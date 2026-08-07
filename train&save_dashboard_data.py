"""


Output: dashboard_data.csv (small file, ready for the Streamlit app)
"""

import pandas as pd
import numpy as np
import gc
import json
import os
import time
from threading import Event, Thread
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

try:
    import psutil
except ImportError:
    psutil = None

# -----------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------
INPUT_PATH = "wearable_data_all_patients.csv"
OUTPUT_DASHBOARD_PATH = "dashboard_data.csv"
PERFORMANCE_METRICS_PATH = "performance_metrics.json"

SIGNAL_COLS = [
    "heart_rate", "heart_rate_variability", "spo2", "steps",
    "respirations_per_minute", "distance", "body_battery",
]

FORECAST_HORIZON_MINUTES = 720
Z_SCORE_THRESHOLD = 2.0
ROLLING_WINDOW_MINUTES = 60
TRAIN_SAMPLE_SIZE = 300_000
DASHBOARD_WINDOW_DAYS = 3        # how many recent days to show per patient
DISPLAY_RESOLUTION_MINUTES = 15  # downsample for a smoother, lighter chart

# -----------------------------------------------------------------
# PERFORMANCE MONITORING
# -----------------------------------------------------------------
# This records the real run time and peak resident memory for the report.
# psutil is optional: install it with "pip install psutil" to include RSS.
run_started = time.perf_counter()
stage_times = {}
peak_rss_bytes = 0
monitor_stop = Event()

def _monitor_memory():
    global peak_rss_bytes
    if psutil is None:
        return
    process = psutil.Process(os.getpid())
    while not monitor_stop.wait(0.05):
        peak_rss_bytes = max(peak_rss_bytes, process.memory_info().rss)

monitor_thread = Thread(target=_monitor_memory, daemon=True)
monitor_thread.start()

def record_stage(name, started):
    """Save elapsed seconds for a pipeline stage and start the next one."""
    stage_times[name] = round(time.perf_counter() - started, 3)
    print(f"{name}: {stage_times[name]:.3f} seconds")
    return time.perf_counter()

# -----------------------------------------------------------------
# STEP 1: Load data
# -----------------------------------------------------------------
print("Loading data...")
stage_started = time.perf_counter()
dtype_map = {col: "float32" for col in SIGNAL_COLS}
dtype_map["user_id"] = "int16"
df = pd.read_csv(INPUT_PATH, dtype=dtype_map)

# The cleaned/merged file was saved by pandas itself earlier, so its
# date format is pandas' own default (YYYY-MM-DD HH:MM:SS), not the
# original raw format. We parse it safely here and verify it worked.
df["received_date"] = pd.to_datetime(df["received_date"])
print(f"received_date dtype after parsing: {df['received_date'].dtype}")
print(f"Sample value: {df['received_date'].iloc[0]}")

df = df.sort_values(["user_id", "received_date"]).reset_index(drop=True)
print(f"Total rows: {len(df):,}")
stage_started = record_stage("load_and_sort_seconds", stage_started)

# -----------------------------------------------------------------
# STEP 2: Personal baseline, z-score, trend features (same as before)
# -----------------------------------------------------------------
print("Computing personal baseline and features...")
for col in SIGNAL_COLS:
    grp = df.groupby("user_id")[col]
    df[f"{col}_mean"] = grp.transform("mean").astype("float32")
    df[f"{col}_std"] = grp.transform("std").astype("float32")

for col in SIGNAL_COLS:
    df[f"{col}_zscore"] = (df[col] - df[f"{col}_mean"]) / df[f"{col}_std"]
zscore_cols = [f"{col}_zscore" for col in SIGNAL_COLS]

trend_cols = []
for col in zscore_cols:
    trend_col = f"{col}_trend"
    df[trend_col] = (
        df.groupby("user_id")[col]
        .transform(lambda s: s.rolling(ROLLING_WINDOW_MINUTES, min_periods=1).mean())
    )
    trend_cols.append(trend_col)

df["risk_event_now"] = (df[zscore_cols].abs() > Z_SCORE_THRESHOLD).any(axis=1)

def make_forecast_target(group):
    future_risk = (
        group["risk_event_now"][::-1]
        .rolling(window=FORECAST_HORIZON_MINUTES, min_periods=1)
        .max()[::-1]
    )
    return future_risk

df["target_future_risk"] = (
    df.groupby("user_id", group_keys=False)
    .apply(make_forecast_target, include_groups=False)
    .astype(int)
)

feature_cols = SIGNAL_COLS + zscore_cols + trend_cols
stage_started = record_stage("feature_engineering_seconds", stage_started)

# -----------------------------------------------------------------
# STEP 3: Train the final model (same as before)
# -----------------------------------------------------------------
print("Training final model...")
all_patients = sorted(df["user_id"].unique())
train_patients, test_patients = train_test_split(
    all_patients, test_size=0.25, random_state=42
)
train_mask = df["user_id"].isin(train_patients)
X_train_full = df.loc[train_mask, feature_cols]
y_train_full = df.loc[train_mask, "target_future_risk"]

if len(X_train_full) > TRAIN_SAMPLE_SIZE:
    sample_idx = X_train_full.sample(n=TRAIN_SAMPLE_SIZE, random_state=42).index
    X_train_small = X_train_full.loc[sample_idx]
    y_train_small = y_train_full.loc[sample_idx]
else:
    X_train_small, y_train_small = X_train_full, y_train_full

final_model = RandomForestClassifier(
    n_estimators=100, max_depth=14, random_state=42,
    class_weight="balanced", n_jobs=2,
)
final_model.fit(X_train_small, y_train_small)
print("Model trained.")
stage_started = record_stage("model_training_seconds", stage_started)

del X_train_full, y_train_full, X_train_small, y_train_small
gc.collect()

# -----------------------------------------------------------------
# STEP 4: Compute risk score + uncertainty for the last few days of
# EVERY patient (this is what the dashboard will show)
# -----------------------------------------------------------------
print(f"Computing risk scores for the last {DASHBOARD_WINDOW_DAYS} days "
      f"of every patient...")

cutoff_per_patient = (
    df.groupby("user_id")["received_date"].transform("max")
    - pd.Timedelta(days=DASHBOARD_WINDOW_DAYS)
)
recent_mask = df["received_date"] >= cutoff_per_patient
df_recent = df[recent_mask].copy()

del df
gc.collect()

X_recent = df_recent[feature_cols]
recent_rows_before_downsampling = len(X_recent)
inference_started = time.perf_counter()
df_recent["risk_score"] = final_model.predict_proba(X_recent)[:, 1]

# Uncertainty: spread of predictions across individual trees.  The previous
# approach stacked every tree's probabilities in one (100 x N) array.  Keeping
# only a running mean and mean-square gives the identical population standard
# deviation while making uncertainty memory O(N), not O(number_of_trees x N).
tree_mean = np.zeros(len(X_recent), dtype="float64")
tree_mean_square = np.zeros(len(X_recent), dtype="float64")
for tree in final_model.estimators_:
    tree_probability = tree.predict_proba(X_recent)[:, 1]
    tree_mean += tree_probability
    tree_mean_square += tree_probability ** 2
tree_mean /= len(final_model.estimators_)
tree_mean_square /= len(final_model.estimators_)
df_recent["uncertainty"] = np.sqrt(np.maximum(tree_mean_square - tree_mean ** 2, 0))
del tree_mean, tree_mean_square, tree_probability
gc.collect()

inference_seconds = time.perf_counter() - inference_started
stage_times["risk_and_uncertainty_seconds"] = round(inference_seconds, 3)
stage_times["per_prediction_milliseconds"] = round(
    inference_seconds * 1000 / max(len(X_recent), 1), 4
)

print("Risk scores computed.")

# -----------------------------------------------------------------
# STEP 4b: NEW - explainability. For each prediction, rank which
# signals are driving the risk score. Combines the model's global
# feature importance (learned during training) with how abnormal
# each signal currently is for this specific patient (their z-score
# right now). No new dependency (SHAP etc.) needed for this.
# -----------------------------------------------------------------
print("Computing per-prediction explainability...")
stage_started = time.perf_counter()

readable_names = {
    "heart_rate": "Heart Rate", "heart_rate_variability": "Heart Rate Variability",
    "spo2": "SpO2", "steps": "Steps", "respirations_per_minute": "Respiration Rate",
    "distance": "Distance", "body_battery": "Body Battery",
}

# 'steps' and 'distance' are cumulative daily counters that reset to 0 every
# day and climb monotonically until the next reset - their z-score/trend
# doesn't reflect a genuine physiological deviation the way it does for the
# other signals, it mostly tracks time-of-day. Excluding them here so the
# explanation reflects real physiological drivers, consistent with the
# earlier ablation study (which found body_battery mattered most and
# steps/distance barely mattered at all).
EXPLAINABILITY_SIGNALS = [c for c in SIGNAL_COLS if c not in ("steps", "distance")]

importances = dict(zip(feature_cols, final_model.feature_importances_))
# Combine the raw + zscore + trend importance for each underlying signal,
# since the model sees each signal as 3 separate features.
signal_importance = {
    col: importances[col] + importances[f"{col}_zscore"] + importances[f"{col}_zscore_trend"]
    for col in EXPLAINABILITY_SIGNALS
}

contribution_matrix = np.zeros((len(X_recent), len(EXPLAINABILITY_SIGNALS)))
for i, col in enumerate(EXPLAINABILITY_SIGNALS):
    contribution_matrix[:, i] = signal_importance[col] * X_recent[f"{col}_zscore"].abs().values

top_indices = np.argsort(-contribution_matrix, axis=1)[:, :3]
for rank in range(3):
    df_recent[f"top_factor_{rank + 1}"] = [
        readable_names[EXPLAINABILITY_SIGNALS[idx]] for idx in top_indices[:, rank]
    ]

print("Explainability columns ready (top_factor_1/2/3).")
stage_started = record_stage("explainability_seconds", stage_started)

# -----------------------------------------------------------------
# STEP 5: Downsample for a lighter, smoother dashboard file
# -----------------------------------------------------------------
print(f"Downsampling to {DISPLAY_RESOLUTION_MINUTES}-minute resolution...")
stage_started = time.perf_counter()

display_cols = (
    ["user_id", "received_date"] + SIGNAL_COLS
    + ["risk_score", "uncertainty", "top_factor_1", "top_factor_2", "top_factor_3"]
)
df_recent = df_recent[display_cols]

numeric_cols = SIGNAL_COLS + ["risk_score", "uncertainty"]
categorical_cols = ["top_factor_1", "top_factor_2", "top_factor_3"]

dashboard_parts = []
for uid, g in df_recent.groupby("user_id"):
    g_indexed = g.set_index("received_date")
    numeric_resampled = g_indexed[numeric_cols].resample(f"{DISPLAY_RESOLUTION_MINUTES}min").mean()
    # Take the 3 factor columns from a single representative row (the most
    # recent one in each window), NOT an independent per-column vote - voting
    # per column separately can pick different original rows for each rank,
    # which can produce duplicates across top_factor_1/2/3 even though every
    # individual original prediction always had 3 distinct signals.
    categorical_resampled = g_indexed[categorical_cols].resample(
        f"{DISPLAY_RESOLUTION_MINUTES}min"
    ).last()
    combined = numeric_resampled.join(categorical_resampled)
    combined["user_id"] = uid
    combined = combined.reset_index()
    dashboard_parts.append(combined)

dashboard_data = pd.concat(dashboard_parts, ignore_index=True)
dashboard_data = dashboard_data.dropna()

dashboard_data.to_csv(OUTPUT_DASHBOARD_PATH, index=False)
stage_started = record_stage("downsample_and_write_seconds", stage_started)

# A compact, machine-readable record for the latency/memory/scalability
# section of the report. Values are deliberately measured, not hard-coded.
monitor_stop.set()
monitor_thread.join()
if psutil is not None:
    peak_rss_bytes = max(peak_rss_bytes, psutil.Process(os.getpid()).memory_info().rss)
performance_metrics = {
    "dataset_rows": int(len(df_recent)),
    "dashboard_rows": int(len(dashboard_data)),
    "display_resolution_minutes": DISPLAY_RESOLUTION_MINUTES,
    "row_reduction_percent": round(
        100 * (1 - len(dashboard_data) / max(recent_rows_before_downsampling, 1)), 2
    ),
    "dashboard_file_bytes": os.path.getsize(OUTPUT_DASHBOARD_PATH),
    "peak_rss_megabytes": round(peak_rss_bytes / 1024 ** 2, 1) if psutil is not None else None,
    "peak_rss_note": "Install psutil to measure RSS" if psutil is None else "Sampled every 50 ms",
    "timings_seconds": stage_times,
    "total_runtime_seconds": round(time.perf_counter() - run_started, 3),
    "memory_optimisation": "Streaming tree uncertainty uses O(N) memory instead of O(100*N).",
}
with open(PERFORMANCE_METRICS_PATH, "w", encoding="utf-8") as metrics_file:
    json.dump(performance_metrics, metrics_file, indent=2)
print(f"Performance metrics saved: {PERFORMANCE_METRICS_PATH}")
print(f"\nSaved: {OUTPUT_DASHBOARD_PATH}")
print(f"Dashboard file has {len(dashboard_data):,} rows "
      f"for {dashboard_data['user_id'].nunique()} patients.")
print("Now run the Streamlit app (app.py) to view the dashboard.")
