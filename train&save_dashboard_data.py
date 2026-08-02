"""
STEP 1 - Run this ONCE (or whenever you want to refresh the dashboard
with new data). It trains the final model and pre-computes the risk
score + uncertainty for the last few days of every patient, saving a
small file that the web app reads instantly (no need to reload 10
million rows every time you open the dashboard).

Output: dashboard_data.csv (small file, ready for the Streamlit app)
"""

import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------
INPUT_PATH = "C:/Users/monic/Desktop/wearable_data_all_patients.csv"
OUTPUT_DASHBOARD_PATH = "C:/Users/monic/Desktop/project_practica/dashboard_data.csv"

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
# STEP 1: Load data
# -----------------------------------------------------------------
print("Loading data...")
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
df_recent["risk_score"] = final_model.predict_proba(X_recent)[:, 1]

# Uncertainty: spread of predictions across individual trees
tree_probs = np.array([
    tree.predict_proba(X_recent)[:, 1] for tree in final_model.estimators_
])
df_recent["uncertainty"] = tree_probs.std(axis=0)

print("Risk scores computed.")

# -----------------------------------------------------------------
# STEP 5: Downsample for a lighter, smoother dashboard file
# -----------------------------------------------------------------
print(f"Downsampling to {DISPLAY_RESOLUTION_MINUTES}-minute resolution...")

display_cols = ["user_id", "received_date"] + SIGNAL_COLS + ["risk_score", "uncertainty"]
df_recent = df_recent[display_cols]

dashboard_parts = []
for uid, g in df_recent.groupby("user_id"):
    g = g.set_index("received_date").resample(f"{DISPLAY_RESOLUTION_MINUTES}min").mean()
    g["user_id"] = uid
    g = g.reset_index()
    dashboard_parts.append(g)

dashboard_data = pd.concat(dashboard_parts, ignore_index=True)
dashboard_data = dashboard_data.dropna()

dashboard_data.to_csv(OUTPUT_DASHBOARD_PATH, index=False)
print(f"\nSaved: {OUTPUT_DASHBOARD_PATH}")
print(f"Dashboard file has {len(dashboard_data):,} rows "
      f"for {dashboard_data['user_id'].nunique()} patients.")
print("Now run the Streamlit app (app.py) to view the dashboard.")