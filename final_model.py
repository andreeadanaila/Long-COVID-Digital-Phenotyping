"""
FINAL MODEL - adds trend features (personalization over the last hour)
and uncertainty estimation (variance across Random Forest trees).

Compares this final model against the best baseline from before
(HistGradientBoosting, AUROC 0.908) to show whether the extra
complexity is actually worth it.
"""

import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# -----------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------
INPUT_PATH = "C:/Users/monic/Desktop/wearable_data_all_patients.csv"

SIGNAL_COLS = [
    "heart_rate",
    "heart_rate_variability",
    "spo2",
    "steps",
    "respirations_per_minute",
    "distance",
    "body_battery",
]

FORECAST_HORIZON_MINUTES = 720
Z_SCORE_THRESHOLD = 2.0
ROLLING_WINDOW_MINUTES = 60      # trend window: last 1 hour

PATIENT_SAMPLE = None            # set to e.g. 10 to test faster first
TRAIN_SAMPLE_SIZE = 300_000      # rows used to train the final model

# -----------------------------------------------------------------
# STEP 1: Load data
# -----------------------------------------------------------------
print("Loading data...")
dtype_map = {col: "float32" for col in SIGNAL_COLS}
dtype_map["user_id"] = "int16"

df = pd.read_csv(
    INPUT_PATH,
    dtype=dtype_map,
    parse_dates=["received_date"],
    date_format="%d/%m/%Y %H:%M:%S",
)

if PATIENT_SAMPLE is not None:
    sample_ids = sorted(df["user_id"].unique())[:PATIENT_SAMPLE]
    df = df[df["user_id"].isin(sample_ids)].copy()
    print(f"TEST RUN on {PATIENT_SAMPLE} patients")

df = df.sort_values(["user_id", "received_date"]).reset_index(drop=True)
print(f"Total rows used: {len(df):,}")

# -----------------------------------------------------------------
# STEP 2: Personal baseline + z-score (same as before)
# -----------------------------------------------------------------
print("Computing personal baseline per patient...")
personal_stats = df.groupby("user_id")[SIGNAL_COLS].agg(["mean", "std"])
personal_stats.columns = ["_".join(c) for c in personal_stats.columns]
df = df.merge(personal_stats, on="user_id", how="left")

for col in SIGNAL_COLS:
    df[f"{col}_zscore"] = (df[col] - df[f"{col}_mean"]) / df[f"{col}_std"]

zscore_cols = [f"{col}_zscore" for col in SIGNAL_COLS]

# -----------------------------------------------------------------
# STEP 3: NEW - rolling trend features (last hour, per patient)
# This is what adds real "personalization over time" to the model:
# is the patient's z-score trending up, or was it just a 1-minute spike?
# -----------------------------------------------------------------
print(f"Computing {ROLLING_WINDOW_MINUTES}-minute rolling trend features...")

trend_cols = []
for col in zscore_cols:
    trend_col = f"{col}_trend"
    df[trend_col] = (
        df.groupby("user_id")[col]
        .transform(lambda s: s.rolling(ROLLING_WINDOW_MINUTES, min_periods=1).mean())
    )
    trend_cols.append(trend_col)

print("Trend features ready.")

# -----------------------------------------------------------------
# STEP 4: Risk event + forecasting target (same logic as before)
# -----------------------------------------------------------------
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

print(f"Positive class percentage: {df['target_future_risk'].mean()*100:.2f}%")

# -----------------------------------------------------------------
# STEP 5: Train/test split BY PATIENT (same 60/20 split as before,
# same random_state, so it's directly comparable to the baselines)
# -----------------------------------------------------------------
feature_cols = SIGNAL_COLS + zscore_cols + trend_cols

all_patients = sorted(df["user_id"].unique())
train_patients, test_patients = train_test_split(
    all_patients, test_size=0.25, random_state=42
)

train_df = df[df["user_id"].isin(train_patients)]
test_df = df[df["user_id"].isin(test_patients)]

X_train = train_df[feature_cols]
y_train = train_df["target_future_risk"]
X_test = test_df[feature_cols]
y_test = test_df["target_future_risk"]

print(f"Train rows: {len(X_train):,}, test rows: {len(X_test):,}")

del df, train_df, test_df
gc.collect()

if len(X_train) > TRAIN_SAMPLE_SIZE:
    sample_idx = X_train.sample(n=TRAIN_SAMPLE_SIZE, random_state=42).index
    X_train_small = X_train.loc[sample_idx]
    y_train_small = y_train.loc[sample_idx]
    print(f"Training sample: {len(X_train_small):,} rows")
else:
    X_train_small, y_train_small = X_train, y_train

# -----------------------------------------------------------------
# STEP 6: Train the FINAL model (Random Forest, so we can measure
# uncertainty from the spread across individual trees)
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING FINAL MODEL (with trend features + uncertainty)")
print("=" * 60)

final_model = RandomForestClassifier(
    n_estimators=100, max_depth=14, random_state=42,
    class_weight="balanced", n_jobs=2,
)
final_model.fit(X_train_small, y_train_small)

# Main prediction: average probability across all trees (as usual)
final_probs = final_model.predict_proba(X_test)[:, 1]

# Uncertainty: standard deviation of each tree's individual prediction
# for the same patient/moment - high spread = low confidence
tree_probs = np.array([
    tree.predict_proba(X_test)[:, 1] for tree in final_model.estimators_
])
uncertainty = tree_probs.std(axis=0)

auroc_final = roc_auc_score(y_test, final_probs)
prauc_final = average_precision_score(y_test, final_probs)
brier_final = brier_score_loss(y_test, final_probs)

print(f"\nFinal model - AUROC: {auroc_final:.3f}")
print(f"Final model - PR-AUC: {prauc_final:.3f}")
print(f"Final model - Brier score (calibration, lower is better): {brier_final:.3f}")
print(f"\nMean uncertainty across test set: {uncertainty.mean():.3f}")
print(f"Uncertainty range: {uncertainty.min():.3f} to {uncertainty.max():.3f}")

print("\nCompare to the best baseline (HistGradientBoosting, no trend "
      "features, no uncertainty): AUROC 0.908, PR-AUC 0.979")
print("If AUROC/PR-AUC above are higher, the trend features helped.")
print("Either way, this model now also outputs a confidence score per "
      "prediction, which the baselines did not.")

# -----------------------------------------------------------------
# STEP 7: A few example predictions with their uncertainty, to show
# in the report/demo what the output actually looks like
# -----------------------------------------------------------------
print("\n=== Example predictions (risk score + confidence) ===")
example_df = pd.DataFrame({
    "risk_score": final_probs[:10],
    "uncertainty": uncertainty[:10],
    "actual_label": y_test.values[:10],
})
print(example_df.round(3))