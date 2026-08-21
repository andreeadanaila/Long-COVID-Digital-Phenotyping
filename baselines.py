"""
FULL PIPELINE - risk label construction + 5 baseline models +
ablation study + multi-seed robustness check.

Input: wearable_data_all_patients.csv 

"""

import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, average_precision_score

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

FORECAST_HORIZON_MINUTES = 720   # 12 hours ahead (use 2880 for 48h)
Z_SCORE_THRESHOLD = 2.0          # how far from personal normal = "risk event"


PATIENT_SAMPLE = None

TREE_MODEL_SAMPLE_SIZE = 300_000  # training rows used for tree-based models

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
    print(f"TEST RUN on {PATIENT_SAMPLE} patients: {sample_ids}")

df = df.sort_values(["user_id", "received_date"]).reset_index(drop=True)
print(f"Total rows used: {len(df):,}")

# -----------------------------------------------------------------
# STEP 2: Personal baseline per patient (mean + std per signal)
# -----------------------------------------------------------------
print("Computing personal baseline per patient...")
personal_stats = df.groupby("user_id")[SIGNAL_COLS].agg(["mean", "std"])
personal_stats.columns = ["_".join(c) for c in personal_stats.columns]
df = df.merge(personal_stats, on="user_id", how="left")

# -----------------------------------------------------------------
# STEP 3: Z-score per signal (how far the current value is from the
# patient's own personal normal)
# -----------------------------------------------------------------
for col in SIGNAL_COLS:
    df[f"{col}_zscore"] = (df[col] - df[f"{col}_mean"]) / df[f"{col}_std"]

zscore_cols = [f"{col}_zscore" for col in SIGNAL_COLS]

# -----------------------------------------------------------------
# STEP 4: Define "risk event" - any signal with |z-score| above
# threshold, AT THIS MOMENT
# -----------------------------------------------------------------
df["risk_event_now"] = (df[zscore_cols].abs() > Z_SCORE_THRESHOLD).any(axis=1)

print(f"Risk events found: {df['risk_event_now'].sum():,} "
      f"out of {len(df):,} rows ({df['risk_event_now'].mean()*100:.2f}%)")

# -----------------------------------------------------------------
# STEP 5: FORECASTING target - will a risk event occur in the next
# FORECAST_HORIZON_MINUTES, starting from this minute? (done per
# patient, so we never mix patients together)
# -----------------------------------------------------------------
print(f"Building forecasting target ({FORECAST_HORIZON_MINUTES} min ahead)...")

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

print(f"Target distribution (risk will occur in the next "
      f"{FORECAST_HORIZON_MINUTES} min):")
print(df["target_future_risk"].value_counts())
print(f"Positive class percentage: {df['target_future_risk'].mean()*100:.2f}%")

# -----------------------------------------------------------------
# STEP 6: Build X / y + train/test split BY PATIENT (not by row -
# otherwise rows from the same patient could leak into both train
# and test, which would be data leakage)
# -----------------------------------------------------------------
feature_cols = SIGNAL_COLS + zscore_cols

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

print(f"\nTrain patients: {len(train_patients)}, test patients: {len(test_patients)}")
print(f"Train rows: {len(X_train):,}, test rows: {len(X_test):,}")


del df, train_df, test_df
gc.collect()

# Per-minute data is highly autocorrelated (one minute looks almost
# identical to the next) - tree-based models don't need all rows.
# We use a smaller random sample just for them. Logistic Regression
# stays on the full X_train.
if len(X_train) > TREE_MODEL_SAMPLE_SIZE:
    sample_idx = X_train.sample(n=TREE_MODEL_SAMPLE_SIZE, random_state=42).index
    X_train_small = X_train.loc[sample_idx]
    y_train_small = y_train.loc[sample_idx]
    print(f"Reduced sample for tree-based models: "
          f"{len(X_train_small):,} rows (out of {len(X_train):,})")
else:
    X_train_small, y_train_small = X_train, y_train

# -----------------------------------------------------------------
# STEP 7: The 5 baseline models
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING AND EVALUATING - 5 BASELINE MODELS")
print("=" * 60)

results = []

# B1 - Simple threshold rule: risk = any z-score already over threshold
b1_preds = (X_test[zscore_cols].abs() > Z_SCORE_THRESHOLD).any(axis=1).astype(int)
results.append(("B1 - Threshold rule", None,
                 roc_auc_score(y_test, b1_preds),
                 average_precision_score(y_test, b1_preds)))

# B2 - Logistic Regression 
b2 = LogisticRegression(max_iter=1000, class_weight="balanced")
b2.fit(X_train, y_train)
b2_probs = b2.predict_proba(X_test)[:, 1]
results.append(("B2 - Logistic Regression", b2,
                 roc_auc_score(y_test, b2_probs),
                 average_precision_score(y_test, b2_probs)))
del b2
gc.collect()

# B3 - Random Forest 
b3 = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42,
                             class_weight="balanced", n_jobs=2)
b3.fit(X_train_small, y_train_small)
b3_probs = b3.predict_proba(X_test)[:, 1]
results.append(("B3 - Random Forest", b3,
                 roc_auc_score(y_test, b3_probs),
                 average_precision_score(y_test, b3_probs)))
del b3
gc.collect()

# B4 - Isolation Forest 
b4 = IsolationForest(random_state=42, contamination="auto", n_jobs=2)
b4.fit(X_train_small)
b4_scores = -b4.score_samples(X_test)  # higher score = more abnormal
results.append(("B4 - Isolation Forest", b4,
                 roc_auc_score(y_test, b4_scores),
                 average_precision_score(y_test, b4_scores)))
del b4
gc.collect()

# B5 - Gradient Boosting: using HistGradientBoostingClassifier, the
# variant built for large datasets 
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_small)
b5 = HistGradientBoostingClassifier(random_state=42, max_iter=150)
b5.fit(X_train_small, y_train_small, sample_weight=sample_weights)
b5_probs = b5.predict_proba(X_test)[:, 1]
results.append(("B5 - Gradient Boosting (Hist)", b5,
                 roc_auc_score(y_test, b5_probs),
                 average_precision_score(y_test, b5_probs)))
del b5
gc.collect()

print("\n=== FINAL RESULTS (AUROC and PR-AUC, not just accuracy!) ===")
print(f"{'Model':<30} {'AUROC':>10} {'PR-AUC':>10}")
for name, model, auroc, prauc in results:
    print(f"{name:<30} {auroc:>10.3f} {prauc:>10.3f}")

# ===================================================================
# STEP 8: ABLATION STUDY - remove one signal at a time, using the
# winning model (HistGradientBoosting)
# ===================================================================
print("\n" + "=" * 60)
print("ABLATION STUDY - removing one signal at a time")
print("=" * 60)

baseline_auroc_ablation = [r[2] for r in results if "Hist" in r[0]][0]
print(f"\nBASELINE (all signals, HistGradientBoosting): "
      f"AUROC = {baseline_auroc_ablation:.3f}\n")

ablation_results = []
for signal in SIGNAL_COLS:
    cols_to_drop = [signal, f"{signal}_zscore"]
    reduced_cols = [c for c in feature_cols if c not in cols_to_drop]

    X_train_reduced = X_train_small[reduced_cols]
    X_test_reduced = X_test[reduced_cols]

    model = HistGradientBoostingClassifier(random_state=42, max_iter=150)
    model.fit(X_train_reduced, y_train_small, sample_weight=sample_weights)
    auroc = roc_auc_score(y_test, model.predict_proba(X_test_reduced)[:, 1])

    drop = baseline_auroc_ablation - auroc
    ablation_results.append((signal, auroc, drop))
    print(f"Without '{signal}': AUROC = {auroc:.3f}  (drop: {drop:+.3f})")

ablation_sorted = sorted(ablation_results, key=lambda x: -x[2])
print("\nMost important signal:", ablation_sorted[0][0])
print("Least important signal:", ablation_sorted[-1][0])

# ===================================================================
# STEP 9: ROBUSTNESS CHECK - same model, 3 different seeds
# ===================================================================
print("\n" + "=" * 60)
print("ROBUSTNESS - same model, 3 different seeds")
print("=" * 60)

seeds = [42, 123, 2024]
seed_aurocs = []

for seed in seeds:
    model = HistGradientBoostingClassifier(random_state=seed, max_iter=150)
    model.fit(X_train_small, y_train_small, sample_weight=sample_weights)
    auroc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    seed_aurocs.append(auroc)
    print(f"Seed {seed}: AUROC = {auroc:.3f}")

mean_auroc = np.mean(seed_aurocs)
std_auroc = np.std(seed_aurocs)
print(f"\nMean AUROC: {mean_auroc:.3f} +/- {std_auroc:.3f}")

print("\nDone. Next step: the final personalized forecasting model")
print("(with uncertainty estimation), then federated learning.")
