"""
Hyperparameter optimization for the risk forecasting model.

Uses RandomizedSearchCV to search over Random Forest and Gradient Boosting
hyperparameters (the two best-performing model types from the baseline
comparison), instead of using guessed/fixed values. Saves the best
parameters and a comparison table.
"""

import pandas as pd
import numpy as np
import time
import json
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, make_scorer

# -----------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------
INPUT_PATH = "wearable_data_all_patients.csv"
OUTPUT_DIR = "hpo_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIGNAL_COLS = [
    "heart_rate", "heart_rate_variability", "spo2", "steps",
    "respirations_per_minute", "distance", "body_battery",
]

FORECAST_HORIZON_MINUTES = 720
Z_SCORE_THRESHOLD = 2.0
SUSTAINED_MINUTES = 15   # same fix as the federated learning script
TRAIN_SAMPLE_SIZE = 150_000  # subsample for search speed (search tries many models)
N_ITER = 15                   # how many random hyperparameter combinations to try
CV_FOLDS = 3
SEED = 42

# -----------------------------------------------------------------
# STEP 1: Load data
# -----------------------------------------------------------------
print("Loading data...")
dtype_map = {col: "float32" for col in SIGNAL_COLS}
dtype_map["user_id"] = "int16"
df = pd.read_csv(INPUT_PATH, dtype=dtype_map, usecols=["user_id", "received_date"] + SIGNAL_COLS)
df["received_date"] = pd.to_datetime(df["received_date"])
df = df.sort_values(["user_id", "received_date"]).reset_index(drop=True)
print(f"Loaded {len(df):,} rows, {df['user_id'].nunique()} patients")

if df["user_id"].nunique() < 70 or len(df) < 9_000_000:
    raise SystemExit(
        f"STOP: expected ~80 patients and ~10,368,000 rows, got "
        f"{df['user_id'].nunique()} patients and {len(df):,} rows. "
        f"The upload is likely incomplete - re-upload and re-run."
    )

# -----------------------------------------------------------------
# STEP 2: Same personalized features + sustained-deviation label as
# the federated learning script (kept identical for consistency)
# -----------------------------------------------------------------
print("Building features and labels...")
for col in SIGNAL_COLS:
    grp = df.groupby("user_id")[col]
    df[f"{col}_mean"] = grp.transform("mean").astype("float32")
    df[f"{col}_std"] = grp.transform("std").astype("float32")

for col in SIGNAL_COLS:
    df[f"{col}_zscore"] = (df[col] - df[f"{col}_mean"]) / df[f"{col}_std"]
zscore_cols = [f"{col}_zscore" for col in SIGNAL_COLS]

df["risk_event_now"] = (df[zscore_cols].abs() > Z_SCORE_THRESHOLD).any(axis=1)

def sustained_flag(series_bool_as_int, group_ids):
    return (
        series_bool_as_int.groupby(group_ids)
        .apply(lambda s: s.rolling(window=SUSTAINED_MINUTES, min_periods=SUSTAINED_MINUTES).min())
        .reset_index(level=0, drop=True)
        .fillna(0)
        .astype(bool)
    )

df["risk_event_now"] = sustained_flag(df["risk_event_now"].astype(int), df["user_id"])

def make_forecast_target(group):
    return (
        group["risk_event_now"][::-1]
        .rolling(window=FORECAST_HORIZON_MINUTES, min_periods=1)
        .max()[::-1]
    )

df["target_future_risk"] = (
    df.groupby("user_id", group_keys=False)
    .apply(make_forecast_target, include_groups=False)
    .astype(int)
)
print(f"Positive class: {df['target_future_risk'].mean()*100:.2f}%")

feature_cols = SIGNAL_COLS + zscore_cols

# -----------------------------------------------------------------
# STEP 3: Train/test split BY PATIENT - same 42 seed as baselines.py
# -----------------------------------------------------------------
all_patients = sorted(df["user_id"].unique())
train_patients, test_patients = train_test_split(all_patients, test_size=0.25, random_state=42)
train_mask = df["user_id"].isin(train_patients)
X_train_full = df.loc[train_mask, feature_cols]
y_train_full = df.loc[train_mask, "target_future_risk"]

if len(X_train_full) > TRAIN_SAMPLE_SIZE:
    sample_idx = X_train_full.sample(n=TRAIN_SAMPLE_SIZE, random_state=SEED).index
    X_train = X_train_full.loc[sample_idx]
    y_train = y_train_full.loc[sample_idx]
else:
    X_train, y_train = X_train_full, y_train_full

test_mask = df["user_id"].isin(test_patients)
X_test = df.loc[test_mask, feature_cols]
y_test = df.loc[test_mask, "target_future_risk"]

print(f"Train (subsampled for search): {len(X_train):,} rows | Test: {len(X_test):,} rows")

del df
import gc
gc.collect()

# -----------------------------------------------------------------
# STEP 4: Hyperparameter search - Random Forest
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("SEARCHING: Random Forest hyperparameters")
print("=" * 60)

rf_param_dist = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [8, 10, 14, 18, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}

t0 = time.time()
rf_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=2),
    param_distributions=rf_param_dist,
    n_iter=N_ITER, cv=CV_FOLDS, scoring="roc_auc",
    random_state=SEED, n_jobs=1, verbose=1,
)
rf_search.fit(X_train, y_train)
rf_search_time = time.time() - t0

rf_best = rf_search.best_estimator_
rf_test_probs = rf_best.predict_proba(X_test)[:, 1]
rf_test_auroc = roc_auc_score(y_test, rf_test_probs)
rf_test_prauc = average_precision_score(y_test, rf_test_probs)

print(f"\nBest RF params: {rf_search.best_params_}")
print(f"Best RF CV AUROC: {rf_search.best_score_:.4f}")
print(f"RF on held-out test set -> AUROC: {rf_test_auroc:.4f}  PR-AUC: {rf_test_prauc:.4f}")
print(f"Search time: {rf_search_time:.1f}s")

# -----------------------------------------------------------------
# STEP 5: Hyperparameter search - Gradient Boosting
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("SEARCHING: Gradient Boosting hyperparameters")
print("=" * 60)

gb_param_dist = {
    "max_iter": [50, 100, 150],
    "max_depth": [None, 6, 10, 14],
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "min_samples_leaf": [10, 20, 40],
    "l2_regularization": [0.0, 0.1, 1.0],
}

t0 = time.time()
gb_search = RandomizedSearchCV(
    HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    param_distributions=gb_param_dist,
    n_iter=N_ITER, cv=CV_FOLDS, scoring="roc_auc",
    random_state=SEED, n_jobs=1, verbose=1,
)
gb_search.fit(X_train, y_train)
gb_search_time = time.time() - t0

gb_best = gb_search.best_estimator_
gb_test_probs = gb_best.predict_proba(X_test)[:, 1]
gb_test_auroc = roc_auc_score(y_test, gb_test_probs)
gb_test_prauc = average_precision_score(y_test, gb_test_probs)

print(f"\nBest GB params: {gb_search.best_params_}")
print(f"Best GB CV AUROC: {gb_search.best_score_:.4f}")
print(f"GB on held-out test set -> AUROC: {gb_test_auroc:.4f}  PR-AUC: {gb_test_prauc:.4f}")
print(f"Search time: {gb_search_time:.1f}s")

# -----------------------------------------------------------------
# STEP 6: Compare against the untuned baseline (fixed guessed params,
# same ones used in final_model.py: n_estimators=100, max_depth=14)
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("COMPARISON: tuned vs untuned (guessed) hyperparameters")
print("=" * 60)

untuned_rf = RandomForestClassifier(
    n_estimators=100, max_depth=14, random_state=SEED,
    class_weight="balanced", n_jobs=2,
)
untuned_rf.fit(X_train, y_train)
untuned_probs = untuned_rf.predict_proba(X_test)[:, 1]
untuned_auroc = roc_auc_score(y_test, untuned_probs)
untuned_prauc = average_precision_score(y_test, untuned_probs)

print(f"Untuned RF (n_estimators=100, max_depth=14) -> AUROC: {untuned_auroc:.4f}  PR-AUC: {untuned_prauc:.4f}")
print(f"Tuned RF (best found)                        -> AUROC: {rf_test_auroc:.4f}  PR-AUC: {rf_test_prauc:.4f}")
print(f"Tuned Gradient Boosting (best found)          -> AUROC: {gb_test_auroc:.4f}  PR-AUC: {gb_test_prauc:.4f}")

# -----------------------------------------------------------------
# SAVE RESULTS
# -----------------------------------------------------------------
results = {
    "search_config": {
        "n_iter": N_ITER, "cv_folds": CV_FOLDS, "train_sample_size": len(X_train),
    },
    "random_forest": {
        "best_params": rf_search.best_params_,
        "cv_auroc": rf_search.best_score_,
        "test_auroc": rf_test_auroc,
        "test_prauc": rf_test_prauc,
        "search_time_sec": rf_search_time,
    },
    "gradient_boosting": {
        "best_params": gb_search.best_params_,
        "cv_auroc": gb_search.best_score_,
        "test_auroc": gb_test_auroc,
        "test_prauc": gb_test_prauc,
        "search_time_sec": gb_search_time,
    },
    "untuned_baseline": {
        "params": {"n_estimators": 100, "max_depth": 14},
        "test_auroc": untuned_auroc,
        "test_prauc": untuned_prauc,
    },
}

with open(f"{OUTPUT_DIR}/hpo_results.json", "w") as f:
    json.dump(results, f, indent=2)

summary_df = pd.DataFrame([
    {"model": "Untuned RF (guessed params)", "auroc": untuned_auroc, "prauc": untuned_prauc},
    {"model": "Tuned RF (RandomizedSearchCV)", "auroc": rf_test_auroc, "prauc": rf_test_prauc},
    {"model": "Tuned Gradient Boosting (RandomizedSearchCV)", "auroc": gb_test_auroc, "prauc": gb_test_prauc},
])
summary_df.to_csv(f"{OUTPUT_DIR}/hpo_summary_table.csv", index=False)

import shutil
shutil.make_archive(OUTPUT_DIR, "zip", OUTPUT_DIR)

print(f"\nSaved: {OUTPUT_DIR}/hpo_results.json, {OUTPUT_DIR}/hpo_summary_table.csv")
print(f"Saved: {OUTPUT_DIR}.zip (right-click this file in the Colab file panel to download it)")
print("\nDone.")
