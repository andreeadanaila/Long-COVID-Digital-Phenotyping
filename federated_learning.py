# -*- coding: utf-8 -*-

"""
FEDERATED LEARNING - Long COVID risk forecasting
Reuses the exact same risk-label logic as baselines.py / final_model.py
(personal z-score baseline, 12h forecast window) so results are directly
comparable to the team's existing table.

Compares 3 setups head-to-head, as required by the project brief:
  1. CENTRALIZED  - one model trained on all pooled training data (upper bound)
  2. LOCAL-ONLY   - each client trains alone, never shares anything (lower bound)
  3. FEDERATED    - clients train locally, server averages weights (FedAvg)

Designed to run in Google Colab. Upload the dataset file first (see chat
instructions), then just Run All.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------
# CONFIG - change INPUT_PATH to wherever your file is in Colab
# -----------------------------------------------------------------
INPUT_PATH = "wearable_data_all_patients.csv"   # <-- change if needed

SIGNAL_COLS = [
    "heart_rate", "heart_rate_variability", "spo2", "steps",
    "respirations_per_minute", "distance", "body_battery",
]

FORECAST_HORIZON_MINUTES = 720   # 12 hours ahead (same as baselines.py)
Z_SCORE_THRESHOLD = 2.0

N_CLIENTS = 4          # how many federated "sites" to simulate
N_ROUNDS = 15           # federated communication rounds
LOCAL_EPOCHS = 2        # local training epochs per round, per client
ROWS_PER_CLIENT = 60_000  # subsample per client, keeps Colab CPU fast
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------------------------------------------
# STEP 1: Load data (date format fixed vs. the original scripts -
# this file uses YYYY-MM-DD, not DD/MM/YYYY)
# -----------------------------------------------------------------
print("Loading data...")
dtype_map = {col: "float32" for col in SIGNAL_COLS}
dtype_map["user_id"] = "int16"

df = pd.read_csv(INPUT_PATH, dtype=dtype_map, usecols=["user_id", "received_date"] + SIGNAL_COLS)
df["received_date"] = pd.to_datetime(df["received_date"])  # auto-infers format
df = df.sort_values(["user_id", "received_date"]).reset_index(drop=True)
print(f"Loaded {len(df):,} rows, {df['user_id'].nunique()} patients")

# -----------------------------------------------------------------
# STEP 2-5: Same risk label construction as baselines.py
# -----------------------------------------------------------------
print("Building personalized risk labels (same logic as baselines.py)...")
personal_stats = df.groupby("user_id")[SIGNAL_COLS].agg(["mean", "std"])
personal_stats.columns = ["_".join(c) for c in personal_stats.columns]
df = df.merge(personal_stats, on="user_id", how="left")

for col in SIGNAL_COLS:
    df[f"{col}_zscore"] = (df[col] - df[f"{col}_mean"]) / df[f"{col}_std"]
zscore_cols = [f"{col}_zscore" for col in SIGNAL_COLS]

df["risk_event_now"] = (df[zscore_cols].abs() > Z_SCORE_THRESHOLD).any(axis=1)
print(f"Risk events at a single minute: {df['risk_event_now'].mean()*100:.2f}% of rows")

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
print(f"Positive class (risk in next 12h): {df['target_future_risk'].mean()*100:.2f}%")
print("^ If this is above ~50%, flag it to the team - the label may be too easy to satisfy by chance.")

# -----------------------------------------------------------------
# STEP 6: Train/test split BY PATIENT - same split as baselines.py
# (random_state=42, test_size=0.25) so results are comparable
# -----------------------------------------------------------------
feature_cols = SIGNAL_COLS + zscore_cols
all_patients = sorted(df["user_id"].unique())
train_patients, test_patients = train_test_split(all_patients, test_size=0.25, random_state=42)
print(f"Train patients: {len(train_patients)}, test patients: {len(test_patients)}")

train_df = df[df["user_id"].isin(train_patients)].copy()
test_df = df[df["user_id"].isin(test_patients)].copy()
del df

# Scale features (helps neural net training - fit on train only, no leakage)
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

X_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
y_test = test_df["target_future_risk"].values

# -----------------------------------------------------------------
# STEP 7: Split train patients into N_CLIENTS federated "sites"
# (non-IID by construction - each site is a different set of real
# patients, which is the realistic version of federated health data)
# -----------------------------------------------------------------
rng = np.random.RandomState(SEED)
shuffled = rng.permutation(train_patients)
client_patient_groups = np.array_split(shuffled, N_CLIENTS)

client_data = []
for i, patients in enumerate(client_patient_groups):
    sub = train_df[train_df["user_id"].isin(patients)]
    if len(sub) > ROWS_PER_CLIENT:
        sub = sub.sample(n=ROWS_PER_CLIENT, random_state=SEED)
    Xc = torch.tensor(sub[feature_cols].values, dtype=torch.float32)
    yc = torch.tensor(sub["target_future_risk"].values, dtype=torch.float32)
    client_data.append((Xc, yc))
    print(f"Client {i}: {len(patients)} patients, {len(sub):,} rows")

# -----------------------------------------------------------------
# STEP 8: Model definition - small MLP (needs to support weight
# averaging, which tree models like Random Forest can't do)
# -----------------------------------------------------------------
class RiskMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

n_features = len(feature_cols)
criterion = nn.BCEWithLogitsLoss()

def train_local(model, X, y, epochs, lr=0.01):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        opt.step()
    return model

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test)).numpy()
    auroc = roc_auc_score(y_test, probs)
    prauc = average_precision_score(y_test, probs)
    return auroc, prauc

def get_weights(model):
    return copy.deepcopy(model.state_dict())

def average_weights(weight_list, sample_counts):
    total = sum(sample_counts)
    avg = copy.deepcopy(weight_list[0])
    for key in avg:
        avg[key] = sum(w[key] * (n / total) for w, n in zip(weight_list, sample_counts))
    return avg

# -----------------------------------------------------------------
# SETUP 1: CENTRALIZED baseline - one model, all pooled train data
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("SETUP 1: CENTRALIZED (upper bound)")
print("=" * 60)
X_train_all = torch.cat([c[0] for c in client_data])
y_train_all = torch.cat([c[1] for c in client_data])
central_model = RiskMLP(n_features)
central_model = train_local(central_model, X_train_all, y_train_all, epochs=N_ROUNDS * LOCAL_EPOCHS)
c_auroc, c_prauc = evaluate(central_model, X_test, torch.tensor(y_test, dtype=torch.float32))
print(f"Centralized  -> AUROC: {c_auroc:.3f}  PR-AUC: {c_prauc:.3f}")

# -----------------------------------------------------------------
# SETUP 2: LOCAL-ONLY baseline - each client trains alone, we
# report the average performance across clients (no sharing at all)
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("SETUP 2: LOCAL-ONLY (lower bound, no collaboration)")
print("=" * 60)
local_aurocs, local_praucs = [], []
for i, (Xc, yc) in enumerate(client_data):
    m = RiskMLP(n_features)
    m = train_local(m, Xc, yc, epochs=N_ROUNDS * LOCAL_EPOCHS)
    a, p = evaluate(m, X_test, torch.tensor(y_test, dtype=torch.float32))
    local_aurocs.append(a)
    local_praucs.append(p)
    print(f"  Client {i} alone -> AUROC: {a:.3f}  PR-AUC: {p:.3f}")
print(f"Local-only average -> AUROC: {np.mean(local_aurocs):.3f}  PR-AUC: {np.mean(local_praucs):.3f}")

# -----------------------------------------------------------------
# SETUP 3: FEDERATED (FedAvg) - the actual proposed method
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("SETUP 3: FEDERATED (FedAvg)")
print("=" * 60)
global_model = RiskMLP(n_features)
comm_bytes_per_round = sum(p.numel() for p in global_model.parameters()) * 4 * 2  # up+down, float32
total_comm_bytes = 0

for rnd in range(N_ROUNDS):
    local_weights, sample_counts = [], []
    for Xc, yc in client_data:
        local_model = RiskMLP(n_features)
        local_model.load_state_dict(global_model.state_dict())
        local_model = train_local(local_model, Xc, yc, epochs=LOCAL_EPOCHS)
        local_weights.append(get_weights(local_model))
        sample_counts.append(len(Xc))
    global_model.load_state_dict(average_weights(local_weights, sample_counts))
    total_comm_bytes += comm_bytes_per_round * N_CLIENTS

    if (rnd + 1) % 5 == 0 or rnd == N_ROUNDS - 1:
        a, p = evaluate(global_model, X_test, torch.tensor(y_test, dtype=torch.float32))
        print(f"Round {rnd+1}/{N_ROUNDS} -> AUROC: {a:.3f}  PR-AUC: {p:.3f}")

f_auroc, f_prauc = evaluate(global_model, X_test, torch.tensor(y_test, dtype=torch.float32))

# -----------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY - for your report table")
print("=" * 60)
print(f"{'Setup':<20}{'AUROC':<10}{'PR-AUC':<10}")
print(f"{'Centralized':<20}{c_auroc:<10.3f}{c_prauc:<10.3f}")
print(f"{'Local-only (avg)':<20}{np.mean(local_aurocs):<10.3f}{np.mean(local_praucs):<10.3f}")
print(f"{'Federated (FedAvg)':<20}{f_auroc:<10.3f}{f_prauc:<10.3f}")
print(f"\nTotal communication cost: {total_comm_bytes/1e6:.2f} MB over {N_ROUNDS} rounds, "
      f"{N_CLIENTS} clients ({comm_bytes_per_round/1e3:.1f} KB per client per round)")
print(f"Model size: {sum(p.numel() for p in global_model.parameters()):,} parameters")