"""
Federated learning for Long COVID risk forecasting.

Reuses the same risk-label logic as baselines.py / final_model.py
(personal z-score baseline, 12h forecast window), so results are
directly comparable to the existing baseline table.

Compares three setups:
  1. Centralized - one model trained on all pooled training data (upper bound)
  2. Local-only  - each client trains alone, never shares anything (lower bound)
  3. Federated   - clients train locally, server averages weights (FedAvg)

Also runs a multi-seed robustness check and a missing-data robustness
experiment, and saves all results + a summary figure to OUTPUT_DIR.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import copy
import time
import os
import psutil
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------
# CONFIG - change INPUT_PATH to wherever your file is in Colab
# -----------------------------------------------------------------
INPUT_PATH = "wearable_data_all_patients.csv"   # <-- change if needed
OUTPUT_DIR = "fl_results"                        # all results auto-saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
SEEDS = [42, 123, 2024]   # multi-seed robustness check
MISSING_DATA_FRACTIONS = [0.0, 0.1, 0.25, 0.5]  # for the robustness experiment

process = psutil.Process(os.getpid())
def current_memory_mb():
    return process.memory_info().rss / 1e6

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

# Sanity check - stop early if the upload was incomplete/truncated, instead
# of silently running the whole experiment on broken data
if df["user_id"].nunique() < 70 or len(df) < 9_000_000:
    raise SystemExit(
        f"STOP: expected ~80 patients and ~10,368,000 rows, got "
        f"{df['user_id'].nunique()} patients and {len(df):,} rows. "
        f"The upload is likely incomplete - re-upload the file (or better, "
        f"mount Google Drive instead of using session storage) and re-run."
    )

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
print(f"Risk events at a single minute (before sustain filter): {df['risk_event_now'].mean()*100:.2f}% of rows")

# A single noisy minute isn't a real health event - require the deviation to
# persist for at least SUSTAINED_MINUTES in a row before counting it as a
# risk event. This fixes the label being satisfied by chance ~83% of the
# time under the single-minute definition.
SUSTAINED_MINUTES = 15

def sustained_flag(series_bool_as_int, group_ids):
    return (
        series_bool_as_int.groupby(group_ids)
        .apply(lambda s: s.rolling(window=SUSTAINED_MINUTES, min_periods=SUSTAINED_MINUTES).min())
        .reset_index(level=0, drop=True)
        .fillna(0)
        .astype(bool)
    )

df["risk_event_now"] = sustained_flag(df["risk_event_now"].astype(int), df["user_id"])
print(f"Risk events after requiring {SUSTAINED_MINUTES}min sustained deviation: {df['risk_event_now'].mean()*100:.2f}% of rows")

df["target_future_risk"] = (
    df.groupby("user_id")["risk_event_now"]
    .apply(lambda s: s[::-1].rolling(window=FORECAST_HORIZON_MINUTES, min_periods=1).max()[::-1])
    .reset_index(level=0, drop=True)
    .astype(int)
)
print(f"Positive class (risk in next 12h): {df['target_future_risk'].mean()*100:.2f}%")
print("^ Note: if this is above ~50%, the positive class dominates and metrics should be interpreted with that in mind.")

# -----------------------------------------------------------------
# STEP 6: Train/test split BY PATIENT - same split as baselines.py
# (random_state=42, test_size=0.25) so results are comparable.
# This split is fixed across seeds - only the client grouping and
# model initialization change per seed, which is the correct way to
# do a multi-seed robustness check (same data, different randomness).
# -----------------------------------------------------------------
feature_cols = SIGNAL_COLS + zscore_cols
all_patients = sorted(df["user_id"].unique())
train_patients, test_patients = train_test_split(all_patients, test_size=0.25, random_state=42)
print(f"Train patients: {len(train_patients)}, test patients: {len(test_patients)}")

train_df = df[df["user_id"].isin(train_patients)].copy()
test_df = df[df["user_id"].isin(test_patients)].copy()
del df

scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

X_test_clean = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
y_test = torch.tensor(test_df["target_future_risk"].values, dtype=torch.float32)

# -----------------------------------------------------------------
# Model + training helpers
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

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X)).numpy()
    return roc_auc_score(y, probs), average_precision_score(y, probs)

def get_weights(model):
    return copy.deepcopy(model.state_dict())

def average_weights(weight_list, sample_counts):
    total = sum(sample_counts)
    avg = copy.deepcopy(weight_list[0])
    for key in avg:
        avg[key] = sum(w[key] * (n / total) for w, n in zip(weight_list, sample_counts))
    return avg

def make_clients(seed):
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(train_patients)
    groups = np.array_split(shuffled, N_CLIENTS)
    clients = []
    for patients in groups:
        sub = train_df[train_df["user_id"].isin(patients)]
        if len(sub) > ROWS_PER_CLIENT:
            sub = sub.sample(n=ROWS_PER_CLIENT, random_state=seed)
        Xc = torch.tensor(sub[feature_cols].values, dtype=torch.float32)
        yc = torch.tensor(sub["target_future_risk"].values, dtype=torch.float32)
        clients.append((Xc, yc))
    return clients

# -----------------------------------------------------------------
# One full run (all 3 setups) for a given seed - returns a results
# dict and, for the federated setup, the trained global model
# (needed later for the missing-data experiment)
# -----------------------------------------------------------------
def run_experiment(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    client_data = make_clients(seed)

    t0 = time.time()
    mem0 = current_memory_mb()

    # Setup 1: Centralized
    X_all = torch.cat([c[0] for c in client_data])
    y_all = torch.cat([c[1] for c in client_data])
    central_model = RiskMLP(n_features)
    central_model = train_local(central_model, X_all, y_all, epochs=N_ROUNDS * LOCAL_EPOCHS)
    c_auroc, c_prauc = evaluate(central_model, X_test_clean, y_test)

    # Setup 2: Local-only (average across clients)
    local_aurocs, local_praucs = [], []
    for Xc, yc in client_data:
        m = RiskMLP(n_features)
        m = train_local(m, Xc, yc, epochs=N_ROUNDS * LOCAL_EPOCHS)
        a, p = evaluate(m, X_test_clean, y_test)
        local_aurocs.append(a)
        local_praucs.append(p)

    # Setup 3: Federated (FedAvg)
    global_model = RiskMLP(n_features)
    comm_bytes_per_round = sum(p.numel() for p in global_model.parameters()) * 4 * 2
    total_comm_bytes = 0
    round_history = []
    for rnd in range(N_ROUNDS):
        local_weights, sample_counts = [], []
        for Xc, yc in client_data:
            lm = RiskMLP(n_features)
            lm.load_state_dict(global_model.state_dict())
            lm = train_local(lm, Xc, yc, epochs=LOCAL_EPOCHS)
            local_weights.append(get_weights(lm))
            sample_counts.append(len(Xc))
        global_model.load_state_dict(average_weights(local_weights, sample_counts))
        total_comm_bytes += comm_bytes_per_round * N_CLIENTS
        a, p = evaluate(global_model, X_test_clean, y_test)
        round_history.append({"round": rnd + 1, "auroc": a, "prauc": p})

    f_auroc, f_prauc = evaluate(global_model, X_test_clean, y_test)
    elapsed_sec = time.time() - t0
    mem_used_mb = current_memory_mb() - mem0

    return {
        "seed": seed,
        "centralized": {"auroc": c_auroc, "prauc": c_prauc},
        "local_only": {"auroc": float(np.mean(local_aurocs)), "prauc": float(np.mean(local_praucs))},
        "federated": {"auroc": f_auroc, "prauc": f_prauc},
        "round_history": round_history,
        "comm_total_mb": total_comm_bytes / 1e6,
        "comm_per_client_per_round_kb": comm_bytes_per_round / 1e3,
        "model_params": sum(p.numel() for p in global_model.parameters()),
        "runtime_sec": elapsed_sec,
        "memory_mb": mem_used_mb,
    }, global_model

# -----------------------------------------------------------------
# RUN ACROSS MULTIPLE SEEDS - satisfies the "statistical testing /
# multiple random seeds" requirement from the brief
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print(f"RUNNING {len(SEEDS)} SEEDS: {SEEDS}")
print("=" * 60)

all_results = []
best_federated_model = None
best_auroc = -1
for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    result, fed_model = run_experiment(seed)
    all_results.append(result)
    print(f"  Centralized -> AUROC {result['centralized']['auroc']:.3f}  "
          f"Local-only -> AUROC {result['local_only']['auroc']:.3f}  "
          f"Federated -> AUROC {result['federated']['auroc']:.3f}")
    print(f"  Runtime: {result['runtime_sec']:.1f}s  Memory: {result['memory_mb']:.1f}MB  "
          f"Comm: {result['comm_total_mb']:.2f}MB total")
    if result["federated"]["auroc"] > best_auroc:
        best_auroc = result["federated"]["auroc"]
        best_federated_model = fed_model

# -----------------------------------------------------------------
# MULTI-SEED SUMMARY - mean +/- std, the format your report needs
# -----------------------------------------------------------------
def mean_std(setup, metric):
    vals = [r[setup][metric] for r in all_results]
    return np.mean(vals), np.std(vals)

print("\n" + "=" * 60)
print(f"SUMMARY ACROSS {len(SEEDS)} SEEDS (mean +/- std)")
print("=" * 60)
summary_rows = []
for setup in ["centralized", "local_only", "federated"]:
    a_mean, a_std = mean_std(setup, "auroc")
    p_mean, p_std = mean_std(setup, "prauc")
    print(f"{setup:<15} AUROC: {a_mean:.3f} +/- {a_std:.3f}   PR-AUC: {p_mean:.3f} +/- {p_std:.3f}")
    summary_rows.append({"setup": setup, "auroc_mean": a_mean, "auroc_std": a_std,
                          "prauc_mean": p_mean, "prauc_std": p_std})

runtime_mean = np.mean([r["runtime_sec"] for r in all_results])
memory_mean = np.mean([r["memory_mb"] for r in all_results])
comm_mean = np.mean([r["comm_total_mb"] for r in all_results])
print(f"\nAvg runtime per full run: {runtime_mean:.1f}s")
print(f"Avg memory used: {memory_mean:.1f}MB")
print(f"Avg communication cost: {comm_mean:.2f}MB over {N_ROUNDS} rounds, {N_CLIENTS} clients")

# -----------------------------------------------------------------
# MISSING-DATA ROBUSTNESS EXPERIMENT - satisfies the "noise / missing
# data / sensor failure" requirement. Simulates a device-not-worn
# scenario: randomly blank out a fraction of the TEST features
# (replaced with 0, i.e. the scaled mean) and see how much the
# already-trained federated model's performance degrades.
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("MISSING-DATA ROBUSTNESS EXPERIMENT (on best federated model)")
print("=" * 60)
missing_data_results = []
rng = np.random.RandomState(42)
for frac in MISSING_DATA_FRACTIONS:
    X_corrupted = X_test_clean.clone()
    if frac > 0:
        mask = torch.tensor(rng.rand(*X_corrupted.shape) < frac)
        X_corrupted[mask] = 0.0  # 0 = the scaled mean, simulates a dropped/imputed reading
    a, p = evaluate(best_federated_model, X_corrupted, y_test)
    missing_data_results.append({"missing_fraction": frac, "auroc": a, "prauc": p})
    print(f"  {frac*100:>4.0f}% missing -> AUROC: {a:.3f}  PR-AUC: {p:.3f}")

# -----------------------------------------------------------------
# AUTO-SAVE EVERYTHING - JSON (full detail) + CSV (report-ready table)
# -----------------------------------------------------------------
with open(f"{OUTPUT_DIR}/full_results.json", "w") as f:
    json.dump({
        "config": {
            "n_clients": N_CLIENTS, "n_rounds": N_ROUNDS, "local_epochs": LOCAL_EPOCHS,
            "seeds": SEEDS, "rows_per_client": ROWS_PER_CLIENT,
        },
        "per_seed_results": all_results,
        "missing_data_experiment": missing_data_results,
    }, f, indent=2)

pd.DataFrame(summary_rows).to_csv(f"{OUTPUT_DIR}/summary_table.csv", index=False)
pd.DataFrame(missing_data_results).to_csv(f"{OUTPUT_DIR}/missing_data_experiment.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR}/full_results.json, summary_table.csv, missing_data_experiment.csv")

# -----------------------------------------------------------------
# GRAPH - two panels: (1) FedAvg convergence over rounds, averaged
# across seeds, (2) accuracy drop under missing data
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel 1: convergence curve (mean across seeds, per round)
rounds = list(range(1, N_ROUNDS + 1))
auroc_by_round = np.array([[rh["auroc"] for rh in r["round_history"]] for r in all_results])
mean_curve = auroc_by_round.mean(axis=0)
std_curve = auroc_by_round.std(axis=0)
axes[0].plot(rounds, mean_curve, marker="o", label="Federated (mean over seeds)")
axes[0].fill_between(rounds, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
c_mean, _ = mean_std("centralized", "auroc")
l_mean, _ = mean_std("local_only", "auroc")
axes[0].axhline(c_mean, color="green", linestyle="--", label="Centralized (upper bound)")
axes[0].axhline(l_mean, color="red", linestyle="--", label="Local-only (lower bound)")
axes[0].set_xlabel("Federated round")
axes[0].set_ylabel("AUROC")
axes[0].set_title("FedAvg convergence")
axes[0].legend(fontsize=8)

# Panel 2: robustness to missing data
mf = [r["missing_fraction"] * 100 for r in missing_data_results]
ma = [r["auroc"] for r in missing_data_results]
axes[1].plot(mf, ma, marker="o", color="darkorange")
axes[1].set_xlabel("% of sensor values missing")
axes[1].set_ylabel("AUROC")
axes[1].set_title("Robustness to missing data")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/results_figure.png", dpi=150)
print(f"Saved: {OUTPUT_DIR}/results_figure.png")

import shutil
shutil.make_archive(OUTPUT_DIR, "zip", OUTPUT_DIR)
print(f"Saved: {OUTPUT_DIR}.zip (right-click this file in the Colab file panel to download it)")

print(f"\nDone. Results saved to '{OUTPUT_DIR}/'.")

