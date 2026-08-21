"""
COMPARARE MODELE


"""

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------
# Rezultatele 
# -----------------------------------------------------------------
RESULTS = {
    "B1 - Threshold rule":        {"auroc": 0.561, "prauc": 0.849},
    "B2 - Logistic Regression":   {"auroc": 0.678, "prauc": 0.909},
    "B3 - Random Forest":         {"auroc": 0.907, "prauc": 0.979},
    "B4 - Isolation Forest":      {"auroc": 0.655, "prauc": 0.906},
    "B5 - Gradient Boosting":     {"auroc": 0.908, "prauc": 0.979},
    "Final (RF + trend + unc.)":  {"auroc": 0.933, "prauc": 0.985},
}

models = list(RESULTS.keys())
auroc_vals = [RESULTS[m]["auroc"] for m in models]
prauc_vals = [RESULTS[m]["prauc"] for m in models]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(x - width/2, auroc_vals, width, label="AUROC", color="#4C72B0")
bars2 = ax.bar(x + width/2, prauc_vals, width, label="PR-AUC", color="#DD8452")


ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)

ax.set_ylabel("Scor")
ax.set_title("Comparatie modele - forecasting risc PEM la 12h\n(80 pacienti, 10.37M randuri)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha="right")
ax.set_ylim(0, 1.05)
ax.legend()


for bars in (bars1, bars2):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=200)
print("Salvat: model_comparison.png")
plt.show()
