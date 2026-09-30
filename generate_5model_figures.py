#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/Users/pavanaksshay/quantum"
EXP45_DIR = os.path.join(BASE_DIR, "results/exp45")
FIGURES_DIR = os.path.join(EXP45_DIR, "figures")
SUB_FIG_DIR = os.path.join(BASE_DIR, "paper/submission/figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(SUB_FIG_DIR, exist_ok=True)

# 5 Selected Models
SELECTED_MODELS = [
    "Linear SVM",
    "Logistic Regression",
    "RBF SVM",
    "Random Forest",
    "MLP"
]

MODEL_LABELS = {
    "Linear SVM": "Linear SVM",
    "Logistic Regression": "Logistic Reg.",
    "RBF SVM": "Matched RBF",
    "Random Forest": "Random Forest",
    "MLP": "MLP Neural Net"
}

COLORS = {
    "Linear SVM": "#1f77b4",
    "Logistic Regression": "#5dade2",
    "RBF SVM": "#e67e22",
    "Random Forest": "#8e44ad",
    "MLP": "#e74c3c"
}

summary_path = os.path.join(EXP45_DIR, "classical_baseline_summary.csv")
df = pd.read_csv(summary_path)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)

datasets = ["SMS", "CEAS", "MeAJOR"]

for ax_idx, rep in enumerate(["Full TF-IDF", "Reduced 8D TF-IDF"]):
    ax = axes[ax_idx]
    sub_df = df[df["Representation"] == rep]
    
    x = np.arange(len(datasets))
    width = 0.80 / len(SELECTED_MODELS)
    
    for m_idx, model in enumerate(SELECTED_MODELS):
        m_sub = sub_df[sub_df["Model"] == model].set_index("Dataset")
        f1s = [m_sub.loc[d, "Mean_F1"] if d in m_sub.index else 0 for d in datasets]
        sds = [m_sub.loc[d, "SD_F1"] if d in m_sub.index else 0 for d in datasets]
        
        c = COLORS.get(model, "#333333")
        label = MODEL_LABELS.get(model, model)
        offset = (m_idx - len(SELECTED_MODELS) / 2 + 0.5) * width
        ax.bar(x + offset, f1s, width, yerr=sds, label=label, color=c, capsize=2.5, alpha=0.9, edgecolor="black", linewidth=0.5)
        
    ax.set_xticks(x)
    ax.set_xticklabels(["SMS Spam", "CEAS 2008", "MeAJOR"], fontweight="bold")
    ax.set_title(f"({chr(97+ax_idx)}) {rep}", fontweight="bold", pad=8)
    ax.set_xlabel("Cybersecurity Corpus", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0.5, 1.03)

axes[0].set_ylabel("Test F1 Score (Mean ± SD)", fontweight="bold")
axes[1].legend(loc="lower right", framealpha=0.95)
plt.tight_layout()

for out_dir in [FIGURES_DIR, SUB_FIG_DIR]:
    fig_path = os.path.join(out_dir, "fig1_classical_f1_across_datasets.png")
    plt.savefig(fig_path)
    print(f"Saved: {fig_path}")

plt.close()
