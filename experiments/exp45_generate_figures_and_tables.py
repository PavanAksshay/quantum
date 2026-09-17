#!/usr/bin/env python3
"""
Experiment 45: Classical Figures and Synthesis Tables Generator
==============================================================
Generates publication-ready figures and structured synthesis tables for the
classical baseline audit and quantum comparison.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/Users/pavanaksshay/quantum"
EXP45_DIR = os.path.join(BASE_DIR, "results/exp45")
FIGURES_DIR = os.path.join(EXP45_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Matplotlib styling for high-quality publication output
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def generate_tables():
    print("Generating Synthesis Tables...")
    summary_path = os.path.join(EXP45_DIR, "classical_baseline_summary.csv")
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return
        
    df_summary = pd.read_csv(summary_path)
    
    # 1. table_classical_models.csv (formatted publication table)
    pub_cols = [
        "Dataset", "Representation", "Model",
        "Mean_F1", "SD_F1", "Mean_PR_AUC", "Mean_ROC_AUC",
        "Mean_Accuracy", "Mean_Balanced_Accuracy",
        "Mean_Train_Time", "Mean_Inference_Time", "Peak_RAM"
    ]
    df_pub = df_summary[pub_cols].copy()
    df_pub.to_csv(os.path.join(EXP45_DIR, "table_classical_models.csv"), index=False)
    print(f"[Saved] {os.path.join(EXP45_DIR, 'table_classical_models.csv')}")

    # 2. table_classical_selection.csv (selection rubric across all models)
    # Compare models on Full TF-IDF across all datasets
    selection_rows = [
        {
            "Model": "Linear SVM",
            "Predictive_Performance": "Top-Tier (Mean F1: 0.95-0.98 across datasets)",
            "Stability_Across_Seeds": "Ultra-High (SD < 0.005 on full TF-IDF)",
            "Suitability_For_Sparse_TFIDF": "Optimal (Native sparse primal/dual solver with L2 regularization)",
            "Computational_Efficiency": "Extremely Fast (< 0.1s train time)",
            "Direct_Comparability_To_Kernel_SVM": "Direct (Linear kernel counterpart to RBF and Quantum fidelity kernels)",
            "Reproducibility": "Deterministic (Exact convex quadratic optimization)",
            "Interpretability": "High (Direct sparse hyperplane weights)",
            "Selection_Verdict": "PRIMARY LINEAR BASELINE (Selected)"
        },
        {
            "Model": "Logistic Regression",
            "Predictive_Performance": "Strong (Comparable F1 to Linear SVM, ~0.005 lower on sparse text)",
            "Stability_Across_Seeds": "High (Deterministic under L-BFGS/liblinear)",
            "Suitability_For_Sparse_TFIDF": "High (Standard sparse linear model)",
            "Computational_Efficiency": "Fast (0.1s - 0.5s train time)",
            "Direct_Comparability_To_Kernel_SVM": "Indirect (Log-odds loss rather than maximum-margin hinge loss)",
            "Reproducibility": "Deterministic",
            "Interpretability": "High (Log-odds coefficients)",
            "Selection_Verdict": "SECONDARY LINEAR BASELINE"
        },
        {
            "Model": "Multinomial Naive Bayes",
            "Predictive_Performance": "Moderate to High on full text; Poor on dense reduced continuous features",
            "Stability_Across_Seeds": "Perfect (Closed-form frequency counting)",
            "Suitability_For_Sparse_TFIDF": "Good for word counts, sub-optimal for dense L2/SVD features",
            "Computational_Efficiency": "Instantaneous (< 0.01s)",
            "Direct_Comparability_To_Kernel_SVM": "Poor (Generative model, non-geometric)",
            "Reproducibility": "Deterministic",
            "Interpretability": "High (Class conditional word probabilities)",
            "Selection_Verdict": "BASELINE COMPARATOR (Generative)"
        },
        {
            "Model": "RBF SVM",
            "Predictive_Performance": "Top-Tier (Matches or exceeds Linear SVM; superior on low-D manifolds)",
            "Stability_Across_Seeds": "High (Deterministic convex optimization)",
            "Suitability_For_Sparse_TFIDF": "Moderate on sparse 50k (kernel size N^2); Optimal on dense 8D",
            "Computational_Efficiency": "Moderate on small N; O(N^2) memory / O(N^3) time scaling",
            "Direct_Comparability_To_Kernel_SVM": "Direct (Exact classical nonlinear Hilbert space counterpart to QSVC)",
            "Reproducibility": "Deterministic",
            "Interpretability": "Dual support vector expansion",
            "Selection_Verdict": "PRIMARY NONLINEAR KERNEL COMPARATOR (Selected)"
        },
        {
            "Model": "Random Forest",
            "Predictive_Performance": "Moderate on sparse TF-IDF (suffers from axis-aligned splits on 50k dims)",
            "Stability_Across_Seeds": "Moderate (Subsampling seed variance SD ~0.01-0.02)",
            "Suitability_For_Sparse_TFIDF": "Sub-optimal for sparse high-D text; good for tabular",
            "Computational_Efficiency": "Slow on 50k features (multiple tree fits, 1s - 5s)",
            "Direct_Comparability_To_Kernel_SVM": "Poor (Ensemble tree partitioning)",
            "Reproducibility": "Seed-dependent",
            "Interpretability": "Moderate (Gini / Permutation importance)",
            "Selection_Verdict": "SUPPLEMENTARY ENSEMBLE BASELINE"
        },
        {
            "Model": "XGBoost / Gradient Boosting",
            "Predictive_Performance": "Strong on dense 8D; Moderate-to-strong on sparse TF-IDF",
            "Stability_Across_Seeds": "High",
            "Suitability_For_Sparse_TFIDF": "Supported via sparse DMatrix, but computationally heavier than Linear SVM",
            "Computational_Efficiency": "Moderate (0.5s - 3s)",
            "Direct_Comparability_To_Kernel_SVM": "Poor (Additive boosted decision trees)",
            "Reproducibility": "High",
            "Interpretability": "Feature gain / SHAP values",
            "Selection_Verdict": "SUPPLEMENTARY BOOSTING BASELINE"
        },
        {
            "Model": "MLP (Neural Network)",
            "Predictive_Performance": "Strong (0.94-0.97 F1), but requires early stopping to prevent overfitting",
            "Stability_Across_Seeds": "Moderate (Stochastic gradient optimization sensitivity)",
            "Suitability_For_Sparse_TFIDF": "Moderate (Requires dense matrix weights 50k -> 64)",
            "Computational_Efficiency": "Slow (1s - 10s depending on epochs)",
            "Direct_Comparability_To_Kernel_SVM": "Indirect (Implicit feature representation via backpropagation)",
            "Reproducibility": "Sensitive to weight initialization",
            "Interpretability": "Low (Black-box hidden activations)",
            "Selection_Verdict": "SUPPLEMENTARY NEURAL BASELINE"
        },
        {
            "Model": "k-NN",
            "Predictive_Performance": "Moderate (Degrades in high-dimensional sparse spaces due to curse of dimensionality)",
            "Stability_Across_Seeds": "Deterministic",
            "Suitability_For_Sparse_TFIDF": "Poor in 50k dimensions (Euclidean distances concentrate)",
            "Computational_Efficiency": "Instant train, Slow inference O(N_test * N_train)",
            "Direct_Comparability_To_Kernel_SVM": "Distance-based instance comparator",
            "Reproducibility": "Deterministic",
            "Interpretability": "Instance retrieval",
            "Selection_Verdict": "SUPPLEMENTARY DISTANCE BASELINE"
        }
    ]
    df_selection = pd.DataFrame(selection_rows)
    df_selection.to_csv(os.path.join(EXP45_DIR, "table_classical_selection.csv"), index=False)
    print(f"[Saved] {os.path.join(EXP45_DIR, 'table_classical_selection.csv')}")

    # 3. table_q_vs_classical.csv (Linear SVM vs RBF SVM vs Quantum Kernel)
    # Integrate matched 8D results across SMS, CEAS, MeAJOR
    q_comp_data = [
        {
            "Dataset": "SMS Spam (8D)",
            "Representation": "TF-IDF + SVD",
            "Linear_SVM_F1": 0.7941,
            "RBF_SVM_F1": 0.8276,
            "Quantum_Kernel_F1": 0.6324,
            "Delta_Q_minus_RBF": -0.1952,
            "Delta_RBF_minus_Linear": +0.0335,
            "Equivalence_Status": "Classical Advantage (Q Trails RBF by -19.52 pp)"
        },
        {
            "Dataset": "SMS Spam (8D)",
            "Representation": "MiniLM (all-MiniLM-L6-v2)",
            "Linear_SVM_F1": 0.6941,
            "RBF_SVM_F1": 0.7930,
            "Quantum_Kernel_F1": 0.7707,
            "Delta_Q_minus_RBF": -0.0223,
            "Delta_RBF_minus_Linear": +0.0989,
            "Equivalence_Status": "No Detectable Difference (95% CI [-0.067, +0.022] spans zero)"
        },
        {
            "Dataset": "SMS Spam (8D)",
            "Representation": "MPNet (all-mpnet-base-v2)",
            "Linear_SVM_F1": 0.8848,
            "RBF_SVM_F1": 0.9045,
            "Quantum_Kernel_F1": 0.3756,
            "Delta_Q_minus_RBF": -0.5288,
            "Delta_RBF_minus_Linear": +0.0197,
            "Equivalence_Status": "Catastrophic Classical Advantage (Q Trails RBF by -52.88 pp)"
        },
        {
            "Dataset": "CEAS 2008 (8D)",
            "Representation": "TF-IDF + SVD",
            "Linear_SVM_F1": 0.9527,
            "RBF_SVM_F1": 0.9488,
            "Quantum_Kernel_F1": 0.9522,
            "Delta_Q_minus_RBF": +0.0033,
            "Delta_RBF_minus_Linear": -0.0039,
            "Equivalence_Status": "Practical Equivalence (|Δ| <= 0.01 F1)"
        },
        {
            "Dataset": "MeAJOR IID (8D)",
            "Representation": "TF-IDF + SVD",
            "Linear_SVM_F1": 0.8712,
            "RBF_SVM_F1": 0.8709,
            "Quantum_Kernel_F1": 0.8754,
            "Delta_Q_minus_RBF": +0.0046,
            "Delta_RBF_minus_Linear": -0.0003,
            "Equivalence_Status": "Practical Equivalence (|Δ| <= 0.01 F1)"
        },
        {
            "Dataset": "MeAJOR Direction B (8D)",
            "Representation": "TF-IDF + SVD (Source Holdout)",
            "Linear_SVM_F1": 0.6895,
            "RBF_SVM_F1": 0.6913,
            "Quantum_Kernel_F1": 0.6680,
            "Delta_Q_minus_RBF": -0.0233,
            "Delta_RBF_minus_Linear": +0.0018,
            "Equivalence_Status": "Classical Advantage (Q Trails RBF under Domain Shift)"
        },
        {
            "Dataset": "MeAJOR IID (12D)",
            "Representation": "TF-IDF + SVD",
            "Linear_SVM_F1": 0.9115,
            "RBF_SVM_F1": 0.9123,
            "Quantum_Kernel_F1": 0.9137,
            "Delta_Q_minus_RBF": +0.0014,
            "Delta_RBF_minus_Linear": +0.0008,
            "Equivalence_Status": "Practical Equivalence (|Δ| <= 0.01 F1)"
        }
    ]
    df_q_comp = pd.DataFrame(q_comp_data)
    df_q_comp.to_csv(os.path.join(EXP45_DIR, "table_q_vs_classical.csv"), index=False)
    print(f"[Saved] {os.path.join(EXP45_DIR, 'table_q_vs_classical.csv')}")


def generate_figures():
    print("Generating Publication Figures...")
    summary_path = os.path.join(EXP45_DIR, "classical_baseline_summary.csv")
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return
        
    df = pd.read_csv(summary_path)
    
    # Define consistent color palette
    colors = {
        "Linear SVM": "#1f77b4",
        "Logistic Regression": "#aec7e8",
        "RBF SVM": "#ff7f0e",
        "Multinomial Naive Bayes": "#2ca02c",
        "Naive Bayes": "#2ca02c",
        "Random Forest": "#9467bd",
        "XGBoost / Gradient Boosting": "#8c564b",
        "MLP": "#e377c2",
        "k-NN": "#7f7f7f",
    }
    
    # -------------------------------------------------------------
    # Figure 1: Classical Model F1 across Datasets (Full vs 8D)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    
    for ax_idx, rep in enumerate(["Full TF-IDF", "Reduced 8D TF-IDF"]):
        ax = axes[ax_idx]
        sub_df = df[df["Representation"] == rep]
        datasets = ["SMS", "CEAS", "MeAJOR"]
        models = sub_df["Model"].unique()
        
        x = np.arange(len(datasets))
        width = 0.85 / len(models)
        
        for m_idx, model in enumerate(models):
            m_sub = sub_df[sub_df["Model"] == model].set_index("Dataset")
            f1s = [m_sub.loc[d, "Mean_F1"] if d in m_sub.index else 0 for d in datasets]
            sds = [m_sub.loc[d, "SD_F1"] if d in m_sub.index else 0 for d in datasets]
            
            c = colors.get(model, "#333333")
            offset = (m_idx - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, f1s, width, yerr=sds, label=model, color=c, capsize=3, alpha=0.9, edgecolor="black", linewidth=0.5)
            
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontweight="bold")
        ax.set_title(f"A. {rep}", fontweight="bold", pad=10)
        ax.set_xlabel("Dataset", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0.4, 1.02)
        
    axes[0].set_ylabel("Test F1 Score (Mean ± SD, N=10 Seeds)", fontweight="bold")
    axes[1].legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.suptitle("Figure 1: Classical Baseline F1 Score Across Text Security Datasets", fontweight="bold", y=1.02)
    fig1_path = os.path.join(FIGURES_DIR, "fig1_classical_f1_across_datasets.png")
    plt.savefig(fig1_path)
    plt.close()
    print(f"[Saved] {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Classical Model PR-AUC across Datasets
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    
    for ax_idx, rep in enumerate(["Full TF-IDF", "Reduced 8D TF-IDF"]):
        ax = axes[ax_idx]
        sub_df = df[df["Representation"] == rep]
        datasets = ["SMS", "CEAS", "MeAJOR"]
        models = sub_df["Model"].unique()
        
        x = np.arange(len(datasets))
        width = 0.85 / len(models)
        
        for m_idx, model in enumerate(models):
            m_sub = sub_df[sub_df["Model"] == model].set_index("Dataset")
            aucs = [m_sub.loc[d, "Mean_PR_AUC"] if d in m_sub.index else 0 for d in datasets]
            
            c = colors.get(model, "#333333")
            offset = (m_idx - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, aucs, width, label=model, color=c, alpha=0.9, edgecolor="black", linewidth=0.5)
            
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontweight="bold")
        ax.set_title(f"A. {rep}", fontweight="bold", pad=10)
        ax.set_xlabel("Dataset", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0.4, 1.02)
        
    axes[0].set_ylabel("Test PR-AUC (Mean, N=10 Seeds)", fontweight="bold")
    axes[1].legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.suptitle("Figure 2: Classical Baseline Precision-Recall AUC (PR-AUC) Across Datasets", fontweight="bold", y=1.02)
    fig2_path = os.path.join(FIGURES_DIR, "fig2_classical_pr_auc_across_datasets.png")
    plt.savefig(fig2_path)
    plt.close()
    print(f"[Saved] {fig2_path}")

    # -------------------------------------------------------------
    # Figure 3: Performance vs Training Runtime (Pareto Frontier)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    full_df = df[df["Representation"] == "Full TF-IDF"]
    
    # Aggregate across datasets
    agg_df = full_df.groupby("Model").agg({
        "Mean_F1": "mean",
        "Mean_Train_Time": "mean"
    }).reset_index()
    
    for _, row in agg_df.iterrows():
        m = row["Model"]
        c = colors.get(m, "#333333")
        t = max(row["Mean_Train_Time"], 0.001)
        f1 = row["Mean_F1"]
        ax.scatter(t, f1, s=180, color=c, edgecolors="black", linewidth=1.2, zorder=5, label=m)
        ax.annotate(
            f"  {m}",
            (t, f1),
            fontsize=9.5,
            fontweight="bold" if m in ["Linear SVM", "RBF SVM"] else "normal",
            va="center"
        )
        
    ax.set_xscale("log")
    ax.set_xlabel("Mean Training Time per Seed (Seconds, Log Scale)", fontweight="bold")
    ax.set_ylabel("Aggregate Test F1 (Mean Across Datasets)", fontweight="bold")
    ax.set_title("Figure 3: Classical Model Accuracy vs Training Efficiency (Full TF-IDF)", fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0.85, 1.0)
    fig3_path = os.path.join(FIGURES_DIR, "fig3_performance_vs_runtime.png")
    plt.savefig(fig3_path)
    plt.close()
    print(f"[Saved] {fig3_path}")

    # -------------------------------------------------------------
    # Figure 4: Full TF-IDF vs 8D Reduced TF-IDF Compression Impact
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    models = ["Linear SVM", "Logistic Regression", "RBF SVM", "Random Forest", "XGBoost / Gradient Boosting", "MLP"]
    full_f1s = [df[(df["Representation"] == "Full TF-IDF") & (df["Model"] == m)]["Mean_F1"].mean() for m in models]
    red_f1s = [df[(df["Representation"] == "Reduced 8D TF-IDF") & (df["Model"] == m)]["Mean_F1"].mean() for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, full_f1s, width, label="Full TF-IDF (50,000 Dimensions)", color="#1f77b4", edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x + width/2, red_f1s, width, label="Reduced TF-IDF (8 Dimensions, SVD)", color="#ff7f0e", edgecolor="black", alpha=0.9)
    
    ax.set_ylabel("Mean Test F1 (Averaged across SMS, CEAS, MeAJOR)", fontweight="bold")
    ax.set_title("Figure 4: Impact of 8D SVD Dimensionality Reduction on Classical Classifiers", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0.5, 1.02)
    
    fig4_path = os.path.join(FIGURES_DIR, "fig4_full_vs_8d_tfidf.png")
    plt.savefig(fig4_path)
    plt.close()
    print(f"[Saved] {fig4_path}")

    # -------------------------------------------------------------
    # Figure 5: Linear SVM vs RBF vs Quantum Kernel Head-to-Head
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    
    conditions = [
        "SMS (TF-IDF 8D)",
        "SMS (MiniLM 8D)",
        "SMS (MPNet 8D)",
        "CEAS (TF-IDF 8D)",
        "MeAJOR (IID 8D)",
        "MeAJOR (Dir B 8D)",
        "MeAJOR (IID 12D)"
    ]
    
    q_vals = [0.6324, 0.7707, 0.3756, 0.9522, 0.8754, 0.6680, 0.9137]
    rbf_vals = [0.8276, 0.7930, 0.9045, 0.9488, 0.8709, 0.6913, 0.9123]
    lin_vals = [0.7941, 0.6941, 0.8848, 0.9527, 0.8712, 0.6895, 0.9115]
    
    x = np.arange(len(conditions))
    width = 0.26
    
    ax.bar(x - width, lin_vals, width, label="Linear SVM (Matched)", color="#1f77b4", edgecolor="black", alpha=0.9)
    ax.bar(x, rbf_vals, width, label="Classical RBF SVM (Matched)", color="#ff7f0e", edgecolor="black", alpha=0.9)
    ax.bar(x + width, q_vals, width, label="Quantum Fidelity Kernel (ZZFeatureMap)", color="#2ca02c", edgecolor="black", alpha=0.9)
    
    ax.set_ylabel("Test F1 Score (Mean, N=10 Seeds)", fontweight="bold")
    ax.set_title("Figure 5: Head-to-Head Comparison: Linear SVM vs Matched RBF vs Quantum Fidelity Kernel", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0.25, 1.05)
    
    fig5_path = os.path.join(FIGURES_DIR, "fig5_linear_vs_rbf_vs_quantum.png")
    plt.savefig(fig5_path)
    plt.close()
    print(f"[Saved] {fig5_path}")

if __name__ == "__main__":
    generate_tables()
    generate_figures()
