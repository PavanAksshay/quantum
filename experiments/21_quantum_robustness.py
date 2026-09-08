"""
=============================================================================
21_quantum_robustness.py
=============================================================================
Rigorous Multi-Seed Robustness Evaluation of the RoBERTa + 2-Qubit Quantum Kernel
Pipeline vs. Tuned Classical RBF-SVM.

Research Question:
    Is the previously observed performance of:
        RoBERTa -> PCA(2) -> 2-qubit Quantum Kernel SVM
    robust across different random train/validation/test splits?

Methodological Guardrails:
    - 5 Random Seeds: [42, 123, 456, 789, 999]
    - Pool all 550 RoBERTa 768-d embeddings from results/roberta_quantum_features/
    - Stratified ~60/20/20 train/val/test splits (330 train / 110 val / 110 test)
    - Zero sample leakage across partitions (verified per seed)
    - StandardScaler & PCA(2) fitted strictly on the training partition
    - Threshold search (0.20 to 0.80, step 0.01) conducted exclusively on validation
    - Single held-out test evaluation
    - Paired Wilcoxon signed-rank hypothesis test

Author: Antigravity AI & Research Team
=============================================================================
"""

import os
import sys
import time
import json
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.fast_quantum_kernel import FastQuantumKernel

# Suppress expected user/future warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================
SEEDS = [42, 123, 456, 789, 999]
N_QUBITS = 2
N_LAYERS = 1
PCA_COMPONENTS = 2
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "metrics")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "results", "roberta_quantum_features")


# =============================================================================
# DATA POOLING & LOADING
# =============================================================================
def load_roberta_pool() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and concatenate all available RoBERTa embeddings (550 samples)
    with deterministic sample IDs.
    """
    X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(FEATURES_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(FEATURES_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

    X_pool = np.concatenate([X_train, X_val, X_test], axis=0)
    y_pool = np.concatenate([y_train, y_val, y_test], axis=0)
    sample_ids = np.arange(len(y_pool))

    print("=" * 80)
    print("ROBERTA EMBEDDING POOL LOADED")
    print("=" * 80)
    print(f"Total samples: {len(X_pool)} (Shape: {X_pool.shape})")
    print(f"Overall class distribution: Ham (0) = {np.sum(y_pool == 0)}, Spam (1) = {np.sum(y_pool == 1)}")
    return X_pool, y_pool, sample_ids


# =============================================================================
# THRESHOLD SELECTION & METRICS EVALUATION
# =============================================================================
def find_best_threshold(probabilities: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Search thresholds from 0.20 to 0.80 (step 0.01) maximizing validation F1."""
    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(0.20, 0.81, 0.01):
        preds = (probabilities >= threshold).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def evaluate_predictions(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    """Calculate standard binary classification metrics on test set."""
    preds = (probabilities >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(labels, preds)),
        "Precision": float(precision_score(labels, preds, zero_division=0)),
        "Recall": float(recall_score(labels, preds, zero_division=0)),
        "F1": float(f1_score(labels, preds, zero_division=0)),
        "PR_AUC": float(average_precision_score(labels, probabilities)),
        "ROC_AUC": float(roc_auc_score(labels, probabilities))
    }


# =============================================================================
# MAIN ROBUSTNESS EXPERIMENT
# =============================================================================
def run_robustness_experiment():
    X_pool, y_pool, sample_ids = load_roberta_pool()

    runs_results = []
    details_log = []

    print("\n" + "=" * 80)
    print(f"STARTING MULTI-SEED EVALUATION ACROSS {len(SEEDS)} SEEDS")
    print("=" * 80)

    for seed in SEEDS:
        print(f"\n>>> SEED {seed} <<<")

        # 1. Stratified 60/20/20 train/val/test split
        train_idx, temp_idx = train_test_split(
            sample_ids,
            test_size=0.40,
            random_state=seed,
            stratify=y_pool
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=seed,
            stratify=y_pool[temp_idx]
        )

        # 2. Overlap Verification
        set_train = set(train_idx)
        set_val = set(val_idx)
        set_test = set(test_idx)

        overlap_train_val = len(set_train & set_val)
        overlap_train_test = len(set_train & set_test)
        overlap_val_test = len(set_val & set_test)

        assert overlap_train_val == 0, f"Leakage detected: Train & Val overlap = {overlap_train_val}"
        assert overlap_train_test == 0, f"Leakage detected: Train & Test overlap = {overlap_train_test}"
        assert overlap_val_test == 0, f"Leakage detected: Val & Test overlap = {overlap_val_test}"

        print(f"  Split sizes: Train = {len(train_idx)}, Val = {len(val_idx)}, Test = {len(test_idx)}")
        print(f"  Overlap checks: Train∩Val={overlap_train_val}, Train∩Test={overlap_train_test}, Val∩Test={overlap_val_test} (Passed: Zero Overlap)")
        print(f"  Class counts -> Train: {dict(pd.Series(y_pool[train_idx]).value_counts())}, Val: {dict(pd.Series(y_pool[val_idx]).value_counts())}, Test: {dict(pd.Series(y_pool[test_idx]).value_counts())}")

        X_train_raw = X_pool[train_idx]
        y_train = y_pool[train_idx]
        X_val_raw = X_pool[val_idx]
        y_val = y_pool[val_idx]
        X_test_raw = X_pool[test_idx]
        y_test = y_pool[test_idx]

        # 3. StandardScaler fitted ONLY on training set
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # 4. PCA(2) fitted ONLY on training set
        pca = PCA(n_components=PCA_COMPONENTS, random_state=seed)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        exp_var = pca.explained_variance_ratio_
        cum_var = np.sum(exp_var)
        print(f"  PCA(2) Explained Variance: {exp_var} (Cumulative = {cum_var:.4f})")

        # 5. Feature Normalization using Training Bounds
        train_min = X_train_pca.min(axis=0)
        train_max = X_train_pca.max(axis=0)
        denom = train_max - train_min
        denom[denom == 0] = 1.0

        # Classical normalized features [0, 1]
        X_train_c = (X_train_pca - train_min) / denom
        X_val_c = (X_val_pca - train_min) / denom
        X_test_c = (X_test_pca - train_min) / denom

        # Quantum angles [-pi, pi]
        X_train_q = 2.0 * np.pi * X_train_c - np.pi
        X_val_q = 2.0 * np.pi * X_val_c - np.pi
        X_test_q = 2.0 * np.pi * X_test_c - np.pi

        # =====================================================================
        # MODEL 1: TUNED CLASSICAL RBF-SVM
        # =====================================================================
        print("  Evaluating Classical RBF-SVM (C=100.0, gamma=0.1)...")
        classical_model = SVC(
            kernel="rbf",
            C=100.0,
            gamma=0.1,
            class_weight="balanced",
            probability=True,
            random_state=seed
        )

        c_train_start = time.time()
        classical_model.fit(X_train_c, y_train)
        c_train_time = time.time() - c_train_start

        # Validation threshold tuning
        c_val_probs = classical_model.predict_proba(X_val_c)[:, 1]
        c_threshold, c_val_f1 = find_best_threshold(c_val_probs, y_val)

        # Held-out test evaluation
        c_infer_start = time.time()
        c_test_probs = classical_model.predict_proba(X_test_c)[:, 1]
        c_infer_time = time.time() - c_infer_start

        c_metrics = evaluate_predictions(c_test_probs, y_test, c_threshold)
        print(f"    Classical Results -> Val F1: {c_val_f1:.4f} | Test F1: {c_metrics['F1']:.4f} | PR-AUC: {c_metrics['PR_AUC']:.4f} | ROC-AUC: {c_metrics['ROC_AUC']:.4f} (Thresh: {c_threshold:.2f})")

        # =====================================================================
        # MODEL 2: 2-QUBIT QUANTUM KERNEL SVM
        # =====================================================================
        print("  Evaluating Quantum Kernel SVM (FastQuantumKernel 2-qubits)...")
        kernel = FastQuantumKernel(n_qubits=N_QUBITS, n_layers=N_LAYERS)

        # Kernel matrices
        q_train_k_start = time.time()
        K_train = kernel.matrix(X_train_q)
        q_train_k_time = time.time() - q_train_k_start

        q_val_k_start = time.time()
        K_val = kernel.matrix(X_val_q, X_train_q)
        q_val_k_time = time.time() - q_val_k_start

        q_test_k_start = time.time()
        K_test = kernel.matrix(X_test_q, X_train_q)
        q_test_k_time = time.time() - q_test_k_start

        total_kernel_time = q_train_k_time + q_val_k_time + q_test_k_time

        quantum_model = SVC(
            kernel="precomputed",
            class_weight="balanced",
            probability=True,
            random_state=seed
        )

        q_train_start = time.time()
        quantum_model.fit(K_train, y_train)
        q_train_time = time.time() - q_train_start

        # Validation threshold tuning
        q_val_probs = quantum_model.predict_proba(K_val)[:, 1]
        q_threshold, q_val_f1 = find_best_threshold(q_val_probs, y_val)

        # Held-out test evaluation
        q_infer_start = time.time()
        q_test_probs = quantum_model.predict_proba(K_test)[:, 1]
        q_infer_time = time.time() - q_infer_start

        q_metrics = evaluate_predictions(q_test_probs, y_test, q_threshold)
        print(f"    Quantum Results   -> Val F1: {q_val_f1:.4f} | Test F1: {q_metrics['F1']:.4f} | PR-AUC: {q_metrics['PR_AUC']:.4f} | ROC-AUC: {q_metrics['ROC_AUC']:.4f} (Thresh: {q_threshold:.2f})")

        # ---------------------------------------------------------------------
        # Record Run Metrics
        # ---------------------------------------------------------------------
        runs_results.append({
            "Seed": seed,
            "Model": "Classical RBF-SVM",
            "F1": c_metrics["F1"],
            "PR_AUC": c_metrics["PR_AUC"],
            "ROC_AUC": c_metrics["ROC_AUC"],
            "Accuracy": c_metrics["Accuracy"],
            "Precision": c_metrics["Precision"],
            "Recall": c_metrics["Recall"],
            "Validation_F1": c_val_f1,
            "Threshold": c_threshold,
            "Training_Time": c_train_time,
            "Inference_Time": c_infer_time,
            "Kernel_Time": 0.0
        })

        runs_results.append({
            "Seed": seed,
            "Model": "Quantum Kernel SVM",
            "F1": q_metrics["F1"],
            "PR_AUC": q_metrics["PR_AUC"],
            "ROC_AUC": q_metrics["ROC_AUC"],
            "Accuracy": q_metrics["Accuracy"],
            "Precision": q_metrics["Precision"],
            "Recall": q_metrics["Recall"],
            "Validation_F1": q_val_f1,
            "Threshold": q_threshold,
            "Training_Time": q_train_time,
            "Inference_Time": q_infer_time,
            "Kernel_Time": total_kernel_time
        })

        # Detailed per-seed log
        details_log.append({
            "seed": seed,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "train_class_counts": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().items()},
            "val_class_counts": {str(k): int(v) for k, v in pd.Series(y_val).value_counts().items()},
            "test_class_counts": {str(k): int(v) for k, v in pd.Series(y_test).value_counts().items()},
            "pca_explained_variance_ratio": [float(x) for x in exp_var],
            "pca_cumulative_variance": float(cum_var),
            "classical_rbf": {
                "threshold": c_threshold,
                "validation_f1": c_val_f1,
                "metrics": c_metrics,
                "train_time_sec": c_train_time,
                "infer_time_sec": c_infer_time
            },
            "quantum_kernel": {
                "threshold": q_threshold,
                "validation_f1": q_val_f1,
                "metrics": q_metrics,
                "svm_train_time_sec": q_train_time,
                "infer_time_sec": q_infer_time,
                "kernel_train_time_sec": q_train_k_time,
                "kernel_val_time_sec": q_val_k_time,
                "kernel_test_time_sec": q_test_k_time,
                "total_kernel_time_sec": total_kernel_time
            }
        })

    runs_df = pd.DataFrame(runs_results)

    # =========================================================================
    # STATISTICAL SUMMARY & AGGREGATION
    # =========================================================================
    summary_rows = []
    for model_name in ["Classical RBF-SVM", "Quantum Kernel SVM"]:
        m_df = runs_df[runs_df["Model"] == model_name]
        summary_rows.append({
            "Model": model_name,
            "Mean_F1": float(m_df["F1"].mean()),
            "Std_F1": float(m_df["F1"].std(ddof=1)),
            "Mean_PR_AUC": float(m_df["PR_AUC"].mean()),
            "Std_PR_AUC": float(m_df["PR_AUC"].std(ddof=1)),
            "Mean_ROC_AUC": float(m_df["ROC_AUC"].mean()),
            "Std_ROC_AUC": float(m_df["ROC_AUC"].std(ddof=1)),
            "Mean_Validation_F1": float(m_df["Validation_F1"].mean()),
            "Std_Validation_F1": float(m_df["Validation_F1"].std(ddof=1)),
            "Mean_Accuracy": float(m_df["Accuracy"].mean()),
            "Std_Accuracy": float(m_df["Accuracy"].std(ddof=1)),
            "Mean_Training_Time": float(m_df["Training_Time"].mean()),
            "Mean_Inference_Time": float(m_df["Inference_Time"].mean()),
            "Mean_Kernel_Time": float(m_df["Kernel_Time"].mean()),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Paired comparisons
    classical_f1s = runs_df[runs_df["Model"] == "Classical RBF-SVM"]["F1"].to_numpy()
    quantum_f1s = runs_df[runs_df["Model"] == "Quantum Kernel SVM"]["F1"].to_numpy()

    quantum_wins = np.sum(quantum_f1s > classical_f1s)
    quantum_win_rate = (quantum_wins / len(SEEDS)) * 100.0

    mean_paired_f1_diff = float(np.mean(quantum_f1s - classical_f1s))
    quantum_mean_f1 = float(np.mean(quantum_f1s))
    classical_mean_f1 = float(np.mean(classical_f1s))

    # Wilcoxon signed-rank test
    diffs = quantum_f1s - classical_f1s
    if np.all(diffs == 0):
        w_stat, w_pvalue = 0.0, 1.0
    else:
        try:
            w_res = stats.wilcoxon(quantum_f1s, classical_f1s)
            w_stat = float(w_res.statistic)
            w_pvalue = float(w_res.pvalue)
        except Exception as e:
            w_stat, w_pvalue = np.nan, np.nan

    # Add paired summary columns
    summary_df["Quantum_Mean_F1"] = quantum_mean_f1
    summary_df["Classical_Mean_F1"] = classical_mean_f1
    summary_df["Mean_F1_Difference"] = mean_paired_f1_diff
    summary_df["Quantum_Win_Rate"] = quantum_win_rate
    summary_df["Wilcoxon_Statistic"] = w_stat
    summary_df["Wilcoxon_PValue"] = w_pvalue

    # =========================================================================
    # SAVE ARTIFACTS
    # =========================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    runs_path = os.path.join(OUTPUT_DIR, "quantum_robustness_runs.csv")
    summary_path = os.path.join(OUTPUT_DIR, "quantum_robustness_summary.csv")
    details_path = os.path.join(OUTPUT_DIR, "quantum_robustness_details.json")

    runs_df.to_csv(runs_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with open(details_path, "w") as f:
        json.dump(details_log, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[Saved] Runs saved to:    {runs_path}")
    print(f"[Saved] Summary saved to: {summary_path}")
    print(f"[Saved] Details saved to: {details_path}")

    # =========================================================================
    # PRINT RESULTS TABLES
    # =========================================================================
    print("\n" + "=" * 80)
    print("MULTI-SEED ROBUSTNESS RESULTS")
    print("=" * 80)
    display_runs = runs_df[[
        "Seed", "Model", "F1", "PR_AUC", "ROC_AUC", "Validation_F1", "Training_Time"
    ]].copy()
    display_runs["Time (s)"] = runs_df["Training_Time"] + runs_df["Kernel_Time"]
    display_table = display_runs[["Seed", "Model", "F1", "PR_AUC", "ROC_AUC", "Validation_F1", "Time (s)"]].copy()

    for col in ["F1", "PR_AUC", "ROC_AUC", "Validation_F1"]:
        display_table[col] = display_table[col].apply(lambda x: f"{x:.4f}")
    display_table["Time (s)"] = display_table["Time (s)"].apply(lambda x: f"{x:.3f}")

    print(display_table.to_string(index=False))

    # Summary
    c_summary = summary_df[summary_df["Model"] == "Classical RBF-SVM"].iloc[0]
    q_summary = summary_df[summary_df["Model"] == "Quantum Kernel SVM"].iloc[0]

    print("\n" + "=" * 80)
    print("ROBUSTNESS SUMMARY")
    print("=" * 80)
    print(f"Classical RBF:")
    print(f"  F1      = {c_summary['Mean_F1']:.4f} ± {c_summary['Std_F1']:.4f}")
    print(f"  PR-AUC  = {c_summary['Mean_PR_AUC']:.4f} ± {c_summary['Std_PR_AUC']:.4f}")
    print(f"  ROC-AUC = {c_summary['Mean_ROC_AUC']:.4f} ± {c_summary['Std_ROC_AUC']:.4f}")
    print()
    print(f"Quantum Kernel:")
    print(f"  F1      = {q_summary['Mean_F1']:.4f} ± {q_summary['Std_F1']:.4f}")
    print(f"  PR-AUC  = {q_summary['Mean_PR_AUC']:.4f} ± {q_summary['Std_PR_AUC']:.4f}")
    print(f"  ROC-AUC = {q_summary['Mean_ROC_AUC']:.4f} ± {q_summary['Std_ROC_AUC']:.4f}")
    print()
    print(f"Quantum win rate: {quantum_win_rate:.1f}% ({quantum_wins}/{len(SEEDS)} seeds)")
    print(f"Mean paired F1 difference (Quantum - Classical): {mean_paired_f1_diff:+.4f}")
    print(f"Wilcoxon statistic: {w_stat}")
    print(f"Wilcoxon p-value:   {w_pvalue:.4f}")

    # Empirical conclusion statement
    print("-" * 80)
    if w_pvalue < 0.05:
        if mean_paired_f1_diff > 0:
            conclusion = "ROBUST AND STATISTICALLY SUPERIOR (p < 0.05)"
        else:
            conclusion = "ROBUST BUT STATISTICALLY INFERIOR (p < 0.05)"
    else:
        if abs(mean_paired_f1_diff) < 0.015 and q_summary["Std_F1"] < 0.05:
            conclusion = "ROBUST AND CONSISTENTLY COMPETITIVE (Differences are statistically non-significant, performance remains tight across random splits)"
        elif quantum_win_rate >= 40.0:
            conclusion = "COMPETITIVE BUT VARIABLE (Performance fluctuates with data splits, no statistically significant dominance)"
        else:
            conclusion = "NOT ROBUST (Classical baseline consistently dominates across splits)"

    print(f"RESEARCH CONCLUSION: {conclusion}")
    print("=" * 80)


if __name__ == "__main__":
    run_robustness_experiment()
