#!/usr/bin/env python3
"""
Experiment 41: Controlled Representation Screening (3 Seeds)
============================================================
Investigates: "Does quantum-kernel behavior depend more strongly on text representation than on kernel choice?"

Matrix:
  - 3 Datasets: SMS, CEAS, MeAJOR
  - 4 Candidate Representations: TF-IDF, MiniLM, RoBERTa, MPNet
  - 3 Models: Linear SVM, Classical RBF SVM, Quantum Fidelity Kernel SVM
  - 3 Seeds: 42, 123, 456
  - Target Dimension: 8D

Zero Test Leakage Guarantee:
  - Preprocessing, SVD projections, and Scalers fit strictly on Training splits.
  - Validation set used exclusively for 200-step decision threshold selection.
  - Test set evaluated strictly out-of-sample.
"""

import os
import sys
import time
import json
import resource
import platform
import subprocess
import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score
)

# Project base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app.model_engine.quantum import simulate_zz_feature_map, compute_von_neumann_entropy
from app.representations.registry import representation_registry

# Output directory for Exp 41
EXP41_DIR = os.path.join(BASE_DIR, "results/exp41")
CACHE_DIR = os.path.join(EXP41_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Datasets and Seeds
DATASETS = ["sms", "ceas", "meajor"]
REPRESENTATIONS = ["tfidf", "minilm", "roberta", "mpnet"]
MODELS = ["linear", "rbf", "quantum"]
SEEDS = [42, 123, 456]
DIMENSION = 8

# Simulation Device
device = torch.device("cpu")


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR).decode("utf-8").strip()
    except Exception:
        return "UNKNOWN_UNTRACKED"


def load_dataset_splits(dataset_name: str, seed: int, sample_size: int = 600):
    """
    Loads frozen train, validation, and test splits.
    Downsamples deterministically per seed for screening runtime tractability while preserving class balance.
    """
    split_dir = os.path.join(BASE_DIR, "results/frozen_splits", dataset_name)
    df_train = pd.read_csv(os.path.join(split_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(split_dir, "validation.csv"))
    df_test = pd.read_csv(os.path.join(split_dir, "test.csv"))

    def sample_stratified(df, n, s):
        if len(df) <= n:
            return df.sample(frac=1.0, random_state=s).reset_index(drop=True)
        fractions = df["target"].value_counts(normalize=True)
        indices = []
        for t_val, frac in fractions.items():
            sub = df[df["target"] == t_val]
            n_sub = int(round(n * frac))
            sampled = sub.sample(n=min(n_sub, len(sub)), random_state=s)
            indices.extend(sampled.index.tolist())
        return df.loc[indices].sample(frac=1.0, random_state=s).reset_index(drop=True)

    tr_sub = sample_stratified(df_train, sample_size, seed)
    va_sub = sample_stratified(df_val, sample_size // 2, seed)
    te_sub = sample_stratified(df_test, sample_size // 2, seed)

    return tr_sub, va_sub, te_sub


def get_or_compute_dense_embeddings(dataset_name: str, rep_id: str, texts: list, split_name: str, seed: int) -> np.ndarray:
    """
    Caches and retrieves dense transformer embeddings to avoid redundant passes.
    """
    rep_cache_dir = os.path.join(CACHE_DIR, dataset_name, rep_id)
    os.makedirs(rep_cache_dir, exist_ok=True)
    cache_path = os.path.join(rep_cache_dir, f"{split_name}_seed_{seed}.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path)

    rep = representation_registry.get(rep_id)
    dense_emb = rep.encode_original(texts)
    np.save(cache_path, dense_emb)
    return dense_emb


def compute_quantum_gram_matrix(states_1: torch.Tensor, states_2: torch.Tensor) -> np.ndarray:
    """Computes exact complex128 statevector fidelity matrix."""
    inner = torch.matmul(states_1, states_2.conj().T)
    K = (torch.abs(inner) ** 2).cpu().numpy().astype(np.float64)
    return np.clip(K, 0.0, 1.0)


def select_best_threshold(y_val: np.ndarray, val_scores: np.ndarray, num_steps: int = 200) -> tuple:
    """200-step grid search for optimal F1 threshold exclusively on validation set."""
    thresholds = np.linspace(val_scores.min(), val_scores.max(), num_steps)
    best_f1, best_th = -1.0, 0.0
    for th in thresholds:
        f1 = f1_score(y_val, (val_scores >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return float(best_th), float(best_f1)


def evaluate_predictions(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.0) -> dict:
    """Computes all primary and secondary evaluation metrics."""
    preds = (scores >= threshold).astype(int)
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]
    margin = float(pos_scores.mean() - neg_scores.mean()) if len(pos_scores) > 0 and len(neg_scores) > 0 else 0.0

    return {
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0,
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0,
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "decision_margin": float(margin),
        "threshold": float(threshold)
    }


def compute_geometry_diagnostics(states_test: torch.Tensor, K_test: np.ndarray, K_rbf: np.ndarray, y_test: np.ndarray) -> dict:
    """Computes geometric entropy, diversity, target alignment, and Gram correlation."""
    N = states_test.shape[0]
    dim = int(np.log2(states_test.shape[1]))

    # 1. Dispersion Entropy (Mean von Neumann entropy across test statevectors)
    entropies = [compute_von_neumann_entropy(states_test[i], n_qubits=dim) for i in range(min(50, N))]
    mean_entropy = float(np.mean(entropies))

    # 2. Kernel Diversity (Standard deviation of off-diagonal elements)
    off_diag_mask = ~np.eye(K_test.shape[0], dtype=bool)
    if off_diag_mask.sum() > 0:
        diversity = float(np.std(K_test[off_diag_mask]))
    else:
        diversity = 0.0

    # 3. Target-Label Alignment A(K, y y^T) = <K, y y^T> / (||K||_F ||y y^T||_F)
    y_centered = (2 * y_test - 1).astype(np.float64)
    Y = np.outer(y_centered, y_centered)
    
    def alignment(K_mat):
        frob_K = np.linalg.norm(K_mat, 'fro')
        frob_Y = np.linalg.norm(Y, 'fro')
        if frob_K > 0 and frob_Y > 0:
            return float(np.sum(K_mat * Y) / (frob_K * frob_Y))
        return 0.0

    align_q = alignment(K_test)
    align_rbf = alignment(K_rbf)

    # 4. Gram Pearson correlation
    flat_q = K_test.flatten()
    flat_rbf = K_rbf.flatten()
    if np.std(flat_q) > 1e-8 and np.std(flat_rbf) > 1e-8:
        gram_r = float(np.corrcoef(flat_q, flat_rbf)[0, 1])
    else:
        gram_r = 0.0

    return {
        "dispersion_entropy_bits": round(mean_entropy, 4),
        "kernel_diversity": round(diversity, 4),
        "target_label_alignment": round(align_q, 4),
        "target_label_alignment_rbf": round(align_rbf, 4),
        "gram_pearson_r": round(gram_r, 4)
    }


def run_experiment_41():
    print("=" * 80)
    print("EXPERIMENT 41: CONTROLLED REPRESENTATION SCREENING (3 SEEDS)")
    print(f"Datasets: {DATASETS} | Reps: {REPRESENTATIONS} | Seeds: {SEEDS} | Dim: {DIMENSION}D")
    print("=" * 80)

    screening_results = []
    geometry_results = []
    runtime_records = []
    start_total_time = time.time()

    for ds_name in DATASETS:
        print(f"\n>>> PROCESSING DATASET: {ds_name.upper()} <<<")

        for seed in SEEDS:
            print(f"\n  --- Seed {seed} ---")
            df_tr, df_va, df_te = load_dataset_splits(ds_name, seed)
            
            texts_tr = df_tr["text"].fillna("").astype(str).tolist()
            texts_va = df_va["text"].fillna("").astype(str).tolist()
            texts_te = df_te["text"].fillna("").astype(str).tolist()

            y_tr = df_tr["target"].to_numpy().astype(int)
            y_va = df_va["target"].to_numpy().astype(int)
            y_te = df_te["target"].to_numpy().astype(int)

            for rep_id in REPRESENTATIONS:
                t0_rep = time.time()
                rep = representation_registry.get(rep_id)
                meta = rep.get_metadata()

                # 1. Feature Representation Extraction & Leakage-Safe Projection
                if rep.is_sparse:
                    # TF-IDF (fit vectorizer strictly on training data)
                    tfidf = TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                        max_features=50000,
                        norm="l2"
                    )
                    X_tr_raw = tfidf.fit_transform(texts_tr)
                    X_va_raw = tfidf.transform(texts_va)
                    X_te_raw = tfidf.transform(texts_te)
                    actual_orig_dim = X_tr_raw.shape[1]
                else:
                    # Dense Transformer (extract/load cached frozen embeddings)
                    X_tr_raw = get_or_compute_dense_embeddings(ds_name, rep_id, texts_tr, "train", seed)
                    X_va_raw = get_or_compute_dense_embeddings(ds_name, rep_id, texts_va, "val", seed)
                    X_te_raw = get_or_compute_dense_embeddings(ds_name, rep_id, texts_te, "test", seed)
                    actual_orig_dim = X_tr_raw.shape[1]

                t_emb = time.time() - t0_rep

                # 2. TruncatedSVD + StandardScaler Projection (fit strictly on training data)
                t0_proj = time.time()
                svd = TruncatedSVD(n_components=DIMENSION, random_state=seed)
                scaler = StandardScaler()

                X_tr_8d = scaler.fit_transform(svd.fit_transform(X_tr_raw))
                X_va_8d = scaler.transform(svd.transform(X_va_raw))
                X_te_8d = scaler.transform(svd.transform(X_te_raw))

                # MinMax Phase Encoding to [0, pi] (computed using train bounds)
                min_v = float(np.min(X_tr_8d))
                max_v = float(np.max(X_tr_8d))
                span = max(max_v - min_v, 1e-8)
                
                X_tr_phase = np.clip(((X_tr_8d - min_v) / span) * np.pi, 0.0, np.pi)
                X_va_phase = np.clip(((X_va_8d - min_v) / span) * np.pi, 0.0, np.pi)
                X_te_phase = np.clip(((X_te_8d - min_v) / span) * np.pi, 0.0, np.pi)

                t_proj = time.time() - t0_proj

                # Model 1: Linear SVM baseline
                t0_lin = time.time()
                clf_lin = LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=2000, dual="auto")
                clf_lin.fit(X_tr_8d, y_tr)
                t_fit_lin = time.time() - t0_lin

                sc_va_lin = clf_lin.decision_function(X_va_8d)
                th_lin, _ = select_best_threshold(y_va, sc_va_lin)

                t0_inf_lin = time.time()
                sc_te_lin = clf_lin.decision_function(X_te_8d)
                t_inf_lin = time.time() - t0_inf_lin

                m_lin = evaluate_predictions(y_te, sc_te_lin, th_lin)

                # Model 2: Classical Gaussian RBF SVM
                t0_rbf = time.time()
                clf_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=seed)
                clf_rbf.fit(X_tr_8d, y_tr)
                t_fit_rbf = time.time() - t0_rbf

                sc_va_rbf = clf_rbf.decision_function(X_va_8d)
                th_rbf, _ = select_best_threshold(y_va, sc_va_rbf)

                t0_inf_rbf = time.time()
                sc_te_rbf = clf_rbf.decision_function(X_te_8d)
                t_inf_rbf = time.time() - t0_inf_rbf

                m_rbf = evaluate_predictions(y_te, sc_te_rbf, th_rbf)

                # Model 3: Quantum Fidelity Kernel SVM (2-Layer Cyclic ZZFeatureMap)
                t0_qk = time.time()
                states_tr = simulate_zz_feature_map(torch.tensor(X_tr_phase, dtype=torch.float64, device=device), DIMENSION)
                states_va = simulate_zz_feature_map(torch.tensor(X_va_phase, dtype=torch.float64, device=device), DIMENSION)
                states_te = simulate_zz_feature_map(torch.tensor(X_te_phase, dtype=torch.float64, device=device), DIMENSION)

                K_tr = compute_quantum_gram_matrix(states_tr, states_tr)
                np.fill_diagonal(K_tr, 1.0)
                K_va = compute_quantum_gram_matrix(states_va, states_tr)
                K_te = compute_quantum_gram_matrix(states_te, states_tr)
                t_k_q = time.time() - t0_qk

                t0_fit_q = time.time()
                clf_q = SVC(kernel="precomputed", C=1.0, class_weight="balanced", random_state=seed)
                clf_q.fit(K_tr, y_tr)
                t_fit_q = time.time() - t0_fit_q

                sc_va_q = clf_q.decision_function(K_va)
                th_q, _ = select_best_threshold(y_va, sc_va_q)

                t0_inf_q = time.time()
                sc_te_q = clf_q.decision_function(K_te)
                t_inf_q = time.time() - t0_inf_q

                m_q = evaluate_predictions(y_te, sc_te_q, th_q)

                # Square Test-Test Gram Matrices for Geometry Diagnostics
                K_te_te = compute_quantum_gram_matrix(states_te, states_te)
                np.fill_diagonal(K_te_te, 1.0)

                gamma_val = 1.0 / (DIMENSION * float(np.var(X_tr_8d))) if np.var(X_tr_8d) > 0 else 1.0
                dist_sq_te = np.sum((X_te_8d[:, None, :] - X_te_8d[None, :, :]) ** 2, axis=-1)
                K_rbf_te_te = np.exp(-gamma_val * dist_sq_te)

                # Geometry Diagnostics
                geom = compute_geometry_diagnostics(states_te, K_te_te, K_rbf_te_te, y_te)
                geometry_results.append({
                    "dataset": ds_name,
                    "representation": rep_id,
                    "seed": seed,
                    "dimension": DIMENSION,
                    **geom
                })

                # Record individual screening results
                base_info = {
                    "dataset": ds_name,
                    "representation": rep_id,
                    "representation_name": meta["name"],
                    "representation_family": meta["representation_type"],
                    "original_dimension": actual_orig_dim,
                    "projected_dimension": DIMENSION,
                    "is_sparse": meta["is_sparse"],
                    "seed": seed,
                    "dimension": DIMENSION,
                    "canonical_status": "CANONICAL" if meta["is_canonical"] else "EXPLORATORY"
                }

                screening_results.append({
                    **base_info,
                    "model": "Linear SVM",
                    "runtime_sec": round(t_emb + t_proj + t_fit_lin + t_inf_lin, 4),
                    **m_lin
                })
                screening_results.append({
                    **base_info,
                    "model": "Classical RBF",
                    "runtime_sec": round(t_emb + t_proj + t_fit_rbf + t_inf_rbf, 4),
                    **m_rbf
                })
                screening_results.append({
                    **base_info,
                    "model": "Quantum Fidelity Kernel",
                    "runtime_sec": round(t_emb + t_proj + t_k_q + t_fit_q + t_inf_q, 4),
                    **m_q
                })

                # Detailed runtime logging
                runtime_records.append({
                    "dataset": ds_name,
                    "representation": rep_id,
                    "seed": seed,
                    "embedding_time_sec": round(t_emb, 4),
                    "projection_time_sec": round(t_proj, 4),
                    "quantum_kernel_time_sec": round(t_k_q, 4),
                    "quantum_train_time_sec": round(t_fit_q, 4),
                    "quantum_inf_time_sec": round(t_inf_q, 4),
                    "rbf_train_time_sec": round(t_fit_rbf, 4),
                    "rbf_inf_time_sec": round(t_inf_rbf, 4),
                    "linear_train_time_sec": round(t_fit_lin, 4),
                    "linear_inf_time_sec": round(t_inf_lin, 4)
                })

                delta_f1 = m_q["f1"] - m_rbf["f1"]
                print(f"    [{rep_id.upper():7s}] Q_F1={m_q['f1']:.4f} | RBF_F1={m_rbf['f1']:.4f} | Lin_F1={m_lin['f1']:.4f} | Δ(Q-RBF)={delta_f1:+.4f}")

    # Build DataFrames
    df_screening = pd.DataFrame(screening_results)
    df_geometry = pd.DataFrame(geometry_results)
    df_runtime = pd.DataFrame(runtime_records)

    # Export Primary Raw Results
    df_screening.to_csv(os.path.join(EXP41_DIR, "exp41_screening_results.csv"), index=False)
    df_geometry.to_csv(os.path.join(EXP41_DIR, "exp41_geometry.csv"), index=False)
    df_runtime.to_csv(os.path.join(EXP41_DIR, "exp41_runtime.csv"), index=False)

    # Compute Aggregated Summary Table (Mean ± Std over 3 seeds)
    summary_rows = []
    for (ds, rep, model), g in df_screening.groupby(["dataset", "representation", "model"]):
        f1_mean = g["f1"].mean()
        f1_std = g["f1"].std()
        pr_mean = g["pr_auc"].mean()
        roc_mean = g["roc_auc"].mean()
        rt_mean = g["runtime_sec"].mean()
        summary_rows.append({
            "dataset": ds,
            "representation": rep,
            "model": model,
            "f1_mean": round(float(f1_mean), 4),
            "f1_std": round(float(f1_std), 4),
            "pr_auc_mean": round(float(pr_mean), 4),
            "roc_auc_mean": round(float(roc_mean), 4),
            "runtime_mean_sec": round(float(rt_mean), 4),
            "canonical_status": g["canonical_status"].iloc[0]
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(EXP41_DIR, "exp41_summary.csv"), index=False)

    # Compute Representation Comparison & Ranking Inversion Analysis
    comparison_rows = []
    for ds in DATASETS:
        q_rbf_diffs = []
        for rep in REPRESENTATIONS:
            q_f1 = df_summary[(df_summary["dataset"] == ds) & (df_summary["representation"] == rep) & (df_summary["model"] == "Quantum Fidelity Kernel")]["f1_mean"].values[0]
            rbf_f1 = df_summary[(df_summary["dataset"] == ds) & (df_summary["representation"] == rep) & (df_summary["model"] == "Classical RBF")]["f1_mean"].values[0]
            lin_f1 = df_summary[(df_summary["dataset"] == ds) & (df_summary["representation"] == rep) & (df_summary["model"] == "Linear SVM")]["f1_mean"].values[0]
            
            diff = q_f1 - rbf_f1
            q_rbf_diffs.append((rep, diff))

            comparison_rows.append({
                "dataset": ds,
                "representation": rep,
                "quantum_f1": q_f1,
                "rbf_f1": rbf_f1,
                "linear_f1": lin_f1,
                "delta_f1_q_minus_rbf": round(diff, 4),
                "observed_quantum_edge": "Quantum Edge" if diff > 0.01 else ("RBF Advantage" if diff < -0.01 else "Practical Parity"),
                "canonical_status": "CANONICAL" if rep == "tfidf" else ("CANONICAL_ABLATION" if (rep == "roberta" and ds == "ceas") else "EXPLORATORY")
            })

        # Calculate representation-conditioned variation range for this dataset
        diff_vals = [d[1] for d in q_rbf_diffs]
        rep_range = max(diff_vals) - min(diff_vals)
        has_rank_reversal = (min(diff_vals) < -0.005) and (max(diff_vals) > 0.005)
        print(f"\nDataset {ds.upper()}: Range of Q-RBF across representations = {rep_range:.4f} (Rank Reversal: {has_rank_reversal})")

    df_comparison = pd.DataFrame(comparison_rows)
    df_comparison.to_csv(os.path.join(EXP41_DIR, "exp41_representation_comparison.csv"), index=False)

    # Write Metadata JSON
    metadata = {
        "experiment": "Exp41_Representation_Screening",
        "objective": "Screen representation-conditioned variation across TF-IDF, MiniLM, RoBERTa, MPNet",
        "date_executed": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_runtime_seconds": round(time.time() - start_total_time, 2),
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "system_platform": platform.platform(),
        "processor": platform.processor(),
        "seeds": SEEDS,
        "dimension": DIMENSION,
        "datasets": DATASETS,
        "representations": REPRESENTATIONS,
        "models": MODELS,
        "peak_ram_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), 2)
    }
    with open(os.path.join(EXP41_DIR, "exp41_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("EXPERIMENT 41 SCREENING COMPLETE!")
    print(f"Results successfully saved to: {EXP41_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_41()
