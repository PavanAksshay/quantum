#!/usr/bin/env python3
"""
Experiment 42: Confirmatory Representation-Kernel Analysis (10 Seeds)
=====================================================================
Primary Research Question:
  "Does text representation materially alter the relative behavior of quantum fidelity kernels versus classical RBF kernels?"

Secondary Research Question:
  "Are the observed effects associated with measurable changes in kernel geometry?"

Promoted Candidates (3):
  A. SMS + MiniLM + 8D (Confirm/Reject Exp41 Q-vs-RBF rank reversal)
  B. SMS + MPNet + 8D (Confirm/Reject severe quantum geometric degradation)
  C. CEAS + TF-IDF + 8D (Confirm/Reject small positive Q-vs-RBF effect in high-diversity sparse lexical baseline)
  [Baseline: SMS + TF-IDF + 8D for representation effect comparison]

Seeds (10):
  [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

Strict Protocols:
  - 10-seed paired evaluation
  - Preprocessing, SVD, and Scaler fit strictly on Training partition
  - 200-step validation-only decision threshold selection
  - Test set evaluated strictly out-of-sample
  - Exact PyTorch complex128 2-layer cyclic ZZFeatureMap statevector simulation
  - Comprehensive Hilbert space geometry diagnostics (square test-test Gram matrices)
  - Rigorous inferential statistics: Paired permutation test (10k), Bootstrap CI (10k), Student-t CI, BH-FDR
  - Practical equivalence margin: epsilon = 0.01 F1
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
import scipy
import scipy.stats as stats
from scipy.linalg import eigvalsh
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
    recall_score,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app.model_engine.quantum import simulate_zz_feature_map, compute_von_neumann_entropy
from app.representations.registry import representation_registry

EXP42_DIR = os.path.join(BASE_DIR, "results/exp42")
CACHE_DIR = os.path.join(EXP42_DIR, "cache")
FIGS_DIR = os.path.join(EXP42_DIR, "figures")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# 10 Canonical Seeds
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

# Candidates to evaluate
CANDIDATES = [
    {"candidate_id": "A", "dataset": "sms", "representation": "minilm", "name": "SMS + MiniLM (8D)", "purpose": "Test Q-vs-RBF rank reversal"},
    {"candidate_id": "B", "dataset": "sms", "representation": "mpnet", "name": "SMS + MPNet (8D)", "purpose": "Test geometric degradation"},
    {"candidate_id": "C", "dataset": "ceas", "representation": "tfidf", "name": "CEAS + TF-IDF (8D)", "purpose": "Test sparse lexical reference"},
    {"candidate_id": "REF_SMS_TFIDF", "dataset": "sms", "representation": "tfidf", "name": "SMS + TF-IDF (8D)", "purpose": "Baseline for SMS representation shift"},
]

DIMENSION = 8
EPSILON = 0.01  # Practical equivalence margin

device = torch.device("cpu")


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR).decode("utf-8").strip()
    except Exception:
        return "UNKNOWN_UNTRACKED"


def load_dataset_splits(dataset_name: str, seed: int, sample_size: int = 600):
    """
    Loads frozen train, validation, and test splits.
    Downsamples deterministically per seed for runtime tractability while preserving exact class balance.
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


def get_or_create_embeddings(dataset_name: str, rep_name: str, seed: int, df_train, df_val, df_test):
    """
    Generates or loads cached raw embeddings.
    For TF-IDF, vectorizer is strictly fit on train only.
    For Transformers, representations are generated and cached per seed.
    """
    cache_subdir = os.path.join(CACHE_DIR, dataset_name, rep_name)
    os.makedirs(cache_subdir, exist_ok=True)
    
    # Also check Exp41 cache for reuse
    exp41_cache = os.path.join(BASE_DIR, "results/exp41/cache", dataset_name, rep_name)

    train_path = os.path.join(cache_subdir, f"train_seed_{seed}.npy")
    val_path = os.path.join(cache_subdir, f"val_seed_{seed}.npy")
    test_path = os.path.join(cache_subdir, f"test_seed_{seed}.npy")

    exp41_tr = os.path.join(exp41_cache, f"train_seed_{seed}.npy")
    exp41_va = os.path.join(exp41_cache, f"val_seed_{seed}.npy")
    exp41_te = os.path.join(exp41_cache, f"test_seed_{seed}.npy")

    if rep_name == "tfidf":
        vec = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=50000,
            norm="l2"
        )
        X_tr = vec.fit_transform(df_train["text"].tolist()).toarray().astype(np.float32)
        X_va = vec.transform(df_val["text"].tolist()).toarray().astype(np.float32)
        X_te = vec.transform(df_test["text"].tolist()).toarray().astype(np.float32)
        original_dim = X_tr.shape[1]
        model_info = {"model_name": "TfidfVectorizer", "version": sklearn.__version__, "original_dim": original_dim}
        return X_tr, X_va, X_te, model_info

    # Dense Transformer representations
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        X_tr = np.load(train_path)
        X_va = np.load(val_path)
        X_te = np.load(test_path)
        original_dim = X_tr.shape[1]
        model_info = {"model_name": rep_name, "version": "cached", "original_dim": original_dim}
        return X_tr, X_va, X_te, model_info
    elif os.path.exists(exp41_tr) and os.path.exists(exp41_va) and os.path.exists(exp41_te):
        X_tr = np.load(exp41_tr)
        X_va = np.load(exp41_va)
        X_te = np.load(exp41_te)
        # save into exp42 cache as well
        np.save(train_path, X_tr)
        np.save(val_path, X_va)
        np.save(test_path, X_te)
        original_dim = X_tr.shape[1]
        model_info = {"model_name": rep_name, "version": "cached_exp41", "original_dim": original_dim}
        return X_tr, X_va, X_te, model_info

    rep_obj = representation_registry.get(rep_name)

    t0 = time.time()
    X_tr = rep_obj.encode_original(df_train["text"].tolist())
    X_va = rep_obj.encode_original(df_val["text"].tolist())
    X_te = rep_obj.encode_original(df_test["text"].tolist())

    np.save(train_path, X_tr)
    np.save(val_path, X_va)
    np.save(test_path, X_te)

    original_dim = X_tr.shape[1]
    model_info = {
        "model_name": getattr(rep_obj, "name", rep_name),
        "version": "1.0.0",
        "original_dim": original_dim,
        "extraction_time_s": time.time() - t0
    }
    return X_tr, X_va, X_te, model_info


def project_to_target_dimension(X_tr, X_va, X_te, target_dim: int, seed: int):
    """
    Applies TruncatedSVD and StandardScaler fit STRICTLY on training split.
    Maps to [0, pi] for quantum angle encoding.
    """
    svd = TruncatedSVD(n_components=target_dim, random_state=seed)
    scaler = StandardScaler()

    Z_tr = svd.fit_transform(X_tr)
    Z_va = svd.transform(X_va)
    Z_te = svd.transform(X_te)

    Z_tr = scaler.fit_transform(Z_tr)
    Z_va = scaler.transform(Z_va)
    Z_te = scaler.transform(Z_te)

    # Angle mapping to [0, pi] using training boundaries
    mins = Z_tr.min(axis=0)
    maxs = Z_tr.max(axis=0)
    ranges = np.where(maxs - mins > 1e-8, maxs - mins, 1.0)

    A_tr = np.clip((Z_tr - mins) / ranges * np.pi, 0, np.pi)
    A_va = np.clip((Z_va - mins) / ranges * np.pi, 0, np.pi)
    A_te = np.clip((Z_te - mins) / ranges * np.pi, 0, np.pi)

    return Z_tr, Z_va, Z_te, A_tr, A_va, A_te


def compute_quantum_gram_matrix(A_rows: np.ndarray, A_cols: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Computes exact PyTorch complex128 statevector fidelity Gram matrix.
    K(i, j) = |<psi(x_i)|psi(x_j)>|^2
    """
    t_rows = torch.tensor(A_rows, dtype=torch.float64, device=device)
    states_rows = simulate_zz_feature_map(t_rows, n_qubits=n_qubits)
    B_rows = states_rows.shape[0]
    states_rows_flat = states_rows.reshape(B_rows, -1)

    if A_rows is A_cols or (len(A_rows) == len(A_cols) and np.array_equal(A_rows, A_cols)):
        states_cols_flat = states_rows_flat
        B_cols = B_rows
    else:
        t_cols = torch.tensor(A_cols, dtype=torch.float64, device=device)
        states_cols = simulate_zz_feature_map(t_cols, n_qubits=n_qubits)
        B_cols = states_cols.shape[0]
        states_cols_flat = states_cols.reshape(B_cols, -1)

    # Inner products in PyTorch complex128
    inner_prods = torch.matmul(states_rows_flat, states_cols_flat.conj().T)
    gram = (inner_prods.abs() ** 2).cpu().numpy().astype(np.float64)
    return np.clip(gram, 0.0, 1.0)


def compute_comprehensive_geometry(K_q_test: np.ndarray, K_rbf_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Computes Hilbert space geometric diagnostics on square test-test Gram matrices.
    """
    N = K_q_test.shape[0]
    
    # 1. Quantum Diagnostics
    q_sym_err = float(np.max(np.abs(K_q_test - K_q_test.T)))
    q_finite = bool(np.all(np.isfinite(K_q_test)))
    q_diag = np.diag(K_q_test)
    q_diag_min = float(np.min(q_diag))
    q_diag_max = float(np.max(q_diag))
    q_diag_mean = float(np.mean(q_diag))

    # Eigenvalue spectrum of Quantum Gram matrix
    q_eigvals = np.linalg.eigvalsh(K_q_test)
    q_psd = bool(np.min(q_eigvals) >= -1e-6)
    q_eig_pos = np.clip(q_eigvals, 1e-12, None)
    q_eig_norm = q_eig_pos / np.sum(q_eig_pos)
    q_spectral_entropy = float(-np.sum(q_eig_norm * np.log2(q_eig_norm)))
    q_top1_eig_fraction = float(np.max(q_eigvals) / np.sum(q_eig_pos))

    # Off-diagonal elements
    triu_idx = np.triu_indices(N, k=1)
    q_offdiag = K_q_test[triu_idx]
    q_mean = float(np.mean(q_offdiag))
    q_std = float(np.std(q_offdiag))
    q_diversity = float(1.0 - q_mean)

    # 2. RBF Diagnostics
    rbf_sym_err = float(np.max(np.abs(K_rbf_test - K_rbf_test.T)))
    rbf_finite = bool(np.all(np.isfinite(K_rbf_test)))
    rbf_eigvals = np.linalg.eigvalsh(K_rbf_test)
    rbf_psd = bool(np.min(rbf_eigvals) >= -1e-6)
    rbf_offdiag = K_rbf_test[triu_idx]
    rbf_mean = float(np.mean(rbf_offdiag))
    rbf_std = float(np.std(rbf_offdiag))
    rbf_diversity = float(1.0 - rbf_mean)

    # 3. Label Alignment
    y_vec = np.where(y_test == 1, 1.0, -1.0).reshape(-1, 1)
    Y = np.dot(y_vec, y_vec.T)
    norm_Y = np.linalg.norm(Y, 'fro')
    
    norm_K_q = np.linalg.norm(K_q_test, 'fro')
    q_label_alignment = float(np.sum(K_q_test * Y) / (norm_K_q * norm_Y + 1e-12))

    norm_K_rbf = np.linalg.norm(K_rbf_test, 'fro')
    rbf_label_alignment = float(np.sum(K_rbf_test * Y) / (norm_K_rbf * norm_Y + 1e-12))

    # 4. Q-vs-RBF Geometry Correlation
    if np.std(q_offdiag) > 1e-8 and np.std(rbf_offdiag) > 1e-8:
        pearson_r, _ = stats.pearsonr(q_offdiag, rbf_offdiag)
        spearman_rho, _ = stats.spearmanr(q_offdiag, rbf_offdiag)
    else:
        pearson_r, spearman_rho = 0.0, 0.0

    return {
        "q_symmetry_error": q_sym_err,
        "q_psd_valid": q_psd,
        "q_finite_check": q_finite,
        "q_diag_min": q_diag_min,
        "q_diag_max": q_diag_max,
        "q_diag_mean": q_diag_mean,
        "q_kernel_mean": q_mean,
        "q_kernel_std": q_std,
        "q_kernel_diversity": q_diversity,
        "q_spectral_entropy": q_spectral_entropy,
        "q_top1_eig_fraction": q_top1_eig_fraction,
        "q_label_alignment": q_label_alignment,
        "rbf_symmetry_error": rbf_sym_err,
        "rbf_psd_valid": rbf_psd,
        "rbf_finite_check": rbf_finite,
        "rbf_kernel_mean": rbf_mean,
        "rbf_kernel_std": rbf_std,
        "rbf_kernel_diversity": rbf_diversity,
        "rbf_label_alignment": rbf_label_alignment,
        "gram_pearson_r": float(pearson_r),
        "gram_spearman_rho": float(spearman_rho),
    }


def optimize_threshold(decision_scores: np.ndarray, y_true: np.ndarray, n_steps: int = 200) -> float:
    """
    Selects optimal decision threshold maximizing F1 on VALIDATION SPLIT ONLY.
    """
    s_min, s_max = float(np.min(decision_scores)), float(np.max(decision_scores))
    if s_min == s_max:
        return 0.0
    grid = np.linspace(s_min, s_max, n_steps)
    best_thresh = 0.0
    best_f1 = -1.0
    for t in grid:
        preds = (decision_scores >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = float(t)
    return best_thresh


def evaluate_model_metrics(clf, X_eval, y_eval, opt_thresh: float, is_precomputed: bool = False) -> dict:
    """
    Computes all primary and secondary evaluation metrics out-of-sample.
    """
    scores = clf.decision_function(X_eval)
    preds = (scores >= opt_thresh).astype(int)

    f1 = float(f1_score(y_eval, preds, zero_division=0))
    acc = float(accuracy_score(y_eval, preds))
    bacc = float(balanced_accuracy_score(y_eval, preds))
    prec = float(precision_score(y_eval, preds, zero_division=0))
    rec = float(recall_score(y_eval, preds, zero_division=0))

    try:
        pr_auc = float(average_precision_score(y_eval, scores))
    except Exception:
        pr_auc = float("nan")

    try:
        roc_auc = float(roc_auc_score(y_eval, scores))
    except Exception:
        roc_auc = float("nan")

    return {
        "f1": f1,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "precision": prec,
        "recall": rec,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "decision_scores": scores,
        "predictions": preds,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "opt_thresh": opt_thresh
    }


def paired_permutation_test(deltas: np.ndarray, n_perm: int = 10000) -> float:
    """
    Calculates exact two-sided p-value under the null hypothesis that delta is symmetric around 0.
    """
    obs = np.abs(np.mean(deltas))
    n = len(deltas)
    signs = np.random.choice([-1, 1], size=(n_perm, n))
    perm_means = np.abs(np.mean(signs * deltas, axis=1))
    p_val = np.mean(perm_means >= obs)
    return float(p_val)


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 10000, alpha: float = 0.05) -> tuple:
    """
    Computes 95% Percentile Bootstrap Confidence Interval.
    """
    n = len(deltas)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = np.random.choice(deltas, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    low = float(np.percentile(boot_means, 100 * (alpha / 2)))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return low, high


def compute_mcnemar_test(preds_a: np.ndarray, preds_b: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Computes McNemar test statistic and p-value for paired binary predictions.
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    b = np.sum(correct_a & ~correct_b)  # A correct, B incorrect
    c = np.sum(~correct_a & correct_b)  # A incorrect, B correct

    stat = float((abs(b - c) - 1) ** 2 / (b + c + 1e-12))
    p_val = float(1.0 - stats.chi2.cdf(stat, df=1)) if (b + c) > 0 else 1.0

    return {"mcnemar_b": int(b), "mcnemar_c": int(c), "mcnemar_stat": stat, "mcnemar_p": p_val}


def benjamini_hochberg(p_values: list) -> list:
    """
    Applies Benjamini-Hochberg FDR correction.
    """
    n = len(p_values)
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    cum_min = 1.0

    for rank_idx in range(n - 1, -1, -1):
        orig_idx, p = sorted_pairs[rank_idx]
        rank = rank_idx + 1
        adj_p = min(1.0, p * n / rank)
        cum_min = min(cum_min, adj_p)
        adjusted[orig_idx] = float(cum_min)

    return adjusted


def main():
    print("=" * 80)
    print("EXPERIMENT 42: CONFIRMATORY REPRESENTATION-KERNEL ANALYSIS (10 SEEDS)")
    print(f"Seeds: {SEEDS} (N=10)")
    print(f"Candidates: {[c['name'] for c in CANDIDATES]}")
    print("=" * 80)

    start_total_time = time.time()
    seed_results = []
    geometry_results = []
    paired_pred_store = {}  # key: (candidate_id, seed)

    for cand in CANDIDATES:
        c_id = cand["candidate_id"]
        d_name = cand["dataset"]
        r_name = cand["representation"]
        c_name = cand["name"]

        print(f"\n>>> EXECUTING CANDIDATE [{c_id}]: {c_name} <<<")

        for seed in SEEDS:
            t0_seed = time.time()
            df_train, df_val, df_test = load_dataset_splits(d_name, seed)
            y_tr = df_train["target"].to_numpy().astype(int)
            y_va = df_val["target"].to_numpy().astype(int)
            y_te = df_test["target"].to_numpy().astype(int)

            # 1. Embeddings & Preprocessing
            X_tr_raw, X_va_raw, X_te_raw, model_info = get_or_create_embeddings(
                d_name, r_name, seed, df_train, df_val, df_test
            )

            # 2. Dimensional Reduction to Matched 8D
            Z_tr, Z_va, Z_te, A_tr, A_va, A_te = project_to_target_dimension(
                X_tr_raw, X_va_raw, X_te_raw, target_dim=DIMENSION, seed=seed
            )

            # 3. Model A: Linear SVM
            t0_lin = time.time()
            lin_svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
            lin_svm.fit(Z_tr, y_tr)
            val_scores_lin = lin_svm.decision_function(Z_va)
            opt_thresh_lin = optimize_threshold(val_scores_lin, y_va)
            metrics_lin = evaluate_model_metrics(lin_svm, Z_te, y_te, opt_thresh_lin)
            lin_time = time.time() - t0_lin

            # 4. Model B: Classical RBF SVM
            t0_rbf = time.time()
            rbf_svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=seed)
            rbf_svm.fit(Z_tr, y_tr)
            val_scores_rbf = rbf_svm.decision_function(Z_va)
            opt_thresh_rbf = optimize_threshold(val_scores_rbf, y_va)
            metrics_rbf = evaluate_model_metrics(rbf_svm, Z_te, y_te, opt_thresh_rbf)
            rbf_time = time.time() - t0_rbf

            # 5. Model C: Quantum Fidelity Kernel SVM
            t0_q = time.time()
            # Construct Gram matrices
            K_q_tr_tr = compute_quantum_gram_matrix(A_tr, A_tr, n_qubits=DIMENSION)
            K_q_va_tr = compute_quantum_gram_matrix(A_va, A_tr, n_qubits=DIMENSION)
            K_q_te_tr = compute_quantum_gram_matrix(A_te, A_tr, n_qubits=DIMENSION)
            # Square test-test Gram matrix for geometry diagnostics
            K_q_te_te = compute_quantum_gram_matrix(A_te, A_te, n_qubits=DIMENSION)

            q_svm = SVC(kernel="precomputed", C=1.0, class_weight="balanced", random_state=seed)
            q_svm.fit(K_q_tr_tr, y_tr)
            val_scores_q = q_svm.decision_function(K_q_va_tr)
            opt_thresh_q = optimize_threshold(val_scores_q, y_va)
            metrics_q = evaluate_model_metrics(q_svm, K_q_te_tr, y_te, opt_thresh_q, is_precomputed=True)
            q_time = time.time() - t0_q

            # 6. Geometry Diagnostics
            # Compute RBF test-test Gram matrix
            gamma_val = 1.0 / (DIMENSION * np.var(Z_tr) + 1e-12)
            dists_sq = np.sum((Z_te[:, np.newaxis, :] - Z_te[np.newaxis, :, :]) ** 2, axis=-1)
            K_rbf_te_te = np.exp(-gamma_val * dists_sq)

            geom = compute_comprehensive_geometry(K_q_te_te, K_rbf_te_te, y_te)
            geom.update({
                "candidate_id": c_id,
                "dataset": d_name,
                "representation": r_name,
                "seed": seed,
                "dimension": DIMENSION
            })
            geometry_results.append(geom)

            # Store predictions for paired tests
            paired_pred_store[(c_id, seed)] = {
                "q_preds": metrics_q["predictions"],
                "rbf_preds": metrics_rbf["predictions"],
                "lin_preds": metrics_lin["predictions"],
                "y_true": y_te
            }

            delta_q_rbf = metrics_q["f1"] - metrics_rbf["f1"]
            delta_q_lin = metrics_q["f1"] - metrics_lin["f1"]
            delta_rbf_lin = metrics_rbf["f1"] - metrics_lin["f1"]

            seed_results.append({
                "candidate_id": c_id,
                "candidate_name": c_name,
                "dataset": d_name,
                "representation": r_name,
                "seed": seed,
                "dimension": DIMENSION,
                "quantum_f1": metrics_q["f1"],
                "rbf_f1": metrics_rbf["f1"],
                "linear_f1": metrics_lin["f1"],
                "delta_f1_q_rbf": delta_q_rbf,
                "delta_f1_q_lin": delta_q_lin,
                "delta_f1_rbf_lin": delta_rbf_lin,
                "quantum_pr_auc": metrics_q["pr_auc"],
                "rbf_pr_auc": metrics_rbf["pr_auc"],
                "linear_pr_auc": metrics_lin["pr_auc"],
                "quantum_roc_auc": metrics_q["roc_auc"],
                "rbf_roc_auc": metrics_rbf["roc_auc"],
                "linear_roc_auc": metrics_lin["roc_auc"],
                "quantum_bacc": metrics_q["balanced_accuracy"],
                "rbf_bacc": metrics_rbf["balanced_accuracy"],
                "linear_bacc": metrics_lin["balanced_accuracy"],
                "quantum_opt_thresh": metrics_q["opt_thresh"],
                "rbf_opt_thresh": metrics_rbf["opt_thresh"],
                "linear_opt_thresh": metrics_lin["opt_thresh"],
                "q_kernel_diversity": geom["q_kernel_diversity"],
                "rbf_kernel_diversity": geom["rbf_kernel_diversity"],
                "q_label_alignment": geom["q_label_alignment"],
                "rbf_label_alignment": geom["rbf_label_alignment"],
                "gram_pearson_r": geom["gram_pearson_r"],
                "runtime_q_s": q_time,
                "runtime_rbf_s": rbf_time,
                "runtime_lin_s": lin_time,
                "total_seed_time_s": time.time() - t0_seed
            })

            print(f"  Seed {seed:4d} | Q_F1={metrics_q['f1']:.4f} | RBF_F1={metrics_rbf['f1']:.4f} | Lin_F1={metrics_lin['f1']:.4f} | Δ(Q-RBF)={delta_q_rbf:+.4f} | Q_Div={geom['q_kernel_diversity']:.4f}")

    # ============================================================
    # STATISTICAL ANALYSIS ACROSS PROMOTED CANDIDATES
    # ============================================================
    df_seeds = pd.DataFrame(seed_results)
    df_geom = pd.DataFrame(geometry_results)

    stats_list = []
    promoted_candidates = ["A", "B", "C"]
    raw_p_values = []

    for c_id in ["A", "B", "C", "REF_SMS_TFIDF"]:
        sub = df_seeds[df_seeds["candidate_id"] == c_id]
        deltas = sub["delta_f1_q_rbf"].to_numpy()
        q_f1s = sub["quantum_f1"].to_numpy()
        rbf_f1s = sub["rbf_f1"].to_numpy()
        lin_f1s = sub["linear_f1"].to_numpy()

        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas, ddof=1))
        median_delta = float(np.median(deltas))
        min_delta = float(np.min(deltas))
        max_delta = float(np.max(deltas))

        # Student-t CI (95%)
        t_crit = stats.t.ppf(0.975, df=len(deltas)-1)
        se = std_delta / np.sqrt(len(deltas))
        t_ci_low = float(mean_delta - t_crit * se)
        t_ci_high = float(mean_delta + t_crit * se)

        # Paired permutation test (10k)
        perm_p = paired_permutation_test(deltas, n_perm=10000)
        
        # Bootstrap CI (10k)
        boot_low, boot_high = bootstrap_ci(deltas, n_boot=10000)

        # Paired t-test
        t_stat, t_p = stats.ttest_rel(q_f1s, rbf_f1s)

        # Seed counts
        q_wins = int(np.sum(deltas > EPSILON))
        rbf_wins = int(np.sum(deltas < -EPSILON))
        ties = int(np.sum(np.abs(deltas) <= EPSILON))

        # Practical equivalence classification
        is_stat_sig = (perm_p < 0.05) and not (t_ci_low <= 0 <= t_ci_high)
        if is_stat_sig and mean_delta >= EPSILON:
            classification = "A. Confirmed practical Q advantage"
        elif is_stat_sig and mean_delta <= -EPSILON:
            classification = "D. Classical advantage (|Δ| >= 0.01)"
        elif is_stat_sig and abs(mean_delta) < EPSILON:
            classification = "B. Statistically detectable but practically equivalent"
        else:
            classification = "C. No statistically detectable difference (spans zero)"

        stat_record = {
            "candidate_id": c_id,
            "candidate_name": sub["candidate_name"].iloc[0],
            "dataset": sub["dataset"].iloc[0],
            "representation": sub["representation"].iloc[0],
            "n_seeds": len(deltas),
            "q_mean_f1": float(np.mean(q_f1s)),
            "q_std_f1": float(np.std(q_f1s, ddof=1)),
            "rbf_mean_f1": float(np.mean(rbf_f1s)),
            "rbf_std_f1": float(np.std(rbf_f1s, ddof=1)),
            "linear_mean_f1": float(np.mean(lin_f1s)),
            "linear_std_f1": float(np.std(lin_f1s, ddof=1)),
            "delta_mean": mean_delta,
            "delta_std": std_delta,
            "delta_median": median_delta,
            "delta_min": min_delta,
            "delta_max": max_delta,
            "student_t_ci_95_low": t_ci_low,
            "student_t_ci_95_high": t_ci_high,
            "bootstrap_ci_95_low": boot_low,
            "bootstrap_ci_95_high": boot_high,
            "permutation_p_raw": perm_p,
            "paired_t_p": float(t_p),
            "seed_q_wins": q_wins,
            "seed_rbf_wins": rbf_wins,
            "seed_ties": ties,
            "epsilon_classification": classification
        }
        stats_list.append(stat_record)
        if c_id in promoted_candidates:
            raw_p_values.append(perm_p)

    # Benjamini-Hochberg FDR correction across the 3 promoted candidates
    bh_p_values = benjamini_hochberg(raw_p_values)
    for i, c_id in enumerate(promoted_candidates):
        for rec in stats_list:
            if rec["candidate_id"] == c_id:
                rec["permutation_p_bh_fdr"] = bh_p_values[i]
                break

    # Non-promoted baseline ref gets raw p as placeholder
    for rec in stats_list:
        if "permutation_p_bh_fdr" not in rec:
            rec["permutation_p_bh_fdr"] = rec["permutation_p_raw"]

    df_stats = pd.DataFrame(stats_list)

    # ============================================================
    # REPRESENTATION EFFECT ANALYSIS (SMS: MiniLM vs MPNet vs TF-IDF)
    # ============================================================
    sms_tfidf = df_stats[df_stats["candidate_id"] == "REF_SMS_TFIDF"].iloc[0]
    sms_minilm = df_stats[df_stats["candidate_id"] == "A"].iloc[0]
    sms_mpnet = df_stats[df_stats["candidate_id"] == "B"].iloc[0]

    rep_effects = [
        {
            "comparison": "MiniLM vs TF-IDF on SMS",
            "q_f1_shift": sms_minilm["q_mean_f1"] - sms_tfidf["q_mean_f1"],
            "rbf_f1_shift": sms_minilm["rbf_mean_f1"] - sms_tfidf["rbf_mean_f1"],
            "linear_f1_shift": sms_minilm["linear_mean_f1"] - sms_tfidf["linear_mean_f1"],
            "delta_q_rbf_shift": sms_minilm["delta_mean"] - sms_tfidf["delta_mean"],
            "interpretation": "Evaluates whether dense MiniLM shifts relative kernel preference vs lexical TF-IDF"
        },
        {
            "comparison": "MPNet vs TF-IDF on SMS",
            "q_f1_shift": sms_mpnet["q_mean_f1"] - sms_tfidf["q_mean_f1"],
            "rbf_f1_shift": sms_mpnet["rbf_mean_f1"] - sms_tfidf["rbf_mean_f1"],
            "linear_f1_shift": sms_mpnet["linear_mean_f1"] - sms_tfidf["linear_mean_f1"],
            "delta_q_rbf_shift": sms_mpnet["delta_mean"] - sms_tfidf["delta_mean"],
            "interpretation": "Evaluates extent of quantum geometric degradation on MPNet contrastive embeddings"
        },
        {
            "comparison": "MPNet vs MiniLM on SMS",
            "q_f1_shift": sms_mpnet["q_mean_f1"] - sms_minilm["q_mean_f1"],
            "rbf_f1_shift": sms_mpnet["rbf_mean_f1"] - sms_minilm["rbf_mean_f1"],
            "linear_f1_shift": sms_mpnet["linear_mean_f1"] - sms_minilm["linear_mean_f1"],
            "delta_q_rbf_shift": sms_mpnet["delta_mean"] - sms_minilm["delta_mean"],
            "interpretation": "Contrasts two dense sentence transformer architectures (compact 384D vs 768D masked/permuted)"
        }
    ]
    df_rep_effects = pd.DataFrame(rep_effects)

    # ============================================================
    # MPNet SPECIFIC FAILURE-MODE DIAGNOSTICS
    # ============================================================
    mpnet_geom = df_geom[df_geom["candidate_id"] == "B"]
    minilm_geom = df_geom[df_geom["candidate_id"] == "A"]
    tfidf_sms_geom = df_geom[df_geom["candidate_id"] == "REF_SMS_TFIDF"]

    mpnet_diag = [
        {
            "metric": "Quantum Kernel Diversity (Mean ± Std)",
            "sms_mpnet": f"{mpnet_geom['q_kernel_diversity'].mean():.4f} ± {mpnet_geom['q_kernel_diversity'].std():.4f}",
            "sms_minilm": f"{minilm_geom['q_kernel_diversity'].mean():.4f} ± {minilm_geom['q_kernel_diversity'].std():.4f}",
            "sms_tfidf": f"{tfidf_sms_geom['q_kernel_diversity'].mean():.4f} ± {tfidf_sms_geom['q_kernel_diversity'].std():.4f}",
            "notes": "Low diversity reflects extreme concentration of off-diagonal quantum fidelity values"
        },
        {
            "metric": "RBF Kernel Diversity (Mean ± Std)",
            "sms_mpnet": f"{mpnet_geom['rbf_kernel_diversity'].mean():.4f} ± {mpnet_geom['rbf_kernel_diversity'].std():.4f}",
            "sms_minilm": f"{minilm_geom['rbf_kernel_diversity'].mean():.4f} ± {minilm_geom['rbf_kernel_diversity'].std():.4f}",
            "sms_tfidf": f"{tfidf_sms_geom['rbf_kernel_diversity'].mean():.4f} ± {tfidf_sms_geom['rbf_kernel_diversity'].std():.4f}",
            "notes": "Classical RBF adapts bandwidth gamma=scale to maintain healthy feature dispersion"
        },
        {
            "metric": "Quantum Target Label Alignment",
            "sms_mpnet": f"{mpnet_geom['q_label_alignment'].mean():.4f} ± {mpnet_geom['q_label_alignment'].std():.4f}",
            "sms_minilm": f"{minilm_geom['q_label_alignment'].mean():.4f} ± {minilm_geom['q_label_alignment'].std():.4f}",
            "sms_tfidf": f"{tfidf_sms_geom['q_label_alignment'].mean():.4f} ± {tfidf_sms_geom['q_label_alignment'].std():.4f}",
            "notes": "Alignment of quantum Gram matrix with outer product of test class labels"
        },
        {
            "metric": "Q vs RBF Gram Pearson Correlation (r)",
            "sms_mpnet": f"{mpnet_geom['gram_pearson_r'].mean():.4f} ± {mpnet_geom['gram_pearson_r'].std():.4f}",
            "sms_minilm": f"{minilm_geom['gram_pearson_r'].mean():.4f} ± {minilm_geom['gram_pearson_r'].std():.4f}",
            "sms_tfidf": f"{tfidf_sms_geom['gram_pearson_r'].mean():.4f} ± {tfidf_sms_geom['gram_pearson_r'].std():.4f}",
            "notes": "Correlation between quantum fidelity and classical RBF pairwise distances"
        },
        {
            "metric": "Spectral Entropy of Quantum Kernel (Bits)",
            "sms_mpnet": f"{mpnet_geom['q_spectral_entropy'].mean():.4f} ± {mpnet_geom['q_spectral_entropy'].std():.4f}",
            "sms_minilm": f"{minilm_geom['q_spectral_entropy'].mean():.4f} ± {minilm_geom['q_spectral_entropy'].std():.4f}",
            "sms_tfidf": f"{tfidf_sms_geom['q_spectral_entropy'].mean():.4f} ± {tfidf_sms_geom['q_spectral_entropy'].std():.4f}",
            "notes": "Eigenvalue distribution spread in quantum Hilbert space"
        }
    ]
    df_mpnet_diag = pd.DataFrame(mpnet_diag)

    # Save CSV files
    df_seeds.to_csv(os.path.join(EXP42_DIR, "exp42_seed_results.csv"), index=False)
    df_geom.to_csv(os.path.join(EXP42_DIR, "exp42_geometry.csv"), index=False)
    df_stats.to_csv(os.path.join(EXP42_DIR, "exp42_statistics.csv"), index=False)
    df_rep_effects.to_csv(os.path.join(EXP42_DIR, "exp42_representation_effects.csv"), index=False)
    df_mpnet_diag.to_csv(os.path.join(EXP42_DIR, "exp42_mpnet_diagnostics.csv"), index=False)
    
    # Save Summary CSV
    summary_rows = []
    for _, row in df_stats.iterrows():
        summary_rows.append({
            "candidate_id": row["candidate_id"],
            "name": row["candidate_name"],
            "dataset": row["dataset"],
            "representation": row["representation"],
            "quantum_f1": f"{row['q_mean_f1']:.4f} ± {row['q_std_f1']:.4f}",
            "rbf_f1": f"{row['rbf_mean_f1']:.4f} ± {row['rbf_std_f1']:.4f}",
            "linear_f1": f"{row['linear_mean_f1']:.4f} ± {row['linear_std_f1']:.4f}",
            "delta_q_rbf": f"{row['delta_mean']:+.4f} [{row['student_t_ci_95_low']:+.4f}, {row['student_t_ci_95_high']:+.4f}]",
            "perm_p_raw": f"{row['permutation_p_raw']:.4f}",
            "perm_p_bh_fdr": f"{row['permutation_p_bh_fdr']:.4f}",
            "seed_wins (Q/RBF/Tie)": f"{row['seed_q_wins']}/{row['seed_rbf_wins']}/{row['seed_ties']}",
            "verdict": row["epsilon_classification"]
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(EXP42_DIR, "exp42_summary.csv"), index=False)

    # ============================================================
    # GENERATE 6 PUBLICATION FIGURES
    # ============================================================
    print("\nGenerating 6 Publication Figures in results/exp42/figures/ ...")

    # Figure 1: Seed-level Q-RBF F1 differences for all 3 candidates
    plt.figure(figsize=(10, 6))
    fig1_df = df_seeds[df_seeds["candidate_id"].isin(["A", "B", "C"])].copy()
    candidate_labels = {"A": "Candidate A: SMS + MiniLM", "B": "Candidate B: SMS + MPNet", "C": "Candidate C: CEAS + TF-IDF"}
    
    for c_id, grp in fig1_df.groupby("candidate_id"):
        plt.plot(grp["seed"].astype(str), grp["delta_f1_q_rbf"], marker='o', label=candidate_labels[c_id], linewidth=2)
    
    plt.axhline(0.0, color='black', linestyle='-', linewidth=1)
    plt.axhline(EPSILON, color='gray', linestyle='--', linewidth=1, label=r'$\epsilon = +0.01$ Equivalence')
    plt.axhline(-EPSILON, color='gray', linestyle='--', linewidth=1, label=r'$\epsilon = -0.01$ Equivalence')
    plt.fill_between(range(10), -EPSILON, EPSILON, color='gray', alpha=0.15)
    plt.title("Figure 1: Seed-Level Paired Q-vs-RBF F1 Differences (Exp 42, N=10 Seeds)", fontsize=12, fontweight='bold')
    plt.xlabel("Random Seed (10 Canonical Seeds)", fontsize=11)
    plt.ylabel(r"$\Delta\mathrm{F1} = \mathrm{Quantum\ F1} - \mathrm{RBF\ F1}$", fontsize=11)
    plt.legend(loc='lower left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "fig1_seed_deltas.png"), dpi=300)
    plt.close()

    # Figure 2: Representation x Classifier F1 Comparison
    plt.figure(figsize=(10, 6))
    cand_order = ["A", "B", "C", "REF_SMS_TFIDF"]
    names = [df_stats[df_stats["candidate_id"] == c]["candidate_name"].iloc[0] for c in cand_order]
    q_means = [df_stats[df_stats["candidate_id"] == c]["q_mean_f1"].iloc[0] for c in cand_order]
    rbf_means = [df_stats[df_stats["candidate_id"] == c]["rbf_mean_f1"].iloc[0] for c in cand_order]
    lin_means = [df_stats[df_stats["candidate_id"] == c]["linear_mean_f1"].iloc[0] for c in cand_order]

    x = np.arange(len(names))
    width = 0.25
    plt.bar(x - width, lin_means, width, label='Linear SVM baseline', color='#475569')
    plt.bar(x, rbf_means, width, label='Classical RBF SVM', color='#2563eb')
    plt.bar(x + width, q_means, width, label='Quantum Fidelity Kernel SVM', color='#7c3aed')

    plt.xticks(x, names, rotation=12, ha='right', fontsize=10)
    plt.ylabel("Test F1 Score (Mean over 10 Seeds)", fontsize=11)
    plt.title("Figure 2: Representation × Classifier F1 Comparison (Exp 42)", fontsize=12, fontweight='bold')
    plt.ylim(0.5, 1.0)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "fig2_rep_classifier_f1.png"), dpi=300)
    plt.close()

    # Figure 3: Quantum vs RBF Kernel Diversity
    plt.figure(figsize=(8, 6))
    sns_df = df_seeds[df_seeds["candidate_id"].isin(["A", "B", "C", "REF_SMS_TFIDF"])].copy()
    plt.scatter(sns_df["rbf_kernel_diversity"], sns_df["q_kernel_diversity"], c=sns_df["delta_f1_q_rbf"], cmap="coolwarm", s=80, edgecolors='k')
    cbar = plt.colorbar()
    cbar.set_label(r"$\Delta\mathrm{F1}\ (\mathrm{Q} - \mathrm{RBF})$", fontsize=10)
    plt.plot([0, 0.4], [0, 0.4], 'k--', alpha=0.5, label='Equality Line')
    plt.xlabel("Classical RBF Kernel Diversity", fontsize=11)
    plt.ylabel("Quantum Kernel Diversity", fontsize=11)
    plt.title("Figure 3: Quantum vs RBF Kernel Diversity across Seeds", fontsize=12, fontweight='bold')
    plt.legend(loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "fig3_kernel_diversity.png"), dpi=300)
    plt.close()

    # Figure 4: Quantum Label Alignment vs Q-RBF F1 Difference
    plt.figure(figsize=(8, 6))
    plt.scatter(df_seeds["q_label_alignment"], df_seeds["delta_f1_q_rbf"], color='#7c3aed', s=70, alpha=0.8, edgecolors='black')
    z = np.polyfit(df_seeds["q_label_alignment"], df_seeds["delta_f1_q_rbf"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df_seeds["q_label_alignment"].min(), df_seeds["q_label_alignment"].max(), 50)
    plt.plot(x_range, p(x_range), 'r--', linewidth=1.5, label=f"Trendline (slope={z[0]:+.2f})")
    plt.axhline(0.0, color='black', linestyle='-', linewidth=0.8)
    plt.xlabel("Quantum Target Label Alignment", fontsize=11)
    plt.ylabel(r"$\Delta\mathrm{F1}\ (\mathrm{Quantum} - \mathrm{RBF})$", fontsize=11)
    plt.title("Figure 4: Quantum Label Alignment vs Q-RBF Performance Difference", fontsize=12, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "fig4_alignment_vs_delta.png"), dpi=300)
    plt.close()

    # Figure 5: MPNet Geometry Diagnostics (Compression Comparison)
    plt.figure(figsize=(9, 5))
    metrics_names = ["Kernel Diversity", "Label Alignment", "Gram Pearson r", "Spectral Entropy / 10"]
    sms_mp_vals = [
        mpnet_geom["q_kernel_diversity"].mean(),
        mpnet_geom["q_label_alignment"].mean(),
        mpnet_geom["gram_pearson_r"].mean(),
        mpnet_geom["q_spectral_entropy"].mean() / 10.0
    ]
    sms_tf_vals = [
        tfidf_sms_geom["q_kernel_diversity"].mean(),
        tfidf_sms_geom["q_label_alignment"].mean(),
        tfidf_sms_geom["gram_pearson_r"].mean(),
        tfidf_sms_geom["q_spectral_entropy"].mean() / 10.0
    ]
    x_idx = np.arange(len(metrics_names))
    plt.bar(x_idx - 0.15, sms_tf_vals, 0.3, label="SMS + TF-IDF (Sparse Lexical)", color='#0284c7')
    plt.bar(x_idx + 0.15, sms_mp_vals, 0.3, label="SMS + MPNet (Dense Contrastive)", color='#e11d48')
    plt.xticks(x_idx, metrics_names, fontsize=10)
    plt.ylabel("Normalized Metric Value", fontsize=11)
    plt.title("Figure 5: MPNet Geometric Collapse vs Canonical TF-IDF Geometry", fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "fig5_mpnet_geometry.png"), dpi=300)
    plt.close()

    # Figure 6: MiniLM Rank-Reversal Seed Distribution
    plt.figure(figsize=(8, 5))
    minilm_deltas = df_seeds[df_seeds["candidate_id"] == "A"]["delta_f1_q_rbf"].to_numpy()
    plt.hist(minilm_deltas, bins=7, color='#8b5cf6', edgecolor='black', alpha=0.8)
    plt.axvline(0.0, color='black', linestyle='-', linewidth=1.5, label="Zero Parity Line")
    plt.axvline(EPSILON, color='gray', linestyle='--', linewidth=1, label=r"$+\epsilon = +0.01$")
    plt.axvline(-EPSILON, color='gray', linestyle='--', linewidth=1, label=r"$-\epsilon = -0.01$")
    plt.axvline(np.mean(minilm_deltas), color='red', linestyle='-', linewidth=2, label=f"Mean Δ = {np.mean(minilm_deltas):+.4f}")
    plt.xlabel(r"$\Delta\mathrm{F1}\ (\mathrm{Quantum} - \mathrm{RBF})$ on SMS + MiniLM", fontsize=11)
    plt.ylabel("Frequency (Count of Seeds out of 10)", fontsize=11)
    plt.title("Figure 6: MiniLM Rank-Reversal Seed Distribution (10 Seeds)", fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "fig6_minilm_seed_dist.png"), dpi=300)
    plt.close()

    total_time = time.time() - start_total_time
    print("\n" + "=" * 80)
    print("EXPERIMENT 42 CONFIRMATION COMPLETE!")
    print(f"Total Execution Time: {total_time:.2f}s")
    print(f"Results saved to: {EXP42_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
