#!/usr/bin/env python3
"""
Experiment 43: Representation-Aware Quantum Encoding Diagnostic Study
======================================================================
Scientific Objective:
  Investigate WHY the SMS + MPNet condition produces the severe quantum-kernel
  failure observed in Exp 42 (Q F1 = 0.3756 vs RBF F1 = 0.9045, ΔF1 = -0.5288).

Primary Research Question:
  "Can representation-aware quantum encoding choices recover the quantum-kernel
   alignment lost when dense MPNet sentence embeddings are projected into an
   8-dimensional quantum feature space?"

Secondary Research Question:
  "Do changes in quantum-kernel geometry correspond to changes in classification performance?"

Ablation Dimensions:
  1. Angular Mapping Diagnostic (Depth = 2):
     - A1: Canonical [0, π]
     - A2: [-π, π]
     - A3: [0, 2π]
     - A4: Train-only normal CDF angular mapping: π * 0.5 * (1 + erf(Z / sqrt(2)))
  2. Feature-Map Depth Diagnostic (Angle = [0, π]):
     - B1: 1-layer cyclic ZZFeatureMap
     - B2: 2-layer cyclic ZZFeatureMap (Canonical replication)
     - B3: 3-layer cyclic ZZFeatureMap

Total Conditions (7):
  1. canonical_2layer_0_pi
  2. angle_minus_pi_pi
  3. angle_0_2pi
  4. angle_train_normalized
  5. depth_1
  6. depth_2
  7. depth_3

Dataset: SMS Spam Collection (Only)
Representation: MPNet (all-mpnet-base-v2, 768D -> 8D TruncatedSVD + StandardScaler)
Seeds (10): [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
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
import scipy.special as special
from scipy.linalg import eigvalsh
import sklearn
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app.representations.registry import representation_registry

EXP43_DIR = os.path.join(BASE_DIR, "results/exp43")
CACHE_DIR = os.path.join(EXP43_DIR, "cache")
FIGS_DIR = os.path.join(EXP43_DIR, "figures")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# 10 Canonical Seeds
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
DATASET = "sms"
REPRESENTATION = "mpnet"
DIMENSION = 8
EPSILON = 0.01  # Practical equivalence boundary

device = torch.device("cpu")
dtype = torch.complex128

# 7 Predefined Experimental Conditions
CONDITIONS = [
    {
        "cond_id": "canonical_2layer_0_pi",
        "category": "Angle Mapping / Canonical Baseline",
        "family": "angle",
        "angle_mapping": "0_pi",
        "depth": 2,
        "label": "A1/B2: Canonical [0, π] (2-Layer)"
    },
    {
        "cond_id": "angle_minus_pi_pi",
        "category": "Angle Mapping",
        "family": "angle",
        "angle_mapping": "minus_pi_pi",
        "depth": 2,
        "label": "A2: Symmetric [-π, π] (2-Layer)"
    },
    {
        "cond_id": "angle_0_2pi",
        "category": "Angle Mapping",
        "family": "angle",
        "angle_mapping": "0_2pi",
        "depth": 2,
        "label": "A3: Full Circle [0, 2π] (2-Layer)"
    },
    {
        "cond_id": "angle_train_normalized",
        "category": "Angle Mapping",
        "family": "angle",
        "angle_mapping": "train_normalized",
        "depth": 2,
        "label": "A4: Train Normal CDF [0, π] (2-Layer)"
    },
    {
        "cond_id": "depth_1",
        "category": "Feature-Map Depth",
        "family": "depth",
        "angle_mapping": "0_pi",
        "depth": 1,
        "label": "B1: 1-Layer Cyclic ZZFeatureMap"
    },
    {
        "cond_id": "depth_2",
        "category": "Feature-Map Depth",
        "family": "depth",
        "angle_mapping": "0_pi",
        "depth": 2,
        "label": "B2: 2-Layer Cyclic ZZFeatureMap"
    },
    {
        "cond_id": "depth_3",
        "category": "Feature-Map Depth",
        "family": "depth",
        "angle_mapping": "0_pi",
        "depth": 3,
        "label": "B3: 3-Layer Cyclic ZZFeatureMap"
    }
]


def load_dataset_splits(dataset_name: str, seed: int, sample_size: int = 600):
    """
    Loads frozen train, validation, and test splits with stratified subsampling per seed.
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


def get_or_load_mpnet_embeddings(dataset_name: str, seed: int, df_train, df_val, df_test):
    """
    Reuses existing cached MPNet embeddings from Exp42 or generates them deterministically.
    """
    cache_subdir = os.path.join(CACHE_DIR, dataset_name, "mpnet")
    os.makedirs(cache_subdir, exist_ok=True)
    exp42_cache = os.path.join(BASE_DIR, "results/exp42/cache", dataset_name, "mpnet")
    exp41_cache = os.path.join(BASE_DIR, "results/exp41/cache", dataset_name, "mpnet")

    train_path = os.path.join(cache_subdir, f"train_seed_{seed}.npy")
    val_path = os.path.join(cache_subdir, f"val_seed_{seed}.npy")
    test_path = os.path.join(cache_subdir, f"test_seed_{seed}.npy")

    exp42_tr = os.path.join(exp42_cache, f"train_seed_{seed}.npy")
    exp42_va = os.path.join(exp42_cache, f"val_seed_{seed}.npy")
    exp42_te = os.path.join(exp42_cache, f"test_seed_{seed}.npy")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        return np.load(train_path), np.load(val_path), np.load(test_path)
    elif os.path.exists(exp42_tr) and os.path.exists(exp42_va) and os.path.exists(exp42_te):
        X_tr, X_va, X_te = np.load(exp42_tr), np.load(exp42_va), np.load(exp42_te)
        np.save(train_path, X_tr)
        np.save(val_path, X_va)
        np.save(test_path, X_te)
        return X_tr, X_va, X_te

    rep_obj = representation_registry.get("mpnet")
    X_tr = rep_obj.encode_original(df_train["text"].tolist())
    X_va = rep_obj.encode_original(df_val["text"].tolist())
    X_te = rep_obj.encode_original(df_test["text"].tolist())

    np.save(train_path, X_tr)
    np.save(val_path, X_va)
    np.save(test_path, X_te)
    return X_tr, X_va, X_te


def project_svd_scaler(X_tr, X_va, X_te, target_dim: int, seed: int):
    """
    Fits TruncatedSVD and StandardScaler strictly on training partition.
    Returns standardized 8D representations Z_tr, Z_va, Z_te.
    """
    svd = TruncatedSVD(n_components=target_dim, random_state=seed)
    scaler = StandardScaler()

    Z_tr = scaler.fit_transform(svd.fit_transform(X_tr))
    Z_va = scaler.transform(svd.transform(X_va))
    Z_te = scaler.transform(svd.transform(X_te))

    return Z_tr, Z_va, Z_te


def apply_angular_transformation(Z_tr: np.ndarray, Z_va: np.ndarray, Z_te: np.ndarray, mapping_type: str):
    """
    Applies deterministic angular mapping based strictly on training set bounds/statistics.
    Preserves exact 8D dimensionality.
    """
    mins = Z_tr.min(axis=0)
    maxs = Z_tr.max(axis=0)
    ranges = np.where(maxs - mins > 1e-8, maxs - mins, 1.0)

    if mapping_type == "0_pi":
        # Canonical [0, pi]
        A_tr = np.clip((Z_tr - mins) / ranges * np.pi, 0.0, np.pi)
        A_va = np.clip((Z_va - mins) / ranges * np.pi, 0.0, np.pi)
        A_te = np.clip((Z_te - mins) / ranges * np.pi, 0.0, np.pi)
    elif mapping_type == "minus_pi_pi":
        # Symmetric [-pi, pi]
        A_tr = np.clip((Z_tr - mins) / ranges * (2.0 * np.pi) - np.pi, -np.pi, np.pi)
        A_va = np.clip((Z_va - mins) / ranges * (2.0 * np.pi) - np.pi, -np.pi, np.pi)
        A_te = np.clip((Z_te - mins) / ranges * (2.0 * np.pi) - np.pi, -np.pi, np.pi)
    elif mapping_type == "0_2pi":
        # Full circle [0, 2pi]
        A_tr = np.clip((Z_tr - mins) / ranges * (2.0 * np.pi), 0.0, 2.0 * np.pi)
        A_va = np.clip((Z_va - mins) / ranges * (2.0 * np.pi), 0.0, 2.0 * np.pi)
        A_te = np.clip((Z_te - mins) / ranges * (2.0 * np.pi), 0.0, 2.0 * np.pi)
    elif mapping_type == "train_normalized":
        # Gaussian standard normal CDF: pi * 0.5 * (1 + erf(Z / sqrt(2)))
        # Strictly uses standard normal integral on zero-mean unit-variance train representation
        A_tr = np.pi * 0.5 * (1.0 + special.erf(Z_tr / np.sqrt(2.0)))
        A_va = np.pi * 0.5 * (1.0 + special.erf(Z_va / np.sqrt(2.0)))
        A_te = np.pi * 0.5 * (1.0 + special.erf(Z_te / np.sqrt(2.0)))
        A_tr = np.clip(A_tr, 0.0, np.pi)
        A_va = np.clip(A_va, 0.0, np.pi)
        A_te = np.clip(A_te, 0.0, np.pi)
    else:
        raise ValueError(f"Unknown angle mapping type: {mapping_type}")

    return A_tr, A_va, A_te


def simulate_zz_feature_map_depth(X: torch.Tensor, n_qubits: int = 8, reps: int = 2) -> torch.Tensor:
    """
    Simulates cyclic ZZFeatureMap for arbitrary depth 'reps' in exact PyTorch complex128.
    X: [Batch, n_qubits]
    Returns: Statevector tensor [Batch, 2^n_qubits]
    """
    B = X.shape[0]
    state = torch.zeros([B] + [2] * n_qubits, dtype=dtype, device=device)
    state[(slice(None),) + (0,) * n_qubits] = 1.0 + 0.0j

    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    H_mat = torch.tensor([[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]], dtype=dtype, device=device)

    def apply_hadamard_all(s: torch.Tensor, n: int) -> torch.Tensor:
        for q in range(n):
            s = torch.tensordot(s, H_mat, dims=([q + 1], [1]))
            perm = list(range(n + 1))
            perm.insert(q + 1, perm.pop(-1))
            s = s.permute(perm)
        return s

    def apply_rz_all(s: torch.Tensor, x: torch.Tensor, n: int) -> torch.Tensor:
        for q in range(n):
            theta = 2.0 * x[:, q]
            phase_0 = torch.exp(-1j * (theta / 2.0)).view(B, *([1] * n))
            phase_1 = torch.exp(1j * (theta / 2.0)).view(B, *([1] * n))
            idx0 = [slice(None)] + [slice(None)] * n
            idx1 = [slice(None)] + [slice(None)] * n
            idx0[q + 1] = 0
            idx1[q + 1] = 1
            s[tuple(idx0)] = s[tuple(idx0)] * phase_0.squeeze(q + 1)
            s[tuple(idx1)] = s[tuple(idx1)] * phase_1.squeeze(q + 1)
        return s

    def apply_rzz_cyclic(s: torch.Tensor, x: torch.Tensor, n: int) -> torch.Tensor:
        for q in range(n):
            q_next = (q + 1) % n
            phi = 2.0 * (np.pi - x[:, q]) * (np.pi - x[:, q_next])
            phase_same = torch.exp(-1j * (phi / 2.0)).view(B, *([1] * n))
            phase_diff = torch.exp(1j * (phi / 2.0)).view(B, *([1] * n))

            mesh_shape = [1] * n
            mesh_shape[q] = 2
            mesh_shape[q_next] = 2

            s = s * torch.where(
                (torch.arange(2, device=device).view(*[2 if i == q else 1 for i in range(n)]) ==
                 torch.arange(2, device=device).view(*[2 if i == q_next else 1 for i in range(n)])).unsqueeze(0),
                phase_same,
                phase_diff
            )
        return s

    for _ in range(reps):
        state = apply_hadamard_all(state, n_qubits)
        state = apply_rz_all(state, X, n_qubits)
        state = apply_rzz_cyclic(state, X, n_qubits)

    return state.view(B, 2 ** n_qubits)


def compute_quantum_gram_matrix_depth(A_rows: np.ndarray, A_cols: np.ndarray, n_qubits: int, depth: int) -> np.ndarray:
    """
    Computes exact PyTorch complex128 fidelity Gram matrix for arbitrary feature map depth.
    """
    t_rows = torch.tensor(A_rows, dtype=torch.float64, device=device)
    states_rows = simulate_zz_feature_map_depth(t_rows, n_qubits=n_qubits, reps=depth)
    B_rows = states_rows.shape[0]
    states_rows_flat = states_rows.reshape(B_rows, -1)

    if A_rows is A_cols or (len(A_rows) == len(A_cols) and np.array_equal(A_rows, A_cols)):
        states_cols_flat = states_rows_flat
        B_cols = B_rows
    else:
        t_cols = torch.tensor(A_cols, dtype=torch.float64, device=device)
        states_cols = simulate_zz_feature_map_depth(t_cols, n_qubits=n_qubits, reps=depth)
        B_cols = states_cols.shape[0]
        states_cols_flat = states_cols.reshape(B_cols, -1)

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
    q_diag_mean = float(np.mean(q_diag))
    q_diag_std = float(np.std(q_diag))

    # Eigenvalue spectrum of Quantum Gram matrix
    q_eigvals = np.linalg.eigvalsh(K_q_test)
    q_psd = bool(np.min(q_eigvals) >= -1e-6)
    q_eig_pos = np.clip(q_eigvals, 1e-12, None)
    q_eig_norm = q_eig_pos / np.sum(q_eig_pos)
    q_spectral_entropy = float(-np.sum(q_eig_norm * np.log2(q_eig_norm)))
    q_effective_rank = float(np.exp(-np.sum(q_eig_norm * np.log(q_eig_norm))))
    q_largest_eig_fraction = float(np.max(q_eigvals) / np.sum(q_eig_pos))

    # Off-diagonal elements
    triu_idx = np.triu_indices(N, k=1)
    q_offdiag = K_q_test[triu_idx]
    q_mean = float(np.mean(q_offdiag))
    q_std = float(np.std(q_offdiag))
    q_diversity = float(1.0 - q_mean)

    # 2. RBF Diagnostics
    rbf_sym_err = float(np.max(np.abs(K_rbf_test - K_rbf_test.T)))
    rbf_finite = bool(np.all(np.isfinite(K_rbf_test)))
    rbf_diag = np.diag(K_rbf_test)
    rbf_diag_mean = float(np.mean(rbf_diag))
    rbf_diag_std = float(np.std(rbf_diag))
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
        "q_finite_check": q_finite,
        "q_diag_mean": q_diag_mean,
        "q_diag_std": q_diag_std,
        "q_kernel_mean": q_mean,
        "q_kernel_std": q_std,
        "q_kernel_diversity": q_diversity,
        "q_spectral_entropy": q_spectral_entropy,
        "q_effective_rank": q_effective_rank,
        "q_largest_eig_fraction": q_largest_eig_fraction,
        "q_label_alignment": q_label_alignment,
        "rbf_symmetry_error": rbf_sym_err,
        "rbf_finite_check": rbf_finite,
        "rbf_diag_mean": rbf_diag_mean,
        "rbf_diag_std": rbf_diag_std,
        "rbf_kernel_mean": rbf_mean,
        "rbf_kernel_std": rbf_std,
        "rbf_kernel_diversity": rbf_diversity,
        "rbf_label_alignment": rbf_label_alignment,
        "gram_pearson_r": float(pearson_r),
        "gram_spearman_rho": float(spearman_rho),
    }


def optimize_threshold(decision_scores: np.ndarray, y_true: np.ndarray, n_steps: int = 200) -> float:
    """
    Selects optimal decision threshold on VALIDATION SPLIT ONLY.
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


def evaluate_model_metrics(clf, X_eval, y_eval, opt_thresh: float) -> dict:
    """
    Computes out-of-sample test metrics.
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
        "predictions": preds,
        "decision_scores": scores,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "opt_thresh": opt_thresh
    }


def paired_permutation_test(deltas: np.ndarray, n_perm: int = 10000) -> float:
    """Two-sided paired permutation test."""
    obs = np.abs(np.mean(deltas))
    n = len(deltas)
    signs = np.random.choice([-1, 1], size=(n_perm, n))
    perm_means = np.abs(np.mean(signs * deltas, axis=1))
    p_val = np.mean(perm_means >= obs)
    return float(p_val)


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 10000, alpha: float = 0.05) -> tuple:
    """95% Percentile Bootstrap Confidence Interval."""
    n = len(deltas)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = np.random.choice(deltas, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    low = float(np.percentile(boot_means, 100 * (alpha / 2)))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return low, high


def benjamini_hochberg(p_values: list) -> list:
    """Benjamini-Hochberg FDR correction."""
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
    print("EXPERIMENT 43: REPRESENTATION-AWARE QUANTUM ENCODING DIAGNOSTIC STUDY")
    print("Ablating Angular Mappings (A1-A4) & Feature-Map Depths (B1-B3) on SMS + MPNet 8D")
    print(f"Seeds: {SEEDS} (N=10) | Total Conditions: {len(CONDITIONS)} (70 Quantum Runs)")
    print("=" * 80)

    start_total_time = time.time()
    seed_records = []
    geometry_records = []
    runtime_records = []

    # Store baseline references per seed: RBF & Linear SVM
    rbf_reference_store = {}
    linear_reference_store = {}

    for seed in SEEDS:
        print(f"\n--- Seed {seed:4d} ---")
        t0_prep = time.time()
        df_train, df_val, df_test = load_dataset_splits(DATASET, seed)
        y_tr = df_train["target"].to_numpy().astype(int)
        y_va = df_val["target"].to_numpy().astype(int)
        y_te = df_test["target"].to_numpy().astype(int)

        # 1. Load canonical MPNet embeddings
        X_tr_raw, X_va_raw, X_te_raw = get_or_load_mpnet_embeddings(DATASET, seed, df_train, df_val, df_test)

        # 2. Canonical SVD + StandardScaler to matched 8D
        Z_tr, Z_va, Z_te = project_svd_scaler(X_tr_raw, X_va_raw, X_te_raw, target_dim=DIMENSION, seed=seed)
        prep_time = time.time() - t0_prep

        # 3. Canonical RBF Reference Model (Fit once per seed)
        t0_rbf_fit = time.time()
        rbf_svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=seed)
        rbf_svm.fit(Z_tr, y_tr)
        rbf_fit_time = time.time() - t0_rbf_fit

        val_scores_rbf = rbf_svm.decision_function(Z_va)
        opt_thresh_rbf = optimize_threshold(val_scores_rbf, y_va)

        t0_rbf_inf = time.time()
        metrics_rbf = evaluate_model_metrics(rbf_svm, Z_te, y_te, opt_thresh_rbf)
        rbf_inf_time = time.time() - t0_rbf_inf

        rbf_reference_store[seed] = metrics_rbf

        # Compute RBF test-test Gram matrix for geometry correlation
        gamma_val = 1.0 / (DIMENSION * np.var(Z_tr) + 1e-12)
        dists_sq = np.sum((Z_te[:, np.newaxis, :] - Z_te[np.newaxis, :, :]) ** 2, axis=-1)
        K_rbf_te_te = np.exp(-gamma_val * dists_sq)

        # 4. Contextual Linear SVM Baseline
        t0_lin_fit = time.time()
        lin_svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
        lin_svm.fit(Z_tr, y_tr)
        lin_fit_time = time.time() - t0_lin_fit
        val_scores_lin = lin_svm.decision_function(Z_va)
        opt_thresh_lin = optimize_threshold(val_scores_lin, y_va)
        metrics_lin = evaluate_model_metrics(lin_svm, Z_te, y_te, opt_thresh_lin)
        linear_reference_store[seed] = metrics_lin

        print(f"  Classical References: RBF F1={metrics_rbf['f1']:.4f} | Linear F1={metrics_lin['f1']:.4f}")

        # 5. Evaluate all 7 Quantum Encoding Conditions
        for cond in CONDITIONS:
            c_id = cond["cond_id"]
            mapping_type = cond["angle_mapping"]
            depth = cond["depth"]
            label = cond["label"]

            t0_cond_total = time.time()

            # Angular mapping
            t0_ang = time.time()
            A_tr, A_va, A_te = apply_angular_transformation(Z_tr, Z_va, Z_te, mapping_type)
            angle_time = time.time() - t0_ang

            # Quantum Gram construction
            t0_q_gram = time.time()
            K_q_tr_tr = compute_quantum_gram_matrix_depth(A_tr, A_tr, n_qubits=DIMENSION, depth=depth)
            K_q_va_tr = compute_quantum_gram_matrix_depth(A_va, A_tr, n_qubits=DIMENSION, depth=depth)
            K_q_te_tr = compute_quantum_gram_matrix_depth(A_te, A_tr, n_qubits=DIMENSION, depth=depth)
            K_q_te_te = compute_quantum_gram_matrix_depth(A_te, A_te, n_qubits=DIMENSION, depth=depth)
            q_gram_time = time.time() - t0_q_gram

            # Quantum SVM fitting
            t0_q_fit = time.time()
            q_svm = SVC(kernel="precomputed", C=1.0, class_weight="balanced", random_state=seed)
            q_svm.fit(K_q_tr_tr, y_tr)
            q_fit_time = time.time() - t0_q_fit

            # Validation threshold optimization
            val_scores_q = q_svm.decision_function(K_q_va_tr)
            opt_thresh_q = optimize_threshold(val_scores_q, y_va)

            # Test inference
            t0_q_inf = time.time()
            metrics_q = evaluate_model_metrics(q_svm, K_q_te_tr, y_te, opt_thresh_q)
            q_inf_time = time.time() - t0_q_inf

            cond_total_time = time.time() - t0_cond_total

            # Geometry diagnostics
            geom = compute_comprehensive_geometry(K_q_te_te, K_rbf_te_te, y_te)
            geom.update({
                "cond_id": c_id,
                "label": label,
                "seed": seed,
                "angle_mapping": mapping_type,
                "depth": depth
            })
            geometry_records.append(geom)

            delta_q_rbf = metrics_q["f1"] - metrics_rbf["f1"]
            delta_q_lin = metrics_q["f1"] - metrics_lin["f1"]

            seed_records.append({
                "cond_id": c_id,
                "label": label,
                "family": cond["family"],
                "angle_mapping": mapping_type,
                "depth": depth,
                "seed": seed,
                "quantum_f1": metrics_q["f1"],
                "rbf_f1": metrics_rbf["f1"],
                "linear_f1": metrics_lin["f1"],
                "delta_f1_q_rbf": delta_q_rbf,
                "delta_f1_q_lin": delta_q_lin,
                "quantum_pr_auc": metrics_q["pr_auc"],
                "rbf_pr_auc": metrics_rbf["pr_auc"],
                "quantum_roc_auc": metrics_q["roc_auc"],
                "rbf_roc_auc": metrics_rbf["roc_auc"],
                "quantum_bacc": metrics_q["balanced_accuracy"],
                "rbf_bacc": metrics_rbf["balanced_accuracy"],
                "quantum_opt_thresh": metrics_q["opt_thresh"],
                "rbf_opt_thresh": metrics_rbf["opt_thresh"],
                "q_kernel_diversity": geom["q_kernel_diversity"],
                "rbf_kernel_diversity": geom["rbf_kernel_diversity"],
                "q_label_alignment": geom["q_label_alignment"],
                "rbf_label_alignment": geom["rbf_label_alignment"],
                "gram_pearson_r": geom["gram_pearson_r"],
                "q_spectral_entropy": geom["q_spectral_entropy"],
                "q_effective_rank": geom["q_effective_rank"],
                "q_largest_eig_fraction": geom["q_largest_eig_fraction"],
                "q_diag_mean": geom["q_diag_mean"],
                "q_offdiag_mean": geom["q_kernel_mean"],
                "q_offdiag_std": geom["q_kernel_std"]
            })

            runtime_records.append({
                "cond_id": c_id,
                "seed": seed,
                "preprocessing_time_s": prep_time,
                "angle_mapping_time_s": angle_time,
                "kernel_construction_time_s": q_gram_time,
                "svm_fitting_time_s": q_fit_time,
                "inference_time_s": q_inf_time,
                "total_condition_time_s": cond_total_time,
                "rbf_fitting_time_s": rbf_fit_time,
                "rbf_inference_time_s": rbf_inf_time
            })

            print(f"    [{c_id:24s}] Q_F1={metrics_q['f1']:.4f} | Δ(Q-RBF)={delta_q_rbf:+.4f} | Div={geom['q_kernel_diversity']:.4f} | Align={geom['q_label_alignment']:.4f} | r={geom['gram_pearson_r']:.4f}")

    df_seeds = pd.DataFrame(seed_records)
    df_geom = pd.DataFrame(geometry_records)
    df_runtime = pd.DataFrame(runtime_records)

    # ============================================================
    # BASELINE CONSISTENCY CHECK (Condition 1 vs Exp42 Canonical)
    # ============================================================
    canon_sub = df_seeds[df_seeds["cond_id"] == "canonical_2layer_0_pi"]
    canon_q_mean = float(canon_sub["quantum_f1"].mean())
    canon_rbf_mean = float(canon_sub["rbf_f1"].mean())
    canon_delta_mean = float(canon_sub["delta_f1_q_rbf"].mean())

    print("\n" + "=" * 80)
    print("BASELINE CONSISTENCY VERIFICATION:")
    print(f"Exp43 Canonical Baseline (SMS + MPNet 8D, N=10 Seeds):")
    print(f"  Quantum F1 Mean: {canon_q_mean:.4f} (Exp42 Reference: 0.3756 ± 0.1133)")
    print(f"  Classical RBF Mean: {canon_rbf_mean:.4f} (Exp42 Reference: 0.9045 ± 0.0511)")
    print(f"  Mean Δ(Q - RBF): {canon_delta_mean:+.4f} (Exp42 Reference: -0.5288)")
    print(f"  Baseline Difference: |Δ_Exp43 - Δ_Exp42| = {abs(canon_delta_mean - (-0.5288)):.4f}")
    if abs(canon_delta_mean - (-0.5288)) < 0.01:
        print("  -> BASELINE CONSISTENCY CHECK: PASSED (Exact Replication within 0.01 F1)")
    else:
        print("  -> BASELINE CONSISTENCY CHECK: MINOR DRIFT DIAGNOSED")
    print("=" * 80)

    # ============================================================
    # INFERENTIAL STATISTICAL ANALYSIS ACROSS 7 CONDITIONS
    # ============================================================
    stats_list = []
    raw_p_values = []

    for cond in CONDITIONS:
        c_id = cond["cond_id"]
        sub = df_seeds[df_seeds["cond_id"] == c_id]
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
        t_crit = stats.t.ppf(0.975, df=len(deltas) - 1)
        se = std_delta / np.sqrt(len(deltas))
        t_ci_low = float(mean_delta - t_crit * se)
        t_ci_high = float(mean_delta + t_crit * se)

        # Paired permutation test (10,000 permutations)
        perm_p = paired_permutation_test(deltas, n_perm=10000)
        raw_p_values.append(perm_p)

        # Bootstrap CI (10,000 resamples)
        boot_low, boot_high = bootstrap_ci(deltas, n_boot=10000)

        # Seed counts
        q_wins = int(np.sum(deltas > EPSILON))
        rbf_wins = int(np.sum(deltas < -EPSILON))
        ties = int(np.sum(np.abs(deltas) <= EPSILON))

        # Practical equivalence verdict
        is_stat_sig = (perm_p < 0.05) and not (t_ci_low <= 0 <= t_ci_high)
        if is_stat_sig and mean_delta >= EPSILON:
            classification = "A. Confirmed practical Q advantage"
        elif is_stat_sig and mean_delta <= -EPSILON:
            classification = "D. Classical advantage (|Δ| >= 0.01)"
        elif is_stat_sig and abs(mean_delta) < EPSILON:
            classification = "B. Statistically detectable but practically equivalent"
        else:
            classification = "C. No statistically detectable difference (spans zero)"

        stats_list.append({
            "cond_id": c_id,
            "label": cond["label"],
            "family": cond["family"],
            "angle_mapping": cond["angle_mapping"],
            "depth": cond["depth"],
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
            "seed_q_wins": q_wins,
            "seed_rbf_wins": rbf_wins,
            "seed_ties": ties,
            "verdict": classification
        })

    # Apply BH-FDR correction across the 7 primary hypotheses
    bh_p_values = benjamini_hochberg(raw_p_values)
    for i, rec in enumerate(stats_list):
        rec["permutation_p_bh_fdr"] = bh_p_values[i]

    df_stats = pd.DataFrame(stats_list)

    # ============================================================
    # SUB-ANALYSIS 1: ANGLE-MAPPING DIAGNOSTIC (A1, A2, A3, A4)
    # ============================================================
    angle_conds = ["canonical_2layer_0_pi", "angle_minus_pi_pi", "angle_0_2pi", "angle_train_normalized"]
    df_angle = df_stats[df_stats["cond_id"].isin(angle_conds)].copy()

    # ============================================================
    # SUB-ANALYSIS 2: FEATURE-MAP DEPTH ABLATION (B1, B2, B3)
    # ============================================================
    depth_conds = ["depth_1", "depth_2", "depth_3"]
    df_depth = df_stats[df_stats["cond_id"].isin(depth_conds)].copy()

    # ============================================================
    # SUB-ANALYSIS 3: GEOMETRY-PERFORMANCE CORRELATION ANALYSIS (70 RUNS)
    # ============================================================
    all_q_f1 = df_seeds["quantum_f1"].to_numpy()
    all_deltas = df_seeds["delta_f1_q_rbf"].to_numpy()
    all_div = df_seeds["q_kernel_diversity"].to_numpy()
    all_align = df_seeds["q_label_alignment"].to_numpy()
    all_r = df_seeds["gram_pearson_r"].to_numpy()
    all_eff_rank = df_seeds["q_effective_rank"].to_numpy()

    def safe_corr(x, y):
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return 0.0, 1.0
        return stats.pearsonr(x, y)

    r_div_f1, p_div_f1 = safe_corr(all_div, all_q_f1)
    r_align_f1, p_align_f1 = safe_corr(all_align, all_q_f1)
    r_geom_f1, p_geom_f1 = safe_corr(all_r, all_q_f1)
    r_div_delta, p_div_delta = safe_corr(all_div, all_deltas)
    r_align_delta, p_align_delta = safe_corr(all_align, all_deltas)
    r_rank_f1, p_rank_f1 = safe_corr(all_eff_rank, all_q_f1)

    geom_assoc_records = [
        {"metric_pair": "Kernel Diversity vs Quantum F1", "pearson_r": r_div_f1, "p_value": p_div_f1, "notes": "Correlation across all 70 quantum runs"},
        {"metric_pair": "Label Alignment vs Quantum F1", "pearson_r": r_align_f1, "p_value": p_align_f1, "notes": "Correlation across all 70 quantum runs"},
        {"metric_pair": "Q/RBF Geometry Correlation (r) vs Quantum F1", "pearson_r": r_geom_f1, "p_value": p_geom_f1, "notes": "Correlation with Gram Pearson correlation"},
        {"metric_pair": "Kernel Diversity vs Q-RBF ΔF1", "pearson_r": r_div_delta, "p_value": p_div_delta, "notes": "Correlation with relative performance differential"},
        {"metric_pair": "Label Alignment vs Q-RBF ΔF1", "pearson_r": r_align_delta, "p_value": p_align_delta, "notes": "Correlation with relative performance differential"},
        {"metric_pair": "Effective Rank vs Quantum F1", "pearson_r": r_rank_f1, "p_value": p_rank_f1, "notes": "Correlation with exponential spectral entropy"}
    ]
    df_geom_assoc = pd.DataFrame(geom_assoc_records)

    # Save CSV outputs
    df_seeds.to_csv(os.path.join(EXP43_DIR, "exp43_seed_results.csv"), index=False)
    df_geom.to_csv(os.path.join(EXP43_DIR, "exp43_geometry.csv"), index=False)
    df_stats.to_csv(os.path.join(EXP43_DIR, "exp43_statistics.csv"), index=False)
    df_angle.to_csv(os.path.join(EXP43_DIR, "exp43_angle_mapping.csv"), index=False)
    df_depth.to_csv(os.path.join(EXP43_DIR, "exp43_depth_ablation.csv"), index=False)
    df_geom_assoc.to_csv(os.path.join(EXP43_DIR, "exp43_geometry_associations.csv"), index=False)
    df_runtime.to_csv(os.path.join(EXP43_DIR, "exp43_runtime.csv"), index=False)

    # Save formatted summary CSV
    summary_rows = []
    for _, row in df_stats.iterrows():
        summary_rows.append({
            "cond_id": row["cond_id"],
            "label": row["label"],
            "quantum_f1": f"{row['q_mean_f1']:.4f} ± {row['q_std_f1']:.4f}",
            "rbf_f1": f"{row['rbf_mean_f1']:.4f} ± {row['rbf_std_f1']:.4f}",
            "linear_f1": f"{row['linear_mean_f1']:.4f} ± {row['linear_std_f1']:.4f}",
            "delta_q_rbf": f"{row['delta_mean']:+.4f} [{row['student_t_ci_95_low']:+.4f}, {row['student_t_ci_95_high']:+.4f}]",
            "perm_p_raw": f"{row['permutation_p_raw']:.4f}",
            "perm_p_bh_fdr": f"{row['permutation_p_bh_fdr']:.4f}",
            "seed_wins (Q/RBF/Tie)": f"{row['seed_q_wins']}/{row['seed_rbf_wins']}/{row['seed_ties']}",
            "verdict": row["verdict"]
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(EXP43_DIR, "exp43_summary.csv"), index=False)

    # ============================================================
    # GENERATE 8 PUBLICATION FIGURES
    # ============================================================
    print("\nGenerating 8 Publication Figures in results/exp43/figures/ ...")

    cond_labels_short = [
        "A1: [0, π] (2L)",
        "A2: [-π, π] (2L)",
        "A3: [0, 2π] (2L)",
        "A4: Norm CDF (2L)",
        "B1: Depth 1",
        "B2: Depth 2",
        "B3: Depth 3"
    ]
    cond_ids_ordered = [c["cond_id"] for c in CONDITIONS]

    # FIGURE 1: Q-RBF F1 Difference by Encoding Condition (Seed dots + Mean + CI)
    plt.figure(figsize=(11, 6))
    for i, c_id in enumerate(cond_ids_ordered):
        sub = df_seeds[df_seeds["cond_id"] == c_id]
        y_vals = sub["delta_f1_q_rbf"].to_numpy()
        x_vals = np.random.normal(i, 0.06, size=len(y_vals))
        plt.scatter(x_vals, y_vals, alpha=0.7, color='#7c3aed', s=50, edgecolors='black')

        stat_row = df_stats[df_stats["cond_id"] == c_id].iloc[0]
        m = stat_row["delta_mean"]
        low = stat_row["student_t_ci_95_low"]
        high = stat_row["student_t_ci_95_high"]
        plt.errorbar(i, m, yerr=[[m - low], [high - m]], fmt='o', color='red', capsize=6, elinewidth=2.5, markeredgewidth=1.5, markersize=8)

    plt.axhline(0.0, color='black', linestyle='-', linewidth=1)
    plt.axhline(EPSILON, color='gray', linestyle='--', linewidth=1, label=r'$\epsilon = +0.01$ Practical Equivalence')
    plt.axhline(-EPSILON, color='gray', linestyle='--', linewidth=1, label=r'$\epsilon = -0.01$ Practical Equivalence')
    plt.fill_between([-0.5, 6.5], -EPSILON, EPSILON, color='gray', alpha=0.15)
    plt.xticks(range(7), cond_labels_short, rotation=15, ha='right', fontsize=10)
    plt.ylabel(r"$\Delta\mathrm{F1} = \mathrm{Quantum\ F1} - \mathrm{RBF\ F1}$", fontsize=11)
    plt.title("Figure 1: Paired Q-vs-RBF F1 Differences Across 7 Quantum Encodings (Exp 43, N=10 Seeds)", fontsize=12, fontweight='bold')
    plt.legend(loc='lower left', fontsize=9)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "fig1_q_rbf_deltas.png"), dpi=300)
    plt.close()

    # FIGURE 2: Quantum F1 by Angle-Mapping Variant
    plt.figure(figsize=(8, 5))
    angle_names = ["A1: [0, π]", "A2: [-π, π]", "A3: [0, 2π]", "A4: Normal CDF"]
    angle_q_means = [df_stats[df_stats["cond_id"] == c]["q_mean_f1"].iloc[0] for c in angle_conds]
    angle_q_stds = [df_stats[df_stats["cond_id"] == c]["q_std_f1"].iloc[0] for c in angle_conds]
    rbf_ref_mean = df_stats["rbf_mean_f1"].mean()

    bars = plt.bar(angle_names, angle_q_means, yerr=angle_q_stds, capsize=5, color='#8b5cf6', edgecolor='black', alpha=0.85)
    plt.axhline(rbf_ref_mean, color='#2563eb', linestyle='--', linewidth=2, label=f"Classical RBF Reference ({rbf_ref_mean:.4f})")
    plt.ylabel("Test F1 Score (Mean ± Std)", fontsize=11)
    plt.title("Figure 2: Quantum F1 Performance by Angle-Mapping Strategy (Depth = 2)", fontsize=12, fontweight='bold')
    plt.ylim(0.0, 1.0)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "fig2_angle_mapping_f1.png"), dpi=300)
    plt.close()

    # FIGURE 3: Quantum F1 by Feature-Map Depth
    plt.figure(figsize=(7, 5))
    depth_names = ["Depth 1 Layer", "Depth 2 Layers", "Depth 3 Layers"]
    depth_q_means = [df_stats[df_stats["cond_id"] == c]["q_mean_f1"].iloc[0] for c in depth_conds]
    depth_q_stds = [df_stats[df_stats["cond_id"] == c]["q_std_f1"].iloc[0] for c in depth_conds]

    plt.plot(depth_names, depth_q_means, marker='o', linewidth=2.5, markersize=8, color='#7c3aed', label="Quantum Fidelity Kernel")
    plt.errorbar(depth_names, depth_q_means, yerr=depth_q_stds, fmt='none', color='#7c3aed', capsize=5)
    plt.axhline(rbf_ref_mean, color='#2563eb', linestyle='--', linewidth=2, label=f"Classical RBF Reference ({rbf_ref_mean:.4f})")
    plt.ylabel("Test F1 Score (Mean ± Std)", fontsize=11)
    plt.title("Figure 3: Quantum Classification Performance vs Feature-Map Depth", fontsize=12, fontweight='bold')
    plt.ylim(0.0, 1.0)
    plt.legend(loc='center right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "fig3_depth_f1.png"), dpi=300)
    plt.close()

    # FIGURE 4: Quantum Kernel Diversity by Condition
    plt.figure(figsize=(10, 5))
    div_means = [df_geom[df_geom["cond_id"] == c]["q_kernel_diversity"].mean() for c in cond_ids_ordered]
    div_stds = [df_geom[df_geom["cond_id"] == c]["q_kernel_diversity"].std() for c in cond_ids_ordered]
    rbf_div_mean = df_geom["rbf_kernel_diversity"].mean()

    plt.bar(cond_labels_short, div_means, yerr=div_stds, capsize=4, color='#0284c7', edgecolor='black', alpha=0.85)
    plt.axhline(rbf_div_mean, color='#ea580c', linestyle='--', linewidth=1.5, label=f"Classical RBF Diversity ({rbf_div_mean:.4f})")
    plt.ylabel("Quantum Kernel Diversity (1 - Mean Off-Diag)", fontsize=11)
    plt.title("Figure 4: Quantum Kernel Diversity across Encoding Conditions", fontsize=12, fontweight='bold')
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "fig4_kernel_diversity.png"), dpi=300)
    plt.close()

    # FIGURE 5: Quantum Label Alignment by Condition
    plt.figure(figsize=(10, 5))
    align_means = [df_geom[df_geom["cond_id"] == c]["q_label_alignment"].mean() for c in cond_ids_ordered]
    align_stds = [df_geom[df_geom["cond_id"] == c]["q_label_alignment"].std() for c in cond_ids_ordered]
    rbf_align_mean = df_geom["rbf_label_alignment"].mean()

    plt.bar(cond_labels_short, align_means, yerr=align_stds, capsize=4, color='#10b981', edgecolor='black', alpha=0.85)
    plt.axhline(rbf_align_mean, color='#ea580c', linestyle='--', linewidth=1.5, label=f"Classical RBF Alignment ({rbf_align_mean:.4f})")
    plt.ylabel("Target Label Alignment", fontsize=11)
    plt.title("Figure 5: Target Label Alignment across Quantum Encoding Conditions", fontsize=12, fontweight='bold')
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "fig5_label_alignment.png"), dpi=300)
    plt.close()

    # FIGURE 6: Quantum-vs-RBF Geometry Correlation by Condition
    plt.figure(figsize=(10, 5))
    r_means = [df_geom[df_geom["cond_id"] == c]["gram_pearson_r"].mean() for c in cond_ids_ordered]
    r_stds = [df_geom[df_geom["cond_id"] == c]["gram_pearson_r"].std() for c in cond_ids_ordered]

    plt.bar(cond_labels_short, r_means, yerr=r_stds, capsize=4, color='#f59e0b', edgecolor='black', alpha=0.85)
    plt.ylabel("Gram Pearson Correlation (r)", fontsize=11)
    plt.title("Figure 6: Quantum vs Classical RBF Hilbert Space Correlation by Condition", fontsize=12, fontweight='bold')
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "fig6_geom_correlation.png"), dpi=300)
    plt.close()

    # FIGURE 7: Kernel Diversity vs Q-RBF Delta F1 (All 70 runs)
    plt.figure(figsize=(8, 6))
    plt.scatter(df_seeds["q_kernel_diversity"], df_seeds["delta_f1_q_rbf"], c=df_seeds["quantum_f1"], cmap='viridis', s=60, edgecolors='k')
    cb7 = plt.colorbar()
    cb7.set_label("Quantum F1", fontsize=10)
    z7 = np.polyfit(df_seeds["q_kernel_diversity"], df_seeds["delta_f1_q_rbf"], 1)
    p7 = np.poly1d(z7)
    x7_line = np.linspace(df_seeds["q_kernel_diversity"].min(), df_seeds["q_kernel_diversity"].max(), 50)
    plt.plot(x7_line, p7(x7_line), 'r--', label=f"Trend (r={r_div_delta:+.2f})")
    plt.axhline(0.0, color='black', linestyle='-', linewidth=0.8)
    plt.xlabel("Quantum Kernel Diversity", fontsize=11)
    plt.ylabel(r"$\Delta\mathrm{F1}\ (\mathrm{Quantum} - \mathrm{RBF})$", fontsize=11)
    plt.title(f"Figure 7: Kernel Diversity vs Q-RBF Performance Differential (N=70 Runs)", fontsize=12, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "fig7_diversity_vs_delta.png"), dpi=300)
    plt.close()

    # FIGURE 8: Label Alignment vs Q-RBF Delta F1 (All 70 runs)
    plt.figure(figsize=(8, 6))
    plt.scatter(df_seeds["q_label_alignment"], df_seeds["delta_f1_q_rbf"], c=df_seeds["quantum_f1"], cmap='plasma', s=60, edgecolors='k')
    cb8 = plt.colorbar()
    cb8.set_label("Quantum F1", fontsize=10)
    z8 = np.polyfit(df_seeds["q_label_alignment"], df_seeds["delta_f1_q_rbf"], 1)
    p8 = np.poly1d(z8)
    x8_line = np.linspace(df_seeds["q_label_alignment"].min(), df_seeds["q_label_alignment"].max(), 50)
    plt.plot(x8_line, p8(x8_line), 'r--', label=f"Trend (r={r_align_delta:+.2f})")
    plt.axhline(0.0, color='black', linestyle='-', linewidth=0.8)
    plt.xlabel("Quantum Target Label Alignment", fontsize=11)
    plt.ylabel(r"$\Delta\mathrm{F1}\ (\mathrm{Quantum} - \mathrm{RBF})$", fontsize=11)
    plt.title(f"Figure 8: Target Label Alignment vs Q-RBF Performance Differential (N=70 Runs)", fontsize=12, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "fig8_alignment_vs_delta.png"), dpi=300)
    plt.close()

    total_time = time.time() - start_total_time
    print("\n" + "=" * 80)
    print("EXPERIMENT 43 EXECUTION COMPLETE!")
    print(f"Total Execution Time: {total_time:.2f}s")
    print(f"Results Directory: {EXP43_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
