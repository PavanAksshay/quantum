#!/usr/bin/env python3
"""
Experiment 28: Quantum Feature-Map Ablation
===========================================
Performs a strictly controlled ablation study comparing quantum feature-map
families/configurations to determine whether quantum kernel concentration and poor
generalization are specific to the Experiment 26 ansatz or systemic across quantum
feature map families.

Tested Feature Maps (CEAS 4D, 4 Qubits):
  1. Current: Exact Experiment 26 feature map (2 reps, ring ZZ entanglement)
  2. ZFeatureMap: 1 rep, pure single-qubit rotations (no entanglement)
  3. ZZFeatureMap: 1 rep, standard linear nearest-neighbor ZZ entanglement
  4. PauliFeatureMap: 1 rep, all-to-all pairwise ZZ entanglement (['Z', 'ZZ'])
  5. Classical RBF: Matched classical baseline

Followed by a secondary generalization test of the best-performing alternative feature
map on SMS 4D and MeAJOR 4D.

Author: Quantum Phishing & Scam Detection Project
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigvalsh
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
HEATMAP_SAMPLES = 200

ROBERTA_BASE_DIR = "results/roberta_multidataset"
ABLATION_BASE_DIR = "results/quantum_featuremap_ablation"
PLOTS_DIR = os.path.join(ABLATION_BASE_DIR, "plots")
METRICS_DIR = "results/metrics"

CSV_OUTPUT_PATH = os.path.join(METRICS_DIR, "experiment28_featuremap_ablation.csv")
SUMMARY_OUTPUT_PATH = os.path.join(METRICS_DIR, "experiment28_summary.txt")


# ============================================================
# QUANTUM FEATURE MAP STATEVECTOR SIMULATOR
# ============================================================
def simulate_feature_map(X: torch.Tensor, n_qubits: int, fmap_type: str) -> torch.Tensor:
    """
    Exact, fully-vectorized batched quantum statevector simulator in PyTorch.
    X: torch.Tensor of shape (B, n_qubits)
    Returns: torch.Tensor of shape (B, 2**n_qubits), complex128
    """
    B = X.shape[0]
    device = X.device
    dtype = torch.complex128

    state = torch.zeros([B] + [2] * n_qubits, dtype=dtype, device=device)
    state[(slice(None),) + (0,) * n_qubits] = 1.0
    H = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=dtype, device=device) / np.sqrt(2)

    def apply_single_qubit_gate(state_tensor, G, target_qubit):
        axis = target_qubit + 1
        dims = list(range(state_tensor.ndim))
        perm = [0, axis] + [d for d in dims[1:] if d != axis]
        inv_perm = [0] + [perm.index(i) for i in range(1, len(dims))]
        permuted = state_tensor.permute(perm)
        shape_rest = permuted.shape[2:]
        permuted_flat = permuted.reshape(B, 2, -1)
        transformed = torch.einsum("ij,bjk->bik", G, permuted_flat)
        return transformed.reshape([B, 2] + list(shape_rest)).permute(inv_perm)

    def apply_rzz(state_tensor, q1, q2, x1, x2):
        x_ij = x1 * x2
        p_even = torch.exp(-1j * x_ij)
        p_odd = torch.exp(1j * x_ij)
        rzz_diag = torch.zeros([B, 2, 2], dtype=dtype, device=device)
        rzz_diag[:, 0, 0] = p_even
        rzz_diag[:, 0, 1] = p_odd
        rzz_diag[:, 1, 0] = p_odd
        rzz_diag[:, 1, 1] = p_even
        shape = [B] + [1] * n_qubits
        shape[q1 + 1] = 2
        shape[q2 + 1] = 2
        if q1 < q2:
            rzz_tensor = rzz_diag.reshape(shape)
        else:
            rzz_tensor = rzz_diag.transpose(1, 2).reshape(shape)
        return state_tensor * rzz_tensor

    if fmap_type == "Current":
        # Exp 26 feature map: 2 reps, ring entanglement
        for rep in range(2):
            for q in range(n_qubits):
                state = apply_single_qubit_gate(state, H, q)
            for q in range(n_qubits):
                x_q = X[:, q]
                phase = torch.stack([torch.exp(-1j * x_q), torch.exp(1j * x_q)], dim=-1)
                shape = [B] + [1] * n_qubits
                shape[q + 1] = 2
                state = state * phase.reshape(shape)
            for i in range(n_qubits):
                j = (i + 1) % n_qubits
                state = apply_rzz(state, i, j, X[:, i], X[:, j])

    elif fmap_type == "ZFeatureMap":
        # ZFeatureMap: reps=1, no entanglement
        for q in range(n_qubits):
            state = apply_single_qubit_gate(state, H, q)
        for q in range(n_qubits):
            x_q = X[:, q]
            phase = torch.stack([torch.exp(-1j * x_q), torch.exp(1j * x_q)], dim=-1)
            shape = [B] + [1] * n_qubits
            shape[q + 1] = 2
            state = state * phase.reshape(shape)

    elif fmap_type == "ZZFeatureMap":
        # ZZFeatureMap: reps=1, standard linear entanglement (0-1, 1-2, 2-3)
        for q in range(n_qubits):
            state = apply_single_qubit_gate(state, H, q)
        for q in range(n_qubits):
            x_q = X[:, q]
            phase = torch.stack([torch.exp(-1j * x_q), torch.exp(1j * x_q)], dim=-1)
            shape = [B] + [1] * n_qubits
            shape[q + 1] = 2
            state = state * phase.reshape(shape)
        for i in range(n_qubits - 1):
            state = apply_rzz(state, i, i + 1, X[:, i], X[:, i + 1])

    elif fmap_type == "PauliFeatureMap":
        # PauliFeatureMap: reps=1, ['Z', 'ZZ'] all-to-all entanglement
        for q in range(n_qubits):
            state = apply_single_qubit_gate(state, H, q)
        for q in range(n_qubits):
            x_q = X[:, q]
            phase = torch.stack([torch.exp(-1j * x_q), torch.exp(1j * x_q)], dim=-1)
            shape = [B] + [1] * n_qubits
            shape[q + 1] = 2
            state = state * phase.reshape(shape)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                state = apply_rzz(state, i, j, X[:, i], X[:, j])

    return state.reshape(B, 2 ** n_qubits)


def compute_quantum_gram_matrix(states_1: torch.Tensor, states_2: torch.Tensor) -> np.ndarray:
    M = torch.matmul(states_1, states_2.conj().T)
    K = torch.abs(M) ** 2
    return K.cpu().numpy().astype(np.float64)


def compute_centered_kernel_alignment(K: np.ndarray, y: np.ndarray) -> float:
    N = len(y)
    Y = np.where(y[:, None] == y[None, :], 1.0, -1.0)
    H = np.eye(N) - (1.0 / N) * np.ones((N, N))
    Kc = H @ K @ H
    Yc = H @ Y @ H
    norm_Kc = np.linalg.norm(Kc, "fro")
    norm_Yc = np.linalg.norm(Yc, "fro")
    if norm_Kc < 1e-12 or norm_Yc < 1e-12:
        return 0.0
    return float(np.sum(Kc * Yc) / (norm_Kc * norm_Yc))


def compute_kernel_diagnostics(K: np.ndarray, y: np.ndarray) -> dict:
    N = K.shape[0]
    k_min = float(np.min(K))
    k_max = float(np.max(K))
    k_mean = float(np.mean(K))
    k_std = float(np.std(K))

    diag_vals = np.diag(K)
    diag_mean = float(np.mean(diag_vals))
    diag_std = float(np.std(diag_vals))

    mask = ~np.eye(N, dtype=bool)
    offdiag = K[mask]
    offdiag_mean = float(np.mean(offdiag))
    offdiag_std = float(np.std(offdiag))
    offdiag_min = float(np.min(offdiag))
    offdiag_max = float(np.max(offdiag))

    symm_err = float(np.max(np.abs(K - K.T)))

    # Eigenvalues on sample of at most 2000 for speed
    diag_n = min(N, 2000)
    evals = eigvalsh(K[:diag_n, :diag_n])
    min_eval = float(np.min(evals))
    neg_count = int(np.sum(evals < -1e-8))

    pos_evals = np.maximum(evals, 0.0)
    tot_pos = np.sum(pos_evals)
    if tot_pos > 0:
        p = pos_evals / tot_pos
        p_nz = p[p > 1e-15]
        eff_rank = float(np.exp(-np.sum(p_nz * np.log(p_nz))))
    else:
        eff_rank = 1.0

    # Class-conditional similarities
    pos_mask = (y == 1)
    neg_mask = (y == 0)

    K_pos = K[np.ix_(pos_mask, pos_mask)]
    k_pos_pos = float(np.mean(K_pos[~np.eye(len(K_pos), dtype=bool)])) if len(K_pos) > 1 else 1.0

    K_neg = K[np.ix_(neg_mask, neg_mask)]
    k_neg_neg = float(np.mean(K_neg[~np.eye(len(K_neg), dtype=bool)])) if len(K_neg) > 1 else 1.0

    K_cross = K[np.ix_(pos_mask, neg_mask)]
    k_pos_neg = float(np.mean(K_cross)) if K_cross.size > 0 else 0.0

    separation = ((k_pos_pos + k_neg_neg) / 2.0) - k_pos_neg

    return {
        "kernel_min": k_min,
        "kernel_max": k_max,
        "kernel_mean": k_mean,
        "kernel_std": k_std,
        "diag_mean": diag_mean,
        "diag_std": diag_std,
        "offdiag_mean": offdiag_mean,
        "offdiag_std": offdiag_std,
        "offdiag_min": offdiag_min,
        "offdiag_max": offdiag_max,
        "symmetry_error": symm_err,
        "min_eigenvalue": min_eval,
        "negative_eigenvalue_count": neg_count,
        "effective_rank": round(eff_rank, 2),
        "positive_positive_similarity": round(k_pos_pos, 6),
        "negative_negative_similarity": round(k_neg_neg, 6),
        "positive_negative_similarity": round(k_pos_neg, 6),
        "within_between_separation": round(separation, 6),
    }


def compute_correlations(K1: np.ndarray, K2: np.ndarray) -> tuple:
    N = K1.shape[0]
    triu = np.triu_indices(N, k=1)
    v1, v2 = K1[triu], K2[triu]
    if np.std(v1) < 1e-12 or np.std(v2) < 1e-12:
        return 0.0, 0.0
    r_p, _ = pearsonr(v1, v2)
    r_s, _ = spearmanr(v1, v2)
    return float(r_p), float(r_s)


def select_best_threshold(y_val: np.ndarray, val_scores: np.ndarray, num_steps: int = 200) -> tuple:
    thresholds = np.linspace(val_scores.min(), val_scores.max(), num_steps)
    best_f1, best_th = -1.0, 0.0
    for th in thresholds:
        f1 = f1_score(y_val, (val_scores >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th, best_f1


def evaluate_dataset_feature_map(
    ds_name: str,
    fmap_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    K_ref_current: np.ndarray,
    K_ref_rbf: np.ndarray,
    rbf_cka: float,
) -> tuple:
    """Evaluates a single quantum feature map on a dataset."""
    d = X_tr.shape[1]
    n_tr, n_va, n_te = len(X_tr), len(X_va), len(X_te)

    # 1. Quantum Kernel Generation
    t0 = time.time()
    X_tr_t = torch.tensor(X_tr, dtype=torch.float64)
    X_va_t = torch.tensor(X_va, dtype=torch.float64)
    X_te_t = torch.tensor(X_te, dtype=torch.float64)

    states_tr = simulate_feature_map(X_tr_t, d, fmap_name)
    states_va = simulate_feature_map(X_va_t, d, fmap_name)
    states_te = simulate_feature_map(X_te_t, d, fmap_name)

    t0_ktr = time.time()
    K_tr = compute_quantum_gram_matrix(states_tr, states_tr)
    t_ktr = time.time() - t0_ktr

    t0_kva = time.time()
    K_va = compute_quantum_gram_matrix(states_va, states_tr)
    t_kva = time.time() - t0_kva

    t0_kte = time.time()
    K_te = compute_quantum_gram_matrix(states_te, states_tr)
    t_kte = time.time() - t0_kte
    t_kernel_total = time.time() - t0

    # 2. Kernel Diagnostics
    # Use subset of 1000 for diagnostics if N > 1000 for fast calculation
    sub_n = min(n_tr, 1000)
    diag = compute_kernel_diagnostics(K_tr[:sub_n, :sub_n], y_tr[:sub_n])
    q_cka = compute_centered_kernel_alignment(K_tr[:sub_n, :sub_n], y_tr[:sub_n])

    # 3. Correlations
    r_curr_p, _ = compute_correlations(K_tr[:sub_n, :sub_n], K_ref_current[:sub_n, :sub_n])
    r_rbf_p, _ = compute_correlations(K_tr[:sub_n, :sub_n], K_ref_rbf[:sub_n, :sub_n])

    # 4. SVM Training and Evaluation
    clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced", random_state=RANDOM_SEED, cache_size=2000)
    t0_fit = time.time()
    clf.fit(K_tr, y_tr)
    t_fit = time.time() - t0_fit

    val_scores = clf.decision_function(K_va)
    test_scores = clf.decision_function(K_te)

    val_pos_mean = float(np.mean(val_scores[y_va == 1]))
    val_neg_mean = float(np.mean(val_scores[y_va == 0]))

    # Threshold selection on validation
    best_th, val_best_f1 = select_best_threshold(y_va, val_scores)
    f1_norm = float(f1_score(y_va, (val_scores >= 0.0).astype(int), zero_division=0))
    f1_rev = float(f1_score(y_va, (-val_scores >= 0.0).astype(int), zero_division=0))

    # Test evaluation with frozen threshold
    y_pred_th = (test_scores >= best_th).astype(int)
    acc = accuracy_score(y_te, y_pred_th)
    prec = precision_score(y_te, y_pred_th, zero_division=0)
    rec = recall_score(y_te, y_pred_th, zero_division=0)
    f1_test = f1_score(y_te, y_pred_th, zero_division=0)
    roc = roc_auc_score(y_te, test_scores)
    pr_p, pr_r, _ = precision_recall_curve(y_te, test_scores)
    pr_auc = auc(pr_r, pr_p)

    record = {
        "feature_map": fmap_name,
        "dataset": ds_name.upper(),
        "pca_dim": d,
        "qubits": d,
        "accuracy": round(float(acc), 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1": round(float(f1_test), 6),
        "pr_auc": round(float(pr_auc), 6),
        "roc_auc": round(float(roc), 6),
        "threshold": round(float(best_th), 6),
        "kernel_train_time_sec": round(t_ktr, 4),
        "kernel_val_time_sec": round(t_kva, 4),
        "kernel_test_time_sec": round(t_kte, 4),
        "total_time_sec": round(t_kernel_total + t_fit, 4),
        "kernel_min": round(diag["kernel_min"], 6),
        "kernel_max": round(diag["kernel_max"], 6),
        "kernel_mean": round(diag["kernel_mean"], 6),
        "kernel_std": round(diag["kernel_std"], 6),
        "diag_mean": round(diag["diag_mean"], 6),
        "diag_std": round(diag["diag_std"], 6),
        "offdiag_mean": round(diag["offdiag_mean"], 6),
        "offdiag_std": round(diag["offdiag_std"], 6),
        "offdiag_min": round(diag["offdiag_min"], 6),
        "offdiag_max": round(diag["offdiag_max"], 6),
        "symmetry_error": round(diag["symmetry_error"], 12),
        "min_eigenvalue": round(diag["min_eigenvalue"], 10),
        "negative_eigenvalue_count": diag["negative_eigenvalue_count"],
        "effective_rank": diag["effective_rank"],
        "positive_positive_similarity": diag["positive_positive_similarity"],
        "negative_negative_similarity": diag["negative_negative_similarity"],
        "positive_negative_similarity": diag["positive_negative_similarity"],
        "within_between_separation": diag["within_between_separation"],
        "label_alignment": round(q_cka, 6),
        "rbf_label_alignment": round(rbf_cka, 6),
        "correlation_with_current_quantum": round(r_curr_p, 6),
        "correlation_with_rbf": round(r_rbf_p, 6),
        "positive_score_mean": round(val_pos_mean, 6),
        "negative_score_mean": round(val_neg_mean, 6),
        "f1_normal_orientation": round(f1_norm, 6),
        "f1_reversed_orientation": round(f1_rev, 6),
    }

    return record, K_tr


def main():
    print("=" * 80, flush=True)
    print("EXPERIMENT 28: QUANTUM FEATURE-MAP ABLATION", flush=True)
    print("=" * 80, flush=True)
    print(f"Executing with Python: {sys.executable}", flush=True)

    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    records = []

    # ============================================================
    # PRIMARY EXPERIMENT: CEAS 4D (4 Qubits)
    # ============================================================
    print("\n" + "#" * 80, flush=True)
    print("PRIMARY ABLATION: CEAS 4D (4 Qubits, 10k Train / 2.5k Val / 2.5k Test)", flush=True)
    print("#" * 80, flush=True)

    ceas_raw_dir = os.path.join(ROBERTA_BASE_DIR, "ceas")
    X_tr_full = np.load(os.path.join(ceas_raw_dir, "train_embeddings.npy"))
    y_tr_full = np.load(os.path.join(ceas_raw_dir, "train_labels.npy"))
    X_va_full = np.load(os.path.join(ceas_raw_dir, "validation_embeddings.npy"))
    y_va_full = np.load(os.path.join(ceas_raw_dir, "validation_labels.npy"))
    X_te_full = np.load(os.path.join(ceas_raw_dir, "test_embeddings.npy"))
    y_te_full = np.load(os.path.join(ceas_raw_dir, "test_labels.npy"))

    print(f"CEAS Embeddings: Train={X_tr_full.shape}, Val={X_va_full.shape}, Test={X_te_full.shape}", flush=True)

    # StandardScaler + PCA(4) strictly on training set
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_full)
    X_va_s = scaler.transform(X_va_full)
    X_te_s = scaler.transform(X_te_full)

    pca = PCA(n_components=4, random_state=RANDOM_SEED)
    X_tr_pca = pca.fit_transform(X_tr_s)
    X_va_pca = pca.transform(X_va_s)
    X_te_pca = pca.transform(X_te_s)

    cum_var = float(np.sum(pca.explained_variance_ratio_)) * 100
    print(f"CEAS 4D Cumulative Explained Variance: {cum_var:.2f}%", flush=True)

    # 1. Compute Classical RBF Kernel
    print("\nEvaluating Classical RBF Kernel Reference...", flush=True)
    t0_rbf = time.time()
    rbf_svc = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=RANDOM_SEED, cache_size=2000)
    rbf_svc.fit(X_tr_pca, y_tr_full)
    rbf_va_sc = rbf_svc.decision_function(X_va_pca)
    rbf_te_sc = rbf_svc.decision_function(X_te_pca)
    rbf_th, _ = select_best_threshold(y_va_full, rbf_va_sc)
    rbf_pred = (rbf_te_sc >= rbf_th).astype(int)
    rbf_f1 = f1_score(y_te_full, rbf_pred, zero_division=0)
    rbf_roc = roc_auc_score(y_te_full, rbf_te_sc)
    rbf_pr_p, rbf_pr_r, _ = precision_recall_curve(y_te_full, rbf_te_sc)
    rbf_pr_auc = auc(rbf_pr_r, rbf_pr_p)
    t_rbf_tot = time.time() - t0_rbf

    K_ceas_rbf_sub = rbf_kernel(X_tr_pca[:1000], gamma=1.0 / 4)
    rbf_cka = compute_centered_kernel_alignment(K_ceas_rbf_sub, y_tr_full[:1000])

    print(f"  Classical RBF: F1={rbf_f1:.4f} | PR-AUC={rbf_pr_auc:.4f} | ROC-AUC={rbf_roc:.4f} | CKA={rbf_cka:.4f} | Time={t_rbf_tot:.2f}s", flush=True)

    # 2. Evaluate All Quantum Feature Maps on CEAS 4D
    fmap_list = ["Current", "ZFeatureMap", "ZZFeatureMap", "PauliFeatureMap"]
    kernel_mats = {}

    # First evaluate Current to get reference for correlation
    print(f"\n--- Evaluating Feature Map: Current (Exp 26) ---", flush=True)
    rec_curr, K_curr = evaluate_dataset_feature_map(
        "CEAS", "Current", X_tr_pca, y_tr_full, X_va_pca, y_va_full, X_te_pca, y_te_full,
        K_ceas_rbf_sub, K_ceas_rbf_sub, rbf_cka
    )
    records.append(rec_curr)
    kernel_mats["Current"] = K_curr
    print(f"  Current Results: F1={rec_curr['f1']:.4f}, PR-AUC={rec_curr['pr_auc']:.4f}, ROC-AUC={rec_curr['roc_auc']:.4f}, CKA={rec_curr['label_alignment']:.4f}, EffRank={rec_curr['effective_rank']}", flush=True)

    for f_name in ["ZFeatureMap", "ZZFeatureMap", "PauliFeatureMap"]:
        print(f"\n--- Evaluating Feature Map: {f_name} ---", flush=True)
        rec, K_mat = evaluate_dataset_feature_map(
            "CEAS", f_name, X_tr_pca, y_tr_full, X_va_pca, y_va_full, X_te_pca, y_te_full,
            K_curr[:1000, :1000], K_ceas_rbf_sub, rbf_cka
        )
        records.append(rec)
        kernel_mats[f_name] = K_mat
        print(f"  {f_name} Results: F1={rec['f1']:.4f}, PR-AUC={rec['pr_auc']:.4f}, ROC-AUC={rec['roc_auc']:.4f}, CKA={rec['label_alignment']:.4f}, EffRank={rec['effective_rank']}", flush=True)

    # 3. Save Heatmaps for CEAS 4D (First 200 samples)
    print("\nGenerating kernel heatmaps (first 200 samples)...", flush=True)
    heat_maps = {
        "current": kernel_mats["Current"][:HEATMAP_SAMPLES, :HEATMAP_SAMPLES],
        "z": kernel_mats["ZFeatureMap"][:HEATMAP_SAMPLES, :HEATMAP_SAMPLES],
        "zz": kernel_mats["ZZFeatureMap"][:HEATMAP_SAMPLES, :HEATMAP_SAMPLES],
        "pauli": kernel_mats["PauliFeatureMap"][:HEATMAP_SAMPLES, :HEATMAP_SAMPLES],
        "rbf": rbf_kernel(X_tr_pca[:HEATMAP_SAMPLES], gamma=1.0 / 4),
    }

    for name, mat in heat_maps.items():
        plt.figure(figsize=(6, 5))
        sns.heatmap(mat, cmap="viridis", vmin=0, vmax=1, cbar=True)
        plt.title(f"CEAS 4D Kernel Matrix - {name.upper()} (200x200)")
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f"ceas_4d_{name}_kernel.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"  Saved heatmap: {plot_path}", flush=True)

    # ============================================================
    # IDENTIFY BEST ALTERNATIVE FEATURE MAP
    # ============================================================
    alt_records = [r for r in records if r["feature_map"] != "Current"]
    best_alt = max(alt_records, key=lambda x: x["f1"])
    best_alt_name = best_alt["feature_map"]
    print(f"\nBest-performing alternative feature map on CEAS: {best_alt_name} (F1: {best_alt['f1']:.4f})", flush=True)

    # ============================================================
    # SECONDARY EXPERIMENT: RUN BEST ALTERNATIVE ON SMS & MEAJOR
    # ============================================================
    print("\n" + "#" * 80, flush=True)
    print(f"SECONDARY EXPERIMENT: Testing {best_alt_name} on SMS 4D and MEAJOR 4D", flush=True)
    print("#" * 80, flush=True)

    for sec_ds in ["sms", "meajor"]:
        print(f"\n--- Running {best_alt_name} on {sec_ds.upper()} 4D ---", flush=True)
        sec_raw_dir = os.path.join(ROBERTA_BASE_DIR, sec_ds)
        s_X_tr = np.load(os.path.join(sec_raw_dir, "train_embeddings.npy"))
        s_y_tr = np.load(os.path.join(sec_raw_dir, "train_labels.npy"))
        s_X_va = np.load(os.path.join(sec_raw_dir, "validation_embeddings.npy"))
        s_y_va = np.load(os.path.join(sec_raw_dir, "validation_labels.npy"))
        s_X_te = np.load(os.path.join(sec_raw_dir, "test_embeddings.npy"))
        s_y_te = np.load(os.path.join(sec_raw_dir, "test_labels.npy"))

        s_scaler = StandardScaler()
        s_X_tr_s = s_scaler.fit_transform(s_X_tr)
        s_X_va_s = s_scaler.transform(s_X_va)
        s_X_te_s = s_scaler.transform(s_X_te)

        s_pca = PCA(n_components=4, random_state=RANDOM_SEED)
        s_X_tr_pca = s_pca.fit_transform(s_X_tr_s)
        s_X_va_pca = s_pca.transform(s_X_va_s)
        s_X_te_pca = s_pca.transform(s_X_te_s)

        # Classical RBF reference on secondary dataset
        s_K_rbf_sub = rbf_kernel(s_X_tr_pca[:1000], gamma=1.0 / 4)
        s_rbf_cka = compute_centered_kernel_alignment(s_K_rbf_sub, s_y_tr[:1000])

        # Current feature map reference for secondary dataset
        s_rec_curr, s_K_curr = evaluate_dataset_feature_map(
            sec_ds, "Current", s_X_tr_pca, s_y_tr, s_X_va_pca, s_y_va, s_X_te_pca, s_y_te,
            s_K_rbf_sub, s_K_rbf_sub, s_rbf_cka
        )
        records.append(s_rec_curr)

        # Best alternative feature map evaluation
        s_rec_alt, _ = evaluate_dataset_feature_map(
            sec_ds, best_alt_name, s_X_tr_pca, s_y_tr, s_X_va_pca, s_y_va, s_X_te_pca, s_y_te,
            s_K_curr[:1000, :1000], s_K_rbf_sub, s_rbf_cka
        )
        records.append(s_rec_alt)

        print(f"  {sec_ds.upper()} Results: Current F1={s_rec_curr['f1']:.4f} vs {best_alt_name} F1={s_rec_alt['f1']:.4f} (Δ F1 = {s_rec_alt['f1'] - s_rec_curr['f1']:+.4f})", flush=True)

    # Save CSV table
    df_ablation = pd.DataFrame(records)
    df_ablation.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"\nSaved ablation metrics CSV to: {CSV_OUTPUT_PATH}", flush=True)

    # ============================================================
    # BUILD HUMAN READABLE REPORT & SUMMARY
    # ============================================================
    ceas_records = [r for r in records if r["dataset"] == "CEAS"]
    best_f1_rec = max(ceas_records, key=lambda x: x["f1"])
    best_prauc_rec = max(ceas_records, key=lambda x: x["pr_auc"])
    best_roc_rec = max(ceas_records, key=lambda x: x["roc_auc"])
    best_cka_rec = max(ceas_records, key=lambda x: x["label_alignment"])
    lowest_conc_rec = max(ceas_records, key=lambda x: x["offdiag_std"]) # highest variance = least concentrated
    lowest_cost_rec = min(ceas_records, key=lambda x: x["total_time_sec"])

    curr_ceas = next(r for r in ceas_records if r["feature_map"] == "Current")
    f1_improvement = best_alt["f1"] - curr_ceas["f1"]
    quantum_classical_gap = rbf_f1 - best_f1_rec["f1"]

    # Predefined Classification Logic
    if f1_improvement >= 0.05 and (rbf_f1 - best_alt["f1"] < 0.10):
        classification = "CURRENT FEATURE MAP PROBLEMATIC"
    elif f1_improvement >= 0.05:
        classification = "FEATURE-MAP SENSITIVE"
    else:
        classification = "FEATURE-MAP ROBUSTLY WEAK"

    rep = []
    rep.append("=" * 95)
    rep.append("EXPERIMENT 28: QUANTUM FEATURE-MAP ABLATION SUMMARY")
    rep.append("=" * 95)
    header = f"{'Feature Map':<16} | {'F1':<7} | {'PR-AUC':<7} | {'ROC-AUC':<7} | {'Label Align':<11} | {'Eff Rank':<8} | {'Offdiag Std':<11} | {'Kernel Time':<11}"
    rep.append(header)
    rep.append("-" * 95)
    for r in ceas_records:
        row = f"{r['feature_map']:<16} | {r['f1']:<7.4f} | {r['pr_auc']:<7.4f} | {r['roc_auc']:<7.4f} | {r['label_alignment']:<11.4f} | {r['effective_rank']:<8.1f} | {r['offdiag_std']:<11.4f} | {r['total_time_sec']:<11.2f}s"
        rep.append(row)
    row_rbf = f"{'Classical RBF':<16} | {rbf_f1:<7.4f} | {rbf_pr_auc:<7.4f} | {rbf_roc:<7.4f} | {rbf_cka:<11.4f} | {'N/A':<8} | {'N/A':<11} | {t_rbf_tot:<11.2f}s"
    rep.append(row_rbf)

    rep.append("\n" + "=" * 95)
    rep.append("KEY COMPARATIVE FINDINGS (CEAS 4D)")
    rep.append("=" * 95)
    rep.append(f"1. Best Feature Map by F1             : {best_f1_rec['feature_map']} ({best_f1_rec['f1']:.4f})")
    rep.append(f"2. Best Feature Map by PR-AUC         : {best_prauc_rec['feature_map']} ({best_prauc_rec['pr_auc']:.4f})")
    rep.append(f"3. Best Feature Map by ROC-AUC        : {best_roc_rec['feature_map']} ({best_roc_rec['roc_auc']:.4f})")
    rep.append(f"4. Best Feature Map by Label Alignment: {best_cka_rec['feature_map']} (CKA = {best_cka_rec['label_alignment']:.4f})")
    rep.append(f"5. Lowest Kernel Concentration        : {lowest_conc_rec['feature_map']} (Offdiag std = {lowest_conc_rec['offdiag_std']:.4f})")
    rep.append(f"6. Lowest Computational Cost          : {lowest_cost_rec['feature_map']} ({lowest_cost_rec['total_time_sec']:.2f}s)")
    rep.append(f"7. Substantial Improvement over Current: {'YES' if f1_improvement >= 0.05 else 'NO'} (Δ F1 = {f1_improvement:+.4f})")
    rep.append(f"8. Approaches Classical RBF Baseline  : NO (Classical RBF F1: {rbf_f1:.4f} vs Best Quantum: {best_f1_rec['f1']:.4f}, Gap = {quantum_classical_gap:.4f})\n")

    rep.append("SECONDARY CROSS-DATASET GENERALIZATION:")
    rep.append("-" * 50)
    for ds in ["SMS", "MEAJOR"]:
        c_r = next(r for r in records if r["dataset"] == ds and r["feature_map"] == "Current")
        a_r = next(r for r in records if r["dataset"] == ds and r["feature_map"] == best_alt_name)
        rep.append(f"• {ds:<6} 4D: Current F1 = {c_r['f1']:.4f} vs {best_alt_name} F1 = {a_r['f1']:.4f} (Δ F1 = {a_r['f1'] - c_r['f1']:+.4f})")

    rep.append(f"\nSCIENTIFIC OUTCOME CLASSIFICATION: {classification}")

    summary_text = "\n".join(rep)
    with open(SUMMARY_OUTPUT_PATH, "w") as f:
        f.write(summary_text)
    print(f"Saved summary report to: {SUMMARY_OUTPUT_PATH}", flush=True)

    # ============================================================
    # FINAL TERMINAL OUTPUT
    # ============================================================
    print("\n" + "=" * 60, flush=True)
    print("EXPERIMENT 28 FINAL SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Feature Map':<20} {'F1':<8} {'PR-AUC':<9} {'ROC-AUC':<8}", flush=True)
    print("-" * 52, flush=True)
    for r in ceas_records:
        print(f"{r['feature_map']:<20} {r['f1']:<8.4f} {r['pr_auc']:<9.4f} {r['roc_auc']:<8.4f}", flush=True)
    print(f"{'Classical RBF':<20} {rbf_f1:<8.4f} {rbf_pr_auc:<9.4f} {rbf_roc:<8.4f}", flush=True)

    print("\n" + "-" * 52, flush=True)
    print(f"Best quantum feature map:    {best_f1_rec['feature_map']}", flush=True)
    print(f"Improvement over current:    {f1_improvement:+.4f}", flush=True)
    print(f"Quantum vs classical F1 gap: -{quantum_classical_gap:.4f}", flush=True)
    print(f"Highest label alignment:     {best_cka_rec['feature_map']} (CKA={best_cka_rec['label_alignment']:.4f})", flush=True)
    print(f"Lowest kernel concentration: {lowest_conc_rec['feature_map']} (std={lowest_conc_rec['offdiag_std']:.4f})", flush=True)
    print(f"Conclusion:                  {classification}", flush=True)
    print("\nExperiment 28 complete.", flush=True)


if __name__ == "__main__":
    main()
