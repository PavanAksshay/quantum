#!/usr/bin/env python3
"""
Experiment 26: Matched Classical RBF vs Quantum Kernel on RoBERTa Features
==========================================================================
Conducts a strictly matched, head-to-head empirical evaluation between:
  1. Classical RBF Support Vector Machine
  2. Quantum Kernel Support Vector Machine (ZZFeatureMap)

Both models receive EXACTLY identical RoBERTa PCA features:
  - StandardScaler fitted ONLY on training embeddings.
  - PCA fitted ONLY on training embeddings for d in [2, 4, 6, 8].
  - Exactly matched qubit mapping: d PCA components -> d qubits.
  - Strict leakage prevention: threshold selected on Validation, evaluated on Test.
  - Comprehensive kernel matrix diagnostics (symmetry, diagonal, PSD spectrum).

Author: Quantum Phishing & Scam Detection Project
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigvalsh
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
PCA_DIMS = [2, 4, 6, 8]
DATASETS = ["sms", "ceas", "meajor"]

ROBERTA_BASE_DIR = "results/roberta_multidataset"
QUANTUM_MATCHED_DIR = "results/quantum_matched"
METRICS_DIR = "results/metrics"

CSV_OUTPUT_PATH = os.path.join(METRICS_DIR, "experiment26_matched_kernel.csv")
SUMMARY_OUTPUT_PATH = os.path.join(METRICS_DIR, "experiment26_summary.txt")


def simulate_zz_feature_map_torch(X: torch.Tensor, n_qubits: int, reps: int = 2) -> torch.Tensor:
    """
    Exact, fully-vectorized batched quantum statevector simulator for ZZFeatureMap in PyTorch.
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

    for rep in range(reps):
        # 1. Hadamard on all qubits
        for q in range(n_qubits):
            state = apply_single_qubit_gate(state, H, q)

        # 2. Single-qubit RZ(2 * x_q)
        for q in range(n_qubits):
            x_q = X[:, q]
            phase_0 = torch.exp(-1j * x_q)
            phase_1 = torch.exp(1j * x_q)
            shape = [B] + [1] * n_qubits
            shape[q + 1] = 2
            rz_diag = torch.stack([phase_0, phase_1], dim=-1).reshape(shape)
            state = state * rz_diag

        # 3. Entangling ZZ interactions: RZZ(2 * x_i * x_j)
        if n_qubits > 1:
            for i in range(n_qubits):
                j = (i + 1) % n_qubits
                x_ij = X[:, i] * X[:, j]
                p_even = torch.exp(-1j * x_ij)
                p_odd = torch.exp(1j * x_ij)

                rzz_diag = torch.zeros([B, 2, 2], dtype=dtype, device=device)
                rzz_diag[:, 0, 0] = p_even
                rzz_diag[:, 0, 1] = p_odd
                rzz_diag[:, 1, 0] = p_odd
                rzz_diag[:, 1, 1] = p_even

                shape = [B] + [1] * n_qubits
                shape[i + 1] = 2
                shape[j + 1] = 2
                if i < j:
                    rzz_tensor = rzz_diag.reshape(shape)
                else:
                    rzz_tensor = rzz_diag.transpose(1, 2).reshape(shape)

                state = state * rzz_tensor

    return state.reshape(B, 2 ** n_qubits)


def compute_quantum_gram_matrix(states_1: torch.Tensor, states_2: torch.Tensor) -> np.ndarray:
    """
    Computes fidelity Gram matrix: K_ij = |<psi(x_i) | psi(x_j)>|^2
    """
    M = torch.matmul(states_1, states_2.conj().T)
    K = torch.abs(M) ** 2
    return K.cpu().numpy().astype(np.float64)


def select_best_threshold(y_val: np.ndarray, val_scores: np.ndarray, num_steps: int = 200) -> tuple:
    """
    Searches for threshold on validation decision scores that maximizes Validation F1 score.
    """
    thresholds = np.linspace(val_scores.min(), val_scores.max(), num_steps)
    best_f1 = -1.0
    best_threshold = 0.0

    for th in thresholds:
        pred = (val_scores >= th).astype(int)
        score = f1_score(y_val, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = th

    return best_threshold, best_f1


def evaluate_model_metrics(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
) -> dict:
    """
    Computes all standard classification metrics given decision scores and frozen threshold.
    """
    y_pred = (test_scores >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, test_scores)
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, test_scores)
    pr_auc = auc(pr_rec, pr_prec)

    return {
        "accuracy": round(float(acc), 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1": round(float(f1), 6),
        "pr_auc": round(float(pr_auc), 6),
        "roc_auc": round(float(roc), 6),
    }


def main():
    print("=" * 80, flush=True)
    print("EXPERIMENT 26: MATCHED CLASSICAL RBF VS QUANTUM KERNEL ON ROBERTA FEATURES", flush=True)
    print("=" * 80, flush=True)

    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(QUANTUM_MATCHED_DIR, exist_ok=True)

    results_records = []
    kernel_diagnostics_log = []

    for ds_name in DATASETS:
        ds_upper = ds_name.upper()
        ds_raw_dir = os.path.join(ROBERTA_BASE_DIR, ds_name)
        ds_save_dir = os.path.join(QUANTUM_MATCHED_DIR, ds_name)
        os.makedirs(ds_save_dir, exist_ok=True)

        print("\n" + "#" * 80, flush=True)
        print(f"DATASET: {ds_upper}", flush=True)
        print("#" * 80, flush=True)

        # 1. Load RoBERTa 768-d embeddings and labels
        X_tr_raw = np.load(os.path.join(ds_raw_dir, "train_embeddings.npy"))
        y_tr = np.load(os.path.join(ds_raw_dir, "train_labels.npy"))
        X_va_raw = np.load(os.path.join(ds_raw_dir, "validation_embeddings.npy"))
        y_va = np.load(os.path.join(ds_raw_dir, "validation_labels.npy"))
        X_te_raw = np.load(os.path.join(ds_raw_dir, "test_embeddings.npy"))
        y_te = np.load(os.path.join(ds_raw_dir, "test_labels.npy"))

        n_train, n_val, n_test = len(X_tr_raw), len(X_va_raw), len(X_te_raw)
        tr_pos_pct = np.mean(y_tr) * 100
        va_pos_pct = np.mean(y_va) * 100
        te_pos_pct = np.mean(y_te) * 100

        print(f"Samples: Train={n_train} ({tr_pos_pct:.2f}% pos), Val={n_val} ({va_pos_pct:.2f}% pos), Test={n_test} ({te_pos_pct:.2f}% pos)", flush=True)

        # 2. Fit StandardScaler strictly on training embeddings
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw)
        X_va_scaled = scaler.transform(X_va_raw)
        X_te_scaled = scaler.transform(X_te_raw)

        # 3. Iterate over PCA dimensions
        for d in PCA_DIMS:
            print("\n" + "-" * 70, flush=True)
            print(f"[{ds_upper}] PCA Dimension: {d} | Qubits: {d}", flush=True)
            print("-" * 70, flush=True)

            # Fit PCA strictly on training data
            pca = PCA(n_components=d, random_state=RANDOM_SEED)
            X_tr_pca = pca.fit_transform(X_tr_scaled)
            X_va_pca = pca.transform(X_va_scaled)
            X_te_pca = pca.transform(X_te_scaled)

            # Save PCA features for matched persistence
            np.save(os.path.join(ds_save_dir, f"train_pca_{d}.npy"), X_tr_pca)
            np.save(os.path.join(ds_save_dir, f"val_pca_{d}.npy"), X_va_pca)
            np.save(os.path.join(ds_save_dir, f"test_pca_{d}.npy"), X_te_pca)

            cum_var = float(np.sum(pca.explained_variance_ratio_)) * 100
            print(f"PCA Cumulative Explained Variance: {cum_var:.2f}%", flush=True)

            # ========================================================
            # MODEL A: CLASSICAL RBF SVM
            # ========================================================
            print(f"\n--- Training Classical RBF SVM (d={d}) ---", flush=True)
            rbf_svc = SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                random_state=RANDOM_SEED,
                cache_size=2000,
            )

            t0_rbf_tr = time.time()
            rbf_svc.fit(X_tr_pca, y_tr)
            t_rbf_tr = time.time() - t0_rbf_tr

            t0_rbf_va = time.time()
            rbf_va_scores = rbf_svc.decision_function(X_va_pca)
            t_rbf_va = time.time() - t0_rbf_va

            t0_rbf_te = time.time()
            rbf_te_scores = rbf_svc.decision_function(X_te_pca)
            t_rbf_te = time.time() - t0_rbf_te

            rbf_thresh, rbf_val_f1 = select_best_threshold(y_va, rbf_va_scores)
            rbf_metrics = evaluate_model_metrics(y_te, rbf_te_scores, rbf_thresh)
            t_rbf_total = t_rbf_tr + t_rbf_va + t_rbf_te

            print(f"Classical RBF Results (Test): F1={rbf_metrics['f1']:.4f} | PR-AUC={rbf_metrics['pr_auc']:.4f} | ROC-AUC={rbf_metrics['roc_auc']:.4f} | ValThresh={rbf_thresh:.4f} | Time={t_rbf_total:.3f}s", flush=True)

            results_records.append({
                "dataset": ds_upper,
                "pca_dim": d,
                "qubits": d,
                "model": "Classical RBF",
                "train_samples": n_train,
                "val_samples": n_val,
                "test_samples": n_test,
                "accuracy": rbf_metrics["accuracy"],
                "precision": rbf_metrics["precision"],
                "recall": rbf_metrics["recall"],
                "f1": rbf_metrics["f1"],
                "pr_auc": rbf_metrics["pr_auc"],
                "roc_auc": rbf_metrics["roc_auc"],
                "threshold": round(float(rbf_thresh), 6),
                "train_time_sec": round(t_rbf_tr, 4),
                "val_inference_time_sec": round(t_rbf_va, 4),
                "test_inference_time_sec": round(t_rbf_te, 4),
                "kernel_train_time_sec": 0.0,
                "kernel_val_time_sec": 0.0,
                "kernel_test_time_sec": 0.0,
                "total_time_sec": round(t_rbf_total, 4),
            })

            # ========================================================
            # MODEL B: QUANTUM KERNEL SVM (ZZFeatureMap)
            # ========================================================
            print(f"\n--- Computing Quantum Kernel (d={d}, qubits={d}) ---", flush=True)
            X_tr_t = torch.tensor(X_tr_pca, dtype=torch.float64)
            X_va_t = torch.tensor(X_va_pca, dtype=torch.float64)
            X_te_t = torch.tensor(X_te_pca, dtype=torch.float64)

            # 1. Statevector Simulation
            t0_q_states = time.time()
            states_tr = simulate_zz_feature_map_torch(X_tr_t, d)
            states_va = simulate_zz_feature_map_torch(X_va_t, d)
            states_te = simulate_zz_feature_map_torch(X_te_t, d)
            t_q_states = time.time() - t0_q_states

            # 2. Kernel Gram Matrices
            t0_k_tr = time.time()
            K_tr_tr = compute_quantum_gram_matrix(states_tr, states_tr)
            t_k_tr = time.time() - t0_k_tr

            t0_k_va = time.time()
            K_va_tr = compute_quantum_gram_matrix(states_va, states_tr)
            t_k_va = time.time() - t0_k_va

            t0_k_te = time.time()
            K_te_tr = compute_quantum_gram_matrix(states_te, states_tr)
            t_k_te = time.time() - t0_k_te

            # 3. Kernel Matrix Quality & PSD Diagnostics
            k_min = float(np.min(K_tr_tr))
            k_max = float(np.max(K_tr_tr))
            k_mean = float(np.mean(K_tr_tr))
            diag_mean = float(np.mean(np.diag(K_tr_tr)))
            symm_err = float(np.max(np.abs(K_tr_tr - K_tr_tr.T)))

            # Eigenvalue spectrum check (subset if N is very large for fast logging)
            diag_sample_n = min(n_train, 2000)
            sample_sub = K_tr_tr[:diag_sample_n, :diag_sample_n]
            min_eig = float(np.min(eigvalsh(sample_sub)))

            diag_info = (
                f"[{ds_upper} d={d}] Kernel Diagnostics: min={k_min:.4f}, max={k_max:.4f}, mean={k_mean:.4f}, "
                f"diag_mean={diag_mean:.4f}, symm_err={symm_err:.2e}, min_eigval={min_eig:.4e}"
            )
            print(diag_info, flush=True)
            kernel_diagnostics_log.append(diag_info)

            # 4. Train Precomputed Quantum SVM
            q_svc = SVC(
                kernel="precomputed",
                class_weight="balanced",
                random_state=RANDOM_SEED,
                cache_size=2000,
            )

            t0_q_fit = time.time()
            q_svc.fit(K_tr_tr, y_tr)
            t_q_fit = time.time() - t0_q_fit

            t0_q_va_inf = time.time()
            q_va_scores = q_svc.decision_function(K_va_tr)
            t_q_va_inf = time.time() - t0_q_va_inf

            t0_q_te_inf = time.time()
            q_te_scores = q_svc.decision_function(K_te_tr)
            t_q_te_inf = time.time() - t0_q_te_inf

            q_thresh, q_val_f1 = select_best_threshold(y_va, q_va_scores)
            q_metrics = evaluate_model_metrics(y_te, q_te_scores, q_thresh)

            t_q_total = t_q_states + t_k_tr + t_k_va + t_k_te + t_q_fit + t_q_va_inf + t_q_te_inf
            print(f"Quantum Kernel Results (Test): F1={q_metrics['f1']:.4f} | PR-AUC={q_metrics['pr_auc']:.4f} | ROC-AUC={q_metrics['roc_auc']:.4f} | ValThresh={q_thresh:.4f} | TotalTime={t_q_total:.3f}s", flush=True)

            results_records.append({
                "dataset": ds_upper,
                "pca_dim": d,
                "qubits": d,
                "model": "Quantum Kernel",
                "train_samples": n_train,
                "val_samples": n_val,
                "test_samples": n_test,
                "accuracy": q_metrics["accuracy"],
                "precision": q_metrics["precision"],
                "recall": q_metrics["recall"],
                "f1": q_metrics["f1"],
                "pr_auc": q_metrics["pr_auc"],
                "roc_auc": q_metrics["roc_auc"],
                "threshold": round(float(q_thresh), 6),
                "train_time_sec": round(t_q_fit, 4),
                "val_inference_time_sec": round(t_q_va_inf, 4),
                "test_inference_time_sec": round(t_q_te_inf, 4),
                "kernel_train_time_sec": round(t_k_tr, 4),
                "kernel_val_time_sec": round(t_k_va, 4),
                "kernel_test_time_sec": round(t_k_te, 4),
                "total_time_sec": round(t_q_total, 4),
            })

    # Save CSV metrics
    df_results = pd.DataFrame(results_records)
    df_results.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"\nSaved matched benchmark metrics to: {CSV_OUTPUT_PATH}", flush=True)

    # ============================================================
    # COMPARATIVE SUMMARY TABLE & ANALYSIS
    # ============================================================
    summary_lines = []
    summary_lines.append("=" * 110)
    summary_lines.append("EXPERIMENT 26: MATCHED CLASSICAL RBF VS QUANTUM KERNEL COMPARATIVE SUMMARY")
    summary_lines.append("=" * 110)
    header = f"{'Dataset':<8} | {'Dim':<4} | {'Classical F1':<12} | {'Quantum F1':<10} | {'Δ F1 (Q-C)':<10} | {'Class PR-AUC':<12} | {'Quant PR-AUC':<12} | {'Class ROC':<9} | {'Quant ROC':<9} | {'Class(s)':<8} | {'Quant(s)':<8}"
    summary_lines.append(header)
    summary_lines.append("-" * 110)

    print("\n" + "\n".join(summary_lines), flush=True)

    analysis_data = {}
    for ds_name in DATASETS:
        ds_upper = ds_name.upper()
        analysis_data[ds_upper] = []

        for d in PCA_DIMS:
            c_row = df_results[(df_results["dataset"] == ds_upper) & (df_results["pca_dim"] == d) & (df_results["model"] == "Classical RBF")].iloc[0]
            q_row = df_results[(df_results["dataset"] == ds_upper) & (df_results["pca_dim"] == d) & (df_results["model"] == "Quantum Kernel")].iloc[0]

            delta_f1 = q_row["f1"] - c_row["f1"]
            delta_prauc = q_row["pr_auc"] - c_row["pr_auc"]
            delta_roc = q_row["roc_auc"] - c_row["roc_auc"]
            time_ratio = q_row["total_time_sec"] / max(c_row["total_time_sec"], 1e-4)

            row_str = (
                f"{ds_upper:<8} | {d:<4} | {c_row['f1']:<12.4f} | {q_row['f1']:<10.4f} | {delta_f1:>+9.4f}  | "
                f"{c_row['pr_auc']:<12.4f} | {q_row['pr_auc']:<12.4f} | {c_row['roc_auc']:<9.4f} | {q_row['roc_auc']:<9.4f} | "
                f"{c_row['total_time_sec']:<8.3f} | {q_row['total_time_sec']:<8.3f}"
            )
            print(row_str, flush=True)
            summary_lines.append(row_str)

            analysis_data[ds_upper].append({
                "dim": d,
                "c_f1": c_row["f1"],
                "q_f1": q_row["f1"],
                "delta_f1": delta_f1,
                "c_pr_auc": c_row["pr_auc"],
                "q_pr_auc": q_row["pr_auc"],
                "delta_pr_auc": delta_prauc,
                "c_roc_auc": c_row["roc_auc"],
                "q_roc_auc": q_row["roc_auc"],
                "delta_roc_auc": delta_roc,
                "c_time": c_row["total_time_sec"],
                "q_time": q_row["total_time_sec"],
                "time_ratio": time_ratio,
            })

    # Scientific Observations Section
    summary_lines.append("\n" + "=" * 80)
    summary_lines.append("SCIENTIFIC ANALYSIS & RIGOROUS EVALUATION")
    summary_lines.append("=" * 80)

    for ds_upper, records in analysis_data.items():
        best_q = max(records, key=lambda x: x["q_f1"])
        best_c = max(records, key=lambda x: x["c_f1"])
        summary_lines.append(f"\n[{ds_upper} Summary]")
        summary_lines.append(f"  • Best Quantum Configuration : d={best_q['dim']} (Qubits={best_q['dim']}) -> F1: {best_q['q_f1']:.4f}, PR-AUC: {best_q['q_pr_auc']:.4f}, ROC-AUC: {best_q['q_roc_auc']:.4f}")
        summary_lines.append(f"  • Best Classical Configuration: d={best_c['dim']} -> F1: {best_c['c_f1']:.4f}, PR-AUC: {best_c['c_pr_auc']:.4f}, ROC-AUC: {best_c['c_roc_auc']:.4f}")
        summary_lines.append(f"  • Head-to-Head Performance Gap: Best Q F1 ({best_q['q_f1']:.4f}) vs Best C F1 ({best_c['c_f1']:.4f}) => Δ F1 = {best_q['q_f1'] - best_c['c_f1']:+.4f}")
        avg_ratio = np.mean([r["time_ratio"] for r in records])
        summary_lines.append(f"  • Average Computational Cost Overhead (Quantum / Classical): {avg_ratio:.2f}x")

    summary_lines.append("\n[Dimension Scaling Effect]")
    for ds_upper, records in analysis_data.items():
        dim_f1s = ", ".join([f"d={r['dim']}: {r['q_f1']:.4f}" for r in records])
        summary_lines.append(f"  • {ds_upper:<8}: Quantum F1 progression ({dim_f1s})")

    summary_lines.append("\n[Kernel Diagnostics Check]")
    for diag in kernel_diagnostics_log:
        summary_lines.append(f"  • {diag}")

    summary_text = "\n".join(summary_lines)
    with open(SUMMARY_OUTPUT_PATH, "w") as f:
        f.write(summary_text)
    print(f"\nSaved summary report to: {SUMMARY_OUTPUT_PATH}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("EXPERIMENT 26 COMPLETE", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
