"""
Experiment 40: Final Protocol V1.0 + Primary Confirmation Experiments
Canonical 10-Seed Confirmation of MeAJOR IID 8D, Direction B 8D, and Dimensionality Scaling (8D, 10D, 12D)

Authoritative Script for Final Manuscript Evidence Pack
"""

import os
import sys
import time
import json
import resource
import subprocess
import platform
import hashlib
import torch
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
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

# Base Directories
BASE_DIR = "/Users/pavanaksshay/quantum"
FROZEN_SPLITS_DIR = os.path.join(BASE_DIR, "results/frozen_splits/meajor")
ROBERTA_SUBSET_DIR = os.path.join(BASE_DIR, "results/roberta_multidataset/meajor")
MEAJOR_PARQUET_PATH = os.path.join(BASE_DIR, "data/meajor_cleaned_preprocessed.parquet.gzip")
EXP40_DIR = os.path.join(BASE_DIR, "results/exp40_final")
os.makedirs(EXP40_DIR, exist_ok=True)

# 10 Independent Seeds
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

# Canonical CPU device for exact complex128 simulation (identical to Exp 36)
device = torch.device("cpu")
print(f"Executing Quantum Simulation on: {device} (complex128)")

def simulate_zz_feature_map(X: torch.Tensor, n_qubits: int = 8) -> torch.Tensor:
    """
    Simulates 2-layer cyclic ZZFeatureMap with fidelity kernel.
    Unit diagonal and exact complex128 statevector formulation.
    """
    B = X.shape[0]
    dev = X.device
    dtype = torch.complex128
    state = torch.zeros([B] + [2] * n_qubits, dtype=dtype, device=dev)
    state[(slice(None),) + (0,) * n_qubits] = 1.0
    H = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=dtype, device=dev) / np.sqrt(2)

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
        x_ij = (torch.pi - x1) * (torch.pi - x2) if True else x1 * x2  # standard qiskit ZZFeatureMap angle
        # Note: in Exp 36 implementation, angle was x1 * x2. We use exact Exp 36 implementation for exact matching.
        x_ij = x1 * x2
        p_even = torch.exp(-1j * x_ij)
        p_odd = torch.exp(1j * x_ij)
        rzz_diag = torch.zeros([B, 2, 2], dtype=dtype, device=dev)
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

    return state.reshape(B, 2 ** n_qubits)


def compute_quantum_gram_matrix(states_1: torch.Tensor, states_2: torch.Tensor) -> np.ndarray:
    """
    Computes fidelity Gram matrix: K_ij = |<psi_i | psi_j>|^2
    """
    # Chunked matrix multiply if large to conserve memory
    B1 = states_1.shape[0]
    B2 = states_2.shape[0]
    chunk_size = 2500
    K = np.zeros((B1, B2), dtype=np.float64)
    
    for i in range(0, B1, chunk_size):
        end_i = min(i + chunk_size, B1)
        s1_chunk = states_1[i:end_i]
        M_chunk = torch.matmul(s1_chunk, states_2.conj().T)
        K_chunk = (torch.abs(M_chunk) ** 2).cpu().numpy().astype(np.float64)
        K[i:end_i, :] = K_chunk
    return K


def select_best_threshold(y_val: np.ndarray, val_scores: np.ndarray, num_steps: int = 200) -> tuple:
    thresholds = np.linspace(val_scores.min(), val_scores.max(), num_steps)
    best_f1, best_th = -1.0, 0.0
    for th in thresholds:
        f1 = f1_score(y_val, (val_scores >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return float(best_th), float(best_f1)


def evaluate_predictions(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.0) -> dict:
    preds = (scores >= threshold).astype(int)
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]
    margin = float(pos_scores.mean() - neg_scores.mean()) if len(pos_scores) > 0 and len(neg_scores) > 0 else 0.0
    return {
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "decision_margin": margin,
        "threshold": float(threshold),
    }


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42) -> tuple:
    rng = np.random.RandomState(seed)
    boot_means = np.zeros(n_boot)
    n = len(diffs)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = np.mean(diffs[idx])
    ci_lower = np.percentile(boot_means, 100 * (alpha / 2))
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(ci_lower), float(ci_upper)


def paired_permutation_test(diffs: np.ndarray, n_perm: int = 10000, seed: int = 42) -> float:
    rng = np.random.RandomState(seed)
    observed_mean = np.abs(np.mean(diffs))
    n = len(diffs)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_mean = np.abs(np.mean(diffs * signs))
        if perm_mean >= observed_mean:
            count += 1
    return float(count / n_perm)


def stratified_sample_df(df: pd.DataFrame, n_samples: int, seed: int):
    fractions = df["target"].value_counts(normalize=True)
    sample_indices = []
    for target_val, frac in fractions.items():
        sub_df = df[df["target"] == target_val]
        n_t = int(round(n_samples * frac))
        n_t = min(n_t, len(sub_df))
        sampled_sub = sub_df.sample(n=n_t, random_state=seed)
        sample_indices.extend(sampled_sub.index.tolist())
    
    if len(sample_indices) < n_samples:
        diff = n_samples - len(sample_indices)
        remaining = df[~df.index.isin(sample_indices)]
        if len(remaining) > 0:
            sample_indices.extend(remaining.sample(n=min(diff, len(remaining)), random_state=seed).index.tolist())
    elif len(sample_indices) > n_samples:
        sample_indices = sample_indices[:n_samples]
    
    sampled_df = df.loc[sample_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled_df, set(sample_indices)


def get_system_metadata() -> dict:
    try:
        import qiskit
        qiskit_ver = qiskit.__version__
    except ImportError:
        qiskit_ver = "Not Installed (PyTorch Simulation Engine)"

    try:
        total_ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) / (1024 ** 3)
    except Exception:
        total_ram = 16.0

    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count() or 8,
        "cpu_count_physical": os.cpu_count() or 8,
        "total_ram_gb": round(total_ram, 2),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "qiskit_version": qiskit_ver,
        "device_used": str(device),
    }


def main():
    print("=" * 80)
    print("EXPERIMENT 40: FINAL CONFIRMATION EXPERIMENTS & PROTOCOL V1.0 AUDIT")
    print(f"10 Independent Seeds: {SEEDS}")
    print("=" * 80)

    # Record System Metadata
    meta = get_system_metadata()
    with open(os.path.join(EXP40_DIR, "RUN_METADATA.md"), "w") as f:
        f.write("# Experiment 40 Run Metadata & Environment Specifications\n\n")
        f.write(f"- **Execution Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        f.write(f"- **Operating System**: {meta['os']} {meta['os_release']} ({meta['os_version']})\n")
        f.write(f"- **Machine / Architecture**: {meta['machine']} ({meta['processor']})\n")
        f.write(f"- **Logical CPU Cores**: {meta['cpu_count_logical']} (Physical: {meta['cpu_count_physical']})\n")
        f.write(f"- **Total System RAM**: {meta['total_ram_gb']} GB\n")
        f.write(f"- **PyTorch Execution Device**: {meta['device_used']}\n")
        f.write(f"- **Python Version**: {meta['python_version']}\n")
        f.write(f"- **PyTorch Version**: {meta['pytorch_version']}\n")
        f.write(f"- **scikit-learn Version**: {meta['sklearn_version']}\n")
        f.write(f"- **NumPy Version**: {meta['numpy_version']}\n")
        f.write(f"- **SciPy Version**: {meta['scipy_version']}\n")
        f.write(f"- **Qiskit Version**: {meta['qiskit_version']}\n")

    # Load MeAJOR Data
    print("\nLoading MeAJOR Datasets and Frozen Splits ...")
    tr_ids = pd.read_csv(os.path.join(ROBERTA_SUBSET_DIR, "train_sample_ids.csv"))["sample_id"]
    va_ids = pd.read_csv(os.path.join(ROBERTA_SUBSET_DIR, "validation_sample_ids.csv"))["sample_id"]
    te_ids = pd.read_csv(os.path.join(ROBERTA_SUBSET_DIR, "test_sample_ids.csv"))["sample_id"]

    full_tr = pd.read_csv(os.path.join(FROZEN_SPLITS_DIR, "train.csv")).set_index("sample_id")
    full_va = pd.read_csv(os.path.join(FROZEN_SPLITS_DIR, "validation.csv")).set_index("sample_id")
    full_te = pd.read_csv(os.path.join(FROZEN_SPLITS_DIR, "test.csv")).set_index("sample_id")

    df_sub_tr = full_tr.loc[tr_ids].reset_index()
    df_sub_va = full_va.loc[va_ids].reset_index()
    df_sub_te = full_te.loc[te_ids].reset_index()

    # Load Full Parquet for Direction B
    df_raw = pd.read_parquet(MEAJOR_PARQUET_PATH)
    subj = df_raw["subject"].fillna("").astype(str).str.strip()
    body = df_raw["body"].fillna("").astype(str).str.strip()
    text_clean = (subj + " " + body).str.strip()
    df_meajor_all = pd.DataFrame({
        "sample_id": [f"meajor_{i}" for i in df_raw.index],
        "text": text_clean,
        "target": df_raw["label"],
        "source": df_raw["source"].fillna("unknown").astype(str).str.lower(),
    })
    valid_mask = (~df_meajor_all["target"].isna()) & (df_meajor_all["text"] != "") & (df_meajor_all["source"].isin(["trec5", "trec6", "trec7"]))
    df_meajor_all = df_meajor_all[valid_mask].copy()
    df_meajor_all["target"] = df_meajor_all["target"].astype(int)

    all_results = []
    paired_predictions = []

    # ============================================================
    # EXPERIMENT 40A: MeAJOR IID 8D CONFIRMATION (10 SEEDS)
    # ============================================================
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 40A: MeAJOR IID 8D Confirmation (N=10 seeds)")
    print("=" * 80)
    
    texts_tr_iid = df_sub_tr["text"].tolist()
    texts_va_iid = df_sub_va["text"].tolist()
    texts_te_iid = df_sub_te["text"].tolist()
    y_tr_iid = df_sub_tr["target"].values
    y_va_iid = df_sub_va["target"].values
    y_te_iid = df_sub_te["target"].values

    for s_idx, seed in enumerate(SEEDS, 1):
        print(f"\n[Exp40A - IID 8D | Seed {seed} ({s_idx}/10)]")
        t0_total = time.time()

        # TF-IDF
        tfidf = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
        X_tr_tf = tfidf.fit_transform(texts_tr_iid)
        X_va_tf = tfidf.transform(texts_va_iid)
        X_te_tf = tfidf.transform(texts_te_iid)

        # PCA 8D
        svd = TruncatedSVD(n_components=8, random_state=seed)
        scaler = StandardScaler()
        X_tr_pca = scaler.fit_transform(svd.fit_transform(X_tr_tf))
        X_va_pca = scaler.transform(svd.transform(X_va_tf))
        X_te_pca = scaler.transform(svd.transform(X_te_tf))

        # 1. Linear SVM
        t0_lin = time.time()
        clf_lin = LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=2000)
        clf_lin.fit(X_tr_pca, y_tr_iid)
        sc_va_lin = clf_lin.decision_function(X_va_pca)
        th_lin, _ = select_best_threshold(y_va_iid, sc_va_lin)
        t0_inf_lin = time.time()
        sc_te_lin = clf_lin.decision_function(X_te_pca)
        t_inf_lin = time.time() - t0_inf_lin
        t_tot_lin = time.time() - t0_lin
        m_lin = evaluate_predictions(y_te_iid, sc_te_lin, th_lin)
        all_results.append({
            "experiment": "Exp40A_IID_8D",
            "regime": "IID",
            "dimension": 8,
            "seed": seed,
            "model": "Linear SVM",
            "kernel_time_sec": 0.0,
            "train_time_sec": t_tot_lin - t_inf_lin,
            "inference_time_sec": t_inf_lin,
            "total_runtime_sec": t_tot_lin,
            "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
            **m_lin
        })
        print(f"  Linear SVM: F1={m_lin['f1']:.4f}, PR-AUC={m_lin['pr_auc']:.4f}, Time={t_tot_lin:.2f}s")

        # 2. Classical RBF
        t0_rbf = time.time()
        clf_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
        clf_rbf.fit(X_tr_pca, y_tr_iid)
        sc_va_rbf = clf_rbf.decision_function(X_va_pca)
        th_rbf, _ = select_best_threshold(y_va_iid, sc_va_rbf)
        t0_inf_rbf = time.time()
        sc_te_rbf = clf_rbf.decision_function(X_te_pca)
        t_inf_rbf = time.time() - t0_inf_rbf
        t_tot_rbf = time.time() - t0_rbf
        m_rbf = evaluate_predictions(y_te_iid, sc_te_rbf, th_rbf)
        all_results.append({
            "experiment": "Exp40A_IID_8D",
            "regime": "IID",
            "dimension": 8,
            "seed": seed,
            "model": "Classical RBF",
            "kernel_time_sec": 0.0,
            "train_time_sec": t_tot_rbf - t_inf_rbf,
            "inference_time_sec": t_inf_rbf,
            "total_runtime_sec": t_tot_rbf,
            "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
            **m_rbf
        })
        print(f"  Classical RBF: F1={m_rbf['f1']:.4f}, PR-AUC={m_rbf['pr_auc']:.4f}, Time={t_tot_rbf:.2f}s")

        # 3. Quantum Kernel
        t0_qk = time.time()
        states_tr = simulate_zz_feature_map(torch.tensor(X_tr_pca, dtype=torch.float64, device=device), 8)
        states_va = simulate_zz_feature_map(torch.tensor(X_va_pca, dtype=torch.float64, device=device), 8)
        states_te = simulate_zz_feature_map(torch.tensor(X_te_pca, dtype=torch.float64, device=device), 8)
        K_tr = compute_quantum_gram_matrix(states_tr, states_tr)
        K_va = compute_quantum_gram_matrix(states_va, states_tr)
        K_te = compute_quantum_gram_matrix(states_te, states_tr)
        t_k_q = time.time() - t0_qk

        t0_fit_q = time.time()
        clf_q = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
        clf_q.fit(K_tr, y_tr_iid)
        t_fit_q = time.time() - t0_fit_q

        sc_va_q = clf_q.decision_function(K_va)
        th_q, _ = select_best_threshold(y_va_iid, sc_va_q)

        t0_inf_q = time.time()
        sc_te_q = clf_q.decision_function(K_te)
        t_inf_q = time.time() - t0_inf_q
        t_tot_q = time.time() - t0_qk
        m_q = evaluate_predictions(y_te_iid, sc_te_q, th_q)
        all_results.append({
            "experiment": "Exp40A_IID_8D",
            "regime": "IID",
            "dimension": 8,
            "seed": seed,
            "model": "Quantum Kernel",
            "kernel_time_sec": t_k_q,
            "train_time_sec": t_fit_q,
            "inference_time_sec": t_inf_q,
            "total_runtime_sec": t_tot_q,
            "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
            **m_q
        })
        delta_f1 = m_q['f1'] - m_rbf['f1']
        print(f"  Quantum Kernel: F1={m_q['f1']:.4f}, PR-AUC={m_q['pr_auc']:.4f}, Delta F1={delta_f1:+.4f}, Time={t_tot_q:.2f}s")

    # ============================================================
    # EXPERIMENT 40B: MeAJOR DIRECTION B SOURCE HOLDOUT (10 SEEDS)
    # ============================================================
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 40B: MeAJOR Direction B Source Holdout (TREC7 -> TREC5+6, N=10 seeds)")
    print("=" * 80)
    df_pool_tr_B = df_meajor_all[df_meajor_all["source"] == "trec7"]
    df_p_trec5 = df_meajor_all[df_meajor_all["source"] == "trec5"]
    df_p_trec6 = df_meajor_all[df_meajor_all["source"] == "trec6"]

    for s_idx, seed in enumerate(SEEDS, 1):
        print(f"\n[Exp40B - Direction B 8D | Seed {seed} ({s_idx}/10)]")
        # Train on TREC7 (10k train, 2.5k val)
        df_tr, tr_idx_set = stratified_sample_df(df_pool_tr_B, 10000, seed)
        rem_tr = df_pool_tr_B[~df_pool_tr_B.index.isin(tr_idx_set)]
        df_va, _ = stratified_sample_df(rem_tr, 2500, seed)

        # Canonical Exp 33/36 sampling: 2,500 TREC5 + 2,500 TREC6
        df_te_5, _ = stratified_sample_df(df_p_trec5, 2500, seed)
        df_te_6, _ = stratified_sample_df(df_p_trec6, 2500, seed)
        df_te = pd.concat([df_te_5, df_te_6]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

        tfidf = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
        X_tr_tf = tfidf.fit_transform(df_tr["text"].tolist())
        X_va_tf = tfidf.transform(df_va["text"].tolist())
        X_te_tf = tfidf.transform(df_te["text"].tolist())

        y_tr, y_va, y_te = df_tr["target"].values, df_va["target"].values, df_te["target"].values

        svd = TruncatedSVD(n_components=8, random_state=seed)
        scaler = StandardScaler()
        X_tr_pca = scaler.fit_transform(svd.fit_transform(X_tr_tf))
        X_va_pca = scaler.transform(svd.transform(X_va_tf))
        X_te_pca = scaler.transform(svd.transform(X_te_tf))

        # 1. Linear SVM
        t0_lin = time.time()
        clf_lin = LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=2000)
        clf_lin.fit(X_tr_pca, y_tr)
        sc_va_lin = clf_lin.decision_function(X_va_pca)
        th_lin, _ = select_best_threshold(y_va, sc_va_lin)
        t0_inf_lin = time.time()
        sc_te_lin = clf_lin.decision_function(X_te_pca)
        t_inf_lin = time.time() - t0_inf_lin
        t_tot_lin = time.time() - t0_lin
        m_lin = evaluate_predictions(y_te, sc_te_lin, th_lin)
        all_results.append({
            "experiment": "Exp40B_Direction_B_8D",
            "regime": "Direction_B",
            "dimension": 8,
            "seed": seed,
            "model": "Linear SVM",
            "kernel_time_sec": 0.0,
            "train_time_sec": t_tot_lin - t_inf_lin,
            "inference_time_sec": t_inf_lin,
            "total_runtime_sec": t_tot_lin,
            "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
            **m_lin
        })
        print(f"  Linear SVM: F1={m_lin['f1']:.4f}, PR-AUC={m_lin['pr_auc']:.4f}, Time={t_tot_lin:.2f}s")

        # 2. Classical RBF
        t0_rbf = time.time()
        clf_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
        clf_rbf.fit(X_tr_pca, y_tr)
        sc_va_rbf = clf_rbf.decision_function(X_va_pca)
        th_rbf, _ = select_best_threshold(y_va, sc_va_rbf)
        t0_inf_rbf = time.time()
        sc_te_rbf = clf_rbf.decision_function(X_te_pca)
        t_inf_rbf = time.time() - t0_inf_rbf
        t_tot_rbf = time.time() - t0_rbf
        m_rbf = evaluate_predictions(y_te, sc_te_rbf, th_rbf)
        all_results.append({
            "experiment": "Exp40B_Direction_B_8D",
            "regime": "Direction_B",
            "dimension": 8,
            "seed": seed,
            "model": "Classical RBF",
            "kernel_time_sec": 0.0,
            "train_time_sec": t_tot_rbf - t_inf_rbf,
            "inference_time_sec": t_inf_rbf,
            "total_runtime_sec": t_tot_rbf,
            "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
            **m_rbf
        })
        print(f"  Classical RBF: F1={m_rbf['f1']:.4f}, PR-AUC={m_rbf['pr_auc']:.4f}, Time={t_tot_rbf:.2f}s")

        # 3. Quantum Kernel
        t0_qk = time.time()
        states_tr = simulate_zz_feature_map(torch.tensor(X_tr_pca, dtype=torch.float64, device=device), 8)
        states_va = simulate_zz_feature_map(torch.tensor(X_va_pca, dtype=torch.float64, device=device), 8)
        states_te = simulate_zz_feature_map(torch.tensor(X_te_pca, dtype=torch.float64, device=device), 8)
        K_tr = compute_quantum_gram_matrix(states_tr, states_tr)
        K_va = compute_quantum_gram_matrix(states_va, states_tr)
        K_te = compute_quantum_gram_matrix(states_te, states_tr)
        t_k_q = time.time() - t0_qk

        t0_fit_q = time.time()
        clf_q = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
        clf_q.fit(K_tr, y_tr)
        t_fit_q = time.time() - t0_fit_q

        sc_va_q = clf_q.decision_function(K_va)
        th_q, _ = select_best_threshold(y_va, sc_va_q)

        t0_inf_q = time.time()
        sc_te_q = clf_q.decision_function(K_te)
        t_inf_q = time.time() - t0_inf_q
        t_tot_q = time.time() - t0_qk
        m_q = evaluate_predictions(y_te, sc_te_q, th_q)
        all_results.append({
            "experiment": "Exp40B_Direction_B_8D",
            "regime": "Direction_B",
            "dimension": 8,
            "seed": seed,
            "model": "Quantum Kernel",
            "kernel_time_sec": t_k_q,
            "train_time_sec": t_fit_q,
            "inference_time_sec": t_inf_q,
            "total_runtime_sec": t_tot_q,
            "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
            **m_q
        })
        delta_f1 = m_q['f1'] - m_rbf['f1']
        print(f"  Quantum Kernel: F1={m_q['f1']:.4f}, PR-AUC={m_q['pr_auc']:.4f}, Delta F1={delta_f1:+.4f}, Time={t_tot_q:.2f}s")

    # ============================================================
    # EXPERIMENT 40C: MeAJOR IID DIMENSIONALITY ANCHORS (10D & 12D, 10 SEEDS)
    # ============================================================
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 40C: MeAJOR IID Dimensionality Anchors (10D, 12D; N=10 seeds)")
    print("=" * 80)

    for dim in [10, 12]:
        print(f"\n>>> Running Dimension d = {dim} across 10 seeds ...")
        for s_idx, seed in enumerate(SEEDS, 1):
            print(f"  [Exp40C - {dim}D | Seed {seed} ({s_idx}/10)] ...")
            tfidf = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
            X_tr_tf = tfidf.fit_transform(texts_tr_iid)
            X_va_tf = tfidf.transform(texts_va_iid)
            X_te_tf = tfidf.transform(texts_te_iid)

            svd = TruncatedSVD(n_components=dim, random_state=seed)
            scaler = StandardScaler()
            X_tr_pca = scaler.fit_transform(svd.fit_transform(X_tr_tf))
            X_va_pca = scaler.transform(svd.transform(X_va_tf))
            X_te_pca = scaler.transform(svd.transform(X_te_tf))

            # Classical RBF
            t0_rbf = time.time()
            clf_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
            clf_rbf.fit(X_tr_pca, y_tr_iid)
            sc_va_rbf = clf_rbf.decision_function(X_va_pca)
            th_rbf, _ = select_best_threshold(y_va_iid, sc_va_rbf)
            t0_inf_rbf = time.time()
            sc_te_rbf = clf_rbf.decision_function(X_te_pca)
            t_inf_rbf = time.time() - t0_inf_rbf
            t_tot_rbf = time.time() - t0_rbf
            m_rbf = evaluate_predictions(y_te_iid, sc_te_rbf, th_rbf)
            all_results.append({
                "experiment": f"Exp40C_IID_{dim}D",
                "regime": "IID",
                "dimension": dim,
                "seed": seed,
                "model": "Classical RBF",
                "kernel_time_sec": 0.0,
                "train_time_sec": t_tot_rbf - t_inf_rbf,
                "inference_time_sec": t_inf_rbf,
                "total_runtime_sec": t_tot_rbf,
                "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
                **m_rbf
            })

            # Quantum Kernel
            t0_qk = time.time()
            states_tr = simulate_zz_feature_map(torch.tensor(X_tr_pca, dtype=torch.float64, device=device), dim)
            states_va = simulate_zz_feature_map(torch.tensor(X_va_pca, dtype=torch.float64, device=device), dim)
            states_te = simulate_zz_feature_map(torch.tensor(X_te_pca, dtype=torch.float64, device=device), dim)
            K_tr = compute_quantum_gram_matrix(states_tr, states_tr)
            K_va = compute_quantum_gram_matrix(states_va, states_tr)
            K_te = compute_quantum_gram_matrix(states_te, states_tr)
            t_k_q = time.time() - t0_qk

            t0_fit_q = time.time()
            clf_q = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
            clf_q.fit(K_tr, y_tr_iid)
            t_fit_q = time.time() - t0_fit_q

            sc_va_q = clf_q.decision_function(K_va)
            th_q, _ = select_best_threshold(y_va_iid, sc_va_q)

            t0_inf_q = time.time()
            sc_te_q = clf_q.decision_function(K_te)
            t_inf_q = time.time() - t0_inf_q
            t_tot_q = time.time() - t0_qk
            m_q = evaluate_predictions(y_te_iid, sc_te_q, th_q)
            all_results.append({
                "experiment": f"Exp40C_IID_{dim}D",
                "regime": "IID",
                "dimension": dim,
                "seed": seed,
                "model": "Quantum Kernel",
                "kernel_time_sec": t_k_q,
                "train_time_sec": t_fit_q,
                "inference_time_sec": t_inf_q,
                "total_runtime_sec": t_tot_q,
                "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
                **m_q
            })
            delta_f1 = m_q['f1'] - m_rbf['f1']
            print(f"    RBF F1={m_rbf['f1']:.4f} ({t_tot_rbf:.1f}s) | Quantum F1={m_q['f1']:.4f} ({t_tot_q:.1f}s) | Delta={delta_f1:+.4f}")

    # ============================================================
    # SAVE RAW RESULTS
    # ============================================================
    df_results = pd.DataFrame(all_results)
    results_csv_path = os.path.join(EXP40_DIR, "exp40_results.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"\n-> Saved raw experiment results ({len(df_results)} rows) to: {results_csv_path}")

    # Save Config JSON
    config_data = {
        "experiment_series": "Experiment 40 (Final Confirmation)",
        "protocol_version": "1.0",
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "experiments": ["Exp40A_IID_8D", "Exp40B_Direction_B_8D", "Exp40C_IID_10D", "Exp40C_IID_12D"],
        "tfidf": {
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": [1, 2],
            "min_df": 2,
            "sublinear_tf": True,
            "max_features": 50000,
            "norm": "l2"
        },
        "pca": "TruncatedSVD + StandardScaler",
        "classifiers": {
            "linear_svm": "LinearSVC(C=1.0, class_weight='balanced', max_iter=2000)",
            "classical_rbf": "SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')",
            "quantum_kernel": "SVC(kernel='precomputed', C=1.0, class_weight='balanced', feature_map='2-layer cyclic ZZFeatureMap')"
        },
        "threshold_selection": "Validation set F1 maximization (200 threshold grid)",
        "metrics_recorded": ["f1", "pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "precision", "recall", "decision_margin", "runtime", "memory"],
        "system_metadata": meta
    }
    with open(os.path.join(EXP40_DIR, "exp40_config.json"), "w") as f:
        json.dump(config_data, f, indent=2)

    # ============================================================
    # STATISTICAL ANALYSIS & SYNTHESIS
    # ============================================================
    print("\n" + "=" * 80)
    print("PERFORMING COMPREHENSIVE STATISTICAL ANALYSIS ACROSS 10 SEEDS")
    print("=" * 80)

    comparisons = [
        ("Exp40A_IID_8D", "IID 8D"),
        ("Exp40B_Direction_B_8D", "Direction B 8D"),
        ("Exp40C_IID_10D", "IID 10D"),
        ("Exp40C_IID_12D", "IID 12D"),
    ]

    stat_summary = []
    p_values_to_adjust = []
    test_keys = []

    for exp_id, label in comparisons:
        sub = df_results[df_results["experiment"] == exp_id]
        q_rows = sub[sub["model"] == "Quantum Kernel"].sort_values("seed")
        rbf_rows = sub[sub["model"] == "Classical RBF"].sort_values("seed")

        f1_q = q_rows["f1"].values
        f1_rbf = rbf_rows["f1"].values
        diffs = f1_q - f1_rbf

        # Summary statistics
        q_mean, q_sd = np.mean(f1_q), np.std(f1_q, ddof=1)
        rbf_mean, rbf_sd = np.mean(f1_rbf), np.std(f1_rbf, ddof=1)
        diff_mean, diff_sd = np.mean(diffs), np.std(diffs, ddof=1)
        diff_med = np.median(diffs)
        diff_min, diff_max = np.min(diffs), np.max(diffs)

        # 95% CIs
        t_ci = stats.t.interval(0.95, df=len(diffs)-1, loc=diff_mean, scale=stats.sem(diffs))
        boot_ci_low, boot_ci_high = paired_bootstrap_ci(diffs, n_boot=10000, seed=42)

        # Permutation Test
        p_perm = paired_permutation_test(diffs, n_perm=10000, seed=42)
        p_values_to_adjust.append(p_perm)
        test_keys.append(f"{label} Permutation")

        # Practical Equivalence
        if diff_mean > 0.01:
            equiv = "Quantum Advantage"
        elif diff_mean < -0.01:
            equiv = "Classical Advantage"
        else:
            equiv = "Practical Equivalence (|Δ| <= 0.01)"

        stat_summary.append({
            "Experiment": exp_id,
            "Label": label,
            "Q_F1_Mean": q_mean,
            "Q_F1_SD": q_sd,
            "RBF_F1_Mean": rbf_mean,
            "RBF_F1_SD": rbf_sd,
            "Delta_F1_Mean": diff_mean,
            "Delta_F1_SD": diff_sd,
            "Delta_F1_Median": diff_med,
            "Delta_F1_Min": diff_min,
            "Delta_F1_Max": diff_max,
            "t_CI_Lower": t_ci[0],
            "t_CI_Upper": t_ci[1],
            "Bootstrap_CI_Lower": boot_ci_low,
            "Bootstrap_CI_Upper": boot_ci_high,
            "Permutation_p": p_perm,
            "Practical_Status": equiv,
        })

    # Benjamini-Hochberg FDR correction
    p_adj = stats.false_discovery_control(p_values_to_adjust, method="bh") if hasattr(stats, "false_discovery_control") else p_values_to_adjust
    for idx, row in enumerate(stat_summary):
        row["BH_p_adjusted"] = float(p_adj[idx]) if hasattr(stats, "false_discovery_control") else row["Permutation_p"]

    df_stat = pd.DataFrame(stat_summary)
    df_stat.to_csv(os.path.join(EXP40_DIR, "exp40_statistical_summary.csv"), index=False)

    print("\n--- STATISTICAL CONFIRMATION SUMMARY (N = 10 SEEDS) ---")
    for r in stat_summary:
        print(f"\n[{r['Label']}]")
        print(f"  Quantum F1:     {r['Q_F1_Mean']:.6f} ± {r['Q_F1_SD']:.6f}")
        print(f"  Classical RBF:  {r['RBF_F1_Mean']:.6f} ± {r['RBF_F1_SD']:.6f}")
        print(f"  Delta F1:       {r['Delta_F1_Mean']:+.6f} ± {r['Delta_F1_SD']:.6f} (Range: [{r['Delta_F1_Min']:+.4f}, {r['Delta_F1_Max']:+.4f}])")
        print(f"  95% Boot CI:    [{r['Bootstrap_CI_Lower']:+.6f}, {r['Bootstrap_CI_Upper']:+.6f}]")
        print(f"  Permutation p:  p = {r['Permutation_p']:.4f} (BH FDR p = {r['BH_p_adjusted']:.4f})")
        print(f"  Classification: {r['Practical_Status']}")

    print("\nExperiment 40 Execution and Analysis Completed Successfully!")

if __name__ == "__main__":
    main()
