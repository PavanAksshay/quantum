"""
Kernel Geometry & Alignment Diagnostics Engine.
Calculates Gram matrices, target label alignment deficits, and entropy-diversity correlations.
"""

from typing import Dict, Any, List
import numpy as np
import torch
from app.model_engine.quantum import simulate_zz_feature_map
from app.model_engine.classical import compute_rbf_kernel_matrix


def compute_target_label_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    Computes Kernel-Target Label Alignment:
    A(K, y) = <K, y y^T>_F / (||K||_F * ||y y^T||_F)
    """
    y_vec = y.reshape(-1, 1).astype(float)
    y_mat = np.dot(y_vec, y_vec.T)
    numerator = np.sum(K * y_mat)
    denom = np.linalg.norm(K, 'fro') * np.linalg.norm(y_mat, 'fro')
    if denom == 0:
        return 0.0
    return float(numerator / denom)


def compute_sample_gram_matrices(X_phase: np.ndarray, sample_size: int = 20) -> Dict[str, Any]:
    """Generates sample 20x20 Gram matrices for Quantum Fidelity Kernel and Matched RBF."""
    sub_X = X_phase[:sample_size]
    n_samples, n_dim = sub_X.shape
    
    # 1. Quantum Kernel
    tensor_X = torch.tensor(sub_X, dtype=torch.float64)
    states = simulate_zz_feature_map(tensor_X, n_qubits=n_dim)
    inner = torch.matmul(states, states.conj().T)
    K_quantum = (torch.abs(inner) ** 2).cpu().numpy()
    np.fill_diagonal(K_quantum, 1.0)
    
    # 2. Classical RBF Kernel
    K_rbf = compute_rbf_kernel_matrix(sub_X, sub_X)
    np.fill_diagonal(K_rbf, 1.0)

    # 3. Off-diagonal correlation
    triu_idx = np.triu_indices(n_samples, k=1)
    q_vals = K_quantum[triu_idx]
    rbf_vals = K_rbf[triu_idx]
    
    corr = float(np.corrcoef(q_vals, rbf_vals)[0, 1]) if len(q_vals) > 1 else 0.0

    return {
        "sample_size": sample_size,
        "dimension": n_dim,
        "quantum_gram": [[round(float(val), 4) for val in row] for row in K_quantum],
        "rbf_gram": [[round(float(val), 4) for val in row] for row in K_rbf],
        "pearson_correlation": round(corr, 4)
    }
