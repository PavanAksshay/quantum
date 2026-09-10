"""
Classical Baseline Models: Matched Gaussian RBF SVM and Linear SVM Baseline.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.svm import SVC, LinearSVC


def build_classical_models(C: float = 1.0, gamma: str = "scale") -> Tuple[SVC, LinearSVC]:
    """Initializes matched Classical RBF and Linear SVM baselines."""
    rbf_clf = SVC(kernel="rbf", C=C, gamma=gamma, random_state=42)
    linear_clf = LinearSVC(C=C, random_state=42, max_iter=2000, dual="auto")
    return rbf_clf, linear_clf


def compute_rbf_kernel_matrix(X1: np.ndarray, X2: np.ndarray, gamma: float = None) -> np.ndarray:
    """Computes Gaussian RBF Gram matrix K(x, z) = exp(-gamma * ||x - z||^2)."""
    if gamma is None:
        gamma = 1.0 / (X1.shape[1] * np.var(X1)) if np.var(X1) > 0 else 1.0 / X1.shape[1]
    
    dists = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1, keepdims=True).T - 2.0 * np.dot(X1, X2.T)
    dists = np.maximum(dists, 0.0)
    return np.exp(-gamma * dists)
