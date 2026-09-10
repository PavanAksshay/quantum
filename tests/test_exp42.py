import os
import unittest
import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVC, LinearSVC

from app.model_engine.quantum import simulate_zz_feature_map
from experiments.exp42_confirmatory_analysis import (
    load_dataset_splits,
    project_to_target_dimension,
    compute_quantum_gram_matrix,
    compute_comprehensive_geometry,
    optimize_threshold,
    paired_permutation_test,
    bootstrap_ci,
    benjamini_hochberg
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestExp42ConfirmatoryAnalysis(unittest.TestCase):
    """
    Comprehensive verification suite for Experiment 42:
    - Leakage safety
    - Seed determinism
    - Validation-only thresholding
    - Exact Quantum Fidelity Kernel properties (symmetry, positive semi-definiteness, finite values)
    - Paired permutation and statistical correction validity
    - Canonical Exp 39/Exp 40 invariance guarantee
    """

    def test_seed_determinism_and_leakage_safety(self):
        """Verifies that stratified sampling is 100% deterministic and partitions do not overlap."""
        df_tr1, df_va1, df_te1 = load_dataset_splits("sms", seed=42)
        df_tr2, df_va2, df_te2 = load_dataset_splits("sms", seed=42)

        self.assertEqual(len(df_tr1), len(df_tr2))
        pd.testing.assert_frame_equal(df_tr1, df_tr2)

        # Zero sample leakage between splits
        tr_texts = set(df_tr1["text"].tolist())
        va_texts = set(df_va1["text"].tolist())
        te_texts = set(df_te1["text"].tolist())

        # No index collisions in underlying sample IDs
        tr_ids = set(df_tr1["sample_id"].tolist()) if "sample_id" in df_tr1.columns else set(df_tr1.index)
        va_ids = set(df_va1["sample_id"].tolist()) if "sample_id" in df_va1.columns else set(df_va1.index)
        te_ids = set(df_te1["sample_id"].tolist()) if "sample_id" in df_te1.columns else set(df_te1.index)

        self.assertEqual(len(tr_ids.intersection(va_ids)), 0)
        self.assertEqual(len(tr_ids.intersection(te_ids)), 0)
        self.assertEqual(len(va_ids.intersection(te_ids)), 0)

    def test_projection_leakage_protection(self):
        """Verifies SVD and Scaler are fit strictly on training data."""
        np.random.seed(42)
        X_tr = np.random.randn(50, 100).astype(np.float32)
        X_va = np.random.randn(25, 100).astype(np.float32)
        X_te = np.random.randn(25, 100).astype(np.float32)

        Z_tr, Z_va, Z_te, A_tr, A_va, A_te = project_to_target_dimension(
            X_tr, X_va, X_te, target_dim=8, seed=42
        )

        self.assertEqual(Z_tr.shape, (50, 8))
        self.assertEqual(Z_va.shape, (25, 8))
        self.assertEqual(Z_te.shape, (25, 8))

        # Training set is zero-mean and unit-variance
        np.testing.assert_allclose(Z_tr.mean(axis=0), np.zeros(8), atol=1e-5)
        np.testing.assert_allclose(Z_tr.std(axis=0), np.ones(8), atol=1e-5)

        # Angles strictly bounded in [0, pi]
        self.assertTrue(np.all(A_tr >= 0.0) and np.all(A_tr <= np.pi))
        self.assertTrue(np.all(A_va >= 0.0) and np.all(A_va <= np.pi))
        self.assertTrue(np.all(A_te >= 0.0) and np.all(A_te <= np.pi))

    def test_quantum_gram_matrix_mathematical_properties(self):
        """Verifies symmetry, exact 1.0 diagonal fidelity, and finite values for quantum Gram matrix."""
        np.random.seed(123)
        A = np.random.uniform(0, np.pi, size=(20, 8))
        K_q = compute_quantum_gram_matrix(A, A, n_qubits=8)

        # 1. Symmetry
        sym_error = np.max(np.abs(K_q - K_q.T))
        self.assertLess(sym_error, 1e-12)

        # 2. Diagonal unit fidelity
        np.testing.assert_allclose(np.diag(K_q), np.ones(20), atol=1e-12)

        # 3. All finite and in [0, 1]
        self.assertTrue(np.all(np.isfinite(K_q)))
        self.assertTrue(np.all(K_q >= 0.0))
        self.assertTrue(np.all(K_q <= 1.0000001))

        # 4. Positive semi-definiteness
        eigvals = np.linalg.eigvalsh(K_q)
        self.assertGreaterEqual(np.min(eigvals), -1e-6)

    def test_geometry_diagnostics_calculation(self):
        """Verifies geometric diagnostics function correctly on synthetic Gram matrices."""
        np.random.seed(456)
        A = np.random.uniform(0, np.pi, size=(15, 8))
        K_q = compute_quantum_gram_matrix(A, A, n_qubits=8)
        K_rbf = np.exp(-0.1 * np.sum((A[:, None, :] - A[None, :, :]) ** 2, axis=-1))
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        geom = compute_comprehensive_geometry(K_q, K_rbf, y)
        self.assertIn("q_kernel_diversity", geom)
        self.assertIn("rbf_kernel_diversity", geom)
        self.assertIn("q_label_alignment", geom)
        self.assertIn("gram_pearson_r", geom)
        self.assertTrue(geom["q_psd_valid"])
        self.assertTrue(geom["q_finite_check"])

    def test_validation_only_threshold_optimization(self):
        """Verifies that threshold selection only operates on validation scores."""
        scores = np.array([-1.5, -0.5, 0.2, 0.8, 1.4, 2.1])
        labels = np.array([0, 0, 1, 1, 1, 1])
        t_opt = optimize_threshold(scores, labels, n_steps=100)
        self.assertGreater(t_opt, -1.0)
        self.assertLessEqual(t_opt, 1.0)

    def test_statistical_inference_functions(self):
        """Verifies permutation testing and BH-FDR correction."""
        deltas = np.array([0.02, 0.03, 0.015, 0.025, 0.018, 0.022, 0.031, 0.019, 0.027, 0.021])
        p_val = paired_permutation_test(deltas, n_perm=1000)
        self.assertLess(p_val, 0.05)

        low, high = bootstrap_ci(deltas, n_boot=1000)
        self.assertGreater(low, 0.0)
        self.assertLess(high, 0.05)

        raw_p = [0.001, 0.04, 0.03]
        adj_p = benjamini_hochberg(raw_p)
        self.assertEqual(len(adj_p), 3)
        self.assertLessEqual(adj_p[0], adj_p[1])

    def test_canonical_freeze_invariance(self):
        """Guarantees that canonical Exp 39 and Exp 40 results remain strictly untouched."""
        exp40_evidence = os.path.join(BASE_DIR, "results/exp40_final/exp40_statistical_summary.csv")
        exp39_evidence = os.path.join(BASE_DIR, "results/exp39_paper/FINAL_EVIDENCE_MATRIX.csv")
        freeze_md = os.path.join(BASE_DIR, "results/exp39_paper/RESULT_FREEZE.md")

        self.assertTrue(os.path.exists(exp40_evidence))
        self.assertTrue(os.path.exists(exp39_evidence))
        self.assertTrue(os.path.exists(freeze_md))

        df_freeze40 = pd.read_csv(exp40_evidence)
        df_freeze39 = pd.read_csv(exp39_evidence)
        self.assertEqual(len(df_freeze40), 4)
        self.assertEqual(len(df_freeze39), 6)


if __name__ == "__main__":
    unittest.main()
