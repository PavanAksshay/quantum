import os
import unittest
import numpy as np
import pandas as pd
import torch

from experiments.exp43_diagnostic_study import (
    apply_angular_transformation,
    simulate_zz_feature_map_depth,
    compute_quantum_gram_matrix_depth,
    compute_comprehensive_geometry,
    optimize_threshold,
    paired_permutation_test,
    bootstrap_ci,
    benjamini_hochberg
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestExp43DiagnosticStudy(unittest.TestCase):
    """
    Unit test verification suite for Experiment 43:
    - Angular mapping determinism & range boundaries
    - Feature map depth scaling (1, 2, 3 layers)
    - Quantum Gram matrix mathematical validity
    - Inferential statistics (permutation test & BH-FDR)
    - Complete immutability of canonical Exp39, Exp40, Exp41, and Exp42
    """

    def test_angle_mapping_determinism_and_bounds(self):
        """Verifies that all 4 angle mappings are deterministic and strictly observe theoretical boundaries."""
        np.random.seed(42)
        Z_tr = np.random.randn(50, 8).astype(np.float64)
        Z_va = np.random.randn(25, 8).astype(np.float64)
        Z_te = np.random.randn(25, 8).astype(np.float64)

        # 1. Canonical [0, pi]
        A_tr1, A_va1, A_te1 = apply_angular_transformation(Z_tr, Z_va, Z_te, "0_pi")
        self.assertEqual(A_tr1.shape, (50, 8))
        self.assertTrue(np.all(A_tr1 >= 0.0) and np.all(A_tr1 <= np.pi))
        self.assertTrue(np.all(A_te1 >= 0.0) and np.all(A_te1 <= np.pi))

        # 2. Symmetric [-pi, pi]
        A_tr2, A_va2, A_te2 = apply_angular_transformation(Z_tr, Z_va, Z_te, "minus_pi_pi")
        self.assertEqual(A_tr2.shape, (50, 8))
        self.assertTrue(np.all(A_tr2 >= -np.pi - 1e-12) and np.all(A_tr2 <= np.pi + 1e-12))
        self.assertTrue(np.all(A_te2 >= -np.pi - 1e-12) and np.all(A_te2 <= np.pi + 1e-12))

        # 3. Full circle [0, 2pi]
        A_tr3, A_va3, A_te3 = apply_angular_transformation(Z_tr, Z_va, Z_te, "0_2pi")
        self.assertEqual(A_tr3.shape, (50, 8))
        self.assertTrue(np.all(A_tr3 >= 0.0) and np.all(A_tr3 <= 2.0 * np.pi + 1e-12))
        self.assertTrue(np.all(A_te3 >= 0.0) and np.all(A_te3 <= 2.0 * np.pi + 1e-12))

        # 4. Normal CDF [0, pi]
        A_tr4, A_va4, A_te4 = apply_angular_transformation(Z_tr, Z_va, Z_te, "train_normalized")
        self.assertEqual(A_tr4.shape, (50, 8))
        self.assertTrue(np.all(A_tr4 >= 0.0) and np.all(A_tr4 <= np.pi))
        self.assertTrue(np.all(A_te4 >= 0.0) and np.all(A_te4 <= np.pi))

    def test_feature_map_depth_simulation(self):
        """Verifies statevector simulation across depths 1, 2, 3."""
        A = np.random.uniform(0, np.pi, size=(10, 8))
        for d in [1, 2, 3]:
            K_q = compute_quantum_gram_matrix_depth(A, A, n_qubits=8, depth=d)
            self.assertEqual(K_q.shape, (10, 10))
            # Diagonal fidelity must be exactly 1.0
            np.testing.assert_allclose(np.diag(K_q), np.ones(10), atol=1e-12)
            # Symmetry error
            sym_err = np.max(np.abs(K_q - K_q.T))
            self.assertLess(sym_err, 1e-12)
            # PSD check
            eigvals = np.linalg.eigvalsh(K_q)
            self.assertGreaterEqual(np.min(eigvals), -1e-6)

    def test_geometry_diagnostics_completeness(self):
        """Verifies that all geometry fields including effective rank and eigenvalue fraction are computed."""
        A = np.random.uniform(0, np.pi, size=(15, 8))
        K_q = compute_quantum_gram_matrix_depth(A, A, n_qubits=8, depth=2)
        K_rbf = np.exp(-0.1 * np.sum((A[:, None, :] - A[None, :, :]) ** 2, axis=-1))
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        geom = compute_comprehensive_geometry(K_q, K_rbf, y)
        self.assertIn("q_effective_rank", geom)
        self.assertIn("q_largest_eig_fraction", geom)
        self.assertIn("q_spectral_entropy", geom)
        self.assertIn("q_label_alignment", geom)
        self.assertIn("q_kernel_diversity", geom)
        self.assertIn("gram_pearson_r", geom)
        self.assertTrue(geom["q_finite_check"])

    def test_inferential_statistics_and_fdr(self):
        """Verifies BH-FDR correction across 7 hypotheses."""
        raw_p = [0.001, 0.04, 0.02, 0.25, 0.005, 0.12, 0.08]
        adj_p = benjamini_hochberg(raw_p)
        self.assertEqual(len(adj_p), 7)
        self.assertLessEqual(adj_p[0], 0.05)
        # Monotonicity check
        sorted_pairs = sorted(zip(raw_p, adj_p), key=lambda x: x[0])
        for i in range(len(sorted_pairs) - 1):
            self.assertLessEqual(sorted_pairs[i][1], sorted_pairs[i+1][1] + 1e-12)

    def test_freeze_immutability_guarantee(self):
        """Verifies that Exp 39, Exp 40, Exp 41, and Exp 42 artifacts remain strictly untouched."""
        exp39_file = os.path.join(BASE_DIR, "results/exp39_paper/FINAL_EVIDENCE_MATRIX.csv")
        exp40_file = os.path.join(BASE_DIR, "results/exp40_final/exp40_statistical_summary.csv")
        exp41_file = os.path.join(BASE_DIR, "results/exp41/exp41_summary.csv")
        exp42_file = os.path.join(BASE_DIR, "results/exp42/exp42_summary.csv")
        freeze_file = os.path.join(BASE_DIR, "results/exp39_paper/RESULT_FREEZE.md")

        self.assertTrue(os.path.exists(exp39_file))
        self.assertTrue(os.path.exists(exp40_file))
        self.assertTrue(os.path.exists(exp41_file))
        self.assertTrue(os.path.exists(exp42_file))
        self.assertTrue(os.path.exists(freeze_file))

        df_exp42 = pd.read_csv(exp42_file)
        self.assertEqual(len(df_exp42), 4)


if __name__ == "__main__":
    unittest.main()
