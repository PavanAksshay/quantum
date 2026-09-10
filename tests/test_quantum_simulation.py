"""
Tests for Quantum Statevector Simulation & Entropy Engine using unittest.
"""

import unittest
import numpy as np
import torch
from app.model_engine.quantum import simulate_zz_feature_map, compute_von_neumann_entropy, extract_top_basis_states


class TestQuantumSimulation(unittest.TestCase):

    def test_statevector_normalization_and_dimension(self):
        n_qubits = 8
        batch_size = 2
        X_phase = torch.tensor(np.random.uniform(0.0, np.pi, size=(batch_size, n_qubits)), dtype=torch.float64)
        
        statevector = simulate_zz_feature_map(X_phase, n_qubits=n_qubits)
        self.assertEqual(statevector.shape, (batch_size, 2 ** n_qubits))
        
        # Verify exact unit norm (sum of probabilities == 1.0)
        probs = torch.sum(torch.abs(statevector) ** 2, dim=1).cpu().numpy()
        self.assertTrue(np.allclose(probs, 1.0, atol=1e-10))

    def test_von_neumann_entropy(self):
        n_qubits = 4
        X_phase = torch.tensor(np.random.uniform(0.0, np.pi, size=(1, n_qubits)), dtype=torch.float64)
        statevector = simulate_zz_feature_map(X_phase, n_qubits=n_qubits)
        
        entropy = compute_von_neumann_entropy(statevector[0], n_qubits=n_qubits)
        self.assertTrue(0.0 <= entropy <= 1.0)  # single-qubit reduced entropy bounded by 1 bit

    def test_top_basis_extraction(self):
        n_qubits = 8
        X_phase = torch.tensor(np.random.uniform(0.0, np.pi, size=(1, n_qubits)), dtype=torch.float64)
        statevector = simulate_zz_feature_map(X_phase, n_qubits=n_qubits)
        
        top_basis = extract_top_basis_states(statevector[0], n_qubits=n_qubits, top_k=8)
        self.assertEqual(len(top_basis), 8)
        self.assertTrue(all("basis" in item and "probability" in item for item in top_basis))
        self.assertGreaterEqual(top_basis[0]["probability"], top_basis[-1]["probability"])


if __name__ == "__main__":
    unittest.main()
