"""
PyTorch complex128 Quantum Statevector Simulation & Fidelity Kernel Engine.
Implements the canonical 2-layer cyclic ZZFeatureMap and von Neumann entanglement entropy.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import torch

DEVICE = torch.device("cpu")
DTYPE = torch.complex128


def simulate_zz_feature_map(X: torch.Tensor, n_qubits: int = 8) -> torch.Tensor:
    """
    Simulates a 2-layer cyclic ZZFeatureMap with fidelity kernel in complex128.
    X: [Batch, n_qubits] in range [0, pi]
    Returns: statevector tensor of shape [Batch, 2^n_qubits]
    """
    B = X.shape[0]
    state = torch.zeros([B] + [2] * n_qubits, dtype=DTYPE, device=DEVICE)
    state[(slice(None),) + (0,) * n_qubits] = 1.0 + 0.0j

    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    H_mat = torch.tensor([[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]], dtype=DTYPE, device=DEVICE)

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
                (torch.arange(2, device=DEVICE).view(*[2 if i == q else 1 for i in range(n)]) ==
                 torch.arange(2, device=DEVICE).view(*[2 if i == q_next else 1 for i in range(n)])).unsqueeze(0),
                phase_same,
                phase_diff
            )
        return s

    # Layer 1
    state = apply_hadamard_all(state, n_qubits)
    state = apply_rz_all(state, X, n_qubits)
    state = apply_rzz_cyclic(state, X, n_qubits)

    # Layer 2
    state = apply_hadamard_all(state, n_qubits)
    state = apply_rz_all(state, X, n_qubits)
    state = apply_rzz_cyclic(state, X, n_qubits)

    return state.view(B, 2 ** n_qubits)


def compute_von_neumann_entropy(statevector: torch.Tensor, n_qubits: int = 8) -> float:
    """Computes single-qubit reduced von Neumann entropy in bits."""
    state_tensor = statevector.view(*([2] * n_qubits))
    trace_dims = list(range(1, n_qubits))
    rho_q0 = torch.tensordot(state_tensor, state_tensor.conj(), dims=(trace_dims, trace_dims))
    eigvals = torch.linalg.eigvalsh(rho_q0).real
    eigvals = eigvals[eigvals > 1e-12]
    entropy = float(-torch.sum(eigvals * torch.log2(eigvals)).item())
    return max(0.0, entropy)


def extract_top_basis_states(statevector: torch.Tensor, n_qubits: int = 8, top_k: int = 8) -> List[Dict[str, Any]]:
    """Extracts top K computational basis states and their measurement collapse probabilities."""
    probs = (torch.abs(statevector) ** 2).cpu().numpy()
    top_indices = np.argsort(-probs)[:top_k]
    return [
        {
            "basis": f"|{bin(idx)[2:].zfill(n_qubits)}⟩",
            "index": int(idx),
            "probability": float(probs[idx]),
            "amplitude_real": float(statevector[idx].real.item()),
            "amplitude_imag": float(statevector[idx].imag.item())
        }
        for idx in top_indices
    ]
