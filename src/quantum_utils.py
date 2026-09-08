import numpy as np


def prepare_quantum_features(
    X,
    n_qubits
):
    """
    Convert PCA features into bounded
    quantum rotation angles.
    """

    X = np.asarray(
        X,
        dtype=np.float64
    )

    if X.ndim != 2:

        raise ValueError(
            "X must be a 2D array."
        )

    if X.shape[1] < n_qubits:

        raise ValueError(
            f"X contains {X.shape[1]} "
            f"features but {n_qubits} "
            f"qubits were requested."
        )

    X = X[
        :,
        :n_qubits
    ]

    # Bound extreme PCA values
    X = np.tanh(X)

    # Convert to rotation angles
    X = X * np.pi

    return X