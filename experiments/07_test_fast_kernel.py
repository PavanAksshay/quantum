import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

from src.fast_quantum_kernel import (
    FastQuantumKernel
)


def main():

    X = np.load(
        "results/quantum_split/X_train.npy"
    )

    X = X[:, :4]

    # Quantum angle preparation

    X = np.tanh(X) * np.pi

    print(
        "Dataset:",
        X.shape
    )

    kernel = FastQuantumKernel(
        n_qubits=4,
        n_layers=1
    )

    print(
        "\nComputing optimized kernel..."
    )

    start = time.time()

    K = kernel.matrix(X)

    elapsed = time.time() - start

    print(
        f"Kernel time: "
        f"{elapsed:.2f} seconds"
    )

    print(
        "Kernel shape:",
        K.shape
    )

    print(
        "Kernel minimum:",
        K.min()
    )

    print(
        "Kernel maximum:",
        K.max()
    )

    print(
        "Kernel diagonal mean:",
        np.mean(np.diag(K))
    )


if __name__ == "__main__":
    main()