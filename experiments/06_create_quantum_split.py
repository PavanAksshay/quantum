import os
import sys

import numpy as np


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


from src.quantum_benchmark import (
    create_balanced_split,
    save_quantum_split
)


def main():

    X = np.load(
        "results/X_train_pca.npy"
    )

    y = np.load(
        "results/y_train.npy"
    )

    print(
        "Available training data:",
        X.shape
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = create_balanced_split(
        X,
        y
    )

    print("\n")
    print("=" * 60)
    print("CONTROLLED QUANTUM BENCHMARK")
    print("=" * 60)

    print(
        "Training:",
        X_train.shape,
        np.bincount(y_train)
    )

    print(
        "Validation:",
        X_val.shape,
        np.bincount(y_val)
    )

    print(
        "Testing:",
        X_test.shape,
        np.bincount(y_test)
    )

    save_quantum_split(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test
    )

    print(
        "\nSaved to:"
        " results/quantum_split/"
    )


if __name__ == "__main__":
    main()