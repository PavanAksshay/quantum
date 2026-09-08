import os
import numpy as np


def create_balanced_split(
    X,
    y,
    train_per_class=150,
    val_per_class=50,
    test_per_class=75,
    seed=42
):
    """
    Creates one fixed balanced train/validation/test split.

    The same split can then be reused by every
    quantum experiment.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []
    test_indices = []

    for label in [0, 1]:

        indices = np.where(
            y == label
        )[0]

        rng.shuffle(indices)

        required = (
            train_per_class
            + val_per_class
            + test_per_class
        )

        if len(indices) < required:

            raise ValueError(
                f"Not enough samples for class {label}. "
                f"Required {required}, "
                f"available {len(indices)}."
            )

        train_indices.extend(
            indices[
                :train_per_class
            ]
        )

        val_indices.extend(
            indices[
                train_per_class:
                train_per_class + val_per_class
            ]
        )

        test_indices.extend(
            indices[
                train_per_class + val_per_class:
                required
            ]
        )

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return (
        X[train_indices],
        X[val_indices],
        X[test_indices],
        y[train_indices],
        y[val_indices],
        y[test_indices]
    )


def save_quantum_split(
    output_dir="results/quantum_split",
    **arrays
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    for name, array in arrays.items():

        np.save(
            os.path.join(
                output_dir,
                f"{name}.npy"
            ),
            array
        )


def load_quantum_split(
    output_dir="results/quantum_split"
):

    names = [
        "X_train",
        "X_val",
        "X_test",
        "y_train",
        "y_val",
        "y_test"
    ]

    data = {}

    for name in names:

        path = os.path.join(
            output_dir,
            f"{name}.npy"
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Missing benchmark file: {path}"
            )

        data[name] = np.load(path)

    return data