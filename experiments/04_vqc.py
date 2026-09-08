import os
import sys
import time

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(
    PROJECT_ROOT
)


from src.quantum_model import (
    HybridVQC
)


N_QUBITS = 4

N_LAYERS = 2

TRAIN_PER_CLASS = 100

TEST_PER_CLASS = 50

EPOCHS = 30


def balanced_subset(
    X,
    y,
    samples_per_class,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    X = np.asarray(X)
    y = np.asarray(y)

    indices = []

    for label in [0, 1]:

        class_indices = np.where(
            y == label
        )[0]

        chosen = rng.choice(
            class_indices,
            size=samples_per_class,
            replace=False
        )

        indices.extend(
            chosen
        )

    rng.shuffle(
        indices
    )

    return (
        X[indices],
        y[indices]
    )


def main():

    # ==========================================
    # Load PCA data
    # ==========================================

    X_train = np.load(
        "results/X_train_pca.npy"
    )

    X_test = np.load(
        "results/X_test_pca.npy"
    )

    y_train = np.load(
        "results/y_train.npy"
    )

    y_test = np.load(
        "results/y_test.npy"
    )

    # ==========================================
    # Balanced benchmark
    # ==========================================

    X_train, y_train = (
        balanced_subset(
            X_train,
            y_train,
            TRAIN_PER_CLASS,
            42
        )
    )

    X_test, y_test = (
        balanced_subset(
            X_test,
            y_test,
            TEST_PER_CLASS,
            43
        )
    )

    print(
        "Training shape:",
        X_train.shape
    )

    print(
        "Testing shape:",
        X_test.shape
    )

    print(
        "Training distribution:",
        np.bincount(y_train)
    )

    print(
        "Testing distribution:",
        np.bincount(y_test)
    )

    # ==========================================
    # Model
    # ==========================================

    model = HybridVQC(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        learning_rate=0.02,
        epochs=EPOCHS
    )

    # ==========================================
    # Train
    # ==========================================

    print(
        "\nStarting Hybrid VQC training..."
    )

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.time() - start
    )

    # ==========================================
    # Predict
    # ==========================================

    print(
        "\nEvaluating..."
    )

    start = time.time()

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    inference_time = (
        time.time() - start
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # ==========================================
    # Metrics
    # ==========================================

    results = {

        "Model":
            "DistilBERT + Hybrid VQC",

        "Qubits":
            N_QUBITS,

        "Layers":
            N_LAYERS,

        "Accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "Precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "PR_AUC":
            average_precision_score(
                y_test,
                probabilities
            ),

        "ROC_AUC":
            roc_auc_score(
                y_test,
                probabilities
            ),

        "Training_Time":
            training_time,

        "Inference_Time":
            inference_time
    }

    # ==========================================
    # Print
    # ==========================================

    print("\n")
    print("=" * 65)
    print("HYBRID VQC RESULTS")
    print("=" * 65)

    for key, value in results.items():

        if isinstance(
            value,
            float
        ):

            print(
                f"{key}: {value:.4f}"
            )

        else:

            print(
                f"{key}: {value}"
            )

    # ==========================================
    # Save
    # ==========================================

    os.makedirs(
        "results/metrics",
        exist_ok=True
    )

    pd.DataFrame(
        [results]
    ).to_csv(
        "results/metrics/hybrid_vqc.csv",
        index=False
    )

    print(
        "\nSaved:"
        " results/metrics/hybrid_vqc.csv"
    )


if __name__ == "__main__":

    main()