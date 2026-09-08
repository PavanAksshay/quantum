import os
import sys
import time

import numpy as np
import pandas as pd

from sklearn.svm import SVC

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

sys.path.append(PROJECT_ROOT)


from src.quantum_kernel import (
    QuantumKernel
)

from src.quantum_utils import (
    prepare_quantum_features
)

from src.quantum_benchmark import (
    load_quantum_split
)


N_QUBITS = 4


def evaluate_at_threshold(
    probabilities,
    y_true,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "Accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "Precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "PR_AUC":
            average_precision_score(
                y_true,
                probabilities
            ),

        "ROC_AUC":
            roc_auc_score(
                y_true,
                probabilities
            )
    }


def find_validation_threshold(
    probabilities,
    y_true
):

    best_threshold = 0.5
    best_f1 = -1

    for threshold in np.arange(
        0.10,
        0.91,
        0.01
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    return (
        best_threshold,
        best_f1
    )


def main():

    # ==========================================
    # Load fixed benchmark
    # ==========================================

    data = load_quantum_split()

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

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

    # ==========================================
    # Quantum features
    # ==========================================

    X_train = prepare_quantum_features(
        X_train,
        N_QUBITS
    )

    X_val = prepare_quantum_features(
        X_val,
        N_QUBITS
    )

    X_test = prepare_quantum_features(
        X_test,
        N_QUBITS
    )

    # ==========================================
    # Quantum kernel
    # ==========================================

    quantum_kernel = QuantumKernel(
        n_qubits=N_QUBITS,
        n_layers=2
    )

    # ==========================================
    # Training kernel
    # ==========================================

    print(
        "\nComputing training kernel..."
    )

    start = time.time()

    K_train = quantum_kernel.matrix(
        X_train
    )

    train_kernel_time = (
        time.time() - start
    )

    print(
        f"Training kernel time: "
        f"{train_kernel_time:.2f}s"
    )

    # ==========================================
    # Validation kernel
    # ==========================================

    print(
        "\nComputing validation kernel..."
    )

    start = time.time()

    K_val = quantum_kernel.matrix(
        X_val,
        X_train
    )

    val_kernel_time = (
        time.time() - start
    )

    print(
        f"Validation kernel time: "
        f"{val_kernel_time:.2f}s"
    )

    # ==========================================
    # Test kernel
    # ==========================================

    print(
        "\nComputing test kernel..."
    )

    start = time.time()

    K_test = quantum_kernel.matrix(
        X_test,
        X_train
    )

    test_kernel_time = (
        time.time() - start
    )

    print(
        f"Test kernel time: "
        f"{test_kernel_time:.2f}s"
    )

    # ==========================================
    # SVM
    # ==========================================

    print(
        "\nTraining quantum-kernel SVM..."
    )

    model = SVC(
        kernel="precomputed",
        class_weight="balanced",
        probability=True,
        random_state=42
    )

    model.fit(
        K_train,
        y_train
    )

    # ==========================================
    # Validation threshold
    # ==========================================

    val_probabilities = (
        model.predict_proba(
            K_val
        )[:, 1]
    )

    threshold, val_f1 = (
        find_validation_threshold(
            val_probabilities,
            y_val
        )
    )

    print(
        f"\nSelected threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{val_f1:.4f}"
    )

    # ==========================================
    # FINAL TEST
    # ==========================================

    test_probabilities = (
        model.predict_proba(
            K_test
        )[:, 1]
    )

    metrics = evaluate_at_threshold(
        test_probabilities,
        y_test,
        threshold
    )

    # ==========================================
    # Results
    # ==========================================

    results = {

        "Model":
            "DistilBERT + Quantum Kernel",

        "Qubits":
            N_QUBITS,

        "Threshold":
            threshold,

        "Validation_F1":
            val_f1,

        **metrics,

        "Kernel_Train_Time":
            train_kernel_time,

        "Kernel_Validation_Time":
            val_kernel_time,

        "Kernel_Test_Time":
            test_kernel_time
    }

    print("\n")
    print("=" * 65)
    print("CONTROLLED QUANTUM KERNEL RESULTS")
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
        "results/metrics/"
        "quantum_kernel_controlled.csv",
        index=False
    )

    print(
        "\nSaved controlled results."
    )


if __name__ == "__main__":
    main()