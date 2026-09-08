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


from src.fast_quantum_kernel import (
    FastQuantumKernel
)

from src.quantum_utils import (
    prepare_quantum_features
)

from src.quantum_benchmark import (
    load_quantum_split
)


# ==========================================
# Experiment configuration
# ==========================================

QUBIT_COUNTS = [
    2,
    4,
    6,
    8
]


def find_threshold(
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

    results = []

    # ==========================================
    # Run each qubit configuration
    # ==========================================

    for n_qubits in QUBIT_COUNTS:

        print("\n")
        print("=" * 70)
        print(
            f"RUNNING {n_qubits}-QUBIT EXPERIMENT"
        )
        print("=" * 70)

        # --------------------------------------
        # Prepare features
        # --------------------------------------

        train_features = (
            prepare_quantum_features(
                X_train,
                n_qubits
            )
        )

        val_features = (
            prepare_quantum_features(
                X_val,
                n_qubits
            )
        )

        test_features = (
            prepare_quantum_features(
                X_test,
                n_qubits
            )
        )

        print(
            "Feature shape:",
            train_features.shape
        )

        # --------------------------------------
        # Create kernel
        # --------------------------------------

        kernel = FastQuantumKernel(
            n_qubits=n_qubits,
            n_layers=1
        )

        # --------------------------------------
        # Training kernel
        # --------------------------------------

        print(
            "\nComputing training kernel..."
        )

        start = time.time()

        K_train = kernel.matrix(
            train_features
        )

        train_time = (
            time.time() - start
        )

        print(
            f"Training kernel: "
            f"{train_time:.3f}s"
        )

        # --------------------------------------
        # Validation kernel
        # --------------------------------------

        print(
            "\nComputing validation kernel..."
        )

        start = time.time()

        K_val = kernel.matrix(
            val_features,
            train_features
        )

        val_time = (
            time.time() - start
        )

        print(
            f"Validation kernel: "
            f"{val_time:.3f}s"
        )

        # --------------------------------------
        # Test kernel
        # --------------------------------------

        print(
            "\nComputing test kernel..."
        )

        start = time.time()

        K_test = kernel.matrix(
            test_features,
            train_features
        )

        test_time = (
            time.time() - start
        )

        print(
            f"Test kernel: "
            f"{test_time:.3f}s"
        )

        # --------------------------------------
        # Train SVM
        # --------------------------------------

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

        # --------------------------------------
        # Validation threshold
        # --------------------------------------

        val_probabilities = (
            model.predict_proba(
                K_val
            )[:, 1]
        )

        threshold, val_f1 = (
            find_threshold(
                val_probabilities,
                y_val
            )
        )

        # --------------------------------------
        # Test
        # --------------------------------------

        test_probabilities = (
            model.predict_proba(
                K_test
            )[:, 1]
        )

        predictions = (
            test_probabilities >= threshold
        ).astype(int)

        # --------------------------------------
        # Metrics
        # --------------------------------------

        result = {

            "Qubits":
                n_qubits,

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
                    test_probabilities
                ),

            "ROC_AUC":
                roc_auc_score(
                    y_test,
                    test_probabilities
                ),

            "Threshold":
                threshold,

            "Validation_F1":
                val_f1,

            "Train_Kernel_Time":
                train_time,

            "Validation_Kernel_Time":
                val_time,

            "Test_Kernel_Time":
                test_time,

            "Total_Kernel_Time":
                train_time
                + val_time
                + test_time
        }

        results.append(
            result
        )

        # --------------------------------------
        # Print
        # --------------------------------------

        print("\nResults:")

        print(
            f"F1:       {result['F1']:.4f}"
        )

        print(
            f"PR-AUC:   {result['PR_AUC']:.4f}"
        )

        print(
            f"ROC-AUC:  {result['ROC_AUC']:.4f}"
        )

        print(
            f"Threshold: {result['Threshold']:.2f}"
        )

        print(
            f"Total kernel time: "
            f"{result['Total_Kernel_Time']:.3f}s"
        )

    # ==========================================
    # Save
    # ==========================================

    results_df = pd.DataFrame(
        results
    )

    os.makedirs(
        "results/metrics",
        exist_ok=True
    )

    results_df.to_csv(
        "results/metrics/"
        "qubit_scaling.csv",
        index=False
    )

    print("\n")
    print("=" * 80)
    print("QUBIT SCALING RESULTS")
    print("=" * 80)

    print(
        results_df[
            [
                "Qubits",
                "F1",
                "PR_AUC",
                "ROC_AUC",
                "Validation_F1",
                "Total_Kernel_Time"
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
        " results/metrics/qubit_scaling.csv"
    )


if __name__ == "__main__":
    main()