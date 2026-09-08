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


DIMENSIONS = [2, 4, 6, 8]


def find_best_threshold(
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


def evaluate(
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


def run_classical(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    dimensions
):

    X_train = X_train[:, :dimensions]
    X_val = X_val[:, :dimensions]
    X_test = X_test[:, :dimensions]

    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    )

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    train_time = (
        time.time() - start
    )

    val_probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    threshold, val_f1 = (
        find_best_threshold(
            val_probabilities,
            y_val
        )
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    metrics = evaluate(
        test_probabilities,
        y_test,
        threshold
    )

    return {
        "Model": "Classical RBF-SVM",
        "Dimensions": dimensions,
        "Threshold": threshold,
        "Validation_F1": val_f1,
        "Training_Time": train_time,
        **metrics
    }


def run_quantum(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    dimensions
):

    train_features = (
        prepare_quantum_features(
            X_train,
            dimensions
        )
    )

    val_features = (
        prepare_quantum_features(
            X_val,
            dimensions
        )
    )

    test_features = (
        prepare_quantum_features(
            X_test,
            dimensions
        )
    )

    kernel = FastQuantumKernel(
        n_qubits=dimensions,
        n_layers=1
    )

    # ------------------------------------------
    # Training kernel
    # ------------------------------------------

    start = time.time()

    K_train = kernel.matrix(
        train_features
    )

    train_kernel_time = (
        time.time() - start
    )

    # ------------------------------------------
    # Validation kernel
    # ------------------------------------------

    start = time.time()

    K_val = kernel.matrix(
        val_features,
        train_features
    )

    val_kernel_time = (
        time.time() - start
    )

    # ------------------------------------------
    # Test kernel
    # ------------------------------------------

    start = time.time()

    K_test = kernel.matrix(
        test_features,
        train_features
    )

    test_kernel_time = (
        time.time() - start
    )

    # ------------------------------------------
    # SVM
    # ------------------------------------------

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

    # ------------------------------------------
    # Validation
    # ------------------------------------------

    val_probabilities = (
        model.predict_proba(
            K_val
        )[:, 1]
    )

    threshold, val_f1 = (
        find_best_threshold(
            val_probabilities,
            y_val
        )
    )

    # ------------------------------------------
    # Test
    # ------------------------------------------

    test_probabilities = (
        model.predict_proba(
            K_test
        )[:, 1]
    )

    metrics = evaluate(
        test_probabilities,
        y_test,
        threshold
    )

    return {
        "Model": "Quantum Kernel SVM",
        "Dimensions": dimensions,
        "Threshold": threshold,
        "Validation_F1": val_f1,
        "Training_Time":
            train_kernel_time
            + val_kernel_time
            + test_kernel_time,
        "Kernel_Train_Time":
            train_kernel_time,
        "Kernel_Validation_Time":
            val_kernel_time,
        "Kernel_Test_Time":
            test_kernel_time,
        **metrics
    }


def main():

    # ==========================================
    # Load controlled benchmark
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
    # Run experiments
    # ==========================================

    for dimensions in DIMENSIONS:

        print("\n")
        print("=" * 70)
        print(
            f"DIMENSION = {dimensions}"
        )
        print("=" * 70)

        # --------------------------------------
        # Classical
        # --------------------------------------

        print(
            "\nRunning classical RBF-SVM..."
        )

        classical_result = run_classical(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            dimensions
        )

        results.append(
            classical_result
        )

        print(
            f"Classical F1: "
            f"{classical_result['F1']:.4f}"
        )

        print(
            f"Classical PR-AUC: "
            f"{classical_result['PR_AUC']:.4f}"
        )

        # --------------------------------------
        # Quantum
        # --------------------------------------

        print(
            "\nRunning quantum kernel..."
        )

        quantum_result = run_quantum(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            dimensions
        )

        results.append(
            quantum_result
        )

        print(
            f"Quantum F1: "
            f"{quantum_result['F1']:.4f}"
        )

        print(
            f"Quantum PR-AUC: "
            f"{quantum_result['PR_AUC']:.4f}"
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

    output_path = (
        "results/metrics/"
        "classical_vs_quantum.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    # ==========================================
    # Display
    # ==========================================

    print("\n")
    print("=" * 90)
    print(
        "CLASSICAL VS QUANTUM RESULTS"
    )
    print("=" * 90)

    print(
        results_df[
            [
                "Model",
                "Dimensions",
                "F1",
                "PR_AUC",
                "ROC_AUC",
                "Validation_F1",
                "Training_Time"
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()