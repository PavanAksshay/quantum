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

from src.fast_quantum_kernel import FastQuantumKernel


# ============================================================
# CONFIGURATION
# ============================================================

PCA_DIR = "results/roberta_pca_features"

OUTPUT_DIR = "results/metrics"

DIMENSIONS = [2, 4, 6, 8]

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_pca_data(dimensions):

    directory = os.path.join(
        PCA_DIR,
        f"pca_{dimensions}"
    )

    X_train = np.load(
        f"{directory}/X_train.npy"
    )

    y_train = np.load(
        f"{directory}/y_train.npy"
    )

    X_val = np.load(
        f"{directory}/X_val.npy"
    )

    y_val = np.load(
        f"{directory}/y_val.npy"
    )

    X_test = np.load(
        f"{directory}/X_test.npy"
    )

    y_test = np.load(
        f"{directory}/y_test.npy"
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# QUANTUM FEATURE SCALING
# ============================================================

def scale_for_quantum(
    X_train,
    X_val,
    X_test
):

    # PCA outputs can have different scales.
    #
    # Scale each feature using TRAINING statistics only.
    #
    # This prevents validation/test leakage.

    train_min = X_train.min(
        axis=0
    )

    train_max = X_train.max(
        axis=0
    )

    denominator = (
        train_max - train_min
    )

    denominator[
        denominator == 0
    ] = 1.0

    def transform(X):

        X = (
            X - train_min
        ) / denominator

        # Map [0,1] → [-pi, pi]

        X = (
            2.0 * np.pi * X
            - np.pi
        )

        return X

    return (
        transform(X_train),
        transform(X_val),
        transform(X_test)
    )


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    probabilities,
    labels
):

    best_threshold = 0.50

    best_f1 = -1.0

    for threshold in np.arange(
        0.20,
        0.81,
        0.01
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            labels,
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


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    probabilities,
    labels,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {

        "Accuracy":
            accuracy_score(
                labels,
                predictions
            ),

        "Precision":
            precision_score(
                labels,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                labels,
                predictions,
                zero_division=0
            ),

        "F1":
            f1_score(
                labels,
                predictions,
                zero_division=0
            ),

        "PR_AUC":
            average_precision_score(
                labels,
                probabilities
            ),

        "ROC_AUC":
            roc_auc_score(
                labels,
                probabilities
            )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "ROBERTA + QUANTUM KERNEL BENCHMARK"
    )
    print("=" * 80)

    results = []

    for dimensions in DIMENSIONS:

        print("\n")
        print("=" * 80)

        print(
            f"DIMENSION = {dimensions}"
        )

        print("=" * 80)

        # ----------------------------------------------------
        # Load PCA data
        # ----------------------------------------------------

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test
        ) = load_pca_data(
            dimensions
        )

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

        # ----------------------------------------------------
        # Scale for quantum angles
        # ----------------------------------------------------

        (
            X_train_q,
            X_val_q,
            X_test_q
        ) = scale_for_quantum(
            X_train,
            X_val,
            X_test
        )

        # ----------------------------------------------------
        # Quantum kernel
        # ----------------------------------------------------

        quantum_kernel = FastQuantumKernel(
            n_qubits=dimensions,
            n_layers=1
        )

        # ----------------------------------------------------
        # Training kernel
        # ----------------------------------------------------

        print(
            "\nComputing training kernel..."
        )

        start = time.time()

        K_train = quantum_kernel.matrix(
            X_train_q
        )

        train_kernel_time = (
            time.time() - start
        )

        print(
            f"Training kernel:"
            f" {train_kernel_time:.3f}s"
        )

        # ----------------------------------------------------
        # Validation kernel
        # ----------------------------------------------------

        print(
            "\nComputing validation kernel..."
        )

        start = time.time()

        K_val = quantum_kernel.matrix(
            X_val_q,
            X_train_q
        )

        val_kernel_time = (
            time.time() - start
        )

        print(
            f"Validation kernel:"
            f" {val_kernel_time:.3f}s"
        )

        # ----------------------------------------------------
        # Test kernel
        # ----------------------------------------------------

        print(
            "\nComputing test kernel..."
        )

        start = time.time()

        K_test = quantum_kernel.matrix(
            X_test_q,
            X_train_q
        )

        test_kernel_time = (
            time.time() - start
        )

        print(
            f"Test kernel:"
            f" {test_kernel_time:.3f}s"
        )

        # ----------------------------------------------------
        # Quantum SVM
        # ----------------------------------------------------

        print(
            "\nTraining quantum-kernel SVM..."
        )

        model = SVC(

            kernel="precomputed",

            class_weight="balanced",

            probability=True,

            random_state=RANDOM_STATE
        )

        start = time.time()

        model.fit(
            K_train,
            y_train
        )

        svm_training_time = (
            time.time() - start
        )

        # ----------------------------------------------------
        # Validation threshold
        # ----------------------------------------------------

        val_probabilities = (
            model.predict_proba(
                K_val
            )[:, 1]
        )

        (
            threshold,
            validation_f1
        ) = find_best_threshold(
            val_probabilities,
            y_val
        )

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

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

        total_kernel_time = (
            train_kernel_time
            + val_kernel_time
            + test_kernel_time
        )

        result = {

            "Model":
                "RoBERTa + Quantum Kernel",

            "Qubits":
                dimensions,

            "Threshold":
                threshold,

            "Validation_F1":
                validation_f1,

            "Accuracy":
                metrics["Accuracy"],

            "Precision":
                metrics["Precision"],

            "Recall":
                metrics["Recall"],

            "F1":
                metrics["F1"],

            "PR_AUC":
                metrics["PR_AUC"],

            "ROC_AUC":
                metrics["ROC_AUC"],

            "Train_Kernel_Time":
                train_kernel_time,

            "Validation_Kernel_Time":
                val_kernel_time,

            "Test_Kernel_Time":
                test_kernel_time,

            "Total_Kernel_Time":
                total_kernel_time,

            "SVM_Training_Time":
                svm_training_time
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print("\nResults:")

        print(
            f"F1:       "
            f"{metrics['F1']:.4f}"
        )

        print(
            f"PR-AUC:   "
            f"{metrics['PR_AUC']:.4f}"
        )

        print(
            f"ROC-AUC:  "
            f"{metrics['ROC_AUC']:.4f}"
        )

        print(
            f"Threshold:"
            f" {threshold:.2f}"
        )

        print(
            f"Validation F1:"
            f" {validation_f1:.4f}"
        )

        print(
            f"Total kernel time:"
            f" {total_kernel_time:.3f}s"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_path = (
        f"{OUTPUT_DIR}/"
        "roberta_quantum_kernel.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print("\n")
    print("=" * 100)
    print(
        "ROBERTA QUANTUM KERNEL RESULTS"
    )
    print("=" * 100)

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
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":

    main()