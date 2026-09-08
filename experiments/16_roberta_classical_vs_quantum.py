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
# LOAD PCA DATA
# ============================================================

def load_data(dimensions):

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
# SCALE FEATURES FOR RBF SVM
# ============================================================

def scale_classical_features(
    X_train,
    X_val,
    X_test
):

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

        return (
            X - train_min
        ) / denominator

    return (
        transform(X_train),
        transform(X_val),
        transform(X_test)
    )


# ============================================================
# SCALE FEATURES FOR QUANTUM CIRCUIT
# ============================================================

def scale_quantum_features(
    X_train,
    X_val,
    X_test
):

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

        # [0, 1] → [-π, π]

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
# CLASSICAL RBF-SVM
# ============================================================

def run_classical(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):

    print(
        "\nRunning classical RBF-SVM..."
    )

    (
        X_train,
        X_val,
        X_test
    ) = scale_classical_features(
        X_train,
        X_val,
        X_test
    )

    model = SVC(

        kernel="rbf",

        C=2.0,

        gamma="scale",

        class_weight="balanced",

        probability=True,

        random_state=RANDOM_STATE
    )

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.time() - start
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    (
        threshold,
        validation_f1
    ) = find_best_threshold(
        val_probabilities,
        y_val
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    start = time.time()

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    inference_time = (
        time.time() - start
    )

    metrics = evaluate(
        test_probabilities,
        y_test,
        threshold
    )

    return {

        "Model":
            "Classical RBF-SVM",

        "Threshold":
            threshold,

        "Validation_F1":
            validation_f1,

        **metrics,

        "Training_Time":
            training_time,

        "Inference_Time":
            inference_time,

        "Kernel_Time":
            0.0
    }


# ============================================================
# QUANTUM KERNEL
# ============================================================

def run_quantum(
    dimensions,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):

    print(
        "\nRunning quantum kernel..."
    )

    (
        X_train,
        X_val,
        X_test
    ) = scale_quantum_features(
        X_train,
        X_val,
        X_test
    )

    quantum_kernel = FastQuantumKernel(

        n_qubits=dimensions,

        n_layers=1
    )

    # --------------------------------------------------------
    # Training kernel
    # --------------------------------------------------------

    start = time.time()

    K_train = quantum_kernel.matrix(
        X_train
    )

    train_kernel_time = (
        time.time() - start
    )

    # --------------------------------------------------------
    # Validation kernel
    # --------------------------------------------------------

    start = time.time()

    K_val = quantum_kernel.matrix(
        X_val,
        X_train
    )

    val_kernel_time = (
        time.time() - start
    )

    # --------------------------------------------------------
    # Test kernel
    # --------------------------------------------------------

    start = time.time()

    K_test = quantum_kernel.matrix(
        X_test,
        X_train
    )

    test_kernel_time = (
        time.time() - start
    )

    total_kernel_time = (
        train_kernel_time
        + val_kernel_time
        + test_kernel_time
    )

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

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

    training_time = (
        time.time() - start
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    start = time.time()

    test_probabilities = (
        model.predict_proba(
            K_test
        )[:, 1]
    )

    inference_time = (
        time.time() - start
    )

    metrics = evaluate(
        test_probabilities,
        y_test,
        threshold
    )

    return {

        "Model":
            "Quantum Kernel SVM",

        "Threshold":
            threshold,

        "Validation_F1":
            validation_f1,

        **metrics,

        "Training_Time":
            training_time,

        "Inference_Time":
            inference_time,

        "Kernel_Time":
            total_kernel_time
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print(
        "ROBERTA: CLASSICAL VS QUANTUM KERNEL"
    )
    print("=" * 90)

    all_results = []

    for dimensions in DIMENSIONS:

        print("\n")
        print("=" * 90)

        print(
            f"DIMENSION = {dimensions}"
        )

        print("=" * 90)

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test
        ) = load_data(
            dimensions
        )

        print(
            "\nTraining:",
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

        # ====================================================
        # CLASSICAL
        # ====================================================

        classical_result = run_classical(

            X_train,
            y_train,

            X_val,
            y_val,

            X_test,
            y_test
        )

        classical_result[
            "Dimensions"
        ] = dimensions

        all_results.append(
            classical_result
        )

        print(
            f"\nClassical F1:"
            f" {classical_result['F1']:.4f}"
        )

        print(
            f"Classical PR-AUC:"
            f" {classical_result['PR_AUC']:.4f}"
        )

        print(
            f"Classical ROC-AUC:"
            f" {classical_result['ROC_AUC']:.4f}"
        )

        # ====================================================
        # QUANTUM
        # ====================================================

        quantum_result = run_quantum(

            dimensions,

            X_train,
            y_train,

            X_val,
            y_val,

            X_test,
            y_test
        )

        quantum_result[
            "Dimensions"
        ] = dimensions

        all_results.append(
            quantum_result
        )

        print(
            f"\nQuantum F1:"
            f" {quantum_result['F1']:.4f}"
        )

        print(
            f"Quantum PR-AUC:"
            f" {quantum_result['PR_AUC']:.4f}"
        )

        print(
            f"Quantum ROC-AUC:"
            f" {quantum_result['ROC_AUC']:.4f}"
        )

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------
    # Add performance gap
    # --------------------------------------------------------

    classical = (
        results_df[
            results_df["Model"]
            == "Classical RBF-SVM"
        ]
        .set_index("Dimensions")
        .reindex(DIMENSIONS)
    )

    quantum = (
        results_df[
            results_df["Model"]
            == "Quantum Kernel SVM"
        ]
        .set_index("Dimensions")
        .reindex(DIMENSIONS)
    )

    gap_df = pd.DataFrame({
        "Dimensions": DIMENSIONS,
        "Classical_F1": classical["F1"].to_numpy(),
        "Quantum_F1": quantum["F1"].to_numpy(),
        "F1_Gap_Quantum_Minus_Classical": (
            quantum["F1"].to_numpy() - classical["F1"].to_numpy()
        ),
        "Classical_PR_AUC": classical["PR_AUC"].to_numpy(),
        "Quantum_PR_AUC": quantum["PR_AUC"].to_numpy(),
        "PR_AUC_Gap_Quantum_Minus_Classical": (
            quantum["PR_AUC"].to_numpy() - classical["PR_AUC"].to_numpy()
        ),
        "Classical_ROC_AUC": classical["ROC_AUC"].to_numpy(),
        "Quantum_ROC_AUC": quantum["ROC_AUC"].to_numpy(),
        "Quantum_Kernel_Time": quantum["Kernel_Time"].to_numpy(),
    })

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    full_path = (
        f"{OUTPUT_DIR}/"
        "roberta_classical_vs_quantum.csv"
    )

    gap_path = (
        f"{OUTPUT_DIR}/"
        "roberta_quantum_gap.csv"
    )

    results_df.to_csv(
        full_path,
        index=False
    )

    gap_df.to_csv(
        gap_path,
        index=False
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n")
    print("=" * 100)
    print(
        "CLASSICAL VS QUANTUM SUMMARY"
    )
    print("=" * 100)

    print(
        gap_df.to_string(
            index=False
        )
    )

    print("\n")
    print(
        f"Saved: {full_path}"
    )

    print(
        f"Saved: {gap_path}"
    )


if __name__ == "__main__":

    main()