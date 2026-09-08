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

sys.path.append(PROJECT_ROOT)

from src.quantum_model import HybridVQC

from src.quantum_benchmark import (
    load_quantum_split
)


# ==========================================
# Configuration
# ==========================================

N_QUBITS = 4
N_LAYERS = 2

EPOCHS = 30
BATCH_SIZE = 16

LEARNING_RATE = 0.01

PATIENCE = 6

SEEDS = [
    42,
    123,
    456
]


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


def main():

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

    all_results = []

    # ==========================================
    # Multiple seeds
    # ==========================================

    for seed in SEEDS:

        print("\n")
        print("=" * 70)
        print(
            f"VQC RUN — SEED {seed}"
        )
        print("=" * 70)

        model = HybridVQC(
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            seed=seed
        )

        # --------------------------------------
        # Train
        # --------------------------------------

        start = time.time()

        model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            patience=PATIENCE
        )

        training_time = (
            time.time() - start
        )

        # --------------------------------------
        # Validation
        # --------------------------------------

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

        # --------------------------------------
        # Test
        # --------------------------------------

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

        result = {

            "Model":
                "DistilBERT + Hybrid VQC",

            "Seed":
                seed,

            "Qubits":
                N_QUBITS,

            "Layers":
                N_LAYERS,

            "Threshold":
                threshold,

            "Validation_F1":
                val_f1,

            "Training_Time":
                training_time,

            "Inference_Time":
                inference_time,

            **metrics
        }

        all_results.append(
            result
        )

        # --------------------------------------
        # Print
        # --------------------------------------

        print("\nFinal test results:")

        print(
            f"F1:       "
            f"{result['F1']:.4f}"
        )

        print(
            f"PR-AUC:   "
            f"{result['PR_AUC']:.4f}"
        )

        print(
            f"ROC-AUC:  "
            f"{result['ROC_AUC']:.4f}"
        )

        print(
            f"Threshold:"
            f" {result['Threshold']:.2f}"
        )

        print(
            f"Training:"
            f" {result['Training_Time']:.2f}s"
        )

    # ==========================================
    # Results dataframe
    # ==========================================

    results_df = pd.DataFrame(
        all_results
    )

    # ==========================================
    # Summary
    # ==========================================

    summary = {

        "Model":
            "DistilBERT + Hybrid VQC",

        "Qubits":
            N_QUBITS,

        "Layers":
            N_LAYERS,

        "Seeds":
            len(SEEDS),

        "F1_Mean":
            results_df["F1"].mean(),

        "F1_Std":
            results_df["F1"].std(),

        "PR_AUC_Mean":
            results_df["PR_AUC"].mean(),

        "PR_AUC_Std":
            results_df["PR_AUC"].std(),

        "ROC_AUC_Mean":
            results_df["ROC_AUC"].mean(),

        "ROC_AUC_Std":
            results_df["ROC_AUC"].std(),

        "Training_Time_Mean":
            results_df[
                "Training_Time"
            ].mean()
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    # ==========================================
    # Print summary
    # ==========================================

    print("\n")
    print("=" * 75)
    print("FINAL HYBRID VQC SUMMARY")
    print("=" * 75)

    print(
        f"F1: "
        f"{summary['F1_Mean']:.4f} "
        f"+/- "
        f"{summary['F1_Std']:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{summary['PR_AUC_Mean']:.4f} "
        f"+/- "
        f"{summary['PR_AUC_Std']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{summary['ROC_AUC_Mean']:.4f} "
        f"+/- "
        f"{summary['ROC_AUC_Std']:.4f}"
    )

    print(
        f"Mean training time: "
        f"{summary['Training_Time_Mean']:.2f}s"
    )

    # ==========================================
    # Save
    # ==========================================

    os.makedirs(
        "results/metrics",
        exist_ok=True
    )

    results_df.to_csv(
        "results/metrics/"
        "final_vqc_runs.csv",
        index=False
    )

    summary_df.to_csv(
        "results/metrics/"
        "final_vqc_summary.csv",
        index=False
    )

    # Save training history from last run

    if hasattr(model, "history"):

        pd.DataFrame(
            model.history
        ).to_csv(
            "results/metrics/"
            "vqc_training_history.csv",
            index=False
        )

    print("\nSaved:")

    print(
        "results/metrics/final_vqc_runs.csv"
    )

    print(
        "results/metrics/final_vqc_summary.csv"
    )

    print(
        "results/metrics/vqc_training_history.csv"
    )


if __name__ == "__main__":
    main()