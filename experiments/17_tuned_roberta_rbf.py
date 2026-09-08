import os
import time

import numpy as np
import pandas as pd

from typing import Literal

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


PCA_DIR = "results/roberta_pca_features"

OUTPUT_DIR = "results/metrics"

DIMENSIONS = [2, 4, 6, 8]

C_VALUES = [
    0.1,
    1.0,
    10.0,
    100.0
]

GAMMA_VALUES: list[Literal["scale", "auto"] | float] = [
    "scale",
    0.1,
    1.0,
    10.0
]


def load_data(dimensions):

    directory = os.path.join(
        PCA_DIR,
        f"pca_{dimensions}"
    )

    return (
        np.load(
            f"{directory}/X_train.npy"
        ),
        np.load(
            f"{directory}/y_train.npy"
        ),
        np.load(
            f"{directory}/X_val.npy"
        ),
        np.load(
            f"{directory}/y_val.npy"
        ),
        np.load(
            f"{directory}/X_test.npy"
        ),
        np.load(
            f"{directory}/y_test.npy"
        )
    )


def scale_features(
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


def find_threshold(
    probabilities,
    labels
):

    best_threshold = 0.50

    best_f1 = -1

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


def main():

    print("=" * 90)
    print(
        "TUNED CLASSICAL RBF-SVM CONTROL"
    )
    print("=" * 90)

    results = []

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

        (
            X_train,
            X_val,
            X_test
        ) = scale_features(
            X_train,
            X_val,
            X_test
        )

        best_model = None

        best_validation_f1 = -1

        best_C = None

        best_gamma = None

        # ====================================================
        # HYPERPARAMETER SEARCH
        # ====================================================

        for C in C_VALUES:

            for gamma in GAMMA_VALUES:

                model = SVC(

                    kernel="rbf",

                    C=C,

                    gamma=gamma,

                    class_weight="balanced",

                    probability=True,

                    random_state=42
                )

                model.fit(
                    X_train,
                    y_train
                )

                val_probabilities = (
                    model.predict_proba(
                        X_val
                    )[:, 1]
                )

                (
                    threshold,
                    validation_f1
                ) = find_threshold(
                    val_probabilities,
                    y_val
                )

                print(
                    f"C={C:<6} "
                    f"gamma={str(gamma):<7} "
                    f"Val F1={validation_f1:.4f}"
                )

                if (
                    validation_f1
                    >
                    best_validation_f1
                ):

                    best_validation_f1 = (
                        validation_f1
                    )

                    best_model = model

                    best_C = C

                    best_gamma = gamma

        # ====================================================
        # FINAL TEST
        # ====================================================

        if best_model is None:
            raise RuntimeError(
                f"Hyperparameter tuning failed to find a valid model for dimension {dimensions}."
            )

        val_probabilities = (
            best_model.predict_proba(
                X_val
            )[:, 1]
        )

        (
            threshold,
            validation_f1
        ) = find_threshold(
            val_probabilities,
            y_val
        )

        test_probabilities = (
            best_model.predict_proba(
                X_test
            )[:, 1]
        )

        metrics = evaluate(
            test_probabilities,
            y_test,
            threshold
        )

        result = {

            "Model":
                "Tuned Classical RBF-SVM",

            "Dimensions":
                dimensions,

            "Best_C":
                best_C,

            "Best_Gamma":
                best_gamma,

            "Threshold":
                threshold,

            "Validation_F1":
                validation_f1,

            **metrics
        }

        results.append(
            result
        )

        print("\nBest configuration:")

        print(
            f"C:       {best_C}"
        )

        print(
            f"Gamma:   {best_gamma}"
        )

        print(
            f"Threshold: {threshold:.2f}"
        )

        print(
            f"Test F1: "
            f"{metrics['F1']:.4f}"
        )

        print(
            f"Test PR-AUC: "
            f"{metrics['PR_AUC']:.4f}"
        )

        print(
            f"Test ROC-AUC: "
            f"{metrics['ROC_AUC']:.4f}"
        )

    # ========================================================
    # SAVE
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
        "tuned_roberta_rbf.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n")
    print("=" * 100)
    print(
        "TUNED RBF RESULTS"
    )
    print("=" * 100)

    print(
        results_df[
            [
                "Dimensions",
                "Best_C",
                "Best_Gamma",
                "F1",
                "PR_AUC",
                "ROC_AUC",
                "Validation_F1"
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