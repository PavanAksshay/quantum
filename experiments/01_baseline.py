import os
import sys

import pandas as pd


# Allow importing from src/
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


from src.preprocessing import (
    load_dataset,
    split_dataset
)

from src.classical_models import (
    get_models,
    evaluate_model
)


def main():

    # -------------------------
    # Load dataset
    # -------------------------

    df = load_dataset()

    # -------------------------
    # Train/test split
    # -------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_dataset(df)

    print("\n")
    print("=" * 50)
    print("TRAINING CLASSICAL BASELINES")
    print("=" * 50)

    models = get_models()

    results = []

    # -------------------------
    # Train models
    # -------------------------

    for name, model in models.items():

        print(
            f"\nTraining: {name}"
        )

        metrics = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        metrics["Model"] = name

        results.append(metrics)

        print(
            f"F1:       {metrics['F1']:.4f}"
        )

        print(
            f"PR-AUC:   {metrics['PR_AUC']:.4f}"
        )

        print(
            f"ROC-AUC:  {metrics['ROC_AUC']:.4f}"
        )

        print(
            f"Precision: {metrics['Precision']:.4f}"
        )

        print(
            f"Recall:    {metrics['Recall']:.4f}"
        )

    # -------------------------
    # Save results
    # -------------------------

    os.makedirs(
        "results/metrics",
        exist_ok=True
    )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        "results/metrics/baseline.csv",
        index=False
    )

    # -------------------------
    # Display final results
    # -------------------------

    print("\n")
    print("=" * 70)
    print("FINAL BASELINE RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "PR_AUC",
                "ROC_AUC"
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nResults saved to:"
        " results/metrics/baseline.csv"
    )


if __name__ == "__main__":
    main()