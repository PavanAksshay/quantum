import os
import time
import pandas as pd
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# CONFIGURATION
# ============================================================

PCA_DIR = "results/roberta_pca_features/pca_2"

OUTPUT_DIR = "results/metrics"

N_QUBITS = 2

N_LAYERS = 2

EPOCHS = 40

LEARNING_RATE = 0.05

SEEDS = [
    42,
    123,
    456,
    789,
    999
]


# ============================================================
# LOAD DATA
# ============================================================

X_train = np.load(
    f"{PCA_DIR}/X_train.npy"
)

y_train = np.load(
    f"{PCA_DIR}/y_train.npy"
)

X_val = np.load(
    f"{PCA_DIR}/X_val.npy"
)

y_val = np.load(
    f"{PCA_DIR}/y_val.npy"
)

X_test = np.load(
    f"{PCA_DIR}/X_test.npy"
)

y_test = np.load(
    f"{PCA_DIR}/y_test.npy"
)


# ============================================================
# QUANTUM INPUT SCALING
# ============================================================

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


def scale_features(X):

    X = (
        X - train_min
    ) / denominator

    X = (
        2.0 * np.pi * X
        - np.pi
    )

    return X


X_train_q = scale_features(
    X_train
)

X_val_q = scale_features(
    X_val
)

X_test_q = scale_features(
    X_test
)


# ============================================================
# QUANTUM DEVICE
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


# ============================================================
# CIRCUIT
# ============================================================

@qml.qnode(
    dev,
    interface="autograd"
)
def circuit(
    x,
    weights
):

    # --------------------------------------------------------
    # Data encoding
    # --------------------------------------------------------

    for i in range(
        N_QUBITS
    ):

        qml.RY(
            x[i],
            wires=i
        )

        qml.RZ(
            x[i],
            wires=i
        )

    # --------------------------------------------------------
    # Variational layers
    # --------------------------------------------------------

    for layer in range(
        N_LAYERS
    ):

        for qubit in range(
            N_QUBITS
        ):

            qml.RY(
                weights[
                    layer,
                    qubit,
                    0
                ],
                wires=qubit
            )

            qml.RZ(
                weights[
                    layer,
                    qubit,
                    1
                ],
                wires=qubit
            )

        # Entanglement

        for qubit in range(
            N_QUBITS - 1
        ):

            qml.CNOT(
                wires=[
                    qubit,
                    qubit + 1
                ]
            )

    return qml.expval(
        qml.PauliZ(0)
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict(
    X,
    weights
):

    outputs = []

    for x in X:

        value = circuit(
            x,
            weights
        )

        # Convert [-1,1] → [0,1]

        probability = (
            (value + 1.0)
            / 2.0
        )

        outputs.append(
            probability
        )

    return pnp.stack(
        outputs
    )


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
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
            probabilities
            >= threshold
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
        probabilities
        >= threshold
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
# SINGLE VQC RUN
# ============================================================

def run_vqc(
    seed
):

    print("\n")
    print("=" * 80)

    print(
        f"VQC RUN — SEED {seed}"
    )

    print("=" * 80)

    pnp.random.seed(
        seed
    )

    # --------------------------------------------------------
    # Initialize weights
    # --------------------------------------------------------

    weights = pnp.array(

        0.01
        * np.random.randn(
            N_LAYERS,
            N_QUBITS,
            2
        ),

        requires_grad=True
    )

    optimizer = qml.AdamOptimizer(
        stepsize=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Binary cross entropy
    # --------------------------------------------------------

    def loss_fn(
        weights
    ):

        probabilities = predict(
            X_train_q,
            weights
        )

        losses = []

        for probability, label in zip(
            probabilities,
            y_train
        ):

            probability = pnp.clip(
                probability,
                1e-7,
                1 - 1e-7
            )

            losses.append(

                -(
                    label
                    * pnp.log(
                        probability
                    )
                    +
                    (1 - label)
                    * pnp.log(
                        1 - probability
                    )
                )
            )

        return pnp.mean(
            pnp.stack(
                losses
            )
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    start_time = time.time()

    for epoch in range(
        EPOCHS
    ):

        weights, loss = (
            optimizer.step_and_cost(
                loss_fn,
                weights
            )
        )

        if (
            epoch == 0
            or
            (epoch + 1) % 5 == 0
        ):

            val_probabilities = (
                np.asarray(
                    predict(
                        X_val_q,
                        weights
                    )
                )
            )

            _, val_f1 = (
                find_best_threshold(
                    val_probabilities,
                    y_val
                )
            )

            print(
                f"Epoch "
                f"{epoch + 1:02d}/{EPOCHS}"
                f" | Loss: {float(loss):.6f}"
                f" | Val F1: {val_f1:.4f}"
            )

    training_time = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_probabilities = (
        np.asarray(
            predict(
                X_val_q,
                weights
            )
        )
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

    start_time = time.time()

    test_probabilities = (
        np.asarray(
            predict(
                X_test_q,
                weights
            )
        )
    )

    inference_time = (
        time.time()
        - start_time
    )

    metrics = evaluate(
        test_probabilities,
        y_test,
        threshold
    )

    print("\nFinal test results:")

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

    return {

        "Seed":
            seed,

        "Qubits":
            N_QUBITS,

        "Layers":
            N_LAYERS,

        "Threshold":
            threshold,

        "Validation_F1":
            validation_f1,

        **metrics,

        "Training_Time":
            training_time,

        "Inference_Time":
            inference_time
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print(
        "FINAL ROBERTA + HYBRID VQC"
    )
    print("=" * 90)

    print(
        "\nTraining:",
        X_train_q.shape,
        np.bincount(y_train)
    )

    print(
        "Validation:",
        X_val_q.shape,
        np.bincount(y_val)
    )

    print(
        "Testing:",
        X_test_q.shape,
        np.bincount(y_test)
    )

    results = []

    for seed in SEEDS:

        result = run_vqc(
            seed
        )

        results.append(
            result
        )

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    summary = {

        "Model":
            "RoBERTa + 2-Qubit Hybrid VQC",

        "Qubits":
            N_QUBITS,

        "Layers":
            N_LAYERS,

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

        "Mean_Training_Time":
            results_df[
                "Training_Time"
            ].mean(),

        "Mean_Inference_Time":
            results_df[
                "Inference_Time"
            ].mean()
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results_path = (
        f"{OUTPUT_DIR}/"
        "final_roberta_vqc_runs.csv"
    )

    summary_path = (
        f"{OUTPUT_DIR}/"
        "final_roberta_vqc_summary.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n")
    print("=" * 100)
    print(
        "FINAL ROBERTA VQC SUMMARY"
    )
    print("=" * 100)

    print(
        f"F1:"
        f" {summary['F1_Mean']:.4f}"
        f" +/- "
        f"{summary['F1_Std']:.4f}"
    )

    print(
        f"PR-AUC:"
        f" {summary['PR_AUC_Mean']:.4f}"
        f" +/- "
        f"{summary['PR_AUC_Std']:.4f}"
    )

    print(
        f"ROC-AUC:"
        f" {summary['ROC_AUC_Mean']:.4f}"
        f" +/- "
        f"{summary['ROC_AUC_Std']:.4f}"
    )

    print(
        f"Mean training time:"
        f" {summary['Mean_Training_Time']:.2f}s"
    )

    print(
        "\nSaved:"
    )

    print(
        results_path
    )

    print(
        summary_path
    )


if __name__ == "__main__":

    main()