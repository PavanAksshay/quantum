import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModel
)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


warnings.filterwarnings(
    "ignore"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

MAX_LENGTH = 128

BATCH_SIZE = 32


MODELS = {

    "DistilBERT":
        "distilbert-base-uncased",

    "BERT":
        "bert-base-uncased",

    "RoBERTa":
        "roberta-base"
}


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device(
        "mps"
    )

elif torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


print(
    "Using device:",
    DEVICE
)


# ============================================================
# DATASET
# ============================================================

def load_dataset():

    path = "data/SMSSpamCollection"

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=[
            "label",
            "text"
        ]
    )

    df["target"] = (
        df["label"]
        .map({
            "ham": 0,
            "spam": 1
        })
    )

    df = df.dropna()

    return df


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

def extract_embeddings(
    texts,
    model_name
):

    print(
        f"\nLoading {model_name}..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
    if tokenizer is None:
        raise ValueError(f"Failed to load tokenizer for {model_name}")

    model = AutoModel.from_pretrained(
        model_name
    )

    model.to(
        DEVICE
    )

    model.eval()

    embeddings = []

    start_time = time.time()

    with torch.no_grad():

        for start in range(
            0,
            len(texts),
            BATCH_SIZE
        ):

            batch_texts = [
                str(t) for t in texts[
                    start:
                    start + BATCH_SIZE
                ]
            ]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(DEVICE)
                for key, value
                in encoded.items()
            }

            outputs = model(
                **encoded
            )

            # ------------------------------------------------
            # Mean pooling
            # ------------------------------------------------

            hidden = (
                outputs.last_hidden_state
            )

            attention_mask = (
                encoded["attention_mask"]
            )

            mask = (
                attention_mask
                .unsqueeze(-1)
                .expand(hidden.size())
                .float()
            )

            summed = (
                torch.sum(
                    hidden * mask,
                    dim=1
                )
            )

            counts = (
                torch.clamp(
                    mask.sum(dim=1),
                    min=1e-9
                )
            )

            pooled = (
                summed / counts
            )

            embeddings.append(
                pooled.cpu().numpy()
            )

            if (
                start == 0
                or
                (start // BATCH_SIZE) % 10 == 0
            ):

                print(
                    f"Processed "
                    f"{min(start + BATCH_SIZE, len(texts))}"
                    f"/{len(texts)}"
                )

    embeddings = np.vstack(
        embeddings
    )

    elapsed = (
        time.time() - start_time
    )

    print(
        f"Embedding shape:"
        f" {embeddings.shape}"
    )

    print(
        f"Embedding time:"
        f" {elapsed:.2f}s"
    )

    return (
        embeddings,
        elapsed
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def evaluate_classifier(
    X_train,
    X_test,
    y_train,
    y_test
):

    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
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

    start = time.time()

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    inference_time = (
        time.time() - start
    )

    return {

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

        "Classifier_Training_Time":
            training_time,

        "Classifier_Inference_Time":
            inference_time
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "TRANSFORMER REPRESENTATION BENCHMARK"
    )
    print("=" * 80)

    df = load_dataset()

    texts = df["text"].to_numpy()

    labels = df["target"].to_numpy()

    (
        train_texts,
        test_texts,
        y_train,
        y_test
    ) = train_test_split(

        texts,

        labels,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=labels
    )

    print(
        "\nDataset:",
        len(df)
    )

    print(
        "Training:",
        len(train_texts)
    )

    print(
        "Testing:",
        len(test_texts)
    )

    results = []

    # ========================================================
    # Each transformer
    # ========================================================

    for name, model_name in MODELS.items():

        print("\n")
        print("=" * 80)
        print(
            f"MODEL: {name}"
        )
        print("=" * 80)

        # ----------------------------------------------------
        # Extract train embeddings
        # ----------------------------------------------------

        X_train, train_embedding_time = (
            extract_embeddings(
                train_texts,
                model_name
            )
        )

        # ----------------------------------------------------
        # Extract test embeddings
        # ----------------------------------------------------

        X_test, test_embedding_time = (
            extract_embeddings(
                test_texts,
                model_name
            )
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        metrics = evaluate_classifier(
            X_train,
            X_test,
            y_train,
            y_test
        )

        result = {

            "Model":
                name,

            "Embedding_Dimension":
                X_train.shape[1],

            "Embedding_Train_Time":
                train_embedding_time,

            "Embedding_Test_Time":
                test_embedding_time,

            **metrics
        }

        results.append(
            result
        )

        print("\nResults:")

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
            f"Embedding dimension:"
            f" {result['Embedding_Dimension']}"
        )

    # ========================================================
    # Save
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "F1",
        ascending=False
    )

    os.makedirs(
        "results/metrics",
        exist_ok=True
    )

    output_path = (
        "results/metrics/"
        "transformer_benchmark.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n")
    print("=" * 100)
    print(
        "TRANSFORMER BENCHMARK RESULTS"
    )
    print("=" * 100)

    print(
        results_df[
            [
                "Model",
                "Embedding_Dimension",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "PR_AUC",
                "ROC_AUC",
                "Embedding_Train_Time"
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