import os
import time

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModel
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "roberta-base"

MAX_LENGTH = 128

BATCH_SIZE = 16

INPUT_DIR = (
    "results/standard_quantum_split"
)

OUTPUT_DIR = (
    "results/roberta_quantum_features"
)


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
    "Device:",
    DEVICE
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading RoBERTa..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
if tokenizer is None:
    raise ValueError(f"Failed to load tokenizer for {MODEL_NAME}")

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.to(
    DEVICE
)

model.eval()


# ============================================================
# EMBEDDINGS
# ============================================================

def extract_embeddings(
    texts
):

    embeddings = []

    start_time = time.time()

    with torch.no_grad():

        for start in range(
            0,
            len(texts),
            BATCH_SIZE
        ):

            batch = texts[
                start:
                start + BATCH_SIZE
            ]

            encoded = tokenizer(

                batch,

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

            hidden = (
                outputs.last_hidden_state
            )

            attention_mask = (
                encoded[
                    "attention_mask"
                ]
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

            processed = min(
                start + BATCH_SIZE,
                len(texts)
            )

            print(
                f"\rProcessed "
                f"{processed}/{len(texts)}",
                end=""
            )

    print()

    embeddings = np.vstack(
        embeddings
    )

    elapsed = (
        time.time() - start_time
    )

    print(
        "Shape:",
        embeddings.shape
    )

    print(
        f"Time: {elapsed:.2f}s"
    )

    return embeddings


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    filename
):

    path = os.path.join(
        INPUT_DIR,
        filename
    )

    df = pd.read_csv(
        path
    )

    print("\n")
    print("=" * 70)
    print(
        f"PROCESSING {filename}"
    )
    print("=" * 70)

    embeddings = extract_embeddings(
        df["text"].tolist()
    )

    labels = df[
        "target"
    ].to_numpy()

    return (
        embeddings,
        labels
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    X_train, y_train = (
        process_split(
            "train.csv"
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    X_val, y_val = (
        process_split(
            "validation.csv"
        )
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    X_test, y_test = (
        process_split(
            "test.csv"
        )
    )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    np.save(
        f"{OUTPUT_DIR}/X_train.npy",
        X_train
    )

    np.save(
        f"{OUTPUT_DIR}/y_train.npy",
        y_train
    )

    np.save(
        f"{OUTPUT_DIR}/X_val.npy",
        X_val
    )

    np.save(
        f"{OUTPUT_DIR}/y_val.npy",
        y_val
    )

    np.save(
        f"{OUTPUT_DIR}/X_test.npy",
        X_test
    )

    np.save(
        f"{OUTPUT_DIR}/y_test.npy",
        y_test
    )

    print("\n")
    print("=" * 70)
    print("ROBERta FEATURES SAVED")
    print("=" * 70)

    print(
        "Training:",
        X_train.shape
    )

    print(
        "Validation:",
        X_val.shape
    )

    print(
        "Testing:",
        X_test.shape
    )

    print(
        "\nOutput:",
        OUTPUT_DIR
    )


if __name__ == "__main__":

    main()