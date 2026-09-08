import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TRAIN_SIZE = 300
VAL_SIZE = 100
TEST_SIZE = 150

DATASET_PATH = "data/SMSSpamCollection"

OUTPUT_DIR = "results/standard_quantum_split"


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH,
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

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# CREATE SPLIT
# ============================================================

def create_quantum_split(df):

    # --------------------------------------------------------
    # First select 550 samples
    # --------------------------------------------------------

    selected, _ = train_test_split(

        df,

        train_size=(
            TRAIN_SIZE
            + VAL_SIZE
            + TEST_SIZE
        ),

        random_state=RANDOM_STATE,

        stratify=df["target"]
    )

    # --------------------------------------------------------
    # 300 train / 250 temporary
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(

        selected,

        train_size=TRAIN_SIZE,

        random_state=RANDOM_STATE,

        stratify=selected["target"]
    )

    # --------------------------------------------------------
    # 100 validation / 150 test
    # --------------------------------------------------------

    val_df, test_df = train_test_split(

        temp_df,

        train_size=VAL_SIZE,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=temp_df["target"]
    )

    return (
        train_df,
        val_df,
        test_df
    )


# ============================================================
# SAVE
# ============================================================

def save_split(
    df,
    filename
):

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    df.to_csv(
        path,
        index=False
    )

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "STANDARDIZED QUANTUM DATASET"
    )
    print("=" * 70)

    df = load_dataset()

    print(
        "\nFull dataset:",
        df.shape
    )

    print(
        "\nFull distribution:"
    )

    print(
        df["target"]
        .value_counts()
        .sort_index()
    )

    train_df, val_df, test_df = (
        create_quantum_split(df)
    )

    # --------------------------------------------------------
    # Display distributions
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("QUANTUM BENCHMARK SPLIT")
    print("=" * 70)

    print(
        "\nTraining:",
        train_df.shape
    )

    print(
        "Training distribution:",
        np.bincount(
            train_df["target"]
        )
    )

    print(
        "\nValidation:",
        val_df.shape
    )

    print(
        "Validation distribution:",
        np.bincount(
            val_df["target"]
        )
    )

    print(
        "\nTesting:",
        test_df.shape
    )

    print(
        "Testing distribution:",
        np.bincount(
            test_df["target"]
        )
    )

    # --------------------------------------------------------
    # Check overlap
    # --------------------------------------------------------

    train_ids = set(
        train_df.index
    )

    val_ids = set(
        val_df.index
    )

    test_ids = set(
        test_df.index
    )

    print("\n")
    print(
        "Overlap checks:"
    )

    print(
        "Train ∩ Validation:",
        len(train_ids & val_ids)
    )

    print(
        "Train ∩ Test:",
        len(train_ids & test_ids)
    )

    print(
        "Validation ∩ Test:",
        len(val_ids & test_ids)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    train_path = save_split(
        train_df,
        "train.csv"
    )

    val_path = save_split(
        val_df,
        "validation.csv"
    )

    test_path = save_split(
        test_df,
        "test.csv"
    )

    print("\n")
    print(
        "Saved:"
    )

    print(
        train_path
    )

    print(
        val_path
    )

    print(
        test_path
    )


if __name__ == "__main__":

    main()