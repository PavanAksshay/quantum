import os
import pandas as pd

from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def load_dataset(
    path="data/SMSSpamCollection"
):
    """
    Load the UCI SMS Spam Collection dataset.

    Format:
        ham<TAB>message
        spam<TAB>message
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Make sure SMSSpamCollection is inside the data folder."
        )

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8"
    )

    # Keep only required columns
    df = df[["label", "text"]]

    # Remove missing values
    df = df.dropna()

    # Clean text
    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    # Remove empty messages
    df = df[
        df["text"].str.len() > 0
    ]

    # Remove duplicate messages
    df = df.drop_duplicates(
        subset=["text"]
    )

    # Convert labels
    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    # Remove anything that wasn't mapped
    df = df.dropna(
        subset=["label"]
    )

    df["label"] = df["label"].astype(int)

    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)

    print(
        f"Number of messages: {len(df)}"
    )

    print("\nClass distribution:")

    print(
        df["label"]
        .value_counts()
        .rename(
            index={
                0: "Ham / Legitimate",
                1: "Spam"
            }
        )
    )

    print("\nClass percentages:")

    print(
        (
            df["label"]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    return df


def split_dataset(df):

    X = df["text"]
    y = df["label"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )


if __name__ == "__main__":

    df = load_dataset()

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_dataset(df)

    print("\n" + "=" * 50)
    print("TRAIN / TEST SPLIT")
    print("=" * 50)

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples:  {len(X_test)}"
    )

    print("\nExample message:")

    print(
        X_train.iloc[0]
    )

    print(
        "Label:",
        y_train.iloc[0]
    )
    