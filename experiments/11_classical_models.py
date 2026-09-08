import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

MAX_FEATURES = 30000

NGRAM_RANGE = (1, 2)

MIN_DF = 2


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(
    PROJECT_ROOT
)


# ============================================================
# DATA LOADING
# ============================================================

def find_dataset():

    possible_paths = [

        "data/spam.csv",

        "data/dataset.csv",

        "data/spam_dataset.csv",

        "data/SMSSpamCollection",

        "data/messages.csv",

        "data/raw/spam.csv",

        "data/raw/dataset.csv",

    ]

    for path in possible_paths:

        if os.path.exists(path):

            return path

    raise FileNotFoundError(
        "\nCould not find the dataset.\n"
        "Update the `find_dataset()` function "
        "with the path to your dataset."
    )


def load_dataset():

    path = find_dataset()

    print(
        f"Loading dataset: {path}"
    )

    # --------------------------------------------------------
    # Handle tab-separated SMS Spam Collection
    # --------------------------------------------------------

    if path.endswith(
        "SMSSpamCollection"
    ):

        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=[
                "label",
                "text"
            ]
        )

    else:

        df = pd.read_csv(
            path
        )

    # --------------------------------------------------------
    # Automatically detect columns
    # --------------------------------------------------------

    text_column = None
    label_column = None

    for column in df.columns:

        column_lower = str(
            column
        ).lower()

        if column_lower in [
            "text",
            "message",
            "sms",
            "content"
        ]:

            text_column = column

        if column_lower in [
            "label",
            "category",
            "class",
            "target"
        ]:

            label_column = column

    if text_column is None:

        raise ValueError(
            "Could not automatically find "
            "the text column."
        )

    if label_column is None:

        raise ValueError(
            "Could not automatically find "
            "the label column."
        )

    df = df[
        [
            label_column,
            text_column
        ]
    ].copy()

    df.columns = [
        "label",
        "text"
    ]

    # Remove missing values

    df = df.dropna()

    df["text"] = (
        df["text"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Normalize labels
    # --------------------------------------------------------

    def convert_label(label):

        label = str(
            label
        ).strip().lower()

        if label in [
            "spam",
            "1",
            "phishing",
            "scam"
        ]:

            return 1

        if label in [
            "ham",
            "0",
            "legitimate",
            "normal"
        ]:

            return 0

        raise ValueError(
            f"Unknown label: {label}"
        )

    df["target"] = (
        df["label"]
        .apply(convert_label)
    )

    print(
        "\nDataset shape:",
        df.shape
    )

    print(
        "\nClass distribution:"
    )

    print(
        df["target"]
        .value_counts()
        .sort_index()
    )

    return df


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def create_split(df):

    from sklearn.model_selection import train_test_split

    X = df["text"].values

    y = df["target"].values

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print(
        "\nTraining:",
        len(X_train)
    )

    print(
        "Testing:",
        len(X_test)
    )

    print(
        "Training distribution:",
        np.bincount(y_train)
    )

    print(
        "Testing distribution:",
        np.bincount(y_test)
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# METRICS
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Get probabilities / scores
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

    elif hasattr(
        model,
        "decision_function"
    ):

        probabilities = (
            model.decision_function(
                X_test
            )
        )

    else:

        probabilities = predictions

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
            )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EXPANDED CLASSICAL NLP BENCHMARK")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_dataset()

    (
        X_train_text,
        X_test_text,
        y_train,
        y_test
    ) = create_split(
        df
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BUILDING TF-IDF REPRESENTATION")
    print("=" * 70)

    vectorizer = TfidfVectorizer(

        lowercase=True,

        strip_accents="unicode",

        sublinear_tf=True,

        max_features=MAX_FEATURES,

        ngram_range=NGRAM_RANGE,

        min_df=MIN_DF
    )

    start = time.time()

    X_train = vectorizer.fit_transform(
        X_train_text
    )

    X_test = vectorizer.transform(
        X_test_text
    )

    tfidf_time = (
        time.time() - start
    )

    print(
        "Training shape:",
        X_train.shape
    )

    print(
        "Testing shape:",
        X_test.shape
    )

    print(
        f"TF-IDF time: "
        f"{tfidf_time:.3f}s"
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "Naive Bayes":
            MultinomialNB(
                alpha=0.1
            ),

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                C=2.0,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),

        "Linear SVM":
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),

        "RBF SVM":
            SVC(
                kernel="rbf",
                C=2.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "MLP":
            MLPClassifier(
                hidden_layer_sizes=(
                    128,
                    64
                ),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=64,
                learning_rate_init=0.001,
                max_iter=50,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5,
                random_state=RANDOM_STATE
            )
    }

    results = []

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    for name, model in models.items():

        print("\n")
        print("-" * 70)
        print(
            f"TRAINING: {name}"
        )
        print("-" * 70)

        start = time.time()

        # XGBoost and MLP can be slower
        # but use the same TF-IDF representation.

        model.fit(
            X_train,
            y_train
        )

        training_time = (
            time.time() - start
        )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        start = time.time()

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        inference_time = (
            time.time() - start
        )

        result = {

            "Model":
                name,

            **metrics,

            "Training_Time":
                training_time,

            "Inference_Time":
                inference_time
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            f"Accuracy:  "
            f"{metrics['Accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrics['Precision']:.4f}"
        )

        print(
            f"Recall:    "
            f"{metrics['Recall']:.4f}"
        )

        print(
            f"F1:        "
            f"{metrics['F1']:.4f}"
        )

        print(
            f"PR-AUC:    "
            f"{metrics['PR_AUC']:.4f}"
        )

        print(
            f"ROC-AUC:   "
            f"{metrics['ROC_AUC']:.4f}"
        )

        print(
            f"Training:  "
            f"{training_time:.3f}s"
        )

        print(
            f"Inference: "
            f"{inference_time:.3f}s"
        )

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "F1",
        ascending=False
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_dir = (
        "results/metrics"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        output_dir,
        "expanded_classical.csv"
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
        "FINAL CLASSICAL BENCHMARK"
    )
    print("=" * 100)

    print(
        results_df[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "PR_AUC",
                "ROC_AUC",
                "Training_Time"
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
        f" {output_path}"
    )


if __name__ == "__main__":

    main()