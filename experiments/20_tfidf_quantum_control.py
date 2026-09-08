import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.fast_quantum_kernel import FastQuantumKernel


warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "data/SMSSpamCollection"

OUTPUT_DIR = "results/metrics"

RANDOM_STATE = 42

DIMENSIONS = [2, 4, 6, 8]

MAX_FEATURES = 15000


# ============================================================
# NLTK
# ============================================================

try:

    nltk.data.find(
        "corpora/wordnet"
    )

except LookupError:

    nltk.download(
        "wordnet"
    )

try:

    nltk.data.find(
        "corpora/omw-1.4"
    )

except LookupError:

    nltk.download(
        "omw-1.4"
    )


lemmatizer = WordNetLemmatizer()


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

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
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    # URLs

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URLTOKEN ",
        text
    )

    # Email addresses

    text = re.sub(
        r"\S+@\S+",
        " EMAILTOKEN ",
        text
    )

    # Phone numbers

    text = re.sub(
        r"\b\d[\d\s\-]{7,}\b",
        " PHONETOKEN ",
        text
    )

    # Numbers

    text = re.sub(
        r"\b\d+(?:\.\d+)?\b",
        " NUMTOKEN ",
        text
    )

    # Keep alphabetic characters,
    # token markers and spaces.

    text = re.sub(
        r"[^a-zA-Z_]+",
        " ",
        text
    )

    # Normalize whitespace

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LEMMATIZATION
# ============================================================

def lemmatize_text(text):

    tokens = text.split()

    lemmatized = []

    for token in tokens:

        # Preserve our special tokens

        if token in {
            "urltoken",
            "emailtoken",
            "phonetoken",
            "numtoken"
        }:

            lemmatized.append(
                token
            )

        else:

            lemmatized.append(
                lemmatizer.lemmatize(
                    token
                )
            )

    return " ".join(
        lemmatized
    )


# ============================================================
# COMPLETE PREPROCESSING
# ============================================================

def preprocess(text):

    cleaned = clean_text(
        text
    )

    return lemmatize_text(
        cleaned
    )


# ============================================================
# LOAD + SPLIT
# ============================================================

def create_split(df):

    from sklearn.model_selection import train_test_split

    train_df, temp_df = (
        train_test_split(

            df,

            train_size=4457,

            random_state=RANDOM_STATE,

            stratify=df["target"]
        )
    )

    # This reproduces the approximate
    # 80/20 split used in the classical
    # benchmark.

    return (
        train_df,
        temp_df
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
# QUANTUM SCALING
# ============================================================

def scale_for_quantum(
    X_train,
    X_val,
    X_test
):

    minimum = X_train.min(
        axis=0
    )

    maximum = X_train.max(
        axis=0
    )

    denominator = (
        maximum - minimum
    )

    denominator[
        denominator == 0
    ] = 1.0

    def transform(X):

        X = (
            X - minimum
        ) / denominator

        X = (
            2.0
            * np.pi
            * X
            - np.pi
        )

        return X

    return (
        transform(X_train),
        transform(X_val),
        transform(X_test)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)

    print(
        "LEMMATIZED TF-IDF → "
        "CLASSICAL VS QUANTUM"
    )

    print("=" * 90)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = load_dataset()

    print(
        "\nDataset:",
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

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_df, test_df = (
        create_split(
            df
        )
    )

    print(
        "\nTraining:",
        train_df.shape
    )

    print(
        "Testing:",
        test_df.shape
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    print(
        "\nPreprocessing text..."
    )

    train_text = [
        preprocess(text)
        for text
        in train_df["text"]
    ]

    test_text = [
        preprocess(text)
        for text
        in test_df["text"]
    ]

    y_train_full = (
        train_df["target"]
        .values
    )

    y_test_full = (
        test_df["target"]
        .values
    )

    # --------------------------------------------------------
    # Create validation split
    # --------------------------------------------------------

    from sklearn.model_selection import train_test_split

    (
        train_text,
        val_text,
        y_train,
        y_val
    ) = train_test_split(

        train_text,

        y_train_full,

        test_size=0.20,

        random_state=RANDOM_STATE,

        stratify=y_train_full
    )

    print(
        "\nFinal split:"
    )

    print(
        "Training:",
        len(train_text),
        np.bincount(y_train)
    )

    print(
        "Validation:",
        len(val_text),
        np.bincount(y_val)
    )

    print(
        "Testing:",
        len(test_text),
        np.bincount(y_test_full)
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)

    print(
        "BUILDING LEMMATIZED TF-IDF"
    )

    print("=" * 90)

    vectorizer = TfidfVectorizer(

        max_features=MAX_FEATURES,

        ngram_range=(1, 2),

        sublinear_tf=True,

        min_df=2
    )

    start = time.time()

    X_train_tfidf = (
        vectorizer.fit_transform(
            train_text
        )
    )

    X_val_tfidf = (
        vectorizer.transform(
            val_text
        )
    )

    X_test_tfidf = (
        vectorizer.transform(
            test_text
        )
    )

    tfidf_time = (
        time.time()
        - start
    )

    print(
        "Training shape:",
        X_train_tfidf.shape
    )

    print(
        "Validation shape:",
        X_val_tfidf.shape
    )

    print(
        "Testing shape:",
        X_test_tfidf.shape
    )

    print(
        f"TF-IDF time:"
        f" {tfidf_time:.3f}s"
    )

    # ========================================================
    # PCA
    # ========================================================

    results = []

    for dimensions in DIMENSIONS:

        print("\n")
        print("=" * 90)

        print(
            f"DIMENSION = {dimensions}"
        )

        print("=" * 90)

        # ----------------------------------------------------
        # Convert to dense
        # ----------------------------------------------------

        X_train_dense = (
            X_train_tfidf.toarray()
        )

        X_val_dense = (
            X_val_tfidf.toarray()
        )

        X_test_dense = (
            X_test_tfidf.toarray()
        )

        # ----------------------------------------------------
        # Standardize
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = (
            scaler.fit_transform(
                X_train_dense
            )
        )

        X_val_scaled = (
            scaler.transform(
                X_val_dense
            )
        )

        X_test_scaled = (
            scaler.transform(
                X_test_dense
            )
        )

        # ----------------------------------------------------
        # PCA
        # ----------------------------------------------------

        pca = PCA(
            n_components=dimensions,
            random_state=RANDOM_STATE
        )

        X_train_pca = (
            pca.fit_transform(
                X_train_scaled
            )
        )

        X_val_pca = (
            pca.transform(
                X_val_scaled
            )
        )

        X_test_pca = (
            pca.transform(
                X_test_scaled
            )
        )

        print(
            "PCA explained variance:",
            f"{pca.explained_variance_ratio_.sum():.4f}"
        )

        # ====================================================
        # CLASSICAL RBF
        # ====================================================

        print(
            "\nRunning classical RBF-SVM..."
        )

        classical_model = SVC(

            kernel="rbf",

            C=1.0,

            gamma="scale",

            class_weight="balanced",

            probability=True,

            random_state=RANDOM_STATE
        )

        start = time.time()

        classical_model.fit(
            X_train_pca,
            y_train
        )

        classical_training_time = (
            time.time()
            - start
        )

        val_prob = (
            classical_model
            .predict_proba(
                X_val_pca
            )[:, 1]
        )

        (
            classical_threshold,
            classical_val_f1
        ) = find_best_threshold(
            val_prob,
            y_val
        )

        test_prob = (
            classical_model
            .predict_proba(
                X_test_pca
            )[:, 1]
        )

        classical_metrics = evaluate(
            test_prob,
            y_test_full,
            classical_threshold
        )

        print(
            f"Classical F1:"
            f" {classical_metrics['F1']:.4f}"
        )

        print(
            f"Classical PR-AUC:"
            f" {classical_metrics['PR_AUC']:.4f}"
        )

        # ====================================================
        # QUANTUM
        # ====================================================

        print(
            "\nRunning quantum kernel..."
        )

        (
            X_train_q,
            X_val_q,
            X_test_q
        ) = scale_for_quantum(

            X_train_pca,

            X_val_pca,

            X_test_pca
        )

        kernel = FastQuantumKernel(

            n_qubits=dimensions,

            n_layers=1
        )

        # ----------------------------------------------------
        # Training kernel
        # ----------------------------------------------------

        start = time.time()

        K_train = kernel.matrix(
            X_train_q
        )

        train_kernel_time = (
            time.time()
            - start
        )

        # ----------------------------------------------------
        # Validation kernel
        # ----------------------------------------------------

        start = time.time()

        K_val = kernel.matrix(
            X_val_q,
            X_train_q
        )

        val_kernel_time = (
            time.time()
            - start
        )

        # ----------------------------------------------------
        # Test kernel
        # ----------------------------------------------------

        start = time.time()

        K_test = kernel.matrix(
            X_test_q,
            X_train_q
        )

        test_kernel_time = (
            time.time()
            - start
        )

        total_kernel_time = (
            train_kernel_time
            + val_kernel_time
            + test_kernel_time
        )

        quantum_model = SVC(

            kernel="precomputed",

            class_weight="balanced",

            probability=True,

            random_state=RANDOM_STATE
        )

        quantum_model.fit(
            K_train,
            y_train
        )

        val_prob = (
            quantum_model
            .predict_proba(
                K_val
            )[:, 1]
        )

        (
            quantum_threshold,
            quantum_val_f1
        ) = find_best_threshold(
            val_prob,
            y_val
        )

        test_prob = (
            quantum_model
            .predict_proba(
                K_test
            )[:, 1]
        )

        quantum_metrics = evaluate(
            test_prob,
            y_test_full,
            quantum_threshold
        )

        print(
            f"Quantum F1:"
            f" {quantum_metrics['F1']:.4f}"
        )

        print(
            f"Quantum PR-AUC:"
            f" {quantum_metrics['PR_AUC']:.4f}"
        )

        print(
            f"Quantum ROC-AUC:"
            f" {quantum_metrics['ROC_AUC']:.4f}"
        )

        # ----------------------------------------------------
        # Save both
        # ----------------------------------------------------

        results.append({

            "Model":
                "Classical RBF-SVM",

            "Dimensions":
                dimensions,

            "F1":
                classical_metrics["F1"],

            "PR_AUC":
                classical_metrics["PR_AUC"],

            "ROC_AUC":
                classical_metrics["ROC_AUC"],

            "Validation_F1":
                classical_val_f1,

            "Threshold":
                classical_threshold,

            "Training_Time":
                classical_training_time,

            "Kernel_Time":
                0.0
        })

        results.append({

            "Model":
                "Quantum Kernel SVM",

            "Dimensions":
                dimensions,

            "F1":
                quantum_metrics["F1"],

            "PR_AUC":
                quantum_metrics["PR_AUC"],

            "ROC_AUC":
                quantum_metrics["ROC_AUC"],

            "Validation_F1":
                quantum_val_f1,

            "Threshold":
                quantum_threshold,

            "Training_Time":
                0.0,

            "Kernel_Time":
                total_kernel_time
        })

    # ========================================================
    # RESULTS
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
        "tfidf_classical_vs_quantum.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n")
    print("=" * 100)

    print(
        "LEMMATIZED TF-IDF: "
        "CLASSICAL VS QUANTUM"
    )

    print("=" * 100)

    print(
        results_df[
            [
                "Model",
                "Dimensions",
                "F1",
                "PR_AUC",
                "ROC_AUC",
                "Validation_F1",
                "Threshold"
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