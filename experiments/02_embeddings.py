import os
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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


from src.preprocessing import load_dataset, split_dataset
from src.embeddings import generate_embeddings, DistilBERTEmbedder


def main():

    # ==========================================
    # Load dataset
    # ==========================================

    df = load_dataset()

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_dataset(df)

    X_train = X_train.reset_index(
        drop=True
    )

    X_test = X_test.reset_index(
        drop=True
    )

    y_train = y_train.reset_index(
        drop=True
    )

    y_test = y_test.reset_index(
        drop=True
    )

    # ==========================================
    # Generate embeddings
    # ==========================================

    train_path = (
        "results/"
        "distilbert_train_embeddings.npy"
    )

    test_path = (
        "results/"
        "distilbert_test_embeddings.npy"
    )

    if (
        os.path.exists(train_path)
        and os.path.exists(test_path)
    ):

        print(
            "\nLoading cached embeddings..."
        )

        X_train_embeddings = np.load(
            train_path
        )

        X_test_embeddings = np.load(
            test_path
        )

    else:

        embedder = (
            DistilBERTEmbedder()
        )

        print(
            "\nGenerating training embeddings..."
        )

        X_train_embeddings = (
            embedder.encode(
                list(X_train)
            )
        )

        print(
            "\nGenerating test embeddings..."
        )

        X_test_embeddings = (
            embedder.encode(
                list(X_test)
            )
        )

        os.makedirs(
            "results",
            exist_ok=True
        )

        np.save(
            train_path,
            X_train_embeddings
        )

        np.save(
            test_path,
            X_test_embeddings
        )

    print(
        "\nTrain embedding shape:",
        X_train_embeddings.shape
    )

    print(
        "Test embedding shape:",
        X_test_embeddings.shape
    )

    # ==========================================
    # Scale embeddings
    # ==========================================

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train_embeddings
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test_embeddings
        )
    )

    # ==========================================
    # PCA
    # ==========================================

    pca = PCA(
        n_components=8,
        random_state=42
    )

    X_train_pca = (
        pca.fit_transform(
            X_train_scaled
        )
    )

    X_test_pca = (
        pca.transform(
            X_test_scaled
        )
    )

    explained_variance = (
        pca.explained_variance_ratio_.sum()
    )

    print(
        "\nPCA output shape:",
        X_train_pca.shape
    )

    print(
        f"PCA explained variance: "
        f"{explained_variance:.4f}"
    )

    # ==========================================
    # Classical classifier
    # ==========================================

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    print(
        "\nTraining DistilBERT + "
        "Logistic Regression..."
    )

    model.fit(
        X_train_pca,
        y_train
    )

    probabilities = (
        model.predict_proba(
            X_test_pca
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # ==========================================
    # Evaluation
    # ==========================================

    print("\n")
    print("=" * 60)
    print("DISTILBERT RESULTS")
    print("=" * 60)

    print(
        f"Accuracy:  "
        f"{accuracy_score(y_test, predictions):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_test, predictions):.4f}"
    )

    print(
        f"Recall:    "
        f"{recall_score(y_test, predictions):.4f}"
    )

    print(
        f"F1:        "
        f"{f1_score(y_test, predictions):.4f}"
    )

    print(
        f"PR-AUC:    "
        f"{average_precision_score(y_test, probabilities):.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{roc_auc_score(y_test, probabilities):.4f}"
    )

    # ==========================================
    # Save PCA features
    # ==========================================

    np.save(
        "results/X_train_pca.npy",
        X_train_pca
    )

    np.save(
        "results/X_test_pca.npy",
        X_test_pca
    )

    np.save(
        "results/y_train.npy",
        y_train.to_numpy()
    )

    np.save(
        "results/y_test.npy",
        y_test.to_numpy()
    )

    print(
        "\nSaved PCA features."
    )


if __name__ == "__main__":

    main()