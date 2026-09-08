import time

import numpy as np

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Train and evaluate a classification model.
    """

    # -------------------------
    # Training
    # -------------------------

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.time() - start
    )

    # -------------------------
    # Prediction
    # -------------------------

    start = time.time()

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    inference_time = (
        time.time() - start
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # -------------------------
    # Metrics
    # -------------------------

    results = {

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

        "ROC_AUC":
            roc_auc_score(
                y_test,
                probabilities
            ),

        "PR_AUC":
            average_precision_score(
                y_test,
                probabilities
            ),

        "Training_Time":
            training_time,

        "Inference_Time":
            inference_time
    }

    return results


def get_models():

    models = {

        "TF-IDF + Logistic Regression":

            Pipeline([
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=10000,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        min_df=2
                    )
                ),

                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ]),

        "TF-IDF + SVM":

            Pipeline([
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=10000,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        min_df=2
                    )
                ),

                (
                    "classifier",
                    SVC(
                        probability=True,
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ])
    }

    return models