import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute standard classification evaluation metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        except Exception:
            roc_auc = roc_auc_score(y_true, y_pred)
            pr_auc = average_precision_score(y_true, y_pred)
    else:
        roc_auc = roc_auc_score(y_true, y_pred)
        pr_auc = average_precision_score(y_true, y_pred)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc
    }


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Fit a model on training data and evaluate it on test data.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test)
            if probs.ndim == 2 and probs.shape[1] > 1:
                y_prob = probs[:, 1]
            else:
                y_prob = probs.ravel()
        except Exception:
            pass
    elif hasattr(model, "decision_function"):
        try:
            y_prob = model.decision_function(X_test)
        except Exception:
            pass

    return compute_metrics(y_test, y_pred, y_prob)
