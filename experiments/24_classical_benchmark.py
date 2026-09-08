#!/usr/bin/env python3
"""
Experiment 24: Multi-Dataset Classical NLP Benchmark
=====================================================
Establishes standardized classical NLP baselines across the three frozen datasets:
  1. SMS Spam Collection (results/frozen_splits/sms/)
  2. CEAS-08 (results/frozen_splits/ceas/)
  3. MeAJOR (results/frozen_splits/meajor/)

Models evaluated:
  1. Multinomial Naive Bayes (MultinomialNB)
  2. Logistic Regression (LogisticRegression, class_weight='balanced')
  3. Linear SVM (LinearSVC, class_weight='balanced')
  4. RBF SVM (SVC, kernel='rbf', class_weight='balanced')

Text Representations:
  A. Raw text
  B. Lemmatized text (reproducible WordNetLemmatizer, NLTK 3.10.3)

Evaluation:
  - TF-IDF fitted strictly on training data (ngram_range=(1,2), min_df=2, max_features=50000).
  - Metrics computed on frozen test sets: Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC.
  - Wall-clock training and inference times recorded.
  - Outputs saved under results/metrics/:
      - multidataset_classical.csv
      - multidataset_classical_summary.csv

Author: Quantum Phishing & Scam Detection Project
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
import nltk
from nltk.stem import WordNetLemmatizer

# Suppress minor scikit-learn deprecation/convergence warnings during benchmark
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION & PATHS
# ============================================================
RANDOM_SEED = 42

DATASET_PATHS = {
    "SMS": {
        "train": "results/frozen_splits/sms/train.csv",
        "validation": "results/frozen_splits/sms/validation.csv",
        "test": "results/frozen_splits/sms/test.csv",
    },
    "CEAS": {
        "train": "results/frozen_splits/ceas/train.csv",
        "validation": "results/frozen_splits/ceas/validation.csv",
        "test": "results/frozen_splits/ceas/test.csv",
    },
    "MeAJOR": {
        "train": "results/frozen_splits/meajor/train.csv",
        "validation": "results/frozen_splits/meajor/validation.csv",
        "test": "results/frozen_splits/meajor/test.csv",
    },
}

METRICS_DIR = "results/metrics"
FULL_METRICS_PATH = os.path.join(METRICS_DIR, "multidataset_classical.csv")
SUMMARY_METRICS_PATH = os.path.join(METRICS_DIR, "multidataset_classical_summary.csv")


# Initialize NLTK WordNet Lemmatizer
try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("testing")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    lemmatizer = WordNetLemmatizer()


def lemmatize_text(text: str) -> str:
    """
    Apply reproducible English lemmatization on lowercased whitespace-tokenized words.
    Sample-wise transformation with zero statistical fitting or cross-split leakage.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    words = text.lower().split()
    return " ".join(lemmatizer.lemmatize(w) for w in words)


def get_models() -> dict:
    """Return dictionary of classical model instances with fixed seeds and balanced weights."""
    return {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "RBF SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            cache_size=2000,
        ),
    }


def evaluate_dataset_representation(
    dataset_name: str,
    representation_name: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> list:
    """
    Fit TF-IDF on training text, transform val and test, and benchmark all 4 classical classifiers.
    """
    print("=" * 60)
    print(f"DATASET: {dataset_name}")
    print(f"REPRESENTATION: {representation_name.upper()}")
    print("=" * 60)

    n_train = len(df_train)
    n_val = len(df_val)
    n_test = len(df_test)

    print(f"Train samples:      {n_train}")
    print(f"Validation samples: {n_val}")
    print(f"Test samples:       {n_test}")

    # Prepare text representations
    if representation_name.lower() == "lemmatized":
        print("\nApplying WordNet lemmatization...")
        t0_lem = time.time()
        X_train_raw = df_train["text"].fillna("").apply(lemmatize_text).values
        X_val_raw = df_val["text"].fillna("").apply(lemmatize_text).values
        X_test_raw = df_test["text"].fillna("").apply(lemmatize_text).values
        print(f"Lemmatization completed in {time.time() - t0_lem:.2f}s")
    else:
        X_train_raw = df_train["text"].fillna("").values
        X_val_raw = df_val["text"].fillna("").values
        X_test_raw = df_test["text"].fillna("").values

    y_train = df_train["target"].astype(int).values
    y_val = df_val["target"].astype(int).values
    y_test = df_test["target"].astype(int).values

    # TF-IDF Vectorization: Fit ONLY on training data
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=50000,
    )

    t0_vec = time.time()
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_val_tfidf = vectorizer.transform(X_val_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    vec_time = time.time() - t0_vec

    tfidf_features = X_train_tfidf.shape[1]
    print(f"TF-IDF dimensions:  {tfidf_features} features (fitted in {vec_time:.2f}s)\n")

    results = []
    models = get_models()

    print(f"{'Model':<25} | {'Acc':<7} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'PR-AUC':<7} | {'ROC-AUC':<7} | {'Train(s)':<8} | {'Infer(s)':<8}")
    print("-" * 105)

    for model_name, model in models.items():
        # Training
        t_start_train = time.time()
        model.fit(X_train_tfidf, y_train)
        train_time = time.time() - t_start_train

        # Inference on Test Set
        t_start_infer = time.time()
        y_pred = model.predict(X_test_tfidf)
        infer_time = time.time() - t_start_infer

        # Continuous decision scores for PR-AUC and ROC-AUC
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test_tfidf)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test_tfidf)
        else:
            scores = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, scores)

        pr_prec, pr_rec, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(pr_rec, pr_prec)

        print(
            f"{model_name:<25} | {acc:<7.4f} | {prec:<7.4f} | {rec:<7.4f} | "
            f"{f1:<7.4f} | {pr_auc:<7.4f} | {roc:<7.4f} | {train_time:<8.2f} | {infer_time:<8.4f}"
        )

        results.append({
            "Dataset": dataset_name,
            "Representation": representation_name,
            "Model": model_name,
            "Train_Samples": n_train,
            "Validation_Samples": n_val,
            "Test_Samples": n_test,
            "TFIDF_Features": tfidf_features,
            "Accuracy": round(acc, 6),
            "Precision": round(prec, 6),
            "Recall": round(rec, 6),
            "F1": round(f1, 6),
            "PR_AUC": round(pr_auc, 6),
            "ROC_AUC": round(roc, 6),
            "Training_Time": round(train_time, 4),
            "Inference_Time": round(infer_time, 4),
        })

    print()
    return results


def main():
    print("=" * 60)
    print("EXPERIMENT 24: MULTI-DATASET CLASSICAL NLP BENCHMARK")
    print("============================================================")
    print("Evaluating 4 Classical Classifiers across 3 Frozen Datasets × 2 Representations\n")

    os.makedirs(METRICS_DIR, exist_ok=True)
    all_results = []

    # Iterate over datasets
    for dataset_name, paths in DATASET_PATHS.items():
        if not os.path.exists(paths["train"]):
            raise FileNotFoundError(f"Missing frozen train split: {paths['train']}")
        if not os.path.exists(paths["validation"]):
            raise FileNotFoundError(f"Missing frozen validation split: {paths['validation']}")
        if not os.path.exists(paths["test"]):
            raise FileNotFoundError(f"Missing frozen test split: {paths['test']}")

        df_train = pd.read_csv(paths["train"])
        df_val = pd.read_csv(paths["validation"])
        df_test = pd.read_csv(paths["test"])

        # 1. Raw text representation
        res_raw = evaluate_dataset_representation(
            dataset_name=dataset_name,
            representation_name="raw",
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
        )
        all_results.extend(res_raw)

        # 2. Lemmatized text representation
        res_lem = evaluate_dataset_representation(
            dataset_name=dataset_name,
            representation_name="lemmatized",
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
        )
        all_results.extend(res_lem)

    # Convert results to DataFrame
    df_metrics = pd.DataFrame(all_results)
    df_metrics.to_csv(FULL_METRICS_PATH, index=False)
    print(f"\nSaved full benchmark metrics to: {FULL_METRICS_PATH} ({len(df_metrics)} evaluations)")

    # Compute Summary: Best model per Dataset × Representation (by F1)
    summary_rows = []
    for (ds, rep), grp in df_metrics.groupby(["Dataset", "Representation"]):
        best_row = grp.sort_values(by="F1", ascending=False).iloc[0]
        summary_rows.append(best_row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(SUMMARY_METRICS_PATH, index=False)
    print(f"Saved best model summary to:    {SUMMARY_METRICS_PATH}")

    # ============================================================
    # FINAL SUMMARY REPORT
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL BENCHMARK SUMMARY (Sorted by Dataset, Representation, F1 Descending)")
    print("=" * 60)
    df_sorted = df_metrics.sort_values(by=["Dataset", "Representation", "F1"], ascending=[True, True, False])
    print(f"{'Dataset':<8} | {'Rep':<10} | {'Model':<24} | {'F1':<7} | {'PR-AUC':<7} | {'ROC-AUC':<7} | {'Train(s)':<8} | {'Infer(s)':<8}")
    print("-" * 95)
    for _, row in df_sorted.iterrows():
        print(
            f"{row['Dataset']:<8} | {row['Representation']:<10} | {row['Model']:<24} | "
            f"{row['F1']:<7.4f} | {row['PR_AUC']:<7.4f} | {row['ROC_AUC']:<7.4f} | {row['Training_Time']:<8.2f} | {row['Inference_Time']:<8.4f}"
        )

    print("\n" + "=" * 60)
    print("BEST CLASSICAL MODEL PER DATASET")
    print("=" * 60)
    for ds in ["SMS", "CEAS", "MeAJOR"]:
        ds_subset = df_metrics[df_metrics["Dataset"] == ds].sort_values(by="F1", ascending=False)
        best_m = ds_subset.iloc[0]
        print(f"{ds:<8}: {best_m['Model']} ({best_m['Representation']}) -> F1: {best_m['F1']:.4f}, PR-AUC: {best_m['PR_AUC']:.4f}, ROC-AUC: {best_m['ROC_AUC']:.4f}, Acc: {best_m['Accuracy']:.4f}")

    print("\n" + "=" * 60)
    print("LEMMATIZATION IMPACT ANALYSIS (Δ F1 = Lemmatized - Raw)")
    print("=" * 60)
    models_list = ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM", "RBF SVM"]
    for ds in ["SMS", "CEAS", "MeAJOR"]:
        print(f"\n--- {ds} ---")
        for m in models_list:
            raw_f1 = df_metrics[(df_metrics["Dataset"] == ds) & (df_metrics["Representation"] == "raw") & (df_metrics["Model"] == m)]["F1"].values[0]
            lem_f1 = df_metrics[(df_metrics["Dataset"] == ds) & (df_metrics["Representation"] == "lemmatized") & (df_metrics["Model"] == m)]["F1"].values[0]
            delta = lem_f1 - raw_f1
            status = "IMPROVED (+)" if delta > 0.0001 else "DECREASED (-)" if delta < -0.0001 else "IDENTICAL (=)"
            print(f"  {m:<24}: Raw={raw_f1:.4f} | Lem={lem_f1:.4f} | Δ F1={delta:+.4f} ({status})")

    print("\nClassical NLP benchmark completed successfully.")


if __name__ == "__main__":
    main()
