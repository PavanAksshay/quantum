#!/usr/bin/env python3
"""
Experiment 45: Classical Baseline Audit & Empirical Justification Suite
======================================================================
Evaluates classical machine learning baselines across 3 canonical text security datasets:
  1. SMS Spam Collection (results/frozen_splits/sms/)
  2. CEAS-08 (results/frozen_splits/ceas/ canonical experimental subset)
  3. MeAJOR (results/roberta_multidataset/meajor/ canonical experimental subset)

Across 2 Representations:
  A. Full-dimensional canonical TF-IDF (50,000 features max, ngrams 1-2, sublinear_tf=True, norm='l2')
  B. Matched 8D Reduced TF-IDF (TF-IDF -> TruncatedSVD(8D) -> StandardScaler)

Across 8 Classical Models:
  1. Multinomial Naive Bayes / GaussianNB (for continuous 8D)
  2. Logistic Regression (C=1, class_weight='balanced')
  3. Linear SVM (LinearSVC, C=1, class_weight='balanced')
  4. RBF SVM (SVC, kernel='rbf', C=1, gamma='scale', class_weight='balanced')
  5. Random Forest (RandomForestClassifier, n_estimators=100, class_weight='balanced')
  6. XGBoost / Gradient Boosting (XGBClassifier, n_estimators=100)
  7. Multi-Layer Perceptron (MLPClassifier, hidden=(64,), max_iter=500)
  8. k-Nearest Neighbors (KNeighborsClassifier, n_neighbors=5, weights='distance')

Across 10 Canonical Random Seeds:
  [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

Saves outputs to results/exp45/
"""

import os
import sys
import time
import tracemalloc
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

warnings.filterwarnings("ignore")

BASE_DIR = "/Users/pavanaksshay/quantum"
EXP45_DIR = os.path.join(BASE_DIR, "results/exp45")
FIGURES_DIR = os.path.join(EXP45_DIR, "figures")
os.makedirs(EXP45_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

def load_datasets():
    """Load canonical frozen splits for SMS, CEAS, and MeAJOR."""
    datasets = {}
    
    # 1. SMS Spam
    sms_dir = os.path.join(BASE_DIR, "results/frozen_splits/sms")
    sms_tr = pd.read_csv(os.path.join(sms_dir, "train.csv"))
    sms_va = pd.read_csv(os.path.join(sms_dir, "validation.csv"))
    sms_te = pd.read_csv(os.path.join(sms_dir, "test.csv"))
    datasets["SMS"] = {
        "train": sms_tr,
        "val": sms_va,
        "test": sms_te,
        "subset_info": "Full frozen split (3,343 Train / 1,114 Val / 1,115 Test)",
    }
    
    # 2. CEAS-08 (Canonical subset used in Exp 26/41/42: 10k train / 2.5k val / 2.5k test)
    ceas_dir = os.path.join(BASE_DIR, "results/frozen_splits/ceas")
    ceas_subset_dir = os.path.join(BASE_DIR, "results/roberta_multidataset/ceas")
    if os.path.exists(os.path.join(ceas_subset_dir, "train_sample_ids.csv")):
        ceas_tr_ids = pd.read_csv(os.path.join(ceas_subset_dir, "train_sample_ids.csv"))["sample_id"]
        ceas_va_ids = pd.read_csv(os.path.join(ceas_subset_dir, "validation_sample_ids.csv"))["sample_id"]
        ceas_te_ids = pd.read_csv(os.path.join(ceas_subset_dir, "test_sample_ids.csv"))["sample_id"]
        ceas_full_tr = pd.read_csv(os.path.join(ceas_dir, "train.csv")).set_index("sample_id")
        ceas_full_va = pd.read_csv(os.path.join(ceas_dir, "validation.csv")).set_index("sample_id")
        ceas_full_te = pd.read_csv(os.path.join(ceas_dir, "test.csv")).set_index("sample_id")
        ceas_tr = ceas_full_tr.loc[ceas_tr_ids].reset_index()
        ceas_va = ceas_full_va.loc[ceas_va_ids].reset_index()
        ceas_te = ceas_full_te.loc[ceas_te_ids].reset_index()
        ceas_info = "Canonical experimental subset (10,000 Train / 2,500 Val / 2,500 Test)"
    else:
        ceas_tr = pd.read_csv(os.path.join(ceas_dir, "train.csv"))
        ceas_va = pd.read_csv(os.path.join(ceas_dir, "validation.csv"))
        ceas_te = pd.read_csv(os.path.join(ceas_dir, "test.csv"))
        ceas_info = "Full frozen split (23,494 Train / 7,830 Val / 7,830 Test)"
        
    datasets["CEAS"] = {
        "train": ceas_tr,
        "val": ceas_va,
        "test": ceas_te,
        "subset_info": ceas_info,
    }
    
    # 3. MeAJOR (Canonical experimental subset: 6k train / 2k val / 2k test)
    meajor_dir = os.path.join(BASE_DIR, "results/frozen_splits/meajor")
    meajor_subset_dir = os.path.join(BASE_DIR, "results/roberta_multidataset/meajor")
    meajor_tr_ids = pd.read_csv(os.path.join(meajor_subset_dir, "train_sample_ids.csv"))["sample_id"]
    meajor_va_ids = pd.read_csv(os.path.join(meajor_subset_dir, "validation_sample_ids.csv"))["sample_id"]
    meajor_te_ids = pd.read_csv(os.path.join(meajor_subset_dir, "test_sample_ids.csv"))["sample_id"]
    meajor_full_tr = pd.read_csv(os.path.join(meajor_dir, "train.csv")).set_index("sample_id")
    meajor_full_va = pd.read_csv(os.path.join(meajor_dir, "validation.csv")).set_index("sample_id")
    meajor_full_te = pd.read_csv(os.path.join(meajor_dir, "test.csv")).set_index("sample_id")
    meajor_tr = meajor_full_tr.loc[meajor_tr_ids].reset_index()
    meajor_va = meajor_full_va.loc[meajor_va_ids].reset_index()
    meajor_te = meajor_full_te.loc[meajor_te_ids].reset_index()
    
    datasets["MeAJOR"] = {
        "train": meajor_tr,
        "val": meajor_va,
        "test": meajor_te,
        "subset_info": "Canonical experimental subset (6,000 Train / 2,000 Val / 2,000 Test)",
    }
    
    return datasets


def select_best_threshold(y_val, val_scores, num_steps=200):
    """Find threshold that maximizes F1 on validation set."""
    if len(np.unique(val_scores)) <= 1:
        return 0.5, float(f1_score(y_val, (val_scores >= 0.5).astype(int), zero_division=0))
    thresholds = np.linspace(val_scores.min(), val_scores.max(), num_steps)
    best_f1, best_th = -1.0, 0.5
    for th in thresholds:
        f1 = f1_score(y_val, (val_scores >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return float(best_th), float(best_f1)


def evaluate_model(y_true, scores, threshold):
    """Compute all evaluation metrics."""
    preds = (scores >= threshold).astype(int)
    
    try:
        pr_auc = float(average_precision_score(y_true, scores))
    except Exception:
        pr_auc = 0.0
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except Exception:
        roc_auc = 0.5
        
    return {
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "threshold": float(threshold),
    }


def get_scores_and_status(model, X):
    """Extract continuous decision scores / probabilities and check convergence."""
    convergence_status = "CONVERGED"
    
    if hasattr(model, "n_iter_"):
        max_iter = getattr(model, "max_iter", None)
        if max_iter is not None and np.all(model.n_iter_ >= max_iter):
            convergence_status = "MAX_ITER_REACHED"
            
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim > 1:
            scores = scores[:, 1]
    elif hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.shape[1] > 1:
            scores = probs[:, 1]
        else:
            scores = probs[:, 0]
    else:
        scores = model.predict(X).astype(float)
        
    return scores, convergence_status


def count_parameters(model, input_dim):
    """Estimate number of trainable/model parameters."""
    if isinstance(model, (LinearSVC, LogisticRegression)):
        return int(input_dim + 1)
    elif isinstance(model, SVC):
        if hasattr(model, "support_vectors_"):
            return int(model.support_vectors_.shape[0] * input_dim)
        return int(input_dim)
    elif isinstance(model, MLPClassifier):
        if hasattr(model, "coefs_"):
            return int(sum(c.size for c in model.coefs_) + sum(i.size for i in model.intercepts_))
        return int(input_dim * 64 + 64 + 64 + 1)
    elif isinstance(model, (RandomForestClassifier, XGBClassifier)):
        return "Ensemble (100 Trees)"
    elif isinstance(model, (MultinomialNB, GaussianNB)):
        return int(2 * input_dim)
    elif isinstance(model, KNeighborsClassifier):
        return "Non-parametric (Instance-based)"
    return "N/A"


def run_benchmark():
    print("=" * 80)
    print("EXP 45: CLASSICAL BASELINE AUDIT BENCHMARK")
    print(f"Seeds ({len(SEEDS)}): {SEEDS}")
    print("=" * 80)
    
    datasets = load_datasets()
    all_runs = []
    
    for d_name, d_data in datasets.items():
        print(f"\n>>> DATASET: {d_name} ({d_data['subset_info']})")
        df_tr = d_data["train"]
        df_va = d_data["val"]
        df_te = d_data["test"]
        
        texts_tr = df_tr["text"].fillna("").astype(str).tolist()
        texts_va = df_va["text"].fillna("").astype(str).tolist()
        texts_te = df_te["text"].fillna("").astype(str).tolist()
        
        y_tr = df_tr["target"].values.astype(int)
        y_va = df_va["target"].values.astype(int)
        y_te = df_te["target"].values.astype(int)
        
        # 1. Full TF-IDF Representation
        print("  --> Representation: Full TF-IDF (50,000 Dimensions)")
        tfidf = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=50000,
            norm="l2"
        )
        X_tr_full = tfidf.fit_transform(texts_tr)
        X_va_full = tfidf.transform(texts_va)
        X_te_full = tfidf.transform(texts_te)
        feat_dim_full = X_tr_full.shape[1]
        
        # Deterministic models on full TF-IDF (fit once, replicate metrics across 10 seeds)
        deterministic_models_full = {
            "Multinomial Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42),
            "Linear SVM": LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42),
            "RBF SVM": SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=False, cache_size=2000, random_state=42),
            "k-NN": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
        }
        
        for m_name, model in deterministic_models_full.items():
            t0 = time.time()
            tracemalloc.start()
            model.fit(X_tr_full, y_tr)
            t_train = time.time() - t0
            curr_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_ram = peak_mem / (1024 * 1024)
            
            va_scores, _ = get_scores_and_status(model, X_va_full)
            best_th, _ = select_best_threshold(y_va, va_scores)
            
            t0_inf = time.time()
            te_scores, conv_status = get_scores_and_status(model, X_te_full)
            t_inf = time.time() - t0_inf
            m_dict = evaluate_model(y_te, te_scores, best_th)
            n_params = count_parameters(model, feat_dim_full)
            
            print(f"      [Full TF-IDF] {m_name}: F1={m_dict['f1']:.4f}, PR-AUC={m_dict['pr_auc']:.4f}, TrainTime={t_train:.2f}s", flush=True)
            
            for seed in SEEDS:
                all_runs.append({
                    "Dataset": d_name,
                    "Representation": "Full TF-IDF",
                    "Model": m_name,
                    "Seed": seed,
                    "F1": m_dict["f1"],
                    "PR_AUC": m_dict["pr_auc"],
                    "ROC_AUC": m_dict["roc_auc"],
                    "Accuracy": m_dict["accuracy"],
                    "Balanced_Accuracy": m_dict["balanced_accuracy"],
                    "Precision": m_dict["precision"],
                    "Recall": m_dict["recall"],
                    "Optimal_Threshold": m_dict["threshold"],
                    "Train_Time_Sec": round(t_train, 4),
                    "Inference_Time_Sec": round(t_inf, 4),
                    "Peak_RAM_MB": round(peak_ram, 2),
                    "Feature_Dim": feat_dim_full,
                    "Num_Parameters": str(n_params),
                    "Convergence_Status": conv_status,
                })
                
        # Stochastic models on full TF-IDF (fit across all 10 seeds)
        for s_idx, seed in enumerate(SEEDS, 1):
            stochastic_models_full = {
                "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1),
                "XGBoost / Gradient Boosting": XGBClassifier(n_estimators=100, random_state=seed, eval_metric="logloss", n_jobs=-1),
                "MLP": MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=seed, early_stopping=True),
            }
            for m_name, model in stochastic_models_full.items():
                t0 = time.time()
                tracemalloc.start()
                model.fit(X_tr_full, y_tr)
                t_train = time.time() - t0
                curr_mem, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_ram = peak_mem / (1024 * 1024)
                
                va_scores, _ = get_scores_and_status(model, X_va_full)
                best_th, _ = select_best_threshold(y_va, va_scores)
                
                t0_inf = time.time()
                te_scores, conv_status = get_scores_and_status(model, X_te_full)
                t_inf = time.time() - t0_inf
                m_dict = evaluate_model(y_te, te_scores, best_th)
                n_params = count_parameters(model, feat_dim_full)
                
                all_runs.append({
                    "Dataset": d_name,
                    "Representation": "Full TF-IDF",
                    "Model": m_name,
                    "Seed": seed,
                    "F1": m_dict["f1"],
                    "PR_AUC": m_dict["pr_auc"],
                    "ROC_AUC": m_dict["roc_auc"],
                    "Accuracy": m_dict["accuracy"],
                    "Balanced_Accuracy": m_dict["balanced_accuracy"],
                    "Precision": m_dict["precision"],
                    "Recall": m_dict["recall"],
                    "Optimal_Threshold": m_dict["threshold"],
                    "Train_Time_Sec": round(t_train, 4),
                    "Inference_Time_Sec": round(t_inf, 4),
                    "Peak_RAM_MB": round(peak_ram, 2),
                    "Feature_Dim": feat_dim_full,
                    "Num_Parameters": str(n_params),
                    "Convergence_Status": conv_status,
                })
            print(f"      [Full TF-IDF Seed {seed} ({s_idx}/10)] Completed RF, XGBoost, MLP", flush=True)
            
        # 2. Reduced 8D TF-IDF Representation (Evaluated across all 10 seeds)
        print("  --> Representation: Reduced 8D TF-IDF (SVD + StandardScaler)")
        for s_idx, seed in enumerate(SEEDS, 1):
            svd = TruncatedSVD(n_components=8, random_state=seed)
            scaler = StandardScaler()
            X_tr_8d = scaler.fit_transform(svd.fit_transform(X_tr_full))
            X_va_8d = scaler.transform(svd.transform(X_va_full))
            X_te_8d = scaler.transform(svd.transform(X_te_full))
            feat_dim_8d = 8
            
            models_8d = {
                "Naive Bayes": GaussianNB(),
                "Logistic Regression": LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed),
                "Linear SVM": LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed),
                "RBF SVM": SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=False, cache_size=2000, random_state=seed),
                "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1),
                "XGBoost / Gradient Boosting": XGBClassifier(n_estimators=100, random_state=seed, eval_metric="logloss", n_jobs=-1),
                "MLP": MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=seed, early_stopping=True),
                "k-NN": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
            }
            
            for m_name, model in models_8d.items():
                t0 = time.time()
                tracemalloc.start()
                model.fit(X_tr_8d, y_tr)
                t_train = time.time() - t0
                curr_mem, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_ram = peak_mem / (1024 * 1024)
                
                va_scores, _ = get_scores_and_status(model, X_va_8d)
                best_th, _ = select_best_threshold(y_va, va_scores)
                
                t0_inf = time.time()
                te_scores, conv_status = get_scores_and_status(model, X_te_8d)
                t_inf = time.time() - t0_inf
                m_dict = evaluate_model(y_te, te_scores, best_th)
                n_params = count_parameters(model, feat_dim_8d)
                
                all_runs.append({
                    "Dataset": d_name,
                    "Representation": "Reduced 8D TF-IDF",
                    "Model": m_name,
                    "Seed": seed,
                    "F1": m_dict["f1"],
                    "PR_AUC": m_dict["pr_auc"],
                    "ROC_AUC": m_dict["roc_auc"],
                    "Accuracy": m_dict["accuracy"],
                    "Balanced_Accuracy": m_dict["balanced_accuracy"],
                    "Precision": m_dict["precision"],
                    "Recall": m_dict["recall"],
                    "Optimal_Threshold": m_dict["threshold"],
                    "Train_Time_Sec": round(t_train, 4),
                    "Inference_Time_Sec": round(t_inf, 4),
                    "Peak_RAM_MB": round(peak_ram, 2),
                    "Feature_Dim": feat_dim_8d,
                    "Num_Parameters": str(n_params),
                    "Convergence_Status": conv_status,
                })
            print(f"      [8D Seed {seed} ({s_idx}/10)] Completed all 8 models", flush=True)

    df_runs = pd.DataFrame(all_runs)
    df_runs.to_csv(os.path.join(EXP45_DIR, "classical_baselines.csv"), index=False)
    df_runs.to_csv(os.path.join(EXP45_DIR, "classical_baseline_seed_results.csv"), index=False)
    print(f"\n[Saved] {len(df_runs)} total classical baseline runs to classical_baselines.csv")
    
    # Compute Aggregated Summary
    summary_rows = []
    grouped = df_runs.groupby(["Dataset", "Representation", "Model"])
    for (d_name, rep_name, m_name), g in grouped:
        summary_rows.append({
            "Dataset": d_name,
            "Representation": rep_name,
            "Model": m_name,
            "Mean_F1": round(g["F1"].mean(), 4),
            "SD_F1": round(g["F1"].std(), 4),
            "Median_F1": round(g["F1"].median(), 4),
            "Min_F1": round(g["F1"].min(), 4),
            "Max_F1": round(g["F1"].max(), 4),
            "Mean_PR_AUC": round(g["PR_AUC"].mean(), 4),
            "Mean_ROC_AUC": round(g["ROC_AUC"].mean(), 4),
            "Mean_Accuracy": round(g["Accuracy"].mean(), 4),
            "Mean_Balanced_Accuracy": round(g["Balanced_Accuracy"].mean(), 4),
            "Mean_Precision": round(g["Precision"].mean(), 4),
            "Mean_Recall": round(g["Recall"].mean(), 4),
            "Mean_Train_Time": round(g["Train_Time_Sec"].mean(), 4),
            "Mean_Inference_Time": round(g["Inference_Time_Sec"].mean(), 4),
            "Peak_RAM": round(g["Peak_RAM_MB"].max(), 2),
            "Convergence_Status": "CONVERGED" if (g["Convergence_Status"] == "CONVERGED").all() else "MIXED",
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values(["Dataset", "Representation", "Mean_F1"], ascending=[True, True, False])
    df_summary.to_csv(os.path.join(EXP45_DIR, "classical_baseline_summary.csv"), index=False)
    print(f"[Saved] Summary table ({len(df_summary)} rows) to classical_baseline_summary.csv")
    
    return df_runs, df_summary


if __name__ == "__main__":
    run_benchmark()
