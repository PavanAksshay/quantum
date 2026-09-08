#!/usr/bin/env python3
"""
Experiment 25: Multi-Dataset RoBERTa Representation Benchmark
=============================================================
Evaluates pretrained RoBERTa-base feature representations across three frozen datasets:
  1. SMS Spam Collection (results/frozen_splits/sms/)
  2. CEAS-08 (results/frozen_splits/ceas/)
  3. MeAJOR (results/frozen_splits/meajor/)

Workflow:
  1. Load frozen splits (SMS full; CEAS 10k/2.5k/2.5k stratified; MeAJOR 10k/2.5k/2.5k joint source-target stratified).
  2. Save exact subset IDs under results/roberta_multidataset/subset_ids/.
  3. Extract 768-dimensional embeddings using roberta-base with masked mean pooling (max_length=256, torch.no_grad, MPS acceleration).
  4. Save numpy embeddings (.npy), labels (.npy), and sample IDs (.csv) under results/roberta_multidataset/{sms,ceas,meajor}/.
  5. Train Logistic Regression on training embeddings and evaluate on frozen test sets.
  6. Fit StandardScaler and PCA strictly on training embeddings and compute cumulative explained variance for [2, 4, 6, 8, 16, 32, 64] components.
  7. Save benchmark metrics to results/metrics/roberta_multidataset.csv and PCA variance table to results/metrics/roberta_multidataset_pca.csv.

Author: Quantum Phishing & Scam Detection Project
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================
RANDOM_SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 64
PCA_COMPONENTS_LIST = [2, 4, 6, 8, 16, 32, 64]

FROZEN_SPLITS_DIR = "results/frozen_splits"
BASE_ROBERTA_DIR = "results/roberta_multidataset"
SUBSET_IDS_DIR = os.path.join(BASE_ROBERTA_DIR, "subset_ids")
METRICS_DIR = "results/metrics"

FULL_METRICS_PATH = os.path.join(METRICS_DIR, "roberta_multidataset.csv")
PCA_METRICS_PATH = os.path.join(METRICS_DIR, "roberta_multidataset_pca.csv")


def get_device() -> torch.device:
    """Select best available acceleration device (MPS -> CUDA -> CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def prepare_data_subsets() -> dict:
    """
    Prepare data splits for SMS, CEAS, and MeAJOR:
      - SMS: Full frozen splits
      - CEAS: Deterministic stratified subset (10k train, 2.5k val, 2.5k test)
      - MeAJOR: Deterministic joint source+target stratified subset (10k train, 2.5k val, 2.5k test)
    """
    os.makedirs(SUBSET_IDS_DIR, exist_ok=True)
    datasets = {}

    # 1. SMS (Full frozen splits)
    sms_tr = pd.read_csv(os.path.join(FROZEN_SPLITS_DIR, "sms/train.csv"))
    sms_va = pd.read_csv(os.path.join(FROZEN_SPLITS_DIR, "sms/validation.csv"))
    sms_te = pd.read_csv(os.path.join(FROZEN_SPLITS_DIR, "sms/test.csv"))

    sms_tr["sample_id"].to_csv(os.path.join(SUBSET_IDS_DIR, "sms_train_ids.csv"), index=False)
    sms_va["sample_id"].to_csv(os.path.join(SUBSET_IDS_DIR, "sms_val_ids.csv"), index=False)
    sms_te["sample_id"].to_csv(os.path.join(SUBSET_IDS_DIR, "sms_test_ids.csv"), index=False)

    datasets["SMS"] = {"train": sms_tr, "validation": sms_va, "test": sms_te}

    # 2. CEAS (Stratified subset: 10,000 Train / 2,500 Val / 2,500 Test)
    ceas_tr_full = pd.read_csv(os.path.join(FROZEN_SPLITS_DIR, "ceas/train.csv"))
    ceas_va_full = pd.read_csv(os.path.join(FROZEN_SPLITS_DIR, "ceas/validation.csv"))
    ceas_te_full = pd.read_csv(os.path.join(FROZEN_SPLITS_DIR, "ceas/test.csv"))

    ceas_tr, _ = train_test_split(ceas_tr_full, train_size=10000, stratify=ceas_tr_full["target"], random_state=RANDOM_SEED)
    ceas_va, _ = train_test_split(ceas_va_full, train_size=2500, stratify=ceas_va_full["target"], random_state=RANDOM_SEED)
    ceas_te, _ = train_test_split(ceas_te_full, train_size=2500, stratify=ceas_te_full["target"], random_state=RANDOM_SEED)

    ceas_tr["sample_id"].to_csv(os.path.join(SUBSET_IDS_DIR, "ceas_train_ids.csv"), index=False)
    ceas_va["sample_id"].to_csv(os.path.join(SUBSET_IDS_DIR, "ceas_val_ids.csv"), index=False)
    ceas_te["sample_id"].to_csv(os.path.join(SUBSET_IDS_DIR, "ceas_test_ids.csv"), index=False)

    datasets["CEAS"] = {"train": ceas_tr.reset_index(drop=True), "validation": ceas_va.reset_index(drop=True), "test": ceas_te.reset_index(drop=True)}

    # 3. MeAJOR (Joint Source + Target Stratified subset: 10,000 Train / 2,500 Val / 2,500 Test)
    meajor_tr_full = pd.read_csv(os.path.join(FROZEN_SPLITS_DIR, "meajor/train.csv"))
    meajor_va_full = pd.read_csv(os.path.join(FROZEN_SPLITS_DIR, "meajor/validation.csv"))
    meajor_te_full = pd.read_csv(os.path.join(FROZEN_SPLITS_DIR, "meajor/test.csv"))
    meajor_meta = pd.read_csv(os.path.join(FROZEN_SPLITS_DIR, "meajor/metadata.csv")).set_index("sample_id")

    meajor_tr_full["source"] = meajor_meta.loc[meajor_tr_full["sample_id"], "source"].values
    meajor_va_full["source"] = meajor_meta.loc[meajor_va_full["sample_id"], "source"].values
    meajor_te_full["source"] = meajor_meta.loc[meajor_te_full["sample_id"], "source"].values

    meajor_tr_full["strata"] = meajor_tr_full["source"] + "_" + meajor_tr_full["target"].astype(str)
    meajor_va_full["strata"] = meajor_va_full["source"] + "_" + meajor_va_full["target"].astype(str)
    meajor_te_full["strata"] = meajor_te_full["source"] + "_" + meajor_te_full["target"].astype(str)

    meajor_tr, _ = train_test_split(meajor_tr_full, train_size=10000, stratify=meajor_tr_full["strata"], random_state=RANDOM_SEED)
    meajor_va, _ = train_test_split(meajor_va_full, train_size=2500, stratify=meajor_va_full["strata"], random_state=RANDOM_SEED)
    meajor_te, _ = train_test_split(meajor_te_full, train_size=2500, stratify=meajor_te_full["strata"], random_state=RANDOM_SEED)

    meajor_tr["sample_id"].to_csv(os.path.join(SUBSET_IDS_DIR, "meajor_train_ids.csv"), index=False)
    meajor_va["sample_id"].to_csv(os.path.join(SUBSET_IDS_DIR, "meajor_val_ids.csv"), index=False)
    meajor_te["sample_id"].to_csv(os.path.join(SUBSET_IDS_DIR, "meajor_test_ids.csv"), index=False)

    datasets["MeAJOR"] = {"train": meajor_tr.reset_index(drop=True), "validation": meajor_va.reset_index(drop=True), "test": meajor_te.reset_index(drop=True)}

    return datasets


def extract_roberta_embeddings(
    texts: list,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract 768-dimensional embeddings using masked mean pooling of final hidden states.
    """
    embeddings = []
    n_samples = len(texts)

    for i in range(0, n_samples, batch_size):
        batch_texts = [str(t) if pd.notna(t) else "" for t in texts[i : i + batch_size]]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state  # shape: (B, Seq_Len, 768)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()
            embeddings.append(mean_pooled)

    return np.vstack(embeddings)


def main():
    print("=" * 60, flush=True)
    print("EXPERIMENT 25: MULTI-DATASET ROBERTA REPRESENTATION BENCHMARK", flush=True)
    print("=" * 60, flush=True)

    device = get_device()
    print(f"Hardware Acceleration Device: {device}", flush=True)

    # 1. Prepare and store data subsets
    print("\nPreparing frozen datasets and deterministic subsets...", flush=True)
    datasets = prepare_data_subsets()

    for ds_name, splits in datasets.items():
        print(f"  {ds_name:<8}: Train={len(splits['train'])} | Val={len(splits['validation'])} | Test={len(splits['test'])}", flush=True)

    # 2. Load RoBERTa-base
    print("\nLoading pretrained roberta-base model and tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModel.from_pretrained("roberta-base")
    model.to(device)
    model.eval()

    os.makedirs(METRICS_DIR, exist_ok=True)
    benchmark_records = []
    pca_records = []

    # 3. Extract embeddings, fit validation classifier, and run PCA
    for ds_name, splits in datasets.items():
        print("\n" + "=" * 60, flush=True)
        print(f"PROCESSING DATASET: {ds_name}", flush=True)
        print("=" * 60, flush=True)

        ds_out_dir = os.path.join(BASE_ROBERTA_DIR, ds_name.lower())
        os.makedirs(ds_out_dir, exist_ok=True)

        emb_dict = {}
        labels_dict = {}
        ids_dict = {}

        for split_name in ["train", "validation", "test"]:
            df_split = splits[split_name]
            texts = df_split["text"].tolist()
            labels = df_split["target"].astype(int).values
            sample_ids = df_split["sample_id"].tolist()

            print(f"Extracting {ds_name} {split_name} embeddings (N = {len(texts)})...", flush=True)
            t0 = time.time()
            emb = extract_roberta_embeddings(texts, tokenizer, model, device, batch_size=BATCH_SIZE)
            t_elapsed = time.time() - t0
            print(f"  Shape: {emb.shape} | Time: {t_elapsed:.2f}s ({len(texts)/t_elapsed:.1f} samples/s)", flush=True)

            # Save arrays and sample IDs
            np.save(os.path.join(ds_out_dir, f"{split_name}_embeddings.npy"), emb)
            np.save(os.path.join(ds_out_dir, f"{split_name}_labels.npy"), labels)
            pd.DataFrame({"sample_id": sample_ids}).to_csv(
                os.path.join(ds_out_dir, f"{split_name}_sample_ids.csv"), index=False
            )

            emb_dict[split_name] = emb
            labels_dict[split_name] = labels
            ids_dict[split_name] = sample_ids

        # 4. Train Validation Classifier (Logistic Regression)
        X_tr = emb_dict["train"]
        y_tr = labels_dict["train"]
        X_va = emb_dict["validation"]
        y_va = labels_dict["validation"]
        X_te = emb_dict["test"]
        y_te = labels_dict["test"]

        print(f"\nTraining Logistic Regression classifier on {ds_name} RoBERTa embeddings...", flush=True)
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )

        t0_train = time.time()
        clf.fit(X_tr, y_tr)
        t_train = time.time() - t0_train

        # Inference on Test set
        t0_infer = time.time()
        y_pred = clf.predict(X_te)
        t_infer = time.time() - t0_infer

        scores = clf.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        roc = roc_auc_score(y_te, scores)
        pr_prec, pr_rec, _ = precision_recall_curve(y_te, scores)
        pr_auc = auc(pr_rec, pr_prec)

        print(f"Test Evaluation ({ds_name}):", flush=True)
        print(f"  Accuracy:  {acc:.4f}", flush=True)
        print(f"  Precision: {prec:.4f}", flush=True)
        print(f"  Recall:    {rec:.4f}", flush=True)
        print(f"  F1 Score:  {f1:.4f}", flush=True)
        print(f"  PR-AUC:    {pr_auc:.4f}", flush=True)
        print(f"  ROC-AUC:   {roc:.4f}", flush=True)
        print(f"  Train Time:{t_train:.3f}s | Infer Time: {t_infer:.4f}s", flush=True)

        benchmark_records.append({
            "Dataset": ds_name,
            "Train_Samples": len(X_tr),
            "Validation_Samples": len(X_va),
            "Test_Samples": len(X_te),
            "Embedding_Dimension": X_tr.shape[1],
            "Accuracy": round(acc, 6),
            "Precision": round(prec, 6),
            "Recall": round(rec, 6),
            "F1": round(f1, 6),
            "PR_AUC": round(pr_auc, 6),
            "ROC_AUC": round(roc, 6),
            "Training_Time": round(t_train, 4),
            "Inference_Time": round(t_infer, 4),
        })

        # 5. PCA Explained Variance Analysis
        print(f"\nComputing PCA cumulative explained variance for {ds_name}...", flush=True)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)

        max_comp = max(PCA_COMPONENTS_LIST)
        pca = PCA(n_components=max_comp, random_state=RANDOM_SEED)
        pca.fit(X_tr_scaled)

        cum_var = np.cumsum(pca.explained_variance_ratio_)

        print(f"{'Components':<12} | {'Cumulative Explained Variance':<30}", flush=True)
        print("-" * 45, flush=True)
        for k in PCA_COMPONENTS_LIST:
            var_k = cum_var[k - 1]
            print(f"{k:<12} | {var_k * 100:>28.2f}%", flush=True)
            pca_records.append({
                "Dataset": ds_name,
                "Components": k,
                "Cumulative_Explained_Variance": round(float(var_k), 6),
                "Cumulative_Explained_Variance_Pct": round(float(var_k * 100), 2),
            })

    # Save CSV outputs
    df_bench = pd.DataFrame(benchmark_records)
    df_bench.to_csv(FULL_METRICS_PATH, index=False)
    print(f"\nSaved benchmark metrics to: {FULL_METRICS_PATH}", flush=True)

    df_pca = pd.DataFrame(pca_records)
    df_pca.to_csv(PCA_METRICS_PATH, index=False)
    print(f"Saved PCA explained variance table to: {PCA_METRICS_PATH}", flush=True)

    # ============================================================
    # FINAL REPORT
    # ============================================================
    print("\n" + "=" * 60, flush=True)
    print("ROBERTA MULTI-DATASET RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Dataset':<10} | {'Samples':<9} | {'F1':<7} | {'PR-AUC':<7} | {'ROC-AUC':<7} | {'Time (s)':<8}", flush=True)
    print("-" * 65, flush=True)
    for r in benchmark_records:
        total_samples = r["Train_Samples"] + r["Validation_Samples"] + r["Test_Samples"]
        tot_time = r["Training_Time"] + r["Inference_Time"]
        print(
            f"{r['Dataset']:<10} | {total_samples:<9} | {r['F1']:<7.4f} | {r['PR_AUC']:<7.4f} | {r['ROC_AUC']:<7.4f} | {tot_time:<8.3f}",
            flush=True,
        )

    print("\n" + "=" * 60, flush=True)
    print("PCA EXPLAINED VARIANCE SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Dataset':<10} | {'Components':<10} | {'Explained Variance':<20}", flush=True)
    print("-" * 48, flush=True)
    for r in pca_records:
        print(f"{r['Dataset']:<10} | {r['Components']:<10} | {r['Cumulative_Explained_Variance_Pct']:>17.2f}%", flush=True)

    print("\nRoBERTa representation benchmark complete.", flush=True)


if __name__ == "__main__":
    main()
