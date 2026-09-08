"""
=============================================================================
19_nlp_preprocessing_benchmark.py
=============================================================================
Rigorous NLP Preprocessing Benchmark for SMS Spam/Scam Detection.

Compares 4 text preprocessing strategies:
    1. RAW: Original text (matches existing classical TF-IDF baseline)
    2. CLEANED: Lightweight normalization (URLs, phones, numbers, currency,
                contractions, character repetition, preserved punctuation)
    3. STEMMED: CLEANED + NLTK PorterStemmer (with domain-aware stopwords)
    4. LEMMATIZED: CLEANED + Contextual POS-tagged NLTK WordNetLemmatizer

Evaluates across 7 classical machine learning classifiers:
    - Multinomial Naive Bayes
    - Logistic Regression
    - Linear SVM
    - RBF SVM
    - Random Forest
    - XGBoost
    - MLP (Multi-Layer Perceptron)

Outputs:
    - results/metrics/preprocessing_benchmark.csv
    - results/metrics/preprocessing_summary.csv

Author: Antigravity AI & Research Team
=============================================================================
"""

import os
import sys
import time
import re
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
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

# NLTK components
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as nltk_stopwords

# Configure warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIGURATION & REPRODUCIBILITY
# =============================================================================
RANDOM_STATE = 42
MAX_FEATURES = 30000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
TEST_SIZE = 0.20

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


# =============================================================================
# NLTK SETUP & INITIALIZATION
# =============================================================================
def init_nltk():
    """Ensure required NLTK data packages are loaded."""
    required_packages = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/stopwords", "stopwords")
    ]
    for path_check, pkg_name in required_packages:
        try:
            nltk.data.find(path_check)
        except LookupError:
            try:
                nltk.download(pkg_name, quiet=True)
            except Exception as e:
                print(f"[Warning] Could not download NLTK package {pkg_name}: {e}")

init_nltk()

# Initialize Stemmer & Lemmatizer
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

# Thoughtful stopword selection: retain critical negations and spam triggers
RAW_STOPWORDS = set(nltk_stopwords.words("english"))
PRESERVED_WORDS = {
    "no", "not", "nor", "won", "won't", "don", "don't", "free", "urgent",
    "win", "winner", "prize", "cash", "claim", "call", "now", "stop", "txt",
    "mobile", "reply", "claim", "contact", "order", "credit", "money"
}
STOPWORDS = RAW_STOPWORDS - PRESERVED_WORDS

# Contraction mapping for standard SMS/English expansion
CONTRACTIONS = {
    r"\bcan['’]t\b": "cannot",
    r"\bwon['’]t\b": "will not",
    r"\bdon['’]t\b": "do not",
    r"\bdidn['’]t\b": "did not",
    r"\bdoesn['’]t\b": "does not",
    r"\bisn['’]t\b": "is not",
    r"\baren['’]t\b": "are not",
    r"\bwasn['’]t\b": "was not",
    r"\bweren['’]t\b": "were not",
    r"\bhaven['’]t\b": "have not",
    r"\bhasn['’]t\b": "has not",
    r"\bhadn['’]t\b": "had not",
    r"\bwouldn['’]t\b": "would not",
    r"\bshouldn['’]t\b": "should not",
    r"\bcouldn['’]t\b": "could not",
    r"\bi['’]m\b": "i am",
    r"\byou['’]re\b": "you are",
    r"\bhe['’]s\b": "he is",
    r"\bshe['’]s\b": "she is",
    r"\bit['’]s\b": "it is",
    r"\bwe['’]re\b": "we are",
    r"\bthey['’]re\b": "they are",
    r"\bi['’]ve\b": "i have",
    r"\byou['’]ve\b": "you have",
    r"\bwe['’]ve\b": "we have",
    r"\bthey['’]ve\b": "they have",
    r"\bi['’]ll\b": "i will",
    r"\byou['’]ll\b": "you will",
    r"\bhe['’]ll\b": "he will",
    r"\bshe['’]ll\b": "she will",
    r"\bwe['’]ll\b": "we will",
    r"\bthey['’]ll\b": "they will",
    r"\bu\b": "you",
    r"\bur\b": "your",
    r"\br\b": "are",
    r"\b2\b": "to",
    r"\b4\b": "for",
    r"\btxt\b": "text",
    r"\bmsg\b": "message",
    r"\bpls\b": "please",
    r"\bplz\b": "please",
}


# =============================================================================
# DATASET LOADING & SPLITTING
# =============================================================================
def find_dataset() -> str:
    """Locate the dataset file."""
    possible_paths = [
        "data/SMSSpamCollection",
        "data/spam.csv",
        "data/dataset.csv",
        "data/spam_dataset.csv",
        "data/messages.csv",
    ]
    for path in possible_paths:
        full_path = os.path.join(PROJECT_ROOT, path) if not os.path.isabs(path) else path
        if os.path.exists(full_path):
            return full_path
    raise FileNotFoundError("Could not find dataset SMSSpamCollection.")


def load_dataset() -> pd.DataFrame:
    """Load and format the SMS Spam dataset matching classical benchmark."""
    path = find_dataset()
    print(f"Loading dataset: {path}")

    if path.endswith("SMSSpamCollection"):
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["label", "text"],
            encoding="utf-8"
        )
    else:
        df = pd.read_csv(path)

    # Detect text and label columns
    text_col = next((c for c in df.columns if str(c).lower() in ["text", "message", "sms", "content"]), None)
    label_col = next((c for c in df.columns if str(c).lower() in ["label", "category", "class", "target"]), None)

    if not text_col or not label_col:
        raise ValueError(f"Unable to detect text/label columns from {df.columns}")

    df = df[[label_col, text_col]].copy()
    df.columns = ["label", "text"]
    df = df.dropna()
    df["text"] = df["text"].astype(str)

    def convert_label(label: Any) -> int:
        val = str(label).strip().lower()
        if val in ["spam", "1", "phishing", "scam"]:
            return 1
        if val in ["ham", "0", "legitimate", "normal"]:
            return 0
        raise ValueError(f"Unknown label value: {label}")

    df["target"] = df["label"].apply(convert_label)

    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution: {dict(df['target'].value_counts())}")
    return df


def create_split(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform stratified train/test split with identical random seed."""
    X = df["text"].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Train samples: {len(X_train)} (Distribution: {dict(pd.Series(y_train).value_counts())})")
    print(f"Test samples:  {len(X_test)} (Distribution: {dict(pd.Series(y_test).value_counts())})")
    return X_train, X_test, y_train, y_test


# =============================================================================
# PREPROCESSING IMPLEMENTATIONS
# =============================================================================
def preprocess_raw(text: str) -> str:
    """
    Variant 1: RAW
    Minimal processing, preserving original text verbatim.
    """
    return str(text)


def preprocess_cleaned(text: str) -> str:
    """
    Variant 2: CLEANED
    Spam-aware normalization:
    - Normalizes URLs -> URLTOKEN
    - Normalizes Email addresses -> EMAILTOKEN
    - Normalizes Phone numbers / shortcodes -> PHONETOKEN
    - Normalizes Currency amounts -> CURRTOKEN NUMTOKEN
    - Normalizes Numbers -> NUMTOKEN
    - Expands SMS contractions
    - Reduces repeated characters (e.g. 'coooool' -> 'cool', '!!!!!' -> '!!')
    - Preserves expressive punctuation (!, ?, %, etc.)
    - Normalizes whitespace & lowercase
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Contraction expansion (case-insensitive)
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 2. Lowercase
    text = text.lower()

    # 3. URLs
    text = re.sub(r"https?://\S+|www\.\S+|bit\.ly/\S+|tinyurl\.com/\S+", " URLTOKEN ", text)

    # 4. Email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " EMAILTOKEN ", text)

    # 5. Currency symbols and amounts (e.g. £1000, $500, €50, 50p)
    text = re.sub(r"[£$€¥₹]\s*\d+([.,]\d+)?", " CURRTOKEN NUMTOKEN ", text)
    text = re.sub(r"\b\d+([.,]\d+)?\s*(pounds?|dollars?|euros?|quid|pence|p)\b", " NUMTOKEN CURRTOKEN ", text)
    text = re.sub(r"[£$€¥₹]", " CURRTOKEN ", text)

    # 6. Phone numbers / UK shortcodes / international mobile formats
    # 5-digit premium shortcodes (e.g. 88066, 86688) or 10-12 digit phone numbers
    text = re.sub(r"\b(?:\+?44|0044|0)?7\d{9}\b", " PHONETOKEN ", text)
    text = re.sub(r"\b\d{5,6}\b", " PHONETOKEN ", text)
    text = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", " PHONETOKEN ", text)

    # 7. Remaining numeric sequences
    text = re.sub(r"\b\d+([.,]\d+)?\b", " NUMTOKEN ", text)

    # 8. Character repetition reduction (>=3 consecutive identical chars -> 2 chars)
    # e.g., 'freereee' -> 'free', 'hurrrry' -> 'hurry', '!!!!!!' -> '!!'
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 9. Isolate punctuation to maintain n-gram boundary signals (!, ?, %)
    text = re.sub(r"([!?,%])", r" \1 ", text)

    # 10. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_stemmed(text: str) -> str:
    """
    Variant 3: STEMMED
    CLEANED preprocessing followed by PorterStemmer with domain-aware stopword handling.
    """
    cleaned = preprocess_cleaned(text)
    # Tokenize words while keeping special tokens and punctuation intact
    tokens = re.findall(r"\b[A-Za-z0-9_]+\b|[!?,%]", cleaned)

    stemmed_tokens = []
    special_tokens = {"urltoken", "emailtoken", "currtoken", "phonetoken", "numtoken"}

    for token in tokens:
        token_lower = token.lower()
        if token_lower in special_tokens or token in {"!", "?", ",", "%"}:
            stemmed_tokens.append(token)
        elif token_lower not in STOPWORDS:
            stemmed_tokens.append(STEMMER.stem(token_lower))

    return " ".join(stemmed_tokens)


def _get_wordnet_pos(treebank_tag: str) -> str:
    """Map Penn Treebank POS tags to WordNet POS constants."""
    if treebank_tag.startswith("J"):
        return wn.ADJ
    elif treebank_tag.startswith("V"):
        return wn.VERB
    elif treebank_tag.startswith("N"):
        return wn.NOUN
    elif treebank_tag.startswith("R"):
        return wn.ADV
    else:
        return wn.NOUN


def preprocess_lemmatized(text: str) -> str:
    """
    Variant 4: LEMMATIZED
    CLEANED preprocessing followed by WordNet lemmatization with contextual POS tagging.
    """
    cleaned = preprocess_cleaned(text)
    tokens = re.findall(r"\b[A-Za-z0-9_]+\b|[!?,%]", cleaned)

    # Separate words from punctuation for POS tagging
    special_tokens = {"urltoken", "emailtoken", "currtoken", "phonetoken", "numtoken"}
    words_to_tag = [t for t in tokens if t not in {"!", "?", ",", "%"} and t.lower() not in special_tokens]

    pos_tags = dict(nltk.pos_tag(words_to_tag)) if words_to_tag else {}

    lemmatized_tokens = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in special_tokens or token in {"!", "?", ",", "%"}:
            lemmatized_tokens.append(token)
        elif token_lower not in STOPWORDS:
            ptb_tag = pos_tags.get(token, "NN")
            wn_tag = _get_wordnet_pos(ptb_tag)
            lemma = LEMMATIZER.lemmatize(token_lower, pos=wn_tag)
            lemmatized_tokens.append(lemma)

    return " ".join(lemmatized_tokens)


PREPROCESSING_PIPELINES = {
    "Raw": preprocess_raw,
    "Cleaned": preprocess_cleaned,
    "Stemmed": preprocess_stemmed,
    "Lemmatized": preprocess_lemmatized,
}


def apply_preprocessing(texts: np.ndarray, method_name: str) -> List[str]:
    """Batch apply a preprocessing function across a corpus."""
    transform_fn = PREPROCESSING_PIPELINES[method_name]
    return [transform_fn(t) for t in texts]


# =============================================================================
# MODEL EVALUATION & FACTORY
# =============================================================================
def get_models() -> Dict[str, Any]:
    """Return dictionary of classical model instances with fixed seeds."""
    return {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            C=2.0,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Linear SVM": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "RBF SVM": SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
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
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
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


def evaluate_model(model: Any, X_test: Any, y_test: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive classification metrics."""
    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        probabilities = model.decision_function(X_test)
    else:
        probabilities = predictions

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "PR_AUC": average_precision_score(y_test, probabilities),
        "ROC_AUC": roc_auc_score(y_test, probabilities)
    }


# =============================================================================
# BENCHMARK EXECUTION
# =============================================================================
def run_benchmark() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete 4 Preprocessing x 7 Model benchmark."""
    print("=" * 80)
    print("STARTING NLP PREPROCESSING BENCHMARK EXPERIMENT")
    print("=" * 80)

    # 1. Load Data
    df = load_dataset()
    X_train_raw, X_test_raw, y_train, y_test = create_split(df)

    benchmark_rows = []

    for prep_name in ["Raw", "Cleaned", "Stemmed", "Lemmatized"]:
        print("\n" + "=" * 80)
        print(f"PREPROCESSING STRATEGY: {prep_name.upper()}")
        print("=" * 80)

        # Preprocess text
        prep_start = time.time()
        X_train_prep = apply_preprocessing(X_train_raw, prep_name)
        X_test_prep = apply_preprocessing(X_test_raw, prep_name)
        prep_time = time.time() - prep_start
        print(f"Applied {prep_name} preprocessing to {len(X_train_prep) + len(X_test_prep)} messages in {prep_time:.2f}s")

        # Show sample transformed message
        sample_idx = 0
        print(f"Sample [{prep_name}]: {X_train_prep[sample_idx][:90]}...")

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            min_df=MIN_DF
        )

        tfidf_start = time.time()
        X_train_vec = vectorizer.fit_transform(X_train_prep)
        X_test_vec = vectorizer.transform(X_test_prep)
        tfidf_fit_time = time.time() - tfidf_start

        vocab_size = len(vectorizer.vocabulary_)
        print(f"TF-IDF Fitted. Vocabulary size: {vocab_size} | Fit time: {tfidf_fit_time:.4f}s")
        print(f"Train matrix shape: {X_train_vec.shape}, Test matrix shape: {X_test_vec.shape}")

        # Train & Evaluate Models
        models = get_models()

        for model_name, model in models.items():
            print(f"  Training {model_name:<20} ...", end="", flush=True)

            # Fit
            train_start = time.time()
            model.fit(X_train_vec, y_train)
            train_time = time.time() - train_start

            # Infer
            infer_start = time.time()
            metrics = evaluate_model(model, X_test_vec, y_test)
            infer_time = time.time() - infer_start

            print(f" Done. F1: {metrics['F1']:.4f} | PR-AUC: {metrics['PR_AUC']:.4f} | ROC-AUC: {metrics['ROC_AUC']:.4f}")

            benchmark_rows.append({
                "Preprocessing": prep_name,
                "Model": model_name,
                "Pipeline": f"{prep_name} + {model_name}",
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "PR_AUC": metrics["PR_AUC"],
                "ROC_AUC": metrics["ROC_AUC"],
                "TFIDF_Vocab_Size": vocab_size,
                "TFIDF_Fit_Time": tfidf_fit_time,
                "Training_Time": train_time,
                "Inference_Time": infer_time
            })

    benchmark_df = pd.DataFrame(benchmark_rows)

    # =========================================================================
    # CREATE SUMMARY: BEST PREPROCESSING PER CLASSIFIER
    # =========================================================================
    summary_rows = []
    for model_name in get_models().keys():
        model_subset = benchmark_df[benchmark_df["Model"] == model_name]
        raw_row = model_subset[model_subset["Preprocessing"] == "Raw"].iloc[0]

        # Best based primarily on F1, secondarily on PR-AUC
        best_row = model_subset.sort_values(by=["F1", "PR_AUC", "ROC_AUC"], ascending=False).iloc[0]

        f1_delta = best_row["F1"] - raw_row["F1"]
        prauc_delta = best_row["PR_AUC"] - raw_row["PR_AUC"]
        rocauc_delta = best_row["ROC_AUC"] - raw_row["ROC_AUC"]

        summary_rows.append({
            "Model": model_name,
            "Best_Preprocessing": best_row["Preprocessing"],
            "Best_Pipeline": best_row["Pipeline"],
            "Best_F1": best_row["F1"],
            "Raw_F1": raw_row["F1"],
            "F1_Delta": f1_delta,
            "Best_PR_AUC": best_row["PR_AUC"],
            "Raw_PR_AUC": raw_row["PR_AUC"],
            "PR_AUC_Delta": prauc_delta,
            "Best_ROC_AUC": best_row["ROC_AUC"],
            "Raw_ROC_AUC": raw_row["ROC_AUC"],
            "ROC_AUC_Delta": rocauc_delta,
            "Improved_Over_Raw": f1_delta > 0.0001
        })

    summary_df = pd.DataFrame(summary_rows)
    return benchmark_df, summary_df


# =============================================================================
# SAVE RESULTS & DISPLAY OUTPUTS
# =============================================================================
def save_and_display(benchmark_df: pd.DataFrame, summary_df: pd.DataFrame):
    """Save results to CSVs and print formatted benchmark tables."""
    output_dir = os.path.join(PROJECT_ROOT, "results", "metrics")
    os.makedirs(output_dir, exist_ok=True)

    benchmark_path = os.path.join(output_dir, "preprocessing_benchmark.csv")
    summary_path = os.path.join(output_dir, "preprocessing_summary.csv")

    benchmark_df.to_csv(benchmark_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n[Saved] Full benchmark saved to: {benchmark_path}")
    print(f"[Saved] Summary table saved to:   {summary_path}")

    # Display comparison table requested
    print("\n")
    print("=" * 80)
    print("PREPROCESSING COMPARISON")
    print("=" * 80)
    display_table = benchmark_df[[
        "Pipeline", "Accuracy", "Precision", "Recall", "F1", "PR_AUC", "ROC_AUC"
    ]].copy()
    display_table.columns = ["Model / Pipeline", "Accuracy", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]

    # Format numbers for readability
    for col in ["Accuracy", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]:
        display_table[col] = display_table[col].apply(lambda x: f"{x:.4f}")

    print(display_table.to_string(index=False))

    # Summary table
    print("\n")
    print("=" * 80)
    print("BEST PREPROCESSING METHOD PER CLASSIFIER")
    print("=" * 80)
    sum_display = summary_df[[
        "Model", "Best_Preprocessing", "Best_F1", "Raw_F1", "F1_Delta", "Best_PR_AUC", "Best_ROC_AUC", "Improved_Over_Raw"
    ]].copy()
    for col in ["Best_F1", "Raw_F1", "F1_Delta", "Best_PR_AUC", "Best_ROC_AUC"]:
        sum_display[col] = sum_display[col].apply(lambda x: f"{x:+.4f}" if "Delta" in col else f"{x:.4f}")
    print(sum_display.to_string(index=False))

    # Best overall winners
    best_overall_row = benchmark_df.sort_values(by=["F1", "PR_AUC", "ROC_AUC"], ascending=False).iloc[0]
    best_f1_row = benchmark_df.loc[benchmark_df["F1"].idxmax()]
    best_prauc_row = benchmark_df.loc[benchmark_df["PR_AUC"].idxmax()]
    best_rocauc_row = benchmark_df.loc[benchmark_df["ROC_AUC"].idxmax()]

    # Average F1 per preprocessing strategy
    avg_f1_by_prep = benchmark_df.groupby("Preprocessing")["F1"].mean().sort_values(ascending=False)
    best_prep_by_avg = avg_f1_by_prep.index[0]

    print("\n" + "=" * 80)
    print("FINAL RESEARCH SUMMARY & VERDICT")
    print("=" * 80)
    print(f"BEST OVERALL PREPROCESSING: {best_prep_by_avg} (Mean F1 across models: {avg_f1_by_prep[best_prep_by_avg]:.4f})")
    print(f"BEST MODEL / PIPELINE:     {best_overall_row['Pipeline']}")
    print(f"BEST F1:                   {best_f1_row['F1']:.4f} ({best_f1_row['Pipeline']})")
    print(f"BEST PR-AUC:               {best_prauc_row['PR_AUC']:.4f} ({best_prauc_row['Pipeline']})")
    print(f"BEST ROC-AUC:              {best_rocauc_row['ROC_AUC']:.4f} ({best_rocauc_row['Pipeline']})")
    print("-" * 80)
    print("MEAN F1 BY PREPROCESSING STRATEGY:")
    for prep, score in avg_f1_by_prep.items():
        print(f"  - {prep:<12}: {score:.4f}")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    benchmark_df, summary_df = run_benchmark()
    save_and_display(benchmark_df, summary_df)
