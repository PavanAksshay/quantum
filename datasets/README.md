# Quantum Text Security Benchmark Datasets Manifest

This directory contains all standardized raw corpora and canonical frozen split partitions in `.csv` format used in the empirical study:
**Evaluating a Parameter-Free Quantum Fidelity Kernel for Text Security Under Representation and Domain Shift**.

---

## 1. Raw Source Datasets
| File Name | Description | Total Records | Positive Class % | Source Domain |
| :--- | :--- | :---: | :---: | :--- |
| `sms_spam_raw.csv` | Full SMS Spam Collection | 5,572 | 13.41% | Mobile SMS communications |
| `ceas_2008_raw.csv` | Full CEAS 2008 Phishing/Ham Archive | 39,154 | 55.78% | Enterprise email challenge |
| `meajor_archive_raw.csv` | Multi-Source Email Archive (TREC 5/6/7) | 108,685 | 44.20% | TREC 2005, 2006, 2007 feeds |

---

## 2. Canonical Controlled Experimental Partitions
All splits enforce strict stratification, deterministic pseudo-random shuffling across 10 frozen seeds, and leakage-safe preprocessing (feature extraction and dimensionality reduction fitted strictly on training subsets).

### SMS Spam Partitions ($N = 5,572$)
- `sms_split_train.csv`: 3,343 samples (60%)
- `sms_split_validation.csv`: 1,114 samples (20%)
- `sms_split_test.csv`: 1,115 samples (20%)

### CEAS 2008 Partitions ($N = 15,000$ Controlled Canonical Subset)
- `ceas_2008_split_train.csv`: 10,000 samples (Spam Prevalence: 18.90%)
- `ceas_2008_split_validation.csv`: 2,500 samples (Spam Prevalence: 18.90%)
- `ceas_2008_split_test.csv`: 2,500 samples (Spam Prevalence: 18.90%)

### MeAJOR In-Distribution (IID) Partitions ($N = 15,000$ Canonical Subset)
- `meajor_iid_split_train.csv`: 10,000 samples (Spam Prevalence: 19.33%)
- `meajor_iid_split_validation.csv`: 2,500 samples (Spam Prevalence: 19.33%)
- `meajor_iid_split_test.csv`: 2,500 samples (Spam Prevalence: 19.33%)

### MeAJOR Cross-Source Domain Shift Partitions (Direction B: TREC 2007 $	o$ TREC 2005/2006)
- `meajor_holdout_train_trec7.csv`: 10,000 training samples from TREC 2007
- `meajor_holdout_test_trec5_trec6.csv`: 5,000 out-of-distribution test samples (balanced mixture of TREC 2005 and TREC 2006)

---

## 3. Data Hygiene Protocol
1. **Header Stripping**: Auxiliary metadata headers stripped to isolate core natural language body content.
2. **Zero Leakage**: Tokenizers, TF-IDF vectorizers (50,000 n-grams), TruncatedSVD projections ($d \in [2, 12]$), and StandardScalers fitted strictly on training splits.
3. **Threshold Tuning**: Decision thresholds $\tau \in [0.01, 0.99]$ tuned strictly on validation partitions and applied unconditionally to test evaluation.
