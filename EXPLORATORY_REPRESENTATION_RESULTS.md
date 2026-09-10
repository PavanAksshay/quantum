# Exploratory Representation Results Register

**Project**: When Do Quantum Kernels Help for Text Security?  
**Standard**: Exploratory Representation Ablation Subsystem  
**Status**: ACTIVE REGISTER (Decoupled from Frozen Paper Claims)  

---

## 1. Scope & Scientific Protocol
This document registers candidate exploratory representation models evaluated within the interactive Representation Lab. In accordance with the project's freeze protocol (`RESULT_FREEZE.md`), these exploratory ablations do **not** modify the frozen manuscript claims, abstract, conclusion, or evidence matrix unless validated through 10-seed paired replication and explicitly versioned.

### Evaluation Criteria:
1. **Upstream Embedding Type**: Subword / Dense contextual / Sparse TF-IDF.
2. **Fixed Dimensionality Projection**: Dense embeddings are projected via `TruncatedSVD` followed by `StandardScaler` to matching dimensions ($2\text{D}, 4\text{D}, 6\text{D}, 8\text{D}, 10\text{D}, 12\text{D}$).
3. **Statistical Margin & Practical Significance**: Performance differences ($\Delta\text{F1}$) are evaluated against the pre-specified equivalence region $\epsilon = 0.01$.

---

## 2. Representation Registry & Exploration Status

| Representation | Family | Model / Spec | Original Dim | Projected Dims Evaluated | Status | Canonical Paper Inclusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + SVD** | Bag-of-Words / N-Gram | 50,000 max features, sublinear TF | 50,000 | 2D, 4D, 6D, 8D, 10D, 12D | `CANONICAL` | **Yes** (Primary baseline across all datasets) |
| **RoBERTa** | Transformer / Deep Contextual | `roberta-base` (frozen, mean-pool) | 768 | 8D | `CANONICAL` | **Yes** (CEAS 2008 representation ablation) |
| **MiniLM** | Sentence Transformer | `all-MiniLM-L6-v2` (frozen, mean-pool) | 384 | 8D | `EXPLORATORY` | **No** (Exploratory representation lab candidate) |
| **MPNet** | Sentence Transformer | `all-mpnet-base-v2` (frozen, mean-pool) | 768 | 8D | `EXPLORATORY` | **No** (Exploratory representation lab candidate) |
| **FastText** | Static Subword Embedding | `fasttext.wiki.en.300d` (subword bag) | 300 | 8D | `EXPLORATORY` | **No** (Optional subword exploratory baseline) |

---

## 3. Exploratory Research Question

> **Research Question (Exploratory)**: *“Does quantum-kernel behavior depend more strongly on text representation than on kernel choice?”*

The canonical research establishes that representation choice materially alters quantum-vs-RBF behavior on CEAS 2008. Experiment 41 tests whether this observation generalizes across multiple benchmark domains (SMS, CEAS, MeAJOR) and representation families (TF-IDF, MiniLM, RoBERTa, MPNet).

---

## 4. Experiment 41: Screening Protocol & Design

| Parameter | Specification | Scientific Rationale |
| :--- | :--- | :--- |
| **Datasets** | SMS Spam Collection, CEAS 2008, MeAJOR | Multi-domain text security benchmarks |
| **Representations** | TF-IDF (50k n-gram), MiniLM (`all-MiniLM-L6-v2`), RoBERTa (`roberta-base`), MPNet (`all-mpnet-base-v2`) | Sparse lexical vs compact dense vs base contextual vs contrastive sentence representations |
| **Dimensionality** | 8D (all models matched via `TruncatedSVD` + `StandardScaler`) | Controlled fair comparison; prevents quantum statevector exponential blowup |
| **Classifiers** | Linear SVM (`LinearSVC`), Classical RBF SVM (`SVC(kernel='rbf')`), Quantum Fidelity Kernel SVM (`ZZFeatureMap` 2-layer cyclic) | Matched parameterization ($C=1$, balanced class weights) |
| **Seeds** | `[42, 123, 456]` (3-seed initial screening) | Identifies candidate configurations prior to 10-seed confirmation (Exp 42) |
| **Thresholding** | 200-step validation-only grid | Strict test split isolation; zero test leakage |
| **Diagnostics** | Kernel diversity, label alignment, Q/RBF kernel correlation, representation dispersion | Geometric properties analyzed without causal attribution |

---

## 5. Scientific Wording & Epistemic Standards

1. **Observed Difference vs Quantum Advantage**: Raw positive $\Delta\text{F1}$ is termed an **“Observed quantum edge”** or **“Observed Q-RBF difference”** until 10-seed statistical confirmation with FDR control is achieved.
2. **Association vs Causation**: Representation variation is described as **“representation-conditioned variation in Q-RBF performance”** rather than claiming representation causes quantum advantage.
3. **Equivalence Region**: Differences within $\epsilon = \pm 0.01$ are classified as **practically equivalent**.
4. **Timing Scope**: Benchmark simulation runtime is explicitly separated from live single-request inference latency.

