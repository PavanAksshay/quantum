# Methodology Section Architecture and Protocol Specifications

**Document Version**: 1.0 (Frozen)  
**Date**: September 5, 2026  
**Scope**: Complete methodological specifications for Section 3 (Methodology), strictly harmonized with `FINAL_PROTOCOL_V1.md`.

---

## 3.1 Research Questions
- **Primary RQ**: *"Under controlled matched conditions, do quantum kernel methods provide a consistent and practically meaningful advantage over matched classical RBF kernels for text-based scam and phishing detection?"*
- **Secondary RQs**:
  - **RQ1 (Model Comparison)**: Quantum vs Matched Classical RBF in-distribution performance.
  - **RQ2 (Dimensionality Scaling)**: Impact of representation dimension $d \in [2, 16]$.
  - **RQ3 (Representation Interaction)**: TF-IDF vs dense RoBERTa embeddings.
  - **RQ4 (Kernel Geometry)**: Gram matrix correlation, target label alignment, state dispersion.
  - **RQ5 (Cross-Source Generalization)**: Domain transfer degradation under MeAJOR source holdout.
  - **RQ6 (Computational Cost)**: Wall-clock simulation runtime and memory scaling.

---

## 3.2 Benchmark Datasets
- **SMS Spam Collection**: 5,572 short mobile SMS messages (4,825 ham, 747 spam; 13.4% positive).
- **CEAS 2008 Email Corpus**: 39,154 raw emails cleaned to a canonical 15,000 subset (10k train / 2.5k val / 2.5k test).
- **MeAJOR Archive**: 108,684 usable emails across TREC 2005, TREC 2006, and TREC 2007 collections. Text is standardized strictly as `Subject + " " + Body`, with all non-NLP metadata fields excluded.

---

## 3.3 Leakage-Safe Splitting Protocols
1. **In-Distribution (IID) Splits**:
   - Fixed sample IDs frozen in `results/roberta_multidataset/meajor/{train,validation,test}_sample_ids.csv`.
   - Training: $N_{\text{train}} = 10,000$; Validation: $N_{\text{val}} = 2,500$; Test: $N_{\text{test}} = 2,500$.
   - Sample IDs are strictly invariant across independent random seeds.
2. **Cross-Source Domain-Holdout Splits**:
   - **Direction A**: Train on TREC 2005 + TREC 2006 ($10,000$ train, $2,500$ val); Test on TREC 2007 ($5,000$).
   - **Direction B (Canonical)**: Train on TREC 2007 ($10,000$ train, $2,500$ val); Test on balanced mixture of TREC 2005 ($2,500$) and TREC 2006 ($2,500$) ($N_{\text{test}} = 5,000$).

---

## 3.4 Text Preprocessing Pipeline
- Concatenate email subject and body text: `text = Subject.strip() + " " + Body.strip()`.
- Filter empty documents, invalid encoding artifacts, and non-target rows.
- Auxiliary metadata (headers, routing paths, dates, sender domains, URL counts) are strictly stripped to enforce pure NLP evaluation.

---

## 3.5 TF-IDF Feature Representation
- `lowercase`: `True`
- `strip_accents`: `"unicode"`
- `ngram_range`: `(1, 2)` (unigrams and bigrams)
- `min_df`: `2`
- `sublinear_tf`: `True` ($1 + \log(\text{tf})$)
- `max_features`: `50,000`
- `norm`: `"l2"`
- **Fitting Scope**: Fitted strictly on training text (`fit_transform` on train; `transform` on validation and test). Zero test vocabulary leakage.

---

## 3.6 Dimensionality Reduction (PCA / SVD)
- **Algorithm**: `sklearn.decomposition.TruncatedSVD` followed by `sklearn.preprocessing.StandardScaler` (zero mean, unit variance).
- **Primary Canonical Anchor**: $d = 8$ components (8 qubits).
- **Dimensionality Sweep**: $d \in \{2, 4, 6, 8, 10, 12, 16\}$.
- **Fitting Scope**: SVD and StandardScaler are fitted strictly on training TF-IDF matrices.

---

## 3.7 Quantum Feature Map
- **Circuit Architecture**: 2-layer cyclic $ZZFeatureMap$ ($\text{depth} = 2$, parameter-free).
  $$\mathcal{U}_{\Phi(x)} = \left( U_{\Phi(x)} H^{\otimes n} \right)^2$$
- **Single-Qubit Rotations**: $R_z(2x_i)$ applied to all qubits $i \in \{0, \dots, n-1\}$.
- **Two-Qubit Entangling Gates**: $R_{zz}(2x_i x_j)$ applied cyclically to pairs $(i, (i+1) \bmod n)$.
- **State Preparation**: Exact double-precision (`complex128`) statevector simulation.

---

## 3.8 Quantum Fidelity Kernel
- **Kernel Function**: Pure quantum state overlap fidelity:
  $$k(x, z) = |\langle \psi(x) | \psi(z) \rangle|^2$$
- **Properties**: Strictly unit diagonal ($k(x, x) = 1.0$), symmetric, positive semi-definite (PSD).
- **Classifier**: `sklearn.svm.SVC(kernel="precomputed", C=1.0, class_weight="balanced")`.

---

## 3.9 Classical RBF Baseline
- `sklearn.svm.SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")`
- Evaluated on identical $d$-dimensional scaled PCA representations as the quantum kernel.

---

## 3.10 Contextual Linear Baseline
- `sklearn.svm.LinearSVC(C=1.0, class_weight="balanced", max_iter=2000)`
- Evaluated on $d$-dimensional PCA for low-dimensional linear reference, and 50k TF-IDF for high-dimensional ceiling.

---

## 3.11 Out-of-Sample Threshold Optimization
- Continuous decision function scores $s_i \in \mathbb{R}$ are obtained for validation and test samples.
- The optimal decision threshold $\tau^*$ is selected strictly on the validation set via exhaustive search across 200 equidistant points in $[\min(s_{\text{val}}), \max(s_{\text{val}})]$ to maximize validation F1 score.
- Test predictions are computed strictly out-of-sample: $\hat{y}_{\text{test}} = \mathbb{I}(s_{\text{test}} \ge \tau^*)$. Test labels are never used for tuning.

---

## 3.12 Evaluation Metrics
- **Primary Metric**: F1 Score on the positive class ($\text{spam}/\text{phishing} = 1$).
- **Secondary Metrics**: Precision-Recall AUC (PR-AUC / Average Precision), ROC-AUC.
- **Diagnostics**: Accuracy, Balanced Accuracy, Precision, Recall, Decision Margin.

---

## 3.13 Inferential Statistical Testing
- **Paired Comparisons**: Seed-level paired differences $\Delta \text{F1} = \text{F1}_{\text{Quantum}} - \text{F1}_{\text{RBF}}$.
- **Confidence Intervals**: 95% Percentile Bootstrap CIs computed over $B = 10,000$ resamples.
- **Significance Testing**: Two-sided paired permutation tests ($10,000$ iterations; $H_0: \Delta = 0$).
- **Multiple Testing Correction**: Benjamini-Hochberg False Discovery Rate (FDR) control applied across all concurrent tests ($\alpha = 0.05$).

---

## 3.14 Predefined Practical Equivalence Region
- **Threshold**: $\varepsilon = 0.01$ F1 ($\pm 1.0$ percentage points).
- **Classification Rules**:
  - $\Delta \text{F1} > +0.01$: Meaningful Quantum Advantage.
  - $|\Delta \text{F1}| \le 0.01$: Practical Equivalence / Parity.
  - $\Delta \text{F1} < -0.01$: Meaningful Classical Advantage.

---

## 3.15 Computational Measurement Protocol
- Recorded wall-clock execution times:
  1. Kernel matrix construction time ($K_{\text{train}}$, $K_{\text{val}}$, $K_{\text{test}}$).
  2. Classifier fitting time ($t_{\text{train}}$).
  3. Inference score computation time ($t_{\text{inf}}$).
  4. Total pipeline execution time ($t_{\text{total}}$).
  5. Peak memory consumption via OS process metrics.

---

## 3.16 Reproducibility and Environment Protocol
- Full software/hardware stack captured in `RUN_METADATA.md` (Python 3.12, PyTorch 2.12, scikit-learn 1.9, NumPy 2.5, SciPy 1.18, Darwin arm64).
- Deterministic 10-seed suite: $\mathcal{S} = \{42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021\}$.
