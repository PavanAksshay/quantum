# Section 3: Methodology and Experimental Protocol

This section details the canonical experimental design, dataset hygiene protocols, feature engineering, mathematical models, threshold optimization procedures, and inferential statistical framework utilized throughout this study. All methodologies are harmonized with the frozen benchmark protocol (`FINAL_PROTOCOL_V1.md`) and adhere to strict zero-leakage standards.

---

## 3.1 Research Questions

The empirical investigation is structured around one primary research question and five supporting research questions:

- **Central Research Question**:  
  *Under controlled matched conditions, do quantum kernel methods provide a consistent and practically meaningful advantage over matched classical RBF kernels for text-based scam and phishing detection?*

- **Supporting Research Questions**:
  - **RQ1 (In-Distribution Model Performance)**: Does the parameter-free quantum fidelity kernel outperform a matched classical Gaussian RBF kernel under in-distribution (IID) evaluation when both models operate on identical low-dimensional representations?
  - **RQ2 (Representation and Dimensionality Scaling)**: How do representation dimensionality ($d \in [2, 16]$) and representation type (sparse n-gram TF-IDF vs dense contextual RoBERTa embeddings) affect the relative performance and scaling trajectory of quantum and classical kernels?
  - **RQ3 (Cross-Source Generalization)**: Does the evaluated quantum fidelity kernel provide an empirical robustness advantage over matched classical baselines when subjected to out-of-distribution source domain shift?
  - **RQ4 (Kernel Geometry and Alignment)**: How do quantum Gram matrix structure, classical RBF geometry, target label alignment, statevector dispersion, and pairwise kernel diversity relate to observed classification behavior?
  - **RQ5 (Computational Cost and Scaling)**: What computational execution time and memory trade-offs arise as representation dimensionality increases for classical statevector-simulated quantum kernels?

---

## 3.2 Dataset Selection and Audit

To evaluate classification performance across diverse text lengths, vocabulary distributions, and class imbalances, we benchmark three curated corpora comprising over 152,000 text samples:

1. **UCI SMS Spam Collection**:  
   The official corpus consists of 5,574 mobile SMS messages [Almeida et al., 2011; DOI: 10.24432/C5CC84]. Following automated text cleaning and invalid encoding removal, the final usable classification dataset comprises **5,572** instances (4,825 legitimate ham and 747 malicious spam; positive class prevalence: $13.41\%$). Texts are short and telegraphic, with a median document length of 12 words.

2. **CEAS 2008 Phishing Corpus**:  
   Curated from the 2008 Collaboration Against Spam and Phishing Conference, the dataset contains 39,154 raw email messages. After encoding normalization and HTML artifact extraction, the corpus provides a canonical experimental subset of **15,000** instances (12,165 legitimate and 2,835 phishing emails; positive class prevalence: $18.90\%$), with a median document length of 48 words.

3. **MeAJOR Email Archive**:  
   The Multi-source Email Archive for Journey and Open Research (MeAJOR) contains 108,685 raw email records spanning three major historical benchmark collections: TREC 2005 (49,583 emails), TREC 2006 (15,005 emails), and TREC 2007 (44,096 emails). Following the removal of one missing-label record, the usable dataset comprises **108,684** emails (87,678 legitimate and 21,006 phishing/spam; positive class prevalence: $19.33\%$). MeAJOR represents a heterogeneous, long-form text corpus with a median document length of 84 words.

These three datasets provide complementary experimental environments, varying systematically across document length (short SMS vs long-form emails), corpus scale ($5.5\text{k}$ to $108\text{k}$ samples), class balance ($13.4\%\text{--}19.3\%$), and source composition (single-source mobile traffic vs multi-source organizational email archives).

---

## 3.3 Input Representation and Metadata Hygiene

In real-world security deployments, non-NLP metadata fields (such as sender IP addresses, routing headers, domain registrations, and transmission timestamps) often exhibit severe collection-specific bias. If included in experimental evaluation, classifiers frequently learn trivial metadata associations rather than semantic language features.

To enforce pure natural language evaluation, all auxiliary metadata fields are strictly stripped from the primary input representation:
- Excluded fields: `sender`, `receiver`, `sender_domain`, `date`, `urls`, `url_count`, `language`, and `source_id`.
- The primary textual input is constructed strictly from natural language content:
  - For SMS: the raw message body.
  - For CEAS and MeAJOR: the standardized concatenation of subject and body preview:
    \begin{equation}
    \text{text}_i = \text{Subject}_i.\text{strip}() + \text{" "} + \text{Body}_i.\text{strip}()
    \end{equation}

---

## 3.4 Leakage-Safe Data Partitioning and Duplicate Audit

Data leakage from overlapping duplicate messages or improper train-test partitioning represents a major source of inflated accuracy in security NLP literature. We apply strict, hash-verified splitting protocols:

### Split Partitions
- **SMS Spam Collection**: 3,343 training samples ($60.0\%$), 1,114 validation samples ($20.0\%$), and 1,115 test samples ($20.0\%$).
- **CEAS 2008 Corpus**: 23,494 training samples, 7,830 validation samples, and 7,830 test samples across the full corpus. The canonical experimental subset utilizes 10,000 training, 2,500 validation, and 2,500 test samples.
- **MeAJOR Corpus (IID Benchmark)**: 65,211 training samples, 21,737 validation samples, and 21,736 test samples across the full archive. The canonical IID benchmark subset utilizes 10,000 training, 2,500 validation, and 2,500 test samples.

### Duplicate Audit and Group Awareness
Exact-string and near-duplicate text messages occur naturally in email and SMS corpora due to automated spam distribution. A comprehensive duplicate audit was conducted across all three corpora:
- **SMS Spam**: 705 duplicate samples distributed across 5,157 unique text groups.
- **CEAS 2008**: 274 duplicate samples distributed across 38,963 unique text groups.
- **MeAJOR Archive**: 5,200 duplicate samples distributed across 104,529 unique text groups.

Partitions are generated using stratified random assignment where all instances belonging to the same text group are constrained to the same split partition, preventing duplicate-group leakage across training, validation, and test sets. Furthermore:
1. All vocabulary extractors, scaling transforms, and dimensionality reduction matrices are fitted **strictly on training partitions**.
2. Threshold optimization is performed **strictly on validation partitions**.
3. Test partitions are inspected **strictly once** for out-of-sample performance estimation.

---

## 3.5 Cross-Source Domain-Holdout Protocol

To evaluate out-of-distribution generalization under source shift, we establish a separate source-holdout evaluation protocol on the MeAJOR archive, decoupling cross-domain transfer from the standard IID benchmark:

- **Direction A (Multi-Source Transfer to Single-Source)**:
  - *Training Pool*: TREC 2005 ($49,583$) + TREC 2006 ($15,005$).
  - *Evaluation Target*: TREC 2007 ($44,096$).
  - *Partition Sizes*: Stratified sampling of $10,000$ training and $2,500$ validation samples from the training pool; out-of-sample evaluation on $5,000$ samples from TREC 2007.

- **Direction B (Single-Source Transfer to Balanced Multi-Source Target — Canonical Protocol)**:
  - *Training Pool*: TREC 2007 ($44,096$).
  - *Evaluation Target*: TREC 2005 ($49,583$) + TREC 2006 ($15,005$).
  - *Partition Sizes*: Stratified sampling of $10,000$ training and $2,500$ validation samples from TREC 2007; out-of-sample evaluation on a balanced target test mixture of $2,500$ TREC 2005 emails and $2,500$ TREC 2006 emails ($N_{\text{test}} = 5,000$).

---

## 3.6 Text Vectorization: Canonical TF-IDF Configuration

All text features are extracted using canonical n-gram term frequency-inverse document frequency (TF-IDF) vectorization:
- `lowercase`: `True`
- `strip_accents`: `"unicode"`
- `ngram_range`: `(1, 2)` (extracting both individual words and contiguous bigrams)
- `min_df`: `2` (pruning terms that appear in fewer than two training documents)
- `sublinear_tf`: `True` (applying logarithmic term-frequency scaling, $1 + \log(\text{tf})$)
- `max_features`: `50,000` (retaining the 50,000 most frequent n-grams in the training corpus)
- `norm`: `"l2"` (normalizing feature vectors to unit Euclidean norm)

The vectorizer is fitted strictly on the training partition: $\mathbf{X}_{\text{train}} \in \mathbb{R}^{N_{\text{train}} \times 50000}$. Validation and test sets are transformed using the fixed training vocabulary and IDF weights.

---

## 3.7 Dimensionality Reduction via TruncatedSVD

Because quantum circuit simulation and near-term quantum hardware are constrained to modest qubit counts ($n \le 16$), high-dimensional classical feature spaces must be projected into low-dimensional representations.

We utilize **TruncatedSVD-based dimensionality reduction** followed by standard feature standardization:
1. `sklearn.decomposition.TruncatedSVD`: Projects the sparse $50,000$-dimensional TF-IDF matrix into a $d$-dimensional subspace:
   \begin{equation}
   \mathbf{Z}_{\text{train}} = \mathbf{X}_{\text{train}} \mathbf{V}_d \in \mathbb{R}^{N_{\text{train}} \times d}
   \end{equation}
   where $\mathbf{V}_d$ contains the top $d$ right singular vectors fitted strictly on training data.
2. `sklearn.preprocessing.StandardScaler`: Standardizes each reduced component to zero empirical mean and unit variance:
   \begin{equation}
   \tilde{\mathbf{z}}_{i, j} = \frac{\mathbf{z}_{i, j} - \mu_j}{\sigma_j}, \quad j \in \{1, \dots, d\}
   \end{equation}

- **Primary Canonical Anchor**: $d = 8$ dimensions (mapped to an 8-qubit feature map).
- **Dimensionality Scaling Sweep**: $d \in \{2, 4, 6, 8, 10, 12, 16\}$.

*Methodological Note on Terminology*: We explicitly designate this procedure as TruncatedSVD-based dimensionality reduction rather than standard PCA, as it operates directly on sparse term-document matrices without centering the high-dimensional matrix prior to decomposition. Crucially, representation dimensions and quantum qubits are not intrinsically equivalent physical entities: rather, an 8-dimensional reduced classical representation is mapped into an 8-qubit quantum feature map.

---

## 3.8 Quantum Feature Map and Fidelity Kernel Formulation

The quantum classifier maps each standardized classical feature vector $\tilde{\mathbf{z}} \in \mathbb{R}^d$ into an $n$-qubit state ($n = d$) using the canonical parameter-free, two-layer cyclic $ZZFeatureMap$ [Havlíček et al., 2019]:
\begin{equation}
\mathcal{U}_{\Phi(\tilde{\mathbf{z}})} = \left( U_{\Phi(\tilde{\mathbf{z}})} H^{\otimes n} \right)^2
\end{equation}
where $H^{\otimes n}$ denotes a layer of Hadamard gates applied to all $n$ qubits, and $U_{\Phi(\tilde{\mathbf{z}})}$ applies diagonal single-qubit and two-qubit entangling phase rotations:
\begin{equation}
U_{\Phi(\tilde{\mathbf{z}})} = \exp\left( -i \sum_{j=0}^{n-1} \tilde{z}_j Z_j - i \sum_{j=0}^{n-1} \tilde{z}_j \tilde{z}_{(j+1) \bmod n} Z_j Z_{(j+1) \bmod n} \right)
\end{equation}
where $Z_j$ denotes the Pauli-$Z$ operator acting on qubit $j$. The two-qubit entangling interactions follow a closed cyclic nearest-neighbor topology.

The resulting quantum state is $|\psi(\tilde{\mathbf{z}})\rangle = \mathcal{U}_{\Phi(\tilde{\mathbf{z}})} |0\rangle^{\otimes n}$. The quantum fidelity kernel between samples $\mathbf{x}$ and $\mathbf{z}$ is defined as the transition probability:
\begin{equation}
k_Q(\mathbf{x}, \mathbf{z}) = |\langle \psi(\tilde{\mathbf{x}}) | \psi(\tilde{\mathbf{z}}) \rangle|^2
\end{equation}
Kernel matrices are computed via exact double-precision (`complex128`) statevector simulation. The resulting Gram matrix $\mathbf{K}_Q \in \mathbb{R}^{N \times N}$ is symmetric, positive semi-definite (PSD) within machine precision, and strictly unit diagonal ($k_Q(\mathbf{x}, \mathbf{x}) = 1.0$).

The precomputed Gram matrix is supplied to a Support Vector Classifier (`sklearn.svm.SVC`):
- `kernel`: `"precomputed"`
- `C`: $1.0$
- `class_weight`: `"balanced"` (adjusting misclassification penalties inversely proportional to class frequencies).

*Simulation Scope*: We explicitly emphasize that this evaluation represents a software-simulated quantum statevector calculation executed on classical hardware, rather than an execution on physical quantum processors.

---

## 3.9 Matched Classical Baseline: Radial Basis Function (RBF) Kernel

To ensure an exact, rigorous comparison, the classical non-linear baseline is implemented using the Gaussian RBF kernel Support Vector Classifier:
- `model`: `sklearn.svm.SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")`
- `kernel function`:
  \begin{equation}
  k_{\text{RBF}}(\tilde{\mathbf{x}}, \tilde{\mathbf{z}}) = \exp\left(-\gamma \|\tilde{\mathbf{x}} - \tilde{\mathbf{z}}\|_2^2\right), \quad \gamma = \frac{1}{d \cdot \text{Var}(\tilde{\mathbf{Z}})}
  \end{equation}

**Critical Experimental Control**: The classical RBF classifier operates on **precisely the same standardized low-dimensional vector $\tilde{\mathbf{z}} \in \mathbb{R}^d$** supplied to the quantum feature map. This guarantees that any observed performance divergence reflects differences in the induced kernel geometries rather than discrepancies in input representation, dimensionality, or regularization parameters.

---

## 3.10 Contextual Baseline: Linear Support Vector Classifier

As a contextual linear reference, we evaluate a Linear Support Vector Classifier:
- `model`: `sklearn.svm.LinearSVC(C=1.0, class_weight="balanced", max_iter=2000)`
- Evaluated on:
  1. Reduced $d$-dimensional representations (to quantify the linear separability of the compressed subspace).
  2. Full $50,000$-dimensional TF-IDF representations (to establish the high-dimensional classical performance ceiling).

---

## 3.11 Out-of-Sample Threshold Optimization

Because security text datasets exhibit class imbalance ($13\%\text{--}19\%$ malicious samples), the default zero decision threshold on SVM continuous scores $s_i = \mathbf{w}^T \phi(\mathbf{x}_i) + b$ is frequently suboptimal for F1 score maximization.

We implement an out-of-sample threshold selection protocol:
1. For each fitted model, continuous decision function scores $s_{\text{val}} \in \mathbb{R}^{N_{\text{val}}}$ are extracted on the validation set.
2. A grid of 200 equidistant candidate thresholds $\mathcal{T} = \{\tau_1, \dots, \tau_{200}\}$ spanning $[\min(s_{\text{val}}), \max(s_{\text{val}})]$ is evaluated.
3. The optimal threshold $\tau^*$ is selected strictly to maximize validation F1 score:
   \begin{equation}
   \tau^* = \arg\max_{\tau \in \mathcal{T}} \text{F1}\left(y_{\text{val}}, \mathbb{I}(s_{\text{val}} \ge \tau)\right)
   \end{equation}
4. The selected threshold $\tau^*$ is applied without modification to test decision scores: $\hat{y}_{\text{test}} = \mathbb{I}(s_{\text{test}} \ge \tau^*)$. Test partition labels are strictly quarantined during threshold selection.

---

## 3.12 Evaluation Metrics and Diagnostics

Model performance is evaluated across a structured hierarchy of classification and diagnostic metrics:

### Primary Metric
- **F1 Score**: The harmonic mean of precision and recall on the positive (scam/phishing) class:
  \begin{equation}
  \text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  \end{equation}

### Secondary Performance Metrics
- **Precision-Recall AUC (PR-AUC / Average Precision)**: The area under the precision-recall curve across all classification thresholds, particularly informative under class imbalance.
- **Receiver Operating Characteristic AUC (ROC-AUC)**: Area under the true positive rate vs false positive rate trajectory.
- **Accuracy & Balanced Accuracy**: Overall instance accuracy and unweighted mean of class-specific recalls.
- **Precision & Recall**: Individual operating point error rates on malicious traffic.

### Geometric and Diagnostic Metrics
- **Decision Margin**: Empirical separation between mean scores of malicious and legitimate test instances: $\Delta s = \mathbb{E}[s | y=1] - \mathbb{E}[s | y=0]$.
- **Kernel-Target Label Alignment**: Frobenius inner product alignment between the kernel matrix $\mathbf{K}$ and the ideal label matrix $\mathbf{y}\mathbf{y}^T$ [Cortes et al., 2012]:
  \begin{equation}
  A(\mathbf{K}, \mathbf{y}) = \frac{\langle \mathbf{K}, \mathbf{y}\mathbf{y}^T \rangle_F}{\|\mathbf{K}\|_F \|\mathbf{y}\mathbf{y}^T\|_F}
  \end{equation}
- **Gram Matrix Correlation ($r_{\text{Q,RBF}}$)**: Pearson correlation between the off-diagonal entries of the quantum and classical RBF Gram matrices:
  \begin{equation}
  r_{\text{Q,RBF}} = \text{Corr}\left(\text{vec}(\mathbf{K}_Q)_{i \neq j}, \text{vec}(\mathbf{K}_{\text{RBF}})_{i \neq j}\right)
  \end{equation}
- **Single-State Dispersion (Entropy & Participation Ratio)**: The Shannon/von Neumann entropy of the statevector amplitudes:
  \begin{equation}
  S(\psi) = -\sum_{k=0}^{2^n-1} |\alpha_k|^2 \ln |\alpha_k|^2, \quad \text{PR}(\psi) = \frac{1}{\sum_{k=0}^{2^n-1} |\alpha_k|^4}
  \end{equation}
- **Pairwise Kernel Diversity**: Empirical variance of the off-diagonal entries of the Gram matrix: $\text{Var}(K_{i,j})_{i \neq j}$.

---

## 3.13 Statistical Protocol and Practical Equivalence Framework

To account for stochastic variability in SVD projections and data sampling, all primary confirmation experiments are executed across a frozen suite of **10 independent random seeds**:
\begin{equation}
\mathcal{S} = \{42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021\}
\end{equation}

For each evaluated configuration, we report:
- Sample Mean ($\bar{x}$) and Sample Standard Deviation ($s$).
- Median, Minimum, and Maximum.
- 95% Parametric Student-$t$ Confidence Interval.
- 95% Non-parametric Percentile Bootstrap Confidence Interval computed over $B = 10,000$ resamples of paired differences $\Delta_k = \text{F1}_{\text{Quantum}, k} - \text{F1}_{\text{RBF}, k}$.

### Inferential Significance Testing
- **Paired Permutation Tests**: Exact two-sided paired permutation tests ($10,000$ iterations) evaluating the null hypothesis of exchangeable model performance ($H_0: \Delta = 0$).
- **Paired McNemar Tests**: Evaluated on $2 \times 2$ classification contingency tables of instance-level prediction disagreements.
- **Multiple Testing Control**: Benjamini-Hochberg False Discovery Rate (FDR) adjustments applied across all concurrent hypothesis tests ($\alpha = 0.05$).

### Predefined Practical Equivalence Region ($\varepsilon = 0.01$)
In large-sample empirical benchmarks, minute performance differences (e.g., $+0.002$ F1) can achieve formal statistical significance ($p < 0.05$) while remaining practically negligible in real-world engineering deployments. We establish an *a priori* practical equivalence threshold of $\varepsilon = 0.01$ F1 ($\pm 1.0$ percentage points):
- **Meaningful Quantum Advantage**: $\Delta \text{F1} > +0.01$ with 95% CI strictly above $+0.01$.
- **Practical Equivalence / Parity**: $|\Delta \text{F1}| \le 0.01$ (observed differences lie within the $[ -0.01, +0.01 ]$ band).
- **Meaningful Classical Advantage**: $\Delta \text{F1} < -0.01$ with 95% CI strictly below $-0.01$.

*Methodological Rule on Language*: If an observed difference is statistically distinguishable from zero ($p < 0.05$) but its magnitude satisfies $|\Delta \text{F1}| \le 0.01$, we describe it as *"statistically detectable but within the predefined practical-equivalence region."* If $p \ge 0.05$, we describe it as showing *"no statistically detectable difference."*

---

## 3.14 Computational Measurement Protocol

Computational execution costs are measured empirically under a standardized hardware environment (Apple M1 / arm64 architecture, 8 CPU cores, 8.0 GB RAM, Python 3.12.4, PyTorch 2.12.1 CPU engine, scikit-learn 1.9.0):
- **Kernel Matrix Construction Time ($t_{\text{kernel}}$)**: Wall-clock time required to simulate statevectors and compute pairwise Gram matrices for train, validation, and test partitions.
- **Classifier Training Time ($t_{\text{train}}$)**: Dual quadratic programming optimization time for the SVM.
- **Inference Time ($t_{\text{inf}}$)**: Time required to evaluate decision scores on out-of-sample test instances.
- **Total Pipeline Execution Time ($t_{\text{total}}$)**: $t_{\text{kernel}} + t_{\text{train}} + t_{\text{inf}}$.
- **Peak Resident Memory**: Maximum resident set size (RSS) tracked via operating system process monitoring.

*Framing Guardrail*: We explicitly document that measured runtimes reflect classical software statevector simulation on CPU architectures, rather than physical quantum device execution times or fundamental asymptotic computational complexity.

---

## 3.15 Geometry Diagnostics: Dispersion vs Kernel Diversity

In theoretical QML literature, quantum state dispersion is frequently conflated with pairwise kernel diversity. In our diagnostic framework, we strictly distinguish:
1. **Single-State Dispersion (Entropy $S(\psi)$ and Participation Ratio)**: Measures how uniformly an individual sample's quantum statevector spreads across the $2^n$ basis states.
2. **Pairwise Kernel Diversity ($\text{Var}(K_{i,j})$)**: Measures the geometric spread and variability of pairwise inner products across distinct samples in the dataset.

Diagnostic regressions across our benchmark corpora confirm that single-state entropy and pairwise kernel diversity exhibit a strong inverse association ($r \approx -0.78$ to $-0.83$), demonstrating that as individual statevectors become maximally dispersed (high entropy), pairwise fidelities concentrate, reducing kernel diversity.

---

## 3.16 Experimental Protocol Summary

Table \ref{tab:protocol_summary} synthesizes the core operational parameters of the benchmark.

\begin{table}[h!]
\centering
\small
\caption{Summary of Canonical Benchmark Protocol Specifications}
\label{tab:protocol_summary}
\begin{tabular}{ll}
\toprule
\textbf{Protocol Dimension} & \textbf{Canonical Implementation Specification} \\
\midrule
Benchmark Datasets & SMS Spam (5.5k), CEAS 2008 (15k), MeAJOR (108k) \\
Evaluated NLP Content & Message Text (SMS); Subject + Body (CEAS/MeAJOR); Metadata Stripped \\
Text Representation & Training-fitted TF-IDF (1–2 n-grams, min-df=2, sublinear-tf, max-feat=50k, L2) \\
Dimensionality Reduction & TruncatedSVD + StandardScaler (Fitted strictly on training partition) \\
Primary Dimension Anchor & $d = 8$ components $\to$ 8-qubit feature map \\
Dimensionality Sweep & $d \in \{2, 4, 6, 8, 10, 12, 16\}$ \\
Quantum Feature Map & Parameter-free 2-layer cyclic $ZZFeatureMap$ \\
Quantum Kernel & State fidelity $k_Q(\mathbf{x}, \mathbf{z}) = |\langle \psi(\mathbf{x}) | \psi(\mathbf{z}) \rangle|^2$ (Exact complex128 simulation) \\
Matched Classical Baseline & Gaussian RBF Kernel SVC ($C=1.0$, $\gamma=\text{"scale"}$, balanced weights) \\
Contextual Reference & LinearSVC ($C=1.0$, balanced weights, max-iter=2000) \\
Evaluation Regimes & In-Distribution (IID) and Cross-Source Holdout (Direction A \& B) \\
Replication Seeds & $N=10$ independent seeds ($\mathcal{S} = \{42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021\}$) \\
Primary Metric & Positive-class F1 Score \\
Threshold Selection & Validation set F1 maximization over 200-step grid (Test set strictly quarantined) \\
Statistical Protocol & 95\% Bootstrap CIs ($B=10,000$), Paired Permutation Tests ($10,000$), BH FDR \\
Practical Equivalence Region & $\varepsilon = 0.01$ F1 ($\pm 1.0$ percentage points) \\
Reproducibility Guarantee & Hash-verified frozen data splits, open-source code and execution logs \\
\bottomrule
\end{tabular}
\end{table}

<!-- CLAIM AUDIT:
- Section 3.1: Central RQ + RQ1-RQ5 defined neutrally without pre-judging outcomes. Evidence: FINAL_PROTOCOL_V1.md.
- Section 3.2: SMS (5,572 usable, DOI: 10.24432/C5CC84), CEAS (15,000 canonical), MeAJOR (108,684 usable: TREC5=49,583; TREC6=15,005; TREC7=44,096). Evidence: FINAL_PROTOCOL_V1.md, table_1_dataset_characteristics.csv.
- Section 3.3: Text-only representation, metadata stripped. Evidence: canonical_protocol.json, FINAL_PROTOCOL_V1.md.
- Section 3.4: Frozen splits (SMS 3343/1114/1115; CEAS 23494/7830/7830; MeAJOR 65211/21737/21736); duplicate counts (SMS 705/5157; CEAS 274/38963; MeAJOR 5200/104529). Evidence: split_hashes.csv, EXP36_REPRODUCIBILITY_AUDIT.md.
- Section 3.5: Direction A (TREC5+6 -> TREC7) & Direction B (TREC7 -> 2.5k TREC5 + 2.5k TREC6 = 5k test). Evidence: canonical_protocol.json, Exp 36, Exp 40B.
- Section 3.6: TF-IDF (1-2 ngrams, min_df=2, sublinear_tf=True, max_features=50000, norm="l2", train-only fit). Evidence: FINAL_PROTOCOL_V1.md.
- Section 3.7: TruncatedSVD + StandardScaler, 8D primary anchor, 2-16D sweep, TruncatedSVD terminology enforced. Evidence: FINAL_PROTOCOL_V1.md.
- Section 3.8: 2-layer cyclic ZZFeatureMap, exact complex128 fidelity kernel, SVC precomputed, statevector simulation scope. Evidence: FINAL_PROTOCOL_V1.md, 40_confirmation_experiments.py.
- Section 3.9: Classical RBF matched control on identical representation. Evidence: FINAL_PROTOCOL_V1.md.
- Section 3.10: Linear SVM contextual reference. Evidence: FINAL_PROTOCOL_V1.md.
- Section 3.11: 200-step validation F1 grid threshold selection, zero test leakage. Evidence: FINAL_PROTOCOL_V1.md, select_best_threshold.
- Section 3.12: Metrics (F1, PR-AUC, ROC-AUC, accuracy, margin, alignment, Gram r, state entropy, participation ratio, runtime, RAM). Evidence: FINAL_PROTOCOL_V1.md, evaluate_predictions.
- Section 3.13: 10 seeds (42..2021), 10k bootstrap, 10k permutation, BH FDR, epsilon = 0.01 practical equivalence region. Evidence: exp40_config.json, exp40_statistical_summary.csv.
- Section 3.14: Computational profiling (12D Quantum 108.8s vs RBF 1.7s, 64x ratio scoped to simulation). Evidence: exp40_results.csv, Table 8.
- Section 3.15: Geometry (Gram r = 0.55-0.65; alignment 50-60% lower; entropy vs diversity r = -0.78 to -0.83). Evidence: Table 7, Exp 27, Exp 32.
- Section 3.16: Protocol summary table. Evidence: Table 1-8 synthesis.
-->
