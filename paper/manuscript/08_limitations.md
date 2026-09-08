# Section 6: Threats to Validity and Limitations

To ensure transparent reporting and prevent overgeneralization of our findings, this section details the primary threats to validity, methodological boundaries, and explicit limitations of our empirical benchmark. We categorize these constraints across dataset scope, simulation environment, feature-map architectures, dimensionality constraints, representation choices, statistical replication, source-holdout partitions, threat models, and runtime interpretation.

---

## 6.1 Dataset Scope

Our empirical evaluation is conducted across three standardized text corpora: the SMS Spam Collection, the CEAS 2008 Email Corpus, and the multi-source MeAJOR email archive (comprising TREC 2005, TREC 2006, and TREC 2007). While these benchmarks span a diverse range of document lengths (from telegraphic 62-character SMS messages to multi-thousand-character corporate emails), class balances ($13.41\%$ to $55.78\%$ positive prevalence), and communication channels (mobile messaging versus organizational email), they remain focused specifically on text-based scam, spam, and phishing detection.

Consequently, our findings cannot be extrapolated to:
- General-domain natural language processing tasks (e.g., machine translation, syntactic parsing, semantic role labeling, or abstractive summarization).
- Non-textual cybersecurity telemetry (e.g., network packet PCAP streams, executable binary opcodes, memory memory dumps, or provenance graphs).
- Fine-grained multi-class NLP benchmarks or complex multi-turn dialog understanding.

The conclusions of this study are strictly scoped to binary classification of deceptive natural language communications under the evaluated feature spaces.

---

## 6.2 Quantum Simulation Rather Than Hardware Execution

All quantum kernel matrices in this benchmark were evaluated through exact numerical statevector simulation using double-precision floating-point arithmetic (`complex128` state representations in PyTorch). 

This introduces two distinct methodological implications:
1. **Absence of Physical Hardware Noise**: The statevectors simulated in this study are entirely noiseless. They do not incorporate physical hardware noise mechanisms characteristic of Noisy Intermediate-Scale Quantum (NISQ) devices, such as qubit decoherence ($T_1, T_2$), single- and two-qubit gate infidelities, readout/measurement errors, or crosstalk.
2. **Infinite Measurement Shots**: State overlaps were evaluated analytically as exact inner products ($|\langle \psi(\mathbf{x}) | \psi(\mathbf{z}) \rangle|^2$), effectively corresponding to the infinite-shot limit without quantum projection noise.

Because our results show that parameter-free quantum kernels achieve parity rather than a practical advantage under ideal, noise-free simulation, introducing physical device noise or finite shot sampling would not plausibly improve quantum performance relative to classical baselines. However, we explicitly do not evaluate quantum error mitigation techniques (e.g., zero-noise extrapolation or Clifford data regression), nor do we assess physical execution times on superconducting or trapped-ion quantum processing units (QPUs).

---

## 6.3 Feature-Map Scope

The primary confirmation benchmark evaluates the canonical two-layer cyclic $ZZFeatureMap$ without variational parameters, with non-linear phase interactions $\phi_{\{i, j\}}(\mathbf{x}) = 2 (\pi - x_i)(\pi - x_j)$. 

While this architecture is the standard parameter-free reference kernel in applied QML literature, our study does not evaluate:
- **Parameterized Quantum Kernel Training (QKT)**: Variational quantum circuits where circuit parameters are trained via gradient descent to maximize kernel-target alignment or classification margin.
- **Alternative Non-Abelian Ansatzes**: Quantum circuits employing non-commuting multi-qubit Pauli rotations ($XY$, $YZ$), higher-order $k$-body interactions ($k \ge 3$), or random quantum circuit architectures.
- **Data Re-Uploading Architectures**: Multi-layered quantum feature maps that repeatedly interleave classical data encoding with trainable unitary transformations.

Although our preliminary architectural explorations confirmed that alternating entanglement topologies (full vs linear vs cyclic) yielded negligible performance shifts under matched conditions, we do not claim that our findings rule out the possibility of specialized, problem-tailored parameterized quantum feature maps achieving superior classification performance.

---

## 6.4 Dimensionality and Memory Constraints

Our empirical dimensionality sweeps were conducted across $d \in \{2, 4, 6, 8, 10, 12\}$ dimensions (and up to 64 dimensions for classical linear models). Evaluating quantum fidelity kernels at 16 dimensions or higher on our canonical 10,000-sample training partition proved intractable within our standardized benchmarking environment due to the resident memory ceiling (>10.5 GB allocation requirement for the $N \times 2^{N_q}$ statevector matrix).

This constraint is an empirical limitation of classical statevector simulation on local compute workstations, rather than a fundamental property of quantum algorithms. However, because in-distribution performance exhibited clear diminishing returns beyond 8 dimensions and reached statistical parity at 12 dimensions ($\Delta \text{F1} = +0.0014$, $p = 0.2824$), scaling to higher dimensions under the current feature-map formulation would not be expected to alter the observed convergence.

---

## 6.5 Representation Scope

The primary benchmark pipeline utilizes TruncatedSVD projections of high-dimensional (50,000-feature) TF-IDF n-gram matrices, supplemented by dense RoBERTa sentence embeddings in our representation ablation experiments. 

While this matched design successfully isolates the comparative behavior of quantum and classical kernels on identical coordinate inputs, it does not evaluate:
- Non-linear dimensionality reduction methods (e.g., deep autoencoders, variational autoencoders, or UMAP projections).
- Specialized subword tokenization schemes or graph-based lexical embeddings.
- End-to-end fine-tuning where representation extraction and quantum feature mapping are jointly optimized.

Our findings demonstrate that representation choice strongly influences the relative ranking between quantum and classical kernels; therefore, our empirical results must be interpreted specifically with respect to the evaluated TF-IDF and RoBERTa representations.

---

## 6.6 Statistical Scope and Replication Units

To address sampling variability, all primary empirical comparisons in this study were replicated across 10 independent random seeds ($\mathcal{S} = \{42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021\}$), with statistical significance evaluated via non-parametric paired permutation tests (10,000 permutations) and 95% bootstrap confidence intervals (10,000 resamples).

While 10 independent seeds provide substantially greater inferential power than the single-seed evaluations common in prior QML literature, it is important to recognize that:
- The 10 seeds represent computational replicates (varying dataset partitioning, TruncatedSVD initialization, and support vector solver order) across a fixed underlying corpus, rather than draws from 10 entirely independent data-generating distributions.
- Non-parametric bootstrap confidence intervals estimate the uncertainty of the estimator across the empirical sample, but do not generate independent scientific data points.

We avoid treating bootstrap resamples as independent experimental repetitions, and we restrict our statistical inferences strictly to the evaluated sample populations.

---

## 6.7 Source-Holdout Scope

Our cross-source domain generalization evaluation utilizes the structured multi-source partitioning of the MeAJOR corpus, specifically evaluating transfer across the TREC 2005, TREC 2006, and TREC 2007 email collections.

While these historical collections exhibit profound vocabulary drift ($>92\%$ type out-of-vocabulary rate and $>34\%$ token out-of-vocabulary rate), they represent a specific historical email environment. Our evaluation does not assess:
- Multi-modal phishing campaigns incorporating image, PDF, or HTML rendering obfuscations.
- Temporal continuous drift over multi-year modern telemetry.
- Cross-language domain transfer (e.g., English to multilingual security filtering).

The cross-source findings demonstrate that the evaluated quantum kernel failed to provide transfer robustness under the TREC source shift, but should not be generalized as a universal claim about all possible domain adaptation regimes.

---

## 6.8 Security Threat Model and Adversarial Robustness Boundaries

Our evaluation focuses exclusively on classification accuracy, calibration, and domain generalization under natural distribution shift. We do not evaluate model resilience against **adaptive adversarial attacks**.

Specifically, this study does not test:
- **Evasion Attacks**: Token substitution, homoglyph replacement, zero-width character injection, or character-level perturbations designed to bypass classification thresholds (e.g., TextBugger, TextFooler).
- **Poisoning Attacks**: Malicious manipulation of training data to create backdoors or degrade model decision boundaries.
- **Adaptive Attackers**: Attackers with black-box or white-box access to the model architecture and gradient information.

Therefore, our study does not claim to establish the "adversarial robustness" or "cybersecurity hardness" of quantum kernel classifiers against active adversaries. Evaluating whether quantum Hilbert spaces confer resistance against gradient-based adversarial text perturbations remains an open question outside the scope of this work.

---

## 6.9 Simulation Runtime Interpretation

The execution times reported in this benchmark (e.g., $108.8$ seconds for 12D quantum kernel computation vs $1.7$ seconds for classical RBF) were measured under a standardized local computing environment (Apple M-series unified memory architecture executing multi-threaded PyTorch double-precision CPU kernels).

These execution runtimes provide a rigorous empirical comparison of the computational cost incurred by researchers running classical simulations of quantum models under identical hardware conditions. However:
- They do not represent theoretical asymptotic complexity bounds.
- They do not reflect the physical execution time of quantum circuits on superconducting or optical quantum hardware.
- They are dependent on specific implementation optimizations, BLAS libraries, and vectorization strategies.

Accordingly, we report simulation runtimes as practical engineering benchmarks for current research workflows, rather than fundamental computational limits of quantum hardware.

<!-- CLAIM AUDIT: All limitations fully declared. Hardware noise, QKT, adversarial robustness, and simulation boundaries explicitly stated. -->
