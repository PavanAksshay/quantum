# Section 2: Related Work

The intersection of quantum computing, machine learning, and natural language processing has produced an expanding literature exploring quantum-enhanced representations and kernel methods. In this section, we review prior work across five foundational domains: (1) quantum machine learning in cybersecurity; (2) quantum kernel methods for text classification; (3) classical text classification baselines; (4) distribution shift and cross-domain generalization in security NLP; and (5) benchmarking standards in empirical QML. Finally, we formulate the precise research gap addressed by this study.

---

## 2.1 Quantum Machine Learning for Cybersecurity

The exploration of quantum machine learning (QML) for cybersecurity applications has expanded significantly over the past five years, motivated by the prospective computational and representational capacities of quantum states [Sagingalieva et al., 2022; Ahmed et al., 2022 [VERIFY CITATION]]. Within the security domain, researchers have investigated variational quantum classifiers (VQC), quantum support vector machines (QSVM), and quantum neural networks (QNN) across several high-impact threat categories, including network intrusion detection [Sagingalieva et al., 2022], malware analysis [Ahmed et al., 2022 [VERIFY CITATION]], phishing URL identification [Shahriyar et al., 2025], and email/SMS scam detection [Hridi et al., 2026; Guddanti et al., 2026].

A recent systematic review of QML applications in phishing detection emphasizes that quantum approaches have demonstrated technical feasibility across diverse attack vectors [Hridi et al., 2026]. For instance, Shahriyar et al. [2025] introduced *PhishVQC*, combining correlation-based feature selection with variational quantum circuits to classify malicious URLs. Similarly, Guddanti et al. [2026] evaluated quantum support vector classifiers (QSVC) and VQC models for detecting fraudulent phishing accounts in Ethereum transaction networks, deploying quantum random access code (QRAC) and Pauli $ZZ$ encodings on IBM Heron superconducting quantum processors. In the domain of electronic messaging, Hridi et al. [2026] evaluated batch-wise ensembles of classical and quantum SVMs using $ZZFeatureMap$ encodings on email corpora.

While these studies establish the practical implementation of quantum pipelines in security contexts, an audit of the literature reveals recurring methodological limitations. First, existing empirical evaluations almost universally rely on small, heavily subsampled datasets—frequently containing fewer than $500$ to $2,000$ instances—due to the severe computational bottlenecks of classical simulation and limited quantum processing unit (QPU) access [Hridi et al., 2026; Shahriyar et al., 2025]. Second, classical controls are frequently incomplete or unmatched; several studies compare tuned quantum models against un-tuned or lower-capacity linear baselines without controlling for dimensionality reduction artifacts. Third, data preprocessing and tokenization are rarely isolated in a leakage-safe manner, with feature selection and scaling often computed over entire datasets prior to train-test partitioning. Fourth, evaluations in cybersecurity QML have focused almost exclusively on in-distribution cross-validation, leaving performance under real-world domain shifts and source transfers untested. Finally, wall-clock execution times and memory allocations are rarely documented with sufficient precision to assess practical utility [Li et al., 2026].

---

## 2.2 Quantum Kernels for Text Classification

Quantum kernel estimation, formalized by Havlíček et al. [2019] and Schuld & Killoran [2019], frames quantum machine learning as the implicit evaluation of inner products within a high-dimensional quantum Hilbert space. For an input vector $\mathbf{x} \in \mathbb{R}^d$, a parameterized quantum circuit $\mathcal{U}_{\Phi(\mathbf{x})}$ prepares a quantum state $|\psi(\mathbf{x})\rangle = \mathcal{U}_{\Phi(\mathbf{x})}|0\rangle^{\otimes n}$. The quantum fidelity kernel computes the transition probability between two encoded states:
\begin{equation}
k_Q(\mathbf{x}, \mathbf{z}) = |\langle \psi(\mathbf{x}) | \psi(\mathbf{z}) \rangle|^2 = |\langle 0|^{\otimes n} \mathcal{U}_{\Phi(\mathbf{x})}^\dagger \mathcal{U}_{\Phi(\mathbf{z})} |0\rangle^{\otimes n}|^2
\end{equation}
Because the state space dimension scales exponentially ($2^n$) with the number of qubits $n$, quantum feature maps can generate complex, non-linear geometric structures. The primary scientific hypothesis motivating quantum kernels in NLP is that specific parameterized circuits—such as those incorporating non-linear multi-qubit entanglement—might capture higher-order semantic interactions between features that differ beneficially from classical radial basis function (RBF) kernels [Havlíček et al., 2019; Garg et al., 2024].

In natural language processing, quantum kernel methods have been explored through two primary avenues: categorical compositional models (DisCoCat) that map grammatical syntax to quantum tensor networks [Coecke et al., 2020; Lorenz et al., 2021 [VERIFY CITATION]], and hybrid vector-embedding pipelines that encode compressed classical text representations into parameterized circuits [Garg et al., 2024; Shukla et al., 2023; Di Sipio et al., 2021 [VERIFY CITATION]]. Recent literature in 2024–2026 has increasingly focused on the statistical and geometric properties of these embeddings. For example, Garg et al. [2024] combined pretrained transformer embeddings with QSVC models using PCA compression on standard sentiment benchmarks, reporting competitive accuracy with classical SVMs on small subsets.

However, theoretical research has identified profound mathematical constraints governing quantum kernel geometries. Thanasilp et al. [2024] demonstrated that generic, highly expressive quantum feature maps suffer from exponential concentration: as the number of qubits grows, the off-diagonal entries of the Gram matrix concentrate exponentially around their mean, leading to vanishing gradient variance and poor generalization unless the feature map is strictly tailored to the underlying data structure. Similarly, Huang et al. [2021] proved that classical learning algorithms with sufficient data can rigorously match or exceed quantum models on classical data distributions unless the data exhibits specific quantum-computational structure. These theoretical bounds highlight that mapping classical text into a quantum Hilbert space does not inherently guarantee representational superiority.

---

## 2.3 Classical Text Classification as the Benchmarking Reference

To evaluate whether a quantum kernel confers an empirical advantage, establishing rigorous classical baselines is an indispensable requirement [Huang et al., 2021; Li et al., 2026]. In text classification and security NLP, classical linear and kernel methods represent mature, highly optimized technologies with decades of empirical validation [Cortes & Vapnik, 1995; Al-Sallami et al., 2023].

Standard n-gram term frequency-inverse document frequency (TF-IDF) representations, when paired with Linear Support Vector Machines (LinearSVC), regularly achieve state-of-the-art or near-ceiling performance on spam and phishing detection benchmarks [Cortes & Vapnik, 1995; Verma & Hossain, 2017 [VERIFY CITATION]]. When nonlinear decision boundaries are required, the classical Gaussian Radial Basis Function (RBF) kernel,
\begin{equation}
k_{\text{RBF}}(\mathbf{x}, \mathbf{z}) = \exp\left(-\gamma \|\mathbf{x} - \mathbf{z}\|_2^2\right), \quad \gamma > 0
\end{equation}
provides an established universal approximator capable of separating complex topologies in continuous vector spaces. Furthermore, modern transformer backbones (such as BERT and RoBERTa) provide dense contextual semantic embeddings that capture deep syntactic dependencies, achieving benchmark F1 scores exceeding $0.98$ on security tasks [Garg et al., 2024; Al-Sallami et al., 2023].

A critical methodological hazard in applied QML is the *unmatched baseline fallacy*: comparing a quantum model operating on compressed low-dimensional features against an arbitrary or misconfigured classical model, or comparing a quantum model on dense embeddings against a linear model on raw bag-of-words. When evaluating non-linear kernel capacity, the classical RBF kernel must receive the exact same representation, dimensionality, scaling, and threshold optimization as the quantum kernel. Without strictly matched classical controls, observed performance differences reflect representation mismatches rather than quantum kernel properties [Li et al., 2026].

---

## 2.4 Distribution Shift and Cross-Domain Evaluation

A fundamental limitation of existing QML text benchmarks is their exclusive reliance on independent and identically distributed (IID) random train-test splits. In cybersecurity applications, in-distribution test accuracy is notoriously deceptive [Al-Sallami et al., 2023; Cova et al., 2008]. Phishing and scam detection models operate in non-stationary environments characterized by three distinct modes of distribution shift:

1. **Source Shift**: E-mail and SMS corpora are collected from diverse communication platforms, server infrastructures, and organizational domains. Classifiers trained on one source (e.g., corporate enron archives) frequently fail when transferred to different collections due to domain-specific vocabulary and stylistic conventions [Al-Sallami et al., 2023].
2. **Lexical and Concept Drift**: Attackers actively modify terminology, obfuscate tokens, and alter message themes over time to evade static filter rules [Ren et al., 2022].
3. **Class Prior Shifts**: The relative proportion of malicious to legitimate traffic fluctuates across organizational contexts and temporal windows.

Evaluating models under cross-source holdout protocols—such as training on specific historical collections and evaluating out-of-sample on distinct, unseen sources—is therefore essential for testing whether a classifier has learned generalizable semantic patterns or merely memorized source-specific lexical artifacts. Prior to this study, cross-source domain holdout had never been rigorously evaluated for quantum kernel text classifiers.

---

## 2.5 Benchmarking, Negative Results, and Reproducibility in QML

The field of quantum machine learning has increasingly recognized the necessity of methodological discipline, standardized benchmarking protocols, and the transparent reporting of negative and conditional results [Preskill, 2018; Cerezo et al., 2021; Biamonte et al., 2017]. In a comprehensive tabular benchmarking study across nine OpenML datasets, Li et al. [2026] demonstrated that quantum fidelity kernels showed no statistically significant performance advantage over tuned classical RBF kernels when evaluated across multiple independent seeds, with feature representation accounting for over $70\%$ of performance variance.

Rigorous empirical evaluation in QML requires adherence to five foundational principles:
- **Zero Test Leakage**: Preprocessing pipelines (vocabulary construction, scaling parameters, dimensionality reduction matrices) must be fitted strictly on training data partitions.
- **Fixed Frozen Partitions**: Benchmark splits must be permanently recorded and hash-verified to eliminate optimistic split cherry-picking.
- **Multi-Seed Replication**: Models must be evaluated across multiple independent random seeds to quantify stochastic variance in optimization and projection.
- **Predefined Practical Effect-Size Thresholds**: Statistical significance testing must be paired with practical equivalence thresholds ($\varepsilon$) to prevent marginal, sub-percentage-point fluctuations from being mischaracterized as practical quantum advantages.
- **Comprehensive Computational Profiling**: Wall-clock runtimes and memory allocations must be documented across all pipeline stages (kernel construction, training, inference).

In this work, our contribution is explicitly structured as an empirical benchmarking and diagnostic study adhering strictly to these standards, rather than an algorithmic proposal.

---

## 2.6 The Unresolved Research Gap

In summary, the literature demonstrates that while quantum kernel methods can be implemented for security-related classification, existing evidence remains fragmented by small sample sizes, unmatched classical controls, lack of representation isolation, absence of domain-shift evaluations, and omitted computational scaling costs. It remains entirely unresolved whether quantum fidelity kernels provide any consistent, statistically robust, or practically meaningful advantage over matched classical RBF baselines when evaluated under controlled conditions on real-world natural language text.

This study directly addresses this research gap by establishing a controlled benchmark that isolates:
1. **Model Capacity**: Comparing parameter-free quantum fidelity kernels against matched classical RBF kernels on identical inputs.
2. **Representation and Dimensionality**: Evaluating sparse n-gram TF-IDF versus dense contextual embeddings across systematic dimensionality sweeps ($2\text{D}\text{--}16\text{D}$).
3. **Out-of-Distribution Robustness**: Measuring cross-source transfer degradation under multi-source email holdouts.
4. **Feature Space Geometry**: Measuring Gram matrix correlations, target label alignment, state dispersion, and pairwise kernel diversity.
5. **Computational Cost**: Profiling exact wall-clock execution scaling and memory demands under classical statevector simulation.

<!-- CLAIM AUDIT:
- Section 2.1: QML cybersecurity review, phishing apps, VQC/QSVC. Limitations: small N, unmatched baselines, leakage, no domain shift, no runtimes. Evidence: Literature Comparison Matrix (P01, P02, P03, P08, P17), Hridi2026, Guddanti2026, Shahriyar2025.
- Section 2.2: QSVC formulation, fidelity kernel, Hilbert space mapping, concentration theory, data bounds. Evidence: Havlicek2019, Schuld2019, Thanasilp2024, Huang2021, Garg2024.
- Section 2.3: Classical baselines (TF-IDF, LinearSVM, RBF, Transformers), unmatched baseline fallacy. Evidence: Cortes1995, Al-Sallami2023, Li2026.
- Section 2.4: Distribution shift in security NLP (source shift, lexical drift, class prior shifts). Evidence: Al-Sallami2023, Ren2022, Cova2008.
- Section 2.5: Benchmarking standards, negative results, reproducibility, 5 principles. Evidence: Preskill2018, Cerezo2021, Li2026, FINAL_PROTOCOL_V1.md.
- Section 2.6: Formulation of exact unresolved research gap and 5 study components. Evidence: NOVELTY_STATEMENT.md, PAPER_BLUEPRINT.md.
-->
