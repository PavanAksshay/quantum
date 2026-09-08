# Related Work Taxonomy and Literature Analysis

**Document Version**: 1.0 (Frozen)  
**Date**: September 5, 2026  
**Scope**: Literature categorization, prior art analysis, identified gaps, and explicit differentiators for Section 2 (Related Work).

---

## 2.1 Quantum Kernels and Quantum Feature Spaces
- **What Prior Work Establishes**:
  - Foundational framework for quantum kernel estimation and quantum support vector classifiers (QSVC) mapping classical data into Hilbert spaces via parameterized unitaries $\mathcal{U}_{\Phi(x)}$ [Havlíček et al., 2019; Schuld & Killoran, 2019].
  - Theoretical conditions under which quantum kernels can provide provable computational speedups or separation from classical learners, particularly on structured group-theoretic problems [Liu et al., 2021; Huang et al., 2021].
  - Geometric characterization of quantum kernels, including kernel-target alignment [Wang et al., 2021; Hubregtsen et al., 2022] and expressivity vs trainability trade-offs.
- **What Remains Unresolved**:
  - Whether theoretical separations translate to empirical performance advantages on unstructured, real-world continuous or text data.
- **How Our Work Differs**:
  - We evaluate quantum fidelity kernels empirically on real text data against rigorously matched classical RBF kernels across multiple scales.

---

## 2.2 Quantum Machine Learning for Cybersecurity
- **What Prior Work Establishes**:
  - Exploratory applications of variational quantum classifiers (VQC) and quantum kernels to network intrusion detection, malware classification, and spam detection [e.g., Sagingalieva et al., 2022; Ahmed et al., 2022; [VERIFY CITATION]].
  - Hypotheses that high-dimensional quantum Hilbert spaces provide resilience against adversarial evasion or obfuscation.
- **What Remains Unresolved**:
  - The vast majority of security QML studies evaluate tiny sample subsets ($N < 500$), lack independent cross-validation seeds, omit leakage checks, and compare against un-tuned classical baselines.
- **How Our Work Differs**:
  - We establish a 152,000+ sample benchmark with hash-verified frozen splits, 10-seed independent replications, and strict leakage elimination.

---

## 2.3 Quantum Kernels for NLP and Text Classification
- **What Prior Work Establishes**:
  - Quantum Natural Language Processing (QNLP) frameworks utilizing categorical compositional models (DisCoCat) [Coecke et al., 2020; Lorenz et al., 2021 [VERIFY CITATION]].
  - Quantum bag-of-words and embedding-based quantum kernel classifiers evaluated on sentiment analysis or topic classification [e.g., Di Sipio et al., 2021; Li et al., 2022 [VERIFY CITATION]].
- **What Remains Unresolved**:
  - Contradictory empirical findings in literature: some papers report quantum superiority while others report severe underperformance, with no systematic investigation into the role of text representation (TF-IDF vs dense embeddings) or PCA information loss.
- **How Our Work Differs**:
  - We systematically evaluate the representation interaction (TF-IDF vs RoBERTa) and demonstrate that representation sparsity dictates whether quantum kernels track or fail against classical models.

---

## 2.4 Representation and Dimensionality Constraints in QML
- **What Prior Work Establishes**:
  - Because NISQ devices and classical simulators are constrained to low qubit counts ($n \le 16\text{--}20$), high-dimensional classical data must be compressed via PCA, autoencoders, or random projections.
  - Theoretical phenomena of barren plateaus and exponential concentration of quantum kernels as qubit count grows [McClean et al., 2018; Thanasilp et al., 2022; Kübler et al., 2021].
- **What Remains Unresolved**:
  - How much of observed low-dimensional QML weakness is due to PCA information loss versus intrinsic quantum feature map properties.
- **How Our Work Differs**:
  - We isolate the dimensionality scaling trajectory from 2D to 16D, proving that low-dimensional weakness is an information bottleneck that recovers monotonically up to 12 qubits.

---

## 2.5 Distribution Shift and Robustness in Security NLP
- **What Prior Work Establishes**:
  - Natural language security classifiers suffer severe performance drops under domain shift, temporal drift, and cross-source transfer [e.g., Cova et al., 2008; Toolan & Carthy, 2010; Verma & Hossain, 2017 [VERIFY CITATION]].
  - Standard in-distribution cross-validation drastically overestimates real-world phishing detection efficacy.
- **What Remains Unresolved**:
  - Prior QML literature has never evaluated cross-source domain holdout in text classification under rigorous controls.
- **How Our Work Differs**:
  - We conduct the first multi-source holdout evaluation (MeAJOR TREC 2007 $\to$ TREC 2005/2006) for quantum kernels, demonstrating that quantum models degrade more severely than classical RBF.

---

## 2.6 Benchmarking, Negative Results, and Reproducibility in QML
- **What Prior Work Establishes**:
  - Growing calls in the QML community for rigorous classical baselines, standardized datasets, and disciplined reporting of negative or conditional findings [Preskill, 2018; Cerezo et al., 2021; Biamonte et al., 2017].
- **What Remains Unresolved**:
  - Standardized benchmarks with predefined practical equivalence boundaries and wall-clock simulation scaling profiles remain rare.
- **How Our Work Differs**:
  - We release complete open-source protocols, configuration hashes, exact seed-level results, and wall-clock computational cost breakdowns.
