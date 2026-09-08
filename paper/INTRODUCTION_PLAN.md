# Introduction Section Architecture and Paragraph Blueprint

**Document Version**: 1.0 (Frozen)  
**Date**: September 5, 2026  
**Scope**: Paragraph-by-paragraph logical flow, argumentative claims, and evidence anchors for Section 1 (Introduction).

---

## Paragraph-by-Paragraph Structural Blueprint

### Paragraph 1: The Problem of Security-Oriented Text Classification
- **Core Theme**: High-stakes nature of phishing, scam, and social engineering text detection.
- **Key Argument**: Text-based cyberattacks inflict billions of dollars in annual global losses. Although deep learning and standard NLP models achieve high in-distribution accuracy, they frequently suffer severe degradation under distribution shifts, lexical drift, and cross-source transfer.
- **Evidence & Citations**: Real-world phishing loss statistics; vulnerability of automated security filters to distribution shifts.

### Paragraph 2: Theoretical Promise of Quantum Kernel Methods
- **Core Theme**: Mathematical motivation for quantum feature maps in machine learning.
- **Key Argument**: Quantum kernel methods map classical input vectors into high-dimensional Hilbert spaces via parameterized quantum circuits $\mathcal{U}_{\Phi(x)}$, computing inner products $k(x, z) = |\langle \psi(x) | \psi(z) \rangle|^2$ that may be classically intractable to evaluate directly. This has led to hypotheses that quantum kernels could uncover non-linear separability patterns inaccessible to classical kernels.
- **Evidence & Citations**: Havlíček et al. (2019), Schuld & Killoran (2019), Huang et al. (2021).

### Paragraph 3: Methodological Limitations in Current Cybersecurity QML Literature
- **Core Theme**: Critical audit of prior QML text classification studies.
- **Key Argument**: Despite growing interest, existing empirical QML studies in cybersecurity suffer from severe methodological confounds:
  1. Reliance on tiny synthetic or toy datasets ($N < 500$ samples).
  2. Data leakage from improper train-test split hygiene or fitting representations across entire corpora.
  3. Comparison against weak or arbitrarily unmatched classical baselines (e.g., comparing tuned quantum models against unscaled linear classifiers).
  4. Complete absence of cross-domain or cross-source shift evaluations.
  5. Omission of exact computational runtime and memory scaling profiles.
- **Tone**: Objective, constructive scientific audit without dismissive rhetoric.

### Paragraph 4: The Unresolved Research Gap
- **Core Theme**: Formulation of the central scientific question.
- **Key Argument**: It remains completely unproven whether quantum kernels provide any genuine, reproducible advantage over matched classical nonlinear baselines (such as RBF kernels) when evaluated under controlled, leakage-free conditions on realistic natural language text.
- **Primary Research Question**: *"Under controlled matched conditions, do quantum kernel methods provide a consistent and practically meaningful advantage over matched classical RBF kernels for text-based scam and phishing detection?"*

### Paragraph 5: Overview of the Controlled Multi-Dataset Benchmark
- **Core Theme**: Scope and methodology of our study.
- **Key Argument**: We design a benchmark spanning three frozen corpora totaling 152,000+ texts: SMS Spam Collection (5.5k), CEAS 2008 (39k raw $\to$ 15k canonical), and MeAJOR (108k). We ensure strict input matching (identical TF-IDF and PCA representations), zero test leakage, validation-only thresholding, and evaluation across 10 independent random seeds.
- **Scope**: Evaluates in-distribution scaling (2D to 12D), cross-source domain holdout (TREC 2007 $\to$ TREC 2005/2006), kernel geometry, and classical simulation overhead.

### Paragraph 6: Summary of Main Findings
- **Core Theme**: High-level empirical takeaways.
- **Key Findings**:
  1. *In-Distribution Parity*: Quantum kernels closely track classical RBF kernels at 8D ($\text{F1} = 0.8754$ vs $0.8709$, $\Delta = +0.0046$), reaching complete parity at 12D ($\Delta = +0.0014$, $p=0.2824$), well within the predefined practical equivalence margin ($\varepsilon = 0.01$).
  2. *Domain Generalization Vulnerability*: Under source holdout, the quantum kernel exhibits greater degradation than classical RBF ($\Delta \text{F1} = -0.0233$, $p=0.0046$).
  3. *Representation Primacy*: Text representation sparsity and dimensionality exert orders of magnitude larger performance effects than kernel choice.
  4. *Simulation Cost*: Exact quantum simulation incurs a $64\times$ runtime penalty at 12 qubits without delivering accuracy gains.

### Paragraph 7: Summary of Contributions
- **Core Theme**: Explicit listing of the 5 publication contributions codified in `paper/CONTRIBUTIONS.md`.
