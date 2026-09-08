# Paper Contributions Architecture

**Document Version**: 1.0 (Frozen)  
**Date**: September 5, 2026  
**Scope**: Publication-ready scientific contribution declarations for the manuscript.

---

## Final Declarations of Contribution

The contributions of this study are conservative, strictly grounded in empirical evidence, and framed around rigorous methodology rather than algorithmic novelty:

1. **Controlled Multi-Dataset Benchmark Protocol**:
   We establish a rigorous, leakage-safe empirical benchmark evaluating parameter-free quantum fidelity kernels against matched classical RBF kernels on security-oriented text classification across three benchmark corpora (SMS Spam Collection, CEAS 2008, and MeAJOR, totaling 152,000+ evaluated samples).

2. **Dimensionality Scaling and Parity Analysis**:
   We demonstrate empirically that low-dimensional quantum weakness ($<8\text{D}$) is an information bottleneck caused by PCA compression, and that increasing representation capacity monotonically restores classification performance to practical parity with classical RBF kernels ($|\Delta \text{F1}| \le 0.01$) at 12 dimensions.

3. **Domain-Shift and Cross-Source Evaluation**:
   We provide a cross-source holdout evaluation (MeAJOR TREC 2007 $\to$ TREC 2005/2006) demonstrating that the evaluated quantum kernel exhibits no inherent domain robustness advantage, suffering a statistically significant degradation penalty relative to matched classical baselines ($\Delta \text{F1} = -0.0233$, $p = 0.0046$).

4. **Geometric and Computational Diagnostic Profiling**:
   We connect classification behavior to underlying feature space geometry through Gram matrix entry correlations, target label alignment, statevector dispersion, and pairwise kernel diversity, while quantifying the substantial computational scaling overhead of classical quantum simulation ($64\times$ total runtime penalty at 12 qubits).

5. **Open Science and Reproducible Artifact Package**:
   We release a fully reproducible experimental framework with hash-verified frozen data splits, 10-seed independent replications, predefined practical equivalence boundaries ($\varepsilon = 0.01$), paired permutation tests with FDR adjustment, and complete open-source configurations.

---

## Prohibited Phrasing and Claim Boundaries
- Do NOT claim to be the "first" study unless strictly scoped to "multi-dataset controlled comparison for NLP text security".
- Do NOT use "definitive proof" or "universal inferiority/superiority".
- Do NOT claim a "novel quantum algorithm" or "novel feature map".
- Frame contributions around rigorous evaluation, negative/conditional findings, and representation-geometry diagnostics.
