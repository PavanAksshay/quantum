# Summary of Contributions

The contributions of this paper are strictly empirical, methodological, and diagnostic, resolving an open comparative question in applied quantum machine learning without overclaiming algorithmic novelty:

1. **Controlled Multi-Dataset Benchmark Protocol**:  
   We introduce a rigorous, leakage-safe empirical benchmark evaluating parameter-free quantum fidelity kernels against matched classical RBF kernels for security-oriented text classification across three benchmark corpora (SMS Spam Collection, CEAS 2008, and the MeAJOR archive, comprising over 152,000 text samples). Both model families operate on strictly identical low-dimensional representations ($d \in [2, 16]$ components/qubits) derived from training-fitted TF-IDF features and PCA projections, with validation-only threshold tuning.

2. **Dimensionality Scaling and Representation Dominance Analysis**:  
   We demonstrate empirically that low-dimensional quantum weakness ($<8\text{D}$) is an information bottleneck resulting from aggressive PCA compression rather than an inherent failure of quantum feature space geometry. Monotonic scaling from 2D to 12D restores classification performance to practical parity with classical RBF ($\Delta \text{F1} = +0.0014$, $p = 0.2824$ at 12D). Furthermore, we show that representation sparsity (TF-IDF vs RoBERTa) and dimensionality exert an order of magnitude larger effect on classification accuracy ($>10$ to $>26$ percentage points) than the mathematical choice of kernel function ($<1.0$ percentage point).

3. **Cross-Source Domain Generalization Evaluation**:  
   We provide a multi-source domain-holdout evaluation (MeAJOR TREC 2007 $\to$ TREC 2005/2006 across 10 independent seeds) testing the hypothesis that quantum Hilbert spaces provide inherent robustness under distribution shift. We find that the evaluated quantum kernel suffers a statistically significant degradation penalty relative to the matched classical RBF baseline ($\text{F1} = 0.6680$ vs $0.6913$; $\Delta \text{F1} = -0.0233$, $p = 0.0046$), demonstrating that the quantum feature map provides no domain robustness advantage under lexical transfer.

4. **Geometric and Computational Diagnostic Profiling**:  
   We analyze the geometric properties of quantum feature spaces by quantifying Gram matrix entry correlations ($r = 0.55\text{--}0.65$ with classical RBF), target label alignment ($50\%\text{--}60\%$ lower for quantum), statevector dispersion, and pairwise kernel diversity ($r = -0.80$ inverse correlation). Concurrently, we document the practical computational cost of exact classical quantum simulation, demonstrating a $64\times$ total runtime penalty at 12 dimensions ($108.8\text{s}$ vs $1.7\text{s}$) without corresponding accuracy gains.

5. **Reproducible Open-Science Benchmark Suite**:  
   We release a fully reproducible experimental framework featuring hash-verified frozen splits, 10-seed independent execution logs, complete configuration files, and exact software/hardware environment metadata. Statistical evaluation incorporates non-parametric percentile bootstrap confidence intervals ($B = 10,000$), paired permutation tests, and Benjamini-Hochberg False Discovery Rate control under a predefined practical-equivalence threshold ($\varepsilon = 0.01$ F1).

<!-- CLAIM AUDIT:
- Contribution 1: Multi-dataset benchmark on SMS, CEAS, MeAJOR (152k+ samples, matched TF-IDF + PCA, zero leakage). Evidence: FINAL_PROTOCOL_V1.md, Exp 24-36.
- Contribution 2: Dimensionality scaling 2D -> 12D, parity at 12D (Delta = +0.0014, p = 0.2824), representation dominance. Evidence: Exp 35, Exp 29, exp40_statistical_summary.csv.
- Contribution 3: Domain holdout Direction B (Q = 0.6680 vs RBF 0.6913, Delta = -0.0233, p = 0.0046). Evidence: Exp 33, Exp 40B, exp40_statistical_summary.csv.
- Contribution 4: Geometry (Gram r = 0.55-0.65, alignment deficit 50-60%) & Simulation runtime (108.8s vs 1.7s, 64x penalty). Evidence: Exp 27, Exp 31, Exp 35, Table 7, Table 8.
- Contribution 5: Reproducibility, 10 seeds, bootstrap CIs, permutation tests, BH FDR, epsilon = 0.01. Evidence: exp40_results.csv, RUN_METADATA.md, FINAL_PROTOCOL_V1.md.
-->
