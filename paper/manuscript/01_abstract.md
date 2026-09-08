# Abstract

**Working Title**: *When Do Quantum Kernels Help for Text Security? A Multi-Dataset Evaluation of Representation, Geometry, Generalization, and Computational Cost*

Text-based social engineering attacks, including email phishing and SMS scams, impose severe security threats that frequently degrade under real-world distribution shifts. Quantum kernel methods have been theoretically hypothesized to offer representational advantages in complex feature spaces; however, empirical evaluations in natural language processing largely rely on small synthetic datasets, unvalidated splits, or unmatched classical baselines. 

In this work, we present a controlled empirical evaluation comparing parameter-free quantum fidelity kernels with matched classical radial basis function (RBF) kernels across three benchmark corpora comprising over 152,000 text samples: SMS Spam Collection, CEAS 2008, and the MeAJOR archive. Using frozen, leakage-safe protocols and 10 independent random seeds, we evaluate in-distribution scaling ($2\text{--}12$ qubits), cross-source domain holdout, feature space geometry, and computational overhead. 

In-distribution at 8 dimensions on MeAJOR, the quantum kernel achieves $\text{F1} = 0.8754 \pm 0.0029$, tracking classical RBF ($\text{F1} = 0.8709 \pm 0.0030$; $\Delta \text{F1} = +0.0046$, 95% CI $[+0.0030, +0.0061]$), which constitutes a statistically detectable but practically small improvement within the predefined practical-equivalence region ($\varepsilon = 0.01$). Scaling dimensionality restores practical parity at 12 dimensions ($\text{F1} = 0.9137$ vs $0.9123$; $\Delta \text{F1} = +0.0014$, $p = 0.2824$). Under cross-source domain transfer (Direction B), the quantum kernel exhibits greater vulnerability, suffering a significant deficit relative to RBF ($\text{F1} = 0.6680$ vs $0.6913$; $\Delta \text{F1} = -0.0233$, $p = 0.0046$). Geometric diagnostics reveal $50\%\text{--}60\%$ lower target label alignment, while classical simulation incurs a $64\times$ runtime penalty at 12 dimensions. 

Our results demonstrate that representation sparsity and dimensionality dominate classification outcomes over kernel selection, and no consistent, practically meaningful quantum advantage is demonstrated under the evaluated conditions.

<!-- CLAIM AUDIT:
- Motivation/Problem: Text security attacks & degradation under shift. Evidence: General security NLP literature.
- Gap: Prior QML literature uses toy sets/unmatched baselines. Evidence: Literature comparison matrix (P01, P03, P09, P17).
- Benchmark Scope: SMS, CEAS, MeAJOR (152k+ samples, 10 seeds, leak-free). Evidence: FINAL_PROTOCOL_V1.md, Exp 40.
- IID 8D Result: Q = 0.8754 ± 0.0029, RBF = 0.8709 ± 0.0030, Delta = +0.0046 (95% CI [+0.0030, +0.0061]). Evidence: exp40_statistical_summary.csv.
- IID 10D/12D Result: 10D Delta = +0.0057; 12D Q = 0.9137, RBF = 0.9123, Delta = +0.0014, p = 0.2824. Evidence: exp40_statistical_summary.csv.
- Direction B Result: Q = 0.6680 ± 0.0094, RBF = 0.6913 ± 0.0161, Delta = -0.0233, p = 0.0046. Evidence: exp40_statistical_summary.csv.
- Geometry & Simulation Cost: Alignment 50-60% lower; 12D runtime 108.8s vs 1.7s (64x). Evidence: exp40_results.csv, Table 7.
- Primary Conclusion: No consistent practically meaningful quantum advantage demonstrated under evaluated conditions. Evidence: UPDATED_PAPER_CLAIMS.md (C1, C2, C9).
-->
