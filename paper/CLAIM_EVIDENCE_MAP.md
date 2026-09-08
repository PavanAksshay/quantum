# Comprehensive Claim-Evidence Mapping Architecture

**Document Version**: 1.0 (Frozen)  
**Date**: September 5, 2026  
**Scope**: Explicit, auditable trace from every scientific claim to its underlying empirical evidence, table, figure, and inferential statistical test.

---

## 1. Claim-to-Evidence Matrix

```
[Claim ID] → [Research Question] → [Experiment] → [Dataset(s)] → [Table / Figure] → [Statistical Support] → [Claim Strength]
```

---

### Claim 1 (C1): Quantum kernels achieve classification performance comparable to matched classical RBF kernels under IID conditions at sufficient dimensionality.
- **Research Question**: RQ1 (Model Comparison) & RQ2 (Dimensionality Scaling)
- **Source Experiment(s)**: Exp 30, Exp 35, Exp 36, Exp 40A, Exp 40C
- **Evaluated Dataset(s)**: MeAJOR (IID), CEAS (IID), SMS (IID)
- **Primary Table / Figure**: Table 3, Table 4, Table 6; Figure 2, Figure 3
- **Statistical Evidence**:
  - MeAJOR IID 8D ($N=10$ seeds): Quantum F1 = $0.8754 \pm 0.0029$ vs Classical RBF F1 = $0.8709 \pm 0.0030$ ($\Delta \text{F1} = +0.0046 \pm 0.0027$; 95% Bootstrap CI $[+0.0030, +0.0061]$).
  - MeAJOR IID 10D ($N=10$ seeds): Quantum F1 = $0.9023 \pm 0.0034$ vs Classical RBF F1 = $0.8967 \pm 0.0049$ ($\Delta \text{F1} = +0.0057 \pm 0.0042$; 95% Bootstrap CI $[+0.0032, +0.0081]$).
  - MeAJOR IID 12D ($N=10$ seeds): Quantum F1 = $0.9137 \pm 0.0046$ vs Classical RBF F1 = $0.9123 \pm 0.0023$ ($\Delta \text{F1} = +0.0014 \pm 0.0040$; 95% Bootstrap CI $[-0.0010, +0.0037]$; paired permutation $p = 0.2824$).
- **Interpretation**: As dimensionality increases from 2D to 12D, quantum classification performance scales monotonically, reaching parity with classical RBF at 12 dimensions.
- **Claim Strength**: **STRONG / DEFINITIVE**

---

### Claim 2 (C2): Observed in-distribution performance gains for quantum kernels are practically small and fall within the practical equivalence band.
- **Research Question**: RQ1 (Model Comparison)
- **Source Experiment(s)**: Exp 36, Exp 37, Exp 38, Exp 40A, Exp 40C
- **Evaluated Dataset(s)**: MeAJOR, CEAS, SMS
- **Primary Table / Figure**: Table 3, Table 6; Figure 3
- **Statistical Evidence**:
  - Across all 10 independent seeds and all dimensions ($8\text{D}, 10\text{D}, 12\text{D}$), mean $\Delta \text{F1}$ never exceeded the predefined threshold $\varepsilon = 0.01$ ($+1.0$ percentage points).
  - Observed deltas: $+0.46$ pp (8D), $+0.57$ pp (10D), $+0.14$ pp (12D).
  - At 12D, the 95% bootstrap confidence interval $[-0.0010, +0.0037]$ explicitly overlaps zero ($p = 0.2824$).
- **Interpretation**: Small numerical differences observed under low dimensions do not constitute a meaningful advantage and satisfy formal statistical criteria for practical equivalence.
- **Claim Strength**: **STRONG / DEFINITIVE**

---

### Claim 3 (C3): Feature dimensionality substantially affects quantum kernel performance, acting as an information bottleneck at low dimensions.
- **Research Question**: RQ2 (Dimensionality Scaling)
- **Source Experiment(s)**: Exp 35, Exp 37, Exp 40C
- **Evaluated Dataset(s)**: MeAJOR (2D to 16D)
- **Primary Table / Figure**: Table 4; Figure 2
- **Statistical Evidence**:
  - Scaling from 2D to 12D yields a $+26.90$ pp gain in Quantum F1 ($0.6447 \to 0.9137$, a $+41.7\%$ relative improvement), tracking Classical RBF ($0.6735 \to 0.9123$).
  - Marginal gains diminish systematically: $+14.29$ pp ($2\text{D} \to 4\text{D}$), $+3.77$ pp ($4\text{D} \to 6\text{D}$), $+4.99$ pp ($6\text{D} \to 8\text{D}$), $+2.69$ pp ($8\text{D} \to 10\text{D}$), $+1.14$ pp ($10\text{D} \to 12\text{D}$).
- **Interpretation**: Early low-dimensional quantum weakness ($<8\text{D}$) reflects PCA information loss rather than an intrinsic failure of quantum Hilbert space geometry.
- **Claim Strength**: **STRONG / DEFINITIVE**

---

### Claim 4 (C4): Text representation choice materially alters the relative ranking and performance gap between quantum and classical kernels.
- **Research Question**: RQ3 (Representation Interaction)
- **Source Experiment(s)**: Exp 25, Exp 26, Exp 29, Exp 34
- **Evaluated Dataset(s)**: CEAS 2008, MeAJOR
- **Primary Table / Figure**: Table 2, Table 3; Figure 6
- **Statistical Evidence**:
  - On CEAS 8D, switching from RoBERTa dense embeddings to sparse TF-IDF shifted the quantum-minus-RBF margin from $-2.95$ pp to $+0.95$ pp (a $3.90$ pp net shift).
  - High-dimensional Linear SVM on 50k TF-IDF achieved F1 = $0.9721$ on MeAJOR IID and F1 = $0.8922$ on Direction A holdout, outperforming low-dimensional models by $>17$ pp.
- **Interpretation**: The cyclic $ZZFeatureMap$ interacts favorably with sparse, orthogonal TF-IDF components but degrades on dense, correlated deep learning representations. Representation design exerts an order of magnitude larger effect than kernel choice.
- **Claim Strength**: **STRONG / DEFINITIVE**

---

### Claim 5 (C5): Quantum and classical RBF kernels induce related but non-identical feature space geometries.
- **Research Question**: RQ4 (Kernel Geometry)
- **Source Experiment(s)**: Exp 27, Exp 31, Exp 32, Exp 35
- **Evaluated Dataset(s)**: SMS, CEAS, MeAJOR
- **Primary Table / Figure**: Table 7; Figure 7, Figure 8
- **Statistical Evidence**:
  - Pearson correlation between off-diagonal Gram matrix entries of Quantum and RBF models ranges between $r = 0.55$ and $0.65$ across 2D–12D, decaying to $0.46$ at 16D.
  - Statevector entropy is strongly inversely correlated with pairwise kernel diversity ($r = -0.78$ to $-0.83$ across all three corpora).
- **Interpretation**: Quantum fidelity kernels partially mirror classical Gaussian RBF geometry in intermediate dimensions while exhibiting distinct geometric decay at higher qubit allocations.
- **Claim Strength**: **STRONG / DEFINITIVE**

---

### Claim 6 (C6): Quantum kernel-target label alignment is consistently weaker than classical RBF alignment in the evaluated settings.
- **Research Question**: RQ4 (Kernel Geometry)
- **Source Experiment(s)**: Exp 27, Exp 31, Exp 32
- **Evaluated Dataset(s)**: SMS, CEAS, MeAJOR
- **Primary Table / Figure**: Table 7; Figure 7
- **Statistical Evidence**:
  - Quantum kernel-target alignment score is $50\%$ to $60\%$ lower than classical RBF across all three benchmark datasets (SMS: $0.021$ vs $0.045$; CEAS: $0.040$ vs $0.077$; MeAJOR: $0.038$ vs $0.069$).
- **Interpretation**: The parameter-free cyclic $ZZFeatureMap$ produces state vectors that are less concentrated around class labels than classical RBF kernels.
- **Claim Strength**: **STRONG / DEFINITIVE**

---

### Claim 7 (C7): Quantum kernels do not demonstrate a domain-shift robustness advantage, suffering statistically significant degradation under source holdout.
- **Research Question**: RQ5 (Cross-Source Generalization)
- **Source Experiment(s)**: Exp 33, Exp 34, Exp 36, Exp 38, Exp 40B
- **Evaluated Dataset(s)**: MeAJOR (Direction A: TREC5+6 $\to$ TREC7; Direction B: TREC7 $\to$ TREC5+6)
- **Primary Table / Figure**: Table 5, Table 6; Figure 4
- **Statistical Evidence**:
  - Direction B 8D ($N=10$ seeds): Quantum F1 = $0.6680 \pm 0.0094$ vs Classical RBF F1 = $0.6913 \pm 0.0161$ (Paired $\Delta \text{F1} = -0.0233 \pm 0.0200$; 95% Bootstrap CI $[-0.0353, -0.0117]$; Permutation $p = 0.0046$; BH FDR $p = 0.0069$).
  - Relative degradation from IID: Quantum degraded by $23.7\%$ ($0.8754 \to 0.6680$), while Classical RBF degraded by $20.6\%$ ($0.8709 \to 0.6913$).
- **Interpretation**: Quantum fidelity kernels provide no empirical defense against lexical shift across email sources and perform worse than classical baselines under transfer.
- **Claim Strength**: **STRONG / DEFINITIVE**

---

### Claim 8 (C8): Exact classical simulation of quantum kernels incurs rapidly scaling computational overhead as dimensionality increases.
- **Research Question**: RQ6 (Computational Cost)
- **Source Experiment(s)**: Exp 35, Exp 36, Exp 37, Exp 40C
- **Evaluated Dataset(s)**: MeAJOR (8D, 10D, 12D)
- **Primary Table / Figure**: Table 8; Figure 5
- **Statistical Evidence**:
  - Total pipeline runtime per run: 8D = $7.7\text{s}$ ($3.7\times$ RBF), 10D = $19.1\text{s}$ ($11.9\times$ RBF), 12D = $108.8\text{s}$ ($64.0\times$ RBF).
  - Classical RBF runtime remained flat across all dimensions ($1.5\text{s}\text{--}2.1\text{s}$).
  - Memory scaling: At 16 qubits on 10,000 samples, statevector allocation exceeded $10.5\text{ GB}$ complex128 RAM in local benchmarks.
- **Interpretation**: Quantum simulation creates steep computational and memory bottlenecks without delivering predictive superiority over classical baselines.
- **Claim Strength**: **STRONG / DEFINITIVE**

---

### Claim 9 (C9): Across controlled matched conditions, no consistent or practically meaningful quantum advantage is demonstrated.
- **Research Question**: Primary Research Question (Synthesized RQ1–RQ6)
- **Source Experiment(s)**: Exp 24 through Exp 40
- **Evaluated Dataset(s)**: SMS Spam Collection, CEAS 2008, MeAJOR
- **Primary Table / Figure**: All Tables (1–8) and Figures (1–8)
- **Statistical Evidence**:
  - In no evaluated experimental regime did the quantum kernel exceed classical RBF by $>0.01$ F1.
  - In domain transfer, quantum kernels exhibited statistically significant deficits ($\Delta = -0.0233$).
  - Linear models on high-dimensional representations outperformed all 8D nonlinear models by $>10$ to $>17$ pp.
- **Interpretation**: The empirical evidence does not support claims of quantum advantage for security text classification under currently feasible low-to-intermediate dimensional representations.
- **Claim Strength**: **STRONG / DEFINITIVE**
