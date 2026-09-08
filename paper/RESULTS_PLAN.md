# Results Section Architecture and Subsections Blueprint

**Document Version**: 1.0 (Frozen)  
**Date**: September 5, 2026  
**Scope**: Structure, empirical evidence, tables/figures, interpretations, and limitation framing for Section 4/5 (Results), organized by Research Question.

---

## 5.1 RQ1: In-Distribution Model Performance (Quantum vs Matched Classical RBF)

- **Research Question**: *Under matched low-dimensional representations, how does the classification performance of the parameter-free quantum fidelity kernel compare against the classical RBF kernel?*
- **Empirical Evidence**:
  - Primary anchor: MeAJOR IID 8D across $N=10$ independent seeds (Exp40A).
  - Cross-corpus replication: SMS Spam (8D) and CEAS 2008 (8D) (Exp 30).
- **Primary Results**:
  - **MeAJOR IID 8D**: Quantum F1 = $0.8754 \pm 0.0029$, PR-AUC = $0.9372$, ROC-AUC = $0.9515$. Classical RBF F1 = $0.8709 \pm 0.0030$, PR-AUC = $0.9423$, ROC-AUC = $0.9535$. Linear SVM (8D) F1 = $0.8445 \pm 0.0022$.
  - **Paired Difference**: $\Delta \text{F1} = +0.0046 \pm 0.0027$ (+0.46 pp; 95% Bootstrap CI $[+0.0030, +0.0061]$).
  - **CEAS 8D**: Quantum F1 = $0.9736$ vs Classical RBF F1 = $0.9641$ ($\Delta = +0.0095$).
  - **SMS 8D**: Quantum F1 = $0.8841$ vs Classical RBF F1 = $0.8912$ ($\Delta = -0.0071$).
- **Interpretation**: Both nonlinear kernels comfortably outperform the low-dimensional linear baseline (+2.6 pp to +3.1 pp), demonstrating valid nonlinear boundary formation. However, across all corpora, the quantum-classical difference remains strictly inside the predefined practical-equivalence margin ($\varepsilon = 0.01$).
- **Section Limitation**: Evaluation in low dimensions is constrained by PCA information loss.

---

## 5.2 RQ2: Dimensionality Scaling Trajectory and Information Recovery

- **Research Question**: *How does classification performance scale with representation dimensionality ($d \in [2, 16]$ qubits/components), and does quantum performance saturate or recover?*
- **Empirical Evidence**:
  - Full dimensionality sweep: $d \in \{2, 4, 6, 8, 10, 12, 16\}$ (Exp 35).
  - 10-seed confirmation anchors: $8\text{D}, 10\text{D}, 12\text{D}$ (Exp40C).
- **Primary Results**:
  - **Monotonic Recovery**: Quantum F1 scales from $0.6447$ (2D) $\to$ $0.7876$ (4D) $\to$ $0.8253$ (6D) $\to$ $0.8754$ (8D) $\to$ $0.9023$ (10D) $\to$ $0.9137$ (12D), representing a $+41.7\%$ relative gain ($+0.2690$ F1).
  - **RBF Tracking**: Classical RBF scales from $0.6735$ (2D) $\to$ $0.9123$ (12D).
  - **Convergence to Parity**: At 12D, $\Delta \text{F1} = +0.0014 \pm 0.0040$ (95% Bootstrap CI $[-0.0010, +0.0037]$; Permutation $p = 0.2824$).
- **Interpretation**: Early low-dimensional quantum underperformance ($<8\text{D}$) is primarily an information bottleneck caused by aggressive PCA compression. Scaling dimensions restores parity with classical kernels.
- **Section Limitation**: Direct simulation of the quantum kernel at 16D on 10,000 samples is computationally and memory-infeasible ($>10.5\text{ GB}$).

---

## 5.3 RQ3: Representation Interaction (Sparse TF-IDF vs Dense Embeddings)

- **Research Question**: *How does the choice of text representation (sparse n-gram TF-IDF vs dense RoBERTa embeddings) interact with quantum kernel mapping geometry and classification accuracy?*
- **Empirical Evidence**:
  - Representation ablation on CEAS and MeAJOR (Exp 25, Exp 26, Exp 29, Exp 34).
- **Primary Results**:
  - On CEAS 8D, switching from RoBERTa dense embeddings to sparse TF-IDF shifted the quantum-minus-RBF margin from $-2.95$ pp to $+0.95$ pp (a $3.90$ pp net inversion).
  - High-dimensional Linear SVM on 50k TF-IDF achieved F1 = $0.9721$ on MeAJOR IID and F1 = $0.8922$ on Direction A holdout, outperforming low-dimensional models by $>17$ pp.
- **Interpretation**: The cyclic $ZZFeatureMap$ performs well on sparse, orthogonal TF-IDF components but degrades on dense, highly correlated embedding dimensions. Representation design exerts an order of magnitude larger effect than kernel substitution.
- **Section Limitation**: Evaluated single transformer backbone (RoBERTa-base) and single linear embedding projection.

---

## 5.4 RQ4: Quantum Kernel Geometry and Label Alignment

- **Research Question**: *What geometric properties (Gram matrix correlation, target label alignment, state dispersion) characterize quantum feature spaces relative to classical RBF?*
- **Empirical Evidence**:
  - Geometric diagnostic audit across SMS, CEAS, and MeAJOR (Exp 27, Exp 31, Exp 32, Exp 35).
- **Primary Results**:
  - **Gram Correlation**: Quantum and RBF off-diagonal Gram entries exhibit moderate-to-strong correlation ($r = 0.55\text{--}0.65$ across 2D–12D), falling to $0.46$ at 16D.
  - **Label Alignment Deficit**: Target label alignment of the quantum kernel is $50\%\text{--}60\%$ lower than classical RBF across all corpora (e.g., $0.038$ vs $0.069$ on MeAJOR; $0.040$ vs $0.077$ on CEAS).
  - **Dispersion vs Diversity**: Statevector entropy is strongly negatively correlated with pairwise kernel diversity ($r = -0.78$ to $-0.83$).
- **Interpretation**: Quantum fidelity kernels partially mirror classical Gaussian RBF geometry in intermediate dimensions, but exhibit substantially weaker class alignment.
- **Section Limitation**: Diagnostic metrics are evaluated on classical statevector inner products.

---

## 5.5 RQ5: Cross-Source Domain Generalization and Robustness

- **Research Question**: *How robust are quantum kernels compared to classical RBF and linear baselines under severe cross-source domain shift?*
- **Empirical Evidence**:
  - MeAJOR source holdout Direction A (TREC5+6 $\to$ TREC7) (Exp 33, Exp 36).
  - MeAJOR source holdout Direction B (TREC7 $\to$ TREC5+6) across 10 seeds (Exp40B).
- **Primary Results**:
  - **Direction B 8D ($N=10$ seeds)**: Quantum F1 = $0.6680 \pm 0.0094$ vs Classical RBF F1 = $0.6913 \pm 0.0161$ ($\Delta \text{F1} = -0.0233 \pm 0.0200$; 95% Bootstrap CI $[-0.0353, -0.0117]$; Permutation $p = 0.0046$; BH FDR $p = 0.0069$).
  - **Relative Degradation**: Quantum degraded by $23.7\%$ ($0.8754 \to 0.6680$), whereas Classical RBF degraded by $20.6\%$ ($0.8709 \to 0.6913$).
- **Interpretation**: The quantum fidelity kernel exhibits no inherent domain-shift robustness, suffering a statistically significant disadvantage relative to classical baselines under transfer.
- **Section Limitation**: Domain transfer is evaluated across email collections from the same historical era (TREC 2005–2007).

---

## 5.6 RQ6: Computational Runtime and Memory Scaling

- **Research Question**: *What is the computational execution time and memory scaling profile of classical quantum statevector simulation compared to classical RBF evaluation?*
- **Empirical Evidence**:
  - Runtime audit and 10-seed confirmation benchmarking (Exp 35, Exp 36, Exp 40C).
- **Primary Results**:
  - **Wall-Clock Runtimes per Run**:
    - 8D: Quantum = $7.7\text{s}$ ($3.7\times$ RBF); RBF = $2.1\text{s}$.
    - 10D: Quantum = $19.1\text{s}$ ($11.9\times$ RBF); RBF = $1.6\text{s}$.
    - 12D: Quantum = $108.8\text{s}$ ($64.0\times$ RBF); RBF = $1.7\text{s}$.
  - **Memory Bottleneck**: At 16 qubits on 10,000 samples, statevector allocation exceeded $10.5\text{ GB}$ complex128 RAM in local execution.
- **Interpretation**: Quantum simulation incurs rapidly escalating computational execution time and memory demands without yielding predictive accuracy gains over classical models.
- **Section Limitation**: Measurements reflect software statevector simulation on classical CPUs/GPUs, not physical quantum hardware execution.

---

## 5.7 Inferential Statistical Synthesis and Equivalence Testing
- Comprehensive table compiling paired differences, 95% bootstrap CIs, permutation tests, McNemar tests, and Benjamini-Hochberg FDR adjustments (Table 6).
- Confirmation that all IID differences satisfy practical equivalence ($|\Delta \text{F1}| \le 0.01$) while Direction B shift confirms a statistically significant classical advantage ($\Delta \text{F1} = -0.0233$, $p=0.0046$).
