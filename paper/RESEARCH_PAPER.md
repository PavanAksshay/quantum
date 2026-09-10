# When Do Quantum Kernels Help for Text Security?
## A Multi-Dataset Evaluation of Representation, Geometry, Generalization, and Computational Cost

**Author:** Academic Research Group &bull; Quantum Machine Learning & Text Security  
**Audit Protocol:** Exp 40 Controlled Multi-Seed Benchmark ($N = 10$ Seeds)  
**Target Submission:** IEEE Transactions on Information Forensics and Security / ACM TOPS  

---

## Abstract

Quantum kernel methods have attracted substantial interest for natural language processing and cybersecurity, motivated by theoretical conjectures that mapping classical data into exponentially large Hilbert spaces could yield superior non-linear decision boundaries or enhanced generalization under distribution shift. However, empirical studies often rely on small sample sizes, unmatched classical baselines, or unvalidated split partitions. In this work, we present a controlled empirical evaluation comparing parameter-free quantum fidelity kernels with matched classical radial basis function (RBF) kernels across three benchmark corpora drawing from more than 150,000 audited text records: the **SMS Spam Collection** (5,572 records), the **CEAS 2008 Email Corpus** (15,000 records), and the multi-source **MeAJOR archive** (108,684 records). Using frozen, leakage-safe protocols and 10 independent random seeds, we evaluate in-distribution scaling ($2\text{--}12$ qubits), cross-source domain holdouts (TREC 2007 $\to$ TREC 2005/2006), feature space geometry, and computational overhead.

Under matched in-distribution (IID) conditions, the quantum kernel is competitive with classical RBF, displaying minor statistically detectable improvements at intermediate dimensions ($+0.46$ percentage points at 8D, $p = 0.0016$; $+0.57$ pp at 10D, $p = 0.0052$) that remain strictly within the predefined practical-equivalence threshold ($\varepsilon = 0.01$ F1), converging to complete parity at 12D ($+0.14$ pp, $p = 0.2824$). Under cross-source domain transfer, the quantum kernel exhibits a statistically significant and practically meaningful performance deficit ($\Delta\text{F1} = -0.0233$, $p = 0.0046$, Benjamini–Hochberg adjusted $p = 0.0069$). Representation ablations demonstrate that upstream feature representation dominates kernel selection by an order of magnitude: switching from TF-IDF to dense RoBERTa embeddings shifts relative performance by $3.90$ percentage points. Geometrically, quantum and RBF Gram matrices show moderate correlation ($r \approx 0.55\text{--}0.65$), while single-state entropy is strongly inversely associated with pairwise kernel diversity ($r = -0.78$ to $-0.83$). Computationally, classical statevector simulation of the quantum kernel requires $108.8\text{s}$ per run at 12D compared to $1.7\text{s}$ for RBF ($\approx 64\times$ penalty). We conclude that under the evaluated conditions, parameter-free quantum fidelity kernels do not produce a consistent practical advantage for text security classification, and we outline rigorous benchmarking standards for future applied QML evaluations.

---

## 1. Introduction

Text-based social engineering attacks, including email phishing, SMS scams, and fraudulent communications, represent one of the most pervasive threat vectors in modern digital infrastructure. Automated defense mechanisms rely heavily on natural language processing (NLP) and machine learning classifiers to filter malicious content before it reaches end users. However, building robust classifiers for text security presents distinct methodological challenges:

1. **High Dimensionality & Sparsity:** Lexical vocabularies span tens of thousands of tokens.
2. **Adversarial Distribution Shift:** Attackers actively mutate lexical patterns and token distributions across campaigns and organizational sources.
3. **Severe Class Imbalance:** Legitimate messages vastly outnumber targeted phishing attacks in wild enterprise streams.

![Figure 1: End-to-End Experimental Framework](file:///Users/pavanaksshay/quantum/paper/submission/figures/figure_1_experimental_framework.png)
*Figure 1: Leakage-safe end-to-end experimental framework comparing Classical Gaussian RBF and Quantum Fidelity Kernels across matched feature representations.*

In recent years, quantum machine learning (QML) has emerged as an alternative paradigm for non-linear pattern recognition. In a quantum support vector classifier (QSVC), classical input vectors $\mathbf{x} \in \mathbb{R}^d$ are mapped into quantum states $|\psi(\mathbf{x})\rangle$ residing in a $2^{N_q}$-dimensional complex Hilbert space via a parameterized unitary circuit $\mathcal{U}_{\Phi(\mathbf{x})}$. Pairwise quantum state fidelities form a kernel matrix:

$$K_Q(\mathbf{x}, \mathbf{z}) = |\langle \psi(\mathbf{x}) | \psi(\mathbf{z}) \rangle|^2$$

Despite theoretical excitement, prior literature in applied QML has suffered from pervasive methodological weaknesses: small sample sizes ($N < 1,000$), un-regularized or unmatched classical controls, and zero out-of-distribution evaluation. This paper establishes a rigorous, leakage-safe empirical benchmark to definitively test whether quantum kernels confer real utility in text security.

---

## 2. Methodology & Mathematical Framework

### 2.1 Quantum Fidelity Kernel Formulation
We evaluate the canonical two-layer cyclic $ZZFeatureMap$ on $N_q = d$ qubits:

$$\mathcal{U}_{\Phi(\mathbf{x})} = \left( U_{\Phi(\mathbf{x})} H^{\otimes N_q} \right)^2$$

$$U_{\Phi(\mathbf{x})} = \exp\left( i \sum_{j=1}^{N_q} x_j Z_j + i \sum_{j=1}^{N_q} (\pi - x_j)(\pi - x_{(j \bmod N_q) + 1}) Z_j Z_{(j \bmod N_q) + 1} \right)$$

where $Z_j$ is the Pauli-$Z$ operator on qubit $j$. The resulting Gram matrix is positive semi-definite (PSD) with unit diagonal entries ($K_Q(\mathbf{x}, \mathbf{x}) = 1.0$).

### 2.2 Matched Classical Gaussian RBF Kernel
The classical control evaluates the Gaussian RBF kernel on identical feature vectors:

$$K_{\text{RBF}}(\mathbf{x}, \mathbf{z}) = \exp\left( -\gamma \|\mathbf{x} - \mathbf{z}\|_2^2 \right), \quad \gamma = \frac{1}{d \cdot \text{Var}(X)}$$

### 2.3 Benchmark Datasets

| Corpus | Raw Count | Usable Size | Positive (Spam/Phish) % | Train / Val / Test Split | Domain Nature |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SMS Spam** | 5,574 | 5,572 | 13.41% | 3,343 / 1,114 / 1,115 | Mobile SMS text streams |
| **CEAS 2008** | 39,154 | 15,000 | 18.90% | 10,000 / 2,500 / 2,500 | Phishing vs Ham email collections |
| **MeAJOR Archive** | 108,685 | 108,684 | 19.33% | 10,000 / 2,500 / 2,500 | Multi-source enterprise email (TREC 5/6/7) |

*Table 1: Benchmark Dataset Characteristics and Audit Partitions.*

---

## 3. Empirical Results

### 3.1 In-Distribution Dimensionality Scaling

Across 10 random seeds on the MeAJOR in-distribution benchmark, expanding dimensionality from 2 to 12 qubits produces rapid performance convergence:

| Dimension ($d$) | Quantum F1 ($\bar{x} \pm s$) | Classical RBF F1 ($\bar{x} \pm s$) | Paired $\Delta$F1 | 95% Bootstrap CI | Permutation $p$ | Practical Conclusion ($\varepsilon=0.01$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2D** | $0.6447 \pm 0.0038$ | $0.6735 \pm 0.0031$ | $-0.0288$ | $[-0.0312, -0.0264]$ | $< 0.001$ | Classical Advantage |
| **4D** | $0.7876 \pm 0.0030$ | $0.7918 \pm 0.0025$ | $-0.0042$ | $[-0.0065, -0.0019]$ | $0.0180$ | Practical Equivalence |
| **6D** | $0.8253 \pm 0.0026$ | $0.8254 \pm 0.0022$ | $-0.0001$ | $[-0.0021, +0.0018]$ | $0.9410$ | Practical Equivalence |
| **8D** | $0.8754 \pm 0.0029$ | $0.8709 \pm 0.0030$ | $+0.0046$ | $[+0.0030, +0.0061]$ | $0.0016$ | Practical Equivalence |
| **10D** | $0.9023 \pm 0.0034$ | $0.8967 \pm 0.0049$ | $+0.0057$ | $[+0.0025, +0.0089]$ | $0.0052$ | Practical Equivalence |
| **12D** | $0.9137 \pm 0.0046$ | $0.9123 \pm 0.0023$ | $+0.0014$ | $[-0.0010, +0.0037]$ | $0.2824$ | **Strict Statistical Parity** |

*Table 2: Dimensionality scaling trajectory across 10 computational random seeds on MeAJOR.*

| ![Figure 2](file:///Users/pavanaksshay/quantum/paper/submission/figures/figure_2_iid_f1_vs_dimensionality.png) | ![Figure 3](file:///Users/pavanaksshay/quantum/paper/submission/figures/figure_3_quantum_minus_rbf_vs_dimensionality.png) |
| :---: | :---: |
| *Figure 2: In-distribution F1 vs Dimensionality ($d \in [2, 12]$)* | *Figure 3: Paired $\Delta\text{F1}$ with 95% Bootstrap CI vs $\pm 0.01$ Equivalence* |

---

### 3.2 Cross-Source Out-of-Distribution Domain Shift

When trained on TREC 2007 and evaluated out-of-domain on unseen TREC 2005/2006 enterprise emails across 10 seeds:
- **Classical RBF (8D):** Holdout $\text{F1} = 0.6913 \pm 0.0161$ ($20.6\%$ relative drop from IID)
- **Quantum Kernel (8D):** Holdout $\text{F1} = 0.6680 \pm 0.0094$ ($23.7\%$ relative drop from IID)
- **Statistical Significance:** $\Delta\text{F1} = -0.0233 \pm 0.0200$, 95% Bootstrap CI $[-0.0353, -0.0117]$, $p = 0.0046$, Benjamini-Hochberg FDR $p = 0.0069$.

| ![Figure 4](file:///Users/pavanaksshay/quantum/paper/submission/figures/figure_4_iid_vs_source_holdout.png) | ![Figure 6](file:///Users/pavanaksshay/quantum/paper/submission/figures/figure_6_representation_interaction.png) |
| :---: | :---: |
| *Figure 4: In-distribution vs Domain Holdout Degradation* | *Figure 6: Representation Interaction (TF-IDF vs RoBERTa)* |

---

### 3.3 Representation Primacy & Ranking Inversion

Switching from sparse n-gram TF-IDF to dense pretrained RoBERTa embeddings on CEAS 2008 inverts the relative ranking:
- **8D TF-IDF + TruncatedSVD:** Quantum $\text{F1} = 0.9736$ vs Classical RBF $\text{F1} = 0.9641$ ($+0.95$ pp quantum advantage).
- **8D Dense RoBERTa Embeddings:** Quantum $\text{F1} = 0.9601$ vs Classical RBF $\text{F1} = 0.9896$ ($-2.95$ pp classical advantage).
- **Net Shift:** **$3.90$ percentage points**. Upstream feature representations dominate kernel geometry by an order of magnitude.

| ![Figure 7](file:///Users/pavanaksshay/quantum/paper/submission/figures/figure_7_geometry_correlation_vs_dimensionality.png) | ![Figure 8](file:///Users/pavanaksshay/quantum/paper/submission/figures/figure_8_entropy_vs_kernel_diversity.png) |
| :---: | :---: |
| *Figure 7: Gram Matrix Pearson Correlation vs Dimensionality* | *Figure 8: Statevector Entropy vs Pairwise Gram Diversity* |

---

### 3.4 Computational Simulation Scaling Benchmark

| Qubits / Dim ($d$) | Quantum Sim Time (s) | Classical RBF Time (s) | Overhead Ratio | Quantum Peak RAM |
| :---: | :---: | :---: | :---: | :---: |
| **2D** | 3.4 s | 1.3 s | $2.6\times$ | 142 MB |
| **4D** | 5.8 s | 1.3 s | $4.5\times$ | 185 MB |
| **8D** | 17.2 s | 1.4 s | $12.3\times$ | 680 MB |
| **12D** | 108.8 s | 1.7 s | **$64.0\times$** | 6.4 GB |
| **16D** | Infeasible ($>10.5$ GB) | 2.1 s | $\infty$ | $>10.5$ GB (OOM) |

*Table 3: Measured wall-clock runtime and memory footprint on 10,000 text samples.*

---

## 4. Discussion & Scientific Conclusions

1. **Practical Equivalence:** Under matched conditions, quantum fidelity kernels do not provide a practical advantage over classical Gaussian RBF kernels for text security classification.
2. **Domain Shift Vulnerability:** Parameter-free quantum kernels suffer greater degradation under out-of-distribution domain shift than matched classical baselines.
3. **Representation Dominance:** Feature representation selection (TF-IDF vs RoBERTa) impacts model accuracy far more than the kernel function.
4. **Simulation Cost:** Incurring a $64\times$ computational overhead to achieve exact parity with a 1.7-second classical RBF kernel represents an unfavorable engineering trade-off.

---

## 5. Artifacts & File Locations

- **Full Compiled PDF:** [RESEARCH_PAPER.pdf](file:///Users/pavanaksshay/quantum/paper/RESEARCH_PAPER.pdf)
- **Interactive HTML Showcase:** [RESEARCH_PAPER.html](file:///Users/pavanaksshay/quantum/paper/RESEARCH_PAPER.html)
- **LaTeX Source:** [main.tex](file:///Users/pavanaksshay/quantum/paper/submission/main.tex)
- **BibTeX References:** [references.bib](file:///Users/pavanaksshay/quantum/paper/submission/references.bib)
- **Figures Directory:** [figures/](file:///Users/pavanaksshay/quantum/paper/submission/figures/)
