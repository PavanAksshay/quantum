# Research at a Glance (Executive Summary for Professor Review)

**Manuscript Title**: *When Do Quantum Kernels Behave Differently for Text Security? A Multi-Dataset Evaluation of Representation, Geometry, Generalization, and Computational Cost*  
**Authors**: Anonymous Authors  
**Target Venue**: IEEE Transactions on Information Forensics and Security / ACM TOPS  
**Read Time**: 2–3 Minutes  

---

## 1. The Core Scientific Question
> **"Under strictly matched classical baselines, leakage-free data hygiene, and out-of-distribution domain shift, do parameter-free quantum fidelity kernels provide a consistent and practically meaningful advantage for text security classification?"**

---

## 2. Benchmark Architecture at a Glance

| Evaluation Dimension | Benchmark Specification |
| :--- | :--- |
| **Datasets Evaluated (3)** | **SMS Spam Collection** ($N=5,572$), **CEAS 2008 Email Phishing** ($N=15,000$), **MeAJOR Multi-Source Archive** ($N=108,684$). |
| **Classical ML Suite (8)** | **Linear SVM** (Primary Linear), **RBF SVM** (Primary Nonlinear Comparator), Logistic Regression, Multinomial Naive Bayes, Random Forest, XGBoost, MLP Neural Network, k-NN. |
| **Text Representations (4)** | **Sparse Lexical TF-IDF** (50,000 n-grams) vs. **Contextual Sentence Transformers** (all-MiniLM-L6-v2, RoBERTa-base, all-mpnet-base-v2). |
| **Quantum Kernel Method** | Parameter-free 2-layer cyclic **ZZFeatureMap** on $N_q \in [2, 12]$ qubits, evaluated via exact double-precision statevector fidelity inner products ($|\langle\psi(\mathbf{x})|\psi(\mathbf{z})\rangle|^2$). |
| **Statistical Protocol** | **10 Canonical Seeds** (`[42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]`), 10,000-sample paired permutation tests, 10,000-sample bootstrap 95% CIs, Benjamini-Hochberg FDR control. |
| **Practical Equivalence Threshold** | Pre-registered $\varepsilon = \pm 0.01$ F1 ($1.0$ percentage point). |

---

## 3. Key Empirical Findings

```
                                  IN-DISTRIBUTION (IID)
    +0.46 pp at 8D (p=0.0016)  ─────────►  PRACTICAL EQUIVALENCE (|Δ| <= 0.01 F1)
    +0.14 pp at 12D (p=0.2824) ─────────►  COMPLETE STATISTICAL PARITY

                              CROSS-SOURCE DOMAIN SHIFT
    TREC 2007 ──► TREC 2005/2006 ───────►  CLASSICAL RBF ADVANTAGE (Δ = -2.33 pp, p=0.0046)
                                           Quantum degrades by 23.7% vs 20.6% for RBF

                              UPSTREAM REPRESENTATION
    Sparse TF-IDF ──► Dense MPNet ──────►  CATASTROPHIC QUANTUM COLLAPSE (Δ = -52.88 pp)
                                           Metric clustering induces destructive phase wrapping
```

1. **In-Distribution Parity**: On MeAJOR 8D, Quantum achieves $\text{F1} = 0.8754 \pm 0.0029$ vs. Classical RBF $\text{F1} = 0.8709 \pm 0.0030$ ($\Delta = +0.0046$, $p = 0.0016$). At 12D, it converges to complete parity ($\Delta = +0.0014$, $p = 0.2824$). All in-distribution differences remain strictly within the practical equivalence zone ($\varepsilon = 0.01$).
2. **Domain Transfer Deficit**: Under cross-source holdout (TREC 2007 $\to$ TREC 2005/2006), the quantum kernel degrades more severely than classical RBF ($\Delta\text{F1} = -0.0233$, $p = 0.0046$, BH $p = 0.0069$).
3. **Representation Primacy**: Upstream embedding geometry dominates kernel choice. On dense contrastive MPNet embeddings, classical RBF achieves $0.9045$ F1 while the unparameterized quantum kernel collapses to $0.3756$ ($-52.88$ pp deficit) due to phase-wrapping across compact Euclidean clusters.
4. **Computational Cost**: Classical statevector simulation incurs steep scaling: 12-qubit simulation requires $108.8\text{s}$ per run vs. $1.7\text{s}$ for RBF ($64\times$ penalty), reaching memory infeasibility ($>10.5\text{ GB}$) at 16 qubits.

---

## 4. Methodological Justifications

### Why Linear SVM as Primary Linear Baseline?
On uncompressed 50,000-D sparse TF-IDF, **Linear SVM achieves dominant performance** across all datasets ($0.9559$ F1 on SMS, $0.9954$ on CEAS, $0.9721$ on MeAJOR) while executing in **$0.03\text{--}0.18\text{s}$** via deterministic primal coordinate descent. It matches or outperforms complex neural (MLP) and boosted tree (XGBoost) baselines at $1/50\text{th}$ to $1/500\text{th}$ the latency, providing the optimal linear reference.

### Why RBF SVM as Primary Nonlinear Comparator?
RBF SVM shares the **exact dual quadratic programming formulation and maximum-margin objective** with the Quantum Support Vector Classifier (QSVC). By holding the loss function, regularization parameter ($C=1.0$), and class balancing identical, substituting $K_{\text{RBF}}$ with $K_Q$ isolates differences strictly to **classical RKHS geometry vs. quantum Hilbert space geometry**.

### What Does This Study Add to the Literature?
Prior QML text and phishing studies (including Rahevar et al., CMES 2026 and the MDPI MAKE 2026 review) evaluate isolated datasets, single seeds, or small low-data samples. **0% of prior QML phishing studies evaluated out-of-distribution domain shift.** Our work provides the first multi-dataset benchmark systematically evaluating the **representation $\times$ geometry $\times$ domain-shift $\times$ computational-cost interaction** under 10-seed pre-registered equivalence testing.

---

## 5. Final Scientific Conclusion
> **Under controlled matched conditions, parameter-free quantum fidelity kernels do not provide a consistent, practically meaningful advantage over matched classical RBF kernels for text security.** Rather than treating QML as universally superior or inferior, this study demonstrates that quantum kernel utility is strictly conditional on representation geometry, dimensionality compression, domain stability, and computational budget.
