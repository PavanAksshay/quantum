# Evaluating a Parameter-Free Quantum Fidelity Kernel for Text Security Under Representation and Domain Shift

**Anonymous Authors**  
*Manuscript Prepared for Formal Academic Review*  

---

### Abstract
Quantum kernel methods have attracted substantial interest for natural language processing and cybersecurity, motivated by theoretical conjectures that mapping classical data into exponentially large Hilbert spaces could yield superior non-linear decision boundaries or enhanced generalization under distribution shift. However, empirical studies often rely on small sample sizes, unmatched classical baselines, or unvalidated split partitions. In this work, we present a controlled empirical evaluation examining an unparameterized quantum fidelity kernel (cyclic $ZZFeatureMap$) against matched classical radial basis function (RBF) kernels and an audited suite of eight classical machine learning baselines across three benchmark corpora comprising 153,410 usable text records (SMS Spam Collection, CEAS 2008 Email Corpus, and the multi-source MeAJOR archive). Using frozen, leakage-safe protocols and 10 independent random seeds, we evaluate in-distribution scaling ($2\text{--}12$ qubits), cross-source domain holdouts (TREC 2007 $\to$ TREC 2005/2006), feature space geometry, classical baseline suitability, and computational overhead.

Under matched in-distribution (IID) conditions, the quantum kernel is competitive with classical RBF, displaying minor statistically detectable improvements at intermediate dimensions ($+0.46$ percentage points at 8D, $p = 0.0016$; $+0.57$ pp at 10D, $p = 0.0052$) that remain strictly within the predefined practical-equivalence threshold ($\varepsilon = 0.01$ F1) under formal Two One-Sided Tests (TOST), converging to complete parity at 12D ($+0.14$ pp, $p = 0.2824$). When compared against a validation-tuned RBF baseline, the quantum margin narrows to parity across all dimensions. Under cross-source domain transfer, the quantum kernel exhibits a statistically significant performance deficit ($\Delta\text{F1} = -0.0233$, $p = 0.0046$, Benjamini–Hochberg adjusted $p = 0.0069$). Representation ablations demonstrate that upstream feature representation dominates kernel selection by an order of magnitude: switching from TF-IDF to dense sentence embeddings shifts relative performance by up to $52.88$ percentage points. Geometrically, quantum and RBF Gram matrices show moderate correlation ($r \approx 0.55\text{--}0.65$), while single-state entropy is strongly inversely associated with pairwise kernel diversity ($r = -0.78$ to $-0.83$). Computationally, classical statevector simulation of the quantum kernel requires $108.8\text{s}$ per run at 12D compared to $1.7\text{s}$ for RBF ($\approx 64\times$ penalty), with 16D exceeding local workstation memory limits ($>10.5\text{ GB}$). We conclude that under the evaluated conditions, parameter-free quantum fidelity kernels do not produce a consistent practical advantage for text security classification, and we outline rigorous benchmarking standards for future applied QML evaluations.

---

## 1. Executive Summary & Research at a Glance

```
========================================================================================
                                 RESEARCH AT A GLANCE
========================================================================================
Primary RQ:        Do parameter-free quantum kernels provide a practical advantage for text security?
Corpora (3):       SMS Spam (5.5k), CEAS 2008 Phishing (39.1k raw / 15k exp), MeAJOR Archive (108.6k)
Accounting:        153,413 Raw -> 153,410 Usable -> 35,572 Controlled Experimental Subsets
Classifiers (8):   Linear SVM, Matched RBF, Tuned RBF, Logistic Regression, Naive Bayes,
                   Random Forest, XGBoost, MLP Neural Net, k-NN
Quantum Setup:     2-Layer Cyclic ZZFeatureMap on 2-12 Qubits, Statevector Fidelity Kernel
Statistical Protocol: 10 Canonical Seeds, 10k Permutations, 10k Bootstrap CIs, TOST, BH-FDR
Equivalence Zone:  ε = ±0.01 F1 (Two One-Sided Tests practical equivalence boundary)

MAJOR SCIENTIFIC FINDINGS:
1. In-Distribution Parity:      +0.46 pp at 8D (p=0.0016) -> TOST Practically Equivalent (|Δ| <= 0.01)
                                +0.14 pp at 12D (p=0.2824) -> Statistical Parity
                                +0.12 pp vs Tuned RBF at 8D (p=0.1840) -> Parity
2. Domain Shift Vulnerability:  -2.33 pp deficit under cross-source holdout (p=0.0046, BH p=0.0069)
3. Representation Dominance:    Sentence transformers alter ranking by up to 52.88 pp
4. Classical Simulation Cost:   64x runtime overhead at 12D; >10.5 GB RAM limit at 16D
5. Baseline Selection:          Linear SVM is dominant linear model (0.95-0.99 F1 in 0.03-0.18s)
                                RBF SVM is exact matched dual QP nonlinear comparator
========================================================================================
```

---

## 2. Dataset Accounting & Stratification Tiers

To prevent ambiguity regarding dataset sizes and class distributions, four counting tiers are strictly maintained:

| Corpus | Raw Archived Records | Usable Cleaned Records | Full Source Positive % | Canonical Exp. Subset ($N$) | Exp. Split (Train / Val / Test) | Exp. Positive % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SMS Spam** | 5,574 | 5,572 | 13.41% | 5,572 | 3,343 / 1,114 / 1,115 | 13.41% |
| **CEAS 2008** | 39,154 | 39,154 | 55.78% | 15,000 | 10,000 / 2,500 / 2,500 | 18.90% |
| **MeAJOR (TREC 5/6/7)** | 108,685 | 108,684 | 44.20% | 15,000 | 10,000 / 2,500 / 2,500 | 19.33% |
| **Total** | **153,413** | **153,410** | — | **35,572** | **23,343 / 6,114 / 6,115** | — |

---

## 3. Seed Randomness & Statistical Equivalence Testing

### Randomness Controlled by 10 Computational Seeds
All primary experiments are evaluated across 10 frozen random seeds:
$$\mathcal{S} = \{42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021\}$$
Each seed controls:
1. **Stratified Partition Resampling**: Generating balanced train/validation/test index assignments.
2. **TruncatedSVD Initialization**: Random seed for projection matrix solver fitting.
3. **Stochastic Solvers**: Iterative weight initialization in MLP, Random Forest, and XGBoost.
4. **SVM Solver Optimization & Tie-Breaking**: Dual coordinate descent and threshold selection.

The unit of inference is the paired difference $\Delta\text{F1}(s) = \text{F1}_Q(s) - \text{F1}_{\text{RBF}}(s)$ across computational runs.

### Two One-Sided Tests (TOST) Formulation
We test practical equivalence relative to $\varepsilon = 0.01$ (1.0 percentage point F1):
$$H_0^-: \mu_\Delta \le -\varepsilon \quad \text{vs.} \quad H_1^-: \mu_\Delta > -\varepsilon$$
$$H_0^+: \mu_\Delta \ge +\varepsilon \quad \text{vs.} \quad H_1^+: \mu_\Delta < +\varepsilon$$
Rejection of both composite null hypotheses at $\alpha = 0.05$ demonstrates that $\mu_\Delta \in (-\varepsilon, +\varepsilon)$.

### Sensitivity Analysis:
- **$\varepsilon = 0.020$**: All IID differences at $d \ge 4$ lie strictly within practical equivalence.
- **$\varepsilon = 0.010$**: Differences at $d \in \{6, 8, 10, 12\}$ are practically equivalent.
- **$\varepsilon = 0.005$**: The 12D result ($[-0.0010, +0.0037] \subset [-0.005, +0.005]$) satisfies strict practical equivalence.

---

## 4. Complete Classical Baseline Audit (8 Models across 3 Corpora)

| Corpus | Model Architecture | Full TF-IDF F1 | Full PR-AUC | Full ROC-AUC | Matched 8D SVD F1 | 8D PR-AUC | 8D ROC-AUC | Train Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SMS Spam** | Linear SVM | $0.9559 \pm 0.000$ | $0.9822$ | $0.9931$ | $0.8273 \pm 0.0108$ | $0.8895$ | $0.9816$ | 0.03 s |
| | Matched RBF SVM | $0.9498 \pm 0.000$ | $0.9829$ | $0.9927$ | $0.8156 \pm 0.0119$ | $0.8767$ | $0.9793$ | 1.53 s |
| | Tuned RBF SVM | $0.9521 \pm 0.000$ | $0.9840$ | $0.9935$ | $0.8285 \pm 0.0105$ | $0.8912$ | $0.9820$ | 5.20 s |
| | Logistic Regression | $0.9346 \pm 0.000$ | $0.9803$ | $0.9944$ | $0.8226 \pm 0.0116$ | $0.8900$ | $0.9818$ | 0.09 s |
| | Random Forest | $0.9423 \pm 0.0047$ | $0.9780$ | $0.9915$ | $0.8500 \pm 0.0060$ | $0.9239$ | $0.9782$ | 0.90 s |
| | XGBoost | $0.9059 \pm 0.000$ | $0.9477$ | $0.9834$ | $0.8452 \pm 0.0132$ | $0.9276$ | $0.9822$ | 0.86 s |
| | MLP (Neural Net) | $0.9324 \pm 0.0083$ | $0.9781$ | $0.9909$ | $0.8284 \pm 0.0143$ | $0.8841$ | $0.9802$ | 4.68 s |
| | Multinomial / Gauss NB | $0.9158 \pm 0.000$ | $0.9644$ | $0.9889$ | $0.8195 \pm 0.0070$ | $0.8788$ | $0.9675$ | 0.01 s |
| | k-NN ($k=5$) | $0.6900 \pm 0.000$ | $0.5903$ | $0.7633$ | $0.8401 \pm 0.0187$ | $0.8802$ | $0.9506$ | 0.00 s |
| **CEAS 2008** | Linear SVM | $0.9954 \pm 0.000$ | $0.9999$ | $0.9999$ | $0.9535 \pm 0.0014$ | $0.9857$ | $0.9845$ | 0.15 s |
| | Matched RBF SVM | $0.9968 \pm 0.000$ | $0.9998$ | $0.9998$ | $0.9638 \pm 0.0006$ | $0.9882$ | $0.9873$ | 57.15 s |
| | Tuned RBF SVM | $0.9972 \pm 0.000$ | $0.9999$ | $0.9999$ | $0.9691 \pm 0.0005$ | $0.9910$ | $0.9902$ | 185.2 s |
| | Logistic Regression | $0.9935 \pm 0.000$ | $0.9996$ | $0.9994$ | $0.9505 \pm 0.0006$ | $0.9886$ | $0.9860$ | 0.26 s |
| | Random Forest | $0.9895 \pm 0.0009$ | $0.9995$ | $0.9993$ | $0.9816 \pm 0.0010$ | $0.9980$ | $0.9973$ | 1.76 s |
| | XGBoost | $0.9886 \pm 0.000$ | $0.9994$ | $0.9992$ | $0.9810 \pm 0.0011$ | $0.9978$ | $0.9970$ | 23.24 s |
| | MLP (Neural Net) | $0.9966 \pm 0.0003$ | $0.9999$ | $0.9999$ | $0.9614 \pm 0.0033$ | $0.9911$ | $0.9888$ | 46.28 s |
| | Multinomial / Gauss NB | $0.9928 \pm 0.000$ | $0.9994$ | $0.9992$ | $0.7165 \pm 0.000$ | $0.9249$ | $0.9091$ | 0.01 s |
| | k-NN ($k=5$) | $0.9957 \pm 0.000$ | $0.9978$ | $0.9986$ | $0.9823 \pm 0.0002$ | $0.9947$ | $0.9943$ | 0.02 s |
| **MeAJOR** | Linear SVM | $0.9721 \pm 0.000$ | $0.9964$ | $0.9972$ | $0.8448 \pm 0.0022$ | $0.9315$ | $0.9404$ | 0.18 s |
| | Matched RBF SVM | $0.9700 \pm 0.000$ | $0.9960$ | $0.9969$ | $0.8709 \pm 0.0030$ | $0.9421$ | $0.9535$ | 71.84 s |
| | Tuned RBF SVM | $0.9734 \pm 0.000$ | $0.9968$ | $0.9975$ | $0.8742 \pm 0.0028$ | $0.9450$ | $0.9560$ | 240.5 s |
| | Logistic Regression | $0.9574 \pm 0.000$ | $0.9935$ | $0.9949$ | $0.8458 \pm 0.0022$ | $0.9315$ | $0.9404$ | 0.35 s |
| | Random Forest | $0.9516 \pm 0.0020$ | $0.9897$ | $0.9922$ | $0.9014 \pm 0.0058$ | $0.9696$ | $0.9733$ | 2.16 s |
| | XGBoost | $0.9453 \pm 0.000$ | $0.9902$ | $0.9923$ | $0.9005 \pm 0.0048$ | $0.9687$ | $0.9725$ | 26.59 s |
| | MLP (Neural Net) | $0.9727 \pm 0.0026$ | $0.9965$ | $0.9973$ | $0.8795 \pm 0.0021$ | $0.9564$ | $0.9624$ | 42.12 s |
| | Multinomial / Gauss NB | $0.9481 \pm 0.000$ | $0.9915$ | $0.9924$ | $0.7258 \pm 0.0055$ | $0.7870$ | $0.8195$ | 0.01 s |
| | k-NN ($k=5$) | $0.9532 \pm 0.000$ | $0.9844$ | $0.9894$ | $0.8870 \pm 0.0025$ | $0.9459$ | $0.9581$ | 0.02 s |

---

## 5. Main Quantum vs. Classical Scaling (MeAJOR IID, 10 Seeds)

| Dim ($d$) | Quantum F1 | Classical RBF F1 | Tuned RBF F1 | Paired $\Delta$F1 (Q - Matched) | 95% Bootstrap CI | Permutation $p$ | TOST ($\varepsilon=0.01$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2D** | $0.6447 \pm 0.0038$ | $0.6735 \pm 0.0031$ | $0.6780 \pm 0.0029$ | $-0.0288$ | $[-0.0310, -0.0264]$ | $p < 0.001$ | Classical Superior |
| **4D** | $0.7876 \pm 0.0030$ | $0.7918 \pm 0.0025$ | $0.7954 \pm 0.0024$ | $-0.0042$ | $[-0.0058, -0.0024]$ | $p = 0.0180$ | Equivalent ($\varepsilon=0.01$) |
| **6D** | $0.8253 \pm 0.0026$ | $0.8254 \pm 0.0022$ | $0.8291 \pm 0.0021$ | $-0.0001$ | $[-0.0018, +0.0017]$ | $p = 0.9410$ | Equivalent ($\varepsilon=0.01$) |
| **8D** | $0.8754 \pm 0.0029$ | $0.8709 \pm 0.0030$ | $0.8742 \pm 0.0028$ | $+0.0046$ | $[+0.0030, +0.0061]$ | $p = 0.0016$ | Equivalent ($\varepsilon=0.01$) |
| **10D** | $0.9023 \pm 0.0034$ | $0.8967 \pm 0.0049$ | $0.9015 \pm 0.0038$ | $+0.0057$ | $[+0.0032, +0.0081]$ | $p = 0.0052$ | Equivalent ($\varepsilon=0.01$) |
| **12D** | $0.9137 \pm 0.0046$ | $0.9123 \pm 0.0023$ | $0.9148 \pm 0.0020$ | $+0.0014$ | $[-0.0010, +0.0037]$ | $p = 0.2824$ | Equivalent ($\varepsilon=0.005$) |

---

## 6. Upstream Representation Ranking Reversals

| Corpus | Representation | Quantum F1 | Classical RBF F1 | Paired Difference | Representation Effect |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **CEAS 2008 (8D)** | Sparse TF-IDF | 0.9736 | 0.9641 | $+0.0095$ (+0.95 pp) | Modest non-linear quantum expansion |
| | Dense RoBERTa | 0.9601 | 0.9896 | $-0.0295$ (-2.95 pp) | RBF exploits Euclidean continuous clustering |
| | **Net Shift** | **-1.35 pp** | **+2.55 pp** | **-3.90 pp Shift** | **Complete ranking inversion** |
| **SMS Spam (8D)** | Sparse TF-IDF | 0.6324 | 0.8276 | $-0.1952$ (-19.52 pp) | RBF superior on sparse projections |
| | Dense all-MiniLM | 0.7707 | 0.7930 | $-0.0223$ (-2.23 pp) | Moderate gap reduction |
| | Dense all-MPNet | 0.3756 | 0.9045 | $-0.5288$ (-52.88 pp) | Catastrophic phase-wrapping collapse |

---

## 7. Cross-Source Domain Shift Generalization (TREC 2007 $\to$ TREC 2005/2006)

| Model Architecture | In-Distribution F1 | Domain Holdout F1 | Absolute Drop | Relative Drop |
| :--- | :---: | :---: | :---: | :---: |
| **Linear SVM (8D)** | $0.8445 \pm 0.0022$ | $0.6622 \pm 0.0142$ | $-0.1823$ | -21.6% |
| **Classical RBF (8D)** | $0.8709 \pm 0.0030$ | $0.6913 \pm 0.0161$ | $-0.1796$ | -20.6% |
| **Quantum Kernel (8D)** | $0.8754 \pm 0.0029$ | $0.6680 \pm 0.0094$ | $-0.2074$ | -23.7% |

Paired Quantum vs. RBF Holdout Delta: $\Delta\text{F1} = -0.0233 \pm 0.0200$ (Permutation $p = 0.0046$, Benjamini–Hochberg adjusted $p = 0.0069$).

---

## 8. Feature Space Geometry & Statevector Dispersion

1. **Centered Kernel-Target Alignment (CKA)**:
   $$\text{CKA}(K, Y) = \frac{\langle H K H, H Y H \rangle_F}{\|H K H\|_F \|H Y H\|_F}$$
   Quantum kernel displays a $50\%\text{--}60\%$ alignment deficit relative to RBF ($\sim 0.022\text{--}0.040$ vs $\sim 0.060\text{--}0.077$).
2. **Effective Rank ($R_{\text{eff}}$)**:
   $$R_{\text{eff}}(K) = \exp\left( -\sum_{i=1}^N \tilde{\lambda}_i \ln \tilde{\lambda}_i \right)$$
   MeAJOR 8D: Quantum $R_{\text{eff}} = 14.2$ vs Classical RBF $R_{\text{eff}} = 18.6$.
3. **Entropy-Diversity Decoupling**:
   Strong inverse association between single-state von Neumann entropy and pairwise Gram diversity ($r = -0.8257$ on SMS, $-0.8170$ on CEAS, $-0.7822$ on MeAJOR).

---

## 9. Computational Simulation Scaling on 10,000 Samples

| Dimension ($d$) | Quantum Time (s) | Classical RBF Time (s) | Runtime Ratio | Peak Quantum RAM | Feasibility Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2 Qubits** | 3.4 s | 1.3 s | $2.6\times$ | 142 MB | Fully Feasible |
| **4 Qubits** | 5.8 s | 1.3 s | $4.5\times$ | 185 MB | Fully Feasible |
| **8 Qubits** | 17.2 s | 1.4 s | $12.3\times$ | 680 MB | Fully Feasible |
| **12 Qubits** | 108.8 s | 1.7 s | $64.0\times$ | 6.4 GB | Heavy Simulation |
| **16 Qubits** | Infeasible | 2.1 s | — | $>10.5$ GB | Out-of-Memory Boundary |

---

## 10. Reproducibility & Provenance

- **Workstation**: Apple Silicon ARM64, macOS Darwin 25.6.0, 16 GB Unified Memory.
- **Environment**: Python 3.12.4, PyTorch 2.13.0 (`complex128` exact statevectors), scikit-learn 1.9.0, NumPy 2.5.2, SciPy 1.18.1, pandas 2.3.3.
- **Repository Commit**: `772017a`
- **Confirmation Command**:
  ```bash
  python3 experiments/40_confirmation_experiments.py
  ```
