# Experiment 42: Confirmatory Research Claims & Epistemic Boundaries

**Document Type**: Scientific Claims Register & Epistemic Protocol  
**Originating Experiment**: Experiment 42 (10-Seed Confirmatory Protocol, $N=40$ conditions)  
**Status**: FORMAL CONFIRMATORY REGISTER  

---

## 1. SUPPORTED CLAIMS (Statistically Confirmed)

1. **Representation-Conditioned Kernel Divergence**:
   Switching between text representations (sparse lexical TF-IDF vs compact dense MiniLM vs contrastive MPNet) produces statistically detectable shifts in the relative performance ($\Delta\text{F1} = \text{Quantum F1} - \text{RBF F1}$) of parameter-free quantum fidelity kernels versus classical RBF kernels.
2. **Failure of Parameter-Free ZZFeatureMap on Dense Contrastive Representations**:
   On SMS Spam with dense contrastive sentence representations (`all-mpnet-base-v2`), the parameter-free 2-layer cyclic `ZZFeatureMap` fidelity kernel experiences severe geometric concentration (low kernel diversity and high off-diagonal concentration), yielding a statistically significant performance deficit relative to classical RBF ($\Delta\text{F1} < -0.15$, $p_{\text{BH}} < 0.05$).
3. **High-Diversity Sparse Lexical Parity on Email Security**:
   On CEAS 2008 with sparse lexical TF-IDF projected to 8D via TruncatedSVD, the parameter-free quantum fidelity kernel demonstrates high kernel diversity ($>0.30$) and maintains near-practical equivalence ($|\Delta\text{F1}| < 0.01$) with matched classical RBF kernels.
4. **Matched Dimensionality Control**:
   When upstream representations are matched at exactly 8 dimensions under strictly controlled preprocessing without test leakage, neither quantum fidelity kernels nor classical RBF kernels achieve a universal, cross-domain advantage over one another.

---

## 2. CAUTIOUS CLAIMS (Contextually Qualified)

1. **Model Rank Inversions**:
   On SMS Spam with MiniLM embeddings, the observed positive Q-vs-RBF difference is sensitive to seed variation and hyperparameter thresholding; while individual seeds exhibit rank reversals ($\text{Quantum} > \text{RBF}$), the aggregate 10-seed paired difference must be interpreted strictly within its empirical confidence interval.
2. **Geometric Diagnostics as Associations**:
   High kernel diversity and balanced label alignment correlate positively with competitive fidelity kernel performance, but geometric indicators represent observational associations rather than proven causal mechanisms.
3. **Task-Specific Representation Interaction**:
   The degree to which representation alters kernel ranking depends on the intrinsic vocabulary sparsity and semantic clustering of the underlying security domain.

---

## 3. CLAIMS THAT MUST NOT BE MADE (Prohibited Epistemic Overreaches)

| Prohibited Claim | Scientific Reason for Prohibition |
| :--- | :--- |
| **"Quantum Advantage" / "Quantum Supremacy"** | No statistically confirmed, practically meaningful ($|\Delta\text{F1}| \ge 0.01$) general quantum advantage was established across the benchmark suite. |
| **"Quantum Computing is Inherently Superior/Inferior"** | Kernel behavior is strictly representation- and dataset-dependent; no universal superiority exists. |
| **"Quantum Fidelity Kernels are Inherently Slower/Faster"** | Measured runtime differences reflect local classical CPU statevector simulation overhead ($O(B \cdot 2^n)$), not fault-tolerant quantum hardware execution. |
| **"MPNet Universally Causes Quantum Failure"** | The observed deficit is specific to the parameter-free 2-layer cyclic ZZFeatureMap with TruncatedSVD angle encodings; trainable or parameterized feature maps were not evaluated. |
| **"Representation Always Dominates Kernel Choice"** | While representation variation induces large performance swings, kernel choice remains decisive under specific geometric conditions. |
| **"Geometric Diversity Causes Performance"** | Kernel diversity and label alignment are statistical descriptors of the Gram matrix, not proven causal drivers. |
| **"Quantum Kernels Provide Adversarial Robustness"** | Predefined source-shift and domain-shift evaluations demonstrated no inherent robustness edge for quantum kernels. |
| **"Hardware Quantum Advantage Demonstrated"** | All experiments were conducted via exact PyTorch `complex128` classical simulation on Apple Silicon CPU. |
