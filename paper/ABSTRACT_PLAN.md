# Manuscript Abstract Architecture and Structured Draft

**Document Version**: 1.0 (Frozen)  
**Date**: September 5, 2026  
**Scope**: Structured section-by-section abstract blueprint and verified numerical draft for the final paper.

---

## 1. Abstract Component Blueprint

| Component | Target Word Count | Core Content & Scientific Requirement |
| :--- | :---: | :--- |
| **1. Problem** | ~25 words | Text-based scam and phishing attacks impose massive economic costs; robust NLP detection remains critical under distribution shift. |
| **2. Gap** | ~35 words | Quantum kernel methods are hypothesized to provide advantages in high-dimensional feature spaces, but prior evaluations rely on small synthetic datasets, unvalidated splits, or unmatched classical baselines. |
| **3. Method** | ~35 words | We establish a leak-free benchmark comparing parameter-free cyclic $ZZFeatureMap$ quantum fidelity kernels against matched classical RBF kernels on identical low-dimensional representations. |
| **4. Datasets** | ~25 words | Evaluated across three frozen corpora totaling 152,000+ samples: SMS Spam Collection, CEAS 2008, and MeAJOR archive across 10 independent seeds. |
| **5. Main Results** | ~60 words | Under matched 8D in-distribution conditions, Quantum achieves $\text{F1} = 0.8754 \pm 0.0029$ vs Classical RBF $0.8709 \pm 0.0030$ ($\Delta = +0.0046$, 95% CI $[+0.0030, +0.0061]$). Monotonic scaling restores parity at 12D ($\Delta = +0.0014$, $p=0.282$). Under cross-source holdout, Quantum degrades significantly ($\text{F1} = 0.6680$ vs RBF $0.6913$, $\Delta = -0.0233$, $p=0.0046$). |
| **6. Interpretation** | ~35 words | Representation sparsity and dimensionality dominate classification outcomes over kernel selection, while exact quantum simulation incurs steep computational scaling ($64\times$ runtime penalty at 12D). |
| **7. Conclusion** | ~25 words | Under controlled matched conditions, quantum kernels provide no consistent, statistically superior, or practically meaningful advantage over classical RBF kernels. |

**Total Target Word Count**: **210–240 words**

---

## 2. Working Structured Draft (Verified Numbers Only)

> **Abstract**  
> Text-based social engineering attacks, including email phishing and SMS scams, continue to present severe security threats that degrade under real-world distribution shift. Quantum kernel methods have been hypothesized to offer representational advantages in complex feature spaces; however, prior quantum machine learning studies in natural language processing largely rely on small synthetic datasets, unvalidated splits, or unmatched classical baselines. 
> 
> In this work, we conduct a controlled empirical evaluation comparing parameter-free quantum fidelity kernels against matched classical radial basis function (RBF) kernels across three benchmark corpora comprising over 152,000 text samples: SMS Spam Collection, CEAS 2008, and the MeAJOR archive. Using frozen leakage-safe protocols and 10 independent random seeds, we evaluate in-distribution scaling ($2\text{--}12$ qubits), cross-source domain holdout, feature space geometry, and computational overhead. 
> 
> In-distribution at 8 dimensions, the quantum kernel achieves $\text{F1} = 0.8754 \pm 0.0029$, tracking classical RBF ($\text{F1} = 0.8709 \pm 0.0030$) within the predefined practical-equivalence boundary of $\pm 0.01$ F1. Performance scales monotonically with dimensionality, reaching complete parity at 12 dimensions ($\Delta \text{F1} = +0.0014$, $p = 0.2824$). Under cross-source domain transfer, the quantum kernel exhibits greater vulnerability, suffering a statistically significant deficit relative to RBF ($\text{F1} = 0.6680$ vs $0.6913$; $\Delta \text{F1} = -0.0233$, $p = 0.0046$). Geometric diagnostics reveal lower target label alignment, while classical simulation incurs a $64\times$ runtime penalty at 12 dimensions. 
> 
> Our results demonstrate that representation sparsity and dimensionality exert substantially larger effects than kernel substitution, and that evaluated quantum kernels provide no consistent practical advantage for security text classification.
