# Paper Novelty Positioning and Formal Statements

**Document Version**: 1.0 (Frozen)  
**Date**: September 5, 2026  
**Scope**: Definitive framing of the paper's scientific positioning across multiple dissemination formats.

---

## Core Positioning Strategy

Prior literature has proposed quantum kernel methods, explored preliminary QML applications in cybersecurity, and established theoretical bounds on kernel expressivity. However, prior empirical NLP evaluations have frequently suffered from severe experimental confounds: small synthetic datasets, unvalidated or leaky splits, unmatched classical baselines, absence of domain-shift testing, and omission of exact simulation overhead. 

This paper does **not** claim to introduce a new quantum algorithm or an optimized feature map. Instead, our scientific contribution lies in delivering a **rigorously controlled, multi-corpus empirical study** that systematically isolates the effects of text representation, feature dimensionality, kernel geometry, cross-source domain shift, and simulation computational cost under strictly matched quantum and classical conditions.

---

## Version A: One-Sentence Novelty Statement
> *"This study provides a controlled, multi-dataset empirical evaluation of quantum fidelity kernels against matched classical RBF baselines for text security, demonstrating that representation dimensionality and sparsity dominate classification outcomes while quantum kernels achieve in-distribution parity but fail to provide domain-shift robustness or practical advantage."*

---

## Version B: Abstract-Length Novelty Statement
> *"While quantum kernel methods are theoretically hypothesized to offer advantages in high-dimensional feature spaces, empirical evidence in natural language security classification remains fragmented by small-sample regimes and unmatched baselines. Rather than proposing a new quantum architecture, this work provides a comprehensive, leak-free benchmark evaluating parameter-free quantum fidelity kernels across three email and SMS corpora (152,000+ samples) under identical low-dimensional representations. By systematically testing in-distribution scaling (2–12 qubits), cross-source domain holdout, feature space geometry, and computational cost across 10 independent seeds, we demonstrate that quantum kernels achieve practical equivalence with classical RBF kernels in-distribution ($|\Delta \text{F1}| \le 0.01$) but incur a statistically significant disadvantage under domain transfer ($\Delta \text{F1} = -0.0233$, $p=0.0046$), showing that text representation design dominates kernel substitution."*

---

## Version C: Reviewer-Response Novelty Statement
> *"We thank the reviewer for raising the question of novelty and algorithmic contribution. We wish to clarify that the primary objective and scientific value of this paper is not the introduction of a new quantum circuit or feature map, but rather the rigorous empirical resolution of an open, contested question in applied QML: under strictly matched, leak-free conditions, do quantum kernels provide an empirical advantage over classical RBF baselines for security text classification?*
>
> *Existing literature in security QML frequently reports quantum advantages on small, synthetic, or unvalidated datasets without controlling for feature representation or isolating PCA compression bottlenecks. In contrast, our work establishes: (1) a multi-corpus, leak-free benchmark with 10-seed statistical rigor; (2) empirical proof that low-dimensional quantum weakness is an information bottleneck rather than a kernel failure; (3) the first multi-source domain holdout evaluation disproving hypothesized quantum robustness advantages in text classification; and (4) geometric and computational cost analyses explaining why quantum kernels mirror RBF geometry while incurring a $64\times$ classical simulation penalty at 12 qubits. This provides the community with an authoritative, reproducible baseline and a negative/conditional finding critical for realistic QML roadmapping."*
