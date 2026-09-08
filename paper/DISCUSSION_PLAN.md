# Discussion Section Architecture and Subsection Blueprint

**Document Version**: 1.0 (Frozen)  
**Date**: September 5, 2026  
**Scope**: Ten structured discussion subsections providing deep scientific interpretation, contextualization with theory, and disciplined boundary setting.

---

## 6.1 What the Empirical Results Establish
- Under matched low-to-intermediate dimensional representations ($d \le 12$), parameter-free quantum fidelity kernels act as valid nonlinear classifiers that closely track classical Gaussian RBF kernels.
- In-distribution performance scales monotonically with qubit/feature dimensionality, reaching practical equivalence ($|\Delta \text{F1}| \le 0.01$) with classical RBF at 12 dimensions.
- Under cross-source domain shift (TREC 2007 $\to$ TREC 2005/2006), the evaluated quantum kernel suffers a statistically significant disadvantage relative to classical RBF ($\Delta \text{F1} = -0.0233$, $p=0.0046$).
- Classical simulation of quantum kernels incurs steep computational scaling ($64\times$ execution time penalty at 12 qubits) for identical classification accuracy.

---

## 6.2 What the Empirical Results Do Not Establish
- Our findings do **not** establish a universal mathematical proof that quantum kernels cannot outperform classical models in all NLP settings.
- Our findings do **not** evaluate physical quantum computing hardware noise, quantum error mitigation, or fault-tolerant execution.
- Our findings do **not** evaluate parameterized quantum kernel training (QKT) or variational quantum circuits (VQC) with learned ansatz parameters.
- Our findings do **not** generalize beyond the evaluated text security datasets and the 2-layer cyclic $ZZFeatureMap$ family.

---

## 6.3 Representation Dominance vs Kernel Substitution
- Switching from dense RoBERTa embeddings to sparse TF-IDF inverted the quantum margin by $3.90$ pp on CEAS, while scaling PCA from 2D to 12D produced a $+26.9$ pp gain.
- In contrast, substituting classical RBF with the quantum kernel at matched dimensionality shifted performance by only $+0.14$ to $+0.57$ pp in IID settings.
- *Core Scientific Takeaway*: Text representation design, feature sparsity, and dimensionality reduction dominate classification outcomes by orders of magnitude over the mathematical choice of kernel function.

---

## 6.4 Dimensionality as an Information Bottleneck
- The severe underperformance of quantum models at 2D–4D ($0.6447\text{--}0.7876$ F1) was primarily an artifact of aggressive linear PCA compression from 50,000 TF-IDF features to 2–4 dimensions.
- Monotonic recovery up to 12D ($0.9137$ F1) proves that the quantum kernel is fully capable of utilizing additional feature dimensions when provided, mirroring classical RBF capacity.

---

## 6.5 Quantum vs Classical Kernel Geometry
- Pearson correlation between quantum and RBF Gram entries remains moderate-to-strong ($r = 0.55\text{--}0.65$), indicating that the cyclic $ZZFeatureMap$ produces distance metrics that partially approximate Gaussian radial decay in intermediate dimensions.
- However, target label alignment of the quantum kernel is $50\%\text{--}60\%$ lower than classical RBF across all three benchmark corpora, explaining why quantum models fail to exceed classical performance.

---

## 6.6 Generalization Under Domain and Source Shift
- Hypotheses that quantum Hilbert spaces provide inherent robustness against distribution shift are not supported by the empirical data.
- The quantum fidelity kernel degraded by $23.7\%$ under Direction B transfer, compared to $20.6\%$ for classical RBF.
- Without explicit regularization or domain-adaptation training, quantum kernel mappings remain vulnerable to out-of-distribution lexical drift across distinct email sources.

---

## 6.7 Computational Trade-offs and Simulation Overhead
- In practical classical simulation workloads, exact quantum statevector evaluation scales rapidly with dimensionality, requiring $108.8\text{s}$ per run at 12D compared to $1.7\text{s}$ for classical RBF ($64\times$ penalty).
- For classical simulation users, evaluating quantum kernels on large datasets ($N > 10,000$) introduces substantial computational and memory burdens without delivering measurable predictive gains.

---

## 6.8 Implications for Applied QML Benchmarking
- Applied QML research must adopt standardized benchmarking rigor:
  1. Identical, leak-free input representations for quantum and classical baselines.
  2. Predefined practical-equivalence regions ($\varepsilon$) to prevent overclaiming small numerical margins.
  3. Multi-seed replications with formal non-parametric inferential testing.
  4. Mandatory reporting of classical simulation runtimes and memory allocations.

---

## 6.9 Implications for Security-Oriented NLP
- For practical text-based scam and phishing detection systems, classical linear classifiers on high-dimensional representations (50k TF-IDF: F1 = $0.9721$) or fine-tuned transformer backbones (RoBERTa: F1 = $0.9841$) drastically outperform low-dimensional nonlinear models (F1 $\approx 0.87\text{--}0.91$).
- Constraining feature dimensionality to accommodate NISQ-era qubit counts creates a severe security penalty that nonlinear quantum mappings cannot overcome.

---

## 6.10 Relation to Previous Theoretical Findings
- Our empirical observations are **consistent with** theoretical results on quantum kernel concentration and expressivity bounds [Thanasilp et al., 2022; Kübler et al., 2021; Huang et al., 2021].
- As dimensionality increases, generic parameter-free quantum feature maps tend to exhibit geometric concentration that prevents them from generating superior inductive biases on unstructured classical data unless problem-specific geometric symmetry is embedded into the ansatz.
- *Tone Guardrail*: We do not claim our empirical benchmark "proves" these theorems; we state our observations are *consistent with* their theoretical predictions.
