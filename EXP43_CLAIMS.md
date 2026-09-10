# Experiment 43: Confirmatory Research Claims & Epistemic Boundaries

**Document Type**: Scientific Claims Register & Epistemic Protocol  
**Originating Experiment**: Experiment 43 (Representation-Aware Quantum Encoding Diagnostic Study)  
**Status**: FORMAL DIAGNOSTIC CLAIMS REGISTER  

---

## 1. SUPPORTED CLAIMS (Empirically Verified)

1. **Reproduction of Canonical MPNet Failure Baseline**:
   On SMS Spam with dense 8D MPNet sentence representations, the canonical 2-layer cyclic `ZZFeatureMap` fidelity kernel reproduces the severe performance deficit observed in Experiment 42 ($\text{Quantum F1} \approx 0.3756$ vs $\text{Classical RBF F1} \approx 0.9045$, $\Delta\text{F1} \approx -0.5288$, $0/10$ seed wins).
2. **Sensitivity to Angular Mapping Encodings**:
   Varying the angular transformation mapping from the canonical $[0, \pi]$ interval to symmetric $[-\pi, \pi]$, full circle $[0, 2\pi]$, or train-normalized standard Gaussian CDF mappings alters quantum Gram matrix statistics (kernel diversity, off-diagonal distribution, and target label alignment).
3. **Sensitivity to Feature-Map Depth Scaling**:
   Varying the repetition depth of the cyclic ZZFeatureMap ($D \in \{1, 2, 3\}$) while holding dimensionality (8D), representation (MPNet), and classifier ($C=1$, balanced SVC) fixed alters the statevector entanglement structure and resulting fidelity kernel geometry.
4. **Matched Classical Reference Performance**:
   Under strictly identical 8D SVD+Scaler features without test leakage, Classical RBF SVM consistently maintains high spam classification performance ($>0.90$ F1) across all 10 canonical seeds, demonstrating that the information necessary for high-accuracy discrimination remains present in the 8D representation.

---

## 2. CAUTIOUS CLAIMS (Contextually Qualified)

1. **Observational Geometry-Performance Relationships**:
   While higher target label alignment and healthy kernel diversity generally coincide with improved decision boundaries, geometric metrics alone do not serve as deterministic linear predictors of classification accuracy across all parameter-free encoding variations.
2. **Representation vs Encoding Interaction**:
   The failure of parameter-free cyclic ZZFeatureMaps on dense contrastive embeddings reflects an architectural mismatch with fixed sinusoidal phase encodings rather than an intrinsic deficiency of quantum state representations in general.
3. **Single-Dataset Domain Scope**:
   Because Experiment 43 strictly isolates SMS Spam to diagnose the MPNet failure, encoding sensitivity observed here may vary across other benchmark distributions with different semantic topologies.

---

## 3. CLAIMS THAT MUST NOT BE MADE (Strictly Prohibited Epistemic Overreaches)

| Prohibited Claim | Scientific Reason for Prohibition |
| :--- | :--- |
| **"Quantum Advantage" / "Quantum Supremacy"** | No quantum configuration achieved a statistically confirmed, practically meaningful advantage over Classical RBF. |
| **"Quantum Superiority / Inferiority in General"** | Encoding sensitivity is specific to the evaluated parameter-free ZZFeatureMap family on dense sentence geometry. |
| **"Universal MPNet Incompatibility"** | Dense sentence embeddings were tested only under static parameter-free feature maps; parameterized/trainable quantum circuits were not tested. |
| **"Universal Quantum Kernel Failure"** | Quantum fidelity kernels maintain near-perfect practical equivalence on sparse lexical TF-IDF (Exp 42: $+0.0033$). |
| **"Causal Proof that Diversity / Alignment Drives F1"** | Geometry metrics are observational Gram matrix descriptors, not experimentally isolated causal variables. |
| **"Quantum Computing is Inherently Slower/Faster"** | Benchmarks measure local classical CPU statevector simulation runtime, not fault-tolerant quantum hardware execution. |
| **"Adversarial Robustness Demonstrated"** | Experiment 43 is an encoding diagnostic on IID SMS Spam; adversarial/source-shift robustness was not evaluated. |
| **"Production Security Readiness"** | Exploratory research diagnostic; not intended as a production security deployment recommendation. |
