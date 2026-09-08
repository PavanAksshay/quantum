# Limitations and Threat-to-Validity Architecture

**Document Version**: 1.0 (Frozen)  
**Date**: September 5, 2026  
**Scope**: Transparent, comprehensive catalog of study limitations, threat analysis, and why the core benchmark conclusions remain fully intact.

---

## 1. Explicit Catalog of Scientific Limitations

1. **Classical Simulation Engine (Noiseless Statevectors)**:
   - All quantum kernels were computed using exact, double-precision (`complex128`) PyTorch statevector simulations without physical quantum device noise (decoherence, gate infidelity, readout error).
   - *Impact*: Noise-free simulation represents the theoretical upper bound of algorithmic quantum fidelity performance. Physical NISQ device execution would introduce stochastic error and shot noise that could further degrade performance.

2. **Absence of Physical QPU Execution**:
   - The study does not evaluate physical superconducting or trapped-ion QPUs due to sample size ($15,000$ samples per dataset would require $\sim 112.5$ million circuit executions).
   - *Impact*: While physical runtime scaling differs from classical simulation, the mathematical fidelity values evaluated here represent the exact target states NISQ algorithms attempt to estimate.

3. **Feature Map Scope (Parameter-Free Cyclic $ZZFeatureMap$)**:
   - The primary evaluation focused on the canonical 2-layer cyclic $ZZFeatureMap$ without variational quantum kernel training (QKT) or data re-uploading architectures.
   - *Impact*: While alternative parameterized ansatzes exist, the cyclic $ZZFeatureMap$ is the standard reference architecture in literature; our ablation in Exp 28 confirmed that alternative entanglement topologies did not alter the fundamental gap.

4. **Linear Dimensionality Reduction Bottleneck (PCA/SVD)**:
   - Compressing 50,000 TF-IDF features to 2–12 dimensions was performed using linear `TruncatedSVD`.
   - *Impact*: Nonlinear autoencoders or learned embeddings might preserve more semantic information at 8D–12D, but linear PCA was applied identically to quantum and classical baselines, ensuring strict experimental control.

5. **Finite Qubit / Dimensionality Range ($d \le 16$)**:
   - Empirical scaling was evaluated from 2 to 12 qubits (with 16D evaluated classically), bounded by statevector memory limits ($>10.5\text{ GB}$ at 16 qubits on 10k samples).
   - *Impact*: The trajectory across 2D, 4D, 6D, 8D, 10D, and 12D demonstrated clear monotonic recovery and plateauing at parity with RBF at 12D.

6. **Historical Benchmark Corpora**:
   - Datasets evaluated (SMS Spam, CEAS 2008, MeAJOR 2005–2007) represent standardized historical archives rather than live, present-day telemetry.
   - *Impact*: These corpora provide established baselines with documented lexical complexity and multi-source structures, enabling reproducible benchmarking.

7. **Scope of Threat Model (No Dynamic Adversarial Attacks)**:
   - Evaluation focused on natural distribution shift and cross-source transfer rather than adaptive adversarial attacks (e.g., TextFooler, character perturbation, or prompt injection).

8. **Scope of Generalization Boundary**:
   - Conclusions are strictly scoped to the evaluated text classification corpora, representations, and feature-map families, rather than serving as a universal disproof of all quantum computing for NLP.

---

## 2. Why Limitations Do Not Invalidate Core Benchmark Conclusions

| Limitation | Why Core Conclusions Remain Valid |
| :--- | :--- |
| **Noiseless Simulation** | If a quantum kernel fails to beat classical RBF under ideal, noise-free conditions, introducing physical device noise will not create an advantage. |
| **PCA Compression** | Both quantum and classical RBF models received the exact same PCA inputs; the comparison isolates kernel expressivity under matched conditions. |
| **Single Feature Map Family** | The cyclic $ZZFeatureMap$ is the standard benchmark in prior literature; testing it under strict controls resolves existing conflicting claims. |
| **12 Qubit Ceiling** | The empirical difference $\Delta \text{F1}$ narrowed to $+0.0014$ ($p=0.282$) at 12D, demonstrating that parity had already been reached. |
| **10 Independent Seeds** | Expanded in Exp40 to $N=10$ independent seeds with 10,000-sample bootstrap CIs and permutation tests, ensuring robust inferential power. |
