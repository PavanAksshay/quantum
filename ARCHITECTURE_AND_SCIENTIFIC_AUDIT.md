# Architecture and Scientific Audit

**Repository**: `PavanAksshay/quantum`  
**Date**: September 10, 2026  
**Auditor**: Antigravity Research Assistant  
**Standard**: Final Protocol V1.0 (Frozen) & Result Freeze Register  

---

## 1. Executive Summary & Audit Purpose
This audit provides a comprehensive structural and scientific review of the Quantum Text Security application prior to upgrading it into a modular, representation-first research laboratory. The paper's core empirical thesis is:

$$\text{Representation} \longrightarrow \text{Geometry} \longrightarrow \text{Kernel} \longrightarrow \text{Generalization} \longrightarrow \text{Computational Cost}$$

Under controlled matched conditions (10 independent seeds, 8D–12D, $\epsilon = 0.01\text{ F1}$ practical equivalence bound), parameter-free quantum fidelity kernels ($2$-layer cyclic $\text{ZZFeatureMap}$) achieve statistical parity with classical Gaussian RBF kernels, but provide **no consistent practical advantage**, experience **material degradation under cross-source domain shift**, and incur **exponential computational simulation penalties** at higher dimensions.

This audit establishes the baseline code components, identifies prohibited scientific language, reconciles runtime timing scopes, and partitions canonical results from new exploratory representation ablations.

---

## 2. Baseline Architecture Audit

### 2.1 Backend Architecture (`app/`)
- **`app/api.py`**: FastAPI application exposing endpoints for health, presets, live multi-model prediction, research metrics, dimensionality scaling, runtime profiling, geometry diagnostics, and table retrieval.
- **`app/model_engine.py`**: Monolithic class handling TF-IDF fitting, TruncatedSVD dimensionality reduction, StandardScaler standardization, phase-angle MinMax normalization to $[0, \pi]$, PyTorch `complex128` statevector simulation, fidelity kernel matrix multiplication, dual-model SVM classification (Quantum Kernel, Gaussian RBF, Linear SVM), and von Neumann entropy extraction.

### 2.2 Frontend Architecture (`frontend/src/`)
- **`frontend/src/App.jsx`**: React 19 application managing tab navigation (`prediction`, `dashboard`, `dimensionality`, `representation`, `generalization`, `geometry`, `runtime`, `tables`), asynchronous data fetching from port 8000, and Recharts rendering.
- **`frontend/src/index.css`**: Vanilla CSS styling with custom design tokens, clean light-mode surfaces, rounded borders, and typography (`Outfit`, `Inter`, `JetBrains Mono`).

---

## 3. Scientific Terminology & Conceptual Audit

The audit identified specific areas requiring scientific rigor and vocabulary corrections:

| Current / Prohibited Terminology | Scientific Reason for Correction | Approved Replacement Language |
| :--- | :--- | :--- |
| *"early low-dimensional weakness was caused by TruncatedSVD information loss"* | Causal claims without explicit causal intervention proofs are mathematically overextended. | *"Low-dimensional performance weakness is strongly associated with the information bottleneck introduced by aggressive TruncatedSVD compression."* |
| *"higher single-state entropy causes pairwise fidelities to concentrate"* | Correlation does not prove causal mechanism; entropy is a state dispersion metric. | *"Higher single-state entropy is strongly inversely associated with pairwise kernel diversity."* |
| *"Full Vocabulary Ceiling"* | "Ceiling" implies a theoretical maximum rather than an empirical linear baseline. | *"Full-dimensional TF-IDF baseline"* |
| *"Contextual Linear SVM"* | Standard TF-IDF + SVD LinearSVC is a bag-of-words / subword baseline, not intrinsically contextual. | *"Linear SVM baseline"* |
| *"adversarial robustness"* | The evaluation measures cross-source dataset transfer (TREC 2007 $\to$ 2005/06), not targeted adversarial perturbations. | *"Cross-source distribution shift"* |
| *"Malicious Probability: X%"* (for raw SVM) | Standard SVM decision boundaries output uncalibrated signed decision margins $f(\mathbf{x}) = \mathbf{w}^T \phi(\mathbf{x}) + b$, not calibrated posterior probabilities. | *"Decision Score: X"* or *"Estimated Probability (Sigmoid-mapped)"* |
| *"Quantum inference is 64× slower"* | Confuses single-sample interactive request latency with full $10{,}000$-sample quadratic Gram matrix construction. | *"Research benchmark runtime — 10,000 samples ($64\times$ classical simulation ratio)"* vs. *"Live request latency"* |

---

## 4. Runtime & Timing Scope Source of Truth

Audit of `results/exp39_paper/tables/table_8_runtime_scalability.csv` and `results/exp40_final/RUN_METADATA.md` establishes the authoritative timing scopes:

### 4.1 Research Benchmark Workload (10,000 Samples, Apple Silicon ARM64 CPU)
- **8D (8 Qubits)**:
  - Quantum Total: $12.95\text{ s}$ (Kernel Construction: $6.20\text{ s}$)
  - Classical RBF Total: $6.74\text{ s}$ (Kernel Construction: $\sim 0.45\text{ s}$)
  - Total Pipeline Ratio: $1.92\times$
- **10D (10 Qubits)**:
  - Quantum Total: $25.13\text{ s}$ (Kernel Construction: $18.50\text{ s}$)
  - Classical RBF Total: $6.49\text{ s}$ (Kernel Construction: $\sim 0.45\text{ s}$)
  - Total Pipeline Ratio: $3.88\times$
- **12D (12 Qubits)**:
  - Quantum Total: $85.44\text{ s}$ – $108.8\text{ s}$ (Kernel Construction: $78.90\text{ s}$)
  - Classical RBF Total: $6.55\text{ s}$ – $1.7\text{ s}$ (Kernel Construction: $\sim 0.45\text{ s}$)
  - Total Pipeline Ratio: $13.04\times$ – $64.0\times$ (Kernel Construction Ratio: $\sim 175\times$)
- **16D (16 Qubits)**:
  - Memory-infeasible ($>10.5\text{ GB}$ complex128 RAM ceiling for 10k statevectors).

### 4.2 Live Request Latency Scope (Single Message, 8D)
- Vectorization + Projection: $\sim 1.0 - 2.5\text{ ms}$
- PyTorch Statevector Contraction (Single test state vs. 500 support states): $\sim 12.0 - 25.0\text{ ms}$
- Classical RBF SVM Kernel Evaluation: $\sim 0.5 - 1.2\text{ ms}$
- Linear SVM Evaluation: $\sim 0.1 - 0.4\text{ ms}$

---

## 5. Result Partitioning & Experiment Hierarchy

To guarantee that new exploratory embeddings do not inadvertently alter the paper's frozen claims:

1. **CANONICAL (`results/exp39_paper/` & `results/exp40_final/`)**:
   - Primary Datasets: MeAJOR Archive, CEAS 2008, SMS Spam Collection.
   - Authoritative Models: Canonical 8D TF-IDF + SVD, 8D RoBERTa ablation on CEAS 2008, Dimensionality scaling (2D–12D), Cross-source holdout (Direction B).
   - Statistical Framework: 10 seeds, paired $\Delta\text{F1}$, 95% bootstrap CIs, permutation tests, Benjamini-Hochberg FDR correction.
2. **EXPLORATORY (`results/exploratory_representation/` & `EXPLORATORY_REPRESENTATION_RESULTS.md`)**:
   - Candidate Representations: `sentence-transformers/all-MiniLM-L6-v2` (384D), `sentence-transformers/all-mpnet-base-v2` (768D), Subword FastText (300D).
   - Rules: All models are frozen with zero fine-tuning; projection to $d$-dimensions is strictly standardized with TruncatedSVD + StandardScaler; missing combinations must explicitly display `NOT EVALUATED` or `UNAVAILABLE`. Never extrapolate or fabricate experimental cells.

---

## 6. Architectural Refactoring Blueprint

```
app/
├── api.py                            # FastAPI Routing & Endpoints
├── representations/                  # Modular Representation Subsystem
│   ├── __init__.py
│   ├── base.py                       # BaseRepresentation Abstract Class
│   ├── tfidf.py                      # Canonical 50k TF-IDF + SVD Representation
│   ├── roberta.py                    # Frozen RoBERTa-base (768D)
│   ├── minilm.py                     # Frozen MiniLM-L6-v2 (384D)
│   ├── mpnet.py                      # Frozen MPNet-base-v2 (768D)
│   ├── fasttext.py                   # Optional FastText Subword Representation
│   └── registry.py                   # Singleton Representation Registry & Cache
├── research/                         # Research Result Management
│   ├── __init__.py
│   ├── registry.py                   # Central Experiment Registry (CANONICAL vs EXPLORATORY)
│   ├── metrics.py                    # Immutable Authoritative Metrics Loader
│   ├── tables.py                     # Frozen CSV Table Parsers
│   └── exploratory.py                # Exploratory Data Store
├── model_engine/                     # Computational Kernel & Simulation
│   ├── __init__.py
│   ├── quantum.py                    # PyTorch complex128 ZZFeatureMap & Entropy Engine
│   ├── classical.py                  # Matched Gaussian RBF & Linear Baselines
│   ├── inference.py                  # Live Prediction Coordinator & Feature Tracing
│   ├── diagnostics.py                # Gram Matrix & Target Label Alignment
│   └── sampling.py                   # 2D Visualization Projection Sampler
├── schemas/                          # Pydantic Request & Response Schemas
│   ├── __init__.py
│   ├── prediction.py
│   ├── representation.py
│   └── research.py
└── model_engine.py                   # Backwards Compatibility Bridge
```
