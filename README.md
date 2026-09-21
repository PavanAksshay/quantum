---
title: Quantum Text Security Research Platform
emoji: ⚛️
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: server.py
pinned: false
---

# ⚛️ When Do Quantum Kernels Help for Text Security?
### *A Multi-Dataset Evaluation of Representation, Geometry, Generalization, and Computational Cost*

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel%20Frontend-blue?style=for-the-badge&logo=vercel)](https://quantum-three-hazel.vercel.app)
[![API Status](https://img.shields.io/badge/API%20Backend-FastAPI%20%2B%20PyTorch-green?style=for-the-badge&logo=fastapi)](https://quantum-backend.onrender.com/docs)
[![Paper PDF](https://img.shields.io/badge/Paper-10--Page%20Preprint%20PDF-red?style=for-the-badge&logo=adobeacrobatreader)](results/exp47/SAMPLE_PAPER.pdf)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

---

## 🔬 Research at a Glance

This repository contains the complete experimental code, statistical verification pipelines, academic manuscript sources, and an interactive full-stack research platform for evaluating **parameter-free quantum fidelity kernels** against **matched classical radial basis function (RBF) kernels** and **eight classical machine learning baselines** on real-world text security threats (phishing, email fraud, and SMS spam).

```
====================================================================================================
                                      RESEARCH SPECIFICATION MATRIX
====================================================================================================
Primary Research Question:   Do quantum kernels provide a practical advantage for text security?
Literature Scope:            28 Peer-Reviewed & Foundational Papers Synthesized and Audited
Benchmark Corpora (3):       SMS Spam (5.5k), CEAS 2008 Phishing (15k), MeAJOR Multi-Source (108k texts)
Classifiers Audited (8):     Linear SVM, Classical RBF SVM, Logistic Regression, Multinomial NB,
                             Random Forest, XGBoost, Multi-Layer Perceptron (MLP), k-NN
Quantum Setup:               2-Layer Cyclic ZZFeatureMap on 2–12 Qubits, Statevector Fidelity Kernel
Statistical Protocol:        10 Canonical Seeds, 10,000 Permutations, 10,000 Bootstrap CIs, BH-FDR
Equivalence Boundary:        ε = ±0.01 F1 (Pre-registered practical equivalence boundary)
====================================================================================================
```

---

## 📊 Major Scientific Findings

1. **In-Distribution Practical Equivalence ($\varepsilon = \pm 0.01\text{ F1}$)**:
   - At intermediate dimensions ($8\text{D}$ and $10\text{D}$), the quantum fidelity kernel achieves minor statistically detectable gains ($+0.46\text{ pp}$ at $8\text{D}$, $p = 0.0016$; $+0.57\text{ pp}$ at $10\text{D}$, $p = 0.0052$) that remain strictly within the predefined practical equivalence boundary ($|\Delta| \le 0.01$).
   - At $12\text{D}$, quantum and classical RBF converge to full statistical parity ($+0.14\text{ pp}$, $p = 0.2824$).

2. **Representation Primacy & Ranking Inversion**:
   - Upstream feature representation dominates kernel selection by an order of magnitude.
   - On **CEAS 2008**, switching from sparse TF-IDF to dense RoBERTa embeddings reverses the quantum advantage into a classical advantage (net shift: $-3.90\text{ pp}$).
   - On **SMS Spam**, contrastive sentence embeddings (**all-mpnet-base-v2**) cause a **$-52.88\text{ pp}$ catastrophic quantum collapse** due to phase-wrapping under cyclic Pauli-$Z$ gates.

3. **Cross-Source Domain Transfer Deficit**:
   - Under cross-source domain shift (TREC 2007 $\to$ TREC 2005/2006 holdouts), the quantum kernel suffers greater degradation than matched classical RBF ($\Delta\text{F1} = -0.0233$, $p = 0.0046$, Benjamini–Hochberg FDR $p = 0.0069$).

4. **Classical Statevector Simulation Overhead**:
   - Simulating the 12-qubit fidelity kernel requires **$108.8\text{s}$** per run versus **$1.7\text{s}$** for classical RBF ($\approx 64\times$ penalty), with memory exceeding $10.5\text{ GB}$ at $16\text{D}$.

5. **Empirical Baseline Selection Hierarchy**:
   - **Linear SVM** is the dominant linear baseline ($0.95\text{--}0.99\text{ F1}$ on full TF-IDF in $0.03\text{--}0.18\text{s}$).
   - **RBF SVM** is the exact matched nonlinear comparator (identical dual QP solver, $C=1.0$, balanced class weighting), isolating RKHS vs. Hilbert space geometry.

---

## 🏛️ Tripartite Architecture & Methodology

```
                               THE TRIPARTITE BASELINE ARCHITECTURE
                               
     Input Text Sample (x)
              │
              ├──► Linear SVM (Primal Coordinate Descent)  ────► Tests Linear Separability
              │
              ├──► Classical RBF SVM (Gaussian Dual QP)   ────► Tests Classical RKHS Geometry
              │
              └──► QSVC (ZZFeatureMap Fidelity Dual QP)    ────► Tests Quantum Hilbert Geometry
```

### Mathematical Formulation
- **Quantum Fidelity Kernel**:
  $$\mathcal{U}_{\Phi(\mathbf{x})} = \left( U_{\Phi(\mathbf{x})} H^{\otimes N_q} \right)^2$$
  $$U_{\Phi(\mathbf{x})} = \exp\Biggl( i \sum_{j=1}^{N_q} x_j Z_j + i \sum_{j=1}^{N_q} (\pi - x_j)(\pi - x_{j'}) Z_j Z_{j'} \Biggr)$$
  $$K_Q(\mathbf{x}, \mathbf{z}) = |\langle \psi(\mathbf{x}) | \psi(\mathbf{z}) \rangle|^2$$
- **Matched Classical RBF Kernel**:
  $$K_{\text{RBF}}(\mathbf{x}, \mathbf{z}) = \exp\left( -\gamma \|\mathbf{x} - \mathbf{z}\|_2^2 \right), \quad \gamma = \frac{1}{d \cdot \text{Var}(X)}$$
- **Dual Support Vector Classification**:
  $$\max_{\boldsymbol{\alpha}} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) \quad \text{s.t.} \quad 0 \le \alpha_i \le C \cdot w_{y_i}, \; \sum_{i=1}^N \alpha_i y_i = 0$$

---

## 📚 28-Paper Systematic Literature Taxonomy

The experimental benchmark is positioned against **28 referenced foundational, theoretical, and empirical works**:

```
+---------------------------------------------------------------------------------------------------+
|                                28-PAPER SYSTEMATIC LITERATURE TAXONOMY                            |
+---------------------------------+---------------------------------+-------------------------------+
|  A. Theory & Geometry (6)       |  B. QNLP & Text (6)             |  C. Security & Phishing (6)   |
|  1. Havlíček et al. (Nature 19) |  7. Rahevar et al. (CMES 2026)  | 13. Ammar et al. (MAKE 2026)  |
|  2. Schuld & Killoran (PRL 19)  |  8. Garg et al. (IEEE TQE 2024) | 14. Guddanti et al. (2026)    |
|  3. Huang et al. (Nat Comm 21)  |  9. Shukla et al. (Access 2023) | 15. Hridi et al. (QPAIN 2026) |
|  4. Thanasilp et al. (Nat Comm) | 10. Di Sipio et al. (TQE 2022)  | 16. Shahriyar et al. (2025)   |
|  5. Kübler et al. (NeurIPS 21)  | 11. Coecke et al. (2020)        | 17. Al-Sarem et al. (2023)    |
|  6. Glick et al. (npj QI 2022)  | 12. Lorenzo et al. (QST 2023)   | 18. Lu et al. (PR Research 20)|
+---------------------------------+---------------------------------+-------------------------------+
|  D. Benchmarking Rigor (5)      |  E. Text Representations & Security Benchmarks (5)              |
| 19. Li et al. (2026)            | 24. Devlin et al. (BERT 2019)                                   |
| 20. Bowles et al. (PRL 2024)    | 25. Reimers & Gurevych (Sentence-BERT 2019)                    |
| 21. Meyer et al. (Quantum 2023) | 26. Song et al. (MPNet 2020)                                    |
| 22. Shaydulin & Wild (TQE 2022) | 27. Al-Sallami et al. (ACM TOPS 2023)                           |
| 23. Abbas et al. (Nat CS 2021)  | 28. Cortes et al. (JMLR 2012)                                   |
+---------------------------------+-----------------------------------------------------------------+
```

---

## 📁 Repository Structure

```
quantum/
├── app/
│   ├── api.py                     # FastAPI backend REST service
│   ├── inference.py               # Live model inference (QSVC, RBF, Linear)
│   └── research/
│       ├── registry.py            # Experiment and representation query registries
│       └── research_store.py      # Authoritative frozen experimental data store
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── RepresentationLab.jsx     # Encoder screening & geometry matrix
│   │   │   ├── Exp41ScreeningView.jsx    # Multi-dataset screening dashboard
│   │   │   ├── ModelHierarchyExplainer.jsx # Tripartite architecture visualizer
│   │   │   └── TextRepresentationInspector.jsx # Real-time vector inspector
│   │   └── data/
│   │       └── researchFallbackData.js   # 100% offline CDN research fallback
│   └── package.json
├── experiments/
│   ├── 39_multi_dataset_eval.py   # Multi-dataset 10-seed in-distribution sweep
│   ├── 40_confirmation_experiments.py # Authoritative confirmation experiments
│   ├── 41_representation_screening.py # Upstream text representation sweeps
│   ├── 45_classical_baseline_audit.py # 480-run 8-classifier classical audit
│   └── 46_manuscript_update.py    # Submission package generator
├── paper/
│   ├── submission/
│   │   ├── main.tex               # LaTeX research manuscript source
│   │   ├── supplementary.tex      # Supplementary materials & mathematical proofs
│   │   ├── references.bib         # 28-paper comprehensive bibliography
│   │   └── figures/               # High-resolution vector PDF figures
├── results/
│   └── exp47/
│       ├── SAMPLE_PAPER.pdf       # 10-page complete compiled PDF
│       ├── SAMPLE_PAPER.tex       # Standalone paper LaTeX source
│       ├── LITERATURE_GAP_TABLE.csv
│       ├── MODEL_COMPARISON_TABLE.csv
│       └── CLAIM_EVIDENCE_MATRIX.csv
├── server.py                      # Gradio + FastAPI deployment entrypoint
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start & Reproduction

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/PavanAksshay/quantum.git
cd quantum

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full-Stack Research Platform Locally

**Start the FastAPI Backend:**
```bash
uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```

**Start the React Frontend:**
```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` to explore the interactive dashboard.

### 3. Run Confirmation Experiments
```bash
# Execute 10-seed confirmation sweep (Exp 40)
python3 experiments/40_confirmation_experiments.py

# Execute classical baseline audit across 8 models (Exp 45)
python3 experiments/45_classical_baseline_audit.py
```

### 4. Compile the Academic Paper PDF
```bash
tectonic results/exp47/SAMPLE_PAPER.tex
```

---

## 📖 Citation

If you use this benchmark, experimental protocols, or dataset partitions in your research, please cite:

```bibtex
@article{anonymous2026quantumtextsecurity,
  title={When Do Quantum Kernels Help for Text Security? A Multi-Dataset Evaluation of Representation, Geometry, Generalization, and Computational Cost},
  author={Anonymous Authors},
  journal={Archival Research Preprint},
  year={2026},
  note={Under Peer Review}
}
```

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
