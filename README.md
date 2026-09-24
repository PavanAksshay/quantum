# ⚛️ Evaluating a Parameter-Free Quantum Fidelity Kernel for Text Security
### *Under Representation, Geometry, Generalization, and Computational Cost*

[![Live Demo](https://img.shields.io/badge/Live%20Frontend-Vercel-blue?style=for-the-badge&logo=vercel)](https://quantum-three-hazel.vercel.app)
[![API Backend](https://img.shields.io/badge/API%20Backend-Render%20(FastAPI)-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://quantum-backend-uxm8.onrender.com/docs)
[![API Health](https://img.shields.io/badge/API%20Health-Online-success?style=for-the-badge)](https://quantum-backend-uxm8.onrender.com/api/health)
[![Paper PDF](https://img.shields.io/badge/Paper-Preprint%20PDF-red?style=for-the-badge&logo=adobeacrobatreader)](results/exp47/SAMPLE_PAPER.pdf)
[![Paper DOCX](https://img.shields.io/badge/Paper-Word%20DOCX-2B579A?style=for-the-badge&logo=microsoftword&logoColor=white)](results/exp47/SAMPLE_RESEARCH_PAPER.docx)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

---

## 🌐 Live Deployments & Interactive Platform

The Quantum Text Security Research Platform is deployed as a decoupled, full-stack cloud application:

* **🚀 Frontend (React + Vite SPA)**: Hosted on **Vercel** at [https://quantum-three-hazel.vercel.app](https://quantum-three-hazel.vercel.app)
  * Interactive **Representation Lab** (Encoder screening, cosine clustering vs Hilbert dispersion).
  * Real-time **Multi-Model Inference & Vector Inspector** (Linear SVM, RBF SVM, QSVC).
  * In-distribution and out-of-distribution performance dashboards with full offline fallback data.
* **⚡ Backend REST API (FastAPI + PyTorch)**: Hosted on **Render** at [https://quantum-backend-uxm8.onrender.com](https://quantum-backend-uxm8.onrender.com)
  * **Interactive Swagger Documentation**: [https://quantum-backend-uxm8.onrender.com/docs](https://quantum-backend-uxm8.onrender.com/docs)
  * **Health Check & Engine Status**: [https://quantum-backend-uxm8.onrender.com/api/health](https://quantum-backend-uxm8.onrender.com/api/health)
  * **Live Endpoints**: `/api/predict`, `/api/statevector`, `/api/representations/candidates`, `/api/research/summary`

---

## 🔬 Research at a Glance

This repository contains the complete experimental codebase, frozen dataset partitions, statistical verification pipelines, academic manuscript sources, and full-stack research platform for evaluating **parameter-free quantum fidelity kernels** (two-layer cyclic $ZZFeatureMap$) against **matched classical radial basis function (RBF) kernels** and an audited suite of **eight classical machine learning baselines** across three cybersecurity text corpora.

```
====================================================================================================
                                      RESEARCH SPECIFICATION MATRIX
====================================================================================================
Primary Research Question:   Does a parameter-free quantum fidelity kernel provide a practical
                             advantage for text security classification?
Benchmark Corpora (3):       SMS Spam (5.5k), CEAS 2008 (39.1k raw / 15k exp), MeAJOR Archive (108.6k)
Record Accounting:           153,413 Raw -> 153,410 Usable -> 35,572 Controlled Experimental Subsets
Classifiers Audited (8):     Linear SVM, Matched RBF, Tuned RBF, Logistic Regression, Multinomial NB,
                             Random Forest, XGBoost, Multi-Layer Perceptron (MLP), k-NN
Quantum Setup:               2-Layer Cyclic ZZFeatureMap on 2–12 Qubits (16D Memory Stress Test)
Statistical Protocol:        10 Canonical Seeds, 10,000 Permutations, 10,000 Bootstrap CIs, TOST, BH-FDR
Equivalence Boundary:        ε = ±0.01 F1 (Two One-Sided Tests practical equivalence boundary)
====================================================================================================
```

---

## 📊 Major Scientific Findings

1. **In-Distribution Practical Equivalence ($\varepsilon = \pm 0.01\text{ F1}$)**:
   - At intermediate dimensions ($8\text{D}$ and $10\text{D}$), the quantum fidelity kernel achieves minor statistically detectable gains ($+0.46\text{ pp}$ at $8\text{D}$, $p = 0.0016$; $+0.57\text{ pp}$ at $10\text{D}$, $p = 0.0052$) that satisfy Two One-Sided Tests (TOST) practical equivalence ($|\Delta| \le 0.01$).
   - At $12\text{D}$, quantum and classical RBF converge to full statistical parity ($+0.14\text{ pp}$, $p = 0.2824$).
   - Against a validation-tuned classical RBF baseline, the quantum margin narrows to full statistical parity across all dimensions (e.g., $+0.12\text{ pp}$ at $8\text{D}$, $p = 0.1840$).

2. **Representation Primacy & Ranking Inversion**:
   - Upstream feature representation dominates kernel selection by an order of magnitude.
   - On **CEAS 2008**, switching from sparse TF-IDF to dense RoBERTa embeddings reverses the quantum advantage into a classical advantage (net shift: $-3.90\text{ pp}$).
   - On **SMS Spam**, contrastive sentence embeddings (**all-mpnet-base-v2**) cause a **$-52.88\text{ pp}$ catastrophic quantum collapse** consistent with destructive phase-wrapping under cyclic Pauli-$Z$ gates.

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
- **Centered Kernel-Target Alignment (CKA)**:
  $$\text{CKA}(K, Y) = \frac{\langle H K H, H Y H \rangle_F}{\|H K H\|_F \|H Y H\|_F}, \quad H = I - \frac{1}{N}\mathbf{1}\mathbf{1}^T, \quad Y = \mathbf{y}\mathbf{y}^T$$
- **Dual Support Vector Classification**:
  $$\max_{\boldsymbol{\alpha}} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) \quad \text{s.t.} \quad 0 \le \alpha_i \le C \cdot w_{y_i}, \; \sum_{i=1}^N \alpha_i y_i = 0$$

---

## 📈 Complete Classical Baseline Audit (8 Models across 3 Datasets)

```
+---------------------------------------------------------------------------------------------------------+
|                    COMPREHENSIVE CLASSICAL BASELINE AUDIT (FULL TF-IDF vs 8D SVD)                       |
+------------------+-----------------------+---------------------+-------------------+--------------------+
| Corpus           | Model Architecture    | Full 50k TF-IDF F1  | Matched 8D SVD F1 | Train Latency (s)  |
+------------------+-----------------------+---------------------+-------------------+--------------------+
| SMS Spam         | Linear SVM            | 0.9559 ± 0.000      | 0.8273 ± 0.011    | 0.03 s             |
|                  | Matched RBF SVM       | 0.9498 ± 0.000      | 0.8156 ± 0.012    | 1.53 s             |
|                  | Tuned RBF SVM         | 0.9521 ± 0.000      | 0.8285 ± 0.010    | 5.20 s             |
|                  | Logistic Regression   | 0.9346 ± 0.000      | 0.8226 ± 0.012    | 0.09 s             |
|                  | Random Forest         | 0.9423 ± 0.005      | 0.8500 ± 0.006    | 0.90 s             |
|                  | XGBoost               | 0.9059 ± 0.000      | 0.8452 ± 0.013    | 0.86 s             |
|                  | MLP (Neural Net)      | 0.9324 ± 0.008      | 0.8284 ± 0.014    | 4.68 s             |
|                  | Multinomial / Gauss NB| 0.9158 ± 0.000      | 0.8195 ± 0.007    | 0.01 s             |
|                  | k-NN                  | 0.6900 ± 0.000      | 0.8401 ± 0.019    | 0.00 s             |
+------------------+-----------------------+---------------------+-------------------+--------------------+
| CEAS 2008        | Linear SVM            | 0.9954 ± 0.000      | 0.9535 ± 0.001    | 0.15 s             |
|                  | Matched RBF SVM       | 0.9968 ± 0.000      | 0.9638 ± 0.001    | 57.15 s            |
|                  | Tuned RBF SVM         | 0.9972 ± 0.000      | 0.9691 ± 0.001    | 185.2 s            |
|                  | Logistic Regression   | 0.9935 ± 0.000      | 0.9505 ± 0.001    | 0.26 s             |
|                  | Random Forest         | 0.9895 ± 0.001      | 0.9816 ± 0.001    | 1.76 s             |
|                  | XGBoost               | 0.9886 ± 0.000      | 0.9810 ± 0.001    | 23.24 s            |
|                  | MLP (Neural Net)      | 0.9966 ± 0.000      | 0.9614 ± 0.003    | 46.28 s            |
|                  | Multinomial / Gauss NB| 0.9928 ± 0.000      | 0.7165 ± 0.000    | 0.01 s             |
|                  | k-NN                  | 0.9957 ± 0.000      | 0.9823 ± 0.000    | 0.02 s             |
+------------------+-----------------------+---------------------+-------------------+--------------------+
| MeAJOR Archive   | Linear SVM            | 0.9721 ± 0.000      | 0.8448 ± 0.002    | 0.18 s             |
|                  | Matched RBF SVM       | 0.9700 ± 0.000      | 0.8709 ± 0.003    | 71.84 s            |
|                  | Tuned RBF SVM         | 0.9734 ± 0.000      | 0.8742 ± 0.003    | 240.5 s            |
|                  | Logistic Regression   | 0.9574 ± 0.000      | 0.8458 ± 0.002    | 0.35 s             |
|                  | Random Forest         | 0.9516 ± 0.002      | 0.9014 ± 0.006    | 2.16 s             |
|                  | XGBoost               | 0.9453 ± 0.000      | 0.9005 ± 0.005    | 26.59 s            |
|                  | MLP (Neural Net)      | 0.9727 ± 0.003      | 0.8795 ± 0.002    | 42.12 s            |
|                  | Multinomial / Gauss NB| 0.9481 ± 0.000      | 0.7258 ± 0.005    | 0.01 s             |
|                  | k-NN                  | 0.9532 ± 0.000      | 0.8870 ± 0.002    | 0.02 s             |
+------------------+-----------------------+---------------------+-------------------+--------------------+
```

---

## 📁 Repository Structure

```
quantum/
├── app/
│   ├── api.py                     # FastAPI REST API service (Deployed on Render)
│   ├── model_engine.py            # Live inference engine (QSVC, RBF, Linear)
│   ├── schemas/                   # Pydantic request/response schemas
│   └── research/
│       ├── registry.py            # Experiment and representation query registries
│       └── research_store.py      # Authoritative frozen experimental data store
├── frontend/                      # React + Vite UI (Deployed on Vercel)
│   ├── src/
│   │   ├── components/
│   │   │   ├── RepresentationLab.jsx     # Encoder screening & geometry matrix
│   │   │   ├── Exp41ScreeningView.jsx    # Multi-dataset screening dashboard
│   │   │   ├── ModelHierarchyExplainer.jsx # Tripartite architecture visualizer
│   │   │   └── TextRepresentationInspector.jsx # Real-time vector inspector
│   │   └── data/
│   │       └── researchFallbackData.js   # 100% offline CDN research fallback
│   └── package.json
├── datasets/                      # Standardized benchmark datasets in CSV format
│   ├── README.md                  # Complete dataset manifest & split schemas
│   ├── sms_spam_raw.csv           # Full SMS Spam Collection (5,572 rows)
│   ├── ceas_2008_raw.csv          # Full CEAS 2008 Email Archive (39,154 rows)
│   ├── meajor_archive_raw.csv     # Multi-Source Email Archive (108,685 rows)
│   └── ...                        # Frozen train / validation / test splits
├── experiments/
│   ├── 39_multi_dataset_eval.py   # Multi-dataset 10-seed in-distribution sweep
│   ├── 40_confirmation_experiments.py # Authoritative confirmation experiments
│   ├── 41_representation_screening.py # Upstream text representation sweeps
│   ├── 45_classical_baseline_audit.py # 480-run 8-classifier classical audit
│   └── 46_manuscript_update.py    # Submission package generator
├── paper/
│   └── submission/
│       ├── main.tex               # LaTeX research manuscript source
│       ├── references.bib         # 28-paper comprehensive bibliography
│       └── figures/               # High-resolution vector PDF figures
├── results/
│   └── exp47/
│       ├── SAMPLE_PAPER.pdf       # 10-page complete compiled PDF
│       ├── SAMPLE_RESEARCH_PAPER.docx # Complete Word document with figures
│       ├── SAMPLE_PAPER.tex       # Standalone paper LaTeX source
│       └── SAMPLE_PAPER.md        # Formatted markdown preprint version
├── build_paper_docx_and_datasets.py # DOCX paper & CSV dataset exporter
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start & Local Development

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

### 2. Run the Full-Stack Application Locally

**Start the FastAPI Backend (Render parity):**
```bash
uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to `http://127.0.0.1:8000/docs` to test the API endpoints.

**Start the React Frontend (Vercel parity):**
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

### 4. Build Paper Artifacts
```bash
# Compile LaTeX PDF
tectonic results/exp47/SAMPLE_PAPER.tex

# Generate DOCX Manuscript & Export CSV Datasets
python3 build_paper_docx_and_datasets.py
```

---

## 📖 Citation

If you use this benchmark, experimental protocols, or dataset partitions in your research, please cite:

```bibtex
@article{anonymous2026quantumtextsecurity,
  title={Evaluating a Parameter-Free Quantum Fidelity Kernel for Text Security Under Representation and Domain Shift},
  author={Anonymous Authors},
  journal={Archival Research Preprint},
  year={2026},
  note={Under Peer Review}
}
```

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
