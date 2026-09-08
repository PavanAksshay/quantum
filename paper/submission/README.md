# Final LaTeX Submission Package

**Manuscript Title**: When Do Quantum Kernels Help for Text Security? A Multi-Dataset Evaluation of Representation, Geometry, Generalization, and Computational Cost  
**Submission Package Location**: `paper/submission/`  
**Date**: September 2026  

---

## 1. Directory and File Inventory

```
paper/submission/
├── main.tex                    # Complete LaTeX source for main conference paper
├── supplementary.tex           # Complete LaTeX source for supplementary material document
├── references.bib              # Complete BibTeX bibliography database
├── figures/                    # Vector PDF and PNG publication figures (1 to 8)
│   ├── figure_1_experimental_framework.pdf / .png
│   ├── figure_2_iid_f1_vs_dimensionality.pdf / .png
│   ├── figure_3_quantum_minus_rbf_vs_dimensionality.pdf / .png
│   ├── figure_4_iid_vs_source_holdout.pdf / .png
│   ├── figure_5_runtime_vs_dimensionality.pdf / .png
│   ├── figure_6_representation_interaction.pdf / .png
│   ├── figure_7_geometry_correlation_vs_dimensionality.pdf / .png
│   └── figure_8_entropy_vs_kernel_diversity.pdf / .png
├── tables/                     # Audited CSV tables from Exp 39/40 evidence pack
│   ├── table_1_dataset_characteristics.csv
│   ├── table_2_classical_baselines.csv
│   ├── table_3_canonical_comparison.csv
│   ├── table_4_dimensionality_scaling.csv
│   ├── table_5_source_holdout.csv
│   ├── table_6_statistical_tests.csv
│   ├── table_7_geometry_diagnostics.csv
│   └── table_8_runtime_scalability.csv
├── README.md                   # This submission guide
└── SUBMISSION_AUDIT.md         # Final comprehensive publication audit report
```

---

## 2. Compilation Instructions

The LaTeX documents are formatted using standard LaTeX2e packages (`article`, `amsmath`, `booktabs`, `graphicx`, `hyperref`, `cite`).

### Standard Compilation Commands
To compile the main paper:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

To compile the supplementary document:
```bash
pdflatex supplementary.tex
bibtex supplementary
pdflatex supplementary.tex
pdflatex supplementary.tex
```

*(Note: The package is fully compatible with Overleaf, arXiv TeX Live environments, and conference submission portals.)*

---

## 3. Provenance of Figures and Tables

- **Figures 1–8**: Generated directly from frozen empirical results in `results/exp39_paper/figures/` and archived in both vector PDF and raster PNG formats.
- **Tables 1–8**: Extracted from audited CSV evidence in `results/exp39_paper/tables/` and confirmed against the 10-seed confirmation package in `results/exp40_final/`.

---

## 4. Software and Hardware Environment

The headline confirmation experiments (Exp 40) were executed and audited under the following frozen environment:
- **Hardware Architecture**: Apple Silicon ARM64
- **Operating System**: macOS Darwin Kernel Version 25.6.0
- **Python**: Version 3.12.4
- **PyTorch**: Version 2.13.0 (configured with `torch.complex128` double-precision tensors)
- **scikit-learn**: Version 1.9.0
- **NumPy**: Version 2.5.2
- **SciPy**: Version 1.18.1
- **pandas**: Version 2.3.3
- **Simulation Engine**: Exact PyTorch statevector tensor simulator (zero network or quantum cloud dependencies)

---

## 5. Main Paper vs Supplementary Material Architecture

- **Main Paper (`main.tex`)**: Formatted for a standard two-column conference submission (approx. 7,500 words). Presents the motivation, methodology, six core empirical results, substantive discussion, explicit limitations, reproducibility summary, and conclusions.
- **Supplementary Document (`supplementary.tex`)**: Provides the exhaustive empirical backbone across 13 dedicated sections (S1–S13), including per-seed performance tables for all 10 seeds, full classical baseline metrics, geometric derivations, out-of-vocabulary overlap tables, and the complete claim-evidence verification matrix.
