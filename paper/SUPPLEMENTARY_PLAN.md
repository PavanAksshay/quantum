# Supplementary Material Architecture and Appendix Blueprint

**Document Version**: 1.0 (Frozen)  
**Date**: September 5, 2026  
**Scope**: Complete inventory and section layout for the Supplementary Material and Appendices.

---

## 1. Supplementary Section Structure

### Appendix A: Extended Dataset and Corpus Statistics
- Complete lexical distributions, vocabulary size curves, duplicate counts, and token length histograms across SMS, CEAS 2008, and MeAJOR collections.
- Full sample identifier hash tables (`split_hashes.csv`) and verification checksums for frozen training, validation, and test sets.
- MeAJOR source breakdown: sample distributions across TREC 2005 (trec5), TREC 2006 (trec6), and TREC 2007 (trec7).

---

### Appendix B: Complete Classical Baseline Suite
- Full metric tables for all evaluated classical models:
  - Multinomial Naive Bayes (MNB)
  - Logistic Regression (L2 / ElasticNet)
  - Random Forest Classifier (100 trees)
  - Linear SVM (LinearSVC, C=1.0)
  - Classical RBF SVM (SVC, C=1.0, gamma='scale')
  - Fine-Tuned Transformer Baselines (RoBERTa-base 768D embeddings).

---

### Appendix C: Comprehensive Dimensionality Scaling Record (2D to 16D)
- Detailed seed-level results across all evaluated dimensions ($d \in \{2, 4, 6, 8, 10, 12, 16\}$):
  - F1, PR-AUC, ROC-AUC, Accuracy, Balanced Accuracy, Precision, Recall, Decision Margin.
  - Variance explained by PCA components across all dimensions ($2\text{D}\text{--}16\text{D}$).
  - Full breakdown of linear vs nonlinear performance recovery trajectories.

---

### Appendix D: Detailed Geometric Diagnostics and Spectral Properties
- Mathematical definitions and complete metric tables for:
  - Frobenius norm distance and Pearson correlation between Gram matrices ($r_{\text{Q,RBF}}$).
  - Target label alignment formulation: $A(K, y) = \frac{\langle K, y y^T \rangle_F}{\|K\|_F \|y y^T\|_F}$.
  - Statevector von Neumann / Shannon entropy: $S(\psi) = -\sum_i |\alpha_i|^2 \ln |\alpha_i|^2$.
  - Pairwise kernel diversity and effective rank of Gram matrices.
  - State dispersion vs text length regressions.

---

### Appendix E: Statistical Methodology and Inferential Test Protocols
- Detailed algorithmic procedures for:
  - Paired Percentile Bootstrap ($B = 10,000$ resamples) for confidence interval construction.
  - Two-sided Paired Permutation Test ($10,000$ iterations) for exact non-parametric $p$-values.
  - McNemar's test on paired classification 2x2 contingency tables.
  - Benjamini-Hochberg (BH) step-up procedure for controlling False Discovery Rate (FDR) at $\alpha = 0.05$.

---

### Appendix F: Computational Measurement and Hardware Environment
- Exact hardware environment specification (`RUN_METADATA.md`):
  - OS: Darwin 25.6.0 (Apple Silicon arm64, 8 cores, 8.0 GB RAM).
  - Python 3.12.4, PyTorch 2.12.1 (CPU statevector engine), scikit-learn 1.9.0, NumPy 2.5.0, SciPy 1.18.0.
- Wall-clock runtime logs per seed, kernel matrix construction time vs SVM fit time, and RAM footprint profiling.

---

### Appendix G: Experiment Lineage and Audit Trail (Exp 24 to Exp 40)
- Complete lineage audit table tracing the evolution from Exp 24 (preprocessing audit) through Exp 36 (harmonization), Exp 38 (inferential statistics), and Exp 40 (10-seed confirmation).

---

### Appendix H: Reproducibility Guide and Open-Source Code Structure
- Step-by-step instructions for reproducing all tables and figures from raw data files and frozen split identifiers using provided scripts (`experiments/40_confirmation_experiments.py`, `canonical_protocol.json`).
