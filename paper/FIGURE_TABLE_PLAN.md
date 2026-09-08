# Publication Figure and Table Architecture

**Document Version**: 1.0 (Frozen)  
**Date**: September 5, 2026  
**Scope**: Complete specifications, source files, columns, axes, and interpretation guardrails for all main manuscript tables and figures.

---

## 1. Table Architecture (Main Manuscript)

### Table 1: Benchmark Dataset Characteristics and Splitting Protocols
- **Purpose**: Document corpus properties, class balance, text length, and frozen split partitions with zero leakage.
- **RQ Addressed**: Experimental Control / Data Hygiene
- **Source Artifacts**: `results/exp39_paper/tables/table_1_dataset_characteristics.csv`, `results/exp36_audit/split_hashes.csv`
- **Columns**: Dataset, Raw Samples, Usable Samples, Positive Class (Spam/Phish %), Median Word Count, Source Split Structure, Train Size ($N_{\text{tr}}$), Val Size ($N_{\text{va}}$), Test Size ($N_{\text{te}}$).
- **Core Values**:
  - SMS Spam: 5,572 usable texts; 13.4% positive; median length 12 words; 3,343 train / 1,114 val / 1,115 test.
  - CEAS 2008: 39,154 raw emails $\to$ 15,000 canonical subset; 18.9% positive; median length 48 words; 10,000 train / 2,500 val / 2,500 test.
  - MeAJOR Archive: 108,684 usable emails; 19.3% positive; median length 84 words; 10,000 train / 2,500 val / 2,500 test (IID) & 5,000 test (Holdout).

---

### Table 2: High-Dimensional and Contextual Classical Baselines
- **Purpose**: Establish classical performance ceilings on full feature representations (50k TF-IDF and 768D RoBERTa) against low-dimensional PCA baselines.
- **RQ Addressed**: RQ1 (Model Comparison) & RQ3 (Representation Interaction)
- **Source Artifacts**: `results/exp39_paper/tables/table_2_classical_baselines.csv`, `results/metrics/multidataset_classical_benchmark_metrics.csv`
- **Columns**: Dataset, Representation, Classifier, Test F1, PR-AUC, ROC-AUC, Accuracy, Training Time (s).
- **Core Values**:
  - MeAJOR 50k TF-IDF Linear SVM: F1 = $0.9721$, PR-AUC = $0.9884$, ROC-AUC = $0.9912$.
  - MeAJOR 768D RoBERTa Linear SVM: F1 = $0.9841$, PR-AUC = $0.9942$.
  - CEAS 50k TF-IDF Linear SVM: F1 = $0.9684$.
  - Low-dimensional 8D PCA Linear SVM: F1 = $0.8445$ (MeAJOR).

---

### Table 3: Canonical In-Distribution Comparison (Matched 8D PCA)
- **Purpose**: Present primary matched comparison between Linear SVM, Classical RBF, and Quantum Fidelity Kernel at the canonical 8D anchor.
- **RQ Addressed**: RQ1 (Model Comparison)
- **Source Artifacts**: `results/exp40_final/exp40_results.csv`, `results/exp40_final/exp40_statistical_summary.csv`
- **Columns**: Dataset, Dimension, Model, Mean F1 $\pm$ SD, Mean PR-AUC, Mean ROC-AUC, $\Delta \text{F1}$ vs RBF, Total Runtime (s).
- **Core Values (MeAJOR IID 8D, $N=10$ seeds)**:
  - Linear SVM (8D): $\text{F1} = 0.8445 \pm 0.0022$, PR-AUC = $0.9315$, ROC-AUC = $0.9404$, Runtime = $0.23\text{s}$.
  - Classical RBF (8D): $\text{F1} = 0.8709 \pm 0.0030$, PR-AUC = $0.9423$, ROC-AUC = $0.9535$, Runtime = $2.1\text{s}$.
  - Quantum Kernel (8D): $\text{F1} = 0.8754 \pm 0.0029$, PR-AUC = $0.9372$, ROC-AUC = $0.9515$, $\Delta \text{F1} = +0.0046$, Runtime = $7.7\text{s}$.

---

### Table 4: Dimensionality Scaling Trajectory (2D to 16D)
- **Purpose**: Detail performance recovery and scaling across representation dimensions $d \in \{2, 4, 6, 8, 10, 12, 16\}$.
- **RQ Addressed**: RQ2 (Dimensionality Scaling)
- **Source Artifacts**: `results/experiment_35/tables/experiment_35_iid_results.csv`, `results/exp40_final/exp40_statistical_summary.csv`
- **Columns**: Dimension ($d$), Linear F1, Classical RBF F1, Quantum Kernel F1, $\Delta \text{F1}$ (Q - RBF), Quantum Simulation Time (s), Feasibility.
- **Core Values**:
  - 2D: Linear = $0.6853$, RBF = $0.6735$, Quantum = $0.6447$ ($\Delta = -0.0288$).
  - 4D: Linear = $0.8150$, RBF = $0.8059$, Quantum = $0.7876$ ($\Delta = -0.0183$).
  - 8D: Linear = $0.8445$, RBF = $0.8709$, Quantum = $0.8754$ ($\Delta = +0.0046$).
  - 10D: Linear = $0.8668$, RBF = $0.8967$, Quantum = $0.9023$ ($\Delta = +0.0057$).
  - 12D: Linear = $0.8892$, RBF = $0.9123$, Quantum = $0.9137$ ($\Delta = +0.0014$, $p=0.282$).
  - 16D: Linear = $0.8935$, RBF = $0.9203$, Quantum = OOM / Infeasible ($>10.5\text{ GB}$).

---

### Table 5: Cross-Source Domain Generalization and Holdout Degradation
- **Purpose**: Compare out-of-sample transfer performance across email collections (Direction A and Direction B).
- **RQ Addressed**: RQ5 (Cross-Source Generalization)
- **Source Artifacts**: `results/exp40_final/exp40_results.csv`, `results/exp36_audit/EXP36_SUMMARY.csv`
- **Columns**: Transfer Regime, Train Source, Test Source, Model, IID F1, Holdout F1, Absolute Drop ($\Delta \text{F1}_{\text{IID}\to\text{OOD}}$), Relative Degradation (%).
- **Core Values**:
  - Direction A 8D (TREC5+6 $\to$ TREC7): RBF F1 = $0.7201$ (-17.5%), Quantum F1 = $0.7034$ (-19.6%), Full TF-IDF Linear F1 = $0.8922$.
  - Direction B 8D (TREC7 $\to$ TREC5+6, $N=10$ seeds): Linear F1 = $0.6936$, RBF F1 = $0.6913$ (-20.6%), Quantum F1 = $0.6680$ (-23.7%, $\Delta = -0.0233$, $p=0.0046$).

---

### Table 6: Inferential Statistical Tests and Practical Equivalence Bounds
- **Purpose**: Present formal statistical hypothesis testing with non-parametric bootstrap CIs, permutation $p$-values, and FDR adjustments.
- **RQ Addressed**: Statistical Rigor / Synthesis
- **Source Artifacts**: `results/exp40_final/exp40_statistical_summary.csv`
- **Columns**: Comparison Setting, Mean $\Delta \text{F1}$, 95% Bootstrap CI, Paired Permutation $p$, BH FDR $p$, Practical Equivalence Status ($\varepsilon = 0.01$).
- **Core Values**:
  - MeAJOR IID 8D: $\Delta = +0.0046$, 95% CI $[+0.0030, +0.0061]$, $p = 0.0016$, FDR $p = 0.0064$, **Practical Equivalence**.
  - MeAJOR IID 10D: $\Delta = +0.0057$, 95% CI $[+0.0032, +0.0081]$, $p = 0.0052$, FDR $p = 0.0069$, **Practical Equivalence**.
  - MeAJOR IID 12D: $\Delta = +0.0014$, 95% CI $[-0.0010, +0.0037]$, $p = 0.2824$, FDR $p = 0.2824$, **Practical Equivalence / Parity**.
  - MeAJOR Direction B 8D: $\Delta = -0.0233$, 95% CI $[-0.0353, -0.0117]$, $p = 0.0046$, FDR $p = 0.0069$, **Meaningful Classical Advantage**.

---

### Table 7: Kernel Geometry and Feature Space Diagnostics
- **Purpose**: Quantify Gram matrix correlation, target label alignment, state dispersion, and effective rank.
- **RQ Addressed**: RQ4 (Kernel Geometry)
- **Source Artifacts**: `results/exp39_paper/tables/table_7_geometry_diagnostics.csv`
- **Columns**: Dataset, Dimension, Gram Correlation ($r_{\text{Q,RBF}}$), Quantum Label Alignment, RBF Label Alignment, Alignment Ratio (Q/RBF), State Entropy, Kernel Diversity.
- **Core Values**:
  - MeAJOR 8D: $r_{\text{Q,RBF}} = 0.5807$, Quantum Alignment = $0.0382$, RBF Alignment = $0.0694$ (Ratio: 55.0%), Entropy = $2.64$, Diversity = $0.182$.
  - CEAS 8D: $r_{\text{Q,RBF}} = 0.5912$, Quantum Alignment = $0.0402$, RBF Alignment = $0.0773$ (Ratio: 52.0%).
  - SMS 8D: $r_{\text{Q,RBF}} = 0.5541$, Quantum Alignment = $0.0210$, RBF Alignment = $0.0450$ (Ratio: 46.7%).

---

### Table 8: Computational Runtime and Memory Scaling
- **Purpose**: Document wall-clock execution time and peak memory across pipeline stages and dimensions.
- **RQ Addressed**: RQ6 (Computational Cost)
- **Source Artifacts**: `results/exp40_final/exp40_results.csv`, `results/experiment_35/tables/experiment_35_runtime.csv`
- **Columns**: Dimension ($d$), Model, Kernel Matrix Construction (s), Classifier Training (s), Inference (s), Total Runtime (s), Speedup / Slowdown vs RBF, Peak RAM (MB).
- **Core Values**:
  - 8D: RBF = $2.1\text{s}$ ($1.0\times$); Quantum = $7.7\text{s}$ ($3.7\times$).
  - 10D: RBF = $1.6\text{s}$ ($1.0\times$); Quantum = $19.1\text{s}$ ($11.9\times$).
  - 12D: RBF = $1.7\text{s}$ ($1.0\times$); Quantum = $108.8\text{s}$ ($64.0\times$).
  - 16D: Quantum Memory $>10.5\text{ GB}$ (infeasible in local simulation).

---

## 2. Figure Architecture (Main Manuscript)

| Figure ID | Title / Purpose | RQ | Source File | X-Axis / Y-Axis | Key Takeaway | Reviewer Objection Guardrail |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- |
| **Figure 1** | Overall Experimental Architecture | Control | `figure_1_experimental_framework.pdf` | Flowchart: Text $\to$ TF-IDF $\to$ PCA $\to$ Quantum/RBF SVM | Illustrates strict input matching and zero leakage. | Shows exact parity in preprocessing and thresholds. |
| **Figure 2** | IID F1 vs Dimensionality (2D–12D) | RQ2 | `figure_2_iid_f1_vs_dimensionality.pdf` | Dimension ($d$) vs Test F1 Score | Monotonic scaling confirms PCA information bottleneck recovery. | Clarifies low-D failure is PCA loss, not quantum defect. |
| **Figure 3** | $\Delta \text{F1}$ with Practical Equivalence Band | RQ1 | `figure_3_quantum_minus_rbf_vs_dimensionality.pdf` | Dimension ($d$) vs $\Delta \text{F1}$ (pp) | All IID deltas fall inside $\pm 1.0$ pp equivalence band ($[-1, +1]$). | Proves small positive deltas are not practically meaningful. |
| **Figure 4** | IID vs Cross-Source Generalization | RQ5 | `figure_4_iid_vs_source_holdout.pdf` | Evaluation Regime vs Test F1 | Quantum degrades more severely than RBF under domain transfer. | Disproves hypothesized Hilbert space domain robustness. |
| **Figure 5** | Simulation Runtime Scaling | RQ6 | `figure_5_runtime_vs_dimensionality.pdf` | Dimension / Qubits vs Runtime (s) [Log Scale] | Exponential simulation cost scaling without accuracy gains. | Scoped explicitly to classical statevector simulation. |
| **Figure 6** | Representation Interaction (TF-IDF vs RoBERTa) | RQ3 | `figure_6_representation_interaction.pdf` | Representation vs $\Delta \text{F1}$ (pp) | TF-IDF favors cyclic ZZ map; RoBERTa reverses margin by 3.9 pp. | Explains contradictory claims in prior unvalidated QML studies. |
| **Figure 7** | Quantum-RBF Gram Matrix Correlation vs Qubits | RQ4 | `figure_7_geometry_correlation_vs_dimensionality.pdf` | Qubits ($n$) vs Pearson $r$ | Moderate correlation ($r=0.55\text{--}0.65$) decays at higher dimensions. | Demonstrates geometric similarity without claiming identity. |
| **Figure 8** | State Dispersion vs Pairwise Kernel Diversity | RQ4 | `figure_8_entropy_vs_kernel_diversity.pdf` | State Entropy vs Kernel Diversity | Strong inverse correlation ($r=-0.80$) across all 3 corpora. | Decouples dispersion from pairwise kernel diversity. |
