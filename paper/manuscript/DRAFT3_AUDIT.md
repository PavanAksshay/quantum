# Manuscript Draft 3 Quality Control Audit Report

**Date**: September 5, 2026  
**Audited Section**: `paper/manuscript/06_results.md` (Results and Statistical Synthesis)  
**Authoritative Evidence Source**: `results/exp40_final/`, `results/exp39_paper/`, and `paper/PAPER_BLUEPRINT.md`

---

## 1. Numerical Consistency and Verification Audit

| Parameter / Metric | Value in Draft 3 | Frozen Source File | Status | Verification Detail |
| :--- | :--- | :--- | :---: | :--- |
| **MeAJOR IID 8D Quantum F1** | $0.8754 \pm 0.0029$ | `exp40_statistical_summary.csv` | **PASSED** | Exact match with Exp40A 10-seed confirmation. |
| **MeAJOR IID 8D Classical RBF F1** | $0.8709 \pm 0.0030$ | `exp40_statistical_summary.csv` | **PASSED** | Exact match with Exp40A 10-seed confirmation. |
| **MeAJOR IID 8D Linear SVM F1** | $0.8445 \pm 0.0022$ | `exp40_results.csv` | **PASSED** | Exact match with Exp40A 10-seed confirmation. |
| **MeAJOR IID 8D Paired Delta** | $+0.004565 \pm 0.002657$ | `exp40_statistical_summary.csv` | **PASSED** | 95% Bootstrap CI: $[+0.003040, +0.006139]$. |
| **MeAJOR IID 8D Permutation $p$** | $p = 0.0016$ | `exp40_statistical_summary.csv` | **PASSED** | BH FDR adjusted: $p = 0.0064$. |
| **MeAJOR IID 10D Quantum F1** | $0.9023 \pm 0.0034$ | `exp40_statistical_summary.csv` | **PASSED** | Classical RBF: $0.8967 \pm 0.0049$. |
| **MeAJOR IID 10D Paired Delta** | $+0.0057 \pm 0.0042$ | `exp40_statistical_summary.csv` | **PASSED** | 95% CI: $[+0.0032, +0.0081]$, $p = 0.0052$. |
| **MeAJOR IID 12D Quantum F1** | $0.9137 \pm 0.0046$ | `exp40_statistical_summary.csv` | **PASSED** | Classical RBF: $0.9123 \pm 0.0023$. |
| **MeAJOR IID 12D Paired Delta** | $+0.0014 \pm 0.0040$ | `exp40_statistical_summary.csv` | **PASSED** | 95% CI: $[-0.0010, +0.0037]$, $p = 0.2824$. |
| **MeAJOR Direction B Quantum F1** | $0.6680 \pm 0.0094$ | `exp40_statistical_summary.csv` | **PASSED** | Classical RBF: $0.6913 \pm 0.0161$. |
| **MeAJOR Direction B Linear F1** | $0.6936 \pm 0.0264$ | `exp40_results.csv` | **PASSED** | Linear SVM contextual holdout baseline. |
| **MeAJOR Direction B Paired Delta** | $-0.023307 \pm 0.019988$ | `exp40_statistical_summary.csv` | **PASSED** | 95% CI: $[-0.035280, -0.011675]$, $p = 0.0046$. |
| **Direction B Degradation Rates** | Quantum $23.69\%$, RBF $20.62\%$ | `FINAL_EVIDENCE_SYNTHESIS.md` | **PASSED** | Exact relative degradation from IID baselines. |
| **Direction A 8D Metrics** | Linear $0.7205$, RBF $0.7201$, Q $0.7034$ | `table_5_source_holdout.csv` | **PASSED** | Direction A transfer values verified. |
| **CEAS Representation Ablation (8D)**| TF-IDF (+0.95 pp) vs RoBERTa (-2.95 pp) | `table_2_classical_baselines.csv` | **PASSED** | 3.90 pp net shift verified. |
| **Dimensionality Scaling (2D–12D)** | 2D: 0.6447/0.6735; 4D: 0.7876/0.8059; 6D: 0.8253/0.8226; 8D: 0.8752/0.8731; 10D: 0.9037/0.8977; 12D: 0.9119/0.9123 | `table_4_dimensionality_scaling.csv` | **PASSED** | Complete scaling sweep matches Exp 35. |
| **Lexical OOV Rates (Dir A & B)** | Dir A: Type 92.73%, Token 35.34%; Dir B: Type 93.91%, Token 34.02% | Exp 34 audit logs | **PASSED** | TruncatedSVD centroid shifts: 0.514 (A), 0.399 (B). |
| **Gram Matrix Correlations** | 2D: 0.6516; 4D: 0.5812; 8D: 0.5807; 12D: 0.5529; 16D: 0.4566 | `table_7_geometry_diagnostics.csv` | **PASSED** | Exact Pearson correlation values verified. |
| **Label Alignment Ratios** | CEAS: 52.01%; SMS: 52.90%; MeAJOR: 55.04% | `table_7_geometry_diagnostics.csv` | **PASSED** | Exact 50–60% deficit range verified. |
| **Entropy vs Diversity Regressions** | SMS: -0.8257; CEAS: -0.8170; MeAJOR: -0.7822 | `table_7_geometry_diagnostics.csv` | **PASSED** | Exact within-dataset inverse correlations verified. |
| **12D Simulation Runtime Ratio** | Quantum: 108.8s vs RBF: 1.7s ($64.0\times$) | `exp40_results.csv` | **PASSED** | Scoped strictly to measured simulation runtime. |

---

## 2. Statistical Discipline and Terminology Audit

| Statistical Check | Verification Criteria | Status | Applied Wording & Context |
| :--- | :--- | :---: | :--- |
| **Practical Equivalence ($\varepsilon = 0.01$)** | All sub-1.0 pp differences treated as practically equivalent. | **PASSED** | 8D (+0.46 pp) and 10D (+0.57 pp) explicitly classified as "statistically detectable but within the predefined practical-equivalence region." |
| **Non-Significant Framing** | Non-significant differences not labeled "statistically identical." | **PASSED** | 12D (+0.14 pp, $p = 0.2824$) described as showing "no statistically detectable difference" and establishing "practical parity." |
| **Domain-Shift Disadvantage** | Statistically significant negative deltas exceeding $\varepsilon$. | **PASSED** | Direction B (-2.33 pp, $p = 0.0046$) classified as a "statistically supported classical advantage exceeding the practical-equivalence threshold." |
| **Paired Evaluation Rigor** | Explicit paired difference evaluation. | **PASSED** | Seed-level pairing used across all 10 independent seeds with 10,000 bootstrap resamples and permutation tests. |

---

## 3. Scientific and Scope Guardrails Audit

| Risk / Guardrail | Verification Check | Status | Verification Detail |
| :--- | :--- | :---: | :--- |
| **No Quantum Advantage Claims** | Check for ungrounded advantage statements. | **PASSED** | Zero claims of quantum advantage; central conclusion explicitly states no practical advantage demonstrated. |
| **No Universal Inferiority Claims** | Check for overgeneralized negative statements. | **PASSED** | Conclusions strictly scoped to evaluated text security corpora, TruncatedSVD representations, and cyclic $ZZFeatureMap$. |
| **No Causal Correlation Claims** | Check state dispersion and document length interpretations. | **PASSED** | Described as "associated with" and "inversely correlated with"; causal claims between length/entropy and F1 avoided. |
| **TruncatedSVD Terminology** | Dimensionality reduction named accurately. | **PASSED** | Consistently referred to as "TruncatedSVD-based dimensionality reduction" throughout the text. |
| **Qubit vs Dimension Distinction** | Classical dimensions and qubits separated. | **PASSED** | Formulated as: "$d$-dimensional classical representation mapped to an $n$-qubit feature map." |
| **State Dispersion vs Diversity** | Single-state entropy decoupled from pairwise kernel diversity. | **PASSED** | Dedicated subsection 4.9.3 explicitly formalizes the distinction and proves inverse correlation ($r \approx -0.80$). |
| **Simulation Runtime Scope** | Wall-clock execution time not generalized to quantum hardware. | **PASSED** | Explicitly stated as classical CPU statevector simulation overhead, with no claims of universal asymptotic complexity laws. |

---

## 4. Structural and Narrative Harmony Audit

- **Research Question Organization**: Results are structured strictly by research question (RQ1–RQ5, Sections 4.2–4.10) rather than experiment chronology. Experiment numbers appear only parenthetically for lineage provenance.
- **Table and Figure Integration**: All 8 planned publication tables (Tables 1–8) and figures (Figures 1–8) are explicitly referenced and discussed in context.
- **Methodology Quarantine**: Methodology descriptions are concise references back to Section 3 without redundant procedural repetition.
- **Draft 1 & Draft 2 Alignment**: Perfect harmony across dataset counts (SMS: 5,572; CEAS: 15,000; MeAJOR: 108,684), split sizes, seed suites ($N=10$), and contribution statements.

---

## 5. Quantitative Summary

- **File Created**: `paper/manuscript/06_results.md`
- **Section Word Count**: **3,892 words** (Target: 3,500–4,500 words) — **PASSED**.
- **Cumulative Manuscript Word Count (Drafts 1 + 2 + 3)**: **10,496 words** (excluding audit metadata).
- **Numerical Discrepancies Found**: **ZERO (0)**.
- **Unverified Claims Found**: **ZERO (0)**.

---

## 6. Final Draft 3 Approval Gate

$$\mathbf{DRAFT\ 3\ STATUS:\ APPROVED\ \&\ FROZEN}$$

Manuscript Draft 3 (Results and Statistical Synthesis) is complete, rigorously audited, and frozen. All empirical evidence is strictly aligned with the frozen project record. Ready to proceed to **Manuscript Draft 4 (Discussion + Limitations + Conclusion)** upon your instruction.
