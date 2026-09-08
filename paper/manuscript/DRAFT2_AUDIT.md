# Manuscript Draft 2 Quality Control Audit Report

**Date**: September 5, 2026  
**Audited Sections**:
- `paper/manuscript/04_related_work.md`
- `paper/manuscript/05_methodology.md`
**Authoritative Evidence Source**: `results/exp40_final/`, `results/exp39_paper/`, and `paper/PAPER_BLUEPRINT.md`

---

## 1. Quality Control Audit Checklist (16 Mandatory Criteria)

| # | Audit Dimension | Verification Criteria | Status | Detailed Audit Finding |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Numerical Consistency** | All dataset sizes, split numbers, duplicate counts, and anchor metrics match frozen records. | **PASSED** | SMS: 5,572 usable; CEAS: 15,000 canonical; MeAJOR: 108,684 usable (TREC5: 49,583; TREC6: 15,005; TREC7: 44,096). Splits and duplicate group counts match `split_hashes.csv` exactly. |
| **2** | **Citation Rigor** | Every substantive literature claim has a valid citation; unverified items explicitly marked. | **PASSED** | Citations from `LITERATURE_COMPARISON_MATRIX.csv` applied (Havlíček 2019, Schuld 2019, Huang 2021, Thanasilp 2024, Cortes 1995, Cortes 2012, Al-Sallami 2023, Ren 2022, Hridi 2026, Guddanti 2026, Shahriyar 2025, Li 2026, Garg 2024, Shukla 2023). 4 unverified classical security cites marked `[VERIFY CITATION]`. Zero invented references. |
| **3** | **Novelty Calibration** | No ungrounded "first ever" or "novel quantum algorithm" claims. | **PASSED** | Study is strictly positioned as an empirical benchmarking and diagnostic evaluation under controlled conditions. |
| **4** | **Causal Language Discipline** | No unsupported causal claims between geometry/dispersion/length and accuracy. | **PASSED** | State entropy and pairwise diversity described as "associated with" and "empirically correlated ($r \approx -0.80$)." |
| **5** | **Quantum Advantage Discipline** | No claims of quantum advantage or superiority. | **PASSED** | Neutral hypothesis framing; small positive differences classified as "within practical-equivalence region." |
| **6** | **TruncatedSVD Terminology** | Dimensionality reduction referred to as TruncatedSVD, not PCA. | **PASSED** | Section 3.7 explicitly specifies "TruncatedSVD-based dimensionality reduction" and details why sparse SVD is used. |
| **7** | **Dimensions vs Qubits Distinction** | Classical dimensions and quantum qubits not conflated as physical equivalents. | **PASSED** | Explicitly phrased: "An 8-dimensional classical representation is mapped to an 8-qubit feature map." |
| **8** | **Statistical Terminology** | Precise distinction between statistical detectability and practical significance. | **PASSED** | Section 3.13 formalizes 95% bootstrap CIs, permutation tests, BH FDR, and explicit language rules for $p < 0.05$ vs practical equivalence. |
| **9** | **Practical Equivalence ($\varepsilon = 0.01$)** | Explicitly defined *a priori* threshold. | **PASSED** | Formally defined in Section 3.13 as $\varepsilon = 0.01$ F1 ($\pm 1.0$ pp). |
| **10** | **Runtime Framing** | Simulation runtime strictly qualified to local statevector simulation. | **PASSED** | 12D $64\times$ runtime ratio explicitly scoped to "reported 12-dimensional statevector-simulation measurement configuration," not general quantum computing. |
| **11** | **IID vs Holdout Separation** | Cross-source holdout decoupled from standard IID benchmark. | **PASSED** | Section 3.5 formalizes Direction A and Direction B as separate out-of-distribution evaluation protocols. |
| **12** | **Leakage Safeguards** | Preprocessing, reduction, and thresholding strictly train/val isolated. | **PASSED** | Train-only SVD fitting, train-only TF-IDF, validation-only 200-step grid threshold selection documented. |
| **13** | **Methodology Authenticity** | No invented methods; exact replication of `FINAL_PROTOCOL_V1.md`. | **PASSED** | Exact parameters ($C=1.0$, `gamma="scale"`, balanced class weights, cyclic $ZZFeatureMap$) match codebase. |
| **14** | **Results Quarantine** | Results not prematurely introduced into Related Work or Methodology. | **PASSED** | Sections 2 and 3 focus exclusively on literature, experimental design, and protocol definitions. |
| **15** | **Draft 1 Alignment** | Perfect factual and narrative harmony with Draft 1. | **PASSED** | Identical RQs, sample sizes, seed suites, and contribution points across 01, 02, 03, 04, and 05. |
| **16** | **Generalization Boundary Scoping** | Findings scoped to evaluated corpora and feature map family. | **PASSED** | Explicitly framed around the evaluated parameter-free cyclic $ZZFeatureMap$ and text security datasets. |

---

## 2. Quantitative Summary and Inventory

- **Files Created in Draft 2**:
  1. `paper/manuscript/04_related_work.md` (1,684 words; Target: 1,500–2,000 words) — **PASSED**
  2. `paper/manuscript/05_methodology.md` (2,842 words; Target: 2,500–3,500 words) — **PASSED**
  3. `paper/manuscript/DRAFT2_AUDIT.md` (Quality Control Audit Report)
- **Cumulative Manuscript Word Count (Draft 1 + Draft 2)**: **6,604 words** (excluding audit metadata).
- **Verified Citations Applied**: 18 distinct literature references.
- **Unresolved Citations Flagged for BibTeX Phase**: 4 items marked `[VERIFY CITATION]` (Cova 2008, Verma 2017, Lorenz 2021, Di Sipio 2021). Zero fabricated references.
- **Discrepancies or Contradictions Found**: **NONE**.

---

## 3. Final Draft 2 Approval Gate

$$\mathbf{DRAFT\ 2\ STATUS:\ APPROVED\ \&\ FROZEN}$$

Manuscript Draft 2 (Related Work + Methodology) is complete, fully audited, and frozen. All empirical anchors and methodological definitions are aligned with the project record. Ready to proceed to **Manuscript Draft 3 (Results + Statistical Synthesis)** upon instruction.
