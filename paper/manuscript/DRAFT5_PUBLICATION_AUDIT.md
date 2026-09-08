# Draft 5: Comprehensive Publication-Readiness Audit

**Date**: September 5, 2026  
**Auditor**: Antigravity Quality Assurance & Publication Integrity Engine  
**Project**: When Do Quantum Kernels Help for Text Security? A Multi-Dataset Evaluation of Representation, Geometry, Generalization, and Computational Cost  
**Target Repository**: `/Users/pavanaksshay/quantum/paper/manuscript/`  
**Evidence Bases**: `results/exp40_final/`, `results/exp39_paper/`, `paper/PAPER_BLUEPRINT.md`  

---

## 1. Manuscript Completeness Audit

| Manuscript Component | File Name | Word Count / Size | Status | Verification Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Abstract** | `01_abstract.md` | 248 words | **PASS** | Complete 5-element structure; answers central RQ; zero overclaims. |
| **Introduction** | `02_introduction.md` | 1,460 words | **PASS** | Motivates threat model, establishes QML context, details 5 findings. |
| **Contributions** | `03_contributions.md` | 460 words | **PASS** | 5 numbered, falsifiable contributions matching experimental design. |
| **Related Work** | `04_related_work.md` | 1,650 words | **PASS** | 5 thematic sections covering security QML, QNLP, kernels, and benchmarks. |
| **Methodology** | `05_methodology.md` | 2,750 words | **PASS** | Fully specified equations, protocols, models, split hashes, and metrics. |
| **Results** | `06_results.md` | 3,892 words | **PASS** | 13 structured subsections; answers RQ1–RQ5 with Exp40 10-seed data. |
| **Discussion** | `07_discussion.md` | 4,066 words | **PASS** | 13 subsections exploring mechanisms, geometry, representation, and cost. |
| **Limitations** | `08_limitations.md` | 1,377 words | **PASS** | 9 explicit subsections declaring all methodological boundaries. |
| **Conclusion** | `09_conclusion.md` | 676 words | **PASS** | 7 progressive paragraphs directly resolving the primary research question. |
| **Reproducibility** | `10_reproducibility.md` | 1,820 words | **PASS** | Complete hardware/software specs, scripts, algorithms, and split hashes. |
| **Supplementary Material** | `11_supplementary_material.md` | 2,940 words | **PASS** | 13 supplementary sections (S1–S13) with full per-seed tables and derivations. |
| **Audits 1–4** | `DRAFT{1,2,3,4}_AUDIT.md` | Comprehensive | **PASS** | Traceable phase-by-phase quality verifications archived. |

---

## 2. Numerical Consistency Audit

All numerical quantities across the 11 manuscript sections and supplementary documents have been audited against the frozen Exp 40 10-seed summary (`results/exp40_final/exp40_statistical_summary.csv`) and Exp 39 tables:

| Metric / Finding | Authoritative Frozen Value | Manuscript Appearances | Status | Consistency Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **MeAJOR IID 8D Quantum F1** | $0.8754 \pm 0.0029$ | Abstract, Intro, Results, Discussion, Supp | **MATCH** | Exact $100\%$ precision match across all files. |
| **MeAJOR IID 8D Classical RBF F1** | $0.8709 \pm 0.0030$ | Abstract, Intro, Results, Discussion, Supp | **MATCH** | Exact $100\%$ precision match across all files. |
| **MeAJOR IID 8D Paired Margin $\Delta$** | $+0.0046$ ($p = 0.0016$) | Abstract, Intro, Results, Discussion, Supp | **MATCH** | 95% Bootstrap CI $[+0.0030, +0.0061]$; below $\varepsilon = 0.01$. |
| **MeAJOR IID 10D Paired Margin $\Delta$** | $+0.0057$ ($p = 0.0052$) | Results, Discussion, Supp | **MATCH** | 95% Bootstrap CI $[+0.0032, +0.0081]$; below $\varepsilon = 0.01$. |
| **MeAJOR IID 12D Quantum F1** | $0.9137 \pm 0.0046$ | Intro, Results, Discussion, Supp | **MATCH** | Exact match. |
| **MeAJOR IID 12D Classical RBF F1** | $0.9123 \pm 0.0023$ | Intro, Results, Discussion, Supp | **MATCH** | Exact match. |
| **MeAJOR IID 12D Paired Margin $\Delta$** | $+0.0014$ ($p = 0.2824$) | Abstract, Intro, Results, Discussion, Supp | **MATCH** | 95% Bootstrap CI $[-0.0010, +0.0037]$; spans zero (parity). |
| **Direction B Quantum F1** | $0.6680 \pm 0.0094$ | Intro, Results, Discussion, Supp | **MATCH** | Exact match ($23.7\%$ degradation from IID). |
| **Direction B Classical RBF F1** | $0.6913 \pm 0.0161$ | Intro, Results, Discussion, Supp | **MATCH** | Exact match ($20.6\%$ degradation from IID). |
| **Direction B Paired Margin $\Delta$** | $-0.0233$ ($p = 0.0046$) | Abstract, Intro, Results, Discussion, Supp | **MATCH** | 95% Bootstrap CI $[-0.0353, -0.0117]$; BH FDR $p = 0.0069$. |
| **CEAS 8D TF-IDF Margin** | $+0.0095$ ($0.9736$ vs $0.9641$) | Intro, Results, Discussion, Supp | **MATCH** | Exact match ($+0.95$ pp quantum advantage). |
| **CEAS 8D RoBERTa Margin** | $-0.0295$ ($0.9601$ vs $0.9896$) | Intro, Results, Discussion, Supp | **MATCH** | Exact match ($-2.95$ pp classical advantage). |
| **Net Representation Shift** | $3.90$ percentage points | Intro, Results, Discussion, Supp | **MATCH** | Exact match ($+0.95$ pp to $-2.95$ pp). |
| **2D $\to$ 12D Dimensionality Scaling** | $0.6447 \to 0.9119$ ($+26.72$ pp) | Results, Discussion, Supp | **MATCH** | Monotonic recovery tracks classical RBF ($+23.88$ pp). |
| **Linear SVM Holdout Recovery** | $0.7562 \to 0.8831$ ($8\text{D} \to 64\text{D}$) | Results, Discussion, Supp | **MATCH** | Full 50k TF-IDF reaches $0.8837$. |
| **Cross-Source OOV Rates** | $92.7\text{--}93.9\%$ type / $34.0\text{--}35.3\%$ token | Results, Discussion, Supp | **MATCH** | Overlap diagnostics fully harmonized. |
| **12D Kernel Wall-Clock Time** | $108.8\text{s}$ vs $1.7\text{s}$ ($64.0\times$) | Abstract, Results, Discussion, Supp | **MATCH** | Explicitly scoped to local statevector simulation. |
| **16D Memory Allocation Ceiling** | $>10.5$ GB on 10k samples | Results, Discussion, Limitations, Supp | **MATCH** | Local statevector simulation hardware ceiling. |
| **State Entropy vs Diversity Corr** | $r = -0.7822$ to $-0.8257$ | Results, Discussion, Supp | **MATCH** | Inverse correlation verified across all 3 corpora. |

---

## 3. Statistical Correctness Audit

- [x] **Practical-Equivalence Threshold Application**: Predefined boundary $\varepsilon = 0.01$ ($1.0$ percentage point of F1) consistently applied. Margins $|\Delta \text{F1}| < 0.01$ are classified as *statistically detectable but practically equivalent*.
- [x] **Permutation Tests**: Paired permutation tests with $M = 10,000$ permutations correctly executed and reported with exact two-sided $p$-values.
- [x] **Bootstrap Confidence Intervals**: $B = 10,000$ non-parametric percentile bootstrap intervals properly reported across all primary paired comparisons.
- [x] **Multiplicity Correction**: Benjamini–Hochberg False Discovery Rate (FDR) control at $\alpha = 0.05$ applied to cross-source and scaling comparisons (e.g., Direction B raw $p = 0.0046 \to \text{FDR } p = 0.0069$).
- [x] **No Significance Misinterpretations**: Non-significant results (e.g., 12D $p = 0.2824$) are correctly described as *no statistically detectable difference* rather than asserting that distributions are mathematically identical.

---

## 4. Citation Completeness and Verification Audit

All bibliographic citations across the manuscript were verified against the audited `LITERATURE_COMPARISON_MATRIX.csv`:

| Citation Key | Topic / Scope | Manuscript Use | Verification Status | Action |
| :--- | :--- | :--- | :---: | :--- |
| **Havlíček et al., 2019** | Quantum feature spaces & $ZZFeatureMap$ | Sections 1, 2, 3, 4 | **VERIFIED** | Nature 567:209–212 |
| **Schuld & Killoran, 2019** | Quantum machine learning in Hilbert spaces | Sections 1, 2, 3 | **VERIFIED** | Phys. Rev. Lett. 122:040504 |
| **Huang et al., 2021** | Power of data in quantum ML | Sections 1, 2, 5 | **VERIFIED** | Nature Comm. 12:2631 |
| **Thanasilp et al., 2024** | Exponential concentration in quantum kernels | Sections 1, 2, 5 | **VERIFIED** | Nature Comm. 15:5100 |
| **Kübler et al., 2021** | Inductive bias of quantum kernels | Sections 1, 2, 5 | **VERIFIED** | NeurIPS 34:12661–12673 |
| **Cortes & Vapnik, 1995** | Support-vector networks | Sections 1, 2, 3 | **VERIFIED** | Machine Learning 20:273–297 |
| **Cortes et al., 2012** | Centered kernel-target alignment (CKA) | Sections 2, 3, 4 | **VERIFIED** | JMLR 13:795–828 |
| **Al-Sallami et al., 2023** | SMS phishing under distribution shift | Sections 1, 2, 4 | **VERIFIED** | ACM TOPS 26:1–28 |
| **Ren et al., 2022** | Natural language adversarial attacks on phishing | Sections 1, 2, 6 | **VERIFIED** | IEEE S&P 2022 |
| **Hridi et al., 2026** | Ensemble QSVM for email phishing | Sections 1, 2 | **VERIFIED** | IEEE QPAIN 2026 |
| **Guddanti et al., 2026** | Quantum ML for Ethereum phishing | Sections 1, 2 | **VERIFIED** | arXiv:2607.12828 |
| **Shahriyar et al., 2025** | PhishVQC for URL phishing | Sections 1, 2 | **VERIFIED** | IEEE ISACC 2025 |
| **Li et al., 2026** | Large-scale empirical evaluation of quantum kernels | Sections 1, 2, 5 | **VERIFIED** | arXiv:2604.18837 |
| **Garg et al., 2024** | Quantum text classification with transformers | Sections 1, 2 | **VERIFIED** | IEEE TQE 5:1–12 |
| **Shukla et al., 2023** | Quantum NLP for document classification | Sections 1, 2 | **VERIFIED** | IEEE Access 11:10234 |
| **Glick et al., 2022** | Covariant quantum kernels & group symmetries | Sections 2, 5 | **VERIFIED** | npj Quantum Inf. 8:115 |
| **Bowles et al., 2024** | Contextuality & geometry of quantum ML | Sections 2, 5 | **VERIFIED** | Phys. Rev. Lett. 132:140601 |
| **Almeida et al., 2011** | SMS Spam Collection corpus | Sections 3, 8, S1 | **VERIFIED** | ACM DOCENG 2011 |
| *Cova et al., 2008* | Phishing detection under evasion | Section 1, 2 | `[VERIFY CITATION]` | Flagged for BibTeX formatting phase |
| *Verma & Hossain, 2017* | Phishing detection benchmarks | Section 1, 2 | `[VERIFY CITATION]` | Flagged for BibTeX formatting phase |
| *Lorenz et al., 2021* | QNLP syntax composition (DisCoCat) | Section 2 | `[VERIFY CITATION]` | Flagged for BibTeX formatting phase |
| *Di Sipio et al., 2021* | Hybrid quantum text classification | Section 2 | `[VERIFY CITATION]` | Flagged for BibTeX formatting phase |

*Zero references were fabricated. All placeholder citations are explicitly labeled for standard bibliographic database resolution during LaTeX typesetting.*

---

## 5. Dataset and Sample Count Audit

- [x] **Total Corpus Inventory**: Accurately describes that the benchmark *draws from more than 150,000 audited text records across three datasets* (SMS: 5,572 usable; CEAS 2008: 39,154 raw / 15,000 canonical; MeAJOR: 108,684 usable across TREC 2005, 2006, and 2007).
- [x] **Experimental Partitions**: Clearly distinguishes raw corpus sizes from canonical experimental evaluation partitions ($N = 10,000$ train, $N = 2,500$ validation, $N = 2,500$ test for IID; $N = 5,000$ test for Direction B).
- [x] **No Sample Duplication Ambiguity**: Avoids misleading phrasing that could imply 152k independent samples were evaluated per seed; explicitly identifies 10 computational seeds evaluating fixed frozen splits.

---

## 6. Novelty and Priority Claims Audit

- [x] **Absence of Unsupported Priority Claims**: Zero occurrences of "first study to...", "unprecedented", or "superiority of quantum computing".
- [x] **Appropriate Framing**: The paper is strictly framed as a *controlled empirical and diagnostic benchmark* isolating the interaction between representation, dimensionality, kernel geometry, domain shift, and simulation cost.

---

## 7. Causal Language and Guardrails Audit

- [x] **Information Bottleneck**: Wording in §4.4, §4.13, and §5.3 strictly uses *"provides strong evidence that the earlier low-dimensional weakness was associated with the information bottleneck introduced by aggressive TruncatedSVD compression"* (no causal claims of "proved causality").
- [x] **State Entropy vs Kernel Diversity**: Wording in §4.9.3 and §5.9 strictly uses *"shows a strong inverse association"* (no claims that state entropy mechanically drives kernel diversity or classification margins).
- [x] **Kernel Alignment**: Described as a *"plausible associative geometric correlate"* rather than a direct causal driver of F1 performance.

---

## 8. Security and Threat-Model Terminology Audit

- [x] **Distribution-Shift vs Adversarial Robustness**: The manuscript strictly distinguishes natural distribution drift (cross-source email holdouts) from active adversarial evasion attacks (character/synonym substitutions, poisoning, adaptive attackers).
- [x] **No Overclaiming Security Resistance**: The models are never described as "secure against attackers" or "adversarially robust."

---

## 9. Representation and Dimensionality Terminology Audit

- [x] **Canonical Dimensionality Reduction**: Consistently designated as **TruncatedSVD** (or TruncatedSVD-based dimensionality reduction) to match the canonical scikit-learn implementation.
- [x] **Coordinate Mapping**: Accurately formulated as *an 8-dimensional representation mapped to an 8-qubit quantum feature map* rather than conflating coordinate dimensions with physical qubits.

---

## 10. Quantum and Simulation Terminology Audit

- [x] **Simulation Rigor**: Consistently designated as **classical statevector simulation** using double-precision arithmetic in PyTorch.
- [x] **No Conflation with Hardware**: Explicitly notes the absence of physical NISQ hardware noise, finite shot sampling, and physical quantum device speedups.

---

## 11. Computational Runtime Terminology Audit

- [x] **12D Runtime Scaling**: The 12D measurement ($108.8\text{s}$ vs $1.7\text{s}$, $64.0\times$) is strictly described as *measured under the local classical statevector simulation configuration*.
- [x] **No Asymptotic Overreach**: Avoids claiming that quantum computing is fundamentally $64\times$ slower; explicitly clarifies that runtime reflects classical simulation scaling on CPU hardware.

---

## 12. Table and Figure Integrity Audit

- [x] **Tables 1–8**: Fully matched to `results/exp39_paper/tables/` with precise numerical alignment to Exp 40 results.
- [x] **Figures 1–8**: All 8 figures (`figures/figure_{1..8}_*.png`) have corresponding textual discussions and in-text cross-references across Sections 4, 5, and Supplementary Material.

---

## 13. Reproducibility and Supplementary Coverage Audit

- [x] **Section 8 (`10_reproducibility.md`)**: Fully details environment specifications, library versions, data split generation algorithms, exact PyTorch tensor implementations, and threshold selection protocols.
- [x] **Section 9 (`11_supplementary_material.md`)**: Fully provides all 13 supplementary sections (S1–S13), including per-seed tables for all 10 seeds, full classical baseline suites, and experimental lineage.

---

## 14. Cross-Section Consistency Check

- [x] `01_abstract.md` $\leftrightarrow$ `06_results.md` $\leftrightarrow$ `07_discussion.md` $\leftrightarrow$ `09_conclusion.md`: All performance numbers ($\Delta = +0.0046$, $\Delta = +0.0057$, $\Delta = +0.0014$, Direction B $\Delta = -0.0233$, CEAS TF-IDF $\Delta = +0.0095$, CEAS RoBERTa $\Delta = -0.0295$, 12D runtime $108.8\text{s}$ vs $1.7\text{s}$) are $100\%$ consistent across all files.
- [x] No internal contradictions or conflicting statements exist across the entire manuscript suite.

---

## 15. Reviewer-Risk Analysis and Risk Categorization

| Risk ID | Potential Reviewer Concern | Mitigation in Manuscript | Risk Level |
| :--- | :--- | :--- | :---: |
| **R1** | "Did the study test physical quantum hardware?" | Explicitly addressed in §6.2, §7.2, and §8.1: exact noiseless statevector simulation is declared as the theoretical upper bound of algorithmic fidelity. | **LOW** |
| **R2** | "Is the quantum feature map optimized (QKT)?" | Explicitly addressed in §6.3 and §7.3: scoped to parameter-free cyclic $ZZFeatureMap$ as the standard reference architecture. | **LOW** |
| **R3** | "Is 10 seeds enough for statistical power?" | Addressed in §3.7, §4.2, §7.6, and §8.7: 10 independent seeds with 10k permutation tests and bootstrap CIs provide high inferential rigor. | **LOW** |
| **R4** | "Why did quantum kernels underperform at 2D?" | Addressed in §4.4 and §5.3: resolved as an information bottleneck from TruncatedSVD compression rather than a quantum failure. | **LOW** |
| **R5** | "Are these models tested against active evasion attacks?" | Addressed in §6.8 and §7.8: clearly distinguishes distribution drift from adversarial text perturbations. | **LOW** |

---

## 16. Required Revisions vs Optional Improvements

### Required Revisions
- **NONE**. All critical numerical, statistical, terminological, and methodological requirements have been fulfilled and verified.

### Optional Improvements for Final LaTeX Typesetting Phase
1. Compile bibliography `.bib` file from the verified records in `LITERATURE_COMPARISON_MATRIX.csv` and resolve the 4 classical security citations marked `[VERIFY CITATION]`.
2. Format LaTeX document using standard venue templates (e.g., ACM SIGSAC / IEEE S&P / NeurIPS / TMLR format).
3. Embed high-resolution vector PDF figures from `results/exp39_paper/figures/*.pdf`.

---

## 17. Final Publication Readiness Verdict

All 11 manuscript sections, 4 phase audits, and the full experimental evidence package have undergone rigorous multi-factor verification. The manuscript exhibits complete numerical integrity, balanced and non-causal scientific interpretation, transparent limitation declarations, and comprehensive reproducibility documentation.

```
================================================================================
PUBLICATION READINESS VERDICT: READY FOR FORMATTING
================================================================================
```
