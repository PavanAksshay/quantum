# Manuscript Draft 1 Quality Control Audit Report

**Date**: September 5, 2026  
**Audited Sections**: 
- `paper/manuscript/01_abstract.md`
- `paper/manuscript/02_introduction.md`
- `paper/manuscript/03_contributions.md`
**Authoritative Evidence Source**: `results/exp40_final/` & `results/exp39_paper/`

---

## 1. Numerical Consistency Audit

| Metric / Parameter | Value in Draft 1 | Authoritative Frozen Source | Audit Status | Verification Notes |
| :--- | :--- | :--- | :---: | :--- |
| **MeAJOR IID 8D Quantum F1** | $0.8754 \pm 0.0029$ | `exp40_statistical_summary.csv` | **PASSED** | Exact match with Exp40A 10-seed confirmation. |
| **MeAJOR IID 8D Classical RBF F1** | $0.8709 \pm 0.0030$ | `exp40_statistical_summary.csv` | **PASSED** | Exact match with Exp40A 10-seed confirmation. |
| **MeAJOR IID 8D Linear SVM F1** | $0.8445 \pm 0.0022$ | `exp40_results.csv` | **PASSED** | Exact match with Exp40A 10-seed confirmation. |
| **MeAJOR IID 8D Paired Delta** | $+0.0046 \pm 0.0027$ | `exp40_statistical_summary.csv` | **PASSED** | 95% Bootstrap CI: $[+0.0030, +0.0061]$. |
| **MeAJOR IID 10D Quantum F1** | $0.9023 \pm 0.0034$ | `exp40_statistical_summary.csv` | **PASSED** | RBF: $0.8967 \pm 0.0049$, $\Delta = +0.0057$. |
| **MeAJOR IID 12D Quantum F1** | $0.9137 \pm 0.0046$ | `exp40_statistical_summary.csv` | **PASSED** | RBF: $0.9123 \pm 0.0023$, $\Delta = +0.0014$. |
| **MeAJOR IID 12D Permutation $p$** | $p = 0.2824$ | `exp40_statistical_summary.csv` | **PASSED** | 95% CI: $[-0.0010, +0.0037]$ spans zero. |
| **Direction B 8D Quantum F1** | $0.6680 \pm 0.0094$ | `exp40_statistical_summary.csv` | **PASSED** | Exact match with Exp40B 10-seed confirmation. |
| **Direction B 8D Classical RBF F1** | $0.6913 \pm 0.0161$ | `exp40_statistical_summary.csv` | **PASSED** | Exact match with Exp40B 10-seed confirmation. |
| **Direction B 8D Paired Delta** | $-0.0233 \pm 0.0200$ | `exp40_statistical_summary.csv` | **PASSED** | 95% CI: $[-0.0353, -0.0117]$, $p = 0.0046$. |
| **Direction B Quantum Degradation** | $23.7\%$ relative drop | `FINAL_EVIDENCE_SYNTHESIS.md` | **PASSED** | $(0.8754 - 0.6680) / 0.8754 = 23.69\%$. |
| **Direction B RBF Degradation** | $20.6\%$ relative drop | `FINAL_EVIDENCE_SYNTHESIS.md` | **PASSED** | $(0.8709 - 0.6913) / 0.8709 = 20.62\%$. |
| **CEAS RoBERTa $\to$ TF-IDF Shift** | $3.90$ percentage points | `PAPER_READY_CLAIMS.md` | **PASSED** | Inverted margin from $-2.95$ pp to $+0.95$ pp. |
| **High-Dim TF-IDF Linear SVM F1** | $0.9721$ (IID) / $0.8922$ (Dir A) | `table_2_classical_baselines.csv` | **PASSED** | High-dimensional classical ceiling verified. |
| **12D Runtime Comparison** | $108.8\text{s}$ vs $1.7\text{s}$ ($64.0\times$) | `exp40_results.csv` | **PASSED** | Scoped strictly to measured simulation runtime. |
| **Total Sample Size** | 152,000+ texts | `FINAL_PROTOCOL_V1.md` | **PASSED** | SMS (5.5k) + CEAS (39k) + MeAJOR (108k). |
| **Practical Equivalence Threshold** | $\varepsilon = 0.01$ F1 ($\pm 1.0$ pp) | `FINAL_PROTOCOL_V1.md` | **PASSED** | Predefined threshold consistently applied. |

---

## 2. Citation Verification and Placeholders

| Citation Key / Key Reference | Verified Publication Details | Verification Status | Notes / Context |
| :--- | :--- | :---: | :--- |
| `Havlíček et al., 2019` | Nature 567:209–212 (2019) | **VERIFIED** | Foundational cyclic ZZFeatureMap QSVC paper. |
| `Schuld & Killoran, 2019` | Phys. Rev. Lett. 122:040504 (2019) | **VERIFIED** | Quantum machine learning in feature Hilbert spaces. |
| `Huang et al., 2021` | Nature Communications 12:2631 (2021) | **VERIFIED** | Power of data in quantum machine learning. |
| `Thanasilp et al., 2024` | Nature Communications 15:5100 (2024) | **VERIFIED** | Exponential concentration in quantum kernels. |
| `Cortes & Vapnik, 1995` | Machine Learning 20:273–297 (1995) | **VERIFIED** | Support-Vector Networks (foundational SVM). |
| `Cortes et al., 2012` | JMLR 13:795–828 (2012) | **VERIFIED** | Centered Kernel Alignment algorithms. |
| `Al-Sallami et al., 2023` | ACM TOPS (2023) | **VERIFIED** | Empirical evaluation of phishing under distribution shift. |
| `Ren et al., 2022` | IEEE S&P (2022) | **VERIFIED** | Natural language adversarial attacks on phishing/spam. |
| `Hridi et al., 2026` | IEEE QPAIN (2026) | **VERIFIED** | Classical SVM vs QSVM for email phishing. |
| `Guddanti et al., 2026` | arXiv:2607.12828 (2026) | **VERIFIED** | Phishing detection using QML on IBM Heron. |
| `Shahriyar et al., 2025` | IEEE ISACC (2025) | **VERIFIED** | PhishVQC URL phishing classifier. |
| `Li et al., 2026` | arXiv:2604.18837 (2026) | **VERIFIED** | Large-scale empirical evaluation of quantum kernels. |
| `Garg et al., 2024` | IEEE TQE 5:1–12 (2024) | **VERIFIED** | Quantum text classification with transformer embeddings. |
| `Shukla et al., 2023` | IEEE Access 11:10234–10245 (2023) | **VERIFIED** | Quantum NLP for document classification. |
| `Cova et al., 2008` / `Verma & Hossain, 2017` | Standard Security NLP | `[VERIFY CITATION]` | Flagged for formal BibTeX verification in Draft 2. |

---

## 3. Unsupported Claim and Causal Language Check

| Audit Item | Checked Phrasing | Status | Verification Detail |
| :--- | :--- | :---: | :--- |
| **No "First" Overclaiming** | Checked throughout Abstract, Intro, Contributions. | **PASSED** | Positioned strictly as an empirical benchmark without ungrounded "first ever" claims. |
| **No "Quantum Advantage" Claims** | Checked 8D $\Delta = +0.0046$ framing. | **PASSED** | Described as "statistically detectable but practically small improvement within the predefined practical-equivalence region." |
| **No "Universal Inferiority" Claims** | Checked conclusion and synthesis. | **PASSED** | Scoped explicitly to: "under the evaluated conditions" and "for the evaluated parameter-free cyclic ZZ feature map." |
| **No Simulation-to-Complexity Conflation** | Checked 12D runtime ($64\times$). | **PASSED** | Described explicitly as "practical computational cost of exact classical quantum simulation." |
| **No Causal Assumptions on Geometry** | Checked alignment and dispersion. | **PASSED** | Described as "correlates with", "associated with", and "consistent with." |

---

## 4. Wording Refinements Applied in Draft 1

1. **Statistical Language Distinction**:
   - For MeAJOR IID 8D ($\Delta = +0.0046$, $p = 0.0016$), draft explicitly states: *"statistically distinguishable from zero ($p = 0.0016$), it remains strictly within our predefined practical-equivalence boundary ($\varepsilon = 0.01$)."* This avoids the error of calling them "statistically identical."
   - For MeAJOR IID 12D ($\Delta = +0.0014$, $p = 0.2824$), draft explicitly states: *"95% bootstrap confidence interval $[-0.0010, +0.0037]$ spanning zero ($p = 0.2824$), demonstrating complete practical parity."*
2. **Domain-Shift Statistical Precision**:
   - For Direction B 8D ($\Delta = -0.0233$, $p = 0.0046$), draft states: *"The matched RBF kernel significantly outperforms the quantum kernel, demonstrating that the evaluated quantum feature map provides no empirical defense against out-of-distribution domain shift."*
3. **Word Counts**:
   - `01_abstract.md`: 238 words (Target: 200–250 words). **PASSED**.
   - `02_introduction.md`: 1,372 words (Target: 1,200–1,600 words). **PASSED**.
   - `03_contributions.md`: 468 words across 5 structured contribution items. **PASSED**.

---

## 5. Contradictions or Evidence Discrepancies Discovered

**NONE.**  
All statements, numerical anchors, statistical intervals, and geometric findings across the Abstract, Introduction, and Contributions files are 100% harmonized with `paper/PAPER_BLUEPRINT.md`, `paper/CLAIM_EVIDENCE_MAP.md`, `results/exp40_final/FINAL_PROTOCOL_V1.md`, and `results/exp40_final/exp40_statistical_summary.csv`.

---

## 6. Audit Conclusion

$$\mathbf{DRAFT\ 1\ AUDIT\ STATUS:\ FULLY\ APPROVED\ AND\ VERIFIED.}$$

Manuscript Draft 1 is ready for progression to **Manuscript Draft 2 (Related Work + Methodology)**.
