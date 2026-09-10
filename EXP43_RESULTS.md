# Experiment 43: Representation-Aware Quantum Encoding Diagnostic Study Results

**Document Type**: Diagnostic Empirical Report & Mechanism Synthesis  
**Experiment Identifier**: `EXP-43-ENCODING-DIAGNOSTIC`  
**Status**: DIAGNOSTIC STUDY COMPLETE  

---

## 1. Executive Summary & Core Diagnostic Questions

Experiment 43 conducted a controlled 10-seed diagnostic ablation study ($70$ quantum runs) on the **SMS Spam + MPNet (8D)** benchmark to resolve why parameter-free cyclic `ZZFeatureMap` fidelity kernels suffer severe classification failure relative to Classical RBF ($0.3756$ vs $0.9045$, $\Delta\text{F1} = -0.5288$).

### Answers to Core Diagnostic Questions:
1. **“Is the MPNet quantum-kernel failure sensitive to angular mapping?”**
   - **YES**. Angular mapping choices materially alter quantum Gram matrix geometry and classification F1. Canonical $[0, \pi]$ achieves the highest performance ($0.3756 \pm 0.113$), whereas expanding the angular range to $[0, 2\pi]$ ($0.2732 \pm 0.051$) or symmetric $[-\pi, \pi]$ ($0.2099 \pm 0.028$) exacerbates phase-wrapping and degrades F1 by up to $-16.57\text{ percentage points}$.
2. **“Is the MPNet quantum-kernel failure sensitive to feature-map depth?”**
   - **YES**. A strong, monotonic depth penalty was uncovered: reducing depth from 3 layers to 1 layer yields a **$+21.30\text{ percentage point}$ F1 improvement** (1-Layer: $0.5334 \pm 0.102$, 2-Layer: $0.3756 \pm 0.113$, 3-Layer: $0.3204 \pm 0.060$).
3. **“Do geometry changes track performance changes?”**
   - **YES**. Quantum-vs-RBF Gram Pearson correlation ($r_{\text{geom}}$) strongly and significantly correlates with Quantum F1 ($r = 0.7707, p = 6.04 \times 10^{-15}$). Preserving geometric congruence with classical Euclidean pairwise distances is strongly associated with quantum classification performance.
4. **“Did any tested parameter-free encoding fully recover the MPNet failure?”**
   - **NO**. While 1-layer ZZFeatureMap improves F1 to $0.5334$, a statistically significant classical advantage remains ($\Delta\text{F1} = -0.3710, p_{\text{BH}} = 0.0023$) relative to Classical RBF ($0.9045$).

---

## 2. Statistical Evidence Table (All 7 Predefined Conditions, N=10 Seeds)

| Condition ID | Ablation Family | Angle Mapping | Depth ($D$) | Quantum F1 (Mean ± Std) | Classical RBF F1 (Mean ± Std) | Paired $\Delta\text{F1}$ [95% Student-t CI] | Permutation $p_{\text{raw}}$ | BH-FDR $p_{\text{adj}}$ | Seed Wins (Q / RBF / Tie) | Equivalence Verdict ($\epsilon = \pm 0.01$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A1 / B2** | **Canonical Ref** | $[0, \pi]$ | 2 | $0.3756 \pm 0.113$ | $0.9045 \pm 0.051$ | **-0.5288** $[-0.6233, -0.4344]$ | $0.0018$ | $0.0023$ | $0\ /\ 10\ /\ 0$ | **D. Classical advantage ($|\Delta| \ge 0.01$)** |
| **A2** | **Angle Mapping** | $[-\pi, \pi]$ | 2 | $0.2099 \pm 0.028$ | $0.9045 \pm 0.051$ | **-0.6946** $[-0.7359, -0.6532]$ | $0.0023$ | $0.0023$ | $0\ /\ 10\ /\ 0$ | **D. Classical advantage ($|\Delta| \ge 0.01$)** |
| **A3** | **Angle Mapping** | $[0, 2\pi]$ | 2 | $0.2732 \pm 0.051$ | $0.9045 \pm 0.051$ | **-0.6312** $[-0.6910, -0.5715]$ | $0.0022$ | $0.0023$ | $0\ /\ 10\ /\ 0$ | **D. Classical advantage ($|\Delta| \ge 0.01$)** |
| **A4** | **Angle Mapping** | Normal CDF | 2 | $0.2863 \pm 0.064$ | $0.9045 \pm 0.051$ | **-0.6181** $[-0.6753, -0.5609]$ | $0.0014$ | $0.0023$ | $0\ /\ 10\ /\ 0$ | **D. Classical advantage ($|\Delta| \ge 0.01$)** |
| **B1** | **Depth Scaling** | $[0, \pi]$ | 1 | **$0.5334 \pm 0.102$** | $0.9045 \pm 0.051$ | **-0.3710** $[-0.4491, -0.2930]$ | $0.0020$ | $0.0023$ | $0\ /\ 10\ /\ 0$ | **D. Classical advantage ($|\Delta| \ge 0.01$)** |
| **B2** | **Depth Scaling** | $[0, \pi]$ | 2 | $0.3756 \pm 0.113$ | $0.9045 \pm 0.051$ | **-0.5288** $[-0.6233, -0.4344]$ | $0.0015$ | $0.0023$ | $0\ /\ 10\ /\ 0$ | **D. Classical advantage ($|\Delta| \ge 0.01$)** |
| **B3** | **Depth Scaling** | $[0, \pi]$ | 3 | $0.3204 \pm 0.060$ | $0.9045 \pm 0.051$ | **-0.5841** $[-0.6486, -0.5196]$ | $0.0016$ | $0.0023$ | $0\ /\ 10\ /\ 0$ | **D. Classical advantage ($|\Delta| \ge 0.01$)** |

---

## 3. Sub-Analysis 1: Angular Mapping Diagnostics

| Condition | Angular Interval | Quantum F1 (Mean ± Std) | $\Delta\text{F1}$ vs RBF | Target Label Alignment | Q/RBF Gram Pearson $r$ | Geometric Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **A1 (Canonical)** | $[0, \pi]$ | **0.3756 ± 0.113** | -0.5288 | 0.1085 ± 0.002 | 0.1232 ± 0.016 | Baseline phase scaling across half circle. |
| **A2 (Symmetric)** | $[-\pi, \pi]$ | **0.2099 ± 0.028** | -0.6946 | 0.0998 ± 0.002 | 0.0435 ± 0.017 | Negative angle signs cause destructive interference and lowest Gram correlation. |
| **A3 (Full Circle)** | $[0, 2\pi]$ | **0.2732 ± 0.051** | -0.6312 | 0.1256 ± 0.006 | 0.1349 ± 0.020 | Periodic boundary wrap-around ($0 \equiv 2\pi$) distorts relative distance ranking. |
| **A4 (Normal CDF)** | $[0, \pi]$ | **0.2863 ± 0.064** | -0.6181 | 0.1026 ± 0.002 | 0.0649 ± 0.015 | Smooth erf scaling concentrates angles around $\pi/2$, compressing feature separation. |

---

## 4. Sub-Analysis 2: Feature-Map Depth Scaling

| Feature-Map Depth | Repeated Layers | Quantum F1 (Mean ± Std) | $\Delta\text{F1}$ vs RBF | Target Label Alignment | Q/RBF Gram Pearson $r$ | Diagnostic Mechanism |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Depth 1 Layer (B1)** | 1 | **0.5334 ± 0.102** | -0.3710 | 0.1065 ± 0.004 | **0.1894 ± 0.038** | Minimal phase scrambling preserves higher classical geometric congruence ($r=0.189$). |
| **Depth 2 Layers (B2)** | 2 | **0.3756 ± 0.113** | -0.5288 | 0.1085 ± 0.002 | **0.1232 ± 0.016** | Intermediate entanglement layer compounds phase interference on dense embeddings. |
| **Depth 3 Layers (B3)** | 3 | **0.3204 ± 0.060** | -0.5841 | 0.1206 ± 0.003 | **0.1126 ± 0.012** | Excessive unparameterized cyclic $R_{zz}$ rotations scramble semantic clusters into Hilbert space noise. |

---

## 5. Geometry-Performance Association Analysis (All 70 Quantum Runs)

| Evaluated Metric Pair | Pearson Correlation ($r$) | Two-Sided $p$-value | Empirical Scientific Conclusion |
| :--- | :---: | :---: | :--- |
| **Q/RBF Geometry Correlation vs Quantum F1** | **+0.7707** | $\mathbf{6.04 \times 10^{-15}}$ | **Strong, highly significant association**: Preservation of classical Euclidean distance ranking in Hilbert space tracks classification performance. |
| **Effective Rank vs Quantum F1** | **-0.2814** | $0.0183$ | Moderate inverse association: Overly dispersed eigenvalue spectra reflect geometric noise rather than semantic separation. |
| **Kernel Diversity vs Quantum F1** | **+0.0887** | $0.4651$ | No significant linear association across dense sentence encoding variants ($p > 0.05$). |
| **Target Label Alignment vs Quantum F1** | **+0.0484** | $0.6909$ | Small alignment shifts within a narrow baseline band ($0.098 - 0.132$) do not independently dictate F1 without distance structure. |

---

## 6. Generated Publication Figures

Saved under [`results/exp43/figures/`](file:///Users/pavanaksshay/quantum/results/exp43/figures/):
1. `fig1_q_rbf_deltas.png`: Seed-level scatter and error bars for all 7 conditions against $\epsilon = \pm 0.01$ bands.
2. `fig2_angle_mapping_f1.png`: Bar chart contrasting Quantum F1 across angular mapping strategies against Classical RBF.
3. `fig3_depth_f1.png`: Line plot showing the monotonic decline in Quantum F1 as depth increases from 1 to 3 layers.
4. `fig4_kernel_diversity.png`: Quantum kernel diversity across all 7 conditions.
5. `fig5_label_alignment.png`: Target label alignment by condition.
6. `fig6_geom_correlation.png`: Hilbert space Pearson correlation with Classical RBF Gram matrices.
7. `fig7_diversity_vs_delta.png`: Scatter plot of Kernel Diversity vs $\Delta\text{F1}$ ($N=70$ runs).
8. `fig8_alignment_vs_delta.png`: Scatter plot of Label Alignment vs $\Delta\text{F1}$ ($N=70$ runs).

---

## 7. Decision Tree Classification & Scientific Verdict

- **Decision Tree Outcome**: **Case 4 + Case 5 (Hybrid)**
  - *Case 4 (Depth Sensitivity Confirmed)*: Feature-map depth exerts a substantial, statistically detectable effect ($+21.30\text{ pp}$ improvement by reducing depth from 3 to 1 layer).
  - *Case 5 (Failure Robust to Static Encodings)*: Neither angular interval scaling nor unparameterized depth reduction alone recovers the $-37.10\text{ to } -52.88\text{ pp}$ deficit to achieve practical parity with Classical RBF ($0.9045$).
- **Experiment 44 Recommendation**: **Do not proceed to standard blind variational ansatz tuning yet**. Instead, prioritize evaluating **supervised dimensionality projection** (e.g. Linear Discriminant Analysis or supervised metric learning before SVD) to structure the 8D manifold prior to quantum phase encoding.
