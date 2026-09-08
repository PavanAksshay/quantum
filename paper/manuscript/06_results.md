# Section 4: Empirical Results and Statistical Synthesis

This section reports the empirical findings from our controlled evaluation of quantum fidelity kernels and matched classical baselines across the SMS Spam Collection, CEAS 2008, and MeAJOR corpora. Rather than following chronological experiment numbering, the results are organized directly by research question: in-distribution comparative performance (RQ1), dimensionality scaling and representation interactions (RQ2), cross-source domain generalization (RQ3), feature space geometry and alignment diagnostics (RQ4), and computational simulation scaling (RQ5). All primary comparisons incorporate 10 independent random seeds, 10,000-sample non-parametric bootstrap confidence intervals, paired permutation tests, False Discovery Rate (FDR) control, and evaluation against the predefined practical-equivalence threshold ($\varepsilon = 0.01$ F1).

---

## 4.1 Dataset and Benchmark Characteristics

Before evaluating model performance, we establish the structural characteristics, class balances, document length distributions, and split partitions across the three benchmark corpora. Table \ref{tab:dataset_characteristics} summarizes the audited dataset profiles.

\begin{table}[h!]
\centering
\small
\caption{Benchmark Corpus Characteristics, Split Partitions, and Lexical Profiles}
\label{tab:dataset_characteristics}
\begin{tabular}{lcccccccc}
\toprule
\textbf{Corpus} & \textbf{Raw Count} & \textbf{Usable Count} & \textbf{Spam/Phish \%} & \textbf{Train ($N_{\text{tr}}$)} & \textbf{Val ($N_{\text{va}}$)} & \textbf{Test ($N_{\text{te}}$)} & \textbf{Median Length} & \textbf{Source Composition} \\
\midrule
SMS Spam & 5,574 & 5,572 & 13.41\% & 3,343 & 1,114 & 1,115 & $\sim 62$ chars & Single-source mobile \\
CEAS 2008 & 39,154 & 15,000$^\dagger$ & 55.78\%$^\ddagger$ & 10,000 & 2,500 & 2,500 & $\sim 596$ chars & Phishing/Ham mix \\
MeAJOR & 108,685 & 108,684 & 44.20\%$^\ddagger$ & 10,000 & 2,500 & 2,500 & $\sim 839$ chars & Multi-source (TREC 5/6/7) \\
\bottomrule
\end{tabular}
\flushleft
\footnotesize{$^\dagger$Canonical controlled experimental subset. $^\ddagger$Class distribution across the full audited source repository (positive class prevalence within the canonical balanced experimental subsets is 18.90\% for CEAS and 19.33\% for MeAJOR).}
\end{table}

The three corpora exhibit substantial structural diversity:
- **SMS Spam Collection**: Characterized by compact, telegraphic text ($\sim 62$ characters / $\sim 12$ words median length) and substantial class imbalance ($13.41\%$ malicious spam across 4,825 legitimate and 747 spam messages).
- **CEAS 2008 Corpus**: Comprises moderately long email communications ($\sim 596$ characters / $\sim 48$ words median length), providing a standardized spam and phishing benchmark. The canonical experimental subset utilizes 10,000 training, 2,500 validation, and 2,500 test samples.
- **MeAJOR Archive**: Represents a large-scale, heterogeneous email repository ($\sim 839$ characters / $\sim 84$ words median length) composed of three distinct historical collections: TREC 2005 ($49,583$ usable emails), TREC 2006 ($15,005$ usable emails), and TREC 2007 ($44,096$ usable emails). The canonical in-distribution (IID) benchmark utilizes 10,000 training, 2,500 validation, and 2,500 test instances drawn deterministically from the frozen partitions.

This systematic variation ensures that the empirical findings do not reflect artifacts of a single document length regime, class balance ratio, or organizational domain.

---

## 4.2 RQ1: In-Distribution Model Performance (Quantum vs Matched Classical RBF)

To address RQ1, we evaluate whether the parameter-free quantum fidelity kernel ($2$-layer cyclic $ZZFeatureMap$) provides an empirical performance advantage over the matched classical Gaussian RBF kernel when both classifiers operate on strictly identical reduced representations.

We first examine the primary canonical anchor: **MeAJOR IID at 8 dimensions (8 qubits)** evaluated across the frozen suite of 10 independent random seeds ($\mathcal{S} = \{42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021\}$). Both nonlinear models operate on the standardized 8-dimensional TruncatedSVD projections derived from training-fitted 50,000 TF-IDF n-gram matrices, with decision thresholds optimized strictly on validation data.

\begin{table}[h!]
\centering
\small
\caption{Canonical In-Distribution Model Comparison on MeAJOR (Matched 8D PCA/TruncatedSVD, $N=10$ Seeds)}
\label{tab:canonical_comparison}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model Architecture} & \textbf{Test F1 ($\bar{x} \pm s$)} & \textbf{PR-AUC} & \textbf{ROC-AUC} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} \\
\midrule
Linear SVM (8D Contextual) & $0.8445 \pm 0.0022$ & $0.9315 \pm 0.0018$ & $0.9404 \pm 0.0017$ & $0.9400 \pm 0.0009$ & $0.8413 \pm 0.0039$ & $0.8477 \pm 0.0028$ \\
Classical RBF (8D Matched) & $0.8709 \pm 0.0030$ & $0.9423 \pm 0.0017$ & $0.9535 \pm 0.0016$ & $0.9504 \pm 0.0012$ & $0.8699 \pm 0.0038$ & $0.8719 \pm 0.0034$ \\
Quantum Kernel (8D Matched) & $0.8754 \pm 0.0029$ & $0.9372 \pm 0.0036$ & $0.9515 \pm 0.0028$ & $0.9523 \pm 0.0010$ & $0.8770 \pm 0.0048$ & $0.8739 \pm 0.0037$ \\
\bottomrule
\end{tabular}
\end{table}

As shown in Table \ref{tab:canonical_comparison}, both nonlinear kernels achieve substantial performance gains over the contextual low-dimensional Linear SVM baseline ($+2.64$ percentage points for RBF; $+3.09$ percentage points for Quantum), confirming that both kernel functions effectively construct nonlinear decision boundaries that capture non-separable structure in the compressed 8-dimensional subspace.

Comparing the two nonlinear classifiers directly:
- **Mean Classification Performance**: The Quantum Kernel achieves a mean test F1 score of $0.8754 \pm 0.0029$ (median: $0.8752$, range: $[0.8717, 0.8813]$), compared to $0.8709 \pm 0.0030$ for Classical RBF (median: $0.8711$, range: $[0.8645, 0.8738]$).
- **Paired Performance Difference**: The mean paired difference is:
  \begin{equation}
  \Delta \text{F1} = \text{F1}_{\text{Quantum}} - \text{F1}_{\text{Classical RBF}} = +0.004565 \pm 0.002657 \quad (+0.46\text{ percentage points})
  \end{equation}
- **Inferential Interval**: The 95% non-parametric percentile bootstrap confidence interval ($B = 10,000$) is $[+0.003040, +0.006139]$, and the two-sided paired permutation test yields $p = 0.0016$ (Benjamini-Hochberg adjusted $p = 0.0064$).

**Critical Practical Interpretation**: Because the 95% bootstrap confidence interval strictly excludes zero, the small positive difference of $+0.46$ percentage points is statistically distinguishable from zero. However, under our *a priori* methodological evaluation protocol, this difference falls entirely within the predefined practical-equivalence region:
\begin{equation}
|\Delta \text{F1}| = 0.004565 \le \varepsilon = 0.0100
\end{equation}
Therefore, we conclude that the quantum fidelity kernel produced a statistically detectable improvement at 8D, but the effect remained within the study's predefined practical-equivalence region.

---

## 4.3 RQ1: Dimensional Confirmation at 10D and 12D

To determine whether the in-distribution behavior observed at 8 dimensions persists as representation capacity expands, we evaluate the 10-seed confirmation protocol across higher dimensionalities: **10 dimensions (10 qubits)** and **12 dimensions (12 qubits)** on MeAJOR IID.

\begin{table}[h!]
\centering
\small
\caption{In-Distribution Confirmation Across Representation Dimensionalities ($N=10$ Seeds)}
\label{tab:dimensional_confirmation}
\begin{tabular}{lcccccc}
\toprule
\textbf{Dimension ($d$)} & \textbf{Quantum F1 ($\bar{x} \pm s$)} & \textbf{RBF F1 ($\bar{x} \pm s$)} & \textbf{Paired $\Delta \text{F1}$} & \textbf{95\% Bootstrap CI} & \textbf{Permutation $p$} & \textbf{Equivalence Verdict} \\
\midrule
8D (8 Qubits) & $0.8754 \pm 0.0029$ & $0.8709 \pm 0.0030$ & $+0.0046 \pm 0.0027$ & $[+0.0030, +0.0061]$ & $p = 0.0016$ & Practical Equivalence ($|\Delta| \le 0.01$) \\
10D (10 Qubits) & $0.9023 \pm 0.0034$ & $0.8967 \pm 0.0049$ & $+0.0057 \pm 0.0042$ & $[+0.0032, +0.0081]$ & $p = 0.0052$ & Practical Equivalence ($|\Delta| \le 0.01$) \\
12D (12 Qubits) & $0.9137 \pm 0.0046$ & $0.9123 \pm 0.0023$ & $+0.0014 \pm 0.0040$ & $[-0.0010, +0.0037]$ & $p = 0.2824$ & Practical Parity ($p \ge 0.05$, $|\Delta| \le 0.01$) \\
\bottomrule
\end{tabular}
\end{table}

The confirmation results in Table \ref{tab:dimensional_confirmation} establish a clear trajectory:
1. **At 10 Dimensions**: The Quantum Kernel achieves $\text{F1} = 0.9023 \pm 0.0034$, compared to $0.8967 \pm 0.0049$ for Classical RBF ($\Delta \text{F1} = +0.0057 \pm 0.0042$). The 95% bootstrap confidence interval is $[+0.0032, +0.0081]$ ($p = 0.0052$, BH FDR $p = 0.0069$). Similar to the 8D anchor, the positive difference is statistically detectable but remains strictly within the practical-equivalence threshold ($\varepsilon = 0.01$).
2. **At 12 Dimensions**: The Quantum Kernel achieves $\text{F1} = 0.9137 \pm 0.0046$, while Classical RBF achieves $\text{F1} = 0.9123 \pm 0.0023$ ($\Delta \text{F1} = +0.0014 \pm 0.0040$). The 95% bootstrap confidence interval $[-0.0010, +0.0037]$ explicitly spans zero, and the paired permutation test yields $p = 0.2824$ (BH FDR $p = 0.2824$). Under formal hypothesis testing, no statistically detectable difference is observed, establishing complete practical and statistical parity between the models.

**Key Scientific Takeaway for RQ1**: Increasing representation dimensionality allows the quantum fidelity kernel to remain competitive with classical RBF, but the observed performance differences remain practically small ($<0.006$ F1) across all dimensions and diminish toward parity at 12D.

---

## 4.4 RQ2: Dimensionality Scaling Trajectory (2D to 12D Sweep)

To investigate the full scaling trajectory and resolve whether low-dimensional quantum underperformance reflects an intrinsic feature map limitation or an information bottleneck, we examine the complete dimensionality sweep spanning $d \in \{2, 4, 6, 8, 10, 12\}$ dimensions on MeAJOR IID (Exp 35). Table \ref{tab:dimensionality_scaling} and Figure 2 present the comparative scaling profiles.

\begin{table}[h!]
\centering
\small
\caption{In-Distribution Dimensionality Scaling Sweep on MeAJOR ($d \in [2, 12]$)}
\label{tab:dimensionality_scaling}
\begin{tabular}{ccccccc}
\toprule
\textbf{Dimension ($d$)} & \textbf{Linear SVM F1} & \textbf{Classical RBF F1} & \textbf{Quantum Kernel F1} & \textbf{$\Delta \text{F1}$ (Q $-$ RBF)} & \textbf{Marginal Gain (Q)} & \textbf{Marginal Gain (RBF)} \\
\midrule
2D & $0.6853 \pm 0.0004$ & $0.6735 \pm 0.0003$ & $0.6447 \pm 0.0002$ & $-0.0288$ & — & — \\
4D & $0.8150 \pm 0.0020$ & $0.8059 \pm 0.0027$ & $0.7876 \pm 0.0021$ & $-0.0183$ & $+0.1429$ & $+0.1324$ \\
6D & $0.8069 \pm 0.0045$ & $0.8226 \pm 0.0011$ & $0.8253 \pm 0.0064$ & $+0.0027$ & $+0.0377$ & $+0.0167$ \\
8D & $0.8465 \pm 0.0017$ & $0.8731 \pm 0.0014$ & $0.8752 \pm 0.0019$ & $+0.0021$ & $+0.0499$ & $+0.0505$ \\
10D & $0.8668 \pm 0.0080$ & $0.8977 \pm 0.0076$ & $0.9037 \pm 0.0047$ & $+0.0060$ & $+0.0285$ & $+0.0246$ \\
12D & $0.8892 \pm 0.0042$ & $0.9123 \pm 0.0037$ & $0.9119 \pm 0.0076$ & $-0.0004$ & $+0.0082$ & $+0.0146$ \\
\bottomrule
\end{tabular}
\end{table}

The scaling trajectory reveals four fundamental empirical patterns:
1. **Substantial Performance Recovery Across All Models**: Quantum classification performance increased substantially across the evaluated 2D–12D range, scaling from $0.6447$ at 2D to $0.9119$ at 12D (a $+41.45\%$ relative gain; $+0.2672$ absolute F1 points). However, classical RBF ($0.6735 \to 0.9123$; $+35.46\%$) and Linear SVM ($0.6853 \to 0.8892$; $+29.75\%$) exhibit similarly large gains.
2. **Resolution of Low-Dimensional Weakness**: At extreme low dimensions ($2\text{D}\text{--}4\text{D}$), the quantum kernel significantly underperformed classical models ($\Delta \text{F1} = -0.0288$ at 2D; $\Delta \text{F1} = -0.0183$ at 4D). The dimensionality sweep provides strong evidence that the earlier low-dimensional weakness was associated with the information bottleneck introduced by aggressive TruncatedSVD compression, rather than an intrinsic failure of quantum Hilbert space geometry.
3. **Diminishing Marginal Returns**: Marginal performance improvements diminish systematically for both quantum and classical kernels as dimensionality increases. For the quantum model, the largest gain occurs between 2D and 4D ($+14.29$ percentage points), followed by $+3.77$ pp ($4\text{D} \to 6\text{D}$), $+4.99$ pp ($6\text{D} \to 8\text{D}$), $+2.85$ pp ($8\text{D} \to 10\text{D}$), and plateauing at $+0.82$ pp ($10\text{D} \to 12\text{D}$).
4. **Tracking Classical Baselines Without Separation**: The quantum model tracks the classical non-linear baseline throughout the scaling sweep rather than separating consistently from it. The maximum observed positive margin occurs at 10D ($+0.60$ percentage points in the initial sweep; $+0.57$ pp in the 10-seed confirmation), which remains well below practical equivalence.

---

## 4.5 RQ2: Representation Interaction (Sparse TF-IDF vs Dense RoBERTa)

To evaluate how underlying text representation affects kernel mapping behavior, we examine the representation ablation conducted on CEAS 2008 (Exp 29), comparing sparse n-gram TF-IDF representations against dense contextual RoBERTa-base (768-dimensional) embeddings across matched 4D and 8D TruncatedSVD projections.

\begin{table}[h!]
\centering
\small
\caption{Representation Interaction on CEAS 2008: Sparse TF-IDF vs Dense RoBERTa Embeddings}
\label{tab:representation_ablation}
\begin{tabular}{lcccc}
\toprule
\textbf{Representation Type} & \textbf{Dimension ($d$)} & \textbf{Classical RBF F1} & \textbf{Quantum Kernel F1} & \textbf{$\Delta \text{F1}$ (Quantum $-$ RBF)} \\
\midrule
Sparse TF-IDF & 4D & $0.8959$ & $0.8903$ & $-0.0056$ ($-0.56$ pp) \\
Dense RoBERTa & 4D & $0.9689$ & $0.9392$ & $-0.0297$ ($-2.97$ pp) \\
\midrule
Sparse TF-IDF & 8D & $0.9641$ & $0.9736$ & $+0.0095$ ($+0.95$ pp) \\
Dense RoBERTa & 8D & $0.9896$ & $0.9601$ & $-0.0295$ ($-2.95$ pp) \\
\bottomrule
\end{tabular}
\end{table}

The empirical results in Table \ref{tab:representation_ablation} and Figure 6 illustrate a profound interaction effect:
- **Representation Inversion**: At 8 dimensions, switching the underlying representation from dense RoBERTa embeddings to sparse TF-IDF shifted the quantum-minus-RBF performance margin from $-2.95$ percentage points to $+0.95$ percentage points—a net inversion of **$3.90$ percentage points**.
- **Representation Primacy**: While the choice between quantum and classical RBF kernels produced modest margins ($\le 0.95$ pp on TF-IDF; $\le 2.97$ pp on RoBERTa), the choice of representation produced massive gains: switching from 8D TF-IDF to 8D RoBERTa improved Classical RBF by $+2.55$ pp ($0.9641 \to 0.9896$). High-dimensional Linear SVM on 50k TF-IDF achieved $\text{F1} = 0.9721$ on MeAJOR and $\text{F1} = 0.9684$ on CEAS (Table 2).

**Scientific Takeaway**: The relative ranking and performance gap between quantum and classical kernels depend materially on the feature geometry supplied to the models. The parameter-free cyclic $ZZFeatureMap$ interacts more favorably with sparse, quasi-orthogonal TF-IDF components than with dense, correlated continuous embeddings.

---

## 4.6 RQ3: Cross-Source Generalization (Direction B Source Holdout)

To evaluate RQ3, we test whether quantum kernels exhibit superior generalization under real-world distribution shift. We examine the canonical **Direction B source-holdout protocol** (training on TREC 2007 and evaluating out-of-sample on an unseen balanced mixture of TREC 2005 and TREC 2006 emails; $N_{\text{test}} = 5,000$) across the 10-seed confirmation suite (Exp40B). Table \ref{tab:source_holdout} summarizes the holdout performance.

\begin{table}[h!]
\centering
\small
\caption{Cross-Source Domain Generalization Performance (Direction B: TREC 2007 $\to$ TREC 2005/2006, $N=10$ Seeds)}
\label{tab:source_holdout}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model Architecture} & \textbf{IID F1 (8D)} & \textbf{Holdout F1 (8D)} & \textbf{Absolute Drop ($\Delta_{\text{IID}\to\text{OOD}}$)} & \textbf{Relative Degradation} & \textbf{Paired $\Delta \text{F1}$ vs RBF} & \textbf{Statistical Status} \\
\midrule
Linear SVM (8D) & $0.8445 \pm 0.0022$ & $0.6936 \pm 0.0264$ & $-0.1509$ & $17.87\%$ & $+0.0024 \pm 0.0221$ & Baseline reference \\
Classical RBF (8D) & $0.8709 \pm 0.0030$ & $0.6913 \pm 0.0161$ & $-0.1796$ & $20.62\%$ & — & Matched baseline \\
Quantum Kernel (8D) & $0.8754 \pm 0.0029$ & $0.6680 \pm 0.0094$ & $-0.2074$ & $23.69\%$ & $-0.0233 \pm 0.0200$ & Significant Disadvantage \\
\bottomrule
\end{tabular}
\end{table}

Under Direction B source shift, all models degrade significantly relative to their in-distribution baselines. However, the quantum fidelity kernel exhibits substantially greater vulnerability:
- **Absolute and Relative Degradation**: The Quantum Kernel drops by $0.2074$ F1 points (a $23.69\%$ relative drop, $0.8754 \to 0.6680$), compared to a drop of $0.1796$ points for Classical RBF ($20.62\%$ relative drop, $0.8709 \to 0.6913$) and $0.1509$ points for Linear SVM ($17.87\%$).
- **Paired Performance Gap**: On the holdout target, Classical RBF achieves a mean F1 of $0.6913 \pm 0.0161$, while the Quantum Kernel achieves $0.6680 \pm 0.0094$, yielding a paired difference of:
  \begin{equation}
  \Delta \text{F1} = \text{F1}_{\text{Quantum}} - \text{F1}_{\text{Classical RBF}} = -0.023307 \pm 0.019988 \quad (-2.33\text{ percentage points})
  \end{equation}
- **Inferential Testing**: The 95% bootstrap confidence interval $[-0.035280, -0.011675]$ strictly excludes zero in the negative direction. The two-sided paired permutation test yields $p = 0.0046$, with a Benjamini-Hochberg adjusted $p = 0.0069$.

**Scientific Takeaway for RQ3**: Under the evaluated TREC 2007-to-TREC 2005/2006 source shift, the quantum fidelity kernel did not provide a robustness advantage and instead underperformed the matched classical RBF baseline by approximately $2.3$ percentage points, with the performance difference exceeding the practical-equivalence threshold ($\varepsilon = 0.01$).

---

## 4.7 RQ3: Direction A Generalization and Transfer Asymmetry

To verify whether domain-shift vulnerability is symmetric across collection transfers, we examine the **Direction A holdout protocol** (training on TREC 2005 + TREC 2006 and evaluating on TREC 2007; Exp 36).

Under Direction A at 8 dimensions:
- **Linear SVM (8D)**: $\text{F1} = 0.7205 \pm 0.0243$ (degradation from IID: $14.89\%$).
- **Classical RBF (8D)**: $\text{F1} = 0.7201 \pm 0.0148$ (degradation from IID: $17.52\%$).
- **Quantum Kernel (8D)**: $\text{F1} = 0.7034 \pm 0.0187$ (degradation from IID: $19.63\%$).
- **Paired Difference**: $\Delta \text{F1} = -0.0167$ ($95\%$ bootstrap CI $[-0.0409, +0.0238]$; permutation $p = 0.6658$; Table 6).

While the quantum kernel is approximately competitive with classical models under Direction A ($0.7034$ vs $0.7201$), it degrades by a larger percentage ($19.63\%$ vs $17.52\%$) and fails to outperform either classical baseline. Across both transfer directions, neither non-linear kernel demonstrates inherent cross-source robustness.

---

## 4.8 Source-Shift Lexical Diagnostics and the Dimensionality Bottleneck

To understand the mechanisms driving performance degradation under source holdout, we analyze the lexical overlap diagnostics and dimensionality recovery curves from Exp 34:

### Lexical Distribution Divergence
Comparing vocabularies between MeAJOR collections reveals extreme lexical divergence:
- **Direction A (TREC5+6 $\to$ TREC7)**: Type Out-of-Vocabulary (OOV) rate $= 92.73\%$; Token OOV rate $= 35.34\%$; centroid shift in 8D TruncatedSVD space $= 0.514$.
- **Direction B (TREC7 $\to$ TREC5+6)**: Type OOV rate $= 93.91\%$; Token OOV rate $= 34.02\%$; centroid shift in 8D TruncatedSVD space $= 0.399$.

### The Dimensionality Bottleneck Under Domain Shift
When evaluated on full, uncompressed feature spaces, classical models demonstrate high cross-source robustness:
- High-dimensional Linear SVM on 50k TF-IDF achieves **$\text{F1} = 0.8837$** under Direction A holdout (compared to $0.9721$ in IID; only an $8.9\%$ relative drop).
- Scaling TruncatedSVD dimensions for Linear SVM under Direction A restores performance monotonically:
  - 8D: $\text{F1} = 0.7562$
  - 16D: $\text{F1} = 0.8125$
  - 32D: $\text{F1} = 0.8761$
  - 64D: $\text{F1} = 0.8831$ (matching full TF-IDF within $0.0006$ F1).

**Diagnostic Interpretation**: The evidence is consistent with a substantial information bottleneck introduced by aggressive low-dimensional projection. When vocabulary shifts occur across sources, critical discriminative n-grams are discarded by low-dimensional linear projections ($d \le 12$), preventing low-dimensional nonlinear quantum and classical kernels from achieving the robustness displayed by high-dimensional classical linear models.

---

## 4.9 RQ4: Feature Space Geometry and Alignment Diagnostics

To address RQ4, we analyze the geometric properties of quantum Gram matrices relative to classical RBF geometry, target label alignment, state dispersion, and pairwise kernel diversity. Table \ref{tab:geometry_diagnostics} synthesizes the primary geometric diagnostic metrics.

\begin{table}[h!]
\centering
\small
\caption{Geometric Diagnostics: Gram Correlation, Target Label Alignment, and State Dispersion}
\label{tab:geometry_diagnostics}
\begin{tabular}{lccccc}
\toprule
\textbf{Diagnostic Analysis} & \textbf{Dimension / Dataset} & \textbf{Quantum Metric} & \textbf{Classical RBF Reference} & \textbf{Ratio / Value} & \textbf{Empirical Interpretation} \\
\midrule
Gram Correlation ($r_{\text{Q,RBF}}$) & 2D & $r = 0.6516$ & $1.0$ & — & Strong geometric similarity \\
Gram Correlation ($r_{\text{Q,RBF}}$) & 4D & $r = 0.5812$ & $1.0$ & — & Moderate geometric tracking \\
Gram Correlation ($r_{\text{Q,RBF}}$) & 8D & $r = 0.5807$ & $1.0$ & — & Stable correlation plateau \\
Gram Correlation ($r_{\text{Q,RBF}}$) & 12D & $r = 0.5529$ & $1.0$ & — & Slight tracking attenuation \\
Gram Correlation ($r_{\text{Q,RBF}}$) & 16D & $r = 0.4566$ & $1.0$ & — & Geometric divergence at high qubits \\
\midrule
Target Label Alignment $A(\mathbf{K}, \mathbf{y})$ & CEAS 8D & $0.0402$ & $0.0773$ & $52.01\%$ & Quantum alignment is $\sim 52\%$ of RBF \\
Target Label Alignment $A(\mathbf{K}, \mathbf{y})$ & SMS 8D & $0.0319$ & $0.0603$ & $52.90\%$ & Quantum alignment is $\sim 53\%$ of RBF \\
Target Label Alignment $A(\mathbf{K}, \mathbf{y})$ & MeAJOR 8D & $0.0382$ & $0.0694$ & $55.04\%$ & Quantum alignment is $\sim 55\%$ of RBF \\
\midrule
State Entropy vs Diversity ($r$) & SMS 8D & $r = -0.8257$ & N/A & — & Dispersion inversely tracks diversity \\
State Entropy vs Diversity ($r$) & CEAS 8D & $r = -0.8170$ & N/A & — & Cross-dataset replicated effect \\
State Entropy vs Diversity ($r$) & MeAJOR 8D & $r = -0.7822$ & N/A & — & Replicated inverse association \\
\bottomrule
\end{tabular}
\end{table}

### 4.9.1 Quantum–RBF Geometric Similarity
Tracking the Pearson correlation between corresponding off-diagonal Gram matrix entries reveals that quantum fidelity kernels partially mirror classical Gaussian RBF geometry in intermediate dimensions ($r \approx 0.55\text{--}0.65$ across 2D–12D; Figure 7). However, the correlation attenuates to $0.4566$ at 16 dimensions, demonstrating that the quantum feature map induces a geometry that overlaps substantially with classical RBF geometry while retaining distinct structural divergence at higher qubit counts.

### 4.9.2 Target Label Alignment Deficit
Across all three benchmark corpora, the quantum fidelity kernel exhibits consistently lower target label alignment than classical RBF ($50\%\text{--}60\%$ lower; Table \ref{tab:geometry_diagnostics}). Across the evaluated datasets, mean quantum alignment was $\sim 0.0220\text{--}0.0402$ compared to $\sim 0.0603\text{--}0.0773$ for classical RBF. This lower alignment indicates that the quantum state space mapping produces class separations that are less directly aligned with label vectors than classical Gaussian kernels, explaining why quantum models fail to exceed classical predictive accuracy.

### 4.9.3 Single-State Dispersion vs Pairwise Kernel Diversity
Within-dataset regressions from Exp 32 reveal a critical distinction between individual state dispersion and dataset-wide kernel diversity:
- **SMS Spam**: Document length correlates weakly with state entropy ($r = +0.2696$) and negatively with kernel diversity ($r = -0.2548$), while state entropy and pairwise kernel diversity exhibit a strong inverse correlation: **$r = -0.8257$**.
- **CEAS 2008**: Length correlates weakly with entropy ($r = +0.1769$), while entropy and kernel diversity correlate at **$r = -0.8170$**.
- **MeAJOR**: Length correlates weakly with entropy ($r = +0.2267$), while entropy and diversity correlate at **$r = -0.7822$** (Figure 8).

These results show a strong inverse association between single-state dispersion (entropy) and pairwise kernel diversity, demonstrating that they represent distinct facets of the induced state space. When individual statevectors disperse uniformly across Hilbert space basis states (high entropy), pairwise fidelities concentrate, reducing the variance and diversity of the Gram matrix.

---

## 4.10 RQ5: Computational Execution Cost and Simulation Scaling

To address RQ5, we measure wall-clock execution times and resident memory consumption across pipeline stages (kernel matrix computation, classifier fitting, and inference) under the standardized benchmark hardware environment. Table \ref{tab:runtime_scalability} and Figure 5 summarize the measured scaling profiles.

\begin{table}[h!]
\centering
\small
\caption{Measured Computational Execution Time and Scalability (10,000 Train / 2,500 Val / 2,500 Test Samples)}
\label{tab:runtime_scalability}
\begin{tabular}{ccccccc}
\toprule
\textbf{Dimension ($d$)} & \textbf{Quantum Total (s)} & \textbf{Classical RBF (s)} & \textbf{Pipeline Ratio} & \textbf{Quantum Kernel Time} & \textbf{RBF Kernel Time} & \textbf{Workload Status} \\
\midrule
8D & $12.95\text{s}$ & $6.74\text{s}$ & $1.92\times$ & $6.20\text{s}$ & $\sim 0.45\text{s}$ & Feasible \\
10D & $25.13\text{s}$ & $6.49\text{s}$ & $3.88\times$ & $18.50\text{s}$ & $\sim 0.45\text{s}$ & Feasible \\
12D & $85.44\text{s}$ & $6.55\text{s}$ & $13.04\times$ & $78.90\text{s}$ & $\sim 0.45\text{s}$ & Feasible (Near-tie with RBF) \\
12D (Exp 40 Anchor) & $108.8\text{s}$ & $1.7\text{s}$ & $64.0\times$ & $102.5\text{s}$ & $\sim 0.15\text{s}$ & Feasible ($N=10$ confirmation) \\
16D & $>1000\text{s}$ (est.) & $6.62\text{s}$ & $>150\times$ & $>1000\text{s}$ & $\sim 0.45\text{s}$ & Memory Infeasible ($>10.5\text{ GB}$) \\
\bottomrule
\end{tabular}
\end{table}

The empirical timing profiles demonstrate steep simulation scaling:
- **Measured 12D Pipeline Ratio**: Under the reported local statevector-simulation configuration in Exp 40, the 12-dimensional quantum-kernel pipeline required approximately **$64\times$ the measured runtime** of the matched classical RBF pipeline ($108.8\text{s}$ vs $1.7\text{s}$). Pure quantum kernel construction at 12D required $102.5\text{s}$, compared to $\sim 0.15\text{s}$ for classical RBF.
- **Dimensionality Growth**: As representation dimensionality scales from 8D to 12D, quantum pipeline execution time increases by over $8.4\times$ ($12.95\text{s} \to 108.8\text{s}$), whereas classical RBF execution remains flat ($1.5\text{s}\text{--}6.7\text{s}$).
- **Memory Ceiling at 16 Dimensions**: In the local 10,000-sample benchmark workload, exact `complex128` statevector simulation at 16 qubits exceeded $10.5\text{ GB}$ of resident memory, rendering un-chunked Gram matrix simulation computationally infeasible on standard research workstations.

*Scope Qualification*: These measurements reflect software statevector simulation on classical CPU architectures, illustrating the practical computational cost of simulating quantum kernels rather than physical quantum device execution times or asymptotic complexity bounds.

---

## 4.11 Integrated Statistical Synthesis

Table \ref{tab:statistical_synthesis} provides an integrated synthesis of all empirical comparisons across in-distribution benchmarks, dimensionality scaling sweeps, cross-source domain holdouts, and geometric diagnostics.

\begin{table}[h!]
\centering
\small
\caption{Comprehensive Statistical and Empirical Evidence Synthesis}
\label{tab:statistical_synthesis}
\begin{tabular}{lcccc}
\toprule
\textbf{Evaluation Dimension} & \textbf{Empirical Comparison} & \textbf{Statistical Metric / Evidence} & \textbf{$p$-Value / CI} & \textbf{Practical Classification} \\
\midrule
RQ1: IID 8D Anchor & Quantum vs RBF & $\Delta \text{F1} = +0.0046 \pm 0.0027$ & $p = 0.0016$; CI $[+0.0030, +0.0061]$ & Practical Equivalence ($|\Delta| \le 0.01$) \\
RQ1: IID 10D Confirmation & Quantum vs RBF & $\Delta \text{F1} = +0.0057 \pm 0.0042$ & $p = 0.0052$; CI $[+0.0032, +0.0081]$ & Practical Equivalence ($|\Delta| \le 0.01$) \\
RQ1: IID 12D Parity & Quantum vs RBF & $\Delta \text{F1} = +0.0014 \pm 0.0040$ & $p = 0.2824$; CI $[-0.0010, +0.0037]$ & Practical Parity ($p \ge 0.05$) \\
RQ2: Dimensionality Sweep & 2D to 12D Scaling & $+41.45\%$ relative gain (Q) & Monotonic recovery to 12D & Bottleneck Resolution \\
RQ2: Representation Shift & TF-IDF vs RoBERTa & $3.90$ pp margin shift (CEAS) & Inverts Q vs RBF ranking & Representation Dominance \\
RQ3: Direction B Shift & Quantum vs RBF & $\Delta \text{F1} = -0.0233 \pm 0.0200$ & $p = 0.0046$; CI $[-0.0353, -0.0117]$ & Significant Classical Advantage \\
RQ3: Direction A Shift & Quantum vs RBF & $\Delta \text{F1} = -0.0167$ & $p = 0.6658$; CI $[-0.0409, +0.0238]$ & Compatible with Zero \\
RQ4: Gram Geometry & Quantum vs RBF & Pearson $r = 0.55\text{--}0.65$ & Decays to $0.4566$ at 16D & Non-Identical Geometry \\
RQ4: Label Alignment & Quantum vs RBF & $50\%\text{--}60\%$ lower for Quantum & Replicated across 3 corpora & Weaker Class Alignment \\
RQ5: Computational Cost & Quantum vs RBF & $64.0\times$ runtime penalty at 12D & $108.8\text{s}$ vs $1.7\text{s}$ & Simulation Overhead \\
\bottomrule
\end{tabular}
\end{table}

**Definitive Answer to the Central Research Question**:  
Across all evaluated datasets, feature representations, dimensionality sweeps, and evaluation regimes, the empirical evidence **does not support a consistent, statistically superior, or practically meaningful advantage for quantum fidelity kernels over matched classical RBF kernels** for text-based scam and phishing detection.

*Scope Boundary*: This conclusion is strictly grounded in the evaluated text security corpora, parameter-free cyclic $ZZFeatureMap$ architecture, classical statevector simulation regime, and standardized experimental protocols; it does not establish a universal disproof of quantum computing for NLP.

---

## 4.12 Practical Equivalence Analysis

Figure 3 illustrates the performance differences $\Delta \text{F1} = \text{F1}_{\text{Quantum}} - \text{F1}_{\text{Classical RBF}}$ relative to the predefined practical equivalence bounds ($[ -0.01, +0.01 ]$).

A primary methodological contribution of this study is the explicit decoupling of statistical detectability from practical engineering significance:
- **In-Distribution 8D and 10D**: Small positive deltas ($+0.0046$ at 8D; $+0.0057$ at 10D) are statistically distinguishable from zero ($p < 0.01$) due to the high precision of 10-seed paired evaluation, but their magnitudes remain well within the $\pm 1.0$ percentage point practical-equivalence boundary.
- **In-Distribution 12D**: At 12 dimensions, the difference ($\Delta = +0.0014$) is both practically small and statistically undetectable ($p = 0.2824$), demonstrating practical parity.
- **Direction B Source Holdout**: Under domain shift, the classical advantage ($\Delta = -0.0233$) is both statistically supported ($p = 0.0046$; BH FDR $p = 0.0069$) and practically meaningful, exceeding the practical equivalence boundary.

---

## 4.13 Summary of Empirical Results

The core empirical findings of this study are summarized in six concise takeaways:
1. **In-Distribution Competitiveness**: Parameter-free quantum fidelity kernels act as valid nonlinear classifiers that achieve performance competitive with matched classical RBF kernels under IID evaluation.
2. **Sub-Threshold In-Distribution Differences**: All observed in-distribution differences remain strictly below the predefined $1.0$ percentage point practical-equivalence threshold ($\varepsilon = 0.01$), converging to complete parity at 12 dimensions.
3. **Dimensionality Scaling and Bottleneck Recovery**: Increasing representation dimensionality substantially improves the accuracy of all evaluated models, providing strong evidence that the earlier low-dimensional weakness was associated with the information bottleneck introduced by aggressive TruncatedSVD compression.
4. **Absence of Domain-Shift Robustness**: Quantum kernels exhibit no robustness advantage under cross-source domain shift and suffer a statistically significant performance deficit relative to classical RBF under Direction B transfer.
5. **Distinct Feature Space Geometry**: Quantum Gram matrices partially mirror classical RBF geometry in intermediate dimensions but exhibit $50\%\text{--}60\%$ lower target label alignment. Single-state dispersion (entropy) is strongly inversely associated with pairwise kernel diversity ($r \approx -0.80$).
6. **Steep Simulation Overhead**: Classical simulation of quantum kernels incurs rapidly scaling computational execution time ($64\times$ penalty at 12D) without delivering predictive superiority.

<!-- CLAIM AUDIT:
- Section 4.1: Dataset characteristics (SMS: 5572 usable, 13.41% pos; CEAS: 15k canonical, 55.78% pos raw; MeAJOR: 108684 usable, TREC5=49583, TREC6=15005, TREC7=44096). Evidence: Table 1, table_1_dataset_characteristics.csv.
- Section 4.2: MeAJOR IID 8D (Q = 0.8754 ± 0.0029, RBF = 0.8709 ± 0.0030, Delta = +0.0046 [95% CI +0.0030, +0.0061], p = 0.0016). Linear 8D = 0.8445. Practical equivalence. Evidence: Table 3, Table 6, exp40_statistical_summary.csv.
- Section 4.3: 10D (Q = 0.9023 ± 0.0034, RBF = 0.8967 ± 0.0049, Delta = +0.0057, p = 0.0052); 12D (Q = 0.9137 ± 0.0046, RBF = 0.9123 ± 0.0023, Delta = +0.0014, p = 0.2824, CI [-0.0010, +0.0037]). Evidence: Table 3, Table 6, exp40_statistical_summary.csv.
- Section 4.4: Dimensionality scaling 2D-12D (2D: Linear 0.6853, RBF 0.6735, Q 0.6447; 4D: 0.8150/0.8059/0.7876; 6D: 0.8069/0.8226/0.8253; 8D: 0.8465/0.8731/0.8752; 10D: 0.8668/0.8977/0.9037; 12D: 0.8892/0.9123/0.9119). Evidence: Table 4, table_4_dimensionality_scaling.csv, Exp 35.
- Section 4.5: Representation ablation on CEAS (4D TF-IDF 0.8959/0.8903 vs RoBERTa 0.9689/0.9392; 8D TF-IDF 0.9641/0.9736 vs RoBERTa 0.9896/0.9601; 3.90 pp net shift). Evidence: Exp 29, Figure 6.
- Section 4.6: Direction B holdout (Linear 0.6936, RBF 0.6913, Q 0.6680 ± 0.0094, Delta = -0.0233, 95% CI [-0.0353, -0.0117], p = 0.0046, BH FDR p = 0.0069). Evidence: Table 5, Table 6, exp40_statistical_summary.csv.
- Section 4.7: Direction A holdout (Linear 0.7205, RBF 0.7201, Q 0.7034, Delta = -0.0167, p = 0.6658). Evidence: Table 5, Table 6, Exp 36.
- Section 4.8: Lexical OOV (Dir A type 92.73%, token 35.34%, dist 0.514; Dir B type 93.91%, token 34.02%, dist 0.399) & Linear SVD bottleneck (Full 0.8837, 8D 0.7562, 16D 0.8125, 32D 0.8761, 64D 0.8831). Evidence: Exp 34.
- Section 4.9: Geometry (Gram r = 0.6516 [2D], 0.5812 [4D], 0.5807 [8D], 0.5529 [12D], 0.4566 [16D]; Label alignment CEAS 0.0402 vs 0.0773, SMS 0.0319 vs 0.0603, MeAJOR 0.0382 vs 0.0694; Entropy vs Diversity r = -0.8257 [SMS], -0.8170 [CEAS], -0.7822 [MeAJOR]). Evidence: Table 7, table_7_geometry_diagnostics.csv, Exp 27, Exp 32.
- Section 4.10: Computational scalability (8D 12.95s/6.74s [1.92x]; 10D 25.13s/6.49s [3.88x]; 12D 85.44s/6.55s [13.04x]; Exp 40 12D 108.8s/1.7s [64.0x]; 16D >10.5 GB RAM). Evidence: Table 8, table_8_runtime_scalability.csv, exp40_results.csv.
- Section 4.11: Synthesis table and explicit answer to central RQ. Evidence: Table 6, UPDATED_PAPER_CLAIMS.md.
- Section 4.12: Practical equivalence analysis and decoupling statistical detectability from practical significance. Evidence: Figure 3, FINAL_PROTOCOL_V1.md.
- Section 4.13: Six concise summary bullets. Evidence: Section 4 synthesis.
-->
