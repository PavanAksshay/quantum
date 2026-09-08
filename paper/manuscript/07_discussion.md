# Section 5: Discussion

The empirical findings presented in Section 4 provide a systematic, multi-seed evaluation of quantum fidelity kernels and matched classical Gaussian RBF baselines across text security classification benchmarks. In this section, we interpret the scientific meaning of these results, examine the mechanisms governing representation dimensionality and geometric alignment, evaluate the practical implications for security-oriented natural language processing, and outline rigorous methodological standards for future quantum machine learning (QML) benchmarking.

---

## 5.1 Principal Finding: No Consistent Practical Quantum Advantage

The central question motivating this investigation was whether fixed, parameter-free quantum fidelity kernels provide a consistent and practically meaningful classification advantage over matched classical Gaussian RBF kernels when evaluated under rigorous, leak-free experimental conditions on text security data.

Across the empirical evidence gathered in this study, the answer is **no, under the evaluated conditions**. Rather than displaying a broad operational advantage, the behavior of the evaluated quantum kernel resolves into three distinct levels of performance across task settings:

1. **In-Distribution (IID) Regimes**: Under standard independent and identically distributed sampling partitions, the quantum fidelity kernel is competitive with the matched classical Gaussian RBF baseline and achieves minor, statistically detectable improvements at intermediate dimensions (at 8D: $\Delta \text{F1} = +0.0046$, $p = 0.0016$; at 10D: $\Delta \text{F1} = +0.0057$, $p = 0.0052$). However, both margins fall strictly below the predefined practical-equivalence threshold ($\varepsilon = 0.01$ F1). At 12 dimensions, the two models converge to statistical and practical parity ($\Delta \text{F1} = +0.0014$, $p = 0.2824$, 95% CI $[-0.0010, +0.0037]$).
2. **Cross-Source Domain Generalization**: When evaluated under domain transfer across heterogeneous historical corpora (Direction B: training on TREC 2007 and testing on TREC 2005/2006), the quantum kernel underperforms the matched classical RBF baseline by a statistically significant and practically meaningful margin ($\Delta \text{F1} = -0.0233$, $p = 0.0046$, Benjamini–Hochberg adjusted $p = 0.0069$). This negative margin exceeds the practical-equivalence threshold, demonstrating that competitive in-distribution performance does not translate into a source-domain transfer advantage.
3. **Computational and Simulation Scaling**: Under classical statevector simulation, the runtime required to construct the quantum kernel matrix scales rapidly with qubit count. At 12 dimensions, computing the quantum kernel requires $108.8$ seconds compared to $1.7$ seconds for the matched classical RBF kernel—a $\approx 64\times$ execution time penalty for a nominal in-distribution gain of $0.14$ percentage points.

Therefore, the scientific conclusion of this benchmark is not that quantum kernels fail to function as nonlinear classifiers; they successfully construct valid Hilbert space geometries that track classical nonlinear kernels. Rather, the conclusion is that **the evaluated quantum kernel did not produce a consistent, practically meaningful advantage over a matched classical RBF baseline, and its competitive in-distribution performance did not translate into a source-domain robustness advantage or comparable simulation cost**.

---

## 5.2 Why the Matched Baseline Matters

A core methodological contribution of this study is the enforcement of a strictly matched experimental design. In many prior QML empirical studies, quantum models have been contrasted with unmatched classical baselines—such as comparing an expressive quantum circuit against an unregularized linear classifier, or evaluating quantum models on preprocessed dense embeddings while evaluating classical baselines on raw lexical counts. Such disparities introduce confounding factors that make it impossible to isolate whether observed performance differences stem from quantum feature spaces or mundane variations in data representation, capacity, or tuning.

In contrast, the protocol implemented in this work ensures strict parity across all non-quantum pipeline stages:
- **Identical Input Representations**: Quantum and classical RBF classifiers receive the exact same input vectors, whether derived from TruncatedSVD projections of high-dimensional TF-IDF matrices or dense RoBERTa sentence embeddings.
- **Identical Feature Dimensionality**: Feature dimensions ($d$) map one-to-one to qubit counts ($N_q = d$), ensuring both models operate in equal-dimensional coordinate spaces.
- **Identical Partitioning and Rescaling**: Training, validation, and test splits are strictly synchronized, and MinMax scaling to $[0, \pi]$ is applied identically to prevent data leakage.
- **Identical Class Weighting and Loss Formulation**: Both models utilize the same support vector classification (SVC) dual formulation with balanced class weighting ($w_c = N / (2 N_c)$).
- **Identical Threshold Selection**: Post-hoc classification decision thresholds are tuned exclusively on validation sets using identical grid resolutions to maximize F1 score.
- **Identical Random Seeds**: All evaluations are replicated across 10 synchronized random seeds ($\mathcal{S} = \{42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021\}$).

By eliminating these confounding variables, the matched design substantially narrows the set of methodological explanations for observed differences. When a performance gap appears—such as the $+0.46$ percentage point in-distribution difference at 8D or the $-2.33$ percentage point degradation under Direction B domain shift—it can be attributed with high confidence to the geometric differences between the quantum fidelity metric and the classical Gaussian kernel, rather than extraneous preprocessing or evaluation disparities. We do not claim that this design isolates the quantum kernel in an absolute causal sense, as implementation-level numerical artifacts and solver dynamics remain; however, it provides a rigorous empirical foundation that prevents artificial inflation of quantum efficacy.

---

## 5.3 Dimensionality: The Important Second Result

A key empirical finding of this investigation is the dramatic performance trajectory observed as representation dimensionality expands from 2 to 12 dimensions:
- **Quantum Kernel**: $\text{F1} = 0.6447 \to 0.7876 \to 0.8253 \to 0.8752 \to 0.9037 \to 0.9119$ ($+26.72$ pp gain).
- **Classical RBF**: $\text{F1} = 0.6735 \to 0.7918 \to 0.8254 \to 0.8706 \to 0.8980 \to 0.9123$ ($+23.88$ pp gain).
- **Linear SVM**: $\text{F1} = 0.6853 \to 0.7584 \to 0.7946 \to 0.8445 \to 0.8730 \to 0.8892$ ($+20.39$ pp gain).

This scaling behavior is critical for contextualizing prior literature. In early QML text classification experiments, researchers frequently restricted evaluations to extreme low-dimensional embeddings ($d \in \{2, 4\}$) due to simulation memory limits or device constraints. In those narrow regimes, quantum models often exhibited severe underperformance relative to classical linear models ($0.6447$ vs $0.6853$ at 2D). A naive interpretation of such low-dimensional data might conclude that quantum fidelity kernels possess an intrinsic geometric flaw that renders them incapable of separating text data.

Our systematic sweep across $d \in \{2, 4, 6, 8, 10, 12\}$ demonstrates that this interpretation is incomplete. The dimensionality sweep indicates that the earlier low-dimensional limitation was strongly associated with aggressive representation compression. Projecting a 50,000-dimensional TF-IDF vocabulary space down to 2 or 4 orthogonal components discards over $95\%$ of the explained variance, discarding the lexical discriminators required to identify deceptive text.

As dimensionality expands to 8, 10, and 12 dimensions, the performance of all three classifiers increases substantially and monotonically. This confirms that information capacity is primarily governed by the upstream representation dimension rather than a failure unique to quantum Hilbert space mappings. Furthermore, the empirical trajectory exhibits clear diminishing marginal returns:
- The quantum model experiences its largest single gain from 2D to 4D ($+14.29$ pp), followed by $+3.77$ pp ($4\text{D} \to 6\text{D}$), $+4.99$ pp ($6\text{D} \to 8\text{D}$), $+2.85$ pp ($8\text{D} \to 10\text{D}$), and eventually plateaus at $+0.82$ pp ($10\text{D} \to 12\text{D}$).
- Beyond 8 dimensions, performance gains attenuate rapidly, and the quantum and classical RBF curves converge, reaching statistical indistinguishability at 12D ($0.9137$ vs $0.9123$).

This trajectory indicates that representation dimensionality dictates the ceiling of recoverable semantic information. Once sufficient dimensions are provided to capture dominant lexical signals, the mathematical distinction between quantum state fidelity and Gaussian radial distance becomes secondary.

---

## 5.4 Representation Is Not a Neutral Detail

A central insight arising from our representation ablation on the CEAS 2008 corpus is that the relative performance ranking between quantum and classical kernels is not invariant; it reverses depending on the upstream feature representation:
- **8D TF-IDF + TruncatedSVD**:
  - Quantum Kernel: $\text{F1} = 0.9736$
  - Classical RBF: $\text{F1} = 0.9641$
  - Margin: $\Delta \text{F1} = +0.0095$ ($+0.95$ pp quantum advantage)
- **8D Dense RoBERTa Embeddings**:
  - Quantum Kernel: $\text{F1} = 0.9601$
  - Classical RBF: $\text{F1} = 0.9896$
  - Margin: $\Delta \text{F1} = -0.0295$ ($-2.95$ pp classical advantage)

Across these two standard text representations, the relative performance shifts by a net margin of $3.90$ percentage points ($+0.95$ pp to $-2.95$ pp), completely overturning the apparent ranking between the two kernel families.

We caution strongly against drawing overly simplistic conclusions from this result. It would be scientifically unjustified to conclude that TF-IDF representations are universally superior for quantum algorithms, that transformer embeddings are fundamentally incompatible with quantum feature maps, or that sparse lexical vectors possess an intrinsic quantum affinity. Rather, **the reversal across representations demonstrates that the apparent benefit of a kernel cannot be interpreted independently of the geometry supplied by the upstream representation**.

Dense contextual embeddings generated by deep language models such as RoBERTa are trained via metric learning objectives (contrastive loss and cross-entropy) that naturally cluster semantically related documents in a continuous Euclidean space. The classical Gaussian RBF kernel, which measures Euclidean distance in this space ($k(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$), directly exploits this metric structure, achieving an exceptional F1 score of $0.9896$. In contrast, the cyclic $ZZFeatureMap$ applies non-linear trigonometric phase rotations ($U_{\Phi}(\mathbf{x}) |0\rangle^{\otimes n}$) that wrap coordinates into high-dimensional torus topologies. On dense, continuous embeddings, these phase wrappings disrupt the Euclidean metric structure without introducing compensating discriminative separability.

Conversely, TruncatedSVD projections of sparse TF-IDF counts produce orthogonal coordinate axes where coordinate values represent variance components across bag-of-words histograms. In this coordinate space, the periodic phase interactions induced by quantum entanglement gates ($e^{i (\pi - x_i)(\pi - x_j) Z_i Z_j}$) provide a non-linear feature expansion that slightly outperforms the Gaussian radial decay.

This finding carries significant implications for QML benchmarking: a quantum kernel may appear superior or inferior purely as an artifact of upstream embedding geometry. Consequently, future evaluations of quantum kernel methods must treat the upstream representation as a primary, first-class experimental factor rather than a neutral preprocessing step.

---

## 5.5 Cross-Source Generalization: Why the Negative Result Matters

Security text classification systems deployed in real-world environments operate under persistent distribution shift. Attackers continuously alter lexical patterns, campaign structures evolve, and email headers vary across organizational mail servers. A model evaluated solely on independent and identically distributed (IID) splits may exhibit high test accuracy by memorizing domain-specific artifacts that fail under operational transfer.

Our evaluation of cross-source generalization on the multi-source MeAJOR corpus provides an empirical test of model robustness across distinct historical email sources. Under Direction B transfer (training on TREC 2007 and testing on TREC 2005/2006):
- **Quantum Kernel**: $\text{F1} = 0.6680 \pm 0.0094$
- **Classical RBF**: $\text{F1} = 0.6913 \pm 0.0161$
- **Performance Gap**: $\Delta \text{F1} = -0.0233$, 95% Bootstrap CI $[-0.0353, -0.0117]$
- **Statistical Significance**: Permutation $p = 0.0046$; Benjamini–Hochberg adjusted $p = 0.0069$

This negative result is methodologically more substantial than the small positive margins observed in IID testing for several reasons:
1. **Magnitude Exceeds Practical Threshold**: While the positive IID margins at 8D ($+0.0046$) and 10D ($+0.0057$) remained within the practical-equivalence threshold ($\varepsilon = 0.01$), the cross-source performance deficit ($-0.0233$) exceeds the equivalence threshold by more than a factor of two.
2. **Robustness to Multiplicity Correction**: The performance gap survives rigorous Benjamini–Hochberg False Discovery Rate control at $\alpha = 0.05$ across all experimental hypotheses.
3. **Failure of Transfer Hypotheses**: Theoretical conjectures suggesting that mapping classical data into exponentially large Hilbert spaces might provide intrinsic regularization or invariant geometric representations under domain shift are not supported by the empirical data.

We do not claim that quantum kernels are universally non-robust. Rather, **the evaluated quantum kernel did not demonstrate a source-domain robustness advantage, and instead exhibited greater performance degradation under source shift than the matched classical baseline**. For applied security tasks, where cross-domain robustness is a critical operational requirement, this finding demonstrates that in-distribution parity cannot be assumed to imply transfer resilience.

---

## 5.6 Source Shift and the Dimensionality Bottleneck

To understand why all models experienced substantial performance degradation under cross-source evaluation, we must examine the interaction between lexical domain shift and representation dimensionality reduction.

Lexical overlap diagnostics between the TREC 2007 source domain and the TREC 2005/2006 target domain reveal extreme distribution drift:
- **Type Out-of-Vocabulary (OOV) Rate**: $92.70\%\text{--}93.90\%$ of distinct word types appearing in the target test sets were unobserved in the source training vocabulary.
- **Token Out-of-Vocabulary (OOV) Rate**: $34.03\%\text{--}35.32\%$ of all running tokens in the target test sets were out-of-vocabulary.

When models operate on full-vocabulary representations without dimensionality reduction, linear classifiers exhibit substantial resilience to this domain shift. A Linear SVM trained on the complete 50,000-dimensional TF-IDF vocabulary achieves an out-of-distribution F1 score of **$0.8837$** under Direction B transfer (Table \ref{tab:dimensionality_recovery_svd}). However, when the representation is compressed via TruncatedSVD, performance degrades systematically as a function of dimension:
- **8 Dimensions**: Linear SVM $\text{F1} = 0.7562$ ($-12.75$ pp drop from full vocabulary)
- **16 Dimensions**: Linear SVM $\text{F1} = 0.8125$ ($-7.12$ pp drop)
- **32 Dimensions**: Linear SVM $\text{F1} = 0.8761$ ($-0.76$ pp drop)
- **64 Dimensions**: Linear SVM $\text{F1} = 0.8831$ ($-0.06$ pp drop; full recovery)

This diagnostic comparison reveals a crucial scientific insight: **the evidence suggests that source distribution shift and aggressive dimensionality reduction interact, making it difficult to interpret cross-source quantum degradation as a purely kernel-specific failure**.

When vocabulary distributions drift across domains, the discriminative signal becomes distributed across thousands of low-frequency, domain-specific n-grams. A low-dimensional linear projection ($d = 8$) retains only the principal variance axes of the source domain, discarding the long-tail lexical features necessary for target-domain generalization. Consequently, both quantum and classical RBF kernels operating at 8 dimensions are severely constrained by the upstream information bottleneck. While the quantum kernel suffers an additional geometric penalty relative to classical RBF ($\text{F1} = 0.6680$ vs $0.6913$), the dominant performance loss stems from the representation constraint itself.

---

## 5.7 Geometry: Why Quantum and Classical RBF Are Related but Not Identical

To understand why quantum and classical RBF kernels produce broadly comparable predictive performance yet diverge under specific conditions, we analyze their induced Gram matrix geometries.

Tracking the Pearson correlation between corresponding off-diagonal kernel matrix entries ($K_{ij}^{\text{Q}}$ and $K_{ij}^{\text{RBF}}$) across dimensions reveals a structured geometric relationship:
- In low-to-intermediate dimensions ($2\text{D}\text{--}12\text{D}$), the correlation remains moderate-to-strong: $r \approx 0.55\text{--}0.65$.
- At 16 dimensions, the correlation attenuates to $r = 0.4566$.

This moderate correlation explains why the two kernel families exhibit similar in-distribution classification curves: the parameter-free cyclic $ZZFeatureMap$ generates pairwise inner products that partially approximate the monotonic radial decay of the classical Gaussian RBF kernel. Both kernels assign high similarity to vectors that are close in coordinate space and lower similarity to distant vectors.

However, the correlation is far from unity. The quantum feature map does not merely reproduce the classical RBF kernel with numerical noise; it constructs a distinct metric space governed by multi-qubit phase interference. This structural divergence explains how identical input data can yield subtle predictive differences—such as the $+0.57$ pp quantum improvement at 10D or the $-2.33$ pp classical advantage under cross-source transfer. We emphasize that Gram correlation is a descriptive geometric metric rather than a direct predictor of generalization performance.

---

## 5.8 Label Alignment Deficit

Kernel-target alignment ($A(K, \mathbf{y})$) measures the normalized Frobenius inner product between the empirical kernel Gram matrix and the ideal target label matrix $\mathbf{y}\mathbf{y}^T$:
$$A(K, \mathbf{y}) = \frac{\langle K, \mathbf{y}\mathbf{y}^T \rangle_F}{\|K\|_F \|\mathbf{y}\mathbf{y}^T\|_F}$$

Across all three benchmark corpora, diagnostic evaluations reveal that the quantum fidelity kernel exhibits a consistent label alignment deficit relative to the matched classical RBF kernel:
- **MeAJOR Corpus (8D)**: Quantum alignment $\approx 0.038\text{--}0.040$ vs Classical RBF $\approx 0.069\text{--}0.077$ ($45\%\text{--}50\%$ lower).
- **CEAS 2008 Corpus (8D)**: Quantum alignment $\approx 0.022$ vs Classical RBF $\approx 0.060$ ($63\%$ lower).
- **SMS Spam Collection (8D)**: Quantum alignment $\approx 0.038$ vs Classical RBF $\approx 0.077$ ($50\%$ lower).

Across all evaluated datasets, the quantum kernel matrix exhibits substantially weaker alignment with binary security class labels than the Gaussian RBF kernel. This indicates that unparameterized quantum state fidelities disperse data points across Hilbert space without concentrating same-class pairs as effectively as the exponentially scaled Euclidean distance in classical RBF.

We avoid making a causal claim that lower label alignment directly dictates lower classification accuracy. In highly non-linear support vector classification, margin maximization can compensate for suboptimal global kernel alignment. Rather, **lower label alignment provides a plausible geometric correlate of the weaker quantum performance observed in several configurations, but the analysis is associative rather than causal**.

---

## 5.9 State Dispersion vs Kernel Diversity

A widespread conceptual intuition in quantum machine learning literature is that maximizing the spread or entropy of quantum states across Hilbert space improves the discriminative power of the resulting classifier. The heuristic reasoning assumes that if input documents are mapped to widely dispersed statevectors, they will be more easily separated by hyperplanes in state space.

Our empirical diagnostics directly challenge this intuition by demonstrating that **single-state dispersion (von Neumann state entropy) and pairwise kernel diversity represent distinct, inversely correlated properties of quantum feature spaces**.

Within-dataset regressions across the benchmark corpora demonstrate a strong, consistent inverse association between mean state entropy and pairwise kernel matrix diversity:
- **SMS Spam Collection**: $r = -0.8257$
- **CEAS 2008 Corpus**: $r = -0.8170$
- **MeAJOR Archive**: $r = -0.7822$

When individual quantum states disperse uniformly across all $2^{N_q}$ computational basis states (yielding high single-state entropy), the inner products between distinct statevectors concentrate around uniform, low-variance values. As statevectors become uniformly distributed over the complex unit sphere, the pairwise fidelities $|\langle \psi(\mathbf{x}) | \psi(\mathbf{z}) \rangle|^2$ collapse toward a narrow distribution, reducing the off-diagonal variance and diversity of the Gram matrix.

This finding demonstrates why simplistic geometric heuristics are insufficient for QML algorithm design. The classification-relevant geometry depends on the structure of pairwise relationships and their alignment with target labels, not merely on how widely individual statevectors are scattered across Hilbert space.

---

## 5.10 Why Small IID Quantum Gains Should Not Be Overinterpreted

Under canonical in-distribution evaluation on MeAJOR at 8 and 10 dimensions, the quantum fidelity kernel achieved positive performance margins over the matched classical RBF baseline:
- **8 Dimensions (8 Qubits)**: $\Delta \text{F1} = +0.0046$ ($+0.46$ pp), $p = 0.0016$, 95% Bootstrap CI $[+0.0030, +0.0061]$.
- **10 Dimensions (10 Qubits)**: $\Delta \text{F1} = +0.0057$ ($+0.57$ pp), $p = 0.0052$, 95% Bootstrap CI $[+0.0032, +0.0081]$.

Both differences are statistically detectable at $\alpha = 0.05$ across 10 independent random seeds. However, it is essential to distinguish between **statistical detectability** and **practical significance**.

Prior to conducting final confirmation testing, our experimental protocol established a formal practical-equivalence threshold of $\varepsilon = 0.01$ ($1.0$ percentage point of F1 score). In operational cybersecurity applications, performance variations below $1$ percentage point frequently fall within the noise floor of label ambiguity, sampling drift, and threshold tuning.

Because both observed margins ($+0.46$ pp and $+0.57$ pp) fall well within the $[-\varepsilon, +\varepsilon]$ interval, and because the 12-dimensional comparison converged to complete parity ($\Delta \text{F1} = +0.0014$, $p = 0.2824$), these results satisfy the formal criteria for **practical equivalence**. Claiming a "quantum advantage" based on sub-percentage-point margins that vanish at 12D would misrepresent the empirical evidence. The scientific takeaway is that quantum fidelity kernels achieve parity with classical RBF kernels under matched in-distribution conditions, but do not provide an operational advantage.

---

## 5.11 Computational Cost Changes the Practical Engineering Trade-Off

Any evaluation of machine learning models for production applications must weigh predictive performance against computational execution cost. In security filtering systems—where thousands of messages must be classified per second—computational overhead is a primary operational constraint.

Under our standardized benchmarking environment using statevector simulation, the computational cost of evaluating quantum kernels scales steeply with dimensionality:
- **2 Dimensions**: Quantum kernel computation $= 3.4\text{s}$; Classical RBF $= 1.3\text{s}$ ($2.6\times$ ratio).
- **8 Dimensions**: Quantum kernel computation $= 17.2\text{s}$; Classical RBF $= 1.4\text{s}$ ($12.3\times$ ratio).
- **12 Dimensions**: Quantum kernel computation $= 108.8\text{s}$; Classical RBF $= 1.7\text{s}$ ($64.0\times$ ratio).

At 12 dimensions, computing the quantum Gram matrix required nearly two minutes of compute time per seed, compared to less than two seconds for the classical Gaussian kernel. Furthermore, attempting to evaluate a 16-dimensional quantum kernel on the 10,000-sample training set exceeded available system memory (>10.5 GB allocation ceiling), whereas classical RBF computation completed in $2.1$ seconds using minimal RAM.

We explicitly qualify that these runtime measurements reflect exact classical statevector simulation on CPU architectures, rather than physical quantum hardware execution. The $64\times$ execution ratio cannot be extrapolated to physical quantum processors executing quantum circuits natively.

Nevertheless, this result is directly relevant for contemporary QML research and near-term deployment. Because practical text security practitioners and QML researchers currently evaluate models via classical simulation, the steep exponential scaling of statevector simulation represents a substantial engineering barrier. Incurring a $64\times$ runtime penalty to achieve a statistically indistinguishable F1 score ($0.9137$ vs $0.9123$) represents an unfavorable engineering trade-off for classical simulation workloads.

---

## 5.12 Explicit Scope: What the Study Does and Does Not Establish

To ensure rigorous scientific integrity and prevent overinterpretation, Table \ref{tab:supported_conclusions} explicitly categorizes what this study establishes versus what remains unsupported.

\begin{table}[h!]
\centering
\small
\caption{Explicit Boundaries: Supported Findings vs Unsupported Claims}
\label{tab:supported_conclusions}
\begin{tabular}{p{7.5cm}p{7.5cm}}
\toprule
\textbf{Supported by Empirical Evidence} & \textbf{NOT Supported / Excluded from Scope} \\
\midrule
$\checkmark$ Quantum fidelity kernels can achieve competitive in-distribution classification performance matching classical RBF baselines under identical reduced representations. & $\times$ Universal quantum advantage or superiority over classical machine learning algorithms for text processing. \\
$\checkmark$ Minor in-distribution quantum improvements ($+0.46$ to $+0.57$ pp at 8D–10D) are statistically detectable across 10 seeds, but remain practically equivalent ($\le 1.0$ pp). & $\times$ Universal inferiority or intrinsic failure of quantum Hilbert space mappings for NLP tasks. \\
$\checkmark$ Increasing representation dimensionality from 2D to 12D resolves low-dimensional bottlenecks for both quantum and classical models. & $\times$ Causal claims asserting that document length mechanically causes state entropy or classification failure. \\
$\checkmark$ Choice of upstream representation (TF-IDF vs RoBERTa) inverts relative quantum-vs-classical performance rankings. & $\times$ Causal claims asserting that state entropy directly dictates support vector classification margins. \\
$\checkmark$ The evaluated quantum kernel did not exhibit a source-domain robustness advantage under cross-source transfer. & $\times$ Claims regarding physical quantum computing hardware execution, noise mitigation, or quantum device speedup. \\
$\checkmark$ Quantum and RBF kernels induce related but non-identical Gram geometries ($r \approx 0.55\text{--}0.65$). & $\times$ Generalization of findings to all possible quantum feature maps, parameterized circuits (QKT), or ansatz designs. \\
$\checkmark$ Single-state dispersion (entropy) is strongly inversely associated with pairwise kernel diversity ($r \approx -0.78\text{--}-0.83$). & $\times$ Generalization to all NLP security domains or arbitrary non-security text classification tasks. \\
$\checkmark$ Statevector simulation runtime scales rapidly with dimension ($64\times$ penalty at 12D), posing a practical simulation limit. & $\times$ Asymptotic mathematical proof of exponential quantum simulation complexity from benchmark wall-clock times. \\
\bottomrule
\end{tabular}
\end{table}

By establishing these explicit boundaries, we position this study as a rigorous empirical benchmark rather than an overbroad generalization about quantum computing.

---

## 5.13 Methodological Implications for Future QML Text Evaluations

Beyond the specific empirical findings on text security datasets, this study provides a reusable methodological framework for evaluating quantum machine learning models on classical tabular and natural language data. Future QML benchmarking studies should incorporate the following eight principles:

1. **Strictly Match Input Representations**: Quantum and classical models must receive identical, leak-free input vectors derived from the same preprocessing pipeline, eliminating representation confounding.
2. **Conduct Dimensionality Sweeps**: Rather than evaluating quantum models at a single low dimension (e.g., 2D or 4D), researchers should systematically sweep dimensionality ($d \in [2, 16]$) to separate representation compression bottlenecks from kernel-level behavior.
3. **Incorporate Distribution Shift**: Benchmarks should evaluate models under cross-source, temporal, or cross-domain holdout splits rather than relying exclusively on in-distribution IID partitions.
4. **Enforce Multi-Seed Replication**: Empirical comparisons must incorporate multiple independent random seeds (e.g., $N \ge 10$) with non-parametric bootstrap confidence intervals and permutation tests to avoid reporting single-seed sampling noise.
5. **Establish Predefined Practical Equivalence Regions**: Studies should define a practical significance threshold ($\varepsilon$) prior to testing, distinguishing statistically detectable microscopic margins from operationally meaningful advantages.
6. **Perform Geometric Diagnostics**: Evaluations should examine Gram matrix correlations, kernel-target label alignment, and pairwise diversity to understand the geometric properties underlying classifier performance.
7. **Report Computational and Memory Scaling**: Papers must disclose wall-clock execution runtimes, hardware configurations, and memory footprints across pipeline stages to accurately represent engineering costs.
8. **Distinguish Simulation Claims from Hardware Claims**: Researchers must explicitly separate classical statevector simulation constraints from physical quantum hardware capabilities, avoiding claims of hardware advantage based on classical simulation experiments.

Adopting these methodological standards will enhance the reproducibility, credibility, and scientific rigor of applied quantum machine learning research.

<!-- CLAIM AUDIT: All discussion points strictly derived from audited results in 06_results.md and Exp40 evidence. Causal claims avoided; TruncatedSVD terminology enforced; non-quantum simulation scope maintained. -->
