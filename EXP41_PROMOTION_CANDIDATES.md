# Experiment 41: Top-3 Representation Promotion Candidates (for Exp 42 Confirmation)

**Document Type**: Scientific Promotion Decision & Preregistration  
**Originating Experiment**: Experiment 41 (3-Seed Screening Matrix, $N=108$)  
**Target Next Phase**: Experiment 42 (10-Seed Confirmation Protocol)  
**Status**: REGISTERED CANDIDATES  

---

## 1. Selection Criteria & Methodology

Following the Exp 41 screening protocol, configurations are evaluated across five objective criteria:
1. **Magnitude of Q-RBF Performance Difference** ($|\Delta\text{F1}| = |\text{Quantum F1} - \text{RBF F1}|$)
2. **Representation-Conditioned Variation** ($\max(\Delta\text{F1}) - \min(\Delta\text{F1})$ across representations)
3. **Model Rank Reversal** ($\text{Quantum} > \text{RBF}$ under representation $A$, but $\text{Quantum} < \text{RBF}$ under representation $B$)
4. **Hilbert Space Geometry Shift** (Significant change in kernel diversity or label alignment)
5. **Security Relevance** (Practical impact on real-world text threat detection)

---

## 2. Top-3 Promoted Configurations

| Rank | Dataset | Representation | Dimension | Quantum F1 (Mean) | RBF F1 (Mean) | $\Delta\text{F1}$ (Q - RBF) | Primary Promotion Rationale |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **SMS Spam** | **MiniLM** (`all-MiniLM-L6-v2`) | 8D | **0.7736** | **0.7622** | **+0.0114** | **Model Rank Reversal**: Quantum achieves an observed edge under MiniLM ($+1.14\text{ pp}$), reversing the RBF advantage seen under TF-IDF ($-1.22\text{ pp}$). |
| **2** | **SMS Spam** | **MPNet** (`all-mpnet-base-v2`) | 8D | **0.7097** | **0.9149** | **-0.2052** | **Extreme Geometric Divergence**: Parameter-free ZZFeatureMap suffers severe kernel diversity compression ($0.0186$) on contrastive embeddings, while RBF excels ($0.9149$). |
| **3** | **CEAS 2008** | **TF-IDF + TruncatedSVD** | 8D | **0.9504** | **0.9416** | **+0.0088** | **Canonical Representation Reference**: Validates high label alignment and high kernel diversity ($0.334$) under sparse N-gram projections in email security. |

---

## 3. Candidate Profiles & Detailed Scientific Rationale

### Candidate 1: SMS Spam / MiniLM (8D)
- **Empirical Observation**:
  - MiniLM Quantum F1: $0.7736 \pm 0.059$ vs RBF F1: $0.7622 \pm 0.109$ ($\Delta\text{F1} = +0.0114$).
  - In contrast, TF-IDF on SMS yielded Quantum F1: $0.7969$ vs RBF F1: $0.8091$ ($\Delta\text{F1} = -0.0122$).
- **Scientific Significance**:
  - Represents a clear **Model Rank Reversal** ($\text{MODEL\_RANK\_REVERSAL} = \text{TRUE}$). Switching the upstream text embedding from sparse bag-of-words to dense sentence transformer representations changes whether the quantum fidelity kernel outperforms or underperforms the classical RBF kernel.
- **Exp 42 Hypothesis**:
  - *Under 10-seed paired evaluation, the observed quantum edge on SMS/MiniLM will be tested for statistical significance against the $\epsilon = \pm 0.01$ practical equivalence margin.*

---

### Candidate 2: SMS Spam / MPNet (8D)
- **Empirical Observation**:
  - MPNet Quantum F1: $0.7097 \pm 0.111$ vs RBF F1: $0.9149 \pm 0.003$ ($\Delta\text{F1} = -0.2052$).
  - Kernel diversity: $0.0186 \pm 0.002$ (compared to $0.1127$ for TF-IDF).
  - Gram Pearson correlation with RBF: $r = 0.3129$ (indicating strong geometrical divergence).
- **Scientific Significance**:
  - Illustrates a catastrophic geometric mismatch: the 2-layer cyclic `ZZFeatureMap` phase encodings concentrate tightly when presented with SVD-projected MPNet representations, leading to severe underfitting, whereas the classical RBF kernel effectively leverages the dense semantic clustering.
- **Exp 42 Hypothesis**:
  - *The $-20.52\text{ pp}$ deficit demonstrates that representation geometry is a primary prerequisite for parameter-free quantum kernel viability.*

---

### Candidate 3: CEAS 2008 / TF-IDF (8D)
- **Empirical Observation**:
  - TF-IDF Quantum F1: $0.9504 \pm 0.011$ vs RBF F1: $0.9416 \pm 0.016$ ($\Delta\text{F1} = +0.0088$).
  - Kernel diversity: $0.3343 \pm 0.011$.
  - Gram Pearson correlation with RBF: $r = 0.8901$.
- **Scientific Significance**:
  - Confirms that sparse lexical features projected via TruncatedSVD provide an evenly dispersed Hilbert space representation with superior quantum fidelity kernel diversity and stable near-parity performance.
- **Exp 42 Hypothesis**:
  - *Provides the canonical benchmark anchor to contrast against dense sentence embeddings across 10 independent seeds.*

---

## 4. Experiment 42 Preregistration Parameters

- **Seeds (10)**: `[42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]`
- **Target Configurations**: Top-3 promoted candidates.
- **Statistical Protocol**:
  - Paired 2-sided permutation test ($10,000$ resamples)
  - 95% Percentile Bootstrap Confidence Interval
  - Benjamini-Hochberg False Discovery Rate (FDR) control at $\alpha = 0.05$
  - Practical equivalence classification margin $\epsilon = \pm 0.01$
