# Experiment 41: Comprehensive Research & System Audit

**Audit Date**: September 2026  
**Auditor**: Antigravity Research Agent  
**Subsystem**: Controlled Representation Screening (Exp 41)  
**Status**: EXPLORATORY AUDIT COMPLETE  

---

## 1. Executive Summary & Verification Matrix

| Audit Item | Verification Status | Notes |
| :--- | :--- | :--- |
| **Existing Full-Stack Platform** | PASSED | Live prediction, geometry, benchmark, and narrative views fully operational |
| **Canonical Results Preservation** | PASSED | `results/exp39_paper/`, `results/exp40_final/`, `RESULT_FREEZE.md`, and `FINAL_EVIDENCE_MATRIX.csv` are strictly untouched |
| **Executable Candidate Representations** | PASSED | MiniLM (`all-MiniLM-L6-v2`), RoBERTa (`roberta-base`), MPNet (`all-mpnet-base-v2`), and TF-IDF executed natively |
| **Leakage Protection Protocol** | PASSED | `TfidfVectorizer`, `TruncatedSVD`, and `StandardScaler` fit strictly on training partitions; validation-only 200-step threshold grid |
| **Quantum Exact Simulation** | PASSED | 2-layer cyclic `ZZFeatureMap`, $K_Q(x,z) = \|\langle\psi(x)\|\psi(z)\rangle\|^2$, PyTorch `complex128` exact statevectors |
| **Matched Classical Baselines** | PASSED | Matched 8D representation supplied to Classical RBF SVM and Linear SVM baselines |
| **Caching Subsystem** | PASSED | `results/exp41/cache/<dataset>/<rep>/` verified with `.npy` matrices |
| **Frontend Integration** | PASSED | Exp 41 screening dashboard with Q-RBF $\Delta\text{F1}$ chart, $\epsilon=\pm0.01$ bands, heatmap, geometry diagnostics, and 108-run table |
| **Scientific Wording & Terminology** | PASSED | Decision score semantics enforced; non-causal language applied; timing scopes distinguished |

---

## 2. Model Availability & Verification

- **TF-IDF + TruncatedSVD**: `AVAILABLE` (50,000 max features, sublinear TF, 8D SVD projection).
- **MiniLM**: `AVAILABLE` (`sentence-transformers/all-MiniLM-L6-v2`, 384D original, mean-pool, 8D SVD projection).
- **RoBERTa**: `AVAILABLE` (`roberta-base`, 768D original, mean-pool, 8D SVD projection).
- **MPNet**: `AVAILABLE` (`sentence-transformers/all-mpnet-base-v2`, 768D original, mean-pool, 8D SVD projection).
- **FastText**: `EXPLORATORY / NOT EVALUATED` (reserved for future subword exploration).

---

## 3. Preprocessing & Leakage Integrity Audit

- **Partition Isolation**: Every dataset split (`train`, `val`, `test`) is deterministically partitioned by random seed before any preprocessing occurs.
- **Fitting Constraints**:
  - `TfidfVectorizer.fit(X_train)`
  - `TruncatedSVD.fit(X_train_emb)`
  - `StandardScaler.fit(X_train_proj)`
- **Evaluation Independence**: Validation split is used exclusively to find the optimal F1 decision boundary over a 200-point uniform grid. Test split metrics are computed using this frozen threshold.

---

## 4. Hardware & Environment Metadata

- **Platform**: macOS ARM64 (Apple Silicon)
- **Python Runtime**: Python 3.12 (CPython)
- **PyTorch Version**: 2.1.2 (`torch.complex128` statevector simulation)
- **Scikit-Learn Version**: 1.4.0
- **NumPy Version**: 1.26.4
- **SciPy Version**: 1.12.0
- **FastAPI / Uvicorn**: 0.109.0 / 0.27.0
- **Frontend Stack**: React 19, Vite 8, Recharts 2.12, Lucide React

---

## 5. Timing Scope & Terminology Compliance

1. **Benchmark vs Live Timing**:
   - Simulation overhead (e.g. 108.8s Quantum vs 1.7s RBF on 10k benchmark) is explicitly scoped as **“Research benchmark runtime”**.
   - Interactive demo measurements are scoped as **“Live single-sample inference latency”**.
2. **Decision Scores**: All SVM model outputs in UI and API return `decision_score` values, avoiding false probability calibration claims.
3. **Epistemic Modesty**: Raw positive Q-RBF performance differences are documented as **“Observed quantum edge”** or **“Observed Q-RBF difference”** pending Exp 42 10-seed confirmation.
