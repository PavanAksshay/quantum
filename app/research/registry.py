"""
Central Experiment Registry for Quantum Text Security.
Maintains authoritative catalog of experiments, distinguishing CANONICAL paper results from EXPLORATORY ablations.
"""

from enum import Enum
from typing import Dict, List, Any, Optional


class ExperimentStatus(str, Enum):
    CANONICAL = "CANONICAL"          # Frozen peer-reviewed protocol (Exp 39/40)
    EXPLORATORY = "EXPLORATORY"      # Active exploratory representation lab ablation
    NOT_EVALUATED = "NOT_EVALUATED"  # Combination not experimentally executed
    UNAVAILABLE = "UNAVAILABLE"      # Required model/weights/hardware unavailable


class ExperimentRecord:
    def __init__(
        self,
        dataset: str,
        representation: str,
        dimension: int,
        kernel: str,
        f1: Optional[float],
        delta_f1: Optional[float] = None,
        pr_auc: Optional[float] = None,
        roc_auc: Optional[float] = None,
        runtime_s: Optional[float] = None,
        p_value: Optional[float] = None,
        ci_95: Optional[List[float]] = None,
        status: ExperimentStatus = ExperimentStatus.CANONICAL,
        notes: str = ""
    ):
        self.dataset = dataset
        self.representation = representation
        self.dimension = dimension
        self.kernel = kernel
        self.f1 = f1
        self.delta_f1 = delta_f1
        self.pr_auc = pr_auc
        self.roc_auc = roc_auc
        self.runtime_s = runtime_s
        self.p_value = p_value
        self.ci_95 = ci_95
        self.status = status
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "representation": self.representation,
            "dimension": self.dimension,
            "kernel": self.kernel,
            "f1": self.f1,
            "delta_f1": self.delta_f1,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "runtime_s": self.runtime_s,
            "p_value": self.p_value,
            "ci_95": self.ci_95,
            "status": self.status.value,
            "notes": self.notes
        }


class ExperimentRegistry:
    """
    Registry indexing all evaluated model and experiment combinations.
    Ensures that unsupported combinations cleanly return NOT_EVALUATED and never fabricate numbers.
    """

    def __init__(self):
        self._records: List[ExperimentRecord] = []
        self._populate_canonical_records()
        self._populate_exploratory_records()

    def _populate_canonical_records(self):
        # 1. Canonical MeAJOR In-Distribution Scaling (TF-IDF + SVD)
        canonical_scaling = [
            (2, 0.6447, 0.6735, 0.6853, -0.0288, 3.4),
            (4, 0.7876, 0.7918, 0.7584, -0.0042, 5.8),
            (6, 0.8253, 0.8254, 0.7946, -0.0001, 9.4),
            (8, 0.8754, 0.8709, 0.8445, 0.0046, 17.2),
            (10, 0.9023, 0.8967, 0.8730, 0.0057, 44.6),
            (12, 0.9137, 0.9123, 0.8892, 0.0014, 108.8)
        ]
        for dim, q_f1, rbf_f1, lin_f1, delta, runtime in canonical_scaling:
            self._records.append(ExperimentRecord("MeAJOR", "tfidf", dim, "quantum", q_f1, delta_f1=delta, runtime_s=runtime, status=ExperimentStatus.CANONICAL))
            self._records.append(ExperimentRecord("MeAJOR", "tfidf", dim, "rbf", rbf_f1, runtime_s=1.5, status=ExperimentStatus.CANONICAL))
            self._records.append(ExperimentRecord("MeAJOR", "tfidf", dim, "linear", lin_f1, runtime_s=0.2, status=ExperimentStatus.CANONICAL))

        # 2. Canonical CEAS 2008 Representation Ablation (8D)
        # TF-IDF
        self._records.append(ExperimentRecord("CEAS_08", "tfidf", 8, "quantum", 0.9736, delta_f1=0.0095, pr_auc=0.9812, roc_auc=0.9854, status=ExperimentStatus.CANONICAL, notes="TF-IDF canonical baseline"))
        self._records.append(ExperimentRecord("CEAS_08", "tfidf", 8, "rbf", 0.9641, pr_auc=0.9754, roc_auc=0.9801, status=ExperimentStatus.CANONICAL))
        self._records.append(ExperimentRecord("CEAS_08", "tfidf", 8, "linear", 0.9520, pr_auc=0.9640, roc_auc=0.9710, status=ExperimentStatus.CANONICAL))
        
        # RoBERTa
        self._records.append(ExperimentRecord("CEAS_08", "roberta", 8, "quantum", 0.9601, delta_f1=-0.0295, pr_auc=0.9680, roc_auc=0.9740, status=ExperimentStatus.CANONICAL, notes="RoBERTa canonical ablation"))
        self._records.append(ExperimentRecord("CEAS_08", "roberta", 8, "rbf", 0.9896, pr_auc=0.9940, roc_auc=0.9960, status=ExperimentStatus.CANONICAL))
        self._records.append(ExperimentRecord("CEAS_08", "roberta", 8, "linear", 0.9780, pr_auc=0.9820, roc_auc=0.9870, status=ExperimentStatus.CANONICAL))

        # 3. Canonical Cross-Source Holdout (Direction B, 8D TF-IDF)
        self._records.append(ExperimentRecord("TREC_Holdout_B", "tfidf", 8, "quantum", 0.6680, delta_f1=-0.0233, p_value=0.0046, ci_95=[-0.0353, -0.0117], status=ExperimentStatus.CANONICAL, notes="Severe vocabulary drift (>92% type OOV)"))
        self._records.append(ExperimentRecord("TREC_Holdout_B", "tfidf", 8, "rbf", 0.6913, status=ExperimentStatus.CANONICAL))
        self._records.append(ExperimentRecord("TREC_Holdout_B", "tfidf", 8, "linear", 0.6810, status=ExperimentStatus.CANONICAL))

    def _populate_exploratory_records(self):
        # Exploratory sentence embeddings on CEAS 2008 (8D)
        # MiniLM
        self._records.append(ExperimentRecord("CEAS_08", "minilm", 8, "quantum", 0.9645, delta_f1=-0.0185, pr_auc=0.9710, roc_auc=0.9765, status=ExperimentStatus.EXPLORATORY, notes="Exploratory MiniLM sentence embedding"))
        self._records.append(ExperimentRecord("CEAS_08", "minilm", 8, "rbf", 0.9830, pr_auc=0.9890, roc_auc=0.9920, status=ExperimentStatus.EXPLORATORY))
        self._records.append(ExperimentRecord("CEAS_08", "minilm", 8, "linear", 0.9715, pr_auc=0.9780, roc_auc=0.9830, status=ExperimentStatus.EXPLORATORY))

        # MPNet
        self._records.append(ExperimentRecord("CEAS_08", "mpnet", 8, "quantum", 0.9682, delta_f1=-0.0192, pr_auc=0.9740, roc_auc=0.9790, status=ExperimentStatus.EXPLORATORY, notes="Exploratory MPNet sentence embedding"))
        self._records.append(ExperimentRecord("CEAS_08", "mpnet", 8, "rbf", 0.9874, pr_auc=0.9925, roc_auc=0.9950, status=ExperimentStatus.EXPLORATORY))
        self._records.append(ExperimentRecord("CEAS_08", "mpnet", 8, "linear", 0.9760, pr_auc=0.9810, roc_auc=0.9860, status=ExperimentStatus.EXPLORATORY))

    def query(
        self,
        dataset: Optional[str] = None,
        representation: Optional[str] = None,
        dimension: Optional[int] = None,
        kernel: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Queries records with filters."""
        results = []
        for rec in self._records:
            if dataset and rec.dataset.lower() != dataset.lower():
                continue
            if representation and rec.representation.lower() != representation.lower():
                continue
            if dimension and rec.dimension != dimension:
                continue
            if kernel and rec.kernel.lower() != kernel.lower():
                continue
            results.append(rec.to_dict())
        return results

    def get_comparison_matrix(self) -> List[Dict[str, Any]]:
        """Returns the Representation Comparison Matrix."""
        reps = ["tfidf", "roberta", "minilm", "mpnet", "fasttext"]
        matrix = []
        for rep in reps:
            q_recs = self.query(dataset="CEAS_08", representation=rep, dimension=8, kernel="quantum")
            rbf_recs = self.query(dataset="CEAS_08", representation=rep, dimension=8, kernel="rbf")
            lin_recs = self.query(dataset="CEAS_08", representation=rep, dimension=8, kernel="linear")

            if q_recs and rbf_recs:
                q_rec = q_recs[0]
                rbf_rec = rbf_recs[0]
                lin_rec = lin_recs[0] if lin_recs else None
                matrix.append({
                    "representation_id": rep,
                    "name": "TF-IDF + TruncatedSVD" if rep == "tfidf" else ("RoBERTa-base" if rep == "roberta" else ("MiniLM-L6" if rep == "minilm" else ("MPNet-base" if rep == "mpnet" else "FastText"))),
                    "original_dim": 50000 if rep == "tfidf" else (768 if rep in ["roberta", "mpnet"] else (384 if rep == "minilm" else 300)),
                    "projected_dim": 8,
                    "type": "Lexical N-Gram (Sparse)" if rep == "tfidf" else ("Transformer Contextual (Dense)" if rep == "roberta" else ("Sentence Transformer (Dense)" if rep in ["minilm", "mpnet"] else "Static Subword (Dense)")),
                    "is_sparse": rep == "tfidf",
                    "quantum_f1": q_rec["f1"],
                    "rbf_f1": rbf_rec["f1"],
                    "linear_f1": lin_rec["f1"] if lin_rec else None,
                    "delta_f1": q_rec["delta_f1"],
                    "pr_auc": q_rec["pr_auc"],
                    "roc_auc": q_rec["roc_auc"],
                    "status": q_rec["status"],
                    "evaluated": True
                })
            else:
                matrix.append({
                    "representation_id": rep,
                    "name": "FastText Subword" if rep == "fasttext" else rep.upper(),
                    "original_dim": 300 if rep == "fasttext" else 768,
                    "projected_dim": 8,
                    "type": "Static Subword Embedding" if rep == "fasttext" else "Dense Embedding",
                    "is_sparse": False,
                    "quantum_f1": None,
                    "rbf_f1": None,
                    "linear_f1": None,
                    "delta_f1": None,
                    "pr_auc": None,
                    "roc_auc": None,
                    "status": "EXPLORATORY",
                    "evaluated": False
                })
        return matrix


# Singleton experiment registry
experiment_registry = ExperimentRegistry()
