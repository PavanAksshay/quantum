"""
Exp 41: Representation Screening Analysis & API Data Provider.
Decoupled exploratory results provider strictly partitioned from canonical Exp 39/40 claims.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP41_DIR = os.path.join(BASE_DIR, "results/exp41")


class Exp41ResearchManager:
    """Manages Exp41 screening results, comparison matrix, geometry, and status."""

    def __init__(self):
        self.exp_dir = EXP41_DIR

    @property
    def is_available(self) -> bool:
        return os.path.exists(os.path.join(self.exp_dir, "exp41_summary.csv"))

    def get_screening_summary(self) -> List[Dict[str, Any]]:
        """Returns aggregated summary (mean ± std across 3 seeds)."""
        csv_path = os.path.join(self.exp_dir, "exp41_summary.csv")
        if not os.path.exists(csv_path):
            return []
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")

    def get_screening_results(self, dataset: Optional[str] = None, representation: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns all 108 raw screening records with optional dataset/rep filters."""
        csv_path = os.path.join(self.exp_dir, "exp41_screening_results.csv")
        if not os.path.exists(csv_path):
            return []
        df = pd.read_csv(csv_path)
        if dataset and dataset != "all":
            df = df[df["dataset"] == dataset.lower()]
        if representation and representation != "all":
            df = df[df["representation"] == representation.lower()]
        return df.to_dict(orient="records")

    def get_representation_comparison(self) -> List[Dict[str, Any]]:
        """Returns representation-conditioned Q-RBF performance comparisons and ranges."""
        csv_path = os.path.join(self.exp_dir, "exp41_representation_comparison.csv")
        if not os.path.exists(csv_path):
            return []
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")

    def get_geometry_diagnostics(self) -> List[Dict[str, Any]]:
        """Returns geometry diagnostics (entropy, diversity, alignment, gram correlation)."""
        csv_path = os.path.join(self.exp_dir, "exp41_geometry.csv")
        if not os.path.exists(csv_path):
            return []
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")

    def get_metadata(self) -> Dict[str, Any]:
        """Returns exact reproducibility metadata."""
        meta_path = os.path.join(self.exp_dir, "exp41_metadata.json")
        if not os.path.exists(meta_path):
            return {
                "status": "NOT_EVALUATED",
                "message": "Experiment 41 screening has not yet been executed."
            }
        with open(meta_path, "r") as f:
            return json.load(f)

    def get_status(self) -> Dict[str, Any]:
        """Returns current experiment lifecycle status and workload estimation."""
        return {
            "experiment_id": "exp41",
            "name": "Controlled Representation Screening (3 Seeds)",
            "status": "READY" if self.is_available else "RUNNING",
            "canonical_status": "EXPLORATORY",
            "total_runs": 108,
            "dimension": 8,
            "seeds": [42, 123, 456],
            "datasets": ["sms", "ceas", "meajor"],
            "representations": ["tfidf", "minilm", "roberta", "mpnet"],
            "models": ["Linear SVM", "Classical RBF", "Quantum Fidelity Kernel"],
            "timing_scope": "Research benchmark runtime (3-seed screening)",
            "practical_equivalence_margin": 0.01
        }


# Singleton manager
exp41_manager = Exp41ResearchManager()
