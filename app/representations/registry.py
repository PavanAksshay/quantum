"""
Central Representation Registry & Manager.
Maintains singleton instances, lifecycle status, caching, and verified metadata.
"""

from typing import Dict, List, Any, Optional
from app.representations.base import BaseRepresentation, ModelStatus
from app.representations.tfidf import TfidfRepresentation
from app.representations.roberta import RoBERTaRepresentation
from app.representations.minilm import MiniLMRepresentation
from app.representations.mpnet import MPNetRepresentation
from app.representations.fasttext import FastTextRepresentation


class RepresentationRegistry:
    """
    Singleton registry managing all representation modules.
    Guarantees that models are loaded once, cached, and never silently substituted.
    """

    def __init__(self):
        self._representations: Dict[str, BaseRepresentation] = {
            "tfidf": TfidfRepresentation(target_dimension=8),
            "roberta": RoBERTaRepresentation(target_dimension=8),
            "minilm": MiniLMRepresentation(target_dimension=8),
            "mpnet": MPNetRepresentation(target_dimension=8),
            "fasttext": FastTextRepresentation(target_dimension=8)
        }
        self.is_initialized = False

    def get(self, representation_id: str) -> BaseRepresentation:
        """Retrieves representation by ID. Raises KeyError if unknown."""
        rep_id = representation_id.lower().strip()
        if rep_id not in self._representations:
            available = list(self._representations.keys())
            raise KeyError(f"Representation '{representation_id}' not found. Available: {available}")
        return self._representations[rep_id]

    def list_all(self) -> List[Dict[str, Any]]:
        """Returns metadata for all registered representations."""
        return [rep.get_metadata() for rep in self._representations.values()]

    def list_available(self) -> List[Dict[str, Any]]:
        """Returns metadata only for ready/available representations."""
        return [
            rep.get_metadata() for rep in self._representations.values()
            if rep.status in [ModelStatus.READY, ModelStatus.AVAILABLE]
        ]

    def fit_all(self, training_texts: List[str]):
        """Fits canonical TF-IDF representation and stores reference texts for lazy model fitting."""
        self._reference_texts = training_texts
        # 1. Canonical TF-IDF (Always pre-fitted on startup)
        try:
            self._representations["tfidf"].fit(training_texts)
        except Exception as e:
            print(f"Error fitting TF-IDF: {e}")

        self.is_initialized = True
        print("RepresentationRegistry initialization complete (TF-IDF ready, transformers lazy-fit).")

    def ensure_fitted(self, representation_id: str):
        """Ensures the requested representation is fitted with projection matrices."""
        rep = self.get(representation_id)
        if not rep.is_fitted:
            texts = getattr(self, "_reference_texts", None)
            if texts is not None:
                rep.fit(texts[:100] if len(texts) > 100 else texts)


# Global singleton instance
representation_registry = RepresentationRegistry()
