"""
Base representation abstraction for Quantum Text Security research platform.
Defines interface, metadata structures, and lifecycle states.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np


class ModelStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    DOWNLOADING = "DOWNLOADING"
    LOADING = "LOADING"
    READY = "READY"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


class BaseRepresentation(ABC):
    """
    Abstract Base Class for all text representations.
    Provides standard fit, transform, encode, and dimensionality projection contracts.
    """

    def __init__(
        self,
        id: str,
        name: str,
        representation_type: str,
        is_sparse: bool,
        original_dimension: int,
        target_dimension: int = 8,
        description: str = "",
        preprocessing_details: str = "",
        is_canonical: bool = False
    ):
        self.id = id
        self.name = name
        self.representation_type = representation_type  # e.g., "lexical_ngram", "transformer_contextual", "subword_static"
        self.is_sparse = is_sparse
        self.original_dimension = original_dimension
        self.target_dimension = target_dimension
        self.description = description
        self.preprocessing_details = preprocessing_details
        self.is_canonical = is_canonical
        self.status = ModelStatus.AVAILABLE
        self.status_message = "Ready for initialization"
        self.is_fitted = False

    @abstractmethod
    def fit(self, texts: List[str], y: Optional[np.ndarray] = None) -> "BaseRepresentation":
        """Fits the vocabulary or projection matrices on training texts."""
        pass

    @abstractmethod
    def encode_original(self, texts: List[str]) -> np.ndarray:
        """Extracts the original (unprojected) representation."""
        pass

    @abstractmethod
    def transform(self, texts: List[str], target_dim: Optional[int] = None) -> np.ndarray:
        """Transforms texts into the standardized d-dimensional projected representation."""
        pass

    def encode(self, texts: List[str], target_dim: Optional[int] = None) -> np.ndarray:
        """Convenience method returning projected representation."""
        return self.transform(texts, target_dim=target_dim)

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Returns verified representation metadata."""
        return {
            "id": self.id,
            "name": self.name,
            "representation_type": self.representation_type,
            "is_sparse": self.is_sparse,
            "original_dimension": self.original_dimension,
            "target_dimension": self.target_dimension,
            "status": self.status.value,
            "status_message": self.status_message,
            "description": self.description,
            "preprocessing_details": self.preprocessing_details,
            "is_canonical": self.is_canonical,
            "is_fitted": self.is_fitted
        }
