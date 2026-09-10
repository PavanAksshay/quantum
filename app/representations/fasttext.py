"""
Optional FastText Subword Representation (Exploratory).
Provides subword n-gram vector averaging with graceful fallback if fasttext binaries are unavailable.
"""

from typing import List, Dict, Any, Optional
import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from app.representations.base import BaseRepresentation, ModelStatus


class FastTextRepresentation(BaseRepresentation):
    """
    Optional exploratory FastText subword representation.
    Pipeline: FastText (300D subword bag) -> TruncatedSVD (d-dim) -> StandardScaler.
    """

    def __init__(self, target_dimension: int = 8, model_path: Optional[str] = None):
        super().__init__(
            id="fasttext",
            name="FastText Subword (Optional)",
            representation_type="Static Subword Embedding (Dense)",
            is_sparse=False,
            original_dimension=300,
            target_dimension=target_dimension,
            description="Static subword n-gram embedding representation capturing morphological variations and OOV tokens.",
            preprocessing_details="fasttext-wiki-news-300d (mean token pooling) -> TruncatedSVD -> StandardScaler",
            is_canonical=False
        )
        self.model_path = model_path
        self.model = None
        self.svd_models: Dict[int, TruncatedSVD] = {}
        self.scalers: Dict[int, StandardScaler] = {}
        self.min_max_bounds: Dict[int, tuple] = {}
        self.status = ModelStatus.UNAVAILABLE
        self.status_message = "FastText unavailable — optional exploratory representation."

    def _ensure_model_loaded(self):
        """Attempts to load FastText binary if available."""
        if self.model is not None:
            return

        if not self.model_path or not os.path.exists(self.model_path):
            self.status = ModelStatus.UNAVAILABLE
            self.status_message = "FastText unavailable — optional exploratory representation (binary not present)."
            raise FileNotFoundError("FastText binary not found locally.")

        try:
            import fasttext
            self.status = ModelStatus.LOADING
            self.model = fasttext.load_model(self.model_path)
            self.status = ModelStatus.READY
            self.status_message = "FastText 300D binary loaded successfully"
        except Exception as e:
            self.status = ModelStatus.UNAVAILABLE
            self.status_message = f"FastText unavailable: {str(e)}"
            raise e

    def encode_original(self, texts: List[str]) -> np.ndarray:
        """Extracts 300D mean sentence vectors via subwords."""
        self._ensure_model_loaded()
        vectors = []
        for text in texts:
            clean = text.replace("\n", " ").strip()
            vectors.append(self.model.get_sentence_vector(clean))
        return np.array(vectors, dtype=np.float32)

    def fit(self, texts: List[str], y: Optional[np.ndarray] = None) -> "FastTextRepresentation":
        """Fits projection matrices if FastText is loaded."""
        try:
            dense_emb = self.encode_original(texts)
            for dim in [2, 4, 6, 8, 10, 12]:
                svd = TruncatedSVD(n_components=dim, random_state=42)
                X_svd = svd.fit_transform(dense_emb)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_svd)
                
                self.svd_models[dim] = svd
                self.scalers[dim] = scaler
                self.min_max_bounds[dim] = (float(np.min(X_scaled)), float(np.max(X_scaled)))

            self.is_fitted = True
            self.status = ModelStatus.READY
            self.status_message = f"Fitted FastText projections on {len(texts)} samples"
        except Exception as e:
            self.status = ModelStatus.UNAVAILABLE
            self.status_message = f"FastText unavailable — optional exploratory representation."
            # Graceful: do not crash application
        return self

    def transform(self, texts: List[str], target_dim: Optional[int] = None) -> np.ndarray:
        """Transforms text to projected FastText vector."""
        if not self.is_fitted or self.status != ModelStatus.READY:
            raise RuntimeError("FastText unavailable — optional exploratory representation.")
        dim = target_dim or self.target_dimension
        dense_emb = self.encode_original(texts)
        if dim not in self.svd_models:
            svd = TruncatedSVD(n_components=dim, random_state=42)
            X_svd = svd.fit_transform(dense_emb)
            scaler = StandardScaler()
            return scaler.fit_transform(X_svd)

        X_svd = self.svd_models[dim].transform(dense_emb)
        return self.scalers[dim].transform(X_svd)

    def get_phase_coordinates(self, texts: List[str], target_dim: Optional[int] = None) -> np.ndarray:
        X_scaled = self.transform(texts, target_dim=target_dim)
        dim = target_dim or self.target_dimension
        min_v, max_v = self.min_max_bounds.get(dim, (float(np.min(X_scaled)), float(np.max(X_scaled))))
        span = max(max_v - min_v, 1e-8)
        X_norm = (X_scaled - min_v) / span
        return np.clip(X_norm * np.pi, 0.0, np.pi)

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta.update({
            "is_optional": True,
            "supported_dimensions": [2, 4, 6, 8, 10, 12]
        })
        return meta
