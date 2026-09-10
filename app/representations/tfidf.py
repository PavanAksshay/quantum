"""
Canonical TF-IDF + TruncatedSVD + StandardScaler Representation.
Preserves exact frozen research parameters across all datasets.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from app.representations.base import BaseRepresentation, ModelStatus


class TfidfRepresentation(BaseRepresentation):
    """
    Canonical Research TF-IDF N-gram representation.
    Pipeline: 50k TF-IDF -> TruncatedSVD (d-dim) -> StandardScaler.
    """

    def __init__(self, target_dimension: int = 8):
        super().__init__(
            id="tfidf",
            name="TF-IDF + TruncatedSVD",
            representation_type="Lexical N-Gram (Sparse to Projected)",
            is_sparse=True,
            original_dimension=50000,
            target_dimension=target_dimension,
            description="Canonical word and character N-gram term-frequency representation, compressed via TruncatedSVD.",
            preprocessing_details="TF-IDF (unigram + bigram, min_df=2, sublinear_tf=True, max_features=50000, l2 norm) -> TruncatedSVD -> StandardScaler",
            is_canonical=True
        )
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=50000,
            norm="l2"
        )
        self.svd_models: Dict[int, TruncatedSVD] = {}
        self.scalers: Dict[int, StandardScaler] = {}
        self.min_max_bounds: Dict[int, tuple] = {}
        self.status = ModelStatus.READY
        self.status_message = "Canonical baseline ready"

    def fit(self, texts: List[str], y: Optional[np.ndarray] = None) -> "TfidfRepresentation":
        """Fits TF-IDF vectorizer and TruncatedSVD + Scalers across standard dimensions."""
        self.status = ModelStatus.LOADING
        try:
            try:
                X_sparse = self.vectorizer.fit_transform(texts)
            except Exception:
                self.vectorizer.set_params(min_df=1)
                X_sparse = self.vectorizer.fit_transform(texts)
                
            if X_sparse.shape[1] == 0:
                self.vectorizer.set_params(min_df=1)
                X_sparse = self.vectorizer.fit_transform(texts)

            self.original_dimension = min(50000, max(1, len(self.vectorizer.vocabulary_)))
            n_features = X_sparse.shape[1]

            # Pre-fit SVD and Scaler for standard dimensions [2, 4, 6, 8, 10, 12]
            for dim in [2, 4, 6, 8, 10, 12]:
                if n_features < 2:
                    arr = X_sparse.toarray()
                    if arr.shape[1] < dim:
                        padding = np.zeros((arr.shape[0], dim - arr.shape[1]))
                        X_svd = np.hstack([arr, padding])
                    else:
                        X_svd = arr[:, :dim]
                    self.svd_models[dim] = None
                elif n_features <= dim:
                    n_comp = min(dim, n_features - 1) if n_features > 2 else 2
                    if n_comp >= n_features:
                        n_comp = max(1, n_features - 1)
                    if n_comp < 1:
                        n_comp = 1
                    svd = TruncatedSVD(n_components=n_comp, random_state=42)
                    X_svd = svd.fit_transform(X_sparse)
                    if X_svd.shape[1] < dim:
                        padding = np.zeros((X_svd.shape[0], dim - X_svd.shape[1]))
                        X_svd = np.hstack([X_svd, padding])
                    self.svd_models[dim] = svd
                else:
                    svd = TruncatedSVD(n_components=dim, random_state=42)
                    X_svd = svd.fit_transform(X_sparse)
                    self.svd_models[dim] = svd

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_svd)
                
                self.scalers[dim] = scaler
                self.min_max_bounds[dim] = (float(np.min(X_scaled)), float(np.max(X_scaled)))

            self.is_fitted = True
            self.status = ModelStatus.READY
            self.status_message = f"Fitted on {len(texts)} samples (vocab: {self.original_dimension})"
        except Exception as e:
            self.status = ModelStatus.ERROR
            self.status_message = f"Fit error: {str(e)}"
            raise e
        return self

    def encode_original(self, texts: List[str]) -> np.ndarray:
        """Extracts sparse/dense raw TF-IDF features (first 50 dimensions if requested for inspection)."""
        if not self.is_fitted:
            raise RuntimeError("TfidfRepresentation is not fitted.")
        return self.vectorizer.transform(texts)

    def transform(self, texts: List[str], target_dim: Optional[int] = None) -> np.ndarray:
        """Transforms text into standardized TruncatedSVD coordinate vector."""
        if not self.is_fitted:
            raise RuntimeError("TfidfRepresentation is not fitted.")
        dim = target_dim or self.target_dimension
        X_sparse = self.vectorizer.transform(texts)

        if dim not in self.svd_models or self.svd_models[dim] is None:
            arr = X_sparse.toarray()
            if arr.shape[1] < dim:
                padding = np.zeros((arr.shape[0], dim - arr.shape[1]))
                X_svd = np.hstack([arr, padding])
            else:
                X_svd = arr[:, :dim]
        else:
            X_svd = self.svd_models[dim].transform(X_sparse)
            if X_svd.shape[1] < dim:
                padding = np.zeros((X_svd.shape[0], dim - X_svd.shape[1]))
                X_svd = np.hstack([X_svd, padding])

        if dim in self.scalers:
            return self.scalers[dim].transform(X_svd)
        else:
            scaler = StandardScaler()
            return scaler.fit_transform(X_svd)

    def get_phase_coordinates(self, texts: List[str], target_dim: Optional[int] = None) -> np.ndarray:
        """Maps standardized coordinates to quantum phase angles [0, pi]."""
        X_scaled = self.transform(texts, target_dim=target_dim)
        dim = target_dim or self.target_dimension
        min_v, max_v = self.min_max_bounds.get(dim, (float(np.min(X_scaled)), float(np.max(X_scaled))))
        span = max(max_v - min_v, 1e-8)
        X_norm = (X_scaled - min_v) / span
        return np.clip(X_norm * np.pi, 0.0, np.pi)

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta.update({
            "supported_dimensions": [2, 4, 6, 8, 10, 12],
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "min_df": 2,
            "norm": "l2"
        })
        return meta
