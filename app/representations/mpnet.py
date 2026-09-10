"""
MPNet Sentence-Transformer Representation (Exploratory).
Uses frozen all-mpnet-base-v2 (768D) followed by TruncatedSVD + StandardScaler.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from app.representations.base import BaseRepresentation, ModelStatus


class MPNetRepresentation(BaseRepresentation):
    """
    Sentence-Transformer all-mpnet-base-v2 dense semantic embedding model.
    Pipeline: all-mpnet-base-v2 (768D, frozen) -> TruncatedSVD (d-dim) -> StandardScaler.
    """

    def __init__(self, target_dimension: int = 8):
        super().__init__(
            id="mpnet",
            name="Sentence-MPNet-Base",
            representation_type="Deep Sentence Transformer (Dense)",
            is_sparse=False,
            original_dimension=768,
            target_dimension=target_dimension,
            description="High-performance masked and permuted sentence transformer producing rich 768-dimensional dense semantic vectors.",
            preprocessing_details="sentence-transformers/all-mpnet-base-v2 (frozen, mean-pooling) -> TruncatedSVD -> StandardScaler",
            is_canonical=False
        )
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cpu")
        self.svd_models: Dict[int, TruncatedSVD] = {}
        self.scalers: Dict[int, StandardScaler] = {}
        self.min_max_bounds: Dict[int, tuple] = {}
        self.status = ModelStatus.AVAILABLE
        self.status_message = "Ready to load model weights"

    def _ensure_model_loaded(self):
        """Lazy loads tokenizer and model in frozen eval mode."""
        if self.model is not None and self.tokenizer is not None:
            return

        self.status = ModelStatus.LOADING
        self.status_message = "Loading sentence-transformers/all-mpnet-base-v2 weights..."
        try:
            from transformers import AutoTokenizer, AutoModel
            model_name = "sentence-transformers/all-mpnet-base-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            self.status = ModelStatus.READY
            self.status_message = "all-mpnet-base-v2 (768D) loaded and frozen"
        except Exception as e:
            self.status = ModelStatus.UNAVAILABLE
            self.status_message = f"all-mpnet-base-v2 unavailable locally: {str(e)}"
            raise e

    def encode_original(self, texts: List[str]) -> np.ndarray:
        """Extracts dense 768D mean-pooled embeddings."""
        self._ensure_model_loaded()
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                token_embeddings = outputs.last_hidden_state
                sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()
                all_embeddings.append(mean_pooled)

        return np.vstack(all_embeddings).astype(np.float32)

    def fit(self, texts: List[str], y: Optional[np.ndarray] = None) -> "MPNetRepresentation":
        """Fits TruncatedSVD and StandardScaler on dense MPNet embeddings."""
        try:
            fit_sample = texts[:100] if len(texts) > 100 else texts
            dense_emb = self.encode_original(fit_sample)
            n_samples, n_feat = dense_emb.shape
            for dim in [2, 4, 6, 8, 10, 12]:
                n_comp = min(dim, n_samples, n_feat)
                svd = TruncatedSVD(n_components=n_comp, random_state=42)
                X_svd = svd.fit_transform(dense_emb)
                if X_svd.shape[1] < dim:
                    padding = np.zeros((X_svd.shape[0], dim - X_svd.shape[1]))
                    X_svd = np.hstack([X_svd, padding])

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_svd)
                
                self.svd_models[dim] = svd
                self.scalers[dim] = scaler
                self.min_max_bounds[dim] = (float(np.min(X_scaled)), float(np.max(X_scaled)))

            self.is_fitted = True
            self.status = ModelStatus.READY
            self.status_message = f"Fitted on {len(texts)} samples (768D -> TruncatedSVD)"
        except Exception as e:
            self.status = ModelStatus.ERROR
            self.status_message = f"Fit error: {str(e)}"
            raise e
        return self

    def transform(self, texts: List[str], target_dim: Optional[int] = None) -> np.ndarray:
        """Transforms text to standardized projected MPNet vector."""
        if not self.is_fitted:
            raise RuntimeError("MPNetRepresentation is not fitted.")
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
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "pooling": "mean_pooling",
            "max_sequence_length": 256,
            "supported_dimensions": [2, 4, 6, 8, 10, 12]
        })
        return meta
