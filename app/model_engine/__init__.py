"""
Model Engine module exports and unified ModelEngineBridge.
"""

from typing import Dict, Any, List, Optional
from app.model_engine.quantum import simulate_zz_feature_map, compute_von_neumann_entropy, extract_top_basis_states
from app.model_engine.classical import build_classical_models, compute_rbf_kernel_matrix
from app.model_engine.diagnostics import compute_target_label_alignment, compute_sample_gram_matrices
from app.model_engine.sampling import generate_2d_visualization_sample
from app.model_engine.inference import inference_engine, InferenceEngine
from app.representations.registry import representation_registry


class ModelEngineBridge:
    """Bridge providing compatibility with API callers while routing to modular engines."""

    def __init__(self):
        self._engine = inference_engine

    @property
    def is_initialized(self) -> bool:
        return self._engine.is_initialized

    @property
    def preset_samples(self) -> List[Dict[str, Any]]:
        return self._engine.preset_samples

    def initialize(self):
        self._engine.initialize()

    def predict_text(
        self,
        text: str,
        subject: Optional[str] = None,
        body: Optional[str] = None,
        representation_id: str = "tfidf",
        dimension: int = 8,
        dataset: str = "MeAJOR"
    ) -> Dict[str, Any]:
        return self._engine.predict(
            text=text,
            subject=subject,
            body=body,
            representation_id=representation_id,
            dimension=dimension,
            dataset=dataset
        )

    def get_sample_gram_matrix(self, representation_id: str = "tfidf", sample_size: int = 20) -> Dict[str, Any]:
        if not self.is_initialized:
            self.initialize()
        rep = representation_registry.get(representation_id)
        if not rep.is_fitted:
            rep.fit(self._engine.train_texts)
        X_phase = rep.get_phase_coordinates(self._engine.train_texts, target_dim=8)
        return compute_sample_gram_matrices(X_phase, sample_size=sample_size)

    def get_2d_embedding_sample(self, representation_id: str = "tfidf", max_points: int = 150) -> Dict[str, Any]:
        if not self.is_initialized:
            self.initialize()
        rep = representation_registry.get(representation_id)
        if not rep.is_fitted:
            rep.fit(self._engine.train_texts)
        X_scaled = rep.transform(self._engine.train_texts, target_dim=8)
        return generate_2d_visualization_sample(
            embeddings=X_scaled,
            labels=self._engine.train_y,
            texts=self._engine.train_texts,
            max_points=max_points,
            projection_method="PCA"
        )


# Global singleton instance
model_engine = ModelEngineBridge()

__all__ = [
    "simulate_zz_feature_map",
    "compute_von_neumann_entropy",
    "extract_top_basis_states",
    "build_classical_models",
    "compute_rbf_kernel_matrix",
    "compute_target_label_alignment",
    "compute_sample_gram_matrices",
    "generate_2d_visualization_sample",
    "inference_engine",
    "InferenceEngine",
    "model_engine",
    "ModelEngineBridge"
]
