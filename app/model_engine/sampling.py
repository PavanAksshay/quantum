"""
Visualization-only 2D Embedding Space Sampler.
Produces 2D coordinates for interactive UI scatterplots with explicit projection labeling.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.decomposition import PCA


def generate_2d_visualization_sample(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    max_points: int = 150,
    projection_method: str = "PCA"
) -> Dict[str, Any]:
    """
    Generates a 2D sample for dashboard scatterplots.
    Explicitly marked as a visualization-only projection so as not to confuse with the TruncatedSVD pipeline.
    """
    n_total = len(texts)
    if n_total == 0:
        return {"points": [], "total_points": 0, "sample_size": 0, "projection_method": projection_method}

    sample_size = min(max_points, n_total)
    # Stratified or evenly spaced index selection
    indices = np.linspace(0, n_total - 1, sample_size, dtype=int)

    sub_emb = embeddings[indices]
    sub_labels = labels[indices]
    sub_texts = [texts[i] for i in indices]

    # Fit 2D PCA for visual projection
    pca_2d = PCA(n_components=2, random_state=42)
    coords_2d = pca_2d.fit_transform(sub_emb)

    points = []
    for i in range(sample_size):
        preview = sub_texts[i][:90] + ("..." if len(sub_texts[i]) > 90 else "")
        points.append({
            "id": int(indices[i]),
            "x": round(float(coords_2d[i, 0]), 4),
            "y": round(float(coords_2d[i, 1]), 4),
            "label": int(sub_labels[i]),
            "label_name": "Malicious / Phishing" if int(sub_labels[i]) == 1 else "Legitimate / Ham",
            "preview": preview
        })

    return {
        "points": points,
        "sample_size": sample_size,
        "total_points": n_total,
        "projection_label": "Visualization projection: PCA (Visualization-only projection)",
        "explained_variance_ratio": [round(float(v), 4) for v in pca_2d.explained_variance_ratio_]
    }
