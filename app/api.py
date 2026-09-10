"""
FastAPI REST API Backend for Quantum Text Security Research Platform & Representation Lab.
Provides endpoints for live multi-model prediction, quantum statevector simulation,
modular representation diagnostics, and audited research evidence.
"""

import os
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.schemas.prediction import PredictionRequest, StatevectorRequest
from app.model_engine import model_engine
from app.representations.registry import representation_registry
from app.research.registry import experiment_registry
from app.research.metrics import get_authoritative_metrics
from app.research.tables import list_tables, get_table_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "results", "exp39_paper", "figures")

app = FastAPI(
    title="Quantum Text Security Research Platform API",
    description="Interactive backend supporting live quantum vs classical text classification, Representation Lab, and research evidence exploration.",
    version="2.0.0"
)

# Enable CORS for local React/Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount figures directory if available
if os.path.exists(FIGURES_DIR):
    app.mount("/static/figures", StaticFiles(directory=FIGURES_DIR), name="figures")


@app.on_event("startup")
async def startup_event():
    """Warm up representations and reference models on startup."""
    print("Starting up Quantum Text Security API server v2.0...")
    try:
        model_engine.initialize()
    except Exception as e:
        print(f"Warning during model initialization: {e}")


@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    return {
        "status": "online",
        "engine": "FastAPI + PyTorch Statevector Engine (complex128)",
        "hardware": "Apple Silicon ARM64 / CPU",
        "models_loaded": model_engine.is_initialized,
        "active_protocol": "Final Protocol V1.0 (Frozen)",
        "representations_registered": len(representation_registry.list_all())
    }


@app.get("/api/samples")
def get_sample_presets() -> List[Dict[str, Any]]:
    """Returns library of test preset texts representing emails and SMS."""
    if not model_engine.is_initialized:
        model_engine.initialize()
    return model_engine.preset_samples


@app.post("/api/predict")
def predict_security_text(req: PredictionRequest) -> Dict[str, Any]:
    """Runs live comparative prediction across Quantum, Classical RBF, and Linear SVM."""
    input_text = req.text or ""
    if not input_text.strip() and not (req.subject or req.body):
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    try:
        return model_engine.predict_text(
            text=input_text,
            subject=req.subject,
            body=req.body,
            representation_id=req.representation or "tfidf",
            dimension=req.dimension or 8,
            dataset=req.dataset or "MeAJOR"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/quantum/statevector")
def compute_quantum_statevector(req: StatevectorRequest) -> Dict[str, Any]:
    """Computes exact statevector amplitudes and basis probabilities for a given representation."""
    if not req.text or len(req.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    pred = model_engine.predict_text(
        text=req.text,
        representation_id=req.representation or "tfidf",
        dimension=req.dimension or 8
    )
    return {
        "text": req.text,
        "quantum_diagnostics": pred["quantum_diagnostics"],
        "feature_trace": pred["feature_trace"],
        "models": pred["models"]
    }


# ============================================================
# REPRESENTATION LAB ENDPOINTS
# ============================================================

@app.get("/api/representations")
def list_representations() -> List[Dict[str, Any]]:
    """Returns metadata for all candidate and canonical representations."""
    return representation_registry.list_all()


@app.get("/api/representations/comparison-matrix")
def get_representation_comparison_matrix() -> List[Dict[str, Any]]:
    """Returns the Representation Comparison Matrix."""
    return experiment_registry.get_comparison_matrix()


@app.get("/api/representations/{rep_id}/geometry")
def get_representation_geometry(rep_id: str) -> Dict[str, Any]:
    """Returns geometry flow and diagnostics for the specified representation."""
    rep_key = rep_id.lower().strip()
    
    # Authoritative diagnostics from Table 7 & Exp 31
    geometry_data = {
        "tfidf": {
            "representation_name": "Canonical TF-IDF + TruncatedSVD (8D)",
            "dispersion_entropy_bits": 0.9421,
            "kernel_diversity": 0.4120,
            "target_label_alignment": 0.0402,
            "target_label_alignment_rbf": 0.0773,
            "alignment_deficit_pct": "-48.0%",
            "gram_pearson_r": 0.5985,
            "quantum_f1": 0.8754,
            "rbf_f1": 0.8709,
            "delta_f1": "+0.0046",
            "status": "CANONICAL",
            "evaluated": True,
            "interpretation": "Strong inverse association between single-state entropy and pairwise kernel diversity (r = -0.78 to -0.83)."
        },
        "roberta": {
            "representation_name": "Dense RoBERTa-base (8D)",
            "dispersion_entropy_bits": 0.8650,
            "kernel_diversity": 0.4890,
            "target_label_alignment": 0.0220,
            "target_label_alignment_rbf": 0.0603,
            "alignment_deficit_pct": "-63.5%",
            "gram_pearson_r": 0.5420,
            "quantum_f1": 0.9601,
            "rbf_f1": 0.9896,
            "delta_f1": "-0.0295",
            "status": "CANONICAL",
            "evaluated": True,
            "interpretation": "Dense contextual geometry causes Classical RBF to gain +2.95 pp over quantum fidelity kernel."
        },
        "minilm": {
            "representation_name": "Sentence-MiniLM-L6 (8D)",
            "dispersion_entropy_bits": 0.8920,
            "kernel_diversity": 0.4630,
            "target_label_alignment": 0.0285,
            "target_label_alignment_rbf": 0.0650,
            "alignment_deficit_pct": "-56.2%",
            "gram_pearson_r": 0.5610,
            "quantum_f1": 0.9645,
            "rbf_f1": 0.9830,
            "delta_f1": "-0.0185",
            "status": "EXPLORATORY",
            "evaluated": True,
            "interpretation": "Sentence-level semantic geometry shows classical RBF advantage on dense semantic embeddings."
        },
        "mpnet": {
            "representation_name": "Sentence-MPNet-Base (8D)",
            "dispersion_entropy_bits": 0.8810,
            "kernel_diversity": 0.4780,
            "target_label_alignment": 0.0298,
            "target_label_alignment_rbf": 0.0682,
            "alignment_deficit_pct": "-56.3%",
            "gram_pearson_r": 0.5530,
            "quantum_f1": 0.9682,
            "rbf_f1": 0.9874,
            "delta_f1": "-0.0192",
            "status": "EXPLORATORY",
            "evaluated": True,
            "interpretation": "Dense semantic geometry produces consistent classical RBF superiority across sentence transformers."
        },
        "fasttext": {
            "representation_name": "FastText Subword (8D)",
            "dispersion_entropy_bits": None,
            "kernel_diversity": None,
            "target_label_alignment": None,
            "target_label_alignment_rbf": None,
            "alignment_deficit_pct": None,
            "gram_pearson_r": None,
            "quantum_f1": None,
            "rbf_f1": None,
            "delta_f1": None,
            "status": "EXPLORATORY",
            "evaluated": False,
            "interpretation": "Geometry diagnostics not yet evaluated for this representation."
        }
    }

    if rep_key not in geometry_data:
        raise HTTPException(status_code=404, detail=f"Representation '{rep_id}' geometry not found.")
    return geometry_data[rep_key]


@app.get("/api/representations/{rep_id}/embedding-sample")
def get_embedding_sample(rep_id: str, sample_size: int = Query(default=120, ge=20, le=250)) -> Dict[str, Any]:
    """Returns 2D visualization scatterplot points with explicit projection labeling."""
    try:
        return model_engine.get_2d_embedding_sample(representation_id=rep_id, max_points=sample_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding sample error: {str(e)}")


# ============================================================
# RESEARCH EVIDENCE & BENCHMARK METRICS
# ============================================================

@app.get("/api/research/metrics")
def get_headline_metrics() -> Dict[str, Any]:
    """Returns authoritative Exp 40 10-seed headline metrics with rigorous statistical interpretation."""
    return get_authoritative_metrics()


@app.get("/api/research/tables")
def list_available_tables() -> List[Dict[str, Any]]:
    """Lists all audited research tables with status."""
    return list_tables()


@app.get("/api/research/tables/{table_id}")
def get_table_content(table_id: str) -> Dict[str, Any]:
    """Returns JSON parsed rows and columns for any paper table."""
    try:
        return get_table_data(table_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Table {table_id} not found.")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Table file for {table_id} not found on disk.")


@app.get("/api/geometry/gram")
def get_gram_matrix_data(
    representation: str = Query(default="tfidf"),
    size: int = Query(default=20, ge=5, le=40)
) -> Dict[str, Any]:
    """Returns interactive Gram matrix comparison data."""
    try:
        return model_engine.get_sample_gram_matrix(representation_id=representation, sample_size=size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gram matrix error: {str(e)}")


@app.get("/api/geometry/diagnostics")
def get_geometry_diagnostics() -> Dict[str, Any]:
    """Returns canonical kernel-target alignment and entropy vs diversity diagnostics."""
    return {
        "target_label_alignment": [
            {"corpus": "SMS Spam Collection (8D)", "quantum": 0.0382, "classical_rbf": 0.0768, "deficit_pct": "-50.3%", "status": "CANONICAL"},
            {"corpus": "CEAS 2008 Corpus (8D)", "quantum": 0.0220, "classical_rbf": 0.0603, "deficit_pct": "-63.5%", "status": "CANONICAL"},
            {"corpus": "MeAJOR Archive (8D)", "quantum": 0.0402, "classical_rbf": 0.0773, "deficit_pct": "-48.0%", "status": "CANONICAL"}
        ],
        "entropy_diversity_correlation": [
            {"corpus": "SMS Spam Collection", "correlation_r": -0.8257, "interpretation": "Higher single-state entropy is strongly inversely associated with pairwise kernel diversity", "status": "CANONICAL"},
            {"corpus": "CEAS 2008 Corpus", "correlation_r": -0.8170, "interpretation": "Higher single-state entropy is strongly inversely associated with pairwise kernel diversity", "status": "CANONICAL"},
            {"corpus": "MeAJOR Archive", "correlation_r": -0.7822, "interpretation": "Higher single-state entropy is strongly inversely associated with pairwise kernel diversity", "status": "CANONICAL"}
        ],
        "gram_correlation_by_dim": [
            {"dimension": "2D", "qubits": 2, "pearson_r": 0.5812},
            {"dimension": "4D", "qubits": 4, "pearson_r": 0.6124},
            {"dimension": "6D", "qubits": 6, "pearson_r": 0.6350},
            {"dimension": "8D", "qubits": 8, "pearson_r": 0.5985},
            {"dimension": "10D", "qubits": 10, "pearson_r": 0.5740},
            {"dimension": "12D", "qubits": 12, "pearson_r": 0.5510},
            {"dimension": "16D", "qubits": 16, "pearson_r": 0.4566}
        ]
    }


@app.get("/api/dimensionality/scaling")
def get_dimensionality_scaling_data(representation: str = Query(default="tfidf")) -> List[Dict[str, Any]]:
    """Returns dimensionality scaling trajectory for the requested representation."""
    rep_key = representation.lower().strip()
    if rep_key == "tfidf":
        return [
            {"dim": "2D", "qubits": 2, "quantum_f1": 0.6447, "rbf_f1": 0.6735, "linear_f1": 0.6853, "delta": -0.0288, "ci_lower": -0.0345, "ci_upper": -0.0231, "status": "CANONICAL"},
            {"dim": "4D", "qubits": 4, "quantum_f1": 0.7876, "rbf_f1": 0.7918, "linear_f1": 0.7584, "delta": -0.0042, "ci_lower": -0.0078, "ci_upper": -0.0006, "status": "CANONICAL"},
            {"dim": "6D", "qubits": 6, "quantum_f1": 0.8253, "rbf_f1": 0.8254, "linear_f1": 0.7946, "delta": -0.0001, "ci_lower": -0.0035, "ci_upper": 0.0033, "status": "CANONICAL"},
            {"dim": "8D", "qubits": 8, "quantum_f1": 0.8754, "rbf_f1": 0.8709, "linear_f1": 0.8445, "delta": 0.0046, "ci_lower": 0.0030, "ci_upper": 0.0061, "status": "CANONICAL"},
            {"dim": "10D", "qubits": 10, "quantum_f1": 0.9023, "rbf_f1": 0.8967, "linear_f1": 0.8730, "delta": 0.0057, "ci_lower": 0.0032, "ci_upper": 0.0081, "status": "CANONICAL"},
            {"dim": "12D", "qubits": 12, "quantum_f1": 0.9137, "rbf_f1": 0.9123, "linear_f1": 0.8892, "delta": 0.0014, "ci_lower": -0.0010, "ci_upper": 0.0037, "status": "CANONICAL"}
        ]
    elif rep_key in ["roberta", "minilm", "mpnet"]:
        # Show 8D point with explicit exploratory notice for other dimensions
        return [
            {"dim": "8D", "qubits": 8, "quantum_f1": 0.9601 if rep_key == "roberta" else (0.9645 if rep_key == "minilm" else 0.9682), "rbf_f1": 0.9896 if rep_key == "roberta" else (0.9830 if rep_key == "minilm" else 0.9874), "linear_f1": 0.9780 if rep_key == "roberta" else (0.9715 if rep_key == "minilm" else 0.9760), "delta": -0.0295 if rep_key == "roberta" else (-0.0185 if rep_key == "minilm" else -0.0192), "status": "CANONICAL" if rep_key == "roberta" else "EXPLORATORY"}
        ]
    else:
        return []


@app.get("/api/runtime/scaling")
def get_runtime_scaling_data() -> Dict[str, Any]:
    """Returns research benchmark computational time and memory scaling curves with timing scope."""
    return {
        "timing_scope": "Research benchmark runtime — 10,000 samples (Apple Silicon ARM64 / CPU)",
        "description": "Under the reported local statevector-simulation benchmark, the 12D quantum pipeline required approximately 64x the measured RBF runtime.",
        "data": [
            {"dim": "2D", "qubits": 2, "quantum_s": 3.4, "rbf_s": 1.3, "ratio": 2.6, "quantum_ram_mb": 142, "rbf_ram_mb": 95},
            {"dim": "4D", "qubits": 4, "quantum_s": 5.8, "rbf_s": 1.3, "ratio": 4.5, "quantum_ram_mb": 185, "rbf_ram_mb": 98},
            {"dim": "6D", "qubits": 6, "quantum_s": 9.4, "rbf_s": 1.4, "ratio": 6.7, "quantum_ram_mb": 310, "rbf_ram_mb": 102},
            {"dim": "8D", "qubits": 8, "quantum_s": 17.2, "rbf_s": 1.4, "ratio": 12.3, "quantum_ram_mb": 680, "rbf_ram_mb": 108},
            {"dim": "10D", "qubits": 10, "quantum_s": 44.6, "rbf_s": 1.5, "ratio": 29.7, "quantum_ram_mb": 2150, "rbf_ram_mb": 115},
            {"dim": "12D", "qubits": 12, "quantum_s": 108.8, "rbf_s": 1.7, "ratio": 64.0, "quantum_ram_mb": 6400, "rbf_ram_mb": 122},
            {"dim": "16D", "qubits": 16, "quantum_s": None, "rbf_s": 2.1, "ratio": None, "quantum_ram_mb": 10500, "rbf_ram_mb": 135}
        ]
    }


# ============================================================
# EXPERIMENT 41: REPRESENTATION SCREENING ENDPOINTS
# ============================================================

from app.research.exp41 import exp41_manager


@app.get("/api/research/representation/exp41")
def get_exp41_screening_data(
    dataset: Optional[str] = Query(default=None),
    representation: Optional[str] = Query(default=None)
) -> Dict[str, Any]:
    """Returns Exp 41 screening data and 3-seed summary."""
    return {
        "status": "READY" if exp41_manager.is_available else "NOT_EVALUATED",
        "canonical_status": "EXPLORATORY",
        "timing_scope": "Research benchmark runtime (3-seed screening)",
        "summary": exp41_manager.get_screening_summary(),
        "screening_results": exp41_manager.get_screening_results(dataset=dataset, representation=representation)
    }


@app.get("/api/research/representation/comparison")
def get_exp41_representation_comparison() -> List[Dict[str, Any]]:
    """Returns representation-conditioned Q-RBF performance comparisons and variation ranges."""
    return exp41_manager.get_representation_comparison()


@app.get("/api/research/representation/geometry")
def get_exp41_geometry() -> List[Dict[str, Any]]:
    """Returns geometry diagnostics (entropy, diversity, alignment) across representations."""
    return exp41_manager.get_geometry_diagnostics()


@app.get("/api/research/representation/status")
def get_exp41_status() -> Dict[str, Any]:
    """Returns Exp 41 lifecycle status and workload estimation."""
    return exp41_manager.get_status()


@app.get("/api/research/representation/space")
def get_exp41_representation_space(
    representation: str = Query(default="tfidf"),
    sample_size: int = Query(default=120, ge=20, le=250)
) -> Dict[str, Any]:
    """Returns 2D embedding space scatter coordinates for visualization."""
    return model_engine.get_2d_embedding_sample(representation_id=representation, max_points=sample_size)
