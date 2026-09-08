"""
FastAPI REST API Backend for Quantum Text Security Research Platform & Live Prediction.
Provides endpoints for live prediction, quantum statevector simulation,
research evidence data, geometry diagnostics, and statistical equivalence.
"""

import os
import time
import glob
import pandas as pd
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.model_engine import model_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP39_DIR = os.path.join(BASE_DIR, "results", "exp39_paper")
EXP40_DIR = os.path.join(BASE_DIR, "results", "exp40_final")
TABLES_DIR = os.path.join(EXP39_DIR, "tables")
FIGURES_DIR = os.path.join(EXP39_DIR, "figures")

app = FastAPI(
    title="Quantum Text Security Research Platform API",
    description="Interactive backend supporting live quantum vs classical text classification and research evidence exploration.",
    version="1.0.0"
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


class PredictionRequest(BaseModel):
    text: str

class StatevectorRequest(BaseModel):
    text: str


@app.on_event("startup")
async def startup_event():
    # Warm up models on startup
    print("Starting up Quantum Text Security API server...")
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
        "active_protocol": "Final Protocol V1.0 (Frozen)"
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
    if not req.text or len(req.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    try:
        return model_engine.predict_text(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/quantum/statevector")
def compute_quantum_statevector(req: StatevectorRequest) -> Dict[str, Any]:
    """Computes exact 8-qubit statevector amplitudes and basis probabilities."""
    if not req.text or len(req.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    pred = model_engine.predict_text(req.text)
    return {
        "text": req.text,
        "quantum_diagnostics": pred["quantum_diagnostics"],
        "reduced_features_8d": pred["reduced_features_8d"],
        "quantum_phase_coords": pred["quantum_phase_coords"]
    }


@app.get("/api/research/metrics")
def get_headline_metrics() -> Dict[str, Any]:
    """Returns authoritative Exp 40 10-seed headline metrics."""
    return {
        "title": "When Do Quantum Kernels Help for Text Security?",
        "primary_rq": "Under controlled matched conditions, do quantum kernel methods provide a consistent and practically meaningful advantage over matched classical RBF kernels?",
        "answer": "No, under the evaluated conditions.",
        "practical_equivalence_threshold": 0.01,
        "headline_results": {
            "iid_8d": {
                "dataset": "MeAJOR IID 8D (8 Qubits)",
                "quantum_f1": 0.8754,
                "quantum_std": 0.0029,
                "rbf_f1": 0.8709,
                "rbf_std": 0.0030,
                "delta_f1": 0.0046,
                "ci_95": [0.0030, 0.0061],
                "permutation_p": 0.0016,
                "status": "Statistically detectable, practically equivalent (|Δ| < 0.01)"
            },
            "iid_10d": {
                "dataset": "MeAJOR IID 10D (10 Qubits)",
                "quantum_f1": 0.9023,
                "quantum_std": 0.0034,
                "rbf_f1": 0.8967,
                "rbf_std": 0.0049,
                "delta_f1": 0.0057,
                "ci_95": [0.0032, 0.0081],
                "permutation_p": 0.0052,
                "status": "Statistically detectable, practically equivalent (|Δ| < 0.01)"
            },
            "iid_12d": {
                "dataset": "MeAJOR IID 12D (12 Qubits)",
                "quantum_f1": 0.9137,
                "quantum_std": 0.0046,
                "rbf_f1": 0.9123,
                "rbf_std": 0.0023,
                "delta_f1": 0.0014,
                "ci_95": [-0.0010, 0.0037],
                "permutation_p": 0.2824,
                "status": "Statistical and practical parity (CI spans zero)"
            },
            "holdout_direction_b": {
                "dataset": "MeAJOR Direction B (TREC 2007 -> TREC 2005/2006)",
                "quantum_f1": 0.6680,
                "quantum_std": 0.0094,
                "rbf_f1": 0.6913,
                "rbf_std": 0.0161,
                "delta_f1": -0.0233,
                "ci_95": [-0.0353, -0.0117],
                "permutation_p": 0.0046,
                "bh_fdr_p": 0.0069,
                "status": "Classical RBF advantage exceeds practical threshold (|Δ| > 0.01)"
            },
            "representation_inversion": {
                "dataset": "CEAS 2008 (8D)",
                "tfidf_quantum": 0.9736,
                "tfidf_rbf": 0.9641,
                "tfidf_delta": 0.0095,
                "roberta_quantum": 0.9601,
                "roberta_rbf": 0.9896,
                "roberta_delta": -0.0295,
                "net_shift": 0.0390,
                "status": "Representation dominates kernel choice by 3.90 percentage points"
            },
            "simulation_runtime_12d": {
                "quantum_time_s": 108.8,
                "rbf_time_s": 1.7,
                "runtime_ratio": 64.0,
                "status": "64x simulation time penalty for identical predictive accuracy"
            }
        }
    }


@app.get("/api/research/tables")
def list_available_tables() -> List[Dict[str, str]]:
    """Lists all frozen research tables available."""
    tables = [
        {"id": "table_1", "title": "Table 1: Benchmark Dataset Characteristics", "file": "table_1_dataset_characteristics.csv"},
        {"id": "table_2", "title": "Table 2: Comprehensive Classical Baselines", "file": "table_2_classical_baselines.csv"},
        {"id": "table_3", "title": "Table 3: Canonical In-Distribution Comparison (8D)", "file": "table_3_canonical_comparison.csv"},
        {"id": "table_4", "title": "Table 4: Dimensionality Scaling Trajectory (2D-12D)", "file": "table_4_dimensionality_scaling.csv"},
        {"id": "table_5", "title": "Table 5: Cross-Source Domain Holdout (Direction B)", "file": "table_5_source_holdout.csv"},
        {"id": "table_6", "title": "Table 6: Statistical Hypothesis Tests and CIs", "file": "table_6_statistical_tests.csv"},
        {"id": "table_7", "title": "Table 7: Feature Space Geometry & Alignment Diagnostics", "file": "table_7_geometry_diagnostics.csv"},
        {"id": "table_8", "title": "Table 8: Computational Runtime & Memory Scaling", "file": "table_8_runtime_scalability.csv"},
    ]
    return tables


@app.get("/api/research/tables/{table_id}")
def get_table_content(table_id: str) -> Dict[str, Any]:
    """Returns JSON parsed rows and columns for any paper table."""
    table_map = {
        "table_1": "table_1_dataset_characteristics.csv",
        "table_2": "table_2_classical_baselines.csv",
        "table_3": "table_3_canonical_comparison.csv",
        "table_4": "table_4_dimensionality_scaling.csv",
        "table_5": "table_5_source_holdout.csv",
        "table_6": "table_6_statistical_tests.csv",
        "table_7": "table_7_geometry_diagnostics.csv",
        "table_8": "table_8_runtime_scalability.csv",
    }
    if table_id not in table_map:
        raise HTTPException(status_code=404, detail=f"Table {table_id} not found.")

    file_name = table_map[table_id]
    csv_path = os.path.join(TABLES_DIR, file_name)
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"Table file {file_name} not found.")

    df = pd.read_csv(csv_path)
    # Fill NaN
    df = df.fillna("")
    return {
        "table_id": table_id,
        "file_name": file_name,
        "columns": df.columns.tolist(),
        "rows": df.to_dict(orient="records")
    }


@app.get("/api/geometry/gram")
def get_gram_matrix_data(size: int = Query(default=20, ge=5, le=40)) -> Dict[str, Any]:
    """Returns interactive Gram matrix comparison data."""
    return model_engine.get_sample_gram_matrix(sample_size=size)


@app.get("/api/geometry/diagnostics")
def get_geometry_diagnostics() -> Dict[str, Any]:
    """Returns kernel-target alignment and entropy vs diversity diagnostics."""
    return {
        "target_label_alignment": [
            {"corpus": "SMS Spam Collection (8D)", "quantum": 0.0382, "classical_rbf": 0.0768, "deficit_pct": "-50.3%"},
            {"corpus": "CEAS 2008 Corpus (8D)", "quantum": 0.0220, "classical_rbf": 0.0603, "deficit_pct": "-63.5%"},
            {"corpus": "MeAJOR Archive (8D)", "quantum": 0.0402, "classical_rbf": 0.0773, "deficit_pct": "-48.0%"}
        ],
        "entropy_diversity_correlation": [
            {"corpus": "SMS Spam Collection", "correlation_r": -0.8257, "interpretation": "Strong inverse association"},
            {"corpus": "CEAS 2008 Corpus", "correlation_r": -0.8170, "interpretation": "Strong inverse association"},
            {"corpus": "MeAJOR Archive", "correlation_r": -0.7822, "interpretation": "Strong inverse association"}
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
def get_dimensionality_scaling_data() -> List[Dict[str, Any]]:
    """Returns MeAJOR in-distribution dimensionality scaling trajectory."""
    return [
        {"dim": "2D", "qubits": 2, "quantum_f1": 0.6447, "rbf_f1": 0.6735, "linear_f1": 0.6853, "delta": -0.0288, "ci_lower": -0.0345, "ci_upper": -0.0231},
        {"dim": "4D", "qubits": 4, "quantum_f1": 0.7876, "rbf_f1": 0.7918, "linear_f1": 0.7584, "delta": -0.0042, "ci_lower": -0.0078, "ci_upper": -0.0006},
        {"dim": "6D", "qubits": 6, "quantum_f1": 0.8253, "rbf_f1": 0.8254, "linear_f1": 0.7946, "delta": -0.0001, "ci_lower": -0.0035, "ci_upper": 0.0033},
        {"dim": "8D", "qubits": 8, "quantum_f1": 0.8754, "rbf_f1": 0.8709, "linear_f1": 0.8445, "delta": 0.0046, "ci_lower": 0.0030, "ci_upper": 0.0061},
        {"dim": "10D", "qubits": 10, "quantum_f1": 0.9023, "rbf_f1": 0.8967, "linear_f1": 0.8730, "delta": 0.0057, "ci_lower": 0.0032, "ci_upper": 0.0081},
        {"dim": "12D", "qubits": 12, "quantum_f1": 0.9137, "rbf_f1": 0.9123, "linear_f1": 0.8892, "delta": 0.0014, "ci_lower": -0.0010, "ci_upper": 0.0037}
    ]


@app.get("/api/runtime/scaling")
def get_runtime_scaling_data() -> List[Dict[str, Any]]:
    """Returns computational time and memory scaling curves."""
    return [
        {"dim": "2D", "qubits": 2, "quantum_s": 3.4, "rbf_s": 1.3, "ratio": 2.6, "quantum_ram_mb": 142, "rbf_ram_mb": 95},
        {"dim": "4D", "qubits": 4, "quantum_s": 5.8, "rbf_s": 1.3, "ratio": 4.5, "quantum_ram_mb": 185, "rbf_ram_mb": 98},
        {"dim": "6D", "qubits": 6, "quantum_s": 9.4, "rbf_s": 1.4, "ratio": 6.7, "quantum_ram_mb": 310, "rbf_ram_mb": 102},
        {"dim": "8D", "qubits": 8, "quantum_s": 17.2, "rbf_s": 1.4, "ratio": 12.3, "quantum_ram_mb": 680, "rbf_ram_mb": 108},
        {"dim": "10D", "qubits": 10, "quantum_s": 44.6, "rbf_s": 1.5, "ratio": 29.7, "quantum_ram_mb": 2150, "rbf_ram_mb": 115},
        {"dim": "12D", "qubits": 12, "quantum_s": 108.8, "rbf_s": 1.7, "ratio": 64.0, "quantum_ram_mb": 6400, "rbf_ram_mb": 122},
        {"dim": "16D", "qubits": 16, "quantum_s": None, "rbf_s": 2.1, "ratio": None, "quantum_ram_mb": 10500, "rbf_ram_mb": 135}
    ]
