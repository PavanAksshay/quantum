"""
Audited Paper Tables Loader.
Reads frozen CSV evidence tables from results/exp39_paper/tables/.
"""

import os
from typing import Dict, List, Any
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLES_DIR = os.path.join(BASE_DIR, "results", "exp39_paper", "tables")

TABLE_METADATA = [
    {"id": "table_1", "title": "Table 1: Benchmark Dataset Characteristics", "file": "table_1_dataset_characteristics.csv", "status": "CANONICAL", "type": "Dataset Characteristics"},
    {"id": "table_2", "title": "Table 2: Comprehensive Classical Baselines", "file": "table_2_classical_baselines.csv", "status": "CANONICAL", "type": "Classical Baselines"},
    {"id": "table_3", "title": "Table 3: Canonical In-Distribution Comparison (8D)", "file": "table_3_canonical_comparison.csv", "status": "CANONICAL", "type": "In-Distribution Benchmark"},
    {"id": "table_4", "title": "Table 4: Dimensionality Scaling Trajectory (2D-12D)", "file": "table_4_dimensionality_scaling.csv", "status": "CANONICAL", "type": "Dimensionality Trajectory"},
    {"id": "table_5", "title": "Table 5: Cross-Source Domain Holdout (Direction B)", "file": "table_5_source_holdout.csv", "status": "CANONICAL", "type": "Distribution Shift"},
    {"id": "table_6", "title": "Table 6: Statistical Hypothesis Tests and CIs", "file": "table_6_statistical_tests.csv", "status": "CANONICAL", "type": "Statistical Synthesis"},
    {"id": "table_7", "title": "Table 7: Feature Space Geometry & Alignment Diagnostics", "file": "table_7_geometry_diagnostics.csv", "status": "CANONICAL", "type": "Geometry & Alignment"},
    {"id": "table_8", "title": "Table 8: Computational Runtime & Memory Scaling", "file": "table_8_runtime_scalability.csv", "status": "CANONICAL", "type": "Computational Profiling"}
]


def list_tables() -> List[Dict[str, Any]]:
    """Returns available tables with canonical/exploratory status."""
    return TABLE_METADATA


def get_table_data(table_id: str) -> Dict[str, Any]:
    """Parses table CSV and returns columns and row records."""
    match = next((t for t in TABLE_METADATA if t["id"] == table_id), None)
    if not match:
        raise KeyError(f"Table '{table_id}' not found.")

    file_path = os.path.join(TABLES_DIR, match["file"])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Table file '{match['file']}' not found on disk.")

    df = pd.read_csv(file_path).fillna("")
    return {
        "id": match["id"],
        "title": match["title"],
        "file_name": match["file"],
        "status": match["status"],
        "type": match["type"],
        "columns": df.columns.tolist(),
        "rows": df.to_dict(orient="records")
    }
