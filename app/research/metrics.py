"""
Immutable Authoritative Research Metrics & Statistical Interpretations.
Source of Truth: results/exp39_paper/ and results/exp40_final/.
"""

from typing import Dict, Any


def get_authoritative_metrics() -> Dict[str, Any]:
    """Returns canonical headline metrics with rigorous statistical interpretation."""
    return {
        "title": "When Do Quantum Kernels Help for Text Security?",
        "subtitle": "A controlled empirical study of representation, kernel geometry, dimensionality, generalization, and computational cost.",
        "primary_research_question": "Under controlled matched conditions, do parameter-free quantum fidelity kernels provide a consistent and practically meaningful advantage over matched classical RBF kernels?",
        "primary_conclusion": "No consistent practically meaningful quantum advantage was observed.",
        "practical_equivalence_threshold": 0.01,
        "statistical_protocol": {
            "seeds": 10,
            "bootstrap_replicates": 10000,
            "significance_level_alpha": 0.05,
            "multiple_testing_correction": "Benjamini-Hochberg FDR",
            "practical_equivalence_margin_epsilon": 0.01
        },
        "headline_results": {
            "iid_8d": {
                "dataset": "MeAJOR IID (8D / 8 Qubits)",
                "representation": "Canonical TF-IDF + TruncatedSVD",
                "quantum_f1": 0.8754,
                "quantum_std": 0.0029,
                "rbf_f1": 0.8709,
                "rbf_std": 0.0030,
                "delta_f1": 0.0046,
                "ci_95": [0.0030, 0.0061],
                "permutation_p": 0.0016,
                "status": "CANONICAL",
                "interpretation": "Statistically detectable but practically small (|Δ| < 0.01)"
            },
            "iid_10d": {
                "dataset": "MeAJOR IID (10D / 10 Qubits)",
                "representation": "Canonical TF-IDF + TruncatedSVD",
                "quantum_f1": 0.9023,
                "quantum_std": 0.0034,
                "rbf_f1": 0.8967,
                "rbf_std": 0.0049,
                "delta_f1": 0.0057,
                "ci_95": [0.0032, 0.0081],
                "permutation_p": 0.0052,
                "status": "CANONICAL",
                "interpretation": "Statistically detectable but practically small (|Δ| < 0.01)"
            },
            "iid_12d": {
                "dataset": "MeAJOR IID (12D / 12 Qubits)",
                "representation": "Canonical TF-IDF + TruncatedSVD",
                "quantum_f1": 0.9137,
                "quantum_std": 0.0046,
                "rbf_f1": 0.9123,
                "rbf_std": 0.0023,
                "delta_f1": 0.0014,
                "ci_95": [-0.0010, 0.0037],
                "permutation_p": 0.2824,
                "status": "CANONICAL",
                "interpretation": "No statistically detectable difference was observed (95% CI spans zero)"
            },
            "holdout_direction_b": {
                "dataset": "Cross-Source Holdout Direction B (TREC 2007 -> TREC 2005/2006)",
                "representation": "Canonical TF-IDF + TruncatedSVD",
                "evaluation_type": "Cross-source distribution shift across distinct dataset sources",
                "quantum_f1": 0.6680,
                "quantum_std": 0.0094,
                "rbf_f1": 0.6913,
                "rbf_std": 0.0161,
                "delta_f1": -0.0233,
                "ci_95": [-0.0353, -0.0117],
                "permutation_p": 0.0046,
                "bh_fdr_p": 0.0069,
                "status": "CANONICAL",
                "interpretation": "Statistically supported and practically meaningful classical RBF advantage (|Δ| > 0.01)"
            },
            "representation_dominance": {
                "dataset": "CEAS 2008 (8D)",
                "tfidf_quantum": 0.9736,
                "tfidf_rbf": 0.9641,
                "tfidf_delta": 0.0095,
                "roberta_quantum": 0.9601,
                "roberta_rbf": 0.9896,
                "roberta_delta": -0.0295,
                "net_shift": 0.0390,
                "status": "CANONICAL",
                "interpretation": "Representation choice alters model ranking by 3.90 percentage points, dominating kernel choice"
            },
            "simulation_overhead": {
                "benchmark_workload": "10,000 samples (Apple Silicon ARM64 CPU / complex128)",
                "dimension": "12D",
                "quantum_time_s": 108.8,
                "rbf_time_s": 1.7,
                "runtime_ratio": 64.0,
                "status": "CANONICAL",
                "interpretation": "Under the reported local statevector-simulation benchmark, the 12D quantum pipeline required approximately 64x the measured RBF runtime"
            }
        }
    }
