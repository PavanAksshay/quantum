"""
Multi-Model Live Inference Coordinator & Feature Trace Engine.
Executes live predictions across Quantum, Gaussian RBF, and Linear SVM models.
"""

import time
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVC, LinearSVC

from app.representations.registry import representation_registry
from app.model_engine.quantum import simulate_zz_feature_map, compute_von_neumann_entropy, extract_top_basis_states
from app.model_engine.classical import build_classical_models, compute_rbf_kernel_matrix


class InferenceEngine:
    """
    Coordinates dataset loading, model training, and real-time comparative inference.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.is_initialized = False
        
        # Training state references
        self.train_texts: List[str] = []
        self.train_y: np.ndarray = np.array([])
        
        # Model storage keyed by (representation, dimension)
        self.quantum_clfs: Dict[Tuple[str, int], SVC] = {}
        self.rbf_clfs: Dict[Tuple[str, int], SVC] = {}
        self.linear_clfs: Dict[Tuple[str, int], LinearSVC] = {}
        self.train_states: Dict[Tuple[str, int], torch.Tensor] = {}
        
        self.preset_samples: List[Dict[str, Any]] = []

    def initialize(self):
        """Fits representations and pre-trains inference models on reference training set."""
        if self.is_initialized:
            return

        print("Initializing InferenceEngine on reference dataset...")
        t0 = time.perf_counter()

        # Load representative training partition from MeAJOR / CEAS
        meajor_path = os.path.join(self.data_dir, "meajor_cleaned_preprocessed.parquet.gzip")
        sms_path = os.path.join(self.data_dir, "SMSSpamCollection")
        
        texts = []
        labels = []

        if os.path.exists(meajor_path):
            df = pd.read_parquet(meajor_path)
            # Sample balanced subset for real-time inference support vectors
            df_pos = df[df["label"] == 1].head(350)
            df_neg = df[df["label"] == 0].head(350)
            df_sub = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=42)
            if "text" in df_sub.columns:
                texts = df_sub["text"].astype(str).tolist()
            else:
                texts = ("Subject: " + df_sub["subject"].fillna("").astype(str) + "\n\n" + df_sub["body"].fillna("").astype(str)).tolist()
            labels = df_sub["label"].astype(int).to_numpy()
        elif os.path.exists(sms_path):
            df = pd.read_csv(sms_path, sep='\t', names=['label_str', 'text'])
            df['label'] = (df['label_str'] == 'spam').astype(int)
            texts = df['text'].astype(str).tolist()[:500]
            labels = df['label'].to_numpy()[:500]
        else:
            # Fallback synthetic reference corpus for standalone cloud deployments
            texts = [
                "URGENT: Your bank account has been locked. Verify identity at http://secure-login.com",
                "Congratulations! You won the international lottery prize of $1,000,000. Claim now.",
                "Security Alert: Suspicious login attempt from unknown device. Reset password here.",
                "Dear customer, your invoice is overdue. Please download attachment to review charges.",
                "Final Notice: Your email service will be terminated within 24 hours without confirmation.",
                "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121",
                "Hi Pavan, are we still meeting tomorrow for the weekly research engineering sync?",
                "The project presentation has been rescheduled to Thursday at 3 PM in conference room A.",
                "Please find attached the updated research manuscript draft for review.",
                "Let me know when you have time to review the experimental benchmark figures."
            ] * 20
            labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0] * 20)

        self.train_texts = texts
        self.train_y = labels

        # Initialize representation registry
        representation_registry.fit_all(texts)

        # Train primary models (Canonical TF-IDF at standard 8D and other dimensions)
        self._train_models_for_representation("tfidf", 8)
        
        # Build preset library
        self._build_preset_samples()

        self.is_initialized = True
        print(f"InferenceEngine initialized successfully in {(time.perf_counter() - t0):.2f}s.")

    def _train_models_for_representation(self, rep_id: str, dim: int):
        """Fits Quantum, RBF, and Linear SVM for a given representation and dimension."""
        key = (rep_id, dim)
        if key in self.rbf_clfs:
            return

        rep = representation_registry.get(rep_id)
        if not rep.is_fitted:
            rep.fit(self.train_texts)

        train_texts_sub = self.train_texts[:200] if len(self.train_texts) > 200 else self.train_texts
        train_y_sub = self.train_y[:200] if len(self.train_y) > 200 else self.train_y

        X_scaled = rep.transform(train_texts_sub, target_dim=dim)
        X_phase = rep.get_phase_coordinates(train_texts_sub, target_dim=dim)

        # 1. Linear SVM baseline
        clf_linear = LinearSVC(C=1.0, random_state=42, max_iter=2000, dual="auto")
        clf_linear.fit(X_scaled, train_y_sub)
        self.linear_clfs[key] = clf_linear

        # 2. Matched Classical RBF SVM
        clf_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
        clf_rbf.fit(X_scaled, train_y_sub)
        self.rbf_clfs[key] = clf_rbf

        # 3. Quantum Fidelity Kernel SVM
        tensor_X = torch.tensor(X_phase, dtype=torch.float64)
        train_states = simulate_zz_feature_map(tensor_X, n_qubits=dim)
        inner = torch.matmul(train_states, train_states.conj().T)
        K_train = (torch.abs(inner) ** 2).cpu().numpy()
        np.fill_diagonal(K_train, 1.0)

        clf_quantum = SVC(kernel="precomputed", C=1.0, random_state=42)
        clf_quantum.fit(K_train, train_y_sub)

        self.quantum_clfs[key] = clf_quantum
        self.train_states[key] = train_states

    def predict(
        self,
        text: str,
        subject: Optional[str] = None,
        body: Optional[str] = None,
        representation_id: str = "tfidf",
        dimension: int = 8,
        dataset: str = "MeAJOR"
    ) -> Dict[str, Any]:
        """
        Executes live comparative classification across models.
        Supports single text or subject/body format.
        """
        if not self.is_initialized:
            self.initialize()

        t_start = time.perf_counter()

        # Combine text if subject and body are provided
        if subject is not None and body is not None:
            full_text = f"Subject: {subject.strip()}\n\n{body.strip()}"
        else:
            full_text = text.strip()

        rep_id = representation_id.lower().strip()
        rep = representation_registry.get(rep_id)
        
        # Ensure models are trained for requested (rep, dim)
        key = (rep_id, dimension)
        if key not in self.rbf_clfs:
            try:
                self._train_models_for_representation(rep_id, dimension)
            except Exception as e:
                # If quantum inference is unavailable for this combination:
                raise RuntimeError(f"Inference unavailable for representation '{rep_id}' at {dimension}D: {str(e)}")

        # 1. Feature Representation & Projection
        t_rep0 = time.perf_counter()
        X_scaled = rep.transform([full_text], target_dim=dimension)
        X_phase = rep.get_phase_coordinates([full_text], target_dim=dimension)
        rep_latency = (time.perf_counter() - t_rep0) * 1000.0

        # 2. Linear SVM Baseline
        t_lin0 = time.perf_counter()
        clf_linear = self.linear_clfs[key]
        score_lin = float(clf_linear.decision_function(X_scaled)[0])
        pred_lin = int(score_lin > 0.0)
        prob_lin = float(1.0 / (1.0 + np.exp(-score_lin)))
        time_lin = (time.perf_counter() - t_lin0) * 1000.0

        # 3. Classical Gaussian RBF SVM
        t_rbf0 = time.perf_counter()
        clf_rbf = self.rbf_clfs[key]
        score_rbf = float(clf_rbf.decision_function(X_scaled)[0])
        pred_rbf = int(score_rbf > 0.0)
        prob_rbf = float(1.0 / (1.0 + np.exp(-score_rbf)))
        time_rbf = (time.perf_counter() - t_rbf0) * 1000.0

        # 4. Quantum Fidelity Kernel SVM
        t_q0 = time.perf_counter()
        clf_quantum = self.quantum_clfs[key]
        train_states = self.train_states[key]

        tensor_test = torch.tensor(X_phase, dtype=torch.float64)
        test_state = simulate_zz_feature_map(tensor_test, n_qubits=dimension)
        inner = torch.matmul(test_state, train_states.conj().T)
        K_test = (torch.abs(inner) ** 2).cpu().numpy()
        K_test = np.clip(K_test, 0.0, 1.0)

        score_quantum = float(clf_quantum.decision_function(K_test)[0])
        pred_quantum = int(score_quantum > 0.0)
        prob_quantum = float(1.0 / (1.0 + np.exp(-score_quantum)))
        time_quantum = (time.perf_counter() - t_q0) * 1000.0

        # Quantum diagnostics (Amplitudes and von Neumann Entropy)
        entropy = compute_von_neumann_entropy(test_state[0], n_qubits=dimension)
        top_basis = extract_top_basis_states(test_state[0], n_qubits=dimension, top_k=8)

        total_time = (time.perf_counter() - t_start) * 1000.0

        # Feature Trace metadata for the Text-Level Inspector
        feature_trace = {
            "character_count": len(full_text),
            "word_count": len(full_text.split()),
            "representation_name": rep.name,
            "original_dimension": rep.original_dimension,
            "projected_dimension": dimension,
            "projected_coordinates": [round(float(x), 4) for x in X_scaled[0]],
            "phase_coordinates_rad": [round(float(x), 4) for x in X_phase[0]],
            "phase_mapping_formula": "Standardized vector -> MinMax normalization -> [0, pi] phase-angle encoding"
        }

        return {
            "input_text": full_text,
            "dataset": dataset,
            "representation": rep.get_metadata(),
            "dimension": dimension,
            "feature_trace": feature_trace,
            "timing_scope": "Live request latency (single sample evaluation)",
            "models": {
                "quantum_fidelity_kernel": {
                    "name": f"Quantum Fidelity Kernel ({dimension} Qubits, 2-Layer ZZ)",
                    "prediction": pred_quantum,
                    "label": "Malicious / Phishing" if pred_quantum == 1 else "Legitimate / Ham",
                    "decision_score": round(score_quantum, 4),
                    "estimated_probability": round(prob_quantum, 4),
                    "latency_ms": round(time_quantum, 2),
                    "is_malicious": bool(pred_quantum == 1),
                    "score_description": "Signed SVM decision margin (uncalibrated score; probability estimated via sigmoid mapping)"
                },
                "classical_rbf": {
                    "name": f"Matched Classical Gaussian RBF ({dimension}D)",
                    "prediction": pred_rbf,
                    "label": "Malicious / Phishing" if pred_rbf == 1 else "Legitimate / Ham",
                    "decision_score": round(score_rbf, 4),
                    "estimated_probability": round(prob_rbf, 4),
                    "latency_ms": round(time_rbf, 2),
                    "is_malicious": bool(pred_rbf == 1),
                    "score_description": "Signed SVM decision margin (uncalibrated score; probability estimated via sigmoid mapping)"
                },
                "linear_svm": {
                    "name": f"Linear SVM baseline ({dimension}D)",
                    "prediction": pred_lin,
                    "label": "Malicious / Phishing" if pred_lin == 1 else "Legitimate / Ham",
                    "decision_score": round(score_lin, 4),
                    "estimated_probability": round(prob_lin, 4),
                    "latency_ms": round(time_lin, 2),
                    "is_malicious": bool(pred_lin == 1),
                    "score_description": "Signed LinearSVC hyperplane distance"
                }
            },
            "quantum_diagnostics": {
                "n_qubits": dimension,
                "hilbert_dimension": 2 ** dimension,
                "state_entropy_bits": round(entropy, 4),
                "top_basis_probabilities": top_basis
            },
            "total_latency_ms": round(total_time, 2)
        }

    def _build_preset_samples(self):
        self.preset_samples = [
            {
                "id": "sample_phish_1",
                "title": "Urgent Account Suspension Alert",
                "category": "Phishing / Scam",
                "subject": "CRITICAL: Immediate Account Security Verification Required",
                "body": "Dear Valued Customer,\n\nWe detected unauthorized login attempts to your corporate portal from an unrecognized IP address. Your access will be suspended within 24 hours unless you re-verify your identity.\n\nPlease follow the secure verification link: http://auth-portal-secure-update.com/verify?token=938210\n\nIT Security Department",
                "text": "Subject: CRITICAL: Immediate Account Security Verification Required\n\nDear Valued Customer, We detected unauthorized login attempts to your corporate portal. Your access will be suspended within 24 hours unless you re-verify your identity at http://auth-portal-secure-update.com/verify"
            },
            {
                "id": "sample_phish_2",
                "title": "International Lottery Prize Award",
                "category": "Phishing / Scam",
                "subject": "NOTIFICATION OF AWARD WINNINGS: Ref #8491/2026",
                "body": "CONGRATULATIONS!\n\nYour email address was selected in the Global Postal Promo Draw. You have been awarded the sum of $2,500,000 USD.\n\nTo claim your cash disbursement, reply immediately with your full name, banking routing number, and mobile phone number to claims-dept@intl-postal-lottery.org.",
                "text": "Subject: NOTIFICATION OF AWARD WINNINGS\n\nCONGRATULATIONS! You have been awarded $2,500,000 USD. To claim your cash disbursement, reply with your full name, banking routing number, and mobile phone number."
            },
            {
                "id": "sample_ham_1",
                "title": "GitHub Commit & CI Notification",
                "category": "Legitimate / Ham",
                "subject": "[GitHub] Build Passed: Fix tensor contraction indexing in quantum pipeline",
                "body": "Hi team,\n\nPull Request #42 has passed all unit tests and lint checks in CI pipeline (Workflow: Research Testing / Run 108).\n\nCommit: a8f910b (Fix tensor contraction indexing)\nAuthor: quantum-researcher <researcher@lab.org>\nBranch: main\n\nYou can review the complete test report in the repository dashboard.",
                "text": "Subject: [GitHub] Build Passed: Fix tensor contraction indexing\n\nPull Request #42 has passed all unit tests and lint checks in CI pipeline. Commit a8f910b on main branch."
            },
            {
                "id": "sample_ham_2",
                "title": "Weekly Research Engineering Sync",
                "category": "Legitimate / Ham",
                "subject": "Agenda: Quantum Kernel Geometry & Scalability Sync (Friday 2pm)",
                "body": "Hi all,\n\nHere is the proposed agenda for our weekly research sync this Friday at 2:00 PM EST:\n\n1. Review Exp 40 10-seed paired confirmation statistics\n2. Discuss dimensionality scaling trajectory (2D to 12D)\n3. Profiling simulation runtime vs classical RBF baselines\n\nMeeting link: https://meet.company.internal/quantum-sync-weekly\n\nBest regards,\nPavan",
                "text": "Subject: Agenda: Quantum Kernel Geometry Sync (Friday 2pm)\n\nHere is the proposed agenda for our weekly research sync: Review Exp 40 10-seed paired confirmation statistics, dimensionality scaling trajectory, and runtime profiling."
            }
        ]


# Base directory definition
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
inference_engine = InferenceEngine(DATA_DIR)
