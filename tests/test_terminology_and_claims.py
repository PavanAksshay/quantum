"""
Scientific Terminology & Conservative Claims Assertion Tests using unittest.
"""

import unittest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


class TestTerminologyAndClaims(unittest.TestCase):

    def test_no_prohibited_terminology_in_metrics(self):
        res = client.get("/api/research/metrics")
        self.assertEqual(res.status_code, 200)
        text = res.text.lower()
        
        # Prohibited claims
        self.assertNotIn("adversarial robustness", text)
        self.assertNotIn("contextual linear svm", text)
        self.assertNotIn("full vocabulary ceiling", text)

    def test_runtime_scope_labeling(self):
        res = client.get("/api/runtime/scaling")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("timing_scope", data)
        self.assertIn("10,000 samples", data["timing_scope"])

    def test_prediction_decision_score_terminology(self):
        payload = {
            "text": "Subject: Test security email\n\nPlease find the attached report.",
            "representation": "tfidf",
            "dimension": 8
        }
        res = client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        
        models = data["models"]
        for model_key in ["quantum_fidelity_kernel", "classical_rbf", "linear_svm"]:
            m = models[model_key]
            self.assertIn("decision_score", m)
            self.assertIn("score_description", m)


if __name__ == "__main__":
    unittest.main()
