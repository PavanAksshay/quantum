"""
Integration tests for FastAPI REST Endpoints using unittest.
"""

import unittest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


class TestApiEndpoints(unittest.TestCase):

    def test_health_endpoint(self):
        res = client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("representations_registered", data)

    def test_representations_list_endpoint(self):
        res = client.get("/api/representations")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(len(data), 5)
        ids = [item["id"] for item in data]
        self.assertIn("tfidf", ids)
        self.assertIn("roberta", ids)
        self.assertIn("minilm", ids)
        self.assertIn("mpnet", ids)
        self.assertIn("fasttext", ids)

    def test_comparison_matrix_endpoint(self):
        res = client.get("/api/representations/comparison-matrix")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(len(data), 5)
        tfidf_row = next(r for r in data if r["representation_id"] == "tfidf")
        self.assertEqual(tfidf_row["quantum_f1"], 0.9736)
        self.assertEqual(tfidf_row["rbf_f1"], 0.9641)

    def test_predict_endpoint_single_text(self):
        payload = {
            "text": "Subject: Urgent security alert: Your account has been locked. Verify immediately.",
            "representation": "tfidf",
            "dimension": 8
        }
        res = client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("models", data)
        self.assertIn("quantum_fidelity_kernel", data["models"])
        self.assertIn("classical_rbf", data["models"])
        self.assertIn("linear_svm", data["models"])
        self.assertIn("feature_trace", data)
        self.assertEqual(data["feature_trace"]["projected_dimension"], 8)

    def test_predict_endpoint_subject_and_body(self):
        payload = {
            "subject": "CRITICAL: Urgent action required on payroll",
            "body": "Please click the following link to confirm your bank account credentials.",
            "representation": "tfidf",
            "dimension": 8
        }
        res = client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("Subject: CRITICAL", data["input_text"])

    def test_research_metrics_endpoint(self):
        res = client.get("/api/research/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("headline_results", data)
        self.assertEqual(data["practical_equivalence_threshold"], 0.01)

    def test_tables_endpoint(self):
        res = client.get("/api/research/tables")
        self.assertEqual(res.status_code, 200)
        tables = res.json()
        self.assertEqual(len(tables), 8)
        self.assertTrue(all(t["status"] == "CANONICAL" for t in tables))

        # Fetch Table 3
        res_t3 = client.get("/api/research/tables/table_3")
        self.assertEqual(res_t3.status_code, 200)
        t3_data = res_t3.json()
        self.assertGreater(len(t3_data["rows"]), 0)


if __name__ == "__main__":
    unittest.main()
