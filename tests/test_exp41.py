"""
Tests for Experiment 41: Representation Screening Subsystem & Endpoints.
Verifies schema, status, comparison metrics, geometry, and canonical-vs-exploratory partitioning.
"""

import os
import unittest
from fastapi.testclient import TestClient
from app.api import app
from app.research.exp41 import exp41_manager

client = TestClient(app)


class TestExp41Screening(unittest.TestCase):

    def test_exp41_status_endpoint(self):
        res = client.get("/api/research/representation/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["experiment_id"], "exp41")
        self.assertEqual(data["canonical_status"], "EXPLORATORY")
        self.assertEqual(data["total_runs"], 108)
        self.assertEqual(data["seeds"], [42, 123, 456])
        self.assertEqual(data["dimension"], 8)
        self.assertIn("tfidf", data["representations"])
        self.assertIn("minilm", data["representations"])
        self.assertIn("roberta", data["representations"])
        self.assertIn("mpnet", data["representations"])

    def test_exp41_screening_endpoint_structure(self):
        res = client.get("/api/research/representation/exp41")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("status", data)
        self.assertEqual(data["canonical_status"], "EXPLORATORY")
        self.assertIn("summary", data)
        self.assertIn("screening_results", data)

    def test_exp41_comparison_endpoint_structure(self):
        res = client.get("/api/research/representation/comparison")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data, list)

    def test_exp41_geometry_endpoint_structure(self):
        res = client.get("/api/research/representation/geometry")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data, list)

    def test_exp41_space_endpoint(self):
        res = client.get("/api/research/representation/space?representation=tfidf&sample_size=30")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("points", data)
        self.assertIn("projection_label", data)
        self.assertLessEqual(len(data["points"]), 30)

    def test_canonical_claims_remain_unmodified(self):
        """Verifies that canonical Exp 40 headline metrics remain strictly intact."""
        res = client.get("/api/research/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["headline_results"]["iid_8d"]["delta_f1"], 0.0046)
        self.assertEqual(data["headline_results"]["iid_12d"]["delta_f1"], 0.0014)
        self.assertEqual(data["headline_results"]["holdout_direction_b"]["delta_f1"], -0.0233)


if __name__ == "__main__":
    unittest.main()
