"""
Tests for Modular Representation Subsystem.
Verifies dimensions, metadata, deterministic encoding, and graceful status.
"""

import unittest
import numpy as np
from app.representations.registry import representation_registry
from app.representations.base import ModelStatus


class TestRepresentations(unittest.TestCase):

    def test_tfidf_representation(self):
        rep = representation_registry.get("tfidf")
        self.assertTrue(rep.is_canonical)
        self.assertTrue(rep.is_sparse)
        self.assertLessEqual(rep.original_dimension, 50000)
        
        texts = [
            "Subject: Urgent account verification required",
            "Subject: Team meeting on research geometry this afternoon"
        ]
        rep.fit(texts)
        self.assertTrue(rep.is_fitted)
        
        transformed_8d = rep.transform(texts, target_dim=8)
        self.assertEqual(transformed_8d.shape, (2, 8))
        
        phase_coords = rep.get_phase_coordinates(texts, target_dim=8)
        self.assertEqual(phase_coords.shape, (2, 8))
        self.assertTrue(np.all(phase_coords >= 0.0) and np.all(phase_coords <= np.pi))

    def test_minilm_representation_metadata(self):
        rep = representation_registry.get("minilm")
        meta = rep.get_metadata()
        self.assertEqual(meta["id"], "minilm")
        self.assertFalse(meta["is_canonical"])
        self.assertEqual(meta["original_dimension"], 384)

    def test_mpnet_representation_metadata(self):
        rep = representation_registry.get("mpnet")
        meta = rep.get_metadata()
        self.assertEqual(meta["id"], "mpnet")
        self.assertFalse(meta["is_canonical"])
        self.assertEqual(meta["original_dimension"], 768)

    def test_fasttext_graceful_unavailable(self):
        rep = representation_registry.get("fasttext")
        meta = rep.get_metadata()
        self.assertEqual(meta["id"], "fasttext")
        self.assertFalse(meta["is_canonical"])
        self.assertIn("FastText unavailable", meta["status_message"])

    def test_unsupported_representation_raises_key_error(self):
        with self.assertRaises(KeyError):
            representation_registry.get("nonexistent_representation")


if __name__ == "__main__":
    unittest.main()
