import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.models.feature_store import FeatureChunk


class FeatureStoreTests(unittest.TestCase):
    def arrays(self):
        return {"last": np.arange(6, dtype=np.float32).reshape(2, 3),
                "mean": np.ones((2, 3), dtype=np.float32),
                "raw_final_last": np.zeros(3, dtype=np.float32),
                "raw_final_mean": np.ones(3, dtype=np.float32)}

    def test_partial_resume_retains_order_and_exact_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = FeatureChunk(root, ["a", "b"], "source", layers=2, width=3)
            first.write_row(0, self.arrays())
            with self.assertRaises(FileNotFoundError):
                first.finalize()
            resumed = FeatureChunk(root, ["a", "b"], "source", layers=2, width=3)
            resumed.write_row(1, self.arrays())
            complete = resumed.finalize()
            self.assertEqual(complete["shapes"]["last"], [2, 2, 3])
            self.assertFalse(resumed.row_path(0).exists())
            self.assertEqual(resumed.completed(), complete)

    def test_resume_with_reordered_ids_or_changed_source_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            FeatureChunk(root, ["a", "b"], "source", layers=2, width=3)
            for ids, source in ((["b", "a"], "source"), (["a", "b"], "new-source")):
                with self.assertRaises(ValueError):
                    FeatureChunk(root, ids, source, layers=2, width=3)

    def test_nonfinite_and_duplicate_observations_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            chunk = FeatureChunk(Path(temporary), ["a"], "source", layers=2, width=3)
            bad = self.arrays()
            bad["last"][0, 0] = np.nan
            with self.assertRaises(ValueError):
                chunk.write_row(0, bad)
            chunk.write_row(0, self.arrays())
            with self.assertRaises(ValueError):
                chunk.write_row(0, self.arrays())

    def test_corrupt_completed_shard_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunk = FeatureChunk(root, ["a"], "source", layers=2, width=3)
            chunk.write_row(0, self.arrays())
            chunk.finalize()
            (root / "features.npz").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                chunk.completed()
