import importlib.util
import json
import unittest
from pathlib import Path

from src.data.splits import candidate_split, validate_split

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("scipy"), "pinned research environment provides SciPy")
class LargerSplitTests(unittest.TestCase):
    def test_actual_v5_allocations_are_repeatable_and_group_disjoint(self):
        rows = [json.loads(line) for line in (ROOT / "data/candidates/v5/sandbox_pairs.jsonl").read_text().splitlines()]
        for seed in (1729, 2718, 31415):
            actual = candidate_split(rows, seed)
            validate_split(rows, actual)
            self.assertEqual(actual["counts"], {"train": 1080, "development": 360, "test": 360})
            self.assertTrue(actual["exact_60_20_20"])
            self.assertFalse(actual["final_freeze"])
            self.assertEqual(actual, candidate_split(rows, seed))
            by_id = {r["id"]: r for r in rows}
            for ids in actual["indices"].values():
                self.assertEqual({by_id[k]["label"] for k in ids},
                                 {"reversible", "partially_reversible", "irreversible"})
