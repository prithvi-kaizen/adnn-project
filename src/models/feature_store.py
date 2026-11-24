"""Immutable 128-case feature shards with resumable row-level work files."""

import hashlib
import json
from pathlib import Path

import numpy as np

KEYS = ("last", "mean", "raw_final_last", "raw_final_mean")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


class FeatureChunk:
    def __init__(self, root: Path, ids: list[str], fingerprint: str, layers=25, width=896):
        if not ids or len(ids) != len(set(ids)) or len(ids) > 128:
            raise ValueError("A shard needs 1–128 distinct IDs")
        self.root, self.ids, self.layers, self.width = root, ids, layers, width
        root.mkdir(parents=True, exist_ok=True)
        metadata = {"ids": ids, "fingerprint": fingerprint, "layers": layers, "width": width,
                    "dtype": "float32"}
        path = root / "metadata.json"
        if path.exists() and json.loads(path.read_text()) != metadata:
            raise ValueError("Resume would change IDs, order, source or feature shape")
        if not path.exists():
            path.write_text(json.dumps(metadata, indent=2) + "\n")

    def row_path(self, index):
        if type(index) is not int or not 0 <= index < len(self.ids):
            raise ValueError("Unknown shard row")
        return self.root / f"row_{index:03d}.npz"

    def read_row(self, index):
        with np.load(self.row_path(index), allow_pickle=False) as saved:
            if str(saved["id"]) != self.ids[index]:
                raise ValueError("Saved row ID differs")
            result = {key: saved[key] for key in KEYS}
        self.validate(result)
        return result

    def validate(self, arrays):
        for key in KEYS:
            expected = (self.layers, self.width) if key in ("last", "mean") else (self.width,)
            value = np.asarray(arrays[key])
            if value.shape != expected or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("Feature shape, dtype or finite-value check failed")

    def write_row(self, index, arrays):
        path = self.row_path(index)
        if path.exists():
            raise ValueError("Never overwrite an observed feature row")
        self.validate(arrays)
        temporary = path.with_suffix(".part")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, id=np.array(self.ids[index]), **arrays)
        temporary.rename(path)
        self.read_row(index)

    def completed(self):
        result = self.root / "complete.json"
        if not result.exists():
            return None
        record = json.loads(result.read_text())
        if record["ids"] != self.ids or sha(self.root / "features.npz") != record["sha256"]:
            raise ValueError("Completed shard integrity differs")
        return record

    def finalize(self):
        if self.completed() is not None:
            return self.completed()
        rows = [self.read_row(i) for i in range(len(self.ids))]
        arrays = {key: np.stack([r[key] for r in rows], axis=1 if key in ("last", "mean") else 0)
                  for key in KEYS}
        target = self.root / "features.npz"
        with target.with_suffix(".part").open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        target.with_suffix(".part").rename(target)
        with np.load(target, allow_pickle=False) as check:
            if any(not np.array_equal(check[key], arrays[key]) for key in KEYS):
                raise ValueError("Compressed shard differs from the observed rows")
        record = {"ids": self.ids, "sha256": sha(target), "bytes": target.stat().st_size,
                  "shapes": {key: list(value.shape) for key, value in arrays.items()},
                  "row_work_cleanup": "only_after_lossless_shard_verification"}
        (self.root / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
        for index in range(len(self.ids)):
            self.row_path(index).unlink()
        return record
