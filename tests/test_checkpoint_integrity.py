import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.stage_course_checkpoint import file_receipt


class CheckpointIntegrityTests(unittest.TestCase):
    def test_git_blob_identity_is_distinct_from_plain_sha1(self):
        content = b'{"model_type":"qwen2"}\n'
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_bytes(content)
            receipt = file_receipt(path, {"bytes": len(content), "lfs_sha256": None, "etag": blob})
            self.assertTrue(receipt["pinned_identity_verified"])
            with self.assertRaises(ValueError):
                file_receipt(path, {"bytes": len(content), "lfs_sha256": None,
                                    "etag": hashlib.sha1(content).hexdigest()})

    def test_corrupted_lfs_payload_cannot_pass_even_with_matching_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "payload"
            path.write_bytes(b"bad")
            expected = {"bytes": 3, "lfs_sha256": hashlib.sha256(b"good"[:3]).hexdigest()}
            with self.assertRaises(ValueError):
                file_receipt(path, expected)

    def test_truncated_copy_never_passes_identity_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "payload"
            path.write_bytes(b"a")
            with self.assertRaises(ValueError):
                file_receipt(path, {"bytes": 2, "lfs_sha256": hashlib.sha256(b"ab").hexdigest()})


if __name__ == "__main__":
    unittest.main()
