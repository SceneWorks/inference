"""Mock upstream metadata checks; no network, model payload or GPU work."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.ci.pinned_hub_source import verify_pinned_hub_files


REPOSITORY = "speridlabs/iris-3b"
REVISION = "7445443349bc9abe3c96f01ff793e2098ca012b3"


def metadata(sibling, revision=REVISION):
    return json.dumps({"id": REPOSITORY, "sha": revision, "private": False,
                       "gated": False, "siblings": [sibling]}).encode()


class PinnedHubSource(unittest.TestCase):
    def verify(self, root, sibling, revision=REVISION):
        with patch("scripts.ci.pinned_hub_source.urllib.request.urlopen",
                   return_value=io.BytesIO(metadata(sibling, revision))):
            return verify_pinned_hub_files(REPOSITORY, REVISION, root, ("model.safetensors",))

    def test_lfs_content_must_match_upstream_sha256_even_when_size_and_path_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            file = root / "model.safetensors"
            file.write_bytes(b"original")
            sibling = {"rfilename": file.name, "size": file.stat().st_size,
                       "lfs": {"size": file.stat().st_size,
                               "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}}
            result = self.verify(root, sibling)
            self.assertEqual(result["status"], "hub_content_verified")
            self.assertEqual(result["files"][0]["kind"], "lfs_sha256")
            self.assertEqual(len(result["manifest_sha256"]), 64)
            file.write_bytes(b"replaced")
            with self.assertRaisesRegex(ValueError, "content mismatch"):
                self.verify(root, sibling)

    def test_wrong_revision_and_missing_digest_metadata_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.safetensors").write_bytes(b"payload")
            sibling = {"rfilename": "model.safetensors", "size": 7}
            with self.assertRaisesRegex(ValueError, "public pinned source"):
                self.verify(root, sibling, "0" * 40)
            with self.assertRaisesRegex(ValueError, "Git blob metadata invalid"):
                self.verify(root, sibling)

    def test_regular_git_blob_sidecar_uses_git_blob_sha1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            file = root / "model.safetensors"
            file.write_bytes(b"config")
            digest = hashlib.sha1(b"blob 6\0config").hexdigest()
            result = self.verify(root, {"rfilename": file.name, "size": 6, "blobId": digest})
            self.assertEqual(result["files"][0]["kind"], "git_blob_sha1")


if __name__ == "__main__":
    unittest.main()
