"""Fail-closed byte identity for the terminal cross-backend adapter transfer."""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.ci.qwen21_adapter_imports import materialize, validate_manifest
from scripts.ci.qwen21_weights_root import verify_root


class AdapterImportTests(unittest.TestCase):
    def manifest(self):
        return {
            "repository": "SceneWorks/inference", "releaseId": 123,
            "adapters": [{"name": "cuda_lora", "file": "test.safetensors",
                          "kind": "lora", "assetId": 456, "size": 13, "assetName": "test.safetensors",
                          "sha256": hashlib.sha256(b"adapter bytes").hexdigest()}],
        }

    def response(self, request, payload):
        if request.full_url.endswith('/releases/123'):
            return io.BytesIO(json.dumps({'assets': [{'id': 456, 'name': 'test.safetensors', 'size': 13}]}).encode())
        return io.BytesIO(payload)

    def test_hash_mismatch_never_installs_or_leaves_a_partial_asset(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"GH_TOKEN": "test"}):
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                materialize(self.manifest(), root, lambda request: self.response(request, b"wrong bytes"))
            self.assertFalse((root / "test.safetensors").exists())
            self.assertFalse((root / "test.download").exists())
            self.assertFalse((root / "adapter-imports-resolved.json").exists())

    def test_authenticated_fetch_and_verified_reuse(self):
        requests = []
        def fetch(request):
            requests.append(request)
            return self.response(request, b"adapter bytes")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"GH_TOKEN": "test"}):
            root = Path(directory)
            resolved = materialize(self.manifest(), root, fetch)
            self.assertTrue(resolved.is_file())
            materialize(self.manifest(), root, fetch)
            self.assertEqual(len(requests), 3)
            self.assertEqual(requests[0].get_header("Authorization"), "Bearer test")
            self.assertEqual(requests[1].full_url,
                "https://api.github.com/repos/SceneWorks/inference/releases/assets/456")
            (root / "test.safetensors").write_bytes(b"corrupted")
            materialize(self.manifest(), root, fetch)
            self.assertEqual(len(requests), 5)

    def test_manifest_rejects_missing_hash_duplicate_and_path_escape(self):
        for key, value in [("file", "../bad.safetensors"), ("sha256", ""),
                           ("kind", "unknown"), ("assetId", -1)]:
            bad = self.manifest()
            bad["adapters"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest(bad)
        duplicate = self.manifest()
        duplicate["adapters"].append(copy.deepcopy(duplicate["adapters"][0]))
        with self.assertRaises(ValueError):
            validate_manifest(duplicate)

    def test_release_ownership_and_size_fail_before_download(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"GH_TOKEN": "test"}):
            for assets in ([], [{"id": 456, "name": "test.safetensors", "size": 12}]):
                with self.subTest(assets=assets), self.assertRaisesRegex(ValueError, "does not own"):
                    materialize(self.manifest(), Path(directory),
                        lambda request: io.BytesIO(json.dumps({"assets": assets}).encode()))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_task_root_accepts_internal_blob_links_and_rejects_external_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "internal"
            hub = root / "hub"
            hub.mkdir(parents=True)
            blob = hub / "blob"
            blob.write_bytes(b"weights")
            link = hub / "snapshot-link"
            try:
                link.symlink_to(blob)
            except OSError as exc:
                self.skipTest(f"symlink privilege unavailable: {exc}")
            self.assertEqual(verify_root(root), root.resolve())
            external = Path(directory) / "external"
            external.mkdir()
            (root / "xet").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                verify_root(root)


if __name__ == "__main__":
    unittest.main()
