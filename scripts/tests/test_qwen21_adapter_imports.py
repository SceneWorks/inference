"""Fail-closed byte identity for the terminal cross-backend adapter transfer."""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
import zipfile
from unittest.mock import patch

from scripts.ci.qwen21_adapter_imports import (
    APPROVED_MANIFEST_BINDING_SHA256,
    APPROVED_SOURCE_ARTIFACT,
    CACHE_RECEIPT,
    cache_directory,
    extract_source_archive,
    manifest_binding_sha256,
    materialize,
    source_cache_status,
    stage_source_cache,
    validate_manifest,
    validate_source_artifact_metadata,
)
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

    def source_artifact_metadata(self, binding=None):
        binding = binding or APPROVED_SOURCE_ARTIFACT
        return {
            "id": binding["artifactId"],
            "name": binding["name"],
            "size_in_bytes": binding["sizeInBytes"],
            "digest": binding["digest"],
            "expired": False,
            "workflow_run": {"id": binding["runId"]},
        }

    def write_source(self, source, manifest, payload=b"adapter bytes"):
        source.mkdir()
        (source / manifest["adapters"][0]["file"]).write_bytes(payload)
        (source / "adapter-imports-resolved.json").write_text(
            json.dumps({**manifest, "directory": "/historical/run/imports/adapters"}),
            encoding="utf-8",
        )

    def approved_manifest(self, manifest):
        return patch(
            "scripts.ci.qwen21_adapter_imports.APPROVED_MANIFEST_BINDING_SHA256",
            manifest_binding_sha256(manifest),
        )

    def write_archive(self, archive, manifest, *, unsafe=False):
        receipt = json.dumps({**manifest, "directory": "/historical/run/imports/adapters"})
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
            output.writestr("imports/adapters/adapter-imports-resolved.json", receipt)
            output.writestr("imports/adapters/test.safetensors", b"adapter bytes")
            output.writestr("failed-candidate/edit.safetensors", b"excluded")
            if unsafe:
                output.writestr("../escape", b"unsafe")
        return {
            **APPROVED_SOURCE_ARTIFACT,
            "sizeInBytes": archive.stat().st_size,
            "digest": f"sha256:{hashlib.sha256(archive.read_bytes()).hexdigest()}",
        }

    def test_hash_mismatch_never_installs_or_leaves_a_partial_asset(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"GH_TOKEN": "test"}):
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                materialize(self.manifest(), root, lambda request: self.response(request, b"wrong content"))
            self.assertFalse((root / "test.safetensors").exists())
            self.assertEqual(list(root.glob("*.download")), [])
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

    def test_approved_cache_hit_materializes_without_token_or_network(self):
        manifest = self.manifest()
        with tempfile.TemporaryDirectory() as directory, self.approved_manifest(manifest):
            root = Path(directory)
            cache_root = root / "cache"; cache_root.mkdir()
            source = root / "source"
            self.write_source(source, manifest)
            staged = stage_source_cache(
                manifest, source, cache_root, self.source_artifact_metadata()
            )
            before = {path.name: path.stat().st_mtime_ns for path in staged.iterdir()}
            self.assertEqual(
                stage_source_cache(manifest, source, cache_root, self.source_artifact_metadata()),
                staged,
            )
            self.assertEqual(
                {path.name: path.stat().st_mtime_ns for path in staged.iterdir()}, before,
                "an already verified cache must be a no-op",
            )
            with patch.dict(os.environ, {}, clear=True):
                resolved = materialize(
                    manifest,
                    root / "destination",
                    opener=lambda _: self.fail("verified cache hit attempted network access"),
                    source_cache_root=cache_root,
                )
            self.assertTrue(resolved.is_file())
            self.assertEqual((resolved.parent / "test.safetensors").read_bytes(), b"adapter bytes")

    def test_missing_cache_uses_release_path_and_propagates_metadata_403(self):
        manifest = self.manifest()
        with tempfile.TemporaryDirectory() as directory, self.approved_manifest(manifest), \
                patch.dict(os.environ, {"GH_TOKEN": "test"}):
            root = Path(directory); cache_root = root / "cache"; cache_root.mkdir()
            requests = []
            def fetch(request):
                requests.append(request)
                return self.response(request, b"adapter bytes")
            materialize(manifest, root / "success", fetch, cache_root)
            self.assertEqual(len(requests), 2)
            with self.assertRaises(urllib.error.HTTPError) as caught:
                materialize(
                    manifest, root / "failure",
                    lambda _: (_ for _ in ()).throw(
                        urllib.error.HTTPError("metadata", 403, "Forbidden", {}, None)
                    ),
                    cache_root,
                )
            caught.exception.close()

    def test_staging_rejects_wrong_bytes_and_unexpected_failed_outputs(self):
        manifest = self.manifest()
        cases = [(b"short", "byte length mismatch"), (b"wrong content", "SHA-256 mismatch")]
        for payload, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory, \
                    self.approved_manifest(manifest):
                root = Path(directory); cache_root = root / "cache"; cache_root.mkdir()
                source = root / "source"; self.write_source(source, manifest, payload)
                with self.assertRaisesRegex(ValueError, message):
                    stage_source_cache(manifest, source, cache_root, self.source_artifact_metadata())
                self.assertFalse(cache_directory(manifest, cache_root).exists())
        with tempfile.TemporaryDirectory() as directory, self.approved_manifest(manifest):
            root = Path(directory); cache_root = root / "cache"; cache_root.mkdir()
            source = root / "source"; self.write_source(source, manifest)
            (source / "failed-edit-output.safetensors").write_bytes(b"excluded")
            with self.assertRaisesRegex(ValueError, "unexpected or partial"):
                stage_source_cache(manifest, source, cache_root, self.source_artifact_metadata())

    def test_archive_digest_crc_safe_paths_and_selected_extraction(self):
        manifest = self.manifest()
        with tempfile.TemporaryDirectory() as directory, self.approved_manifest(manifest):
            root = Path(directory); archive = root / "artifact.zip"
            binding = self.write_archive(archive, manifest)
            with patch.dict(APPROVED_SOURCE_ARTIFACT, binding, clear=True):
                extracted = extract_source_archive(
                    manifest, archive, root / "source", self.source_artifact_metadata(binding)
                )
            self.assertEqual(
                {path.name for path in extracted.iterdir()},
                {"adapter-imports-resolved.json", "test.safetensors"},
            )
            self.assertFalse((root / "source/failed-candidate").exists())

        for mutation in ("unsafe-path", "crc", "whole-digest"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory, \
                    self.approved_manifest(manifest):
                root = Path(directory); archive = root / "artifact.zip"
                binding = self.write_archive(archive, manifest, unsafe=mutation == "unsafe-path")
                if mutation == "crc":
                    body = archive.read_bytes()
                    marker = body.index(b"adapter bytes")
                    archive.write_bytes(body[:marker] + b"corrupt bytes" + body[marker + 13:])
                    binding["digest"] = f"sha256:{hashlib.sha256(archive.read_bytes()).hexdigest()}"
                elif mutation == "whole-digest":
                    binding["digest"] = "sha256:" + "0" * 64
                with patch.dict(APPROVED_SOURCE_ARTIFACT, binding, clear=True), \
                        self.assertRaisesRegex(ValueError, "unsafe path|CRC mismatch|SHA-256 mismatch"):
                    extract_source_archive(
                        manifest, archive, root / "source", self.source_artifact_metadata(binding)
                    )

    def test_partial_or_mismatched_existing_cache_fails_before_network(self):
        manifest = self.manifest()
        with tempfile.TemporaryDirectory() as directory, self.approved_manifest(manifest):
            root = Path(directory); cache_root = root / "cache"; cache_root.mkdir()
            partial = cache_directory(manifest, cache_root); partial.mkdir(parents=True)
            (partial / "test.safetensors").write_bytes(b"adapter bytes")
            with self.assertRaisesRegex(ValueError, "partial"):
                materialize(
                    manifest, root / "destination",
                    opener=lambda _: self.fail("partial cache attempted network access"),
                    source_cache_root=cache_root,
                )
            (partial / CACHE_RECEIPT).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected fields"):
                source_cache_status(manifest, cache_root)

    def test_cache_and_artifact_identity_mutants_fail_closed(self):
        manifest = self.manifest()
        validate_source_artifact_metadata(self.source_artifact_metadata())
        for key, value in [
            ("id", 1), ("name", "other"), ("size_in_bytes", 1),
            ("digest", "sha256:" + "0" * 64), ("expired", True),
        ]:
            metadata = self.source_artifact_metadata(); metadata[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_source_artifact_metadata(metadata)
        metadata = self.source_artifact_metadata(); metadata["workflow_run"]["id"] = 1
        with self.assertRaises(ValueError):
            validate_source_artifact_metadata(metadata)
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory)
            with self.assertRaisesRegex(ValueError, "approved immutable manifest"):
                source_cache_status(manifest, cache_root)

    def test_cache_directory_symlink_is_rejected(self):
        manifest = self.manifest()
        with tempfile.TemporaryDirectory() as directory, self.approved_manifest(manifest):
            root = Path(directory); cache_root = root / "cache"; cache_root.mkdir()
            parent = cache_root / "qwen21-transfer-cache"; parent.mkdir()
            external = root / "external"; external.mkdir()
            try:
                cache_directory(manifest, cache_root).symlink_to(external, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlink privilege unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "real directory"):
                source_cache_status(manifest, cache_root)

    def test_repository_manifest_keeps_the_approved_cache_binding(self):
        manifest_path = Path(__file__).resolve().parents[2] / "scripts/ci/qwen21_adapter_imports.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest_binding_sha256(manifest), APPROVED_MANIFEST_BINDING_SHA256)

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
