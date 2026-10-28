"""Offline checks for the exact manual Iris quant asset binding; no model execution."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.ci import prepare_iris_quant_proof_assets as assets


class IrisQuantProofAssets(unittest.TestCase):
    def test_secondary_mac_uses_its_own_pinned_cache_and_rejects_other_runners(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            hub = home / "sceneworks-rw-weights" / "hub"
            hub.mkdir(parents=True)
            with patch.object(Path, "home", return_value=home):
                paths = assets.snapshot_paths("mlx", {"RUNNER_OS": "macOS", "RUNNER_NAME": "nax-macos-2"})
                self.assertEqual(paths[0][2], hub / "models--speridlabs--iris-3b" / "snapshots" / assets.IRIS_REVISION)
                self.assertEqual(paths[1][2], hub / "models--Qwen--Qwen3-VL-4B-Instruct" / "snapshots" / assets.QWEN_REVISION)
                with self.assertRaisesRegex(ValueError, "secondary Mac"):
                    assets.snapshot_paths("mlx", {"RUNNER_OS": "macOS", "RUNNER_NAME": "nax-macos"})
                hub.rmdir()
                with self.assertRaisesRegex(ValueError, "hub is missing"):
                    assets.snapshot_paths("mlx", {"RUNNER_OS": "macOS", "RUNNER_NAME": "nax-macos-2"})

    def test_missing_windows_cache_root_fails_before_export(self):
        for root in ("", "relative-cache"):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, "absolute"):
                assets.snapshot_paths("candle-cuda", {"RUNNER_OS": "Windows", "CANDLE_GEN_MODELS_ROOT": root})

    def test_receipt_carries_both_exact_upstream_manifests_and_required_file_digests(self):
        expected = (("IRIS_WEIGHTS_DIR", "speridlabs/iris-3b", assets.IRIS_REVISION, assets.IRIS_FILES),
                    ("IRIS_TEXT_ENCODER_DIR", "Qwen/Qwen3-VL-4B-Instruct", assets.QWEN_REVISION, assets.QWEN_FILES))
        rows = []
        paths = {}
        for name, repository, revision, files in expected:
            path = f"/cache/models--{repository.replace('/', '--')}/snapshots/{revision}"
            paths[name] = path
            rows.append({"environment": name, "path": path, "upstream": {
                "status": "hub_content_verified", "repository": repository, "revision": revision,
                "readback_url": f"https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true",
                "manifest_sha256": "a" * 64,
                "files": [{"path": file, "bytes": 1, "kind": "lfs_sha256", "digest": "b" * 64}
                          for file in files]}})
        receipt = {"schema_version": 1, "backend": "mlx", "assets": rows}
        self.assertIs(assets.validate_receipt(receipt, "mlx", paths), receipt)
        rows[0]["upstream"]["manifest_sha256"] = "missing"
        with self.assertRaisesRegex(ValueError, "upstream claim mismatch"):
            assets.validate_receipt(receipt, "mlx", paths)


if __name__ == "__main__":
    unittest.main()
