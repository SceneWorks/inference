"""Weights-free checks for the manual conditional asset inventory protocol."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.ci import flux1_conditional_asset_inventory as assets
from scripts.ci.flux1_pair import conditional


def put(root, relative, content=b"fixed"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


class ConditionalAssetInventory(unittest.TestCase):
    def test_huggingface_gemma_snapshot_links_stage_from_confined_blobs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            hub = root / "hub"
            snapshot = hub / "models--Efficient-Large-Model--gemma-2-2b-it/snapshots" / ("b" * 40)
            snapshot.mkdir(parents=True)
            for name, content in (("config.json", b"config"), ("tokenizer.json", b"tokenizer"),
                                  ("gemma-2-2b-it.safetensors", b"weights")):
                blob = put(hub, "models--Efficient-Large-Model--gemma-2-2b-it/blobs/" + name, content)
                (snapshot / name).symlink_to(blob)
            result = assets.inventory(hub, "nax-macos-2")
            row = result["roles"]["pid_gemma"]["candidates"][0]
            self.assertEqual(result["roles"]["pid_gemma"]["status"], "unique")
            work = root / "proof"
            work.mkdir()
            with patch.dict(os.environ, FLUX1_HUB=str(hub)):
                _, staged = conditional.prepare_assets(work, assets.ROOT,
                                                       {"pid_gemma": {"path": row["path"],
                                                                      "files": row["files"],
                                                                      "revision": row["revision"]}})
            self.assertEqual((Path(staged["pid_gemma"]["staged"]) / "gemma-2-2b-it.safetensors").read_bytes(),
                             b"weights")
            external = put(root, "external.safetensors", b"other")
            (snapshot / "gemma-2-2b-it.safetensors").unlink()
            (snapshot / "gemma-2-2b-it.safetensors").symlink_to(external)
            with self.assertRaisesRegex(ValueError, "escapes configured model hub"):
                assets.inventory(hub, "nax-macos-2")

    def test_config_only_gemma_is_incomplete_and_cannot_stage(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            hub = root / "hub"
            snapshot = hub / "models--google--gemma-2-2b-it/snapshots" / ("a" * 40)
            put(hub, snapshot.relative_to(hub) / "config.json", b"config")
            result = assets.inventory(hub, "nax-macos-2")
            self.assertEqual(result["roles"]["pid_gemma"]["status"], "incomplete")
            self.assertIn("pid_gemma", result["incomplete_roles"])
            self.assertFalse(result["cases"]["pulid-pid"]["ready"])
            work = root / "proof"
            work.mkdir()
            with patch.dict(os.environ, FLUX1_HUB=str(hub)):
                with self.assertRaisesRegex(ValueError, "PiD Gemma needs"):
                    conditional.prepare_assets(work, assets.ROOT,
                                               {"pid_gemma": {"path": str(snapshot),
                                                              "files": {"config.json": hashlib.sha256(b"config").hexdigest()},
                                                              "revision": "cache-snapshot"}})

    def test_indexed_gemma_requires_every_declared_shard(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            put(directory, "config.json", b"config")
            put(directory, "tokenizer.json", b"tokenizer")
            shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
            put(directory, shards[0], b"first")
            put(directory, "model.safetensors.index.json",
                json.dumps({"weight_map": {"a": shards[0], "b": shards[1]}}).encode())
            with self.assertRaisesRegex(ValueError, "incomplete"):
                conditional.require_pid_gemma_footprint(directory, {
                    "config.json", "tokenizer.json", "model.safetensors.index.json", shards[0]})
            put(directory, shards[1], b"second")
            conditional.require_pid_gemma_footprint(directory, {
                "config.json", "tokenizer.json", "model.safetensors.index.json", *shards})

    def test_fixed_roles_and_complete_gemma_map_from_confined_snapshot(self):
        with TemporaryDirectory() as tmp:
            hub = Path(tmp) / "hub"
            hub.mkdir()
            rev = "a" * 40
            snapshot = Path("models--Shakker-Labs--FLUX.1-dev-ControlNet-Union-Pro-2.0/snapshots") / rev
            control = put(hub, snapshot / "diffusion_pytorch_model.safetensors", b"control")
            put(hub, "models--XLabs-AI--flux-ip-adapter/snapshots/" + rev + "/ip_adapter.safetensors")
            put(hub, "models--openai--clip-vit-large-patch14/snapshots/" + rev + "/model.safetensors")
            put(hub, "models--guozinan--PuLID/snapshots/" + rev + "/pulid_flux_v0.9.1.safetensors")
            put(hub, "converted/eva02_clip_l_14_336.safetensors")
            for name in ("scrfd_10g", "arcface_iresnet100", "bisenet_parsing"):
                put(hub, "face/" + name + ".safetensors")
            put(hub, "pid/flux_2k.safetensors")
            gemma = "models--google--gemma-2-2b-it/snapshots/" + rev
            put(hub, gemma + "/config.json", b"config")
            put(hub, gemma + "/tokenizer.json", b"tokenizer")
            put(hub, gemma + "/gemma-2-2b-it.safetensors", b"gemma")
            result = assets.inventory(hub, "nax-macos-2")
            self.assertEqual(result["missing_roles"], [])
            self.assertEqual(result["ambiguous_roles"], [])
            self.assertTrue(all(row["status"] == "unique" for row in result["roles"].values()))
            self.assertTrue(all(row["ready"] for row in result["cases"].values()))
            control_row = result["roles"]["control"]["candidates"][0]
            self.assertEqual(control_row["path"], str(control.resolve()))
            self.assertEqual(control_row["sha256"], hashlib.sha256(b"control").hexdigest())
            self.assertEqual(control_row["revision"], rev)
            self.assertEqual(control_row["revision_provenance"], "huggingface-cache-snapshot-path")
            files = result["roles"]["pid_gemma"]["candidates"][0]["files"]
            self.assertEqual(files, {"config.json": hashlib.sha256(b"config").hexdigest(),
                                     "tokenizer.json": hashlib.sha256(b"tokenizer").hexdigest(),
                                     "gemma-2-2b-it.safetensors": hashlib.sha256(b"gemma").hexdigest()})
            self.assertNotIn("pulid-real", assets.inventory(hub, "cuda-windows")["cases"])

    def test_missing_ambiguous_and_symlink_escape(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            hub = root / "hub"
            hub.mkdir()
            put(hub, "flux-a/control.safetensors")
            put(hub, "flux-b/control.safetensors")
            result = assets.inventory(hub, "nax-macos-2")
            self.assertEqual(result["roles"]["control"]["status"], "ambiguous")
            self.assertEqual(result["roles"]["pulid"]["status"], "missing")
            self.assertIn("control", result["ambiguous_roles"])
            self.assertIn("pulid", result["missing_roles"])
            self.assertFalse(result["cases"]["control-pose"]["ready"])
            external = put(root, "external.safetensors")
            (hub / "flux-a/ip_adapter.safetensors").symlink_to(external)
            with self.assertRaisesRegex(ValueError, "escapes configured model hub"):
                assets.inventory(hub, "nax-macos-2")

    def test_hash_budget_and_exact_runner_cli(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            hub = root / "hub"
            hub.mkdir()
            put(hub, "flux/control.safetensors", b"control")
            with patch.object(assets, "MAX_HASH_BYTES", 2):
                with self.assertRaisesRegex(ValueError, "budget exceeded"):
                    assets.inventory(hub, "nax-macos-2")
            output = root / "inventory.json"
            command = [sys.executable, "-I", str(Path(assets.__file__)), "--runner", "nax-macos-2",
                       "--output", str(output)]
            env = dict(os.environ, RUNNER_NAME="nax-macos-2", FLUX1_HUB=str(hub))
            subprocess.run(command, env=env, check=True)
            self.assertEqual(json.loads(output.read_text())["runner"], "nax-macos-2")
            env["RUNNER_NAME"] = "nax-macos"
            self.assertNotEqual(subprocess.run(command, env=env, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
