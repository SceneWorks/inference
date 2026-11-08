"""Weights-free checks for the manual conditional comparison contract."""
import hashlib
import math
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import tomllib
import unittest
from unittest.mock import patch

from scripts.ci import flux1_conditional_comparison as comparison
from scripts.ci.flux1_pair import conditional, conditional_evidence, conditional_probe


class ConditionalComparison(unittest.TestCase):
    def test_probe_manifests_promote_image_without_duplicate_dependency(self):
        for package, family in (("mlx-gen-flux", "mlx-gen"), ("mlx-gen-pulid", "mlx-gen"),
                                ("candle-gen-flux", "candle-gen"),
                                ("candle-gen-pulid", "candle-gen")):
            with self.subTest(package=package):
                path = (comparison.ROOT / "crates/media" / family / package / "Cargo.toml")
                source = path.read_text(encoding="utf-8")
                changed = conditional_probe.proof_manifest(source, package)
                parsed = tomllib.loads(changed)
                self.assertEqual(changed.count("\nimage = { workspace = true }\n"), 1)
                self.assertIn("image", parsed["dependencies"])
                self.assertNotIn("image", parsed["dev-dependencies"])
                self.assertEqual(changed.count("\nserde_json = { workspace = true }\n"), 1)
                self.assertIn("serde_json", parsed["dependencies"])

    def test_fixed_case_matrix_rejects_missing_or_unpinned_assets(self):
        self.assertEqual(len(conditional.CASES), 11)
        for case, (_, backends, roles) in conditional.CASES.items():
            for backend in backends:
                with self.subTest(case=case, backend=backend):
                    prefix = "C:\\models\\" if backend == "candle-cuda" else "/models/"
                    assets = {role: ({"path": prefix + "gemma", "files": {
                                      "config.json": "a" * 64, "tokenizer.json": "a" * 64,
                                      "gemma-2-2b-it.safetensors": "a" * 64},
                                      "revision": "commit-1"} if role == "pid_gemma" else
                                     {"path": prefix + "weights", "sha256": "a" * 64,
                                      "revision": "commit-1"}) for role in roles}
                    self.assertEqual(conditional.case_metadata(case, backend, assets)[0],
                                     conditional.variant(case))
                    role = roles[0]
                    broken = dict(assets)
                    broken[role] = dict(broken[role], revision="")
                    with self.assertRaises(ValueError):
                        conditional.case_metadata(case, backend, broken)
                    broken = dict(assets)
                    broken.pop(role)
                    with self.assertRaises(ValueError):
                        conditional.case_metadata(case, backend, broken)
                    if role == "pid_gemma":
                        broken = dict(assets)
                        broken[role] = dict(assets[role], files={"config.json": "a" * 64})
                        with self.assertRaisesRegex(ValueError, "PiD Gemma needs"):
                            conditional.case_metadata(case, backend, broken)
        with self.assertRaises(ValueError):
            conditional.case_metadata("ip-dev-cfg", "candle-cuda", {})

    def test_staged_assets_are_detached_from_mutable_source(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            hub = root / "hub"
            hub.mkdir()
            file = hub / "overlay.safetensors"
            file.write_bytes(b"fixed overlay")
            gemma = hub / "gemma"
            gemma.mkdir()
            (gemma / "config.json").write_bytes(b"fixed config")
            (gemma / "tokenizer.json").write_bytes(b"fixed tokenizer")
            (gemma / "gemma-2-2b-it.safetensors").write_bytes(b"fixed weights")
            digest = lambda data: hashlib.sha256(data).hexdigest()
            rows = {"control": {"path": str(file), "sha256": digest(file.read_bytes()),
                                   "revision": "caller-commit"},
                    "pid_gemma": {"path": str(gemma),
                                  "files": {"config.json": digest(b"fixed config"),
                                            "tokenizer.json": digest(b"fixed tokenizer"),
                                            "gemma-2-2b-it.safetensors": digest(b"fixed weights")},
                                  "revision": "caller-commit"}}
            source = comparison.ROOT
            with patch.dict(os.environ, FLUX1_HUB=str(hub)):
                _, staged = conditional.prepare_assets(root, source, rows)
            file.write_bytes(b"changed")
            (gemma / "config.json").write_bytes(b"changed")
            self.assertEqual(Path(staged["control"]["staged"]).read_bytes(), b"fixed overlay")
            self.assertEqual((Path(staged["pid_gemma"]["staged"]) / "config.json").read_bytes(),
                             b"fixed config")
            self.assertEqual(conditional.REFERENCE_SHA256,
                             hashlib.sha256(Path(staged["reference_image"]["staged"]).read_bytes()).hexdigest())

    def test_literal_tilde_hub_is_expanded_and_still_confines_assets(self):
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            hub = home / "hub"
            hub.mkdir()
            inside = hub / "overlay.safetensors"
            inside.write_bytes(b"inside")
            outside = home / "outside.safetensors"
            outside.write_bytes(b"outside")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            (home / "inside-proof").mkdir()
            (home / "outside-proof").mkdir()
            with patch.dict(os.environ, HOME=str(home), FLUX1_HUB="~/hub"):
                _, staged = conditional.prepare_assets(home / "inside-proof", comparison.ROOT,
                                                       {"control": {"path": str(inside),
                                                                    "sha256": digest(inside),
                                                                    "revision": "caller-commit"}})
                self.assertEqual(Path(staged["control"]["staged"]).read_bytes(), b"inside")
                with self.assertRaisesRegex(ValueError, "outside the configured model hub"):
                    conditional.prepare_assets(home / "outside-proof", comparison.ROOT,
                                               {"control": {"path": str(outside),
                                                            "sha256": digest(outside),
                                                            "revision": "caller-commit"}})

    def test_preload_plan_rejects_wrong_source_or_route(self):
        scale_bits = struct.unpack("<I", struct.pack("<f", 0.7))[0]
        plan = {"provider": "flux1_dev_control", "backend": "mlx",
                "inputs": {"requested_seed": 104729,
                           "width": 256, "height": 256, "count": 1, "steps": 3, "sampler": "euler",
                           "negative_prompt": None, "true_cfg_bits": None,
                           "timestep_to_start_cfg": None, "use_pid": False,
                           "conditioning": {"Control": {"kind": "pose", "width": 256,
                                                        "height": 256, "scale_bits": scale_bits}}},
                "preparation": {"adapters": [], "sequential_offload": False,
                                "deferred_materialization": False, "control": {"File": "/x"},
                                "ip_adapter": None, "identity": None, "pid": None},
                "pass": {"sigmas": [1.0, math.exp(0.5) / (math.exp(0.5) + 0.5),
                                    math.exp(0.5) / (math.exp(0.5) + 2), 0.0],
                         "phase": {"first_step": 0, "end_step": 3, "sampler": "euler", "guidance": 3.5,
                                   "stream": {"width": 256, "height": 256, "patch_stride": 16,
                                              "tokens": 256, "token_width": 64}}}}
        conditional_evidence.validate_plan(plan, "control-pose", "mlx")
        plan["inputs"]["conditioning"] = {"Reference": {}}
        with self.assertRaises(ValueError):
            conditional_evidence.validate_plan(plan, "control-pose", "mlx")


if __name__ == "__main__":
    unittest.main()
