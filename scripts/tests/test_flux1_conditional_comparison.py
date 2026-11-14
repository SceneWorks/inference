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
    def test_probe_install_fetches_the_locked_graph_before_offline_steps(self):
        import subprocess
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs.get("timeout")))
            return subprocess.CompletedProcess(command, 0)

        with TemporaryDirectory() as tmp, patch.object(conditional_probe.subprocess, "run", side_effect=run), \
                patch.object(conditional_probe.overlay, "install", side_effect=RuntimeError("stop")):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                conditional_probe.install(Path(tmp), "mlx", "head", "tier1", "control-pose")
        self.assertEqual(calls, [(["cargo", "fetch", "--locked"],
                                  conditional_probe.CARGO_FETCH_TIMEOUT_SECONDS)])
        for failure, message in ((subprocess.TimeoutExpired("cargo", 1), "exceeded 1 s"),
                                 (subprocess.CalledProcessError(101, "cargo"), "exit 101")):
            with patch.object(conditional_probe.subprocess, "run", side_effect=failure):
                with self.assertRaisesRegex(ValueError, message):
                    conditional_probe.fetch_locked(Path("."), timeout=1)

    def test_child_caps_are_explicit_per_case(self):
        from types import SimpleNamespace
        gib = 1024**3
        policy = lambda host, gpu, deadline=600: SimpleNamespace(
            deadline_seconds=deadline, child_footprint_cap_bytes=host, child_gpu_cap_bytes=gpu)
        for case in conditional.CASES:
            caps = comparison.case_caps("candle-cuda", case, policy(24 * gib, 18 * gib))
            expected_host = 48 * gib if case == "pulid-pid" else 24 * gib
            self.assertEqual(caps["child_footprint_cap_bytes"], expected_host, case)
            expected_gpu = 40 * gib if case == "pulid-pid" else 18 * gib
            self.assertEqual(caps["child_gpu_cap_bytes"], expected_gpu, case)
            for backend, physical in comparison.RUNNER_HOST_BYTES.items():
                self.assertLess(expected_host, physical, (case, backend))
            with self.assertRaisesRegex(ValueError, "host cap"):
                comparison.case_caps("candle-cuda", case, policy(expected_host + 1, 18 * gib))
            with self.assertRaisesRegex(ValueError, "GPU cap"):
                comparison.case_caps("candle-cuda", case, policy(24 * gib, expected_gpu + 1))
            with self.assertRaisesRegex(ValueError, "20 minute"):
                comparison.case_caps("candle-cuda", case, policy(gib, gib, 1201))
        # The pulid-pid caps admit run 43553801528's 24.79 GB whole-GPU use and run 43560615425's
        # 28.87 GB host footprint, with headroom over the 32.3 GB estimated host peak.
        comparison.case_caps("candle-cuda", "pulid-pid", policy(48 * gib, 40 * gib))
        self.assertGreater(40 * gib, 24.79e9)
        self.assertGreater(48 * gib, 32.3e9 * 1.5)
        # MLX unified memory: one footprint cap, no CUDA cap to record.
        mlx = comparison.case_caps("mlx", "pulid-pid", policy(48 * gib, None))
        self.assertEqual(mlx, {"child_footprint_cap_bytes": 48 * gib, "child_gpu_cap_bytes": None})
        self.assertEqual(comparison.case_caps("mlx", "control-pose", policy(24 * gib, None)),
                         {"child_footprint_cap_bytes": 24 * gib, "child_gpu_cap_bytes": None})

    def test_candle_pulid_probes_freeze_the_comparison_sampler(self):
        for case in conditional.CASES:
            if not case.startswith("pulid-") or "candle-cuda" not in conditional.CASES[case][1]:
                continue
            with self.subTest(case=case):
                setup = conditional_probe.candle_setup(case)
                self.assertIn('sampler: Some("euler".into())', setup)
                self.assertEqual(conditional_evidence.SETTINGS["sampler"], "euler")

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
