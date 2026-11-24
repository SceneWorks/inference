"""Metadata-only checks for the explicitly dispatched Mochi/SVD and SCAIL-2 proofs."""
from contextlib import ExitStack
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.ci import video_tier1_cohorts as cohorts
from scripts.ci import video_tier1_comparison as pair
from scripts.ci import video_tier1_evidence as evidence


class VideoTier1Cohorts(unittest.TestCase):
    def test_committed_recipe_shapes_and_segment_counts(self):
        for selection, lane_counts in ((cohorts.MOCHI_SVD, (3, 4)), (cohorts.SCAIL, (2, 2))):
            for backend, count in zip(pair.LANES, lane_counts, strict=True):
                cases = cohorts.cases(selection, backend)
                self.assertEqual(len(cases), count)
                self.assertEqual(len(pair.phases(cases)), 10 * count)
                for case in cases:
                    with self.subTest(case=case), TemporaryDirectory() as directory:
                        spec = evidence.case_spec(case)
                        descriptor = cohorts.BY_NAME[case]
                        self.assertEqual(spec["settings"]["depth"], descriptor.depth)
                        self.assertEqual(spec["settings"]["seed"], 104729)
                        steps = 8 if "scail2-replacement" in case else 4
                        self.assertEqual(spec["outer_steps"], steps)
                        out = Path(directory)
                        shape = spec["tensor_shape"]
                        size = math.prod(shape)
                        raw = struct.pack("<f", 1.) + bytes(4 * (size - 1))
                        (out / "state-0.f32").write_bytes(raw)
                        (out / f"state-{steps}.f32").write_bytes(raw)
                        row = {"case": case, "trace": spec["trace"], "tensor_shape": shape, "warmups": 1,
                               "allocator_peak_bytes": 100, "settings": spec["settings"],
                               "samples": [{"elapsed_ns": 100, "outer_steps": steps}]}
                        (out / "result.json").write_text(json.dumps(row), encoding="utf-8")
                        obs = pair.read_observed(out, case)
                        def capture():
                            return evidence.capture(obs["result"], bytes.fromhex(obs["initial_le_f32_hex"]),
                                bytes.fromhex(obs["terminal_le_f32_hex"]), case, spec, pair.SETTINGS,
                                SimpleNamespace(deadline_seconds=10))
                        self.assertEqual(capture()["terminal"]["shape"], shape)
                        obs["result"]["trace"].reverse()
                        with self.assertRaisesRegex(ValueError, "ordered trace"):
                            capture()

    def test_two_packages_discovered_once_each_arm_and_bound_to_their_binary(self):
        windows = os.name == "nt"
        backend = "candle-cuda" if windows else "mlx"
        selection = cohorts.MOCHI_SVD
        contract = {"backend": backend, "cases": cohorts.cases(selection, backend),
                    "request": {"selection": selection}}
        models = cohorts.models(contract["cases"])
        self.assertEqual(models, ("mochi", "svd"))
        with TemporaryDirectory() as directory:
            work = Path(directory)
            def cargo(args, *, cwd, env, **_):
                package = args[args.index("-p") + 1]
                target = package.replace("-", "_")
                row = pair.case_row(backend, package.rsplit("-", 1)[-1])
                exe = Path(env["CARGO_TARGET_DIR"]) / "debug" / "deps" / (target + ("-abcd.exe" if windows else "-abcd"))
                exe.parent.mkdir(parents=True, exist_ok=True)
                exe.write_bytes(target.encode())
                artifact = {"reason": "compiler-artifact", "profile": {"test": True},
                            "target": row["cargo_target_metadata"], "executable": str(exe),
                            "package_id": f"path+file:///src/{package}#0.0.0", "features": ["model-inference-proof", "cuda", "default"] if windows else ["model-inference-proof"]}
                stdout = json.dumps(artifact) + f'\n{row["fqn"]}: test\n1 test, 0 benchmarks\n'
                return SimpleNamespace(stdout=stdout)
            with patch.object(pair, "checkout_and_overlay", side_effect=lambda work, contract, arm: (work/arm, {})), \
                 patch.object(pair.proof, "verified_toolchain", return_value={}), patch.object(pair.subprocess, "run", side_effect=cargo) as call:
                for arm in pair.ARMS:
                    _, binaries, _, record = pair.build(work, contract, arm, SimpleNamespace(backend=pair.proof.POLICY_BACKENDS[backend], cuda_device_uuid="mock"))
                    self.assertEqual(set(binaries), set(models))
                    self.assertNotEqual(binaries["mochi"], binaries["svd"])
                    mismatched = deepcopy(record["targets"]["mochi"])
                    mismatched["discovery"]["executable"] = str(binaries["svd"])
                    with self.assertRaisesRegex(ValueError, "target"):
                        pair.verify_target(mismatched, backend, "mochi", arm)
            self.assertEqual(call.call_count, 4)

    def test_scail_dependency_overlay_rejects_another_file_or_changed_getter(self):
        from scripts.ci.scail_pair import overlay
        src = "crates/media/candle-gen/candle-gen-scail2/src/"
        scheduler = "crates/media/candle-gen/candle-gen-wan/src/scheduler.rs"
        original = b"// production\n"
        edits = {src + name: original + b"\n#[cfg(test)] mod probe {}"
                 for name in ("model.rs", "generate.rs", "pipeline.rs")}
        edits[scheduler] = original + overlay.CANDLE_SAMPLER_READBACK.encode()
        for mutation in (None, "getter", "extra", "production"):
            with self.subTest(mutation=mutation):
                data = dict(edits)
                if mutation == "getter":
                    data[scheduler] = data[scheduler].replace(b"self.sampler", b"Sampler::UniPC")
                if mutation == "extra":
                    data[src + "lib.rs"] = b"another file"
                if mutation == "production":
                    data[src + "model.rs"] = bytes(reversed(original)) + data[src + "model.rs"]
                with patch.object(overlay, "render", return_value=(b"probe", data)):
                    def install():
                        return cohorts.overlay_files("a"*40, "candle-cuda", "base", "scail2", lambda *_: original, None)
                    if mutation is None:
                        self.assertEqual(install()[1], b"probe")
                    else:
                        with self.assertRaises(ValueError):
                            install()

    def test_counter_preflight_fails_before_fixture_or_cargo(self):
        from scripts.tests.test_wan_tier1_comparison import request
        selection = cohorts.MOCHI_SVD
        backend = "candle-cuda"
        contract = {"backend": backend, "cases": cohorts.cases(selection, backend),
                    "request": dict(request(backend, cohorts.VIDEO_BASE), selection=selection)}
        with TemporaryDirectory() as location, ExitStack() as patches:
            root = Path(location)
            patches.enter_context(patch.dict(os.environ, RUNNER_TEMP=location, RUNNER_NAME="runner",
                MODEL_PROOF_RUNNER_ENVIRONMENT="self-hosted", GITHUB_RUN_ID="7", GITHUB_RUN_ATTEMPT="1",
                GITHUB_ENV=str(root / "bindings")))
            patches.enter_context(patch.object(pair.proof, "require_host"))
            patches.enter_context(patch.object(pair.paired, "hardware", return_value={"uuid": "mock"}))
            patches.enter_context(patch.object(pair, "cuda_adapter", side_effect=ValueError("missing counter")))
            fixture = patches.enter_context(patch.object(cohorts, "fixture"))
            build = patches.enter_context(patch.object(pair, "build"))
            output = root / "evidence"
            with self.assertRaisesRegex(ValueError, "missing counter"):
                pair.execute(SimpleNamespace(output=output), contract)
            fixture.assert_not_called()
            build.assert_not_called()
            receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
            self.assertFalse(receipt["accepted"])
            self.assertEqual(receipt["observations"], {})

    def test_only_comparable_dedicated_usage_can_resolve_a_wide_wddm_interval(self):
        def run(peak, luid="a", pid_peak=64, unknown=2):
            return {"initial": "same", "terminal": "same", "trace": [], "trace_sha256": "same",
                    "peaks": {"pool": 64}, "samples": [{"elapsed_ns": 10, "outer_steps": 4}],
                    "cuda_process": {"basis": "wddm-bounded" if unknown else "owned-pid",
                                     "lower": pid_peak, "upper": 200 if unknown else pid_peak,
                                     "dedicated": {"adapter": {"luid": luid}, "sampled_peak_bytes": peak}}}
        base = [run(100)] * 5
        for candidate, expected in ((run(102), "pass"), (run(103), "fail"), (run(None), "inconclusive"),
                                    (run(100, "other"), "inconclusive")):
            result = evidence.compare(base, [candidate] * 5, pair.SETTINGS)
            self.assertEqual(result["metrics"]["wddm_dedicated_process_bytes"]["verdict"], expected)
            self.assertEqual(result["metrics"]["cuda_process_bytes"]["verdict"], "inconclusive")
            self.assertEqual(result["accepted"], expected == "pass")
        result = evidence.compare([run(100, unknown=0)] * 5, [run(100, pid_peak=96, unknown=0)] * 5, pair.SETTINGS)
        self.assertEqual(result["metrics"]["cuda_process_bytes"]["verdict"], "fail")
        self.assertFalse(result["accepted"])
