"""Metadata-only checks for the manual Wan base/candidate controller."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.ci import wan_tier1_comparison as pair
from scripts.ci import wan_tier1_evidence as evidence
from scripts.media_campaign_supervisor import runtime_guarded_admission


def request(backend, base):
    policy = {"schemaVersion": 1, "backend": pair.proof.POLICY_BACKENDS[backend],
              "deadlineSeconds": 90, "pollMillis": 100, "termGraceMillis": 500,
              "hostFreeReserveBytes": pair.GIB, "childFootprintCapBytes": 20 * pair.GIB,
              "stdoutCapBytes": 65536, "stderrCapBytes": 65536, "eventCapBytes": 65536}
    if backend == "candle-cuda":
        policy.update(cudaDeviceUuid="GPU-00000000-1111-2222-3333-444444444444",
                      gpuFreeReserveBytes=pair.GIB, childGpuCapBytes=10 * pair.GIB)
    return {"selection": pair.SELECTION, "base": base, "policy": policy}


class WanPlanningTests(unittest.TestCase):
    def test_ancestry_overlay_and_fixed_selection(self):
        with TemporaryDirectory() as temporary, ExitStack() as patches:
            root = Path(temporary) / "repo"
            root.mkdir()
            def git(*args):
                return pair.paired.git(root, "-c", "user.name=proof-test", "-c", "user.email=test@localhost", *args)
            git("init", "-q")
            src = root / "crates/media/mlx-gen/mlx-gen-wan/src"
            src.mkdir(parents=True)
            original = b"// Existing production code.\n"
            (src / "lib.rs").write_bytes(original)
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            git("checkout", "-qb", "unrelated")
            git("commit", "--allow-empty", "-qm", "unrelated")
            other = git("rev-parse", "HEAD")
            git("checkout", "-q", "--detach", base)
            bundle = root / "scripts/ci/wan_pair"
            bundle.mkdir(parents=True)
            (bundle / "mlx_probe.rs.in").write_bytes(b"test\n@WAN_PAIR_COMMON@\n@WAN_MLX_SWAP_SCOPE@")
            (bundle / "common.rs.in").write_bytes(b"common")
            (bundle / "fixture.py").write_text("# Fake metadata-only fixture")
            (bundle / "headers.json").write_text("{}")
            git("add", ".")
            git("commit", "-qm", "candidate")
            head = git("rev-parse", "HEAD")
            patches.enter_context(patch.object(pair, "ROOT", root))
            patches.enter_context(patch.object(pair.proof, "ROOT", root))
            patches.enter_context(patch.object(pair, "BASE", base))
            value = request("mlx", base)
            contract = pair.plan("mlx", value, head)
            self.assertEqual(contract["runner"][-1], "rw-mage")
            self.assertEqual(contract["base"], base)
            self.assertEqual(contract["cases"], list(pair.CASES["mlx"]))
            work = Path(temporary) / "work"
            work.mkdir()
            builds = [pair.checkout_and_overlay(work, contract, arm) for arm in pair.ARMS]
            self.assertEqual((src / "lib.rs").read_bytes(), original)
            for arm, (arm_root, record) in zip(pair.ARMS, builds):
                self.assertEqual((arm_root / src.relative_to(root) / "lib.rs").read_bytes(),
                                 original + b"\n#[cfg(test)]\nmod wan_pair_fixture;\n")
                self.assertEqual(record["fixture_source_sha256"], hashlib.sha256(b"test\ncommon\n" + (b"" if arm == "base" else b"None,")).hexdigest())
                self.assertEqual(record["fixture_source_sha256"], contract["fixture_source_sha256"][arm])
            for invalid in (dict(value, command=["bash"]), dict(value, base=other),
                            dict(value, policy=dict(value["policy"], deadlineSeconds=1201))):
                with self.assertRaises(ValueError):
                    pair.plan("mlx", invalid, head)
            with patch.object(pair, "BASE", other), self.assertRaises(subprocess.CalledProcessError):
                pair.plan("mlx", dict(value, base=other), head)
            (src / "lib.rs").write_bytes(original + b"// dirty\n")
            with self.assertRaisesRegex(ValueError, "modifications"):
                pair.plan("mlx", value, head)

    def test_environment_drops_unreviewed_commands_and_model_settings(self):
        with patch.dict(os.environ, {"WAN_PAIR_CASE": "override", "RUSTC_WRAPPER": "unreviewed",
                                     "HF_TOKEN": "sensitive", "CARGO_ENCODED_RUSTFLAGS": "override"}, clear=True):
            env = pair.build_env(SimpleNamespace(backend="darwin-mlx"))
        self.assertEqual(env, {"CARGO_INCREMENTAL": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                               "MACOSX_DEPLOYMENT_TARGET": "26.2"})
        self.assertEqual(pair.LANES["candle-cuda"], ["self-hosted", "windows", "cuda", "real-weights"])
        self.assertEqual(len(pair.CASES["candle-cuda"]), 1)
        self.assertEqual(pair.fixture_directory(Path("work"), pair.CASES["mlx"][1]),
                         pair.fixture_directory(Path("work"), pair.CASES["mlx"][2]))


class WanEvidenceTests(unittest.TestCase):
    case = pair.CASES["candle-cuda"][0]
    spec = {"tensor_shape": [2], "outer_steps": 4, "trace": [{"event": "step", "current": i, "total": 4}
                                                              for i in (1, 2, 2, 3, 3, 4, 4)]}

    def setUp(self):
        self.raw = struct.pack("<ff", 0.125, -0.75)
        self.output = {"case": self.case, "tensor_shape": [2], "warmups": 1,
                       "samples": [{"elapsed_ns": 1000, "outer_steps": 4} for _ in range(5)],
                       "trace": self.spec["trace"], "allocator_peak_bytes": 64}
        self.policy = SimpleNamespace(deadline_seconds=90)

    def capture(self, output=None, raw=None):
        return evidence.capture(output or self.output, self.raw if raw is None else raw,
                                self.case, self.spec, pair.SETTINGS, self.policy)

    def test_numerical_and_performance_verdicts_remain_independent(self):
        base = dict(self.capture(), peaks={"process_footprint_bytes": 1000})
        head = deepcopy(base)
        self.assertTrue(evidence.compare(base, head, pair.SETTINGS)["accepted"])
        head["samples"][0]["elapsed_ns"] = 1050
        measured = evidence.compare(base, head, pair.SETTINGS)
        self.assertEqual(measured["parity"], "pass")
        self.assertEqual(measured["metrics"]["denoise_seconds"]["verdict"], "inconclusive")
        self.assertFalse(measured["accepted"])
        for sample in head["samples"]:
            sample["elapsed_ns"] = 1030
        head["peaks"]["process_footprint_bytes"] = 1021
        head["terminal"] = self.capture(raw=struct.pack("<ff", 0.125, -0.75001))["terminal"]
        measured = evidence.compare(base, head, pair.SETTINGS)
        self.assertEqual(measured["parity"], "fail")
        self.assertEqual(measured["metrics"]["seconds_per_outer_step"]["verdict"], "fail")
        self.assertEqual(measured["metrics"]["process_footprint_bytes"]["verdict"], "fail")

    def test_incomplete_or_nonfinite_observations_never_count(self):
        for raw in (b"", struct.pack("<ff", float("nan"), 1), struct.pack("<ff", float("inf"), 1),
                    struct.pack("<ff", 0.125, 0.125)):
            with self.assertRaises(ValueError):
                self.capture(raw=raw)
        for invalid in (dict(self.output, case=pair.CASES["mlx"][0]),
                        dict(self.output, trace=self.output["trace"][::-1]),
                        dict(self.output, samples=self.output["samples"][:4]),
                        dict(self.output, warmups=True), dict(self.output, result_path="/unexpected"),
                        dict(self.output, samples=[{"elapsed_ns": 10, "outer_steps": 3}] * 5)):
            with self.assertRaises(ValueError):
                self.capture(output=invalid)

    def test_body_device_and_cuda_high_water_are_independently_required(self):
        with TemporaryDirectory() as directory:
            policy = pair.read_policy(Path(directory), "policy.json", request("candle-cuda", pair.BASE)["policy"])
        observation = {"result": self.output, "terminal_le_f32_hex": self.raw.hex()}
        stdout = (f"MODEL_INFERENCE_DEVICE {pair.FQN} candle-cuda\n"
                  f"MODEL_INFERENCE_BODY_COMPLETED {pair.FQN}\n"
                  "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured\n")
        admission = runtime_guarded_admission(policy)
        admission.update(hostAvailableBytes=100 * pair.GIB, gpuAvailableBytes=50 * pair.GIB)
        guard = {"pid": 17, "returncode": 0, "elapsed_seconds": 1, "peak_host_bytes": 128, "peak_gpu_bytes": 64, "admission": admission}
        pool = {"pid": 17, "cudaDeviceUuid": policy.cuda_device_uuid, "poolUsedBytes": 16,
                "poolReservedBytes": 32, "poolUsedHighBytes": 32, "poolReservedHighBytes": 64}
        with patch.object(evidence, "case_spec", return_value=self.spec):
            result = pair.evaluate(observation, self.case, "candle-cuda", policy, pair.SETTINGS, stdout, guard, pool)
            self.assertEqual(result["body"]["body_completed"], 1)
            self.assertEqual(result["peaks"]["cuda_pool_reserved_high_bytes"], 64)
            for log, bad_pool in ((stdout.replace("candle-cuda", "candle-cpu"), pool),
                                  (stdout.replace("MODEL_INFERENCE_BODY_COMPLETED", "skipped"), pool),
                                  (stdout, dict(pool, pid=18)), (stdout, dict(pool, poolReservedHighBytes=63))):
                with self.assertRaises(ValueError):
                    pair.evaluate(observation, self.case, "candle-cuda", policy, pair.SETTINGS, log, guard, bad_pool)

    def test_upload_verifier_replays_raw_parity_and_detects_missing_or_tampered_evidence(self):
        backend = "candle-cuda"
        plan = {"backend": backend, "cases": [self.case], "base": pair.BASE, "head": "1" * 40,
                "base_tree": "2" * 40, "head_tree": "3" * 40, "request": request(backend, pair.BASE),
                "fixture_source_sha256": dict.fromkeys(pair.ARMS, "4" * 64)}
        uuid = plan["request"]["policy"]["cudaDeviceUuid"]
        with TemporaryDirectory() as directory, ExitStack() as patches:
            tmp = Path(directory)
            output = tmp / "evidence"
            output.mkdir()
            patches.enter_context(patch.dict(os.environ, RUNNER_TEMP=directory, GITHUB_RUN_ID="1", GITHUB_RUN_ATTEMPT="1"))
            patches.enter_context(patch.object(evidence, "case_spec", return_value=self.spec))
            receipt = {"protocol": "wan-curated-tier1-v1", **plan, "run_id": "1", "run_attempt": "1",
                       "accepted": True, "physical_claim": {"post_execution_verified": True, "device_uuid": uuid},
                       "hardware": {"uuid": uuid}, "builds": {}, "logs": {}, "observations": {},
                       "fixtures": {self.case: {"fixture_schema": 1, "depth": 1, "seed": 104729, "variant": "5b",
                                               "source": "example", "files": {"tensor.safetensors": "0" * 64}}}}
            policy = pair.read_policy(tmp, "policy.json", plan["request"]["policy"])
            admission = runtime_guarded_admission(policy)
            admission.update(hostAvailableBytes=40 * pair.GIB, gpuAvailableBytes=20 * pair.GIB)
            guard = {"pid": 17, "returncode": 0, "elapsed_seconds": 1, "peak_host_bytes": 128,
                     "peak_gpu_bytes": 64, "admission": admission}
            pool = {"pid": 17, "cudaDeviceUuid": uuid, "poolUsedBytes": 16,
                    "poolReservedBytes": 32, "poolUsedHighBytes": 32, "poolReservedHighBytes": 64}
            stdout = (f"MODEL_INFERENCE_DEVICE {pair.FQN} candle-cuda\nMODEL_INFERENCE_BODY_COMPLETED {pair.FQN}\n"
                      "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured\n")
            observed = {"result": self.output, "terminal_le_f32_hex": self.raw.hex()}
            result = pair.evaluate(observed, self.case, backend, policy, pair.SETTINGS, stdout, guard, pool)
            for arm in pair.ARMS:
                stem = f"{arm}-{self.case}"
                for suffix, contents in {"stdout.log": stdout, "stderr.log": "", "supervisor.json": json.dumps(guard),
                                         "cuda-pool.json": json.dumps(pool)}.items():
                    name = f"{stem}.{suffix}"
                    path = output / name
                    path.write_bytes(contents.encode())
                    receipt["logs"][name] = {"path": name, "sha256": pair.sha(path), "bytes": path.stat().st_size}
                receipt["observations"][stem] = {"observed": deepcopy(observed), "result": deepcopy(result)}
                receipt["builds"][arm] = {"revision": plan[arm], "tree": plan[arm + "_tree"],
                                          "fixture_source_sha256": plan["fixture_source_sha256"][arm],
                                          "binary_sha256": "5" * 64, "overlay_sha256": "6" * 64,
                                          "discovery": {"target": {"name": pair.case_row(backend)["target"]}},
                                          "build_command": pair.proof.command(pair.case_row(backend), discover=True)}
            receipt["comparisons"] = {self.case: evidence.compare(result, result, pair.SETTINGS)}
            args = SimpleNamespace(output=output)
            def verify():
                (output / "receipt.json").write_text(json.dumps(receipt))
                pair.verify(args, plan)
            verify()
            original = deepcopy(receipt)
            receipt["observations"][f"head-{self.case}"]["observed"]["terminal_le_f32_hex"] = struct.pack("<ff", 1, 3).hex()
            with self.assertRaisesRegex(ValueError, "raw observations"):
                verify()
            receipt = original
            (output / f"base-{self.case}.stdout.log").write_text("")
            with self.assertRaisesRegex(ValueError, "log identity"):
                verify()
