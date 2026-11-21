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

from scripts.ci import video_tier1_comparison as pair
from scripts.ci import video_tier1_evidence as evidence
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


def retained_cuda_claim():
    # The five actual admission rows and seals from Wan run 44893811394.
    uuid = "GPU-e4b79931-7be6-f216-460a-f5405cfafffe"
    profile = "non-isolated-correctness-headroom"
    samples = [{"sample_number": i, "selected_index_matches_admission": True, "predicates": {"unique_uuid": True},
                "selected_card": {"total_mib": 97887, "free_mib": free, "used_mib": used,
                                  "index": 1, "utilization_percent": 0}}
               for i, (free, used) in enumerate([(63284, 34155)] * 3 + [(63281, 34158)] * 2, 1)]
    return {"activity_profile": profile, "device_uuid": uuid, "physical_index": 1, "runner": "cuda-windows-2",
            "activity_policy": "non-isolated correctness; exact-card identity and policy cap plus reserve free "
                               "on a quiet card (memory.used spread/growth, utilization level/spread bounded); "
                               "foreign contexts are diagnostic only",
            "isolated_benchmark_claim": False, "lease_released": True, "post_execution_verified": True,
            "mechanism": "existing PUBLIC/SceneWorks/gpu-locks UUID lock",
            "admission_baseline": {"uuid": uuid, "admitted_total_mib": 97887,
                                   "process_identities": [], "reserved_baseline_mib": 41818},
            "admission_utilization_percent": [0] * 5,
            "admission_background_activity": {"activity_profile": profile, "status": "retained-existing-samples",
                                              "sample_count": 5, "samples": samples}}


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
            (root / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.96.0"\n')
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            git("checkout", "-qb", "unrelated")
            git("commit", "--allow-empty", "-qm", "unrelated")
            other = git("rev-parse", "HEAD")
            git("checkout", "-q", "--detach", base)
            bundle = root / "scripts/ci/wan_pair"
            bundle.mkdir(parents=True)
            (bundle / "mlx_probe.rs.in").write_bytes(b"test\n@WAN_PAIR_COMMON@\n@WAN_MLX_SWAP_ARM@")
            (bundle / "common.rs.in").write_bytes(b"common")
            (bundle / "fixture.py").write_text("# Fake metadata-only fixture")
            (bundle / "headers.json").write_text(json.dumps({v: {"source": "pinned", "geometry": {"model.safetensors": {}}} for v in pair.VARIANTS.values()}))
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
                self.assertEqual(record["fixture_source_sha256"], hashlib.sha256(b"test\ncommon\n" + arm.encode()).hexdigest())
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
        self.assertIn("mlx-gen/test-support", pair.case_row("mlx")["cargo_features"])
        self.assertEqual(pair.LANES["candle-cuda"], ["self-hosted", "windows", "cuda", "real-weights"])
        self.assertEqual(len(pair.CASES["candle-cuda"]), 1)
        self.assertEqual(pair.fixture_directory(Path("work"), pair.CASES["mlx"][1]),
                         pair.fixture_directory(Path("work"), pair.CASES["mlx"][2]))


class WanEvidenceTests(unittest.TestCase):
    case = pair.CASES["candle-cuda"][0]
    spec = {"settings": {"seed": 104729, "sampler": "heun"}, "tensor_shape": [2], "outer_steps": 4, "trace": [{"event": "step", "current": i, "total": 4}
                                                              for i in (1, 2, 2, 3, 3, 4, 4)]}

    def setUp(self):
        self.raw = struct.pack("<ff", 0.125, -0.75)
        self.output = {"case": self.case, "tensor_shape": [2], "warmups": 1,
                       "samples": [{"elapsed_ns": 1000, "outer_steps": 4}], "settings": self.spec["settings"],
                       "trace": self.spec["trace"], "allocator_peak_bytes": 64}
        self.policy = SimpleNamespace(deadline_seconds=90)

    def capture(self, output=None, raw=None):
        return evidence.capture(output or self.output, self.raw, self.raw if raw is None else raw,
                                self.case, self.spec, pair.SETTINGS, self.policy)

    def test_numerical_and_performance_verdicts_remain_independent(self):
        base = [deepcopy(dict(self.capture(), peaks={"process_footprint_bytes": 1000})) for _ in range(5)]
        head = deepcopy(base)
        self.assertTrue(evidence.compare(base, head, pair.SETTINGS)["accepted"])
        head[0]["samples"][0]["elapsed_ns"] = 1050
        measured = evidence.compare(base, head, pair.SETTINGS)
        self.assertEqual(measured["parity"], "pass")
        self.assertEqual(measured["metrics"]["denoise_seconds"]["verdict"], "inconclusive")
        for row in head:
            row["samples"][0]["elapsed_ns"] = 1030
        head[0]["peaks"]["process_footprint_bytes"] = 1021
        head[0]["terminal"] = self.capture(raw=struct.pack("<ff", 0.125, -0.75001))["terminal"]
        measured = evidence.compare(base, head, pair.SETTINGS)
        self.assertEqual(measured["parity"], "fail")
        self.assertEqual(measured["metrics"]["seconds_per_outer_step"]["verdict"], "fail")
        self.assertEqual(measured["metrics"]["process_footprint_bytes"]["verdict"], "fail")
        head[1]["initial"] = head[0]["terminal"]
        self.assertEqual(evidence.compare(base, head, pair.SETTINGS)["parity"], "inconclusive")

    def test_incomplete_or_nonfinite_observations_never_count(self):
        for raw in (b"", struct.pack("<ff", float("nan"), 1), struct.pack("<ff", float("inf"), 1),
                    struct.pack("<ff", 0.125, 0.125)):
            with self.assertRaises(ValueError):
                self.capture(raw=raw)
        for invalid in (dict(self.output, case=pair.CASES["mlx"][0]),
                        dict(self.output, trace=self.output["trace"][::-1]),
                        dict(self.output, samples=self.output["samples"] * 5),
                        dict(self.output, warmups=True), dict(self.output, result_path="/unexpected"),
                        dict(self.output, samples=[{"elapsed_ns": 10, "outer_steps": 3}] * 5)):
            with self.assertRaises(ValueError):
                self.capture(output=invalid)

    def test_body_device_and_cuda_high_water_are_independently_required(self):
        with TemporaryDirectory() as directory:
            policy = pair.read_policy(Path(directory), "policy.json", request("candle-cuda", pair.BASE)["policy"])
        observation = {"result": self.output, "initial_le_f32_hex": self.raw.hex(), "terminal_le_f32_hex": self.raw.hex()}
        stdout = (f"MODEL_INFERENCE_DEVICE {pair.FQN} candle-cuda\n"
                  f"MODEL_INFERENCE_BODY_COMPLETED {pair.FQN}\n"
                  "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured\n")
        admission = runtime_guarded_admission(policy)
        admission.update(hostAvailableBytes=100 * pair.GIB, gpuAvailableBytes=50 * pair.GIB)
        guard = {"pid": 17, "returncode": 0, "elapsed_seconds": 1, "peak_host_bytes": 128, "peak_gpu_bytes": 64, "admission": admission, "cuda_attribution": {
                        "pid_peak_bytes": 64, "unknown_pid_polls": 0, "global_upper_bytes": 0, "pool_fallback_peak_bytes": 0}}
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

            from scripts.ci import wddm_process_memory as wddm
            adapter = {"cuda_uuid": policy.cuda_device_uuid, "luid": "luid_0x00000000_0x00000001",
                       "physical_node": 0, "node_mask": 1, "counter": wddm.COUNTER}
            guard.update(peak_gpu_bytes=None, cuda_attribution={
                "pid_peak_bytes": 0, "unknown_pid_polls": 2, "global_upper_bytes": 180,
                "pool_fallback_peak_bytes": 0, "wddm_dedicated": {"pid": 17, "created_filetime": 123,
                    "adapter": adapter, "samples": [[130, 111], [140, 120]], "error": None, "missing_active_polls": 0}})
            result = pair.evaluate(observation, self.case, "candle-cuda", policy, pair.SETTINGS, stdout, guard, pool, adapter)
            self.assertEqual(result["cuda_process"]["dedicated"]["sampled_peak_bytes"], 120)
            self.assertEqual(result["peaks"]["cuda_pool_reserved_high_bytes"], 64)
            for invalid in (dict(pool, pid=18), dict(pool, poolReservedHighBytes=65)):
                with self.assertRaises(ValueError):
                    pair.evaluate(observation, self.case, "candle-cuda", policy, pair.SETTINGS, stdout, guard, invalid, adapter)
            guard["cuda_attribution"]["wddm_dedicated"]["missing_active_polls"] = 1
            with self.assertRaisesRegex(ValueError, "missing attributable GPU sample"):
                pair.evaluate(observation, self.case, "candle-cuda", policy, pair.SETTINGS, stdout, guard, pool, adapter)

    def test_observed_shared_headroom_rejects_wrong_policy_unlocked_or_unstable_card(self):
        claim = retained_cuda_claim()
        hardware = {"name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "uuid": claim["device_uuid"],
                    "index": "1", "memory.total": "97887", "driver_version": "596.36"}
        policy = SimpleNamespace(cuda_device_uuid=hardware["uuid"], child_gpu_cap_bytes=12 * pair.GIB,
                                 gpu_free_reserve_bytes=6 * pair.GIB)
        pair.verify_cuda_claim(claim, hardware, policy)
        for alter in (
            lambda c: c.update(activity_profile="strict-zero-utilization", idle_policy="strict"),
            lambda c: c.update(lease_released=False),
            lambda c: c.update(post_execution_verified=False),
            lambda c: c.update(isolated_benchmark_claim=True),
            lambda c: c.update(physical_index=0),
            lambda c: c.update(device_uuid="GPU-other"),
            lambda c: c["admission_background_activity"]["samples"].pop(),
            lambda c: c["admission_background_activity"]["samples"][1]["predicates"].update(unique_uuid=False),
            lambda c: c["admission_background_activity"]["samples"][1]["selected_card"].update(free_mib=200),
            lambda c: c["admission_background_activity"]["samples"][1]["selected_card"].update(used_mib=1024),
            lambda c: c["admission_background_activity"]["samples"][1]["selected_card"].update(utilization_percent=100),
        ):
            altered = deepcopy(claim)
            alter(altered)
            with self.assertRaises(ValueError):
                pair.verify_cuda_claim(altered, hardware, policy)

    def test_upload_verifier_replays_raw_parity_and_detects_missing_or_tampered_evidence(self):
        backend = "candle-cuda"
        plan = {"backend": backend, "cases": [self.case], "base": pair.BASE, "head": "1" * 40,
                "base_tree": "2" * 40, "head_tree": "3" * 40, "request": request(backend, pair.BASE),
                "fixture_source_sha256": dict.fromkeys(pair.ARMS, "4" * 64),
                "fixture_contracts": {self.case: {"variant":"5b", "geometry_sha256":"1"*64,
                                    "source":{"snapshot":"v"}, "files":["tensor.safetensors"]}},
                "toolchain_contracts": {arm: {"channel":"1.96.0","declaration_sha256":"7"*64} for arm in pair.ARMS}}
        from scripts.ci import wddm_process_memory
        claim = retained_cuda_claim()
        uuid = plan["request"]["policy"]["cudaDeviceUuid"] = claim["device_uuid"]
        adapter = {"cuda_uuid": uuid, "luid": "luid_0x00000000_0x00020c10", "node_mask": 1,
                   "physical_node": 0, "counter": wddm_process_memory.COUNTER}
        with TemporaryDirectory() as directory, ExitStack() as patches:
            tmp = Path(directory)
            output = tmp / "evidence"
            output.mkdir()
            patches.enter_context(patch.dict(os.environ, RUNNER_TEMP=directory, RUNNER_NAME=claim["runner"], GITHUB_RUN_ID="1", GITHUB_RUN_ATTEMPT="1"))
            patches.enter_context(patch.object(evidence, "case_spec", return_value=self.spec))
            hw = {"uuid":uuid, "index":"1","name":"CUDA card","memory.total":"97887","driver_version":"1"}
            tc = {"channel":"1.96.0","declaration_sha256":"7"*64,"rustc_version":"rustc selected",
                  "cargo_version":"cargo selected","binaries":{"rustc":"selected", "cargo":"selected"}}
            patches.enter_context(patch.object(pair.proof, "verified_toolchain", return_value=tc))
            patches.enter_context(patch.object(pair.paired, "hardware", return_value=hw))
            patches.enter_context(patch.object(pair, "cuda_adapter", return_value=adapter))
            receipt = {"dedicated_adapter": adapter, "protocol": "wan-curated-tier1-v2", **plan, "run_id": "1", "run_attempt": "1",
                       "accepted": True, "host":pair.platform.platform(), "runner_name":claim["runner"],
                       "execution_order": [], "physical_claim": claim,
                       "hardware": deepcopy(hw), "builds": {}, "logs": {}, "observations": {},
                       "fixtures": {self.case: {"fixture_schema": 1, "depth": 1, "seed": 104729, "variant": "5b",
                                               "source": {"snapshot":"v"}, "geometry_sha256":"1"*64, "files": {"tensor.safetensors": "0" * 64}}}}
            policy = pair.read_policy(tmp, "policy.json", plan["request"]["policy"])
            admission = runtime_guarded_admission(policy)
            admission.update(hostAvailableBytes=40 * pair.GIB, gpuAvailableBytes=20 * pair.GIB)
            guard = {"pid": 17, "returncode": 0, "elapsed_seconds": 1, "peak_host_bytes": 128,
                     "peak_gpu_bytes": 64, "admission": admission, "cuda_attribution": {
                        "pid_peak_bytes": 64, "unknown_pid_polls": 0, "global_upper_bytes": 0, "pool_fallback_peak_bytes": 0}}
            guard["cuda_attribution"]["wddm_dedicated"] = {
                "adapter": adapter, "pid": 17, "created_filetime": 12345,
                "samples": [[14000, 64], [15000, 72]], "missing_active_polls": 0, "error": None,
            }
            pool = {"pid": 17, "cudaDeviceUuid": uuid, "poolUsedBytes": 16,
                    "poolReservedBytes": 32, "poolUsedHighBytes": 32, "poolReservedHighBytes": 64}
            stdout = (f"MODEL_INFERENCE_DEVICE {pair.FQN} candle-cuda\nMODEL_INFERENCE_BODY_COMPLETED {pair.FQN}\n"
                      "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured\n")
            observed = {"result": self.output, "initial_le_f32_hex": self.raw.hex(), "terminal_le_f32_hex": self.raw.hex()}
            result = pair.evaluate(observed, self.case, backend, policy, pair.SETTINGS, stdout, guard, pool, adapter)
            for counter, (_, arm, stem) in enumerate(pair.measurements(self.case)):
                receipt["execution_order"].append({"phase":stem,"start_ns":counter*100+1,"end_ns":counter*100+2})
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
                                          "toolchain": deepcopy(tc), "discovery": {"target": pair.case_row(backend)["cargo_target_metadata"],
                                              "features":["default","model-inference-proof","cuda"],"model_bodies_executed":0,
                                              "package_id":"path+file:///src/candle-gen-wan#0.0.0",
                                              "executable":f"C:/src/{arm}-target/debug/deps/candle_gen_wan-abcd.exe"},
                                          "build_command": pair.proof.command(pair.case_row(backend), discover=True)}
            receipt["comparisons"] = {self.case: evidence.compare([result]*5, [result]*5, pair.SETTINGS)}
            args = SimpleNamespace(output=output)
            def verify():
                (output / "receipt.json").write_text(json.dumps(receipt))
                # Re-bind here to exercise semantic checks independently of producer commitment.
                with patch.dict(os.environ, WAN_PAIR_RECEIPT_SHA256=pair.sha(output / "receipt.json")):
                    pair.verify(args, plan)
            verify()
            original = deepcopy(receipt)
            with patch.dict(os.environ, WAN_PAIR_RECEIPT_SHA256=pair.sha(output / "receipt.json")):
                changed = deepcopy(original)
                changed["fixtures"][self.case]["files"]["tensor.safetensors"] = "1"*64
                (output / "receipt.json").write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "supervisor receipt"):
                    pair.verify(args, plan)
            for change in (
                lambda r: r["fixtures"][self.case].update(variant="wrong"),
                lambda r: r["fixtures"][self.case].update(source={"snapshot":"forged"}),
                lambda r: r["fixtures"][self.case].update(files={"fake.safetensors":"3"*64}),
                lambda r: r["fixtures"][self.case].update(geometry_sha256="0"*64),
                lambda r: r["builds"]["head"].update(toolchain={}),
                lambda r: r["builds"]["head"]["discovery"].update(features=["cuda"]),
                lambda r: r["builds"]["head"]["discovery"]["target"].update(kind=["bin"]),
                lambda r: r.update(hardware={"uuid":uuid}),
                lambda r: r.update(dedicated_adapter=dict(adapter, node_mask=2)),
                lambda r: r.pop("dedicated_adapter"),
                lambda r: r.update(physical_claim={"post_execution_verified":True,"runner":"runner"}),
                lambda r: r["execution_order"].reverse(),
                lambda r: r["execution_order"][1].update(start_ns=1),
            ):
                receipt = deepcopy(original)
                change(receipt)
                with self.assertRaises((ValueError,KeyError)):
                    verify()
            receipt = deepcopy(original)
            receipt["observations"][f"{self.case}-p1-head"]["observed"]["terminal_le_f32_hex"] = struct.pack("<ff", 1, 3).hex()
            with self.assertRaisesRegex(ValueError, "raw observations"):
                verify()
            receipt = original
            (output / f"{self.case}-p1-base.stdout.log").write_text("")
            with self.assertRaisesRegex(ValueError, "log identity"):
                verify()

    def test_wddm_intervals_and_native_pid_process_regression(self):
        def arm(pid, unknown, upper, fallback):
            row = dict(self.capture(), peaks={"pool":64})
            row["cuda_process"] = evidence.cuda_process({"peak_gpu_bytes":max(pid,fallback), "cuda_attribution":{
                "pid_peak_bytes":pid,"unknown_pid_polls":unknown,"global_upper_bytes":upper,
                "pool_fallback_peak_bytes":fallback}}, 64)
            return [deepcopy(row) for _ in range(5)]
        for base, head, verdict in ((arm(100,0,0,0),arm(110,0,0,0),"fail"),
                                    (arm(100,0,0,0),arm(100,1,121,64),"inconclusive"),
                                    (arm(64,1,80,64),arm(64,1,80,64),"inconclusive"),
                                    (arm(64,1,65,64),arm(64,1,65,64),"pass")):
            self.assertEqual(evidence.compare(base, head, pair.SETTINGS)["metrics"]["cuda_process_bytes"]["verdict"], verdict)

    def test_artifact_root_traversal_and_link_rejected_before_open(self):
        with TemporaryDirectory() as directory, TemporaryDirectory() as outside:
            root = Path(directory)
            link = root / "linked"
            link.symlink_to(outside, target_is_directory=True)
            with patch.dict(os.environ, RUNNER_TEMP=directory):
                for invalid in (root, root / "child" / ".." / "evidence", root / ".." / Path(outside).name, link):
                    with self.assertRaises(ValueError):
                        pair.verify(SimpleNamespace(output=invalid), {})
