"""CPU-only selected-card tests for the owner's physical GPU1 shared window."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ci"))
sys.path.insert(0, str(ROOT / "scripts/tests"))
import test_yue2_empty_gpu0 as empty
import yue2_cuda_idle_context as idle
import yue2_precision_proof as proof

SHA = "a" * 40
LUID = "luid_0x00000000_0x0001f78f"
ACTOR = "    1        888   C+G      -      -      -      -      -      -      0      0    desktop.exe"


def make_shared_probe(root: Path) -> None:
    empty.make_probe(root)
    for path in root.glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if path.name == "manifest.json":
            value["selectedGpuIndex"] = 1
        text = json.dumps(value).replace(empty.UUID, idle.SHARED_GPU1_UUID)
        text = text.replace(empty.PCI, "0000:C1:00.0")
        text = text.replace(empty.LUID, LUID)
        text = text.replace("8C-0B-02-00-00-00-00-00", "8F-F7-01-00-00-00-00-00")
        value = json.loads(text)
        if path.name.startswith(("gpu-sample-", "driver-mode-", "gpu-before-", "gpu-after-")):
            value["output"] = [line.replace("0, ", "1, ", 1) for line in value["output"]]
        if path.name.startswith("gpu-sample-"):
            value["output"][0] = value["output"][0].replace(", 0, 97438, 0, 0",
                                                             ", 553, 96886, 25, 18")
        if path.name.startswith(("pmon-1-", "pmon-0-final")):
            value["output"] = empty.PMON_EMPTY[:2] + [ACTOR]
        if path.name.startswith("compute-apps-1-"):
            value["output"] = ["888, desktop.exe, [N/A]"]
        if path.name.startswith(("gpu-before-", "gpu-after-")):
            value["output"][0] = value["output"][0].replace(", 0, 0", ", 553, 25")
        path.write_text(json.dumps(value), encoding="utf-8")


def set_window(root: Path, used: list[int], util: list[int], free: list[int] | None = None) -> None:
    """Rewrite the five chronological selected-card samples: three gpu-samples, then before/after."""
    free = free or [97439 - value for value in used[:3]]
    for index in range(3):
        path = root / f"gpu-sample-{index}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        fields = [field.strip() for field in row["output"][0].split(",")]
        fields[6], fields[7], fields[8] = str(used[index]), str(free[index]), str(util[index])
        row["output"][0] = ", ".join(fields)
        path.write_text(json.dumps(row), encoding="utf-8")
    for index, name in ((3, "gpu-before-cuda-properties"), (4, "gpu-after-cuda-properties")):
        path = root / f"{name}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        fields = [field.strip() for field in row["output"][0].split(",")]
        fields[3], fields[4] = str(used[index]), str(util[index])
        row["output"][0] = ", ".join(fields)
        path.write_text(json.dumps(row), encoding="utf-8")


class FakeLock:
    def __init__(self, *args, **kwargs):
        self.pauses = []

    def acquire(self, deadline, notice):
        pass

    def release(self):
        pass

    def pause(self, seconds, deadline, notice):
        self.pauses.append(seconds)


class SharedGpu1Tests(unittest.TestCase):
    def verify(self, root: Path, *, admission: bool = True) -> dict:
        with patch.dict("os.environ", {"EXPECTED_ENGINE_SHA": SHA, "GITHUB_SHA": SHA,
                                    "RUNNER_NAME": "cuda-windows"}):
            return idle._shared_gpu1_summary(root, admission=admission)

    def test_physical1_maps_to_logical0_and_foreign_actors_are_observed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_shared_probe(root)
            self.assertEqual(len(list(root.glob("*.json"))), 29)
            # Activity on physical GPU0 is not admission activity on GPU1.
            for name in ("gpu-sample-0", "gpu-sample-1", "gpu-sample-2"):
                path = root / f"{name}.json"
                row = json.loads(path.read_text(encoding="utf-8"))
                row["output"].append("0, GPU-other, 00000000:21:00.0, RTX, 596, 97887, 84816, 12623, 99, 99")
                path.write_text(json.dumps(row), encoding="utf-8")
            for index in range(3):
                pmon = root / f"pmon-0-{index}.json"
                row = json.loads(pmon.read_text(encoding="utf-8"))
                row["output"].append("0 4000 C 99 90 0 0 0 0 84816 0 owner-llm")
                pmon.write_text(json.dumps(row), encoding="utf-8")
            result = self.verify(root)
            self.assertEqual((result["physicalIndex"], result["cudaOrdinal"], result["luid"]),
                             (1, 0, LUID))
            self.assertEqual(result["observedActors"][0], [{"pid": 888, "type": "C+G"}])
            # A coexisting actor may change residency/activity during our child; only
            # admission judges the quiet window, never the post-child release census.
            for index in range(3):
                path = root / f"gpu-sample-{index}.json"
                row = json.loads(path.read_text(encoding="utf-8"))
                row["output"][0] = row["output"][0].replace(", 553, 96886, 25, 18",
                                                              ", 70000, 27000, 80, 65")
                path.write_text(json.dumps(row), encoding="utf-8")
            self.verify(root, admission=False)
            with self.assertRaisesRegex(idle.SharedGpu1NotAdmissible, r"utilization 80% \(> 30%\)"):
                self.verify(root, admission=True)

    def test_busy_but_stable_foreign_compute_process_is_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_shared_probe(root)
            # A foreign compute process holds 70 GiB and runs at a steady 20%: present, not refused.
            set_window(root, [70000, 70040, 70020, 70060, 70050], [20, 22, 21, 20, 23])
            for name in ("pmon-1-0", "pmon-1-1", "pmon-1-2", "pmon-0-final"):
                path = root / f"{name}.json"
                row = json.loads(path.read_text(encoding="utf-8"))
                row["output"].append("    1       4000     C     20      9      -      -      -      -  70000      0    owner-llm")
                path.write_text(json.dumps(row), encoding="utf-8")
            for index in range(3):
                path = root / f"windows-counters-{index}.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                for row in value["counters"]:
                    if row["counter"] == r"\GPU Engine(*)\Utilization Percentage":
                        row["samples"][0]["cookedValue"] = 90  # Windows engine counter is evidence only.
                path.write_text(json.dumps(value), encoding="utf-8")
            result = self.verify(root)
            self.assertIn({"pid": 4000, "type": "C"}, result["observedActors"][0])
            self.assertEqual(result["windowUsedMiB"], [70000, 70040, 70020, 70060, 70050])
            self.assertEqual(result["baselineUsedMiB"], 70050)

    def test_growth_instability_and_insufficient_free_refuse_with_reason(self):
        cases = (
            ("growth", [553, 590, 625, 660, 700], [25] * 5, None, r"memory\.used grew 147 MiB \(> 128\)"),
            ("spread", [553, 900, 553, 553, 553], [25] * 5, None, r"memory\.used moved 347 MiB \(> 256\)"),
            ("busy", [553] * 5, [25, 25, 31, 25, 25], None, r"utilization 31% \(> 30%\)"),
            ("jitter", [553] * 5, [5, 25, 5, 5, 5], None, r"utilization moved 20 points \(> 15\)"),
            ("free", [553] * 5, [25] * 5, [96886, 16000, 96886], r"16000 MiB free < 16384 MiB required"),
        )
        for name, used, util, free, reason in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_shared_probe(root)
                set_window(root, used, util, free)
                self.verify(root, admission=False)
                with self.assertRaisesRegex(idle.SharedGpu1NotAdmissible, reason):
                    self.verify(root)

    def test_not_admissible_census_is_retryable_but_identity_refusal_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_shared_probe(root)
            set_window(root, [553, 900, 553, 553, 553], [25] * 5)
            def probe(command, **kwargs):
                output = Path(command[command.index("-OutputDirectory") + 1])
                for file in root.iterdir():
                    (output / file.name).write_bytes(file.read_bytes())
                return type("Completed", (), {"returncode": 0, "stderr": ""})()
            with patch.object(idle, "check_shared_gpu1_dispatch"), \
                 patch.dict("os.environ", {"EXPECTED_ENGINE_SHA": SHA, "GITHUB_SHA": SHA,
                                           "RUNNER_NAME": "cuda-windows"}), \
                 patch.object(idle.subprocess, "run", side_effect=probe):
                raw, verified = idle.census_shared_gpu1(admission=True)
                self.assertFalse(verified)
                self.assertEqual(proof.retryable_admission(raw),
                                 "selected GPU1 not admissible: memory.used moved 347 MiB (> 256)")
                (root / "gpu-sample-1.json").write_text(
                    (root / "gpu-sample-1.json").read_text(encoding="utf-8").replace(
                        idle.SHARED_GPU1_UUID, "GPU-other"), encoding="utf-8")
                raw, verified = idle.census_shared_gpu1(admission=True)
                self.assertFalse(verified)
                self.assertIsNone(proof.retryable_admission(raw))

    def test_admission_waits_with_logged_reason_and_releases_the_lock(self):
        unstable = json.dumps({"retryableAdmission": True,
                               "refusal": "selected GPU1 not admissible: memory.used grew 300 MiB (> 128)"})
        census = [(unstable, ["physical GPU1 shared-device proof refused"])] * 2 + [("admitted", [])]
        lock = FakeLock()
        stderr = io.StringIO()
        with patch.object(proof, "cuda_physical_census", side_effect=census) as physical, \
             patch("sys.stderr", stderr):
            self.assertEqual(proof.admitted_cuda_census(time.monotonic() + 600, lock), ("admitted", []))
        self.assertEqual(physical.call_count, 3)
        self.assertEqual(lock.pauses, [proof.CUDA_ADMISSION_POLL_SECONDS] * 2)
        # One notice per minute, naming the failed condition and that foreign actors never refuse.
        self.assertEqual(stderr.getvalue().count("CUDA admission waiting:"), 1)
        self.assertIn("memory.used grew 300 MiB (> 128)", stderr.getvalue())
        self.assertIn("never a refusal", stderr.getvalue())
        # Identity/evidence refusals are returned at once; an expired deadline stops the wait.
        refused = json.dumps({"refusal": "selected physical GPU1 UUID/PCI changed"})
        with patch.object(proof, "cuda_physical_census", return_value=(refused, ["refused"])) as physical:
            self.assertEqual(proof.admitted_cuda_census(time.monotonic() + 600, lock)[1], ["refused"])
            physical.assert_called_once_with()
        with patch.object(proof, "cuda_physical_census", return_value=(unstable, ["refused"])) as physical:
            self.assertEqual(proof.admitted_cuda_census(time.monotonic() - 1, lock)[1], ["refused"])
            physical.assert_called_once_with()

    def test_host_global_uuid_lock_waits_with_notice_then_holds_and_releases(self):
        class Locker:
            LK_NBLCK, LK_UNLCK = 2, 0
            def __init__(self):
                self.calls = []
                self.refusals = 2
            def locking(self, fd, mode, size):
                self.calls.append(mode)
                if mode == self.LK_NBLCK and self.refusals:
                    self.refusals -= 1
                    raise OSError("locked")
        with tempfile.TemporaryDirectory() as directory:
            locker = Locker()
            sleeps = []
            lock = idle.SharedGpu1Lock(Path(directory), locker, sleep=sleeps.append)
            stderr = io.StringIO()
            with patch("sys.stderr", stderr):
                lock.acquire(time.monotonic() + 600, {})
            self.assertEqual((Path(directory) / f"{idle.SHARED_GPU1_UUID}.lock").read_bytes(), b"0")
            self.assertEqual(locker.calls, [2, 2, 2])
            self.assertEqual(len(sleeps), 2)
            self.assertIn("UUID lock held by another CI job", stderr.getvalue())
            lock.pause(3, time.monotonic() + 600, {})
            self.assertEqual(locker.calls[3:], [0, 2])
            lock.release()
            lock.release()
            self.assertEqual(locker.calls[-1], 0)
            locker.refusals = 1
            with self.assertRaisesRegex(RuntimeError, "UUID lock still held"):
                lock.acquire(time.monotonic() - 1, {})

    def test_child_memory_is_peak_minus_locked_baseline_and_cotenant_change_contaminates(self):
        def census(used, actors):
            return json.dumps({"validatedDevice": {"baselineUsedMiB": used,
                               "observedActors": [actors] * 4}})
        samples = [{"raw": f"t, 1, {value}, 1000", "physical_gpu_index": 1}
                   for value in (70100, 82000, 75000)]
        foreign = [{"pid": 4000, "type": "C"}, {"pid": 888, "type": "C+G"}]
        own = [{"pid": 9001, "type": "C"}]
        clean = proof.child_memory_measurement(census(70000, foreign), census(70100, foreign + own),
                                               samples, {9001})
        self.assertEqual((clean["baseline_used_mib"], clean["peak_used_mib"],
                          clean["peak_minus_baseline_mib"], clean["contaminated"]),
                         (70000, 82000, 12000, False))
        moved = proof.child_memory_measurement(census(70000, foreign), census(74000, foreign), samples, set())
        self.assertTrue(moved["contaminated"])
        self.assertIn("baseline moved +4000 MiB", moved["contamination"][0])
        changed = proof.child_memory_measurement(census(70000, foreign), census(70000, foreign[1:]),
                                                 samples, set())
        self.assertEqual(changed["contamination"], ["foreign compute actors changed: [4000] -> []"])
        # A desktop C+G window opening or closing is not a CUDA co-tenant change.
        desktop = proof.child_memory_measurement(census(70000, foreign), census(70000, foreign[:1]),
                                                 samples, set())
        self.assertFalse(desktop["contaminated"])
        self.assertEqual(proof.headline_peak_minus_baseline(
            [dict(clean, label="quant", peak_minus_baseline_mib=300),
             dict(moved, label="vae", peak_minus_baseline_mib=99000),
             dict(clean, label="precision")]), 12000)
        self.assertIsNone(proof.headline_peak_minus_baseline([moved, changed]))
        missing = proof.child_memory_measurement(census(70000, foreign), None, samples, set())
        self.assertTrue(missing["contaminated"])
        self.assertEqual(missing["peak_minus_baseline_mib"], 12000)

    def test_selected_identity_capacity_and_telemetry_mutations_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_shared_probe(root)
            mutations = (
                ("cuda-adapter-map", lambda row: row["devices"][0].__setitem__("pciBusId", "0000:21:00.0")),
                ("cuda-adapter-map", lambda row: row["devices"][0].__setitem__("ordinal", 1)),
                ("gpu-sample-1", lambda row: row["output"].__setitem__(
                    0, row["output"][0].replace(idle.SHARED_GPU1_UUID, "GPU-other"))),
                ("gpu-sample-1", lambda row: row["output"].__setitem__(
                    0, row["output"][0].replace(", 553, 96886,", ", 48944, 48495,"))),
                ("gpu-sample-1", lambda row: row["output"].__setitem__(
                    0, row["output"][0].replace(", 553, 96886,", ", 553, 16000,"))),
                ("gpu-sample-1", lambda row: row["output"].__setitem__(
                    0, row["output"][0].replace(", 25, 18", ", 51, 18"))),
                ("pmon-1-1", lambda row: row["output"].append("1 999 X 10 0 0 0 0 0 0 0 unknown")),
                ("pmon-0-final", lambda row: row["output"].append("1 999 C+G 101 0 0 0 0 0 0 0 unknown")),
                ("windows-counters-1", lambda row: row["counters"][0].pop("samples")),
                ("windows-counters-1", lambda row: row["counters"][0]["samples"][0].__setitem__(
                    "status", "unavailable")),
            )
            for name, mutate in mutations:
                with self.subTest(name=name, mutate=mutate):
                    path = root / f"{name}.json"
                    original = path.read_bytes()
                    row = json.loads(original)
                    mutate(row)
                    path.write_text(json.dumps(row), encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        self.verify(root)
                    path.write_bytes(original)
            self.verify(root)

    def test_pdh_full_paths_distinguish_duplicate_short_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_shared_probe(root)
            for index in range(3):
                path = root / f"windows-counters-{index}.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                for family, metric, amount in (("gpu engine", "utilization percentage", 0),
                                               ("gpu process memory", "dedicated usage", 270336)):
                    row = next(row for row in value["counters"]
                               if row["counter"].lower() ==
                               f"\\{family}(*)\\{metric}")
                    short = row["samples"][0]["instance"]
                    row["samples"][0]["path"] = f"\\\\host\\{family}({short})\\{metric}"
                    duplicate = copy.deepcopy(row["samples"][0])
                    duplicate["path"] = f"\\\\host\\{family}({short}#1)\\{metric}"
                    duplicate["cookedValue"] = amount
                    row["samples"].append(duplicate)
                path.write_text(json.dumps(value), encoding="utf-8")
            self.verify(root)
            path = root / "windows-counters-1.json"
            original = path.read_bytes()
            for mutate in (
                lambda row: row["samples"][-1].__setitem__("path", row["samples"][0]["path"]),
                lambda row: row["samples"][-1].__setitem__(
                    "path", row["samples"][-1]["path"].replace("gpu engine", "gpu process memory")),
                lambda row: row["samples"][-1].__setitem__(
                    "path", row["samples"][-1]["path"].replace("#1", "_other#1")),
                lambda row: row["samples"][-1].pop("path"),
                lambda row: row["samples"][-1].__setitem__("status", "unavailable"),
            ):
                with self.subTest(mutate=mutate):
                    value = json.loads(original)
                    row = next(row for row in value["counters"]
                               if row["counter"] == r"\GPU Engine(*)\Utilization Percentage")
                    mutate(row)
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        self.verify(root)
            path.write_bytes(original)
            self.verify(root)

    def test_dispatch_and_sampler_bind_physical1_without_old_receipt(self):
        env = {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "YUE2_IDLE_CONTEXT_RUN_ID": "", "YUE2_CUDA_SCHEDULING_MODE": "shared-host",
               "GITHUB_REPOSITORY": "SceneWorks/inference", "GITHUB_JOB": "cuda",
               "GITHUB_RUN_ATTEMPT": "1", "RUNNER_NAME": "cuda-windows",
               "EXPECTED_ENGINE_SHA": SHA, "EXPECTED_CONTROL_SHA": SHA, "GITHUB_SHA": SHA}
        with patch("os.name", "nt"), patch.dict("os.environ", env):
            idle.check_shared_gpu1_dispatch()
            with patch.dict("os.environ", {"YUE2_CUDA_SCHEDULING_MODE": "shared-gpu1"}):
                idle.check_shared_gpu1_dispatch()
            with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
                with self.assertRaises(RuntimeError):
                    idle.check_shared_gpu1_dispatch()
            with patch.dict("os.environ", {"YUE2_IDLE_CONTEXT_RUN_ID": "old"}):
                with self.assertRaises(RuntimeError):
                    idle.check_shared_gpu1_dispatch()
            completed = type("Completed", (), {"returncode": 0,
                "stdout": "2026/10/04 20:00:00, 1, 12000, 85000\n", "stderr": ""})()
            with patch.object(proof.subprocess, "run", return_value=completed) as smi:
                sample = proof.sample_cuda()
            self.assertEqual(sample["physical_gpu_index"], 1)
            self.assertEqual(smi.call_args.args[0][2], "1")

    def test_workflow_cuda_mode_argument_reaches_selected_device_preflight(self):
        workflow = (ROOT / ".github/workflows/yue2-precision-proof.yml").read_text(encoding="utf-8")
        self.assertIn("'--cuda-scheduling-mode', $env:YUE2_CUDA_SCHEDULING_MODE", workflow)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            reference.mkdir()
            (reference / "vae_real_reference.safetensors").write_bytes(b"fixture")
            binary = root / "test-binary"
            binary.write_bytes(b"test")
            argv = ["yue2_precision_proof.py", "run", "--backend", "cuda",
                    "--binary", str(binary), "--quant-smoke-binary", str(binary),
                    "--vae-smoke-binary", str(binary), "--reference", str(reference),
                    "--evidence", str(root / "evidence"), "--work-dir", str(root / "listening"),
                    "--engine-sha", SHA, "--control-sha", SHA,
                    "--cuda-scheduling-mode", "shared-gpu1"]
            env = {"RUNNER_NAME": "cuda-windows", "CUDA_VISIBLE_DEVICES": "1",
                   "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "YUE2_IDLE_CONTEXT_RUN_ID": "",
                   "YUE2_CUDA_SCHEDULING_MODE": "shared-gpu1",
                   "GITHUB_REPOSITORY": "SceneWorks/inference", "GITHUB_JOB": "cuda",
                   "GITHUB_RUN_ATTEMPT": "1", "EXPECTED_ENGINE_SHA": SHA,
                   "EXPECTED_CONTROL_SHA": SHA, "GITHUB_SHA": SHA,
                   "YUE2_PRECISION_JOB_STARTED_UTC_NS": str(time.time_ns())}
            real_preflight = idle.check_shared_gpu1_dispatch
            def selected_preflight():
                with patch.object(idle.os, "name", "nt"):
                    real_preflight()
            with patch.object(proof.sys, "argv", argv), patch.dict("os.environ", env), \
                 patch.object(proof, "sha256", return_value=proof.REFERENCE_SHA256), \
                 patch.object(proof, "verify_revisions"), \
                 patch.object(proof.subprocess, "run", return_value=type("Result", (), {"stdout": ""})()), \
                 patch.object(idle, "check_shared_gpu1_dispatch", side_effect=selected_preflight) as selected, \
                 patch.object(idle, "SharedGpu1Lock", FakeLock), \
                 patch.object(proof, "cuda_physical_census",
                              side_effect=RuntimeError("selected GPU1 census reached")) as census, \
                 patch.object(proof.subprocess, "Popen") as launched:
                with self.assertRaisesRegex(RuntimeError, "selected GPU1 census reached"):
                    proof.main()
            selected.assert_called_once_with()
            census.assert_called_once_with()
            launched.assert_not_called()

    def test_workflow_and_app_capture_use_physical1(self):
        engine = (ROOT / ".github/workflows/yue2-precision-proof.yml").read_text(encoding="utf-8")
        app = (ROOT / ".github/workflows/yue2-app-precision-profile.yml").read_text(encoding="utf-8")
        capture = (ROOT / "scripts/ci/yue2_app_precision_profile.py").read_text(encoding="utf-8")
        self.assertEqual(engine.count('CUDA_VISIBLE_DEVICES: "1"'), 2)
        self.assertIn('-SelectedGpuIndex 1', engine)
        self.assertIn('CUDA_VISIBLE_DEVICES: "1"', app)
        self.assertIn('--platform cuda --gpu-id 1 --profile-install-only', app)
        self.assertIn('"--gpu-id", "1"', capture)

    def test_only_owned_gpu_pid_must_disappear_after_child(self):
        raw = json.dumps({"physicalMode": "shared-gpu1", "validatedDevice": {
            "observedActors": [[{"pid": 888, "type": "C+G"}] for _ in range(4)]}})
        self.assertTrue(proof.owned_gpu_pid_released(raw, 900))
        self.assertFalse(proof.owned_gpu_pid_released(raw, 888))
        with self.assertRaisesRegex(RuntimeError, "actor inventory unavailable"):
            proof.owned_gpu_pid_released(json.dumps({"physicalMode": "shared-gpu1"}), 888)

    def test_owned_process_tree_tracks_generation_and_refuses_descendant_leaks(self):
        parents = {500: 1, 501: 500, 502: 501, 999: 5}
        births = {500: 100, 501: 101, 502: 102, 999: 9}
        self.assertEqual(proof.descendant_generations(parents, births, 500, 100),
                         {501: 101, 502: 102})
        with self.assertRaisesRegex(RuntimeError, "reused"):
            proof.descendant_generations(parents, births, 500, 99)
        births[502] = 90
        with self.assertRaisesRegex(RuntimeError, "older than parent"):
            proof.descendant_generations(parents, births, 500, 100)
        births.pop(502)
        with self.assertRaisesRegex(RuntimeError, "creation identity unavailable"):
            proof.descendant_generations(parents, births, 500, 100)
        result = {"exit_code": 0, "timed_out": False, "wait_error": None,
                  "released": True, "exact_one_test_passed": True, "sample_count": 1,
                  "binary_unchanged_after_child": True, "sampler_faults": [],
                  "owned_gpu_pid_released": True, "owned_descendants_released": False,
                  "post_census_error": None, "post_census_busy": []}
        self.assertFalse(proof.child_stage_succeeded(result))
        result["owned_descendants_released"] = True
        self.assertTrue(proof.child_stage_succeeded(result))

    def test_adapter_luid_binding_survives_foreign_actor_changes(self):
        original = {"physicalMode": "shared-gpu1", "validatedDevice": {
            "uuid": idle.SHARED_GPU1_UUID, "pci": idle.SHARED_GPU1_PCI,
            "luid": LUID, "physicalIndex": 1, "cudaOrdinal": 0,
            "observedActors": [[{"pid": 888}]]}}
        changed = json.loads(json.dumps(original))
        changed["validatedDevice"]["observedActors"] = [[{"pid": 999}]]
        self.assertTrue(proof.same_selected_cuda_device(json.dumps(original), json.dumps(changed)))
        for key, value in (("luid", "other"), ("uuid", "GPU-other"),
                           ("physicalIndex", 0), ("cudaOrdinal", 1)):
            with self.subTest(key=key):
                mutated = json.loads(json.dumps(changed))
                mutated["validatedDevice"][key] = value
                self.assertFalse(proof.same_selected_cuda_device(json.dumps(original),
                                                                 json.dumps(mutated)))


if __name__ == "__main__":
    unittest.main()
