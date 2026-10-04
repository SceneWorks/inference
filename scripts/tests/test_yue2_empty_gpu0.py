"""CPU-only refusal checks for the post-reboot process-free GPU0 lane."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import yue2_cuda_idle_context as idle
import yue2_precision_proof as proof

SHA = "a" * 40
LUID = "luid_0x00000000_0x00020b8c"
PCI = "0000:21:00.0"
UUID = "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0"
PMON_EMPTY = [
    "# gpu         pid   type     sm    mem    enc    dec    jpg    ofa     fb   ccpm    command",
    "# Idx           #    C/G      %      %      %      %      %      %     MB     MB    name",
    "    0          -     -      -      -      -      -      -      -      -      -    -",
]


def make_probe(root: Path) -> None:
    def put(name, body):
        (root / f"{name}.json").write_text(json.dumps(body) + "\n", encoding="utf-8")
    put("manifest", {"completed": True, "targetPid": 0, "engineSha": SHA,
                     "controlSha": SHA, "runner": "cuda-windows"})
    for name in ("process-before", "process-after"):
        put(name, {"pid": 0, "status": "no-target-process"})
    put("windows-counter-catalog", {"targetPid": 0, "sets": [
        {"name": "GPU Engine", "paths": [r"\GPU Engine(*)\Utilization Percentage"]},
        {"name": "GPU Process Memory", "paths": [
            r"\GPU Process Memory(*)\Dedicated Usage", r"\GPU Process Memory(*)\Shared Usage",
            r"\GPU Process Memory(*)\Total Committed"]},
        {"name": "GPU Adapter Memory", "paths": [
            r"\GPU Adapter Memory(*)\Dedicated Usage", r"\GPU Adapter Memory(*)\Shared Usage",
            r"\GPU Adapter Memory(*)\Total Committed"]}]})
    put("cuda-adapter-map", {"cuInit": 0, "cuDeviceGetCount": 0, "devices": [
        {"ordinal": 0, "cuDeviceGet": 0, "cuDeviceGetPCIBusId": 0,
         "cuDeviceGetLuid": 0, "nodeMask": 1,
         "luidBytes": "8C-0B-02-00-00-00-00-00", "pciBusId": PCI}]})
    paths = {"engine": r"\GPU Engine(*)\Utilization Percentage",
             "processDedicated": r"\GPU Process Memory(*)\Dedicated Usage",
             "processShared": r"\GPU Process Memory(*)\Shared Usage",
             "processCommitted": r"\GPU Process Memory(*)\Total Committed",
             "adapterDedicated": r"\GPU Adapter Memory(*)\Dedicated Usage",
             "adapterShared": r"\GPU Adapter Memory(*)\Shared Usage",
             "adapterCommitted": r"\GPU Adapter Memory(*)\Total Committed"}
    values = {"processDedicated": (4_493_312, 0),
              "processShared": (262_144, 0),
              "processCommitted": (4_755_456, 262_144),
              "adapterDedicated": 4_493_312,
              "adapterShared": 262_144,
              "adapterCommitted": 5_017_600}
    def counter_samples(key, index):
        if key == "engine":
            return [{"instance": f"pid_{40804 if index == 2 and number == 0 else 4}_{LUID}_phys_0_eng_{number}_engtype_3d",
                     "status": "0", "cookedValue": 0} for number in range(22)]
        if key.startswith("adapter"):
            return [{"instance": f"{LUID}_phys_0", "status": "0", "cookedValue": values[key]}]
        samples = [{"instance": f"pid_{pid}_{LUID}_phys_0", "status": "0", "cookedValue": amount}
                   for pid, amount in zip((4, 7976), values[key])]
        if index < 2 or key == "processCommitted":
            pid = 23028 if index < 2 else 40804
            samples.append({"instance": f"pid_{pid}_{LUID}_phys_0", "status": "0", "cookedValue": 0})
        return samples
    for index in range(3):
        put(f"gpu-sample-{index}", {"exitCode": 0, "output": [
            f"0, {UUID}, {PCI}, RTX, 596, 97887, 0, 97438, 0, 0"]})
        put(f"driver-mode-{index}", {"exitCode": 0, "output": [f"0, {UUID}, WDDM, Enabled"]})
        for gpu in (0, 1):
            put(f"pmon-{gpu}-{index}", {"exitCode": 0, "output": PMON_EMPTY})
            put(f"compute-apps-{gpu}-{index}", {"exitCode": 0, "output": []})
        put(f"windows-counters-{index}", {"targetPid": 0, "counters": [
            {"counter": path, "samples": counter_samples(key, index)}
            for key, path in paths.items()]})
    for name in ("gpu-before-cuda-properties", "gpu-after-cuda-properties"):
        put(name, {"exitCode": 0, "output": [f"0, {UUID}, {PCI}, 0, 0"]})
    put("pmon-0-final", {"exitCode": 0, "output": PMON_EMPTY})


class EmptyDeviceTests(unittest.TestCase):
    def test_process_free_census_retains_all_29_bytes_and_rejects_new_actor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_probe(root)
            files, encoded = idle.diagnostic_file_pairs(root)
            raw = json.dumps({"physicalMode": "empty-gpu0", "commandExit": 0,
                              "validatedDevice": {"physicalMode": "empty-gpu0"},
                              "diagnosticFiles": files, "diagnosticFileBytesB64": encoded})
            result = type("Result", (), {"returncode": 0, "stdout": "\n".join(PMON_EMPTY),
                                         "stderr": ""})()
            with patch.dict("os.environ", {"YUE2_IDLE_CONTEXT_RUN_ID": ""}), \
                 patch.object(proof.subprocess, "run", return_value=result), \
                 patch.object(idle, "census_empty_device", return_value=(raw, True)) as empty:
                self.assertEqual(proof.cuda_physical_census(), (raw, []))
                empty.assert_called_once()
            with patch.object(proof.subprocess, "run", return_value=type(
                    "Result", (), {"returncode": 0,
                                   "stdout": PMON_EMPTY[0] + "\n0 123 C 4 0 0 0 0 0 0\n",
                                   "stderr": ""})()), \
                 patch.object(idle, "census_empty_device") as empty:
                self.assertTrue(proof.cuda_physical_census()[1])
                empty.assert_not_called()

    def test_windows_probe_keeps_original_raw_family_inventory(self):
        source = (ROOT / "scripts/ci/yue2_cuda_context_diagnostic.ps1").read_text(encoding="utf-8")
        workflow = (ROOT / ".github/workflows/yue2-precision-proof.yml").read_text(encoding="utf-8")
        self.assertIn("[ValidateRange(0, 2147483647)]", source)
        self.assertIn("$TargetPid -eq 0 -or $pattern -like '*GPU Adapter Memory*'", source)
        self.assertIn("Save-CudaAdapterMap", source)
        self.assertIn("Invoke-Smi 'pmon-0-final'", source)
        self.assertIn("diagnostic_pid must be 0 or a positive decimal PID", workflow)
        self.assertIn("if: inputs.stage == 'cuda-diagnostic'", workflow)

    def test_same_29_raw_files_validate_device_without_an_old_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_probe(root)
            self.assertEqual(len(list(root.iterdir())), 29)
            with patch.dict("os.environ", {"EXPECTED_ENGINE_SHA": SHA, "GITHUB_SHA": SHA,
                                        "RUNNER_NAME": "cuda-windows"}):
                result = idle._empty_gpu0_summary(root)
                self.assertEqual((result["physicalMode"], result["gpu"]["usedMiB"]),
                                 ("empty-gpu0", 0))
                self.assertEqual(result["counters"]["adapterDedicated"],
                                 (f"{LUID}_phys_0", 4_493_312))
                self.assertEqual(result["counters"]["processDedicated"],
                                 [(f"pid_4_{LUID}_phys_0", 4_493_312)])
                self.assertEqual(result["counters"]["processCommitted"], [
                    (f"pid_4_{LUID}_phys_0", 4_755_456),
                    (f"pid_7976_{LUID}_phys_0", 262_144)])
                self.assertEqual(result["counters"]["engine"], "all-zero")
                self.assertIn("pid_23028", (root / "windows-counters-0.json").read_text(encoding="utf-8"))
                self.assertIn("pid_40804", (root / "windows-counters-2.json").read_text(encoding="utf-8"))
                # GPU1 may hold foreign work; selected physical GPU0 is the
                # admission target. The separate all-runner watch owns job scope.
                gpu1_pmon = root / "pmon-1-0.json"
                gpu1_apps = root / "compute-apps-1-0.json"
                gpu1_counters = root / "windows-counters-0.json"
                saved_gpu1 = (gpu1_pmon.read_bytes(), gpu1_apps.read_bytes(),
                              gpu1_counters.read_bytes())
                try:
                    gpu1_pmon.write_text(json.dumps({"exitCode": 0, "output": [
                        PMON_EMPTY[0], "1 33528 C 4400 30 0 0 0 0 0"]}), encoding="utf-8")
                    gpu1_apps.write_text(json.dumps({"exitCode": 0, "output": [
                        "33528, sc24163_cross.exe, 4400"]}), encoding="utf-8")
                    other_adapter = json.loads(saved_gpu1[2])
                    for row in other_adapter["counters"]:
                        if row["counter"].startswith(r"\GPU Adapter Memory"):
                            row["samples"].append({"instance": "luid_0x00000000_0x00020c10_phys_1",
                                                   "status": "0", "cookedValue": 4_400_000_000})
                    gpu1_counters.write_text(json.dumps(other_adapter), encoding="utf-8")
                    self.assertEqual(idle._empty_gpu0_summary(root)["gpu"]["usedMiB"], 0)
                finally:
                    gpu1_pmon.write_bytes(saved_gpu1[0])
                    gpu1_apps.write_bytes(saved_gpu1[1])
                    gpu1_counters.write_bytes(saved_gpu1[2])
                mutations = (
                    ("gpu-sample-1", lambda row: row["output"].__setitem__(
                        0, row["output"][0].replace(", 0, 0", ", 1, 0"))),
                    ("pmon-0-0", lambda row: row["output"].__setitem__(
                        1, "0 123 C 12 0 0 0 0 0 0")),
                    ("pmon-0-final", lambda row: row["output"].__setitem__(
                        1, "0 123 G 0 0 0 0 0 0 0")),
                    ("compute-apps-0-0", lambda row: row["output"].append("123, worker, 1")),
                    ("process-before", lambda row: row.__setitem__("status", "not_found")),
                    ("windows-counter-catalog", lambda row: row["sets"][1]["paths"].pop()),
                    ("cuda-adapter-map", lambda row: row["devices"][0].__setitem__(
                        "pciBusId", "0000:22:00.0")),
                    ("windows-counters-1", lambda row: row["counters"][-1]["samples"][0].__setitem__(
                        "cookedValue", 5_017_601)),
                    ("windows-counters-1", lambda row: row["counters"][1]["samples"][0].__setitem__(
                        "cookedValue", 4_493_313)),
                    ("windows-counters-1", lambda row: row["counters"][1]["samples"].pop(0)),
                    ("windows-counters-1", lambda row: row["counters"][1]["samples"].append(
                        {"instance": f"pid_999_{LUID}_phys_0", "status": "0", "cookedValue": 1})),
                    ("windows-counters-1", lambda row: row["counters"][1]["samples"].append(
                        dict(row["counters"][1]["samples"][-1]))),
                    ("windows-counters-1", lambda row: row["counters"][1]["samples"][-1].__setitem__(
                        "status", "unavailable")),
                    ("windows-counters-1", lambda row: row["counters"][0]["samples"][0].__setitem__(
                        "cookedValue", 1)),
                    ("windows-counters-1", lambda row: row["counters"][0]["samples"][0].__setitem__(
                        "status", "unavailable")),
                    ("windows-counters-1", lambda row: row["counters"][0]["samples"].append(
                        dict(row["counters"][0]["samples"][0]))),
                    ("windows-counters-1", lambda row: row["counters"][0].__setitem__("samples", None)),
                    ("windows-counters-1", lambda row: row["counters"][2].__setitem__("error", "unavailable")),
                    ("windows-counters-1", lambda row: row["counters"][3]["samples"][0].__setitem__(
                        "cookedValue", float("nan"))),
                    ("gpu-sample-0", lambda row: row["output"].__setitem__(
                        0, row["output"][0].replace(", 0, 97438, 0, 0", ", 1, 97437, 0, 0"))),
                )
                for name, mutate in mutations:
                    with self.subTest(name=name):
                        original = (root / f"{name}.json").read_bytes()
                        row = json.loads(original)
                        mutate(row)
                        (root / f"{name}.json").write_text(json.dumps(row), encoding="utf-8")
                        with self.assertRaises(RuntimeError):
                            idle._empty_gpu0_summary(root)
                        (root / f"{name}.json").write_bytes(original)
                # Identical missing fields in every epoch are still missing
                # telemetry, not a stable empty process/engine family.
                for family in ("engine", "processDedicated"):
                    with self.subTest(missing_samples=family):
                        saved_counters = {}
                        try:
                            for index in range(3):
                                path = root / f"windows-counters-{index}.json"
                                saved_counters[path] = path.read_bytes()
                                row = json.loads(saved_counters[path])
                                required = {"engine": r"\GPU Engine(*)\Utilization Percentage",
                                            "processDedicated": r"\GPU Process Memory(*)\Dedicated Usage"}
                                selected = next(item for item in row["counters"]
                                                if item["counter"] == required[family])
                                selected.pop("samples")
                                path.write_text(json.dumps(row), encoding="utf-8")
                            with self.assertRaisesRegex(RuntimeError, "samples unavailable"):
                                idle._empty_gpu0_summary(root)
                        finally:
                            for path, original in saved_counters.items():
                                path.write_bytes(original)
                # A stable Windows counter reservation is allowed; actual
                # selected-card NVML residency still must remain zero.
                names = [*(f"gpu-sample-{index}" for index in range(3)),
                         "gpu-before-cuda-properties", "gpu-after-cuda-properties"]
                saved = {name: (root / f"{name}.json").read_bytes() for name in names}
                try:
                    for name in names:
                        path = root / f"{name}.json"
                        row = json.loads(saved[name])
                        if name.startswith("gpu-sample"):
                            row["output"][0] = row["output"][0].replace(
                                ", 0, 97438, 0, 0", ", 1, 97437, 0, 0")
                        else:
                            row["output"][0] = row["output"][0].replace(", 0, 0", ", 1, 0")
                        path.write_text(json.dumps(row), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "residency changed or nonzero"):
                        idle._empty_gpu0_summary(root)
                finally:
                    for name, raw in saved.items():
                        (root / f"{name}.json").write_bytes(raw)

                # Coherent device re-enumeration must not transfer the owner's
                # GPU0 authorization to a different physical card.
                saved = {path: path.read_bytes() for path in root.glob("*.json")}
                try:
                    for path, raw in saved.items():
                        changed = (raw.decode("utf-8").replace(UUID, "GPU-other")
                                   .replace(PCI, "0000:22:00.0")
                                   .replace(LUID, "luid_0x00000000_0x00020c10")
                                   .replace("8C-0B-02-00-00-00-00-00", "10-0C-02-00-00-00-00-00"))
                        path.write_text(changed, encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "different physical GPU0"):
                        idle._empty_gpu0_summary(root)
                finally:
                    for path, raw in saved.items():
                        path.write_bytes(raw)

    def test_empty_route_requires_source_gpu0_and_no_old_receipt(self):
        with patch.object(idle, "check_device_selection"), patch.dict("os.environ", {
            "YUE2_IDLE_CONTEXT_RUN_ID": "", "EXPECTED_ENGINE_SHA": SHA,
            "EXPECTED_CONTROL_SHA": SHA, "GITHUB_SHA": SHA,
            "RUNNER_NAME": "cuda-windows", "GITHUB_REPOSITORY": "SceneWorks/inference",
            "GITHUB_JOB": "cuda", "GITHUB_RUN_ATTEMPT": "1",
            "YUE2_CUDA_SCHEDULING_MODE": "shared-host"}):
            idle.check_empty_dispatch()
            with patch.dict("os.environ", {"YUE2_IDLE_CONTEXT_RUN_ID": "old"}):
                with self.assertRaisesRegex(RuntimeError, "cannot select"):
                    idle.check_empty_dispatch()
            with patch.dict("os.environ", {"GITHUB_SHA": "b" * 40}):
                with self.assertRaisesRegex(RuntimeError, "source/control"):
                    idle.check_empty_dispatch()
            with patch.dict("os.environ", {"GITHUB_RUN_ATTEMPT": "2"}):
                with self.assertRaisesRegex(RuntimeError, "source/control"):
                    idle.check_empty_dispatch()

    def test_empty_pmon_rejects_graphics_and_unknown_rows(self):
        idle._pmon_empty(PMON_EMPTY, "test")
        for row in ("0 3 C - - - - - - 4 - worker", "0 3 G - - - - - - 4 - desktop",
                    "0 3 X - - - - - - 4 - unknown", "0 - - 0 - - - - - - - -",
                    "0 - - -", "1 - - - - - - - - - - -"):
            with self.subTest(row=row), self.assertRaises(RuntimeError):
                idle._pmon_empty(PMON_EMPTY[:2] + [row], "test")
        for rows in (PMON_EMPTY[1:], [PMON_EMPTY[1], PMON_EMPTY[0]],
                     [PMON_EMPTY[0], PMON_EMPTY[0], PMON_EMPTY[2]]):
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                idle._pmon_empty(rows, "test")
