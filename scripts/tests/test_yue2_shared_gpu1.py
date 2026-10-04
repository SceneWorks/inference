"""CPU-only selected-card tests for the owner's physical GPU1 shared window."""
import json
import copy
from pathlib import Path
import sys
import tempfile
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
            # A coexisting actor may change residency/activity during our child.
            for index in range(3):
                path = root / f"gpu-sample-{index}.json"
                row = json.loads(path.read_text(encoding="utf-8"))
                row["output"][0] = row["output"][0].replace(", 553, 96886, 25, 18",
                                                              ", 70000, 27000, 80, 65")
                path.write_text(json.dumps(row), encoding="utf-8")
            self.verify(root, admission=False)
            with self.assertRaisesRegex(RuntimeError, "approved occupancy"):
                self.verify(root, admission=True)

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
                ("windows-counters-1", lambda row: row["counters"][0]["samples"][0].__setitem__(
                    "cookedValue", 51)),
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
                lambda row: row["samples"][-1].__setitem__("cookedValue", 51),
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
