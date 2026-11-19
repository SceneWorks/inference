"""Model-free checks for independently witnessed CUDA reference measurements."""

import contextlib
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.ci.sensenova_loop_assets import guarded, reference_measurements
from scripts.ci.sensenova_loop_reference import FIXTURES, STEPS
from scripts.media_campaign_supervisor import (
    SupervisionError, load_policy, runtime_guarded_admission,
)

GPU = "GPU-e4b79931-7be6-f216-460a-f5405cfafffe"
POLICY = {
    "schemaVersion": 1, "backend": "windows-cuda", "deadlineSeconds": 10,
    "pollMillis": 100, "termGraceMillis": 1000,
    "hostFreeReserveBytes": 100, "childFootprintCapBytes": 1000,
    "stdoutCapBytes": 2000, "stderrCapBytes": 2000, "eventCapBytes": 9000,
    "cudaDeviceUuid": GPU, "gpuFreeReserveBytes": 1000, "childGpuCapBytes": 10000,
}
COMPLETION = {
    "scope": "supervisor-owned-windows-job", "root_pid": 123,
    "root_returncode": 0, "waited_returncode": 0, "job_member_pids": [], "owned_pids": [],
}
MEASUREMENT = {
    "device_uuid": GPU, "visible_devices": 1, "device_ordinal": 0, "allocator_backend": "native",
    "allocator_scope": "PyTorch CUDA allocator", "live_tensor_dtype": "bfloat16",
    "live_tensor_bytes": 2000, "allocated_after_load_bytes": 3000,
    "peak_allocated_bytes": 5000, "peak_reserved_bytes": 8000,
    "forward_passes": len(FIXTURES) * STEPS * 3,
}


class ReferenceSupervisionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        policy = self.work / "policy.json"
        policy.write_text(json.dumps(POLICY), encoding="utf-8")
        self.policy = load_policy(policy)
        admission = runtime_guarded_admission(self.policy)
        admission.update(hostAvailableBytes=20000, gpuAvailableBytes=70000)
        self.supervision = {
            "device_claim": {
                "device_uuid": GPU, "physical_index": 1, "post_execution_verified": True,
                "lease_released": True, "owned_execution_completion": deepcopy(COMPLETION),
            },
            "process": {
                "pid": 123, "returncode": 0, "owned_completion": deepcopy(COMPLETION),
                "peak_gpu_bytes": None, "admission": admission,
                "samples": [
                    {"phase": "cuda-global-used-upper-bound-sample", "cudaDeviceUuid": GPU,
                     "gpuFreeBytes": 20000, "globalUsedUpperBoundBytes": 18000,
                     "childGlobalUsedGrowthBytes": 9000},
                ],
            },
        }

    def test_unknown_attribution_preserves_separate_pytorch_and_wddm_measurements(self):
        result = reference_measurements({"cuda_witness": MEASUREMENT}, POLICY, self.supervision)
        self.assertIsNone(result["external_supervisor_peak_bytes"])
        self.assertEqual(result["external_supervisor_sample"], "unknown")
        self.assertEqual(result["pytorch"]["peak_reserved_bytes"], 8000)
        self.assertEqual(result["wddm_global_peak_used_bytes"], 18000)
        self.assertEqual(result["wddm_global_peak_growth_bytes"], 9000)

    def test_bad_uuid_dtype_peak_or_missing_forward_fails_even_with_valid_lease(self):
        for key, value in (
            ("device_uuid", "GPU-00000000-0000-0000-0000-000000000001"),
            ("visible_devices", 2), ("device_ordinal", 1), ("live_tensor_dtype", "float32"),
            ("allocator_backend", "cudaMallocAsync"), ("live_tensor_bytes", 0),
            ("peak_allocated_bytes", 2000), ("peak_reserved_bytes", POLICY["childGpuCapBytes"] + 1),
            ("forward_passes", 1),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                reference_measurements({"cuda_witness": {**MEASUREMENT, key: value}}, POLICY, self.supervision)

    def test_allocator_witness_does_not_excuse_failed_reserve_or_unsealed_release(self):
        for claim, field, value in (
            ("device_claim", "physical_index", 0),
            ("device_claim", "lease_released", False),
            ("process", "owned_completion", {**COMPLETION, "job_member_pids": [124]}),
            ("process", "peak_gpu_bytes", 0),
            ("process", "admission", {**self.supervision["process"]["admission"], "gpuAvailableBytes": 1}),
            ("process", "samples", [{**self.supervision["process"]["samples"][0], "gpuFreeBytes": 1}]),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                supervision = deepcopy(self.supervision)
                supervision[claim][field] = value
                reference_measurements({"cuda_witness": MEASUREMENT}, POLICY, supervision)

    def test_error_ownership_is_forwarded_before_release_and_primary_is_retained(self):
        for fail_release in (False, True):
            with self.subTest(fail_release=fail_release):
                primary = SupervisionError("deadline", "original child failure")
                primary.owned_completion = deepcopy(COMPLETION)
                stage = f"reference-{fail_release}"

                @contextlib.contextmanager
                def mock_claim(backend, policy, work, env, *, constraints, require_owned_completion):
                    self.assertTrue(require_owned_completion)
                    self.assertEqual(constraints, {"required_cuda_physical_index": 1})
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], GPU)
                    claim = {"claim_deadline_monotonic": 1}
                    yield claim
                    self.assertEqual(claim["owned_execution_completion"], COMPLETION)
                    claim["lease_released"] = True
                    if fail_release:
                        raise RuntimeError("secondary release failure")
                    claim["post_execution_verified"] = True

                with (
                    patch("scripts.ci.sensenova_loop_assets.execution_claim", mock_claim),
                    patch("scripts.ci.sensenova_loop_assets.run_guarded", side_effect=primary) as run,
                    self.assertRaises(SupervisionError) as caught,
                ):
                    guarded(["python"], "candle-cuda", self.policy, self.work, {}, label=stage)
                self.assertIs(caught.exception, primary)
                self.assertFalse(run.call_args.kwargs["require_gpu_sample"])
                failure = json.loads((self.work / f"{stage}.failure.json").read_bytes())
                self.assertEqual(failure["supervision_reason"], "deadline")
                self.assertEqual(failure["owned_completion"], COMPLETION)
                self.assertEqual(failure["release_failure"], "RuntimeError" if fail_release else None)


if __name__ == "__main__":
    unittest.main()
