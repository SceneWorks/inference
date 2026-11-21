"""Selected-card pricing and end-of-render observations, without model execution."""

import copy
import unittest
from unittest.mock import MagicMock

from scripts.ci import wddm_process_memory as wddm
from scripts.ci.scail_plural import contract, cuda_stage

GPU = contract.GPU
PID, CREATED = 41, 99


def fixture():
    adapter = {
        "cuda_uuid": GPU,
        "luid": "luid_0x00000000_0x0001e266",
        "node_mask": 1,
        "physical_node": 0,
        "counter": wddm.COUNTER,
    }
    series = {
        "adapter": adapter,
        "pid": PID,
        "created_filetime": CREATED,
        "error": None,
        "samples": [[120, 50], [170, 12]],
        "missing_active_polls": 1,
    }
    other = dict(series, samples=[[121, 0], [171, 6]])
    common = {"pid": PID, "created_filetime": CREATED, "cuda_uuid": GPU}

    def retries(previous):
        return {
            "attempts": 1,
            "events": [
                {
                    "start_ns": 375,
                    "end_ns": 390,
                    "retried": True,
                    "local_filetime_before": 180,
                    "local_filetime_after": 185,
                    "previous": previous,
                    "retry_sample": None,
                    "recovered": False,
                }
            ],
        }

    stage = {
        "pool_ticks": 2,
        "pool_growth": [
            dict(
                card_index=1,
                at_ns=200,
                read_ns=185,
                cuda_uuid=GPU,
                free_bytes=190,
                global_used_upper_bytes=125,
                child_growth_bytes=78,
            ),
            dict(
                card_index=2,
                at_ns=300,
                read_ns=286,
                cuda_uuid=GPU,
                free_bytes=189,
                global_used_upper_bytes=124,
                child_growth_bytes=76,
            ),
        ],
        "pool": dict(
            common,
            card_index=1,
            at_ns=200,
            basis="verified-output-pool",
            pool_reserved_high_bytes=60,
        ),
        "zero": dict(
            common,
            card_index=3,
            at_ns=400,
            basis="owned-process-attributed",
            before="live",
            after="exited_empty",
        ),
    }
    raw = {
        "wddm_shared": other,
        "wddm_retry": {"dedicated": retries([170, 12]), "shared": retries([171, 6])},
        "card_samples_monotonic_ns_free_owned_or_fallback_bytes": [
            [100, 200, 75],
            [200, 190, 60],
            [300, 189, 60],
            [400, 275, 0],
        ],
        "cuda_render_end": stage,
    }
    growth = {
        "phase": "cuda-global-used-upper-bound-sample",
        "at_ns": 2000,
        "cudaDeviceUuid": GPU,
        "childGlobalUsedGrowthBytes": 75,
        "globalUsedUpperBoundBytes": 125,
        "gpuFreeBytes": 200,
    }
    guard = {
        "pid": PID,
        "admission": {"gpuGlobalUsedBaselineBytes": 50},
        "fit_observation": raw,
        "cuda_attribution": {
            "wddm_dedicated": series,
            "unknown_pid_polls": 3,
            "global_upper_bytes": 125,
        },
        "samples": [growth],
    }
    return guard, {
        "poolReservedHighBytes": 60,
        "poolReservedBytes": 60,
        "poolUsedHighBytes": 55,
        "poolUsedBytes": 45,
        "pid": PID,
        "cudaDeviceUuid": GPU,
    }


class CudaRenderEnd(unittest.TestCase):
    def test_underlying_tenant_growth_is_retained_without_another_device_read(self):
        system, owner = MagicMock(), object()
        observer = cuda_stage.KeepSystemGrowth(system)
        system.gpu_usage_basis = "global-used-upper-bound"
        system.policy.cuda_device_uuid = GPU
        system.last_global_used_upper_bound_bytes = 125
        system.gpu_free_and_tree_bytes.return_value = (190, 78)
        self.assertEqual(observer.gpu_free_and_tree_bytes(owner), (190, 78))
        system.gpu_free_and_tree_bytes.assert_called_once_with(owner)
        self.assertEqual(observer.current["child_growth_bytes"], 78)
        self.assertEqual(observer.current["global_used_upper_bytes"], 125)
        system.gpu_usage_basis = "owned-process-attributed"
        system.gpu_free_and_tree_bytes.return_value = (189, 0)
        self.assertEqual(observer.gpu_free_and_tree_bytes(owner), (189, 0))
        self.assertIsNone(observer.current)
        system.gpu_free_and_tree_bytes.side_effect = OSError("device read failed")
        with self.assertRaises(OSError):
            observer.gpu_free_and_tree_bytes(owner)
        self.assertIsNone(observer.current)

    def test_substituted_pool_tick_cannot_be_omitted_crossed_or_underpriced(self):
        for value, field in [
            (75, "read_ns"),
            (100, "at_ns"),
            (200, "free_bytes"),
            ("other", "cuda_uuid"),
            (74, "child_growth_bytes"),
            (124, "global_used_upper_bytes"),
        ]:
            with self.subTest(field=field):
                g, _ = fixture()
                entry = g["fit_observation"]["cuda_render_end"]["pool_growth"][0]
                if field == "global_used_upper_bytes":
                    entry["child_growth_bytes"] = 125
                entry[field] = value
                with self.assertRaisesRegex(ValueError, "pool substitution"):
                    cuda_stage.tenant_growth(g, GPU)
        g, _ = fixture()
        g["fit_observation"]["cuda_render_end"]["pool_growth"].pop()
        with self.assertRaisesRegex(ValueError, "missing original"):
            cuda_stage.tenant_growth(g, GPU)
        g, _ = fixture()
        g["cuda_attribution"]["global_upper_bytes"] = 126
        with self.assertRaisesRegex(ValueError, "card high-water"):
            cuda_stage.tenant_growth(g, GPU)

    def test_verified_render_keeps_actual_peaks_and_preserves_raw_gaps(self):
        g, pool = fixture()
        original = copy.deepcopy(g)
        growth, count = cuda_stage.tenant_growth(g, GPU)
        self.assertEqual((growth, count), (78, 1))
        scope = cuda_stage.render_scope(g, pool, GPU, count)
        self.assertEqual(
            scope,
            dict(
                pid=PID,
                created_filetime=CREATED,
                cuda_uuid=GPU,
                pool_card_index=1,
                zero_card_index=3,
                dedicated_missing_after_output=1,
                shared_missing_after_output=1,
            ),
        )
        self.assertEqual(g, original)
        dedicated = g["cuda_attribution"]["wddm_dedicated"]
        self.assertIsNone(
            wddm.sampled_peak(dedicated, pid=PID, adapter=dedicated["adapter"])
        )

    def test_full_native_outcome_uses_growth_and_preserves_hard_limits(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts.ci.scail2_plural_fit import measured_outcome
        from scripts.tests.test_scail2_plural_fit import native

        g, pool = fixture()
        g.update(
            returncode=0,
            elapsed_seconds=20,
            admission={"gpuGlobalUsedBaselineBytes": 50},
            peak_host_bytes=50,
            peak_gpu_bytes=60,
        )
        g["cuda_attribution"].update(
            pid_peak_bytes=0, pool_fallback_peak_bytes=60, global_upper_bytes=125
        )
        raw = g["fit_observation"]
        raw.update(
            host_baseline=2,
            host_samples_monotonic_ns_bytes=[[1, 2], [2, 50]],
            windows_peak={
                "owned_process_count": 1,
                "peak_job_commit_bytes": 50,
                "peak_private_commit_bytes": 49,
                "peak_working_set_bytes": 45,
            },
            wddm_shared_counter="GPU Process Memory/Shared Usage",
        )
        n = native("candle-cuda", 6)
        n.update(pid=PID, allocator_peak=60)
        adapter = g["cuda_attribution"]["wddm_dedicated"]["adapter"]
        policy = SimpleNamespace(
            child_footprint_cap_bytes=90,
            child_gpu_cap_bytes=80,
            poll_millis=250,
            sha256="policy",
        )

        def outcome():
            return measured_outcome(
                n,
                g,
                pool,
                adapter,
                {"uuid": GPU, "name": "board", "driver_version": "v", "index": 1},
                "candle-cuda",
                6,
                policy,
                "f" * 40,
            )

        with patch.object(contract, "validate_admission"):
            result = outcome()
            self.assertTrue(result["accepted"])
            candidate = result["candidate_row_pending_physical_review"]
            self.assertEqual(candidate["witness"]["gross_host_bytes"], 56)
            self.assertEqual(candidate["witness"]["gross_device_bytes"], 78)
            self.assertEqual(candidate["cuda_render_scope"]["pool_card_index"], 1)
            policy.child_gpu_cap_bytes = 76
            with self.assertRaisesRegex(ValueError, "exceeds the guard"):
                outcome()
            policy.child_gpu_cap_bytes = 80
            del raw["cuda_render_end"]
            self.assertEqual(
                outcome(),
                {"accepted": False, "telemetry_failure": "incomplete_wddm_samples"},
            )

    def test_foreign_series_falsified_watermark_and_unowned_zero_fail(self):
        for path, val in [
            (("samples", 0, "cudaDeviceUuid"), "crossed"),
            (("samples", 0, "childGlobalUsedGrowthBytes"), 150),
            (("fit_observation", "cuda_render_end", "pool", "created_filetime"), 100),
            (("fit_observation", "cuda_render_end", "zero", "before"), "exited_empty"),
            (("fit_observation", "cuda_render_end", "pool_ticks"), 10),
            (
                (
                    "fit_observation",
                    "cuda_render_end",
                    "pool",
                    "pool_reserved_high_bytes",
                ),
                80,
            ),
            (
                (
                    "fit_observation",
                    "card_samples_monotonic_ns_free_owned_or_fallback_bytes",
                    3,
                    2,
                ),
                1,
            ),
        ]:
            with self.subTest(path=path):
                guard, pool = fixture()
                if path[-1] == "before":
                    guard["fit_observation"]["cuda_render_end"]["zero"]["after"] = (
                        "live"
                    )
                obj = guard
                for field in path[:-1]:
                    obj = obj[field]
                obj[path[-1]] = val
                with self.assertRaises(ValueError):
                    _, n = cuda_stage.tenant_growth(guard, GPU)
                    cuda_stage.render_scope(guard, pool, GPU, n)

    def test_legacy_or_unlocated_or_in_render_missing_evidence_cannot_excuse_gaps(self):
        for mutation in [
            "no_marker",
            "early",
            "after_zero",
            "before_later_use",
            "not_last",
            "no_zero",
            "truncated",
        ]:
            with self.subTest(mutation=mutation):
                g, pool = fixture()
                r = g["fit_observation"]
                miss = r["wddm_retry"]["dedicated"]["events"][0]
                if mutation == "no_marker":
                    del r["cuda_render_end"]
                if mutation == "early":
                    miss.update(start_ns=115, end_ns=190)
                if mutation == "after_zero":
                    miss.update(start_ns=410, end_ns=420)
                if mutation == "before_later_use":
                    miss.update(start_ns=220, end_ns=230)
                if mutation == "not_last":
                    miss["previous"] = [120, 50]
                if mutation == "no_zero":
                    r["cuda_render_end"]["zero"] = None
                if mutation == "truncated":
                    r["wddm_retry"]["dedicated"]["events"] = []
                self.assertIsNone(cuda_stage.render_scope(g, pool, GPU, 1))

    def test_observer_binds_native_pool_and_owned_zero_on_the_original_job(self):
        g, pool = fixture()
        expected = g["fit_observation"]["cuda_render_end"]
        stage = cuda_stage.RenderEnd()
        probe = MagicMock()
        owner = MagicMock()
        owner.members.return_value = {PID}
        owner.root_identity.return_value = (CREATED, False)
        probe.dedicated.record = g["cuda_attribution"]["wddm_dedicated"]
        probe.policy.cuda_device_uuid = GPU
        probe.pool_fallback_peak_bytes = pool["poolReservedHighBytes"]
        probe.system.gpu_usage_basis = "global-used-upper-bound"
        probe.gpu_usage_basis = "owned-process-attributed"
        probe.system.current = {
            k: v
            for k, v in expected["pool_growth"][0].items()
            if k not in ("card_index", "at_ns")
        }
        stage.observe(owner, probe, None, 1, 200, 60)
        probe.system.current = {
            k: v
            for k, v in expected["pool_growth"][1].items()
            if k not in ("card_index", "at_ns")
        }
        stage.observe(owner, probe, stage.before(owner), 2, 300, 60)
        previous = stage.before(owner)
        owner.members.return_value = set()
        owner.root_identity.return_value = (CREATED, True)
        owner.exited_root_and_empty.return_value = True
        probe.system.gpu_usage_basis = "owned-process-attributed"
        stage.observe(owner, probe, previous, 3, 400, 0)
        self.assertEqual(stage.record, expected)
        owner.root_identity.return_value = (CREATED + 1, True)
        with self.assertRaisesRegex(ValueError, "identity"):
            stage.before(owner)
