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
        "rejected_growth": [],
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


def fixture_with_rejected_growth():
    guard, pool = fixture()
    guard["admission"].update(
        cudaDeviceUuid=GPU, childGpuCapBytes=80, gpuFreeReserveBytes=50
    )
    guard["cuda_attribution"]["unknown_pid_polls"] += 2
    end = guard["fit_observation"]["cuda_render_end"]
    event = dict(
        cuda_uuid=GPU,
        pid=PID,
        created_filetime=CREATED,
        before="live",
        after="live",
        reason="probe-failure",
        detail="CUDA pool proof exceeds concurrent global use",
        pool=pool.copy(),
        next_card_index=3,
        read_ns=315,
        at_ns=320,
        free_bytes=250,
        child_growth_bytes=55,
        global_used_upper_bytes=100,
    )
    end["rejected_growth"] = [
        event,
        dict(
            event,
            read_ns=325,
            at_ns=340,
            free_bytes=280,
            child_growth_bytes=25,
            global_used_upper_bytes=72,
        ),
    ]
    return guard, pool


class CudaRenderEnd(unittest.TestCase):
    def test_rejected_substitutions_use_real_measurements_and_exact_outcomes(self):
        guard, pool = fixture_with_rejected_growth()
        peak, count = cuda_stage.tenant_growth(guard, GPU)
        self.assertEqual((peak, count), (78, 1))
        self.assertIsNotNone(cuda_stage.render_scope(guard, pool, GPU, count))
        raw = guard["fit_observation"]["cuda_render_end"]
        last = raw["rejected_growth"][-1]
        last.update(
            detail="CUDA pool proof has no valid owned PID/UUID allocation",
            after="exited_empty",
            child_growth_bytes=80,
            global_used_upper_bytes=125,
        )
        raw["zero"]["before"] = "exited_empty"
        self.assertEqual(cuda_stage.tenant_growth(guard, GPU), (80, 1))
        self.assertIsNotNone(cuda_stage.render_scope(guard, pool, GPU, count))

    def test_rejected_attempt_cannot_be_lost_crossed_underpriced_or_guessed(self):
        for field, value in [
            ("next_card_index", 1),
            ("next_card_index", 4),
            ("read_ns", 285),
            ("at_ns", 401),
            ("cuda_uuid", "crossed"),
            ("pid", 1),
            ("created_filetime", 1),
            ("before", "exited_empty"),
            ("after", "exited_empty"),
            ("reason", "device-memory"),
            ("pool", {"pid": 1}),
            ("free_bytes", 49),
            ("child_growth_bytes", 81),
            ("global_used_upper_bytes", 120),
            ("detail", "CUDA pool proof is malformed"),
            ("detail", "CUDA pool proof has no valid owned PID/UUID allocation"),
            ("omit", None),
            ("no_zero", None),
            ("foreign_pool", None),
            ("swap", None),
        ]:
            with self.subTest(field=field):
                guard, pool = fixture_with_rejected_growth()
                end = guard["fit_observation"]["cuda_render_end"]
                records = end["rejected_growth"]
                if field == "omit":
                    records.pop()
                elif field == "swap":
                    records.reverse()
                elif field == "no_zero":
                    end["zero"] = None
                elif field == "foreign_pool":
                    records[0]["pool"]["cudaDeviceUuid"] = "foreign"
                else:
                    records[0][field] = value
                with self.assertRaises(ValueError):
                    _, count = cuda_stage.tenant_growth(guard, GPU)
                    cuda_stage.render_scope(guard, pool, GPU, count)

    def test_underlying_tenant_growth_is_retained_without_another_device_read(self):
        system, owner = MagicMock(), object()
        observer = cuda_stage.KeepSystemGrowth(system)
        system.gpu_usage_basis = "global-used-upper-bound"
        system.policy.cuda_device_uuid = GPU
        system.policy.child_gpu_cap_bytes = 78
        system.policy.gpu_free_reserve_bytes = 180
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

    def test_same_watchdog_tick_records_two_real_rejections_then_only_one_completed_zero(
        self,
    ):
        from unittest.mock import patch
        from scripts.ci import scail2_plural_fit as runner
        from scripts.ci.flux1_pair.cuda_probe import PairCudaProbe
        from scripts.media_campaign_supervisor import read_gpu

        g, pool = fixture_with_rejected_growth()
        raw = g["fit_observation"]
        stage = raw["cuda_render_end"]
        system, owner, policy = MagicMock(), MagicMock(), MagicMock()
        policy.cuda_device_uuid = GPU
        policy.gpu_free_reserve_bytes = 50
        policy.child_gpu_cap_bytes = 80
        system.policy = policy
        with patch("scripts.ci.flux1_pair.cuda_probe.SystemProbe", return_value=system):
            pair = PairCudaProbe(policy, None)
        pair.unknown_pid_polls = 3
        pair.pid_peak_bytes = 0
        pair.pool_fallback_peak_bytes = 60
        pair.global_upper_bytes = 125
        pair.system = cuda_stage.KeepSystemGrowth(system)
        pair.dedicated = MagicMock()
        pair.dedicated.record = g["cuda_attribution"]["wddm_dedicated"]
        observer = runner.Observer.__new__(runner.Observer)
        observer.base = pair
        observer.shared = MagicMock()
        observer.render_end = cuda_stage.RenderEnd()
        observer.render_end.record = dict(stage, rejected_growth=[], zero=None)
        observer.card_samples = raw[
            "card_samples_monotonic_ns_free_owned_or_fallback_bytes"
        ][:3]
        owner.root_identity.return_value = (CREATED, False)
        owner.members.return_value = {PID}
        readings = iter(
            [
                ("global-used-upper-bound", 250, 55, 100),
                ("global-used-upper-bound", 280, 25, 72),
                ("owned-process-attributed", 275, 0, None),
            ]
        )

        def gpu_read(_owner):
            basis, free, used, absolute = next(readings)
            system.gpu_usage_basis = basis
            system.last_global_used_upper_bound_bytes = absolute
            return free, used

        system.gpu_free_and_tree_bytes.side_effect = gpu_read
        with (
            patch(
                "scripts.ci.flux1_pair.cuda_probe._read_pool_proof", return_value=pool
            ) as read_pool,
            patch("time.monotonic_ns", side_effect=[315, 320, 325, 340, 400]),
        ):
            self.assertEqual(
                read_gpu(lambda: observer.gpu_free_and_tree_bytes(owner)), (275, 0)
            )
        self.assertEqual(system.gpu_free_and_tree_bytes.call_count, 3)
        self.assertEqual(read_pool.call_count, 2)
        self.assertEqual(pair.unknown_pid_polls, 5)
        self.assertEqual(observer.shared.poll.call_count, 3)
        self.assertEqual(
            observer.card_samples,
            raw["card_samples_monotonic_ns_free_owned_or_fallback_bytes"],
        )
        self.assertEqual(
            observer.render_end.record["rejected_growth"], stage["rejected_growth"]
        )
        raw["cuda_render_end"] = observer.render_end.record
        _, primary = cuda_stage.tenant_growth(g, GPU)
        self.assertIsNotNone(cuda_stage.render_scope(g, pool, GPU, primary))

    def test_actual_source_guard_precedes_pool_substitution_and_all_retries(self):
        from scripts.ci.flux1_pair.cuda_probe import PairCudaProbe
        from scripts.media_campaign_supervisor import read_gpu, SupervisionError
        from unittest.mock import patch

        system = MagicMock()
        policy = system.policy
        policy.child_gpu_cap_bytes = 78
        policy.gpu_free_reserve_bytes = 180
        policy.cuda_device_uuid = GPU
        probe = PairCudaProbe.__new__(PairCudaProbe)
        probe.system = cuda_stage.KeepSystemGrowth(system)
        probe.dedicated = None
        probe.policy = policy
        for basis in ("owned-process-attributed", "global-used-upper-bound"):
            system.gpu_usage_basis = basis
            system.last_global_used_upper_bound_bytes = 120
            for reading in ((179, 77), (180, 79)):
                with self.subTest(basis=basis, reading=reading):
                    system.gpu_free_and_tree_bytes.reset_mock()
                    system.gpu_free_and_tree_bytes.return_value = reading
                    with patch(
                        "scripts.ci.flux1_pair.cuda_probe._read_pool_proof"
                    ) as pool:
                        with self.assertRaises(SupervisionError) as caught:
                            read_gpu(lambda: probe.gpu_free_and_tree_bytes(object()))
                        pool.assert_not_called()
                    system.gpu_free_and_tree_bytes.assert_called_once()
                    self.assertEqual(caught.exception.reason, "device-memory")
                    memory = caught.exception.watchdog_device_memory
                    self.assertEqual(memory["gpuFreeBytes"], reading[0])
                    self.assertEqual(memory["gpuUsageBasis"], basis)
                    self.assertEqual(
                        memory["ownedGpuBytes"]
                        if basis == "owned-process-attributed"
                        else memory["childGlobalUsedGrowthBytes"],
                        reading[1],
                    )

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
            repaired, _pool = fixture_with_rejected_growth()
            g["admission"].update(repaired["admission"])
            policy.cuda_device_uuid = GPU
            policy.gpu_free_reserve_bytes = 50
            end = raw["cuda_render_end"]
            end["rejected_growth"] = repaired["fit_observation"]["cuda_render_end"][
                "rejected_growth"
            ]
            g["cuda_attribution"]["unknown_pid_polls"] += 2
            self.assertTrue(outcome()["accepted"])
            g["admission"]["gpuFreeReserveBytes"] = 49
            with self.assertRaisesRegex(ValueError, "approved guard"):
                outcome()
            g["admission"]["gpuFreeReserveBytes"] = 50
            end["rejected_growth"] = []
            g["cuda_attribution"]["unknown_pid_polls"] -= 2
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
