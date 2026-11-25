import copy
import hashlib
import itertools
import json
import os
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from scripts.ci import scail2_plural_fit as runner
from scripts.ci.scail_plural import contract as c


def native(backend, count):
    size = (2 * count + 90) * 832 * 480 * 3
    digest = hashlib.sha256(b"scail2-plural-weights-v1")
    for n in (
        "config.json",
        "tokenizer.json",
        "dit.safetensors",
        "t5_encoder.safetensors",
        "clip.safetensors",
        "vae.safetensors",
    ):
        digest.update(bytes.fromhex(c.FILES[n][1]))
    return {
        "schema": 1,
        "backend": backend,
        "count": count,
        "pid": 31,
        "files": [{"name": n, "sha256": v[1]} for n, v in c.FILES.items()],
        "axes": {
            "backend": "mlx" if backend == "mlx" else "cuda",
            "artifact_tier": "q4",
            "effective_tier": "q4",
            "floor_host_bytes": 0,
            "floor_unified_bytes": 0,
            "floor_device_bytes": 0,
            "windows": [{"mlx_unsplit_ffn": backend == "mlx"}],
            "width": 832,
            "height": 480,
            "references": count,
            "history_order": 4,
            "classifier_free_guidance": True,
            "artifact_sha256": digest.hexdigest(),
            "source_bytes": size,
        },
        "character_hashes": copy.deepcopy(c.CARRIERS["ordered_pairs"][:count]),
        "driving_hashes": [c.CARRIERS["driving_rgb"], c.CARRIERS["driving_masks"][str(count)]],
        "render_ns": 100,
        "process_work_ns": 120,
        "allocator_baseline": 1,
        "allocator_peak": 300,
        "source_bytes": size,
        "phases": [{"references": count}],
        "steps": [[i, 4] for i in range(1, 5)],
        "prompt_tokens": [512, 512],
        "execution": {
            "provider_id": "scail2_14b",
            "backend": "mlx" if backend == "mlx" else "candle",
            "sampling": "unipc",
            "quality_policy": "scail2_decode"
            if backend == "mlx"
            else "z16_causal_streaming_no_spatial_tiles",
        },
        "output": {
            "width": 832,
            "height": 480,
            "fps": 16,
            "frames": 48 if backend == "mlx" else 45,
            "sha256": "a" * 64,
        },
    }


class PluralQualification(unittest.TestCase):
    def test_registration_only_allows_reviewed_policies_and_full_batches(self):
        for backend, counts in c.COUNTS.items():
            request = {"selection": c.SELECTION, "policy": c.policy(backend)}
            with patch.object(
                runner.proof,
                "repository_identity",
                return_value={"tree": "tree", "revision": "head"},
            ):
                contract = runner.plan(backend, request, "head")
                self.assertEqual(contract["counts"], counts)
                self.assertEqual(contract["hub_snapshots"][c.SLOT]["subfolder"], "q4")
                self.assertEqual(contract["case"]["fqn"], c.FQN)
                for bad in [
                    dict(request, counts=[2]),
                    dict(request, policy=dict(request["policy"], deadlineSeconds=2401)),
                    dict(request, selection="scail2-tier1"),
                ]:
                    with self.assertRaises(ValueError):
                        runner.plan(backend, bad, "head")
            self.assertNotIn("@BACKEND@", runner.source(backend))
            self.assertNotIn("@PINS@", runner.source(backend))

    def test_gpu0_must_be_explicit_and_keeps_all_counts_guards_and_native_code(self):
        with patch.object(runner.proof, "repository_identity", return_value={"tree": "tree", "revision": "head"}):
            request = {"selection": c.SELECTION, "policy": c.policy("candle-cuda", cuda_uuid=c.GPU)}
            previous = runner.plan("candle-cuda", request, "head")
            selected = dict(request, policy=c.policy("candle-cuda", cuda_uuid=c.GPU0))
            plan = runner.plan("candle-cuda", selected, "head")
            self.assertEqual(plan, dict(previous, request=selected, cuda_completion="supervisor-owned-windows-job",
                fixture_module_sha256=hashlib.sha256(runner.source("candle-cuda", c.GPU0).encode()).hexdigest()))
            self.assertEqual(previous["fixture_module_sha256"], "ea14d6259e4c2bfc903c69bb34178f413585e91eeeec48649d78d6a67377a67c")
            self.assertEqual(selected["policy"], dict(request["policy"], cudaDeviceUuid=c.GPU0))
            self.assertEqual(previous["request"]["policy"]["cudaDeviceUuid"], c.GPU)
            self.assertNotEqual(c.digest_json(plan), c.digest_json(previous))
            for key, value in (("cudaDeviceUuid", "0"), ("cudaDeviceUuid", "GPU-unknown"),
                               ("childGpuCapBytes", 83*c.GIB), ("gpuFreeReserveBytes", 5*c.GIB)):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    runner.plan("candle-cuda", dict(selected, policy=dict(selected["policy"], **{key: value})), "head")
        limits = c.cuda_constraints(c.GPU0)
        self.assertEqual(limits, {"required_cuda_device_uuid": c.GPU0, "required_cuda_physical_index": 0})
        approved = SimpleNamespace(backend="windows-cuda", cuda_device_uuid=c.GPU0)
        self.assertEqual(runner.tier1.build_env(approved)["CUDA_VISIBLE_DEVICES"], c.GPU0)
        adapter = {"cuda_uuid": c.GPU0, "luid": "luid_0x00000000_0x00000001",
                   "physical_node": 0, "node_mask": 1, "counter": runner.wddm.COUNTER}
        with patch.object(runner.tier1.subprocess, "check_output", return_value=json.dumps(adapter).encode()) as preflight:
            self.assertEqual(runner.tier1.cuda_adapter(approved), adapter)
            self.assertEqual(preflight.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], c.GPU0)
        hw = {"uuid": c.GPU0, "index": "0", "memory.total": "98000"}
        claim = dict(post_execution_verified=True, lease_released=True, device_uuid=c.GPU0, physical_index=0)
        samples = [dict(sample_number=i, predicates={"unique_uuid":True}, selected_index_matches_admission=True,
                        post_owned_exit_primary_metadata_valid=True,
                        selected_card={"index":0,"total_mib":98000,"used_mib":500,"free_mib":97000}) for i in range(1,6)]
        release = claim["release_background_activity"] = {"samples":samples, "sample_count":5}
        claim["release_metadata_verification"] = {"status":"verified"}
        with patch.object(runner.tier1, "verify_cuda_claim") as replay:
            runner.require_release(claim, "candle-cuda", hw, approved)
            replay.assert_called_once_with(claim, hw, approved)
            samples[-1]["selected_card"]["index"] = 1
            with self.assertRaisesRegex(ValueError, "release sample"):
                runner.require_release(claim, "candle-cuda", hw, approved)
            samples[-1]["selected_card"]["index"] = 0
            release.update(samples=samples[:2], sample_count=2)
            claim["release_metadata_verification"] = {"status":"inconclusive-driver-query-timing","diagnostic":release}
            runner.require_release(claim, "candle-cuda", hw, approved)
            for wrong in (dict(claim, physical_index=1), dict(claim, physical_index=False), dict(claim, device_uuid=c.GPU)):
                with self.assertRaisesRegex(ValueError, "different physical"):
                    runner.require_release(wrong, "candle-cuda", hw, approved)
            for crossed in (dict(hw, uuid=c.GPU), dict(hw, index="1"), dict(hw, index=False)):
                with self.assertRaisesRegex(ValueError, "wrong Windows GPU"):
                    runner.require_release(claim, "candle-cuda", crossed, approved)

    def test_mlx_guard_accepts_the_tightened_policy_and_refuses_the_previous_cap(self):
        policy = c.policy("mlx")
        self.assertEqual(policy["childFootprintCapBytes"], 80 * c.GIB)
        self.assertEqual(policy["hostFreeReserveBytes"], 12 * c.GIB)
        with patch.object(runner.proof, "repository_identity", return_value={"tree": "tree", "revision": "head"}):
            request = {"selection": c.SELECTION, "policy": policy}
            self.assertEqual(runner.plan("mlx", request, "head")["counts"], [6, 2])
            with self.assertRaises(ValueError):
                runner.plan("mlx", dict(request, policy=dict(policy, childFootprintCapBytes=96 * c.GIB)), "head")

    @unittest.skipUnless(os.name == "posix", "process-group supervision needs POSIX")
    def test_mlx_admission_uses_the_reviewed_estimate_not_the_child_cap(self):
        """An idle 128 GB Mac (96,999,358,464 bytes available, run 2026-11-24) was refused at the
        80 GiB cap fallback. The reviewed estimate admits it; below estimate plus reserve refuses."""
        import sys
        from scripts import media_campaign_supervisor as safety

        class Probe:
            def __init__(self, free):
                self.free = free

            def host_admission(self):
                host = safety.darwin_host_memory(4096, {
                    "freePages": self.free // 4096, "speculativePages": 0, "purgeablePages": 0,
                    "inactivePages": 0, "fileBackedPages": 0, "anonymousPages": 0,
                    "throttledPages": 0, "activePages": 0})
                return host["availableBytes"], host

            def tree_footprint(self, _pgid):
                return 1024

        self.assertEqual(c.admission_estimate("mlx"), {
            "static_floor_host_bytes": 32 * c.GIB,
            "host_estimate_source": "scail2-plural-reviewed-unified-peak-v1"})
        self.assertEqual(c.admission_estimate("candle-cuda")["static_floor_gpu_bytes"], 62 * c.GIB)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "policy.json").write_text(json.dumps(c.policy("mlx")), encoding="utf-8")
            policy = safety.load_policy(root / "policy.json")

            def run(free, estimate):
                for name in ("stdout", "stderr"):
                    (root / name).unlink(missing_ok=True)
                return safety.run_guarded(
                    [sys.executable, "-c", "pass"], cwd=root, env=dict(os.environ), policy=policy,
                    stdout_path=root / "stdout", stderr_path=root / "stderr", probe=Probe(free),
                    **estimate)

            idle_mac = 96_999_358_464
            # The previous cap fallback refuses the idle Mac exactly as the failed capture did.
            with self.assertRaisesRegex(safety.SupervisionError, "child-footprint-cap-fallback"):
                run(idle_mac, {})
            result = run(idle_mac, c.admission_estimate("mlx"))
            admission = safety.validate_admission(result.admission, policy_sha256=policy.sha256)
            self.assertEqual((admission["hostEstimateSource"], admission["hostEstimateBytes"],
                              admission["childFootprintCapBytes"], admission["hostFreeReserveBytes"]),
                             (c.MLX_HOST_ESTIMATE_SOURCE, 32 * c.GIB, 80 * c.GIB, 12 * c.GIB))
            needed = policy.host_free_reserve_bytes + c.MLX_HOST_ESTIMATE_BYTES
            with self.assertRaisesRegex(safety.SupervisionError, "preflight-memory") as caught:
                run(needed - 4096, c.admission_estimate("mlx"))
            self.assertIn(c.MLX_HOST_ESTIMATE_SOURCE, caught.exception.detail)
            run(needed, c.admission_estimate("mlx"))

    def test_cuda_default_names_no_card_and_runs_on_whichever_registered_card_fits(self):
        request = {"selection": c.SELECTION, "policy": c.policy("candle-cuda")}
        self.assertNotIn("cudaDeviceUuid", request["policy"])
        with patch.object(runner.proof, "repository_identity", return_value={"tree": "tree", "revision": "head"}):
            plan = runner.plan("candle-cuda", request, "head")
            gpu0 = runner.plan("candle-cuda", {"selection": c.SELECTION,
                                               "policy": c.policy("candle-cuda", cuda_uuid=c.GPU0)}, "head")
        self.assertEqual(plan["cuda_card_selection"], runner.CARD_SELECTION_RULE)
        self.assertEqual((plan["counts"], plan["cuda_completion"]), (c.COUNTS["candle-cuda"], "supervisor-owned-windows-job"))
        self.assertNotIn("cuda_card_selection", gpu0)
        self.assertIn("cuda_identity()", runner.source("candle-cuda", c.ANY_CARD))
        # GPU 0 is short (another process holds 40 GB) and GPU 1 is free: the campaign takes GPU 1.
        def reader(uuid, _deadline):
            used = 40000 if uuid == c.GPU0 else 3000
            return [{"uuid": uuid, "index": c.CUDA_INDICES[uuid], "total_mib": 97887,
                     "used_mib": used, "free_mib": 97887 - used, "utilization_percent": 0}]
        with TemporaryDirectory() as directory:
            chosen, selection = runner.select_card("candle-cuda", Path(directory), reader=reader)
        self.assertEqual((chosen, selection["chosen_index"]), (c.GPU, 1))
        self.assertEqual([card.get("refusal", "")[:30] for card in selection["cards"]],
                         ["57887 MiB free < 69632 MiB req", ""])
        receipt = {"cuda_card_selection": selection, "hardware": {"uuid": c.GPU, "index": "1"}}
        contract = dict(plan, backend="candle-cuda")
        self.assertEqual(runner.executed_policy(contract, receipt), c.policy("candle-cuda", cuda_uuid=c.GPU))
        for mutate in (lambda r: r["cuda_card_selection"].update(chosen_uuid=c.GPU0, chosen_index=0),
                       lambda r: r["hardware"].update(uuid=c.GPU0),
                       lambda r: r["cuda_card_selection"]["cards"].reverse(),
                       lambda r: r["cuda_card_selection"]["cards"][1].update(required_free_bytes=1)):
            bad = copy.deepcopy(receipt)
            mutate(bad)
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "most-free registered card"):
                runner.executed_policy(contract, bad)
        # A named-card campaign (both accepted GPU 0 runs) records no selection and keeps its policy.
        self.assertEqual(runner.executed_policy(gpu0, {}), c.policy("candle-cuda", cuda_uuid=c.GPU0))
        with self.assertRaises(ValueError):
            runner.executed_policy(gpu0, receipt)

    def test_cuda_counts_state_the_reviewed_device_estimate_to_the_claim_and_supervisor(self):
        estimate = c.admission_estimate("candle-cuda")
        self.assertEqual(estimate, {
            "static_floor_host_bytes": 58 * c.GIB, "host_estimate_source": "scail2-plural-reviewed-host-peak-v1",
            "static_floor_gpu_bytes": 62 * c.GIB, "gpu_estimate_source": "scail2-plural-reviewed-device-peak-v1"})
        with TemporaryDirectory() as directory:
            loaded = runner.tier1.read_policy(Path(directory), "policy.json", c.policy("candle-cuda", cuda_uuid=c.GPU0))
        claimed = c.claim_estimate(loaded)
        self.assertEqual(claimed.sha256, loaded.sha256)
        from scripts.ci.model_proof_resources import claim_required_free_bytes
        self.assertEqual(claim_required_free_bytes(claimed), 68 * c.GIB)
        from scripts import media_campaign_supervisor as safety
        admission = safety.runtime_guarded_admission(loaded, **estimate)
        self.assertEqual((admission["hostEstimateBytes"], admission["gpuEstimateBytes"]), (58 * c.GIB, 62 * c.GIB))

    def test_a_count_priced_at_another_estimate_is_rejected_and_old_cap_records_still_verify(self):
        from scripts import media_campaign_supervisor as safety
        for backend in ("mlx", "candle-cuda"):
            with TemporaryDirectory() as directory:
                uuid = {"cuda_uuid": c.GPU0} if backend == "candle-cuda" else {}
                loaded = runner.tier1.read_policy(Path(directory), "policy.json", c.policy(backend, **uuid))
            stated = safety.runtime_guarded_admission(loaded, **c.admission_estimate(backend))
            c.validate_stated_admission(stated, backend)
            # The accepted GPU 0 and MLX runs were admitted at the caps (no stated estimate).
            c.validate_stated_admission(safety.runtime_guarded_admission(loaded), backend)
            wrong = dict(c.admission_estimate(backend))
            for floor, source in (("static_floor_host_bytes", "host_estimate_source"),
                                  ("static_floor_gpu_bytes", "gpu_estimate_source")):
                if floor not in wrong:
                    continue
                bad = dict(wrong, **{floor: wrong[floor] - c.GIB})
                with self.subTest(backend=backend, floor=floor), self.assertRaisesRegex(ValueError, "different estimate"):
                    c.validate_stated_admission(safety.runtime_guarded_admission(loaded, **bad), backend)
                renamed = dict(wrong, **{source: "other-estimate"})
                with self.subTest(backend=backend, source=source), self.assertRaisesRegex(ValueError, "different estimate"):
                    c.validate_stated_admission(safety.runtime_guarded_admission(loaded, **renamed), backend)
        # The claim's recorded device estimate must be the campaign's too.
        from scripts.ci.model_proof_resources import require_claimed_card
        executed = c.policy("candle-cuda", cuda_uuid=c.GPU0)
        claim = {"device_uuid": c.GPU0, "claim_gpu_estimate": {"source": c.CUDA_DEVICE_ESTIMATE_SOURCE,
                                                               "estimate_bytes": c.CUDA_DEVICE_ESTIMATE_BYTES}}
        require_claimed_card(claim, executed, runner.CARD_ESTIMATE)
        require_claimed_card({"device_uuid": c.GPU0}, executed, runner.CARD_ESTIMATE)
        for bad in (dict(claim, device_uuid=c.GPU),
                    dict(claim, claim_gpu_estimate={"source": c.CUDA_DEVICE_ESTIMATE_SOURCE, "estimate_bytes": 60 * c.GIB})):
            with self.assertRaises(ValueError):
                require_claimed_card(bad, executed, runner.CARD_ESTIMATE)

    def test_all_counts_both_backend_shapes_and_mutated_native_results(self):
        for backend, counts in c.COUNTS.items():
            for count in counts:
                data = native(backend, count)
                c.validate_native(data, backend, count)
                if backend == "mlx":
                    bad = copy.deepcopy(data)
                    bad["axes"]["windows"][0]["mlx_unsplit_ffn"] = False
                    with self.assertRaises(ValueError):
                        c.validate_native(bad, backend, count)
                for path, value in [
                    (("axes", "artifact_sha256"), "a" * 64),
                    (("axes", "references"), 1),
                    (("output", "frames"), 45 if backend == "mlx" else 48),
                    (("output", "fps"), 30),
                    (("prompt_tokens", 0), 511),
                    (("steps", 3), [3, 4]),
                    (("execution", "quality_policy"), "unknown"),
                    (("character_hashes", 0), data["character_hashes"][1]),
                    (("files", 4), {"name": "tokenizer.json", "sha256": "1" * 64}),
                ]:
                    bad = copy.deepcopy(data)
                    bad[path[0]][path[1]] = value
                    with (
                        self.subTest(backend=backend, count=count, path=path),
                        self.assertRaises(ValueError),
                    ):
                        c.validate_native(bad, backend, count)

    def test_stage_binds_six_complete_checked_files(self):
        files = [
            {"path": "q4/" + n, "bytes": v[0], "kind": "lfs_sha256", "digest": v[1]}
            for n, v in c.FILES.items()
        ]
        files[1].update(kind="git_blob_sha1", digest="2f1647f3edc2ec8d1c9660f3910a91f45d06a675")
        root = Path("cache/q4")
        stage = dict(
            repository=c.REPO,
            revision=c.REV,
            subfolder="q4",
            path=str(root),
            status="hub_content_verified",
            files=files,
        )
        c.validate_stage(stage, root)
        for bad in [
            dict(stage, revision="0" * 40),
            dict(stage, files=files[:-1]),
            dict(stage, files=files[::-1]),
            dict(stage, files=[dict(files[0], bytes=1), *files[1:]]),
        ]:
            with self.assertRaises(ValueError):
                c.validate_stage(bad, root)

    def test_one_same_tick_retry_retains_fresh_data_only(self):
        from scripts.ci import wddm_process_memory as w
        from scripts.ci.scail_plural import wddm_retry

        adapter = dict(cuda_uuid=c.GPU, luid="luid_0x00000000_0x0001e266",
                       physical_node=0, node_mask=1, counter=w.COUNTER)
        for outcome, retry, gap in [
            ("fresh", [110, 0], 0), ("missing", None, 1), ("cached", [100, 4], 1),
            ("stale", [102, 2], 1), ("late", [110, 2], 1),
            ("exit", [110, 2], 1), ("changed-root", [110, 2], 1),
        ]:
            with self.subTest(outcome=outcome):
                owner = MagicMock()
                owner.members.return_value = {31}
                owner.root_identity.return_value = (99, False)
                owner.exited_root_and_empty.return_value = True
                values = iter([[100, 4], None, retry])
                @contextmanager
                def query(*args, **kwargs):
                    def read():
                        value = next(values)
                        if value == retry and value is not None:
                            if outcome == "exit":
                                owner.members.return_value = set()
                            if outcome == "changed-root":
                                owner.root_identity.return_value = (77, False)
                        return value
                    yield read

                ticks = [1000, 2000, 2010, 2020, 2_000_000 if outcome == "late" else 2080]
                with patch.object(w, "_query", query), patch.object(wddm_retry, "local_filetime", side_effect=[109,111]), \
                        patch.object(wddm_retry.time, "sleep"), \
                        patch.object(wddm_retry.time, "monotonic_ns", side_effect=ticks):
                    probe = runner.SameTickSamples(w.DedicatedSamples(adapter, c.GPU), 1)
                    probe.poll(owner, cuda_active=True)
                    probe.poll(owner, cuda_active=True)
                self.assertEqual(probe.record["missing_active_polls"], gap)
                wddm_retry.validate_retry(probe.diagnostics, probe.record, 1)
                event = probe.diagnostics["events"][0]
                self.assertEqual(event["recovered"], outcome == "fresh")
                self.assertEqual(probe.diagnostics["attempts"], 1)
                if outcome == "fresh":
                    self.assertEqual(probe.record["samples"], [[100, 4], [110, 0]])
                    self.assertEqual(w.sampled_peak(probe.record, pid=31, adapter=adapter, allow_zero=True), 4)
                if outcome == "changed-root":
                    self.assertIn("identity changed", probe.record["error"])
                probe.close()

    def test_retries_preserve_earlier_missing_polls_and_stop_at_the_audit_limit(self):
        from scripts.ci.scail_plural import wddm_retry

        clock = patch.object(wddm_retry.time, "monotonic_ns", side_effect=itertools.count(1, 100).__next__)
        clock.start()
        self.addCleanup(clock.stop)
        gauge = MagicMock()
        gauge.record = {"samples": [], "missing_active_polls": 0, "error": None}
        probe = runner.SameTickSamples(gauge, 250)
        def miss(owner, *, cuda_active):
            gauge.record["missing_active_polls"] += 1
        def sample(owner, *, cuda_active):
            gauge.record["samples"].append([9, 0])
        # A new process's first actual sample can qualify; an older miss survives.
        reads = iter([miss, miss, miss, sample])
        gauge.poll.side_effect = lambda owner, **kwargs: next(reads)(owner, **kwargs)
        with patch.object(wddm_retry.time, "sleep"), patch.object(
            wddm_retry, "local_filetime", side_effect=[1, 5, 7, 10]
        ):
            probe.poll(object(), cuda_active=True)
            probe.poll(object(), cuda_active=True)
        self.assertEqual(gauge.record["missing_active_polls"], 1)
        wddm_retry.validate_retry(probe.diagnostics, gauge.record, 250)
        gauge.poll.side_effect = miss
        with patch.object(wddm_retry.time, "sleep"), patch.object(wddm_retry, "local_filetime", return_value=11):
            for _ in range(wddm_retry.MAX_RETRY_EVENTS):
                probe.poll(object(), cuda_active=True)
        self.assertEqual(len(probe.diagnostics["events"]), wddm_retry.MAX_RETRY_EVENTS)
        self.assertEqual(probe.diagnostics["attempts"], wddm_retry.MAX_RETRY_EVENTS + 2)
        wddm_retry.validate_retry(probe.diagnostics, gauge.record, 250)
        probe.diagnostics["events"][1]["local_filetime_before"] = 10
        with self.assertRaisesRegex(ValueError, "fresh same-tick"):
            wddm_retry.validate_retry(probe.diagnostics, gauge.record, 250)

    def test_partial_campaign_replays_each_completed_count_and_refuses_changed_results(self):
        with TemporaryDirectory() as directory:
            dest = Path(directory) / "evidence"
            dest.mkdir()
            contract = {"request": {"policy": c.policy("candle-cuda", cuda_uuid=c.GPU)}, "head": "head",
                        "counts": [6, 2, 3, 4, 5], "backend": "candle-cuda", "case": {}}
            receipt = {"contract": contract, "request": contract["request"], "run_id": "1",
                       "run_attempt": "1", "accepted": False, "failure": "incomplete campaign",
                       "logs": {}, "adapter": {"fixture": True}, "hardware": {"uuid": c.GPU, "index": "1"}, "counts": {}}
            claim = {"post_execution_verified": True, "lease_released": True, "device_uuid": c.GPU, "physical_index": 1}
            gap = {"accepted": False, "telemetry_failure": "incomplete_wddm_samples"}
            passed = {"accepted": True, "candidate_row_pending_physical_review": {"price": 200}}
            for n, result in [(6, gap), (2, passed)]:
                receipt["counts"][str(n)] = dict(result, native={"count": n}, body={"ok": True}, claim=claim)
                for suffix in ["stdout.log", "stderr.log", "supervisor.json", "cuda-pool.json"]:
                    name = f"count-{n}.{suffix}"
                    (dest / name).write_text("{}", encoding="utf-8")
                    receipt["logs"][name] = {"sha256": runner.tier1.sha(dest / name)}
            def verify():
                path = dest / "receipt.json"
                path.write_text(json.dumps(receipt), encoding="utf-8")
                with patch.dict(os.environ, {"RUNNER_TEMP": directory, "GITHUB_RUN_ID": "1",
                        "GITHUB_RUN_ATTEMPT": "1", "SCAIL_PLURAL_RECEIPT_SHA256": runner.tier1.sha(path)}):
                    runner.verify(SimpleNamespace(output=dest), contract)
            records = receipt["counts"]
            # Count 3 cannot follow 6 if 2 was omitted, even in a failed campaign.
            records["3"] = records.pop("2")
            with self.assertRaisesRegex(ValueError, "not a prefix"):
                verify()
            records["2"] = records.pop("3")
            # Every attempted predecessor must replay; only the last attempted
            # count may have aborted before completing physical verification.
            preceding = records["6"]
            records["6"] = {"accepted": False}
            with self.assertRaisesRegex(ValueError, "after a hard failure"):
                verify()
            records["6"] = preceding
            last = records["2"]
            records["2"] = {"accepted": False}
            with patch.object(runner.proof, "verify_execution", return_value={"ok": True}), \
                    patch.object(c, "measured_row", side_effect=c.IncompleteWddmSamples) as earlier:
                verify()
                self.assertEqual(earlier.call_count, 1)
            records["2"] = last
            with patch.object(runner.proof, "verify_execution", return_value={"ok": True}), \
                    patch.object(c, "measured_row", side_effect=[
                        c.IncompleteWddmSamples(), {"price": 200},
                        c.IncompleteWddmSamples(), {"price": 201},
                    ]) as measured:
                verify()
                self.assertEqual(measured.call_count, 2)
                with self.assertRaisesRegex(ValueError, "receipt changed"):
                    verify()
                receipt["accepted"] = True
                with self.assertRaisesRegex(ValueError, "incomplete campaign"):
                    verify()
                receipt["accepted"] = False
                claim["lease_released"] = False
                with self.assertRaisesRegex(ValueError, "release unverified"):
                    verify()

    def test_no_card_fitting_before_the_deadline_leaves_a_verifiable_unsuccessful_receipt(self):
        from scripts.ci import model_proof_resources as resources
        refusal = {"rule": resources.CARD_SELECTION_RULE, "polls": 2,
                   "cards": [{"uuid": uuid, "index": c.CUDA_INDICES[uuid], "refusal": "1000 MiB free < 69632 MiB required"}
                             for uuid in sorted(c.CUDA_INDICES, key=c.CUDA_INDICES.get)],
                   "admission_error": "no registered CUDA card had estimate plus reserve free"}
        receipt = {}
        self.assertTrue(resources.record_selection_refusal(
            receipt, resources.CudaClaimRejected(refusal, "no registered CUDA card fits this proof")))
        self.assertFalse(resources.record_selection_refusal({}, ValueError("other")))
        with TemporaryDirectory() as directory:
            dest = Path(directory) / "evidence"
            dest.mkdir()
            contract = {"request": {"policy": c.policy("candle-cuda")}, "head": "head", "cuda_card_selection": "rule",
                        "counts": [6, 2, 3, 4, 5], "backend": "candle-cuda", "case": {}, "cuda_completion": "job"}
            receipt.update({"contract": contract, "request": contract["request"], "run_id": "1", "run_attempt": "1",
                            "accepted": False, "failure": "no registered CUDA card fits this proof", "logs": {},
                            "counts": {}})

            def verify():
                path = dest / "receipt.json"
                path.write_text(json.dumps(receipt), encoding="utf-8")
                with patch.dict(os.environ, {"RUNNER_TEMP": directory, "GITHUB_RUN_ID": "1",
                        "GITHUB_RUN_ATTEMPT": "1", "SCAIL_PLURAL_RECEIPT_SHA256": runner.tier1.sha(path)}):
                    runner.verify(SimpleNamespace(output=dest), contract)

            verify()
            receipt["accepted"] = True
            with self.assertRaises(ValueError):
                verify()
            receipt["accepted"] = False
            receipt[resources.SELECTION_REFUSAL] = dict(refusal, cards=refusal["cards"][:1])
            with self.assertRaisesRegex(ValueError, "invalid card selection refusal"):
                verify()
            # A campaign that failed before choosing a card (staging or build) also verifies.
            del receipt[resources.SELECTION_REFUSAL]
            verify()
            receipt["counts"] = {"6": {"accepted": False}}
            with self.assertRaises(ValueError):
                verify()

    def test_shared_gauge_accounts_for_a_retryable_failed_nvidia_tick(self):
        base = MagicMock()
        observer = runner.Observer.__new__(runner.Observer)
        observer.base = base
        observer.render_end = runner.RenderEnd()
        observer.shared = MagicMock()
        owner = object()
        base.gpu_free_and_tree_bytes.side_effect = runner.SupervisionError("probe-failure", "NVIDIA timeout")
        with self.assertRaisesRegex(runner.SupervisionError, "NVIDIA timeout"):
            observer.gpu_free_and_tree_bytes(owner)
        observer.shared.poll.assert_called_once_with(owner, cuda_active=True)

    def test_physical_measurements_require_owned_gpu_pool_proc_and_shared_counters(self):
        from scripts.ci import wddm_process_memory as w

        data = native("candle-cuda", 6)
        adapter = dict(
            cuda_uuid=c.GPU,
            luid="luid_0x00000000_0x00000001",
            node_mask=1,
            physical_node=0,
            counter=w.COUNTER,
        )
        raw = dict(
            adapter=adapter,
            pid=31,
            created_filetime=99,
            samples=[[1, 500], [2, 600]],
            missing_active_polls=0,
            error=None,
        )
        shared = dict(raw, samples=[[1, 0], [2, 0]])
        guard = dict(
            pid=31,
            returncode=0,
            elapsed_seconds=10,
            admission={"cudaDeviceUuid": c.GPU},
            samples=[],
            peak_host_bytes=800,
            peak_gpu_bytes=600,
            fit_observation=dict(
                host_baseline=100,
                host_samples_monotonic_ns_bytes=[[1, 100], [2, 800]],
                windows_peak={
                    "owned_process_count": 1,
                    "peak_job_commit_bytes": 800,
                    "peak_working_set_bytes": 700,
                    "peak_private_commit_bytes": 800,
                },
                card_samples_monotonic_ns_free_owned_or_fallback_bytes=[[1, 9000, 600]],
                wddm_shared_counter="GPU Process Memory/Shared Usage",
                wddm_retry={k: {"attempts": 0, "events": []} for k in ("shared", "dedicated")},
                wddm_shared=shared,
            ),
            cuda_attribution=dict(
                pid_peak_bytes=600,
                unknown_pid_polls=0,
                global_upper_bytes=0,
                pool_fallback_peak_bytes=0,
                wddm_dedicated=raw,
            ),
        )
        pool = dict(pid=31, cudaDeviceUuid=c.GPU, poolUsedHighBytes=299, poolReservedHighBytes=300)
        machine = dict(uuid=c.GPU, index=1, name="GPU", driver_version="driver")
        approved = SimpleNamespace(
            cuda_device_uuid=c.GPU, sha256="hash", child_footprint_cap_bytes=10000, child_gpu_cap_bytes=10000, poll_millis=250
        )
        # The stated-estimate check has its own test; this fixture's admission is partial.
        with patch.object(c, "validate_admission"), patch.object(c, "validate_stated_admission"):
            row = c.measured_row(
                data, guard, pool, adapter, machine, "candle-cuda", 6, approved, "f" * 40
            )
            self.assertEqual(row["witness"]["gross_device_bytes"], 600)
            self.assertEqual(row["witness"]["gross_host_bytes"], 800)
            self.assertEqual(row["witness"]["text_tokens"], 512)
            self.assertIsNone(row["witness"]["mlx_decode_budget_bytes"])
            selected = SimpleNamespace(**dict(vars(approved), cuda_device_uuid=c.GPU0))
            board = dict(machine, uuid=c.GPU0, index="0", **{"pci.bus_id":c.GPU0_PCI})
            selected_pool = dict(pool, cudaDeviceUuid=c.GPU0)
            selected_adapter = dict(adapter, cuda_uuid=c.GPU0)
            selected_guard = copy.deepcopy(guard)
            selected_guard["admission"]["cudaDeviceUuid"] = c.GPU0
            selected_guard["cuda_attribution"]["wddm_dedicated"]["adapter"] = selected_adapter
            selected_guard["fit_observation"]["wddm_shared"]["adapter"] = selected_adapter
            gpu0_data = dict(data, cuda_device=c.cuda_identity(board, selected_adapter))
            def measure(g=selected_guard, p=selected_pool, a=selected_adapter, h=board, n=gpu0_data):
                return c.measured_row(n, g, p, a, h, "candle-cuda", 6, selected, "f"*40)
            self.assertEqual(measure()["witness"], row["witness"])
            for field, value in (("uuid", c.GPU),("pci_bus_id","00000000:22:00.0"),("ordinal",False),
                                 ("visible_count",2),("luid","luid_0x00000000_0x00000009"),("node_mask",2)):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    measure(n=dict(data, cuda_device=dict(gpu0_data["cuda_device"], **{field:value})))
            for crossed in ({"g": guard}, {"p": pool}, {"a": adapter}, {"h": machine}, {"n": data},
                            {"h":dict(board, **{"pci.bus_id":"00000000:22:00.0"})}):
                with self.assertRaises(ValueError):
                    measure(**crossed)
            missing = copy.deepcopy(guard)
            obs = missing["fit_observation"]
            obs["wddm_shared"]["missing_active_polls"] = 1
            obs["wddm_retry"]["shared"] = {"attempts": 1, "events": [{
                "start_ns": 500, "end_ns": 900, "retried": True, "previous": [2, 0],
                "retry_sample": None, "local_filetime_before": 5, "local_filetime_after": 7,
                "recovered": False,
            }]}
            def measure_missing():
                return runner.measured_outcome(data, missing, pool, adapter, machine, "candle-cuda",
                                               6, approved, "f" * 40)
            self.assertEqual(measure_missing(), {"accepted": False, "telemetry_failure": "incomplete_wddm_samples"})
            obs["windows_peak"]["peak_working_set_bytes"] = 10001
            with self.assertRaisesRegex(ValueError, "exceeds approved cap"):
                measure_missing()
            obs["windows_peak"]["peak_working_set_bytes"] = 700
            obs["wddm_shared"]["error"] = "owned process identity changed"
            with self.assertRaisesRegex(ValueError, "ownership"):
                measure_missing()
            for mutation in [
                "shared_gap",
                "pid",
                "dedicated_gap",
                "wrong_gpu",
                "missing_card",
                "overcap",
                "missing_host",
            ]:
                bad = copy.deepcopy(guard)
                invalid_pool = dict(pool)
                if mutation == "shared_gap":
                    bad["fit_observation"]["wddm_shared"]["missing_active_polls"] = 1
                elif mutation == "pid":
                    invalid_pool["pid"] = 90
                elif mutation == "dedicated_gap":
                    bad["cuda_attribution"]["wddm_dedicated"]["error"] = "lost ownership"
                elif mutation == "wrong_gpu":
                    invalid_pool["cudaDeviceUuid"] = "GPU-other"
                elif mutation == "missing_card":
                    bad["fit_observation"][
                        "card_samples_monotonic_ns_free_owned_or_fallback_bytes"
                    ] = []
                elif mutation == "overcap":
                    bad["peak_host_bytes"] = 10001
                else:
                    bad["fit_observation"]["host_samples_monotonic_ns_bytes"] = []
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    c.measured_row(
                        data,
                        bad,
                        invalid_pool,
                        adapter,
                        machine,
                        "candle-cuda",
                        6,
                        approved,
                        "f" * 40,
                    )
            mlx = native("mlx", 2)
            guard["lifetime_peak_host_bytes"] = 800
            approved.child_footprint_cap_bytes = c.policy("mlx")["childFootprintCapBytes"]
            result = c.measured_row(
                mlx,
                guard,
                None,
                None,
                {"chip": "Apple M5 Max", "os_build": "26B131"},
                "mlx",
                2,
                approved,
                "f" * 40,
            )
            self.assertEqual(result["witness"]["mlx_decode_budget_bytes"], 28 * c.GIB)
            guard["lifetime_peak_host_bytes"] = None
            with self.assertRaises(ValueError):
                c.measured_row(
                    mlx,
                    guard,
                    None,
                    None,
                    {"chip": "Apple M5 Max", "os_build": "26B131"},
                    "mlx",
                    2,
                    approved,
                    "f" * 40,
                )
