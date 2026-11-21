import copy
import hashlib
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

    def test_shared_gauge_accounts_for_a_retryable_failed_nvidia_tick(self):
        base = MagicMock()
        observer = runner.Observer.__new__(runner.Observer)
        observer.base = base
        observer.shared = MagicMock()
        owner = object()
        base.gpu_free_and_tree_bytes.side_effect = runner.SupervisionError("probe-failure", "NVIDIA timeout")
        with self.assertRaisesRegex(runner.SupervisionError, "NVIDIA timeout"):
            observer.gpu_free_and_tree_bytes(owner)
        observer.shared.poll.assert_called_once_with(owner, cuda_active=True)

    def test_physical_measurements_require_owned_gpu_pool_proc_and_shared_counters(self):
        from types import SimpleNamespace
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
            admission={},
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
            sha256="hash", child_footprint_cap_bytes=10000, child_gpu_cap_bytes=10000
        )
        with patch.object(c, "validate_admission"):
            row = c.measured_row(
                data, guard, pool, adapter, machine, "candle-cuda", 6, approved, "f" * 40
            )
            self.assertEqual(row["witness"]["gross_device_bytes"], 600)
            self.assertEqual(row["witness"]["gross_host_bytes"], 800)
            self.assertEqual(row["witness"]["text_tokens"], 512)
            self.assertIsNone(row["witness"]["mlx_decode_budget_bytes"])
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
            approved.child_footprint_cap_bytes = 96 * c.GIB
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
