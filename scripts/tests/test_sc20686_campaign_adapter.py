import argparse
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ADAPTER = Path(__file__).parents[1] / "sc20686_campaign_adapter.py"
REDUCER = Path(__file__).parents[1] / "sc20686_cache_attribution.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CampaignAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load(ADAPTER, "sc20686_adapter_test")
        cls.reducer = load(REDUCER, "sc20686_reducer_test")
        cls.coverage = json.loads(cls.adapter.COVERAGE.read_text(encoding="utf-8"))
        cls.source_map_hash = cls.adapter.digest(cls.adapter.SOURCE_MAP.read_bytes())

    def runner_safety(self, deadline=3):
        class Probe:
            def host_free(self):
                return 10**12
            def host_admission(self):
                return self.host_free(), None
            def tree_footprint(self, _pgid):
                return 1024
            def gpu_free(self):
                return 10**12
            def gpu_free_and_tree_bytes(self, _pgid):
                return 10**12, 1024
        policy = self.adapter.supervisor.SafetyPolicy(
            "linux-cuda", deadline, 10, 20, 1024, 10**9, 10**6, 10**6, 10**7,
            "GPU-12345678-1234-1234-1234-123456789abc", 1024, 10**9, "test", b"{}\n",
        )
        return {"safety_policy": policy, "probe": Probe()}

    def measured_admission(self, policy):
        """A CUDA admission as run_guarded records it: with the host and device measurements
        covering each estimate (the cap fallback) plus its reserve."""
        admission = self.adapter.supervisor.runtime_guarded_admission(policy)
        admission["hostAvailableBytes"] = policy.host_free_reserve_bytes + admission["hostEstimateBytes"]
        admission["gpuAvailableBytes"] = policy.gpu_free_reserve_bytes + admission["gpuEstimateBytes"]
        return admission

    def fake_estimator(self, root, document):
        """An executable answering `--sc20686-estimate` with `document` (or exiting 3 when None)."""
        script = root / "estimator"
        body = "import sys\n" + (
            "sys.exit(3)\n" if document is None else
            f"assert sys.argv[1:5] == ['--sc20686-estimate', '--variant', 'route', '--snapshot']\n"
            f"print('progress noise')\nprint({json.dumps(json.dumps(document))})\n")
        script.write_text(f"#!/usr/bin/env python3\n{body}", encoding="utf-8")
        script.chmod(0o755)
        return script

    def test_product_admission_estimate_is_parsed_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = {"schema": self.adapter.PRODUCT_ADMISSION_ESTIMATE_SCHEMA, "route": "route",
                    "source": "product-admission-profile", "estimateBytes": 700,
                    "phases": {"conditioning": 100, "decode": 700}, "components": {}}
            estimate = self.adapter.product_admission_estimate(
                self.fake_estimator(root, good), root, "route", ("--width", "512"))
            self.assertEqual(estimate["estimateBytes"], 700)
            for broken in ({**good, "schema": "v0"}, {**good, "route": "other"},
                           {**good, "source": "guess"}, {**good, "estimateBytes": 0},
                           {**good, "estimateBytes": 699}, {**good, "phases": {}}, None):
                with self.subTest(broken=broken):
                    with self.assertRaisesRegex(ValueError, "product admission estimate"):
                        self.adapter.product_admission_estimate(
                            self.fake_estimator(root, broken), root, "route")

    def test_runner_admits_on_the_product_estimate_plus_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            executable = root / "producer"
            executable.write_text(
                "#!/usr/bin/env python3\nimport sys\n"
                "open(sys.argv[sys.argv.index('--sc20686-events') + 1], 'w').write('{}\\n')\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            safety = self.runner_safety()
            policy, probe = safety["safety_policy"], safety["probe"]
            estimate = {"schema": self.adapter.PRODUCT_ADMISSION_ESTIMATE_SCHEMA, "route": "route",
                        "source": "product-admission-profile", "estimateBytes": 4096,
                        "phases": {"decode": 4096}, "components": {}}

            def run(measured=None, product=estimate):
                result = self.adapter.run_entrypoint(
                    executable, snapshot, "route", "normal", "a" * 40, "sequential", (),
                    timeout_seconds=5, failure_root=root / "failed", product_estimate=product,
                    measured_peak_host_bytes=measured, **safety)
                self.adapter.cleanup_campaign_run(result)
                return result.supervision

            # Exactly estimate plus reserve is admitted; far below the cap plus reserve.
            probe.host_free = lambda: policy.host_free_reserve_bytes + 4096
            supervision = run()
            admission = self.adapter.supervisor.validate_admission(
                supervision["admission"], policy_sha256=policy.sha256)
            self.assertEqual((admission["hostEstimateSource"], admission["hostEstimateBytes"]),
                             ("product-admission-profile", 4096))
            self.assertEqual(supervision["productAdmissionEstimate"], estimate)
            # One byte less is refused before spawn.
            probe.host_free = lambda: policy.host_free_reserve_bytes + 4095
            with self.assertRaisesRegex(ValueError, "preflight-memory"):
                run()
            # A completed arm of the same coordinate that measured more raises the estimate.
            probe.host_free = lambda: policy.host_free_reserve_bytes + 4096
            with self.assertRaisesRegex(ValueError, "preflight-memory"):
                run(measured=4097)
            probe.host_free = lambda: policy.host_free_reserve_bytes + 4097
            admission = run(measured=4097)["admission"]
            self.assertEqual((admission["hostEstimateSource"], admission["hostEstimateBytes"]),
                             (self.adapter.supervisor.MEASURED_PEAK_ESTIMATE_SOURCE, 4097))
            # A product estimate above the cap (or none) falls back to the cap.
            probe.host_free = lambda: 10**12
            for product in ({**estimate, "estimateBytes": policy.child_footprint_cap_bytes + 1,
                             "phases": {"decode": policy.child_footprint_cap_bytes + 1}}, None):
                admission = run(product=product)["admission"]
                self.assertEqual((admission["hostEstimateSource"], admission["hostEstimateBytes"]),
                                 (self.adapter.supervisor.CAP_FALLBACK_ESTIMATE_SOURCE,
                                  policy.child_footprint_cap_bytes))

    def test_runner_admits_by_runtime_guards_and_seals_refusal_or_abort_as_unaccepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            executable = root / "producer"
            executable.write_text(
                "#!/usr/bin/env python3\nimport sys, time\n"
                "open(sys.argv[sys.argv.index('--sc20686-events') + 1], 'w').write('{}\\n')\n"
                "time.sleep(float(sys.argv[-3]))\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            safety = self.runner_safety()
            policy = safety["safety_policy"]
            run = self.adapter.run_entrypoint(
                executable, snapshot, "route", "normal", "a" * 40, "sequential", ("0.05",),
                timeout_seconds=5, failure_root=root / "failed", **safety,
            )
            admission = self.adapter.supervisor.validate_admission(
                run.supervision["admission"], policy_sha256=policy.sha256)
            self.assertIsNone(admission["wholeProcessPeakBoundBytes"])
            self.assertEqual(admission["childGpuCapBytes"], policy.child_gpu_cap_bytes)
            # SC-20686 coordinates have no static estimate in the campaign path: host and device
            # fall back to their caps.
            fallback = self.adapter.supervisor.CAP_FALLBACK_ESTIMATE_SOURCE
            self.assertEqual((admission["hostEstimateSource"], admission["hostEstimateBytes"]),
                             (fallback, policy.child_footprint_cap_bytes))
            self.assertEqual((admission["gpuEstimateSource"], admission["gpuEstimateBytes"]),
                             (fallback, policy.child_gpu_cap_bytes))
            self.adapter.cleanup_campaign_run(run)
            self.assertFalse((root / "failed").exists())

            probe = safety["probe"]
            for outcome, name, override, expected in (
                ("refused", "host_free",
                 lambda: policy.host_free_reserve_bytes + policy.child_footprint_cap_bytes - 1,
                 "preflight-memory"),
                ("refused", "gpu_free",
                 lambda: policy.gpu_free_reserve_bytes + policy.child_gpu_cap_bytes - 1,
                 "CUDA free .* is below reserve .* plus estimate .*child-footprint-cap-fallback"),
                ("aborted", "tree_footprint",
                 lambda _owner: policy.child_footprint_cap_bytes + 1, "child-footprint"),
            ):
                setattr(probe, name, override)
                with self.assertRaisesRegex(ValueError, expected):
                    self.adapter.run_entrypoint(
                        executable, snapshot, "route", "normal", "a" * 40, "sequential",
                        ("5",), timeout_seconds=5, failure_root=root / "failed", **safety,
                    )
                delattr(probe, name)
            records = [json.loads(path.read_bytes())
                       for path in (root / "failed").glob("incomplete-*/unaccepted.json")]
            self.assertEqual(sorted(record["outcome"] for record in records),
                             ["aborted", "refused", "refused"])
            self.assertEqual(sorted(record["reason"] for record in records),
                             ["child-footprint", "preflight-memory", "preflight-memory"])
            for record in records:
                self.assertIs(record["accepted"], False)
                self.assertEqual(record["coordinate"], "route/normal")
                self.assertEqual(record["pid"] is None, record["outcome"] == "refused")
                self.adapter.supervisor.validate_admission(record["admission"], policy_sha256=policy.sha256,
                                                           admitted=record["outcome"] != "refused")

    def test_runner_seals_spawn_timeout_and_evidence_failures_as_unaccepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            malformed = root / "malformed"
            malformed.write_text("#!/usr/bin/env python3\nprint('no observer events')\n", encoding="utf-8")
            malformed.chmod(0o755)
            for label, entrypoint, timeout, expected in (
                ("spawn", root / "missing-entrypoint", 5, "spawn-failure"),
                ("timeout", malformed, 0.5, "invalid-timeout"),
                ("evidence", malformed, 5, "missing or malformed observer transcript"),
            ):
                failed = root / f"failed-{label}"
                with self.assertRaisesRegex(ValueError, expected):
                    self.adapter.run_entrypoint(
                        entrypoint, snapshot, "route", "normal", "a" * 40, "sequential",
                        timeout_seconds=timeout, failure_root=failed, **self.runner_safety(),
                    )
                [path] = failed.glob("incomplete-*/unaccepted.json")
                record = json.loads(path.read_bytes())
                self.assertIs(record["accepted"], False)
                outcome = {"spawn": "failed", "timeout": "refused", "evidence": "failed"}[label]
                self.assertEqual(record["outcome"], outcome)
                self.assertEqual(record["pid"] is None, label != "evidence")
                self.assertEqual(record["admission"] is None, label == "timeout")

    def flux_events(self, *, variant="flux2_klein_9b_edit", cancel=False):
        geometry = {
            "batch": 1, "resolution": "512x512", "reference_count": 1, "frames": 1,
            "prompt": "6c8d6812785e96e6241dc0e9b6d7d8d1542c6606fb70e3872156f82130f27238",
            "guidance": "1", "layers": 24, "heads": 24, "head_dimension": 128,
            "sq": 4096, "skv": 1024, "dtype": "BF16", "mask": "none", "rope": "4-axis",
        }
        metadata = {
            "phase": "metadata", "source_ref": "a" * 40, "snapshot_sha256": "b" * 64,
            "model_snapshot_revision": "b" * 40,
            "residency_strategy": self.adapter.PRODUCT_RESIDENCY.get(variant, "sequential"),
            "snapshot_bytes": 1, "variant": variant, "geometry": geometry,
            "real_weights": True, "full_generation": not cancel, "attention_kind": "cross",
            "cancellation_armed": cancel,
        }
        if cancel:
            metadata["cancellation_arm_id"] = "a:flux2_klein_9b_edit"
        terminal = "cancelled" if cancel else "generation-end"
        dense_reference = self.adapter.dense_reference_kv_bytes(geometry)
        metrics = {
            "phase": "metrics", "sample_kind": "allocator", "peak_bytes": 10 * 1024**3,
            "current_persistent_bytes": 0, "current_read_transient_bytes": dense_reference,
            "candidate_persistent_bytes": 100 * 1024**2,
            "candidate_read_transient_bytes": dense_reference,
            "generation_duration_ms": 1000, "cache_read_duration_ms": 0,
            "joint_attention_context_duration_ms": 100,
            "reference_runtime_attribution_available": False,
            "reused_requests": 2, "minimum_cache_reads": 2,
        }
        base = {"sample_kind": "allocator", "peak_bytes": 10 * 1024**3}
        allocator = {
            "allocator_before_bytes": 1024**3,
            "allocator_after_bytes": 1024**3,
            "allocator_high_bytes": 1024**3 + dense_reference,
            "allocator_reserved_bytes": 10 * 1024**3,
            "allocator_measurement_available": True,
        }
        remnant = {
            "allocator_before_bytes": 512 * 1024**2,
            "allocator_after_bytes": 512 * 1024**2,
            "allocator_high_bytes": 512 * 1024**2,
            "allocator_reserved_bytes": 2 * 1024**3,
            "allocator_measurement_available": True,
        }
        return [
            metadata,
            {"phase": "generation-start", **base},
            {"phase": "cross-kv-created", "operation": "DoubleAttention::to_k/to_v(reference-slice)",
             "persistent_bytes": 0, "transient_bytes": dense_reference, **base},
            {"phase": "cross-kv-read", "transient_bytes": dense_reference,
             "operation": "DoubleAttention::attention(joint-context-non-attributable)",
             "reused": 1,
             **allocator, **base},
            {"phase": "cross-kv-read", "transient_bytes": dense_reference,
             "operation": "DoubleAttention::attention(joint-context-non-attributable)",
             "reused": 1,
             **allocator, **base},
            {"phase": terminal, **base},
            metrics,
            {"phase": "invalidated", **base},
            {"phase": "released", **remnant, **base},
            {"phase": "process-sample", "sample_kind": "process", "peak_bytes": 10 * 1024**3},
        ]

    def config(self):
        return {
            "inference_revision": "a" * 40,
            "model_snapshot_revision": "b" * 40,
            "residency_strategy": "sequential",
            "route_manifest_sha256": "c" * 64,
            "source_map_sha256": self.source_map_hash,
            "input_file_sha256": {"reference-0": "d" * 64},
            "evidence_artifact_sha256": {"run.media.json": "e" * 64},
        }

    def row_args(self, cancel=False):
        return argparse.Namespace(
            fake=False, family="flux2-klein", variant="flux2_klein_9b_edit",
            coordinate_name="edit-512-ref1-cfg1", cancel_campaign=cancel,
        )

    def test_flux_route_identity_is_product_operation_not_config_id(self):
        row = self.adapter.make_row(
            self.row_args(), self.config(), "b" * 64, 1, self.flux_events()
        )
        self.assertEqual(row["variant"], "flux2_klein_9b_edit")
        with self.assertRaisesRegex(ValueError, "exact product route"):
            self.adapter.make_row(
                self.row_args(), self.config(), "b" * 64, 1,
                self.flux_events(variant="flux2_klein_9b"),
            )

    def test_row_binds_repository_model_and_residency_as_separate_axes(self):
        mutations = (
            ("source_ref", "c" * 40, "inference revision"),
            ("model_snapshot_revision", "c" * 40, "model revision"),
            ("residency_strategy", "resident", "residency"),
        )
        for field, value, expected in mutations:
            with self.subTest(field=field):
                events = self.flux_events()
                events[0][field] = value
                with self.assertRaisesRegex(ValueError, expected):
                    self.adapter.make_row(
                        self.row_args(), self.config(), "b" * 64, 1, events
                    )

    def test_inference_revision_must_match_the_exact_checkout(self):
        actual = self.adapter.subprocess.check_output(
            ["git", "-C", str(self.adapter.INFERENCE_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        self.assertEqual(self.adapter.verify_inference_revision(actual), actual)
        wrong = "0" * 40 if actual != "0" * 40 else "1" * 40
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            self.adapter.verify_inference_revision(wrong)

    def test_active_high_water_and_post_release_remnant_are_required(self):
        broken_read = self.flux_events()
        next(event for event in broken_read if event["phase"] == "cross-kv-read")[
            "allocator_measurement_available"
        ] = False
        with self.assertRaisesRegex(ValueError, "active allocator high-water"):
            self.adapter.make_row(
                self.row_args(), self.config(), "b" * 64, 1, broken_read
            )
        broken_release = self.flux_events()
        next(event for event in broken_release if event["phase"] == "released")[
            "allocator_measurement_available"
        ] = False
        with self.assertRaisesRegex(ValueError, "post-release remnant"):
            self.adapter.make_row(
                self.row_args(), self.config(), "b" * 64, 1, broken_release
            )

    def test_reserved_high_must_cover_reserved_current_and_used_high(self):
        for field in ("allocator_reserved_bytes", "allocator_high_bytes"):
            broken = self.flux_events()
            read = next(event for event in broken if event["phase"] == "cross-kv-read")
            if field == "allocator_high_bytes":
                read["allocator_reserved_bytes"] = read["allocator_after_bytes"]
            read["peak_bytes"] = read[field] - 1
            with self.assertRaisesRegex(ValueError, "allocator high-water/remnant ordering"):
                self.adapter.make_row(
                    self.row_args(), self.config(), "b" * 64, 1, broken
                )

    def test_flux_joint_duration_is_context_not_reference_runtime(self):
        events = self.flux_events()
        row = self.adapter.make_row(
            self.row_args(), self.config(), "b" * 64, 1, events
        )
        self.assertFalse(row["reference_runtime_attribution_available"])
        self.assertEqual(row["cache_read_duration_ms"], 0)
        self.assertEqual(row["joint_attention_context_duration_ms"], 100)
        metrics = next(event for event in events if event["phase"] == "metrics")
        metrics["cache_read_duration_ms"] = metrics["joint_attention_context_duration_ms"]
        with self.assertRaisesRegex(ValueError, "runtime duration|non-attributable"):
            self.adapter.make_row(
                self.row_args(), self.config(), "b" * 64, 1, events
            )

    def test_raw_receipt_sidecar_hashes_exact_unsigned_artifact(self):
        sealed, raw, sidecar = self.adapter.seal({"value": 1}, "row.json")
        self.reducer.verify_seal_artifact(sealed, raw, sidecar, "row.json")
        with self.assertRaises(ValueError):
            self.reducer.verify_seal_artifact(sealed, raw + b" ", sidecar, "row.json")

    def test_streaming_runner_keeps_carriage_return_progress_out_of_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "producer"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import json,os,sys,time\n"
                "event_path = sys.argv[sys.argv.index('--sc20686-events') + 1]\n"
                "out = sys.argv[sys.argv.index('--out') + 1]\n"
                "os.makedirs(out)\n"
                "open(os.path.join(out, 'frame-0000.png'), 'wb').write(b'media')\n"
                "with open(event_path, 'w', encoding='utf-8') as events:\n"
                "    for i in range(10000):\n"
                "        events.write(json.dumps({'phase':'noise','payload':'x'*256}) + '\\n')\n"
                "print('\\rprovider progress 100%', end='', flush=True)\n"
                "print('diagnostic'*10000, flush=True)\n"
                "print(os.getcwd(), flush=True)\n"
                "time.sleep(.2)\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            run = self.adapter.run_entrypoint(
                executable, snapshot, "route", "normal", "a" * 40, "sequential",
                timeout_seconds=5,
                **self.runner_safety(),
            )
            self.assertEqual(len(run.events) - len(run.process_samples), 10000)
            self.assertGreater(len(run.stdout), 64 * 1024)
            self.assertIn(b"\rprovider progress", run.stdout)
            self.assertNotIn(b"\r", run.event_transcript)
            self.assertTrue(run.process_samples)
            event_path = Path(run.command[run.command.index("--sc20686-events") + 1])
            output_path = Path(run.command[run.command.index("--out") + 1])
            self.assertEqual(event_path.parent, output_path.parent)
            self.assertEqual(event_path.parent.name, "sealed-run")
            self.assertIn(str(event_path.parent).encode(), run.stdout)
            self.assertEqual(
                run.command[run.command.index("--sc20686-source-ref") + 1], "a" * 40
            )
            self.assertEqual(
                run.command[run.command.index("--sc20686-residency") + 1], "sequential"
            )
            self.assertTrue(run.media_output.is_dir())
            self.adapter.cleanup_campaign_run(run)
            self.assertFalse(run.cleanup_root.exists())

    def test_streaming_runner_fails_closed_on_missing_or_malformed_event_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            executable = root / "producer"
            for payload, expected in ((None, "missing or malformed"), ("{not-json}\\n", "missing or malformed")):
                body = "#!/usr/bin/env python3\nimport sys\n"
                if payload is not None:
                    body += (
                        "event_path = sys.argv[sys.argv.index('--sc20686-events') + 1]\n"
                        f"open(event_path, 'w', encoding='utf-8').write({payload!r})\n"
                    )
                executable.write_text(body, encoding="utf-8")
                executable.chmod(0o755)
                with self.assertRaisesRegex(ValueError, expected):
                    self.adapter.run_entrypoint(
                        executable, snapshot, "route", "normal", "a" * 40,
                        "sequential", timeout_seconds=5,
                        **self.runner_safety(),
                    )

    def test_streaming_runner_enforces_timeout_and_terminates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "producer"
            executable.write_text(
                "#!/usr/bin/env python3\nimport time\nprint('{}', flush=True)\ntime.sleep(5)\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            with self.assertRaisesRegex(ValueError, "deadline"):
                self.adapter.run_entrypoint(
                    executable, snapshot, "route", "normal", "a" * 40,
                    "sequential", timeout_seconds=0.1,
                    **self.runner_safety(deadline=0.08),
                )

    def _manifest_fixture(self, root):
        prompt_by_hash = {
            "ff8b29180ce69f56369bd1222522f384564bc4154035ebb06c1efbdc30e36be5":
                "SC-20686 representative still-motion study",
            "c19800ff5dc04d6cbd7d00d636218c569318b28a3df652255a531c4c3b3de713":
                "SC-20686 representative long-motion study",
        }
        image = root / "image.png"
        image.write_bytes(b"image")
        reference = root / "reference.png"
        reference.write_bytes(b"reference")
        control = root / "control"
        mask = root / "mask"
        control.mkdir()
        mask.mkdir()
        (control / "0.png").write_bytes(b"control")
        (mask / "0.png").write_bytes(b"mask")
        entries = {}
        for route in self.adapter.WAN_ROUTES:
            binary = root / self.adapter.WAN_ENTRYPOINT_STEMS[route]
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            binary.chmod(0o755)
            snapshot = root / route / ("1" * 40)
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            coordinates = {}
            for name, expected in self.coverage["families"]["wan"][route]["coordinates"].items():
                width, height = expected["resolution"].split("x")
                values = [
                    "--width", width, "--height", height, "--frames", str(expected["frames"]),
                    "--prompt", prompt_by_hash[expected["prompt"]], "--guidance",
                    expected["guidance"], "--steps", "4",
                ]
                if route == "wan2_2_i2v_14b":
                    values[0:0] = ["--image", str(image)]
                if route in ("wan_vace", "wan2_2_vace_fun_14b"):
                    values[0:0] = [
                        "--control-dir", str(control), "--mask-dir", str(mask),
                    ]
                    if expected["reference_count"]:
                        values[0:0] = ["--reference", str(reference)]
                coordinates[name] = values
            entries[route] = {
                "provider_id": route, "entrypoint": str(binary),
                "snapshot": str(snapshot),
                "residency_strategy": self.adapter.PRODUCT_RESIDENCY[route],
                "coordinates": coordinates,
            }
        return entries

    def test_manifest_closes_frozen_coordinates_and_hashes_file_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = self._manifest_fixture(root)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(entries), encoding="utf-8")
            loaded = self.adapter.load_wan_manifest(manifest)
            self.assertEqual(set(loaded), set(self.adapter.WAN_ROUTES))
            i2v = next(iter(loaded["wan2_2_i2v_14b"].values()))
            self.assertTrue(any(item["flag"] == "--image" for item in i2v.input_file_inventory))
            vace = loaded["wan_vace"]["landscape-33f-reference"]
            self.assertEqual(
                {item["flag"] for item in vace.input_file_inventory},
                {"--reference", "--control-dir", "--mask-dir"},
            )
            self.adapter.verify_coordinate_inputs(i2v)
            image = next(
                Path(item["path"])
                for item in i2v.input_file_inventory if item["flag"] == "--image"
            )
            image.write_bytes(b"changed after resolution")
            with self.assertRaisesRegex(ValueError, "route input changed"):
                self.adapter.verify_coordinate_inputs(i2v)
            del entries[self.adapter.WAN_ROUTES[-1]]["coordinates"][
                next(iter(entries[self.adapter.WAN_ROUTES[-1]]["coordinates"]))
            ]
            manifest.write_text(json.dumps(entries), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen coverage"):
                self.adapter.load_wan_manifest(manifest)

    def test_diffusers_layout_and_nested_q4_model_revision_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision = "2" * 40
            diffusers = root / revision
            (diffusers / "transformer").mkdir(parents=True)
            (diffusers / "model_index.json").write_text("{}", encoding="utf-8")
            (diffusers / "transformer" / "config.json").write_text(
                "{}", encoding="utf-8"
            )
            self.adapter.validate_snapshot_layout(diffusers, "FLUX campaign")
            self.assertEqual(self.adapter.model_snapshot_revision(diffusers), revision)

            q4 = diffusers / "q4"
            q4.mkdir()
            (q4 / "config.json").write_text("{}", encoding="utf-8")
            self.adapter.validate_snapshot_layout(q4, "Wan campaign")
            self.assertEqual(self.adapter.model_snapshot_revision(q4), revision)
            identity = self.adapter.snapshot_identity(q4)
            (diffusers / "unselected.safetensors").write_bytes(b"sibling")
            self.assertEqual(self.adapter.snapshot_identity(q4), identity)
            (q4 / "weights.safetensors").write_bytes(b"selected")
            self.assertNotEqual(self.adapter.snapshot_identity(q4), identity)

    def test_worker_assembled_vace_snapshot_is_accepted_and_hashes_linked_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hf" / "transformer"
            source.mkdir(parents=True)
            (source / "config.json").write_text("{}", encoding="utf-8")
            (source / "diffusion_pytorch_model.safetensors").write_bytes(b"vace-1.3b")
            assembled = root / "wan_vace"
            assembled.mkdir()
            (assembled / "transformer").symlink_to(source, target_is_directory=True)
            for name in ("t5_encoder.safetensors", "vae.safetensors", "tokenizer.json"):
                (assembled / name).write_bytes(name.encode("utf-8"))
            revision = "3" * 40
            (assembled / ".snapshot-revision").write_text(revision, encoding="utf-8")
            self.adapter.validate_snapshot_layout(
                assembled, "Wan manifest wan_vace", "wan_vace", "mlx-metal"
            )
            # The assembled layout is scoped per route: VACE-Fun also needs its low-noise expert,
            # and a non-VACE route never takes it.
            with self.assertRaisesRegex(ValueError, "assembled wan2_2_vace_fun_14b"):
                self.adapter.validate_snapshot_layout(
                    assembled, "Wan manifest", "wan2_2_vace_fun_14b", "mlx-metal"
                )
            with self.assertRaisesRegex(ValueError, "exact component/tier root"):
                self.adapter.validate_snapshot_layout(assembled, "Wan manifest", "wan2_2_t2v_14b")
            self.assertEqual(self.adapter.model_snapshot_revision(assembled), revision)
            identity = self.adapter.snapshot_identity(assembled)
            (source / "diffusion_pytorch_model.safetensors").write_bytes(b"vace-14b")
            self.assertNotEqual(self.adapter.snapshot_identity(assembled), identity)
            # A dangling link or a link cycle is refused, never skipped.
            (assembled / "dangling").symlink_to(root / "missing")
            with self.assertRaisesRegex(ValueError, "broken link"):
                self.adapter.snapshot_identity(assembled)
            (assembled / "dangling").unlink()
            (source / "loop").symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "link cycle"):
                self.adapter.snapshot_identity(assembled)
            (source / "loop").unlink()
            (assembled / "tokenizer.json").unlink()
            with self.assertRaisesRegex(ValueError, "assembled wan_vace"):
                self.adapter.validate_snapshot_layout(
                    assembled, "Wan manifest wan_vace", "wan_vace", "mlx-metal"
                )

    def test_metal_tiered_routes_require_the_product_q4_tier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for route, relative in self.adapter.METAL_PACKED_TIER_CONFIG.items():
                for bits, accepted in ((4, True), (8, False), (None, False)):
                    tier = root / route / str(bits)
                    (tier / relative).parent.mkdir(parents=True, exist_ok=True)
                    (tier / "config.json").write_text("{}", encoding="utf-8")
                    quantization = {} if bits is None else {"quantization": {"bits": bits}}
                    (tier / relative).write_text(json.dumps(quantization), encoding="utf-8")
                    if accepted:
                        self.adapter.validate_snapshot_layout(tier, route, route, "mlx-metal")
                    else:
                        with self.assertRaisesRegex(ValueError, "default q4 tier"):
                            self.adapter.validate_snapshot_layout(tier, route, route, "mlx-metal")
                    # The CUDA lane's Candle snapshots are not the Mac product's tiers.
                    self.adapter.validate_snapshot_layout(tier, route, route, "candle-cuda")

    def test_manifest_rejects_non_product_residency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = self._manifest_fixture(root)
            entries["wan_vace"]["residency_strategy"] = "sequential"
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(entries), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "malformed"):
                self.adapter.load_wan_manifest(manifest)

    def test_manifest_cannot_override_adapter_owned_event_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = self._manifest_fixture(root)
            route = self.adapter.WAN_ROUTES[0]
            coordinate = next(iter(entries[route]["coordinates"]))
            entries[route]["coordinates"][coordinate].extend(("--sc20686-events", "-"))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(entries), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overrides campaign identity"):
                self.adapter.load_wan_manifest(manifest)

    def test_every_file_bearing_route_argument_is_content_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file_path = root / "weights.safetensors"
            file_path.write_bytes(b"weights")
            folder = root / "frames"
            folder.mkdir()
            (folder / "0.png").write_bytes(b"frame")
            arguments = []
            for flag in sorted(self.adapter.FILE_ARGUMENT_FLAGS):
                arguments.extend((flag, str(folder if flag.endswith("-dir") else file_path)))
            hashes, inventory = self.adapter.hash_file_arguments(arguments)
            self.assertEqual(len(hashes), len(self.adapter.FILE_ARGUMENT_FLAGS))
            self.assertEqual(len(inventory), len(self.adapter.FILE_ARGUMENT_FLAGS))
            before = dict(hashes)
            file_path.write_bytes(b"changed")
            after, _ = self.adapter.hash_file_arguments(arguments)
            self.assertNotEqual(before, after)

    def test_model_snapshot_mutation_after_arm_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            config = snapshot / "config.json"
            config.write_text("{}", encoding="utf-8")
            expected = self.adapter.snapshot_identity(snapshot)
            spec = argparse.Namespace(variant="route", name="coordinate", snapshot=snapshot)
            self.adapter.verify_snapshot_identity(spec, expected)
            config.write_text('{"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model snapshot changed during campaign arm"):
                self.adapter.verify_snapshot_identity(spec, expected)

    def test_cancel_arm_requires_product_identity_and_metrics(self):
        row = self.adapter.make_row(
            self.row_args(cancel=True), self.config(), "b" * 64, 1,
            self.flux_events(cancel=True),
        )
        self.assertEqual(row["arm"], "cancel")
        broken = self.flux_events(cancel=True)
        next(event for event in broken if event["phase"] == "metrics").pop(
            "minimum_cache_reads"
        )
        with self.assertRaisesRegex(ValueError, "metrics"):
            self.adapter.make_row(
                self.row_args(cancel=True), self.config(), "b" * 64, 1, broken
            )

    def test_wan_reuse_is_per_cache_and_every_cache_is_released(self):
        expected = self.coverage["families"]["wan"]["wan2_2_ti2v_5b"][
            "coordinates"
        ]["square-17f"]
        geometry = {
            **expected, "layers": 2, "heads": 2, "head_dimension": 64,
            "sq": 1024, "skv": 128, "dtype": "BF16", "mask": "none",
            "rope": "none",
        }
        base = {"sample_kind": "allocator", "peak_bytes": 10 * 1024**3}
        def measured(before, transient=0):
            return {
                "allocator_before_bytes": before,
                "allocator_after_bytes": before,
                "allocator_high_bytes": before + transient,
                "allocator_reserved_bytes": 10 * 1024**3,
                "allocator_measurement_available": True,
            }
        events = [
            {"phase": "metadata", "source_ref": "a" * 40,
             "model_snapshot_revision": "b" * 40,
             "residency_strategy": "sequential",
             "snapshot_sha256": "b" * 64, "snapshot_bytes": 1,
             "variant": "wan2_2_ti2v_5b", "geometry": geometry,
             "real_weights": True, "full_generation": True,
             "attention_kind": "cross"},
            {"phase": "generation-start", **base},
            {"phase": "cross-kv-created", "cache_id": 1,
             "persistent_bytes": 400 * 1024**2,
             "candidate_persistent_bytes": 100 * 1024**2, **base},
            {"phase": "cross-kv-created", "cache_id": 2,
             "persistent_bytes": 400 * 1024**2,
             "candidate_persistent_bytes": 100 * 1024**2, **base},
            {"phase": "cross-kv-read", "cache_id": 1,
             "transient_bytes": 64 * 1024**2, "reused": 1,
             **measured(1024, 64 * 1024**2),
             **base},
            {"phase": "cross-kv-read", "cache_id": 2,
             "transient_bytes": 64 * 1024**2, "reused": 1,
             **measured(2048, 64 * 1024**2),
             **base},
            {"phase": "cross-kv-read", "cache_id": 1,
             "transient_bytes": 64 * 1024**2, "reused": 1,
             **measured(3072, 64 * 1024**2),
             **base},
            {"phase": "cross-kv-read", "cache_id": 2,
             "transient_bytes": 64 * 1024**2, "reused": 1,
             **measured(4096, 64 * 1024**2),
             **base},
            {"phase": "cross-kv-released", "cache_id": 1,
             "persistent_bytes": 400 * 1024**2,
             "candidate_persistent_bytes": 100 * 1024**2, **measured(4096), **base},
            {"phase": "cross-kv-released", "cache_id": 2,
             "persistent_bytes": 400 * 1024**2,
             "candidate_persistent_bytes": 100 * 1024**2, **measured(2048), **base},
            {"phase": "generation-end", **base},
            {"phase": "metrics", "current_persistent_bytes": 800 * 1024**2,
             "current_read_transient_bytes": 64 * 1024**2,
             "candidate_persistent_bytes": 200 * 1024**2,
             "candidate_read_transient_bytes": 64 * 1024**2,
             "generation_duration_ms": 1000, "cache_read_duration_ms": 100,
             "joint_attention_context_duration_ms": 0,
             "reference_runtime_attribution_available": True,
             "reused_requests": 4, "minimum_cache_reads": 2, **base},
            {"phase": "invalidated", **base},
            {"phase": "released", **measured(1024), **base},
            {"phase": "process-sample", "sample_kind": "process",
             "peak_bytes": 10 * 1024**3},
        ]
        args = argparse.Namespace(
            fake=False, family="wan", variant="wan2_2_ti2v_5b",
            coordinate_name="square-17f", cancel_campaign=False,
        )
        row = self.adapter.make_row(args, self.config(), "b" * 64, 1, events)
        self.assertEqual(row["minimum_cache_reads"], 2)
        metrics = next(event for event in events if event["phase"] == "metrics")
        metrics["candidate_persistent_bytes"] -= 1
        with self.assertRaisesRegex(ValueError, "exact simultaneous residency"):
            self.adapter.make_row(args, self.config(), "b" * 64, 1, events)
        metrics["candidate_persistent_bytes"] += 1
        first_read = next(event for event in events if event["phase"] == "cross-kv-read")
        first_read["allocator_high_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "active allocator high-water"):
            self.adapter.make_row(args, self.config(), "b" * 64, 1, events)
        first_read["allocator_high_bytes"] -= 1
        next(event for event in events if event["phase"] == "metrics")[
            "minimum_cache_reads"
        ] = 4
        with self.assertRaisesRegex(ValueError, "per-cache minimum"):
            self.adapter.make_row(args, self.config(), "b" * 64, 1, events)
        next(event for event in events if event["phase"] == "metrics")[
            "minimum_cache_reads"
        ] = 2
        events = [
            event for event in events
            if not (event.get("phase") == "cross-kv-released" and event.get("cache_id") == 2)
        ]
        with self.assertRaisesRegex(ValueError, "release every exact cache"):
            self.adapter.make_row(args, self.config(), "b" * 64, 1, events)

    def test_atomic_publication_removes_staging_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "campaign"
            coordinates = [object(), object()]
            calls = 0

            def runner(_coordinate, _arm):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise ValueError("boom")
                media = Path(directory) / "first-media"
                media.write_bytes(b"media")
                return self.adapter.CampaignRun(
                    [], b"", b"", ("producer", "--out", str(media)), ({},),
                    media_output=media,
                )

            with self.assertRaisesRegex(ValueError, "boom"):
                self.adapter.publish_campaign(
                    coordinates, runner, lambda *_: {}, destination
                )
            self.assertFalse(destination.exists())
            self.assertFalse(any(Path(directory).glob(".campaign.staging-*")))

    def test_resume_preserves_validated_normal_arm_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resume = root / "resume"
            spec = SimpleNamespace(variant="wan_vace")
            media = root / "media"
            media.mkdir()
            (media / "frame.png").write_bytes(b"real output bytes")
            event = self.adapter.canonical({"phase": "metadata"})
            calls = []
            policy = self.runner_safety()["safety_policy"]
            identity_raw = self.adapter.canonical({"safetyPolicySha256": policy.sha256})
            identity_sha = self.adapter.digest(identity_raw)

            def runner(_spec, arm):
                calls.append(arm)
                if arm == "cancel":
                    raise ValueError("interrupted")
                return self.adapter.CampaignRun(
                    [{"phase": "metadata"}, {"phase": "process-sample", "sample_kind": "process", "peak_bytes": 1}],
                    b"stdout", b"", ("producer", "--sc20686-events", str(root / "events"), "--out", str(media)),
                    ({"phase": "process-sample", "sample_kind": "process", "peak_bytes": 1},),
                    event, media, None,
                    {"pid": 123, "exitCode": 0, "ownedProcessGroupReaped": True,
                     "admission": self.measured_admission(self.runner_safety()["safety_policy"])},
                )

            def row_builder(_spec, arm, _events, _hashes):
                return {"family": "wan", "variant": "wan_vace", "coordinate_name": "small", "arm": arm}

            resume.mkdir()
            (resume / "units").mkdir()
            (resume / "identity.json").write_bytes(identity_raw)
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "interrupted"):
                    self.adapter.publish_campaign(
                        [spec], runner, row_builder, root / "final",
                        resume_root=resume, resume_identity_sha=identity_sha,
                    )
            self.assertEqual(calls, ["normal", "cancel", "cancel"])
            unit = resume / "units" / "run-00-normal"
            self.assertTrue((unit / "record.json").is_file())
            self.assertEqual(self.adapter._load_unit(
                resume, "run-00-normal", identity_sha, "wan_vace", "normal",
            ).supervision["admission"]["mode"], "runtime-guarded")
            # A unit sealed without its runtime-guarded admission is neither saved nor resumed.
            supervision = json.loads((unit / "supervision.json").read_bytes())
            del supervision["admission"]
            (unit / "supervision.json").write_bytes(self.adapter.canonical(supervision))
            record = self.adapter.canonical({
                "schema": "sc-20686-resume-unit-v1", "identitySha256": identity_sha,
                "stem": "run-00-normal", "files": self.adapter._unit_files(unit),
            })
            (unit / "record.json").write_bytes(record)
            (unit / "record.json.sha256").write_text(
                f"{self.adapter.digest(record)}  record.json\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "invalid-admission"):
                self.adapter._load_unit(resume, "run-00-normal", identity_sha, "wan_vace", "normal")
            unadmitted = self.adapter.CampaignRun(
                [], b"", b"", ("producer", "--sc20686-events", "e", "--out", str(media)), (),
                b"", media, None, supervision,
            )
            with self.assertRaisesRegex(ValueError, "invalid-admission"):
                self.adapter._save_unit(resume, "run-09-normal", identity_sha, "wan_vace", "normal", unadmitted)
            self.assertFalse((resume / "units" / "run-09-normal").exists())
            foreign = dict(supervision, admission={
                **self.adapter.supervisor.runtime_guarded_admission(policy), "policySha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "invalid-admission"):
                self.adapter._save_unit(resume, "run-09-normal", identity_sha, "wan_vace", "normal",
                                        unadmitted.__class__(**{**unadmitted.__dict__, "supervision": foreign}))
            self.assertFalse((resume / "units" / "run-09-normal").exists())
            self.assertFalse((resume / "units" / ".run-09-normal.partial").exists())
            (unit / "stdout").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "corrupted"):
                self.adapter._load_unit(resume, "run-00-normal", identity_sha, "wan_vace", "normal")

    def test_a_refused_completed_arm_retains_its_transcript_unaccepted(self):
        # SC-20686 D2: a 40-minute arm whose transcript the row builder refused vanished with the
        # run's private directory, taking its decode measurements with it.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resume = root / "resume"
            private = root / "sc20686-events-x"
            sealed = private / "sealed-run"
            media = sealed / "media"
            media.mkdir(parents=True)
            (media / "frame.png").write_bytes(b"real output bytes")
            (sealed / "events.jsonl").write_bytes(b'{"phase":"metadata"}\n')
            policy = self.runner_safety()["safety_policy"]
            identity_raw = self.adapter.canonical({"safetyPolicySha256": policy.sha256})
            identity_sha = self.adapter.digest(identity_raw)
            resume.mkdir()
            (resume / "units").mkdir()
            (resume / "identity.json").write_bytes(identity_raw)
            admission = self.measured_admission(policy)

            def runner(_spec, _arm):
                return self.adapter.CampaignRun(
                    [{"phase": "metadata"}], b"stdout", b"",
                    ("producer", "--sc20686-events", str(sealed / "events.jsonl"), "--out", str(media)),
                    ({"phase": "process-sample", "sample_kind": "process", "peak_bytes": 1},),
                    b'{"phase":"metadata"}\n', media, private,
                    {"pid": 321, "exitCode": 0, "ownedProcessGroupReaped": True, "admission": admission},
                )

            def row_builder(*_args):
                raise ValueError("entrypoint must identify a real product cross-attention route")

            with self.assertRaisesRegex(ValueError, "real product cross-attention route.*retained at"):
                self.adapter.publish_campaign(
                    [SimpleNamespace(variant="wan2_2_ti2v_5b")], runner, row_builder, root / "final",
                    resume_root=resume, resume_identity_sha=identity_sha,
                )
            [retained] = (resume / "failed").glob("incomplete-*")
            self.assertFalse(private.exists())
            self.assertEqual((retained / "sealed-run" / "events.jsonl").read_bytes(), b'{"phase":"metadata"}\n')
            record = json.loads((retained / "unaccepted.json").read_bytes())
            self.assertEqual(
                (record["accepted"], record["outcome"], record["reason"], record["pid"], record["coordinate"]),
                (False, "failed", "invalid-evidence", 321, "wan2_2_ti2v_5b/normal"),
            )
            self.assertFalse((resume / "units" / "run-00-normal").exists())

    def test_operator_stop_halts_between_arms_and_resume_completes_the_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resume = root / "resume"
            spec = SimpleNamespace(variant="wan_vace")
            media = root / "media"
            media.mkdir()
            (media / "frame.png").write_bytes(b"real output bytes")
            event = self.adapter.canonical({"phase": "metadata"})
            policy = self.runner_safety()["safety_policy"]
            identity_raw = self.adapter.canonical({"safetyPolicySha256": policy.sha256})
            identity_sha = self.adapter.digest(identity_raw)
            stop = resume / "STOP"
            calls, touch_during = [], {"normal"}
            sample = {"phase": "process-sample", "sample_kind": "process", "peak_bytes": 1}

            def runner(_spec, arm):
                calls.append(arm)
                if arm in touch_during:
                    stop.write_bytes(b"")  # the operator asks while this arm is running
                output = media if arm == "normal" else root / f"absent-{arm}"
                return self.adapter.CampaignRun(
                    [{"phase": "metadata"}, sample], b"stdout", b"",
                    ("producer", "--sc20686-events", str(root / "events"), "--out", str(output)),
                    (sample,), event, output, None,
                    {"pid": 100 + len(calls), "exitCode": 0, "ownedProcessGroupReaped": True,
                     "admission": self.measured_admission(policy)},
                )

            def row_builder(_spec, arm, _events, _hashes):
                return {"family": "wan", "variant": "wan_vace", "coordinate_name": "small", "arm": arm}

            seen = []

            def publish():
                return self.adapter.publish_campaign(
                    [spec], runner, row_builder, root / "final",
                    resume_root=resume, resume_identity_sha=identity_sha,
                    stop_files=self.adapter.supervisor.operator_stop_files(None, resume),
                    on_unit=lambda coordinate, arm, run: seen.append(
                        (coordinate.variant, arm, run.supervision["pid"])),
                )

            resume.mkdir()
            (resume / "units").mkdir()
            (resume / "identity.json").write_bytes(identity_raw)
            stop.write_bytes(b"")
            with self.assertRaises(self.adapter.supervisor.OperatorStop) as caught:
                publish()
            self.assertEqual(calls, [])
            self.assertEqual((caught.exception.index, caught.exception.before), (0, "run-00-normal"))
            self.assertEqual(json.loads(caught.exception.record.read_bytes())["status"], "stopped-by-operator")
            stop.unlink()
            with self.assertRaises(self.adapter.supervisor.OperatorStop) as caught:
                publish()
            # The running arm finished and was sealed; the next arm never started.
            self.assertEqual(calls, ["normal"])
            self.assertEqual((caught.exception.index, caught.exception.before), (1, "run-00-cancel"))
            self.assertTrue((resume / "units" / "run-00-normal" / "record.json").is_file())
            self.assertFalse((resume / "units" / "run-00-cancel").exists())
            touch_during.clear()
            stop.unlink()
            # Resume: the sealed arm is reused, only the stopped arm runs, and the loop completes
            # into publication (which this fixture deliberately refuses for lacking inputs).
            with self.assertRaisesRegex(ValueError, "requires sealed resolved inputs"):
                publish()
            self.assertEqual(calls, ["normal", "cancel"])
            self.assertTrue((resume / "units" / "run-00-cancel" / "record.json").is_file())
            # Every completed arm, fresh or resumed, reaches the next arm's admission estimate.
            self.assertEqual(seen, [("wan_vace", "normal", 101), ("wan_vace", "normal", 101),
                                    ("wan_vace", "cancel", 102)])

    def test_resume_directory_tolerates_the_operator_stop_file(self):
        with tempfile.TemporaryDirectory() as directory:
            resume = Path(directory) / "resume"
            policy = self.runner_safety()["safety_policy"]
            resolved = {"schema": "test"}
            first = self.adapter._prepare_resume(resume, resolved, policy)
            (resume / "STOP").write_bytes(b"")
            (resume / "halt").write_bytes(b"")
            (resume / "logs").mkdir()
            stops = self.adapter.supervisor.operator_stop_files(resume / "halt", resume)
            self.assertEqual(self.adapter._prepare_resume(resume, resolved, policy, stops), first)
            (resume / "stray").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "unexpected entries"):
                self.adapter._prepare_resume(resume, resolved, policy, stops)

    def test_complete_bundle_is_reproducible_and_tampering_fails(self):
        coordinates = []
        for family, variants in self.coverage["families"].items():
            for variant, variant_spec in variants.items():
                for name in variant_spec["coordinates"]:
                    coordinates.append(argparse.Namespace(
                        family=family, variant=variant, name=name,
                    ))
        resolved_coordinates = []
        route_hashes = {}
        for coordinate in coordinates:
            document = {
                "route": coordinate.variant, "coordinate": coordinate.name,
                "entrypoint": "product-entrypoint", "entrypoint_sha256": "d" * 64,
                "snapshot": "/snapshot", "model_snapshot_revision": "b" * 40,
                "residency_strategy": self.adapter.PRODUCT_RESIDENCY[coordinate.variant],
                "args": [], "input_files": [],
            }
            route_hash = self.adapter.digest(self.adapter.canonical(document))
            route_hashes[(coordinate.family, coordinate.variant, coordinate.name)] = route_hash
            resolved_coordinates.append({
                "family": coordinate.family, "variant": coordinate.variant,
                "name": coordinate.name, "entrypoint": "product-entrypoint",
                "entrypoint_sha256": "d" * 64, "snapshot": "/snapshot",
                "snapshot_sha256": "c" * 64, "snapshot_bytes": 1, "args": [],
                "model_snapshot_revision": "b" * 40,
                "residency_strategy": self.adapter.PRODUCT_RESIDENCY[coordinate.variant],
                "route_manifest_sha256": route_hash, "input_files": [],
            })
        resolved = {
            "schema": "sc-20686-resolved-inputs-v4",
            "backend": "candle-cuda",
            "inference_revision": "a" * 40,
            "coordinates": resolved_coordinates,
        }

        def runner(coordinate, arm):
            expected = self.coverage["families"][coordinate.family][coordinate.variant][
                "coordinates"
            ][coordinate.name]
            geometry = {
                **expected, "layers": 2, "heads": 2, "head_dimension": 64,
                "sq": 1024, "skv": 128, "dtype": "BF16", "mask": "none",
                "rope": "none",
            }
            exact_dense = self.reducer.dense_reference_kv_bytes(geometry)
            transient = exact_dense if coordinate.family == "flux2-klein" else 1
            measured = {
                "allocator_before_bytes": 1024,
                "allocator_after_bytes": 1024,
                "allocator_high_bytes": 1024 + transient,
                "allocator_reserved_bytes": 10 * 1024**3,
                "allocator_measurement_available": True,
            }
            metadata = {
                "phase": "metadata", "sample_kind": "allocator",
                "peak_bytes": 10 * 1024**3,
                "cancellation_armed": arm == "cancel",
            }
            if arm == "cancel":
                metadata["cancellation_arm_id"] = (
                    f"test:{coordinate.variant}:{coordinate.name}"
                )
            process = {
                "phase": "process-sample", "sample_kind": "process",
                "peak_bytes": 10 * 1024**3, "at_ns": 1,
            }
            exact_candidate = self.reducer.packed_group32_kv_bytes(1, 2, 128, 64)
            created = {
                "phase": "cross-kv-created", "sample_kind": "allocator",
                "peak_bytes": 10 * 1024**3,
                "candidate_persistent_bytes": exact_candidate,
            }
            if coordinate.family == "flux2-klein":
                created.update({
                    "operation": "DoubleAttention::to_k/to_v(reference-slice)",
                    "transient_bytes": exact_dense,
                })
            read = {
                "phase": "cross-kv-read", "sample_kind": "allocator",
                "peak_bytes": 10 * 1024**3, "transient_bytes": transient,
                "operation": (
                    "DoubleAttention::attention(joint-context-non-attributable)"
                    if coordinate.family == "flux2-klein" else "wan-cross-attention"
                ),
                **measured,
            }
            release_events = []
            if coordinate.family == "wan":
                release_events.append({
                    "phase": "cross-kv-released", "sample_kind": "allocator",
                    "peak_bytes": 10 * 1024**3, **measured,
                })
            released = {
                "phase": "released", "sample_kind": "allocator",
                "peak_bytes": 10 * 1024**3, **measured,
            }
            terminal = "cancelled" if arm == "cancel" else "generation-end"
            observer_events = [
                metadata,
                {"phase": "generation-start"},
                created,
                read,
                dict(read),
                *release_events,
                {"phase": terminal},
                {"phase": "metrics"},
                {"phase": "invalidated"},
                released,
            ]
            run_root = media_root / f"{coordinate.variant}-{coordinate.name}-{arm}"
            run_directory = run_root / "sealed-run"
            run_directory.mkdir(parents=True)
            output_name = (
                "media.png" if coordinate.family == "flux2-klein" else "media"
            )
            media_output = run_directory / output_name
            if arm == "normal":
                if coordinate.family == "flux2-klein":
                    media_output.write_bytes(b"png-media")
                else:
                    media_output.mkdir()
                    (media_output / "frame-0000.png").write_bytes(b"frame-media")
            argv = [
                "product-entrypoint", "--sc20686-campaign", "--sc20686-events",
                str(run_directory / "events.jsonl"),
                "--sc20686-source-ref", "a" * 40,
                "--sc20686-residency",
                self.adapter.PRODUCT_RESIDENCY[coordinate.variant],
                "--snapshot", "/snapshot", "--variant", coordinate.variant,
            ]
            if arm == "cancel":
                argv.append("--sc20686-cancel")
            argv.extend(("--out", str(media_output)))
            return self.adapter.CampaignRun(
                [*observer_events, process],
                b"".join(self.adapter.canonical(event) for event in observer_events), b"",
                tuple(argv), (process,), b"".join(self.adapter.canonical(event) for event in observer_events),
                media_output,
            )

        def row_builder(coordinate, arm, events, evidence_hashes):
            expected = self.coverage["families"][coordinate.family][coordinate.variant][
                "coordinates"
            ][coordinate.name]
            geometry = {
                **expected, "layers": 2, "heads": 2, "head_dimension": 64,
                "sq": 1024, "skv": 128, "dtype": "BF16", "mask": "none",
                "rope": "none",
            }
            exact_candidate = self.reducer.packed_group32_kv_bytes(
                geometry["batch"], geometry["heads"], geometry["skv"],
                geometry["head_dimension"],
            )
            exact_dense = self.reducer.dense_reference_kv_bytes(geometry)
            return {
                "producer": self.adapter.PRODUCER, "backend": "candle-cuda",
                "family": coordinate.family,
                "variant": coordinate.variant, "coordinate_name": coordinate.name,
                "coordinate_id": self.adapter.digest(self.adapter.canonical(geometry))[:16],
                "arm": arm, "source_ref": "a" * 40,
                "model_snapshot_revision": "b" * 40,
                "residency_strategy": self.adapter.PRODUCT_RESIDENCY[coordinate.variant],
                "route_manifest_sha256": route_hashes[
                    (coordinate.family, coordinate.variant, coordinate.name)
                ],
                "source_map_sha256": self.source_map_hash,
                "model_snapshot_sha256": "c" * 64, "model_snapshot_bytes": 1,
                "input_file_sha256": {},
                "evidence_artifact_sha256": evidence_hashes,
                "geometry": geometry,
                "media_manifest_sha256": next(
                    item_hash for name, item_hash in evidence_hashes.items()
                    if name.endswith(".media.json")
                ),
                "lifecycle": {
                    "created": 1, "reused": 2, "invalidated": 1,
                    "cancelled": int(arm == "cancel"), "released": 1,
                },
                "allocator_samples": [
                    {"phase": event["phase"], "peak_bytes": event["peak_bytes"]}
                    for event in events if event.get("sample_kind") == "allocator"
                ],
                "process_samples": [{"phase": "process-sample", "peak_bytes": 10 * 1024**3}],
                "observer_events": events,
                "raw_receipt_sha256": "", "raw_receipt_sidecar_sha256": "",
                "real_weights": True, "full_generation": arm == "normal",
                "attention_kind": "cross",
                "current_persistent_bytes": 0 if coordinate.family == "flux2-klein" else 1,
                "current_read_transient_bytes": (
                    exact_dense if coordinate.family == "flux2-klein" else 1
                ), "candidate_persistent_bytes": (
                    exact_candidate * geometry["layers"]
                    if coordinate.family == "flux2-klein" else exact_candidate
                ),
                "candidate_read_transient_bytes": (
                    exact_dense if coordinate.family == "flux2-klein" else 1
                ), "generation_duration_ms": 1000,
                "cache_read_duration_ms": 0 if coordinate.family == "flux2-klein" else 10,
                "joint_attention_context_duration_ms": (
                    10 if coordinate.family == "flux2-klein" else 0
                ),
                "reference_runtime_attribution_available": coordinate.family == "wan",
                "reused_requests": 2,
                "minimum_cache_reads": 2,
            }

        with tempfile.TemporaryDirectory() as directory:
            media_root = Path(directory) / "run-media"
            destination = Path(directory) / "campaign"
            self.adapter.publish_campaign(
                coordinates, runner, row_builder, destination, {
                    "campaign-inputs.resolved.json": (
                        json.dumps(resolved, indent=2, sort_keys=True) + "\n"
                    ).encode("utf-8"),
                    "safety-policy.json": b"{\"schemaVersion\":1}\n",
                    "resume-identity.json": b"{\"schema\":\"test\"}\n",
                }
            )
            result = self.reducer.verify_campaign_bundle(destination)
            self.assertEqual(set(result["decisions"]), {"flux2-klein", "wan"})
            normal_media_manifest = json.loads(
                (destination / "run-00-normal.media.json").read_text(encoding="utf-8")
            )
            self.assertEqual(normal_media_manifest["output_kind"], "file")
            self.assertEqual(normal_media_manifest["output_name"], "media.png")
            self.assertEqual(len(normal_media_manifest["files"]), 1)
            media_path = destination / normal_media_manifest["files"][0]["artifact"]
            self.assertTrue(media_path.is_file())
            media_original = media_path.read_bytes()
            media_path.write_bytes(media_original + b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                self.reducer.verify_campaign_bundle(destination)
            media_path.write_bytes(media_original)
            cancel_media_manifest = json.loads(
                (destination / "run-00-cancel.media.json").read_text(encoding="utf-8")
            )
            self.assertEqual(cancel_media_manifest["output_kind"], "absent")
            self.assertEqual(cancel_media_manifest["files"], [])
            markdown = (destination / "campaign.md").read_text(encoding="utf-8")
            self.assertIn("## Sealed source map", markdown)
            self.assertIn("- Group size: `32`", markdown)
            self.assertIn("- Pending key tail: dense f32", markdown)
            self.assertIn("packed-group-affine", json.dumps(result["source_map"]))
            resolved_path = destination / "campaign-inputs.resolved.json"
            resolved_sidecar = destination / "campaign-inputs.resolved.json.sha256"
            campaign_path = destination / "campaign.json"
            campaign_sidecar = destination / "campaign.json.sha256"
            originals = {
                path: path.read_bytes()
                for path in (resolved_path, resolved_sidecar, campaign_path, campaign_sidecar)
            }
            malicious = json.loads(resolved_path.read_text(encoding="utf-8"))
            malicious["coordinates"][0]["entrypoint_sha256"] = "e" * 64
            malicious_raw = (
                json.dumps(malicious, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            malicious_hash = self.adapter.digest(malicious_raw)
            resolved_path.write_bytes(malicious_raw)
            resolved_sidecar.write_text(
                f"{malicious_hash}  campaign-inputs.resolved.json\n", encoding="utf-8"
            )
            campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
            campaign["artifact_sha256"]["campaign-inputs.resolved.json"] = malicious_hash
            campaign_raw = (
                json.dumps(campaign, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            campaign_path.write_bytes(campaign_raw)
            campaign_sidecar.write_text(
                f"{self.adapter.digest(campaign_raw)}  campaign.json\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "cannot be recomputed"):
                self.reducer.verify_campaign_bundle(destination)
            for path, payload in originals.items():
                path.write_bytes(payload)
            transcript = destination / "run-00-normal.stdout"
            transcript.write_bytes(transcript.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                self.reducer.verify_campaign_bundle(destination)


if __name__ == "__main__":
    unittest.main()
