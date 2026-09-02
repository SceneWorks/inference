import argparse
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

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
            "evidence_artifact_sha256": {"run.stdout": "e" * 64},
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
            with self.assertRaisesRegex(ValueError, "timed out"):
                self.adapter.run_entrypoint(
                    executable, snapshot, "route", "normal", "a" * 40,
                    "sequential", timeout_seconds=0.1,
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
                return self.adapter.CampaignRun([], b"", b"", ("producer",), ({},))

            with self.assertRaisesRegex(ValueError, "boom"):
                self.adapter.publish_campaign(
                    coordinates, runner, lambda *_: {}, destination
                )
            self.assertFalse(destination.exists())
            self.assertFalse(any(Path(directory).glob(".campaign.staging-*")))

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
            "schema": "sc-20686-resolved-inputs-v3",
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
            }
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
            release_events.append({
                "phase": "released", "sample_kind": "allocator",
                "peak_bytes": 10 * 1024**3, **measured,
            })
            observer_events = [metadata, created, read, dict(read), *release_events]
            argv = [
                "product-entrypoint", "--sc20686-campaign", "--sc20686-events",
                "/adapter-owned/sealed-run/events.jsonl",
                "--sc20686-source-ref", "a" * 40,
                "--sc20686-residency",
                self.adapter.PRODUCT_RESIDENCY[coordinate.variant],
                "--snapshot", "/snapshot", "--variant", coordinate.variant,
            ]
            if arm == "cancel":
                argv.append("--sc20686-cancel")
            argv.extend(("--out", "/adapter-owned/sealed-run/media"))
            return self.adapter.CampaignRun(
                [*observer_events, process],
                b"".join(self.adapter.canonical(event) for event in observer_events), b"",
                tuple(argv), (process,), b"".join(self.adapter.canonical(event) for event in observer_events),
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
                "producer": self.adapter.PRODUCER, "family": coordinate.family,
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
                "lifecycle": {"created": 1, "reused": 2, "invalidated": 1, "released": 1},
                "allocator_samples": [
                    {"phase": event["phase"], "peak_bytes": event["peak_bytes"]}
                    for event in events if event.get("sample_kind") == "allocator"
                ],
                "process_samples": [{"phase": "process-sample", "peak_bytes": 10 * 1024**3}],
                "observer_events": events,
                "raw_receipt_sha256": "", "raw_receipt_sidecar_sha256": "",
                "real_weights": True, "full_generation": arm == "normal",
                "attention_kind": "cross", "current_persistent_bytes": 1,
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
            destination = Path(directory) / "campaign"
            self.adapter.publish_campaign(
                coordinates, runner, row_builder, destination, {
                    "campaign-inputs.resolved.json": (
                        json.dumps(resolved, indent=2, sort_keys=True) + "\n"
                    ).encode("utf-8"),
                }
            )
            result = self.reducer.verify_campaign_bundle(destination)
            self.assertEqual(set(result["decisions"]), {"flux2-klein", "wan"})
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
