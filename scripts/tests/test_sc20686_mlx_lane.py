"""SC-20686 MLX Metal lane: admission, route mapping, refusal paths, transcript validation, and
sealed-bundle reduction for the darwin-mlx measurement lane (no model execution)."""

import argparse
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ADAPTER = Path(__file__).parents[1] / "sc20686_campaign_adapter.py"
REDUCER = Path(__file__).parents[1] / "sc20686_cache_attribution.py"
ROOT = Path(__file__).parents[2]
GIB = 1024**3
PROMPTS = {
    "ff8b29180ce69f56369bd1222522f384564bc4154035ebb06c1efbdc30e36be5":
        "SC-20686 representative still-motion study",
    "c19800ff5dc04d6cbd7d00d636218c569318b28a3df652255a531c4c3b3de713":
        "SC-20686 representative long-motion study",
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MetalLaneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load(ADAPTER, "sc20686_adapter_mlx_test")
        cls.reducer = load(REDUCER, "sc20686_reducer_mlx_test")
        cls.coverage = json.loads(cls.adapter.COVERAGE.read_text(encoding="utf-8"))
        cls.source_map = json.loads(cls.adapter.SOURCE_MAP.read_text(encoding="utf-8"))
        cls.source_map_hash = cls.adapter.digest(cls.adapter.SOURCE_MAP.read_bytes())
        cls.mlx_coverage = cls.reducer.lane_coverage(cls.coverage, "mlx-metal")

    # --- transcript fixtures -------------------------------------------------------------------

    def entry(self, variant, backend="mlx-metal"):
        return self.source_map["lanes"][backend]["variants"][variant]

    def geometry(self, variant, coordinate=None):
        family = self.entry(variant)["family"]
        coordinates = self.mlx_coverage[family][variant]["coordinates"]
        name = coordinate or sorted(coordinates)[0]
        return name, {
            **coordinates[name], "layers": 2, "heads": 2, "head_dimension": 64,
            "sq": 1024, "skv": 128, "dtype": "BF16", "mask": "none", "rope": "none",
        }

    def mlx_events(self, variant, cancel=False, coordinate=None):
        entry = self.entry(variant)
        kind, attributable = entry["cache_kind"], entry["runtime_attribution"]
        operations = entry["operations"]
        _name, geometry = self.geometry(variant, coordinate)
        base = {"backend": "mlx-metal", "sample_kind": "allocator", "peak_bytes": 10 * GIB}

        def alloc(before, high, after=None):
            after = before if after is None else after
            return {
                "allocator_before_bytes": before, "allocator_after_bytes": after,
                "allocator_high_bytes": high, "allocator_reserved_bytes": high + 1024,
                "allocator_measurement_available": True,
            }

        def window(name, index=0):
            return {
                "phase": "phase-window", "window": name, "window_index": index,
                "phys_footprint_bytes": 4 * GIB, "phys_footprint_peak_bytes": 5 * GIB,
                **alloc(GIB, 2 * GIB), **base,
            }

        metadata = {
            "phase": "metadata", "source_ref": "a" * 40, "snapshot_sha256": "b" * 64,
            "model_snapshot_revision": "b" * 40,
            "residency_strategy": self.adapter.PRODUCT_RESIDENCY[variant],
            "snapshot_bytes": 1, "variant": variant, "geometry": geometry,
            "real_weights": True, "full_generation": not cancel, "attention_kind": "cross",
            "cancellation_armed": cancel, **base,
        }
        if cancel:
            metadata["cancellation_arm_id"] = f"mlx-metal:{'a' * 40}:{variant}"
        events = [window("encode"), metadata, {"phase": "generation-start", **base},
                  window("prepare-cache")]
        dense = self.adapter.dense_reference_kv_bytes(geometry)
        if kind == "persistent":
            kv_batch = 2 if entry["family"] == "wan" else 1
            cache_dense = self.adapter.dense_reference_kv_bytes(geometry, kv_batch)
            candidate = self.reducer.packed_group32_kv_bytes(kv_batch, 2, 128, 64)
            for cache_id in (1, 2):
                events.append({
                    "phase": "cross-kv-created", "operation": operations["create"],
                    "cache_id": cache_id, "persistent_bytes": cache_dense,
                    "candidate_persistent_bytes": candidate, "kv_batch": kv_batch,
                    "transient_bytes": 0, **base,
                })
            for cache_id in (1, 2, 1, 2):
                events.append({
                    "phase": "cross-kv-read", "operation": operations["read"],
                    "cache_id": cache_id, "transient_bytes": 1000, "reused": 1,
                    **alloc(GIB, GIB + 1000), **base,
                })
            events.append(window("denoise-step", 1))
            for cache_id in (1, 2):
                events.append({
                    "phase": "cross-kv-released", "operation": "drop", "cache_id": cache_id,
                    "persistent_bytes": cache_dense, "candidate_persistent_bytes": candidate,
                    **alloc(GIB, GIB), **base,
                })
            metrics = {
                "current_persistent_bytes": 2 * cache_dense,
                "candidate_persistent_bytes": 2 * candidate,
                "current_read_transient_bytes": 1000, "candidate_read_transient_bytes": 1000,
                "reused_requests": 4, "minimum_cache_reads": 2,
            }
        else:
            for _forward in range(2):
                for _layer in range(geometry["layers"]):
                    events.append({
                        "phase": "cross-kv-created", "operation": operations["create"],
                        "cache_id": 0, "persistent_bytes": 0, "transient_bytes": dense,
                        "kv_batch": 1, **base,
                    })
                    events.append({
                        "phase": "cross-kv-read", "operation": operations["read"],
                        "cache_id": 0, "transient_bytes": dense, "reused": 1,
                        **alloc(GIB, GIB + dense), **base,
                    })
            events.append(window("denoise-step", 1))
            metrics = {
                "current_persistent_bytes": 0,
                "candidate_persistent_bytes": (
                    self.reducer.packed_group32_kv_bytes(1, 2, 128, 64) * geometry["layers"]
                ),
                "current_read_transient_bytes": dense, "candidate_read_transient_bytes": dense,
                "reused_requests": 2, "minimum_cache_reads": 2,
            }
        if not cancel:
            events.append(window("decode", 1))
        terminal = "cancelled" if cancel else "generation-end"
        events.append({"phase": terminal, **base})
        events.append({
            "phase": "metrics", **base, **metrics, "generation_duration_ms": 1000.0,
            "cache_read_duration_ms": 10.0 if attributable else 0.0,
            "joint_attention_context_duration_ms": 0.0 if attributable else 10.0,
            "reference_runtime_attribution_available": attributable,
        })
        events.append({"phase": "invalidated", **base})
        events.append({"phase": "released", **alloc(GIB, GIB), **base})
        events.append({"phase": "process-sample", "sample_kind": "process", "peak_bytes": 12 * GIB})
        return events

    def config(self, variant):
        return {
            "inference_revision": "a" * 40, "model_snapshot_revision": "b" * 40,
            "residency_strategy": self.adapter.PRODUCT_RESIDENCY[variant],
            "route_manifest_sha256": "c" * 64, "source_map_sha256": self.source_map_hash,
            "input_file_sha256": {}, "evidence_artifact_sha256": {"run.media.json": "e" * 64},
        }

    def row_args(self, variant, cancel=False, backend="mlx-metal", coordinate=None):
        name, _geometry = self.geometry(variant, coordinate)
        return argparse.Namespace(
            fake=False, family=self.entry(variant)["family"], variant=variant,
            coordinate_name=name, cancel_campaign=cancel, backend=backend,
        )

    def make_row(self, variant, events, cancel=False, backend="mlx-metal"):
        return self.adapter.make_row(
            self.row_args(variant, cancel, backend), self.config(variant), "b" * 64, 1, events,
        )

    # --- lane registration and coverage --------------------------------------------------------

    def test_policy_backend_selects_exactly_one_lane(self):
        self.assertEqual(
            self.adapter.LANE_BY_POLICY,
            {"darwin-mlx": "mlx-metal", "linux-cuda": "candle-cuda", "windows-cuda": "candle-cuda"},
        )
        self.assertEqual(tuple(self.source_map["lanes"]), self.reducer.LANES)

    def test_metal_coverage_is_the_cuda_matrix_plus_the_mlx_only_kv_route(self):
        cuda = self.reducer.lane_coverage(self.coverage, "candle-cuda")
        for family, variants in cuda.items():
            for variant, spec in variants.items():
                # Identical native geometry: no Metal coordinate is narrowed or re-shaped.
                self.assertEqual(self.mlx_coverage[family][variant], spec, variant)
        extra = {
            (family, variant)
            for family, variants in self.mlx_coverage.items()
            for variant in variants if variant not in cuda.get(family, {})
        }
        self.assertEqual(extra, {("flux2-klein", "flux2_klein_9b_kv_edit")})
        self.assertEqual(
            self.mlx_coverage["flux2-klein"]["flux2_klein_9b_kv_edit"],
            cuda["flux2-klein"]["flux2_klein_9b_edit"],
        )

    def test_metal_route_to_entrypoint_and_residency_map(self):
        expected = {
            "flux2_klein_9b_edit": ("sc20686_flux2_edit", "recomputed", "sequential"),
            "flux2_klein_9b_kv_edit": ("sc20686_flux2_edit", "persistent", "sequential"),
            "wan2_2_ti2v_5b": ("sc20686_wan", "persistent", "sequential"),
            "wan2_2_t2v_14b": ("sc20686_wan", "persistent", "sequential"),
            "wan2_2_i2v_14b": ("sc20686_wan", "persistent", "sequential"),
            "wan_vace": ("sc20686_wan", "recomputed", "resident"),
            "wan2_2_vace_fun_14b": ("sc20686_wan", "recomputed", "sequential"),
        }
        actual = {
            variant: (entry["entrypoint_stem"], entry["cache_kind"], entry["product_residency"])
            for variant, entry in self.source_map["lanes"]["mlx-metal"]["variants"].items()
        }
        self.assertEqual(actual, expected)
        # The CUDA lane is untouched: same Candle entrypoints, same frozen kinds.
        cuda = self.source_map["lanes"]["candle-cuda"]["variants"]
        for route, stem in self.adapter.WAN_ENTRYPOINT_STEMS.items():
            self.assertEqual(cuda[route]["entrypoint_stem"], stem)
            self.assertEqual(cuda[route]["cache_kind"], "persistent")
        self.assertEqual(cuda["flux2_klein_9b_edit"]["cache_kind"], "recomputed")
        self.assertNotIn("flux2_klein_9b_kv_edit", cuda)

    def _wan_manifest(self, root, stem_for):
        image = root / "image.png"
        image.write_bytes(b"image")
        reference = root / "reference.png"
        reference.write_bytes(b"reference")
        for name in ("control", "mask"):
            (root / name).mkdir(exist_ok=True)
            (root / name / "0.png").write_bytes(name.encode())
        entries = {}
        for route in self.adapter.WAN_ROUTES:
            binary = root / "bin" / stem_for(route)
            binary.parent.mkdir(exist_ok=True)
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            binary.chmod(0o755)
            snapshot = root / route / ("1" * 40) / "q4"
            snapshot.mkdir(parents=True, exist_ok=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            coordinates = {}
            for name, expected in self.mlx_coverage["wan"][route]["coordinates"].items():
                width, height = expected["resolution"].split("x")
                values = [
                    "--width", width, "--height", height, "--frames", str(expected["frames"]),
                    "--prompt", PROMPTS[expected["prompt"]], "--guidance", expected["guidance"],
                    "--steps", "4",
                ]
                if route == "wan2_2_i2v_14b":
                    values[0:0] = ["--image", str(image)]
                if route in ("wan_vace", "wan2_2_vace_fun_14b"):
                    values[0:0] = ["--control-dir", str(root / "control"),
                                   "--mask-dir", str(root / "mask")]
                    if expected["reference_count"]:
                        values[0:0] = ["--reference", str(reference)]
                coordinates[name] = values
            entries[route] = {
                "provider_id": route, "entrypoint": str(binary), "snapshot": str(snapshot),
                "residency_strategy": self.adapter.PRODUCT_RESIDENCY[route],
                "coordinates": coordinates,
            }
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(entries), encoding="utf-8")
        return manifest

    def test_metal_manifest_maps_every_wan_route_to_the_mlx_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._wan_manifest(root, lambda _route: "sc20686_wan")
            loaded = self.adapter.load_wan_manifest(manifest, "mlx-metal")
            self.assertEqual(set(loaded), set(self.adapter.WAN_ROUTES))
            self.assertTrue(all(
                spec.entrypoint.stem == "sc20686_wan"
                for specs in loaded.values() for spec in specs.values()
            ))
            # A Candle entrypoint cannot measure the Metal lane, nor the MLX one the CUDA lane.
            with self.assertRaisesRegex(ValueError, "does not match registered route"):
                self.adapter.load_wan_manifest(manifest, "candle-cuda")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._wan_manifest(root, lambda route: self.adapter.WAN_ENTRYPOINT_STEMS[route])
            self.adapter.load_wan_manifest(manifest, "candle-cuda")
            with self.assertRaisesRegex(ValueError, "does not match registered route"):
                self.adapter.load_wan_manifest(manifest, "mlx-metal")

    def _flux_inputs(self, root, stem):
        entrypoint = root / stem
        entrypoint.write_text("#!/bin/sh\n", encoding="utf-8")
        entrypoint.chmod(0o755)
        snapshots = {}
        for name, revision in (("edit", "2" * 40), ("kv", "3" * 40)):
            snapshot = root / name / revision / "q4"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            snapshots[name] = snapshot
        (root / "ref.png").write_bytes(b"ref")
        (root / "ref2.png").write_bytes(b"ref2")
        return entrypoint, snapshots, root / "ref.png", root / "ref2.png"

    def test_metal_flux_coordinates_add_the_kv_route_and_refuse_narrowing(self):
        with tempfile.TemporaryDirectory() as directory:
            entrypoint, snapshots, ref, ref2 = self._flux_inputs(Path(directory), "sc20686_flux2_edit")
            specs = self.adapter.flux_coordinates(
                entrypoint, snapshots["edit"], ref, ref2, "mlx-metal", snapshots["kv"],
            )
            self.assertEqual(
                sorted((spec.variant, spec.name) for spec in specs),
                sorted(
                    (variant, name)
                    for variant in ("flux2_klein_9b_edit", "flux2_klein_9b_kv_edit")
                    for name in ("edit-512-ref1-cfg1", "edit-768x512-ref2-cfg2")
                ),
            )
            kv = [spec for spec in specs if spec.variant == "flux2_klein_9b_kv_edit"]
            self.assertTrue(all(spec.model_snapshot_revision == "3" * 40 for spec in kv))
            with self.assertRaisesRegex(ValueError, "requires a flux2_klein_9b_kv_edit snapshot"):
                self.adapter.flux_coordinates(entrypoint, snapshots["edit"], ref, ref2, "mlx-metal")
            with self.assertRaisesRegex(ValueError, "no flux2_klein_9b_kv_edit route"):
                self.adapter.flux_coordinates(
                    entrypoint, snapshots["edit"], ref, ref2, "candle-cuda", snapshots["kv"],
                )
        with tempfile.TemporaryDirectory() as directory:
            entrypoint, snapshots, ref, ref2 = self._flux_inputs(Path(directory), "flux2-edit")
            cuda = self.adapter.flux_coordinates(entrypoint, snapshots["edit"], ref, ref2)
            self.assertEqual({spec.variant for spec in cuda}, {"flux2_klein_9b_edit"})
            with self.assertRaisesRegex(ValueError, "does not match registered route"):
                self.adapter.flux_coordinates(
                    entrypoint, snapshots["edit"], ref, ref2, "mlx-metal", snapshots["kv"],
                )

    def _run_main(self, argv):
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", ["adapter", *argv]), contextlib.redirect_stderr(stderr):
            code = self.adapter.main()
        return code, stderr.getvalue()

    def _policy(self, root, backend):
        policy = {
            "schemaVersion": 1, "backend": backend, "deadlineSeconds": 600, "pollMillis": 100,
            "termGraceMillis": 1000, "hostFreeReserveBytes": 1024, "childFootprintCapBytes": 2**30,
            "stdoutCapBytes": 2**20, "stderrCapBytes": 2**20, "eventCapBytes": 2**24,
        }
        if backend != "darwin-mlx":
            policy.update({
                "cudaDeviceUuid": "GPU-12345678-1234-1234-1234-123456789abc",
                "gpuFreeReserveBytes": 1024, "childGpuCapBytes": 2**30,
            })
        path = root / f"{backend}.json"
        path.write_text(json.dumps(policy), encoding="utf-8")
        return path

    def test_main_admits_darwin_mlx_and_refuses_before_any_launch(self):
        head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._wan_manifest(root, lambda _route: "sc20686_wan")
            entrypoint, snapshots, ref, ref2 = self._flux_inputs(root, "sc20686_flux2_edit")
            common = [
                "--campaign", "--matrix", "--inference-revision", head,
                "--wan-manifest", str(manifest), "--flux-entrypoint", str(entrypoint),
                "--flux-snapshot", str(snapshots["edit"]), "--flux-reference", str(ref),
                "--flux-reference2", str(ref2), "--matrix-output", str(root / "out"),
            ]
            launched = mock.patch.object(
                self.adapter, "publish_campaign",
                side_effect=AssertionError("refusal must precede any launch"),
            )
            with launched:
                # darwin-mlx is admitted as the Metal lane, but the kv route may not be dropped.
                code, stderr = self._run_main([
                    *common, "--safety-policy", str(self._policy(root, "darwin-mlx")),
                    "--resume-dir", str(root / "resume-mlx"),
                ])
                self.assertEqual(code, 1)
                self.assertIn("requires a flux2_klein_9b_kv_edit snapshot", stderr)
                # The CUDA lane refuses the MLX-only route and the MLX entrypoints.
                code, stderr = self._run_main([
                    *common, "--flux-kv-snapshot", str(snapshots["kv"]),
                    "--safety-policy", str(self._policy(root, "linux-cuda")),
                    "--resume-dir", str(root / "resume-cuda"),
                ])
                self.assertEqual(code, 1)
                self.assertIn("does not match registered route", stderr)
            self.assertFalse((root / "resume-mlx").exists())
            self.assertFalse((root / "out").exists())

            # A complete Metal matrix reaches publication with 7 routes x 2 coordinates.
            captured = {}

            def publish(coordinates, _runner, _builder, destination, inputs, **_kwargs):
                captured["coordinates"] = coordinates
                captured["resolved"] = json.loads(inputs["campaign-inputs.resolved.json"])

            with mock.patch.object(self.adapter, "publish_campaign", side_effect=publish):
                code, stderr = self._run_main([
                    *common, "--flux-kv-snapshot", str(snapshots["kv"]),
                    "--safety-policy", str(self._policy(root, "darwin-mlx")),
                    "--resume-dir", str(root / "resume-mlx"),
                ])
            self.assertEqual(code, 0, stderr)
            self.assertEqual(captured["resolved"]["backend"], "mlx-metal")
            self.assertEqual(captured["resolved"]["schema"], "sc-20686-resolved-inputs-v4")
            self.assertEqual(len(captured["coordinates"]), 14)
            self.assertEqual(
                {spec.variant for spec in captured["coordinates"]},
                set(self.source_map["lanes"]["mlx-metal"]["variants"]),
            )
            # The resume directory now belongs to the Metal lane; a CUDA policy cannot reuse it.
            code, stderr = self._run_main([
                *common, "--safety-policy", str(self._policy(root, "linux-cuda")),
                "--resume-dir", str(root / "resume-mlx"),
            ])
            self.assertEqual(code, 1)
            self.assertIn("different measurement lane", stderr)

    # --- transcript validation -----------------------------------------------------------------

    def test_every_metal_route_kind_produces_a_valid_row(self):
        for variant in self.source_map["lanes"]["mlx-metal"]["variants"]:
            for cancel in (False, True):
                with self.subTest(variant=variant, cancel=cancel):
                    row = self.make_row(variant, self.mlx_events(variant, cancel), cancel)
                    self.assertEqual(row["backend"], "mlx-metal")
                    self.assertEqual(row["full_generation"], not cancel)
        vace = self.make_row("wan_vace", self.mlx_events("wan_vace"))
        self.assertEqual(vace["current_persistent_bytes"], 0, "MLX VACE recomputes its text K/V")
        wan = self.make_row("wan2_2_t2v_14b", self.mlx_events("wan2_2_t2v_14b"))
        self.assertGreater(wan["current_persistent_bytes"], 0)
        kv = self.make_row("flux2_klein_9b_kv_edit", self.mlx_events("flux2_klein_9b_kv_edit"))
        self.assertGreater(kv["current_persistent_bytes"], 0)
        self.assertFalse(kv["reference_runtime_attribution_available"])

    def assert_refused(self, variant, events, pattern, cancel=False, backend="mlx-metal"):
        with self.assertRaisesRegex(ValueError, pattern):
            self.make_row(variant, events, cancel, backend)

    def test_backend_identity_is_required_and_cannot_cross_lanes(self):
        events = self.mlx_events("wan2_2_t2v_14b")
        del next(e for e in events if e["phase"] == "metadata")["backend"]
        self.assert_refused("wan2_2_t2v_14b", events, "backend differs")
        events = self.mlx_events("wan2_2_t2v_14b")
        del next(e for e in events if e["phase"] == "cross-kv-read")["backend"]
        self.assert_refused("wan2_2_t2v_14b", events, "mlx-metal backend identity")
        # A Metal transcript can never seal a CUDA-lane row, and vice versa for the MLX-only route.
        self.assert_refused(
            "wan2_2_t2v_14b", self.mlx_events("wan2_2_t2v_14b"), "backend differs",
            backend="candle-cuda",
        )
        self.assert_refused(
            "flux2_klein_9b_kv_edit", self.mlx_events("flux2_klein_9b_kv_edit"),
            "not a registered candle-cuda route", backend="candle-cuda",
        )

    def test_phase_windows_are_required_and_bounded(self):
        events = [
            e for e in self.mlx_events("wan2_2_t2v_14b")
            if not (e["phase"] == "phase-window" and e["window"] == "decode")
        ]
        self.assert_refused("wan2_2_t2v_14b", events, "denoise and decode")
        events = [
            e for e in self.mlx_events("wan2_2_t2v_14b")
            if not (e["phase"] == "phase-window" and e["window"] == "denoise-step")
        ]
        self.assert_refused("wan2_2_t2v_14b", events, "denoise and decode")
        events = self.mlx_events("wan2_2_t2v_14b")
        window = next(e for e in events if e["phase"] == "phase-window")
        events.remove(window)
        events.insert(len(events) - 2, window)  # after the terminal event
        self.assert_refused("wan2_2_t2v_14b", events, "after the terminal")
        events = self.mlx_events("wan2_2_t2v_14b")
        next(e for e in events if e["phase"] == "phase-window")["phys_footprint_bytes"] = 0
        self.assert_refused("wan2_2_t2v_14b", events, "phys_footprint")
        events = self.mlx_events("wan2_2_t2v_14b")
        next(e for e in events if e["phase"] == "phase-window")["window"] = "attention"
        self.assert_refused("wan2_2_t2v_14b", events, "missing or unnamed")
        events = self.mlx_events("wan2_2_t2v_14b")
        window = next(e for e in events if e["phase"] == "phase-window")
        window["allocator_high_bytes"] = window["allocator_before_bytes"] - 1
        self.assert_refused("wan2_2_t2v_14b", events, "ordering")

    def test_persistent_bytes_bind_the_exact_live_cfg_batch(self):
        for kv_batch in (3, 1):
            events = self.mlx_events("wan2_2_t2v_14b")
            for event in events:
                if event["phase"] == "cross-kv-created":
                    event["kv_batch"] = kv_batch
            # 3 is not a CFG batch; 1 disagrees with the retained CFG-stacked bytes.
            self.assert_refused("wan2_2_t2v_14b", events, "exact live tensors")

    def test_recomputed_and_persistent_semantics_cannot_be_swapped(self):
        events = self.mlx_events("wan_vace")
        next(e for e in events if e["phase"] == "cross-kv-created")["persistent_bytes"] = 1
        self.assert_refused("wan_vace", events, "non-persistent K/V honestly")
        events = self.mlx_events("flux2_klein_9b_kv_edit")
        next(e for e in events if e["phase"] == "cross-kv-read")["operation"] = (
            "DoubleAttention::attention(joint-context-non-attributable)"
        )
        self.assert_refused("flux2_klein_9b_kv_edit", events, "exact K/V operations")
        events = self.mlx_events("flux2_klein_9b_kv_edit")
        metrics = next(e for e in events if e["phase"] == "metrics")
        metrics["reference_runtime_attribution_available"] = True
        metrics["cache_read_duration_ms"] = 10.0
        metrics["joint_attention_context_duration_ms"] = 0.0
        self.assert_refused("flux2_klein_9b_kv_edit", events, "non-attributable")
        events = self.mlx_events("wan2_2_t2v_14b")
        next(e for e in events if e["phase"] == "metrics")["current_persistent_bytes"] += 1
        self.assert_refused("wan2_2_t2v_14b", events, "exact simultaneous residency")

    # --- reduction and sealed bundle -----------------------------------------------------------

    def sealed_rows(self, variants=None, backend_override=None):
        rows = []
        for family, family_variants in self.mlx_coverage.items():
            for variant, spec in family_variants.items():
                if variants is not None and variant not in variants:
                    continue
                for name in sorted(spec["coordinates"]):
                    for cancel in (False, True):
                        args = self.row_args(variant, cancel, coordinate=name)
                        row = self.adapter.make_row(
                            args, self.config(variant), "b" * 64, 1,
                            self.mlx_events(variant, cancel, coordinate=name),
                        )
                        if backend_override:
                            row["backend"] = backend_override
                        sealed, _raw, _sidecar = self.adapter.seal(row, "row.json")
                        rows.append(sealed)
        return rows

    def test_metal_reduction_decides_each_family_for_the_metal_lane_only(self):
        result = self.reducer.reduce(self.sealed_rows())
        self.assertEqual(result["backend"], "mlx-metal")
        self.assertIn(
            "flux2_klein_9b_kv_edit/edit-512-ref1-cfg1",
            result["decisions"]["flux2-klein"]["coordinates"],
        )
        self.assertIn(result["decisions"]["wan"]["decision"], ("go", "no-go"))
        # Dropping the MLX-only kv route blocks the FLUX family instead of silently narrowing it.
        partial = self.sealed_rows(variants=set(self.mlx_coverage["wan"]) | {"flux2_klein_9b_edit"})
        blocked = self.reducer.reduce(partial)["decisions"]["flux2-klein"]
        self.assertEqual(blocked["decision"], "blocked")
        self.assertTrue(any("flux2_klein_9b_kv_edit" in item for item in map(str, blocked["missing"])))
        # Each row below is individually valid for its own lane; together they are refused.
        mixed = self.sealed_rows(variants={"wan2_2_t2v_14b"})
        mixed[0]["backend"] = "candle-cuda"
        for event in mixed[0]["observer_events"]:
            event.pop("backend", None)
        self.reducer.validate(mixed[0], self.source_map_hash, self.source_map)
        with self.assertRaisesRegex(ValueError, "mix measurement backends"):
            self.reducer.reduce(mixed)

    def test_metal_bundle_round_trip_and_backend_tamper(self):
        coordinates = [
            argparse.Namespace(family=family, variant=variant, name=name)
            for family, variants in self.mlx_coverage.items()
            for variant, spec in variants.items()
            for name in sorted(spec["coordinates"])
        ]
        route_hashes, resolved_coordinates = {}, []
        for coordinate in coordinates:
            document = {
                "route": coordinate.variant, "coordinate": coordinate.name,
                "entrypoint": "product-entrypoint", "entrypoint_sha256": "d" * 64,
                "snapshot": "/snapshot", "model_snapshot_revision": "b" * 40,
                "residency_strategy": self.adapter.PRODUCT_RESIDENCY[coordinate.variant],
                "args": [], "input_files": [],
            }
            route_hash = self.adapter.digest(self.adapter.canonical(document))
            route_hashes[(coordinate.variant, coordinate.name)] = route_hash
            resolved_coordinates.append({
                "family": coordinate.family, "variant": coordinate.variant,
                "name": coordinate.name, "entrypoint": "product-entrypoint",
                "entrypoint_sha256": "d" * 64, "snapshot": "/snapshot",
                "snapshot_sha256": "b" * 64, "snapshot_bytes": 1, "args": [],
                "model_snapshot_revision": "b" * 40,
                "residency_strategy": self.adapter.PRODUCT_RESIDENCY[coordinate.variant],
                "route_manifest_sha256": route_hash, "input_files": [],
            })
        resolved = {
            "schema": "sc-20686-resolved-inputs-v4", "backend": "mlx-metal",
            "inference_revision": "a" * 40, "coordinates": resolved_coordinates,
        }

        with tempfile.TemporaryDirectory() as directory:
            media_root = Path(directory) / "media"

            def runner(coordinate, arm):
                events = self.mlx_events(coordinate.variant, arm == "cancel", coordinate.name)
                observed = [e for e in events if e.get("sample_kind") != "process"]
                process = [e for e in events if e.get("sample_kind") == "process"]
                run_directory = media_root / f"{coordinate.variant}-{coordinate.name}-{arm}" / "sealed-run"
                run_directory.mkdir(parents=True)
                output = run_directory / (
                    "media.png" if coordinate.family == "flux2-klein" else "media"
                )
                if arm == "normal":
                    if coordinate.family == "flux2-klein":
                        output.write_bytes(b"png")
                    else:
                        output.mkdir()
                        (output / "frame_0000.png").write_bytes(b"frame")
                argv = [
                    "product-entrypoint", "--sc20686-campaign", "--sc20686-events",
                    str(run_directory / "events.jsonl"), "--sc20686-source-ref", "a" * 40,
                    "--sc20686-residency", self.adapter.PRODUCT_RESIDENCY[coordinate.variant],
                    "--snapshot", "/snapshot", "--variant", coordinate.variant,
                ]
                if arm == "cancel":
                    argv.append("--sc20686-cancel")
                argv.extend(("--out", str(output)))
                transcript = b"".join(self.adapter.canonical(event) for event in observed)
                return self.adapter.CampaignRun(
                    [*observed, *process], transcript, b"", tuple(argv), tuple(process),
                    transcript, output,
                )

            def row_builder(coordinate, arm, events, evidence_hashes):
                config = dict(self.config(coordinate.variant))
                config["route_manifest_sha256"] = route_hashes[(coordinate.variant, coordinate.name)]
                config["evidence_artifact_sha256"] = evidence_hashes
                args = self.row_args(coordinate.variant, arm == "cancel", coordinate=coordinate.name)
                return self.adapter.make_row(args, config, "b" * 64, 1, events)

            destination = Path(directory) / "campaign"
            self.adapter.publish_campaign(coordinates, runner, row_builder, destination, {
                "campaign-inputs.resolved.json": (
                    json.dumps(resolved, indent=2, sort_keys=True) + "\n"
                ).encode("utf-8"),
                "safety-policy.json": b"{\"schemaVersion\":1}\n",
                "resume-identity.json": b"{\"schema\":\"test\"}\n",
            })
            result = self.reducer.verify_campaign_bundle(destination)
            self.assertEqual(result["backend"], "mlx-metal")
            markdown = (destination / "campaign.md").read_text(encoding="utf-8")
            self.assertIn("Measurement lane: `mlx-metal`", markdown)
            self.assertIn("### `flux2_klein_9b_kv_edit`", markdown)
            # Re-sealing the resolved inputs under the other lane is detected.
            resolved_path = destination / "campaign-inputs.resolved.json"
            forged = json.loads(resolved_path.read_text(encoding="utf-8"))
            forged["backend"] = "candle-cuda"
            raw = (json.dumps(forged, indent=2, sort_keys=True) + "\n").encode("utf-8")
            resolved_path.write_bytes(raw)
            (destination / "campaign-inputs.resolved.json.sha256").write_text(
                f"{self.adapter.digest(raw)}  campaign-inputs.resolved.json\n", encoding="utf-8",
            )
            campaign_path = destination / "campaign.json"
            campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
            campaign["artifact_sha256"]["campaign-inputs.resolved.json"] = self.adapter.digest(raw)
            campaign_raw = (json.dumps(campaign, indent=2, sort_keys=True) + "\n").encode("utf-8")
            campaign_path.write_bytes(campaign_raw)
            (destination / "campaign.json.sha256").write_text(
                f"{self.adapter.digest(campaign_raw)}  campaign.json\n", encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "resolved measurement backend"):
                self.reducer.verify_campaign_bundle(destination)

    # --- source contracts ----------------------------------------------------------------------

    def test_mlx_example_manifest_keeps_the_cuda_coordinates(self):
        scripts = ROOT / "scripts"
        mlx = json.loads(
            (scripts / "sc20686_mlx_wan_campaign_manifest.example.json").read_text(encoding="utf-8")
        )
        cuda = json.loads(
            (scripts / "sc20686_wan_campaign_manifest.example.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(mlx), set(cuda))
        for route, entry in cuda.items():
            self.assertEqual(mlx[route]["coordinates"], entry["coordinates"], route)
            self.assertEqual(mlx[route]["residency_strategy"], entry["residency_strategy"], route)
            self.assertEqual(Path(mlx[route]["entrypoint"]).name, "sc20686_wan", route)

    def test_campaign_doc_describes_the_metal_lane_launch(self):
        document = (ROOT / "docs/architecture/SC_20686_PERSISTENT_KV_CAMPAIGN.md").read_text(
            encoding="utf-8"
        )
        for phrase in (
            "## Metal (MLX) lane", "Mac product-path lane", "--flux-kv-snapshot",
            "sc20686_mlx_wan_campaign_manifest.example.json", "darwin-mlx",
            "recomputed text K/V", "phase-window", "kv_batch",
        ):
            self.assertIn(phrase, document)

    def test_metal_entrypoints_and_observer_carry_the_campaign_contract(self):
        mlx = ROOT / "crates/media/mlx-gen"
        for relative in (
            "mlx-gen-wan/examples/sc20686_wan.rs",
            "mlx-gen-flux2/examples/sc20686_flux2_edit.rs",
        ):
            source = (mlx / relative).read_text(encoding="utf-8")
            for token in (
                '"--sc20686-campaign"', '"--sc20686-events"', '"--sc20686-source-ref"',
                '"--sc20686-residency"', '"--sc20686-cancel"', '.filter(|path| path != "-")',
                "dedicated --sc20686-events <file>", "campaign_cancelled()",
                "with_offload_policy(",
            ):
                self.assertIn(token, source, f"{relative}: {token}")
        wan = (mlx / "mlx-gen-wan/examples/sc20686_wan.rs").read_text(encoding="utf-8")
        for route in self.adapter.WAN_ROUTES:
            self.assertIn(f'"{route}"', wan)
        flux = (mlx / "mlx-gen-flux2/examples/sc20686_flux2_edit.rs").read_text(encoding="utf-8")
        self.assertIn("load_klein_9b_kv_edit", flux)
        self.assertIn("load_klein_9b_edit", flux)
        observer = (mlx / "src/sc20686.rs").read_text(encoding="utf-8")
        self.assertIn('pub const BACKEND: &str = "mlx-metal";', observer)
        self.assertIn('"backend": BACKEND', observer)
        self.assertIn('if PENDING.with(|slot| slot.borrow().is_none()) {\n        return run(on_progress);', observer)


if __name__ == "__main__":
    unittest.main()
