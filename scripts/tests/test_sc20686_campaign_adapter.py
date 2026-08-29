import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import importlib.util

ADAPTER = Path(__file__).parents[1] / "sc20686_campaign_adapter.py"
GEOMETRY = {"resolution": "512x512", "reference_count": 1, "frames": 1,
            "prompt": "f" * 64, "guidance": "1.0", "layers": 1, "heads": 2,
            "head_dimension": 64, "sq": 1, "skv": 1024, "dtype": "bf16",
            "mask": "causal", "rope": "native"}


class CampaignAdapterTests(unittest.TestCase):
    def test_wan_manifest_requires_exact_routes_and_real_files(self):
        spec = importlib.util.spec_from_file_location("adapter", ADAPTER)
        adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); binary = root / "wan"; binary.write_text("#!/bin/sh\n", encoding="utf-8"); binary.chmod(0o755)
            entries = {}
            for route in adapter.WAN_ROUTES:
                snapshot = root / route; snapshot.mkdir(); (snapshot / "config.json").write_text("{}", encoding="utf-8")
                entries[route] = {"entrypoint": str(binary), "snapshot": str(snapshot), "args": []}
            manifest = root / "manifest.json"; manifest.write_text(json.dumps(entries), encoding="utf-8")
            self.assertEqual(set(adapter.load_wan_manifest(manifest)), set(adapter.WAN_ROUTES))
            del entries[adapter.WAN_ROUTES[-1]]; manifest.write_text(json.dumps(entries), encoding="utf-8")
            with self.assertRaises(ValueError): adapter.load_wan_manifest(manifest)

    def test_dispatch_campaign_covers_each_route_and_cancel_cleanup(self):
        calls = []
        def runner(variant, arm):
            calls.append((variant, arm))
            return [{"phase": "cross-kv-created"}, {"phase": "cancelled" if arm == "cancel" else "generation-end"}, {"phase": "released"}]
        spec = importlib.util.spec_from_file_location("adapter", ADAPTER)
        adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
        runs = adapter.dispatch_campaign(runner)
        self.assertEqual(len(runs), 10)
        self.assertEqual(calls, [(route, arm) for route in adapter.WAN_ROUTES for arm in ("normal", "cancel")])
        self.assertEqual(sum(any(event["phase"] == "cancelled" for event in events) for _, arm, events in runs if arm == "cancel"), 5)

    def test_caller_authored_jsonl_is_rejected(self):
        spec = importlib.util.spec_from_file_location("adapter", ADAPTER)
        adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "snapshot"; root.mkdir()
            (root / "config.json").write_text("{}", encoding="utf-8")
            snapshot_hash, snapshot_bytes = adapter.snapshot_identity(root)
            geometry = {**GEOMETRY, "resolution": "64x64", "frames": 5}
            events = [{"phase": "metadata", "source_ref": "a" * 40,
                       "snapshot_sha256": snapshot_hash, "snapshot_bytes": snapshot_bytes,
                       "variant": "wan2_2_t2v_14b", "geometry": geometry,
                       "real_weights": True, "full_generation": True, "attention_kind": "cross"},
                      {"phase": "generation-start", "sample_kind": "allocator", "peak_bytes": 1024, "at_ns": 1},
                      {"phase": "cross-kv-created", "persistent_bytes": 700, "transient_bytes": 0, "sample_kind": "allocator", "peak_bytes": 1024, "at_ns": 2},
                      {"phase": "cross-kv-read", "persistent_bytes": 700, "transient_bytes": 10, "sample_kind": "allocator", "peak_bytes": 1024, "at_ns": 3},
                      {"phase": "generation-end", "sample_kind": "allocator", "peak_bytes": 1024, "at_ns": 4},
                      {"phase": "invalidated", "sample_kind": "allocator", "peak_bytes": 1024, "at_ns": 5},
                      {"phase": "released", "sample_kind": "allocator", "peak_bytes": 1, "at_ns": 6},
                      {"phase": "process-sample", "sample_kind": "process", "peak_bytes": 2048, "at_ns": 7},
                      {"phase": "metrics", "current_persistent_bytes": 1024**3, "current_read_transient_bytes": 10,
                       "candidate_persistent_bytes": 100, "candidate_read_transient_bytes": 10,
                       "generation_duration_ms": 100, "cache_read_duration_ms": 5, "reused_requests": 2}]
            events_path = root.parent / "events.json"
            events_path.write_text(json.dumps(events), encoding="utf-8")
            output = root.parent / "decision.json"
            completed = subprocess.run([sys.executable, str(ADAPTER), "--campaign", "--family", "wan",
                                        "--variant", "wan2_2_t2v_14b", "--snapshot", str(root),
                                        "--events", str(events_path), "--output", str(output)],
                                       check=False, text=True, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output.exists())

    def test_flux_nonpersistent_route_requires_dense_read_evidence(self):
        spec = importlib.util.spec_from_file_location("adapter", ADAPTER)
        adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
        geometry = {**GEOMETRY, "sq": 64, "skv": 16}
        events = [
            {"phase": "metadata", "source_ref": "a" * 40, "snapshot_sha256": "b" * 64,
             "snapshot_bytes": 1, "variant": "flux2_klein_9b_edit", "geometry": geometry,
             "real_weights": True, "full_generation": True, "attention_kind": "cross"},
            {"phase": "generation-start", "sample_kind": "allocator", "peak_bytes": 100},
            {"phase": "cross-kv-created", "persistent_bytes": 0, "sample_kind": "allocator", "peak_bytes": 100},
            {"phase": "cross-kv-read", "persistent_bytes": 0, "transient_bytes": 64, "sample_kind": "allocator", "peak_bytes": 100},
            {"phase": "generation-end", "sample_kind": "allocator", "peak_bytes": 100},
            {"phase": "invalidated", "sample_kind": "allocator", "peak_bytes": 100},
            {"phase": "released", "sample_kind": "allocator", "peak_bytes": 100},
            {"phase": "process-sample", "sample_kind": "process", "peak_bytes": 100},
            {"phase": "metrics", "current_persistent_bytes": 0, "current_read_transient_bytes": 64,
             "candidate_persistent_bytes": 32, "candidate_read_transient_bytes": 0,
             "generation_duration_ms": 10, "cache_read_duration_ms": 1, "reused_requests": 0},
        ]
        args = type("Args", (), {"fake": False, "family": "flux2-klein",
                                   "variant": "flux2_klein_9b_edit", "cancel_campaign": False})()
        row = adapter.make_row(args, {}, "b" * 64, 1, events)
        self.assertEqual(row["current_persistent_bytes"], 0)
        events[2]["persistent_bytes"] = 1
        with self.assertRaisesRegex(ValueError, "non-persistent"):
            adapter.make_row(args, {}, "b" * 64, 1, events)

    def test_cancel_arm_requires_product_metrics_and_cancel_identity(self):
        spec = importlib.util.spec_from_file_location("adapter", ADAPTER)
        adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
        events = [
            {"phase": "metadata", "source_ref": "a" * 40, "snapshot_sha256": "b" * 64,
             "snapshot_bytes": 1, "variant": "flux2_klein_9b_edit", "geometry": GEOMETRY,
             "real_weights": True, "full_generation": False, "attention_kind": "cross",
             "cancellation_armed": True, "cancellation_arm_id": "a:b"},
            {"phase": "generation-start", "sample_kind": "allocator", "peak_bytes": 100},
            {"phase": "cross-kv-created", "persistent_bytes": 0, "sample_kind": "allocator", "peak_bytes": 100},
            {"phase": "cross-kv-read", "persistent_bytes": 0, "transient_bytes": 64, "sample_kind": "allocator", "peak_bytes": 100},
            {"phase": "cancelled", "sample_kind": "allocator", "peak_bytes": 100},
            {"phase": "invalidated", "sample_kind": "allocator", "peak_bytes": 100},
            {"phase": "released", "sample_kind": "allocator", "peak_bytes": 100},
            {"phase": "process-sample", "sample_kind": "process", "peak_bytes": 100},
            {"phase": "metrics", "current_persistent_bytes": 0, "current_read_transient_bytes": 64,
             "candidate_persistent_bytes": 32, "candidate_read_transient_bytes": 0,
             "generation_duration_ms": 2, "cache_read_duration_ms": 1, "reused_requests": 0},
        ]
        args = type("Args", (), {"fake": False, "family": "flux2-klein",
                                   "variant": "flux2_klein_9b_edit", "cancel_campaign": True})()
        self.assertEqual(adapter.make_row(args, {}, "b" * 64, 1, events)["arm"], "cancel")
        events[-1].pop("generation_duration_ms")
        with self.assertRaisesRegex(ValueError, "metrics"):
            adapter.make_row(args, {}, "b" * 64, 1, events)

    def test_fake_both_families_emit_sealed_rows(self):
        for family in ("flux2-klein", "wan"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "snapshot"
                root.mkdir(); (root / "config.json").write_text(
                    json.dumps({"source_ref": "frozen-test", "sc20686_geometry": GEOMETRY}),
                    encoding="utf-8")
                output = Path(directory) / "decision.json"
                command = [sys.executable, str(ADAPTER), "--campaign", "--fake",
                           "--family", family, "--variant", "test", "--snapshot", str(root),
                           "--output", str(output)]
                completed = subprocess.run(command, check=False, text=True,
                                           encoding="utf-8", capture_output=True)
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse(output.exists())

    def test_normal_invocation_and_missing_hook_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "config.json").write_text("{}", encoding="utf-8")
            base = [sys.executable, str(ADAPTER), "--family", "wan", "--variant", "x",
                    "--snapshot", str(root), "--output", str(root / "o.json")]
            self.assertNotEqual(subprocess.run(base, check=False).returncode, 0)
            command = base + ["--campaign"]
            self.assertNotEqual(subprocess.run(command, check=False).returncode, 0)


if __name__ == "__main__":
    unittest.main()
