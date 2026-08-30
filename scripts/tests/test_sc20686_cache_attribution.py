import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "sc20686_cache_attribution.py"
ADAPTER = Path(__file__).parents[1] / "sc20686_campaign_adapter.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reducer = load(SCRIPT, "sc20686_reducer_contract_test")
        cls.adapter = load(ADAPTER, "sc20686_adapter_contract_test")
        cls.coverage = json.loads(cls.reducer.MANIFEST.read_text(encoding="utf-8"))
        cls.source_map_hash = cls.reducer.sha256(cls.reducer.SOURCE_MAP.read_bytes())

    def row(self, family, variant, coordinate, arm="normal", **overrides):
        expected = self.coverage["families"][family][variant]["coordinates"][coordinate]
        geometry = {
            **expected,
            "layers": 32, "heads": 32, "head_dimension": 128, "sq": 4096,
            "skv": 1024, "dtype": "BF16", "mask": "none", "rope": "native",
        }
        coordinate_id = self.reducer.sha256(
            (json.dumps(geometry, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )[:16]
        exact_candidate = self.reducer.packed_group32_kv_bytes(
            geometry["batch"], geometry["heads"], geometry["skv"],
            geometry["head_dimension"],
        )
        row = {
            "producer": "sc20686-campaign-adapter-v2", "family": family,
            "variant": variant, "coordinate_name": coordinate,
            "coordinate_id": coordinate_id, "arm": arm, "source_ref": "d" * 40,
            "route_manifest_sha256": "e" * 64,
            "source_map_sha256": self.source_map_hash,
            "model_snapshot_sha256": "a" * 64, "model_snapshot_bytes": 1_000_000_000,
            "input_file_sha256": {},
            "evidence_artifact_sha256": {"run.stdout": "b" * 64},
            "geometry": geometry,
            "lifecycle": {"created": 1, "reused": 2, "invalidated": 1, "released": 1},
            "allocator_samples": [{"peak_bytes": 10 * 1024**3}],
            "process_samples": [{"peak_bytes": 10 * 1024**3}],
            "observer_events": [{
                "phase": "cross-kv-created",
                "candidate_persistent_bytes": exact_candidate,
            }],
            "raw_receipt_sha256": "c" * 64,
            "raw_receipt_sidecar_sha256": "f" * 64,
            "real_weights": True, "full_generation": arm == "normal",
            "attention_kind": "cross",
            "current_persistent_bytes": 1, "current_read_transient_bytes": 1,
            "candidate_persistent_bytes": (
                exact_candidate * geometry["layers"] if family == "flux2-klein"
                else exact_candidate
            ),
            "candidate_read_transient_bytes": 1,
            "generation_duration_ms": 1000, "cache_read_duration_ms": 10,
            "reused_requests": 2, "minimum_cache_reads": 2,
        }
        row.update(overrides)
        return row

    def complete_rows(self):
        rows = []
        for family, variants in self.coverage["families"].items():
            for variant, spec in variants.items():
                for coordinate in spec["coordinates"]:
                    rows.extend((
                        self.row(family, variant, coordinate, "normal"),
                        self.row(family, variant, coordinate, "cancel"),
                    ))
        return rows

    def test_cross_coordinate_maxima_cannot_manufacture_a_go(self):
        rows = self.complete_rows()
        wan_normals = [row for row in rows if row["family"] == "wan" and row["arm"] == "normal"]
        large = wan_normals[0]
        large.update({
            "current_persistent_bytes": 700 * 1024**2,
            "candidate_persistent_bytes": 100 * 1024**2,
            "minimum_cache_reads": 1,
        })
        # Another row has reuse >= 2, but its one-byte opportunity cannot donate that fact.
        result = self.reducer.reduce(rows)
        self.assertEqual(result["decisions"]["wan"]["decision"], "no-go")
        # Both frozen coordinates for the exact variant must qualify independently. A single good
        # coordinate never promotes its sibling.
        exact_variant_normals = [
            row for row in wan_normals if row["variant"] == large["variant"]
        ]
        for row in exact_variant_normals:
            row.update({
                "current_persistent_bytes": 700 * 1024**2,
                "candidate_persistent_bytes": 100 * 1024**2,
                "minimum_cache_reads": 2,
                "cache_read_duration_ms": 100,
            })
        result = self.reducer.reduce(rows)
        self.assertEqual(result["decisions"]["wan"]["decision"], "go")
        self.assertEqual(result["decisions"]["wan"]["qualifying_variants"], [large["variant"]])

    def test_allocator_and_process_domains_are_sealed_but_never_summed(self):
        row = self.complete_rows()[0]
        row.update({
            "current_persistent_bytes": 400 * 1024**2,
            "current_read_transient_bytes": 200 * 1024**2,
            "candidate_persistent_bytes": 100 * 1024**2,
            "candidate_read_transient_bytes": 200 * 1024**2,
            "allocator_samples": [{"peak_bytes": 4 * 1024**3}],
            "process_samples": [{"peak_bytes": 12 * 1024**3}],
        })
        decision = self.reducer.coordinate_decision(row)
        self.assertEqual(decision["allocator_peak_bytes"], 4 * 1024**3)
        self.assertEqual(decision["process_rss_peak_bytes"], 12 * 1024**3)
        self.assertNotIn("current_whole_process_bytes", decision)

    def test_family_decisions_are_separate_and_variant_scoped(self):
        rows = self.complete_rows()
        for row in rows:
            if row["family"] == "flux2-klein" and row["arm"] == "normal":
                row.update({
                    "current_read_transient_bytes": 700 * 1024**2,
                    "candidate_read_transient_bytes": 100 * 1024**2,
                    "cache_read_duration_ms": 100,
                })
        result = self.reducer.reduce(rows)
        self.assertEqual(result["decisions"]["flux2-klein"]["decision"], "go")
        self.assertEqual(result["decisions"]["wan"]["decision"], "no-go")

    def test_component_reductions_are_evaluated_independently(self):
        rows = self.complete_rows()
        row = next(item for item in rows if item["family"] == "wan" and item["arm"] == "normal")
        row.update({
            "current_persistent_bytes": 900 * 1024**2,
            "current_read_transient_bytes": 100 * 1024**2,
            "candidate_persistent_bytes": 100 * 1024**2,
            "candidate_read_transient_bytes": 200 * 1024**2,
            "cache_read_duration_ms": 100,
        })
        result = self.reducer.reduce(rows)
        coordinate = f"{row['variant']}/{row['coordinate_name']}"
        decision = result["decisions"]["wan"]["coordinates"][coordinate]
        self.assertTrue(decision["persistent_reduction_qualifies"])
        self.assertFalse(decision["read_transient_reduction_qualifies"])
        self.assertEqual(decision["decision"], "go")

    def test_complete_frozen_coordinates_and_matching_arms_are_required(self):
        rows = self.complete_rows()
        missing = rows[:-1]
        result = self.reducer.reduce(missing)
        self.assertEqual(result["decisions"]["wan"]["decision"], "blocked")
        pair = self.complete_rows()
        cancel = next(row for row in pair if row["arm"] == "cancel")
        cancel["geometry"] = {**cancel["geometry"], "batch": cancel["geometry"]["batch"] + 1}
        expected_candidate = self.reducer.packed_group32_kv_bytes(
            cancel["geometry"]["batch"], cancel["geometry"]["heads"],
            cancel["geometry"]["skv"], cancel["geometry"]["head_dimension"],
        )
        cancel["candidate_persistent_bytes"] = (
            expected_candidate * cancel["geometry"]["layers"]
            if cancel["family"] == "flux2-klein" else expected_candidate
        )
        cancel["observer_events"][0]["candidate_persistent_bytes"] = expected_candidate
        cancel["coordinate_id"] = self.reducer.sha256(
            (json.dumps(cancel["geometry"], sort_keys=True, separators=(",", ":")) + "\n").encode()
        )[:16]
        with self.assertRaisesRegex(ValueError, "geometry differs"):
            self.reducer.reduce(pair)

    def test_checked_in_source_map_is_required_and_closed(self):
        rows = self.complete_rows()
        rows[0]["source_map_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checked-in source map"):
            self.reducer.reduce(rows)
        result = self.reducer.reduce(self.complete_rows())
        self.assertEqual(
            set(result["source_map"]["variants"]),
            {
                variant
                for family in self.coverage["families"].values()
                for variant in family
            },
        )

    def test_source_instrumentation_excludes_wan_self_attention_and_releases_ids(self):
        root = SCRIPT.parents[1]
        transformer = (root / "crates/media/candle-gen/candle-gen-wan/src/transformer.rs").read_text(
            encoding="utf-8"
        )
        observer = (root / "crates/media/candle-gen/candle-gen-wan/src/sc20686_observer.rs").read_text(
            encoding="utf-8"
        )
        self.assertIn("self.prepare_kv_impl(context, false)?", transformer)
        self.assertIn("self.forward_prepared_impl(hidden, &kv, rope, false)", transformer)
        self.assertIn("if !ACTIVE.with(|slot| slot.borrow().is_some())", observer)
        self.assertIn("slot.borrow_mut().remove(&cache_id)", observer)

    def test_flux_source_map_anchors_live_double_attention_kv(self):
        source_map = json.loads(self.reducer.SOURCE_MAP.read_text(encoding="utf-8"))
        entry = source_map["variants"]["flux2_klein_9b_edit"]
        self.assertEqual(
            {anchor["symbol"] for anchor in entry["creation"]},
            {"to_k.forward", "to_v.forward", "add_k.forward", "add_v.forward"},
        )
        self.assertTrue(all(anchor["path"].endswith("flux2/src/transformer.rs") for anchor in entry["creation"]))
        self.assertEqual(entry["reads"][0]["symbol"], "attention")

    def test_standalone_reducer_rejects_unsealed_json_list(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rows.json"
            output = Path(directory) / "out.json"
            source.write_text(json.dumps(self.complete_rows()), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), str(output)],
                check=False, capture_output=True, text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("standalone raw-row reduction is forbidden", completed.stderr)
            self.assertFalse(output.exists())

    def test_exact_raw_row_and_self_sidecar_are_still_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "row.json"
            output = Path(directory) / "out.json"
            row = self.complete_rows()[0]
            _sealed, raw, sidecar = self.adapter.seal(row, raw_path.name)
            raw_path.write_bytes(raw)
            sidecar_path = Path(f"{raw_path}.sha256")
            sidecar_path.write_bytes(sidecar)
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), str(raw_path), str(output), "--sidecar", str(sidecar_path)],
                check=False, capture_output=True, text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("standalone raw-row reduction is forbidden", completed.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
