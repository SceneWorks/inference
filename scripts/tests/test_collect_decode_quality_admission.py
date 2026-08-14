import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "ci" / "collect_decode_quality_admission.py"
SPEC = importlib.util.spec_from_file_location("decode_quality_collector", SCRIPT)
assert SPEC and SPEC.loader
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def receipt(seed: int, error: int) -> dict:
    token = f"{seed:064x}"
    return {
        "family": "sdxl",
        "resolvedRoute": "realvisxl",
        "backend": "mlx",
        "tier": "q4",
        "loadShape": "deferred_materialization",
        "artifact": {
            "repository": "SceneWorks/realvisxl-mlx",
            "revision": "a" * 40,
            "variant": "q4",
            "fingerprint": f"SceneWorks/realvisxl-mlx@{'a' * 40}:q4",
        },
        "implementationFingerprint": collector.IMPLEMENTATION_FINGERPRINT,
        "mode": "text_to_image",
        "overlay": None,
        "geometry": {
            "width": 1024,
            "height": 1024,
            "batch": 1,
            "frames": 1,
            "referenceCount": 0,
        },
        "usePid": False,
        "tileEdge": 896,
        "overlap": 192,
        "metric": "max_abs_rgb_u8",
        "maximumError": 48,
        "seed": seed,
        "productionLatentProvenance": "realvisxl@abc q4 production prompt=p9-v1",
        "productionLatentSha256": token,
        "denseOutputSha256": token,
        "tiledOutputSha256": token,
        "observedError": error,
    }


class DecodeQualityAdmissionCollectorTests(unittest.TestCase):
    def write_log(self, row: dict) -> Path:
        temporary = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        with temporary:
            temporary.write(f"DECODE_QUALITY_V2 {json.dumps(row)}\n")
        return Path(temporary.name)

    def test_seals_a_multi_seed_coordinate_and_retains_failure_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "quality.log"
            log.write_text(
                "noise\n"
                + "\n".join(
                    f"test output DECODE_QUALITY_V2 {json.dumps(row)}"
                    for row in (receipt(7, 47), receipt(99, 52))
                )
                + "\n",
                encoding="utf-8",
            )

            policies = collector.seal(collector.read_receipts([log]))

        self.assertEqual(len(policies), 1)
        policy = policies[0]
        self.assertEqual([fixture["seed"] for fixture in policy["fixtures"]], [7, 99])
        self.assertEqual(
            policy["disposition"],
            {
                "kind": "refused",
                "reason": "max_abs_rgb_u8 exceeded 48: seed 99=52",
            },
        )
        self.assertEqual(len(policy["productionEvidenceSha256"]), 64)
        self.assertEqual(
            policy["productionEvidenceSha256"],
            collector.seal([receipt(99, 52), receipt(7, 47)])[0][
                "productionEvidenceSha256"
            ],
        )

    def test_rejects_nonsemantic_fields_instead_of_absorbing_measurements(self) -> None:
        row = receipt(7, 47)
        row["elapsedMs"] = 12

        with self.assertRaisesRegex(ValueError, "semantic allowlist"):
            collector.read_receipts([self.write_log(row)])

    def test_input_root_deterministically_merges_isolated_geometry_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "z-route" / "decode-quality-v2-sdxl-realvisxl-1024x1024.log"
            second = root / "a-route" / "decode-quality-v2-sdxl-realvisxl-768x768.log"
            first.parent.mkdir()
            second.parent.mkdir()
            for path, width in ((first, 1024), (second, 768)):
                rows = [receipt(7, 1), receipt(99, 2)]
                for row in rows:
                    row["geometry"]["width"] = width
                    row["geometry"]["height"] = width
                path.write_text(
                    "".join(f"DECODE_QUALITY_V2 {json.dumps(row)}\n" for row in rows),
                    encoding="utf-8",
                )
            (root / "unrelated.log").write_text("ignored\n", encoding="utf-8")

            paths = collector.receipt_log_paths([], root)
            policies = collector.seal(collector.read_receipts(paths))

        self.assertEqual(paths, [second, first])
        self.assertEqual(
            [policy["geometry"]["width"] for policy in policies],
            [768, 1024],
        )

    def test_input_root_rejects_duplicate_explicit_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "decode-quality-v2-sdxl-realvisxl-1024x1024.log"
            log.write_text("unused\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate decode-quality"):
                collector.receipt_log_paths([log], root)

    def test_isolated_cell_shape_requires_one_row_with_all_five_seeds(self) -> None:
        policies = collector.seal([receipt(seed, 1) for seed in (1234, 7, 99, 20260805, 424242)])
        collector.require_policy_shape(policies, 1, 5)
        with self.assertRaisesRegex(ValueError, "expected 4 fixtures"):
            collector.require_policy_shape(policies, 1, 4)
        with self.assertRaisesRegex(ValueError, "expected 2 policy"):
            collector.require_policy_shape(policies, 2, 5)

    def test_rejects_malformed_semantic_coordinates(self) -> None:
        cases = [
            ("usePid", 0, "usePid must be a boolean"),
            ("overlay", "bad=value", "overlay must be"),
            ("tileEdge", 128, "tileEdge must exceed"),
        ]
        for field, value, message in cases:
            with self.subTest(field=field):
                row = receipt(7, 47)
                if field == "tileEdge":
                    row["overlap"] = 128
                row[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    collector.read_receipts([self.write_log(row)])

    def test_load_shape_artifact_implementation_and_tile_pair_are_exact_coordinates(self) -> None:
        rows = [receipt(7, 1), receipt(99, 2)]
        variants: list[dict] = []
        for mutate in (
            lambda row: row.update(loadShape="eager_materialization"),
            lambda row: row["artifact"].update(variant="q8"),
            lambda row: row.update(tileEdge=768, overlap=128),
        ):
            pair = [receipt(7, 1), receipt(99, 2)]
            for row in pair:
                mutate(row)
            variants.extend(pair)

        policies = collector.seal([*rows, *variants])
        self.assertEqual(len(policies), 4)
        coordinates = {
            (
                policy["loadShape"],
                policy["artifact"]["variant"],
                policy["implementationFingerprint"],
                policy["tileEdge"],
                policy["overlap"],
            )
            for policy in policies
        }
        self.assertEqual(len(coordinates), 4)

    def test_rejects_malformed_load_and_source_identity(self) -> None:
        cases = [
            (lambda row: row.update(loadShape="deferred"), "unsupported loadShape"),
            (
                lambda row: row["artifact"].update(revision="A" * 40),
                "artifact.revision",
            ),
            (
                lambda row: row["artifact"].update(extra="ambient"),
                "exact ABI-2 identity axes",
            ),
            (
                lambda row: row.update(implementationFingerprint="g" * 64),
                "implementationFingerprint",
            ),
            (
                lambda row: row.update(implementationFingerprint="f" * 64),
                "running inference source closure",
            ),
        ]
        for mutate, message in cases:
            with self.subTest(message=message):
                row = receipt(7, 1)
                mutate(row)
                with self.assertRaisesRegex(ValueError, message):
                    collector.read_receipts([self.write_log(row)])


if __name__ == "__main__":
    unittest.main()
