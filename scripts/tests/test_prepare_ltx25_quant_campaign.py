"""Tests for autonomous LTX-2.5 terminal campaign input preparation."""

import importlib.util
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[1] / "release" / "prepare_ltx25_quant_campaign.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_ltx25_quant_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


REVISION = "a" * 40


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def readback(*, private=False, gated=False, revision=REVISION) -> bytes:
    return json.dumps(
        {
            "id": MODULE.PUBLIC_REPOSITORY,
            "sha": revision,
            "private": private,
            "gated": gated,
            "siblings": [{"rfilename": "README.md", "size": 12}],
        }
    ).encode()


def selection(case_ids=None):
    return {
        "policyId": MODULE.POLICY_ID,
        "reviewedBy": "codex-independent-review",
        "selectedCaseIds": case_ids or ["ltx25-int8-convrot-blackwell-v1"],
        "minimumReferencePsnr": 20.0,
        "minimumReferenceSsim": 0.8,
        "maximumTemporalBoundaryDrift": 0.1,
        "minimumReplayPsnr": 100.0,
        "minimumReplaySsim": 0.99,
        "maximumReplayTemporalBoundaryDrift": 0.001,
        "requireReplayOutputHashMatch": True,
    }


class PrepareLtx25QuantCampaignTests(unittest.TestCase):
    def test_public_readback_is_anonymous_public_and_expanded(self):
        calls = []

        def opener(request, timeout):
            calls.append((request, timeout))
            return Response(readback())

        self.assertEqual(MODULE.fetch_public_readback(REVISION, opener=opener), readback())
        self.assertEqual(calls[0][1], 120)
        self.assertNotIn("authorization", dict(calls[0][0].header_items()))
        self.assertTrue(calls[0][0].full_url.endswith(f"/{REVISION}?blobs=true"))

        for private, gated in ((True, False), (False, True)):
            with self.subTest(private=private, gated=gated):
                with self.assertRaisesRegex(ValueError, "private=false, and gated=false"):
                    MODULE.fetch_public_readback(
                        REVISION,
                        opener=lambda *_args, **_kwargs: Response(
                            readback(private=private, gated=gated)
                        ),
                    )

    def test_materialization_is_full_anonymous_and_canonical(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            expected = (
                cache
                / "models--SceneWorks--ltx-2.5-mlx"
                / "snapshots"
                / REVISION
            )
            calls = []

            def download(**kwargs):
                calls.append(kwargs)
                expected.mkdir(parents=True)
                return str(expected)

            self.assertEqual(
                MODULE.materialize_public_snapshot(REVISION, cache, download),
                expected,
            )
            self.assertEqual(
                calls,
                [
                    {
                        "repo_id": MODULE.PUBLIC_REPOSITORY,
                        "revision": REVISION,
                        "cache_dir": str(cache),
                        "token": False,
                    }
                ],
            )
            self.assertNotIn("allow_patterns", calls[0])
            self.assertNotIn("local_dir", calls[0])

    def test_campaign_manifest_contains_exact_nine_public_rows(self):
        snapshot = Path("/cache/models--SceneWorks--ltx-2.5-mlx/snapshots") / REVISION
        document = MODULE.campaign_manifest(snapshot, REVISION)
        self.assertEqual(document["schemaVersion"], MODULE.CAMPAIGN_SCHEMA)
        self.assertEqual(
            MODULE.TERMINAL_CASES,
            (
                ("ltx25-bf16-blackwell-v1", "distilled", "distilled/bf16", None),
                ("ltx25-packed-q4-blackwell-v1", "distilled", "distilled/q4", None),
                ("ltx25-packed-q8-blackwell-v1", "distilled", "distilled/q8", None),
                (
                    "ltx25-int8-convrot-blackwell-v1",
                    "distilled",
                    "bundles/distilled/int8-convrot",
                    "bundles/distilled/int8-convrot/text_encoders/"
                    "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                ),
                (
                    "ltx25-nvfp4-blackwell-v1",
                    "distilled",
                    "bundles/distilled/nvfp4",
                    "bundles/distilled/nvfp4/text_encoders/"
                    "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                ),
                ("ltx25-bf16-blackwell-dev-v1", "dev", "dev/bf16", None),
                ("ltx25-packed-q4-blackwell-dev-v1", "dev", "dev/q4", None),
                ("ltx25-packed-q8-blackwell-dev-v1", "dev", "dev/q8", None),
                (
                    "ltx25-int8-convrot-blackwell-dev-v1",
                    "dev",
                    "bundles/dev/int8-convrot",
                    "bundles/dev/int8-convrot/text_encoders/"
                    "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                ),
            ),
        )
        self.assertEqual(len(document["cases"]), 9)
        advanced = [
            case for case in document["cases"] if "bf16TextEncoderSubpath" in case
        ]
        self.assertEqual(len(advanced), 3)
        self.assertTrue(
            all("text_encoders/gemma4-12b-with-proj" in case["bf16TextEncoderSubpath"] for case in advanced)
        )
        self.assertEqual({case["snapshotRoot"] for case in document["cases"]}, {str(snapshot)})

    def test_snapshot_inventory_matches_sizes_lfs_hashes_and_canonical_blob_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repo = root / "models--SceneWorks--ltx-2.5-mlx"
            blobs = repo / "blobs"
            snapshot = repo / "snapshots" / REVISION
            blobs.mkdir(parents=True)
            snapshot.mkdir(parents=True)
            weights = b"reviewed public weights"
            digest = hashlib.sha256(weights).hexdigest()
            blob = blobs / digest
            blob.write_bytes(weights)
            (snapshot / "weights.safetensors").symlink_to(Path("../../blobs") / digest)
            (snapshot / "README.md").write_bytes(b"public readme")
            raw = json.dumps(
                {
                    "id": MODULE.PUBLIC_REPOSITORY,
                    "sha": REVISION,
                    "private": False,
                    "gated": False,
                    "siblings": [
                        {"rfilename": "README.md", "size": 13},
                        {
                            "rfilename": "weights.safetensors",
                            "size": len(weights),
                            "lfs": {"size": len(weights), "sha256": digest},
                        },
                    ],
                }
            ).encode()
            MODULE.validate_snapshot_against_readback(snapshot, REVISION, raw)

            blob.write_bytes(b"reviewed public weightz")
            with self.assertRaisesRegex(ValueError, "LFS SHA-256 differs"):
                MODULE.validate_snapshot_against_readback(snapshot, REVISION, raw)
            blob.write_bytes(weights)

            extra = snapshot / "extra.json"
            extra.write_text("{}")
            with self.assertRaisesRegex(ValueError, "extra=.*extra.json"):
                MODULE.validate_snapshot_against_readback(snapshot, REVISION, raw)
            extra.unlink()
            (snapshot / "README.md").unlink()
            with self.assertRaisesRegex(ValueError, "missing=.*README.md"):
                MODULE.validate_snapshot_against_readback(snapshot, REVISION, raw)

    def test_snapshot_inventory_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repo = root / "models--SceneWorks--ltx-2.5-mlx"
            (repo / "blobs").mkdir(parents=True)
            snapshot = repo / "snapshots" / REVISION
            snapshot.mkdir(parents=True)
            outside = root / "outside.safetensors"
            outside.write_bytes(b"weights")
            (snapshot / "weights.safetensors").symlink_to(outside)
            raw = json.dumps(
                {
                    "id": MODULE.PUBLIC_REPOSITORY,
                    "sha": REVISION,
                    "private": False,
                    "gated": False,
                    "siblings": [{"rfilename": "weights.safetensors", "size": 7}],
                }
            ).encode()
            with self.assertRaisesRegex(ValueError, "outside canonical blob store"):
                MODULE.validate_snapshot_against_readback(snapshot, REVISION, raw)

    def test_snapshot_inventory_rejects_windows_directory_junction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repo = root / "models--SceneWorks--ltx-2.5-mlx"
            (repo / "blobs").mkdir(parents=True)
            snapshot = repo / "snapshots" / REVISION
            junction = snapshot / "redirected"
            junction.mkdir(parents=True)
            raw = json.dumps(
                {
                    "id": MODULE.PUBLIC_REPOSITORY,
                    "sha": REVISION,
                    "private": False,
                    "gated": False,
                    "siblings": [{"rfilename": "redirected/file.json", "size": 2}],
                }
            ).encode()

            real_isjunction = MODULE.os.path.isjunction

            def isjunction(path):
                return Path(path) == junction or real_isjunction(path)

            with mock.patch.object(MODULE.os.path, "isjunction", side_effect=isjunction):
                with self.assertRaisesRegex(ValueError, "directory symlink, junction"):
                    MODULE.validate_snapshot_against_readback(snapshot, REVISION, raw)

    def test_windows_name_surrogate_reparse_tag_is_an_unsafe_directory_link(self):
        metadata = mock.Mock(
            st_mode=0o040000,
            st_reparse_tag=MODULE.WINDOWS_NAME_SURROGATE_REPARSE_BIT,
        )
        self.assertTrue(
            MODULE._is_unsafe_directory_link(Path("synthetic-directory"), metadata)
        )

    def test_promotion_input_contains_only_explicit_reviewed_winners(self):
        snapshot = Path("/cache/models--SceneWorks--ltx-2.5-mlx/snapshots") / REVISION
        raw = selection(
            [
                "ltx25-nvfp4-blackwell-v1",
                "ltx25-int8-convrot-blackwell-dev-v1",
            ]
        )
        document = MODULE.promotion_input(
            snapshot,
            REVISION,
            Path("/evidence/public-readback.json"),
            raw,
        )
        self.assertEqual(document["selection"], raw)
        self.assertEqual(
            [case["caseId"] for case in document["cases"]],
            [
                "ltx25-nvfp4-blackwell-v1",
                "ltx25-int8-convrot-blackwell-dev-v1",
            ],
        )
        self.assertTrue(all(case["publicModelRevision"] == REVISION for case in document["cases"]))

    def test_selection_rejects_implicit_or_duplicate_variant_winners(self):
        with self.assertRaisesRegex(ValueError, "exactly the required"):
            MODULE.validate_selection({"selectedCaseIds": []})
        with self.assertRaisesRegex(ValueError, "at most one winner"):
            MODULE.validate_selection(
                selection(
                    [
                        "ltx25-int8-convrot-blackwell-v1",
                        "ltx25-nvfp4-blackwell-v1",
                    ]
                )
            )


if __name__ == "__main__":
    unittest.main()
