"""Explicit-proof asset setups download a missing pinned input into the runner cache (SW-7217).

Fake downloaders only: nothing touches the Hub, and no model is loaded.
"""
from __future__ import annotations

import fnmatch
import functools
import io
import json
import os
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.ci import prepare_bespoke_training_proof_assets as bespoke
from scripts.ci import prepare_cohort_training_proof_assets as cohort
from scripts.ci import prepare_krea_proof_assets as krea
from scripts.ci import proof_asset_snapshots as snapshots
from scripts.ci.proof_asset_snapshots import Input

ROOT = Path(__file__).resolve().parents[2]
REVISION = "0123456789abcdef0123456789abcdef01234567"
ROW = f'''schema_version = 1

[[models]]
key = "fixture-proof"
repository = "org/fixture"
revision = "{REVISION}"
requires_auth = true
environment = ["FIXTURE_DIR"]
download_files = ["bf16/*"]
expected_files = ["bf16/config.json", "bf16/model.safetensors"]
'''


class FakeHub:
    """`snapshot_download(cache_dir=...)` that writes the allow-listed files into the hub layout."""

    def __init__(self, files=("bf16/config.json", "bf16/model.safetensors")):
        self.files = files
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        snapshot = (Path(kwargs["cache_dir"]) / ("models--" + kwargs["repo_id"].replace("/", "--"))
                    / "snapshots" / kwargs["revision"])
        for name in self.files:
            if any(fnmatch.fnmatch(name, pattern) for pattern in kwargs.get("allow_patterns", ["*"])):
                (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
                (snapshot / name).write_text("{}", encoding="utf-8")
        return str(snapshot)


class MaterializeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "hub"
        self.manifest = self.root / "models.toml"
        self.manifest.write_text(ROW, encoding="utf-8")
        self.environ = {"HF_HUB_CACHE": str(self.cache), "GITHUB_ACTIONS": "true"}
        self.snapshot = self.cache / "models--org--fixture" / "snapshots" / REVISION

    def materialize(self, download, bindings=None):
        with redirect_stdout(io.StringIO()):
            return snapshots.materialize(bindings or {"FIXTURE_DIR": Input("fixture-proof", "bf16")},
                                         environ=self.environ, download=download, manifest=self.manifest)

    def test_missing_input_downloads_its_allow_listed_slice_into_the_runner_cache(self):
        hub = FakeHub()
        paths = self.materialize(hub, {"FIXTURE_DIR": Input("fixture-proof", "bf16"),
                                       "FIXTURE_ROOT": Input("fixture-proof")})
        self.assertEqual(paths, {"FIXTURE_DIR": self.snapshot / "bf16", "FIXTURE_ROOT": self.snapshot})
        # One fetch per row, through the cache (no local_dir), the slice only, the runner credential.
        self.assertEqual(hub.calls, [{"cache_dir": str(self.cache), "repo_id": "org/fixture",
                                      "revision": REVISION, "token": True, "allow_patterns": ["bf16/*"]}])

    def test_complete_input_is_reused_with_no_transfer(self):
        for name in ("bf16/config.json", "bf16/model.safetensors"):
            (self.snapshot / name).parent.mkdir(parents=True, exist_ok=True)
            (self.snapshot / name).write_text("{}", encoding="utf-8")

        def refuse(**_):
            raise AssertionError("a complete pinned input must not be downloaded")

        self.assertEqual(self.materialize(refuse), {"FIXTURE_DIR": self.snapshot / "bf16"})

    def test_download_short_of_the_expected_files_fails(self):
        with self.assertRaisesRegex(RuntimeError, "bf16/model.safetensors"):
            self.materialize(FakeHub(files=("bf16/config.json",)))

    def test_setup_mains_download_through_the_runner_cache(self):
        """Each setup's entry point routes its bindings through the download path."""
        hub = FakeHub()
        bound = functools.partial(snapshots.materialize, environ=self.environ, download=hub,
                                  manifest=self.manifest)
        export = self.root / "github-env"
        environment = {"GITHUB_ENV": str(export), "MODEL_PROOF_REQUEST": json.dumps(
            {"selection": "candle-gen-x/trainer_e2e::case", "configuration": "fixed_cuda"})}
        with patch.dict(os.environ, environment), redirect_stdout(io.StringIO()):
            with patch.object(cohort, "materialize", bound), patch.dict(
                    cohort.BINDINGS, {"candle-gen-x/trainer_e2e::case": {"FIXTURE_DIR": Input("fixture-proof", "bf16")}},
                    clear=True), patch.object(cohort, "OVERLAYS", {}):
                cohort.main()
            with patch.object(krea, "materialize", bound), patch.dict(
                    krea.KREA_PROOFS, {"krea-training": {"KREA_RAW_DIR": Input("fixture-proof")}}, clear=True), \
                    patch("sys.argv", ["prepare_krea_proof_assets.py", "--setup", "krea-training"]):
                krea.main()
        self.assertEqual(len(hub.calls), 1)  # the cohort run fetched; the Krea run found it complete
        lines = export.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, [f"FIXTURE_DIR={(self.snapshot / 'bf16').as_posix()}", f"KREA_RAW_DIR={self.snapshot}"])

    def test_bespoke_downloads_before_its_metadata_admission(self):
        calls = []

        def fake(bindings):
            calls.append(bindings)
            raise RuntimeError("download refused")

        case = next(iter(bespoke.BESPOKE_TRAINING_PROOFS))
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"backend": "candle-cuda", "selected": 1, "bindings": [case],
                                    "configurations": {case: {"name": "fixed_cuda"}}}), encoding="utf-8")
        with patch.object(bespoke, "materialize", fake), patch("sys.argv", ["x", "--plan", str(plan)]), \
                patch.dict(os.environ, {"GITHUB_RUN_ID": "1", "GITHUB_RUN_ATTEMPT": "1"}):
            with self.assertRaisesRegex(RuntimeError, "download refused"):
                bespoke.main()
        self.assertEqual(calls, [bespoke.BESPOKE_TRAINING_PROOFS[case]])


class KolorsTokenizerOverlayTests(unittest.TestCase):
    def test_missing_derived_tokenizer_is_placed_and_an_existing_one_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kolors, derived = root / "kolors", root / "overlay" / "tokenizer.json"
            kolors.mkdir()
            derived.parent.mkdir()
            derived.write_text('{"derived": true}', encoding="utf-8")
            calls = []

            def fake(bindings):
                calls.append(bindings)
                return {name: derived for name in bindings}

            with patch.object(cohort, "materialize", fake), redirect_stdout(io.StringIO()):
                cohort.overlay_derived_files({"KOLORS_TRAIN_SNAPSHOT": kolors})
                target = kolors / "tokenizer" / "tokenizer.json"
                self.assertEqual(target.read_text(encoding="utf-8"), '{"derived": true}')
                self.assertEqual(calls, [{"KOLORS_TRAIN_SNAPSHOT": Input("kolors-chatglm3-tokenizer-proof",
                                                                         "tokenizer.json")}])
                target.write_text('{"installed": true}', encoding="utf-8")
                cohort.overlay_derived_files({"KOLORS_TRAIN_SNAPSHOT": kolors})
                cohort.overlay_derived_files({"SDXL_SNAPSHOT": kolors})
            self.assertEqual(target.read_text(encoding="utf-8"), '{"installed": true}')
            self.assertEqual(len(calls), 1)


class ManifestCoverageTests(unittest.TestCase):
    def test_every_setup_input_is_an_exact_allow_listed_manifest_row(self):
        models = {row["key"]: row for row in tomllib.loads(
            (ROOT / "release/real-weight-models.toml").read_text(encoding="utf-8"))["models"]}
        tables = {
            "bespoke": bespoke.BESPOKE_TRAINING_PROOFS,
            "cohort": cohort.BINDINGS,
            "cohort-overlay": {name: {name: source} for name, (source, _) in cohort.OVERLAYS.items()},
            "krea": krea.KREA_PROOFS,
        }
        for label, table in tables.items():
            for case, bindings in table.items():
                for name, item in bindings.items():
                    with self.subTest(setup=label, case=case, variable=name):
                        model = models[item.model]
                        self.assertRegex(model["revision"], r"^[0-9a-f]{40}$")
                        self.assertIn(name, model["environment"])
                        # Never a whole-repository fetch: an allow-list that covers every expected file.
                        patterns = model.get("download_files")
                        self.assertTrue(patterns)
                        for expected in model["expected_files"]:
                            self.assertTrue(any(fnmatch.fnmatch(expected, p) for p in patterns), expected)
                        # The bound path is inside the verified slice.
                        self.assertTrue(any(expected == item.path or expected.startswith(item.path + "/")
                                            for expected in model["expected_files"]) if item.path else True)
        self.assertEqual(bespoke.MANIFEST_KEYS | cohort.MANIFEST_KEYS | krea.MANIFEST_KEYS,
                         {item.model for table in tables.values() for bindings in table.values()
                          for item in bindings.values()})


if __name__ == "__main__":
    unittest.main()
