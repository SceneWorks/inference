"""Weightless CI gate for the MMAudio torch-parity reference fixtures (sc-17285).

The fixtures are produced by real weights on a dev box, but everything that can rot without them is
checked here: the deterministic inputs are regenerated and compared against the committed bytes,
every fixture's SHA-256 / tensor population / geometry is validated, the vendored reference tree is
re-digested, and the snapshot revisions the fixtures were produced from must still equal the pins in
`release/real-weight-models.toml` — so bumping a pin without regenerating fails in ordinary CI
instead of silently comparing a new model against old numbers on the next weekly real-weight run.

The mutation tests matter as much as the positive ones. A verifier that never fails is exactly the
false green sc-17285 exists to remove, so each check below is also shown to reject a targeted
corruption.
"""

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.reference import mmaudio_reference as ref
from scripts.reference import mmaudio_reference_environment as environment

LOCK_IN = ref.VENDOR / "requirements-oracles.in"
LOCK_TXT = ref.VENDOR / "requirements-oracles.txt"


class MMAudioReferenceFixtureTests(unittest.TestCase):
    def test_committed_fixtures_verify(self) -> None:
        self.assertEqual(ref.verify_fixtures(), [])

    def test_every_gate_has_a_fixture_and_a_schema(self) -> None:
        # The five gates sc-17285 unblocked. A fixture silently dropped from either mapping would
        # take its gate's coverage with it.
        self.assertEqual(set(ref.SCHEMA), set(ref.FIXTURES))
        self.assertEqual(len(ref.FIXTURES), 5)
        for name in ref.FIXTURES:
            self.assertTrue((ref.FIXTURE_DIR / name).is_file(), name)
            self.assertTrue(ref.SCHEMA[name], f"{name} declares no tensors")

    def test_closed_form_latent_matches_the_committed_fixture_bytes(self) -> None:
        """The one constant duplicated in Rust, in Python, and in a committed fixture.

        `conformance_output.rs::fixed_latent` recomputes this latent at real-weight time and asserts
        it equals the fixture's copy; this is the half of that contract ordinary CI can enforce. If
        the generator here is edited, the committed bytes stop matching and this fails without
        weights.
        """
        committed = ref.fixture_tensor_bytes(
            ref.FIXTURE_DIR / "mmaudio_parity_output_16k.safetensors", "latent"
        )
        self.assertEqual(committed, ref.fixed_latent_16k().tobytes())

    def test_deterministic_inputs_are_reproducible_and_non_degenerate(self) -> None:
        clip = ref.clip_frames_u8()
        sync = ref.sync_frames_u8()
        self.assertEqual(clip.shape, (64, 384, 384, 3))
        self.assertEqual(sync.shape, (200, 224, 224, 3))
        # Regenerating must be bit-stable, or the fixtures could never be reproduced.
        self.assertEqual(clip.tobytes(), ref.clip_frames_u8().tobytes())
        # A constant clip would make the CLIP/Synchformer features degenerate and the parity gates
        # correspondingly weak, so pin that the frames actually move and actually differ in space.
        self.assertGreater(int(clip.max()) - int(clip.min()), 200)
        self.assertNotEqual(clip[0].tobytes(), clip[1].tobytes(), "frames must vary over time")
        self.assertNotEqual(sync[0].tobytes(), sync[1].tobytes(), "frames must vary over time")
        latent = ref.fixed_latent_16k()
        self.assertEqual(latent.shape, (1, 20, 48))
        self.assertGreater(float(latent.std()), 0.05, "the latent must not be near-constant")

    def test_metadata_records_a_cpu_float32_run_of_the_pinned_reference(self) -> None:
        metadata = ref.read_metadata()
        self.assertEqual(metadata["device"], "cpu")
        self.assertEqual(metadata["dtype"], "float32")
        self.assertEqual(metadata["upstreamRepository"], ref.UPSTREAM_REPOSITORY)
        self.assertEqual(metadata["upstreamRevision"], ref.UPSTREAM_REVISION)
        self.assertEqual(metadata["snapshotRevisions"], ref.manifest_revisions())
        # The producer environment has to be the PINNED one, not merely recorded: torch changes
        # numerics across releases, so "regenerate it" is only a reproducible instruction if the
        # stack is fixed.
        self.assertEqual(metadata["producerEnvironment"], environment.REFERENCE_PACKAGES)

    def test_producer_pins_and_the_hash_lock_agree(self) -> None:
        """The two halves of the producer-environment contract must not drift apart.

        `REFERENCE_PACKAGES` is what `dump` enforces at run time; `requirements-oracles.in` is what
        an operator installs. If they disagree, a correctly-installed environment fails validation
        (or worse, a wrong one passes).
        """
        direct = {}
        for line in LOCK_IN.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, _, version = line.partition("==")
            self.assertTrue(version, f"unpinned direct requirement: {line}")
            direct[name] = version
        self.assertEqual(direct, environment.REFERENCE_PACKAGES)

        lock = LOCK_TXT.read_text(encoding="utf-8")
        pinned = dict(re.findall(r"(?m)^([A-Za-z0-9._-]+)==([^\s;\\]+)", lock))
        self.assertGreater(len(pinned), len(direct), "the lock must carry the full closure")
        for name, version in environment.REFERENCE_PACKAGES.items():
            # The lock normalizes `open_clip_torch` to `open-clip-torch`, so compare canonically.
            key = name.lower().replace("_", "-")
            self.assertEqual(pinned.get(key), version, f"{name} differs between the pins and lock")
        # Every requirement must be hash-locked, or `--require-hashes` buys nothing.
        self.assertNotIn("--hash=sha512:", lock)
        self.assertGreaterEqual(lock.count("--hash=sha256:"), len(pinned))

    def test_producer_environment_validation_rejects_a_wrong_stack(self) -> None:
        with mock.patch.object(
            environment.importlib.metadata, "version", return_value="0.0.0"
        ):
            with self.assertRaises(ref.ReferenceError):
                environment.validate_reference_environment(ref.ReferenceError)
        with mock.patch.object(
            environment.importlib.metadata,
            "version",
            side_effect=environment.importlib.metadata.PackageNotFoundError,
        ):
            with self.assertRaises(ref.ReferenceError):
                environment.validate_reference_environment(ref.ReferenceError)
        with self.assertRaises(ref.ReferenceError):
            environment.validate_python_version((3, 9, 0), ref.ReferenceError)

    def test_manifest_keys_cover_every_component_the_gates_load(self) -> None:
        # Read through `verify_model_snapshot.load_model`, never a hand-rolled TOML scan: a key that
        # disappears from the manifest must be an error here, not a silently missing revision.
        for key in ref.MANIFEST_KEYS:
            model = ref.load_model(ref.MANIFEST_PATH, key)
            self.assertEqual(model["key"], key)
            self.assertRegex(model["revision"], r"^[0-9a-f]{40}$")


class MMAudioReferenceVerifierDiscriminationTests(unittest.TestCase):
    """Each check must reject a targeted corruption — a verifier that cannot fail proves nothing."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.sandbox = Path(self._tmp.name) / "fixtures"
        shutil.copytree(ref.FIXTURE_DIR, self.sandbox)
        self._original_dir = ref.FIXTURE_DIR
        self._original_metadata = ref.METADATA_PATH
        ref.FIXTURE_DIR = self.sandbox
        ref.METADATA_PATH = self.sandbox / self._original_metadata.name
        # The copy itself must still verify, or every mutation below would "pass" vacuously.
        self.assertEqual(ref.verify_fixtures(), [])

    def tearDown(self) -> None:
        ref.FIXTURE_DIR = self._original_dir
        ref.METADATA_PATH = self._original_metadata
        self._tmp.cleanup()

    def _rewrite_metadata(self, **changes: object) -> None:
        document = json.loads(ref.METADATA_PATH.read_text(encoding="utf-8"))
        document.update(changes)
        ref.METADATA_PATH.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    def test_rejects_a_corrupted_fixture(self) -> None:
        target = ref.FIXTURE_DIR / "mmaudio_parity_output_16k.safetensors"
        payload = bytearray(target.read_bytes())
        payload[-1] ^= 0xFF  # one flipped bit in the last sample
        target.write_bytes(bytes(payload))
        self.assertTrue(any("SHA-256" in error for error in ref.verify_fixtures()))

    def test_rejects_a_missing_fixture(self) -> None:
        (ref.FIXTURE_DIR / "mmaudio_parity_mmdit.safetensors").unlink()
        self.assertTrue(ref.verify_fixtures())

    def test_rejects_a_stale_snapshot_pin(self) -> None:
        revisions = dict(ref.manifest_revisions())
        revisions["mmaudio-small-16k"] = "0" * 40
        self._rewrite_metadata(snapshotRevisions=revisions)
        self.assertTrue(
            any("snapshot revisions" in error for error in ref.verify_fixtures()),
            "a pin bumped without regenerating the fixtures must fail",
        )

    def test_rejects_an_edited_generator(self) -> None:
        digests = dict(ref.deterministic_input_digests())
        digests["fixed_latent_16k"] = "0" * 64
        self._rewrite_metadata(deterministicInputs=digests)
        self.assertTrue(
            any("deterministic inputs" in error for error in ref.verify_fixtures()),
            "an input generator that no longer reproduces the fixtures must fail",
        )

    def test_rejects_an_edited_vendored_reference(self) -> None:
        self._rewrite_metadata(vendorTreeSha256="0" * 64)
        self.assertTrue(
            any("vendored reference tree" in error for error in ref.verify_fixtures()),
            "a local edit to _vendor/mmaudio must invalidate the fixtures",
        )

    def test_rejects_a_changed_generation_configuration(self) -> None:
        for field, value in (
            ("prompt", "something else"),
            ("seed", 43),
            ("cfgStrength", 3.0),
            ("numSteps", 10),
            ("duration", 4.0),
            ("device", "cuda"),
            ("dtype", "bfloat16"),
        ):
            with self.subTest(field=field):
                self._rewrite_metadata(**{field: value})
                self.assertTrue(ref.verify_fixtures(), f"{field} drift must fail")
                # Restore from the pristine committed copy rather than guessing the value back, so
                # each subTest starts from a state that verifies.
                shutil.copy2(self._original_metadata, ref.METADATA_PATH)
                self.assertEqual(ref.verify_fixtures(), [])

    def test_rejects_a_fixture_whose_tensor_population_changed(self) -> None:
        # Swap a fixture's bytes for another fixture's: the SHA check and the schema check are
        # independent, and this must trip on population/geometry as well as on the digest.
        source = self._original_dir / "mmaudio_parity_output_16k.safetensors"
        target = ref.FIXTURE_DIR / "mmaudio_parity_reference_44k_decoder.safetensors"
        shutil.copy2(source, target)
        errors = ref.verify_fixtures()
        self.assertTrue(any("SHA-256" in error for error in errors))
        self.assertTrue(
            any("tensor population differs" in error for error in errors),
            f"expected a population mismatch among {errors}",
        )


if __name__ == "__main__":
    unittest.main()
