"""Weightless CI gate for the MOSS-Audio-Tokenizer encode reference-parity fixture (sc-17270).

The fixture itself is produced by real weights on a dev box, but everything that can rot without
them is checked here: the clip is regenerated and compared against the committed bytes, the codes
CSV is validated for shape and range, and the three places the codec revision is written down
(the manifest, the generator, the fixture metadata) must agree — so bumping the pin without
regenerating fails in ordinary CI instead of on the next weekly real-weight run.
"""

import json
import struct
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from scripts.reference.moss_audio_codec_reference import (
    CLIP_FRAMES,
    CLIP_PATH,
    CLIP_PEAK,
    CLIP_SAMPLES,
    CODES_PATH,
    FRAME_SAMPLES,
    MANIFEST_PATH,
    METADATA_PATH,
    NUM_QUANTIZERS,
    SAMPLE_RATE,
    UPSTREAM_REPOSITORY,
    UPSTREAM_REVISION,
    UPSTREAM_SOURCES,
    ReferenceError,
    bytes_to_clip,
    codes_to_csv,
    manifest_model,
    manifest_revision,
    parse_codes_csv,
    sha256_hex,
    synthesize_clip,
    validate_codes,
    verify_fixture,
)

#: The test whose reference-parity arm this fixture feeds, and the workflow step that runs it.
CONFORMANCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "audio"
    / "candle-audio-moss-tts-realtime"
    / "tests"
    / "conformance.rs"
)
WORKFLOW_PATH = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "real-weights.yml"
GITATTRIBUTES_PATH = Path(__file__).resolve().parents[2] / ".gitattributes"

#: `math.sin`/`math.cos` are libm calls, so a regenerated clip can differ from the committed one in
#: the last ulp on a different platform. Compare with a tolerance far below anything that could
#: change a code, and let the committed bytes stay the single source of truth for what the two
#: encoders actually saw.
CLIP_TOLERANCE = 1e-6


class MossAudioCodecFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clip_bytes = CLIP_PATH.read_bytes()
        self.codes_text = CODES_PATH.read_text(encoding="utf-8")
        self.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    # --- the fixture is present and well-formed ---------------------------------------------

    def test_clip_has_the_documented_geometry(self) -> None:
        self.assertEqual(len(self.clip_bytes), CLIP_SAMPLES * 4)
        self.assertEqual(CLIP_SAMPLES, CLIP_FRAMES * FRAME_SAMPLES)
        # The clip must be a whole number of frames: a part-padded trailing frame is where the
        # reference's ceil-padded count and its reported valid length diverge.
        self.assertEqual(len(self.clip_bytes) // 4 % FRAME_SAMPLES, 0)
        self.assertEqual(len(self.clip_bytes) // 4 / SAMPLE_RATE, 8.0)
        samples = bytes_to_clip(self.clip_bytes)
        peak = max(abs(value) for value in samples)
        self.assertAlmostEqual(peak, CLIP_PEAK, places=5)
        self.assertLess(peak, 1.0, "a clipped sample would not survive the f32 round-trip")

    def test_codes_have_the_documented_shape_and_range(self) -> None:
        frames = parse_codes_csv(self.codes_text)
        validate_codes(frames, expected_frames=CLIP_FRAMES)
        self.assertEqual(len(frames), CLIP_FRAMES)
        self.assertTrue(all(len(frame) == NUM_QUANTIZERS for frame in frames))

    def test_codes_exercise_every_quantizer(self) -> None:
        """A residual quantizer stuck on one code would gate nothing at that depth."""
        frames = parse_codes_csv(self.codes_text)
        for quantizer in range(NUM_QUANTIZERS):
            distinct = {frame[quantizer] for frame in frames}
            self.assertGreater(
                len(distinct),
                1,
                f"codebook {quantizer} emits a single code across all {CLIP_FRAMES} frames",
            )

    def test_csv_round_trips(self) -> None:
        self.assertEqual(codes_to_csv(parse_codes_csv(self.codes_text)), self.codes_text)

    # --- the fixture matches its own provenance record ---------------------------------------

    def test_metadata_hashes_match_the_committed_files(self) -> None:
        self.assertEqual(self.metadata["clip"]["sha256"], sha256_hex(self.clip_bytes))
        self.assertEqual(self.metadata["codes"]["sha256"], sha256_hex(self.codes_text.encode()))

    def test_metadata_records_the_upstream_provenance(self) -> None:
        upstream = self.metadata["upstream"]
        self.assertEqual(upstream["repository"], UPSTREAM_REPOSITORY)
        self.assertEqual(upstream["revision"], UPSTREAM_REVISION)
        self.assertEqual(upstream["license"], "Apache-2.0")
        self.assertFalse(upstream["vendored"], "the reference sources are fetched, not vendored")
        self.assertEqual(upstream["sources"], dict(sorted(UPSTREAM_SOURCES.items())))

    def test_verify_fixture_reports_no_problems(self) -> None:
        self.assertEqual(list(verify_fixture(CLIP_PATH, CODES_PATH, METADATA_PATH)), [])

    # --- the pin cannot move without a regeneration -------------------------------------------

    def test_manifest_generator_and_metadata_pin_the_same_revision(self) -> None:
        pin = manifest_revision()
        self.assertEqual(
            pin,
            UPSTREAM_REVISION,
            "release/real-weight-models.toml moved the moss-audio-tokenizer pin; regenerate the "
            "fixture with scripts/reference/moss_audio_codec_reference.py",
        )
        self.assertEqual(self.metadata["upstream"]["revision"], pin)

    def test_manifest_revision_rejects_an_unknown_key(self) -> None:
        with self.assertRaises(ReferenceError):
            manifest_revision(key="not-a-model")

    def test_manifest_entry_declares_the_files_the_generator_stages(self) -> None:
        """The generator reads `expected_files` instead of restating them, so they cannot drift."""
        model = manifest_model()
        self.assertEqual(model["repository"], UPSTREAM_REPOSITORY)
        self.assertIn("config.json", model["expected_files"])
        self.assertTrue(
            any(name.endswith(".safetensors") for name in model["expected_files"]),
            model["expected_files"],
        )

    def test_metadata_binds_the_codes_to_the_weights_that_produced_them(self) -> None:
        """A revision string is a label; the inventory digest is a claim about actual bytes."""
        digest = self.metadata["upstream"].get("snapshot_inventory_sha256")
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 64, digest)
        self.assertEqual(digest, digest.lower().strip())

    # --- the clip generator still produces the committed clip -----------------------------------

    def test_regenerating_the_clip_reproduces_the_committed_bytes(self) -> None:
        regenerated = synthesize_clip()
        committed = bytes_to_clip(self.clip_bytes)
        self.assertEqual(len(regenerated), len(committed))
        worst = max(abs(a - b) for a, b in zip(regenerated, committed))
        self.assertLess(
            worst,
            CLIP_TOLERANCE,
            "the clip generator no longer reproduces the committed fixture (worst delta "
            f"{worst:.3e}); regenerate the codes too, they are only valid for this clip",
        )

    def test_clip_is_not_silent_and_not_constant(self) -> None:
        samples = bytes_to_clip(self.clip_bytes)
        self.assertGreater(len(set(samples)), CLIP_SAMPLES // 100)
        energy = sum(value * value for value in samples) / len(samples)
        self.assertGreater(energy, 1e-4, "the clip carries no meaningful energy")

    # --- the validators actually reject bad input ------------------------------------------------

    def test_validate_codes_rejects_a_wrong_frame_count(self) -> None:
        frames = parse_codes_csv(self.codes_text)
        with self.assertRaises(ReferenceError):
            validate_codes(frames[:-1], expected_frames=CLIP_FRAMES)

    def test_validate_codes_rejects_a_wrong_codebook_count(self) -> None:
        frames = parse_codes_csv(self.codes_text)
        frames[0] = frames[0][:-1]
        with self.assertRaises(ReferenceError):
            validate_codes(frames, expected_frames=CLIP_FRAMES)

    def test_validate_codes_rejects_an_out_of_range_code(self) -> None:
        frames = parse_codes_csv(self.codes_text)
        frames[0][0] = 1024
        with self.assertRaises(ReferenceError):
            validate_codes(frames, expected_frames=CLIP_FRAMES)

    def test_validate_codes_rejects_a_collapsed_encode(self) -> None:
        frames = parse_codes_csv(self.codes_text)
        collapsed = [list(frames[0]) for _ in frames]
        with self.assertRaises(ReferenceError):
            validate_codes(collapsed, expected_frames=CLIP_FRAMES)

    def test_verify_fixture_flags_a_tampered_clip(self) -> None:
        with _temporary_copy(CLIP_PATH) as scratch:
            payload = bytearray(CLIP_PATH.read_bytes())
            payload[:4] = struct.pack("<f", 0.123456)
            scratch.write_bytes(bytes(payload))
            problems = list(verify_fixture(scratch, CODES_PATH, METADATA_PATH))
        self.assertTrue(any("clip sha256" in problem for problem in problems), problems)

    def test_verify_fixture_flags_tampered_codes(self) -> None:
        with _temporary_copy(CODES_PATH) as scratch:
            frames = parse_codes_csv(self.codes_text)
            frames[0][0] = (frames[0][0] + 1) % 1024
            scratch.write_text(codes_to_csv(frames), encoding="utf-8")
            problems = list(verify_fixture(CLIP_PATH, scratch, METADATA_PATH))
        self.assertTrue(any("codes sha256" in problem for problem in problems), problems)


@contextmanager
def _temporary_copy(source: Path) -> Iterator[Path]:
    """A scratch path for a tampered copy of ``source``, outside the repository working tree.

    Deliberately not a sibling of the fixture: a crashed test would leave a stray file in
    `crates/`, and a dirty tree is its own failure mode for anything that checks `git status`.
    """
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory) / source.name


class ReferenceArmStaysWiredTests(unittest.TestCase):
    """Guard the consumer, not just the fixture.

    Everything above checks that the committed files are healthy. None of it would notice a future
    edit that reintroduced a skip in `conformance.rs` — ordinary CI would stay green, and the
    weekly lane's `test result: ok. 1 passed` grep cannot see a branch that was not taken. That is
    precisely the failure this story exists to fix, one level up, so it gets its own gate here
    where it costs nothing to run.
    """

    def setUp(self) -> None:
        self.conformance = CONFORMANCE_PATH.read_text(encoding="utf-8")

    def test_conformance_defaults_to_both_committed_fixture_files(self) -> None:
        for name in (CLIP_PATH.name, CODES_PATH.name):
            self.assertIn(
                name,
                self.conformance,
                f"{CONFORMANCE_PATH.name} no longer names {name}; the reference-parity arm has "
                "stopped defaulting to the committed fixture",
            )

    def test_the_reference_arm_has_no_skip_branch(self) -> None:
        self.assertNotIn(
            "SKIPPED",
            self.conformance,
            "a 'SKIPPED' println is back in the MOSS conformance suite — sc-17270 removed the one "
            "that let the reference cross-check no-op while still reporting `1 passed`",
        )

    def test_the_reference_arm_still_asserts_against_the_reference(self) -> None:
        for fragment in ("cb0_rate", "all_rate", "worst_rate"):
            self.assertIn(
                fragment,
                self.conformance,
                f"the reference-parity arm no longer computes {fragment}",
            )

    def test_the_lane_runs_the_test_and_no_longer_calls_it_half_gated(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertIn("run_one moss_audio_codec_encode_roundtrip_and_reference", workflow)
        self.assertNotIn(
            "HALF-GATED",
            workflow,
            "real-weights.yml still carries a HALF-GATED note; if a new one is intentional, it "
            "needs its own story rather than inheriting sc-17270's",
        )

    def test_the_clip_extension_is_declared_binary(self) -> None:
        """`* text=auto eol=lf` would rewrite CR LF pairs inside a clip that opens on audio."""
        attributes = GITATTRIBUTES_PATH.read_text(encoding="utf-8")
        declared = [
            line
            for line in attributes.splitlines()
            if line.strip().startswith("*.f32") and "binary" in line
        ]
        self.assertTrue(declared, "*.f32 is not declared binary in .gitattributes")


if __name__ == "__main__":
    unittest.main()
