#!/usr/bin/env python3
"""Regenerate the MOSS-Audio-Tokenizer encode reference-parity fixture (sc-17270).

`crates/audio/candle-audio-moss-tts-realtime/tests/conformance.rs` cross-checks the native
`MossAudioCodec::encode` port against the reference PyTorch `codec.encode` on a byte-identical
clip. Before sc-17270 that arm sat behind two environment variables nothing in the repository set,
so it silently took an `else` branch and the test still reported `1 passed`. The fixture this
script produces is committed, so the arm now runs on every real-weight lane with no operator setup.

Two independent halves, deliberately split so ordinary CI can verify the cheap one:

* ``synth-clip`` writes the reference clip. It is **stdlib only** — a deterministic, synthetic,
  speech-shaped waveform with no third-party audio and therefore no licence to clear. Ordinary CI
  (`scripts/tests/test_moss_audio_codec_reference.py`) regenerates it and compares against the
  committed bytes, so an edited generator or a corrupted fixture fails without weights.
* ``encode`` writes the reference codes CSV by running the **upstream** PyTorch codec. It needs
  ~7.1 GB of real weights plus torch, so it is a dev-box / operator command, never CI.

The reference implementation is *not* vendored. It is fetched at regeneration time from the same
pinned revision `release/real-weight-models.toml` pins for the weights, and each file's SHA-256 is
asserted so a silent upstream edit is caught. Weights are taken only from an explicit environment
variable; this script never resolves repository ids or derives a Hugging Face cache location
(epic 13657).

Regenerating (dev box with the codec snapshot materialized)::

    export MOSS_AUDIO_TOKENIZER_SNAPSHOT=/path/to/MOSS-Audio-Tokenizer/snapshots/<rev>
    python3 scripts/reference/moss_audio_codec_reference.py synth-clip
    python3 scripts/reference/moss_audio_codec_reference.py encode

Both default to the committed fixture paths, so the two commands above rewrite the fixture in
place. Re-run them whenever the pinned codec revision changes — `test_moss_audio_codec_reference`
fails in ordinary CI if the fixture metadata and the manifest pin disagree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

# Run directly (`python3 scripts/reference/moss_audio_codec_reference.py`) and Python puts this
# file's own directory on `sys.path`, not the repository root, so the cross-package import below
# fails. The sibling `if __package__:` idiom in `scripts/release/` only covers same-directory
# imports. Bootstrap the root so both `python3 <path>` and `-m scripts.reference....` work, which
# matters because the unit tests import this module while operators run it as a script.
if not __package__:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.release.verify_model_snapshot import load_model, snapshot_inventory  # noqa: E402

FIXTURE_DIR = (
    REPO_ROOT
    / "crates"
    / "audio"
    / "candle-audio-moss-tts-realtime"
    / "tests"
    / "fixtures"
)
CLIP_PATH = FIXTURE_DIR / "moss_codec_ref_clip.f32"
CODES_PATH = FIXTURE_DIR / "moss_codec_ref_codes.csv"
METADATA_PATH = FIXTURE_DIR / "moss_codec_ref_metadata.json"

MANIFEST_PATH = REPO_ROOT / "release" / "real-weight-models.toml"
MANIFEST_KEY = "moss-audio-tokenizer"

UPSTREAM_REPOSITORY = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
#: The pinned revision. Kept in lockstep with `release/real-weight-models.toml`; the unit test
#: fails if they drift, which forces a fixture regeneration when the pin moves.
UPSTREAM_REVISION = "3cd226ba2947efa357ef453bcad111b6eafba782"
#: Apache-2.0 remote-code modules, fetched at regeneration time and checksum-asserted rather than
#: vendored — the numbers below are this repository's, the upstream source stays upstream.
UPSTREAM_SOURCES = {
    "__init__.py": "ed0b4910a5e53b1dfc3234cfcf96e2c6890d339f9134a3dfed8481d01b3768fd",
    "configuration_moss_audio_tokenizer.py": (
        "349b7ff7e1b3f160f9c80df9a0311672b326b8b73e90459122fb39e6878962bf"
    ),
    "modeling_moss_audio_tokenizer.py": (
        "65cae7744845f1b8ac65957e918cea508efe331a38e87b882b7530b6c8d7caa5"
    ),
}
#: Weight/config files come from the materialized snapshot rather than being re-downloaded, and the
#: list of them is read from the manifest's `expected_files` — never restated here, so it cannot
#: drift from the pin the real-weight lanes enforce.
SNAPSHOT_ENV = "MOSS_AUDIO_TOKENIZER_SNAPSHOT"

# --- Clip geometry -------------------------------------------------------------------------
#: The codec's native rate; `MossAudioCodec::encode` resamples to it, and an equal-rate resample is
#: byte-identical, so a clip authored here reaches the encoder unchanged.
SAMPLE_RATE = 24_000
#: `config.downsample_rate` — waveform samples per RVQ frame (24 kHz / 1920 = 12.5 Hz).
FRAME_SAMPLES = 1_920
#: 100 frames = 8.0 s. An exact multiple of `FRAME_SAMPLES`, so the reference's ceil-padded frame
#: count and its reported valid length (`floor`) coincide and no trailing part-frame is at stake.
#: 8.0 s also keeps the first analysis stage's attention inside its causal context
#: (192000 / 240 = 800 positions <= floor(100 Hz * 10 s) = 1000), so `encode`'s auto path resolves
#: to single-shot — the path this cross-check therefore gates. Chunked-vs-single-shot equivalence
#: is a separate test (`moss_audio_codec_chunked_encode_matches_single_shot`, sc-14181/sc-17263).
CLIP_FRAMES = 100
CLIP_SAMPLES = CLIP_FRAMES * FRAME_SAMPLES
#: `config.rvq` on the MOSS-TTS-Realtime side: the AR brain emits 16 codebooks per frame, so the
#: port loads 16 quantizers and the reference must be asked for the same prefix.
NUM_QUANTIZERS = 16
#: Peak amplitude of the synthesized clip. Below 1.0 so no sample clips when cast to f32.
CLIP_PEAK = 0.9


@dataclass(frozen=True)
class Segment:
    """One synthesized utterance-like segment of the reference clip."""

    start_s: float
    duration_s: float
    #: ``"voiced"`` (harmonic stack under a three-formant envelope) or ``"fricative"``
    #: (high-passed deterministic noise).
    kind: str
    #: Start/end fundamental in Hz, linearly glided. Ignored for fricatives.
    f0_start: float = 0.0
    f0_end: float = 0.0
    gain: float = 1.0


#: A fixed "sentence": alternating voiced syllables with pitch glides and fricative bursts,
#: separated by silences. Broadband and non-stationary, so the analysis stages and every residual
#: quantizer see real structure rather than a single stationary tone.
SEGMENTS: tuple[Segment, ...] = (
    Segment(0.28, 0.52, "voiced", 132.0, 116.0, 1.00),
    Segment(0.88, 0.24, "fricative", gain=0.55),
    Segment(1.20, 0.64, "voiced", 118.0, 154.0, 0.92),
    Segment(1.98, 0.46, "voiced", 149.0, 121.0, 0.80),
    Segment(2.60, 0.30, "fricative", gain=0.48),
    Segment(3.02, 0.72, "voiced", 108.0, 138.0, 1.00),
    Segment(3.90, 0.38, "voiced", 141.0, 132.0, 0.70),
    Segment(4.44, 0.26, "fricative", gain=0.60),
    Segment(4.82, 0.68, "voiced", 126.0, 104.0, 0.88),
    Segment(5.66, 0.50, "voiced", 112.0, 146.0, 0.95),
    Segment(6.30, 0.22, "fricative", gain=0.45),
    Segment(6.66, 0.74, "voiced", 136.0, 110.0, 1.00),
    Segment(7.54, 0.34, "voiced", 120.0, 128.0, 0.75),
)

#: Three Lorentzian resonances (centre Hz, bandwidth Hz, weight) standing in for vowel formants.
FORMANTS: tuple[tuple[float, float, float], ...] = (
    (620.0, 90.0, 1.00),
    (1_180.0, 130.0, 0.62),
    (2_640.0, 220.0, 0.30),
)
#: Harmonics summed per voiced segment; 40 * ~104 Hz still lands under the 12 kHz Nyquist.
NUM_HARMONICS = 40
#: Numerical Recipes LCG — the same integer recurrence the conformance test uses for its
#: self-contained round-trip pattern, so the clip is bit-reproducible without a PRNG library.
LCG_SEED = 17_270
LCG_MULT = 1_664_525
LCG_ADD = 1_013_904_223
LCG_MOD = 1 << 32
#: One-pole high-pass coefficient shaping the fricative noise (a plain `y = a*(y + x - x_prev)`).
FRICATIVE_HP = 0.86


class ReferenceError(RuntimeError):
    """A regeneration precondition failed (bad checksum, missing snapshot, bad geometry)."""


# --- Clip synthesis (stdlib only) ------------------------------------------------------------


def _lcg(seed: int = LCG_SEED):
    """Yield uniform samples in ``[-1, 1)`` from a fixed 32-bit LCG."""
    state = seed
    while True:
        state = (state * LCG_MULT + LCG_ADD) % LCG_MOD
        yield (state / LCG_MOD) * 2.0 - 1.0


def _formant_gain(frequency_hz: float) -> float:
    """Three-resonance spectral envelope evaluated at ``frequency_hz``."""
    gain = 0.0
    for centre, bandwidth, weight in FORMANTS:
        offset = (frequency_hz - centre) / bandwidth
        gain += weight / (1.0 + offset * offset)
    return gain


def _envelope(position: float) -> float:
    """Raised-cosine attack/decay over a segment, ``position`` in ``[0, 1]``."""
    edge = 0.18
    if position < edge:
        return 0.5 - 0.5 * math.cos(math.pi * position / edge)
    if position > 1.0 - edge:
        return 0.5 - 0.5 * math.cos(math.pi * (1.0 - position) / edge)
    return 1.0


def synthesize_clip() -> list[float]:
    """Build the deterministic reference clip as ``CLIP_SAMPLES`` float samples.

    Pure stdlib and pure arithmetic on IEEE-754 doubles, so any machine reproduces it to within
    libm's last-ulp differences on `sin`/`cos` — which is why the unit test compares with a small
    tolerance rather than by hash, and the committed bytes stay the single source of truth for
    what the encoders actually see.
    """
    samples = [0.0] * CLIP_SAMPLES
    noise = _lcg()
    for segment in SEGMENTS:
        start = int(round(segment.start_s * SAMPLE_RATE))
        length = int(round(segment.duration_s * SAMPLE_RATE))
        if start < 0 or start + length > CLIP_SAMPLES:
            raise ReferenceError(
                f"segment at {segment.start_s}s (+{segment.duration_s}s) falls outside the "
                f"{CLIP_SAMPLES}-sample clip"
            )
        if segment.kind == "voiced":
            phase = 0.0
            for i in range(length):
                position = i / length
                f0 = segment.f0_start + (segment.f0_end - segment.f0_start) * position
                phase += 2.0 * math.pi * f0 / SAMPLE_RATE
                value = 0.0
                for harmonic in range(1, NUM_HARMONICS + 1):
                    frequency = f0 * harmonic
                    if frequency >= SAMPLE_RATE / 2.0:
                        break
                    value += _formant_gain(frequency) * math.sin(phase * harmonic) / harmonic
                samples[start + i] += segment.gain * _envelope(position) * value
        elif segment.kind == "fricative":
            previous_input = 0.0
            filtered = 0.0
            for i in range(length):
                raw = next(noise)
                filtered = FRICATIVE_HP * (filtered + raw - previous_input)
                previous_input = raw
                samples[start + i] += segment.gain * _envelope(i / length) * filtered
        else:  # pragma: no cover - guarded by the SEGMENTS literal above
            raise ReferenceError(f"unknown segment kind {segment.kind!r}")

    peak = max(abs(value) for value in samples)
    if peak <= 0.0:
        raise ReferenceError("synthesized clip is silent")
    scale = CLIP_PEAK / peak
    return [value * scale for value in samples]


def clip_to_bytes(samples: Sequence[float]) -> bytes:
    """Pack samples as the raw little-endian f32 mono the fixture and env contract use."""
    return struct.pack(f"<{len(samples)}f", *samples)


def bytes_to_clip(payload: bytes) -> list[float]:
    """Unpack a raw f32-LE mono clip."""
    if len(payload) % 4 != 0:
        raise ReferenceError(f"clip length {len(payload)} is not a whole number of f32 samples")
    return list(struct.unpack(f"<{len(payload) // 4}f", payload))


# --- Codes CSV -------------------------------------------------------------------------------


def codes_to_csv(frames: Sequence[Sequence[int]]) -> str:
    """Render ``frames[T][nq]`` as the one-row-per-frame CSV the Rust test reads."""
    return "".join(",".join(str(code) for code in frame) + "\n" for frame in frames)


def parse_codes_csv(text: str) -> list[list[int]]:
    """Parse the codes CSV back into ``frames[T][nq]``."""
    frames: list[list[int]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            frames.append([int(cell.strip()) for cell in line.split(",")])
        except ValueError as exc:
            raise ReferenceError(f"codes CSV line {line_no} is not all integers: {exc}") from exc
    return frames


def validate_codes(frames: Sequence[Sequence[int]], *, expected_frames: int) -> None:
    """Assert the shape and range invariants the conformance test relies on."""
    if len(frames) != expected_frames:
        raise ReferenceError(
            f"expected {expected_frames} code frames for a {CLIP_SAMPLES}-sample clip, "
            f"got {len(frames)}"
        )
    for index, frame in enumerate(frames):
        if len(frame) != NUM_QUANTIZERS:
            raise ReferenceError(
                f"frame {index} has {len(frame)} codebooks, expected {NUM_QUANTIZERS}"
            )
        for quantizer, code in enumerate(frame):
            if not 0 <= code < 1024:
                raise ReferenceError(
                    f"frame {index} codebook {quantizer} = {code} is outside [0, 1024)"
                )
    # The clip opens and closes on silence, so a handful of repeated frames is expected; a
    # collapsed encoder (or one running on randomly initialized weights) repeats far more than
    # that. The synthesized clip's 13 distinct segments comfortably clear this floor — the
    # committed fixture measures 98/100.
    distinct = len({tuple(frame) for frame in frames})
    if distinct * 2 < len(frames):
        raise ReferenceError(
            f"only {distinct} of {len(frames)} code frames are distinct — the reference encode "
            "looks degenerate (collapsed output, or weights that did not load)"
        )


# --- Manifest pin ----------------------------------------------------------------------------


def manifest_model(key: str = MANIFEST_KEY) -> dict:
    """The pinned manifest entry for ``key``, parsed by the shared release helper.

    Deliberately `scripts.release.verify_model_snapshot.load_model` (real `tomllib`) rather than a
    hand-rolled line scanner: a scanner that tracks `[[models]]` headers reads the wrong revision
    the moment an entry grows a nested table, and this pin is the whole anti-staleness mechanism.
    `tomllib` is already a hard requirement of the same CI job that runs these tests
    (`scripts/check-workspace.py` imports it one step earlier).
    """
    try:
        return load_model(MANIFEST_PATH, key)
    except (OSError, RuntimeError) as exc:
        raise ReferenceError(f"cannot read the pin for {key!r} from {MANIFEST_PATH}: {exc}") from exc


def manifest_revision(key: str = MANIFEST_KEY) -> str:
    """The pinned revision for ``key``."""
    return manifest_model(key)["revision"]


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_metadata(
    *,
    clip_bytes: bytes,
    codes_text: str,
    runtime: dict[str, str],
    inventory: dict,
) -> dict[str, object]:
    """The provenance record committed beside the fixture."""
    return {
        "story": "sc-17270",
        "generator": "scripts/reference/moss_audio_codec_reference.py",
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "revision": UPSTREAM_REVISION,
            "license": "Apache-2.0",
            "sources": dict(sorted(UPSTREAM_SOURCES.items())),
            "vendored": False,
            # The digest of every file in the snapshot that produced these codes, from
            # `verify_model_snapshot.snapshot_inventory`. The revision above is a label; this is
            # the claim about bytes, and it is what makes the codes auditable against the weights
            # rather than merely stamped with a pin someone could have typed.
            "snapshot_inventory_sha256": inventory["inventory_sha256"],
        },
        "clip": {
            "format": "raw f32 little-endian, mono",
            "sample_rate": SAMPLE_RATE,
            "samples": CLIP_SAMPLES,
            "frames": CLIP_FRAMES,
            "peak": CLIP_PEAK,
            "origin": "synthesized by this script's `synth-clip`; no third-party audio",
            "sha256": sha256_hex(clip_bytes),
        },
        "codes": {
            "format": "CSV, one row per frame, frames[T][nq]",
            "num_quantizers": NUM_QUANTIZERS,
            "frames": len(parse_codes_csv(codes_text)),
            "sha256": sha256_hex(codes_text.encode()),
        },
        "runtime": runtime,
    }


# --- Upstream remote code ---------------------------------------------------------------------


def fetch_upstream_sources(destination: Path) -> dict[str, str]:
    """Download the pinned Apache-2.0 remote-code modules into ``destination``.

    Each file's SHA-256 must match :data:`UPSTREAM_SOURCES`; a mismatch means the pin moved (or a
    proxy rewrote the body) and the fixture must not be regenerated from it silently.
    """
    destination.mkdir(parents=True, exist_ok=True)
    fetched: dict[str, str] = {}
    for name, expected in UPSTREAM_SOURCES.items():
        url = (
            f"https://huggingface.co/{UPSTREAM_REPOSITORY}/resolve/"
            f"{UPSTREAM_REVISION}/{name}"
        )
        # A timeout matters: without one a stalled Hub connection hangs the regeneration forever
        # with no output, which reads as "the model is loading" rather than "the network is gone".
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https host
            payload = response.read()
        actual = sha256_hex(payload)
        if actual != expected:
            raise ReferenceError(
                f"{name} at {UPSTREAM_REVISION} hashes {actual}, expected {expected}; the pinned "
                "upstream source changed — review it before regenerating the fixture"
            )
        (destination / name).write_bytes(payload)
        fetched[name] = actual
    return fetched


def stage_model_dir(
    snapshot: Path, remote_code: Path, staged: Path, snapshot_files: Sequence[str]
) -> Path:
    """Link the snapshot weights and the fetched remote code into one loadable directory.

    The verified snapshot is never written to: `ensure_model_snapshot.py` pins its exact file list,
    and dropping `.py` files into it would make that check fail on the next run.

    `transformers` copies trust-remote-code modules into
    `~/.cache/huggingface/modules/transformers_modules/<basename of this directory>`, so the
    basename carries the revision — a generic name would let a stale copy of a previous pin's
    modeling code shadow the freshly fetched one.

    Every link is recreated unconditionally. The work directory persists between runs, and
    `Path.exists()` follows symlinks: a "skip if it exists" staging silently keeps the *previous*
    run's weights when the operator re-points `MOSS_AUDIO_TOKENIZER_SNAPSHOT` at a different
    directory, and raises a confusing `FileExistsError` when the old target has been deleted.
    """
    staged.mkdir(parents=True, exist_ok=True)
    sources = {name: snapshot / name for name in snapshot_files}
    sources.update({name: remote_code / name for name in UPSTREAM_SOURCES})
    for name, source in sources.items():
        if not source.is_file():
            raise ReferenceError(f"staging source is missing: {source}")
        link = staged / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.unlink(missing_ok=True)
        link.symlink_to(source.resolve())
    return staged


def resolve_snapshot() -> Path:
    raw = os.environ.get(SNAPSHOT_ENV, "").strip()
    if not raw:
        raise ReferenceError(
            f"set {SNAPSHOT_ENV} to a materialized MOSS-Audio-Tokenizer snapshot directory"
        )
    snapshot = Path(raw).expanduser()
    if not snapshot.is_dir():
        raise ReferenceError(f"{SNAPSHOT_ENV} is not a directory: {snapshot}")
    return snapshot


def reference_encode(clip: Sequence[float], staged: Path) -> tuple[list[list[int]], dict[str, str]]:
    """Run the upstream PyTorch `codec.encode` and return ``frames[T][nq]`` plus runtime versions.

    Float32 on CPU: the port loads the same weights as f32 (`VarBuilder::from_mmaped_safetensors`
    with `DType::F32`, and the checkpoint is f32 on disk), so this is the like-for-like reference
    rather than a reduced-precision accelerator run.
    """
    import torch  # noqa: PLC0415 - heavy import kept lazy so `synth-clip` needs no torch
    import transformers  # noqa: PLC0415
    from transformers import AutoModel  # noqa: PLC0415

    torch.set_grad_enabled(False)
    load_kwargs = {"trust_remote_code": True, "output_loading_info": True}
    try:
        model, loading_info = AutoModel.from_pretrained(
            staged, dtype=torch.float32, **load_kwargs
        )
    except TypeError:  # transformers < 4.56 spells it `torch_dtype`
        model, loading_info = AutoModel.from_pretrained(
            staged, torch_dtype=torch.float32, **load_kwargs
        )
    # A checkpoint whose keys do not reach the module tree loads *successfully* with those
    # parameters randomly initialized, and would produce plausible-looking but meaningless codes
    # that then get committed as "the reference". Fail instead.
    for label in ("missing_keys", "mismatched_keys"):
        offenders = loading_info.get(label) or []
        if offenders:
            raise ReferenceError(
                f"reference checkpoint has {len(offenders)} {label} "
                f"(e.g. {offenders[:3]}) — refusing to encode against partly random weights"
            )
    model.eval()

    waveform = torch.tensor(list(clip), dtype=torch.float32).view(1, 1, -1)
    output = model.encode(waveform, num_quantizers=NUM_QUANTIZERS, return_dict=True)
    audio_codes = output.audio_codes  # (nq, B, T)
    valid = int(output.audio_codes_lengths[0].item())
    codes = audio_codes[:, 0, :valid].to(torch.int64).tolist()  # (nq, valid)
    frames = [[int(codes[q][t]) for q in range(len(codes))] for t in range(valid)]

    # Recorded, not hard-pinned — a deliberate divergence from `sa3_reference.py`, which fails on a
    # runtime-version mismatch. That script's goldens are float tensors, where a kernel change moves
    # the numbers; these are discrete argmax codes off f32 CPU weights, and the fixture's own
    # correctness is re-checked by the parity assertion itself (a reference that drifted would stop
    # agreeing with the port at 1.000). Hard-pinning torch here would only block regeneration on a
    # newer interpreter without protecting anything the parity gate does not already catch.
    runtime = {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": "cpu",
        "dtype": "float32",
    }
    return frames, runtime


# --- Commands ---------------------------------------------------------------------------------


def command_synth_clip(args: argparse.Namespace) -> int:
    samples = synthesize_clip()
    payload = clip_to_bytes(samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(
        f"wrote {args.output} ({len(payload)} bytes, {CLIP_SAMPLES} samples, "
        f"{CLIP_FRAMES} frames, sha256 {sha256_hex(payload)})"
    )
    return 0


def command_encode(args: argparse.Namespace) -> int:
    model = manifest_model()
    if model["revision"] != UPSTREAM_REVISION:
        raise ReferenceError(
            f"{MANIFEST_PATH.name} pins {MANIFEST_KEY} at {model['revision']} but this script pins "
            f"{UPSTREAM_REVISION}; reconcile them before regenerating"
        )
    snapshot = resolve_snapshot()
    # Bind the fixture to the weights that actually produced it. Comparing two in-repo strings
    # proves nothing about what is on disk: a snapshot materialized at a different revision has the
    # same architecture, so it loads with no missing keys and its codes would be committed stamped
    # with a revision they did not come from. `snapshot_inventory` calls `verify_snapshot` (which
    # enforces the revision marker/basename and `expected_files`) and then digests every file, so
    # the recorded `inventory_sha256` is a claim about bytes.
    try:
        inventory = snapshot_inventory(model, snapshot)
    except RuntimeError as exc:
        raise ReferenceError(f"{SNAPSHOT_ENV} does not match the pinned {MANIFEST_KEY}: {exc}") from exc
    clip_bytes = args.clip.read_bytes()
    clip = bytes_to_clip(clip_bytes)
    if len(clip) != CLIP_SAMPLES:
        raise ReferenceError(
            f"{args.clip} holds {len(clip)} samples, expected {CLIP_SAMPLES}; regenerate it with "
            "`synth-clip` first"
        )

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    remote_code = work / "remote_code"
    fetch_upstream_sources(remote_code)
    staged = stage_model_dir(
        snapshot,
        remote_code,
        work / f"moss_audio_tokenizer_{UPSTREAM_REVISION[:12]}",
        model["expected_files"],
    )

    frames, runtime = reference_encode(clip, staged)
    validate_codes(frames, expected_frames=CLIP_FRAMES)
    codes_text = codes_to_csv(frames)

    args.output_codes.parent.mkdir(parents=True, exist_ok=True)
    args.output_codes.write_text(codes_text, encoding="utf-8")
    metadata = build_metadata(
        clip_bytes=clip_bytes, codes_text=codes_text, runtime=runtime, inventory=inventory
    )
    args.output_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output_codes} ({len(frames)} frames x {NUM_QUANTIZERS} codebooks)")
    print(f"wrote {args.output_metadata}")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    """Re-check the committed fixture without weights: geometry, hashes, and the manifest pin."""
    problems = list(verify_fixture(args.clip, args.codes, args.metadata))
    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)
    if problems:
        return 1
    print("fixture OK")
    return 0


def verify_fixture(clip_path: Path, codes_path: Path, metadata_path: Path) -> Iterable[str]:
    """Yield a message per fixture invariant that does not hold. Empty means healthy."""
    clip_bytes = clip_path.read_bytes()
    codes_text = codes_path.read_text(encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if len(clip_bytes) != CLIP_SAMPLES * 4:
        yield f"clip is {len(clip_bytes)} bytes, expected {CLIP_SAMPLES * 4}"
    if sha256_hex(clip_bytes) != metadata["clip"]["sha256"]:
        yield "clip sha256 does not match the metadata record"
    if sha256_hex(codes_text.encode()) != metadata["codes"]["sha256"]:
        yield "codes sha256 does not match the metadata record"
    if metadata["upstream"]["revision"] != UPSTREAM_REVISION:
        yield (
            f"metadata pins {metadata['upstream']['revision']}, this script pins "
            f"{UPSTREAM_REVISION}"
        )
    if not metadata["upstream"].get("snapshot_inventory_sha256"):
        yield "metadata records no snapshot inventory digest — regenerate with `encode`"
    pin = manifest_revision()
    if pin != UPSTREAM_REVISION:
        yield f"{MANIFEST_PATH.name} pins {pin}, this script pins {UPSTREAM_REVISION}"
    try:
        validate_codes(parse_codes_csv(codes_text), expected_frames=CLIP_FRAMES)
    except ReferenceError as exc:
        yield str(exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    synth = sub.add_parser("synth-clip", help="write the deterministic reference clip (no weights)")
    synth.add_argument("--output", type=Path, default=CLIP_PATH)
    synth.set_defaults(func=command_synth_clip)

    encode = sub.add_parser("encode", help="run the upstream PyTorch codec (needs real weights)")
    encode.add_argument("--clip", type=Path, default=CLIP_PATH)
    encode.add_argument("--output-codes", type=Path, default=CODES_PATH)
    encode.add_argument("--output-metadata", type=Path, default=METADATA_PATH)
    encode.add_argument(
        "--work-dir",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/tmp")) / "moss-audio-codec-reference",
        help="scratch directory for the fetched remote code and the staged model directory",
    )
    encode.set_defaults(func=command_encode)

    verify = sub.add_parser("verify", help="re-check the committed fixture (no weights)")
    verify.add_argument("--clip", type=Path, default=CLIP_PATH)
    verify.add_argument("--codes", type=Path, default=CODES_PATH)
    verify.add_argument("--metadata", type=Path, default=METADATA_PATH)
    verify.set_defaults(func=command_verify)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ReferenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
