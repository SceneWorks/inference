#!/usr/bin/env python3
"""Regenerate the MMAudio torch-parity reference fixtures (sc-17285).

`crates/audio/candle-audio-mmaudio/tests/` carries five numerical parity gates against the
PyTorch MMAudio reference. Until sc-17285 every one of them was gated on an environment variable
pointing at a safetensors dump produced by a `scratchpad/*.py` script git had never tracked, so
none of them could run: three test binaries were unreachable and two more early-`return`ed to a
passing result when their variable was unset. This script is the producer those five gates were
always missing, and the fixtures it writes are **committed**, so the gates now run with no
operator setup at all — there is no environment variable left to forget.

Five fixtures, one per gate:

=======================================  ======================================================
fixture (``.safetensors``)               gate
=======================================  ======================================================
``mmaudio_parity_reference``             ``parity_reference.rs`` — assembled 16 kHz pipeline
``mmaudio_parity_reference_44k``         ``parity_reference_44k.rs`` — assembled 44.1 kHz pipeline
``mmaudio_parity_reference_44k_decoder`` ``parity_reference_44k_decoder.rs`` — v1-44 + BigVGAN v2
``mmaudio_parity_output_16k``            ``conformance_output.rs`` — v1-16 + 16 kHz BigVGAN
``mmaudio_parity_mmdit``                 ``mmdit_conformance.rs`` — MM-DiT and its Euler sampler
=======================================  ======================================================

The reference implementation **is** vendored, at
``crates/audio/candle-audio-mmaudio/_vendor/mmaudio`` — see ``_vendor/VENDORED.md`` for the
provenance and why the licence permits it (MIT, into this Apache-2.0 repository, the same call
epic 14034 made for Mage-Flow). Weights come only from explicit snapshot environment variables;
this script never resolves a repository id or derives a Hugging Face cache location (epic 13657).

Two halves, deliberately split so ordinary CI can verify the cheap one:

* the **deterministic inputs** — the synthetic video, the prompts, the seeded prior and the
  closed-form 16 kHz latent — are stdlib/numpy only and carry no licence. Ordinary CI
  (`scripts/tests/test_mmaudio_reference.py`) regenerates them and compares against the digests
  recorded in the fixture metadata, so an edited generator fails without weights.
* the **reference outputs** need ~8 GB of real weights plus torch, so `dump` is a dev-box /
  operator command, never CI.

Regenerating (dev box with the three snapshots materialized)::

    python3.12 -m venv /tmp/mmaudio-reference
    /tmp/mmaudio-reference/bin/python -m pip install --require-hashes \
      -r crates/audio/candle-audio-mmaudio/_vendor/requirements-oracles.txt

    export MMAUDIO_MMAUDIO_SNAPSHOT=/path/to/models--hkchengrex--MMAudio/snapshots/<rev>
    export MMAUDIO_CLIP_SNAPSHOT=/path/to/models--apple--DFN5B-CLIP-ViT-H-14-384/snapshots/<rev>
    export MMAUDIO_BIGVGAN_V2_SNAPSHOT=/path/to/models--nvidia--bigvgan_v2_.../snapshots/<rev>
    /tmp/mmaudio-reference/bin/python scripts/reference/mmaudio_reference.py dump

The producer stack is **pinned**, not merely reported: `mmaudio_reference_environment.py` restates
the lock's direct pins and `dump` refuses to run in an environment that does not match, because
these fixtures record numbers and torch moves them between releases.

`dump` rewrites the committed fixtures in place and takes roughly half an hour on CPU, most of it
in the 44.1 kHz sampler. Re-run it whenever a pinned snapshot revision changes — the unit test
fails in ordinary CI when the fixture metadata and `release/real-weight-models.toml` disagree,
so a pin bump cannot silently leave stale numbers behind.

Everything runs in float32 on the CPU. The reference's own default is bfloat16 on CUDA; f32/CPU is
chosen so the fixture records the reference's *mathematics* rather than one accelerator's rounding,
which is what the candle side is being held to.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Run directly (`python3 scripts/reference/mmaudio_reference.py`) and Python puts this file's own
# directory on `sys.path`, not the repository root, so the cross-package import below fails.
# Bootstrap the root so both `python3 <path>` and `-m scripts.reference....` work, which matters
# because the unit tests import this module while operators run it as a script.
if not __package__:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reference.mmaudio_reference_environment import (  # noqa: E402
    REFERENCE_PACKAGES,
    validate_reference_environment,
)
from scripts.release.verify_model_snapshot import load_model  # noqa: E402

CRATE = REPO_ROOT / "crates" / "audio" / "candle-audio-mmaudio"
VENDOR = CRATE / "_vendor"
FIXTURE_DIR = CRATE / "tests" / "fixtures"
METADATA_PATH = FIXTURE_DIR / "mmaudio_parity_metadata.json"
MANIFEST_PATH = REPO_ROOT / "release" / "real-weight-models.toml"

#: Upstream provenance of the vendored reference. `_vendor/VENDORED.md` carries the per-file
#: digests; this is the pin the tree is verified against.
UPSTREAM_REPOSITORY = "hkchengrex/MMAudio"
UPSTREAM_REVISION = "974010a026c731054592d8f777218bd9d85a6c24"

#: Snapshot environment variables, identical to the ones `tests/common/mod.rs` already requires so
#: an operator regenerating fixtures needs no new setup. `MMAUDIO_BIGVGAN_V2_SNAPSHOT` names the
#: *nvidia* 44.1 kHz repo, NOT hkchengrex/MMAudio's 16 kHz `best_netG.pt` (sc-17266).
MMAUDIO_SNAPSHOT_ENV = "MMAUDIO_MMAUDIO_SNAPSHOT"
CLIP_SNAPSHOT_ENV = "MMAUDIO_CLIP_SNAPSHOT"
BIGVGAN_V2_SNAPSHOT_ENV = "MMAUDIO_BIGVGAN_V2_SNAPSHOT"

#: manifest key -> the snapshot environment variable that supplies it. Every revision written into
#: the fixture metadata is read back out of the manifest through `verify_model_snapshot.load_model`,
#: never restated here, so a pin bump cannot drift from the fixture.
MANIFEST_KEYS = {
    "mmaudio-small-16k": MMAUDIO_SNAPSHOT_ENV,
    "mmaudio-large-44k-v2": MMAUDIO_SNAPSHOT_ENV,
    "mmaudio-vae-16k": MMAUDIO_SNAPSHOT_ENV,
    "mmaudio-vae-44k": MMAUDIO_SNAPSHOT_ENV,
    "mmaudio-bigvgan-16k": MMAUDIO_SNAPSHOT_ENV,
    "synchformer": MMAUDIO_SNAPSHOT_ENV,
    "dfn5b-clip-vit-h14-384": CLIP_SNAPSHOT_ENV,
    "nvidia-bigvgan-v2-44khz-128band-512x": BIGVGAN_V2_SNAPSHOT_ENV,
}

#: In-repo relative checkpoint paths, mirroring the `pub const *_WEIGHTS_PATH` the candle crate
#: joins onto the same snapshot dirs (`src/mmdit.rs`, `src/output.rs`, `src/model.rs`, `src/clip.rs`).
DIT_16K = "weights/mmaudio_small_16k.pth"
DIT_44K = "weights/mmaudio_large_44k_v2.pth"
VAE_16K = "ext_weights/v1-16.pth"
VAE_44K = "ext_weights/v1-44.pth"
BIGVGAN_16K = "ext_weights/best_netG.pt"
SYNCHFORMER = "ext_weights/synchformer_state_dict.pth"
CLIP_WEIGHTS = "open_clip_pytorch_model.bin"

#: The generation the fixtures record. These are the reference's own documented defaults
#: (`demo.py`: 8 s, CFG 4.5, 25 Euler steps, seed 42), so the fixture pins the shipping
#: configuration rather than a corner of it.
PROMPT = "a wooden door creaks open, then footsteps on gravel"
NEGATIVE_PROMPT = "music"
SEED = 42
CFG_STRENGTH = 4.5
NUM_STEPS = 25
DURATION = 8.0
CLIP_FPS = 8
SYNC_FPS = 25
CLIP_SIZE = 384
SYNC_SIZE = 224

#: The closed-form 16 kHz latent, duplicated bit-for-bit in `conformance_output.rs::fixed_latent`.
#: The fixture carries it so the Rust side can assert its own copy still agrees — a cross-language
#: constant that drifts silently is exactly how a parity gate goes quietly vacuous.
OUTPUT_16K_CHANNELS = 20
OUTPUT_16K_LENGTH = 48

#: The single sampler timestep the MM-DiT `predict_flow` arm is compared at. Not 0 or 1: an
#: endpoint can make a broken time embedding look correct.
MMDIT_T = 0.37

FIXTURES = (
    "mmaudio_parity_reference.safetensors",
    "mmaudio_parity_reference_44k.safetensors",
    "mmaudio_parity_reference_44k_decoder.safetensors",
    "mmaudio_parity_output_16k.safetensors",
    "mmaudio_parity_mmdit.safetensors",
)

#: Every tensor each fixture must carry, with its dtype and exact shape. The verifier rejects a
#: fixture whose population or geometry differs, so a producer that silently stops writing a tensor
#: fails in ordinary CI rather than at the next weekly real-weight run.
SCHEMA: dict[str, dict[str, tuple[str, list[int]]]] = {
    "mmaudio_parity_reference.safetensors": {
        "clip_f": ("F32", [1, 64, 1024]),
        "sync_f": ("F32", [1, 192, 768]),
        "text_f": ("F32", [1, 77, 1024]),
        "neg_text_f": ("F32", [1, 77, 1024]),
        "x0": ("F32", [1, 250, 20]),
        "ref_wave": ("F32", [1, 1, 128000]),
        "scalars": ("F32", [4]),
    },
    "mmaudio_parity_reference_44k.safetensors": {
        "clip_f": ("F32", [1, 64, 1024]),
        "sync_f": ("F32", [1, 192, 768]),
        "text_f": ("F32", [1, 77, 1024]),
        "neg_text_f": ("F32", [1, 77, 1024]),
        "x0": ("F32", [1, 345, 40]),
        "ref_wave": ("F32", [1, 1, 353280]),
        "scalars": ("F32", [4]),
    },
    "mmaudio_parity_reference_44k_decoder.safetensors": {
        "z": ("F32", [1, 40, 345]),
        "ref_mel": ("F32", [1, 128, 690]),
        "ref_wave": ("F32", [1, 1, 353280]),
    },
    "mmaudio_parity_output_16k.safetensors": {
        "latent": ("F32", [1, 20, 48]),
        "ref_mel": ("F32", [1, 80, 96]),
        "ref_wave": ("F32", [1, 1, 24576]),
    },
    "mmaudio_parity_mmdit.safetensors": {
        "clip_f": ("F32", [1, 64, 1024]),
        "sync_f": ("F32", [1, 192, 768]),
        "text_f": ("F32", [1, 77, 1024]),
        "latent": ("F32", [1, 250, 20]),
        "t": ("F32", [1]),
        "flow": ("F32", [1, 250, 20]),
        "x0": ("F32", [1, 250, 20]),
        "x1_unnorm": ("F32", [1, 250, 20]),
    },
}


class ReferenceError(RuntimeError):
    """A fixture, snapshot or producer environment that cannot be trusted."""


# -------------------------------------------------------------------------------------------
# Deterministic inputs — numpy only, no weights, no torch. Ordinary CI regenerates these.
# -------------------------------------------------------------------------------------------


def synthetic_frames(count: int, fps: int, size: int) -> np.ndarray:
    """A deterministic RGB clip: drifting colour fields plus a moving bright bar.

    Closed-form in `(t, u, v)` and quantized to uint8, so it is byte-reproducible on any host and
    needs no video asset (and therefore no footage licence). It is only ever fed to the reference
    encoders — the parity claim the fixtures support is about everything *downstream* of them, so
    the clip's provenance does not enter it; what matters is that the features are in-distribution
    and that the bytes never move.

    Returns `(count, size, size, 3)` uint8.
    """
    t = (np.arange(count, dtype=np.float64) / fps)[:, None, None]
    axis = (np.arange(size, dtype=np.float64) + 0.5) / size
    u = axis[None, None, :]
    v = axis[None, :, None]
    tau = 2.0 * np.pi
    red = 0.5 + 0.5 * np.sin(tau * (u + 0.25 * t))
    green = 0.5 + 0.5 * np.sin(tau * (v - 0.15 * t) + tau / 3.0)
    blue = 0.5 + 0.5 * np.sin(tau * (0.5 * (u + v) + 0.35 * t) + 2.0 * tau / 3.0)
    frames = np.broadcast_arrays(red, green, blue)
    stacked = np.stack([np.broadcast_to(c, (count, size, size)) for c in frames], axis=-1)
    # A hard moving edge. The smooth fields alone are band-limited; a synchronised transient is
    # what gives the Synchformer stream something to lock onto.
    bar = np.mod(0.12 * t, 1.0)
    on = (np.abs(u - bar) < 0.02)[..., None]
    stacked = np.where(on, np.minimum(stacked + 0.6, 1.0), stacked)
    return np.rint(stacked * 255.0).astype(np.uint8)


@functools.lru_cache(maxsize=1)
def clip_frames_u8() -> np.ndarray:
    """`(64, 384, 384, 3)` uint8 — the 8 fps CLIP stream over `DURATION` seconds.

    Cached and marked read-only: this is 28 MP, and the ordinary-CI gate re-derives it on every
    `verify_fixtures()` call while exercising a dozen mutation cases.
    """
    frames = synthetic_frames(int(DURATION * CLIP_FPS), CLIP_FPS, CLIP_SIZE)
    frames.flags.writeable = False
    return frames


@functools.lru_cache(maxsize=1)
def sync_frames_u8() -> np.ndarray:
    """`(200, 224, 224, 3)` uint8 — the 25 fps Synchformer stream over `DURATION` seconds."""
    frames = synthetic_frames(int(DURATION * SYNC_FPS), SYNC_FPS, SYNC_SIZE)
    frames.flags.writeable = False
    return frames


def fixed_latent_16k() -> np.ndarray:
    """`(1, 20, 48)` f32, bit-identical to `conformance_output.rs::fixed_latent`.

    The Rust side computes this itself in f64 and stores f32; the same order of operations is
    reproduced here so the two languages cannot drift.
    """
    channel = np.arange(OUTPUT_16K_CHANNELS, dtype=np.float64)[:, None]
    index = np.arange(OUTPUT_16K_LENGTH, dtype=np.float64)[None, :]
    value = 0.3 * np.sin(0.11 * channel + 0.023 * index) + 0.2 * np.cos(
        0.007 * index - 0.05 * channel
    )
    return value.astype(np.float32).reshape(1, OUTPUT_16K_CHANNELS, OUTPUT_16K_LENGTH)


def deterministic_input_digests() -> dict[str, str]:
    """SHA-256 of every weight-free input, for the ordinary-CI gate to re-derive."""
    return {
        "clip_frames_u8": sha256_bytes(clip_frames_u8().tobytes()),
        "sync_frames_u8": sha256_bytes(sync_frames_u8().tobytes()),
        "fixed_latent_16k": sha256_bytes(fixed_latent_16k().tobytes()),
    }


# -------------------------------------------------------------------------------------------
# Digests / manifest
# -------------------------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vendor_tree_digest() -> str:
    """One digest over the vendored reference: sorted `relpath\\0content` for every file.

    The fixtures are only as trustworthy as the code that produced them, so the tree that produced
    them is pinned alongside their own hashes. A local edit to `_vendor/mmaudio` moves this and the
    ordinary-CI gate reports the fixtures as stale.
    """
    digest = hashlib.sha256()
    root = VENDOR / "mmaudio"
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def manifest_revisions() -> dict[str, str]:
    """`{manifest key: pinned revision}` read from the manifest, never restated here."""
    return {
        key: load_model(MANIFEST_PATH, key)["revision"] for key in sorted(MANIFEST_KEYS)
    }


def snapshot_dir(env: str) -> Path:
    """The **required** snapshot directory named by `env`.

    Required, not derived: inference never self-fetches or resolves a Hugging Face cache location
    (epic 13657), and neither does its reference producer.
    """
    raw = os.environ.get(env)
    if not raw:
        raise ReferenceError(f"set {env} to the materialized snapshot directory")
    path = Path(raw)
    if not path.is_dir():
        raise ReferenceError(f"{env} is not a directory: {path}")
    return path


def verify_snapshot_revisions() -> dict[str, str]:
    """Check every snapshot on disk against its manifest pin, and return the pinned revisions."""
    revisions = manifest_revisions()
    for key, env in sorted(MANIFEST_KEYS.items()):
        directory = snapshot_dir(env)
        marker = directory / ".sceneworks-model-revision"
        actual = (
            marker.read_text(encoding="utf-8").strip()
            if marker.is_file()
            else directory.resolve().name
        )
        if actual != revisions[key]:
            raise ReferenceError(
                f"{key}: {env} resolves to revision {actual!r}, manifest pins "
                f"{revisions[key]!r} — regenerate against the pinned snapshot"
            )
    return revisions


# -------------------------------------------------------------------------------------------
# Fixture verification — safetensors only, no torch, no weights.
# -------------------------------------------------------------------------------------------


def read_metadata() -> dict[str, Any]:
    if not METADATA_PATH.is_file():
        raise ReferenceError(f"fixture metadata is missing: {METADATA_PATH}")
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ReferenceError(f"fixture metadata is not valid JSON: {error}") from error


def safetensors_header(path: Path) -> dict[str, Any]:
    """Parse a safetensors header without importing safetensors.

    Deliberately stdlib: the ordinary-CI gate runs on a lane that installs nothing, and the header
    format (little-endian u64 length, then that many bytes of JSON) is stable and trivial.
    """
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ReferenceError(f"{path.name} is truncated before its header length")
        length = int.from_bytes(raw_length, "little")
        if length <= 0 or length > 100 * 1024 * 1024:
            raise ReferenceError(f"{path.name} declares an implausible header length {length}")
        payload = handle.read(length)
        if len(payload) != length:
            raise ReferenceError(f"{path.name} is truncated inside its header")
    try:
        header = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReferenceError(f"{path.name} has an unreadable header: {error}") from error
    if not isinstance(header, dict):
        raise ReferenceError(f"{path.name} header is not an object")
    return header


def fixture_schema(path: Path) -> dict[str, tuple[str, list[int]]]:
    """`{tensor: (dtype, shape)}` for a committed fixture."""
    header = safetensors_header(path)
    schema: dict[str, tuple[str, list[int]]] = {}
    for name, record in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(record, dict) or "dtype" not in record or "shape" not in record:
            raise ReferenceError(f"{path.name}:{name} has no dtype/shape")
        schema[name] = (str(record["dtype"]), list(record["shape"]))
    return schema


def fixture_tensor_bytes(path: Path, name: str) -> bytes:
    """The raw little-endian payload of one tensor in a committed fixture.

    Stdlib, like `safetensors_header`: this exists so the ordinary-CI gate can compare a committed
    tensor against a regenerated one **byte for byte** without installing anything.
    """
    header = safetensors_header(path)
    record = header.get(name)
    if not isinstance(record, dict) or "data_offsets" not in record:
        raise ReferenceError(f"{path.name} has no tensor named {name!r}")
    offsets = record["data_offsets"]
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ReferenceError(f"{path.name}:{name} has malformed data_offsets")
    start, end = int(offsets[0]), int(offsets[1])
    if start < 0 or end < start:
        raise ReferenceError(f"{path.name}:{name} has a negative or inverted extent")
    with path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        handle.seek(8 + header_length + start)
        payload = handle.read(end - start)
    if len(payload) != end - start:
        raise ReferenceError(f"{path.name}:{name} is truncated")
    return payload


def verify_fixtures() -> list[str]:
    """Every check that does not need weights or torch. Returns the accumulated failures."""
    found: list[str] = []
    metadata = read_metadata()

    if metadata.get("schema") != 1:
        found.append(f"metadata schema is {metadata.get('schema')!r}, expected 1")
    if metadata.get("upstreamRepository") != UPSTREAM_REPOSITORY:
        found.append("metadata upstreamRepository does not match the vendored reference")
    if metadata.get("upstreamRevision") != UPSTREAM_REVISION:
        found.append("metadata upstreamRevision does not match the vendored reference pin")
    if metadata.get("vendorTreeSha256") != vendor_tree_digest():
        found.append(
            "the vendored reference tree has changed since the fixtures were produced — "
            "re-run `mmaudio_reference.py dump`"
        )
    for name, expected in (
        ("prompt", PROMPT),
        ("negativePrompt", NEGATIVE_PROMPT),
    ):
        if metadata.get(name) != expected:
            found.append(f"metadata {name} is stale")
    for name, expected in (
        ("seed", SEED),
        ("numSteps", NUM_STEPS),
    ):
        if metadata.get(name) != expected:
            found.append(f"metadata {name} is {metadata.get(name)!r}, expected {expected!r}")
    if metadata.get("cfgStrength") != CFG_STRENGTH:
        found.append("metadata cfgStrength is stale")
    if metadata.get("duration") != DURATION:
        found.append("metadata duration is stale")
    if metadata.get("device") != "cpu" or metadata.get("dtype") != "float32":
        found.append("fixtures must record a cpu/float32 producer run")
    if metadata.get("producerEnvironment") != REFERENCE_PACKAGES:
        found.append(
            "fixtures were produced by a stack that is not the pinned producer environment in "
            "scripts/reference/mmaudio_reference_environment.py — bump the pins and regenerate, "
            "or regenerate against the pins"
        )

    expected_revisions = manifest_revisions()
    if metadata.get("snapshotRevisions") != expected_revisions:
        found.append(
            "fixture snapshot revisions differ from release/real-weight-models.toml — a pin "
            "moved without regenerating the fixtures"
        )

    if metadata.get("deterministicInputs") != deterministic_input_digests():
        found.append(
            "the weight-free deterministic inputs no longer reproduce the digests the fixtures "
            "were produced from — the generator and the fixtures disagree"
        )

    records = metadata.get("files")
    if not isinstance(records, dict) or set(records) != set(FIXTURES):
        found.append("metadata files population does not match the expected fixture set")
        return found

    for name in FIXTURES:
        path = FIXTURE_DIR / name
        record = records[name]
        if not path.is_file():
            found.append(f"fixture is missing: {path}")
            continue
        if not isinstance(record, dict) or set(record) != {"bytes", "sha256"}:
            found.append(f"{name}: metadata record is malformed")
            continue
        if path.stat().st_size != record["bytes"]:
            found.append(f"{name}: size differs from the metadata record")
        if sha256_file(path) != record["sha256"]:
            found.append(f"{name}: SHA-256 differs from the metadata record")
        actual = fixture_schema(path)
        expected = SCHEMA[name]
        if set(actual) != set(expected):
            found.append(
                f"{name}: tensor population differs — missing "
                f"{sorted(set(expected) - set(actual))}, unexpected "
                f"{sorted(set(actual) - set(expected))}"
            )
            continue
        for key, (dtype, shape) in expected.items():
            if actual[key] != (dtype, shape):
                found.append(f"{name}:{key} is {actual[key]}, expected {(dtype, shape)}")
    return found


# -------------------------------------------------------------------------------------------
# Reference execution — needs torch and ~8 GB of real weights.
# -------------------------------------------------------------------------------------------


def _load_reference() -> Any:
    """Import the vendored reference, after asserting the producer environment is the pinned one."""
    # Validated BEFORE the reference is imported: regenerating fixtures from an unpinned stack is
    # the failure this exists to prevent, and it must fail before it can produce a single number.
    validate_reference_environment(ReferenceError)
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - operator-only path
        raise ReferenceError(
            "`dump` needs the pinned producer environment; see the module docstring"
        ) from error
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Belt and braces on epic 13657: every checkpoint below is loaded from an explicit snapshot
    # path, so any hub round-trip left in the reference is a bug. Make it fail loudly instead of
    # silently pulling a different revision than the manifest pins.
    os.environ["HF_HUB_OFFLINE"] = "1"
    return torch


def _clip_model(torch: Any) -> tuple[Any, Any]:
    """Build the DFN5B CLIP tower from the local snapshot, never from the hub.

    open_clip's own entry point for this checkpoint is `hf-hub:apple/DFN5B-CLIP-ViT-H-14-384`,
    which resolves through the Hugging Face hub. The snapshot already on disk carries both halves
    it needs — `open_clip_config.json` and `open_clip_pytorch_model.bin` — so the config is
    registered under a private name and the weights are loaded from the explicit path. The
    vendored reference is not edited to achieve this (the Mage precedent): `FeaturesUtils` is
    constructed with `enable_conditions=False` and the encoders are attached here.
    """
    import open_clip
    from open_clip import factory

    snapshot = snapshot_dir(CLIP_SNAPSHOT_ENV)
    config = json.loads((snapshot / "open_clip_config.json").read_text(encoding="utf-8"))
    name = "sceneworks-dfn5b-clip-vit-h-14-384"
    factory._MODEL_CONFIGS[name] = config["model_cfg"]
    model = factory.create_model(
        name,
        pretrained=str(snapshot / CLIP_WEIGHTS),
        precision="fp32",
        device="cpu",
        require_pretrained=True,
    )
    # 'ViT-H-14-378-quickgelu' shares this checkpoint's vocabulary; open_clip ships the BPE merges
    # in-package, so this resolves with no network access.
    tokenizer = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")
    return model.eval(), tokenizer


def _features_utils(torch: Any, mode: str) -> Any:
    """`FeaturesUtils` for `mode`, with every checkpoint taken from an explicit snapshot path."""
    from mmaudio.model.utils.features_utils import FeaturesUtils, patch_clip
    from mmaudio.ext.synchformer import Synchformer

    mmaudio_snapshot = snapshot_dir(MMAUDIO_SNAPSHOT_ENV)
    if mode == "16k":
        vae_path = mmaudio_snapshot / VAE_16K
        vocoder_path: Path | None = mmaudio_snapshot / BIGVGAN_16K
    else:
        vae_path = mmaudio_snapshot / VAE_44K
        # The 44.1 kHz branch builds NVIDIA BigVGAN v2 through `from_pretrained`, which
        # `_patch_bigvgan_v2_to_local` has already redirected at the explicit nvidia snapshot.
        vocoder_path = None

    utils = FeaturesUtils(
        tod_vae_ckpt=str(vae_path),
        bigvgan_vocoder_ckpt=str(vocoder_path) if vocoder_path is not None else None,
        synchformer_ckpt=None,
        enable_conditions=False,
        mode=mode,
        need_vae_encoder=False,
    )
    clip_model, tokenizer = _clip_model(torch)
    utils.clip_model = patch_clip(clip_model)
    utils.tokenizer = tokenizer
    from torchvision.transforms import Normalize

    utils.clip_preprocess = Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    synchformer = Synchformer()
    synchformer.load_state_dict(
        torch.load(
            str(mmaudio_snapshot / SYNCHFORMER), weights_only=True, map_location="cpu"
        )
    )
    utils.synchformer = synchformer.eval()
    return utils.eval().float()


def _patch_bigvgan_v2_to_local(torch: Any) -> None:
    """Point the 44.1 kHz vocoder at the explicit nvidia snapshot instead of the hub id.

    `AutoEncoderModule` builds it with `BigVGANv2.from_pretrained('nvidia/bigvgan_v2_...')`, which
    resolves through the Hugging Face hub; inference never self-fetches (epic 13657). The class
    attribute is rebound rather than the vendored source edited, so `_vendor/mmaudio` stays a
    verbatim upstream copy — the same discipline the Mage oracle harness keeps.
    `mmaudio.ext.autoencoder.autoencoder` holds a reference to this very class object, so patching
    the class reaches it without depending on which module name the import bound.
    """
    from mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN as BigVGANv2

    snapshot = str(snapshot_dir(BIGVGAN_V2_SNAPSHOT_ENV))
    original = BigVGANv2.from_pretrained

    def from_local(_model_id: str, *args: Any, **kwargs: Any) -> Any:
        return original(snapshot, *args, **kwargs)

    BigVGANv2.from_pretrained = staticmethod(from_local)


def _network(torch: Any, variant: str, checkpoint: Path) -> Any:
    from mmaudio.model.networks import get_my_mmaudio

    net = get_my_mmaudio(variant).to("cpu", torch.float32).eval()
    net.load_weights(torch.load(str(checkpoint), map_location="cpu", weights_only=True))
    return net


def _encode_conditions(torch: Any, utils: Any) -> tuple[Any, Any, Any, Any]:
    """Run the reference's own CLIP / Synchformer / text encoders on the synthetic clip."""
    clip_u8 = clip_frames_u8()
    sync_u8 = sync_frames_u8()
    # `load_video`'s transforms: CLIP frames land in [0, 1]; Synchformer frames are additionally
    # normalized to [-1, 1]. Reproduced here rather than round-tripping through a video file.
    clip_video = torch.from_numpy(clip_u8.astype(np.float32) / 255.0).permute(0, 3, 1, 2)[None]
    sync_video = torch.from_numpy(sync_u8.astype(np.float32) / 255.0).permute(0, 3, 1, 2)[None]
    sync_video = (sync_video - 0.5) / 0.5
    with torch.inference_mode():
        clip_f = utils.encode_video_with_clip(clip_video).float().clone()
        sync_f = utils.encode_video_with_sync(sync_video).float().clone()
        text_f = utils.encode_text([PROMPT]).float().clone()
        neg_text_f = utils.encode_text([NEGATIVE_PROMPT]).float().clone()
    return clip_f, sync_f, text_f, neg_text_f


def _save(torch: Any, name: str, tensors: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    path = FIXTURE_DIR / name
    payload = {key: value.detach().cpu().contiguous().float() for key, value in tensors.items()}
    for key, (dtype, shape) in SCHEMA[name].items():
        if key not in payload:
            raise ReferenceError(f"{name}: producer did not emit {key}")
        if list(payload[key].shape) != shape:
            raise ReferenceError(
                f"{name}:{key} is {list(payload[key].shape)}, schema says {shape} ({dtype})"
            )
        if not bool(torch.isfinite(payload[key]).all()):
            raise ReferenceError(f"{name}:{key} contains non-finite values")
    save_file(payload, str(path))
    print(f"  wrote {path.relative_to(REPO_ROOT)} ({path.stat().st_size} bytes)")


def _sample(torch: Any, net: Any, fm: Any, x0: Any, cond: Any, empty: Any) -> Any:
    def ode(t: Any, x: Any) -> Any:
        return net.ode_wrapper(t, x, cond, empty, CFG_STRENGTH)

    return fm.to_data(ode, x0)


def dump(only: Iterable[str] | None = None) -> None:
    """Run the reference and rewrite every committed fixture."""
    torch = _load_reference()
    from mmaudio.model.flow_matching import FlowMatching
    from mmaudio.model.sequence_config import CONFIG_16K, CONFIG_44K

    wanted = set(only) if only else set(FIXTURES)
    revisions = verify_snapshot_revisions()
    _patch_bigvgan_v2_to_local(torch)
    mmaudio_snapshot = snapshot_dir(MMAUDIO_SNAPSHOT_ENV)
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=NUM_STEPS)

    def prior(seq_len: int, dim: int, seed: int = SEED) -> Any:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        return torch.randn(1, seq_len, dim, device="cpu", dtype=torch.float32, generator=rng)

    # ---- 16 kHz: features, MM-DiT parity, and the assembled pipeline -----------------------
    needs_16k = wanted & {
        "mmaudio_parity_reference.safetensors",
        "mmaudio_parity_mmdit.safetensors",
    }
    if needs_16k:
        print("16k: loading reference (CLIP + Synchformer + small_16k + v1-16 + BigVGAN)")
        utils = _features_utils(torch, "16k")
        net = _network(torch, "small_16k", mmaudio_snapshot / DIT_16K)
        net.update_seq_lengths(
            CONFIG_16K.latent_seq_len, CONFIG_16K.clip_seq_len, CONFIG_16K.sync_seq_len
        )
        clip_f, sync_f, text_f, neg_text_f = _encode_conditions(torch, utils)
        x0 = prior(CONFIG_16K.latent_seq_len, net.latent_dim)

        with torch.inference_mode():
            cond = net.preprocess_conditions(clip_f, sync_f, text_f)

            if "mmaudio_parity_mmdit.safetensors" in wanted:
                print("16k: MM-DiT predict_flow + full CFG sample")
                # `sample_default` on the candle side uses the DEFAULT empty conditions (no
                # negative text), so the reference must too — otherwise the two sides sample
                # different distributions and the gate compares unlike things.
                empty_default = net.get_empty_conditions(1)
                # A DIFFERENT seed from the sampler prior. Sharing one would let a `predict_flow`
                # arm and a `sample` arm agree for the wrong reason, and would leave the single-step
                # comparison covering the same input the 25-step one already starts from.
                latent = prior(CONFIG_16K.latent_seq_len, net.latent_dim, seed=SEED + 1)
                t = torch.tensor([MMDIT_T], dtype=torch.float32)
                flow = net.predict_flow(latent.clone(), t, cond)
                x1 = _sample(torch, net, fm, x0.clone(), cond, empty_default)
                # `unnormalize` is in-place on the reference; clone first so `x1` is not aliased.
                x1_unnorm = net.unnormalize(x1.clone())
                _save(
                    torch,
                    "mmaudio_parity_mmdit.safetensors",
                    {
                        "clip_f": clip_f,
                        "sync_f": sync_f,
                        "text_f": text_f,
                        "latent": latent,
                        "t": t,
                        "flow": flow,
                        "x0": x0,
                        "x1_unnorm": x1_unnorm,
                    },
                )

            if "mmaudio_parity_reference.safetensors" in wanted:
                print("16k: assembled pipeline (negative-text CFG, decode, vocode)")
                empty = net.get_empty_conditions(1, negative_text_features=neg_text_f)
                x1 = _sample(torch, net, fm, x0.clone(), cond, empty)
                latents = net.unnormalize(x1.clone())
                mel = utils.decode(latents)
                wave = utils.vocode(mel)
                _save(
                    torch,
                    "mmaudio_parity_reference.safetensors",
                    {
                        "clip_f": clip_f,
                        "sync_f": sync_f,
                        "text_f": text_f,
                        "neg_text_f": neg_text_f,
                        "x0": x0,
                        "ref_wave": wave,
                        "scalars": torch.tensor(
                            [CFG_STRENGTH, float(NUM_STEPS), DURATION, float(SYNC_FPS)],
                            dtype=torch.float32,
                        ),
                    },
                )
        del utils, net

    # ---- 16 kHz decoder on the closed-form latent -------------------------------------------
    if "mmaudio_parity_output_16k.safetensors" in wanted:
        print("16k: decoder stage on the closed-form latent")
        from mmaudio.ext.autoencoder import AutoEncoderModule

        module = (
            AutoEncoderModule(
                vae_ckpt_path=str(mmaudio_snapshot / VAE_16K),
                vocoder_ckpt_path=str(mmaudio_snapshot / BIGVGAN_16K),
                mode="16k",
                need_vae_encoder=False,
            )
            .eval()
            .float()
        )
        latent = torch.from_numpy(fixed_latent_16k())
        with torch.inference_mode():
            mel = module.decode(latent)
            wave = module.vocode(mel)
        _save(
            torch,
            "mmaudio_parity_output_16k.safetensors",
            {"latent": latent, "ref_mel": mel, "ref_wave": wave},
        )
        del module

    # ---- 44.1 kHz: the assembled pipeline, and the decoder on its own latent -----------------
    needs_44k = wanted & {
        "mmaudio_parity_reference_44k.safetensors",
        "mmaudio_parity_reference_44k_decoder.safetensors",
    }
    if needs_44k:
        print("44k: loading reference (CLIP + Synchformer + large_44k_v2 + v1-44 + BigVGAN v2)")
        utils = _features_utils(torch, "44k")
        net = _network(torch, "large_44k_v2", mmaudio_snapshot / DIT_44K)
        net.update_seq_lengths(
            CONFIG_44K.latent_seq_len, CONFIG_44K.clip_seq_len, CONFIG_44K.sync_seq_len
        )
        clip_f, sync_f, text_f, neg_text_f = _encode_conditions(torch, utils)
        x0 = prior(CONFIG_44K.latent_seq_len, net.latent_dim)
        with torch.inference_mode():
            print("44k: assembled pipeline (this is the slow one — 25 CFG steps on CPU)")
            cond = net.preprocess_conditions(clip_f, sync_f, text_f)
            empty = net.get_empty_conditions(1, negative_text_features=neg_text_f)
            x1 = _sample(torch, net, fm, x0.clone(), cond, empty)
            latents = net.unnormalize(x1.clone())
            mel = utils.decode(latents)
            wave = utils.vocode(mel)
            if "mmaudio_parity_reference_44k.safetensors" in wanted:
                _save(
                    torch,
                    "mmaudio_parity_reference_44k.safetensors",
                    {
                        "clip_f": clip_f,
                        "sync_f": sync_f,
                        "text_f": text_f,
                        "neg_text_f": neg_text_f,
                        "x0": x0,
                        "ref_wave": wave,
                        "scalars": torch.tensor(
                            [CFG_STRENGTH, float(NUM_STEPS), DURATION, float(SYNC_FPS)],
                            dtype=torch.float32,
                        ),
                    },
                )
            if "mmaudio_parity_reference_44k_decoder.safetensors" in wanted:
                # The decoder gate's `z` is the pipeline's OWN latent, not a synthetic one. An
                # arbitrary closed-form latent drives the 44.1 kHz VAE far out of distribution and
                # the vocoder's final tanh saturates, which compresses exactly the differences the
                # gate exists to detect — a saturated waveform scores a high cosine even against a
                # materially wrong port.
                #
                # `FeaturesUtils.decode` transposes `(B, N, C)` into the VAE's `(B, C, N)`, which is
                # the layout `AudioDecoder44k::decode_latent` takes, so `z` is that transpose and
                # the mel/waveform above are already this gate's references — no second decode.
                _save(
                    torch,
                    "mmaudio_parity_reference_44k_decoder.safetensors",
                    {
                        "z": latents.transpose(1, 2).contiguous(),
                        "ref_mel": mel,
                        "ref_wave": wave,
                    },
                )
        del utils, net

    _write_metadata(revisions)


def _write_metadata(revisions: dict[str, str]) -> None:
    # Re-validated rather than remembered from `_load_reference`, so the recorded versions are the
    # ones that were live for the whole run.
    packages = validate_reference_environment(ReferenceError)

    document = {
        "schema": 1,
        "upstreamRepository": UPSTREAM_REPOSITORY,
        "upstreamRevision": UPSTREAM_REVISION,
        "vendorTreeSha256": vendor_tree_digest(),
        "device": "cpu",
        "dtype": "float32",
        "prompt": PROMPT,
        "negativePrompt": NEGATIVE_PROMPT,
        "seed": SEED,
        "cfgStrength": CFG_STRENGTH,
        "numSteps": NUM_STEPS,
        "duration": DURATION,
        "snapshotRevisions": revisions,
        "producerEnvironment": packages,
        "deterministicInputs": deterministic_input_digests(),
        "files": {
            name: {
                "bytes": (FIXTURE_DIR / name).stat().st_size,
                "sha256": sha256_file(FIXTURE_DIR / name),
            }
            for name in FIXTURES
        },
    }
    METADATA_PATH.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {METADATA_PATH.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dump_parser = sub.add_parser("dump", help="run the reference and rewrite the fixtures")
    dump_parser.add_argument(
        "--only",
        action="append",
        choices=FIXTURES,
        help="regenerate a single fixture (repeatable); defaults to all five",
    )
    sub.add_parser("verify", help="weight-free structural verification of the committed fixtures")
    args = parser.parse_args()

    try:
        if args.command == "dump":
            dump(args.only)
        else:
            errors = verify_fixtures()
            if errors:
                for error in errors:
                    print(f"error: {error}", file=sys.stderr)
                return 1
            print(f"verified {len(FIXTURES)} MMAudio parity fixtures under {FIXTURE_DIR}")
    except ReferenceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
