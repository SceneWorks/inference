#!/usr/bin/env python3
"""Generate a frozen end-to-end Stable Audio 3 small-music provider oracle.

Generation is offline and accepts only explicit paths. Every stochastic draw is
replaced with a portable deterministic tensor so the Rust provider can replay
the exact initial, Pingpong, SoftNorm, and decoder-token noise sequence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from unittest import mock

from sa3_reference import (
    InvalidReference,
    SPEC_BY_KEY,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    _localize_t5,
    _torch_modules,
    load_snapshot_lock,
    portable_values as _portable_values,
    sha256_file,
    snapshot_revision,
    validate_upstream_checkout,
)


STORY = "sc-14543"
SNAPSHOT = SPEC_BY_KEY["small-music"]
SEED = 14_543
PROMPT = "Warm analog synth pulses, crisp percussion, spacious stereo field"
DURATION_SECONDS = 30.0
STEPS = 8
ARTIFACT = "provider-output.safetensors"
EXPECTED_MANIFEST_SHA256 = (
    "a893ff4040612942d217dbaa369566909db26bf88d81499ee231834a6acbc982"
)
EXPECTED_ARTIFACT_BYTES = 9_883_912
EXPECTED_ARTIFACT_SHA256 = (
    "0b217e13ba379c714697ac3f34443d0c89c1ad3b2302d885c1c3d7911dc0376f"
)
EXPECTED_SOURCE_SHA256 = {
    "stable_audio_3/inference/sampling.py": (
        "76e96bc3a0c1fb346d13038f09a227ec2f9c7f2e56fd64623e8bbe6ff722bf55"
    ),
    "stable_audio_3/model.py": (
        "8f3a18f1b8f02ffb1ce97f8e64dcb319eb2a6a864d8c405f89d40f6933f3a701"
    ),
    "stable_audio_3/models/autoencoders.py": (
        "44939e6b1c2a72690736757cec3f46cb7f8f6b9ccf9161b738daac320bdfa171"
    ),
    "stable_audio_3/models/bottleneck.py": (
        "1d19a7713dba78de6b87fc003f74a54c40c9ac1604917c827f730363e2214c88"
    ),
}
EXPECTED_TENSORS = {
    "audio": {"dtype": "F16", "shape": [1, 2, 1_323_000]},
    "decoded_chunk_0": {"dtype": "F32", "shape": [1, 2, 524_288]},
    "sampled_latents": {"dtype": "F32", "shape": [1, 256, 388]},
}
ROOT = Path(__file__).resolve().parents[2]


def portable_values(torch, shape: tuple[int, ...], stream: int):
    return _portable_values(torch, shape, stream, seed=SEED)


def validate_upstream(path: Path) -> None:
    validate_upstream_checkout(
        path,
        UPSTREAM_COMMIT,
        error_type=InvalidReference,
        allow_venv=True,
        allow_pycache=True,
        include_ignored=True,
    )


class PortableNoise:
    def __init__(self, torch):
        self.torch = torch
        self.records: list[dict] = []

    def _draw(self, shape, *, device, dtype):
        stream = len(self.records)
        shape = tuple(int(dim) for dim in shape)
        value = portable_values(self.torch, shape, stream).to(
            device=device, dtype=dtype
        )
        self.records.append({"stream": stream, "shape": list(shape)})
        return value

    def randn(self, *shape, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return self._draw(
            shape,
            device=kwargs.get("device", "cpu"),
            dtype=kwargs.get("dtype", self.torch.float32),
        )

    def randn_like(self, reference, **kwargs):
        return self._draw(
            reference.shape,
            device=kwargs.get("device", reference.device),
            dtype=kwargs.get("dtype", reference.dtype),
        )


def generate(upstream: Path, snapshot: Path, output: Path) -> None:
    validate_upstream(upstream)
    if snapshot_revision(snapshot) != SNAPSHOT.revision:
        raise InvalidReference("small-music snapshot revision mismatch")
    locks = load_snapshot_lock()["snapshots"]["small-music"]["files"]
    for name in ("model_config.json", "model.safetensors"):
        if sha256_file(snapshot / name) != locks[name]["sha256"]:
            raise InvalidReference(f"small-music {name} payload mismatch")

    torch, save_file, _sample_diffusion, loaders, versions = _torch_modules(upstream)
    _load_autoencoder, load_diffusion_cond = loaders
    config = json.loads((snapshot / "model_config.json").read_text(encoding="utf-8"))
    wrapper = load_diffusion_cond(
        _localize_t5(config, snapshot),
        str(snapshot / "model.safetensors"),
        device="cpu",
        model_half=False,
    )
    # sc-14537 established CPU F32 as the canonical cross-runtime text policy. Transformers
    # otherwise honors the checkpoint's BF16 declaration on CPU, while the Candle CPU backend
    # deliberately promotes those on-disk weights before compute.
    wrapper.conditioner.conditioners["prompt"].model.float()
    from stable_audio_3.model import StableAudioModel

    model = StableAudioModel(wrapper, config, "cpu", False)
    captured: dict[str, object] = {}
    original_decode = model.same.decode
    original_direct_decode = model.same.model.decode

    def capture_decode(latents, *args, **kwargs):
        captured["sampled_latents"] = latents.detach().cpu()
        return original_decode(latents, *args, **kwargs)

    def capture_direct_decode(latents, *args, **kwargs):
        output = original_direct_decode(latents, *args, **kwargs)
        captured.setdefault("decoded_chunks", []).append(output.detach().cpu())
        return output

    noise = PortableNoise(torch)
    started = time.monotonic()
    with (
        mock.patch.object(torch, "randn", noise.randn),
        mock.patch.object(torch, "randn_like", noise.randn_like),
        mock.patch.object(model.same, "decode", capture_decode),
        mock.patch.object(model.same.model, "decode", capture_direct_decode),
        torch.inference_mode(),
    ):
        audio = model.generate(
            prompt=PROMPT,
            duration=DURATION_SECONDS,
            steps=STEPS,
            cfg_scale=1.0,
            seed=SEED,
            sampler_type="pingpong",
            apg_scale=1.0,
            chunked_decode=True,
        )
    elapsed = time.monotonic() - started
    sampled_latents = captured.get("sampled_latents")
    decoded_chunks = captured.get("decoded_chunks")
    if sampled_latents is None:
        raise InvalidReference("upstream decode did not expose sampled latents")
    if not isinstance(decoded_chunks, list) or len(decoded_chunks) != 4:
        raise InvalidReference("upstream outer decode did not execute four direct chunks")
    expected_shapes = [
        [1, 256, 388],
        *([[1, 256, 388]] * STEPS),
        *(
            shape
            for _ in range(4)
            for shape in ([1, 256, 128], [128, 16, 768])
        ),
    ]
    actual_shapes = [record["shape"] for record in noise.records]
    if actual_shapes != expected_shapes:
        raise InvalidReference(
            f"stochastic call order drifted: {actual_shapes}, expected {expected_shapes}"
        )
    if list(sampled_latents.shape) != [1, 256, 388]:
        raise InvalidReference("sampled latent geometry drifted")
    if list(audio.shape) != [1, 2, 1_323_000]:
        raise InvalidReference("30-second output geometry drifted")
    if not torch.isfinite(audio).all() or float(audio.abs().max()) > 1.0:
        raise InvalidReference("upstream audio is non-finite or unclamped")

    output.mkdir(parents=True, exist_ok=True)
    artifact = output / ARTIFACT
    save_file(
        {
            "sampled_latents": sampled_latents.to(torch.float32).contiguous(),
            "decoded_chunk_0": decoded_chunks[0].to(torch.float32).contiguous(),
            "audio": audio.detach().cpu().to(torch.float16).contiguous(),
        },
        artifact,
        metadata={"story": STORY},
    )
    source_files = (
        "stable_audio_3/model.py",
        "stable_audio_3/inference/sampling.py",
        "stable_audio_3/models/autoencoders.py",
        "stable_audio_3/models/bottleneck.py",
    )
    manifest = {
        "schemaVersion": 1,
        "story": STORY,
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "sourceSha256": {
                name: sha256_file(upstream / name) for name in source_files
            },
        },
        "snapshot": {
            "repository": SNAPSHOT.repository,
            "revision": SNAPSHOT.revision,
            "modelConfigSha256": locks["model_config.json"]["sha256"],
            "modelSha256": locks["model.safetensors"]["sha256"],
        },
        "runtime": versions,
        "computePolicy": {
            "root": "F32",
            "textWeightsOnDisk": "BF16",
            "textCompute": "F32",
            "basis": "sc-14537 canonical CPU policy",
        },
        "request": {
            "prompt": PROMPT,
            "negativePrompt": None,
            "durationSeconds": DURATION_SECONDS,
            "steps": STEPS,
            "guidance": 1.0,
            "sampler": "pingpong",
            "apgScale": 1.0,
            "seed": SEED,
            "chunkedDecode": True,
        },
        "noise": {
            "algorithm": "portable-lcg-u32-affine",
            "draws": noise.records,
        },
        "output": {
            "sampleRate": 44_100,
            "channels": 2,
            "frames": 1_323_000,
            "artifact": ARTIFACT,
            "artifactBytes": artifact.stat().st_size,
            "artifactSha256": sha256_file(artifact),
        },
        "elapsedSeconds": elapsed,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify(output, enforce_repository_pins=False)


def inspect_safetensors(path: Path) -> dict[str, dict]:
    with path.open("rb") as handle:
        header_length_raw = handle.read(8)
        if len(header_length_raw) != 8:
            raise InvalidReference("truncated provider artifact header")
        header_length = int.from_bytes(header_length_raw, "little")
        if header_length <= 0 or header_length > path.stat().st_size - 8:
            raise InvalidReference("invalid provider artifact header length")
        try:
            header = json.loads(handle.read(header_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InvalidReference("invalid provider artifact header") from error
    if not isinstance(header, dict) or header.pop("__metadata__", None) != {
        "story": STORY
    }:
        raise InvalidReference("provider artifact metadata mismatch")
    if set(header) != set(EXPECTED_TENSORS):
        raise InvalidReference("provider artifact tensor set mismatch")
    expected_start = 0
    inspected = {}
    for name, record in sorted(
        header.items(), key=lambda item: item[1].get("data_offsets", [-1])[0]
    ):
        expected = EXPECTED_TENSORS[name]
        if (
            not isinstance(record, dict)
            or record.get("dtype") != expected["dtype"]
            or record.get("shape") != expected["shape"]
        ):
            raise InvalidReference(f"provider artifact tensor header mismatch: {name}")
        offsets = record.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or offsets[0] != expected_start
            or offsets[1] <= offsets[0]
        ):
            raise InvalidReference(f"provider artifact tensor offsets mismatch: {name}")
        expected_start = offsets[1]
        inspected[name] = {
            "dtype": record["dtype"],
            "shape": record["shape"],
        }
    if 8 + header_length + expected_start != path.stat().st_size:
        raise InvalidReference("provider artifact payload length mismatch")
    return inspected


def expected_noise_draws() -> list[dict]:
    shapes = [
        [1, 256, 388],
        *([[1, 256, 388]] * STEPS),
        *(
            shape
            for _ in range(4)
            for shape in ([1, 256, 128], [128, 16, 768])
        ),
    ]
    return [{"shape": shape, "stream": stream} for stream, shape in enumerate(shapes)]


def verify(output: Path, *, enforce_repository_pins: bool = True) -> None:
    manifest_path = output / "manifest.json"
    if enforce_repository_pins and sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise InvalidReference("provider manifest repository pin mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != 1 or manifest.get("story") != STORY:
        raise InvalidReference("provider manifest identity mismatch")
    if manifest.get("upstream") != {
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "sourceSha256": EXPECTED_SOURCE_SHA256,
    }:
        raise InvalidReference("provider upstream provenance mismatch")
    locks = load_snapshot_lock()["snapshots"]["small-music"]["files"]
    if manifest.get("snapshot") != {
        "repository": SNAPSHOT.repository,
        "revision": SNAPSHOT.revision,
        "modelConfigSha256": locks["model_config.json"]["sha256"],
        "modelSha256": locks["model.safetensors"]["sha256"],
    }:
        raise InvalidReference("provider snapshot provenance mismatch")
    if manifest.get("computePolicy") != {
        "root": "F32",
        "textWeightsOnDisk": "BF16",
        "textCompute": "F32",
        "basis": "sc-14537 canonical CPU policy",
    }:
        raise InvalidReference("provider compute policy mismatch")
    if manifest.get("request") != {
        "prompt": PROMPT,
        "negativePrompt": None,
        "durationSeconds": DURATION_SECONDS,
        "steps": STEPS,
        "guidance": 1.0,
        "sampler": "pingpong",
        "apgScale": 1.0,
        "seed": SEED,
        "chunkedDecode": True,
    }:
        raise InvalidReference("provider request mismatch")
    if manifest.get("noise") != {
        "algorithm": "portable-lcg-u32-affine",
        "draws": expected_noise_draws(),
    }:
        raise InvalidReference("provider stochastic schedule mismatch")
    expected_output = {
        "sampleRate": 44_100,
        "channels": 2,
        "frames": 1_323_000,
        "artifact": ARTIFACT,
        "artifactBytes": EXPECTED_ARTIFACT_BYTES,
        "artifactSha256": EXPECTED_ARTIFACT_SHA256,
    }
    if manifest.get("output") != expected_output:
        raise InvalidReference("provider output record mismatch")
    artifact = output / ARTIFACT
    if (
        artifact.stat().st_size != EXPECTED_ARTIFACT_BYTES
        or sha256_file(artifact) != EXPECTED_ARTIFACT_SHA256
    ):
        raise InvalidReference("provider artifact repository pin mismatch")
    if inspect_safetensors(artifact) != EXPECTED_TENSORS:
        raise InvalidReference("provider artifact tensor records mismatch")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/migration/sa3-small-music-provider-reference",
    )
    args = parser.parse_args()
    if not args.verify and not all((args.upstream, args.snapshot)):
        parser.error("generation requires --upstream and --snapshot")
    return args


def main() -> None:
    args = parse_args()
    if args.verify:
        verify(args.output.resolve())
    else:
        generate(
            args.upstream.resolve(), args.snapshot.resolve(), args.output.resolve()
        )


if __name__ == "__main__":
    try:
        main()
    except (InvalidReference, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
