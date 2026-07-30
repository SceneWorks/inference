#!/usr/bin/env python3
"""Generate or verify frozen SAME outer-chunking parity evidence for sc-14540."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import struct
import subprocess
import sys
from pathlib import Path
from unittest import mock


UPSTREAM_COMMIT = "124e8a799f57a1f665495ecb72e547d0a62867f1"
UPSTREAM_REPOSITORY = "https://github.com/Stability-AI/stable-audio-3.git"
SAME_S_REVISION = "fbeb3dcf53a326e5682f38e22e7f740202d44232"
SAME_L_REVISION = "41acf79dd242877d6499a1108ca5dba5d5eecfc5"
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "torch": "2.7.1",
    "torchaudio": "2.7.1",
    "transformers": "5.8.0",
}
SEED = 14_540
LATENTS = 225
RATIO = 4096
CHUNK_SIZE = 128
OVERLAP = 32
HOP = CHUNK_SIZE - OVERLAP
HALF_OVERLAP = OVERLAP // 2
ARTIFACT = "chunked-f32.safetensors"
OUTPUTS_ARTIFACT = "chunked-outputs-f16.safetensors"

# Updated only after independent review of regenerated evidence. Generation does not bless itself.
EXPECTED_MANIFEST_SHA256 = "8127254d720a92c74f115d92a32da67bbee33efffd913c78dd519fa686cc1a80"
EXPECTED_ARTIFACT_SHA256 = "ab537ae1803d0b74834e5c32c3a3c677e16bd0d41156116971ae881e556996a3"
EXPECTED_ARTIFACT_BYTES = 2_035_792
EXPECTED_OUTPUTS_SHA256 = "7a1a3eaf67506d8c6e0a62d9004b2886826d97e10adf1e7b617670e17615d965"
EXPECTED_OUTPUTS_BYTES = 22_365_984


class InvalidReference(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_revision(path: Path, revision: str, label: str) -> None:
    if path.resolve().name != revision:
        raise InvalidReference(
            f"{label} revision mismatch: {path.resolve().name}, expected {revision}"
        )

def require_snapshot_files(
    snapshot: Path, revision: str, label: str, lock_entry: dict
) -> dict[str, str]:
    require_revision(snapshot, revision, label)
    hashes = {}
    for filename in ("model_config.json", "model.safetensors"):
        path = snapshot / filename
        actual = sha256_file(path)
        expected = lock_entry["files"][filename]["sha256"]
        if actual != expected:
            raise InvalidReference(
                f"{label} {filename} hash mismatch: {actual}, expected {expected}"
            )
        hashes[filename] = actual
    return hashes


def validate_upstream(path: Path) -> None:
    revision = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if revision != UPSTREAM_COMMIT:
        raise InvalidReference(f"upstream revision mismatch: {revision}")
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if status:
        raise InvalidReference(f"upstream checkout is not clean: {status}")


def portable_values(torch, shape, stream: int):
    count = 1
    for dim in shape:
        count *= dim
    index = torch.arange(count, dtype=torch.int64)
    bits = (index * 1_664_525 + (SEED + stream) * 1_013_904_223) & 0xFFFF_FFFF
    return (bits.to(torch.float64) / 2_147_483_648.0 - 1.0).to(
        torch.float32
    ).reshape(shape)


def chunk_starts(total: int) -> list[int]:
    starts = list(range(0, total - CHUNK_SIZE + 1, HOP))
    if starts[-1] != total - CHUNK_SIZE:
        starts.append(total - CHUNK_SIZE)
    return starts


def ownership(starts: list[int]) -> list[dict]:
    result = []
    for index, start in enumerate(starts):
        output_start = 0 if index == 0 else start + HALF_OVERLAP
        output_end = LATENTS if index + 1 == len(starts) else starts[index + 1] + HALF_OVERLAP
        result.append(
            {
                "chunkIndex": index,
                "chunkStart": start,
                "source": [output_start - start, output_end - start],
                "output": [output_start, output_end],
            }
        )
    return result


def noise_patch(torch, base_stream: int, records: list[dict]):
    calls = 0

    def injected(reference, *args, **kwargs):
        nonlocal calls
        stream = base_stream + calls
        records.append({"stream": stream, "shape": list(reference.shape)})
        calls += 1
        return portable_values(torch, tuple(reference.shape), stream).to(
            device=reference.device, dtype=reference.dtype
        )

    return injected


def capture_chunk_calls(model, method_name: str, run):
    original = getattr(model, method_name)
    chunks = []

    def wrapped(*args, **kwargs):
        output = original(*args, **kwargs)
        chunks.append(output.detach().cpu())
        return output

    with mock.patch.object(model, method_name, wrapped):
        output = run()
    return output.detach().cpu(), chunks


def add_slices(
    tensors: dict,
    prefix: str,
    value,
    starts: list[int],
    width: int,
    boundary_scale: int = 1,
):
    length = value.shape[-1]
    positions = {0, max(0, length - min(width, length))}
    for start in starts:
        boundary = (start + HALF_OVERLAP) * boundary_scale
        positions.add(min(max(0, boundary - width // 2), max(0, length - width)))
    for start in sorted(positions):
        tensors[f"{prefix}.slice_{start}"] = value[
            ..., start : start + min(width, length)
        ].contiguous()


def spectral_boundaries(torch, value, starts: list[int]) -> list[dict]:
    width = 2048
    window = torch.hann_window(width, periodic=True)
    records = []
    for start in starts[1:]:
        boundary = (start + HALF_OVERLAP) * RATIO
        left = value[..., boundary - width : boundary] * window
        right = value[..., boundary : boundary + width] * window
        left_spectrum = torch.log1p(torch.fft.rfft(left).abs())
        right_spectrum = torch.log1p(torch.fft.rfft(right).abs())
        records.append(
            {
                "sample": boundary,
                "jumpMaxAbs": float(
                    (value[..., boundary] - value[..., boundary - 1])
                    .abs()
                    .max()
                    .item()
                ),
                "logMagnitudeL1": float(
                    (left_spectrum - right_spectrum).abs().mean().item()
                ),
            }
        )
    return records


def tensor_records(torch, tensors: dict) -> dict:
    from safetensors.torch import save

    records = {}
    for name, value in tensors.items():
        value = value.detach().cpu().contiguous()
        payload_bytes = value.numel() * value.element_size()
        serialized = save({"x": value})
        records[name] = {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(
                serialized[-payload_bytes:] if payload_bytes else b""
            ).hexdigest(),
        }
    return records


def inspect_safetensors(path: Path, story: str) -> dict:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise InvalidReference(f"truncated safetensors: {path}")
    header_length = struct.unpack("<Q", raw[:8])[0]
    data_start = 8 + header_length
    header = json.loads(raw[8:data_start].decode("utf-8"))
    if header.pop("__metadata__", None) != {"story": story}:
        raise InvalidReference(f"metadata mismatch: {path}")
    records = {}
    for name, entry in header.items():
        start, end = entry["data_offsets"]
        records[name] = {
            "dtype": {"F32": "float32", "F16": "float16"}[entry["dtype"]],
            "shape": entry["shape"],
            "sha256": hashlib.sha256(
                raw[data_start + start : data_start + end]
            ).hexdigest(),
        }
    return records


def run_model(torch, model, label: str, tensors: dict, outputs: dict) -> dict:
    starts = chunk_starts(LATENTS)
    audio = portable_values(torch, (1, 2, LATENTS * RATIO), 900) * 0.25
    latents = portable_values(torch, (1, 256, LATENTS), 901) * 0.125

    encode_noise = []
    with mock.patch.object(
        torch, "randn_like", noise_patch(torch, 0, encode_noise)
    ), torch.inference_mode():
        encoded, encoded_chunks = capture_chunk_calls(
            model,
            "encode",
            lambda: model.encode_audio(
                audio, chunked=True, chunk_size=CHUNK_SIZE, overlap=OVERLAP
            ),
        )

    decode_noise = []
    with mock.patch.object(
        torch, "randn_like", noise_patch(torch, 100, decode_noise)
    ), torch.inference_mode():
        decoded, decoded_chunks = capture_chunk_calls(
            model,
            "decode",
            lambda: model.decode_audio(
                latents, chunked=True, chunk_size=CHUNK_SIZE, overlap=OVERLAP
            ),
        )

    zero_noise = lambda reference, *args, **kwargs: torch.zeros_like(reference)
    with mock.patch.object(torch, "randn_like", zero_noise), torch.inference_mode():
        decoded_direct_zero = model.decode_audio(latents, chunked=False)
        decoded_chunked_zero = model.decode_audio(
            latents, chunked=True, chunk_size=CHUNK_SIZE, overlap=OVERLAP
        )
    decoded_direct_zero = decoded_direct_zero.detach().cpu()
    decoded_chunked_zero = decoded_chunked_zero.detach().cpu()

    add_slices(tensors, f"{label}.encoded", encoded, starts, 64)
    add_slices(
        tensors,
        f"{label}.decoded",
        decoded,
        starts,
        RATIO,
        boundary_scale=RATIO,
    )
    for index, chunk in enumerate(encoded_chunks):
        add_slices(tensors, f"{label}.encoded_chunk_{index}", chunk, [], 64)
    for index, chunk in enumerate(decoded_chunks):
        add_slices(tensors, f"{label}.decoded_chunk_{index}", chunk, [], RATIO)
    outputs[f"{label}.encoded"] = encoded.to(torch.float16)
    outputs[f"{label}.decoded"] = decoded.to(torch.float16)
    outputs[f"{label}.decoded_direct_zero"] = decoded_direct_zero.to(torch.float16)
    outputs[f"{label}.decoded_chunked_zero"] = decoded_chunked_zero.to(torch.float16)

    direct_metrics = spectral_boundaries(torch, decoded_direct_zero, starts)
    chunked_metrics = spectral_boundaries(torch, decoded_chunked_zero, starts)
    for direct, chunked in zip(direct_metrics, chunked_metrics, strict=True):
        chunked["directLogMagnitudeL1"] = direct["logMagnitudeL1"]
        chunked["addedLogMagnitudeL1"] = abs(
            chunked["logMagnitudeL1"] - direct["logMagnitudeL1"]
        )

    return {
        "encodeNoise": encode_noise,
        "decodeNoise": decode_noise,
        "encodedShape": list(encoded.shape),
        "decodedShape": list(decoded.shape),
        "spectralBoundariesZeroNoise": chunked_metrics,
    }


def generate(args) -> None:
    upstream = args.upstream.resolve()
    same_s = args.same_s.resolve()
    same_l = args.same_l.resolve()
    output = args.output.resolve()
    validate_upstream(upstream)
    require_revision(same_s, SAME_S_REVISION, "SAME-S")
    require_revision(same_l, SAME_L_REVISION, "SAME-L")
    versions = {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "torchaudio": importlib.metadata.version("torchaudio"),
        "transformers": importlib.metadata.version("transformers"),
    }
    if versions != EXPECTED_RUNTIME:
        raise InvalidReference(f"runtime mismatch: {versions}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(upstream))
    import torch
    from safetensors.torch import save_file
    from stable_audio_3.loading_utils import load_autoencoder
    import stable_audio_3.models.transformer as upstream_transformer

    upstream_transformer.flex_attention_available = False
    upstream_transformer.flex_attention_compiled = None
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    lock = json.loads(
        (root / "docs/migration/sa3-reference/snapshot-files.json").read_text(
            encoding="utf-8"
        )
    )
    snapshot_hashes = {
        "same_s": require_snapshot_files(
            same_s, SAME_S_REVISION, "SAME-S", lock["snapshots"]["same-s"]
        ),
        "same_l": require_snapshot_files(
            same_l, SAME_L_REVISION, "SAME-L", lock["snapshots"]["same-l"]
        ),
    }
    tensors = {}
    outputs = {}
    cases = {}
    for label, snapshot in (("same_s", same_s), ("same_l", same_l)):
        model = load_autoencoder(
            str(snapshot / "model_config.json"),
            str(snapshot / "model.safetensors"),
            device="cpu",
        ).eval()
        cases[label] = run_model(torch, model, label, tensors, outputs)
        del model

    artifact = output / ARTIFACT
    save_file(
        {name: value.contiguous() for name, value in tensors.items()},
        str(artifact),
        metadata={"story": "sc-14540"},
    )
    outputs_artifact = output / OUTPUTS_ARTIFACT
    save_file(
        {name: value.contiguous() for name, value in outputs.items()},
        str(outputs_artifact),
        metadata={"story": "sc-14540"},
    )
    upstream_file = upstream / "stable_audio_3/models/autoencoders.py"
    manifest = {
        "schemaVersion": 1,
        "story": "sc-14540",
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "autoencodersSha256": sha256_file(upstream_file),
            "encodeLines": [550, 594],
            "decodeLines": [596, 638],
        },
        "runtime": versions,
        "portableNoise": {
            "seed": SEED,
            "equation": "((i*1664525 + (seed+stream)*1013904223) & 0xffffffff)/2147483648 - 1",
            "audioStream": 900,
            "audioScale": 0.25,
            "latentStream": 901,
            "latentScale": 0.125,
        },
        "snapshots": {
            "same_s": {
                "revision": SAME_S_REVISION,
                "configSha256": snapshot_hashes["same_s"]["model_config.json"],
                "modelSha256": snapshot_hashes["same_s"]["model.safetensors"],
            },
            "same_l": {
                "revision": SAME_L_REVISION,
                "configSha256": snapshot_hashes["same_l"]["model_config.json"],
                "modelSha256": snapshot_hashes["same_l"]["model.safetensors"],
            },
        },
        "plan": {
            "totalLatents": LATENTS,
            "ratio": RATIO,
            "chunkSize": CHUNK_SIZE,
            "overlap": OVERLAP,
            "hop": HOP,
            "halfOverlapFloor": HALF_OVERLAP,
            "starts": chunk_starts(LATENTS),
            "ownership": ownership(chunk_starts(LATENTS)),
        },
        "cases": cases,
        "artifact": {
            "file": ARTIFACT,
            "bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
            "tensors": tensor_records(torch, tensors),
        },
        "outputsArtifact": {
            "file": OUTPUTS_ARTIFACT,
            "bytes": outputs_artifact.stat().st_size,
            "sha256": sha256_file(outputs_artifact),
            "storageDtype": "float16",
            "tensors": tensor_records(torch, outputs),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify(output, enforce_repository_pins=False)


def verify(output: Path, *, enforce_repository_pins: bool = True) -> None:
    manifest_path = output / "manifest.json"
    if enforce_repository_pins and (
        not EXPECTED_MANIFEST_SHA256
        or sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256
    ):
        raise InvalidReference("manifest repository pin mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != 1 or manifest.get("story") != "sc-14540":
        raise InvalidReference("manifest identity mismatch")
    if manifest["plan"]["starts"] != [0, 96, 97]:
        raise InvalidReference("frozen final-anchor plan mismatch")
    if [entry["output"] for entry in manifest["plan"]["ownership"]] != [
        [0, 112],
        [112, 113],
        [113, 225],
    ]:
        raise InvalidReference("frozen later-writer ownership mismatch")
    artifact = output / manifest["artifact"]["file"]
    outputs = output / manifest["outputsArtifact"]["file"]
    if sha256_file(artifact) != manifest["artifact"]["sha256"]:
        raise InvalidReference("F32 artifact manifest hash mismatch")
    if sha256_file(outputs) != manifest["outputsArtifact"]["sha256"]:
        raise InvalidReference("outputs artifact manifest hash mismatch")
    if inspect_safetensors(artifact, "sc-14540") != manifest["artifact"]["tensors"]:
        raise InvalidReference("F32 tensor records mismatch")
    if (
        inspect_safetensors(outputs, "sc-14540")
        != manifest["outputsArtifact"]["tensors"]
    ):
        raise InvalidReference("output tensor records mismatch")
    if enforce_repository_pins and (
        artifact.stat().st_size != EXPECTED_ARTIFACT_BYTES
        or sha256_file(artifact) != EXPECTED_ARTIFACT_SHA256
        or outputs.stat().st_size != EXPECTED_OUTPUTS_BYTES
        or sha256_file(outputs) != EXPECTED_OUTPUTS_SHA256
    ):
        raise InvalidReference("artifact repository pin mismatch")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--same-s", type=Path)
    parser.add_argument("--same-l", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/migration/sa3-chunked-reference"),
    )
    args = parser.parse_args()
    if not args.verify and not all((args.upstream, args.same_s, args.same_l)):
        parser.error("generation requires --upstream, --same-s, and --same-l")
    return args


def main() -> None:
    args = parse_args()
    if args.verify:
        verify(args.output.resolve())
    else:
        generate(args)


if __name__ == "__main__":
    try:
        main()
    except (InvalidReference, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
