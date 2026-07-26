#!/usr/bin/env python3
"""Generate/verify provenance-locked SAME-L sliding-window parity evidence.

Generation is offline and accepts only explicit immutable snapshot/check-out paths.  Large token
noise is reconstructed from a portable integer sequence instead of being committed, so long
duration evidence stays compact while exercising the exact upstream override points.
"""

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
UPSTREAM_FILES = ("autoencoders.py", "bottleneck.py", "transformer.py")
SAME_L_REVISION = "41acf79dd242877d6499a1108ca5dba5d5eecfc5"
MEDIUM_REVISION = "27b5a21b791b1b033d193a9e1e3ce78493f102f9"
MEDIUM_BASE_REVISION = "b32993f73c3bdc3864043a72d8032606bba737c8"
SEED = 14539
ARTIFACT = "same-l.safetensors"
EXTENDED_ARTIFACT = "same-l-extended.safetensors"
OUTPUTS_ARTIFACT = "same-l-outputs-f16.safetensors"
RESOURCE_EVIDENCE = "resource-evidence.json"
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "torch": "2.7.1",
    "torchaudio": "2.7.1",
    "transformers": "5.8.0",
}
CASES = {
    "short": 16_384,
    "ten_seconds": 441_000,
    "long_120_seconds": 5_292_000,
}
BOUNDARY_WIDTH = 35
QUERY_TILE = 1024
# Updated only after independently reviewing a regenerated fixture. Generation intentionally does
# not bless its own output; the standalone verifier fails closed against these repository pins.
EXPECTED_MANIFEST_SHA256 = "dd24d788938333ead1abc6958d96e5e257f79adf248d0190177ccdb57339c104"
EXPECTED_ARTIFACT_SHA256 = "b77cdc73eac861f3f8fdd6b29271caf38b5de47198ce63b4dcea7b60f2645e94"
EXPECTED_ARTIFACT_BYTES = 72_828_184
EXPECTED_EXTENDED_ARTIFACT_SHA256 = "256271358a45b42ce4400206874fa2673336e0802fdd0cfc7d0dc1876b053598"
EXPECTED_EXTENDED_ARTIFACT_BYTES = 57_273_040
EXPECTED_OUTPUTS_ARTIFACT_SHA256 = "fac3744ffaba6461767f6ad49105f46333e5b611e5c35a1e7b6a3cc68b7a4aab"
EXPECTED_OUTPUTS_ARTIFACT_BYTES = 47_522_248
EXPECTED_RESOURCE_EVIDENCE_SHA256 = (
    "f51bc9cc6fa6b245b915ec5cee8bf654e186ab47a73f42f2d6a705b3cf4d1e11"
)
EXPECTED_RESOURCE_EVIDENCE_BYTES = 1_229


class InvalidReference(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def safetensors_prefix_digest(path: Path, prefix: str) -> dict:
    """Hash a namespaced tensor payload without materializing it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        header_length_raw = handle.read(8)
        if len(header_length_raw) != 8:
            raise InvalidReference(f"truncated safetensors header: {path}")
        header_length = struct.unpack("<Q", header_length_raw)[0]
        header = json.loads(handle.read(header_length).decode("utf-8"))
        data_start = 8 + header_length
        selected = sorted(
            (name, entry)
            for name, entry in header.items()
            if name != "__metadata__" and name.startswith(prefix)
        )
        if not selected:
            raise InvalidReference(f"no tensors use prefix {prefix!r}: {path}")
        payload_bytes = 0
        for name, entry in selected:
            start, end = entry["data_offsets"]
            length = end - start
            digest.update(name.removeprefix(prefix).encode("utf-8"))
            digest.update(b"\0")
            digest.update(entry["dtype"].encode("ascii"))
            digest.update(b"\0")
            digest.update(json.dumps(entry["shape"], separators=(",", ":")).encode("ascii"))
            digest.update(b"\0")
            handle.seek(data_start + start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise InvalidReference(f"truncated tensor payload: {path}:{name}")
                digest.update(chunk)
                remaining -= len(chunk)
            payload_bytes += length
    return {
        "prefix": prefix,
        "tensors": len(selected),
        "bytes": payload_bytes,
        "sha256": digest.hexdigest(),
    }


def require_revision(path: Path, revision: str, label: str) -> None:
    if path.resolve().name != revision:
        raise InvalidReference(
            f"{label} revision mismatch: {path.resolve().name}, expected {revision}"
        )


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
        ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    if any(not line.endswith(" .venv/") for line in status):
        raise InvalidReference(f"upstream checkout is not clean: {status}")


def portable_values(torch, shape, stream: int):
    """Exact u32 LCG values mapped to [-1,1]; deliberately not a Gaussian distribution."""
    count = 1
    for dim in shape:
        count *= dim
    index = torch.arange(count, dtype=torch.int64)
    bits = (index * 1_664_525 + (SEED + stream) * 1_013_904_223) & 0xFFFF_FFFF
    return (bits.to(torch.float64) / 2_147_483_648.0 - 1.0).to(
        torch.float32
    ).reshape(shape)


def fixed_audio(torch, samples: int):
    return portable_values(torch, (1, 2, samples), 100) * 0.25


def slice_starts(length: int) -> list[int]:
    width = min(BOUNDARY_WIDTH, length)
    starts = {0, max(0, length - width)}
    for boundary in (17, QUERY_TILE, length // 2):
        if 0 < boundary < length:
            starts.add(min(max(0, boundary - width // 2), length - width))
    return sorted(starts)


def capture_layers(block, prefix: str, tensors: dict):
    handles = []
    for index, layer in enumerate(block.transformers):
        def hook(_module, _args, output, index=index):
            value = output.detach()
            for start in slice_starts(value.shape[1]):
                tensors[f"{prefix}.block_{index}.slice_{start}"] = value[
                    :, start : start + min(BOUNDARY_WIDTH, value.shape[1]), :
                ]

        handles.append(layer.register_forward_hook(hook))
    return handles


def output_slices(value, prefix: str, tensors: dict, width: int):
    for start in sorted({0, max(0, value.shape[-1] - min(width, value.shape[-1]))}):
        tensors[f"{prefix}.slice_{start}"] = value[
            ..., start : start + min(width, value.shape[-1])
        ]


def run_case(
    torch,
    model,
    label: str,
    samples: int,
    tensors: dict,
    outputs: dict,
    stride_override: int | None = None,
):
    encoder = next(
        layer
        for layer in model.encoder.layers
        if layer.__class__.__name__ == "TransformerResamplingBlock"
    )
    decoder = next(
        layer
        for layer in model.decoder.layers
        if layer.__class__.__name__ == "TransformerResamplingBlock"
    )
    handles = capture_layers(encoder, f"{label}.encoder", tensors)
    handles += capture_layers(decoder, f"{label}.decoder", tensors)
    calls = 0

    def injected(reference, *args, **kwargs):
        nonlocal calls
        value = portable_values(torch, tuple(reference.shape), calls)
        calls += 1
        return value.to(device=reference.device, dtype=reference.dtype)

    audio = fixed_audio(torch, samples)
    with mock.patch.object(torch, "randn_like", injected), torch.inference_mode():
        kwargs = (
            {"override_stride": [stride_override]}
            if stride_override is not None
            else {}
        )
        latents = model.encode(audio, **kwargs)
        decoded = model.decode(latents, **kwargs)
    for handle in handles:
        handle.remove()
    if calls != 3:
        raise InvalidReference(f"{label}: expected three injected noise calls, got {calls}")
    tensors[f"{label}.audio_samples"] = torch.tensor([samples], dtype=torch.int64)
    output_slices(latents, f"{label}.latents", tensors, 64)
    output_slices(decoded, f"{label}.decoded", tensors, 4096)
    outputs[f"{label}.latents"] = latents
    outputs[f"{label}.decoded"] = decoded
    return {
        "inputSamples": samples,
        "paddedSamples": int(decoded.shape[-1]),
        "latentLength": int(latents.shape[-1]),
        "activeStride": stride_override or 16,
        "encoderPackedLength": int(
            latents.shape[-1] * ((stride_override or 16) + 1)
        ),
        "noiseStreams": [0, 1, 2],
    }


def tensor_records(torch, tensors):
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


def inspect_safetensors(path: Path):
    raw = path.read_bytes()
    if len(raw) < 8:
        raise InvalidReference("truncated safetensors")
    header_len = struct.unpack("<Q", raw[:8])[0]
    data_start = 8 + header_len
    header = json.loads(raw[8:data_start].decode("utf-8"))
    if header.pop("__metadata__", None) != {"story": "sc-14539"}:
        raise InvalidReference("safetensors metadata mismatch")
    records = {}
    for name, entry in header.items():
        start, end = entry["data_offsets"]
        records[name] = {
            "dtype": {
                "F32": "float32",
                "F16": "float16",
                "I64": "int64",
            }[entry["dtype"]],
            "shape": entry["shape"],
            "sha256": hashlib.sha256(
                raw[data_start + start : data_start + end]
            ).hexdigest(),
        }
    return records


def generate(args) -> None:
    upstream = args.upstream.resolve()
    same_l = args.same_l.resolve()
    medium = args.medium.resolve()
    medium_base = args.medium_base.resolve()
    output = args.output.resolve()
    validate_upstream(upstream)
    require_revision(same_l, SAME_L_REVISION, "SAME-L")
    require_revision(medium, MEDIUM_REVISION, "medium")
    require_revision(medium_base, MEDIUM_BASE_REVISION, "medium-base")
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

    # CPU flex_attention's eager fallback materializes dense scores. Exercise the frozen upstream
    # chunked-halo fallback explicitly, which is the source path being ported.
    upstream_transformer.flex_attention_available = False
    upstream_transformer.flex_attention_compiled = None

    output.mkdir(parents=True, exist_ok=True)
    tensors = {}
    extended_tensors = {}
    outputs = {}
    cases = {}
    for snapshot_label, snapshot in (("standalone", same_l), ("embedded", medium)):
        model = load_autoencoder(
            str(snapshot / "model_config.json"),
            str(snapshot / "model.safetensors"),
            device="cpu",
        ).eval()
        durations = {"short": CASES["short"]} if args.quick else CASES
        for duration, samples in durations.items():
            label = f"{snapshot_label}.{duration}"
            destination = (
                tensors
                if snapshot_label == "standalone" or duration == "short"
                else extended_tensors
            )
            cases[label] = run_case(
                torch, model, label, samples, destination, outputs
            )
        if snapshot_label == "standalone":
            label = "standalone.stride7"
            cases[label] = run_case(
                torch,
                model,
                label,
                CASES["short"],
                extended_tensors,
                outputs,
                stride_override=7,
            )
        del model
    artifact = output / ARTIFACT
    save_file(
        {
            name: value.detach().cpu().contiguous().clone()
            for name, value in tensors.items()
        },
        str(artifact),
        metadata={"story": "sc-14539"},
    )
    extended_artifact = output / EXTENDED_ARTIFACT
    save_file(
        {
            name: value.detach().cpu().contiguous().clone()
            for name, value in extended_tensors.items()
        },
        str(extended_artifact),
        metadata={"story": "sc-14539"},
    )
    outputs_artifact = output / OUTPUTS_ARTIFACT
    save_file(
        {
            name: value.detach().cpu().to(torch.float16).contiguous().clone()
            for name, value in outputs.items()
        },
        str(outputs_artifact),
        metadata={"story": "sc-14539"},
    )
    lock = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "docs/migration/sa3-reference/snapshot-files.json"
        ).read_text(encoding="utf-8")
    )
    embedded_config = json.loads(
        (medium / "model_config.json").read_text(encoding="utf-8")
    )["model"]["pretransform"]["config"]
    embedded_base_config = json.loads(
        (medium_base / "model_config.json").read_text(encoding="utf-8")
    )["model"]["pretransform"]["config"]
    if embedded_config != embedded_base_config:
        raise InvalidReference("medium and medium-base autoencoder configs differ")
    embedded_payload = safetensors_prefix_digest(
        medium / "model.safetensors", "pretransform.model."
    )
    embedded_base_payload = safetensors_prefix_digest(
        medium_base / "model.safetensors", "pretransform.model."
    )
    if embedded_payload != embedded_base_payload:
        raise InvalidReference("medium and medium-base autoencoder payloads differ")
    manifest = {
        "schemaVersion": 1,
        "story": "sc-14539",
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "files": {
                name: sha256_file(upstream / "stable_audio_3/models" / name)
                for name in UPSTREAM_FILES
            },
        },
        "runtime": versions,
        "environment": {
            "system": platform.system(),
            "machine": platform.machine(),
            "torchDevice": "cpu",
            "slidingBackend": "_sliding_window_chunked_halo_sdpa",
        },
        "portableNoise": {
            "seed": SEED,
            "equation": "((i*1664525 + (seed+stream)*1013904223) & 0xffffffff)/2147483648 - 1",
            "order": ["encoder_tokens", "softnorm_decode", "decoder_tokens"],
            "scales": [0.001, 0.001, 0.1],
            "audioStream": 100,
            "audioScale": 0.25,
        },
        "snapshots": {
            "standalone": {
                "revision": SAME_L_REVISION,
                "modelSha256": lock["snapshots"]["same-l"]["files"]["model.safetensors"]["sha256"],
                "prefix": "",
            },
            "embedded": {
                "revision": MEDIUM_REVISION,
                "modelSha256": lock["snapshots"]["medium"]["files"]["model.safetensors"]["sha256"],
                "prefix": "pretransform.model.",
            },
            "embeddedBase": {
                "revision": MEDIUM_BASE_REVISION,
                "modelSha256": lock["snapshots"]["medium-base"]["files"]["model.safetensors"]["sha256"],
                "prefix": "pretransform.model.",
            },
        },
        "embeddedIdentity": {
            "configSha256": canonical_json_sha256(embedded_config),
            "payload": embedded_payload,
        },
        "architecture": {
            "stride": 16,
            "subchunk": 17,
            "window": [17, 17],
            "queryTile": QUERY_TILE,
            "depth": 12,
            "dim": 1536,
            "heads": 24,
            "sinusoidalBlocks": list(range(5, 12)),
        },
        "cases": cases,
        "artifact": {
            "file": ARTIFACT,
            "bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
            "tensors": tensor_records(torch, tensors),
        },
        "extendedArtifact": {
            "file": EXTENDED_ARTIFACT,
            "bytes": extended_artifact.stat().st_size,
            "sha256": sha256_file(extended_artifact),
            "tensors": tensor_records(torch, extended_tensors),
        },
        "outputsArtifact": {
            "file": OUTPUTS_ARTIFACT,
            "bytes": outputs_artifact.stat().st_size,
            "sha256": sha256_file(outputs_artifact),
            "storageDtype": "float16",
            "tensors": tensor_records(
                torch,
                {
                    name: value.detach().cpu().to(torch.float16)
                    for name, value in outputs.items()
                },
            ),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify(output, enforce_repository_pins=False)


def verify(output: Path, *, enforce_repository_pins: bool = True) -> None:
    manifest_path = output / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if enforce_repository_pins and manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise InvalidReference("manifest repository pin mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != 1 or manifest.get("story") != "sc-14539":
        raise InvalidReference("manifest identity mismatch")
    artifact = output / manifest["artifact"]["file"]
    artifact_sha256 = sha256_file(artifact)
    if enforce_repository_pins and (
        artifact.stat().st_size != EXPECTED_ARTIFACT_BYTES
        or artifact_sha256 != EXPECTED_ARTIFACT_SHA256
    ):
        raise InvalidReference("artifact repository pin mismatch")
    if (
        artifact.stat().st_size != manifest["artifact"]["bytes"]
        or artifact_sha256 != manifest["artifact"]["sha256"]
    ):
        raise InvalidReference("artifact hash/size mismatch")
    actual = inspect_safetensors(artifact)
    if actual != manifest["artifact"]["tensors"]:
        raise InvalidReference("artifact tensor inventory mismatch")
    for manifest_key, expected_bytes, expected_sha256 in (
        (
            "extendedArtifact",
            EXPECTED_EXTENDED_ARTIFACT_BYTES,
            EXPECTED_EXTENDED_ARTIFACT_SHA256,
        ),
        (
            "outputsArtifact",
            EXPECTED_OUTPUTS_ARTIFACT_BYTES,
            EXPECTED_OUTPUTS_ARTIFACT_SHA256,
        ),
    ):
        record = manifest[manifest_key]
        path = output / record["file"]
        path_sha256 = sha256_file(path)
        if enforce_repository_pins and (
            path.stat().st_size != expected_bytes
            or path_sha256 != expected_sha256
        ):
            raise InvalidReference(f"{manifest_key} repository pin mismatch")
        if path.stat().st_size != record["bytes"] or path_sha256 != record["sha256"]:
            raise InvalidReference(f"{manifest_key} hash/size mismatch")
        if inspect_safetensors(path) != record["tensors"]:
            raise InvalidReference(f"{manifest_key} tensor inventory mismatch")
    resource_evidence_sha256 = None
    if enforce_repository_pins:
        resource_evidence = output / RESOURCE_EVIDENCE
        resource_evidence_sha256 = sha256_file(resource_evidence)
        if (
            resource_evidence.stat().st_size != EXPECTED_RESOURCE_EVIDENCE_BYTES
            or resource_evidence_sha256 != EXPECTED_RESOURCE_EVIDENCE_SHA256
        ):
            raise InvalidReference("resource evidence repository pin mismatch")
        resource_record = json.loads(resource_evidence.read_text(encoding="utf-8"))
        if (
            resource_record.get("schemaVersion") != 1
            or resource_record.get("story") != "sc-14539"
            or set(resource_record.get("runs", {}))
            != {"literal380Seconds", "exactMaximum"}
        ):
            raise InvalidReference("resource evidence identity mismatch")
    print(
        json.dumps(
            {
                "status": "verified",
                "manifest": manifest_sha256,
                "artifact": artifact_sha256,
                "extendedArtifact": manifest["extendedArtifact"]["sha256"],
                "outputsArtifact": manifest["outputsArtifact"]["sha256"],
                "resourceEvidence": resource_evidence_sha256,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "docs/migration/sa3-same-l-reference",
    )
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--same-l", type=Path)
    parser.add_argument("--medium", type=Path)
    parser.add_argument("--medium-base", type=Path)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="generate only short standalone/embedded and stride-7 smoke artifacts",
    )
    args = parser.parse_args()
    try:
        if args.verify:
            verify(args.output.resolve())
        elif args.upstream and args.same_l and args.medium and args.medium_base:
            generate(args)
        else:
            raise InvalidReference(
                "generation requires --upstream, --same-l, --medium, and --medium-base"
            )
    except (InvalidReference, OSError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
