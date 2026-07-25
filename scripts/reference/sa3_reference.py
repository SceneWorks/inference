#!/usr/bin/env python3
"""Generate compact Stable Audio 3 PyTorch reference tensors without network access.

The script deliberately accepts model snapshots only through explicit environment
variables. It never resolves repository ids or derives paths from a Hugging Face
cache. Heavy imports are lazy so snapshot and artifact verification remain usable
from the repository's ordinary Python environment.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


UPSTREAM_COMMIT = "124e8a799f57a1f665495ecb72e547d0a62867f1"
UPSTREAM_REPOSITORY = "https://github.com/Stability-AI/stable-audio-3.git"
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "torch": "2.7.1",
    "torchaudio": "2.7.1",
    "transformers": "5.8.0",
}
SEED = 14534
PROMPT = "Warm analog synth pulses, crisp percussion, spacious stereo field, 112 BPM"
LATENT_LENGTH = 16
AUDIO_SAMPLES = 16_384
TIMESTEP = 0.5
SNAPSHOT_LOCK_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "migration"
    / "sa3-reference"
    / "snapshot-files.json"
)


@dataclass(frozen=True)
class SnapshotSpec:
    key: str
    repository: str
    revision: str
    env: str
    kind: str


SNAPSHOTS = (
    SnapshotSpec(
        "small-music",
        "stabilityai/stable-audio-3-small-music",
        "0fef1392cd842149a2b6d445e181c97608faac06",
        "SA3_SMALL_MUSIC_SNAPSHOT",
        "dit",
    ),
    SnapshotSpec(
        "small-sfx",
        "stabilityai/stable-audio-3-small-sfx",
        "ae12755283df9d62ca39a9b050a39a0b607b8c20",
        "SA3_SMALL_SFX_SNAPSHOT",
        "dit",
    ),
    SnapshotSpec(
        "medium",
        "stabilityai/stable-audio-3-medium",
        "27b5a21b791b1b033d193a9e1e3ce78493f102f9",
        "SA3_MEDIUM_SNAPSHOT",
        "dit",
    ),
    SnapshotSpec(
        "small-music-base",
        "stabilityai/stable-audio-3-small-music-base",
        "eab5ceee5ad9c1ed38800aff30a8e49d1161c539",
        "SA3_SMALL_MUSIC_BASE_SNAPSHOT",
        "dit",
    ),
    SnapshotSpec(
        "small-sfx-base",
        "stabilityai/stable-audio-3-small-sfx-base",
        "cc5ddb990e30daa68336ac61c140c37c7033ab7c",
        "SA3_SMALL_SFX_BASE_SNAPSHOT",
        "dit",
    ),
    SnapshotSpec(
        "medium-base",
        "stabilityai/stable-audio-3-medium-base",
        "b32993f73c3bdc3864043a72d8032606bba737c8",
        "SA3_MEDIUM_BASE_SNAPSHOT",
        "dit",
    ),
    SnapshotSpec(
        "same-s",
        "stabilityai/SAME-S",
        "fbeb3dcf53a326e5682f38e22e7f740202d44232",
        "SA3_SAME_S_SNAPSHOT",
        "same",
    ),
    SnapshotSpec(
        "same-l",
        "stabilityai/SAME-L",
        "41acf79dd242877d6499a1108ca5dba5d5eecfc5",
        "SA3_SAME_L_SNAPSHOT",
        "same",
    ),
)
SPEC_BY_KEY = {spec.key: spec for spec in SNAPSHOTS}
COMMON_FILES = (
    "model_config.json",
    "model.safetensors",
    "LICENSE.md",
    "LICENSE_GEMMA.md",
    "NOTICE",
)
T5_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
SAFETENSORS_DTYPES = {
    "BOOL": "bool",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "U8": "uint8",
    "U16": "uint16",
    "U32": "uint32",
    "U64": "uint64",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
}


def reference_inputs() -> dict[str, Any]:
    return {
        "seed": SEED,
        "prompt": PROMPT,
        "audioSamples": AUDIO_SAMPLES,
        "latentLength": LATENT_LENGTH,
        "timestep": TIMESTEP,
        "sampler": "pingpong",
        "samplerSteps": 8,
    }


class InvalidReference(RuntimeError):
    """The requested reference evidence is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_revision(path: Path) -> str:
    revision = path.resolve().name
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        marker = path / ".snapshot-revision"
        if marker.is_file():
            revision = marker.read_text(encoding="utf-8").strip()
    return revision


def required_snapshot_files(spec: SnapshotSpec) -> tuple[str, ...]:
    required = list(COMMON_FILES)
    if spec.kind == "dit":
        required.extend(f"t5gemma-b-b-ul2/{name}" for name in T5_FILES)
    return tuple(required)


def _validate_snapshot_lock(lock: dict[str, Any]) -> dict[str, Any]:
    if lock.get("schemaVersion") != 1:
        raise InvalidReference("snapshot payload lock schema must be 1")
    snapshots = lock.get("snapshots")
    if not isinstance(snapshots, dict) or set(snapshots) != set(SPEC_BY_KEY):
        raise InvalidReference("snapshot payload lock must contain exactly all eight snapshots")
    for spec in SNAPSHOTS:
        record = snapshots[spec.key]
        if record.get("repository") != spec.repository:
            raise InvalidReference(f"{spec.key} payload lock repository mismatch")
        if record.get("revision") != spec.revision:
            raise InvalidReference(f"{spec.key} payload lock revision mismatch")
        files = record.get("files")
        if not isinstance(files, dict) or set(files) != set(required_snapshot_files(spec)):
            raise InvalidReference(f"{spec.key} payload lock file inventory mismatch")
        for name, payload in files.items():
            if (
                not isinstance(payload, dict)
                or not isinstance(payload.get("bytes"), int)
                or payload["bytes"] < 0
                or not isinstance(payload.get("sha256"), str)
                or len(payload["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in payload["sha256"])
            ):
                raise InvalidReference(f"{spec.key}/{name} has an invalid payload lock")
    return lock


def load_snapshot_lock(path: Path = SNAPSHOT_LOCK_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise InvalidReference(f"missing pinned snapshot payload lock: {path}")
    return _validate_snapshot_lock(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _resolve_snapshot_paths(
    environ: dict[str, str] | None = None,
) -> dict[str, Path]:
    environ = environ or os.environ
    missing_env = [spec.env for spec in SNAPSHOTS if not environ.get(spec.env)]
    if missing_env:
        raise InvalidReference(
            "explicit snapshot environment variables are required: "
            + ", ".join(missing_env)
        )

    resolved = {}
    for spec in SNAPSHOTS:
        path = Path(environ[spec.env]).expanduser().resolve()
        if not path.is_dir():
            raise InvalidReference(f"{spec.env} is not a directory: {path}")
        revision = snapshot_revision(path)
        if revision != spec.revision:
            raise InvalidReference(
                f"{spec.key} revision mismatch: {revision!r}, expected {spec.revision}"
            )
        missing = [
            name for name in required_snapshot_files(spec) if not (path / name).is_file()
        ]
        if missing:
            raise InvalidReference(
                f"{spec.key} snapshot is incomplete; missing: {', '.join(missing)}"
            )
        resolved[spec.key] = path
    return resolved


def build_snapshot_lock(paths: dict[str, Path]) -> dict[str, Any]:
    digest_cache: dict[tuple[int, int, int], str] = {}
    snapshots = {}
    for spec in SNAPSHOTS:
        files = {}
        for name in required_snapshot_files(spec):
            path = paths[spec.key] / name
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino, stat.st_size)
            digest = digest_cache.get(identity)
            if digest is None:
                digest = sha256_file(path)
                digest_cache[identity] = digest
            files[name] = {"bytes": stat.st_size, "sha256": digest}
        snapshots[spec.key] = {
            "repository": spec.repository,
            "revision": spec.revision,
            "files": files,
        }
    return {"schemaVersion": 1, "snapshots": snapshots}


def resolve_snapshots(
    environ: dict[str, str] | None = None,
    snapshot_lock: dict[str, Any] | None = None,
) -> dict[str, Path]:
    paths = _resolve_snapshot_paths(environ)
    lock = (
        _validate_snapshot_lock(snapshot_lock)
        if snapshot_lock is not None
        else load_snapshot_lock()
    )
    for spec in SNAPSHOTS:
        for name, expected in lock["snapshots"][spec.key]["files"].items():
            path = paths[spec.key] / name
            actual_size = path.stat().st_size
            if actual_size != expected["bytes"]:
                raise InvalidReference(
                    f"{spec.key}/{name} size mismatch: {actual_size}, "
                    f"expected {expected['bytes']}"
                )
            actual_hash = sha256_file(path)
            if actual_hash != expected["sha256"]:
                raise InvalidReference(
                    f"{spec.key}/{name} SHA-256 mismatch: {actual_hash}, "
                    f"expected {expected['sha256']}"
                )
    return paths


def _config_evidence(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    diffusion = model["diffusion"]
    return {
        "modelType": config["model_type"],
        "sampleRate": config["sample_rate"],
        "sampleSize": config["sample_size"],
        "ioChannels": model.get("io_channels"),
        "diffusionObjective": diffusion.get("diffusion_objective", "v"),
        "crossAttentionCondIds": diffusion.get("cross_attention_cond_ids", []),
        "globalCondIds": diffusion.get("global_cond_ids", []),
        "inputConcatIds": diffusion.get("input_concat_ids", []),
        "localAddCondIds": diffusion.get("local_add_cond_ids", []),
        "prependCondIds": diffusion.get("prepend_cond_ids", []),
        "modularLocalCondConfigs": diffusion.get("modular_local_cond_configs", []),
        "distributionShiftOptions": diffusion.get("distribution_shift_options"),
        "samplingDistributionShiftOptions": diffusion.get(
            "sampling_distribution_shift_options"
        ),
        "maskPaddingAttention": diffusion.get("mask_padding_attention", False),
        "useEffectiveLengthForSchedule": diffusion.get(
            "use_effective_length_for_schedule", False
        ),
        "dit": diffusion["config"],
        "conditioning": model["conditioning"],
        "pretransform": model["pretransform"],
        "svdBasesPath": config.get("svd_bases_path"),
    }


def build_snapshot_records(paths: dict[str, Path]) -> list[dict[str, Any]]:
    records = []
    for spec in SNAPSHOTS:
        path = paths[spec.key]
        record: dict[str, Any] = {
            "key": spec.key,
            "repository": spec.repository,
            "revision": spec.revision,
            "pathEnvironmentVariable": spec.env,
            "kind": spec.kind,
            "modelConfigSha256": sha256_file(path / "model_config.json"),
            "modelBytes": (path / "model.safetensors").stat().st_size,
        }
        if spec.kind == "dit":
            config = json.loads(
                (path / "model_config.json").read_text(encoding="utf-8")
            )
            record["consumedConfig"] = _config_evidence(config)
            record["t5Gemma"] = {
                "subfolder": "t5gemma-b-b-ul2",
                "configSha256": sha256_file(
                    path / "t5gemma-b-b-ul2" / "config.json"
                ),
                "modelBytes": (
                    path / "t5gemma-b-b-ul2" / "model.safetensors"
                ).stat().st_size,
            }
        records.append(record)
    return records


def validate_runtime_versions(actual: dict[str, str]) -> None:
    if set(actual) != set(EXPECTED_RUNTIME):
        raise InvalidReference("runtime version inventory is incomplete")
    for package, expected in EXPECTED_RUNTIME.items():
        if actual[package] != expected:
            raise InvalidReference(
                f"{package} {expected} required, got {actual[package]}"
            )


def _torch_modules(upstream_root: Path) -> tuple[Any, Any, Any, Any, dict[str, str]]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    python_version = platform.python_version()
    if python_version != EXPECTED_RUNTIME["python"]:
        raise InvalidReference(
            f"python {EXPECTED_RUNTIME['python']} required, got {python_version}"
        )
    root = str(upstream_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import torch
        import torchaudio  # noqa: F401
        import transformers  # noqa: F401
        from safetensors.torch import save_file
    except ImportError as error:
        raise InvalidReference(
            "run generation from the pinned upstream pyproject environment"
        ) from error
    versions = {
        "python": python_version,
        "torch": importlib.metadata.version("torch"),
        "torchaudio": importlib.metadata.version("torchaudio"),
        "transformers": importlib.metadata.version("transformers"),
    }
    validate_runtime_versions(versions)
    # Import upstream model code only after the complete environment has passed.
    try:
        from stable_audio_3.inference.sampling import sample_diffusion
        from stable_audio_3.loading_utils import load_autoencoder, load_diffusion_cond
    except ImportError as error:
        raise InvalidReference("pinned upstream source is not importable") from error
    return (
        torch,
        save_file,
        sample_diffusion,
        (load_autoencoder, load_diffusion_cond),
        versions,
    )


def validate_upstream_checkout(
    upstream_root: Path, expected_commit: str = UPSTREAM_COMMIT
) -> str:
    revision = subprocess.run(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if revision != expected_commit:
        raise InvalidReference(
            f"upstream checkout mismatch: {revision}, expected {expected_commit}"
        )
    tracked_status = subprocess.run(
        [
            "git",
            "-C",
            str(upstream_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if tracked_status:
        raise InvalidReference(
            "upstream checkout has tracked modifications: "
            + tracked_status.replace("\n", "; ")
        )
    return revision


def _parse_safetensors_bytes(data: bytes) -> tuple[dict[str, Any], int]:
    if len(data) < 8:
        raise InvalidReference("truncated safetensors header")
    header_length = int.from_bytes(data[:8], "little")
    data_start = 8 + header_length
    if header_length <= 0 or data_start > len(data):
        raise InvalidReference("invalid safetensors header length")
    try:
        header = json.loads(data[8:data_start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidReference("invalid safetensors JSON header") from error
    if not isinstance(header, dict):
        raise InvalidReference("safetensors header must be an object")
    return header, data_start


def _tensor_payload_hash(
    data: bytes, header: dict[str, Any], data_start: int, name: str
) -> str:
    record = header.get(name)
    if not isinstance(record, dict):
        raise InvalidReference(f"missing safetensors tensor {name}")
    offsets = record.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(value, int) for value in offsets)
        or offsets[0] < 0
        or offsets[1] < offsets[0]
        or data_start + offsets[1] > len(data)
    ):
        raise InvalidReference(f"{name} has invalid safetensors data offsets")
    return hashlib.sha256(
        data[data_start + offsets[0] : data_start + offsets[1]]
    ).hexdigest()


def _save_tensors(
    save_file: Any,
    output: Path,
    name: str,
    tensors: dict[str, Any],
    metadata: dict[str, str],
) -> dict[str, Any]:
    path = output / f"{name}.safetensors"
    portable = {}
    tensor_records = {}
    for key, tensor in tensors.items():
        # Clone so callback views (for example step_00_x and initial_noise) do
        # not retain shared storage that safetensors correctly rejects.
        value = tensor.detach().cpu().contiguous().clone()
        portable[key] = value
        # NumPy cannot represent bfloat16. A one-tensor safetensors
        # serialization exposes the exact portable payload bytes.
        tensor_save = __import__("safetensors.torch", fromlist=["save"]).save
        raw = tensor_save({"tensor": value})
        raw_header, raw_data_start = _parse_safetensors_bytes(raw)
        tensor_records[key] = {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "sha256": _tensor_payload_hash(
                raw, raw_header, raw_data_start, "tensor"
            ),
        }
    save_file(portable, str(path), metadata=metadata)
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "tensors": tensor_records,
    }


def _fixed_audio(torch: Any, device: str) -> Any:
    samples = torch.arange(AUDIO_SAMPLES, dtype=torch.float32, device=device)
    time_axis = samples / 44_100.0
    left = (
        0.25 * torch.sin(2 * torch.pi * 220.0 * time_axis)
        + 0.08 * torch.sin(2 * torch.pi * 997.0 * time_axis)
    )
    right = (
        0.22 * torch.cos(2 * torch.pi * 330.0 * time_axis)
        + 0.06 * torch.sin(2 * torch.pi * 1499.0 * time_axis)
    )
    return torch.stack([left, right]).unsqueeze(0)


def _same_reference(
    torch: Any,
    save_file: Any,
    loaders: tuple[Any, Any],
    key: str,
    snapshot: Path,
    output: Path,
    device: str,
) -> dict[str, Any]:
    load_autoencoder, _ = loaders
    torch.manual_seed(SEED)
    autoencoder = load_autoencoder(
        str(snapshot / "model_config.json"),
        str(snapshot / "model.safetensors"),
        device=device,
    ).eval()
    captures: dict[str, Any] = {}

    def hook(prefix: str) -> Any:
        def capture(_module: Any, inputs: tuple[Any, ...], result: Any) -> None:
            captures[f"{prefix}_input"] = inputs[0].detach()
            captures[f"{prefix}_output"] = (
                result[0].detach() if isinstance(result, tuple) else result.detach()
            )

        return capture

    encoder_block = next(
        layer
        for layer in autoencoder.encoder.layers
        if layer.__class__.__name__ == "TransformerResamplingBlock"
    )
    decoder_block = next(
        layer
        for layer in autoencoder.decoder.layers
        if layer.__class__.__name__ == "TransformerResamplingBlock"
    )
    encoder_handle = encoder_block.register_forward_hook(hook("encoder_resampling_0"))
    decoder_handle = decoder_block.register_forward_hook(hook("decoder_resampling_0"))
    try:
        with torch.inference_mode():
            audio = _fixed_audio(torch, device)
            latents = autoencoder.encode(audio)
            decoded = autoencoder.decode(latents)
    finally:
        encoder_handle.remove()
        decoder_handle.remove()

    tensors = {"audio_input": audio, "latents": latents, "decoded_audio": decoded}
    tensors.update(captures)
    return _save_tensors(
        save_file,
        output,
        f"{key}-same",
        tensors,
        {
            "component": key,
            "seed": str(SEED),
            "upstreamCommit": UPSTREAM_COMMIT,
        },
    )


def _localize_t5(config: dict[str, Any], snapshot: Path) -> dict[str, Any]:
    localized = copy.deepcopy(config)
    for item in localized["model"]["conditioning"]["configs"]:
        if item["type"] == "t5gemma":
            item["config"].pop("repo_id", None)
            item["config"].pop("subfolder", None)
            item["config"]["model_path"] = str(snapshot / "t5gemma-b-b-ul2")
    return localized


def _conditioning_reference(
    torch: Any, wrapper: Any, device: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt_conditioner = wrapper.conditioner.conditioners["prompt"]
    encoded = prompt_conditioner.tokenizer(
        [PROMPT],
        truncation=True,
        max_length=prompt_conditioner.max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device).bool()
    prompt_conditioner.model.to(device).eval()
    prompt_conditioner.proj_out.to(device)
    with torch.inference_mode():
        raw = prompt_conditioner.model(
            input_ids=input_ids, attention_mask=attention_mask
        )["last_hidden_state"]
        projected = prompt_conditioner.proj_out(raw)
        padded = prompt_conditioner.apply_padding(projected, attention_mask)
        conditioning_tensors = wrapper.conditioner(
            [{"prompt": PROMPT, "seconds_total": 0.25}], device
        )
    return (
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "last_hidden_state": raw,
            "projected_padded": padded,
        },
        conditioning_tensors,
    )


def _dit_reference(
    torch: Any,
    save_file: Any,
    sample_diffusion: Any,
    loaders: tuple[Any, Any],
    key: str,
    snapshot: Path,
    output: Path,
    device: str,
) -> dict[str, Any]:
    _, load_diffusion_cond = loaders
    config = json.loads(
        (snapshot / "model_config.json").read_text(encoding="utf-8")
    )
    wrapper = load_diffusion_cond(
        _localize_t5(config, snapshot),
        str(snapshot / "model.safetensors"),
        device=device,
        model_half=False,
    )
    torch.manual_seed(SEED)
    t5_tensors, conditioning_tensors = _conditioning_reference(torch, wrapper, device)
    mask = torch.zeros((1, 1, LATENT_LENGTH), device=device)
    masked_input = torch.zeros((1, 256, LATENT_LENGTH), device=device)
    conditioning_tensors["inpaint_mask"] = [mask]
    conditioning_tensors["inpaint_masked_input"] = [masked_input]
    cond_inputs = wrapper.get_conditioning_inputs(conditioning_tensors)
    model_dtype = next(wrapper.model.parameters()).dtype
    cond_inputs = {
        name: value.to(model_dtype) if value is not None else None
        for name, value in cond_inputs.items()
    }

    torch.manual_seed(SEED)
    noise = torch.randn(
        (1, 256, LATENT_LENGTH), device=device, dtype=model_dtype
    )
    timestep = torch.full((1,), TIMESTEP, device=device, dtype=torch.float32)
    with torch.inference_mode():
        prediction = wrapper.model(
            noise,
            timestep,
            **cond_inputs,
            cfg_scale=1.0,
            batch_cfg=True,
            rescale_cfg=True,
            apg_scale=1.0,
        )

    trajectory: dict[str, Any] = {}

    def callback(state: dict[str, Any]) -> None:
        index = int(state["i"])
        trajectory[f"step_{index:02d}_x"] = state["x"].detach()
        trajectory[f"step_{index:02d}_denoised"] = state["denoised"].detach()
        trajectory[f"step_{index:02d}_sigma"] = torch.as_tensor(
            state["sigma"], device=device
        ).reshape(1)

    torch.manual_seed(SEED)
    sampler_noise = torch.randn(
        (1, 256, LATENT_LENGTH), device=device, dtype=model_dtype
    )
    with torch.inference_mode():
        final = sample_diffusion(
            model=wrapper.model,
            noise=sampler_noise,
            cond_inputs=cond_inputs,
            diffusion_objective=wrapper.diffusion_objective,
            steps=8,
            cfg_scale=1.0,
            conditioning=[{"prompt": PROMPT, "seconds_total": 0.25}],
            sample_rate=wrapper.sample_rate,
            pretransform=wrapper.pretransform,
            mask_padding_attention=True,
            use_effective_length_for_schedule=True,
            headroom_seconds=6.0,
            dist_shift=wrapper.sampling_dist_shift,
            sampler_type="pingpong",
            batch_cfg=True,
            rescale_cfg=True,
            apg_scale=1.0,
            callback=callback,
            disable_tqdm=True,
            decode=False,
        )
    trajectory["sampler_initial_noise"] = sampler_noise
    trajectory["sampler_final"] = final
    tensors = {f"t5_{name}": value for name, value in t5_tensors.items()}
    tensors.update(
        {
            "dit_noise": noise,
            "dit_timestep": timestep,
            "dit_prediction": prediction,
        }
    )
    tensors.update(trajectory)
    return _save_tensors(
        save_file,
        output,
        f"{key}-reference",
        tensors,
        {
            "component": key,
            "prompt": PROMPT,
            "seed": str(SEED),
            "sampler": "pingpong",
            "steps": "8",
            "upstreamCommit": UPSTREAM_COMMIT,
        },
    )


def _runtime_environment(
    versions: dict[str, str], device: str
) -> dict[str, str]:
    return {
        **versions,
        "platform": platform.platform(),
        "device": device,
    }


def generate(
    upstream_root: Path,
    output: Path,
    device: str,
    selected: Iterable[str],
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    validate_upstream_checkout(upstream_root)
    paths = resolve_snapshots(environ)
    selected_keys = list(selected)
    unknown = sorted(set(selected_keys) - set(SPEC_BY_KEY))
    if unknown:
        raise InvalidReference(f"unknown components: {', '.join(unknown)}")
    if len(selected_keys) != len(SPEC_BY_KEY) or set(selected_keys) != set(SPEC_BY_KEY):
        raise InvalidReference(
            "reference generation requires exactly all eight components"
        )
    output.mkdir(parents=True, exist_ok=True)
    torch, save_file, sample_diffusion, loaders, versions = _torch_modules(upstream_root)
    if device == "mps" and not torch.backends.mps.is_available():
        raise InvalidReference("MPS requested but unavailable")
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    artifacts = {}
    for key in selected_keys:
        spec = SPEC_BY_KEY[key]
        print(f"generating {key}", flush=True)
        if spec.kind == "same":
            artifact = _same_reference(
                torch, save_file, loaders, key, paths[key], output, device
            )
        else:
            artifact = _dit_reference(
                torch,
                save_file,
                sample_diffusion,
                loaders,
                key,
                paths[key],
                output,
                device,
            )
        artifacts[key] = artifact
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    manifest = {
        "schemaVersion": 1,
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
        },
        "inputs": reference_inputs(),
        "referenceEnvironment": _runtime_environment(versions, device),
        "snapshots": build_snapshot_records(paths),
        "artifacts": artifacts,
        "elapsedSeconds": round(time.monotonic() - started, 3),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify_artifacts(output)
    return manifest


def expected_tensor_keys(component: str) -> set[str]:
    if SPEC_BY_KEY[component].kind == "same":
        return {
            "audio_input",
            "latents",
            "decoded_audio",
            "encoder_resampling_0_input",
            "encoder_resampling_0_output",
            "decoder_resampling_0_input",
            "decoder_resampling_0_output",
        }
    keys = {
        "t5_input_ids",
        "t5_attention_mask",
        "t5_last_hidden_state",
        "t5_projected_padded",
        "dit_noise",
        "dit_timestep",
        "dit_prediction",
        "sampler_initial_noise",
        "sampler_final",
    }
    for index in range(8):
        keys.update(
            {
                f"step_{index:02d}_x",
                f"step_{index:02d}_denoised",
                f"step_{index:02d}_sigma",
            }
        )
    return keys


def expected_artifact_filename(component: str) -> str:
    suffix = "same" if SPEC_BY_KEY[component].kind == "same" else "reference"
    return f"{component}-{suffix}.safetensors"


def expected_artifact_metadata(component: str) -> dict[str, str]:
    metadata = {
        "component": component,
        "seed": str(SEED),
        "upstreamCommit": UPSTREAM_COMMIT,
    }
    if SPEC_BY_KEY[component].kind == "dit":
        metadata.update(
            {
                "prompt": PROMPT,
                "sampler": "pingpong",
                "steps": "8",
            }
        )
    return metadata


def _validate_safetensors_artifact(
    component: str, path: Path, record: dict[str, Any]
) -> None:
    data = path.read_bytes()
    header, data_start = _parse_safetensors_bytes(data)
    metadata = header.pop("__metadata__", None)
    if metadata != expected_artifact_metadata(component):
        raise InvalidReference(f"{component} safetensors metadata mismatch")
    expected_keys = expected_tensor_keys(component)
    if set(header) != expected_keys:
        raise InvalidReference(f"{component} safetensors tensor inventory mismatch")
    payload_hashes = {
        name: _tensor_payload_hash(data, header, data_start, name)
        for name in expected_keys
    }
    cursor = 0
    ranges = sorted(
        (header[name]["data_offsets"][0], header[name]["data_offsets"][1], name)
        for name in expected_keys
    )
    for start, end, name in ranges:
        if start != cursor:
            raise InvalidReference(
                f"{component}/{name} has a non-contiguous payload range"
            )
        cursor = end
    if data_start + cursor != len(data):
        raise InvalidReference(f"{component} has trailing or truncated tensor payload")
    tensors = record.get("tensors")
    if not isinstance(tensors, dict) or set(tensors) != expected_keys:
        raise InvalidReference(f"{component} manifest tensor inventory mismatch")
    for name in sorted(expected_keys):
        tensor_header = header[name]
        tensor_record = tensors[name]
        if not isinstance(tensor_header, dict) or not isinstance(tensor_record, dict):
            raise InvalidReference(f"{component}/{name} tensor record is invalid")
        dtype = SAFETENSORS_DTYPES.get(tensor_header.get("dtype"))
        if dtype is None or tensor_record.get("dtype") != dtype:
            raise InvalidReference(f"{component}/{name} dtype mismatch")
        if tensor_record.get("shape") != tensor_header.get("shape"):
            raise InvalidReference(f"{component}/{name} shape mismatch")
        actual_hash = payload_hashes[name]
        if tensor_record.get("sha256") != actual_hash:
            raise InvalidReference(f"{component}/{name} payload hash mismatch")


def verify_artifacts(output: Path) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise InvalidReference(f"missing artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != 1:
        raise InvalidReference("artifact manifest schema must be 1")
    if manifest.get("upstream") != {
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
    }:
        raise InvalidReference("artifact manifest has the wrong upstream source")
    if manifest.get("inputs") != reference_inputs():
        raise InvalidReference("artifact manifest reference inputs mismatch")
    reference_environment = manifest.get("referenceEnvironment")
    if not isinstance(reference_environment, dict):
        raise InvalidReference("artifact manifest reference environment is missing")
    validate_runtime_versions(
        {name: reference_environment.get(name) for name in EXPECTED_RUNTIME}
    )
    if not isinstance(reference_environment.get("device"), str) or not isinstance(
        reference_environment.get("platform"), str
    ):
        raise InvalidReference("artifact manifest platform/device metadata is missing")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(SPEC_BY_KEY):
        raise InvalidReference("artifact manifest must contain exactly all eight components")
    for key, record in artifacts.items():
        if not isinstance(record, dict):
            raise InvalidReference(f"{key} artifact record is invalid")
        if record.get("file") != expected_artifact_filename(key):
            raise InvalidReference(f"{key} artifact filename mismatch")
        path = output / record["file"]
        if not path.is_file():
            raise InvalidReference(f"{key} artifact is missing: {path.name}")
        if record.get("bytes") != path.stat().st_size:
            raise InvalidReference(f"{key} artifact size mismatch")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise InvalidReference(
                f"{key} artifact hash mismatch: {actual}, expected {record['sha256']}"
            )
        _validate_safetensors_artifact(key, path, record)
    return manifest


def _snapshot_env_help() -> str:
    return "\n".join(f"  {spec.env}=<path>  # {spec.repository}@{spec.revision}" for spec in SNAPSHOTS)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Required explicit snapshot paths:\n" + _snapshot_env_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify-snapshots")

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--upstream-root", required=True, type=Path)
    generate_parser.add_argument("--output", required=True, type=Path)
    generate_parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    generate_parser.add_argument(
        "--components",
        nargs="+",
        choices=tuple(SPEC_BY_KEY),
        default=tuple(SPEC_BY_KEY),
    )

    verify_parser = subparsers.add_parser("verify-artifacts")
    verify_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "verify-snapshots":
            paths = resolve_snapshots()
            records = build_snapshot_records(paths)
            print(json.dumps({"snapshots": records}, indent=2, sort_keys=True))
        elif args.command == "generate":
            generate(
                args.upstream_root.resolve(),
                args.output.resolve(),
                args.device,
                args.components,
            )
        else:
            verify_artifacts(args.output.resolve())
            print(f"SA3 reference artifacts: OK ({args.output.resolve()})")
    except (InvalidReference, OSError, ValueError) as error:
        print(f"SA3 reference error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
