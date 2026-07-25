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
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


UPSTREAM_COMMIT = "124e8a799f57a1f665495ecb72e547d0a62867f1"
SEED = 14534
PROMPT = "Warm analog synth pulses, crisp percussion, spacious stereo field, 112 BPM"
LATENT_LENGTH = 16
AUDIO_SAMPLES = 16_384
TIMESTEP = 0.5


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
    "model.safetensors",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


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


def resolve_snapshots(environ: dict[str, str] | None = None) -> dict[str, Path]:
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
        required = list(COMMON_FILES)
        if spec.kind == "dit":
            required.extend(f"t5gemma-b-b-ul2/{name}" for name in T5_FILES)
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise InvalidReference(
                f"{spec.key} snapshot is incomplete; missing: {', '.join(missing)}"
            )
        resolved[spec.key] = path
    return resolved


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


def _torch_modules(upstream_root: Path) -> tuple[Any, Any, Any, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    root = str(upstream_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import torch
        from safetensors.torch import save_file
        from stable_audio_3.inference.sampling import sample_diffusion
        from stable_audio_3.loading_utils import load_autoencoder, load_diffusion_cond
    except ImportError as error:
        raise InvalidReference(
            "run generation from the pinned upstream pyproject environment"
        ) from error
    return torch, save_file, sample_diffusion, (load_autoencoder, load_diffusion_cond)


def _upstream_revision(upstream_root: Path) -> str:
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


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
        # NumPy cannot represent bfloat16. Hashing a one-tensor safetensors
        # serialization covers the exact dtype, shape, and payload portably.
        tensor_save = __import__("safetensors.torch", fromlist=["save"]).save
        raw = tensor_save({"tensor": value})
        tensor_records[key] = {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
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


def _runtime_environment(torch: Any, device: str) -> dict[str, str]:
    try:
        import torchaudio
        import transformers
    except ImportError as error:
        raise InvalidReference("pinned reference packages are incomplete") from error
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
    }


def generate(
    upstream_root: Path,
    output: Path,
    device: str,
    selected: Iterable[str],
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    revision = _upstream_revision(upstream_root)
    if revision != UPSTREAM_COMMIT:
        raise InvalidReference(
            f"upstream checkout mismatch: {revision}, expected {UPSTREAM_COMMIT}"
        )
    paths = resolve_snapshots(environ)
    selected_keys = list(selected)
    unknown = sorted(set(selected_keys) - set(SPEC_BY_KEY))
    if unknown:
        raise InvalidReference(f"unknown components: {', '.join(unknown)}")
    output.mkdir(parents=True, exist_ok=True)
    torch, save_file, sample_diffusion, loaders = _torch_modules(upstream_root)
    if torch.__version__.split("+", 1)[0] != "2.7.1":
        raise InvalidReference(f"torch 2.7.1 required, got {torch.__version__}")
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
            "repository": "https://github.com/Stability-AI/stable-audio-3.git",
            "commit": UPSTREAM_COMMIT,
        },
        "inputs": {
            "seed": SEED,
            "prompt": PROMPT,
            "audioSamples": AUDIO_SAMPLES,
            "latentLength": LATENT_LENGTH,
            "timestep": TIMESTEP,
            "sampler": "pingpong",
            "samplerSteps": 8,
        },
        "referenceEnvironment": _runtime_environment(torch, device),
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


def verify_artifacts(output: Path) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise InvalidReference(f"missing artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("upstream", {}).get("commit") != UPSTREAM_COMMIT:
        raise InvalidReference("artifact manifest has the wrong upstream commit")
    for key, record in manifest.get("artifacts", {}).items():
        path = output / record["file"]
        if not path.is_file():
            raise InvalidReference(f"{key} artifact is missing: {path.name}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise InvalidReference(
                f"{key} artifact hash mismatch: {actual}, expected {record['sha256']}"
            )
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
