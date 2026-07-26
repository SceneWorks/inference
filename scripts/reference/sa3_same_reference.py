#!/usr/bin/env python3
"""Generate and verify the sc-14538 SAME-S frozen-upstream oracle.

Generation is deliberately offline: callers provide an immutable upstream checkout, the two
snapshot directories, and the already-downloaded public-domain music source. Verification needs
only the Python standard library.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import struct
import subprocess
import sys
from pathlib import Path
from unittest import mock


UPSTREAM_COMMIT = "124e8a799f57a1f665495ecb72e547d0a62867f1"
UPSTREAM_REPOSITORY = "https://github.com/Stability-AI/stable-audio-3.git"
UPSTREAM_FILE_HASHES = {
    "autoencoders.py": "44939e6b1c2a72690736757cec3f46cb7f8f6b9ccf9161b738daac320bdfa171",
    "bottleneck.py": "1d19a7713dba78de6b87fc003f74a54c40c9ac1604917c827f730363e2214c88",
    "transformer.py": "7436d5d1f040ed74af38899ca57c1eb7b6f3ee5e5c27a3e7402f39f22a52a83a",
}
SAME_S_REVISION = "fbeb3dcf53a326e5682f38e22e7f740202d44232"
SAME_S_MODEL_SHA256 = (
    "c19698ce3a0b462acb967ee495e9eb7945221f236968c50206cce8cf22b3d305"
)
EMBEDDED_REVISION = "0fef1392cd842149a2b6d445e181c97608faac06"
EMBEDDED_MODEL_SHA256 = (
    "da85866b11b01d0694d990785f6abbd79c8064df1b0e6f8aea52935e0ef84b64"
)
MUSIC_URL = (
    "https://librosa.org/data/audio/"
    "Hungarian_Dance_number_5_-_Allegro_in_F_sharp_minor_(string_orchestra).hq.ogg"
)
MUSIC_SHA256 = "8e93ff0182a93168b15346c497b164cb49d2a97bf1e987a1149ea579e914532e"
MUSIC_OFFSET = 5 * 44_100
MUSIC_SAMPLES = 10 * 44_100
SEED = 14538
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "torch": "2.7.1",
    "torchaudio": "2.7.1",
    "transformers": "5.8.0",
}
ARTIFACTS = (
    "oracle.safetensors",
    "music-roundtrip.safetensors",
    "two-stage.safetensors",
)
SYNTHETIC_CONFIG = "two-stage-config.json"
SYNTHETIC_CONFIG_SHA256 = (
    "a7bf22177267ef9aa26905c950d69008de621da3e4d44389cc03d6ad808f412a"
)
EXPECTED_ARTIFACTS = {
    "oracle.safetensors": {
        "bytes": 15_363_456,
        "sha256": "2a692ed3e93420ff82984228445e479d670d82ab66c6e7fd0914410fc9d2db54",
        "tensors": 80,
    },
    "music-roundtrip.safetensors": {
        "bytes": 12_597_016,
        "sha256": "75b37a732f4d245d0565bb99a5e675242ae508e5cac6776060490962e830fe90",
        "tensors": 5,
    },
    "two-stage.safetensors": {
        "bytes": 100_708,
        "sha256": "e851a9141cb18ef6f03f0ede30d9f78d79cc02605012cb0f6bea7dd48af7e3d1",
        "tensors": 161,
    },
}
EXPECTED_ARCHITECTURE = {
    "attentionTokens": 34,
    "subchunkTokens": 17,
    "blocks": 6,
    "midpointSplit": [3, 3],
    "midpointShiftTokens": 17,
    "ropePositions": [0, 33],
    "patchSize": 256,
    "stride": 16,
    "latentDim": 256,
    "alignmentSamples": 8192,
}
EXPECTED_QUALITY = {
    "center": True,
    "mrstft": 3.05304217338562,
    "mrstftEquation": (
        "mean_r(||Mhat-M||_F/||M||_F + "
        "mean(abs(log(Mhat+1e-7)-log(M+1e-7))))"
    ),
    "padMode": "reflect",
    "resolutions": [
        {
            "hop": 128,
            "logMagnitudeL1": 2.557659864425659,
            "spectralConvergence": 0.25792503356933594,
            "window": 512,
        },
        {
            "hop": 256,
            "logMagnitudeL1": 2.8407068252563477,
            "spectralConvergence": 0.26160508394241333,
            "window": 1024,
        },
        {
            "hop": 512,
            "logMagnitudeL1": 2.975397825241089,
            "spectralConvergence": 0.26583194732666016,
            "window": 2048,
        },
    ],
    "snrDb": 7.799531936645508,
    "snrEquation": "10*log10(sum(reference^2)/sum((reference-decoded)^2))",
    "window": "periodic Hann",
}
EXPECTED_BOUNDS = {
    "snrDbMinimum": 7.6000000000000005,
    "mrstftMaximum": 3.0549999999999997,
}
EXPECTED_BACKEND_SENSITIVITY = {
    "injection": "frozen unit-normal perturbation after decoder block 0",
    "unitSeed": 15_338,
    "scales": {
        "1e-6": {
            "afterBlock0MaxAbs": 4.537286258710083e-06,
            "cosine": 0.9725882386915404,
            "maxAbs": 2.293635368347168,
            "relativeL2": 0.234568382716466,
            "snrDb": 12.594610529587925,
        },
        "3e-6": {
            "afterBlock0MaxAbs": 1.361185968562495e-05,
            "cosine": 0.9485032091841693,
            "maxAbs": 3.6628634929656982,
            "relativeL2": 0.3313625060979783,
            "snrDb": 9.593932677293033,
        },
    },
}
SAFETENSORS_DTYPES = {
    "BOOL": ("bool", 1),
    "U8": ("uint8", 1),
    "I8": ("int8", 1),
    "I16": ("int16", 2),
    "U16": ("uint16", 2),
    "F16": ("float16", 2),
    "BF16": ("bfloat16", 2),
    "I32": ("int32", 4),
    "U32": ("uint32", 4),
    "F32": ("float32", 4),
    "F64": ("float64", 8),
    "I64": ("int64", 8),
    "U64": ("uint64", 8),
}


class InvalidReference(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise InvalidReference(f"{path.name} duplicate JSON key {key}")
            value[key] = item
        return value

    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidReference(f"invalid {path.name}: {error}") from error


def inspect_safetensors(path: Path):
    raw = path.read_bytes()
    if len(raw) < 8:
        raise InvalidReference(f"{path.name} truncated safetensors prefix")
    header_length = struct.unpack("<Q", raw[:8])[0]
    data_start = 8 + header_length
    if header_length == 0 or data_start > len(raw):
        raise InvalidReference(f"{path.name} invalid safetensors header length")
    try:
        header = json.loads(raw[8:data_start].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise InvalidReference(f"{path.name} invalid safetensors header") from error
    if not isinstance(header, dict) or header.get("__metadata__") != {
        "story": "sc-14538"
    }:
        raise InvalidReference(f"{path.name} safetensors metadata mismatch")

    records = {}
    intervals = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict) or set(entry) != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            raise InvalidReference(f"{path.name} tensor header mismatch for {name}")
        dtype = entry["dtype"]
        shape = entry["shape"]
        offsets = entry["data_offsets"]
        if (
            dtype not in SAFETENSORS_DTYPES
            or not isinstance(shape, list)
            or any(not isinstance(dim, int) or dim < 0 for dim in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(offset, int) for offset in offsets)
        ):
            raise InvalidReference(f"{path.name} tensor header mismatch for {name}")
        start, end = offsets
        elements = math.prod(shape)
        expected_bytes = elements * SAFETENSORS_DTYPES[dtype][1]
        if start < 0 or end < start or end - start != expected_bytes:
            raise InvalidReference(f"{path.name} tensor offset mismatch for {name}")
        payload_start = data_start + start
        payload_end = data_start + end
        if payload_end > len(raw):
            raise InvalidReference(f"{path.name} tensor payload truncated for {name}")
        intervals.append((start, end, name))
        records[name] = {
            "dtype": SAFETENSORS_DTYPES[dtype][0],
            "shape": shape,
            "sha256": hashlib.sha256(raw[payload_start:payload_end]).hexdigest(),
        }

    expected_start = 0
    for start, end, name in sorted(intervals):
        if start != expected_start:
            raise InvalidReference(f"{path.name} tensor layout gap before {name}")
        expected_start = end
    if data_start + expected_start != len(raw):
        raise InvalidReference(f"{path.name} trailing safetensors payload")
    return records


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
        raise InvalidReference(
            f"upstream revision mismatch: {revision}, expected {UPSTREAM_COMMIT}"
        )
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    if any(not line.endswith(" .venv/") for line in status):
        raise InvalidReference(f"upstream checkout is not clean: {status}")


def runtime(upstream: Path):
    versions = {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "torchaudio": importlib.metadata.version("torchaudio"),
        "transformers": importlib.metadata.version("transformers"),
    }
    if versions != EXPECTED_RUNTIME:
        raise InvalidReference(
            f"runtime mismatch: {versions}, expected {EXPECTED_RUNTIME}"
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(upstream))
    import torch
    import torchaudio
    from safetensors.torch import save_file
    from stable_audio_3.factory import create_autoencoder_from_config
    from stable_audio_3.loading_utils import load_autoencoder

    return (
        torch,
        torchaudio,
        save_file,
        load_autoencoder,
        create_autoencoder_from_config,
        versions,
    )


def fixed_audio(torch):
    samples = torch.arange(16_384, dtype=torch.float32)
    t = samples / 44_100.0
    return torch.stack(
        [
            0.25 * torch.sin(2 * torch.pi * 220.0 * t)
            + 0.08 * torch.sin(2 * torch.pi * 997.0 * t),
            0.22 * torch.cos(2 * torch.pi * 330.0 * t)
            + 0.06 * torch.sin(2 * torch.pi * 1499.0 * t),
        ]
    ).unsqueeze(0)


def transformer_sequence(
    raw, index: int, depth: int, batch: int, chunk: int, midpoint: bool
):
    sequence = raw.reshape(batch, -1, raw.shape[-1])
    if midpoint and index >= depth // 2:
        shift = chunk // 2
        sequence = sequence[:, shift:-shift]
    return sequence


def selected_segments(
    raw,
    depth: int,
    batch: int,
    stride: int,
    chunk: int,
    decoder: bool,
    midpoint: bool,
):
    sequence = transformer_sequence(
        raw, depth - 1, depth, batch, chunk, midpoint
    )
    subchunk = stride + 1
    output = stride if decoder else 1
    return sequence.reshape(batch, -1, subchunk, sequence.shape[-1])[
        :, :, -output:, :
    ].reshape(batch, -1, sequence.shape[-1])


def capture_block(
    torch,
    block,
    prefix: str,
    batch: int,
    captures: dict,
    *,
    decoder: bool,
    active_stride: int | None = None,
):
    handles = []
    depth = len(block.transformers)
    stride = active_stride or block.stride
    effective_chunk = block.chunk_size + block.chunk_size // stride
    midpoint = block.chunk_midpoint_shift

    def block_pre(_module, args):
        captures[f"{prefix}.mapped_sequence"] = args[0].transpose(1, 2).detach()

    handles.append(block.register_forward_pre_hook(block_pre))
    if not decoder:

        def mapping(_module, _args, output):
            captures[f"{prefix}.mapped_sequence"] = output.transpose(1, 2).detach()

        handles.append(block.mapping.register_forward_hook(mapping))
    for index, layer in enumerate(block.transformers):

        def hook(_module, args, output, index=index):
            captures[f"{prefix}.block_{index}_input"] = args[0].detach()
            if index == 0:
                captures[f"{prefix}.folded_input"] = args[0].detach()
                unfolded = args[0].reshape(batch, -1, args[0].shape[-1])
                output_segment = stride if decoder else 1
                captures[f"{prefix}.expanded_tokens"] = unfolded.reshape(
                    batch, -1, stride + 1, unfolded.shape[-1]
                )[:, :, -output_segment:, :].reshape(
                    batch, -1, unfolded.shape[-1]
                )
            captures[f"{prefix}.block_{index}"] = transformer_sequence(
                output.detach(), index, depth, batch, effective_chunk, midpoint
            )
            if index == depth - 1:
                captures[f"{prefix}.selected_segments"] = selected_segments(
                    output.detach(),
                    depth,
                    batch,
                    stride,
                    effective_chunk,
                    decoder,
                    midpoint,
                )

        handles.append(layer.register_forward_hook(hook))

    def result(_module, _args, output):
        captures[f"{prefix}.output"] = output.detach()

    handles.append(block.register_forward_hook(result))
    return handles


def run_decode_with_saved_noise(torch, autoencoder, latents, override_stride=None):
    noises = []
    original = torch.randn_like

    def record(tensor, *args, **kwargs):
        value = original(tensor, *args, **kwargs)
        noises.append(value.detach().clone())
        return value

    with mock.patch.object(torch, "randn_like", record), torch.inference_mode():
        decoded = autoencoder.decode(latents, override_stride=override_stride)
    if len(noises) != 2:
        raise InvalidReference(f"expected two decoder noises, captured {len(noises)}")
    batch = latents.shape[0]
    mask = noises[1].reshape(batch, -1, noises[1].shape[-1]) * 0.01
    return decoded, noises[0], mask


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


def save_artifact(torch, save_file, path: Path, tensors):
    portable = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in tensors.items()
    }
    save_file(portable, str(path), metadata={"story": "sc-14538"})
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "tensors": tensor_records(torch, portable),
    }


def metrics(torch, reference, decoded):
    error = reference - decoded
    snr = 10.0 * torch.log10(reference.square().sum() / error.square().sum())
    resolutions = [(512, 128), (1024, 256), (2048, 512)]
    terms = []
    details = []
    for window, hop in resolutions:
        hann = torch.hann_window(window, periodic=True)
        ref_mag = torch.stft(
            reference.reshape(-1, reference.shape[-1]),
            window,
            hop_length=hop,
            window=hann,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        ).abs()
        out_mag = torch.stft(
            decoded.reshape(-1, decoded.shape[-1]),
            window,
            hop_length=hop,
            window=hann,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        ).abs()
        convergence = torch.linalg.vector_norm(out_mag - ref_mag) / torch.linalg.vector_norm(
            ref_mag
        )
        log_magnitude = (
            (torch.log(out_mag + 1e-7) - torch.log(ref_mag + 1e-7)).abs().mean()
        )
        term = convergence + log_magnitude
        terms.append(term)
        details.append(
            {
                "window": window,
                "hop": hop,
                "spectralConvergence": float(convergence),
                "logMagnitudeL1": float(log_magnitude),
            }
        )
    return {
        "snrDb": float(snr),
        "mrstft": float(torch.stack(terms).mean()),
        "resolutions": details,
        "snrEquation": "10*log10(sum(reference^2)/sum((reference-decoded)^2))",
        "mrstftEquation": "mean_r(||Mhat-M||_F/||M||_F + mean(abs(log(Mhat+1e-7)-log(M+1e-7))))",
        "window": "periodic Hann",
        "center": True,
        "padMode": "reflect",
    }


def sensitivity_metrics(torch, reference, actual):
    reference = reference.to(torch.float64).reshape(-1)
    actual = actual.to(torch.float64).reshape(-1)
    delta = actual - reference
    relative_l2 = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference)
    return {
        "cosine": float(
            torch.dot(actual, reference)
            / (torch.linalg.vector_norm(actual) * torch.linalg.vector_norm(reference))
        ),
        "maxAbs": float(delta.abs().max()),
        "relativeL2": float(relative_l2),
        "snrDb": float(-20.0 * torch.log10(relative_l2)),
    }


def perturb_decoder_after_first_block(
    torch, decoder, model, latents, override_stride, rng_state, unit, scale
):
    def perturb(_module, _args, output):
        return output + unit.to(dtype=output.dtype, device=output.device) * scale

    handle = decoder.transformers[0].register_forward_hook(perturb)
    torch.random.set_rng_state(rng_state)
    try:
        decoded, regularization, mask = run_decode_with_saved_noise(
            torch, model, latents, override_stride=override_stride
        )
    finally:
        handle.remove()
    return decoded, regularization, mask


def two_stage_config(same_snapshot: Path):
    full = json.loads(
        (same_snapshot / "model_config.json").read_text(encoding="utf-8")
    )
    model = copy.deepcopy(full["model"])
    model["pretransform"]["config"].update({"patch_size": 2, "channels": 2})
    encoder = model["encoder"]["config"]
    encoder.update(
        {
            "in_channels": 4,
            "channels": 8,
            "latent_dim": 4,
            "c_mults": [1, 2],
            "strides": [2, 4],
            "transformer_depths": [1, 1],
            "sliding_window": None,
            "dim_heads": 8,
            "chunk_size": 8,
            "chunk_midpoint_shift": False,
            "mask_noise": 0.0,
            "conv_mapping": False,
        }
    )
    decoder = model["decoder"]["config"]
    decoder.update(
        {
            "out_channels": 4,
            "channels": 8,
            "latent_dim": 4,
            "c_mults": [1, 2],
            "strides": [2, 4],
            "transformer_depths": [1, 1],
            "sinusoidal_blocks": [0, 0],
            "sliding_window": None,
            "dim_heads": 8,
            "chunk_size": 8,
            "chunk_midpoint_shift": False,
            "mask_noise": 0.0,
            "conv_mapping": True,
        }
    )
    model["bottleneck"]["config"].update(
        {
            "dim": 4,
            "noise_augment_dim": 0,
            "noise_regularize": False,
            "auto_scale": True,
        }
    )
    model.update(
        {
            "latent_dim": 4,
            "downsampling_ratio": 16,
            "io_channels": 2,
        }
    )
    return {
        "model_type": "autoencoder",
        "sample_rate": 44_100,
        "sample_size": 32,
        "audio_channels": 2,
        "model": model,
    }


def synthetic_two_stage_reference(
    torch,
    save_file,
    create_autoencoder_from_config,
    same_snapshot: Path,
    output: Path,
):
    config = two_stage_config(same_snapshot)
    (output / SYNTHETIC_CONFIG).write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch.manual_seed(SEED + 200)
    model = create_autoencoder_from_config(
        config["model"], config["sample_rate"]
    ).eval()
    audio = (
        torch.arange(64, dtype=torch.float32).reshape(1, 2, 32) / 63.0 - 0.5
    )
    tensors = {
        f"weights.{name}": value
        for name, value in model.state_dict().items()
    }
    tensors["audio"] = audio
    encoder_blocks = [
        layer
        for layer in model.encoder.layers
        if layer.__class__.__name__ == "TransformerResamplingBlock"
    ]
    decoder_blocks = [
        layer
        for layer in model.decoder.layers
        if layer.__class__.__name__ == "TransformerResamplingBlock"
    ]
    for label, strides in (("order24", [2, 4]), ("order42", [4, 2])):
        captures = {}
        handles = []
        for stage, block in enumerate(encoder_blocks):
            handles += capture_block(
                torch,
                block,
                f"{label}.encoder{stage}",
                1,
                captures,
                decoder=False,
                active_stride=strides[stage],
            )
        for stage, block in enumerate(decoder_blocks):
            handles += capture_block(
                torch,
                block,
                f"{label}.decoder{stage}",
                1,
                captures,
                decoder=True,
                active_stride=strides[stage],
            )
        with torch.inference_mode():
            latents = model.encode(audio, override_stride=strides)
            decoded = model.decode(latents, override_stride=strides)
        for handle in handles:
            handle.remove()
        tensors[f"{label}.latents"] = latents
        tensors[f"{label}.decoded"] = decoded
        tensors.update(captures)
    return save_artifact(
        torch, save_file, output / ARTIFACTS[2], tensors
    )


def generate(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    upstream = args.upstream.resolve()
    same = args.same_s.resolve()
    embedded = args.embedded.resolve()
    music = args.music.resolve()
    validate_upstream(upstream)
    require_revision(same, SAME_S_REVISION, "SAME-S")
    require_revision(embedded, EMBEDDED_REVISION, "embedded small")
    if sha256_file(music) != MUSIC_SHA256:
        raise InvalidReference("music source hash mismatch")
    (
        torch,
        torchaudio,
        save_file,
        load_autoencoder,
        create_autoencoder_from_config,
        versions,
    ) = runtime(upstream)

    torch.manual_seed(SEED)
    model = load_autoencoder(
        str(same / "model_config.json"), str(same / "model.safetensors"), device="cpu"
    ).eval()
    audio = fixed_audio(torch)
    captures = {}
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
    handles = capture_block(
        torch, encoder, "encoder", 1, captures, decoder=False
    )
    handles += capture_block(
        torch, decoder, "decoder", 1, captures, decoder=True
    )
    with torch.inference_mode():
        latents = model.encode(audio)
    decoded, regularization_noise, mask_noise = run_decode_with_saved_noise(
        torch, model, latents
    )
    for handle in handles:
        handle.remove()
    oracle_tensors = {
        "audio": audio,
        "latents": latents,
        "regularization_noise": regularization_noise,
        "decoder_mask_noise": mask_noise,
        "decoded": decoded,
        **captures,
    }

    stride_captures = {}
    stride_handles = capture_block(
        torch,
        encoder,
        "stride8.encoder",
        1,
        stride_captures,
        decoder=False,
        active_stride=8,
    )
    stride_handles += capture_block(
        torch,
        decoder,
        "stride8.decoder",
        1,
        stride_captures,
        decoder=True,
        active_stride=8,
    )
    torch.manual_seed(SEED + 8)
    with torch.inference_mode():
        stride_latents = model.encode(audio, override_stride=[8])
    stride_decode_rng_state = torch.random.get_rng_state()
    (
        stride_decoded,
        stride_regularization_noise,
        stride_mask_noise,
    ) = run_decode_with_saved_noise(
        torch, model, stride_latents, override_stride=[8]
    )
    for handle in stride_handles:
        handle.remove()
    perturbation_generator = torch.Generator(device="cpu").manual_seed(SEED + 800)
    perturbation_unit = torch.randn(
        (2, 36, 768), generator=perturbation_generator, dtype=torch.float32
    )
    stride_perturbed = {}
    sensitivity = {}
    for label, scale in (("1e-6", 1e-6), ("3e-6", 3e-6)):
        perturbed, perturb_regularization, perturb_mask = (
            perturb_decoder_after_first_block(
                torch,
                decoder,
                model,
                stride_latents,
                [8],
                stride_decode_rng_state,
                perturbation_unit,
                scale,
            )
        )
        if not torch.equal(perturb_regularization, stride_regularization_noise):
            raise InvalidReference("stride sensitivity changed SoftNorm noise")
        if not torch.equal(perturb_mask, stride_mask_noise):
            raise InvalidReference("stride sensitivity changed decoder token noise")
        stride_perturbed[f"stride8.perturbation_{label}.decoded"] = perturbed
        sensitivity[label] = {
            "afterBlock0MaxAbs": float(perturbation_unit.abs().max() * scale),
            **sensitivity_metrics(torch, stride_decoded, perturbed),
        }
    oracle_tensors.update(
        {
            "stride8.latents": stride_latents,
            "stride8.regularization_noise": stride_regularization_noise,
            "stride8.decoder_mask_noise": stride_mask_noise,
            "stride8.decoded": stride_decoded,
            "stride8.perturbation_unit": perturbation_unit,
            **stride_perturbed,
            **stride_captures,
        }
    )
    oracle = save_artifact(torch, save_file, output / ARTIFACTS[0], oracle_tensors)

    source_audio, sample_rate = torchaudio.load(str(music))
    if sample_rate != 44_100 or source_audio.shape[0] != 2:
        raise InvalidReference(
            f"music source must be stereo 44.1kHz, got {source_audio.shape}, {sample_rate}"
        )
    clip = source_audio[:, MUSIC_OFFSET : MUSIC_OFFSET + MUSIC_SAMPLES].unsqueeze(0)
    if clip.shape[-1] != MUSIC_SAMPLES:
        raise InvalidReference("music source is too short")
    torch.manual_seed(SEED + 1)
    with torch.inference_mode():
        music_latents = model.encode(clip)
    music_decoded, music_regularization, music_mask = run_decode_with_saved_noise(
        torch, model, music_latents
    )
    cropped = music_decoded[:, :, :MUSIC_SAMPLES]
    quality = metrics(torch, clip, cropped)
    music_artifact = save_artifact(
        torch,
        save_file,
        output / ARTIFACTS[1],
        {
            "audio": clip,
            "latents": music_latents,
            "regularization_noise": music_regularization,
            "decoder_mask_noise": music_mask,
            "decoded_padded": music_decoded,
        },
    )
    two_stage_artifact = synthetic_two_stage_reference(
        torch,
        save_file,
        create_autoencoder_from_config,
        same,
        output,
    )

    snapshot_lock = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "docs/migration/sa3-reference/snapshot-files.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    manifest = {
        "schemaVersion": 1,
        "story": "sc-14538",
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "files": {
                name: sha256_file(upstream / "stable_audio_3/models" / name)
                for name in ("autoencoders.py", "bottleneck.py", "transformer.py")
            },
        },
        "runtime": versions,
        "environment": {
            "system": platform.system(),
            "machine": platform.machine(),
            "torchDevice": "cpu",
        },
        "seed": SEED,
        "snapshots": {
            "standalone": {
                "revision": SAME_S_REVISION,
                "modelSha256": snapshot_lock["snapshots"]["same-s"]["files"][
                    "model.safetensors"
                ]["sha256"],
                "prefix": "",
            },
            "embedded": {
                "revision": EMBEDDED_REVISION,
                "modelSha256": snapshot_lock["snapshots"]["small-music"]["files"][
                    "model.safetensors"
                ]["sha256"],
                "prefix": "pretransform.model.",
            },
        },
        "architecture": EXPECTED_ARCHITECTURE,
        "syntheticTwoStage": {
            "config": SYNTHETIC_CONFIG,
            "configSha256": sha256_file(output / SYNTHETIC_CONFIG),
            "overrideOrders": [[2, 4], [4, 2]],
        },
        "backendSensitivity": {
            "injection": "frozen unit-normal perturbation after decoder block 0",
            "unitSeed": SEED + 800,
            "scales": sensitivity,
        },
        "music": {
            "sourceUrl": MUSIC_URL,
            "sourceSha256": MUSIC_SHA256,
            "license": "Public Domain Mark 1.0",
            "attribution": "Brahms Hungarian Dance No. 5, US Army Strings, via librosa",
            "sampleRate": 44100,
            "channels": 2,
            "offsetSamples": MUSIC_OFFSET,
            "samples": MUSIC_SAMPLES,
            "quality": quality,
            "bounds": {
                "snrDbMinimum": math.floor(quality["snrDb"] * 10.0) / 10.0 - 0.1,
                "mrstftMaximum": math.ceil(quality["mrstft"] * 1000.0) / 1000.0
                + 0.001,
            },
        },
        "artifacts": [oracle, music_artifact, two_stage_artifact],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify(output)


def verify(output: Path):
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise InvalidReference("missing manifest.json")
    manifest = load_json(manifest_path)
    if set(manifest) != {
        "schemaVersion",
        "story",
        "upstream",
        "runtime",
        "environment",
        "seed",
        "snapshots",
        "architecture",
        "syntheticTwoStage",
        "backendSensitivity",
        "music",
        "artifacts",
    }:
        raise InvalidReference("manifest top-level inventory mismatch")
    if manifest.get("schemaVersion") != 1 or manifest.get("story") != "sc-14538":
        raise InvalidReference("manifest identity mismatch")
    if manifest.get("upstream") != {
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "files": UPSTREAM_FILE_HASHES,
    }:
        raise InvalidReference("upstream provenance mismatch")
    if manifest.get("runtime") != EXPECTED_RUNTIME:
        raise InvalidReference("runtime provenance mismatch")
    if manifest.get("environment") != {
        "system": "Darwin",
        "machine": "arm64",
        "torchDevice": "cpu",
    } or manifest.get("seed") != SEED:
        raise InvalidReference("generation environment mismatch")
    if manifest.get("snapshots") != {
        "standalone": {
            "revision": SAME_S_REVISION,
            "modelSha256": SAME_S_MODEL_SHA256,
            "prefix": "",
        },
        "embedded": {
            "revision": EMBEDDED_REVISION,
            "modelSha256": EMBEDDED_MODEL_SHA256,
            "prefix": "pretransform.model.",
        },
    }:
        raise InvalidReference("snapshot provenance mismatch")
    if manifest.get("architecture") != EXPECTED_ARCHITECTURE:
        raise InvalidReference("architecture provenance mismatch")
    if manifest.get("syntheticTwoStage") != {
        "config": SYNTHETIC_CONFIG,
        "configSha256": SYNTHETIC_CONFIG_SHA256,
        "overrideOrders": [[2, 4], [4, 2]],
    }:
        raise InvalidReference("synthetic provenance mismatch")
    if manifest.get("backendSensitivity") != EXPECTED_BACKEND_SENSITIVITY:
        raise InvalidReference("backend sensitivity provenance mismatch")
    synthetic_config = output / SYNTHETIC_CONFIG
    if not synthetic_config.is_file():
        raise InvalidReference(f"missing {SYNTHETIC_CONFIG}")
    if sha256_file(synthetic_config) != SYNTHETIC_CONFIG_SHA256:
        raise InvalidReference("synthetic config hash mismatch")
    if manifest.get("music") != {
        "sourceUrl": MUSIC_URL,
        "sourceSha256": MUSIC_SHA256,
        "license": "Public Domain Mark 1.0",
        "attribution": "Brahms Hungarian Dance No. 5, US Army Strings, via librosa",
        "sampleRate": 44_100,
        "channels": 2,
        "offsetSamples": MUSIC_OFFSET,
        "samples": MUSIC_SAMPLES,
        "quality": EXPECTED_QUALITY,
        "bounds": EXPECTED_BOUNDS,
    }:
        raise InvalidReference("music provenance mismatch")
    quality = manifest["music"]["quality"]
    bounds = manifest["music"]["bounds"]
    if (
        quality["snrDb"] < bounds["snrDbMinimum"]
        or quality["mrstft"] > bounds["mrstftMaximum"]
    ):
        raise InvalidReference("music metric bound mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or [a.get("file") for a in artifacts] != list(
        ARTIFACTS
    ):
        raise InvalidReference("artifact inventory mismatch")
    for record in artifacts:
        if not isinstance(record, dict) or set(record) != {
            "file",
            "bytes",
            "sha256",
            "tensors",
        }:
            raise InvalidReference("artifact record mismatch")
        expected = EXPECTED_ARTIFACTS[record["file"]]
        if (
            record["bytes"] != expected["bytes"]
            or record["sha256"] != expected["sha256"]
        ):
            raise InvalidReference(f"{record['file']} frozen record mismatch")
        if (
            not isinstance(record["tensors"], dict)
            or len(record["tensors"]) != expected["tensors"]
        ):
            raise InvalidReference(f"{record['file']} tensor inventory mismatch")
        path = output / record["file"]
        if not path.is_file():
            raise InvalidReference(f"missing artifact {path.name}")
        if path.stat().st_size != expected["bytes"]:
            raise InvalidReference(f"{path.name} size mismatch")
        actual_tensors = inspect_safetensors(path)
        if set(record["tensors"]) != set(actual_tensors):
            raise InvalidReference(f"{path.name} tensor inventory mismatch")
        for name, expected_tensor in record["tensors"].items():
            if (
                not isinstance(expected_tensor, dict)
                or set(expected_tensor) != {"dtype", "shape", "sha256"}
                or expected_tensor != actual_tensors[name]
            ):
                raise InvalidReference(f"{path.name} tensor record mismatch for {name}")
        if sha256_file(path) != expected["sha256"]:
            raise InvalidReference(f"{path.name} hash mismatch")
    print(
        json.dumps(
            {
                "status": "verified",
                "manifest": sha256_file(manifest_path),
                "artifacts": {name: sha256_file(output / name) for name in ARTIFACTS},
            },
            sort_keys=True,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "docs/migration/sa3-same-s-reference",
    )
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--same-s", type=Path)
    parser.add_argument("--embedded", type=Path)
    parser.add_argument("--music", type=Path)
    args = parser.parse_args()
    try:
        if args.verify:
            verify(args.output.resolve())
        else:
            missing = [
                name
                for name in ("upstream", "same_s", "embedded", "music")
                if getattr(args, name) is None
            ]
            if missing:
                raise InvalidReference(
                    "generation requires " + ", ".join("--" + m.replace("_", "-") for m in missing)
                )
            generate(args)
    except (InvalidReference, OSError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
