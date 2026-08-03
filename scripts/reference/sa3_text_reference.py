#!/usr/bin/env python3
"""Generate or verify the frozen T5Gemma-b-b-ul2 Stable Audio 3 text oracle.

Generation is deliberately offline and accepts both the exact Stable Audio 3
source checkout and the exact small-music snapshot as explicit paths. Heavy
imports are lazy so ordinary repository tests can verify provenance and hashes
without installing PyTorch or Transformers.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

try:
    from sa3_reference import sha256_file, validate_upstream_checkout
except ModuleNotFoundError:
    from scripts.reference.sa3_reference import sha256_file, validate_upstream_checkout


UPSTREAM_REPOSITORY = "https://github.com/Stability-AI/stable-audio-3.git"
UPSTREAM_COMMIT = "124e8a799f57a1f665495ecb72e547d0a62867f1"
UPSTREAM_UV_LOCK_SHA256 = (
    "f6422550a634b3e44664235910835bde49ec930b8026ec34f28e4ddc2e724150"
)
SNAPSHOT_REPOSITORY = "stabilityai/stable-audio-3-small-music"
SNAPSHOT_REVISION = "0fef1392cd842149a2b6d445e181c97608faac06"
TEXT_MODEL_BYTES = 1_183_022_944
TEXT_MODEL_SHA256 = (
    "9b05ea5a4f211d023832f706fb2c0e83e4fc721b6da35ab69ceb0b55eb7800d3"
)
TEXT_CONFIG_SHA256 = (
    "575334409716886ac2952f5a275ed92868deef8a0ea560258d9970a431c6fb3a"
)
TOKENIZER_SHA256 = (
    "7794135caa3ea73918949c902a781cc61dab674a4b59c17d85931c77c1114cbd"
)
TEXT_CONFIG_BYTES = 2_540
TOKENIZER_BYTES = 34_362_429
ORACLE_BYTES = 3_549_352
ORACLE_SHA256 = (
    "51b37c402de952ea19022baf491755bd3b1f3aa06ac4f1d380dae6a7f46fdaf4"
)
CPU_ORACLE_BYTES = 4_729_000
CPU_ORACLE_SHA256 = (
    "beb5e15e40c89da227d430da2d65ea47b8fba7339af1570e0af184c814410eb8"
)
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "torch": "2.7.1",
    "transformers": "5.8.0",
}
MAX_LENGTH = 256
LONG_PROMPT_HEAD_WORDS = (
    "amber",
    "violin",
    "ocean",
    "thunder",
    "velvet",
    "rhythm",
    "copper",
    "forest",
    "lantern",
    "piano",
    "silver",
    "rain",
    "echo",
    "drum",
    "meadow",
    "sunrise",
    "crystal",
    "bass",
    "canyon",
    "pulse",
    "maple",
    "choir",
    "comet",
    "guitar",
    "harbor",
    "tempo",
    "winter",
    "flute",
    "aurora",
    "melody",
    "stone",
    "wave",
)
LONG_PROMPT_TAIL_WORDS = ("xylophone", "kumquat", "zucchini", "zephyr", "quasar")
LONG_PROMPT_HEAD = " ".join(LONG_PROMPT_HEAD_WORDS * 12)
LONG_PROMPT = f"{LONG_PROMPT_HEAD} {' '.join(LONG_PROMPT_TAIL_WORDS)}"
EXPECTED_RAW_TOKEN_COUNTS = (3, 17, 394)
EXPECTED_TRUNCATION_PROBE = {
    "retainedBoundaryTokenIds": [46107, 11030, 7830, 74458, 94510, 53188, 9566, 9591],
    "discardedTailStartsAt": 384,
    "discardedTailTokenIds": [
        54824,
        545,
        3953,
        52524,
        90083,
        105165,
        4949,
        219990,
        700,
        62026,
    ],
}
PROMPTS = (
    ("short", "A bell."),
    (
        "medium",
        "Warm analog synth pulses, crisp percussion, spacious stereo field, 112 BPM",
    ),
    ("truncated", LONG_PROMPT),
)
TENSOR_DTYPES = {
    "input_ids": "I64",
    "attention_mask": "U8",
    "raw_embeddings": "BF16",
    "conditioned_embeddings": "F32",
    "padding_embedding": "F32",
}
TENSOR_SHAPES = {
    "input_ids": [3, 256],
    "attention_mask": [3, 256],
    "raw_embeddings": [3, 256, 768],
    "conditioned_embeddings": [3, 256, 768],
    "padding_embedding": [768],
}
CPU_TENSOR_DTYPES = {**TENSOR_DTYPES, "raw_embeddings": "F32"}


class InvalidReference(RuntimeError):
    """Reference provenance or payload did not match the frozen contract."""


def verify_sources(upstream: Path, snapshot: Path) -> dict[str, Any]:
    upstream = upstream.resolve()
    snapshot = snapshot.resolve()
    validate_upstream_checkout(
        upstream,
        UPSTREAM_COMMIT,
        error_type=InvalidReference,
        allow_venv=False,
        include_ignored=False,
    )
    uv_lock = upstream / "uv.lock"
    if sha256_file(uv_lock) != UPSTREAM_UV_LOCK_SHA256:
        raise InvalidReference("Stable Audio 3 uv.lock hash mismatch")
    if snapshot.name != SNAPSHOT_REVISION:
        raise InvalidReference("small-music snapshot revision mismatch")
    text_root = snapshot / "t5gemma-b-b-ul2"
    files = {
        "model.safetensors": (TEXT_MODEL_BYTES, TEXT_MODEL_SHA256),
        "config.json": (None, TEXT_CONFIG_SHA256),
        "tokenizer.json": (None, TOKENIZER_SHA256),
    }
    payloads = {}
    for name, (expected_bytes, expected_sha) in files.items():
        path = text_root / name
        if not path.is_file():
            raise InvalidReference(f"missing text artifact: {path}")
        actual_bytes = path.stat().st_size
        if expected_bytes is not None and actual_bytes != expected_bytes:
            raise InvalidReference(f"{name} byte length mismatch")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise InvalidReference(f"{name} sha256 mismatch")
        payloads[name] = {"bytes": actual_bytes, "sha256": actual_sha}
    return {
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "uvLockSha256": UPSTREAM_UV_LOCK_SHA256,
        },
        "snapshot": {
            "repository": SNAPSHOT_REPOSITORY,
            "revision": SNAPSHOT_REVISION,
            "textFiles": payloads,
        },
    }


def verify_runtime() -> dict[str, str]:
    actual = {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
    }
    if actual != EXPECTED_RUNTIME:
        raise InvalidReference(f"runtime mismatch: {actual!r}")
    return actual


def generate(upstream: Path, snapshot: Path, output_dir: Path) -> None:
    provenance = verify_sources(upstream, snapshot)
    runtime = verify_runtime()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoTokenizer, T5GemmaEncoderModel

    text_root = snapshot.resolve() / "t5gemma-b-b-ul2"
    if not torch.backends.mps.is_available():
        raise InvalidReference("the Metal acceptance oracle requires PyTorch MPS")
    tokenizer = AutoTokenizer.from_pretrained(text_root, local_files_only=True)
    if tokenizer.padding_side != "right" or tokenizer.truncation_side != "right":
        raise InvalidReference("tokenizer is not right-padding/right-truncating")
    tokenizer_json = json.loads(
        (text_root / "tokenizer.json").read_text(encoding="utf-8")
    )
    post_processor = tokenizer_json.get("post_processor")
    if (
        not isinstance(post_processor, dict)
        or post_processor.get("type") != "TemplateProcessing"
        or post_processor.get("special_tokens") != {}
        or post_processor.get("single")
        != [{"Sequence": {"id": "A", "type_id": 0}}]
    ):
        raise InvalidReference("tokenizer.json post-processor is not identity-only")

    raw_lengths = [
        len(tokenizer(text, add_special_tokens=True)["input_ids"])
        for _, text in PROMPTS
    ]
    long_ids = tokenizer(LONG_PROMPT, add_special_tokens=True)["input_ids"]
    long_head_ids = tokenizer(LONG_PROMPT_HEAD, add_special_tokens=True)["input_ids"]
    truncation_probe = {
        "retainedBoundaryTokenIds": long_ids[MAX_LENGTH - 8 : MAX_LENGTH],
        "discardedTailStartsAt": len(long_head_ids),
        "discardedTailTokenIds": long_ids[len(long_head_ids) :],
    }
    if (
        tuple(raw_lengths) != EXPECTED_RAW_TOKEN_COUNTS
        or truncation_probe != EXPECTED_TRUNCATION_PROBE
        or not set(truncation_probe["retainedBoundaryTokenIds"]).isdisjoint(
            truncation_probe["discardedTailTokenIds"]
        )
    ):
        raise InvalidReference(
            f"heterogeneous right-truncation probe drifted: {raw_lengths}, "
            f"{truncation_probe}"
        )
    if not (raw_lengths[0] < raw_lengths[1] < MAX_LENGTH < raw_lengths[2]):
        raise InvalidReference(f"prompt coverage is not short/medium/truncated: {raw_lengths}")
    encoded_cpu = tokenizer(
        [text for _, text in PROMPTS],
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    config = AutoConfig.from_pretrained(text_root, local_files_only=True)
    config.is_encoder_decoder = False
    model = T5GemmaEncoderModel.from_pretrained(
        text_root,
        config=config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().to("mps")
    if model.config.encoder._attn_implementation != "eager":
        raise InvalidReference("T5Gemma encoder did not select eager attention")
    encoded = {key: value.to("mps") for key, value in encoded_cpu.items()}
    with torch.inference_mode():
        raw = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            return_dict=True,
        ).last_hidden_state

    with safe_open(
        snapshot.resolve() / "model.safetensors", framework="pt", device="cpu"
    ) as root_weights:
        padding = root_weights.get_tensor(
            "conditioner.conditioners.prompt.padding_embedding"
        )
    mask = encoded["attention_mask"].bool().unsqueeze(-1)
    conditioned = torch.where(
        mask, raw, padding.to("mps").reshape(1, 1, -1)
    )
    if raw.dtype != torch.bfloat16 or conditioned.dtype != torch.float32:
        raise InvalidReference(
            f"unexpected dtype order: raw={raw.dtype}, conditioned={conditioned.dtype}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "text.safetensors"
    save_file(
        {
            "input_ids": encoded["input_ids"].cpu().contiguous(),
            "attention_mask": encoded["attention_mask"]
            .to(torch.uint8)
            .cpu()
            .contiguous(),
            "raw_embeddings": raw.cpu().contiguous(),
            "conditioned_embeddings": conditioned.cpu().contiguous(),
            "padding_embedding": padding.cpu().contiguous(),
        },
        tensor_path,
    )
    del model
    cpu_model = T5GemmaEncoderModel.from_pretrained(
        text_root,
        config=config,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    with torch.inference_mode():
        cpu_raw = cpu_model(
            input_ids=encoded_cpu["input_ids"],
            attention_mask=encoded_cpu["attention_mask"],
            return_dict=True,
        ).last_hidden_state
    cpu_conditioned = torch.where(
        encoded_cpu["attention_mask"].bool().unsqueeze(-1),
        cpu_raw,
        padding.reshape(1, 1, -1),
    )
    cpu_tensor_path = output_dir / "text-cpu-f32.safetensors"
    save_file(
        {
            "input_ids": encoded_cpu["input_ids"].contiguous(),
            "attention_mask": encoded_cpu["attention_mask"]
            .to(torch.uint8)
            .contiguous(),
            "raw_embeddings": cpu_raw.contiguous(),
            "conditioned_embeddings": cpu_conditioned.contiguous(),
            "padding_embedding": padding.cpu().contiguous(),
        },
        cpu_tensor_path,
    )
    manifest = {
        "schemaVersion": 1,
        "story": "sc-14537",
        **provenance,
        "runtime": runtime,
        "model": {
            "attentionImplementation": "eager",
            "attentionSoftcap": 50.0,
            "attentionSoftmaxDtype": "float32",
            "canonicalReference": "cpu-f32",
            "mpsBf16Role": "backend-variance-evidence",
        },
        "tokenizer": {
            "library": "tokenizers 0.22 via Transformers 5.8.0",
            "paddingSide": "right",
            "truncationSide": "right",
            "maxLength": MAX_LENGTH,
            "padTokenId": tokenizer.pad_token_id,
            "postProcessor": "identity-no-special-tokens",
        },
        "prompts": [
            {
                "key": key,
                "text": text,
                "rawTokenCount": raw_length,
                "validTokenCount": min(raw_length, MAX_LENGTH),
                "truncated": raw_length > MAX_LENGTH,
            }
            for (key, text), raw_length in zip(PROMPTS, raw_lengths)
        ],
        "truncationProbe": truncation_probe,
        "tensorArtifact": {
            "file": tensor_path.name,
            "bytes": tensor_path.stat().st_size,
            "sha256": sha256_file(tensor_path),
            "dtypes": TENSOR_DTYPES,
            "shapes": TENSOR_SHAPES,
            "role": "backend-variance-evidence",
        },
        "cpuTensorArtifact": {
            "file": cpu_tensor_path.name,
            "bytes": cpu_tensor_path.stat().st_size,
            "sha256": sha256_file(cpu_tensor_path),
            "dtypes": CPU_TENSOR_DTYPES,
            "shapes": TENSOR_SHAPES,
            "computePolicy": "BF16 on disk, F32 CPU compute and raw output",
            "role": "canonical",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify(output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != 1 or manifest.get("story") != "sc-14537":
        raise InvalidReference("manifest identity mismatch")
    if manifest.get("runtime") != EXPECTED_RUNTIME:
        raise InvalidReference("manifest runtime mismatch")
    if manifest.get("model") != {
        "attentionImplementation": "eager",
        "attentionSoftcap": 50.0,
        "attentionSoftmaxDtype": "float32",
        "canonicalReference": "cpu-f32",
        "mpsBf16Role": "backend-variance-evidence",
    }:
        raise InvalidReference("manifest model execution contract mismatch")
    if manifest.get("upstream") != {
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "uvLockSha256": UPSTREAM_UV_LOCK_SHA256,
    }:
        raise InvalidReference("manifest upstream provenance mismatch")
    if manifest.get("snapshot") != {
        "repository": SNAPSHOT_REPOSITORY,
        "revision": SNAPSHOT_REVISION,
        "textFiles": {
            "model.safetensors": {
                "bytes": TEXT_MODEL_BYTES,
                "sha256": TEXT_MODEL_SHA256,
            },
            "config.json": {
                "bytes": TEXT_CONFIG_BYTES,
                "sha256": TEXT_CONFIG_SHA256,
            },
            "tokenizer.json": {
                "bytes": TOKENIZER_BYTES,
                "sha256": TOKENIZER_SHA256,
            },
        },
    }:
        raise InvalidReference("manifest snapshot provenance mismatch")
    expected_prompts = [
        {
            "key": key,
            "text": text,
            "rawTokenCount": raw_count,
            "validTokenCount": min(raw_count, MAX_LENGTH),
            "truncated": raw_count > MAX_LENGTH,
        }
        for (key, text), raw_count in zip(PROMPTS, EXPECTED_RAW_TOKEN_COUNTS)
    ]
    if manifest.get("prompts") != expected_prompts:
        raise InvalidReference("manifest prompt metadata mismatch")
    if manifest.get("truncationProbe") != EXPECTED_TRUNCATION_PROBE:
        raise InvalidReference("manifest truncation probe mismatch")
    tokenizer = manifest.get("tokenizer", {})
    if tokenizer != {
        "library": "tokenizers 0.22 via Transformers 5.8.0",
        "paddingSide": "right",
        "truncationSide": "right",
        "maxLength": MAX_LENGTH,
        "padTokenId": 0,
        "postProcessor": "identity-no-special-tokens",
    }:
        raise InvalidReference("manifest tokenizer contract mismatch")
    artifact = manifest.get("tensorArtifact", {})
    if artifact != {
        "file": "text.safetensors",
        "bytes": ORACLE_BYTES,
        "sha256": ORACLE_SHA256,
        "dtypes": TENSOR_DTYPES,
        "shapes": TENSOR_SHAPES,
        "role": "backend-variance-evidence",
    }:
        raise InvalidReference("manifest tensor artifact contract mismatch")
    path = output_dir / "text.safetensors"
    if (
        not path.is_file()
        or path.stat().st_size != ORACLE_BYTES
        or sha256_file(path) != ORACLE_SHA256
    ):
        raise InvalidReference("tensor artifact identity mismatch")
    with path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_length))
    header.pop("__metadata__", None)
    if set(header) != set(TENSOR_DTYPES):
        raise InvalidReference("tensor artifact inventory mismatch")
    for key, shape in TENSOR_SHAPES.items():
        if header[key].get("shape") != shape:
            raise InvalidReference(f"{key} shape mismatch")
        if header[key].get("dtype") != TENSOR_DTYPES[key]:
            raise InvalidReference(f"{key} dtype mismatch")
    cpu_artifact = manifest.get("cpuTensorArtifact", {})
    if cpu_artifact != {
        "file": "text-cpu-f32.safetensors",
        "bytes": CPU_ORACLE_BYTES,
        "sha256": CPU_ORACLE_SHA256,
        "dtypes": CPU_TENSOR_DTYPES,
        "shapes": TENSOR_SHAPES,
        "computePolicy": "BF16 on disk, F32 CPU compute and raw output",
        "role": "canonical",
    }:
        raise InvalidReference("manifest CPU tensor artifact contract mismatch")
    cpu_path = output_dir / "text-cpu-f32.safetensors"
    if (
        not cpu_path.is_file()
        or cpu_path.stat().st_size != CPU_ORACLE_BYTES
        or sha256_file(cpu_path) != CPU_ORACLE_SHA256
    ):
        raise InvalidReference("CPU tensor artifact identity mismatch")
    with cpu_path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        cpu_header = json.loads(handle.read(header_length))
    cpu_header.pop("__metadata__", None)
    if set(cpu_header) != set(CPU_TENSOR_DTYPES):
        raise InvalidReference("CPU tensor artifact inventory mismatch")
    for key, shape in TENSOR_SHAPES.items():
        if cpu_header[key].get("shape") != shape:
            raise InvalidReference(f"CPU {key} shape mismatch")
        if cpu_header[key].get("dtype") != CPU_TENSOR_DTYPES[key]:
            raise InvalidReference(f"CPU {key} dtype mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--upstream", required=True, type=Path)
    generate_parser.add_argument("--snapshot", required=True, type=Path)
    generate_parser.add_argument("--output-dir", required=True, type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            generate(args.upstream, args.snapshot, args.output_dir)
        else:
            verify(args.output_dir)
    except (InvalidReference, OSError, ValueError, KeyError) as error:
        print(f"SA3 text reference error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
