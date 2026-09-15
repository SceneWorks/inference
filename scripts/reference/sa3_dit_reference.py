#!/usr/bin/env python3
"""Generate/verify the compact frozen-upstream sc-14541 DiT oracle.

Generation is offline and accepts only explicit paths. Verification has no Torch
dependency and locks source, snapshot, P0 input, tensor inventory, and payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from sa3_reference import (
    InvalidReference,
    SPEC_BY_KEY,
    UPSTREAM_COMMIT,
    _localize_t5,
    _torch_modules,
    load_snapshot_lock,
    sha256_file,
    snapshot_revision,
)


ROOT = Path(__file__).resolve().parents[2]
STORY = "sc-14541"
COMPONENT = "small-music"
SNAPSHOT = SPEC_BY_KEY[COMPONENT]
P0 = ROOT / "docs/migration/sa3-reference/small-music-reference.safetensors"
DEFAULT_OUTPUT = ROOT / "docs/migration/sa3-dit-reference"
ARTIFACT = "dit-intermediates.safetensors"
SOURCE_FILES = (
    "stable_audio_3/models/blocks.py",
    "stable_audio_3/models/conditioners.py",
    "stable_audio_3/models/diffusion.py",
    "stable_audio_3/models/dit.py",
    "stable_audio_3/models/transformer.py",
)
EXPECTED_SOURCE_SHA256 = {
    "stable_audio_3/models/blocks.py": "eae8aaa6eeee47d39b2276caea12627413ecd2816a61be5cce126453a69c3f01",
    "stable_audio_3/models/conditioners.py": "11d3496a04c8390f0de2ea869e7188c621d97859734bef4930027ff626989a47",
    "stable_audio_3/models/diffusion.py": "b0eab27f7965b4a21a34bf009c932128217524c09cddbfd601bffe866e4b6732",
    "stable_audio_3/models/dit.py": "4b28dcb74937cdcab9ab5f745221178304f998f53aa60d2ee9e74b8fa76fd4fb",
    "stable_audio_3/models/transformer.py": "7436d5d1f040ed74af38899ca57c1eb7b6f3ee5e5c27a3e7402f39f22a52a83a",
}
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "torch": "2.7.1",
    "transformers": "5.8.0",
}
EXPECTED_INPUTS = {
    "seed": 14534,
    "secondsTotal": 0.25,
    "timestep": 0.5,
    "latentShape": [1, 256, 16],
    "promptShape": [1, 256, 768],
    "localOrder": ["inpaint_mask", "inpaint_masked_input"],
    "crossMask": None,
    "paddingSemantics": "zero-v-only",
}
TENSOR_KEYS = {
    "number_features",
    "number_embedding",
    "raw_context",
    "projected_context",
    "duration_global",
    "timestep_features",
    "timestep_embedding",
    "combined_global",
    "global_modulation",
    "preprocessed",
    "projected_input",
    "with_memory",
    "rotary_frequencies",
    "layer0_local",
    "layer0_output",
    "trimmed",
    "projected_output",
    "output",
    "padding_mask",
    "partial_padding_output",
}


def _validate_upstream(path: Path) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise InvalidReference("upstream checkout revision mismatch")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    unexpected = [
        line
        for line in status
        if "__pycache__/" not in line and not line.endswith(".pyc")
    ]
    if unexpected:
        raise InvalidReference(f"upstream checkout has source changes: {unexpected[:3]}")


def _hash(path: Path) -> str:
    return sha256_file(path)


def _tensor_header(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 8:
        raise InvalidReference("truncated safetensors artifact")
    header_len = int.from_bytes(data[:8], "little")
    if header_len > len(data) - 8:
        raise InvalidReference("invalid safetensors header length")
    return json.loads(data[8 : 8 + header_len])


def _tensor_records(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    header_len = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + header_len])
    data_start = 8 + header_len
    records = {}
    for name, record in header.items():
        if name == "__metadata__":
            continue
        start, end = record["data_offsets"]
        records[name] = {
            "dtype": record["dtype"],
            "shape": record["shape"],
            "sha256": hashlib.sha256(data[data_start + start : data_start + end]).hexdigest(),
        }
    return records


def generate(upstream: Path, snapshot: Path, output: Path) -> dict[str, Any]:
    _validate_upstream(upstream)
    if snapshot_revision(snapshot) != SNAPSHOT.revision:
        raise InvalidReference("small-music snapshot revision mismatch")
    locks = load_snapshot_lock()["snapshots"][COMPONENT]["files"]
    for name in ("model_config.json", "model.safetensors"):
        path = snapshot / name
        if _hash(path) != locks[name]["sha256"]:
            raise InvalidReference(f"small-music {name} payload mismatch")
    if not P0.is_file():
        raise InvalidReference("missing committed sc-14534 P0 artifact")

    torch, save_file, _sample_diffusion, loaders, versions = _torch_modules(upstream)
    _load_autoencoder, load_diffusion_cond = loaders
    config = json.loads((snapshot / "model_config.json").read_text(encoding="utf-8"))
    wrapper = load_diffusion_cond(
        _localize_t5(config, snapshot),
        str(snapshot / "model.safetensors"),
        device="cpu",
        model_half=False,
    )
    dit = wrapper.model.model.eval()
    from safetensors.torch import load_file

    p0 = load_file(str(P0), device="cpu")
    noise = p0["dit_noise"]
    timestep = p0["dit_timestep"].float()
    prompt = p0["t5_projected_padded"].float()
    seconds = torch.tensor([0.25], dtype=torch.float32)
    number = wrapper.conditioner.conditioners["seconds_total"]
    normalized = seconds.clamp(number.min_val, number.max_val)
    normalized = (normalized - number.min_val) / (number.max_val - number.min_val)
    number_features = number.embedder.embedding[0](normalized)
    number_embedding = number.embedder.embedding[1](number_features)
    duration_token = number_embedding.unsqueeze(1)
    raw_context = torch.cat([prompt, duration_token], dim=1)
    projected_context = dit.to_cond_embed(raw_context)
    duration_global = dit.to_global_embed(number_embedding)
    timestep_features = dit.timestep_features(timestep[:, None])
    timestep_embedding = dit.to_timestep_embed(timestep_features)
    combined_global = duration_global + timestep_embedding

    preprocessed = dit.preprocess_conv(noise) + noise
    projected_input = dit.transformer.project_in(preprocessed.transpose(1, 2))
    memory = dit.transformer.memory_tokens.expand(projected_input.shape[0], -1, -1)
    with_memory = torch.cat([memory, projected_input], dim=1)
    rotary = dit.transformer.rotary_pos_emb.forward_from_seq_len(with_memory.shape[1])
    rotary_frequencies = rotary[0]
    global_modulation = dit.transformer.global_cond_embedder(combined_global)
    local = torch.zeros((1, 16, 257), dtype=torch.float32)
    first = dit.transformer.layers[0]
    local_latent = first.to_local_embed(local)
    layer0_local = torch.nn.functional.pad(local_latent, (0, 0, 64, 0))

    with torch.inference_mode():
        hidden = with_memory
        for index, layer in enumerate(dit.transformer.layers):
            hidden = layer(
                hidden,
                context=projected_context,
                global_cond=global_modulation,
                local_add_cond=local,
                rotary_pos_emb=rotary,
                padding_mask=None,
            )
            if index == 0:
                layer0_output = hidden
        trimmed = hidden[:, 64:, :]
        projected_output = dit.transformer.project_out(trimmed)
        output_bct = projected_output.transpose(1, 2)
        final_output = dit.postprocess_conv(output_bct) + output_bct

        padding_mask = torch.tensor(
            [[True] * 8 + [False] * 8], dtype=torch.bool
        )
        partial_padding_output = dit._forward(
            noise,
            timestep,
            cross_attn_cond=raw_context,
            global_embed=number_embedding,
            local_add_cond=torch.zeros((1, 257, 16), dtype=torch.float32),
            padding_mask=padding_mask,
        )

    max_abs = (final_output - p0["dit_prediction"]).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        final_output.flatten().double(),
        p0["dit_prediction"].flatten().double(),
        dim=0,
    ).item()
    if cosine < 0.999999 or max_abs != 0.0:
        raise InvalidReference(
            f"manual frozen forward drifted from P0: cosine={cosine} max_abs={max_abs}"
        )

    tensors = {
        "number_features": number_features,
        "number_embedding": number_embedding,
        "raw_context": raw_context,
        "projected_context": projected_context,
        "duration_global": duration_global,
        "timestep_features": timestep_features,
        "timestep_embedding": timestep_embedding,
        "combined_global": combined_global,
        "global_modulation": global_modulation,
        "preprocessed": preprocessed,
        "projected_input": projected_input,
        "with_memory": with_memory,
        "rotary_frequencies": rotary_frequencies,
        "layer0_local": layer0_local,
        "layer0_output": layer0_output,
        "trimmed": trimmed,
        "projected_output": projected_output,
        "output": final_output,
        "padding_mask": padding_mask.to(torch.uint8),
        "partial_padding_output": partial_padding_output,
    }
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / ARTIFACT
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
        str(artifact),
        metadata={
            "story": STORY,
            "upstreamCommit": UPSTREAM_COMMIT,
            "snapshotRevision": SNAPSHOT.revision,
        },
    )
    manifest = {
        "schemaVersion": 1,
        "story": STORY,
        "upstream": {
            "commit": UPSTREAM_COMMIT,
            "sourceSha256": {
                name: _hash(upstream / name) for name in SOURCE_FILES
            },
        },
        "snapshot": {
            "repository": SNAPSHOT.repository,
            "revision": SNAPSHOT.revision,
            "configSha256": locks["model_config.json"]["sha256"],
            "modelSha256": locks["model.safetensors"]["sha256"],
        },
        "p0Sha256": _hash(P0),
        "runtime": {
            "python": versions["python"],
            "torch": versions["torch"],
            "transformers": versions["transformers"],
        },
        "inputs": EXPECTED_INPUTS,
        "artifact": {
            "file": ARTIFACT,
            "bytes": artifact.stat().st_size,
            "sha256": _hash(artifact),
            "tensors": _tensor_records(artifact),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify(output, upstream)
    return manifest


def verify(output: Path, upstream: Path | None = None) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != 1 or manifest.get("story") != STORY:
        raise InvalidReference("DiT oracle schema/story mismatch")
    if manifest.get("upstream", {}).get("commit") != UPSTREAM_COMMIT:
        raise InvalidReference("DiT oracle upstream revision mismatch")
    if manifest["upstream"].get("sourceSha256") != EXPECTED_SOURCE_SHA256:
        raise InvalidReference("DiT oracle source provenance mismatch")
    if upstream is not None:
        _validate_upstream(upstream)
        actual_sources = {name: _hash(upstream / name) for name in SOURCE_FILES}
        if actual_sources != manifest["upstream"]["sourceSha256"]:
            raise InvalidReference("DiT oracle frozen source payload mismatch")
    if manifest.get("p0Sha256") != _hash(P0):
        raise InvalidReference("DiT oracle P0 dependency mismatch")
    locks = load_snapshot_lock()["snapshots"][COMPONENT]["files"]
    if manifest.get("snapshot") != {
        "repository": SNAPSHOT.repository,
        "revision": SNAPSHOT.revision,
        "configSha256": locks["model_config.json"]["sha256"],
        "modelSha256": locks["model.safetensors"]["sha256"],
    }:
        raise InvalidReference("DiT oracle snapshot provenance mismatch")
    if manifest.get("runtime") != EXPECTED_RUNTIME:
        raise InvalidReference("DiT oracle runtime mismatch")
    if manifest.get("inputs") != EXPECTED_INPUTS:
        raise InvalidReference("DiT oracle input contract mismatch")
    artifact = output / manifest["artifact"]["file"]
    if artifact.stat().st_size != manifest["artifact"]["bytes"]:
        raise InvalidReference("DiT oracle artifact byte length mismatch")
    if _hash(artifact) != manifest["artifact"]["sha256"]:
        raise InvalidReference("DiT oracle artifact hash mismatch")
    records = _tensor_records(artifact)
    if set(records) != TENSOR_KEYS or records != manifest["artifact"]["tensors"]:
        raise InvalidReference("DiT oracle tensor inventory/payload mismatch")
    metadata = _tensor_header(artifact).get("__metadata__")
    if metadata != {
        "story": STORY,
        "upstreamCommit": UPSTREAM_COMMIT,
        "snapshotRevision": SNAPSHOT.revision,
    }:
        raise InvalidReference("DiT oracle artifact metadata mismatch")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--upstream", type=Path, required=True)
    generate_parser.add_argument("--snapshot", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify_parser.add_argument("--upstream", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            manifest = generate(args.upstream, args.snapshot, args.output)
        else:
            manifest = verify(args.output, args.upstream)
    except (InvalidReference, OSError, KeyError, ValueError) as error:
        print(f"invalid SA3 DiT reference: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
