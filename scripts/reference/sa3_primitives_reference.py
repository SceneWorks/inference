#!/usr/bin/env python3
"""Generate/verify the independently versioned SA3 primitive oracle (sc-14536)."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

UPSTREAM_COMMIT = "124e8a799f57a1f665495ecb72e547d0a62867f1"
STORY = "sc-14536"
SCHEMA_VERSION = 1
EXPECTED_ENVIRONMENT = {
    "python": "3.12.13",
    "torch": "2.7.1",
    "torchaudio": "2.7.1",
    "transformers": "5.8.0",
}
UV_LOCK_SHA256 = "f6422550a634b3e44664235910835bde49ec930b8026ec34f28e4ddc2e724150"
SOURCE_LOCK_KEYS = {
    "small": "small-music",
    "medium": "medium",
    "sameS": "same-s",
    "sameL": "same-l",
}
ROOT = Path(__file__).resolve().parents[2]
P0_MANIFEST = ROOT / "docs/migration/sa3-reference/manifest.json"
SNAPSHOT_LOCK = ROOT / "docs/migration/sa3-reference/snapshot-files.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_sources() -> dict:
    p0 = json.loads(P0_MANIFEST.read_text(encoding="utf-8"))
    file_lock = json.loads(SNAPSHOT_LOCK.read_text(encoding="utf-8"))
    if p0["upstream"]["commit"] != UPSTREAM_COMMIT:
        raise SystemExit("sc-14534 manifest upstream revision drift")
    if p0["referenceEnvironment"] != {
        "device": "cpu",
        "platform": "macOS-26.5.2-arm64-arm-64bit",
        **EXPECTED_ENVIRONMENT,
    }:
        raise SystemExit("sc-14534 reference environment drift")
    p0_by_key = {entry["key"]: entry for entry in p0["snapshots"]}
    result = {}
    for name, lock_key in SOURCE_LOCK_KEYS.items():
        snapshot = file_lock["snapshots"][lock_key]
        p0_snapshot = p0_by_key[lock_key]
        model = snapshot["files"]["model.safetensors"]
        if (
            snapshot["revision"] != p0_snapshot["revision"]
            or model["bytes"] != p0_snapshot["modelBytes"]
            or snapshot["repository"] != p0_snapshot["repository"]
        ):
            raise SystemExit(f"{lock_key}: sc-14534 manifest/snapshot lock disagree")
        result[name] = {
            "lockKey": lock_key,
            "repository": snapshot["repository"],
            "revision": snapshot["revision"],
            "model": "model.safetensors",
            "modelBytes": model["bytes"],
            "modelSha256": model["sha256"],
        }
    return result


def require_upstream(path: Path) -> None:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True, encoding="utf-8"
    ).strip()
    if head != UPSTREAM_COMMIT:
        raise SystemExit(f"upstream HEAD {head} != pinned {UPSTREAM_COMMIT}")
    if sha256(path / "uv.lock") != UV_LOCK_SHA256:
        raise SystemExit("upstream uv.lock hash mismatch")
    sys.path.insert(0, str(path))


def load_prefixed(path: Path, prefix: str) -> dict:
    from safetensors import safe_open

    result = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                result[key.removeprefix(prefix)] = handle.get_tensor(key)
    if not result:
        raise SystemExit(f"{path}: no tensors below {prefix}")
    return result


def tensor_metadata(tensors: dict) -> dict:
    result = {}
    for name, tensor in sorted(tensors.items()):
        raw = tensor.detach().cpu().contiguous().numpy().tobytes()
        result[name] = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return result


def file_tensor_metadata(path: Path) -> dict:
    """Inspect a safetensors artifact using only the standard library."""
    data = path.read_bytes()
    header_length = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + header_length])
    payload = 8 + header_length
    dtype_names = {
        "BOOL": "bool",
        "F32": "float32",
    }
    result = {}
    for name, entry in sorted(header.items()):
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        result[name] = {
            "dtype": dtype_names.get(entry["dtype"], entry["dtype"].lower()),
            "shape": entry["shape"],
            "sha256": hashlib.sha256(data[payload + start : payload + end]).hexdigest(),
        }
    return result


def generate(args: argparse.Namespace) -> None:
    require_upstream(args.upstream)
    import torch
    import torch.nn.functional as F
    import torchaudio
    import transformers
    from unittest.mock import patch as mock_patch
    from safetensors.torch import safe_open, save_file
    from stable_audio_3.models.blocks import WNConv1d
    from stable_audio_3.models.bottleneck import SoftNormBottleneck
    from stable_audio_3.models.pretransforms import PatchedPretransform
    from stable_audio_3.models.transformer import (
        Attention,
        FeedForward,
        LayerNorm,
        LayerScale,
        RMSNorm,
        RotaryEmbedding,
        TransformerBlock,
    )

    environment = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
    }
    if environment != EXPECTED_ENVIRONMENT:
        raise SystemExit(
            f"runtime environment {environment} != pinned {EXPECTED_ENVIRONMENT}"
        )
    torch.manual_seed(14536)
    torch.set_grad_enabled(False)
    outputs = {}
    sources = {}
    source_locks = locked_sources()

    def source(name: str, snapshot: Path) -> None:
        expected = source_locks[name]
        actual = {
            **expected,
            "revision": snapshot.parent.name,
            "model": snapshot.name,
            "modelBytes": snapshot.stat().st_size,
            "modelSha256": sha256(snapshot),
        }
        if actual != expected:
            raise SystemExit(f"{name}: snapshot does not match sc-14534 locks")
        sources[name] = actual

    source("small", args.small)
    source("medium", args.medium)
    source("sameS", args.same_s)
    source("sameL", args.same_l)

    # Small: ordinary self/cross attention, RMS/force-fp32, AdaLN, local projection, FF mult 4.
    small_prefix = "model.model.transformer.layers.0."
    small = TransformerBlock(
        1024,
        dim_heads=64,
        cross_attend=True,
        dim_context=1024,
        global_cond_dim=768,
        local_add_cond_dim=257,
        norm_type="rms_norm",
        norm_kwargs={"force_fp32": True, "eps": 1e-5},
        attn_kwargs={"qk_norm": "rms", "qk_norm_eps": 1e-6, "differential": False},
        ff_kwargs={"mult": 4.0},
    ).eval()
    small.load_state_dict(load_prefixed(args.small, small_prefix), strict=True)
    small_x = torch.randn(1, 4, 1024)
    small_context = torch.randn(1, 3, 1024)
    small_global = torch.randn(1, 6144)
    small_local = torch.randn(1, 4, 257)
    small_padding = torch.tensor([[True, True, False, True]])
    small_rope = RotaryEmbedding(32).forward_from_seq_len(4)
    outputs.update(
        small_x=small_x,
        small_context=small_context,
        small_global=small_global,
        small_local=small_local,
        small_padding=small_padding,
        small_rope=small_rope[0],
        small_block=small(
            small_x,
            context=small_context,
            global_cond=small_global,
            local_add_cond=small_local,
            rotary_pos_emb=small_rope,
            padding_mask=small_padding,
        ),
    )
    with safe_open(args.small, framework="pt", device="cpu") as handle:
        memory = handle.get_tensor("model.model.transformer.memory_tokens")
    outputs["small_memory"] = memory
    outputs["small_memory_prepend"] = torch.cat(
        (memory.unsqueeze(0), small_x), dim=1
    )
    outputs["small_memory_mask"] = torch.cat(
        (torch.ones(1, 64, dtype=torch.bool), small_padding), dim=1
    )

    # Medium: the direct-subtraction differential attention branch.
    medium_prefix = "model.model.transformer.layers.0.self_attn."
    medium = Attention(
        1536,
        dim_heads=64,
        qk_norm="rms",
        qk_norm_eps=1e-6,
        differential=True,
        zero_init_output=False,
    ).eval()
    medium.load_state_dict(load_prefixed(args.medium, medium_prefix), strict=True)
    medium_x = torch.randn(1, 3, 1536)
    medium_padding = torch.tensor([[True, False, True]])
    medium_rope = RotaryEmbedding(32).forward_from_seq_len(3)
    outputs.update(
        medium_x=medium_x,
        medium_padding=medium_padding,
        medium_rope=medium_rope[0],
        medium_attention=medium(
            medium_x, rotary_pos_emb=medium_rope, padding_mask=medium_padding
        ),
    )

    # SAME-S: DyT + differential attention + SwiGLU mult 3.
    same_s_prefix = "encoder.layers.0.transformers.0."
    same_s = TransformerBlock(
        768,
        dim_heads=64,
        norm_type="dyt",
        add_rope=True,
        attn_kwargs={"qk_norm": "dyt", "differential": True},
        ff_kwargs={"mult": 3.0},
        zero_init_branch_outputs=False,
    ).eval()
    same_s.load_state_dict(load_prefixed(args.same_s, same_s_prefix), strict=True)
    same_s_x = torch.randn(1, 5, 768)
    outputs.update(same_s_x=same_s_x, same_s_block=same_s(same_s_x))

    # SAME-L: strict `(12 - i) < 8` selects indices 5..11. Index 5 proves Sin(pi*x).
    same_l_prefix = "decoder.layers.3.transformers.5.ff."
    same_l_ff = FeedForward(
        1536, mult=3.0, sinusoidal=True, zero_init_output=False
    ).eval()
    same_l_ff.load_state_dict(load_prefixed(args.same_l, same_l_prefix), strict=True)
    same_l_x = torch.randn(1, 2, 1536)
    outputs.update(same_l_x=same_l_x, same_l_sin_ff=same_l_ff(same_l_x))

    # Compact frozen-upstream branch oracle for missing mutation-sensitive paths.
    branch_ln = LayerNorm(
        4, bias=False, fix_scale=True, force_fp32=True, eps=1e-5
    ).eval()
    branch_ln.gamma.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
    branch_ln_x = torch.tensor(
        [[[1000.0, 1000.5, 999.0, 1001.0], [-3.0, 2.0, 7.0, -5.0]]],
        dtype=torch.float16,
    )
    outputs.update(
        branch_ln_x=branch_ln_x,
        branch_ln_output=branch_ln(branch_ln_x),
        **{
            f"branch_ln.{key}": value
            for key, value in branch_ln.state_dict().items()
        },
    )

    branch_rms = RMSNorm(4, fix_scale=True, force_fp32=True, eps=1e-5).eval()
    branch_rms.gamma.copy_(torch.tensor([0.25, 0.75, 1.25, 2.0]))
    branch_rms_x = torch.tensor(
        [[[1000.0, 1000.5, 999.0, 1001.0], [-3.0, 2.0, 7.0, -5.0]]],
        dtype=torch.float16,
    )
    outputs.update(
        branch_rms_x=branch_rms_x,
        branch_rms_output=branch_rms(branch_rms_x),
        **{
            f"branch_rms.{key}": value
            for key, value in branch_rms.state_dict().items()
        },
    )

    branch_scale = LayerScale(4, init_val=1e-5).eval()
    branch_scale.scale.copy_(torch.tensor([0.25, -0.5, 2.0, 3.0]))
    branch_scale_x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    outputs.update(
        branch_scale_x=branch_scale_x,
        branch_scale_output=branch_scale(branch_scale_x),
        **{
            f"branch_scale.{key}": value
            for key, value in branch_scale.state_dict().items()
        },
    )

    branch_cross = Attention(
        4,
        dim_heads=2,
        dim_context=4,
        qk_norm="none",
        differential=True,
        zero_init_output=False,
    ).eval()
    branch_cross_x = torch.randn(1, 3, 4)
    branch_cross_context = torch.randn(1, 4, 4)
    branch_cross_padding = torch.tensor([[True, False, True, True]])
    branch_cross_additive = torch.tensor(
        [[[[0.0, float("-inf"), -0.25, -1.0],
           [-0.5, float("-inf"), 0.0, -0.75],
           [-1.0, float("-inf"), -0.25, 0.0]]]]
    )
    original_sdpa = F.scaled_dot_product_attention

    def masked_sdpa(q, k, v, *positional, **kwargs):
        if positional or kwargs.get("attn_mask") is not None:
            raise RuntimeError("unexpected upstream SDPA mask arguments")
        kwargs["attn_mask"] = branch_cross_additive
        return original_sdpa(q, k, v, **kwargs)

    with mock_patch.object(F, "scaled_dot_product_attention", masked_sdpa):
        branch_cross_output = branch_cross(
            branch_cross_x,
            context=branch_cross_context,
            padding_mask=branch_cross_padding,
        )
    outputs.update(
        branch_cross_x=branch_cross_x,
        branch_cross_context=branch_cross_context,
        branch_cross_padding=branch_cross_padding,
        branch_cross_additive=branch_cross_additive,
        branch_cross_output=branch_cross_output,
        **{
            f"branch_cross.{key}": value
            for key, value in branch_cross.state_dict().items()
        },
    )

    # Attention qk_norm="ln" is torch.nn.LayerNorm (weight/bias keys), unlike the custom block
    # LayerNorm above (gamma/beta keys). Non-default affine values make normalization mandatory.
    branch_qk_ln = Attention(
        4,
        dim_heads=2,
        qk_norm="ln",
        zero_init_output=False,
    ).eval()
    branch_qk_ln.q_norm.weight.copy_(torch.tensor([0.5, 1.5]))
    branch_qk_ln.q_norm.bias.copy_(torch.tensor([-0.25, 0.75]))
    branch_qk_ln.k_norm.weight.copy_(torch.tensor([2.0, -0.5]))
    branch_qk_ln.k_norm.bias.copy_(torch.tensor([0.125, -0.375]))
    branch_qk_ln_x = torch.randn(1, 3, 4)
    outputs.update(
        branch_qk_ln_x=branch_qk_ln_x,
        branch_qk_ln_output=branch_qk_ln(branch_qk_ln_x),
        **{
            f"branch_qk_ln.{key}": value
            for key, value in branch_qk_ln.state_dict().items()
        },
    )

    # Execute the frozen upstream WNConv1d object with actual SAME-S legacy g/v weights.
    with safe_open(args.same_s, framework="pt", device="cpu") as handle:
        g = handle.get_tensor("encoder.layers.0.mapping.weight_g")
        v = handle.get_tensor("encoder.layers.0.mapping.weight_v")
        bias = handle.get_tensor("encoder.layers.0.mapping.bias")
    wn = WNConv1d(512, 768, 1, bias=True).eval()
    wn.load_state_dict({"weight_g": g, "weight_v": v, "bias": bias}, strict=True)
    wn_x = torch.randn(1, 512, 7)
    outputs.update(
        wn_x=wn_x,
        wn_g=g,
        wn_v=v,
        wn_bias=bias,
        wn_output=wn(wn_x),
    )

    # Patch size 256: non-divisible input round-trips its prefix and preserves a zero tail.
    patch = PatchedPretransform(channels=2, patch_size=256)
    patch_x = torch.randn(1, 2, 259)
    patch_encoded = patch.encode(patch_x)
    outputs.update(
        patch_x=patch_x,
        patch_encoded=patch_encoded,
        patch_decoded=patch.decode(patch_encoded),
    )

    # SoftNorm uses actual weights. Patch only the RNG source so the frozen upstream decode method
    # itself executes at train and eval while the exact sampled noise remains replayable in Rust.
    soft = SoftNormBottleneck(
        dim=256, noise_regularize=True, auto_scale=True
    ).eval()
    soft.load_state_dict(load_prefixed(args.same_s, "bottleneck."), strict=True)
    soft_x = torch.randn(1, 256, 5)
    soft_noise = torch.randn_like(soft_x)
    encoded = soft.encode(soft_x)
    with mock_patch("torch.randn_like", return_value=soft_noise):
        soft.eval()
        soft_eval = soft.decode(encoded)
    with mock_patch("torch.randn_like", return_value=soft_noise):
        soft.train()
        soft_train = soft.decode(encoded)
    outputs.update(
        soft_x=soft_x,
        soft_noise=soft_noise,
        soft_encoded=encoded,
        soft_eval=soft_eval,
        soft_train=soft_train,
    )

    # Candle's mmap backend does not accept BOOL safetensors entries. Masks retain their exact
    # zero/one values as f32 in the portable artifact (the upstream calls above used bool masks).
    outputs = {
        name: value.detach().cpu().to(torch.float32).contiguous()
        for name, value in outputs.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    artifact = args.output / "primitives.safetensors"
    save_file(outputs, artifact)
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "story": STORY,
        "upstreamCommit": UPSTREAM_COMMIT,
        "environment": environment,
        "upstreamLock": {
            "file": "uv.lock",
            "sha256": UV_LOCK_SHA256,
        },
        "sources": sources,
        "artifact": {
            "file": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": sha256(artifact),
            "tensors": tensor_metadata(outputs),
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schemaVersion"] != SCHEMA_VERSION:
        raise SystemExit("primitive oracle schema version mismatch")
    if manifest.get("story") != STORY:
        raise SystemExit("primitive oracle story mismatch")
    if manifest["upstreamCommit"] != UPSTREAM_COMMIT:
        raise SystemExit("primitive oracle upstream revision mismatch")
    if manifest.get("environment") != EXPECTED_ENVIRONMENT:
        raise SystemExit("primitive oracle environment mismatch")
    if manifest.get("upstreamLock") != {
        "file": "uv.lock",
        "sha256": UV_LOCK_SHA256,
    }:
        raise SystemExit("primitive oracle upstream lock mismatch")
    if manifest.get("sources") != locked_sources():
        raise SystemExit("primitive oracle source provenance mismatch")
    artifact = output / manifest["artifact"]["file"]
    if artifact.stat().st_size != manifest["artifact"]["bytes"]:
        raise SystemExit("primitive oracle byte length mismatch")
    if sha256(artifact) != manifest["artifact"]["sha256"]:
        raise SystemExit("primitive oracle artifact hash mismatch")
    if file_tensor_metadata(artifact) != manifest["artifact"]["tensors"]:
        raise SystemExit("primitive oracle tensor metadata/hash mismatch")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--verify", action="store_true")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--upstream", type=Path)
    result.add_argument("--small", type=Path)
    result.add_argument("--medium", type=Path)
    result.add_argument("--same-s", type=Path)
    result.add_argument("--same-l", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.verify:
        verify(args.output)
    else:
        for name in ("upstream", "small", "medium", "same_s", "same_l"):
            if getattr(args, name) is None:
                raise SystemExit(f"--{name.replace('_', '-')} is required for generation")
        generate(args)


if __name__ == "__main__":
    main()
