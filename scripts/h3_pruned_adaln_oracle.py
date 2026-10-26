#!/usr/bin/env python3
"""Generate weights-free S5 proof from pinned executable WanGP CPU arithmetic.

No downloaded table/checkpoint is used. --reference-dir contains the three pinned
textual source files, fetched separately; offline regeneration verifies their hashes.
Runtime inference never imports Python or reconstructs a full basis.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

REVISION = "47b06ff920c16b4b7b36de068367a5de0e0968e9"
SOURCES = {
    "transformer.py": "5f0b81959522333830c7f3d422c2b0138b5f4fd37287a98ebb1705fd04f0ca50",
    "prune_checkpoint.py": "91b0c55a1fa7dcff0acaad5d89df68a40cc54545f14c1a9d37342f703027d1f1",
    "components/packing.py": "8374dcd5e4023acca6fbaed2ed2fc98eabf2f24b019f5c4bfae88b7e36a1fe5b",
}


def pinned_definitions(root, path, names, namespace):
    data = (root / path).read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCES[path]:
        raise ValueError(f"pinned reference hash mismatch: {path}")
    tree = ast.parse(data, filename=path)
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"missing pinned executable definitions in {path}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, "exec"), namespace)


def attention(qkv_list, recycle_q=False):
    # Only the external attention dispatch is supplied locally. The pinned block,
    # projections, norms, modulation, residuals and final layer execute unchanged.
    q, k, v = qkv_list
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


def deterministic(name, shape, norm=False):
    salt = int.from_bytes(hashlib.sha256(name.encode()).digest()[:2], "little")
    x = ((torch.arange(math.prod(shape), dtype=torch.float32) * 13 + salt) % 41 - 20) / 128
    return (x * 0.1 + 1 if norm else x).reshape(shape)


def flat(tensor):
    return tensor.detach().float().flatten().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    ns = {"torch": torch, "nn": nn, "F": F, "math": math, "pay_attention": attention}
    pinned_definitions(args.reference_dir, "prune_checkpoint.py", ["_factor_curve", "_compress_projection", "_interpolated_table"], ns)
    pinned_definitions(args.reference_dir, "transformer.py", ["_take", "_modulate", "_gated_residual", "MLP", "Attention", "AdalnProj", "DiTBlock", "FinalLayer"], ns)
    pinned_definitions(args.reference_dir, "components/packing.py", ["build_row_timesteps"], ns)
    config = dict(num_attention_heads=2, attention_head_dim=8, hidden_size=8, num_layers=2,
                  num_refiner_layers=1, ffn_dim=16, in_channels=4, audio_in_channels=4,
                  patch_size=[1, 1, 1], text_dim=8, freq_dim=8, time_embed_hidden_dim=16,
                  time_embed_dim=16, rope_freq_dim=1, rope_theta=10000.0,
                  norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5)
    blocks = [ns["DiTBlock"](8, 2, 8, 16, 8, 1e-5, 1e-5, apply_silu=False,
                              adaln_dtype=torch.float32, ffn_chunk_size=0, dtype=torch.float32, device="cpu") for _ in range(2)]
    final = ns["FinalLayer"](8, 8, 4, 4, 1e-5, apply_silu=False, adaln_dtype=torch.float32, dtype=torch.float32, device="cpu")
    tensors = {}

    def record(name, value, dtype="F32"):
        tensors[name] = {"shape": list(value.shape), "dtype": dtype, "data": flat(value)}

    t = torch.linspace(0, 1, 1025, dtype=torch.float32)
    curve = torch.stack([torch.sin(t * (j + 1) * 1.7) * 0.3 + t.square() * (j + 1) / 20 + (j - 4) / 9 for j in range(16)], 1)
    table, basis, mean, metrics = ns["_factor_curve"](curve, 8)
    record("adaln_t_table", table)
    with torch.no_grad():
        for index, module in enumerate([*blocks, final]):
            prefix = f"transformer_blocks.{index}" if index < 2 else "norm_out"
            for name, param in module.named_parameters():
                if name.startswith("adaln_proj."):
                    continue
                param.copy_(deterministic(f"{prefix}.{name}", tuple(param.shape), norm="norm" in name))
            adaln = module.adaln_proj.linear
            original = {
                "p.weight": deterministic(f"{prefix}.full.weight", (adaln.weight.shape[0], 16)).to(torch.bfloat16),
                "p.bias": deterministic(f"{prefix}.full.bias", (adaln.weight.shape[0],)).to(torch.float16),
            }
            source = SimpleNamespace(get_tensor=original.__getitem__)
            folded_bias, reduced_weight = ns["_compress_projection"](source, "p.bias", basis, mean, "cpu", 64)
            stored_weight, stored_bias = reduced_weight.to(torch.bfloat16), folded_bias.to(torch.float16)
            # Pinned transformer locks curve projection compute to F32. Widening the
            # supplied source operands is lossless; it does not invent F32 source bytes.
            adaln.weight.copy_(stored_weight.float())
            adaln.bias.copy_(stored_bias.float())
            reduced_prefix = f"{prefix}.adaln_proj.linear" if index < 2 else "norm_out.linear"
            record(reduced_prefix + ".weight", stored_weight, "BF16")
            record(reduced_prefix + ".bias", stored_bias, "F16")
            if index < 2:
                record(prefix + ".norm1.weight", module.norm1.weight)
                record(prefix + ".norm2.weight", module.norm2.weight)
                qkv = module.attn.qkv_proj.weight.reshape(2, 3, 8, 8)
                for part, key in enumerate(["to_q", "to_k", "to_v"]):
                    record(prefix + f".attn.{key}.weight", qkv[:, part].reshape(16, 8))
                record(prefix + ".attn.norm_q.weight", module.attn.q_norm.weight)
                record(prefix + ".attn.norm_k.weight", module.attn.k_norm.weight)
                record(prefix + ".attn.to_out.0.weight", module.attn.out_proj.weight)
                gate, value = module.mlp.fc1.weight.chunk(2, 0)
                record(prefix + ".ff.net.0.proj.weight", torch.cat((value, gate), 0))
                record(prefix + ".ff.net.2.weight", module.mlp.fc2.weight)
            else:
                record("norm_out.norm.weight", module.norm.weight)
                for target, head in [("proj_out", module.video_out), ("audio_proj_out", module.audio_out)]:
                    record(target + ".weight", head.weight)
                    record(target + ".bias", head.bias)

        hidden = deterministic("input_hidden", (5, 8)) * 3
        layout = SimpleNamespace(sequence_length=5, token_tags=torch.tensor([1, 0, 2, 2, 0]),
                                 video_indices=torch.tensor([1, 4]), audio_indices=torch.tensor([2, 3]),
                                 num_condition_video_rows=1, num_condition_audio_rows=1)
        cases = []
        for video_t in [-0.1, 0.0, 0.0001, 0.12548828125, 0.23, 0.5, 0.999, 0.9995, 1.0, 1.1]:
            # Four source row classes deliberately retain duplicate coordinates at endpoints.
            times = torch.tensor([video_t, max(video_t, .999), 1 - video_t, 1], dtype=torch.float32)
            _, packed_indices = ns["build_row_timesteps"](layout, float(times[0]), float(times[2]), float(times[1]), 1.0)
            unique_times, _ = ns["build_row_timesteps"](layout, float(times[0]), float(times[2]), float(times[1]), 1.0)
            packed_times = unique_times[packed_indices]
            assert torch.equal(packed_times, times[torch.tensor([0, 1, 3, 2, 0])])
            coordinates = ns["_interpolated_table"](table, times)
            state = hidden.clone()
            block_outputs, modulations = [], []
            adaln_indices = [1, 3, 11, 8, 0]
            segments = [(i, i + 1, row) for i, row in enumerate(adaln_indices)]
            for block in blocks:
                modulations.append(flat(block.adaln_proj.linear(coordinates)))
                state = block([state], coordinates, segments, rope=None)
                block_outputs.append(flat(state))
            norm_projection = final.adaln_proj.linear(coordinates)
            video, audio = final([state], coordinates, (4, 5, 0), (3, 4, 2))
            cases.append(dict(timesteps=flat(times), coordinates=flat(coordinates),
                              block_modulations=modulations, norm_modulation=flat(norm_projection),
                              block_outputs=block_outputs, video_output=flat(video), audio_output=flat(audio)))
    result = dict(story="sc-26684", reference_revision=REVISION, reference_sha256=SOURCES,
                  torch=torch.__version__, device="cpu", config=config, tensors=tensors,
                  input_hidden=flat(hidden), adaln_indices=[1, 3, 11, 8, 0], timestep_indices=[0, 1, 3, 2, 0],
                  positions=[0.0] * 15, cases=cases, synthetic_factorization=metrics,
                  precision="Stored table F32, reduced weight BF16, folded bias F16; F32 projection compute as pinned curve runtime. Body F32.",
                  tolerances=dict(coordinates_abs=2e-6, projection_abs=2e-5, candle_block_abs=2e-5, mlx_block_abs=2e-3))
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"generated {len(cases)} off-grid/endpoint/conditioned cases, {len(tensors)} synthetic native operands; cpu only")


if __name__ == "__main__":
    main()
