"""SYNTHETIC CPU-only training AE oracle; never downloads or uses release weights.

Run with trusted Python -I from a trusted cwd. Arguments: frozen BFL autoencoder.py,
frozen HF source directory (vae.py/resnet.py/attention_processor.py plus the two
trainers and autoencoder_kl_flux2.py), existing tiny AE weights, old inference
golden, output path. Every external executable source byte is hash checked.

The BFL Encoder verifies the unchanged old inference golden FIRST. The actual
hash-verified HF Encoder/Resnet/Attention classes supply the training oracle,
with strict complete native-key conversion (including 1x1 qkv->linear) and a
separate quant_conv. HF uses nn.SiLU, unlike separately rounded BFL swish.
Convolution, group norm and SDPA execute trusted CPU PyTorch selected-dtype
operations, not final-output-only casts. Half CPU kernels are NOT CUDA evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import types
import sys
import re
import json
import importlib.metadata
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

HASHES = {
    "autoencoder.py": "960e2a98046bcb78f2d36fbbd4848db726f14810c7887485f7e2682346f95189",
    "vae.py": "59b51fdf2d951da45d697569271e0c99adf4a959ec0f91885f3d9314e925a12c",
    "resnet.py": "77ad799719fa51bf0fc703f0486915b791c9b69673983abeb3e32c6edcc76650",
    "attention_processor.py": "59871fd4e66fc14a26f0e0464706ab24680e13dcd93161c55123678a5e671772",
    "autoencoder_kl_flux2.py": "1afe08d76a93e994a453bd9036b57d173516211c2e6b9794ea64f485bea57f57",
    "train_dreambooth_lora_flux3_klein.py": "89cc2ff1af1909c1652deefdee7a60e589c3d4e7169d22ab7e05e3ca0c49be72",
    "train_dreambooth_lora_flux3_klein_img2img.py": "c33420f2f487476474ae12e5490655874a6f5226712b2e74f64e19dce6a630c9",
}


def checked(path: Path) -> str:
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == HASHES[path.name], path
    return data.decode("utf-8")


def pack(mean, bn_mean, bn_std):
    # Frozen trainers: deterministic mode, patchify, ORIGINAL F32 BN stats.
    _, e, hh, ww = mean.shape
    h, w = hh // 2, ww // 2
    packed = mean.reshape(1, e, h, 2, w, 2).permute(0, 1, 3, 5, 2, 4).reshape(1, 4 * e, h, w)
    normalized = (packed.float() - bn_mean) / bn_std
    return normalized[0].permute(1, 2, 0).reshape(h * w, 4 * e).contiguous()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bfl", type=Path)
    p.add_argument("hf", type=Path)
    p.add_argument("hf_source", type=Path, help="verified exact HF source tree's src directory")
    p.add_argument("--hf-tree", type=Path, required=True, help="official pinned Git tree JSON (recursive)")
    p.add_argument("--overlay", type=Path, help="optional isolated hash-pinned pure-Python dependencies")
    p.add_argument("--overlay-receipt", type=Path, help="exact --require-hashes dependency pins")
    p.add_argument("weights", type=Path)
    p.add_argument("old_golden", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    tree = json.loads(a.hf_tree.read_text(encoding="utf-8"))
    assert tree["sha"] == "7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941" and not tree["truncated"]
    manifest = sorted((e["path"], e["sha"]) for e in tree["tree"] if e["type"] == "blob" and e["path"].startswith("src/"))
    manifest_sha = hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()
    assert manifest_sha == "bf28920b75c20666f3f6a06494ab09c7dd9a5f954784341ae1a1da6fd6b63af6"
    verified = 0
    for entry in tree["tree"]:
        if entry["type"] != "blob" or not entry["path"].startswith("src/"):
            continue
        data = (a.hf_source / entry["path"].removeprefix("src/")).read_bytes()
        assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == entry["sha"], entry["path"]
        verified += 1
    assert verified == 1099
    code = checked(a.bfl)
    sources = {n: checked(a.hf / n) for n in HASHES if n != "autoencoder.py"}
    assert "self.conv_act = nn.SiLU()" in sources["vae.py"]
    assert "force_upcast" not in sources["autoencoder_kl_flux2.py"]
    for name in ("train_dreambooth_lora_flux3_klein.py", "train_dreambooth_lora_flux3_klein_img2img.py"):
        assert ".latent_dist.mode()" in sources[name]
        assert "vae.to(**to_kwargs)" in sources[name]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(26240)
    module = types.ModuleType("hash_verified_bfl_ae")
    exec(compile(code, str(a.bfl), "exec"), module.__dict__)
    module.CH, module.CH_MULT, module.NUM_RES_BLOCKS, module.EMBED_DIM = 32, (1, 2, 4, 4), 2, 4
    module.LATENT_CHANNELS = 16
    state = load_file(str(a.weights), device="cpu")
    old = load_file(str(a.old_golden), device="cpu")
    encoder = module.Encoder().eval()
    encoder.load_state_dict({k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}, strict=True)
    assert all(not k.startswith("decoder.") for k in encoder.state_dict())
    bn_mean = state["bn.running_mean"].float().reshape(1, -1, 1, 1).clone()
    bn_std = (state["bn.running_var"].float().reshape(1, -1, 1, 1) + 1e-4).sqrt()
    image = old["image"].unsqueeze(0)
    with torch.no_grad():
        # First prove existing F32 inference golden and source graph unchanged.
        legacy = pack(encoder(image)[:, :4], bn_mean, bn_std)
        assert (legacy - old["tokens"]).abs().max().item() < 1e-5
        # Execute ACTUAL frozen HF classes, not a claimed-equivalent BFL half graph.
        if a.overlay:
            assert a.overlay_receipt is not None
            sys.path.insert(0, str(a.overlay.resolve()))
        sys.path.insert(0, str(a.hf_source.resolve()))
        from diffusers.models.autoencoders.vae import Encoder as HFEncoder
        hf_encoder = HFEncoder(in_channels=3, out_channels=4,
            down_block_types=("DownEncoderBlock2D",) * 4,
            block_out_channels=(32, 64, 128, 128), layers_per_block=2,
            norm_num_groups=32, act_fn="silu", double_z=True).eval()
        mapped = {}
        quant = {}
        for key, value in state.items():
            if not key.startswith("encoder."):
                continue
            key = key.removeprefix("encoder.")
            if key.startswith("quant_conv."):
                quant[key.removeprefix("quant_conv.")] = value
                continue
            key = re.sub(r"^down\.(\d+)\.block\.(\d+)\.", r"down_blocks.\1.resnets.\2.", key)
            key = re.sub(r"^down\.(\d+)\.downsample\.", r"down_blocks.\1.downsamplers.0.", key)
            key = key.replace("nin_shortcut.", "conv_shortcut.")
            key = key.replace("mid.block_1.", "mid_block.resnets.0.").replace("mid.block_2.", "mid_block.resnets.1.")
            key = key.replace("norm_out.", "conv_norm_out.")
            if key.startswith("mid.attn_1."):
                key = key.replace("mid.attn_1.norm.", "mid_block.attentions.0.group_norm.")
                for source, dest in (("q", "to_q"), ("k", "to_k"), ("v", "to_v"), ("proj_out", "to_out.0")):
                    if key.startswith(f"mid.attn_1.{source}."):
                        key = key.replace(f"mid.attn_1.{source}.", f"mid_block.attentions.0.{dest}.")
                        if key.endswith("weight"):
                            assert value.shape[2:] == (1, 1)
                            value = value[:, :, 0, 0].contiguous()
                        break
            assert key not in mapped, key
            mapped[key] = value
        hf_encoder.load_state_dict(mapped, strict=True)
        quant_conv = torch.nn.Conv2d(8, 8, 1).eval()
        quant_conv.load_state_dict(quant, strict=True)
        encoder = torch.nn.Sequential(hf_encoder, quant_conv).eval()
        assert sum(v.numel() for v in encoder.state_dict().values()) == sum(v.numel() for k, v in state.items() if k.startswith("encoder."))
        outputs = {"image": image.contiguous(), "original_bn_mean": bn_mean, "original_bn_std": bn_std}
        means = {}
        for label, dtype in (("f32", torch.float32), ("bf16", torch.bfloat16), ("f16", torch.float16)):
            model = copy.deepcopy(encoder).to(dtype=dtype)
            assert all(v.dtype == dtype for v in model.parameters())
            # Selected input cast happens on CPU BEFORE execution.
            x = image.to(dtype=dtype)
            assert x.dtype == dtype
            observed = []
            handles = []
            def observe(mod, args, output):
                assert args[0].dtype == dtype and output.dtype == dtype
                observed.append(type(mod).__name__)
            for mod in model.modules():
                if isinstance(mod, (torch.nn.Conv2d, torch.nn.GroupNorm, torch.nn.SiLU)):
                    handles.append(mod.register_forward_hook(observe))
            moments = model(x)
            for handle in handles:
                handle.remove()
            assert len(observed) > 50 and "SiLU" in observed
            assert moments.dtype == dtype
            mean = moments[:, :4]
            tokens = pack(mean, bn_mean, bn_std)
            assert tokens.dtype == torch.float32 and tokens.shape == (6, 16)
            outputs[f"mean_{label}"] = mean.float().contiguous()
            outputs[f"tokens_{label}"] = tokens
            means[label] = mean
            if dtype != torch.float32:
                wrong_bn = pack(mean, bn_mean.to(dtype).float(), bn_std.to(dtype).float())
                outputs[f"wrong_bn_{label}"] = wrong_bn
                assert (tokens - wrong_bn).abs().max().item() > 1e-4
        final_only = pack(means["f32"].to(torch.bfloat16), bn_mean, bn_std)
        assert (outputs["tokens_bf16"] - final_only).abs().max().item() > 1e-3
        outputs["wrong_final_only_bf16"] = final_only
        # Fused conv+bias must round ONCE. A separately rounded convolution is wrong.
        x = torch.tensor([[[[1.0078125]]]], dtype=torch.bfloat16)
        weight = torch.tensor([[[[1.0078125]]]], dtype=torch.bfloat16)
        bias = torch.tensor([0.00390625], dtype=torch.bfloat16)
        fused = F.conv2d(x.float(), weight.float(), bias.float()).to(torch.bfloat16)
        split = (F.conv2d(x.float(), weight.float()).to(torch.bfloat16) + bias.reshape(1, 1, 1, 1))
        assert not torch.equal(fused, split)
        outputs.update(conv_input=x, conv_weight=weight, conv_bias=bias, conv_fused=fused, conv_wrong_split=split)
        silu_input = torch.linspace(-8, 8, 257).to(torch.bfloat16).reshape(1,1,1,-1)
        silu_fused = F.silu(silu_input)
        silu_wrong_split = silu_input * torch.sigmoid(silu_input)
        assert not torch.equal(silu_fused, silu_wrong_split)
        outputs.update(silu_input=silu_input, silu_fused=silu_fused, silu_wrong_split=silu_wrong_split)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in outputs.items()}, str(a.output), metadata={
        "fixture": "SYNTHETIC; existing seeded tiny weights, NOT downloaded base weights",
        "source_bfl": "45e57c9fb06f9751cdced29a65c5d2171003be05",
        "source_hf": "7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941",
        "source_hashes": json.dumps(HASHES, sort_keys=True), "torch": torch.__version__,
        "verified_source_blobs": str(verified), "source_manifest_sha256": manifest_sha,
        "python": sys.version, "safetensors": importlib.metadata.version("safetensors"),
        "transformers": importlib.metadata.version("transformers"),
        "Pillow": importlib.metadata.version("Pillow"),
        "overlay_pins_sha256": hashlib.sha256(a.overlay_receipt.read_bytes()).hexdigest() if a.overlay_receipt else "none",
        "overlay_pins": a.overlay_receipt.read_text(encoding="utf-8") if a.overlay_receipt else "none",
        "idna": importlib.metadata.version("idna"),
        "native_policy": "retained selected parameters/activations; transient F32 opmath; fused conv+bias and SiLU low writes; CPU half emulation, CUDA unrun",
        "execution": "CPU PyTorch selected precision; not CUDA kernel evidence",
        "grid": "2x3; channel=c*4+ph*2+pw; F32 original BN normalization exactly once",
    })
    print(f"SYNTHETIC CPU oracle: {a.output}; {len(outputs)} tensors; F32/BF16/F16")


if __name__ == "__main__":
    main()
