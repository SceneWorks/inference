"""Independent pinned Comfy VAE reference from fixed RGB, never native latents.

Reference tooling only. Import the unmodified pinned Comfy implementation and
guide function, load the installed published weights strictly, and compare the
native captures only AFTER independently producing and saving expected latents.
"""
import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
import re
from pathlib import Path
import subprocess
import sys
import time

import torch
from safetensors.torch import load_file, save_file, save

COMFY_PIN = "7a5dad695fe1cae25efcb2550530fb20ef68da3d"
UPSCALER_PIN = "36cb612ec3df30094eeb4fec66f1528925ebba73"
VAE_PIN = "6818f6c32d12b210915e44ad56a4228c2608f160"
# Declared before the real-input experiment. Both limits must hold, with no
# per-case adaptation. These predeclared bounds remain unchanged when the
# experimental VAE precision is corrected to published F32 weights.
TOLERANCES = {"max_abs": 0.08, "max_relative": 0.015}
CAPTURES = ("source.raw", "source.normalized", "guide.normalized")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_captures(expected, native):
    results = {}
    for name in CAPTURES:
        reference = expected[name].float()
        actual = native[name].float()
        if reference.shape != actual.shape:
            raise ValueError(f"{name}: shape mismatch")
        if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
            raise ValueError(f"{name}: nonfinite values")
        difference = (actual - reference).abs()
        maximum = float(difference.max())
        relative = maximum / max(float(reference.abs().max()), 1e-12)
        results[name] = {
            "shape": list(reference.shape), "max_abs": maximum,
            "max_relative": relative,
            "rms": float(difference.square().mean().sqrt()),
            "pass": maximum <= TOLERANCES["max_abs"]
            and relative <= TOLERANCES["max_relative"],
        }
    return results


def published_to_comfy(state):
    """Lossless published split projections -> pinned decoder's fused names.

    Q/K/V are interleaved PER HEAD because the pinned forward views its fused
    output as [heads, 3*head_dim], then chunks the last dimension. No checkpoint
    tensor is discarded. The published mask token is absent and unused at
    inference; its pinned non-learned buffer is explicitly zeroed by the caller.
    """
    result = dict(state)
    for key in list(result):
        if key.startswith("encoder.down_blocks."):
            converted = re.sub(r"encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.",
                               r"encoder.down.\1.block.\2.", key)
            converted = re.sub(r"encoder\.down_blocks\.(\d+)\.downsamplers\.0\.",
                               r"encoder.down.\1.downsample.", converted)
            converted = converted.replace(".conv_shortcut.", ".nin_shortcut.")
            if converted == key or converted in result:
                raise ValueError("ambiguous published encoder mapping " + key)
            result[converted] = result.pop(key)
    for block in range(36):
        prefix = f"decoder.transformer_blocks.{block}."
        for suffix in ("weight", "bias"):
            projections = [result.pop(prefix + f"attn.to_{part}.{suffix}")
                           for part in ("q", "k", "v")]
            tail = projections[0].shape[1:]
            result[prefix + f"attn.to_qkv.{suffix}"] = torch.stack(
                [p.reshape(32, 64, *tail) for p in projections], dim=1
            ).reshape(32 * 3 * 64, *tail)
            for old, new in (("attn.to_out.0", "attn.to_out"),
                             ("ff.net.0.proj", "ff.w1"),
                             ("ff.net.2", "ff.w2")):
                tensor = result.pop(prefix + old + "." + suffix)
                if old == "ff.net.0.proj":
                    # The official Diffusers converter swaps gate/value halves.
                    # Reverse that physical transform, not just the key rename.
                    value, gate = tensor.chunk(2, dim=0)
                    tensor = torch.cat([gate, value], dim=0).contiguous()
                result[prefix + new + "." + suffix] = tensor
    for suffix in ("weight", "bias"):
        result["decoder.x_embedder." + suffix] = result.pop("decoder.proj_in." + suffix)
    return result


def compare_saved_reference(reference_root, evidence_root, out):
    """Compare saved independent exports, binding RGB and both tensor hashes."""
    out.mkdir(parents=True, exist_ok=False)
    all_pass = True
    for case in ("h3", "other-model", "live-action"):
        metadata_path = reference_root / (case + ".json")
        metadata = json.loads(metadata_path.read_text())
        reference_path = reference_root / (case + ".safetensors")
        directory = evidence_root / case
        native_path = directory / "guided.intermediates.safetensors"
        if metadata["tolerances"] != TOLERANCES or metadata["comfy_commit"] != COMFY_PIN:
            raise ValueError("changed reference pin or tolerance")
        if sha256(reference_path) != metadata["reference_sha256"]:
            raise ValueError("changed independent reference tensors")
        if sha256(directory / "source.rgb") != metadata["rgb_sha256"]:
            raise ValueError("native and reference RGB inputs differ")
        expected = load_file(str(reference_path), device="cpu")
        native = load_file(str(native_path), device="cpu")
        comparison = compare_captures(expected, native)
        passed = all(value["pass"] for value in comparison.values())
        all_pass &= passed
        # Demonstrate every real source/guide assertion rejects a changed native
        # tensor, with expected exports and all other native captures unchanged.
        mutations = {}
        for name in CAPTURES:
            changed = {k: v.clone() for k, v in native.items()}
            changed[name].flatten()[7] += max(1., float(expected[name].abs().max()))
            errors = compare_captures(expected, changed)[name]
            mutations[name] = {
                "rejected": not errors["pass"],
                "reason": "numerical absolute/relative limits violated; RGB and reference hashes valid; native checksum is computed from current values, not checked against an expected native checksum",
                "errors": errors,
                "changed_tensor_payload_sha256": hashlib.sha256(save({name: changed[name].contiguous()})).hexdigest(),
            }
        if not all(value["rejected"] for value in mutations.values()):
            raise ValueError("real native capture mutation escaped assertion")
        report = {**metadata, "pass": passed, "captures": comparison,
                  "native_sha256": sha256(native_path),
                  "reference_manifest_sha256": sha256(metadata_path),
                  "comparison_script_sha256": sha256(Path(__file__)),
                  "native_capture_mutations_rejected": mutations}
        (out / (case + ".json")).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"case": case, "pass": passed, "captures": comparison}), flush=True)
    return all_pass


def main():
    parser = argparse.ArgumentParser()
    for name in ("comfy-root", "upstream-guide", "vae", "evidence-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--export-only", action="store_true", help="save independent references without reading native captures")
    parser.add_argument("--reference-root", type=Path, help="compare saved independent exports on CPU instead of running a model")
    parser.add_argument("--cases", nargs="+", choices=("h3", "other-model", "live-action"), default=("h3", "other-model", "live-action"))
    args = parser.parse_args()
    if args.reference_root is not None:
        if not compare_saved_reference(args.reference_root, args.evidence_root, args.out):
            raise SystemExit("independent real VAE source/guide comparison failed")
        return
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "GPU-e4b79931-7be6-f216-460a-f5405cfafffe":
        raise ValueError("assign the authorized GPU UUID explicitly")
    actual_pin = subprocess.check_output(
        ["git", "-C", str(args.comfy_root), "rev-parse", "HEAD"], text=True).strip()
    if actual_pin != COMFY_PIN:
        raise ValueError("unexpected Comfy reference revision")
    if subprocess.check_output(["git", "-C", str(args.comfy_root), "status", "--porcelain"], text=True).strip():
        raise ValueError("Comfy reference checkout has changes")
    args.out.mkdir(parents=True, exist_ok=False)
    # Persist the acceptance bounds BEFORE importing/loading/running any model.
    (args.out / "declared-tolerances.json").write_text(json.dumps({
        "captures": CAPTURES, "tolerances": TOLERANCES,
        "basis": "Fixed predeclared backend accumulation bound; both absolute and relative limits required; no case-dependent tolerance",
    }, indent=2) + "\n")
    sys.path.insert(0, str(args.comfy_root))
    module = importlib.import_module("comfy.ldm.minimax.vae")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Construct parameters on meta, then strictly assign the published tensors.
    # Published learned tensors and pinned normalization buffers are F32.
    with torch.device("meta"):
        vae = module.MiniMaxH3VideoVAE()
    files = sorted(args.vae.glob("*.safetensors"))
    if len(files) != 3 or args.vae.parent.name != VAE_PIN:
        raise ValueError("expected the three immutable H3 VAE shards")
    state = {}
    weight_hashes = {}
    for file in files:
        weight_hashes[file.name] = sha256(file)
        for name, tensor in load_file(str(file), device="cpu").items():
            if name in state:
                raise ValueError("duplicate VAE weight " + name)
            state[name] = tensor.to(device="cuda:0", dtype=torch.float32)
    published_count = len(state)
    state = published_to_comfy(state)
    state["latents_mean"] = torch.tensor(module.LATENTS_MEAN, device="cuda:0", dtype=torch.float32)
    state["latents_std"] = torch.tensor(module.LATENTS_STD, device="cuda:0", dtype=torch.float32)
    state["decoder.mask_token"] = torch.zeros((1, 1, 2048), device="cuda:0", dtype=torch.float32)
    vae.load_state_dict(state, strict=True, assign=True)
    del state
    # Nonpersistent buffers are not in state_dict and must be materialized.
    vae.pixel_mean = torch.tensor(module.IMAGENET_MEAN, device="cuda:0").view(1, 3, 1, 1, 1)
    vae.pixel_std = torch.tensor(module.IMAGENET_STD, device="cuda:0").view(1, 3, 1, 1, 1)
    for block in vae.decoder.transformer_blocks:
        block.attn.qk_norm_scale = torch.ones(64, device="cuda:0", dtype=torch.float32)
    vae.decoder.pos_embed = module.RotaryEmbeddingND(48, 100., n_dim=3).to("cuda:0")
    for name, tensor in list(vae.named_parameters()) + list(vae.named_buffers()):
        if tensor.is_meta:
            raise ValueError("unmaterialized reference tensor " + name)
    vae.eval()
    tree = ast.parse(args.upstream_guide.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "guide_pixels"]
    if len(nodes) != 1:
        raise ValueError("pinned guide_pixels definition missing")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(args.upstream_guide), "exec"), scope)
    source_hashes = {str(p.relative_to(args.comfy_root)): sha256(p) for p in (
        args.comfy_root / "comfy/ldm/minimax/vae.py", args.comfy_root / "comfy/ops.py",
        args.comfy_root / "comfy/quant_ops.py", args.comfy_root / "comfy/rmsnorm.py",
        args.comfy_root / "comfy/ldm/modules/attention.py")}
    source_hashes["upscale_guide.py"] = sha256(args.upstream_guide)
    all_pass = True
    for case in args.cases:
        directory = args.evidence_root / case
        rgb_path = directory / "source.rgb"
        rgb = rgb_path.read_bytes()
        if len(rgb) != 39 * 288 * 512 * 3:
            raise ValueError("unexpected fixed input size")
        pixels = torch.frombuffer(bytearray(rgb), dtype=torch.uint8).reshape(1, 39, 288, 512, 3)
        pixels = pixels.permute(0, 4, 1, 2, 3).float().div(255).to("cuda:0")
        expected = {}
        original_encode = vae.encode_temporal
        def capture_moments(x, device):
            moments = original_encode(x, device)
            expected["source.raw"] = moments.float().chunk(2, dim=1)[0].cpu()
            return moments
        vae.encode_temporal = capture_moments
        started = time.monotonic()
        with torch.inference_mode():
            source = vae.encode(pixels * 2 - 1)
            expected["source.normalized"] = source.float().cpu()
            vae.encode_temporal = original_encode
            decoded = vae.decode(source)[:, :, :39]
            expected["source.decoded.rgb"] = decoded.float().cpu()
            guide_pixels = scope["guide_pixels"](decoded[0].permute(1, 2, 3, 0), 1024, 576, 2)
            guide = guide_pixels.permute(3, 0, 1, 2).unsqueeze(0).to("cuda:0")
            expected["guide.pixels.rgb"] = guide.float().cpu()
            expected["guide.normalized"] = vae.encode(guide * 2 - 1).float().cpu()
        torch.cuda.synchronize()
        reference_path = args.out / (case + ".safetensors")
        save_file({k: v.contiguous() for k, v in expected.items()}, str(reference_path))
        # This is the first native-data read. Every expected tensor is independent.
        native_path = directory / "guided.intermediates.safetensors"
        comparison = {} if args.export_only else compare_captures(expected, load_file(str(native_path), device="cpu"))
        passed = all(value["pass"] for value in comparison.values())
        all_pass &= passed
        report = {"case": case, "pass": None if args.export_only else passed, "captures": comparison,
                  "tolerances": TOLERANCES, "comfy_commit": COMFY_PIN,
                  "upsampler_commit": UPSCALER_PIN, "vae_revision": VAE_PIN,
                  "reference": "Unmodified pinned real Comfy VAE, published F32 learned weights and pixel/latent normalization buffers, cuDNN/SDPA without autocast or TF32; native captures are comparison-only",
                  "torch": torch.__version__, "reference_seconds": time.monotonic() - started,
                  "published_tensor_count": published_count, "weight_sha256": weight_hashes,
                  "source_sha256": source_hashes, "rgb_sha256": sha256(rgb_path),
                  "native_sha256": None if args.export_only else sha256(native_path), "reference_sha256": sha256(reference_path),
                  "script_sha256": sha256(Path(__file__)),
                  "dependencies": {name: importlib.metadata.version(name) for name in ("comfy-kitchen", "comfy-aimdo")}}
        (args.out / (case + ".json")).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"case": case, "pass": passed, "captures": comparison}), flush=True)
        del pixels, source, decoded, guide_pixels, guide
    if not all_pass:
        raise SystemExit("independent real VAE source/guide comparison failed")


if __name__ == "__main__":
    main()
