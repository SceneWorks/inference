#!/usr/bin/env python3
"""Regenerate the Anima STAGE-7 end-to-end parity golden (sc-10524, epic 10512) for all three variants.

    Regenerate with (from the repo root, using the prepared reference venv; MPS recommended):
        ANIMA_CONVERT_DIR=<dir-with-fetched-convert-scripts> \
        <venv>/bin/python mlx-gen-anima/tests/fixtures/gen_anima_stage7_golden.py

WHAT / WHY
----------
Compares the MLX port's end-to-end output against the diffusers 0.39.0 reference (Apache-2.0) for
`anima_base`, `anima_aesthetic`, `anima_turbo`. To avoid chaos-limited cross-backend drift, the Rust
test injects the **identical initial latent** (a deterministic Gaussian, reproduced bit-for-bit here
and in Rust via the same LCG + Box-Muller) and both sides run **deterministic Euler** over the
**identical sigma schedule** (`linspace(1,1/N,N)` + static shift 3.0). With the same start, schedule,
and solver, residual drift is Metal-vs-MPS float error, not chaos.

Runs fp32 both sides (the Rust `AnimaPipeline::denoise_from_latent` takes a `dtype`). Commits only the
computed golden JSON (final-latent + decoded-image summary stats + a fixed subsample; <200 KB total) —
no weights, no upstream source. PNG pairs are written to $ANIMA_STAGE7_OUT for a visual check.

CHECKPOINTS
-----------
`Anima-Base-v1.0-Diffusers` is **base-only**. `anima_aesthetic` / `anima_turbo` are converted from the
single-file `circlestone-labs/Anima` checkpoints **in memory** (no disk write — the box has little free
space) using the upstream `convert_anima_to_diffusers.py` / `convert_cosmos_to_diffusers.py` functions.

⚠️ FINDING (reported on sc-10524): the upstream convert script is **base-only** — both
`split_anima_transformer_checkpoint` (`adapter_prefix = "net.llm_adapter."`) and
`convert_transformer` (`PREFIX_KEY = "net."`) hardcode the `net.` root. aesthetic/turbo root at
`model.diffusion_model`, so this generator first **normalizes their root prefix to `net.`** before
handing the state dict to the upstream functions. Without that, the convert fails on those two variants.
The convert scripts are fetched (not vendored) from huggingface/diffusers @ v0.39.0.
"""
import importlib.util
import io
import json
import math
import os
import sys
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "e2e_golden.json"
PNG_OUT = Path(os.environ.get("ANIMA_STAGE7_OUT", "/tmp/anima_sc10524_stage7"))
HF_HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

CONVERT_URLS = {
    "convert_anima_to_diffusers": "https://raw.githubusercontent.com/huggingface/diffusers/v0.39.0/scripts/convert_anima_to_diffusers.py",
    "convert_cosmos_to_diffusers": "https://raw.githubusercontent.com/huggingface/diffusers/v0.39.0/scripts/convert_cosmos_to_diffusers.py",
}

# Story defaults per variant: (steps, guidance, uses_cfg).
VARIANTS = {
    "anima_base": ("anima-base-v1.0.safetensors", 30, 4.5, True),
    "anima_aesthetic": ("anima-aesthetic-v1.0.safetensors", 30, 4.5, True),
    "anima_turbo": ("anima-turbo-v1.0.safetensors", 10, 1.0, False),
}
PROMPT = "an anime girl with long silver hair and blue eyes, detailed illustration, masterpiece"
NEGATIVE = ""
W = H = 1024
INIT_SEED = 7  # shared init latent across all variants


# ---------------- provenance ----------------
def diffusers_base_snapshot() -> Path:
    base = HF_HUB / "models--circlestone-labs--Anima-Base-v1.0-Diffusers" / "snapshots"
    return next(d for d in sorted(base.iterdir()) if (d / "transformer").is_dir())


def single_file_snapshot() -> Path:
    base = HF_HUB / "models--circlestone-labs--Anima" / "snapshots"
    return next(d for d in sorted(base.iterdir()) if (d / "split_files" / "diffusion_models").is_dir())


# ---------------- reproducible Gaussian init (bit-identical to the Rust test) ----------------
def gauss_fill(n: int, seed: int) -> np.ndarray:
    """LCG uniforms in (0,1) -> Box-Muller Gaussian. Reproduced bit-for-bit in Rust (same recurrence,
    f64 transcendentals -> f32). u = (s+0.5)/2147483648 with s the 31-bit LCG state."""
    out = np.empty(n, dtype=np.float64)
    s = seed & 0x7FFFFFFF
    i = 0
    two_pi = 2.0 * math.pi
    while i < n:
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        u1 = (s + 0.5) / 2147483648.0
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        u2 = (s + 0.5) / 2147483648.0
        r = math.sqrt(-2.0 * math.log(u1))
        out[i] = r * math.cos(two_pi * u2)
        i += 1
        if i < n:
            out[i] = r * math.sin(two_pi * u2)
            i += 1
    return out.astype(np.float32)


# ---------------- sigma schedule (mirrors anima_sigmas / stage 5) ----------------
def anima_sigmas(steps: int) -> list[float]:
    n = max(steps, 1)
    shift = 3.0
    sig = []
    for i in range(n):
        s = 1.0 if n == 1 else 1.0 + i * (1.0 / n - 1.0) / (n - 1)
        sig.append(float(np.float32(shift * s / (1.0 + (shift - 1.0) * s))))
    sig.append(0.0)
    return sig


# ---------------- torchvision-free identity resize shim for the DiT padding mask ----------------
def _install_transforms_shim():
    import diffusers.models.transformers.transformer_cosmos as tc

    class _F:
        @staticmethod
        def resize(img, size, interpolation=None):
            assert list(img.shape[-2:]) == list(size)
            return img

    class _Mode:
        NEAREST = "nearest"

    class _T:
        functional = _F
        InterpolationMode = _Mode

    tc.transforms = _T


# ---------------- fetch upstream convert functions (not vendored) ----------------
def load_convert_module(name: str):
    d = os.environ.get("ANIMA_CONVERT_DIR")
    if d and (Path(d) / f"{name}.py").is_file():
        src = (Path(d) / f"{name}.py").read_text()
    else:
        with urllib.request.urlopen(CONVERT_URLS[name], timeout=60) as r:
            src = r.read().decode("utf-8")
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    exec(compile(src, f"<{name}@v0.39.0>", "exec"), mod.__dict__)
    return mod


# ---------------- transformer + conditioner per variant ----------------
def load_transformer_and_conditioner(variant: str, base_snap: Path, sf_snap: Path, anima_conv):
    from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
    from diffusers.models.condition_embedders.condition_embedder_anima import AnimaTextConditioner

    if variant == "anima_base":
        dit = CosmosTransformer3DModel.from_pretrained(base_snap / "transformer").float().eval().to(DEVICE)
        cond = AnimaTextConditioner.from_pretrained(base_snap / "text_conditioner").float().eval().to(DEVICE)
        return dit, cond

    # aesthetic / turbo: convert single-file in memory, normalizing the root prefix -> net.
    dit_file = sf_snap / "split_files" / "diffusion_models" / VARIANTS[variant][0]
    raw = load_file(str(dit_file), device="cpu")
    anchor = ".x_embedder.proj.1.weight"
    prefix = next(k[: -len(anchor)] for k in raw if k.endswith(anchor))
    if prefix != "net":
        raw = {("net." + k[len(prefix) + 1 :]): v for k, v in raw.items()}
    tsd, csd = anima_conv.split_anima_transformer_checkpoint(raw)
    with redirect_stdout(io.StringIO()):  # the convert prints every key mapping
        dit = anima_conv.convert_transformer(
            "Cosmos-2.0-Diffusion-2B-Text2Image", state_dict=tsd, weights_only=True
        )
        cond = anima_conv.convert_text_conditioner(csd)
    return dit.float().eval().to(DEVICE), cond.float().eval().to(DEVICE)


# ---------------- encode / denoise / decode (fp32) ----------------
def encode(prompt, text_encoder, conditioner, qwen_tok, t5_tok):
    q = qwen_tok(prompt, padding="longest", max_length=512, truncation=True, return_tensors="pt")
    ids, mask = q.input_ids.to(DEVICE), q.attention_mask.to(DEVICE)
    if ids.shape[-1] == 0:
        ids, mask = ids.new_zeros((1, 1)), mask.new_zeros((1, 1))
    with torch.no_grad():
        src = text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        src = src * mask.to(src.dtype).unsqueeze(-1)
        t5 = t5_tok(prompt, padding="longest", max_length=512, truncation=True, return_tensors="pt").input_ids.to(DEVICE)
        return conditioner(source_hidden_states=src, target_input_ids=t5)


def denoise(dit, cond, uncond, init, steps, guidance):
    sigmas = anima_sigmas(steps)
    x = init.clone()
    pm = torch.zeros(1, 1, init.shape[-2], init.shape[-1], device=DEVICE)
    for i in range(steps):
        s, sn = sigmas[i], sigmas[i + 1]
        ts = torch.tensor([s], device=DEVICE)
        with torch.no_grad():
            vc = dit(hidden_states=x, timestep=ts, encoder_hidden_states=cond, padding_mask=pm).sample
            if uncond is not None:
                vu = dit(hidden_states=x, timestep=ts, encoder_hidden_states=uncond, padding_mask=pm).sample
                v = vu + guidance * (vc - vu)
            else:
                v = vc
        x = x + (sn - s) * v
    return x


def decode_to_uint8(vae, latent):
    lm = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latent)
    ls = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latent)
    z = latent * ls + lm  # == MLX QwenVae.decode's baked de-norm (latent*std + mean)
    with torch.no_grad():
        img = vae.decode(z, return_dict=False)[0][:, :, 0]  # [1,3,H,W] in [-1,1]
    img = ((img.clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return img[0].permute(1, 2, 0).cpu().numpy()  # HWC uint8


# ---------------- summarize (tiny golden) ----------------
def summarize(arr, n=48):
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    total = int(flat.size)
    idx = np.unique(np.linspace(0, total - 1, min(n, total)).astype(np.int64))
    return {
        "shape": list(np.asarray(arr).shape),
        "count": total,
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "l2": float(np.sqrt((flat * flat).sum())),
        "sample_indices": [int(i) for i in idx],
        "sample_values": [float(flat[i]) for i in idx],
    }


def img_summary(rgb, n=48):
    d = summarize(rgb.astype(np.float64), n)
    d["per_channel_mean"] = [float(rgb[..., c].mean()) for c in range(3)]
    d["per_channel_std"] = [float(rgb[..., c].std()) for c in range(3)]
    return d


def main():
    import diffusers  # noqa: F401
    import transformers
    from transformers import AutoModel, AutoTokenizer

    _install_transforms_shim()
    load_convert_module("convert_cosmos_to_diffusers")  # anima convert imports this by name
    anima_conv = load_convert_module("convert_anima_to_diffusers")

    base_snap = diffusers_base_snapshot()
    sf_snap = single_file_snapshot()
    PNG_OUT.mkdir(parents=True, exist_ok=True)

    # Shared across variants: Qwen3 text encoder + Qwen-Image VAE + tokenizers (base cached diffusers).
    text_encoder = AutoModel.from_pretrained(base_snap / "text_encoder").float().eval().to(DEVICE)
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(base_snap / "vae").float().eval().to(DEVICE)
    qwen_tok = AutoTokenizer.from_pretrained(base_snap / "tokenizer")
    t5_tok = AutoTokenizer.from_pretrained(base_snap / "t5_tokenizer")

    lat_shape = [1, 16, 1, H // 8, W // 8]
    init_np = gauss_fill(int(np.prod(lat_shape)), INIT_SEED).reshape(lat_shape)
    init = torch.from_numpy(init_np).to(DEVICE)

    results = {}
    for variant, (_, steps, guidance, uses_cfg) in VARIANTS.items():
        print(f"[{variant}] steps={steps} guidance={guidance} cfg={uses_cfg} device={DEVICE}", flush=True)
        dit, cond_model = load_transformer_and_conditioner(variant, base_snap, sf_snap, anima_conv)
        cond = encode(PROMPT, text_encoder, cond_model, qwen_tok, t5_tok)
        uncond = encode(NEGATIVE, text_encoder, cond_model, qwen_tok, t5_tok) if uses_cfg else None
        latent = denoise(dit, cond, uncond, init, steps, guidance)
        if DEVICE == "mps":
            torch.mps.synchronize()
        rgb = decode_to_uint8(vae, latent)
        # save PNG for a visual check
        try:
            from PIL import Image as PILImage

            PILImage.fromarray(rgb).save(PNG_OUT / f"{variant}_diffusers.png")
        except Exception as e:  # noqa: BLE001
            print(f"  (PNG save skipped: {e})", flush=True)
        results[variant] = {
            "steps": steps,
            "guidance": guidance,
            "uses_cfg": uses_cfg,
            "final_latent": summarize(latent.float().cpu().numpy()),
            "image": img_summary(rgb),
        }
        print(f"  latent std={results[variant]['final_latent']['std']:.4f} "
              f"img per-ch mean={results[variant]['image']['per_channel_mean']}", flush=True)
        del dit, cond_model
        if DEVICE == "mps":
            torch.mps.empty_cache()

    doc = {
        "meta": {
            "story": "sc-10524",
            "epic": "10512",
            "reference": "diffusers 0.39.0 Anima components (CosmosTransformer3DModel + AnimaTextConditioner "
            "+ Qwen3Model + AutoencoderKLQwenImage) in a manual fp32 deterministic-Euler loop.",
            "reference_packages": {
                "diffusers": __import__("diffusers").__version__,
                "transformers": transformers.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": sys.version.split()[0],
            },
            "device": DEVICE,
            "diffusers_repo": "circlestone-labs/Anima-Base-v1.0-Diffusers",
            "diffusers_snapshot_sha": base_snap.name,
            "single_file_repo": "circlestone-labs/Anima",
            "single_file_snapshot_sha": sf_snap.name,
            "convert_scripts": CONVERT_URLS,
            "aesthetic_turbo_conversion": "in-memory; root prefix normalized model.diffusion_model.->net. "
            "(upstream convert is base-only: hardcodes net.llm_adapter. and PREFIX_KEY='net.').",
            "regen_command": "ANIMA_CONVERT_DIR=<dir> <venv>/bin/python "
            "mlx-gen-anima/tests/fixtures/gen_anima_stage7_golden.py",
            "generator": "mlx-gen-anima/tests/fixtures/gen_anima_stage7_golden.py",
            "prompt": PROMPT,
            "negative": NEGATIVE,
            "width": W,
            "height": H,
            "init": {"kind": "gaussian_boxmuller_lcg", "seed": INIT_SEED, "shape": lat_shape},
            "sampler": "euler (deterministic flow-match), x += (sigma_next-sigma)*v; timestep=raw sigma",
            "dtype": "fp32 both sides (Rust denoise_from_latent(dtype=Float32)); the MLX Qwen3 encode is "
            "bf16-locked, but fp32 reference is empirically the closer target for MLX-bf16 than a torch-bf16 "
            "encode (two different bf16 roundings diverge more than fp32-truth-vs-bf16). Trajectory in fp32.",
            "note": "Parity = no quality regression. Same injected init + Euler + schedule -> residual is "
            "Metal-vs-MPS float error, not chaos.",
        },
        "variants": results,
    }
    OUT_JSON.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT_JSON} ({OUT_JSON.stat().st_size} bytes); PNGs in {PNG_OUT}")


if __name__ == "__main__":
    main()
