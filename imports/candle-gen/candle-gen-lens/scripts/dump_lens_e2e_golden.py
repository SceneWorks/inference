#!/usr/bin/env python
"""Dump an end-to-end golden for the full Lens-Turbo T2I pipeline (candle-gen sc-5115).

Constructs the authoritative vendor ``LensPipeline`` (SceneWorks ``_vendor/lens``) from the cached
``microsoft/Lens-Turbo`` snapshot and runs one **4-step turbo** generation (guidance 1.0) with an
**injected** initial latent — so the candle port feeds byte-identical starting noise and the only
divergence is bf16 candle-vs-torch op-order (the e2e is cross-build, gated on structural cosine, per
the FLUX-hyper / cross-backend precedent).

Production dtypes: encoder + transformer **bf16** (MXFP4 experts dequantize to dense bf16), VAE
**f32**. Resolution **512×512** (latent 32×32 = 1024 image tokens) keeps the run tractable while
exercising the whole wiring + the turbo schedule. Runs on CUDA.

Golden contents:
  - ``input_ids``     [1, L] int64 — the positive harmony-rendered ids (the candle e2e re-tokenizes
                      with ``date_utf8`` and asserts it reproduces these, validating the tokenizer
                      inside the e2e);
  - ``init_latents``  [1, h·w, 128] f32 — the injected starting noise;
  - ``final_latents`` [1, h·w, 128] f32 — the torch denoise output (pre-VAE);
  - ``image``         [1, 3, H, W] f32 in [-1,1] — the decoded image (full e2e incl. the VAE shim);
  - ``date_utf8``     [n] uint8 — the harmony-preamble date used (candle has no metadata loader).

Run (from the worktree root) with the transformers-5.8 lens-venv:
  & "C:\\Users\\Michael\\AppData\\Roaming\\SceneWorks\\python\\lens-venv\\Scripts\\python.exe" \\
      candle-gen-lens\\scripts\\dump_lens_e2e_golden.py [out_dir]

Default out_dir: .scratch/lens-e2e-goldens/  (not committed — large + regenerable).
"""
from __future__ import annotations

import datetime
import glob
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, Mxfp4Config

VENDOR_PARENT_CANDIDATES = [
    r"D:\repos\SceneWorks\apps\worker\scene_worker",
    r"D:\repos\SceneWorks\apps\desktop\python-src\scene_worker",
]

PROMPT = "a red fox sitting in a snowy forest at sunrise, photorealistic"
NEGATIVE = ""
HEIGHT = WIDTH = 512
NUM_STEPS = 4
GUIDANCE = 1.0
SEED = 0
DEVICE = "cuda"


def find_snapshot() -> str:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    snaps = sorted(p for p in glob.glob(str(hub / "models--microsoft--Lens-Turbo" / "snapshots" / "*")) if os.path.isdir(p))
    if not snaps:
        sys.exit("no microsoft/Lens-Turbo snapshot found")
    return snaps[-1]


@torch.no_grad()
def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".scratch/lens-e2e-goldens")
    out_dir.mkdir(parents=True, exist_ok=True)

    vendor_parent = next((p for p in VENDOR_PARENT_CANDIDATES if os.path.isdir(os.path.join(p, "_vendor", "lens"))), None)
    if vendor_parent is None:
        sys.exit("no _vendor/lens package found")
    sys.path.insert(0, os.path.join(vendor_parent, "_vendor"))

    snap = find_snapshot()
    print(f"snapshot: {snap}", flush=True)

    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from lens import LensPipeline, LensTransformer2DModel
    from lens.text_encoder import LensGptOssEncoder

    tok = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))

    te_cfg = AutoConfig.from_pretrained(os.path.join(snap, "text_encoder"))
    te_cfg._attn_implementation = "eager"
    te_cfg._experts_implementation = "eager"
    print("loading text_encoder (MXFP4 → bf16)…", flush=True)
    text_encoder = LensGptOssEncoder.from_pretrained(
        os.path.join(snap, "text_encoder"),
        config=te_cfg,
        quantization_config=Mxfp4Config(dequantize=True),
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    ).eval()

    print("loading transformer (bf16)…", flush=True)
    transformer = LensTransformer2DModel.from_pretrained(
        os.path.join(snap, "transformer"), torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()

    print("loading vae (f32)…", flush=True)
    vae = AutoencoderKLFlux2.from_pretrained(
        os.path.join(snap, "vae"), torch_dtype=torch.float32
    ).to(DEVICE).eval()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(snap, "scheduler"))
    pipe = LensPipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tok, transformer=transformer)

    latent_h, latent_w = HEIGHT // pipe.vae_scale_factor, WIDTH // pipe.vae_scale_factor
    seq_len = latent_h * latent_w
    g = torch.Generator(device="cpu").manual_seed(SEED)
    init = torch.randn((1, seq_len, 128), generator=g, dtype=torch.float32)

    device = torch.device(DEVICE)
    input_ids, _ = pipe._build_chat_inputs([PROMPT], 512, device)
    current_date = datetime.date.today().isoformat()
    print(f"input_ids L={input_ids.shape[1]}  date={current_date}", flush=True)

    print(f"denoising {NUM_STEPS} steps @ {WIDTH}x{HEIGHT}…", flush=True)
    final_latents = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        latents=init.to(device=device, dtype=transformer.dtype),
        output_type="latent",
    ).images  # [1, seq, 128]

    print("decoding…", flush=True)
    decoded = pipe._decode(final_latents, latent_h, latent_w)  # [1, 3, H, W] in [-1,1]

    date_bytes = list(current_date.encode("utf-8"))
    tensors = {
        "input_ids": input_ids.to(torch.int64).cpu(),
        "init_latents": init.to(torch.float32).cpu(),
        "final_latents": final_latents.to(torch.float32).cpu(),
        "image": decoded.clamp(-1.0, 1.0).to(torch.float32).cpu().contiguous(),
        "date_utf8": torch.tensor(date_bytes, dtype=torch.uint8),
    }
    dst = out_dir / "lens_e2e_golden.safetensors"
    save_file(tensors, str(dst))
    print(f"wrote {dst}\n  final_latents {tuple(final_latents.shape)}  image {tuple(decoded.shape)}  date={current_date}")


if __name__ == "__main__":
    main()
