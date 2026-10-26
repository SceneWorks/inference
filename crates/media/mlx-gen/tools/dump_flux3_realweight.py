"""Real-weight upstream reference for the native FLUX 3 port (sc-26234): the frozen source over the
PINNED distilled package on CPU, at a 64x64 output (a 4x4 token grid) so every component golden is
small enough to commit (`tests/fixtures/flux3_real_golden_64.safetensors`) and the Mac lane reads
it next to the weights.

What it records (upstream layouts; the Rust `real_weights::` cases convert at the boundary):

* `text/{input_ids, mask, positions, tokens}` — `TextEncoder.encode(prompt)` with the REAL
  processor/tokenizer and the bf16 tower (`tokens` is the `[L, 20480]` conditioning, bf16 values).
* `dit/{x, image_ids, t, t_ref, velocity, velocity_bf16}` — one flow-map forward on 16 image
  tokens: `velocity` from the f32 DiT (bf16 weights cast to f32, f32 inputs), `velocity_bf16` from
  the bf16 DiT under `torch.autocast("cpu", bfloat16)` (the CPU twin of the release's policy).
  Metadata `dit_bf16_distance` = max |velocity_bf16 - velocity| / peak: upstream's OWN bf16
  distance on this input, the yardstick for the native bf16 gate.
* `ae/{tokens_rand, decoded}` — `AutoEncoder.decode` of seeded N(0, 1) tokens at 4x4, f32.
* `e2e/{noise, schedule, tokens, tokens_bf16, image, rgb8}` — `denoise_flow_map` over the 16-step
  release schedule from injected bf16-valued noise, f32 and bf16-autocast, the f32 decode and its
  RGB8. Metadata `e2e_bf16_distance` likewise.

Inputs: `FLUX3_SRC` (frozen source), `FLUX3_PACKAGE` (the pinned package snapshot: `config.json`,
`model.safetensors`, `ae.safetensors`, `text_encoder/`), optional `FLUX3_REAL_OUT` (default: the
committed fixture path). Runs on CPU; ~35 GB of RAM (f32 DiT 22 GB + bf16 DiT 11 GB + tower).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _flux3_common import FIXTURE_DIR, FLUX3_COMMIT, FLUX3_PACKAGE_REVISION, import_upstream, save_safetensors  # noqa: E402

import_upstream()

import numpy as np  # noqa: E402
import torch  # noqa: E402

PROMPT = "a red fox sitting in fresh snow at golden hour, soft light, photograph"
SIZE = 64
STEPS = 16
ALPHA = 6.93
PINNED = {
    "config.json": "c4fc638a9618e8ddc25f013c009be4d2aa9bd631d89dcf45149a377dd472c67d",
}


def package_root() -> Path:
    root = Path(os.environ.get("FLUX3_PACKAGE", ""))
    if not root.is_dir():
        raise SystemExit("FLUX3_PACKAGE is required: the pinned black-forest-labs/flux-3-image-klein snapshot")
    for rel, want in PINNED.items():
        got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"FLUX3_PACKAGE/{rel}: sha256 {got} != pinned {want}")
    return root


def main() -> None:
    package = package_root()
    out_path = Path(os.environ.get("FLUX3_REAL_OUT", FIXTURE_DIR / "flux3_real_golden_64.safetensors"))
    torch.set_num_threads(max(4, os.cpu_count() or 4))
    dit_module = importlib.import_module("flux3.dit")
    checkpoint_module = importlib.import_module("flux3.checkpoint")
    pipeline_module = importlib.import_module("flux3.pipeline")
    sampling_module = importlib.import_module("flux3.sampling")
    autoencoder_module = importlib.import_module("flux3.autoencoder")
    text_encoder_module = importlib.import_module("flux3.text_encoder")
    tensors: dict[str, torch.Tensor] = {}
    meta = {
        "upstream_commit": FLUX3_COMMIT,
        "package_revision": FLUX3_PACKAGE_REVISION,
        "prompt": PROMPT,
        "size": SIZE,
        "steps": STEPS,
        "alpha": ALPHA,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }

    # --- text encoder (bf16, the real processor) ---
    t0 = time.time()
    enc = text_encoder_module.TextEncoder(str(package / "text_encoder"), device="cpu")
    tok = enc.processor.tokenizer
    text = enc.processor.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
    with torch.no_grad():
        tokens, ids = enc.encode(PROMPT)
    length = int(tokens.shape[0])
    inputs = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
    mask = inputs["attention_mask"][0]
    positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
    tensors["text/input_ids"] = inputs["input_ids"][0].to(torch.int32)
    tensors["text/mask"] = mask.to(torch.int32)
    tensors["text/positions"] = positions.to(torch.int32)
    tensors["text/tokens"] = tokens.float()
    tensors["text/ids"] = ids.float()
    meta["text_seconds"] = f"{time.time() - t0:.1f}"
    print(f"text: {int(mask.sum())} real / {length} window, {time.time() - t0:.1f}s", flush=True)
    del enc

    # --- the DiT, f32 and bf16 ---
    h, w = SIZE // 16, SIZE // 16
    g = torch.Generator().manual_seed(26234)
    x = torch.randn((h * w, dit_module.IN_CHANNELS["image"]), generator=g).to(torch.bfloat16).float()
    image_ids = pipeline_module.scaled_ids(pipeline_module.latent_ids(h, w, "cpu"))
    schedule = sampling_module.timesteps(STEPS, ALPHA)
    t, t_ref = float(schedule[3]), float(schedule[4])
    text_pair = (tokens, ids)

    def run_dit(dtype: torch.dtype, autocast: bool):
        t0 = time.time()
        with torch.device("meta"):
            dit = dit_module.DiT(flow_map=True)
        dit = checkpoint_module.load_checkpoint(dit, str(package), "cpu", dtype)
        print(f"dit {dtype} loaded in {time.time() - t0:.1f}s", flush=True)
        packed = pipeline_module.pack([[text_pair, None, (x.clone(), image_ids)]])
        if not autocast:
            packed.data = {m: v.float() for m, v in packed.data.items()}
            packed.timesteps = {m: v.float() for m, v in packed.timesteps.items()}
        packed.timesteps["image"].fill_(t)
        t0 = time.time()
        if autocast:
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                velocity = dit(packed, timesteps_ref=torch.tensor([t_ref]))
        else:
            with torch.no_grad():
                velocity = dit(packed, timesteps_ref=torch.tensor([t_ref]))
        print(f"dit {dtype} forward in {time.time() - t0:.1f}s", flush=True)
        # the 16-step flow-map loop from the same noise
        packed = pipeline_module.pack([[text_pair, None, (x.clone(), image_ids)]])
        if not autocast:
            packed.data = {m: v.float() for m, v in packed.data.items()}
            packed.timesteps = {m: v.float() for m, v in packed.timesteps.items()}
        t0 = time.time()
        if autocast:
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                e2e = sampling_module.denoise_flow_map(dit, packed, schedule)
        else:
            with torch.no_grad():
                e2e = sampling_module.denoise_flow_map(dit, packed, schedule)
        print(f"dit {dtype} {STEPS}-step loop in {time.time() - t0:.1f}s", flush=True)
        del dit
        return velocity.float(), e2e.float()

    velocity, e2e_tokens = run_dit(torch.float32, autocast=False)
    velocity_bf16, e2e_tokens_bf16 = run_dit(torch.bfloat16, autocast=True)
    tensors["dit/x"] = x
    tensors["dit/image_ids"] = image_ids
    tensors["dit/t"] = torch.tensor([t])
    tensors["dit/t_ref"] = torch.tensor([t_ref])
    tensors["dit/velocity"] = velocity
    tensors["dit/velocity_bf16"] = velocity_bf16
    peak = velocity.abs().max().item()
    meta["dit_bf16_distance"] = f"{(velocity_bf16 - velocity).abs().max().item() / max(peak, 1.0):.6e}"
    tensors["e2e/noise"] = x.clone()
    tensors["e2e/schedule"] = schedule
    tensors["e2e/tokens"] = e2e_tokens
    tensors["e2e/tokens_bf16"] = e2e_tokens_bf16
    peak = e2e_tokens.abs().max().item()
    meta["e2e_bf16_distance"] = f"{(e2e_tokens_bf16 - e2e_tokens).abs().max().item() / max(peak, 1.0):.6e}"
    print(f"bf16 distances: dit {meta['dit_bf16_distance']} e2e {meta['e2e_bf16_distance']}", flush=True)

    # --- the autoencoder, f32 ---
    ae = checkpoint_module.load_checkpoint(autoencoder_module.AutoEncoder(), str(package / "ae.safetensors"), "cpu", torch.float32)
    lat_ids = autoencoder_module.latent_ids(h, w, "cpu")
    tokens_rand = torch.randn((h * w, autoencoder_module.LATENT_CHANNELS), generator=g)
    with torch.no_grad():
        decoded_rand = ae.decode(tokens_rand, lat_ids)
        image = ae.decode(e2e_tokens, lat_ids)
    tensors["ae/tokens_rand"] = tokens_rand
    tensors["ae/decoded"] = decoded_rand
    tensors["e2e/image"] = image
    tensors["e2e/rgb8"] = torch.from_numpy(np.array(pipeline_module.to_pil(image))).to(torch.uint8)
    save_safetensors(out_path, tensors, meta)
    (out_path.with_suffix(".json")).write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
