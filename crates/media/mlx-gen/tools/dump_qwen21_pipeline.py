"""Write the tiny Qwen-Image 2.1 snapshot + end-to-end pipeline goldens (sc-24108).

Builds the seeded tiny `QwenImage21Pipeline` (see `_qwen21_common.py`), saves it with
`save_pretrained` in the exact `Qwen/Qwen-Image-2.1` snapshot layout under
`mlx-gen-qwen-image-2-1/tests/fixtures/tiny-snapshot/`, then runs the frozen upstream pipeline on
fixed packed latents and records every per-step latent plus the decoded RGBA image:

  * `t2i/*`   — 32x32, 3 steps, no guidance (upstream default `true_cfg_scale = 1.0`);
  * `cfg/*`   — 32x32, 3 steps, negative prompt + `true_cfg_scale = 2.5` (true CFG branch);
  * `wide/*`  — 64x32, 2 steps: a non-square target (2x4 latent, two vision slots) so the RoPE
                centring and the multi-slot expansion are covered end to end.

`use_kv_cache=False` on every run: the Rust port evaluates the full block-causal joint sequence at
every step (the exact `QwenImage21AttnProcessor` prefill path); upstream documents the cached and
uncached paths as equally valid but not bit-identical.

Run with the pinned venv (CPU): `python3.12 tools/dump_qwen21_pipeline.py`
Output: `tests/fixtures/tiny-snapshot/**` and `tests/fixtures/qwen21_pipeline.safetensors`.
"""

from __future__ import annotations

import shutil

import torch

from _qwen21_common import (
    FIXTURE_DIR,
    SNAPSHOT_DIR,
    Z_DIM,
    build_tiny_pipeline,
    load_tiny_pipeline,
    save_safetensors,
)

PROMPT = "a red fox in the forest"
NEGATIVE = "blurry low quality photo"


def run_case(pipe, name, width, height, steps, latents, out, meta, **kwargs):
    captured = {}

    def on_step_end(pipeline, i, t, callback_kwargs):
        captured[i] = callback_kwargs["latents"].detach().clone()
        return {}

    image = pipe(
        prompt=PROMPT,
        width=width,
        height=height,
        num_inference_steps=steps,
        latents=latents.clone(),
        output_type="pt",
        use_kv_cache=False,
        callback_on_step_end=on_step_end,
        **kwargs,
    ).images[0]
    out[f"{name}/latents_init"] = latents.clone()
    for i in range(steps):
        out[f"{name}/latents_after_step_{i}"] = captured[i]
    out[f"{name}/image_rgba"] = image  # [4, H, W] in [0, 1]
    meta[f"{name}/width"] = width
    meta[f"{name}/height"] = height
    meta[f"{name}/steps"] = steps
    print(f"{name}: image {tuple(image.shape)} min={image.min():.4f} max={image.max():.4f}")


def main() -> None:
    if SNAPSHOT_DIR.exists():
        shutil.rmtree(SNAPSHOT_DIR)
    pipe = build_tiny_pipeline()
    pipe.save_pretrained(str(SNAPSHOT_DIR), safe_serialization=True)
    # `save_pretrained` writes the processor under `processor/`; the real snapshot ships the same.
    assert (SNAPSHOT_DIR / "processor" / "tokenizer.json").is_file()
    assert (SNAPSHOT_DIR / "transformer" / "diffusion_pytorch_model.safetensors").is_file()
    assert (SNAPSHOT_DIR / "vae" / "diffusion_pytorch_model.safetensors").is_file()
    assert (SNAPSHOT_DIR / "text_encoder" / "model.safetensors").is_file()

    # Re-read the snapshot so the goldens come from the bytes the Rust loader reads.
    pipe = load_tiny_pipeline()
    out, meta = {}, {"prompt": PROMPT, "negative_prompt": NEGATIVE}

    g = torch.Generator("cpu").manual_seed(7)
    latents_32 = torch.randn((1, 4, Z_DIM), generator=g)  # packed: 2x2 latent = 4 tokens
    run_case(pipe, "t2i", 32, 32, 3, latents_32, out, meta)
    run_case(
        pipe, "cfg", 32, 32, 3, latents_32, out, meta, negative_prompt=NEGATIVE, true_cfg_scale=2.5
    )
    meta["cfg/true_cfg_scale"] = 2.5
    latents_64x32 = torch.randn((1, 8, Z_DIM), generator=g)  # 2 rows x 4 cols
    run_case(pipe, "wide", 64, 32, 2, latents_64x32, out, meta)

    save_safetensors(FIXTURE_DIR / "qwen21_pipeline.safetensors", out, meta)


if __name__ == "__main__":
    main()
