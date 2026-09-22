"""Write the Qwen-Image 2.1 **reference / edit** goldens (sc-24110).

Reloads the tiny snapshot `dump_qwen21_pipeline.py` wrote (so these oracles and the component
oracles are the same bytes the Rust loader reads) and drives the frozen upstream
`QwenImage21Pipeline` down its condition-image branch, recording every stage the Rust port has to
reproduce:

  * the per-reference **preprocessing** — `calculate_dimensions` fit, the single LANCZOS resize,
    the Qwen3-VL `pixel_values` / `grid_thw`, and the RGBA `[-1, 1]` VAE input;
  * the **text conditioning** — `encode_prompt(image=…)`'s `prompt_embeds` and `image_pad_mask`,
    i.e. the full Qwen3-VL vision tower + DeepStack + interleaved-M-RoPE path;
  * the **reference latents** — `prepare_latents`' packed, mode-sampled, normalised condition
    latents, per reference and in order;
  * the **end-to-end latents** after a fixed number of Euler steps from fixed initial noise.

Cases (`output_resolution = 64`, the tiny processor's pixel budget; targets 64x64):

  * `ref1`      — one reference (the single-image "edit" call);
  * `ref2`      — two references (multi-reference composition);
  * `ref2_swap` — the same two references in the opposite order; the Rust ordering test asserts
                  this differs from `ref2`, which is what makes "order is semantic" testable;
  * `ref10`     — the documented **boundary**, ten references;
  * `annotated` — a reference with a drawn-in annotation (a filled rectangle standing for the
                  README's circle / paint stroke), which upstream consumes as an ordinary
                  reference: no mask argument exists;
  * `mask_ref`  — a photo plus a **separate binary mask image** passed as a second ordinary
                  reference, which is the only way upstream accepts one.

`use_kv_cache=False` on every run, as the other generators do.

Run with the pinned venv (CPU): `python3.12 tools/dump_qwen21_edit.py`
Output: `mlx-gen-qwen-image-2-1/tests/fixtures/qwen21_edit.safetensors`
"""

from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image as PILImage

from _qwen21_common import FIXTURE_DIR, Z_DIM, load_tiny_pipeline, save_safetensors

# Every word below is in the tiny WordLevel vocabulary (see `_qwen21_common.VOCAB_WORDS`).
PROMPT = "a red fox in the forest"
NEGATIVE = "blurry low quality photo"
OUTPUT_RESOLUTION = 64
TARGET = (64, 64)
STEPS = 2


def synthetic(seed: int, width: int, height: int) -> PILImage.Image:
    """A deterministic RGB8 picture with structure in both axes (so a swapped reference is
    genuinely a different input, and so the resize is not a no-op on a flat field)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack(
        [
            (x * 3 + y * 5 + seed * 17) % 256,
            (x * 7 + seed * 29) % 256,
            (y * 11 + seed * 41) % 256,
        ],
        axis=-1,
    ).astype(np.float64)
    noise = rng.integers(0, 24, size=base.shape)
    return PILImage.fromarray(((base + noise) % 256).astype(np.uint8), mode="RGB")


def annotated(seed: int, width: int, height: int) -> PILImage.Image:
    """`synthetic` with an annotation drawn **into** it — the README's circle / painted mark.
    Upstream has no mask argument, so this is how a local edit is specified."""
    array = np.array(synthetic(seed, width, height))
    y0, y1 = height // 4, height // 4 + max(2, height // 3)
    x0, x1 = width // 4, width // 4 + max(2, width // 3)
    array[y0:y1, x0:x1] = [255, 0, 0]
    array[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = np.array(synthetic(seed, width, height))[
        y0 + 1 : y1 - 1, x0 + 1 : x1 - 1
    ]
    return PILImage.fromarray(array, mode="RGB")


def binary_mask(width: int, height: int) -> PILImage.Image:
    """A hard black/white mask, carried as an ordinary extra reference image."""
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = 255
    return PILImage.fromarray(array, mode="RGB")


def preprocess(pipe, images):
    """Steps 1 of `QwenImage21Pipeline.__call__`: RGBA convert, per-image `calculate_dimensions`
    fit, one resize feeding both the processor copy and the VAE copy."""
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

    sizes, vision_inputs, vae_inputs = [], [], []
    for img in images:
        rgba = img.convert("RGBA") if img.mode != "RGBA" else img
        w, h = rgba.size
        iw, ih, _ = calculate_dimensions(OUTPUT_RESOLUTION * OUTPUT_RESOLUTION, w / h)
        sizes.append((iw, ih))
        vision_inputs.append(pipe.image_processor.resize(rgba, width=iw, height=ih))
        vae_inputs.append(pipe.image_processor.preprocess(rgba, width=iw, height=ih).unsqueeze(2))
    return sizes, vision_inputs, vae_inputs


def run_case(pipe, name, images, out, meta, negative=None, true_cfg=1.0, stages=True):
    width, height = TARGET
    sizes, vision_inputs, vae_inputs = preprocess(pipe, images)

    # Raw inputs, so the Rust test conditions on the very same pixels.
    for i, img in enumerate(images):
        out[f"{name}/source_{i}"] = torch.from_numpy(np.array(img.convert("RGB")))
    grids = []
    for i, img in enumerate(vision_inputs):
        processed = pipe.processor.image_processor(images=[img.convert("RGB")], return_tensors="pt")
        grids.append(processed["image_grid_thw"].to(torch.int32))
        # The per-stage preprocessing oracles are ~160 KB per reference, so they are recorded for
        # the small cases only; the ten-reference boundary case is about the joint layout, and its
        # inputs are the same `synthetic` pixels the small cases prove the preprocessing on.
        if stages:
            out[f"{name}/vae_input_{i}"] = vae_inputs[i].squeeze(2)
            out[f"{name}/pixel_values_{i}"] = processed["pixel_values"]
            out[f"{name}/grid_thw_{i}"] = processed["image_grid_thw"].to(torch.int32)

    # 2. Text conditioning through the Qwen3-VL vision tower + DeepStack + interleaved M-RoPE.
    prompt_embeds, prompt_embeds_mask, image_pad_mask = pipe.encode_prompt(
        image=vision_inputs, prompt=PROMPT, device=torch.device("cpu")
    )
    assert prompt_embeds_mask is None, "a single unpadded prompt carries no padding mask"
    out[f"{name}/prompt_embeds"] = prompt_embeds
    out[f"{name}/image_pad_mask"] = image_pad_mask.to(torch.int32)

    # 3. Reference latents, per reference and in order.
    latent_h, latent_w = height // 16, width // 16
    torch.manual_seed(7)
    init = torch.randn(1, latent_h * latent_w, Z_DIM)
    for i, vae_input in enumerate(vae_inputs):
        _, packed = pipe.prepare_latents(
            [vae_input], 1, Z_DIM, height, width, torch.float32, torch.device("cpu"), None,
            latents=init.clone(),
        )
        out[f"{name}/ref_latents_{i}"] = packed
    out[f"{name}/latents_init"] = init.clone()

    # 4. End to end from the same initial noise.
    kwargs = {}
    if negative is not None:
        kwargs["negative_prompt"] = negative
        kwargs["true_cfg_scale"] = true_cfg
    latents = pipe(
        prompt=PROMPT,
        image=list(images),
        width=width,
        height=height,
        num_inference_steps=STEPS,
        latents=init.clone(),
        output_type="latent",
        output_resolution=OUTPUT_RESOLUTION,
        use_kv_cache=False,
        **kwargs,
    ).images
    out[f"{name}/latents_final"] = latents

    meta[name] = {
        "prompt": PROMPT,
        "negative_prompt": negative,
        "true_cfg_scale": true_cfg,
        "width": width,
        "height": height,
        "steps": STEPS,
        "output_resolution": OUTPUT_RESOLUTION,
        "references": len(images),
        "source_sizes": [list(img.size) for img in images],
        "fitted_sizes": [list(s) for s in sizes],
        "vision_slots": [int(g[0].prod()) // 4 for g in grids],
    }
    print(f"{name}: {len(images)} refs, fitted {sizes}, text {tuple(prompt_embeds.shape)}")


def main() -> None:
    pipe = load_tiny_pipeline()
    pipe.set_progress_bar_config(disable=True)
    out, meta = {}, {}

    photo_a = synthetic(1, 48, 48)
    photo_b = synthetic(2, 80, 80)

    run_case(pipe, "ref1", [photo_a], out, meta)
    run_case(pipe, "ref2", [photo_a, photo_b], out, meta)
    run_case(pipe, "ref2_swap", [photo_b, photo_a], out, meta, stages=False)
    run_case(pipe, "ref2_cfg", [photo_a, photo_b], out, meta, negative=NEGATIVE, true_cfg=2.5, stages=False)
    run_case(pipe, "ref10", [synthetic(10 + i, 64, 64) for i in range(10)], out, meta, stages=False)
    run_case(pipe, "annotated", [annotated(3, 64, 64)], out, meta)
    run_case(pipe, "mask_ref", [photo_a, binary_mask(64, 64)], out, meta)

    # The port's own bounds, recorded so the Rust refusal tests cite this file rather than a
    # constant typed twice. Upstream's pipeline does not enforce a cap; the README documents ten.
    meta["_limits"] = {
        "max_reference_images": 10,
        "refused_reference_counts": [0, 11],
        "mask_conditioning": "refused: upstream exposes no mask tensor; pass the mask as an "
        "ordered reference and name it in the prompt",
    }
    # Metadata values are JSON so the Rust tests parse one stable format (safetensors
    # metadata is string-valued, and `str(dict)` would be a Python repr).
    save_safetensors(
        FIXTURE_DIR / "qwen21_edit.safetensors",
        out,
        {k: json.dumps(v, sort_keys=True) for k, v in meta.items()},
    )


if __name__ == "__main__":
    main()
