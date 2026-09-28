"""Write the Qwen-Image 2.1 **transparency / RGBA** goldens (sc-24111).

Reloads the tiny snapshot `dump_qwen21_pipeline.py` wrote (so these oracles are the same bytes the
Rust loader reads) and drives the frozen upstream `QwenImage21Pipeline` down the two transparency
paths the story ports, recording what the Rust side has to reproduce.

**There is no transparency flag upstream.** `QwenImage21Pipeline.__call__` takes no `rgba` /
`transparent` argument; the VAE is four-channel in and out, so the pipeline *always* decodes four
channels and `image_processor.postprocess(..., "pil")` *always* returns a PIL image in mode
`RGBA`. Whether a render is actually transparent is decided by the **prompt** (the README's
"Transparent Image Generation (RGBA)" example is a prompt, not a mode). This file therefore
records the same single decode in both of the forms the port emits:

  * `<case>/image_rgba`    — upstream's `output_type="pt"` tensor, `[4, H, W]` in `[0, 1]`
    (the float oracle S1/S2 already hold their VAE decode to);
  * `<case>/image_rgba_u8` — upstream's `output_type="pil"` **uint8** RGBA, `[H, W, 4]`, i.e.
    `(v * 255).round().astype(uint8)` of the above with **straight, un-premultiplied** alpha.
    This is the byte-exact oracle for the new `RgbaImage` output surface.

Cases:

  * `t2i_rgba`  — text-to-image with the README's transparent-image prompt form ("This is an RGBA
    image with transparency … the background is transparent"), the T2I half of the story;
  * `extract`   — an **RGBA reference in → RGBA out**: subject extraction / transparent-layer
    editing. The reference genuinely carries alpha (a soft-edged disc over transparency), which is
    what makes the two consumers distinguishable. Upstream splits them deliberately:

      - the **vision** copy is flattened over WHITE before the Qwen3-VL processor sees it
        (`_get_qwen_prompt_embeds`: "the checkpoint was trained with the alpha composited over
        white for the vision encoder" — `white.paste(img, mask=img.getchannel("A"))`), recorded
        here as `extract/vision_rgb_0` so the Rust port's flatten is checked against upstream's
        own bytes rather than against a re-derivation;
      - the **VAE** copy keeps all four channels (`image_processor.preprocess(img, …)`), recorded
        as `extract/vae_input_0` — a real alpha plane, not the constant `+1.0` an opaque
        reference produces.

  * `extract_opaque` — the SAME geometry with a fully opaque reference, so the Rust test can
    assert that the RGB reference route is exactly the `A = 255` special case of the RGBA one
    (upstream's `img.convert("RGBA")` widening) rather than a separate path.

`use_kv_cache=False` on every run, as the other generators do. Every prompt word is in the tiny
WordLevel vocabulary (`_qwen21_common.VOCAB_WORDS`), which is asserted below.

Run with the pinned venv (CPU): `python3.12 tools/dump_qwen21_rgba.py`
Output: `mlx-gen-qwen-image-2-1/tests/fixtures/qwen21_rgba.safetensors`
"""

from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image as PILImage

from _qwen21_common import FIXTURE_DIR, Z_DIM, load_tiny_pipeline, save_safetensors

# The README's transparent-image prompt form, restricted to the tiny vocabulary.
TRANSPARENT_PROMPT = (
    "This is an RGBA image with transparency . a red fox sticker . "
    "The image has alpha channel and the background is transparent"
)
# Subject extraction / transparent-layer editing, likewise in-vocabulary.
EXTRACT_PROMPT = "the red fox on a transparent background"

OUTPUT_RESOLUTION = 64
TARGET = (64, 64)
T2I_TARGET = (32, 32)
STEPS = 2


def _assert_in_vocab(pipe, prompt: str) -> None:
    """Every prompt word must be a real token, not `[UNK]` — otherwise the golden silently
    conditions on a different prompt than the Rust test believes it sent."""
    unk = pipe.processor.tokenizer.unk_token_id
    ids = pipe.processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    assert unk not in ids, f"prompt {prompt!r} hits [UNK]; add the word to VOCAB_WORDS"


def transparent_subject(seed: int, width: int, height: int) -> PILImage.Image:
    """A deterministic RGBA picture: a structured subject over a **transparent** field, with a
    soft (anti-aliased) edge.

    The soft edge matters. A hard binary matte survives any resampling order, so it could not tell
    "resize RGBA together" apart from "resize RGB, then resize alpha"; a gradient edge cannot. The
    colour is carried underneath the transparent region too (it is *not* zeroed), which is exactly
    why a consumer must composite rather than drop the fourth byte.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    rgb = np.stack(
        [
            (x * 3 + y * 5 + seed * 17) % 256,
            (x * 7 + seed * 29) % 256,
            (y * 11 + seed * 41) % 256,
        ],
        axis=-1,
    ).astype(np.float64)
    rgb = (rgb + rng.integers(0, 24, size=rgb.shape)) % 256

    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    radius = min(width, height) * 0.36
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    # Linear ramp over a 3-px band at the disc boundary: opaque inside, transparent outside.
    alpha = np.clip((radius + 1.5 - dist) / 3.0, 0.0, 1.0) * 255.0

    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1).astype(np.uint8)
    return PILImage.fromarray(rgba, mode="RGBA")


def opaque_subject(seed: int, width: int, height: int) -> PILImage.Image:
    """`transparent_subject`'s colours with a fully opaque alpha — upstream's `convert("RGBA")` on
    an ordinary photograph."""
    array = np.array(transparent_subject(seed, width, height))
    array[..., 3] = 255
    return PILImage.fromarray(array, mode="RGBA")


def preprocess(pipe, images):
    """Step 1 of `QwenImage21Pipeline.__call__`: RGBA convert, per-image `calculate_dimensions`
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


def flatten_over_white(rgba: PILImage.Image) -> PILImage.Image:
    """Upstream's vision-tower flatten, verbatim from `_get_qwen_prompt_embeds`."""
    white = PILImage.new("RGB", rgba.size, (255, 255, 255))
    white.paste(rgba, mask=rgba.getchannel("A"))
    return white


def record_images(pipe, name, out, **call_kwargs) -> None:
    """Run the pipeline twice from the same inputs, once for each output form, and record both.

    `output_type="pt"` and `output_type="pil"` differ only in the final quantisation
    (`postprocess` shares everything before it), and the denoise is fully determined by the
    supplied `latents`, so the two runs decode the same image.
    """
    float_image = pipe(output_type="pt", **call_kwargs).images[0]
    pil_image = pipe(output_type="pil", **call_kwargs).images[0]
    assert pil_image.mode == "RGBA", (
        f"{name}: upstream returned mode {pil_image.mode}; the 2.1 VAE is four-channel and "
        "postprocess must produce RGBA"
    )
    out[f"{name}/image_rgba"] = float_image  # [4, H, W] in [0, 1]
    out[f"{name}/image_rgba_u8"] = torch.from_numpy(np.array(pil_image))  # [H, W, 4] uint8
    alpha = np.array(pil_image)[..., 3]
    print(
        f"{name}: {pil_image.mode} {pil_image.size} "
        f"alpha min={alpha.min()} max={alpha.max()} mean={alpha.mean():.1f}"
    )


def run_t2i(pipe, out, meta) -> None:
    width, height = T2I_TARGET
    latent_h, latent_w = height // 16, width // 16
    torch.manual_seed(11)
    init = torch.randn(1, latent_h * latent_w, Z_DIM)
    out["t2i_rgba/latents_init"] = init.clone()
    record_images(
        pipe,
        "t2i_rgba",
        out,
        prompt=TRANSPARENT_PROMPT,
        width=width,
        height=height,
        num_inference_steps=STEPS,
        latents=init.clone(),
        use_kv_cache=False,
    )
    meta["t2i_rgba"] = {
        "prompt": TRANSPARENT_PROMPT,
        "width": width,
        "height": height,
        "steps": STEPS,
        "references": 0,
    }


def run_reference(pipe, name, image, out, meta) -> None:
    width, height = TARGET
    sizes, vision_inputs, vae_inputs = preprocess(pipe, [image])

    # The raw reference, exactly as a request carries it: HWC RGBA8.
    out[f"{name}/source_rgba_0"] = torch.from_numpy(np.array(image))
    # Upstream's WHITE-composited vision copy of the RESIZED reference — the oracle for the
    # port's flatten. Recorded as bytes so the Rust side cannot "prove" its composite against
    # its own re-derivation of the rule.
    whitened = flatten_over_white(vision_inputs[0])
    out[f"{name}/vision_rgb_0"] = torch.from_numpy(np.array(whitened))
    # ...and the four-channel VAE copy, alpha intact.
    out[f"{name}/vae_input_0"] = vae_inputs[0].squeeze(2)

    processed = pipe.processor.image_processor(images=[whitened], return_tensors="pt")
    out[f"{name}/pixel_values_0"] = processed["pixel_values"]
    out[f"{name}/grid_thw_0"] = processed["image_grid_thw"].to(torch.int32)

    prompt_embeds, prompt_embeds_mask, image_pad_mask = pipe.encode_prompt(
        image=vision_inputs, prompt=EXTRACT_PROMPT, device=torch.device("cpu")
    )
    assert prompt_embeds_mask is None, "a single unpadded prompt carries no padding mask"
    out[f"{name}/prompt_embeds"] = prompt_embeds
    out[f"{name}/image_pad_mask"] = image_pad_mask.to(torch.int32)

    latent_h, latent_w = height // 16, width // 16
    torch.manual_seed(7)
    init = torch.randn(1, latent_h * latent_w, Z_DIM)
    _, packed = pipe.prepare_latents(
        [vae_inputs[0]], 1, Z_DIM, height, width, torch.float32, torch.device("cpu"), None,
        latents=init.clone(),
    )
    out[f"{name}/ref_latents_0"] = packed
    out[f"{name}/latents_init"] = init.clone()

    # The end-to-end latents, from the same initial noise (the S3 reference gate's oracle).
    out[f"{name}/latents_final"] = pipe(
        prompt=EXTRACT_PROMPT,
        image=[image],
        width=width,
        height=height,
        num_inference_steps=STEPS,
        latents=init.clone(),
        output_type="latent",
        output_resolution=OUTPUT_RESOLUTION,
        use_kv_cache=False,
    ).images

    record_images(
        pipe,
        name,
        out,
        prompt=EXTRACT_PROMPT,
        image=[image],
        width=width,
        height=height,
        num_inference_steps=STEPS,
        latents=init.clone(),
        output_resolution=OUTPUT_RESOLUTION,
        use_kv_cache=False,
    )

    alpha = np.array(image)[..., 3]
    meta[name] = {
        "prompt": EXTRACT_PROMPT,
        "width": width,
        "height": height,
        "steps": STEPS,
        "output_resolution": OUTPUT_RESOLUTION,
        "references": 1,
        "source_sizes": [list(image.size)],
        "fitted_sizes": [list(s) for s in sizes],
        "source_alpha_min": int(alpha.min()),
        "source_alpha_max": int(alpha.max()),
    }


def main() -> None:
    pipe = load_tiny_pipeline()
    pipe.set_progress_bar_config(disable=True)
    _assert_in_vocab(pipe, TRANSPARENT_PROMPT)
    _assert_in_vocab(pipe, EXTRACT_PROMPT)

    out, meta = {}, {}
    run_t2i(pipe, out, meta)
    run_reference(pipe, "extract", transparent_subject(5, 48, 48), out, meta)
    run_reference(pipe, "extract_opaque", opaque_subject(5, 48, 48), out, meta)

    # The contract the Rust refusal tests cite, rather than a constant typed twice.
    meta["_semantics"] = {
        "upstream_transparency_flag": None,
        "alpha": "straight (un-premultiplied); postprocess clamps each channel to [0,1] "
        "independently, then (v*255).round() to uint8",
        "reference_vision_copy": "composited over white before the Qwen3-VL processor "
        "(_get_qwen_prompt_embeds)",
        "reference_vae_copy": "all four channels, VaeImageProcessor.preprocess 2x-1",
        "how_transparency_is_requested": "by prompt; see t2i_rgba/prompt",
    }
    save_safetensors(
        FIXTURE_DIR / "qwen21_rgba.safetensors",
        out,
        {k: json.dumps(v, sort_keys=True) for k, v in meta.items()},
    )


if __name__ == "__main__":
    main()
