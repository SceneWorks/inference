# Float-to-RGB8 quantization policy

SceneWorks inference converts decoded floating-point image and video samples to RGB8 by clamping to
the display range, scaling to `[0, 255]`, **rounding to nearest with midpoint ties to even**, and only
then casting to `u8`. This matches PyTorch/MLX round in the diffusers reference pipelines and avoids the
systematic one-LSB darkening caused by a direct truncating float-to-integer cast.

In shorthand, decoded `[-1, 1]` pixels use:

```text
rgb8 = round_ties_even(clamp(decoded, -1, 1) * 0.5 * 255 + 127.5) as u8
```

Equivalent `[0, 1] * 255` formulations follow the same round-before-cast rule. Exact half-integer
cases are intentionally nearest-even: `0.5 -> 0`, `1.5 -> 2`, `2.5 -> 2`, `3.5 -> 4`. In
particular, decoded zero maps to RGB8 `128`, not truncation's `127`.

## Audited runtime inventory

The `sc-12534` sweep classified every `Uint8`/`U8` cast and float-to-`u8` host conversion under
`crates/media/{mlx-gen,candle-gen}`. These decoded-image/video output sites implement this policy:

- Shared MLX: `mlx-gen/src/image.rs` (used by Anima, Boogu, Chroma, FLUX, FLUX.2, Krea, Lens,
  Qwen-Image, SANA, SD3, SeedVR2, Sensenova, and Z-Image).
- MLX family-local: `mlx-gen-sdxl/src/pipeline.rs`, `mlx-gen-lens/src/pipeline.rs`,
  `mlx-gen-ideogram/src/pipeline.rs`, `mlx-gen-svd/src/model.rs`, `mlx-gen-ltx/src/pipeline.rs`,
  `mlx-gen-mochi/src/pipeline.rs`, `mlx-gen-scail2/src/generate.rs`, and both z16/z48 conversion
  paths in `mlx-gen-wan/src/pipeline.rs`.
- Candle image families: `candle-gen-anima/src/pipeline.rs`, `candle-gen-chroma/src/pipeline.rs`,
  `candle-gen-flux/src/pipeline.rs`, `candle-gen-flux2/src/lib.rs`,
  `candle-gen-ideogram/src/pipeline.rs`, `candle-gen-kolors/src/common.rs`,
  `candle-gen-qwen-image/src/control_common.rs`, `candle-gen-sana/src/pipeline.rs`,
  `candle-gen-sd3/src/pipeline.rs`, both SDXL conversion paths in
  `candle-gen-sdxl/src/{pipeline,denoise}.rs`, and the Krea and Lens inference/training preview
  paths in `candle-gen-krea/src/{pipeline,training}.rs` and
  `candle-gen-lens/src/{lib,training,vae}.rs`.
- Candle video families: `candle-gen-ltx/src/pipeline.rs`, `candle-gen-mochi/src/pipeline.rs`,
  `candle-gen-seedvr2/src/pipeline.rs`, `candle-gen-svd/src/pipeline.rs`, and
  `candle-gen-wan/src/pipeline.rs`.
- Developer output utilities that emit real decoded RGB8 are covered too:
  `candle-gen-krea/examples/krea-control-infer.rs` and
  `candle-gen-qwen-image/src/comfyui_vae_validate.rs`.

The following `u8` conversions are deliberately outside this policy: attention and segmentation
masks, quantized weight codes/scales and packed metadata, UTF-8 golden metadata, already-quantized
RGB arrays being copied to host `Image` values, input/conditioning resize pixels, alpha compositing
and depth/face raster operations that already specify their own rounding, and synthetic test/example
image construction. A future decoded-output path should use the shared MLX helper when its layout
fits, or explicitly apply the backend's round operation immediately before its `u8` cast and add it
to this inventory.

Changing this policy moves any affected output pixel by at most one LSB. Pixel-exact generated-image
goldens produced by a formerly truncating path must therefore be regenerated intentionally; model,
latent, and floating-point parity goldens are unaffected.
