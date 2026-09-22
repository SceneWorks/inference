# Frozen upstream — Qwen-Image 2.1 (sc-24108)

Every numeric leaf in this crate mirrors one of the pinned sources below; the constants in
`src/lib.rs` (`UPSTREAM_*`) carry the same values and `tests/registration.rs` asserts they agree
with this file.

| What | Where | Revision |
| --- | --- | --- |
| Weights, `config.json`s, tokenizer, scheduler config, licence | `Qwen/Qwen-Image-2.1` (Hugging Face) | `790c92633540aa0cb11d9abf19eb46d861714758` (2026-09-21) |
| README presets, defaults, RGBA prompt form, prompt-rewrite tooling | `QwenLM/Qwen-Image-2.1` (GitHub) | `fb7ae1d1f9611cd91524d03c53c5246b36ac8577` (2026-09-20) |
| `QwenImage21Pipeline`, `QwenImage21Transformer2DModel`, `AutoencoderKLQwenImage21`, `FlowMatchEulerDiscreteScheduler` | `huggingface/diffusers` (`src/diffusers/pipelines/qwenimage21/`, `models/transformers/transformer_qwenimage21.py`, `models/autoencoders/autoencoder_kl_qwenimage21.py`, `schedulers/scheduling_flow_match_euler_discrete.py`) | `8b3c707ebd3ec4881f4190cf42931da07eaf3b65` (2026-09-22, main; day-0 support landed in PR #14804) |
| Qwen3-VL language tower (`Qwen3VLForConditionalGeneration`, `model.language_model.*`) | `transformers` 5.17.0 (the version the fixtures were dumped with) | — |

Licence: **Qwen Research License Agreement** (research / evaluation only) — see `NOTICE`.

## Snapshot layout the loader expects

```
<root>/model_index.json
<root>/processor/tokenizer.json            (tokenizer_config.json, chat_template.jinja, …)
<root>/scheduler/scheduler_config.json
<root>/text_encoder/config.json + model-0000N-of-00004.safetensors
<root>/transformer/config.json + diffusion_pytorch_model-0000N-of-00002.safetensors
<root>/vae/config.json + diffusion_pytorch_model.safetensors
```

Config keys read: `transformer/config.json` (`in_channels`, `out_channels`, `num_layers`,
`attention_head_dim`, `num_attention_heads`, `context_in_dim`, `mlp_ratio`, `axes_dims_rope`,
`eps`, `causal_condition`, `patch_size` must be 1); `text_encoder/config.json` → `text_config`
(`vocab_size`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`,
`head_dim`, `intermediate_size`, `rms_norm_eps`, `rope_theta`); `vae/config.json` (`base_dim`,
`decoder_base_dim`, `z_dim`, `dim_mult`, `num_res_blocks`, `temperal_downsample`, `in_channels`,
`out_channels`, `latents_mean`, `latents_std`, `scale_factor_spatial`, `is_residual` must be true,
`patch_size` must be null); `scheduler/scheduler_config.json` (`base_image_seq_len`,
`max_image_seq_len`, `base_shift`, `max_shift`, `shift_terminal`, `num_train_timesteps`,
`use_dynamic_shifting` must be true, `time_shift_type` must be `exponential`).

## Production geometry (frozen values)

* DiT: 32 single-stream layers, 32 heads × 128 (`inner_dim` 4096), `context_in_dim` 4096,
  `mlp_ratio` 3, RoPE axes `[16, 56, 56]` at θ = 10000, `eps` 1e-6, `causal_condition` true,
  64 latent channels in and out, unpatched.
* Text tower: Qwen3-VL-8B language model — 36 layers, hidden 4096, GQA 32/8 × 128, SwiGLU
  12288, RMSNorm 1e-6, RoPE θ = 5e6 (mRoPE collapses to 1-D for text-only prompts). Conditioning
  is the last layer's output **before** the final norm, minus the 14 system-prefix tokens.
* VAE: RGBA in/out, `z_dim` 64, `dim_mult [1, 2, 4, 8, 8]` (16× spatial), encoder width 96,
  decoder width 144, 2 residual blocks per stage, `temperal_downsample [false, true, true, true]`.
* Scheduler: `linspace(1, 1/N, N)`, exponential shift with `mu = calculate_shift(tokens, 256,
  8192, 0.5, 0.9)`, stretched to a 0.02 terminal, trailing 0. Default 40 steps, no guidance
  (`true_cfg_scale` 1.0).
* Presets (`width × height`): 1:1 2048×2048 (default), 4:3 2400×1792, 3:4 1792×2400,
  3:2 2528×1696, 2:3 1696×2528, 16:9 2752×1536, 9:16 1536×2752. Any size that is a multiple of
  32 px per side is accepted (one Qwen3-VL vision slot per 2×2 latent group).

## Fixtures

`tests/fixtures/` is produced by `tools/dump_qwen21_*.py` (shared setup in
`tools/_qwen21_common.py`) from the diffusers/transformers classes at the revisions above on
miniature seeded configs, with the tiny snapshot written by `save_pretrained` in the exact layout
above. Each Rust parity test names the tolerance it holds the port to.

## Reference conditioning and local editing (sc-24110)

Everything below is read off the pinned `QwenImage21Pipeline.__call__` /
`_get_qwen_prompt_embeds` / `prepare_latents` and
`QwenImage21Transformer2DModel.forward` / `build_token_metadata` / `QwenImage21Rope.forward`,
plus the pinned GitHub README. It is stated here because the shape of the feature is not
obvious from the class names.

### There is no edit pipeline, and no mask tensor

Upstream ships **one** pipeline. `QwenImage21Pipeline.__call__` takes an optional
`image: PipelineImageInput | None`; text-to-image is that argument being `None`. "Editing",
"multi-reference composition" and "local editing" are all the same call with one or more
condition images. There is no `QwenImage21EditPipeline`, no `strength`, no `mask_image`, no
latent blending and **no inpaint**: nothing in the pipeline preserves source pixels outside a
region, and the target latents are drawn from pure noise exactly as in text-to-image.

The README's "specify local edits via circles, painted annotations, or separate masks" describes
*what the checkpoint was trained to read inside its condition images*, not extra pipeline inputs.
The pinned README and the pinned pipeline agree: the only image input is the ordered `image` list.
Concretely:

* **Annotated reference** — the caller draws the circle / paint stroke **into the reference image
  itself** and describes the intent in the prompt ("remove the watch inside the red circle"). The
  annotated image is an ordinary condition image; the marks reach the model both through the
  Qwen3-VL vision tower and through the VAE latents.
* **Separate mask** — the mask is an ordinary **additional condition image** in the same ordered
  list, referred to by the prompt ("use the second image as the mask"). It is not a tensor with
  its own argument, it is not composited, and the model is under no obligation to respect it
  pixel-exactly.
* **"Identity preservation"** upstream means the checkpoint's own training behaviour (portrait and
  product fidelity across references). It is **not** a mechanism: there is no pinning, no
  masking and no latent copy. Nothing in the code path guarantees any pixel of a reference
  survives into the output.

This port therefore implements exactly the ordered-reference contract and refuses
`Conditioning::Mask` with a typed error that names the workaround, rather than inventing an
inpaint semantic upstream does not have.

### Up to ten references, and the order is meaningful

The README caps composition at **10 reference images**. The pipeline itself does not enforce a cap;
this port does (`MAX_REFERENCE_IMAGES = 10`), because past that the joint sequence is outside
anything upstream documents or trained.

Order is semantic, not incidental, in three separate ways:

1. The prompt template numbers the images — `<image1>`, `<image2>`, … — so a prompt that says
   "put the hat from image 3 on the person in image 1" is resolved by position.
2. The target geometry, when the caller does not pin `width`/`height`, is derived from the
   **last** image's aspect ratio (`image[-1].size`).
3. Each reference occupies its own RoPE frame position in sequence order, and attention is
   **block-causal**, so a reference can only be attended to by the text and references that follow
   it. Swapping two references is a different request and produces a different image.

### Preprocessing (one resize feeds both consumers)

Per condition image, in list order:

1. Converted to `RGBA` if it is not already (`img.convert("RGBA")`).
2. `calculate_dimensions(output_resolution², w/h)` → `w' = round(sqrt(A·r)/32)·32`,
   `h' = round((w'/r)/32)·32` with `output_resolution = 1024`. This is a *per-image* resize target
   derived from that image's own aspect ratio, not from the output size.
3. **One** resize to `(w', h')` feeds both consumers:
   * `image_processor.resize(...)` → the PIL image handed to the **Qwen3-VL processor** (RGBA is
     flattened over **white** for the vision tower only — `_get_qwen_prompt_embeds` composites an
     RGBA condition onto a white background before the processor sees it);
   * `image_processor.preprocess(...)` → the `[-1, 1]` NCHW tensor, `unsqueeze(2)` for the VAE's
     temporal axis, handed to the **VAE** with all four channels.
4. VAE encode uses `sample_mode="argmax"` — the posterior **mode**, not a sample — then the same
   `(z − latents_mean) / latents_std` normalisation the target latents use.
5. The encoded reference is packed unpatched (`[1, (h'/16)·(w'/16), 64]`) and the packed references
   are concatenated **in list order** and prepended to the target noise:
   `latent_model_input = cat([*reference_latents, latents], dim=1)`.

The processor's own `smart_resize` runs on top of step 2 with `patch_size·merge_size = 32`, so for
an image already sized to a multiple of 32 within the `[256², 4096²]` pixel budget it is a no-op —
which is what makes the vision grid and the VAE grid line up (see below). A reference small enough
to fall under `min_pixels` **is** enlarged by `smart_resize`, and the two grids then disagree; this
port rejects that case with a typed error rather than silently mis-binding the blocks.

### The text encoder reads the references (the Qwen3-VL vision tower is required)

The condition images go through the **text encoder as well as** the VAE. `_get_qwen_prompt_embeds`
builds the image-conditioned template

```
<|im_start|>system\n{SYSTEM}<|im_end|>\n
<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>[ <imageN><|vision_start|><|image_pad|><|vision_end|>]*{prompt}<|im_end|>\n
<|im_start|>assistant\n
```

(note the single leading space before `<image2>` onwards) and calls
`processor(text=..., images=[...])`, so `pixel_values` and `image_grid_thw` reach
`Qwen3VLForConditionalGeneration.forward`. That means the full VLM path runs:

* `Qwen2VLImageProcessorFast` geometry — `patch_size 16`, `merge_size 2`, `temporal_patch_size 2`,
  `mean = std = 0.5`, `min_pixels 65536`, `max_pixels 16777216`, bicubic;
* the 27-layer Qwen3-VL ViT (`vision_config`: hidden 1152, 16 heads, MLP 4304,
  `num_position_embeddings 2304` bilinearly resampled, 2-D rotary, 2×2 patch merger to
  `out_hidden_size 4096`);
* **DeepStack** — the ViT taps layers `[8, 16, 24]`, and those merged feature sets are added to the
  visual-token rows after decoder layers 0, 1 and 2 of the language tower;
* **interleaved M-RoPE** over the language tower (`mrope_section [24, 20, 20]`), which for a
  text-only prompt collapses to the plain 1-D RoPE the text-to-image path already used — that
  collapse is why the T2I path is bit-unchanged by this story.

The hidden state taken is still the last decoder layer's output **before** the final RMSNorm, minus
the system-prefix tokens.

**What the vision tower does and does not reach.** The DiT overwrites every `<|image_pad|>` row of
the joint sequence with the VAE latents (`joint_hidden_states[:, image_pad_mask] = hidden_states`),
so the ViT features never reach the DiT directly. They reach it *indirectly and materially*: the
language tower's attention lets every text token read the image rows, so the text conditioning the
DiT does consume is a function of the reference pixels. Skipping the tower would therefore produce
a structurally valid but wrongly-conditioned render, not a degraded one.

### The joint sequence

`img_mask` is the `<|image_pad|>` mask over the **VLM** sequence, with `target_tokens / 4` extra
`True` slots appended for the target image. Each `True` slot stands for a **2×2 group of latent
tokens**, so the DiT expands those positions four-fold
(`repeat_interleave(img_mask, where(img_mask, 4, 1))`) and drops the packed latents into them. The
resulting joint layout is

```
[text …] [ref 1 latents] [text …] [ref 2 latents] … [text …] [target latents]
```

i.e. each reference's latents sit **at the placeholder position the VLM reserved for it**, in the
middle of the text, and the target's latents are appended last. `img_shapes` is
`[(1, h'/16, w'/16) per reference in order, (1, H/16, W/16) for the target]`, and
`build_token_metadata` checks that `sum(prod(shape))` equals the number of expanded image
positions. This is the identity `(h'/32)·(w'/32) merged patches × 4 = (h'/16)·(w'/16)` latent
tokens — the reason the two resizes have to agree.

Block structure follows from `img_shapes`, **not** from runs of `True`: two adjacent references
with no text between them stay two blocks, so they never attend to each other bidirectionally.
Within the joint sequence:

* attention is `(q_idx >= kv_idx) or same_image_block` — causal overall, bidirectional inside each
  image block;
* RoPE gives each image block a frame position frozen at the text cursor and a zero-centred
  `(h, w)` grid, then advances the cursor by `max(h, w)`;
* under `causal_condition`, text **and reference** tokens modulate from the extra `t = 0` row; only
  the target tokens read the sampled timestep;
* the velocity is sliced to the last `target_tokens` rows.

`JointLayout` (`src/transformer.rs`) already models all of this; this story supplies the segments.

### What this port does with all of that

* `Conditioning::Reference { image, strength }` and `Conditioning::MultiReference { images }` are
  both accepted and flattened, **in request order**, into one ordered reference list of 1–10
  images. `strength` has no upstream meaning on this route and is refused unless it is `1.0`.
* `Conditioning::Mask { image }` is refused (typed `Unsupported`) naming the upstream fact and the
  workaround: pass the mask as an ordered reference and name it in the prompt.
* With references present, `width`/`height` still come from the request (SceneWorks always sends
  them); upstream's "derive from `image[-1]`" fallback is reproduced by
  `reference_derived_size`, which callers may use to fill them.
* Zero references on a request that carries a conditioning list, eleven or more references, an
  empty or zero-dimension image, and a reference whose `smart_resize` grid disagrees with its VAE
  grid are all typed, actionable refusals.

## Native transparency: RGBA output, subject extraction, transparent-layer editing (sc-24111)

### There is no transparency flag

`QwenImage21Pipeline.__call__` takes **no** `rgba` / `transparent` / `mode` argument. The 2.1 VAE
is four-channel in and out (`vae/config.json` `out_channels: 4`), so the pipeline *always* decodes
four channels:

```python
image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]   # [B, 4, H, W]
image = self.image_processor.postprocess(image, output_type=output_type)
```

and `postprocess(..., "pil")` therefore *always* returns a PIL image in mode `RGBA`
(`numpy_to_pil` → `Image.fromarray` on a 4-channel array). Verified against the pinned revision on
the committed tiny snapshot: every case in `tools/dump_qwen21_rgba.py` asserts `mode == "RGBA"`.

**Whether a render is actually transparent is decided by the prompt.** The model card's
"Transparent Image Generation (RGBA)" section is a *prompt recipe*, not a mode:

> "This is an RGBA image with transparency. A cute cartoon dragon sticker. The image has alpha
> channel and the background is transparent."

There is **no separate matting model**, no alpha post-processor and no segmentation step anywhere
in the pipeline. The alpha channel is whatever the VAE decoded — so there are **no guarantees
about it beyond that**. An ordinary opaque prompt yields a near-opaque alpha, not an exactly-255
one; a transparency prompt yields a soft matte the model chose. Nothing validates, thresholds or
cleans it up, and a caller must not assume `A ∈ {0, 255}`.

**Subject extraction** is likewise a prompt over the ordinary reference route ("extract the
subject onto a transparent background" — this port's fixture uses the in-vocabulary
"the red fox on a transparent background"), and **transparent-layer editing** is the ordinary
reference route with a reference that happens to carry alpha. Neither is a distinct API, a
distinct pipeline class or a distinct code path upstream.

### The alpha convention: straight, per-channel clamp

`VaeImageProcessor.postprocess` is, for every channel *independently*:

```python
image = (image * 0.5 + 0.5).clamp(0, 1)        # denormalize
images = (images * 255).round().astype("uint8")  # numpy_to_pil
```

So the emitted alpha is **straight (un-premultiplied)**, clamped to `[0, 1]` by the same clamp as
the colour, and rounded half-away-from-zero at 8 bits. Nothing premultiplies, and `A = 0` does
**not** imply `RGB = 0`: a fully transparent pixel still carries whatever colour the decoder
painted there, which is why a consumer that flattens must composite rather than drop the fourth
byte.

This port emits exactly that as `gen_core::RgbaImage` when the request sets
`output_channels: OutputChannels::Rgba` (gated by `Capabilities::supports_alpha_output`, which
this engine sets and every other provider leaves `false`). The default, `OutputChannels::Rgb`,
composites the same decode over white and emits `Image` — byte-for-byte the pre-sc-24111
behaviour. There is **one** decode either way; the request field selects only what happens to the
alpha the decoder already produced.

### RGBA reference inputs: the VAE sees the alpha, the vision tower sees it over white

Upstream converts **every** condition image to RGBA up front and then feeds the two consumers
*differently*:

```python
# __call__, step 1 — one resize, both consumers
if hasattr(img, "mode") and img.mode != "RGBA":
    img = img.convert("RGBA")
input_images.append(self.image_processor.resize(img, width=iw, height=ih))          # vision
vae_images.append(self.image_processor.preprocess(img, width=iw, height=ih).unsqueeze(2))  # VAE

# _get_qwen_prompt_embeds — the vision copy only
if img.mode == "RGBA":
    # "The checkpoint was trained with the alpha composited over white for the vision encoder."
    white = PILImage.new("RGB", img.size, (255, 255, 255))
    white.paste(img, mask=img.getchannel("A"))
    img = white
```

* the **VAE** encode receives all four channels, `2x − 1` normalised (`preprocess`), so a
  transparent layer's alpha is real signal in the condition latents;
* the **Qwen3-VL vision tower** receives the reference composited over **white**.

PIL's `paste` with an `"L"` mask is exactly `round(rgb·a + 255·(1 − a))` with `a = A/255` —
verified exhaustively over all 256×256 (colour, alpha) pairs against Pillow, and implemented as
`gen_core::RgbaImage::to_rgb_over_white`.

An ordinary RGB reference is the `A = 255` special case of this path (that is what
`img.convert("RGBA")` produces), so the white composite is the identity and the VAE alpha plane is
the constant `+1.0` — which is why the RGB reference route is unchanged by this story.

**Sending a flattened RGB reference is a different request from sending the transparent layer.**
The flattened copy hands the VAE white pixels where the layer is transparent; the transparent
layer hands it the alpha. This port therefore carries the two on distinct conditioning variants
(`Conditioning::Reference` and `Conditioning::ReferenceRgba`), both flattened into the one ordered
reference list, freely interleavable, with the same 1..=10 bound and the same "no strength" rule.

### The reference resize runs in PREMULTIPLIED space

`PIL.Image.resize` does **not** resample an RGBA image's four bands straight. It special-cases
`LA`/`RGBA` for every filter but `NEAREST`:

```python
if self.mode in ["LA", "RGBA"] and resample != Resampling.NEAREST:
    im = self.convert({"LA": "La", "RGBA": "RGBa"}[self.mode])
    im = im.resize(size, resample, box)
    return im.convert(self.mode)
```

so upstream's condition-image fit is **premultiply → LANCZOS-resample the four premultiplied bands
(with PIL's `clip8` between the horizontal and vertical passes) → un-premultiply**, not a
straight four-band resample. On a soft matte edge the difference is large, not cosmetic: measured
on this crate's `extract` fixture, resampling straight gives `pixel_values` `max|Δ| = 4.4e-1`
against a `2.1e-5` bound and reference latents `max|Δ| = 1.7e-1` against `2.5e-2`; reproducing the
premultiplied resample brings the latents to `2.9e-4`.

PIL's integer rules, verified exhaustively over all 256×256 (channel, alpha) pairs and implemented
in `gen_core::imageops::resize_lanczos_rgba_u8`:

* premultiply `c' = (c·a + 127) / 255` (round-half-up; `floor(c·a/255)` is off by one);
* un-premultiply `c = min(255, c'·255 / a)` for `a > 0`, and `c = c'` for `a = 0`.

For a fully opaque image both conversions are the identity and the alpha band resamples to a
constant 255, so this is byte-identical to the three-channel resize the RGB reference route always
used.

### What this port does NOT do

* It does not synthesise alpha. `supports_alpha_output` means "the decoder is natively
  four-channel", never "this provider can matte an opaque render".
* It does not threshold, clean up, premultiply or otherwise post-process the decoded alpha.
* It does not make transparency requestable as a mode. A caller asking for a transparent result
  writes a transparency prompt, exactly as upstream documents; `output_channels: Rgba` only
  decides whether the resulting alpha is delivered or composited away.

## Deliberate divergences

* No prefix KV cache: every step evaluates the full block-causal joint sequence (upstream's exact
  `QwenImage21AttnProcessor` prefill path). Upstream documents the cached and uncached paths as
  equally valid but not bit-identical. **On the reference route this is a real cost, not just a
  numerical choice**: every condition image is fitted to `output_resolution` whatever the target
  size, so ten references are ~41k prefix tokens that upstream encodes once and this port
  re-encodes at every step. Reference-heavy requests scale with `steps × references` here where
  upstream scales with `references + steps`.
* Latents stay f32 between Euler steps (upstream rounds to bf16 each step).
* The text tower runs f32 activations over bf16 weights (upstream: bf16 end to end).
* Noise is MLX-seeded (`mlx.random.normal` under `key(seed)`), not torch-seeded.
* RGB emission composites the RGBA decode over white — the `OutputChannels::Rgb` default.
  `OutputChannels::Rgba` emits the four-channel decode unflattened as `RgbaImage`
  (sc-24111); see *Native transparency* above.
