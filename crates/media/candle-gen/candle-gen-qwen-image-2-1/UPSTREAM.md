# Frozen upstream — Qwen-Image 2.1, candle port (sc-24109)

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

This crate commits **no fixtures of its own**. Its parity tests read the *same* committed
`.safetensors` oracles and the same miniature snapshot the MLX twin reads, across the backend
boundary by relative path:

```
crates/media/mlx-gen/mlx-gen-qwen-image-2-1/tests/fixtures/
```

(the idiom `candle-gen-ltx`, `candle-gen-mochi` and `candle-gen-krea` already use). They were
produced by `crates/media/mlx-gen/tools/dump_qwen21_*.py` (shared setup in `_qwen21_common.py`) from
the diffusers/transformers classes at the revisions above, on miniature seeded configs, with the
tiny snapshot written by `save_pretrained` in the exact layout above. One oracle per component means
the two backends cannot drift apart against two copies. Each Rust parity test names the tolerance it
holds the port to and prints its measured error.

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
the colour, and rounded **half-to-even** at 8 bits (numpy's `ndarray.round`, which is banker's
rounding — `[0.5, 1.5, 2.5, 3.5] -> [0, 2, 2, 4]`). That is what `candle_gen::round_rgb8` and
`mlx_rs::ops::round` implement; Rust's own `f32::round` is half-away-from-zero and would diverge
at an exact `.5`, so neither emission path uses it. Nothing premultiplies, and `A = 0` does
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

### Capability discovery: the catalog does not publish the flag

`runtime-catalog` does **not** emit `supports_alpha_output` in its provider rows — consistent with
`supports_hdr`, which it also omits. The catalog publishes the conditioning kinds (so
`referenceRgba` does appear there) but not the boolean output-surface flags. The SceneWorks half
therefore infers availability from the manifest's own `supportsAlphaOutput` key rather than from
the catalog dump; the engine-side flag remains the authority that the shared request floor
enforces, so a manifest that disagrees with the descriptor gets a typed refusal at `validate`
rather than a wrong render.

### What this port does NOT do

* It does not synthesise alpha. `supports_alpha_output` means "the decoder is natively
  four-channel", never "this provider can matte an opaque render".
* It does not threshold, clean up, premultiply or otherwise post-process the decoded alpha.
* It does not make transparency requestable as a mode. A caller asking for a transparent result
  writes a transparency prompt, exactly as upstream documents; `output_channels: Rgba` only
  decides whether the resulting alpha is delivered or composited away.
## Installable tiers (sc-24112)

Candle has **no affine-quantize-at-load path** for this family, and this story does not add one. What
it adds is the other half: candle *installs* the pre-quantized tiers
`mlx_gen_qwen_image_2_1::convert` produces, on exactly the artefacts MLX loads. Both packed
components bind through `candle_gen::quant::AdaptLinear::linear_detect_gs`, which reads the packed
triple straight into the quantized weight on the target device — no dense weight is ever
materialized.

**A tier is a whole-pipeline contract**: selecting q4 runs q4 through every packable component.

| component | bf16 | q8 | q4 |
|---|---|---|---|
| `transformer/` | dense | packed Q8, group 64 | packed Q4, group 64 |
| `text_encoder/` Qwen3 language tower | dense | packed Q8, group 64 | packed Q4, group 64 |
| `text_encoder/` token embedding + norms | dense (`quant::guard_dense`) | dense | dense |
| `vae/` | dense (`quant::guard_dense`) | dense | dense |

`Capabilities::supported_quants` therefore reads `[Q4, Q8]` on this backend after this story, and
`component_precision_floors` is **empty** — nothing is resident above the selected width.
`LoadSpec::quantize` is a **tier selector** here: it is accepted only when the snapshot on disk
already is that tier, a mismatch is refused with the snapshot to point at instead, and a Q4/Q8
request against a **dense** snapshot is still the same typed `Unsupported` ("no on-the-fly
quantization") it was before. A snapshot's tier is read from its packed code/scale shapes as well as
its `config.json` marker, so a q4 label over q8 weights is a hard error rather than a q8 render under
a q4 label.

The production tiers are produced by the MLX converter and re-hosted at
**`SceneWorks/qwen-image-2-1-mlx`** (`q8/`, `q4/` subdirectories, each with its `SHA256SUMS`); that is
the repository the SceneWorks manifest half pins.

The text encoder is packed here and dense in the 2512 `candle-gen-qwen-image` crate; the reasons are
in `mlx-gen-qwen-image-2-1/UPSTREAM.md` and in `crate::quant`.

## Memory (sc-24112)

`memory_strategy` publishes the shared ladder — `Resident`, `StagedResidency` and `BoundedDecode`
implemented, the two bounded-DiT rungs classified — and prices each component from the snapshot's own
tensor headers at the width **its** loader materializes it at. The text encoder is priced over the
loaded `model.language_model.*` prefix only: the checkpoint's `lm_head` and its whole `model.visual.*`
tower are on disk but materialized by nothing on this route.

The derived tier and activation tables are owned once, in `mlx_gen_qwen_image_2_1::memory_strategy`,
and are not duplicated here — parameter counts and joint-token arithmetic are properties of the model,
not of the backend. `memory_strategy::admission_geometry` reports the same consumer envelope as the
MLX twin, including that each of up to 10 reference images adds `REFERENCE_FIT_TOKENS` (4 096) rather
than the target's own token count: `reference::prepare` fits every condition image to
`OUTPUT_RESOLUTION` (1024²) first. References are **priced**, not refused.

## Turbo: the distilled checkpoint, `qwen_image_2_1_turbo` (sc-25901)

| What | Where | Revision |
| --- | --- | --- |
| Weights, `config.json`s, `model_index.json` (`sample_sigmas`), scheduler config, licence | `Qwen/Qwen-Image-2.1-Turbo` (Hugging Face; card `base_model: Qwen/Qwen-Image-2.1`) | `fc13ba1986fcb905854dc8de2f52eee395f49b5b` |
| Saved-sigma semantics of `QwenImage21Pipeline.__call__` | `huggingface/diffusers` PR #14950 (`pipelines/qwenimage21/pipeline_qwenimage21.py`) | — |

`src/lib.rs` carries the repository and revision as `TURBO_UPSTREAM_HF_REPO` /
`TURBO_UPSTREAM_HF_REVISION`. The licence is the same Qwen Research License Agreement (the card
declares `license_name: qwen-research` and ships the same `LICENSE`).

One architecture, served by this crate under a second engine id. Against the base repository the
Turbo snapshot has the **same** `transformer/config.json`, presets and licence, and differs in:

| | base `Qwen/Qwen-Image-2.1` | `Qwen/Qwen-Image-2.1-Turbo` | what this port does |
| --- | --- | --- | --- |
| `model_index.json` | no `sample_sigmas` | `sample_sigmas: [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]` (terminal 0 excluded) | read by `loader::load_turbo_schedule`; it is also the base/Turbo discriminator |
| `scheduler/scheduler_config.json` | `use_dynamic_shifting: true`, `shift_terminal: 0.02` | `use_dynamic_shifting: false`, `shift: 1.0`, `shift_terminal: null` | `TurboSchedule::from_values` requires exactly that static form |
| `vae/diffusion_pytorch_model.safetensors` | f32, 1.35 GB | bf16, 0.68 GB | nothing: candle casts on read, so both land on the lane's compute dtype |
| `text_encoder/` | four `model-0000N-of-00004.safetensors` shards | one `model.safetensors` | nothing: `candle_gen::loader::load_sorted_mmap` reads whatever shards the directory holds |

### The schedule

Upstream: with no call-time `sigmas`, the pipeline takes `config.sample_sigmas` and hands it to
`FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=…)`. With dynamic shifting off the scheduler
applies the static shift `shift·σ / (1 + (shift − 1)·σ)`, which is the identity at `shift: 1.0`;
`shift_terminal` is null, so there is no stretch; it then appends `0`. The eight saved sigmas are
therefore used **unchanged**: no `mu`, no terminal stretch, and nothing depends on the resolution.
`true_cfg_scale` defaults to 1 ("Generation uses CFG=1 by default", the model card), and a negative
prompt with a scale above 1 is still accepted.

This port (`scheduler::saved_sigmas`, `resolve_turbo_run_params`):

* `steps` unset, or equal to the saved length → exactly the saved list, then `0.0`. The default
  step count is the saved list's length, read from the snapshot (eight for the pinned revision).
* any other `steps = N ≥ 1` → the saved curve resampled — see *Deliberate divergences*.
* `guidance` / `true_cfg` default to 1 (off), exactly as on the base id.
* the curated `sampler` axis integrates over the same sigma list.
* the curated `scheduler` axis goes through the same `candle_gen::resolve_flow_schedule` seam as
  the base id, at `mu = 0` (`ln shift`): unset keeps the saved/resampled list; a curated name
  (`normal`, `karras`, `beta`, …) **replaces** it with that scheduler's own unshifted ramp at the
  same step count. That is off the sigmas the checkpoint was distilled over — upstream's card
  says other schedules were not evaluated for this checkpoint — and is offered because the axis is a caller's
  explicit choice on every other engine.

### Loading the wrong snapshot is refused

The two repositories share every tensor name and shape, so a swap would load. It is refused before
any weight is read (`loader::check_snapshot_variant`, from `load` and from the memory
registration), with a typed `Unsupported` naming the mismatch and the id to use instead:

* a snapshot that saves `sample_sigmas` under `qwen_image_2_1` — any present, non-null value,
  a malformed list included (the base generator **and** the base trainer);
* a snapshot that saves none — or saves them beside a dynamic-shift scheduler config — under
  `qwen_image_2_1_turbo`.

Deliberate consequence: the base id now reads `model_index.json` (for that one key), so a base
snapshot whose `model_index.json` is present but **unparsable** is refused where it used to load —
it cannot be shown not to be a Turbo snapshot. The MLX twin refuses it too. A snapshot with no
`model_index.json` loads as before.

### What is not under the Turbo id

No trainer (`TRAINER_ID` stays base-only). Reference conditioning, RGBA output and adapters are
advertised because the descriptor is the base descriptor under a second id and the code path is
shared; none of them has been run on real Turbo weights by this story.

### Installable tiers of the Turbo checkpoint (sc-25906)

The Turbo checkpoint tiers exactly as the base one does (*Installable tiers* above): candle loads
the q8/q4 trees the MLX converter writes, and that converter reads which checkpoint it is
re-packing off the source snapshot, so a Turbo tier still carries `model_index.json` with its
`sample_sigmas`, the static scheduler config and the bf16 VAE, byte for byte (plus a `NOTICE`
file with the licence's attribution, which only a Turbo tier carries). Consequences here,
all inherited rather than added — `tests/turbo_tiers.rs` holds each one:

* a Turbo q8/q4 tier loads under `qwen_image_2_1_turbo` with the matching `quantize` selector and
  renders over the sigmas **its own** `model_index.json` saves;
* each id refuses the other checkpoint's tier with the typed refusal its dense snapshot gets
  (`loader::check_snapshot_variant`), through `load` and through the memory registration, before
  the tier/request rule is consulted;
* the tier/request rules are `quant::resolve_requested_tier`'s, unchanged: a **dense** Turbo
  snapshot with a Q4/Q8 request is the same typed "no on-the-fly quantization" refusal the base id
  raises (the MLX twin quantizes a dense snapshot at load instead — on both ids);
* the contract prices the bf16 VAE at `memory_strategy::compute_width()`, the width candle holds
  it at (f32 on the CPU lane, bf16 on a GPU lane), so the Turbo and base checkpoints price alike
  on this backend — unlike MLX, which holds the VAE at its on-disk dtype — under the Turbo id's
  own `TURBO_MEMORY_CALIBRATION_FINGERPRINT`. Nothing here is measured.

## Prefix KV cache (sc-25904)

Upstream's `use_kv_cache=True` default is ported, as on the MLX twin (sc-25903) and with the same
public shape: `QwenImage21KVCache` → `PrefixKvCache` (`src/transformer.rs`),
`kv_cache_mode="extract"` → `forward_joint_extract`, `"cached"` → `forward_joint_cached`, and the
pipeline's loop → `denoise` / `denoise_reported` (`src/pipeline.rs`).

**Why it is valid.** The joint sequence is `[prefix = text + condition-image blocks][target]`.
Under `causal_condition` the prefix tokens modulate from the `t = 0` row, and block-causal
attention never lets a prefix token attend to the target. So every block's prefix hidden states —
and its prefix keys and values — are the same at every model evaluation of a request. The cache is
refused without `causal_condition` (the transformer returns a typed `Unsupported`; the loop, as
upstream's `use_kv_cache and transformer.config.causal_condition`, runs the prefill path and
reports it).

**What is stored.** Per block, exactly what `_qwenimage21_prepare_qkv` stores: the prefix rows of
`key` **after** QK-norm and RoPE and of `value` (the raw `to_v` projection), in the compute dtype
(f32 on the CPU lane, bf16 on CUDA/Metal) — plus the target block's RoPE rows. Upstream and the MLX
twin lay them out `[1, prefix_tokens, heads, head_dim]`; this crate's attention runs heads-first,
so they are held `[1, heads, prefix_tokens, head_dim]` (same values; it saves a transposing copy of
the whole prefix at every cached step). `2 · blocks · prefix_tokens · inner_dim · itemsize` bytes
per prompt branch: at the ten-reference boundary in bf16 (`2 · 32 · 41 216 · 4096 · 2`) about
20 GiB per branch, twice that under true CFG.

**How a cached evaluation runs.** Only the target tokens go through `img_in` and the blocks; each
block projects/norms/rotates the target's own `q`, `k`, `v`, prepends the cached prefix `k`/`v`,
and makes one unmasked attention call (the block-causal mask degenerates to full attention for
target rows). That call is `block_causal_attention` with no prefix segments — the prefill's own
target call, through the same `sdpa_budgeted_bhsd` at `ATTN_SCORES_BUDGET`. It has to be: at the
largest preset with ten references the call's scores are `32 · 16 800 · 58 016 ≈ 3.1e10` elements,
~14× `i32::MAX`, so it always runs query-chunked. The per-token modulation
and `norm_out` rows are the sampled-timestep row for every token, read directly (a broadcast of
that row, exact — as upstream's `torch.where` and the MLX twin's `broadcast_to`). Upstream still projects the whole text and image stream on a
cached step and then slices it away; this port does not compute what it would discard.

**When.** Upstream extracts at loop index 0. This port extracts at each branch's **first model
evaluation** and decodes at every later one: the curated solver axis may evaluate the model more
than once per step or off the schedule (`heun`, `dpmpp_sde`), and the cache does not depend on the
step. With true CFG the negative branch has its own, independent cache.

**Candle specifics.** Tensors are eager, so there is nothing to materialize — but a `narrow` is a
view that shares its source's storage, so each cache entry is copied (`force_contiguous`, which
always allocates; `contiguous` returns the view itself when it already counts as contiguous) into
a storage of exactly its own size, inside the block that produced it. The block's full
`[1, heads, prefix + target, head_dim]` K/V are released when its attention returns, as on the
prefill path. The extract step's live set is therefore the prefill forward's plus the cache it is
building:

    extract peak = prefill peak + 2 · blocks · prefix_tokens · inner_dim · itemsize
                   (+ the target's RoPE rows, 2 · target_tokens · head_dim/2 · 4 bytes)

The caches are owned by the denoise loop's velocity closure, which the sampler consumes, so they
are dropped when the loop returns — on success, cancellation or error — and nothing of them
reaches the VAE decode. A cache only exists once a complete extract has returned, so there is no
empty or partial cache to decode from.

**The switch.** `DenoiseInputs::kv_cache` (`KvCacheMode::{On, Off}`). The generator (`src/lib.rs`)
drives it from the request (sc-25905, below): `On` for a request that selects
`PrefixKvCacheMode::Cached`, `Off` otherwise. `Off` is the prefill loop, byte for byte. `denoise_reported` fills a `DenoiseReport`:
the path that ran (`KvCachePath`), the bytes the caches held, and every transformer forward
(`ModelEvaluation`: branch, `Prefill` / `Extract` / `Cached`, tokens the blocks ran over). The
trainer's preview render stays on `Off`: it runs inside the trainer's memory envelope, which prices
the prefill path.

**Numerics.** Upstream documents the two paths as equally valid but not bit-identical (on the tiny
fixture, torch CPU f32, they differ by ≤ 1.2e-6 after three steps). Here the cached step's
attention call consumes the same key/value values through the same call as the prefill's target
call, and reads the same modulation row, exactly, as the prefill does. The residual difference
between the two paths is the CPU GEMM's kernel selection by row count (the cached step projects
target rows only): bit-identical on arm64, not guaranteed on x86-64. `tests/kv_cache.rs` gates it
at `|Δ| ≤ 1e-5 + 1e-5·|want|`; it also gates the cached
path — and the off path, against upstream's `use_kv_cache=False` latents — on the fixture the MLX
twin reads (`mlx-gen-qwen-image-2-1/tests/fixtures/qwen21_kv_cache.safetensors`,
`mlx-gen/tools/dump_qwen21_kv_cache.py`). The bf16 the released checkpoint runs in on CUDA/Metal
is not reachable from those tests (candle's CPU backend has no bf16 matmul).

**History — the prefill's modulation select (2026-10-13, sc-25904).** Until this date the candle
prefill forward selected each token's modulation and `norm_out` row with a blend,
`zero + (real − zero)·mask`, where upstream's `_select_modulation_rows` is `torch.where`. It was
exact on prefix rows and rounded on target rows; simulated on 1e6 N(0, 1) pairs, f32 was inexact
on 36% of entries (max abs 2.4e-7), and bf16 — the dtype production runs on CUDA/Metal — on 36%
with median relative error 0.6%, p99 9%, max abs 0.031 and some entries zeroed outright.
`JointGeometry::select` is now an exact select (each token's row is a copy of the `t = 0` row or
of the sampled row), on the prefill, the extract and the trainer forward alike. Base 2.1 candle
output in bf16 therefore changed, toward upstream, by up to those figures; f32 output changed in
the last bits.

**Priced per request, selected per request (sc-25905).** Upstream's `use_kv_cache=True` is a
default with no memory model behind it; here the cache is a **request selection** —
`GenerationMemory::prefix_kv_cache`, a typed execution domain declared on
`Capabilities::execution` for both engine ids (`gen_core::execution_domains`) — and it is priced:

* **What it costs.** K and V for all 32 blocks over the prefix (the prompt rows and every
  reference image), at the dtype the DiT computes in, once per prompt branch:
  `2 · 32 · prefix_tokens · 4096 · width`. **On this backend the width follows the device, not
  the tier**: a packed (q8 / q4) Linear dequantizes its weight to the activation's dtype
  (`QLinear::forward`), so the hidden states and the cached K/V stay at the dtype the model was
  loaded at — bf16 on CUDA and Metal (0.5 MiB per prefix token), f32 on a CPU device (1 MiB), on
  every tier (observed on the miniature q8 tier by `tests/kv_admission.rs`). No tier shrinks the
  cache. The MLX twin differs: its packed tiers compute in f32.
* **What is priced: an upper bound.** The real prefix is known only after tokenizing and fitting,
  so the surcharge (`memory_strategy::prefix_kv_cache_demand_bytes(width, height, references,
  negative)`) is over two published bounds that err high: 1024 text rows per branch
  (`PREFIX_KV_CACHE_CONDITIONING_TOKENS` — chosen, not measured) and 4920 tokens per reference
  (`PREFIX_KV_CACHE_REFERENCE_TOKENS` — the largest grid any aspect ratio fits to: 13120×96, 20%
  above the square 4096; derived by search over the ratios the processor accepts). At bf16:
  0.5 GiB with no reference, 2.40 GiB more per reference, 24.5 GiB at ten; twice that at f32;
  doubled with a negative branch; plus the target RoPE rows each cache keeps. It is priced for the
  build's production device (`compute_width`); `prefix_kv_cache_demand_bytes_at` with
  `compute_width_on(device)` prices a GPU build loaded onto a CPU device.
* **A cached render is held to that price.** With the real layouts in hand the generator computes
  what the caches would hold (`memory_strategy::prefix_kv_cache_overflow`); a request past its
  priced figure — a prompt branch over the text bound, or a generator on a wider device than the
  build prices — runs **prefill** instead: never cached above its price, never a failed render.
  The report carries both byte counts (`PrefixKvCacheReport::overflow`) and a
  `prefix_kv_cache_shed_over_priced_bound` warning.
* **What does not move.** This crate publishes no activation table and installs no allocator
  budget, so the cache-shed demand is whatever the caller already prices. The memory contract, its
  asset facts, its formula, the admission envelope and both calibration identities are unchanged.
  A request that selects nothing runs the prefill path — the deliberate divergence from
  upstream's default, because a request admitted against a demand that never priced the cache
  must not run above it.
* **Who decides, and what the engine does not check.** The caller that owns the budget:
  `PrefixKvCacheMode::select(budget, shed, cache)` answers `Cached` when both fit, `Prefill` when
  only the shed demand fits (the request is admitted and runs without the cache), and `None` when
  neither does (the caller refuses as it always did). Two things are the caller's alone. *Order:*
  the request scope's `configure_request` replaces `GenerationRequest::memory` (with `None` for a
  `Resident` selection), so the selection must be written **after** it, creating a block if there
  is none — written before, it is lost and the render runs prefill in silence (pinned by
  `a_selection_written_before_the_scope_configures_the_request_is_lost`). *Admission:* the engine
  honours `Cached`; it does not verify the request was admitted against the cached demand.
* **What ran.** `Generator::generate_with_report` → `GenerationReport::prefix_kv_cache`: the path
  the denoise loop took (`cached` / `prefill`), the bytes its caches held, and `overflow` when a
  cached selection was shed at the bound — written from the loop's own `DenoiseReport`.
* **Approximations that remain.** The admission envelope still assumes 256 text rows and a square
  4096-token reference (pre-existing; errs low for a long prompt or an extreme-aspect reference).

**Every cache figure above is derived — exact tensor arithmetic over the shapes the cache stores,
held against a real cache on the miniature fixture — and none is measured on a checkpoint.** The
epic's terminal story measures one cached ten-reference render.

## Deliberate divergences

### From upstream (shared with the MLX twin)

* Prefix KV cache: extract is keyed on each branch's first model evaluation rather than loop
  index 0, and a cached step skips the prefix projections upstream computes and discards — see
  *Prefix KV cache* above.
* Prefix KV cache default (sc-25905): upstream runs the cache unless told otherwise
  (`use_kv_cache=True`); here a request that selects nothing runs the **prefill** path, and the
  cache is selected per request by the caller that priced it — see *Prefix KV cache* above.
* Latents stay f32 between Euler steps (upstream rounds to bf16 each step).
* On the **CPU parity lane** the text tower runs f32 (upstream: bf16 end to end). On CUDA/Metal
  it runs bf16 end to end **like upstream** — see *Compute dtype* below (sc-24114).
* Noise is seeded from the shared launch-portable CPU `StdRng` (`candle_gen::seed`), not torch's
  generator; seed parity across frameworks is not a goal.
* RGB emission composites the RGBA decode over white — the `OutputChannels::Rgb` default.
  `OutputChannels::Rgba` emits the four-channel decode unflattened as `RgbaImage`
  (sc-24111); see *Native transparency* above.

### On the Turbo id: off-count step requests are resampled (a SceneWorks choice)

Upstream gives `num_inference_steps` no effect on the Turbo checkpoint: "Setting
`num_inference_steps` alone does not override" the saved schedule (model card), and the only way
to another count is an explicit call-time `sigmas` list the caller invents. SceneWorks keeps
`steps` a live knob instead (epic 25899 E2), so a request for `N ≠ K` steps (`K` = the saved
length) **resamples the saved curve**. This is our rule, not upstream's, and it is the same rule on
both backends:

* knots: the saved sigmas `s_0 … s_{K−1}` followed by `s_K = 0.0`, at positions `u_j = j / K`
  for `j = 0 … K`;
* for `i = 0 … N−1`: `σ_i` = the piecewise-linear interpolation of the knots at `u = i / N`;
* then append `0.0`.

Computed in f64 and rounded to f32 at the end. `N = K` reproduces the saved list exactly (this
crate takes the knot index and fraction from the integer product `i·K`, so a sample on a knot
reads it back bit-for-bit). For the pinned revision, twelve steps are

```
1.0, 0.98563533, 0.970362, 0.95418, 0.9358107, 0.9161107, 0.89508,
0.861792, 0.79827667, 0.704534, 0.5112233, 0.27637867, 0.0
```

pinned as literals in `src/scheduler.rs` and `tests/turbo.rs`. Nothing about image quality at an
off-count is claimed: the checkpoint was distilled over the eight saved sigmas only.

### From the MLX twin (this crate is the candle sibling)

* `backend = "candle"`, `mac_only = false`.
* **No on-the-fly Q4/Q8.** MLX quantizes the DiT's Linears at load from `LoadSpec::quantize`;
  candle has no affine-quantize-at-load path, so `load` refuses that spec with a typed
  `Unsupported` and the descriptor advertises no `supported_quants`. An already-MLX-packed snapshot
  still loads: every DiT Linear goes through `candle_gen::quant::AdaptLinear::linear_detect_gs`,
  the packed-detect seam `candle-gen-qwen-image` uses, at the same group size the MLX `quantize`
  would have written (64, or 32 for a narrower input width).
* Tensors stay NCHW end to end. The MLX VAE works channels-last because mlx convolutions are NHWC;
  candle is NCHW natively and the torch weights already ship `[out, in, kh, kw]`, so the layout
  shuffles disappear and the two parameter-free VAE shortcuts (`AvgDown3D`, `DupUp3D`) are
  re-derived on the channel axis rather than transcribed.
* The Qwen3 decoder block is ported into `src/text_encoder.rs` rather than reused: the MLX twin
  reaches `mlx_gen_z_image::text_encoder::EncoderLayer`, but candle's Z-Image Qwen3 decoder is a
  private vendored module pinned to Z-Image's layer[-2] / `model.` prefix conventions.
* Noise draw and per-image seed derivation come from `candle_gen::seed` (`image_seed`,
  `seeded_normal_vec`), so a seed reproduces within the candle backend but not across the two
  engines.
* **Compute dtype** is the backend's, for *every* component: f32 on CPU (the parity lane), bf16 on
  CUDA/Metal — where MLX always loads at the checkpoint's on-disk dtype. That includes the Qwen3-VL
  text encoder (language tower and ViT): `loader::load_text_encoder_from` builds the tower's
  `VarBuilder` at `compute_dtype()`, so a CUDA/Metal build runs the tower in bf16 exactly as
  upstream's `QwenImage21Pipeline` runs `Qwen3VLForConditionalGeneration` (`torch_dtype=bfloat16`),
  and the memory contract prices it at 2 B/param — the resident-bytes basis the SceneWorks floors
  were derived at. Until sc-24114 the tower was read at f32 on every backend and priced at
  4 B/param, which put the candle bf16 resident set at ~42.7 GiB against the 28.61 GiB the
  derivation states. The S2/S3/S4 CPU parity lanes are unchanged: the loaders materialize at
  `loader::compute_dtype_on(device)`, which is f32 for a CPU device on every build, a CUDA/Metal
  build included (candle's CPU backend has no bf16 matmul). Production loads onto
  `candle_gen::default_device()`, so a GPU build still runs every component at bf16.
* **Reference route (sc-24110).** The semantics above are the frozen upstream ones and are held to
  the *same* committed fixture (`qwen21_edit.safetensors`) as the MLX twin. The mechanical
  divergences are:
  * The Qwen3-VL vision tower is `candle_llm::models::Qwen35VisionModel` /
    `Qwen35VisionConfig` — candle-llm carries Qwen3-VL and Qwen3.6 as **one** tower under the
    `Qwen35…` spelling, where mlx-llm names it `Qwen3VLVision…`. Same architecture, same weights,
    same `model.visual.*` prefix.
  * The route's typed refusals — a `Mask` conditioning, an unadvertised conditioning kind, eleven
    references, a per-reference `strength`, a snapshot with no tower, a `smart_resize` grid
    disagreement — are `CandleError::Unsupported` with the **identical message text** the MLX twin
    raises, and cross the `From<CandleError> for gen_core::Error` bridge as
    `gen_core::Error::Unsupported` (sc-24114; `CandleError` gained the variant for exactly this,
    where it used to arrive as `Msg`). The gen-core-testkit validate-honesty check requires the
    typed variant for every undeclared conditioning kind.
  * `prepare_reference` / `prepare_references` take an explicit `&Device`: candle tensors are
    device-bound at construction, where mlx `Array`s are not.
  * The ViT tower's weights are pulled out of the `text_encoder/` shards by an mmapped,
    prefix-filtered read (`loader::load_vision_tower_weights`) rather than filtered out of an
    already-resident `Weights` map as on the MLX side — candle's loader hands providers a
    `VarBuilder`, which cannot be enumerated. Only `model.visual.*` is materialised, so attaching
    the tower never stages a second copy of the language tower. Each visual tensor is cast to
    `compute_dtype()` on the way in, so both halves of the one encoder run in the same dtype as
    the language tower's `VarBuilder`: bf16 end to end on CUDA/Metal (an `Arc`-clone no-op on the
    released bf16 ViT), f32 end to end on the CPU parity lane.
  * The decoder block's rotary embedding grew an `Option<(&cos, &sin)>` parameter: `None` keeps
    the precomputed 1-D table the text-to-image path has always used (that path stays
    **bit-identical**, asserted by `edit_parity::the_text_to_image_path_is_unchanged_by_the_reference_route`),
    and `Some` applies the interleaved M-RoPE tables from
    `candle_llm::primitives::Rope::mrope_interleaved_cos_sin` through `candle_llm::primitives::apply_rope`.
    Both are the same NeoX half-split convention.
