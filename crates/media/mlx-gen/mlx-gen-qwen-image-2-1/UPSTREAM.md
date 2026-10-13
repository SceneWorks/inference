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
`use_dynamic_shifting` must be true, `time_shift_type` must be `exponential`). The distilled Turbo
snapshot differs here — see *The distilled Turbo checkpoint* below.

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

Qwen-Image 2.1 ships **pre-quantized**: a tier is a complete standalone snapshot in the layout above
whose weight-bearing components are already affine-quantized, produced offline by
`convert::prequantize_turnkey` and loaded with no dense transient. `LoadSpec::quantize` selects a
tier; it is a transform request only against a dense snapshot.

**A tier is a whole-pipeline contract.** Selecting q4 runs q4 through every packable component.
There is no per-component promotion, no `component_precision_floors` declaration and no override:
`Tier::text_encoder_bits` is literally `Tier::transformer_bits`.

| component | bf16 | q8 | q4 |
|---|---|---|---|
| `transformer/` (7.12 B params in 232 Linears) | dense bf16 | packed Q8, group 64 | packed Q4, group 64 |
| `text_encoder/` Qwen3 language tower (6.95 B params in 252 decoder Linears) | dense bf16 | packed Q8, group 64 | packed Q4, group 64 |
| `text_encoder/` token embedding (622 M) | dense | dense | dense |
| `text_encoder/` `lm_head`, `model.visual.*` | dense, **not loaded** by this route | dense | dense |
| `vae/` (338 M) | dense f32 | dense f32 | dense f32 |

(The distilled Turbo checkpoint's `vae/` ships bf16 and rides through at bf16 — see *Installable
tiers of the Turbo checkpoint* below.)

Three decisions, each deliberate:

* **The text encoder is packed**, unlike the 2512 `mlx-gen-qwen-image` crate. 2512 pairs a ~20 B DiT
  with a ~7 B tower, so a dense tower is a minority of its footprint. 2.1 pairs a **7.12 B** DiT with
  a **7.57 B** tower: under `Sequential` the resident floor is `max(tower, DiT + VAE)`, so a dense
  tower would pin that floor at ~14.5 GiB at every tier and the Q4 tier would buy ~0.3 GiB over Q8.
  Reusing 2512's table here would have produced a tier that cannot do its job.
* **No component precision floor.** An earlier revision held the tower at Q8 on the q4 tier, on
  the strength of `mlx_gen_mage`'s measured Qwen-LM-tower sweep. That was **withdrawn**: it is a
  prior from a different model, it minted a "q4" tier that was not q4, and it had to be carried as a
  `component_precision_floors` declaration every caller then has to reason about. A tier that needs a
  floor to be usable is a measurement result, not a default — and this route has no measurement. If
  a real sweep later shows the q4 tower is unusable, the honest outcome is a narrower
  `supported_quants`, not a q4 tier that quietly runs q8.
* **Group size 64, uniformly.** A tier declares exactly one `quantization.group_size`, and the
  packed bit-width is derived from the packed shapes *at the group size the loader passes*, so a tier
  mixing group sizes would be decoded at the wrong width. A `Linear` narrower than 64 therefore stays
  dense. The released geometry has no such width (its four DiT input widths are 64 / 4096 / 4096 /
  12288 and the tower's are 4096 / 12288), so in production the converter packs everything; only the
  miniature parity snapshot is partly dense.

A **load-time** tier (a dense snapshot plus `LoadSpec::quantize`) is whole-pipeline or nothing: a
geometry the single group size cannot cover is a typed refusal naming the widths, never a silently
dense text encoder under a quantized label. The miniature parity snapshot is exactly such a geometry.

The converted artefacts are **byte-reproducible** (keys are sorted before serialization and the
config merge is deterministic) and each packed triple is **byte-identical to `mlx_rs::ops::quantize`**
over the bf16 source — the same op the load-time quantizer runs. `tests/tiers.rs` pins both, plus a
device-independent dequantization check against the dense weights, and `tests/fixtures/tiers/`
carries the committed miniature tier weights the Candle backend reads.

A snapshot's tier is read from its **packed code/scale shapes** as well as its `config.json` marker;
a q4 label over q8 weights is a hard error, because such a snapshot renders perfectly well at q8 and
nothing downstream would otherwise notice.

### Producing a tier

```sh
QWEN21_SRC=<dense snapshot> QWEN21_TIER=q4 QWEN21_DST=~/SceneWorks/qwen-image-2-1-tiers \
  cargo run --release --example qwen_image_2_1_prequant -p mlx-gen-qwen-image-2-1
```

The **converter itself** (`convert::prequantize_turnkey`, sc-24114) writes two files beside the
weights of every tier it assembles: `<tier>/CHANGES.md`, the change record the Qwen Research
License §3(b) requires of a modified redistribution (which components were re-packed, at what
bits/group, from which upstream revision, by which converter version), and `<tier>/SHA256SUMS`, the
`sha256sum`-format manifest over every other file. The example only selects the tier and echoes
the manifest. The production tiers are re-hosted at **`SceneWorks/qwen-image-2-1-mlx`**, one
subdirectory per tier (`q8/`, `q4/`), each carrying both files — that is the repository the
SceneWorks manifest half pins.

## Memory (sc-24112)

`memory_strategy` publishes the shared ladder: `Resident`, `StagedResidency` and `BoundedDecode` are
implemented; `BoundedAttention` and `BoundedTransformerResidency` are classified
`StructurallyNotApplicable` (block-causal per-segment attention has no chunked variant here, and
there is no block-streaming loader) rather than left to the ladder's cost order.

`memory_strategy::derived` publishes a closed-form estimate of resident weights and of the warm
activation transient for every tier at every preset. **Every number there is derived** — parameter
counts × dtype width, plus a structural count of the live tensors in this crate's own forward — and
this route therefore registers **no** `ActivationMemoryRegistration`: that carrier publishes only
real on-device measurements, so filing a derivation there would launder an estimate into evidence.
`activation_memory_bytes_1024("qwen_image_2_1")` answers `None`, and callers fall back to the
asset-facts + generic-headroom estimate path where the estimate safety margin applies. None of it is
copied from the 2512 route.

`memory_strategy::admission_geometry` reports the envelope a consumer must gate on: the largest
preset **area** (2400×1792, which is neither the widest preset nor the square default), the 16 800
latent tokens it contributes, and the fact that each of up to 10 reference images adds a further
**4 096** tokens — `reference::prepare` fits every condition image to `OUTPUT_RESOLUTION` (1024²)
before the vision tower and the VAE see it, so a reference costs the same whatever the target size.
The worst-case joint sequence is therefore `256 + 16 800 + 10 × 4 096 = 58 016` tokens. References
are **priced** by the memory contract, not refused; the 1..=10 bound and every per-image geometry
check stay request-time refusals in `reference`.

## The distilled Turbo checkpoint — `qwen_image_2_1_turbo` (sc-25900)

| What | Where | Revision |
| --- | --- | --- |
| Weights, `config.json`s, tokenizer, scheduler config, `sample_sigmas`, licence | `Qwen/Qwen-Image-2.1-Turbo` (Hugging Face) | `fc13ba1986fcb905854dc8de2f52eee395f49b5b` |
| `QwenImage21Pipeline.config.sample_sigmas` ("Configure fixed sampling sigmas on QwenImage21Pipeline", PR 14950) | `huggingface/diffusers` (`src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py`) | `da1d3829cf08d4f329b526d89e17cc035c049d8d` |

A second engine id served by this same crate. `Qwen/Qwen-Image-2.1-Turbo` is a distillation of
Qwen-Image 2.1 with the **same** `transformer/config.json`, presets, reference/RGBA routes and
licence (its `LICENSE` is byte-identical to the base repository's). Everything above this section
applies to it unchanged; what differs is the snapshot, and through it the schedule.

### What the Turbo snapshot carries

* `model_index.json` has one extra key, `sample_sigmas`:
  `[1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]` — eight sigmas, the
  terminal `0` excluded.
* `scheduler/scheduler_config.json` turns every scheduler transform off: `use_dynamic_shifting:
  false`, `shift: 1.0`, `shift_terminal: null` (the base repository has `true` / `1.0` / `0.02`).
  The remaining keys are identical.
* `vae/diffusion_pytorch_model.safetensors` is bf16 (base: f32), and `text_encoder/` is a single
  `model.safetensors` (base: four shards). Neither needs loader code: the VAE computes in its
  weight dtype and every component loads whatever `*.safetensors` its directory holds.

### What upstream does with it

`QwenImage21Pipeline.__call__` (at the diffusers revision above):

```python
if sigmas is None:
    sigmas = self.config.sample_sigmas
    if sigmas is None:
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
mu = calculate_shift(...)
timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
```

and `FlowMatchEulerDiscreteScheduler.set_timesteps`, with dynamic shifting off, applies the static
shift `shift·σ / (1 + (shift − 1)·σ)` — the identity at `shift: 1.0` — skips the terminal stretch
(`shift_terminal` is null) and appends `0`. So the schedule **is** the saved list plus a trailing
zero: `mu` is computed and ignored, and `num_inference_steps` is ignored too (the docstring:
"Ignored when `sigmas` or the pipeline's configured `sample_sigmas` is used; the length of that
schedule determines the number of steps"). `true_cfg_scale` still defaults to `1.0`, and a negative
prompt with a scale above 1 still runs the second forward.

### What this port does

* `qwen_image_2_1_turbo` is registered beside `qwen_image_2_1` (`model::TURBO_REGISTRATION`, its
  own memory strategy / fixture / behaviour registrations; **no** trainer — training stays on the
  base checkpoint). `model::Variant` picks the id and `model::Schedule` the schedule family; every
  other line of the load and render path is shared.
* The schedule is read from the snapshot at load (`loader::load_saved_sigma_schedule` →
  `config::SavedSigmaSchedule`), never from constants in this crate. A request with no step count
  runs one step per saved sigma; `steps` equal to the saved length is the same list, bit for bit.
* A snapshot of the other checkpoint is a typed refusal that names the mismatch and the id to use:
  the base id refuses a `model_index.json` that saves `sample_sigmas`; the Turbo id refuses one
  that does not, and refuses a scheduler config that would move the saved values
  (`use_dynamic_shifting: true`, `shift` other than `1.0`, a non-null `shift_terminal`).
* A curated `scheduler` name goes through the same `resolve_flow_schedule` seam as the base id,
  built over an unshifted flow model (`mu = ln(shift) = 0`). It **replaces** the saved sigmas with
  that scheduler's own curve at the requested step count — it is the caller leaving the distilled
  schedule, not a re-spacing of it. The `sampler` axis integrates whichever schedule was chosen.
* One step is accepted (`[σ₀, 0]`); the base id's `steps >= 2` floor exists only because its
  terminal stretch divides by zero there.

### Installable tiers of the Turbo checkpoint (sc-25906)

The same converter, with no variant flag: `convert::prequantize_turnkey` reads which checkpoint
its **source snapshot** is (`convert::source_variant`) from the two signals the loader refuses the
wrong engine id on — `model_index.json` saving `sample_sigmas`, under a static-shift scheduler
config — and a source that is neither checkpoint is the loader's own typed refusal, raised before
anything is written. A Turbo q8/q4 tier is the *Installable tiers* layout above with these
differences, all inherited from the snapshot rather than decided by the converter:

| in the tier | Turbo | base |
|---|---|---|
| `model_index.json` | byte-for-byte copy, `sample_sigmas` included | byte-for-byte copy |
| `scheduler/scheduler_config.json` | byte-for-byte copy (static identity shift) | byte-for-byte copy (dynamic shift) |
| `vae/` | byte-for-byte copy, **bf16** | byte-for-byte copy, f32 |
| `text_encoder/model.safetensors` | packed from upstream's one file | packed from upstream's four shards |
| `CHANGES.md` | names `Qwen/Qwen-Image-2.1-Turbo` at its pinned revision, and that the tier loads as `qwen_image_2_1_turbo` | names `Qwen/Qwen-Image-2.1` at its pinned revision — wording unchanged by this story |
| `NOTICE` | the licence's §3(c) attribution notice, verbatim, bound by `SHA256SUMS`; `CHANGES.md` points at it | not written — a base tier is produced byte for byte as before, so published base tiers still reproduce; its attribution is the notice quoted in the `LICENSE` it carries |

Because the schedule signals ride in files a tier copies verbatim, a tier is the same checkpoint as
its source: a Turbo tier is refused under `qwen_image_2_1` and a base tier under
`qwen_image_2_1_turbo` with exactly the typed refusals the dense snapshots get, through `load` and
through the memory registration. The tier/request rules (`quant::needs_load_time_quant`) are the
base checkpoint's, unchanged, and the bf16 tier is still refused rather than copied.

**Derived memory — nothing here is measured.** The Turbo checkpoint's **only** derived difference
from the base one is the autoencoder's resident bytes: it ships bf16 and MLX holds it at the
on-disk dtype (675 480 808 B against 1 350 961 616). `memory_strategy::derived`'s
`*_for(variant, …)` functions carry that through the totals built on the resident set (resident
total, staged floor, peak totals, tier tree); the un-suffixed ones remain the base checkpoint's.
**Every activation and transient term — denoise, decode head and tail, reference encode, the
request budget — is the base checkpoint's, unchanged.** A bf16 decode stream *may* be cheaper (the
head's attention matrices would be half as wide), but every `ChannelNorm` still widens to f32, the
bounded decode's tile term already runs low against its one f32 trace, and nothing has been
measured on a bf16 VAE; that saving is deliberately not claimed until the terminal story measures
it.

| Turbo, derived | bf16 | q8 | q4 |
|---|---|---|---|
| tower (same as base) | 14.10 GiB | 8.03 GiB | 4.80 GiB |
| DiT (same as base) | 13.25 GiB | 7.04 GiB | 3.73 GiB |
| VAE (bf16) | 0.63 GiB | 0.63 GiB | 0.63 GiB |
| resident total | 27.98 GiB | 15.70 GiB | 9.15 GiB |
| staged floor `max(tower, DiT + VAE)` — the **tower** binds | 14.10 GiB | 8.03 GiB | 4.80 GiB |
| default-path peak at the largest-area preset (base less the VAE difference) | 34.35 GiB | 22.08 GiB | 15.53 GiB |
| tier tree weight files (`derived::tier_tree_bytes`) | 30.21 GiB | 17.94 GiB | 11.39 GiB |

The Turbo id carries its own calibration identity,
`memory_strategy::TURBO_MEMORY_CALIBRATION_FINGERPRINT`, and registers no activation anchor.

## Prefix KV cache (sc-25903)

Upstream's `use_kv_cache=True` default is ported: `QwenImage21KVCache` → `PrefixKvCache`
(`src/transformer.rs`), `kv_cache_mode="extract"` → `forward_joint_extract`, `"cached"` →
`forward_joint_cached`, and the pipeline's loop → `denoise` / `denoise_reported`
(`src/pipeline.rs`).

**Why it is valid.** The joint sequence is `[prefix = text + condition-image blocks][target]`.
Under `causal_condition` the prefix tokens modulate from the `t = 0` row, and block-causal
attention never lets a prefix token attend to the target. So every block's prefix hidden states —
and its prefix keys and values — are the same at every model evaluation of a request. The cache is
refused without `causal_condition` (the transformer returns a typed `Unsupported`; the loop, as
upstream's `use_kv_cache and transformer.config.causal_condition`, runs the prefill path and
reports it).

**What is stored.** Per block, exactly what `_qwenimage21_prepare_qkv` stores: the prefix rows of
`key` **after** QK-norm and RoPE and of `value` (the raw `to_v` projection), as
`[1, prefix_tokens, heads, head_dim]` in the compute dtype — plus, here, the target block's RoPE
rows. `2 · blocks · prefix_tokens · inner_dim · itemsize` bytes per prompt branch: at the
ten-reference boundary in bf16 (`2 · 32 · ~41 200 · 4096 · 2`) about 20 GiB per branch.

**How a cached evaluation runs.** Only the target tokens go through `img_in` and the blocks;
each block projects/norms/rotates the target's own `q`, `k`, `v`, prepends the cached prefix
`k`/`v`, and makes one unmasked attention call (the block-causal mask degenerates to full attention
for target rows). The modulation and `norm_out` rows are the sampled-timestep row for every token.
Upstream still projects the whole text and image stream on a cached step and then slices it away;
this port does not compute what it would discard (the projections are row-wise, so the target rows
are the same).

**When.** Upstream extracts at loop index 0. This port extracts at each branch's **first model
evaluation** and decodes at every later one: the curated solver axis may evaluate the model more
than once per step or off the schedule (`heun`, `dpmpp_sde`), and the cache does not depend on the
step. With true CFG the negative branch has its own, independent cache.

**MLX specifics.** The cache is materialized (`eval`) during the extract forward, **block by
block** — a lazy cache would re-run the prefix graph on every step and keep it alive, and a cache
evaluated only at the end of the forward would keep every block's full K/V alive until then (each
entry is cut from that block's `[1, prefix + target, heads, head_dim]` K and V). Each cached
tensor is copied into its own buffer: an MLX slice is a view, so caching the slice would pin the
block's whole prefill K/V (upstream's "`clone()`, not `contiguous()`"). The extract step's live
set is therefore the prefill forward's plus the cache it is building
(`2 · blocks · prefix_tokens · inner_dim · itemsize`, and the target's RoPE rows). The
caches are owned by the denoise loop's velocity closure and are dropped when the loop returns —
on success, cancellation or error — so nothing of them reaches the VAE decode. A cancellation
(the shared sampler's step gate) ends the loop, as upstream's `break`; a cache only exists once a
complete extract has been evaluated, so there is no empty or partial cache to decode from.

**The switch.** `DenoiseInputs::kv_cache` (`KvCacheMode::{On, Off}`). The render path
(`src/model.rs`) drives it from the request (sc-25905, below): `On` for a request that selects
`PrefixKvCacheMode::Cached`, `Off` otherwise. `Off` is the prefill loop, bit for bit. `denoise_reported` fills a
`DenoiseReport`: the path that ran (`KvCachePath`), the bytes the caches held, and every
transformer forward (`ModelEvaluation`: branch, `Prefill` / `Extract` / `Cached`, tokens the
blocks ran over). The trainer's preview render stays on `Off`: it runs inside the trainer's
memory envelope, which prices the prefill path.

**Numerics.** Upstream documents the two paths as equally valid but not bit-identical (on the tiny
fixture, torch CPU f32, they differ by ≤ 1.2e-6 after three steps). Here the cached step's
attention call consumes the same key/value values in the same kernel as the prefill's target call,
and on Metal the two paths are **bit-identical** at every evaluation and after the whole denoise —
on the f32 tiny fixture and on a `head_dim = 128` model in f32 and bf16 (the geometry at which MLX
selects its fused attention kernel on Metal; the dispatch itself is not asserted by the tests).
`tests/kv_cache.rs` holds that to an absolute +
relative bound and gates the cached path against upstream's own
(`tests/fixtures/qwen21_kv_cache.safetensors`, `tools/dump_qwen21_kv_cache.py`).

**Priced per request, selected per request (sc-25905).** Upstream's `use_kv_cache=True` is a
default with no memory model behind it; here the cache is a **request selection** —
`GenerationMemory::prefix_kv_cache`, a typed execution domain declared on
`Capabilities::execution` for both engine ids (`gen_core::execution_domains`) — and it is priced:

* **What it costs.** K and V for all 32 blocks over the prefix (the prompt and every reference
  image), at the compute dtype, once per prompt branch: `2 · 32 · prefix_tokens · 4096 · 2` bytes
  in bf16, 0.5 MiB per prefix token. A 256-token prompt is 0.125 GiB; each reference (fitted to
  1024², 4096 tokens) adds 2.0 GiB; ten references are 20.1 GiB; a negative branch doubles all of
  it. No tier shrinks it — it is activations, not weights
  (`memory_strategy::derived::prefix_kv_cache_bytes`).
* **How a cached request is budgeted.** At its cache-shed budget plus those bytes plus the target
  RoPE rows each cache keeps (`derived::prefix_kv_cache_demand_bytes`,
  `derived::request_transient_budget_bytes_for`), which is what the generator installs as the
  request's allocator bounds when it runs cached. The bound behind that: the extract evaluation's
  peak is at most the prefill forward's peak plus the caches plus those RoPE rows, and every
  later evaluation carries the target-only transient plus the caches, which is lower.
* **What does not move.** The cache-shed budget, the derived table and every floor are the
  numbers published before the cache existed, to the byte, for both ids. A request that selects
  nothing runs the prefill path under that budget — the deliberate divergence from upstream's
  default, because a request admitted against a budget that never priced the cache must not run
  above it.
* **Who decides.** The caller that owns the budget: `PrefixKvCacheMode::select(budget, shed,
  cache)` answers `Cached` when both fit, `Prefill` when only the shed demand fits (the request
  is admitted and runs without the cache), and `None` when neither does (the caller refuses as it
  always did). Nothing shrinks an image, drops a reference or changes a tier.
* **What ran.** `Generator::generate_with_report` → `GenerationReport::prefix_kv_cache`: the path
  the denoise loop took (`cached` / `prefill`) and the bytes its caches held, written from the
  loop's own `DenoiseReport`.

**Every cache figure above is derived — exact tensor arithmetic over the shapes the cache stores,
held against a real cache on the miniature fixture — and none is measured on a checkpoint.** The
peak bound was checked on the CPU device at fixture scale only. The epic's terminal story
measures one cached ten-reference render.

## Deliberate divergences

* **Turbo: `steps` is a live knob (sc-25900).** Upstream ignores `num_inference_steps` whenever
  the pipeline saves `sample_sigmas`, so the distilled checkpoint always runs its eight saved
  steps. Here a step count other than the saved length `K` **resamples the saved curve**
  (`scheduler::resample_saved_sigmas`): the knots are the saved sigmas `s_0 … s_{K−1}` followed by
  `s_K = 0`, at `u_j = j / K`; step `i` of `N` takes the piecewise-linear interpolation at
  `u = i / N`; a `0` is appended. In integers: `t = i·K`, `j = t div N`, `r = t mod N`,
  `σ_i = s_j + (s_{j+1} − s_j)·(r / N)` in `f64`, rounded to `f32` at the end. `N = K` reproduces
  the saved list exactly (upstream's schedule), `σ_0` is the first saved sigma, the list strictly
  descends and ends on `0`. This is a SceneWorks choice with no upstream counterpart — a resampled
  schedule is off the distillation's training grid, and the candle twin implements the same
  arithmetic so both lanes render the same schedule for the same request.
* Prefix KV cache: extract is keyed on the first model evaluation rather than loop index 0, and a
  cached step skips the prefix projections upstream computes and discards — see *Prefix KV cache*.
* Prefix KV cache default (sc-25905): upstream runs the cache unless told otherwise
  (`use_kv_cache=True`); here a request that selects nothing runs the **prefill** path, and the
  cache is selected per request by the caller that priced it — see *Prefix KV cache* above.
* Latents stay f32 between Euler steps (upstream rounds to bf16 each step).
* The text tower runs f32 activations over bf16 weights (upstream: bf16 end to end).
* Noise is MLX-seeded (`mlx.random.normal` under `key(seed)`), not torch-seeded.
* RGB emission composites the RGBA decode over white — the `OutputChannels::Rgb` default.
  `OutputChannels::Rgba` emits the four-channel decode unflattened as `RgbaImage`
  (sc-24111); see *Native transparency* above.
