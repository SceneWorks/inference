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

## Deliberate divergences

### From upstream (shared with the MLX twin)

* No prefix KV cache: every step evaluates the full block-causal joint sequence (upstream's exact
  `QwenImage21AttnProcessor` prefill path). Upstream documents the cached and uncached paths as
  equally valid but not bit-identical.
* Latents stay f32 between Euler steps (upstream rounds to bf16 each step).
* The text tower runs f32 activations (upstream: bf16 end to end).
* Noise is seeded from the shared launch-portable CPU `StdRng` (`candle_gen::seed`), not torch's
  generator; seed parity across frameworks is not a goal.
* RGB emission composites the RGBA decode over white until gen-core carries an RGBA surface
  (sc-24111); `QwenImage21Vae::decode_rgba` is the four-channel path.

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
* Compute dtype is the backend's: f32 on CPU (the parity lane), bf16 on CUDA/Metal — where MLX
  always loads at the checkpoint's on-disk dtype.
