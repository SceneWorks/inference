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

## Installable tiers (sc-24112)

Qwen-Image 2.1 ships **pre-quantized**: a tier is a complete standalone snapshot in the layout above
whose weight-bearing components are already affine-quantized, produced offline by
`convert::prequantize_turnkey` and loaded with no dense transient. `LoadSpec::quantize` selects a
tier; it is a transform request only against a dense snapshot.

| component | bf16 | q8 | q4 |
|---|---|---|---|
| `transformer/` (7.12 B params in 232 Linears) | dense bf16 | packed Q8, group 64 | packed Q4, group 64 |
| `text_encoder/` Qwen3 language tower (6.95 B params in 252 decoder Linears) | dense bf16 | packed Q8, group 64 | **packed Q8**, group 64 |
| `text_encoder/` token embedding (622 M) | dense | dense | dense |
| `text_encoder/` `lm_head`, `model.visual.*` | dense, **not loaded** by this route | dense | dense |
| `vae/` (338 M) | dense f32 | dense f32 | dense f32 |

Three decisions, each deliberate:

* **The text encoder is packed**, unlike the 2512 `mlx-gen-qwen-image` crate. 2512 pairs a ~20 B DiT
  with a ~7 B tower, so a dense tower is a minority of its footprint. 2.1 pairs a **7.12 B** DiT with
  a **7.57 B** tower: under `Sequential` the resident floor is `max(tower, DiT + VAE)`, so a dense
  tower would pin that floor at ~14.5 GiB at every tier and the Q4 tier would buy ~0.3 GiB over Q8.
  Reusing 2512's table here would have produced a tier that cannot do its job.
* **The Q4 tier holds the tower at Q8** — declared through
  `Capabilities::component_precision_floors`, never silent. That floor is a *prior* carried from
  `mlx_gen_mage`'s measured Qwen-LM-tower sweep (a Q4 SwiGLU MLP collapses generation quality; Q8
  attention + MLP holds), not a measurement of this model. Confirming or retiring it on real weights
  belongs to the epic's terminal measurement story.
* **Group size 64, uniformly.** A tier declares exactly one `quantization.group_size`, and the
  packed bit-width is derived from the packed shapes *at the group size the loader passes*, so a tier
  mixing group sizes would be decoded at the wrong width. A `Linear` narrower than 64 therefore stays
  dense. The released geometry has no such width (its four DiT input widths are 64 / 4096 / 4096 /
  12288 and the tower's are 4096 / 12288), so in production the converter packs everything; only the
  miniature parity snapshot is partly dense.

The converted artefacts are **byte-reproducible** (keys are sorted before serialization and the
config merge is deterministic) and each packed triple is **byte-identical to `mlx_rs::ops::quantize`**
over the bf16 source — the same op the load-time quantizer runs. `tests/tiers.rs` pins both, and
`tests/fixtures/tiers/` carries the committed miniature tiers the Candle backend reads.

### Producing a tier

```sh
QWEN21_SRC=<dense snapshot> QWEN21_TIER=q4 QWEN21_DST=~/SceneWorks/qwen-image-2-1-tiers \
  cargo run --release --example qwen_image_2_1_prequant -p mlx-gen-qwen-image-2-1
```

The example prints a SHA-256 manifest of everything it wrote.

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
preset **area** (2400×1792, which is neither the widest preset nor the square default), the latent
tokens it contributes, and the fact that each of up to 10 reference images adds a full block of image
tokens to the joint sequence.

## Deliberate divergences

* No prefix KV cache: every step evaluates the full block-causal joint sequence (upstream's exact
  `QwenImage21AttnProcessor` prefill path). Upstream documents the cached and uncached paths as
  equally valid but not bit-identical.
* Latents stay f32 between Euler steps (upstream rounds to bf16 each step).
* The text tower runs f32 activations over bf16 weights (upstream: bf16 end to end).
* Noise is MLX-seeded (`mlx.random.normal` under `key(seed)`), not torch-seeded.
* RGB emission composites the RGBA decode over white until gen-core carries an RGBA surface
  (sc-24111); `QwenImage21Vae::decode_rgba` is the four-channel path.
