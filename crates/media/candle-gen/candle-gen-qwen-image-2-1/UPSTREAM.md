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

`tests/fixtures/` is produced by `tools/dump_qwen21_*.py` (shared setup in
`tools/_qwen21_common.py`) from the diffusers/transformers classes at the revisions above on
miniature seeded configs, with the tiny snapshot written by `save_pretrained` in the exact layout
above. Each Rust parity test names the tolerance it holds the port to.

## Deliberate divergences

* No prefix KV cache: every step evaluates the full block-causal joint sequence (upstream's exact
  `QwenImage21AttnProcessor` prefill path). Upstream documents the cached and uncached paths as
  equally valid but not bit-identical.
* Latents stay f32 between Euler steps (upstream rounds to bf16 each step).
* The text tower runs f32 activations over bf16 weights (upstream: bf16 end to end).
* Noise is MLX-seeded (`mlx.random.normal` under `key(seed)`), not torch-seeded.
* RGB emission composites the RGBA decode over white until gen-core carries an RGBA surface
  (sc-24111); `QwenImage21Vae::decode_rgba` is the four-channel path.
