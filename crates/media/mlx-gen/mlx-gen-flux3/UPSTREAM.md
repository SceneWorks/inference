# Frozen upstream — FLUX 3 Image Klein (epic sc-26233)

Every numeric leaf in this crate mirrors one of the pinned sources below; the constants in
`gen-core/src/flux3.rs` carry the same values and a unit test asserts they agree with this file.
Python appears only in the offline oracle (`../tools/dump_flux3_*.py`), never at runtime.

| What | Where | Revision |
| --- | --- | --- |
| Inference code (`dit.py`, `autoencoder.py`, `text_encoder.py`, `sampling.py`, `pipeline.py`) | `black-forest-labs/flux-3-image-klein` (GitHub) | `45e57c9fb06f9751cdced29a65c5d2171003be05` |
| Distilled package: `config.json`, `model.safetensors` (5.6B flow-map DiT, bf16), `ae.safetensors` (f32), `text_encoder/` | `black-forest-labs/flux-3-image-klein` (Hugging Face, gated) | `8454017f9ebaada30c998fccd9f67a80c7580f68` |
| The bundled `text_encoder/` is byte-identical to this snapshot (same LFS sha256 of both shards, same tokenizer and config) | `Qwen/Qwen3-VL-4B-Instruct` (Hugging Face) | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |

`model.safetensors` sha256 `c3e9813cdd049c831c836b2e9db61150e49a93fc1a4c63b256494642fee0bd7c`;
`ae.safetensors` sha256 `868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3`
(both the HF LFS objects; the DiT's is also in the package's `manifest.json`).

Licences: the inference code is Apache-2.0; the FLUX 3 weights (DiT + autoencoder) are under the
**FLUX Kommunity License v1.0** (`LICENSE.md` of the package — non-commercial weights, a
commercial-output exception for Qualifying Users under US$5M revenue, content-filter and
attribution obligations); the text encoder is Apache-2.0 (Qwen). See `NOTICE` and the catalog's
licence rows (`gen_core::license`).

The base checkpoint (`flux-3-image-klein-base`), the prompt upsampler (`flux-3-image-klein-pup`),
reference images and boxes are later stories of the epic (sc-26239, sc-26238, sc-26236/7); nothing
here pins them.

## Source authority

The executable source is the authority wherever it and the package's `config.json` disagree: the
source never reads `config.json` (its widths are module constants). The one known disagreement is
the text window — `text_encoder.py` clamps the prompt at **4096** tokens, `config.json` says
`max_tokens: 8192` — and the port applies the source's clamp
(`gen_core::flux3::TextEncoderConfig::window_max_length`). The native graph reads its widths from
`config.json` so the miniature fixtures load through the same code as the release; the release
`config.json` is asserted against the source constants in `gen-core`.

## Resources (E5)

```
weights (LoadSpec::weights, a directory)      = the HF package root
  config.json                                  dit / text_encoder / vae / sampler sections (read + validated)
  model.safetensors                            DiT state dict, bf16 on disk, upstream key names
components["text_encoder"] (a directory)       optional; default <root>/text_encoder (config `text_encoder.path`)
  config.json, tokenizer.json, tokenizer_config.json
  model.safetensors.index.json + model-0000N-of-00002.safetensors   language tower only
components["autoencoder"] (a file)             optional; default <root>/ae.safetensors (config `vae.path`)
```

A missing/incomplete resource is a typed load error that names the path and the resource. A
quantized variant (`variants/fp8r`, `variants/nvfp4`; `config.json` `quantization != null`) is a
typed refusal until sc-26241.

## Coverage table (source → native, S1 = distilled text-to-image)

| Upstream behaviour | Source | MLX (`mlx-gen-flux3`) | Status |
| --- | --- | --- | --- |
| Chat template (single user turn + generation prompt, no system turn) tokenized as one string; clamp 4096; right-pad with `<\|endoftext\|>` to a multiple of 80; mask; pad positions = 1 | `text_encoder.py` | `gen_core::flux3::assemble_window`, `text_encoder.rs` | S1 |
| Qwen3-VL language tower, bf16, padded forward with the pad keys masked for every query (pad queries attend the real keys at position 1) | transformers `Qwen3VLForConditionalGeneration` | shared `mlx_llm::CausalLm` (`qwen3_vl`), explicit positions + additive mask | S1 |
| Conditioning = the 8 hidden states `[4, 8, …, 32]` (HF numbering) concatenated per token → `[L, 20480]` bf16; ids `(0, 0, 0, pos)` | `text_encoder.py` | `text_encoder.rs` | S1 |
| Autoencoder: 8× KL VAE + 2×2 patchify + BatchNorm-stat latents, f32; decode = denorm → unpatchify → decoder | `autoencoder.py` | `autoencoder.rs` (delegated leaf) | S1 |
| Image ids `(0, row·0.5, col·0.5, 0)` | `pipeline.scaled_ids` | `gen_core::flux3::scaled_image_ids` | S1 |
| 4-axis RoPE tables in f64 → f32; adjacent-pair rotation after f32 RMSNorm of q/k | `dit.rope`, `dit.rmsnorm_rope` | `gen_core::flux3::rope_cos_sin`, `dit.rs` | S1 |
| Timestep embedding (×1000 in the conditioning dtype, 256-d cos‖sin), `time_in` MLP, `ref_time_in` target-time MLP (flow map) | `dit.timestep_embedding`, `MLPEmbedder` | `dit.rs` | S1 |
| Per-mode modulation (`early_stream_modulations`, `single_stream_modulations`): shift/scale/gate from `silu(vec)` | `ModeModulation` | `dit.rs` | S1 |
| 5 per-mode blocks (image / image_cond / text), then 28 shared single-stream blocks over the per-sample `[image, image_cond, text]` concatenation; per-sample attention bounds; parallel attention + SwiGLU MLP with one gate | `Block`, `DiT.forward` | `dit.rs` | S1 (image_cond empty: the phantom token is never attended and never placed, so the stream is skipped — identical output) |
| `LastLayer`: adaLN (shift, scale) → LayerNorm → linear to 128 channels | `LastLayer` | `dit.rs` | S1 |
| Flow-map Euler: `x += dt · dit(x, t, t_ref = schedule[i+1])` over `timesteps(steps, 6.93)`; bf16-valued f32 noise, f32 state | `sampling.py`, `pipeline.generate` | `pipeline.rs` | S1 |
| Noise `torch.randn(...).to(bf16).float()` | `pipeline.generate` | seeded MLX normal, rounded through bf16 (not bit-reproducible with torch; parity uses injected noise) | S1 |
| Output `clamp(−1, 1)` → RGB8 | `pipeline.to_pil` | `mlx_gen::image::decoded_to_image` | S1 |
| Default seed `int(sha256(caption).hexdigest(), 16) % 2**32` when the request carries none | `generate.py` | `gen_core::flux3::prompt_seed` | S1 |
| CFG sampling (base checkpoint) | `sampling.denoise` | — | sc-26239 |
| Reference images (`image_cond`), boxes, upsampler, quantized variants, LoRA | `pipeline.encode_references`, `bbox.py`, `pup.py`, `fp8.py`/`nvfp4.py` | — | sc-26236/7/8, sc-26241, sc-26239 |

## Precision policy

Upstream runs the DiT under CUDA autocast(bf16) over bf16 weights: linears, attention and the
residual streams are bf16; `F.layer_norm` and the q/k RMSNorm + RoPE run in f32; the Euler state is
f32 (the noise is `.float()`) with `dt · velocity` multiplied at f32 opmath and its product
rounded to bf16 before the add (the Python-float `dt` itself is never rounded); the timestep `1000·t` is formed in bf16.
The port mirrors this at `Precision::Bf16` (the default) and runs everything in f32 at
`Precision::Fp32` (the oracle's CPU path, which has no autocast). The autoencoder is f32 on both.

## Fixtures

`tests/fixtures/tiny-snapshot/` is a miniature package in the released layout
(`config.json`, `model.safetensors`, `ae.safetensors`, `text_encoder/`), produced by
`../tools/dump_flux3_golden.py` (DiT, text encoder, sampler, end-to-end goldens) and
`../tools/dump_flux3_ae_golden.py` (autoencoder golden) from the frozen source with seeded
miniature widths (`../tools/_flux3_common.py`). Real-weight goldens
(`../tools/dump_flux3_realweight.py`, CPU torch over the pinned package at a 64×64 output) are
read by the `#[ignore]`d `real_weights::` cases from `FLUX3_REAL_GOLDEN`.
