# Frozen upstream — FLUX 3 Image Klein (epic sc-26233)

Every numeric leaf in this crate mirrors one of the pinned sources below; the constants in
`gen-core/src/flux3.rs` carry the same values and a unit test asserts they agree with this file.
Python appears only in the offline oracle (`../tools/dump_flux3_*.py`), never at runtime.

| What | Where | Revision |
| --- | --- | --- |
| Inference code (`dit.py`, `autoencoder.py`, `text_encoder.py`, `sampling.py`, `pipeline.py`) | `black-forest-labs/flux-3-image-klein` (GitHub) | `45e57c9fb06f9751cdced29a65c5d2171003be05` |
| Distilled package: `config.json`, `model.safetensors` (5.6B flow-map DiT, bf16), `ae.safetensors` (f32), `text_encoder/` | `black-forest-labs/flux-3-image-klein` (Hugging Face, gated) | `8454017f9ebaada30c998fccd9f67a80c7580f68` |
| Base CFG package (S6): native DiT without target-time embedder; 50 steps, CFG 3.5 | `black-forest-labs/flux-3-image-klein-base` (Hugging Face, gated) | `1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4` |
| The bundled `text_encoder/` is byte-identical to this snapshot (same LFS sha256 of both shards, same tokenizer and config) | `Qwen/Qwen3-VL-4B-Instruct` (Hugging Face) | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Prompt upsampler code (`pup.py`, `bbox.py`, `pup_prompts.json`; sc-26238) | `black-forest-labs/flux-3-image-klein` (GitHub) | `45e57c9fb06f9751cdced29a65c5d2171003be05` (`pup.py` sha256 `e3561ae3…`, `bbox.py` `fbea10c1…`) |
| Prompt upsampler package: `config.json` (`qwen3_5`), `model.safetensors` (9B VLM, bf16, 18.8 GB, sha256 `f5062a1f…`), tokenizer + `chat_template.jinja`, `system_prompts.json` (sha256 `b5da3ed8…`, identical to `pup_prompts.json`) | `black-forest-labs/flux-3-image-klein-pup` (Hugging Face, ungated, Apache-2.0) | `8b069751eb5dcab23829d9c5eccb470eb8b8a4ed` |

`model.safetensors` sha256 `c3e9813cdd049c831c836b2e9db61150e49a93fc1a4c63b256494642fee0bd7c`;
`ae.safetensors` sha256 `868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3`
(both the HF LFS objects; the DiT's is also in the package's `manifest.json`).

Licences: the inference code is Apache-2.0; the FLUX 3 weights (DiT + autoencoder) are under the
**FLUX Kommunity License v1.0** (`LICENSE.md` of the package — non-commercial weights, a
commercial-output exception for Qualifying Users under US$5M revenue, content-filter and
attribution obligations); the text encoder is Apache-2.0 (Qwen). See `NOTICE` and the catalog's
licence rows (`gen_core::license`).

S6 registers the base checkpoint separately as `flux3_klein_base`; S3 adds ordered reference conditioning, while structured boxes remain sc-26237. The prompt upsampler is sc-26238 (the
`pup` module here and `gen_core::flux3::pup`): an **optional** co-requisite, loaded only when the
`prompt_upsampler` component names its package and accounted separately from the image model and
its text encoder.

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
components["prompt_upsampler"] (a directory)   optional (sc-26238); the flux-3-image-klein-pup package root
  config.json, model.safetensors, tokenizer.json, tokenizer_config.json, chat_template.jinja,
  preprocessor_config.json, system_prompts.json (optional: the embedded copy is the fallback)
```

The upsampler is never read from the image package: it is a separate 18.8 GB model, a second
resident beside the DiT when loaded, and a request with `enhance_prompt` on a generator loaded
without it is a typed refusal (never a silent pass-through).

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
| CFG sampling (base checkpoint): [unconditional, conditional], default 50/3.5, no target-time; explicit negative caption independently encoded | `sampling.denoise` | `pipeline::denoise_cfg`, shared request resolver | S6 |
| Additive LoRA with explicit finite scale and base/revision/rank/alpha/target admission | frozen diffusers converter at `7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941` | `gen_core::flux3::adapters`, `adapters.rs`, existing native residual hosts | S6 |
| Ordered 1–10 RGB/RGBA references: single ≤4 Mi-pixel padded canvas (Lanczos, centred reflect/symmetric), multi each ≤1 Mi-pixel with centred 16px crop; first-reference auto size and single auto output crop only when neither dimension was explicit | `pipeline.prepare_reference`, `prepare_multi_reference`, `default_size`, `generate.py` | `gen_core::flux3::references`, typed `GenerationRequest::flux3_reference_output`; scope verifies admission against resolved geometry before normalizing numeric defaults | S3 |
| Reference AE tokens cast bf16, ordered negative l-axis IDs, spatial axes ×0.5; identical image_cond in negative/positive base rows even CFG0/1 | `pipeline.encode_references`, `pack`, both samplers | `references.rs`, `pipeline::*_with_references`; generation Qwen text encoding remains text-only, not VLM/deepstack | S3 |
| Original EXIF-oriented RGBA forks before AE RGB-drop-alpha versus existing PUP white-composite policy; invalid count/pixels/strength rejected before generation | `pipeline.prepare_reference`, `pup.policy_reference` | family-only authorized-file decoder, existing `ReferenceRgba`/ordered `Reference` carriers, existing `UpsamplerInput` modes | S3 |
| Boxes, quantized variants | `bbox.py`, `fp8.py`/`nvfp4.py` | — | sc-26237, sc-26241 |
| **Prompt upsampler (sc-26238)** — four modes by system prompt (`t2i`, `i2i`, `t2i_bbox`, `i2i_bbox`): `bbox.plan` (passed boxes → canvas rows; a table already in the prompt locks by itself), the `ORIGINAL PROMPT:` / `LOCKED BOUNDING-BOX ANNOTATIONS` input, the two `I2I_SCAFFOLD`s, the chat template with `enable_thinking=False`, greedy `top_p 0.95 / top_k 20 / max_tokens 3584`, presence penalty 1.5 (t2i) / 0.0 (i2i), `max_model_len 8192`, `bbox.reconcile` (IoU ≥ 0.9 rename, locked rows verbatim adopting the desc, valid cited rows appended), `generate.py`'s `--no-pup` caption and finished-answer skip | `pup.py`, `bbox.py` | `gen_core::flux3::pup` (tensor-neutral, byte-for-byte text), `pup.rs` (the engine) | S5 |
| Upsampler reference policy: EXIF transpose, alpha over white, `reference_size` budget (1 MP / 724² / 512² by reference count, multiples of 32, aspect ≤ 200:1 else the prompt as typed), Pillow `LANCZOS`, JPEG q92 | `pup.policy_reference` | `gen_core::flux3::pup::policy_reference` (bit-exact Pillow RGB/alpha/LANCZOS and libjpeg's q92 integer 4:2:0 round trip, pixel-equality fixtures) | S5 |
| Qwen3.5 VLM serving (hybrid linear/full attention, Qwen3-VL ViT, `<\|image_pad\|>` expansion by the processor's `smart_resize`) | vLLM | shared `mlx_llm` provider (`mlx-llama`, `qwen3_5`) | S5 |

## Precision policy

Upstream runs the DiT under CUDA autocast(bf16) over bf16 weights: linears, attention and the
residual streams are bf16; `F.layer_norm` and the q/k RMSNorm + RoPE run in f32; the Euler state is
f32 (the noise is `.float()`) with `dt · velocity` multiplied at f32 opmath and its product
rounded to bf16 before the add (the Python-float `dt` itself is never rounded); the timestep `1000·t` is formed in bf16.
The port mirrors this at `Precision::Bf16` (the default) and runs everything in f32 at
`Precision::Fp32` (the oracle's CPU path, which has no autocast). The autoencoder is f32 on both.

## Reference output envelope (S3)

Frozen `generate.py:65..77` requires positive 16-pixel-grid output sides; `pipeline.MIN_SIDE`
(64) is the **single-reference padding** floor, not the multi-reference or decoder kernel floor.
Reference-derived 16/32/48-pixel sides are therefore accepted, including after the memory scope
normalizes the numeric request canvas. Plain no-reference requests retain their existing 64-pixel
minimum.

Both native providers retain their existing advertised **4096-pixel output-side ceiling**
(`model::MAX_SIZE`). This is an explicit native allocation-envelope deviation, not an upstream
renderer or kernel limit: larger explicit canvases or very wide multi-reference canvases are
refused before generation rather than expanding that public envelope without validation.

## Fixtures

`tests/fixtures/tiny-snapshot/` is a miniature package in the released layout
(`config.json`, `model.safetensors`, `ae.safetensors`, `text_encoder/`), produced by
`../tools/dump_flux3_golden.py` (DiT, text encoder, sampler, end-to-end goldens) and
`../tools/dump_flux3_ae_golden.py` (autoencoder golden) from the frozen source with seeded
miniature widths (`../tools/_flux3_common.py`). Real-weight goldens
(`../tools/dump_flux3_realweight.py`, CPU torch over the pinned package at a 64×64 output) are
read by the `#[ignore]`d `real_weights::` cases from `FLUX3_REAL_GOLDEN`.

`tests/fixtures/tiny-pup-snapshot/` (sc-26238) is a miniature upsampler package — a seeded
`Qwen3_5ForConditionalGeneration` (4 hybrid layers, a 2-layer ViT with no deepstack, as the
release) with a WordLevel tokenizer carrying the PINNED `chat_template.jinja`, `system_prompts.json`
and processor configs verbatim — and `flux3_pup_golden.safetensors` its greedy run over the four
modes (expanded prompt ids, the reference pixels after upstream's policy, generated ids, top-2
logit margins), produced by `../tools/dump_flux3_pup_golden.py`, which also writes the text
goldens and the Pillow policy fixture read by `gen-core/tests/flux3_pup.rs`. The real-weight case
`real_weights::real_prompt_upsampler_rewrites_the_readme_request` reads the pinned package from
`FLUX3_PUP_DIR`.

## S6 adapter identity and mapping

`LoadSpec::adapters` carries the explicit user scale. Safetensors metadata must name
`base_model` (the canonical base or distilled Hub repo) and its frozen `base_revision`. An explicit
finite positive global alpha is read from native `alpha` / `lora_alpha` / `ss_network_alpha` or
standard JSON `lora_adapter_metadata`; duplicate declarations must agree, never fall back to rank.
Every declared native/PEFT rank must match the actual factors. Plain native and diffusers A/B
adapters are supported; rsLoRA/DoRA, fan-in/fan-out and nonempty rank/alpha patterns are typed
refusals rather than silently applying the wrong gain. Only complete `lora_A.weight` /
`lora_B.weight` pairs are admitted; all tensors must map, and LoKr/pass/expert overrides fail typed.
The mapping is the explicit inverse of diffusers' pinned FLUX3 converter, not FLUX2 aliases:
`mode_blocks` ↔ `content_mode_blocks`, `single_transformer_blocks` ↔ `single_blocks`,
`txt_mode_blocks` unchanged; `attn.to_q/to_k/to_v/to_out.0` ↔ `q_proj/k_proj/v_proj/attn_out`,
`ff.linear_in/linear_out` ↔ `mlp_in/mlp_out`. Residual scale is user scale × alpha/rank.

`base-hub-identity.json` records pinned API metadata: base DiT LFS SHA256
`5c921759dc03663dbb67df11d8e3d317d2d5b893b31b969c95dc77ebf861b19c`, the same AE LFS object
as distilled, and all fourteen identical encoder git blob ids. LICENSE.md git blob id
`044cf48568c278150d132a7479ea237a79766ea7` also equals distilled. These are metadata identity
claims, not successful base-file download or locally hashed weight receipts; actual pinned
base config access returned HTTP403 for the authenticated account on 2026-10-16.
`dump_flux3_base_golden.py` runs only CPU tiny fixtures through the hash-verified frozen source:
CFG0/1/3.5, nonempty negative conditioning, and six-projection additive adapter on/off.
MLX requires hosted macOS execution; Windows CPU fixture success is not MLX verification.
