# Frozen upstream — FLUX 3 Image Klein on Candle (epic sc-26233, story sc-26235)

`candle-gen-flux3` is the Candle (Windows/Linux CUDA) twin of `mlx-gen-flux3`. The coverage table,
the resources layout, the precision policy and the fixtures are documented once, in the MLX twin's
[`UPSTREAM.md`](../../mlx-gen/mlx-gen-flux3/UPSTREAM.md); this file records the pinned sources the
Candle port is accountable to and the deviations that are Candle's own. Every numeric leaf in this
crate mirrors one of the pinned sources below; the constants in `gen-core/src/flux3.rs` carry the
same values and a unit test asserts they agree with this file. Python appears only in the offline
oracle (`crates/media/mlx-gen/tools/dump_flux3_*.py`), never at runtime.

| What | Where | Revision |
| --- | --- | --- |
| Inference code (`dit.py`, `autoencoder.py`, `text_encoder.py`, `sampling.py`, `pipeline.py`) | `black-forest-labs/flux-3-image-klein` (GitHub) | `45e57c9fb06f9751cdced29a65c5d2171003be05` |
| Distilled package: `config.json`, `model.safetensors` (5.6B flow-map DiT, bf16), `ae.safetensors` (f32), `text_encoder/` | `black-forest-labs/flux-3-image-klein` (Hugging Face, gated) | `8454017f9ebaada30c998fccd9f67a80c7580f68` |
| Base CFG package (S6): native DiT without target-time embedder; 50 steps, CFG 3.5 | `black-forest-labs/flux-3-image-klein-base` (Hugging Face, gated) | `1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4` |
| The bundled `text_encoder/` is byte-identical to this snapshot (same LFS sha256 of both shards, same tokenizer and config) | `Qwen/Qwen3-VL-4B-Instruct` (Hugging Face) | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |

Licences: the inference code is Apache-2.0; the FLUX 3 weights (DiT + autoencoder) are under the
**FLUX Kommunity License v1.0**; the text encoder is Apache-2.0 (Qwen). See the MLX twin's `NOTICE`
and the catalog's licence rows (`gen_core::license`, `candle-gen-catalog/src/licenses.rs`).

## Coverage (S2 = the distilled text-to-image route, Candle column)

Every row of the MLX twin's coverage table marked S1 is ported here one-to-one, against the same
`gen_core_flux3` contract and the same committed goldens:

| Upstream behaviour | Candle (`candle-gen-flux3`) |
| --- | --- |
| Chat template, clamp 4096, pad-80 window, pad positions = 1, padded causal attention | `text_encoder.rs` over `candle_llm::CausalLm::hidden_states_with_mask_at` (explicit positions + additive mask) |
| Conditioning = the 8 hidden states concatenated per token → `[L, 20480]` bf16 | `text_encoder.rs` |
| Autoencoder: 8× KL VAE + 2×2 patchify + BatchNorm-stat latents, f32 | `autoencoder.rs` (NCHW; every 3×3 conv through `candle_gen::conv2d_budgeted`, the mid attention through the shared i32-safe SDPA) |
| Image ids, 4-axis RoPE tables, timestep embedding, per-mode modulation, per-mode + single-stream blocks, `LastLayer` | `dit.rs` + `nn.rs` (`gen_core_flux3` tables) |
| Flow-map Euler, `dt · velocity` rounded to the compute dtype, f32 state | `pipeline.rs` (written against `gen_core_flux3::flow_map_schedule`, not the shared sigma-convention sampler driver) |
| Output `clamp(−1, 1)` → RGB8 (truncating cast) | `pipeline::float_image_to_rgb8` |
| Default seed `sha256(caption)` when the request carries none | `gen_core_flux3::prompt_seed` |
| Base CFG50/3.5, explicit negative conditioning, additive identity-checked LoRA | `pipeline::denoise_cfg`, `adapters.rs`, native `AdaptLinear` residuals (S6) |
| Optional prompt upsampler | separate `pup.rs` model and shared four-mode contract (S5) |
| Ordered 1–10 RGB/RGBA references; single Lanczos fit/reflect-or-symmetric padding, multi-reference fit/centre crop; optional output-size intent and single auto output crop | `gen_core_flux3::references`, `references.rs`, immutable `image_cond` in both flow-map and both base CFG samples (S3); generation Qwen encoding stays text-only |
| Source EXIF/RGB conversion versus PUP alpha-over-white fork | caller-authorized asset decode preserves oriented RGBA; AE drops alpha, existing PUP sees original RGBA; no raw-path generation-request carrier |
| Boxes, quantized variants | typed `Unsupported` refusals (sc-26237, sc-26241) |

## Deliberate differences from the MLX twin

* `backend = "candle"`, `mac_only = false`; the same generator id `flux3_klein` and family `flux3`.
* The autoencoder runs in **NCHW** (candle's native conv layout); the MLX twin runs NHWC. The
  packed-channel law `c·4 + ph·2 + pw` and the row-major token order are the same.
* **Noise** is the shared launch-portable CPU `StdRng` draw (`candle_gen::seed`), rounded through
  bf16 as upstream's `torch.randn(...).to(bfloat16).float()`. Not bit-compatible with torch's or
  MLX's RNG; parity is measured with injected noise on every backend.
* **Attention math runs in f32 on every backend** over compute-dtype (bf16) q/k/v, rounding once
  at the output — the `q·kᵀ` logits, the softmax and `p·v`. Candle has no fused SDPA, and a
  composed bf16 path that rounds the logits to bf16 before the softmax is measurably noisier than
  upstream's own bf16 autocast (7.2 % vs 2.5 % of peak on the real DiT at 64²); the fused kernels
  upstream runs in production (cuDNN) and the MLX twin's Metal SDPA both keep the logits and the
  softmax in f32.
* **Precision on the Candle CPU backend** (no half-precision GEMM): the bf16 policy runs each GEMM
  in f32 over bf16-rounded operands and rounds the result to bf16 — what a bf16 GEMM computes —
  and the Qwen3-VL tower runs in f32 (`candle_llm`'s CPU rule, the conditioning still cast to bf16
  as upstream). On CUDA the GEMMs and the tower are the real bf16 kernels, and the CUDA lane holds
  them to the same fixtures.
* The memory contract's calibration fingerprint is Candle's own
  (`flux3-klein-candle-derived-…`): the two backends never share calibration evidence.

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

The parity tests read the MLX twin's committed fixtures
(`crates/media/mlx-gen/mlx-gen-flux3/tests/fixtures/`): the miniature package `tiny-snapshot/`,
`flux3_text_golden.safetensors`, `flux3_dit_golden.safetensors`, `flux3_ae_golden.safetensors`,
`flux3_e2e_golden.safetensors`, and the real-weight CPU reference `flux3_real_golden_64.safetensors`
read by the `#[ignore]`d `real_weights::` cases.

### Prompt-upsampler oracle versus native device behavior (sc-26683)

`tests/pup_parity.rs` runs all four frozen `flux3_pup_golden.safetensors` decoder cases through
the real native loader on **CPU/F32**, including reference tokenization, public-token projection,
locked boxes, stopping and raw-token accounting. The fixture generator
`tools/dump_flux3_pup_golden.py` in the MLX tree computes those IDs and per-step margins with
Torch CPU/F32. An isolated subprocess sets the existing `CANDLE_LLM_DEVICE=cpu` override;
the original 0.02 margin allowance and minimum matching prefix are unchanged. The override is
never applied to production or mutated in the parent test process.

The separate `default_backend_executes_all_modes_with_ordered_references_and_locked_boxes`
test uses the build-selected native loader (BF16 on CUDA/Metal, F32 on CPU). It executes all four
modes, exact rendered prompts and prompt/reference token counts, ordered reference images,
deterministic greedy adapter generation, ordered content progress, exact locked rows, and typed
pre-cancellation. It does **not** compare BF16 token identities to the F32 oracle. A CPU/F32
oracle margin is not a confidence certificate for a BF16 argmax; the MLX mixed-precision oracle
also does not certify Candle CUDA arithmetic. The native loader's compute dtype follows its
device; `core_llm::LoadSpec` has no F32-on-CUDA override (`quantize` controls weight storage).
Numeric GPU parity requires a separately dtype/backend-matched oracle, not wider tolerances or
re-pinning the CPU golden.
