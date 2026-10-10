# Frozen upstream — Iris-3B (epic sc-25678)

Every numeric leaf in this crate mirrors one of the pinned sources below; the constants in
`src/lib.rs` (`UPSTREAM_*`) carry the same values and a unit test asserts they agree with this
file. Python appears only in the offline oracle (`../tools/dump_iris_*.py`), never at runtime.

| What | Where | Revision |
| --- | --- | --- |
| Model / pipeline / text conditioning / solver code | `speridlabs/iris-3b` (GitHub) | `a8d15239dea469aba042cfa56ca3bb4e450d5ebc` |
| Generation weights + `config.yaml` (also `depth/`, `upscaler/` — later stories) | `speridlabs/iris-3b` (Hugging Face) | `7445443349bc9abe3c96f01ff793e2098ca012b3` |
| Text encoder: tokenizer, `config.json`, both safetensors shards | `Qwen/Qwen3-VL-4B-Instruct` (Hugging Face) | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Paper | arXiv 2610.09450v1 | — |

Licences: Iris-3B code and weights Apache-2.0; Qwen3-VL-4B-Instruct Apache-2.0. See `NOTICE`.

## Task resources (E4 task identity)

The generation task is **backbone + Qwen3-VL**. The text encoder is a separate resource — never
baked into the backbone loader — so the depth and restoration tasks (later stories) load the same
backbone code without it.

```
weights (LoadSpec::weights, a directory)      = the HF repo root (generation task)
  config.yaml                                  model / text_encoder / flow sections (read + validated)
  model.safetensors                            IrisDiT state dict, FP32 on disk, upstream key names
components["text_encoder"] (a directory)       = the Qwen/Qwen3-VL-4B-Instruct snapshot
  config.json                                  text_config read (36 layers, 2560 wide, …)
  tokenizer.json                               the pinned tokenizer (prefix/caption/suffix ids)
  model.safetensors.index.json + model-0000N-of-00002.safetensors   language tower only
                                               (`model.language_model.*`; the vision tower is never read)
```

A missing/incomplete resource is a typed load error that names the path and the task.

## Generation coverage table (source → native)

Later stories extend this table (depth, restoration, full control surface S3). The Candle column is
sc-25680 (`crates/media/candle-gen/candle-gen-iris`), which reads the same `gen_core::iris` contract
and is held to the same fixtures.

| Upstream control / behaviour | Source | MLX (`mlx-gen-iris`) | Candle (`candle-gen-iris`) | Status |
| --- | --- | --- | --- | --- |
| Chat template prefix/suffix (system "Describe the image…", user turn, assistant marker) | `text/qwen3_vl.py` `_PROMPT_PREFIX`/`_PROMPT_SUFFIX` | `text_encoder.rs` `PROMPT_PREFIX`/`PROMPT_SUFFIX` | `text_encoder.rs` (same `gen_core::iris` constants) | ported verbatim; prefix, caption, suffix tokenized **separately** (`add_special_tokens=False`) |
| 300-token conditioning window, caption truncated to `max_length - len(suffix)`, suffix always kept | `_run`, `caption_budget` | `assemble_window` | `assemble_window` (shared) | ported; overflow golden |
| Right padding with `pad_token_id` (fallback `eos`), mask = 1 over caption + suffix | `_run` | `assemble_window` | same (runs only the real tokens) | ported; the tower is causal and the pads trail, so real rows never see a pad and pad rows are zeroed by the mask — the port runs only the real tokens and writes zeros (the pad id cannot affect the output) |
| `on_caption_overflow` (`warn`/`error`/`silent`) | `TextEncoderConfig` | — | — (same release default) | release default `warn` (truncate) honoured; the knob is not exposed |
| 12 post-block hidden states `[2,5,…,35]` (1-based), sliced to the window, pad rows zeroed | `_run`, `hidden_layers` | `IrisTextEncoder::encode` | `IrisTextEncoder::encode_window` | ported; index law asserted by the oracle |
| Qwen3-VL language tower (GQA 32/8×128, q/k RMSNorm, SwiGLU, θ = 5e6; interleaved mRoPE collapses to 1-D for text) | transformers `Qwen3VLTextModel` | the shared generic decoder `mlx_llm::CausalLm` (`qwen3_vl` architecture, HF-ordered hidden states) | the shared generic decoder `candle_llm::CausalLm` (`qwen3_vl`, HF-ordered hidden states); bf16 on CUDA, f32 on the Candle CPU lane (no CPU half GEMM) | reused; bf16, the release's `text_encoder.dtype` |
| CFG null = the negative prompt run through the **same** template (empty ⇒ training dropout null) | `null()` | `encode` of the negative prompt | `pipeline::encode` | ported (the on-disk null cache is a pure memo, not behaviour) |
| Layerwise text adapter: 2 per-token layer-attention blocks (32 heads, MLP×1.3, no qk-norm), `layer_pool` Linear(12→1), refiner Linear + 2 masked blocks + RMSNorm | `LayerwiseTextEmbedder`, `TransformerTextEmbedder` | `dit.rs` `LayerwiseTextEmbedder` | `dit.rs` `TextAdapter` | ported; key mask OR diagonal |
| Learned text position table `y_pos_embedding` | `IrisDiT.forward` | `dit.rs` | `dit.rs` | ported |
| Patch embed (`unfold` p=16, channel-major patch vectors) + timestep embed (period 10, 256 freqs, cos‖sin) | `PatchEmbedder`, `TimestepEmbedder` | `dit.rs` | `dit.rs` | ported |
| Hybrid trunk: 8 dual-stream MM-DiT + 16 single-stream; GQA 20/5; sigmoid attention gate; sandwich RMSNorm; shared-bias adaLN (img/txt cores, single-stream on the img core); text-first joint attention | `blocks/mmdit.py`, `blocks/single_stream.py`, `nn/modulation.py` | `dit.rs` | `dit.rs` (GQA by interleaved `repeat_kv`; i32-safe budgeted SDPA) | ported |
| Image RoPE: axial 2D, isotropic `[0,16]` span, interleaved (x,y) pairs, complex-pair rotation; text RoPE 1-D θ = 1e4 | `nn/rope.py` | `dit.rs` `rope_2d`/`rope_1d`/`apply_rope` | `nn.rs` `Rope` (fp32 rotation) | ported (fp32 rotation) |
| Trunk attention is **unmasked** (pad text keys participate) | `JointAttention`, `SingleStreamBlock` | `dit.rs` | `dit.rs` | ported as-is (source fidelity) |
| Timestep re-fusion `silu(t_emb + s)` | `IrisDiT.forward` | `dit.rs` | `dit.rs` | ported |
| PiT pixel head: per-pixel Linear(3→16) + full-resolution 2D sincos, 4 post-modulation blocks (pixel-wise adaLN, patch compaction 256·16→1280, 10-head RoPE attention over the patch grid, GELU MLP), RMSNorm + Linear(16→3), `fold` | `blocks/pit.py`, `PixelEmbedder`, `FinalLayer` | `dit.rs` | `dit.rs` (`gelu_erf`) | ported |
| Flow schedule: `σ = 1 − linspace(1, 0.001, N+1)`, shift `s·σ/(1+(s−1)σ)` with `flow.shift` = 4, model time `1000·t` | `flow/schedule.py`, `FlowDPMSolver.time_grid` | `solver.rs` `time_grid` (f64) | `gen_core::iris::time_grid` (shared, f64) | ported |
| FlowDPM-Solver++ order 2 multistep, lower-order warm-up and final (terminal step = exact x0 projection), v-prediction `x0 = x − t·v` | `flow/solver.py` | `solver.rs` | `solver.rs` (shared plan; FP32 state) | ported; FP32 integration state |
| CFG on the raw model output, `uncond + s·(cond − uncond)`, gated strictly inside `cfg_interval` (0, 1) | `_model_out` | `solver.rs` | `solver.rs` `cfg_combine` | ported; interval fixed at the release default |
| `steps` (default 100), `cfg_scale` (default 3.0), `negative_prompt` (default ""), `seed`, `width`/`height` (default 1024², multiples of 16) | `scripts/sample.py`, `SampleConfig` | `GenerationRequest` `steps`/`guidance`/`negative_prompt`/`seed`/`width`/`height` | same `GenerationRequest` fields (`model.rs`) | honoured |
| `order` (1..), `shift` override, `cfg_interval`, `prediction="x"`, `--set` config overrides | `scripts/sample.py` | — | — (same refusals, `reject_unhonored_generation_controls`) | not exposed in S1 (fixed at release values; S3 widens the surface) — never accepted-and-ignored |
| Output `clamp(−1, 1)` → `[0, 255]` (torchvision `save_image(normalize, value_range=(−1,1))`) | `generate`, `sample.py` | `pipeline.rs` `to_image` | `pipeline.rs` `to_image` (nearest-even rounding) | ported (`round(255·(x+1)/2)`) |
| Noise `torch.randn` on the generator device | `generate` | seeded MLX normal (repo convention) | launch-portable CPU `StdRng` (`candle_gen::seed`) — not bit-reproducible with torch or MLX; parity uses injected noise | **not bit-reproducible** with torch RNG; parity is measured with injected noise |

## Precision

Upstream samples on CUDA under `torch.autocast(bfloat16)` over FP32 parameters (and in plain FP32 on
CPU). The default native compute (`Precision::Bf16`) mirrors autocast: matmuls and attention in
bf16, RMSNorm/RoPE/adaLN biases/residual streams in f32 exactly where torch's type promotion leaves
them, the solver state in f32. `Precision::Fp32` is upstream's CPU path (f32 everywhere).

The Candle twin (`candle-gen-iris`) applies the same policy op for op on CUDA (bf16 GEMM and
attention, f32 norms/RoPE/residuals/solver; candle has no implicit promotion, so `nn.rs` promotes
explicitly exactly where torch does). The Candle CPU backend has no half-precision GEMM, so on CPU a
bf16 matmul/attention runs as bf16-rounded operands with f32 accumulation and one rounding of the
result, and the Qwen3-VL tower runs f32 (the `candle_llm` CPU policy).

## Fixtures and tolerances

`tests/fixtures/` is produced by `../tools/dump_iris_golden.py` and `../tools/dump_iris_tokenizer.py`
(shared setup and the upstream source sha256 pins in `../tools/_iris_common.py`) on miniature seeded
configs that keep every architectural switch of the released `config.yaml`. The text encoder is
driven through upstream's real `Qwen3VLTextEncoder.__init__` over a miniature
`Qwen3VLForConditionalGeneration` + WordLevel tokenizer snapshot (`tests/fixtures/tiny-snapshot/`),
which is also the snapshot the generator-contract test loads through the catalog.

| Gate | Native | Tolerance (of peak) | Measured (MLX) | Measured (Candle CPU) | Why |
| --- | --- | --- | --- | --- | --- |
| Template ids, window, truncation, masks | — | exact | exact | exact | integer logic |
| Release 12-of-36 selected layers, pad rows zeroed (`text_parity`) | bf16 tower | 4e-2 | 1.7e-2–2.5e-2 | 2.0e-2–2.5e-2 (f32 tower on the Candle CPU lane — no half GEMM — so this is upstream's own bf16-vs-fp32 distance) | both sides bf16; MLX vs torch-CPU rounding compounded over 36 blocks (upstream's own bf16-vs-fp32 tower: 2.2e-2) |
| Layerwise adapter + backbone + pixel head (`dit_parity`) | FP32, MLX CPU stream | 1e-4 | 5.5e-7 / 2.8e-5 | 9.0e-7 / 1.8e-5 | summation order only (Metal f32 GEMM is reduced precision, so the f32 gate runs on the CPU stream) |
| Same, release bf16 autocast policy | bf16, GPU | 5e-2 | 2.7e-2 | 2.2e-2 (bf16 operands, f32 accumulation on CPU) | bf16 matmul/attention |
| Solver trajectory, 7 steps, CFG, shift (`solver_parity`) | f32 | 1e-5 | 8.8e-8 | 0 (bit-identical) | identical f32 coefficients |
| 100-step default grid | f64 | 1e-15 abs | exact | exact | f64 on both sides |
| End to end, 6 steps, CFG 3 (`e2e_parity`) | bf16 tower + FP32 backbone | 6e-2 | 3.1e-2 | 4.0e-2 (f32 tower, see `text_parity`) | tower rounding amplified by CFG (upstream's own bf16-vs-fp32 tower: 4.0e-2) |

`tests/fixtures/iris_tokenizer_ids.json` pins the real Qwen3-VL tokenizer's prefix / suffix / caption
ids for a prompt battery; the ignored real-weight test checks the loaded tokenizer against it.
