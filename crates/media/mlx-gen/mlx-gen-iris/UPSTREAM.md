# Frozen upstream — Iris-3B (epic sc-25678)

Every numeric leaf in this crate mirrors one of the pinned sources below; the constants in
`src/lib.rs` (`UPSTREAM_*`) carry the same values and a unit test asserts they agree with this
file. Python appears only in the offline oracle (`../tools/dump_iris_*.py`), never at runtime.

| What | Where | Revision |
| --- | --- | --- |
| Model / pipeline / text conditioning / solver code | `speridlabs/iris-3b` (GitHub) | `a8d15239dea469aba042cfa56ca3bb4e450d5ebc` |
| Generation weights + `config.yaml`; the `depth/` export (sc-25682); `upscaler/` (later story) | `speridlabs/iris-3b` (Hugging Face) | `7445443349bc9abe3c96f01ff793e2098ca012b3` |
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

The **depth** task (`iris_3b_depth`, sc-25682) is its own closure — the `depth/` folder only, no text
encoder (`gen_core::iris::downstream::TaskExport`):

```
weights (LoadSpec::weights, a directory)      = the HF repo's depth/ folder
  config.yaml                                  model / text_encoder / flow + `task: {name: depth}`
  model.safetensors                            `pixel.*` (the widened IrisDiT) + `depth_reducer.*`, FP32
  empty_prompt.safetensors                     `embeddings` [1, 300, 12, 2560] F32 + `mask` [1, 300] BOOL
```

Any staged component (e.g. `text_encoder`), `LoadSpec::text_encoder`, the generation checkpoint (a
`config.yaml` without a `task` section) or another task's export (`task.name != depth`) is a typed
`Unsupported` refusal naming the wrong-task artifact.

## Generation coverage table (source → native)

Later stories extend this table (depth, restoration, full control surface S3). The Candle column is
sc-25680 (`crates/media/candle-gen/candle-gen-iris`), which reads the same `gen_core::iris` contract
and is held to the same fixtures.

| Upstream control / behaviour | Source | MLX (`mlx-gen-iris`) | Candle (`candle-gen-iris`) | Status |
| --- | --- | --- | --- | --- |
| Chat template prefix/suffix (system "Describe the image…", user turn, assistant marker) | `text/qwen3_vl.py` `_PROMPT_PREFIX`/`_PROMPT_SUFFIX` | `text_encoder.rs` `PROMPT_PREFIX`/`PROMPT_SUFFIX` | `text_encoder.rs` (same `gen_core::iris` constants) | ported verbatim; prefix, caption, suffix tokenized **separately** (`add_special_tokens=False`) |
| 300-token conditioning window, caption truncated to `max_length - len(suffix)`, suffix always kept | `_run`, `caption_budget` | `assemble_window` | `assemble_window` (shared) | ported; overflow golden |
| Right padding with `pad_token_id` (fallback `eos`), mask = 1 over caption + suffix | `_run` | `assemble_window` | same (runs only the real tokens) | ported; the tower is causal and the pads trail, so real rows never see a pad and pad rows are zeroed by the mask — the port runs only the real tokens and writes zeros (the pad id cannot affect the output) |
| `on_caption_overflow` (`warn`/`error`/`silent`) | `TextEncoderConfig`, `_tokenize_captions` (also `--set text_encoder.on_caption_overflow=…`) | config value, overridden per request by `GenerationRequest::caption_overflow`; `gen_core::iris::apply_caption_overflow_policy` in `IrisTextEncoder::encode_with_policy` | same (`text_encoder.rs`) | ported (sc-25681): `warn` truncates, logs and returns a `caption_truncated` `GenerationWarning` (`generate_with_report`); `silent` truncates quietly; `error` refuses the request before the tower runs. Applies to the negative prompt too (upstream's `null()` shares `_tokenize_captions`). Any other config value is refused at load. Upstream rate-limits its log line to once per 100 calls — a logging cadence, not behaviour; the port reports every truncation |
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
| FlowDPM-Solver++ multistep, lower-order warm-up and final (terminal step = exact x0 projection) | `flow/solver.py` | `solver.rs` | `solver.rs` (shared plan; FP32 state) | ported; FP32 integration state |
| `prediction`: `v` (`x0 = x − t·out`) or `x` (`x0 = out`) — a checkpoint property | `FlowConfig.prediction`, `_pred_x0` | `config.yaml` `flow.prediction` → `gen_core::iris::Prediction` → `solver::sample` | same | ported (sc-25681); any other value is refused at load (upstream raises) |
| CFG on the raw model output, `uncond + s·(cond − uncond)`, gated strictly inside `cfg_interval` | `_model_out` | `solver.rs` `cfg_combine`, `GenerationParams::cfg_at` | same | ported |
| Output `clamp(−1, 1)` → `[0, 255]` (torchvision `save_image(normalize, value_range=(−1,1))`) | `generate`, `sample.py` | `pipeline.rs` `to_image` | `pipeline.rs` `to_image` (nearest-even rounding) | ported (`round(255·(x+1)/2)`) |
| Noise `torch.randn` on the generator device | `generate` | seeded MLX normal (repo convention) | launch-portable CPU `StdRng` (`candle_gen::seed`) — not bit-reproducible with torch or MLX; parity uses injected noise | **not bit-reproducible** with torch RNG; parity is measured with injected noise |

### Generation control surface (sc-25681)

Every public argument of `iris3b.sampling.generate` / `scripts/sample.py`, its request field, and the
native computation it changes. "Test" names the fixture case (`tests/controls_parity.rs`, both
backends, against `iris_controls_golden.safetensors`) and the provider-level check
(`tests/controls_contract.rs::every_advertised_control_changes_the_render`, both backends).

| Upstream control (default) | Request field | MLX | Candle | Test | Notes |
| --- | --- | --- | --- | --- | --- |
| `prompts: list[str]` | `prompt`, or `prompt_batch` (non-empty ⇒ `prompt` empty) | `pipeline::encode` per prompt → one batched `denoise` (`[B, …]` text states, per-row masks, null expanded over the batch in `cat([uncond, cond])`) | same | `batch` | `prompt_batch.len() × count ≤ 8` images; image `k = c·B + j` draws noise from seed `k` (MLX `seed + k`, Candle `image_seed(seed, k)`), so a batched image equals its single render up to batched-GEMM rounding (`a_prompt_batch_renders_one_image_per_prompt_per_count`, FP32 ≤ 1 code value) |
| `height` / `width` (1024²) | `height` / `width` (multiples of 16, 16..=2048) | noise geometry, patch grid, 2-D RoPE grid, PiT sincos table | same | `portrait` (12×8) vs `base` (8×12) | aspect ratio is the pair; no separate aspect control upstream |
| `steps` (100) | `steps` | `dpm_solver_plan(steps, …)` | same | every case (5 steps) | |
| `order` (2) | `sampler`: `dpmpp_2m` = order 2 (default), `euler` = order 1 | `GenerationParams::order` → plan | same | `order1` | order 1 is the first-order DPM-Solver++ update `x ← (t/s)·x + ((s−t)/s)·x0` on every step — exactly the flow Euler (= DDIM) step (`gen_core::iris` `the_order_one_plan_is_the_euler_step`). Upstream runs the 2M update for every `order ≥ 2`, so 3+ is not a distinct solver and has no name; any other sampler name is refused |
| `cfg_scale` (3.0) | `guidance` | `cfg_combine`; `1.0` skips the unconditional branch and its encode | same | `negative` (4.0), `cfg_off` (1.0) | |
| `cfg_interval` ((0, 1)) | `cfg_interval: Option<(f32, f32)>` | `GenerationParams::cfg_at` | same | `interval` (0.3, 0.8) | `0 ≤ lo < hi ≤ 1` (model time `t` ∈ (0, 1)); an interval holding none of the plan's evaluation times while guidance ≠ 1 is refused (the guidance would be accepted and never run), and any interval at guidance 1.0 is refused (CFG is off, so the interval would do nothing) |
| `shift` (checkpoint `flow.shift`) | `scheduler_shift` | `time_grid(steps, shift)` | same | `shift2` | `> 0`; default is the checkpoint's own `flow.shift` |
| `negative_prompt` ("") | `negative_prompt` | encoded through the same template as the unconditional | same | `negative` | refused at guidance 1.0 (never evaluated) |
| `generator` seed | `seed` | MLX seeded normal | `candle_gen::seed` CPU `StdRng` | `every_advertised_control_changes_the_render` | not bit-compatible with `torch.randn` (parity runs on injected noise) |
| `noise` | — | — | — | — | test-only injection (`denoise` takes the noise tensor) |
| `num_train_timesteps`, `prediction` | — (checkpoint `flow.*`) | read from `config.yaml` | same | `prediction_x` | checkpoint properties, not request controls |
| `--set KEY=VALUE` config overrides | — | the inference-relevant keys are the rows above (`sample.*` → request fields; `flow.shift` → `scheduler_shift`; `text_encoder.on_caption_overflow` → `caption_overflow`); architecture keys are the checkpoint's and are validated, never overridden | same | — | an override of an architecture key would mis-load the checkpoint upstream too |
| `flow.shift_law` (`none`/`sd3`/`flux`) | — | accepted, inert at inference | same | `gen_core::iris` `unsupported_switches_are_typed_refusals` | `scripts/sample.py` samples with `flow.shift` verbatim whatever the law — the law only resolves the *training* stage shift (`RectifiedFlow`). Unknown laws are refused (upstream's `resolution_shift` raises) |
| `sample:` section of a checkpoint config | — | ignored | same | — | upstream's `inference_config` keeps only `model`/`text_encoder`/`flow` (`INFERENCE_SECTIONS`); `SampleConfig` defaults always apply |

Refused by name (`gen_core::iris::reject_unhonored_generation_controls` and the shared floor): every
other request field — `true_cfg`, `timestep_to_start_cfg`, `guidance_method` and the APG knobs,
`scheduler` names, `strength`, conditioning inputs, video/audio fields, PiD, phases.

### Adapters (sc-25681)

Upstream Iris-3B ships no adapter code; LoRA/LoKr use the repository's adapter conventions, keyed by
the upstream `IrisDiT` module path (the checkpoint key stem, e.g. `blocks.3.attn_proj`,
`y_embedder.refiner.proj`, `pixel_blocks.0.fc1`).

| Concern | MLX (`src/adapters.rs`) | Candle (`src/adapters.rs`) |
| --- | --- | --- |
| Application | forward-time residual on the projection (`mlx_gen::adapters::AdaptableLinear`, `apply_adapters_strict`), base never mutated; LoKr via the structured Kronecker product `w1·X·w2ᵀ` | `W += δ` folded into the f32 weight at the safetensors-key level before the compute-dtype cast (`candle_gen::train::merge` convention); LoKr `δ = scale·(alpha/rank)·kron(w1, w2)` |
| Formats | PEFT/diffusers LoRA (`transformer.` / `diffusion_model.` / bare; `lora_A/B`, `lora_down/up`; per-target `.alpha` or `lora_adapter_metadata`), kohya `lora_unet_…`, PEFT-stamped LoKr (`networkType=lokr`), LyCORIS-layout LoKr/LoHa factors — key layouts only: every one of them must also carry the identity stamps below | same set |
| Identity | `gen_core::iris::check_adapter_identity`: `family=iris` and `irisTask=<task>` **required**, `baseModel` must match when present — the task backbones share one architecture, so only the stamp tells a depth adapter from a generation one. Required for **every** format: a file exported by a third-party trainer (kohya, LyCORIS) carries neither stamp and is refused until it is re-stamped; unstamped third-party files are not supported | same |
| Strictness | a target that resolves to no projection, a file that lands nothing, a diff-patch (`.diff`/`.diff_b`) file, per-pass scales / MoE expert → typed error | same, plus a delta whose shape differs from its projection |
| Provenance | `Generator::adapter_apply_reports()` — one `AdapterApplyReport { adapter_path, applied, skipped }` per file, from the most recent backbone load | same |

Fixtures: `iris_lora.safetensors` / `iris_lokr.safetensors` (four targets: text adapter, a dual-stream
block, a single-stream block, the pixel head) and `iris_lora_depth_task.safetensors` (refused);
`iris_adapter_golden.safetensors` holds upstream's merged-weight forwards and the expected deltas.

### Live step preview (sc-25681)

Iris denoises in pixel space, so the preview needs no latent→RGB fit: each solver step emits the
step's predicted clean image `x0` (`FlowDPMSolver._pred_x0`, CFG-combined) of the batch's first row,
average-pooled over each patch cell (the backbone's own token grid — 64×64 for a 1024² render) and
decoded with the exact pixel map `(x + 1)/2` (`pipeline::preview_image`, both backends, unit-tested
for exactness). The last frame is the output (the terminal update is the exact x0 projection).
Upstream has no preview; nothing in its semantics makes `x0` misleading — it is the same estimate
the solver integrates. Both descriptors advertise `supports_preview: true`; an inert sink costs one
branch per step.

## Depth coverage table (source → native, sc-25682)

Source: `src/iris3b/downstream/{__init__,depth}.py`, `scripts/depth.py`, `scripts/export_downstream.py`
at the pinned commit. Host-side steps live once in `gen_core::iris::depth` (both backends call
`estimate_with`); the tensor forward is `mlx_gen_iris::depth` / `candle_gen_iris::depth`.

| Upstream control / behaviour | Source | MLX (`mlx-gen-iris`) | Candle (`candle-gen-iris`) | Status |
| --- | --- | --- | --- | --- |
| Export layout + task identity (`task.name == "depth"`), strict key set (`pixel.*` + `depth_reducer.*`) | `load_export`, `export_downstream.py` | `depth::load_depth` over `gen_core::iris::downstream::TaskExport` | same | ported; wrong-task artifacts are typed refusals; no text encoder in the closure (E4) |
| Empty-prompt conditioning `[1, T, L, D]` F32 + BOOL mask, shape-checked against the config | `load_export`, `DepthPredictor.__init__` | `EmptyPrompt::read` (shared) | same | ported |
| Widened input projections (`s_embedder.proj` p²·4, `pixel_embedder.proj` 4 → 16), output stays 3 channels | `IrisDepth._widen` | `IrisDiT::from_weights_widened` | `IrisDiT::from_checkpoint_widened` | ported |
| Input = RGB ‖ zero channel, t = `flow.num_train_timesteps` (1000), one forward, empty-prompt mask | `IrisDepth.forward` | `IrisDepth::forward` | `IrisDepth::forward` | ported |
| `depth_reducer` 1×1 conv 3 → 1 (+ bias) | `IrisDepth.depth_reducer` | `addmm` in the compute dtype | `Linear::from_parts` in the compute dtype | ported (autocast runs the conv in bf16; FP32 path in f32) |
| EXIF orientation (`ImageOps.exif_transpose`) + `convert("RGB")` | `DepthPredictor.__call__` | — | — | the **caller's** job (an `Image` carries no EXIF; the SceneWorks half orients before calling) |
| `max_side` (default 1024; 0 = native): `scale = min(1, max_side / max(w, h))`, never upscale | `__call__`, `scripts/depth.py --max-side` | `DepthResolution::{Capped(n), Native}` (default `Capped(1024)`; `Capped(0)` refused — use `Native`) | same (shared) | ported |
| Patch-grid size law `max(p, round(side·scale/p)·p)`, Python round-half-even | `__call__` | `plan_depth_input` | same | ported (`round_ties_even`) |
| Lanczos resize to the model size (only when it changes), PIL 8-bit fixed-point | `image.resize(LANCZOS)` | `gen_core::imageops::resize_lanczos_u8` | same | ported, **bit-exact** (fixture gate: exact) |
| `rgb / 255 * 2 - 1` in f32 | `__call__` | `prepare_depth_input` | same | ported, exact |
| Bilinear resize of the prediction back to the source size (`align_corners=False`, no antialias), only when resized | `F.interpolate` | `interpolate_bilinear` | same | ported |
| Output: raw `[H, W]` f32 relative log depth, -1 near … +1 far, not clamped/normalized/metric | `__call__` docstring | `DepthMap` + `DepthMetadata` (model/config revision, preprocessing, value convention) | same | ported (E5) |
| `colorize` (near = bright over the map's own 2nd–98th percentiles, `inferno`) | `colorize` | `near_bright_unit` / `near_bright_control_image` (control adapter) / `colorize_inferno` (preview) — pure, never mutate the map | same (shared) | ported (matplotlib's 256-entry inferno LUT) |
| `.npy` + `.png` writing, `--out`, batch of paths | `scripts/depth.py` | — | — | consumer (SceneWorks) persistence, not inference |
| BF16 autocast on CUDA, FP32 elsewhere | `__call__` | `Precision::Bf16` (default) / `Precision::Fp32` | same | ported (the default is the CUDA release policy on both backends) |

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
| Every generation control, 9 cases (`controls_parity`, `tools/dump_iris_controls.py`) | bf16 tower + FP32 backbone (MLX); f32 tower + FP32 backbone vs the oracle's **fp32-tower** render (Candle) | 6e-2 (MLX) / 2e-4 (Candle) | 8.4e-3–4.5e-2 | 1.1e-5–5.8e-5 | MLX: tower rounding amplified by CFG (upstream's own bf16-vs-fp32 tower: 1.7e-2–7.3e-2 across the cases); every control's golden sits ≥ 0.15 from `base`, so an ignored control fails |
| LoRA / LoKr / both on the backbone (`adapter_parity`) | FP32 | 2e-3 (MLX) / 1e-4 forward, 1e-6 delta (Candle) | 2.5e-5 / 8.8e-4 / 8.6e-4 | forwards 2.4e-5 / 3.3e-5; deltas ≤ 1.5e-8 | MLX's shared LoKr path reconstructs its factors in bf16 (PARITY-BF16); each adapter moves the output by 0.11–0.44 |

The oracle renders the fp32-tower references with a **separate** null-embedding cache directory:
upstream memoizes `null("")` on disk keyed by repo, window and layers but not by dtype, so a shared
directory hands an fp32 tower the bf16 null.

`tests/fixtures/iris_tokenizer_ids.json` pins the real Qwen3-VL tokenizer's prefix / suffix / caption
ids for a prompt battery; the ignored real-weight test checks the loaded tokenizer against it.

### Depth fixtures and tolerances (sc-25682)

`tests/fixtures/tiny-depth/` + `iris_depth_golden.safetensors` come from `../tools/dump_iris_depth.py`
(upstream's real `DepthPredictor` over a miniature export in the `export_downstream.py` layout; the
depth modules are sha256-pinned there). Cases: odd non-square native (37×23), capped (50×30 at
max side 32), below one patch (3×5), already on the grid (24×16), round-half-even ties (10×14), a
cap that would upscale (20×12 at 1024).

| Gate | Native | Tolerance (of peak) | Measured (MLX) | Measured (Candle CPU) |
| --- | --- | --- | --- | --- |
| Size law, Lanczos input, `[-1, 1]` mapping (`depth_parity`) | host f32 | exact | exact | exact |
| Raw `[1, 1, h, w]` forward, FP32 | MLX CPU stream / Candle CPU | 1e-4 | ≤1.5e-5 | ≤1.5e-5 |
| Source-size map after the bilinear resize back, FP32 | — | 1e-4 | ≤1.5e-5 | ≤1.5e-5 |
| Raw forward, release bf16 policy | GPU / CPU bf16 operands | 5e-2 | 1.1e-2 | 1.0e-2 |
| `colorize_inferno` vs upstream `colorize` | host | ≤ 4 RGB levels | within bound | — |

Real weights (`tests/depth_real_weights.rs`, ignored; `../tools/dump_iris_depth_realweight.py`): the
repo photo `_vendor/mage_flow/assets/dog.jpg` (1024×2048 → model 512×1024 at the default max side),
upstream FP32 CPU as the reference. Upstream's OWN bf16-autocast distance on that photo is recorded in
`iris_depth_real_reference.safetensors`: full-res mean 2.9e-3 / max 0.21; 16×16-pooled mean 2.6e-3 /
max 4.6e-2 / pearson 0.99999. Native MLX bf16 (Apple GPU): model input exact; full-res mean 2.5e-3 /
max 0.31 (fur edges) / pearson 0.99999; pooled mean 2.3e-3 / max 4.2e-2; from the `image`-crate JPEG
decode (not PIL) pooled mean 3.3e-3 / max 7.0e-2. Bounds: pearson ≥ 0.9995, mean ≤ 1e-2, pooled max
≤ 0.15. The Candle/CUDA job (`iris` profile) checks the same pooled reference: bf16 on `cuda:0`
(`image`-crate decode) measured pooled mean 2.8e-3 / max 0.11 / pearson 0.99999, 0.4 s estimate,
8.4 GB VRAM peak (run 38076560406).

## Restoration (`iris_3b_restore`, sc-25683)

Source: `src/iris3b/downstream/restoration.py`, `src/iris3b/downstream/__init__.py`,
`scripts/upscale.py`, `scripts/export_downstream.py` (sha256-pinned in
`../tools/dump_iris_restoration.py`). Registered as a **transform** (`gen_core::Transform`, the
non-prompt image→image contract) under `iris_3b_restore` on both backends
(`mlx-gen-iris::restoration`, `candle-gen-iris::restoration`). The host-side pixel path is ONE
backend-neutral implementation in `gen_core::iris::restoration` (planner + driver); each backend
supplies only a tile's velocity.

### Resources (E4)

```
weights (LoadSpec::weights, a directory)   = the `upscaler/` folder of speridlabs/iris-3b
  config.yaml                               parent model/text_encoder/flow sections + `task:` {name: restoration, sigma: 0.5, tile: 1024}
  model.safetensors                         the fine-tuned IrisDiT, FP32, upstream key names (strict load)
  empty_prompt.safetensors                  `embeddings` [1, 300, 12, 2560] F32 + `mask` [1, 300] BOOL
```

No text encoder is loaded or accepted: a `text_encoder` component (or any other) is a typed refusal,
as is a generation backbone (no `task` section) or a depth export (`task.name: depth`) staged as the
weights — upstream `load_export`'s task-name check. A missing file names the path.

### Coverage table (source → native)

| Upstream control / behaviour | Source | Shared (`gen_core::iris::restoration`) | MLX (`mlx-gen-iris`) | Candle (`candle-gen-iris`) | Status |
| --- | --- | --- | --- | --- | --- |
| Task identity (`task.name == "restoration"`), `sigma`, `tile` from `config.yaml` | `load_export`, `Restorer.__init__` | `downstream::TaskExport` (S4's shared export closure) + `RestorationSettings::from_task/validate` | `IrisRestorer::load` | `IrisRestorer::load` | ported; tile must be a multiple of the patch; v-prediction required (upstream's check) |
| Empty-prompt conditioning, no text encoder | `Restorer.__init__` | `downstream::EmptyPrompt::read` (shape-checked against `text_len` / `text_lap_num_layers` / `text_dim`) | `[1, T, L, D]` array, read once | same | ported |
| One-step forward at `t = sigma · num_train_timesteps` (500), restored = `x − sigma · v` in f32 | `restore_tile` | `restore_detailed` (the subtraction) | `IrisRestorer::velocity` (shared `IrisDiT`) | same | ported; bf16 compute (default) mirrors the CUDA autocast, `Precision::Fp32` = upstream's CPU path |
| `scale` (default 4.0, any positive float incl. 1×): `F.interpolate(scale_factor, bicubic, align_corners=False)` — output `floor(side · scale)`, coordinate scale `1 / scale`, Keys a = −0.75, clamped taps | `__call__` | `bicubic_scale_factor`, `plan` | shared | shared | ported (`TargetSize::Scale`; `TargetSize::ModelDefault` = 4×); min-edge / explicit resolution refused (upstream has neither) |
| Small output (short side ≤ tile): enlarged so the short side is one tile (`round` half-even per side), antialiased bicubic (a = −0.5, clipped renormalized window), resized back afterwards | `__call__` | `plan` (`processing`, `enlarged`), `bicubic_aa_resize` (horizontal pass first, an unchanged axis skipped) | shared | shared | ported |
| Zero pad of `x · 2 − 1` to the patch grid (bottom / right), crop after | `__call__` | `restore_detailed` | shared | shared | ported |
| 1024-px tiles, stride `tile // 2` (50 % overlap), last tile flush; one pass when the image fits a tile | `tile_positions`, `tiled` | `tile_positions`, `plan` (`tile_rows` / `tile_cols`) | shared | shared | ported |
| Gaussian fusion window (variance 0.01, row centre `tile / 2`, column centre `(tile − 1) / 2`), weighted average | `gaussian_window`, `tiled` | `gaussian_window`, `restore_detailed` | shared | shared | ported verbatim incl. the asymmetric centres |
| Wavelet colour fix (à-trous 3×3 binomial, dilations 1…16, replicate pad): restored high frequencies + bicubic-reference low frequencies | `wavelet_color_fix` | `wavelet_color_fix` | shared | shared | ported; `TransformRequest::color_fix` (default on, `--no-color-fix` = `Some(false)`) |
| Input budget (`fit_budget`: short ≤ 512, long ≤ 1024, never upscale, PIL Lanczos on RGB8) vs `--no-budget` | `scripts/upscale.py` | `fit_budget_dims`, `fit_budget_image` (bit-exact PIL fixed point) | shared | shared | ported; `InputSizing::Budgeted` (default, the script's) / `InputSizing::Original` — never switched implicitly |
| Output `clamp(0, 1) · 255`, `round` (half-even) → RGB8 | `__call__` | `Planes::to_rgb8` | shared | shared | ported |
| Per-tile progress; cancel between tiles | — | `restore` (`Progress::Step` per tile, `Progress::Decoding` for post-processing; `Error::Canceled` and no image) | shared | shared | native addition |
| Tiles per forward | `restore_tile` takes `[B, …]`; upstream calls it with B = 1 | one tile per forward | — | — | upstream's call pattern |
| `ImageOps.exif_transpose(...).convert("RGB")` | `__call__` | — | — | — | the consumer's job: the transform receives display-oriented RGB8. For an EXIF-rotated (90°) file the script budgets in storage orientation, so PIL's two 8-bit passes run in the other order — a ±1-level difference on such files only |
| Seed / strength / step count | — | `RestorationOptions::from_request` | — | — | not upstream controls: refused by name (`steps: 1` accepted) |

### Planner (consumer preview)

`gen_core::iris::restoration::plan_request(req, TileGeometry::RELEASE, MODEL_ID)` (or `plan(dims,
&options, …)`) returns, before any execution and without weights: source, budgeted input, output
(`floor(input · scale)`), processing size, whether it was enlarged, padded size, tile rows/columns
and `forward_count()`. A scale whose output rounds to zero pixels (or overflows) is a typed refusal;
the provider plans with the loaded export's geometry, which equals `TileGeometry::RELEASE` for the
release export.

### Fixtures and tolerances

`tests/fixtures/tiny-snapshot/upscaler/` is a miniature export written by upstream's own
`export_downstream.py restoration` (the miniature backbone perturbed, a 3-token empty prompt,
`tile` 32, `sigma` 0.5); `tests/fixtures/iris_restoration_golden.safetensors` holds the op-level and
end-to-end references (`../tools/dump_iris_restoration.py`).

| Gate | Tolerance | Measured (MLX) | Measured (Candle CPU) |
| --- | --- | --- | --- |
| `scale_factor` bicubic (×4, ×2.5 on odd sides, ×1, ×0.75, ×3) | 1e-5 of peak | ≤ 1.4e-6 | shared host code |
| Antialiased resize (up, down, one axis, odd) | 1e-5 | ≤ 7.2e-7 | shared |
| Gaussian window (32 full; 1024 centre row / column / subsample) | 1e-6 | ≤ 6.0e-8 | shared |
| Wavelet colour fix (29×37, smaller than dilation 16) | 1e-5 | 1.8e-7 | shared |
| Tile positions, budget sizes | exact | exact | shared |
| `fit_budget` PIL Lanczos bytes (7 shapes) | sha256-exact | exact | shared |
| End to end FP32, 7 cases (enlarge → resize back, multi-tile, 1×, 2.5×, single tile, portrait 3×, budgeted): fused tiles / colour-fixed float | 1e-4 of peak | ≤ 2.2e-5 / ≤ 7.2e-6 | ≤ 2.1e-5 / ≤ 6.9e-6 |
| Same, RGB8 | ≤ 1 level | ≤ 1 (at most 6 of 61 440 values differ) | ≤ 1 |

Real weights (`tests/restoration_real_weights.rs`, `../tools/dump_iris_restoration_realweight.py`):
the native bf16 path is held to 1.5× upstream's **own** bf16-autocast-vs-FP32 mean RGB8 distance on
the same input (FP32: 0.5 level mean). `tests/fixtures/iris_restoration_real_small.safetensors` is
the committed upstream reference of the small case (RGB8 only), so the CUDA job checks parity too.

| Case | Upstream bf16 vs fp32 (the bound's basis) | MLX bf16 vs upstream fp32 | Candle CUDA bf16 vs upstream fp32 |
| --- | --- | --- | --- |
| 512×384 → 2048×1536 (6 tiles), machine-local golden | max 68, mean 0.521, PSNR 47.7 dB | max 60, mean 0.423, PSNR 49.0 dB (restore 10.8 s, peak footprint 11.7 GB) | — (no Python oracle on the runner): release-scale render 512×384 → 2048×1536, 6 tiles in 2.6 s, VRAM peak 10.1 GB |
| 64×48 → 256×192 (enlarged, 2 tiles), committed | max 23, mean 0.522, PSNR 48.1 dB | max 19, mean 0.459, PSNR 48.7 dB | max 23, mean 0.701, PSNR 46.0 dB (bound 0.783; restore 1.0 s, VRAM peak 10.1 GB; run 38085270480) |
