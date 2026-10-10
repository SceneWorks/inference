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

## Generation training coverage table (sc-25685)

The backend-neutral half (request surface, plan, positional randomness, schedule, data walk and
preprocessing, Muon routing, random-init law, artifact schemas, checkpoint layout) is
`gen_core::iris::train`; the MLX half is `src/train/`. Oracle: `../tools/dump_iris_train.py`
(frozen `train/*.py`, `flow/*.py`, `seeding.py` sha256-pinned; Dion `microsoft/dion@58d38adb`, the
commit upstream's `pyproject.toml` pins). The release ships no downstream training loop; generation
training follows `train/trainer.py`. Upstream trains no adapters: LoRA/LoKr are the standard PEFT /
LyCORIS parameterizations over the frozen upstream Linears, under the same objective and loop.

| Upstream behaviour | Source | MLX (`src/train`, `gen_core::iris::train`) | Candle | Status |
| --- | --- | --- | --- | --- |
| Rectified-flow loss: `x_t = (1−σ)x0 + σε`, target `ε − x0`, per-sample MSE then mean; v- or x-prediction (`v̂ = (x_t − x̂0)/max(σ, x_pred_sigma_min)`) | `flow/transport.py` | `model::flow_loss` | S8 | ported; `loss_v`/`loss_x` fixtures 3e-7 |
| 1000-point shifted training grid, integer-truncated model time; `shift_law` none/sd3/flux | `flow/schedule.py` | `TrainSchedule`, `resolution_shift` | S8 | ported (f64, exact values pinned) |
| Logit-normal / uniform timestep index sampling | `flow/timesteps.py` | `TimestepSampler` (from `TrainingConfig::timestep_type`) | S8 | ported law; draws from the host positional RNG (not torch's stream) |
| Positional seeding `mix_seed(seed, rank, epoch, position)` for dropout/timesteps/noise; caption choice `mix_seed(seed, epoch, idx)` | `seeding.py`, `trainer.py`, `datasets.py` | `mix_seed` (values pinned), `batch_seed`, `caption_seed`, `HostRng` | S8 (same host draws ⇒ same samples) | ported; RNG stream is SplitMix64/Box–Muller, identical on both backends |
| CFG caption dropout → the `null("")` states AND mask per dropped row | `trainer.py` | `Run::batch` | S8 | ported; mutation-tested |
| Frozen Qwen3-VL conditioning, encoded on the fly | `trainer.py` | `text_conditioning: on_the_fly` (default) or the bit-identical `cached` memo | S8 | ported; on-the-fly == cached bit for bit |
| `on_caption_overflow` warn / error / silent | `text/qwen3_vl.py` | `TextSource::encode` (option `on_caption_overflow`) | S8 | ported |
| Fixed-square data policy: shortest-side PIL bicubic resize, center crop, `[−1, 1]` | `data/datasets.py` | `preprocess_image` (PIL-exact `resize_bicubic_u8`) | S8 | ported (bucket/area policies: S11) |
| Single-device ranged walk (`chunks = min(640, n)`, tail beyond `chunks·⌊n/chunks⌋` unassigned, last partial batch kept) | `data/samplers.py` | `DataWalk` | S8 | ported (note: 641–1279 items cover only 640) |
| `caption_field` / `caption_fields` (uniform among present fields) | `data/datasets.py` | `select_caption`, item `model_options.captions` | S8 | ported |
| Gradient accumulation (`loss / grad_accum`; windows counted globally across epochs, accelerate) | `trainer.py` + accelerate | `Window` | S8 | ported; oracle-checked |
| `clip_grad_norm_(0.5)` | `trainer.py` | `optim::clip_grads` | S8 | ported |
| EMA `p_ema = d·p_ema + (1−d)·p`, updated from the **pre-step** weights | `train/ema.py`, `trainer.py` | `optim::ema_update` in `Window::update` | S8 | ported; mutation-tested |
| AdamW (betas, eps 1e-8, decoupled wd, bias correction) | `train/optim.py` | `IrisOptimizer` (`Slot::Adam`) | S8 | ported; rel ≤ 1e-2 (near-zero-gradient elements: f32 noise × m/√v) |
| Hybrid Muon: hidden matrices orthogonalized (bf16 quintic Newton–Schulz, Nesterov μ = 0.95, `rms_norm` lr adjust, fused QKV / shared adaLN cores split per row block with per-block lr scales), AdamW for embeddings/heads/vectors/boundary matrices | `train/optim.py` + Dion `Muon` | `IrisOptimizer` (`Slot::Muon`), `full_param_route` | S8 | ported; rel ≤ 3e-2 (bf16 NS rounding); low-rank adaLN-core updates bounded by per-block norms (see `train_parity`) |
| Parameters that get no gradient (the discarded text tail of a final dual block under `keep`) are skipped | torch `grad is None` | `ParamRoute::Frozen` | S8 | ported |
| `LambdaLR` constant / cosine, warmup ramping from **0** | `train/lr.py` | `lr_factor` | S8 | ported (linear: refused) |
| `scale_lr` auto_lr none/sqrt/linear | `train/optim.py` | `scale_lr` | S8 | ported |
| Mixed precision bf16 autocast over f32 masters / `no` | accelerate | `TrainingConfig::train_dtype` bf16 / f32 (fp16 refused) | S8 | ported (the provider's autocast policy, traced casts) |
| Random init (`initialize_weights`: xavier patch embed, N(0,.02) timestep MLP, zero head, adaLN-zero, kaiming Linear, ones norms, N(0,1) text positions) | `models/dit.py` | `init_kind`, `random_init` | S8 | ported law (MLX RNG) |
| Weights-only start (`load_from`) | `trainer.py` | `init: weights` / `load_from` | S8 | ported |
| Checkpoint: model + EMA + optimizer + scheduler + step + epoch + data position + RNG, atomic temp+rename, `latest` | `train/ckpt.py` | `CheckpointState`, `publish_checkpoint` | S8 | ported; cancel+resume bit-exact (full/Muon, LoRA/accum) |
| Retention `keep_last_checkpoints` + `milestone_steps` | `train/ckpt.py` | `prune_checkpoints` | S8 | ported |
| `resume_data_policy` exact / new_phase, `override_lr_on_resume` | `trainer.py` | `check_resume`, `Run::prepare` | S8 | ported |
| `nan_loss_tolerance` (drop the window, abort past the tolerance) | `trainer.py` | `Run::execute` | S8 | ported |
| Validation samples from the training state (seeded, 100 steps, CFG 3 at the stage size) | `trainer.py` `_render_validation` | `TrainingProgress::Sample` (steps / cfg / prompts from the request, `preview_weights` raw or EMA) | S8 | ported (upstream renders raw; EMA selectable) |
| Export: `config.yaml` (model/text_encoder/flow) + `model.safetensors` (EMA, else raw; fp32 or bf16) | `scripts/export_checkpoint.py` | `Run::export`, `export_config_yaml` | S8 | ported; loads in the provider |
| `activation_checkpointing`, REPA/iREPA, bucket/area shapes, stage presets, frozen-grid holdout validation, distributed training, fp16 | `trainer.py` etc. | refused (typed) | — | S11 |

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
