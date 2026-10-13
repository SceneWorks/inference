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

## Generation training coverage table (sc-25685)

The backend-neutral half (request surface, plan, positional randomness, schedule, data walk and
preprocessing, Muon routing, random-init law, artifact schemas, checkpoint layout) is
`gen_core::iris::train`; the MLX half is `src/train/`. Oracle: `../tools/dump_iris_train.py`
(frozen `train/*.py`, `flow/*.py`, `seeding.py` sha256-pinned; Dion `microsoft/dion@58d38adb`, the
commit upstream's `pyproject.toml` pins). The Candle half (sc-25686) is `candle-gen-iris/src/train/`,
held to the same fixture by its own `train_parity` (Candle CPU, FP32) and the same lifecycle cases. The release ships no downstream training loop; generation
training follows `train/trainer.py`. Upstream trains no adapters: LoRA/LoKr are the standard PEFT /
LyCORIS parameterizations over the frozen upstream Linears, under the same objective and loop.

| Upstream behaviour | Source | MLX (`src/train`, `gen_core::iris::train`) | Candle | Status |
| --- | --- | --- | --- | --- |
| Rectified-flow loss: `x_t = (1−σ)x0 + σε`, target `ε − x0`, per-sample MSE then mean; v- or x-prediction (`v̂ = (x_t − x̂0)/max(σ, x_pred_sigma_min)`) | `flow/transport.py` | `model::flow_loss` | `train::model::flow_loss` | ported; `loss_v`/`loss_x` fixtures 3e-7 (Candle 5e-7, also through the autograd forward) |
| 1000-point shifted training grid, integer-truncated model time; `shift_law` none/sd3/flux | `flow/schedule.py` | `TrainSchedule`, `resolution_shift` | shared contract | ported (f64, exact values pinned) |
| Logit-normal / uniform timestep index sampling | `flow/timesteps.py` | `TimestepSampler` (from `TrainingConfig::timestep_type`) | shared contract | ported law; draws from the host positional RNG (not torch's stream) |
| Positional seeding `mix_seed(seed, rank, epoch, position)` for dropout/timesteps/noise; caption choice `mix_seed(seed, epoch, idx)` | `seeding.py`, `trainer.py`, `datasets.py` | `mix_seed` (values pinned), `batch_seed`, `caption_seed`, `HostRng` | shared contract (same host draws ⇒ same samples) | ported; RNG stream is SplitMix64/Box–Muller, identical on both backends |
| CFG caption dropout → the `null("")` states AND mask per dropped row | `trainer.py` | `Run::batch` | `train::Run::batch` | ported; mutation-tested (both) |
| Frozen Qwen3-VL conditioning, encoded on the fly | `trainer.py` | `text_conditioning: on_the_fly` (default) or the bit-identical `cached` memo | `train::TextSource` (same modes) | ported; on-the-fly == cached bit for bit |
| `on_caption_overflow` warn / error / silent | `text/qwen3_vl.py` | `TextSource::encode` (option `on_caption_overflow`) | `train::TextSource::encode` | ported |
| Fixed-square data policy: shortest-side PIL bicubic resize, center crop, `[−1, 1]` | `data/datasets.py` | `preprocess_image` (PIL-exact `resize_bicubic_u8`) | shared contract | ported (bucket / area policies: below) |
| Single-device ranged walk (`chunks = min(640, n)`, tail beyond `chunks·⌊n/chunks⌋` unassigned, last partial batch kept) | `data/samplers.py` | `DataWalk` | shared contract (`unused_tail_warning` on stderr) | ported (note: 641–1279 items cover only 640) |
| `caption_field` / `caption_fields` (uniform among present fields) | `data/datasets.py` | `select_caption`, item `model_options.captions` | shared contract | ported |
| Gradient accumulation (`loss / grad_accum`; windows counted globally across epochs, accelerate) | `trainer.py` + accelerate | `Window` | `train::Window` | ported; oracle-checked (both) |
| `clip_grad_norm_(0.5)` | `trainer.py` | `optim::clip_grads` | `train::optim::clip_grads` | ported |
| EMA `p_ema = d·p_ema + (1−d)·p`, updated from the **pre-step** weights | `train/ema.py`, `trainer.py` | `optim::ema_update` in `Window::update` | `train::optim::ema_update` in `train::Window::update` | ported; mutation-tested (both) |
| AdamW (betas, eps 1e-8, decoupled wd, bias correction) | `train/optim.py` | `IrisOptimizer` (`Slot::Adam`) | `train::optim::IrisOptimizer` (`Slot::Adam`; scalar ops against f32 scalar tensors, true division) | ported; rel ≤ 1e-2 (near-zero-gradient elements: f32 noise × m/√v) |
| Hybrid Muon: hidden matrices orthogonalized (bf16 quintic Newton–Schulz, Nesterov μ = 0.95, `rms_norm` lr adjust, fused QKV / shared adaLN cores split per row block with per-block lr scales), AdamW for embeddings/heads/vectors/boundary matrices | `train/optim.py` + Dion `Muon` | `IrisOptimizer` (`Slot::Muon`), `full_param_route` | `train::optim::IrisOptimizer` (`Slot::Muon`): bf16 NS as native bf16 GEMMs on CUDA (torch's bf16 `@`), the f32-product-rounded-once form on the CPU backend (no half GEMM) — the `ns.*` fixtures match exactly on CPU | ported; rel ≤ 3e-2 (bf16 NS rounding); low-rank adaLN-core updates bounded by per-block norms (see `train_parity`) |
| Parameters that get no gradient (the discarded text tail of a final dual block under `keep`) are skipped | torch `grad is None` | `ParamRoute::Frozen` | same route (a param the graph never reads gets a zero gradient, then is skipped) | ported |
| `LambdaLR` constant / cosine, warmup ramping from **0** | `train/lr.py` | `lr_factor` | shared contract | ported (linear: refused) |
| `scale_lr` auto_lr none/sqrt/linear | `train/optim.py` | `scale_lr` | shared contract | ported |
| Mixed precision bf16 autocast over f32 masters / `no` | accelerate | `TrainingConfig::train_dtype` bf16 / f32 / fp16 (fp16: below) | candle autograd over f32 `Var` masters; the provider's casts on the tape; the fused RMSNorm / softmax kernels (no candle backward) swapped for the same math in composable ops under autograd only (`nn.rs`), each chain on the tape as ONE node (`tape::fold3`, sc-25690: candle keeps every primitive's output alive until the loss drops, so the plain chains held five to six `[B, heads, N, N]` tensors per attention and three activations per norm) — see *The folded tape* in the Candle crate's `UPSTREAM.md` | ported (the provider's autocast policy, traced casts; the folded nodes' gradients are the plain chains' bit for bit) |
| Random init (`initialize_weights`: xavier patch embed, N(0,.02) timestep MLP, zero head, adaLN-zero, kaiming Linear, ones norms, N(0,1) text positions) | `models/dit.py` | `init_kind`, `random_init` | `train::model::random_init` (host `HostRng` keyed `mix_seed(seed, fnv(key))`, device-portable) | ported law (MLX RNG / host RNG) |
| Weights-only start (`load_from`) | `trainer.py` | `init: weights` / `load_from` | same | ported |
| Checkpoint: model + EMA + optimizer + scheduler + step + epoch + data position + RNG, atomic temp+rename, `latest` | `train/ckpt.py` | `CheckpointState`, `publish_checkpoint` | same layout key for key (`backend: candle`), staged + `publish_checkpoint` (fsync) | ported; cancel+resume bit-exact (full/Muon, LoRA/accum; Candle also LoKr) |
| Retention `keep_last_checkpoints` + `milestone_steps` | `train/ckpt.py` | `prune_checkpoints` | shared contract | ported |
| `resume_data_policy` exact / new_phase, `override_lr_on_resume` | `trainer.py` | `check_resume`, `Run::prepare` | `check_resume`, `train::Run::prepare` | ported |
| `nan_loss_tolerance` (drop the window, abort past the tolerance) | `trainer.py` | `Run::execute` | `train::Run::execute` | ported |
| Validation samples from the training state (seeded, 100 steps, CFG 3 at the stage size) | `trainer.py` `_render_validation` | `TrainingProgress::Sample` (steps / cfg / prompts from the request, `preview_weights` raw or EMA) | `train::render_preview_with` (the provider's sampler and forward; an adapter preview == the exported file loaded through `load_backbone_with_adapters`, bit for bit) | ported (upstream renders raw; EMA selectable) |
| Export: `config.yaml` (model/text_encoder/flow) + `model.safetensors` (EMA, else raw; fp32 or bf16) | `scripts/export_checkpoint.py` | `Run::export`, `export_config_yaml` | `train::Run::export` (staged directory published by rename + fsync, sc-25690) | ported; loads in the provider — and cross-backend (`train_cross_backend`: Candle-trained LoRA/LoKr/full load in MLX and vice versa, FP32 forwards agree to 1e-4 for every artifact; the MLX recording was re-produced on the MLX CPU device in sc-25690, when MLX stopped rounding a LoKr delta to bf16 under FP32 compute — the artifacts reproduced byte for byte, the LoKr forward moved by the removed rounding) |
| Official stages `stage1_256 → stage2_512 → stage3_1024 → sft_1024` | `configs/*.yaml` | `upstream::STAGE_PRESETS` (the YAML vendored verbatim) → `stage_preset_config`: an editable `TrainingConfig` stamped `stage` / `parent_stage`; the run records both in `state.json`, and `check_lineage` refuses a parent from another stage | shared contract | ported (sc-25689); lifecycle chain 256 → 512 on both backends |
| Shape policies `fixed` / `bucket` (`shared21-512`, `shared21-1024`, closest ratio, first on a tie) / `area` (`shape_align`, `shape_max_ratio`) with the covering resize + center crop | `data/shapes.py`, `data/buckets.py`, `data/datasets.py` | `train::shapes::ShapePolicy`, `preprocess_to_shape` (PIL-exact bicubic) | shared contract | ported; every target and the 31×47 crop exact vs upstream |
| Ranged walk (640 canonical chunks, world-portable resume state, resharding a cursor to another world) + shape batcher (`drop_last`, per-shape batches, regroup history replayed on a micro-batch change, `limit_batches`) | `data/samplers.py` | `train::shapes::{RangedSampler, ShapeBatchSampler}`, `train::pipeline::DataPipeline` | shared contract | ported; plans and resume states equal upstream's key for key |
| `rope_aspect` square / isotropic (stage 1 trains square) | `models/dit.py` | `rope_2d_angles_aspect`, the trunk's RoPE grid | same | ported; non-square bucket step under both laws: loss 1e-5, grads 1e-4 vs upstream |
| REPA / iREPA: frozen DINOv2 teacher (`x_norm_patchtokens`, bicubic input resize, ImageNet norm, interpolated position table; `_reg` variants antialiased), MLP (REPA) / conv (iREPA) projector on block `layer`'s tokens, bilinear token-grid resize to the smaller grid, iREPA spatial normalization, `−cos` loss × `weight` | `train/repa.py` | `train::teacher::Teacher` (hub and Hugging Face naming), `model::repa_loss`; the teacher is the provisioned `repa_teacher` component (`facebook/dinov2-base` @ f9e44c81 = the hub `dinov2_vitb14_pretrain.pth`, checked tensor for tensor), never fetched | `train::teacher::Teacher`, `train::model::repa_loss` | ported; tokens 1e-4, losses 1e-5, projector + backbone grads 1e-4 (three grid relations); the release teacher on real weights 1e-3 (`IRIS_REPA_TEACHER_DIR`). `teacher_source: hf` (DINOv3 through `transformers`) refused typed: the native teacher is the dinov2 hub family |
| Activation checkpointing `none` / `full` / `selective_layer` (`ac_selective_every`) / `selective_op` | `models/ac.py` | `mlx_rs::transforms::checkpoint` per trunk / PiT block | segmented recompute: the region runs off the tape (`with_composable`, so it computes exactly what the replay computes), boundary `Var`s, reverse surrogate backward | ported. **f32** (`train_dtype: f32`): every policy's gradients equal the plain graph's ≤ 1e-6 and upstream's ≤ 1e-4 (upstream's own are bit-identical), and one optimizer step lands on the same next loss. **bf16** (Candle, sc-25690): the forward is bit-identical (the same loss), the gradients are not — a region's cotangent is rounded where it crosses the boundary — and what is held is that checkpointing adds no error of its own: per trained tensor the checkpointed gradient is no further from the f32 gradient than the plain bf16 gradient is (declared bound `e_full ≤ 2·e_none + 1e-3`, cosine to the plain bf16 gradient ≥ 0.999; measured on the miniature: mean relative distance from f32 3.6e-2 for both, worst ratio 1.003, worst cosine 0.99998). AdamW's first steps move every weight by ±lr whatever the gradient's size, so that rounding shows up in the next loss at the size of the bf16-vs-f32 gap itself (miniature: 4.2e-4 vs 4.3e-4) — two bf16 runs that differ only in the policy print different losses from step 2 on, as a bf16 and an f32 run do. **MLX** (sc-25690, measured for the depth and restoration trainers on the miniature, CPU stream — see *Activation checkpointing of the depth and restoration trainers*): the checkpoint transform re-runs the block's own graph, so f32 **and** bf16 gradients came out bit-equal to the plain graph's (max\|Δ\| 0; bf16 worst ratio 1.000, worst cosine 1.000000) under the same declared bounds; the generation trainer's MLX bf16 gradients are not measured. `selective_op` (torch SAC keeps matmul outputs) recomputes the whole block natively: same gradients, a different memory profile |
| Holdout validation: items flagged `model_options.split = "holdout"`, the frozen `(noise, t)` grid (`VAL_GRID_SEED`, `VAL_T_STRIDE`), `val_every_steps`, `val_samples`, banded `val/loss_t0..3` | `trainer.py` `_val_loss` | `train::holdout`, `Run::validate_holdout` → `<checkpoints>/metrics.jsonl` | same | ported; grid noise exact, per-sample loss 1e-5 |
| Metrics (`log_every`, `log_block_grad_norms`, REPA raw / weighted / total, loss scale) | `trainer.py` | `<checkpoints>/metrics.jsonl` (run header, steps, grad groups, validation) | same | ported (upload to a tracker is the product layer's) |
| fp16 mixed precision with `GradScaler` (65536, ×2 per 2000 clean steps, ×0.5 and skip on overflow) | accelerate | `MixedPrecision::Fp16` + `GradScaler`, checkpointed | CUDA only (candle's CPU backend has no half GEMM: refused typed there) | ported |
| `resume_optimizer` full / core (a new REPA projector starts cold; a projector the resumed run drops — `repa.weight: 0` — is refused under `full` on a replicated (`ddp`) run, as upstream's `_validate_optimizer_layout` refuses the changed parameter routing (`ckpt.py`), and dropped on an FSDP2 run, as upstream's `_drop_stale_aux_group` forgets it (`trainer.py`); `core` resumes without it either way); `save_every_epochs`; retention; exact resume of a pre-sc-25689 checkpoint (no `data_position`: its identity's saved `batch` / `accum` rebuild the legacy identity and cursor) | `trainer.py`, `train/ckpt.py` | `check_resume`, `Run::prepare` | same | ported |
| Distributed: DDP and FSDP2 (`full`, `grad_op`, `hybrid`, `hybrid_grad_op`), mesh resolution, `param_dtype` (the gathered forward copy rounded), `reduce_dtype` (rank-order reductions, bf16 rounding), `expected_global_batch` = batch × world × accum, `scale_warmup_by_world`, `checkpoint_io` rank0 / per_node / dcp, resharding a checkpoint to another world | `train/parallel.py`, `world.py`, `trainer.py` | `train::dist` (backend-neutral plan, `Collective`, the host `TcpCollective`); MLX transports `mlx_ring` / `mlx_jaccl` / `mlx_mpi` / `mlx_any` (MLX's own collectives on the CPU stream, behind a bounded TCP liveness barrier, each run on a worker thread waited on for at most the timeout — a rank lost inside an MLX collective is a typed `RankLost`; the blocked worker is abandoned and the process must exit) | the host TCP transport (no NCCL: the repo had no distributed primitives and Windows has no NCCL); the MLX transports refused typed | ported; see *Curricula and distributed training* below |
| Invalid meshes, a global-batch mismatch, a split world, a lost / stalled rank, a desynchronized collective | `parallel.py`, `world.py` | typed `DistError` (`InvalidMesh`, `GlobalBatchMismatch`, `SplitWorld`, `RankLost`, `Desync`, `Unsupported`) at `Trainer::train`, every wait bounded by `dist.world.timeout_secs` (default 5400 s, upstream's process-group timeout) — the rendezvous separately and shorter by `dist.world.rendezvous_timeout_secs` (default 600 s, torchrun's c10d join timeout; also bounds MLX's own `distributed::init`, run on an abandonable worker), every socket wait polled on a non-blocking socket (no byte lost to an expired socket timeout, which leaves a Windows socket indeterminate) so it observes the run's cancel flag (the rendezvous: a typed `Canceled` at once; a collective: once a pending cancel outlived `dist.world.cancel_grace_secs` (default 120 s, at least 1; one budget per transport, not renewed per frame or collective), so the peers can still reach the cooperative cancel agreement — after which, and around every checkpoint / export write, the grace is disarmed and the waits for a slow writer are bounded by the transport timeout only), and TCP keepalive on every transport socket (a peer that vanishes without a reset — power loss, a partition — fails its connection after ~90 s, a keepalive `TimedOut` being a lost rank; the probe timing is best effort: `SIO_KEEPALIVE_VALS` on Windows before 10 1709, else the system's); the TCP rendezvous drops connections that are not a rank of the job (malformed, another protocol, a wrong `SCENEWORKS_DIST_TOKEN` per-job secret, an out-of-range or duplicate rank) and keeps accepting; no frame is allocated from an untrusted length (bounded by what the receiver knows: 4 KiB hellos, exact reduce payloads, gather shards ≤ shard 0), and mismatched member payloads (length, op, precision, barrier tag) are a `Desync` | same | ported; multi-process tests below |
| Checkpoint lineage, resume agreement, atomic exports (sc-25690) | `train/ckpt.py`, `trainer.py` (one job, one checkpoint directory; every rank loads the same file) | `train::lineage`: a fresh run (or one resuming from outside its checkpoint root) moves every checkpoint already under the root aside to the one sibling `<root>.prior` (`lineage::prior_checkpoint_root`; the next retirement replaces it — bounded disk — and the product deletes it with the training's checkpoint root), after the resume source was validated (a refused resume moves nothing; a resume from `step_K` inside the root moves only the newer, abandoned steps), so retention and `latest` see only the run's own lineage; `agree_resume` — every rank resumes from one step with one saved state (a `latest` disagreement falls back to the newest step every rank holds; otherwise typed `SplitWorld` / `Desync`); `elect_checkpoint_writer` — under `per_node`, one writer per storage (nodes sharing a filesystem never publish into one directory concurrently); a checkpoint saved after the walker entered a new pass (a cancel at its first batch) carries the position captured when the boundary's pass ended; multi-file exports (generation full model, depth, restoration) staged in a sibling directory and published by rename — every failing step removes the staging directory and keeps (or restores) the previous export, and an interrupted publication's leftovers are swept (a hidden previous export restored) by the next | same | ported; `train_lineage` (both backends: cross-epoch cancel → resume bit-exact, fresh-run lineage, whole-directory export), `train_distributed::ranks_that_see_different_checkpoints_refuse_to_resume`, gen-core `lineage` tests (per-node election, agreement fallback / refusals) |

### Curricula and distributed training (sc-25689)

Oracle: the curricula section of `../tools/dump_iris_train.py` (frozen `train/repa.py`,
`models/ac.py`, `data/shapes.py`, `data/buckets.py`, `data/samplers.py`, `data/datasets.py`
sha256-pinned; DINOv2 `facebookresearch/dinov2@7764ea0f`) → `iris_curricula_golden.safetensors`, and
`iris_repa_teacher_real.safetensors` (the release teacher's tokens). Both backends hold
`curricula_parity` (MLX CPU stream / Candle CPU, FP32) and `curricula_lifecycle` (the stage chain
256 → 512 with REPA on a provisioned teacher, `area` batches + exact resume, AC leaving the run
unchanged, holdout metrics, fp16, `resume_optimizer: core`).

**Distributed semantics.** Every rank holds the same positional draws upstream's do (`mix_seed(seed,
rank, epoch, position)`) and its slice of the world-portable ranged walk; the per-rank epoch length
is agreed as the world minimum (upstream's `_agree_batch_count`). DDP all-reduces the window's
gradients (in `bucket_cap_mb` buckets) to the global mean in `reduce_dtype`; FSDP2 shards every
trained tensor by rows (`torch.chunk`: the first shards take the rows), reduces within the shard
group then across replicate groups, clips on the global norm, updates EMA and AdamW shard-wise, and
orthogonalizes each Muon matrix **whole** (the shards' rows gathered, Newton–Schulz once, each rank
keeping its rows) — so a sharded step equals the replicated one: AdamW ≤ 1e-6 (the f32 order of the
per-shard norm sum), Muon bit-identical with the clip out of reach. `param_dtype` rounds the gathered
forward copy (FSDP2's `MixedPrecisionPolicy`); the f32 shards stay the masters. `checkpoint_io: dcp`
writes one file per shard (`shard_NNNNN.safetensors`, zero-row tensors omitted) and any world
reassembles them; `rank0` / `per_node` write the consolidated layout. Transports: the host TCP
collective (both backends; rank 0 coordinates, every reduction in rank order, every wait bounded) and,
on MLX, MLX's own `ring` / `jaccl` / `mpi` collectives (CPU stream; the ring backend has no
sub-group `split`, so hybrid sub-group reductions are world all-gathers reduced per group — through
the length-padded gather, since a replicate group's row shard differs in length across groups and the
ring all-gather needs equal rows (sc-25690; `hybrid_sharding_over_the_mlx_ring_reduces_odd_row_shards`,
4 ring ranks, 3 rows over 2 shards); a TCP
liveness barrier precedes each MLX collective because MLX's socket backends block on a vanished
peer). `mlx_jaccl` is available only when the MLX build links `libjaccl` (source builds; prebuilt
cells compile a typed-unavailable stub).

**Multi-process evidence (`train_distributed`, both backends, real OS processes on one host):**
DDP over TCP vs over MLX ring on loopback — bit-identical trainable / EMA / optimizer / export; FSDP
`full` and `grad_op` (2 ranks) and `hybrid` 2 × 2 (4 ranks) vs DDP of the same world (AdamW ≤ 1e-6,
Muon exact); `param_dtype: bf16` honored (moves the result by 4.6e-4); a `dcp` checkpoint resumed on
the same mesh reproduces the uninterrupted run bit for bit and reshards to one process; a rank that
exits, or stalls, surfaces on the survivor as `RankLost` (the stall naming the transport timeout) and
nothing hangs; invalid meshes / a global-batch mismatch / an MLX transport on Candle are typed
refusals. Multi-host MLX (two Macs over ring or JACCL) is supported by the hostfile path but is not
exercised: the only machine this story may load is nax-macos-2. On candle/CUDA a rank trains on
`cuda:<local_rank>` (upstream's `world.py`); a rank whose ordinal has no GPU is a typed
`InvalidMesh` refusal unless the run opts into `dist.world.share_device` (several ranks on one
card — same semantics, shared memory and compute; never a silent fallback). The
`iris-curricula-cuda` lane runs the same tests on the Windows CUDA box (two RTX PRO 6000 Blackwell,
96 GB each, per `nvidia-smi`), whose runner listener exposes one of them to CUDA (`cuda:1` is an
invalid ordinal there): it proves the refusal on one visible card, then runs the multi-rank
scenarios with `share_device` on — both ranks on that card, over the host TCP transport (no NCCL on
Windows).

**Real weights (dispatch-only):** `iris-curricula-mlx` (`mlx-iris-curricula-train`, nax-macos-2)
and `iris-curricula-cuda` (`candle-iris-curricula`) run the release teacher against upstream and the
bounded official curriculum (`curricula_real_weights`: the shipped `stage1_256` with REPA for two
steps → `stage2_512` as a `new_phase` to step 4, batch 1).

**Upstream config map.** Every leaf of `configs/` (`TrainConfig` and its sections) and the trainer
CLI, as `gen_core::iris::train::upstream::UPSTREAM_OPTIONS` records it; `translate_upstream` turns a
config + dotlist overrides into the native request, refusing a refused key that is set away from its
default, and `check_resources` holds every `resource` key to the provisioned backbone / text encoder.
*honored* = carried into the request; *mapped* = a framework / performance flag with the named native
counterpart (never changes the update); *inert* = upstream's trainer never reads it either;
*resource* = fixed by the provisioned resource (a set value must match); *refused* = typed refusal
unless at its default.

| Upstream key | Disposition | Native |
| --- | --- | --- |
| `name` | honored | iris.run_name (state.json, metrics.jsonl) |
| `work_dir` | mapped | the request's output_dir (checkpoints under <output_dir>/<stem>.checkpoints) |
| `wandb_project` | honored | iris.tracking.project (metrics.jsonl header) |
| `report_to` | mapped | metrics are written to <checkpoints>/metrics.jsonl; uploading them to wandb / tensorboard is the product layer's (inference links no network client) |
| `tags` | honored | iris.tracking.tags (metrics.jsonl header) |
| `model.block` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.dual_depth` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.final_block_text` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.hidden_size` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.depth` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.num_heads` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.num_kv_heads` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.gated_attention` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.sandwich_norm` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.patch_size` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.in_channels` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.mlp_ratio` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.qkv_bias` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.qk_norm` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.norm_eps` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.attn_backend` | mapped | sdpa \| torch_flash \| torch_cudnn \| fa3 \| fa4 select a torch attention kernel; natively MLX's fused SDPA / Candle's SDPA compute the same attention |
| `model.modulation` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.modulation_rank` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.timestep_max_period` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.adaln_zero_init` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.rope_theta` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.rope_scale` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.rope_aspect` | honored | iris.rope_aspect (square \| isotropic; the trained and exported image RoPE law — stage 1 trains square) |
| `model.rope_frame_pairs` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.rope_frame_theta` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.text_rope` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.text_rope_theta` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.text_abs_pos_embed` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.text_dim` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.text_len` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.text_adapter` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.text_lap_num_layers` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.text_lap_num_heads` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.text_lap_mlp_ratio` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.repa_layer` | honored | iris.repa.layer |
| `model.pixel.enabled` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.pixel.depth` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.pixel.hidden_size` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.pixel.attn_hidden_size` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.pixel.num_heads` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.pixel.mlp_ratio` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.pixel.modulation` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `model.pixel.abs_pos_embed` | resource | the trained architecture is the backbone resource's config.yaml (random init reads it); a set value must equal it — provision a backbone config with that architecture to train one |
| `flow.num_train_timesteps` | resource | the 1000-point grid of the backbone's flow section |
| `flow.shift` | honored | iris.flow_shift |
| `flow.timestep_sampler` | honored | TrainingConfig::timestep_type |
| `flow.logit_mean` | honored | iris.logit_mean |
| `flow.logit_std` | honored | iris.logit_std |
| `flow.prediction` | honored | iris.prediction |
| `flow.x_pred_sigma_min` | honored | iris.x_pred_sigma_min |
| `flow.shift_law` | honored | iris.shift_law |
| `flow.shift_base_tokens` | honored | iris.shift_base_tokens |
| `text_encoder.name` | resource | the frozen conditioning is the text-encoder resource and the backbone's text_encoder section; a set value must equal it |
| `text_encoder.pretrained` | resource | the frozen conditioning is the text-encoder resource and the backbone's text_encoder section; a set value must equal it |
| `text_encoder.dim` | resource | the frozen conditioning is the text-encoder resource and the backbone's text_encoder section; a set value must equal it |
| `text_encoder.max_length` | resource | the frozen conditioning is the text-encoder resource and the backbone's text_encoder section; a set value must equal it |
| `text_encoder.dtype` | resource | the frozen conditioning is the text-encoder resource and the backbone's text_encoder section; a set value must equal it |
| `text_encoder.attn_implementation` | mapped | a Hugging Face attention backend switch; the shared native decoder runs the same attention |
| `text_encoder.hidden_layers` | resource | the frozen conditioning is the text-encoder resource and the backbone's text_encoder section; a set value must equal it |
| `text_encoder.null_embed_dir` | mapped | the on-disk null-embedding memo; natively the null is encoded once per run in memory (bit-identical: the encoder is frozen and deterministic) |
| `text_encoder.compile` | mapped | torch.compile of the frozen decoder; natively the decoder runs eagerly (same math) |
| `text_encoder.on_caption_overflow` | honored | iris.on_caption_overflow |
| `repa.weight` | honored | iris.repa.weight |
| `repa.variant` | honored | iris.repa.variant |
| `repa.spatial_norm_gamma` | honored | iris.repa.spatial_norm_gamma |
| `repa.teacher_source` | honored | iris.repa.teacher_source (hub: the native DINOv2 teacher; hf / DINOv3: refused, see TeacherArch) |
| `repa.teacher_hub` | honored | iris.repa.teacher_hub (facebookresearch/dinov2 only) |
| `repa.teacher` | honored | iris.repa.teacher (dinov2_vit{s,b,l,g}14[_reg]) |
| `repa.teacher_weights` | honored | iris.repa.teacher_weights: the teacher resource directory (LoadSpec component repa_teacher); never downloaded |
| `repa.teacher_dim` | honored | iris.repa.teacher_dim |
| `repa.proj_hidden_dim` | honored | iris.repa.proj_hidden_dim |
| `repa.teacher_image_size` | honored | iris.repa.teacher_image_size |
| `repa.teacher_patch_size` | honored | iris.repa.teacher_patch_size |
| `repa.teacher_match_student` | honored | iris.repa.teacher_match_student |
| `data.data_dirs` | refused | native training reads the request's TrainingItems (the product layer resolves WIDS shards into items) |
| `data.type` | honored | iris.data_type (pixel \| pixel_multiscale; the synthetic smoke datasets are refused) |
| `data.image_size` | honored | TrainingConfig::resolution |
| `data.shape_policy` | honored | iris.shape_policy |
| `data.aspect_ratio_bucket` | honored | iris.aspect_ratio_bucket |
| `data.shape_align` | honored | iris.shape_align |
| `data.shape_max_ratio` | honored | iris.shape_max_ratio |
| `data.caption_field` | honored | iris.caption_field |
| `data.caption_fields` | honored | iris.caption_fields |
| `data.val_data_dirs` | refused | the holdout set is the request items flagged model_options.split = "holdout" |
| `data.num_workers` | mapped | dataloader worker processes; natively the host decodes each batch in process |
| `train.batch_size` | honored | TrainingConfig::batch_size (per rank) |
| `train.grad_accum_steps` | honored | TrainingConfig::gradient_accumulation |
| `train.expected_global_batch` | honored | iris.expected_global_batch |
| `train.num_epochs` | honored | iris.num_epochs |
| `train.max_steps` | honored | TrainingConfig::steps (= max_steps × grad_accum micro-steps; 0 = the epoch horizon) |
| `train.mixed_precision` | honored | TrainingConfig::train_dtype (bf16 \| fp16 \| no) |
| `train.activation_checkpointing` | honored | iris.activation_checkpointing |
| `train.ac_selective_every` | honored | iris.ac_selective_every |
| `train.gradient_clip` | honored | iris.gradient_clip |
| `train.text_dropout` | honored | iris.text_dropout |
| `train.seed` | honored | TrainingConfig::seed |
| `train.optimizer.name` | honored | TrainingConfig::optimizer |
| `train.optimizer.lr` | honored | TrainingConfig::learning_rate |
| `train.optimizer.betas` | honored | iris.betas |
| `train.optimizer.weight_decay` | honored | TrainingConfig::weight_decay |
| `train.optimizer.muon_momentum` | honored | iris.muon_momentum |
| `train.optimizer.muon_nesterov` | honored | iris.muon_nesterov |
| `train.optimizer.muon_adjust_lr` | honored | iris.muon_adjust_lr |
| `train.optimizer.auto_lr` | honored | iris.auto_lr |
| `train.optimizer.base_batch_size` | honored | iris.base_batch_size |
| `train.optimizer.schedule` | honored | TrainingConfig::lr_scheduler |
| `train.optimizer.warmup_steps` | honored | TrainingConfig::lr_warmup_steps (× grad_accum) |
| `train.optimizer.scale_warmup_by_world` | honored | iris.scale_warmup_by_world |
| `train.ema.enabled` | honored | iris.ema_enabled |
| `train.ema.decay` | honored | iris.ema_decay |
| `train.dist.strategy` | honored | iris.dist.strategy |
| `train.dist.sharding` | honored | iris.dist.sharding |
| `train.dist.mesh.dp_shard` | honored | iris.dist.mesh.dp_shard |
| `train.dist.mesh.dp_replicate` | honored | iris.dist.mesh.dp_replicate |
| `train.dist.param_dtype` | honored | iris.dist.param_dtype |
| `train.dist.reduce_dtype` | honored | iris.dist.reduce_dtype |
| `train.dist.reshard_after_forward` | honored | iris.dist.reshard_after_forward |
| `train.dist.checkpoint_io` | honored | iris.dist.checkpoint_io |
| `train.dist.ddp.find_unused_parameters` | honored | iris.dist.ddp.find_unused_parameters |
| `train.dist.ddp.gradient_as_bucket_view` | honored | iris.dist.ddp.gradient_as_bucket_view |
| `train.dist.ddp.batched_grad_copy` | honored | iris.dist.ddp.batched_grad_copy |
| `train.dist.ddp.skip_all_reduce_unused_params` | honored | iris.dist.ddp.skip_all_reduce_unused_params |
| `train.dist.ddp.bucket_cap_mb` | honored | iris.dist.ddp.bucket_cap_mb |
| `train.dist.ddp.forward_sync_buffers` | honored | iris.dist.ddp.forward_sync_buffers |
| `train.load_from` | honored | iris.load_from (a native checkpoint / exported model dir; a torch .pth is refused — export it with scripts/export_checkpoint.py) |
| `train.resume_from` | honored | iris.resume_from ("latest" = TrainingConfig::resume) |
| `train.resume_data_world_size` | refused | migrates torch checkpoints written before data_position was persisted; every native checkpoint records its data position |
| `train.resume_data_policy` | honored | iris.resume_data_policy |
| `train.resume_optimizer` | honored | iris.resume_optimizer |
| `train.override_lr_on_resume` | honored | iris.override_lr_on_resume |
| `train.save_every_steps` | honored | TrainingConfig::save_every (× grad_accum) |
| `train.save_every_epochs` | honored | iris.save_every_epochs |
| `train.keep_last_checkpoints` | honored | iris.keep_last_checkpoints |
| `train.milestone_steps` | honored | iris.milestone_steps |
| `train.log_every` | honored | iris.log_every |
| `train.log_block_grad_norms` | honored | iris.log_block_grad_norms |
| `train.sample_every_steps` | honored | TrainingConfig::sample_every (× grad_accum) + iris.preview_at_step_one |
| `train.val_every_steps` | honored | iris.val_every_steps |
| `train.val_samples` | honored | iris.val_samples |
| `train.validation_prompts` | honored | iris.upstream_validation_prompts (true: the default seven; a list: the config's own — every prompt, uncapped) |
| `train.nan_loss_tolerance` | honored | iris.nan_loss_tolerance |
| `train.perf.foreach_ema` | mapped | multi-tensor EMA kernels; the native EMA update is the same elementwise blend |
| `train.perf.fused_adamw` | mapped | torch's fused AdamW kernel; the native AdamW is the same update |
| `train.perf.skip_dropped_text` | mapped | dropped rows skip the encoder; natively a dropped row always takes the null states without encoding (identical: the encoder consumes no randomness) |
| `train.perf.lazy_sync` | honored | iris.perf.lazy_sync (the non-finite guard on the clipped grad norm) |
| `train.perf.compile` | mapped | torch.compile (numerics within training noise); natively eager |
| `train.perf.compile_mode` | mapped | torch.compile mode; natively eager |
| `train.perf.compile_scope` | mapped | torch.compile scope; natively eager |
| `train.perf.compile_dynamic` | mapped | torch.compile shape dynamism; natively eager |
| `train.perf.compile_fullgraph` | mapped | torch.compile fullgraph; natively eager |
| `train.perf.recompile_limit` | mapped | dynamo cache budget; natively eager (no recompilation) |
| `sample.steps` | inert | scripts/sample.py only; the trainer's previews use 100 steps |
| `sample.order` | honored | iris.preview_order |
| `sample.cfg_scale` | inert | scripts/sample.py only; the trainer's previews use CFG 3 |
| `sample.cfg_interval` | inert | scripts/sample.py only; the trainer's previews use (0, 1) |
| `sample.negative_prompt` | inert | scripts/sample.py only; the trainer's previews pass none |
| `cli.--config` | honored | translate_upstream's yaml |
| `cli.--expected-nodes` | honored | iris.dist.world.expected_nodes |
| `cli.--prestage` | mapped | fills node-local torch caches (hub teacher, shape-plan keys); natively the teacher is a provisioned resource and shape keys are computed in the run's preparing pass — Trainer::validate is the dry run |
| `cli.overrides` | honored | translate_upstream's dotlist overrides |

### Generation control surface (sc-25681)

Every public argument of `iris3b.sampling.generate` / `scripts/sample.py`, its request field, and the
native computation it changes. "Test" names the fixture case (`tests/controls_parity.rs`, both
backends, against `iris_controls_golden.safetensors`) and the provider-level check
(`tests/controls_contract.rs::every_advertised_control_changes_the_render`, both backends).

| Upstream control (default) | Request field | MLX | Candle | Test | Notes |
| --- | --- | --- | --- | --- | --- |
| `prompts: list[str]` | `prompt`, or `prompt_batch` (non-empty ⇒ `prompt` empty) | `pipeline::encode` per prompt → one batched `denoise` (`[B, …]` text states, per-row masks, null expanded over the batch in `cat([uncond, cond])`) | same | `batch` | `prompt_batch.len() × count ≤ 8` images; image `k = c·B + j` draws noise from seed `k` (MLX `seed + k`, Candle `image_seed(seed, k)`), so a batched image equals its single render up to batched-GEMM rounding (`a_prompt_batch_renders_one_image_per_prompt_per_count`, FP32 ≤ 1 code value) |
| `height` / `width` (1024²) | `height` / `width` (multiples of 16, 16..=2048) | noise geometry, patch grid, 2-D RoPE grid, PiT sincos table | same | `square` (8×8), `portrait` (12×8), `landscape` (8×16) vs `base` (8×12, also landscape) | aspect ratio is the pair; no separate aspect control upstream |
| `steps` (100) | `steps` | `dpm_solver_plan(steps, …)` | same | `steps` (8) vs every other case (5 steps) | |
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
| Application | forward-time residual on the projection (`mlx_gen::adapters::AdaptableLinear`, `apply_adapters_strict`), base never mutated; LoRA `scale·(x·Aᵀ)·Bᵀ`; LoKr / LoHa the reconstructed `[out, in]` delta (`Adapter::Lokr` — the projections are dense, so the shared loader materializes rather than taking its packed-base structured `w1·X·w2ᵀ` form) built in the backbone compute dtype (`AdaptableHost::lycoris_dtype`: bf16 under `Precision::Bf16`, f32 under `Precision::Fp32` — sc-25690) | forward-time residual on the projection (`nn::Residual`, sc-25686 — previously folded `W += δ`, which rounds a sub-ulp delta away under bf16 compute and is not the forward the trainer optimizes), base never mutated; LoRA `(x·Aᵀ)·(Bᵀ·scale·alpha/rank)`, factors and GEMMs in f32; LoKr (stamped or LyCORIS) the structured `Residual::Kron` — its Kronecker factors held **f32** with `scale·(alpha/rank)` baked into `w2`, applied as `w1·X·w2ᵀ` (GEMMs in the activation's dtype, f32 on the CPU backend) without forming `kron(w1, w2)` — falling back to the reconstructed delta (`Residual::Delta`, compute dtype) only when the factors do not decompose the projection; LoHa its exact rank-`r₁r₂` factorisation (`Residual::LowRank`, compute dtype) when `r₁r₂·(in + out) < in·out`, else the reconstructed delta (`Residual::Delta`, compute dtype); a stamped LoKr's per-target `.alpha` must equal its metadata alpha |
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
them, the solver state in f32. `Precision::Fp32` is upstream's CPU path for the backbone (f32
everywhere in the DiT, its adapters and the solver).

`LoadSpec::precision` selects the **backbone's** compute dtype only (`model::compute_dtype`, both
backends). It never reaches the Qwen3-VL text tower, which follows the backbone's
`text_encoder.dtype` (`bfloat16` in the release — `IrisConfig::validate_supported` refuses any other
value): on MLX the tower is the shared `mlx_llm::CausalLm`, which always computes in bf16 on MLX's
default device, so an FP32 load still conditions on bf16-computed hidden states (read back as f32);
on Candle, `text_encoder::tower_dtype` takes `text_encoder.dtype` on a GPU device and f32 on the CPU
device (no CPU half GEMM), and the device is `candle_gen::default_device()` (`cuda:0` on a CUDA
build, else CPU) — chosen by the build, never by `Precision`. The restoration route has no text
tower (E4).

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
| Every generation control, 12 cases (`controls_parity`, `tools/dump_iris_controls.py`) | bf16 tower + FP32 backbone (MLX); f32 tower + FP32 backbone vs the oracle's **fp32-tower** render (Candle) | 6e-2 (MLX) / 2e-4 (Candle) | 8.4e-3–4.5e-2 | 1.1e-5–5.8e-5 | MLX: tower rounding amplified by CFG (upstream's own bf16-vs-fp32 tower: 1.7e-2–7.3e-2 across the cases); every control's golden sits ≥ 0.15 from `base`, so an ignored control fails |
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

## Depth training coverage table (`iris_3b_depth` trainer, sc-25687)

The frozen release has **no downstream training loop**: the model and export are upstream code
(`src/iris3b/downstream/depth.py` `IrisDepth`, `scripts/export_downstream.py depth` — whose `depth()`
asserts a "direct-regression (stage 0)" checkpoint carrying its own `conditioning`), the recipe is the
paper (arXiv 2610.09450v1 §5.2 "Recipe." / "Data.", Eq. 5). Nothing below is a verbatim port of a
training script; each row names its source, and where the paper is silent the choice is a documented
assumption **and** a `model_options.iris` option (never a hidden constant). Contract:
`gen_core::iris::train::depth` (plan, items, paired transforms, normalization, Eq. 5 host reference,
export layout, resume identity); tensors: `mlx_gen_iris::train::depth` / `candle_gen_iris::train::depth`.

| Choice | Source | MLX | Candle | Status |
| --- | --- | --- | --- | --- |
| Model = widened projections + zero 4th channel + 1×1 reducer + empty prompt, one forward at `t = flow.num_train_timesteps` | `IrisDepth.__init__` / `.forward` | `depth::depth_forward` (shared with the estimator) | same | ported — training and inference share one graph |
| Full fine-tune (no adapters) | §5.2 "We fully fine-tune with AdamW" | trainer refuses `full_finetune = false` | same | ported |
| Target: relative log depth, per image, 2nd/98th percentiles of the **valid** GT → −1 / +1 | §5.2 "normalized per image by the 2nd and 98th percentiles of the valid ground truth"; `depth.py` docstring "mapped to -1 (near) and +1 (far)" | `normalize_log_depth` (numpy `linear` percentile) | same | ported; `low_percentile` / `high_percentile` options. **Assumption:** normalized over the training crop's valid pixels (after the paired resize / crop); the paper says only "per image" |
| Target clamp to [−1, 1] | paper silent (Marigold clips; the stated map does not) | off | same | **assumption**, option `clip_target` |
| Loss Eq. 5: `1/|M| Σ_{p∈M} |d̂−d| + 5‖∇d̂−∇d‖₁` | §5.2 Eq. 5 | `train::depth::depth_loss` | same | ported; `gradient_weight` option (default 5) |
| ∇ = forward differences in x and y; a pair counts when both pixels are valid | paper silent (∇ undefined) | fixed | same | **assumption** (documented; the conventional masked gradient-matching term) |
| `|M|` over the micro-batch | Eq. 5 has one `M` | `valid_pixels` | same | literal reading; option `loss_reduction: per_image` |
| Invalid / beyond-far-plane pixels masked | §5.2 "masked from the loss" | per-item `mask_path`, `min_depth`, `max_depth` (+ finite, `d > 0`) | same | ported |
| No-valid-depth / degenerate examples | paper silent | `invalid_examples: reject` (default, refuses naming the item) / `skip`; a crop-emptied row is masked out | same | explicit policy, tested |
| AdamW β (0.9, 0.999), wd 0.01, LR 3e-5 constant, no warmup, clip 1.0, global batch 32, 10K steps | §5.2 | β + clip are the depth defaults; `apply_paper_recipe` sets the shared fields | same | ported (Muon also honoured when asked) |
| Raw weights evaluated (no EMA) | §5.2 "evaluate the raw weights" | `ema_enabled` default false, `export_weights: raw` | same | ported; EMA optional |
| Random horizontal flips | §5.2 "with random horizontal flips" | `hflip_prob` 0.5, paired over RGB / depth / mask | same | ported (½ = **assumption**, the conventional flip) |
| Training sizes 480×640 / 352×1216 (native), 9:1 per batch | §5.2 "Data." | one training size per run (`depth_size`); `resize: cover` (bicubic RGB, nearest depth + mask) or `none`; `crop: center` / `random` | same | resolution and mixing are dataset/job choices (SceneWorks); crop/resize **assumption** + options |
| Parent init: widened columns | paper silent | zeros (the channel is always 0: no forward effect, no gradient) | same | **assumption**, tested |
| Parent init: reducer | paper silent; `nn.Conv2d` default in `IrisDepth.__init__` | `reducer_init: torch_default` (U(±1/√C)), or `channel_mean` | same | upstream-construction default + option |
| Parent empty prompt | export carries `payload["conditioning"]` | `empty_prompt` component file, or encode `""` once with `text_encoder` | same | ported; a `depth/` export uses its own |
| Export | `export_downstream.py depth` | `<out>/<stem>/{config.yaml (model/text_encoder/flow + task: depth), model.safetensors (FP32 `pixel.*` + `depth_reducer.*`), empty_prompt.safetensors}` + `irisTask=depth` / `irisArtifact=full_model` / `irisBaseIdentity` stamps | same | ported; runs in `depth::load` unchanged; generation / restoration loaders refuse it |
| Lifecycle (cancel, checkpoints, resume incl. optimizer/scheduler/EMA/RNG/data position, new phase, retention, previews) | generation trainer contract (sc-25685) | shared `gen_core::iris::train` machinery; `state.json` `task = depth` | same | ported |

Fixtures: `iris_depth_train_golden.safetensors` from `../tools/dump_iris_depth_train.py` (upstream
`IrisDepth` on `tiny-depth`, torch FP32 CPU, the paper's normalization + Eq. 5, one
`clip_grad_norm_(1.0)` + `torch.optim.AdamW(β=(0.9, 0.999), wd=0.01)` step at lr 1e-3). Measured
(`tests/depth_train_parity.rs`, FP32 CPU): normalized targets 1.2e-7; prediction 2.0e-5 of peak; loss
2.5215921 vs 2.5215914; per-tensor gradient norms ≤ 1e-3 rel; selected gradients ≤ 9e-5 of peak (MLX)
/ ≤ 8.8e-5 (Candle); the AdamW update matches to ≤ 3e-8 on every well-conditioned element (elements
whose gradient is at rounding level may flip `sign(g)` in Adam's first step, bounded by 2·lr);
post-step loss 2.3403609 vs 2.3403635. MLX ↔ Candle parity is transitive through this one reference.
Hand-derived Eq. 5 values and finite-difference subgradients: `gen_core` unit tests. Candle's `abs`
backward is +1 at exactly 0 (torch: 0) — reachable only at an exactly-zero valid error.

Real weights (`tests/depth_train_real_weights.rs`, ignored, both backends): the `depth/` export
(~3B parameters) fully fine-tuned for 4 bf16 AdamW steps (lr 3e-5, wd 0.01, clip 1) at 128², batch 1,
on the repo photo with a synthetic depth ramp, then the export is loaded by the estimator unchanged.
Candle/CUDA (RTX PRO 6000, run 38133787177): losses 0.818 → 0.411 → 0.227 → 0.343, 2.7 s/step,
train + checkpoint + export 230 s, VRAM peak 64.6 GB; estimator mean |pred − target| 0.662 (release)
→ 0.112 (trained). MLX on nax-macos-2 (dispatch-only `iris-depth-train-mlx` profile: 128 GB RAM,
`memory_pressure` 94% = 120 GB free at start, 80 GB phys_footprint guard; run 38136750391): losses
0.830 → 0.355 → 0.325 → 0.219, 1.1 s/step, train + checkpoint + export 14.1 s, peak phys_footprint
59.0 GB, 19 s test wall; estimator mean |pred − target| 0.660 (release) → 0.083 (trained).

## Activation checkpointing of the depth and restoration trainers (sc-25690)

`TrainingConfig::gradient_checkpointing` selects `full` checkpointing on the MLX depth and
restoration trainers, as on Candle (the typed refusal is gone; the option is the same on both
backends). Upstream ships neither task with checkpointing; it changes memory and compute only, so
it is no part of a run's identity (a checkpointed export carries the plain export's stamps). Default
off.

| | Depth (`iris_3b_depth`) | Restoration (`iris_3b_restore`) |
| --- | --- | --- |
| Regions | the generation trainer's (`train::model::region_hooks`: one `mlx_rs::transforms::checkpoint` per trunk block and per PiT pixel block, each rebuilt from the block's own traced tensors by `TrainModel::region_source` + `IrisDiT::load_trunk_block` / `load_pixel_block`) around the **widened** backbone — the widening touches only `s_embedder.proj` and `pixel_embedder.proj`, outside every region | the same regions around the generator in the **generator** step; an adapter run rebuilds a region's block from the frozen base plus the traced LoRA / LoKr factors, as the generation trainer does |
| Seam | `depth::depth_forward_hooked` (`depth_forward` = the same with `NoHooks`), `DepthModel::ac` | `Frozen::restore_hooked` (`restore` = the same with `NoHooks`), `RestoreTrainPlan::ac` |
| Outside the regions | the patch / text embedders, the final layer, the 1×1 reducer, Eq. 5 | the embedders and final layer, `x̂ = x_lq − σ·v`, the L2 / LPIPS / critic terms; the critic step (no generator gradient) is never checkpointed |
| Previews / estimator forward | plain (`DepthModel::predict`) | plain (`Frozen::restore`) |

There is no explicit release of the loss graph (Candle's `backward_segmented` needs one because its
tape holds every node until the loss drops): MLX has no tape — `keyed_value_and_grad` builds one lazy
graph and `eval` frees each array once its last reader has run, and every reader of the LPIPS / critic
activations is upstream of the first region's recompute in the backward. Measured, not assumed: at
1024² the checkpointed transient above the resident set is 27.5 GB for restoration, 27.1 GB for depth
and 27.0 GB for generation — the head terms add nothing to the peak.

Gradient equality, miniature, CPU stream (`tests/depth_train_parity.rs`,
`tests/restoration_train_lifecycle.rs`, `activation_checkpointing_gradients_equal_the_plain_graph`):
f32 — declared ≤ 1e-6 of `max(1, tensor peak)` for both tasks (the generation trainer's bound; Candle
needs 1e-5 for restoration), **measured 0** for both, loss terms bit-equal; bf16 — the Candle rule
(`e_full ≤ 2·e_none + 1e-3` against the f32 gradient, cosine to the plain bf16 gradient ≥ 0.999,
step-1 loss bit-equal): depth 175 tensors mean `e_none` = mean `e_full` = 5.694e-2, worst ratio 1.000,
worst cosine 1.000000; restoration 173 tensors 6.664e-2 for both, worst ratio 1.000, worst cosine
1.000000. Lifecycle (`activation_checkpointing_leaves_the_run_unchanged`, both tasks; restoration
also as LoRA): the checkpointed run's export equals the plain run's (declared ≤ 1e-6, measured 0),
carries the same stamps and loads in the estimator / `iris_3b_restore`; cancel → checkpoint → resume
under checkpointing reproduces the uninterrupted checkpointed run bit for bit (weights, EMA,
optimizer(s), critic head, data position, export). The default path is unchanged bit for bit: the same
export, checkpoint members and step losses as the parent commit on five miniature runs (depth f32 /
bf16, restoration f32 / bf16 / LoRA).

Real weights, nax-macos-2 (dispatch-only `iris-memory-mlx`, steady pass: batch 1, full fine-tune, bf16
over f32 masters, 6 optimizer steps, MLX cache limit 0; run 38405154244; decimal GB, MLX's own
active-byte high-water, cumulative over the run):

| Shape | Policy | Active peak after steps 1…6 (GB) | s / step | Process footprint peak (GB) | Completes under the 115 GB guard |
| --- | --- | --- | --- | --- | --- |
| depth 1024² | none | 68.67 / 92.58 / 92.58 / 92.58 / 92.58 / 92.58 | 5.1 | 92.40 | yes |
| depth 1024² | **full** | 48.68 / 62.10 / 62.10 / 62.10 / 63.01 / 63.01 | 5.9 | 63.21 | yes |
| restoration 512² | none | 62.79 / 71.57 / 71.57 / 71.57 / 71.57 / 71.57 | 9.3 | 71.87 | yes |
| restoration 512² | **full** | 62.87 / 63.21 / 63.21 / 63.21 / 63.21 / 63.21 | 9.9 | 62.98 | yes |
| restoration 1024² | none | — (guard stop above 115 GB, earlier runs) | — | > 115 | **no** |
| restoration 1024² | **full** | 63.15 / 77.24 / 77.24 / 77.24 / 77.25 / 77.25 | 33.5 | 77.48 | **yes** |

A restoration step is one generator window plus one critic window. The step losses of a checkpointed
run equal the plain run's to the printed digit at both measured pairs (depth 1024²: 0.4472 → 0.0958;
restoration 512²: 3.8255 → 3.3667). The 1024² restoration full fine-tune — which does not fit a 128 GB
Mac plain — runs in 77.5 GB checkpointed; where both policies ran, checkpointing cost 6 % (restoration
512²) to 16 % (depth 1024²) more time per step.

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
| `scale` (default 4.0, any positive float incl. 1×): `F.interpolate(scale_factor, bicubic, align_corners=False)` — output `floor(side · scale)`, coordinate scale `1 / scale`, Keys a = −0.75, clamped taps | `__call__` | `bicubic_scale_factor`, `plan` | shared | shared | ported (`TargetSize::Scale`; `TargetSize::ModelDefault` = 4×); the request's f32 is read as its shortest round-trip decimal (`1.3f32` → the Python float `1.3`), so `floor(1000 · 1.3) = 1300` like torch, not 1299; min-edge / explicit resolution refused (upstream has neither) |
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
and `forward_count()`. A scale whose output rounds to zero pixels, or whose output, enlarged
processing size or patch-padded size does not fit `u32` (e.g. a 1×5,000,000 image, whose long side
the enlarge multiplies by 1024), is a typed refusal;
the provider plans with the loaded export's geometry, which equals `TileGeometry::RELEASE` for the
release export.

### Fixtures and tolerances

`tests/fixtures/tiny-snapshot/upscaler/` is a miniature export written by upstream's own
`export_downstream.py restoration` (the miniature backbone perturbed, a 3-token empty prompt,
`tile` 32, `sigma` 0.5); `tests/fixtures/iris_restoration_golden.safetensors` holds the op-level and
end-to-end references (`../tools/dump_iris_restoration.py`).

| Gate | Tolerance | Measured (MLX) | Measured (Candle CPU) |
| --- | --- | --- | --- |
| `scale_factor` bicubic (×4, ×2.5 on odd sides, ×1, ×0.75, ×3, non-dyadic ×1.3) | 1e-5 of peak | ≤ 1.4e-6 | shared host code |
| Antialiased resize (up, down, one axis, odd) | 1e-5 | ≤ 7.2e-7 | shared |
| Gaussian window (32 full; 1024 centre row / column / subsample) | 1e-6 | ≤ 6.0e-8 | shared |
| Wavelet colour fix (29×37, smaller than dilation 16) | 1e-5 | 1.8e-7 | shared |
| Tile positions, budget sizes | exact | exact | shared |
| `fit_budget` PIL Lanczos bytes (7 shapes) | sha256-exact | exact | shared |
| End to end FP32 through the request path, 8 cases (enlarge → tile → resize back, multi-tile, 1×, 2.5×, single tile, portrait 3×, budgeted, non-dyadic 1.3×), planned output size and enlarge branch exact: fused tiles / colour-fixed float | 1e-4 of peak | ≤ 1.9e-5 / ≤ 9.2e-6 | ≤ 1.9e-5 / ≤ 8.8e-6 |
| Same, RGB8 | ≤ 1 level | ≤ 1 (at most 6 of 61 440 values differ) | ≤ 1 |

Real weights (`tests/restoration_real_weights.rs`, `../tools/dump_iris_restoration_realweight.py`):
the native bf16 path is held to 1.5× upstream's **own** bf16-autocast-vs-FP32 mean RGB8 distance on
the same input (FP32: 0.5 level mean). `tests/fixtures/iris_restoration_real_small.safetensors` is
the committed upstream reference of the small case (RGB8 only), so the CUDA job checks parity too.

| Case | Upstream bf16 vs fp32 (the bound's basis) | MLX bf16 vs upstream fp32 | Candle CUDA bf16 vs upstream fp32 |
| --- | --- | --- | --- |
| 512×384 → 2048×1536 (6 tiles), machine-local golden | max 68, mean 0.521, PSNR 47.7 dB | max 60, mean 0.423, PSNR 49.0 dB (restore 10.8 s, peak footprint 11.7 GB) | — (no Python oracle on the runner): release-scale render 512×384 → 2048×1536, 6 tiles in 2.6 s, VRAM peak 10.1 GB |
| 64×48 → 256×192 (enlarged, 2 tiles), committed | max 23, mean 0.522, PSNR 48.1 dB | max 19, mean 0.459, PSNR 48.7 dB | max 23, mean 0.701, PSNR 46.0 dB (bound 0.783; restore 1.0 s, VRAM peak 10.1 GB; run 38085270480) |

## Restoration training (trainer `iris_3b_restore`, sc-25688)

The frozen release ships the restoration export and its inference but **no training loop**. The
recipe is the paper's (arXiv 2610.09450v1 §5.3, Eq. 6: "follows the one-step recipe of HYPIR"); every
choice the paper leaves open is traced to HYPIR's reference trainer, `XPixelGroup/HYPIR` @
`b61d107c6cef38f01a93c7833558869731cfa8c1` (`HYPIR/trainer/base.py`, `HYPIR/model/{D,backbone}.py`,
`HYPIR/dataset/{realesrgan,batch_transform,utils,diffjpeg}.py`, `HYPIR/utils/degradation.py`,
`configs/sd2_train.yaml`; sha256-pinned in `../tools/dump_iris_restoration_train.py`). One
backend-neutral contract (`gen_core::iris::restoration_train` + `gen_core::iris::degradation`), two
native trainers (`mlx-gen-iris::train::restore`, `candle-gen-iris::train::restore`) with their own
LPIPS-VGG (`train::lpips`) and critic (`train::critic`).

### Training-only resources (separate from every export; the caller checks the sha256)

| Component | What (reference use) | Pinned source | sha256 |
| --- | --- | --- | --- |
| `critic_trunk` | OpenCLIP `convnext_xxlarge` `laion2b_s34b_b82k_augreg_soup` (`backbone.py` L38–39); only `visual.*` read | `laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup` @ `9f3e8ee3f383c672388d9178afe70af9e63ac9df` `open_clip_pytorch_model.bin` | `d571d992…e2f155` |
| `lpips_vgg16` | `lpips.LPIPS(net="vgg")` trunk (`base.py` L129) = torchvision `vgg16-397923af` | `timm/vgg16.tv_in1k` @ `b8d8aa2dd860af9233c8c67385a8097fd6c35d3f` `model.safetensors` (`features.*` bit-identical to torchvision's, checked) | `57b02691…203fad8` |
| `lpips_lin` | LPIPS v0.1 VGG heads (`lpips==0.1.4`) | `vivym/lpips` @ `270571f1fb2a2c4c5f920cec96cd87838c345982` `vgg_lpips_linear.safetensors` (bit-identical to `richzhang/PerceptualSimilarity@082bb24` `vgg.pth`, checked) | `0c6387bf…4333ac` |

Full hashes, sizes and URLs: `gen_core::iris::restoration_train::TRAINING_RESOURCES`.

### Recipe trace (source → native)

| Choice | Source | Native (`gen_core` / MLX / Candle) | Status |
| --- | --- | --- | --- |
| One-step generator `x̂ = x_lq − σ·v(x_lq, σ·N, ∅)` on the bicubic-upsampled input, empty-prompt conditioning, no noise | paper §5.3; `restoration.py` `restore_tile`; HYPIR `forward_generator` (one step at `model_t`) | `Frozen::restore` (both) — the transform's exact forward at the export's `task.sigma` (release 0.5 ⇒ t = 500) | ported (HYPIR's SD2 `model_t = 200` is a different backbone's choice; the Iris export's own `sigma` is the identity trained) |
| Loss `‖x̂−y‖₂² + 5·LPIPS + 0.5·L_adv` | paper Eq. 6; `base.py` L282–284; yaml `lambda_*` | `generator_terms` (both); `loss_l2/lpips/gan` options | ported; every term on (zeroing one is a request, never a default) |
| Pixel term `F.mse_loss(reduction="mean")` | `base.py` L282 | mean over every element | ported |
| LPIPS-VGG v0.1, inputs in [−1, 1], `.mean()` | `base.py` L129, L283 | `train::lpips` (both): ScalingLayer, VGG16 taps relu1_2…relu5_3, channel-normalize (`+1e-10`), lin, spatial mean, sum | ported; f32 (not under autocast); norm floored at `sqrt(1e-30)` so an all-zero pixel's gradient is finite (torch gives NaN there) |
| Critic trunk features | `D.py` L77–81, `backbone.py` `_visual_forward(return_pooled_feats=True)` @ timm 1.0.15 | `ConvNextTrunk::features`: CLIP normalize of `x·0.5+0.5`, stages 0/1/2 maps + pooled `head.norm`→`proj` embedding | ported (timm ≥1.0.20 changed `forward_intermediates` — pinned 1.0.15, HYPIR's) |
| Critic head `MultiLevelDConv(level=4, in_ch1=[384,768,1536], in_ch2=1024, out_ch=512, down=2)` | `D.py` L8–49, L61 | `head_logits` (both): BlurPool(pad 2)→SN conv3×3 s2→LeakyReLU 0.2→BlurPool(pad 1)→SN conv1×1 s2; SN Linear→LeakyReLU→SN Linear | ported |
| Spectral norm (`torch.nn.utils.spectral_norm`, 1 power iteration per train-mode forward, none in eval; σ = uᵀWv with u, v constants) | `vision_aided_loss` `spectral_norm` | `power_iteration` + `spectral_weight`; `u`/`v` live in the checkpoint | ported; two iterations per critic micro-batch (`D(y)` then `D(x̂)`), none in generator windows |
| Adversarial loss: `multilevel_loss(alpha=0.8)` = BCE-with-logits, target 0.8 (real / generator), 0 (fake), spatial mean per level, summed | `D.py` L62; `cv_losses.py` | `multilevel_loss` (both); `gan_real_label` option | ported (non-saturating BCE with one-sided label smoothing) |
| Critic loss `mean D_real + mean D_fake`, x̂ regenerated under no-grad | `base.py` L296–304 | `critic_grads` (both) | ported |
| Alternation: window `(batch_count // accum) % 2` even = G, odd = D; one step = G + D window | `base.py` L327, L346–348 | `restoration_train::phase_of`; `steps` / `save_every` / `sample_every` must be multiples of `2·accum` | ported |
| Optimizers: AdamW for G and D, lr 1e-5 both, betas (0.9, 0.999), torch defaults eps 1e-8, weight_decay 0.01 | paper §5.3 ("We fully fine-tune with AdamW at an LR of 10⁻⁵ for both generator and critic"); `base.py` L164–176; yaml `lr_G`, `lr_D`, `opt_kwargs` | `IrisOptimizer` (AdamW route) ×2. The values come from the **request** (`TrainingConfig::learning_rate` / `weight_decay`; `lr_critic`, `betas`, `critic_betas`, `critic_weight_decay` options); the recipe values are `restoration_train::recipe_training_config()` / `apply_recipe` (`RECIPE_LR = 1e-5`, `RECIPE_WEIGHT_DECAY = 0.01`, full fine-tune, constant LR), the trainer's documented default | ported |
| Clip `max_grad_norm = 1.0` per network | `base.py` L288, L307 | `clip_grads` per window; `gradient_clip`, `critic_gradient_clip` | ported |
| EMA 0.999, updated after each G optimizer step; EMA weights exported | paper ("EMA … decay 0.999 is scored"); `base.py` L343, `ema.py` | `ema_update` after the G step; `export_weights` default `ema` | ported |
| LR schedule | paper (Iris arm: constant) | constant (default) or cosine; warmup in steps | constant ported; cosine = the shared knob; linear refused |
| Crops 1024², horizontal flip | paper; `realesrgan.py` `augment`, `utils.py` `random_crop_arr(min_crop_frac=0.7)` / `center_crop_arr` (PIL BOX halving + BICUBIC) | `training_pair` (`crop`: random / center / none), PIL-exact BOX/BICUBIC (`imageops`) | ported; `resolution` is the crop |
| Second-order Real-ESRGAN degradation at 4× incl. USM sharpening of the GT | paper; `batch_transform.py` `__call__`; yaml dataset + batch_transform | `gen_core::iris::degradation` (kernels, sinc via J₁, reflect `filter2D`, area/bilinear/bicubic `F.interpolate`, Gaussian/Poisson (gray) noise, `DiffJPEG(differentiable=False)`, USM, 8-bit round) | ported; per-op parity + end-to-end 8-bit LQ bit-exact on recorded draws |
| Batch-level resize/mode draws + 256-entry training-pair queue | `batch_transform.py` (`_dequeue_and_enqueue`) | drawn **per item** from `mix_seed(seed, RESTORE1, epoch, item)` | deliberate mapping: the queue only diversifies per-batch draws; per-item draws do that directly and keep each pair a function of `(seed, epoch, item)` (cancel/resume bit-exact) |
| Aligned degraded/clean pairs | (not in HYPIR) | `TrainingItem::model_options["degraded_image"]`; integer-scale aligned crops, torch-exact bicubic upsample | native addition (story AC); all-or-none per run |
| Full fine-tune — **the recipe** | paper §5.3 ("We fully fine-tune …") | `full_finetune: true` (set by `recipe_training_config()`) | ported; the default request |
| LoRA (`init_lora_weights="gaussian"`, `lora_alpha = rank`) merged into the export — HYPIR's arm, supported but **not** the default | HYPIR's own arm (`sd2.py` `LoraConfig`) | LoRA / LoKr over `blocks.*`, `lora_init` gaussian default, merged (f32) at export | ported; export is always the full `upscaler/` layout |
| Mixed precision bf16; critic trunk `precision="bf16"` | yaml `mixed_precision: bf16`; `base.py` L151 | `train_dtype`, `critic_dtype` (convs/linears in bf16, LayerNorm/reductions f32) | ported; Candle CPU refuses a bf16 critic (no CPU bf16 GEMM) |
| Data order (shuffled DataLoader) | `base.py` L188 | `gen_core::iris::train::DataWalk` (the generation trainer's ranged walk) | native (reused machinery, deterministic) |
| Checkpoints: `accelerator.save_state` + EMA dict, `checkpoints_total_limit` | `base.py` L431–453 | generation-trainer checkpoint machinery + `critic.safetensors` (head + `u`/`v`), `critic_optimizer.safetensors`, `state.json` (`iris-restoration-train-checkpoint-v1`) | ported; resume restores generator, EMA, both optimizers, spectral buffers, data cursor |
| Export | — | `upscaler/` layout: `config.yaml` + `empty_prompt.safetensors` copied verbatim from the base export, `model.safetensors` (FP32, `irisTask=restoration`, `irisArtifact=full_model`) | loads in `iris_3b_restore` with no training-only component; the depth loader and the generation loader (`downstream::refuse_task_export`) refuse it |
| Previews | `log_images` | the first item's degraded input restored with `preview_weights` every `sample_every` | native (no prompts — `sample_prompts` refused) |

### Verification

* `gen_core::iris::degradation` parity (`degradation_parity.rs`): HYPIR kernels ≤1e-8, every op on its
  recorded input ≤2e-5 (JPEG: ≤2 % of elements past 1e-3), the full chain's 8-bit LQ bit-exact on all
  three scripted chains.
* `tests/restoration_train_parity.rs` (both): LPIPS (formula weights), miniature ConvNeXt critic
  features, generator term + input gradient, the critic step (power iterations, losses, head
  gradients, clip + AdamW), each Eq. 6 term and its gradient — Candle CPU f32 to 1e-4 of peak (the
  64-px LPIPS gradient to 1.5e-2: deep-tap ReLU gates within float noise of zero); MLX on Metal
  (the production path; TF32-class f32 accumulation) to 2e-2 / 2e-3. The MLX CPU stream's
  run-to-run gradient drift seen while writing these was MLX's CPU direct conv writing past its
  output (a 7x7 depthwise conv on a 2x2 map) — fixed in the mlx-rs fork
  (`cpu-conv-border-clamp.patch`, upstream ml-explore/mlx af9a5924).
* `tests/restoration_train_lifecycle.rs` (both, miniature networks): every term contributes to the
  generator update; overfitting two pairs improves PSNR and LPIPS; cancel → resume bit-exact incl.
  critic, spectral buffers and both optimizers; export → `iris_3b_restore` ×2; other tasks refuse.
* Real weights (`tests/restoration_train_real_weights.rs`, `#[ignore]`): pinned LPIPS-VGG and
  ConvNeXt-XXL against `iris_restoration_train_real.safetensors`
  (`../tools/dump_iris_restoration_train_realweight.py`; Candle CPU: LPIPS ≤1.3e-8, trunk ≤2.1e-6 of
  peak), and one bounded train → export → restore run (dispatch-only `iris` CUDA profile,
  `iris-restore-mlx` macOS profile).
* Real-weight evidence (2026-10-11), one bounded LoRA run of the full recipe (rank 8, 256² crop, two
  steps = G, D, G, D, bf16, EMA) → merged `upscaler/` export → `iris_3b_restore` 64×48 → 256×192:
  CUDA (`iris` profile, run 38137202298): train 58.4 s, VRAM peak 29.3 GB, trunk ≤2.8e-6 / LPIPS
  ≤1e-6 of peak; MLX on nax-macos-2 (`iris-restore-mlx`, run 38140601912): train 8.1 s, guard peak
  25.2 GB (cap 40), trunk ≤2.8e-3 of peak (Metal), LPIPS ≤5e-7. The generator loss at the first step
  (~5e4) is the untrained critic read through spectral norm's random initial `u`/`v` (HYPIR's first
  generator window runs in eval mode, before any power iteration); it falls to ~5 by the second step.
  The paper's full fine-tune of the 3B generator needs f32 masters + grads + AdamW m/v + EMA ≈ 5 × 12 GB
  = 60 GB before activations, the critic and LPIPS — above this lane's 40 GB guard; it is exercised on
  the miniature (`restoration_train_lifecycle`).


## Memory-strategy contract (sc-25690)

Upstream ships no memory policy; this is the repo's shared ladder (`gen_core::MemoryProviderContract`)
declared for what each MLX route actually runs. `src/memory_strategy.rs` owns all three contracts, and
`register_providers` registers each with its weights-free fixture and behavior seam. `iris_3b` is a
registry generator; depth (a provider-specific API) and restoration (a transform) are registered as
composed routes.

| | `iris_3b` | `iris_3b_depth` | `iris_3b_restore` |
|---|---|---|---|
| Memory mode key | `text_to_image` | `depth` | `restoration` |
| Resident | implemented | implemented | implemented (one backbone forward per export tile, always) |
| StagedResidency | implemented (text encoder → encode → drop → backbone) | not applicable (no text encoder) | not applicable (no text encoder) |
| BoundedDecode | not applicable (pixel space) | not applicable | not applicable |
| BoundedAttention | missing | missing | missing |
| BoundedTransformerResidency | missing | missing | missing |
| Calibration fingerprint | `iris-3b-mlx-derived-2026-10-12-v1` | `iris-3b-depth-mlx-derived-2026-10-12-v1` | `iris-3b-restore-mlx-derived-2026-10-12-v1` |
| Witnessed load surfaces | `bf16:resident:eager`, `bf16:sequential:eager` | same | same |
| Admission geometry | request width × height; batch = prompts × count | `depth_memory_geometry(plan)` (the model grid) | `restoration_memory_geometry(plan)` (the processing size); `restoration_forward_pixels(plan)` is the one-tile forward bound |
| Resident bytes (materialized snapshot) | text tower (`model.language_model.*` + untied `lm_head`, bf16) + backbone at the compute dtype + adapter overlays | `empty_prompt.safetensors` (stored f32) + backbone and reducer at the compute dtype | `empty_prompt.safetensors` + backbone at the compute dtype |

Staged residency is a per-request decision on generation: the request scope writes
`GenerationMemory::stage_residency`, and a `Sequential` load stages every request. The scope bounds
`prompts × count` by the admitted batch (a prompt batch denoises its rows together), and
`Iris3b::validate` refuses the controls the route has no mechanism for (`tile_vae_decode`,
`chunk_attention`, `stream_transformer_blocks`). No loader has a deferred-materialization load
shape: all three refuse `LoadShape::DeferredMaterialization` at load, and a deferred or quantized
spec has no contract. The loaded models publish the same contract their registration builds and the
registered safety check (`Generator::memory_strategy_contract` / `memory_strategy_safety_check` for
generation; the inherent methods of the same names on `IrisDepth` and `IrisRestorer`). An adapted
generation load prices its stack as the typed overlay components
`iris_3b.adapters.forward_residuals` and `iris_3b.adapters.materialized_deltas`, and its behavior
fixture carries the overlay key `lora`. Nothing numeric beyond header-priced resident bytes is published, and no activation
anchor is registered: those are measurements the epic's terminal campaign owns.

Verification: `memory_strategy` unit tests (declaration, admission gate, pricing, registry
conformance) and `tests/memory_contract.rs` (a staged selection on a warm generator loads the two
components in staged order and renders the identical image; the encoder-free loaders publish their
registered contract).
