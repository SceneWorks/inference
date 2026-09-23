# sc-24134 — Shared CUDA-graph runner over `StepModel` (decode + verify), POC-gated: evidence

Story S6 of epic sc-24128. Host: Windows 11, **RTX Pro 6000 / sm_120** (GPU 0), CUDA 12.9 driver, MSVC 14.44,
`CUDA_COMPUTE_CAP=120`, release build with `--features cuda`. Model: `Qwen/Qwen3.8-27B` @
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (`bonsai-qwen38-parent`), BF16 greedy, 97 prompt tokens, 256 new
tokens per row — the same fixture and harness as `docs/migration/evidence/sc-24132/`.

## Outcome: closed with findings (E8) — runner landed, opt-in off, no win measurable on this candle revision

The runner (`crates/llm/candle-llm/src/decode/graph.rs`, `GraphRunner`) captures, verifies and replays a
step model's decode and verify steps end to end — proven bit-exact on a synthetic step model built from
capturable ops (below) — but **no Qwen3.5/3.8 step is replayable at candle `1e6aa85e`**: every candle CUDA
op that takes a layout uploads its `[dims, strides]` from a transient host `Vec` inside the step, and the
driver records that as a memcpy node that re-reads freed host memory at every replay. The runner's census of
every captured graph refuses such a graph by name (`host_upload_in_capture`) before anything is launched, so
graphs-on is token-identical to eager (AC1) by falling back, decode_bench records graphs-on vs graphs-off with
the reason (AC2: no win → **default off**, `CANDLE_LLM_CUDA_GRAPHS=1` opts in), and every refusal is a named
fallback that never fails the request or changes device (AC3).

## POC findings (step 0), in the order the story asked

1. **Allocator.** candle `1e6aa85e` allocates every temporary through cudarc 0.19 `CudaStream::alloc`, which is
   `cuMemAllocAsync` when the context reports memory-pool support (`has_async_alloc() == true` on this device).
   Inside a capture the temporaries become graph memory nodes, allocated and freed inside the graph — no
   pre-planned workspace is needed. (`poc_allocator_is_stream_ordered_and_the_stream_is_capturable`.)
2. **The stream.** `Device::new_cuda` puts the model on the legacy NULL stream, which **cannot be captured**
   (`cuStreamBeginCapture` → `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`). `select_device` now builds the model with
   `Device::new_cuda_with_stream` and switches cudarc's per-slice event tracking off (one stream: nothing to
   order; its `cuStreamWaitEvent` on events recorded before the capture would invalidate it). The runner
   refuses a context that still tracks events (`event_tracking`).
3. **cuBLAS / cuBLASLt.** cuBLAS GEMMs capture and replay bit-exactly with the handle candle created on that
   stream (no `cublasSetWorkspace` needed at this toolkit; the NVFP4 cuBLASLt path holds its own persistent
   32 MiB workspace). A cuBLAS call from *another* thread during a `THREAD_LOCAL` capture fails the captured
   call with `CUBLAS_STATUS_INTERNAL_ERROR` — a named `step_failed_in_capture` fallback, seen only when the
   CUDA unit tests were run multi-threaded (they are single-threaded under cargo).
4. **Host traffic.** A pageable host→device upload inside a capture invalidates it (`capture_invalidated`); a
   device→host read (`to_vec`) is silently recorded as a memcpy node into a buffer that no longer exists — the
   census names it `host_read_in_capture`, and this crate's `note_host_sync` counter names a noted read
   `sync_in_capture`. The S5 sampler's argmax copy therefore stays outside the captured region (the runner
   captures the forward only; sampling runs on the copied-out logits).
5. **Position as data.** Inside a captured Qwen3.5/3.8 step the positions are still Rust-side scalars
   (`cos_sin(offset)` builds the RoPE tables on the host, `slice_set(offset)` writes the KV, `narrow(len)`
   bounds attention) — each is a host upload the census refuses. The data-driven form the runner needs
   (`capacity_mask` + `sdpa_gqa_masked` over the full preallocated buffer, a device position tensor read by
   `index_select` / `scatter_set`) is landed as primitives with its bit survey, and is **not** wired in: on
   this candle revision `index_select`, `scatter_set`, comparisons and broadcasts *themselves* upload their
   layouts, and the masked full-capacity attention is **not bit-identical** to the bounded `narrow` view —
   cuBLAS selects its kernel by the key extent (survey below: 40 / 384 decode lengths and 122 / 381 verify
   lengths differ by one bf16 ULP), so wiring it would move the static path's bits for no replay in return.
   A data-driven length needs an attention kernel whose arithmetic does not depend on the extent (a seam
   kernel), not a mask over cuBLAS.
6. **What breaks capture, precisely.** `candle-core/src/cuda_backend/mod.rs`: `SlicePtrOrNull::params_from_layout`
   (unary / affine / copy on a non-contiguous operand), the reductions, `index_select`, gather / scatter /
   index_add, `where_cond`, the comparisons and the binary ops on non-contiguous operands each call
   `dev.clone_htod(&[dims, strides].concat())` — a pageable `cuMemcpyHtoDAsync` from a `Vec` that is dropped
   when the op returns. Contiguous unary / binary / affine ops, cuBLAS matmuls, `softmax_last_dim`, `copy2d`
   (`slice_set`) and the nvrtc-seam kernels (scalar arguments) capture cleanly. **The candle change that would
   fix it:** pass layouts by value as kernel parameters, or cache the `[dims, strides]` device buffers per
   layout, so no per-op host upload exists. (`poc_contiguous_ops_replay_bit_exact_and_index_select_is_refused`,
   `poc_single_op_census`.)
7. **State that outlives the step.** A tensor allocated inside the capture and kept (the S1 hybrid cache's
   replaced DeltaNet state) is an allocation node with no free node — `allocation_escaped_capture`; dropping a
   tensor allocated *before* the capture inside it (`cuMemFreeAsync` on foreign memory) fails the step outright
   (`step_failed_in_capture`); `Tensor::copy` inside a capture returns `CUDA_ERROR_INVALID_VALUE`. The hybrid
   cache declares `deltanet_state_unstable` before any capture; S3's stable-address ring lifts that
   declaration, after which the layout uploads (6) are what remains.

## What the runner does (E0, E2, E4, E5, E6)

* One runner over any `StepModel` + `DecodeCache` (`decode/graph.rs`); no graph logic in model files. The
  seams model files implement: `StepModel::graph_support` (a MoE router's host read → `moe_router_host_read`),
  `DecodeCache::{graph_support, stage_positions, replay_advance, graph_identity}`; `StaticKvCache::advance`.
* Per step shape (token count `M`, logits scope, hidden wanted): eager warm-up (kernels compile outside any
  capture) → capture → **census** (refuse host uploads / host reads / escaped allocations by name) →
  instantiate (`cuGraphInstantiateWithFlags(0)`, explicit upload) → first launch + bit-exact self-check
  against the eager step at the same position → verified first replay at a new position (the check that
  catches a stale scalar or an unstable address) → replay. Outputs are copied into preallocated staging
  tensors inside the capture and handed back as copies; the two self-checks cost exactly one host sync each,
  once per shape.
* Every eager step through the runner is counted with its reason on the per-thread `GraphTally` →
  `DecodeRecord::cuda_graphs` (`graph: <label> replayed=<n> eager=<n> captured=<n> fallback=<reason>`), the
  provider's record, and decode_bench's `cuda_graphs` column.
* Without `cuda` the runner is a pass-through (`cuda_feature_off`); off-device `not_cuda`; the switch off
  `disabled`; prefills `shape`. `GraphRunner::workspace` reports the staging bytes and the driver's graph
  memory reservation delta for admission (E6); a failed staging allocation or instantiation is a named
  fallback (`instantiate_failed`), never an OOM mid-decode.

## Runner proof on a capturable step model (CUDA unit tests, GPU 0)

`decode::graph::cuda_tests` — a synthetic recurrent step model built only from contiguous ops, cuBLAS
matmuls, `tanh`, `cat` (`copy2d`) and in-place `slice_set` (what candle can replay today):

| test | result |
|---|---|
| `synthetic_decode_through_the_runner_is_token_identical_and_replays` | 40 greedy tokens identical to eager; `graph: mixed replayed=37 eager=5 captured=1`; census `nodes=20 kernels=8 memcpy(dtod=0, htod=0, dtoh=0) alloc=6 free=6 escaped=0`; staging 100 B |
| `synthetic_speculative_k3_through_the_runner_is_token_identical_and_replays` | K = 3 through the engine (4-token verify steps, rollbacks, replay-forwards): 48 tokens identical; `graph: mixed replayed=41 eager=11 captured=2`; one sync per verify step plus exactly two self-check syncs per captured shape |
| `synthetic_misbehaviour_falls_back_with_a_named_reason` | declared-unstable cache → `mock_cache_declared_unstable`; a noted host read during capture → `sync_in_capture`; a kept allocation → `allocation_escaped_capture` (census `alloc=7 free=6 escaped=1`); every request finishes token-identical, nothing replayed, device unchanged |
| `qwen35_static_step_is_refused_by_the_census_with_layout_uploads` | tiny attention-only Qwen3.5 on the static cache: 1-token step census **`nodes=320 kernels=100 memcpy(htod=10) alloc=105 free=105`** → `host_upload_in_capture`, tokens identical; the hybrid config → `deltanet_state_unstable` before any capture |
| `poc_masked_capacity_attention_vs_narrow_survey` | 27B decode shape, bf16, cap 384: q=1 → 40 / 384 lengths differ (first 127, max Δ 2⁻⁸); q=4 → 122 / 381 (first 65) |
| `synthetic_replay_timing` (ignored, debug build) | 2048 steps: eager 300.0 µs/step, graphs 227.4 µs/step (`replayed=2045 eager=5 captured=1`) — the launch overhead a graph removes on an 8-kernel step |

Run: `RUST_TEST_THREADS=1 CUDA_VISIBLE_DEVICES=0 <candle_llm lib test exe> graph:: --nocapture --include-ignored`.

<!-- REAL-WEIGHT SECTIONS FILLED FROM THE RUNS BELOW -->
