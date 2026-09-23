# sc-24134 — Shared CUDA-graph runner over `StepModel` (decode + verify), POC-gated: evidence

Story S6 of epic sc-24128. Host: Windows 11, **RTX Pro 6000 / sm_120** (GPU 0), CUDA 12.9, MSVC 14.44,
`CUDA_COMPUTE_CAP=120`, release build with `--features cuda`. Model: `Qwen/Qwen3.8-27B` @
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (`bonsai-qwen38-parent`), BF16 greedy, 97 prompt tokens, 256 new
tokens per row. The fixture and harness are the same as `docs/migration/evidence/sc-24132/`.

## Outcome: closed with findings (E8). The runner is landed and opt-in; the default is **off**

The runner lives in `crates/llm/candle-llm/src/decode/graph.rs` (`GraphRunner`). It captures, checks and replays a
step model's decode steps (M = 1) and verify steps (M = K + 1) end to end. On a synthetic step model built from
capturable ops it is token-identical to eager and **~28 % faster per step** in release.

**No Qwen3.5/3.8 step can be replayed at candle `1e6aa85e`.** Three things block it:

- **Layout uploads.** A real 27B decode step records **3927 kernel launches and 851 host uploads**. Every candle
  CUDA op that takes a layout uploads its `[dims, strides]` from a temporary host `Vec`, and each upload is recorded
  as a memcpy that re-reads freed host memory at every replay.
- **Replaced state.** A decode step leaves **96 allocations alive past the step**: the 48 linear layers' conv and
  SSM states, which the S1 hybrid cache replaces instead of writing in place.
- **Scalar positions.** Positions are Rust-side scalars. `Qwen35Model::graph_support` declares this
  (`positions_host_scalar`), so no Qwen3.5/3.8 step is ever recorded by the runner.

How the three acceptance criteria come out:

- **AC1:** The 27B cache declares `deltanet_state_unstable` before any capture (the runner checks the cache's
  declaration before the model's), so graphs-on is token-identical to eager because it falls back to eager.
- **AC2:** decode_bench records graphs off vs on with that named reason. No win is measurable, so the default is
  off and `CANDLE_LLM_CUDA_GRAPHS=1` opts in.
- **AC3:** Every refusal is a named fallback. None fails the request or changes the device.

## POC findings (step 0)

1. **Allocator — works.** candle `1e6aa85e` allocates every temporary through cudarc 0.19's `CudaStream::alloc`.
   That call uses `cuMemAllocAsync` when the context reports memory-pool support (`has_async_alloc() == true` here).
   Inside a capture the temporaries become graph memory nodes, allocated and freed inside the graph. The step needs
   no pre-planned workspace. A contiguous POC step records `alloc=4 free=4`.
2. **Stream — the model's own stream only when graphs are on; the legacy stream stays the default.**
   - `Device::new_cuda` puts the model on the legacy NULL stream, which cannot be captured: `cuStreamBeginCapture`
     fails with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.
   - `Device::new_cuda_with_stream` gives the model its own non-blocking stream; `select_device` then turns off
     cudarc's per-slice event tracking. While tracking is on, cudarc creates two CUDA events per allocation and waits
     on them across streams, and a wait on an event recorded before a capture invalidates it.
   - Tracking off means nothing orders the model's stream against any other stream, so every tensor its kernels touch
     must come from that one device. That is not safe as a process-wide default:
     - StarVector-1B used to build a fresh device per request for its pixels — on the own stream a second stream, whose
       conv and `cat` writes into memory allocated on the model's stream nothing orders (candle's per-op device check
       compares the GPU ordinal only). It now keeps the load-time device.
     - candle-flash-attn launches its kernels on stream 0, which does not synchronize with a non-blocking stream.
   - So `select_device` keeps the **legacy stream by default** (`CudaStreamKind::resolve`). It builds the own stream
     only when the CUDA-graph runner is switched on at that moment (`CANDLE_LLM_CUDA_GRAPHS=1` at load) or
     `CANDLE_LLM_CUDA_STREAM=own` asks for it, and never in a `flash-attn` build (the runner refuses such a build as
     `flash_attn_stream`). A model on the legacy stream is refused as `legacy_stream`.
   - **Measured — no speed difference either way.** A 3× A/B of the two streams on this host (graphs off, 256 tokens,
     every row token-identical):

     | row | own stream tok/s | legacy stream tok/s |
     |---|---|---|
     | step_model | 14.91 / 14.68 / 15.10 | 14.72 / 14.14 / 14.73 |
     | reference | 14.81 / 14.62 / 14.69 | 14.94 / 15.06 / 13.92 |

     Run to run, one configuration spreads by about 4 % (up to 8 % with the 13.92 reference run), and the two streams
     overlap. The stream is chosen for capture and safety, not for speed.
3. **cuBLAS / cuBLASLt — works.**
   - GEMMs capture and replay bit-exactly on the handle candle created on the stream. No `cublasSetWorkspace` is
     needed with this toolkit, and the NVFP4 cuBLASLt path keeps its own persistent 32 MiB workspace.
   - One caveat: a cuBLAS call from another thread during a `THREAD_LOCAL` capture fails with
     `CUBLAS_STATUS_INTERNAL_ERROR`, which is named `step_failed_in_capture`. This showed up only when the CUDA unit
     tests ran multi-threaded; cargo runs them single-threaded.
4. **Host traffic.**
   - A pageable host→device upload inside a capture is either refused by the driver (`capture_invalidated`) or
     recorded as a host-sourced memcpy node (`host_upload_in_capture`). Both runs are in the logs.
   - A device→host read (`to_vec`) is silently recorded into a buffer that no longer exists. The census names it
     `host_read_in_capture`, and this crate's `note_host_sync` counter names a noted read `sync_in_capture`.
   - The S5 sampler's scalar copy therefore stays outside the captured region. The runner captures only the forward
     step; sampling runs on the logits it copies out.
5. **Positions as data — not viable on this revision.** Inside a Qwen3.5/3.8 step the positions are Rust-side
   scalars: `cos_sin(offset)` builds RoPE tables on the host, `slice_set(offset)` writes the KV, and `narrow(len)`
   bounds attention. `Qwen35Model::graph_support` declares it (`positions_host_scalar`).
   - The POC measured the data-driven form — attention over the full preallocated buffer with a device length mask
     (built with a comparison) — and did **not** keep it: the experiment's primitives are removed again, since
     nothing could call them.
   - It cannot be wired in on this revision because `index_select`, `scatter_set`, comparisons and broadcasts upload
     their own layouts.
   - The masked full-capacity attention is also **not bit-identical** to the bounded `narrow` view. On the 27B shape
     (measured at `7699d507a`), 40 of 384 decode lengths and 122 of 381 verify lengths differ by one bf16 ULP, because
     cuBLAS picks its kernel by the key extent. Wiring it in would move the static path's bits and still not make the
     step replayable.
   - A data-driven length needs an attention kernel whose arithmetic does not depend on the extent (a seam kernel),
     not a mask over cuBLAS.
6. **What breaks capture, precisely.** In `candle-core/src/cuda_backend/mod.rs`, these call
   `dev.clone_htod(&[dims, strides].concat())` — a pageable `cuMemcpyHtoDAsync` from a `Vec` dropped when the op
   returns:
   - `SlicePtrOrNull::params_from_layout` (unary, affine and copy on a non-contiguous operand)
   - the reductions
   - `index_select`
   - gather, scatter and index_add
   - `where_cond`
   - the comparisons
   - binary ops on strided operands

   These capture cleanly: contiguous unary / binary / affine ops, cuBLAS matmuls, `softmax_last_dim`, `copy2d`
   (`slice_set`, `cat`), and the nvrtc-seam kernels (scalar arguments). **The candle change that would fix it:** pass
   layouts by value as kernel parameters, or cache the per-layout device buffers, so no op uploads from the host.
7. **State that outlives the step.**
   - A tensor allocated inside the capture and kept (the hybrid cache's replaced DeltaNet state) becomes an allocation
     node with no free node. The census reports `escaped`, and the runner refuses it as
     `allocation_escaped_capture`.
   - Dropping a tensor inside the capture that was allocated before it fails the step outright: `cuMemFreeAsync`
     returns `CUDA_ERROR_INVALID_VALUE`, named `step_failed_in_capture`. A rollback checkpoint pruned mid-step does
     this.
   - `Tensor::copy` inside a capture also returns `CUDA_ERROR_INVALID_VALUE`.
   - The hybrid cache declares `deltanet_state_unstable` before any capture. S3's stable-address ring lifts that
     declaration; after it, finding 6 is what remains.

## What the runner does (E0, E2, E4, E5, E6)

- **Structure.** There is one runner over any `StepModel` + `DecodeCache`, and model files carry no graph logic —
  only declarations. The seams are:
  - `StepModel::graph_support`: `Qwen35Model` declares `moe_router_host_read` (the MoE router's host read) and
    otherwise `positions_host_scalar`.
  - `DecodeCache::{graph_support, stage_positions, replay_advance, graph_identity}`. `Qwen35Cache` declares
    `deltanet_state_unstable` / `growing_kv`; it has no replay bookkeeping (its model refuses first). Only the
    synthetic test cache implements `replay_advance`.
- **Capability checks before any capture (E5).** The runner checks, in order: switch, `cuda` feature, CUDA device,
  not a `flash-attn` build, capturable stream, stream-ordered allocator, event tracking off, the cache's declaration,
  then the model's.
- **Per step shape** (token count `M`, logits scope, hidden wanted):
  1. Eager warm-up, so kernels compile outside any capture.
  2. Capture.
  3. **Census** of the recorded graph. Host uploads, host reads and escaped allocations are refused by name.
  4. `cuGraphInstantiateWithFlags(0)` and an explicit upload.
  5. First launch, plus a bit-exact self-check against the eager step at the same position.
  6. First replay at a *new* position, verified against eager. This catches a stale scalar or an unstable address.
  7. Replay.

  Each self-check runs the eager step first, rolls back, runs the graph at the same position, compares, then rolls
  back and re-runs eager, so the cache always holds the eager state. Outputs are copied into preallocated staging
  tensors inside the capture and handed back as copies. The two self-checks cost exactly one host sync each, once per
  shape. A capture ends even if the step panics (a drop guard ends it).
- **Refusal and fallback (AC3).** A refusal drops **every** graph and staging tensor the runner holds. The failed
  step rolls the cache back to its start and runs eager; this covers staging (`staging_failed`), the census,
  instantiation, launches, and a self-check that cannot run (`self_check_failed`) or disagrees (`replay_mismatch`).
  The one case that fails the step is a cache that cannot roll back to a position it restored or checkpointed earlier
  in the same step (`rollback_unavailable`): then neither the graph's state nor the eager one can be trusted.
- **Telemetry (E2).** Every step through the runner is counted with its reason on the per-thread `GraphTally`. It
  is reported as `DecodeRecord::cuda_graphs` (`graph: <label> replayed=<n> eager=<n> captured=<n>
  fallback=<reason>`), in the provider's record, and in decode_bench's `cuda graphs` column.
- **Build variants (E4).** Without `cuda` the runner is a pass-through (`cuda_feature_off`).
- **Memory (E6).**
  - Admission prices `graph_workspace_admission_bytes`: staging plus one step's working set, for every shape the
    engine can present (`M = 1..=K+1`, two scopes). The provider adds this term when the runner is on and MTP runs.
  - `GraphRunner::workspace()` is telemetry: the staging bytes, and how much the device's graph-memory reservation
    grew across the runner's captures. The pool is device-wide, so that number is approximate (it reads 0 when a live
    graph's reservation already covers the step). A destroyed graph trims the pool (`cuDeviceGraphMemTrim`), so a
    runner that drops its graphs gives the reservation back (32 MiB for the synthetic step).
  - A failed staging allocation is the named fallback `staging_failed`; a failed instantiation `instantiate_failed`.

## AC1 — graphs on is token-identical to eager (real weights)

`tests/cuda_graphs.rs::ac1_graphs_on_is_token_identical_to_eager_for_spec_off_and_mtp_k3` passes at `dd91ecc9e`
(`parity-and-census.log`):

| row | graphs off vs on | graph tally with the runner on |
|---|---|---|
| spec off (step driver, static KV) | **256 / 256 identical** | `graph: eager replayed=0 eager=256 captured=0 fallback=deltanet_state_unstable` |
| MTP K = 3 (unified engine) | **256 / 256 identical**, acceptance identical (167 / 261) | `graph: eager replayed=0 eager=137 captured=0 fallback=deltanet_state_unstable` |

Identity holds because the runner refuses the 27B hybrid by name before any capture. The replay path's own
exactness (capture → replay bit-identical, decode and K = 3 verify) is proven on the synthetic step model below.

Regression checks:

- `static_kv_parity::ac1_static_kv_greedy_fixture_is_token_identical_to_attn_kv` (S4's AC1) still passes on the
  dedicated stream (`static-kv-ac1-regression.log`).
- Every row of the three sealed bench runs is token-identical to the sealed S2 head run
  (`sc-24130/decode-bench/head-bd33b65a7`).

## AC2 — decode_bench, graphs off vs on (sealed, `decode-bench/`)

Three sealed runs of the same binary at `7699d507a`. None had a co-tenant on GPU 0 (`gpu.co_tenants_at_start` is
`[]`). Full table: `decode-bench/comparison.md`. Re-verify the seals with
`python scripts/release/decode_bench.py table <run dirs>`.

| run | stream | graphs | reference tok/s | **spec off** tok/s | **MTP K=3** tok/s | syncs/tok (off / K=3) | cuda graphs (spec off / K=3) |
|---|---|---|---|---|---|---|---|
| `head-7699d507a-legacy-stream-graphs-off` | legacy (pre-story) | off | 15.18 | 15.49 | 19.33 | 1.00 / 0.35 | — |
| `head-7699d507a-graphs-off` | own | off | 13.41 | **13.30** | **18.85** | 1.00 / 0.35 | — |
| `head-7699d507a-graphs-on` | own | **on** | 15.01 | **14.71** | **18.30** | 1.00 / 0.35 | `0 replayed / 256 eager (deltanet_state_unstable)` / `0 replayed / 137 eager (deltanet_state_unstable)` |

(These runs predate the legacy-stream default: at `7699d507a` the own stream was the default, so the graphs-off run set
`CANDLE_LLM_CUDA_STREAM=own` explicitly, which still selects it.)

- **The noise envelope.** Run to run, one configuration of this code spreads by about 4 % on this host (the 3× stream
  A/B in finding 2: 14.68–15.10 and 14.14–14.73 tok/s for step_model). The graphs-off run's **13.41 reference row is
  an outlier**: the reference row never goes through the runner and ran 15.18 and 15.01 in the other two runs, and
  13.92–15.06 in the A/B. Its 13.30 spec-off row sits with it, so that run was slow as a whole.
- **What graphs-on actually ran.** Its spec-off and K = 3 rows ran the same eager steps as graphs-off (every step is
  a named fallback). 14.71 vs 13.30 is that slow graphs-off run; 18.30 vs 18.85 (3 %) is inside the spread.
- **Syncs.** Syncs per token are unchanged: no capture means no self-check sync.
- **Decision (E8).** No win is measured, so the runner stays **opt-in, default off**.
- **Stream.** The own-stream vs legacy-stream rows show no speed difference (finding 2). The own stream is used only
  when capture needs it.

## AC3 — fallback with a named reason, never a failure (samples)

From `graph-unit-tests.log` (the CUDA unit tests) and the AC1 run (`dd91ecc9e`):

```text
graph: eager replayed=0 eager=256 captured=0 fallback=deltanet_state_unstable      # 27B hybrid, spec off
graph: eager replayed=0 eager=8 captured=0 fallback=positions_host_scalar          # tiny Qwen3.5, attention-only static cache
graph: eager replayed=0 eager=8 captured=0 fallback=deltanet_state_unstable        # tiny Qwen3.5, hybrid static cache
graph: eager replayed=0 eager=24 captured=0 fallback=mock_cache_declared_unstable  # cache declares itself unstable
graph: eager replayed=0 eager=25 captured=0 fallback=sync_in_capture               # a noted device->host read during capture
graph: eager replayed=0 eager=25 captured=0 fallback=allocation_escaped_capture    # census alloc=7 free=6 escaped=1
graph: mixed replayed=1 eager=26 captured=0 fallback=replay_mismatch               # a Rust scalar baked into the capture
graph: mixed replayed=4 eager=9 captured=1 fallback=launch_failed                  # a launch fails mid-request
```

In each case the request finishes token-identical to the bare model, on the same device, and the runner keeps no
graph.

Which reasons have a test (CPU or CUDA unit tests unless noted):

- the capability refusals `disabled`, `not_cuda`, `cuda_feature_off`, `legacy_stream`; `flash_attn_stream` only
  compile-checked (`--features cuda,flash-attn`; its stream rule is unit-tested)
- the declarations `mock_cache_declared_unstable`, `positions_host_scalar`, `deltanet_state_unstable`
- the census `sync_in_capture`, `allocation_escaped_capture`; `host_upload_in_capture` and `host_read_in_capture` in
  the POC tests and `census_step`
- `replay_mismatch` (the synthetic `scalar` misbehaviour: the first replay at a new position disagrees)
- `shape` (`oversized_steps_are_named_shape_fallbacks`: a 17-token step runs eager; decode steps still capture)
- `launch_failed` (`a_failed_launch_rolls_back_and_falls_back_eager`, through a test-only launch-failure seam: the
  replay's bookkeeping is rolled back and every step's logits and cache length match the bare model)
- `reference_path` (provider: switch on, MTP off → `graph: none … fallback=reference_path`)
- `capture_invalidated` (the legacy stream in the POC test)
- no test: `staging_failed`, `self_check_failed`, `instantiate_failed`, `no_async_alloc`, `event_tracking`,
  `step_failed_in_capture`, `rollback_unavailable`

## The census of a real 27B step (`qwen38_27b_step_census`)

This records one step as a graph without launching anything:

```text
27B decode (1 token) step, 64 layers: nodes=14206 kernels=3927 memcpy(dtod=0, htod=851,  dtoh=0) alloc=4762 free=4666 escaped=96 — ~61.4 kernels, 13.3 host uploads per layer
27B verify (4 tokens) step, 64 layers: nodes=23995 kernels=5751 memcpy(dtod=0, htod=2386, dtoh=0) alloc=7977 free=7881 escaped=96 — ~89.9 kernels, 37.3 host uploads per layer
```

- `escaped=96` is the 48 linear layers × (conv state, SSM state) that the S1 cache replaces each step.
- `htod` is candle's per-op layout metadata plus the host-built RoPE tables.
- A replay would remove the host-side issue cost of ~3.9 k launches (and ~4.8 k allocation calls) per decode step. On
  the synthetic model, graphs removed about 4.4 µs of host cost per captured kernel. At the same rate that would be
  on the order of 17 ms of a ~65–75 ms 27B decode step. This is an **upper bound**, reachable only after the three
  blockers above are removed. It is not a measured win.

## Runner proof on a capturable step model (CUDA unit tests, `decode::graph::cuda_tests`)

This is a synthetic recurrent step model built only from contiguous ops, cuBLAS matmuls, `tanh`, `cat` (`copy2d`)
and in-place `slice_set` — what candle can replay today:

| test | result (release, `7699d507a`) |
|---|---|
| `synthetic_decode_through_the_runner_is_token_identical_and_replays` | 40 greedy tokens identical to eager. `graph: mixed replayed=37 eager=6 captured=1`. Census `nodes=20 kernels=8 htod=0 alloc=6 free=6 escaped=0`. Staging 100 B |
| `synthetic_speculative_k3_through_the_runner_is_token_identical_and_replays` | K = 3 through the engine (4-token verify steps, rejections, rollbacks, replay forwards): 48 tokens identical, acceptance identical. `graph: mixed replayed=41 eager=13 captured=2`. Exactly one sync per verify step, plus two self-check syncs per captured shape |
| `synthetic_misbehaviour_falls_back_with_a_named_reason` | Declared unstable / host read / escaping allocation / baked-in scalar → the four named reasons above. Tokens identical to the same model run bare, no graph kept, device unchanged |
| `oversized_steps_are_named_shape_fallbacks`, `a_failed_launch_rolls_back_and_falls_back_eager` | `shape` and `launch_failed`, as listed under AC3 |
| `a_refusal_drops_every_captured_shape` | A verified 1-token graph is dropped, with its staging, when the 2-token shape is refused |
| `a_panic_inside_the_capture_still_ends_it` | After a panic inside a capture the stream is out of capture mode and usable |
| `graph_memory_is_reported_and_trimmed_when_the_graphs_go` | From a trimmed pool, one capture reports `graph_reserved_bytes = 33554432` (the device's whole reservation); `reset` trims it back to 0 |
| `qwen35_steps_are_refused_by_declaration_and_the_census_finds_layout_uploads` | Tiny attention-only Qwen3.5 on the static cache → `positions_host_scalar`, no recording. `census_step` on a warmed 1-token step: `kernels=99 htod=11` → `host_upload_in_capture` (the runner's own recording at `7699d507a`, before the declaration existed, counted `kernels=100 htod=10`). The hybrid config → `deltanet_state_unstable` |
| `poc_contiguous_ops_replay_bit_exact_and_index_select_is_refused` | matmul + softmax + affine + fused QK-norm/RoPE: 2 replays at new inputs, bit-exact. Adding one `index_select` → `htod=1` → refused |
| `synthetic_replay_timing` | 2048 steps: **eager 125.5 µs/step, graphs 90.6 µs/step** (`replayed=2045`). An earlier release run at `6c07bcce3` measured 142.2 → 92.7 |

## What it would take for graphs to pay on Qwen3.8

These are ordered; each gate is visible as the runner's fallback reason:

1. **S3's stable-address DeltaNet ring** lifts `deltanet_state_unstable` (and the `escaped=96`).
2. **Positions as device data** lift `positions_host_scalar`: RoPE tables gathered from a device position, KV written
   at a device offset, attention over the capacity with a device length, through seam kernels whose arithmetic keeps
   the static path's bits. The model then drops its declaration and implements the cache's `replay_advance`.
3. **A candle revision that stops uploading layouts from host `Vec`s** lifts `host_upload_in_capture`, the census
   gate that follows.

The runner, the census, the self-checks, the telemetry and the admission term need no change for any of them. Each
step only removes a named reason.

## Reproduce

```sh
# PowerShell, vcvars64 14.44, CUDA_COMPUTE_CAP=120
cargo test --locked --release --features cuda -p candle-llm --test decode_bench --test cuda_graphs --test static_kv_parity --lib --no-run
# Git Bash (clean checkout at the binary's sha; outputs outside the checkout, then copied in)
SNAP=<Qwen3.8-27B snapshot dir>
for run in "legacy-stream-graphs-off CANDLE_LLM_CUDA_STREAM=legacy" "graphs-off CANDLE_LLM_CUDA_STREAM=own" \
           "graphs-on CANDLE_LLM_CUDA_STREAM=own CANDLE_LLM_CUDA_GRAPHS=1"; do
  set -- $run; name=$1; shift
  env "$@" python scripts/release/decode_bench.py run --binary <decode_bench exe> --run-name head-<sha>-$name \
    --runtime-sha <sha> --snapshot "$SNAP" --output <outside>/head-<sha>-$name \
    --rows reference,step_model,mtp --drafts 3 --gpu-index 0
done
python scripts/release/decode_bench.py table <the three dirs> --output comparison.md
CUDA_VISIBLE_DEVICES=0 BONSAI_QWEN38_SNAPSHOT="$SNAP" <cuda_graphs exe> --ignored --nocapture --test-threads=1
CUDA_VISIBLE_DEVICES=0 RUST_TEST_THREADS=1 <candle_llm lib exe> graph:: device:: starvector:: --nocapture
```

The sealed runs above were made at `7699d507a`, when the own stream was the default. At this revision the stream
follows `CudaStreamKind::resolve`, and the three commands still select the same streams (`CANDLE_LLM_CUDA_STREAM` is
explicit in each, and graphs-on also switches the runner on). The `cuda_graphs` tests select their device with the
runner switched on, so both of their rows run on the own stream as before.
