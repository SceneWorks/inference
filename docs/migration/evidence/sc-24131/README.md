# sc-24131 — DeltaNet per-token state checkpoints: speculative rollback without a replay forward

Story S3 of epic sc-24128. Host: Windows 11, **RTX Pro 6000 / sm_120** (GPU 1, the lane shared with
S10's Qwen3-8B runs — see co-tenancy below), CUDA 12.9, MSVC 14.44, `CUDA_COMPUTE_CAP=120`, release
build with `--features cuda`. Model: `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
(`bonsai-qwen38-parent`), BF16 greedy, 97 prompt tokens, 256 new tokens per bench row — the same
fixture and harness as `sc-24129/`, `sc-24132/` and `sc-24130/`.

## What landed

* **One recurrent-cache implementation in `primitives/`** (E0): `primitives/gated_delta.rs`'s
  `DeltaNetCache` now owns the per-token checkpoint **ring** — a preallocated `[slots, B, K-1, C]`
  conv-tail tensor and a `[slots, B, Hv, Dv, Dk]` f32 SSM tensor per linear layer (`RingSpec`).
  Every forward writes the state after each of its last `slots` tokens **in place**
  (`Tensor::slice_set` into slot `position % slots`; no per-token allocation), the live state is a
  **view of the newest slot**, and `rollback_to(n)` is a **slot-index change — no copy**. The
  model file (`models/qwen35.rs`) only wires it: the layer runs its conv through
  `causal_depthwise_conv_traced` (which keeps the conv input so the tail after *every* token is a
  slice) and hands the recurrence to `DeltaNetCache::advance`.
* **The checkpoint is an output of the recurrence step.** `gated_delta_recurrence_with_sink`
  calls `sink(ti, state_after_token_ti)` after every token; `advance` writes the ring from that
  sink. This is the contract a fused DeltaNet decode kernel (sc-24000, epic 23989) keeps by
  writing each token's state into its ring slot directly — the ring's slot addresses are stable
  for the kernel and for the CUDA-graph runner (S6) alike.
* **The S1 start-of-step checkpoints are gone**, replaced (not supplemented) by the ring: no
  `DeltaCheckpoint` list, no `Qwen35Cache::checkpoint()` at forward start. Admission prices the
  ring **exactly** (E6): `Qwen35Model::recurrent_state_bytes(states)` — per linear layer, one
  slot is `48 × 128 × 128 × 4 B` of SSM state plus a `3 × 10240 × 2 B` bf16 conv tail — is what
  `Qwen35Cache::recurrent_bytes()` reports for that cache, and the provider's
  `step_memory_geometry(drafts)` charges `K + 2` slots for a `K`-draft request (the step start
  plus the `K + 1` verify positions). The old term was an upper bound over all 64 layers with a
  conv tail sized as `Hv × Dv × kernel`; the new one is the allocation. Allocation fails closed:
  `StepModel::new_cache_for` preallocates every ring and returns the device's error.
* **Rollback contract**: any position inside the last verify step (and the step start) is
  restorable; anything older is the typed `Error::RollbackUnavailable { n, have }` — the engine's
  fallback path is untouched, it just never fires on this cache. `retain_checkpoints` on the
  `DecodeCache` trait now returns `Result` (deepening a ring allocates).
* **Telemetry (E2)**: on top of S2's `replays` / `replay_forwards` (the `RollbackUnavailable`
  → replay fallback count), `SpeculativeStats` / `DecodeRecord` carry `direct_rollbacks`,
  `DecodeRecord::target_forwards_per_verify_step()` is
  `(verify_steps + replay_forwards) / verify_steps`, and the bench table gains `fwd/verify` and
  `replay forwards` columns (JSON: `verify_steps`, `direct_rollbacks`, `replay_forwards`,
  `target_forwards_per_verify_step`).

## AC1 — rollback to every position of a verify step restores the exact DeltaNet state

**Tiny config (f32, CPU), exact.** `models::qwen35::tests::verify_step_rollback_to_every_position_matches_a_fresh_decode`:
for every `K in 1..=5` and every `j in 0..=K + 1`, after a `K + 1`-token verify forward,
`rollback_to(prompt + j)` leaves every linear layer's conv tail and SSM state at max abs error
**0.0** (gate `1e-6`) against a fresh token-at-a-time decode of `j` tokens on the reference cache;
the rings' addresses are unchanged across the verify step and every rollback; the next token then
decodes within `1e-5` of a fresh same-kind cache (the residual is the CPU GEMM's row-count
rounding, S2's finding, not the state). At the primitive,
`primitives::gated_delta::tests::ring_restores_every_position_of_a_verify_step_exactly` does the
same over the recurrence alone (also exactly 0), and
`sink_receives_every_post_token_state_in_order` pins the per-token sink to the prefix recurrence.

**Real weights (Qwen3.8-27B, BF16, GPU 1)** — `deltanet-ring-real-weight.log`
(`tests/deltanet_ring.rs`, K = 1..5, every j):

| oracle | conv max abs err | ssm max abs err |
|---|---|---|
| `j = 0` vs the prompt-only state | **0** | **0** |
| `j = K + 1` vs a ring-less reference cache fed the same `K + 1`-token forward | **0** | **0** |
| interior `j` vs a second ring cache fed the same first `j` tokens + a *different* suffix | **0** | **0** |
| any `j >= 1` vs a fresh **token-at-a-time** decode of `j` tokens (the literal wording) | 0.25 – 0.625 (one bf16 ULP of the conv tail's `in_proj_qkv` rows at their magnitude) | 0.026 – 0.044 |
| ring-free envelope: the reference cache after one `K + 1`-token forward vs after `K + 1` single-token forwards (no ring involved) | 0.25 – 0.625 | 0.026 – 0.044 |

The ring is exact under every oracle whose arithmetic is the verify forward's own; the literal
token-at-a-time oracle differs by exactly the class of error the **ring-free** envelope shows,
because a `M = K + 1`-row cuBLAS projection GEMM rounds its rows differently from `M = 1` — the
root cause S2 documented for the knife-edge (`sc-24130/README.md`). That envelope is a property
of the projections, not of the rollback: at `j = K + 1` the ring's literal error *equals* the
envelope to every digit (the ring holds the reference's own state), and at every interior `j`
the restored slot is closer to the fresh `j`-token state than to the `j ± 1`-token states
(a wrong slot would be closest to a neighbour). The literal `<= 1e-6` is therefore met on the f32
tiny config and under the same-arithmetic oracles on the 27B; on BF16 weights against a
token-at-a-time oracle it cannot be met by any rollback mechanism, and the gate the test enforces
there is the same-arithmetic exactness plus neighbour discrimination within the envelope.

## AC2 — one target forward per verify step, zero replay forwards, K = 1..5

`decode-bench/head-1d0f5cd9a/` (sealed at the PR's head, after S2's fix pass was merged;
`comparison.md` sets it beside S2's sealed head `bd33b65a7` and the earlier sealed run of this
story, `head-6937eff53`, taken before that merge — tokens, acceptance, `fwd/tok` and every
recovery counter are identical between the two runs of this story):

| row | tok/s | acceptance | fwd/tok | syncs/verify | **fwd/verify** | **replay forwards** | direct rollbacks / verify steps | first divergence |
|---|---|---|---|---|---|---|---|---|
| MTP off (reference, growing kv) | 14.48 | n/a | 1.000 | n/a | n/a | n/a | n/a | (ref) |
| MTP off (StepModel, static kv) | 14.61 | n/a | 1.000 | n/a | n/a | n/a | n/a | yes (identical) |
| MTP K=1 | 18.86 | 0.827 | 0.551 | 1.00 | **1.00** | **0** | 24 / 140 | 124 (exact bf16 tie) |
| MTP K=2 | 24.24 | 0.757 | 0.402 | 1.00 | **1.00** | **0** | 32 / 102 | 124 (exact bf16 tie) |
| MTP K=3 | 23.16 | 0.569 | 0.375 | 1.00 | **1.00** | **0** | 59 / 95 | 125 (knife-edge) |
| MTP K=4 | 22.80 | 0.467 | 0.355 | 1.00 | **1.00** | **0** | 71 / 90 | 106 (exact bf16 tie) |
| MTP K=5 | 22.96 | 0.465 | 0.305 | 1.00 | **1.00** | **0** | 65 / 77 | 124 (exact bf16 tie) |

* **`fwd/verify` is 1.00 and `replay forwards` is 0 on every row**, across the whole 256-token
  run at every K (S2's head paid 2 forwards on every partially rejected step: 24–71 of the
  77–140 verify steps here were partial rejections, each now a direct rollback). The in-test
  gate (`[ac2]` in `deltanet-ring-real-weight.log`, 48 tokens) reports the same counters.
* **E1 parity**: the StepModel row is token-identical to the reference; every MTP row's first
  divergence (124, 124, 125, 106, 124) is one of S2's enumerated reference knife-edge positions
  (`51, 74, 99, 102, 103, 106, 107, 124, 125, 154, 161, 177, 186, 202, 243`); no new position.
  Acceptance rates are a property of the sequence actually decoded, so past the divergence they
  are not the same statistic as S2's rows (K=1 reproduces 0.827; K=2..5 part from the reference at
  or before S2's rows did and land at 0.757 / 0.569 / 0.467 / 0.465 vs 0.668 / 0.640 / 0.549 /
  0.480) — the like-for-like comparison is `fwd/tok`, which is what the removed replay changes.
* **tok/s (honest reading)**: this run's reference is 14.48 tok/s against S2's 11.21 — the base
  branch has since gained the fused primitives (S7) and the NVFP4/sampler work (S5/S8), and this
  is a different GPU on a different day, so cross-run absolute numbers are not the evidence;
  within-run ratios are. S2's MTP rows sat at **+27–31 %** over their reference (K=1..4;
  +22 % at K=5); this run's sit at **+30 % (K=1) and +57–67 % (K=2..5)** — the
  partial-rejection-heavy rows gain the most, as expected from removing one target forward per
  rejected step: `fwd/tok` fell from 0.645 / 0.613 / 0.535 / 0.535 / 0.547 (S2) to
  0.551 / 0.402 / 0.375 / 0.355 / 0.305 at K=1..5. The earlier run of this story
  (`head-6937eff53`: reference 15.66, K=1..5 21.57 / 21.54 / 24.58 / 24.61 / 24.53) shows the
  same counters and the same within-run gain band; the absolute spread between the two runs
  (10–15 %) is this box's clock drift, as `sc-24132/README.md` documents.
* **Memory**: the StepModel row's final cache reports 168.9 MiB live / 146.8 MiB checkpoints
  (one ring slot: `generate_step` asks for overshoot 0, depth 1) against S2's 168.9 / 293.6 (two
  start-of-step clones). An MTP request's ring is `K + 2` slots × 146.8 MiB (K=1: 440 MiB, K=5:
  1.03 GiB; the `[ac2]` lines show the exact per-K figures), priced as such at admission — more
  than S1's flat three-state bound from K=3 up, which is the honest cost of restoring any
  position of a K-draft verify without a replay.
* **Co-tenancy**: GPU 1 is the shared lane. `run.json` records the co-tenants at start — for
  `head-6937eff53` another story's CUDA test executable
  (`sc-24138-target-cuda\debug\deps\tools-*.exe`) plus the desktop compositor processes, for
  `head-1d0f5cd9a` only the compositor processes; per-process memory is not exposed by the WDDM
  driver. GPU 1 held < 1.4 GiB before every run here.

## Weights-free gates

* Git Bash: `cargo test --locked -p candle-llm --lib` (311 passed, 9 ignored; new: the ring's
  primitive tests, the tiny-config AC1 sweep, the engine's `direct_rollbacks` /
  `replays` counters with forced partial rejections at K=1..5 and a step-start-only mock
  cache that still replays, admission pricing equal to the ring bytes for K=0..5), CPU clippy
  `-D warnings`, `cargo fmt --check`, CPU rustdoc `-D warnings`, `check-workspace.py`,
  `check_docs.py`, `pytest scripts/tests/test_decode_bench.py` (13 passed).
* PowerShell (MSVC 14.44 vcvars, `CUDA_COMPUTE_CAP=120`, `CUDA_VISIBLE_DEVICES=1`):
  `cargo test --locked --lib --tests -p candle-llm --features cuda`, CUDA clippy `-D warnings`,
  CUDA rustdoc `-D warnings` — see the PR for the run summary.

## Files

* `deltanet-ring-real-weight.log` — `tests/deltanet_ring.rs` on the 27B, K = 1..5 (AC1 oracles
  per j, the ring-free envelope, the `[ac2]` engine counters and ring sizes).
* `decode-bench/head-1d0f5cd9a/` — the sealed bench run at the PR's head (reference, StepModel,
  MTP K=1..5); `decode-bench/head-6937eff53/` — the same suite before S2's fix pass was merged;
  `decode-bench/comparison.md` — both beside S2's sealed head `bd33b65a7`.

## Reproduce

```text
BONSAI_QWEN38_SNAPSHOT=E:\huggingface\hub\models--Qwen--Qwen3.8-27B\snapshots\1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
DELTANET_RING_DRAFTS=1,2,3,4,5 CUDA_VISIBLE_DEVICES=1
  cargo test --release --features cuda -p candle-llm --test deltanet_ring -- --ignored --nocapture

python scripts/release/decode_bench.py run --binary <decode_bench exe> --run-name head-<sha> \
  --runtime-sha <sha> --snapshot <snapshot dir> --output docs/migration/evidence/sc-24131/decode-bench/head-<sha> \
  --gpu-index 1 --rows reference,step_model,mtp --drafts 1,2,3,4,5
python scripts/release/decode_bench.py table docs/migration/evidence/sc-24130/decode-bench/head-bd33b65a7 \
  docs/migration/evidence/sc-24131/decode-bench/head-<sha> --output docs/migration/evidence/sc-24131/decode-bench/comparison.md
```
