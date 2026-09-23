# sc-24130 — Unified speculative engine (MTP / n-gram / draft proposers): evidence

Story S2 of epic sc-24128. Host: Windows 11, **RTX Pro 6000 / sm_120** (GPU 0), CUDA 12.9, MSVC 14.44,
`CUDA_COMPUTE_CAP=120`, release build with `--features cuda`. Model: `Qwen/Qwen3.8-27B` @
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (`bonsai-qwen38-parent`), BF16 greedy, 97 prompt tokens,
256 new tokens per row — the same fixture and harness as `sc-24129/` and `sc-24132/`. Every run here had
**no co-tenant on GPU 0** (`gpu.co_tenants_at_start` in `run.json`).

## What landed

* **One speculative loop** — `crates/llm/candle-llm/src/decode/engine.rs` (`generate_speculative`)
  over the S1 seams (`StepModel` + `DecodeCache`) with a `Proposer` trait; the proposers in
  `decode/proposers.rs`: `MtpProposer` (the `Qwen35Mtp` head, which stays in the model file),
  `NgramProposer` (prompt lookup), `DraftModelProposer<D: StepModel>` (a second model), and
  `NoProposer` (the token-at-a-time loop, `proposer=none`). The `qwen_mtp.rs` loop's verify /
  accept / replay logic moved into the engine and that file is deleted; the pre-epic `CausalLm`
  loops in `decode/speculative.rs` stay for the llama family until S10. The provider dispatches
  MTP requests through the engine (on the static KV cache), and n-gram now runs on qwen35.
* **Backend-neutral policy in core-llm** (`crates/contracts/core-llm/src/speculative.rs`):
  `ProposerKind`, `MtpPlan` / `resolve_mtp_plan` (mode resolution: `Off`; `Auto` → the advertised
  width or `none`; `Enabled` → the requested width) and `greedy_commit`; the stochastic rule is the
  unchanged `accept_token` (E7).
* **The verify decision is one device→host transfer** (AC2): a plain greedy run keeps its drafts on
  the device (each draft step's argmax tensor feeds the next; `StepRequest` accepts device ids), the
  target's per-position argmax is computed on the device, and `[argmax (K+1) ‖ drafts (K)]` comes to
  the host as one `2K+1`-integer copy; penalties / constraints / stochastic runs pull the `K+1` logit
  rows in one copy instead. The pre-engine loop paid `K+1` transfers per verify step.
* **Recovery**: the engine first asks `rollback_to(start + 1 + accepted)`; the S1 `Qwen35Cache`
  answers the typed `RollbackUnavailable`, so it rolls back to the step start and replays the kept
  prefix — the same forwards as the old clone-restore loop, so the acceptance statistics are
  unchanged (the tiny-config test ran both loops side by side before the old one was deleted:
  tokens / proposed / accepted / forwards identical for K = 1..5, greedy and stochastic). S3's
  per-token checkpoints will make the first call succeed with no change here.
* **Telemetry (E2)**: `DecodeRecord.proposer` (`none` / `mtp` / `ngram` / `draft`),
  `verify_steps`, `verify_host_syncs`, `host_syncs_per_verify_step()`; the bench reports
  `proposer` and `syncs/verify` per row.

## The token-107 knife-edge: root cause

`tests/speculative_engine_parity.rs::op_isolation_survey_which_primitive_depends_on_the_row_count`
feeds each primitive, at the 27B decode shapes, one row alone (`M = 1`) and the same row as row 0 of
an `M = 2..6`-row input, and bit-compares row 0 (`parity-survey.log`):

| primitive (27B shape) | row 0 with `M = 2..6` vs alone |
|---|---|
| projection GEMMs — attention q/k/v/o, DeltaNet `in_proj_qkv` / `out_proj`, MLP gate / down, `lm_head` | **differs, every one, every M** (last-bit changes; up to hundreds of bf16 ULP on near-zero elements) |
| `sdpa_gqa_causal` (`q_len = M`, row 0 attends the same keys) at 128 / 205 / 353 keys | bit-identical (one 1-ULP element at 353 keys, `M = 3`) |
| `gated_delta_recurrence` (`T = M`, step 0) | bit-identical (token-sequential by construction) |
| `causal_depthwise_conv` (`S = M`, row 0) | bit-identical |
| `rms_norm` (row 0) | bit-identical |

So the M=1 decode forward and the M=K+1 verify forward differ in exactly one class of op: **every
cuBLAS projection GEMM** picks a different kernel — and so a different fp32 reduction order and last
bf16 bit — for `M >= 2` rows than for a single row (the M=1 case is a GEMV-class kernel). Attention,
the DeltaNet recurrence, the conv and the norms are row-count-invariant. This is the same class of
property S4 documented for expanded-vs-unexpanded GQA GEMMs, now on the projections.

**Why no formulation removes it.** There is no per-row formulation of a verify forward short of
running its projections one row at a time, which streams the 54 GB of weights `K+1` times per step —
the cost speculation exists to avoid. Padding the reference decode to `M = K+1` rows would tie the
parity oracle to one draft width and make each K its own reference. Neither is option (a); the
finding is inherent to cuBLAS kernel selection, so this story takes **option (b)**: the parity gate is
"token-identical except at reference knife-edge positions where the reference's top-2 logit gap is
within one bf16 ULP of its top logit", with the positions enumerated (below), and AC1's literal
wording is recorded as not met (see BLOCKED in the PR).

### The knife-edge positions on the fixture

Positions are the 0-based index of the generated token the logits pick (the bench's `first
divergence` convention). The single-token reference's top-2 logit gap is `<= 1` bf16 ULP of its top
logit (`0.125` at these logit magnitudes) at **15 of 256 positions**: 51, 74, 99, 102, 103, 106 (exact
tie), **107 (exact tie — the token-107 knife-edge of every sealed run since S1)**, 124 (tie), 125, 154,
161, 177, 186, 202, 243 (tie). Teacher-forced, row 0 of an `M = K+1` verify forward against the
single-token forward at the same position (`parity-teacher-forced-row0.log`, whose labels are one
lower — the position of the token fed, not the token picked): the argmax flips only at knife-edge
positions — 51, 74, 154, 243 for `M = 2`; 106, 154 for `M = 3`; 106 for `M = 4`; 106, 125, 243 for
`M = 5`; 243 for `M = 6` — and nowhere else; the max `|Δlogit|` between the two forwards over the
whole fixture is 0.64 (`M = 2, 3`) to 1.39 (`M = 5`), the same last-bit class of change S4 measured.
The all-rows gate (every row of the verify forward, not only row 0, against the single-token argmax
at the position that row picks) is `teacher_forced_verify_shaped_forward_vs_single_token_knife_edge_gate`
(`parity-final.log`): over the 5 065 (position, row) pairs of `M = 2..6` it finds **34 argmax
flips, every one at an enumerated knife-edge** (74, 106, 107, 124, 125, 154, 161, 177, 186), and the
largest reference top-2 gap at any flipped position is **1.0 bf16 ULP**; it passes.

## AC1 / AC2 — engine rows against speculative-off (`parity-final.log`, `decode-bench/`)

Free-running (each row a full greedy decode; `match ref` = token identity with the reference loop,
which the `StepModel` static-KV row reproduces 256/256):

| row | first divergence | reference gap there | acceptance | S1 sealed acceptance (old loop) | fwd/tok | syncs/verify | tok/s |
|---|---|---|---|---|---|---|---|
| MTP K=1 | 124 | 0 (exact bf16 tie) | **0.827** | 0.827 | 0.645 | **1.00** | 14.18 |
| MTP K=2 | 125 | 1 bf16 ULP | 0.668 | 0.779 | 0.613 | **1.00** | 14.60 |
| MTP K=3 | 186 | 1 bf16 ULP | 0.640 | 0.607 | 0.535 | **1.00** | 14.53 |
| MTP K=4 | 124 | 0 (exact bf16 tie) | 0.549 | 0.493 | 0.535 | **1.00** | 14.67 |
| MTP K=5 | 124 | 0 (exact bf16 tie) | 0.480 | 0.480 | 0.547 | **1.00** | 13.66 |
| n-gram K=3 | 186 | 1 bf16 ULP | 0.167 | n/a (could not run on qwen35 before) | 1.145 | **1.00** | 8.86 |

`ac1_ac2_engine_greedy_fixture_rows_against_speculative_off` passes with this gate: 0 of 6 rows are
token-identical over 256 tokens, and every first divergence is an enumerated knife-edge (two exact
ties and two 1-ULP gaps). The literal AC1 wording is not met and cannot be on this hardware (root
cause above).

* **Acceptance vs the pre-engine loop.** K=1 and K=5 reproduce S1's sealed acceptance to three
  decimals (0.827, 0.480) with the same forwards per token (0.645, 0.547); the old loop's rows all
  parted from their reference at the 107 tie, the engine's at the 124 tie — both are exact bf16 ties
  in the enumerated set, and which side a tie falls on is the last bit of a GEMM. K=2..4 diverge from the reference earlier or later than the old
  loop did, and an acceptance rate is a property of the sequence actually decoded, so once the
  sequences part the rates are not the same statistic: the like-for-like comparison is the
  tiny-config side-by-side (identical) and K=1 / K=5 above. Every MTP row's forwards per token is
  `< 1` as before.
* **AC2**: every engine row reports exactly **1.00** host syncs per verify step (the old loop: 1.64
  → 3.21 syncs per generated token at K=1..5 on S1's head, i.e. `K+1` per verify step). Per token the
  engine issues 0.55 → 0.30 syncs at K=1..5.
* **AC3**: `tests/qwen38_mtp.rs::frozen_qwen38_provider_executes_ar_mtp_tools_and_stops` (frozen
  tokenizer, CPU) — the same checkpoint without an MTP head advertises no MTP; `MtpMode::Auto`
  decodes normally with `proposer=none` (`DecodePath::Reference`), `Enabled` is refused
  (`Unsupported`); with the head, the engine's record says `proposer=mtp`, `kv_cache=static`,
  `host_syncs_per_verify_step = 1`.
* **tok/s**: the MTP rows run on the **static KV cache** now (the old loop: growing). K=1..4 sit at
  14.2–14.7 tok/s against 11.2 (reference) / 11.4 (StepModel) in the same run — +27–31 % — where S1's
  head measured 15.0 / 16.7 / 14.9 / 13.3 / 13.4 tok/s at K=1..5 on a different day (this box's
  clocks drift 10–20 % between runs; within-run deltas are the evidence, see `sc-24132/README.md`).
  The n-gram row is slower than the reference on this prose fixture (acceptance 0.167, 1.145
  forwards per token): prompt lookup pays a replay forward on most steps here; it is the row that
  proves the proposer runs on qwen35, not a speed claim.

## Weights-free gates

* Git Bash: `cargo test --locked -p candle-llm --lib` — 288 passed (16 new: the engine's parity vs
  the step driver for MTP K=1..5 / n-gram / draft model / no proposer, one-sync-per-verify accounting
  on the greedy, penalized and constrained paths, direct-rollback vs replay forward counts, chi-square
  of the stochastic decision, stop / caller-stop / cancel contracts, cache consistency after a
  mid-verify cancel, the prefilled and multimodal prompt paths, the rope-delta seam), `cargo test -p
  core-llm` (199 + 4), `cargo test -p candle-llm --test qwen38_mtp -- --ignored` (frozen tokenizer,
  3 passed), CPU clippy `-D warnings`, `cargo fmt --check`, `check-workspace.py`, `check_docs.py`,
  `check_clock_assertions.py --check-baseline`, `pytest scripts/tests/test_decode_bench.py`
  (11 passed).
* PowerShell (MSVC 14.44 vcvars, `CUDA_COMPUTE_CAP=120`): CUDA clippy `-D warnings`, rustdoc
  `-D warnings`, `cargo test --locked --lib --tests -p candle-llm --features cuda`.

## Files

* `decode-bench/head-bd33b65a7/` — the sealed bench run (reference, StepModel, MTP K=1..5, n-gram
  K=3); `decode-bench/comparison.md` — against the pre-epic baseline, S1's and S4's sealed heads.
* `parity-survey.log` — the op-isolation survey and the first (row-0) AC1 pass.
* `parity-teacher-forced-row0.log` — the row-0 teacher-forced gate.
* `parity-final.log` — the AC1 rows with each divergence's reference gap and the all-rows
  teacher-forced gate, at the PR's code sha.
