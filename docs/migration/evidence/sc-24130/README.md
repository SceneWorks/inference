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
  tokens / proposed / accepted / forwards identical for K = 1..5, greedy and stochastic; the
  real-weight identical-prefix comparison is below). S3's per-token checkpoints will make the
  first call succeed with no change here.
* **Telemetry (E2)**: `DecodeRecord.proposer` (`none` / `mtp` / `ngram` / `draft`),
  `verify_steps`, `verify_host_syncs`, `host_syncs_per_verify_step()`, and
  `replay_forwards` (`SpeculativeStats.replays`: the `RollbackUnavailable` → replay fallbacks,
  one per rejected verify step on the S1 cache, `0` on a cache with per-position rollback — the
  engine test pins both); the bench reports `proposer` (from the record, not the row name),
  `syncs/verify` and `replay_forwards` per row.
* **The stream contract (E7)**: the engine checks the cancel flag right after the verify forward,
  as the old loop did, and rolls the cache back to the step start before returning `Cancelled`;
  every early exit of the commit loop (stop token, caller stop, cancel, budget) settles the cache
  so it never holds a position the committed history does not. Device-resident greedy drafts
  cannot stop at a stop token while drafting, so the verify decision truncates them at the first
  stop token: nothing past an accepted stop token is counted as proposed or accepted.

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

* **Acceptance vs the pre-engine loop, on the identical prefix** (`decode-bench/acceptance-prefix/`).
  Over 256 tokens the 256-token rows above and the old loop's sealed rows decode *different*
  sequences past their first tie (the old loop's rows parted from their reference at the 107 tie,
  the engine's at 124 / 125 / 186), and an acceptance rate is a property of the sequence actually
  decoded — so the 256-token rates (K=2: 0.779 old vs 0.668 engine; K=4: 0.493 vs 0.549) are not
  the same statistic. The like-for-like comparison is the two loops on the prefix they share: the
  old `qwen_mtp` loop (built from `87478b336`, the commit before the engine landed, run through
  its own `decode_bench`) and the engine at this PR's code sha (`51621c288`), the same fixture,
  **124 new tokens** (the first old-vs-new divergence, the 124 tie), K=1..5, GPU 0 with no
  co-tenant. Every row of both runs is token-identical to the reference over the 124 tokens, and
  **proposed / accepted / target forwards match exactly for every K**
  (`acceptance-prefix/comparison.md`, the two sealed runs beside it):

  | K | proposed old / new | accepted old / new | target forwards old / new | acceptance | replay forwards (engine) |
  |---|---|---|---|---|---|
  | 1 | 65 / 65 | 57 / 57 | 75 / 75 | 0.877 | 8 |
  | 2 | 91 / 91 | 77 / 77 | 57 / 57 | 0.846 | 10 |
  | 3 | 117 / 117 | 83 / 83 | 58 / 58 | 0.709 | 17 |
  | 4 | 132 / 132 | 90 / 90 | 53 / 53 | 0.682 | 19 |
  | 5 | 154 / 154 | 92 / 92 | 54 / 54 | 0.597 | 22 |

  No draft-side knife-edge had to be invoked: the MTP head proposed the same drafts and the target
  accepted the same runs on the growing (old) and static (engine) caches, and the engine's replay
  forwards are exactly the old loop's clone-restore replays (`target_forwards` equal). The
  256-token rates differ only because the sequences differ after the tie. Every MTP row's forwards
  per token is `< 1` as before.
* **AC2**: every engine row reports exactly **1.00** host syncs per verify step (the old loop: 1.64
  → 3.21 syncs per generated token at K=1..5 on S1's head, i.e. `K+1` per verify step). Per token the
  engine issues 0.55 → 0.30 syncs at K=1..5. The 1.00 is the **plain-greedy** figure (the bench
  rows and every provider request without penalties, a constraint or sampling): with a repetition
  penalty or a constraint the drafts are sampled on the host, one whole-vocab transfer each, and
  the verify decision pulls the `K+1` rows in one copy — `K+1` syncs per verify step, the old
  loop's figure on every path; a stochastic run adds a shaped-distribution copy per draft,
  `2K+1`. Both are pinned by the tiny-config tests
  (`penalized_and_constrained_verify_steps_cost_one_sync_per_draft_plus_one`,
  `stochastic_runs_are_seed_deterministic_and_bounded`).
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

* Git Bash: `cargo test --locked -p candle-llm --lib` — 313 passed after the merge of S5 + S8 (the engine's parity vs the
  step driver for MTP K=1..5 / n-gram / draft model / no proposer, one sync per verify step on the
  plain-greedy path and `K+1` on the penalized / constrained paths, direct-rollback vs replay
  forward counts with the replay counter pinned (0 / 8), chi-square of the stochastic decision over
  a point-mass and the real draft `q`, stop / caller-stop / cancel contracts, cache consistency
  after a mid-verify cancel and after a cancel fired inside the verify forward, the budget exit
  settling an over-proposed run, device-greedy drafts truncated at a stop token, the prefilled and
  multimodal prompt paths, the rope-delta seam; in `provider.rs`: an `Enabled{3}` request priced on
  the step geometry — checkpoints plus the K-position overshoot — and AC3 on a synthetic Qwen3.5
  snapshot without an MTP head: `Auto` decodes normally with `proposer=none`, `Enabled` is
  refused), `cargo test -p core-llm` (199 + 4), `cargo test -p candle-llm --test qwen38_mtp --
  --ignored` (frozen tokenizer, 3 passed), CPU clippy `-D warnings`, `cargo fmt --check`,
  `check-workspace.py`, `check_docs.py`, `pytest scripts/tests/test_decode_bench.py` (11 passed).
* PowerShell (MSVC 14.44 vcvars, `CUDA_COMPUTE_CAP=120`): CUDA clippy `-D warnings`, rustdoc
  `-D warnings`, `cargo test --locked --lib --tests -p candle-llm --features cuda`.

## Files

* `decode-bench/head-bd33b65a7/` — the sealed bench run (reference, StepModel, MTP K=1..5, n-gram
  K=3); `decode-bench/comparison.md` — against the pre-epic baseline, S1's and S4's sealed heads.
* `decode-bench/acceptance-prefix/` — the identical-prefix comparison: `old-loop-87478b336/` (the
  pre-engine loop's own bench, 124 tokens, K=1..5), `head-51621c288/` (the engine at this PR's
  code sha, same run), `comparison.md` (the table above, generated from the two sealed JSONs).
* `parity-survey.log` — the op-isolation survey and the first (row-0) AC1 pass.
* `parity-teacher-forced-row0.log` — the row-0 teacher-forced gate.
* `parity-final.log` — the AC1 rows with each divergence's reference gap and the all-rows
  teacher-forced gate: both non-survey parity tests in **one** invocation at the PR's code sha
  (`51621c288`, release, `--features cuda`), unfiltered `test … ok` / `test result` lines
  included; the log's paths are relative. (The earlier version of this file was grep-filtered from
  two separate invocations and embedded absolute scratchpad paths; the first invocation's
  `1 passed; 1 failed` line named neither the failing test nor its assertion, because the filter
  dropped them. This re-run — both tests, one invocation, at the code sha — replaces it: `2 passed;
  0 failed`, the 3 filtered out being the survey and the two `common::fixture_*` tests the
  binary also carries.)
