# sc-24132 — Static (preallocated, in-place, zero-copy GQA) KV cache: evidence

Story S4 of epic sc-24128. Host: Windows 11, **RTX Pro 6000 / sm_120** (GPU 0), CUDA 12.9, MSVC 14.44,
`CUDA_COMPUTE_CAP=120`, release build with `--features cuda`. Model: `Qwen/Qwen3.8-27B` @
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (`bonsai-qwen38-parent`), BF16 greedy, 97 prompt tokens,
256 new tokens per row — the same fixture and harness as `docs/migration/evidence/sc-24129/`.

## The decision taken (review of PR #1024)

Bit identity between the un-expanded GQA matmul and the pre-S4 `repeat_kv`-expanded GEMM is a **cuBLAS
kernel-selection property** — the kernel, and with it the fp32 reduction order that decides the last bf16
bit, is chosen by the GEMM's `m`, batch count and strides (survey below) — and is not attainable without
re-materializing the expansion AC2 exists to remove. The coordinator's decision, implemented at
`23905fe60`:

1. **The reference oracle attends through `sdpa_gqa_causal` too.** The growing `AttnKv` slot — the reference
   `Decode` loop (`generate_with`, the provider), the `StepModel` driver with the growing cache selected,
   the MTP head layers and the MTP target verify — runs the same un-expanded arithmetic as the static
   cache, so static-vs-reference parity holds **by construction**.
2. **The pre-S4 expanded formulation stays selectable only as a labelled comparison row** against the
   sealed pre-epic / S1 baseline: `AttnFormulation::Expanded` (`Qwen35Model::set_attn_formulation`;
   `DECODE_BENCH_ATTN=expanded` in the bench). The static cache always attends un-expanded, whatever the
   selector says, and its record says so.
3. **The reference numerics changed at S4 by at most one bf16 ULP at attention-GEMM knife-edges** — the
   token-107 example and the survey table below. The reference row of a head run at the fix sha therefore
   reads `no @107` against the pre-epic baseline's reference row, and the `expanded` row reads `yes`.
4. **AC1 is the real-weight test `ac1_static_kv_greedy_fixture_is_token_identical_to_attn_kv`**, which
   passes **256 / 256** against the new reference (below). The teacher-forced knife-edge report
   (`teacher_forced_static_vs_attn_kv_logit_parity_report`) is a **diagnostic only**: it selects the
   `expanded` arithmetic explicitly and documents the knife-edge; it is not cited as the AC1 gate.

Every decode record and every bench row now names both the KV cache (`kv_cache`: `growing` / `static`) and
the attention formulation (`attn_formulation`: `gqa` / `expanded`) that produced it — what actually ran, not
what was configured.

## Decode bench (sealed, `decode-bench/`)

Sealed head runs of `tests/decode_bench.rs` (MTP-off rows) through `scripts/release/decode_bench.py`,
tabulated in `decode-bench/comparison.md` against the pre-epic baseline (`sc-24129/decode-bench/baseline-d2b8cb335`,
the table's `match baseline ref` column) and S1's sealed head (`sc-24129/decode-bench/head-4a059a87b`); re-verify
the seals with `python scripts/release/decode_bench.py table <the run dirs>`. Every run here had **no
co-tenant on GPU 0** (`gpu.co_tenants_at_start` in each `run.json`) and GPU 1 idle (< 1 GB, 0 %).

| run | row | match ref | match pre-epic ref | tok/s | cache live | cache checkpoints |
|---|---|---|---|---|---|---|
| pre-epic baseline-d2b8cb335 | MTP off (reference) | (ref) | (ref) | 11.57 | n/a | n/a |
| S1 head-4a059a87b | MTP off (reference) | (ref) | yes | 12.17 | n/a | n/a |
| S1 head-4a059a87b | MTP off (StepModel, growing `AttnKv`) | yes | yes | 11.85 | 168.8 MiB | 293.6 MiB |
| **head-23905fe60** (the fix) | MTP off (reference, growing kv, **gqa** attn) | (ref) | no @107 | 11.27 | n/a | n/a |
| **head-23905fe60** (the fix) | **MTP off (StepModel, static kv, gqa attn)** | **yes** | no @107 | **11.38** | 168.9 MiB | 293.6 MiB |
| head-978307d88-expanded-attn (comparison) | MTP off (reference, growing kv, **expanded** attn) | (ref) | **yes** | 9.29 | n/a | n/a |
| head-978307d88-expanded-attn (comparison) | MTP off (StepModel, static kv, gqa attn) | no @107 | no @107 | 9.99 | 168.9 MiB | 293.6 MiB |

`head-23905fe60` and `head-978307d88-expanded-attn` are the same binary (`decode_bench-a589f4c981fe54a8.exe`,
sha256 `ab4dc2e6…` recorded in each `run.json`, built at the code sha `23905fe60`); the second sha only adds
the first run's evidence files. The two earlier sealed runs, `head-e7805eee4` (static row `no @107` against
its expanded reference) and `head-adce2370e-growing-kv` (the growing control), predate the decision and stay
sealed as the record of the finding that led to it; `comparison.md` lists them too.

### What the rows say

* **Static KV is token-identical to the reference** (`match ref` = yes at the fix sha) and decodes at
  **11.38 vs 11.27 tok/s** (+1.0 %) — inside the ±0.3–0.7 tok/s run-to-run noise S1 recorded. Within a run
  is the only valid comparison on this box (below).
* **The `expanded` comparison row reproduces the pre-epic baseline's tokens** (`match pre-epic ref` = yes):
  the pre-S4 arithmetic is still there, selectable, and still bit-exact with the sealed baseline — which is
  what shows the S4 change to the reference is exactly the formulation and nothing else. Against that
  expanded reference the static row diverges at 107, the knife-edge below, and nowhere earlier.
* **Across runs the absolute numbers move by ~10–20 %** (reference 12.17 → 11.27 → 9.29 across three idle-box
  runs) — the two Max-Q cards share the workstation's power/thermal budget and the clocks drift between runs;
  S1 recorded the same. Within-run deltas are the evidence; absolute tok/s across runs are not.
* A first head run at `40a874ef8` (an intermediate sha of the same fix) was **discarded, never sealed**: its
  `run.json` recorded a sibling story's CUDA test binary as a co-tenant on GPU 0 and its reference row read
  3.55 tok/s. The runs above were taken behind a gate that requires zero compute processes on GPU 0.
* **Why attention is invisible in tok/s here:** a Qwen3.8-27B bf16 decode step streams ~54 GB of weights and
  launches several hundred kernels across 64 layers; the 16 full-attention layers' KV traffic at L ≤ 353 is
  ~12 MB and a handful of launches per step. Removing the per-step `cat`, `repeat_kv` and transpose copies is
  well below this bench's noise floor. What the static cache buys is the property the next stories need —
  stable device addresses and no per-step allocation (S6 CUDA graphs) — plus a memory profile that is the
  request's bound from the first step (`cache live` 168.9 MiB = 22.1 MiB of preallocated KV + 146.9 MiB of
  live DeltaNet state; the growing row's 168.8 MiB is its *final* size).

## Real-weight acceptance (`tests/static_kv_parity.rs`, `--ignored`, GPU 0, binary built at `23905fe60`)

| test | result |
|---|---|
| `ac1_static_kv_greedy_fixture_is_token_identical_to_attn_kv` (**AC1**) | **ok — 256 / 256** tokens identical across the reference loop / the static cache / the growing cache through the step driver; 0 KV materializations over the static run; records `kv_cache: Static, attn_formulation: Gqa` and `kv_cache: Growing, attn_formulation: Gqa`. |
| `ac3_static_kv_device_pointers_are_stable_across_the_fixture_and_a_rollback` (**AC3**) | **ok** — 16 attention buffers' CUDA device pointers unchanged across the 256-token greedy run (checked every 32 steps) and a `rollback_to(349)` + re-decode; 23 134 208 bytes preallocated (`Qwen35Model::static_kv_bytes` exact). |
| `teacher_forced_static_vs_attn_kv_logit_parity_report` (**diagnostic, not a gate**) | ok — static (gqa) vs the `expanded` reference, both fed the reference's 256 tokens: argmax agrees at **255 / 256**; the one disagreement is position 107, where the expanded reference's own top-2 gap is **0.125 = 1 bf16 ULP** of its top logit and the gqa path lands on an exact tie; max \|Δlogit\| there 0.125; over the fixture mean per-position max \|Δlogit\| 0.254, max 2.70 (position 56, on a row of range 41.1). |

**Mutation of the AC1 gate on real weights:** with the reference loop forced onto the `expanded`
arithmetic (one line in the test), AC1 fails at token 107 — the gate distinguishes the two formulations on
the real fixture (see the PR's mutation list).

### The reference numerics changed at S4 — by ≤ 1 bf16 ULP at attention-GEMM knife-edges

The pre-S4 reference attended over `repeat_kv`-expanded heads: `batch = b·H = 24`, `m = 1`. The S4
reference (and the static cache) attends over the un-expanded K/V with the query groups folded onto the
sequence axis: `batch = b·Hkv = 4`, `m = groups = 6`, strided K/V views. cuBLAS picks its kernel — and so
the fp32 reduction order that decides the last bf16 bit — by `m`, batch count and strides, so the two
disagree in the last bit of a few attention outputs at some key lengths.
`primitives::attention::tests::gqa_variants_bit_match_survey` (CUDA, `#[ignore]`d, kept as evidence) compared
six un-expanded formulations to the expanded reference at the 27B decode shape over 131 key lengths
(90..=1000 step 7):

| formulation | key lengths with a bf16-bit mismatch (of 131) |
|---|---|
| V0 folded groups, strided K (`OP_T`), strided V — **the shipped `sdpa_gqa_causal`** | 52 |
| V1 folded, contiguous Kᵀ (`OP_N`), strided V | 52 |
| V2 folded, contiguous Kᵀ and V | 52 |
| V3 per-KV-head, stride-0 broadcast (m = 1, batch = groups) | 32 |
| V4 V3 + contiguous Kᵀ | 28 |
| V5 V4 + contiguous V | 28 |

Every mismatch is a single bf16 ULP in a few elements. Only the expanded reference's own calls reproduce its
bits, i.e. only the expansion. **The token-107 example:** on the 256-token fixture the expanded reference's
top-2 logit gap at position 107 is 0.125 — exactly one bf16 ULP of its top logit (8 significant bits) —
and the un-expanded arithmetic lands on an exact tie there, so greedy decoding picks the other token
(S1's sealed MTP rows diverge from the reference at the same position for the same reason: their verify
GEMMs have `m > 1`). Everywhere else on the fixture the two formulations agree on the argmax. That is the
whole extent of the change to the reference numerics: a different last bit at a knife-edge, never a
different winner where the reference had a clear one.

## Other review items resolved in the same change

* `sdpa_gqa_causal` **no longer tries the `flash-attn` kernel** (the option the review offered): un-expanded,
  narrowed K/V through `try_flash_attn` is a GQA + stride combination no test covers, and `flash-attn` is not
  part of this repository's CUDA lane; the fused kernel stays out of this path until a flash-vs-eager parity
  test on those views exists. `sdpa` (the expanded path) is unchanged.
* `StaticKvCache` is no longer `Clone` (the deep device copy could only `expect`): `try_clone() -> Result`,
  threaded through `Qwen35LayerCache::try_clone` / `Qwen35Cache::try_clone`; the MTP loop's per-step
  trial-cache copy propagates a failed device copy instead of panicking (`kv_cache::static_tests::bytes_are_the_full_preallocation_and_try_clone_is_deep`).
* `StepModel::new_cache_for(capacity, overshoot)`: callers declare the speculative overshoot (a verify step
  writes `K + 1` positions) and the static cache is sized for `capacity + overshoot`; doc test on the trait,
  `provider::tests::qwen35_admission_covers_the_static_kv_preallocation` checks the sizing and pricing.
* `Qwen35Model::new_static_cache(0, ..)` is `Error::Msg`; `Error::KvCapacityExceeded` is kept for a capacity
  past `max_position_embeddings` (`static_kv_capacity_is_bounded_and_fails_closed`).
* Selector proof on the tiny config (`static_kv_decode_steps_materialize_nothing_and_hold_memory_flat`): the
  growing gqa path records **1** KV materialization per attention layer per step (the `cat` pair), the
  `expanded` selection **3** (plus two `repeat_kv`), the static cache **0** under either selection;
  `attn_formulation_selector_switches_the_growing_arithmetic_and_is_recorded` shows the two formulations
  agree to f32 reduction order and that the record names what ran (a static cache reports `gqa` even with
  `expanded` selected).

## Weights-free gates (at `23905fe60`)

* Git Bash: `cargo test --locked -p candle-llm --lib` — 272 passed (12 new since the base), 9 ignored; the
  `new_cache_for` doc test; CPU clippy `-D warnings`, `cargo fmt --check`, rustdoc `-D warnings`,
  `check-workspace.py`, `check_docs.py`, `check_clock_assertions.py --check-baseline`,
  `pytest scripts/tests/test_decode_bench.py` (11 passed).
* PowerShell (MSVC 14.44 vcvars, `CUDA_COMPUTE_CAP=120`): `cargo test --locked --lib --tests -p candle-llm
  --features cuda` all green (275 lib tests, every test binary), including
  `static_tests::cuda_device_pointers_are_stable_across_100_steps_and_rollback` and
  `gqa_causal_matches_repeat_kv_sdpa_on_cuda_bf16_27b_shape`; CUDA clippy `-D warnings`.
