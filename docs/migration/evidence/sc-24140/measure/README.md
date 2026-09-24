# sc-24140 — feature-end review, round 1 (measurement and the llama family): evidence

Epic sc-24128, feature-end fix story, findings A–E. Host: Windows 11, **RTX Pro 6000 / sm_120**
(GPU 1, `CUDA_VISIBLE_DEVICES=1`), CUDA 12.9, MSVC 14.44, `CUDA_COMPUTE_CAP=120`, release builds
with `--features cuda`. Real-weight model: `Qwen/Qwen3-8B` @ `b968826d9c46`
(`qwen3-8b`, pinned in `release/real-weight-models.toml`, config sha256 `f7c4eadf…` pinned in
`scripts/release/decode_bench.py`). Measured code: `6867f52d9` (this branch merged with the
feature head carrying S11 #1035 and the round-1 decode fixes #1036), clean tree. The code after it
adds only the per-thread mask accounting in tests (`attention.rs`) and boxes the `CausalLm` LM
head with its device (CUDA clippy `large_enum_variant`) — no arithmetic change.

## The E1 rule for speculative rows

Decided at this feature-end review (finding D); stated once here, applied on both families in
`crates/llm/candle-llm/tests/speculative_engine_parity.rs`:

* **Exact rows are gated token-identical**: every path with no verify forward — the reference
  loop, the static step seam, the fused-off loops and the CUDA-graph fallback.
* **The E1 gate for speculative rows is the teacher-forced verify-shaped forward**: every row of
  an `M = 2..6` verify forward over the reference fixture, from a cache the single-token path wrote,
  may disagree with the single-token argmax only at a reference position whose top-2 logit gap is
  `<= 1` bf16 ULP of its top logit (the S2 mechanism: cuBLAS rounds an `M`-row GEMM differently
  from `M = 1`).
* **Free-running speculative rows are recorded, not gated**: first divergence and the reference's
  top-2 gap in ULP. A free-running row compounds the same rounding — its later positions attend
  K/V that earlier verify forwards wrote — so any fixed ULP bound on it would be arbitrary.

The case that decided it: Qwen3-8B, n-gram K=3, free-running, first diverges at **@51 with a
2.00-ULP** reference gap, while the teacher-forced gate **passes** (no single verify forward flips
@51; every teacher-forced flip is at ≤ 1 ULP). Qwen3.8-27B keeps its gate in
`teacher_forced_verify_shaped_forward_vs_single_token_knife_edge_gate`; its AC1 free-running rows
are now recorded the same way.

## A — NVFP4 reaches the llama family (`CausalLm`)

`CausalLm::from_weights_format(w, prefix, cfg, Option<&ProjectionFormat>)` is the one loader for
every format; the Q4/Q8 entry points (`from_weights_with`, `from_weights_dtype`,
`from_weights_on_device`) wrap it. Under NVFP4 every attention (q/k/v/o, Phi-3's split `qkv`,
MLA's low-rank projections), MLP / MoE-expert projection **and the LM head** load through
`Projection::load_eligible` in `primitives/projection.rs` — `load_as` plus the llama-family shape
policy: a shape the FP4 GEMM cannot serve (`nvfp4_shape_refusal`: `N % 16 != 0`, or past the
quantizer's 32-bit indexing) stays dense and is counted under `dense`, never under `nvfp4`. No
quantization logic lives in `models/llama.rs` beyond calling the shared loader (E0). An MLX affine
triple under NVFP4 is the typed `Unsupported("nvfp4: …")`. The decode GEMV (sc-24136) serves the
≤ 8-row forwards automatically. `CausalLm::weight_census()` reports the census, and the provider's
`LoadRecord` carries it for every llama-family load (safetensors and GGUF).

The family rule lives in one function, `provider::nvfp4_family_refusal`, called by
`nvfp4_model_gate` — the gate the load runs **and** S11's per-snapshot `nvfp4_support` probe calls,
so the two cannot disagree. Served: the qwen3_5 hybrid and every `CausalLm` architecture (Llama /
Mistral, Qwen3 dense, Phi-3, Qwen2-MoE, Gemma 2/4, GLM-4, DeepSeek-V2, the Qwen3-VL decoder).
Refused by name: GGUF (already block-quantized), Prism/Bonsai (packed affine-2); LLaVA and both
StarVectors keep refusing in their own providers' gates. The probe's tests
(`backend.rs`, `runtime-cpu`) now assert that a Llama and a Qwen3 snapshot reach the device gate.

**Qwen3-8B, bf16 vs NVFP4** — `nvfp4-qwen3-8b/nvfp4_evidence.json` (`tests/nvfp4_evidence.rs`,
which now dispatches the family from `config.json` like the provider; same fixture prompt, same
perplexity text as S7 — `docs/migration/evidence/sc-24135/ppl_slice_source.txt`, sha256
`f238f90b…` — first 2048 tokens of **this model's** plain encoding):

| | bf16 | NVFP4 |
|---|---|---|
| projections (count / params) | 253 dense / 7.568 G | **253 NVFP4** / 7.568 G (36 layers × 7 + the head; 0 dense) |
| projection bits/param | 16.0 | **4.50** |
| resident weights, total | 16.38 GB (16.0 bits/param) | **5.50 GB (5.37 bits/param)** — embeddings and norms stay bf16 |
| perplexity, 2048 tokens (mean NLL) | **26.038** (3.2595) | **27.971** (3.3312) — +7.4 % (S7 on Qwen3.8-27B: +7.4 %, 7.84 → 8.41) |
| 256-token greedy fixture | reference | coherent; **first divergence at token 43** |
| provider `Quantize::Nvfp4` load | — | `LoadRecord` census = the direct load's (253 NVFP4, 5.37 bits/param) |

The NVFP4 text is the same answer re-worded from token 43 on ("…greedy vs. sampled…"); both
fixtures are in the JSON.

## B — the one measurement home expresses the AT2 matrix

`tests/decode_bench.rs` + `scripts/release/decode_bench.py`:

* **Formats**: `DECODE_BENCH_FORMAT` / `--format` ∈ `bf16 | q8 | q4 | nvfp4` for both families
  (`q8`/`q4` are the GGML load-time quantization of `Quantize::Q8/Q4`); the document records
  `weight_format` and the weight census; `run` refuses a document whose `weight_format` differs from
  the request, and `--cuda-graphs on|off` forces `CANDLE_LLM_CUDA_GRAPHS` and requires the document
  to record the same switch.
* **Weight format is a row dimension**: tables merge runs of different formats, with `format` and
  `graphs` columns; each row compares with the baseline of its own format — and a non-bf16 row with
  no same-format baseline with its head's own bf16 reference row, labelled `(vs <run> bf16 ref)`.
* **The pre-epic Qwen3-8B baseline**: `BASELINE_STUB` carries the pre-epic `CausalLm`
  `decode_logits` + `generate_from_prefill` reference (bf16 / q8 / q4). The rewrite now compiles
  at `d2b8cb335` — it did **not** before this story: the shared body called
  `mtp.set_attn_formulation`, which the baseline lacks (moved into the head-only block; the Python
  test now scans everything the baseline compiles for head-only calls). Sealed run:
  `decode-bench/baseline-d2b8cb335-qwen3-8b/` (reference 24.62 tok/s, sampled 22.58 tok/s, 256
  tokens).
* **`campaign`** runs — or collects already-sealed runs of — {models} × {bf16, q8, nvfp4} ×
  {speculative off, MTP K=1..5, n-gram} × {graphs off, on} (+ the stochastic rows), refuses a dirty
  checkout / an output inside it / an unpinned snapshot before anything runs, refuses collected
  runs that are dirty, from two commits, outside the matrix, or cells that are missing (unless
  `--allow-partial`), records MTP on the llama family as `n/a (no MTP head)`, copies every run and
  baseline into the campaign directory, and seals `INDEX.md` + `index.json` (one table per model,
  baselines first). `campaign-verify` re-checks the campaign seal and every run's seal.
  Since round 2 of the feature-end review ([`../round2/README.md`](../round2/README.md)), every
  model also needs a bf16 S1 baseline, or the campaign lists it as missing. `campaign-verify` fails
  a campaign that has missing cells unless it was sealed with `--allow-partial`. A head run's binary
  must be built with `CANDLE_LLM_BUILD_PROVENANCE=1`, so that it embeds the runtime SHA and a clean
  tree. The label must match the probed device.

**Smoke** (not the campaign): `decode-bench/campaign-smoke/` — Qwen3-8B × {bf16, q8, nvfp4} ×
{graphs off, on} × {off, MTP (n/a), n-gram K=3} + sampled rows, 16 tokens, with the 16-token
pre-epic baseline (`baselines/`). 6 cells, all `ok`; `campaign-verify` passes on the committed
copy. Selected rows (merged head `6867f52d9`):

| format, graphs | reference / StepModel / n-gram K=3 / sampled StepModel (tok/s, 16 tokens) | vs baseline | nvfp4 path (StepModel) | sampler |
|---|---|---|---|---|
| bf16 (pre-epic baseline `d2b8cb335`) | 32.41 / — / — / — | (ref) | n/a | n/a |
| bf16, off | 74.53 / 77.98 / 97.37 / 78.95 | yes | none | device, 0 rows→host |
| bf16, on | 73.49 / 80.81 / 103.01 / 77.70 | yes | none | device |
| q8, off | 41.16 / 44.99 / 61.73 / 43.68 | yes (vs head bf16 ref) | none | device |
| q8, on | 71.93 / 74.87 / 96.75 / 71.91 | yes (vs head bf16 ref) | none | device |
| nvfp4, off | 115.57 / 124.19 / 152.69 / 111.93 | yes (vs head bf16 ref) | 3796 GEMV / 252 cuBLASLt (`rows` = prefill) | device |
| nvfp4, on | 114.93 / 127.53 / 152.49 / 111.94 | yes (vs head bf16 ref) | same | device |

Graphs on: every step-seam row runs eager through the runner, fallback `positions_host_scalar`
(a `CausalLm` step is not replayable). Sixteen tokens is a smoke, not a measurement: the q8
graphs-off / graphs-on spread is warm-up noise at this length.

`decode-bench/campaign-smoke-e155acb09/` is the same smoke before the #1036 merge: its
`sampled (StepModel)` rows read `1 device / 0 host, 0.94 logits rows→host/tok` (the engine's K = 0
host draw); at the merged head they read `16 device, 0.00` — the round-1 decode fix, visible in
the one measurement home. The full campaign was **not** run (terminal story).

## C — stochastic rows on both families

`sampled` (the reference loop) and `sampled_step_model` (the step seam) under the seeded
`DECODE_BENCH_SAMPLING` = `temperature,top_p,seed` (default `0.7,0.9,0`), on the hybrid and the
llama family; a `sampled_step_model` row compares with its run's `sampled` row. **Every** row
records `sampler`: path (`device` / `host:<reason>` / `none`), device and host draws, whole logits
rows copied to the host, and both per generated token (`RequestSpan` bracketing). On real weights:
the smoke above (both sampled rows on every cell).

## D — the llama-family knife-edge gate (Qwen3-8B, 256 tokens)

`single_token_gaps` / `knife_edges` are generic over `StepModel`; the free-running record is
`free_running_record`. One test, `llama_family_qwen3_8b_exact_rows_and_teacher_forced_knife_edge_gate`,
applies the E1 rule above. Log: `knife-edge-qwen3-8b.log` (commit `8ea9d21e0`, clean tree, GPU 1):
**passes**.

| row | vs the reference loop (growing, `gqa`) |
|---|---|
| static step seam (`generate_step`, now the engine with no proposer) | **identical 256/256** |
| reference loop, fused primitives off | **identical** |
| static step seam, fused off | **identical** |
| CUDA-graph runner, switch on (own stream) | **identical**; 0 replayed / 256 eager, fallback `positions_host_scalar` |
| teacher-forced verify-shaped forwards, M = 2..6, every row (**the E1 gate**) | **holds**: 30 disagreements, every one at 65 (0 ULP), 90 / 134 / 141 (1.0 ULP) or 132 (0.5 ULP) |
| free-running n-gram K=2 (recorded) | first divergence @65 — top-2 gap **0.00** bf16 ULP (a knife-edge) |
| free-running n-gram K=4 (recorded) | @132 — **0.50** ULP (a knife-edge) |
| free-running n-gram K=3 (recorded) | @51 — **2.00** ULP (not a knife-edge) |

Reference knife-edges (gap ≤ 1 ULP): 17, 65, 90, 132, 134, 141, 148, 193, 228. The S10 README's
statement that @51 is a knife-edge is **not true** under the decided rule (it was never
enumerated).

Position 51 never flips in a single verify forward: the free-running K=3 row reaches a 2-ULP
flip through compounding — its later positions attend K/V that earlier verify forwards (M = 4
GEMMs) wrote. Under the E1 rule that row is recorded, not gated.

## E — `REQUIRE_SM120=1`

`candle_quant_kernels::sm120_gate`: `skip_without_sm120(reason)` prints `skipping: …` or, under
`REQUIRE_SM120` (set, non-empty, not `0`), panics. Routed through it: `tests/nvfp4_gemv.rs` (3),
`qwen35.rs`'s NVFP4 loader test, `projection.rs` (3), the new llama NVFP4 test, and every sm_120
skip in `candle-quant-kernels` (`nvfp4_gemv.rs` 5, `nvfp4_weight.rs` 4). `require-sm120.log`:
with every GPU hidden (`CUDA_VISIBLE_DEVICES=-1`) the tests skip and pass; with `REQUIRE_SM120=1`
they fail (3 of 3 and 9 of 9) naming the reason; on GPU 1 with `REQUIRE_SM120=1` they run and pass.

## Mutations

`mutations.log` — 55 mutations, one per added or changed assertion, each applied alone to a
touched source, the named gate run, and the source restored: **55 RED** (P19 reds through the
campaign's own coverage refusal, a `ValueError`, rather than the test's assertion). Python
harness (P1–P21), Rust CPU (R1–R16, S1–S2 — the S11 probe after the merge —, T1), Rust CUDA on
GPU 1 (C1–C5), real weights on GPU 1 (G1: the reference loop computing `Expanded` fails the exact
rows at 65; G2: counting every disagreement fails the teacher-forced gate). R1b / R2b / C1b / C2b
re-run the head mutations on the final (boxed-head) code. After the E1 decision (G3, G4, R17,
R18 on the combined Qwen3-8B test and the free-running record): counting every teacher-forced
disagreement fails the E1 gate (G3, GPU 1); the reference computing `Expanded` fails the exact
rows (G4, GPU 1); the record reporting the raw gap (R17) or marking every divergence a knife-edge
(R18) fails its unit test. G1, G2 and R11 were run against the pre-decision tests
(`…_exact_rows_and_ngram_knife_edge_gate`, `…_teacher_forced_verify_shaped_knife_edge_gate`,
`off_edge_divergences`), which the combined test and `free_running_record` replace.
