# sc-24138 — The llama family, StarCoder2, Gemma 4, LLaVA and StarVector on the step seams: evidence

Story S10 of epic sc-24128. Host: Windows 11, **RTX Pro 6000 / sm_120** (GPU 1 — the lane shared
with S3), CUDA 12.9, MSVC 14.44, `CUDA_COMPUTE_CAP=120`, release build with `--features cuda`.
Real-weight model: `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218` (`qwen3-8b`, pinned in
`release/real-weight-models.toml`; config sha256 `f7c4eadf…` pinned in `scripts/release/decode_bench.py`),
BF16 greedy, 256 new tokens per row.

## What landed

* **One shared step-seam cache** — `primitives/step_kv_cache.rs::StepKvCache`, the `DecodeCache`
  every softmax decoder hands `StepModel`: a **static** backing (one preallocated `StaticKvCache` per
  layer — per-layer shapes, so Gemma 4's two layer types and DeepSeek-V2 MLA's distinct key/value
  widths fit; `StaticKvCache::with_value_dim` is the small primitive extension that allows the latter),
  a **growing** backing (the reference `ContiguousKvCache` concat) and a **paged** backing
  (`PagedKvCache`, kept behind the seam rather than replaced: the continuous-batching and
  prefix-sharing paths are built on its shared block pool and copy-on-write prefix blocks, which a
  per-request preallocation cannot express). It carries the M-RoPE position delta too.
* **`CausalLm` implements `StepModel`** — the whole llama family (Llama, Qwen3 dense — the Qwen3-8B
  block —, Gemma 2/4, GLM-4, DeepSeek-V2 MLA, the Qwen3-VL decoder). Static KV by default, attending
  the un-expanded K/V through `sdpa_gqa_causal` wherever a layer can (plain causal, no soft-cap, no
  sliding window). The `attn_formulation` selector defaults to **`Gqa`** on the reference loop and
  the growing / paged backings too, so the reference and the static path are one arithmetic and
  token-identical by construction (256 / 256 on Qwen3-8B, below); **`Expanded`** — the
  pre-migration `repeat_kv` + `sdpa` arithmetic — is the explicitly selected comparison and
  reproduces the pre-migration tree bit for bit (the AC1 goldens). The llama layers now call the S9
  fused entry points where the math matches (`rms_norm_residual`, `swiglu`, `rms_norm_rope`);
  Gemma's sandwich norms and GeGLU keep their op chains.
* **The reference numerics changed — by ≤ 1 bf16 ULP at attention-GEMM knife-edges**, exactly as
  S4 changed the Qwen3.5 hybrid's (`docs/migration/evidence/sc-24132/README.md`, "The reference
  numerics changed at S4"): the default reference attends the un-expanded K/V with the query groups
  folded onto the sequence axis instead of `repeat_kv`-expanded heads, cuBLAS picks its kernel (and
  so the fp32 reduction order that decides the last bf16 bit) by `m`, batch count and strides, and
  the two formulations disagree in the last bit of a few attention outputs at some key lengths. On
  the Qwen3-8B fixture the expanded and the default reference loop agree on the first 65 tokens and
  first differ at token index 65 (0-based, as every `@n` below). On CPU f32 the tiny-config
  goldens move by ≤ 6.7e-6 per logit, tokens unchanged (AC1).
* **The provider decodes the llama family through the engine** (`provider.rs`): a text request
  (Qwen3-8B included) and the Gemma 4 soft-token splice run `generate_speculative_with` over
  `StepModel` on the **static** KV cache. The family has no MTP head, so `resolve_mtp_plan` gives
  `Off` and the proposer is `NoProposer` (no request field selects n-gram or a draft model on this
  provider); the request is priced on the `Off` plan, whose geometry covers the static
  preallocation (E6, below), and the Gemma 4 splice — whose soft-token spans are only known after
  its front-ends run — admits the preallocation for its expanded prompt before allocating it. The
  `DecodeRecord` says `step_model` / `static` / `gqa` (Gemma 4: `expanded`, its sliding layers) /
  `proposer=none`. `LlamaProvider::set_causal_decode_path(DecodePath::Reference)` keeps the
  `Decode` loop selectable as the oracle. A Qwen-VL multimodal request (DeepStack / M-RoPE prefill)
  stays on the `VlmDecode` reference loop it shares with the hybrid.
* **`StarCoder2` (StarVector-8B) and the StarVector-1B GPTBigCode decoder implement `StepModel`**.
  StarCoder2 has the same selector as `CausalLm`: `Gqa` by default on its reference `Decode` and
  its growing backing, `Expanded` selectable (`StarCoder2::set_attn_formulation`). The 1B decoder
  no longer owns a KV cache (its layers held one each, reset by the provider): the forward is
  `&self` and the request's K/V live in its `StepKvCache`.
* **No private decode loops remain.** LLaVA's caption loop and the StarVector-1B provider's loop are
  gone; LLaVA and both StarVector providers prefill their conditioning into the step cache and decode
  through the engine (`decode::generate_step_from_prefill` — the engine with `NoProposer`). They run
  on the cache's growing backing: these providers have no admission surface, and a static cache would
  preallocate the whole caption / SVG budget up front (16k positions on StarVector-8B) — the static
  backing of each of them is exercised by the parity tests below.
* **`decode/speculative.rs` holds no decode loop** (AC3) — only `SpeculativeStats`. The pre-epic
  `CausalLm` prompt-lookup and draft-model loops are deleted; prompt lookup and draft-model
  speculation run through the engine's `NgramProposer` / `DraftModelProposer` (`decode/proposers.rs`),
  `tests/speculative.rs` ported onto them with its assertions unchanged. The retired draft loop's
  vocab-mismatch refusal moved into the engine (`Proposer::vocab_size`), checked before any
  inference.
* **Admission (E6)** prices the widest layer: the causal family's geometry carries the
  `(kv_heads, head width)` of the **one** layer with the largest `kv_heads × max(key_dim,
  value_dim)` (`KvLayout::widest_layer`), which covers Gemma 4's full-attention layers and MLA's
  full-head keys (the scalar `head_dim` / `num_key_value_heads` it read before under-priced both)
  without crossing one layer type's head count with another's width (on a Gemma 4 12B-style stack —
  sliding 8×256, full `k_eq_v` 1×512 — independent maxima would price 8×512, twice any layer).
  `static_kv_bytes` equals what the static cache allocates; `provider.rs::causal_admission_covers_the_static_kv_preallocation`
  holds llama and Gemma 4 `k_eq_v` to a KV term **equal** to the preallocation (no inflation) and a
  non-`k_eq_v` Gemma 4 (full 2×16 wider than sliding 2×8) to the full layers' geometry.
* **Telemetry (E2)**: every seam run's `DecodeRecord` names `kv_cache` (`static` / `growing`) and
  `attn_formulation` (`gqa` only when every layer ran un-expanded); the provider's reference record
  reports the causal selector's effective formulation instead of a hard-coded `expanded`.

## AC1 — tiny-config parity (CPU f32)

`crates/llm/candle-llm/tests/step_seam_migration.rs`; goldens in
`crates/llm/candle-llm/tests/goldens/sc24138/`, **written at commit `a226b3e0c`** — the pre-migration
tree plus only the test file and a behaviour-neutral geometry knob on the StarVector-1B decoder — by
running the pre-migration paths: the `CausalLm` reference loop, LLaVA's own caption loop, StarCoder2
through the shared reference loop, and the StarVector-1B decoder with its layer-owned cache. Each
golden: prompt, 12 greedy tokens, the 12 logit rows they were drawn from, and 12 sampled tokens
(temperature 0.8, top-p 0.9, repetition penalty 1.1). The golden producers now select
**`Expanded`** explicitly (`causal_before`, `gemma4_mm_before`, `llava_before` through
`LlavaModel::set_attn_formulation`, `starcoder2_before`), since the default is `Gqa`. Full log:
`tiny-config-parity-cpu.log`.

| model | `Expanded` selected (reference / growing / paged) vs golden | default (`Gqa`) reference loop, growing and paged backings | static step cache: greedy + sampled tokens | static: max \|Δlogit\| vs golden | engine (none / n-gram / self-draft) | extra |
|---|---|---|---|---|---|---|
| llama (`LlamaForCausalLM`) | bit-exact | golden tokens; rows = static, bit for bit | identical | 1.0e-6 | identical | paged backing: = static (default), = golden (expanded) |
| Qwen3 dense (Qwen3-8B block: q/k-norm, head_dim ≠ hidden/heads) | bit-exact | golden tokens; rows = static, bit for bit | identical | 1.1e-6 | identical | |
| Gemma 4 (sliding 3 < prompt 9, two layer types, `k_eq_v`) | bit-exact | golden tokens; rows = static, bit for bit | identical | 6.7e-6 | identical | `expanded` label (sliding layers) |
| Gemma 4 soft-token splice (`gemma4_mm` prefill) | bit-exact | growing rows = static, bit for bit | identical (static + growing) | 6.2e-6 | identical | prefill through `step_prefill_from_embeds` |
| LLaVA (SigLIP + Llama) | bit-exact (the caption through the seam) | golden tokens; caption rows = static, bit for bit | identical | — | — | post-prefill cancel = empty `Cancelled`, as before |
| StarCoder2 (StarVector-8B decoder) | bit-exact | golden tokens; reference and growing rows = static, bit for bit | identical | 1.9e-6 | — | conditioning prefix through the seam |
| StarVector-1B (GPTBigCode, MQA) | bit-exact (stateless decoder; its MQA fold is un-expanded either way) | — | identical | bit-exact | — | past `n_positions` fails closed |
| DeepSeek-V2 MLA (`tests/mla.rs`) | — | — | identical to `generate` | — | — | static K/V of different widths; bytes = priced |

Tolerance for the static rows: `STATIC_LOGIT_TOL = 1e-4`; the observed maxima above are two orders
below it (the un-expanded attention GEMMs round differently in the last bits).

**Portability of the goldens.** "Bit-exact against a golden" holds in the configuration the goldens
were measured on — Windows x86_64 MSVC, which both the Windows CPU lane and the `--features cuda`
lane reproduce (these tests run on `Device::Cpu`). On WSL Ubuntu 24.04 (glibc, same Rust 1.96.0)
the pre-fix suite failed 12 of 17 tests: every golden **token** was reproduced, and the logits
differed in the last bits (platform libm / GEMM rounding). As `architecture_forward.rs` scopes its
goldens, the suite now compares a golden's logits bit for bit only in the measured configuration
(`goldens_bit_exact`, pinned by `golden_bit_exactness_is_limited_to_the_measured_configuration`)
and within `STATIC_LOGIT_TOL` elsewhere; tokens stay exact everywhere, and every comparison between
two paths of this tree (default growing vs static, reference vs static, …) stays bit for bit on
every platform. On WSL Ubuntu at `c5cfe30a3`: **18 / 18** pass; with the bit-exact rule forced on
every platform (the pre-fix behaviour) the same build fails 13 (the 12 above plus the selection
test). Log: `tiny-config-parity-wsl-ubuntu.log`.

## Review-fix mutations (sc-24138 review)

Every assertion added or changed by the review fixes was run against a mutation, one at a time on
a touched source, and turned red (CPU lane unless noted); restored, all pass. Per-mutation
failure lines: `review-fix-mutations.log`.

| assertion | mutation | result |
|---|---|---|
| default growing backing = static cache, bit for bit (`causal_after`) | `CausalLm` growing step computes `Expanded` while labelled `Gqa` | red |
| default growing label = static label; `Gqa` is the default | `CausalLm` default `Gqa` → `Expanded` | red (llama, Qwen3 dense, Gemma 4, Gemma 4 splice, LLaVA) |
| default reference loop = static cache, bit for bit | `CausalLm` reference loop computes `Expanded` | red (Qwen3 dense; LLaVA caption rows) |
| `Expanded` selected on growing / paged reproduces the golden | growing / paged backings ignore the selector | red (llama label; Gemma 4 rows) |
| golden producers select `Expanded` | `causal_before` / `gemma4_mm_before` / `llava_before` / `starcoder2_before` without the selection | red (each) |
| `LlavaModel::set_attn_formulation` reaches the decoder | the knob is a no-op | red |
| StarCoder2: default growing = static; default reference = static; `Expanded` growing = golden | growing computes `Expanded` while labelled `Gqa`; `logits_from_embeds` hard-codes `Expanded`; growing ignores the selector; default → `Expanded` | red (each) |
| goldens bit-exact only in the measured configuration | `goldens_bit_exact` always true | red (Windows: selection test; WSL: 13 tests) |
| `widest_layer` is one layer's pair (unit tests) | independent maxima (the pre-fix code) | red |
| admission: geometry = widest layer; no inflation (`k_eq_v`); full layers widest (non-`k_eq_v`) | independent maxima; scalar `num_key_value_heads × head_dim` | red (each) |
| admission: priced `>=` the static preallocation (9 + 200 positions) | the `Off` plan's geometry loses its KV term | red |
| seam overshoot: cache = `static_kv_bytes(capacity + 3)` | `new_cache_for` ignores the overshoot | red |
| provider: default causal request → `step_model` / `static` / `gqa` / `proposer=none` | the engine route disabled | red (`thinking.rs`) |
| provider: reference selectable → `reference` / `growing` | `set_causal_decode_path` does not store | red (`thinking.rs`, `gemma4_multimodal.rs`) |
| provider: only `step_model` / `reference` accepted | `PromptLookup` accepted | red |
| provider: Gemma 4 splice through the engine | the splice routed to the reference loop | red |
| provider: the two loops give the same tokens (CPU f32) | the reference loop draws with another seed (`thinking.rs`); halves the Gemma 4 prefill embeddings (`gemma4_multimodal.rs`) | red (each). A first Gemma 4 mutation, one token fewer, stayed green: the fixture stops on its EOS before the budget, so it changed nothing. |
| `architecture_forward` selects `Expanded` (its goldens are that arithmetic) | the selection removed — **CUDA lane**, Windows+CUDA golden | red; restored 7 / 7 |

## AC2 — Qwen3-8B through the shared engine (RTX Pro 6000 / sm_120)

`decode-bench/` — sealed by `scripts/release/decode_bench.py run` (binary, source sha, model pin,
hardware, co-tenants and memory samples in each `run.json`; `SEAL.json` hashes). **Final code:
`c5cfe30a3`** (clean tree, GPU 1 idle at start), one release binary, the prose fixture (53 prompt
tokens, 256 new), rows `reference,reference_unfused,step_model,ngram`, `DECODE_BENCH_NGRAM_DRAFTS=2,3,4`:

* `head-c5cfe30a3-gqa-ref` — the default: every row `gqa`;
* `head-c5cfe30a3-expanded-ref` — `DECODE_BENCH_ATTN=expanded`, the labelled comparison: the
  reference rows run the pre-migration `repeat_kv` arithmetic ("expanded attn").

| run | row | tok/s | acceptance | fwd/tok | syncs/tok | syncs/verify | tokens vs that run's reference |
|---|---|---|---|---|---|---|---|
| gqa-ref (default) | **reference** (`CausalLm` loop, growing kv, gqa) | 72.09 | — | 1.000 | 1.00 | — | (ref) |
| gqa-ref | reference, **fused kernels off** | 32.60 | — | 1.000 | 1.00 | — | **identical (256/256)** |
| gqa-ref | **StepModel** (static kv, gqa) | **75.23** | — | 1.000 | 1.00 | 1.00 | **identical (256/256)** |
| gqa-ref | n-gram K=2 / 3 / 4 (engine, static kv) | 74.98 / 73.29 / 74.05 | 0.147 / 0.087 / 0.076 | 0.875 / 0.887 / 0.875 | 0.88 / 0.89 / 0.88 | 1.00 | @65 / @51 / @132 |
| expanded-ref (comparison) | reference, **expanded attn** (the pre-migration path) | 64.34 | — | 1.000 | 1.00 | — | (ref); vs the default reference: @65 |
| expanded-ref | reference, expanded, fused kernels off | 28.93 | — | 1.000 | 1.00 | — | **identical (256/256)** |
| expanded-ref | StepModel (static kv, gqa) | 74.62 | — | 1.000 | 1.00 | 1.00 | @65 (identical to the default reference) |
| expanded-ref | n-gram K=2 / 3 / 4 | 75.22 / 72.72 / 72.55 | 0.147 / 0.087 / 0.076 | 0.875 / 0.887 / 0.875 | 0.88 / 0.89 / 0.88 | 1.00 | @132 / @51 / @65 |

Reading it:

* **The reference is the static path's arithmetic now.** The default reference loop and the
  StepModel row on the static cache emit the same **256 / 256** tokens; the n-gram rows diverge at
  knife-edges (the S2 finding: a multi-row verify forward's projection GEMMs pick a different cuBLAS
  kernel than the M = 1 decode), with exactly **one** device→host transfer per verify step.
  *Correction (sc-24140, `docs/migration/evidence/sc-24140/measure/`): enumerated, @65 (0 ULP) and
  @132 (0.5 ULP) are ≤ 1 bf16 ULP knife-edges, but @51 (n-gram K=3) is a **2.0 ULP** reference gap
  — not a knife-edge under the decided rule; a single verify-shaped forward never flips it, the
  free-running row reaches it through K/V earlier verify forwards wrote.*
* **The old expanded reference is the labelled comparison.** It agrees with the default reference on
  the first 65 tokens and diverges at index 65 — the ≤ 1 bf16 ULP formulation knife-edge above;
  its StepModel row is token-identical to the default reference.
* **Throughput.** Against the default reference the static cache is within noise-to-small-gain
  (75.2 vs 72.1 tok/s; 72.7 vs 70.9 and 70.6 vs 71.1 in the earlier runs below): at 256 tokens on an
  8B model the win is the attention arithmetic, not the cache. The **+16 % / +22 %** earlier runs
  showed (74.3 vs 64.1, 74.4 vs 60.8 tok/s) — **+16 %** here (74.6 vs 64.3) — is the gain over the
  **old expanded reference**: most of it is the move from `repeat_kv` expansion to the un-expanded
  formulation, which the default reference now shares.
* **Fused kernels on vs off (S9 entry points in the llama layers: `rms_norm_residual`, `swiglu`,
  `rms_norm_rope`).** `reference_unfused` runs the reference loop with the fused switch off for that
  row only (`off: 0 fused / 46336 ref`, against `on: 46336 fused / 0 ref`): **token-identical over
  all 256 tokens** under both formulations, at 2.2× the tok/s with them on.
* **CUDA graphs:** S6 has not landed on `feature/sc-24128-fast-decode-blackwell`, so there is no
  graphs-on row; every row is eager.

`comparison.md` merges the same-prompt runs, the final-code default run first. The earlier runs
(kept, sealed) were taken when `Expanded` was still the `CausalLm` default, so their "expanded-ref"
runs are the default-at-the-time reference and their "gqa-ref" runs selected `Gqa` with
`DECODE_BENCH_ATTN=gqa`: `833e61542` is the story's code before merging the S2 head (Rust sources
identical to `110fcddf1`, where that binary was built), `0e3d5a1d3` after it. Every row's tokens,
divergence points, acceptance and forwards are identical across `833e61542`, `0e3d5a1d3` and
`c5cfe30a3`; only wall-clock throughput moves.

| run | row | tok/s | acceptance | syncs/verify | tokens vs that run's reference |
|---|---|---|---|---|---|
| 833e61542 expanded-ref | reference (expanded, then the default) | 64.09 | — | — | (ref) |
| 833e61542 expanded-ref | StepModel (static kv, gqa) | 74.33 | — | 1.00 | @65 |
| 833e61542 expanded-ref | n-gram K=2 / 3 / 4 | 74.18 / 75.15 / 75.56 | 0.147 / 0.087 / 0.076 | 1.00 | @132 / @51 / @65 |
| 833e61542 gqa-ref | reference (gqa selected) | 70.87 | — | — | (ref) |
| 833e61542 gqa-ref | StepModel (static kv, gqa) | 72.74 | — | 1.00 | identical (256/256) |
| 833e61542 gqa-ref | n-gram K=2 / 3 / 4 | 74.09 / 67.97 / 73.15 | 0.147 / 0.087 / 0.076 | 1.00 | @65 / @51 / @132 |
| 833e61542 gqa-ref-structured (JSON-repeat prompt, 80 tokens) | reference (gqa) | 70.77 | — | — | (ref) |
| 833e61542 gqa-ref-structured | StepModel (static kv, gqa) | 73.27 | — | 1.00 | identical |
| 833e61542 gqa-ref-structured | n-gram K=2 / 3 / 4 / 6 | 79.51 / 81.00 / 78.55 / **88.14** | 0.232 / 0.164 / 0.128 / 0.137 | 1.00 | @58 / @58 / @58 / @55 |
| 0e3d5a1d3 expanded-ref | reference (expanded, then the default) | 60.83 | — | — | (ref) |
| 0e3d5a1d3 expanded-ref | StepModel (static kv, gqa) | 74.42 | — | 1.00 | @65 |
| 0e3d5a1d3 expanded-ref | n-gram K=2 / 3 / 4 | 72.25 / 68.23 / 74.31 | 0.147 / 0.087 / 0.076 | 1.00 | @132 / @51 / @65 |
| 0e3d5a1d3 gqa-ref | reference (gqa selected) | 71.05 | — | — | (ref) |
| 0e3d5a1d3 gqa-ref | StepModel (static kv, gqa) | 70.61 | — | 1.00 | identical (256/256) |
| 0e3d5a1d3 gqa-ref | n-gram K=2 / 3 / 4 | 71.72 / 75.31 / 75.39 | 0.147 / 0.087 / 0.076 | 1.00 | @65 / @51 / @132 |

On the structured prompt n-gram K = 6 reaches **88.1 tok/s, +25 %** over its (gqa) reference.
`head-833e61542-contended` (run name `head-833e61542`) is a first run of the same binary while
another story's `device_sampler` test held GPU 1 at 94 % utilization (listed in its `run.json`
co-tenants): 17.9 / 39.0 / 20–22 tok/s — kept, sealed, as the record of what lane contention does to
these numbers.

## Real-weight speculative suites (`tests/speculative.rs`, `--ignored`)

The ported prompt-lookup and draft-model suites, run at `c5cfe30a3` from the release `--features
cuda` binary on GPU 1 (idle, 880 MiB used at start), `--ignored --test-threads=1`, log
`speculative-real-weight-ignored.log`. `CANDLE_LLM_QWEN3_MODEL` = `Qwen/Qwen3-8B` @ `b968826d9c46`
(the epic's pinned snapshot, `E:\huggingface`); `CANDLE_LLM_TEST_MODEL` = the llama-architecture
`HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f77366` (cached in the user HF cache; the llama
suites and the vocab-mismatch pair need a llama snapshot with a vocabulary other than Qwen3's).
The draft suites need no separate draft snapshot: the draft is the target's own weights quantized
on load (Q4, falling back to Q8).

| test | result |
|---|---|
| `prompt_lookup_qwen3` | ok — K = 0 equals non-speculative on both prompts; repetitive prompt: 48 tokens in 13 target forwards (3.69 tok/forward), 35 / 35 drafts accepted, tracks non-speculative 48 / 48 |
| `draft_speculative_qwen3` (Q4 draft of Qwen3-8B) | ok — K = 0 exact; 48 tokens in 12 target forwards (4.00 tok/forward), 36 / 43 drafts accepted, tracks dense 48 / 48; seeded sampling deterministic |
| `prompt_lookup_llama` (SmolLM2) | ok — 48 tokens in 13 forwards, 35 / 35 accepted, tracks 48 / 48 |
| `draft_speculative_llama` (SmolLM2, quantized draft) | ok — 48 tokens in 12 forwards, 36 / 41 accepted, tracks dense 48 / 48 |
| `draft_speculative_rejects_vocab_mismatch` (SmolLM2 vs Qwen3-8B) | ok — refused |

## AC3

`tests/step_seam_migration.rs::speculative_rs_holds_no_decode_loop` reads `src/decode/speculative.rs`
and fails on any code line containing `fn `, `loop {`, `while `, `for `, `CausalLm`, `decode_logits`
or `KvCache`, and checks the n-gram / draft proposers are in `decode/proposers.rs`.

## Gates run (final code `c5cfe30a3`)

* CPU (Git Bash, Windows): `cargo test --locked -p candle-llm` (lib, integration, doc) — all green
  except `architecture_forward::every_architecture_forward_is_bit_identical_to_the_base_branch`, which
  is **pre-existing**: it fails with the identical message (`llama` index 1, `0xbf80e46b` vs
  `0xbf80e46c`) on the untouched pre-fix head `90fa2a1d2` in this configuration (Windows CPU without
  `--features cuda` reads the non-Windows golden). With the un-expanded default and no `Expanded`
  selection it failed differently (index 0) — the suite now selects `Expanded`, the arithmetic its
  goldens were captured with. `cargo clippy --locked -p candle-llm --all-targets -- -D warnings`,
  `cargo fmt -p candle-llm -- --check`, `RUSTDOCFLAGS=-D warnings cargo doc --no-deps -p candle-llm`,
  `cargo check -p candle-llm -p runtime-cpu`, `python -m pytest scripts/tests/test_decode_bench.py`,
  `scripts/check-workspace.py`, `scripts/check_docs.py`.
* CUDA (PowerShell, MSVC 14.44 vcvars, `CUDA_COMPUTE_CAP=120`, `CUDA_VISIBLE_DEVICES=1`):
  `cargo test --locked --lib --tests -p candle-llm --features cuda` — **all green** (39 binaries +
  lib), including `architecture_forward` against the Windows+CUDA golden;
  `cargo clippy --locked -p candle-llm --all-targets --features cuda -- -D warnings`;
  `RUSTDOCFLAGS=-D warnings cargo doc --no-deps -p candle-llm --features cuda`.
* Linux (WSL Ubuntu 24.04, Rust 1.96.0, CPU): `cargo test --locked -p candle-llm` — all green
  except the same pre-existing `architecture_forward` failure, identical (same index and bits) on
  the untouched pre-fix head there too; `step_seam_migration` 18 / 18.
