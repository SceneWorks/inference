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
  sliding window); a new `attn_formulation` selector (`Expanded` by default) keeps the `CausalLm`
  reference loop **bit-identical** to the pre-migration tree (E2) and can select `Gqa`, which makes
  the reference and the static path token-identical by construction (see the bench). The llama
  layers now call the S9 fused entry points where the math matches (`rms_norm_residual`, `swiglu`,
  `rms_norm_rope`); Gemma's sandwich norms and GeGLU keep their op chains.
* **`StarCoder2` (StarVector-8B) and the StarVector-1B GPTBigCode decoder implement `StepModel`**.
  The 1B decoder no longer owns a KV cache (its layers held one each, reset by the provider): the
  forward is `&self` and the request's K/V live in its `StepKvCache`.
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
  vocab-mismatch refusal moved into the engine (`Proposer::vocab_size`), and the engine's bonus draw
  is now the reference sampler bit for bit (`sample_host`: the same argmax fallback for a fully-masked
  row, and no uniform consumed there) — so the engine with no proposer is exactly the reference loop.
* **Admission (E6)** prices the widest layer: the causal family's geometry now carries the widest
  per-layer `(kv_heads, head_dim)` from `CausalLm::kv_layout`, which covers Gemma 4's full-attention
  layers and MLA's full-head keys (the scalar `head_dim` / `num_key_value_heads` it read before
  under-priced both). `static_kv_bytes` equals what the static cache allocates.
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
(temperature 0.8, top-p 0.9, repetition penalty 1.1). Full log: `tiny-config-parity-cpu.log`.

| model | reference path (bit-exact tokens + logits) | static step cache: greedy + sampled tokens | static: max \|Δlogit\| vs golden | growing step cache | engine (none / n-gram / self-draft) | extra |
|---|---|---|---|---|---|---|
| llama (`LlamaForCausalLM`) | yes | identical | 1.0e-6 | bit-exact | identical | paged backing bit-exact; `gqa` label |
| Qwen3 dense (Qwen3-8B block: q/k-norm, head_dim ≠ hidden/heads) | yes | identical | 1.1e-6 | bit-exact | identical | reference with `Gqa` selected = static, bit for bit |
| Gemma 4 (sliding 3 < prompt 9, two layer types, `k_eq_v`) | yes | identical | 6.7e-6 | bit-exact | identical | `expanded` label (sliding layers) |
| Gemma 4 soft-token splice (`gemma4_mm` prefill) | yes | identical (static + growing) | 6.2e-6 | bit-exact | identical | prefill through `step_prefill_from_embeds` |
| LLaVA (SigLIP + Llama) | yes (the caption now decodes through the seam) | identical | — | bit-exact (the provider path) | — | post-prefill cancel = empty `Cancelled`, as before |
| StarCoder2 (StarVector-8B decoder) | yes | identical | 1.9e-6 | bit-exact | — | conditioning prefix through the seam |
| StarVector-1B (GPTBigCode, MQA) | yes (stateless decoder, growing backing) | identical | bit-exact | bit-exact | — | past `n_positions` fails closed |
| DeepSeek-V2 MLA (`tests/mla.rs`) | — | identical to `generate` | — | — | — | static K/V of different widths; bytes = priced |

Tolerance for the static rows: `STATIC_LOGIT_TOL = 1e-4`; the observed maxima above are two orders
below it (the un-expanded attention GEMMs round differently in the last bits).

## AC2 — Qwen3-8B through the shared engine (RTX Pro 6000 / sm_120)

`decode-bench/` — sealed by `scripts/release/decode_bench.py run` (binary, source sha, model pin,
hardware, co-tenants and memory samples in each `run.json`; `SEAL.json` hashes). All runs are
`833e61542` (Rust sources identical to `110fcddf1`, where the binary was built; the one commit between
is the harness's llama-family checkpoint rule). `comparison.md` merges the three same-prompt runs.

| run | row | tok/s | acceptance | fwd/tok | syncs/tok | syncs/verify | tokens vs that run's reference |
|---|---|---|---|---|---|---|---|
| expanded-ref | reference (`CausalLm` loop, growing kv, expanded — the pre-migration path) | 64.09 | — | 1.000 | 1.00 | — | (ref) |
| expanded-ref | StepModel (static kv, gqa) | **74.33** | — | 1.000 | 1.00 | 1.00 | first divergence @65 |
| expanded-ref | n-gram K=2 (engine, static kv) | 74.18 | 0.147 | 0.875 | 0.88 | 1.00 | @132 |
| expanded-ref | n-gram K=3 | 75.15 | 0.087 | 0.887 | 0.89 | 1.00 | @51 |
| expanded-ref | n-gram K=4 | 75.56 | 0.076 | 0.875 | 0.88 | 1.00 | @65 |
| gqa-ref | reference (growing kv, **gqa** selected) | 70.87 | — | 1.000 | 1.00 | — | (ref) |
| gqa-ref | StepModel (static kv, gqa) | 72.74 | — | 1.000 | 1.00 | 1.00 | **identical (256/256)** |
| gqa-ref | n-gram K=2 / 3 / 4 | 74.09 / 67.97 / 73.15 | 0.147 / 0.087 / 0.076 | 0.875 / 0.887 / 0.875 | 0.88 / 0.89 / 0.88 | 1.00 | @65 / @51 / @132 |
| gqa-ref-structured (JSON-repeat prompt, 80 tokens) | reference (gqa) | 70.77 | — | 1.000 | 1.00 | — | (ref) |
| gqa-ref-structured | StepModel (static kv, gqa) | 73.27 | — | 1.000 | 1.00 | 1.00 | identical |
| gqa-ref-structured | n-gram K=2 / 3 / 4 / 6 | 79.51 / 81.00 / 78.55 / **88.14** | 0.232 / 0.164 / 0.128 / 0.137 | 0.770 / 0.762 / 0.770 / 0.695 | 0.77 / 0.76 / 0.77 / 0.70 | 1.00 | @58 / @58 / @58 / @55 |

Reading it:

* The step seam's static cache is **+16 %** over the pre-migration `CausalLm` reference on the prose
  fixture (74.3 vs 64.1 tok/s): no per-step `cat` of the history and no `repeat_kv` expansion.
* With the `Gqa` formulation selected on the reference loop, reference and static step path are
  **token-identical over all 256 tokens** — the same arithmetic on two caches. Against the expanded
  reference the static path diverges at token 65, the S4 knife-edge class (expanded vs un-expanded
  attention GEMMs differ in the last bf16 bit).
* The n-gram rows diverge from the single-token reference at knife-edge positions — the S2 finding
  (a multi-row verify forward's projection GEMMs pick a different cuBLAS kernel than the M = 1
  decode); acceptance is low on free prose (8–15 %) and higher on the structured prompt, where K = 6
  reaches **88.1 tok/s, +25 %** over the reference. Every verify step costs exactly **one**
  device→host transfer (`syncs/verify` 1.00).
* `head-833e61542-contended` (run name `head-833e61542`) is the first run of the same binary while
  another story's `device_sampler` test held GPU 1 at 94 % utilization (listed in its `run.json`
  co-tenants): 17.9 / 39.0 / 20–22 tok/s. Kept, sealed, as the record of what lane contention does to
  these numbers; the other runs had GPU 1 idle at start.
* **CUDA graphs:** S6 has not landed on `feature/sc-24128-fast-decode-blackwell`, so there is no
  graphs-on row; every row is eager.

## AC3

`tests/step_seam_migration.rs::speculative_rs_holds_no_decode_loop` reads `src/decode/speculative.rs`
and fails on any code line containing `fn `, `loop {`, `while `, `for `, `CausalLm`, `decode_logits`
or `KvCache`, and checks the n-gram / draft proposers are in `decode/proposers.rs`.

## Gates run

* CPU (Git Bash): `cargo test --locked -p candle-llm` (lib, integration, doc) — all green except
  `architecture_forward::every_architecture_forward_is_bit_identical_to_the_base_branch`, which fails
  identically on the untouched S2 head (`833265c03`, clean target dir) in this configuration: Windows
  CPU without `--features cuda` reads the non-Windows golden (`forward_candle.json`) and differs by one
  ULP; the Windows+CUDA configuration's own golden passes (below). `cargo clippy --locked -p candle-llm
  --all-targets -- -D warnings`, `cargo fmt -p candle-llm -- --check`, `RUSTDOCFLAGS=-D warnings cargo
  doc --no-deps -p candle-llm`, `python -m pytest scripts/tests/test_decode_bench.py`,
  `scripts/check-workspace.py`, `scripts/check_docs.py`.
* CUDA (PowerShell, MSVC 14.44 vcvars, `CUDA_COMPUTE_CAP=120`, `CUDA_VISIBLE_DEVICES=1`):
  `cargo test --locked --lib --tests -p candle-llm --features cuda` (37 binaries, all green, including
  `architecture_forward` against the Windows+CUDA golden), `cargo clippy --locked -p candle-llm
  --all-targets --features cuda -- -D warnings`.
