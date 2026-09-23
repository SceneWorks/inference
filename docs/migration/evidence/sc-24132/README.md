# sc-24132 — Static (preallocated, in-place, zero-copy GQA) KV cache: evidence

Story S4 of epic sc-24128. Host: Windows 11, **RTX Pro 6000 / sm_120** (GPU 0), CUDA 12.9, MSVC 14.44,
`CUDA_COMPUTE_CAP=120`, release build with `--features cuda`. Model: `Qwen/Qwen3.8-27B` @
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (`bonsai-qwen38-parent`), BF16 greedy, 97 prompt tokens,
256 new tokens per row — the same fixture and harness as `docs/migration/evidence/sc-24129/`.

## Decode bench (sealed, `decode-bench/`)

Two sealed head runs of `tests/decode_bench.rs` (MTP-off rows) through `scripts/release/decode_bench.py`,
compared against S1's sealed head (`sc-24129/decode-bench/head-4a059a87b`) in `decode-bench/comparison.md`
(re-verify the seals with `python scripts/release/decode_bench.py table <the three run dirs>`):

| run | row | match ref | tok/s | cache live | cache checkpoints |
|---|---|---|---|---|---|
| S1 head-4a059a87b | MTP off (reference, `AttnKv`) | (ref) | 12.17 | n/a | n/a |
| S1 head-4a059a87b | MTP off (StepModel, growing `AttnKv`) | yes | 11.85 | 168.8 MiB | 293.6 MiB |
| head-e7805eee4 | MTP off (reference, `AttnKv`) | (ref) | 11.23 | n/a | n/a |
| head-e7805eee4 | **MTP off (StepModel, static kv)** | no @107 | **11.25** | 168.9 MiB | 293.6 MiB |
| head-adce2370e-growing-kv | MTP off (reference, `AttnKv`) | (ref) | 10.17 | n/a | n/a |
| head-adce2370e-growing-kv | MTP off (StepModel, growing kv) — `DECODE_BENCH_KV_CACHE=growing` | yes | 10.12 | 168.8 MiB | 293.6 MiB |

`head-e7805eee4` and `head-adce2370e-growing-kv` are the same binary (`decode_bench-a589f4c981fe54a8.exe`,
sha256 recorded in each `run.json`); the second sha only adds the first run's evidence files.

### Throughput: no regression, no measurable win — and why

* **Within a run** (the only valid comparison on this box — see below): static KV decodes at **11.25 vs
  11.23 tok/s** for the reference row (+0.2 %); the growing control decodes at 10.12 vs 10.17 (−0.5 %);
  S1's growing StepModel row was 11.85 vs 12.17 (−2.6 %). All three deltas are inside the ±0.3–0.7 tok/s
  run-to-run noise S1 recorded.
* **Across runs the absolute numbers move by ~10 %** (reference 12.17 → 11.23 → 10.17) with the *other*
  card's load: the two RTX Pro 6000 Max-Q cards share the workstation's power/thermal budget and PCIe.
  `nvidia-smi` sampled right before each run: GPU 0 idle for all three; GPU 1 at 885 MiB / 0 % for
  `head-e7805eee4`, at **54 GB / 62 %** (a sibling story's 27B run) for `head-adce2370e-growing-kv`. A first
  static run at an earlier sha, taken while GPU 1 was under the same load, read 9.17 (reference) / 9.40
  (static) and was discarded for that reason (it is not part of the sealed evidence).
* **Why attention is invisible in tok/s here:** a Qwen3.8-27B bf16 decode step streams ~54 GB of weights
  and launches several hundred kernels across 64 layers; the 16 full-attention layers' KV traffic at
  L ≤ 353 is ~12 MB and a handful of launches per step. Removing the per-step `cat`, `repeat_kv` and
  transpose copies (≈ 3 × H·L·d per layer) is well below the noise floor of this bench. What the static
  cache buys is the property the next stories need — stable device addresses and no per-step allocation
  (S6 CUDA graphs) — plus a memory profile that is the request's bound from the first step
  (`cache live` 168.9 MiB = 22.1 MiB of preallocated KV + 146.9 MiB of live DeltaNet state; the growing
  row's 168.8 MiB is its *final* size).

## Real-weight acceptance (`tests/static_kv_parity.rs`, `--ignored`, GPU 0)

| test | result |
|---|---|
| `ac3_static_kv_device_pointers_are_stable_across_the_fixture_and_a_rollback` | **ok** — 16 attention buffers' CUDA device pointers unchanged across the 256-token greedy run (checked every 32 steps) and a `rollback_to(349)` + re-decode; 23 134 208 bytes preallocated (`Qwen35Model::static_kv_bytes` exact). |
| `teacher_forced_static_vs_attn_kv_logit_parity_report` | **ok** — both paths fed the reference's 256 tokens: sampler argmax agrees at **255 / 256** positions; the one disagreement is position 107, where the reference's own top-2 gap is **0.125 = 1 bf16 ULP** of its top logit and the static path lands on an exact tie; mean per-position max \|Δlogit\| 0.254, max 2.70 (position 56, on a row of range 41.1). Gate: every disagreement is a reference knife-edge (top-2 gap ≤ 4 ULPs). |
| `ac1_static_kv_greedy_fixture_is_token_identical_to_attn_kv` | **FAILED at 107** — the literal AC1 (all 256 greedy tokens identical) is not met: the two decodes share tokens 0..106 and diverge at the knife-edge above. The growing `AttnKv` path through the same driver stays identical to the reference loop (S1's AC1, re-confirmed by the `growing-kv` bench row). |

### Why bit-exactness is out of reach for an un-expanded attention (the AC1 finding)

Bit-exact parity with the reference is a property of **cuBLAS kernel selection**, not of the cache. The
reference attends over `repeat_kv`-expanded heads: `batch = b·H = 24`, `m = 1`. Any formulation that does
not materialize the expansion issues different GEMMs (`m = groups = 6` and/or `batch = b·Hkv = 4`, or a
stride-0 broadcast), and cuBLAS picks its kernel — and with it the fp32 reduction order that decides the
last bf16 bit — by `m`, batch count and strides. `primitives::attention::tests::gqa_variants_bit_match_survey`
(CUDA, `#[ignore]`d, kept as evidence) compared six formulations to the reference at the 27B decode shape
over 131 key lengths (90..=1000 step 7):

| formulation | key lengths with a bf16-bit mismatch (of 131) |
|---|---|
| V0 folded groups, strided K (`OP_T`), strided V — **the shipped `sdpa_gqa_causal`** | 52 |
| V1 folded, contiguous Kᵀ (`OP_N`), strided V | 52 |
| V2 folded, contiguous Kᵀ and V | 52 |
| V3 per-KV-head, stride-0 broadcast (m = 1, batch = groups) | 32 |
| V4 V3 + contiguous Kᵀ | 28 |
| V5 V4 + contiguous V | 28 |

Every mismatch is a single bf16 ULP in a few elements. Only the reference's own calls (batch 24 over
expanded heads) reproduce the reference's bits, i.e. only the expansion AC2 removes. The 256-token fixture
carries a knife-edge at position 107 — S1's sealed MTP rows diverge from the reference at the same position.

Options (a product decision, not taken in this PR):

1. Gate AC1 as "token-identical except at reference knife-edges" — the `teacher_forced_…` test above
   (every disagreement must be a reference top-2 gap ≤ 4 bf16 ULPs).
2. Route the growing `AttnKv` path through `sdpa_gqa_causal` as well, so both paths share the attention
   arithmetic and are token-identical by construction. This changes the reference oracle's numerics (its
   tokens would no longer match S1's sealed pre-epic baseline at 107), so it is not done unilaterally.
3. Keep the expansion (a per-step `repeat_kv`-equivalent copy) for bit-exactness — defeats AC2.

## Weights-free gates (this head)

* Git Bash: `cargo test --locked -p candle-llm --lib` — 271 passed (11 new), 9 ignored; CPU clippy
  `-D warnings`, `cargo fmt --check`, rustdoc `-D warnings`, `check-workspace.py`, `check_docs.py`,
  `check_clock_assertions.py --check-baseline`, `pytest scripts/tests/test_decode_bench.py` (11 passed).
* PowerShell (MSVC 14.44 vcvars, `CUDA_COMPUTE_CAP=120`): `cargo test --locked --lib --tests -p candle-llm
  --features cuda` all green, including `static_tests::cuda_device_pointers_are_stable_across_100_steps_and_rollback`
  and `gqa_causal_matches_repeat_kv_sdpa_on_cuda_bf16_27b_shape`; CUDA clippy `-D warnings`.
