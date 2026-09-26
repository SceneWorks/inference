# candle-audio-yue2 test fixtures

| Fixture | Generator | Used by |
| --- | --- | --- |
| `vae_tiny/{standard,legacy}/{config.json,model.safetensors}`, `vae_tiny_reference.{safetensors,json}` | `scripts/reference/yue2/vae_reference.py tiny` | lib tests of `vae`, `decode`, `latent` (always run in CI) |
| `vae_real_reference.json` | `scripts/reference/yue2/vae_reference.py real` | `tests/vae_real_weights.rs` (`#[ignore]`d) |
| `protocol/` | `scripts/reference/yue2/protocol_fixtures.py` | tokenizer / protocol / plan tests (see `protocol/README.md`) |
| `ar_synthetic.json`, `sampler.json`, `ar_real_weights.json` | `scripts/reference/yue2/ar_fixtures.py` | `parity` lib tests (see the section below) |
| `nar_synthetic.{json,safetensors}`, `nar_real_reference.json` | `scripts/reference/yue2/nar_fixtures.py` | `nar` lib tests, `tests/nar_real_weights.rs` (`#[ignore]`d) (see the section below) |

Provenance: upstream `github.com/multimodal-art-projection/YuE` @
`92a73cc7652fcc1f937855e4b765e0a0edd7ff2e` (the installed pinned `yue2` package, never a copy), run
on the CPU in FP32 in the shared reference environment (`scripts/reference/yue2/setup_reference_env.sh`);
each JSON's `reference` block records Python, torch, numpy, safetensors and transformers versions.

- The `vae_tiny` models are **synthetic**: the released VAE topology (strides `[2, 2, 4, 4, 5, 6]`,
  64 latent channels, SnakeBeta, weight-norm, stereo, no final tanh) at toy widths with seeded random
  weights. They contain no YuE2 weights.
- The real-weight reference uses `m-a-p/YuE2-Vae` @ `95535e72a97bc0f09b8ada125d26b4009428c0e8` and
  `m-a-p/YuE2-Vae-legacy` @ `b54118f0fc462f08999d1ec07e88817f4ee3f770`. Its waveforms derive from
  CC BY-NC 4.0 weights, so the tensors stay outside the repository
  (`~/.cache/sceneworks-yue2-fixtures/vae/vae_real_reference.safetensors`); only their SHA-256,
  shapes and statistics are committed here, and the test refuses a reference file with another hash.

## YuE2 autoregressive-stage fixtures (sc-22991)

Produced by `scripts/reference/yue2/ar_fixtures.py` from the **pinned upstream** code, imported
from the shared reference environment (`scripts/reference/yue2/README.md`):

| What | Pin |
| --- | --- |
| Upstream source | `github.com/multimodal-art-projection/YuE` @ `92a73cc7652fcc1f937855e4b765e0a0edd7ff2e` — `yue2.modeling_yue2`, `yue2.sampling`, `yue2.protocol`, `yue2.tokenization_yue2` |
| Python stack | Python 3.12.13, torch 2.10.0 (CPU), numpy 2.2.6, transformers 4.57.6, tiktoken 0.12.0 (each file's `reference` block) |
| Weights (`ar_real_weights.json` only) | `m-a-p/YuE2-3B` @ `1a96eca688d6ae5d7f0feb88573fec89920fcd19`, `model.safetensors` SHA-256 checked against its `weights_manifest.json` before loading; BF16 checkpoint upcast to F32 |

No weights are committed. The files hold token ids, logit summaries and SHA-256 digests.

### Files

| File | Subcommand | Consumer |
| --- | --- | --- |
| `ar_synthetic.json` | `synthetic` | `parity::synthetic_decodes_match_upstream` (CI, `--lib`) |
| `sampler.json` | `synthetic` | `parity::sampler_rows_match_upstream`, `parity::rms_norm_and_rope_match_upstream` (CI) |
| `ar_real_weights.json` | `real` | `parity::real_weight_decodes_match_upstream` (`#[ignore]`, `YUE2_HF_HUB`) |

* **`ar_synthetic.json`** — upstream `generate_tokens` on a 2-layer, 32-wide `YuE2ForCausalLM`
  with the real architecture (GQA 4/2, per-head Q/K norm, non-square Q projection, RoPE θ = 10⁶)
  and the full 184 704-id vocabulary. Its weights are an integer hash of (tensor name, element
  index), rebuilt bit-identically by `model::synthetic`. Cases: greedy semantic, greedy with
  guidance 1.5 and an exact-score negative, greedy ABC, `cot = off` legacy with guidance 1.01 and
  an instruction-only negative, three stochastic decodes with injected uniforms, a natural
  `MUSIC_END` and a natural `ABC_END`, and a 606-token prefix (longer than one 512-position Rust
  prefill chunk).
* **`sampler.json`** — upstream `distribution` on three fixed 184 704-wide rows (a hash with two
  million levels) under eight sampling setups, in F32 and in the BF16 `legacy_off` arithmetic; the
  CFG line `u + s·(c − u)` in F32 and BF16; and upstream `RMSNorm` / `_apply_rotary` on a fixed
  `[1, 7, 16, 128]` input at cache positions 300–306.
* **`ar_real_weights.json`** — the exact prompt ids of five requests built by the upstream
  tokenizer and `token_prefixes` / `negative_prefix` from `examples/song.json` (style + lyrics):
  `cot` full (guidance 1.0), melody (1.5), off (default 1.01, instruction-only negative), a
  supplied full score `examples/score.abc` (2.0) and a supplied melody score `examples/melody.abc`
  (1.0). Planned modes decode 32 greedy ABC tokens, then every mode 24 greedy semantic tokens
  (`min_tokens` = budget, so all truncate); plus a planner that ends on its own (`ABC_END` after the
  teacher-forced melody score), a stochastic ABC decode with the released ABC controls and a
  stochastic `cot = off` semantic decode with the released semantic controls, both with injected
  draws. Every step records the logits row it sampled from as: top-8 ids/values over the phase's
  allow mask, fixed probe ids, and F64 Σx, Σx², logsumexp over the allowed row. The last step's
  row is also recomputed without a cache.

### Deliberate substitutions in the reference

1. **The categorical draw.** Torch's generator cannot be reproduced natively and a seed is no
   cross-platform bit-exact guarantee (epic E9). During each recorded decode `torch.multinomial`
   is replaced by the native draw — the first id, in vocabulary order, whose F64 cumulative
   probability exceeds `u · Σp` — with `u` taken from the recorded `uniforms` (SplitMix64,
   24-bit). `draw_margins` records how far each `u · Σp` landed from a CDF boundary (relative to
   Σp); every other line of upstream sampling runs unmodified.
2. **Tie order in top-p.** Upstream sorts with `torch.sort(descending=True)`, leaving the order of
   equal scores to the backend (CUDA's radix sort keeps ascending ids; the CPU's introsort does
   not). The native sampler uses ascending ids; the reference runs with `stable=True` so the order
   is the same. It only decides which of several **equal** scores straddling the top-p cut is
   removed — common only in BF16.

### Measured tolerances

Each bound was set after measuring, and each is documented next to its constant in
`src/parity.rs`.

| Check | Measured | Bound |
| --- | --- | --- |
| Synthetic model logits, 93 steps (F32) | max \|Δ\| 3.3e-6, \|Δ lse\| 6.3e-7 | 2e-5 (min reference top-1 margin 6e-3) |
| Sampler scores (F32, BF16-legacy), CFG mix, BF16 RMSNorm/RoPE | bit-identical | SHA-256 |
| F32 sampler probabilities | ≤ 1.3e-6 relative | 1e-5 |
| F32 RMSNorm / RoPE | ≤ 4.8e-7 | 4e-6 |
| Real weights: logits of 258 decode steps, every mode (F32) | max \|Δ\| 3.6e-5, \|Δ lse\| 3.6e-5, moments 1.2e-5 | 2e-4 (min reference top-1 margin 1.2e-2) |
| Real weights: cached decode vs native full recompute | 1.6e-5 | 1e-4 |

Token sequences (greedy and injected-draw) and truncation flags are compared exactly. Each
step's top-k ids: the top-1 id and the top-k set exactly, and an order swap counts only when the
reference gap between the swapped entries exceeds twice the logit tolerance (the smallest
adjacent top-8 gap on real weights is 6.9e-5, below twice the measured noise). On real weights
all 258 steps' tokens matched with no top-k mismatch.

The `modes` records also carry each `request` (style, lyrics, cot, seed, external ABC, CFG
scale): the native test rebuilds every prefix through its own tokenizer and `SymbolicPlan` and
checks it id for id against the upstream ids recorded beside it.

Run cost (Apple M-series CPU, F32): the reference `real` subcommand peaks at 21.0 GB RSS
(`ru_maxrss`: both MoT paths in F32 plus the mapped checkpoint) and takes ~50 s; the native
`real_weight_decodes_match_upstream` peaks at 9.5 GB (AR path only) and takes ~210 s. Never run the
two at once.

## YuE2 acoustic-stage (NAR) fixtures (sc-22992)

Produced by `scripts/reference/yue2/nar_fixtures.py` from the **pinned upstream** `yue2.nar`
(`synthesize`, `song_chunks`, `CachedNAR`) and `YuE2ForCausalLM.nar_velocity`, imported from the
shared reference environment. The only instrumentation is a wrapper around `CachedNAR.velocity`
that records each evaluation's input state, raw `t` and output; nothing is re-implemented.

| File | Subcommand | Consumer |
| --- | --- | --- |
| `nar_synthetic.json`, `nar_synthetic.safetensors` | `synthetic` | `nar::tests` (CI, `--lib`) |
| `nar_real_reference.json` | `real` | `tests/nar_real_weights.rs` (`#[ignore]`, `YUE2_HF_HUB`) |

* **Synthetic** — the 2-layer, 32-wide real-architecture MoT of `ar_synthetic.json` plus NAR heads
  (`vae2llm`, `llm2vae`, `time_embedder`, and a 64-row `latent_pos_embed.pe`) whose weights are the
  same integer hash, rebuilt bit-identically by `nar::synthetic`. torch runs single-threaded so the
  committed values are reproducible. The **injected noise** is upstream's own full-song draw
  (`song_chunks` with seed 831001), recorded per case and fed to the native side unchanged. Cases:
  the default 32-step midpoint solve; 7 and 1 steps; three original chunks (context cut to 8 frames
  per chunk, 21 frames); chunk edges at exactly two full chunks (16 frames) and one frame over (17);
  a 70-frame chunk whose 72 NAR positions exceed the 64-row position table (the clamp); upstream
  tiling by 5 query rows with `offload_ar=True` (final only); `CachedNAR` with `nar_cond_end = 5`
  (text-only visibility); `timestep_shift = 3`; and `joint/*` — the cached velocity against
  upstream's joint hybrid-mask forward `nar_velocity` (upstream agrees with itself to 1.7e-6).
* **Real** — `m-a-p/YuE2-3B` @ `1a96eca688d6ae5d7f0feb88573fec89920fcd19` in F32 on the CPU, the
  exact 357-token semantic prefix of the `supplied_full` request of `ar_real_weights.json`, and
  hashed codec ids: `multi_chunk_32` (100 frames, context cut to 48 frames per chunk: chunks of 48,
  48 and a 4-frame tail, the released 32 steps) and `single_chunk_5` (24 frames, released context,
  5 steps). The latents derive from CC BY-NC 4.0 weights, so the tensors stay outside the
  repository (`~/.cache/sceneworks-yue2-fixtures/nar/nar_real_reference.safetensors`); only the
  cases and the file's SHA-256 are committed, and the test refuses a reference with another hash.

### Measured tolerances

| Check | Measured | Bound |
| --- | --- | --- |
| Synthetic: every evaluation's input + velocity and the final latents, 8 song-level cases (F32) | max \|Δ\| 2.1e-6, rel L2 4.8e-7 | 2e-5 / 5e-6 |
| Synthetic: cached velocity vs upstream joint forward and cached velocity | max \|Δ\| 2.1e-6 | 2e-5 / 5e-6 |
| Synthetic: `nar_cond_end = 5` | max \|Δ\| 2.0e-6 | 2e-5 / 5e-6 |
| Synthetic: native query tiles / score budgets / offload vs default | measured 0 on macOS CPU, 7.2e-7 on Linux CI | ≤ 1e-5 |
| Timestep schedule (`logit(t)` as the model's F32) | exact | exact |
| Real weights: every evaluation's input + velocity and the final latents, 202 evaluations (F32) | max \|Δ\| 4.4e-5 (final 1.9e-5), rel L2 5.9e-6 | 5e-4 / 6e-5 |
| Real weights: `Rows(7)` + AR offload vs default | measured 0 on macOS CPU | ≤ 1e-5 |

Every assertion was checked against a mutation that must fail it (run one at a time); the smallest
latent movement among them was 7.3e-3 (midpoint time `t − dt` instead of `t − dt/2`):

| Mutation | Red tests (max \|Δ\| where a bound applies) |
| --- | --- |
| Euler (second evaluation at `x, t`) | synthetic parity (0.10), `cond_end` (0.82), tiling-vs-upstream (0.073) |
| Midpoint time `t − dt` | synthetic parity, `cond_end`, tiling-vs-upstream (7.3e-3) |
| No `MUSIC_END` in a chunk's AR sequence | synthetic parity (AR sequence), tiling-vs-upstream (0.44) |
| NAR RoPE positions off by one | joint (0.62), synthetic parity (0.53), `cond_end` (0.72), tiling |
| Attention drops the first visible key | joint (0.97), synthetic parity (0.36), `cond_end` (0.76), tiling |
| A query tile drops the last key | `every_query_tile_attends_every_key`, tiling (0.54) |
| Latent positions shifted by one | joint (4.0), synthetic parity (3.4), `cond_end` (3.2), tiling |
| `nar_cond_end` ignored | joint (2.2), `cond_end` (1.5) |
| `timestep_shift` ignored | synthetic parity `shift_3` (0.10) |
| Chunk noise rows misaligned | noise slicing, long-song composition, synthetic parity `multi_chunk` (6.1) |
| Offloaded AR path not restored | cancellation, tiling (AR path left offloaded) |
| No cancellation poll before the midpoint evaluation | cancellation (poll count) |
| Timestep features `cos`/`sin` swapped | joint (3.1), synthetic parity (2.3), `cond_end` (2.4), tiling |
| Cache not truncated between evaluations | 8 tests (capacity error / wrong keys) |
| No `logit` clamp | synthetic parity, `cond_end` (raw `t` schedule) |
| `ScoreBytes` ignores the budget (all rows) | `query_tile_rows_are_bounded_by_the_budget` |
| Budget divided by one score tile instead of `SCORE_TILES_LIVE` (3) | `query_tile_rows_are_bounded_by_the_budget` |
| `Rows(0)` rounded to one row | `query_tile_rows_…`, `invalid_requests_are_refused` |
| Too-small `ScoreBytes` rounded up to one row | `query_tile_rows_…`, `invalid_requests_are_refused` |
| Tile not validated before the prefill | `invalid_requests_are_refused` (poll count) |
| No `catch_unwind` around the solve | `a_panic_mid_solve_still_restores_the_ar_path` |
| Restore failure replaces the original error | `a_failed_restore_keeps_the_original_error` |
| Compute dtype left out of the stage identity | `stage_identity_covers_the_inputs_not_the_memory_controls` |
| Stage identity depends on the query tile | tiling (stage identity compared across settings) |
| A chunk exactly at `CONTEXT` refused (`>=`) | `chunks_at_the_released_context_limit` |
| No position-range check (one over runs) | `chunks_at_the_released_context_limit`, `invalid_requests_are_refused` |

**Where offload is measured.** On the CPU lanes (every test above, and the real-weight run) the
model already lives in host memory, so `offload_ar` only marks the AR path unavailable — the moves,
byte accounting and restore-to-device never execute there. `nar::tests::ar_offload_moves_exactly_the_ar_weights_on_a_gpu`
(`#[cfg(any(feature = "cuda", feature = "metal"))]`) covers them on a real device: exact bytes moved
(`embed_tokens + lm_head + Σ layer.ar`), every AR tensor on the host while offloaded with the NAR
twins and final norm left on the device, offload vs no-offload latents, and every AR tensor back on
the device after a completed and a cancelled synthesis. It runs on the manual CUDA lane
(`Candle CUDA packages (Windows, manual)`, `ci.yml` dispatched with `lanes=windows-cuda`).

**Context boundary.** `chunks_at_the_released_context_limit` runs the tiny model at the released
`CONTEXT` (24 576): a 12 282-frame song is one chunk whose AR + NAR positions are exactly 24 576 plus
a one-frame tail, and a chunk one position longer is refused (18–47 s, 2.1 GB RSS in a debug test
build on an M-series CPU).

Run cost (Apple M-series CPU, F32, measured 2026-09-26): the reference `real` subcommand peaks at
21.0 GB RSS (`ru_maxrss`: both MoT paths in F32 plus the mapped checkpoint) and takes ~80 s after
loading; the native `nar_real_weights` test (release) peaks at 23.2 GB `ru_maxrss` / 16.4 GB
footprint (the file-backed mapping counts toward `ru_maxrss` during the load) and takes 263 s: 22 s
to verify and load, 211 s for the three-chunk 32-step case, 14 s per run of the 5-step case. Never
run the two at once.
