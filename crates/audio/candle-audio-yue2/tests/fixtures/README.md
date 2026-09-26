# YuE2 autoregressive-stage fixtures (sc-22991)

Produced by `scripts/reference/yue2/ar_fixtures.py` from the **pinned upstream** code, imported
from the shared reference environment (`scripts/reference/yue2/README.md`):

| What | Pin |
| --- | --- |
| Upstream source | `github.com/multimodal-art-projection/YuE` @ `92a73cc7652fcc1f937855e4b765e0a0edd7ff2e` — `yue2.modeling_yue2`, `yue2.sampling`, `yue2.protocol`, `yue2.tokenization_yue2` |
| Python stack | Python 3.12.13, torch 2.10.0 (CPU), numpy 2.2.6, transformers 4.57.6, tiktoken 0.12.0 (each file's `reference` block) |
| Weights (`ar_real_weights.json` only) | `m-a-p/YuE2-3B` @ `1a96eca688d6ae5d7f0feb88573fec89920fcd19`, `model.safetensors` SHA-256 checked against its `weights_manifest.json` before loading; BF16 checkpoint upcast to F32 |

No weights are committed. The files hold token ids, logit summaries and SHA-256 digests.

## Files

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

## Deliberate substitutions in the reference

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

## Measured tolerances

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

Token sequences (greedy and injected-draw) and truncation flags are compared exactly; on real
weights all 258 steps' tokens and every step's top-8 ids matched.

Run cost (Apple M-series CPU, F32): the reference `real` subcommand peaks at 21.0 GB RSS
(`ru_maxrss`: both MoT paths in F32 plus the mapped checkpoint) and takes ~50 s; the native
`real_weight_decodes_match_upstream` peaks at 9.5 GB (AR path only) and takes ~210 s. Never run the
two at once.
