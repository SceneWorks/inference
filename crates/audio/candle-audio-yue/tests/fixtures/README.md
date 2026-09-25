# YuE stage-2 parity fixtures (sc-19381)

Both files are produced by `scripts/reference/yue_stage2_reference.py`, which runs **upstream
YuE-v1's own** `stage2_generate` / `stage2_inference` / `BlockTokenRangeProcessor` — lifted
verbatim out of the pinned `inference/infer.py` (commit `6d4f0b1f`, every upstream file SHA-256
pinned) and executed against the upstream `CodecManipulator` and mm tokenizer. Nothing upstream is
vendored and no weights are committed.

| File | Consumer | What |
| --- | --- | --- |
| `stage2_mock_reference.json` (33 KB) | `src/stage2/tests.rs` (every lane) | The upstream loop driven by a deterministic integer mock model — the chunk schedule, batch groups, ragged tail, sub-chunk track and `fix_output` repair. |
| `stage2_real_reference.json` (69 KB) | `tests/stage2_real_weights.rs` (`#[ignore]`, `YUE_S2_SNAPSHOT`) | The upstream loop on the real `m-a-p/YuE-s2-1B-general` (rev `9dfa90b7`), CPU. |

## Mock fixture

`MockModel` (Python) and `MockLm` (Rust) are twins: row scores are a keyed permutation of
`[0, 4194301)` plus `2·P` on codes `< 16` of the codebook after the last token's, skipped when
`key % 8 == 0`. All integer, all exact in f32, no ties. It makes about a quarter of the residuals
land in the wrong codebook, so upstream's `fix_output` rewrites codes in every case:

| Case | Frames | `batch_size` | Schedule | Repaired |
| --- | --- | --- | --- | --- |
| `groups_partial_and_tail` | 937 | 2 | [300, 300] + [300] + tail 37 | 2039 |
| `one_chunk_and_tail` | 337 | 4 | [300] + tail 37 | 680 |
| `tail_only` | 45 | 4 | tail 45 | 114 |

Codebooks are stored as 3 hex digits per code (8 rows per case). Codebook-0 inputs are the formula
`(t·37 + (t·t) mod 101 + seed) mod 1024`, not stored.

## Real-weight fixture

- **Inputs**: codebook 0 of the upstream xcodec encode (`SoundStream`, `target_bw=0.5`, exactly
  infer.py's ICL path) of a 13 s synthetic, arithmetic 16 kHz clip (a bass tone, a plucked
  arpeggio and a vibrato lead — no third-party audio), 650 frames. Three cases, with the
  `stage2_generate` calls upstream made (recorded in the fixture):

  | Case | Frames | Upstream calls | Covers |
  | --- | --- | --- | --- |
  | `encode_0_40` | 40 | tail 40 | sub-chunk track (patched `num_batch == 0` path) |
  | `encode_17_33` | 16 | tail 16 | sub-chunk track |
  | `encode_0_650` | 650 | **one 2-row group** of 300 (`batch_size` 2) + tail 50 | upstream's **unpatched** batched branch; full per-chunk context (positions to 2702, the whole static-cache capacity) |

  The 650-frame golden takes ~75 min on an M-series CPU: upstream re-prefills the whole context
  every frame.
- **Compute dtype: float32.** The checkpoint is bf16; upstream loads it as bf16 and computes in
  bf16. candle-llm computes in f32 on the CPU (bf16 weights upcast), so the golden is the same
  bf16 weights upcast in torch (`--compute-dtype float32`). The Rust CPU run reproduces it exactly
  — all 8 codebooks, all three cases (2026-09-24, torch 2.14.0, transformers 5.17.0).
- **Why not bf16 compute**: re-running the reference itself in bf16 (upstream's setting) changed
  166 of 320 codes of a 40-frame case and 21 of 128 of a 16-frame case against its own f32 run
  (measured on the fixture's first revision, cut from a 1 s clip).
  bf16 logits tie or near-tie often (8 significant bits across a 7168-wide slice), the first flip
  changes the teacher-forced context, and every later pick can follow. bf16 output is therefore a
  numerics-dependent realisation, not a golden — which is the divergence the Metal (bf16) tests
  characterise instead of asserting equality (`stage2::tests::metal`,
  `stage2_metal_bf16_divergence_is_characterised_on_real_weights`).
- **Measured tier divergence** (`stage2_every_tier_loads_through_production_and_is_characterised`,
  CPU, 2026-09-25, teacher-forced on the reference stream over the 40-frame case): bf16 through the
  production loader 0 of 280 picks differ; q8 4 of 280 (max |Δlogit| 1.26 at logit scale 30.1); q4
  26 of 280 (max |Δlogit| 7.1). Free-running, the first flip cascades: q8 keeps 121/280 and
  103/112, q4 77/280 and 14/112 residual codes of the reference (40- / 16-frame cases). Peak RSS
  19.6 GB.

## Metal (bf16) divergence — measured (AC3)

Run 2026-09-25 on the dev Mac (Apple Silicon, `--features metal`, release) with
`sc-19381-metal-run.sh`: Metal computes in bf16, the reference is the CPU in f32, both
teacher-forced along the CPU's own stream (so a flip cannot cascade). Flips = residual picks whose
argmax differs; relative Δ = `max |Δlogit| / max |logit|` over the 7168-wide slice. Deterministic:
a second run reproduced every number.

| Test / case | Frames | Flips | Flip fraction | max \|Δlogit\| | Logit scale | Relative Δ | Flip margins (logits) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic 2-layer (`stage2::tests::metal`) | 24 | 17 / 168 | 0.101 | 0.089 | 12.9 | 0.0069 | 0.001–0.046 |
| real 1B `encode_0_40` | 40 | 9 / 280 | 0.032 | 0.539 | 30.1 | 0.0179 | 0.004–0.087 |
| real 1B `encode_17_33` | 16 | 2 / 112 | 0.018 | 2.279 | 25.2 | 0.0905 | 0.010, 0.082 |
| real 1B `encode_0_650`, first chunk | 300 | 13 / 2100 | 0.0062 | 0.345 | 40.7 | 0.0085 | 0.004–0.166 |

Every flip is a near-tie (margin ≤ 0.17 logits at a logit scale of 25–41). The real-weight
CPU stream still equals the golden in every case. Tolerances, ~2× the measured maximum:

| Constant | Test | Value | Measured max |
| --- | --- | --- | --- |
| `MAX_REL_LOGIT_DELTA` | synthetic | 0.014 | 0.0069 |
| `MAX_FLIP_FRACTION` | synthetic | 0.20 | 0.101 |
| `METAL_MAX_REL_LOGIT_DELTA` | real 1B (per case) | 0.18 | 0.0905 |
| `METAL_MAX_FLIP_FRACTION` | real 1B (per case) | 0.065 | 0.032 |

Mutation: scaling the GPU logits ×1.5 inside `CandleStage2Lm::step` fails the real-weight test on
its first case (relative Δ 0.506 > 0.18). The flip count does not change under that mutation,
because scaling preserves the argmax; the relative-Δ bound is what catches it. Cost of the run:
~12.5 min wall (synthetic 9 s, real 1B 11.5 min), peak RSS 16.7 GB.

## Two upstream defects the producer works around (and the port handles)

1. **Sub-chunk tracks crash upstream.** With fewer than 300 frames, `stage2_inference` still calls
   `stage2_generate(prompt[:, :0], batch_size=0)`, whose `offset_tok_ids` takes `max()` of an
   empty array and raises. The producer patches only that call to return an empty id row (zero full
   chunks → zero output; the tail call right after handles the frames); the Rust port does the
   same by construction (`stage2::chunk_plan`).
2. **The open vocab tail aborts upstream.** The upstream processors block `[0, 46358)` and
   `[53526, 83738)` (the *tokenizer's* vocab size), leaving the checkpoint's untrained
   `[83738, 83840)` rows open, but `ids2npy` asserts every id is `< 57622`. The port argmaxes over
   the slice `[46358, 53525]` only — identical wherever upstream returns output. The mock scores
   that tail zero so upstream never trips on it.

## Regenerating

In the epic's YuE-v1 reference environment (torch 2.14.0 CPU, transformers 5.17.0, plus the
upstream `requirements.txt` set):

```text
export YUE_REF_INFERENCE_DIR=<YuE-v1 clone>/inference      # with xcodec_mini_infer beside it
export YUE_S2_BF16_SNAPSHOT=<yue-s2-1b-general-candle>/bf16
python scripts/reference/yue_stage2_reference.py mock
python scripts/reference/yue_stage2_reference.py real --compute-dtype float32
```

`real` takes about 80 minutes on an M-series CPU (the 650-frame case) and peaked at 12.5 GB RSS.
