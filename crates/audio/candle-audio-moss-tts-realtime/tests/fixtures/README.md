# MOSS-Audio-Tokenizer encode reference-parity fixture (sc-17270)

`moss_audio_codec_encode_roundtrip_and_reference` in `../conformance.rs` cross-checks the native
`MossAudioCodec::encode` port against the **upstream PyTorch** `codec.encode`. That comparison
needs a clip both encoders see byte-for-byte identically, plus the codes the reference emits for
it. Before sc-17270 those came from `MOSS_CODEC_CLIP` / `MOSS_CODEC_REF_CODES`, which nothing in
this repository ever set — so the arm took an `else` branch, printed `SKIPPED`, and the test still
reported `1 passed`. The real-weight lane's run-count assertion cannot see that, so the strongest
claim the port makes was not regression-protected at all.

These three files close that hole. They are committed, the test defaults to them, and the two
environment variables now only *override* the comparison — they can no longer switch it off.

| File | What it is |
| --- | --- |
| `moss_codec_ref_clip.f32` | The input clip: raw **f32 little-endian, mono, 24 kHz**, 192 000 samples (8.000 s = 100 frames of `downsample_rate` 1920). 750 KiB. |
| `moss_codec_ref_codes.csv` | The reference codes, `frames[100][16]` — one comma-separated row per frame, values in `[0, 1024)`. |
| `moss_codec_ref_metadata.json` | Provenance: upstream repository/revision/licence, the SHA-256 of each fetched reference source, the SHA-256 of the two files above, `snapshot_inventory_sha256` (a digest of every file in the weight snapshot that produced the codes), and the runtime they were produced on. |

`snapshot_inventory_sha256` is the part that makes the provenance a claim about *bytes*. A revision
string is only a label: a snapshot materialized at some other revision has the same architecture,
so it loads with no missing keys and its codes would be committed stamped with a pin they never
came from. The generator runs `verify_model_snapshot.snapshot_inventory`, which enforces the
revision and `expected_files` and then digests the whole snapshot, so that cannot happen quietly.

## Licence

There is **no third-party audio here.** The clip is synthesized from scratch by
`scripts/reference/moss_audio_codec_reference.py` — a fixed "sentence" of harmonic stacks under a
three-formant envelope, alternating with high-passed LCG noise, all from stdlib arithmetic on a
fixed seed. Nothing is sampled, recorded, or derived from a corpus, so no clip licence applies.

The reference implementation (`modeling_moss_audio_tokenizer.py` and friends, Apache-2.0) is
**not vendored**. The generator fetches it at regeneration time from the pinned revision and
asserts each file's SHA-256, so a silent upstream edit is caught. Only the *numbers* it produces
are committed.

## Regenerating

```bash
export MOSS_AUDIO_TOKENIZER_SNAPSHOT=/path/to/MOSS-Audio-Tokenizer/snapshots/<revision>
python3 scripts/reference/moss_audio_codec_reference.py synth-clip   # stdlib only, ~1 s
python3 scripts/reference/moss_audio_codec_reference.py encode       # needs torch + the weights
```

Both default to the paths above. The `encode` step refuses to run unless three things hold: its
own pin matches `release/real-weight-models.toml`; the snapshot on disk actually *is* that
revision, with every `expected_files` entry present (`verify_model_snapshot.verify_snapshot`, not
a string comparison); and the checkpoint loads with no missing or mismatched keys — a partly
random-init reference would otherwise be committed as ground truth.

**Regenerate whenever the pinned `moss-audio-tokenizer` revision moves.** Ordinary CI
(`scripts/tests/test_moss_audio_codec_reference.py`, no weights needed) fails if the manifest pin,
the generator's pin, and the metadata's pin drift apart, so a bump cannot land with a stale
fixture.

## Why these numbers

- **8.000 s / 100 frames.** An exact multiple of `downsample_rate`, so the reference's ceil-padded
  frame count and its reported valid length coincide and no trailing part-padded frame is at
  stake. It also keeps the first analysis stage at 192000 / 240 = 800 positions, inside its
  causal context of ⌊100 Hz × 10 s⌋ = 1000 — so `encode`'s auto path resolves to **single-shot**.
  A clip past 10 s would silently switch the arm to the chunked path instead.
- **16 codebooks.** `config.rvq` on the MOSS-TTS-Realtime side: the AR brain emits 16 per frame,
  so the port loads 16 quantizers and the reference is asked for the same prefix.
- **float32 on CPU.** The port mmaps these weights as `DType::F32` and the checkpoint is f32 on
  disk, so a CPU f32 reference run is like-for-like rather than a reduced-precision accelerator
  comparison.
- **Measured agreement is 1.000** on every one of the 16 codebooks. The test bounds codebook 0 at
  0.99, the pooled rate at 0.98, and — separately — the *worst single* codebook at 0.95. The third
  bound is not redundant: pooled over 1600 comparisons, 0.98 tolerates 32 mismatches, so with
  codebook 0 separately bounded a regression confined to one deep quantizer could be wrong on 32
  of 100 frames and still pass. Verified: perturbing 8 frames of quantizer 7 yields cb0 1.000 and
  pooled 0.995 — clearing both original bounds — and is caught only by the per-codebook bound.

## What is gated where

| Check | Where it runs | Needs weights |
| --- | --- | --- |
| Port vs reference codes (this fixture) | `real-weights.yml`, weekly | yes (~7.1 GB) |
| Clip regenerates to the committed bytes; codes shape/range; pin agreement across manifest, generator and metadata; the conformance arm still defaults to the fixture and has no skip branch; `*.f32` declared binary | ordinary CI, every PR | no |

The second row matters as much as the first. The fixture files being healthy would not catch a
future edit that reintroduced a skip in `conformance.rs` — ordinary CI would stay green and the
weekly lane's `test result: ok. 1 passed` grep cannot see a branch that was not taken, which is
this story's own bug one level up. `scripts/tests/test_moss_audio_codec_reference.py` asserts the
consumer, not just the fixture.
