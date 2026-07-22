# SC-13691 — Kokoro (iSTFT-Net vocoder) on macOS/Metal (real weights, GPU) validation

Date: 2026-07-22

Owner: the macOS Apple-Silicon box (this story is intrinsically owned by it — the Metal half of the
`upsample_nearest1d` gap can only be validated on a real Metal GPU).

## Purpose

`upsample_nearest1d` is unimplemented on **both** of candle-core's GPU backends at the pinned rev
`1e6aa85e`: a hard `bail!` on Metal (this story, sc-13691) and on CUDA (its sibling sc-13886). Two
audio providers hit it — Kokoro's iSTFT-Net vocoder (`AdainResBlk1d` ×2 time upsample) and
Chatterbox's S3Gen flow encoder — so both were stuck CPU-only on GPU platforms. The backend-agnostic
fix (express nearest ×k upsample as `unsqueeze`+`broadcast_as`+`reshape`) landed on `main` via
**PR #188** (sc-13886, merge commit `79a8e2fa`); its own note deferred the end-to-end **Metal**
validation to this story:

> "End-to-end Metal validation of Kokoro on this candle rev is tracked by sc-13691 (must be verified
> on the Mac)."

This document is that record — the Metal counterpart of `SC_13324_AUDIO_CUDA_VALIDATION.md`
(Recommendation #1 there: *"Verify on CUDA here and on Metal on the Mac."*), scoped to the two call
sites the fix touches.

## Environment

- Host: macOS 26.5.2 (Darwin 25.5.0, build 25F84), **Apple M5 Max**, 128 GB unified memory.
- Toolchain: rustc 1.96.0, candle rev `1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d`.
- Build: `--features metal` (forwards `candle-audio/metal` + `candle-nn/metal`).
- Device seam: `candle_audio::default_device()` returns `Device::new_metal(0)` under `--features
  metal` (`crates/audio/candle-audio/src/lib.rs`), and Kokoro loads onto it
  (`candle-audio-kokoro/src/model.rs:197`). There is **no CPU-fallback arm** — the `?` on
  `new_metal(0)` propagates — so a passing `--features metal` run executes genuinely on the Metal
  device. Proven directly by the mutation check below.
- Weights: real pinned `hexgrad/Kokoro-82M` snapshot (`f3ff3571…`, config.json + kokoro-v1_0.pth +
  voices/), staged locally and pointed at via `KOKORO_SNAPSHOT` (inference never self-fetches — epic
  13657).
- Code under test: `main` at the PR #188 merge; the fix files are byte-identical between the tested
  commit `c8d30b87` and the merge, so this validates exactly what shipped.
- Method: ran the existing `#[ignore]`d, env-gated real-weights tests with `--ignored --nocapture`.

## Result — `kokoro_82m` generates on Metal ✅

| Test (`--features metal`, real weights) | Verdict | Evidence |
| --- | --- | --- |
| `kokoro_conformance` (shared audio-generator suite) | ✅ pass | validate-honesty + typed cancellation + seed determinism, all green on Metal (51.9 s) |
| `kokoro_wav_conformance` (real-WAV DoD) | ✅ pass | 3.17 s @ 24 kHz mono; interior RMS **0.0510** (>0.01); peak-frame RMS **0.1168** (>0.05); frame-RMS CV **>0.25**; voiced autocorrelation **0.867 @ 316 Hz** (>0.4 — the af_heart reference); sub-4 kHz energy **>10×** supra-8 kHz; byte-identical re-synthesis. A 152 KB playable WAV was written. |

These are the same rigorous non-degeneracy/speech-shape gates the macOS-CPU runs assert — not a
"wrote some bytes" pass.

### Device-execution proof (mutation check)

To rule out any silent CPU execution, the fixed call site
(`candle-audio-kokoro/src/nn.rs`, `nearest_upsample1d(x, 2)`) was temporarily reverted to
`x.upsample_nearest1d(t * 2)` on the **same** `--features metal` test binary
(`target/debug/deps/conformance-16348ce35d84cbc8`):

    generate() failed on the cheap request: backend op failed: Metal upsample_nearest1d not implemented

`kokoro_conformance` went **RED** with candle's Metal-specific bail (exit 101), and green again once
restored. CPU *implements* `upsample_nearest1d`, so a CPU run would have passed the mutation — the
failure proves the tensors were on the Metal device and that `nearest_upsample1d` is precisely what
unblocks the vocoder there.

### CPU golden preserved (bit-identity)

The reroute is bit-identical to candle's own CPU `upsample_nearest1d` for an exact integer factor
(pure data movement, no arithmetic):

- `candle_audio::ops` unit tests: 3/3 pass — `matches_candle_upsample_nearest1d_bit_for_bit`
  (k ∈ {1,2,3,4}), per-frame repetition, per-channel independence.
- `kokoro_regression_fixture` on this **macos/aarch64** box (the fixture's canonical platform, so the
  exact-hash assertion *fires*, not the skip branch): PCM SHA-256
  `e358255a6058b849…831076a3` **matches** the committed fixture. Duration 3.175 s @ 24 kHz mono,
  −25.93 LUFS, −9.50 dBTP, 0 clipped. The fix changed which ops run, not the samples produced.

## Chatterbox — the fix's second call site on Metal (bonus)

The shared helper's other call site is Chatterbox's S3Gen flow encoder
(`candle-audio-chatterbox/src/flow_encoder.rs`, `nearest_upsample1d(x, UP_STRIDE)`). Running
`candle-audio-chatterbox` conformance `--features metal` on real weights (`ResembleAI/chatterbox` +
`SceneWorks/perth-implicit`): **10/11 pass**.

- `flow_synthesizes_a_sane_mel_from_speech_tokens` (the `UpsampleConformerEncoder` owning the fixed
  call site) — ✅ pass on Metal, confirming the second call site runs on-device.
- `hift_vocodes_a_real_mel_to_nonsilent_waveform` (ConvTranspose1d upsample — a *different* op that
  Metal implements) — ✅ 105,600 samples @ 24 kHz, peak 0.3482, RMS 0.0472.
- `t3`, `s3tokenizer`, `campplus`, `perth` component tests — ✅ pass on Metal.
- `chatterbox_clones_a_reference_voice_end_to_end` — ❌ fails, but **not** on the upsample op:
  `Backend(device mismatch in conv1d, lhs: Metal{…}, rhs: Metal{…})`. This is a distinct
  Metal device-plumbing bug (multiple independent `default_device()` → non-equal `new_metal(0)`
  instances), filed as **sc-13922** (relates sc-13501/13691/13886). Orthogonal to this story.

## Scope — no other Metal `upsample_nearest1d` sites

A tree-wide grep finds exactly the two call sites the fix touches (kokoro `nn.rs`, chatterbox
`flow_encoder.rs`). The story's "check chatterbox HiFT / other vocoders" resolves clean:

- Chatterbox HiFT upsamples via `ConvTranspose1d` (implemented on Metal; `hift` test green above).
- MMAudio's `nearest-exact` interpolation is a custom `u32` index-gather (`mmaudio/.../mmdit.rs`),
  not `upsample_nearest1d`, so it never hits the bail.

A full Metal sweep of the remaining audio providers (the Metal counterpart of the SC-13324 CUDA
matrix) is out of this story's Kokoro scope and belongs to enabling audio Metal on macOS (sc-13501).

## Parity vs the macOS-CPU baseline

Cross-device **exact** parity is neither attempted nor expected — the one committed numeric baseline
(`kokoro_regression_fixture`) pins an exact PCM hash to `macos/aarch64` **CPU** and runs only there
(candle's Metal conv/matmul kernels are not bit-identical to CPU). Metal parity is established by the
same statistical quality gates passing on Metal that the CPU runs assert, within expected cross-device
floating-point differences.

## Acceptance

Met: **Kokoro-82M generates a real, non-degenerate WAV on Metal, device confirmed Metal (mutation
proof), quality gates green.** The fix ships on `main` via PR #188; this record is sc-13691's
deliverable.

## Follow-ups

1. **sc-13922** — chatterbox full-clone `device mismatch in conv1d` on Metal (repeated
   `default_device()`/`new_metal(0)` instances). Also check whether CUDA shares it (sc-13886 verified
   only the upsample-gated components, not the full end-to-end clone).
2. **sc-13928** — no committed **Metal** regression envelope for Kokoro yet: Metal output is guarded
   only by per-run statistical conformance + within-run determinism, so future Metal drift is
   ungated. An exact hash isn't viable on Metal; a committed metric envelope (LUFS/dBTP/duration/
   frame-CV bands) would close it.
