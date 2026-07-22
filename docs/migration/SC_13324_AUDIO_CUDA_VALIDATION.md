# SC-13324 — Audio providers on Windows/CUDA (real weights, GPU) validation

Date: 2026-07-22

Owner: the Windows/CUDA self-hosted box (this story is intrinsically owned by it; the 2026-07-20
hold was lifted by running here).

## Purpose

Epic 12833's audio lane was developed and real-weights-conformance-tested on the macOS box (Candle
CPU + MLX). The candle-native cross-platform decision (sc-12901) is premised on **one**
implementation running on CPU/CUDA/Metal, but the Windows/CUDA CI lane only does a **compile** check
— no audio provider had been proven to actually *generate* real output on a CUDA GPU. This validates
that gap.

## Environment

- Host: Windows 11, dual **NVIDIA RTX PRO 6000 Blackwell Max-Q** (`sm_120`, 97 GB each), CUDA 12.9.
- Toolchain: rustc 1.96.0, MSVC 19.44.35227 (14.44 — under CUDA 12.9 nvcc's 14.51 ceiling), cudarc
  0.19.8, candle rev `1e6aa85e`.
- Build: every crate built `--features cuda` with `CUDA_COMPUTE_CAP=120` and the VS2022 BuildTools
  `vcvars64` (per CLAUDE.md).
- Device seam: `candle_audio::default_device()` returns `Device::new_cuda(0)` under `--features
  cuda`, so tests that call it run genuinely on the CUDA device (not a CPU fallback).
- Weights: real pinned snapshots at the `release/real-weight-models.toml` revisions, staged locally
  and pointed at via each crate's `*_SNAPSHOT` env var (inference never self-fetches — epic 13657).
- Method: ran each shipped provider's existing `#[ignore]`d, env-gated real-weights conformance with
  `--ignored --nocapture --no-fail-fast`.

## Shipped provider set (re-scanned from `candle-audio-catalog` at execution time)

8 generators — `kokoro_82m`, `moss_sfx_v2`, `acestep_v15_turbo`, `moss_tts_realtime`,
`chatterbox_tts`, `mmaudio_small_16k`, `mmaudio_large_44k`, `moss_ttsd_v05` — plus `chatterbox_ve`
(voice embedder), `openvoice_v2` (audio transform), `whisper_base` (transcriber), `clap_htsat_unfused`
(audio embedder). 12 registered providers.

## Result matrix

| Provider (kind) | CUDA verdict | Evidence |
| --- | --- | --- |
| `moss_sfx_v2` (SFX gen) | ✅ **generates** | `moss_sfx_wav_conformance`: 4.00 s @ 48 kHz, RMS 0.108, frame-CV 2.105, peak-bin 30.5 %, 4/7 octave bands, flatness 0.003 — real broadband SFX |
| `acestep_v15_turbo` (music/edit) | ✅ **generates** | `acestep_music_wav_conformance`: 12.00 s @ 48 kHz stereo, RMS 0.140, frame-CV 0.144, beat-autocorr 0.324, 6/7 bands. `acestep_edit_repaint_wav_conformance`: region [4,8]s edited rel-L2 1.014 / corr 0.201, untouched corr 1.000 |
| `acestep_v15_turbo` **Cover** mode | ❌ **fails** — sc-13888 | non-contiguous `matmul` in the sft Cover DiT (`[30,2048]×[2048,6]`). Music/edit (distilled turbo DiT) pass; only Cover's sft DiT path hits it |
| `moss_ttsd_v05` (dialogue TTS) | ✅ **generates** | 4/4 render tests: valid 8-codebook delay-pattern frames; 2-speaker script shapes the stream; multi-speaker 5.12 s (cross-speaker cos 0.495 < self-sim 0.749 → distinct); non-English zh 5.20 s |
| `moss_tts_realtime` (streaming TTS) | ✅ **generates** | 5/5: RVQ frames; incremental first-frame 73.6 ms; streaming gate real 1.60 s WAV @ 24 kHz (CV 2.193); **ASR round-trip mean CER 0.095**; codec decode |
| `clap_htsat_unfused` (audio embed) | ✅ **embeds** | `embedding_is_deterministic` (Kokoro-independent, tone/noise) passes on GPU — the test's internal asserts (512-d, unit-norm, determinism) hold; log line is a bare `ok`. `cross_modal` blocked (uses a Kokoro clip — sc-13886) |
| `chatterbox_tts` — PerTh component | ✅ **runs** | `perth_watermark_roundtrips_and_is_imperceptible`: 32 kHz SNR 15.96 dB (det 1.000 / clean 0.008); 24 kHz SNR 15.21 dB (det 0.998 / clean 0.000) |
| `kokoro_82m` (TTS) | ❌ **fails** — sc-13886 | `upsample-nearest1d is not supported on cuda` (iSTFTNet vocoder, `kokoro/nn.rs:241`) |
| `chatterbox_tts` (full clone) | ⛔ **blocked / fails** — sc-13886 | 10/11 tests die at the Kokoro reference synth (`kokoro generate: upsample-nearest1d`) before the provider runs; its own `flow_encoder.rs:277` uses the same op (verified statically, not reached at runtime). Only the Kokoro-independent PerTh component test passes |
| `mmaudio_small_16k` (video→audio) | ❌ **fails** — sc-13888 | non-contiguous `matmul` in DFN CLIP encode (`mmaudio/clip.rs:363`), before the MM-DiT |
| `mmaudio_large_44k` (video→audio) | ❌ **fails** — sc-13888 | same CLIP gap (shared encoder) |
| `chatterbox_ve` (voice embed) | ⛔ **blocked** — sc-13886 | harness synthesizes a Kokoro reference → dies at `kokoro generate: upsample-nearest1d`; own model unverified on CUDA |
| `openvoice_v2` (voice conversion) | ⛔ **blocked** — sc-13886 | harness converts a Kokoro source clip → same block; own model unverified |
| `whisper_base` (ASR) | ⛔ **blocked** — sc-13886 | both tests transcribe a Kokoro-synthesized clip → same block; own model unverified |

**Generate real output on CUDA: 5 full providers** — moss_sfx_v2, acestep_v15_turbo (music/edit),
moss_ttsd_v05, moss_tts_realtime, clap_htsat_unfused. The chatterbox **PerTh component** also runs on
CUDA, but the chatterbox_tts provider *as a whole* fails. **Own-model CUDA failures: kokoro_82m and
both mmaudio providers** (directly observed op errors) **+ acestep Cover mode**. **Blocked before the
provider-under-test runs** (harness synthesizes a Kokoro clip that hits sc-13886): chatterbox_tts,
chatterbox_ve, openvoice_v2, whisper_base — their own models are **unverified** on CUDA. (chatterbox_tts
*would* also hit the same op in its own `flow_encoder.rs:277`, verified statically, but the test never
reaches it — it dies at the Kokoro reference synth first.)

## Root cause — two candle-CUDA-backend op gaps explain every failure

**A. `upsample_nearest1d` unsupported on the candle CUDA backend (sc-13886).** Two call sites, both
our code, both exact integer-factor nearest upsample: `candle-audio-kokoro/src/nn.rs:241` and
`candle-audio-chatterbox/src/flow_encoder.rs:277`. Fails `kokoro_82m` and `chatterbox_tts` on their
own models, and — because the TTS/voice conformance harnesses synthesize live Kokoro reference/
round-trip clips — **transitively blocks** `chatterbox_ve`, `openvoice_v2`, and `whisper_base`. This
is the CUDA sibling of the already-filed **Metal** gap sc-13691 (same op, same vocoders). The fix
(express nearest ×k as `unsqueeze`+`broadcast`+`reshape`) is backend-agnostic and bit-identical on
CPU, so it resolves both backends at once and preserves the macOS regression hash.

**B. candle CUDA `matmul` rejects non-contiguous operands (sc-13888).** A transposed/strided operand
that CPU (and Metal) tolerate makes CUDA `matmul` `bail!`. Fails `mmaudio_small_16k` /
`mmaudio_large_44k` (DFN CLIP encode, `mmaudio/clip.rs:363`) and `acestep_v15_turbo` **Cover** (sft
DiT). Fix: `.contiguous()` on the operand before the matmul. NB: MMAudio fails at its *first* stage
(CLIP), so its deeper stages (Synchformer / MM-DiT / VAE / BigVGAN) are still **untested** on CUDA
and may hide further gaps (MMAudio real-weights CI wiring is separately tracked by sc-13474).

## Parity vs the macOS-CPU baseline

Cross-device **exact** parity is not attempted and is not expected: the one committed numeric
baseline, `kokoro_regression_fixture`, pins an exact PCM hash to `macos/aarch64` and intentionally
runs only on that canonical platform. Parity here is established by **the same quality gates passing
on CUDA** that the macOS-CPU runs assert — non-degeneracy (RMS/frame-CV/spectral/beat), content
fidelity (ASR CER, voice-distinctness cosine margins, watermark SNR/detection), and within-device
byte-level determinism (re-synthesis is byte-identical for the passing generators). Every provider
that generates on CUDA does so within those envelopes; no divergence beyond expected cross-device
floating-point differences was observed for the passing set.

## Not-CUDA failures (recorded for honesty; excluded from the verdict above)

- `moss_sfx_conformance` / `acestep_conformance` fail a **device-independent** assertion — the shared
  *image* conformance suite's `64×64 above max_size 0` oversize check applied to a skip-size audio
  model (the pre-existing testkit issue sc-13705). These fail identically on macOS-CPU; the audio
  real-artifact tests (`*_wav_conformance`) are the CUDA generation gate and pass.
- `moss-tts` `codec_decode_frames_from_file` needs the optional `MOSS_TTSD_CODES_IN` env (a debug
  helper fed an external codes file), not a real-weights conformance.
- `moss_ttsd_v05`'s passing renders print `skipping: unsupported storage type BoolStorage` (8×/render)
  — candle's upstream `.pth`/`.ckpt` pickle loader skipping bool buffer tensors at *load* time. It is
  device-independent (identical on CPU), benign, and the render still produces real, speaker-distinct
  audio; noted only so the log line is not mistaken for a CUDA fault.

## Recommendations

1. Fix sc-13886 (nearest-upsample via reshape/broadcast at the two call sites) — one small,
   bit-identical-on-CPU change unblocks `kokoro_82m`, `chatterbox_tts`, and the CUDA validation of
   `chatterbox_ve` / `openvoice_v2` / `whisper_base`, **and** closes the Metal gap sc-13691. Verify
   on CUDA here and on Metal on the Mac.
2. Fix sc-13888 (`.contiguous()` before the affected matmuls) — unblocks `mmaudio` past CLIP and
   `acestep` Cover; then re-run to discover any deeper MMAudio CUDA gaps.
3. Consider decoupling the TTS/voice conformance harnesses from live Kokoro synthesis (a cached
   reference clip, or a non-Kokoro reference), so a single provider's op gap does not transitively
   block CUDA validation of unrelated providers.
