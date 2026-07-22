# SC-13888 — candle CUDA `matmul` rejects non-contiguous operands (fix + validation)

Date: 2026-07-22

Owner: the Windows/CUDA self-hosted box (this bug is CUDA-only; it does not reproduce on CPU/Metal).

Relates to: sc-13324 (the real-weights validation that surfaced it — see
`SC_13324_AUDIO_CUDA_VALIDATION.md`), sc-13474 (MMAudio real-weights CI wiring), sc-13251 / sc-13756
(ACE-Step Cover), epic 12833.

## Symptom

candle's CUDA `matmul` (`cuda_backend/mod.rs::gemm_config`) only accepts operands whose last two
strides are either row-major (**C-contiguous**) or column-major (**F-contiguous** / a plain
transpose). A genuinely *strided* operand — one that is neither — makes it `bail!` with
`matmul is only supported for contiguous tensors`. CPU and Metal tolerate the strided operand, so
code that omits an explicit `.contiguous()` works there and only fails on CUDA.

In both affected sites the offending operand is the **left-hand side**: a strided slice of a larger
tensor (its row stride is the parent's full row span, not the slice width). The right-hand side is a
transposed `Linear`/projection weight, which is F-contiguous and which `gemm_config` already accepts
via `CUBLAS_OP_T` — so the RHS never needed touching; only the strided LHS did.

## Root cause, pinned to `gemm_config` (candle rev `1e6aa85e`)

For `A[m,k] @ B[k,n]`, `gemm_config` derives `transb`/`ldb` from the LHS's last two strides
`(lhs_m1, lhs_m2)`:

- accepted (C): `lhs_m1 == 1 && lhs_m2 == k`
- accepted (F/transpose): `lhs_m2 == 1 && lhs_m1 == m`
- otherwise → `MatMulNonContiguous` bail

The CLS-pool / query-token slices have `lhs_m1 == 1` but `lhs_m2 == (parent row span) != k`, so they
hit neither branch and bail. `.contiguous()` makes `lhs_m2 == k`, satisfying the C branch. It is a
no-op on already-contiguous tensors and semantically identical on every backend.

## The two sites (both the strided LHS)

| Provider(s) | File:line | Matmul | LHS (strided) |
| --- | --- | --- | --- |
| `mmaudio_small_16k` / `mmaudio_large_44k` | `candle-audio-mmaudio/src/clip.rs` `encode_image` | `pooled.matmul(visual.proj)` — DFN CLIP visual projection `1280→1024` | `hs.i((.., 0, ..))` CLS-token slice |
| `acestep_v15_turbo` **Cover** | `candle-audio-acestep/src/tokenizer.rs` `AttentionPooler::forward` → FSQ `project_in` | `Linear(2048→6)` (`x.matmul(w.t())`) in `ResidualFsq::forward` | `x.narrow(1,0,1).squeeze(1)` pooler query token |

Fix: `.contiguous()` on the strided LHS slice at each site (2 lines + explanatory comments).
acestep music / edit-repaint use the distilled turbo DiT and never hit the FSQ pooler, so they were
never affected; only the non-distilled sft Cover path is.

## Regression coverage

An always-on, device-independent unit test guards the ACE-Step site:
`candle-audio-acestep` `tokenizer::tests::pooler_query_token_is_contiguous_for_fsq_matmul` builds a
synthetic `AttentionPooler` (zero-init `VarMap`) and asserts `forward(...).is_contiguous()`.
Contiguity is a layout property independent of device/values, so it runs on the default CPU test
lane and flips **RED** if the `.contiguous()` is removed (mutation-verified). The MMAudio CLS-pool
site is not cheaply unit-testable — `DfnClipEncoder` has fixed ViT-H/14 dims (≈630 M params), too
heavy to synthesize — so it remains guarded by the real-weights CUDA conformance above; a fast
per-PR guard for it is tracked as the follow-up **sc-13932**.

## Reproduce → verify (real weights, this box)

Environment identical to `SC_13324_AUDIO_CUDA_VALIDATION.md`: Windows 11, RTX PRO 6000 Blackwell
(`sm_120`), CUDA 12.9, MSVC 14.44, `--features cuda`, `CUDA_COMPUTE_CAP=120`, candle `1e6aa85e`.
Snapshots staged locally and pointed at via each crate's `*_SNAPSHOT` env (epic 13657).

**Before (RED, unfixed tree, verbatim):**

- `acestep_cover_wav_conformance` → panics in `ResidualFsq::forward` (`tokenizer.rs:326`) via
  `candle_nn::linear::forward`:
  `matmul ... contiguous ... lstride [30,2048] stride [12288,1] rstride [2048,6] stride [1,2048] mnk (30,6,2048)`
- `mmaudio_video_to_audio_conformance` → panics in `DfnClipEncoder::encode_image` (`clip.rs:363`):
  `matmul ... contiguous ... lstride [8,1280] stride [934400,1] rstride [1280,1024] stride [1,1280] mnk (8,1024,1280)`

**After (GREEN, fixed tree):**

| Test (`--features cuda --ignored`) | Result | Evidence |
| --- | --- | --- |
| `acestep_cover_wav_conformance` | ✅ ok (17.6 s) | 6.00 s cover; chroma gate passes — per-direction matched A 0.7907 / B 0.6745 (floor 0.4), margins +0.1195 / +0.0426 (floor 0.03), aggregate margin +0.0811 |
| `mmaudio_video_to_audio_conformance` | ✅ ok | `check_video_to_audio: PASS` |
| `mmaudio_synced_foley_wav` | ✅ ok | real Foley WAV 1.504 s, peak 0.548, rms 0.0185 |
| `mmaudio_44k_video_to_audio_conformance` | ✅ ok | `check_video_to_audio (44k): PASS` |
| `mmaudio_44k_synced_foley_wav` | ✅ ok | real 44k Foley WAV 1.509 s, peak 0.403, rms 0.0180 |

## sc-13324 caveat now closed

sc-13324 warned that MMAudio failed at its *first* stage (CLIP), leaving Synchformer / MM-DiT / VAE /
BigVGAN **untested** on CUDA. With the CLIP fix, both `mmaudio_small_16k` and `mmaudio_large_44k` now
run the **entire** pipeline through the BigVGAN vocoder to a real WAV — so no further candle-CUDA op
gap exists in the MMAudio graph. The remaining CUDA audio gap is the separate `upsample_nearest1d`
one (sc-13886), untouched here.
