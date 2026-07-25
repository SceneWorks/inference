# Audio provider reference parity

What each Candle audio provider has been verified against upstream, how that verification was done,
and which defects it found. Read this **before** investigating an audio-quality report — several
hypotheses below were disproved at real cost and should not be re-litigated.

Investigation date: 2026-07-24. Stories: sc-14441, sc-14442, sc-14443, sc-14448.

---

## The method that worked

Three real defects were found in one session. All three were located the same way, and **none** were
found by black-box reasoning about our own code:

1. Reproduce the failure locally with real weights.
2. Fetch the **upstream reference implementation** (diffusers for ACE-Step, the HF repo's
   `modeling_*.py` / `processing_*.py` for MOSS models) and diff conventions line by line.
3. Where a diff is suspected, dump the reference's intermediate tensors to **safetensors** — candle
   reads them natively — and compare stage by stage until one cosine drops.
4. Fix, then re-verify the whole chain end to end.

Counter-evidence for the alternative approach: over the same session, black-box hypotheses produced
a string of confident-but-wrong candidates — a frame-splice in the TTSD un-shift, segmented codec
decode, timestep scaling, channel-layout swaps, a missing `time_embed_r`. Every one was disproved by
checking the reference or measuring. **Go to the reference first.**

### Practical notes for standing up a reference

- A Python env already exists at `~/sceneworks-pytorch-harness/.venv` (torch + transformers).
  **Do not modify it.** Install extra packages with `pip install --target <scratch-dir>` and put that
  dir on `PYTHONPATH`, so the harness stays untouched.
- Model code that ships in the HF repo (`trust_remote_code`) can be run against the **local snapshot**
  — symlink `model.safetensors` rather than copying multi-GB weights.
- Version pinning bites: MOSS-TTSD v0.5's custom `_sample` targets transformers 4.52 while its config
  needs ≥4.53's `layer_type_validation`. Workaround: skip `generate()` and drive the AR loop manually,
  calling the model's own `_generate_next_tokens_with_scores` / `_process_multi_channel_tokens`.
- Comparing waveforms: reference audio is often **planar** `[L…L R…R]` while our WAVs are
  **interleaved**. Comparing them naively yields ≈0 cosine and looks like catastrophic failure. This
  cost real time — de-interleave before comparing.

---

## Per-provider status

### ACE-Step (`acestep_v15_turbo`) — three defects found, all fixed (sc-14442)

Verified against `diffusers.pipelines.ace_step`. End-to-end waveform parity **L 0.999950 /
R 0.999971** given identical initial noise.

| Stage | Parity |
|---|---|
| Qwen3 text encoder | cosine 1.00000000 |
| Condition encoder | cosine 1.00000000 |
| Context latents (silence + mask) | cosine 1.00000000 |
| DiT (per-step velocity) | cosine 1.00000000 |
| 8-step trajectory | identical to 5–6 decimals |

Defects, all in **conditioning** — the DiT itself was always exact:

1. **Prompt was the wrong format.** The model is trained on the reference `SFT_GEN_PROMPT`
   structured markdown document (instruction / caption / metas + `<|endoftext|>`), not a
   comma-joined caption. 23 tokens vs the reference's 67. Absent metadata renders an explicit
   `N/A`; the lines are **not** omitted.
2. **Lyric stream dropped for instrumentals.** The reference always encodes
   `# Languages\n{lang}\n\n# Lyric\n{lyrics}<|endoftext|>` — ~11 conditioning rows even when the
   user supplies no lyrics. `vocal_language` rides the **lyric** prompt, not the text prompt.
3. **Timbre row — the source of the audible drone.** The reference encodes a fixed 30 s slice of
   VAE-encoded silence (`timbre_fix_frame = ceil(30 × latents_per_second)` = 750 frames) through the
   timbre encoder and CLS-pools row 0. The port fed the bare learned `special_token`, which the
   reference's `AceStepTimbreEncoder.forward` **never reads**. The diffusers source documents the
   symptom verbatim: an OOD timbre input *"produces drone-like audio (observed on all text2music
   outputs)"*.

**Disproved along the way** (do not re-investigate): VAE decode (synthetic round-trip cosine
0.9985), channel interleaving, `silence_latent` usage, flow shift (3.0 is correct), timestep scaling
(σ is correct — ×1000 makes it *worse*), CFG (turbo correctly bypasses it), step count, duration,
`time_embed_r` (already implemented correctly), and the `[context_latents, hidden]` concat order
(matches the reference).

### MOSS-SoundEffect (`moss_sfx_v2`) — one defect found, fixed (sc-14441)

The reference **always** denoises a fixed `max_inference_seconds` (30 s) latent window and crops the
decoded waveform; `max_inference_seconds` defaults to `None`. The port shortened the window to
`ceil(seconds)` as a speed optimization, putting the DiT ~10× out of distribution. CFG tripled the
error and an accurate solve converged onto a degenerate latent decoding to a −74 dBFS floor.

The failure was **step-count dependent**, which is what made it confusing: a coarse solve stepped
over the degeneracy. The conformance test hardcoded `STEPS = 30` and passed, while the model's own
`DEFAULT_STEPS = 100` — the reference-documented default — produced silence.

> **Lesson worth generalizing:** a test that pins a parameter never exercises the default. Real-weight
> tests should cover the value an unset request actually resolves to.

**Cost of the fix:** 10× the sequence length with quadratic attention. 159 s on Metal vs 35+ min
unfinished on CPU for a 3 s clip. The `audio-metal` bundle feature is effectively required.

### MOSS-TTSD (`moss_ttsd_v05`) — no defect; the port is faithful (sc-14448)

Reported symptoms (choppy dialogue, words cut between turns, rushed delivery) **reproduce in the
PyTorch reference itself**. Every stage verified exact:

| Stage | Parity |
|---|---|
| Prompt grid, chat template, `[S1]`→`<speaker1>`, system prompt | identical (source diff) |
| `_shift_inputs` delay-pattern shift | identical |
| AR loop: channel constraints, teacher forcing, drain | identical (source diff) |
| Sampler config | identical (v0.5 ships `do_samples: null` → flat config on all channels) |
| KV cache | cached vs stateless on real weights: **cosine 1.000000** |
| `unshift` frame drops | 0 leading, 0 mid-stream, 86/86 kept (**measured**) |
| Backbone logits, all 8 channels | **cosine 1.00000000**, identical argmaxes |
| XY_Tokenizer codec | cosine 1.0 vs torch (sc-13518) |

Decisive test: tokens sampled by the **reference** model+sampler, decoded through **our** codec,
sound equivalent to our own end-to-end output. The reference also truncates harder — 68 frames
(5.44 s) where our port reaches 86 (6.88 s) on the same three-turn script.

**Conclusion: v0.5 is the model ceiling.** Quality work means evaluating MOSS-TTSD **v1.0** (upstream
has moved; v0.5 is under `legacy/`). RNG parity with torch's multinomial is explicitly **not** worth
doing for quality — reference sampling produced no better audio.

### Others

`kokoro_82m` and `moss_tts_realtime` were reported working and were not re-verified this session.

---

## Cross-cutting: the consumer WAV writer (sc-14443)

SceneWorks' `sceneworks-worker::video_jobs::write_wav_pcm16` peak-normalized **every** clip
(`scale = i16::MAX / peak`) rather than clamping and scaling by `i16::MAX`. Two effects:

- Absolute loudness destroyed for every audio asset — everything forced to 0 dBFS.
- A failed near-silent render received enormous make-up gain. The collapsed SFX render (peak 0.0205)
  was amplified **+33.8 dB**, turning its residual noise floor into full-scale hiss and registering it
  as a finished asset.

This is why a *dead render* presented as *"random noise"* and cost significant diagnosis time. The
dead-render guard missed it too — it tested `peak == 0.0`, and the peak was 0.02.

**Diagnostic fingerprint:** a rendered asset whose peak is *exactly* 32767 has been peak-normalized.
Compare crest factors — if they match the unnormalized render, it is pure gain, not distortion.

Consumers should use the audited `candle_audio::wav::encode_wav_pcm16` rather than reimplementing it.

---

## Seeds

`gen_core::default_seed()` is **wall-clock nanoseconds**. A request with no seed is therefore not
reproducible run to run. This matters for stochastic providers: measured TTSD output RMS varied
0.033–0.101 across seeds on one script. When triaging a quality report, establish whether the seed
was pinned before attributing anything to a code change.
