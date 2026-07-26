# Stable Audio 3 small-SFX provider (`sc-14544`)

This checkpoint registers the second post-trained Stable Audio 3 small checkpoint as
`stable_audio_3_small_sfx` in every shipped audio bundle, beside the
`stable_audio_3_small_music` provider landed by `sc-14543`. It is a
checkpoint swap plus a second registration, not a port: the two shipped
`model_config.json` files are architecturally identical.

- upstream `Stability-AI/stable-audio-3` commit
  `124e8a799f57a1f665495ecb72e547d0a62867f1`;
- `stabilityai/stable-audio-3-small-sfx` revision
  `ae12755283df9d62ca39a9b050a39a0b607b8c20`;
- the snapshot-local encoder-only T5Gemma weights, config, tokenizer JSON, and
  tokenizer model.

## What actually differs from small-music

A full recursive diff of the two shipped configs finds nine differing leaf paths
in four groups, and only the first group is read by the inference path:

| Config path | small-music | small-sfx |
|---|---|---|
| `model.conditioning.configs[0].config.repo_id` | `stabilityai/stable-audio-3-small-music` | `stabilityai/stable-audio-3-small-sfx` |
| `training.arc.discriminator.freeze_backbone` | `false` | `true` |
| `training.arc.discriminator_base_ckpt` | training-only path | training-only path |
| `training.demo.demo_cond[0..3].prompt` | music prompts (9–119 s) | SFX prompts (9–35 s) |
| `training.demo.demo_cond[0..1].seconds_total` | 119, 100 | 10, 10 |

Only `repo_id` is consumed at inference time; the ARC discriminator fields and
the demo prompts are training/demo-only. The demo prompts are still useful — the
committed provider and conformance gates each drive a checkpoint with a real
shipped `demo_cond` prompt from its own config.

Everything the loader consumes is identical: `embed_dim` 1,024, `depth` 20,
`num_heads` 16, non-differential attention, `sample_size` 5,292,032 (120.0 s),
the same SAME-S pretransform, the same conditioning shape, the same
`rf_denoiser` objective, and the same 685-tensor / 438-DiT-key inventory. The
bundled `t5gemma-b-b-ul2` config, weights, and both tokenizer files are
byte-identical between the two snapshots. The root `model.safetensors` files are
both exactly 2,270,384,940 bytes and differ only in payload.

## Variant-bound loading

`gen_core::ModelRegistration::load` receives a `LoadSpec` and no provider id, so
a single shared loader registered under two ids could serve music weights from
`stable_audio_3_small_sfx` and nothing would notice. Loading is therefore
variant-bound: both registrations call
`model::load_variant(Variant, &LoadSpec)`, and the expected variant comes from
the registration site.

Two independent checks run before any tensor is materialized:

1. `pipeline::validate_small_layout` compares the snapshot's conditioner
   `repo_id` against the expected variant's repository. This is the only
   identifying field in the shipped configs.
2. `verify_snapshot_identity` authenticates the byte length and SHA-256 of every
   consumed root/T5 config, weight, and tokenizer file against the pinned
   revision. The safetensors header carries no identity metadata, so for the two
   equal-sized root checkpoints this hash is the only discriminator.

| Pinned file | Bytes | SHA-256 |
|---|---:|---|
| `model_config.json` | 10,454 | `a8aa5d45ae3d6524d3cd4e85e0d6e7d8d401267e7c6f28214bca8aae7b77bdeb` |
| `model.safetensors` | 2,270,384,940 | `ed9cf1b6172f1a8c2921a9560c21109ff3239524563ced9dce6dcdef41e2f515` |
| `t5gemma-b-b-ul2/config.json` | 2,540 | `575334409716886ac2952f5a275ed92868deef8a0ea560258d9970a431c6fb3a` |
| `t5gemma-b-b-ul2/model.safetensors` | 1,183,022,944 | `9b05ea5a4f211d023832f706fb2c0e83e4fc721b6da35ab69ceb0b55eb7800d3` |
| `t5gemma-b-b-ul2/tokenizer.json` | 34,362,429 | `7794135caa3ea73918949c902a781cc61dab674a4b59c17d85931c77c1114cbd` |
| `t5gemma-b-b-ul2/tokenizer.model` | 4,241,003 | `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2` |

## Registered contract

The SFX descriptor mirrors the proven small-music surface: family
`stable_audio_3`, backend `candle`, 44.1 kHz stereo, a 120-second maximum,
negative prompts, guidance, the four native samplers (`pingpong`, `euler`,
`rk4`, `dpmpp`), and the three mapped guidance methods (`cfg`, `apg`,
`cfg_rescale`). Both post-trained objectives share the same batch-CFG/APG/rescale
math, so SFX is deliberately **not** distinguished from music by a false
`supports_guidance=false` / `supports_negative_prompt=false` flag. The defaults
are 120 seconds, eight steps, guidance 1, Pingpong. Exact output length is
`floor(seconds × 44100)` frames.

Request validation, cancellation points, progress events, and the request-local
RNG draw order are the shared small-variant behaviour documented in
[`SC_14543_SMALL_MUSIC_PROVIDER.md`](SC_14543_SMALL_MUSIC_PROVIDER.md).

## Relationship to the shipped `moss_sfx_v2` provider

Both generate sound effects; they are not interchangeable and the descriptor
contract cannot machine-encode the difference today.

| | `stable_audio_3_small_sfx` | `moss_sfx_v2` |
|---|---|---|
| Sample rate | 44,100 Hz | 48,000 Hz |
| Channels | 2 (stereo) | 1 (mono) |
| Max duration | 120 s | 30 s |
| Weight license | Stability AI Community License | see `release/model-weight-licenses.json` |

If a product needs structured domain / channel / quality selection rather than
two documented ids, that is an additive descriptor-contract change, filed as
**sc-15041** — `family` is not overloaded here.

## Acceptance evidence

The `sc-14534` frozen PyTorch artifacts contain both checkpoints run from
identical inputs (seed 14534, same prompt, same noise, same sigmas, same
tokenizer ids and attention mask). Reading the committed
`sa3-reference/small-music-reference.safetensors` and
`sa3-reference/small-sfx-reference.safetensors` gives the reference divergence
envelope that the runtime gates are derived from, rather than an invented
threshold:

| Frozen Torch boundary | Music vs SFX |
|---|---:|
| shared `dit_noise` / `sampler_initial_noise` / `t5_last_hidden_state` | byte-identical |
| single-step `dit_prediction` cosine | 0.601294 |
| eight-step `sampler_final` cosine | 0.018599 |
| eight-step `sampler_final` normalized RMS delta | 1.002213 |

`tests/variant_divergence.rs` re-derives all three at test time (so oracle drift
breaks the gate's premise rather than silently weakening it), reproduces the
single-step DiT cosine through the Candle DiT within ±0.02, and then requires the
two registered providers — same prompt, seed, duration, steps, sampler, and
therefore the same request-local noise stream — to produce different PCM hashes,
a waveform cosine at or below 0.15, and a normalized RMS delta at or above 0.9,
**at every seed in the sweep**.

Measured on this branch, Metal, release, both pinned snapshots present:

| Runtime gate | seed 14544 | seed 7 | seed 2026 | Threshold |
|---|---:|---:|---:|---|
| 2 s / 8-step waveform cosine | 0.060349 | 0.062972 | 0.062954 | ≤ 0.15 |
| 2 s / 8-step normalized RMS delta | 1.290740 | 1.390828 | 1.439228 | ≥ 0.9 |
| music / SFX PCM SHA-256 | `159fdc89…` / `5cdfc118…` | differ | differ | must differ |

| Runtime gate | Measured | Threshold |
|---|---:|---|
| Candle single-step DiT music-vs-SFX cosine | 0.601294 | frozen Torch 0.601294 ± 0.02 |
| shared-weight null (music vs itself, same seed) | cosine 1.000000, delta 0.000000 | must violate both thresholds |
| partial-mis-wiring control (music blended toward SFX) | cosine 0.250000, delta 1.224745 | cosine > 0.15 and < the shipped 0.35 |

The thresholds are one-sided: they detect *agreement*, so a shared weight path
(cosine 1, delta 0) is rejected, and the self-comparison control proves the two
metrics actually register that. They cannot certify that the divergence is the
*right* divergence — the frozen-Torch `dit_prediction` reproduction does that.
The seed spread is 0.0026 in cosine and 0.15 in delta, so the committed
thresholds sit 2.4x and 1.4x outside the measured envelope.

What tightening from 0.35 / 0.5 buys is the middle of the range, and that is now
demonstrated rather than argued. The self-comparison null is byte-identical by
construction, so it lands at exactly cosine 1.0 and can only prove the metrics
register *total* agreement — it never exercises the 0.15 … 1.0 band. The
partial-mis-wiring control fills that gap: the music take Gram-Schmidt blended
toward the SFX take at exactly cosine 0.25, which is the signature of two
registrations sharing a conditioner or a subset of DiT blocks. It is asserted to
be rejected by the committed 0.15 *and* admitted by the shipped 0.35, so the
tightening has committed gate strength behind it rather than a narrative. A naive
mix overshoots to 0.304 because the two takes are only near-orthogonal
(|cos| ≈ 0.06) — hence the orthogonalization, and hence the achieved cosine is
asserted rather than assumed.

The control also records what the RMS-delta gate does *not* do: the blend sits at
delta 1.224745, well clear of the 0.9 threshold, so **only** the cosine gate
closes the partial-mis-wiring hole. That is asserted too, so the claim cannot go
stale silently.

The Candle DiT reproduces the frozen Torch cosine to six decimal places because
each variant is driven with its own `t5_projected_padded`, exactly as the frozen
run was.

`tests/variant_binding.rs` mutates real snapshots and requires every
cross-variant path to be rejected: the music snapshot under the SFX id, the SFX
snapshot under the music id, the music root safetensors under the SFX config,
the SFX root under the music config, and either config bolted onto the other
checkpoint's DiT. It also asserts that the two unmutated snapshots load under
their own registrations, so the suite cannot pass by rejecting everything.

It additionally closes the load-to-use window. `load_variant` verifies the pins,
but tensors are mmapped later in the lazy `pipeline()`, which historically
re-ran only the `repo_id` config check. Swapping `model.safetensors` after load
while leaving `model_config.json` in place was therefore served without
complaint — verified: with the second verification removed, the SFX registration
returns music audio. `pipeline()` now re-runs `verify_snapshot_identity` before
any tensor is opened. Measured cost: +6.9 s once per generator (SHA-256 over the
3.45 GB of pinned files at ~500 MB/s), against a load-plus-first-generate of
6.9 s without it. The control is the same generator instance serving real audio
once the authentic root is restored.

That pass runs with both the generation and pipeline mutexes held, so it roughly
doubles the cold-start window — and a request cancelled inside it would not have
observed the cancellation until the whole verification *and* load had finished
(~14 s rather than ~7 s). `verify_snapshot_identity` therefore polls the caller's
`CancelFlag` between pins, which bounds the unobserved window to one file while
keeping the check off the hashing hot path. The gate
(`cold_start_snapshot_verification_observes_cancellation_before_hashing`) is
discriminating: it asserts that an *uncancelled* call against a missing snapshot
reaches the filesystem and returns the I/O error, and that a cancelled call
against the same path returns `Canceled` instead. With the poll removed the
second leg would return the I/O error too, so the test cannot pass vacuously.

`tests/provider.rs` requires a real shipped SFX demo prompt to produce audio
that is exactly `floor(seconds × 44100)` frames, 44.1 kHz, finite, in range,
non-silent by RMS and peak, genuinely stereo by side/mid energy ratio — measured
both globally and as a median over ~23 ms windows, so a localized artefact
cannot stand in for a two-channel image — not white noise (lag-1
autocorrelation above 0.2), and not a pure tone (zero-crossing interval spread
above 0.05, computed fail-closed so a DC or sub-16 Hz channel scores 0 rather
than "infinitely spread").

### Side/mid floors are calibrated where they are enforced

Each variant's side/mid floor is calibrated by a committed 25-sample sweep — all
four of that checkpoint's shipped `demo_cond` prompts plus the prompt its own
per-run gate renders, at five seeds including the gate's seed 42.

The governing rule is that **a gate is only enforced at a configuration where it
has been measured**. The sweeps therefore default to 30 s / 8 steps, the exact
duration and step count the CI render steps enforce at, and they run on **both**
backends that enforce — Metal *and* CUDA — immediately before that backend's
render. An earlier revision of this branch calibrated on Metal at 10 s and
enforced on CUDA at 30 s; that is not a calibrated gate, and the sweep steps in
`.github/workflows/real-weights.yml` now make the two coincide by construction.

| Variant | Backend | `min_global` | `min_median_window` | Floor | Margin |
|---|---|---:|---:|---:|---:|
| SFX | Metal | 5.71451e-4 | 4.78244e-4 | 2e-4 | 2.39x |
| SFX | CUDA | 5.71427e-4 | 4.78424e-4 | 2e-4 | 2.39x |
| music | Metal | 1.64826e-1 | 2.00091e-1 | 1e-2 | 16.48x |
| music | CUDA | 1.64825e-1 | 2.00093e-1 | 1e-2 | 16.48x |

The two backends agree to five significant figures on both sweeps, which is
worth stating plainly: the margins are not absorbing a cross-backend
discrepancy, because there isn't one. That is now a measured result rather than
an assumption — it is exactly what the previous revision asserted without
checking. Every number in this table was produced by the committed sweeps
running in `.github/workflows/real-weights.yml` on this branch.

The SFX distribution is bimodal: 20 of 25 samples land in 5.7e-4 … 1.4e-3, and
the "Sparkling fantasy energy swirl" prompt lands near 1.0 at every seed. The
checkpoint genuinely renders a near-centred image on most prompts, so this is a
*duplicated-mono* gate, not a stereo-width quality bar. The music checkpoint's
image is two to three orders wider; its `1e-2` floor is left where sc-14543
shipped it rather than raised into the measured distribution, for the same
reason. The sweeps assert in both directions: every sample must clear the floor,
and the floor must not sit further below the measured minimum than the variant
allows (10x for SFX, 50x for music).

The per-run renders reproduce their sweep entries exactly — SFX seed 42 at
`global` 7.73298e-4 / `median_window` 4.78244e-4, music seed 42 at 4.51496e-1 /
4.66927e-1 — confirming the sweep covers the enforced configuration rather than
merely resembling it.

The music sweep is new in this revision. sc-14544 added the per-window-median
assertion to the shared `assert_real_audio` helper, which applies it to the music
variant too; that assertion previously had no music measurement behind it at any
duration.

### The floors, and the controls that discriminate them

A weight-free test
(`the_quality_gates_reject_the_degeneracies_they_are_named_for`) proves each of
these heuristics fails on the degeneracy it names — duplicated mono, mono plus
numerical dust, mono plus one loud localized burst, a pure tone, DC, a 10 Hz
tone, and white noise — each with a passing control alongside it, so the
analysis itself is gated in the ordinary test lane.

The SFX floor was raised from `1e-4` to `2e-4`. That change needs a control that
distinguishes the two values, or it halves the margin for nothing:

| Control | side/mid | `1e-4` | `2e-4` |
|---|---:|---|---|
| duplicated mono | 0.0 | rejects | rejects |
| mono + numerical dust (~-89 dB) | 3.4644e-5 | rejects | rejects |
| **mono + ~-77 dB differential** | **1.38569e-4** | **admits** | **rejects** |
| two independent channels | ~1.0 | admits | admits |

Only the third row discriminates, and it is asserted in both directions — it
must pass `1e-4` (or it proves nothing about the tightening) and must fail
`2e-4`. The dust control is a sanity check on the measurement, not evidence for
the tightening; an earlier revision of this branch labelled it as "the near-mono
case the old 1e-4 floor let through", which was simply wrong — `1e-4` rejects it
by a factor of three.

## Runtime and bundle gates

`candle-audio-catalog` composes both variants, the shared preparer, and both
sets of composite/component license rows into `runtime-cpu`, `runtime-macos`,
and `runtime-cuda`. SA3 registrations are contiguous and ordered music-then-SFX
across the crate registry, the catalog, and all three bundles' ordered-surface
tests.

The `sa3-small-sfx` real-weight profile materializes **both** snapshots — the
identity and divergence gates compare the checkpoints against each other — and
runs the mutation gates, the divergence gate, SFX provider conformance,
concurrent RNG isolation, and a real 30-second eight-step WAV on Metal and CUDA.

Root weights are governed by the Stability AI Community License, including its
revenue threshold and prohibited-use terms; the bundled T5Gemma component is
separately attributed under the Gemma Terms and Prohibited Use Policy. Both rows
plus the composite effective-restriction row are in
`release/model-weight-licenses.json`.

## Verification

```bash
export SA3_SMALL_MUSIC_SNAPSHOT=/models/sa3/small-music/0fef1392cd842149a2b6d445e181c97608faac06
export SA3_SMALL_SFX_SNAPSHOT=/models/sa3/small-sfx/ae12755283df9d62ca39a9b050a39a0b607b8c20

cargo test --locked -p candle-audio-stable-audio-3 --features metal \
  --test variant_binding -- --ignored --nocapture

cargo test --locked -p candle-audio-stable-audio-3 --features metal \
  --test variant_divergence -- --ignored --nocapture

SA3_TEST_DURATION=30 SA3_TEST_STEPS=8 \
SA3_SMALL_SFX_WAV_OUT=/tmp/sa3-small-sfx.wav \
  cargo test --locked -p candle-audio-stable-audio-3 --features metal \
    --test provider connected_sfx_generation_is_stereo_finite_and_exact_length \
    -- --ignored --nocapture

cargo test --locked -p candle-audio-stable-audio-3 --features metal \
  --test conformance registered_sfx_provider_passes_full_audio_conformance -- --ignored --nocapture
```
