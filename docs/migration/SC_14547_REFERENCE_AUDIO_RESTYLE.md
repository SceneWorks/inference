# sc-14547 — audio→audio restyle on Stable Audio 3

Adds the second upstream headline mode to all six registered Stable Audio 3 providers: pass an
existing recording plus a prompt and get a restyled variation. The source is SAME-encoded, mixed
into the sampler's initial noise, and denoised from a partial-strength schedule instead of from
pure noise.

| | value |
|---|---|
| Contract surface | `Conditioning::ReferenceAudio { audio, strength }` |
| Advertised on | all six ids — `stable_audio_3_{small_music,small_sfx,medium,small_music_base,small_sfx_base,medium_base}` |
| `strength` orientation | **retention** — higher preserves more of the source |
| `strength` domain | finite, `0.0..=1.0`; omitted ⇒ `0.1` |
| Off-rate source audio | **resampled** to 44.1 kHz, never rejected |
| Channels | mono duplicates, stereo passes, `>2` keeps the first two |
| New sampler entry point | `sample_dit_initialized_with_interval_and_cancel` |
| New SAME seam | `SameAutoencoder::encode_audio_with_request_rng` |

## The sign trap, and what gates it

Two same-named `strength` parameters with **opposite** meanings meet on this seam.

* **Contract side.** `Conditioning::ReferenceAudio.strength` documents itself as mirroring the
  per-reference img2img strength. This workspace's img2img strength is mflux-derived and
  **retention**-oriented: `init_time_step(steps, strength)` is the loop *start* index and the
  schedule is sliced `sigmas[start..]`, so a higher strength runs fewer steps and preserves **more**
  of the source. That is the inverse of the diffusers convention, which is exactly why the collision
  is easy to miss.
* **Sampler side.** Stable Audio 3's already-landed `strength` (`sampler.rs` `build_schedule` /
  `initialize_latents`) is upstream's `init_noise_level`: `1.0` is pure noise.

The adopted mapping keeps the contract's retention reading and converts once, in
`model::reference_noise_level`:

```
init_noise_level = 1.0 - contract_strength
```

| contract `strength` | init noise level | result |
|---|---|---|
| `0.0` | `1.0` | pure generation; the source has no influence |
| `0.1` (default) | `0.9` | a loose restyle |
| `1.0` | `0.0` | the prepared source, returned without a single DiT forward |

A silent inversion here would still run, still emit plausible audio, and do the opposite of what the
caller asked — no shape or quality check would reveal it. It has **three distinct shapes**, one per
seam the value crosses on its way from the request to the sampler's `strength` argument, and each
needs its own gate because each is blind to the other two. Two successive adversarial reviews each
found the next shape uncovered: the first landed shape (2) as a mutation and watched the whole
weight-free suite stay green, the second did the same with shape (3).

1. **The conversion is wrong.** Gated weight-free by `tests/reference_audio.rs`
   `contract_strength_is_retention_and_a_flipped_sign_fails_here`, which drives the shipped
   conversion into the shipped schedule builder and the shipped init mix and asserts contract `1.0`
   returns the prepared source bit-for-bit while contract `0.0` returns the sampler noise
   bit-for-bit. It also rejects the identity mapping and `|1-2s|` (which agrees at both endpoints) by
   pinning the interior and requiring the distance-to-source to fall strictly as retention rises.
2. **The conversion is right but the wrong field travels** — the provider builds the pipeline's
   `ReferenceAudio` out of the resolved `strength` where `noise_level` belongs. Gate (1) is blind to
   this: the conversion is still correct, it simply never reaches the sampler. Gated weight-free by
   `the_request_surface_hands_the_pipeline_the_converted_noise_level`, which builds a real
   `GenerationRequest` carrying `Conditioning::ReferenceAudio`, calls the shipped
   `resolve_reference_audio` and the shipped `model::reference_audio_for` — the same function
   `StableAudio3Generator::generate` calls — and compares the constructed struct field for field, at
   strengths where retention and noise level differ. Verified by re-running the exact mutation
   (`noise_level: reference.noise_level` → `reference.strength`): the case FAILS with
   `the pipeline must receive the converted init noise level 0, not the contract retention 1`.
3. **The right value reaches the pipeline and the pipeline hands the sampler something else** —
   `1.0 - reference.noise_level` where the reference and text-only paths converge on one scalar.
   Gates (1) and (2) are both blind to it, and so is the sampler's own `schedule[0] == strength`
   cross-check: `build_schedule` and `sample_dit_initialized_with_interval_and_cancel` are handed
   that one scalar, so inverting it moves both sides of the comparison together and they still
   agree. Mutating either consumer alone is loud; mutating their shared input was not. Gated
   weight-free by `the_pipeline_hands_the_sampler_the_converted_noise_level_as_strength`, which
   feeds the output of `reference_audio_for` straight into `pipeline::sampler_strength_for` — so
   (2) and (3) together are one continuous request → sampler-strength assertion — and additionally
   requires the sampler strength to fall strictly as retention rises, plus the `None` ⇒ `1.0`
   text-only arm. Verified by re-running the exact mutation (`reference.noise_level` →
   `1.0 - reference.noise_level` inside `sampler_strength_for`): the case FAILS with
   `the sampler must receive the init noise level 0, not the contract retention 1`.

`reference_audio_for` was extracted out of `generate` precisely so shape (2) is reachable without
weights, and `sampler_strength_for` was extracted out of `synthesize_with_reference_traced` for
shape (3). Extraction alone was not enough for (3), and neither was one further step. Three
successive reviews each found the same defect one expression downstream:

1. a pre-computed `strength` **argument** forwarded into the weights-only `sample` helper;
2. that argument removed, but replaced by `let strength = sampler_strength_for(reference)` **inside**
   `sample`.

Both shapes are one expression read by both consumers, so an edit to it moves `build_schedule` and
the DiT sample call together and the sampler's `schedule[0] == strength` check still agrees with
itself. Measured on shape (2): `let strength = 1.0 - sampler_strength_for(reference)` — a complete
user-visible inversion on all six ids — left the whole suite green, as did substituting `None`.

So `sample` takes `Option<&ReferenceAudio>` and calls `sampler_strength_for` **inline at each of the
two consumer sites**, with no binding shared between them; the text-only replay path passes `None`
rather than spelling `1.0` out again.

Everything past that `strength` argument — the sampling itself and below — needs a snapshot, and is
covered by **the real-weight case**
`real_reference_restyle_is_bounded_and_ordered_on_all_six_variants`, which requires the measured
source correlation at contract `1.0` to exceed the one at contract `0.0` by a wide margin. So the
honest statement of the split is: the request → sampler-`strength` chain is gated weight-free end to
end; what the sampler then *does* with it is gated only with real weights.

### What a single-token edit can still do

Every site below was mutated individually and the weight-free CI step re-run. Sites in `model.rs`
and in `pipeline::sampler_strength_for` fail the suite. Sites inside the weights-only methods
(`StableAudio3Generator::generate`, `StableAudio3Pipeline::synthesize_with_reference_traced`,
`StableAudio3Pipeline::sample`) do **not** — no weight-free case evaluates them — so what is stated
for those is which *runtime* check rejects them, each measured against `sampler::initialized_start`
directly rather than inferred.

| site | one-token edit | what catches it |
|---|---|---|
| `model::reference_noise_level` body | drop the `1.0 -` | weight-free: `contract_strength_is_retention_and_a_flipped_sign_fails_here` |
| `model::resolve_reference_audio` `noise_level` / `strength` fields | swap the two | weight-free: both of the other two gates |
| `model::DEFAULT_REFERENCE_STRENGTH` | `0.1` → `0.9` | weight-free: all three gates (the `assert_ne!` discriminator trips) |
| `model::reference_audio_for` field selection | `noise_level` → `strength` | weight-free: `the_request_surface_hands_the_pipeline_the_converted_noise_level` |
| `pipeline::sampler_strength_for` body (either arm) | invert, or change the `None` constant | weight-free: `the_pipeline_hands_the_sampler_the_converted_noise_level_as_strength` |
| `pipeline::reference_halves_agree` body | flip or neuter the predicate | weight-free: the tail of the same gate |
| either `sampler_strength_for(reference)` call site in `sample` | invert it, or swap `reference` for `None` | runtime, fail-closed: it now disagrees with the other site, and `initialized_start` bails `init noise strength must equal every schedule's first sigma`. Measured at levels `0.75/0.9/0.0/1.0/0.5`; the only pairs that pass are the ones where the edit does not change the value (inversion at level `0.5`, `None` at level `1.0`), which are behaviour-neutral. |
| `sample`'s `init_latents` or `reference` argument at the call site | drop one, keep the other | runtime, fail-closed: `reference_halves_agree` rejects the disagreement. Before this PR's third cycle, dropping `reference` alone was **silent** — the source was still encoded and then discarded at strength `1.0`, degrading a restyle to plain text-to-audio. |
| `let reference = reference_audio_for(request)` in `generate` → `None` | delete the feature | real-weight only: `real_reference_restyle_is_bounded_and_ordered_on_all_six_variants` would measure a source correlation of ~0 at contract `1.0`. Nothing weight-free sees it; `generate` needs a loaded pipeline. |
| `prepare_reference_pcm`'s target size in `reference_latents` | halve it | runtime, fail-closed: the SAME-encode shape check in `reference_latents` |

Two things are worth stating plainly rather than as reassurance. First, a **coordinated edit to both
`sampler_strength_for` call sites at once** agrees with itself, passes `initialized_start`, and is
green weight-free — it is caught only by the real-weight case. That is a two-site edit, not a
one-token one, but it is reachable. Second, deleting the feature at `generate` is likewise
weight-free-invisible. Neither is claimed to be closed here; both are named so the next reader
starts from them.

Measured source correlation on Metal, 5 s / 4 steps, seed 7 (M-series, `--release --features metal`):

| id | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| `small_music` | -0.003934 | 0.574467 | 0.909590 | 0.951326 | 0.966623 |
| `small_sfx` | 0.005339 | 0.718412 | 0.937722 | 0.952103 | 0.966623 |
| `medium` | 0.003340 | 0.903839 | 0.945623 | 0.962782 | 0.985503 |
| `small_music_base` | 0.006965 | 0.766585 | 0.942801 | 0.959396 | 0.966623 |
| `small_sfx_base` | 0.000483 | 0.649653 | 0.920254 | 0.956970 | 0.966623 |
| `medium_base` | -0.006359 | 0.905855 | 0.961553 | 0.977967 | 0.985503 |

The shipped floors — `full > 0.5`, `full > none + 0.3`, `|none| < 0.2`, `full < 0.9999` — are read
off that table with wide margin, not fitted to it. Two columns are load-bearing rather than
incidental. The `1.0` column is *identical* across all four SAME-S ids and across both SAME-L ids
and is independent of the prompt, because at full retention the DiT is skipped entirely and what is
measured is a pure autoencoder round trip of the same prepared buffer. The `0.0` column sits within
`0.007` of zero on every id, which is what makes the divergence floor a floor.

Note the consequence of the mapping: `sampler.rs`'s `strength == 0` DiT short-circuit is reached at
contract `strength = 1`, so "returns the prepared source" is a **contract-endpoint** claim and is
tested there rather than at the sampler's.

That short circuit has a second, caller-visible consequence worth stating: every initialized sampler
entry point returns on `skip_model` **before invoking its progress callback even once**, so a request
at contract `strength = 1.0` emits zero `Progress::Step` events and then goes straight to
`Progress::Decoding`. A progress bar driven off step counts sits at zero for the whole sample phase.
That is accurate — there genuinely are no steps — but it is behaviour of a documented endpoint, so it
is recorded here and on `sampler::InitializedStart::skip_model` and `model::reference_noise_level`
rather than left to be discovered.

## Resample, do not reject

The open decision on the story is settled by evidence rather than preference.

1. The shared resampler already implements and gates 48 kHz → 44.1 kHz — the 160:147 ratio, on
   interleaved stereo, with passband/alias and impulse-alignment tests in `candle-audio`'s own
   `dsp` suite. It needs ~215 taps × 147 phases, far under the crate's `RESAMPLE_MAX_TAPS_PER_PHASE`.
2. **The ACE-Step precedent is hollow.** That crate contains no resampler of any kind, so its "audio
   edit source must be 48000 Hz" is a *capability gap*, not a policy about how music models should
   treat caller audio.
3. Both other audio generators in this lane emit 48 kHz, so rejecting would refuse audio produced
   one step earlier in the same product.
4. `candle-audio-stable-audio-3`'s own crate docs already mandate using `candle_audio::dsp::resample`
   rather than provider-local DSP.

The same ruling is intended for sc-14548 (inpaint).

## Preprocessing order

1. resolve the target duration → the adapted sample size (requested duration **plus** the 6 s
   `DEFAULT_DURATION_PADDING`, never the source's extent);
2. `dsp::resample` the **whole** buffer to 44.1 kHz;
3. trim, or right-zero-pad from offset 0, to exactly that sample size;
4. conform channels **after** padding — mono duplicates, stereo passes, `>2` keeps the first two;
5. SAME-encode the complete adapted buffer.

Steps 1–3 genuinely depend on their predecessor and their results are asserted by
`sizing_geometry_comes_from_the_requested_duration_not_the_source_extent` and
`prepared_reference_pcm_is_resampled_channel_conformed_and_target_sized`.

**Step 4's position is not observable, and is not claimed to be gated.** Because the pad value is
zero, conforming channels before padding and conforming after it produce byte-identical output —
duplicating a zero commutes with padding zeros, as does keeping the first two of four zeros. A
reviewer confirmed this empirically by rewriting `prepare_reference_pcm` to conform first and
watching the five weight-free cases that existed at the time still pass. The spec's "conform after padding" bullet is
therefore satisfied **by construction, not by a test**, and is recorded that way rather than
overclaimed; it would only become observable if the pad value ever stopped being zero. What the test
does pin is the channel-conformance *result*, which is real.

The attention/padding mask is unchanged — it still derives from the requested duration plus the
headroom. `local` inpaint conditioning stays exactly zero: on this path the source is `init_data`
only, never masked local input.

The geometry claim is asserted at the seam that could actually get it wrong. `prepare_reference_pcm`
returns `target_frames * CHANNELS` unconditionally, so sweeping *source* lengths through it proves
nothing; the number that decides sizing is `SynthesisParameters::duration_secs`, resolved by
`model::synthesis_parameters` from `audio.target_duration` (or the variant default) and never from
the clip. That resolution is asserted on all six ids across three source lengths, together with the
exact geometry arithmetic behind it — `valid_lengths[0] == 172` for a 10 s request (108 latent frames
of content plus 64 of headroom), two-sided, with each term separately shown to move the result.

## Draw order

The request-local stream draws the sampler's initial noise **first**, then the source encode's
draws, then Pingpong, then SAME decode. Encoding first would move every later draw, so the same seed
would sound different merely for having a clip attached. This is enforced two ways, neither of which
is as strong as "fails closed" unqualified would suggest:

* the pipeline **guards against a future edit**: it errors if `draws_after_initial_noise != 1`. As
  shipped that cannot fire — the `SeededNoise` is constructed three lines above the check and drawn
  from exactly once — so it is a tripwire for something being inserted between the two, not a live
  check on the current code. And under a genuine reordering it only discriminates where the encode
  itself draws, i.e. on the two SAME-L ids (`medium`, `medium_base`); on the four SAME-S ids it
  cannot fire at all;
* `real_initial_sampler_noise_precedes_the_source_encode` asserts the counts through
  `synthesize_with_reference_traced`, on all six, with the same SAME-S caveat below.

This is pinned as a **structural** invariant, not as an upstream-parity claim: no frozen SA3 fork is
vendored in this repository, so the upstream order cannot be substantiated here.

Measured, and the reason the real-weight case runs all six rather than one: **SAME-S consumes zero
draws on encode**, so on the four small ids `draws_after_source_encode == draws_after_initial_noise`
and the ordering assertion is *vacuous* — swapping the two operations would not move a count. Only
medium's **SAME-L** encode draws (`1` → `2`), so only `medium` and `medium_base` can falsify the
invariant. The case therefore also asserts that at least one checkpoint reported a drawing encode,
so a future change making every encode deterministic fails loudly instead of degrading the gate into
a tautology. Verified by mutation: moving the encode ahead of the initial draw fails on `medium`
with `initial sampler noise must be the request stream's first draw, saw 2`, and passes unchanged on
every small.

## New seams, and why each was needed

* **`SameAutoencoder::encode_audio_with_request_rng`.** The existing `encode_audio_with_rng` takes a
  concrete `SameNoiseRng` — a stochastic stream *independent* of the request — so a provider using
  it could not state, let alone gate, where the source encode's draws sit. The new entry point
  mirrors `decode_audio_with_request_rng` through the existing `RequestSameNoise` adapter, and polls
  cancellation before every SAME dispatch.
* **`sampler::initialized_start` + `sample_dit_initialized_with_interval_and_cancel`.** The restyle
  path needs the frozen `x0 = init*(1-s) + noise*s` initialization *and* the cancellable,
  guidance-interval-aware DiT loop at once. Composing them in the provider would have put the
  `strength == 0` no-DiT short circuit back under a caller convention, which is what
  `sample_initialized` exists to prevent, so both forms now share one preamble.
* **`model::validate_request_for`.** Every request-shaping rule this family owns is weight-free;
  exposing the variant/request pair lets the whole rejection surface be gated in the PR lane instead
  of only on a real-weight runner.

## Validation

Beyond the generic floor (which enforces only that an explicit `strength` is finite), the provider
rejects: more than one `ReferenceAudio`; `ReferenceAudio` combined with `AudioEdit`; empty or
non-finite PCM; a zero sample rate; zero channels; a sample count that is not a whole number of
frames; and a `strength` outside `0.0..=1.0`.

The error **type** carries a deliberate split, and both sides are asserted by type so it cannot
drift. The two arity/combination refusals — more than one clip, and a clip alongside an `AudioEdit` —
are statements about what this family *can do* and are typed `Error::Unsupported`. Everything else is
about the caller's data rather than the model's capability and stays `Error::Msg`.

The `AudioEdit` combination check is defence in depth and is labelled as such in the source: this
family advertises no audio-edit modes, so the generic floor already refuses such an item on its own.
The check exists so that advertising audio editing later (sc-14548) cannot silently make the
*combination* reachable, which would hand one source clip two contradictory roles.

## CI

| lane | selects |
|---|---|
| `ci.yml` "Test Stable Audio 3 weight-free quality gates" | `--test reference_audio` — 7 weight-free cases, including **all three** sign gates |
| `real-weights.yml` `sa3-base-identity-metal` | `--test reference_audio -- --ignored` |
| `real-weights.yml` `sa3-base-identity-cuda` | `--test reference_audio -- --ignored` |

The two `sa3-base-identity` jobs are the only lanes that provision all six pinned snapshots, which is
what the story's "every registered variant" acceptance requires.
`scripts/tests/test_sa3_ci_target_coverage.py` enforces the weight-free half of that wiring; the
real-weight half is verified by reading the jobs' actual `--test` flags.

## Follow-ups

* A **preprocessing oracle**: the shared Rust resampler is not numerically identical to torchaudio,
  so any future cross-framework comparison of this path inherits an unattributed delta until one
  exists.
* Resample cost at medium's 380 s cap is a scalar single-threaded polyphase pass, ~7 G MAC on the
  host before any GPU work. Not a correctness issue; it is a real-weight lane timing consideration.
