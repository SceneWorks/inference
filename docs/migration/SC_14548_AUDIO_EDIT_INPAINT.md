# sc-14548 — Stable Audio 3 inpaint / repaint / extend (`Conditioning::AudioEdit`)

Bounded source editing on all six registered Stable Audio 3 checkpoints: regenerate a `[start, end)`
region of an existing clip from the prompt while keeping the rest, and continue a clip past its end
with the same mechanism.

## What was already landed, and what this story actually built

The **DiT half was complete before this story started** and is untouched by it. `DitInputs` already
carried a `[batch, 257, time]` `local_conditioning` ordered `[inpaint_mask, inpaint_masked_input]`,
with a shape gate, a dtype cast and the transpose into the block stack; every block already had its
`LocalConditioning` MLP; the six configs already pinned `local_add_cond_ids` and
`local_add_cond_dim = 257`; and the batch-2 CFG forward already repeated the identical tensor on both
branches.

What was missing was the **producer**. `StableAudio3Pipeline` handed the DiT a zero tensor and the
provider advertised no `audio_edit_modes` at all, so every `AudioEdit` request died as typed
`Unsupported` on the shared capability allowlist. This story is: descriptor advertisement, request
validation, region resolution, source preparation and SAME encoding, mask construction, the
257-channel concat, and the PCM stitch.

## Advertised surface

| | |
|---|---|
| `ConditioningKind` | `ReferenceAudio` (sc-14547) **and** `AudioEdit` |
| `audio_edit_modes` | `[Inpaint, Repaint, Extend]` — on all six ids |
| `AudioEditMode::Cover` | **not advertised** → typed `Unsupported` from the generic allowlist |
| `AudioEdit.strength` | **refused** as typed `Unsupported`, never ignored |

### Why `Cover` is off, and why that is not a narrowing

Two independent reasons. Each of the six checkpoints pins its **complete** conditioner surface in
config — `global = [seconds_total]`, `local = [inpaint_mask, inpaint_masked_input]` — so there is no
style/cover conditioner in any of them for `Cover` to map onto; gen-core's own doc for that mode names
ACE-Step, which does ship it. And the capability a caller means by "cover" is a whole-clip restyle,
which is not dropped: it is `Conditioning::ReferenceAudio` (sc-14547), advertised alongside on the same
six ids, with a retention `strength` knob this surface does not have. The `AudioEdit` refusal message
names it.

`AudioEditMode::Cover` stays in gen-core — ACE-Step implements it.

### Why `strength` is refused rather than honoured or ignored

Stable Audio 3's inpaint conditioner is a hard binary mask times the encoded source, concatenated as
channels. There is no scalar anywhere on that path a "strength" could modulate, so honouring it would
mean *inventing* a semantic that silently changes the sampler trajectory. gen-core carries
`AudioEdit.strength` as a first-class float, so accepting and discarding it is the "appears to work,
does nothing" failure — the shipped anti-pattern being avoided is `chatterbox/src/model.rs`, which
destructures `ReferenceAudio { audio, .. }` and drops the strength without a word. The idiom followed
instead is `candle-gen-mage/src/lib.rs`, which refuses per-reference strength inside its conditioning
walk.

Gated by asserting **both** halves: `Some(0.5)` errors typed `Unsupported` *and* `None` succeeds. Only
one of those on its own is satisfiable by a family that refuses every edit, or by one that ignores the
field.

### `Repaint` ≡ `Inpaint`

gen-core describes `Inpaint` as silence-substituted and `Repaint` as context-conditioned. That
distinction is written against ACE-Step's two native tasks; upstream Stable Audio 3 exposes three
paths in total (text-to-audio, audio-to-audio via `init_audio`, inpaint) and has one inpaint
mechanism. So on this family the two are aliases and must be byte-identical for the same request and
seed.

That is **structural**, not merely tested: the mode is consumed entirely in `model::audio_edit_for`,
and `pipeline::AudioEdit` has no mode field for anything downstream to branch on. `Extend` differs only
in where the region sits and how long the output is, both of which are already numbers in that struct.
`real_repaint_is_byte_identical_to_inpaint` asserts the consequence anyway, because a structural
argument does not survive somebody adding a field.

## The canonical path

1. **Resolve the region and the output length** (`model::resolve_audio_edit` / `audio_edit_for`). The
   source's duration is measured on the *post-resample* 44.1 kHz timeline. Inpaint/Repaint output
   exactly the source's duration; Extend outputs `region.end_secs`. `synthesis_parameters` reads that
   number, so an extend with no `target_duration` renders its region's end rather than the variant's
   120 s / 380 s default.
2. **Adapt the geometry** with the already-landed `sampler::adapt_sample_size_for_max`:
   `ceil_to(ceil_to((d + 6s) * 44_100, 4096), 8192).min(sample_size)`. Not reimplemented.
3. **Draw the sampler's initial noise first**, then encode the source — the frozen order, so attaching
   a clip does not move the draw a text-only request at the same seed would have made.
4. **Prepare the source** with sc-14547's `pipeline::prepare_reference_pcm`: resample the whole buffer
   to 44.1 kHz, trim/right-zero-pad from offset 0 to the adapted size, conform channels. Resample, not
   reject — the ruling recorded on sc-14547 and inherited here.
5. **Build the keep mask at audio-sample resolution over the adapted size** (`edit_keep_mask`): ones
   keep, zero `[start*44_100, end*44_100)`, and *also* zero `[seconds_total*44_100, adapted_size)` for
   training parity.
6. **Resize to latent resolution** with `candle_audio::ops::nearest_downsample1d` (new; see below).
7. **`masked_input = SAME_encode(prepared) * mask`**, then concat channel-first
   `[mask (1), masked_input (256)] = 257`.
8. **Sample from pure seeded noise** — no `init_data`, no `init_noise_level`. The source reaches the
   model only through the local conditioner.
9. **Stitch** the canonical prepared source PCM back over everything outside `[start, end)`,
   bit-exactly.

### Pinned alignment, reproduced exactly

| case | adapted | latent | edit latents | effective boundary |
|---|---|---|---|---|
| 10 s inpaint `[2, 7)` | 712,704 | 174 | `[22, 76)` | 108 (padding `[108, 174)` zero) |
| 10 s → 18 s extend | 1,064,960 | 260 | `[108, 194)` | 194 (padding `[194, 260)` zero) |

### The stitch is exact equality, not a bound

Frozen upstream regenerates and decodes the whole clip and pastes nothing back. This repository's
`AudioEdit` contract promises the rest of the clip is preserved, so the frozen behaviour is preserved
through the raw decode and the preservation is then satisfied by an explicit stitch. Because the
preserved span is written from `prepare_reference_pcm`'s own output, the assertion is **bit
equality** — the story's original "preserved to a tight numeric bound" is amended, since a tolerance
would pass a stitch that had slipped a frame, which is the entire failure class this exists to
exclude. Any future crossfade must live wholly inside `[start, end)`.

## New shared op: `candle_audio::ops::nearest_downsample1d`

`candle_audio::ops` had `nearest_upsample1d` and no downsample; candle-core has none on any backend.
Audio → latent is a downsample by exactly 4096, and an ad-hoc stride at the call site is precisely the
kind of arithmetic that goes untested, so it is a named op with its own gates in that module.

The rule is `dst[j] = src[j * k]` — the **first** frame of each block, matching nearest resizing in
both directions. For a binary mask that is load-bearing rather than cosmetic: under it, a zeroed audio
span `[start, end)` maps to `[ceil(start/4096), ceil(end/4096))`; under a last-of-block or
any-nonzero rule the span widens and the edit window moves with no shape change. Three tests pin it:
the element rule, the span property directly, and a round trip against `nearest_upsample1d`.

## The `ReferenceAudio` + `AudioEdit` refusal is now live

sc-14547 shipped that refusal and labelled it honestly as defence in depth: `AudioEdit` was not
advertised, so the generic allowlist refused the item on its own and the specific check never fired.
Advertising the kind here turns it on.

It was therefore **hoisted** out of `validate_reference_audio` into
`reject_reference_and_edit_combination`, called once from `validate_request` before both per-kind
validators, so the caller sees the same message regardless of which conditioning item is written
first — asserted by comparing the two orderings' errors. `tests/reference_audio.rs`'s combination case
was also rewritten: it used `AudioEditMode::Cover`, which the generic floor refuses on its own, so it
was proving the mode allowlist rather than the combination. It now carries a well-formed `Inpaint`.

## Mutation matrix — every gate verified RED under its own mutation

Applied to the shipped code, suite re-run, reverted.

| mutation | result | failing case |
|---|---|---|
| mask polarity inverted | **RED** | `the_keep_mask_zeroes_the_region_and_the_padding_and_nothing_else`, `the_local_conditioning_is_mask_first_then_the_masked_source` |
| concat order → `[masked_input, mask]` | **RED** | `the_local_conditioning_is_mask_first_then_the_masked_source` |
| mask built at the un-adapted size | **RED** | `the_edit_geometry_reproduces_the_pinned_alignment_examples` + 2 |
| region seconds applied at the caller's rate (pre-resample) | **RED** | `the_edit_geometry_reproduces_the_pinned_alignment_examples` + 2 |
| latent indices rounded instead of derived | **RED** | `the_edit_geometry_reproduces_the_pinned_alignment_examples` + 1 |
| ones left in the padding | **RED** | `the_keep_mask_zeroes_the_region_and_the_padding_and_nothing_else` + 1 |
| unmasked source latents (drop the multiply) | **RED** | `the_local_conditioning_is_mask_first_then_the_masked_source` |
| negative CFG branch zeroed | **RED** | `the_negative_cfg_branch_receives_the_same_local_conditioning` |
| `strength` accepted and ignored | **RED** | `audio_edit_validation_rejects_every_malformed_request_on_every_variant` |
| second `AudioEdit` silently dropped | **RED** | `audio_edit_validation_rejects_every_malformed_request_on_every_variant` |
| `nearest_downsample1d` keeps the last of each block | **RED** | `ops::tests::downsample_keeps_the_first_frame_of_each_block`, `..._maps_a_zeroed_span_to_the_ceiling_span` |

### What a single-token edit can still do

Named rather than claimed closed — the sc-14547 review history is that "closed" is what let the next
hole survive.

| edit | weight-free lane | what catches it |
|---|---|---|
| the stitch invocation in `synthesize_traced` → `audio` (skip the stitch) | **green** | `real_inpaint_preserves_the_outside_exactly_and_changes_the_inside`, `real_extend_keeps_the_source_prefix_and_bridges_the_seam` |
| `audio_edit_for(request)` → `None` at the forward in `generate` | **green** | fails **closed** at runtime via `pipeline::conditioning_is_forwarded`; any real render, i.e. every real-weight case |
| `reference_audio_for(request)` → `None` at the same forward | **green** | same guard, same lane |
| the local-tensor handoff into `sample` replaced with zeros | **green** | the real-weight inpaint/extend cases (the region would stop being conditioned on the source) |

Two structural choices shrank that list rather than documenting it larger:

* `edit_local_conditioning` takes the encoded source as a plain **tensor** instead of encoding it, so
  the whole mask → resize → multiply → concat chain is drivable with synthetic latents and no weights.
  Inside the SAME-encoding method it would have been real-weight-only, which is exactly how
  sc-14547's three sign inversions each stayed green;
* there is **no `match`** selecting a synthesis method in `generate`. `synthesize_conditioned` takes
  both optional conditionings and decides internally, so "route the edit into the wrong arm" is not a
  reachable edit.

## Real-weight results — all six, Metal, `--release`

### Inpaint, 6 s source at 48 kHz mono, region `[2 s, 4 s)`, 4 steps

Outside the region: **bit-exact** against `prepare_reference_pcm`'s own output, on every id, at
**both** seeds. Interior source energy `0.124632` on every row (the same prepared buffer).

| id | source divergence | seed divergence |
|---|---|---|
| `stable_audio_3_small_music` | 0.049032 | 0.060890 |
| `stable_audio_3_small_sfx` | 0.040207 | 0.034959 |
| `stable_audio_3_medium` | 0.023271 | 0.018139 |
| `stable_audio_3_small_music_base` | 0.060283 | 0.043349 |
| `stable_audio_3_small_sfx_base` | 0.072864 | 0.069078 |
| `stable_audio_3_medium_base` | 0.022106 | 0.018048 |

**The first calibration was wrong and is recorded rather than rewritten.** The case originally
asserted a single measurement — `source_divergence > 0.25 * source_energy` — against a number picked
before anything was measured. On the real-weight run `medium` failed it at `0.023271` against a
`0.031158` floor **while genuinely regenerating the region**. That is the story's own named hazard
("thresholds must be derived, not eyeballed, and floors taken from the low mode of a bimodal
sweep"), and the spread here is exactly bimodal by autoencoder family: the two SAME-L ids sit at
`0.018`–`0.023`, the four SAME-S ids at `0.035`–`0.073`.

Two things changed as a result, not one:

* the floor is now `0.08 * energy = 0.009971`, taken from the **low** mode at `1.8x` below its
  tightest member rather than fitted to the average;
* a second, **threshold-free** measurement was added. `source_divergence` is the weak one: a full
  SAME round trip already diverges from its input (sc-14547 measured source correlation `0.966`–`0.986`
  at full retention, not `1.0`), so a region copied through the autoencoder still scores well above
  zero. `seed_divergence` — the same region at two seeds — is the one a copy cannot pass at *any*
  threshold, because a copied or whole-clip-overwritten region is seed-independent and scores
  identically zero. Every id is now measured before anything is asserted, so a single failing
  checkpoint no longer hides the other five's numbers, which is what made the first floor hard to
  calibrate in the first place.

### Extend, 10 s → 18 s, 4 steps

Prefix: **exactly** the 441,000-frame prepared source on every id. Output exactly 793,800 frames.
The seam bound is the material's own 99.9th-percentile frame-to-frame step over the second before
the seam (`0.163181`), so it is measured rather than chosen.

| id | seam step | tail energy |
|---|---|---|
| `stable_audio_3_small_music` | 0.039421 | 0.008514 |
| `stable_audio_3_small_sfx` | 0.040421 | 0.044111 |
| `stable_audio_3_medium` | 0.033704 | 0.012578 |
| `stable_audio_3_small_music_base` | 0.037924 | 0.010317 |
| `stable_audio_3_small_sfx_base` | 0.038054 | 0.036780 |
| `stable_audio_3_medium_base` | 0.045983 | 0.059855 |

Every seam step is **below** the material's own typical step, i.e. the join is smoother than the
source's own waveform activity — not merely inside a generous bound. The prepared buffer is silent
past the source, so a non-zero tail energy is proof the tail came from the model.

### Repaint ≡ Inpaint

Byte-identical on all six ids for the same request and seed, with a non-silent control so two
all-zero buffers cannot pass.

### Draw order

`draws_after_initial_noise == 1` on all six. `draws_after_source_encode`: `1` on the four SAME-S ids
(`small_music`, `small_sfx`, `small_music_base`, `small_sfx_base`) and `2` on `medium` /
`medium_base`. This reconfirms sc-14547's finding on the edit path: **SAME-S consumes zero draws on
encode**, so the ordering assertion discriminates on only two of the six, which is why the case runs
all six *and* separately requires at least one drawing encode rather than quietly degrading into a
tautology.

## No torch oracle — what is and is not claimed

**There is no frozen-PyTorch inpaint oracle on this machine and none is vendored in the repository.**
Upstream lives in an external checkout per `SC_14534_SA3_REFERENCE_PARITY.md`, which is not present.
No cross-framework parity is claimed for this path, and none should be inferred from the geometry
agreeing with the numbers on the story — those were derived from this repository's own landed
`adapt_sample_size_for_max` and from the resize rule, not from a torch run.

What *is* claimed: internal consistency against the shipped preprocessing (exact outside-region
preservation, measured interior divergence, alias byte-equality, a seam bound derived from the
material), plus the mutation matrix above. Producing the oracle is filed as a follow-up.

## CI wiring

* **Weight-free** — `--test audio_edit` in `ci.yml`'s "Test Stable Audio 3 weight-free quality gates"
  step: **10 cases**. `scripts/tests/test_sa3_ci_target_coverage.py` passes; the step's comment counts
  are updated (33 → 43 SA3 weight-free cases across nine named targets).
* **Real weights** — `--test audio_edit -- --ignored` on `sa3-base-identity-metal` and
  `sa3-base-identity-cuda` (profile `sa3-base-identity`), the only jobs provisioning all six pinned
  snapshots. Verified against those jobs' actual `--test` flags rather than their comment blocks.

## Related

* `SC_14547_REFERENCE_AUDIO_RESTYLE.md` — the audio→audio restyle path this one deliberately does not
  duplicate, and the source of `prepare_reference_pcm`, `encode_audio_with_request_rng` and the
  resample-not-reject ruling.
