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
| `edit_local_conditioning_is_present` stops rejecting the disagreement | **RED** | `the_local_conditioning_handoff_must_still_be_present_where_it_crosses_into_the_dit` |
| `edit_retained_latent_count` ignores the padding boundary | **RED** | same case — the retained count stops agreeing with the tensor `edit_local_conditioning` builds |
| `edit_geometry_matches_request` drops the effective-span comparison | **RED** | `the_resolved_edit_geometry_must_match_the_geometry_the_request_is_sampled_at` |
| `tensor_has_nonzero`'s `total > 0.0` → `total >= 0.0` | **RED** | `the_local_conditioning_handoff_must_still_be_present_where_it_crosses_into_the_dit` — "[0s, 16s): the retained-latent count and the built tensor must agree on whether the DiT is conditioned at all, left: false right: true" |
| `tensor_has_nonzero` drops the `.abs()` before `sum_all` | **RED** | same case — "a local conditioning whose elements cancel to a zero sum is still present; the presence reduction must be over \|x\|" |

### Every weights-only call site on the edit path, mutated and run

The whole edit path outside the weight-free helpers is two places: the `edit` arm of
`StableAudio3Pipeline::synthesize_traced` (plus its tail) and the conditioning receipt built in
`StableAudio3Generator::generate`. **Every** argument and binding in those was mutated one token at a
time, the mutated tree rebuilt `--release`, and the relevant real-weight case run against the six
pinned snapshots on Metal. The right-hand column is what the run *did*, quoted from its output — not
what reading the code suggests. Round-2 review found a row in the previous version of this table that
was reasoned rather than run and was wrong.

None of these are reachable by the weight-free lane, and that is the standing property of this path
rather than a gap this table closes: `generate` and `synthesize_traced` both need multi-gigabyte
weights. What the table records is which case fails, and how fast.

| # | site → mutation | result | what the run reported |
|---|---|---|---|
| 1 | `prepare_and_encode(..., edit.sample_rate, ...)` → `SAMPLE_RATE` | **RED** | `real_inpaint…`: "frame 0 channel 0 is outside [2s, 4s) and must be the prepared source exactly" |
| 2 | `prepare_and_encode(..., edit.channels, ...)` → `1` | **GREEN** on `real_inpaint…` / `real_repaint…`, **RED** on `real_extend…` | see below — the inpaint fixture's source is **mono**, so this is a no-op there. `real_extend…`: "the first 10 s must be the prepared source bit for bit" |
| 3 | `edit_geometry(&edit, &geometry, parameters.duration_secs)` → `edit.end_secs` | **RED** | `real_inpaint…`: "the edit's effective span covers 44 latents, but the request's own duration covers 65" (`edit_geometry_matches_request`, new) |
| 4 | `local = edit_local_conditioning(…)` → `let _ = …` | **RED**, 15 s | `real_inpaint…`: "an audio edit whose keep mask retains source latents (true) must reach the DiT as non-zero local conditioning, saw false" (`edit_local_conditioning_is_present`, new) |
| 5 | `edit_local_conditioning(&resolved, &latents, …)` → `&latents.ones_like()?` | **RED** | `real_inpaint…`: "two inpaints differing only in their source rendered the *identical* interior, which is what an unconditioned region looks like" |
| 6 | `edit_state = Some((prepared, resolved))` → `None` | **RED** | `real_inpaint…`: the same presence guard, in the *other* direction — "retains source latents (false) … saw true" |
| 7 | `stitch_outside_region(&audio, prepared, resolved)` → `audio` | **RED** | `real_inpaint…`: "frame 0 channel 0 is outside [2s, 4s) and must be the prepared source exactly" |
| 8 | `&local` argument to `self.sample(…)` → `&local.zeros_like()?` (*after* the presence guard) | **RED**, 16 s | `real_inpaint…`: "two inpaints differing only in their source rendered the *identical* interior, which is what an unconditioned region looks like" — the `conditioning_divergence` byte-inequality |
| 9 | `init_latents` left `None` on the edit path → `Some(latents.clone())` | **RED** | `real_inpaint…`: "reference-audio init latents and the reference itself must be supplied together, saw init_latents=true reference=false" — sc-14547's `reference_halves_agree`, catching an sc-14548 mutation |
| 10 | `order = Some(ReferenceDrawOrder {…})` in the edit arm → `None` | **RED** | `real_edit_initial_sampler_noise_precedes_the_source_encode`: "an edit render reports its draw order" |
| 11 | receipt field `edit: audio_edit_for(request)` → `None` | **RED** | `real_inpaint…`: "a request carrying an audio edit (true) must forward it to the pipeline, saw false" |
| 12 | receipt field `request_has_edit: …any(…)` → `false` | **RED** | `real_inpaint…`: "a request carrying an audio edit (false) must forward it to the pipeline, saw true" |
| 13 | the `edit` the receipt is destructured into, forwarded to `synthesize_traced` one line after `conditioning.check()` → `None` | **RED**, 14 s | `real_inpaint…` at `tests/audio_edit.rs:1458`: "stable_audio_3_small_music: frame 0 channel 0 is outside [2s, 4s) and must be the prepared source exactly — left: -0.027633887, right: -0.44984582" |

Rows 11 and 12 are new coverage, not a restatement. Round-1 checked the forward where the values were
**computed** and left the call's own argument list unguarded — writing `None` in the argument slot
satisfied that guard, because the guard read the local. The resolved values now travel as a
`pipeline::ForwardedConditioning` receipt carrying the request booleans with them, re-checked inside
`synthesize_conditioned`.

**What that receipt does is *move* the nullable seam out of `generate` and into
`synthesize_conditioned` — it does not remove it, and the earlier claim in this document that there
was "no longer a separate `edit` argument to null out" was false.** Round-2 review ran it. Inside
`synthesize_conditioned` the receipt is checked and then destructured, and the destructured `edit` is
forwarded to `synthesize_traced` as a plain argument one line later; `check()` reads the receipt, not
the argument, so `edit` → `None` there is still one token. It is **row 13** above, and it is caught —
by row 7's catcher, the bit-exact-outside assertion in `real_inpaint_…`, with weights. What the
receipt buys is that nulling a *field* is refused, so the single-token edit on the receipt itself
becomes two tokens (a field plus its matching boolean). The three sites that carried the false
wording — `pipeline.rs`'s `ForwardedConditioning` doc, `tests/audio_edit.rs`'s
`a_resolved_edit_must_be_the_one_the_pipeline_is_handed` doc, and this paragraph — now say that.

**Row 2 is a real hole in the fixture, not in the code, and it is left open deliberately.** The
inpaint case's source is a 48 kHz **mono** clip, so hard-coding `1` for `edit.channels` changes
nothing there and the case ran to completion green on all six ids. The catcher is
`real_extend_…` — and **only** that case: run under the same mutation it is RED with "the first 10 s
must be the prepared source bit for bit". A fixture-level fix (making the inpaint source stereo)
would move the row rather than change the class, since the mutation would then be invisible on a mono
request instead. Stated here rather than silently omitted.

An earlier version of this paragraph also named `real_repaint_is_byte_identical_to_inpaint` as a
catcher. That was reasoned rather than run, and running it disproved it: under the same mutation the
case is **GREEN** on all six ids (414 s). The general property, worth stating once because it applies
to every row of this table and not just this one:

> **`real_repaint_is_byte_identical_to_inpaint` compares two renders of the same code, so it is
> structurally blind to any mutation applied uniformly to both.** Both renders are mutated
> identically, the equality holds, and the `energy > 1e-4` non-silence control still passes. It gates
> the `Repaint` ≡ `Inpaint` alias and nothing else. For a shared-path mutation the catcher to name is
> `real_inpaint_…` or `real_extend_…`. The same note is recorded on the case itself, so a future
> reader of the source reaches it without this document.

**Row 9** was expected to be the weak one and is not. The edit path deliberately leaves
`init_latents` at `None` — an edit starts from pure seeded noise and the source reaches the model
only through the local conditioner; supplying the encoded source as `init_data` as well is the
restyle contract, and it would change the render without changing its length, rate, finiteness,
bit-exact outside, or any of the three interior divergences in a *signed* way. What refuses it is
`pipeline::reference_halves_agree`, shipped by **sc-14547** for a different failure entirely (an
encoded source travelling without the reference that produced it). It is recorded here because a gate
catching a mutation it was not written for is the thing that stops being true silently.

Two structural choices keep that list this short rather than longer:

* `edit_local_conditioning` takes the encoded source as a plain **tensor** instead of encoding it, so
  the whole mask → resize → multiply → concat chain is drivable with synthetic latents and no weights.
  Inside the SAME-encoding method it would have been real-weight-only, which is exactly how
  sc-14547's three sign inversions each stayed green;
* there is **no `match`** selecting a synthesis method in `generate`. `synthesize_conditioned` takes
  the whole receipt and decides internally, so "route the edit into the wrong arm" is not a reachable
  edit.

## Real-weight results — all six, Metal, `--release`

### Inpaint, 6 s source at 48 kHz mono, region `[2 s, 4 s)`, 4 steps

Outside the region: **bit-exact** against `prepare_reference_pcm`'s own output, on every id, at
**both** seeds. Interior source energy `0.124632` on every row (the same prepared buffer).

| id | source divergence | seed divergence | conditioning divergence |
|---|---|---|---|
| `stable_audio_3_small_music` | 0.049025 | 0.060849 | 0.164266 |
| `stable_audio_3_small_sfx` | 0.040199 | 0.034961 | 0.181044 |
| `stable_audio_3_medium` | 0.023272 | 0.018142 | 0.165492 |
| `stable_audio_3_small_music_base` | 0.060284 | 0.043346 | 0.166105 |
| `stable_audio_3_small_sfx_base` | 0.072861 | 0.068967 | 0.225991 |
| `stable_audio_3_medium_base` | 0.022110 | 0.018060 | 0.178970 |

All three columns are **run-to-run unstable in the fifth decimal on Metal** — reduction order, not a
behaviour change. Measured across three separate `--release` runs of this suite on the same tree, the
`small_music` row reported `source_divergence` `0.049025`/`0.049032`, `seed_divergence`
`0.060849`/`0.060890` and `conditioning_divergence` `0.164266`/`0.164221`. The tables in this section
are one such run and should be read to about four significant figures; every floor below is set far
enough back that this spread cannot move a verdict. Do not treat a digit here as a pin. The third
column is new; see below.

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

### The two floors above are **wrong-signed** against a dropped conditioning handoff

Round-2 review finding, and the reason there is now a third measurement. `local =
edit_local_conditioning(...)` → `let _ = edit_local_conditioning(...)` inside `synthesize_traced` is
one token, and it leaves the DiT's local conditioner as the zero tensor the text-only path
allocates: every inpaint, repaint and extend interior silently becomes plain text-to-audio. Nothing
above could see it, and not by a narrow margin:

* the bit-exact outside is written by `stitch_outside_region` from `prepared`, which never reads
  `local`;
* `source_divergence` and `seed_divergence` both get **larger** when the interior is unconditioned —
  an unconditioned region wanders further from the source and further between seeds — so tightening
  either floor makes the mutation *easier* to pass, not harder;
* `real_repaint_is_byte_identical_to_inpaint` compares two renders that would both be unconditioned,
  so they stay byte-identical;
* `real_extend_...` asserts a non-silent tail, and text-to-audio is non-silent.

So a third measurement was added: **`conditioning_divergence`**, the same region rendered from a
*different source clip* at the identical seed, prompt, region, duration and geometry. If the source
never reaches the DiT the two renders are bit-identical, so the assertion beside the floor is a
byte-inequality — correctly signed, and unfalsifiable-proof in the direction that matters, since the
mutation drives the quantity to exactly zero rather than to something merely small.

| id | conditioning divergence | × source energy |
|---|---|---|
| `stable_audio_3_small_music` | 0.164266 | 1.318 |
| `stable_audio_3_small_sfx` | 0.181044 | 1.453 |
| `stable_audio_3_medium` | 0.165492 | 1.328 |
| `stable_audio_3_small_music_base` | 0.166105 | 1.333 |
| `stable_audio_3_small_sfx_base` | 0.225991 | 1.813 |
| `stable_audio_3_medium_base` | 0.178970 | 1.436 |

The graded floor is `0.70 * source_energy`, `1.88x` below the tightest measured row — the same
relative distance the other two floors take from their own low mode, and set after the measurement,
not before it.

The weight-free half of the same seam is `pipeline::edit_local_conditioning_is_present`, evaluated
in `synthesize_traced` immediately before `sample`. Its call site needs weights, exactly like
`conditioning_is_forwarded`'s, so what the PR lane gates is the **rule** (all four boolean
combinations, plus the agreement between `edit_retained_latent_count` and the tensor
`edit_local_conditioning` actually builds); what the real-weight lane gets is a zeroed handoff that
is refused before any sampling happens instead of rendering plausible audio.

The `observed` half of that rule is `pipeline::tensor_has_nonzero`. Round-2 review found it private,
with the weight-free case **reimplementing** the predicate as `any(|v| *v != 0.0)` rather than
calling it — so mutating the shipped one was invisible to the only lane that could have seen it. It
is now exported `#[doc(hidden)] pub` and driven directly, which puts both of its own mutations in the
weight-free matrix above: `> 0.0` → `>= 0.0` makes the guard vacuous (caught on the degenerate
whole-span row, where the answer must be `false`), and dropping the `.abs()` makes a sign-cancelling
conditioner read as absent (caught by an explicit cancelling tensor added to the same case, since
every other row in it is non-negative and cannot see that one).

### Extend, 10 s → 18 s, 4 steps

Prefix: **exactly** the 441,000-frame prepared source on every id. Output exactly 793,800 frames.

The seam predicate, stated in full rather than as "it is measured": **`seam <= typical`**, where
`typical` is this fixture's own 99.9th-percentile frame-to-frame step over the second before the
seam, `0.163181`. The *yardstick* is measured from the material; the multiplier is `1.0` and that is
a choice. It is the choice it is because a join no sharper than the sharpest transition the source
itself makes is not audible as an edit, while one sharper than that is — the click an extend produces
when the tail is generated without reference to the prefix, or when the stitch boundary slips a
frame. `typical` is separately asserted to be above `0.02`, so a future fixture with a nearly flat
waveform fails loudly here instead of silently tightening the bound into a flake.

**The shipped round-1 predicate was `(typical * 8.0).max(0.05)` and it was unfalsifiable.** At
`typical = 0.163181` that bounds the seam at `1.305` — above the largest step reachable on this
fixture at all, since `source_clip`'s envelope decays to zero at the 10 s mark and PCM here is in
[-1, 1]. Every tail passed it, discontinuous or not. The doc claimed the bound was "measured rather
than chosen" while the assertion multiplied the measurement by a chosen 8 and floored it at a chosen
0.05; that claim was false and is corrected here.

| id | seam step | tail energy |
|---|---|---|
| `stable_audio_3_small_music` | 0.039416 | 0.008523 |
| `stable_audio_3_small_sfx` | 0.040418 | 0.043943 |
| `stable_audio_3_medium` | 0.033704 | 0.012602 |
| `stable_audio_3_small_music_base` | 0.037928 | 0.010388 |
| `stable_audio_3_small_sfx_base` | 0.038057 | 0.036689 |
| `stable_audio_3_medium_base` | 0.045981 | 0.059821 |

These carry the same fifth-decimal Metal instability as the inpaint table above (`0.039416` /
`0.039421` for `small_music`'s seam step across two runs); `typical_step` is `0.163181` on every row
of every run, because it is measured from the fixture rather than from a render.

Every seam step is `0.034`–`0.046` against a bound of `0.163181`, i.e. ~3.5x inside it. That headroom
is what the multiplier `1.0` costs against checkpoint variation, and it is headroom against a bound
that a genuinely discontinuous tail cannot clear — the whole point of dropping the 8x. The prepared
buffer is silent past the source, so a non-zero tail energy is proof the tail came from the model.

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
  step: **12 cases**, measured by running that step's exact command on this tree, not counted by
  eye. `scripts/tests/test_sa3_ci_target_coverage.py` passes; the step's comment counts are updated
  (33 → 48 SA3 weight-free cases across ten named targets). Two of the twelve are round-2's
  (`edit_local_conditioning_is_present` and `edit_geometry_matches_request`); round 3 added no case,
  only assertions inside an existing one, so the count is unchanged from the previous revision.
* **Real weights** — `--test audio_edit -- --ignored` on `sa3-base-identity-metal` and
  `sa3-base-identity-cuda` (profile `sa3-base-identity`), the only jobs provisioning all six pinned
  snapshots. Verified against those jobs' actual `--test` flags rather than their comment blocks.

## Related

* `SC_14547_REFERENCE_AUDIO_RESTYLE.md` — the audio→audio restyle path this one deliberately does not
  duplicate, and the source of `prepare_reference_pcm`, `encode_audio_with_request_rng` and the
  resample-not-reject ruling.
