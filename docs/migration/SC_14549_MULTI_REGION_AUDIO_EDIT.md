# SC-14549 — Multi-region audio editing: an additive gen-core contract extension

Several non-contiguous spans of a source clip regenerated in a **single** pass, sharing one
denoising trajectory. This is not equivalent to N sequential single-region edits: each sequential
pass re-noises and re-decodes, so the spans do not share a trajectory and the seams differ.

Two halves: an additive `gen-core` contract carrier that every provider in the repository rides on,
and Stable Audio 3 support for it.

## The contract shape, and why this one

`Conditioning::AudioEditRegions { audio, mode, regions: Vec<TimeRegion>, strength }`, mapped to a
**new** `ConditioningKind::AudioEditRegions`, plus a borrowed `AudioEditRegionsRef` accessor and a
`GenerationRequest::audio_edit_regions()` getter.

Three shapes were measured against the tree before choosing:

| shape | sites broken | verdict |
|---|---|---|
| add `regions: Vec<TimeRegion>` to the existing `AudioEdit` variant | 14 sites / 3 files | **rejected** — not source-compatible in Rust (every constructor and exact destructuring pattern breaks), and it makes every single-region caller carry a field it must ignore |
| a new variant reusing `ConditioningKind::AudioEdit` + a default-false `Capabilities` flag | `kind()` **+ 81 `Capabilities` literal sites across 70 files**, 37 of them macOS-only | **rejected** — see below |
| a new variant with its **own** `ConditioningKind` | **1 site** | **adopted** |

The 81-site figure is the count of `Capabilities { … }` literals in the repository that spell every
field with no `..Default::default()`; a new field breaks all of them, and 37 compile only on the
macOS lane, so a Linux `cargo check` would be a false green for exactly those.

The deeper problem with reusing the existing kind is not the count. `Capabilities::accepts(kind)`
already rejects any unadvertised kind as a typed `Error::Unsupported`. Reusing `AudioEdit` would let
ACE-Step — the one other provider advertising it — sail straight through that allowlist, so the
capability flag would exist *purely to re-close a hole that choice opened*. With a distinct kind,
**default-deny is free and correct**: every audio provider that has not opted in rejects
multi-region cleanly with no code change, no flag, and no descriptor edit.

### The single deliberate break

`Conditioning::kind()` (`crates/contracts/gen-core/src/generator.rs`) has no catch-all by design —
its own doc calls it the single place a new variant must be classified. It is the **only** site in
the workspace that breaks, verified empirically by removing the new arm and rebuilding: exactly one
`E0004`, in the lib and its test target.

Of the 57 `match` blocks over `Conditioning` repo-wide, the other 56 compile unchanged (47 have a
literal `_ =>`; 2 use a named catch-all already returning a typed error). `ConditioningKind` is
pattern-matched in **zero** match arms, has no serde, and derives only
`Clone/Copy/Debug/PartialEq/Eq`, so adding a variant there has no compile-time blast radius at all.

`Conditioning::AudioEdit`, `AudioEditRef` and `GenerationRequest::audio_edit()` are **untouched**.

## Semantics — this repository's, not claimed as upstream parity

`TimeRegion` gains three questions when it becomes a list. All three are answered explicitly, and
the answers are stated in the `Conditioning::AudioEditRegions` doc comment rather than left to the
provider.

- **Order is not significant.** Regions may arrive in any order and are normalized into a canonical
  union. `[a, b]` and `[b, a]` are the same edit.
- **Overlapping, touching and duplicate regions are accepted, not rejected**, and merged. Refusing
  them would reject requests with an unambiguous meaning and would make the result depend on the
  order the caller happened to write.
- **`end_secs` must be `Some` on every region.** `None` means "to the end of the clip", which is
  only well-defined for a *final* region — and since order is not significant, "final" is not
  well-defined here. Rather than leave that ambiguity, `None` is refused outright. This costs no
  capability: the single-region `Conditioning::AudioEdit` keeps the `None` shorthand unchanged, and
  `AudioEditMode::Extend` stays a single-tail operation on that carrier.

Everything else is inherited: finite bounds, `start_secs >= 0`, `end_secs > start_secs`, plus a
non-empty-list check. Clip-bound and latent-collapse checks stay with the provider, matching the
existing division of labour.

> ### ⚠ No cross-framework parity is claimed
>
> The frozen upstream (`Stability-AI/stable-audio-3`) is an external `/tmp` checkout per
> `SC_14534_SA3_REFERENCE_PARITY.md` and is **not present on this machine**; `inpaint_mask_start_seconds`
> appears nowhere in this tree. The union rule recorded on sc-14549 (integer-sample merge, one mask,
> one trajectory, one decode) could not be substantiated against upstream, so it is implemented and
> documented as **this repository's stated semantics**. sc-15431 tracks the missing conditioning
> oracle. The pinned geometry below is derived from this repo's own landed `adapt_sample_size_for_max`
> and resize rule; that it agrees with the figures recorded on the story is worth noting and is not
> evidence of parity.

## ⚠ The list defeats the finiteness floor's own safety mechanism

`GenerationRequest::first_nonfinite_float` destructures without `..` **deliberately**, so that
adding a *field* breaks the build and forces the author to classify it. That is the mechanism which
has kept the finiteness floor from lagging the request surface.

**A `Vec` defeats it.** The new field satisfies the exhaustive destructure exactly once, and then
hides an unbounded number of floats behind it. A guard written as `regions[0]` — or a loop that a
later refactor turns into `.take(1)` — compiles, passes every pre-existing test, and lets a NaN in
region two flow into the provider's mask rasterisation and poison it silently.

So the floor loops over **every** region, and the gates put their bad value in region **two**, never
region one:

- `sceneworks-gen-core` `every_region_is_floored_not_just_the_first` — NaN/±Inf in region two *and*
  region three, on both bounds, with a well-formed control asserting `first_nonfinite_float() == None`
  so the case discriminates rather than firing on everything.
- `sceneworks-gen-core-testkit` `check_multi_region_audio_edit` — six malformed shapes, every one of
  them in region two.

**Mutation-verified**: changing the loop to `regions.iter().take(1)` fails
`every_region_is_floored_not_just_the_first` at the region-TWO assertion (`left: None`,
`right: Some("conditioning.audio_edit_regions.regions.start_secs")`) and at nothing else.

### The refusal names which region

`first_nonfinite_float` returns a `&'static str` key, so its multi-region arm can only report
`conditioning.audio_edit_regions.regions.start_secs` — with no index. An earlier revision recorded
that as acceptable on the grounds that `Capabilities::validate_request` messages do name the index.
**That was true for range violations only.** The two indexed guards are `r.start_secs < 0.0` and
`end <= r.start_secs`, and **both evaluate `false` for NaN** — so a non-finite bound never reached
an indexed message. For the exact failure mode this whole gate exists to close, the caller of a
ten-region repaint was told a region was malformed but not which one.

`validate_request` therefore runs an **indexed** finiteness pass over `regions`, returning
`Error::Msg` naming `region {i}`. It sits **before** `req.ensure_finite_floats()?` because that call
returns on the first non-finite float anywhere in the request; placed after it, the indexed loop
would be unreachable. No public signature changes.

`ensure_finite_floats` stays intact as the backstop, and both layers are load-bearing: providers
with a bespoke `validate` (flux1's IP-Adapter carve-out, `mlx-gen-flux`) call it directly without
ever entering `validate_request`. This is defence in depth, not a replacement.

**Mutation-verified**: deleting the indexed pass fails
`a_non_finite_region_bound_is_reported_with_its_index` — the request is still rejected, by the
index-free backstop, but the message no longer names `region 1`.

## The testkit gate, and why it is shaped that way

`check_multi_region_audio_edit` joins `audio_conformance` (now 12 checks). A provider that does not
advertise the kind takes the `expect_unsupported` branch — which is how the acceptance criterion
"every non-multi-region audio provider rejects multi-region cleanly" is proven, against the shared
allowlist rather than against any one provider.

The defect it exists to catch is "honours `regions[0]`, silently drops the rest". That defect
produces a *completely plausible* render: one region genuinely regenerated, the rest of the clip
preserved, right length, non-silent, finite, reproducible. Every well-formedness and divergence
measurement passes.

**So the probe is correctly signed** — sc-14548's carry-forward applied before the defect could
recur. Two renders are compared that are identical in seed, prompt, source, mode and `regions[0]`,
differing **only in where region two sits**. Honouring region two ⇒ they differ; dropping it ⇒ they
are byte-identical and the quantity is exactly **zero**. A broken implementation cannot score better
here; it can only collapse to identity.

Non-vacuity is proven by two deliberately-broken stubs in the testkit's own suite:

| stub | defect | caught by |
|---|---|---|
| `first_region_only` | renders from `regions[0]` alone | the moved-region-two probe, by byte-identity |
| `blind_validate` | hand-rolled floor walking `regions[0]` | the region-two shape cases (`open-ended`, NaN, ±Inf, inverted, negative) |

`conformance_panics_on_a_first_region_only_stub` pins that the aggregate suite fails too, not just
the individual check.

## Provider: Stable Audio 3

Advertises `ConditioningKind::AudioEditRegions` on all six ids, alongside `AudioEdit`. **One path,
not a parallel copy** — that is the central design constraint, because a second path is a second
place for the geometry to drift.

- `pipeline::AudioEdit` now carries `regions: Vec<EditRegionSecs>` instead of a
  `start_secs`/`end_secs` pair. Single-region editing is the **one-element case** and travels this
  same code. `model::resolve_audio_edit` resolves **both** carriers into that one shape, so nothing
  downstream can tell them apart — there is no arm for the multi-region case to drift away from, and
  no way for it to quietly reuse only the machinery the single-region case happened to exercise.
- `EditGeometry` carries `spans: Vec<EditSpan>` — the normalized union, **sorted, disjoint and
  non-touching**. There is deliberately **no** promoted `start_sample`/`end_sample` pair: keeping
  the first span in named fields alongside "the others" is exactly what makes a drop-the-rest
  regression easy to write and invisible to read.
- `merge_edit_regions` normalizes at integer 44.1 kHz sample resolution — the resolution the mask is
  *built* at. Seconds would leave two spans that round to the same boundary looking distinct; latent
  resolution would coalesce spans that are genuinely separate in the PCM the stitch preserves.
  Touching counts as overlapping (`start <= last.end`), so the union has one canonical form.
- `edit_keep_mask` zeroes every span plus the padding tail, in **one** mask. `edit_local_conditioning`,
  the SAME encode, the sampler trajectory and the decode all run **once** — that is the capability.
- `stitch_outside_region` preserves everything outside **every** span, walking the sorted union once
  so no frame is visited twice. Any future crossfade therefore lands wholly inside one merged span
  and no frame can receive two fades — structural, not a rule to remember.
- `edit_retained_latent_count` takes a **union** in latent space rather than a sum — see the
  correction below for what that does and does not buy.
- Latent-collapse is checked per **requested** region, not per merged span, so a caller who names one
  usable window and one degenerate one is told about the degenerate one instead of having it silently
  vanish into the union.
- Carrier arity is counted across **both** kinds together. Counting them separately would let the
  *mixed* case — one legacy and one multi-region carrier — through at one apiece, and
  `audio_edit()` / `audio_edit_regions()` are each first-match-only and neither sees the other.
- `AudioEditMode::Extend` is refused on the multi-region carrier as a typed `Unsupported`: there is
  exactly one tail, and the output length an extend implies is the region's end, which is not
  well-defined when order is not significant.
- `strength` is refused on this carrier for the same reason as on the legacy one.
- The `ReferenceAudio` + edit combination refusal now covers both carriers.

### Pinned geometry (this repository's own, cross-checked against the story's figures)

A 20 s source with `[2,6)` and `[14,18)`: adapted `1,146,880`; latent `280`; effective boundary
`216` latents; spans `[88,200, 264,600)` → `[22, 65)` and `[617,400, 793,800)` → `[151, 194)`;
padding-local zero from `216`. Preserved spans `[0,2)`, `[6,14)`, `[18,20)`.

Both single-region pinned examples from sc-14548 still reproduce exactly, unchanged.

## What a single-token edit can still do

Stated, not claimed closed.

| edit | weight-free | catcher |
|---|---|---|
| `merge_edit_regions` `start <= last.1` → `<` (touching no longer merges) | RED | `merge_edit_regions_is_an_order_independent_disjoint_union` |
| the keep-mask loop → `spans.iter().take(1)` | RED | `the_multi_region_geometry_and_keep_mask_cover_every_span` |
| the stitch loop → `spans[0]` only | RED | `the_stitch_preserves_every_span_outside_the_union` (middle preserved span) |
| `edit_retained_latent_count` union → sum | RED *(only after the correction below)* | `the_retained_latent_count_unions_overlapping_latent_ranges` |
| gen-core floor → `regions[0]` | RED | `every_region_is_floored_not_just_the_first` |
| latent-collapse check → first region only | RED | `multi_region_validation_rejects_every_malformed_request_on_every_variant` |
| carrier arity counted per-kind | RED | the mixed-carrier case in the same test |
| `resolve_audio_edit` drops the `AudioEditRegions` arm | RED | `one_region_on_the_new_carrier_is_the_legacy_shape_exactly` |
| the sampler run split into one pass per span | green weight-free | `real_multi_region_inpaint_…`'s `sequential != base` assertion |
| the local-tensor handoff zeroed | green weight-free | `edit_local_conditioning_is_present` (fails closed at runtime) + real-weight |

The last two are honestly green in the PR lane and named rather than papered over, exactly as
sc-14548 recorded its own residue.


## A vacuous test, found by running the mutation rather than trusting the reasoning

The `edit_retained_latent_count` case shipped in an earlier revision of this branch **passed under
the very mutation it was credited with catching**, and the mutation run is what exposed it.

The claim was: `ceil(x/4096)` is not injective, so audio-disjoint spans can land on overlapping
latent ranges, and summing their widths would double-count. The first half is true and the
conclusion does not follow. `merge_edit_regions` leaves the audio spans **disjoint and
non-touching**, `ceil` is monotonic, and a region narrower than one latent frame is already refused —
so for consecutive spans `a2 > b1` gives `ceil(a2/4096) >= ceil(b1/4096)`, i.e.
`start_latent[i+1] >= end_latent[i]`. Latent ranges can touch; they can never overlap.

**Normalized input is exactly the input on which union and sum agree**, so a test that reaches the
function only through `edit_geometry` cannot discriminate between the two formulas, however many
overlapping *requests* it feeds in. All three of the original cases went through the normalizer, and
two of them merged into a single span before the count ever ran.

What the union actually buys is independence from an invariant established in a **different
function**. Delete the merge, or weaken it, and `[2,6)` + `[4,8)` becomes latent `[22,65)` +
`[44,87)`, whose 21 shared positions a sum double-counts — reporting 130 retained where 151 is
correct. So the case now constructs an overlapping-span `EditGeometry` **by hand**, bypassing
normalization, and separately pins the invariant (`spans[i].end_latent <= spans[i+1].start_latent`)
where it is actually established. Re-verified: the sum mutation now fails with
`a sum reports 130 here`.

The general shape, which is the reusable part: **a test that can only construct inputs satisfying an
invariant cannot gate code whose purpose is to survive that invariant breaking.** Reaching the
discriminating input required bypassing the constructor.
