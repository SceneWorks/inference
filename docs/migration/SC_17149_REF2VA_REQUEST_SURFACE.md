# SC-17149 — `Conditioning::ReferenceVideo`: giving a reference clip a request-surface carrier

MiniMax-H3's `ref2va` task takes an ordered, heterogeneous reference list. Two of its three
modalities already had a gen-core carrier — `Conditioning::Reference` for an image,
`Conditioning::ReferenceAudio` for a waveform. The third, a **video reference**, had none, and was
routed through `Conditioning::VideoClip`. This adds the missing carrier and completes the triple.

A reference video may carry **its own soundtrack**, conditioned on as that reference's own,
rotary-aligned with its video rows and sharing their origin. That is a different request from the
same waveform sent as a standalone `ReferenceAudio`, which takes its own rotary slot and consumes one
of the three standalone audio-reference cap slots.

## The shape, and why this one

`Conditioning::ReferenceVideo { frames: Vec<Image>, fps: f32, audio: Option<AudioTrack> }`, mapped to
a **new** `ConditioningKind::ReferenceVideo`. The payload is the engine type
(`mlx_gen_minimax_h3::reference::VideoReference`) verbatim.

Three shapes were measured against the tree before choosing:

| shape | sites broken | verdict |
|---|---|---|
| add `audio: Option<AudioTrack>` to the existing `VideoClip` variant | **9 errors / 7 files** | **rejected** — not source-compatible, and it closes only one of the two holes (see below) |
| document the standalone-only limitation as intended | 0 | **rejected** — the engine implements the aligned form and the packer consumes it; the gap was request vocabulary only |
| a new variant with its **own** `ConditioningKind` | **1 site** | **adopted** |

The 9-error figure is measured, not estimated: a probe field was added to `VideoClip` and the tree
rebuilt. `sceneworks-gen-core` breaks at 3 sites (one `E0027` exhaustive destructure in
`video_clips()`, two `E0063` constructions); with those patched, `mlx-gen-krea-realtime` breaks at 3
(across `src/pipeline.rs`, `tests/generate_smoke.rs` and `tests/style_lora_real_weights.rs`),
`mlx-gen-bernini` at 1, `candle-gen-bernini` at 1, and `mlx-gen-minimax-h3` at 1. `mlx-gen-ltx` and
`mlx-gen-seedvr2` match with `..` and are unaffected. **Five of the seven files compile only on the
macOS lane**, so a Linux-only `cargo check` would be a false green for all but two of them.

### The single deliberate break

`Conditioning::kind()` (`crates/contracts/gen-core/src/generator.rs`) has no catch-all by design —
its own doc calls it the single place a new variant must be classified. It is the **only** site in
the workspace that breaks, verified empirically by removing the new arm and rebuilding: exactly two
`E0004`, both pointing at the `kind()` match itself — one for the lib target and one for its test
target.

Statically: of the **49** `match` blocks over `Conditioning` repo-wide, **48** carry a catch-all arm
and compile unchanged; the one that does not is `kind()`. `ConditioningKind` has **no `impl` blocks
at all** — no `Display`, no serde — and is pattern-matched only inside `matches!` guards, so adding a
variant there has no compile-time blast radius. `Conditioning` derives only `Clone, Debug` and is not
serialized anywhere in this repository, so there is no serialized-field compatibility question on the
inference side; the SceneWorks payload work rides sc-17160 and mode reachability rides sc-17159.

`Conditioning::VideoClip` and `Conditioning::VideoSync` are **untouched**.

## Why `VideoClip` could not be the carrier

`VideoClip` is the LTX in-context latent-append path. Its *latent handling* does describe a reference
block — VAE-encoded, appended as extra rows, never written by the denoise loop — which is why the
first revision of `request_references` rode it. Its *payload* is the problem.

**Two of its three fields are unusable by a reference, and were rejected rather than ignored:**

- `frame_idx` is a position in the generated timeline. A reference has none — that is precisely what
  separates it from a keyframe. The old code required `0`.
- `strength` is a `1 − strength` denoise mask. Reference rows are fully pinned at the checkpoint's own
  conditioning timestep (`KEYFRAME_NOISE_AUG` for visual rows, `REFERENCE_AUDIO_TIMESTEP` for audio),
  never caller-selectable. The old code required `1.0`.

That alone is a poor vocabulary — two of three fields are traps a provider must reject — but it is not
the deciding argument. The deciding argument is what `VideoClip` **cannot carry at all**:

- **The clip's own soundtrack.** No audio field, so a soundtrack could only arrive as a separate
  `ReferenceAudio` — legal, but a different request silently substituted for the intended one.
- **The clip's own frame rate.** No rate field. This one turned out to be worse than a missing
  capability; see below.

## ⚠ The rate hole was a live defect, not just a gap

`VideoReference::fps` is required data rather than a hint because MiniMax-H3 resamples every reference
onto its own 24 fps by dropping and duplicating whole frames — a clip whose real rate was lost is
conditioned on **at the wrong speed with nothing raising**.

`VideoClip` has no rate, so the old mapping read the request-level `req.fps`:

```rust
fps: req.fps.map_or(crate::denoise::MINIMAX_H3_FPS, f64::from),
```

But `request_geometry` independently **rejects** `req.fps` unless it is exactly `MINIMAX_H3_FPS`:

```rust
if let Some(fps) = req.fps {
    if f64::from(fps) != crate::denoise::MINIMAX_H3_FPS { return Err(...) }
}
```

Between them, a reference's declared rate could only ever resolve to `24.0` — `None` defaulted to it,
`Some(24.0)` was it, and every other value was a hard error on a *different* field. So a 30 fps
reference clip was normalized as though it were 24 fps: `normalize_reference_clip` took its
`(fps - target_fps).abs() < f64::EPSILON` early return, dropped no frames, and conditioned the render
on a clip playing 25% fast. No error, no warning, plausible output.

The two rates are genuinely different quantities and conflating them is what created this:
`GenerationRequest::fps` is the rate of the **generated output**; a reference's is the rate of
**supplied input media**. For a variant whose defining property is that it does not bind the output
timeline, a model may legally reject an output rate it happily accepts as an input rate — which is
exactly what MiniMax-H3 does. The new carrier therefore puts the rate on the variant. This does not
contradict the single-source-of-truth argument `Conditioning::VideoSync` makes for reading `req.fps`:
there the clip *drives* the output timing, so it is the same quantity.

### The resample branch had no test, because it was unreachable

`normalize_reference_clip`'s resampling branch had **zero** coverage in the crate. That was not an
oversight in isolation — through the request surface the branch could not be reached, since `fps`
always equalled the target. Making it reachable is this change's doing, so the gate is added here:

`a_clip_is_resampled_from_the_rate_it_declares` (`src/reference.rs`) pins all three cases with
solid-colour tagged frames, asserting *which* source frames survive rather than only how many —
24 fps → identity `[10,20,30,40,50]`; 30 fps → frames **dropped**, `[10,20,40,50]`; 12 fps → frames
**duplicated**, `[10,10,20,20,30,30]` — plus that every output frame is a held source frame and never
an interpolation.

## Gates

In `sceneworks-gen-core` (`generator.rs`), six cases:

- `reference_video_maps_to_its_own_kind` — the discriminant is `ReferenceVideo`, and the variant is
  **not** collected by `video_clips()` / `keyframes()` / `control_clip()`.
- `reference_video_accepted_when_advertised` — with and without a soundtrack.
- `reference_video_unsupported_on_a_non_advertising_model` — a provider advertising `VideoClip` but
  not `ReferenceVideo` still rejects a reference, typed `Error::Unsupported`. This is the arm that
  makes separating the two kinds worth anything. It also pins the **layer ordering** (see below).
- `reference_video_empty_frames_is_a_msg_range_error`.
- `reference_video_rejects_a_rate_that_has_no_reading` — `0.0`, `-24.0`, NaN and `+Inf` are all
  refused, with a well-formed `30.0` control so the case discriminates rather than firing on
  everything.
- `reference_video_rate_joins_the_finiteness_floor` — `first_nonfinite_float` names
  `conditioning.reference_video.fps`.

The rate is checked for **positivity** and not only finiteness, unlike every other conditioning float
the floor owns. Those feed denoise math where `0.0` is a meaningful inert value; a rate of zero or
below has no reading at all — it makes the resample stride undefined or negative, so the frames the
model would then read are arbitrary rather than merely unweighted. So the rate is refused by two
different layers, and which one fires depends on the value: NaN and `±Inf` are caught by
`ensure_finite_floats`, `0.0` and negatives by `validate_request`'s range pass.

### The floor precedes the conditioning allowlist

An assumption made while writing these gates was that the capability verdict always outranks the
payload verdict — that a malformed reference on a model which does not admit the kind reports
`Unsupported`. **That is true for payload *shape* and false for non-finite floats**, and the test run
is what corrected it: `ensure_finite_floats` runs ahead of the allowlist for every float in the
request, so a NaN rate on a non-advertising model reports
`Msg("conditioning.reference_video.fps must be finite (got NaN)")`, not `Unsupported`.

This is pre-existing gen-core ordering that every other conditioning float already shares, and it was
**not** changed here — reordering the floor would alter the error every existing provider returns for
every non-finite value. Both behaviours are now pinned in
`reference_video_unsupported_on_a_non_advertising_model` (empty frames → `Unsupported`; NaN rate →
the floor's `Msg`), so a future reordering is a visible decision rather than a silent change in which
error a caller sees.

In `mlx-gen-minimax-h3` (`src/model.rs`), three cases:

- `an_in_context_video_clip_is_refused_as_unsupported` — `ConditioningKind::VideoClip` was **dropped**
  from the descriptor, so default-deny turns an in-context clip into a typed `Error::Unsupported`
  rather than letting it through to a model with no in-context clip mechanism.
- `a_reference_clip_carries_its_own_frame_rate` — a 30 fps reference survives the mapping as 30 fps
  *while the request's own `fps` stays unset*, and validates.
- `a_reference_clips_own_soundtrack_rides_the_clip` — the soundtrack is on the video reference, the
  list has one entry not two, and `audio_count()` is `0`; contrasted against the standalone form,
  which yields two references and `audio_count() == 1`.

**Mutation-verified**, each failing exactly one test and nothing else:

| mutation | result |
|---|---|
| `fps: f64::from(*fps)` → `fps: MINIMAX_H3_FPS` (restore the old pinning) | `a_reference_clip_carries_its_own_frame_rate` FAILED |
| `audio: audio.clone()` → `audio: None` on the `ReferenceVideo` arm | `a_reference_clips_own_soundtrack_rides_the_clip` FAILED |

> ### ⚠ No real-weight validation of this change
>
> The `MiniMaxAI/MiniMax-H3` snapshot is present on this machine, but the render-path tests that
> would exercise a reference end-to-end load the 66 GB `transformer_ref` partition and are `#[ignore]`d
> behind `MINIMAX_H3_SNAPSHOT`. This change is request-surface plumbing whose downstream — the packer
> and the audio-row encode that consume `VideoReference::{fps, audio}` — landed and was gated under
> sc-17149 itself (`tests/ref2va_conditioning.rs`, which covers the audio-bearing clip geometry
> directly). The claim made here is that the mapping now carries both fields to that already-gated
> code, and that is what the tests above pin. **No claim is made that a reference-with-soundtrack
> render has been executed against real weights.**

## ⚠ The tiering seam: `ref2va`'s partition follows the tier, not the root

Landed when this branch merged sc-17150 (q4/q8/bf16 quant tiers, PR #570). The two stories touch the
same call sites from opposite directions and the textual merge did **not** surface the whole overlap.

sc-17149 gives the DiT a *per-task* partition: `t2va`/`fl2va` read `transformer/`, `ref2va` reads
`transformer_ref/`. sc-17150 makes the DiT a *tiered component*: a `q4` install stages it from the
`SceneWorks/minimax-h3-mlx` mirror, **outside the snapshot root**, so `MiniMaxH3Dit::load(root,
partition, …)` no longer names it at all and sc-17150 replaced that seam with `load_dir(dit_dir, …)`.

Resolved with `MiniMaxH3::task_dit_dir(task)`, which resolves the reference partition as
`dit_dir`'s **sibling**:

| layout | `dit_dir` | `ref2va` resolves to |
|---|---|---|
| flat upstream | `root/transformer` | `root/transformer_ref` — unchanged from before tiering |
| tiered / split | `{tier}/transformer` | `{tier}/transformer_ref` — same tier, both partitions |

This is the shape sc-17150 already anticipated rather than a new invention: `crate::convert` and
`examples/minimax_h3_prequant.rs` both document pre-quantizing "`transformer/` **or**
`transformer_ref/`", so a tier directory holds the pair side by side. The load-time probe and
`reconcile_tier` were moved to the reference partition's resolved location for the same reason.

### What the clean merge hid

`generate` has **two** DiT load sites — the `t2va`/`fl2va` arm and the `ref2va` arm — and only the
first conflicted. The `ref2va` arm merged with no marker while still resolving against `&self.root`.
On a tiered install that resolves to a path the staged tier does not occupy; worse, on a machine
that also has a flat snapshot it would find a **dense** `transformer_ref` beside a `q4` base and mix
tiers across two branches of one render, which nothing downstream checks. Both sites now go through
`task_dit_dir`.

This is the concrete reason the merge was re-verified rather than trusted: a conflict-free hunk is
not evidence that the surrounding invariant survived.

### The source gate was restated, not relaxed

`every_dit_load_site_is_driven_by_the_task` counted raw `"transformer"` occurrences in `model.rs` and
required exactly one (`BASE_DIT_PARTITION`'s value). sc-17150 legitimately added a third —
`DIT_COMPONENT` — so that count was measuring the wrong thing. It now asserts the property that
actually matters: every occurrence of either partition literal is a `pub const` binding and none sits
at a call site, comment lines excluded so the gate tracks code rather than prose. It also newly
asserts that `task_dit_dir` switches on `MiniMaxH3Task::partition()`, because a source scan that
counts a helper cannot see whether the helper is correct.

**Mutation-verified**, each guard alone, each reverted byte-identical:

| mutation | result |
|---|---|
| `ref2va` load site back to `load(&self.root, task.partition(), …)` | `every_dit_load_site_is_driven_by_the_task` — "1 of 2 site(s) do not" |
| `task_dit_dir` resolves `self.root.join(partition)` | `the_reference_partition_follows_the_tier_not_the_root` — the tiered case fails |
| `task_dit_dir` switches on the task enum instead of `partition()` | `every_dit_load_site_is_driven_by_the_task` — the `partition()` assertion fails |
| a bare `"transformer_ref"` returns to a call site | "appears in 2 non-comment line(s) but only 1 are `pub const` bindings" |

> No real-weight validation: confirming a `q4` `ref2va` render reads `{tier}/transformer_ref` needs a
> staged tier plus the 66 GB reference checkpoint. `the_reference_partition_follows_the_tier_not_the_root`
> pins the path arithmetic on both layouts; the real-weight assertion belongs with sc-17150's tier
> smokes.
