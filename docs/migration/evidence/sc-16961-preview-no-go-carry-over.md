# sc-16961 — the carried-over preview no-go set (epic 16948)

**This document exists to stop a future author spending CUDA-box time re-deriving a negative result
that epic 16624 already paid for.** It settles nothing new. It writes down what was measured, what was
*not* measured, and which candle route rides which finding — so that "should we fit a preview for Wan?"
has an answer in the repository instead of in someone's GPU queue.

Base branch `origin/main` @ `dbb435e8`. **No measurement was run for this story**, and none may be:
running a producer here would be the misread the story names in its own non-goals.

> **If a future method makes any of these viable, it reopens as a NEW story with a NEW measurement.**
> Nothing below is "to be decided". A route leaves this set by being measured past the bar and wired,
> in a story that owns both halves — not by an author concluding the numbers look close enough.

## 1. What epic 16624 measured, and what it did not

The bar is **holdout R² ≥ 0.88**. Three temporal / high-channel latent spaces were measured on real
weights against it and **rejected**. The remaining spaces were closed without measurement.

| Latent space | Fit R² | Holdout R² | Bar | Outcome |
| --- | --- | --- | --- | --- |
| **LTX** (128 ch) | `0.984291` | **`0.618575`** | 0.88 | **no-go** — sc-16638 |
| **Mage** (128 ch, spatial) | `0.938091` | **`0.806216`** | 0.88 | **no-go** — sc-16639 |
| **Mochi** (12 ch) | `0.846932` | **`0.807202`** | 0.88 | **no-go** — sc-16640 |
| **Wan z16** (`vae16::WanVae16`) | *not measured* | *not measured* | 0.88 | **closed under the temporal program gate** — sc-16637 |
| **Wan z48** (`vae::WanVae`) | *not measured* | *not measured* | 0.88 | **closed under the temporal program gate** — sc-16637 |
| **SVD** (temporal 4 ch) | *not measured* | *not measured* | 0.88 | **closed under the temporal program gate** — sc-16636; routed to Tier 3 by sc-16633; candle-side sc-16954 |
| **SeedVR2** | — | — | — | **excluded on shape**, not on a number — no story measured it |

**Read the two R² columns as different things, because they are.** The fit column is in-sample: it is
what the least-squares solution scores on the very renders it was solved over, and it is *always*
flattering. The holdout column is out-of-sample, on renders that never contributed a normal equation,
and it is the only one the 0.88 bar is applied to. LTX is the whole argument in one row: a fit R² of
`0.984291` — better than several *shipped* families score in-sample — collapsing to `0.618575` the
moment it is asked about a render it has not seen. sc-16954 was bounced for comparing an in-sample
number against a holdout one, and this document is precisely where that distinction must be
unambiguous, so every number here is labelled.

Note also what the two 0.80 holdout rows mean. Mage and Mochi are not near-misses to be nudged over
the line by a bigger corpus. **Mochi's fit R² is `0.846932` — below the 0.88 bar in-sample**, so the
linear model does not describe that space even on the renders it was solved over; no corpus size fixes
that. Mage's fit clears the bar at `0.938091` but drops 0.13 out-of-sample, which is the ordinary
overfitting signature rather than a sampling accident.

sc-16636 additionally settled the contract question, and settled it as a **rejection**: `PreviewFrame`
carries a single `Image`. Contact-sheet packing of several latent frames into one image was rejected,
and so was widening the contract to carry a frame index or a clip. What was retained is only a
middle-latent deterministic output-index anchor shape, held in reserve for a future *viable* method.
So a future author who solves the R² problem for a video space still does not inherit a contract that
can carry a video preview — that is a second, separate piece of work.

## 2. The measurement transfers to candle; it is not a backend fact

`RGB_FACTORS` / `RGB_BIAS` are least-squares constants over a **VAE latent space**. They contain no
tensor library. A linear latent→RGB approximation that fails holdout on MLX fails it on candle too:
same VAE weights, same algebra, different matmul implementation. There is nothing for candle to
re-measure, and re-measuring would produce the same rejection at the cost of a CUDA box.

This is the same reasoning the epic's *positive* stories used in the other direction — sc-16950
through sc-16959 reused epic 16624's committed fits rather than refitting them, each after proving in
tensor bytes that candle loads the same VAE. The no-go set is that argument run backwards, and it is
not weaker for it.

## 3. Which candle route rides which finding

19 registered generator ids stay preview-inert. Grouping them by "they are all video" would be an
assumption; the table below is grounded in what each crate actually loads.

| Route id(s) | Crate | Settled by | Basis | Lineage evidence |
| --- | --- | --- | --- | --- |
| `wan2_2_ti2v_5b` | `candle-gen-wan` | sc-16637 | **not measured** — Wan **z48**, closed under the temporal program gate | `src/lib.rs` builds `VaeConfig::ti2v_5b()` → `vae::WanVae`; `src/config.rs` gives that config `z_dim: 48`, `base_dim: 256`, `patch_size: 2` |
| `wan2_2_t2v_14b`, `wan2_2_i2v_14b`, `wan_vace` | `candle-gen-wan` | sc-16637 | **not measured** — Wan **z16**, same program gate | `src/wan14b.rs` and `src/model_vace.rs` build `Vae16Config::wan21()` → `vae16::WanVae16`; that config is `z_dim: 16`, `base_dim: 96` |
| `bernini_renderer`, `bernini` | `candle-gen-bernini` | sc-16637 | **not measured** — same Wan **z16** space | `src/components.rs` builds `WanVae16::new_with_encoder(&Vae16Config::wan21(), …)`; `Cargo.toml` takes `candle-gen-wan` as a path dependency |
| `scail2_14b` | `candle-gen-scail2` | sc-16637 | **not measured** — same Wan **z16** space | `src/pipeline.rs` builds `WanVae16::new_with_encoder(&Vae16Config::wan21(), …)`; `src/generate.rs` holds a `WanVae16` field |
| `ltx_2_3_distilled` | `candle-gen-ltx` | sc-16638 | **measured** — fit `0.984291` / holdout `0.618575` | LTX 128-channel space |
| `mochi_1` | `candle-gen-mochi` | sc-16640 | **measured** — fit `0.846932` / holdout `0.807202` | Mochi 12-channel space |
| `mage_flow`, `mage_flow_base`, `mage_flow_turbo`, `mage_flow_edit`, `mage_flow_edit_base`, `mage_flow_edit_turbo` | `candle-gen-mage` | sc-16639 | **measured** — fit `0.938091` / holdout `0.806216` | Mage 128-channel spatial space |
| `svd_xt` | `candle-gen-svd` | sc-16636 (gate); sc-16633 (routed); sc-16954 (candle) | **not measured** — temporal video space, closed under the program gate | `AutoencoderKLTemporalDecoder`, `latent_channels: 4` with a **temporal** decoder |
| `seedvr2`, `seedvr2_3b`, `seedvr2_7b` | `candle-gen-seedvr2` | — | **not measured, and not measurable in this shape** — one-step super-resolution, not a txt2img denoise | crate docs: "**One-step Euler**"; the input is a low-resolution image or clip, not noise |

### `candle-gen-wan` is two latent spaces, not one — z16 and z48

The single crate registers routes in **two structurally different VAE latent spaces**, and collapsing
them would erase the finding sc-16637 was careful to preserve. Its closure comment reads: "the
registered family spans z16 `WanVae` IDs … and the distinct z48 `Wan22Vae` `wan2_2_ti2v_5b`; a single
fit would never have covered the full surface."

* **z48 — `wan2_2_ti2v_5b` alone.** `candle-gen-wan/src/lib.rs` sets `vae_cfg: VaeConfig::ti2v_5b()`
  and loads `vae::WanVae` (`AutoencoderKLWan`). `src/config.rs` gives that config `z_dim: 48`,
  `base_dim: 256`, `num_res_blocks: 2`, `patch_size: 2`, `conv_out_channels: 12`, and the decoder is
  residual.
* **z16 — the A14B pair and VACE.** `src/wan14b.rs` (`wan2_2_t2v_14b` / `wan2_2_i2v_14b`) and
  `src/model_vace.rs` (`wan_vace`) both set `Vae16Config::wan21()` and load `vae16::WanVae16`:
  `z_dim: 16`, `base_dim: 96`, `dim_mult [1,2,4,4]`.

`vae16.rs`'s own module docs enumerate the differences and are the primary source here: `WanVae16` is
"the temporal VAE used by **both** A14B MoE variants (`wan2_2_t2v_14b` / `wan2_2_i2v_14b`)", and is
"Distinct from the 5B's z48 `crate::vae` `AutoencoderKLWan` on three structural axes" — z16/base 96 vs
z48/base 256, non-residual vs `is_residual`, and no spatial patchify vs a 2×2 unpatchify, giving 8×
rather than 16× spatial scale.

Neither space was measured, so the disposition is the same either way. What differs is the recorded
lineage, and that is exactly why nothing else would have caught it being wrong: a future author who
trips the no-go assertion on `wan2_2_ti2v_5b` must be told about `vae::WanVae`, not about a VAE that
route never loads. `the_wan_routes_are_recorded_in_the_latent_space_their_provider_builds` asserts the
id→space assignment **and** re-checks it against those provider files, so the record cannot drift from
the sources it cites.

### Bernini and Scail2 inherit Wan **z16** — the *unmeasured* row, not a measured one

Both are in the Wan orbit, and it would be easy to wave at "temporal, therefore rejected like LTX".
That would attach them to a number that was never measured for their space. What the sources say:

* `candle-gen-bernini` depends on `candle-gen-wan` by path and its `Components::load` constructs
  `WanVae16::new_with_encoder(&Vae16Config::wan21(), …)`. The crate's own docs state the renderer **is**
  Wan2.2-T2V-A14B, finetuned, reusing the z16 VAE, the UMT5 text encoder, the dual-expert
  `WanTransformer`, the flow/UniPC scheduler and the RoPE table wholesale.
* `candle-gen-scail2` depends on `candle-gen-wan` by path and its `Scail2Pipeline` construction and
  `Components` both hold a `WanVae16`. SCAIL-2 is Wan2.1-14B I2V.

So both occupy **literally the Wan z16 latent space** — the same VAE code over the same weight family
— and therefore ride **sc-16637**, which closed Wan under the temporal program gate **without
measuring it**. Neither inherits LTX's `0.618575`, Mage's `0.806216`, or Mochi's `0.807202`. Quoting
one of those for Bernini or Scail2 would be a fabricated provenance, which is the failure mode this
row exists to prevent. Their status is "closed, unmeasured", and if a future story wants a number for
the Wan z16 space it must go and get one — and a z16 number would still say nothing about z48.

### SVD is excluded on its latent space, not on a holdout number — and not by sc-16637

**Which story closed SVD, stated precisely, because the obvious guess is wrong.** sc-16637 is "Tier 3:
fit the Wan latent space and wire wan — check bernini, scail2, krea-realtime". Neither its description
nor either of its comments mentions SVD at all; citing it here would send a reader to a story with
nothing about SVD in it. The actual chain is:

* **sc-16633** routed it there — "Route svd to Tier 3" in its scope, and its preflight comment records
  "`mlx-gen-svd` is registered video with temporal latents and remains false pending sc-16636's
  temporal contract".
* **sc-16636** is the program gate that closed it: its final decision declares "NO-GO for the remaining
  Tier 3 rollout stories" on the strength of the LTX 128-channel measurement, having first settled what
  a `PreviewFrame` could even mean for a temporal latent.
* **sc-16954** is the candle-side adjudication — it enumerated SVD alongside kolors and instantid and
  left it inert without re-measuring anything.


`svd_xt` runs a genuine multi-step EDM denoise (it advertises the curated sampler menu), so it is not
excluded structurally the way SeedVR2 is. It is excluded because its latent space is a temporal video
space that nobody measured: `AutoencoderKLTemporalDecoder`, `latent_channels: 4`, where the decoder is
temporal even though the encode is spatial. Two consequences worth stating so they are not
rediscovered:

* Its four channels are **not** SDXL's four channels. It is a different checkpoint, and sc-16957 /
  sc-16958 already established for the 16-channel spaces that a matching channel count proves nothing
  (see §4). Reaching for `candle_gen_sdxl::preview` here would be exactly that mistake at 4 channels.
* Even if a per-frame encoder-side fit existed, the function a preview approximates is the *decode*,
  and SVD's decode is temporal — it mixes frames. A per-frame linear map is not an approximation of
  that function, so the ordinary fit shape does not apply without new work.

### SeedVR2 is excluded on its shape — no holdout number was ever measured, and none is quoted

`seedvr2` / `seedvr2_3b` / `seedvr2_7b` are a **restoration / super-resolution upscaler**, not a
txt2img denoise. The crate docs name it a "one-step diffusion-transformer super-resolution upscaler"
running **one-step Euler**, with a precomputed negative-prompt embedding and no runtime text encoder;
the input is a low-resolution image (`Reference`) or clip (`VideoClip`), not noise.

A per-step preview needs a multi-step progression from noise toward an image. A one-step route has no
progression: the only frame that could ever be emitted is the finished output, which the caller is
about to receive anyway. That is the reason, and it is a structural one. **No holdout R² was ever
measured for SeedVR2 and none is quoted here** — citing a number for it would be inventing evidence.

## 4. Four VAE-relation shapes this epic actually observed

A future author deciding whether to measure will first ask "is this space one we already have?" This
epic answered that question ten times and found **four distinct relation shapes**. The lesson in both
directions: *same channel count never implies same latent space, and a different file does not imply a
different one.* Nothing short of comparing tensor bytes settles it.

| # | Relation | Instance | Measurement |
| --- | --- | --- | --- |
| 1 | **Identical file** | Qwen-Image lanes; Z-Image ↔ FLUX.1-dev (`f5b59a26…40a3`, whose `vae/config.json` names `flux-dev` as its origin); Krea Turbo ↔ Raw | one SHA-256 |
| 2 | **Different file, same tensors** | Anima ↔ Qwen after the production key rename `convert_vae_key`: **194 of 194** tensors, 126,892,531 values, bit-identical, both bf16. Boogu ↔ FLUX.1: a different f32 container (`8c717328…4c94`, 244 tensors) whose **244 of 244** tensors, 83,819,683 values, round — round-to-nearest-even — exactly onto FLUX.1's bf16 bits, same key set, same `vae/config.json` | tensor walk after rename / after cast |
| 3 | **Same everything except the numbers** | SD3.5 vs FLUX.1-dev: same architecture, same 167,666,902-byte container, same 244 bf16 keys at the same shapes — and **0 of 244** tensors identical, 244 differing over 83,819,683 values | tensor walk |
| 4 | **Partial overlap** | Sana base vs Sprint: 375 tensors / 312,250,275 values → **320 identical, 55 differing**, and every one of the 55 is decoder-side; the entire 179-tensor encoder is identical | tensor walk |

Shape 3 is the one that would have caused a wrong reuse if anybody had trusted architecture, container
size, key set, shape and dtype — all five agreed and the space was still different. Shape 4 is the one
that would have caused a wrong reuse if anybody had trusted "the encoder matches": Sana base and
Sprint share an encoder and still need two separate committed fits, because a preview approximates the
*decode*.

Sharing a Rust *type* proves the least of all. `z_image::vae::AutoEncoderKL` is the type behind
Z-Image, SD3.5 and Boogu, and those are two different latent spaces plus a third file that turns out
to be one of them.

## 5. Two more traps a re-measuring author would hit

**SenseNova's `0.99998292` is not a benchmark for anything.** sc-16960 is the epic's one genuinely new
fit and it scored a holdout overall R² of `0.99998292` — orders of magnitude above every other family.
That is not evidence that its latent space is unusually well-behaved. SenseNova-U1 has **no VAE at
all**: it denoises in pixel space, the running state of its loop *is* the image in `[-1, 1]`, and its
"decode" is an affine map. So its fit is 3-channel and its R² is close to arithmetic rather than close
to evidence — it measures how well a linear map approximates a linear map. Do not read it as a bar, a
target, or proof that a hard space might come out fine after all.

**Rail-clipping is not reliably the discriminating statistic.** Both sc-16959 and sc-16960 found cases
where a scaling error *shrinks* the latent rather than saturating it, collapsing the projection toward
the fit's own intercept. sc-16959 measured the uncorrected Sana Sprint prior at a rail-clipped
fraction of `0.0000` — **lower** than the correctly scaled one at `0.0003` — while its spread about the
intercept was `12.96` against the corrected `25.93`. A rail-fraction check would have passed the wrong
frame. The statistic that discriminates is **contrast about the intercept**. Any future validation of
a preview projection should bound that, and treat a rail fraction as something to record rather than
something to gate on.

## 6. What is executable, and where

The record above is enforced by tests in `crates/media/candle-gen/candle-gen-catalog/src/lib.rs`, in
the `preview_advertising` module:

| Test | What it catches |
| --- | --- |
| `temporal_and_super_resolution_routes_stay_outside_preview_advertising` | a no-go id starting to advertise `supports_preview`, or appearing in the wired allowlist — and it now names the settling story and the reason in the failure message, rather than failing on a bare boolean |
| `the_no_go_set_and_the_wired_set_partition_every_shipped_route` | a newly registered route that is in none of the three classes, a route that quietly changes class, and the no-go set going stale as the catalog grows |
| `no_no_go_family_acquires_a_preview_fit_or_a_fit_producer` | an `RGB_FACTORS` / `RGB_BIAS` / `LATENT_RGB` / `LATENT_TO_RGB` constant, **any `[[f32; 3]; N]` fit table whatever it is named**, a `PreviewHook` / `PreviewSink` / `project_latents` / `emit_preview` reference, a `src/preview.rs`, or a producer under `tests/` named `fit_*.rs` or `*preview*.rs`, appearing in any of the eight no-go crates — i.e. someone starting the work this document says not to start |
| `the_recorded_no_go_measurements_stay_labelled_fit_versus_holdout` | the numbers above being edited into an unlabelled or swapped form, **and any variant other than LTX/Mage/Mochi acquiring a `(fit, holdout)` pair at all** |
| `the_wan_routes_are_recorded_in_the_latent_space_their_provider_builds` | the z16/z48 assignment being edited or drifting from the provider sources it cites |

`no_no_go_family_acquires_a_preview_fit_or_a_fit_producer` carries **positive controls**: the marker
scan is run against `candle-gen-flux` (which has a committed fit) and must trip on **both** the name
`RGB_FACTORS` and the shape `[[f32; 3]; 16]`; the producer scan is run against `candle-gen-sensenova`
(which has `tests/fit_preview_rgb.rs`) and must trip too. A detector that silently stopped matching
would otherwise read as "no no-go crate has a fit" forever.

**Honest limits of that scan.** The marker list is exact substrings, so on its own it is a *name*
heuristic — `tests/fit_ltx_rgb.rs` holding a `LATENT_TO_RGB` table, `src/rgb_projection.rs` holding a
`LATENT_RGB` one, and `tests/preview_fit.rs` all slipped past the original four filenames and eight
names. Two things close that: the **shape** check matches `[[f32; 3]; N]` regardless of the constant's
name (excluding `N == 3`, which is an ordinary 3×3 kernel — `candle-gen-seedvr2/src/color.rs` ships
one, and no no-go space has three latent channels), and the loosened producer patterns under `tests/`.
All three evasions above were planted and confirmed caught. What is still **not** caught is a fit
stored in some other representation entirely — a `Vec<Vec<f32>>`, or coefficients read from a data
file. That is well past "someone has started the work", and anything that actually *emits* still has
to touch `PreviewHook` / `PreviewSink` / `project_latents`, which are matched by name.

The bare word `preview` is deliberately not a marker: **seven of the eight no-go crates contain the
token today** — all but `candle-gen-mage`, whose `Capabilities` literal ends `..Default::default()`
and so never writes a `supports_preview` line. The other seven do, plus training preview samples
(sc-8650) and Mochi's repo id `genmo/mochi-1-preview`. A substring match on it could never have been
written.

## 7. The three classes, at `dbb435e8`

The catalog registers **51** generators. Every one is in exactly one class, and the partition test
above is what keeps that true:

| Class | Count | Meaning |
| --- | --- | --- |
| **wired** — `PREVIEW_ROUTE_IDS` | **29** | emits per-step previews and advertises `supports_preview: true` |
| **no-go** — `PREVIEW_INERT_ROUTE_IDS` | **19** | this document |
| **deferred but viable** — `PREVIEW_DEFERRED_ROUTE_IDS` | **3** | `boogu_image`, `boogu_image_turbo`, `boogu_image_edit` |

The deferred class is the reason the no-go set must not be "everything that does not advertise". Boogu
does not advertise previews today, and it is **not** a no-go: sc-16956 proved its VAE *is* FLUX.1's, so
its space already has a committed fit that clears the bar. It is unwired only because the wiring has
not been done, tracked as **sc-17218**. Filing it under "rejected" would lose a viable family; leaving
it out of every list would let the no-go set grow stale unnoticed.

**This story moves the wired count by zero.** It measures nothing, wires nothing, and flips no
descriptor. 29 before, 29 after.

## 8. Follow-ups

* **sc-17218 — Boogu previews.** Reuse `candle_gen_flux::preview`'s 16-channel fit (sc-16956 proved
  the VAE identity) and wire the three `run_flow_sampler` sites sc-16955 enumerated. Not this story's
  work, and explicitly not a no-go.
* **sc-17309 — two committed fits over one latent space.** `mlx-gen-flux/src/preview.rs` and
  `mlx-gen-z-image/src/preview.rs` are independent OLS solutions over a byte-identical VAE. Collapsing
  them moves MLX preview bytes on whichever family loses its fit, so it belongs to a story that owns
  both engines. sc-16958 established that SD3.5 must **not** gain a row there: its space is genuinely
  distinct (shape 3 above).

Nothing else was discovered that belongs to another story.
