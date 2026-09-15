# sc-16952 candle Qwen-Image latent-preview evidence

Epic 16948 wires `PreviewSink` into the candle engines. Qwen-Image is Tier 1: it reuses the QwenVae
RGB fit epic 16624 committed on the MLX side and adds **no** new fit. This file records what makes
that reuse legitimate, where the projection has to happen on this family, and what the real-weight run
on the CUDA box actually showed for all three shipped lanes.

## Route surface — three lanes, one advertised id

`git grep -n 'run_flow_sampler(' crates/media/candle-gen/candle-gen-qwen-image/src` returns exactly
three call sites. There is no `run_curated_sampler` / `run_scm_sampler` site and no bespoke denoise
loop, so every lane opts in the sc-16949 way: a projector closure handed to the shared driver, with no
loop restructured.

| lane | site | reached by | descriptor id |
| --- | --- | --- | --- |
| base txt2img | `lib.rs` — `Pipeline::denoise_and_decode` | registry | `qwen_image` |
| reference edit | `edit.rs` — `QwenEdit::denoise_and_decode` | bespoke, worker drives by name | *(none)* |
| 2512-Fun ControlNet | `control_fun.rs` — `QwenFunControl::generate` | bespoke, worker drives by name | *(none)* |

Two things worth stating because they change what "cover every route" means here:

- **The base t2i site is in `lib.rs`, not `pipeline.rs`.** `pipeline.rs` owns latent geometry
  (`unpack_latents` / `pack_latents` / `create_noise`) and the σ schedules; it drives no sampler.
  `preview::tests::the_geometry_module_drives_no_sampler` pins that as a negative, so a route added
  there cannot slip past an inventory that names three files.
- **Routes are not ids.** This crate registers one generator descriptor. Edit and ControlNet/Fun are
  bespoke providers, so each gains a `preview: PreviewSink` field on its request type — the shape
  sc-16950 used for `Krea2ControlRequest` — rather than a second id. `PREVIEW_ROUTE_IDS` therefore
  gains exactly one entry while the sc-16951 route inventory carries all three.

There is no dark site: this crate has no trainer and no second denoise loop.

## Projection runs AFTER the unpack — verified, not assumed

Qwen-Image is one of only two candle families with a packed latent seam, and this is where the story
could have gone wrong silently. All three lanes denoise in the **packed token** space: `create_noise`
samples `[1, (H/16)·(W/16), 64]` and the running latent handed to the preview hook stays there until
the decode tail unpacks it. The committed fit is defined over the *spatial* VAE latent
`[1, 16, H/8, W/8]`.

Projecting the packed sequence directly does not merely mis-colour a frame: `[1, seq, 64]` is rank 3,
so it fails the shared `[1, C, h, w]` contract outright and **every** frame would be swallowed as a
projection failure — a preview that shows nothing, which is indistinguishable from the bug this epic
exists to fix. `preview::project_packed_latents` therefore runs `pipeline::unpack_latents` first, the
same inverse patchify the decode tail already applies before the VAE.

Pinned by three separate rows rather than by reading the source:

- `a_packed_latent_is_not_projectable_without_the_unpack` — the spatial projector rejects the packed
  latent, so "just call the spatial projector" reads as the mistake it is.
- `packed_projection_equals_the_spatial_projection_of_the_unpacked_latent` — the packed projector is
  exactly unpack-then-project, byte for byte; the constants have one code path.
- `packed_projection_is_vae_latent_resolution_not_token_grid_resolution` — frames come out at
  `H/8 × W/8`, asserted **unequal** to the `H/16 × W/16` token grid so the two cannot be confused.

The real-weight rows carry the same assertion, so the property is checked against actual renders as
well as against synthetic latents.

## The fit is reused, not refitted

`crates/media/candle-gen/candle-gen-qwen-image/src/preview.rs` carries `RGB_FACTORS` / `RGB_BIAS`
transcribed verbatim from `crates/media/mlx-gen/mlx-gen-qwen-image/src/preview.rs`. There is
deliberately no candle producer: a second least-squares solve of the same latent space would be a
second source of truth for one set of numbers. The MLX producer
(`mlx-gen-qwen-image/tests/fit_preview_rgb.rs`) remains the only way these constants are re-derived.

## Learned-basis transfer — grounded in tensor bytes

The claim being checked is not "both crates name a type `QwenVae`" but "every candle Qwen-Image lane
loads the VAE weights the fit was measured against". Here that lands **stronger** than it did for Krea
(sc-16950), which needed a tensor-by-tensor argument because its published `vae/` is an f32 container
of values the fit donor stores as bf16. There is no container difference to argue past on this family:
every snapshot publishes the identical file.

| snapshot | revision | `vae/…safetensors` SHA-256 | `vae/config.json` SHA-256 |
| --- | --- | --- | --- |
| `Qwen/Qwen-Image-2512` | `25468b98e3276ca6700de15c6628e51b7de54a26` | `0c8bc8b7…d0a8344` | `c448160d…7b56a65` |
| `Qwen/Qwen-Image-Edit-2511` | `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9` | `0c8bc8b7…d0a8344` | `c448160d…7b56a65` |
| `SceneWorks/qwen-image-mlx` `q4/` and `q8/` | `8080a4171f1c8b7fca6c30491eafbe6ffab754bf` | `0c8bc8b7…d0a8344` | `c448160d…7b56a65` |
| `SceneWorks/qwen-image-edit-2511-mlx` `q4/` and `q8/` | `0dfbf3a018bcee42d77de14494c35f97a7531def` | `0c8bc8b7…d0a8344` | `c448160d…7b56a65` |

Full digests:

- weights `0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344` (253,806,966 bytes)
- config `c448160dba5ce79c965cb075ee02e18d1c42eb6424f787e5869790d577b56a65`

Row three is the fit donor itself — the snapshot epic 16624 measured `RGB_FACTORS` / `RGB_BIAS` on.
So the transfer is not "equal values in a different container", it is the same bytes.

- **Every tier keeps the VAE dense.** The packed q4 / q8 tiers quantize the DiT only, so the reuse
  holds on the quantized lanes the desktop app actually ships, not just on the bf16 reference.
- **`latents_mean` / `latents_std`** — the per-channel de-normalization that *defines* the normalized
  16-channel space the fit lives in — are identical, which the byte-identical `vae/config.json`
  already implies and which the rows also assert from the parsed JSON.
- **The ControlNet/Fun lane has no VAE of its own.** `QwenFunControl::load` builds both its `QwenVae`
  decoder and its `QwenVaeEncoder` from `QwenFunControlPaths::qwen_base`, and the
  `alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union` overlay (and the `SceneWorks` packed tiers of it)
  ship a control branch only. Its provenance is pinned in its own row rather than inherited from the
  t2i row, because `qwen_base` can name a different snapshot.

The hashes are pinned as constants in `tests/preview_real_weights.rs`, so a snapshot swap fails there
rather than silently applying a fit that belongs to a different latent space.

## What the hook is allowed to see

Three properties, all structural rather than defensive, and all pinned by weights-free rows that drive
the **real** sampler with a predict closure shaped like the route's:

- **CFG never reaches the preview.** All three lanes are true-CFG, and both forwards plus
  `pipeline::compute_guided_noise` run *inside* the predict closure, which returns one combined
  velocity. No fused `[2, …]` batch exists anywhere in the sampler.
  (`cfg_never_exposes_the_unconditional_half_to_the_preview`)
- **Edit reference tokens never reach the preview.** `edit.rs` concatenates the VAE-encoded references
  onto the sequence axis inside the closure and narrows the forward's result back to the noise prefix,
  so the sampler's latent stays the target alone. This is the hazard that makes an edit preview show
  the wrong picture. (`edit_previews_project_target_tokens_only`)
- **The control hint never reaches the preview.** `control_fun.rs` keeps its packed 132-channel VACE
  context in a closure capture, constant across steps and never part of the running latent.

## One frame per outer solver step

`multi_eval_solvers_still_emit_exactly_one_frame_per_outer_step` drives `heun` and `dpmpp_sde` through
the real driver and requires exactly N frames for N steps. The guard is non-vacuous in both
directions: it asserts the evaluation count **exceeds** the step count first, so a solver that fell
back to Euler could not satisfy it silently, and removing the dedup from
`candle_gen::preview::PreviewCounter::advance` was confirmed to turn the 6-step heun run into
`[1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]`.

## Real-weight run — CUDA box

`2× RTX PRO 6000 Blackwell (97,887 MiB each)`, CUDA 12.9 / MSVC 14.44 vcvars, `CUDA_COMPUTE_CAP=120`,
`--release --features cuda`. All lanes on the packed **q4** tiers.

```sh
QWEN_PREVIEW_T2I_DIR=…/SceneWorks/qwen-image-mlx/q4 \
QWEN_PREVIEW_EDIT_DIR=…/SceneWorks/qwen-image-edit-2511-mlx/q4 \
QWEN_PREVIEW_EDIT_REFERENCE=…/t2i_1024_s12_final.png \
QWEN_PREVIEW_CONTROL_BASE_DIR=…/SceneWorks/qwen-image-mlx/q4 \
QWEN_PREVIEW_CONTROL_NET=…/qwen-image-2512-fun-controlnet-union/q4/model.safetensors \
QWEN_PREVIEW_CONTROL_HINT=…/t2i_1024_s12_final.png \
QWEN_PREVIEW_ARTIFACT_DIR=…/out/sc-16952 \
  cargo test --locked --release --features cuda -p candle-gen-qwen-image \
    --test preview_real_weights -- --ignored --nocapture
```

`test result: ok. 5 passed; 0 failed` in 240 s.

Each lane rendered twice at one seed — once with an inert sink, once with a live one — and the two
outputs were **byte-identical**, on all three lanes. Each emitted exactly 12 numbered frames
`1..=12` at `H/8 × W/8`.

| lane | size / steps | mean \|Δ\| to final, frame 1 → 12 | coarse correlation, frame 1 → 12 |
| --- | --- | --- | --- |
| t2i | 1024², 12 | 63.55 → 11.73 | +0.292 → +0.994 |
| edit | 768², 12 | 55.12 → 12.46 | +0.553 → +0.988 |
| control | 768², 12 | 57.27 → 12.06 | +0.340 → +0.992 |

Both series are strictly monotone in every lane — distance falls at every step, resemblance rises at
every step — and every adjacent frame pair differs (mean |Δ| between 3.1 and 8.3), so no lane is
emitting one image N times.

**The edit lane's target-only property, measured rather than argued.** The reference was supplied at
1024² while the edit rendered at 768², so a reference-derived frame could not have carried the
asserted latent size; and the last frame's coarse correlation was **+0.988 with the edited output**
versus **−0.026 with the reference**. A preview that projected reference tokens would invert that.

Committed artifacts, in `docs/migration/evidence/sc-16952/`:

- `t2i_1024_s12_strip.png` — 12 frames of the base lane, left to right.
- `edit_768_s12_strip.png` — the edit lane.
- `control_768_s12_strip.png` — the ControlNet/Fun lane.
- `finals-contact-sheet.png` — the three finished renders at 320², in the same order, so each strip
  can be judged against what it converged on.

### A note on the first-frame correlation bound

The shared strip analysis asserts a **rise**, not an absolute floor on frame 1. Correlation is taken
over flattened RGB triplets, so it carries channel-mean structure as well as spatial structure, and
the fit's intercept `(0.406, 0.386, 0.287)` is itself R > G > B — as every warm-lit render also is. A
frame of genuine pre-denoise noise therefore starts at a non-zero, scene-dependent floor: +0.292 for
the snowy-forest t2i prompt, +0.553 for the summer-meadow edit. Porting sc-16950's `r_first < 0.35`
would have read that floor as resemblance and failed an honest lane for the colour of its prompt.
What cannot be faked is the rise — a strip that opened on the finished image has nowhere to rise to —
so the bound is `r_last − r_first > 0.30` plus a loose `r_first < 0.75` ceiling, layered with the
strictly monotone rise, the falling mean |Δ|, and the per-frame movement floor.

The ControlNet/Fun row is deliberately measured with an ordinary RGB hint rather than a real pose map.
The 2512-Fun branch is input-agnostic by design (no mode index — pose, canny and depth share one
path) and simply VAE-encodes whatever it is handed, and what this row measures is preview convergence,
not control fidelity. The render it steers is not judged.

## Advertising

`supports_preview` flips to `true` on the one `qwen_image` descriptor, and the sc-16951
`preview_advertising` guard in `candle-gen-catalog` is amended in the same PR, all three ways its
protocol requires: `qwen_image` added to `PREVIEW_ROUTE_IDS`, the descriptor flipped, and the
`candle-gen-qwen-image` route inventory filled in as `control_fun.rs` 1 hooked / `edit.rs` 1 hooked /
`lib.rs` 1 hooked, zero dark sites.

Both halves of that guard were confirmed non-vacuous by mutation, and reverted:

| mutation | result |
| --- | --- |
| `edit.rs` sampler site `Some(&preview)` → `None` | `every_wired_crate_pins_its_exact_route_inventory` and `a_wired_crate_leaves_no_undeclared_dark_sampler_site` FAIL; the crate-local `every_qwen_image_render_route_passes_a_preview_hook` FAILS |
| descriptor `supports_preview` → `false` | `preview_capability_matches_every_wired_shipped_route_bidirectionally` and `source_level_wiring_and_advertised_capability_agree_for_every_provider_crate` FAIL |
| dedup removed from `PreviewCounter::advance` | `multi_eval_solvers_still_emit_exactly_one_frame_per_outer_step` FAILS with 11 frames for 6 steps |
