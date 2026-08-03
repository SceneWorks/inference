# sc-16957 — candle Z-Image per-step latent previews (epic 16948)

Wires `gen_core::PreviewSink` into **`candle-gen-z-image`**: two registered descriptors
(`z_image_turbo`, `z_image`), two name-driven worker providers (Fun-ControlNet, img2img/edit), and
**nine** denoise lanes across three source files — more emitting lanes than any other crate in this epic
(Krea's eight is next), and the first where a *single* name-driven provider carries **both** wiring
layers, because its distilled and undistilled modes denoise differently.

No fit is introduced. The epic-16624 16-channel constants committed at
`crates/media/mlx-gen/mlx-gen-z-image/src/preview.rs` are transcribed verbatim into
`crates/media/candle-gen/candle-gen-z-image/src/preview.rs`; `mlx-gen-z-image/tests/fit_preview_rgb.rs`
remains the only producer.

---

## 1. The headline finding: Z-Image's latent space **is** FLUX.1-dev's

The epic asked this story to settle which 16-channel space Z-Image occupies, because sc-16956 had just
pinned that "16 channels" alone does not make two latent spaces the same (Boogu's 16-channel VAE turned
out to *be* FLUX.1's; the FLUX.2 32-channel fit correctly did not apply to it). The answer for Z-Image
is the same, and it is exact:

| repo | file | SHA-256 | bytes | tensors |
| --- | --- | --- | --- | --- |
| `Tongyi-MAI/Z-Image-Turbo` @ `f332072aa78be7aecdf3ee76d5c247082da564a6` | `vae/diffusion_pytorch_model.safetensors` | `f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3` | 167,666,902 | 244 |
| `Tongyi-MAI/Z-Image` @ `04cc4abb7c5069926f75c9bfde9ef43d49423021` | `vae/diffusion_pytorch_model.safetensors` | `f5b59a26…40a3` — **identical** | 167,666,902 | 244 |
| `black-forest-labs/FLUX.1-dev` @ `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` | `vae/diffusion_pytorch_model.safetensors` | `f5b59a26…40a3` — **identical** | 167,666,902 | 244 |

One file, three repos. `f5b59a26…40a3` is the exact constant sc-16956 pinned as
`DIFFUSERS_VAE_SHA256` in `candle-gen-flux/tests/preview_real_weights.rs`. The tensor walk confirms the
hash: **244 of 244 tensors, 83,819,683 values, bit-identical**, same key set, same shapes, same dtypes.
The `vae/config.json` records where it came from — `"_name_or_path": "flux-dev"` (Turbo) /
`"../checkpoints/flux-dev"` (base) — with `latent_channels: 16`, `scaling_factor: 0.3611`,
`shift_factor: 0.1159`, matching `candle_transformers::models::z_image::vae::VaeConfig::z_image()`
exactly. The repo already half-knew this: `candle-gen-z-image/Cargo.toml` says "Z-Image ships the
FLUX.1 16-ch VAE, so it aliases the `flux` PiD latent-space student", and `common::decode` says
"Z-Image aliases the FLUX.1 latent space". This is that claim measured rather than asserted.

### Consequence: epic 16624 committed **two** fits over **one** latent space

`mlx-gen-flux/src/preview.rs` and `mlx-gen-z-image/src/preview.rs` are two independent OLS solutions
over the same 16-channel VAE, measured on different render sets. They are close but not equal — same
sign and comparable magnitude on all 16 rows (e.g. row 0 `[-0.0125, +0.0163, +0.0434]` vs
`[-0.0132, +0.0206, +0.0503]`; row 13 `[-0.0802, -0.0311, -0.0829]` vs `[-0.0728, -0.0102, -0.0743]`),
which is exactly what two samples of one linear relationship look like.

This is a **duplication, not a contradiction**. Either fit would preview either family; the difference
is decorative colour nuance, and the denoise path never reads these constants.

**Decision: `candle-gen-z-image` keeps the Z-Image-measured fit.** Three reasons, in order:

1. It is what sc-16957's acceptance criteria name ("reuse the fit, do not refit").
2. It was measured on Z-Image-Turbo renders, so its in-sample R² (`0.98133`) describes *this* family's
   latent distribution.
3. It keeps candle's Z-Image previews byte-comparable with the MLX Z-Image lane's, which is the
   cross-backend parity property every other family in this epic preserves.

Collapsing the two into one fit would change **MLX** preview bytes on one of the two families, so it is
a cross-engine decision, not a candle one. Recorded as a follow-up in §8 rather than taken here.

### The fit donor is the same file, re-containered

`SceneWorks/z-image-turbo-mlx` @ `bb2bc9893b3c49ae96c813350775f791a2e8bc80`, `bf16/vae/model.safetensors`
— SHA-256 `0fbab8b661f6ee6af81c88a6eb1501ec1f7b4b8fe4ad29803507ebe0cf863810`, 167,666,968 bytes, 244
tensors — is the container `mlx-gen-z-image/src/preview.rs` names. Its hash differs from the diffusers
file's and its length differs by **66 bytes**, which is the safetensors header's `__metadata__`
(`{"format":"pt"}`) that the MLX writer omits. Every learned tensor underneath is bit-identical, at both
the Turbo and the base snapshot:

```
fit donor vs Z-Image-Turbo: 244 tensors, 83819683 values, bit-identical
fit donor vs Z-Image base : 244 tensors, bit-identical
z-image vs flux1          : 244 tensors, 83819683 values, bit-identical
```

That is why the reuse gate is a **tensor** comparison and not a hash equality: a hash test would have
reported a mismatch for a file whose weights are identical.

Rows: `the_committed_fit_donor_is_the_shipped_z_image_vae`, `the_z_image_vae_is_the_flux1_one` in
`crates/media/candle-gen/candle-gen-z-image/tests/preview_real_weights.rs`.

---

## 2. Lane enumeration — every user-reachable denoise lane

`git grep 'run_flow_sampler(\|run_curated_sampler(\|run_scm_sampler('` over `candle-gen-z-image/src`,
plus a sweep for bespoke `for`-loop denoise bodies and for direct `emit_preview*` / `.emit*` calls.
**Seven sampler sites and three bespoke loops; ten lanes, of which nine are user-reachable renders and
one is the trainer's sample render (dark).** Sites are cited by **file**, not by line, matching the
catalog table — the scanner re-derives the exact positions from the shipped module tree on every run.

The epic's scoping said "Z-Image drives `candle_gen::run_flow_sampler`". That is **true of the base
halves and of every registered route, and false of three distilled-Turbo lanes**, which own bespoke
flow-match Euler loops. This is the third time the epic's sampler scoping has been incomplete
(sc-16954 found SDXL/Kolors' second lanes; sc-16956 found the `unpack_latents` claim misleading), so it
was re-enumerated from source rather than inherited.

| file | lane | driver | default for |
| --- | --- | --- | --- |
| `pipeline.rs` | `render` — Turbo resident txt2img / img2img | `run_flow_sampler` | **`z_image_turbo`, the default path** |
| `pipeline.rs` | `denoise_sequential` — Turbo staged residency | `run_flow_sampler` | `z_image_turbo` when `req.memory.stage_residency` |
| `pipeline.rs` | `render_base` — base resident txt2img / img2img, real CFG | `run_flow_sampler` | **`z_image`, the default path** |
| `pipeline.rs` | `denoise_base_sequential` — base staged residency | `run_flow_sampler` | `z_image` when `req.memory.stage_residency` |
| `control.rs` | `generate_turbo` — Turbo control, resident | **bespoke Euler loop** | **`z_image_turbo_control`, the default path** |
| `control.rs` | `denoise_turbo_with` — Turbo control, staged | **bespoke Euler loop** | `z_image_turbo_control` when phase-loaded |
| `control.rs` | `generate_base` — base control, resident, real CFG | `run_flow_sampler` | **`z_image_control`, the default path** |
| `control.rs` | `denoise_base_with` — base control, staged | `run_flow_sampler` | `z_image_control` when phase-loaded |
| `edit.rs` | `generate` — img2img / masked edit, reduced schedule | **bespoke Euler loop** | the worker's `zimage_identity` candle lane |
| `training.rs` | `render_sample` — trainer periodic sample | `run_flow_sampler` | **dark on purpose** (§5) |

`adapters.rs`, `base.rs`, `comfyui.rs`, `common.rs`, `dit.rs`, `lib.rs`, `memory_strategy.rs`,
`packed_dit.rs`, `packed_te.rs`, `preview.rs` and `quant.rs` hold weights, geometry, remapping or
registration and drive no sampler — pinned as a negative row
(`no_other_shipped_module_drives_a_sampler_or_emits`), with
`the_inventory_covers_every_file_in_src` asserting the file list is the crate's whole `src/` surface.

The four `*_validate.rs` files are out-of-line `#[cfg(test)] mod` GPU-validation harnesses. They are
**not** shipped code and are deliberately excluded from `candle-gen-catalog`'s module-tree walk; they
were not wired and are not counted. They do carry the new `preview` field, set to the inert
`PreviewSink::default()`, because the request structs are exhaustive.

### `z_image_turbo_control` / `z_image_control` are memory strategies, not descriptors

Both ids register a `gen_core::MemoryRegistration` and nothing else. They have no `ModelDescriptor`, so
they have no `supports_preview` to flip and cannot join `PREVIEW_ROUTE_IDS` — the same shape as
`candle-gen-flux`'s control/IP providers. That is why **two** ids cover **nine** render lanes.
`edit.rs`'s provider is the same: a name-driven worker stream carrying a `preview` field on its own
request type.

### The "one site, N callers" check — this crate is structurally immune

sc-16955's Lens finding was one sampler site reached by three callers, only two of which had a sink; a
site-level assertion could not see the third. Z-Image cannot have that shape, and the reason is
structural rather than lucky: **no Z-Image lane takes a hook or a sink as a parameter.** Every emitting
lane takes the whole request — `&GenerationRequest` in `pipeline.rs`, `&ZImageControlRequest` in
`control.rs`, `&ZImageEditRequest` in `edit.rs` — and reads `req.preview` *at the site itself*. A caller
has nothing to drop: forwarding the request is forwarding the sink, and a caller that did not forward
the request could not call the lane.

`every_emitting_lane_reads_the_sink_off_its_own_request` pins exactly that: all six hook constructions
are literally `crate::preview::hook(&req.preview)`, all three direct emissions pass `&req.preview`, and
no scanned file declares a `: &PreviewHook` / `: &PreviewSink` parameter.

**That row is also what caught a real miss during implementation**: `control::generate_base` (the
resident base-control lane) was still passing `None` after the first wiring pass, and
`every_shipped_render_lane_emits_a_preview` failed on it by position before any GPU time was spent.

---

## 3. The latent shape at the emission point — verified, not assumed

The epic explicitly flagged Z-Image: *"MLX FLUX.1 and Z-Image project post-unpack, so those two stories
must verify the actual latent shape at the emission point rather than porting or omitting an unpack step
by assumption."*

Verified. **Candle Z-Image is not packed at all.** There is no `unpack_latents` in this crate and none
is needed — the patchify/unpatchify pair lives entirely *inside*
`candle_transformers::models::z_image::transformer`'s forward, so the running latent never enters the
packed token space. What it does have is a **rank-5** layout:

| stage | shape |
| --- | --- |
| `common::seed_noise` | `[1, 16, H/8, W/8]` |
| after `z_image::preprocess::prepare_inputs` (`latents.unsqueeze(2)`) | `[1, 16, 1, H/8, W/8]` — what the sampler integrates |
| after dropping the frame axis | `[1, 16, H/8, W/8]` — the fitted space |

So the recovery is one squeeze of the singleton **frame** axis. That is the third distinct latent
convention this epic has met at an emission point (Krea rank-4 spatial, Qwen packed rank-3, Anima 5-D
Cosmos, SDXL/Kolors raw VE σ-space, FLUX.2 packed + bn de-normalize + unpatchify, FLUX.1 packed): closest
to Anima's, but with a leading batch axis MLX Z-Image does not have (MLX denoises `[16, 1, h, w]` and
reaches the fitted space through its own `pipeline::unpack_latents`). Porting either neighbour would
have been wrong.

**Bound to one source.** The axis is spelled `crate::common::LATENT_FRAME_AXIS`, introduced by this
story and used by *both* `common::decode` (which previously wrote a bare `squeeze(2)`) and
`preview::drop_frame_axis`. Because the rest of the geometry travels inside the latent, the projector
needs no `width`/`height` argument at all — hook geometry and latent geometry are not merely bound to one
source, there is only one source to bind to.

**The squeeze is checked, not bare.** Candle's `squeeze` is a **no-op** on an axis whose extent is not 1
(the Anima lesson), so a `[1, 16, T>1, h, w]` latent would pass straight through it and fail later with a
message about a contract this family never violates. `drop_frame_axis` gates rank, batch, channel count
and frame extent first. `projection_rejects_every_non_z_image_layout` covers six wrong layouts including
the already-squeezed rank-4 one, the temporal one, and MLX's batch-less `[16, 1, h, w]`.

---

## 4. The σ convention — no correction needed, measured

`run_flow_sampler` integrates a `gen_core::sampling::FlowModelSampling`, whose `input_scale` is
**exactly `1.0` at every σ** under both `TimestepConvention::Sigma` and `OneMinusSigma` (Z-Image drives
the latter). So the running latent already *is* the tensor the fit was measured against, and
`PreviewHook::new` — the σ-less constructor — is correct. The three bespoke Turbo loops scale nothing
either: they hand `latents` straight to the DiT.

sc-16954 found the **opposite** for the discrete ε cohort (SDXL/Kolors denoise in k-diffusion VE σ-space;
the uncorrected first frame clipped **89.4%** of pixels to the 0/255 rails and needed
`PreviewHook::with_sigma`). The cheap decisive signal is taken here regardless of the argument above:

```
flow prior at sigma_max: rail-clipped fraction 0.0000
```

A unit-normal `[1, 16, 1, 32, 32]` latent at `σ_max = 1.0` — exactly what `common::seed_noise` produces
and what the first emission sees — projects to a **readable noise field**, not a saturated binary one.

`the_flow_cohort_needs_no_sigma_correction` is the **only non-`#[ignore]`d row** in the harness file, so
it runs in a plain `cargo test` of that target. That is deliberate: sc-16954 shipped a red row that hid
because `-- --ignored` excluded the only non-ignored row in its file. §7 reports the harness run **both
ways**.

---

## 5. The one dark site

| file | driver | index | reason |
| --- | --- | --- | --- |
| `training.rs` | `run_flow_sampler` | 0 | the trainer's periodic sample render drives the sampler from a synthetic request that carries no `PreviewSink` — its result is delivered as a finished `TrainingProgress::Sample` image, not as a live denoise stream — so it passes `None` on purpose |

The same decision sc-16950 recorded for Krea's trainer, sc-16954 for SDXL's and sc-16955 for Lens's.
Declared as a `DarkSite { driver, index, reason }` in `candle-gen-catalog`'s inventory (so
`a_wired_crate_leaves_no_undeclared_dark_sampler_site` accepts it) and pinned **positively** in the
crate's own `the_trainer_sample_render_is_deliberately_dark` — the argument *is* `None`, asserted by
position, rather than merely absent from a list.

---

## 6. The catalog guard — all three steps in this PR

1. **`supports_preview` flipped** on `candle-gen-z-image/src/lib.rs` (`z_image_turbo`) and
   `candle-gen-z-image/src/base.rs` (`z_image`). `comfyui_descriptor()` derives from `descriptor()`, so
   the in-place ComfyUI variant inherits it — correctly, since it drives the same `pipeline::render`.
2. **`PREVIEW_ROUTE_IDS` extended** with `z_image_turbo` and `z_image`, asserted individually by
   `preview_capability_matches_every_wired_shipped_route_bidirectionally`.
3. **Route inventory added** to `PROVIDER_CRATES`:

   | file | hooked | direct | dark |
   | --- | --- | --- | --- |
   | `control.rs` | 2 | 2 | — |
   | `edit.rs` | 0 | 1 | — |
   | `pipeline.rs` | 4 | 0 | — |
   | `training.rs` | 0 | 0 | `run_flow_sampler` #0 |

   `Denoise::Shared` is retained and verified — this crate does drive shared samplers — while the
   `direct` counts record the three bespoke loops. `candle-gen-kolors` and `candle-gen-sdxl` already mix
   the two layers within a file; what is new here is *why* `control.rs` does: not two lanes of one
   route, but two **modes** of one provider that denoise differently — distilled Turbo owns a loop,
   undistilled base drives the sampler — each in both a resident and a staged-residency form.

   `preview.rs` gets no row: it carries only the reused fit and the frame-axis drop, so it neither
   drives a sampler nor emits.

**Advertised-true set after this PR: 22.** krea ×3, `qwen_image`, anima ×3, `sdxl`, `kolors`, flux2 ×2,
lens ×2, ideogram ×2, `flux1_schnell`, `flux1_dev`, chroma ×3, **`z_image_turbo`, `z_image`**. Nothing
else moved; the bidirectional guard fails if it had.

---

## 7. Real-weight CUDA validation

Hardware: 2× RTX PRO 6000 (`CUDA_COMPUTE_CAP=120`), CUDA 12.9, MSVC 14.44 vcvars64,
`HF_HUB_CACHE=E:\huggingface`. Every snapshot was already cached; nothing was downloaded.

`cargo test -p candle-gen-z-image --release --features cuda --test preview_real_weights`

### 7.1 Without `--ignored` (the plain run)

```
running 7 tests
test the_flow_cohort_needs_no_sigma_correction ... ok
(6 ignored)
test result: ok. 1 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out
```

Green. Reported separately, and first, because sc-16954 shipped a **red** row that `-- --ignored` hid —
and this file reproduces exactly the conditions that hid it: the `--ignored` run below reports
`1 filtered out`, and that one filtered row is the σ-convention row. Neither invocation exercises the
whole file, so both are run and both are reported.

### 7.2 With `--ignored` — the six weight-bearing rows

```
test a_multi_eval_solver_emits_one_frame_per_outer_step ... ok
test the_committed_fit_donor_is_the_shipped_z_image_vae ... ok
test the_control_routes_preview_their_target_latent ... ok
test the_edit_route_previews_its_reduced_schedule ... ok
test the_z_image_vae_is_the_flux1_one ... ok
test z_image_preview_frames_evolve_toward_the_final_image ... ok
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 196.71s
```

**All nine emitting lanes** rendered **twice on one warmed generator at seed 16957**, once with an inert
sink and once with a live one, at 512², plus a tenth strip driving the Turbo lane with `heun` instead of
Euler. Every pair was **pixel-identical**, so an active sink perturbs no render byte
(`assert_eq!(inert.pixels, live.pixels)` on every row).

| lane | wiring | steps | frames | distance to final | coarse correlation | floor |
| --- | --- | --- | --- | --- | --- | --- |
| `z_image_turbo-resident` | hooked | 8 | 8 | ↓ 71.6 % (52.75 → 14.99) | +0.051 → **+0.921** | 0.90 |
| `z_image_turbo-staged` | hooked | 8 | 8 | ↓ 71.6 % (52.75 → 14.99) | +0.051 → **+0.921** | 0.90 |
| `z_image-resident` | hooked | 20 | 20 | ↓ 56.2 % (50.81 → 22.26) | +0.115 → **+0.836** | 0.80 |
| `z_image-staged` | hooked | 20 | 20 | ↓ 56.2 % (50.81 → 22.26) | +0.115 → **+0.836** | 0.80 |
| `z_image_turbo-heun` | hooked | 8 (**15 evals**) | 8 | ↓ 69.1 % (51.25 → 15.83) | +0.086 → **+0.913** | 0.89 |
| `z_image_turbo_control-resident` | **direct** | 8 | 8 | ↓ 73.6 % (49.64 → 13.12) | +0.099 → **+0.942** | 0.92 |
| `z_image_turbo_control-staged` | **direct** | 8 | 8 | ↓ 73.6 % (49.64 → 13.12) | +0.099 → **+0.942** | 0.92 |
| `z_image_control-resident` | hooked | 20 | 20 | ↓ 66.2 % (60.53 → 20.49) | +0.155 → **+0.920** | 0.90 |
| `z_image_control-staged` | hooked | 20 | 20 | ↓ 66.2 % (60.53 → 20.49) | +0.155 → **+0.920** | 0.90 |
| `z_image_edit` | **direct** | 12 @ strength 0.5 → **6** | 6 | ↓ 38.3 % (28.05 → 17.30) | +0.764 → **+0.892** | 0.87 |

Distance to the finished image falls at **every** step and coarse correlation rises at **every** step on
every strip — the two monotonicities are the load-bearing assertions, and no stale, duplicated or
wrongly-scaled latent reproduces them. Strips and finals are committed under
`docs/migration/evidence/sc-16957/` (`*-strip.png` is one horizontal contact sheet per lane;
`pose-skeleton.png` is the control input all four control lanes were driven with).

**Both control modes are rendered in both residencies.** That is not padding: a resident model takes
`generate_turbo` / `generate_base`, while a `stage_residency` load leaves the transformer absent and
routes through `generate_staged` → `denoise_turbo_with` / `denoise_base_with` — four *different
functions*, each with its own emission, and the staged pair is what a memory-constrained SceneWorks run
actually takes. Rendering only the defaults would have left two shipped lanes unproven.

Four things the table shows that prose could not:

* **Every staged-residency lane is the same trajectory as its resident twin**, to three decimal places
  on every metric, on all four pairs — which is the point: each is a *different function* with its own
  emission, and a memory-constrained SceneWorks run takes it instead. Identical metrics are the
  assertion, not a coincidence: staged residency must not change what is rendered, only when the weights
  are live.
* **`heun` evaluated 15 times for 8 outer steps and produced 8 frames.** The evaluation count is
  asserted `> steps` *before* the frame count is, so the dedup guard cannot pass vacuously. 15 rather
  than 16 because the final step's second evaluation lands at σ = 0.
* **The bespoke-loop lanes work end to end.** `z_image_turbo_control` reaches the highest correlation of
  the eight (+0.942) — a pose-locked composition resolves earlier than a free one — and it is one of the
  two lanes with *no* driver at all. The catalog tally can see that a direct emission call exists; only
  this can see that it produces frames.
* **The base path's schedule, not the wiring, is what caps its correlation.** `z_image` reaches +0.836
  where Turbo reaches +0.921, because the static shift=6.0 σ table is heavily back-loaded — its strip is
  visibly noise for two thirds of its length and then resolves fast (frame 13 is +0.306, frame 20 is
  +0.836). The hook emits *before* each solver step, so the largest single advancement is never
  previewed. That is why `min_r_last` is per-lane and each floor carries its measured number.

#### The img2img lane needed its own "develops" pair — and that is a finding, not a threshold fudge

`z_image_edit` opens at **+0.764**, above the `r_first < 0.75` ceiling every other lane passes. That
ceiling is a *txt2img* statement ("the first frame is pre-denoise noise"), and it is simply **false** for
a strength-reduced img2img strip: the first emission is `x_t = (1 − σ_start)·source + σ_start·noise`, so
it opens partly converged **by construction** — that is what a structure-preserving edit means.

Loosening the global constant would have weakened all seven other lanes to accommodate one. Instead the
pair is per-lane: `FROM_NOISE { max_r_first: 0.75, min_rise: 0.30 }` for every txt2img and control lane
(the epic's numbers, unchanged), and `FROM_A_PARTIAL_LATENT { max_r_first: 0.85, min_rise: 0.08 }` for
the one lane that does not start on noise, with both numbers carrying their measurement. The two
monotonicities and the ≥ 25 % distance fall apply unchanged everywhere.

#### What the strips look like

* Turbo (8 steps, linear σ): a clean noise → lighthouse progression, roughly even movement per step
  (mean |Δ| 8.21 → 6.64), which is what an unshifted schedule produces. `crate::pipeline::render`
  documents that the Turbo `Some(mu)` call applies **no** shift under `use_dynamic_shifting=false`, so
  this is the expected shape rather than a defect — and it is why sc-16956's *acceleration* assertion is
  deliberately not ported (it would be asserting the schedule, not the wiring).
* Base (20 steps, static shift 6.0): noise for most of the strip, then a fast resolve.
* Turbo control: a standing figure emerging in the skeleton's pose, from the bespoke loop.
* Edit (6 of 12 steps at strength 0.5): opens on a recognisable, half-noised source and refines.

#### The σ-convention measurement

```
flow prior at sigma_max: rail-clipped fraction 0.0000
```

Zero pixels on the rails, against sc-16954's uncorrected SDXL **0.894**. No `with_sigma` correction is
needed and none is used.

---

## 8. SceneWorks call sites this breaks — deliberate, for the sc-16962 pin bump

Adding a `preview` field to a name-driven request type is a **source-breaking change** for SceneWorks,
and both Z-Image bespoke request types take one. Surveyed at the current pin
(`rev = bf06bb569697391a620e171f983fbb4d11a2ff14`); exactly two files stop compiling, and both are
already holding the sink and discarding it.

| file | line | what breaks | fix |
| --- | --- | --- | --- |
| `crates/sceneworks-worker/src/image_jobs/zimage_control.rs` | 436 | exhaustive `ZImageControlRequest` literal in `ZImageStrictControl::generate_one` → E0063 | rename the `_preview: &gen_core::PreviewSink` parameter at :430 to `preview` and add `preview: preview.clone()` |
| `crates/sceneworks-worker/src/image_jobs/zimage_identity_candle.rs` | 270 | exhaustive `ZImageEditRequest` literal inside the `drive_gen_items_scored` closure → E0063 | bind the closure's `_preview` at :266 as `preview` and add `preview` to the literal |

Neither uses `..Default::default()`, so nothing can silently absorb the field — which is what
`no_bespoke_candle_request_hides_its_preview_behind_a_rest_init`
(`crates/sceneworks-worker/src/candle_preview_wiring_tests.rs:382`) exists to enforce. Both sinks are
already real: `preview_sink_for` (`image_jobs/stream.rs:51`) builds them and `drive_gen_items_scored`
hands them down. The reference implementations to copy are `image_jobs/qwen_control.rs:371/383` and
`image_jobs/krea_control_candle.rs:497/510`.

Two guard tables also need the new lanes at pin time:

* `WIRED_LANES` (`candle_preview_wiring_tests.rs:47`) — add `ZImageControlRequest` and
  `ZImageEditRequest` beside the existing `Krea2ControlRequest` / `QwenFunControlRequest` /
  `QwenEditRequest`.
* `SINK_CONSUMERS` (`candle_preview_wiring_tests.rs:422`) — add `zimage_identity_candle.rs`. It is
  **not** in that list today, so its discarded `_preview` is currently invisible to
  `candle_preview_consumers_do_not_discard_their_sink`.

Three Z-Image lanes need **no** SceneWorks change and light up automatically once the pin bumps, because
they already thread `GenerationRequest.preview`: `image_jobs/zimage.rs` (the registry lane, :192/:439),
`image_jobs/zimage_comfyui_candle.rs` (:210), and the `CandleImageRoute::ZimageEdit` registry route
(`image_jobs.rs:754-767`).

---

## 9. Pre-existing failures, confirmed not made worse

`every_registered_memory_strategy_rejects_cross_route_decode_geometry` under `--features cuda` fails on
`z_image_turbo_control` / `z_image_control` — sc-17087, pre-existing, in this crate. Verified unchanged
by this PR: it is a memory-strategy decode-geometry assertion and touches no preview surface, and the
run reproduces the same two ids before and after. Not fixed here.

---

## 10. Follow-ups

1. **Two committed fits describe one latent space** (§1). `mlx-gen-flux/src/preview.rs` and
   `mlx-gen-z-image/src/preview.rs` are independent OLS solutions over a byte-identical VAE.
   Consolidating them is worth doing but is a **cross-engine** change — collapsing to one set would move
   MLX preview bytes on whichever family loses its own fit — so it belongs to a story that owns both
   engines, not to a candle wiring story. Candidate scope: pick the fit with the better *pooled* holdout
   over both families' render sets, retire the other, and re-point `candle-gen-flux`,
   `candle-gen-chroma`, `candle-gen-pulid`, `candle-gen-boogu` (sc-17218) and `candle-gen-z-image` at
   the survivor.
2. **`candle-gen-boogu` (sc-17218) shares this space too.** sc-16956 proved Boogu's VAE is FLUX.1's;
   §1 proves Z-Image's is as well. Whichever fit that story reuses, it is now known to be one of these
   two rather than a third.
3. **SceneWorks pin bump** (§8) — the two call sites and the two guard tables, at sc-16962.
