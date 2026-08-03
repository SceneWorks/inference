# sc-16955 — candle FLUX.2 family per-step latent previews (epic 16948)

The 32-channel latent space, and the epic's widest single unlock. This story wires
`gen_core::PreviewSink` into **three** candle crates — `candle-gen-flux2`, `candle-gen-lens`,
`candle-gen-ideogram` — and adjudicates a fourth, `candle-gen-boogu`, **out** on measured evidence.

No fit is introduced. The epic-16624 32-channel constants committed at
`crates/media/mlx-gen/mlx-gen-flux2/src/preview.rs:27` are transcribed verbatim into
`crates/media/candle-gen/candle-gen-flux2/src/preview.rs`; `candle-gen-lens` re-exports them and
`candle-gen-ideogram` calls through to them. `mlx-gen-flux2/tests/fit_preview_rgb.rs` remains the only
producer.

## 1. Adjudication — all four families, explicitly

| family | crate | decision | why |
| --- | --- | --- | --- |
| **FLUX.2** | `candle-gen-flux2` | **wired** — 3 lanes | `flux2_klein_9b` + `flux2_dev` descriptors; the fit donor's own latent space |
| **Lens** | `candle-gen-lens` | **wired** — 1 shared site, 2 lanes | `lens` + `lens_turbo`; loads the same `Flux2Vae`, 250/250 tensors round-identical |
| **Ideogram 4** | `candle-gen-ideogram` | **wired** — bespoke, 1 direct emission | `ideogram_4` + `ideogram_4_turbo`; 250/250 tensors **byte-identical**, different packing order |
| **Boogu** | `candle-gen-boogu` | **NOT wired** | loads a plain **16-channel** `AutoencoderKL` with no `bn.*` stats — not this latent space |

The story's acceptance criterion for Boogu is explicit: *"Boogu emits only if its VAE is proven to be
the FLUX.2 one … If it is its own, split it out rather than shipping a borrowed fit."* It is not, so it
is split out. `candle-gen-boogu/src/pipeline.rs:34` imports
`candle_transformers::models::z_image::vae::{AutoEncoderKL, Encoder, VaeConfig}` — the FLUX.1 /
Z-Image lineage — and `config.rs:45` declares `in_channels: 16`. Measured below, and pinned as a
shipped test row so a future snapshot swap that *did* make them one file would be noticed.

### The Lens "dual dependency" premise did not hold

The story anticipated that `candle-gen-lens` depends on both `candle-gen-flux2` **and**
`candle-gen-sd3`, and asked which VAE each Lens route loads. On candle it has **no** `candle-gen-sd3`
dependency at all (`candle-gen-lens/Cargo.toml` mentions sd3 only in a comment about a shared
`serde_json` pin). Both registered Lens routes load one `Flux2Vae` from the snapshot's own `vae/` dir
via `lib.rs:285` and `lib.rs:362`; the trainer loads the same type at `training.rs:378` / `:426`. One
VAE, one latent space.

## 2. Lane enumeration — every user-reachable denoise lane, per crate

`git grep 'run_flow_sampler(\|run_curated_sampler(\|run_scm_sampler('` over each crate's `src`, plus a
sweep for bespoke `for`-loop denoise bodies and for direct `emit_preview*` / `.emit*` calls.

### `candle-gen-flux2` — `Denoise::Shared`, 3 sites, 3 lanes, 0 dark

| file | lane | driver | default for |
| --- | --- | --- | --- |
| `lib.rs:645` | registered txt2img | `run_flow_sampler` | **`flux2_klein_9b` and `flux2_dev` — the default and only registered lane** |
| `edit_provider.rs:242` | reference edit (`Flux2EditRequest`) | `run_flow_sampler` | the worker's candle edit path (`flux2_edit_candle.rs`) |
| `control_provider.rs:212` | strict-pose control (`Flux2ControlRequest`) | `run_flow_sampler` | the worker's FLUX.2-dev pose lane |

`pipeline.rs` and `vae.rs` hold the geometry and drive no sampler — pinned as a negative row
(`the_geometry_modules_drive_no_sampler`). The edit and control providers are **name-driven**: they
register no descriptor, so like Qwen-Image's and SDXL's they carry a `preview: PreviewSink` field on
their own request types. `crates/sceneworks-worker/src/image_jobs/flux2_edit_candle.rs:358` already
threads a `_preview` argument its closure currently ignores, and `flux2_control_candle.rs:338` the same
as a `generate_one` parameter — those are the consumers these fields unblock. Adding the fields
**breaks the SceneWorks build** at four exhaustive struct literals; the full list, and the guard that
rejects the lazy fix, are in §8.

### `candle-gen-lens` — `Denoise::Shared`, 2 sites, 1 wired + 1 dark

| file | lane | driver | note |
| --- | --- | --- | --- |
| `lib.rs:500` | `Pipeline::denoise` | `run_flow_sampler` | **one site, three callers** |
| `training.rs:549` | trainer sample render | `run_flow_sampler` | deliberately dark |

The site/lane distinction matters here and is the reason `candle-gen-lens/src/preview.rs` carries a
second, caller-level inventory: `Pipeline::denoise` is reached from `render` (resident, `lib.rs:583`),
`render_sequential` (`lib.rs:688`) and `denoise_for_parity` (`lib.rs:1152`). The first two build a
hook; the parity seam takes injected latents and has no `GenerationRequest`, therefore no sink, and
passes `None`. A crate-level "the site forwards a hook" assertion alone would not have noticed a lane
that dropped its `Some(&preview)`.

### `candle-gen-ideogram` — `Denoise::Bespoke`, **0** sites, 1 lane, 1 direct emission

`git grep run_flow_sampler -- candle-gen-ideogram` is **empty**. The whole family runs one bespoke
flow-match loop at `pipeline.rs:429` (`fn denoise`), reached from the single `render` entry
(`pipeline.rs:364`) by both registered ids and by both conditioning modes (txt2img and the
reference/mask edit — `resolve_edit` selects between them on the same loop). It emits through
`candle_gen::preview::emit_preview_at`.

### `candle-gen-boogu` — enumerated, deliberately left alone

Three `run_flow_sampler` sites in `pipeline.rs` (`:234`, `:328`, `:495`), all still passing `None`, and
the crate's `routes` inventory stays empty. Left untouched rather than wired: see §1.

**Confirmation:** the four crates' shipped `src` trees contain exactly the sites above and exactly one
direct emission call (`candle-gen-ideogram/src/pipeline.rs:522`). `candle-gen-catalog`'s
`preview_advertising` module re-derives all of it independently and is mutation-proven in §6.

## 3. VAE reuse — grounded in tensor bytes, per family

`HF_HUB_CACHE=E:\huggingface`. Two containers publish **one** learned 32-channel
`AutoencoderKLFlux2` (251 tensors: 250 learned + the unused `bn.num_batches_tracked` counter):

| SHA-256 | bytes | dtype | published by |
| --- | --- | --- | --- |
| `ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04` | 168,120,878 | bf16 | `black-forest-labs/FLUX.2-klein-9B` @ `92196c8e`; `SceneWorks/flux2-klein-9b-mlx` @ `1d36c680` **bf16 + q4 + q8**; `SceneWorks/flux2-klein-9b-kv-mlx` @ `fc6579b2` bf16 + q8 |
| `d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5` | 336,213,556 | f32 | `black-forest-labs/FLUX.2-dev` @ `26afe3a7`; `SceneWorks/flux2-dev-mlx` @ `0c9b86f4` q4 + q8; `SceneWorks/Lens` @ `5c5521d4`; `Comfy-Org/Lens` @ `198d6ddf` (as `vae/flux2-vae.safetensors`); `SceneWorks/lens-mlx` @ `4e1349c1` bf16 + q4 + q8; `SceneWorks/lens-turbo-mlx` @ `d3f485c3` bf16 + q4 + q8 |

The **fit donor is the bf16 one** — `mlx-gen-flux2/src/preview.rs` records the fit as measured on eight
FLUX.2 **Klein** renders.

| relation | measured | row |
| --- | --- | --- |
| f32 → bf16 | **250/250** learned tensors, **84,046,371** values, round-to-nearest-even **exact** | `candle-gen-flux2`: `the_flux2_family_ships_one_learned_vae_in_two_container_widths` |
| Lens f32 → donor bf16 | same 250/250, 84,046,371 values | `candle-gen-lens`: `the_lens_vae_rounds_onto_the_flux2_fit_donor` |
| Ideogram bf16 → donor bf16 | **250/250 byte-identical** | `candle-gen-ideogram`: `the_ideogram_vae_is_the_flux2_fit_donor_tensor_for_tensor` |

Ideogram publishes two further containers of the same tensors:

* `SceneWorks/ideogram-4` @ `2e8fb610` `bf16/vae/model.safetensors` — `00089549…409b`, 168,120,846
  bytes. Exactly **32 bytes** smaller than the donor: the donor's safetensors header carries a
  `__metadata__` block and this one does not. The **250 learned tensors are byte-identical**; only
  `bn.num_batches_tracked` differs, in value (its I64 dtype matches the donor's, unlike the packed
  re-host below).
* `SceneWorks/ideogram-4-mlx` @ `a3095855` `q4|q8/vae/model.safetensors` — `bb9ba30d…3bc9`,
  168,120,870 bytes. 250/250 learned tensors byte-identical; only `bn.num_batches_tracked` differs, in
  integer dtype (I32 vs I64) and value. Read by nothing — `Flux2Vae::build` loads
  `bn.running_mean` / `bn.running_var` and never this.

This reproduces the MLX doc block's claims exactly (it cites the same 84,046,371 values and the same
`bn.num_batches_tracked` dtype difference) **for the revisions candle pins**, which is what the story
asked for.

### Boogu, measured

`Boogu/Boogu-Image-0.1-Turbo` @ `7c475e94` `vae/diffusion_pytorch_model.safetensors` —
`8c717328c8ad41faab2ccfd52ae17332505c6833cf176aad56e7b58f2c4d4c94`, 335,306,212 bytes, **244** f32
tensors, **no `bn.*` stats at all**, `decoder.conv_in.weight` with **16** input channels. The
BatchNorm-stats normalization of a packed 128-channel space is the defining feature of
`AutoencoderKLFlux2`; its absence is not a variation, it is a different architecture over a different
latent space. Pinned by
`candle-gen-flux2`: `the_boogu_vae_is_not_the_flux2_one_and_that_is_why_boogu_is_unwired`.

## 4. Latent shape at the emission point — verified, not assumed

FLUX.2 does not denoise the tensor its fit is defined over. Three shapes, and only the last is
projectable:

| stage | shape | projectable |
| --- | --- | --- |
| the sampler's running latent | `[1, (H/16)·(W/16), 128]` packed tokens | no — rank 3 |
| after `pipeline::unpack_latents_at` | `[1, 128, H/16, W/16]` | **no** — 128 "channels" at half resolution, still bn-normalized |
| after `Flux2Vae::raw_latent_from_packed` | `[1, 32, H/8, W/8]` | yes |

Neither wrong row can be projected *quietly*. Against the committed 32-row factor table the rank-3
sequence and the 128-channel grid are both rejected outright — `project_raw_latents` pins the channel
count explicitly rather than inheriting the table's length, and
`the_packed_grid_is_rejected_by_the_raw_projector` proves it for both. The middle row is still the
patch-major trap `mlx-gen-flux2/src/preview.rs:4` names, because it is rank 4 with a plausible channel
count: a 128-row factor table *would* accept it and produce a half-resolution picture rather than an
error.

The failure that is genuinely **silent** is a different one, and no shape check can see it: running
the unpatchify while skipping the bn de-normalize. That yields a perfectly valid `[1, 32, H/8, W/8]`,
passes every check, and projects to a plausible-but-wrong picture. It is closed structurally rather
than by a guard — `decode_packed` and the preview seam call the *same*
`vae::raw_latent_from_packed`, so there is no second copy to drift — and
`packed_projection_equals_the_raw_projection_of_the_recovered_latent` pins that dropping the
de-normalize changes the projected frame, so a refactor that lost it would be red rather than merely
wrong-coloured.

The story noted that `unpack_latents` exists in `candle-gen-flux2` and asked whether to project after
it. The answer is **after it and after two more VAE-owned transforms**: `unpack_latents` only performs
the token→grid fold, and the bn de-normalize + 2×2 unpatchify live inside `Flux2Vae::decode_packed`.
That head is now factored into `vae::raw_latent_from_packed`, which `decode_packed` calls — so the
preview's geometry and the decode's geometry are **one function**, not two agreeing implementations.
`pipeline::unpack_latents` likewise became a wrapper over a new grid-keyed
`pipeline::unpack_latents_at`. The grid is the primitive because the preview hook is parameterised by
one: each route builds its hook from the same `(lat_h, lat_w)` pair it hands its decode tail, which is
what keeps the two geometries from diverging, so the seam has to unpack against that pair directly —
and `candle-gen-lens` drives it from the grid it has already resolved for its own decode. Lens's grid
is *numerically* identical to `latent_dims`' (`VAE_SCALE_FACTOR` is the same 16; the aspect-bucket
table fixes the image dimensions upstream, not this formula) — the grid-keyed form exists for the
shared-pair discipline, not for a divergent one.

**Ideogram is the exception that proves the seam is VAE-owned.** Its DiT packs the same 128 channels as
`(ph, pw, c)` rather than FLUX.2's `(c, ph, pw)`, so a FLUX.2-shaped recovery would de-normalize
against a permuted stat vector and unpatchify along the wrong axes. It owns
`pipeline::raw_latent` — also shared with its own `decode` — and reaches the shared code only at
`project_raw_latents`. Pinned numerically rather than in prose by
`the_patch_major_order_differs_from_the_flux2_channel_major_one`, which applies both folds to one
128-channel cell and asserts the results differ (channel 0's cells: Ideogram `[0, 32, 64, 96]`, FLUX.2
`[0, 1, 2, 3]`).

## 5. σ convention — FLUX.2's `input_scale` is identically 1.0

`run_flow_sampler` integrates `gen_core::sampling::FlowModelSampling`, whose `input_scale` returns
`1.0` at every σ (`gen-core/src/sampling/model_sampling.rs:220`). The running latent therefore already
*is* the tensor the fit was measured against, and the σ-less `PreviewHook::new` constructor is correct;
sc-16954's `with_sigma` correction is **not** needed and is not used.

Asserted rather than argued, and measured with sc-16954's own cheap decisive signal — the
**rail-clipped fraction of the first frame**:

| cohort | first-frame clipped fraction | outcome |
| --- | --- | --- |
| SDXL / Kolors, uncorrected (sc-16954) | **0.894** | needs `with_sigma` |
| FLUX.2 flow prior at σ_max | **0.0000** | readable as-is |

`the_flow_cohort_needs_no_sigma_correction` reads `input_scale` off the real `FlowModelSampling` for
six σ values and then measures the clipping. It is the **only non-`#[ignore]`d row** in the FLUX.2
harness, deliberately: sc-16954 shipped a red row that hid because its file's only non-ignored row was
excluded by `-- --ignored`.

## 6. Catalog guard — the three steps, and the mutation proof

1. `supports_preview: true` on `candle-gen-flux2` (both variants), `candle-gen-lens` (both) and
   `candle-gen-ideogram` (both). `candle-gen-boogu` stays `false`.
2. Six ids added to `PREVIEW_ROUTE_IDS`, each asserted individually by the loop in
   `preview_capability_matches_every_wired_shipped_route_bidirectionally`. The advertised-true set
   moves from **9** to **15**: the previous nine (`krea_2_turbo`, `krea_2_raw`, `krea_2_edit`,
   `qwen_image`, `anima_base`, `anima_aesthetic`, `anima_turbo`, `sdxl`, `kolors`) plus
   `flux2_klein_9b`, `flux2_dev`, `lens`, `lens_turbo`, `ideogram_4`, `ideogram_4_turbo`. Nothing else
   moves — the set equality in that same row is what enforces it.
3. Route inventories, exact per-file:

| crate | `Denoise` | file | hooked | direct | dark |
| --- | --- | --- | --- | --- | --- |
| `candle-gen-flux2` | `Shared` | `control_provider.rs` | 1 | 0 | — |
| | | `edit_provider.rs` | 1 | 0 | — |
| | | `lib.rs` | 1 | 0 | — |
| `candle-gen-ideogram` | **`Bespoke`** | `pipeline.rs` | 0 | **1** | — |
| `candle-gen-lens` | `Shared` | `lib.rs` | 1 | 0 | — |
| | | `training.rs` | 0 | 0 | `run_flow_sampler` #0 |

The single `DarkSite` is Lens's trainer sample render: a synthetic request carrying no `PreviewSink`,
whose result is delivered as a finished `TrainingProgress::Sample` image rather than a live stream —
the same decision sc-16950 recorded for Krea's trainer and sc-16954 for SDXL's. Its `reason` is
non-empty and checked.

Ideogram is the first **wired** `Denoise::Bespoke` crate, and the case `DIRECT_EMISSION_CALLS` was
hardened for in sc-16951's own review. Its `hooked: 0` is the point, not a gap: there is no sampler
call site in the crate to hook, which `the_wiring_table_pins_how_each_crate_denoises` verifies.

**Mutation proof** (both reverted afterwards):

| mutation | result |
| --- | --- |
| `edit_provider.rs`'s `Some(&preview)` → `None` | `a_wired_crate_leaves_no_undeclared_dark_sampler_site` **FAILED** (`["edit_provider.rs: run_flow_sampler #0"]`) and `every_wired_crate_pins_its_exact_route_inventory` **FAILED** |
| ideogram's `emit_preview_at` call deleted | `source_level_wiring_and_advertised_capability_agree_for_every_provider_crate` **FAILED** (`candle-gen-ideogram advertises supports_preview on ["ideogram_4", "ideogram_4_turbo"] but nothing in its shipped sources emits`) |

## 7. Real-weight CUDA validation

2× RTX PRO 6000 Blackwell (97,887 MiB each), MSVC **14.44.35207** vcvars64 (not 14.51),
`CUDA_COMPUTE_CAP=120`, CUDA 12.9, `--release --features cuda`. Every snapshot was already in
`E:\huggingface`; nothing was downloaded.

Each render row runs the lane **twice on one warmed generator at the same seed** — once with an inert
sink, once live — and asserts the two images are pixel-identical before analysing the strip.

| lane | route | size × steps | frames | movement (first → last) | distance fall | correlation (first → last) |
| --- | --- | --- | --- | --- | --- | --- |
| FLUX.2 klein, euler | `flux2_klein_9b` | 1024² × 12 | 12 | 0.448 → 10.321 (**23.0×**) | 32.7% (65.21 → 43.92) | +0.024 → **+0.556** |
| FLUX.2 klein, heun | `flux2_klein_9b` | 768² × 8 | 8 | 0.820 → 10.620 (**12.9×**) | 30.3% (59.65 → 41.57) | +0.129 → **+0.601** |
| Lens, euler | `lens` | 1024² × 12 | 12 | 0.636 → 9.959 (**15.7×**) | 40.1% (60.02 → 35.98) | +0.021 → **+0.625** |
| Ideogram 4 (bespoke) | `ideogram_4` | 768² × 12 | 12 | 3.663 → 7.034 (**1.9×**) | 56.9% (67.22 → 28.97) | +0.032 → **+0.828** |

All four: distance to the finished image falls at **every** step, resemblance rises at **every** step,
and no two consecutive frames are the same picture. Strips and finals are committed under
`docs/migration/evidence/sc-16955/`.

### One frame per outer step, non-vacuously

`a_multi_eval_solver_emits_one_frame_per_outer_step` runs `heun` at 8 steps and asserts the
**evaluation count exceeds the step count first**: the shared driver calls `on_progress` once per
evaluation, so counting `Progress::Step` events *is* counting evaluations. Measured **15 evaluations
for 8 outer steps**, and exactly **8** frames numbered 1..=8. Without the counter's dedup this would
have been 15 frames.

### Why the correlation floors differ per lane — and why that is the honest reading

The fit fixes a **ceiling**, not a floor. FLUX.2's fit R² is 0.76409, a correlation ceiling of
√0.76409 ≈ **0.874**; carrying forward the 86.8%-of-ceiling fraction the QwenVae (0.85 / in-sample
0.9586) and SDXL (0.83 / in-sample 0.91849) lanes were held to gives a derived floor of **0.75**. The
in-sample R² is used on both sides — matching an in-sample number against a holdout one is the error
sc-16954 corrected mid-story.

Ideogram **reaches** that derived floor (+0.828 = 95% of ceiling). FLUX.2 and Lens do not, and the
reason is measured rather than assumed: `min_r_last` also measures *how far the trajectory has
travelled one step from the end*, which is a property of the **schedule** — the hook emits before each
solver step, so the final advancement is never previewed. FLUX.2's empirical-μ flow schedule is
strongly back-loaded (23× acceleration; its last previewed step moves more than the first nine
combined) while Ideogram's `LogitNormalSchedule` is not (1.9×). Two independent corroborations:

* Lengthening the FLUX.2 schedule moves the last frame toward the ceiling exactly as that explanation
  predicts — the same render at **28** steps reaches r **+0.663** with a 40.3% distance fall, against
  +0.556 / 32.7% at 12.
* The committed strips show it directly: the FLUX.2 strip resolves in its last two frames, the
  Ideogram strip develops evenly across all twelve.

So both `min_r_last` and the acceleration ratio are **per-lane parameters**, not shared constants —
hard-coding either would assert one family's schedule about another's. They are backstops; the
load-bearing assertions are the three strict monotonicities and the ≥ 0.30 total rise, none of which a
stale, duplicated or wrongly-scaled latent could reproduce.

sc-16950's `r_first < 0.35` ceiling is deliberately **not** ported, per the story: `r_last - r_first >
0.30` with a loose `r_first < 0.75` is used instead. Measured rises: 0.532, 0.472, 0.604, 0.796.

### Harness inputs fail rather than skip

Every env input is resolved through `required_path`, which panics. Demonstrated in passing: a run with
`FLUX2_FIT_VAE` set but `FLUX2_PREVIEW_SNAPSHOT` unset **failed** the render rows with
`FLUX2_PREVIEW_SNAPSHOT must be set for this row — skipping it would report success while proving
nothing`, rather than reporting green.

## 8. Follow-ups

* **Boogu previews — sc-17218.** They need the FLUX.1 / Z-Image **16-channel** fit, not this one. Its
  three `run_flow_sampler` sites are already enumerated in §2; the work is one VAE-identity proof
  against whichever 16-channel fit sc-16956 (FLUX.1) or sc-16957 (Z-Image) lands, plus the wiring.
  Filed separately rather than absorbed here, because shipping a 32-channel fit on a 16-channel latent
  space is precisely what this story's acceptance criteria forbid.
* **SceneWorks consumer wiring — a HARD COMPILE BREAK, not an optional enhancement.**
  `Flux2EditRequest` and `Flux2ControlRequest` now carry a `preview` field, and neither is
  `#[non_exhaustive]`. SceneWorks constructs both with **exhaustive struct literals at four sites**,
  none using `..Default::default()`:

  | site | request |
  | --- | --- |
  | `crates/sceneworks-worker/src/image_jobs/flux2_edit_candle.rs:362` | `Flux2EditRequest` |
  | `crates/sceneworks-worker/src/image_jobs/flux2_control_candle.rs:341` | `Flux2ControlRequest` |
  | `crates/sceneworks-worker/src/flux2_dev_gpu_smoke.rs:237` | `Flux2EditRequest` |
  | `crates/sceneworks-worker/src/flux2_dev_gpu_smoke.rs:322` | `Flux2ControlRequest` |

  (Line numbers against SceneWorks `origin/main` at the time of writing.) **The pin bump does not
  compile until every one of the four is updated.** Both worker lanes already receive the live sink —
  `flux2_edit_candle.rs` takes it as the `_preview` parameter of its `drive_gen_items` closure and
  `flux2_control_candle.rs` as the `_preview` parameter of `generate_one`, each with a standing comment
  saying the field does not exist upstream yet — so the fix is to feed those, not to invent a sink.

  A `..Default::default()` or `preview: Default::default()` edit would make it compile and ship lanes
  that emit nothing. SceneWorks guards against exactly that: `no_candle_image_lane_defaults_its_preview_sink`
  and `no_bespoke_candle_request_hides_its_preview_behind_a_rest_init`
  (`crates/sceneworks-worker/src/candle_preview_wiring_tests.rs`) sweep `src/image_jobs/` and reject
  both spellings, so the two worker sites must thread the real sink. The two `flux2_dev_gpu_smoke.rs`
  sites sit **outside** that sweep's directory and are on the fixer to catch from the compile error
  alone. Lands with the pin bump (sc-16962).
