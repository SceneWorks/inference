# sc-16954 candle SDXL-family latent-preview evidence

Epic 16948 wires `PreviewSink` into the candle engines. SDXL is Tier 1 and the **first non-QwenVae
fit** in the epic: 4 channels rather than 16, and the first family whose registered routes drive
`run_curated_sampler` rather than `run_flow_sampler`. It reuses the RGB fit epic 16624 committed on
the MLX side and adds **no** new fit. This file records what makes that reuse legitimate, what the
real-weight runs actually showed, and the one structural finding this story turned up that the four
predecessors did not.

## Adjudication — the three families the story asked about

| family | decision | why |
| --- | --- | --- |
| **kolors** | **wired**, `supports_preview: true` | One registered descriptor (`kolors`). Both its lanes emit, as do both name-driven providers. Reuses the SDXL fit through `candle_gen_sdxl::preview` — one byte-identical VAE file. |
| **instantid** | **not wired**, stays unadvertised | Registers **no descriptor at all**. It is a `BESPOKE_UTILITY_CRATES` member and `candle-gen-catalog`'s `bespoke_composition_apis_have_no_invented_registration` actively forbids it acquiring one, so there is no `supports_preview` to flip and no `PROVIDER_CRATES` row to inventory. `InstantIdRequest` carries no sink either. MLX left it unadvertised for the same reason. |
| **svd** | **stays preview-inert** | `svd_xt` is already in `PREVIEW_INERT_ROUTE_IDS`, carried over from epic 16624's rejected temporal fits. Its fit was **not** re-run and must not be: the .88 holdout bar is a property of the latent space, not of the backend. Its `run_curated_sampler` site keeps passing `None` and its crate keeps `routes: &[]`. |

InstantID is the interesting one, and the answer is **it does not inherit previews for free**. It
delegates to `candle_gen_sdxl::denoise_curated` and `denoise_ip_multi_control` — *not* to the
registered SDXL generator's `pipeline.rs` render lane — and its production default is the bespoke
ancestral loop. Both of those functions now take a preview argument, and InstantID passes `None` at
both, which is a decision recorded in `model.rs` rather than an omission. Wiring it would mean adding
a `preview` field to `InstantIdRequest` and a worker that sets it; that is listed as a follow-up.

## The fit is reused, not refitted

`crates/media/candle-gen/candle-gen-sdxl/src/preview.rs` carries `RGB_FACTORS` / `RGB_BIAS` copied
verbatim from `mlx-gen-sdxl/src/preview.rs:27`. There is deliberately no candle producer of the
coefficients: `mlx-gen-sdxl/tests/fit_preview_rgb.rs` remains the only way they are re-derived.

`candle-gen-kolors/src/preview.rs` defines **no constants at all** — it calls through to the SDXL
seam, and `preview::tests::the_fit_is_the_shared_sdxl_one` projects the same latent through both and
requires the pixels to be equal, so a copy could not be introduced there without failing.

Fit corpus (unchanged, quoted from the donor): four diverse 512² real-weight SDXL renders, seeds
1663301..1663304, evaluated on two disjoint holdouts (1663391, 1663392), all 12-step ancestral Euler
at CFG 5.0 against 8×8-average-pooled VAE decode targets. Fit R² `(0.91640, 0.92538, 0.91487)` /
overall `0.91849`; holdout R² `(0.86501, 0.84844, 0.86649)` / overall `0.86065`.

## Learned-basis transfer — grounded in tensor bytes

The claim being checked is not "both crates name a type `AutoEncoderKL`". SDXL and Kolors are one
latent space because they ship **one VAE file**, and it is the file the fit was measured against — so
this is settled by a hash equality and needs no tensor-by-tensor argument.

| snapshot | revision | file | SHA-256 | bytes |
| --- | --- | --- | --- | --- |
| `stabilityai/stable-diffusion-xl-base-1.0` | `462165984030d82259a11f4367a4eed129e94a7b` | `vae/diffusion_pytorch_model.fp16.safetensors` | `bcb60880…6161e68` | 167,335,342 |
| `Kwai-Kolors/Kolors-diffusers` | `7e091c75199e910a26cd1b51ed52c28de5db3711` | `vae/diffusion_pytorch_model.fp16.safetensors` | `bcb60880…6161e68` | 167,335,342 |
| `SceneWorks/sdxl-base-mlx` (`bf16`/`q8`/`q4` share one file) | `36699bb8a6353e61c920e3bf19f0e6f8e4151c55` | `*/vae/diffusion_pytorch_model.fp16.safetensors` | `bcb60880…6161e68` | 167,335,342 |
| `SceneWorks/kolors-mlx` (`bf16`/`q8`/`q4` share one file) | `aadbd49f53b66a33ef1be09384eac409cbc44061` | `*/vae/diffusion_pytorch_model.fp16.safetensors` | `bcb60880…6161e68` | 167,335,342 |

Full digest: `bcb60880a46b63dea58e9bc591abe15f8350bde47b405f9c38f4be70c6161e68`.

- **All four are the same bytes**, and it is the digest `mlx-gen-sdxl/src/preview.rs:25` already cites
  as its Kolors grounding — so the candle claim and the MLX claim rest on the same file.
- **Every packed tier keeps the VAE dense.** The MLX packer mirrors `vae/` verbatim
  (`mlx-gen-kolors/src/convert.rs:124`), so no q4/q8 tier has a copy of its own to drift.
- Both `vae/config.json`s declare `latent_channels: 4` and `scaling_factor: 0.13025` — the two numbers
  that *define* the space, and the value `VAE_SCALE` hardcodes on both lanes. The only differences
  between the two configs are metadata (`_diffusers_version`, `_name_or_path`, `force_upcast`).

### The one asymmetry, recorded rather than glossed

Candle SDXL **decodes** through the caller-staged `madebyollin/sdxl-vae-fp16-fix`
(`loaders::load_sdxl_vae`), *not* the snapshot's own VAE:

| snapshot | revision | file | SHA-256 | bytes |
| --- | --- | --- | --- | --- |
| `madebyollin/sdxl-vae-fp16-fix` | `207b116dae70ace3637169f1ddd2434b91b3a8cd` | `diffusion_pytorch_model.safetensors` | `1b909373…7b0ace2c` | 334,643,238 |

Measured against the original: same 248 keys, same 248 shapes, **and not one tensor equal in value**
— 108/108 encoder-side and 140/140 decoder-side differ. It is a genuine fine-tune, not a precision
variant, and a "the VAEs match" claim here would have been false.

What that does and does not move:

- **The fit's input domain is unaffected.** The UNet that produces these latents is byte-identical
  across engines and `VAE_SCALE` is unchanged at 0.13025, so the latents the fit reads are the same
  latents. `madebyollin/sdxl-vae-fp16-fix` is a documented drop-in for that space.
- **The fit's colour target could in principle move**, since the decode is a different decoder. That
  is settled empirically rather than by assertion: every convergence number below is measured against
  the image *this* decoder actually produced, and the curated lane's last frame still reaches
  r **+0.885**.
- **Kolors is the closer match to the fit corpus, not the looser one** — it decodes with the
  snapshot's own VAE, so its whole chain is the fit donor's.

`the_decode_vae_is_a_different_checkpoint_and_that_is_recorded` pins both digests and asserts they are
*not* equal, so if the two ever converge a reader is forced to revisit this section.

## The latent shape at the emission point — verified, not assumed

Every predecessor in this epic had a different layout and would have silently swallowed every frame if
ported by assumption (Krea rank-4 spatial, Qwen-Image packed rank-3, Anima 5-D Cosmos). SDXL is the
easy case, and it was checked rather than presumed:

- `pipeline::Pipeline::render` builds `Tensor::from_vec(noise, (1, 4, lat_h, lat_w), …)` directly;
- `denoise::seeded_sigma_prior` returns the same NCHW `[1, 4, height/8, width/8]`;
- the decode tail `pipeline::tiled_vae_decode` takes exactly `[1, 4, h, w]`;
- Kolors' `common::initial_noise` builds `(1, 4, lat_h, lat_w)` on the same contract.

So the running latent is **rank-4 `[1, 4, H/8, W/8]`** on every lane, and **no unpack step exists or
was written**. Batch is always 1: `req.count` is served sequentially through
`candle_gen::for_each_image_seed`, one fresh prior per image. Confirmed at runtime — every frame in
every strip landed at exactly `H/8 × W/8` (128×128 at 1024², 96×96 at 768²).

## The finding: the latent *convention* is NOT the one the flow cohort has

This is the part of sc-16954 that did not exist in the four predecessor stories.

`run_curated_sampler` hands the hook the **running** latent `x`, never the `c_in`-scaled model input
`x_in`, and sc-16949 documents that as the property that makes the hook see "the tensor a family's
linear RGB fit was measured against". That is true for the flow-match cohort — `FlowModelSampling`'s
`input_scale` is exactly `1.0` at every σ — and it is **false for the discrete ε-prediction cohort**.

SDXL and Kolors denoise in k-diffusion **VE σ-space**: the prior is `unit noise · σ_max` with
σ_max ≈ 14.6, and `DiscreteModelSampling::input_scale` supplies the `1/√(σ²+1)` renormalization inside
the driver. The fit was measured on 12-step **ancestral Euler**, whose sampler folds that
renormalization into its own step — so the fit's domain is the renormalized latent. Projecting `x`
raw would push the early frames to roughly `σ·ε` against ~0.17 slopes.

Measured (`the_ve_correction_is_what_makes_the_early_frames_readable`, weights-free, at σ = 14.6):

| projection | fraction of pixels clipped to 0 or 255 |
| --- | --- |
| raw VE latent | **> 0.50** |
| `x · 1/√(σ²+1)` | **< 0.05** |

Uncorrected, the first frames are a saturated binary field rather than the noise-to-image progression
the fit describes. At the last emission σ is small, `c_in → 1`, and the two agree — so the correction
only ever changes the early frames, which is exactly where the uncorrected projection was wrong.

### How it is wired, and why it cannot drift

`PreviewHook::with_sigma` is an **additive** constructor on the shared seam: the projector receives
`(latent, Option<f32>)` instead of `latent`. `PreviewHook::new` is unchanged in signature and wraps
the one-argument closure, so every family wired before this story is byte-identical — the flow cohort
never sees a σ. The σ-less SCM driver passes `None`, and `project_ve_latents` treats `None` as an
error rather than as "project unscaled", because silently projecting a raw VE latent is the exact
failure the function exists to prevent.

Which projector each lane uses is read off **what the lane feeds its UNet**, not chosen:

| lane | running latent | projector |
| --- | --- | --- |
| SDXL `Pipeline::denoise_curated`, `denoise::denoise_curated`, Kolors `Pipeline::denoise_curated` | VE σ-space (driver applies `c_in`) | `project_ve_latents` (σ-keyed) |
| SDXL `Pipeline::denoise_lightning` | VE-like; lane holds its own `c.c_in` | direct emission of `latents · c.c_in` |
| Kolors native leading-Euler (×3) | lane holds its own `scale_in(i)` | direct emission of `latents / scale_in(i)` |
| `denoise::denoise_ip_multi_control`, `SdxlEdit::denoise_edit` | already renormalized — ancestral folds it into the step | `project_spatial_latents` |

The bespoke lanes emit the *same tensor they hand the UNet*, so the preview and the denoise cannot come
to disagree about the scaling. `ve_renormalization_matches_discrete_model_sampling_input_scale` pins
the closed form against the real `DiscreteModelSampling` at five σ values.

## CFG never reaches the preview

Every lane fuses `[uncond, cond]` **inside** its predict closure — `Tensor::cat(&[x, x], 0)` on entry,
`chunk(2, 0)` plus the guidance combine before returning — so the tensor the sampler carries as its
running latent is batch 1 at every step. The bespoke ancestral and leading-Euler loops keep the same
discipline (`latents` stays batch 1; only `x_unet` is widened).

This is structurally self-proving rather than merely asserted: a fused `[2, 4, h, w]` latent fails the
`[1, 4, h, w]` contract outright, so a strip that exists **at all** is proof the unconditional half was
never projected — there would be zero frames if it had been. Every run below rendered with CFG on
(guidance 5.0, a real negative prompt) and produced a full strip.

## Real-weight runs — CUDA box

RTX PRO 6000 Blackwell, CUDA 12.9 / MSVC 14.44 (`vcvars64`), `CUDA_COMPUTE_CAP=120`, `--release
--features cuda`, seed 16954. Every lane rendered **twice on one warmed generator at the same seed** —
once with an inert sink, once live — and the two outputs were required to be **byte-identical**.

### SDXL — `stabilityai/stable-diffusion-xl-base-1.0`, and the distilled Lightning checkpoint

| lane | sampler | steps | size | frames | mean \|Δ\| to final | coarse r with final |
| --- | --- | --- | --- | --- | --- | --- |
| curated (**the default lane**) | `ddim` (omitted → curated default) | 12 | 1024² | 12, numbered 1..=12 @128² | 71.99 → **18.30** | +0.215 → **+0.885** |
| Lightning | `lightning` | 8 | 1024² | 8, numbered 1..=8 @128² | 65.06 → **32.31** | +0.243 → **+0.600** |
| multi-eval | `heun` | 8 | 768² | 8, numbered 1..=8 @96² | 79.53 → **20.57** | +0.203 → **+0.887** |

Distance to the finished image fell at **every** step and resemblance rose at **every** step on all
three. `test result: ok. 5 passed; 0 failed`.

### Kolors — `Kwai-Kolors/Kolors-diffusers`, its own render

The reuse is a claim about weights, so Kolors gets its own run rather than being covered by SDXL's.

| lane | sampler | steps | size | frames | mean \|Δ\| to final | coarse r with final |
| --- | --- | --- | --- | --- | --- | --- |
| native leading-Euler (**the default lane**) | omitted | 12 | 1024² | 12, numbered 1..=12 @128² | 77.89 → **25.47** | +0.254 → **+0.852** |
| curated | `ddim` | 12 | 1024² | 12, numbered 1..=12 @128² | 76.92 → **24.12** | +0.228 → **+0.848** |
| multi-eval | `heun` | 8 | 768² | 8, numbered 1..=8 @96² | 86.63 → **32.38** | +0.182 → **+0.854** |

All three lanes fell and rose at every step, and Kolors' `heun` likewise produced **15 `Progress::Step`
events for 8 outer steps** and exactly 8 frames. `test result: ok. 4 passed; 0 failed`.

The Lightning lane is held to a lower `r_last` floor (0.55 rather than 0.85) and the reason is
structural, not a lowered bar. The hook emits **before** each solver step, so the last frame is the
latent one advancement short of the render — the fully denoised state is never previewed, the finished
image lands instead. On a 12-step schedule that final step is a small share of the trajectory; on the
few-step Euler-**trailing** Lightning schedule, whose terminal σ is zero, it carries a large share. The
strip's own frame-to-frame movement shows it directly: **2.20 → 3.60 → 4.63 → 5.80 → 6.98 → 8.11 →
9.20**, still *accelerating* at the last emission. The lane pays for the lower floor with an extra
assertion the others do not carry — that movement is **strictly increasing** across the whole strip,
the Euler-trailing signature, which a hook reading a stale or wrongly scaled latent would not produce.

That row was also first run against SDXL **base** weights, where it reached only +0.633 with the same
accelerating signature; it now runs against `SceneWorks/realvisxl-lightning-mlx` `bf16`
(@ `c09fd586989bdc3c658d4acd03e8ae81677ade8e`) so the lane is exercised as it actually ships rather
than as a distilled schedule driven on a non-distilled checkpoint.

### One frame per outer step, on a genuinely multi-eval solver

`heun` at 8 steps produced **15 `Progress::Step` events** and **8 preview frames**.

The guard is non-vacuous *by construction*: the shared driver calls `on_progress` once per model
**evaluation** (`sampler.rs` recomputes the step index every eval and deliberately repeats it), so
counting progress events IS counting evaluations. The test asserts `events > steps` **before** it means
anything by `frames == steps`. 15 > 8 — the extra evaluations were collapsed by
`PreviewCounter`'s σ-keyed dedup, exactly as sc-16949 designed.

### Artifacts

`docs/migration/evidence/sc-16954/` — one horizontal contact-sheet strip plus the finished render per
lane:

- `sdxl-curated-ddim-strip.png`, `sdxl-curated-ddim-final.png`
- `sdxl-lightning-strip.png`, `sdxl-lightning-final.png`
- `sdxl-heun-strip.png`, `sdxl-heun-final.png`
- `kolors-native-euler-strip.png`, `kolors-native-euler-final.png`
- `kolors-curated-ddim-strip.png`, `kolors-curated-ddim-final.png`
- `kolors-heun-strip.png`, `kolors-heun-final.png`

## Thresholds — what is asserted, and what is deliberately not

sc-16950's `r_first < 0.35` correlation ceiling is **deliberately not ported**. Correlation is taken
over flattened RGB triplets, so it carries channel-mean structure as well as spatial structure — and
this fit's intercept is `(0.556, 0.509, 0.492)`, itself R > G > B, as every warm-lit render also is. A
frame of pre-denoise noise therefore starts at a non-zero, *scene-dependent* floor: +0.215, +0.243 and
+0.203 on the three SDXL lanes, all of which a 0.35 ceiling would have passed by luck and a different
prompt would have failed. `the_fit_intercept_is_warm_not_neutral` is a `const` assertion in
`preview.rs` so the reason stays attached to the constants.

What is asserted instead, on every lane: exact frame numbering `1..=steps`, latent resolution,
per-frame movement above a floor, **strictly** falling distance to the final image, a ≥40% total
reduction in that distance, **strictly** rising resemblance, `r_last − r_first > 0.30`, and a loose
`r_first < 0.75`.

### The `r_last` backstop is derived from the fit, not tuned to the runs

A projection cannot correlate with the decode better than the fit itself does. The 16-channel QwenVae
families wired earlier in this epic carry a **holdout R² of 0.9586** — a correlation ceiling of ~0.979
— and were held to `r_last > 0.85`, i.e. **86.8% of their ceiling**. The four-channel SDXL fit is
demonstrably weaker: **holdout R² 0.86065**, a ceiling of ~0.928. The same *relative* strictness is
`0.928 × 0.85 / 0.979 = 0.805`, so **0.80** is the matched floor, and re-using 0.85 here would have
imposed a strictly harsher gate on a strictly weaker fit for no stated reason.

This matters because the measured values straddle the ported number: Kolors' curated lane lands at
**+0.848** and its native lane at **+0.852**. Moving the floor to 0.845 to clear the first would have
been fitting the threshold to the observation; deriving it from the committed fit statistics is not,
and it lands well clear of both. The floor is the "the strip never got close" backstop — the
load-bearing assertions are the strictly monotone rise and fall around it.

The Lightning lane keeps a further per-lane floor of 0.55 for the schedule reason given above, and
pays for it with the monotone-acceleration assertion no other lane carries.

## Catalog guard — all three amendment steps, in this PR

1. **Descriptors flipped.** `candle-gen-sdxl/src/lib.rs` and `candle-gen-kolors/src/config.rs` now
   advertise `supports_preview: true`. The advertised-true set moves from **7 ids to 9** —
   `krea_2_turbo`, `krea_2_raw`, `krea_2_edit`, `qwen_image`, `anima_base`, `anima_aesthetic`,
   `anima_turbo`, **`sdxl`**, **`kolors`** — across 5 descriptor sites. Nothing else moved.
2. **Route ids added** to `PREVIEW_ROUTE_IDS`, asserted individually.
3. **Route inventories added**, exact per-file `hooked` / `direct` / `dark` tallies:

| crate | file | hooked | direct | dark |
| --- | --- | --- | --- | --- |
| `candle-gen-sdxl` | `denoise.rs` | 1 | 1 | — |
| `candle-gen-sdxl` | `edit_provider.rs` | 0 | 1 | — |
| `candle-gen-sdxl` | `pipeline.rs` | 1 | 1 | — |
| `candle-gen-sdxl` | `training.rs` | 0 | 0 | `run_curated_sampler#0` |
| `candle-gen-kolors` | `control.rs` | 0 | 1 | — |
| `candle-gen-kolors` | `ip_provider.rs` | 0 | 1 | — |
| `candle-gen-kolors` | `pipeline.rs` | 1 | 1 | — |

SDXL is the first crate in the epic to mix both wiring layers, because the registered route has two
denoise lanes: a curated driver call **and** a bespoke Lightning loop. Wiring only the driver call —
which the story as scoped implied — would have left a shipped lane of an advertised route silently
dark. The same is true of Kolors, whose bespoke native leading-Euler lane is its **default**.

The one `DarkSite` is the SDXL trainer's periodic sample render (`training.rs`), which drives the
sampler from a synthetic request carrying no `PreviewSink` and delivers its result as a finished
`TrainingProgress::Sample` image rather than a live stream — the same decision sc-16950 recorded for
Krea's trainer.

## Routes wired — the full list

**SDXL (`sdxl`)** — 5 emitting lanes + 1 dark:

1. `pipeline.rs` `Pipeline::denoise_curated` — the registered route's **default** lane (hook)
2. `pipeline.rs` `Pipeline::denoise_lightning` — the registered route's few-step lane (direct)
3. `denoise.rs` `denoise_curated` — the shared curated helper (hook forwarded from its caller)
4. `denoise.rs` `denoise_ip_multi_control` — the bespoke ancestral loop (direct)
5. `edit_provider.rs` `SdxlEdit::denoise_edit` — img2img / inpaint (direct)
6. `training.rs` `preview_latents` — **dark on purpose**

**Kolors (`kolors`)** — 4 emitting lanes, no dark site:

1. `pipeline.rs` curated lane (hook)
2. `pipeline.rs` native leading-Euler lane — the **default** (direct)
3. `control.rs` pose-control provider: curated (hook into SDXL's helper) + native (direct)
4. `ip_provider.rs` IP-Adapter provider: curated (hook into SDXL's helper) + native (direct)

Four request types gained a `preview: PreviewSink` field, defaulting to inert, because they are driven
by name rather than through the registry — the shape sc-16950 used for `Krea2ControlRequest`:
`SdxlEditRequest`, `IpAdapterSdxlRequest`, `KolorsControlRequest`, `IpAdapterKolorsRequest`.

## Reproduce

```sh
# SDXL — 5 rows (2 provenance, 3 render lanes)
SDXL_PREVIEW_SNAPSHOT=E:\huggingface\hub\models--stabilityai--stable-diffusion-xl-base-1.0\snapshots\4621659… \
SDXL_LIGHTNING_SNAPSHOT=E:\huggingface\hub\models--SceneWorks--realvisxl-lightning-mlx\snapshots\c09fd58…\bf16 \
SDXL_TOKENIZER_CLIP_L_DIR=…\models--openai--clip-vit-large-patch14\snapshots\32bd642… \
SDXL_TOKENIZER_CLIP_BIGG_DIR=…\models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k\snapshots\743c27b… \
SDXL_VAE_FP16_FIX_DIR=…\models--madebyollin--sdxl-vae-fp16-fix\snapshots\207b116… \
SDXL_KOLORS_VAE=…\models--Kwai-Kolors--Kolors-diffusers\snapshots\7e091c7…\vae\diffusion_pytorch_model.fp16.safetensors \
SDXL_PREVIEW_ARTIFACT_DIR=docs\migration\evidence\sc-16954 \
  cargo test -p candle-gen-sdxl --release --features cuda --test preview_real_weights \
    -- --ignored --nocapture --test-threads 1

# Kolors — its own render, because the reuse is a claim about weights
KOLORS_PREVIEW_SNAPSHOT=E:\huggingface\hub\models--Kwai-Kolors--Kolors-diffusers\snapshots\7e091c7… \
KOLORS_PREVIEW_ARTIFACT_DIR=docs\migration\evidence\sc-16954 \
  cargo test -p candle-gen-kolors --release --features cuda --test preview_real_weights \
    -- --ignored --nocapture --test-threads 1
```

Every row **fails rather than skips** when its inputs are absent: a row that early-returns on an unset
variable still reports SUCCESS, and in a run log a skipped gate is indistinguishable from one that ran
and proved something. Asking for `--ignored` is already the opt-in.

## Follow-ups

- **InstantID is unwired.** Wiring it needs a `preview` field on `InstantIdRequest` plus a SceneWorks
  worker that sets it, and it would remain unadvertisable through the catalog (no descriptor). Both
  of its denoise entry points already accept a preview argument, so the engine half is a small change.
- **The four name-driven providers emit but cannot be advertised.** `supports_preview` is
  descriptor-keyed and these have no descriptors, so a consumer cannot discover that the SDXL edit /
  IP-Adapter and Kolors pose / IP-Adapter lanes now stream previews. This is the same gap sc-16952
  recorded for the Qwen-Image edit and Fun lanes; it is a property of the capability surface, not of
  this wiring.
