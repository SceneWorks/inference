# sc-16958 candle SD3.5 latent-preview evidence

Epic 16948 (candle `PreviewSink`), Tier 1. Sibling of epic 16624, which rolled the same seam out
across MLX and committed the fits this story reuses.

The story asked for four things: enumerate every shipped candle SD3.5 route before wiring anything;
confirm — rather than assume — that SD3.5's latents are already spatial and need no unpack step;
ground the reuse of the epic-16624 16-channel fit in tensor bytes; and prove on real weights that
frames develop, one per outer solver step, without changing a single output byte.

## Every denoise lane, enumerated

`candle-gen-sd3` contains **one** shared-driver call site and no bespoke denoise loop at all:

```
$ git grep -n 'run_flow_sampler(\|run_curated_sampler(\|run_scm_sampler(' -- crates/media/candle-gen/candle-gen-sd3/src
crates/media/candle-gen/candle-gen-sd3/src/pipeline.rs:576:    let latents = candle_gen::run_flow_sampler(
```

Re-captured against the committed tree. The open paren is part of the pattern deliberately: without it the
same grep also returns the prose mentions in `pipeline.rs`'s and `preview.rs`'s module docs, none of
which is a call.

That single site, in `pipeline::render_core`, is the **one-site / N-callers** shape: every
user-reachable lane funnels through it.

| route id | descriptor | lanes reaching the site | default lane |
| --- | --- | --- | --- |
| `sd3_5_large` | `descriptor()` | txt2img (true CFG); img2img / `Reference` | txt2img |
| `sd3_5_large_turbo` | `descriptor_turbo()` | txt2img (distilled, no CFG); img2img / `Reference` | txt2img |
| `sd3_5_medium` | `descriptor_medium()` | txt2img (true CFG); img2img / `Reference` | txt2img |

Three registered descriptors × two lanes = **six user-reachable lanes**, wired by hooking one site.

The img2img lane is not a second site. `Sd3Generator::generate` resolves the single
`Conditioning::Reference` and its strength, `pipeline::init_time_step` turns that into a fork step,
and `render_core` blends the VAE-encoded source into `x_t` (`x_t = (1−σ)·clean + σ·noise`) and hands
the driver the reduced `sigmas[start..]` tail — all *before* the driver call. `start_step == 0` (no
reference) is the full schedule and is byte-identical to the pre-img2img path.

Things this crate does **not** have, each checked rather than assumed:

* **No name-driven bespoke provider.** `load_variant` refuses control and IP-adapter overlays
  outright (`spec.control` / `spec.extra_controls` / `spec.ip_adapter` → typed `Unsupported`), so
  there is no descriptor-less render lane here — unlike `candle-gen-flux`, `candle-gen-sdxl`,
  `candle-gen-qwen-image` or `candle-gen-z-image`.
* **No trainer.** Nothing to leave deliberately dark.
* **No second crate.** The story's acceptance criteria assert that `candle-gen-sd3` depends on
  `candle-gen-flux2` and that `candle-gen-lens` depends on both. **Neither holds on candle.**
  `candle-gen-sd3/Cargo.toml` depends on `candle-gen`, `candle-transformers`, `candle-core`,
  `candle-nn`, `tokenizers`, `rand`, `rand_distr`, `safetensors` and `serde_json` — no provider
  crate. `candle-gen-lens/Cargo.toml` depends on `candle-gen-flux2` and `candle-gen-pid`, and not on
  `candle-gen-sd3`; the only mention of SD3 in that file is a comment about a shared workspace pin.
  This matches sc-16955's finding that the dual dependency the epic scoping describes is an MLX-side
  relationship. There is therefore no risk of this story and sc-16955 shipping each other's
  constants: they touch disjoint crates over different latent spaces.

Consequently the route inventory is one row: `pipeline.rs` `hooked: 1, direct: 0, dark: &[]`, over
`Denoise::Shared`.

### CFG never reaches the preview

Large and Medium run true classifier-free guidance as **two separate MMDiT forwards inside the
predict closure**, blended into one velocity (`v_uncond + cfg·(v_cond − v_uncond)`) before the
closure returns. No fused `[2, …]` batch is ever the running latent, so there is no unconditional
half to project — the criterion is closed structurally rather than by a guard. Turbo is
guidance-distilled and runs a single forward. At `cfg_scale == 1.0` the uncond encode and forward are
skipped entirely (sc-8993), which does not change the shape of what the preview sees.

## The latent shape at the emission point — verified, not assumed

The story asked this to be confirmed rather than assumed. **SD3.5 is not packed and needs no unpack
step, and unlike Z-Image it has no frame axis to drop.** The MMDiT's patchify/unpatchify pair lives
entirely inside `transformer::Sd3Transformer::forward`, so the sampler's running latent never enters
the packed token space.

| stage | shape |
| --- | --- |
| seeded noise in `render_core` | `[1, 16, H/8, W/8]` |
| after the img2img blend `x_t = (1−σ)·clean + σ·noise` | `[1, 16, H/8, W/8]` |
| what `run_flow_sampler` integrates and hands the hook | `[1, 16, H/8, W/8]` |
| what `decode_image` hands the VAE | the same tensor |

So there is exactly one geometry in play and the projector needs no `width`/`height` argument: hook
geometry and latent geometry are not merely bound to one source, there is only one source to bind to.
The channel count comes from `vae::LATENT_CHANNELS` — the very constant `render_core` seeds noise
with and the VAE decodes — held by a `const` assertion against the committed factor table's own
length, so this module cannot drift from the denoise about how wide the space is.

The runtime rows prove it independently: every emitted frame is exactly `H/8 × W/8`. A latent that
still needed an unpack or a squeeze could not have produced a frame at that size at all.

## The σ convention: no correction needed

`run_flow_sampler` integrates a `FlowModelSampling`, whose `input_scale` is **identically 1.0** at
every σ, so the running latent already *is* the tensor the fit was measured against and
`PreviewHook::new` is the correct constructor rather than `PreviewHook::with_sigma`.

`the_flow_cohort_needs_no_sigma_correction` reads `input_scale` off the very `ModelSampling` the
driver integrates rather than asserting it about the family in prose, confirms the pipeline's own
schedule starts at `σ_max = 1.0`, and then measures the consequence sc-16954 named — the first
frame's rail-clipped fraction on this family's own unit-normal prior:

```
  flow prior at sigma_max: rail-clipped fraction 0.0065
```

**0.0065** against sc-16954's uncorrected SDXL **0.894** — 137× less, and the difference between a
readable noise field and a saturated binary one. It is not zero and should not be: on a unit-normal
prior the projection's per-channel spread is `σ_(R,G,B) = (0.169, 0.135, 0.156)` about the fit's
intercept `(0.646, 0.626, 0.615)`, which puts the upper rail 2.09 / 2.78 / 2.47 σ away, so ≈0.9% of
pixels clip by construction. The row's own bound is `rails < 0.05`, chosen loose enough that a
rounding change cannot flip it. This is the only non-`#[ignore]`d row in
`tests/preview_real_weights.rs` and runs on the committed constants alone, so it appears in a plain
`cargo test` of the file. sc-16954 shipped a red row that hid because the sole non-ignored row in its
file was excluded by `-- --ignored`; both invocations are reported below.

## The fit is reused, not refitted

`RGB_FACTORS` / `RGB_BIAS` in `candle-gen-sd3/src/preview.rs` are the epic-16624 constants transcribed
verbatim from `mlx-gen-sd3/src/preview.rs`. Candle ships **no producer** — `mlx-gen-sd3`'s
`tests/fit_preview_rgb.rs` remains the only way they are re-derived.

Fit R² `(R,G,B) = (0.97402, 0.98248, 0.98505)`, overall `0.98031`; holdout R²
`(0.86452, 0.87254, 0.95631)`, overall `0.91459` — measured on four real-weight SD3.5-Large renders
with two disjoint prompt/seed holdouts, all 256² at eight static-shift-3 flow-Euler steps, CFG 3.5,
against 8×8-average-pooled native VAE decodes.

### One fit covers three routes — by artifact identity

`the_three_sd3_snapshots_ship_one_identical_vae` requires all three snapshots and hashes each:

```
  sd3_5_large        vae/  8f53304a79335b55e13ec50f63e5157fee4deb2f30d5fae0654e2b2653c109dc  167666902 bytes
  sd3_5_large        vae/config.json  58557f2439dfa867450caef425b5d11160be8aa9c34d60dbf23a94a6a94cb060
  sd3_5_large_turbo  vae/  8f53304a79335b55e13ec50f63e5157fee4deb2f30d5fae0654e2b2653c109dc  167666902 bytes
  sd3_5_large_turbo  vae/config.json  58557f2439dfa867450caef425b5d11160be8aa9c34d60dbf23a94a6a94cb060
  sd3_5_medium       vae/  8f53304a79335b55e13ec50f63e5157fee4deb2f30d5fae0654e2b2653c109dc  167666902 bytes
  sd3_5_medium       vae/config.json  58557f2439dfa867450caef425b5d11160be8aa9c34d60dbf23a94a6a94cb060
```

| repo | revision |
| --- | --- |
| `stabilityai/stable-diffusion-3.5-large` | `ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f` |
| `stabilityai/stable-diffusion-3.5-large-turbo` | `ec07796fc06b096cc56de9762974a28f4c632eda` |
| `stabilityai/stable-diffusion-3.5-medium` | `b940f670f0eda2d07fbb75229e779da1ad11eb80` |

The Large revision is the one the epic-16624 fit was measured on. The other two are separate
repositories at separate revisions, which is exactly why the row requires all three rather than
checking one and assuming the rest. `vae/config.json` is pinned alongside the weights because the
`1.5305` / `0.0609` normalization is half the definition of the fitted space — a snapshot that kept
the weights but re-scaled them would project wrong while passing a weights-only check. The row also
asserts the engine's own `vae::SCALING_FACTOR` / `vae::SHIFT_FACTOR` are the values that config
carries.

## Which 16-channel space SD3.5 occupies — its own

Epic 16948 asked every 16-channel story to settle this explicitly, because the two before it did not
land where their scoping expected. sc-16956 found Boogu's 16-channel VAE to be FLUX.1's
(`max |Δ| = 0.0`, a pure bf16→f32 upcast); sc-16957 found Z-Image's to be the *same file* as
FLUX.1-dev's, SHA-256 `f5b59a26…40a3`, meaning epic 16624 had committed two fits over one latent
space (tracked as sc-17309).

**SD3.5 is the opposite finding, and it is the first genuinely distinct 16-channel space in this
epic.** It could only be reached by walking tensors: the two containers are the same architecture,
carry the same 244 keys at the same shapes in the same bf16 dtype, and are the **same
167,666,902-byte size**. A size, key-set, shape or channel-count comparison would say "identical"
about two different VAEs.

`the_sd3_vae_is_not_the_flux1_latent_space`:

```
  sd3.5     vae/  8f53304a79335b55e13ec50f63e5157fee4deb2f30d5fae0654e2b2653c109dc
  flux1-dev vae/  f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3
  walked 244 tensors / 83819683 values: 0 identical, 244 differing
```

The walk compares each tensor's **raw payload bytes**, read straight out of the two containers rather
than widened to `f32` through candle. That distinction is load-bearing for a strong-form "none match"
assertion: `Vec<f32>` equality reports a genuinely identical pair as *differing* if either holds a
NaN, which is the one way the row could have passed vacuously. The row also pins the walked payload at
167,639,366 bytes — `83,819,683 × 2`, the bf16 arithmetic that says every value in both containers was
compared, and the same tensor region the independent header-level walk below hashes.

An independent header-level walk over the same two files (payload SHA-256 over the identical
167,639,366-byte tensor region):

| | SD3.5-Large | FLUX.1-dev |
| --- | --- | --- |
| container SHA-256 | `8f53304a…c109dc` | `f5b59a26…d40a3` |
| container bytes | 167,666,902 | 167,666,902 |
| tensors / dtype | 244 / BF16 | 244 / BF16 |
| payload bytes | 167,639,366 | 167,639,366 |
| payload SHA-256 | `52163be5…7bb2` | `44b97a3d…bcb9` |
| `scaling_factor` / `shift_factor` | 1.5305 / 0.0609 | 0.3611 / 0.1159 |
| `vae/config.json` `_name_or_path` | `../sdxl-vae/` | `../checkpoints/flux-dev` |

Same architecture, different trained weights, different normalization. Two consequences, both worth
stating because both were open questions the epic named:

1. **sc-17309 must not gain an SD3.5 row.** It tracks the Z-Image / FLUX.1 duplication — two fits over
   one space. This is not a third fit over that space; it is a second space.
2. **Sharing a Rust type says nothing about the latent space.** `candle-gen-sd3` reuses
   `candle_transformers::models::z_image::vae::AutoEncoderKL` precisely because the *architecture* is
   shared and the *weights* and scale/shift are not. That is exactly the "matching Rust type"
   reasoning the epic forbids grounding a reuse in, and this crate is the clearest example of why.

The test asserts the strong form — **every** one of the 244 tensors differs — rather than "the hashes
differ". A single matching tensor would mean the two lineages overlap and the reasoning above would
need revisiting; the assertion message says so.

## Wiring

`Pipeline::render` builds one hook from `req.preview` and hands it to each per-seed `render_core`
call. The driver builds a fresh `PreviewCounter` per call, so a batched request restarts each image's
trajectory at frame 1 rather than continuing the previous one's numbering.

Opting in is the sc-16949 projector hook and nothing else — neither the render loop nor `render_core`
changes shape. `candle-gen-sd3/src/preview.rs` holds only the reused fit and a layout check; there is
no unpack, no squeeze and no de-normalize, because the running latent is already the `[1, C, h, w]`
contract `candle_gen::preview::project_latents` takes.

### Catalog guard, all three steps in this PR

1. `supports_preview` flipped to `true` in `descriptor_for`, covering all three variants (they share
   one descriptor builder and one sampler site, so the flag is variant-independent).
2. `sd3_5_large`, `sd3_5_large_turbo` and `sd3_5_medium` added to `PREVIEW_ROUTE_IDS` in
   `candle-gen-catalog`, which asserts each id individually against the shipped descriptors in both
   directions. The advertised-true set goes from 22 to **25**; nothing else moved.
3. The `candle-gen-sd3` `ProviderCrate` row gained its route inventory —
   `FileRoutes { file: "pipeline.rs", hooked: 1, direct: 0, dark: &[] }` — declaring
   `Denoise::Shared`.

### Guards, and what happens when they are mutated

Blanking the one hook (`preview` → `None` at the `run_flow_sampler` site) trips two catalog tests by
name, so the inventory is not decorative:

```
test preview_advertising::every_wired_crate_pins_its_exact_route_inventory ... FAILED
test preview_advertising::source_level_wiring_and_advertised_capability_agree_for_every_provider_crate ... FAILED

candle-gen-sd3 pins a route inventory but emits no previews — an inventory only means something on a
wired crate
```

The preview argument's **position** is pinned separately by the catalog's
`the_sampler_driver_signatures_pin_the_preview_argument_position`, which re-derives `preview_at: 7`
and `arity: 9` from the driver's own signature, and by `sampler_sites`, which classifies the parsed
argument at that index rather than searching the call text — so a moved or mis-split argument is a
loud failure, not a silent "no hook". sc-16957 shipped a guard that pinned an emitter's sink argument
but not its index argument; there is no direct emission call in this crate, so the only positional
surface is the driver argument, and it is pinned.

### The hop the catalog cannot see — closed here and in `candle-gen-sdxl`

The paragraph above is about the driver argument *inside* `render_core`. The hook reaches it through
a second hop — `Pipeline::render` → `render_core` — and the catalog never looks there. While that hop
was an `Option`, changing one caller argument from `Some(&preview)` to `None` took **all six** SD3.5
lanes preview-dark while `hooked: 1`, all three `PREVIEW_ROUTE_IDS` rows and `supports_preview: true`
kept advertising and the full CPU suite stayed green (122 tests, 0 failures, including all twelve
`preview_advertising` guards). The rows that would have caught it are `#[ignore]`d and CUDA-only, so
CI never saw it.

Two changes close it:

* **`candle-gen-sd3`** — `render_core`'s parameter is now `&PreviewHook`, not `Option<&PreviewHook>`.
  Going dark is a type error, the way `candle-gen-chroma` has always been immune. The crate has no
  deliberately dark lane to express (no trainer, no descriptor-less provider), so the `Option` bought
  nothing. `preview.rs`'s `the_render_lane_builds_its_hook_from_the_requests_sink` pins the parameter
  against the declaration so the immunity cannot be quietly widened away, and pins that exactly one
  hook is built in shipped code and built over `req.preview` — the one remaining way to go dark
  without changing a type. Both halves were mutation-proven: restoring the `Option` and blanking the
  caller fails the parameter assertion; building the hook over an inert sink fails the sink assertion.
* **`candle-gen-sdxl`** — the same blankable shape exists in already-merged code:
  `denoise::denoise_curated` takes `Option<&PreviewHook>` and forwards it, so the catalog's
  `denoise.rs hooked: 1` is really about *its* argument, not its callers'. `ip_provider.rs` is the one
  shipped caller and blanking its `Some(&preview)` is invisible everywhere. The type change is not
  available there (`denoise_curated` is `pub`, `candle-gen-kolors` and `candle-gen-instantid` reach it
  too, and InstantID passes `None` on purpose), so `candle-gen-sdxl/src/preview.rs` gains the
  crate-local pin instead: every caller in the crate is classified positionally by the argument in the
  preview slot, with the argument's index re-derived from `denoise_curated`'s own declaration.
  Mutation-proven the same way — blanking `ip_provider.rs` fails the inventory with
  `("ip_provider.rs", "None")` against `("ip_provider.rs", "Some(&preview)")`.

## Real-weight run (CUDA)

2× RTX PRO 6000, CUDA compute cap 120, MSVC 14.44 (VS2022 BuildTools vcvars64), snapshots from
`E:\huggingface`. Every route rendered **twice on one warmed generator at the same seed** — once with
an inert sink, once with a live one — and the two outputs compared byte for byte.

```sh
cargo test --locked --release -j 1 -p candle-gen-sd3 --features cuda \
  --test preview_real_weights -- --ignored --nocapture
...
test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 427.30s
```

| lane | route | steps | size | frames | r first → last | mean \|Δ\| to final, first → last (ratio) |
| --- | --- | --- | --- | --- | --- | --- |
| txt2img | `sd3_5_large` | 12 | 1024² | 12 | +0.348 → **+0.979** | 80.09 → 19.00 (0.237) |
| txt2img | `sd3_5_large_turbo` | 8 | 1024² | 8 | +0.250 → **+0.987** | 78.53 → 16.24 (0.207) |
| txt2img | `sd3_5_medium` | 12 | 1024² | 12 | +0.459 → **+0.985** | 95.59 → 12.11 (0.127) |
| img2img @ 0.6 | `sd3_5_large` | 12 req. | 1024² | **5** | +0.954 → **+0.979** | 56.30 → 18.91 (0.336) |
| txt2img `heun` | `sd3_5_large` | 8 | 768² | 8 | +0.377 → **+0.984** | 85.36 → 19.04 (0.223) |

Both series are strictly monotone on every lane, every frame differs from its predecessor by more
than the movement floor, and every frame arrives at VAE-latent resolution `H/8 × W/8` — which is
itself the runtime proof that no unpack or squeeze was needed. Strips and finals are in
`docs/migration/evidence/sc-16958/`.

The two rows the review round touched — img2img (`max_distance_ratio` tightened 0.42 → **0.40**) and
`heun` (the evaluation count moved ahead of the strip assertions) — were **re-run on the same box**
against the same seed and reproduce every figure above exactly: img2img 56.30 → 18.91 (ratio 0.336),
`heun` 85.36 → 19.04 over 15 evaluations for 8 outer steps.

```
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 6 filtered out; finished in 178.19s
```

**Inert-sink byte-identity holds on all five lanes.** An extra confirmation fell out of the img2img
row: its source image is a `sd3_5_large` txt2img render at the same seed, and it came out SHA-256
`d835f768…6ee4` — byte-identical to the `sd3_5_large` final produced in a *different process run*.
The duplicate file is therefore not committed; `sd3_5_large_1024_s12_final.png` is that image.

### Exactly one frame per outer step, proven non-vacuous

```
── sd3_5_large_heun: 768² × 8 steps, sampler Some("heun")
  heun: 15 evaluations for 8 outer steps
```

The shared driver calls `on_progress` once per **evaluation** — `sampler.rs` recomputes the step count
on every eval and deliberately repeats it — so counting `Progress::Step` events is counting
evaluations. The row asserts `evaluations > steps` **first** (15 > 8), then that the strip is numbered
exactly `1..=8`. Without the first assertion a solver that happened to evaluate once per step would
make the second prove nothing about dedup. 15 rather than 16 because Heun's final step degenerates to
Euler at σ = 0.

That ordering is structural rather than a description of intent: the count is checked through
`render_and_assert`'s `non_vacuity` callback, which runs ahead of every assertion in that function —
`assert_the_strip_converges`, where the frame numbering is actually pinned, included. The transcript
above shows it: the evaluation line is the first thing the row prints, before any strip measurement.

### The img2img lane, and what it changed about the thresholds

The img2img row is the second lane every route has, and running it turned up something worth
recording rather than papering over. Two of the shared "the strip develops" bounds silently assume the
trajectory starts from **pure noise**:

```
sd3_5_large_img2img: the first frame is pre-denoise noise and must not already BE the render (r +0.954)
```

That is correct behaviour, not a defect. At strength 0.6 the fork skips
`init_time_step(12, 0.6) = 7` schedule nodes and blends the VAE-encoded source into `x_t`, so the
first emitted frame already carries the target's structure and the assertion's own message is simply
false for this lane. Rather than loosen `max_r_first` / `min_rise` for everyone — which would have
weakened the four txt2img lanes to accommodate one — that lane declares its own bounds and the weight
shifts onto `max_distance_ratio`, which still discriminates hard: a strip that had opened on the
finished image would sit near 1.0, not 0.336.

The same run also caught the harness assuming the emitted frame count equals the requested step count.
It does not on a forked lane: the driver only ever sees `sigmas[start..]`, so this row emits
`12 − 7 = 5` frames. The expectation is now derived from `pipeline::init_time_step` — the very
function the fork uses, made `pub` for exactly this — so the assertion stays an **exact equality**
rather than being loosened to a range that would accept a genuinely wrong count.

### Per-lane floors

Every floor is derived from that lane's own measured run, with uniform stated headroom: **0.03 under a
measured correlation, 0.06 over a measured distance ratio**. No number is transferred between lanes.

| lane | `min_r_last` | `max_r_first` | `min_rise` | `max_distance_ratio` |
| --- | --- | --- | --- | --- |
| `sd3_5_large` | 0.949 | 0.75 | 0.30 | 0.30 |
| `sd3_5_large_turbo` | 0.957 | 0.75 | 0.30 | 0.27 |
| `sd3_5_medium` | 0.955 | 0.75 | 0.30 | 0.18 |
| `sd3_5_large` img2img | 0.949 | **0.97** | **0.015** | **0.40** |
| `sd3_5_large` heun | 0.954 | 0.75 | 0.30 | 0.28 |

The img2img row is the one place where that headroom rule has to hold tightest, because it is also the
one lane whose `max_r_first` and `min_rise` were deliberately relaxed — `max_distance_ratio` carries
the discriminating weight there. `0.336 + 0.06 = 0.396`, rounded to **0.40**; the other four sit at
0.063 / 0.063 / 0.053 / 0.057 over their own measurements.

`max_r_first = 0.75` and `min_rise = 0.30` are shared across the from-noise lanes deliberately: a
tighter first-frame ceiling would read the fit's own warm intercept (0.646, 0.626, 0.615 — R > G > B,
as most warm-lit renders are) as if it were resemblance, which is why sc-16950's `r_first < 0.35` is
not ported.

### Both invocations of the harness, reported separately

sc-16954 shipped a red row that hid because the only non-`#[ignore]`d row in its file was excluded by
`-- --ignored`, so both are run and both are reported:

| invocation | result |
| --- | --- |
| `cargo test -p candle-gen-sd3 --test preview_real_weights` (no `--ignored`) | **1 passed, 0 failed, 7 ignored** — `the_flow_cohort_needs_no_sigma_correction` |
| `… --features cuda … -- --ignored` | **7 passed, 0 failed, 1 filtered out** |

The two sets are disjoint and together cover all eight rows.

## Limits of this evidence

* The fit is reused, not re-derived. Nothing here re-measures the epic-16624 OLS solution; the rows
  above establish that candle loads the bytes it was measured on, not that the solution is optimal.
* The resemblance floors are per-lane and empirical. Each names the run it came from, and each sits
  under that lane's own measured `r_last`. They are regression floors, not statements about how good
  a preview must look.
* A preview frame is a global linear approximation of the decode (holdout R² 0.9146), so absolute
  distance to the finished image can only ever say "closer", never "resembles". That is why every row
  layers a coarse-thumbnail correlation rise on top of the falling distance.
* Preview failures are decorative by contract: a projection error loses one frame, consumes its
  schedule position, and never reaches the caller's render.
