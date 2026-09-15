# sc-16959 — candle SANA per-step latent previews (epic 16948)

Wires `PreviewSink` into **both** shipped `candle-gen-sana` routes, reusing epic 16624's two committed
SANA 32-channel fits rather than refitting either. This is the last Tier 1 story, the only candle
family in the epic that drives **two** sampler drivers, and the only one carrying **two** fits for one
crate.

Base branch `origin/main` @ `6427219a`.

## Every denoise lane, enumerated

`git grep` of all three shared drivers plus every bespoke emission call across
`crates/media/candle-gen/candle-gen-sana/src`:

```
run_flow_sampler(   → pipeline.rs, in `denoise_cfg`      (1 hit)
run_scm_sampler(    → pipeline.rs, in `denoise_sprint`   (1 hit)
run_curated_sampler( / run_av_curated_sampler(           (0 hits)
emit_preview / emit_preview_at / .emit( / .emit_step(    (0 hits)
hand-written `for`-loop denoise                          (none)
```

Two hits, both in `pipeline.rs`, and no bespoke loop anywhere in the crate.

| route id | driver | denoise | fit | user-reachable lanes | default |
| --- | --- | --- | --- | --- | --- |
| `sana_1600m` | `run_flow_sampler` | true-CFG flow-match Euler, static shift 3.0 | `BASE_RGB_*` | 1 (txt2img) | 20 steps, guidance 4.5, native Euler |
| `sana_sprint_1600m` | `run_scm_sampler` | CFG-free SCM / TrigFlow consistency | `SPRINT_RGB_*` | 1 (txt2img) | 2 steps, embedded guidance 4.5 |

**Two lanes, one per route, one call site each.** Both `load` functions refuse quantization,
LoRA/LoKr and control / IP-adapter overlays outright, so there is no img2img fork, no name-driven
provider and no descriptor-less render lane. The crate ships **no trainer**, so unlike Krea / Lens /
SDXL / Z-Image it has **no deliberately dark site**: `pipeline.rs` is `hooked: 2, direct: 0, dark: []`.

The base lane is reachable under the whole curated epic-7114 sampler menu, which is where the
multi-eval dedup matters — `heun` evaluates twice per outer step through that same single call.
Sprint advertises only the `"default"` sentinel, because the SCM consistency loop is not a curated
`Solver` at all.

**CFG never reaches the preview.** Base SANA runs true CFG as two separate trunk forwards *inside*
`denoise_cfg`'s predict closure, blended into one velocity before the solver advances, so no `[2, …]`
batch is ever the running latent. Sprint has no unconditional branch at all — its guidance is an
embedded scalar handed to the trunk's guidance embedder — so there is no fused half to go looking for.
`tests/preview_wiring.rs::the_base_preview_never_sees_a_fused_unconditional_half` measures this on a
real trunk rather than asserting it.

## Two fits, and *why* two are needed — measured, not assumed

The story's named mistake is shipping one fit for both routes. The reuse is grounded in tensor bytes
per route, and it is the strongest grounding in this epic: **the candle route loads the identical file
the MLX fit was measured on, in both directions.**

| | base | Sprint |
| --- | --- | --- |
| candle snapshot | `Efficient-Large-Model/Sana_1600M_1024px_diffusers` @ `d1b54936…a19a38` | `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` @ `b3c9ce6f…5ca934` |
| `vae/diffusion_pytorch_model.safetensors` SHA-256 | `15a4b09e56d95b768a0ec9da50b702e21d920333fc9b3480d66bb5c7fad9d87f` | `dfd991d1b54ffabf22745c5885589d8f2a7bc59930d95d92bd741c4fc64454bb` |
| container bytes | 1,249,044,836 | 1,249,044,836 |
| MLX donor the fit was measured on | `SceneWorks/Sana_1600M_1024px_mlx` — **same hash** | `SceneWorks/Sana_Sprint_1.6B_1024px_mlx` — **same hash** |
| committed fit | `BASE_RGB_FACTORS` / `BASE_RGB_BIAS` | `SPRINT_RGB_FACTORS` / `SPRINT_RGB_BIAS` |

Both MLX re-hosts were hashed directly out of `E:\huggingface` alongside the two diffusers snapshots;
all four files are 1,249,044,836 bytes and collapse to exactly these two hashes.

### The tensor walk: one latent space, two decoders

The two containers are the same size with the same 375 keys, shapes and dtype, so nothing short of a
tensor walk could say how they relate. `the_two_dc_aes_share_one_encoder_and_differ_only_in_the_decoder_tail`:

```
  base   vae/  15a4b09e56d95b768a0ec9da50b702e21d920333fc9b3480d66bb5c7fad9d87f
  sprint vae/  dfd991d1b54ffabf22745c5885589d8f2a7bc59930d95d92bd741c4fc64454bb
  walked 375 tensors / 312250275 values: 320 identical, 55 differing
```

This is a **fourth** relation, and none of this epic's earlier stories would have predicted it:
sc-16956 found Boogu's "16 channels" to be FLUX.1's, sc-16957 found Z-Image's VAE to be *literally the
same file* as FLUX.1-dev's, sc-16958 found SD3.5's to be a wholly different VAE at an identical size.
SANA's two DC-AEs **partially overlap**:

- **320 of 375 tensors are byte-identical**, including the **entire encoder** — all 179 of its tensors
  — plus `decoder.conv_in` and the whole of `decoder.up_blocks.3`, the decoder stage closest to the
  latent;
- **55 differ, and every one of them is decoder-side**: `decoder.up_blocks.0`, `.1`, `.2`,
  `decoder.norm_out` and `decoder.conv_out` — the last three upsampling stages and the output head.

So DC-AE 1.1 (Sprint) is a **decoder-tail fine-tune** of DC-AE 1.0 (base). The encoder is what defines
the latent space and it is unchanged — but an RGB preview fit is a least-squares map from a latent to
that autoencoder's **decoded pixels**, and the decode is exactly what was retrained. One fit therefore
cannot serve both routes, and the reason is sharper than "two latent spaces": it is **one latent space
with two decoders**.

That also explains why the two committed tables look structurally alike while differing numerically in
every row — which is precisely what makes a copy-paste between them plausible, and the guards below
worth having. The row pins the overlap as an **exact pair** (320 / 55), because *both* endpoints would
falsify the reasoning: `0 differing` would mean one fit could serve both routes, and `375 differing`
would contradict the shared-encoder finding.

The walk compares each tensor's **raw payload bytes**, read straight out of the containers rather than
widened to `f32` through candle: `Vec<f32>` equality reports a genuinely identical pair as *differing*
if either holds a NaN, which is the one way an overlap count could come out wrong. It pins the walked
payload at 1,249,001,100 bytes over 312,250,275 values, so a partial walk cannot pass as a full one.

## The latent shape at the emission point — verified per route

| stage | base (`run_flow_sampler`) | Sprint (`run_scm_sampler`) |
| --- | --- | --- |
| `pipeline::create_noise` | `[1, 32, H/32, W/32]` | `[1, 32, H/32, W/32]` |
| what the driver hands the hook | `[1, 32, H/32, W/32]` | `[1, 32, H/32, W/32]`, **pre-scaled by `σ_data`** |
| what `decode_to_image` hands the DC-AE | `[1, 32, H/32, W/32]` | `[1, 32, H/32, W/32]` |

No packed token space (the Linear-DiT trunk patchifies inside its own forward), no frame axis, so
neither route needs an unpack and neither needs a gated squeeze. The channel count is taken from
`pipeline::LATENT_CHANNELS` and the spatial edge from `pipeline::SPATIAL_SCALE`, the very constants
`create_noise` builds with and the decoder decodes — so `preview.rs` cannot come to disagree with the
denoise about the geometry. Frames therefore arrive at 32×32 for a 1024² render: that is the
deep-compression autoencoder's own latent resolution, confirmed at runtime by every strip below.

## The σ conventions differ between the routes — and only one needs a correction

- **Base needs none.** `run_flow_sampler` integrates a `FlowModelSampling`, whose `input_scale` is
  identically `1.0`, so the running latent already *is* the tensor the base fit was measured against
  and `PreviewHook::new` is correct rather than `with_sigma`. Read off the `ModelSampling` itself.
- **Sprint needs `1/σ_data`.** `run_scm_sampler` multiplies the seed latent by `σ_data` on entry and
  divides it back out on exit, handing the hook the **scaled** running latent — the warning sc-16949
  left in the driver's rustdoc and pinned with
  `scm_preview_receives_the_sigma_data_scaled_running_latent`. `preview::sprint_hook` therefore carries
  `SPRINT_INVERSE_SIGMA_DATA`, derived from `candle_gen::SCM_SIGMA_DATA` rather than restated as `2.0`,
  and `the_scm_scheduler_always_carries_the_sigma_data_this_correction_inverts` binds the constant to
  the value the driver actually divides by, through both public `ScmScheduler` constructors across the
  whole 1–8 step band. This is the candle spelling of `mlx-gen-sana`'s `inverse_sigma_data` argument.

`the_two_routes_sigma_conventions_are_what_the_projectors_assume` (the file's only non-`#[ignore]`d
row) measures the consequence on each route's own prior:

```
  base   flow prior at sigma_max: rail-clipped fraction 0.0007
  sprint SCM prior (corrected):   rail-clipped fraction 0.0003
  sprint SCM prior (uncorrected): rail-clipped fraction 0.0000
  sprint spread about the intercept: corrected 25.93, uncorrected 12.96
```

Both shipped projections are readable noise fields, far below sc-16954's uncorrected SDXL **0.894**.

**A correction to the epic's own prediction, worth recording.** The `run_scm_sampler` rustdoc described
the uncorrected Sprint preview as "2× too bright", and the rail-clipped fraction is *not* the statistic
that catches it: `σ_data = 0.5` **shrinks** the running latent, so an uncorrected Sprint frame collapses
*toward* the fit's intercept rather than toward the rails — its rail fraction is `0.0000`, lower than
the corrected one. The statistic that discriminates is contrast about the intercept, and it is exactly
2× as the arithmetic predicts (25.93 / 12.96 = 2.0008). That is what the row asserts, alongside the
exact identity `project_sprint_latents(x·σ_data, 1/σ_data) == project_sprint_latents(x, 1)`.

**That sentence is corrected in this PR**, in `candle-gen/src/sampler.rs` — not merely recorded here.
It sat in the **shared** driver's rustdoc, which is exactly where the next family wiring an SCM route
would read it, and this story owns the measurement that disproves it. The driver now says an
uncorrected frame is *flatter*, not brighter, gives all three rail fractions, and names contrast about
the intercept as the statistic that discriminates.

## Wiring, and the whole hook path guarded

The hook is built **once per route**, over the request's own sink, in `model.rs`:

- `SanaGenerator::generate` → `preview::base_hook(&req.preview)` → `generate_base_images` →
  `render_seed` → `SanaPipeline::generate_with_conditioning` → `denoise_cfg` → `run_flow_sampler`
- `SanaSprintGenerator::generate` → `preview::sprint_hook(&req.preview)` → `generate_sprint_images` →
  `render_seed` → `SanaSprintPipeline::generate_with_conditioning` → `denoise_sprint` →
  `run_scm_sampler`

**Every hop takes the hook as a non-`Option` `&PreviewHook<'_>`.** sc-16958's reviewer showed what an
`Option` anywhere on that path costs: blanking one caller argument took an entire family dark while the
`hooked` counts, `PREVIEW_ROUTE_IDS` and `supports_preview: true` all went on advertising and the whole
CPU suite stayed green — because `candle-gen-catalog`'s route inventory classifies only the argument at
the *driver* call, several hops further in. Here *widening the seam* is a **type error**.

`preview::tests::both_render_lanes_build_their_hook_from_the_requests_sink` pins all of it against the
crate's own sources: exactly one `preview::base_hook(&req.preview)` and one
`preview::sprint_hook(&req.preview)` in shipped `model.rs` and no third hook anywhere in it; the base
constructor inside `generate_base_images`'s span and the Sprint one after `generate_sprint_images`, so
a swap is a diff; and the exact `preview: &candle_gen::preview::PreviewHook<'_>,` parameter read out of
six declarations in `model.rs` and six in `pipeline.rs`. Shipped `pipeline.rs` builds exactly two
hooks, both the documented **inert** ones in the `generate` convenience wrappers, pinned by count so a
request-sink hook cannot appear there instead.

### What the types do *not* cover — and how far the text goes (sc-16959 review)

The first draft of this section, and of that test's rustdoc, claimed the sinks were the "one way left
to go dark without changing a type". That was **false**, and sc-16959's reviewer demonstrated it by
taking both lanes dark with **zero** type errors and the full CPU + catalog suites green:

- `impl BaseBatchPipeline for SanaPipeline`'s `render_seed` accepted the forwarded hook, **ignored**
  it, and built a fresh `candle_gen::preview::PreviewHook::new(&inert, …)`. The `_hook(` count of two
  could not see it, because `PreviewHook::new(` does not contain the substring `_hook(`.
- `SanaGenerator::generate` rebound `let req = &GenerationRequest { preview: PreviewSink::default(),
  ..req.clone() };` ahead of `preview::base_hook(&req.preview)`. The literal the scan counts was still
  there, exactly once — over a sink that had been emptied.

The root cause is coverage, not typing: nothing on the CPU lane renders through the registered
`Generator` seam with a live sink (`tests/preview_wiring.rs` enters at `denoise_cfg` / `denoise_sprint`,
because everything above them needs a loaded snapshot), so the whole `model.rs` adapter layer is
guarded by text. This PR therefore does two things:

1. **Names the boundary honestly.** The non-`Option` typing is what makes *widening* the seam — an
   `Option` hop, a `None` at the driver — impossible without a diff to a signature. It is not, and was
   never, an absolute immunity against a body that ignores what it was handed.
2. **Closes both demonstrated spellings by count.** Shipped `model.rs` must contain **zero**
   `PreviewHook::new` and **zero** `GenerationRequest {`; shipped `pipeline.rs` must contain **zero**
   `PreviewHook::new` (all three are 0 today). The other spelling of the first mutation —
   `preview::base_hook(&inert)` inside `model.rs` — was already caught, by the `_hook(` count of two.

The same review found the `WANT` parameter tally was a **substring** match, so a hop renamed
`_preview:` — precisely what a hop looks like once it stops using its hook — still counted toward the
six. Both tallies now compare whole trimmed lines instead. `preview_parameter` reads only the *first*
`fn render_seed(` (the trait declaration), so that line-exact count is what holds the other three.

What remains uncaught is an edit that reaches the same end by a third construction (a helper returning
an emptied request; `GenerationRequest{` with no space). Closing that needs a render through the
registered `Generator` seam with a live sink, which needs weights — which is exactly what the
real-weight lane below does, on CUDA, and it is the only place this crate proves the seam end to end.

`both_sampler_calls_pass_the_hook_at_the_drivers_preview_position` pins the two driver arguments
**positionally, including the index**: argument 7 of 9 for `run_flow_sampler` and argument 5 of 7 for
`run_scm_sampler`, both `Some(preview)`, with the argument list parsed rather than string-matched — so
an argument inserted ahead of the hook fails here rather than shifting what the row believes it reads.

## Catalog guard — all three steps in this PR

1. **`supports_preview` flipped** on both `candle-gen-sana` descriptors (`model.rs`), each with a
   comment naming *its own* fit.
2. **`PREVIEW_ROUTE_IDS` gains `sana_1600m` and `sana_sprint_1600m`**, and — the point of this story —
   they are asserted **individually**. The pre-existing guards did not catch a collapse: the
   bidirectional row is a set comparison satisfied by the pair collectively, and
   `source_level_wiring_and_advertised_capability_agree_for_every_provider_crate` only needed *one* id
   of a crate to advertise. See the next section — that hole is not SANA's, and it is now closed for
   every crate.
3. **Route inventory row added**: `candle-gen-sana` → `pipeline.rs` `hooked: 2, direct: 0, dark: []`.
   The only row in that table whose two hooked sites are two *different* drivers.
   `sana_base_and_sprint_are_two_independent_rows` binds that to the two drivers by name: the crate's
   hooked sites must be exactly `["run_flow_sampler", "run_scm_sampler"]`, so one route going dark
   while the other keeps both ids advertising is a failure. That assertion is genuinely SANA-specific
   — no other candle family in this epic drives two shared samplers — and nothing else asserts it.

**Advertised-`supports_preview` set is now exactly 27** — the 25 before this story plus the two SANA
routes. `preview_capability_matches_every_wired_shipped_route_bidirectionally` asserts the registry's
advertising set equals `PREVIEW_ROUTE_IDS` exactly, in both directions, so nothing else moved.

### The per-id hole was never SANA's — generalised (sc-16959 review)

sc-16959's reviewer took the observation above and showed it live **on already-merged code**. Because
the bidirectional row is a set equality and the source-level row only asked for a non-empty
`advertised`, dropping one id from *both* sides of a multi-id crate left everything green. Reproduced
on merged `candle-gen-sd3`: change `supports_preview` to `!matches!(variant, Variant::Medium)` and
delete `"sd3_5_medium"` from `PREVIEW_ROUTE_IDS`, and the whole catalog suite exits **0 with zero
failures** — while `sd3_5_medium`, which reaches the same hooked `run_flow_sampler` site as its two
siblings, silently stops advertising a capability it demonstrably has.

Ten crates ship more than one id and were all exposed: krea ×3, anima ×3, chroma ×3, sd3_5 ×3, flux ×2,
flux2 ×2, lens ×2, ideogram ×2, z-image ×2, sana ×2.

So the per-id half is now **generalised** rather than SANA-shaped: the wired branch of
`source_level_wiring_and_advertised_capability_agree_for_every_provider_crate` loops over every id
`ids_of(provider)` reports and requires each, on its own, to be in `PREVIEW_ROUTE_IDS` *and*
advertising. Thirteen wired crates register **27** ids between them, which is exactly
`PREVIEW_ROUTE_IDS.len()` — the declared and derived halves meet with nothing left over. No
already-merged crate fails the generalised guard. `sana_base_and_sprint_are_two_independent_rows`
keeps only what is SANA's: the two-driver pair, plus binding the two ids to `candle-gen-sana` as their
registrant, which the generalised row cannot do because it reads its ids back out of the registry.

## Real-weight run (CUDA)

2× RTX PRO 6000 Blackwell (97,887 MiB each), CUDA 12.9, MSVC **14.44.35207** vcvars64 (not 14.51),
`CUDA_COMPUTE_CAP=120`, snapshots from `E:\huggingface`. Each route rendered **twice on one warmed
generator at the same seed** — once with an inert sink, once with a live one — and the two outputs
compared byte for byte.

```sh
cargo test --locked --release -j 1 -p candle-gen-sana --features cuda \
  --test preview_real_weights -- --ignored --nocapture
...
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 42.80s
```

| lane | route | driver | steps | size | frames | r first → last | mean \|Δ\| to final, first → last (ratio) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| txt2img | `sana_1600m` | `run_flow_sampler` | 12 | 1024² | 12 | +0.477 → **+0.958** | 49.76 → 19.81 (0.398) |
| txt2img `heun` | `sana_1600m` | `run_flow_sampler` | 8 | 512² | 8 (**15 evals**) | +0.578 → **+0.938** | 48.20 → 21.54 (0.447) |
| txt2img | `sana_sprint_1600m` | `run_scm_sampler` | 4 | 1024² | 4 | +0.254 → **+0.949** | 61.64 → 14.42 (0.234) |

**Base and Sprint were rendered separately, from separate snapshots, into separate artifacts.** A
shared strip is exactly what would have hidden the mistake this story exists to avoid.

Every floor carries **that lane's own** measured number, with uniform stated headroom — 0.03 under a
measured correlation, **0.06 under a measured rise** (a rise differences two correlations, so it
carries the 0.03 allowance of each), 0.06 over a measured distance ratio, rounded to two decimals:

| lane | `min_r_last` | derivation | `min_rise` | derivation | `max_distance_ratio` | derivation |
| --- | --- | --- | --- | --- | --- | --- |
| `sana_1600m` | 0.928 | 0.958 − 0.03 | 0.421 | (0.958 − 0.477) − 0.06 = 0.481 − 0.06 | 0.46 | 0.398 + 0.06 = 0.458 |
| `sana_1600m` `heun` | 0.908 | 0.938 − 0.03 | 0.300 | (0.938 − 0.578) − 0.06 = 0.360 − 0.06 | 0.51 | 0.447 + 0.06 = 0.507 |
| `sana_sprint_1600m` | 0.919 | 0.949 − 0.03 | 0.635 | (0.949 − 0.254) − 0.06 = 0.695 − 0.06 | 0.30 | 0.234 + 0.06 = 0.294 |

`min_rise` was a single shared `0.30` in the first draft, and sc-16959's reviewer flagged it: never
unsound — all three lanes clear it — but it hid how differently the three sit against it. The measured
rises are **+0.481** (Euler), **+0.360** (`heun`) and **+0.695** (Sprint), so one shared floor handed
them 0.181, **0.06** and 0.395 of margin, none of it stated. Each lane now derives its own, and the
shared number falls out as `heun`'s: `0.360 − 0.06 = 0.300`. `heun` is the shallowest by construction —
its *first* frame already sits at +0.578, ahead of Euler's +0.477, so it has less distance left to
travel even though it finishes lower — and that is now visible in the bound rather than being an
accident of a shared constant. The change is strictly tightening: Euler 0.30 → 0.421, Sprint
0.30 → 0.635, `heun` unchanged.

`max_r_first = 0.75` genuinely is shared and stays shared — both routes are txt2img-only, so every
strip starts at its prior. Loose deliberately: a tight `r_first` bound would read a fit's own warm
intercept as if it were resemblance.

Both series are strictly monotone on every lane, every frame differs from its predecessor by more than
the movement floor, and every frame arrives at DC-AE latent resolution `H/32 × W/32` — which is itself
the runtime proof that no unpack or squeeze was needed. Strips and finals are in
`docs/migration/evidence/sc-16959/`.

The Sprint shape is what a four-step consistency schedule should look like beside a twelve-step flow
one: it starts *further* from the render (+0.254 against +0.477, because each SCM step moves much
further) and finishes closer in absolute distance (14.42 against 19.81).

**One frame per outer step on a multi-eval solver**, proven non-vacuous first: `heun` reported **15**
`Progress::Step` events for 8 outer steps — the driver calls `on_progress` once per *evaluation* — and
the row asserts `evaluations > steps` through `render_and_assert`'s `non_vacuity` callback, which runs
ahead of every other assertion in that function, before pinning the numbering to exactly `1..=8`.

### Harness, both ways

| invocation | result |
| --- | --- |
| `cargo test -p candle-gen-sana --features cuda --test preview_real_weights` (**no** `--ignored`) | **1 passed, 0 failed, 5 ignored** — `the_two_routes_sigma_conventions_are_what_the_projectors_assume` |
| `… -- --ignored` | **5 passed, 0 failed, 0 ignored, 1 filtered out** |

The non-`--ignored` row exists deliberately: sc-16954 shipped a red row that hid because the only
non-ignored row in its file was excluded by `-- --ignored`. Every `#[ignore]`d row **fails** rather
than skips on a missing input — a row that early-returns on an unset variable still reports SUCCESS,
and in a run log a skipped gate is indistinguishable from one that ran.

## CPU coverage — `tests/preview_wiring.rs`

Nine rows driving both drivers through the **shipped** `denoise_cfg` / `denoise_sprint` with the
committed tiny goldens `transformer_parity.rs` numerically validates, in the lane that runs on every
PR: one frame per Euler step; one frame per outer step under `heun` (non-vacuity first); the 1-, 2-,
3- and 4-step Sprint schedules; the single-step schedule on its own (`current == 1`, `total == 1`, no
division by zero, no stall, finite latent); the `σ_data`-scaled latent asserted against the caller's
own seed; no fused unconditional half; inert-sink byte-identity on **both** drivers; and a failing
projector costing neither render anything.

## Mutation proofs

| mutation | caught by |
| --- | --- |
| `run_scm_sampler`'s `Some(preview)` → `None` | `both_sampler_calls_pass_the_hook_at_the_drivers_preview_position`, plus catalog `a_wired_crate_leaves_no_undeclared_dark_sampler_site`, `every_wired_crate_pins_its_exact_route_inventory`, `sana_base_and_sprint_are_two_independent_rows` (4 rows) |
| Sprint lane built with `base_hook` instead of `sprint_hook` | `both_render_lanes_build_their_hook_from_the_requests_sink` |
| `SPRINT_RGB_FACTORS := BASE_RGB_FACTORS` (bias untouched) | `the_base_and_sprint_fits_share_no_row`, `the_committed_constants_are_the_sana_ones` |
| full collapse — factors **and** bias | the two above plus `projecting_one_latent_through_both_fits_gives_different_frames` |
| `sprint_hook` drops the `1/σ_data` correction | `the_sprint_hook_applies_the_sigma_data_correction_it_carries` |

The factors-only and full-collapse rows are listed separately on purpose: the projection row compares
two rendered frames, so it only fires once *both* halves of the fit have been copied — the row-level
`the_base_and_sprint_fits_share_no_row` is what catches a partial copy-paste, which is the shape a real
accident takes.

## Source-breaking changes for consumers

`denoise_cfg`, `denoise_sprint`, both `SanaPipeline`/`SanaSprintPipeline::generate_with` and both
`generate_with_conditioning` gained a trailing `preview: &candle_gen::preview::PreviewHook<'_>`
parameter. **No SceneWorks call site is affected** — `git grep candle_gen_sana` across SceneWorks
`crates/` and `apps/` is empty; the worker reaches SANA only through `gen_core`'s registry by model id,
and `GenerationRequest.preview` already exists. Inside this repo the affected call sites are
`tests/sprint_scm.rs` (3) and `tests/nvfp4_sana_dit_gpu.rs` (1), both updated to pass an inert hook.
`SanaGenerateRequest` is **unchanged** — no new field on a name-driven request type.
