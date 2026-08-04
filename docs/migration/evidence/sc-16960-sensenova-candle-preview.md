# sc-16960 — candle SenseNova-U1 per-step previews, and the epic's one measured fit (epic 16948)

Wires `PreviewSink` into `candle-gen-sensenova` and ships a **newly measured** RGB fit for its latent
space. This is the epic's **Tier 2** story: every other family maps onto a latent space epic 16624
already fitted on MLX, and SenseNova does not.

Base branch `origin/main` @ `67f81b25`.

**Outcome: GO.** Holdout overall R² **0.99998292** against the epic's **0.88** holdout bar. Both
registered ids are wired and advertising; the advertised-`supports_preview` set moves from 27 to **29**.

## The epic's premise was wrong: SenseNova-U1 has **no VAE**

The story was scoped as *"SenseNova-U1 has its own VAE and MLX never fitted it"*. The second half is
true. The first is not, and the difference decides everything below.

`candle-gen-sensenova`'s own crate docs have said so since it was written — *"there is no separate VAE
or text encoder"* — and three independent checks agree, all of them run rather than asserted
(`tests/fit_preview_rgb.rs::the_snapshot_ships_no_autoencoder`):

| check | result |
| --- | --- |
| component directories beside the checkpoint | no `vae/`, no `autoencoder/`, no `first_stage_model/` |
| `config.json` sections | no `vae` / `vae_config` / `autoencoder` / `autoencoder_config` key |
| shard tensor headers | 2,292 tensors under exactly three top-level subtrees — `language_model`, `fm_modules`, `vision_model`; **zero** keys matching `vae` / `autoencoder` / `first_stage` |

**SenseNova-U1 denoises in pixel space.** The flow-matching head predicts `3·(patch·merge)² = 3,072`
values per backbone token; `fm::unpatchify` folds them straight back into `[1, 3, H, W]`; the running
state of the denoise loop *is* the image in the model's `[-1, 1]` space; and the "decode" is the
affine map `t2i::tensor_to_image` applies — `x·0.5 + 0.5`, clamped, ×255, ties to even.

Two consequences carried through the whole story:

* **A reuse was never available.** The seven epic-16624 fits are over 4-, 16- and 32-channel VAE
  latents. This is a **3**-channel pixel space belonging to a checkpoint with no autoencoder, so the
  channel count alone settles it — no hash comparison is needed, and none is offered.
* **The fit is near-exact, and that is a measurement rather than an escape.** An OLS from an affine
  map's input to its output recovers the map. The only thing stopping it being exact is the clamp.
  That is stated up front so the R²s below are read for what they are.

## Registered ids, every denoise lane, and the bespoke loop

`git grep` across `crates/media/candle-gen/candle-gen-sensenova/src`:

```
run_flow_sampler( / run_curated_sampler( / run_scm_sampler( / run_av_curated_sampler(   0 hits
hand-written `for i in 0..steps` denoise loops                                          2
```

**Zero shared-sampler call sites.** That is deliberate, not incidental: `descriptor_for` advertises an
**empty** curated sampler and scheduler menu (`samplers: Vec::new()`), because the unified AR
backbone's `predict_v` mutates a per-step `KvCache` whose length feeds the RoPE/position build, so a
multi-eval curated solver would append to the cache twice per step and desync the AR positions. Native
shifted Euler is the only valid integrator. `Denoise::Bespoke` in the catalog's wiring table is
therefore a statement about the architecture, and the wiring shows up as a **direct emission call**.

| registered id | descriptor | denoise lane | wired |
| --- | --- | --- | --- |
| `sensenova_u1_8b` | `descriptor()` — 50 NFE, CFG 4.0 | `T2iModel::denoise` (`t2i.rs`) | **yes** |
| `sensenova_u1_8b_fast` | `descriptor_fast()` — 8 NFE, CFG 1.0, distill LoRA merged at load | `T2iModel::denoise` (`t2i.rs`) — the same loop | **yes** |

Both ids reach one lane through `SenseNovaGenerator::generate_impl` → `T2iModel::generate` →
`T2iModel::denoise`. They differ only in the id, the generation defaults, and whether the loader merges
the 8-step distill LoRA.

**The crate's other denoise loop stays dark on purpose.** `T2iModel::it2i_denoise` is the off-registry
understanding surface (VQA / Document-Studio interleave), reached only through `interleave_gen`, driven
by the worker directly off the concrete `T2iModel` rather than through the registry, advertised by no
descriptor (`conditioning: []`, `supports_true_cfg: false`), and known-corrupted on the edit path. The
story scopes it out explicitly. `preview::tests::the_bespoke_denoise_loop_emits_exactly_once_per_step`
asserts the shipped source holds **no** `preview` reference at or after `fn it2i_denoise(`, which is
what keeps the catalog's `direct: 1` honest rather than `2`.

**CFG never reaches a frame.** The unconditional pass is a *second forward against a second KV cache*
inside the step body (`predict_v(&cond, rm_u, cache_u, …)`), blended into one `v_pred` by `cfg_blend`
before `euler_step` advances the state. The running `image` is never a fused `[2, …]` batch — the
cond/uncond split lives entirely in the two caches. The real-weight base lane below is rendered at
**true CFG 4.0** precisely so the guided lane is the one measured.

## The latent shape at the emission point — verified, not assumed

| stage | tensor |
| --- | --- |
| `gaussian((1, 3, H, W), seed)` × `noise_scale` | `[1, 3, H, W]` |
| what `emit_step` is handed each step | `[1, 3, H, W]` — the running model-space image |
| what `crate::preview` pools it to | `[1, 3, H/cell, W/cell]`, `cell = patch_size · merge_size = 32` |
| what the frame comes back as | `H/cell × W/cell` RGB8 |
| what `tensor_to_image` hands the caller | `[1, 3, H, W]` → `W × H` RGB8 |

No unpack, no squeeze, no frame axis. The token grid is SenseNova's own latent granularity — one
backbone token *is* one `32 × 32` pixel patch, and the FM head predicts exactly that patch — so a
1024² render previews at 32×32 for the same reason SANA's `f32` DC-AE does. The pool is the same box
average the fit's target is built with, so the projector operates in precisely the space the
coefficients were measured in. `the_pool_is_the_cell_box_average_the_fit_target_uses` proves the pool
is a box average as an identity between two shipped-projector calls (a subsample or a max-pool breaks
it), and the real-weight rows read the frame size back off the emitted frames.

## σ convention, and why rail-clipping is **not** the discriminating statistic here

`step_schedule` builds an **ascending** `t` boundary grid `linspace(0, 1, steps+1)` through
`apply_time_schedule` (`σ = 1 − t`; `σ ← shift·σ / (1 + (shift−1)·σ)`; return `1 − σ`), with the product
inference shift **3.0**. `t = 0` is the prior, `t = 1` is the image, and the state advances by
`euler_step(v, z, t, t_next) = z + (t_next − t)·v`. There is no descending σ array to index into, which
is why frames are keyed on the **step index** (`PreviewCounter::with_steps`) — the same counter shape
the σ-less SCM driver uses. No input scaling is applied to the state the loop advances, so
`PreviewHook::new` is the correct constructor rather than `with_sigma`.

sc-16959 found that rail-clipping is not always the statistic that catches a scaling error. SenseNova
is a sharper case of the same thing, in the opposite direction. The prior is
`N(0,1) · noise_scale` at full pixel resolution (`noise_scale = 2.0` at 512², from the `"resolution"`
mode formula `√(seq / 64) · 1.0`), so the *unpooled* prior clips heavily — but the token-cell pool
averages **1024** independent samples per preview pixel, dividing the prior's standard deviation by 32.
So SenseNova's **first frame is near-flat grey and clips essentially nothing**.

`tests/preview_real_weights.rs` therefore reports the rail-clipped fraction (to record that it does not
discriminate) and bounds **contrast about the fit's own intercept** instead — the mean absolute
distance from the flat grey a fully-zero state projects to. See the measured table below.

## The fit — measured, with an honest holdout

`tests/fit_preview_rgb.rs`, `sensenova_u1_8b` `q8/` tier on CUDA, 512² × 8 flow-match Euler steps at
guidance 4.0, token cell 32 ⇒ 16×16 = **256 pooled samples per render**.

**The split is by whole render.** Four fit renders determine the coefficients; two holdout renders with
**different prompts and different seeds** measure them. The holdout renders are produced *after* the
normal equations are solved, so nothing about them can have reached a coefficient even by accident. It
is not a random subsample of one render's pixels — that would leak each render's own palette into both
halves and is exactly what the epic-16624 standard exists to prevent.

| | prompts | seeds | pooled samples |
| --- | --- | --- | --- |
| **fit** | night market / alpine mountains / studio portrait / tropical-fruit illustration | 1696001–1696004 | 4 × 256 = 1024 |
| **holdout** | library at dusk / macro wildflowers | 1696091, 1696092 | 2 × 256 = 512 |

Predictor: the `cell`-pooled final model-space state. Target: the `cell`-pooled **clamped** decode
(`clamp(0.5·x + 0.5)` then pool — that order, because it is the order the finished image is produced
in). Solved as 4×3 normal equations in `f64` with partial pivoting.

### Fit R² and holdout R², separately

| split | R² (R, G, B) | overall R² |
| --- | --- | --- |
| **fit** — 4 renders, 1024 samples | `0.99999517`, `0.99999016`, `0.99998441` | **`0.99998989`** |
| **holdout** — 2 disjoint renders, 512 samples | `0.99999123`, `0.99998945`, `0.99997544` | **`0.99998292`** |

Per-image spatial R² (each image's own target-centered SST against its raw prediction SSE, so a fit
that only separated prompt-level palettes could not pass) ranges **0.99997004 – 0.99998979** across all
six renders, and every pooled target carries real spatial variance (min channel variance `0.00688`,
against a floor of `0.002`).

### Go / no-go

**The bar is holdout overall R² ≥ 0.88** — the bar epic 16624 used to *reject* LTX (fit .984 / holdout
.619), Mage (.938 / .806) and Mochi (.847 / .807). Measured holdout overall is **0.99998292**, so this
is a **GO**, and `HOLDOUT_OVERALL_R2_FLOOR` in the producer is that bar verbatim rather than a number
tuned to the result. The fit floor is a separate constant for a separate split; the two are never
compared to each other.

The honest reading of the margin is the one stated at the top: there is no VAE, so the decode is an
affine map and OLS recovers it. This story could equally have found a *rejection* — the same producer,
the same bar — and the reason it did not is a property of the architecture, not of the corpus.

### The committed constants, and what the residual is

The producer's `f64` solution, printed to nine decimals:

```
const RGB_FACTORS: [[f32; 3]; 3] = [
    [0.499482566, 0.001114516, 0.001276602],
    [-0.000450976, 0.497581102, -0.000060032],
    [0.000828717, 0.000887129, 0.497608008],
];
const RGB_BIAS: [f32; 3] = [0.500254442, 0.500271806, 0.499321467];
```

Committed as the shortest decimals that round-trip through `f32` — `clippy::excessive_precision`
rejects digits an `f32` cannot hold, and each literal below compiles to the identical `f32`:

```rust
const RGB_FACTORS: [[f32; 3]; 3] = [
    [0.499_482_57, 0.001_114_516, 0.001_276_602],
    [-0.000_450_976, 0.497_581_1, -0.000_060_032],
    [0.000_828_717, 0.000_887_129, 0.497_608],
];
const RGB_BIAS: [f32; 3] = [0.500_254_44, 0.500_271_8, 0.499_321_47];
```

The analytic decode transform is `x·0.5 + 0.5`, exactly diagonal. The solved gains come out a touch
**under** 0.5 with small cross-channel terms, because the target is the *clamped* decode and clipping
compresses. Largest distance from the analytic transform: **0.0024188976** (the green gain).

That number is not left as prose:

* `ANALYTIC_TOLERANCE = 3e-3` is it, rounded up to one significant figure, and a **compile-time**
  `const` block checks all twelve coefficients against the analytic transform. A fit measured against
  the wrong target — an unclamped decode, a differently pooled one, a transposed row — is a build
  error, not a review question. `the_measured_fit_lands_on_the_analytic_decode_transform` restates it
  at runtime for the message.
* `the_committed_fit_is_within_two_rgb8_levels_of_the_models_own_decode` derives the worst-case visual
  cost from the coefficients: over the model's own `[-1, 1]` range the extreme is
  `Σ_j |M[j][c] − 0.5·δ_jc| + |b_c − 0.5|`, which is **0.523 / 1.197 / 1.124** RGB8 levels on R / G / B.
* `the_projector_is_the_models_own_decode_at_token_resolution` (the **only non-`#[ignore]`d row** in
  `tests/preview_real_weights.rs`, so a plain `cargo test` runs it) compares the shipped projector
  against the engine's own `avg_pool2d` + `tensor_to_image` on a synthetic in-range state. Both sides
  are shipped code; there is no second implementation of the maths.

The producer additionally cross-checks the **shipped** projector against an independent `f64`
evaluation of the *solved* coefficients on all six renders: max RGB8 delta **0** on every one — exact
agreement — which is what binds the constants block above to the code that consumes it. (The bound in
the assertion is ≤ 1, to leave room for the `f32`/`f64` split; the measurement is 0.)

### Provenance

There is no autoencoder to name, so the fit's provenance is the **checkpoint**:

| field | value |
| --- | --- |
| repo | `SceneWorks/sensenova-u1-8b-mlx` |
| revision | `b6206ea2e888198418b92f3bed31f5506c6183f9` |
| tier / file | `q8/model.safetensors` |
| bytes | 19,911,123,700 |
| SHA-256 | `8da38dde4c39722259a98cfc47643c88e48cea205595625fdbd9fec097f9dc4f` |
| tensors | 2,292, under `language_model` / `fm_modules` / `vision_model` only |
| channel count | **3** (pixel space) |

## Wiring, and the whole hook path guarded

One hook, built once, over the request's own sink:

```
SenseNovaGenerator::generate → generate_impl
  → preview::t2i_hook(&req.preview, comps.model.cell())
  → T2iModel::generate(…, preview: &PreviewHook<'_>)
  → T2iModel::denoise(…, preview: &PreviewHook<'_>)
  → preview.emit_step(&preview_counter, i, &image)
```

`cell` is bound at that single site from `T2iModel::cell()` — the same accessor `denoise` derives its
token grid from — so the projector cannot come to disagree with the loop about how large a token is.
`the_hook_projects_at_the_cell_it_was_built_with` pins that a hook built with a different cell produces
a different frame size, so a wrong binding is visible without a render.

**Every hop takes the hook as a non-`Option` `&PreviewHook<'_>`**, so widening the seam is a type
error. That is necessary and — as sc-16958 and sc-16959 both demonstrated on merged code — not
sufficient. Both reviewers took a family's lanes dark with **zero** type errors and a green CPU suite:

* a hop that accepts and then **ignores** its forwarded hook and builds a fresh
  `PreviewHook::new(&inert, …)` — a constructor-call tally spelled `_hook(` cannot see it, because
  `PreviewHook::new(` does not contain that substring;
* a `generate` that rebinds `let req = &GenerationRequest { preview: PreviewSink::default(),
  ..req.clone() };` ahead of the hook build — the literal a scan counts is still there, exactly once,
  over an emptied sink.

This story's own review found **two more**, both green against the first draft of the guard, and both
are the reason the table below is per-file rather than per-crate:

* the `_hook(` tally was applied only to `lib.rs`, which left `t2i.rs` free to call
  `crate::preview::t2i_hook(&dark, cell)` a **second** time inside `denoise` and shadow the forwarded
  hook. No `PreviewHook::new`, no `GenerationRequest {`, both parameter lines intact, the `.emit_step`
  literal intact, and the catalog inventory still reading `hooked: 0, direct: 1`;
* the `GenerationRequest {` count blocks only the **struct-literal** spelling. `GenerationRequest`
  derives `Clone` (`gen-core/src/generator.rs:268`) and its `preview` field is `pub` (`:471`), so
  `let mut owned = req.clone(); owned.preview = PreviewSink::default(); let req = &owned;` empties the
  sink with no literal at all.

`preview::tests::the_registered_lane_builds_its_hook_from_the_requests_sink` counts on the **shipped**
half of each file (everything ahead of its first `#[cfg(test)]` item) with whole-line comments dropped,
so the needles count code and the module docs stay free to name the spellings they forbid:

| pin | shipped `lib.rs` | shipped `t2i.rs` |
| --- | --- | --- |
| `preview::t2i_hook(&req.preview, comps.model.cell())` | exactly **1** | — |
| `_hook(` | exactly **1** | **0** |
| `.preview` | exactly **1** (that same site) | **0** |
| `PreviewSink` | **0** | **0** |
| `PreviewHook::` | **0** | **0** |
| `GenerationRequest {` | **0** | **0** |
| `let req` / `\|req` / `req =>` | **0** | — |
| `preview: &PreviewHook<'_>,` (whole trimmed line) | — | exactly **2** (`generate`, `denoise`) |

Four of those rows are new in this revision. `_hook(` and `.preview` are now pinned **per file** rather
than on `lib.rs` alone, which is what closes the shadowed-second-hook edit. `.preview` is what closes
the clone-and-assign edit, and it is stronger than pinning `req.clone()`: plain `clone()` is legitimate
on this path (once in shipped `lib.rs`, three times in shipped `t2i.rs`) so it cannot be pinned to zero,
whereas *reading the request's sink at all* is something this lane does exactly once. `PreviewHook::`
rather than `PreviewHook::new` because `with_sigma` and `over_schedule` are constructors too.

The last row policies the **shadow** instead of what produced it, and it is the one that generalises:
`req` reaches the hook site only as `generate_impl`'s own parameter, so `let req = &…;` fails whatever
built the right-hand side — a type alias (`type Req = GenerationRequest; let owned = Req { preview:
Default::default(), ..req.clone() };`, which evades all three construction needles) or a helper from
another crate alike.

Parameters are counted by **whole trimmed line**, never by substring: a substring tally is satisfied by
the same declaration renamed `_preview:`, which is precisely what a hop looks like once it stops using
its hook.

`preview::tests::no_sibling_module_in_this_crate_touches_the_requests_sink` closes the remaining
in-crate spelling — a helper in a sibling module returning a request with an emptied `preview`, so that
none of the needles above appears in either scanned file. Every module `lib.rs` declares other than
`preview` (which owns the seam), `lib` and `t2i` is pinned to **0** `PreviewSink`, **0** `.preview`,
**0** `PreviewHook` and **0** `_hook(`, over the whole file, and the module list is checked against
`lib.rs`'s own `mod` declarations so a new source cannot arrive as an unscanned blind spot.

The shipped-half split itself is anchored at line start rather than matched as `\n#[cfg(test)]\n`. That
is **defensive, not load-bearing**: a newline-wrapped needle matches nothing in a CRLF checkout, and the
split holds even if one occurred — but one does not occur here. `.gitattributes:22` pins
`* text=auto eol=lf` for the whole tree, set specifically to override `core.autocrlf=true` on Windows;
`git check-attr text eol` reports `text: auto, eol: lf` for all three sources; and `lib.rs`, `t2i.rs`
and `preview.rs` hold **0** CR bytes on disk. What actually rules out a vacuous pass is the helper's
`!shipped.is_empty()` assertion, which fires whatever the line endings turn out to be.

### What is *not* enforced

Stated precisely, because an earlier draft of this section claimed closure it did not have. Move the
emptying helper into a **different crate** and pass its result in **argument position** rather than
rebinding — `self.generate_impl(&candle_gen::preview::without_listener(req), on_progress)` inside
`Generator::generate` — and every pin above is satisfied: no needle of the table appears, no module of
this crate is touched, and `req` is never shadowed so the rebind row has nothing to fire on. That edit
was built and run while writing this revision: **the registered lane went fully dark with all 57 lib
tests green.** Closing it would need a scan of every crate that can produce a `GenerationRequest`, which
is not a bound a test in this crate can hold.

So: the guard pins **where the hook is built, what it is built from, and that nothing on this crate's
own path rebuilds or replaces either.** It does not make the sink unreachable. The only thing that
observes frames actually arriving is `tests/preview_real_weights.rs`, which renders through the
registered `Generator` seam with a live sink — on CUDA, on real weights, and it is the only place this
crate proves the seam end to end.

## Catalog guard — all three steps in this PR

1. **`supports_preview` flipped** on `descriptor_for`, which both ids share, with a comment naming the
   bespoke loop it is true because of. `descriptor_advertises_only_wired_t2i_surface` and
   `both_registered_sensenova_routes_advertise_preview_support` assert it on each id.
2. **`PREVIEW_ROUTE_IDS` gains `sensenova_u1_8b` and `sensenova_u1_8b_fast`**, individually — the
   per-id form sc-16959 generalised the guard into, so one id cannot go dark behind its sibling.
3. **Route inventory row added**: `candle-gen-sensenova` → `t2i.rs` `hooked: 0, direct: 1, dark: []`.
   The second all-`direct` row in the table, after Ideogram's, and the shape `DIRECT_EMISSION_CALLS`
   was hardened for in sc-16951's review: a bespoke loop emitting through `PreviewHook::emit_step` was
   invisible to the original scanner, which listed only the free functions.

**Advertised-`supports_preview` set is now exactly 29** — the 27 before this story plus SenseNova's two
ids. `preview_capability_matches_every_wired_shipped_route_bidirectionally` asserts the registry's
advertising set equals `PREVIEW_ROUTE_IDS` exactly in both directions, so nothing else moved.
Fourteen wired crates register those 29 ids between them, which is exactly `PREVIEW_ROUTE_IDS.len()`.

### `supports_preview` does not collapse when this epic completes

SenseNova is the permanent reason, and it is now demonstrated rather than predicted: `mlx-gen-sensenova`
contains no `PreviewSink` reference at all (`git grep PreviewSink -- crates/media/mlx-gen/mlx-gen-sensenova`
is empty), the family is candle-only, and nothing in this story changes that. At least one route stays
engine-split for good, so a single shipped boolean would be wrong in every ordering.

## Real-weight run (CUDA)

2× RTX PRO 6000 Blackwell (97,887 MiB each), CUDA 12.9, MSVC **14.44.35207** vcvars64 (not 14.51),
`CUDA_COMPUTE_CAP=120`, snapshots from `E:\huggingface`. Each id rendered **twice on one warmed
generator at the same seed** — once with an inert sink, once with a live one — and the two outputs
compared byte for byte.

```sh
cargo test --locked --release -p candle-gen-sensenova --features cuda \
  --test preview_real_weights -- --ignored --nocapture
...
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 242.68s
```

| lane | route | steps | size | frames | r first → last | mean \|Δ\| to final, first → last (ratio) | contrast about the intercept, first → last | rail-clipped, first |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| txt2img, **true CFG 4.0** | `sensenova_u1_8b` | 8 | 512² | 8 | +0.234 → **+0.999** | 104.04 → 28.63 (0.275) | 6.33 → **76.05** | **0.0000** |
| txt2img, CFG 1.0 | `sensenova_u1_8b_fast` | 8 | 512² | 8 | +0.260 → **+1.000** | 62.11 → 18.45 (0.297) | 6.33 → **44.72** | **0.0000** |

**The two ids were rendered separately, from separate snapshots, into separate artifacts.** Both
advertise the flag, so a strip from one is not evidence for the other.

Both series are strictly monotone — distance to the finished image falls at every step, correlation
rises at every step, and contrast about the intercept grows at every step, each asserted separately
because no pair of endpoint bounds implies monotonicity. Every frame differs from its predecessor by
more than the movement floor (smallest step-to-step mean |Δ| observed: 3.14). Every frame arrives at
the token grid `512/32 = 16×16`, read back off the emitted frames — which is itself the runtime proof
that the running state really is `[1, 3, H, W]` and the pool is the model's own `cell`. Strips, per-frame
PNGs and finals are in `docs/migration/evidence/sc-16960/`.

Every floor carries **that lane's own** measured number, with uniform stated headroom — 0.03 under a
measured correlation, 0.06 under a measured rise, 0.06 over a measured distance ratio (rounded up to
two decimals), ±20% on a measured contrast:

| lane | `min_r_last` | `min_rise` | `max_distance_ratio` | `max_first_contrast` | `min_last_contrast` |
| --- | --- | --- | --- | --- | --- |
| `sensenova_u1_8b` | 0.969 (0.999 − 0.03) | 0.705 ((0.999 − 0.234) − 0.06) | 0.34 (0.275 + 0.06) | 7.6 (6.33 × 1.2) | 60.8 (76.05 × 0.8) |
| `sensenova_u1_8b_fast` | 0.970 (1.000 − 0.03) | 0.680 ((1.000 − 0.260) − 0.06) | 0.36 (0.297 + 0.06) | 7.6 (6.33 × 1.2) | 35.7 (44.72 × 0.8) |

`max_first_contrast` is the same number on both lanes and that is not a shared constant standing in
for two measurements: **frame 1 is emitted before any model forward**, so on both lanes it is the same
seeded prior at the same seed and resolution and measures 6.33 on both.

`max_r_first = 0.60` is the one bound not derived from a lane's own measurement, and it is shared and
deliberately loose — measured +0.234 and +0.260, both from a near-flat 16×16 pooled prior whose
correlation with the render is essentially a coin flip. Tightening onto either would read noise as a
contract. This is the same carve-out sc-16959 recorded for SANA's `max_r_first`. What the strip has to
prove is the rise, and that bound *is* per lane.

`max_first_rail_clipped = 0.02` against a measurement of **0.0000** is deliberately a bound on zero,
and it is non-vacuous: the *unpooled* prior at 512² is `N(0,1) · 2.0`, whose decode is `clamp(z + 0.5)`
and therefore rails on `2·Φ(−0.5) ≈ 62%` of its pixels. The ceiling fails the moment the token-cell
pool stops happening. It is also this story's record that rail-clipping — the statistic the epic's
earlier stories reached for — does **not** discriminate here.

The first-frame contrast corroborates the whole analysis numerically. The pooled prior's standard
deviation is `2.0 / 32 = 0.0625` in model space, i.e. `0.03125` after the decode's ×0.5, i.e. `7.97`
RGB8 levels; a normal's mean absolute deviation is `σ·√(2/π) = 0.798·σ`, so the predicted first-frame
contrast about the intercept is **6.36**. Measured: **6.33**, on both lanes. The frame is exactly the
seeded prior averaged over 1024 samples per preview pixel, and nothing else.

### One frame per outer step, against the loop's own counter

`assert_the_strip_converges` does not count frames. The preview sink and the progress callback write
into **one** event log, and the log is compared element-wise against
`[Frame(1,8), Step(1,8), Frame(2,8), Step(2,8), …, Frame(8,8), Step(8,8)]`. That does three things at
once:

* the frame numbering is checked against the very `Progress::Step` counter the bespoke loop advances,
  rather than against a number this test believes in;
* exactly one frame per outer step falls out of the alternation, with no possibility of a duplicate or
  a dropped position hiding in a total;
* the **emit-before-step** ordering is pinned — the contract the shared drivers and Ideogram's bespoke
  loop both hold to, and the reason frame 1 is the prior and frame 8 is one advancement short of the
  render rather than the render itself.

### Inert-sink byte identity

Each lane renders **twice on one warmed generator at the same seed**, once with an inert sink and once
with a live one, and the two outputs are compared byte for byte. `tests/fit_preview_rgb.rs` repeats the
check one level lower, on the final `f32` state rather than the RGB8 output, and additionally asserts
the live render emitted exactly 8 frames over 8 steps.

## Harness, both ways

Both `#[ignore]`d files, run **both ways** on the CUDA lane:

| invocation | result |
| --- | --- |
| `--test fit_preview_rgb` (**no** `--ignored`) | **0 passed, 0 failed, 2 ignored** |
| `--test fit_preview_rgb -- --ignored` | **2 passed, 0 failed, 0 ignored** (179.36s) |
| `--test preview_real_weights` (**no** `--ignored`) | **1 passed, 0 failed, 2 ignored** — `the_projector_is_the_models_own_decode_at_token_resolution` |
| `--test preview_real_weights -- --ignored` | **2 passed, 0 failed, 0 ignored, 1 filtered out** (220.40s) |

The non-`--ignored` row on `preview_real_weights` exists deliberately: sc-16954 shipped a red row that
hid because the only non-ignored row in its file was excluded by `-- --ignored`. `fit_preview_rgb` has
no such row and reports 0/0/2 — it is a producer, and everything in it needs weights.

Every `#[ignore]`d row **fails** rather than skips on a missing input: `required_path` panics with a
message saying the row cannot be skipped into a pass. A row that early-returns on an unset variable
still reports SUCCESS, and in a run log a skipped gate is indistinguishable from one that ran.

### Package suites

| lane | result |
| --- | --- |
| CPU `--lib` — `candle-gen`, `candle-gen-sensenova`, `candle-gen-catalog`, `candle-gen-sana`, `candle-gen-ideogram` | **319 + 56 + 27 + 56 + 36 = 494 passed, 0 failed** |
| CUDA `--lib --tests` — `candle-gen-sensenova`, `candle-gen-catalog` | sensenova green; `candle-gen-catalog --lib` **26 passed, 1 failed** — see below |
| `cargo fmt` (per package), `clippy --all-targets -- -D warnings`, `RUSTDOCFLAGS=-D warnings cargo doc --no-deps`, `crates/media/candle-gen/scripts/check-lock-poison.sh`, `python3 scripts/check_docs.py` | all clean |

**The one CUDA failure is pre-existing and untouched.**
`every_registered_memory_strategy_rejects_cross_route_decode_geometry` fails under `--features cuda`
on `z_image_turbo_control` and `z_image_control` — *"optimized registration lacks a weights-free
behavior seam"* — which is **sc-17087**, recorded by sc-16957 (§9 of
`docs/migration/evidence/sc-16957-z-image-candle-preview.md`) with the same two ids. It is a
memory-strategy decode-geometry assertion, names no SenseNova id, and touches no preview surface. Not
fixed here.

`scripts/check-workspace.py` also fails locally on "expected only the root Cargo.lock" because this
checkout has sibling worktrees; that is a local-environment artifact, not a graph change.

## Mutation proofs

Every row below was **executed**: the mutation applied to the working tree, the affected packages run,
the failing test names recorded, and the mutation reverted.

| mutation | caught by |
| --- | --- |
| **M1** — delete `preview.emit_step(&preview_counter, i, &image);` from the bespoke loop | `preview::tests::the_bespoke_denoise_loop_emits_exactly_once_per_step` (sensenova, 1 failure), plus catalog `every_wired_crate_pins_its_exact_route_inventory` and `source_level_wiring_and_advertised_capability_agree_for_every_provider_crate` (2 failures) — **3 rows across 2 crates** |
| **M2** — hook built over an emptied sink (`let emptied = PreviewSink::default();`) instead of `&req.preview` | `preview::tests::the_registered_lane_builds_its_hook_from_the_requests_sink`. The catalog suite stays **fully green** (27/27) — which is exactly why this row exists: the route inventory can only see that `t2i.rs` emits, never what sink reached it |
| **M3** — a raw `PreviewHook::new(&req.preview, …)` added beside the correct hook (the sc-16959 `render_seed` spelling) | `preview::tests::the_registered_lane_builds_its_hook_from_the_requests_sink`, via the **zero** `PreviewHook::new` count — the `_hook(` tally cannot see it, because `PreviewHook::new(` does not contain that substring |
| **M4** — corrupt one fit coefficient (green gain `0.4976 → 0.4`) | **build error**: `error[E0080]: evaluation panicked: a committed RGB_FACTORS coefficient is further than ANALYTIC_TOLERANCE from the analytic decode transform…` — the compile-time `const` block, so a wrong fit never reaches a test run |
| **M5** — drop the token-cell pool from `project_running_image` | four rows: `the_pool_is_the_cell_box_average_the_fit_target_uses`, `the_shipped_resolutions_preview_at_the_token_grid`, `the_hook_projects_at_the_cell_it_was_built_with`, `a_low_precision_state_pools_and_projects` |

M2 is the load-bearing one and it is listed with its *negative* result on purpose: it is the mutation
sc-16959's reviewer used to darken a family with a green catalog suite, and here the catalog suite is
green under it too. Nothing cross-crate can close that gap — only the crate-local source pins can, and
above them only a real-weight render through the registered `Generator` seam with a live sink, which
is what `tests/preview_real_weights.rs` does.

An accidental finding worth recording: the *naive* spelling of M2 —
`preview::t2i_hook(&gen_core::PreviewSink::default(), …)` — does not even compile
(`error[E0716]: temporary value dropped while borrowed`), because the hook borrows its sink for the
lifetime of the render. The mutation had to be given an explicit `let` binding before it was a real
threat. That is a small property of taking the sink by reference, and it is not relied on.

## Source-breaking changes for consumers

`T2iModel::generate` and the private `T2iModel::denoise` gained a trailing
`preview: &candle_gen::preview::PreviewHook<'_>` parameter. **No SceneWorks call site is affected**:
the worker reaches SenseNova's T2I surface only through `gen_core`'s registry by model id
(`crates/sceneworks-worker/src/image_jobs/sensenova.rs` calls `generator.generate(&request, …)`, and
`GenerationRequest.preview` already exists), and the off-registry understanding carve-outs
(`sensenova_jobs.rs`) drive `T2iModel::vqa` / `T2iModel::interleave_gen`, neither of which changed.
Inside this repo `T2iModel::generate` has exactly one call site, `SenseNovaGenerator::generate_impl`.
`T2iOptions` is unchanged — no new field on a public options struct.
