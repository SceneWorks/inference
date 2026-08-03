# sc-16956 — candle FLUX.1 family per-step latent previews (epic 16948)

The 16-channel latent space. This story wires `gen_core::PreviewSink` into **three** candle crates —
`candle-gen-flux`, `candle-gen-chroma`, `candle-gen-pulid` — across **five** denoise lanes, and answers
a question sc-16955 left open for a fourth crate it deliberately withheld (`candle-gen-boogu`,
sc-17218).

No fit is introduced. The epic-16624 16-channel constants committed at
`crates/media/mlx-gen/mlx-gen-flux/src/preview.rs:32` are transcribed verbatim into
`crates/media/candle-gen/candle-gen-flux/src/preview.rs`; `candle-gen-chroma` and `candle-gen-pulid`
re-export them. `mlx-gen-flux/tests/fit_preview_rgb.rs` remains the only producer.

## 1. Adjudication — flux / chroma / pulid, explicitly

| family | crate | decision | why |
| --- | --- | --- | --- |
| **FLUX.1** | `candle-gen-flux` | **wired** — 3 lanes, 2 ids | `flux1_schnell` + `flux1_dev`; the fit donor's own latent space |
| **Chroma** | `candle-gen-chroma` | **wired** — 1 lane, 3 ids | `chroma1_hd` / `chroma1_base` / `chroma1_flash`; ships a **byte-identical** VAE file |
| **PuLID-FLUX** | `candle-gen-pulid` | **wired** — 1 lane, **0 ids** | composes this crate's own FLUX.1-dev backbone; registers no descriptor |
| **Boogu** | `candle-gen-boogu` | **not wired here** (sc-17218) | but its VAE **is** this one — measured below, so sc-17218 reuses this fit |

### The story's PuLID premise did not hold — and it mattered

The story says *"Candle ships PuLID as a bespoke provider id, unlike MLX's `pulid_flux`; use the candle
id in the descriptor and allowlist."* There **is no candle PuLID id**. `candle-gen-pulid/src/lib.rs:16`
states it outright ("a plain struct driven **directly** by the worker … NOT a gen-core-registered
`Generator`"), `candle_gen_catalog::BESPOKE_UTILITY_CRATES` lists `pulid`, and the shipped
`temporal_and_super_resolution_routes_stay_outside_preview_advertising` asserts by exact id that no
registered descriptor is ever named `pulid` or `pulid_flux`. Inventing one to satisfy the story text
would have broken the guard the epic asked for.

That is not a cosmetic correction. `candle-gen-catalog`'s `PROVIDER_CRATES` table is keyed on a
**registration function**, so before this story no table in that module could see `candle-gen-pulid` at
all — and PuLID owns a `run_flow_sampler` site of its own. Wiring it under the pre-existing guard would
have produced exactly the failure the guard exists to prevent: a shipped, user-reachable lane whose
emission fact nothing checks. §6 describes the `BESPOKE_PROVIDER_CRATES` table added to close it.

### Chroma's VAE claim was inherited from MLX and is re-verified here

`mlx-gen-flux/src/preview.rs` records that "Chroma's three pinned Q4 tiers reuse the same latent basis
and the exact FLUX.1 loader". On candle, Chroma does **not** call `candle_gen_flux::load_vae` — it ships
its own `vae.rs`, a 16-channel adaptation of the FLUX.2 decoder. So the MLX statement is about a
different code path and could not be inherited. It is re-measured directly against the bytes each
snapshot carries (§4), and the answer is stronger than the MLX note: the files are byte-identical.

## 2. Lane enumeration — every user-reachable denoise lane, per crate

`git grep 'run_flow_sampler(\|run_curated_sampler(\|run_scm_sampler('` over each crate's `src`, plus a
sweep for bespoke `for`-loop denoise bodies (`for … in …sigmas`, `for … in 0..steps`, `windows(2)`) and
for direct `emit_preview*` / `.emit*` calls. **Five sampler sites across the three crates, five lanes,
zero bespoke loops, zero dark sites.**

### `candle-gen-flux` — `Denoise::Shared`, 3 sites, 3 lanes, 0 dark

| file | lane | driver | default for |
| --- | --- | --- | --- |
| `pipeline.rs:649` | registered txt2img (`Pipeline::denoise`) | `run_flow_sampler` | **`flux1_schnell` and `flux1_dev` — the default and only registered lane for both** |
| `control_provider.rs:417` | Fun-ControlNet-Union strict pose/canny/depth | `run_flow_sampler` | the worker's `flux1_dev_control` candle lane |
| `ip_provider.rs:324` | XLabs IP-Adapter reference stream | `run_flow_sampler` | the worker's FLUX IP-Adapter lane |

`control.rs`, `ip_adapter.rs`, `ip_dit.rs`, `packed_dit.rs`, `packed_te.rs`, `flux1_load.rs`,
`ref_backbone.rs`, `quant.rs`, `vae/` and `lib.rs` hold weights, geometry or registration and drive no
sampler — pinned as a negative row (`the_geometry_and_weight_modules_drive_no_sampler`).

**The "one site, N callers" check.** `control_provider.rs` is the shape sc-16955's Lens finding warns
about: one sampler site with **two** public entry points, `generate` and `generate_with_injector` (the
compose-ready seam that stacks PuLID / IP identity on top of control). Here they cannot diverge, because
`generate` is a one-line delegation that forwards its whole `req` — sink included — and adds `None` for
the injector. `both_control_entry_points_reach_the_one_hooked_site` pins that against the crate's own
source, so a refactor giving `generate` a body of its own has to come back to it. `ip_provider` and
`pipeline` each have a single caller path.

### `candle-gen-chroma` — `Denoise::Shared`, 1 site, 1 lane, 0 dark

| file | lane | driver | default for |
| --- | --- | --- | --- |
| `pipeline.rs:312` | registered txt2img (`Pipeline::denoise`) | `run_flow_sampler` | **`chroma1_hd`, `chroma1_base` and `chroma1_flash` — one lane, three descriptors** |

Three ids, one render body: the variants differ in the DiT weights and in the σ schedule
(`Pipeline::sigmas` — HD/Flash static-shift `linspace`, Base beta-spaced), not in the denoise. Chroma
has no trainer, no second denoise and no name-driven provider.

### `candle-gen-pulid` — `Denoise::Shared`, 1 site, 1 lane, 0 dark

| file | lane | driver | default for |
| --- | --- | --- | --- |
| `pulid_flux.rs:391` | `PulidFlux::generate` identity T2I | `run_flow_sampler` | the worker's `candle_pulid_flux` lane (no descriptor) |

The epic's scoping said PuLID uses `run_flow_sampler`; **verified, and it is its own call**, not a
delegation to `candle-gen-flux`'s. The candle PuLID runs its own flow loop (unlike the MLX PuLID, which
delegates to the FLUX backbone), which is why the curated `sampler`/`scheduler` knobs are threaded
through `PulidFluxRequest`.

**Nothing was missed.** The negative sweep found no bespoke denoise body in any of the three crates: the
only `for … windows(2)` matches are schedule-monotonicity assertions inside `#[cfg(test)]` modules, and
no crate has a trainer. The catalog's own scanner re-derives all five sites from the shipped module tree
independently of this table.

## 3. The latent shape at the emission point — verified, not assumed

The epic flagged FLUX.1 explicitly: *"`unpack_latents` was found only in `candle-gen-flux2` and
`candle-gen-qwen-image` … do not port an unpack step the candle path does not have, and do not skip one
it does."*

**Candle FLUX.1 does pack its latents — it just spells the recovery differently.** The DiT denoises a
`[1, ⌈H/16⌉·⌈W/16⌉, 64]` token sequence; the native `[1, 16, 2⌈H/16⌉, 2⌈W/16⌉]` VAE latent exists only
after `candle_transformers::models::flux::sampling::unpack`, which `pipeline::decode_latents` calls
before every decode. So an unpack **is** needed, and the reason `git grep unpack_latents` missed it is
that the function is named `unpack` and lives in `candle-transformers`, not in the provider crate.

| stage | shape | projectable? |
| --- | --- | --- |
| the sampler's running latent | `[1, ⌈H/16⌉·⌈W/16⌉, 64]` packed tokens | no — rank 3 |
| after `flux::sampling::unpack` | `[1, 16, 2⌈H/16⌉, 2⌈W/16⌉]` | **yes — the fitted space** |

Unlike FLUX.2 there is **no second transform**: this VAE is a plain diffusers `AutoencoderKL` with no
BatchNorm-stats space, so the unpack alone recovers the fitted latent. That asymmetry is why this module
could not be written by porting `candle-gen-flux2/src/preview.rs`.

`project_packed_tokens` calls the **same** `unpack` the decode calls — one implementation of the fold,
not two agreeing ones — and every route builds its hook from the same `(width, height)` pair it hands
its decode tail. `the_packed_recovery_is_the_one_the_decode_uses` pins the equality *and* pins that a
naive reshape of the same values produces a different picture, so a refactor that dropped the permute
would be red rather than merely wrong-looking.

## 4. VAE reuse, grounded in tensor bytes

`crates/media/candle-gen/candle-gen-flux/tests/preview_real_weights.rs` re-derives every number below
per snapshot. All four containers declare the same `vae/config.json`: `_class_name: AutoencoderKL`,
`latent_channels: 16`, `scaling_factor: 0.3611`, `shift_factor: 0.1159`,
`block_out_channels: [128, 256, 512, 512]`, `layers_per_block: 2`.

| container | SHA-256 | bytes | tensors | loaded by |
| --- | --- | --- | --- | --- |
| **fit donor** `SceneWorks/flux1-dev-mlx` @ `323fd12d…` `q4/vae/model.safetensors` | `e510ed25…4823` | 164,654,042 | 260 (244 learned bf16 + 16 packed arrays) | candle FLUX.1 / PuLID packed tier |
| diffusers bf16 `black-forest-labs/FLUX.1-{dev,schnell}` `vae/diffusion_pytorch_model.safetensors` | `f5b59a26…40a3` | 167,666,902 | 244 | Chroma (all three variants) |
| BFL f32 `black-forest-labs/FLUX.1-{dev,schnell}` `ae.safetensors` | `afc8e282…9e38` | 335,304,388 | 244 | candle FLUX.1 / PuLID **dense** tier |
| q8 tier `SceneWorks/flux1-dev-mlx` `q8/vae/model.safetensors` | `7cbe4841…f24d` | 165,702,660 | 260 | candle FLUX.1 / PuLID q8 tier |

Measured (`the_flux1_family_ships_one_learned_vae_in_three_containers`):

- **fit donor vs diffusers bf16** — **236 of 244** learned tensors byte-identical. The eight that differ
  are exactly the mid-block spatial-attention Q/K/V/out projections (encoder + decoder) the MLX packer
  quantized to `U32` code blocks with separate `scales`/`biases`; they are excluded **by name**, not by
  "whatever failed", so the row cannot degrade into comparing only the tensors that happen to agree.
- **BFL f32 vs diffusers bf16** — all **244** tensors (**83,819,683** values) map from the BFL naming
  onto the diffusers naming and round, round-to-nearest-even, exactly onto its bits. The map is not
  cosmetic: `mid.block_{1,2}` → `mid_block.resnets.{0,1}`, `mid.attn_1.{q,k,v,proj_out,norm}` →
  `mid_block.attentions.0.{to_q,to_k,to_v,to_out.0,group_norm}` (1×1 conv → Linear),
  `nin_shortcut` → `conv_shortcut`, `norm_out` → `conv_norm_out`, and the decoder's `up.{i}` blocks are
  **reversed** (`3 − i`).
- `SceneWorks/flux1-schnell-mlx` @ `bba3ae01…` q4 ships all **260** tensors identical to the fit donor's
  (the 54-byte file-size difference is header metadata), so the schnell tier is the same weights again.

Per family:

- **FLUX.1** — the fit donor *is* one of the tiers this crate loads. Identity, not analogy.
- **Chroma** (`the_chroma_vaes_are_byte_identical_to_the_flux1_one`) — HD `9d99afe1…`, Base
  `e7330dda…` and Flash `6a9cb617…` each ship `vae/diffusion_pytorch_model.safetensors` with SHA-256
  `f5b59a26…40a3`: **byte-identical** to `black-forest-labs/FLUX.1-dev`'s and to `FLUX.1-schnell`'s.
  Four repos, one file. A hash equality is the strongest available instrument here, so it is used
  rather than the tensor machinery the q4/BFL containers need.
- **PuLID** — has **no VAE of its own**. `PulidFlux` holds a `candle_gen_flux::FluxRefBackbone`, whose
  tier-detecting load is the registered `flux1_dev` route's; the worker points it at
  `SceneWorks/flux1-dev-mlx`'s `q4`/`q8`/`bf16` subdirs
  (`sceneworks-worker/src/image_jobs/pulid_candle.rs:38`). Its reuse is the same code and the same
  files, which is why its harness carries no provenance row and says so.

### Boogu (sc-17218): the FLUX.1 16-channel fit is the one to reuse

`Boogu/Boogu-Image-0.1-Turbo` @ `7c475e94…`'s `vae/diffusion_pytorch_model.safetensors` (SHA-256
`8c717328…4c94`, 244 f32 tensors, no `bn.*`) has the **same key set** as the FLUX.1 diffusers container,
the same shapes, and all **244** tensors (**83,819,683** values) round, round-to-nearest-even, exactly
onto its bf16 bits — with an identical `vae/config.json`. It is not a second 16-channel space; it is
this one. `the_boogu_vae_is_the_flux1_one` pins it as a shipped row.

sc-16955 measured the same file against the FLUX.2 **32**-channel fit and correctly refused it. Both
findings stand: "16 channels" alone never made two latent spaces the same, which is why sc-17218 needed
an answer rather than an inference. The answer is `candle_gen_flux::preview` — not `candle-gen-z-image`'s
fit (sc-16957), which is a different checkpoint.

## 5. σ convention and the rail-clip measurement

`run_flow_sampler` integrates a `gen_core::sampling::FlowModelSampling` whose `input_scale` is
**identically 1.0** at every σ (`gen-core/src/sampling/model_sampling.rs:220`), so the running latent
already *is* the tensor the fit was measured against and the σ-less `PreviewHook::new` constructor is
the correct one. This is the property sc-16954 found to be **false** for the discrete ε cohort
(SDXL/Kolors denoise in k-diffusion VE σ-space; uncorrected, 89.4 % of the first frame clipped to the
rails).

Measured anyway, because it is cheap and decisive: projecting a seeded unit-normal packed latent at
σ_max = 1.0 — what this family's first emission actually sees — gives a rail-clipped fraction of
**0.0000**. `the_flow_cohort_needs_no_sigma_correction` is the **only non-`#[ignore]`d row** in
`candle-gen-flux/tests/preview_real_weights.rs`, deliberately: sc-16954 shipped a red row that hid
because `-- --ignored` excluded the only non-ignored row in its file.

## 6. The catalog guard — all three steps, plus the table that had to exist

1. **`supports_preview` flipped** on the five wired ids only: `flux1_schnell`, `flux1_dev`
   (`candle-gen-flux/src/lib.rs`), `chroma1_hd`, `chroma1_base`, `chroma1_flash`
   (`candle-gen-chroma/src/config.rs`). PuLID has no descriptor to flip.
2. **`PREVIEW_ROUTE_IDS` extended** with those five ids, individually. The advertised-true set is now
   **20**: the 15 from sc-16950/16952/16953/16954/16955 plus these five. Nothing else moved.
3. **Route inventories added**, exact per-file `hooked`/`direct`/`dark` counts:
   `candle-gen-flux` = `control_provider.rs` 1/0/—, `ip_provider.rs` 1/0/—, `pipeline.rs` 1/0/—;
   `candle-gen-chroma` = `pipeline.rs` 1/0/—; `candle-gen-pulid` = `pulid_flux.rs` 1/0/—. All three
   crates declare `Denoise::Shared`. **No `DarkSite` anywhere in this story** — none of the three has a
   trainer or a second denoise.

**`BESPOKE_PROVIDER_CRATES` (new).** `PROVIDER_CRATES` is keyed on a registration function, so a crate
that registers nothing cannot appear in it. That cost nothing until now, because none of the six
`BESPOKE_UTILITY_CRATES` members owned a denoise loop — and `candle-gen-pulid` does. The new table lists
all six (`depth`, `face`, `instantid`, `pid`, `pulid`, `sam3`) with a `Denoise` shape and a route
inventory, and `every_bespoke_utility_crate_is_covered_by_the_bespoke_table` derives its membership from
`BESPOKE_UTILITY_CRATES` itself, so a seventh utility crate — or a sampler site appearing in one of the
five that have none — cannot join uninventoried. The three source-derived assertions (denoise shape, no
undeclared dark site, exact inventory) now run over both tables; only the id-keyed ones stay
`PROVIDER_CRATES`-only, because a bespoke crate has no ids to advertise.

Bringing those five crates into the scan exposed one real hole in the scanner: `candle-gen-instantid`
and `candle-gen-pulid` both open `src/validate.rs` with a **file-level inner** `#![cfg(test)]`, which
`code_only` did not recognise (it looked only for the outer `#[cfg(…)]` that precedes an item) and which
its own belt-and-braces sweep then hard-failed on. `inner_cfg_attribute` now handles it, recognised only
before the first block — the one position where an inner attribute means "the whole file", since one
inside an inline `mod` applies to that module alone and treating it as file-scope would under-scan.

### Mutation proof

- Blanking `pulid_flux.rs`'s hook to `None` → `every_wired_crate_pins_its_exact_route_inventory` and
  `every_bespoke_utility_crate_is_covered_by_the_bespoke_table` both go **red**. (Before this story that
  edit was invisible to every test in the module.)
- Reverting `chroma1_*`'s `supports_preview` to `false` →
  `preview_capability_matches_every_wired_shipped_route_bidirectionally` and
  `source_level_wiring_and_advertised_capability_agree_for_every_provider_crate` both go **red**.

## 7. Real-weight CUDA validation

2× RTX PRO 6000, MSVC 14.44 vcvars, `CUDA_COMPUTE_CAP=120`, `HF_HUB_CACHE=E:\huggingface`. Strips and
final renders under `docs/migration/evidence/sc-16956/`. Every render is run **twice on one warmed
model at the same seed** — once inert, once live — and the two outputs are asserted byte-identical, so
"an active sink does not perturb the render" is measured rather than argued.

Snapshots: `SceneWorks/flux1-dev-mlx` @ `323fd12d…` **q4** for FLUX.1 and for PuLID's backbone;
`SceneWorks/chroma1-hd-mlx` @ `9d99afe1…` **q4** for Chroma; `guozinan/PuLID` @ `492b1451…`,
`SceneWorks/pulid-flux-mlx` @ `78ef91f9…` and `SceneWorks/instantid-mlx` @ `bca0cacf…` for the identity
stack. PuLID's reference face is itself a FLUX.1-dev render (`pulid-reference.png`), so the harness
needs no third-party portrait.

| lane | size × steps | frames | movement (open → close, ×, peak) | mean \|Δ\| to final | coarse r (first → last) |
| --- | --- | --- | --- | --- | --- |
| `flux1_dev` euler (**default**) | 1024² × 12 | 12, 1..=12 | 2.813 → 12.288 (4.4×), peak 12.288 | 68.36 → 12.71 (**−81.4 %**) | +0.116 → **+0.970** |
| `flux1_dev` heun | 768² × 8 | 8, 1..=8 | 5.301 → 14.424 (2.7×), peak 14.424 | 65.61 → 12.99 (**−80.2 %**) | +0.027 → **+0.961** |
| `chroma1_hd` euler (**default**) | 1024² × 12 | 12, 1..=12 | 3.122 → 8.797 (2.8×), peak 9.496 | 60.71 → 13.42 (**−77.9 %**) | +0.193 → **+0.800** |
| `chroma1_hd` heun | 768² × 8 | 8, 1..=8 | 4.273 → 15.731 (3.7×), peak 15.731 | 65.32 → 12.64 (**−80.6 %**) | +0.058 → **+0.957** |
| PuLID euler (**default**) | 1024² × 12 | 12, 1..=12 | 4.805 → 14.507 (3.0×), peak 15.989 | 90.43 → 12.34 (**−86.4 %**) | +0.113 → **+0.962** |
| PuLID heun | 768² × 8 | 8, 1..=8 | 8.413 → 19.307 (2.3×), peak 19.307 | 84.37 → 12.34 (**−85.4 %**) | +0.018 → **+0.952** |

Every strip: distance to the finished image falls at **every** step, resemblance rises at **every**
step, and the total rise clears the +0.30 bar with room (+0.607 on the weakest lane, +0.934 on the
strongest). The fit's in-sample R² `0.98224` puts the correlation ceiling at √ ≈ `0.991`, so the FLUX.1
and PuLID lanes land at 96–98 % of it.

**Chroma's 1024² euler lane is the outlier at +0.800**, and the reason is its schedule rather than the
wiring: Chroma HD walks `linspace(1, 1/N)` under a static shift of 3, which leaves a large unpreviewed
terminal step (the hook emits *before* each solver step, so the final advancement is never shown). Its
own 8-step 768 lane, which does not pay that penalty, reaches +0.957 on the same fit. sc-16955 measured
the same effect on FLUX.2 (+0.556 against a 0.874 ceiling); at 80.7 % of ceiling Chroma is well ahead of
that already-accepted precedent. Per-lane floors are set just under each measurement.

**Exactly one frame per outer step on a multi-eval solver**, proven non-vacuously: `heun` produced **15
evaluations for 8 outer steps** on all three families (the driver calls `on_progress` once per
*evaluation*, so the event count IS the evaluation count), and each still emitted exactly 8 frames
numbered 1..=8. The inequality is asserted **before** the frame count, so a solver that silently fell
back to Euler could not make the row pass vacuously.

**One assertion was relaxed on measured evidence.** "Movement rises monotonically into the terminal
step" is a property of the *model*, not the wiring: on the same nominal 1024² × 12-step flow schedule
FLUX.1-dev rises into it (9.729 → 12.288) while Chroma HD (9.496 → 8.797) and PuLID (15.989 → 14.507)
dip — by the last previewed step the latent is nearly converged, so the projection's mean |Δ| saturates
even as the σ interval grows. The terminal pair is now excluded from the monotonicity check and
replaced by a floor on the terminal step as a share of the strip's **peak** movement (≥ 0.5; measured
0.91 on the weakest lane), which still fails a hook that froze or one projecting a stale latent.

### Both ways, separately

The harnesses were run **without** `--ignored` and **with** it, and both results are reported, because
sc-16954 shipped a red row that hid behind `-- --ignored` excluding the only non-ignored row in its
file:

- **without `--ignored`** — `cargo test -p candle-gen-flux -p candle-gen-chroma -p candle-gen-pulid
  --test preview_real_weights`: `1 passed; 0 failed; 5 ignored` (flux — the σ-convention row),
  `0 passed; 0 failed; 3 ignored` (chroma), `0 passed; 0 failed; 2 ignored` (pulid).
- **with `--ignored`** (CUDA, release): flux `5 passed; 0 failed` (both strips + all three provenance
  rows), chroma `3 passed; 0 failed`, pulid `2 passed; 0 failed`.

### An incidental engine fix this story had to make

The first PuLID real-weight run failed **before any preview was involved** — in the *inert* baseline
render, inside `compute_id_embedding`:

```
matmul is only supported for contiguous tensors
lstride: Layout { shape: [1, 16, 577, 64], stride: [64, 64, 1024, 1] }
```

`candle-gen-pulid/src/eva_clip/attention.rs`'s `rope_patch_tokens` joins the untouched CLS token to the
RoPE'd patch tokens with `Tensor::cat` on axis 2. Candle's `cat` takes a **transposing** slow path on a
non-zero axis when its inputs are not all contiguous — and `rope.apply` returns a strided view — so `q`
reached the attention matmul with a transposed layout. Candle's CPU gemm accepts that; its **CUDA**
matmul does not. The whole EVA02-CLIP tower therefore failed at the first attention on the one platform
this crate exists for, while every CPU unit test passed — the standing candle lesson that a green CPU
suite proves a path compiles and runs, not that the CUDA kernels accept its layouts. One `.contiguous()`, numerically a no-op, fixes q and k at once. Without it candle PuLID is
entirely non-functional at real weights on Windows/CUDA, so this was a blocker for the story's own
acceptance criterion rather than adjacent cleanup.

## 8. What this breaks in SceneWorks — and what is left

Adding a `preview: PreviewSink` field to a name-driven request type is a **source-breaking change** for
SceneWorks, which builds these structs as exhaustive literals with no `..Default::default()`. That is
expected and deliberate — a `#[non_exhaustive]` struct would let a lane silently *not* forward its sink,
which is exactly the failure this epic exists to fix. Three call sites break, and the closeout pin bump
(sc-16962) must fix them:

| SceneWorks call site | request type | sink already available? |
| --- | --- | --- |
| `crates/sceneworks-worker/src/image_jobs/flux1_control_candle.rs:319` | `Flux1ControlRequest` | **no** — `CandleStrictControl::generate_one` has no `preview` parameter |
| `crates/sceneworks-worker/src/image_jobs/flux_ipadapter.rs:366` | `IpAdapterFluxRequest` | yes — `drive_gen_items_scored` already passes a `_preview` the closure ignores |
| `crates/sceneworks-worker/src/image_jobs/pulid_candle.rs:448` | `PulidFluxRequest` | yes — same `_preview` argument |

The first needs `CandleStrictControl::generate_one` extended with a `preview: &PreviewSink` parameter.
sc-16955 left the same gap for `flux2_control_candle.rs`, so the trait change serves both lanes and
should be made once.
