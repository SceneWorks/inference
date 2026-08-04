//! SANA's per-step latent preview seam (epic 16948, sc-16959; the MLX original is epic 16624 /
//! `mlx-gen-sana/src/preview.rs`).
//!
//! Schedule numbering, multi-eval dedup and the swallow-on-failure contract live in
//! [`candle_gen::preview`], shared by every candle family (sc-16949). This module owns three things:
//! the **two reused** SANA 32-channel fits, the layout check that says the tensor being projected is
//! one of this family's latents, and the `1/σ_data` correction the SCM driver's running latent needs.
//!
//! ## Two routes, two drivers, two fits
//!
//! `candle-gen-sana` is the only family in this epic that drives **two** shared samplers, and the only
//! one that carries **two** committed fits for one crate. Neither may borrow the other's:
//!
//! | route id | driver | denoise | fit | default |
//! | --- | --- | --- | --- | --- |
//! | `sana_1600m` | [`candle_gen::run_flow_sampler`] | true-CFG flow-match, static shift 3.0 | `BASE_RGB_*` | 20 steps, guidance 4.5, native Euler |
//! | `sana_sprint_1600m` | [`candle_gen::run_scm_sampler`] | CFG-free SCM / TrigFlow consistency | `SPRINT_RGB_*` | 2 steps, embedded guidance 4.5 |
//!
//! Shipping one fit for both routes is the specific mistake this story exists to avoid, so it is
//! pinned three ways rather than left to review: `the_base_and_sprint_fits_share_no_row` asserts every
//! one of the 33 constant rows differs, `projecting_one_latent_through_both_fits_gives_different_frames`
//! asserts the two projectors disagree on the same tensor, and the two hook constructors below take
//! their factor tables from different constants with no shared default.
//!
//! ## Every denoise lane, enumerated
//!
//! A `git grep` of `run_flow_sampler` / `run_curated_sampler` / `run_scm_sampler` across
//! `candle-gen-sana/src` returns **exactly two** hits, both in `pipeline.rs` — `denoise_cfg`'s flow
//! call and `denoise_sprint`'s SCM call — and there is no hand-written `for`-loop denoise anywhere in
//! the crate. Two registered descriptors, two user-reachable lanes, one call site each:
//!
//! * **`sana_1600m` — txt2img only.** Its `Capabilities::conditioning` is empty and `load` refuses
//!   quantization, LoRA/LoKr and control / IP-adapter overlays outright, so there is no img2img fork
//!   and no name-driven provider. The one lane is reachable under the whole curated epic-7114 sampler
//!   menu, which is why the multi-eval dedup matters here: `heun` and `dpmpp_sde` evaluate twice per
//!   outer step through that same single call.
//! * **`sana_sprint_1600m` — txt2img only.** It advertises only the `"default"` sampler / scheduler
//!   sentinel, because the SCM consistency loop is not a curated [`candle_gen::gen_core::sampling::Solver`]
//!   at all. `load_sprint` refuses the same overlays.
//!
//! The crate ships **no trainer**, so unlike Krea / Lens / SDXL / Z-Image it has no deliberately dark
//! site: both sampler calls are hooked and `candle-gen-catalog`'s route inventory pins
//! `hooked: 2, direct: 0, dark: []` on `pipeline.rs`.
//!
//! **CFG never reaches the preview.** Base SANA runs true CFG as *two separate trunk forwards inside
//! `denoise_cfg`'s predict closure*, blended into one velocity (`uncond + scale·(cond − uncond)`); no
//! fused `[2, …]` batch is ever the running latent, so there is no unconditional half to project.
//! Sprint has no unconditional branch at all — its guidance is an embedded scalar handed to the trunk's
//! guidance embedder — so the Sprint path has no fused half to go looking for.
//!
//! ## The latent shape at the emission point — verified per route, not assumed
//!
//! Both routes denoise SANA's native spatial DC-AE latent. There is no packed token space in this
//! family (the Linear-DiT trunk patchifies internally) and no frame axis, so neither route needs an
//! unpack and neither needs a gated squeeze:
//!
//! | stage | base (`run_flow_sampler`) | Sprint (`run_scm_sampler`) |
//! | --- | --- | --- |
//! | `pipeline::create_noise` | `[1, 32, H/32, W/32]` | `[1, 32, H/32, W/32]` |
//! | what the driver hands the hook | `[1, 32, H/32, W/32]` | `[1, 32, H/32, W/32]`, **pre-scaled by `σ_data`** |
//! | what `pipeline::decode_to_image` hands the DC-AE | `[1, 32, H/32, W/32]` | `[1, 32, H/32, W/32]` |
//!
//! The channel count is taken from [`crate::pipeline::LATENT_CHANNELS`] — the very constant
//! `create_noise` builds its noise with and `DcAeDecoder` decodes — so this module cannot come to
//! disagree with the denoise about how wide the space is. The spatial edge is image/32
//! ([`crate::pipeline::SPATIAL_SCALE`]), so a 1024² render previews at 32×32; that is the deep-
//! compression autoencoder's own resolution, not a downsample this module chose.
//!
//! ## The σ convention differs between the two routes — and only one needs a correction
//!
//! * **Base needs none.** [`candle_gen::run_flow_sampler`] integrates a
//!   [`candle_gen::gen_core::sampling::FlowModelSampling`], whose `input_scale` is exactly `1.0` at
//!   every σ, so the running latent already *is* the tensor the base fit was measured against and the
//!   σ-less [`candle_gen::preview::PreviewHook::new`] constructor is correct. `the_flow_route_needs_no_input_scaling`
//!   reads that off the `ModelSampling` itself rather than asserting it in prose.
//! * **Sprint needs `1/σ_data`.** [`candle_gen::run_scm_sampler`] multiplies the seed latent by
//!   `σ_data` on entry and divides it back out on exit, and it hands the hook the **scaled** running
//!   latent. Sprint's projector therefore multiplies by `1/σ_data` before projecting — the same
//!   `inverse_sigma_data` argument `mlx-gen-sana::preview::emit_sprint_preview` carries. The
//!   correction is derived from [`candle_gen::SCM_SIGMA_DATA`], the constant every
//!   [`candle_gen::ScmScheduler`] constructor sets and the driver divides by, and
//!   `the_scm_scheduler_always_carries_the_sigma_data_this_correction_inverts` binds the two.
//!
//! Note the SCM driver has **no σ array**, so `PreviewCounter::new(sigmas)` does not apply to Sprint at
//! all; the driver keys its frames on the step index through
//! [`candle_gen::preview::PreviewCounter::with_steps`]. Sprint is also a 1–4 step schedule, and the
//! single-step case is asserted explicitly below.
//!
//! ## The fits are reused, not refitted — and two are needed because the two DECODERS differ
//!
//! `BASE_RGB_FACTORS` / `BASE_RGB_BIAS` and `SPRINT_RGB_FACTORS` / `SPRINT_RGB_BIAS` (private, hence no
//! links) are the epic-16624 constants transcribed verbatim from `mlx-gen-sana/src/preview.rs`. They
//! are least-squares numbers over a VAE latent with no backend in them; candle reuses them and
//! deliberately ships **no producer** of its own — `mlx-gen-sana/tests/fit_preview_rgb.rs` remains the
//! only way either is re-derived.
//!
//! The reuse is grounded in tensor bytes, per route, and it is the strongest grounding in this epic:
//! **the candle route loads the identical file the MLX fit was measured on, in both directions.**
//!
//! * Base — `Efficient-Large-Model/Sana_1600M_1024px_diffusers` @
//!   `d1b54936033cd7d45410ecadd692c5c502a19a38`, `vae/diffusion_pytorch_model.safetensors`,
//!   1,249,044,836 bytes, SHA-256 `15a4b09e56d95b768a0ec9da50b702e21d920333fc9b3480d66bb5c7fad9d87f`
//!   — **byte-identical** to the `SceneWorks/Sana_1600M_1024px_mlx` file whose SHA-256 the base fit's
//!   provenance record names.
//! * Sprint — `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` @
//!   `b3c9ce6f29ad4161a00fa58a62e476b9c75ca934`, same filename, the same 1,249,044,836 bytes, SHA-256
//!   `dfd991d1b54ffabf22745c5885589d8f2a7bc59930d95d92bd741c4fc64454bb` — **byte-identical** to the
//!   `SceneWorks/Sana_Sprint_1.6B_1024px_mlx` file the Sprint fit's provenance record names.
//!
//! ### Why two fits — measured, not assumed
//!
//! The two containers are the same size with the same 375 keys, shapes and dtype, so nothing short of
//! a tensor walk could say how they relate. `tests/preview_real_weights.rs` walks them, and the answer
//! is one none of this epic's earlier stories would have predicted: the two DC-AEs **partially
//! overlap**. **320 of 375 tensors are byte-identical — including the entire encoder, all 179 of its
//! tensors — and the 55 that differ are every one of them in the `decoder.` subtree**
//! (`up_blocks.0/1/2`, `norm_out`, `conv_out`: the last three upsampling stages and the output head).
//! `decoder.conv_in` and `decoder.up_blocks.3`, the stage closest to the latent, are identical too.
//!
//! DC-AE 1.1 (Sprint) is therefore a **decoder-tail fine-tune** of DC-AE 1.0 (base). The encoder is
//! what defines the latent space and it is unchanged — but an RGB preview fit maps a latent to that
//! autoencoder's **decoded pixels**, and the decode is exactly what was retrained. So one fit cannot
//! serve both routes, and the reason is sharper than "two latent spaces": it is **one latent space
//! with two decoders**. That also explains the shape of the two committed tables — structurally alike,
//! numerically apart — which is precisely the resemblance that makes a copy-paste between them
//! plausible and this module's guards worth having.
//!
//! `tests/preview_real_weights.rs` re-derives every claim above per snapshot; the full record is
//! `docs/migration/evidence/sc-16959-sana-candle-preview.md`.
//!
//! A stale or absent fit degrades preview colour only; the denoise path never reads these constants.

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::{Image, PreviewSink};
use candle_gen::preview::PreviewHook;
use candle_gen::{CandleError, Result};

use crate::pipeline::LATENT_CHANNELS;

/// Base SANA's ordinary-least-squares map from its scaled DC-AE latent to latent-resolution RGB (row
/// *i* maps latent channel *i* to `[r, g, b]`), with `BASE_RGB_BIAS` the intercept.
///
/// **Reused verbatim from `mlx-gen-sana/src/preview.rs`, not refitted.** Fit on four diverse
/// real-weight Base renders and measured on two disjoint prompt/seed holdouts, all 256² with eight
/// static-shift-3 flow-Euler steps at true CFG 4.5, against native DC-AE decodes average-pooled by
/// [`crate::pipeline::SPATIAL_SCALE`]. Fit R² `(R,G,B) = (0.94379, 0.94447, 0.95035)`, overall
/// `0.94601`; holdout R² `(0.89018, 0.90411, 0.90728)`, overall `0.89941`.
///
/// Donor VAE: `vae/diffusion_pytorch_model.safetensors`, 1,249,044,836 bytes, SHA-256
/// `15a4b09e56d95b768a0ec9da50b702e21d920333fc9b3480d66bb5c7fad9d87f` — the file
/// `Efficient-Large-Model/Sana_1600M_1024px_diffusers` @ `d1b5493…` publishes and the one
/// `SceneWorks/Sana_1600M_1024px_mlx` published for the fit, byte for byte.
///
/// **Not Sprint's.** Sprint ships a *decoder-tail fine-tune* of this same DC-AE at the same container
/// size — one identical encoder, 55 differing decoder tensors — so these rows describe a **decode**
/// `SPRINT_RGB_FACTORS` does not. See the module docs for the tensor walk.
///
/// Refit — in `mlx-gen-sana`, never here — whenever the base SANA DC-AE lineage changes.
const BASE_RGB_FACTORS: [[f32; 3]; 32] = [
    [-0.005_173_837, -0.015_042_886, -0.007_488_659],
    [0.000_351_302, 0.001_461_522, -0.000_774_016],
    [0.008_825_719, 0.001_199_267, -0.021_697_014],
    [-0.009_551_8, -0.003_265_284, -0.001_802_139],
    [0.006_764_386, 0.008_381_207, -0.011_200_431],
    [0.012_730_06, 0.009_848_793, 0.001_164_681],
    [-0.064_499_7, 0.017_144_65, 0.030_926_41],
    [0.004_002_025, 0.001_431_185, 0.004_225_923],
    [-0.000_988_014, -0.002_621_293, -0.000_011_099],
    [-0.020_792_159, -0.008_566_003, 0.000_020_908],
    [-0.010_766_551, -0.015_161_791, -0.019_115_523],
    [0.009_740_896, 0.010_433_206, 0.000_820_438],
    [0.009_016, 0.000_445_342, 0.007_900_901],
    [0.012_811_583, -0.003_098_324, -0.001_098_156],
    [0.010_798_576, 0.005_888_571, 0.004_122_824],
    [0.005_488_254, -0.007_242_312, 0.012_080_453],
    [0.012_765_118, -0.007_917_695, 0.008_944_155],
    [-0.005_965_203, -0.008_538_616, -0.005_285_878],
    [0.003_386_742, 0.008_137_628, 0.004_372_295],
    [0.002_402_561, 0.004_276_578, 0.001_985_248],
    [-0.005_492_354, 0.009_353_48, -0.031_596_568],
    [0.002_371_349, 0.001_331_12, 0.006_118_907],
    [-0.003_632_62, -0.003_973_242, -0.005_477_37],
    [0.006_471_009, 0.003_666_282, 0.005_540_972],
    [-0.092_541_26, -0.101_964_61, -0.100_527_29],
    [0.000_860_234, -0.001_277_855, 0.025_332_061],
    [-0.006_872_895, 0.017_089_885, -0.009_946_431],
    [-0.004_895_191, 0.006_142_205, -0.003_318_994],
    [-0.003_770_195, -0.002_957_515, 0.000_958_802],
    [0.005_835_034, 0.003_580_885, 0.000_608_369],
    [-0.007_277_093, 0.000_836_16, 0.008_557_965],
    [-0.000_549_004, 0.001_023_616, -0.008_517_158],
];

/// The base fit's intercept — the near-neutral grey a fully-zero base latent projects to. Reused with
/// `BASE_RGB_FACTORS`.
const BASE_RGB_BIAS: [f32; 3] = [0.467_3, 0.437_615_22, 0.414_471_12];

/// SANA-Sprint's ordinary-least-squares map from its scaled DC-AE latent to latent-resolution RGB.
///
/// **Reused verbatim from `mlx-gen-sana/src/preview.rs`, not refitted.** The four-fit / two-holdout
/// producer used **four SCM steps at embedded guidance 4.5** — a different denoise from the base
/// producer's eight CFG-4.5 flow-Euler steps, over a different autoencoder. Fit R²
/// `(R,G,B) = (0.96315, 0.96731, 0.96115)`, overall `0.96439`; holdout R²
/// `(0.94540, 0.90385, 0.93090)`, overall `0.93066`.
///
/// Donor VAE: `vae/diffusion_pytorch_model.safetensors`, 1,249,044,836 bytes, SHA-256
/// `dfd991d1b54ffabf22745c5885589d8f2a7bc59930d95d92bd741c4fc64454bb` — the file
/// `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` @ `b3c9ce6f…` publishes and the one
/// `SceneWorks/Sana_Sprint_1.6B_1024px_mlx` published for the fit, byte for byte. The official configs
/// carry the same consumed architecture and scaling fields but identify DC-AE Sana **1.0** (base)
/// versus **1.1** (Sprint).
///
/// **Not the base's.** The two containers are the same 1,249,044,836 bytes and share 320 of their 375
/// tensors — the whole encoder among them — but all 55 that differ are in the decoder tail, so the
/// same latent decodes to different pixels. That is exactly why the fits are not shared.
///
/// The projector applies `1/σ_data` before these rows, because the SCM driver's running latent is
/// pre-scaled — see this module's docs and `project_sprint_latents`.
///
/// Refit — in `mlx-gen-sana`, never here — whenever the Sprint DC-AE lineage changes.
const SPRINT_RGB_FACTORS: [[f32; 3]; 32] = [
    [-0.011_039_152, -0.009_622_238, -0.000_219_859],
    [-0.001_787_247, -0.001_421_918, -0.005_986_071],
    [0.011_402_427, -0.002_974_342, -0.015_161_106],
    [-0.007_480_828, 0.001_479_696, -0.002_422_799],
    [0.008_063_252, 0.012_678_597, -0.013_892_456],
    [0.011_412_599, 0.002_060_583, 0.002_145_959],
    [-0.070_273_293, 0.016_512_596, 0.037_036_387],
    [-0.004_700_504, -0.002_757_596, 0.008_739_593],
    [0.001_312_036, -0.008_613_445, -0.002_093_493],
    [-0.018_778_253, -0.019_158_247, 0.007_124_294],
    [-0.005_627_819, -0.005_312_433, -0.017_171_389],
    [-0.001_117_639, -0.001_434_881, -0.002_470_027],
    [0.012_784_191, 0.000_079_727, 0.012_982_718],
    [0.011_755_877, -0.002_326_223, 0.000_341_412],
    [0.002_501_337, -0.001_228_545, -0.002_620_598],
    [-0.004_028_827, -0.012_651_675, 0.010_675_205],
    [0.012_539_394, -0.002_551_714, 0.002_239_369],
    [-0.004_789_931, -0.002_140_59, -0.003_570_349],
    [0.002_870_705, 0.017_316_512, 0.015_326_954],
    [-0.000_136_197, 0.005_923_379, 0.003_242_907],
    [0.004_542_396, 0.012_175_465, -0.032_539_067],
    [-0.001_360_403, 0.000_009_718, 0.001_045_577],
    [-0.014_818_82, -0.014_927_074, -0.011_538_296],
    [-0.004_396_741, 0.005_684_66, 0.008_298_95],
    [-0.095_328_63, -0.109_609_84, -0.106_988_52],
    [0.006_340_783, 0.002_176_268, 0.025_876_263],
    [-0.010_475_607, 0.015_086_319, -0.011_959_834],
    [-0.003_533_343, 0.002_384_034, -0.004_472_533],
    [0.002_721_063, 0.008_446_834, 0.005_142_777],
    [0.018_582_678, -0.002_438_67, -0.001_012_867],
    [-0.005_599_778, 0.005_467_537, 0.008_059_673],
    [0.001_651_694, 0.001_563_015, -0.010_827_64],
];

/// The Sprint fit's intercept. Reused with `SPRINT_RGB_FACTORS`, and deliberately **not** the base's —
/// the two greys differ in every channel.
const SPRINT_RGB_BIAS: [f32; 3] = [0.457_922_9, 0.428_063_12, 0.399_165_84];

/// The latent channel count the fits are defined over, derived from the committed base table's own
/// length so nothing in this crate can drift from it by restating a number.
pub const PREVIEW_LATENT_CHANNELS: usize = BASE_RGB_FACTORS.len();

/// Both fits are the THIRTY-TWO-channel ones, and they are defined over the channel count the rest of
/// the crate already denoises and decodes in. Compile-time, because a runtime row over constants
/// proves nothing a `const` assertion does not prove earlier.
const _: () = assert!(
    PREVIEW_LATENT_CHANNELS == 32
        && PREVIEW_LATENT_CHANNELS == LATENT_CHANNELS
        && SPRINT_RGB_FACTORS.len() == PREVIEW_LATENT_CHANNELS
);

/// The scale Sprint's projector applies before its fit, undoing the `σ_data` pre-scale
/// [`candle_gen::run_scm_sampler`] applies to the running latent.
///
/// Derived from [`candle_gen::SCM_SIGMA_DATA`] — the constant every [`candle_gen::ScmScheduler`]
/// constructor sets and the driver divides by — rather than restated as `2.0`, so the correction
/// cannot come to disagree with the loop it inverts. `the_scm_scheduler_always_carries_the_sigma_data_this_correction_inverts`
/// pins that binding through both public constructors.
///
/// This is the candle spelling of `mlx-gen-sana::preview::emit_sprint_preview`'s `inverse_sigma_data`
/// argument.
pub const SPRINT_INVERSE_SIGMA_DATA: f32 = 1.0 / candle_gen::SCM_SIGMA_DATA;

/// Project **base** SANA's native spatial running latent `[1, 32, h, w]` to a latent-resolution RGB8
/// preview.
///
/// There is nothing to recover first: this is already the `[1, C, h, w]` contract
/// [`candle_gen::preview::project_latents`] takes, it is the same tensor
/// [`crate::pipeline::decode_to_image`] hands the DC-AE, and the flow driver applies no input scaling.
/// The only work here is rejecting a latent that is not one of this family's.
///
/// Errors on any other layout. The caller's frame is then lost and swallowed by
/// [`candle_gen::preview::emit_preview`], which is the intended decorative-failure behaviour.
pub fn project_base_latents(latents: &Tensor) -> Result<Image> {
    check_layout(latents)?;
    candle_gen::preview::project_latents(latents, &BASE_RGB_FACTORS, BASE_RGB_BIAS)
}

/// Project **Sprint**'s running latent `[1, 32, h, w]` to a latent-resolution RGB8 preview, after
/// removing the SCM loop's `σ_data` prior-space scale.
///
/// `inverse_sigma_data` is `1/σ_data` — [`SPRINT_INVERSE_SIGMA_DATA`] for every shipped call. It is a
/// parameter rather than a hard-coded constant for exactly one reason: it is what the identity
/// `project_sprint_latents(x·σ_data, 1/σ_data) == project_sprint_latents(x, 1)` is expressed over, and
/// that identity is how the tests prove the correction inverts the driver's pre-scale instead of
/// merely being present.
///
/// Uses the **Sprint** fit. Handing a Sprint latent to [`project_base_latents`] would produce a frame
/// — a wrongly coloured one, over a latent space that autoencoder does not occupy — which is why the
/// two live behind separate entry points with no shared default.
pub fn project_sprint_latents(latents: &Tensor, inverse_sigma_data: f32) -> Result<Image> {
    check_layout(latents)?;
    let unscaled = latents.affine(inverse_sigma_data as f64, 0.0)?;
    candle_gen::preview::project_latents(&unscaled, &SPRINT_RGB_FACTORS, SPRINT_RGB_BIAS)
}

/// Reject anything that is not one batch-1 latent in the fitted thirty-two-channel space.
///
/// The shared projection would reject most of these anyway, but naming the channel count here makes
/// the failure say *SANA* — and catches the one case it cannot see, a rank-4 latent whose channel
/// count merely happens to match another 32-channel family's (FLUX.2 / Lens / Ideogram).
fn check_layout(latents: &Tensor) -> Result<()> {
    let dims = latents.dims();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != PREVIEW_LATENT_CHANNELS {
        return Err(CandleError::Msg(format!(
            "SANA preview latent must have shape [1, {PREVIEW_LATENT_CHANNELS}, h, w], got {dims:?}"
        )));
    }
    Ok(())
}

/// The preview hook the **base** lane hands [`candle_gen::run_flow_sampler`]: a projector closure over
/// [`project_base_latents`]. The driver owns frame numbering, multi-eval dedup and the
/// swallow-on-failure contract (sc-16949), so the route does not restructure its loop.
///
/// [`candle_gen::preview::PreviewHook::new`] rather than `with_sigma`, because
/// [`candle_gen::gen_core::sampling::FlowModelSampling`]'s `input_scale` is identically `1.0`.
///
/// Build it **per image**: a batched request runs one driver call per seed and each call must start a
/// fresh trajectory at frame 1.
pub(crate) fn base_hook(sink: &PreviewSink) -> PreviewHook<'_> {
    PreviewHook::new(sink, project_base_latents)
}

/// The preview hook the **Sprint** lane hands [`candle_gen::run_scm_sampler`]: a projector closure
/// over [`project_sprint_latents`] carrying [`SPRINT_INVERSE_SIGMA_DATA`].
///
/// Still [`candle_gen::preview::PreviewHook::new`] rather than `with_sigma`: the SCM driver has no σ
/// to report and would hand a `with_sigma` projector `None`, whereas the correction Sprint needs is a
/// schedule-independent constant. The driver keys these frames on the **step index**
/// ([`candle_gen::preview::PreviewCounter::with_steps`]), not on a σ array it does not have.
pub(crate) fn sprint_hook(sink: &PreviewSink) -> PreviewHook<'_> {
    PreviewHook::new(sink, |latents: &Tensor| {
        project_sprint_latents(latents, SPRINT_INVERSE_SIGMA_DATA)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use candle_gen::candle_core::{DType, Device};
    use candle_gen::gen_core::sampling::{FlowModelSampling, ModelSampling, TimestepConvention};
    use candle_gen::gen_core::PreviewFrame;
    use candle_gen::preview::PreviewCounter;
    use candle_gen::{ScmScheduler, SCM_SIGMA_DATA};

    use super::*;

    fn zeros(shape: (usize, usize, usize, usize)) -> Tensor {
        Tensor::zeros(shape, DType::F32, &Device::Cpu).expect("zeros")
    }

    /// A deterministic non-constant latent — a fit applied to zeros returns its bias and would let
    /// two different factor tables look identical.
    fn ramp(channels: usize, h: usize, w: usize) -> Tensor {
        let n = channels * h * w;
        let values: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.37).sin() * 2.5 + ((i % 7) as f32) * 0.11)
            .collect();
        Tensor::from_vec(values, (1, channels, h, w), &Device::Cpu).expect("ramp")
    }

    fn collecting_sink() -> (PreviewSink, Arc<Mutex<Vec<PreviewFrame>>>) {
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| candle_gen::lock_recover(&captured).push(frame));
        (sink, frames)
    }

    // ── The two fits ──────────────────────────────────────────────────────────────────────────────

    /// Each reused table must have one finite row per latent channel, and the channel count they
    /// define must be the one the rest of the crate denoises and decodes in.
    #[test]
    fn both_reused_fits_have_one_finite_row_per_sana_latent_channel() {
        for (label, factors, bias) in [
            ("base", &BASE_RGB_FACTORS, &BASE_RGB_BIAS),
            ("sprint", &SPRINT_RGB_FACTORS, &SPRINT_RGB_BIAS),
        ] {
            assert_eq!(factors.len(), 32, "{label}");
            assert!(
                factors.iter().flatten().all(|v| v.is_finite()),
                "{label}: non-finite factor"
            );
            assert!(
                bias.iter().all(|v| v.is_finite()),
                "{label}: non-finite bias"
            );
        }
        assert_eq!(PREVIEW_LATENT_CHANNELS, LATENT_CHANNELS);
    }

    /// **The story's insurance test.** Base and Sprint decode the same latent with different weights,
    /// so they do not share a fit — and the specific mistake sc-16959 exists to avoid is one set of
    /// constants serving both routes.
    ///
    /// Asserted in the strong form: **every one of the 33 rows differs**, all 32 factor rows and the
    /// bias. A weaker "the tables are not equal" check would pass on a copy-paste that replaced only
    /// one row, which is the shape a real accident takes — and it is a plausible accident precisely
    /// because the two tables are structurally alike: the Sprint DC-AE is a decoder-tail fine-tune of
    /// the base one, sharing 320 of its 375 tensors.
    #[test]
    fn the_base_and_sprint_fits_share_no_row() {
        assert_ne!(
            BASE_RGB_BIAS, SPRINT_RGB_BIAS,
            "base and Sprint must not share the intercept"
        );
        let shared: Vec<usize> = (0..PREVIEW_LATENT_CHANNELS)
            .filter(|&i| BASE_RGB_FACTORS[i] == SPRINT_RGB_FACTORS[i])
            .collect();
        assert!(
            shared.is_empty(),
            "base and Sprint are fits over two DIFFERENT DC-AE decoders (SHA-256 15a4b09e… vs \
             dfd991d1…, same container size, 55 of 375 tensors differing and every one of them \
             decoder-side); rows {shared:?} are identical, which means one route's constants were \
             copied onto the other's"
        );
    }

    /// The constants are the epic-16624 SANA numbers, not another 32-channel family's. Spot-pinned on
    /// both biases — the row most likely to be copied by accident, since every family's is a
    /// near-neutral grey — plus each table's largest-magnitude row.
    ///
    /// FLUX.2's bias is named in the failure message rather than imported: this crate deliberately
    /// does not depend on `candle-gen-flux2`, and the point is that SANA's 32-channel space is its own.
    #[test]
    fn the_committed_constants_are_the_sana_ones() {
        assert_eq!(BASE_RGB_BIAS, [0.467_3, 0.437_615_22, 0.414_471_12]);
        assert_eq!(SPRINT_RGB_BIAS, [0.457_922_9, 0.428_063_12, 0.399_165_84]);
        assert_eq!(
            BASE_RGB_FACTORS[24],
            [-0.092_541_26, -0.101_964_61, -0.100_527_29]
        );
        assert_eq!(
            SPRINT_RGB_FACTORS[24],
            [-0.095_328_63, -0.109_609_84, -0.106_988_52]
        );
        for bias in [BASE_RGB_BIAS, SPRINT_RGB_BIAS] {
            assert_ne!(
                bias,
                [0.440_938_92, 0.424_318_4, 0.409_667_16],
                "this is candle-gen-flux2's 32-channel bias — SANA's 32-channel space is a \
                 different one"
            );
        }
    }

    /// The runtime half of the same guarantee: the two projectors must disagree about the same
    /// tensor. Constants can be distinct on paper and still be wired to the wrong route, and a
    /// zero latent would hide it because both fits would then return only their (different) biases —
    /// hence the non-constant ramp.
    ///
    /// `inverse_sigma_data = 1.0` isolates the fit from the σ_data correction, so a failure here can
    /// only mean the factor tables were collapsed.
    #[test]
    fn projecting_one_latent_through_both_fits_gives_different_frames() {
        let latents = ramp(PREVIEW_LATENT_CHANNELS, 6, 7);
        let base = project_base_latents(&latents).expect("base projection");
        let sprint = project_sprint_latents(&latents, 1.0).expect("sprint projection");
        assert_eq!((base.width, base.height), (sprint.width, sprint.height));
        assert_ne!(
            base.pixels, sprint.pixels,
            "the base and Sprint projectors must not produce the same frame for one latent — if they \
             do, one route is projecting through the other's fit"
        );
    }

    // ── The SCM σ_data correction ─────────────────────────────────────────────────────────────────

    /// The correction is the **exact inverse** of what [`candle_gen::run_scm_sampler`] applies, not
    /// merely "some scaling".
    ///
    /// The driver multiplies the seed latent by `σ_data` on entry and hands the hook that scaled
    /// tensor. So projecting the scaled latent with `1/σ_data` must be byte-identical to projecting
    /// the unscaled one with no correction at all. Dropping the correction, or inverting it, breaks
    /// this equality.
    #[test]
    fn the_sprint_correction_exactly_inverts_the_drivers_sigma_data_pre_scale() {
        let raw = ramp(PREVIEW_LATENT_CHANNELS, 5, 5);
        // Exactly what the driver's `latents.affine(sd, 0.0)` produces on entry.
        let as_the_driver_hands_it = raw
            .affine(SCM_SIGMA_DATA as f64, 0.0)
            .expect("sigma_data pre-scale");

        let corrected = project_sprint_latents(&as_the_driver_hands_it, SPRINT_INVERSE_SIGMA_DATA)
            .expect("corrected projection");
        let reference = project_sprint_latents(&raw, 1.0).expect("reference projection");
        assert_eq!(
            corrected.pixels, reference.pixels,
            "1/σ_data must recover exactly the latent the fit was measured on"
        );

        let uncorrected =
            project_sprint_latents(&as_the_driver_hands_it, 1.0).expect("uncorrected projection");
        assert_ne!(
            uncorrected.pixels, reference.pixels,
            "if the uncorrected projection already matched, this row would prove nothing about the \
             correction"
        );
    }

    /// What the missing correction actually costs, measured rather than asserted in prose: `σ_data`
    /// is `0.5`, so an uncorrected Sprint preview projects a latent of **half** the magnitude and
    /// collapses toward the fit's own intercept — a flat, low-contrast frame rather than the denoise.
    ///
    /// Measured as the mean absolute deviation from the projected all-zero latent (which IS the
    /// intercept), so the statistic is exactly "how far this frame gets from flat grey".
    #[test]
    fn an_uncorrected_sprint_projection_collapses_toward_the_intercept() {
        let raw = ramp(PREVIEW_LATENT_CHANNELS, 8, 8);
        let scaled = raw
            .affine(SCM_SIGMA_DATA as f64, 0.0)
            .expect("sigma_data pre-scale");
        let flat = project_sprint_latents(&zeros((1, PREVIEW_LATENT_CHANNELS, 8, 8)), 1.0)
            .expect("intercept frame");
        let spread = |image: &Image| -> f64 {
            image
                .pixels
                .iter()
                .zip(&flat.pixels)
                .map(|(&v, &g)| (v as i32 - g as i32).unsigned_abs() as f64)
                .sum::<f64>()
                / image.pixels.len() as f64
        };

        let corrected =
            spread(&project_sprint_latents(&scaled, SPRINT_INVERSE_SIGMA_DATA).expect("corrected"));
        let uncorrected = spread(&project_sprint_latents(&scaled, 1.0).expect("uncorrected"));
        assert!(
            corrected > uncorrected * 1.5,
            "an uncorrected Sprint preview must be visibly flatter than a corrected one \
             (corrected {corrected:.2}, uncorrected {uncorrected:.2})"
        );
    }

    /// [`SPRINT_INVERSE_SIGMA_DATA`] inverts the value the driver actually divides by, through **both**
    /// public [`ScmScheduler`] constructors and across the whole 1–4 Sprint operating band (plus a
    /// couple past it).
    ///
    /// The driver reads `scheduler.sigma_data`; this module derives its correction from
    /// [`candle_gen::SCM_SIGMA_DATA`]. This row is what binds the two, so a future scheduler that
    /// carried a per-schedule `sigma_data` would fail here rather than silently mis-scale every Sprint
    /// preview.
    #[test]
    fn the_scm_scheduler_always_carries_the_sigma_data_this_correction_inverts() {
        for steps in 1..=8usize {
            assert_eq!(
                ScmScheduler::new(steps).sigma_data,
                SCM_SIGMA_DATA,
                "ScmScheduler::new({steps})"
            );
        }
        assert_eq!(
            ScmScheduler::from_timesteps(vec![1.2, 0.6, 0.0]).sigma_data,
            SCM_SIGMA_DATA
        );
        assert_eq!(SPRINT_INVERSE_SIGMA_DATA, 1.0 / SCM_SIGMA_DATA);
        assert_eq!(SCM_SIGMA_DATA, 0.5, "the diffusers SCMScheduler value");
    }

    // ── The layout contract ───────────────────────────────────────────────────────────────────────

    /// The native spatial latent projects at latent resolution, through both projectors, with no
    /// unpack and no squeeze — the shape claim the story asked to be confirmed per route rather than
    /// assumed.
    #[test]
    fn the_native_spatial_latent_projects_at_latent_resolution_on_both_routes() {
        for image in [
            project_base_latents(&zeros((1, 32, 3, 5))).expect("base"),
            project_sprint_latents(&zeros((1, 32, 3, 5)), SPRINT_INVERSE_SIGMA_DATA)
                .expect("sprint"),
        ] {
            assert_eq!((image.width, image.height), (5, 3));
            assert_eq!(image.pixels.len(), 5 * 3 * 3);
        }
    }

    /// A 1024² render previews at 32×32, because the DC-AE compresses by
    /// [`crate::pipeline::SPATIAL_SCALE`]. Self-describing rather than a bare number, and derived from
    /// the crate's own noise builder so it cannot drift from what the sampler integrates.
    #[test]
    fn the_shipped_resolutions_preview_at_the_dc_ae_latent_edge() {
        for edge in [256u32, 512, 1024] {
            let latents =
                crate::pipeline::create_noise(&Device::Cpu, 0, edge, edge).expect("seed noise");
            let expected = edge / crate::pipeline::SPATIAL_SCALE;
            for image in [
                project_base_latents(&latents).expect("base"),
                project_sprint_latents(&latents, SPRINT_INVERSE_SIGMA_DATA).expect("sprint"),
            ] {
                assert_eq!((image.width, image.height), (expected, expected));
            }
        }
    }

    /// Anything that is not one batch-1 thirty-two-channel spatial latent is refused by **both**
    /// projectors, and the message says SANA. The rank-3 row is the one that matters: a packed token
    /// sequence is what a copy-paste from a FLUX-family crate would produce, and it must not project.
    #[test]
    fn a_latent_that_is_not_this_family_is_refused_on_both_routes() {
        let packed = Tensor::zeros((1, 256, 128), DType::F32, &Device::Cpu).expect("packed");
        let five_d = Tensor::zeros((1, 32, 1, 4, 4), DType::F32, &Device::Cpu).expect("5-D");
        for bad in [
            zeros((2, 32, 4, 4)), // batched
            zeros((1, 16, 4, 4)), // SD3.5 / FLUX.1's channel width
            zeros((1, 4, 4, 4)),  // SDXL's channel width
            packed,               // a packed FLUX-family token sequence
            five_d,               // Z-Image's 5-D frame-axis latent
        ] {
            for error in [
                project_base_latents(&bad).expect_err("base must refuse"),
                project_sprint_latents(&bad, SPRINT_INVERSE_SIGMA_DATA)
                    .expect_err("sprint must refuse"),
            ] {
                let error = error.to_string();
                assert!(
                    error.contains("SANA preview latent must have shape [1, 32, h, w]"),
                    "got: {error}"
                );
            }
        }
    }

    // ── The hooks ─────────────────────────────────────────────────────────────────────────────────

    /// The base hook emits one frame per schedule position, dedups a repeat before projecting, and
    /// reports the driver's numbering — the multi-eval contract, exercised through this family's own
    /// hook rather than only through the shared module's tests.
    #[test]
    fn the_base_hook_numbers_frames_and_dedups_a_repeated_position() {
        let sigmas = [1.0f32, 0.5, 0.0];
        let (sink, frames) = collecting_sink();
        let hook = base_hook(&sink);
        let counter = PreviewCounter::new(&sigmas);
        let latents = zeros((1, 32, 2, 2));

        hook.emit(&counter, &sigmas, sigmas[0], &latents);
        hook.emit(&counter, &sigmas, sigmas[0], &latents); // a heun-style repeat
        hook.emit(&counter, &sigmas, sigmas[1], &latents);

        let frames = candle_gen::lock_recover(&frames);
        assert_eq!(
            frames
                .iter()
                .map(|f| (f.current, f.total))
                .collect::<Vec<_>>(),
            vec![(1, 2), (2, 2)],
            "a two-step schedule evaluated three times must emit exactly two frames"
        );
    }

    /// The Sprint hook is **step-index keyed**, because the SCM driver has no σ array. One frame per
    /// step over the 4-step schedule, numbered 1..=4 with `total == 4`, and a repeated index emits
    /// nothing.
    #[test]
    fn the_sprint_hook_numbers_frames_by_step_index() {
        let (sink, frames) = collecting_sink();
        let hook = sprint_hook(&sink);
        let counter = PreviewCounter::with_steps(4);
        let latents = zeros((1, 32, 2, 2));

        for step in 0..4 {
            hook.emit_step(&counter, step, &latents);
        }
        hook.emit_step(&counter, 3, &latents); // a repeat

        let frames = candle_gen::lock_recover(&frames);
        assert_eq!(
            frames
                .iter()
                .map(|f| (f.current, f.total))
                .collect::<Vec<_>>(),
            (1..=4).map(|n| (n, 4)).collect::<Vec<_>>()
        );
    }

    /// The **hook** applies the correction, not just the projector it wraps.
    ///
    /// `the_sprint_correction_exactly_inverts_the_drivers_sigma_data_pre_scale` proves
    /// `project_sprint_latents` inverts the pre-scale when handed `1/σ_data`; nothing there would
    /// notice [`sprint_hook`] passing `1.0` instead. This row closes that gap by driving the shipped
    /// hook with the tensor the SCM loop actually hands it and comparing the emitted frame against the
    /// unscaled reference.
    #[test]
    fn the_sprint_hook_applies_the_sigma_data_correction_it_carries() {
        let raw = ramp(PREVIEW_LATENT_CHANNELS, 5, 6);
        let as_the_driver_hands_it = raw
            .affine(candle_gen::SCM_SIGMA_DATA as f64, 0.0)
            .expect("sigma_data pre-scale");

        let (sink, frames) = collecting_sink();
        let hook = sprint_hook(&sink);
        hook.emit_step(&PreviewCounter::with_steps(1), 0, &as_the_driver_hands_it);

        let frames = candle_gen::lock_recover(&frames);
        assert_eq!(frames.len(), 1);
        let reference = project_sprint_latents(&raw, 1.0).expect("reference projection");
        assert_eq!(
            frames[0].image.pixels, reference.pixels,
            "sprint_hook must carry SPRINT_INVERSE_SIGMA_DATA — an uncorrected hook emits a frame \
             over half the latent the fit was measured on"
        );
        // Non-vacuity: the uncorrected frame really is different, so this row can fail.
        let uncorrected =
            project_sprint_latents(&as_the_driver_hands_it, 1.0).expect("uncorrected projection");
        assert_ne!(uncorrected.pixels, reference.pixels);
    }

    /// **The single-step Sprint case, asserted explicitly** (the story names it): exactly one frame,
    /// `current == 1`, `total == 1`, no division by zero and no stall.
    ///
    /// `ScmScheduler::new(1)` is a real request shape — `is_single_step()` is true and the driver
    /// skips the renoise — and a counter built from `steps = 1` is the degenerate case where an
    /// `n - 1` denominator would divide by zero.
    #[test]
    fn a_single_step_sprint_schedule_emits_exactly_one_frame() {
        let scheduler = ScmScheduler::new(1);
        assert!(scheduler.is_single_step());
        assert_eq!(scheduler.num_steps(), 1);

        let (sink, frames) = collecting_sink();
        let hook = sprint_hook(&sink);
        let counter = PreviewCounter::with_steps(scheduler.num_steps());
        assert_eq!(counter.total(), 1);
        let latents = zeros((1, 32, 2, 2));

        hook.emit_step(&counter, 0, &latents);
        // The loop runs once, so index 0 is the only one reached; a second call cannot re-emit.
        hook.emit_step(&counter, 0, &latents);

        let frames = candle_gen::lock_recover(&frames);
        assert_eq!(frames.len(), 1, "a 1-step Sprint render emits one frame");
        assert_eq!((frames[0].current, frames[0].total), (1, 1));
    }

    /// A malformed latent loses exactly one decorative frame and consumes its schedule position on
    /// both routes; the render is unaffected. The failure is swallowed by the shared emitter, never
    /// surfaced.
    #[test]
    fn a_malformed_latent_loses_one_frame_and_consumes_its_position() {
        let sigmas = [1.0f32, 0.5, 0.0];
        let (sink, frames) = collecting_sink();
        let hook = base_hook(&sink);
        let counter = PreviewCounter::new(&sigmas);
        hook.emit(&counter, &sigmas, sigmas[0], &zeros((1, 31, 2, 2)));
        hook.emit(&counter, &sigmas, sigmas[1], &zeros((1, 32, 2, 2)));
        let frames = candle_gen::lock_recover(&frames);
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].current, frames[0].total), (2, 2));

        let (sink, frames) = collecting_sink();
        let hook = sprint_hook(&sink);
        let counter = PreviewCounter::with_steps(2);
        hook.emit_step(&counter, 0, &zeros((1, 31, 2, 2)));
        hook.emit_step(&counter, 1, &zeros((1, 32, 2, 2)));
        let frames = candle_gen::lock_recover(&frames);
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].current, frames[0].total), (2, 2));
    }

    /// An inert sink does no tensor work at all and does not advance either counter — the property
    /// that makes an unwatched render byte-identical to a pre-sc-16959 one.
    #[test]
    fn an_inert_sink_projects_nothing_and_advances_nothing_on_either_route() {
        let inert = PreviewSink::default();
        let sigmas = [1.0f32, 0.0];
        let base = base_hook(&inert);
        assert!(!base.is_active());
        let counter = PreviewCounter::new(&sigmas);
        // A latent that would fail the projection if it were ever reached is not reached.
        base.emit(&counter, &sigmas, sigmas[0], &zeros((1, 31, 2, 2)));
        assert_eq!(counter.next(&sigmas, sigmas[0]), Some(1));

        let sprint = sprint_hook(&inert);
        assert!(!sprint.is_active());
        let counter = PreviewCounter::with_steps(2);
        sprint.emit_step(&counter, 0, &zeros((1, 31, 2, 2)));
        assert_eq!(counter.next_step(0), Some(1));
    }

    // ── The σ convention ──────────────────────────────────────────────────────────────────────────

    /// Base drives `TimestepConvention::Sigma` over a `FlowModelSampling`, whose `input_scale` is
    /// identically 1.0 — so the running latent already is the tensor the base fit was measured against
    /// and [`base_hook`]'s `PreviewHook::new` is the correct constructor rather than `with_sigma`.
    ///
    /// Read off the very `ModelSampling` the driver integrates rather than asserted about the family
    /// in prose. `tests/preview_real_weights.rs` measures the consequence — the first frame's
    /// rail-clipped fraction — on this route's own prior, and on Sprint's separately.
    #[test]
    fn the_flow_route_needs_no_input_scaling() {
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        for sigma in [0.0f32, 0.05, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(
                ms.input_scale(sigma),
                1.0,
                "FlowModelSampling::input_scale must be identically 1.0; at {sigma} it is not, and \
                 base SANA would need PreviewHook::with_sigma"
            );
        }
        // And the schedule the pipeline actually builds starts at the σ that measurement assumes.
        assert_eq!(crate::pipeline::sana_sigmas(None, 20)[0], 1.0);
    }

    // ── The wiring, pinned against this crate's own source ────────────────────────────────────────

    /// The shipped half of a source file — everything ahead of its single `#[cfg(test)] mod tests`.
    ///
    /// The structural rows in those modules drive the hooks over inert sinks, so a scan that counted
    /// them would read test code as shipped wiring.
    fn shipped(source: &'static str, name: &str) -> &'static str {
        const MARKER: &str = "\n#[cfg(test)]\n";
        assert_eq!(
            source.matches(MARKER).count(),
            1,
            "{name} must hold exactly one `#[cfg(test)]` item for this split to be sound — teach \
             this scan about the new one rather than letting it read test code as shipped code"
        );
        &source[..source.find(MARKER).expect("the marker was just counted")]
    }

    fn shipped_model() -> &'static str {
        shipped(include_str!("model.rs"), "model.rs")
    }

    fn shipped_pipeline() -> &'static str {
        shipped(include_str!("pipeline.rs"), "pipeline.rs")
    }

    /// Read the `preview:` parameter out of a function's own declaration, so a pin cannot be satisfied
    /// by the spelling appearing somewhere else in the file.
    ///
    /// The parameter list ends at the first line whose trimmed form opens the return type — matched
    /// indentation-agnostically, because these declarations live at three different nesting depths
    /// (free function, trait method, inherent method).
    fn preview_parameter(source: &str, declaration: &str) -> String {
        let at = source
            .find(declaration)
            .unwrap_or_else(|| panic!("{declaration} must be declared in this file"));
        let mut parameter = None;
        for line in source[at..].lines().map(str::trim) {
            if line.starts_with(") ->") {
                return parameter
                    .unwrap_or_else(|| panic!("{declaration} must take a preview parameter"));
            }
            if line.starts_with("preview:") {
                parameter = Some(line.to_string());
            }
        }
        panic!("{declaration}'s parameter list must end at its return type");
    }

    /// The exact declaration every hop between the request's sink and a driver must carry.
    const WANT: &str = "preview: &candle_gen::preview::PreviewHook<'_>,";

    /// Count `WANT` by **whole trimmed line**, not by substring.
    ///
    /// A substring tally (`source.matches(WANT).count()`) is satisfied by the same declaration
    /// renamed `_preview:` — the spelling a hop takes the moment it stops *using* its hook — so it
    /// would let a hop become an ignored parameter and still count toward the total.
    fn hook_parameters(source: &str) -> usize {
        source.lines().filter(|line| line.trim() == WANT).count()
    }

    /// **Both** registered lanes build their hook over the **request's** sink, each with its own
    /// constructor, and every hop between that sink and the driver carries the hook as a
    /// **non-`Option` reference**.
    ///
    /// Both halves are load-bearing and neither implies the other, because the epic's cross-crate
    /// guard cannot see either. `candle-gen-catalog`'s `preview_advertising` inventory classifies the
    /// preview argument of the driver calls *inside* `pipeline.rs` — several hops past `model.rs`.
    /// sc-16958 showed what an `Option` anywhere on that path costs: blanking one caller argument took
    /// every lane of that family dark while `hooked` counts, `PREVIEW_ROUTE_IDS` and
    /// `supports_preview: true` all went on advertising and the whole CPU suite stayed green.
    ///
    /// A `&PreviewHook` parameter is what makes *widening the seam* — passing `None`, or `Option`-ing
    /// a hop — a **type error**. This row pins every parameter on the path so that property cannot be
    /// undone by quietly widening one back.
    ///
    /// ## Exactly what is and is not mechanically enforced (sc-16959 review)
    ///
    /// The non-`Option` typing is **not** an absolute immunity, and the earlier wording here claimed
    /// it was. sc-16959's reviewer took both lanes dark with **zero** type errors and the full CPU and
    /// catalog suites green, two different ways, both inside `model.rs`:
    ///
    /// * `impl BaseBatchPipeline for SanaPipeline`'s `render_seed` accepted the forwarded hook,
    ///   **ignored** it, and built a fresh `candle_gen::preview::PreviewHook::new(&inert, …)`. Every
    ///   parameter on the path still read `&PreviewHook`; the `_hook(` count still read 2, because
    ///   `PreviewHook::new(` does not contain the substring `_hook(`.
    /// * `SanaGenerator::generate` rebound `let req = &GenerationRequest { preview:
    ///   PreviewSink::default(), ..req.clone() };` ahead of `preview::base_hook(&req.preview)`. The
    ///   literal the scan looks for was still there, exactly once — over a sink that had been emptied.
    ///
    /// Neither is reachable by the CPU suite: `tests/preview_wiring.rs` enters at `denoise_cfg` /
    /// `denoise_sprint`, because everything above them needs a loaded snapshot. So the whole `model.rs`
    /// adapter layer is guarded by **text**, and this row states that plainly rather than implying the
    /// types cover it. What the text now pins, and why each is the mutation's only cheap spelling:
    ///
    /// * **zero `PreviewHook::new`** in shipped `model.rs` *and* shipped `pipeline.rs` — the sinks are
    ///   reached only through `preview::base_hook` / `preview::sprint_hook`, whose call sites are
    ///   counted. Building a hook the other spelling instead (`base_hook(&inert)` in `model.rs`) trips
    ///   the `_hook(` count of 2.
    /// * **zero `GenerationRequest {`** in shipped `model.rs` — the adapters read the caller's request
    ///   and never construct one, so a rebind that swaps `preview` out cannot hide behind the
    ///   still-correct `base_hook(&req.preview)` literal.
    ///
    /// What is *not* caught: an edit that reaches the same end by some third construction — a helper
    /// that returns an emptied request, a `GenerationRequest{` with no space. Closing that needs a
    /// render through the registered `Generator` seam with a live sink, which needs weights; the
    /// real-weight lane in `tests/preview_real_weights.rs` does exactly that, on CUDA, and is the only
    /// place this crate proves the seam end to end.
    #[test]
    fn both_render_lanes_build_their_hook_from_the_requests_sink() {
        let model = shipped_model();

        // The sinks: exactly one hook per route in shipped `model.rs`, each over the request's sink,
        // and no third hook anywhere in the file.
        assert_eq!(
            model.matches("preview::base_hook(&req.preview)").count(),
            1,
            "the base render lane must build exactly one hook, over the REQUEST's sink"
        );
        assert_eq!(
            model.matches("preview::sprint_hook(&req.preview)").count(),
            1,
            "the Sprint render lane must build exactly one hook, over the REQUEST's sink"
        );
        assert_eq!(
            model.matches("_hook(").count(),
            2,
            "shipped model.rs must build exactly two preview hooks — a third render lane must be \
             named in this crate's inventory (and in the catalog's) rather than appearing here"
        );

        // The Sprint hook must NOT be the base one and vice versa: the ids and the constructors are
        // read out of the same file, so a swap is visible here rather than only in a render.
        let base_at = model
            .find("fn generate_base_images(")
            .expect("generate_base_images");
        let sprint_at = model
            .find("fn generate_sprint_images(")
            .expect("generate_sprint_images");
        assert!(
            model[base_at..sprint_at].contains("preview::base_hook(&req.preview)"),
            "the base adapter must build the BASE hook"
        );
        assert!(
            model[sprint_at..].contains("preview::sprint_hook(&req.preview)"),
            "the Sprint adapter must build the SPRINT hook"
        );

        // The two darkening edits the reviewer demonstrated, each with the type system fully
        // satisfied and every count above still reading exactly what it expects. See this row's
        // rustdoc for why text is the only instrument available here.
        assert_eq!(
            model.matches("PreviewHook::new").count(),
            0,
            "shipped model.rs must never CONSTRUCT a hook — it may only call `preview::base_hook` / \
             `preview::sprint_hook` over the request's own sink. A `PreviewHook::new(&inert, …)` \
             inside a `render_seed` that accepts and then ignores its forwarded hook takes the lane \
             dark with no type error, and `PreviewHook::new(` does not contain `_hook(`, so the \
             count of two above cannot see it"
        );
        assert_eq!(
            model.matches("GenerationRequest {").count(),
            0,
            "shipped model.rs must never CONSTRUCT a GenerationRequest — the adapters read the \
             caller's. `let req = &GenerationRequest {{ preview: PreviewSink::default(), \
             ..req.clone() }};` ahead of `preview::base_hook(&req.preview)` empties the sink while \
             leaving that literal intact, exactly once, which is all the scan above checks"
        );

        // Every hop, in both files, takes the hook by non-`Option` reference.
        //
        // Counted by WHOLE trimmed line rather than by substring: `model.matches(WANT)` is satisfied
        // by `_preview: &candle_gen::preview::PreviewHook<'_>,`, so a hop renamed to the ignore-me
        // spelling — the first half of the `render_seed` mutation above — would still count toward
        // the six.
        for (source, declaration) in [
            (model, "fn generate_base_images("),
            (model, "fn generate_sprint_images("),
            (model, "    fn render_seed("),
        ] {
            assert_eq!(
                preview_parameter(source, declaration),
                WANT,
                "{declaration} must take its hook by reference. An `Option` here is blankable at the \
                 caller, and that mutation is invisible to the catalog's route inventory — it \
                 classifies the driver argument several hops further in, which would still read \
                 `Some(preview)`."
            );
        }
        // `render_seed` is declared on two traits and implemented on two pipelines; every one of the
        // four must carry the same non-`Option` parameter. `preview_parameter` above reads only the
        // FIRST `fn render_seed(` — the trait declaration — so this line-exact tally is what holds
        // the other three.
        assert_eq!(
            hook_parameters(model),
            6,
            "both adapters plus all four `render_seed` declarations/impls must take `&PreviewHook` \
             under that exact name — a hop renamed `_preview:` is one that no longer uses it"
        );

        let pipeline = shipped_pipeline();
        for declaration in [
            "pub fn denoise_cfg(",
            "pub fn denoise_sprint(",
            "    pub fn generate_with(",
            "    pub(crate) fn generate_with_conditioning(",
        ] {
            assert_eq!(
                preview_parameter(pipeline, declaration),
                WANT,
                "{declaration} must take its hook by reference"
            );
        }
        // Both `generate_with` / `generate_with_conditioning` pairs (base and Sprint) plus the two
        // free denoise functions: six declarations in shipped `pipeline.rs`, line-exact for the same
        // reason as `model.rs` above.
        assert_eq!(hook_parameters(pipeline), 6);

        // The only hooks shipped `pipeline.rs` builds are the two documented INERT ones in the
        // `generate` convenience wrappers. A hook over anything else there would be a second wiring
        // layer the catalog's inventory cannot distinguish from the first.
        assert_eq!(pipeline.matches("preview::base_hook(&inert)").count(), 1);
        assert_eq!(pipeline.matches("preview::sprint_hook(&inert)").count(), 1);
        assert_eq!(
            pipeline.matches("_hook(").count(),
            2,
            "shipped pipeline.rs must build exactly the two inert convenience hooks — the request's \
             sink is reached only through model.rs"
        );
        assert_eq!(
            pipeline.matches("PreviewHook::new").count(),
            0,
            "shipped pipeline.rs must not CONSTRUCT a hook either — the two inert ones go through \
             `preview::base_hook` / `preview::sprint_hook`, which the count above sees. A third, \
             built directly over some other sink between `generate_with_conditioning` and the driver \
             call, would be invisible to every count in this row"
        );
    }

    /// The two driver calls are hooked, and hooked with `Some(preview)` — the argument
    /// `candle-gen-catalog`'s route inventory classifies.
    ///
    /// Pinned positionally rather than by "the file mentions `Some(preview)`": the preview argument is
    /// index 7 of 9 for `run_flow_sampler` and index 5 of 7 for `run_scm_sampler`, and an argument
    /// inserted ahead of it would otherwise shift what this row believes it is reading.
    #[test]
    fn both_sampler_calls_pass_the_hook_at_the_drivers_preview_position() {
        let pipeline = shipped_pipeline();
        for (call, preview_at, arity) in [
            ("run_flow_sampler(", 7usize, 9usize),
            ("run_scm_sampler(", 5, 7),
        ] {
            let at = pipeline
                .find(call)
                .unwrap_or_else(|| panic!("{call} must appear in shipped pipeline.rs"));
            assert_eq!(
                pipeline.matches(call).count(),
                1,
                "{call}: this crate must drive each sampler from exactly one site — a second site \
                 must join the catalog's route inventory rather than appear silently"
            );
            let arguments = split_arguments(&pipeline[at + call.len()..]);
            assert_eq!(arguments.len(), arity, "{call}: argument count");
            assert_eq!(
                arguments[preview_at], "Some(preview)",
                "{call}: argument {preview_at} is the driver's `preview` parameter and must be the \
                 forwarded hook"
            );
        }
    }

    /// Split a call's argument list, from just past its open paren to the matching close paren,
    /// at top-level commas, dropping rustfmt's trailing comma. Deliberately simple: the two calls it
    /// reads are plain argument lists.
    fn split_arguments(rest: &str) -> Vec<String> {
        let (mut depth, mut current, mut out) = (0i32, String::new(), Vec::new());
        for ch in rest.chars() {
            match ch {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' if depth == 0 => {
                    let tail = current.trim();
                    if !tail.is_empty() {
                        out.push(tail.to_string());
                    }
                    return out;
                }
                ')' | ']' | '}' => depth -= 1,
                ',' if depth == 0 => {
                    out.push(current.trim().to_string());
                    current.clear();
                    continue;
                }
                _ => {}
            }
            current.push(ch);
        }
        panic!("unterminated argument list");
    }
}
