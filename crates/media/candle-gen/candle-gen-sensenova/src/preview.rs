//! SenseNova-U1's per-step preview seam (epic 16948, sc-16960) — the epic's **Tier 2** family, and
//! the only one that needed a fit of its own rather than a reuse.
//!
//! Schedule numbering, dedup and the swallow-on-failure contract live in [`candle_gen::preview`],
//! shared by every candle family (sc-16949). This module owns three things: the fit, the pool that
//! takes the running state to the grid the fit was measured on, and the layout check that says the
//! tensor being projected is this family's.
//!
//! ## SenseNova-U1 has **no VAE** — the premise this story was scoped on is false, and that is the
//! whole reason the measurement came out where it did
//!
//! Epic 16948 filed this story as "SenseNova-U1 has its own VAE and MLX never fitted it". The first
//! half is wrong. `crate`'s own module docs already say so — *"there is no separate VAE or text
//! encoder"* — and the sources agree end to end: `config.json` carries no autoencoder section at all,
//! [`crate::T2iModel`] loads `language_model.*` + `fm_modules.*` and nothing else, and its
//! flow-matching head predicts `3·(patch·merge)²` values per token, which `crate::fm::unpatchify`
//! folds straight back into `[1, 3, H, W]`.
//!
//! **SenseNova-U1 denoises in pixel space.** The running state of the bespoke loop *is* the image, in
//! the model's own `[-1, 1]` space, and the "decode" is the affine map `crate::t2i::tensor_to_image`
//! applies: `x·0.5 + 0.5`, clamped. There is no encoder, no latent channel expansion, and nothing to
//! recover before projecting.
//!
//! Two consequences, both load-bearing:
//!
//! * **This space cannot be one of the seven epic-16624 fitted spaces**, and the reason is structural
//!   rather than a hash comparison: those are 4-, 16- and 32-channel VAE latents, and this is a
//!   **3**-channel pixel space belonging to a checkpoint that ships no autoencoder. A reuse was never
//!   available here, which is what put this family in Tier 2 in the first place.
//! * **The fit is near-exact, and that is a result rather than an escape.** Because the decode is
//!   affine, an ordinary-least-squares map from the state to the decoded pixels can recover it
//!   essentially perfectly; the only thing that stops it being exactly perfect is the clamp, which is
//!   the one non-linearity in the path. The measurement is reported honestly for what it is — see
//!   `RGB_FACTORS` (private, hence no link) and
//!   `docs/migration/evidence/sc-16960-sensenova-candle-preview.md` — and `tests/fit_preview_rgb.rs`
//!   is the producer that derived it, on real weights, with a disjoint holdout that never contributed
//!   a coefficient.
//!
//! ## The preview grid is the model's own token grid
//!
//! [`crate::T2iModel::cell`] = `patch_size · merge_size` = **32** for the shipped 8B-MoT
//! checkpoint, and one backbone token is exactly one `cell × cell` pixel patch — the FM head predicts
//! `3 · cell²` numbers per token. So `H/cell × W/cell` is SenseNova's own latent granularity, not a
//! downsample this module invented, and a 1024² render previews at 32×32 for the same reason SANA's
//! `f32` DC-AE does.
//!
//! [`project_running_image`] therefore average-pools the running `[1, 3, H, W]` state by `cell` before
//! projecting. That pool is **the same average the fit's target uses**, so the projector operates in
//! precisely the space the coefficients were measured in.
//!
//! One consequence worth stating because it inverts the usual diagnostic: pooling 32×32 = 1024
//! independent noise samples divides the prior's standard deviation by 32, so SenseNova's **first
//! frame is not a noise field — it is near-flat grey**, and its rail-clipped fraction is ~0 where a
//! VAE family's would be large. Rail-clipping is not the discriminating statistic here; **contrast
//! about the intercept** is, and `tests/preview_real_weights.rs` measures that one instead.
//!
//! ## What a frame can and cannot contain
//!
//! * **CFG never reaches a frame.** The bespoke `denoise` loop runs the unconditional pass as a
//!   *second forward against a second KV cache* inside the step body and blends the two velocities
//!   into one `v_pred` before `crate::fm::euler_step` advances the state. No fused `[2, …]` batch is
//!   ever the running image, so there is no unconditional half to project.
//! * **Text tokens never reach a frame.** The prompt lives entirely in the prefilled KV cache; the
//!   running state is the image grid alone.
//! * **The it2i / interleave loop is not wired.** [`crate::T2iModel`]'s second denoise
//!   (`it2i_denoise`, reached only through [`crate::T2iModel::interleave_gen`]) is the off-registry
//!   understanding surface, is not advertised by either descriptor, and is out of scope for sc-16960
//!   — the edit path is known-corrupted. It emits nothing, deliberately.
//!
//! A stale or absent fit degrades preview colour only; the denoise path never reads these constants.

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::{Image, PreviewSink};
use candle_gen::preview::PreviewHook;
use candle_gen::{CandleError, Result};

/// SenseNova-U1's ordinary-least-squares map from its **pixel-space** running state to
/// latent-resolution RGB (row *i* maps model-space channel *i* to `[r, g, b]`), with [`RGB_BIAS`] the
/// intercept.
///
/// **Measured, not reused — this is the epic's one new fit.** `tests/fit_preview_rgb.rs` renders four
/// diverse prompt/seed corpora and two *disjoint* holdout prompt/seed corpora at 512² with 8
/// flow-match Euler steps, pairs each render's final `cell`-pooled model-space state with its
/// `cell`-pooled clamped decode, and solves the 4×3 normal equations in `f64`. The holdout renders
/// never contribute a coefficient.
///
/// Measured on the run recorded in `docs/migration/evidence/sc-16960-sensenova-candle-preview.md` —
/// `sensenova_u1_8b`, `q8/` tier, 512² × 8 steps at guidance 4.0, token cell 32 ⇒ 16×16 = 256 pooled
/// samples per render:
///
/// | split | R² (R, G, B) | overall R² |
/// | --- | --- | --- |
/// | **fit** — 4 renders, 1024 samples | `(0.99999517, 0.99999016, 0.99998441)` | **`0.99998989`** |
/// | **holdout** — 2 disjoint renders, 512 samples | `(0.99999123, 0.99998945, 0.99997544)` | **`0.99998292`** |
///
/// The epic's bar is a **holdout** R² ≥ 0.88 — the bar that rejected LTX (.984 fit / .619 holdout),
/// Mage (.938 / .806) and Mochi (.847 / .807). The holdout clears it by a wide margin, and the honest
/// reading of *why* is in the module docs: SenseNova has no VAE, so the "decode" is an affine map and
/// OLS recovers it.
///
/// **The residual is the clamp, and it is visible in these very numbers.** The target is the
/// *clamped* decode, and clipping compresses, so the solved gains come out a touch under the analytic
/// `0.5` (0.4995 / 0.4976 / 0.4976) with small cross-channel terms rather than exactly diagonal. The
/// largest distance from the analytic transform is **0.0024188976**, which is what
/// [`ANALYTIC_TOLERANCE`] is derived from.
///
/// Provenance — the **checkpoint**, because there is no autoencoder to name: repo
/// `SceneWorks/sensenova-u1-8b-mlx` @ revision `b6206ea2e888198418b92f3bed31f5506c6183f9`, tier
/// `q8/`, file `model.safetensors`, 19,911,123,700 bytes, SHA-256
/// `8da38dde4c39722259a98cfc47643c88e48cea205595625fdbd9fec097f9dc4f`. That container holds 2,292
/// tensors under exactly three top-level subtrees — `language_model`, `fm_modules`, `vision_model` —
/// and **no autoencoder**. Channel count **3**, pixel space, which is what makes it impossible for
/// this to be one of the seven epic-16624 VAE latent spaces.
///
/// Refit whenever the SenseNova-U1 output transform changes — i.e. if a future checkpoint stops
/// emitting `[-1, 1]` model-space RGB from its FM head.
///
/// The literals are the producer's `f64` output truncated to the shortest decimal that round-trips
/// through `f32` — the producer prints nine decimals, and `clippy::excessive_precision` rejects digits
/// an `f32` cannot hold. Every value below is the same `f32` the full-precision literal would compile
/// to.
const RGB_FACTORS: [[f32; 3]; 3] = [
    [0.499_482_57, 0.001_114_516, 0.001_276_602],
    [-0.000_450_976, 0.497_581_1, -0.000_060_032],
    [0.000_828_717, 0.000_887_129, 0.497_608],
];

/// The fit's intercept — the near-exact mid-grey a fully-zero model-space state projects to. Measured
/// with [`RGB_FACTORS`] on the same run.
const RGB_BIAS: [f32; 3] = [0.500_254_44, 0.500_271_8, 0.499_321_47];

/// The analytic decode transform the measured fit must land on, kept beside the measurement so the
/// two can be compared rather than merely asserted about in prose.
///
/// `crate::t2i::tensor_to_image` maps the model-space image to RGB with `x·0.5 + 0.5`, per channel,
/// with no cross-channel term. That IS the map an OLS over this space is estimating, so the fit is
/// falsifiable against theory: `the_measured_fit_lands_on_the_analytic_decode_transform` bounds every
/// coefficient's distance from it. A fit that had drifted off this — a transposed row, a wrong pool,
/// a target built in the wrong space — would fail there even though its R² stayed near 1.
const ANALYTIC_GAIN: f32 = 0.5;

/// The analytic intercept, the other half of `x·0.5 + 0.5`.
const ANALYTIC_BIAS: f32 = 0.5;

/// How far a measured coefficient may sit from its analytic value.
///
/// **Measured, not chosen.** The largest absolute deviation across all twelve committed coefficients
/// is **0.0024188976** (`RGB_FACTORS[1][1]`, the green gain); this bound is that number rounded up to
/// one significant figure. The deviation exists at all because the fit's target is the *clamped*
/// decode, and the clamp is the one non-linearity in an otherwise exactly affine path.
const ANALYTIC_TOLERANCE: f32 = 3e-3;

/// The channel count the fit is defined over, derived from the committed table's own length so
/// nothing in this crate can drift from it by restating a number.
///
/// **Three**, because SenseNova denoises in pixel space. That single fact is what rules out every
/// epic-16624 latent space (4, 16 and 32 channels) without needing a hash.
pub const PREVIEW_LATENT_CHANNELS: usize = RGB_FACTORS.len();

/// The fit is the three-channel one, and it is square-and-then-some: one row per model-space colour
/// channel, three columns per row. Compile-time, because a runtime row over constants proves nothing
/// a `const` assertion does not prove earlier.
const _: () = assert!(PREVIEW_LATENT_CHANNELS == 3 && RGB_BIAS.len() == 3);

/// Whether a measured coefficient sits within [`ANALYTIC_TOLERANCE`] of its analytic value.
///
/// Spelled without `f32::abs`, which is not `const`.
const fn within_tolerance(value: f32, expected: f32) -> bool {
    let delta = value - expected;
    delta <= ANALYTIC_TOLERANCE && -delta <= ANALYTIC_TOLERANCE
}

/// **The fit is checked against theory at compile time**, coefficient by coefficient.
///
/// `crate::t2i::tensor_to_image` decodes with `x·0.5 + 0.5` per channel and no cross-channel term, so
/// that IS the map the OLS is estimating and a measured table that had drifted off it — a transposed
/// row, a target built in the wrong space, a wrong pool — is a build error rather than a review
/// question. `the_measured_fit_lands_on_the_analytic_decode_transform` restates it at runtime for the
/// message; this is what makes it unskippable.
const _: () = {
    let mut row = 0;
    while row < PREVIEW_LATENT_CHANNELS {
        let mut column = 0;
        while column < 3 {
            let expected = if row == column { ANALYTIC_GAIN } else { 0.0 };
            assert!(
                within_tolerance(RGB_FACTORS[row][column], expected),
                "a committed RGB_FACTORS coefficient is further than ANALYTIC_TOLERANCE from the \
                 analytic decode transform x·0.5 + 0.5 that crate::t2i::tensor_to_image applies"
            );
            column += 1;
        }
        assert!(
            within_tolerance(RGB_BIAS[row], ANALYTIC_BIAS),
            "a committed RGB_BIAS entry is further than ANALYTIC_TOLERANCE from the analytic \
             intercept of the decode transform x·0.5 + 0.5"
        );
        row += 1;
    }
};

/// Project SenseNova's running model-space state `[1, 3, H, W]` to a token-grid RGB8 preview.
///
/// `cell` is [`crate::T2iModel::cell`] — `patch_size · merge_size`, 32 for the shipped 8B-MoT
/// checkpoint — and both spatial edges must be multiples of it, which the generator's `validate`
/// already enforces on every request. The state is average-pooled by `cell` (the same average the
/// fit's target was built with) and then projected with the committed fit, so the frame is
/// `H/cell × W/cell`.
///
/// The pool runs in `f32` regardless of the incoming dtype: the shipped path is `f32` end to end, but
/// a narrow-dtype pool would quantize the average before the projection ever saw it.
///
/// Errors on any other layout. The caller's frame is then lost and swallowed by
/// [`candle_gen::preview::emit_preview_at`], which is the intended decorative-failure behaviour.
pub fn project_running_image(image: &Tensor, cell: usize) -> Result<Image> {
    check_layout(image, cell)?;
    let pooled = image.to_dtype(DType::F32)?.avg_pool2d(cell)?;
    candle_gen::preview::project_latents(&pooled, &RGB_FACTORS, RGB_BIAS)
}

/// Reject anything that is not one batch-1 three-channel model-space image on this render's grid.
///
/// The shared projection would reject a wrong *rank* anyway, but naming the channel count here makes
/// the failure say *SenseNova* — and catches the two cases it cannot see: a `cell` that does not
/// divide the state (which `avg_pool2d` would silently truncate rather than reject) and a zero `cell`.
fn check_layout(image: &Tensor, cell: usize) -> Result<()> {
    let dims = image.dims();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != PREVIEW_LATENT_CHANNELS {
        return Err(CandleError::Msg(format!(
            "SenseNova preview state must have shape [1, {PREVIEW_LATENT_CHANNELS}, H, W], got {dims:?}"
        )));
    }
    if cell == 0 || !dims[2].is_multiple_of(cell) || !dims[3].is_multiple_of(cell) {
        return Err(CandleError::Msg(format!(
            "SenseNova preview state {}x{} must be a multiple of the {cell}-pixel token cell",
            dims[2], dims[3]
        )));
    }
    Ok(())
}

/// The preview hook [`crate::T2iModel::generate`] threads into its bespoke denoise loop: a projector
/// closure over [`project_running_image`] carrying this model's own token cell.
///
/// `cell` is bound **once**, at the single construction site in this crate's generator, from
/// [`crate::T2iModel::cell`] — the same accessor `denoise` derives its token grid from — so the
/// projector cannot come to disagree with the loop about how large a token is.
///
/// [`candle_gen::preview::PreviewHook::new`] rather than `with_sigma`: this loop walks an ascending
/// `t` boundary grid and applies no input scaling to the state it advances, so the running image
/// already *is* the tensor the fit was measured against. It is keyed on the step index
/// ([`candle_gen::preview::PreviewCounter::with_steps`]) for the same reason — there is no descending
/// σ array to index into.
///
/// Build it **per request**: `candle_gen::for_each_image_seed` runs one `generate` per seed and each
/// must start a fresh trajectory at frame 1, which the per-call counter inside `denoise` already
/// guarantees.
pub(crate) fn t2i_hook(sink: &PreviewSink, cell: usize) -> PreviewHook<'_> {
    PreviewHook::new(sink, move |image: &Tensor| {
        project_running_image(image, cell)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use candle_gen::candle_core::Device;
    use candle_gen::gen_core::PreviewFrame;
    use candle_gen::preview::PreviewCounter;

    use super::*;

    /// The shipped 8B-MoT token cell (`patch_size 16 · merge_size 2`), used by the rows below so they
    /// exercise the real geometry rather than a convenient small one.
    const CELL: usize = 32;

    fn zeros(shape: (usize, usize, usize, usize)) -> Tensor {
        Tensor::zeros(shape, DType::F32, &Device::Cpu).expect("zeros")
    }

    /// A deterministic non-constant state — a fit applied to zeros returns only its bias, which would
    /// let a collapsed factor table look correct.
    fn ramp(channels: usize, h: usize, w: usize) -> Tensor {
        let n = channels * h * w;
        let values: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.37).sin() * 0.9 + ((i % 7) as f32) * 0.05)
            .collect();
        Tensor::from_vec(values, (1, channels, h, w), &Device::Cpu).expect("ramp")
    }

    fn collecting_sink() -> (PreviewSink, Arc<Mutex<Vec<PreviewFrame>>>) {
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| candle_gen::lock_recover(&captured).push(frame));
        (sink, frames)
    }

    // ── The fit ───────────────────────────────────────────────────────────────────────────────────

    /// One finite row per model-space channel, and the channel count is the pixel-space three rather
    /// than any VAE width.
    #[test]
    fn the_fit_has_one_finite_row_per_pixel_channel() {
        assert_eq!(PREVIEW_LATENT_CHANNELS, 3);
        assert!(RGB_FACTORS.iter().flatten().all(|v| v.is_finite()));
        assert!(RGB_BIAS.iter().all(|v| v.is_finite()));
        for width in [4usize, 16, 32] {
            assert_ne!(
                PREVIEW_LATENT_CHANNELS, width,
                "a {width}-channel fit would be an epic-16624 VAE latent space; SenseNova has no VAE"
            );
        }
    }

    /// **The story's insurance test.** The measured fit must land on the analytic decode transform
    /// `x·0.5 + 0.5`, coefficient by coefficient.
    ///
    /// This is what makes a near-unity R² falsifiable rather than self-congratulatory. A fit measured
    /// against the wrong target — an unclamped decode, a differently pooled one, a transposed row —
    /// would still report a high R² against *its own* target while sitting visibly off the transform
    /// `crate::t2i::tensor_to_image` actually applies. Every one of the twelve coefficients is
    /// checked, so a single wrong entry cannot hide behind eleven right ones.
    #[test]
    fn the_measured_fit_lands_on_the_analytic_decode_transform() {
        for (row, factors) in RGB_FACTORS.iter().enumerate() {
            for (column, value) in factors.iter().enumerate() {
                let expected = if row == column { ANALYTIC_GAIN } else { 0.0 };
                assert!(
                    (value - expected).abs() <= ANALYTIC_TOLERANCE,
                    "RGB_FACTORS[{row}][{column}] = {value} is more than {ANALYTIC_TOLERANCE} from \
                     the analytic decode transform's {expected}"
                );
            }
        }
        for (channel, value) in RGB_BIAS.iter().enumerate() {
            assert!(
                (value - ANALYTIC_BIAS).abs() <= ANALYTIC_TOLERANCE,
                "RGB_BIAS[{channel}] = {value} is more than {ANALYTIC_TOLERANCE} from the analytic \
                 intercept {ANALYTIC_BIAS}"
            );
        }
    }

    /// The fit is not an epic-16624 family's intercept wearing three rows. Each of the seven committed
    /// biases is a *warm* grey a little under 0.5; SenseNova's is the exact mid-grey the analytic
    /// transform demands, and that difference is the point.
    #[test]
    fn the_intercept_is_the_analytic_mid_grey_not_a_borrowed_one() {
        for borrowed in [
            [0.440_938_92f32, 0.424_318_4, 0.409_667_16], // candle-gen-flux2, 32ch
            [0.467_3, 0.437_615_22, 0.414_471_12],        // candle-gen-sana base, 32ch
        ] {
            assert_ne!(
                RGB_BIAS, borrowed,
                "this is another family's VAE-space intercept — SenseNova's space is pixel space"
            );
        }
        assert!(RGB_BIAS.iter().all(|v| (v - 0.5).abs() < 0.01));
    }

    // ── The pool and the layout contract ──────────────────────────────────────────────────────────

    /// The pool is the plain **box average** over one token cell — the same average the fit's target
    /// is built with — proven against a hand-computed cell rather than against `avg_pool2d`'s docs.
    ///
    /// Stated as an identity between two calls of the shipped projector rather than as hard-coded
    /// pixel values, so it measures the *pool* rather than re-encoding the committed coefficients: a
    /// 2×2 cell projected at `cell = 2` must equal its hand-computed mean projected at `cell = 1`.
    /// A max-pool, a stride-2 subsample, or a differently weighted kernel all break it.
    #[test]
    fn the_pool_is_the_cell_box_average_the_fit_target_uses() {
        // One 2×2 cell per channel: R = mean(0, 1, 0, 1) = 0.5, G = mean(−1,−1,−1,−1) = −1,
        // B = mean(0.25, 0.75, −0.5, 0.5) = 0.25, in MODEL space.
        let cell_values = vec![
            0.0f32, 1.0, 0.0, 1.0, // R
            -1.0, -1.0, -1.0, -1.0, // G
            0.25, 0.75, -0.5, 0.5, // B
        ];
        let state = Tensor::from_vec(cell_values, (1, 3, 2, 2), &Device::Cpu).expect("state");
        let pooled = project_running_image(&state, 2).expect("pooled projection");
        assert_eq!((pooled.width, pooled.height), (1, 1));

        let means = Tensor::from_vec(vec![0.5f32, -1.0, 0.25], (1, 3, 1, 1), &Device::Cpu)
            .expect("hand-computed cell means");
        let direct = project_running_image(&means, 1).expect("direct projection");
        assert_eq!(
            pooled.pixels, direct.pixels,
            "the token-cell pool must be the plain box average — a subsample or a max-pool would \
             disagree here"
        );

        // And the frame really is the decode of that mean: the analytic transform puts R and B on
        // the bright side of mid-grey and G hard against the dark rail.
        assert!(pooled.pixels[0] > 180 && pooled.pixels[2] > 130 && pooled.pixels[1] < 10);
    }

    /// **How far the committed fit sits from the model's own decode, in RGB8 levels** — computed from
    /// the coefficients rather than asserted in prose.
    ///
    /// Over the model's own `[-1, 1]` output range the worst-case difference between
    /// `RGB_FACTORS · x + RGB_BIAS` and the analytic `x·0.5 + 0.5` is
    /// `Σ_j |M[j][c] − 0.5·δ_jc| + |b_c − 0.5|`, because the extreme is attained at a corner of the
    /// cube. That is the entire visual cost of shipping the measured fit instead of the analytic map,
    /// and it is **under 1.2 RGB8 levels** on every channel.
    ///
    /// The bound is what makes the near-unity R² concrete: a preview frame is within a code value or
    /// two of the picture the model would have produced from the same pooled state.
    #[test]
    fn the_committed_fit_is_within_two_rgb8_levels_of_the_models_own_decode() {
        for channel in 0..3 {
            let mut worst = (RGB_BIAS[channel] - ANALYTIC_BIAS).abs() as f64;
            for (row, factors) in RGB_FACTORS.iter().enumerate() {
                let expected = if row == channel { ANALYTIC_GAIN } else { 0.0 };
                worst += (factors[channel] - expected).abs() as f64;
            }
            let levels = worst * 255.0;
            assert!(
                levels < 1.2,
                "channel {channel}: the committed fit differs from the analytic decode by up to \
                 {levels:.4} RGB8 levels over the model's own [-1, 1] range"
            );
        }
    }

    /// A render previews at its **token grid**, `H/cell × W/cell`, for the shipped 32-pixel cell — the
    /// shape claim the story asked to be confirmed at the emission point rather than assumed.
    #[test]
    fn the_shipped_resolutions_preview_at_the_token_grid() {
        for (h, w) in [(256usize, 256usize), (512, 512), (1024, 1024), (1152, 2048)] {
            let image = project_running_image(&zeros((1, 3, h, w)), CELL).expect("projection");
            assert_eq!(
                (image.width, image.height),
                ((w / CELL) as u32, (h / CELL) as u32)
            );
            assert_eq!(image.pixels.len(), (w / CELL) * (h / CELL) * 3);
        }
    }

    /// Anything that is not one batch-1 three-channel state on this render's grid is refused, and the
    /// message says SenseNova. The 4/16/32-channel rows are the ones that matter: a copy-paste from
    /// any epic-16624 family would produce exactly those, and they must not project.
    #[test]
    fn a_state_that_is_not_this_family_is_refused() {
        let packed = Tensor::zeros((1, 256, 128), DType::F32, &Device::Cpu).expect("packed");
        let five_d = Tensor::zeros((1, 3, 1, 64, 64), DType::F32, &Device::Cpu).expect("5-D");
        for bad in [
            zeros((2, 3, 64, 64)),  // batched
            zeros((1, 4, 64, 64)),  // SDXL's channel width
            zeros((1, 16, 64, 64)), // FLUX.1 / SD3.5 / Z-Image's
            zeros((1, 32, 64, 64)), // FLUX.2 / Lens / Ideogram / SANA's
            packed,
            five_d,
        ] {
            let error = project_running_image(&bad, CELL)
                .expect_err("must refuse")
                .to_string();
            assert!(
                error.contains("SenseNova preview state must have shape [1, 3, H, W]"),
                "got: {error}"
            );
        }

        // A grid the cell does not divide is refused rather than silently truncated by the pool, and
        // so is a zero cell.
        for (state, cell) in [
            (zeros((1, 3, 64, 48)), 32usize),
            (zeros((1, 3, 48, 64)), 32),
            (zeros((1, 3, 64, 64)), 0),
        ] {
            let error = project_running_image(&state, cell)
                .expect_err("must refuse")
                .to_string();
            assert!(error.contains("token cell"), "got: {error}");
        }
    }

    /// The projection casts up front, so a narrow-dtype state pools and projects rather than panicking
    /// — the candle dtype trap `candle_gen::preview::project_latents` documents, exercised through the
    /// pool this module adds in front of it.
    #[test]
    fn a_low_precision_state_pools_and_projects() {
        for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
            let state = zeros((1, 3, 64, 64)).to_dtype(dtype).expect("cast");
            let image = project_running_image(&state, CELL)
                .unwrap_or_else(|e| panic!("{dtype:?} state failed to project: {e}"));
            assert_eq!((image.width, image.height), (2, 2));
        }
    }

    // ── The hook ──────────────────────────────────────────────────────────────────────────────────

    /// The hook is step-index keyed: one frame per outer step, numbered `1..=steps` over `total ==
    /// steps`, with a repeated index emitting nothing.
    #[test]
    fn the_hook_numbers_frames_by_step_index() {
        let (sink, frames) = collecting_sink();
        let hook = t2i_hook(&sink, CELL);
        let counter = PreviewCounter::with_steps(4);
        let state = zeros((1, 3, 64, 64));

        for step in 0..4 {
            hook.emit_step(&counter, step, &state);
        }
        hook.emit_step(&counter, 3, &state); // a repeat

        let frames = candle_gen::lock_recover(&frames);
        assert_eq!(
            frames
                .iter()
                .map(|f| (f.current, f.total))
                .collect::<Vec<_>>(),
            (1..=4).map(|n| (n, 4)).collect::<Vec<_>>()
        );
    }

    /// A single-step render emits exactly one frame — the degenerate schedule an `n - 1` denominator
    /// would divide by zero on. `steps == 1` is reachable: `validate` rejects only `steps == 0`.
    #[test]
    fn a_single_step_render_emits_exactly_one_frame() {
        let (sink, frames) = collecting_sink();
        let hook = t2i_hook(&sink, CELL);
        let counter = PreviewCounter::with_steps(1);
        assert_eq!(counter.total(), 1);
        hook.emit_step(&counter, 0, &zeros((1, 3, 64, 64)));
        hook.emit_step(&counter, 0, &zeros((1, 3, 64, 64)));
        assert_eq!(candle_gen::lock_recover(&frames).len(), 1);
    }

    /// The hook carries the **cell it was built with**, so a projector wired to the wrong token size
    /// is visible here rather than only in a render.
    #[test]
    fn the_hook_projects_at_the_cell_it_was_built_with() {
        let state = ramp(3, 64, 64);
        for (cell, edge) in [(32usize, 2u32), (16, 4), (8, 8)] {
            let (sink, frames) = collecting_sink();
            t2i_hook(&sink, cell).emit_step(&PreviewCounter::with_steps(1), 0, &state);
            let frames = candle_gen::lock_recover(&frames);
            assert_eq!(frames.len(), 1);
            assert_eq!(
                (frames[0].image.width, frames[0].image.height),
                (edge, edge)
            );
        }
    }

    /// A malformed state loses exactly one decorative frame and consumes its schedule position; the
    /// render is unaffected and the failure is swallowed by the shared emitter, never surfaced.
    #[test]
    fn a_malformed_state_loses_one_frame_and_consumes_its_position() {
        let (sink, frames) = collecting_sink();
        let hook = t2i_hook(&sink, CELL);
        let counter = PreviewCounter::with_steps(2);
        hook.emit_step(&counter, 0, &zeros((1, 16, 64, 64)));
        hook.emit_step(&counter, 1, &zeros((1, 3, 64, 64)));
        let frames = candle_gen::lock_recover(&frames);
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].current, frames[0].total), (2, 2));
    }

    /// An inert sink does no tensor work at all and does not advance the counter — the property that
    /// makes an unwatched render byte-identical to a pre-sc-16960 one.
    #[test]
    fn an_inert_sink_projects_nothing_and_advances_nothing() {
        let inert = PreviewSink::default();
        let hook = t2i_hook(&inert, CELL);
        assert!(!hook.is_active());
        let counter = PreviewCounter::with_steps(2);
        // A state that would fail the projection if it were ever reached is not reached.
        hook.emit_step(&counter, 0, &zeros((1, 16, 64, 64)));
        assert_eq!(counter.next_step(0), Some(1));
    }

    // ── The wiring, pinned against this crate's own source ────────────────────────────────────────

    /// The shipped half of a source file — everything ahead of its FIRST `#[cfg(test)]` item.
    ///
    /// The structural rows in those modules drive the hook over inert sinks and construct
    /// `GenerationRequest` literals, so a scan that counted them would read test code as shipped
    /// wiring. The expected marker count is asserted so a new test module has to be acknowledged here
    /// rather than silently changing what "shipped" means.
    fn shipped(source: &'static str, name: &str, markers: usize) -> &'static str {
        const MARKER: &str = "#[cfg(test)]";
        // Anchored at line start rather than matched with surrounding newlines: these sources are
        // checked out CRLF on Windows, and a `\n…\n` needle silently matches nothing there — which
        // would hand every scan below an EMPTY shipped half and pass every count vacuously.
        let starts: Vec<usize> = source
            .match_indices(MARKER)
            .filter(|(at, _)| *at == 0 || source[..*at].ends_with('\n'))
            .map(|(at, _)| at)
            .collect();
        assert_eq!(
            starts.len(),
            markers,
            "{name} must hold exactly {markers} `#[cfg(test)]` item(s) for this split to be sound — \
             teach this scan about the new one rather than letting it read test code as shipped code"
        );
        let shipped = &source[..starts[0]];
        assert!(
            !shipped.is_empty(),
            "{name}'s shipped half is empty — every count below would pass vacuously"
        );
        shipped
    }

    fn shipped_lib() -> &'static str {
        shipped(include_str!("lib.rs"), "lib.rs", 2)
    }

    fn shipped_t2i() -> &'static str {
        shipped(include_str!("t2i.rs"), "t2i.rs", 1)
    }

    /// Read the `preview:` parameter out of a function's own declaration, so a pin cannot be satisfied
    /// by the spelling appearing somewhere else in the file.
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

    /// The exact declaration every hop between the request's sink and the bespoke loop must carry.
    const WANT: &str = "preview: &PreviewHook<'_>,";

    /// Count [`WANT`] by **whole trimmed line**, not by substring.
    ///
    /// A substring tally is satisfied by the same declaration renamed `_preview:` — the spelling a hop
    /// takes the moment it stops *using* its hook — so it would let a hop become an ignored parameter
    /// and still count toward the total.
    fn hook_parameters(source: &str) -> usize {
        source.lines().filter(|line| line.trim() == WANT).count()
    }

    /// **The whole hook path, guarded.** The registered lane builds its hook over the *request's* sink
    /// and every hop between that sink and the emission carries it as a non-`Option` reference.
    ///
    /// Both halves are load-bearing and neither implies the other. `candle-gen-catalog`'s
    /// `preview_advertising` inventory can only see that `t2i.rs` makes one direct emission call — it
    /// cannot see what sink reached it, and there is no shared-driver argument to classify because
    /// this crate drives no shared sampler at all. sc-16958 and sc-16959 each showed a reviewer taking
    /// a family's lanes dark with the full CPU suite green:
    ///
    /// * a hop that **accepts and then ignores** its forwarded hook and builds a fresh
    ///   `PreviewHook::new(&inert, …)` — every parameter still reads `&PreviewHook`, and
    ///   `PreviewHook::new(` does not contain `_hook(`, so a constructor-call tally cannot see it;
    /// * a `generate` that rebinds `let req = &GenerationRequest { preview: PreviewSink::default(),
    ///   ..req.clone() };` ahead of the hook build — the literal a scan looks for is still there,
    ///   exactly once, over a sink that has been emptied.
    ///
    /// So shipped `lib.rs` and shipped `t2i.rs` are pinned to **zero** `PreviewHook::new` and **zero**
    /// `GenerationRequest {` between them, and the parameters are counted by whole line.
    #[test]
    fn the_registered_lane_builds_its_hook_from_the_requests_sink() {
        let lib = shipped_lib();
        let t2i = shipped_t2i();

        // The sink: exactly one hook, over the REQUEST's sink, carrying the MODEL's own token cell.
        assert_eq!(
            lib.matches("preview::t2i_hook(&req.preview, comps.model.cell())")
                .count(),
            1,
            "the registered lane must build exactly one hook, over the request's sink, with the \
             cell read from the very model whose denoise loop the frames come from"
        );
        assert_eq!(
            lib.matches("_hook(").count(),
            1,
            "shipped lib.rs must build exactly one preview hook — a second render lane must be named \
             in this crate's inventory (and in the catalog's) rather than appearing here"
        );

        // Neither shipped file may CONSTRUCT a hook or a request: the two darkening edits above.
        for (name, source) in [("lib.rs", lib), ("t2i.rs", t2i)] {
            assert_eq!(
                source.matches("PreviewHook::new").count(),
                0,
                "shipped {name} must never CONSTRUCT a hook — the sink is reached only through \
                 `preview::t2i_hook`, whose single call site is counted above. A \
                 `PreviewHook::new(&inert, …)` in a hop that accepts and then ignores its forwarded \
                 hook takes the lane dark with no type error"
            );
            assert_eq!(
                source.matches("GenerationRequest {").count(),
                0,
                "shipped {name} must never CONSTRUCT a GenerationRequest — it reads the caller's. A \
                 rebind that swaps `preview` for an inert sink empties it while leaving the \
                 `t2i_hook(&req.preview, …)` literal intact, which is all the count above checks"
            );
        }

        // Every hop takes the hook by non-`Option` reference, counted by whole line.
        for declaration in ["    pub fn generate(", "    fn denoise("] {
            assert_eq!(
                preview_parameter(t2i, declaration),
                WANT,
                "{declaration} must take its hook by reference. An `Option` here is blankable at the \
                 caller, and this crate has no shared-driver argument for the catalog's route \
                 inventory to classify instead"
            );
        }
        assert_eq!(
            hook_parameters(t2i),
            2,
            "`generate` and `denoise` must both take `&PreviewHook` under that exact name — a hop \
             renamed `_preview:` is one that no longer uses it"
        );
    }

    /// The bespoke loop emits **exactly once**, keyed on the loop's own 0-based step index, over a
    /// counter built from the very step count the loop iterates and `Progress::Step` reports.
    ///
    /// This crate drives no shared sampler, so there is no call-site argument to inspect; the
    /// source-level fact available here is that the loop emits once, in the shape the catalog's
    /// `Denoise::Bespoke` row declares. The needle is assembled at compile time so this scan does not
    /// match its own source.
    #[test]
    fn the_bespoke_denoise_loop_emits_exactly_once_per_step() {
        let t2i = shipped_t2i();
        assert_eq!(
            t2i.matches(concat!(".emit", "_step(")).count(),
            1,
            "the bespoke loop must make exactly one direct emission call"
        );
        assert!(
            t2i.contains("preview.emit_step(&preview_counter, i, &image);"),
            "the emission must key on the loop's own 0-based index and forward the running state"
        );
        assert!(
            t2i.contains(concat!("PreviewCounter::", "with_steps(steps)")),
            "the counter must be step-index keyed over the loop's own step count"
        );
        // The it2i / interleave loop is the crate's OTHER denoise and is deliberately unwired.
        let it2i = t2i
            .find("fn it2i_denoise(")
            .expect("the understanding-surface loop must still exist");
        assert!(
            !t2i[it2i..].contains("preview"),
            "the it2i / interleave loop is the off-registry understanding surface and is out of \
             scope for sc-16960 — it must emit nothing"
        );
        // No shared sampler anywhere in the crate: the fact that makes this crate Denoise::Bespoke.
        for (file, code) in [
            ("t2i.rs", include_str!("t2i.rs")),
            ("lib.rs", include_str!("lib.rs")),
            ("fm.rs", include_str!("fm.rs")),
            ("qwen3.rs", include_str!("qwen3.rs")),
            ("runtime.rs", include_str!("runtime.rs")),
        ] {
            for driver in [
                concat!("run_flow_", "sampler("),
                concat!("run_curated_", "sampler("),
                concat!("run_scm_", "sampler("),
            ] {
                assert!(
                    !code.contains(driver),
                    "{file} drives {driver} — this crate is declared Denoise::Bespoke"
                );
            }
        }
    }

    /// Both registered SenseNova routes advertise the flag. Weights-free: descriptors only.
    #[test]
    fn both_registered_sensenova_routes_advertise_preview_support() {
        for descriptor in [crate::descriptor(), crate::descriptor_fast()] {
            assert!(
                descriptor.capabilities.supports_preview,
                "{} must advertise preview support",
                descriptor.id
            );
        }
    }
}
