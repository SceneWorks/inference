//! The FLUX.2-family per-step latent preview seam (epic 16948, sc-16955; the MLX original is epic
//! 16624 / `mlx-gen-flux2/src/preview.rs`).
//!
//! This module owns the **32-channel** FLUX.2 fit — the widest latent space in the epic and the
//! largest single unlock: `candle-gen-lens` and `candle-gen-ideogram` both reuse these coefficients
//! through [`project_raw_latents`] rather than restating them, because both load a VAE whose learned
//! tensors are the ones this fit was measured over (below). Schedule numbering, multi-eval dedup and
//! the swallow-on-failure contract all live in [`candle_gen::preview`], shared by every candle family.
//!
//! ## The patch-major trap: WHERE the projection runs is the whole story
//!
//! FLUX.2 does not denoise the tensor its fit is defined over. The transformer works in a
//! **BatchNorm-normalized, 2×2-patchified, 128-channel** space, and the raw 32-channel VAE latent
//! only exists on the far side of two VAE-owned transforms. So the running latent goes through three
//! shapes before it is projectable, and every one of them would silently produce a picture rather
//! than an error if projected in the wrong place:
//!
//! | stage | shape | projectable? |
//! | --- | --- | --- |
//! | the sampler's running latent | `[1, (H/16)·(W/16), 128]` packed tokens | no — rank 3 |
//! | after [`crate::pipeline::unpack_latents_at`] | `[1, 128, H/16, W/16]` | **no — 128 "channels" at half resolution, and still bn-normalized** |
//! | after [`crate::vae::Flux2Vae::raw_latent_from_packed`] | `[1, 32, H/8, W/8]` | yes — the fitted space |
//!
//! The middle row is the trap `mlx-gen-flux2/src/preview.rs` names: it is rank 4 with a plausible
//! channel count, so [`candle_gen::preview::project_latents`] would accept it if handed a 128-row
//! factor table, and a 32-row table simply rejects it. [`project_raw_latents`] therefore pins the
//! channel count explicitly, and the packed seam runs the *VAE's own* de-normalize + unpatchify —
//! [`crate::vae::Flux2Vae::raw_latent_from_packed`], the exact function
//! [`crate::vae::Flux2Vae::decode_packed`] calls — rather than a second copy of those transforms. The
//! preview's geometry and the decode's geometry are one function, not two agreeing implementations.
//!
//! Ideogram is the exception that proves the seam is VAE-owned rather than pipeline-owned: its DiT
//! packs the same 128 channels in a **`(ph, pw, c)`** patch-major order instead of FLUX.2's
//! `(c, ph, pw)`, so it de-normalizes and unpatchifies in its own crate (against the same
//! `bn_stats()`) and reaches this module only at [`project_raw_latents`], with a raw 32-channel latent
//! already in hand.
//!
//! ## What the hook sees on each route
//!
//! All three shipped FLUX.2 routes drive [`candle_gen::run_flow_sampler`], so all three opt in with a
//! projector closure rather than by restructuring a loop:
//!
//! * **CFG never reaches the preview.** klein's true-CFG negative pass runs *inside* the predict
//!   closure and returns one blended velocity (`vn + guidance·(v − vn)`); dev is guidance-distilled
//!   and runs a single forward. No fused `[2, …]` batch ever becomes the running latent.
//! * **Edit reference tokens never reach the preview.** `crate::edit_provider` concatenates the
//!   VAE-encoded references onto the **sequence** axis inside the closure
//!   (`Tensor::cat(&[latents, &ref_tokens], 1)`) and slices the forward's result back to the leading
//!   `target_seq` velocity tokens, so the sampler's latent stays the target grid alone. That is the
//!   hazard that makes an edit preview show the reference picture, and it is closed structurally.
//! * **The control branch never reaches the preview.** `crate::control_provider` keeps its encoded
//!   control latent in a closure capture, constant across steps and never part of the running latent.
//!
//! ## The σ convention: this family needs no correction
//!
//! [`candle_gen::run_flow_sampler`] integrates a [`candle_gen::gen_core::sampling::FlowModelSampling`],
//! whose `input_scale` is exactly `1.0` at every σ, so the running latent already *is* the tensor the
//! fit was measured against and the σ-less [`candle_gen::preview::PreviewHook::new`] constructor is
//! the correct one. This is the property sc-16954 found to be **false** for the discrete ε cohort
//! (SDXL/Kolors denoise in k-diffusion VE σ-space and must apply `1/√(σ²+1)` before projecting); it is
//! stated here rather than assumed, and `tests/preview_real_weights.rs` measures the first frame's
//! rail-clipped fraction to show the uncorrected projection is already readable.
//!
//! ## The fit is reused, not refitted — grounded in tensor bytes
//!
//! `RGB_FACTORS` / `RGB_BIAS` are the epic-16624 constants transcribed verbatim from
//! `mlx-gen-flux2/src/preview.rs`. They are least-squares numbers over a VAE *latent space* with no
//! backend in them; candle reuses them and deliberately ships **no producer** of its own —
//! `mlx-gen-flux2/tests/fit_preview_rgb.rs` remains the only way they are re-derived.
//!
//! The reuse is grounded in the bytes each family actually loads, not in a shared Rust type. Two
//! containers publish **one** learned 32-channel `AutoencoderKLFlux2`:
//!
//! * bf16, SHA-256 `ca70d220…0f04`, 168,120,878 bytes — `black-forest-labs/FLUX.2-klein-9B`, and
//!   every tier of the `SceneWorks/flux2-klein-9b-mlx` / `flux2-klein-9b-kv-mlx` re-hosts;
//! * f32, SHA-256 `d64f3a68…c4b5`, 336,213,556 bytes — `black-forest-labs/FLUX.2-dev`,
//!   `SceneWorks/Lens`, `Comfy-Org/Lens`, and every tier of the `flux2-dev-mlx` / `lens-mlx` /
//!   `lens-turbo-mlx` re-hosts.
//!
//! All **250** learned tensors (84,046,371 values) of the f32 container round, exactly and
//! round-to-nearest-even, onto the bf16 container's bits: two widths of one checkpoint, not two
//! fine-tunes. Ideogram ships a third and fourth *file* whose 250 learned tensors are **byte-identical**
//! to the bf16 container — only the unused `bn.num_batches_tracked` counter differs, in value and (for
//! the packed re-host) in integer dtype. `tests/preview_real_weights.rs` re-derives all of that per
//! snapshot; the full record is `docs/migration/evidence/sc-16955-flux2-candle-preview.md`.
//!
//! `candle-gen-boogu` is deliberately **not** wired here: it loads a plain 244-tensor 16-channel
//! `AutoencoderKL` (the FLUX.1 / Z-Image lineage, `z_image::vae::AutoEncoderKL`) with no `bn.*` stats
//! at all, so this 32-channel fit does not describe its latent space. See the adjudication in the
//! evidence document.
//!
//! A stale or absent fit degrades preview colour only; the denoise path never reads these constants.

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::{Image, PreviewSink};
use candle_gen::preview::PreviewHook;
use candle_gen::{CandleError, Result};

use crate::vae::Flux2Vae;

/// Ordinary-least-squares map from the raw 32-channel FLUX.2 VAE latent to latent-resolution RGB
/// (row *i* maps latent channel *i* to `[r, g, b]`), with [`RGB_BIAS`] the intercept.
///
/// **Reused verbatim from `mlx-gen-flux2/src/preview.rs:27`, not refitted.** Fit on eight diverse
/// real-weight FLUX.2 Klein renders and evaluated on four disjoint prompt/seed holdouts, all 256²
/// with eight flow-Euler steps, against 8×8-average-pooled native VAE decodes. Fit R²
/// `(R,G,B) = (0.77472, 0.77362, 0.74224)`, overall `0.76409`; holdout R²
/// `(0.64247, 0.71049, 0.72381)`, overall `0.69504`. The transform preserves coarse palette and
/// composition and intentionally omits decoder detail.
///
/// Refit — in `mlx-gen-flux2`, never here — whenever the FLUX.2 VAE lineage or the bn-stats
/// normalization changes.
const RGB_FACTORS: [[f32; 3]; 32] = [
    [0.004_606_732, 0.000_919_550, 0.002_941_766],
    [-0.004_998_811, -0.001_859_922, 0.012_573_381],
    [-0.002_657_903, 0.006_393_426, 0.017_933_674],
    [0.041_584_67, 0.071_578_98, 0.086_347_34],
    [-0.003_628_706, -0.003_508_693, -0.006_987_334],
    [0.000_882_318, 0.001_083_956, -0.006_031_806],
    [-0.013_952_778, 0.022_036_728, -0.009_708_006],
    [0.000_767_323, 0.008_682_552, -0.000_544_764],
    [-0.088_695_09, -0.062_523_74, -0.037_046_182],
    [-0.001_179_61, -0.005_790_229, -0.008_324_275],
    [-0.013_809_573, -0.004_775_996, -0.004_229_145],
    [0.028_518_197, 0.002_239_314, -0.029_317_595],
    [-0.000_851_189, 0.003_684_022, 0.010_735_653],
    [0.009_056_681, 0.011_084_957, 0.008_661_977],
    [0.009_438_053, 0.012_882_866, 0.019_065_392],
    [-0.024_013_02, 0.003_596_907, 0.007_431_358],
    [0.003_629_937, 0.009_398_92, 0.007_673_813],
    [0.002_841_817, 0.004_690_581, 0.000_741_530],
    [0.005_932_563, 0.005_661_016, -0.003_041_417],
    [0.029_337_115, -0.007_673_972, -0.010_016_148],
    [-0.000_704_694, -0.004_005_695, -0.003_698_119],
    [0.006_647_336, 0.005_498_843, 0.005_949_094],
    [-0.004_770_506, -0.005_373_847, -0.006_334_095],
    [-0.004_414_479, -0.008_698_589, -0.004_114_577],
    [0.000_000_255, 0.014_090_918, 0.012_625_638],
    [-0.004_015_625, 0.000_087_857, 0.000_038_551],
    [0.005_745_834, -0.004_168_361, -0.002_627_784],
    [0.002_873_869, 0.006_991_146, 0.003_682_432],
    [-0.014_361_657, -0.015_511_271, -0.010_613_72],
    [-0.020_403_308, 0.001_543_391, 0.007_434_33],
    [0.000_384_687, 0.002_038_027, -0.000_277_807],
    [0.014_301_192, 0.004_776_081, 0.000_665_965],
];

/// The fit's intercept — the near-neutral grey a fully-zero latent projects to. Reused with
/// [`RGB_FACTORS`].
const RGB_BIAS: [f32; 3] = [0.440_938_92, 0.424_318_4, 0.409_667_16];

/// The FLUX.2-family latent channel count the fit is defined over. Derived from the committed factor
/// table's own length, so a consumer (Lens, Ideogram) cannot drift from it by restating a number.
pub const PREVIEW_LATENT_CHANNELS: usize = RGB_FACTORS.len();

/// The transformer's packed channel count — 32 latent channels × the 2×2 patch. Named because it is
/// the number that makes the middle row of the module docs' table dangerous: a packed latent is
/// rank 4 and plausible-looking, and only the channel count tells it apart from the fitted space.
pub const PACKED_LATENT_CHANNELS: usize = PREVIEW_LATENT_CHANNELS * 4;

/// The fit is the THIRTY-TWO-channel one, and the packed space is four times wider. Compile-time,
/// because a runtime row over constants proves nothing a `const` assertion does not prove earlier.
const _: () = assert!(PREVIEW_LATENT_CHANNELS == 32 && PACKED_LATENT_CHANNELS == 128);

/// Project a **raw 32-channel** FLUX.2-family latent `[1, 32, h, w]` to a latent-resolution RGB8
/// preview.
///
/// This is the reuse seam `candle-gen-lens` and `candle-gen-ideogram` call: both share this latent
/// space (one learned `AutoencoderKLFlux2`, proven in tensor bytes — see the module docs) and
/// therefore these coefficients, and calling through here is what keeps a second copy of them from
/// existing.
///
/// "Raw" means **after** the bn de-normalize and the 2×2 unpatchify, i.e. the tensor
/// [`crate::vae::Flux2Vae::raw_latent_from_packed`] returns and
/// [`crate::vae::Flux2Vae::decode_latent`] consumes. A packed transformer latent must go through
/// [`project_packed_tokens`] instead.
///
/// Errors on any layout that is not one batch-1 thirty-two-channel spatial latent — notably on the
/// 128-channel packed grid, which is the failure mode this check exists for. The caller's frame is
/// then lost and swallowed by `candle_gen::preview::emit_preview`, the intended decorative-failure
/// behaviour.
pub fn project_raw_latents(latents: &Tensor) -> Result<Image> {
    check_raw_layout(latents)?;
    candle_gen::preview::project_latents(latents, &RGB_FACTORS, RGB_BIAS)
}

/// Project the sampler's **packed token** latent `[1, lat_h·lat_w, 128]` by running the whole
/// VAE-owned recovery first: unpack the token sequence onto the `[1, 128, lat_h, lat_w]` grid, then bn
/// de-normalize and 2×2 unpatchify it into the raw `[1, 32, 2·lat_h, 2·lat_w]` latent, then project.
///
/// `(lat_h, lat_w)` is the **token** grid — `(H/16, W/16)` for FLUX.2, and Lens's own aspect-bucket
/// grid for Lens. It is the same pair the route hands its decode tail, which is what keeps the
/// preview's geometry and the render's geometry from diverging.
///
/// `bn_std` / `bn_mean` come from [`crate::vae::Flux2Vae::bn_stats`], and the recovery itself is
/// [`crate::vae::raw_latent_from_packed`] — the exact function
/// [`crate::vae::Flux2Vae::decode_packed`] calls — so there is one implementation of that seam and not
/// two. [`hook`] is the ergonomic form that reads the pair off a loaded VAE.
///
/// Errors — a latent that is not this route's packed shape, or a grid that does not describe it — are
/// what the shared emitter swallows to lose exactly one decorative frame.
pub fn project_packed_tokens(
    tokens: &Tensor,
    bn_std: &Tensor,
    bn_mean: &Tensor,
    lat_h: usize,
    lat_w: usize,
) -> Result<Image> {
    check_packed_layout(tokens, lat_h, lat_w)?;
    let grid = crate::pipeline::unpack_latents_at(tokens, lat_h, lat_w)?;
    project_raw_latents(&crate::vae::raw_latent_from_packed(&grid, bn_std, bn_mean)?)
}

/// Reject anything that is not one batch-1 latent in the fitted thirty-two-channel space.
///
/// The shared projection would reject most of these anyway, but naming the channel count here makes
/// the failure say *FLUX.2* — and catches the one case it cannot see, a rank-4 latent whose channel
/// count merely happens to match some other family's.
fn check_raw_layout(latents: &Tensor) -> Result<()> {
    let dims = latents.dims();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != PREVIEW_LATENT_CHANNELS {
        return Err(CandleError::Msg(format!(
            "flux2 preview latent must have shape [1, {PREVIEW_LATENT_CHANNELS}, h, w], got {dims:?}"
        )));
    }
    Ok(())
}

/// Reject a packed token sequence that is not this render's grid.
///
/// What this catches is a **sequence length** or channel width that does not describe the declared
/// grid — the realistic failure, which is a hook built for one render size and handed another's
/// trajectory. It deliberately does not claim more: a packed sequence carries no grid of its own, so
/// no check here could tell a `(2, 8)` grid from an `(8, 2)` one. That transposition is ruled out at
/// the call sites instead, by each route deriving the hook's grid from the same pair it hands its
/// decode tail.
///
/// Checked before the reshape rather than left to it so the message names the grid and the caller
/// gets one swallowed decorative frame rather than a raw candle shape error.
fn check_packed_layout(tokens: &Tensor, lat_h: usize, lat_w: usize) -> Result<()> {
    let dims = tokens.dims();
    if dims.len() != 3
        || dims[0] != 1
        || dims[1] != lat_h * lat_w
        || dims[2] != PACKED_LATENT_CHANNELS
    {
        return Err(CandleError::Msg(format!(
            "flux2 preview packed latent must have shape [1, {}, {PACKED_LATENT_CHANNELS}] for a \
             {lat_h}x{lat_w} token grid, got {dims:?}",
            lat_h * lat_w
        )));
    }
    Ok(())
}

/// The per-route preview hook every FLUX.2-family render route hands to
/// [`candle_gen::run_flow_sampler`]: a projector closure over [`project_packed_tokens`] bound to this
/// render's VAE and token grid. The driver owns frame numbering, multi-eval dedup, and the
/// swallow-on-failure contract (sc-16949), so no route restructures its loop.
///
/// Build it **per image**: a batched request runs one driver call per seed and each call must start a
/// fresh trajectory at frame 1. (The driver builds its own counter per call, so this is a property of
/// the call rather than of the hook — building the hook alongside the call keeps the two impossible to
/// separate.)
///
/// Public because `candle-gen-lens`' render lanes drive the same sampler over the same latent space
/// with the same [`Flux2Vae`], so they need this seam rather than one of their own. Ideogram does
/// **not** use it: it owns a bespoke denoise loop and emits through
/// `candle_gen::preview::emit_preview_at` directly.
pub fn hook<'a>(
    sink: &'a PreviewSink,
    vae: &'a Flux2Vae,
    lat_h: usize,
    lat_w: usize,
) -> PreviewHook<'a> {
    let (bn_std, bn_mean) = vae.bn_stats();
    PreviewHook::new(sink, move |tokens: &Tensor| {
        project_packed_tokens(tokens, bn_std, bn_mean, lat_h, lat_w)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use candle_gen::candle_core::{DType, Device};
    use candle_gen::gen_core::PreviewFrame;

    use super::*;

    fn zeros(shape: (usize, usize, usize, usize)) -> Tensor {
        Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap()
    }

    /// The fit is **reused**, not refitted: these are the epic-16624 constants transcribed verbatim
    /// from `mlx-gen-flux2/src/preview.rs`. Pinned as literals so an edit to either copy fails rather
    /// than silently forking one latent space into two colour maps.
    #[test]
    fn committed_fit_matches_the_mlx_source_block() {
        assert_eq!(RGB_FACTORS.len(), 32);
        assert_eq!(PREVIEW_LATENT_CHANNELS, 32);
        assert_eq!(
            RGB_FACTORS[0],
            [0.004_606_732, 0.000_919_550, 0.002_941_766]
        );
        assert_eq!(
            RGB_FACTORS[8],
            [-0.088_695_09, -0.062_523_74, -0.037_046_182]
        );
        assert_eq!(
            RGB_FACTORS[24],
            [0.000_000_255, 0.014_090_918, 0.012_625_638]
        );
        assert_eq!(
            RGB_FACTORS[31],
            [0.014_301_192, 0.004_776_081, 0.000_665_965]
        );
        assert_eq!(RGB_BIAS, [0.440_938_92, 0.424_318_4, 0.409_667_16]);
        assert!(RGB_FACTORS.iter().flatten().all(|v| v.is_finite()));
        assert!(RGB_BIAS.iter().all(|v| v.is_finite()));
    }

    /// A zero latent projects to the fit's intercept — the one place the committed bias is directly
    /// observable, so a typo in [`RGB_BIAS`] cannot pass.
    #[test]
    fn a_zero_latent_projects_to_the_fit_intercept() {
        let image = project_raw_latents(&zeros((1, 32, 2, 3))).unwrap();
        assert_eq!((image.width, image.height), (3, 2));
        // 0.44093892·255 = 112.4, 0.4243184·255 = 108.2, 0.40966716·255 = 104.5
        let expect: Vec<u8> = [112, 108, 104].repeat(6);
        assert_eq!(image.pixels, expect);
    }

    /// The packed grid is the trap: rank 4, plausible, and **wrong**. It must be rejected by the raw
    /// projector rather than projected at half resolution against a mismatched map.
    #[test]
    fn the_packed_grid_is_rejected_by_the_raw_projector() {
        for bad in [
            zeros((1, PACKED_LATENT_CHANNELS, 4, 4)),
            zeros((2, 32, 2, 3)),
            zeros((1, 16, 2, 3)),
        ] {
            let error = project_raw_latents(&bad).unwrap_err().to_string();
            assert!(
                error.contains("flux2 preview latent must have shape [1, 32, h, w]"),
                "unexpected error: {error}"
            );
        }
        let rank_three = Tensor::zeros((1, 16, 128), DType::F32, &Device::Cpu).unwrap();
        assert!(project_raw_latents(&rank_three).is_err());
    }

    /// The packed layout check runs before the reshape, so a hook built for a different render size
    /// loses one decorative frame with a grid-naming error instead of folding onto the wrong shape.
    ///
    /// The limit of the check is pinned in the same row rather than left to the doc comment: a
    /// transposed grid of the same token count is **not** detectable from the sequence alone, and the
    /// call sites are what rule it out.
    #[test]
    fn a_token_sequence_that_does_not_describe_the_grid_is_an_error() {
        let tokens = Tensor::zeros((1, 16, 128), DType::F32, &Device::Cpu).unwrap();
        assert!(check_packed_layout(&tokens, 4, 4).is_ok());

        for (h, w) in [(8usize, 4usize), (4, 5), (2, 3), (1, 15)] {
            let error = check_packed_layout(&tokens, h, w).unwrap_err().to_string();
            assert!(
                error.contains("token grid"),
                "a {h}x{w} grid must be rejected: {error}"
            );
        }

        // Wrong channel width and wrong rank are rejected too.
        let narrow = Tensor::zeros((1, 16, 64), DType::F32, &Device::Cpu).unwrap();
        assert!(check_packed_layout(&narrow, 4, 4).is_err());
        assert!(check_packed_layout(&zeros((1, 128, 4, 4)), 4, 4).is_err());
        let batched = Tensor::zeros((2, 16, 128), DType::F32, &Device::Cpu).unwrap();
        assert!(check_packed_layout(&batched, 4, 4).is_err());

        // The documented limit: a same-length transposition passes this check, by construction.
        assert!(check_packed_layout(&tokens, 2, 8).is_ok());
    }

    /// bf16 is the candle GPU denoise dtype; the shared projection casts to f32 up front, so the
    /// FLUX.2 seam must accept it rather than panicking in the matmul.
    #[test]
    fn projection_accepts_a_bf16_latent() {
        let latents = zeros((1, 32, 2, 2)).to_dtype(DType::BF16).unwrap();
        let image = project_raw_latents(&latents).unwrap();
        assert_eq!(image.pixels[..3], [112, 108, 104]);
    }

    /// Identity bn stats (`std = 1`, `mean = 0`) so a geometry row reasons about the unpatchify alone.
    /// Two `[1, 128, 1, 1]` tensors — no decoder is materialized, which is the point of taking the
    /// stats rather than a `Flux2Vae`.
    fn identity_stats() -> (Tensor, Tensor) {
        let shape = (1usize, PACKED_LATENT_CHANNELS, 1usize, 1usize);
        (
            Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap(),
            Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap(),
        )
    }

    /// The frame is at **VAE-latent** resolution `2·lat_h × 2·lat_w`, never at the token grid: the
    /// unpatchify doubles the spatial extent, and a preview at half the true resolution is exactly
    /// what projecting the packed grid would produce.
    #[test]
    fn packed_projection_is_vae_latent_resolution_not_token_grid_resolution() {
        let (std, mean) = identity_stats();
        for (lat_h, lat_w) in [(4usize, 4usize), (8, 6), (2, 3)] {
            let tokens = Tensor::zeros(
                (1, lat_h * lat_w, PACKED_LATENT_CHANNELS),
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
            let image = project_packed_tokens(&tokens, &std, &mean, lat_h, lat_w).unwrap();
            assert_eq!(
                (image.width, image.height),
                (lat_w as u32 * 2, lat_h as u32 * 2),
                "{lat_h}x{lat_w} must project at VAE-latent resolution"
            );
            assert_ne!((image.width, image.height), (lat_w as u32, lat_h as u32));
        }
    }

    /// The packed seam is exactly "unpack → the VAE's own raw-latent recovery → the raw projector":
    /// same bytes, no second code path for either the constants or the de-normalize/unpatchify.
    /// Non-identity stats, so a dropped de-normalize cannot pass.
    #[test]
    fn packed_projection_equals_the_raw_projection_of_the_recovered_latent() {
        let (lat_h, lat_w) = (3usize, 5usize);
        let shape = (1usize, PACKED_LATENT_CHANNELS, 1usize, 1usize);
        let std = Tensor::rand(0.5f32, 2.0f32, shape, &Device::Cpu).unwrap();
        let mean = Tensor::rand(-1f32, 1f32, shape, &Device::Cpu).unwrap();
        let tokens = Tensor::rand(
            -2f32,
            2f32,
            (1, lat_h * lat_w, PACKED_LATENT_CHANNELS),
            &Device::Cpu,
        )
        .unwrap();

        let grid = crate::pipeline::unpack_latents_at(&tokens, lat_h, lat_w).unwrap();
        assert_eq!(grid.dims(), [1, PACKED_LATENT_CHANNELS, lat_h, lat_w]);
        let raw = crate::vae::raw_latent_from_packed(&grid, &std, &mean).unwrap();
        assert_eq!(raw.dims(), [1, 32, lat_h * 2, lat_w * 2]);

        let via_packed = project_packed_tokens(&tokens, &std, &mean, lat_h, lat_w).unwrap();
        let via_raw = project_raw_latents(&raw).unwrap();
        assert_eq!(via_packed.pixels, via_raw.pixels);
        assert_eq!(
            (via_packed.width, via_packed.height),
            (via_raw.width, via_raw.height)
        );

        // The de-normalize is load-bearing: skipping it changes the picture. Pinned so a refactor
        // that projected the still-normalized grid would fail rather than look plausible.
        let (unit_std, zero_mean) = identity_stats();
        let undenormalized =
            project_packed_tokens(&tokens, &unit_std, &zero_mean, lat_h, lat_w).unwrap();
        assert_ne!(
            via_packed.pixels, undenormalized.pixels,
            "the bn de-normalize must actually change the projected frame"
        );
    }

    /// The seam is the one the decode uses: `Flux2Vae::raw_latent_from_packed` must be exactly the
    /// free function bound to the VAE's own stats. Asserted on the free function's own algebra
    /// (`x·std + mean` then unpatchify) rather than by building a decoder, and hand-computed on a
    /// 1×1 grid so the `c·4 + ph·2 + pw` channel order is pinned too.
    #[test]
    fn the_raw_latent_recovery_is_denormalize_then_channel_major_unpatchify() {
        // One 1x1 token cell, 128 channels carrying their own index as a value.
        let values: Vec<f32> = (0..PACKED_LATENT_CHANNELS).map(|c| c as f32).collect();
        let grid =
            Tensor::from_vec(values, (1, PACKED_LATENT_CHANNELS, 1, 1), &Device::Cpu).unwrap();
        let std = Tensor::full(2f32, (1, PACKED_LATENT_CHANNELS, 1, 1), &Device::Cpu).unwrap();
        let mean = Tensor::full(1f32, (1, PACKED_LATENT_CHANNELS, 1, 1), &Device::Cpu).unwrap();

        let raw = crate::vae::raw_latent_from_packed(&grid, &std, &mean).unwrap();
        assert_eq!(raw.dims(), [1, 32, 2, 2]);
        let flat = raw.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Channel c occupies packed channels c*4 + ph*2 + pw, and de-normalizes to 2*v + 1.
        let expect: Vec<f32> = (0..32)
            .flat_map(|c| (0..4).map(move |p| 2.0 * (c * 4 + p) as f32 + 1.0))
            .collect();
        assert_eq!(flat, expect);
    }

    /// `unpack_latents` (image-size keyed) and `unpack_latents_at` (grid keyed) are one fold: the
    /// wrapper must not drift from the primitive the preview seam uses.
    #[test]
    fn the_grid_keyed_unpack_agrees_with_the_image_size_keyed_one() {
        let (width, height) = (256u32, 128u32);
        let (lat_h, lat_w) = crate::pipeline::latent_dims(width, height);
        let tokens = Tensor::rand(
            -1f32,
            1f32,
            (1, lat_h * lat_w, PACKED_LATENT_CHANNELS),
            &Device::Cpu,
        )
        .unwrap();
        let by_size = crate::pipeline::unpack_latents(&tokens, width, height).unwrap();
        let by_grid = crate::pipeline::unpack_latents_at(&tokens, lat_h, lat_w).unwrap();
        assert_eq!(
            by_size.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            by_grid.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    // --- Driving the real sampler -----------------------------------------------------------------

    /// A small but genuinely FLUX.2-shaped render: 64² → a 4×4 token grid, 16 packed tokens, and an
    /// 8×8 raw latent.
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    const SEQ: usize = 16;

    fn packed_latent(width: u32, height: u32) -> Tensor {
        let (lat_h, lat_w) = crate::pipeline::latent_dims(width, height);
        Tensor::zeros(
            (1, lat_h * lat_w, PACKED_LATENT_CHANNELS),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap()
    }

    fn collecting_sink() -> (PreviewSink, Arc<Mutex<Vec<PreviewFrame>>>) {
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| candle_gen::lock_recover(&captured).push(frame));
        (sink, frames)
    }

    fn frames_of(captured: &Arc<Mutex<Vec<PreviewFrame>>>) -> Vec<(u32, u32)> {
        candle_gen::lock_recover(captured)
            .iter()
            .map(|f| (f.current, f.total))
            .collect()
    }

    /// A hook over the identity bn stats, so these rows exercise the real
    /// [`project_packed_tokens`] path without a VAE.
    fn stats_hook<'a>(
        sink: &'a PreviewSink,
        std: &'a Tensor,
        mean: &'a Tensor,
        width: u32,
        height: u32,
    ) -> PreviewHook<'a> {
        let (lat_h, lat_w) = crate::pipeline::latent_dims(width, height);
        PreviewHook::new(sink, move |tokens: &Tensor| {
            project_packed_tokens(tokens, std, mean, lat_h, lat_w)
        })
    }

    /// A velocity of exactly zero: the flow-Euler step leaves the latent untouched, so the sampler's
    /// output is a pure function of its input and any byte difference is the wiring's.
    fn zero_velocity(x: &Tensor, _t: f32) -> Result<Tensor> {
        Ok(x.zeros_like()?)
    }

    /// Drive the real flow sampler over `sigmas`, in the packed space and with the real schedule the
    /// routes resolve — the same driver, convention and argument order all three call sites use.
    fn run(
        sampler: Option<&str>,
        sigmas: &[f32],
        start: Tensor,
        preview: Option<&PreviewHook<'_>>,
        predict: impl FnMut(&Tensor, f32) -> Result<Tensor>,
    ) -> Result<Tensor> {
        candle_gen::run_flow_sampler(
            sampler,
            candle_gen::gen_core::sampling::TimestepConvention::Sigma,
            sigmas,
            start,
            7,
            &candle_gen::gen_core::CancelFlag::new(),
            &mut |_: candle_gen::gen_core::Progress| {},
            preview,
            predict,
        )
    }

    fn sigmas(steps: usize) -> Vec<f32> {
        crate::pipeline::schedule(steps, WIDTH, HEIGHT)
    }

    /// Euler evaluates once per step: an N-step render emits exactly N frames, 1..=N, each carrying
    /// `total == N`.
    #[test]
    fn euler_emits_exactly_one_numbered_frame_per_step() {
        let (std, mean) = identity_stats();
        for steps in [1usize, 4, 8] {
            let (sink, captured) = collecting_sink();
            let hook = stats_hook(&sink, &std, &mean, WIDTH, HEIGHT);
            run(
                None,
                &sigmas(steps),
                packed_latent(WIDTH, HEIGHT),
                Some(&hook),
                zero_velocity,
            )
            .unwrap();
            assert_eq!(
                frames_of(&captured),
                (1..=steps as u32)
                    .map(|n| (n, steps as u32))
                    .collect::<Vec<_>>(),
                "{steps}-step Euler render"
            );
        }
    }

    /// The candle-specific hazard the shared counter exists for: heun and dpmpp_sde evaluate the
    /// predict closure **twice** per outer step, so an undeduped path would emit 2N frames. The
    /// evaluation count is asserted to exceed the step count first, so a solver that silently fell
    /// back to Euler could not make this pass vacuously.
    #[test]
    fn multi_eval_solvers_still_emit_exactly_one_frame_per_outer_step() {
        let (std, mean) = identity_stats();
        for name in ["heun", "dpmpp_sde"] {
            let steps = 6usize;
            let evaluations = std::cell::Cell::new(0usize);
            let (sink, captured) = collecting_sink();
            let hook = stats_hook(&sink, &std, &mean, WIDTH, HEIGHT);
            run(
                Some(name),
                &sigmas(steps),
                packed_latent(WIDTH, HEIGHT),
                Some(&hook),
                |x, t| {
                    evaluations.set(evaluations.get() + 1);
                    zero_velocity(x, t)
                },
            )
            .unwrap();

            assert!(
                evaluations.get() > steps,
                "{name} must evaluate more than once per step for this test to mean anything \
                 (got {} evaluations for {steps} steps)",
                evaluations.get()
            );
            assert_eq!(
                frames_of(&captured),
                (1..=steps as u32)
                    .map(|n| (n, steps as u32))
                    .collect::<Vec<_>>(),
                "{name} must still emit exactly one frame per outer step"
            );
        }
    }

    /// Every emitted frame is a raw-latent-resolution RGB8 image of the running trajectory — `H/8`,
    /// i.e. twice the token grid, because the unpatchify runs before the projection.
    #[test]
    fn emitted_frames_are_raw_latent_resolution_rgb8() {
        let (std, mean) = identity_stats();
        let (sink, captured) = collecting_sink();
        let hook = stats_hook(&sink, &std, &mean, 128, 64);
        let start = packed_latent(128, 64);
        run(None, &sigmas(2), start, Some(&hook), zero_velocity).unwrap();

        let frames = candle_gen::lock_recover(&captured);
        assert_eq!(frames.len(), 2);
        for frame in frames.iter() {
            assert_eq!((frame.image.width, frame.image.height), (16, 8));
            assert_eq!(frame.image.pixels.len(), 16 * 8 * 3);
        }
    }

    // --- What the hook is allowed to see ----------------------------------------------------------

    /// The CFG hazard, driven through the real sampler with a predict closure shaped like klein's
    /// true-CFG branch: it fuses a two-leg batch internally and returns one blended velocity. The
    /// unconditional half never becomes the running latent, so it can never be projected.
    #[test]
    fn cfg_never_exposes_the_unconditional_half_to_the_preview() {
        let (std, mean) = identity_stats();
        let (sink, _captured) = collecting_sink();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&seen);
        let (lat_h, lat_w) = crate::pipeline::latent_dims(WIDTH, HEIGHT);
        let hook = PreviewHook::new(&sink, move |x: &Tensor| {
            candle_gen::lock_recover(&recorded).push(x.dims().to_vec());
            project_packed_tokens(x, &std, &mean, lat_h, lat_w)
        });

        run(
            None,
            &sigmas(4),
            packed_latent(WIDTH, HEIGHT),
            Some(&hook),
            |x, _t| {
                // klein's `vn + guidance·(v − vn)`: two forwards, blended back to one velocity in
                // the conditional space before returning.
                let fused = Tensor::cat(&[x, x], 0)?;
                assert_eq!(fused.dims()[0], 2);
                let cond = fused.narrow(0, 0, 1)?;
                Ok(cond.zeros_like()?)
            },
        )
        .unwrap();

        let seen = candle_gen::lock_recover(&seen);
        assert_eq!(seen.len(), 4);
        assert!(
            seen.iter()
                .all(|dims| dims == &[1, SEQ, PACKED_LATENT_CHANNELS]),
            "the hook must only ever see the single unfused conditional latent, got {seen:?}"
        );
    }

    /// The edit hazard, in this family's own shape: `edit_provider` concatenates the encoded
    /// reference tokens onto the **sequence** axis inside the closure and keeps the leading
    /// `target_seq` velocity tokens. If the sampler's latent ever carried reference tokens the
    /// preview would show the reference image, and the sequence length is exactly what would give it
    /// away — so the seen shapes are asserted against the target-only length, with the joint length
    /// computed and asserted different so the row cannot pass by the two coinciding.
    #[test]
    fn edit_previews_project_target_tokens_only() {
        let (std, mean) = identity_stats();
        let (sink, captured) = collecting_sink();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&seen);
        let (lat_h, lat_w) = crate::pipeline::latent_dims(WIDTH, HEIGHT);
        let hook = PreviewHook::new(&sink, move |x: &Tensor| {
            candle_gen::lock_recover(&recorded).push(x.dims().to_vec());
            project_packed_tokens(x, &std, &mean, lat_h, lat_w)
        });

        // Two references' worth of clean, constant tokens — the joint image stream.
        let references =
            Tensor::ones((1, 40, PACKED_LATENT_CHANNELS), DType::F32, &Device::Cpu).unwrap();
        let joint_seq = SEQ + 40;
        assert_ne!(joint_seq, SEQ);

        run(
            None,
            &sigmas(4),
            packed_latent(WIDTH, HEIGHT),
            Some(&hook),
            |x, _t| {
                let joint = Tensor::cat(&[x, &references], 1)?;
                assert_eq!(joint.dims()[1], joint_seq);
                // `velocity(...)` keeps the leading `target_seq` tokens.
                Ok(joint.narrow(1, 0, SEQ)?.zeros_like()?)
            },
        )
        .unwrap();

        let seen = candle_gen::lock_recover(&seen);
        assert_eq!(seen.len(), 4);
        assert!(
            seen.iter()
                .all(|dims| dims == &[1, SEQ, PACKED_LATENT_CHANNELS]),
            "the hook must never see a reference token, got {seen:?}"
        );
        for (width, height) in candle_gen::lock_recover(&captured)
            .iter()
            .map(|f| (f.image.width, f.image.height))
        {
            assert_eq!((width, height), (WIDTH / 8, HEIGHT / 8));
        }
    }

    // --- Decorative by contract -------------------------------------------------------------------

    /// An inert sink must be byte-identical to no hook at all, and an ACTIVE sink must be too — the
    /// preview reads the latent and never writes it.
    #[test]
    fn an_inert_sink_is_byte_identical_to_an_unhooked_render() {
        let (std, mean) = identity_stats();
        let s = sigmas(6);
        let start =
            Tensor::rand(-1f32, 1f32, (1, SEQ, PACKED_LATENT_CHANNELS), &Device::Cpu).unwrap();
        let velocity = |x: &Tensor, t: f32| Ok((x * (t as f64 + 0.25))?);
        let bytes = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let bare = run(None, &s, start.clone(), None, velocity).unwrap();

        let inert = PreviewSink::default();
        let inert_hook = stats_hook(&inert, &std, &mean, WIDTH, HEIGHT);
        assert!(!inert_hook.is_active());
        let hooked = run(None, &s, start.clone(), Some(&inert_hook), velocity).unwrap();
        assert_eq!(
            bytes(&bare),
            bytes(&hooked),
            "an inert preview sink must not perturb a single latent byte"
        );

        let (sink, captured) = collecting_sink();
        let active_hook = stats_hook(&sink, &std, &mean, WIDTH, HEIGHT);
        let active = run(None, &s, start, Some(&active_hook), velocity).unwrap();
        assert_eq!(bytes(&bare), bytes(&active));
        assert_eq!(candle_gen::lock_recover(&captured).len(), 6);
    }

    /// A projection failure loses its frame and never fails the render. The realistic shape of that
    /// failure here is a hook whose grid does not describe the running latent.
    #[test]
    fn a_projection_failure_loses_the_frame_and_never_fails_the_render() {
        let (std, mean) = identity_stats();
        let (sink, captured) = collecting_sink();
        // A hook built for a 1024² render, handed a 64² trajectory: every unpack fails.
        let hook = stats_hook(&sink, &std, &mean, 1024, 1024);
        let out = run(
            None,
            &sigmas(5),
            packed_latent(WIDTH, HEIGHT),
            Some(&hook),
            zero_velocity,
        )
        .expect("a failing projection must not fail the render");

        assert_eq!(out.dims(), [1, SEQ, PACKED_LATENT_CHANNELS]);
        assert!(
            candle_gen::lock_recover(&captured).is_empty(),
            "no frame may be emitted when every projection fails"
        );
    }

    // --- Route inventory --------------------------------------------------------------------------

    /// [`candle_gen::run_flow_sampler`]'s argument count before the predict closure. Pinned so a
    /// signature change — or a scanner mis-split — fails this inventory loudly instead of quietly
    /// shifting which argument "the one before the closure" names.
    const SAMPLER_ARGUMENTS_BEFORE_PREDICT: usize = 8;

    /// The arguments of every `run_flow_sampler` call in `source`, one entry per call site, covering
    /// the arguments **before** the predict closure — the window the `preview` argument sits in.
    ///
    /// Ported from sc-16950's Krea inventory. The window is bounded by the call's own bracket balance
    /// and ends at the first top-level `|`; it deliberately does not key off a closure parameter name,
    /// because a route naming that parameter something else would otherwise widen the window to the
    /// next call site (or to end of file) and let any `Some(&preview)` in the swallowed text — prose
    /// included — satisfy a route that was left dark. A missing bound is a failure, not a wider window.
    ///
    /// The match is textual, so writing the driver's name followed by an open paren in prose is read
    /// as a call site: name it without the paren in comments.
    fn sampler_call_sites(file: &str, source: &str) -> Vec<Vec<String>> {
        const CALL: &str = "run_flow_sampler(";
        let mut sites = Vec::new();
        let mut cursor = 0usize;
        while let Some(at) = source[cursor..].find(CALL) {
            let args_start = cursor + at + CALL.len();
            sites.push(sampler_call_arguments(
                file,
                sites.len(),
                &source[args_start..],
            ));
            cursor = args_start;
        }
        sites
    }

    /// The comma-separated top-level arguments of one call, given everything after its open paren.
    fn sampler_call_arguments(file: &str, index: usize, rest: &str) -> Vec<String> {
        let site = format!("{file}: run_flow_sampler call #{index}");
        let normalize = |text: &str| text.split_whitespace().collect::<Vec<_>>().join(" ");

        let mut args: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut depth = 1usize;
        let mut chars = rest.chars().peekable();
        while let Some(ch) = chars.next() {
            match ch {
                // Comments are not code: a `(` or a `|` inside one must not move the scan.
                '/' if chars.peek() == Some(&'/') => {
                    for c in chars.by_ref() {
                        if c == '\n' {
                            break;
                        }
                    }
                    current.push(' ');
                }
                '/' if chars.peek() == Some(&'*') => {
                    chars.next();
                    let (mut nesting, mut prev) = (1usize, '\0');
                    for c in chars.by_ref() {
                        match (prev, c) {
                            ('/', '*') => (nesting, prev) = (nesting + 1, '\0'),
                            ('*', '/') => {
                                nesting -= 1;
                                prev = '\0';
                                if nesting == 0 {
                                    break;
                                }
                            }
                            _ => prev = c,
                        }
                    }
                    assert_eq!(nesting, 0, "{site} has an unterminated block comment");
                    current.push(' ');
                }
                // Nor are string literals.
                '"' => {
                    current.push('"');
                    let mut escaped = false;
                    let mut closed = false;
                    for c in chars.by_ref() {
                        current.push(c);
                        if escaped {
                            escaped = false;
                        } else if c == '\\' {
                            escaped = true;
                        } else if c == '"' {
                            closed = true;
                            break;
                        }
                    }
                    assert!(closed, "{site} has an unterminated string literal");
                }
                '(' | '[' | '{' => {
                    depth += 1;
                    current.push(ch);
                }
                ')' | ']' | '}' => {
                    depth -= 1;
                    assert!(
                        depth > 0,
                        "{site} closes without a predict closure — the scan cannot bound its \
                         preview argument, so no assertion about that argument would mean anything"
                    );
                    current.push(ch);
                }
                ',' if depth == 1 => {
                    args.push(normalize(&current));
                    current.clear();
                }
                // The predict closure's parameter list: the argument window ends here, whatever that
                // parameter is called.
                '|' if depth == 1 => {
                    let trailing = normalize(&current);
                    assert!(
                        trailing.is_empty(),
                        "{site} has unparsed text {trailing:?} between its last argument and the \
                         predict closure — the scan cannot be trusted to have found the preview \
                         argument"
                    );
                    return args;
                }
                _ => current.push(ch),
            }
        }
        panic!("{site} is unterminated: no predict closure and no closing paren before end of file")
    }

    /// Every shipped FLUX.2 render route emits previews, pinned at the source level: the registered
    /// txt2img route (`lib.rs`), the reference edit provider (`edit_provider.rs`) and the strict-pose
    /// control provider (`control_provider.rs`) — one sampler site each, all three passing a hook. A
    /// route left unwired shows the user nothing, and no weights-free test can otherwise reach a
    /// route that needs a 9B or 32B DiT.
    ///
    /// This is the crate-local half of the epic-16948 guard; `candle-gen-catalog`'s
    /// `preview_advertising` module carries the same counts as the family's route inventory and ties
    /// them to the advertised `supports_preview`.
    #[test]
    fn every_flux2_render_route_passes_a_preview_hook() {
        for (file, source) in [
            ("lib.rs", include_str!("lib.rs")),
            ("edit_provider.rs", include_str!("edit_provider.rs")),
            ("control_provider.rs", include_str!("control_provider.rs")),
        ] {
            let sites = sampler_call_sites(file, source);
            assert_eq!(
                sites.len(),
                1,
                "{file}: expected exactly 1 sampler call site, found {}. A new render route must \
                 pass a preview hook and be named in this inventory (and in the catalog's).",
                sites.len()
            );
            let args = &sites[0];
            assert_eq!(
                args.len(),
                SAMPLER_ARGUMENTS_BEFORE_PREDICT,
                "{file}: expected {SAMPLER_ARGUMENTS_BEFORE_PREDICT} arguments before the predict \
                 closure, parsed {args:?}"
            );
            // Positional, not `contains`: the preview is the argument immediately before the predict
            // closure, so this cannot be satisfied by the word appearing anywhere else.
            assert_eq!(
                args.last().map(String::as_str),
                Some("Some(&preview)"),
                "{file} does not pass a preview hook: {args:?}"
            );
        }
    }

    /// `pipeline.rs` and `vae.rs` own the latent geometry, not a denoise loop: neither may hold a
    /// sampler site. Pinned as a negative so a future route added there cannot slip past the
    /// inventory above, which only looks at three named files.
    #[test]
    fn the_geometry_modules_drive_no_sampler() {
        assert!(sampler_call_sites("pipeline.rs", include_str!("pipeline.rs")).is_empty());
        assert!(sampler_call_sites("vae.rs", include_str!("vae.rs")).is_empty());
    }
}
