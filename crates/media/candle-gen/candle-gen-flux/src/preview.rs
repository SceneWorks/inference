//! The FLUX.1-family per-step latent preview seam (epic 16948, sc-16956; the MLX original is epic
//! 16624 / `mlx-gen-flux/src/preview.rs`).
//!
//! This module owns the **16-channel** FLUX.1 fit. `candle-gen-chroma` reuses it through [`hook`] /
//! [`project_packed_tokens`] rather than restating it, and `candle-gen-pulid` reaches it through the
//! FLUX.1 backbone it composes — all three denoise the *same* latent space, proven in tensor bytes
//! below. Schedule numbering, multi-eval dedup and the swallow-on-failure contract all live in
//! [`candle_gen::preview`], shared by every candle family.
//!
//! ## The packed seam: candle FLUX.1 does NOT hold spatial latents
//!
//! Epic 16948's scoping noted that `unpack_latents` appears only in `candle-gen-flux2` and
//! `candle-gen-qwen-image`, and asked this story to verify rather than assume. It is verified, and the
//! answer is that candle FLUX.1 **is** packed — it simply spells the recovery differently. The DiT
//! denoises a `[1, S, 64]` token sequence and the native `[1, 16, H/8, W/8]` VAE latent only exists on
//! the far side of [`candle_transformers::models::flux::sampling::unpack`], which
//! [`crate::decode_latents`] calls before every decode:
//!
//! | stage | shape | projectable? |
//! | --- | --- | --- |
//! | the sampler's running latent | `[1, ⌈H/16⌉·⌈W/16⌉, 64]` packed tokens | no — rank 3 |
//! | after `flux::sampling::unpack` | `[1, 16, 2⌈H/16⌉, 2⌈W/16⌉]` | yes — the fitted space |
//!
//! Unlike FLUX.2 there is **no** second transform: this VAE is a plain diffusers `AutoencoderKL` with
//! no BatchNorm-stats space, so the unpack alone recovers the fitted latent. That asymmetry is the
//! reason this module could not be written by porting `candle-gen-flux2/src/preview.rs`, and
//! `the_packed_recovery_is_the_one_the_decode_uses` pins it against the very function the decode tail
//! calls rather than against a second copy of the fold.
//!
//! The recovery is keyed on the **pixel** `(width, height)`, exactly as `unpack` is, and every route
//! builds its hook from the same pair it hands its decode tail — which is what keeps the preview's
//! geometry and the render's geometry from diverging.
//!
//! ## What the hook sees on each route
//!
//! Every shipped FLUX.1-family route drives [`candle_gen::run_flow_sampler`], so all of them opt in
//! with a projector closure rather than by restructuring a loop:
//!
//! * **CFG never reaches the preview.** FLUX.1 dev is guidance-distilled (a single forward with an
//!   embedded guidance vector) and schnell is timestep-distilled with no guidance at all; neither has a
//!   negative pass. Chroma *does* run true CFG, but the whole `neg + g·(pos − neg)` blend happens
//!   **inside** its predict closure and returns one velocity in the conditional space, so no fused
//!   `[2, …]` batch is ever the running latent.
//! * **The identity and control branches never reach the preview.** PuLID's `id_embedding`, the XLabs
//!   IP-Adapter's reference tokens and the Fun-ControlNet's encoded control latent are all closure
//!   captures, constant across steps, injected inside the DiT forward. The tensor the sampler
//!   integrates — and therefore the tensor the hook sees — stays the target image latent alone. That is
//!   this story's "PuLID's identity-embedding path must not perturb the previewed latent" criterion,
//!   and it is closed structurally rather than by a guard.
//!
//! ## The σ convention: this family needs no correction
//!
//! [`candle_gen::run_flow_sampler`] integrates a [`candle_gen::gen_core::sampling::FlowModelSampling`],
//! whose `input_scale` is exactly `1.0` at every σ, so the running latent already *is* the tensor the
//! fit was measured against and the σ-less [`candle_gen::preview::PreviewHook::new`] constructor is the
//! correct one. sc-16954 found the **opposite** for the discrete ε cohort (SDXL/Kolors denoise in
//! k-diffusion VE σ-space and must apply `1/√(σ²+1)` before projecting, or 89.4% of the first frame
//! clips to the rails). It is stated here rather than assumed, and `tests/preview_real_weights.rs`
//! measures the first frame's rail-clipped fraction to show the uncorrected projection is readable.
//!
//! ## The fit is reused, not refitted — grounded in tensor bytes
//!
//! `RGB_FACTORS` / `RGB_BIAS` are the epic-16624 constants transcribed verbatim from
//! `mlx-gen-flux/src/preview.rs`. They are least-squares numbers over a VAE *latent space* with no
//! backend in them; candle reuses them and deliberately ships **no producer** of its own —
//! `mlx-gen-flux/tests/fit_preview_rgb.rs` remains the only way they are re-derived.
//!
//! The reuse is grounded in the bytes each family actually loads, not in a shared Rust type. sc-16956
//! measured four containers of **one** learned 16-channel `AutoencoderKL`
//! (`latent_channels: 16`, `scaling_factor: 0.3611`, `shift_factor: 0.1159`, `block_out_channels:
//! [128, 256, 512, 512]` in all four `vae/config.json`s):
//!
//! * the **fit donor** — `SceneWorks/flux1-dev-mlx` @ `323fd12d…`, `q4/vae/model.safetensors`, SHA-256
//!   `e510ed25…4823`, 164,654,042 bytes, 260 tensors (diffusers layout; 244 learned bf16 plus the 16
//!   `scales`/`biases` arrays of the eight q4-packed mid-block attention linears). This is the file the
//!   MLX fit block names;
//! * the **diffusers bf16 container** — `black-forest-labs/FLUX.1-dev` and `FLUX.1-schnell`
//!   `vae/diffusion_pytorch_model.safetensors`, SHA-256 `f5b59a26…40a3`, 167,666,902 bytes, 244
//!   tensors. **Byte-identical** across those two repos and all three Chroma re-hosts;
//! * the **BFL f32 container** — `FLUX.1-dev` / `FLUX.1-schnell` `ae.safetensors`, SHA-256
//!   `afc8e282…9e38`, 335,304,388 bytes, 244 tensors in the BFL naming, which is what the dense
//!   `crate::vae::native::AutoEncoder` path loads (a private module, hence no intra-doc link);
//! * the **q8 tier** — `SceneWorks/flux1-dev-mlx` `q8/vae/model.safetensors`, SHA-256 `7cbe4841…f24d`.
//!
//! 236 of the fit donor's 244 learned tensors are **byte-identical** to the diffusers bf16 container;
//! the eight that differ are exactly the mid-block attention linears the MLX packer quantized. All 244
//! tensors of the BFL f32 container map onto the diffusers naming and round — round-to-nearest-even,
//! the rounding a bf16 cast performs — exactly onto its bits (83,819,683 values). `SceneWorks/flux1-schnell-mlx`
//! ships all 260 tensors identical to the fit donor's. `tests/preview_real_weights.rs` re-derives every
//! one of those claims per snapshot; the full record is
//! `docs/migration/evidence/sc-16956-flux1-candle-preview.md`.
//!
//! **This is also the 16-channel space `candle-gen-boogu` loads** (sc-17218): Boogu's
//! `vae/diffusion_pytorch_model.safetensors` (SHA-256 `8c717328…4c94`, 244 f32 tensors) rounds onto the
//! same bf16 bits, key for key, with the same config — so the fit in this module is the one Boogu
//! should reuse, not FLUX.2's 32-channel one. `the_boogu_vae_is_the_flux1_one` is that measurement.
//!
//! A stale or absent fit degrades preview colour only; the denoise path never reads these constants.

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::{Image, PreviewSink};
use candle_gen::preview::PreviewHook;
use candle_gen::{CandleError, Result};
use candle_transformers::models::flux::sampling::unpack;

/// Ordinary-least-squares map from the native FLUX.1 VAE latent to latent-resolution RGB (row *i* maps
/// latent channel *i* to `[r, g, b]`), with [`RGB_BIAS`] the intercept.
///
/// **Reused verbatim from `mlx-gen-flux/src/preview.rs:32`, not refitted.** Fit on four diverse
/// real-weight FLUX.1-dev q4 renders and measured on two disjoint prompt/seed holdouts, all 256² with
/// eight flow-Euler steps, against 8×8-average-pooled native VAE decodes. Fit R²
/// `(R,G,B) = (0.98570, 0.97910, 0.98121)`, overall `0.98224`; holdout R²
/// `(0.89993, 0.89133, 0.94286)`, overall `0.92176`.
///
/// Refit — in `mlx-gen-flux`, never here — whenever the FLUX.1 VAE lineage changes.
const RGB_FACTORS: [[f32; 3]; 16] = [
    [-0.012_527_851, 0.016_290_851, 0.043_425_495],
    [0.013_447_882, 0.033_666_9, 0.052_629_194],
    [0.028_631_245, -0.018_569_952, -0.007_021_428],
    [-0.009_539_733, 0.008_372_801, 0.035_320_2],
    [0.041_892_606, 0.028_447_104, 0.010_048_334],
    [0.006_346_499, 0.017_091_932, 0.013_491_33],
    [0.013_718_514, 0.050_447_08, 0.043_962_63],
    [-0.023_302_208, -0.018_885_406, -0.026_715_806],
    [-0.023_010_249, 0.008_753_829, 0.058_855_95],
    [0.072_232_775, 0.050_650_94, -0.021_402_475],
    [-0.015_363_334, 0.023_666_509, 0.009_932_561],
    [0.045_652_762, 0.013_941_527, 0.009_122_111],
    [0.028_260_466, 0.024_255_602, 0.027_867_623],
    [-0.080_196_46, -0.031_118_49, -0.082_909_666],
    [-0.011_887_487, -0.042_129_558, -0.013_012_416],
    [-0.060_526_643, -0.030_449_005, -0.025_197_53],
];

/// The fit's intercept — the near-neutral grey a fully-zero latent projects to. Reused with
/// `RGB_FACTORS`.
const RGB_BIAS: [f32; 3] = [0.495_903_46, 0.492_634_95, 0.482_487_62];

/// The FLUX.1-family latent channel count the fit is defined over. Derived from the committed factor
/// table's own length, so a consumer (Chroma, PuLID) cannot drift from it by restating a number.
pub const PREVIEW_LATENT_CHANNELS: usize = RGB_FACTORS.len();

/// The transformer's packed channel width — 16 latent channels × the 2×2 patch. Named because it is
/// what tells a packed token sequence apart from anything else: the packed latent is rank 3, so it
/// cannot be confused with the fitted grid by shape alone, but its channel width is the one number that
/// says which family's tokens these are.
pub const PACKED_LATENT_CHANNELS: usize = PREVIEW_LATENT_CHANNELS * 4;

/// The fit is the SIXTEEN-channel one, and the packed space is four times wider. Compile-time, because
/// a runtime row over constants proves nothing a `const` assertion does not prove earlier.
const _: () = assert!(PREVIEW_LATENT_CHANNELS == 16 && PACKED_LATENT_CHANNELS == 64);

/// The `(rows, cols)` **token** grid a `width × height` FLUX.1 render packs into, and therefore the
/// sequence length its running latent carries.
///
/// This is `flux::sampling::unpack`'s own `⌈·/16⌉` arithmetic, restated in one place so the layout
/// check and the routes agree with the decode rather than with each other. The recovered latent is
/// twice this in each axis — `[1, 16, 2·rows, 2·cols]`.
pub fn token_grid(width: u32, height: u32) -> (usize, usize) {
    (
        (height as usize).div_ceil(16),
        (width as usize).div_ceil(16),
    )
}

/// Project a **native 16-channel** FLUX.1-family latent `[1, 16, h, w]` to a latent-resolution RGB8
/// preview.
///
/// "Native" means **after** the 2×2 unpack, i.e. the tensor `flux::sampling::unpack` returns and
/// [`crate::decode_latents`] hands the VAE. A packed transformer latent must go through
/// [`project_packed_tokens`] instead.
///
/// Errors on any layout that is not one batch-1 sixteen-channel spatial latent. The caller's frame is
/// then lost and swallowed by [`candle_gen::preview::emit_preview`], the intended decorative-failure
/// behaviour.
pub fn project_raw_latents(latents: &Tensor) -> Result<Image> {
    check_raw_layout(latents)?;
    candle_gen::preview::project_latents(latents, &RGB_FACTORS, RGB_BIAS)
}

/// Project the sampler's **packed token** latent `[1, ⌈H/16⌉·⌈W/16⌉, 64]` by running the decode's own
/// recovery first: unpack the token sequence onto the native `[1, 16, 2⌈H/16⌉, 2⌈W/16⌉]` VAE latent,
/// then project.
///
/// `(width, height)` is the **pixel** size, the same pair the route hands its decode tail, and the
/// recovery is [`candle_transformers::models::flux::sampling::unpack`] — the exact function
/// [`crate::decode_latents`] calls — so there is one implementation of that seam and not two. [`hook`]
/// is the ergonomic form.
///
/// This is the reuse seam `candle-gen-chroma` calls: Chroma packs and unpacks with the same two
/// functions over the same `AutoencoderKL` bytes (see the module docs), so it shares these coefficients
/// and calling through here is what keeps a second copy of them from existing.
///
/// Errors — a latent that is not this route's packed shape, or a size that does not describe it — are
/// what the shared emitter swallows to lose exactly one decorative frame.
pub fn project_packed_tokens(tokens: &Tensor, width: u32, height: u32) -> Result<Image> {
    check_packed_layout(tokens, width, height)?;
    project_raw_latents(&unpack(tokens, height as usize, width as usize)?)
}

/// Reject anything that is not one batch-1 latent in the fitted sixteen-channel space.
///
/// The shared projection would reject most of these anyway, but naming the channel count here makes the
/// failure say *FLUX.1* — and catches the one case it cannot see, a rank-4 latent whose channel count
/// merely happens to match some other family's.
fn check_raw_layout(latents: &Tensor) -> Result<()> {
    let dims = latents.dims();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != PREVIEW_LATENT_CHANNELS {
        return Err(CandleError::Msg(format!(
            "flux1 preview latent must have shape [1, {PREVIEW_LATENT_CHANNELS}, h, w], got {dims:?}"
        )));
    }
    Ok(())
}

/// Reject a packed token sequence that is not this render's grid.
///
/// What this catches is a **sequence length** or channel width that does not describe the declared
/// size — the realistic failure, which is a hook built for one render size and handed another's
/// trajectory. It deliberately does not claim more: a packed sequence carries no grid of its own, so no
/// check here could tell a `256×1024` render from a `1024×256` one. That transposition is ruled out at
/// the call sites instead, by each route deriving the hook's size from the same pair it hands its
/// decode tail.
///
/// Checked before the reshape rather than left to it so the message names the size and the caller gets
/// one swallowed decorative frame rather than a raw candle shape error.
fn check_packed_layout(tokens: &Tensor, width: u32, height: u32) -> Result<()> {
    let (rows, cols) = token_grid(width, height);
    let dims = tokens.dims();
    if dims.len() != 3
        || dims[0] != 1
        || dims[1] != rows * cols
        || dims[2] != PACKED_LATENT_CHANNELS
    {
        return Err(CandleError::Msg(format!(
            "flux1 preview packed latent must have shape [1, {}, {PACKED_LATENT_CHANNELS}] for a \
             {width}x{height} render, got {dims:?}",
            rows * cols
        )));
    }
    Ok(())
}

/// The per-route preview hook every FLUX.1-family render route hands to
/// [`candle_gen::run_flow_sampler`]: a projector closure over [`project_packed_tokens`] bound to this
/// render's pixel size. The driver owns frame numbering, multi-eval dedup, and the swallow-on-failure
/// contract (sc-16949), so no route restructures its loop.
///
/// Build it **per image**: a batched request runs one driver call per seed and each call must start a
/// fresh trajectory at frame 1. (The driver builds its own counter per call, so this is a property of
/// the call rather than of the hook — building the hook alongside the call keeps the two impossible to
/// separate.)
///
/// Public because `candle-gen-chroma`'s render lane drives the same sampler over the same latent space
/// with the same `AutoencoderKL` bytes, and `candle-gen-pulid` composes this crate's own FLUX.1
/// backbone — both need this seam rather than one of their own.
pub fn hook(sink: &PreviewSink, width: u32, height: u32) -> PreviewHook<'_> {
    PreviewHook::new(sink, move |tokens: &Tensor| {
        project_packed_tokens(tokens, width, height)
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

    fn packed_zeros(width: u32, height: u32) -> Tensor {
        let (rows, cols) = token_grid(width, height);
        Tensor::zeros(
            (1, rows * cols, PACKED_LATENT_CHANNELS),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap()
    }

    /// The fit is **reused**, not refitted: these are the epic-16624 constants transcribed verbatim
    /// from `mlx-gen-flux/src/preview.rs`. Pinned as literals so an edit to either copy fails rather
    /// than silently forking one latent space into two colour maps.
    #[test]
    fn committed_fit_matches_the_mlx_source_block() {
        assert_eq!(RGB_FACTORS.len(), 16);
        assert_eq!(PREVIEW_LATENT_CHANNELS, 16);
        assert_eq!(
            RGB_FACTORS[0],
            [-0.012_527_851, 0.016_290_851, 0.043_425_495]
        );
        assert_eq!(RGB_FACTORS[6], [0.013_718_514, 0.050_447_08, 0.043_962_63]);
        assert_eq!(
            RGB_FACTORS[9],
            [0.072_232_775, 0.050_650_94, -0.021_402_475]
        );
        assert_eq!(
            RGB_FACTORS[13],
            [-0.080_196_46, -0.031_118_49, -0.082_909_666]
        );
        assert_eq!(
            RGB_FACTORS[15],
            [-0.060_526_643, -0.030_449_005, -0.025_197_53]
        );
        assert_eq!(RGB_BIAS, [0.495_903_46, 0.492_634_95, 0.482_487_62]);
        assert!(RGB_FACTORS.iter().flatten().all(|v| v.is_finite()));
        assert!(RGB_BIAS.iter().all(|v| v.is_finite()));
    }

    /// A zero latent projects to the fit's intercept — the one place the committed bias is directly
    /// observable, so a typo in [`RGB_BIAS`] cannot pass.
    #[test]
    fn a_zero_latent_projects_to_the_fit_intercept() {
        let image = project_raw_latents(&zeros((1, 16, 2, 3))).unwrap();
        assert_eq!((image.width, image.height), (3, 2));
        // 0.49590346·255 = 126.5, 0.49263495·255 = 125.6, 0.48248762·255 = 123.0
        let expect: Vec<u8> = [126, 126, 123].repeat(6);
        assert_eq!(image.pixels, expect);
    }

    /// A latent from another family's space must be rejected rather than projected against a
    /// mismatched map — the FLUX.2 32-channel grid and the SDXL 4-channel one both land here.
    #[test]
    fn a_foreign_channel_count_is_rejected_by_the_raw_projector() {
        for bad in [
            zeros((1, 32, 2, 3)),
            zeros((1, 4, 2, 3)),
            zeros((1, PACKED_LATENT_CHANNELS, 4, 4)),
            zeros((2, 16, 2, 3)),
        ] {
            let error = project_raw_latents(&bad).unwrap_err().to_string();
            assert!(
                error.contains("flux1 preview latent must have shape [1, 16, h, w]"),
                "unexpected error: {error}"
            );
        }
        let rank_three = Tensor::zeros((1, 16, 64), DType::F32, &Device::Cpu).unwrap();
        assert!(project_raw_latents(&rank_three).is_err());
    }

    /// The packed layout check runs before the reshape, so a hook built for a different render size
    /// loses one decorative frame with a size-naming error instead of folding onto the wrong shape.
    ///
    /// The limit of the check is pinned in the same row rather than left to the doc comment: a
    /// transposed size of the same token count is **not** detectable from the sequence alone, and the
    /// call sites are what rule it out.
    #[test]
    fn a_token_sequence_that_does_not_describe_the_render_size_is_an_error() {
        let tokens = packed_zeros(64, 64); // a 4x4 token grid, 16 tokens
        assert_eq!(tokens.dims(), [1, 16, PACKED_LATENT_CHANNELS]);
        assert!(check_packed_layout(&tokens, 64, 64).is_ok());

        for (w, h) in [(128u32, 64u32), (64, 128), (32, 32), (256, 256)] {
            let error = check_packed_layout(&tokens, w, h).unwrap_err().to_string();
            assert!(
                error.contains("render"),
                "a {w}x{h} render must be rejected: {error}"
            );
        }

        // Wrong channel width and wrong rank are rejected too.
        let narrow = Tensor::zeros((1, 16, 32), DType::F32, &Device::Cpu).unwrap();
        assert!(check_packed_layout(&narrow, 64, 64).is_err());
        assert!(check_packed_layout(&zeros((1, 64, 4, 4)), 64, 64).is_err());
        let batched =
            Tensor::zeros((2, 16, PACKED_LATENT_CHANNELS), DType::F32, &Device::Cpu).unwrap();
        assert!(check_packed_layout(&batched, 64, 64).is_err());

        // The documented limit: a transposition of the same token count passes, by construction.
        assert!(check_packed_layout(&tokens, 32, 128).is_ok());
    }

    /// bf16 is the candle GPU denoise dtype (FLUX.1 loads at bf16 regardless of the CPU default); the
    /// shared projection casts to f32 up front, so this seam must accept it rather than panicking in
    /// the matmul.
    #[test]
    fn projection_accepts_a_bf16_latent() {
        let latents = zeros((1, 16, 2, 2)).to_dtype(DType::BF16).unwrap();
        let image = project_raw_latents(&latents).unwrap();
        assert_eq!(image.pixels[..3], [126, 126, 123]);
    }

    /// The frame is at **native VAE-latent** resolution — twice the token grid in each axis, because
    /// the 2×2 unpack runs before the projection. A preview at half the true resolution is exactly what
    /// projecting the token grid as if it were spatial would produce.
    #[test]
    fn packed_projection_is_native_latent_resolution_not_token_grid_resolution() {
        for (w, h) in [(64u32, 64u32), (128, 64), (48, 32), (1024, 768)] {
            let image = project_packed_tokens(&packed_zeros(w, h), w, h).unwrap();
            let (rows, cols) = token_grid(w, h);
            assert_eq!(
                (image.width, image.height),
                (cols as u32 * 2, rows as u32 * 2),
                "{w}x{h} must project at native VAE-latent resolution"
            );
            assert_ne!((image.width, image.height), (cols as u32, rows as u32));
        }
    }

    /// The packed seam is exactly "the decode's own unpack → the raw projector": same bytes, no second
    /// implementation of the 2×2 fold. Asserted against
    /// [`candle_transformers::models::flux::sampling::unpack`] itself — which is what
    /// [`crate::decode_latents`] calls — over a non-trivial latent, so a hand-rolled fold that
    /// transposed the patch axes could not agree.
    #[test]
    fn the_packed_recovery_is_the_one_the_decode_uses() {
        let (width, height) = (80u32, 48u32);
        let (rows, cols) = token_grid(width, height);
        let tokens = Tensor::rand(
            -2f32,
            2f32,
            (1, rows * cols, PACKED_LATENT_CHANNELS),
            &Device::Cpu,
        )
        .unwrap();

        let native = unpack(&tokens, height as usize, width as usize).unwrap();
        assert_eq!(
            native.dims(),
            [1, PREVIEW_LATENT_CHANNELS, rows * 2, cols * 2]
        );

        let via_packed = project_packed_tokens(&tokens, width, height).unwrap();
        let via_raw = project_raw_latents(&native).unwrap();
        assert_eq!(via_packed.pixels, via_raw.pixels);
        assert_eq!(
            (via_packed.width, via_packed.height),
            (via_raw.width, via_raw.height)
        );

        // The unpack is load-bearing: projecting the token sequence reshaped naively — the same values
        // in the same order, laid out as a [1, 16, ·, ·] grid — is a different picture. Pinned so a
        // refactor that dropped the permute would be red rather than merely wrong-looking.
        let naive = tokens
            .reshape((1, rows * 2, cols * 2, PREVIEW_LATENT_CHANNELS))
            .unwrap()
            .permute((0, 3, 1, 2))
            .unwrap()
            .contiguous()
            .unwrap();
        let scrambled = project_raw_latents(&naive).unwrap();
        assert_ne!(
            via_packed.pixels, scrambled.pixels,
            "the 2x2 unpack must actually change the projected frame"
        );
    }

    /// The token grid is `flux::sampling::unpack`'s own `ceil(·/16)` arithmetic, including for a size
    /// that is not a multiple of 16 (the providers floor at multiples of 16, but the registered route
    /// does not, so the two must agree there too).
    #[test]
    fn the_token_grid_matches_the_unpack_arithmetic() {
        for (w, h) in [(1024u32, 1024u32), (768, 512), (256, 256), (72, 40)] {
            let (rows, cols) = token_grid(w, h);
            let tokens = Tensor::zeros(
                (1, rows * cols, PACKED_LATENT_CHANNELS),
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
            let native = unpack(&tokens, h as usize, w as usize).unwrap();
            assert_eq!(
                native.dims(),
                [1, PREVIEW_LATENT_CHANNELS, rows * 2, cols * 2],
                "{w}x{h}"
            );
        }
    }

    // --- Driving the real sampler -----------------------------------------------------------------

    /// A small but genuinely FLUX.1-shaped render: 64² → a 4×4 token grid, 16 packed tokens, and an
    /// 8×8 native latent.
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    const SEQ: usize = 16;

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

    /// A velocity of exactly zero: the flow-Euler step leaves the latent untouched, so the sampler's
    /// output is a pure function of its input and any byte difference is the wiring's.
    fn zero_velocity(x: &Tensor, _t: f32) -> Result<Tensor> {
        Ok(x.zeros_like()?)
    }

    /// Drive the real flow sampler over `sigmas`, in the packed space and with the real schedule the
    /// routes resolve — the same driver, convention and argument order all five call sites use.
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
            16956,
            &candle_gen::gen_core::CancelFlag::new(),
            &mut |_: candle_gen::gen_core::Progress| {},
            preview,
            predict,
        )
    }

    /// FLUX.1-dev's own time-shifted schedule at this render's sequence length — the array every route
    /// resolves, not a synthetic ramp.
    fn sigmas(steps: usize) -> Vec<f32> {
        let native: Vec<f32> = candle_transformers::models::flux::sampling::get_schedule(
            steps,
            Some((SEQ, 0.5, 1.15)),
        )
        .iter()
        .map(|&t| t as f32)
        .collect();
        candle_gen::resolve_flow_schedule(
            None,
            crate::flow_mu(crate::Variant::Dev, SEQ),
            steps,
            &native,
        )
    }

    /// Euler evaluates once per step: an N-step render emits exactly N frames, 1..=N, each carrying
    /// `total == N`.
    #[test]
    fn euler_emits_exactly_one_numbered_frame_per_step() {
        for steps in [1usize, 4, 8] {
            let (sink, captured) = collecting_sink();
            let hook = hook(&sink, WIDTH, HEIGHT);
            run(
                None,
                &sigmas(steps),
                packed_zeros(WIDTH, HEIGHT),
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
    /// evaluation count is asserted to exceed the step count first, so a solver that silently fell back
    /// to Euler could not make this pass vacuously.
    #[test]
    fn multi_eval_solvers_still_emit_exactly_one_frame_per_outer_step() {
        for name in ["heun", "dpmpp_sde"] {
            let steps = 6usize;
            let evaluations = std::cell::Cell::new(0usize);
            let (sink, captured) = collecting_sink();
            let hook = hook(&sink, WIDTH, HEIGHT);
            run(
                Some(name),
                &sigmas(steps),
                packed_zeros(WIDTH, HEIGHT),
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

    /// Every emitted frame is a native-latent-resolution RGB8 image of the running trajectory — `H/8`,
    /// i.e. twice the token grid, because the unpack runs before the projection.
    #[test]
    fn emitted_frames_are_native_latent_resolution_rgb8() {
        let (sink, captured) = collecting_sink();
        let hook = hook(&sink, 128, 64);
        run(
            None,
            &sigmas(2),
            packed_zeros(128, 64),
            Some(&hook),
            zero_velocity,
        )
        .unwrap();

        let frames = candle_gen::lock_recover(&captured);
        assert_eq!(frames.len(), 2);
        for frame in frames.iter() {
            assert_eq!((frame.image.width, frame.image.height), (16, 8));
            assert_eq!(frame.image.pixels.len(), 16 * 8 * 3);
        }
    }

    // --- What the hook is allowed to see ----------------------------------------------------------

    /// The CFG hazard, driven through the real sampler with a predict closure shaped like Chroma's
    /// true-CFG branch: it fuses a two-leg batch internally and returns one blended velocity. The
    /// unconditional half never becomes the running latent, so it can never be projected.
    #[test]
    fn cfg_never_exposes_the_unconditional_half_to_the_preview() {
        let (sink, _captured) = collecting_sink();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&seen);
        let hook = PreviewHook::new(&sink, move |x: &Tensor| {
            candle_gen::lock_recover(&recorded).push(x.dims().to_vec());
            project_packed_tokens(x, WIDTH, HEIGHT)
        });

        run(
            None,
            &sigmas(4),
            packed_zeros(WIDTH, HEIGHT),
            Some(&hook),
            |x, _t| {
                // Chroma's `neg + g·(pos − neg)`: two forwards, blended back to one velocity in the
                // conditional space before returning.
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

    /// The identity/control hazard in this family's own shape, and this story's PuLID criterion:
    /// PuLID's `id_embedding`, the XLabs IP residuals and the Fun-ControlNet's encoded control latent
    /// are all injected **inside** the DiT forward. They may change the returned velocity as much as
    /// they like; what they must never do is become part of the tensor the sampler integrates. Driven
    /// through the real sampler with a closure that concatenates a conditioning stream onto the
    /// sequence axis and slices the target tokens back out — the shape that WOULD leak if a route
    /// handed the joint tensor to the driver.
    #[test]
    fn injected_conditioning_never_reaches_the_previewed_latent() {
        let (sink, captured) = collecting_sink();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&seen);
        let hook = PreviewHook::new(&sink, move |x: &Tensor| {
            candle_gen::lock_recover(&recorded).push(x.dims().to_vec());
            project_packed_tokens(x, WIDTH, HEIGHT)
        });

        // A constant identity/control stream — 32 PuLID id tokens' worth of clean conditioning.
        let identity =
            Tensor::ones((1, 32, PACKED_LATENT_CHANNELS), DType::F32, &Device::Cpu).unwrap();
        let joint_seq = SEQ + 32;
        assert_ne!(joint_seq, SEQ);

        run(
            None,
            &sigmas(4),
            packed_zeros(WIDTH, HEIGHT),
            Some(&hook),
            |x, _t| {
                let joint = Tensor::cat(&[x, &identity], 1)?;
                assert_eq!(joint.dims()[1], joint_seq);
                Ok(joint.narrow(1, 0, SEQ)?.zeros_like()?)
            },
        )
        .unwrap();

        let seen = candle_gen::lock_recover(&seen);
        assert_eq!(seen.len(), 4);
        assert!(
            seen.iter()
                .all(|dims| dims == &[1, SEQ, PACKED_LATENT_CHANNELS]),
            "the hook must never see an injected conditioning token, got {seen:?}"
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
        let s = sigmas(6);
        let start =
            Tensor::rand(-1f32, 1f32, (1, SEQ, PACKED_LATENT_CHANNELS), &Device::Cpu).unwrap();
        let velocity = |x: &Tensor, t: f32| Ok((x * (t as f64 + 0.25))?);
        let bytes = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let bare = run(None, &s, start.clone(), None, velocity).unwrap();

        let inert = PreviewSink::default();
        let inert_hook = hook(&inert, WIDTH, HEIGHT);
        assert!(!inert_hook.is_active());
        let hooked = run(None, &s, start.clone(), Some(&inert_hook), velocity).unwrap();
        assert_eq!(
            bytes(&bare),
            bytes(&hooked),
            "an inert preview sink must not perturb a single latent byte"
        );

        let (sink, captured) = collecting_sink();
        let active_hook = hook(&sink, WIDTH, HEIGHT);
        let active = run(None, &s, start, Some(&active_hook), velocity).unwrap();
        assert_eq!(bytes(&bare), bytes(&active));
        assert_eq!(candle_gen::lock_recover(&captured).len(), 6);
    }

    /// A projection failure loses its frame and never fails the render. The realistic shape of that
    /// failure here is a hook whose render size does not describe the running latent.
    #[test]
    fn a_projection_failure_loses_the_frame_and_never_fails_the_render() {
        let (sink, captured) = collecting_sink();
        // A hook built for a 1024² render, handed a 64² trajectory: every unpack fails.
        let hook = hook(&sink, 1024, 1024);
        let out = run(
            None,
            &sigmas(5),
            packed_zeros(WIDTH, HEIGHT),
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
    /// Ported from sc-16955's FLUX.2 inventory. The window is bounded by the call's own bracket balance
    /// and ends at the first top-level `|`; it deliberately does not key off a closure parameter name,
    /// because a route naming that parameter something else would otherwise widen the window to the
    /// next call site (or to end of file) and let any `Some(&preview)` in the swallowed text — prose
    /// included — satisfy a route that was left dark. A missing bound is a failure, not a wider window.
    ///
    /// The match is textual, so writing the driver's name followed by an open paren in prose is read as
    /// a call site: name it without the paren in comments.
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

    /// Every shipped FLUX.1 render route emits previews, pinned at the source level: the registered
    /// txt2img route both `flux1_*` descriptors share (`pipeline.rs`), the name-driven
    /// Fun-ControlNet-Union strict-pose provider (`control_provider.rs`) and the name-driven XLabs
    /// IP-Adapter provider (`ip_provider.rs`) — one sampler site each, all three passing a hook. A route
    /// left unwired shows the user nothing, and no weights-free test can otherwise reach a route that
    /// needs a 12B DiT plus a T5-XXL.
    ///
    /// This is the crate-local half of the epic-16948 guard; `candle-gen-catalog`'s
    /// `preview_advertising` module carries the same counts as the family's route inventory and ties
    /// them to the advertised `supports_preview`.
    ///
    /// The expected argument is pinned per file rather than searched for. All three routes take their
    /// hook as a `&PreviewHook` parameter on a private `denoise` and therefore pass `Some(preview)`;
    /// spelling that out keeps the assertion positional, so it cannot be satisfied by the word
    /// appearing anywhere else in the call.
    #[test]
    fn every_flux1_render_route_passes_a_preview_hook() {
        for (file, source, expected) in [
            ("pipeline.rs", include_str!("pipeline.rs"), "Some(preview)"),
            (
                "control_provider.rs",
                include_str!("control_provider.rs"),
                "Some(preview)",
            ),
            (
                "ip_provider.rs",
                include_str!("ip_provider.rs"),
                "Some(preview)",
            ),
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
                Some(expected),
                "{file} does not pass a preview hook: {args:?}"
            );
        }
    }

    /// The control provider ships **one** sampler site with **two** public entry points —
    /// `generate` (the plain control render) and `generate_with_injector` (the compose-ready seam that
    /// stacks PuLID / IP identity on top of control) — and `generate` is implemented as a delegation to
    /// the other. A site-level assertion cannot see that distinction, and sc-16955's Lens finding was
    /// exactly this shape: one site, several callers, one of which could quietly stop forwarding its
    /// sink.
    ///
    /// Here the two callers cannot diverge, because there is only one body: `generate` forwards its
    /// whole `req` — preview sink included — and adds `None` for the injector. Pinned against the
    /// crate's own source so a future refactor that gave `generate` a body of its own would have to
    /// come back here.
    #[test]
    fn both_control_entry_points_reach_the_one_hooked_site() {
        let source = include_str!("control_provider.rs");
        let stripped: String = source
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            stripped.contains("self.generate_with_injector(req, control_image, None, on_progress)"),
            "`generate` must delegate its whole request to `generate_with_injector`, or the two \
             entry points no longer share the one hooked sampler site"
        );
        assert_eq!(
            stripped.matches("fn generate").count(),
            2,
            "control_provider.rs must expose exactly the two entry points this row reasons about"
        );
        assert_eq!(
            sampler_call_sites("control_provider.rs", source).len(),
            1,
            "both entry points must reach ONE sampler site"
        );
    }

    /// `flux1_load.rs`, the reference backbone and the control/IP weight modules own geometry and
    /// weights, not a denoise loop: none may hold a sampler site. Pinned as a negative so a future route
    /// added there cannot slip past the inventory above, which only looks at three named files.
    ///
    /// `preview.rs` is deliberately absent from this list: **this** module's own test helpers drive the
    /// real sampler (that is what the rows above are), and this crate-local scanner has no `cfg(test)`
    /// strip, so scanning itself would read those helpers as shipped routes. The file's shipped-code
    /// count of zero is asserted where a strip does exist — `candle-gen-catalog`'s `preview_advertising`
    /// module walks this crate's module tree with test-only items removed and pins the exact per-file
    /// tally, so a sampler call added to `preview.rs`'s shipped half would fail there.
    #[test]
    fn the_geometry_and_weight_modules_drive_no_sampler() {
        for (file, source) in [
            ("flux1_load.rs", include_str!("flux1_load.rs")),
            ("ref_backbone.rs", include_str!("ref_backbone.rs")),
            ("control.rs", include_str!("control.rs")),
            ("ip_adapter.rs", include_str!("ip_adapter.rs")),
            ("lib.rs", include_str!("lib.rs")),
        ] {
            assert!(
                sampler_call_sites(file, source).is_empty(),
                "{file} must not drive a sampler"
            );
        }
    }
}
