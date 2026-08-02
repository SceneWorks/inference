//! The QwenVae latent→RGB preview fit — the constants every Qwen-latent-space candle family projects
//! through (epic 16948; the MLX original is epic 16624 / `mlx-gen-qwen-image/src/preview.rs`).
//!
//! This module owns **only** the fitted constants and the spatial projection that applies them.
//! Schedule numbering, emission, dedup, and the swallow-on-failure contract live in
//! [`candle_gen::preview`], shared by every candle family (sc-16949).
//!
//! ## The fit is reused, not refitted (sc-16950)
//!
//! `RGB_FACTORS` / `RGB_BIAS` are the least-squares constants epic 16624 committed at
//! `mlx-gen-qwen-image/src/preview.rs:42`, transcribed verbatim. They are ordinary numbers over a VAE
//! *latent space* with no backend in them, so the correct candle move is to reuse them — but only
//! once the reuse is grounded in **tensor bytes** rather than in a matching Rust type name. That
//! grounding is recorded in `docs/migration/evidence/sc-16950-krea-candle-preview.md` and pinned by
//! `tests::committed_fit_matches_the_mlx_source_block` plus the Krea-side provenance row
//! (`candle-gen-krea/tests/preview_real_weights.rs`):
//!
//! * `krea/Krea-2-Turbo` @ `1161245028ef398cd0a951101b2bbf486464f841` — `vae/` SHA-256
//!   `ab1b61103959913d6c7e628cf793dbb2ca4726a40a3b3ae206c52b8e75bf6f08`;
//! * `krea/Krea-2-Raw` @ `4ad9f4b627a647fad78b3dfeebb09f2654aeb494` — the **same** file SHA-256;
//! * `SceneWorks/qwen-image-mlx` @ `8080a4171f1c8b7fca6c30491eafbe6ffab754bf` — `q4|q8/vae/` SHA-256
//!   `0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344`, the snapshot the MLX fit was
//!   measured against.
//!
//! All **194** tensors are value-identical across the two files (126,892,531 values); the container
//! differs only in width — the published Krea `vae/` is an f32 container whose values are all exactly
//! bf16-representable (zero low-16 mantissa bits, every value), and the MLX snapshot stores those same
//! values as bf16. `latents_mean` / `latents_std`, which *define* the normalized latent space the fit
//! was measured in, are identical in both `vae/config.json` files.
//!
//! A stale or absent fit degrades preview colour only; the denoise path never reads these constants.

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::Image;
use candle_gen::Result;

/// Least-squares latent→RGB factors for the Qwen-Image VAE latent space (16 channels; row *i* maps
/// latent channel *i* to `[r, g, b]`), with [`RGB_BIAS`] the intercept.
///
/// Fit by ordinary least squares on `decoded_rgb ≈ latent · M + b` over (final unpacked latent,
/// 8×-downsampled VAE decode) pairs — 2 prompts/seeds, 8-step Lightning at 1024², 32,768 samples,
/// R² = 0.9586. The producer is `mlx-gen-qwen-image/tests/fit_preview_rgb.rs`; candle has no producer
/// of its own **by design** — a second fit of the same latent space would be a second source of truth
/// for one set of numbers.
///
/// **These are not Qwen-Image-only.** `candle-gen-krea` reuses [`crate::vae::QwenVae`] directly, so
/// the same latent space — and therefore the same fit — applies to the Krea family unchanged, exactly
/// as `mlx-gen-krea` reuses the MLX original. A family needs its own fit only if it has its own VAE.
const RGB_FACTORS: [[f32; 3]; 16] = [
    [-0.00986379, 0.0257554, 0.211834],
    [-0.00150066, -0.00355605, 0.00219657],
    [0.0881243, 0.0565462, 0.0390654],
    [0.166173, 0.180288, 0.0838119],
    [0.0081918, -0.00272948, -0.0139806],
    [0.0276023, -0.0379166, -0.0372937],
    [-0.144053, -0.167288, -0.107295],
    [-0.0423725, -0.004423, 0.00174681],
    [-0.0705916, -0.0879479, -0.17535],
    [-0.0603724, 0.0326614, 0.0934403],
    [0.0473827, 0.121914, 0.0651104],
    [0.0138456, 0.0267495, 0.0120851],
    [-0.0844989, -0.0160223, 0.0123298],
    [-0.0162293, -0.0335703, -0.018524],
    [0.111816, 0.050061, 0.0724697],
    [0.0448471, 0.0208121, 0.0407526],
];

/// Intercept of the [`RGB_FACTORS`] fit — the mid-grey a zero latent projects to.
const RGB_BIAS: [f32; 3] = [0.406258, 0.385829, 0.287052];

/// The QwenVae latent channel count the fit is defined over. A latent that does not carry exactly
/// this many channels is not in the fitted space, and [`project_spatial_latents`] rejects it rather
/// than projecting a mismatched map.
pub const PREVIEW_LATENT_CHANNELS: usize = RGB_FACTORS.len();

/// Project a **spatial** QwenVae latent `[1, 16, h, w]` to a latent-resolution RGB8 preview.
///
/// This is the provider-owned reuse seam for the fitted coefficients. `candle-gen-krea` keeps its
/// denoise state in exactly this layout (`[1, 16, H/8, W/8]`, the normalized space
/// [`crate::vae::QwenVae::decode`] de-normalizes), so its routes hand the sampler's running latent
/// straight here with no unpack step.
///
/// Errors on any other layout: the caller's frame is then lost and swallowed by
/// [`candle_gen::preview::emit_preview`], which is the intended decorative-failure behaviour.
pub fn project_spatial_latents(latents: &Tensor) -> Result<Image> {
    candle_gen::preview::project_latents(latents, &RGB_FACTORS, RGB_BIAS)
}

#[cfg(test)]
mod tests {
    use candle_gen::candle_core::{DType, Device};

    use super::*;

    /// The fit is **reused**, not refitted: these are the epic-16624 constants transcribed verbatim
    /// from `mlx-gen-qwen-image/src/preview.rs`. Pinned as literals here so an edit to either copy
    /// fails rather than silently forking one latent space into two colour maps.
    #[test]
    fn committed_fit_matches_the_mlx_source_block() {
        assert_eq!(RGB_FACTORS.len(), 16);
        assert_eq!(PREVIEW_LATENT_CHANNELS, 16);
        assert_eq!(RGB_FACTORS[0], [-0.00986379, 0.0257554, 0.211834]);
        assert_eq!(RGB_FACTORS[6], [-0.144053, -0.167288, -0.107295]);
        assert_eq!(RGB_FACTORS[15], [0.0448471, 0.0208121, 0.0407526]);
        assert_eq!(RGB_BIAS, [0.406258, 0.385829, 0.287052]);
    }

    /// A zero latent projects to the intercept — the mid-grey a preview opens on before any structure
    /// has emerged, and the cheapest end-to-end check that the constants are wired the right way round.
    #[test]
    fn zero_latent_projects_to_the_intercept_grey() {
        let latents = Tensor::zeros((1, 16, 3, 5), DType::F32, &Device::Cpu).unwrap();
        let image = project_spatial_latents(&latents).unwrap();
        assert_eq!((image.width, image.height), (5, 3));
        // 0.406258·255 = 103.6, 0.385829·255 = 98.4, 0.287052·255 = 73.2
        assert_eq!(image.pixels[..3], [104, 98, 73]);
        assert_eq!(image.pixels.len(), 3 * 5 * 3);
        assert!(image.pixels.chunks_exact(3).all(|p| p == [104, 98, 73]));
    }

    /// The projection is latent-resolution, not image-resolution: a 1024² render's `[1, 16, 128, 128]`
    /// latent yields a 128² frame.
    #[test]
    fn projection_is_latent_resolution() {
        let latents = Tensor::zeros((1, 16, 128, 128), DType::F32, &Device::Cpu).unwrap();
        let image = project_spatial_latents(&latents).unwrap();
        assert_eq!((image.width, image.height), (128, 128));
    }

    /// bf16 is the candle GPU denoise dtype; the shared projection casts to f32 up front, so the
    /// Qwen-space seam must accept it rather than panicking in the matmul.
    #[test]
    fn projection_accepts_a_bf16_latent() {
        let latents = Tensor::zeros((1, 16, 2, 2), DType::F32, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let image = project_spatial_latents(&latents).unwrap();
        assert_eq!(image.pixels[..3], [104, 98, 73]);
    }

    /// A latent outside the fitted space is rejected — the error is what the shared emitter swallows
    /// to lose exactly one decorative frame.
    #[test]
    fn projection_rejects_a_non_qwen_latent_layout() {
        for shape in [(1usize, 4usize, 2usize, 2usize), (1, 32, 2, 2)] {
            let latents = Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap();
            let error = project_spatial_latents(&latents).unwrap_err();
            assert!(error.to_string().contains("does not match latent channel"));
        }

        let packed = Tensor::zeros((1usize, 64usize, 64usize), DType::F32, &Device::Cpu).unwrap();
        let error = project_spatial_latents(&packed).unwrap_err();
        assert!(error.to_string().contains("[1, C, h, w]"));
    }
}
