//! Per-step previews for the FLUX.2 32-channel VAE latent space.
//!
//! FLUX.2 and Lens denoise BatchNorm-normalized 128-channel 2×2-patchified tokens. Ideogram 4
//! uses the same learned VAE latent basis but a different patch-major channel order. Projection is
//! therefore always performed only after the VAE-owned de-normalize + unpatch seam has recovered
//! the true raw 32-channel latent.

use mlx_gen::{PreviewSink, Result};
use mlx_rs::Array;

use crate::Flux2Vae;

/// Ordinary-least-squares map from the true raw 32-channel FLUX.2 VAE latent to RGB. Fit on eight
/// diverse real-weight FLUX.2 Klein renders and measured on four disjoint prompt/seed holdouts,
/// all 256² with eight flow-Euler steps. Targets are 8×8-average-pooled native VAE decodes.
///
/// Fit R² `(R,G,B) = (0.77472, 0.77362, 0.74224)`, overall `0.76409`; holdout R²
/// `(0.64247, 0.71049, 0.72381)`, overall `0.69504`. The transform preserves coarse palette and
/// composition but intentionally omits decoder detail. Reproduce with `tests/fit_preview_rgb.rs`.
///
/// Lens reuse is grounded in tensor values, not Rust type lineage: all 251 VAE tensors (84,046,371
/// values) in the cached Lens base/turbo f32 checkpoints round exactly to the cached FLUX.2 bf16
/// values. The separately downloaded Ideogram 4 q4 VAE has all 250 learned tensors byte-identical
/// to FLUX.2; only the unused `bn.num_batches_tracked` scalar's integer dtype differs. Its distinct
/// `(ph,pw,c)` unpack path is covered here, while full-provider runtime evidence remains unavailable
/// because the multi-gigabyte Ideogram text/DiT components are not locally materialized.
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
const RGB_BIAS: [f32; 3] = [0.440_938_92, 0.424_318_4, 0.409_667_16];

fn project_raw_nhwc(latents: &Array) -> Result<mlx_gen::Image> {
    let nchw = latents.transpose_axes(&[0, 3, 1, 2])?;
    mlx_gen::preview::project_latents(&nchw, &RGB_FACTORS, RGB_BIAS)
}

/// Emit one FLUX/Lens packed-token latent using the FLUX channel-major patch order.
#[allow(clippy::too_many_arguments)]
pub fn emit_flux_preview(
    sink: &PreviewSink,
    counter: &mlx_gen::preview::PreviewCounter,
    sigmas: &[f32],
    sigma: f32,
    packed_tokens: &Array,
    grid_h: i32,
    grid_w: i32,
    vae: &Flux2Vae,
) {
    if !sink.is_active() {
        return;
    }
    mlx_gen::preview::emit_preview(sink, counter, sigmas, sigma, || {
        let shape = packed_tokens.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[1] != grid_h * grid_w || shape[2] != 128 {
            return Err(mlx_gen::Error::Msg(format!(
                "FLUX preview latent must have shape [1, {}, 128], got {shape:?}",
                grid_h * grid_w
            )));
        }
        let packed = packed_tokens.reshape(&[1, grid_h, grid_w, 128])?;
        project_raw_nhwc(&vae.unpack_flux_packed_latents(&packed)?)
    });
}

/// Emit one Ideogram packed-token latent using its patch-major `(ph,pw,c)` order.
#[allow(clippy::too_many_arguments)]
pub fn emit_ideogram_preview(
    sink: &PreviewSink,
    counter: &mlx_gen::preview::PreviewCounter,
    sigmas: &[f32],
    sigma: f32,
    packed_tokens: &Array,
    grid_h: i32,
    grid_w: i32,
    vae: &Flux2Vae,
) {
    if !sink.is_active() {
        return;
    }
    mlx_gen::preview::emit_preview(sink, counter, sigmas, sigma, || {
        project_raw_nhwc(&vae.unpack_ideogram_packed_latents(packed_tokens, grid_h, grid_w)?)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_fit_has_one_row_per_true_latent_channel() {
        assert_eq!(RGB_FACTORS.len(), 32);
        assert!(RGB_FACTORS.iter().flatten().all(|value| value.is_finite()));
        assert!(RGB_BIAS.iter().all(|value| value.is_finite()));
    }
}
