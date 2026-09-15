//! Per-step previews for SANA's native 32-channel DC-AE latent space.

use mlx_gen::PreviewSink;
use mlx_rs::ops::multiply;
use mlx_rs::Array;

/// Base SANA ordinary-least-squares map from its scaled DC-AE latent to RGB.
///
/// Fit on four diverse real-weight Base renders and measured on two disjoint prompt/seed holdouts,
/// all 256x256 with eight static-shift-3 flow-Euler steps at true CFG 4.5. Targets are native DC-AE
/// decodes average-pooled by `SPATIAL_SCALE = 32`. Fit R² `(R,G,B) = (0.94379, 0.94447, 0.95035)`,
/// overall `0.94601`; holdout R² `(0.89018, 0.90411, 0.90728)`, overall `0.89941`.
///
/// Snapshot: `SceneWorks/Sana_1600M_1024px_mlx` revision
/// `ba22f36ba3d1feb78c9a1055a808ad68eda8adf8`, Q4 tier. Its DC-AE file is 1,249,044,836 bytes,
/// SHA-256 `15a4b09e56d95b768a0ec9da50b702e21d920333fc9b3480d66bb5c7fad9d87f`.
/// The upstream Sana-1.0 DC-AE config at revision
/// `d1b54936033cd7d45410ecadd692c5c502a19a38` has SHA-256
/// `4e5669aa03caa77615b73b21c9347c384cb11124a9590683e32a510df0176eb4`.
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
const BASE_RGB_BIAS: [f32; 3] = [0.467_3, 0.437_615_22, 0.414_471_12];

/// SANA-Sprint ordinary-least-squares map from its scaled DC-AE latent to RGB.
///
/// The four-fit/two-holdout producer uses four SCM steps and embedded guidance 4.5. Fit R²
/// `(R,G,B) = (0.96315, 0.96731, 0.96115)`, overall `0.96439`; holdout R²
/// `(0.94540, 0.90385, 0.93090)`, overall `0.93066`.
///
/// Snapshot: `SceneWorks/Sana_Sprint_1.6B_1024px_mlx` revision
/// `0b0d18484cac2fb515e76d25a09a5911ae4ab58e`, Q4 tier. Its DC-AE file is 1,249,044,836 bytes,
/// SHA-256 `dfd991d1b54ffabf22745c5885589d8f2a7bc59930d95d92bd741c4fc64454bb`.
/// This differs from Base, so the fits are intentionally not shared. Official configs have the same
/// consumed architecture/scaling fields but identify DC-AE Sana 1.0 (Base) versus 1.1 (Sprint).
/// The Sprint upstream config at revision `19683c58b7ea290e55cedd8950ae1d86ada7ef96`
/// has SHA-256 `ba6f3d3e44d75d44fdd3760097c069173b5b925e6d14604d5d3582628d09cca6`.
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
const SPRINT_RGB_BIAS: [f32; 3] = [0.457_922_9, 0.428_063_12, 0.399_165_84];

/// Emit one native spatial `[1, 32, h, w]` SANA denoise latent for an actual outer step.
///
/// `latent_scale` converts the observed sampler state into the shared DC-AE denoise space. Base SANA
/// supplies `1`; Sprint supplies `1 / sigma_data` because its SCM loop carries an extra prior-space
/// scale. Layout validation, optional scaling, and projection stay inside the best-effort closure.
/// An inert sink therefore returns before any tensor operation, and decorative errors cannot fail
/// or retry generation.
#[allow(clippy::too_many_arguments)]
fn emit_preview(
    sink: &PreviewSink,
    counter: &mlx_gen::preview::PreviewCounter,
    schedule: &[f32],
    position: f32,
    latents: &Array,
    latent_scale: f32,
    factors: &[[f32; 3]; 32],
    bias: [f32; 3],
) {
    if !sink.is_active() {
        return;
    }
    mlx_gen::preview::emit_preview(sink, counter, schedule, position, || {
        let shape = latents.shape();
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 32 {
            return Err(mlx_gen::Error::Msg(format!(
                "SANA preview latent must have shape [1, 32, h, w], got {shape:?}"
            )));
        }
        let scaled;
        let latents = if latent_scale == 1.0 {
            latents
        } else {
            scaled = multiply(latents, Array::from_slice(&[latent_scale], &[1]))?;
            &scaled
        };
        mlx_gen::preview::project_latents(latents, factors, bias)
    });
}

/// Emit a Base SANA latent, which already lives in scaled DC-AE denoise space.
pub fn emit_base_preview(
    sink: &PreviewSink,
    counter: &mlx_gen::preview::PreviewCounter,
    schedule: &[f32],
    position: f32,
    latents: &Array,
) {
    emit_preview(
        sink,
        counter,
        schedule,
        position,
        latents,
        1.0,
        &BASE_RGB_FACTORS,
        BASE_RGB_BIAS,
    );
}

/// Emit a Sprint latent after removing its additional SCM prior-space `sigma_data` scale.
pub fn emit_sprint_preview(
    sink: &PreviewSink,
    counter: &mlx_gen::preview::PreviewCounter,
    schedule: &[f32],
    position: f32,
    latents: &Array,
    inverse_sigma_data: f32,
) {
    emit_preview(
        sink,
        counter,
        schedule,
        position,
        latents,
        inverse_sigma_data,
        &SPRINT_RGB_FACTORS,
        SPRINT_RGB_BIAS,
    );
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[test]
    fn native_layout_projects_at_actual_latent_resolution() {
        let schedule = [1.0, 0.5, 0.0];
        let counter = mlx_gen::preview::PreviewCounter::new(&schedule);
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));
        let latents = Array::zeros::<f32>(&[1, 32, 3, 5]).unwrap();

        emit_base_preview(&sink, &counter, &schedule, schedule[0], &latents);

        let frames = frames.lock().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].current, frames[0].total), (1, 2));
        assert_eq!((frames[0].image.width, frames[0].image.height), (5, 3));
    }

    #[test]
    fn shipped_resolution_examples_are_self_describing() {
        for (image_edge, latent_edge) in [(256_u32, 8_i32), (1024, 32)] {
            let schedule = [1.0, 0.0];
            let counter = mlx_gen::preview::PreviewCounter::new(&schedule);
            let frames = Arc::new(Mutex::new(Vec::new()));
            let captured = Arc::clone(&frames);
            let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));
            let latents = Array::zeros::<f32>(&[1, 32, latent_edge, latent_edge]).unwrap();

            emit_base_preview(&sink, &counter, &schedule, schedule[0], &latents);

            let frames = frames.lock().unwrap();
            assert_eq!(frames.len(), 1);
            assert_eq!(
                (frames[0].image.width, frames[0].image.height),
                (image_edge / 32, image_edge / 32)
            );
        }
    }

    #[test]
    fn malformed_layout_is_decorative_and_consumes_position() {
        let schedule = [1.0, 0.5, 0.0];
        let counter = mlx_gen::preview::PreviewCounter::new(&schedule);
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));
        let malformed = Array::zeros::<f32>(&[1, 31, 2, 2]).unwrap();
        let valid = Array::zeros::<f32>(&[1, 32, 2, 2]).unwrap();

        emit_base_preview(&sink, &counter, &schedule, schedule[0], &malformed);
        emit_base_preview(&sink, &counter, &schedule, schedule[0], &valid);
        emit_base_preview(&sink, &counter, &schedule, schedule[1], &valid);

        let frames = frames.lock().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].current, frames[0].total), (2, 2));
    }

    #[test]
    fn inert_sink_avoids_layout_validation_scaling_and_tensor_work() {
        let schedule = [1.0, 0.0];
        let counter = mlx_gen::preview::PreviewCounter::new(&schedule);
        let malformed = Array::zeros::<f32>(&[9]).unwrap();

        emit_sprint_preview(
            &PreviewSink::default(),
            &counter,
            &schedule,
            schedule[0],
            &malformed,
            f32::NAN,
        );

        assert_eq!(counter.next(&schedule, schedule[0]), Some(1));
    }

    #[test]
    fn committed_fit_has_one_finite_row_per_latent_channel() {
        for (factors, bias) in [
            (&BASE_RGB_FACTORS, &BASE_RGB_BIAS),
            (&SPRINT_RGB_FACTORS, &SPRINT_RGB_BIAS),
        ] {
            assert_eq!(factors.len(), 32);
            assert!(factors.iter().flatten().all(|value| value.is_finite()));
            assert!(bias.iter().all(|value| value.is_finite()));
        }
    }
}
