use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen_minimax_h3::upscale_prototype::{
    recipe_sigma, resize_axis, GuideLayout, LatentUpscaler,
};

fn compare(actual: &Tensor, expected: &Tensor, tolerance: f32) {
    assert_eq!(actual.dims(), expected.dims());
    let a = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f32::max);
    assert!(
        diff <= tolerance,
        "max absolute difference {diff} exceeds {tolerance}"
    );
}

#[test]
fn upstream_nontrivial_network_and_interpolation() {
    let mut fixture = candle_gen::candle_core::safetensors::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/upscale_network.safetensors"),
        &Device::Cpu,
    )
    .unwrap();
    let x = fixture.remove("input").unwrap();
    let expected = fixture.remove("output").unwrap();
    let resized = fixture.remove("resized").unwrap();
    let weights = fixture
        .into_iter()
        .filter_map(|(k, v)| k.strip_prefix("weight.").map(|s| (s.to_owned(), v)))
        .collect();
    let net = LatentUpscaler::from_weights(weights, DType::F32).unwrap();
    compare(&net.forward_normalized(&x, 4, 6).unwrap(), &expected, 3e-5);
    let fixture = candle_gen::candle_core::safetensors::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/upscale_network.safetensors"),
        &Device::Cpu,
    )
    .unwrap();
    let source =
        candle_gen_minimax_h3::upscale_prototype::normalize_vae_raw(&fixture["boundary.raw"])
            .unwrap();
    compare(&source, &fixture["boundary.normalized"], 1e-6);
    let enlarged = net.upscale_vae_raw(&fixture["boundary.raw"], 4, 6).unwrap();
    compare(&enlarged, &fixture["boundary.upscale"], 3e-5);
    let wrong = net.upscale_latents(&fixture["boundary.raw"], 4, 6).unwrap();
    let difference = (&wrong - &enlarged)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(difference > 0.05, "raw-domain mutation must disagree");
    compare(
        &resize_axis(&resize_axis(&x, 3, 4).unwrap(), 4, 6).unwrap(),
        &resized,
        1e-6,
    );
}

#[test]
fn actual_upstream_strided_layout_and_source_bicubic_guide() {
    let fixture = candle_gen::candle_core::safetensors::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/upscale_network.safetensors"),
        &Device::Cpu,
    )
    .unwrap();
    let layout = GuideLayout::new(3, 12, 36, 64, 65, true).unwrap();
    let positions = Tensor::from_vec(
        layout
            .positions
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        (layout.positions.len(), 3),
        &Device::Cpu,
    )
    .unwrap()
    .to_dtype(DType::F32)
    .unwrap();
    compare(
        &positions,
        &fixture["layout.positions"].to_dtype(DType::F32).unwrap(),
        1e-5,
    );
    assert_eq!(
        layout
            .video_indices
            .iter()
            .map(|&i| i as i64)
            .collect::<Vec<_>>(),
        fixture["layout.video_indices"].to_vec1::<i64>().unwrap()
    );
    assert_eq!(
        layout
            .audio_indices
            .iter()
            .map(|&i| i as i64)
            .collect::<Vec<_>>(),
        fixture["layout.audio_indices"].to_vec1::<i64>().unwrap()
    );
    let updates = fixture["layout.img_update"].to_vec1::<i64>().unwrap();
    assert!(updates[..layout.guide_rows].iter().all(|&v| v == 0));
    assert!(updates[layout.guide_rows..].iter().all(|&v| v == 1));
    let guide =
        candle_gen_minimax_h3::upscale_prototype::source_guide_pixels(&fixture["guide.rgb"], 8, 12)
            .unwrap();
    compare(&guide, &fixture["guide.output"], 1e-6);
    // Merely keeping original pixels (or using an enlarged latent guide) is
    // shape-correct yet fails the reference decode->resize->reencode boundary.
    let wrong = (&fixture["guide.rgb"] - &fixture["guide.output"])
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(wrong > 0.01);
}

#[test]
fn partial_schedule_rejects_raw_sigma_mutation_and_zero_bypasses() {
    let fixture = candle_gen::candle_core::safetensors::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/upscale_network.safetensors"),
        &Device::Cpu,
    )
    .unwrap();
    let denoise = fixture["schedule.denoise"].to_vec1::<f32>().unwrap();
    let sigma = fixture["schedule.sigma"].to_vec1::<f32>().unwrap();
    for (fraction, reference) in denoise.into_iter().zip(sigma) {
        assert!((recipe_sigma(fraction).unwrap().unwrap() - reference).abs() < 1e-6);
    }
    assert!((recipe_sigma(0.1).unwrap().unwrap() - 0.5714286).abs() < 1e-7);
    assert_eq!(recipe_sigma(0.2).unwrap(), Some(0.75));
    assert!((recipe_sigma(0.1).unwrap().unwrap() - 0.1).abs() > 0.4);
    assert_eq!(recipe_sigma(0.0).unwrap(), None);
    assert!(recipe_sigma(f32::NAN).is_err());
    assert!(recipe_sigma(-0.1).is_err());
}

#[test]
fn guide_rows_overlap_every_time_without_stretching_or_audio_updates() {
    let layout = GuideLayout::new(3, 12, 36, 64, 65, true).unwrap();
    assert_eq!(layout.guide_rows, 12 * 9 * 16);
    assert_eq!(layout.update_indices.len(), 12 * 18 * 32);
    assert_eq!(layout.positions.len(), 3 + 1728 + 130 + 6912);
    let target_start = 3 + 1728 + 130;
    // Literal first/last retained grid entries: taking full target grid fails
    // both row-count and positions; resizing a half-grid spans different endpoints.
    for t in 0..12 {
        for r in 0..9 {
            for c in 0..16 {
                assert_eq!(
                    layout.positions[3 + t * 144 + r * 16 + c],
                    layout.positions[target_start + t * 576 + r * 2 * 32 + c * 2]
                );
            }
        }
    }
    assert_eq!(layout.classes[3], 2);
    assert!(layout
        .audio_indices
        .iter()
        .all(|&i| layout.classes[i as usize] == 3));
    assert!(layout
        .update_indices
        .iter()
        .all(|&i| i as usize >= layout.guide_rows));
    assert!(GuideLayout::new(3, 12, 18, 32, 65, true).is_err());
    let unguided = GuideLayout::new(3, 12, 36, 64, 65, false).unwrap();
    assert_eq!(unguided.guide_rows, 0);
    assert_eq!(unguided.update_indices.len(), 6912);
}
#[test]
#[cfg(feature = "cuda")]
#[ignore = "manual bounded CUDA precision fixture; set CUDA_VISIBLE_DEVICES explicitly"]
fn bf16_refinement_retains_f32_scheduler_state() {
    use crate::common::{dit_fixture_config, weights, Golden, DIT_FIXTURE};
    use candle_gen_minimax_h3::{upscale_prototype::refine_once, MiniMaxH3Dit};
    let cfg = dit_fixture_config();
    let fixture = Golden::load(DIT_FIXTURE);
    assert!(
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok(),
        "explicit device assignment required"
    );
    let device = Device::new_cuda(0).unwrap();
    let model = weights(
        fixture
            .model_map(&["src.", "in.", "out.", "layout."])
            .into_iter()
            .map(|(key, value)| {
                (
                    key,
                    value
                        .to_device(&device)
                        .unwrap()
                        .to_dtype(DType::BF16)
                        .unwrap(),
                )
            })
            .collect(),
    );
    let dit = MiniMaxH3Dit::from_weights(&model, &cfg, &device, DType::BF16).unwrap();
    let context = Tensor::zeros((1, 3, cfg.text_dim), DType::BF16, &device).unwrap();
    let enlarged = Tensor::zeros((1, 24, 12, 4, 4), DType::F32, &device).unwrap();
    let noise = Tensor::ones(enlarged.shape(), DType::F32, &device).unwrap();
    let output = refine_once(&dit, &context, &enlarged, None, &noise, None, 0.5714286).unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.dims(), enlarged.dims());
}

#[test]
fn pinned_decoder_stitch_uses_already_blended_canvas_strips() {
    use candle_gen_minimax_h3::spatial_tiling::{BoundedStitch, TilePlan};
    let fixture = candle_gen::candle_core::safetensors::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/upscale_decode_stitch.safetensors"),
        &Device::Cpu,
    )
    .unwrap();
    let rows = TilePlan::split(5, 4, 2, 1).unwrap();
    let cols = TilePlan::split(8, 4, 2, 1).unwrap();
    let output =
        candle_gen_minimax_h3::upscale_prototype::stitch_pinned_decode(&rows, &cols, |i, j| {
            Ok(fixture[&format!("tile.{i}.{j}")].clone())
        })
        .unwrap();
    compare(&output, &fixture["output"], 1e-6);
    let mut original =
        BoundedStitch::new(rows.len(), cols.len(), &rows.overlaps, &cols.overlaps).unwrap();
    for i in 0..rows.len() {
        for j in 0..cols.len() {
            original
                .push(fixture[&format!("tile.{i}.{j}")].clone())
                .unwrap();
        }
    }
    let original = original.finish().unwrap();
    let difference = (&original - &output)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        difference > 0.1,
        "original-neighbour mutation must fail pinned decoder parity"
    );
}
