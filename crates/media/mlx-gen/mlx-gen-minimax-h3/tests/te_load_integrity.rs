//! Scoped real-weight regression for sc-23053. Run once per bf16/q4/q8 component with
//! MINIMAX_H3_SNAPSHOT (base snapshot), MINIMAX_H3_TE (tier component), and
//! MINIMAX_H3_IMAGE (source image). Optional MINIMAX_H3_PROMPT_FILE replays a saved prompt.
//! The first and second forwards exercise fresh-load and already-loaded embedding views;
//! Set MINIMAX_H3_RESIDENT=1 to exercise the resident lazy-load constructor as well.
//! This does not claim to evict the operating system's file cache.
use mlx_gen_boogu::VisionTower;
use mlx_gen_minimax_h3::text_encoder::{
    self as te, MiniMaxH3TeConfig, MiniMaxH3TextEncoder, MiniMaxH3Tokenizer, LM_PREFIX,
    VISION_PREFIX,
};
use mlx_rs::{Array, Dtype};

fn assert_context(context: &Array, sequence: i32) {
    assert_eq!(context.shape(), &[1, sequence, 5120]);
    assert!(
        te::inspect_conditioning("sc-23053 real-weight regression", context)
            .unwrap()
            .is_none()
    );
}

#[test]
#[ignore = "needs MiniMax-H3 weights, source image, and exclusive Metal access"]
fn fresh_and_warm_text_and_grounded_contexts_match() {
    let root = std::path::PathBuf::from(std::env::var("MINIMAX_H3_SNAPSHOT").unwrap());
    let component = std::path::PathBuf::from(std::env::var("MINIMAX_H3_TE").unwrap());
    let prompt = std::env::var("MINIMAX_H3_PROMPT_FILE")
        .map(|p| std::fs::read_to_string(p).unwrap())
        .unwrap_or_else(|_| "A heron turns its head beside a quiet lake.".into());
    let tok = MiniMaxH3Tokenizer::from_snapshot(&root).unwrap();
    let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
    let component_config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(component.join("config.json")).unwrap()).unwrap();
    let tier_bits = component_config["quantization"]["bits"]
        .as_i64()
        .map(|n| n as i32);
    let im = image::open(std::env::var("MINIMAX_H3_IMAGE").unwrap())
        .unwrap()
        .to_rgb8();
    let im = mlx_gen::media::Image {
        width: im.width(),
        height: im.height(),
        pixels: im.into_raw(),
    };
    let fitted = mlx_gen_minimax_h3::keyframe::fit_keyframes(&[&im], 768, 1024).unwrap();
    let grounded = {
        let w = te::map_shards(&root.join("text_encoder"), true).unwrap();
        let tower =
            VisionTower::from_weights(&w, te::minimax_h3_vision_config(), VISION_PREFIX, 64)
                .unwrap();
        let g = te::run_vision(&tower, &[&fitted[0]]).unwrap();
        let mut outputs: Vec<&Array> = g.embeds.iter().collect();
        outputs.extend(g.deepstack.iter().flatten());
        mlx_rs::transforms::eval(outputs).unwrap();
        g
    };
    mlx_gen::residency::drain_allocator_cache();
    for route in ["t2va", "fl2va"] {
        let resident = std::env::var_os("MINIMAX_H3_RESIDENT").is_some();
        let encoder = if resident {
            let w = te::map_shards(&component, false).unwrap();
            MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg).unwrap()
        } else {
            let mut encoder =
                MiniMaxH3TextEncoder::from_dir_deferred(&component, LM_PREFIX, &cfg).unwrap();
            encoder
                .set_block_window(2, mlx_gen::CancelFlag::default())
                .unwrap();
            encoder
        };
        assert_eq!(encoder.token_table_is_quantized(), tier_bits.is_some());
        if resident {
            assert_eq!(encoder.packed_bits().unwrap(), tier_bits);
        }
        let expected_layers = if resident { 50 } else { 0 };
        assert_eq!(encoder.resident_layers(), expected_layers);
        let (ids, mask) = if route == "t2va" {
            tok.encode_prompt(&prompt).unwrap()
        } else {
            let (ids, mask, _) = tok.encode_fl2va(&prompt, &grounded.counts).unwrap();
            (ids, mask)
        };
        let forward = || {
            if route == "t2va" {
                encoder.forward(&ids, &mask)
            } else {
                encoder.forward_with_images(
                    &ids,
                    &mask,
                    &grounded.embeds,
                    &grounded.deepstack,
                    &grounded.grids,
                )
            }
        };
        let first = forward().unwrap();
        assert_context(&first, ids.shape()[1]);
        let second = forward().unwrap();
        assert_context(&second, ids.shape()[1]);
        let error = mlx_rs::ops::max(
            mlx_rs::ops::abs(first.subtract(&second).unwrap()).unwrap(),
            None,
        )
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .try_item::<f32>()
        .unwrap();
        assert_eq!(error, 0., "fresh/warm {route} changed conditioning");
        assert_eq!(encoder.resident_layers(), expected_layers);
        eprintln!("sc-23053 component={} route={route} shape={:?} fresh_warm_max_delta={error} tier_bits={tier_bits:?} token_table_quantized={} resident_layers={} peak_bytes={}",
            component.display(), first.shape(), encoder.token_table_is_quantized(), encoder.resident_layers(), mlx_rs::memory::get_peak_memory());
        drop((first, second, encoder));
        mlx_gen::residency::drain_allocator_cache();
    }
}
