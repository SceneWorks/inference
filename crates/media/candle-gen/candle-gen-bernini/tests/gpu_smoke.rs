//! GPU smoke for the `bernini_renderer` raw t2v render (sc-10994). `#[ignore]`d — needs a dual-expert
//! Wan2.2-T2V-A14B candle snapshot on disk + a CUDA device. The Bernini renderer IS Wan2.2-T2V-A14B
//! (finetuned), so a stock `SceneWorks/wan2.2-t2v-a14b-candle` tier exercises the full renderer
//! machinery (UMT5 → dual-expert boundary switch → APG guidance → z16 VAE decode) end to end.
//!
//! ```sh
//! # point at a diffusers-layout dual-expert snapshot dir (text_encoder/ transformer/ transformer_2/ vae/ tokenizer/)
//! export SCENEWORKS_BERNINI_SNAPSHOT="E:/huggingface/hub/models--SceneWorks--wan2.2-t2v-a14b-candle/snapshots/<hash>/q4"
//! export SCENEWORKS_BERNINI_OUT="D:/tmp/bernini_render.png"   # optional; defaults to the temp dir
//! # Optional exact-shape A/B: both files must target the named Wan expert.
//! export SCENEWORKS_BERNINI_ADAPTER_HIGH="D:/models/high.safetensors"
//! export SCENEWORKS_BERNINI_ADAPTER_LOW="D:/models/low.safetensors"
//! # vcvars 14.44 + CUDA_COMPUTE_CAP=120, then:
//! cargo test -p candle-gen-bernini --release --features cuda gpu_smoke_raw_render -- --ignored --nocapture
//! ```

use std::time::Instant;

use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, LoadSpec, MoeExpert, Progress,
    WeightsSource,
};

#[test]
#[ignore = "GPU smoke: needs a dual-expert Wan2.2-T2V-A14B candle snapshot + CUDA device"]
fn gpu_smoke_raw_render() {
    let snapshot = std::env::var("SCENEWORKS_BERNINI_SNAPSHOT").expect(
        "set SCENEWORKS_BERNINI_SNAPSHOT to a dual-expert Wan2.2-T2V-A14B candle snapshot dir",
    );
    let out_path = std::env::var("SCENEWORKS_BERNINI_OUT").unwrap_or_else(|_| {
        std::env::temp_dir()
            .join("bernini_render.png")
            .to_string_lossy()
            .into_owned()
    });

    let spec = LoadSpec::new(WeightsSource::Dir(snapshot.clone().into()));
    let gen = candle_gen_bernini::provider_registry()
        .unwrap()
        .load("bernini_renderer", &spec)
        .expect("load bernini_renderer");
    eprintln!("[bernini-gpu] loaded bernini_renderer from {snapshot}");

    // A single-frame (t2i) raw render: 256x256, a handful of steps so the dual-expert boundary (0.875)
    // is crossed (high→low). frames=1 ⇒ 1 latent frame ⇒ a still image.
    let req = GenerationRequest {
        prompt: "a red panda sitting on a mossy log in a sunlit forest, cinematic".into(),
        width: 256,
        height: 256,
        frames: Some(1),
        steps: Some(12),
        seed: Some(42),
        sampler: Some("uni_pc".into()),
        ..Default::default()
    };

    let mut last_step = 0u32;
    let mut on_progress = |p: Progress| {
        if let Progress::Step { current, total } = p {
            if current == 1 || current == total || current % 4 == 0 {
                eprintln!("[bernini-gpu] step {current}/{total}");
            }
            last_step = current;
        } else {
            eprintln!("[bernini-gpu] {p:?}");
        }
    };

    let t0 = Instant::now();
    let out = gen
        .generate(&req, &mut on_progress)
        .expect("bernini_renderer generate");
    let dt = t0.elapsed();

    let image = match out {
        GenerationOutput::Images(mut imgs) => imgs.remove(0),
        GenerationOutput::Video { mut frames, .. } => frames.remove(0),
        _ => panic!("expected a visual output, got a non-visual output"),
    };
    assert!(
        image.width >= 16 && image.height >= 16,
        "decoded frame has real dims"
    );
    assert_eq!(
        image.pixels.len(),
        image.width as usize * image.height as usize * 3,
        "decoded frame is a full RGB8 buffer"
    );
    // Not all-constant (a black/uniform frame would be a broken decode).
    let first = image.pixels[0];
    let varied = image.pixels.iter().any(|&p| p != first);
    assert!(varied, "decoded frame must not be a single flat color");

    // Compile-only in ordinary CI. When both expert adapters are explicitly supplied, reload after
    // dropping the baseline so GPU residency stays sequential, then render the exact same request once
    // per expert. Independent divergence proves both expert-tagged selections flow through the full
    // Bernini provider rather than one valid expert masking a silently ignored peer.
    if let (Ok(high), Ok(low)) = (
        std::env::var("SCENEWORKS_BERNINI_ADAPTER_HIGH"),
        std::env::var("SCENEWORKS_BERNINI_ADAPTER_LOW"),
    ) {
        drop(gen);
        for (label, path, expert) in [
            ("high", high, MoeExpert::High),
            ("low", low, MoeExpert::Low),
        ] {
            let mut adapted_spec = LoadSpec::new(WeightsSource::Dir(snapshot.clone().into()));
            adapted_spec.adapters =
                vec![AdapterSpec::new(path.into(), 1.0, AdapterKind::Lora).with_moe_expert(expert)];
            let adapted = candle_gen_bernini::provider_registry()
                .unwrap()
                .load("bernini_renderer", &adapted_spec)
                .unwrap_or_else(|error| panic!("load {label}-adapted bernini_renderer: {error}"));
            let mut adapted_last_step = 0;
            let adapted_out = adapted
                .generate(&req, &mut |progress| {
                    if let Progress::Step { current, .. } = progress {
                        adapted_last_step = current;
                    }
                })
                .unwrap_or_else(|error| panic!("{label}-adapted Bernini render: {error}"));
            let adapted_image = match adapted_out {
                GenerationOutput::Images(mut images) => images.remove(0),
                GenerationOutput::Video { mut frames, .. } => frames.remove(0),
                _ => panic!("expected a {label}-adapted visual output"),
            };
            assert_eq!(
                (adapted_image.width, adapted_image.height),
                (image.width, image.height),
                "{label}-expert adapter A/B must preserve request shape"
            );
            assert_ne!(
                adapted_image.pixels, image.pixels,
                "selected Bernini {label}-expert adapter must change output"
            );
            assert_eq!(adapted_last_step, req.steps.unwrap());
        }
    }

    image::save_buffer(
        &out_path,
        &image.pixels,
        image.width,
        image.height,
        image::ColorType::Rgb8,
    )
    .expect("save png");

    eprintln!(
        "[bernini-gpu] DONE raw t2v render {}x{} in {:.1}s ({} steps) -> {out_path}",
        image.width,
        image.height,
        dt.as_secs_f32(),
        last_step,
    );
}
