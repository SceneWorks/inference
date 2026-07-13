//! SD3.5 **img2img / `Reference`** real-weight GPU validation (sc-11784) — env-driven, `#[ignore]`d
//! integration tests that drive the REGISTERED candle SD3.5 generators
//! (`provider_registry().load("sd3_5_*", …)`) through a [`Conditioning::Reference`] against the deployed
//! hardware (an SD3.5 diffusers snapshot + a source image). The candle/CUDA parity of the mlx-gen-sd3
//! `denoise_img2img_cfg` lane (sc-10189) — one test per variant: true-CFG **Large** + **Medium**, and
//! the distilled few-step **Turbo**.
//!
//! **Gate.** Img2img should (a) actually edit toward the prompt, and (b) honor the fork
//! strength convention — **higher strength ⇒ closer to the source** ([`crate::pipeline::init_time_step`],
//! the INVERSE of the SDXL knob). Measured against the `strength = 1.0` output (the empty-loop source
//! VAE round-trip baseline):
//!  - **low** strength (heavy regeneration) departs from the round-trip the most,
//!  - the diff-vs-round-trip is monotone the INVERSE of SDXL (s=0.25 > s=0.6 > s=0.9),
//!  - two distinct **prompts** at the same seed/strength diverge — the edit follows the text,
//!  - the pure-txt2img path (no `Reference`) diverges more than the heaviest img2img.
//!
//! Run (after deploying a snapshot + a source into local dirs), e.g. for Large:
//! ```text
//! set CUDA_VISIBLE_DEVICES=1
//! set SD35_LARGE_PATH=...\stable-diffusion-3.5-large   # text_encoder*/ transformer/ vae/ + tokenizers
//! set SD35_IMG2IMG_SRC=...\source.ppm                  # a source image (P6 binary PPM)
//! set SD35_IMG2IMG_OUT=...\out
//! cargo test -p candle-gen-sd3 --features cuda --release \
//!   img2img_validate::real_weight_large -- --ignored --nocapture
//! ```

use candle_gen::gen_core::{
    self, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress,
    WeightsSource,
};
use candle_gen::testkit::{env_path, mean_abs_diff, read_ppm, write_ppm};

/// Per-variant sweep config: the registered id, its snapshot env var, an output-name tag, and whether
/// the variant runs real CFG (Large/Medium) or the distilled CFG-free loop (Turbo).
struct Sweep {
    model_id: &'static str,
    snapshot_env: &'static str,
    tag: &'static str,
    /// Real-CFG variants pass `guidance` + a `negative_prompt`; distilled Turbo passes neither (the
    /// descriptor rejects them). `None` ⇒ distilled; `Some(scale)` ⇒ CFG scale.
    guidance: Option<f32>,
    /// Inference steps to request. A few past the distilled default so the reduced img2img tail still
    /// has room to denoise at the higher (source-preserving) strengths.
    steps: u32,
}

/// The shared strength-sweep + monotonicity gate, driven through the REGISTERED generator for `sweep`.
fn run_sweep(sweep: &Sweep) {
    let out_dir = env_path("SD35_IMG2IMG_OUT");
    std::fs::create_dir_all(&out_dir).ok();

    let source = read_ppm(&env_path("SD35_IMG2IMG_SRC"));
    println!(
        "[{}] source {}x{}; resolving registered {} …",
        sweep.tag, source.width, source.height, sweep.model_id
    );

    let spec = LoadSpec::new(WeightsSource::Dir(env_path(sweep.snapshot_env)));
    let t0 = std::time::Instant::now();
    let model = crate::provider_registry()
        .expect("build SD3 provider registry")
        .load(sweep.model_id, &spec)
        .unwrap_or_else(|e| panic!("load {}: {e}", sweep.model_id));
    println!(
        "  resolved id={} backend={} accepts(Reference)={} in {:?}",
        model.descriptor().id,
        model.descriptor().backend,
        model
            .descriptor()
            .capabilities
            .accepts(gen_core::ConditioningKind::Reference),
        t0.elapsed()
    );

    // Fit the source to a clean multiple-of-16 render size (the validate floor).
    let width = source.width - (source.width % 16);
    let height = source.height - (source.height % 16);

    let gen = |strength: Option<f32>, prompt: &str| -> Image {
        let mut req = GenerationRequest {
            prompt: prompt.to_owned(),
            width,
            height,
            steps: Some(sweep.steps),
            seed: Some(12345),
            ..Default::default()
        };
        if let Some(s) = strength {
            req.conditioning = vec![Conditioning::Reference {
                image: source.clone(),
                strength: Some(s),
            }];
        }
        // Real-CFG variants exercise guidance + a negative prompt; distilled Turbo neither.
        if let Some(scale) = sweep.guidance {
            req.guidance = Some(scale);
            req.negative_prompt = Some("blurry, low quality, distorted".to_owned());
        }
        let mut noop = |_p: Progress| {};
        match model
            .generate(&req, &mut noop)
            .unwrap_or_else(|e| panic!("generate (s={strength:?} \"{prompt}\"): {e}"))
        {
            GenerationOutput::Images(mut imgs) => imgs.pop().expect("one image"),
            GenerationOutput::Video { .. } => panic!("expected images, got video"),
        }
    };

    let prompt_a = "a watercolor painting, soft pastel colors, dreamy, artistic";
    let prompt_b =
        "an oil painting, dark dramatic chiaroscuro lighting, heavy impasto brushstrokes";

    // strength 1.0 ⇒ start == steps ⇒ empty loop ⇒ the source's VAE round-trip at the render size: the
    // "no-edit" baseline to measure structure preservation against.
    let t = std::time::Instant::now();
    let roundtrip = gen(Some(1.0), prompt_a);
    println!("  [roundtrip s=1.0] {:?}", t.elapsed());
    let p = |name: &str| out_dir.join(format!("sd3_{}_{name}", sweep.tag));
    write_ppm(&p("img2img_roundtrip.ppm"), &roundtrip);

    let out_regen = gen(Some(0.25), prompt_a);
    write_ppm(&p("img2img_s025.ppm"), &out_regen);
    let out_regen_b = gen(Some(0.25), prompt_b);
    write_ppm(&p("img2img_s025_b.ppm"), &out_regen_b);
    let out_edit = gen(Some(0.6), prompt_a);
    write_ppm(&p("img2img_s06.ppm"), &out_edit);
    let out_preserve = gen(Some(0.9), prompt_a);
    write_ppm(&p("img2img_s09.ppm"), &out_preserve);

    let d_regen = mean_abs_diff(&out_regen, &roundtrip);
    let d_edit = mean_abs_diff(&out_edit, &roundtrip);
    let d_preserve = mean_abs_diff(&out_preserve, &roundtrip);
    let d_prompt = mean_abs_diff(&out_regen, &out_regen_b);
    println!("=== sd3 {} img2img validation ===", sweep.tag);
    println!(
        "  diff vs source round-trip: s=0.25 {d_regen:.2}  s=0.6 {d_edit:.2}  s=0.9 {d_preserve:.2}"
    );
    println!("  prompt A-vs-B diff @ s=0.25: {d_prompt:.2}");
    println!("  outputs: {}", out_dir.display());

    // Gate 1: heavy regeneration (low strength) clearly departs from the source — img2img is wired.
    assert!(
        d_regen > 3.0,
        "low-strength regen diff {d_regen:.2} too small — img2img may not be wired"
    );
    // Gate 2: the edit follows the PROMPT — two distinct prompts at the same seed/strength diverge.
    assert!(
        d_prompt > 3.0,
        "prompt A-vs-B diff {d_prompt:.2} too small — the edit may ignore the prompt"
    );
    // Gate 3 (the correctness proof): the fork strength convention — monotone the INVERSE of SDXL, so a
    // lower strength diverges from the source MORE than a higher one.
    assert!(
        d_regen > d_edit && d_edit > d_preserve,
        "strength monotonicity broken (expected s=0.25 > s=0.6 > s=0.9): {d_regen:.2} / {d_edit:.2} / {d_preserve:.2}"
    );
    // Gate 4: the default-strength edit is a real (non-trivial) change from the source.
    assert!(
        d_edit > 1.0,
        "default-strength edit diff {d_edit:.2} too small"
    );

    // The pure-txt2img path still works through the same registered generator (no Reference ⇒ full
    // schedule from the max-σ prior) and diverges wildly from the source round-trip.
    let txt2img = gen(None, prompt_a);
    write_ppm(&p("txt2img.ppm"), &txt2img);
    let d_txt = mean_abs_diff(&txt2img, &roundtrip);
    println!("  txt2img (no Reference) diff vs round-trip: {d_txt:.2}");
    assert!(
        d_txt > d_regen,
        "txt2img should diverge more than s=0.25 img2img"
    );

    println!(
        "sd3 {} img2img validation PASS ✅ (eyeball the PPMs for edit quality)",
        sweep.tag
    );
}

/// **SD3.5 Large** (true-CFG, guidance + negative prompt): the flagship img2img strength sweep.
#[test]
#[ignore = "real-weight GPU validation; set SD35_LARGE_PATH/SD35_IMG2IMG_SRC/SD35_IMG2IMG_OUT"]
fn real_weight_large_img2img() {
    run_sweep(&Sweep {
        model_id: crate::MODEL_ID,
        snapshot_env: "SD35_LARGE_PATH",
        tag: "large",
        guidance: Some(4.0),
        steps: 28,
    });
}

/// **SD3.5 Medium** (MMDiT-X, true-CFG): the dual-attention img2img strength sweep.
#[test]
#[ignore = "real-weight GPU validation; set SD35_MEDIUM_PATH/SD35_IMG2IMG_SRC/SD35_IMG2IMG_OUT"]
fn real_weight_medium_img2img() {
    run_sweep(&Sweep {
        model_id: crate::MODEL_ID_MEDIUM,
        snapshot_env: "SD35_MEDIUM_PATH",
        tag: "medium",
        guidance: Some(4.5),
        steps: 40,
    });
}

/// **SD3.5 Large Turbo** (guidance-distilled, CFG-free): the few-step img2img strength sweep. NO
/// guidance / NO negative prompt (the descriptor rejects them). A few steps past the 4-step default so
/// the reduced img2img tail still denoises at the source-preserving strengths.
#[test]
#[ignore = "real-weight GPU validation; set SD35_TURBO_PATH/SD35_IMG2IMG_SRC/SD35_IMG2IMG_OUT"]
fn real_weight_turbo_img2img() {
    run_sweep(&Sweep {
        model_id: crate::MODEL_ID_TURBO,
        snapshot_env: "SD35_TURBO_PATH",
        tag: "turbo",
        guidance: None,
        steps: 8,
    });
}
