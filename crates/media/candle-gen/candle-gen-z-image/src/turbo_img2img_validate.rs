//! `z_image_turbo` **img2img / `Reference`** real-weight GPU validation (sc-11783) — an env-driven,
//! `#[ignore]`d integration test that drives the REGISTERED candle Turbo generator
//! (`provider_registry().load("z_image_turbo", …)`) through a [`Conditioning::Reference`] against the
//! deployed hardware (a Z-Image-Turbo snapshot + a source image). The CFG-free Turbo sibling of
//! [`crate::base_img2img_validate`] — same structure-preservation gates, but the distilled few-step loop
//! (no guidance, no negative prompt).
//!
//! **Gate.** Turbo img2img should (a) actually edit toward the prompt, and (b) honor the Z-Image
//! structure-preservation strength convention — **higher strength ⇒ closer to the source** (the fork's
//! `init_time_step`, the inverse of the SDXL knob). Measured against the `strength = 1.0` output (the
//! empty-loop source VAE round-trip baseline):
//!  - **low** strength (heavy regeneration) departs from the round-trip the most,
//!  - the diff-vs-round-trip is monotone the INVERSE of SDXL (s=0.25 > s=0.6 > s=0.9),
//!  - two distinct **prompts** at the same seed/strength diverge — the edit follows the text,
//!  - the pure-txt2img path (no `Reference`) diverges more than the heaviest img2img.
//!
//! Run (after deploying a Turbo snapshot + a source into local dirs):
//! ```text
//! set CUDA_VISIBLE_DEVICES=1
//! set ZIMG_TURBO_SNAPSHOT=...\Z-Image-Turbo    # tokenizer/ text_encoder/ transformer/ vae/
//! set ZIMG_TURBO_SRC=...\source.ppm            # a source image (P6 binary PPM)
//! set ZIMG_TURBO_OUT=...\out
//! cargo test -p candle-gen-z-image --features cuda --release \
//!   turbo_img2img_validate::real_weight -- --ignored --nocapture
//! ```

use candle_gen::gen_core::{
    self, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress,
    WeightsSource,
};
use candle_gen::testkit::{env_path, mean_abs_diff, read_ppm, write_ppm};

#[test]
#[ignore = "real-weight GPU validation; set ZIMG_TURBO_SNAPSHOT/ZIMG_TURBO_SRC/ZIMG_TURBO_OUT"]
fn real_weight_turbo_img2img() {
    let out_dir = env_path("ZIMG_TURBO_OUT");
    std::fs::create_dir_all(&out_dir).ok();

    let source = read_ppm(&env_path("ZIMG_TURBO_SRC"));
    println!(
        "source {}x{}; resolving registered z_image_turbo …",
        source.width, source.height
    );

    let spec = LoadSpec::new(WeightsSource::Dir(env_path("ZIMG_TURBO_SNAPSHOT")));
    let t0 = std::time::Instant::now();
    let model = crate::provider_registry()
        .expect("build z-image provider registry")
        .load("z_image_turbo", &spec)
        .expect("load z_image_turbo");
    println!(
        "resolved id={} backend={} accepts(Reference)={} in {:?}",
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

    // CFG-free distilled Turbo: NO guidance / NO negative prompt (the descriptor rejects them). A few
    // steps past the 4-step default so the reduced img2img tail still has room to denoise.
    let gen = |strength: f32, prompt: &str| -> Image {
        let req = GenerationRequest {
            prompt: prompt.to_owned(),
            width,
            height,
            steps: Some(8),
            seed: Some(12345),
            conditioning: vec![Conditioning::Reference {
                image: source.clone(),
                strength: Some(strength),
            }],
            ..Default::default()
        };
        let mut noop = |_p: Progress| {};
        match model
            .generate(&req, &mut noop)
            .unwrap_or_else(|e| panic!("generate (s={strength} \"{prompt}\"): {e}"))
        {
            GenerationOutput::Images(mut imgs) => imgs.pop().expect("one image"),
            _ => panic!("expected images, got video"),
        }
    };

    let prompt_a = "a watercolor painting, soft pastel colors, dreamy, artistic";
    let prompt_b =
        "an oil painting, dark dramatic chiaroscuro lighting, heavy impasto brushstrokes";

    // strength 1.0 ⇒ start == steps ⇒ empty loop ⇒ the source's VAE round-trip at the render size: the
    // "no-edit" baseline to measure structure preservation against.
    let t = std::time::Instant::now();
    let roundtrip = gen(1.0, prompt_a);
    println!("[roundtrip s=1.0] {:?}", t.elapsed());
    write_ppm(
        &out_dir.join("zimage_turbo_img2img_roundtrip.ppm"),
        &roundtrip,
    );

    let out_regen = gen(0.25, prompt_a);
    write_ppm(&out_dir.join("zimage_turbo_img2img_s025.ppm"), &out_regen);
    let out_regen_b = gen(0.25, prompt_b);
    write_ppm(
        &out_dir.join("zimage_turbo_img2img_s025_b.ppm"),
        &out_regen_b,
    );
    let out_edit = gen(0.6, prompt_a);
    write_ppm(&out_dir.join("zimage_turbo_img2img_s06.ppm"), &out_edit);
    let out_preserve = gen(0.9, prompt_a);
    write_ppm(&out_dir.join("zimage_turbo_img2img_s09.ppm"), &out_preserve);

    let d_regen = mean_abs_diff(&out_regen, &roundtrip);
    let d_edit = mean_abs_diff(&out_edit, &roundtrip);
    let d_preserve = mean_abs_diff(&out_preserve, &roundtrip);
    let d_prompt = mean_abs_diff(&out_regen, &out_regen_b);
    println!("=== z_image_turbo img2img validation ===");
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
    // Gate 3 (the correctness proof): the Z-Image structure-preservation convention — strength is
    // monotone the INVERSE of SDXL, so a lower strength diverges from the source MORE than a higher one.
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
    let txt2img = {
        let req = GenerationRequest {
            prompt: prompt_a.to_owned(),
            width,
            height,
            steps: Some(8),
            seed: Some(12345),
            ..Default::default()
        };
        let mut noop = |_p: Progress| {};
        match model.generate(&req, &mut noop).expect("txt2img generate") {
            GenerationOutput::Images(mut imgs) => imgs.pop().expect("one image"),
            _ => panic!("expected images"),
        }
    };
    write_ppm(&out_dir.join("zimage_turbo_txt2img.ppm"), &txt2img);
    let d_txt = mean_abs_diff(&txt2img, &roundtrip);
    println!("  txt2img (no Reference) diff vs round-trip: {d_txt:.2}");
    assert!(
        d_txt > d_regen,
        "txt2img should diverge more than s=0.25 img2img"
    );

    println!("z_image_turbo img2img validation PASS ✅ (eyeball the PPMs for edit quality)");
}
