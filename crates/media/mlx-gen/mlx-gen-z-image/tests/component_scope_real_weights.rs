//! **SC-15794 end-to-end**: does rung 4's text-encoder scope move the REQUEST peak, through the real
//! `generate()` path, and does the image survive?
//!
//! The isolated encoder numbers live in `text_encoder_window_real_weights`. This is the test that
//! decides whether any of it matters: a phase win that a different phase already bounds is not a win,
//! which is exactly what the sub-512 decode-tile sweep found for rung 2.
//!
//! Two arms per tier, identical in every other respect:
//! - `component = Dit` — the DiT stream only; the encoder runs unscoped.
//! - `component = Both` — the DiT stream plus a windowed encoder.
//!
//! **The `Dit` arm is NOT the pre-SC-15794 baseline**, and reading it as one understates the change.
//! Since SC-15794 a snapshot load builds a *streamable* encoder, so even an unscoped `forward` runs
//! the stack as one full-width window — which drains the weights view per layer instead of holding all
//! 36 layers' tensors alive the way `TextEncoder::from_weights` did. Measured on bf16, that alone is
//! 8.489 → 2.718 GiB, and the extra tightening from a window of 1 is 2.718 → 1.455
//! (`text_encoder_window_real_weights` carries the full ladder including the true resident arm).
//!
//! So this A/B measures only the *second* effect. The first is why bf16 conditioning no longer binds
//! the request at all — before the change its 8.489 GiB conditioning put the bf16 request peak above
//! an 8 GB budget; it is now decode-bound at 4.365.
//!
//! ```sh
//! ZIMAGE_SNAPSHOT=<tier dir, e.g. …/bf16> \
//!   cargo test -p mlx-gen-z-image --release --test component_scope_real_weights \
//!   -- --ignored --nocapture --test-threads=1
//! ```

mod common;

use mlx_gen::gen_core::{GenerationMemory, TransformerComponent};
use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Progress, Quant,
    WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

use common::snapshot;

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Tier from the snapshot path's own directory name, so a run cannot silently measure a tier other
/// than the weights it was pointed at.
fn tier_from_path() -> (Option<Quant>, String) {
    let name = snapshot()
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    match name.as_str() {
        "q4" => (Some(Quant::Q4), name),
        "q8" => (Some(Quant::Q8), name),
        _ => (None, name),
    }
}

struct Run {
    conditioning: u64,
    denoise: u64,
    decode: u64,
    image: Image,
}

impl Run {
    fn request(&self) -> u64 {
        self.conditioning.max(self.denoise).max(self.decode)
    }
    fn binding(&self) -> &'static str {
        let r = self.request();
        if r == self.conditioning {
            "conditioning"
        } else if r == self.denoise {
            "denoise"
        } else {
            "decode"
        }
    }
}

fn run_with(component: TransformerComponent) -> Run {
    let (quant, _) = tier_from_path();
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    spec.offload_policy = OffloadPolicy::Sequential;
    if let Some(q) = quant {
        spec = spec.with_quant(q);
    }
    let size = env_u32("ZIMAGE_SIZE", 1024);
    let req = GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(1234),
        steps: Some(env_u32("ZIMAGE_STEPS", 4)),
        memory: Some(GenerationMemory {
            tile_vae_decode: true,
            chunk_attention: true,
            stream_transformer_blocks: true,
            transformer_window_size: Some(1),
            transformer_window_component: Some(component),
            ..Default::default()
        }),
        ..Default::default()
    };

    let generator = mlx_gen_z_image::load(&spec).expect("load z_image_turbo");
    clear_cache();
    reset_peak_memory();
    let mut conditioning = 0usize;
    let mut denoise = 0usize;
    let mut on_progress = |p: Progress| match p {
        Progress::Step { current: 1, .. } => {
            conditioning = get_peak_memory();
            reset_peak_memory();
        }
        Progress::Decoding if denoise == 0 => {
            denoise = get_peak_memory();
            reset_peak_memory();
        }
        _ => {}
    };
    let out = generator
        .generate(&req, &mut on_progress)
        .expect("generation");
    let decode = get_peak_memory();
    let image = match out {
        GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        other => panic!("expected images, got {other:?}"),
    };
    drop(generator);
    clear_cache();
    assert!(
        conditioning > 0 && denoise > 0 && decode > 0,
        "a phase boundary was not sampled"
    );
    Run {
        conditioning: conditioning as u64,
        denoise: denoise as u64,
        decode: decode as u64,
        image,
    }
}

/// Max per-channel byte difference between two rendered images.
fn image_delta(a: &Image, b: &Image) -> u8 {
    assert_eq!(a.width, b.width, "image width changed");
    assert_eq!(a.height, b.height, "image height changed");
    assert_eq!(a.pixels.len(), b.pixels.len(), "image byte length changed");
    a.pixels
        .iter()
        .zip(&b.pixels)
        .fold(0u8, |m, (x, y)| m.max(x.abs_diff(*y)))
}

/// The whole story in one test: the encoder scope must cut the conditioning phase, must not change the
/// image, and — the part that decides whether it ships — must move the request peak.
#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot + Apple/Metal GPU"]
fn the_encoder_scope_moves_the_request_peak_without_changing_the_image() {
    let (_, tier) = tier_from_path();
    let dit_only = run_with(TransformerComponent::Dit);
    let both = run_with(TransformerComponent::Both);

    println!("\nSC-15794 component scope A/B ({tier} tier, real weights)");
    println!(
        "  {:<12} {:>13} {:>9} {:>8} {:>10} {:>14}",
        "component", "conditioning", "denoise", "decode", "request", "binding phase"
    );
    for (label, r) in [("Dit", &dit_only), ("Both", &both)] {
        println!(
            "  {:<12} {:>12.3}G {:>8.3}G {:>7.3}G {:>9.3}G {:>14}",
            label,
            gib(r.conditioning),
            gib(r.denoise),
            gib(r.decode),
            gib(r.request()),
            r.binding()
        );
    }
    let cond_cut = 1.0 - both.conditioning as f64 / dit_only.conditioning.max(1) as f64;
    let req_cut = 1.0 - both.request() as f64 / dit_only.request().max(1) as f64;
    println!(
        "  => conditioning {:.1}%, request {:.1}%",
        cond_cut * 100.0,
        req_cut * 100.0
    );

    // The 8 GB question this rung exists to answer.
    let reserve = 2.0;
    for (label, r) in [("Dit", &dit_only), ("Both", &both)] {
        let total = gib(r.request()) + reserve;
        println!(
            "  {label:<12} + {reserve:.1} GiB reserve = {total:.3} GiB against 8.0 -> {}",
            if total <= 8.0 { "FITS" } else { "DOES NOT FIT" }
        );
    }
    println!(
        "  DISCLOSURE: this host is NOT a physical 8 GB Mac and MLX's peak counter is ACTIVE bytes.\n  \
         Whether the 2.0 GiB reserve is right for a UNIFIED pool is SC-15611 / SC-15614.\n"
    );

    // 1. Quality is preserved. A rung is quality-preserving by definition, and the encoder feeds every
    //    denoise step, so a drift here would be the whole conditioning signal moving.
    let delta = image_delta(&dit_only.image, &both.image);
    assert_eq!(
        delta, 0,
        "the encoder scope changed the image by {delta}/255 — rung 4 must be quality-preserving"
    );

    // 2. The conditioning phase actually moved. Without this the arms are indistinguishable and the
    //    request comparison below is vacuous.
    assert!(
        cond_cut > 0.20,
        "conditioning moved only {:.1}% ({:.3} -> {:.3} GiB) — the encoder scope did not engage, so \
         this A/B is comparing two identical runs",
        cond_cut * 100.0,
        gib(dit_only.conditioning),
        gib(both.conditioning)
    );

    // 3. The request peak is what the user pays. It moves on the tiers where conditioning was binding
    //    and does not on the tiers where a different phase already bound — both are legitimate, so
    //    this asserts consistency rather than a fixed direction.
    if dit_only.binding() == "conditioning" {
        assert!(
            req_cut > 0.01,
            "conditioning bound the request at {:.3} GiB but the scope moved the request peak only \
             {:.1}% — the saving is being eaten somewhere between the phase and the request",
            gib(dit_only.request()),
            req_cut * 100.0
        );
    } else {
        assert!(
            req_cut.abs() < 0.05,
            "a different phase bound this tier's request, so the encoder scope should leave the \
             request peak alone; it moved {:.1}%",
            req_cut * 100.0
        );
    }
}
