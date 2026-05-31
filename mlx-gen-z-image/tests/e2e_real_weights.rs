//! sc-2352 / sc-2344: end-to-end validation of the Z-Image port against a real-weights golden run.
//!
//! `#[ignore]`d — needs the real `Tongyi-MAI/Z-Image-Turbo` weights in the HF cache and the
//! golden produced by `tools/dump_z_image_golden.py` (gitignored, local). Run with:
//!   cargo test -p mlx-gen-z-image --release --test e2e_real_weights -- --ignored --nocapture
//!
//! The stage tests validate each pipeline stage on real bf16 weights against the fork's
//! intermediates; the final test drives the **public** `load(id, spec).generate(req)` API and
//! confirms the rendered image matches the fork's golden.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{
    FlowMatchEuler, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};
use mlx_gen_z_image::{
    decoded_to_image, denoise, load_text_encoder, load_tokenizer, load_transformer, load_vae,
    slice_valid, unpack_latents,
};
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/z_image_golden.safetensors"
);

/// Locate the Z-Image-Turbo snapshot dir (env override, else the HF cache).
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("ZIMAGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

/// Peak-relative error `max|a-b| / max|b|` — the meaningful metric for high-dynamic-range
/// tensors compared against a bf16 golden.
fn peak_rel(a: &Array, b: &Array) -> f32 {
    // reshape to 1-D forces C-order materialization (decode/transpose views would otherwise
    // expose physical, not logical, order through as_slice).
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    max_diff / peak
}

fn bf16(a: &Array) -> Array {
    a.as_dtype(Dtype::Bfloat16).unwrap()
}

#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_text_encoder_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let num_valid: i32 = g.metadata("num_valid").unwrap().parse().unwrap();

    let enc = load_text_encoder(&snapshot()).unwrap();
    let out = enc
        .forward(
            g.require("input_ids").unwrap(),
            g.require("attention_mask").unwrap(),
        )
        .unwrap();
    let cap = slice_valid(&out, num_valid).unwrap();

    let golden = g.require("cap_feats").unwrap();
    assert_eq!(cap.shape(), golden.shape(), "cap_feats shape");

    let a = cap.as_slice::<f32>();
    let b = golden.as_slice::<f32>();
    let max_abs_g = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mean_diff: f32 =
        a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    // Peak-relative error: the meaningful metric for a high-dynamic-range tensor (values reach
    // ~1.4e4) compared against a bf16 golden after a 35-layer f32 forward.
    let peak_rel = max_diff / max_abs_g;
    println!(
        "cap_feats: max|golden|={max_abs_g:.1} max|diff|={max_diff:.3} peak_rel={peak_rel:.2e} mean|diff|={mean_diff:.5}"
    );
    assert!(
        peak_rel < 2e-3,
        "cap_feats diverged from the fork: peak-relative error {peak_rel:.2e} >= 2e-3"
    );
    println!(
        "✓ text encoder: cap_feats {:?} matches the fork golden (peak-rel {peak_rel:.2e})",
        cap.shape()
    );
}

#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_transformer_single_forward_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let transformer = load_transformer(&snapshot()).unwrap();

    // First step in f32 (rules out bf16): v0 = transformer(init, 1 - sigma[0], cap_feats).
    let timestep0 = 1.0 - sigmas[0];
    let v = transformer
        .forward(
            g.require("init").unwrap(),
            timestep0,
            g.require("cap_feats").unwrap(),
        )
        .unwrap();
    let golden = g.require("v0").unwrap();
    assert_eq!(v.shape(), golden.shape(), "v0 shape");
    let pr = peak_rel(&v, golden);
    println!(
        "transformer single forward: v0 peak_rel={pr:.2e} shape={:?}",
        v.shape()
    );
    assert!(
        pr < 5e-2,
        "single transformer forward diverged at real resolution: peak_rel {pr:.2e}"
    );
    println!("✓ transformer single forward matches golden");
}

#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_denoise_loop_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    // Use the fork's exact sigmas (not a recomputed schedule) so this isolates the loop, not mu.
    let scheduler = FlowMatchEuler { sigmas };
    let transformer = load_transformer(&snapshot()).unwrap();

    // Match the fork's bf16 path: init noise + cap_feats fed to the DiT as bf16.
    let init = bf16(g.require("init").unwrap());
    let cap = bf16(g.require("cap_feats").unwrap());
    let out = denoise(&transformer, &scheduler, init, &cap).unwrap();
    let out = out.as_dtype(Dtype::Float32).unwrap();

    let golden = g.require("final_latents").unwrap();
    assert_eq!(out.shape(), golden.shape(), "final latents shape");
    let pr = peak_rel(&out, golden);
    println!(
        "denoise: final_latents peak_rel={pr:.2e} shape={:?}",
        out.shape()
    );
    // bf16 accumulation over 4 iterative steps (each feeding the next) compounds; the decoded
    // image is near-pixel-perfect, so this peak-relative latent drift is benign.
    assert!(pr < 1e-1, "final latents diverged: peak_rel {pr:.2e}");
    println!("✓ denoise loop matches golden (peak-rel {pr:.2e})");
}

#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_vae_and_image_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let vae = load_vae(&snapshot()).unwrap();

    // golden final_latents [16,1,H,W] -> unpack [1,16,H,W] -> [1,16,1,H,W] for decode.
    let latents = g.require("final_latents").unwrap();
    let unpacked = unpack_latents(latents).unwrap();
    let sh = unpacked.shape();
    let latent5 = unpacked.reshape(&[sh[0], sh[1], 1, sh[2], sh[3]]).unwrap();
    let decoded = vae.decode(&latent5).unwrap(); // f32 (latents f32, weights bf16 -> promote)
    let decoded = decoded.as_dtype(Dtype::Float32).unwrap();

    let golden = g.require("decoded").unwrap();
    assert_eq!(decoded.shape(), golden.shape(), "decoded shape");
    let pr = peak_rel(&decoded, golden);
    println!("vae: decoded peak_rel={pr:.2e} shape={:?}", decoded.shape());
    assert!(pr < 2e-2, "VAE decode diverged: peak_rel {pr:.2e}");

    // RGB8 image: my decoded vs the golden decoded, both through decoded_to_image.
    let img = decoded_to_image(&decoded).unwrap();
    let gimg = decoded_to_image(golden).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 2)
        .count();
    println!(
        "✓ vae+image: {}x{}, {} / {} pixels differ by >2",
        img.width,
        img.height,
        differ,
        img.pixels.len()
    );
    assert!(
        differ < img.pixels.len() / 50,
        "too many pixel diffs: {differ}"
    );
}

/// The integration proof: the full prompt→image pipeline through the **public** Generator API
/// (`mlx_gen::load("z_image_turbo", …).generate(req)`), compared to the fork's golden render.
#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_full_pipeline_generates_fox() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let snap = snapshot();
    let num_valid: i32 = g.metadata("num_valid").unwrap().parse().unwrap();

    // Drive the request from the golden's own metadata so this test tracks whatever
    // (prompt, seed, steps, size) the golden was dumped at — no separate hardcoding to
    // drift. dump_z_image_golden.py honors ZIMAGE_W/H/STEPS/SEED/PROMPT; this reads them back.
    let prompt = g.metadata("prompt").unwrap().to_string();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let steps: u32 = g.metadata("steps").unwrap().parse().unwrap();
    let w: u32 = g.metadata("w").unwrap().parse().unwrap();
    let h: u32 = g.metadata("h").unwrap().parse().unwrap();

    // Tokenizer parity: the prompt with the Qwen chat template reproduces the fork's ids exactly.
    let tok = load_tokenizer(&snap).unwrap();
    let t = tok.tokenize(&prompt).unwrap();
    let take_n =
        |a: &Array| a.reshape(&[-1]).unwrap().as_slice::<i32>()[..num_valid as usize].to_vec();
    assert_eq!(
        take_n(&t.input_ids),
        take_n(g.require("input_ids").unwrap()),
        "tokenizer input_ids diverge from the fork"
    );

    // Full pipeline through the public API: load(id, spec) -> generate(req).
    let spec = LoadSpec::new(WeightsSource::Dir(snap));
    let generator = mlx_gen::load("z_image_turbo", &spec).unwrap();
    let req = GenerationRequest {
        prompt: prompt.clone(),
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };
    let mut last_step = 0u32;
    let out = generator
        .generate(&req, &mut |p| {
            if let Progress::Step { current, total } = p {
                assert_eq!(total, steps, "step total");
                last_step = last_step.max(current);
            }
        })
        .unwrap();
    assert_eq!(
        last_step, steps,
        "expected {steps} denoise-step progress events"
    );

    let img = match out {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1, "count=1 -> one image");
            v.pop().unwrap()
        }
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!((img.width, img.height), (w, h), "image size");

    // Save the Rust render for visual inspection.
    let out_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/golden/rust_fox.png");
    image::save_buffer(
        &out_path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();

    // Compare to the fork's golden image (bf16-loop drift allows a small fraction of pixels to
    // differ).
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    println!(
        "✓ full pipeline (public generate): prompt->image {}x{}; {} / {} pixels differ by >8 from the fork; saved {}",
        img.width,
        img.height,
        differ,
        img.pixels.len(),
        out_path.display()
    );
    assert!(
        differ < img.pixels.len() / 20,
        "full-pipeline image diverges: {differ} pixels"
    );
}
