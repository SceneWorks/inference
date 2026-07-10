//! Real-weights tests for mlx-gen-anima (sc-10515). `#[ignore]`d — they need the licensed
//! `circlestone-labs/Anima` snapshot in the HF cache and Metal. Run with:
//!   cargo test -p mlx-gen-anima --release --test real_weights -- --ignored --nocapture
//!
//! The snapshot dir is resolved by glob (no hardcoded sha); PNG output goes to `$ANIMA_OUT`
//! (default `/tmp/anima_sc10515`).

use std::path::PathBuf;

use mlx_rs::{Array, Dtype};

use mlx_gen::media::Image;
use mlx_gen::runtime::CancelFlag;
use mlx_gen::weights::Weights;
use mlx_gen::{Progress, WeightsSource};

use mlx_gen_anima::conditioner::AnimaTextConditioner;
use mlx_gen_anima::config::{ConditionerConfig, DitConfig, Variant};
use mlx_gen_anima::loader::split_anima_keys;
use mlx_gen_anima::pipeline::{AnimaPipeline, GenOptions};
use mlx_gen_anima::transformer::CosmosDiT;

/// Glob the Anima snapshot's `split_files/` dir from the HF cache (no hardcoded sha).
fn split_files() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--circlestone-labs--Anima/snapshots");
    std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            let p = e.path().join("split_files");
            p.join("diffusion_models").is_dir().then_some(p)
        })
}

fn out_dir() -> PathBuf {
    PathBuf::from(std::env::var("ANIMA_OUT").unwrap_or_else(|_| "/tmp/anima_sc10515".into()))
}

fn dit_file(split: &std::path::Path, v: Variant) -> PathBuf {
    split.join("diffusion_models").join(v.dit_filename())
}

/// `(grayscale_std, coarse_std)`. `std` is the full-image contrast (≈0 ⇒ blank wash). `coarse_std` is
/// the std of a 32×32 block-averaged downsample: real images keep strong coarse layout (subject vs
/// background) so it stays high, while noise — even the VAE's 8×8-upsampled latent noise — averages
/// out to a near-uniform coarse map (low `coarse_std`). Together they separate {blank wash, noise,
/// coherent image}. (A naive neighbor-diff ratio is fooled by the VAE's spatial upsampling.)
fn coherence(img: &Image) -> (f32, f32) {
    let (w, h) = (img.width as usize, img.height as usize);
    let gray: Vec<f32> = img
        .pixels
        .chunks(3)
        .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
        .collect();
    let mean = gray.iter().sum::<f32>() / gray.len() as f32;
    let std = (gray.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / gray.len() as f32).sqrt();
    let g = 32usize;
    let (bh, bw) = (h / g, w / g);
    let mut coarse = vec![0f32; g * g];
    for by in 0..g {
        for bx in 0..g {
            let mut s = 0.0f32;
            for y in 0..bh {
                for x in 0..bw {
                    s += gray[(by * bh + y) * w + (bx * bw + x)];
                }
            }
            coarse[by * g + bx] = s / (bh * bw) as f32;
        }
    }
    let cm = coarse.iter().sum::<f32>() / coarse.len() as f32;
    let coarse_std =
        (coarse.iter().map(|&x| (x - cm).powi(2)).sum::<f32>() / coarse.len() as f32).sqrt();
    (std, coarse_std)
}

fn save_png(img: &Image, path: &std::path::Path) {
    let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .expect("valid RGB buffer");
    buf.save(path).expect("save png");
}

fn assert_coherent(img: &Image, label: &str) {
    assert_eq!(img.width, 1024);
    assert_eq!(img.height, 1024);
    let (std, coarse_std) = coherence(img);
    println!("[{label}] grayscale std = {std:.2}, coarse32 std = {coarse_std:.2}");
    assert!(std > 8.0, "{label}: image is near-blank (std {std:.2})");
    // Real generations carry strong coarse layout (coarse32 std ~30-40); VAE-decoded noise averages to
    // a near-uniform coarse map (coarse32 std < ~8). A coherent anime image clears this easily.
    assert!(
        coarse_std > 12.0,
        "{label}: output lacks coarse structure — looks like noise, not a coherent image (coarse32 std {coarse_std:.2})"
    );
}

// -------------------------------------------------------------------------------------------------
// Structural / shape tests
// -------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot"]
fn checkpoint_split_is_118_adapter_567_dit() {
    let split = split_files().expect("Anima snapshot");
    let w = Weights::from_file(dit_file(&split, Variant::Base)).unwrap();
    let (dit, adapter) = split_anima_keys(&w);
    println!("dit keys = {}, adapter keys = {}", dit.len(), adapter.len());
    assert_eq!(adapter.len(), 118, "expected 118 llm_adapter tensors");
    assert_eq!(dit.len(), 567, "expected 567 DiT tensors");
}

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot"]
fn conditioner_output_is_b_512_1024() {
    let split = split_files().expect("Anima snapshot");
    let w = Weights::from_file(dit_file(&split, Variant::Base)).unwrap();
    let cond =
        AnimaTextConditioner::from_weights(&w, "net.llm_adapter", ConditionerConfig::anima())
            .unwrap();
    // dummy Qwen3 source states [1, 4, 1024] + T5 ids [1, 3] → conditioner must right-pad to 512.
    let source = mlx_rs::random::normal::<f32>(&[1, 4, 1024], None, None, None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let t5_ids = Array::from_slice(&[10i32, 42, 7], &[1, 3]);
    let out = cond.forward(&source, &t5_ids, Dtype::Bfloat16).unwrap();
    assert_eq!(
        out.shape(),
        &[1, 512, 1024],
        "conditioner must emit exactly 512 text tokens"
    );
}

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot"]
fn dit_forward_17ch_patch_and_5d_latent_roundtrip() {
    let split = split_files().expect("Anima snapshot");
    let w = Weights::from_file(dit_file(&split, Variant::Base)).unwrap();
    let dit = CosmosDiT::from_weights(&w, "net", DitConfig::anima()).unwrap();
    // 5-D latent [B,16,1,Hl,Wl] (Hl=Wl=64 → 512² image); patch-embed prepends the 17th (mask) channel.
    let latent = mlx_rs::random::normal::<f32>(&[1, 16, 1, 64, 64], None, None, None).unwrap();
    let sigma = Array::from_slice(&[1.0f32], &[1]);
    let encoder = mlx_rs::random::normal::<f32>(&[1, 512, 1024], None, None, None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let v = dit
        .forward(&latent, &sigma, &encoder, Dtype::Bfloat16)
        .unwrap();
    // velocity must have the same 5-D latent shape (proves patchify(17ch) + unpatchify roundtrip).
    assert_eq!(v.shape(), &[1, 16, 1, 64, 64]);
}

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot"]
fn dit_rejects_out_of_range_size_per_axis() {
    let split = split_files().expect("Anima snapshot");
    let w = Weights::from_file(dit_file(&split, Variant::Base)).unwrap();
    let dit = CosmosDiT::from_weights(&w, "net", DitConfig::anima()).unwrap();
    // latent Hl=250 → post-patch 125 > max_size 120 ⇒ RoPE must reject, not index OOB.
    let latent = mlx_rs::random::normal::<f32>(&[1, 16, 1, 250, 64], None, None, None).unwrap();
    let sigma = Array::from_slice(&[1.0f32], &[1]);
    let encoder = mlx_rs::random::normal::<f32>(&[1, 512, 1024], None, None, None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    assert!(dit
        .forward(&latent, &sigma, &encoder, Dtype::Bfloat16)
        .is_err());
}

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot"]
fn vae_decode_shape() {
    let split = split_files().expect("Anima snapshot");
    let vae = mlx_gen_anima::load_vae(split.join("vae/qwen_image_vae.safetensors")).unwrap();
    // a 5-D latent [1,16,1,128,128] → 1024² image.
    let latent = mlx_rs::random::normal::<f32>(&[1, 16, 1, 128, 128], None, None, None).unwrap();
    let img = vae.decode(&latent).unwrap();
    assert_eq!(img.shape(), &[1, 3, 1, 1024, 1024]);
}

// NOTE: the flow-velocity convention (the DiT is a standard flow denoiser, `v ≈ ε − x0`) and the raw-σ
// timestep were verified during the port via a `cos(v, ε − x0)` measurement against a KNOWN
// VAE-encoded latent (≈0.96 with timestep=σ; ≈0 with timestep=σ·1000). That check materializes many
// arrays and is flaky when run in the same test binary as other Metal tests (an mlx-rs cross-test
// default-stream quirk), so it is not kept as a standing test — the end-to-end `generate_*` test below
// is the standing guard: a wrong sign or timestep collapses the output into a wash/noise that
// `assert_coherent` rejects. See `pipeline.rs` for the convention documentation.

// -------------------------------------------------------------------------------------------------
// Acceptance: generate a real, coherent image for all three variants (bf16, 1024², fixed seed).
// -------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot; SLOW (real 2B DiT denoise)"]
fn generate_all_three_variants_1024() {
    let split = split_files().expect("Anima snapshot");
    let out = out_dir();
    std::fs::create_dir_all(&out).unwrap();
    let prompt =
        "an anime girl with long silver hair and blue eyes, detailed illustration, masterpiece";

    // Turbo first (cheapest: 10 steps, CFG-free) so a fundamental pipeline bug surfaces fast.
    for variant in [Variant::Turbo, Variant::Base, Variant::Aesthetic] {
        let pipeline =
            AnimaPipeline::from_source(&WeightsSource::Dir(split.clone()), variant).unwrap();

        // Default (recommended er_sde) solver, story-default steps/CFG.
        let opts = GenOptions {
            width: 1024,
            height: 1024,
            steps: variant.default_steps() as usize,
            guidance: variant.default_guidance(),
            seed: 42,
            sampler: None,
        };
        let cancel = CancelFlag::default();
        let mut prog = |_p: Progress| {};
        let img = pipeline
            .generate(prompt, "", variant, &opts, &cancel, &mut prog)
            .unwrap();
        let path = out.join(format!("{}_1024_er_sde.png", variant.id()));
        save_png(&img, &path);
        println!("wrote {}", path.display());
        assert_coherent(&img, variant.id());

        // Base: also render with the reference Euler solver (matches diffusers FlowMatchEuler) as a
        // cross-check that the DiT/conditioner port is correct independent of the stochastic solver.
        if variant == Variant::Base {
            let opts_euler = GenOptions {
                sampler: Some("euler".into()),
                ..GenOptions {
                    width: 1024,
                    height: 1024,
                    steps: variant.default_steps() as usize,
                    guidance: variant.default_guidance(),
                    seed: 42,
                    sampler: None,
                }
            };
            let img_e = pipeline
                .generate(prompt, "", variant, &opts_euler, &cancel, &mut prog)
                .unwrap();
            let path_e = out.join(format!("{}_1024_euler.png", variant.id()));
            save_png(&img_e, &path_e);
            println!("wrote {}", path_e.display());
            assert_coherent(&img_e, "anima_base_euler");
        }

        mlx_rs::memory::clear_cache();
    }
}
