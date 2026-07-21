//! sc-2346 S5: end-to-end real-weights parity for FLUX.2-klein single-reference EDIT. `#[ignore]`d
//! — needs the real snapshot + the f32 golden from `tools/dump_flux2_edit_golden.py`:
//!
//!   cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_edit_golden.py
//!   cargo test -p mlx-gen-flux2 --test edit_real_weights -- --ignored --nocapture
//!
//! Two gates (f32):
//!  1. **reference encoding** — the NEW edit chain (preprocess → VAE-encode → 2×2 patchify →
//!     BN-normalize → pack) reproduces the fork's `image_latents` (chaos-free, tight);
//!  2. **full edit generate** — `load("flux2_klein_9b_edit").generate(Reference)` render vs the
//!     fork's decoded image (px>8 coherence/floor, like the txt2img e2e).

use std::path::PathBuf;

use mlx_gen::image::decoded_to_image;
use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::{Conditioning, GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};
use mlx_gen_flux2::{load_vae, pack_latents, patchify_latents, preprocess_ref_image};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

const PROMPT: &str = "make it look like a cold winter morning";

fn snapshot() -> PathBuf {
    let p = std::env::var("MLX_GEN_FLUX2_SNAPSHOT").unwrap_or_else(|_| panic!("set MLX_GEN_FLUX2_SNAPSHOT to the required snapshot dir; inference never self-fetches or derives a cache location (epic 13657)"));
    PathBuf::from(p)
}

fn golden() -> Weights {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/golden/flux2_edit.safetensors");
    Weights::from_file(&path).unwrap_or_else(|_| {
        panic!(
            "missing {} — run tools/dump_flux2_edit_golden.py",
            path.display()
        )
    })
}

/// Reconstruct the reference `Image` from the golden's `ref_u8` `[256,256,3]`.
fn ref_image(g: &Weights) -> Image {
    let a = g.require("ref_u8").unwrap().as_dtype(Dtype::Int32).unwrap();
    let sh = a.shape();
    let (h, w) = (sh[0] as u32, sh[1] as u32);
    let pixels: Vec<u8> = a
        .reshape(&[sh.iter().product::<i32>()])
        .unwrap()
        .as_slice::<i32>()
        .iter()
        .map(|&v| v as u8)
        .collect();
    Image {
        width: w,
        height: h,
        pixels,
    }
}

fn rel(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (xs, ys) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = ys.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let mabs = (ys.iter().map(|y| y.abs()).sum::<f32>() / ys.len() as f32).max(1e-12);
    let max_d = xs
        .iter()
        .zip(ys)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mean_d = xs.iter().zip(ys).map(|(x, y)| (x - y).abs()).sum::<f32>() / xs.len() as f32;
    (max_d / peak, mean_d / mabs)
}

fn px_gt8(a: &Image, b: &Image) -> f32 {
    assert_eq!(a.pixels.len(), b.pixels.len(), "image size mismatch");
    let n = a.pixels.len();
    let c = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() > 8)
        .count();
    100.0 * c as f32 / n as f32
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_edit.safetensors"]
fn reference_encoding_matches_fork() {
    let g = golden();
    let vae = load_vae(&snapshot()).unwrap();
    let img = ref_image(&g);
    // The edit reference chain (256² → no LANCZOS resize), via the public pipeline + VAE APIs.
    let pre = preprocess_ref_image(&img, 256, 256).unwrap(); // NHWC [1,256,256,3]
    let enc = vae
        .encode_mean(&pre)
        .unwrap()
        .transpose_axes(&[0, 3, 1, 2])
        .unwrap(); // [1,32,32,32]
    let patchified = patchify_latents(&enc).unwrap(); // [1,128,16,16]
    let normed = vae.bn_normalize_nchw(&patchified).unwrap();
    let packed = pack_latents(&normed).unwrap(); // [1,256,128]
    let want = g.require("image_latents").unwrap();
    assert_eq!(packed.shape(), want.shape(), "image_latents shape");
    let (peak, mean) = rel(&packed, want);
    println!("flux2 edit reference encoding: peak_rel={peak:.4} mean_rel={mean:.4}");
    assert!(
        mean < 5e-3,
        "reference image_latents diverged: mean_rel={mean}"
    );
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_edit.safetensors"]
fn full_edit_generate_matches_fork() {
    let g = golden();
    let gen = mlx_gen_flux2::provider_registry()
        .unwrap()
        .load(
            "flux2_klein_9b_edit",
            &LoadSpec::new(WeightsSource::Dir(snapshot())),
        )
        .unwrap();
    let req = GenerationRequest {
        prompt: PROMPT.into(),
        width: 256,
        height: 256,
        count: 1,
        seed: Some(0),
        steps: Some(4),
        conditioning: vec![Conditioning::Reference {
            image: ref_image(&g),
            strength: None,
        }],
        ..Default::default()
    };
    let out = gen.generate(&req, &mut |_| {}).unwrap();
    let GenerationOutput::Images(images) = out else {
        panic!("expected images");
    };
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let px = px_gt8(&images[0], &gimg);
    println!("flux2 edit full generate: {px:.2}% px>8 vs fork f32 (NAX-vs-wheel build delta)");
    assert!(
        px < 25.0,
        "edit generate diverged from the fork composition: {px}% px>8"
    );
}

// ---- sc-2645: multi-image (`MultiReference`) edit ---------------------------------------------
//
// Goldens from `tools/dump_flux2_edit_multi_golden.py` (dense f32 + `BITS=8`). Two distinct refs
// (flux2_klein_edit.jpg + flux2_klein.jpg), 256²/4 steps. The FLUX.2 text encoder is a dense Qwen3
// LLM with no vision input, so the prompt path is independent of the references — multi-image
// conditioning flows ONLY through the concatenated reference tokens (t = 10 + 10·i ids).

fn multi_golden(suffix: &str) -> Weights {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../tools/golden/flux2_edit_multi{suffix}.safetensors"
    ));
    Weights::from_file(&path).unwrap_or_else(|_| {
        panic!(
            "missing {} — run tools/dump_flux2_edit_multi_golden.py",
            path.display()
        )
    })
}

/// Reconstruct an `Image` from a named `[256,256,3]` golden tensor.
fn ref_image_named(g: &Weights, key: &str) -> Image {
    let a = g.require(key).unwrap().as_dtype(Dtype::Int32).unwrap();
    let sh = a.shape();
    let (h, w) = (sh[0] as u32, sh[1] as u32);
    let pixels: Vec<u8> = a
        .reshape(&[sh.iter().product::<i32>()])
        .unwrap()
        .as_slice::<i32>()
        .iter()
        .map(|&v| v as u8)
        .collect();
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// Encode one reference to packed latents via the public pipeline + VAE APIs (256² → no resize).
fn encode_ref_packed(vae: &mlx_gen_flux2::Flux2Vae, img: &Image) -> Array {
    let pre = preprocess_ref_image(img, 256, 256).unwrap(); // NHWC [1,256,256,3]
    let enc = vae
        .encode_mean(&pre)
        .unwrap()
        .transpose_axes(&[0, 3, 1, 2])
        .unwrap(); // [1,32,32,32]
    let patchified = patchify_latents(&enc).unwrap(); // [1,128,16,16]
    let normed = vae.bn_normalize_nchw(&patchified).unwrap();
    pack_latents(&normed).unwrap() // [1,256,128]
}

/// Gate: the concatenated 2-ref `image_latents` reproduce the fork's (chaos-free, tight). The
/// packed latents are id-independent (the t = 10 / t = 20 offsets live only in the ids), so the
/// concat over the sequence axis is the wiring under test here.
#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_edit_multi.safetensors"]
fn multi_reference_encoding_matches_fork() {
    let g = multi_golden("");
    let vae = load_vae(&snapshot()).unwrap();
    let p0 = encode_ref_packed(&vae, &ref_image_named(&g, "ref0_u8"));
    let p1 = encode_ref_packed(&vae, &ref_image_named(&g, "ref1_u8"));
    let packed = concatenate_axis(&[&p0, &p1], 1).unwrap(); // [1, 512, 128]
    let want = g.require("image_latents").unwrap();
    assert_eq!(packed.shape(), want.shape(), "2-ref image_latents shape");
    let (peak, mean) = rel(&packed, want);
    println!("flux2 multi-edit reference encoding (2 refs): peak_rel={peak:.4} mean_rel={mean:.4}");
    assert!(mean < 5e-3, "2-ref image_latents diverged: mean_rel={mean}");
}

/// The public multi-image edit render: `load(quant).generate(MultiReference{images})`.
fn render_multi_edit(quant: Option<Quant>, images: Vec<Image>) -> Image {
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    spec.quantize = quant;
    let gen = mlx_gen_flux2::provider_registry()
        .unwrap()
        .load("flux2_klein_9b_edit", &spec)
        .unwrap();
    let req = GenerationRequest {
        prompt: "blend the two scenes into a single dreamlike landscape".into(),
        width: 256,
        height: 256,
        count: 1,
        seed: Some(0),
        steps: Some(4),
        conditioning: vec![Conditioning::MultiReference { images }],
        ..Default::default()
    };
    let GenerationOutput::Images(mut images) = gen.generate(&req, &mut |_| {}).unwrap() else {
        panic!("expected images");
    };
    images.pop().unwrap()
}

/// Gate: the full public 2-ref edit render vs the fork f32 render.
#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_edit_multi.safetensors"]
fn full_multi_edit_generate_matches_fork() {
    let g = multi_golden("");
    let img = render_multi_edit(
        None,
        vec![
            ref_image_named(&g, "ref0_u8"),
            ref_image_named(&g, "ref1_u8"),
        ],
    );
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let px = px_gt8(&img, &gimg);
    println!("flux2 multi-edit full generate (2 refs): {px:.2}% px>8 vs fork f32 (NAX-vs-wheel build delta)");
    assert!(
        px < 25.0,
        "multi-edit generate diverged from the fork composition: {px}% px>8"
    );
}

/// Gate: the **Q8** 2-ref edit render vs the fork's Q8 render (bounded coherence floor: Rust f32
/// acts vs fork bf16, identical quantized weights per sc-2643).
#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_edit_multi_q8.safetensors"]
fn q8_multi_edit_generate_matches_fork() {
    let g = multi_golden("_q8");
    let img = render_multi_edit(
        Some(Quant::Q8),
        vec![
            ref_image_named(&g, "ref0_u8"),
            ref_image_named(&g, "ref1_u8"),
        ],
    );
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let px = px_gt8(&img, &gimg);
    println!("flux2 Q8 multi-edit full generate (2 refs): {px:.2}% px>8 vs fork Q8 (f32-act vs bf16-act + cross-build, chaos-amplified)");
    assert!(
        px < 70.0,
        "Q8 multi-edit generate not coherent: {px}% px>8 (scope/wiring bug?)"
    );
}

/// Single-image regression: with one `Reference`, the multi path must reduce to the original
/// single-ref edit (no divergence vs the existing `flux2_edit.safetensors` decoded image).
#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_edit.safetensors"]
fn single_reference_does_not_regress() {
    let g = golden(); // the original single-ref edit golden
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    spec.quantize = None;
    let gen = mlx_gen_flux2::provider_registry()
        .unwrap()
        .load("flux2_klein_9b_edit", &spec)
        .unwrap();
    let req = GenerationRequest {
        prompt: PROMPT.into(),
        width: 256,
        height: 256,
        count: 1,
        seed: Some(0),
        steps: Some(4),
        conditioning: vec![Conditioning::Reference {
            image: ref_image(&g),
            strength: None,
        }],
        ..Default::default()
    };
    let GenerationOutput::Images(images) = gen.generate(&req, &mut |_| {}).unwrap() else {
        panic!("expected images");
    };
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let px = px_gt8(&images[0], &gimg);
    println!("flux2 single-ref edit (post-multi-wiring) regression: {px:.2}% px>8 vs fork f32");
    assert!(px < 25.0, "single-ref edit regressed: {px}% px>8");
}
