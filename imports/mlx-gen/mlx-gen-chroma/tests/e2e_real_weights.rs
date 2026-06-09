//! sc-3839: real-weight e2e parity vs torch `diffusers` ChromaPipeline (HD), f32 both sides.
//! Golden = `tools/dump_chroma_e2e_golden.py`. `#[ignore]` — needs the ~18GB Chroma1-HD snapshot;
//! run with `cargo test -p mlx-gen-chroma --test e2e_real_weights -- --ignored --nocapture`.
//!
//! Three gates, increasingly integrated:
//!   1. masked T5 encode (sc-3838 numeric) — `encode_prompt` `prompt_embeds` vs diffusers;
//!   2. single real-weight DiT forward (sc-3837 at full scale) — `noise_pred` on fixed inputs;
//!   3. full true-CFG denoise + VAE decode — final latents + image coherence.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{LoadSpec, Progress, WeightsSource};
use mlx_gen_chroma::{encode_prompt, load_chroma, ChromaVariant};
use mlx_rs::ops::{abs, concatenate_axis, max, multiply, subtract, sum};
use mlx_rs::{Array, Dtype};

const PROMPT: &str = "a photograph of an astronaut riding a horse";
const NEG: &str = "";
const W: u32 = 256;
const H: u32 = 256;
const STEPS: u32 = 4;
const GUIDANCE: f32 = 4.0;

fn hf_snapshot() -> PathBuf {
    let cache = std::env::var("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/huggingface/hub")
        });
    let snaps = cache.join("models--lodestones--Chroma1-HD/snapshots");
    std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| panic!("Chroma1-HD snapshot not found under {}", snaps.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn peak_rel(got: &Array, golden: &Array) -> f32 {
    let d = max(abs(subtract(got, golden).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let s = max(abs(golden).unwrap(), None).unwrap().item::<f32>();
    d / s
}

/// Relative L2 `‖got−golden‖₂ / ‖golden‖₂` — robust to single-element outliers (unlike peak-rel).
fn rel_l2(got: &Array, golden: &Array) -> f32 {
    let l2 = |a: &Array| -> f32 {
        sum(multiply(a, a).unwrap(), None)
            .unwrap()
            .item::<f32>()
            .sqrt()
    };
    l2(&subtract(got, golden).unwrap()) / l2(golden)
}

#[test]
#[ignore = "needs the ~18GB Chroma1-HD snapshot + several minutes"]
fn chroma_hd_e2e_matches_diffusers() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let g = Weights::from_file(format!("{dir}/chroma_e2e.safetensors")).unwrap();

    let spec = LoadSpec::new(WeightsSource::Dir(hf_snapshot()));
    let model = load_chroma(ChromaVariant::Hd, &spec).expect("load Chroma1-HD");

    // --- 1. masked T5 encode parity (sc-3838 numeric) ---
    let (pe, pm) = encode_prompt(model.tokenizer_ref(), model.t5_ref(), PROMPT).unwrap();
    let pe_rel = peak_rel(&pe, g.require("prompt_embeds").unwrap());
    eprintln!("prompt_embeds peak-rel = {pe_rel:.4}");
    assert!(pe_rel < 5e-2, "masked T5 prompt_embeds diverged: {pe_rel}");
    let (nege, _negm) = encode_prompt(model.tokenizer_ref(), model.t5_ref(), NEG).unwrap();
    let neg_rel = peak_rel(&nege, g.require("neg_embeds").unwrap());
    eprintln!("neg_embeds peak-rel = {neg_rel:.4}");
    // the transformer text mask (keep-one-extra-pad) must match exactly.
    let pm_diff = max(
        abs(subtract(&pm, g.require("prompt_mask").unwrap()).unwrap()).unwrap(),
        None,
    )
    .unwrap()
    .item::<f32>();
    assert_eq!(pm_diff, 0.0, "transformer text mask diverged");

    // --- 2. single real-weight DiT forward (tight) ---
    let init = g.require("init_latents").unwrap();
    let ts = g.require("timestep").unwrap();
    let img_ids = g.require("img_ids").unwrap();
    // Feed the *golden* prompt_embeds so this gate isolates the DiT (decoupled from any T5 delta).
    let golden_embeds = g.require("prompt_embeds").unwrap();
    let l = golden_embeds.shape()[1];
    let txt_ids = Array::from_slice(&vec![0f32; (l * 3) as usize], &[l, 3]);
    let si = ((H / 16) * (W / 16)) as i32;
    let ones = Array::ones::<f32>(&[1, si]).unwrap();
    let full_mask = concatenate_axis(&[g.require("prompt_mask").unwrap(), &ones], 1).unwrap();
    let noise_pred = model
        .transformer_ref()
        .forward(init, golden_embeds, ts, img_ids, &txt_ids, Some(&full_mask))
        .unwrap();
    let np_rel = peak_rel(&noise_pred, g.require("noise_pred").unwrap());
    eprintln!("noise_pred peak-rel = {np_rel:.4}");
    assert!(np_rel < 5e-2, "single DiT forward diverged: {np_rel}");

    // --- 3. full true-CFG denoise + decode (coherence) ---
    let mut nop = |_p: Progress| {};
    let final_latents = model
        .denoise(PROMPT, NEG, W, H, STEPS, GUIDANCE, init.clone(), &mut nop)
        .unwrap();
    let gfl = g.require("final_latents").unwrap();
    let mymax = max(abs(&final_latents).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let gmax = max(abs(gfl).unwrap(), None).unwrap().item::<f32>();
    eprintln!("final max|mine|={mymax:.4} max|golden|={gmax:.4}");
    let fl_rel = peak_rel(&final_latents, gfl);
    let fl_l2 = rel_l2(&final_latents, gfl);
    eprintln!("final_latents peak-rel = {fl_rel:.4}  rel-L2 = {fl_l2:.4}");
    // rel-L2 over a 4-step g=4 CFG run across torch-MPS-f32 vs mlx-Metal-f32 (different GPU kernels);
    // the decoded image (px>8 below) is the authoritative success criterion.
    assert!(fl_l2 < 0.08, "final latents diverged (rel-L2 {fl_l2})");

    // decoded image coherence: fraction of pixels with |Δ| > 16/255.
    let img = model.decode(&final_latents, W, H).unwrap();
    let golden_img = g.require("image").unwrap(); // [1,3,H,W] in [-1,1]
    let gi: Vec<f32> = golden_img
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let n = (H * W) as usize;
    let (mut p8, mut p16) = (0usize, 0usize);
    for c in 0..3 {
        for p in 0..n {
            let gv = ((gi[c * n + p] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0);
            let mv = img.pixels[p * 3 + c] as f32; // Image is HWC RGB u8
            let d = (gv - mv).abs();
            if d > 8.0 {
                p8 += 1;
            }
            if d > 16.0 {
                p16 += 1;
            }
        }
    }
    let tot = (3 * n) as f32;
    let (f8, f16) = (p8 as f32 / tot, p16 as f32 / tot);
    eprintln!("image px>8 = {f8:.4}  px>16 = {f16:.4}");
    assert!(f8 < 0.08, "decoded image diverged: {f8} px>8");
}
