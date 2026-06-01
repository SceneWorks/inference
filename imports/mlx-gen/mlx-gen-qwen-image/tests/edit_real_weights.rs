//! sc-2465 slice 7a: Qwen-Image-Edit pipeline parity vs the frozen fork. Micro-gated.
//!
//! - **Gate 1 (here)**: the multi-image (dual-latent) RoPE — `QwenRope3d::forward_multi` over
//!   `[noise_grid, cond_grid]` — vs the fork's `QwenEmbedRopeMLX`. Weight-free.
//!
//! Run: `cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_qwen_edit_rope_golden.py`, then
//! `cargo test -p mlx-gen-qwen-image --release --test edit_real_weights -- --ignored --nocapture`

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::CancelFlag;
use mlx_gen_qwen_image::transformer::QwenRope3d;
use mlx_gen_qwen_image::{denoise_edit_with_progress, loader, qwen_scheduler, unpack_latents};
use mlx_rs::Array;

const ROPE_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_edit_rope_golden.safetensors"
);
const EDIT_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_image_edit_golden.safetensors"
);

// Must match tools/dump_qwen_image_edit_golden.py.
const STEPS: usize = 2;
const GUIDANCE: f32 = 4.0;

fn edit_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN_IMAGE_EDIT_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2509/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn rel_errors(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let sum_abs_b: f64 = b.iter().map(|&v| v.abs() as f64).sum();
    let sum_abs_diff: f64 = a.iter().zip(b).map(|(&x, &y)| (x - y).abs() as f64).sum();
    (max_diff / peak, (sum_abs_diff / sum_abs_b) as f32)
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    a.iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
}

// Must match tools/dump_qwen_edit_rope_golden.py.
#[test]
#[ignore = "needs local edit-rope golden"]
fn edit_rope_multi_image_matches_fork() {
    let g = Weights::from_file(ROPE_GOLDEN).unwrap();
    let (ic, is_, tc, ts) = QwenRope3d::qwen_image()
        .forward_multi(&[(8, 12), (6, 6)], 20)
        .unwrap();
    for (name, got, key) in [
        ("img_cos", &ic, "img_cos"),
        ("img_sin", &is_, "img_sin"),
        ("txt_cos", &tc, "txt_cos"),
        ("txt_sin", &ts, "txt_sin"),
    ] {
        let want = g.require(key).unwrap();
        assert_eq!(got.shape(), want.shape(), "{name} shape");
        let d = max_abs_diff(got, want);
        println!("edit rope {name} {:?}: max abs diff {d:.3e}", got.shape());
        assert!(d < 1e-5, "{name} max abs diff {d:.3e}");
    }
}

/// Gate 2: the full dual-latent denoise loop (concat noise+ref → transformer with `cond_grids` →
/// slice → CFG → Euler) vs the fork's edit loop. Feeds the golden noise + prompt embeds + packed
/// reference latents + cond grid (so the tokenizer / VL encoder / VAE-encode — each separately
/// verified — are out of scope), loads the real transformer + VAE from the Edit snapshot, and
/// compares the final latents + decoded image.
#[test]
#[ignore = "needs real Qwen-Image-Edit-2509 transformer+VAE weights + local edit golden"]
fn edit_pipeline_matches_fork() {
    let g = Weights::from_file(EDIT_GOLDEN).unwrap();
    let root = edit_snapshot();
    let transformer = loader::load_transformer(&root).unwrap();
    let vae = loader::load_vae(&root).unwrap();

    let dims = g.require("out_dims").unwrap();
    let dims = dims.as_slice::<i32>();
    let (w, h) = (dims[0] as u32, dims[1] as u32);
    let cg = g.require("cond_grid").unwrap();
    let cg = cg.as_slice::<i32>();
    let cond_grids = vec![(cg[0] as usize, cg[1] as usize)];

    let noise = g.require("noise").unwrap().clone();
    let static_lat = g.require("static_image_latents").unwrap();
    let pos = g.require("pos_embeds").unwrap();
    let neg = g.require("neg_embeds").unwrap();
    let scheduler = qwen_scheduler(STEPS, w, h);

    let latents = denoise_edit_with_progress(
        &transformer,
        &scheduler,
        noise,
        static_lat,
        &cond_grids,
        pos,
        neg,
        GUIDANCE,
        w,
        h,
        &CancelFlag::default(),
        &mut |_| {},
    )
    .unwrap();

    let want = g.require("final_latents").unwrap();
    assert_eq!(latents.shape(), want.shape(), "final_latents shape");
    let (peak, mean) = rel_errors(&latents, want);
    println!("edit final_latents: peak-rel {peak:.3e}  mean-rel {mean:.3e}");
    assert!(mean < 2e-2, "edit final_latents mean-rel {mean:.3e}");
    assert!(peak < 1e-1, "edit final_latents peak-rel {peak:.3e}");

    let unpacked = unpack_latents(&latents, w, h).unwrap();
    let decoded = vae.decode(&unpacked).unwrap();
    let want_dec = g.require("decoded").unwrap();
    let (dpeak, dmean) = rel_errors(&decoded, want_dec);
    println!("edit decoded: peak-rel {dpeak:.3e}  mean-rel {dmean:.3e}");
    assert!(dmean < 5e-2, "edit decoded mean-rel {dmean:.3e}");
}
