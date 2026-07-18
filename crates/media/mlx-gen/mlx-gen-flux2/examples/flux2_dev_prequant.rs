//! One-shot: pre-quantize a dense bf16 FLUX.2-**dev** diffusers snapshot into a turnkey **packed**
//! snapshot the loader reads directly (no per-generate re-quant, no dense transient at generate
//! time). Packs the DiT + Mistral-3 text encoder to Q8 (default), borrows vae/tokenizer/scheduler by
//! absolute symlink, and copies `model_index.json`. Idempotent — re-run skips already-packed parts.
//!
//! The one dense transient is this offline convert read (peaks like a single-component load).
//!
//! Run (reuses the warm workspace target):
//!   FLUX2_DEV_SNAPSHOT=~/Models/aether/flux2-dev FLUX2_DEV_Q8=~/Models/aether/flux2-dev-q8 \
//!     cargo run --release --example flux2_dev_prequant -p mlx-gen-flux2
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_gen_flux2::{quantize_flux2_dit, quantize_flux2_text_encoder_dir};

fn env_req(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| panic!("set {k}"))
}

/// Replace `dst` with an absolute symlink to `src` (idempotent — removes any prior link/file/dir).
fn link_borrowed(src: &Path, dst: &Path) {
    if dst.symlink_metadata().is_ok() {
        let _ = std::fs::remove_file(dst).or_else(|_| std::fs::remove_dir_all(dst));
    }
    std::os::unix::fs::symlink(src, dst)
        .unwrap_or_else(|e| panic!("symlink {} -> {}: {e}", dst.display(), src.display()));
}

fn main() {
    let src = PathBuf::from(env_req("FLUX2_DEV_SNAPSHOT"));
    let dst = PathBuf::from(env_req("FLUX2_DEV_Q8"));
    let bits: i32 = std::env::var("FLUX2_DEV_BITS")
        .ok()
        .map(|b| b.parse().expect("bits"))
        .unwrap_or(8);
    let group_size: i32 = 64;
    std::fs::create_dir_all(&dst).expect("create dst dir");

    // --- transformer (DiT) ---
    let dst_transformer = dst.join("transformer");
    if dst_transformer
        .join("diffusion_pytorch_model.safetensors")
        .exists()
    {
        eprintln!("[prequant] transformer already packed — skipping");
    } else {
        eprintln!("[prequant] packing DiT → Q{bits} (group {group_size}) ...");
        let t = Instant::now();
        quantize_flux2_dit(&src.join("transformer"), &dst_transformer, bits, group_size)
            .expect("pre-quantize dev DiT");
        eprintln!("[prequant] DiT packed in {:.0}s", t.elapsed().as_secs_f32());
    }

    // --- Mistral-3 text encoder ---
    let dst_te = dst.join("text_encoder");
    if dst_te.join("model.safetensors").exists() {
        eprintln!("[prequant] text_encoder already packed — skipping");
    } else {
        eprintln!("[prequant] packing Mistral-3 TE → Q{bits} (group {group_size}) ...");
        let t = Instant::now();
        quantize_flux2_text_encoder_dir(&src.join("text_encoder"), &dst_te, bits, group_size)
            .expect("pre-quantize dev TE");
        eprintln!("[prequant] TE packed in {:.0}s", t.elapsed().as_secs_f32());
    }

    // --- borrow the unquantized components + top-level manifest ---
    for sub in ["vae", "tokenizer", "scheduler"] {
        link_borrowed(&src.join(sub), &dst.join(sub));
    }
    std::fs::copy(src.join("model_index.json"), dst.join("model_index.json"))
        .expect("copy model_index.json");

    let du = |p: &Path| {
        std::fs::metadata(p)
            .map(|m| m.len() as f64 / 1e9)
            .unwrap_or(0.0)
    };
    eprintln!(
        "[prequant] turnkey Q{bits} snapshot ready at {}\n           DiT {:.1} GB + TE {:.1} GB packed on disk",
        dst.display(),
        du(&dst_transformer.join("diffusion_pytorch_model.safetensors")),
        du(&dst_te.join("model.safetensors")),
    );
}
