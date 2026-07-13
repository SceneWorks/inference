//! SCAIL-2 **pre-quantized-snapshot** convert + load smoke (sc-5445, path B).
//!
//! Drives [`mlx_gen_scail2::quantize_scail2_dit`] over the assembled dense bf16 snapshot to produce a
//! packed Q4 (default) / Q8 `dit.safetensors` + `config.json` `quantization` manifest, symlinks the
//! unchanged VAE / UMT5 / CLIP / tokenizer alongside, then proves the result loads through
//! [`Scail2Dit::from_weights`] via the packed path (`config.json`'s manifest → `load_lin_q` builds
//! `from_quantized_parts`, never a dense bf16 weight). This is BOTH the offline converter and its load
//! gate — the low-memory-floor counterpart to the load-time `generate_*_q4_smoke` (those quantize a
//! bf16 snapshot at load; this ships the packs).
//!
//! `#[ignore]` (needs the real ~33 GB bf16 DiT). Run on macOS:
//! `cargo test -p mlx-gen-scail2 --test quantize_snapshot -- --ignored --nocapture`
//! Env: `SCAIL2_SNAPSHOT_DIR` (dense src, default `~/.cache/scail2-mlx-convert`),
//! `SCAIL2_Q4_DIR` (packed dst, default `~/.cache/scail2-mlx-q4`),
//! `SCAIL2_QUANT_BITS` (4 = Q4 default, 8 = Q8).
//!
//! To then run the full `generate()` pipeline against the *pre-quantized* snapshot (and measure its
//! real peak footprint — the sc-5445 `minMemoryGb` input), point the generate smoke at the dst:
//! `SCAIL2_SNAPSHOT_DIR=~/.cache/scail2-mlx-q4 cargo test -p mlx-gen-scail2 --test generate_smoke \
//!   -- --ignored generate_animation_smoke --nocapture`  (the snapshot carries its own quant manifest,
//! so no `with_quant` is needed — `None` loads the packs straight off disk).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_scail2::{quantize_scail2_dit, Scail2Config, Scail2Dit};
use mlx_rs::Dtype;

fn home_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
}

fn env_dir(key: &str, default: &str) -> PathBuf {
    std::env::var(key)
        .map(PathBuf::from)
        .unwrap_or_else(|_| home_dir().join(default))
}

/// Symlink every entry of `src` (except the two the converter rewrote) into `dst`, so `dst` is a
/// complete, loadable turnkey snapshot without copying the ~14 GB of VAE / UMT5 / CLIP weights.
fn link_siblings(src: &std::path::Path, dst: &std::path::Path) {
    for entry in std::fs::read_dir(src).expect("read src snapshot dir") {
        let entry = entry.expect("dir entry");
        let name = entry.file_name();
        if name == "dit.safetensors" || name == "config.json" {
            continue; // the converter writes these (packed DiT + quant manifest)
        }
        let link = dst.join(&name);
        if !link.exists() {
            std::os::unix::fs::symlink(entry.path(), &link).expect("symlink sibling component");
        }
    }
}

#[test]
#[ignore = "real ~33 GB bf16 DiT; run with --ignored on macOS (see module doc)"]
fn quantize_dit_to_packed_snapshot() {
    let src = env_dir("SCAIL2_SNAPSHOT_DIR", ".cache/scail2-mlx-convert");
    let dst = env_dir("SCAIL2_Q4_DIR", ".cache/scail2-mlx-q4");
    let bits: i32 = std::env::var("SCAIL2_QUANT_BITS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    assert!(
        src.join("dit.safetensors").exists(),
        "missing dense snapshot at {} — assemble it first (sc-5445 _materialize_bf16.py)",
        src.display()
    );

    // 1. Convert: pack the dense bf16 DiT → Q4/Q8 on disk + a `quantization` config manifest.
    quantize_scail2_dit(&src, &dst, bits, 64).expect("quantize_scail2_dit");
    link_siblings(&src, &dst);

    // 2. On-disk shape: the predicate Linears are packed (u32 codes + scales), the dense surface is
    //    untouched.
    let dit_w = Weights::from_file(dst.join("dit.safetensors")).expect("packed dit.safetensors");
    let qw = dit_w
        .get("blocks.0.self_attn.q.weight")
        .expect("packed q.weight");
    assert_eq!(qw.dtype(), Dtype::Uint32, "Q{bits} codes are u32-packed");
    assert!(
        dit_w.get("blocks.0.self_attn.q.scales").is_some(),
        "packed q.scales present"
    );
    assert!(
        dit_w.get("blocks.0.ffn.0.scales").is_some(),
        "packed ffn.0.scales present"
    );
    assert_eq!(
        dit_w.get("patch_embedding.weight").unwrap().dtype(),
        Dtype::Bfloat16,
        "patch embedding stays dense bf16"
    );

    // 3. Load through the packed path: `from_model_dir` reads the manifest → `load_lin_q` builds
    //    `from_quantized_parts` (no dense weight materialized).
    let cfg = Scail2Config::from_model_dir(&dst).expect("config with quantization manifest");
    assert_eq!(
        cfg.wan.quantization.map(|q| q.bits),
        Some(bits),
        "config.json quantization manifest parsed"
    );
    let dit = Scail2Dit::from_weights(&dit_w, &cfg).expect("load packed Scail2Dit");
    assert_eq!(dit.num_blocks(), cfg.wan.num_layers, "DiT block count");

    println!(
        "quantize_dit_to_packed_snapshot: Q{bits} snapshot at {} loaded ({} blocks); \
         run generate_smoke with SCAIL2_SNAPSHOT_DIR={} to measure the runtime floor",
        dst.display(),
        dit.num_blocks(),
        dst.display()
    );
}
