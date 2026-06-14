//! SCAIL-2 turnkey-snapshot **load smoke** (sc-5445).
//!
//! Proves the assembled `SceneWorks/scail2-mlx` snapshot is loadable by all four Rust component
//! loaders — the converted DiT, the reused Wan2.1 z16 VAE + UMT5 text encoder, and the de-prefixed
//! open-CLIP ViT-H/14 visual tower — plus the UMT5 tokenizer. This is the gate for sc-5445 (snapshot
//! assembly) and the prerequisite for the live `generate` wiring (sc-5443) and the real-weight 40-layer
//! DiT parity (sc-5446).
//!
//! Snapshot layout (`~/.cache/scail2-mlx-convert`, override with `SCAIL2_SNAPSHOT_DIR`):
//!   * `dit.safetensors`         — SCAIL-2 DiT (bf16)            → [`Scail2Dit`]
//!   * `vae.safetensors`         — Wan2.1 z16 VAE (f32)          → [`mlx_gen_wan::WanVae`]
//!   * `t5_encoder.safetensors`  — umt5-xxl encoder (bf16)       → [`mlx_gen_wan::Umt5Encoder`]
//!   * `clip.safetensors`        — open-CLIP ViT-H/14 (f32)      → [`ScailClip`]
//!   * `tokenizer.json`          — umt5-xxl HF tokenizer         → [`mlx_gen_wan::load_tokenizer`]
//!   * `config.json`             — dims sidecar                  → [`Scail2Config`]
//!
//! Regenerate the snapshot's converted safetensors on the Mac (torch venv):
//!
//! ```text
//! ~/mlx-flux-venv/bin/python _vendor/scail2/_convert_vae_t5_safetensors.py   # vae + t5_encoder + tokenizer
//! ~/mlx-flux-venv/bin/python _vendor/scail2/_gen_clip_realweight_fixtures.py # clip.safetensors
//! ```
//! (`dit.safetensors` + `config.json` come from the DiT converter; `vae`/`t5_encoder` mirror
//! `mlx_gen_wan::convert::convert_dual_a14b` steps 3-4 on SCAIL-2's own stock Wan2.1 VAE + umt5-xxl.)
//!
//! `#[ignore]` (needs the ~46 GB real snapshot, off CI). Run on macOS with:
//! `cargo test -p mlx-gen-scail2 --test snapshot_load -- --ignored --nocapture`.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_scail2::{ClipVisionConfig, Scail2Config, Scail2Dit, ScailClip};
use mlx_gen_wan::{load_tokenizer, Umt5Encoder, WanVae};

fn home_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
}

fn snapshot_dir() -> PathBuf {
    std::env::var("SCAIL2_SNAPSHOT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home_dir().join(".cache/scail2-mlx-convert"))
}

#[test]
#[ignore = "real ~46 GB snapshot (see module doc); run with --ignored on macOS"]
fn snapshot_load_smoke() {
    let root = snapshot_dir();
    let cfg = Scail2Config::from_model_dir(&root).expect("config.json");

    // 1. DiT — the converted 40-layer SCAIL-2 transformer.
    let dit_w = Weights::from_file(root.join("dit.safetensors")).expect("dit.safetensors");
    let dit = Scail2Dit::from_weights(&dit_w, &cfg).expect("load Scail2Dit");
    assert_eq!(dit.num_blocks(), cfg.wan.num_layers, "DiT block count");
    assert_eq!(cfg.wan.num_layers, 40, "SCAIL-2 14B is 40 layers");
    println!("Scail2Dit: {} blocks loaded", dit.num_blocks());
    drop((dit, dit_w));

    // 2. VAE — the reused Wan2.1 z16 VAE (encode + decode).
    let vae_w = Weights::from_file(root.join("vae.safetensors")).expect("vae.safetensors");
    let _vae = WanVae::from_weights(&vae_w).expect("load WanVae");
    println!("WanVae: loaded");
    drop(vae_w);

    // 3. UMT5 text encoder — reused umt5-xxl.
    let t5_w =
        Weights::from_file(root.join("t5_encoder.safetensors")).expect("t5_encoder.safetensors");
    let _t5 = Umt5Encoder::from_weights(&t5_w, &cfg.wan).expect("load Umt5Encoder");
    println!("Umt5Encoder: loaded");
    drop(t5_w);

    // 4. CLIP visual tower — de-prefixed open-CLIP ViT-H/14.
    let clip_w = Weights::from_file(root.join("clip.safetensors")).expect("clip.safetensors");
    let _clip =
        ScailClip::from_weights(&clip_w, &ClipVisionConfig::vit_h_14()).expect("load ScailClip");
    println!("ScailClip: loaded");
    drop(clip_w);

    // 5. UMT5 tokenizer (text_len 512 per config.json).
    let _tok = load_tokenizer(root.join("tokenizer.json"), 512).expect("load_tokenizer");
    println!("umt5 tokenizer: loaded");

    println!(
        "snapshot_load_smoke: all 4 component loaders + tokenizer OK from {}",
        root.display()
    );
}
