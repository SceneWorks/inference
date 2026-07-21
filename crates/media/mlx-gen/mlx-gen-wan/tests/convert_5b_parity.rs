//! sc-3238 byte-parity validation for the native Wan2.2 TI2V-5B converter
//! ([`mlx_gen_wan::convert::convert_ti2v_5b`]).
//!
//! `#[ignore]`d + heavy: needs the native `Wan-AI/Wan2.2-TI2V-5B` checkpoint (3 f32 transformer
//! shards ~20 GB + `models_t5_umt5-xxl-enc-bf16.pth` ~11 GB + `Wan2.2_VAE.pth` ~2.8 GB) and the
//! golden `wan_2_2_ti2v_5b` dir. Runs the full converter in-process and asserts `model.safetensors`
//! (825 bf16), `t5_encoder.safetensors` (242 bf16), and `vae.safetensors` (196 f32) reproduce the
//! golden byte-for-byte, with `config.json` semantically equal. Peak RSS ~30 GB (the f32 transformer).
//!
//! Run with: `cargo test -p mlx-gen-wan --test convert_5b_parity -- --ignored --nocapture`
//! Override paths with `WAN_TI2V_5B_DIR` (golden) / `WAN_5B_CKPT` (native checkpoint dir).

use std::collections::BTreeSet;
use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_wan::convert::convert_ti2v_5b;
use mlx_rs::ops::array_eq;

fn golden_dir() -> PathBuf {
    if let Ok(d) = std::env::var("WAN_TI2V_5B_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap();
    PathBuf::from(home)
        .join("Library/Application Support/SceneWorks/data/models/mlx/wan_2_2_ti2v_5b")
}

fn checkpoint_dir() -> PathBuf {
    let d = std::env::var("WAN_5B_CKPT").unwrap_or_else(|_| panic!("set WAN_5B_CKPT to the required snapshot dir; inference never self-fetches or derives a cache location (epic 13657)"));
    PathBuf::from(d)
}

fn assert_component_parity(golden: &std::path::Path, produced: &std::path::Path, name: &str) {
    let g =
        Weights::from_file(golden.join(name)).unwrap_or_else(|e| panic!("load golden {name}: {e}"));
    let p = Weights::from_file(produced.join(name))
        .unwrap_or_else(|e| panic!("load produced {name}: {e}"));
    let gk: BTreeSet<&str> = g.keys().collect();
    let pk: BTreeSet<&str> = p.keys().collect();
    let missing: Vec<&&str> = gk.difference(&pk).collect();
    let extra: Vec<&&str> = pk.difference(&gk).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "{name}: keyset mismatch — {} missing {:?}, {} extra {:?}",
        missing.len(),
        &missing[..missing.len().min(8)],
        extra.len(),
        &extra[..extra.len().min(8)],
    );
    let mut diffs = 0usize;
    for k in &gk {
        let (gt, pt) = (g.require(k).unwrap(), p.require(k).unwrap());
        if gt.shape() != pt.shape() {
            eprintln!("  {name}/{k}: shape {:?} != {:?}", gt.shape(), pt.shape());
            diffs += 1;
        } else if gt.dtype() != pt.dtype() {
            eprintln!("  {name}/{k}: dtype {:?} != {:?}", gt.dtype(), pt.dtype());
            diffs += 1;
        } else if !array_eq(gt, pt, false).unwrap().item::<bool>() {
            eprintln!("  {name}/{k}: bytes differ (dtype {:?})", gt.dtype());
            diffs += 1;
        }
    }
    assert_eq!(diffs, 0, "{name}: {diffs} of {} tensors differ", gk.len());
    eprintln!("  ✓ {name}: {} tensors byte-identical to golden", gk.len());
}

#[test]
#[ignore = "needs native Wan2.2-TI2V-5B checkpoint (~34 GB) + golden wan_2_2_ti2v_5b"]
fn ti2v_5b_convert_matches_golden() {
    let golden = golden_dir();
    let ckpt = checkpoint_dir();
    assert!(golden.is_dir(), "golden dir missing: {}", golden.display());
    assert!(ckpt.is_dir(), "checkpoint dir missing: {}", ckpt.display());

    let out = std::env::temp_dir().join("mlx_gen_wan_5b_parity_out");
    let _ = std::fs::remove_dir_all(&out);
    eprintln!("converting {} → {}", ckpt.display(), out.display());

    convert_ti2v_5b(&ckpt, &out).unwrap();

    for name in [
        "model.safetensors",
        "t5_encoder.safetensors",
        "vae.safetensors",
    ] {
        assert_component_parity(&golden, &out, name);
    }

    // config.json semantic equality.
    let parse = |p: PathBuf| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
    };
    assert_eq!(
        parse(golden.join("config.json")),
        parse(out.join("config.json")),
        "config.json semantic mismatch"
    );
    eprintln!("  ✓ config.json: semantically equal to golden");

    eprintln!("\nALL Wan TI2V-5B components byte-identical to golden ✓");
}

/// sc-4972 — materialize the dense **TI2V-5B bf16** MLX snapshot into the mlx-gen model cache (only
/// q4/q8 were cached), so the trainer e2e gates (`trainer_e2e.rs`) can run. Converts the native HF
/// checkpoint via [`convert_ti2v_5b`] and copies the shared UMT5 `tokenizer.json` (the converter
/// emits everything else: `model.safetensors` bf16, `t5_encoder.safetensors` bf16, `vae.safetensors`
/// f32, `config.json`). Tokenizer source: the already-cached I2V-A14B bf16 snapshot (same UMT5).
///
///   cargo test -p mlx-gen-wan --release --test convert_5b_parity ti2v_5b_materialize -- --ignored --nocapture
#[test]
#[ignore = "one-shot: writes ~/.cache/mlx-gen-models/wan_2_2_ti2v_5b_mlx_bf16 (needs the native HF checkpoint)"]
fn ti2v_5b_materialize_bf16_snapshot() {
    let ckpt = checkpoint_dir();
    assert!(ckpt.is_dir(), "checkpoint dir missing: {}", ckpt.display());
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let out = home.join(".cache/mlx-gen-models/wan_2_2_ti2v_5b_mlx_bf16");
    eprintln!("converting {} → {}", ckpt.display(), out.display());
    convert_ti2v_5b(&ckpt, &out).unwrap();

    // Copy the shared UMT5 tokenizer from the cached I2V-A14B bf16 snapshot (converter doesn't emit it).
    let tok_src = home.join(".cache/mlx-gen-models/wan2_2_i2v_a14b_mlx_bf16/tokenizer.json");
    assert!(
        tok_src.is_file(),
        "tokenizer source missing: {} (need a cached Wan bf16 snapshot)",
        tok_src.display()
    );
    std::fs::copy(&tok_src, out.join("tokenizer.json")).unwrap();

    for name in [
        "model.safetensors",
        "t5_encoder.safetensors",
        "vae.safetensors",
        "config.json",
        "tokenizer.json",
    ] {
        assert!(out.join(name).is_file(), "missing emitted {name}");
    }
    eprintln!(
        "\n✓ TI2V-5B bf16 snapshot materialized at {}",
        out.display()
    );
}
