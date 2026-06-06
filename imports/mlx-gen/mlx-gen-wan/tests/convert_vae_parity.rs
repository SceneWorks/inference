//! sc-3237 byte-parity validation for the native Wan2.2 VAE converter
//! ([`mlx_gen_wan::convert::convert_vae22`]) + the torch `.pth` reader ([`mlx_gen_wan::pth`]).
//!
//! `#[ignore]`d: needs the cached native source `Wan2.2_VAE.pth` (~2.8 GB, all FloatStorage) and the
//! golden `wan_2_2_ti2v_5b/vae.safetensors` (196 f32 tensors). Reads the `.pth` end-to-end (zip +
//! pickle VM + storage decode), applies the Wan2.2 VAE sanitizer, and asserts the result reproduces
//! the golden byte-for-byte.
//!
//! Run with: `cargo test -p mlx-gen-wan --test convert_vae_parity -- --ignored --nocapture`
//! Override paths with `WAN_TI2V_5B_DIR` (golden) / `WAN_VAE_PTH` (source .pth).

use std::collections::BTreeSet;
use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_wan::convert::convert_vae22;
use mlx_rs::ops::array_eq;

fn golden_vae() -> PathBuf {
    if let Ok(d) = std::env::var("WAN_TI2V_5B_DIR") {
        return PathBuf::from(d).join("vae.safetensors");
    }
    let home = std::env::var("HOME").unwrap();
    PathBuf::from(home).join(
        "Library/Application Support/SceneWorks/data/models/mlx/wan_2_2_ti2v_5b/vae.safetensors",
    )
}

fn source_pth() -> PathBuf {
    if let Ok(p) = std::env::var("WAN_VAE_PTH") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snapshots =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots");
    std::fs::read_dir(&snapshots)
        .unwrap_or_else(|_| panic!("no HF snapshots at {}", snapshots.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.join("Wan2.2_VAE.pth").is_file())
        .unwrap_or_else(|| panic!("Wan2.2_VAE.pth not found under {}", snapshots.display()))
        .join("Wan2.2_VAE.pth")
}

#[test]
#[ignore = "needs Wan2.2_VAE.pth (~2.8 GB) + golden wan_2_2_ti2v_5b/vae.safetensors"]
fn ti2v_5b_vae_convert_matches_golden() {
    let golden_path = golden_vae();
    let source = source_pth();
    assert!(
        golden_path.is_file(),
        "golden vae missing: {}",
        golden_path.display()
    );
    assert!(
        source.is_file(),
        "source .pth missing: {}",
        source.display()
    );

    let out = std::env::temp_dir().join("mlx_gen_wan_vae_parity.safetensors");
    let _ = std::fs::remove_file(&out);
    eprintln!("converting {} → {}", source.display(), out.display());

    // TI2V keeps the encoder (include_encoder = true).
    convert_vae22(&source, &out, true).unwrap();

    let g = Weights::from_file(&golden_path).unwrap();
    let p = Weights::from_file(&out).unwrap();

    let gk: BTreeSet<&str> = g.keys().collect();
    let pk: BTreeSet<&str> = p.keys().collect();
    let missing: Vec<&&str> = gk.difference(&pk).collect();
    let extra: Vec<&&str> = pk.difference(&gk).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "keyset mismatch — {} missing {:?}, {} extra {:?}",
        missing.len(),
        &missing[..missing.len().min(8)],
        extra.len(),
        &extra[..extra.len().min(8)],
    );

    let mut diffs = 0usize;
    for k in &gk {
        let (gt, pt) = (g.require(k).unwrap(), p.require(k).unwrap());
        if gt.shape() != pt.shape() {
            eprintln!("  {k}: shape {:?} != {:?}", gt.shape(), pt.shape());
            diffs += 1;
        } else if gt.dtype() != pt.dtype() {
            eprintln!("  {k}: dtype {:?} != {:?}", gt.dtype(), pt.dtype());
            diffs += 1;
        } else if !array_eq(gt, pt, false).unwrap().item::<bool>() {
            eprintln!(
                "  {k}: bytes differ (dtype {:?}, shape {:?})",
                gt.dtype(),
                gt.shape()
            );
            diffs += 1;
        }
    }
    assert_eq!(
        diffs,
        0,
        "{diffs} of {} tensors differ from golden",
        gk.len()
    );
    eprintln!(
        "\n✓ all {} VAE tensors byte-identical to golden wan_2_2_ti2v_5b ✓",
        gk.len()
    );
}
