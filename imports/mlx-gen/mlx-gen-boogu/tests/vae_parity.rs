//! E4 (sc-6392) — real-weight parity for the Boogu VAE decode. Boogu's FLUX.1 16-ch `AutoencoderKL`
//! loads into the reused `mlx_gen_z_image::vae::Vae`; this checks the decode matches the diffusers
//! `AutoencoderKL` reference on the same latent (de-normalize + decoder, scale/shift 0.3611/0.1159).
//!
//! `#[ignore]` — needs the Base snapshot (`vae/`) + the golden (`tools/dump_boogu_vae_golden.py`):
//!   BOOGU_BASE_DIR=<snapshot> BOOGU_VAE_GOLDEN=<...>/boogu_vae.safetensors \
//!     CARGO_TARGET_DIR=~/Repos/mlx-gen/target \
//!     cargo test -p mlx-gen-boogu --test vae_parity -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_boogu::load_vae;
use mlx_rs::ops::{multiply, sqrt, sum};
use mlx_rs::{Array, Dtype};

fn cosine(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let dot = sum(multiply(&a, &b).unwrap(), false).unwrap();
    let na = sqrt(sum(multiply(&a, &a).unwrap(), false).unwrap()).unwrap();
    let nb = sqrt(sum(multiply(&b, &b).unwrap(), false).unwrap()).unwrap();
    (dot / (na * nb)).item::<f32>()
}

fn snapshot_dir() -> PathBuf {
    PathBuf::from(std::env::var("BOOGU_BASE_DIR").expect("set BOOGU_BASE_DIR to the snapshot root"))
}

fn golden_path() -> PathBuf {
    std::env::var("BOOGU_VAE_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME"))
                .join("Repos/mlx-gen-wt-boogu/reference/goldens/boogu_vae.safetensors")
        })
}

#[test]
#[ignore = "needs real weights + golden (tools/dump_boogu_vae_golden.py)"]
fn vae_decode_matches_reference() {
    let g = Weights::from_file(golden_path()).expect("golden — run tools/dump_boogu_vae_golden.py");
    let vae = load_vae(snapshot_dir()).expect("load Boogu FLUX.1 VAE");

    // `Vae::decode` de-normalizes (z/scaling + shift) internally, so feed the raw latent.
    let out = vae.decode(g.require("z").unwrap()).unwrap();
    let want = g.require("golden").unwrap();
    assert_eq!(
        out.shape(),
        want.shape(),
        "decoded image shape (NCHW [1,3,1,H,W])"
    );

    let c = cosine(&out, want);
    println!("Boogu VAE decode parity cosine = {c:.7}");
    assert!(
        c > 0.999,
        "VAE decode parity cosine {c} too low — structural mismatch"
    );
}

#[test]
#[ignore = "needs real weights + golden (tools/dump_boogu_vae_golden.py)"]
fn vae_encode_matches_reference() {
    let g = Weights::from_file(golden_path()).expect("golden — run tools/dump_boogu_vae_golden.py");
    let vae = load_vae(snapshot_dir()).expect("load Boogu FLUX.1 VAE");

    // img2img path: encoder → posterior mean → (mean − shift) · scaling.
    let out = vae.encode(g.require("img_in").unwrap()).unwrap();
    let want = g.require("enc_golden").unwrap();
    assert_eq!(
        out.shape(),
        want.shape(),
        "encoded latent shape [1,16,H/8,W/8]"
    );

    let c = cosine(&out, want);
    println!("Boogu VAE encode parity cosine = {c:.7}");
    assert!(
        c > 0.999,
        "VAE encode parity cosine {c} too low — structural mismatch"
    );
}
