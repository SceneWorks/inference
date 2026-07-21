//! sc-12438 — **real-weights validation** of the Wan z16 VAE decode across its write-bound stage
//! boundary, on the real `vae.safetensors` (`#[ignore]`d, GPU-heavy). This is the evidence that
//! `VaeTiling::WAN.full_res_channels = 96` is correct — NOT the structurally-tempting 192.
//!
//! The last `UpsampleBlock` does `upsample_nearest` (at its **input** width) then `Conv2d(C→C/2)`, which
//! *looks* like a 192-ch full-resolution transient before the 96-ch reduction. If that transient
//! materialized as one write, a single pass past `96·f·512² = 2^31` (f≈42 at 512²) would corrupt. It
//! does **not**: MLX fuses the broadcast/reshape/conv, so no single >i32::MAX buffer is written there.
//! Measured single-pass-vs-tiled mean|Δ| at 512² (tiled = the correct reference; 256px/16f tiles keep
//! every tile far under the bound):
//!
//! | frames | mean\|Δ\| | note |
//! |---:|---:|---|
//! | 44 | 0.0397 | baseline blend tolerance |
//! | 60 | 0.0429 | |
//! | 68 | 0.0440 | |
//! | 84 | 0.0547 | at/near the 96-ch cap (85f); still ≈ blend, NOT corruption |
//! | 92 | 0.0652 | past the 96-ch cap; gentle rise, not the ±0.3+ of real corruption |
//! | 120 | — | single-pass **errors** (does not silently corrupt) |
//!
//! So the effective materialized width is the 96-ch res-block stage: a single pass is exact through the
//! 96-ch cap and merely *errors* far past it — never the silent conv-style corruption a 192 write would
//! produce. Compare the sharp conv3d/pad corruption in `mlx-gen/tests/mlx_write_bound_probe.rs` (a bare
//! materialized op at the exact boundary) — the fused decode does not exhibit it at these sizes.
//!
//! Run:
//! ```text
//! Z16_VAE=/path/to/models--SceneWorks--wan2.2-t2v-a14b-mlx/snapshots/<h>/bf16/vae.safetensors \
//!   cargo test -p mlx-gen-wan --test vae16_write_bound_real -- --ignored --nocapture
//! ```
//! (With `Z16_VAE` unset it auto-discovers the bf16 snapshot under the HF cache, else skips.)

use std::path::PathBuf;

use mlx_gen::tiling::{SpatialTiling, TemporalTiling, TilingConfig, VaeTiling, MAX_WRITABLE_ELEMS};
use mlx_gen::weights::Weights;
use mlx_gen_wan::WanVae;
use mlx_rs::random;

fn discover_vae() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("Z16_VAE") {
        return Some(PathBuf::from(p));
    }
    let base = PathBuf::from(std::env::var_os("MLX_GEN_MODELS_ROOT")?)
        .join("models--SceneWorks--wan2.2-t2v-a14b-mlx/snapshots");
    let snap = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    let p = snap.join("bf16/vae.safetensors");
    p.exists().then_some(p)
}

/// Decode `out 512×512×F` single-pass and tiled, and return their mean absolute difference over the RGB
/// output. A small config (256 px / 16 frame tiles) keeps every tile's decoder-stage write far under the
/// bound, so the tiled decode is the correct reference; the single pass diverges only if it corrupts.
fn single_vs_tiled_diff(vae: &WanVae, frames: i32) -> f64 {
    let (out_h, out_w) = (512i32, 512i32);
    let (z, t_lat, h_lat, w_lat) = (16, frames / 4, out_h / 8, out_w / 8);
    let key = random::key(0).unwrap();
    let latent =
        random::normal::<f32>(&[1, z, t_lat, h_lat, w_lat], None, None, Some(&key)).unwrap();

    let single = vae.decode(&latent).unwrap();
    mlx_rs::transforms::eval([&single]).unwrap();

    let cfg = TilingConfig {
        spatial: Some(SpatialTiling {
            tile_px: 256,
            overlap_px: 64,
        }),
        temporal: Some(TemporalTiling {
            tile_frames: 16,
            overlap_frames: 8,
        }),
    };
    let tiled = vae.decode_tiled(&latent, &cfg, None).unwrap();
    mlx_rs::transforms::eval([&tiled]).unwrap();

    let (a, b) = (single.as_slice::<f32>(), tiled.as_slice::<f32>());
    assert_eq!(a.len(), b.len());
    let sum: f64 = a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).sum();
    sum / a.len() as f64
}

/// A single-pass z16 decode within the 96-ch write cap is EXACT vs the tiled decode (blend tolerance) —
/// i.e. `full_res_channels = 96` is not under-stated (no silent corruption below the guard threshold).
/// Discriminating: a real conv-style corruption would push mean|Δ| toward ~0.3–0.6 (values span [-1,1]),
/// an order of magnitude over the ~0.05 blend tolerance asserted here.
#[test]
#[ignore = "sc-12438 real z16 vae decode validation; needs Z16_VAE, GPU-heavy"]
fn wan_z16_single_pass_matches_tiled_within_the_96ch_cap() {
    let Some(vae_path) = discover_vae() else {
        eprintln!("skip: no Z16_VAE and no bf16 vae.safetensors under the HF cache");
        return;
    };
    let weights = Weights::from_file(&vae_path).expect("read z16 vae.safetensors");
    let vae = WanVae::from_weights(&weights).expect("WanVae::from_weights");

    let cap = VaeTiling::WAN.writable_frame_cap(512, 512); // 96-ch cap at 512² = 85 frames
    for f in [40i32, 84] {
        assert!(
            (f as i64) <= cap,
            "geometry {f}f must stay within the 96-ch single-pass cap {cap}"
        );
        // Sanity that a 96-ch write at this frame count is under the bound (the premise of "no corruption
        // here"): 96·f·512² ≤ 2^31.
        assert!(96 * f as i64 * 512 * 512 <= MAX_WRITABLE_ELEMS);
        let d = single_vs_tiled_diff(&vae, f);
        eprintln!("[z16 real] frames={f:3}  single-vs-tiled mean|Δ|={d:.5}  (cap={cap})");
        assert!(
            d < 0.1,
            "single-pass z16 decode at {f}f diverged from the tiled decode by {d:.5} — a 96-ch decode \
             within the cap must be exact (blend tolerance ~0.05). A large value would mean 96 is \
             under-stated and the widest materialized write is wider."
        );
    }
}
