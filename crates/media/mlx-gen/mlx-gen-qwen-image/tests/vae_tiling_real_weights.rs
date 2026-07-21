//! sc-11747: mechanism-level parity for the tiled Qwen-Image VAE decode ([`QwenVae::decode_tiled`]),
//! independent of any provider. Loads the Krea snapshot's `vae/` from the HF cache (env
//! `KREA_CONTROL_DIR` or the default) — byte-identical to Qwen-Image's `vae/` — and asserts:
//!   1. the head/tail decomposition is an EXACT identity of the single-pass `decode` (Δ = 0),
//!   2. the upsample tail is spatially LOCAL (a cropped head-tile's interior matches the full decode),
//!   3. the full tiled decode matches the untiled decode within blend tolerance, for both a random and
//!      a VAE-encoded latent (no seams → coherent).
//!
//!   cargo test -p mlx-gen-qwen-image --release --test vae_tiling_real_weights -- --ignored --nocapture

use mlx_gen::tiling::{SpatialTiling, TilingConfig};
use mlx_gen_qwen_image::{load_vae, QwenVae};
use mlx_rs::{random, Array};
use std::path::PathBuf;

fn base_dir() -> PathBuf {
    std::env::var("KREA_CONTROL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let snaps = PathBuf::from(std::env::var("MLX_GEN_MODELS_ROOT").expect("set MLX_GEN_MODELS_ROOT to the explicit models root (holds models--*/snapshots); inference never self-fetches or derives a cache location (epic 13657)"))
                .join("models--SceneWorks--krea-2-turbo-mlx/snapshots");
            std::fs::read_dir(&snaps)
                .unwrap()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.is_dir())
                .unwrap()
                .join("bf16")
        })
}

/// Contiguous [start, end) crop along the spatial H/W axes (3, 4) of an NCTHW array.
fn slice2d(x: &Array, h0: i32, h1: i32, w0: i32, w1: i32) -> Array {
    let hi: Vec<i32> = (h0..h1).collect();
    let wi: Vec<i32> = (w0..w1).collect();
    let x = x
        .take_axis(Array::from_slice(&hi, &[hi.len() as i32]), 3)
        .unwrap();
    x.take_axis(Array::from_slice(&wi, &[wi.len() as i32]), 4)
        .unwrap()
}

/// `(max|Δ|, mean|Δ|)`. Reshapes to 1-D first so a strided (transposed) `decode` view and a contiguous
/// tiled buffer are read in the SAME logical order — `as_slice` returns the physical buffer, so comparing
/// a transposed view directly would read it scrambled.
fn max_mean_abs(a: &Array, b: &Array) -> (f32, f64) {
    let a = a.reshape(&[-1]).unwrap();
    let b = b.reshape(&[-1]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let mut max = 0f32;
    let mut sum = 0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        max = max.max(d);
        sum += d as f64;
    }
    (max, sum / a.len() as f64)
}

fn tile_cfg() -> TilingConfig {
    TilingConfig {
        spatial: Some(SpatialTiling {
            tile_px: 512,
            overlap_px: 64,
        }),
        temporal: None,
    }
}

#[test]
#[ignore = "needs the Krea snapshot vae/ (KREA_CONTROL_DIR / HF cache)"]
fn tiled_decode_matches_untiled() {
    let vae: QwenVae = load_vae(&base_dir()).expect("load vae");

    // A seeded random latent [1,16,1,128,128] (a 1024² image). Random exercises the seams harder than a
    // smooth encoded latent, so it is the stricter reconstruction case.
    let key = random::key(7).unwrap();
    let latent = random::normal::<f32>(&[1, 16, 1, 128, 128], None, None, Some(&key)).unwrap();
    let full = vae.decode(&latent).unwrap();
    full.eval().unwrap();

    // 1. head/tail split is an exact identity of `decode`.
    let head = vae.decode_pre_upsample(&latent).unwrap();
    let manual = vae.decode_upsample_tail(&head).unwrap();
    let (split_max, _) = max_mean_abs(&manual, &full);
    assert!(
        split_max == 0.0,
        "head/tail split must reproduce decode() exactly, got max|Δ|={split_max:.3e}"
    );

    // 2. the upsample tail is spatially LOCAL: a 72-latent corner tile's deep interior matches the full
    //    decode (so overlap+blend can reconstruct). If this were large the tail had a global op.
    let crop = vae
        .decode_upsample_tail(&slice2d(&head, 0, 72, 0, 72))
        .unwrap();
    let (loc_max, _) = max_mean_abs(
        &slice2d(&crop, 128, 448, 128, 448),
        &slice2d(&full, 128, 448, 128, 448),
    );
    assert!(
        loc_max < 1e-2,
        "upsample tail is not spatially local (interior max|Δ|={loc_max:.3e}) — tiling can't work"
    );

    // 3. full tiled vs untiled (random latent) — seam-free within tolerance.
    let tiled = vae.decode_tiled(&latent, &tile_cfg(), None).unwrap();
    let (rand_max, rand_mean) = max_mean_abs(&tiled, &full);
    println!("tiled-vs-untiled (random latent):  max|Δ|={rand_max:.3e} mean|Δ|={rand_mean:.3e}");
    assert!(
        rand_max < 1.5e-1 && rand_mean < 5e-3,
        "random-latent tiled decode diverges: max|Δ|={rand_max:.3e} mean|Δ|={rand_mean:.3e}"
    );

    // 3b. a realistic (VAE-encoded) latent — the smooth case a real render decodes — is far tighter.
    let mut pixels = Vec::with_capacity(1024 * 1024 * 3);
    for y in 0..1024u32 {
        for x in 0..1024u32 {
            pixels.push((x * 255 / 1024) as u8);
            pixels.push((y * 255 / 1024) as u8);
            pixels.push(((x + y) * 127 / 2048) as u8);
        }
    }
    let img = mlx_gen::media::Image {
        width: 1024,
        height: 1024,
        pixels,
    };
    let nchw = mlx_gen::img2img::preprocess_init_image(&img, 1024, 1024).unwrap();
    let enc = vae.encode(&nchw).unwrap();
    let efull = vae.decode(&enc).unwrap();
    let etiled = vae.decode_tiled(&enc, &tile_cfg(), None).unwrap();
    let (enc_max, enc_mean) = max_mean_abs(&etiled, &efull);
    println!("tiled-vs-untiled (encoded latent): max|Δ|={enc_max:.3e} mean|Δ|={enc_mean:.3e}");
    assert!(
        enc_max < 3e-2 && enc_mean < 3e-3,
        "encoded-latent tiled decode diverges: max|Δ|={enc_max:.3e} mean|Δ|={enc_mean:.3e}"
    );
}
