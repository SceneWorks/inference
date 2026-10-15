//! sc-26268 bullet 2 + the low-memory compositions, real weights (Apple/Metal, opt-in).
//!
//! * `fp16_fix_vae_and_tiled_decode_match_the_f32_untiled_decode` — the AC's decode parity on the
//!   real `madebyollin/sdxl-vae-fp16-fix` checkpoint, with the tolerances stated below.
//! * `low_memory_compositions_q4` — the staged q4 request peak for each low-memory lever alone and
//!   all together (sequential CFG, bounded decode, fp16-fix VAE), per entry, report-only, plus the
//!   one output claim that IS asserted: sequential CFG reproduces batched CFG.
//!
//! | env var | meaning |
//! |---|---|
//! | `SDXL_LADDER_SDXL`, `SDXL_LADDER_ILLUSTRIOUS_XL_V2` | snapshot roots holding `q4/` |
//! | `SDXL_VAE_FP16_FIX_DIR` | the `madebyollin/sdxl-vae-fp16-fix` snapshot dir (rev `207b116d`) |
//! | `SDXL_LOWMEM_SIZE`, `SDXL_LOWMEM_STEPS` | as in `low_memory_peak_real_weights.rs` |
//!
//! A missing env var fails the test by name; nothing passes silently.

use std::path::PathBuf;

use mlx_gen::gen_core::{CfgBatching, GenerationMemory};
use mlx_gen::weights::Weights;
use mlx_gen::WeightsSource;
use mlx_rs::{Array, Dtype};

use mlx_gen_sdxl::config::VaeConfig;
use mlx_gen_sdxl::loader::{load_vae_fp16_fix, resolve_vae_fp16_fix_file, VAE_FP16_FIX_COMPONENT};
use mlx_gen_sdxl::memory_strategy::{DECODE_OVERLAP, DECODE_TILE_EDGES};
use mlx_gen_sdxl::pipeline::decoded_to_image;
use mlx_gen_sdxl::vae::Autoencoder;

use super::low_memory_peak_real_weights::{measure, q4_spec, staged_request};

/// fp16-fix VAE at f16 vs the same checkpoint at f32, untiled, in 8-bit pixels: no pixel may move
/// by more than 3/255 and the mean must stay under 0.25/255. f16's 11-bit significand rounds at
/// ~0.05% per op; three levels is the budget for ~20 normalized stages compounding, and an overflow
/// or promotion defect (what this guards) moves pixels by tens of levels.
const FP16_MAX_PIXEL_DELTA: u32 = 3;
const FP16_MEAN_PIXEL_DELTA: f64 = 0.25;
/// f32 tiled vs f32 untiled: the same arithmetic up to reassociation of halo-cropped convolutions,
/// so at most one 8-bit level (a rounding-boundary flip) anywhere.
const F32_TILED_MAX_PIXEL_DELTA: u32 = 1;

fn env_path(var: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| {
        panic!("set {var}; inference never self-fetches or derives a cache location (epic 13657)")
    }))
}

fn max_delta(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| u32::from(x.abs_diff(*y)))
        .max()
        .unwrap_or(0)
}

fn mean_delta(a: &[u8], b: &[u8]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(x.abs_diff(*y)))
        .sum::<f64>()
        / a.len() as f64
}

/// A deterministic, structured RGB test image in `[-1, 1]` (NHWC), so its latent has real edges.
fn test_image(edge: i32) -> Array {
    let mut v = Vec::with_capacity((edge * edge * 3) as usize);
    for y in 0..edge {
        for x in 0..edge {
            let (fx, fy) = (x as f32 / edge as f32, y as f32 / edge as f32);
            v.push((fx * 9.0).sin() * 0.8);
            v.push((fy * 13.0 + fx * 3.0).cos() * 0.7);
            v.push(((fx - 0.5).powi(2) + (fy - 0.5).powi(2)).sqrt() * 2.0 - 0.6);
        }
    }
    Array::from_slice(&v, &[1, edge, edge, 3])
}

#[test]
#[ignore = "real weights + Apple GPU: set SDXL_VAE_FP16_FIX_DIR"]
fn fp16_fix_vae_and_tiled_decode_match_the_f32_untiled_decode() {
    let source = WeightsSource::Dir(env_path("SDXL_VAE_FP16_FIX_DIR"));
    let half = load_vae_fp16_fix(&source).expect("load the fp16-fix VAE at f16");
    assert_eq!(half.dtype(), Dtype::Float16);
    let mut w = Weights::from_file(resolve_vae_fp16_fix_file(&source).unwrap()).unwrap();
    w.cast_all(Dtype::Float32).unwrap();
    let full = Autoencoder::from_weights(&w, &VaeConfig::sdxl_base()).unwrap();
    w.materialize_accessed().unwrap();

    for edge in [512_i32, 1024] {
        let latent = full.encode_mean(&test_image(edge)).unwrap();
        latent.eval().unwrap();
        let reference = decoded_to_image(&full.decode(&latent).unwrap()).unwrap();
        let fp16 = decoded_to_image(&half.decode(&latent).unwrap()).unwrap();
        let (max, mean) = (
            max_delta(&reference.pixels, &fp16.pixels),
            mean_delta(&reference.pixels, &fp16.pixels),
        );
        eprintln!("[sc-26268 parity] {edge}² fp16 untiled: max {max}/255 mean {mean:.4}");
        assert!(max <= FP16_MAX_PIXEL_DELTA && mean <= FP16_MEAN_PIXEL_DELTA);

        for tile in DECODE_TILE_EDGES.iter().filter(|t| (**t as i32) < edge) {
            let tiling =
                mlx_gen::tiling::TilingConfig::spatial_only(*tile as i32, DECODE_OVERLAP as i32);
            let tiled32 =
                decoded_to_image(&full.decode_tiled(&latent, &tiling, None).unwrap()).unwrap();
            let tiled16 =
                decoded_to_image(&half.decode_tiled(&latent, &tiling, None).unwrap()).unwrap();
            let max32 = max_delta(&reference.pixels, &tiled32.pixels);
            let max16 = max_delta(&reference.pixels, &tiled16.pixels);
            let mean16 = mean_delta(&reference.pixels, &tiled16.pixels);
            eprintln!(
                "[sc-26268 parity] {edge}² tile {tile}: f32 max {max32}/255; fp16 max {max16}/255 \
                 mean {mean16:.4}"
            );
            assert!(
                max32 <= F32_TILED_MAX_PIXEL_DELTA,
                "f32 tiled edge {tile}: {max32}/255"
            );
            assert!(
                max16 <= FP16_MAX_PIXEL_DELTA && mean16 <= FP16_MEAN_PIXEL_DELTA,
                "fp16 tiled edge {tile}: max {max16}/255 mean {mean16:.4}"
            );
        }
    }
}

#[test]
#[ignore = "real weights + Apple GPU: set SDXL_LADDER_SDXL, SDXL_LADDER_ILLUSTRIOUS_XL_V2, SDXL_VAE_FP16_FIX_DIR"]
fn low_memory_compositions_q4() {
    let fix = WeightsSource::Dir(env_path("SDXL_VAE_FP16_FIX_DIR"));
    let tiled = GenerationMemory {
        tile_vae_decode: true,
        decode_tile_edge: Some(256),
        decode_overlap: Some(DECODE_OVERLAP),
        ..Default::default()
    };
    let sequential = GenerationMemory {
        cfg_batching: Some(CfgBatching::Sequential),
        ..Default::default()
    };
    for (entry, var) in [
        ("sdxl", "SDXL_LADDER_SDXL"),
        ("illustrious_xl_v2", "SDXL_LADDER_ILLUSTRIOUS_XL_V2"),
    ] {
        let dir = env_path(var).join("q4");
        let plain = q4_spec(&dir);
        let with_fix = plain
            .clone()
            .with_component(VAE_FP16_FIX_COMPONENT, fix.clone());
        let staged = measure(&plain, &staged_request(GenerationMemory::default()));
        let seq = measure(&plain, &staged_request(sequential));
        let tile = measure(&plain, &staged_request(tiled));
        let vae16 = measure(&with_fix, &staged_request(GenerationMemory::default()));
        let all = measure(
            &with_fix,
            &staged_request(GenerationMemory {
                cfg_batching: Some(CfgBatching::Sequential),
                ..tiled
            }),
        );
        for (name, row) in [
            ("staged", &staged),
            ("+ sequential CFG", &seq),
            ("+ tiled decode 256", &tile),
            ("+ fp16-fix VAE", &vae16),
            ("all three", &all),
        ] {
            eprintln!(
                "[sc-26268 composition] {entry} q4: {name:<20} peak {:.4} GiB ({:+.2}% vs staged)",
                row.peak_gib,
                (row.peak_gib / staged.peak_gib - 1.0) * 100.0
            );
        }
        let (max, mean) = (
            max_delta(&staged.pixels, &seq.pixels),
            mean_delta(&staged.pixels, &seq.pixels),
        );
        eprintln!("[sc-26268 composition] {entry}: sequential vs batched CFG max {max}/255 mean {mean:.4}");
        // The shared `CfgBatching` contract declares the class bit-identical.
        assert_eq!(max, 0, "{entry}: sequential CFG changed the image");
    }
}
