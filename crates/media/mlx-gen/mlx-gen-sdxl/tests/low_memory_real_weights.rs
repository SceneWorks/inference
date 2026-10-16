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
/// f32 tiled vs f32 untiled. The tiled tail is the same arithmetic as the dense decode (1-px halo
/// per 3×3 conv, full-image GroupNorm statistics) — `vae::tests::
/// tiled_decode_matches_the_f32_untiled_decode_across_edges` holds it to 1e-4 of peak on the CPU,
/// where f32 really is f32. On Apple Metal it is not: measured on an M5 Max (sc-26268 fix probe),
/// one f32 3×3 conv at 160²×{128,256} differs from the CPU f32 result by 1.3–1.4e-2 of peak, and
/// the halo-tiled conv from the untiled one by 1.6–1.8e-2 — the same class as an f16 conv
/// (1.7–1.9e-2), because the GPU's f32 matmul/conv kernels reduce at reduced internal precision
/// and pick their kernel by spatial size. So the 8-bit budget for f32 tiled-vs-untiled on Metal is
/// the f16 budget, not "one rounding flip": the GPU run measured 1/255 at seven edges and 2/255 at
/// 1024² edge 640 (fp16 untiled was 2/255 at 1024² too). A seam defect — a short halo, per-tile
/// statistics — is the 38–132/255 class this story fixed and stays far outside both bounds.
const F32_TILED_MAX_PIXEL_DELTA: u32 = FP16_MAX_PIXEL_DELTA;
const F32_TILED_MEAN_PIXEL_DELTA: f64 = FP16_MEAN_PIXEL_DELTA;

/// The stated memory bound of `fp16_fix_vae_and_tiled_decode_match_the_f32_untiled_decode`: every
/// single decode's MLX peak (active bytes, reset before it) must stay under this. MLX's own limit
/// is set to it as backpressure and its buffer cache to [`PARITY_CACHE_LIMIT_BYTES`], so freed
/// buffers are returned rather than parked: the unbounded run (default limits, every decode
/// sharing one cache) grew the process footprint to 102 GiB.
const PARITY_DECODE_PEAK_GIB: f64 = 16.0;
const PARITY_CACHE_LIMIT_BYTES: usize = 512 << 20;
const GIB: f64 = (1u64 << 30) as f64;

/// Sets MLX's memory limit and cache limit for one test and restores both on drop (also on panic).
struct MemoryCap {
    memory: usize,
    cache: usize,
}

impl MemoryCap {
    fn new(memory_bytes: usize, cache_bytes: usize) -> Self {
        let memory = mlx_rs::memory::set_memory_limit(memory_bytes);
        let cache = mlx_rs::memory::set_cache_limit(cache_bytes);
        mlx_rs::memory::clear_cache();
        Self { memory, cache }
    }
}

impl Drop for MemoryCap {
    fn drop(&mut self) {
        mlx_rs::memory::set_memory_limit(self.memory);
        mlx_rs::memory::set_cache_limit(self.cache);
    }
}

/// Run one decode to host pixels with a fresh peak counter; every device array it built is dropped
/// and the cache cleared before returning, so no decode's buffers outlive it. Returns the pixels and
/// that decode's MLX peak in GiB.
fn bounded_decode(decode: impl FnOnce() -> mlx_gen::Result<Array>) -> (Vec<u8>, f64) {
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let pixels = {
        let decoded = decode().expect("decode");
        decoded_to_image(&decoded).expect("to image").pixels
    };
    let peak = mlx_rs::memory::get_peak_memory() as f64 / GIB;
    mlx_rs::memory::clear_cache();
    (pixels, peak)
}

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

/// The 1024² latent from four 512² quadrant encodes, so the 1024² parity never runs a 1024² f32
/// encode (as large as the decode it feeds). The latent's provenance is not under test — the
/// decodes are — and each quadrant is a real encoder output with real edges.
fn quadrant_latent(vae: &Autoencoder, edge: i32) -> Array {
    let image = test_image(edge);
    if edge <= 512 {
        let latent = vae.encode_mean(&image).unwrap();
        latent.eval().unwrap();
        return latent;
    }
    let half = edge / 2;
    let crop = |y: i32, x: i32| {
        let rows = image
            .take_axis(
                Array::from_slice(&(y..y + half).collect::<Vec<_>>(), &[half]),
                1,
            )
            .unwrap();
        let tile = rows
            .take_axis(
                Array::from_slice(&(x..x + half).collect::<Vec<_>>(), &[half]),
                2,
            )
            .unwrap();
        let latent = vae.encode_mean(&tile).unwrap();
        latent.eval().unwrap();
        mlx_rs::memory::clear_cache();
        latent
    };
    let top = mlx_rs::ops::concatenate_axis(&[&crop(0, 0), &crop(0, half)], 2).unwrap();
    let bottom = mlx_rs::ops::concatenate_axis(&[&crop(half, 0), &crop(half, half)], 2).unwrap();
    let latent = mlx_rs::ops::concatenate_axis(&[&top, &bottom], 1).unwrap();
    latent.eval().unwrap();
    latent
}

/// A labelled decode with its max/mean pixel bounds.
type Decode<'a> = (
    String,
    u32,
    f64,
    Box<dyn Fn() -> mlx_gen::Result<Array> + 'a>,
);

/// One measured parity row.
struct ParityRow {
    label: String,
    max: u32,
    mean: f64,
    max_bound: u32,
    mean_bound: f64,
    peak_gib: f64,
}

/// Optional diagnostic selection. No selectors means the complete supported sweep; selecting a
/// tile keeps BOTH compute widths compared against a genuinely untiled f32 reference.
fn parity_cases(size: Option<i32>, tile: Option<u32>) -> Result<Vec<(i32, Vec<u32>)>, String> {
    if size.is_some_and(|s| ![512, 1024].contains(&s)) {
        return Err("SDXL_VAE_PARITY_SIZE must be 512 or 1024".into());
    }
    if tile.is_some_and(|t| !DECODE_TILE_EDGES.contains(&t)) {
        return Err("SDXL_VAE_PARITY_TILE must be a supported decode edge".into());
    }
    let cases = [512, 1024]
        .into_iter()
        .filter(|s| size.is_none_or(|wanted| *s == wanted))
        .map(|s| {
            (
                s,
                DECODE_TILE_EDGES
                    .iter()
                    .copied()
                    .filter(|t| (*t as i32) < s && tile.is_none_or(|wanted| *t == wanted))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    if cases.iter().all(|(_, tiles)| tiles.is_empty()) {
        return Err("selected tile must be smaller than the selected image".into());
    }
    Ok(cases)
}

#[test]
fn parity_diagnostic_selection_preserves_both_widths_and_default_coverage() {
    let full = parity_cases(None, None).unwrap();
    for (edge, tiles) in &full {
        assert_eq!(
            tiles,
            &DECODE_TILE_EDGES
                .iter()
                .copied()
                .filter(|t| (*t as i32) < *edge)
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(
        full.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
        vec![512, 1024]
    );
    assert_eq!(
        parity_cases(Some(1024), Some(256)).unwrap(),
        vec![(1024, vec![256])]
    );
    assert!(parity_cases(Some(512), Some(896)).is_err());
    assert!(parity_cases(Some(256), None).is_err());
    assert!(parity_cases(None, Some(1)).is_err());
}

/// Memory-bounded (see [`PARITY_DECODE_PEAK_GIB`]): one decode at a time, each with its own peak
/// counter and a cleared cache, and the run stops at the first decode over the bound. Every row is
/// printed before anything is asserted, so one failing edge never hides the others.
#[test]
#[ignore = "real weights + Apple GPU: set SDXL_VAE_FP16_FIX_DIR"]
fn fp16_fix_vae_and_tiled_decode_match_the_f32_untiled_decode() {
    let size = std::env::var("SDXL_VAE_PARITY_SIZE").ok().map(|v| {
        v.parse::<i32>()
            .expect("SDXL_VAE_PARITY_SIZE is an integer")
    });
    let tile = std::env::var("SDXL_VAE_PARITY_TILE").ok().map(|v| {
        v.parse::<u32>()
            .expect("SDXL_VAE_PARITY_TILE is an integer")
    });
    let cases = parity_cases(size, tile).expect("valid parity selection");
    let expected = cases
        .iter()
        .map(|(_, tiles)| usize::from(tile.is_none()) + 2 * tiles.len())
        .sum::<usize>();
    eprintln!(
        "[sc-26268 parity] selection size={size:?} tile={tile:?}; expected comparisons={expected}"
    );
    let _cap = MemoryCap::new(
        (PARITY_DECODE_PEAK_GIB * GIB) as usize,
        PARITY_CACHE_LIMIT_BYTES,
    );
    let source = WeightsSource::Dir(env_path("SDXL_VAE_FP16_FIX_DIR"));
    let half = load_vae_fp16_fix(&source).expect("load the fp16-fix VAE at f16");
    assert_eq!(half.dtype(), Dtype::Float16);
    let full = {
        let mut w = Weights::from_file(resolve_vae_fp16_fix_file(&source).unwrap()).unwrap();
        w.cast_all(Dtype::Float32).unwrap();
        let full = Autoencoder::from_weights(&w, &VaeConfig::sdxl_base()).unwrap();
        w.materialize_accessed().unwrap();
        full
    };
    mlx_rs::memory::clear_cache();

    let mut rows: Vec<ParityRow> = Vec::new();
    let mut over_bound = false;
    'edges: for (edge, tiles) in cases {
        let latent = quadrant_latent(&full, edge);
        let (reference, ref_peak) = bounded_decode(|| full.decode_evaluated(&latent));
        let mut decodes: Vec<Decode<'_>> = Vec::new();
        if tile.is_none() {
            decodes.push((
                format!("{edge}² fp16 untiled"),
                FP16_MAX_PIXEL_DELTA,
                FP16_MEAN_PIXEL_DELTA,
                Box::new(|| half.decode_evaluated(&latent)),
            ));
        }
        for tile in &tiles {
            let tiling =
                mlx_gen::tiling::TilingConfig::spatial_only(*tile as i32, DECODE_OVERLAP as i32);
            let t2 = tiling;
            let latent = &latent;
            let (full, half) = (&full, &half);
            decodes.push((
                format!("{edge}² f32 tile {tile}"),
                F32_TILED_MAX_PIXEL_DELTA,
                F32_TILED_MEAN_PIXEL_DELTA,
                Box::new(move || full.decode_tiled(latent, &tiling, None)),
            ));
            decodes.push((
                format!("{edge}² fp16 tile {tile}"),
                FP16_MAX_PIXEL_DELTA,
                FP16_MEAN_PIXEL_DELTA,
                Box::new(move || half.decode_tiled(latent, &t2, None)),
            ));
        }
        eprintln!("[sc-26268 parity] {edge}² f32 untiled reference: peak {ref_peak:.2} GiB");
        if ref_peak > PARITY_DECODE_PEAK_GIB {
            over_bound = true;
            eprintln!(
                "[sc-26268 parity] {edge}² reference decode exceeded the stated bound; stopping"
            );
            break 'edges;
        }
        for (label, max_bound, mean_bound, decode) in decodes {
            let (pixels, peak_gib) = bounded_decode(decode);
            let row = ParityRow {
                max: max_delta(&reference, &pixels),
                mean: mean_delta(&reference, &pixels),
                label,
                max_bound,
                mean_bound,
                peak_gib,
            };
            eprintln!(
                "[sc-26268 parity] {}: max {}/255 mean {:.4} (bounds {}/255, {:.2}) peak {:.2} GiB",
                row.label, row.max, row.mean, row.max_bound, row.mean_bound, row.peak_gib
            );
            let stop = row.peak_gib > PARITY_DECODE_PEAK_GIB;
            rows.push(row);
            if stop {
                over_bound = true;
                eprintln!("[sc-26268 parity] decode exceeded the stated bound; stopping");
                break 'edges;
            }
        }
    }

    assert!(
        !over_bound,
        "a decode's MLX peak exceeded the stated {PARITY_DECODE_PEAK_GIB} GiB bound (see the rows above)"
    );
    let failures: Vec<&ParityRow> = rows
        .iter()
        .filter(|r| r.max > r.max_bound || r.mean > r.mean_bound)
        .collect();
    assert!(
        failures.is_empty(),
        "decode parity out of bounds: {}",
        failures
            .iter()
            .map(|r| format!("{} max {}/255 mean {:.4}", r.label, r.max, r.mean))
            .collect::<Vec<_>>()
            .join("; ")
    );
    assert_eq!(
        rows.len(),
        expected,
        "every selected comparison must be measured"
    );
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
        // Byte-identity, not a tolerance: a sequential half projects every conditioning-only input
        // at the pair's batch (`SdxlForwardPlan::cfg_row`), and at SDXL's 512² q4 latent-path
        // shapes Metal's GEMMs are row-exact between `tokens` and `2·tokens` rows. Before that fix
        // this measured 255/255 (sdxl) and 112/255 (illustrious) — `[1, K]` vs `[2, K]` and 77- vs
        // 154-row kernels rounding differently, amplified by CFG × 7 and the ancestral noise.
        assert_eq!(max, 0, "{entry}: sequential CFG changed the image");
    }
}
