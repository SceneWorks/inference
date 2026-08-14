//! Correctness-only production-latent tiled-decode admission for Kolors (SC-18325).
//!
//! This target has no clock, peak-memory, allocator, process-footprint, or calibration-matrix API.
//! It emits only immutable latent/output identities and the precommitted max-RGB error statistic.

use std::path::PathBuf;

use mlx_gen::gen_core::Progress;
use mlx_gen::{PreviewSink, Quant};
use mlx_gen_kolors::memory_strategy as ms;
use mlx_gen_kolors::model::{KolorsHeavy, KolorsText};
use mlx_rs::Dtype::Float16;
use sha2::{Digest, Sha256};

const ROOT_ENV: &str = "KOLORS_QUALITY_ROOT";
const DEFAULT_TIER: &str = "q4";
const DEFAULT_STEPS: usize = 25;
const SEEDS: [u64; 5] = [1234, 7, 99, 20260805, 424242];
/// The correctness admission grid is deliberately the shipped common Kolors/SDXL bucket grid,
/// stricter than the U-Net's minimum structural multiple. Keeping both axes on `/64` prevents an
/// unreviewed odd-resolution coordinate from reaching an expensive real-weight run.
const QUALITY_GEOMETRY_MULTIPLE: u32 = 64;
const DECODER_PARAMETER_MULTIPLE: u32 = 8;

fn tier() -> String {
    let tier = std::env::var("DECODE_QUALITY_TIER").unwrap_or_else(|_| DEFAULT_TIER.to_owned());
    assert!(
        matches!(tier.as_str(), "bf16" | "q4" | "q8"),
        "unsupported Kolors quality tier {tier:?}"
    );
    tier
}

fn tier_dir(tier: &str) -> PathBuf {
    let root = PathBuf::from(
        std::env::var(ROOT_ENV)
            .unwrap_or_else(|_| panic!("SKIPPED-BY-ABSENCE: set {ROOT_ENV} to the snapshot root")),
    );
    let dir = root.join(tier);
    assert!(
        dir.is_dir(),
        "SKIPPED-BY-ABSENCE: {ROOT_ENV} must contain {tier}/"
    );
    dir
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn candidates() -> Vec<(u32, u32, u32, u32)> {
    let rows = std::env::var("DECODE_QUALITY_CANDIDATES")
        .unwrap_or_else(|_| format!("1024x1024:{}:{}", ms::DECODE_TILE_EDGE, ms::DECODE_OVERLAP));
    parse_candidates(&rows)
}

fn parse_candidates(rows: &str) -> Vec<(u32, u32, u32, u32)> {
    let candidates = rows
        .split_ascii_whitespace()
        .map(|row| {
            let fields = row.split(':').collect::<Vec<_>>();
            assert_eq!(fields.len(), 3, "invalid quality candidate {row:?}");
            let (width, height) = fields[0]
                .split_once('x')
                .unwrap_or_else(|| panic!("invalid quality geometry {:?}", fields[0]));
            let candidate = (
                width.parse::<u32>().expect("quality width"),
                height.parse::<u32>().expect("quality height"),
                fields[1].parse::<u32>().expect("quality tile edge"),
                fields[2].parse::<u32>().expect("quality overlap"),
            );
            let (width, height, tile_edge, overlap) = candidate;
            assert!(
                width > 0
                    && height > 0
                    && width.is_multiple_of(QUALITY_GEOMETRY_MULTIPLE)
                    && height.is_multiple_of(QUALITY_GEOMETRY_MULTIPLE),
                "Kolors quality geometry {width}x{height} must be positive and aligned to the shipped /{QUALITY_GEOMETRY_MULTIPLE} grid"
            );
            assert!(
                overlap > 0
                    && overlap < tile_edge
                    && tile_edge <= width.min(height)
                    && tile_edge.is_multiple_of(DECODER_PARAMETER_MULTIPLE)
                    && overlap.is_multiple_of(DECODER_PARAMETER_MULTIPLE),
                "Kolors quality tile {tile_edge}/{overlap} must satisfy 0 < overlap < tile <= min(width,height) and the decoder /{DECODER_PARAMETER_MULTIPLE} domain"
            );
            candidate
        })
        .collect::<Vec<_>>();
    assert!(
        !candidates.is_empty(),
        "Kolors quality candidate grid is empty"
    );
    candidates
}

#[test]
fn quality_candidate_domain_rejects_invalid_geometry_and_tile_axes() {
    assert_eq!(
        parse_candidates("768x768:576:48 1024x1024:768:64 1280x768:576:48 768x1280:576:48"),
        [
            (768, 768, 576, 48),
            (1024, 1024, 768, 64),
            (1280, 768, 576, 48),
            (768, 1280, 576, 48),
        ]
    );
    for invalid in [
        "1280x720:576:48",
        "720x1280:576:48",
        "0x768:576:48",
        "768x768:576:0",
        "768x768:576:576",
        "768x768:776:48",
        "768x768:578:48",
        "768x768:576:50",
        "",
    ] {
        assert!(
            std::panic::catch_unwind(|| parse_candidates(invalid)).is_err(),
            "invalid candidate domain unexpectedly passed: {invalid:?}"
        );
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn array_sha256(array: &mlx_rs::Array) -> String {
    let flat = array
        .as_dtype(mlx_rs::Dtype::Float32)
        .expect("quality latent f32")
        .flatten(None, None)
        .expect("quality latent flatten");
    flat.eval().expect("quality latent readback");
    let mut digest = Sha256::new();
    for dimension in array.shape() {
        digest.update(dimension.to_le_bytes());
    }
    for value in flat.as_slice::<f32>() {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn max_delta(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len(), "pixel buffers differ in length");
    a.iter()
        .zip(b)
        .map(|(left, right)| left.abs_diff(*right) as u32)
        .max()
        .unwrap_or_default()
}

#[test]
#[ignore = "needs the exact SceneWorks/kolors-mlx q4 snapshot and Metal"]
fn production_latent_quality_admission() {
    let revision = std::env::var("DECODE_QUALITY_SOURCE_REVISION")
        .expect("DECODE_QUALITY_SOURCE_REVISION must bind the exact snapshot");
    let repository = std::env::var("QUALITY_REPOSITORY")
        .expect("QUALITY_REPOSITORY must bind the exact snapshot repository");
    let tier = tier();
    let dir = tier_dir(&tier);
    let steps = env_u32("DECODE_QUALITY_STEPS", DEFAULT_STEPS as u32) as usize;
    let quant = match tier.as_str() {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        "bf16" => None,
        _ => unreachable!("validated quality tier"),
    };

    // P1 owns the process-default MLX stream on this thread. Prime that lifetime even though this
    // correctness-only target never enables retained compilation or records a performance datum.
    mlx_gen::nn::prepare_retained_compilation_thread();
    let mut text = KolorsText::load(&dir, Float16).expect("load ChatGLM3");
    if let Some(quant) = quant {
        text.quantize(quant.bits()).expect("quantize text encoder");
    }
    let pos = text
        .encode("a red fox in a snowy forest, photograph")
        .expect("positive conditioning");
    let neg = text
        .encode("blurry, lowres")
        .expect("negative conditioning");
    mlx_rs::transforms::eval([&pos.0, &pos.1, &neg.0, &neg.1]).expect("eval conditioning");
    drop(text);
    mlx_rs::memory::clear_cache();

    let mut heavy = KolorsHeavy::load(&dir, Float16).expect("load production heavy bundle");
    if let Some(quant) = quant {
        heavy.quantize_unet(quant.bits()).expect("quantize U-Net");
    }
    let cancel = mlx_gen::CancelFlag::default();

    for (width, height, tile_edge, overlap) in candidates() {
        let tiling = mlx_gen::tiling::TilingConfig::spatial_only(tile_edge as i32, overlap as i32);
        for seed in SEEDS {
            let latent_height = (height / 8) as i32;
            let latent_width = (width / 8) as i32;
            mlx_rs::random::seed(seed).expect("seed");
            let noise = mlx_rs::random::normal::<f32>(
                &[1, latent_height, latent_width, 4],
                None,
                None,
                None,
            )
            .expect("production noise");
            let latent = heavy
                .denoise_latents_with_preview(
                    &noise,
                    &pos,
                    Some(&neg),
                    steps,
                    5.0,
                    height as i32,
                    width as i32,
                    None,
                    &cancel,
                    &mut |_: Progress| {},
                    &PreviewSink::default(),
                    mlx_gen_sdxl::SdxlForwardPlan::UNBOUNDED,
                )
                .expect("production denoise");
            latent.eval().expect("eval production latent");
            let dense =
                mlx_gen_sdxl::decode_image(heavy.vae(), &latent, None).expect("dense decode");
            let tiled =
                mlx_gen_sdxl::decode_image_tiled(heavy.vae(), &latent, None, Some(&tiling), None)
                    .expect("tiled decode");
            println!(
                "DECODE_QUALITY_V2 {}",
                serde_json::json!({
                    "family": "kolors",
                    "resolvedRoute": "kolors",
                    "backend": "mlx",
                    "tier": tier.as_str(),
                    "loadShape": "eager_materialization",
                    "artifact": {
                        "repository": repository.as_str(),
                        "revision": revision.as_str(),
                        "variant": tier.as_str(),
                        "fingerprint": format!("{repository}@{revision}:{tier}"),
                    },
                    "implementationFingerprint": mlx_gen::gen_core::MEMORY_DECODE_QUALITY_IMPLEMENTATION_FINGERPRINT,
                    "mode": "text_to_image",
                    "overlay": null,
                    "geometry": { "width": width, "height": height, "batch": 1, "frames": 1, "referenceCount": 0 },
                    "usePid": false,
                    "tileEdge": tile_edge,
                    "overlap": overlap,
                    "metric": "max_abs_rgb_u8",
                    "maximumError": ms::DECODE_DRIFT_BAR,
                    "seed": seed,
                    "productionLatentProvenance": format!("kolors@{revision} {tier} production schedule={steps} guidance=5 prompt=p9-red-fox-v1"),
                    "productionLatentSha256": array_sha256(&latent),
                    "denseOutputSha256": sha256(&dense.pixels),
                    "tiledOutputSha256": sha256(&tiled.pixels),
                    "observedError": max_delta(&dense.pixels, &tiled.pixels),
                })
            );
        }
    }
}
