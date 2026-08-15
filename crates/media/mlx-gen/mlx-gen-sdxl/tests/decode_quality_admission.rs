//! Correctness-only production-latent tiled-decode admission for every SDXL catalog route.
//!
//! This target contains no clock, allocator, memory, process-footprint, or calibration API. It
//! captures the exact final latent from the ordinary `Sdxl::generate` denoise body and emits only
//! immutable semantic identities plus the precommitted max-RGB error statistic.

use std::io::Write as _;
use std::path::PathBuf;

use mlx_gen::{
    GenerationOutput, GenerationRequest, LoadShape, LoadSpec, OffloadPolicy, Quant, WeightsSource,
};
use sha2::{Digest, Sha256};

const ROOT_ENV: &str = "SDXL_QUALITY_ROOT";
const DEFAULT_TIER: &str = "q4";
const MAXIMUM_ERROR: u32 = 48;
const DEFAULT_TILE_EDGE: u32 = 896;
const DEFAULT_OVERLAP: u32 = 192;
const SEEDS: [u64; 5] = [1234, 7, 99, 20260805, 424242];

/// The source-closure stamp recorded on every receipt this harness emits.
///
/// Derived by `scripts/ci/decode_quality_implementation_fingerprint.py` in the CI step that runs
/// this test, so the stamp always names the tree that produced the measurement. It is recorded, not
/// matched: sc-19728 removed the comparison against a constant compiled into the binary, because a
/// closure spanning whole crates plus `Cargo.lock` invalidated every measurement on every
/// dependency bump and main sync. Required rather than defaulted — an unstamped receipt is a
/// receipt nobody can date.
fn implementation_fingerprint() -> String {
    let value = std::env::var("DECODE_QUALITY_IMPLEMENTATION_FINGERPRINT").expect(
        "DECODE_QUALITY_IMPLEMENTATION_FINGERPRINT must carry the derived source-closure stamp \
         (scripts/ci/decode_quality_implementation_fingerprint.py)",
    );
    assert!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "DECODE_QUALITY_IMPLEMENTATION_FINGERPRINT must be a lowercase SHA-256: {value:?}"
    );
    value
}

fn tier() -> String {
    let tier = std::env::var("DECODE_QUALITY_TIER").unwrap_or_else(|_| DEFAULT_TIER.to_owned());
    assert!(
        matches!(tier.as_str(), "bf16" | "q4" | "q8"),
        "unsupported SDXL quality tier {tier:?}"
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
    std::env::var("DECODE_QUALITY_CANDIDATES")
        .unwrap_or_else(|_| format!("1024x1024:{DEFAULT_TILE_EDGE}:{DEFAULT_OVERLAP}"))
        .split_ascii_whitespace()
        .map(|row| {
            let fields = row.split(':').collect::<Vec<_>>();
            assert_eq!(fields.len(), 3, "invalid quality candidate {row:?}");
            let (width, height) = fields[0]
                .split_once('x')
                .unwrap_or_else(|| panic!("invalid quality geometry {:?}", fields[0]));
            (
                width.parse().expect("quality width"),
                height.parse().expect("quality height"),
                fields[1].parse().expect("quality tile edge"),
                fields[2].parse().expect("quality overlap"),
            )
        })
        .collect()
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
#[ignore = "needs one exact SDXL-family q4 snapshot and Metal"]
fn production_latent_quality_admission() {
    let resolved_route = std::env::var("SDXL_QUALITY_ENTRY")
        .expect("SDXL_QUALITY_ENTRY must name the catalog route");
    assert!(
        matches!(
            resolved_route.as_str(),
            "sdxl"
                | "realvisxl"
                | "realvisxl_lightning"
                | "illustrious_xl_v1"
                | "illustrious_xl_v2"
        ),
        "unsupported SDXL quality route {resolved_route:?}"
    );
    let revision = std::env::var("DECODE_QUALITY_SOURCE_REVISION")
        .expect("DECODE_QUALITY_SOURCE_REVISION must bind the exact snapshot");
    let repository = std::env::var("QUALITY_REPOSITORY")
        .expect("QUALITY_REPOSITORY must bind the exact snapshot repository");
    let tier = tier();
    let steps = env_u32(
        "DECODE_QUALITY_STEPS",
        if resolved_route == "realvisxl_lightning" {
            5
        } else {
            30
        },
    );
    let guidance = std::env::var("DECODE_QUALITY_GUIDANCE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(if resolved_route == "realvisxl_lightning" {
            1.0
        } else {
            7.0
        });
    let sampler = std::env::var("DECODE_QUALITY_SAMPLER")
        .ok()
        .filter(|value| !value.is_empty());
    let mut spec = LoadSpec::new(WeightsSource::Dir(tier_dir(&tier)))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(LoadShape::DeferredMaterialization);
    spec.quantize = match tier.as_str() {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        "bf16" => None,
        _ => unreachable!("validated quality tier"),
    };
    let model = mlx_gen_sdxl::load_concrete(&spec).expect("load SDXL production route");
    for (width, height, tile_edge, overlap) in candidates() {
        let tiling = mlx_gen::tiling::TilingConfig::spatial_only(tile_edge as i32, overlap as i32);
        for seed in SEEDS {
            let request = GenerationRequest {
                prompt: "a red fox in a snowy forest, photograph".to_owned(),
                negative_prompt: Some("blurry, lowres".to_owned()),
                width,
                height,
                count: 1,
                seed: Some(seed),
                steps: Some(steps),
                guidance: Some(guidance),
                sampler: sampler.clone(),
                ..Default::default()
            };
            let (output, mut samples) = model
                .generate_decode_quality(&request, &tiling, &mut |_| {})
                .expect("production denoise and decode comparison");
            assert_eq!(
                samples.len(),
                1,
                "one request count must capture one latent"
            );
            let sample = samples.pop().unwrap();
            let production_pixels = match output {
                GenerationOutput::Images(mut images) => {
                    images.pop().expect("one production image").pixels
                }
                other => panic!("expected image output, got {other:?}"),
            };
            assert_eq!(
                production_pixels, sample.dense.pixels,
                "the capture seam must leave the ordinary dense production output byte-identical"
            );
            println!(
                "DECODE_QUALITY_V2 {}",
                serde_json::json!({
                    "family": "sdxl",
                    "resolvedRoute": resolved_route.as_str(),
                    "backend": "mlx",
                    "tier": tier.as_str(),
                    "loadShape": "deferred_materialization",
                    "artifact": {
                        "repository": repository.as_str(),
                        "revision": revision.as_str(),
                        "variant": tier.as_str(),
                        "fingerprint": format!("{repository}@{revision}:{tier}"),
                    },
                    "implementationFingerprint": implementation_fingerprint(),
                    "mode": "text_to_image",
                    "overlay": null,
                    "geometry": { "width": width, "height": height, "batch": 1, "frames": 1, "referenceCount": 0 },
                    "usePid": false,
                    "tileEdge": tile_edge,
                    "overlap": overlap,
                    "metric": "max_abs_rgb_u8",
                    "maximumError": MAXIMUM_ERROR,
                    "seed": seed,
                    "productionLatentProvenance": format!("{resolved_route}@{revision} {tier} production schedule={steps} guidance={guidance} sampler={} prompt=p9-red-fox-v1", sampler.as_deref().unwrap_or("euler_ancestral")),
                    "productionLatentSha256": array_sha256(&sample.production_latent),
                    "denseOutputSha256": sha256(&sample.dense.pixels),
                    "tiledOutputSha256": sha256(&sample.tiled.pixels),
                    "observedError": max_delta(&sample.dense.pixels, &sample.tiled.pixels),
                })
            );
            std::io::stdout()
                .flush()
                .expect("flush SDXL quality receipt");
            drop((production_pixels, sample));
            mlx_rs::memory::clear_cache();
        }
    }
}
