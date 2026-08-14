//! Correctness-only production-latent tiled-decode admission for Chroma (SC-18325).
//!
//! This target is intentionally isolated from `memory_ladder_real_weights`: it does not read a
//! clock, allocator counter, process footprint, or memory/calibration matrix. Its only output is an
//! immutable semantic receipt for one exact route/geometry/tile coordinate and multiple fixed seeds.

use std::io::Write as _;
use std::path::PathBuf;

use mlx_gen::{LoadSpec, WeightsSource};
use sha2::{Digest, Sha256};

const DEFAULT_TIER: &str = "q4";
const DEFAULT_TILE_EDGE: u32 = 832;
const DEFAULT_OVERLAP: u32 = 256;
const MAXIMUM_ERROR: u32 = 48;
const SEEDS: [u64; 5] = [1234, 7, 99, 20260805, 424242];

fn route() -> (&'static str, &'static str, mlx_gen_chroma::ChromaVariant) {
    match std::env::var("CHROMA_QUALITY_ENTRY").ok().as_deref() {
        Some("chroma1_hd") => (
            "chroma1_hd",
            "CHROMA_QUALITY_ROOT",
            mlx_gen_chroma::ChromaVariant::Hd,
        ),
        Some("chroma1_flash") => (
            "chroma1_flash",
            "CHROMA_QUALITY_ROOT",
            mlx_gen_chroma::ChromaVariant::Flash,
        ),
        Some("chroma1_base") | None => (
            "chroma1_base",
            "CHROMA_QUALITY_ROOT",
            mlx_gen_chroma::ChromaVariant::Base,
        ),
        Some(other) => panic!("CHROMA_QUALITY_ENTRY: unknown entry {other:?}"),
    }
}

fn tier() -> String {
    let tier = std::env::var("DECODE_QUALITY_TIER").unwrap_or_else(|_| DEFAULT_TIER.to_owned());
    assert!(
        matches!(tier.as_str(), "bf16" | "q4" | "q8"),
        "unsupported Chroma quality tier {tier:?}"
    );
    tier
}

fn tier_dir(root_env: &str, tier: &str) -> PathBuf {
    let root = PathBuf::from(
        std::env::var(root_env)
            .unwrap_or_else(|_| panic!("SKIPPED-BY-ABSENCE: set {root_env} to a snapshot root")),
    );
    let dir = root.join(tier);
    assert!(
        dir.is_dir(),
        "SKIPPED-BY-ABSENCE: {root_env} must contain {tier}/"
    );
    dir
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

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn array_sha256(array: &mlx_rs::Array) -> String {
    let flat = array
        .as_dtype(mlx_rs::Dtype::Float32)
        .expect("quality array f32")
        .flatten(None, None)
        .expect("quality array flatten");
    flat.eval().expect("quality array readback");
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
#[ignore = "needs one exact Chroma q4 snapshot and Metal"]
fn production_latent_quality_admission() {
    let (resolved_route, root_env, variant) = route();
    let tier = tier();
    let dir = tier_dir(root_env, &tier);
    let steps = env_u32("DECODE_QUALITY_STEPS", variant.default_steps());
    let guidance = std::env::var("DECODE_QUALITY_GUIDANCE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or_else(|| variant.default_true_cfg());
    let revision = std::env::var("DECODE_QUALITY_SOURCE_REVISION")
        .expect("DECODE_QUALITY_SOURCE_REVISION must bind the exact snapshot");
    let repository = std::env::var("QUALITY_REPOSITORY")
        .expect("QUALITY_REPOSITORY must bind the exact snapshot repository");
    // The correctness accessor deliberately drives the warm-resident production components. It is
    // not available on `Sequential`, whose components exist only inside the request closure.
    let spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
    let model = mlx_gen_chroma::load_chroma(variant, &spec).expect("load Chroma production route");
    let vae = mlx_gen_chroma::loader::load_vae(&dir).expect("load production VAE");
    for (width, height, tile_edge, overlap) in candidates() {
        let tiling = mlx_gen::tiling::TilingConfig::spatial_only(tile_edge as i32, overlap as i32);
        for seed in SEEDS {
            let noise = mlx_gen_flux::create_noise(seed, width, height).expect("production noise");
            let latent = model
                .denoise_with_sampler_name(
                    "a red fox in a snowy forest, photograph",
                    "blurry, lowres",
                    width,
                    height,
                    steps,
                    guidance,
                    noise,
                    None,
                    &mlx_gen::CancelFlag::default(),
                    &mut |_| {},
                )
                .expect("production denoise");
            latent.eval().expect("eval production latent");
            let unpacked =
                mlx_gen_flux::unpack_latents(&latent, width, height).expect("unpack latent");
            let dense = vae.decode(&unpacked).expect("dense decode");
            dense.eval().expect("eval dense decode");
            let tiled = vae
                .decode_tiled(&unpacked, &tiling, None)
                .expect("tiled decode");
            tiled.eval().expect("eval tiled decode");
            let dense_pixels =
                mlx_gen::image::decoded_to_image(&dense.as_dtype(mlx_rs::Dtype::Float32).unwrap())
                    .expect("dense image")
                    .pixels;
            let tiled_pixels =
                mlx_gen::image::decoded_to_image(&tiled.as_dtype(mlx_rs::Dtype::Float32).unwrap())
                    .expect("tiled image")
                    .pixels;
            println!(
                "DECODE_QUALITY_V2 {}",
                serde_json::json!({
                    "family": "chroma",
                    "resolvedRoute": resolved_route,
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
                    "maximumError": MAXIMUM_ERROR,
                    "seed": seed,
                    "productionLatentProvenance": format!("{resolved_route}@{revision} {tier} production schedule={steps} guidance={guidance} prompt=p9-red-fox-v1"),
                    "productionLatentSha256": array_sha256(&latent),
                    "denseOutputSha256": sha256(&dense_pixels),
                    "tiledOutputSha256": sha256(&tiled_pixels),
                    "observedError": max_delta(&dense_pixels, &tiled_pixels),
                })
            );
            std::io::stdout()
                .flush()
                .expect("flush Chroma quality receipt");
            drop((latent, unpacked, dense, tiled, dense_pixels, tiled_pixels));
            mlx_rs::memory::clear_cache();
        }
    }
}
