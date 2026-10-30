//! Krea Realtime 14B real-weight end-to-end validation (sc-8446, S13).
//!
//! Mirrors `mlx-gen-scail2/tests/generate_smoke.rs`: one CI-safe weights-free gate plus `#[ignore]`
//! real-weight drivers that go through the **public product entry point** —
//! `provider_registry().load("krea_realtime_14b", spec)` → `Generator::generate` — so the registry,
//! snapshot resolution, tier probe, UMT5 encode, AR few-step chunk loop, and z16 VAE decode are all
//! exercised as the worker drives them.
//!
//! Beyond "it runs", these capture the three S13 measurements:
//!   * **coherence** — per-frame dynamic range / plausibility and temporal change (a drifted or
//!     collapsed AR clip shows up as a frozen or blown-out tail, which the *last-third* checks catch);
//!   * **timing** — per-chunk and whole-clip wall time (the latency input for the realtime epic 8432);
//!   * **memory** — the MLX ceiling, sampled across the run, reported as **active** and
//!     **active + cache** separately (`get_peak_memory` is the ACTIVE high-water mark and does **not**
//!     include the buffer cache, so it under-reports what the OS has to have available), plus the
//!     measured KV-cache residency that sizes the SceneWorks manifest's `mlx.minMemoryGb`.
//!
//! ```text
//! KREA_REALTIME_SNAPSHOT_DIR=~/.cache/krea-realtime-mlx-snapshot/q4 \
//! KREA_SMOKE_OUT=/tmp/krea_smoke \
//!   cargo test -p mlx-gen-krea-realtime --test integration generate_smoke:: -- --ignored --nocapture
//! ```
//!
//! ⚠️ **`--ignored` is a blanket, and one of the tests it selects is not a smoke.**
//! [`long_clip_coherence_under_the_bounded_window`] is the sc-15127/sc-15571 research sweep: 85 min at
//! its default single seed, and the seeds it needs to return a verdict at all put it over four hours.
//! The real-weight lane therefore `--skip`s it and dispatches it as its own job (sc-17276); add
//! `--skip long_clip_coherence_under_the_bounded_window` when you want the ~20-minute smoke set.
//!
//! Env: `KREA_REALTIME_SNAPSHOT_DIR` (snapshot root; default `~/.cache/krea-realtime-mlx-convert`),
//! `KREA_SMOKE_W`/`_H` (default 832×480, the reference bucket), `KREA_SMOKE_FRAMES` (default 81),
//! `KREA_SMOKE_STEPS` (default: the config's Self-Forcing schedule), `KREA_SMOKE_SEED`,
//! `KREA_SMOKE_OUT` (frame dump dir).

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use sha2::{Digest, Sha256};

use mlx_gen::testing::env_path_or_home;
use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, WeightsSource,
};
use mlx_gen_krea_realtime::{
    decode_latents_to_video, decode_tiling, FewStepSchedule, KreaRealtimeConfig, KvCacheQuant,
    MODEL_ID,
};

// sc-27342: the shared fixtures, request builders, the S18 drift metric / sweep / verdict and its
// weights-free gates, and the recorded S18 evidence live in `generate_smoke/`. This file keeps the
// `#[ignore]`d real-weight entry points, whose `generate_smoke::<name>` paths are what the proof
// registry, the real-weight lanes and the campaign launchers address, and the whole SC-20684
// observer section, which the SC-20684 contract checker and the observer's sealed source identity
// pin to this file. This file is `#[path]`-loaded by `tests/main.rs`, so its submodules name
// their files explicitly.
#[path = "generate_smoke/drift.rs"]
mod drift;
#[path = "generate_smoke/measured.rs"]
mod measured;
#[path = "generate_smoke/recorded.rs"]
mod recorded;
#[path = "generate_smoke/requests.rs"]
mod requests;
#[path = "generate_smoke/s18.rs"]
mod s18;
#[path = "generate_smoke/s18_gates.rs"]
mod s18_gates;
#[path = "generate_smoke/s18_verdict.rs"]
mod s18_verdict;
#[path = "generate_smoke/support.rs"]
mod support;
use drift::*;
use measured::*;
use recorded::*;
use requests::*;
use s18::*;
use support::*;

/// The converted MLX snapshot root, holding `dit.safetensors`, `t5_encoder.safetensors`,
/// `vae.safetensors` and `tokenizer.json` — one of the published `SceneWorks/krea-realtime-14b-mlx`
/// tiers. Caller-provided like every other component path in this workspace; the default mirrors the
/// sibling SCAIL-2 smoke.
///
/// Kept in this file (not `generate_smoke/support.rs`): the workspace gate's `$HOME`-derivation
/// lint scopes its env-override search to a plain or `pub` `fn`.
fn snapshot_dir() -> PathBuf {
    env_path_or_home(
        "KREA_REALTIME_SNAPSHOT_DIR",
        ".cache/krea-realtime-mlx-convert",
    )
}

/// This suite's own source — this file and every `generate_smoke/` submodule — for the prose gates
/// that read the suite's documentation (they read the single `generate_smoke.rs` before the split).
/// A new submodule must be added here too.
const SUITE_SOURCE: &str = concat!(
    include_str!("generate_smoke.rs"),
    include_str!("generate_smoke/drift.rs"),
    include_str!("generate_smoke/measured.rs"),
    include_str!("generate_smoke/recorded.rs"),
    include_str!("generate_smoke/requests.rs"),
    include_str!("generate_smoke/s18.rs"),
    include_str!("generate_smoke/s18_gates.rs"),
    include_str!("generate_smoke/s18_verdict.rs"),
    include_str!("generate_smoke/support.rs"),
);

// ---------------------------------------------------------------------------------------------
// SC-20684 product-owned packed-Metal campaign observer
// ---------------------------------------------------------------------------------------------

/// The parent Python reducer supplies only a run nonce and a requested matrix coordinate.  This
/// ignored real-weight test owns all identity and measurement fields below, and emits one JSON line
/// only after the packed route, dense parity route, media decode, and cancellation probe are terminal.
/// It is deliberately not a normal smoke test: the coordinator invokes it in fresh paired and
/// dense-baseline processes for every T2V/I2V/V2V × Q8/Q4 cell on the held Metal lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sc20684Mode {
    T2v,
    I2v,
    V2v,
}

const SC20684_V2V_STRENGTH: f32 = 0.6;

/// An `f32` request knob as the decimal the request declared, for the launcher's identity check.
/// `serde_json` widens an `f32` to `f64` exactly, so `0.6f32` would serialize as
/// `0.6000000238418579` and fail the launcher's `v2vStrength == 0.6` (W2 run 36854186685). The
/// shortest decimal that round-trips the `f32` is the requested value; the generation itself still
/// consumes the `f32`.
fn sc20684_identity_decimal(value: f32) -> f64 {
    value
        .to_string()
        .parse()
        .expect("an f32's shortest decimal is a valid f64")
}

/// The V2V conditioning's input identity, exactly as the launcher validates it.
fn sc20684_v2v_input_identity(
    source_sha256: &str,
    frames: usize,
    width: usize,
    height: usize,
    strength: f32,
) -> serde_json::Value {
    serde_json::json!({
        "kind": "deterministic-smooth-motion-clip", "sha256": source_sha256,
        "frameCount": frames, "width": width, "height": height,
        "vaeEncoding": "WanVae.encode-sample",
        "v2vStrength": sc20684_identity_decimal(strength),
    })
}

#[test]
fn sc20684_v2v_strength_identity_is_the_requested_decimal() {
    let emitted = serde_json::to_string(&sc20684_v2v_input_identity(
        &"0".repeat(64),
        25,
        832,
        480,
        SC20684_V2V_STRENGTH,
    ))
    .unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&emitted).unwrap();
    assert_eq!(
        parsed["v2vStrength"].as_f64(),
        Some(0.6),
        "the launcher expects the requested V2V strength, 0.6: {emitted}"
    );
    assert!(emitted.contains(r#""v2vStrength":0.6}"#) || emitted.contains(r#""v2vStrength":0.6,"#));
    let identity = |strength: serde_json::Value| {
        serde_json::to_string(&serde_json::json!({ "v2vStrength": strength })).unwrap()
    };
    // The defect this guards: the raw f32 widens on serialization.
    assert_ne!(
        identity(SC20684_V2V_STRENGTH.into()),
        r#"{"v2vStrength":0.6}"#
    );
    // Still the same f32 the generation consumes.
    assert_eq!(
        sc20684_identity_decimal(SC20684_V2V_STRENGTH) as f32,
        SC20684_V2V_STRENGTH
    );
}
const SC20684_Q8_PARITY_MAX_ABS_ERROR: f32 = 0.25;
const SC20684_Q4_PARITY_MAX_ABS_ERROR: f32 = 0.75;
const SC20684_Q8_MAX_ABS_RGB_U8: u8 = 32;
const SC20684_Q4_MAX_ABS_RGB_U8: u8 = 96;
const SC20684_Q8_MEAN_ABS_RGB_U8: f64 = 1.0;
const SC20684_Q4_MEAN_ABS_RGB_U8: f64 = 3.0;
const SC20684_Q8_TEMPORAL_DELTA_DRIFT: f64 = 4.0;
const SC20684_Q4_TEMPORAL_DELTA_DRIFT: f64 = 12.0;

// This is the measured z16 spatial candidate from the real-latent sweep below. The
// 28-frame decode fits in one temporal tile; the spatial upsample tail is bounded.
// Keep this exact plan common to packed, paired dense, and fresh dense baseline.
fn sc20684_decode_plan() -> mlx_gen::tiling::TilingConfig {
    mlx_gen::tiling::TilingConfig {
        spatial: Some(mlx_gen::tiling::SpatialTiling {
            tile_px: 256,
            overlap_px: 64,
        }),
        temporal: Some(mlx_gen::tiling::TemporalTiling {
            tile_frames: 32,
            overlap_frames: 16,
        }),
    }
}

fn sc20684_decode_plan_identity(raw_decoded_frames: usize) -> serde_json::Value {
    let plan = sc20684_decode_plan();
    let spatial = plan.spatial.expect("SC-20684 spatial decode tile");
    let temporal = plan.temporal.expect("SC-20684 temporal decode tile");
    serde_json::json!({
        "kind": "wan-z16-spatial-tail-v1",
        "spatialTilePx": spatial.tile_px,
        "spatialOverlapPx": spatial.overlap_px,
        "temporalTileFrames": temporal.tile_frames,
        "temporalOverlapFrames": temporal.overlap_frames,
        "rawDecodedFrames": raw_decoded_frames,
    })
}

#[test]
fn sc20684_decode_plan_prices_the_full_raw_frame_count() {
    let plan = sc20684_decode_plan();
    let identity = sc20684_decode_plan_identity(28);
    assert_eq!(plan.spatial.as_ref().expect("spatial bound").tile_px, 256);
    assert_eq!(
        plan.temporal.as_ref().expect("temporal bound").tile_frames,
        32
    );
    assert_eq!(identity["rawDecodedFrames"], 28);
    let estimate = mlx_gen_wan::pipeline::video_decode_peak_bytes_for_vae_and_tiling(
        mlx_gen_wan::WanVae::VAE_TILING,
        832,
        480,
        28,
        Some(&sc20684_decode_plan()),
    );
    assert_eq!(estimate, Some(12_643_205_120));
}

/// Every recorded allocator-sampler gap, with the phase and release windows it spanned.
fn sc20684_sampler_gaps(
    allocator: &mlx_gen::memory_probe::AllocatorProbeReport,
) -> Vec<serde_json::Value> {
    allocator
        .gaps
        .iter()
        .map(|gap| {
            serde_json::json!({
                "startMicros": gap.start_micros,
                "durationMicros": gap.duration_micros,
                "uncoveredMicros": gap.uncovered_micros,
                "releaseMicros": gap.release_micros,
                "releaseWindows": gap.release_windows,
                "phaseAtStart": gap.phase_at_start,
                "phaseAtEnd": gap.phase_at_end,
            })
        })
        .collect()
}

/// One phase-boundary memory reading. It is also printed as it is taken, so a role the launcher's
/// footprint watchdog aborts mid-run still leaves which phases completed, and with what MLX active
/// versus cached bytes, in its bounded stdout transcript.
fn sc20684_phase_memory(name: &str) -> serde_json::Value {
    // The allocator probe attributes each sampler gap to the last phase reached.
    mlx_gen::memory_probe::set_phase(name);
    let value = serde_json::json!({
        "phase": name,
        "process": sc20684_process_memory(),
        "mlxActiveBytes": mlx_rs::memory::get_active_memory() as u64,
        "mlxCacheBytes": mlx_rs::memory::get_cache_memory() as u64,
    });
    println!("SC20684_KREA_PHASE {value}");
    value
}

fn sc20684_source_budget(
    config: &KreaRealtimeConfig,
    mode: Sc20684Mode,
    root: &Path,
    width: usize,
    height: usize,
    frames: usize,
) -> serde_json::Value {
    let frame_tokens = config.ar.frame_seq_length;
    let frame_schedule: &[usize] = if mode == Sc20684Mode::I2v {
        &[1, 3, 3]
    } else {
        &[3, 3, 1]
    };
    let packed_bytes_per_token = config.kv_bytes_per_token().expect("valid SC-20684 KV tier");
    let dense_bytes_per_token = 2 * config.wan.num_layers * config.wan.dim * 2;
    let mut previous_tokens = 0usize;
    let mut append_coexistence = 0usize;
    for &chunk_frames in frame_schedule {
        let new_tokens = frame_tokens * chunk_frames;
        let staged_tokens = previous_tokens + new_tokens;
        append_coexistence = append_coexistence.max(
            (previous_tokens + staged_tokens) * packed_bytes_per_token
                + new_tokens * dense_bytes_per_token,
        );
        previous_tokens = staged_tokens;
    }
    let logical_model_bytes: u64 = [
        "dit.safetensors",
        "t5_encoder.safetensors",
        "vae.safetensors",
    ]
    .iter()
    .map(|name| {
        std::fs::metadata(root.join(name))
            .expect("SC-20684 model metadata")
            .len()
    })
    .sum();
    let raw_decoded_frames = 4 * ((frames - 1) / 4 + 1);
    let decode_plan = sc20684_decode_plan();
    let decode_working_set_estimate =
        mlx_gen_wan::pipeline::video_decode_peak_bytes_for_vae_and_tiling(
            mlx_gen_wan::WanVae::VAE_TILING,
            width as u32,
            height as u32,
            raw_decoded_frames as u32,
            Some(&decode_plan),
        )
        .expect("SC-20684 fixed z16 decode estimate");
    serde_json::json!({
        "modelLogicalBytes": logical_model_bytes,
        "terminalKvBytes": previous_tokens * packed_bytes_per_token,
        "appendOldStagedAndNewDenseBytes": append_coexistence,
        "decodeWorkingSetEstimateBytes": decode_working_set_estimate,
        "runtimeOverheadBytes": null,
        "wholeProcessPeakBoundBytes": null,
        "decodePlan": sc20684_decode_plan_identity(raw_decoded_frames),
    })
}

impl Sc20684Mode {
    fn parse() -> Self {
        match std::env::var("KREA_SC20684_REQUEST_MODE").as_deref() {
            Ok("t2v") => Self::T2v,
            Ok("i2v") => Self::I2v,
            Ok("v2v") => Self::V2v,
            value => {
                panic!("SC-20684 requires KREA_SC20684_REQUEST_MODE=t2v|i2v|v2v, got {value:?}")
            }
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::T2v => "t2v",
            Self::I2v => "i2v",
            Self::V2v => "v2v",
        }
    }
}

fn sc20684_cache_tier() -> KvCacheQuant {
    match std::env::var("KREA_SC20684_CACHE_TIER").as_deref() {
        Ok("q8") => KvCacheQuant::Q8,
        Ok("q4")
            if std::env::var("KREA_SC20684_Q4_QUALITY_ARM").as_deref() == Ok("acknowledged") =>
        {
            KvCacheQuant::Q4
        }
        Ok("q4") => panic!("SC-20684 Q4 requires KREA_SC20684_Q4_QUALITY_ARM=acknowledged"),
        value => panic!("SC-20684 requires KREA_SC20684_CACHE_TIER=q8|q4, got {value:?}"),
    }
}

fn sc20684_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sc20684_sha256_file(path: &Path) -> (u64, String) {
    let mut file = std::fs::File::open(path).expect("open pinned model artifact for hashing");
    let size = file
        .metadata()
        .expect("read pinned model artifact metadata")
        .len();
    assert!(size > 0, "SC-20684 pinned model artifact must be nonempty");
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).expect("hash pinned model artifact");
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    (size, format!("{:x}", digest.finalize()))
}

fn sc20684_repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(4)
        .expect("crate must be nested below the inference repository root")
        .to_path_buf()
}

fn sc20684_command(program: &str, args: &[&str]) -> String {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("SC-20684 cannot execute {program}: {error}"));
    assert!(
        output.status.success(),
        "SC-20684 {program} {:?} failed",
        args
    );
    String::from_utf8(output.stdout)
        .expect("SC-20684 command output must be UTF-8")
        .trim()
        .to_owned()
}

fn sc20684_metal_device() -> String {
    let displays = sc20684_command("/usr/sbin/system_profiler", &["SPDisplaysDataType"]);
    displays
        .lines()
        .find_map(|line| line.trim().strip_prefix("Chipset Model:"))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .expect("SC-20684 system profile must identify the Metal chipset")
        .to_owned()
}

fn sc20684_source_identity(root: &Path) -> serde_json::Value {
    // The launcher freezes provenance before the first arm. Verify its captured
    // behavior bytes and executable before stamping a later resumed observation.
    let identity_path = std::env::var("KREA_SC20684_IDENTITY_PATH")
        .expect("SC-20684 launcher must supply its sealed resume identity");
    let identity: serde_json::Value = serde_json::from_slice(
        &std::fs::read(identity_path).expect("read SC-20684 resume identity"),
    )
    .expect("parse SC-20684 resume identity");
    let executable = std::env::current_exe().expect("resolve SC-20684 observer executable");
    let executable_sha =
        sc20684_sha256(&std::fs::read(executable).expect("hash observer executable"));
    assert_eq!(
        identity["executableSha256"].as_str(),
        Some(executable_sha.as_str()),
        "SC-20684 observer executable differs from captured resume identity"
    );
    let files = [
        "crates/media/mlx-gen/mlx-gen-krea-realtime/src/causal.rs",
        "crates/media/mlx-gen/mlx-gen-krea-realtime/src/compressed_kv.rs",
        "crates/media/mlx-gen/mlx-gen-krea-realtime/src/generate.rs",
        "crates/media/mlx-gen/mlx-gen-krea-realtime/src/t2v.rs",
        "crates/media/mlx-gen/mlx-gen-krea-realtime/src/pipeline.rs",
        "crates/media/mlx-gen/mlx-gen-krea-realtime/tests/generate_smoke.rs",
        "scripts/sc20684_krea_realtime_campaign.py",
    ];
    let mut hashes = serde_json::Map::new();
    for relative in files {
        hashes.insert(
            relative.to_owned(),
            serde_json::Value::String(sc20684_sha256(
                &std::fs::read(root.join(relative)).expect("read product source identity"),
            )),
        );
    }
    let source = identity["source"].clone();
    let head = source["repositoryHead"]
        .as_str()
        .expect("SC-20684 captured source revision");
    assert!(head.len() == 40 && head.bytes().all(|byte| byte.is_ascii_hexdigit()));
    assert_eq!(source["files"], serde_json::Value::Object(hashes));
    source
}

fn sc20684_model_identity(snapshot: &Path) -> serde_json::Value {
    assert_eq!(
        snapshot.file_name().and_then(|value| value.to_str()),
        Some("q4"),
        "SC-20684 snapshot must be the q4 variant"
    );
    assert_eq!(
        snapshot
            .parent()
            .and_then(Path::file_name)
            .and_then(|value| value.to_str()),
        Some("e68e9a3d98187fdf6936838ffcf6df5aa48d6626"),
        "SC-20684 snapshot must be nested below the exact pinned revision"
    );
    let mut files = serde_json::Map::new();
    let mut inventory = Sha256::new();
    for name in [
        "config.json",
        "dit.safetensors",
        "t5_encoder.safetensors",
        "vae.safetensors",
        "tokenizer.json",
    ] {
        let (size, digest) = sc20684_sha256_file(&snapshot.join(name));
        inventory.update(name.as_bytes());
        inventory.update([0]);
        inventory.update(size.to_string().as_bytes());
        inventory.update([0]);
        inventory.update(digest.as_bytes());
        inventory.update(b"\n");
        files.insert(
            name.to_owned(),
            serde_json::json!({"size": size, "sha256": digest}),
        );
    }
    let config_sha256 = files["config.json"]["sha256"]
        .as_str()
        .expect("config identity sha256")
        .to_owned();
    serde_json::json!({
        "repository": "SceneWorks/krea-realtime-14b-mlx",
        "revision": "e68e9a3d98187fdf6936838ffcf6df5aa48d6626",
        "variant": "q4",
        "configSha256": config_sha256,
        "files": files,
        "inventorySha256": format!("{:x}", inventory.finalize()),
    })
}

fn sc20684_route_lifecycle_snapshot(
    receipt: &mlx_gen_krea_realtime::PackedMetalRouteReceipt,
) -> serde_json::Value {
    serde_json::json!({
        "compiledHandleIdentity": receipt.compiled_handle_identity,
        "retainedHandleBytes": receipt.retained_handle_bytes,
        "acceptedForwards": receipt.accepted_forwards,
        "materializedScratchDispatches": receipt.materialized_scratch_dispatches,
        "boundedScratchBytes": receipt.bounded_scratch_bytes,
        "denseWindowBytes": receipt.dense_window_bytes,
        "scoreMatrixBytes": receipt.score_matrix_bytes,
        "dispatchGeometries": receipt.dispatch_geometries.iter().map(|geometry| serde_json::json!({
            "queryTokens": geometry.query_tokens,
            "keyTokens": geometry.key_tokens,
            "acceptedForwards": geometry.accepted_forwards,
        })).collect::<Vec<_>>(),
    })
}

fn sc20684_cancellation_dispatch_token(
    run_id: &str,
    receipt: &mlx_gen_krea_realtime::PackedMetalRouteReceipt,
) -> String {
    let dispatches = receipt
        .dispatch_geometries
        .iter()
        .map(|geometry| {
            format!(
                "{}:{}:{}",
                geometry.query_tokens, geometry.key_tokens, geometry.accepted_forwards
            )
        })
        .collect::<Vec<_>>()
        .join(";");
    sc20684_sha256(
        format!(
            "{run_id}\0{}\0{}\0{}\0{dispatches}",
            receipt.compiled_handle_identity,
            receipt.accepted_forwards,
            receipt.materialized_scratch_dispatches,
        )
        .as_bytes(),
    )
}

fn sc20684_hash_video(output: &GenerationOutput) -> (String, Vec<Image>) {
    let GenerationOutput::Video { frames, .. } = output else {
        panic!("SC-20684 expected a video output")
    };
    let mut hash = Sha256::new();
    for frame in frames {
        hash.update(frame.width.to_le_bytes());
        hash.update(frame.height.to_le_bytes());
        hash.update(&frame.pixels);
    }
    (format!("{:x}", hash.finalize()), frames.clone())
}

fn sc20684_max_rgb_error(left: &[Image], right: &[Image]) -> u8 {
    assert_eq!(left.len(), right.len(), "SC-20684 paired media frame count");
    left.iter()
        .zip(right)
        .flat_map(|(a, b)| {
            assert_eq!(
                (a.width, a.height),
                (b.width, b.height),
                "SC-20684 paired media geometry"
            );
            assert_eq!(
                a.pixels.len(),
                b.pixels.len(),
                "SC-20684 paired media payload length"
            );
            a.pixels.iter().zip(&b.pixels).map(|(x, y)| x.abs_diff(*y))
        })
        .max()
        .unwrap_or(0)
}

fn sc20684_hash_media(frames: &[Image]) -> String {
    let mut hash = Sha256::new();
    for frame in frames {
        hash.update(frame.width.to_le_bytes());
        hash.update(frame.height.to_le_bytes());
        hash.update(&frame.pixels);
    }
    format!("{:x}", hash.finalize())
}

fn sc20684_mean_temporal_delta(frames: &[Image]) -> f64 {
    let deltas: Vec<f64> = frame_stats(frames)
        .into_iter()
        .filter_map(|stat| stat.delta.is_finite().then_some(stat.delta))
        .collect();
    assert!(
        !deltas.is_empty(),
        "SC-20684 temporal metric requires multiple frames"
    );
    deltas.iter().sum::<f64>() / deltas.len() as f64
}

fn sc20684_process_memory() -> serde_json::Value {
    #[cfg(target_os = "macos")]
    {
        fn parse(value: &str) -> Option<u64> {
            let mut parts = value.split_whitespace();
            let number: f64 = parts.next()?.parse().ok()?;
            let multiplier = match parts.next().unwrap_or("B").to_ascii_uppercase().as_str() {
                "B" => 1.0,
                "KB" => 1024.0,
                "MB" => 1024.0 * 1024.0,
                "GB" => 1024.0 * 1024.0 * 1024.0,
                _ => return None,
            };
            let bytes = (number * multiplier).round();
            (number.is_finite() && number >= 0.0 && bytes.is_finite() && bytes <= u64::MAX as f64)
                .then_some(bytes as u64)
        }

        let pid = std::process::id().to_string();
        let output = std::process::Command::new("/usr/bin/footprint")
            .args(["--pid", &pid, "--noCategories", "--wired"])
            .output()
            .expect("SC-20684 execute /usr/bin/footprint");
        assert!(output.status.success(), "SC-20684 footprint command failed");
        let text = String::from_utf8(output.stdout).expect("SC-20684 footprint output UTF-8");
        let field = |name: &str| {
            text.lines()
                .find_map(|line| line.trim().strip_prefix(name))
                .and_then(parse)
                .unwrap_or_else(|| panic!("SC-20684 footprint output missing {name}"))
        };
        let current = field("phys_footprint:");
        let peak = field("phys_footprint_peak:");
        assert!(
            peak >= current,
            "SC-20684 process peak must cover current footprint"
        );
        serde_json::json!({"physFootprintBytes": current, "physFootprintPeakBytes": peak})
    }
    #[cfg(not(target_os = "macos"))]
    {
        panic!("SC-20684 real-weight campaign requires macOS Darwin phys_footprint")
    }
}

/// MLX active bytes the loaded Krea 14B legitimately gains on its FIRST forward and then keeps for the
/// model's lifetime, so the release boundary may exceed the weights-loaded boundary by this much.
///
/// mlx-gen-wan keeps every block's `modulation` `[1,6,5120]` and `norm3` weight/bias `[5120]`, and
/// `head.modulation` `[1,2,5120]`, as lazy f32 casts of the bf16 checkpoint tensors (`Block::load`,
/// `WanTransformer::from_weights_without_blocks`); `Weights::materialize_accessed` evaluates the bf16
/// sources at load but not the casts. The first forward allocates the f32 tables (16 KiB-page-rounded:
/// 131072 + 2×32768 per block, 49152 for the head) and frees their bf16 sources (65536 + 2×10240 per
/// block, 32768 for the head): [`SC20684_LAZY_DENSE_CAST_BYTES`] = 4,440,064 bytes. It is one-time model
/// state, not a per-request leak — the same pipeline on the tiny random-weight config gains exactly its
/// own lazy-cast sum on request 1 and nothing on requests 2 and 3. The SC-20684 campaign recorded
/// +4,456,444 bytes in BOTH the dense-baseline and paired roles; the remaining 16,380 bytes (under one
/// Metal page) are first-use MLX allocations, bounded here by one 16 KiB page. Mirrors
/// `RELEASE_ACTIVE_RESIDUAL_BYTES` in `scripts/sc20684_krea_realtime_campaign.py`.
const SC20684_LAZY_DENSE_CAST_BYTES: u64 =
    40 * (131_072 + 2 * 32_768 - 65_536 - 2 * 10_240) + (49_152 - 32_768);
const SC20684_RELEASE_ACTIVE_RESIDUAL_BYTES: u64 = 4_456_448;
/// Darwin phys-footprint slack between the release boundary and the loaded/terminal boundaries.
const SC20684_RELEASE_FOOTPRINT_SLACK_BYTES: u64 = 512 * 1024 * 1024;

/// The release rule shared by the dense-baseline and paired roles (and re-derived independently by
/// the campaign validator): MLX active memory returns to the loaded model plus its documented one-time
/// residual and to the terminal boundary, the allocator cache is empty of anything the run added, and
/// the settled Darwin footprint is within slack of both boundaries.
#[allow(clippy::too_many_arguments)]
fn sc20684_release_verified(
    release_active: u64,
    weights_loaded_active: u64,
    terminal_active: u64,
    release_cache: u64,
    weights_loaded_cache: u64,
    terminal_cache: u64,
    release_footprint: u64,
    weights_loaded_footprint: u64,
    terminal_footprint: u64,
) -> bool {
    release_active <= weights_loaded_active.saturating_add(SC20684_RELEASE_ACTIVE_RESIDUAL_BYTES)
        && release_active <= terminal_active
        && release_cache <= weights_loaded_cache
        && release_cache <= terminal_cache
        && release_footprint
            <= weights_loaded_footprint.saturating_add(SC20684_RELEASE_FOOTPRINT_SLACK_BYTES)
        && release_footprint
            <= terminal_footprint.saturating_add(SC20684_RELEASE_FOOTPRINT_SLACK_BYTES)
}

#[test]
fn sc20684_release_rule_admits_the_recorded_lazy_cast_residual_and_nothing_more() {
    assert_eq!(SC20684_LAZY_DENSE_CAST_BYTES, 4_440_064);
    assert_eq!(
        SC20684_RELEASE_ACTIVE_RESIDUAL_BYTES,
        SC20684_LAZY_DENSE_CAST_BYTES + 16_384
    );
    // Recorded SC-20684 campaign numbers: identical in the dense-baseline and paired roles.
    let (loaded, release) = (8_887_095_376u64, 8_891_551_820u64);
    let terminal = release + 4 * 1024 * 1024 * 1024;
    let footprint = 20 * 1024 * 1024 * 1024u64;
    let verified = |release_active: u64, release_footprint: u64| {
        sc20684_release_verified(
            release_active,
            loaded,
            terminal,
            0,
            0,
            1024,
            release_footprint,
            footprint,
            footprint + 1024 * 1024 * 1024,
        )
    };
    assert!(verified(release, footprint));
    assert!(verified(
        loaded + SC20684_RELEASE_ACTIVE_RESIDUAL_BYTES,
        footprint
    ));
    // One byte past the documented residual is a leak, and so is anything past the footprint slack.
    assert!(!verified(
        loaded + SC20684_RELEASE_ACTIVE_RESIDUAL_BYTES + 1,
        footprint
    ));
    assert!(!verified(
        release,
        footprint + SC20684_RELEASE_FOOTPRINT_SLACK_BYTES + 1
    ));
}

/// Poll interval, stable-read count, read bound, and unchanged tolerance of a settled footprint read —
/// the same policy as mlx-llm's `campaign_supervisor::settled_phys_footprint`.
const SC20684_FOOTPRINT_SETTLE_POLL: std::time::Duration = std::time::Duration::from_millis(10);
const SC20684_FOOTPRINT_SETTLE_STABLE_READS: u32 = 25;
const SC20684_FOOTPRINT_SETTLE_MAX_READS: u32 = 300;
const SC20684_FOOTPRINT_SETTLE_TOLERANCE_BYTES: u64 = 1024 * 1024;

/// Darwin returns Metal buffers MLX just freed to the process footprint asynchronously: a read taken
/// right after `clear_cache` reports the pre-free footprint for tens of milliseconds and then falls in
/// steps. Poll until `physFootprintBytes` is unchanged (within tolerance) for `stable_reads`
/// consecutive reads, bounded by `max_reads` (then the latest read is returned).
fn sc20684_settle_process_memory(
    mut read: impl FnMut() -> serde_json::Value,
    mut wait: impl FnMut(),
    stable_reads: u32,
    max_reads: u32,
) -> serde_json::Value {
    let current = |value: &serde_json::Value| {
        value["physFootprintBytes"]
            .as_u64()
            .expect("SC-20684 footprint read carries physFootprintBytes")
    };
    let mut latest = read();
    let mut anchor = current(&latest);
    let mut stable = 0;
    for _ in 1..max_reads {
        wait();
        latest = read();
        if current(&latest).abs_diff(anchor) <= SC20684_FOOTPRINT_SETTLE_TOLERANCE_BYTES {
            stable += 1;
            if stable >= stable_reads {
                break;
            }
        } else {
            anchor = current(&latest);
            stable = 0;
        }
    }
    latest
}

/// [`sc20684_process_memory`] once the footprint has stopped changing; every boundary the release rule
/// compares is read this way so a lagging pre-free footprint is never trusted.
fn sc20684_settled_process_memory() -> serde_json::Value {
    sc20684_settle_process_memory(
        sc20684_process_memory,
        || std::thread::sleep(SC20684_FOOTPRINT_SETTLE_POLL),
        SC20684_FOOTPRINT_SETTLE_STABLE_READS,
        SC20684_FOOTPRINT_SETTLE_MAX_READS,
    )
}

#[test]
fn sc20684_settled_footprint_waits_out_an_asynchronous_release() {
    let gib = 1024 * 1024 * 1024u64;
    let row = |current: u64| serde_json::json!({"physFootprintBytes": current, "physFootprintPeakBytes": 30 * gib});
    // Pre-free value held for three reads, then two falling steps, then flat.
    let trace = [20 * gib, 20 * gib, 20 * gib, 15 * gib, 12 * gib];
    let mut reads = 0usize;
    let settled = sc20684_settle_process_memory(
        || {
            reads += 1;
            row(trace.get(reads - 1).copied().unwrap_or(12 * gib))
        },
        || {},
        4,
        100,
    );
    assert_eq!(settled["physFootprintBytes"].as_u64(), Some(12 * gib));
    // Settles on the last step plus four unchanged reads, never on the lagging pre-free plateau.
    assert_eq!(reads, trace.len() + 4);

    // A footprint that never stops moving is bounded: the latest read is returned.
    let mut reads = 0u64;
    let unsettled = sc20684_settle_process_memory(
        || {
            reads += 1;
            row(reads * 2 * 1024 * 1024)
        },
        || {},
        4,
        10,
    );
    assert_eq!(reads, 10);
    assert_eq!(
        unsettled["physFootprintBytes"].as_u64(),
        Some(20 * 1024 * 1024)
    );
}

fn sc20684_conditioning(
    mode: Sc20684Mode,
    vae: &mlx_gen_wan::WanVae,
    config: &KreaRealtimeConfig,
    width: usize,
    height: usize,
    frames: usize,
    seed: u64,
) -> (mlx_gen_krea_realtime::RefConditioning, serde_json::Value) {
    use mlx_gen_wan::preprocess_i2v_image;
    use mlx_rs::ops::concatenate_axis;
    use mlx_rs::random;

    let drop_batch = |z: mlx_rs::Array| {
        let shape = z.shape().to_vec();
        z.reshape(&[shape[1], shape[2], shape[3], shape[4]])
            .expect("SC-20684 drop VAE batch axis")
    };
    match mode {
        Sc20684Mode::T2v => {
            let prompt = b"a red fox trotting through a snowy pine forest at sunrise";
            (
                mlx_gen_krea_realtime::RefConditioning {
                    context_latents: None,
                    source: None,
                },
                serde_json::json!({
                    "kind": "text-only", "sha256": sc20684_sha256(prompt), "frameCount": 0,
                    "width": width, "height": height, "vaeEncoding": "none",
                    "v2vStrength": serde_json::Value::Null,
                }),
            )
        }
        Sc20684Mode::I2v => {
            let image = gradient(width, height, seed as usize);
            let chw = preprocess_i2v_image(&image, width as u32, height as u32)
                .expect("SC-20684 preprocess I2V reference");
            let video = chw
                .expand_dims(1)
                .and_then(|value| value.expand_dims(0))
                .expect("SC-20684 batch I2V reference");
            let latents = drop_batch(vae.encode(&video).expect("SC-20684 encode I2V reference"));
            mlx_rs::transforms::eval([&latents]).expect("SC-20684 materialize I2V reference");
            (
                mlx_gen_krea_realtime::RefConditioning {
                    context_latents: Some(latents),
                    source: None,
                },
                serde_json::json!({
                    "kind": "deterministic-gradient-still",
                    "sha256": sc20684_hash_media(std::slice::from_ref(&image)), "frameCount": 1,
                    "width": width, "height": height, "vaeEncoding": "WanVae.encode-mode",
                    "v2vStrength": serde_json::Value::Null,
                }),
            )
        }
        Sc20684Mode::V2v => {
            let source: Vec<Image> = (0..frames)
                .map(|frame| smooth_frame(width, height, frame + seed as usize))
                .collect();
            let chw: Vec<mlx_rs::Array> = source
                .iter()
                .map(|frame| {
                    preprocess_i2v_image(frame, width as u32, height as u32)
                        .expect("SC-20684 preprocess V2V frame")
                        .expand_dims(1)
                        .expect("SC-20684 add V2V temporal axis")
                })
                .collect();
            let video = concatenate_axis(&chw.iter().collect::<Vec<_>>(), 1)
                .and_then(|value| value.expand_dims(0))
                .expect("SC-20684 stack V2V source");
            let latent_frames = (frames - 1) / 4 + 1;
            let key = random::key(seed).expect("SC-20684 V2V random key");
            let eps = random::normal::<f32>(
                &[
                    1,
                    config.wan.vae_z_dim as i32,
                    latent_frames as i32,
                    (height / 8) as i32,
                    (width / 8) as i32,
                ],
                None,
                None,
                Some(&key),
            )
            .expect("SC-20684 V2V latent noise");
            let latents = drop_batch(
                vae.encode_sample(&video, &eps)
                    .expect("SC-20684 encode V2V source"),
            );
            mlx_rs::transforms::eval([&latents]).expect("SC-20684 materialize V2V source");
            let strength = SC20684_V2V_STRENGTH;
            (
                mlx_gen_krea_realtime::RefConditioning {
                    context_latents: None,
                    source: Some((latents, strength)),
                },
                sc20684_v2v_input_identity(
                    &sc20684_hash_media(&source),
                    frames,
                    width,
                    height,
                    strength,
                ),
            )
        }
    }
}

/// Run one loaded, real-weight product AR route with a caller-owned cache so the observer can read
/// the exact retained kernel/cache facts after execution.  The inputs are product-shaped media
/// latents; I2V warms the cache and V2V uses the source branch, rather than relabelling T2V output.
#[allow(clippy::too_many_arguments)]
fn sc20684_generate_latents(
    transformer: &mlx_gen_krea_realtime::CausalKreaTransformer,
    config: &KreaRealtimeConfig,
    context: &mlx_rs::Array,
    params: &mlx_gen_krea_realtime::ArGenParams,
    conditioning: &mlx_gen_krea_realtime::RefConditioning,
    packed: bool,
) -> (
    mlx_rs::Array,
    mlx_gen_krea_realtime::CausalKvCache,
    std::time::Duration,
    Vec<std::time::Duration>,
) {
    use mlx_gen_krea_realtime::generate_latents_conditioned_into;
    let mut cache = transformer.new_cache();
    if packed {
        cache
            .enable_experimental_packed_metal(matches!(
                config.ar.kv_cache_quant,
                Some(KvCacheQuant { bits: 4, .. })
            ))
            .expect("enable SC-20684 packed Metal before cache mutation");
    }
    let started = Instant::now();
    let mut step_marks = Vec::new();
    let latents = generate_latents_conditioned_into(
        transformer,
        config,
        context,
        params,
        conditioning,
        &mut cache,
        &mlx_gen::CancelFlag::default(),
        &mut |progress| {
            if matches!(progress, Progress::Step { .. }) {
                step_marks.push(started.elapsed());
            }
        },
    )
    .expect("SC-20684 product route succeeds");
    mlx_rs::transforms::eval([&latents]).expect("materialize SC-20684 route output");
    (latents, cache, started.elapsed(), step_marks)
}

fn sc20684_steady_progress(
    step_marks: &[std::time::Duration],
    expected_chunks: usize,
    full_chunks: usize,
    steps_per_chunk: usize,
    frames_per_block: usize,
) -> (f64, f64, f64) {
    let step_marks_ns: Vec<u128> = step_marks.iter().map(|mark| mark.as_nanos()).collect();
    sc20684_steady_progress_from_ns(
        &step_marks_ns,
        expected_chunks,
        full_chunks,
        steps_per_chunk,
        frames_per_block,
    )
}

fn sc20684_steady_progress_from_ns(
    step_marks_ns: &[u128],
    expected_chunks: usize,
    full_chunks: usize,
    steps_per_chunk: usize,
    frames_per_block: usize,
) -> (f64, f64, f64) {
    assert_eq!(
        step_marks_ns.len(),
        expected_chunks * steps_per_chunk,
        "SC-20684 progress must cover every denoise step"
    );
    assert!(
        full_chunks >= 2 && full_chunks <= expected_chunks,
        "SC-20684 steady timing requires a post-warmup full chunk"
    );
    let intervals: Vec<u128> = step_marks_ns
        .windows(2)
        .enumerate()
        .filter(|(index, _)| !(index + 1).is_multiple_of(steps_per_chunk))
        .map(|(_, pair)| {
            pair[1]
                .checked_sub(pair[0])
                .expect("monotonic progress marks")
        })
        .collect();
    assert!(!intervals.is_empty(), "SC-20684 steady step timing");
    let mean_step_ns = intervals.iter().sum::<u128>() / intervals.len() as u128;
    let steady_chunks: Vec<u128> = (1..full_chunks)
        .map(|chunk| {
            step_marks_ns[(chunk + 1) * steps_per_chunk - 1]
                .checked_sub(step_marks_ns[chunk * steps_per_chunk - 1])
                .expect("monotonic chunk marks")
        })
        .collect();
    let mean_chunk_ns = steady_chunks.iter().sum::<u128>() / steady_chunks.len() as u128;
    let mean_chunk_ms = mean_chunk_ns as f64 / 1_000_000.0;
    let decoded_frames_per_full_chunk = frames_per_block * 4;
    (
        mean_step_ns as f64 / 1_000_000.0,
        mean_chunk_ms,
        decoded_frames_per_full_chunk as f64 * 1000.0 / mean_chunk_ms,
    )
}

#[test]
fn sc20684_steady_progress_excludes_chunk_boundaries_and_warmup() {
    let marks_ns: Vec<u128> = (1..=15).map(|step| step * 100_000_000).collect();
    assert_eq!(
        sc20684_steady_progress_from_ns(&marks_ns, 3, 2, 5, 3),
        (100.0, 500.0, 24.0)
    );
}

/// Real-weight, opt-in product observer consumed by `scripts/sc20684_krea_realtime_campaign.py`.
#[test]
#[ignore = "SC-20684 held Metal real-weight campaign; run only through its parent launcher"]
fn sc20684_packed_campaign_observer() {
    use mlx_gen_krea_realtime::{
        decode_latents_to_video, generate_latents_conditioned_into,
        load_krea_realtime_transformer_with_quant, ArGenParams, CausalKreaTransformer,
        RefConditioning,
    };
    use mlx_gen_wan::{load_tokenizer, Umt5Encoder, WanVae};
    use mlx_rs::Array;

    let campaign_started = Instant::now();
    let process_start = sc20684_process_memory();
    mlx_rs::memory::reset_peak_memory();
    mlx_gen::memory_probe::set_phase("process-start");
    let allocator_probe = mlx_gen::memory_probe::AllocatorProbe::start_default();
    let mode = Sc20684Mode::parse();
    let tier = sc20684_cache_tier();
    let run_id = std::env::var("KREA_SC20684_RUN_ID").expect("SC-20684 launcher run id");
    let measurement_role =
        std::env::var("KREA_SC20684_MEASUREMENT_ROLE").expect("SC-20684 launcher measurement role");
    assert!(
        matches!(measurement_role.as_str(), "paired" | "dense-baseline"),
        "SC-20684 measurement role must be paired|dense-baseline"
    );
    let root = PathBuf::from(
        std::env::var("KREA_SC20684_SNAPSHOT_DIR").expect("SC-20684 pinned snapshot path"),
    );
    // Fixed source-owned campaign geometry: callers choose a matrix arm, never measured geometry.
    let (width, height, frames) = (832usize, 480usize, 25usize);
    let (latent_h, latent_w) = (height / 8, width / 8);
    let latent_frames = (frames - 1) / 4 + 1;
    assert!(
        width > 0 && height > 0 && frames > 1 && latent_h > 0 && latent_w > 0,
        "SC-20684 geometry must be positive"
    );

    let mut config = KreaRealtimeConfig::krea_realtime_14b();
    config.ar.local_attn_size = -1; // packed dispatch accepts only a complete physical cache window.
    config.ar.kv_cache_quant = Some(tier);
    config.ar.frame_seq_length =
        (latent_h / config.wan.patch_size.1) * (latent_w / config.wan.patch_size.2);
    config.ar.seq_length = latent_frames * config.ar.frame_seq_length;
    config
        .validate_kv_cache_quant()
        .expect("SC-20684 tier is valid for this model");

    let load_started = Instant::now();
    let tokenizer = load_tokenizer(root.join("tokenizer.json"), config.wan.text_len)
        .expect("load product tokenizer");
    // The product's own UMT5 release (`encode_prompt`): the encoder and its weight map live only
    // inside the phase, its context is materialized, then MLX's cache of their buffers is returned.
    let context = mlx_gen_krea_realtime::materialize_and_release_phase(|| {
        let mut text_weights =
            mlx_gen::weights::Weights::from_file(root.join("t5_encoder.safetensors"))?;
        let encoder = Umt5Encoder::from_weights_quantized(
            &mut text_weights,
            &config.wan,
            mlx_gen_wan::config::WanQuant {
                bits: 8,
                group_size: 64,
            },
        )?;
        encoder.encode(
            &tokenizer,
            "a red fox trotting through a snowy pine forest at sunrise",
        )
    })
    .expect("encode product prompt and release the text encoder");
    drop(tokenizer);
    let dit_weights = mlx_gen::weights::Weights::from_file(root.join("dit.safetensors"))
        .expect("open product DiT");
    let raw: std::collections::HashMap<String, Array> = dit_weights
        .keys()
        .map(|key| {
            (
                key.to_owned(),
                dit_weights.get(key).expect("listed DiT key").clone(),
            )
        })
        .collect();
    let (dit, _) = load_krea_realtime_transformer_with_quant(raw, &config)
        .expect("load product Krea transformer");
    drop(dit_weights);
    let transformer = CausalKreaTransformer::new(dit, &config);
    let vae_weights = mlx_gen::weights::Weights::from_file(root.join("vae.safetensors"))
        .expect("open product VAE");
    let vae = WanVae::from_weights(&vae_weights).expect("load product VAE");
    drop(vae_weights);
    let load_elapsed = load_started.elapsed();
    let weights_loaded = sc20684_settled_process_memory();
    let weights_loaded_active = mlx_rs::memory::get_active_memory() as u64;
    let weights_loaded_cache = mlx_rs::memory::get_cache_memory() as u64;
    let mut phase_memory = vec![sc20684_phase_memory("weights-loaded")];
    let source_budget = sc20684_source_budget(&config, mode, &root, width, height, frames);
    let decode_plan = sc20684_decode_plan();

    let conditioning_started = Instant::now();
    let (conditioning, input) = sc20684_conditioning(mode, &vae, &config, width, height, frames, 7);
    let conditioning_elapsed = conditioning_started.elapsed();
    phase_memory.push(sc20684_phase_memory("conditioning-complete"));
    let generated_latent_frames = if mode == Sc20684Mode::I2v {
        latent_frames
            .checked_sub(1)
            .expect("SC-20684 I2V requires one reference plus continuation latents")
    } else {
        latent_frames
    };
    let params = ArGenParams {
        seed: 7,
        steps: None,
        num_latent_frames: generated_latent_frames,
        latent_height: latent_h,
        latent_width: latent_w,
        fps: 24,
        memory: Default::default(),
    };
    let expected_chunks = generated_latent_frames.div_ceil(config.ar.num_frames_per_block);
    let full_chunks = generated_latent_frames / config.ar.num_frames_per_block;
    let steps_per_chunk = config.ar.denoising_step_list.len();
    let schedule_timesteps: Vec<f64> = if mode == Sc20684Mode::V2v {
        FewStepSchedule::for_strength(
            config.ar.timestep_shift as f64,
            f64::from(SC20684_V2V_STRENGTH),
            steps_per_chunk,
        )
        .expect("resolve SC-20684 V2V strength schedule")
        .step_timesteps()
        .to_vec()
    } else {
        config
            .ar
            .denoising_step_list
            .iter()
            .map(|&value| f64::from(value))
            .collect()
    };

    if measurement_role == "dense-baseline" {
        let (dense_latents, dense_cache, generation_elapsed, dense_step_marks) =
            sc20684_generate_latents(
                &transformer,
                &config,
                &context,
                &params,
                &conditioning,
                false,
            );
        let (steady_step_ms, steady_chunk_ms, steady_output_fps) = sc20684_steady_progress(
            &dense_step_marks,
            expected_chunks,
            full_chunks,
            steps_per_chunk,
            config.ar.num_frames_per_block,
        );
        phase_memory.push(sc20684_phase_memory("dense-generation-complete"));
        let decode_started = Instant::now();
        let dense_media = decode_latents_to_video(
            &vae,
            &dense_latents,
            24,
            Some(frames),
            Some(&decode_plan),
            &mlx_gen::CancelFlag::default(),
        )
        .expect("decode dense SC-20684 baseline media");
        let decode_elapsed = decode_started.elapsed();
        phase_memory.push(sc20684_phase_memory("dense-decode-complete"));
        let (output_sha256, dense_frames) = sc20684_hash_video(&dense_media);
        let output_frame_count = dense_frames.len();
        let key_tokens = dense_cache.stored_tokens();
        let terminal_process = sc20684_settled_process_memory();
        let terminal_active = mlx_rs::memory::get_active_memory() as u64;
        let terminal_cache = mlx_rs::memory::get_cache_memory() as u64;
        let exact_active_peak = mlx_rs::memory::get_peak_memory() as u64;
        drop(dense_media);
        drop(dense_frames);
        drop(dense_cache);
        drop(dense_latents);
        drop(conditioning);
        mlx_gen::memory_probe::clear_cache();
        phase_memory.push(sc20684_phase_memory("release"));
        let release_process = sc20684_settled_process_memory();
        let release_active = mlx_rs::memory::get_active_memory() as u64;
        let release_cache = mlx_rs::memory::get_cache_memory() as u64;
        let allocator = allocator_probe.finish();
        let weights_loaded_footprint = weights_loaded["physFootprintBytes"]
            .as_u64()
            .expect("SC-20684 dense weights-loaded footprint");
        let terminal_footprint = terminal_process["physFootprintBytes"]
            .as_u64()
            .expect("SC-20684 dense terminal footprint");
        let release_footprint = release_process["physFootprintBytes"]
            .as_u64()
            .expect("SC-20684 dense release footprint");
        let release_verified = sc20684_release_verified(
            release_active,
            weights_loaded_active,
            terminal_active,
            release_cache,
            weights_loaded_cache,
            terminal_cache,
            release_footprint,
            weights_loaded_footprint,
            terminal_footprint,
        );
        let request_first_frame_ms =
            (conditioning_elapsed + generation_elapsed + decode_elapsed).as_secs_f64() * 1000.0;
        let mean_output_frame_ms = (generation_elapsed + decode_elapsed).as_secs_f64() * 1000.0
            / output_frame_count as f64;
        let repository = sc20684_repository_root();
        let tool = |program: &str, args: &[&str]| sc20684_command(program, args);
        let observation = serde_json::json!({
            "schemaVersion": 6,
            "producer": "mlx-gen-krea-realtime/sc20684-dense-baseline",
            "runId": run_id,
            "case": {"mode": mode.name(), "cacheTier": if matches!(tier, KvCacheQuant { bits: 8, .. }) { "q8" } else { "q4" }},
            "source": sc20684_source_identity(&repository),
            "model": sc20684_model_identity(&root),
            "input": input,
            "schedule": {
                "kind": "source-owned-self-forcing",
                "timesteps": schedule_timesteps.clone(),
                "steps": schedule_timesteps.len(),
                "seed": params.seed,
            },
            "toolchain": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH, "rustc": tool("rustc", &["--version"]), "cargo": tool("cargo", &["--version"]), "mlx": "mlx-rs-linked", "hardwareModel": tool("sysctl", &["-n", "hw.model"]), "metalDevice": sc20684_metal_device()},
            "geometry": {"batch": 1, "heads": config.wan.num_heads, "queryTokens": config.ar.frame_seq_length * config.ar.num_frames_per_block, "keyTokens": key_tokens, "dispatchGeometries": [], "headDim": config.wan.head_dim(), "groupSize": 64, "queryTile": 8, "keyTile": 8, "dtypes": {"packedCodes": "uint32", "packedMetadata": "bfloat16", "queryAndCurrentKv": "bfloat16", "accumulator": "float32"}, "mask": "block-causal", "width": width, "height": height, "frames": frames, "latentFrames": latent_frames, "generatedLatentFrames": generated_latent_frames},
            "sourceBudget": source_budget,
            "phaseMemory": phase_memory,
            "timing": {
                "label": "fresh-process-full-schedule-dense-read-window-baseline",
                "processWallMs": campaign_started.elapsed().as_secs_f64() * 1000.0,
                "loadMs": load_elapsed.as_secs_f64() * 1000.0,
                "conditioningMs": conditioning_elapsed.as_secs_f64() * 1000.0,
                "generationMs": generation_elapsed.as_secs_f64() * 1000.0,
                "decodeMs": decode_elapsed.as_secs_f64() * 1000.0,
                "requestFirstFrameAvailableMs": request_first_frame_ms,
                "coldProcessFirstFrameAvailableMs": load_elapsed.as_secs_f64() * 1000.0 + request_first_frame_ms,
                "meanOutputFrameMs": mean_output_frame_ms,
                "meanOutputFps": 1000.0 / mean_output_frame_ms,
                "progressStepCount": dense_step_marks.len(),
                "steadyDenoiseStepMeanMs": steady_step_ms,
                "steadyFullChunkMs": steady_chunk_ms,
                "steadyDenoiseEquivalentFps": steady_output_fps,
            },
            "memory": {
                "processStart": process_start,
                "weightsLoaded": weights_loaded,
                "generationTerminal": terminal_process,
                "release": release_process,
                "mlx": {
                    "weightsLoadedActiveBytes": weights_loaded_active,
                    "weightsLoadedCacheBytes": weights_loaded_cache,
                    "generationTerminalActiveBytes": terminal_active,
                    "generationTerminalCacheBytes": terminal_cache,
                    "exactActivePeakBytes": exact_active_peak,
                    "sampledActivePeakBytes": allocator.sampled_active_peak_bytes,
                    "sampledCachePeakBytes": allocator.sampled_cache_peak_bytes,
                    "sampledFootprintPeakBytes": allocator.sampled_footprint_peak_bytes,
                    "footprintPeakActiveBytes": allocator.footprint_peak_active_bytes,
                    "footprintPeakCacheBytes": allocator.footprint_peak_cache_bytes,
                    "sampleCount": allocator.sample_count,
                    "periodicSampleCount": allocator.periodic_sample_count,
                    "samplingSpanMicros": allocator.sampling_span_micros,
                    "intervalMicros": allocator.interval_micros,
                    "maxGapMicros": allocator.max_gap_micros,
                    "maxUncoveredGapMicros": allocator.max_uncovered_gap_micros,
                    "gaps": sc20684_sampler_gaps(&allocator),
                    "gapsNotRecorded": allocator.gaps_not_recorded,
                    "releaseWindowCount": allocator.release_window_count,
                    "realtimeSampler": allocator.realtime_sampler,
                    "releaseActiveBytes": release_active,
                    "releaseCacheBytes": release_cache,
                },
                "releaseVerified": release_verified,
            },
            "output": {"status": "generated", "sha256": output_sha256},
        });
        println!(
            "SC20684_KREA_BASELINE_OBSERVATION {}",
            serde_json::to_string(&observation).expect("serialize SC-20684 baseline observation")
        );
        return;
    }

    let (packed_latents, packed_cache, packed_elapsed, packed_step_marks) =
        sc20684_generate_latents(
            &transformer,
            &config,
            &context,
            &params,
            &conditioning,
            true,
        );
    let (steady_step_ms, steady_chunk_ms, steady_output_fps) = sc20684_steady_progress(
        &packed_step_marks,
        expected_chunks,
        full_chunks,
        steps_per_chunk,
        config.ar.num_frames_per_block,
    );
    phase_memory.push(sc20684_phase_memory("packed-generation-complete"));
    let packed_terminal = sc20684_process_memory();
    let packed_decode_started = Instant::now();
    let packed_media = decode_latents_to_video(
        &vae,
        &packed_latents,
        24,
        Some(frames),
        Some(&decode_plan),
        &mlx_gen::CancelFlag::default(),
    )
    .expect("decode packed product media");
    let packed_decode_elapsed = packed_decode_started.elapsed();
    phase_memory.push(sc20684_phase_memory("packed-decode-complete"));
    let (output_sha256, packed_frames) = sc20684_hash_video(&packed_media);
    let packed_frame_count = packed_frames.len();
    let candidate_terminal = sc20684_process_memory();
    let candidate_terminal_active = mlx_rs::memory::get_active_memory() as u64;
    let candidate_terminal_cache = mlx_rs::memory::get_cache_memory() as u64;
    let exact_active_peak = mlx_rs::memory::get_peak_memory() as u64;
    drop(packed_media);
    let allocator = allocator_probe.finish();
    let candidate_wall_ms = campaign_started.elapsed().as_secs_f64() * 1000.0;
    // The candidate's measurements are taken. Return the packed decode's cached working set before
    // the dense parity generation: its buffers fit no DiT shape, so MLX would otherwise hold them
    // beside the dense run until its ~0.95 x working-set trim, far above the child cap.
    mlx_gen::memory_probe::clear_cache();
    // Printed like every phase reading, but kept out of `phaseMemory`, whose labels the launcher
    // validates in a fixed order. Without the clear this reads the packed decode's ~12 GiB working
    // set; the 1 GiB bound tolerates a Metal completion handler recycling a late temporary between
    // the clear and the read (nothing else allocates here: the allocator probe has finished and
    // only reads counters).
    let dense_parity_entry = sc20684_phase_memory("dense-parity-entry");
    let dense_parity_entry_cache = dense_parity_entry["mlxCacheBytes"]
        .as_u64()
        .expect("SC-20684 dense-parity-entry cache bytes");
    assert!(
        dense_parity_entry_cache < 1 << 30,
        "SC-20684 dense parity must start with the packed decode's cache returned: \
         {dense_parity_entry_cache} cached bytes"
    );

    let (dense_latents, _, dense_elapsed, _) = sc20684_generate_latents(
        &transformer,
        &config,
        &context,
        &params,
        &conditioning,
        false,
    );
    drop(conditioning);
    phase_memory.push(sc20684_phase_memory("dense-generation-complete"));
    mlx_rs::transforms::eval([&packed_latents, &dense_latents])
        .expect("materialize paired parity latents");
    let parity_error = mlx_rs::ops::subtract(&packed_latents, &dense_latents)
        .expect("paired subtract")
        .abs()
        .expect("paired abs")
        .max(None)
        .expect("paired max")
        .item::<f32>();
    let parity_tolerance = if matches!(tier, KvCacheQuant { bits: 8, .. }) {
        SC20684_Q8_PARITY_MAX_ABS_ERROR
    } else {
        SC20684_Q4_PARITY_MAX_ABS_ERROR
    };

    let dense_decode_started = Instant::now();
    let dense_media = decode_latents_to_video(
        &vae,
        &dense_latents,
        24,
        Some(frames),
        Some(&decode_plan),
        &mlx_gen::CancelFlag::default(),
    )
    .expect("decode dense product media");
    let dense_decode_elapsed = dense_decode_started.elapsed();
    phase_memory.push(sc20684_phase_memory("dense-decode-complete"));
    let (dense_output_sha256, dense_frames) = sc20684_hash_video(&dense_media);
    drop(dense_media);
    drop(dense_latents);
    let quality_error = sc20684_max_rgb_error(&packed_frames, &dense_frames);
    let quality_mean_error = mean_abs_delta(&packed_frames, &dense_frames);
    let packed_temporal_delta = sc20684_mean_temporal_delta(&packed_frames);
    let dense_temporal_delta = sc20684_mean_temporal_delta(&dense_frames);
    let temporal_delta_drift = (packed_temporal_delta - dense_temporal_delta).abs();
    drop(dense_frames);
    let quality_tolerance = if matches!(tier, KvCacheQuant { bits: 8, .. }) {
        SC20684_Q8_MAX_ABS_RGB_U8
    } else {
        SC20684_Q4_MAX_ABS_RGB_U8
    };
    let quality_mean_tolerance = if matches!(tier, KvCacheQuant { bits: 8, .. }) {
        SC20684_Q8_MEAN_ABS_RGB_U8
    } else {
        SC20684_Q4_MEAN_ABS_RGB_U8
    };
    let temporal_tolerance = if matches!(tier, KvCacheQuant { bits: 8, .. }) {
        SC20684_Q8_TEMPORAL_DELTA_DRIFT
    } else {
        SC20684_Q4_TEMPORAL_DELTA_DRIFT
    };
    let quality_pass = quality_error <= quality_tolerance
        && quality_mean_error <= quality_mean_tolerance
        && temporal_delta_drift <= temporal_tolerance;

    let artifact_root = PathBuf::from(
        std::env::var("KREA_SC20684_ARTIFACT_DIR")
            .expect("SC-20684 launcher must provide an external artifact directory"),
    );
    assert!(
        !artifact_root.exists(),
        "SC-20684 artifact directory must be new"
    );
    std::fs::create_dir_all(&artifact_root).expect("create SC-20684 artifact directory");
    let mut artifacts = Vec::with_capacity(packed_frames.len());
    for (index, frame) in packed_frames.iter().enumerate() {
        let name = format!("frame-{index:03}.ppm");
        let path = artifact_root.join(&name);
        write_ppm(&path, frame);
        let bytes = std::fs::read(&path).expect("read SC-20684 review artifact");
        artifacts.push(serde_json::json!({
            "path": name,
            "sha256": sc20684_sha256(&bytes),
            "bytes": bytes.len(),
        }));
    }
    drop(packed_frames);

    let receipt = packed_cache
        .packed_metal_route_receipt()
        .expect("packed route owns its retained kernel");
    let key_tokens = packed_cache.stored_tokens();
    drop(packed_cache);
    let mut cancelled_cache = transformer.new_cache();
    cancelled_cache
        .enable_experimental_packed_metal(matches!(tier, KvCacheQuant { bits: 4, .. }))
        .expect("enable packed cancellation probe");
    let cancellation_route_before = cancelled_cache
        .packed_metal_route_receipt()
        .expect("cancellation cache owns its packed route before dispatch");
    let cancellation_context_latents = packed_latents
        .take_axis(Array::from_slice(&[0i32], &[1]), 1)
        .expect("select deterministic cancellation context latent");
    mlx_rs::transforms::eval([&cancellation_context_latents])
        .expect("materialize deterministic cancellation context latent");
    drop(packed_latents);
    let cancellation_conditioning = RefConditioning {
        context_latents: Some(cancellation_context_latents),
        source: None,
    };
    let cancellation_active_before = mlx_rs::memory::get_active_memory() as u64;
    let cancellation_cache_before = mlx_rs::memory::get_cache_memory() as u64;
    let cancelled = mlx_gen::CancelFlag::default();
    let mut cancellation_progress_steps = 0u64;
    let cancelled_result = generate_latents_conditioned_into(
        &transformer,
        &config,
        &context,
        &params,
        &cancellation_conditioning,
        &mut cancelled_cache,
        &cancelled,
        &mut |progress| {
            if matches!(progress, Progress::Step { .. }) && cancellation_progress_steps == 0 {
                cancellation_progress_steps = 1;
                cancelled.cancel();
            }
        },
    );
    let cancellation_observed = matches!(cancelled_result, Err(mlx_gen::Error::Canceled));
    let cancellation_expected_context_tokens = config.ar.frame_seq_length;
    let cancellation_stored_tokens_after = cancelled_cache.stored_tokens();
    let cancellation_state_unchanged =
        cancellation_stored_tokens_after == cancellation_expected_context_tokens;
    let cancellation_route_at_cancel = cancelled_cache
        .packed_metal_route_receipt()
        .expect("cancellation cache owns its packed route after dispatch");
    let cancellation_dispatch_token =
        sc20684_cancellation_dispatch_token(&run_id, &cancellation_route_at_cancel);
    let cancellation_route_before_json =
        sc20684_route_lifecycle_snapshot(&cancellation_route_before);
    let cancellation_route_at_cancel_json =
        sc20684_route_lifecycle_snapshot(&cancellation_route_at_cancel);
    let cancellation_dispatch_delta = cancellation_route_at_cancel
        .accepted_forwards
        .saturating_sub(cancellation_route_before.accepted_forwards);
    let cancellation_scratch_dispatch_delta = cancellation_route_at_cancel
        .materialized_scratch_dispatches
        .saturating_sub(cancellation_route_before.materialized_scratch_dispatches);
    let cancellation_route_allocation_observed = cancellation_dispatch_delta == 1
        && cancellation_scratch_dispatch_delta == cancellation_dispatch_delta
        && cancellation_route_before.bounded_scratch_bytes == 0
        && cancellation_route_at_cancel.bounded_scratch_bytes
            == mlx_gen_krea_realtime::compressed_kv::PACKED_METAL_THREADGROUP_SCRATCH_BYTES;
    drop(cancelled_cache);
    drop(cancellation_conditioning);
    mlx_gen::memory_probe::clear_cache();
    let cancellation_active_after_release = mlx_rs::memory::get_active_memory() as u64;
    let cancellation_cache_after_release = mlx_rs::memory::get_cache_memory() as u64;
    let cancellation_scratch_released = cancellation_active_after_release
        <= cancellation_active_before
        && cancellation_cache_after_release <= cancellation_cache_before;
    let cancellation_clean = cancellation_observed
        && cancellation_progress_steps == 1
        && cancellation_route_allocation_observed
        && cancellation_state_unchanged
        && cancellation_scratch_released;
    phase_memory.push(sc20684_phase_memory("cancellation-complete"));

    let steady_forward_ns = u64::try_from(receipt.steady_evaluated_forward_count)
        .ok()
        .filter(|&count| count > 0)
        .map(|count| receipt.steady_evaluated_forward_ns / count)
        .unwrap_or(0);
    let compile_upper_bound_ns = receipt
        .first_evaluated_forward_ns
        .saturating_sub(steady_forward_ns);
    let append_mean_ns = u64::try_from(receipt.accepted_append_count)
        .ok()
        .filter(|&count| count > 0)
        .map(|count| receipt.accepted_append_ns / count)
        .unwrap_or(0);
    let ns_ms = |value: u64| value as f64 / 1_000_000.0;
    let request_first_frame_ms =
        (conditioning_elapsed + packed_elapsed + packed_decode_elapsed).as_secs_f64() * 1000.0;
    let cold_process_first_frame_ms = load_elapsed.as_secs_f64() * 1000.0 + request_first_frame_ms;
    let mean_output_frame_ms =
        (packed_elapsed + packed_decode_elapsed).as_secs_f64() * 1000.0 / packed_frame_count as f64;

    let verification_terminal = sc20684_settled_process_memory();
    let verification_terminal_active = mlx_rs::memory::get_active_memory() as u64;
    let verification_terminal_cache = mlx_rs::memory::get_cache_memory() as u64;
    mlx_gen::memory_probe::clear_cache();
    phase_memory.push(sc20684_phase_memory("release"));
    let release_process = sc20684_settled_process_memory();
    let release_active = mlx_rs::memory::get_active_memory() as u64;
    let release_cache = mlx_rs::memory::get_cache_memory() as u64;
    let weights_loaded_footprint = weights_loaded["physFootprintBytes"]
        .as_u64()
        .expect("SC-20684 weights-loaded footprint");
    let verification_terminal_footprint = verification_terminal["physFootprintBytes"]
        .as_u64()
        .expect("SC-20684 verification-terminal footprint");
    let release_footprint = release_process["physFootprintBytes"]
        .as_u64()
        .expect("SC-20684 release footprint");
    let release_verified = cancellation_clean
        && sc20684_release_verified(
            release_active,
            weights_loaded_active,
            verification_terminal_active,
            release_cache,
            weights_loaded_cache,
            verification_terminal_cache,
            release_footprint,
            weights_loaded_footprint,
            verification_terminal_footprint,
        );

    let repository = sc20684_repository_root();
    let tool = |program: &str, args: &[&str]| sc20684_command(program, args);
    let observation = serde_json::json!({
        "schemaVersion": 6,
        "producer": "mlx-gen-krea-realtime/sc20684",
        "runId": run_id,
        "case": {"mode": mode.name(), "cacheTier": if matches!(tier, KvCacheQuant { bits: 8, .. }) { "q8" } else { "q4" }},
        "source": sc20684_source_identity(&repository),
        "model": sc20684_model_identity(&root),
        "input": input,
        "schedule": {
            "kind": "source-owned-self-forcing",
            "timesteps": schedule_timesteps.clone(),
            "steps": schedule_timesteps.len(),
            "seed": params.seed,
        },
        "toolchain": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH, "rustc": tool("rustc", &["--version"]), "cargo": tool("cargo", &["--version"]), "mlx": "mlx-rs-linked", "hardwareModel": tool("sysctl", &["-n", "hw.model"]), "metalDevice": sc20684_metal_device()},
        "geometry": {"batch": 1, "heads": config.wan.num_heads, "queryTokens": config.ar.frame_seq_length * config.ar.num_frames_per_block, "keyTokens": key_tokens, "dispatchGeometries": receipt.dispatch_geometries.iter().map(|geometry| serde_json::json!({"queryTokens": geometry.query_tokens, "keyTokens": geometry.key_tokens, "acceptedForwards": geometry.accepted_forwards})).collect::<Vec<_>>(), "headDim": config.wan.head_dim(), "groupSize": 64, "queryTile": 8, "keyTile": 8, "dtypes": {"packedCodes": "uint32", "packedMetadata": "bfloat16", "queryAndCurrentKv": "bfloat16", "accumulator": "float32"}, "mask": "block-causal", "width": width, "height": height, "frames": frames, "latentFrames": latent_frames, "generatedLatentFrames": generated_latent_frames},
        "sourceBudget": source_budget,
        "phaseMemory": phase_memory,
        "compiledHandle": {"identity": receipt.compiled_handle_identity, "retainedBytes": receipt.retained_handle_bytes, "compiled": receipt.accepted_forwards > 0, "acceptedDispatches": receipt.accepted_forwards},
        "bytes": {"persistent": receipt.persistent_bytes, "retainedHandle": receipt.retained_handle_bytes, "boundedScratch": receipt.bounded_scratch_bytes, "denseWindow": receipt.dense_window_bytes, "scoreMatrix": receipt.score_matrix_bytes},
        "timing": {
            "label": "fresh-process-full-schedule-loaded-product",
            "processWallMs": campaign_started.elapsed().as_secs_f64() * 1000.0,
            "candidateWallMs": candidate_wall_ms,
            "loadMs": load_elapsed.as_secs_f64() * 1000.0,
            "conditioningMs": conditioning_elapsed.as_secs_f64() * 1000.0,
            "packedGenerationMs": packed_elapsed.as_secs_f64() * 1000.0,
            "denseGenerationMs": dense_elapsed.as_secs_f64() * 1000.0,
            "packedDecodeMs": packed_decode_elapsed.as_secs_f64() * 1000.0,
            "denseDecodeMs": dense_decode_elapsed.as_secs_f64() * 1000.0,
            "requestFirstFrameAvailableMs": request_first_frame_ms,
            "coldProcessFirstFrameAvailableMs": cold_process_first_frame_ms,
            "meanOutputFrameMs": mean_output_frame_ms,
            "meanOutputFps": 1000.0 / mean_output_frame_ms,
            "progressStepCount": packed_step_marks.len(),
            "steadyDenoiseStepMeanMs": steady_step_ms,
            "steadyFullChunkMs": steady_chunk_ms,
            "steadyDenoiseEquivalentFps": steady_output_fps,
            "coldEvaluatedPackedForwardMs": ns_ms(receipt.first_evaluated_forward_ns),
            "steadyEvaluatedPackedForwardMeanMs": ns_ms(steady_forward_ns),
            "steadyEvaluatedPackedForwardCount": receipt.steady_evaluated_forward_count,
            "compileUpperBoundMs": ns_ms(compile_upper_bound_ns),
            "acceptedPackedAppendMeanMs": ns_ms(append_mean_ns),
            "acceptedPackedAppendCount": receipt.accepted_append_count,
        },
        "memory": {
            "processStart": process_start,
            "weightsLoaded": weights_loaded,
            "packedTerminal": packed_terminal,
            "candidateTerminal": candidate_terminal,
            "verificationTerminal": verification_terminal,
            "release": release_process,
            "mlx": {
                "weightsLoadedActiveBytes": weights_loaded_active,
                "weightsLoadedCacheBytes": weights_loaded_cache,
                "candidateTerminalActiveBytes": candidate_terminal_active,
                "candidateTerminalCacheBytes": candidate_terminal_cache,
                "verificationTerminalActiveBytes": verification_terminal_active,
                "verificationTerminalCacheBytes": verification_terminal_cache,
                "exactActivePeakBytes": exact_active_peak,
                "sampledActivePeakBytes": allocator.sampled_active_peak_bytes,
                "sampledCachePeakBytes": allocator.sampled_cache_peak_bytes,
                "sampledFootprintPeakBytes": allocator.sampled_footprint_peak_bytes,
                "footprintPeakActiveBytes": allocator.footprint_peak_active_bytes,
                "footprintPeakCacheBytes": allocator.footprint_peak_cache_bytes,
                "sampleCount": allocator.sample_count,
                "periodicSampleCount": allocator.periodic_sample_count,
                "samplingSpanMicros": allocator.sampling_span_micros,
                "intervalMicros": allocator.interval_micros,
                "maxGapMicros": allocator.max_gap_micros,
                "maxUncoveredGapMicros": allocator.max_uncovered_gap_micros,
                "gaps": sc20684_sampler_gaps(&allocator),
                "gapsNotRecorded": allocator.gaps_not_recorded,
                "releaseWindowCount": allocator.release_window_count,
                "realtimeSampler": allocator.realtime_sampler,
                "releaseActiveBytes": release_active,
                "releaseCacheBytes": release_cache,
            },
            "releaseVerified": release_verified,
        },
        "parity": {"status": if parity_error <= parity_tolerance { "pass" } else { "fail" }, "candidateTier": if matches!(tier, KvCacheQuant { bits: 8, .. }) { "q8" } else { "q4" }, "maxAbsError": parity_error, "tolerance": parity_tolerance},
        "quality": {
            "status": if quality_pass { "pass" } else { "fail" },
            "candidateTier": if matches!(tier, KvCacheQuant { bits: 8, .. }) { "q8" } else { "q4" },
            "metric": "paired-rgb-and-temporal-delta",
            "maxAbsRgbU8": quality_error,
            "maxAbsRgbU8Tolerance": quality_tolerance,
            "meanAbsRgbU8": quality_mean_error,
            "meanAbsRgbU8Tolerance": quality_mean_tolerance,
            "packedMeanTemporalDelta": packed_temporal_delta,
            "denseMeanTemporalDelta": dense_temporal_delta,
            "temporalDeltaDrift": temporal_delta_drift,
            "temporalDeltaDriftTolerance": temporal_tolerance,
            "acknowledged": matches!(tier, KvCacheQuant { bits: 8, .. }) || std::env::var("KREA_SC20684_Q4_QUALITY_ARM").as_deref() == Ok("acknowledged"),
        },
        "fallback": {"count": receipt.dense_fallbacks, "reason": receipt.last_fallback_reason},
        "cancellation": {
            "status": if cancellation_clean { "pass" } else { "fail" },
            "requests": 1,
            "trigger": "after-first-materialized-denoise-step",
            "progressStepsBeforeCancel": cancellation_progress_steps,
            "typedCancellationObserved": cancellation_observed,
            "dispatchToken": cancellation_dispatch_token,
            "routeBefore": cancellation_route_before_json,
            "routeAtCancel": cancellation_route_at_cancel_json,
            "expectedContextTokens": cancellation_expected_context_tokens,
            "storedTokensAfterCancel": cancellation_stored_tokens_after,
            "activeBytesBefore": cancellation_active_before,
            "activeBytesAfterRelease": cancellation_active_after_release,
            "cacheBytesBefore": cancellation_cache_before,
            "cacheBytesAfterRelease": cancellation_cache_after_release,
            "allocationObserved": cancellation_route_allocation_observed,
            "partialStateMutation": !cancellation_state_unchanged,
            "scratchReleased": cancellation_scratch_released,
        },
        "output": {"status": "generated", "sha256": output_sha256, "denseSha256": dense_output_sha256, "artifacts": artifacts},
    });
    println!(
        "SC20684_KREA_PROVIDER_OBSERVATION {}",
        serde_json::to_string(&observation).expect("serialize SC-20684 observation")
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED generate_smoke::sc20684_packed_campaign_observer");
}

/// **The S13 product-path run:** timing, memory, and the *structural* coherence floor.
///
/// ⚠️ **This gate does NOT certify image quality, and the clip it produces is known to be visibly
/// corrupted** (sc-15325 — the tiled VAE decode blows ~26% of one frame in eight to near-white). The
/// assertions below check flatness, plausible mean, tail motion and gross brightness drift; **none of
/// those is violated by the corruption**, which is precisely why it went unnoticed in the first cut of
/// this work. `report_artifacts` is therefore called on the product clip so the clipping figure is at
/// least *printed* by the story's headline run — a green tick here means "the pipeline ran and produced
/// a structurally sane clip", not "the clip looks right". Tightening this into a real quality gate
/// depends on sc-15325 landing, since today it would fail by design.
#[test]
#[ignore = "real ~20 GB (q4) / ~40 GB (bf16) snapshot; run with --ignored on macOS (see module doc)"]
fn t2v_produces_a_coherent_clip() {
    let w = env_usize("KREA_SMOKE_W", 832);
    let h = env_usize("KREA_SMOKE_H", 480);
    let frames = env_usize("KREA_SMOKE_FRAMES", 81);
    let steps = env_opt_usize("KREA_SMOKE_STEPS");
    let seed = env_usize("KREA_SMOKE_SEED", 7) as u64;

    let cfg = KreaRealtimeConfig::krea_realtime_14b();
    let latent_frames = (frames - 1) / 4 + 1;
    let chunks = latent_frames.div_ceil(cfg.ar.num_frames_per_block);

    let req = smoke_request(w, h, frames, steps, seed);
    let r = run(&req, chunks, "t2v");
    report("t2v", &r, w, h, frames);
    assert_eq!(r.fps, 24, "fps passthrough");
    assert_eq!(
        r.frames.len(),
        frames,
        "the decode must be trimmed back to the requested frame count"
    );
    assert_coherent(&r.frames, w, h, "t2v");
    // Print what `assert_coherent` structurally cannot see (sc-15325). Compare the clipping figure
    // against the ~0.08% a single-pass decode of the same latents achieves.
    report_artifacts(&r.frames, "t2v/product-path");
    dump_frames(&r.frames, "t2v");
}

/// **S7 deferred minor #1 — single-frame anchor vs. the reference's repeated anchor.**
///
/// `generate_i2v` warms **one** clean-context latent frame from the still, while
/// `release_server.py::setup_start_frame` repeats the still to `kv_cache_num_frames` (3) latent frames
/// so the first generated chunk starts frame-block aligned. The engine seam accepts either, so this
/// measures the difference on real weights instead of arguing it: both runs must be coherent, and the
/// per-frame stats are printed so the anchor choice can be judged rather than assumed.
#[test]
#[ignore = "real snapshot; run with --ignored on macOS (see module doc)"]
fn i2v_single_frame_anchor_is_coherent() {
    let w = env_usize("KREA_SMOKE_W", 832);
    let h = env_usize("KREA_SMOKE_H", 480);
    let frames = env_usize("KREA_SMOKE_FRAMES", 33);
    let steps = env_opt_usize("KREA_SMOKE_STEPS");

    let cfg = KreaRealtimeConfig::krea_realtime_14b();
    let latent_frames = (frames - 1) / 4 + 1;
    // i2v generates `total_latent - 1` frames (latent frame 0 is the reference).
    let chunks = (latent_frames - 1).div_ceil(cfg.ar.num_frames_per_block);

    let req = i2v_request(w, h, frames, steps, 11, gradient(w, h, 0));
    let r = run(&req, chunks, "i2v");
    report("i2v/anchor=1", &r, w, h, frames);
    assert_eq!(r.frames.len(), frames);
    assert_coherent(&r.frames, w, h, "i2v/anchor=1");
    dump_frames(&r.frames, "i2v_anchor1");
}

/// **S7 deferred minor #2 — is v2v at `strength = 0` genuinely identity?**
///
/// At `strength = 0` every entry of the strength-scaled schedule is timestep 0, so `σ = 0` throughout:
/// the init is `source·(1−σ) + ε·σ` = the source latents verbatim, each Euler step is
/// `x − 0·v = x`, and each renoise is `(1−0)·x0 + 0·ε = x0`. The **denoise** is therefore an exact
/// identity on the latents. But the *clip* is not identity on pixels, and this pins down why with
/// controls rather than assertions:
///
///   * `A` — source → VAE round-trip through `encode` (the distribution **mode**). The floor any
///     encode/decode pays.
///   * `A'` — mode round-trip → `encode_sample` round-trip. The v2v path deliberately uses
///     `.sample()` (the reference's video-source path), which draws `mean + std·ε` per latent. `A'` is
///     the size of that draw, measured, not assumed.
///   * `C` — mode round-trip → v2v@0. What is left once the VAE mode round-trip is accounted for.
///
/// The finding this encodes: v2v@0 is identity **in the latents** — the decisive form of that is
/// [`v2v_strength_zero_is_latent_identity`], which drives the AR loop on encoded latents with no VAE
/// between the source and the answer and measures the residual directly. What is left in `C` is that
/// residual carried through the decode, not the `.sample()` draw: `A'` measures the draw at ~0.01/255,
/// two orders of magnitude under `C`, so sampling cannot be the explanation. See the MEASURED block on
/// the assertions for the numbers and for what `C` is gated against.
#[test]
#[ignore = "real snapshot; run with --ignored on macOS (see module doc)"]
fn v2v_strength_zero_preserves_the_source() {
    use mlx_gen_wan::{preprocess_i2v_image, WanVae};
    use mlx_rs::ops::concatenate_axis;
    use mlx_rs::random;

    let w = env_usize("KREA_SMOKE_W", 832);
    let h = env_usize("KREA_SMOKE_H", 480);
    let frames = env_usize("KREA_SMOKE_FRAMES", 33);

    // Pin the decode budget across BOTH the product render and the control below (sc-17276).
    // `decode_tiling` is free-aware — `free x 0.85`, `free = MLX limit - resident` — so with no pin the
    // plan is a function of whatever happened to be free at each call, and the two calls are made at
    // very different residencies: the product decodes inside `run` with the DiT and TE resident, the
    // control decodes afterwards with them dropped. A host big enough to fit a single-pass decode also
    // takes the `None` branch for both, which silently changes what `A` measures. That free variable is
    // literally the second cause the `C` assertion names, and it must not be one of this test's inputs.
    // Same pin, same reason, as `long_clip_coherence_under_the_bounded_window`; 20 GiB is the operating
    // point `decode_tiling`'s own docs derive at 832x480.
    std::env::set_var("WAN_VAE_BUDGET_GIB", "20");

    let source: Vec<Image> = (0..frames).map(|i| smooth_frame(w, h, i)).collect();
    let req = v2v_zero_request(w, h, source.clone());

    let cfg = KreaRealtimeConfig::krea_realtime_14b();
    let latent_frames = (frames - 1) / 4 + 1;
    let chunks = latent_frames.div_ceil(cfg.ar.num_frames_per_block);
    let r = run(&req, chunks, "v2v/strength=0");
    report("v2v/strength=0", &r, w, h, frames);
    assert_eq!(r.frames.len(), frames);
    dump_frames(&r.frames, "v2v_strength0");

    // The controls: the same source through the same VAE and nothing else, once at the distribution
    // mode and once sampled (the encode the v2v path actually performs).
    let root = require_snapshot();
    let vw =
        mlx_gen::weights::Weights::from_file(root.join("vae.safetensors")).expect("open the VAE");
    let vae = WanVae::from_weights(&vw).expect("load the z16 Wan VAE");
    let chw: Vec<mlx_rs::Array> = source
        .iter()
        .map(|f| {
            preprocess_i2v_image(f, w as u32, h as u32)
                .expect("preprocess")
                .expand_dims(1)
                .expect("expand")
        })
        .collect();
    let video = concatenate_axis(&chw.iter().collect::<Vec<_>>(), 1)
        .expect("stack the source clip")
        .expand_dims(0)
        .expect("batch axis");

    // Decode through the SAME windowing the product path uses — a single-pass control is not
    // comparable to a tiled product decode, and the difference is not small (see the tile-seam note in
    // the S13 findings).
    let tiling = decode_tiling(h, w, (latent_frames * 4) as i32).expect("plan the decode tiling");
    println!("  control decode tiling: {tiling:?}");
    let decode = |latents: &mlx_rs::Array| -> Vec<Image> {
        match decode_latents_to_video(
            &vae,
            latents,
            24,
            Some(frames),
            tiling.as_ref(),
            &mlx_gen::CancelFlag::default(),
        )
        .expect("VAE decode")
        {
            GenerationOutput::Video { frames, .. } => frames,
            other => panic!("expected a Video output, got {other:?}"),
        }
    };
    let drop_batch = |z: mlx_rs::Array| {
        let s = z.shape().to_vec();
        z.reshape(&[s[1], s[2], s[3], s[4]]).expect("drop batch")
    };

    let z_mode = drop_batch(vae.encode(&video).expect("VAE encode (mode)"));
    let mode_rt = decode(&z_mode);
    let key = random::key(99).expect("key");
    let eps = random::normal::<f32>(
        &[
            1,
            z_mode.shape()[0],
            z_mode.shape()[1],
            z_mode.shape()[2],
            z_mode.shape()[3],
        ],
        None,
        None,
        Some(&key),
    )
    .expect("eps");
    let z_sample = drop_batch(
        vae.encode_sample(&video, &eps)
            .expect("VAE encode (sampled)"),
    );
    let sample_rt = decode(&z_sample);
    // Both decodes are done, so the pin has served its purpose. Dropped HERE rather than at the end of
    // the test so the assertions below cannot leak a 20 GiB budget into whatever runs next in this
    // process — these tests share one process under `--test-threads 1`, and libtest carries on to the
    // next test after a panic. A panic ABOVE this line still leaks, as it does at the other four pin
    // sites in this file; today that is inert because this test sorts last of the six the lane runs.
    std::env::remove_var("WAN_VAE_BUDGET_GIB");
    dump_frames(&mode_rt, "v2v_vae_roundtrip_mode");

    let a = mean_abs_delta(&source, &mode_rt);
    let a_prime = mean_abs_delta(&mode_rt, &sample_rt);
    let b = mean_abs_delta(&source, &r.frames);
    let c = mean_abs_delta(&mode_rt, &r.frames);
    println!(
        "  v2v/strength=0 mean |Δ| (0..255): A source->VAE(mode) = {a:.2}, \
         A' VAE(mode)->VAE(sample) = {a_prime:.2}, B source->v2v@0 = {b:.2}, \
         C VAE(mode)->v2v@0 = {c:.2}"
    );
    // MEASURED (832x480, 33 frames, Q4, budget pinned to 20 GiB ⇒ spatial 256/64, no temporal tiling):
    // A = 2.98, A' = 0.01, B = 2.94, **C = 0.37**.
    //
    // So: v2v at strength 0 IS identity — the output sits ~0.4/255 from a plain VAE round-trip of its
    // own source, and `B` tracks `A` to well under a tenth of `A`, i.e. the whole distance from the
    // source to the v2v@0 clip is the VAE's own round-trip and nothing else moved the picture.
    //
    // WHERE THE REMAINING ~0.4 COMES FROM, and why it is not gated against `A'`: it is the AR loop's
    // numerical residual. [`v2v_strength_zero_is_latent_identity`] measures that residual directly in
    // the latents — it gates it at 2% of the source scale, and on the same Q4 snapshot both this
    // machine and CI run 30787887176 measure it at 0.91% (mean |d| 0.01044 against a source mean |x|
    // of 1.15338). Carried through the decode, that is the ~0.4/255 seen here. It is NOT the
    // `.sample()` draw: `A'` measures that draw at ~0.01/255, ~40x smaller.
    //
    // WHY THIS IS AN ABSOLUTE BUDGET AND NOT `C < A * 0.1` (sc-17276). That relative form was
    // calibrated by sc-8446 (1deefff6, #287) when `A` was ~27.6 — the tiled decode of the day, which
    // sc-15325 (51d65a1a, #292) then fixed. Post-fix a tiled decode reaches single-pass quality (the
    // spatial-only plan this bucket selects is 0.31/255 against single-pass, `decode_tiling`'s docs),
    // so `A` collapsed ~9x to ~3.0 while `C`, an AR-loop residual, did not scale with it at all;
    // 0.1*A became 0.30 against a 0.37-0.38 measurement and the gate has been latently red ever since.
    // `C` is not a fraction of the VAE's error, so it must not be gated as one.
    //
    // AND WHY THE PIN IS STILL LOAD-BEARING even though it did not cause that red (unpinned this
    // measures C = 0.38, pinned 0.37 — the old gate failed at both). It is the same sc-15325 fix that
    // makes it necessary: a control/product decode-plan mismatch used to be worth ~26/255 and was
    // caught by any sane gate, and post-fix it is worth ~0.3/255 — INSIDE this budget. The gate can no
    // longer see that confound, so the pin removes it by construction instead.
    //
    // THE BUDGET'S HEADROOM, both sides, MEASURED — a budget nobody has driven to red is a budget
    // nobody knows the width of. Below: 0.37. Above: the same test body re-run against a
    // strength-0.25 render of the same source, everything else held (sc-17276), measures A = 2.98,
    // B = 13.81, **C = 13.77** — 37x the strength-0 value and 13.8x this budget, and it trips this
    // assertion. So the budget sits between a measured pass at 0.37 and a measured fail at 13.77
    // rather than between a measurement and a guess.
    const V2V0_IDENTITY_BUDGET: f64 = 1.0;
    assert!(
        a > 1.0,
        "the VAE round-trip control is a near no-op (A={a:.2}) — C cannot be interpreted against it"
    );
    assert!(
        c < V2V0_IDENTITY_BUDGET,
        "v2v@0 is {c:.2}/255 from a VAE round-trip of its own source, over the \
         {V2V0_IDENTITY_BUDGET:.2}/255 identity budget — the strength=0 denoise is not behaving as an \
         identity (A={a:.2}, A'={a_prime:.2}). The decode is not a candidate explanation the way it \
         used to be: both clips above went through one pinned plan"
    );
    // NOT an independent measurement — `|B - A| <= C` always, so this can only fire where `C` already
    // could. It is the tighter bound on the DIRECTIONAL part of `C`, which is the part a pipeline that
    // renders instead of preserving produces: the strength-0.25 arm above moves B to 13.81 against A's
    // 2.98. Cheap, and it fails ~3x sooner than the absolute budget when the deviation is one-way.
    assert!(
        (b - a).abs() < a * 0.1,
        "source->v2v@0 is {b:.2}/255 against a {a:.2}/255 VAE round-trip of the same source — a \
         strength-0 render must be its source's round-trip and nothing more, so these two must agree"
    );
    assert!(b > 0.0, "B is exactly zero — the clips are the same object");
    assert_coherent(&r.frames, w, h, "v2v/strength=0");
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED generate_smoke::v2v_strength_zero_preserves_the_source"
    );
}

/// **sc-15325 real-weight regression guard — the new policy reaches single-pass quality at the OLD
/// policy's memory peak; the old policy did not.**
///
/// Real z16 VAE, real encode/decode, the same latents decoded every way. Three arms:
///
///  * **single-pass** — the reference every tiled decode approximates (guarded to stay under the z16
///    write cap; past it MLX writes silently wrong pixels and the reference is garbage, sc-15402);
///  * **the OLD policy**, reproduced verbatim from the deleted `DECODE_TILE_BUDGET_PXFRAMES`
///    arithmetic — this is what makes the test a *guard* rather than a snapshot: it is red on the
///    shipped-before code by construction, because that code emitted exactly this config;
///  * **the NEW policy**, read from `decode_tiling` under a budget pinned to the OLD policy's measured
///    peak (~20 GiB at 832×480), so "did the fix cost memory?" is answered by the same run.
///
/// It then walks a candidate ladder (temporal-only vs. spatially-relieved) for the record, reporting
/// mean abs err, highlight clipping and MLX active peak per decode.
///
/// ⚠️ **Read this test's SPATIAL rows as a memory measurement, not as quality evidence.** Its source is
/// [`smooth_frame`] — a 5-6-cycle sinusoid with essentially no energy above DC, chosen so the VAE's own
/// round-trip error would not swamp the v2v comparisons this module also runs. That stimulus
/// *structurally cannot* exhibit a spatial seam or a starved spatial receptive field: there is no
/// high-frequency content for either to destroy. The peaks it reports are real (memory does not care
/// what the pixels are) and its temporal rows are diagnostic (the z16 collapse is a low-frequency
/// content failure, which this source does show). But the claim "shrinking the spatial tile is nearly
/// free" is evidenced by [`decode_tiling_sweep_against_single_pass`], which runs the same ladder on
/// **real generated latents** and adds a per-column metric that a whole-frame mean would hide.
#[test]
#[ignore = "real snapshot; run with --ignored on macOS (see module doc)"]
fn decode_policy_matches_single_pass_at_the_old_memory_peak() {
    use mlx_gen::tiling::{SpatialTiling, TemporalTiling, TilingConfig, VaeTiling};
    use mlx_gen_wan::{preprocess_i2v_image, WanVae};
    use mlx_rs::ops::concatenate_axis;

    let root = require_snapshot();
    let w = env_usize("KREA_SMOKE_W", 832);
    let h = env_usize("KREA_SMOKE_H", 480);
    let frames = env_usize("KREA_SMOKE_FRAMES", 33);
    let latent_frames = (frames - 1) / 4 + 1;
    let out_frames = (latent_frames * 4) as i32;
    // ⚠️ A single-pass z16 decode is only a valid reference below the write cap (56 output frames at
    // 832×480). Past it MLX writes silently wrong pixels and every tiled candidate looks catastrophic.
    let write_cap = VaeTiling::WAN.writable_frame_cap(h as i32, w as i32);
    assert!(
        out_frames as i64 <= write_cap,
        "needs a valid single-pass reference: {out_frames} output frames exceeds the z16 write cap \
         {write_cap} at {w}x{h}"
    );

    let vw =
        mlx_gen::weights::Weights::from_file(root.join("vae.safetensors")).expect("open the VAE");
    let vae = WanVae::from_weights(&vw).expect("load the z16 Wan VAE");
    let chw: Vec<mlx_rs::Array> = (0..frames)
        .map(|i| {
            preprocess_i2v_image(&smooth_frame(w, h, i), w as u32, h as u32)
                .expect("preprocess")
                .expand_dims(1)
                .expect("expand")
        })
        .collect();
    let video = concatenate_axis(&chw.iter().collect::<Vec<_>>(), 1)
        .expect("stack")
        .expand_dims(0)
        .expect("batch");
    let z = vae.encode(&video).expect("VAE encode");
    let zs = z.shape().to_vec();
    let latents = z
        .reshape(&[zs[1], zs[2], zs[3], zs[4]])
        .expect("drop batch");

    let decode = |tiling: Option<&TilingConfig>| -> Vec<Image> {
        match decode_latents_to_video(
            &vae,
            &latents,
            24,
            Some(frames),
            tiling,
            &mlx_gen::CancelFlag::default(),
        )
        .expect("VAE decode")
        {
            GenerationOutput::Video { frames, .. } => frames,
            other => panic!("expected a Video output, got {other:?}"),
        }
    };
    // Peak per decode: "does the fix work" and "is the fix affordable" answered by the same run.
    let peak_of = |f: &dyn Fn() -> Vec<Image>| -> (Vec<Image>, usize) {
        mlx_rs::memory::clear_cache();
        mlx_rs::memory::reset_peak_memory();
        let out = f();
        (out, mlx_rs::memory::get_peak_memory())
    };

    /// The removed policy, verbatim: a 3.5e6 output px·frame budget snapped to the ×4 stride, with
    /// the sc-8446 overlap floor. At every bucket ≥ ~233k px/frame this is 8 output frames = 2 latent.
    fn old_policy(out_h: usize, out_w: usize, out_frames: i32) -> TilingConfig {
        const BUDGET_PXFRAMES: i64 = 3_500_000;
        let px_per_frame = (out_h as i64) * (out_w as i64);
        let budget_frames = BUDGET_PXFRAMES / px_per_frame.max(1);
        let write_cap = VaeTiling::WAN.writable_frame_cap(out_h as i32, out_w as i32);
        let win = budget_frames.min(write_cap) as i32;
        let tile_frames = (win / 4 * 4).clamp(8, out_frames.max(8));
        TilingConfig::temporal_only(tile_frames, (tile_frames / 4).max(4))
    }

    let (single, single_peak) = peak_of(&|| decode(None));
    let (single_clip_mean, single_clip_worst) = clip_stats(&single);
    println!(
        "  single-pass (reference): clipping {single_clip_mean:.2}% / {single_clip_worst:.2}%, \
         peak {:.2} GiB",
        gib(single_peak)
    );

    let old = old_policy(h, w, out_frames);
    let ot = old.temporal.expect("the old policy is temporal-only");
    let (old_out, old_peak) = peak_of(&|| decode(Some(&old)));
    let old_err = mean_abs_delta(&single, &old_out);
    let (old_clip_mean, old_clip_worst) = clip_stats(&old_out);
    println!(
        "  OLD policy tile {}/overlap {} (latent {}/{}): mean abs err {old_err:.2}/255, clipping \
         {old_clip_mean:.2}% / {old_clip_worst:.2}%, peak {:.2} GiB",
        ot.tile_frames,
        ot.overlap_frames,
        ot.tile_frames / 4,
        (ot.overlap_frames / 4).min(ot.tile_frames / 4 - 1),
        gib(old_peak)
    );

    // Pin the NEW policy's budget to what the OLD one actually cost, so the comparison is like-for-like
    // on memory and the quality result is not bought with a bigger machine.
    let budget_gib = env_usize("KREA_DECODE_BUDGET_GIB", gib(old_peak).ceil() as usize);
    std::env::set_var("WAN_VAE_BUDGET_GIB", budget_gib.to_string());
    let new = decode_tiling(h, w, out_frames)
        .expect("the new policy must be feasible at the old policy's peak")
        .expect("the clip must still tile at that budget");
    std::env::remove_var("WAN_VAE_BUDGET_GIB");
    let (new_out, new_peak) = peak_of(&|| decode(Some(&new)));
    let new_err = mean_abs_delta(&single, &new_out);
    let (new_clip_mean, new_clip_worst) = clip_stats(&new_out);
    println!(
        "  NEW policy {new:?} (budget pinned {budget_gib} GiB): mean abs err {new_err:.2}/255, \
         clipping {new_clip_mean:.2}% / {new_clip_worst:.2}%, peak {:.2} GiB",
        gib(new_peak)
    );
    dump_frames(&single, "policy_single_pass");
    dump_frames(&old_out, "policy_old");
    dump_frames(&new_out, "policy_new");

    // --- the record: temporal-only vs. spatially-relieved at the SAME latent tile ------------------
    println!("  --- candidate ladder (for the operating-point record) ---");
    let ladder: [(i32, i32, Option<i32>); 8] = [
        (8, 4, None),        // the old policy: latent 2 / 1
        (16, 8, None),       // latent 4 / 2
        (32, 8, None),       // latent 8 / 2, full frame
        (32, 16, None),      // latent 8 / 4, full frame
        (32, 16, Some(448)), // latent 8 / 4, spatially relieved
        (32, 16, Some(320)), // latent 8 / 4, spatially relieved further
        (32, 16, Some(256)), // latent 8 / 4, 32 latent px spatial tiles
        (32, 16, Some(192)), // latent 8 / 4, the SMALLEST spatial candidate (24 latent px)
    ];
    for (tile, overlap, tile_px) in ladder {
        let cfg = TilingConfig {
            spatial: tile_px.map(|tile_px| SpatialTiling {
                tile_px,
                overlap_px: 64,
            }),
            temporal: Some(TemporalTiling {
                tile_frames: tile,
                overlap_frames: overlap,
            }),
        };
        let (out, peak) = peak_of(&|| decode(Some(&cfg)));
        let (cm, cw) = clip_stats(&out);
        println!(
            "    latent tile {}/overlap {} spatial {:?}: mean abs err {:.2}/255, clipping \
             {cm:.2}% / {cw:.2}%, peak {:.2} GiB",
            tile / 4,
            overlap / 4,
            tile_px,
            mean_abs_delta(&single, &out),
            gib(peak)
        );
    }

    // --- gates ------------------------------------------------------------------------------------
    // 1. The harness reproduces the defect: the OLD policy is materially worse than single-pass. If
    //    this ever goes green the measurement stopped measuring, and gate 2 proves nothing.
    assert!(
        old_err > 5.0,
        "the OLD policy now matches single-pass to {old_err:.2}/255 — this guard is no longer \
         reproducing sc-15325 and gate 2 below is vacuous"
    );
    // 2. The NEW policy is at single-pass level, not merely better.
    assert!(
        new_err < 4.0 && new_err < old_err * 0.35,
        "the new decode policy is {new_err:.2}/255 from single-pass (old: {old_err:.2}) — sc-15325 \
         requires approximately single-pass quality at the shipping buckets"
    );
    // 3. ...on the metric a viewer actually sees. The old policy blew out whole frames.
    assert!(
        new_clip_worst <= single_clip_worst + 1.5,
        "the new policy's worst-frame highlight clipping is {new_clip_worst:.2}% against \
         {single_clip_worst:.2}% single-pass — highlights are still blowing out"
    );
    // 4. ...and it did not buy that with memory: the pinned budget must have been honoured.
    assert!(
        gib(new_peak) <= budget_gib as f64 * 1.15,
        "the new policy peaked at {:.2} GiB against a {budget_gib} GiB budget — the selector's cost \
         model is under-predicting and `mlx.minMemoryGb` cannot be trusted",
        gib(new_peak)
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED generate_smoke::decode_policy_matches_single_pass_at_the_old_memory_peak");
}

/// **sc-15325 root cause: decode the SAME real t2v latents single-pass vs. every tiling candidate.**
///
/// This is the experiment the whole diagnosis rests on. It generates a real clip's latents once, then
/// decodes them repeatedly — single-pass (the reference every tiled decode approximates) and across a
/// sweep of `(tile_frames, overlap_frames)` — and reports, per decode: mean |Δ| vs single-pass, the
/// worst per-frame Δ and where it lands, highlight-clipping %, and mean saturation. If the shipped
/// tiling's period-8 artifacts vanish single-pass, the tiled decode is the mechanism; the sweep then
/// says whether the fix is *overlap* (a blending problem) or *tile size* (a temporal receptive-field
/// problem), which are different bugs with different fixes.
///
/// It also carries the **spatial** ladder, because the fix's affordability claim ("relieve the memory
/// on the spatial axis instead") has to be evidenced on the same real latents rather than on a
/// band-limited synthetic source that cannot show a seam. Measured (832×480, 36 output frames, latent
/// tile 8 / overlap 4):
///
/// | spatial tile | mean abs err | per-column mean abs err | clipping mean / worst | MLX active peak |
/// |---|---|---|---|---|
/// | none (full frame) | 1.954 /255 | 1.629 … 2.440 | 0.14 % / 1.05 % | 75.77 GiB |
/// | 448 px | 2.021 | 1.696 … 2.505 | 0.14 % / 1.04 % | 38.57 GiB |
/// | 320 px | 2.054 | 1.719 … 2.531 | 0.13 % / 1.04 % | 20.08 GiB |
/// | 256 px | 2.075 | 1.735 … 2.579 | 0.13 % / 1.04 % | 12.99 GiB |
/// | 192 px | 2.121 | 1.769 … 2.630 | 0.13 % / 1.02 % | 7.49 GiB |
///
/// **10.1× less memory for 0.17/255, and no seam**: the per-column error shifts uniformly rather than
/// spiking at the tile stride (worst column 1.24× the clip mean at the smallest tile). Contrast the
/// temporal axis on the same clip: latent 2 → 4 → 8 is 17.09 → 6.43 → 1.95 /255.
#[test]
#[ignore = "real snapshot; run with --ignored on macOS (see module doc)"]
fn decode_tiling_sweep_against_single_pass() {
    use mlx_gen::tiling::{SpatialTiling, TemporalTiling, TilingConfig};
    use mlx_gen_krea_realtime::{
        generate_latents, load_krea_realtime_transformer_with_quant, ArGenParams,
        CausalKreaTransformer,
    };
    use mlx_gen_wan::{load_tokenizer, Umt5Encoder, WanVae};
    use mlx_rs::Array;

    let root = require_snapshot();
    let w = env_usize("KREA_SMOKE_W", 832);
    let h = env_usize("KREA_SMOKE_H", 480);
    let frames = env_usize("KREA_SMOKE_FRAMES", 33);
    let (latent_h, latent_w) = (h / 8, w / 8);
    let latent_frames = (frames - 1) / 4 + 1;

    let mut cfg = KreaRealtimeConfig::krea_realtime_14b();
    cfg.ar.local_attn_size = cfg.ar.streaming_local_attn_frames() as i64;
    cfg.ar.frame_seq_length = (latent_h / cfg.wan.patch_size.1) * (latent_w / cfg.wan.patch_size.2);
    cfg.ar.seq_length = latent_frames * cfg.ar.frame_seq_length;

    // --- Generate one real clip's latents (the same content the shipped clip shows). ---
    let latents = {
        let tokenizer =
            load_tokenizer(root.join("tokenizer.json"), cfg.wan.text_len).expect("tokenizer");
        let mut tw = mlx_gen::weights::Weights::from_file(root.join("t5_encoder.safetensors"))
            .expect("open the TE");
        let context = {
            let enc = Umt5Encoder::from_weights_quantized(
                &mut tw,
                &cfg.wan,
                mlx_gen_wan::config::WanQuant {
                    bits: 8,
                    group_size: 64,
                },
            )
            .expect("UMT5");
            let c = enc
                .encode(
                    &tokenizer,
                    "a red fox trotting through a snowy pine forest at sunrise, drifting snow, \
                     cinematic, shallow depth of field",
                )
                .expect("encode");
            mlx_rs::transforms::eval([&c]).expect("eval context");
            c
        };
        let dw =
            mlx_gen::weights::Weights::from_file(root.join("dit.safetensors")).expect("open DiT");
        let raw: std::collections::HashMap<String, Array> = dw
            .keys()
            .map(|k| (k.to_string(), dw.get(k).expect("listed key").clone()))
            .collect();
        let (dit, _) = load_krea_realtime_transformer_with_quant(raw, &cfg).expect("load the DiT");
        let transformer = CausalKreaTransformer::new(dit, &cfg);
        let params = ArGenParams {
            seed: 7,
            steps: None,
            num_latent_frames: latent_frames,
            latent_height: latent_h,
            latent_width: latent_w,
            fps: 24,
            memory: Default::default(),
        };
        let l = generate_latents(
            &transformer,
            &cfg,
            &context,
            &params,
            &mlx_gen::CancelFlag::default(),
            &mut |_| {},
        )
        .expect("generate latents");
        mlx_rs::transforms::eval([&l]).expect("materialize latents");
        l
    };
    mlx_rs::memory::clear_cache();

    // --- Decode the SAME latents every way. ---
    let vw =
        mlx_gen::weights::Weights::from_file(root.join("vae.safetensors")).expect("open the VAE");
    let vae = WanVae::from_weights(&vw).expect("load the z16 Wan VAE");
    let decode = |tiling: Option<&TilingConfig>| -> Vec<Image> {
        match decode_latents_to_video(
            &vae,
            &latents,
            24,
            Some(frames),
            tiling,
            &mlx_gen::CancelFlag::default(),
        )
        .expect("VAE decode")
        {
            GenerationOutput::Video { frames, .. } => frames,
            other => panic!("expected a Video output, got {other:?}"),
        }
    };

    // Peak is measured per decode: tile size is the decode's memory bound, so "is the fix
    // affordable" is answered by the same run that answers "does the fix work".
    let peak_of = |f: &dyn Fn() -> Vec<Image>| -> (Vec<Image>, usize) {
        mlx_rs::memory::clear_cache();
        mlx_rs::memory::reset_peak_memory();
        let out = f();
        (out, mlx_rs::memory::get_peak_memory())
    };

    // ⚠️ The single-pass reference is only valid BELOW the z16 write bound. At 832x480 the cap is 56
    // output frames (`96 · f · h · w <= i32::MAX`); an 84-frame clip is 1.5x over it, and MLX writes
    // silently WRONG results past that — a corrupt reference that would make every tiled candidate look
    // catastrophically bad (measured: it collapses saturation 0.33 -> 0.07 and inflates every mean |Δ|
    // to 58-71/255). Refuse to compare against garbage.
    let out_frames_total = (latent_frames * 4) as i64;
    let write_cap = mlx_gen::tiling::VaeTiling::WAN.writable_frame_cap(h as i32, w as i32);
    assert!(
        out_frames_total <= write_cap,
        "this sweep needs a VALID single-pass reference, but {out_frames_total} output frames exceeds \
         the z16 write cap of {write_cap} at {w}x{h} — past it the untiled decode is silently wrong. \
         Re-run with KREA_SMOKE_FRAMES <= {}",
        write_cap - 3
    );

    let (single, single_peak) = peak_of(&|| decode(None));
    dump_frames(&single, "sweep_single_pass");
    println!("--- single-pass (the reference) ---");
    println!("    [single] MLX active peak {:.2} GiB", gib(single_peak));
    report_artifacts(&single, "single");

    // The product policy, pinned to a small budget so the sweep is comparable run-to-run rather than
    // reflecting whatever this host happened to have free.
    std::env::set_var("WAN_VAE_BUDGET_GIB", "20");
    let shipped = decode_tiling(h, w, (latent_frames * 4) as i32)
        .expect("the product policy must be feasible")
        .expect("the clip must tile at a 20 GiB budget");
    std::env::remove_var("WAN_VAE_BUDGET_GIB");
    println!("--- product policy at a 20 GiB budget: {shipped:?} ---");
    // ⚠️ The product plan is NOT necessarily temporal. Since sc-15325 the selector relieves memory on
    // the spatial axis, and at a short clip it will happily return a spatial-only plan — which is the
    // *best* possible answer for this defect (no temporal tiling at all ⇒ the decoder sees the whole
    // sequence). Assuming a temporal tile here is how this test first went red after the fix.
    let ship_t = shipped.temporal;
    let mut candidates: Vec<(i32, i32)> = Vec::new();
    if let Some(t) = ship_t {
        candidates.push((t.tile_frames, t.overlap_frames));
    }
    // (8, 8) is not another experiment: the planner clamps latent overlap to tile - 1,
    // making it the same latent 2/1 plan as (8, 4).
    for c in [(8, 4), (16, 4), (16, 8), (32, 8), (32, 16)] {
        if !candidates.contains(&c) {
            candidates.push(c);
        }
    }

    // The full-frame latent-8/4 decode is the reference the SPATIAL ladder below is compared against
    // (it isolates "what does shrinking the spatial tile cost?" from "what does the temporal tile
    // cost?"), so keep it rather than paying a second 70-GiB-class decode for it.
    let mut full_frame_32_16: Option<(Vec<Image>, usize)> = None;
    for (tile, overlap) in candidates {
        let cfg = TilingConfig {
            spatial: None,
            temporal: Some(TemporalTiling {
                tile_frames: tile,
                overlap_frames: overlap,
            }),
        };
        let (out, peak) = peak_of(&|| decode(Some(&cfg)));
        let d = mean_abs_delta(&single, &out);
        let per: Vec<f64> = single
            .iter()
            .zip(out.iter())
            .map(|(a, b)| {
                a.pixels
                    .iter()
                    .zip(b.pixels.iter())
                    .map(|(&x, &y)| (x as f64 - y as f64).abs())
                    .sum::<f64>()
                    / a.pixels.len() as f64
            })
            .collect();
        let (worst_i, worst) =
            per.iter().enumerate().fold(
                (0usize, 0.0f64),
                |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc },
            );
        let is_product =
            ship_t.is_some_and(|t| (t.tile_frames, t.overlap_frames) == (tile, overlap));
        let label = if is_product {
            format!("tile {tile:>2} / overlap {overlap:>2}  [PRODUCT temporal tile]")
        } else {
            format!("tile {tile:>2} / overlap {overlap:>2}")
        };
        println!(
            "  {label}: latent tile {} / overlap {} -> mean |Δ| {d:.2}/255, worst frame {worst_i} \
             at {worst:.2}",
            tile / 4,
            overlap / 4
        );
        println!(
            "    [t{tile}o{overlap}] MLX active peak {:.2} GiB",
            gib(peak)
        );
        report_artifacts(&out, &format!("t{tile}o{overlap}"));
        if is_product {
            dump_frames(&out, "sweep_product_temporal");
        }
        if (tile, overlap) == (32, 16) {
            dump_frames(&out, "sweep_best");
            full_frame_32_16 = Some((out, peak));
        }
    }

    // --- the SPATIAL ladder, on the same real latents ---------------------------------------------
    //
    // sc-15325's affordability claim ("the memory floor drops ~6-10× and it costs almost nothing")
    // used to be evidenced only by `decode_policy_matches_single_pass_at_the_old_memory_peak`, whose
    // source is `smooth_frame` — a 5-6-cycle sinusoid with essentially zero energy above DC. That
    // stimulus *structurally cannot* show a spatial seam or a starved spatial receptive field, so it
    // could not establish the claim it was cited for. These rows are the real-latent version: same
    // clip, same latents, same latent tile 8 / overlap 4, varying only the spatial tile.
    //
    // Both metrics are reported. The whole-frame mean answers "how much error"; the per-COLUMN mean
    // answers "is any of it concentrated at a tile boundary", which the frame mean averages away.
    let (full_out, full_peak) =
        full_frame_32_16.expect("the (32, 16) full-frame row must have run");
    let full_err = mean_abs_delta(&single, &full_out);
    let full_cols = mean_abs_delta_columns(&single, &full_out);
    let col_span = |c: &[f64]| -> (f64, f64) {
        let lo = c.iter().cloned().fold(f64::MAX, f64::min);
        let hi = c.iter().cloned().fold(0.0f64, f64::max);
        (lo, hi)
    };
    let (full_lo, full_hi) = col_span(&full_cols);
    println!("  --- spatial ladder at latent tile 8 / overlap 4 (real latents) ---");
    println!(
        "    spatial none (full frame): mean |Δ| {full_err:.3}/255, per-column {full_lo:.3}..\
         {full_hi:.3}, peak {:.2} GiB",
        gib(full_peak)
    );
    let mut spatial_rows: Vec<(i32, f64, f64, f64)> = Vec::new(); // (px, err, col_hi, peak_gib)
    for tile_px in [448i32, 320, 256, 192] {
        let cfg = TilingConfig {
            spatial: Some(SpatialTiling {
                tile_px,
                overlap_px: 64,
            }),
            temporal: Some(TemporalTiling {
                tile_frames: 32,
                overlap_frames: 16,
            }),
        };
        let (out, peak) = peak_of(&|| decode(Some(&cfg)));
        let err = mean_abs_delta(&single, &out);
        let cols = mean_abs_delta_columns(&single, &out);
        let (lo, hi) = col_span(&cols);
        let (cm, cw) = clip_stats(&out);
        println!(
            "    spatial {tile_px:>3} px: mean |Δ| {err:.3}/255, per-column {lo:.3}..{hi:.3}, \
             clipping {cm:.2}% / {cw:.2}%, peak {:.2} GiB",
            gib(peak)
        );
        spatial_rows.push((tile_px, err, hi, gib(peak)));
    }

    // Gate 1 — the spatial axis is nearly free. The SMALLEST candidate must stay close to the
    // full-frame decode at the same temporal tile; if shrinking the spatial tile ever starts costing
    // real error, "relieve memory spatially" stops being the right answer and the floor has to become
    // a memory trade again.
    let (small_px, small_err, small_hi, small_peak) = *spatial_rows
        .last()
        .expect("the ladder ran at least one row");
    assert!(
        small_err <= full_err + 1.0,
        "the smallest spatial candidate ({small_px} px) is {small_err:.3}/255 against \
         {full_err:.3}/255 full-frame at the same latent tile — spatial relief is no longer nearly \
         free, so sc-15325's affordability argument needs re-deriving"
    );
    // Gate 2 — and none of that error is a SEAM. A trapezoidally blended spatial tile should raise
    // every column a little, not spike at the stride. Compared against the row's own column floor so
    // this measures concentration, not content.
    assert!(
        small_hi <= small_err * 2.0,
        "the {small_px} px spatial plan has a worst column of {small_hi:.3}/255 against a clip mean \
         of {small_err:.3} — that concentration is the signature of a spatial seam, which the \
         whole-frame mean would have hidden"
    );
    // Gate 3 — and it actually bought the memory it is claimed to buy.
    assert!(
        small_peak < gib(full_peak) * 0.5,
        "the {small_px} px spatial plan peaked at {small_peak:.2} GiB against {:.2} GiB full-frame — \
         sc-15325's claim is that the spatial axis is where the memory comes from",
        gib(full_peak)
    );

    // Gate the full PRODUCT plan against the same generated latents decoded single-pass and
    // with the old temporal window. Keep both the absolute quality ceiling and the relative
    // improvement requirement. The latter rejects selecting the old window as the product plan,
    // even when the content makes the old window's absolute error small.
    //
    // The >5/255 known-corruption control belongs to the fixed-source sibling test
    // `decode_policy_matches_single_pass_at_the_old_memory_peak`. Generated content does not have
    // that floor: rc.2 measured old=4.28 and product=0.46, satisfying BOTH product-quality gates.
    // Also, since sc-19753 the old WINDOW runs on a new decoder partition (dense middle blocks,
    // tiled tail); it is not the historical whole-decoder-per-tile implementation.
    let (product_out, product_peak) = peak_of(&|| decode(Some(&shipped)));
    let d_product = mean_abs_delta(&single, &product_out);
    dump_frames(&product_out, "sweep_product_plan");
    println!(
        "  PRODUCT plan {shipped:?}: mean abs err {d_product:.2}/255, peak {:.2} GiB",
        gib(product_peak)
    );
    report_artifacts(&product_out, "product");

    let old_window = TilingConfig {
        spatial: None,
        temporal: Some(TemporalTiling {
            tile_frames: 8,
            overlap_frames: 4,
        }),
    };
    let d_old = mean_abs_delta(&single, &decode(Some(&old_window)));
    println!("  old temporal window: mean abs err {d_old:.2}/255");
    assert!(
        d_product < 4.0 && d_product < d_old * 0.35,
        "the product decode plan is {d_product:.2}/255 from single-pass (old window: {d_old:.2}) — \
         sc-15325 has regressed"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED generate_smoke::decode_tiling_sweep_against_single_pass"
    );
}

/// **The decisive form of S7 minor #2: is v2v@0 identity in the LATENTS?**
///
/// The pixel test above cannot separate the AR loop from the VAE. This drives
/// [`generate_v2v_latents`] directly on VAE-encoded source latents at `strength = 0` and compares the
/// returned latents to the input — no decode, no encode noise, nothing between the source and the
/// answer. At `strength = 0` every scheduled timestep is 0, so the arithmetic says every Euler step and
/// every renoise is an identity; this is the measurement of whether the implementation agrees.
#[test]
#[ignore = "real snapshot; run with --ignored on macOS (see module doc)"]
fn v2v_strength_zero_is_latent_identity() {
    use mlx_gen_krea_realtime::{
        generate_v2v_latents, load_krea_realtime_transformer_with_quant, ArGenParams,
        CausalKreaTransformer,
    };
    use mlx_gen_wan::{load_tokenizer, preprocess_i2v_image, Umt5Encoder, WanVae};
    use mlx_rs::ops::concatenate_axis;
    use mlx_rs::Array;

    let root = require_snapshot();
    let w = env_usize("KREA_SMOKE_W", 832);
    let h = env_usize("KREA_SMOKE_H", 480);
    let frames = env_usize("KREA_SMOKE_FRAMES", 33);
    let (latent_h, latent_w) = (h / 8, w / 8);
    let latent_frames = (frames - 1) / 4 + 1;

    let mut cfg = KreaRealtimeConfig::krea_realtime_14b();
    cfg.ar.local_attn_size = cfg.ar.streaming_local_attn_frames() as i64;
    cfg.ar.frame_seq_length = (latent_h / cfg.wan.patch_size.1) * (latent_w / cfg.wan.patch_size.2);
    cfg.ar.seq_length = latent_frames * cfg.ar.frame_seq_length;

    // Source latents, via the same VAE the pipeline uses.
    let vw =
        mlx_gen::weights::Weights::from_file(root.join("vae.safetensors")).expect("open the VAE");
    let vae = WanVae::from_weights(&vw).expect("load the z16 Wan VAE");
    let chw: Vec<Array> = (0..frames)
        .map(|i| {
            preprocess_i2v_image(&smooth_frame(w, h, i), w as u32, h as u32)
                .expect("preprocess")
                .expand_dims(1)
                .expect("expand")
        })
        .collect();
    let video = concatenate_axis(&chw.iter().collect::<Vec<_>>(), 1)
        .expect("stack")
        .expand_dims(0)
        .expect("batch");
    let z = vae.encode(&video).expect("VAE encode");
    let zs = z.shape().to_vec();
    let source_latents = z
        .reshape(&[zs[1], zs[2], zs[3], zs[4]])
        .expect("drop batch");
    drop(vae);

    // A real prompt context (the cross-attention path must be exercised, not zeroed).
    let tokenizer =
        load_tokenizer(root.join("tokenizer.json"), cfg.wan.text_len).expect("tokenizer");
    let mut tw = mlx_gen::weights::Weights::from_file(root.join("t5_encoder.safetensors"))
        .expect("open the TE");
    let context = {
        let enc = Umt5Encoder::from_weights_quantized(
            &mut tw,
            &cfg.wan,
            mlx_gen_wan::config::WanQuant {
                bits: 8,
                group_size: 64,
            },
        )
        .expect("UMT5");
        let c = enc
            .encode(&tokenizer, "a paper crane on a desk")
            .expect("encode");
        mlx_rs::transforms::eval([&c]).expect("eval context");
        c
    };

    let dw = mlx_gen::weights::Weights::from_file(root.join("dit.safetensors")).expect("open DiT");
    let raw: std::collections::HashMap<String, Array> = dw
        .keys()
        .map(|k| (k.to_string(), dw.get(k).expect("listed key").clone()))
        .collect();
    let (dit, _) = load_krea_realtime_transformer_with_quant(raw, &cfg).expect("load the DiT");
    let transformer = CausalKreaTransformer::new(dit, &cfg);

    let params = ArGenParams {
        seed: 3,
        steps: None,
        num_latent_frames: latent_frames,
        latent_height: latent_h,
        latent_width: latent_w,
        fps: 24,
        memory: Default::default(),
    };
    let out = generate_v2v_latents(
        &transformer,
        &cfg,
        &context,
        &params,
        &source_latents,
        0.0,
        &mlx_gen::CancelFlag::default(),
        &mut |_| {},
    )
    .expect("v2v at strength 0");

    mlx_rs::transforms::eval([&out, &source_latents]).expect("materialize");
    let diff = mlx_rs::ops::subtract(&out, &source_latents)
        .expect("diff")
        .abs()
        .expect("abs");
    let max_abs = diff.max(None).expect("max").item::<f32>();
    let mean_abs = diff.mean(None).expect("mean").item::<f32>();
    let src_scale = source_latents
        .abs()
        .expect("abs")
        .mean(None)
        .expect("mean")
        .item::<f32>();
    println!(
        "  v2v@0 latents vs source: mean |Δ| = {mean_abs:.5}, max |Δ| = {max_abs:.5}, \
         source mean |x| = {src_scale:.5}  ({:.2}% of the source scale)",
        100.0 * mean_abs / src_scale
    );
    assert!(
        src_scale > 0.0,
        "the source latents are all zero — the comparison is inert"
    );
    assert!(
        mean_abs < 0.02 * src_scale,
        "v2v at strength=0 changed the latents by {mean_abs:.5} ({:.2}% of the source scale) — the \
         zero-strength schedule is NOT an identity on the latents",
        100.0 * mean_abs / src_scale
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED generate_smoke::v2v_strength_zero_is_latent_identity");
}

/// **Measured KV-cache residency at the production geometry, per storage tier.** The self-attention
/// KV cache holds post-RoPE **activations**, so the *weight* tier does not shrink it: a Q4 DiT still
/// carries a bf16 cache, which is the whole point of measuring it next to the Q4 weights. What *does*
/// shrink it is [`KreaArConfig::kv_cache_quant`](mlx_gen_krea_realtime::KreaArConfig::kv_cache_quant)
/// (sc-17807), so this drives the real DiT chunk-by-chunk **once per tier** — the shipped bf16 cache
/// and Q8 — over the same bounded window the pipeline uses. sc-17894 trims the cache to the actual
/// next read before measuring: at the shipped geometry that is one 4,680-token chunk, not the
/// 9,360-token attention window (which includes the next chunk's own K/V).
///
/// The gate is the one that makes the published per-token table evidence rather than arithmetic:
/// measured `retained_bytes` must equal
/// [`kv_bytes_per_token`](mlx_gen_krea_realtime::KreaRealtimeConfig::kv_bytes_per_token) × retained
/// tokens **exactly**, on real weights, for both tiers. A table that drifted from the allocator, or a
/// tier that silently stored dense, fails here.
#[test]
#[ignore = "real snapshot; run with --ignored on macOS (see module doc)"]
fn kv_cache_residency_at_the_production_geometry() {
    use mlx_gen_krea_realtime::{
        load_krea_realtime_transformer_with_quant, CausalKreaTransformer, CausalKvCache,
    };
    use mlx_rs::Array;

    let root = require_snapshot();
    let w = env_usize("KREA_SMOKE_W", 832);
    let h = env_usize("KREA_SMOKE_H", 480);
    let frames = env_usize("KREA_SMOKE_FRAMES", 81);

    let (latent_h, latent_w) = (h / 8, w / 8);
    let latent_frames = (frames - 1) / 4 + 1;

    let mut cfg = KreaRealtimeConfig::krea_realtime_14b();
    cfg.ar.local_attn_size = cfg.ar.streaming_local_attn_frames() as i64;
    cfg.ar.frame_seq_length = (latent_h / cfg.wan.patch_size.1) * (latent_w / cfg.wan.patch_size.2);
    cfg.ar.seq_length = latent_frames * cfg.ar.frame_seq_length;

    // Load the real DiT through the product load path.
    let weights =
        mlx_gen::weights::Weights::from_file(root.join("dit.safetensors")).expect("open the DiT");
    let raw: std::collections::HashMap<String, Array> = weights
        .keys()
        .map(|k| (k.to_string(), weights.get(k).expect("listed key").clone()))
        .collect();
    let (dit, packed) =
        load_krea_realtime_transformer_with_quant(raw, &cfg).expect("load the Krea DiT");
    println!("  DiT tier on disk: {packed:?}");
    let transformer = CausalKreaTransformer::new(dit, &cfg);

    // A zero context is enough: this measures cache growth, not image quality.
    let ctx = Array::zeros::<f32>(&[cfg.wan.text_len as i32, cfg.wan.text_dim as i32])
        .expect("zero context");
    let embedded = transformer
        .inner()
        .embed_text(&ctx)
        .expect("embed the text context");
    let cross_kv = transformer
        .prepare_cross_kv(&embedded)
        .expect("cross-attention cache");
    // MLX safetensors loads are lazy, so weights are not resident until something reads them. This
    // evaluation touches only the text embedding and every block's `cross_attn.{k,v}` — **2 of the 10
    // per-block Linears and none of the FFN, roughly 15% of the DiT's bytes** (~1.2 of 7.8 GiB at Q4).
    // So the figure below is a PARTIAL-staging reading, not the DiT's footprint; it is reported as such
    // and the full residency is the per-chunk `MLX active` column, which is measured after a real
    // forward has pulled everything in.
    {
        let mut staged: Vec<&Array> = vec![&embedded];
        for (k, v) in &cross_kv {
            staged.push(k);
            staged.push(v);
        }
        mlx_rs::transforms::eval(staged).expect("stage the DiT");
    }
    let partially_staged = mlx_rs::memory::get_active_memory();

    let fpb = cfg.ar.num_frames_per_block as i32;
    let chunks = latent_frames.div_ceil(cfg.ar.num_frames_per_block);
    // The bound is structural: cached history is the attention window minus the next chunk, plus a
    // disjoint always-on sink prefix (zero in this production row).
    let max_tokens = cfg
        .ar
        .max_attention_size()
        .saturating_sub(cfg.ar.block_size())
        + cfg.ar.sink_tokens();
    let mut peak_by_tier: Vec<(Option<KvCacheQuant>, usize, usize)> = Vec::new();

    for tier in [None, Some(KvCacheQuant::Q8)] {
        let mut tier_cfg = cfg.clone();
        tier_cfg.ar.kv_cache_quant = tier;
        let per_token = tier_cfg
            .kv_bytes_per_token()
            .expect("the tier must be expressible at this backbone's geometry");
        // Built from the config rather than `transformer.new_cache()` so both tiers run against the
        // one loaded DiT (the transformer bakes its tier at construction).
        let mut cache = CausalKvCache::new(
            tier_cfg.wan.num_layers,
            tier_cfg.ar.max_attention_size(),
            tier_cfg.ar.sink_tokens(),
            tier,
        );
        let label = match tier {
            None => "bf16".to_string(),
            Some(q) => format!("q{}/g{}", q.bits, q.group_size),
        };
        println!("  --- KV tier {label} ({per_token} bytes/token) ---");

        let mut start = 0usize;
        let mut peak_retained_bytes = 0usize;
        // The ALLOCATOR's peak, not the cache's own accounting. `retained_bytes` proves the packed
        // cache is smaller; only this proves the packed path is smaller *in practice* — see the
        // assertion after the loop for the mechanism it guards.
        mlx_rs::memory::reset_peak_memory();
        for c in 0..chunks {
            let chunk = Array::zeros::<f32>(&[
                cfg.wan.in_dim as i32,
                fpb,
                latent_h as i32,
                latent_w as i32,
            ])
            .expect("chunk");
            let velocity = transformer
                .forward_chunk(&chunk, 500.0, &cross_kv, start, &mut cache)
                .expect("causal chunk forward");
            start += (fpb as usize) * cfg.ar.frame_seq_length;
            // Supply the ACTUAL next chunk length before measuring. This is the sc-17894 mechanism:
            // `window_prev` evicts exactly the tokens that read cannot consume, and doing it here
            // also proves the packed and dense caches expose the same retained shape.
            cache
                .window_prev(cfg.ar.block_size())
                .expect("trim to the next read");
            // MLX is LAZY: without this `eval` the whole chunk forward is an unexecuted graph, the
            // cache arrays are unmaterialized, and `get_active_memory()` reads ~0 — the measurement
            // would be a shape calculation dressed up as a memory reading. Forcing the cache (and the
            // velocity that depends on the same graph) makes the reported bytes and the allocator's
            // active figure both real, and makes the run take actual GPU time.
            //
            // `eval_retained` materializes the cache **as stored** and `retained_bytes` reports that
            // same representation, so a quantized cache (sc-17807) is measured at its packed cost.
            // Evaluating `layer_kv` and summing its `nbytes` instead would dequantize first and
            // report every tier at the bf16 size — blind to the thing the knob changes.
            mlx_rs::transforms::eval([&velocity]).expect("materialize the chunk forward");
            cache.eval_retained().expect("materialize the KV cache");
            let bytes = cache.retained_bytes();
            peak_retained_bytes = peak_retained_bytes.max(bytes);
            println!(
                "  chunk {c:>2}: committed {} tok, next-read retained {} tok, KV {:.2} GiB, MLX active {:.2} GiB",
                cache.stored_tokens(),
                cache.retained_tokens(),
                gib(bytes),
                gib(mlx_rs::memory::get_active_memory()),
            );
            // The published per-token cost, checked against the allocator on real weights every
            // chunk — including the chunks before eviction starts and the ones after it.
            assert_eq!(
                bytes,
                cache.retained_tokens() * per_token,
                "{label}: {} retained tokens should cost {per_token} bytes each",
                cache.retained_tokens()
            );
        }

        println!(
            "  KV-cache residency at {w}x{h} ({} tok/frame, window {} frames, KV {label}): \
             {:.2} GiB (MLX active after PARTIAL staging -- text embed + cross-attn k/v only, ~15% \
             of the DiT: {:.2} GiB)",
            cfg.ar.frame_seq_length,
            cfg.ar.streaming_local_attn_frames(),
            gib(peak_retained_bytes),
            gib(partially_staged),
        );
        assert!(
            peak_retained_bytes > 0,
            "{label}: the KV cache never retained anything — the measurement is inert"
        );
        assert!(
            cache.retained_tokens() <= max_tokens,
            "{label}: retained {} tokens > the actual next read's {max_tokens} — eviction is not \
             trimming the cache",
            cache.retained_tokens()
        );
        let mlx_peak = mlx_rs::memory::get_peak_memory();
        println!(
            "  {label}: MLX active PEAK across the chunk loop {:.2} GiB",
            gib(mlx_peak)
        );
        peak_by_tier.push((tier, peak_retained_bytes, mlx_peak));
        mlx_rs::memory::clear_cache();
    }

    let (dense_kv, dense_peak) = (peak_by_tier[0].1, peak_by_tier[0].2);
    let (q8_kv, q8_peak) = (peak_by_tier[1].1, peak_by_tier[1].2);
    println!(
        "  Q8 vs bf16 — retained KV {:.2}/{:.2} GiB, MLX active peak {:.2}/{:.2} GiB",
        gib(q8_kv),
        gib(dense_kv),
        gib(q8_peak),
        gib(dense_peak),
    );
    let old_dense_window = cfg.ar.max_attention_size() * cfg.kv_bytes_per_token().unwrap();
    assert_eq!(
        dense_kv * 2,
        old_dense_window,
        "the shipped 6-frame window should retain only the 3-frame cached history the next read uses"
    );
    assert!(
        ((q8_kv as f64 / old_dense_window as f64) - 0.265625).abs() < 1e-9,
        "Q8 plus next-read eviction should compose to 0.265625x of the old bf16 window"
    );

    // **The assertion the prose needed.** `causal.rs` argues that the dequantized read window is an
    // anonymous graph intermediate MLX frees as each layer's attention completes, so it does not
    // land in peak. That argument is sound against the pinned MLX (the tape detaches each node after
    // it evaluates, and `denoise_chunk_inner` drops the window `Vec` before anything evals) — but it
    // is a *premise*, and if it ever fails the quantized path holds all `num_layers` dequantized
    // windows at once and peaks at ~1.53x the dense cache, i.e. WORSE than not quantizing. Nothing
    // else in the tree would notice. This does.
    assert!(
        q8_peak < dense_peak,
        "the Q8 cache peaked at {:.2} GiB against bf16's {:.2} — quantizing did not lower the \
         allocator's peak, which means the dequantized read window is no longer being freed per \
         layer and the packed path now costs MORE than the dense one",
        gib(q8_peak),
        gib(dense_peak),
    );
    // ...and the saving is of the size the per-token table predicts, not a rounding accident. The
    // retained window is identical between tiers, so the whole KV delta should reach the peak.
    let kv_saving = (dense_kv - q8_kv) as f64;
    let peak_saving = dense_peak.saturating_sub(q8_peak) as f64;
    assert!(
        peak_saving > kv_saving * 0.8,
        "the KV cache shrank by {:.2} GiB but the allocator's peak only fell by {:.2} — most of \
         the saving is being spent somewhere else",
        gib(dense_kv - q8_kv),
        gib(dense_peak.saturating_sub(q8_peak)),
    );
}

/// **sc-15127 (S18) — does a long batch clip drift as the bounded KV window rolls, and does a
/// first-chunk sink anchor fix it?**
///
/// Generates the clip once per (row, seed) — varying only the KV read window, the sink anchor and the
/// seed — scores each with [`score_clip`], and prints the table the verdict rests on. Each row's MLX
/// peak is reported too, so the sink's KV cost is measured rather than asserted: a sink is permanently
/// resident, and the bounded window exists precisely because KV is expensive on this host.
///
/// **Rows.** `A` shipped (window 6, sink 0) · `B` sink 1 · `C` sink 3 · `D` wide window 15 · `F`
/// wider bounded window 30 (together A/D/F form the within-regime dose ladder) · `E` the checkpoint's
/// global window (an out-of-regime *reference*, not a floor, and **not attribution evidence** —
/// different attention mask, `n = 1`, no variance estimate) · `Z` the within-regime zero-eviction
/// *rate* row (shipped window, clip short enough never to roll).
///
/// Row Z was intended as a rate floor and **turned out not to be one** — see
/// [`S18Sweep::rate_floor_clause`] for the measured reason. It is still run, still recorded and still
/// reported in the verdict, because "the intended floor does not work, and here is the number that
/// shows it" is the honest form of that result.
///
/// **Driving it.** `KREA_S18_ROWS` (default `ABCDFEZ`) selects rows; `KREA_S18_SEEDS` (comma-separated,
/// default `7`) selects seeds. Each (row, seed) prints one `S18CELL` TSV line, so a long sweep can be
/// run in pieces and re-aggregated without holding a five-hour process open.
///
/// ⚠️ **Geometry — the global reference is the expensive row.** Row E runs the checkpoint's *global*
/// window, whose KV is `latent_frames × frame_seq_length` tokens at **800 KiB/token** (`2 (K and V) ×
/// 40 layers × 5120 dim × 2 bytes` = 819,200 bytes —
/// [`KreaRealtimeConfig::kv_bytes_per_token`](mlx_gen_krea_realtime::KreaRealtimeConfig::kv_bytes_per_token),
/// measured against the allocator by [`kv_cache_residency_at_the_production_geometry`]). At the
/// 832×480 reference bucket a 45-latent-frame clip is 70,200 tokens ≈ **53.6 GiB** of KV before
/// activations — exactly the ~27 GB-of-KV problem
/// [`mac_ar_config`](mlx_gen_krea_realtime::mac_ar_config) exists to dodge, so its cost is a finding
/// rather than a harness bug.
///
/// **Corrected in sc-17807.** This doc priced the cache at 546 kB per DiT token and the clip at
/// 38 GiB — both 1.5× low, which made row E look like it ought to fit a 128 GiB host with room to
/// spare. It did not: the
/// recorded run measured a 63.32 GiB MLX peak and sc-15571 recorded it swapping a host's boot volume
/// full. The corrected figure is what
/// [`the_kv_per_token_figure_in_crate_prose_is_the_computed_one`] gates.
///
/// **Row E is UNMEASURABLE on current infrastructure — decided in sc-17324. Do not keep retrying it.**
///
/// The older claim here was that row E *SIGKILLs* a 128 GiB host at 832×480 (sc-15127: jetsam at
/// step 39/75). That is too strong: CI run 30787887176 (2026-08-03, `nax-macos`, 128 GiB) ran it to
/// completion at 832×480 — `S18CELL E 7 832x480 45 0 55.7356 ...`, a **63.32 GiB** MLX active peak,
/// ~17.5 min. But it fit that box by roughly 0.3 GiB, and sc-15571 records it driving a host into
/// enough swap to fill the boot volume even at 640×384. On `nax-macos-2` (~101 GiB, where the
/// `rw-krea` lane actually runs) it fits at neither bucket.
///
/// So row E is neither impossible nor fine — it is a row with no reproducible home, which is worth
/// naming once rather than rediscovering. `scripts/ci/s18_memory_preflight.py` refuses it by default
/// on both hosts and `krea_s18_rows` no longer includes it. **Nothing downstream depends on it:**
/// [`S18Sweep::verdict`] already declines to use row E as attribution evidence (out of regime,
/// different attention mask, n = 1, no variance estimate), so a sweep without it loses a reference,
/// not a conclusion.
///
/// Row **F** is a different case with a real answer: 46,800 tokens ≈ 49 GiB at 832×480 took
/// `nax-macos-2` down twice (runs 30948568453 and 31051413212, both inside row F seed 7), but at
/// 640×384 it needs ~34 GiB and fits — the bucket the recorded sc-15127 sweep already used for it.
/// The bounded rows run at both buckets, and **both buckets are recorded** — see
/// [`the_recorded_s18_sweep_is_what_the_docs_claim`].
///
/// ⚠️ **Row F is a whole-host risk on a *shared* machine, not only on a CI runner (sc-17807).** The
/// preflight's budget subtracts a fixed 60 GiB of non-MLX overhead, calibrated on a dedicated
/// runner. A dev Mac with several agent worktrees building concurrently is not that box, and its
/// real overhead is not knowable in advance. Running rows A/D/F unattended on one — after checking
/// `uptime`, which reports load and says nothing about memory — OOM-crashed it, at row A's
/// AR→VAE-decode transition where the footprint peaks (the AR peak plus the pinned 20 GiB decode
/// budget). Check free memory (`vm_stat`, free + inactive), require roughly 2× the predicted peak,
/// and re-check before every cell rather than once at the start.
///
/// **The sc-17807 KV-tier A/B therefore ran rows A and D only**, and that costs the comparison
/// nothing structural: [`S18Sweep::kv_tier_comparison`] compares whatever rows are present in both
/// arms, and [`S18Sweep::validate_window_dose_ladder`] returns early without F. What it does cost is
/// the widest *dose* — the tier's saving grows with the window, so row F is where quantizing helps
/// most, and it is exactly the row the A/B has no local measurement for. With sc-17894's exact-read
/// retention its q8 prediction is ~17.1 GiB of KV against 32.1 bf16, i.e. the row that killed
/// `nax-macos-2` becoming affordable on it. That stays a **prediction** until someone runs it on a
/// host that can hold it.
///
/// ⚠️ Must be run on a tree that has sc-15325 (the tiled-decode fix). Before it, the decode injected an
/// 8-output-frame-period artifact that a drift metric reads as AR drift; an earlier attempt at this
/// exact measurement drew the wrong conclusion from it.
#[test]
#[ignore = "real snapshot, hours of GPU; run with --ignored on macOS (see module doc)"]
fn long_clip_coherence_under_the_bounded_window() {
    use mlx_gen_krea_realtime::{
        generate_latents, load_krea_realtime_transformer_with_quant, ArGenParams,
        CausalKreaTransformer,
    };
    use mlx_gen_wan::{load_tokenizer, Umt5Encoder, WanVae};
    use mlx_rs::Array;

    let root = require_snapshot();
    let w = env_usize("KREA_SMOKE_W", 832);
    let h = env_usize("KREA_SMOKE_H", 480);
    let long_lat = env_usize("KREA_S18_LATENT_FRAMES", 45);
    let seeds: Vec<u64> = std::env::var("KREA_S18_SEEDS")
        .unwrap_or_else(|_| "7".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    assert!(!seeds.is_empty(), "KREA_S18_SEEDS parsed to nothing");
    let want_rows = std::env::var("KREA_S18_ROWS").unwrap_or_else(|_| "ABCDFEZ".to_string());
    let (latent_h, latent_w) = (h / 8, w / 8);
    // sc-17807 — the KV cache storage tier. Unset is the shipped bf16 cache, so an unchanged
    // dispatch measures exactly what it measured before. Setting it re-runs the same rows against a
    // quantized cache, which is a SEPARATE arm: it is recorded on its own and compared against the
    // bf16 baseline by `s18_kv_tier_ab_from_accumulated_cells`, never pooled with it.
    //
    // Parsed strictly, and NOT through `env_opt_usize`: that swallows a malformed value into `None`,
    // which here means "silently measure the bf16 arm while the operator believes they dispatched
    // the quantized one" — an hours-long run whose evidence is mislabelled and whose A/B compares a
    // tier against itself. Empty is the one accepted non-numeric form, because that is how a shell
    // spells "unset this for one invocation" (`env KREA_S18_KV_BITS= ...`).
    let kv_quant = parse_kv_tier_env().unwrap_or_else(|e| panic!("{e}"));
    let kv_tier = kv_tier_label(kv_quant);

    let base = KreaRealtimeConfig::krea_realtime_14b();
    let fpb = base.ar.num_frames_per_block; // 3 latent frames per AR chunk
    let shipped_window = base.ar.streaming_local_attn_frames() as i64; // 6 latent frames
    let wide_window = shipped_window * 2 + 3; // 15 latent frames — middle dose (10 rolls at 45f)
    let wider_bounded_window = shipped_window * 5; // 30 latent frames — high dose (5 rolls at 45f)
                                                   // Row Z: the longest clip the SHIPPED window never evicts on. Chunk `c` commits `fpb*(c+1)` latent
                                                   // frames and evicts once that exceeds the window, so `shipped_window` latent frames is exactly it.
    let zero_roll_lat = shipped_window as usize;

    // Every long row is scored over the SAME split, fixed by the shipped window's geometry: the
    // baseline is the output frames the *shipped* configuration produces before its first eviction, and
    // the trend is measured over everything after. A per-row split would make the rows incomparable.
    let pre_len = shipped_window as usize * 4;
    let bucket = fpb * 4; // one AR chunk == one window roll == 12 output frames
                          // Row Z's whole clip is `pre_len` frames long, so it needs its own (shorter) baseline. Its trend is
                          // therefore NOT comparable with the long rows — only its slope is, which is what it is used for.
    let z_pre_len = fpb * 4;

    let all_runs = [
        DriftRun {
            row: 'A',
            label: "A shipped    (window  6, sink 0)",
            window: shipped_window,
            sink: 0,
            latent_frames: None,
        },
        DriftRun {
            row: 'B',
            label: "B anchor 1f  (window  6, sink 1)",
            window: shipped_window,
            sink: 1,
            latent_frames: None,
        },
        DriftRun {
            row: 'C',
            label: "C anchor 3f  (window  6, sink 3)",
            window: shipped_window,
            sink: fpb,
            latent_frames: None,
        },
        DriftRun {
            row: 'D',
            label: "D wide win   (window 15, sink 0)",
            window: wide_window,
            sink: 0,
            latent_frames: None,
        },
        DriftRun {
            row: 'F',
            label: "F wider bound (window 30, sink 0)",
            window: wider_bounded_window,
            sink: 0,
            latent_frames: None,
        },
        DriftRun {
            row: 'E',
            label: "E global ref (global window, sink 0)",
            window: -1,
            sink: 0,
            latent_frames: None,
        },
        DriftRun {
            row: 'Z',
            label: "Z zero-roll  (window  6, sink 0, short)",
            window: shipped_window,
            sink: 0,
            latent_frames: Some(zero_roll_lat),
        },
    ];
    let runs: Vec<&DriftRun> = all_runs
        .iter()
        .filter(|r| want_rows.contains(r.row))
        .collect();
    assert!(
        !runs.is_empty(),
        "KREA_S18_ROWS `{want_rows}` selected no rows"
    );

    // --- Encode the prompt ONCE, then drop the ~11 GB text encoder before any DiT is resident. ---
    let prompt = "a red fox trotting through a snowy pine forest at sunrise, drifting snow, \
                  cinematic, shallow depth of field";
    let context = {
        let tokenizer =
            load_tokenizer(root.join("tokenizer.json"), base.wan.text_len).expect("tokenizer");
        let mut tw = mlx_gen::weights::Weights::from_file(root.join("t5_encoder.safetensors"))
            .expect("open the TE");
        let enc = Umt5Encoder::from_weights_quantized(
            &mut tw,
            &base.wan,
            mlx_gen_wan::config::WanQuant {
                bits: 8,
                group_size: 64,
            },
        )
        .expect("UMT5");
        let c = enc.encode(&tokenizer, prompt).expect("encode");
        // MLX is lazy: without this the context is an unexecuted graph held across the TE's drop.
        mlx_rs::transforms::eval([&c]).expect("eval context");
        c
    };
    mlx_rs::memory::clear_cache();

    let vw =
        mlx_gen::weights::Weights::from_file(root.join("vae.safetensors")).expect("open the VAE");
    let vae = WanVae::from_weights(&vw).expect("load the z16 Wan VAE");

    // Pin the decode budget so every row decodes through the same plan (the comparison is between AR
    // configs; a host-dependent decode plan would be a second free variable).
    std::env::set_var("WAN_VAE_BUDGET_GIB", "20");

    let mut results: Vec<S18Row> = Vec::new();
    for r in &runs {
        for &seed in &seeds {
            let lat = r.latent_frames.unwrap_or(long_lat);
            let row_pre_len = if r.latent_frames.is_some() {
                z_pre_len
            } else {
                pre_len
            };
            let mut cfg = base.clone();
            cfg.ar.local_attn_size = r.window;
            cfg.ar.sink_size = r.sink;
            cfg.ar.kv_cache_quant = kv_quant;
            cfg.ar.frame_seq_length =
                (latent_h / cfg.wan.patch_size.1) * (latent_w / cfg.wan.patch_size.2);
            cfg.ar.seq_length = lat * cfg.ar.frame_seq_length;

            let chunks = lat.div_ceil(fpb);
            // Chunk `c` commits latent frames up to `fpb*(c+1)`; it evicts once that exceeds the
            // window. A global window (`local_attn_size < 0`) never evicts.
            let rolls = eviction_rolls(lat, fpb, r.window);

            println!(
                "=== {} seed {seed} | {lat} latent frames, {chunks} chunks, {rolls} evicting, \
                 baseline {row_pre_len} output frames, window {} tok, KV {kv_tier} ({} B/token)",
                r.label,
                cfg.ar.max_attention_size(),
                cfg.kv_bytes_per_token()
                    .expect("the KV tier must fit this backbone")
            );

            mlx_rs::memory::clear_cache();
            mlx_rs::memory::reset_peak_memory();
            let t0 = Instant::now();
            let latents = {
                let dw = mlx_gen::weights::Weights::from_file(root.join("dit.safetensors"))
                    .expect("DiT");
                let raw: std::collections::HashMap<String, Array> = dw
                    .keys()
                    .map(|k| (k.to_string(), dw.get(k).expect("listed key").clone()))
                    .collect();
                let (dit, _) =
                    load_krea_realtime_transformer_with_quant(raw, &cfg).expect("load the DiT");
                let transformer = CausalKreaTransformer::new(dit, &cfg);
                let params = ArGenParams {
                    seed,
                    steps: None,
                    num_latent_frames: lat,
                    latent_height: latent_h,
                    latent_width: latent_w,
                    fps: 24,
                    memory: Default::default(),
                };
                // Print a per-step mark. This sweep is long enough that a silent run is
                // indistinguishable from a hung one, and the per-step cadence is also how a
                // memory-contended row (the global reference especially) announces itself.
                let mut tstep = Instant::now();
                let l = generate_latents(
                    &transformer,
                    &cfg,
                    &context,
                    &params,
                    &mlx_gen::CancelFlag::default(),
                    &mut |p| {
                        if let Progress::Step { current, total } = p {
                            let now = Instant::now();
                            println!(
                                "    step {current:>3}/{total} (+{:.1?}, {:.1?} elapsed)",
                                now.duration_since(tstep),
                                t0.elapsed()
                            );
                            tstep = now;
                        }
                    },
                )
                .expect("generate latents");
                // MLX is lazy — without this the AR loop has not actually run and the peak below is a
                // shape calculation. Three measurements in this epic read ~0 for exactly this reason.
                mlx_rs::transforms::eval([&l]).expect("materialize latents");
                l
            };
            let ar_peak = mlx_rs::memory::get_peak_memory();
            let ar_wall = t0.elapsed();
            mlx_rs::memory::clear_cache();

            let out_frames = lat * 4;
            let tiling =
                decode_tiling(h, w, out_frames as i32).expect("decode policy must be feasible");
            let frames = match decode_latents_to_video(
                &vae,
                &latents,
                24,
                None,
                tiling.as_ref(),
                &mlx_gen::CancelFlag::default(),
            )
            .expect("VAE decode")
            {
                GenerationOutput::Video { frames, .. } => frames,
                other => panic!("expected a Video output, got {other:?}"),
            };
            drop(latents);
            mlx_rs::memory::clear_cache();

            assert_eq!(frames.len(), out_frames, "{}: decoded frame count", r.label);
            println!(
                "  AR wall {:.1?} ({:.1?}/chunk), MLX active peak {:.2} GiB, decode plan {tiling:?}",
                ar_wall,
                ar_wall / chunks as u32,
                gib(ar_peak)
            );
            report_artifacts(&frames, r.label);
            let d = score_clip(&frames, row_pre_len, bucket);
            print_drift(r.label, &d);
            dump_frames(
                &frames,
                &format!("s18_{}_w{}_sink{}_{lat}_s{seed}", r.row, r.window, r.sink),
            );

            // Two INDEPENDENT cross-checks, because "the drift statistic is small" is not by itself a
            // quality claim:
            //   * highlight clipping — what a viewer actually sees blow out, and a metric with no
            //     construction in common with `score_clip`.
            //   * tail motion — a clip that has stopped moving also has no trend. Freeze and drift are
            //     the two AR failure modes, and a gate that only sees one can be passed by the other.
            let (clip_mean, _) = clip_stats(&frames);
            let deltas: Vec<f64> = frame_stats(&frames)
                .iter()
                .skip(1)
                .map(|s| s.delta)
                .collect();
            let third = (deltas.len() / 3).max(1);
            let head_motion = mean_f64(&deltas[..third]);
            let tail_motion = mean_f64(&deltas[deltas.len() - third..]);
            println!(
                "    motion: head {head_motion:.2}/255 -> tail {tail_motion:.2}/255 per frame; \
                 clipping mean {clip_mean:.2}%"
            );
            // Machine-readable, so a sweep split across processes can be re-aggregated.
            println!(
                "S18CELL\t{}\t{seed}\t{w}x{h}\t{lat}\t{rolls}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t\
                 {:.4}\t{:.4}\t{:.4}\t{}\t{kv_tier}\t{}",
                r.row,
                worst_drift(&d),
                worst_trend(&d),
                worst_excursion(&d),
                worst_slope(&d),
                ar_peak,
                clip_mean,
                head_motion,
                tail_motion,
                worst_component(&d),
                ar_wall.as_millis()
            );
            results.push(S18Row {
                label: format!("{} s{seed}", r.label),
                latent_frames: lat,
                drift: worst_drift(&d),
                trend: worst_trend(&d),
                excursion: worst_excursion(&d),
                slope: worst_slope(&d),
                component: worst_component(&d),
                rolls,
                peak: ar_peak,
                clip_mean,
                head_motion,
                tail_motion,
                row: r.row,
                seed,
                ar_wall_ms: ar_wall.as_millis() as u64,
            });
        }
    }
    std::env::remove_var("WAN_VAE_BUDGET_GIB");

    println!("=== sc-15127 S18 summary ({w}x{h}) ==========================================");
    println!(
        "  row                                       rolls   drift   trend   excur  slope/100f  \
         peak GiB  clip%  tail motion  component"
    );
    for r in &results {
        println!(
            "  {:<41} {:>5}  {:>6.2}  {:>6.2}  {:>6.2}  {:>10.3}  {:>8.2}  {:>5.2}  {:>8.2}  {}",
            r.label,
            r.rolls,
            r.drift,
            r.trend,
            r.excursion,
            r.slope,
            gib(r.peak),
            r.clip_mean,
            r.tail_motion,
            r.component
        );
    }

    // FREEZE gate, every row. A frozen tail is the other way a bounded window can ruin a long clip,
    // and it would read as *excellent* on a drift metric — so a no-drift verdict is only meaningful
    // alongside this.
    for r in &results {
        assert!(
            r.tail_motion > 1.0 && r.tail_motion > r.head_motion * 0.2,
            "{}: the clip's tail moves {:.2}/255 per frame against a head of {:.2} — it has frozen, \
             so its low drift score means nothing",
            r.label,
            r.tail_motion,
            r.head_motion
        );
    }
    // CROSS-CHECK gate: highlight clipping, an independent metric, must agree that the shipped bounded
    // window is a healthy operating point. Judged **within regime** against the wider bounded window
    // (row D) — comparing it against the global row would be comparing against a different attention
    // regime, and against an absolute ceiling, so a run where everything clips cannot pass by being
    // uniformly bad.
    const CLIP_CEILING: f64 = 8.0;
    let mean_clip = |row: char| {
        let v: Vec<f64> = results
            .iter()
            .filter(|r| r.row == row)
            .map(|r| r.clip_mean)
            .collect();
        (!v.is_empty()).then(|| mean_f64(&v))
    };
    if let Some(a) = mean_clip('A') {
        assert!(
            a <= CLIP_CEILING,
            "the shipped bounded window clips {a:.2}% of pixels — over the {CLIP_CEILING:.1}% \
             ceiling, so it is not a healthy operating point whatever the drift statistic says"
        );
        if let Some(d) = mean_clip('D') {
            assert!(
                a <= d * 2.0,
                "the shipped window clips {a:.2}% against the wider bounded window's {d:.2}% — the \
                 within-regime dose-response says tightening the window blows highlights, which the \
                 drift statistic must not be allowed to paper over"
            );
        }
    }
    // Row Z's contribution: the within-regime, zero-eviction *rate*, in the only length-comparable
    // statistic there is. This is REPORTED, not gated — see `S18Sweep::rate_floor_clause` for why the
    // measured Z slope turned out not to bound row A's.
    let mean_slope = |row: char| {
        let v: Vec<f64> = results
            .iter()
            .filter(|r| r.row == row)
            .map(|r| r.slope)
            .collect();
        (!v.is_empty()).then(|| mean_f64(&v))
    };
    if let (Some(a), Some(z)) = (mean_slope('A'), mean_slope('Z')) {
        println!(
            "  within-regime rate: zero-eviction row Z {z:.3}/100f vs shipped row A {a:.3}/100f \
             (A/Z {:.2}x) — a ratio at or above 1 would make Z a floor; below 1 it is not one",
            a / z.max(1e-6)
        );
    }

    let sweep = S18Sweep {
        bucket: format!("{w}x{h}"),
        kv: kv_tier,
        cells: results.iter().map(S18Cell::from_measured).collect(),
    };
    println!("  {}", sweep.summary());
    // A verdict needs the shipped row AND the within-regime dose ladder it is attributed against.
    // Row A alone is *resolvable* — `validate_window_dose_ladder` returns early when D or F is
    // missing — so gating on `A` alone would let a one-row dispatch emit a verdict whose attribution
    // clause has no ladder behind it. Piecewise runs are the normal way to drive this sweep now
    // (sc-17655), so the partial branch is the common path, not the exotic one: measure the rows in
    // whatever pieces the runner can afford, then re-aggregate with
    // [`s18_verdict_from_accumulated_cells`].
    // Structural checks run on EVERY dispatch, complete or not — they are about whether the rows
    // measured are what they claim to be, not about whether there are enough of them. A partial
    // dispatch that mis-parameterises row E or Z, or runs too short a clip, must fail here rather
    // than at re-aggregation hours later.
    if let Err(e) = sweep.structural_checks() {
        panic!("{e}");
    }
    let complete = ['A', 'D', 'F'].iter().all(|row| want_rows.contains(*row));
    if complete {
        match sweep.verdict() {
            Ok(v) => println!("  VERDICT: {v}"),
            Err(e) => panic!("{e}"),
        }
    } else {
        println!(
            "  (partial sweep: rows `{want_rows}` are not the whole A/D/F dose ladder, so no \
             verdict is computed — re-aggregate the S18CELL lines with \
             `KREA_S18_CELLS=<file> cargo test -p mlx-gen-krea-realtime --test integration -- \
             generate_smoke::s18_verdict_from_accumulated_cells --exact --ignored --nocapture`)"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED generate_smoke::long_clip_coherence_under_the_bounded_window");
}

/// **Re-aggregate a piecewise S18 sweep and apply the verdict rule to it.**
///
/// [`long_clip_coherence_under_the_bounded_window`] measures every (row, seed) it is asked for and
/// prints one `S18CELL` TSV line each, and its docs have always said those lines exist "so a long
/// sweep can be run in pieces and re-aggregated without holding a five-hour process open". Nothing
/// exposed that re-aggregation until sc-17655: the verdict was computed only inside the measuring
/// process, so the pieces could be measured but never resolved, and the only way to get a verdict
/// was the whole 4.3-hour sweep in one go — which on `nax-macos-2` head-of-line-blocks `rw-audio`,
/// `rw-llm` and `rw-chroma` for half a night (the sc-16981 failure, and the reason sc-17324's own
/// powered run got cancelled).
///
/// This is that entry point. It reads accumulated `S18CELL` lines, rebuilds the [`S18Sweep`], and
/// applies **the same [`S18Sweep::verdict`]** the live run would have — no second copy of the rule.
///
/// ```text
/// cat run-A.tsv run-B.tsv ... > all-cells.tsv
/// KREA_S18_CELLS=$PWD/all-cells.tsv cargo test -p mlx-gen-krea-realtime --test integration \
///   generate_smoke::s18_verdict_from_accumulated_cells -- --exact --ignored --nocapture
/// ```
///
/// `-p` is not optional: two workspace crates carry a `generate_smoke` test target, and without it
/// the scail2 binary also runs, matches nothing under `--exact`, and exits 0 — the "0 tests, still
/// green" shape this file warns about elsewhere. Use an ABSOLUTE path for the file, too: `cargo
/// test` runs the binary with the CRATE root as its working directory, not the workspace root.
///
/// Input is the artifact the sweep job uploads verbatim — `VERDICT:` lines and blanks are ignored,
/// so `s18-cells.tsv` files concatenate without editing. `KREA_S18_BUCKET` selects the geometry when
/// the input mixes buckets; with one bucket present it is inferred. Duplicate (row, seed) cells are
/// NOT silently dropped: a re-measured cell reaches `validate_window_dose_ladder`, which rejects the
/// ladder rather than quietly averaging two runs of the same configuration.
///
/// `#[ignore]` because it is an operator entry point rather than a gate — it asserts nothing about
/// *which* verdict comes out, only that the rule can be applied to evidence that arrived in pieces.
/// The rule's own outcomes stay gated, without weights, by
/// [`the_s18_verdict_rule_distinguishes_its_outcomes`].
#[test]
#[ignore = "operator entry point: re-aggregates S18CELL evidence named by KREA_S18_CELLS"]
fn s18_verdict_from_accumulated_cells() {
    let path = std::env::var("KREA_S18_CELLS").expect(
        "set KREA_S18_CELLS to a file of accumulated S18CELL lines (concatenated `s18-cells.tsv` \
         artifacts are the expected input)",
    );
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read the accumulated cells at `{path}`: {e}"));
    let want_bucket = std::env::var("KREA_S18_BUCKET").ok();
    let sweep = s18_sweep_from_accumulated(&raw, want_bucket.as_deref())
        .unwrap_or_else(|e| panic!("`{path}`: {e}"));

    let mut rows: Vec<char> = sweep.cells.iter().map(|c| c.row).collect();
    rows.sort_unstable();
    rows.dedup();
    println!(
        "re-aggregated {} cells at {} from `{path}` — rows {}",
        sweep.cells.len(),
        sweep.bucket,
        rows.iter().collect::<String>()
    );
    println!("  {}", sweep.summary());
    // The structural guards apply to re-assembled evidence exactly as they do to a live sweep — a
    // mis-parameterised row E or Z is no less wrong for having been measured in a separate job.
    if let Err(e) = sweep.structural_checks() {
        panic!("{e}");
    }
    match sweep.verdict() {
        Ok(v) => println!("  VERDICT: {v}"),
        Err(e) => panic!("{e}"),
    }
}

/// **Resolve a measured KV-tier A/B — the sc-17807 counterpart of
/// [`s18_verdict_from_accumulated_cells`].**
///
/// Both arms are measured by `long_clip_coherence_under_the_bounded_window` (the quantized one with
/// `KREA_S18_KV_BITS` set), each emitting `S18CELL` lines. This is what turns those two piles of
/// lines into an answer, and it applies **both** rules item 3 asks for:
///
///   * [`S18Sweep::verdict`] — the sweep's existing rule — to the quantized arm on its own, so the
///     arm has to be a structurally valid sweep before it is compared to anything;
///   * [`S18Sweep::kv_tier_comparison`] against the bf16 arm, which is the A/B itself.
///
/// ```text
/// KREA_S18_CELLS=$PWD/cells-bf16.tsv KREA_S18_Q8_CELLS=$PWD/cells-q8.tsv \
///   cargo test -p mlx-gen-krea-realtime --test integration \
///   generate_smoke::s18_kv_tier_ab_from_accumulated_cells -- --exact --ignored --nocapture
/// ```
///
/// **Both arms must come from the same host.** The recorded `MEASURED_640` baseline was measured on
/// `nax-macos`; a second box reproduces its row A seed 7 closely but not bit-exactly (different
/// Metal silicon → a different trajectory, and even a different winning descriptor component). That
/// gap is well inside the between-seed spread, but comparing a local quantized arm against a remote
/// bf16 baseline would confound tier with host, so this takes two evidence files rather than
/// defaulting one side to the recorded table.
///
/// `#[ignore]` because it is an operator entry point over supplied evidence, like its sibling. The
/// rules it applies are gated without weights by
/// [`the_kv_tier_comparison_can_fail_the_quantized_arm`] and
/// [`the_s18_verdict_rule_distinguishes_its_outcomes`].
#[test]
#[ignore = "operator entry point: resolves a KV-tier A/B from KREA_S18_CELLS + KREA_S18_Q8_CELLS"]
fn s18_kv_tier_ab_from_accumulated_cells() {
    let read = |var: &str| -> String {
        let path = std::env::var(var)
            .unwrap_or_else(|_| panic!("set {var} to a file of accumulated S18CELL lines"));
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read `{path}`: {e}"))
    };
    let want_bucket = std::env::var("KREA_S18_BUCKET").ok();
    let baseline = s18_sweep_from_accumulated(&read("KREA_S18_CELLS"), want_bucket.as_deref())
        .unwrap_or_else(|e| panic!("KREA_S18_CELLS: {e}"));
    let arm = s18_sweep_from_accumulated(&read("KREA_S18_Q8_CELLS"), want_bucket.as_deref())
        .unwrap_or_else(|e| panic!("KREA_S18_Q8_CELLS: {e}"));

    println!("baseline: {}", baseline.summary());
    println!("arm:      {}", arm.summary());

    // The arm has to stand on its own before it is compared: same structural guards, same verdict
    // rule the bf16 sweep is held to. (`verdict` answers "does this sweep still say what the docs
    // say", not "is q8 acceptable" — the comparison below is what answers the second question.)
    if let Err(e) = arm.structural_checks() {
        panic!("quantized arm: {e}");
    }
    match arm.verdict() {
        Ok(v) => println!("  ARM VERDICT: {v}"),
        Err(e) => panic!("quantized arm: {e}"),
    }
    match baseline.verdict() {
        Ok(v) => println!("  BASELINE VERDICT: {v}"),
        Err(e) => panic!("baseline arm: {e}"),
    }
    match arm.kv_tier_comparison(&baseline) {
        Ok(v) => println!("  KV TIER A/B: {v}"),
        Err(e) => panic!("{e}"),
    }
}
