//! Real-weight parity of the native YuE2 VAE against the pinned upstream (sc-22993).
//!
//! `#[ignore]`d in ordinary runs (CI has no weights); under `--ignored` a missing `YUE2_HF_HUB` or
//! reference file panics rather than silently passing. Needs:
//!
//! * `YUE2_HF_HUB` — a hub directory holding the pinned `m-a-p/YuE2-Vae` and
//!   `m-a-p/YuE2-Vae-legacy` revisions (see `scripts/reference/yue2/README.md`);
//! * the upstream reference produced by `scripts/reference/yue2/vae_reference.py real` (written
//!   outside the repository because it is derived from CC BY-NC 4.0 weights; its SHA-256 is
//!   committed in `tests/fixtures/vae_real_reference.json` and checked here). Location:
//!   `YUE2_VAE_REFERENCE_DIR`, default `~/.cache/sceneworks-yue2-fixtures/vae`.
//!
//! ```text
//! YUE2_HF_HUB=/path/to/huggingface/hub cargo test --release -p candle-audio-yue2 \
//!   --test vae_real_weights -- --ignored --nocapture --test-threads 1
//! ```
//!
//! CPU only. Measured peak RSS 2.3–2.6 GB (two runs) for the whole file: one 530 MB FP32 VAE
//! mapped plus its folded weights at a time (the encoder test loads both halves) and activations
//! of a ≤ 3 s clip; 37–50 s in release mode.
//!
//! Every waveform compared here is produced by the production entry points
//! ([`decode_latents`] on a [`Yue2Vae::load`]ed verified component) — not a test-only forward.

use std::path::PathBuf;
use std::time::Instant;

use candle_audio::candle_core::{Device, Tensor};
use candle_audio_yue2::decode::{decode_latents, DecodeMode, DecodeOptions, DecodedAudio};
use candle_audio_yue2::inventory::{self, ComponentId};
use candle_audio_yue2::latent::{AcousticLatents, LatentSource};
use candle_audio_yue2::snapshot::resolve_component;
use candle_audio_yue2::vae::{VaeParts, Yue2Vae};
use candle_audio_yue2::SnapshotDirs;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Native (Candle CPU) vs PyTorch CPU on the real decoders: max |Δ| bound on the clamped
/// production waveform. Measured 2026-09-26 (Apple M-series CPU, Candle vs torch 2.10.0 CPU):
/// 1.28e-5 standard, 1.02e-5 legacy, on samples of magnitude ≤ 1 — i.e. ~100 FP32 ulps after 30+
/// layers of up to 2048 channels. The bound is ~15× that, so BLAS/summation-order differences on
/// other CPUs do not flake, while every structural defect this guards (swapped channels, a
/// missing Snake exponent, an off-by-one crop, a missing clamp) moves samples by ≥ 0.1.
const WAVEFORM_MAX_ABS: f32 = 2e-4;
/// Minimum waveform SNR (dB) of native vs upstream — scale-sensitive, unlike cosine similarity.
/// Measured 120.9 / 121.8 dB; a 1% gain error alone would cap SNR at 40 dB.
const WAVEFORM_MIN_SNR_DB: f64 = 90.0;
/// Native tiled vs native full decode, same weights. Measured 0.0 (bit-identical: every output
/// sample's convolution sums the same products in the same order regardless of tile length). The
/// bound allows the native-vs-torch cross-runtime scale of rounding (~1e-5) with headroom, in case
/// another backend's GEMM blocking depends on the sequence length; the zero-halo mutation in the
/// tiny CI test proves the comparison is sensitive to real boundary errors (≫ 1e-3).
const TILED_VS_FULL_MAX_ABS: f32 = 1e-4;
/// Native vs upstream encoder posterior mean / scale. Measured 1.24e-5 / 1.69e-5 (values up to
/// ~10); ~12× headroom.
const ENCODE_MAX_ABS: f32 = 2e-4;

fn hub() -> SnapshotDirs {
    let hub = PathBuf::from(std::env::var_os("YUE2_HF_HUB").unwrap_or_else(|| {
        panic!(
            "real-weight test run without YUE2_HF_HUB (a hub directory holding the pinned repos)"
        )
    }));
    inventory::REPOS
        .iter()
        .fold(SnapshotDirs::new(), |dirs, repo| {
            let dir = hub
                .join(format!("models--{}", repo.id.replace('/', "--")))
                .join("snapshots")
                .join(repo.revision);
            dirs.with(repo.id, dir)
        })
}

fn meta() -> Value {
    serde_json::from_str(include_str!("fixtures/vae_real_reference.json")).unwrap()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The upstream reference tensors, refused unless their SHA-256 equals the committed record.
fn reference() -> std::collections::HashMap<String, Tensor> {
    let dir = std::env::var_os("YUE2_VAE_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME"))
                .join(".cache/sceneworks-yue2-fixtures/vae")
        });
    let meta = meta();
    let path = dir.join(meta["reference_file"].as_str().unwrap());
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e} — run scripts/reference/yue2/vae_reference.py real first",
            path.display()
        )
    });
    assert_eq!(
        hex(&Sha256::digest(&bytes)),
        meta["reference_sha256"].as_str().unwrap(),
        "{} is not the committed reference (regenerate it or the JSON together)",
        path.display()
    );
    candle_audio::candle_core::safetensors::load_buffer(&bytes, &Device::Cpu).unwrap()
}

/// Resolve + verify the component immediately before loading it (the crate's load-boundary rule).
fn load(id: ComponentId, parts: VaeParts) -> Yue2Vae {
    let start = Instant::now();
    let verified = resolve_component(id, &hub()).unwrap_or_else(|e| panic!("{e}"));
    let vae = Yue2Vae::load(&verified, parts, &Device::Cpu).unwrap_or_else(|e| panic!("{e}"));
    println!(
        "{id:?}: verified + loaded ({parts:?}) in {:.1?}",
        start.elapsed()
    );
    vae
}

fn to_vec(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// (max |Δ|, SNR dB of `want` against the difference).
fn compare(got: &[f32], want: &[f32]) -> (f32, f64) {
    assert_eq!(got.len(), want.len());
    let (mut max, mut sig, mut noise) = (0f32, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let d = g - w;
        max = max.max(d.abs());
        sig += (*w as f64).powi(2);
        noise += (d as f64).powi(2);
    }
    (max, 10.0 * (sig / noise.max(1e-30)).log10())
}

fn corr(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn rms(a: &[f32]) -> f64 {
    (a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
}

fn decode(vae: &Yue2Vae, l: &AcousticLatents, o: DecodeOptions) -> DecodedAudio {
    let start = Instant::now();
    let out = decode_latents(vae, l, &o, &|| false, &mut |_, _| {}).unwrap();
    println!(
        "  decode {:?}: {} samples/ch in {:.2?}",
        o.mode,
        out.frames(),
        start.elapsed()
    );
    out
}

/// AC1 + AC3 on real weights: the SAME cached latents (the upstream standard encoder's posterior
/// mean of an asymmetric clip, persisted and re-loaded as an artifact) decode with both pinned
/// decoders to finite 48 kHz stereo matching upstream `YuE2Pipeline.decode` — length, channel
/// order (L carries the loud square wave, R the quiet sine), clamping, and waveform agreement by
/// max |Δ| and SNR — with each output attributed to its decoder and to the one latent identity.
#[test]
#[ignore = "real weights: set YUE2_HF_HUB and generate the reference (see the module docs)"]
fn both_pinned_decoders_match_upstream_on_identical_cached_latents() {
    let r = reference();
    let meta = meta();
    let clip = to_vec(&r["clip"]);
    let samples = meta["frames"].as_u64().unwrap() as usize * 1920;
    let (clip_l, clip_r) = (&clip[..samples], &clip[samples..]);
    let latents = AcousticLatents::from_tensor(
        &r["latent"],
        LatentSource::Encoded {
            vae: "yue2_vae".into(),
            audio_sha256: hex(&Sha256::digest(
                clip.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )),
            sampled: false,
        },
    )
    .unwrap();
    assert_eq!(
        latents.identity().sha256,
        meta["latent_sha256"].as_str().unwrap()
    );
    let cache = tempfile::tempdir().unwrap();
    latents.save(cache.path()).unwrap();
    let npy_before = std::fs::read(cache.path().join("latent.npy")).unwrap();

    for (id, key) in [
        (ComponentId::VaeStandard, "standard"),
        (ComponentId::VaeLegacy, "legacy"),
    ] {
        let rec = &meta["decoders"][key];
        let vae = load(id, VaeParts::DecoderOnly);
        // Identity: exactly the pinned component, from the verified snapshot.
        let ident = vae.identity();
        assert_eq!(ident.release(), key);
        assert_eq!(ident.repo, rec["repo"].as_str().unwrap());
        assert_eq!(ident.revision, rec["revision"].as_str().unwrap());
        assert_eq!(
            ident.weights_sha256,
            rec["weights_sha256"].as_str().unwrap()
        );
        assert_eq!(ident.config_sha256, rec["config_sha256"].as_str().unwrap());
        assert_eq!(
            vae.required_halo(4) as u64,
            rec["required_halo_core_4"].as_u64().unwrap()
        );
        assert_eq!(
            vae.required_halo(1024) as u64,
            rec["required_halo_default"].as_u64().unwrap()
        );

        let cached = AcousticLatents::load(cache.path()).unwrap();
        let out = decode(&vae, &cached, DecodeOptions::production());
        let natural = rec["natural_output_length"].as_u64().unwrap() as usize;
        assert_eq!(out.frames(), natural);
        assert_eq!(out.frames(), 1920 * cached.frames() - 64);
        assert_eq!(out.metadata().sample_rate, 48_000);
        assert_eq!(out.metadata().channels, 2);
        assert!(out
            .samples()
            .iter()
            .all(|s| s.is_finite() && s.abs() <= 1.0));
        assert_eq!(out.metadata().decoder, *ident);
        assert_eq!(out.metadata().latents, *latents.identity());

        let want = to_vec(&r[&format!("{key}.pipeline_default")]);
        let (max, snr) = compare(out.samples(), &want);
        println!("  {key}: native vs upstream pipeline max|Δ| = {max:e}, SNR = {snr:.1} dB");
        assert!(max < WAVEFORM_MAX_ABS, "{key}: max|Δ| {max}");
        assert!(snr > WAVEFORM_MIN_SNR_DB, "{key}: SNR {snr}");

        // Clamping is active on this clip, and agrees with upstream's raw overshoot.
        let raw_py = to_vec(&r[&format!("{key}.full_raw")]);
        let py_beyond = raw_py.iter().filter(|x| x.abs() > 1.0).count();
        let native_beyond = out.metadata().clamped_samples;
        println!("  {key}: clamped {native_beyond} (upstream raw beyond ±1: {py_beyond})");
        assert!(native_beyond > 0);
        assert_eq!(
            py_beyond as u64,
            rec["raw_samples_beyond_unit"].as_u64().unwrap()
        );
        let near_unit = raw_py
            .iter()
            .filter(|x| (x.abs() - 1.0).abs() <= WAVEFORM_MAX_ABS)
            .count();
        assert!(native_beyond.abs_diff(py_beyond) <= near_unit);

        // Channel order: L follows the loud square-wave input, R the quiet sine.
        let (left, right) = (out.channel(0), out.channel(1));
        let (cl, cr) = (corr(&left, clip_l), corr(&right, clip_r));
        let (xl, xr) = (corr(&left, clip_r), corr(&right, clip_l));
        println!(
            "  {key}: rms L {:.3} R {:.3}; corr(L,inL) {cl:.3} corr(R,inR) {cr:.3} \
             corr(L,inR) {xl:.3} corr(R,inL) {xr:.3}",
            rms(&left),
            rms(&right)
        );
        assert!(rms(&left) > 2.0 * rms(&right));
        assert!(cl > 0.5 && cr > 0.5 && cl > xl.abs() && cr > xr.abs());

        // The full reference FP32 path and a many-seam tiling agree with the production output.
        let full = decode(&vae, &cached, DecodeOptions::reference_full());
        let (max_full, _) = compare(full.samples(), out.samples());
        let seams = decode(
            &vae,
            &cached,
            DecodeOptions {
                mode: DecodeMode::Tiled { core_frames: 4 },
                halo_frames: 16,
            },
        );
        let (max_seams, _) = compare(seams.samples(), full.samples());
        println!("  {key}: full vs production {max_full:e}; core-4 tiling vs full {max_seams:e}");
        assert!(max_full < TILED_VS_FULL_MAX_ABS && max_seams < TILED_VS_FULL_MAX_ABS);
    }
    // Decoding with both decoders never touched the cached latents.
    assert_eq!(
        std::fs::read(cache.path().join("latent.npy")).unwrap(),
        npy_before
    );
}

/// AC2 on real weights at song-like length: halo/crop tiles (core 16, five seams, plus a short
/// final tile) against the full reference decode, both decoders, on a 75-frame (3 s) latent.
#[test]
#[ignore = "real weights: set YUE2_HF_HUB and generate the reference (see the module docs)"]
fn halo_crop_tiling_matches_full_decode_on_a_long_latent() {
    let r = reference();
    let base = to_vec(&r["latent"]);
    // 75 frames: the 16-frame reference latent, then four affine variants of it (deterministic,
    // in-distribution scale), truncated.
    let mut values = Vec::new();
    for (k, (a, b)) in [
        (1.0, 0.0),
        (-0.8, 0.05),
        (0.9, -0.1),
        (0.6, 0.2),
        (-1.1, 0.0),
    ]
    .into_iter()
    .enumerate()
    {
        values.extend(base.iter().map(|v| a * v + b + 0.01 * k as f32));
    }
    values.truncate(75 * 64);
    let latents = AcousticLatents::new(
        values,
        75,
        LatentSource::Synthesis {
            stage_identity: "long-affine-fixture".into(),
        },
    )
    .unwrap();
    for id in [ComponentId::VaeStandard, ComponentId::VaeLegacy] {
        let vae = load(id, VaeParts::DecoderOnly);
        let full = decode(&vae, &latents, DecodeOptions::reference_full());
        let tiled = decode(
            &vae,
            &latents,
            DecodeOptions {
                mode: DecodeMode::Tiled { core_frames: 16 },
                halo_frames: 16,
            },
        );
        assert_eq!(full.frames(), 75 * 1920 - 64);
        let (max, snr) = compare(tiled.samples(), full.samples());
        // Seam neighbourhoods specifically (±128 samples around each core boundary).
        let mut seam_max = 0f32;
        for seam in (16usize..75).step_by(16).map(|f| f * 1920) {
            for i in seam.saturating_sub(128)..(seam + 128).min(full.frames()) {
                for c in 0..2 {
                    let k = i * 2 + c;
                    seam_max = seam_max.max((tiled.samples()[k] - full.samples()[k]).abs());
                }
            }
        }
        println!(
            "  {id:?}: tiled(core 16) vs full max|Δ| {max:e} (seams {seam_max:e}), SNR {snr:.1} dB"
        );
        assert!(max < TILED_VS_FULL_MAX_ABS, "{id:?}: {max}");
    }
}

/// The native encoder (both released VAEs) reproduces upstream's posterior of the reference clip,
/// and its posterior-mean latents decode through the production path like upstream's.
#[test]
#[ignore = "real weights: set YUE2_HF_HUB and generate the reference (see the module docs)"]
fn native_encoder_matches_upstream_for_both_vaes() {
    let r = reference();
    for (id, key) in [
        (ComponentId::VaeStandard, "standard"),
        (ComponentId::VaeLegacy, "legacy"),
    ] {
        let vae = load(id, VaeParts::Full);
        let start = Instant::now();
        let post = vae.encode(&r["clip"]).unwrap();
        println!("  {key}: encode in {:.2?}", start.elapsed());
        for (label, got) in [("mean", &post.mean), ("scale", &post.scale)] {
            let (max, snr) = compare(&to_vec(got), &to_vec(&r[&format!("{key}.encode_{label}")]));
            println!("  {key} encode {label}: max|Δ| {max:e}, SNR {snr:.1} dB");
            assert!(max < ENCODE_MAX_ABS, "{key} {label}: {max}");
        }
        if key == "standard" {
            let latents = vae
                .encode_latents(&r["clip"].squeeze(0).unwrap(), None)
                .unwrap();
            assert_eq!(latents.frames(), 16);
            let (max, _) = compare(latents.values(), &to_vec(&r["latent"]));
            assert!(max < ENCODE_MAX_ABS, "encode_latents: {max}");
        }
    }
}
