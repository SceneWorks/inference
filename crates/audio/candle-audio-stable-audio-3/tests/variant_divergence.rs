//! Music-vs-SFX divergence gate (`sc-14544`).
//!
//! The two registered checkpoints are architecturally identical: a mis-wired weight path would
//! produce *valid* audio under the wrong id and pass every shape, conformance, and quality gate.
//! The only thing that catches it is proving the two registrations actually diverge, by an amount
//! the frozen Torch reference says they should.
//!
//! The divergence floor is not invented. `docs/migration/sa3-reference/` carries the committed
//! `sc-14534` PyTorch artifacts for both checkpoints, generated from identical inputs (same seed,
//! same prompt, same noise, same sigmas, same tokenizer ids and attention mask). Reading them here
//! yields the reference envelope directly, and re-verifying it in-test means the gate's premise is
//! re-checked rather than assumed.

use std::path::{Path, PathBuf};

use candle_audio_stable_audio_3::candle_audio::candle_core::{
    safetensors::MmapedSafetensors, DType, Device, Tensor,
};
use candle_audio_stable_audio_3::dit::{DitInputs, StableAudio3Dit};
use candle_audio_stable_audio_3::gen_core::{
    AudioParams, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use candle_audio_stable_audio_3::weights::SnapshotLayout;
use candle_audio_stable_audio_3::Variant;
use sha2::{Digest, Sha256};

/// A prompt neither checkpoint's demo set biases toward, so the comparison is about the weights.
const PROMPT: &str = "a short bright transient followed by a decaying tail";
const SEED: u64 = 14_544;

fn snapshot(env: &str) -> PathBuf {
    PathBuf::from(
        std::env::var(env).unwrap_or_else(|_| panic!("set {env} to the pinned immutable snapshot")),
    )
}

fn reference_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../docs/migration/sa3-reference")
}

fn reference(artifact: &str) -> MmapedSafetensors {
    // Safety: the sc-14534 artifacts are committed, immutable, and hash-pinned by their manifest.
    unsafe { MmapedSafetensors::new(reference_dir().join(artifact)).unwrap() }
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    assert_eq!(left.len(), right.len());
    let mut dot = 0f64;
    let mut aa = 0f64;
    let mut bb = 0f64;
    for (&a, &b) in left.iter().zip(right) {
        dot += a as f64 * b as f64;
        aa += (a as f64).powi(2);
        bb += (b as f64).powi(2);
    }
    dot / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE)
}

fn rms(values: &[f32]) -> f64 {
    (values.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
}

/// `rms(a - b) / rms(a)` — 0 for identical signals, ~sqrt(2) for uncorrelated ones of equal power.
fn normalized_rms_delta(left: &[f32], right: &[f32]) -> f64 {
    let delta = left
        .iter()
        .zip(right)
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    rms(&delta) / rms(left).max(f64::MIN_POSITIVE)
}

fn digest(samples: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for sample in samples {
        hasher.update(sample.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// The frozen Torch music-vs-SFX envelope, read from the committed sc-14534 artifacts.
struct ReferenceDivergence {
    dit_cosine: f64,
    final_latent_cosine: f64,
    final_latent_rms_delta: f64,
}

fn reference_divergence() -> ReferenceDivergence {
    let device = Device::Cpu;
    let music = reference("small-music-reference.safetensors");
    let sfx = reference("small-sfx-reference.safetensors");
    let pair = |name: &str| {
        (
            values(&music.load(name, &device).unwrap()),
            values(&sfx.load(name, &device).unwrap()),
        )
    };

    // The two runs shared every stochastic and textual input, so any divergence below is the
    // checkpoints and nothing else.
    for shared in ["dit_noise", "sampler_initial_noise", "t5_last_hidden_state"] {
        let (music, sfx) = pair(shared);
        assert_eq!(
            music, sfx,
            "sc-14534 {shared} must be shared across variants"
        );
    }

    let (music_dit, sfx_dit) = pair("dit_prediction");
    let (music_final, sfx_final) = pair("sampler_final");
    let divergence = ReferenceDivergence {
        dit_cosine: cosine(&music_dit, &sfx_dit),
        final_latent_cosine: cosine(&music_final, &sfx_final),
        final_latent_rms_delta: normalized_rms_delta(&music_final, &sfx_final),
    };
    eprintln!(
        "frozen torch reference: dit_cosine={:.6} final_latent_cosine={:.6} \
         final_latent_rms_delta={:.6}",
        divergence.dit_cosine, divergence.final_latent_cosine, divergence.final_latent_rms_delta
    );
    divergence
}

fn generate(variant: Variant, env: &str, duration: f32, steps: u32) -> Vec<f32> {
    let generator = candle_audio_stable_audio_3::provider_registry()
        .unwrap()
        .load(
            variant.model_id(),
            &LoadSpec::new(WeightsSource::Dir(snapshot(env))),
        )
        .unwrap_or_else(|error| panic!("load {}: {error}", variant.model_id()));
    let request = GenerationRequest {
        prompt: PROMPT.into(),
        seed: Some(SEED),
        steps: Some(steps),
        sampler: Some("pingpong".into()),
        audio: Some(AudioParams {
            target_duration: Some(duration),
            sample_rate: Some(44_100),
            ..Default::default()
        }),
        ..Default::default()
    };
    match generator.generate(&request, &mut |_| {}).unwrap() {
        GenerationOutput::Audio(track) => track.samples,
        other => panic!("expected audio, got {other:?}"),
    }
}

/// Both registered variants, driven through the production path with an identical prompt, seed,
/// duration, step count, and sampler — and therefore an identical request-local noise stream,
/// which is seeded from the request and never from the weights.
#[test]
#[ignore = "requires both pinned 3.45 GB small snapshots"]
fn music_and_sfx_produce_materially_different_audio_from_the_same_prompt_and_seed() {
    let expected = reference_divergence();
    // Re-verify the premise the runtime gate is derived from: after eight Pingpong steps the frozen
    // Torch checkpoints are effectively orthogonal.
    assert!(
        expected.final_latent_cosine.abs() < 0.05,
        "frozen reference final-latent cosine drifted to {}",
        expected.final_latent_cosine
    );
    assert!(
        expected.final_latent_rms_delta > 0.75,
        "frozen reference final-latent RMS delta drifted to {}",
        expected.final_latent_rms_delta
    );
    assert!(
        (0.5..0.75).contains(&expected.dit_cosine),
        "frozen reference single-step DiT cosine drifted to {}",
        expected.dit_cosine
    );

    let duration = std::env::var("SA3_TEST_DURATION")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2.0f32);
    let steps = std::env::var("SA3_TEST_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8u32);

    // Sequential, so only one 3.45 GB graph is resident at a time.
    let music = generate(
        Variant::SmallMusic,
        "SA3_SMALL_MUSIC_SNAPSHOT",
        duration,
        steps,
    );
    let sfx = generate(Variant::SmallSfx, "SA3_SMALL_SFX_SNAPSHOT", duration, steps);
    assert_eq!(music.len(), sfx.len());

    let music_digest = digest(&music);
    let sfx_digest = digest(&sfx);
    let observed_cosine = cosine(&music, &sfx);
    let observed_rms_delta = normalized_rms_delta(&music, &sfx);
    eprintln!(
        "runtime divergence ({duration}s, {steps} steps): music_sha256={music_digest} \
         sfx_sha256={sfx_digest} audio_cosine={observed_cosine:.6} \
         audio_rms_delta={observed_rms_delta:.6} music_rms={:.9} sfx_rms={:.9}",
        rms(&music),
        rms(&sfx)
    );

    assert_ne!(
        music_digest, sfx_digest,
        "identical PCM from two registrations means one weight path is mis-wired"
    );
    // Floor derived from the frozen reference above: the two checkpoints' eight-step outputs sit
    // at |cos| = 0.0186. This allows an order of magnitude more agreement than the reference while
    // staying far below the ~1.0 a shared-weight path would produce.
    assert!(
        observed_cosine.abs() <= 0.35,
        "music/SFX waveform cosine {observed_cosine} is far above the frozen-reference envelope \
         ({}); the two registrations are probably serving the same weights",
        expected.final_latent_cosine
    );
    // The frozen reference measures a normalized RMS delta of ~1.00 between the two checkpoints.
    assert!(
        observed_rms_delta >= 0.5,
        "music/SFX normalized RMS delta {observed_rms_delta} is below half the frozen-reference \
         envelope ({})",
        expected.final_latent_rms_delta
    );
    // Neither side may be silent — divergence between two silences is meaningless.
    assert!(rms(&music) > 1e-4 && rms(&sfx) > 1e-4);
}

/// The exact reference-derived number, at the DiT boundary where the frozen artifacts are directly
/// comparable.
///
/// Each variant is driven with **its own** `t5_projected_padded`, exactly as the frozen Torch run
/// that produced `dit_prediction` was. The two checkpoints share the raw T5 hidden state but each
/// applies its own learned prompt padding, so substituting one variant's projected prompt for the
/// other's would measure a different quantity than `expected.dit_cosine` and make the tolerance
/// below meaningless. Noise, timestep, seconds, and local conditioning are shared and identical.
#[test]
#[ignore = "requires both pinned 3.45 GB small snapshots"]
fn single_step_dit_divergence_matches_the_frozen_torch_reference() {
    let expected = reference_divergence();
    let device = Device::Cpu;
    let music_reference = reference("small-music-reference.safetensors");
    let noise = music_reference.load("dit_noise", &device).unwrap();
    let timestep = music_reference.load("dit_timestep", &device).unwrap();
    let seconds = Tensor::from_vec(vec![0.25f32], 1, &device).unwrap();
    let local = Tensor::zeros((1, 257, 16), DType::F32, &device).unwrap();

    let mut predictions = Vec::new();
    for (env, artifact) in [
        (
            "SA3_SMALL_MUSIC_SNAPSHOT",
            "small-music-reference.safetensors",
        ),
        ("SA3_SMALL_SFX_SNAPSHOT", "small-sfx-reference.safetensors"),
    ] {
        let prompt = reference(artifact)
            .load("t5_projected_padded", &device)
            .unwrap();
        let layout = SnapshotLayout::from_dir(&snapshot(env)).unwrap();
        let dit = StableAudio3Dit::from_layout(&layout, &device).unwrap();
        predictions.push(values(
            &dit.forward(DitInputs {
                latents: &noise,
                timestep: &timestep,
                prompt: &prompt,
                seconds_total: &seconds,
                local_conditioning: &local,
                padding_mask: None,
            })
            .unwrap(),
        ));
    }
    let observed = cosine(&predictions[0], &predictions[1]);
    eprintln!(
        "candle single-step DiT music-vs-SFX cosine={observed:.6} (frozen torch {:.6})",
        expected.dit_cosine
    );
    assert!(
        (observed - expected.dit_cosine).abs() <= 0.02,
        "candle music-vs-SFX DiT divergence {observed} does not reproduce the frozen torch value {}",
        expected.dit_cosine
    );
}
