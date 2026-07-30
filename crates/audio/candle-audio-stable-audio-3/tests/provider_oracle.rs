//! Frozen upstream, real-weight, end-to-end provider parity for `sc-14543`.

use std::path::{Path, PathBuf};

use candle_audio_stable_audio_3::candle_audio::candle_core::{
    safetensors::MmapedSafetensors, DType, Device, Tensor,
};
use candle_audio_stable_audio_3::dit::Guidance;
use candle_audio_stable_audio_3::pipeline::{StableAudio3Pipeline, SynthesisParameters};
use candle_audio_stable_audio_3::same::SameDecodeChunkNoise;
use candle_audio_stable_audio_3::sampler::SamplerKind;
use candle_audio_stable_audio_3::weights::SnapshotLayout;

const SEED: u64 = 14_543;
const PROMPT: &str = "Warm analog synth pulses, crisp percussion, spacious stereo field";
const LATENTS: usize = 388;
const FRAMES: usize = 1_323_000;

fn portable(shape: &[usize], stream: u64, scale: f32, device: &Device) -> Tensor {
    let count = shape.iter().product();
    let values = (0..count)
        .map(|index| {
            let bits = (index as u64)
                .wrapping_mul(1_664_525)
                .wrapping_add((SEED + stream).wrapping_mul(1_013_904_223))
                & 0xffff_ffff;
            ((bits as f64 / 2_147_483_648.0 - 1.0) as f32) * scale
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, shape.to_vec(), device).unwrap()
}

fn snapshot() -> PathBuf {
    std::env::var_os("SA3_SMALL_MUSIC_SNAPSHOT")
        .map(PathBuf::from)
        .expect("SA3_SMALL_MUSIC_SNAPSHOT must point to the pinned immutable snapshot")
}

fn oracle() -> MmapedSafetensors {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../docs/migration/sa3-small-music-provider-reference");
    // Safety: the generated artifact is immutable and hash-pinned by its committed manifest.
    unsafe { MmapedSafetensors::new(root.join("provider-output.safetensors")).unwrap() }
}

fn metrics(actual: &[f32], expected: &[f32]) -> (f64, f32, f64) {
    assert_eq!(actual.len(), expected.len());
    let mut dot = 0f64;
    let mut aa = 0f64;
    let mut bb = 0f64;
    let mut max_abs = 0f32;
    let mut mean_abs = 0f64;
    for (&a, &b) in actual.iter().zip(expected) {
        dot += a as f64 * b as f64;
        aa += (a as f64).powi(2);
        bb += (b as f64).powi(2);
        let delta = (a - b).abs();
        max_abs = max_abs.max(delta);
        mean_abs += delta as f64;
    }
    (
        dot / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE),
        max_abs,
        mean_abs / actual.len() as f64,
    )
}

/// Deliberately CPU-pinned, in the same sense as `dit_oracle.rs`'s
/// `all_six_cpu_f32_predictions_match_p0`, and **not** switched to the three-way `SA3_TEST_*`
/// selector the rest of the crate's real-weight targets use.
///
/// This gate is bit-reproduction against a frozen Torch CPU-f32 artifact, not backend
/// certification, and its design says so throughout: the noise is a portable host LCG
/// ([`portable`]) precisely so no device RNG can enter, and the seed and step count are fixed. Its
/// bounds are exact-reproduction-grade — latent cosine `>= 0.99999`, exact-latent decode
/// `max_abs <= 1e-3` and `mean_abs <= 1e-4`, and `deltas_gt_0.1 == 0` across 1,323,000 stereo
/// frames — measured on CPU f32 against a CPU f32 reference. A reduced-precision accelerator
/// matmul contributes on the order of 1e-3 relative per op, compounded across eight sampler steps
/// and a full VAE decode, so those bounds are calibrated on *this* device and would not hold on
/// Metal or CUDA. Running this under the selector would convert a parity oracle into a flake.
///
/// The consequence is stated rather than hidden: `sa3-small-music-metal` selects this target while
/// setting `SA3_TEST_METAL`, and this step alone does **not** certify Metal. The Metal small-music
/// path is certified by that job's other steps — `conformance`'s registered-provider and
/// concurrent-RNG cases, the stereo-width calibration sweep, and the 30 s render — every one of
/// which resolves its device through `candle_audio::default_device()`. There is deliberately no
/// CUDA counterpart of this step, for the same reason.
#[test]
#[ignore = "requires the pinned 3.45 GB small-music snapshot"]
fn thirty_second_eight_step_provider_matches_frozen_torch() {
    let device = Device::Cpu;
    let layout = SnapshotLayout::from_dir(&snapshot()).unwrap();
    let pipeline = StableAudio3Pipeline::from_layout(
        &layout,
        candle_audio_stable_audio_3::Variant::SmallMusic.geometry(),
        &device,
    )
    .unwrap();
    let initial = portable(&[1, 256, LATENTS], 0, 1.0, &device);
    let pingpong = (1..=8)
        .map(|stream| portable(&[1, 256, LATENTS], stream, 1.0, &device))
        .collect::<Vec<_>>();
    let decode = (0..4)
        .map(|chunk| {
            let base = 9 + chunk * 2;
            SameDecodeChunkNoise {
                regularization_noise: Some(portable(&[1, 256, 128], base, 1.0, &device)),
                mask_noises: vec![portable(&[128, 16, 768], base + 1, 0.01, &device)
                    .reshape((1, 128 * 16, 768))
                    .unwrap()],
            }
        })
        .collect::<Vec<_>>();
    let mut progress = Vec::new();
    let mut decoding = 0;
    let (latents, audio) = pipeline
        .synthesize_controlled(
            PROMPT,
            None,
            SynthesisParameters {
                duration_secs: 30.0,
                steps: 8,
                sampler: SamplerKind::Pingpong,
                guidance: Guidance {
                    cfg_scale: 1.0,
                    apg_scale: 1.0,
                    cfg_norm_threshold: 0.0,
                    scale_phi: 0.0,
                },
                seed: SEED,
            },
            &initial,
            pingpong,
            &decode,
            &mut |current, total| progress.push((current, total)),
            &mut || decoding += 1,
            &|| false,
        )
        .unwrap();
    assert_eq!(progress, (1..=8).map(|step| (step, 8)).collect::<Vec<_>>());
    assert_eq!(decoding, 1);
    assert_eq!(audio.len(), FRAMES * 2);

    let oracle = oracle();
    let expected_latents_tensor = oracle
        .load("sampled_latents", &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let expected_latents = expected_latents_tensor
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let actual_latents = latents.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let latent_metrics = metrics(&actual_latents, &expected_latents);
    eprintln!(
        "latents: cosine={:.9} max_abs={:.9} mean_abs={:.9}",
        latent_metrics.0, latent_metrics.1, latent_metrics.2
    );
    assert!(latent_metrics.0 >= 0.99999);
    assert!(latent_metrics.1 <= 0.03);
    assert!(latent_metrics.2 <= 0.002);

    let actual_chunk = pipeline
        .decode_chunk_controlled(
            &expected_latents_tensor.narrow(2, 0, 128).unwrap(),
            &decode[0],
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let expected_chunk = oracle
        .load("decoded_chunk_0", &device)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let chunk_metrics = metrics(&actual_chunk, &expected_chunk);
    eprintln!(
        "direct chunk: cosine={:.9} max_abs={:.9} mean_abs={:.9}",
        chunk_metrics.0, chunk_metrics.1, chunk_metrics.2
    );
    assert!(chunk_metrics.0 >= 0.9999);
    assert!(chunk_metrics.1 <= 0.0015);

    let expected_planar = oracle
        .load("audio", &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec3::<f32>()
        .unwrap();
    let expected_audio = (0..FRAMES)
        .flat_map(|frame| [expected_planar[0][0][frame], expected_planar[0][1][frame]])
        .collect::<Vec<_>>();
    let exact_decode = pipeline
        .decode_controlled(&expected_latents_tensor, 30.0, &decode)
        .unwrap();
    let exact_decode_metrics = metrics(&exact_decode, &expected_audio);
    let exact_large = exact_decode
        .iter()
        .zip(&expected_audio)
        .enumerate()
        .filter_map(|(index, (actual, expected))| {
            ((*actual - *expected).abs() > 0.1).then_some(index)
        })
        .collect::<Vec<_>>();
    eprintln!(
        "exact-latent decode: cosine={:.9} max_abs={:.9} mean_abs={:.9} \
         deltas_gt_0.1={} first={:?}",
        exact_decode_metrics.0,
        exact_decode_metrics.1,
        exact_decode_metrics.2,
        exact_large.len(),
        &exact_large[..exact_large.len().min(16)],
    );
    assert!(exact_decode_metrics.0 >= 0.99999);
    assert!(exact_decode_metrics.1 <= 0.001);
    assert!(exact_decode_metrics.2 <= 0.0001);

    let audio_metrics = metrics(&audio, &expected_audio);
    let large_audio_deltas = audio
        .iter()
        .zip(&expected_audio)
        .filter(|(actual, expected)| (*actual - *expected).abs() > 0.1)
        .count();
    eprintln!(
        "audio: cosine={:.9} max_abs={:.9} mean_abs={:.9} deltas_gt_0.1={large_audio_deltas}",
        audio_metrics.0, audio_metrics.1, audio_metrics.2,
    );
    assert!(audio_metrics.0 >= 0.9999);
    assert!(audio_metrics.1 <= 0.1);
    assert!(audio_metrics.2 <= 0.001);
    assert_eq!(large_audio_deltas, 0);
}
