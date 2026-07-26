//! Frozen-upstream and real-weight gates for SAME outer chunking (`sc-14540`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_audio_stable_audio_3::candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio_stable_audio_3::candle_audio::dsp::{hann_window, stft};
use candle_audio_stable_audio_3::same::{
    SameAutoencoder, SameChunkingParameters, SameChunkingPolicy, SameDecodeChunkNoise,
};
use candle_audio_stable_audio_3::weights::SnapshotLayout;
use candle_nn::VarBuilder;

const SEED: u64 = 14_540;
const LATENTS: usize = 225;
const RATIO: usize = 4096;
const CHUNK: usize = 128;

fn snapshot(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} must point to the pinned immutable snapshot"))
}

fn test_device() -> Device {
    if std::env::var_os("SA3_TEST_METAL").is_some() {
        Device::new_metal(0).expect("SA3_TEST_METAL requested but Metal is unavailable")
    } else if std::env::var_os("SA3_TEST_CUDA").is_some() {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0).expect("SA3_TEST_CUDA requested but CUDA is unavailable")
        }
        #[cfg(not(feature = "cuda"))]
        {
            panic!("SA3_TEST_CUDA requires --features cuda")
        }
    } else {
        Device::Cpu
    }
}

fn device_allocated_bytes(device: &Device) -> Option<usize> {
    #[cfg(feature = "metal")]
    {
        device
            .as_metal_device()
            .ok()
            .map(|metal| metal.metal_device().current_allocated_size())
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = device;
        None
    }
}

fn with_peak_device_bytes<T>(device: &Device, operation: impl FnOnce() -> T) -> (T, Option<usize>) {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    let Some(initial) = device_allocated_bytes(device) else {
        return (operation(), None);
    };
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicUsize::new(initial));
    let sampled_device = device.clone();
    let sampled_stop = Arc::clone(&stop);
    let sampled_peak = Arc::clone(&peak);
    let sampler = std::thread::spawn(move || {
        while !sampled_stop.load(Ordering::Relaxed) {
            if let Some(bytes) = device_allocated_bytes(&sampled_device) {
                sampled_peak.fetch_max(bytes, Ordering::Relaxed);
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        if let Some(bytes) = device_allocated_bytes(&sampled_device) {
            sampled_peak.fetch_max(bytes, Ordering::Relaxed);
        }
    });
    let output = operation();
    stop.store(true, Ordering::Relaxed);
    sampler.join().expect("Metal allocation sampler thread");
    (output, Some(peak.load(Ordering::Relaxed)))
}

fn fixture(device: &Device) -> VarBuilder<'static> {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../docs/migration/sa3-chunked-reference");
    // Safety: both generated artifacts are hash-pinned and immutable for the test process.
    unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[
                root.join("chunked-f32.safetensors"),
                root.join("chunked-outputs-f16.safetensors"),
            ],
            DType::F32,
            device,
        )
        .unwrap()
    }
}

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

fn metric(name: &str, actual: &Tensor, expected: &Tensor, max_abs_limit: f32) {
    let actual = actual
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let expected = expected
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(actual.len(), expected.len(), "{name}");
    let mut dot = 0f64;
    let mut aa = 0f64;
    let mut bb = 0f64;
    let mut max_abs = 0f32;
    for (&a, &b) in actual.iter().zip(&expected) {
        dot += a as f64 * b as f64;
        aa += (a as f64).powi(2);
        bb += (b as f64).powi(2);
        max_abs = max_abs.max((a - b).abs());
    }
    let cosine = dot / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE);
    eprintln!("{name}: cosine={cosine:.9} max_abs={max_abs:.9}");
    assert!(cosine >= 0.9999, "{name}: cosine {cosine}");
    assert!(
        max_abs <= max_abs_limit,
        "{name}: max_abs {max_abs} > {max_abs_limit}"
    );
}

fn chunk_encode_noises(label: &str, dim: usize, device: &Device) -> Vec<Vec<Tensor>> {
    if label == "same_s" {
        return (0..3)
            .map(|_| vec![Tensor::zeros((1, CHUNK, dim), DType::F32, device).unwrap()])
            .collect();
    }
    (0..3)
        .map(|index| {
            vec![portable(&[CHUNK, 1, dim], index, 0.001, device)
                .reshape((1, CHUNK, dim))
                .unwrap()]
        })
        .collect()
}

fn chunk_decode_noises(
    dim: usize,
    token_scale: f32,
    zero: bool,
    device: &Device,
) -> Vec<SameDecodeChunkNoise> {
    (0..3)
        .map(|index| {
            let regularization = if zero {
                Tensor::zeros((1, 256, CHUNK), DType::F32, device).unwrap()
            } else {
                portable(&[1, 256, CHUNK], 100 + index * 2, 1.0, device)
            };
            let tokens = if zero {
                Tensor::zeros((1, CHUNK * 16, dim), DType::F32, device).unwrap()
            } else {
                portable(&[CHUNK, 16, dim], 101 + index * 2, token_scale, device)
                    .reshape((1, CHUNK * 16, dim))
                    .unwrap()
            };
            SameDecodeChunkNoise {
                regularization_noise: Some(regularization),
                mask_noises: vec![tokens],
            }
        })
        .collect()
}

fn direct_zero_noises(label: &str, dim: usize, device: &Device) -> (Tensor, Tensor) {
    let padded_latents = if label == "same_s" {
        LATENTS.div_ceil(2) * 2
    } else {
        LATENTS
    };
    (
        Tensor::zeros((1, 256, LATENTS), DType::F32, device).unwrap(),
        Tensor::zeros((1, padded_latents * 16, dim), DType::F32, device).unwrap(),
    )
}

fn spectral_discontinuity(value: &Tensor, boundary: usize) -> (f64, f32) {
    const WINDOW: usize = 2048;
    let values = value
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec3::<f32>()
        .unwrap();
    let hann = hann_window(WINDOW);
    let mut spectral = 0f64;
    let mut count = 0usize;
    let mut jump = 0f32;
    for channel in &values[0] {
        let left = &channel[boundary - WINDOW..boundary];
        let right = &channel[boundary..boundary + WINDOW];
        let left_mag = stft(left, WINDOW, WINDOW, &hann).unwrap().magnitude();
        let right_mag = stft(right, WINDOW, WINDOW, &hann).unwrap().magnitude();
        for (&a, &b) in left_mag.iter().zip(&right_mag) {
            spectral += ((a as f64).ln_1p() - (b as f64).ln_1p()).abs();
            count += 1;
        }
        jump = jump.max((channel[boundary] - channel[boundary - 1]).abs());
    }
    (spectral / count as f64, jump)
}

#[test]
#[ignore = "requires pinned standalone SAME-S/SAME-L snapshots"]
fn chunked_same_s_and_same_l_match_frozen_torch_and_boundary_metrics() {
    let device = test_device();
    let oracle = fixture(&device);
    let selected = std::env::var("SA3_CHUNKED_CASE").ok();
    assert!(
        selected
            .as_deref()
            .is_none_or(|value| matches!(value, "same_s" | "same_l")),
        "SA3_CHUNKED_CASE must be same_s or same_l"
    );
    for (label, env, dim, token_scale) in [
        ("same_s", "SA3_SAME_S_SNAPSHOT", 768usize, 0.01f32),
        ("same_l", "SA3_SAME_L_SNAPSHOT", 1536usize, 0.1f32),
    ] {
        if selected
            .as_deref()
            .is_some_and(|selected| selected != label)
        {
            continue;
        }
        let layout = SnapshotLayout::from_dir(&snapshot(env)).unwrap();
        let model = SameAutoencoder::load(
            layout.config.autoencoder(),
            layout.mmap_builders(DType::F32, &device).unwrap(),
        )
        .unwrap();
        let policy = SameChunkingPolicy::standalone(true);
        let parameters = SameChunkingParameters::default();
        let audio = portable(&[1, 2, LATENTS * RATIO], 900, 0.25, &device);
        let latents = portable(&[1, 256, LATENTS], 901, 0.125, &device);

        let encoded = model
            .encode_audio_with_chunk_noises(
                &audio,
                policy,
                parameters,
                &chunk_encode_noises(label, dim, &device),
            )
            .unwrap();
        metric(
            &format!("{label}.encoded"),
            &encoded,
            &oracle
                .get((1, 256, LATENTS), &format!("{label}.encoded"))
                .unwrap(),
            0.0015,
        );
        for start in [0usize, 80, 81, 161] {
            metric(
                &format!("{label}.encoded.slice_{start}"),
                &encoded.narrow(2, start, 64).unwrap(),
                &oracle
                    .get((1, 256, 64), &format!("{label}.encoded.slice_{start}"))
                    .unwrap(),
                0.0005,
            );
        }

        let decoded = model
            .decode_audio_with_chunk_noises(
                &latents,
                policy,
                parameters,
                &chunk_decode_noises(dim, token_scale, false, &device),
            )
            .unwrap();
        metric(
            &format!("{label}.decoded"),
            &decoded,
            &oracle
                .get((1, 2, LATENTS * RATIO), &format!("{label}.decoded"))
                .unwrap(),
            0.0015,
        );
        for start in [0usize, 456_704, 460_800, 917_504] {
            metric(
                &format!("{label}.decoded.slice_{start}"),
                &decoded.narrow(2, start, RATIO).unwrap(),
                &oracle
                    .get((1, 2, RATIO), &format!("{label}.decoded.slice_{start}"))
                    .unwrap(),
                0.0005,
            );
        }

        let decoded_chunked_zero = model
            .decode_audio_with_chunk_noises(
                &latents,
                policy,
                parameters,
                &chunk_decode_noises(dim, token_scale, true, &device),
            )
            .unwrap();
        let (direct_regularization, direct_tokens) = direct_zero_noises(label, dim, &device);
        let decoded_direct_zero = model
            .decode_with_noise(
                &latents,
                None,
                Some(&direct_regularization),
                Some(&[direct_tokens]),
            )
            .unwrap();
        for (suffix, actual) in [
            ("decoded_chunked_zero", &decoded_chunked_zero),
            ("decoded_direct_zero", &decoded_direct_zero),
        ] {
            let expected_samples = if label == "same_s" && suffix == "decoded_direct_zero" {
                (LATENTS + 1) * RATIO
            } else {
                LATENTS * RATIO
            };
            metric(
                &format!("{label}.{suffix}"),
                actual,
                &oracle
                    .get((1, 2, expected_samples), &format!("{label}.{suffix}"))
                    .unwrap(),
                0.0015,
            );
        }
        for boundary in [112 * RATIO, 113 * RATIO] {
            let (chunked_spectral, jump) = spectral_discontinuity(&decoded_chunked_zero, boundary);
            let (direct_spectral, _) = spectral_discontinuity(&decoded_direct_zero, boundary);
            eprintln!(
                "{label} boundary={boundary}: chunked_logmag_l1={chunked_spectral:.9} \
                 direct_logmag_l1={direct_spectral:.9} jump={jump:.9}"
            );
            assert!(
                chunked_spectral <= direct_spectral + 0.003,
                "{label} boundary {boundary} adds a spectral seam"
            );
            assert!(jump <= 0.03, "{label} boundary {boundary} jump {jump}");
        }

        let odd_parameters = SameChunkingParameters {
            chunk_size: 7,
            overlap: 3,
        };
        let edge_latents = 19usize;
        let batch_audio = portable(&[2, 2, edge_latents * RATIO], 902, 0.25, &device);
        let batch_latents = portable(&[2, 256, edge_latents], 903, 0.125, &device);
        let mut edge_rng =
            candle_audio_stable_audio_3::same::SameNoiseRng::seeded(SEED.wrapping_add(1));
        let batch_encoded = model
            .encode_audio_with_rng(
                &batch_audio,
                SameChunkingPolicy::full_model_encode(true),
                odd_parameters,
                &mut edge_rng,
            )
            .unwrap();
        assert_eq!(batch_encoded.dims3().unwrap(), (2, 256, edge_latents));
        let batch_decoded = model
            .decode_audio_with_rng(
                &batch_latents,
                SameChunkingPolicy::full_model_decode(true, None),
                odd_parameters,
                &mut edge_rng,
            )
            .unwrap();
        assert_eq!(batch_decoded.dims3().unwrap(), (2, 2, edge_latents * RATIO));
        let cropped =
            SameAutoencoder::crop_valid_prefix(&batch_decoded, edge_latents * RATIO - 13).unwrap();
        assert_eq!(cropped.dims3().unwrap(), (2, 2, edge_latents * RATIO - 13));
        assert!(
            SameAutoencoder::crop_valid_prefix(&batch_decoded, edge_latents * RATIO + 1).is_err()
        );

        let misaligned = portable(&[2, 2, edge_latents * RATIO + 1], 904, 0.25, &device);
        assert!(model
            .encode_audio_with_rng(
                &misaligned,
                SameChunkingPolicy::standalone(true),
                odd_parameters,
                &mut edge_rng,
            )
            .is_err());

        let exact_chunk_audio = portable(&[2, 2, 7 * RATIO], 905, 0.25, &device);
        assert_eq!(
            model
                .encode_audio_with_rng(
                    &exact_chunk_audio,
                    SameChunkingPolicy::standalone(true),
                    odd_parameters,
                    &mut edge_rng,
                )
                .unwrap()
                .dims3()
                .unwrap(),
            (2, 256, 7)
        );
        let exact_chunk_latents = portable(&[2, 256, 7], 906, 0.125, &device);
        assert_eq!(
            model
                .decode_audio_with_rng(
                    &exact_chunk_latents,
                    SameChunkingPolicy::full_model_decode(false, Some(true)),
                    odd_parameters,
                    &mut edge_rng,
                )
                .unwrap()
                .dims3()
                .unwrap(),
            (2, 2, 7 * RATIO)
        );

        let missing_regularization = (0..4)
            .map(|_| SameDecodeChunkNoise {
                regularization_noise: None,
                mask_noises: Vec::new(),
            })
            .collect::<Vec<_>>();
        assert!(model
            .decode_audio_with_chunk_noises(
                &batch_latents,
                SameChunkingPolicy::standalone(true),
                odd_parameters,
                &missing_regularization,
            )
            .is_err());
    }
}

#[test]
#[ignore = "requires one pinned SAME snapshot and a fresh process for one resource case"]
fn chunked_same_resource_probe() {
    let case =
        std::env::var("SA3_CHUNKED_CASE").expect("SA3_CHUNKED_CASE must be same_s or same_l");
    let (snapshot_env, dim) = match case.as_str() {
        "same_s" => ("SA3_SAME_S_SNAPSHOT", 768usize),
        "same_l" => ("SA3_SAME_L_SNAPSHOT", 1536usize),
        _ => panic!("SA3_CHUNKED_CASE must be same_s or same_l"),
    };
    let enabled = match std::env::var("SA3_CHUNKED_RESOURCE_MODE")
        .expect("SA3_CHUNKED_RESOURCE_MODE must be direct or chunked")
        .as_str()
    {
        "direct" => false,
        "chunked" => true,
        _ => panic!("SA3_CHUNKED_RESOURCE_MODE must be direct or chunked"),
    };
    let operation = std::env::var("SA3_CHUNKED_RESOURCE_OPERATION")
        .expect("SA3_CHUNKED_RESOURCE_OPERATION must be encode or decode");
    assert!(
        matches!(operation.as_str(), "encode" | "decode"),
        "SA3_CHUNKED_RESOURCE_OPERATION must be encode or decode"
    );
    let latent_len = std::env::var("SA3_CHUNKED_RESOURCE_LATENTS")
        .map(|value| {
            value
                .parse::<usize>()
                .expect("SA3_CHUNKED_RESOURCE_LATENTS must be usize")
        })
        .unwrap_or(1292);
    assert!(
        latent_len >= CHUNK,
        "resource case must execute the chunk scaffold"
    );
    let device = test_device();
    let layout = SnapshotLayout::from_dir(&snapshot(snapshot_env)).unwrap();
    let load_started = Instant::now();
    let model = SameAutoencoder::load(
        layout.config.autoencoder(),
        layout.mmap_builders(DType::F32, &device).unwrap(),
    )
    .unwrap();
    device.synchronize().unwrap();
    let load_seconds = load_started.elapsed().as_secs_f64();
    let load_peak_rss = candle_audio_stable_audio_3::candle_audio::harness::peak_rss_bytes();

    let policy = SameChunkingPolicy::standalone(enabled);
    let parameters = SameChunkingParameters::default();
    let warm_latents = portable(&[1, 256, 4], 700, 0.125, &device);
    let warm_audio = portable(&[1, 2, 4 * RATIO], 701, 0.25, &device);
    let mut warm_rng = candle_audio_stable_audio_3::same::SameNoiseRng::seeded(SEED);
    model
        .encode_audio_with_rng(
            &warm_audio,
            SameChunkingPolicy::standalone(false),
            parameters,
            &mut warm_rng,
        )
        .unwrap();
    model
        .decode_audio_with_rng(
            &warm_latents,
            SameChunkingPolicy::standalone(false),
            parameters,
            &mut warm_rng,
        )
        .unwrap();
    device.synchronize().unwrap();
    let load_device_bytes = device_allocated_bytes(&device);

    let mut rng = candle_audio_stable_audio_3::same::SameNoiseRng::seeded(SEED);
    let audio =
        (operation == "encode").then(|| portable(&[1, 2, latent_len * RATIO], 900, 0.25, &device));
    let latents =
        (operation == "decode").then(|| portable(&[1, 256, latent_len], 901, 0.125, &device));
    device.synchronize().unwrap();
    let started = Instant::now();
    let (output, peak_device_bytes) = with_peak_device_bytes(&device, || {
        let output = if operation == "encode" {
            model
                .encode_audio_with_rng(audio.as_ref().unwrap(), policy, parameters, &mut rng)
                .unwrap()
        } else {
            model
                .decode_audio_with_rng(latents.as_ref().unwrap(), policy, parameters, &mut rng)
                .unwrap()
        };
        device.synchronize().unwrap();
        output
    });
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let peak_rss = candle_audio_stable_audio_3::candle_audio::harness::peak_rss_bytes();
    let expected_shape = if operation == "encode" {
        (1, 256, latent_len)
    } else {
        (1, 2, latent_len * RATIO)
    };
    assert_eq!(output.dims3().unwrap(), expected_shape);
    let checksum = output
        .to_dtype(DType::F32)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let mode = if enabled { "chunked" } else { "direct" };
    eprintln!(
        "SA3_CHUNKED_RESOURCE case={case} mode={mode} operation={operation} \
         device={device:?} dtype=F32 latent_len={latent_len} samples={} dim={dim} \
         chunk_size=128 overlap=32 starts={} processed_latents={} \
         load_seconds={load_seconds:.6} load_peak_rss_bytes={load_peak_rss:?} \
         load_device_bytes={load_device_bytes:?} elapsed_seconds={elapsed_seconds:.6} \
         peak_rss_bytes={peak_rss:?} peak_device_bytes={peak_device_bytes:?} \
         output_shape={expected_shape:?} checksum={checksum:.9}",
        latent_len * RATIO,
        if enabled {
            candle_audio_stable_audio_3::same::SameChunkPlan::build(latent_len, true, parameters)
                .unwrap()
                .starts
                .len()
        } else {
            1
        },
        if enabled {
            candle_audio_stable_audio_3::same::SameChunkPlan::build(latent_len, true, parameters)
                .unwrap()
                .starts
                .len()
                * CHUNK
        } else {
            latent_len
        },
    );
}
