//! SAME-S whole-stage and structural parity gates.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_audio_stable_audio_3::candle_audio::candle_core::{
    DType, Device, Result as CandleResult, Shape, Tensor,
};
use candle_audio_stable_audio_3::candle_audio::dsp::{hann_window, stft};
use candle_audio_stable_audio_3::config::StableAudioConfig;
use candle_audio_stable_audio_3::same::{
    BandLayerTrace, SameAutoencoder, SameNoiseKind, SameNoiseRng,
};
use candle_audio_stable_audio_3::weights::{
    map_weight_key, safetensors_keys, SnapshotKind, SnapshotLayout, StableAudioVarBuilders,
    WeightSection,
};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};

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

fn mmap(path: &Path, device: &Device) -> VarBuilder<'static> {
    mmap_many(&[path.to_path_buf()], device)
}

fn mmap_many(paths: &[PathBuf], device: &Device) -> VarBuilder<'static> {
    // Safety: tests require immutable, hash-pinned artifacts for the full process lifetime.
    unsafe { VarBuilder::from_mmaped_safetensors(paths, DType::F32, device).unwrap() }
}

fn same_l_fixture(device: &Device) -> VarBuilder<'static> {
    let root = std::env::var_os("SA3_SAME_L_REFERENCE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../docs/migration/sa3-same-l-reference")
        });
    mmap_many(
        &[
            root.join("same-l.safetensors"),
            root.join("same-l-extended.safetensors"),
            root.join("same-l-outputs-f16.safetensors"),
        ],
        device,
    )
}

fn same_fixture(name: &str, device: &Device) -> VarBuilder<'static> {
    mmap(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../docs/migration/sa3-same-s-reference")
            .join(name),
        device,
    )
}

struct TrackingBackend {
    component: &'static str,
    inner: VarBuilder<'static>,
    consumed: Arc<Mutex<BTreeSet<String>>>,
}

impl SimpleBackend for TrackingBackend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        hints: Init,
        _dtype: DType,
        _device: &Device,
    ) -> CandleResult<Tensor> {
        self.consumed
            .lock()
            .unwrap()
            .insert(format!("{}.{name}", self.component));
        self.inner
            .get_with_hints(shape, name, hints)
            .map_err(|error| {
                candle_audio_stable_audio_3::candle_audio::candle_core::Error::Msg(format!(
                    "{}.{name}: {error}",
                    self.component
                ))
            })
    }

    fn get_unchecked(&self, name: &str, _dtype: DType, _device: &Device) -> CandleResult<Tensor> {
        self.consumed
            .lock()
            .unwrap()
            .insert(format!("{}.{name}", self.component));
        self.inner.get_unchecked(name).map_err(|error| {
            candle_audio_stable_audio_3::candle_audio::candle_core::Error::Msg(format!(
                "{}.{name}: {error}",
                self.component
            ))
        })
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.contains_tensor(name)
    }
}

fn tracked_builder(
    component: &'static str,
    inner: VarBuilder<'static>,
    consumed: &Arc<Mutex<BTreeSet<String>>>,
    device: &Device,
) -> VarBuilder<'static> {
    VarBuilder::from_backend(
        Box::new(TrackingBackend {
            component,
            inner,
            consumed: consumed.clone(),
        }),
        DType::F32,
        device.clone(),
    )
}

fn load_with_consumption_audit(
    layout: &SnapshotLayout,
    device: &Device,
) -> (SameAutoencoder, BTreeSet<String>) {
    let inner = layout.mmap_builders(DType::F32, device).unwrap();
    let consumed = Arc::new(Mutex::new(BTreeSet::new()));
    let builders = StableAudioVarBuilders {
        encoder: tracked_builder("encoder", inner.encoder, &consumed, device),
        decoder: tracked_builder("decoder", inner.decoder, &consumed, device),
        bottleneck: tracked_builder("bottleneck", inner.bottleneck, &consumed, device),
        dit: None,
        conditioner: None,
        text_encoder: None,
    };
    let model = SameAutoencoder::load(layout.config.autoencoder(), builders).unwrap();
    let consumed = consumed.lock().unwrap().clone();
    (model, consumed)
}

fn expected_autoencoder_weights(layout: &SnapshotLayout) -> BTreeSet<String> {
    safetensors_keys(&layout.weights_path)
        .unwrap()
        .into_iter()
        .filter_map(|key| {
            let mapped = map_weight_key(layout.kind, &key)?;
            let component = match mapped.section {
                WeightSection::Encoder => "encoder",
                WeightSection::Decoder => "decoder",
                WeightSection::Bottleneck => "bottleneck",
                WeightSection::Dit | WeightSection::Conditioner => return None,
            };
            Some(format!("{component}.{}", mapped.local_key))
        })
        .collect()
}

fn portable_noise(shape: &[usize], stream: u64, scale: f32, device: &Device) -> Tensor {
    let count = shape.iter().product();
    let values = (0..count)
        .map(|index| {
            let bits = (index as u64)
                .wrapping_mul(1_664_525)
                .wrapping_add((14_539 + stream).wrapping_mul(1_013_904_223))
                & 0xffff_ffff;
            ((bits as f64 / 2_147_483_648.0 - 1.0) as f32) * scale
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, shape.to_vec(), device).unwrap()
}

fn metric(name: &str, actual: &Tensor, expected: &Tensor, max_abs_limit: f32) -> f32 {
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
    let mut dd = 0f64;
    let mut max_abs = 0f32;
    for (&a, &b) in actual.iter().zip(&expected) {
        dot += a as f64 * b as f64;
        aa += (a as f64).powi(2);
        bb += (b as f64).powi(2);
        dd += ((a - b) as f64).powi(2);
        max_abs = max_abs.max((a - b).abs());
    }
    let cosine = dot / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE);
    let relative_l2 = (dd / bb.max(f64::MIN_POSITIVE)).sqrt();
    let snr_db = -20.0 * relative_l2.log10();
    eprintln!(
        "{name}: cosine={cosine:.9}, max_abs={max_abs:.9}, \
         relative_l2={relative_l2:.9}, snr_db={snr_db:.6}"
    );
    assert!(cosine >= 0.9999, "{name}: cosine {cosine}");
    assert!(
        max_abs <= max_abs_limit,
        "{name}: max_abs {max_abs} > {max_abs_limit}"
    );
    max_abs
}

fn assert_same_l_band_trace(
    name: &str,
    layers: &[BandLayerTrace],
    sequence_len: usize,
    expected_starts: &[usize],
) {
    assert_eq!(
        layers.len(),
        12,
        "{name}: every SAME-L layer must be traced"
    );
    for (index, layer) in layers.iter().enumerate() {
        assert_eq!(layer.layer, index, "{name}: layer index");
        assert_eq!(layer.sequence_len, sequence_len, "{name}: sequence length");
        assert_eq!(
            layer
                .slices
                .iter()
                .map(|slice| slice.start)
                .collect::<Vec<_>>(),
            expected_starts,
            "{name}: required edge/segment/tile/midpoint boundaries"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn backend_sensitive_metric(
    name: &str,
    actual: &Tensor,
    expected: &Tensor,
    cosine_range: (f64, f64),
    max_abs_range: (f32, f32),
    relative_l2_range: (f64, f64),
    snr_db_range: (f64, f64),
) {
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
    let mut dd = 0f64;
    let mut max_abs = 0f32;
    for (&a, &b) in actual.iter().zip(&expected) {
        dot += a as f64 * b as f64;
        aa += (a as f64).powi(2);
        bb += (b as f64).powi(2);
        dd += ((a - b) as f64).powi(2);
        max_abs = max_abs.max((a - b).abs());
    }
    let cosine = dot / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE);
    let relative_l2 = (dd / bb.max(f64::MIN_POSITIVE)).sqrt();
    let snr_db = -20.0 * relative_l2.log10();
    eprintln!(
        "{name}: cosine={cosine:.9}, max_abs={max_abs:.9}, \
         relative_l2={relative_l2:.9}, snr_db={snr_db:.6}"
    );
    assert!(
        (cosine_range.0..=cosine_range.1).contains(&cosine),
        "{name}: cosine {cosine} outside {cosine_range:?}"
    );
    assert!(
        (max_abs_range.0..=max_abs_range.1).contains(&max_abs),
        "{name}: max_abs {max_abs} outside {max_abs_range:?}"
    );
    assert!(
        (relative_l2_range.0..=relative_l2_range.1).contains(&relative_l2),
        "{name}: relative_l2 {relative_l2} outside {relative_l2_range:?}"
    );
    assert!(
        (snr_db_range.0..=snr_db_range.1).contains(&snr_db),
        "{name}: snr_db {snr_db} outside {snr_db_range:?}"
    );
}

#[test]
#[ignore = "requires pinned SAME-L/medium snapshots and the generated sc-14539 artifact"]
fn same_l_short_standalone_and_embedded_match_every_band_layer() {
    let device = test_device();
    let fixture = same_l_fixture(&device);
    let selected = std::env::var("SA3_SAME_L_CASE").ok();
    assert!(
        selected
            .as_deref()
            .is_none_or(|value| matches!(value, "standalone" | "embedded")),
        "SA3_SAME_L_CASE must be standalone or embedded"
    );
    for (env, prefix) in [
        ("SA3_SAME_L_SNAPSHOT", "standalone.short"),
        ("SA3_MEDIUM_SNAPSHOT", "embedded.short"),
    ] {
        if selected.as_deref().is_some_and(|selected| {
            (selected == "standalone" && env != "SA3_SAME_L_SNAPSHOT")
                || (selected == "embedded" && env != "SA3_MEDIUM_SNAPSHOT")
        }) {
            continue;
        }
        let layout = SnapshotLayout::from_dir(&snapshot(env)).unwrap();
        let (model, consumed) = load_with_consumption_audit(&layout, &device);
        assert_eq!(
            consumed,
            expected_autoencoder_weights(&layout),
            "{env} must consume all and only its 472 autoencoder tensors, including 24 RoPE buffers"
        );
        assert_eq!(consumed.len(), 472, "{env}");

        let audio = portable_noise(&[1, 2, 16_384], 100, 0.25, &device);
        let encoder_noise = portable_noise(&[1, 4, 1536], 0, 0.001, &device);
        let (latents, encoder_trace) = model
            .encode_with_noise_and_trace(&audio, None, &[encoder_noise])
            .unwrap();
        metric(
            &format!("{prefix}.latents"),
            &latents,
            &fixture
                .get((1, 256, 4), &format!("{prefix}.latents.slice_0"))
                .unwrap(),
            0.001,
        );
        metric(
            &format!("{prefix}.whole_latents"),
            &latents,
            &fixture
                .get((1, 256, 4), &format!("{prefix}.latents"))
                .unwrap(),
            0.1,
        );
        let encoder = &encoder_trace.stages[0];
        assert_eq!(
            encoder.layout,
            candle_audio_stable_audio_3::same::SameAttentionSchedule::Band {
                latent_left: 1,
                latent_right: 1,
                query_tile: 1024,
            }
        );
        assert!(encoder.block_outputs.is_empty());
        assert_eq!(encoder.band_layers.len(), 12);
        for layer in &encoder.band_layers {
            for slice in &layer.slices {
                metric(
                    &format!(
                        "{prefix}.encoder.block_{}.slice_{}",
                        layer.layer, slice.start
                    ),
                    &slice.values,
                    &fixture
                        .get(
                            slice.values.shape(),
                            &format!(
                                "{prefix}.encoder.block_{}.slice_{}",
                                layer.layer, slice.start
                            ),
                        )
                        .unwrap(),
                    0.001,
                );
            }
        }

        let regularization = portable_noise(&[1, 256, 4], 1, 1.0, &device);
        let decoder_noise = portable_noise(&[1, 64, 1536], 2, 0.1, &device);
        let (decoded, decoder_trace) = model
            .decode_with_trace(
                &latents,
                None,
                Some(&regularization),
                Some(&[decoder_noise]),
            )
            .unwrap();
        let decoder = &decoder_trace.stages[0];
        assert!(decoder.block_outputs.is_empty());
        assert_eq!(decoder.band_layers.len(), 12);
        for layer in &decoder.band_layers {
            for slice in &layer.slices {
                metric(
                    &format!(
                        "{prefix}.decoder.block_{}.slice_{}",
                        layer.layer, slice.start
                    ),
                    &slice.values,
                    &fixture
                        .get(
                            slice.values.shape(),
                            &format!(
                                "{prefix}.decoder.block_{}.slice_{}",
                                layer.layer, slice.start
                            ),
                        )
                        .unwrap(),
                    0.001,
                );
            }
        }
        for start in [0, 12_288] {
            metric(
                &format!("{prefix}.decoded.slice_{start}"),
                &decoded.narrow(2, start, 4096).unwrap(),
                &fixture
                    .get((1, 2, 4096), &format!("{prefix}.decoded.slice_{start}"))
                    .unwrap(),
                0.0001,
            );
        }
        metric(
            &format!("{prefix}.whole_decoded"),
            &decoded,
            &fixture
                .get((1, 2, 16_384), &format!("{prefix}.decoded"))
                .unwrap(),
            0.001,
        );
    }
}

#[test]
#[ignore = "requires the pinned SAME-L snapshot and generated 10s/120s sc-14539 artifact"]
fn same_l_long_durations_match_compact_band_boundaries() {
    let device = test_device();
    let fixture = same_l_fixture(&device);
    let selected = std::env::var("SA3_SAME_L_CASE").ok();
    assert!(
        selected
            .as_deref()
            .is_none_or(|value| matches!(value, "standalone" | "embedded")),
        "SA3_SAME_L_CASE must be standalone or embedded"
    );
    let selected_duration = std::env::var("SA3_SAME_L_DURATION").ok();
    assert!(
        selected_duration
            .as_deref()
            .is_none_or(|value| matches!(value, "ten_seconds" | "long_120_seconds")),
        "SA3_SAME_L_DURATION must be ten_seconds or long_120_seconds"
    );
    for (env, snapshot_label) in [
        ("SA3_SAME_L_SNAPSHOT", "standalone"),
        ("SA3_MEDIUM_SNAPSHOT", "embedded"),
    ] {
        if selected.as_deref().is_some_and(|selected| {
            (selected == "standalone" && env != "SA3_SAME_L_SNAPSHOT")
                || (selected == "embedded" && env != "SA3_MEDIUM_SNAPSHOT")
        }) {
            continue;
        }
        let layout = SnapshotLayout::from_dir(&snapshot(env)).unwrap();
        let builders = layout.mmap_builders(DType::F32, &device).unwrap();
        let model = SameAutoencoder::load(layout.config.autoencoder(), builders).unwrap();
        for (duration, samples) in [
            ("ten_seconds", 441_000usize),
            ("long_120_seconds", 5_292_000usize),
        ] {
            if selected_duration
                .as_deref()
                .is_some_and(|selected| selected != duration)
            {
                continue;
            }
            let label = format!("{snapshot_label}.{duration}");
            let latent_len = samples.div_ceil(4096);
            let padded_samples = latent_len * 4096;
            let packed_len = latent_len * 17;
            let expected_starts: &[usize] = match duration {
                "ten_seconds" => &[0, 901, 1007, 1801],
                "long_120_seconds" => &[0, 1007, 10_965, 21_929],
                _ => unreachable!(),
            };
            let audio = portable_noise(&[1, 2, samples], 100, 0.25, &device);
            let encoder_noise = portable_noise(&[1, latent_len, 1536], 0, 0.001, &device);
            let (latents, encoder_trace) = model
                .encode_with_noise_and_trace(&audio, None, &[encoder_noise])
                .unwrap();
            metric(
                &format!("{label}.whole_latents"),
                &latents,
                &fixture
                    .get((1, 256, latent_len), &format!("{label}.latents"))
                    .unwrap(),
                0.1,
            );
            for start in [0, latent_len.saturating_sub(64)] {
                let width = latent_len.min(64);
                metric(
                    &format!("{label}.latents.slice_{start}"),
                    &latents.narrow(2, start, width).unwrap(),
                    &fixture
                        .get((1, 256, width), &format!("{label}.latents.slice_{start}"))
                        .unwrap(),
                    0.05,
                );
            }
            let encoder = &encoder_trace.stages[0];
            assert!(encoder.block_outputs.is_empty());
            assert_same_l_band_trace(
                &format!("{label}.encoder"),
                &encoder.band_layers,
                packed_len,
                expected_starts,
            );
            for layer in &encoder.band_layers {
                for slice in &layer.slices {
                    metric(
                        &format!(
                            "{label}.encoder.block_{}.slice_{}",
                            layer.layer, slice.start
                        ),
                        &slice.values,
                        &fixture
                            .get(
                                slice.values.shape(),
                                &format!(
                                    "{label}.encoder.block_{}.slice_{}",
                                    layer.layer, slice.start
                                ),
                            )
                            .unwrap(),
                        0.1,
                    );
                }
            }
            drop(encoder_trace);

            let regularization = portable_noise(&[1, 256, latent_len], 1, 1.0, &device);
            let decoder_noise = portable_noise(&[1, latent_len * 16, 1536], 2, 0.1, &device);
            let (decoded, decoder_trace) = model
                .decode_with_trace(
                    &latents,
                    None,
                    Some(&regularization),
                    Some(&[decoder_noise]),
                )
                .unwrap();
            let decoder = &decoder_trace.stages[0];
            assert!(decoder.block_outputs.is_empty());
            assert_same_l_band_trace(
                &format!("{label}.decoder"),
                &decoder.band_layers,
                packed_len,
                expected_starts,
            );
            for layer in &decoder.band_layers {
                for slice in &layer.slices {
                    metric(
                        &format!(
                            "{label}.decoder.block_{}.slice_{}",
                            layer.layer, slice.start
                        ),
                        &slice.values,
                        &fixture
                            .get(
                                slice.values.shape(),
                                &format!(
                                    "{label}.decoder.block_{}.slice_{}",
                                    layer.layer, slice.start
                                ),
                            )
                            .unwrap(),
                        0.1,
                    );
                }
            }
            let expected_decoded = fixture
                .get((1, 2, padded_samples), &format!("{label}.decoded"))
                .unwrap();
            metric(
                &format!("{label}.whole_decoded"),
                &decoded,
                &expected_decoded,
                0.001,
            );
            if duration == "ten_seconds" {
                let (snr_db, mrstft) = roundtrip_metrics(
                    &expected_decoded
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec3()
                        .unwrap(),
                    &decoded.to_dtype(DType::F32).unwrap().to_vec3().unwrap(),
                );
                eprintln!(
                    "{label}.whole_decoded_reference_quality: \
                     snr_db={snr_db:.6}, mrstft={mrstft:.9}"
                );
                assert!(
                    snr_db >= 70.0,
                    "{label}: reference SNR {snr_db} dB is below 70 dB"
                );
                assert!(
                    mrstft <= 0.08,
                    "{label}: reference MR-STFT {mrstft} exceeds 0.08"
                );
            }
            for start in [0, padded_samples - 4096] {
                metric(
                    &format!("{label}.decoded.slice_{start}"),
                    &decoded.narrow(2, start, 4096).unwrap(),
                    &fixture
                        .get((1, 2, 4096), &format!("{label}.decoded.slice_{start}"))
                        .unwrap(),
                    0.1,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires the pinned SAME-L snapshot and generated stride-7 sc-14539 artifact"]
fn same_l_variable_stride_padding_gather_and_noise_order_match_upstream() {
    let device = test_device();
    let fixture = same_l_fixture(&device);
    let layout = SnapshotLayout::from_dir(&snapshot("SA3_SAME_L_SNAPSHOT")).unwrap();
    let model = SameAutoencoder::load(
        layout.config.autoencoder(),
        layout.mmap_builders(DType::F32, &device).unwrap(),
    )
    .unwrap();
    let samples = 16_384;
    let stride = [7usize];
    let latent_len = 10;
    let padded_samples = 17_920;
    let audio = portable_noise(&[1, 2, samples], 100, 0.25, &device);
    let encoder_noise = portable_noise(&[1, latent_len, 1536], 0, 0.001, &device);
    let (latents, encoder_trace) = model
        .encode_with_noise_and_trace(&audio, Some(&stride), std::slice::from_ref(&encoder_noise))
        .unwrap();
    assert_eq!(latents.dims3().unwrap(), (1, 256, latent_len));
    assert_eq!(
        encoder_trace.stages[0].layout,
        candle_audio_stable_audio_3::same::SameAttentionSchedule::Band {
            latent_left: 1,
            latent_right: 1,
            query_tile: 1024,
        }
    );
    metric(
        "standalone.stride7.whole_latents",
        &latents,
        &fixture
            .get((1, 256, latent_len), "standalone.stride7.latents")
            .unwrap(),
        0.1,
    );
    let zero_encoder_noise = Tensor::zeros_like(&encoder_noise).unwrap();
    let without_encoder_noise = model
        .encode_with_noise(&audio, Some(&stride), &[zero_encoder_noise])
        .unwrap();
    assert!(
        max_abs_tensor(&latents, &without_encoder_noise) > 1e-6,
        "encoder token-noise mutation must affect band output"
    );

    let regularization = portable_noise(&[1, 256, latent_len], 1, 1.0, &device);
    let decoder_noise = portable_noise(&[1, latent_len * 7, 1536], 2, 0.1, &device);
    let decoded = model
        .decode_with_noise(
            &latents,
            Some(&stride),
            Some(&regularization),
            Some(std::slice::from_ref(&decoder_noise)),
        )
        .unwrap();
    assert_eq!(decoded.dims3().unwrap(), (1, 2, padded_samples));
    metric(
        "standalone.stride7.whole_decoded",
        &decoded,
        &fixture
            .get((1, 2, padded_samples), "standalone.stride7.decoded")
            .unwrap(),
        0.001,
    );
    let zero_regularization = Tensor::zeros_like(&regularization).unwrap();
    let without_regularization = model
        .decode_with_noise(
            &latents,
            Some(&stride),
            Some(&zero_regularization),
            Some(std::slice::from_ref(&decoder_noise)),
        )
        .unwrap();
    assert!(
        max_abs_tensor(&decoded, &without_regularization) > 1e-6,
        "SoftNorm noise mutation must affect band decode"
    );
    let zero_decoder_noise = Tensor::zeros_like(&decoder_noise).unwrap();
    let without_decoder_noise = model
        .decode_with_noise(
            &latents,
            Some(&stride),
            Some(&regularization),
            Some(&[zero_decoder_noise]),
        )
        .unwrap();
    assert!(
        max_abs_tensor(&decoded, &without_decoder_noise) > 1e-6,
        "decoder token-noise mutation must affect band output"
    );

    let mut rng = SameNoiseRng::capturing(14_539);
    let production_latents = model
        .encode_with_rng(&audio, Some(&stride), &mut rng)
        .unwrap();
    model
        .decode_with_rng(&production_latents, Some(&stride), &mut rng)
        .unwrap();
    let captures = rng.captures();
    assert_eq!(captures.len(), 3);
    assert_eq!(captures[0].kind, SameNoiseKind::EncoderTokens { stage: 0 });
    assert_eq!(captures[0].scale, 0.001);
    assert_eq!(captures[1].kind, SameNoiseKind::SoftNormRegularization);
    assert_eq!(captures[1].scale, 0.001);
    assert_eq!(captures[2].kind, SameNoiseKind::DecoderTokens { stage: 0 });
    assert_eq!(captures[2].scale, 0.1);
}

fn max_abs_tensor(left: &Tensor, right: &Tensor) -> f32 {
    (left - right)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

#[test]
#[ignore = "requires pinned SAME-L weights and executes a fresh-process long-duration roundtrip"]
fn same_l_long_duration_roundtrip_resource_probe() {
    const LITERAL_380_SECOND_SAMPLES: usize = 380 * 44_100;
    let samples = std::env::var("SA3_SAME_L_RESOURCE_SAMPLES")
        .map(|value| {
            value
                .parse()
                .expect("SA3_SAME_L_RESOURCE_SAMPLES must be usize")
        })
        .unwrap_or(LITERAL_380_SECOND_SAMPLES);
    let device = test_device();
    let layout = SnapshotLayout::from_dir(&snapshot("SA3_SAME_L_SNAPSHOT")).unwrap();
    let load_started = Instant::now();
    let model = SameAutoencoder::load(
        layout.config.autoencoder(),
        layout.mmap_builders(DType::F32, &device).unwrap(),
    )
    .unwrap();
    device.synchronize().unwrap();
    let load_seconds = load_started.elapsed().as_secs_f64();
    let load_peak_rss_bytes = candle_audio_stable_audio_3::candle_audio::harness::peak_rss_bytes();

    let warm_audio = portable_noise(&[1, 2, 16_384], 100, 0.25, &device);
    let mut warm_rng = SameNoiseRng::seeded(14_539);
    let warm_latents = model
        .encode_with_rng(&warm_audio, None, &mut warm_rng)
        .unwrap();
    model
        .decode_with_rng(&warm_latents, None, &mut warm_rng)
        .unwrap();
    device.synchronize().unwrap();

    let audio = portable_noise(&[1, 2, samples], 100, 0.25, &device);
    let mut rng = SameNoiseRng::seeded(14_539);
    let encode_started = Instant::now();
    let latents = model.encode_with_rng(&audio, None, &mut rng).unwrap();
    device.synchronize().unwrap();
    let encode_seconds = encode_started.elapsed().as_secs_f64();
    let latent_len = samples.div_ceil(4096);
    assert_eq!(latents.dims3().unwrap(), (1, 256, latent_len));

    let decode_started = Instant::now();
    let decoded = model.decode_with_rng(&latents, None, &mut rng).unwrap();
    device.synchronize().unwrap();
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    let peak_rss_bytes = candle_audio_stable_audio_3::candle_audio::harness::peak_rss_bytes();
    let checksum = decoded
        .to_dtype(DType::F32)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let decoded_values = decoded
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let pcm_sha256 =
        candle_audio_stable_audio_3::candle_audio::harness::pcm_sha256(&decoded_values);
    let padded_samples = latent_len * 4096;
    assert_eq!(decoded.dims3().unwrap(), (1, 2, padded_samples));
    eprintln!(
        "SA3_SAME_L_RESOURCE device={device:?} dtype=F32 input_samples={samples} \
         padded_samples={padded_samples} latent_len={latent_len} packed_len={} query_tile=1024 \
         load_seconds={load_seconds:.6} load_peak_rss_bytes={load_peak_rss_bytes:?} \
         encode_seconds={encode_seconds:.6} decode_seconds={decode_seconds:.6} \
         peak_rss_bytes={peak_rss_bytes:?} checksum={checksum:.9} pcm_sha256={pcm_sha256}",
        latent_len * 17,
    );
}

#[test]
#[ignore = "requires the explicitly provisioned pinned SAME-S snapshot"]
fn existing_whole_stage_reference_matches_encoder() {
    let device = test_device();
    let layout = SnapshotLayout::from_dir(&snapshot("SA3_SAME_S_SNAPSHOT")).unwrap();
    let builders = layout.mmap_builders(DType::F32, &device).unwrap();
    let model = SameAutoencoder::load(layout.config.autoencoder(), builders).unwrap();
    let oracle = mmap(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../docs/migration/sa3-reference/same-s-same.safetensors"),
        &device,
    );
    let audio = oracle.get((1, 2, 16_384), "audio_input").unwrap();
    let actual = model.encode(&audio).unwrap();
    metric(
        "same_s_encoder",
        &actual,
        &oracle.get((1, 256, 4), "latents").unwrap(),
        2e-4,
    );
}

#[test]
#[ignore = "requires the explicitly provisioned pinned SAME-S snapshot"]
fn structural_oracle_locks_all_blocks_midpoint_selection_and_noise() {
    let device = test_device();
    let layout = SnapshotLayout::from_dir(&snapshot("SA3_SAME_S_SNAPSHOT")).unwrap();
    let model = SameAutoencoder::load(
        layout.config.autoencoder(),
        layout.mmap_builders(DType::F32, &device).unwrap(),
    )
    .unwrap();
    let oracle = same_fixture("oracle.safetensors", &device);
    let audio = oracle.get((1, 2, 16_384), "audio").unwrap();
    let (latents, encoder) = model.encode_with_trace(&audio, None).unwrap();
    metric(
        "latents",
        &latents,
        &oracle.get((1, 256, 4), "latents").unwrap(),
        2e-4,
    );
    let encoder = &encoder.stages[0];
    metric(
        "encoder mapped sequence",
        &encoder.mapped_sequence,
        &oracle.get((1, 64, 768), "encoder.mapped_sequence").unwrap(),
        2e-5,
    );
    metric(
        "encoder folded input",
        &encoder.folded_input,
        &oracle.get((2, 34, 768), "encoder.folded_input").unwrap(),
        2e-5,
    );
    assert_eq!(encoder.block_outputs.len(), 6);
    for (index, output) in encoder.block_outputs.iter().enumerate() {
        metric(
            &format!("encoder block {index}"),
            output,
            &oracle
                .get((1, 68, 768), &format!("encoder.block_{index}"))
                .unwrap(),
            1.2e-3,
        );
    }
    metric(
        "encoder selected segments",
        &encoder.selected_segments,
        &oracle
            .get((1, 4, 768), "encoder.selected_segments")
            .unwrap(),
        8e-4,
    );
    metric(
        "encoder output",
        &encoder.output,
        &oracle.get((1, 768, 4), "encoder.output").unwrap(),
        2e-4,
    );

    let regularization_noise = oracle.get((1, 256, 4), "regularization_noise").unwrap();
    let mask_noise = oracle.get((1, 64, 768), "decoder_mask_noise").unwrap();
    let (decoded, decoder) = model
        .decode_with_trace(
            &latents,
            None,
            Some(&regularization_noise),
            Some(std::slice::from_ref(&mask_noise)),
        )
        .unwrap();
    let decoder = &decoder.stages[0];
    metric(
        "decoder mapped sequence",
        &decoder.mapped_sequence,
        &oracle.get((1, 4, 768), "decoder.mapped_sequence").unwrap(),
        3e-4,
    );
    metric(
        "decoder folded input",
        &decoder.folded_input,
        &oracle.get((2, 34, 768), "decoder.folded_input").unwrap(),
        3e-4,
    );
    assert_eq!(decoder.block_outputs.len(), 6);
    for (index, output) in decoder.block_outputs.iter().enumerate() {
        metric(
            &format!("decoder block {index}"),
            output,
            &oracle
                .get((1, 68, 768), &format!("decoder.block_{index}"))
                .unwrap(),
            4e-4,
        );
    }
    metric(
        "decoder selected segments",
        &decoder.selected_segments,
        &oracle
            .get((1, 64, 768), "decoder.selected_segments")
            .unwrap(),
        4e-4,
    );
    metric(
        "decoder resampling output",
        &decoder.output,
        &oracle.get((1, 512, 64), "decoder.output").unwrap(),
        5e-4,
    );
    metric(
        "decoded",
        &decoded,
        &oracle.get((1, 2, 16_384), "decoded").unwrap(),
        5e-4,
    );

    let zeros_regularization = Tensor::zeros_like(&regularization_noise).unwrap();
    let no_regularization = model
        .decode_with_noise(
            &latents,
            None,
            Some(&zeros_regularization),
            Some(std::slice::from_ref(&mask_noise)),
        )
        .unwrap();
    let zeros_mask = Tensor::zeros_like(&mask_noise).unwrap();
    let no_mask = model
        .decode_with_noise(
            &latents,
            None,
            Some(&regularization_noise),
            Some(std::slice::from_ref(&zeros_mask)),
        )
        .unwrap();
    assert!(
        max_abs(&decoded, &no_regularization) > 1e-5,
        "dropping SoftNorm evaluation noise must be observable"
    );
    assert!(
        max_abs(&decoded, &no_mask) > 1e-5,
        "dropping decoder mask noise must be observable"
    );
}

#[test]
#[ignore = "requires the explicitly provisioned pinned SAME-S snapshot"]
fn frozen_stride_eight_oracle_locks_every_resampling_seam() {
    let device = test_device();
    let layout = SnapshotLayout::from_dir(&snapshot("SA3_SAME_S_SNAPSHOT")).unwrap();
    let model = SameAutoencoder::load(
        layout.config.autoencoder(),
        layout.mmap_builders(DType::F32, &device).unwrap(),
    )
    .unwrap();
    let oracle = same_fixture("oracle.safetensors", &device);
    let audio = oracle.get((1, 2, 16_384), "audio").unwrap();
    let (latents, encoder) = model.encode_with_trace(&audio, Some(&[8])).unwrap();
    metric(
        "stride8 latents",
        &latents,
        &oracle.get((1, 256, 8), "stride8.latents").unwrap(),
        2.5e-3,
    );
    let encoder = &encoder.stages[0];
    for (name, actual, shape) in [
        (
            "mapped_sequence",
            &encoder.mapped_sequence,
            vec![1, 64, 768],
        ),
        ("folded_input", &encoder.folded_input, vec![2, 36, 768]),
        ("expanded_tokens", &encoder.expanded_tokens, vec![1, 8, 768]),
        (
            "selected_segments",
            &encoder.selected_segments,
            vec![1, 8, 768],
        ),
        ("output", &encoder.output, vec![1, 768, 8]),
    ] {
        metric(
            &format!("stride8 encoder {name}"),
            actual,
            &oracle
                .get(shape, &format!("stride8.encoder.{name}"))
                .unwrap(),
            1.5e-3,
        );
    }
    assert_eq!(encoder.block_outputs.len(), 6);
    for (index, output) in encoder.block_outputs.iter().enumerate() {
        metric(
            &format!("stride8 encoder block {index}"),
            output,
            &oracle
                .get((1, 72, 768), &format!("stride8.encoder.block_{index}"))
                .unwrap(),
            1.5e-3,
        );
    }

    let regularization = oracle
        .get((1, 256, 8), "stride8.regularization_noise")
        .unwrap();
    let mask = oracle
        .get((1, 64, 768), "stride8.decoder_mask_noise")
        .unwrap();
    // Isolate decoder numerical parity from the accumulated encoder delta. The
    // stride-eight decoder is substantially more sensitive to its latent input
    // than the default stride-sixteen path.
    let decoder_latents = oracle.get((1, 256, 8), "stride8.latents").unwrap();
    let (decoded, decoder) = model
        .decode_with_trace(
            &decoder_latents,
            Some(&[8]),
            Some(&regularization),
            Some(std::slice::from_ref(&mask)),
        )
        .unwrap();
    let decoder = &decoder.stages[0];
    for (name, actual, shape) in [
        ("mapped_sequence", &decoder.mapped_sequence, vec![1, 8, 768]),
        ("folded_input", &decoder.folded_input, vec![2, 36, 768]),
        (
            "expanded_tokens",
            &decoder.expanded_tokens,
            vec![1, 64, 768],
        ),
    ] {
        metric(
            &format!("stride8 decoder {name}"),
            actual,
            &oracle
                .get(shape, &format!("stride8.decoder.{name}"))
                .unwrap(),
            1.5e-3,
        );
    }
    let controlled_inputs = (0..6)
        .map(|index| {
            let folded_batch = if index < 3 { 2 } else { 3 };
            oracle
                .get(
                    (folded_batch, 36, 768),
                    &format!("stride8.decoder.block_{index}_input"),
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    let controlled = model
        .decode_stage_with_controlled_block_inputs(0, 1, Some(8), &controlled_inputs)
        .unwrap();
    assert_eq!(controlled.block_outputs.len(), 6);
    for (index, output) in controlled.block_outputs.iter().enumerate() {
        metric(
            &format!("stride8 controlled decoder block {index}"),
            output,
            &oracle
                .get((1, 72, 768), &format!("stride8.decoder.block_{index}"))
                .unwrap(),
            1.5e-3,
        );
    }
    metric(
        "stride8 controlled decoder selected_segments",
        &controlled.selected_segments,
        &oracle
            .get((1, 64, 768), "stride8.decoder.selected_segments")
            .unwrap(),
        1.5e-3,
    );
    metric(
        "stride8 controlled decoder output",
        &controlled.output,
        &oracle.get((1, 512, 64), "stride8.decoder.output").unwrap(),
        1.5e-3,
    );

    let block2 = oracle.get((1, 72, 768), "stride8.decoder.block_2").unwrap();
    let repeated = Tensor::cat(
        &[
            &block2.narrow(1, 0, 18).unwrap(),
            &block2,
            &block2.narrow(1, 54, 18).unwrap(),
        ],
        1,
    )
    .unwrap()
    .reshape((3, 36, 768))
    .unwrap();
    metric(
        "stride8 midpoint-18 edge repeat and refold",
        &repeated,
        &controlled_inputs[3],
        0.0,
    );
    assert_ne!(
        17 + 72 + 17,
        3 * 36,
        "the default midpoint-17 mutation must not satisfy stride8 folding"
    );

    let block5 = oracle.get((1, 72, 768), "stride8.decoder.block_5").unwrap();
    let gather_final_eight = block5
        .reshape((8, 9, 768))
        .unwrap()
        .narrow(1, 1, 8)
        .unwrap()
        .reshape((1, 64, 768))
        .unwrap();
    metric(
        "stride8 gather final eight",
        &gather_final_eight,
        &controlled.selected_segments,
        1.5e-3,
    );
    let gather_first_eight = block5
        .reshape((8, 9, 768))
        .unwrap()
        .narrow(1, 0, 8)
        .unwrap()
        .reshape((1, 64, 768))
        .unwrap();
    assert!(
        max_abs(&gather_first_eight, &controlled.selected_segments) > 1e-2,
        "gathering the first eight tokens must fail the frozen final-eight oracle"
    );
    assert_eq!(
        decoder.expanded_tokens.dim(1).unwrap(),
        8 * 8,
        "stride8 must expand one learned token to eight tokens per latent"
    );
    assert!(
        model
            .decode_stage_with_controlled_block_inputs(0, 1, Some(16), &controlled_inputs)
            .is_err(),
        "ignoring stride8 must fail the frozen 36-position fold/RoPE input"
    );
    let bypassed = decoder.folded_input.reshape((1, 72, 768)).unwrap();
    assert!(
        max_abs(&bypassed, &controlled.block_outputs[0]) > 1e-2,
        "bypassing decoder block 0 must be observable"
    );
    let mut reordered_inputs = controlled_inputs.clone();
    reordered_inputs.swap(0, 1);
    let reordered = model
        .decode_stage_with_controlled_block_inputs(0, 1, Some(8), &reordered_inputs)
        .unwrap();
    assert!(
        max_abs(&controlled.block_outputs[0], &reordered.block_outputs[0]) > 1e-2,
        "reordering frozen decoder block inputs must be observable"
    );

    let perturbation = oracle
        .get((2, 36, 768), "stride8.perturbation_unit")
        .unwrap()
        .affine(3e-6, 0.0)
        .unwrap();
    let mut mutated_inputs = controlled_inputs.clone();
    mutated_inputs[1] = (&mutated_inputs[1] + perturbation).unwrap();
    let mutated = model
        .decode_stage_with_controlled_block_inputs(0, 1, Some(8), &mutated_inputs)
        .unwrap();
    assert!(
        max_abs(&controlled.block_outputs[1], &mutated.block_outputs[1]) > 1e-5,
        "controlled decoder oracle must detect the frozen post-block-0 perturbation"
    );

    let expected_decoded = oracle.get((1, 2, 16_384), "stride8.decoded").unwrap();
    backend_sensitive_metric(
        "frozen Torch stride8 1e-6 post-block-0 sensitivity",
        &oracle
            .get((1, 2, 16_384), "stride8.perturbation_1e-6.decoded")
            .unwrap(),
        &expected_decoded,
        (0.9725, 0.9727),
        (2.29, 2.30),
        (0.234, 0.235),
        (12.5, 12.7),
    );
    backend_sensitive_metric(
        "frozen Torch stride8 3e-6 post-block-0 sensitivity",
        &oracle
            .get((1, 2, 16_384), "stride8.perturbation_3e-6.decoded")
            .unwrap(),
        &expected_decoded,
        (0.9484, 0.9486),
        (3.66, 3.67),
        (0.331, 0.332),
        (9.5, 9.7),
    );

    let (decoded_repeat, _) = model
        .decode_with_trace(
            &decoder_latents,
            Some(&[8]),
            Some(&regularization),
            Some(std::slice::from_ref(&mask)),
        )
        .unwrap();
    assert_eq!(
        max_abs(&decoded, &decoded_repeat),
        0.0,
        "stride-eight decode must be deterministically repeatable"
    );
    backend_sensitive_metric(
        "stride8 production end-to-end decoded",
        &decoded,
        &expected_decoded,
        (0.95, 1.0),
        (0.0, 2.5),
        (0.0, 0.30),
        (10.5, f64::INFINITY),
    );
}

fn max_abs(left: &Tensor, right: &Tensor) -> f32 {
    left.broadcast_sub(right)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

#[test]
#[ignore = "requires the explicitly provisioned pinned SAME-S snapshot"]
fn production_rng_is_seeded_captured_ordered_scaled_and_mutation_sensitive() {
    let device = test_device();
    let layout = SnapshotLayout::from_dir(&snapshot("SA3_SAME_S_SNAPSHOT")).unwrap();
    let model = SameAutoencoder::load(
        layout.config.autoencoder(),
        layout.mmap_builders(DType::F32, &device).unwrap(),
    )
    .unwrap();
    let oracle = same_fixture("oracle.safetensors", &device);
    let latents = oracle.get((1, 256, 4), "latents").unwrap();

    let mut first_rng = SameNoiseRng::capturing(14538);
    let first = model
        .decode_with_rng(&latents, None, &mut first_rng)
        .unwrap();
    let captures = first_rng.captures();
    assert_eq!(captures.len(), 2);
    assert_eq!(captures[0].kind, SameNoiseKind::SoftNormRegularization);
    assert_eq!(captures[1].kind, SameNoiseKind::DecoderTokens { stage: 0 });
    assert_eq!(captures[0].scale, 1e-3);
    assert_eq!(captures[1].scale, 1e-2);
    assert_eq!(captures[0].unit.dims3().unwrap(), (1, 256, 4));
    assert_eq!(captures[1].unit.dims3().unwrap(), (4, 16, 768));

    let mut second_rng = SameNoiseRng::capturing(14538);
    let second = model
        .decode_with_rng(&latents, None, &mut second_rng)
        .unwrap();
    assert_eq!(max_abs(&first, &second), 0.0);
    assert_eq!(
        captures[0]
            .unit
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        second_rng.captures()[0]
            .unit
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    );
    assert_eq!(
        captures[1]
            .unit
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        second_rng.captures()[1]
            .unit
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    );

    let mask_scaled = captures[1]
        .unit
        .affine(captures[1].scale, 0.0)
        .unwrap()
        .reshape((1, 64, 768))
        .unwrap();
    let replay = model
        .decode_with_noise(
            &latents,
            None,
            Some(&captures[0].unit),
            Some(std::slice::from_ref(&mask_scaled)),
        )
        .unwrap();
    assert_eq!(
        max_abs(&first, &replay),
        0.0,
        "captured production draws must replay the exact output"
    );

    let zero_regularization = Tensor::zeros_like(&captures[0].unit).unwrap();
    let zero_mask = Tensor::zeros_like(&mask_scaled).unwrap();
    let zeroed = model
        .decode_with_noise(
            &latents,
            None,
            Some(&zero_regularization),
            Some(std::slice::from_ref(&zero_mask)),
        )
        .unwrap();
    assert!(max_abs(&first, &zeroed) > 1e-5);
    let regularization_zeroed = model
        .decode_with_noise(
            &latents,
            None,
            Some(&zero_regularization),
            Some(std::slice::from_ref(&mask_scaled)),
        )
        .unwrap();
    assert!(max_abs(&first, &regularization_zeroed) > 1e-5);
    let mask_zeroed = model
        .decode_with_noise(
            &latents,
            None,
            Some(&captures[0].unit),
            Some(std::slice::from_ref(&zero_mask)),
        )
        .unwrap();
    assert!(max_abs(&first, &mask_zeroed) > 1e-5);
    let half_mask = mask_scaled.affine(0.5, 0.0).unwrap();
    let rescaled = model
        .decode_with_noise(
            &latents,
            None,
            Some(&captures[0].unit),
            Some(std::slice::from_ref(&half_mask)),
        )
        .unwrap();
    assert!(max_abs(&first, &rescaled) > 1e-5);
    assert!(model
        .decode_with_noise(
            &latents,
            None,
            Some(&captures[1].unit),
            Some(std::slice::from_ref(&captures[0].unit)),
        )
        .is_err());
    let training = model
        .decode_with_noise_mode(
            &latents,
            None,
            Some(&captures[0].unit),
            Some(std::slice::from_ref(&mask_scaled)),
            true,
        )
        .unwrap();
    assert!(max_abs(&first, &training) > 1e-4);

    let mut disabled_config = layout.config.autoencoder().clone();
    disabled_config.bottleneck.config.noise_regularize = false;
    disabled_config.decoder.config.mask_noise = 0.0;
    let disabled = SameAutoencoder::load(
        &disabled_config,
        layout.mmap_builders(DType::F32, &device).unwrap(),
    )
    .unwrap()
    .decode(&latents)
    .unwrap();
    assert_eq!(
        max_abs(&disabled, &zeroed),
        0.0,
        "disabling both configured noises must equal explicit zero noise"
    );
}

#[test]
#[ignore = "requires explicitly provisioned standalone and embedded SAME-S snapshots"]
fn standalone_and_embedded_namespaces_match_and_extent_rules_hold() {
    let device = test_device();
    let standalone_layout = SnapshotLayout::from_dir(&snapshot("SA3_SAME_S_SNAPSHOT")).unwrap();
    let standalone = SameAutoencoder::load(
        standalone_layout.config.autoencoder(),
        standalone_layout
            .mmap_builders(DType::F32, &device)
            .unwrap(),
    )
    .unwrap();
    let embedded_layout = SnapshotLayout::from_dir(&snapshot("SA3_SMALL_MUSIC_SNAPSHOT")).unwrap();
    let embedded = SameAutoencoder::load(
        embedded_layout.config.autoencoder(),
        embedded_layout.mmap_builders(DType::F32, &device).unwrap(),
    )
    .unwrap();
    let oracle = same_fixture("oracle.safetensors", &device);
    let audio = oracle.get((1, 2, 16_384), "audio").unwrap();
    let standalone_latents = standalone.encode(&audio).unwrap();
    let embedded_latents = embedded.encode(&audio).unwrap();
    assert!(
        max_abs(&standalone_latents, &embedded_latents) < 1e-6,
        "standalone and embedded namespaces must route identical SAME-S weights"
    );
    let regularization = oracle.get((1, 256, 4), "regularization_noise").unwrap();
    let mask = oracle.get((1, 64, 768), "decoder_mask_noise").unwrap();
    let embedded_decoded = embedded
        .decode_with_noise(
            &embedded_latents,
            None,
            Some(&regularization),
            Some(std::slice::from_ref(&mask)),
        )
        .unwrap();
    metric(
        "embedded controlled-noise decoded",
        &embedded_decoded,
        &oracle.get((1, 2, 16_384), "decoded").unwrap(),
        5e-4,
    );

    let non_patch = Tensor::zeros((2, 2, 259), DType::F32, &device).unwrap();
    let non_patch_latents = standalone.encode(&non_patch).unwrap();
    assert_eq!(non_patch_latents.dims3().unwrap(), (2, 256, 2));
    let non_patch_decoded = standalone.decode(&non_patch_latents).unwrap();
    assert_eq!(non_patch_decoded.dims3().unwrap(), (2, 2, 8192));
    assert_eq!(
        SameAutoencoder::crop_valid_prefix(&non_patch_decoded, 259)
            .unwrap()
            .dims3()
            .unwrap(),
        (2, 2, 259)
    );

    let half_alignment = Tensor::zeros((1, 2, 12_288), DType::F32, &device).unwrap();
    let half_latents = standalone.encode(&half_alignment).unwrap();
    assert_eq!(half_latents.dims3().unwrap(), (1, 256, 4));
    assert_eq!(
        standalone.decode(&half_latents).unwrap().dim(2).unwrap(),
        16_384,
        "12,288 is divisible by 4,096 but not by the effective 8,192 alignment"
    );
    let odd_latents = Tensor::zeros((1, 256, 3), DType::F32, &device).unwrap();
    assert_eq!(
        standalone.decode(&odd_latents).unwrap().dim(2).unwrap(),
        16_384,
        "odd latent length must pad to the decoder's two-latent chunk boundary"
    );

    let stride_eight_latents = standalone.encode_with_strides(&audio, Some(&[8])).unwrap();
    assert_eq!(stride_eight_latents.dim(2).unwrap(), 8);
    assert_eq!(
        standalone
            .decode_with_noise(&stride_eight_latents, Some(&[8]), None, None)
            .unwrap()
            .dim(2)
            .unwrap(),
        16_384
    );
    assert!(standalone.encode_with_strides(&audio, Some(&[0])).is_err());
    assert!(standalone.encode_with_strides(&audio, Some(&[7])).is_err());
    assert!(standalone
        .decode_with_noise(&stride_eight_latents, Some(&[7]), None, None)
        .is_err());
    assert!(standalone
        .encode_with_strides(&audio, Some(&[8, 16]))
        .is_err());
}

#[test]
#[ignore = "requires explicitly provisioned standalone and embedded SAME-S snapshots"]
fn runtime_consumes_all_and_only_244_same_s_tensors_in_both_namespaces() {
    let device = test_device();
    for env in ["SA3_SAME_S_SNAPSHOT", "SA3_SMALL_MUSIC_SNAPSHOT"] {
        let layout = SnapshotLayout::from_dir(&snapshot(env)).unwrap();
        assert!(matches!(
            (env, layout.kind),
            ("SA3_SAME_S_SNAPSHOT", SnapshotKind::StandaloneAutoencoder)
                | ("SA3_SMALL_MUSIC_SNAPSHOT", SnapshotKind::Full)
        ));
        let expected = expected_autoencoder_weights(&layout);
        assert_eq!(expected.len(), 244, "{env} autoencoder inventory");
        let (_model, consumed) = load_with_consumption_audit(&layout, &device);
        assert_eq!(consumed.len(), 244, "{env} consumed inventory");
        assert_eq!(consumed, expected, "{env} exact tensor consumption");
    }
}

#[test]
fn frozen_upstream_two_stage_model_locks_override_list_execution_order() {
    let device = Device::Cpu;
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../docs/migration/sa3-same-s-reference");
    let config = StableAudioConfig::from_path(&root.join("two-stage-config.json")).unwrap();
    let oracle = mmap(&root.join("two-stage.safetensors"), &device);
    let model = SameAutoencoder::load(
        config.autoencoder(),
        StableAudioVarBuilders {
            encoder: oracle.pp("weights.encoder"),
            decoder: oracle.pp("weights.decoder"),
            bottleneck: oracle.pp("weights.bottleneck"),
            dit: None,
            conditioner: None,
            text_encoder: None,
        },
    )
    .unwrap();
    let audio = oracle.get((1, 2, 32), "audio").unwrap();
    for (label, strides, latent_shape) in [
        ("order24", [2usize, 4usize], (1, 4, 2)),
        ("order42", [4usize, 2usize], (1, 4, 4)),
    ] {
        let (latents, encoder) = model.encode_with_trace(&audio, Some(&strides)).unwrap();
        metric(
            &format!("{label} latents"),
            &latents,
            &oracle
                .get(latent_shape, &format!("{label}.latents"))
                .unwrap(),
            2e-5,
        );
        assert_eq!(encoder.stages.len(), 2);
        for (stage, trace) in encoder.stages.iter().enumerate() {
            for (name, actual) in [
                ("mapped_sequence", &trace.mapped_sequence),
                ("folded_input", &trace.folded_input),
                ("expanded_tokens", &trace.expanded_tokens),
                ("selected_segments", &trace.selected_segments),
                ("output", &trace.output),
            ] {
                metric(
                    &format!("{label} encoder{stage} {name}"),
                    actual,
                    &oracle
                        .get_unchecked(&format!("{label}.encoder{stage}.{name}"))
                        .unwrap(),
                    2e-5,
                );
            }
            metric(
                &format!("{label} encoder{stage} block0"),
                &trace.block_outputs[0],
                &oracle
                    .get_unchecked(&format!("{label}.encoder{stage}.block_0"))
                    .unwrap(),
                2e-5,
            );
        }
        let (decoded, decoder) = model
            .decode_with_trace(&latents, Some(&strides), None, None)
            .unwrap();
        metric(
            &format!("{label} decoded"),
            &decoded,
            &oracle.get((1, 2, 64), &format!("{label}.decoded")).unwrap(),
            2e-5,
        );
        assert_eq!(decoder.stages.len(), 2);
        for (stage, trace) in decoder.stages.iter().enumerate() {
            for (name, actual) in [
                ("mapped_sequence", &trace.mapped_sequence),
                ("folded_input", &trace.folded_input),
                ("expanded_tokens", &trace.expanded_tokens),
                ("selected_segments", &trace.selected_segments),
                ("output", &trace.output),
            ] {
                metric(
                    &format!("{label} decoder{stage} {name}"),
                    actual,
                    &oracle
                        .get_unchecked(&format!("{label}.decoder{stage}.{name}"))
                        .unwrap(),
                    2e-5,
                );
            }
            metric(
                &format!("{label} decoder{stage} block0"),
                &trace.block_outputs[0],
                &oracle
                    .get_unchecked(&format!("{label}.decoder{stage}.block_0"))
                    .unwrap(),
                2e-5,
            );
        }
    }
}

fn roundtrip_metrics(reference: &[Vec<Vec<f32>>], decoded: &[Vec<Vec<f32>>]) -> (f64, f64) {
    let mut signal = 0f64;
    let mut error = 0f64;
    for (reference_batch, decoded_batch) in reference.iter().zip(decoded) {
        for (reference_channel, decoded_channel) in reference_batch.iter().zip(decoded_batch) {
            for (&a, &b) in reference_channel.iter().zip(decoded_channel) {
                signal += (a as f64).powi(2);
                error += ((a - b) as f64).powi(2);
            }
        }
    }
    let snr = 10.0 * (signal / error).log10();
    let mut resolution_terms = Vec::new();
    for (window, hop) in [(512, 128), (1024, 256), (2048, 512)] {
        let hann = hann_window(window);
        let mut numerator = 0f64;
        let mut denominator = 0f64;
        let mut log_sum = 0f64;
        let mut count = 0usize;
        for (reference_batch, decoded_batch) in reference.iter().zip(decoded) {
            for (reference_channel, decoded_channel) in reference_batch.iter().zip(decoded_batch) {
                let reference_mag = stft(reference_channel, window, hop, &hann)
                    .unwrap()
                    .magnitude();
                let decoded_mag = stft(decoded_channel, window, hop, &hann)
                    .unwrap()
                    .magnitude();
                for (&a, &b) in reference_mag.iter().zip(&decoded_mag) {
                    numerator += ((b - a) as f64).powi(2);
                    denominator += (a as f64).powi(2);
                    log_sum += ((b as f64 + 1e-7).ln() - (a as f64 + 1e-7).ln()).abs();
                    count += 1;
                }
            }
        }
        resolution_terms.push((numerator / denominator).sqrt() + log_sum / count as f64);
    }
    (snr, resolution_terms.iter().sum::<f64>() / 3.0)
}

#[test]
#[ignore = "requires the explicitly provisioned pinned SAME-S snapshot"]
fn ten_second_music_roundtrip_matches_torch_and_measured_quality_bounds() {
    let device = test_device();
    let layout = SnapshotLayout::from_dir(&snapshot("SA3_SAME_S_SNAPSHOT")).unwrap();
    let model = SameAutoencoder::load(
        layout.config.autoencoder(),
        layout.mmap_builders(DType::F32, &device).unwrap(),
    )
    .unwrap();
    let fixture = same_fixture("music-roundtrip.safetensors", &device);
    let audio = fixture.get((1, 2, 441_000), "audio").unwrap();
    let latents = model.encode(&audio).unwrap();
    metric(
        "music latents",
        &latents,
        &fixture.get((1, 256, 108), "latents").unwrap(),
        3e-4,
    );
    let regularization = fixture.get((1, 256, 108), "regularization_noise").unwrap();
    let mask = fixture.get((1, 1_728, 768), "decoder_mask_noise").unwrap();
    let decoded = model
        .decode_with_noise(
            &latents,
            None,
            Some(&regularization),
            Some(std::slice::from_ref(&mask)),
        )
        .unwrap();
    assert_eq!(decoded.dims3().unwrap(), (1, 2, 442_368));
    metric(
        "music decoded padded",
        &decoded,
        &fixture.get((1, 2, 442_368), "decoded_padded").unwrap(),
        7e-4,
    );
    let cropped = SameAutoencoder::crop_valid_prefix(&decoded, 441_000).unwrap();
    let (snr, mrstft) = roundtrip_metrics(
        &audio.to_vec3::<f32>().unwrap(),
        &cropped.to_vec3::<f32>().unwrap(),
    );
    eprintln!("music roundtrip: snr_db={snr:.6}, mrstft={mrstft:.6}");
    assert!(snr >= 7.6, "SNR {snr} dB fell below measured bound");
    assert!(mrstft <= 3.055, "MR-STFT {mrstft} exceeded measured bound");
}
