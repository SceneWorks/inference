//! Real-weight Stable Audio 3 DiT parity and exact key-consumption gates.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_audio_stable_audio_3::candle_audio::candle_core::{
    DType, Device, Result as CandleResult, Shape, Tensor,
};
use candle_audio_stable_audio_3::config::ModelConfig;
use candle_audio_stable_audio_3::dit::{DitInputs, Guidance, StableAudio3Dit};
use candle_audio_stable_audio_3::weights::{
    map_weight_key, safetensors_keys, SnapshotLayout, StableAudioVarBuilders, WeightSection,
};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};

struct Case {
    env: &'static str,
    artifact: &'static str,
}

const CASES: &[Case] = &[
    Case {
        env: "SA3_SMALL_MUSIC_SNAPSHOT",
        artifact: "small-music-reference.safetensors",
    },
    Case {
        env: "SA3_SMALL_SFX_SNAPSHOT",
        artifact: "small-sfx-reference.safetensors",
    },
    Case {
        env: "SA3_SMALL_MUSIC_BASE_SNAPSHOT",
        artifact: "small-music-base-reference.safetensors",
    },
    Case {
        env: "SA3_SMALL_SFX_BASE_SNAPSHOT",
        artifact: "small-sfx-base-reference.safetensors",
    },
    Case {
        env: "SA3_MEDIUM_SNAPSHOT",
        artifact: "medium-reference.safetensors",
    },
    Case {
        env: "SA3_MEDIUM_BASE_SNAPSHOT",
        artifact: "medium-base-reference.safetensors",
    },
];

fn path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} must point to its pinned immutable snapshot"))
}

/// The crate's three-way real-weight device selector, identical to `same_oracle.rs` and
/// `chunked_oracle.rs`.
///
/// Branching on `SA3_TEST_METAL` alone would run every case that calls this on `Device::Cpu` inside
/// the CUDA lanes, which set `SA3_TEST_CUDA`. A requested backend that is unavailable is a hard
/// failure, never a fallback. `all_six_cpu_f32_predictions_match_p0` pins `Device::Cpu` by name and
/// deliberately does not use this.
fn device() -> Device {
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
    // Safety: test inputs are committed or pinned immutable artifacts.
    unsafe {
        VarBuilder::from_mmaped_safetensors(&[path.to_path_buf()], DType::F32, device).unwrap()
    }
}

fn oracle(case: &Case, device: &Device) -> VarBuilder<'static> {
    mmap(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../docs/migration/sa3-reference")
            .join(case.artifact),
        device,
    )
}

fn intermediate_oracle(device: &Device) -> VarBuilder<'static> {
    mmap(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../docs/migration/sa3-dit-reference/dit-intermediates.safetensors"),
        device,
    )
}

fn metric(name: &str, actual: &Tensor, expected: &Tensor) -> (f64, f32) {
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
    for (&left, &right) in actual.iter().zip(&expected) {
        dot += left as f64 * right as f64;
        aa += (left as f64).powi(2);
        bb += (right as f64).powi(2);
        max_abs = max_abs.max((left - right).abs());
    }
    let cosine = dot / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE);
    eprintln!("{name}: cosine={cosine:.9} max_abs={max_abs:.9}");
    assert!(cosine >= 0.999, "{name}: cosine {cosine}");
    // The bound is deliberately independent from cosine and tightened to the measured
    // cross-backend envelope after the real CPU/Metal runs.
    assert!(max_abs <= 0.02, "{name}: max_abs {max_abs}");
    (cosine, max_abs)
}

fn run_case(case: &Case, device: &Device) {
    let layout = SnapshotLayout::from_dir(&path(case.env)).unwrap();
    let model = StableAudio3Dit::from_layout(&layout, device).unwrap();
    let expected = oracle(case, device);
    let noise = expected.get((1, 256, 16), "dit_noise").unwrap();
    let timestep = expected.get(1, "dit_timestep").unwrap();
    let prompt = expected.get((1, 256, 768), "t5_projected_padded").unwrap();
    let seconds = Tensor::from_vec(vec![0.25f32], 1, device).unwrap();
    let local = Tensor::zeros((1, 257, 16), DType::F32, device).unwrap();
    let prediction = model
        .forward(DitInputs {
            latents: &noise,
            timestep: &timestep,
            prompt: &prompt,
            seconds_total: &seconds,
            local_conditioning: &local,
            padding_mask: None,
        })
        .unwrap();
    metric(
        case.env,
        &prediction,
        &expected.get((1, 256, 16), "dit_prediction").unwrap(),
    );
}

#[test]
#[ignore = "requires the pinned small-music snapshot"]
fn small_music_intermediates_and_frozen_v_zero_padding_match() {
    let device = device();
    let case = &CASES[0];
    let layout = SnapshotLayout::from_dir(&path(case.env)).unwrap();
    let model = StableAudio3Dit::from_layout(&layout, &device).unwrap();
    let p0 = oracle(case, &device);
    let expected = intermediate_oracle(&device);
    let noise = p0.get((1, 256, 16), "dit_noise").unwrap();
    let timestep = p0.get(1, "dit_timestep").unwrap();
    let prompt = p0.get((1, 256, 768), "t5_projected_padded").unwrap();
    let seconds = Tensor::from_vec(vec![0.25f32], 1, &device).unwrap();
    let local = Tensor::zeros((1, 257, 16), DType::F32, &device).unwrap();
    let (output, trace) = model
        .forward_with_trace(DitInputs {
            latents: &noise,
            timestep: &timestep,
            prompt: &prompt,
            seconds_total: &seconds,
            local_conditioning: &local,
            padding_mask: None,
        })
        .unwrap();
    let checks = [
        ("number_features", trace.number_features.unwrap()),
        ("number_embedding", trace.number_embedding.unwrap()),
        ("raw_context", trace.raw_context.unwrap()),
        ("projected_context", trace.projected_context.unwrap()),
        ("duration_global", trace.duration_global.unwrap()),
        ("timestep_features", trace.timestep_features.unwrap()),
        ("timestep_embedding", trace.timestep_embedding.unwrap()),
        ("combined_global", trace.combined_global.unwrap()),
        ("global_modulation", trace.global_modulation.unwrap()),
        ("preprocessed", trace.preprocessed.unwrap()),
        ("projected_input", trace.projected_input.unwrap()),
        ("with_memory", trace.with_memory.unwrap()),
        ("rotary_frequencies", trace.rotary_frequencies.unwrap()),
        ("layer0_local", trace.layer0_local.unwrap()),
        ("layer0_output", trace.layer0_output.unwrap()),
        ("trimmed", trace.trimmed.unwrap()),
        ("projected_output", trace.projected_output.unwrap()),
        ("output", output.clone()),
    ];
    for (name, actual) in checks {
        metric(
            name,
            &actual,
            &expected.get(actual.shape().clone(), name).unwrap(),
        );
    }

    let padding_mask = expected.get((1, 16), "padding_mask").unwrap();
    let partial = model
        .forward(DitInputs {
            latents: &noise,
            timestep: &timestep,
            prompt: &prompt,
            seconds_total: &seconds,
            local_conditioning: &local,
            padding_mask: Some(&padding_mask),
        })
        .unwrap();
    metric(
        "partial_padding_output",
        &partial,
        &expected
            .get((1, 256, 16), "partial_padding_output")
            .unwrap(),
    );
    let valid_prefix_delta = partial
        .narrow(2, 0, 8)
        .unwrap()
        .broadcast_sub(&output.narrow(2, 0, 8).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        valid_prefix_delta > 1e-4,
        "frozen CPU zero-V padding must change the valid-prefix softmax"
    );
}

/// Every assertion here is a self-comparison between two real-weight forwards — no frozen artifact
/// is read — so the case is device-portable, and sc-14546 runs it on the accelerator lanes rather
/// than pinning CPU: this is the one gate that separates "no negative conditioning" (the
/// `zero_cross_context_from_batch` path) from "an explicit all-invalid negative prompt", and it has
/// to execute on the backend whose CFG path it is certifying.
#[test]
#[ignore = "requires the pinned small-music snapshot"]
fn real_weights_detect_conditioning_mutations_and_exercise_cfg_apg() {
    let device = device();
    let case = &CASES[0];
    let layout = SnapshotLayout::from_dir(&path(case.env)).unwrap();
    let model = StableAudio3Dit::from_layout(&layout, &device).unwrap();
    let p0 = oracle(case, &device);
    let noise = p0.get((1, 256, 16), "dit_noise").unwrap();
    let timestep = p0.get(1, "dit_timestep").unwrap();
    let prompt = p0.get((1, 256, 768), "t5_projected_padded").unwrap();
    let seconds = Tensor::from_vec(vec![0.25f32], 1, &device).unwrap();
    let changed_seconds = Tensor::from_vec(vec![1.25f32], 1, &device).unwrap();
    let zero_local = Tensor::zeros((1, 257, 16), DType::F32, &device).unwrap();
    let baseline = model
        .forward(DitInputs {
            latents: &noise,
            timestep: &timestep,
            prompt: &prompt,
            seconds_total: &seconds,
            local_conditioning: &zero_local,
            padding_mask: None,
        })
        .unwrap();
    let changed_duration = model
        .forward(DitInputs {
            latents: &noise,
            timestep: &timestep,
            prompt: &prompt,
            seconds_total: &changed_seconds,
            local_conditioning: &zero_local,
            padding_mask: None,
        })
        .unwrap();
    assert!(
        max_abs_diff(&baseline, &changed_duration) > 1e-3,
        "duration must affect both locked conditioning routes"
    );

    let mut mask_first = vec![0f32; 257 * 16];
    let mut mask_last = vec![0f32; 257 * 16];
    for index in 0..16 {
        mask_first[index] = 1.0;
        mask_last[256 * 16 + index] = 1.0;
    }
    let mask_first = Tensor::from_vec(mask_first, (1, 257, 16), &device).unwrap();
    let mask_last = Tensor::from_vec(mask_last, (1, 257, 16), &device).unwrap();
    let first_output = model
        .forward(DitInputs {
            latents: &noise,
            timestep: &timestep,
            prompt: &prompt,
            seconds_total: &seconds,
            local_conditioning: &mask_first,
            padding_mask: None,
        })
        .unwrap();
    let reversed_output = model
        .forward(DitInputs {
            latents: &noise,
            timestep: &timestep,
            prompt: &prompt,
            seconds_total: &seconds,
            local_conditioning: &mask_last,
            padding_mask: None,
        })
        .unwrap();
    assert!(max_abs_diff(&baseline, &first_output) > 1e-3);
    assert!(
        max_abs_diff(&first_output, &reversed_output) > 1e-3,
        "reversing [mask,masked-input] channels must be detected"
    );

    let vanilla = model
        .forward_guided(
            &noise,
            &timestep,
            &prompt,
            None,
            None,
            &seconds,
            &zero_local,
            None,
            Guidance {
                cfg_scale: 2.0,
                apg_scale: 0.0,
                ..Guidance::default()
            },
        )
        .unwrap();
    let apg = model
        .forward_guided(
            &noise,
            &timestep,
            &prompt,
            None,
            None,
            &seconds,
            &zero_local,
            None,
            Guidance {
                cfg_scale: 2.0,
                apg_scale: 1.0,
                ..Guidance::default()
            },
        )
        .unwrap();
    assert!(
        max_abs_diff(&vanilla, &apg) > 1e-4,
        "nontrivial CFG must exercise the APG endpoint"
    );
    assert!(apg
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .is_finite());

    let all_invalid = Tensor::zeros((1, 256), DType::U8, &device).unwrap();
    let masked_negative = model
        .forward_guided(
            &noise,
            &timestep,
            &prompt,
            Some(&prompt),
            Some(&all_invalid),
            &seconds,
            &zero_local,
            None,
            Guidance {
                cfg_scale: 2.0,
                apg_scale: 0.0,
                ..Guidance::default()
            },
        )
        .unwrap();
    assert!(
        max_abs_diff(&vanilla, &masked_negative) > 1e-4,
        "absent negative conditioning must zero the entire cross context, while an explicit \
         all-invalid negative prompt retains its conditioned duration row"
    );
}

fn max_abs_diff(left: &Tensor, right: &Tensor) -> f32 {
    left.to_dtype(DType::F32)
        .unwrap()
        .broadcast_sub(&right.to_dtype(DType::F32).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

#[test]
#[ignore = "requires all six pinned full snapshots and multi-gigabyte CPU execution"]
fn all_six_cpu_f32_predictions_match_p0() {
    let device = Device::Cpu;
    for case in CASES {
        run_case(case, &device);
    }
}

#[test]
#[ignore = "requires one explicitly selected pinned snapshot; set SA3_DIT_CASE_ENV"]
fn selected_real_device_prediction_matches_p0() {
    let selected = std::env::var("SA3_DIT_CASE_ENV")
        .expect("SA3_DIT_CASE_ENV must name one of the six snapshot environment variables");
    let case = CASES.iter().find(|case| case.env == selected).unwrap();
    run_case(case, &device());
}

#[test]
#[ignore = "requires selected real weights; set SA3_DIT_CASE_ENV and SA3_DIT_RESOURCE_LENGTH"]
fn selected_real_device_resource_probe() {
    let selected = std::env::var("SA3_DIT_CASE_ENV")
        .expect("SA3_DIT_CASE_ENV must name one of the six snapshot environment variables");
    let case = CASES.iter().find(|case| case.env == selected).unwrap();
    let length: usize = std::env::var("SA3_DIT_RESOURCE_LENGTH")
        .expect("SA3_DIT_RESOURCE_LENGTH is required")
        .parse()
        .unwrap();
    let device = device();
    let layout = SnapshotLayout::from_dir(&path(case.env)).unwrap();
    let load_started = Instant::now();
    let model = StableAudio3Dit::from_layout(&layout, &device).unwrap();
    device.synchronize().unwrap();
    let load_seconds = load_started.elapsed().as_secs_f64();
    let latents = Tensor::zeros((1, 256, length), DType::F32, &device).unwrap();
    let timestep = Tensor::from_vec(vec![0.5f32], 1, &device).unwrap();
    let prompt = Tensor::zeros((1, 256, 768), DType::F32, &device).unwrap();
    let seconds = Tensor::from_vec(vec![0.25f32], 1, &device).unwrap();
    let local = Tensor::zeros((1, 257, length), DType::F32, &device).unwrap();
    let forward_started = Instant::now();
    let output = model
        .forward(DitInputs {
            latents: &latents,
            timestep: &timestep,
            prompt: &prompt,
            seconds_total: &seconds,
            local_conditioning: &local,
            padding_mask: None,
        })
        .unwrap();
    device.synchronize().unwrap();
    let forward_seconds = forward_started.elapsed().as_secs_f64();
    let checksum = output
        .to_dtype(DType::F32)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    eprintln!(
        "SA3_DIT_RESOURCE env={} device={:?} length={} load_seconds={:.6} \
         forward_seconds={:.6} peak_rss_bytes={:?} checksum={:.9}",
        case.env,
        device,
        length,
        load_seconds,
        forward_seconds,
        candle_audio_stable_audio_3::candle_audio::harness::peak_rss_bytes(),
        checksum,
    );
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
        self.inner.get_with_hints(shape, name, hints)
    }

    fn get_unchecked(&self, name: &str, _dtype: DType, _device: &Device) -> CandleResult<Tensor> {
        self.consumed
            .lock()
            .unwrap()
            .insert(format!("{}.{name}", self.component));
        self.inner.get_unchecked(name)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.contains_tensor(name)
    }
}

fn tracked(
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

#[test]
#[ignore = "requires all six pinned full snapshots"]
fn all_six_consume_every_dit_and_number_conditioner_tensor_exactly() {
    let device = Device::Cpu;
    for case in CASES {
        let layout = SnapshotLayout::from_dir(&path(case.env)).unwrap();
        let inner = layout.mmap_builders(DType::F32, &device).unwrap();
        let consumed = Arc::new(Mutex::new(BTreeSet::new()));
        let builders = StableAudioVarBuilders {
            encoder: inner.encoder,
            decoder: inner.decoder,
            bottleneck: inner.bottleneck,
            dit: Some(tracked("dit", inner.dit.unwrap(), &consumed, &device)),
            conditioner: Some(tracked(
                "conditioner",
                inner.conditioner.unwrap(),
                &consumed,
                &device,
            )),
            text_encoder: inner.text_encoder,
        };
        let config = match &layout.config.model {
            ModelConfig::Diffusion(model) => model,
            _ => unreachable!(),
        };
        let model = StableAudio3Dit::load(config, builders).unwrap();
        assert_eq!(
            model.depth(),
            if case.env.contains("MEDIUM") { 24 } else { 20 }
        );
        let actual = consumed.lock().unwrap().clone();
        let expected: BTreeSet<_> = safetensors_keys(&layout.weights_path)
            .unwrap()
            .into_iter()
            .filter_map(|key| {
                let mapped = map_weight_key(layout.kind, &key)?;
                match mapped.section {
                    WeightSection::Dit => Some(format!("dit.{}", mapped.local_key)),
                    WeightSection::Conditioner => {
                        // Prompt padding belongs sc-14537; this story consumes the two number
                        // tensors and deliberately no SVD/training data.
                        mapped
                            .local_key
                            .starts_with("conditioners.seconds_total.")
                            .then(|| format!("conditioner.{}", mapped.local_key))
                    }
                    _ => None,
                }
            })
            .collect();
        assert_eq!(actual, expected, "{}", case.env);
    }
}
