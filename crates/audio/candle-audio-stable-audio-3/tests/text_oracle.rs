//! Real-weight parity against the provenance-locked Transformers 5.8.0 T5Gemma oracle.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_audio_stable_audio_3::candle_audio;
use candle_audio_stable_audio_3::candle_audio::candle_core::{
    safetensors, DType, Device, Result as CandleResult, Shape, Tensor,
};
use candle_audio_stable_audio_3::config::{ConditionerConfig, ModelConfig};
use candle_audio_stable_audio_3::t5gemma::{
    encoder_weight_keys, T5GemmaConditioner, T5GemmaEncoderConfig,
};
use candle_audio_stable_audio_3::weights::{SnapshotLayout, TextWeightSummary};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};

const VALID_LENGTHS: &[usize] = &[3, 17, 256];
const PROMPT_KEYS: &[&str] = &["short", "medium", "truncated"];
const MINIMUM_TOKEN_COSINE: f64 = 0.999;
// Measured Metal maximum is 0.0625 after the canonical F32 compute/BF16 output boundary.
const METAL_MAXIMUM_ABSOLUTE_DELTA: f32 = 0.075;
// Measured CPU F32 maximum is 0.001459122 on the heterogeneous truncation probe.
const CPU_MAXIMUM_ABSOLUTE_DELTA: f32 = 0.002;
const LONG_PROMPT_HEAD_WORDS: &[&str] = &[
    "amber", "violin", "ocean", "thunder", "velvet", "rhythm", "copper", "forest", "lantern",
    "piano", "silver", "rain", "echo", "drum", "meadow", "sunrise", "crystal", "bass", "canyon",
    "pulse", "maple", "choir", "comet", "guitar", "harbor", "tempo", "winter", "flute", "aurora",
    "melody", "stone", "wave",
];
const LONG_PROMPT_TAIL_WORDS: &[&str] = &["xylophone", "kumquat", "zucchini", "zephyr", "quasar"];
const RETAINED_BOUNDARY_TOKEN_IDS: &[u32] = &[46107, 11030, 7830, 74458, 94510, 53188, 9566, 9591];
const DISCARDED_TAIL_TOKEN_IDS: &[u32] = &[
    54824, 545, 3953, 52524, 90083, 105165, 4949, 219990, 700, 62026,
];

fn prompts() -> Vec<String> {
    let head = LONG_PROMPT_HEAD_WORDS.repeat(12).join(" ");
    let long = format!("{head} {}", LONG_PROMPT_TAIL_WORDS.join(" "));
    vec![
        "A bell.".into(),
        "Warm analog synth pulses, crisp percussion, spacious stereo field, 112 BPM".into(),
        long,
    ]
}

fn assert_heterogeneous_right_truncation(ids: &[Vec<u32>]) {
    let truncated = &ids[2];
    assert_eq!(
        &truncated[256 - RETAINED_BOUNDARY_TOKEN_IDS.len()..],
        RETAINED_BOUNDARY_TOKEN_IDS,
        "the retained boundary must be the heterogeneous prompt head"
    );
    assert!(
        DISCARDED_TAIL_TOKEN_IDS
            .iter()
            .all(|tail_id| !truncated.contains(tail_id)),
        "tokens unique to the heterogeneous tail must be discarded"
    );
}

fn assert_tail_vs_head_mutation(conditioner: &T5GemmaConditioner) {
    let head = LONG_PROMPT_HEAD_WORDS.repeat(12).join(" ");
    let baseline = format!("{head} {}", LONG_PROMPT_TAIL_WORDS.join(" "));
    let tail_mutated = format!("{head} marimba paprika obelisk");
    let mut head_words = LONG_PROMPT_HEAD_WORDS.repeat(12);
    head_words[0] = "saffron";
    let head_mutated = format!(
        "{} {}",
        head_words.join(" "),
        LONG_PROMPT_TAIL_WORDS.join(" ")
    );
    let tokenized = conditioner
        .tokenize(&[baseline, tail_mutated, head_mutated])
        .unwrap();
    assert_eq!(
        tokenized.input_ids[0], tokenized.input_ids[1],
        "mutating only the discarded tail must preserve all retained IDs"
    );
    assert_ne!(
        tokenized.input_ids[0], tokenized.input_ids[2],
        "mutating the retained head must change the retained IDs"
    );
    assert!(tokenized
        .attention_mask
        .iter()
        .all(|mask| mask.iter().all(|&value| value == 1)));
}

fn cpu_fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../docs/migration/sa3-text-reference/text-cpu-f32.safetensors")
}

fn snapshot() -> PathBuf {
    std::env::var_os("SA3_SMALL_MUSIC_SNAPSHOT")
        .map(PathBuf::from)
        .expect("SA3_SMALL_MUSIC_SNAPSHOT must point to the pinned immutable snapshot")
}

fn cosine_and_max_abs(actual: &[f32], expected: &[f32]) -> (f64, f32) {
    assert_eq!(actual.len(), expected.len());
    let mut dot = 0f64;
    let mut actual_norm = 0f64;
    let mut expected_norm = 0f64;
    let mut max_abs = 0f32;
    for (&actual, &expected) in actual.iter().zip(expected) {
        dot += actual as f64 * expected as f64;
        actual_norm += (actual as f64).powi(2);
        expected_norm += (expected as f64).powi(2);
        max_abs = max_abs.max((actual - expected).abs());
    }
    (
        dot / (actual_norm.sqrt() * expected_norm.sqrt()).max(f64::MIN_POSITIVE),
        max_abs,
    )
}

fn host_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

struct AccessTrackingBackend {
    inner: candle_audio_stable_audio_3::candle_audio::candle_core::safetensors::MmapedSafetensors,
    accessed: Arc<Mutex<BTreeSet<String>>>,
    requested_dtypes: Arc<Mutex<Vec<DType>>>,
}

impl SimpleBackend for AccessTrackingBackend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        hints: Init,
        dtype: DType,
        device: &Device,
    ) -> CandleResult<Tensor> {
        self.accessed.lock().unwrap().insert(name.to_string());
        self.requested_dtypes.lock().unwrap().push(dtype);
        SimpleBackend::get(&self.inner, shape, name, hints, dtype, device)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> CandleResult<Tensor> {
        self.accessed.lock().unwrap().insert(name.to_string());
        self.requested_dtypes.lock().unwrap().push(dtype);
        SimpleBackend::get_unchecked(&self.inner, name, dtype, device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        SimpleBackend::contains_tensor(&self.inner, name)
    }
}

#[test]
#[ignore = "requires the pinned small-music snapshot and Metal-capable real-weight execution"]
fn actual_metal_policy_matches_canonical_cpu_f32_oracle() {
    let device = candle_audio::default_device().expect("Metal-capable Candle device");
    assert!(
        !matches!(device, Device::Cpu),
        "the release acceptance gate must exercise Metal"
    );
    let expected = safetensors::load(cpu_fixture_path(), &Device::Cpu).unwrap();
    let layout = SnapshotLayout::from_dir(&snapshot()).unwrap();
    assert_eq!(
        layout.text_keys,
        Some(TextWeightSummary {
            total: 340,
            encoder: 134,
            decoder: 206,
            encoder_params: 281_580_288,
            decoder_params: 309_910_272,
        })
    );
    let requested = encoder_weight_keys();
    assert_eq!(requested.len(), TextWeightSummary::ENCODER);
    assert!(requested
        .iter()
        .all(|key| key.starts_with("model.encoder.")));
    assert!(
        requested.iter().all(|key| !key.contains(".decoder.")),
        "the actual builder request set must be encoder-only"
    );

    let diffusion = match &layout.config.model {
        ModelConfig::Diffusion(config) => config,
        ModelConfig::Autoencoder(_) => unreachable!(),
    };
    let (conditioner_id, conditioner_config) = diffusion
        .conditioning
        .configs
        .iter()
        .find_map(|entry| match entry {
            ConditionerConfig::T5gemma { id, config } => Some((id, config)),
            ConditionerConfig::Number { .. } => None,
        })
        .unwrap();
    let builders = layout
        .mmap_builders_with_text_dtype(DType::F32, DType::F32, &device)
        .unwrap();
    let accessed = Arc::new(Mutex::new(BTreeSet::new()));
    let requested_dtypes = Arc::new(Mutex::new(Vec::new()));
    let tracking_backend = AccessTrackingBackend {
        // Safety: the explicitly pinned snapshot stays immutable for this process.
        inner: unsafe {
            candle_audio_stable_audio_3::candle_audio::candle_core::safetensors::MmapedSafetensors::new(
                layout.text_weights_path.as_ref().unwrap(),
            )
            .unwrap()
        },
        accessed: accessed.clone(),
        requested_dtypes: requested_dtypes.clone(),
    };
    let text_vb = VarBuilder::new_with_args(
        Box::new(tracking_backend) as Box<dyn SimpleBackend>,
        DType::F32,
        &device,
    );
    let text_config =
        T5GemmaEncoderConfig::from_path(layout.text_config_path.as_ref().unwrap()).unwrap();
    let audited_conditioner = T5GemmaConditioner::load(
        conditioner_config,
        diffusion.conditioning.cond_dim,
        DType::BF16,
        &text_config,
        layout.tokenizer_path.as_ref().unwrap(),
        text_vb,
        builders
            .conditioner
            .unwrap()
            .pp("conditioners")
            .pp(conditioner_id),
    )
    .unwrap();
    assert_eq!(
        *accessed.lock().unwrap(),
        requested.into_iter().collect(),
        "actual conditioner construction must request every encoder tensor and no decoder tensor"
    );
    assert!(
        requested_dtypes
            .lock()
            .unwrap()
            .iter()
            .all(|&dtype| dtype == DType::F32),
        "Metal policy must decode BF16-on-disk weights into F32 compute tensors"
    );
    drop(audited_conditioner);
    let conditioner = T5GemmaConditioner::from_layout(&layout, &device).unwrap();
    assert_tail_vs_head_mutation(&conditioner);
    let prompts = prompts();
    let actual = conditioner.encode(&prompts).unwrap();

    let expected_ids = expected["input_ids"]
        .to_dtype(DType::U32)
        .unwrap()
        .to_vec2::<u32>()
        .unwrap();
    let expected_mask = expected["attention_mask"].to_vec2::<u8>().unwrap();
    assert_eq!(actual.input_ids.to_vec2::<u32>().unwrap(), expected_ids);
    assert_heterogeneous_right_truncation(&expected_ids);
    assert_eq!(
        actual.attention_mask.to_vec2::<u8>().unwrap(),
        expected_mask
    );
    assert_eq!(
        expected_mask
            .iter()
            .map(|row| row.iter().map(|&value| value as usize).sum::<usize>())
            .collect::<Vec<_>>(),
        VALID_LENGTHS
    );

    assert_eq!(actual.embeddings.dtype(), DType::F32);
    let actual_values = host_f32(&actual.embeddings);
    // Canonical Torch CPU-F32 output is rounded once at the Metal BF16 raw-output boundary.
    let expected_raw = host_f32(&expected["raw_embeddings"].to_dtype(DType::BF16).unwrap());
    let padding = host_f32(&expected["padding_embedding"]);
    let row_width = 768;
    let prompt_width = 256 * row_width;
    for (prompt, &valid) in VALID_LENGTHS.iter().enumerate() {
        let start = prompt * prompt_width;
        let valid_end = start + valid * row_width;
        let mut minimum_token_cosine = f64::INFINITY;
        let mut maximum_token_abs = 0f32;
        for token in 0..valid {
            let offset = start + token * row_width;
            let (token_cosine, token_max_abs) = cosine_and_max_abs(
                &actual_values[offset..offset + row_width],
                &expected_raw[offset..offset + row_width],
            );
            minimum_token_cosine = minimum_token_cosine.min(token_cosine);
            maximum_token_abs = maximum_token_abs.max(token_max_abs);
        }
        let (cosine, max_abs) = cosine_and_max_abs(
            &actual_values[start..valid_end],
            &expected_raw[start..valid_end],
        );
        eprintln!(
            "prompt {} valid-token parity: aggregate_cosine={cosine:.9}, \
             minimum_token_cosine={minimum_token_cosine:.9}, max_abs={max_abs:.9}, \
             maximum_token_abs={maximum_token_abs:.9}",
            PROMPT_KEYS[prompt]
        );
        assert!(cosine >= 0.999, "prompt {prompt} cosine {cosine}");
        assert!(
            minimum_token_cosine >= MINIMUM_TOKEN_COSINE,
            "prompt {prompt} minimum token cosine {minimum_token_cosine}"
        );
        assert!(
            max_abs <= METAL_MAXIMUM_ABSOLUTE_DELTA,
            "prompt {prompt} max_abs {max_abs} exceeds locked BF16 tolerance"
        );
        for row in valid..256 {
            let offset = start + row * row_width;
            assert_eq!(
                &actual_values[offset..offset + row_width],
                padding.as_slice(),
                "prompt {prompt} learned padding row {row}"
            );
        }
    }
}

#[test]
#[ignore = "requires the pinned small-music snapshot and real-weight CPU execution"]
fn actual_cpu_f32_fallback_matches_transformers_oracle() {
    let device = Device::Cpu;
    let expected = safetensors::load(cpu_fixture_path(), &device).unwrap();
    let layout = SnapshotLayout::from_dir(&snapshot()).unwrap();
    let conditioner = T5GemmaConditioner::from_layout(&layout, &device).unwrap();
    assert_tail_vs_head_mutation(&conditioner);
    let empty = conditioner.encode(&[String::new()]).unwrap();
    assert_eq!(
        empty
            .attention_mask
            .to_vec2::<u8>()
            .unwrap()
            .into_iter()
            .flatten()
            .sum::<u8>(),
        0
    );
    let empty_values = host_f32(&empty.embeddings);
    assert!(empty_values.iter().all(|value| value.is_finite()));
    let expected_padding = host_f32(&expected["padding_embedding"]);
    assert!(empty_values
        .chunks_exact(768)
        .all(|row| row == expected_padding.as_slice()));
    let actual = conditioner.encode(&prompts()).unwrap();
    assert_eq!(actual.embeddings.dtype(), DType::F32);
    let expected_ids = expected["input_ids"]
        .to_dtype(DType::U32)
        .unwrap()
        .to_vec2::<u32>()
        .unwrap();
    assert_eq!(actual.input_ids.to_vec2::<u32>().unwrap(), expected_ids);
    assert_heterogeneous_right_truncation(&expected_ids);
    assert_eq!(
        actual.attention_mask.to_vec2::<u8>().unwrap(),
        expected["attention_mask"].to_vec2::<u8>().unwrap()
    );

    let actual_values = host_f32(&actual.embeddings);
    let expected_raw = host_f32(&expected["raw_embeddings"]);
    let padding = host_f32(&expected["padding_embedding"]);
    let row_width = 768;
    let prompt_width = 256 * row_width;
    for (prompt, &valid) in VALID_LENGTHS.iter().enumerate() {
        let start = prompt * prompt_width;
        let mut minimum_token_cosine = f64::INFINITY;
        let mut maximum_token_abs = 0f32;
        for token in 0..valid {
            let offset = start + token * row_width;
            let (cosine, max_abs) = cosine_and_max_abs(
                &actual_values[offset..offset + row_width],
                &expected_raw[offset..offset + row_width],
            );
            minimum_token_cosine = minimum_token_cosine.min(cosine);
            maximum_token_abs = maximum_token_abs.max(max_abs);
        }
        eprintln!(
            "CPU prompt {}: minimum_token_cosine={minimum_token_cosine:.9}, \
             maximum_token_abs={maximum_token_abs:.9}",
            PROMPT_KEYS[prompt]
        );
        assert!(minimum_token_cosine >= 0.999);
        assert!(maximum_token_abs <= CPU_MAXIMUM_ABSOLUTE_DELTA);
        for row in valid..256 {
            let offset = start + row * row_width;
            assert_eq!(
                &actual_values[offset..offset + row_width],
                padding.as_slice()
            );
        }
    }
}
