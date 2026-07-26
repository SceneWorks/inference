//! `stable_audio_3_small_music` generator contract and lazy provider load.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_audio::gen_core::{
    self, AudioTrack, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    Modality, ModelDescriptor, OffloadPolicy, Precision, Progress, WeightsSource,
};
use sha2::{Digest, Sha256};

use crate::dit::Guidance;
use crate::pipeline::{
    StableAudio3SmallMusicPipeline, SynthesisParameters, CHANNELS, DEFAULT_DURATION_SECS,
    DEFAULT_GUIDANCE, DEFAULT_STEPS, SAMPLE_RATE,
};
use crate::sampler::SamplerKind;
use crate::weights::SnapshotLayout;
use crate::{resolve_device, DevicePolicy};

pub const MODEL_ID: &str = "stable_audio_3_small_music";
pub const HUB_REPO: &str = "stabilityai/stable-audio-3-small-music";
pub const HUB_REVISION: &str = "0fef1392cd842149a2b6d445e181c97608faac06";
pub const MAX_DURATION_SECS: f32 = 120.0;
pub const MAX_STEPS: u32 = 500;
pub const GUIDANCE_RANGE: (f32, f32) = (0.0, 25.0);

struct SnapshotFilePin {
    relative: &'static str,
    bytes: u64,
    sha256: &'static str,
}

const SNAPSHOT_FILE_PINS: &[SnapshotFilePin] = &[
    SnapshotFilePin {
        relative: "model_config.json",
        bytes: 10_341,
        sha256: "100776f25af5aa83f70e0c6b384de6690cb4e5ad01c24f7cfbb6524d18765f06",
    },
    SnapshotFilePin {
        relative: "model.safetensors",
        bytes: 2_270_384_940,
        sha256: "da85866b11b01d0694d990785f6abbd79c8064df1b0e6f8aea52935e0ef84b64",
    },
    SnapshotFilePin {
        relative: "t5gemma-b-b-ul2/config.json",
        bytes: 2_540,
        sha256: "575334409716886ac2952f5a275ed92868deef8a0ea560258d9970a431c6fb3a",
    },
    SnapshotFilePin {
        relative: "t5gemma-b-b-ul2/model.safetensors",
        bytes: 1_183_022_944,
        sha256: "9b05ea5a4f211d023832f706fb2c0e83e4fc721b6da35ab69ceb0b55eb7800d3",
    },
    SnapshotFilePin {
        relative: "t5gemma-b-b-ul2/tokenizer.json",
        bytes: 34_362_429,
        sha256: "7794135caa3ea73918949c902a781cc61dab674a4b59c17d85931c77c1114cbd",
    },
    SnapshotFilePin {
        relative: "t5gemma-b-b-ul2/tokenizer.model",
        bytes: 4_241_003,
        sha256: "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
    },
];

pub const ROOT_WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "LicenseRef-Stability-AI-Community",
    name: "Stability AI Community License",
    source_url: "https://huggingface.co/stabilityai/stable-audio-3-small-music/blob/0fef1392cd842149a2b6d445e181c97608faac06/LICENSE.md",
    attribution: Some("Stable Audio 3 Small Music © Stability AI"),
    commercial_use: false,
    restriction: Some(
        "Use is governed by the Stability AI Community License, including its revenue threshold and prohibited-use terms.",
    ),
};

pub const GEMMA_WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "LicenseRef-Gemma-Terms",
    name: "Gemma Terms of Use",
    source_url: "https://huggingface.co/stabilityai/stable-audio-3-small-music/blob/0fef1392cd842149a2b6d445e181c97608faac06/LICENSE_GEMMA.md",
    attribution: Some("T5Gemma model weights © Google"),
    commercial_use: true,
    restriction: Some("Use is governed by the Gemma Terms of Use and Prohibited Use Policy."),
};

pub const WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] = &[
    gen_core::WeightLicenseEntry {
        provider_id: MODEL_ID,
        component: None,
        license: ROOT_WEIGHT_LICENSE,
    },
    gen_core::WeightLicenseEntry {
        provider_id: MODEL_ID,
        component: Some("root"),
        license: ROOT_WEIGHT_LICENSE,
    },
    gen_core::WeightLicenseEntry {
        provider_id: MODEL_ID,
        component: Some("t5gemma"),
        license: GEMMA_WEIGHT_LICENSE,
    },
];

pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        required_components: &[],
        id: MODEL_ID,
        family: "stable_audio_3",
        backend: "candle",
        modality: Modality::Audio,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: Vec::new(),
            supports_lora: false,
            supports_lokr: false,
            samplers: vec!["pingpong", "euler", "rk4", "dpmpp"],
            schedulers: vec![],
            supported_guidance_methods: vec!["cfg", "apg", "cfg_rescale"],
            min_size: 0,
            max_size: 0,
            max_count: 1,
            mac_only: false,
            audio_sample_rates: vec![SAMPLE_RATE],
            max_audio_duration_secs: Some(MAX_DURATION_SECS),
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
            supported_quants: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            max_speakers: None,
        },
    }
}

fn verify_file_pin(path: &std::path::Path, pin: &SnapshotFilePin) -> gen_core::Result<()> {
    let metadata = std::fs::metadata(path).map_err(|error| {
        gen_core::Error::Msg(format!(
            "{MODEL_ID}: read pinned file {}: {error}",
            path.display()
        ))
    })?;
    if metadata.len() != pin.bytes {
        return Err(gen_core::Error::Msg(format!(
            "{MODEL_ID}: {} byte length {} does not match {HUB_REPO}@{HUB_REVISION}",
            pin.relative,
            metadata.len()
        )));
    }
    let file = std::fs::File::open(path).map_err(|error| {
        gen_core::Error::Msg(format!(
            "{MODEL_ID}: open pinned file {}: {error}",
            path.display()
        ))
    })?;
    let mut reader = std::io::BufReader::with_capacity(1024 * 1024, file);
    let mut digest = Sha256::new();
    std::io::copy(&mut reader, &mut digest).map_err(|error| {
        gen_core::Error::Msg(format!(
            "{MODEL_ID}: hash pinned file {}: {error}",
            path.display()
        ))
    })?;
    let actual = format!("{:x}", digest.finalize());
    if actual != pin.sha256 {
        return Err(gen_core::Error::Msg(format!(
            "{MODEL_ID}: {} SHA-256 does not match {HUB_REPO}@{HUB_REVISION}",
            pin.relative
        )));
    }
    Ok(())
}

fn verify_snapshot_identity(layout: &SnapshotLayout) -> gen_core::Result<()> {
    for pin in SNAPSHOT_FILE_PINS {
        verify_file_pin(&layout.root.join(pin.relative), pin)?;
    }
    Ok(())
}

pub(crate) fn validate_request(
    descriptor: &ModelDescriptor,
    request: &GenerationRequest,
) -> gen_core::Result<()> {
    descriptor
        .capabilities
        .validate_request_audio(descriptor.id, request)?;
    if request.prompt.trim().is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{MODEL_ID}: prompt must not be empty"
        )));
    }
    let audio = request.audio.clone().unwrap_or_default();
    let duration = audio.target_duration.unwrap_or(DEFAULT_DURATION_SECS);
    if duration < 1.0 / SAMPLE_RATE as f32 {
        return Err(gen_core::Error::Msg(format!(
            "{MODEL_ID}: audio.target_duration must contain at least one 44.1 kHz frame"
        )));
    }
    if audio.bpm.is_some() || audio.musical_key.is_some() || audio.lyrics.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID} exposes text-prompt music generation only; BPM, key, and lyrics conditioning are unsupported"
        )));
    }
    if let Some(steps) = request.steps {
        if steps > MAX_STEPS {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID}: steps {steps} exceeds the {MAX_STEPS}-step model limit"
            )));
        }
    }
    if let Some(guidance) = request.guidance {
        if !(GUIDANCE_RANGE.0..=GUIDANCE_RANGE.1).contains(&guidance) {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID}: guidance {guidance} outside {}..={}",
                GUIDANCE_RANGE.0, GUIDANCE_RANGE.1
            )));
        }
    }
    if request.guidance_momentum.unwrap_or(0.0) != 0.0 {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID} does not support APG momentum"
        )));
    }
    let method = request.guidance_method.as_deref();
    if request.guidance_eta.is_some() && method != Some("apg") {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: guidance_eta is only supported with guidance_method=apg"
        )));
    }
    if request.guidance_norm_threshold.is_some() && method != Some("apg") {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: guidance_norm_threshold is only supported with guidance_method=apg"
        )));
    }
    Ok(())
}

fn synthesis_parameters(request: &GenerationRequest) -> SynthesisParameters {
    let audio = request.audio.clone().unwrap_or_default();
    let method = request.guidance_method.as_deref();
    let guidance = Guidance {
        cfg_scale: request.guidance.unwrap_or(DEFAULT_GUIDANCE as f32) as f64,
        apg_scale: match method {
            Some("cfg") | Some("cfg_rescale") => 0.0,
            Some("apg") => 1.0 - request.guidance_eta.unwrap_or(0.0) as f64,
            None => 1.0,
            Some(_) => unreachable!("capability validation rejects unknown methods"),
        },
        cfg_norm_threshold: request.guidance_norm_threshold.unwrap_or(0.0) as f64,
        scale_phi: if method == Some("cfg_rescale") {
            1.0
        } else {
            0.0
        },
    };
    SynthesisParameters {
        duration_secs: audio.target_duration.unwrap_or(DEFAULT_DURATION_SECS),
        steps: request.steps.unwrap_or(DEFAULT_STEPS as u32) as usize,
        sampler: match request.sampler.as_deref() {
            None | Some("pingpong") => SamplerKind::Pingpong,
            Some("euler") => SamplerKind::Euler,
            Some("rk4") => SamplerKind::Rk4,
            Some("dpmpp") => SamplerKind::Dpmpp,
            Some(_) => unreachable!("capability validation rejects unknown samplers"),
        },
        guidance,
        seed: request.seed.unwrap_or_else(gen_core::default_seed),
    }
}

pub struct StableAudio3SmallMusicGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    pipeline: Mutex<Option<Arc<StableAudio3SmallMusicPipeline>>>,
    generation: Mutex<()>,
}

impl StableAudio3SmallMusicGenerator {
    fn pipeline(&self) -> gen_core::Result<Arc<StableAudio3SmallMusicPipeline>> {
        let mut guard = match self.pipeline.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(pipeline) = guard.as_ref() {
            return Ok(pipeline.clone());
        }
        let layout = SnapshotLayout::from_dir(&self.root)?;
        let device = resolve_device(DevicePolicy::Default)?;
        let pipeline = Arc::new(StableAudio3SmallMusicPipeline::from_layout(
            &layout, &device,
        )?);
        *guard = Some(pipeline.clone());
        Ok(pipeline)
    }
}

pub fn load_generator(spec: &LoadSpec) -> gen_core::Result<StableAudio3SmallMusicGenerator> {
    let root = match &spec.weights {
        WeightsSource::Dir(path) => path.clone(),
        WeightsSource::File(path) => {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID} requires a snapshot directory, got {}",
                path.display()
            )));
        }
    };
    if spec.quantize.is_some()
        || spec.precision != Precision::Bf16
        || !spec.adapters.is_empty()
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
        || !spec.components.is_empty()
        || spec.offload_policy != OffloadPolicy::Resident
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID} accepts only its native dense self-contained snapshot"
        )));
    }
    let layout = SnapshotLayout::from_dir(&root)?;
    crate::pipeline::validate_small_music_layout(&layout)?;
    verify_snapshot_identity(&layout)?;
    Ok(StableAudio3SmallMusicGenerator {
        descriptor: descriptor(),
        root,
        pipeline: Mutex::new(None),
        generation: Mutex::new(()),
    })
}

pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_generator(spec)?))
}

impl Generator for StableAudio3SmallMusicGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor, request)
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(request)?;
        if request.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        // Candle's shared Metal graph is not safe to execute concurrently: command-buffer
        // interleaving can change the resulting PCM even when every request owns its RNG stream.
        // Serialize graph execution while keeping all stochastic state request-local.
        let _generation = match self.generation.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let parameters = synthesis_parameters(request);
        let pipeline = self.pipeline()?;
        let cancel = request.cancel.clone();
        let progress = std::cell::RefCell::new(on_progress);
        let mut step_progress = |current: usize, total: usize| {
            (progress.borrow_mut())(Progress::Step {
                current: current as u32,
                total: total as u32,
            });
        };
        let mut decoding = || (progress.borrow_mut())(Progress::Decoding);
        let samples = pipeline.synthesize(
            &request.prompt,
            request.negative_prompt.as_deref(),
            parameters,
            &mut step_progress,
            &mut decoding,
            &|| cancel.is_cancelled(),
        )?;
        Ok(GenerationOutput::Audio(AudioTrack {
            samples,
            sample_rate: SAMPLE_RATE,
            channels: CHANNELS as u16,
            stems: Vec::new(),
        }))
    }
}

candle_audio::register_generators! {
    pub const REGISTRATION = descriptor => load
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::gen_core::{
        AdapterKind, AdapterSpec, AudioParams, Conditioning, Image, Quant,
    };

    fn request() -> GenerationRequest {
        GenerationRequest {
            prompt: "orchestral post-rock with bowed strings".into(),
            audio: Some(AudioParams {
                target_duration: Some(30.0),
                sample_rate: Some(SAMPLE_RATE),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_is_honest_and_conformant() {
        let descriptor = descriptor();
        assert_eq!(descriptor.id, MODEL_ID);
        assert_eq!(descriptor.family, "stable_audio_3");
        assert_eq!(descriptor.backend, "candle");
        assert!(matches!(descriptor.modality, Modality::Audio));
        assert_eq!(descriptor.capabilities.audio_sample_rates, [SAMPLE_RATE]);
        assert_eq!(
            descriptor.capabilities.max_audio_duration_secs,
            Some(MAX_DURATION_SECS)
        );
        assert!(descriptor.capabilities.supports_negative_prompt);
        assert!(descriptor.capabilities.supports_guidance);
        assert_eq!(
            descriptor.capabilities.samplers,
            ["pingpong", "euler", "rk4", "dpmpp"]
        );
    }

    #[test]
    fn request_validation_maps_the_complete_public_surface() {
        let descriptor = descriptor();
        assert!(validate_request(&descriptor, &request()).is_ok());
        let mut valid = request();
        valid.negative_prompt = Some("harsh clipping and speech".into());
        valid.guidance = Some(7.5);
        valid.sampler = Some("dpmpp".into());
        valid.guidance_method = Some("apg".into());
        valid.guidance_eta = Some(0.25);
        valid.guidance_norm_threshold = Some(2.0);
        assert!(validate_request(&descriptor, &valid).is_ok());

        let mut invalid = request();
        invalid.audio.as_mut().unwrap().bpm = Some(120.0);
        assert!(matches!(
            validate_request(&descriptor, &invalid),
            Err(gen_core::Error::Unsupported(_))
        ));

        for field in ["musical_key", "lyrics"] {
            let mut invalid = request();
            match field {
                "musical_key" => {
                    invalid.audio.as_mut().unwrap().musical_key = Some("D minor".into())
                }
                "lyrics" => invalid.audio.as_mut().unwrap().lyrics = Some("sing this".into()),
                _ => unreachable!(),
            }
            assert!(matches!(
                validate_request(&descriptor, &invalid),
                Err(gen_core::Error::Unsupported(_))
            ));
        }

        let mut invalid = request();
        invalid.conditioning.push(Conditioning::Reference {
            image: Image {
                width: 1,
                height: 1,
                pixels: vec![0, 0, 0],
            },
            strength: None,
        });
        assert!(matches!(
            validate_request(&descriptor, &invalid),
            Err(gen_core::Error::Unsupported(_))
        ));

        let mut invalid = request();
        invalid.guidance_method = Some("apg".into());
        invalid.guidance_momentum = Some(0.1);
        assert!(matches!(
            validate_request(&descriptor, &invalid),
            Err(gen_core::Error::Unsupported(_))
        ));

        let mut invalid = request();
        invalid.audio.as_mut().unwrap().target_duration = Some(0.0);
        assert!(validate_request(&descriptor, &invalid).is_err());

        let mut invalid = request();
        invalid.prompt = "  ".into();
        assert!(validate_request(&descriptor, &invalid).is_err());

        let mut invalid = request();
        invalid.steps = Some(MAX_STEPS + 1);
        assert!(validate_request(&descriptor, &invalid).is_err());

        let mut invalid = request();
        invalid.guidance = Some(GUIDANCE_RANGE.1 + 0.1);
        assert!(validate_request(&descriptor, &invalid).is_err());
    }

    #[test]
    fn guidance_methods_map_to_frozen_cfg_apg_endpoints() {
        let mut cfg = request();
        cfg.guidance = Some(4.0);
        cfg.guidance_method = Some("cfg".into());
        assert_eq!(
            synthesis_parameters(&cfg).guidance,
            Guidance {
                cfg_scale: 4.0,
                apg_scale: 0.0,
                cfg_norm_threshold: 0.0,
                scale_phi: 0.0,
            }
        );

        let mut apg = cfg.clone();
        apg.guidance_method = Some("apg".into());
        apg.guidance_eta = Some(0.25);
        apg.guidance_norm_threshold = Some(3.0);
        assert_eq!(synthesis_parameters(&apg).guidance.apg_scale, 0.75);
        assert_eq!(synthesis_parameters(&apg).guidance.cfg_norm_threshold, 3.0);

        let mut rescale = cfg;
        rescale.guidance_method = Some("cfg_rescale".into());
        assert_eq!(synthesis_parameters(&rescale).guidance.apg_scale, 0.0);
        assert_eq!(synthesis_parameters(&rescale).guidance.scale_phi, 1.0);
    }

    #[test]
    fn load_rejects_every_non_native_shape_before_touching_the_snapshot() {
        let missing = PathBuf::from("does-not-exist");
        assert!(load(&LoadSpec::new(WeightsSource::File(missing.clone()))).is_err());

        let dense = || LoadSpec::new(WeightsSource::Dir(missing.clone()));
        let mut specs = Vec::new();
        specs.push(dense().with_quant(Quant::Q4));

        let mut precision = dense();
        precision.precision = Precision::Fp32;
        specs.push(precision);

        specs.push(dense().with_adapters(vec![AdapterSpec::new(
            missing.clone(),
            1.0,
            AdapterKind::Lora,
        )]));
        specs.push(dense().with_control(WeightsSource::File(missing.clone())));
        specs.push(dense().with_extra_control(WeightsSource::File(missing.clone())));
        specs.push(dense().with_ip_adapter(WeightsSource::Dir(missing.clone())));
        specs.push(dense().with_pid(
            WeightsSource::File(missing.clone()),
            WeightsSource::Dir(missing.clone()),
        ));
        specs.push(dense().with_component("unsupported", WeightsSource::Dir(missing.clone())));
        specs.push(dense().with_offload_policy(OffloadPolicy::Sequential));

        let mut text = dense();
        text.text_encoder = Some(WeightsSource::Dir(missing.clone()));
        specs.push(text);

        let mut identity = dense();
        identity.identity = Some(Default::default());
        specs.push(identity);

        for spec in specs {
            assert!(matches!(load(&spec), Err(gen_core::Error::Unsupported(_))));
        }
    }

    #[test]
    fn pinned_file_authentication_rejects_size_and_payload_drift() {
        let root = std::env::temp_dir().join(format!(
            "sa3-provider-pin-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("fixture");
        std::fs::write(&path, b"exact").unwrap();
        let exact = SnapshotFilePin {
            relative: "fixture",
            bytes: 5,
            sha256: "fa79d4746c21cd960a17b92db8976ddef95a7e20b590721f8e0fa7847a05e486",
        };
        verify_file_pin(&path, &exact).unwrap();

        std::fs::write(&path, b"wrong").unwrap();
        assert!(verify_file_pin(&path, &exact).is_err());
        std::fs::write(&path, b"short").unwrap();
        let wrong_size = SnapshotFilePin { bytes: 4, ..exact };
        assert!(verify_file_pin(&path, &wrong_size).is_err());
        let _ = std::fs::remove_dir_all(root);
    }
}
