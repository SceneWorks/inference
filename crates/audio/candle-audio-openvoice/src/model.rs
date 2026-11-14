//! `OpenVoiceTransform` — the [`gen_core::AudioTransform`] implementation for **OpenVoice V2 tone-color
//! voice conversion** on the candle audio lane (sc-13223), plus its [`descriptor`]/[`load`] entry
//! points and the explicit [`REGISTRATION`] wired into `candle-audio-catalog` under the id
//! `"openvoice_v2"` — the first real [`gen_core::AudioTransform`] (the sc-12839 release gate).
//!
//! ## Snapshot layout
//!
//! [`load`] expects a `myshell-ai/OpenVoiceV2` **converter** snapshot directory:
//!
//! ```text
//!   config.json       → converter/config.json (VITS hyperparameters; validated at load)
//!   checkpoint.pth    → converter/checkpoint.pth (the SynthesizerTrn state dict; see crate::weights)
//! ```
//!
//! A `LoadSpec` points at exactly that `converter/` layout: it is staged locally and passed in,
//! never self-fetched (epic 13657). The `HUB_REPO`@`HUB_REVISION` pin records its provenance.
//!
//! ## Request mapping
//!
//! [`apply`](gen_core::AudioTransform::apply) consumes the source clip
//! ([`AudioTransformRequest::audio`]) as the content and the **target tone-color reference**
//! ([`AudioTransformRequest::target_reference`], the additive sc-13223 field) as the voice to
//! transfer. Both are resampled to the model's native 22.05 kHz. `strength` overrides the
//! posterior-sampling temperature `τ` (default [`config::DEFAULT_TAU`]); `seed` fixes the
//! SplitMix64/Box–Muller Gaussian draw. Bitwise output replay has not been attested across
//! checkpoint bytes, devices, or precision modes. Output is exactly **one**
//! [`AudioTrack`] at 22.05 kHz whose duration/content tracks the source and whose timbre is shifted
//! toward the target. Cancellation is checked before any heavy work and between the flow/decoder
//! stages, returning the typed [`gen_core::Error::Canceled`].

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_audio::gen_core::generator::{
    ExecutionDraw, ExecutionFact, ExecutionIdentity, ExecutionRandomness, ExecutionReplaySupport,
    ResolvedExecutionIdentity, EXECUTION_IDENTITY_VERSION,
};
use candle_audio::gen_core::{
    self, AudioTarget, AudioTrack, AudioTransform, AudioTransformCapabilities,
    AudioTransformDescriptor, AudioTransformKind, AudioTransformRequest, CompletedAudioTransform,
    LoadSpec, Progress, WeightsSource,
};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::config;
use crate::pipeline::{OpenVoicePipeline, CHECKPOINT_FILE, CONFIG_FILE};
use candle_audio::dsp::to_mono;
use candle_audio::lock_recover;

/// Registry id (the SceneWorks worker routes an audio-transform request to this exact id).
pub const MODEL_ID: &str = "openvoice_v2";

/// Provider family for the OpenVoice converter.
pub const FAMILY: &str = "openvoice";

/// Hub pin: `myshell-ai/OpenVoiceV2` at an immutable commit SHA (F-029; MIT weights — commercial OK).
pub const HUB_REPO: &str = "myshell-ai/OpenVoiceV2";
pub const HUB_REVISION: &str = "f36e7edfe1684461a8343844af60babc2efbb727";

/// Stable component key for the pinned OpenVoice V2 checkpoint — what `PROVIDER_COMPONENTS`
/// resolves through, and the licence manifest's unique row key.
pub const COMPONENT_KEY: &str = "openvoice_v2";

/// The schema-3 licence row for the pinned OpenVoice V2 checkpoint (sc-16663).
///
/// **Disclosure only.** The row records what the upstream declares so a consumer can show it to a
/// user; nothing here decides whether any use is permitted. `declared` and `gated` were read from
/// the `myshell-ai/OpenVoiceV2` model card on `retrieved`, and `family` normalizes that declaration onto
/// [`candle_audio::gen_core::license::families::MIT`].
pub const COMPONENT_LICENSE: candle_audio::gen_core::ComponentLicense =
    candle_audio::gen_core::ComponentLicense {
        component: COMPONENT_KEY,
        source_url: "https://huggingface.co/myshell-ai/OpenVoiceV2",
        gated: false,
        declared: "mit",
        family: "mit",
        attribution: Some("OpenVoice V2 © MyShell.ai — licensed under MIT"),
        retrieved: "2026-08-02",
    };

/// The converter files inside the pinned repo (both live under `converter/`).
pub const CONVERTER_CONFIG: &str = "converter/config.json";
pub const CONVERTER_CHECKPOINT: &str = "converter/checkpoint.pth";

/// The output sample rate — OpenVoice V2's native converter rate (the decoder always emits here,
/// regardless of the source clip's rate; the source is resampled in).
pub const OUTPUT_SAMPLE_RATE: u32 = config::SAMPLE_RATE;

/// OpenVoice V2's identity + advertised capabilities — constructible without weights.
pub fn descriptor() -> AudioTransformDescriptor {
    AudioTransformDescriptor {
        id: MODEL_ID,
        family: FAMILY,
        backend: "candle",
        capabilities: AudioTransformCapabilities {
            kind: AudioTransformKind::VoiceConversion,
            stem_count: 0,
            is_diffusion: false,
            // `strength` overrides the posterior-sampling temperature τ.
            supports_strength: true,
            // Not a resampler — the rate is the model's native 22.05 kHz (AudioTarget::Preserve).
            supports_resample: false,
            mac_only: false,
        },
    }
}

/// A loaded (lazy) OpenVoice V2 converter. The pipeline (reference encoder + converter weights) is
/// built on first `apply`, not at [`load`] — the sibling providers' lazy-load discipline.
pub struct OpenVoiceTransform {
    descriptor: AudioTransformDescriptor,
    root: PathBuf,
    pipeline: Mutex<Option<Arc<OpenVoicePipeline>>>,
}

/// Values captured by the successful conversion call, before any report is constructed.
struct OpenVoiceObserved {
    source_mono_sha256: String,
    reference_mono_sha256: String,
    source_rate: u32,
    reference_rate: u32,
    source_channels: u16,
    reference_channels: u16,
    source_mono_samples: usize,
    reference_mono_samples: usize,
    tau_bits: u32,
    seed: u64,
    noise_frames: usize,
    device: String,
    dtype: String,
}

fn pcm_sha256(samples: &[f32]) -> String {
    let mut digest = Sha256::new();
    for sample in samples {
        digest.update(sample.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn completed_identity(facts: OpenVoiceObserved, tracks: &[AudioTrack]) -> ExecutionIdentity {
    let mut provider_choices = BTreeMap::new();
    provider_choices.insert(
        "source".into(),
        ExecutionFact::resolved(json!({
            "mono_pcm_f32_sha256":facts.source_mono_sha256,
            "sample_rate":facts.source_rate,
            "channels":facts.source_channels,
            "mono_samples":facts.source_mono_samples,
        })),
    );
    provider_choices.insert(
        "target_reference".into(),
        ExecutionFact::resolved(json!({
            "mono_pcm_f32_sha256":facts.reference_mono_sha256,
            "sample_rate":facts.reference_rate,
            "channels":facts.reference_channels,
            "mono_samples":facts.reference_mono_samples,
        })),
    );
    provider_choices.insert(
        "converter".into(),
        ExecutionFact::resolved(json!({
            "path":"source_and_reference_tone_color_to_vits_flow",
            "native_sample_rate":OUTPUT_SAMPLE_RATE,
            "posterior_tau_bits":format!("f32:{:08x}",facts.tau_bits),
            "noise_scheme":"splitmix64_box_muller",
            "noise_shape":[1,config::INTER_CHANNELS,facts.noise_frames],
        })),
    );
    let outputs = tracks
        .iter()
        .enumerate()
        .map(|(index, track)| {
            json!({
                "index":index,
                "pcm_f32_sha256":pcm_sha256(&track.samples),
                "sample_rate":track.sample_rate,
                "channels":track.channels,
                "samples":track.samples.len(),
            })
        })
        .collect::<Vec<_>>();
    ExecutionIdentity::Resolved {
        identity: Box::new(ResolvedExecutionIdentity {
            version: EXECUTION_IDENTITY_VERSION,
            provider_id: MODEL_ID.into(),
            backend: "candle".into(),
            runtime_id: "openvoice-v2-candle-v1".into(),
            sampling: ExecutionFact::resolved(json!({
                "posterior_tau_bits":format!("f32:{:08x}",facts.tau_bits),
                "noise_scheme":"splitmix64_box_muller",
            })),
            schedule: ExecutionFact::not_applicable("voice conversion has no diffusion schedule"),
            estimator: ExecutionFact::not_applicable("voice conversion has no diffusion estimator"),
            randomness: ExecutionRandomness::Draws {
                scheme: "splitmix64_box_muller".into(),
                draws: vec![ExecutionDraw {
                    output_index: 0,
                    purpose: "posterior_gaussian_noise".into(),
                    seed: Some(facts.seed),
                    noise_sha256: None,
                }],
            },
            deterministic_replay: ExecutionReplaySupport::NonDeterministic {
                reason: "OpenVoice V2 checkpoint bytes and bitwise replay on this device and precision have not been attested".into(),
            },
            geometry: ExecutionFact::not_applicable("audio has no image geometry"),
            quality_policy: ExecutionFact::resolved(json!({
                "tone_color_device":facts.device,
                "tone_color_dtype":facts.dtype,
                "mono_downmix":"channel_mean",
                "resample_to_hz":OUTPUT_SAMPLE_RATE,
            })),
            output_policy: ExecutionFact::resolved(json!({"tracks":outputs})),
            provider_choices,
            output_units: tracks.len(),
        }),
    }
}

impl OpenVoiceTransform {
    fn pipeline(&self) -> gen_core::Result<Arc<OpenVoicePipeline>> {
        let mut guard = lock_recover(&self.pipeline);
        if let Some(p) = guard.as_ref() {
            return Ok(p.clone());
        }
        let device = candle_audio::default_device()?;
        let built = Arc::new(OpenVoicePipeline::from_snapshot(&self.root, &device)?);
        *guard = Some(built.clone());
        Ok(built)
    }

    fn apply_inner(
        &self,
        req: &AudioTransformRequest,
        on_progress: &mut dyn FnMut(Progress),
        capture: bool,
    ) -> gen_core::Result<(Vec<AudioTrack>, Option<OpenVoiceObserved>)> {
        self.validate(req)?;
        // Preserve the legacy pre-load cancellation boundary.
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let reference = req
            .target_reference
            .as_ref()
            .expect("validate_request guarantees Some");

        let pipeline = self.pipeline()?;
        let total = 3u32;
        on_progress(Progress::Step { current: 1, total });

        let src_mono = to_mono(&req.audio.samples, req.audio.channels);
        let tgt_mono = to_mono(&reference.samples, reference.channels);
        let g_src = pipeline.extract_tone_color(&src_mono, req.audio.sample_rate)?;
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let g_tgt = pipeline.extract_tone_color(&tgt_mono, reference.sample_rate)?;
        on_progress(Progress::Step { current: 2, total });

        let tau = req.strength.unwrap_or(config::DEFAULT_TAU);
        if !(tau.is_finite() && tau >= 0.0) {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID}: strength (τ) must be finite and >= 0, got {tau}"
            )));
        }
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);

        on_progress(Progress::Decoding);
        let cancel = req.cancel.clone();
        let (samples, noise_frames) = pipeline
            .convert_observed(
                &src_mono,
                req.audio.sample_rate,
                &g_src,
                &g_tgt,
                tau,
                seed,
                &move || cancel.is_cancelled(),
            )
            .map_err(gen_core::Error::from)?;
        on_progress(Progress::Step { current: 3, total });

        if samples.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID}: conversion produced no samples"
            )));
        }
        if let AudioTarget::SampleRate(r) = req.target {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID}: voice conversion does not resample (requested {r} Hz); the output is \
                 the model's native {OUTPUT_SAMPLE_RATE} Hz — use AudioTarget::Preserve"
            )));
        }
        let observed = capture.then(|| OpenVoiceObserved {
            source_mono_sha256: pcm_sha256(&src_mono),
            reference_mono_sha256: pcm_sha256(&tgt_mono),
            source_rate: req.audio.sample_rate,
            reference_rate: reference.sample_rate,
            source_channels: req.audio.channels,
            reference_channels: reference.channels,
            source_mono_samples: src_mono.len(),
            reference_mono_samples: tgt_mono.len(),
            tau_bits: tau.to_bits(),
            seed,
            noise_frames,
            device: format!("{:?}", g_src.device().location()),
            dtype: format!("{:?}", g_src.dtype()),
        });
        Ok((
            vec![AudioTrack {
                samples,
                sample_rate: OUTPUT_SAMPLE_RATE,
                channels: 1,
                ..Default::default()
            }],
            observed,
        ))
    }
}

/// Shared request validation (weightless): a VoiceConversion converter needs a source clip long
/// enough to analyze AND a target tone-color reference (the additive sc-13223 field). A `None`
/// reference is a typed error — this is a reference-based converter, not a weight-baked one.
pub(crate) fn validate_request(req: &AudioTransformRequest) -> gen_core::Result<()> {
    if req.audio.samples.len() < config::MIN_SAMPLES {
        return Err(gen_core::Error::Msg(format!(
            "{MODEL_ID}: source clip has {} samples (< {}); too short to convert",
            req.audio.samples.len(),
            config::MIN_SAMPLES
        )));
    }
    match &req.target_reference {
        None => Err(gen_core::Error::Msg(format!(
            "{MODEL_ID}: voice conversion needs a target tone-color reference — set \
             AudioTransformRequest::target_reference to the clip whose voice to transfer"
        ))),
        Some(reference) if reference.samples.len() < config::MIN_SAMPLES => {
            Err(gen_core::Error::Msg(format!(
                "{MODEL_ID}: target reference clip has {} samples (< {}); too short to extract a \
                 tone color",
                reference.samples.len(),
                config::MIN_SAMPLES
            )))
        }
        Some(_) => Ok(()),
    }
}

impl AudioTransform for OpenVoiceTransform {
    fn descriptor(&self) -> &AudioTransformDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &AudioTransformRequest) -> gen_core::Result<()> {
        validate_request(req)
    }

    fn apply(
        &self,
        req: &AudioTransformRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<Vec<AudioTrack>> {
        self.apply_inner(req, on_progress, false)
            .map(|(tracks, _)| tracks)
    }

    fn apply_with_execution_report(
        &self,
        req: &AudioTransformRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<CompletedAudioTransform> {
        let (tracks, observed) = self.apply_inner(req, on_progress, true)?;
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let facts = observed.expect("report path captures completed OpenVoice choices");
        let execution = completed_identity(facts, &tracks);
        CompletedAudioTransform::new(tracks, execution, &self.descriptor)
    }
}

/// Construct the (lazy) OpenVoice transform from a [`LoadSpec`]. `spec.weights` must be the
/// converter snapshot directory (see module docs); adapters/quantization/control overlays are
/// rejected — refusing is more honest than silently dropping.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn AudioTransform>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID} expects a converter snapshot directory ({CONFIG_FILE} + \
                 {CHECKPOINT_FILE}), not a single file"
            )));
        }
    };
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID} does not support on-the-fly quantization"
        )));
    }
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID} does not support LoRA/LoKr adapters"
        )));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID} does not support control/IP-adapter overlays"
        )));
    }
    Ok(Box::new(OpenVoiceTransform {
        descriptor: descriptor(),
        root,
        pipeline: Mutex::new(None),
    }))
}

// Explicit catalog registration for `openvoice_v2` (composed by `candle-audio-catalog`).
candle_audio::gen_core::register_audio_transform! {
    pub const REGISTRATION = descriptor => load
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::gen_core::CancelFlag;

    fn track(n: usize, rate: u32) -> AudioTrack {
        AudioTrack {
            samples: vec![0.01; n],
            sample_rate: rate,
            channels: 1,
            ..Default::default()
        }
    }

    #[test]
    fn completed_transform_identity_binds_observed_inputs_seed_runtime_and_output() {
        let source = [0.0f32, -0.0, 1.0];
        let reference = [1.0f32, 0.5];
        let output = track(3, OUTPUT_SAMPLE_RATE);
        let identity = completed_identity(
            OpenVoiceObserved {
                source_mono_sha256: pcm_sha256(&source),
                reference_mono_sha256: pcm_sha256(&reference),
                source_rate: 24_000,
                reference_rate: 44_100,
                source_channels: 2,
                reference_channels: 1,
                source_mono_samples: source.len(),
                reference_mono_samples: reference.len(),
                tau_bits: 0.3f32.to_bits(),
                seed: 42,
                noise_frames: 7,
                device: "Cpu".into(),
                dtype: "F32".into(),
            },
            &[output],
        );
        identity.validate(MODEL_ID, "candle", 1).unwrap();
        let resolved = identity.resolved().unwrap();
        assert_eq!(
            resolved.provider_choices["source"],
            ExecutionFact::resolved(json!({
                "mono_pcm_f32_sha256":pcm_sha256(&source),
                "sample_rate":24_000,"channels":2,"mono_samples":3
            }))
        );
        assert_eq!(
            resolved.provider_choices["target_reference"],
            ExecutionFact::resolved(json!({
                "mono_pcm_f32_sha256":pcm_sha256(&reference),
                "sample_rate":44_100,"channels":1,"mono_samples":2
            }))
        );
        assert_eq!(
            resolved.provider_choices["converter"].clone(),
            ExecutionFact::resolved(json!({
                "path":"source_and_reference_tone_color_to_vits_flow",
                "native_sample_rate":OUTPUT_SAMPLE_RATE,
                "posterior_tau_bits":format!("f32:{:08x}",0.3f32.to_bits()),
                "noise_scheme":"splitmix64_box_muller",
                "noise_shape":[1,config::INTER_CHANNELS,7]
            }))
        );
        let ExecutionRandomness::Draws { draws, .. } = &resolved.randomness else {
            panic!("seeded Gaussian draw");
        };
        assert_eq!(draws[0].seed, Some(42));
        assert_eq!(
            resolved.output_policy,
            ExecutionFact::resolved(json!({"tracks":[{
                "index":0,"pcm_f32_sha256":pcm_sha256(&track(3,OUTPUT_SAMPLE_RATE).samples),
                "sample_rate":OUTPUT_SAMPLE_RATE,"channels":1,"samples":3
            }]}))
        );
        assert!(matches!(
            resolved.deterministic_replay,
            ExecutionReplaySupport::NonDeterministic { .. }
        ));
        assert_ne!(pcm_sha256(&[0.0]), pcm_sha256(&[-0.0]));
    }

    #[test]
    fn descriptor_advertises_the_voice_conversion_surface() {
        let d = descriptor();
        assert_eq!(d.id, "openvoice_v2");
        assert_eq!(d.family, "openvoice");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.capabilities.kind, AudioTransformKind::VoiceConversion);
        assert_eq!(d.capabilities.stem_count, 0);
        assert!(d.capabilities.supports_strength);
        assert!(!d.capabilities.supports_resample);
        assert!(!d.capabilities.mac_only);
    }

    #[test]
    fn validate_requires_a_target_reference_and_long_enough_clips() {
        // Missing target reference → error.
        let req = AudioTransformRequest {
            audio: track(config::MIN_SAMPLES, 24_000),
            target_reference: None,
            ..Default::default()
        };
        assert!(validate_request(&req).is_err());
        // Too-short source → error even with a reference.
        let req = AudioTransformRequest {
            audio: track(8, 24_000),
            target_reference: Some(track(config::MIN_SAMPLES, 24_000)),
            ..Default::default()
        };
        assert!(validate_request(&req).is_err());
        // Too-short reference → error.
        let req = AudioTransformRequest {
            audio: track(config::MIN_SAMPLES, 24_000),
            target_reference: Some(track(8, 24_000)),
            ..Default::default()
        };
        assert!(validate_request(&req).is_err());
        // Both present and long enough → ok.
        let req = AudioTransformRequest {
            audio: track(config::MIN_SAMPLES, 24_000),
            target_reference: Some(track(config::MIN_SAMPLES, 24_000)),
            ..Default::default()
        };
        assert!(validate_request(&req).is_ok());
    }

    #[test]
    fn to_mono_averages_channels() {
        assert_eq!(to_mono(&[1.0, 3.0, 2.0, 4.0], 2), vec![2.0, 3.0]);
        assert_eq!(to_mono(&[1.0, 2.0], 1), vec![1.0, 2.0]);
    }

    #[test]
    fn load_rejects_unsupported_spec_shapes() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        assert!(load(&LoadSpec::new(WeightsSource::File(
            dir.join("checkpoint.pth")
        )))
        .is_err());
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
        spec.quantize = Some(gen_core::Quant::Q4);
        assert!(matches!(load(&spec), Err(gen_core::Error::Unsupported(_))));
    }

    #[test]
    fn pre_tripped_cancel_returns_typed_canceled_before_any_heavy_work() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        let t = load(&LoadSpec::new(WeightsSource::Dir(dir))).unwrap();
        let flag = CancelFlag::new();
        flag.cancel();
        let req = AudioTransformRequest {
            audio: track(config::MIN_SAMPLES, 24_000),
            target_reference: Some(track(config::MIN_SAMPLES, 24_000)),
            cancel: flag,
            ..Default::default()
        };
        let err = t.apply(&req, &mut |_| {}).unwrap_err();
        assert!(matches!(err, gen_core::Error::Canceled));
        assert!(matches!(
            t.apply_with_execution_report(&req, &mut |_| {}),
            Err(gen_core::Error::Canceled)
        ));
    }

    #[test]
    fn apply_on_a_missing_snapshot_fails_cleanly() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        let t = load(&LoadSpec::new(WeightsSource::Dir(dir))).unwrap();
        let req = AudioTransformRequest {
            audio: track(config::MIN_SAMPLES, 24_000),
            target_reference: Some(track(config::MIN_SAMPLES, 24_000)),
            ..Default::default()
        };
        let err = t.apply(&req, &mut |_| {}).unwrap_err();
        assert!(!matches!(err, gen_core::Error::Canceled));
        assert!(t.apply_with_execution_report(&req, &mut |_| {}).is_err());
        assert!(t.apply_with_execution_report(&req, &mut |_| {}).is_err());
    }
}
