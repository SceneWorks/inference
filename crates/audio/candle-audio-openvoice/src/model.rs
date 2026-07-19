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
//! [`resolve_pinned_snapshot`] materializes exactly that layout through the audio lane's pinned-SHA
//! hub path (`candle_audio::hub`, F-029 — never the mutable `main` revision), landing the `converter/`
//! directory in the HF cache and returning it as the snapshot root.
//!
//! ## Request mapping
//!
//! [`apply`](gen_core::AudioTransform::apply) consumes the source clip
//! ([`AudioTransformRequest::audio`]) as the content and the **target tone-color reference**
//! ([`AudioTransformRequest::target_reference`], the additive sc-13223 field) as the voice to
//! transfer. Both are resampled to the model's native 22.05 kHz. `strength` overrides the
//! posterior-sampling temperature `τ` (default [`config::DEFAULT_TAU`]); `seed` makes the Gaussian
//! draw deterministic (same request + seed ⇒ byte-identical samples). Output is exactly **one**
//! [`AudioTrack`] at 22.05 kHz whose duration/content tracks the source and whose timbre is shifted
//! toward the target. Cancellation is checked before any heavy work and between the flow/decoder
//! stages, returning the typed [`gen_core::Error::Canceled`].

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_audio::gen_core::{
    self, AudioTarget, AudioTrack, AudioTransform, AudioTransformCapabilities,
    AudioTransformDescriptor, AudioTransformKind, AudioTransformRequest, LoadSpec, Progress,
    WeightsSource,
};
use candle_audio::hub::hf_get_pinned;
use candle_audio::{AudioError, Result as AudioResult};

use crate::config;
use crate::pipeline::{OpenVoicePipeline, CHECKPOINT_FILE, CONFIG_FILE};

/// Registry id (the SceneWorks worker routes an audio-transform request to this exact id).
pub const MODEL_ID: &str = "openvoice_v2";

/// Provider family for the OpenVoice converter.
pub const FAMILY: &str = "openvoice";

/// Hub pin: `myshell-ai/OpenVoiceV2` at an immutable commit SHA (F-029; MIT weights — commercial OK).
pub const HUB_REPO: &str = "myshell-ai/OpenVoiceV2";
pub const HUB_REVISION: &str = "f36e7edfe1684461a8343844af60babc2efbb727";

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

/// Recover a poisoned mutex — the audio twin of `candle_gen::lock_recover`.
fn lock_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// A loaded (lazy) OpenVoice V2 converter. The pipeline (reference encoder + converter weights) is
/// built on first `apply`, not at [`load`] — the sibling providers' lazy-load discipline.
pub struct OpenVoiceTransform {
    descriptor: AudioTransformDescriptor,
    root: PathBuf,
    pipeline: Mutex<Option<Arc<OpenVoicePipeline>>>,
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
}

/// Down-mix interleaved `channels`-channel PCM to mono by averaging (a no-op for mono).
fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    samples
        .chunks(ch)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect()
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
        self.validate(req)?;
        // Pre-apply cancellation seam: consult the flag before ANY heavy work (weights load +
        // synthesis all come after this).
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

        // Tone colors: the source's own (forward flow) and the target's (reverse flow).
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
        let samples = pipeline
            .convert(
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
        // AudioTarget::Preserve is the only supported target (this is not a resampler); a
        // SampleRate target is refused rather than silently ignored.
        if let AudioTarget::SampleRate(r) = req.target {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID}: voice conversion does not resample (requested {r} Hz); the output is \
                 the model's native {OUTPUT_SAMPLE_RATE} Hz — use AudioTarget::Preserve"
            )));
        }
        Ok(vec![AudioTrack {
            samples,
            sample_rate: OUTPUT_SAMPLE_RATE,
            channels: 1,
            ..Default::default()
        }])
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

/// Materialize the pinned OpenVoice V2 converter snapshot through the audio lane's F-029 hub path:
/// `converter/config.json` + `converter/checkpoint.pth` at [`HUB_REVISION`], landing in the HF
/// cache. Returns the `converter/` directory as a [`WeightsSource::Dir`] ready for a [`LoadSpec`].
pub fn resolve_pinned_snapshot() -> AudioResult<WeightsSource> {
    let cfg = hf_get_pinned(HUB_REPO, HUB_REVISION, CONVERTER_CONFIG)?;
    hf_get_pinned(HUB_REPO, HUB_REVISION, CONVERTER_CHECKPOINT)?;
    let dir = cfg.parent().ok_or_else(|| {
        AudioError::Msg(format!(
            "openvoice_v2: resolved {CONVERTER_CONFIG} path {} has no parent directory",
            cfg.display()
        ))
    })?;
    Ok(WeightsSource::Dir(dir.to_path_buf()))
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
        let dir = std::env::temp_dir();
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
        let dir = std::env::temp_dir().join("openvoice-missing-snapshot");
        std::fs::create_dir_all(&dir).unwrap();
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
    }

    #[test]
    fn apply_on_a_missing_snapshot_fails_cleanly() {
        let dir = std::env::temp_dir().join("openvoice-missing-snapshot");
        std::fs::create_dir_all(&dir).unwrap();
        let t = load(&LoadSpec::new(WeightsSource::Dir(dir))).unwrap();
        let req = AudioTransformRequest {
            audio: track(config::MIN_SAMPLES, 24_000),
            target_reference: Some(track(config::MIN_SAMPLES, 24_000)),
            ..Default::default()
        };
        let err = t.apply(&req, &mut |_| {}).unwrap_err();
        assert!(!matches!(err, gen_core::Error::Canceled));
    }
}
