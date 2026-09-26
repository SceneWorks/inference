//! The reusable transcription provider: load the verified cover closure offline, transcribe a
//! source recording into a [`Transcription`], and unload observably.
//!
//! * **Offline, pinned.** [`Transcriber::load`] takes a [`VerifiedClosure`] — only obtainable from
//!   `candle_audio_yue2::snapshot::resolve_closure(Closure::Cover, …)`, which hashes every pinned
//!   file of both snapshots — and reads only the paths it names. Nothing is downloaded. The
//!   SheetSage2 config's pinned parent (`base_model_revision`, `base_model_sha256`) must equal the
//!   inventory's MERT pin, and the licence gate must permit noncommercial experimentation.
//! * **Unload before generation.** [`Transcriber::unload`] consumes the transcriber and reports
//!   whether the model was actually released; [`live_models`] counts every SheetSage2 model alive in
//!   the process, which the cover path checks before a YuE2 engine is created.

use std::sync::{Arc, Weak};

use candle_audio::candle_core::Device;
use candle_audio_yue2::inventory::ComponentId;
use candle_audio_yue2::license::{authorize_closure, IntendedUse};
use candle_audio_yue2::{Closure, VerifiedClosure};
use sha2::{Digest, Sha256};

use crate::model::weights::Weights;
use crate::model::SheetSage2Model;
use crate::pipeline::Stitcher;
use crate::review::{ClosureIdentity, SourceIdentity, Transcription, TranscriptionSettings};
use crate::Error;

/// The model's sample rate.
pub const SAMPLE_RATE: u32 = 24_000;

/// The transcriber's provider id (the key of its component mapping).
pub const TRANSCRIBER_ID: &str = "sheetsage2";

/// The licence rows of the components this transcriber loads — the cover closure, CC BY-NC 4.0
/// weights with their attributions (recorded by sc-22989 in `candle_audio_yue2::license`). A catalog
/// that composes this transcriber publishes these rows with it.
pub const COMPONENT_LICENSES: &[candle_audio::gen_core::ComponentLicense] = &[
    candle_audio_yue2::license::LICENSE_SHEETSAGE2,
    candle_audio_yue2::license::LICENSE_MERT_V2_FULLSONG,
];

/// Provider → component mapping: the transcriber loads exactly the cover closure.
pub const PROVIDER_COMPONENTS: &[candle_audio::gen_core::ProviderComponents] =
    &[candle_audio::gen_core::ProviderComponents {
        provider_id: TRANSCRIBER_ID,
        components: &["yue2_sheetsage2", "yue2_mert_v2_fullsong"],
    }];

pub use crate::model::live_models;

/// The mono 24 kHz input the model consumes, with how it was derived.
#[derive(Clone, Debug)]
pub struct SourceAudio {
    samples: Vec<f32>,
    name: Option<String>,
    original_sha256: Option<String>,
    conversion: String,
    /// The rate the samples were resampled to, when they were (`None`: supplied at the model rate).
    resampled_to: Option<u32>,
}

impl SourceAudio {
    /// Mono samples already at the model's rate (24 kHz for the pinned model) — upstream's array
    /// path with `sampling_rate=24000`: used unchanged.
    pub fn mono(samples: Vec<f32>) -> Self {
        Self {
            samples,
            name: None,
            original_sha256: None,
            conversion: "none (mono float32 supplied at the model rate)".into(),
            resampled_to: None,
        }
    }

    /// Interleaved samples at any rate: channel mean, then torchaudio's default sinc resampler to
    /// 24 kHz (upstream's array path, `load_audio` with `sampling_rate`).
    pub fn interleaved(samples: &[f32], rate: u32, channels: u16) -> Result<Self, Error> {
        if channels == 0 || rate == 0 {
            return Err(Error::Request(
                "rate and channel count must be positive".into(),
            ));
        }
        let c = usize::from(channels);
        if !samples.len().is_multiple_of(c) {
            return Err(Error::Request(
                "interleaved samples are not a whole number of frames".into(),
            ));
        }
        let mono: Vec<f32> = samples
            .chunks(c)
            .map(|frame| frame.iter().sum::<f32>() / c as f32)
            .collect();
        let resampled = candle_audio::dsp::resample_sinc_hann(&mono, rate, SAMPLE_RATE)
            .map_err(|e| Error::Request(format!("resample: {e}")))?;
        Ok(Self {
            samples: resampled,
            name: None,
            original_sha256: None,
            conversion: format!(
                "{channels}-channel mean, {rate} Hz → 24000 Hz (torchaudio sinc_interp_hann defaults)"
            ),
            resampled_to: Some(SAMPLE_RATE),
        })
    }

    /// Name the source (display only).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Record the SHA-256 of the original encoded file the host decoded.
    pub fn with_original_sha256(mut self, sha256: impl Into<String>) -> Self {
        self.original_sha256 = Some(sha256.into());
        self
    }

    /// The samples.
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Apply `max_seconds` and upstream's validity checks; returns the model input and its identity.
    fn prepare(
        &self,
        max_seconds: Option<f64>,
        rate: u32,
        minimum: usize,
    ) -> Result<(Vec<f32>, SourceIdentity), Error> {
        if self.resampled_to.is_some_and(|r| r != rate) {
            return Err(Error::Request(format!(
                "the source was resampled to {} Hz but the model runs at {rate} Hz",
                self.resampled_to.unwrap_or(0)
            )));
        }
        let mut samples = self.samples.clone();
        let mut conversion = self.conversion.clone();
        if let Some(max) = max_seconds {
            if !(max.is_finite() && max > 0.0) {
                return Err(Error::Request(
                    "max_seconds must be finite and positive".into(),
                ));
            }
            let keep = (max * f64::from(rate)).round_ties_even() as usize;
            if keep < samples.len() {
                samples.truncate(keep);
                conversion.push_str(&format!("; cropped to {max} s"));
            }
        }
        if samples.len() < minimum || samples.iter().any(|s| !s.is_finite()) {
            return Err(Error::Request(format!(
                "audio must contain at least {minimum} finite samples at {rate} Hz"
            )));
        }
        let mut hasher = Sha256::new();
        for s in &samples {
            hasher.update(s.to_le_bytes());
        }
        let identity = SourceIdentity {
            sha256: hasher
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
            samples: samples.len(),
            sample_rate: rate,
            name: self.name.clone(),
            original_sha256: self.original_sha256.clone(),
            conversion,
        };
        Ok((samples, identity))
    }
}

/// Transcription progress, for job reporting and cancellation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Progress {
    /// Encoding window `window` of `windows` (1-based).
    Encoding {
        /// Current window.
        window: usize,
        /// Total windows.
        windows: usize,
    },
    /// Decoding: `tokens` so far in window `window`.
    Decoding {
        /// Current window.
        window: usize,
        /// Total windows.
        windows: usize,
        /// Tokens generated so far in this window.
        tokens: usize,
    },
    /// Building the exports and the review.
    Notation,
}

/// What [`Transcriber::unload`] observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnloadReceipt {
    /// Whether the model was dropped (no reference survived the unload).
    pub released: bool,
    /// Parameter bytes the model held on its device.
    pub parameter_bytes: usize,
    /// SheetSage2 models still alive in the process afterwards.
    pub live_models_after: usize,
}

/// The loaded SheetSage2 + MERT-v2-FullSong transcriber.
pub struct Transcriber {
    loaded: Arc<SheetSage2Model>,
    closure: ClosureIdentity,
}

/// The files of a cover closure, for loading outside a [`VerifiedClosure`] (fixtures).
pub(crate) struct ClosureFiles {
    /// SheetSage2 `config.json`.
    pub sheetsage2_config: serde_json::Value,
    /// MERT `config.json`.
    pub mert_config: serde_json::Value,
    /// SheetSage2 head + adapter weights.
    pub head: Weights,
    /// MERT parent weights.
    pub parent: Weights,
}

fn device_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    }
}

fn read_json(path: &std::path::Path) -> Result<(serde_json::Value, String), Error> {
    let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
    let value = serde_json::from_slice(&bytes)
        .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
    let digest = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    Ok((value, digest))
}

/// Refuse a SheetSage2 config whose pinned parent is not the inventory's MERT pin.
pub fn check_parent_pin(sheetsage2_config: &serde_json::Value) -> Result<(), Error> {
    let mert = ComponentId::MertV2FullSong.component();
    let weights = mert.weights().expect("MERT has a weights file");
    let revision = sheetsage2_config["base_model_revision"].as_str();
    let sha = sheetsage2_config["base_model_sha256"].as_str();
    if revision != Some(mert.repo.revision) || sha != Some(weights.sha256) {
        return Err(Error::Closure(format!(
            "SheetSage2 pins MERT parent {revision:?} / {sha:?}, the inventory pins {} / {}",
            mert.repo.revision, weights.sha256
        )));
    }
    Ok(())
}

impl Transcriber {
    /// Load from the verified cover closure (offline; reads only the verified paths).
    pub fn load(closure: &VerifiedClosure, device: &Device) -> Result<Self, Error> {
        if closure.closure() != Closure::Cover {
            return Err(Error::Closure(
                "the transcriber loads the cover closure".into(),
            ));
        }
        authorize_closure(Closure::Cover, IntendedUse::NoncommercialExperimentation)
            .map_err(|e| Error::Closure(e.to_string()))?;
        let component = |id: ComponentId| {
            closure
                .get(id)
                .ok_or_else(|| Error::Closure(format!("{id:?} missing from the closure")))
        };
        let ss2 = component(ComponentId::SheetSage2)?;
        let mert = component(ComponentId::MertV2FullSong)?;
        let path = |c: &candle_audio_yue2::VerifiedComponent, file: &str| {
            c.path(file)
                .map(std::path::Path::to_path_buf)
                .ok_or_else(|| Error::Closure(format!("{file} was not verified")))
        };
        let (ss2_config, ss2_config_sha) = read_json(&path(ss2, "config.json")?)?;
        let (mert_config, mert_config_sha) = read_json(&path(mert, "config.json")?)?;
        check_parent_pin(&ss2_config)?;
        let head = Weights::load("SheetSage2", &path(ss2, "model.safetensors")?)?;
        let parent = Weights::load("MERT-v2-FullSong", &path(mert, "model.safetensors")?)?;
        let repo = |c: &candle_audio_yue2::VerifiedComponent, config_sha: String| {
            let component = c.component();
            [
                component.repo.id.to_string(),
                component.repo.revision.to_string(),
                component.weights().expect("weights").sha256.to_string(),
                config_sha,
            ]
        };
        let identity = ClosureIdentity {
            sheetsage2: repo(ss2, ss2_config_sha),
            mert: repo(mert, mert_config_sha),
            ported_code_revision: crate::PORTED_CODE_REVISION.into(),
            tokenizer_fingerprint: String::new(),
            device: device_name(device).into(),
        };
        Self::from_files(
            ClosureFiles {
                sheetsage2_config: ss2_config,
                mert_config,
                head,
                parent,
            },
            identity,
            device,
        )
    }

    /// Load from explicit closure files (tiny fixtures and tests). `identity` records what they
    /// are; its tokenizer fingerprint is filled from the model.
    pub(crate) fn from_files(
        files: ClosureFiles,
        mut identity: ClosureIdentity,
        device: &Device,
    ) -> Result<Self, Error> {
        let model = SheetSage2Model::load(
            &files.sheetsage2_config,
            &files.mert_config,
            files.head,
            files.parent,
            device,
        )?;
        identity.tokenizer_fingerprint = model.tokenizer().fingerprint().to_string();
        Ok(Self {
            loaded: Arc::new(model),
            closure: identity,
        })
    }

    /// The loaded model.
    pub fn model(&self) -> &SheetSage2Model {
        &self.loaded
    }

    /// The closure identity every transcription records.
    pub fn closure(&self) -> &ClosureIdentity {
        &self.closure
    }

    /// Transcribe `source` (upstream `transcribe`, default preset, float32): window plan, encode,
    /// grammar-constrained greedy decode, stitching, exports, both scores, octave evidence and the
    /// review. `progress` may cancel by returning an error.
    pub fn transcribe(
        &self,
        source: &SourceAudio,
        settings: &TranscriptionSettings,
        mut progress: impl FnMut(Progress) -> Result<(), Error>,
    ) -> Result<Transcription, Error> {
        let model = &self.loaded;
        let rate = model.config().sampling_rate as u32;
        let minimum = model.config().backbone.minimum_input_samples();
        let (audio, identity) = source.prepare(settings.max_seconds, rate, minimum)?;
        let duration = audio.len() as f64 / f64::from(rate);
        let tokenizer = model.tokenizer();
        let prompts: Vec<&str> = settings.prompts.iter().map(String::as_str).collect();
        let max_len = model.config().max_output_seq_len;
        let mut stitcher = Stitcher::new(
            tokenizer,
            &prompts,
            duration,
            settings.overlap_seconds,
            settings.lookahead_seconds,
            max_len,
        )?;
        let plan = stitcher.plan().to_vec();
        let window_samples = model.config().window_samples();
        for (index, window) in plan.iter().enumerate() {
            progress(Progress::Encoding {
                window: index + 1,
                windows: plan.len(),
            })?;
            // slice_audio: round(start * rate) offset, zero-padded to the window.
            let offset = (window.start * f64::from(rate)).round_ties_even() as usize;
            let mut segment: Vec<f32> = audio
                .get(offset..(offset + window_samples).min(audio.len()))
                .unwrap_or(&[])
                .to_vec();
            segment.resize(window_samples, 0.0);
            let (prefix, prefix_len) = stitcher.prefix(index)?;
            let memory = model.encode(&segment)?;
            let stop = stitcher.stop_time(index);
            let tokens = model.generate(&memory, &prefix, max_len, Some(stop), |step| {
                if (step.position + 1) % 64 == 0 {
                    progress(Progress::Decoding {
                        window: index + 1,
                        windows: plan.len(),
                        tokens: step.position + 1,
                    })?;
                }
                Ok(())
            })?;
            stitcher.accept(index, tokens, prefix_len)?;
        }
        progress(Progress::Notation)?;
        Transcription::build(
            identity,
            settings.clone(),
            self.closure.clone(),
            stitcher.finish(),
            audio,
        )
    }

    /// Drop the model and report whether it was released. Consumes the transcriber: nothing can
    /// transcribe with it afterwards.
    pub fn unload(self) -> UnloadReceipt {
        let parameter_bytes = self.loaded.parameter_bytes();
        let weak: Weak<SheetSage2Model> = Arc::downgrade(&self.loaded);
        drop(self);
        UnloadReceipt {
            released: weak.upgrade().is_none(),
            parameter_bytes,
            live_models_after: live_models(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
