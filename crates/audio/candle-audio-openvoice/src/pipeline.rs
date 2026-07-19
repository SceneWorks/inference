//! The assembled OpenVoice V2 conversion pipeline (sc-13223): load the reference encoder + converter
//! from a `converter/` snapshot (`config.json` + `checkpoint.pth`), extract tone-color embeddings
//! from reference clips, and run the source→target voice conversion end to end.
//!
//! `from_snapshot` validates the shipped `config.json` against [`crate::config`] before building any
//! module — a checkpoint whose hyperparameters drifted from the pinned architecture is a typed
//! error, never a silent mis-load. `convert` reproduces `api.ToneColorConverter.convert`: resample
//! the source to [`config::SAMPLE_RATE`], take the `spectrogram_torch` linear spectrogram, and run
//! `voice_conversion(spec, g_src, g_tgt, tau)`; `extract_tone_color` reproduces `extract_se`
//! (spectrogram → `ref_enc`).

use std::path::Path;

use candle_audio::candle_core::{Device, Tensor};
use candle_audio::{AudioError, Result};

use crate::config;
use crate::converter::VoiceConverter;
use crate::reference_encoder::ReferenceEncoder;
use crate::spectrogram::{resample_to_native, spectrogram};
use crate::weights::state_var_builder;

/// The converter checkpoint filename inside the `converter/` snapshot dir.
pub const CHECKPOINT_FILE: &str = "checkpoint.pth";
/// The converter config filename inside the `converter/` snapshot dir.
pub const CONFIG_FILE: &str = "config.json";

/// A loaded conversion pipeline.
pub struct OpenVoicePipeline {
    ref_enc: ReferenceEncoder,
    converter: VoiceConverter,
    device: Device,
}

impl OpenVoicePipeline {
    /// Build from a `converter/`-shaped snapshot dir (`config.json` + `checkpoint.pth`).
    pub fn from_snapshot(root: &Path, device: &Device) -> Result<Self> {
        validate_config(&root.join(CONFIG_FILE))?;
        let pth = root.join(CHECKPOINT_FILE);
        if !pth.is_file() {
            return Err(AudioError::Msg(format!(
                "openvoice_v2: checkpoint {} missing from the snapshot",
                pth.display()
            )));
        }
        let vb = state_var_builder(&pth, device)?;
        let ref_enc = ReferenceEncoder::new(vb.pp("ref_enc"), device.clone())?;
        let converter = VoiceConverter::new(vb.clone(), device.clone())?;
        Ok(Self {
            ref_enc,
            converter,
            device: device.clone(),
        })
    }

    /// Extract the tone-color embedding `[1, gin, 1]` from one reference clip (any rate, any channel
    /// count already down-mixed to mono by the caller) — the `extract_se` path.
    pub fn extract_tone_color(&self, mono: &[f32], src_rate: u32) -> Result<Tensor> {
        let wav = resample_to_native(mono, src_rate);
        let spec = spectrogram(&wav).ok_or_else(|| {
            AudioError::Msg("openvoice_v2: reference clip too short for a spectrogram frame".into())
        })?;
        self.ref_enc.tone_color(&spec)
    }

    /// Convert `source` (mono, `src_rate`) into the target voice described by tone-color `g_tgt`,
    /// using `g_src` (the source's own tone color) for the forward flow. Returns the converted
    /// waveform at [`config::SAMPLE_RATE`].
    #[allow(clippy::too_many_arguments)]
    pub fn convert(
        &self,
        source: &[f32],
        src_rate: u32,
        g_src: &Tensor,
        g_tgt: &Tensor,
        tau: f32,
        seed: u64,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>> {
        let wav = resample_to_native(source, src_rate);
        let spec = spectrogram(&wav).ok_or_else(|| {
            AudioError::Msg("openvoice_v2: source clip too short for a spectrogram frame".into())
        })?;
        let spec_t = spec_to_tensor(&spec, &self.device)?;
        let noise = gaussian_noise(
            (1, config::INTER_CHANNELS, spec.n_frames),
            seed,
            &self.device,
        )?;
        self.converter
            .voice_conversion(&spec_t, g_src, g_tgt, tau, &noise, cancel)
    }
}

/// Bin-major linear-spectrogram magnitudes → a `[1, spec_channels, T]` tensor (the layout `enc_q`
/// consumes directly, no transpose).
fn spec_to_tensor(spec: &crate::spectrogram::LinearSpectrogram, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        spec.mag.clone(),
        (1, spec.n_bins, spec.n_frames),
        device,
    )?)
}

/// A `[…]`-shaped tensor of deterministic standard-Gaussian samples, seeded so the same request +
/// seed reproduces byte-identical output (the sibling providers' determinism discipline). A small
/// SplitMix64 → Box–Muller generator — the reference draws `torch.randn`; only the temperature `τ`
/// and the posterior statistics are load-bearing for identity, not the exact draw.
fn gaussian_noise(shape: (usize, usize, usize), seed: u64, device: &Device) -> Result<Tensor> {
    let n = shape.0 * shape.1 * shape.2;
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next_u64 = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut unit = || {
        // (0, 1] to keep ln() finite.
        ((next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    };
    let mut vals = Vec::with_capacity(n);
    while vals.len() < n {
        let u1 = unit();
        let u2 = unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        vals.push((r * theta.cos()) as f32);
        if vals.len() < n {
            vals.push((r * theta.sin()) as f32);
        }
    }
    Ok(Tensor::from_vec(vals, shape, device)?)
}

/// Validate the shipped `config.json` against the pinned architecture. A drifted hyperparameter is
/// a typed error, so the port never silently mis-shapes itself against unexpected weights.
fn validate_config(path: &Path) -> Result<()> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| AudioError::Msg(format!("openvoice_v2: read {}: {e}", path.display())))?;
    let cfg: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| AudioError::Msg(format!("openvoice_v2: parse {}: {e}", path.display())))?;
    let data = &cfg["data"];
    let model = &cfg["model"];
    let checks: [(&str, u64, u64); 7] = [
        (
            "data.sampling_rate",
            data["sampling_rate"].as_u64().unwrap_or(0),
            config::SAMPLE_RATE as u64,
        ),
        (
            "data.filter_length",
            data["filter_length"].as_u64().unwrap_or(0),
            config::FILTER_LENGTH as u64,
        ),
        (
            "data.hop_length",
            data["hop_length"].as_u64().unwrap_or(0),
            config::HOP_LENGTH as u64,
        ),
        (
            "data.n_speakers",
            data["n_speakers"].as_u64().unwrap_or(u64::MAX),
            0,
        ),
        (
            "model.inter_channels",
            model["inter_channels"].as_u64().unwrap_or(0),
            config::INTER_CHANNELS as u64,
        ),
        (
            "model.hidden_channels",
            model["hidden_channels"].as_u64().unwrap_or(0),
            config::HIDDEN_CHANNELS as u64,
        ),
        (
            "model.gin_channels",
            model["gin_channels"].as_u64().unwrap_or(0),
            config::GIN_CHANNELS as u64,
        ),
    ];
    for (name, got, want) in checks {
        if got != want {
            return Err(AudioError::Msg(format!(
                "openvoice_v2: {} = {got} in {}, expected {want} — checkpoint architecture drifted \
                 from the pinned port",
                name,
                path.display()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_noise_is_deterministic_and_seed_sensitive() {
        let dev = Device::Cpu;
        let a = gaussian_noise((1, 4, 8), 42, &dev).unwrap();
        let b = gaussian_noise((1, 4, 8), 42, &dev).unwrap();
        let c = gaussian_noise((1, 4, 8), 43, &dev).unwrap();
        let av: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let bv: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        let cv: Vec<f32> = c.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(av, bv, "same seed ⇒ identical noise");
        assert_ne!(av, cv, "different seed ⇒ different noise");
        assert!(av.iter().all(|x| x.is_finite()));
        // Roughly standard-normal: sample mean near 0, std near 1 over 32 draws (loose bounds).
        let mean = av.iter().sum::<f32>() / av.len() as f32;
        assert!(mean.abs() < 0.6, "mean {mean}");
    }

    #[test]
    fn validate_config_rejects_drift() {
        let dir = std::env::temp_dir().join("openvoice-cfg-validate");
        std::fs::create_dir_all(&dir).unwrap();
        let good = dir.join("good.json");
        std::fs::write(
            &good,
            r#"{"data":{"sampling_rate":22050,"filter_length":1024,"hop_length":256,"n_speakers":0},
                "model":{"inter_channels":192,"hidden_channels":192,"gin_channels":256}}"#,
        )
        .unwrap();
        assert!(validate_config(&good).is_ok());
        let bad = dir.join("bad.json");
        std::fs::write(
            &bad,
            r#"{"data":{"sampling_rate":16000,"filter_length":1024,"hop_length":256,"n_speakers":0},
                "model":{"inter_channels":192,"hidden_channels":192,"gin_channels":256}}"#,
        )
        .unwrap();
        assert!(validate_config(&bad).is_err());
    }
}
