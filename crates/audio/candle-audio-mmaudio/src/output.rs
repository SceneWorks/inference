//! MMAudio's **16 kHz output path** (sc-13440): pinned-checkpoint resolution, weight-license
//! surface, and the loaders that assemble the two-stage latent→mel→waveform decoder — the
//! [`crate::vae`] mel-VAE decoder followed by the [`crate::bigvgan`] vocoder.
//!
//! This is the audio-generation tail of MMAudio (the DiT that produces the latent is a later
//! slice). Everything here is **model-internal and UNREGISTERED** — no shipping generator this
//! slice, mirroring how the Synchformer encoder (sc-13438) stayed unregistered until its generator
//! lands.
//!
//! ## Mel configuration (16k)
//!
//! The mel domain the VAE decodes into and BigVGAN vocodes from is MMAudio's 16k `MelConverter`
//! spec: `sampling_rate=16000`, `n_fft=1024`, `num_mels=80`, `hop_size=256`, `win_size=1024`,
//! `fmin=0`, `fmax=8000`, `norm_fn=log10`. The output path never runs the forward STFT (that is the
//! *analysis* path), but the constants are surfaced for consumers that pair this decoder with a
//! matching analysis front-end.

use std::path::{Path, PathBuf};

use candle_audio::candle_core::{DType, Device, Result as CResult, Tensor};
use candle_audio::gen_core::{WeightLicense, WeightLicenseEntry, WeightsSource};
use candle_audio::hub::hf_get_pinned;
use candle_audio::{AudioError, Result};
use candle_nn::VarBuilder;

use crate::bigvgan::BigVganVocoder;
use crate::vae::MelVaeDecoder;

/// Hub pin: MMAudio's model repo (mirrors the GitHub-release `ext_weights/`). Same immutable commit
/// SHA the Synchformer slice pins (F-029 discipline).
pub const HUB_REPO: &str = "hkchengrex/MMAudio";
pub const HUB_REVISION: &str = "eb13a1a98fdbec91753775c57b074ccdfc60587c";

/// The 16k mel-VAE checkpoint (`v1-16.pth`, ~687 MB) inside the pinned repo.
pub const VAE_WEIGHTS_PATH: &str = "ext_weights/v1-16.pth";
/// The 16k BigVGAN generator checkpoint (`best_netG.pt`, ~449 MB) inside the pinned repo.
pub const BIGVGAN_WEIGHTS_PATH: &str = "ext_weights/best_netG.pt";
/// The state-dict key the BigVGAN generator is nested under in `best_netG.pt`.
pub const BIGVGAN_STATE_KEY: &str = "generator";

/// Stable identity of the 16k mel-VAE decoder (weight-license entry key). Not a shipping provider id.
pub const VAE_MODEL_ID: &str = "mmaudio_vae_16k";
/// Stable identity of the 16k BigVGAN vocoder (weight-license entry key). Not a shipping provider id.
pub const BIGVGAN_MODEL_ID: &str = "mmaudio_bigvgan_16k";

// ---- Mel configuration (16k) ------------------------------------------------------------------

/// Waveform sample rate the 16k path produces (Hz).
pub const SAMPLE_RATE: usize = 16_000;
/// STFT window / FFT size of the paired 16k mel analysis (`n_fft`).
pub const N_FFT: usize = 1024;
/// Mel bands (`num_mels`).
pub const NUM_MELS: usize = 80;
/// STFT hop (`hop_size`) — also BigVGAN's total upsampling factor, so `waveform_len = 256·mel_len`.
pub const HOP_SIZE: usize = 256;
/// STFT window length (`win_size`).
pub const WIN_SIZE: usize = 1024;
/// Mel lower bound (`fmin`, Hz).
pub const FMIN: f32 = 0.0;
/// Mel upper bound (`fmax`, Hz).
pub const FMAX: f32 = 8_000.0;

// ---- Weight licenses (sc-13332) ---------------------------------------------------------------

/// Non-commercial restriction note shared by both 16k checkpoints (MMAudio releases every HF
/// checkpoint under CC-BY-NC-4.0).
const NC_RESTRICTION: &str = "Non-commercial only: MMAudio releases all ext_weights checkpoints \
    under CC-BY-NC-4.0 (see the MMAudio README). Additionally the pretrained models were trained on \
    AudioSet/VGGSound/Freesound/AudioCaps/WavCaps, whose dataset terms a downstream user must honor; \
    MMAudio states it does not guarantee suitability for commercial use.";

/// License of the pinned 16k mel-VAE checkpoint (`v1-16.pth`).
///
/// **CC-BY-NC-4.0** — verified against MMAudio's README (the checkpoints are released on Hugging
/// Face under CC-BY-NC-4.0). The VAE *architecture* originates from Make-An-Audio 2 (ByteDance, MIT
/// code) and its EDM2 magnitude-preserving primitives derive from NVIDIA's EDM2 code
/// (CC-BY-NC-SA-4.0); the distributed **weights** are governed by MMAudio's non-commercial
/// checkpoint license. SceneWorks is non-commercial, so the weights are usable, but the restriction
/// MUST be surfaced.
pub const VAE_WEIGHT_LICENSE: WeightLicense = WeightLicense {
    spdx_id: "CC-BY-NC-4.0",
    name: "Creative Commons Attribution-NonCommercial 4.0 International",
    source_url: "https://huggingface.co/hkchengrex/MMAudio",
    attribution: Some(
        "MMAudio 16k mel-VAE (v1-16.pth) © Sony Research Inc. — released under CC-BY-NC-4.0; VAE \
         architecture from Make-An-Audio 2 (ByteDance, MIT); EDM2 magnitude-preserving primitives \
         © NVIDIA (CC-BY-NC-SA-4.0)",
    ),
    commercial_use: false,
    restriction: Some(NC_RESTRICTION),
};

/// License of the pinned 16k BigVGAN checkpoint (`best_netG.pt`).
///
/// **CC-BY-NC-4.0** — verified against MMAudio's README (checkpoints released under CC-BY-NC-4.0).
/// This is the Make-An-Audio 2 16k BigVGAN; the BigVGAN *code* is NVIDIA MIT (adapted from HiFi-GAN,
/// MIT), but the distributed **weights** are governed by MMAudio's non-commercial checkpoint
/// license. Usable for the non-commercial product with the restriction surfaced.
pub const BIGVGAN_WEIGHT_LICENSE: WeightLicense = WeightLicense {
    spdx_id: "CC-BY-NC-4.0",
    name: "Creative Commons Attribution-NonCommercial 4.0 International",
    source_url: "https://huggingface.co/hkchengrex/MMAudio",
    attribution: Some(
        "MMAudio 16k BigVGAN (best_netG.pt) © Sony Research Inc. — released under CC-BY-NC-4.0; 16k \
         BigVGAN pretrained model from Make-An-Audio 2 (ByteDance, MIT); BigVGAN code © NVIDIA \
         (MIT), adapted from HiFi-GAN (MIT)",
    ),
    commercial_use: false,
    restriction: Some(NC_RESTRICTION),
};

/// Weight-license entry for the 16k mel-VAE (keyed by [`VAE_MODEL_ID`]).
pub const VAE_WEIGHT_LICENSE_ENTRY: WeightLicenseEntry = WeightLicenseEntry {
    provider_id: VAE_MODEL_ID,
    license: VAE_WEIGHT_LICENSE,
};

/// Weight-license entry for the 16k BigVGAN (keyed by [`BIGVGAN_MODEL_ID`]).
pub const BIGVGAN_WEIGHT_LICENSE_ENTRY: WeightLicenseEntry = WeightLicenseEntry {
    provider_id: BIGVGAN_MODEL_ID,
    license: BIGVGAN_WEIGHT_LICENSE,
};

// ---- Pinned resolution --------------------------------------------------------------------------

/// Resolve the pinned 16k mel-VAE checkpoint through the audio lane's F-029 hub path.
pub fn resolve_pinned_vae() -> Result<WeightsSource> {
    Ok(WeightsSource::File(hf_get_pinned(
        HUB_REPO,
        HUB_REVISION,
        VAE_WEIGHTS_PATH,
    )?))
}

/// Resolve the pinned 16k BigVGAN checkpoint through the audio lane's F-029 hub path.
pub fn resolve_pinned_bigvgan() -> Result<WeightsSource> {
    Ok(WeightsSource::File(hf_get_pinned(
        HUB_REPO,
        HUB_REVISION,
        BIGVGAN_WEIGHTS_PATH,
    )?))
}

fn source_to_path(source: &WeightsSource, filename: &str, nested: &str) -> PathBuf {
    match source {
        WeightsSource::File(p) => p.clone(),
        WeightsSource::Dir(d) => {
            let nested_path = d.join(nested);
            if nested_path.exists() {
                nested_path
            } else {
                d.join(filename)
            }
        }
    }
}

/// Load the 16k mel-VAE decoder from a `v1-16.pth` file path (weights load as f32, CPU-first).
pub fn load_vae_from_pth(weights: &Path, device: &Device) -> Result<MelVaeDecoder> {
    if !weights.exists() {
        return Err(AudioError::Msg(format!(
            "{VAE_MODEL_ID}: weights file {} not found (resolve_pinned_vae materializes {VAE_WEIGHTS_PATH})",
            weights.display()
        )));
    }
    let vb = VarBuilder::from_pth(weights, DType::F32, device).map_err(AudioError::from)?;
    MelVaeDecoder::load(vb).map_err(AudioError::from)
}

/// Load the 16k BigVGAN vocoder from a `best_netG.pt` file path. The generator state dict is nested
/// under the `generator` key.
pub fn load_bigvgan_from_pth(weights: &Path, device: &Device) -> Result<BigVganVocoder> {
    if !weights.exists() {
        return Err(AudioError::Msg(format!(
            "{BIGVGAN_MODEL_ID}: weights file {} not found (resolve_pinned_bigvgan materializes {BIGVGAN_WEIGHTS_PATH})",
            weights.display()
        )));
    }
    let vb = VarBuilder::from_pth_with_state(weights, DType::F32, BIGVGAN_STATE_KEY, device)
        .map_err(AudioError::from)?;
    BigVganVocoder::load(vb).map_err(AudioError::from)
}

/// Load the mel-VAE decoder from a [`WeightsSource`].
pub fn load_vae(source: &WeightsSource, device: &Device) -> Result<MelVaeDecoder> {
    let path = source_to_path(source, "v1-16.pth", VAE_WEIGHTS_PATH);
    load_vae_from_pth(&path, device)
}

/// Load the BigVGAN vocoder from a [`WeightsSource`].
pub fn load_bigvgan(source: &WeightsSource, device: &Device) -> Result<BigVganVocoder> {
    let path = source_to_path(source, "best_netG.pt", BIGVGAN_WEIGHTS_PATH);
    load_bigvgan_from_pth(&path, device)
}

/// The assembled MMAudio 16k output decoder: latent → mel (VAE) → 16 kHz waveform (BigVGAN).
pub struct AudioDecoder16k {
    vae: MelVaeDecoder,
    vocoder: BigVganVocoder,
    device: Device,
}

impl AudioDecoder16k {
    /// Assemble from the two pinned checkpoints already resolved to files.
    pub fn load_from_paths(vae_pth: &Path, bigvgan_pth: &Path, device: &Device) -> Result<Self> {
        let vae = load_vae_from_pth(vae_pth, device)?;
        let vocoder = load_bigvgan_from_pth(bigvgan_pth, device)?;
        Ok(Self {
            vae,
            vocoder,
            device: device.clone(),
        })
    }

    /// Assemble from two [`WeightsSource`]s.
    pub fn load(
        vae_source: &WeightsSource,
        bigvgan_source: &WeightsSource,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            vae: load_vae(vae_source, device)?,
            vocoder: load_bigvgan(bigvgan_source, device)?,
            device: device.clone(),
        })
    }

    /// The compute device the weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Decode a latent `(B, 20, L)` to an unnormalized 80-band mel `(B, 80, 2L)`.
    pub fn decode_latent(&self, latent: &Tensor) -> CResult<Tensor> {
        self.vae.decode(latent)
    }

    /// Vocode an 80-band mel `(B, 80, T)` to a 16 kHz waveform `(B, 1, 256·T)`.
    pub fn vocode(&self, mel: &Tensor) -> CResult<Tensor> {
        self.vocoder.forward(mel)
    }

    /// Full 16k output path: latent `(B, 20, L)` → 16 kHz waveform `(B, 1, 512·L)`.
    pub fn latent_to_waveform(&self, latent: &Tensor) -> CResult<Tensor> {
        let mel = self.decode_latent(latent)?;
        self.vocode(&mel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_licenses_are_well_formed_non_commercial() {
        assert!(VAE_WEIGHT_LICENSE.is_well_formed());
        assert!(BIGVGAN_WEIGHT_LICENSE.is_well_formed());
        assert!(!VAE_WEIGHT_LICENSE.is_permissive());
        assert!(!BIGVGAN_WEIGHT_LICENSE.is_permissive());
        assert_eq!(VAE_WEIGHT_LICENSE.spdx_id, "CC-BY-NC-4.0");
        assert_eq!(BIGVGAN_WEIGHT_LICENSE.spdx_id, "CC-BY-NC-4.0");
        assert!(VAE_WEIGHT_LICENSE.restriction.is_some());
        assert!(BIGVGAN_WEIGHT_LICENSE.restriction.is_some());
        assert_eq!(VAE_WEIGHT_LICENSE_ENTRY.provider_id, VAE_MODEL_ID);
        assert_eq!(BIGVGAN_WEIGHT_LICENSE_ENTRY.provider_id, BIGVGAN_MODEL_ID);
    }

    #[test]
    fn hub_revision_is_a_full_commit_sha() {
        assert_eq!(HUB_REVISION.len(), 40);
        assert!(HUB_REVISION.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn mel_config_matches_16k_reference() {
        assert_eq!(SAMPLE_RATE, 16_000);
        assert_eq!(N_FFT, 1024);
        assert_eq!(NUM_MELS, 80);
        assert_eq!(HOP_SIZE, 256);
        assert_eq!(HOP_SIZE, crate::bigvgan::HOP);
    }

    #[test]
    fn missing_weights_files_error_clearly() {
        let dev = Device::Cpu;
        let e = match load_vae_from_pth(Path::new("/nonexistent/v1-16.pth"), &dev) {
            Ok(_) => panic!("loading a nonexistent VAE path must fail"),
            Err(e) => e,
        };
        assert!(e.to_string().contains("not found"));
        let e = match load_bigvgan_from_pth(Path::new("/nonexistent/best_netG.pt"), &dev) {
            Ok(_) => panic!("loading a nonexistent BigVGAN path must fail"),
            Err(e) => e,
        };
        assert!(e.to_string().contains("not found"));
    }
}
