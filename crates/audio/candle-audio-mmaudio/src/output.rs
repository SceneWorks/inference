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
use candle_audio::gen_core::{ComponentLicense, WeightsSource};
use candle_audio::{AudioError, Result};
use candle_nn::VarBuilder;

use crate::bigvgan::{self, BigVganVocoder};
use crate::vae::{self, MelVaeDecoder};

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

/// The 44k mel-VAE checkpoint (`v1-44.pth`, ~1.22 GB) inside the pinned MMAudio repo (sc-13441).
pub const VAE_WEIGHTS_PATH_44K: &str = "ext_weights/v1-44.pth";

/// Hub pin: NVIDIA's **BigVGAN v2** repo — a **separate** HF model from `hkchengrex/MMAudio`. The
/// 44k output path vocodes with `nvidia/bigvgan_v2_44khz_128band_512x` (128-band mel, n_fft 2048,
/// hop 512), pinned to an immutable commit SHA (F-029 discipline).
pub const BIGVGAN_V2_HUB_REPO: &str = "nvidia/bigvgan_v2_44khz_128band_512x";
pub const BIGVGAN_V2_HUB_REVISION: &str = "95a9d1dcb12906c03edd938d77b9333d6ded7dfb";
/// The NVIDIA BigVGAN v2 generator checkpoint (`bigvgan_generator.pt`, ~489 MB).
pub const BIGVGAN_V2_WEIGHTS_PATH: &str = "bigvgan_generator.pt";
/// The state-dict key the NVIDIA BigVGAN v2 generator is nested under in `bigvgan_generator.pt`.
pub const BIGVGAN_V2_STATE_KEY: &str = "generator";

/// Stable identity of the 16k mel-VAE decoder (weight-license entry key). Not a shipping provider id.
pub const VAE_MODEL_ID: &str = "mmaudio_vae_16k";
/// Stable identity of the 16k BigVGAN vocoder (weight-license entry key). Not a shipping provider id.
pub const BIGVGAN_MODEL_ID: &str = "mmaudio_bigvgan_16k";
/// Stable identity of the 44k mel-VAE decoder (weight-license entry key). Not a shipping provider id.
pub const VAE_MODEL_ID_44K: &str = "mmaudio_vae_44k";
/// Stable identity of the NVIDIA BigVGAN v2 44k vocoder (weight-license entry key). Not a provider id.
pub const BIGVGAN_V2_MODEL_ID: &str = "nvidia_bigvgan_v2_44khz_128band_512x";

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

// ---- Mel configuration (44k) ------------------------------------------------------------------

/// Waveform sample rate the 44k path produces (Hz).
pub const SAMPLE_RATE_44K: usize = 44_100;
/// STFT window / FFT size of the paired 44k mel analysis (`n_fft`, NVIDIA BigVGAN v2 config.json).
pub const N_FFT_44K: usize = 2048;
/// Mel bands the 44k path uses (`num_mels`).
pub const NUM_MELS_44K: usize = 128;
/// STFT hop (`hop_size`) — also BigVGAN v2's total upsampling factor (`waveform_len = 512·mel_len`).
pub const HOP_SIZE_44K: usize = 512;
/// STFT window length (`win_size`).
pub const WIN_SIZE_44K: usize = 2048;
/// Mel lower bound (`fmin`, Hz).
pub const FMIN_44K: f32 = 0.0;
/// Mel upper bound (`fmax`, Hz) — `fmax=null` in the NVIDIA config resolves to `sampling_rate / 2`.
pub const FMAX_44K: f32 = 22_050.0;

// ---- Weight licenses (sc-13332) ---------------------------------------------------------------

/// Component key for the pinned 16k mel-VAE checkpoint (`v1-16.pth`).
pub const VAE_COMPONENT_KEY: &str = "mmaudio_vae_16k";

/// The schema-3 licence row for the pinned 16k mel-VAE checkpoint (`v1-16.pth`). `declared` was read from the `hkchengrex/MMAudio` model
/// card on `retrieved`; the architecture credits in the attribution are provenance, not terms.
pub const VAE_COMPONENT_LICENSE: ComponentLicense = ComponentLicense {
    component: VAE_COMPONENT_KEY,
    source_url: "https://huggingface.co/hkchengrex/MMAudio",
    gated: false,
    declared: "cc-by-nc-4.0",
    family: "cc-by-nc-4-0",
    attribution: Some(
        "MMAudio 16k mel-VAE (v1-16.pth) © Sony Research Inc. — released under CC-BY-NC-4.0; VAE \
         architecture from Make-An-Audio 2 (ByteDance, MIT); EDM2 magnitude-preserving primitives \
         © NVIDIA (CC-BY-NC-SA-4.0)",
    ),
    retrieved: "2026-08-02",
};

/// Component key for the pinned 16k BigVGAN checkpoint (`best_netG.pt`).
pub const BIGVGAN_COMPONENT_KEY: &str = "mmaudio_bigvgan_16k";

/// The schema-3 licence row for the pinned 16k BigVGAN checkpoint (`best_netG.pt`). `declared` was read from the `hkchengrex/MMAudio` model
/// card on `retrieved`; the architecture credits in the attribution are provenance, not terms.
pub const BIGVGAN_COMPONENT_LICENSE: ComponentLicense = ComponentLicense {
    component: BIGVGAN_COMPONENT_KEY,
    source_url: "https://huggingface.co/hkchengrex/MMAudio",
    gated: false,
    declared: "cc-by-nc-4.0",
    family: "cc-by-nc-4-0",
    attribution: Some(
        "MMAudio 16k BigVGAN (best_netG.pt) © Sony Research Inc. — released under CC-BY-NC-4.0; 16k \
         BigVGAN pretrained model from Make-An-Audio 2 (ByteDance, MIT); BigVGAN code © NVIDIA \
         (MIT), adapted from HiFi-GAN (MIT)",
    ),
    retrieved: "2026-08-02",
};

/// Component key for the pinned 44k mel-VAE checkpoint (`v1-44.pth`).
pub const VAE_COMPONENT_KEY_44K: &str = "mmaudio_vae_44k";

/// The schema-3 licence row for the pinned 44k mel-VAE checkpoint (`v1-44.pth`). `declared` was read from the `hkchengrex/MMAudio` model
/// card on `retrieved`; the architecture credits in the attribution are provenance, not terms.
pub const VAE_COMPONENT_LICENSE_44K: ComponentLicense = ComponentLicense {
    component: VAE_COMPONENT_KEY_44K,
    source_url: "https://huggingface.co/hkchengrex/MMAudio",
    gated: false,
    declared: "cc-by-nc-4.0",
    family: "cc-by-nc-4-0",
    attribution: Some(
        "MMAudio 44k mel-VAE (v1-44.pth) © Sony Research Inc. — released under CC-BY-NC-4.0; VAE \
         architecture from Make-An-Audio 2 (ByteDance, MIT); EDM2 magnitude-preserving primitives \
         © NVIDIA (CC-BY-NC-SA-4.0)",
    ),
    retrieved: "2026-08-02",
};

/// Component key for the NVIDIA BigVGAN v2 44k vocoder artifact.
pub const BIGVGAN_V2_COMPONENT_KEY: &str = "nvidia_bigvgan_v2_44khz_128band_512x";

/// The schema-3 licence row for the pinned NVIDIA **BigVGAN v2** 44k vocoder
/// (`bigvgan_generator.pt`).
///
/// `declared` is the `nvidia/bigvgan_v2_44khz_128band_512x` model card's `license: mit`, read on
/// `retrieved`. Note this resolves to the plain `mit` family and **not** to either NVIDIA-specific
/// family in the table: the artifact declares MIT, and the NVIDIA Open Model / NSCLv1 texts govern
/// other NVIDIA releases.
pub const BIGVGAN_V2_COMPONENT_LICENSE: ComponentLicense = ComponentLicense {
    component: BIGVGAN_V2_COMPONENT_KEY,
    source_url: "https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x",
    gated: false,
    declared: "mit",
    family: "mit",
    attribution: Some(
        "NVIDIA BigVGAN v2 (bigvgan_v2_44khz_128band_512x, bigvgan_generator.pt) © 2024 NVIDIA \
         CORPORATION — MIT License (model card license: mit); BigVGAN code adapted from HiFi-GAN \
         (jik876, MIT).",
    ),
    retrieved: "2026-08-02",
};

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
            "{VAE_MODEL_ID}: weights file {} not found (pass {VAE_WEIGHTS_PATH} in via the LoadSpec)",
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
            "{BIGVGAN_MODEL_ID}: weights file {} not found (pass {BIGVGAN_WEIGHTS_PATH} in via the LoadSpec)",
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

/// Load the 44k mel-VAE decoder from a `v1-44.pth` file path (sc-13441). Same pickle layout as the
/// 16k VAE, loaded with the [`vae::Config::vae_44k`] dims (40-d latent → 128-band mel, hidden 512).
pub fn load_vae_44k_from_pth(weights: &Path, device: &Device) -> Result<MelVaeDecoder> {
    if !weights.exists() {
        return Err(AudioError::Msg(format!(
            "{VAE_MODEL_ID_44K}: weights file {} not found (pass {VAE_WEIGHTS_PATH_44K} in via the LoadSpec)",
            weights.display()
        )));
    }
    let vb = VarBuilder::from_pth(weights, DType::F32, device).map_err(AudioError::from)?;
    MelVaeDecoder::load_with_config(vb, vae::Config::vae_44k()).map_err(AudioError::from)
}

/// Load the NVIDIA BigVGAN v2 44k vocoder from a `bigvgan_generator.pt` file path (sc-13441). The
/// generator state dict is nested under the `generator` key, loaded with the
/// [`bigvgan::Config::bigvgan_v2_44khz_128band_512x`] hyperparameters.
pub fn load_bigvgan_v2_from_pth(weights: &Path, device: &Device) -> Result<BigVganVocoder> {
    if !weights.exists() {
        return Err(AudioError::Msg(format!(
            "{BIGVGAN_V2_MODEL_ID}: weights file {} not found (pass {BIGVGAN_V2_WEIGHTS_PATH} in via the LoadSpec)",
            weights.display()
        )));
    }
    let vb = VarBuilder::from_pth_with_state(weights, DType::F32, BIGVGAN_V2_STATE_KEY, device)
        .map_err(AudioError::from)?;
    BigVganVocoder::load_with_config(vb, bigvgan::Config::bigvgan_v2_44khz_128band_512x())
        .map_err(AudioError::from)
}

/// Load the 44k mel-VAE decoder from a [`WeightsSource`].
pub fn load_vae_44k(source: &WeightsSource, device: &Device) -> Result<MelVaeDecoder> {
    let path = source_to_path(source, "v1-44.pth", VAE_WEIGHTS_PATH_44K);
    load_vae_44k_from_pth(&path, device)
}

/// Load the NVIDIA BigVGAN v2 44k vocoder from a [`WeightsSource`].
pub fn load_bigvgan_v2(source: &WeightsSource, device: &Device) -> Result<BigVganVocoder> {
    let path = source_to_path(source, "bigvgan_generator.pt", BIGVGAN_V2_WEIGHTS_PATH);
    load_bigvgan_v2_from_pth(&path, device)
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

/// The assembled MMAudio **44.1 kHz** output decoder (sc-13441): latent `(B, 40, L)` → 128-band mel
/// (44k VAE) → 44.1 kHz waveform (NVIDIA BigVGAN v2). The VAE upsamples ×2 (`mel_len = 2·L`) and the
/// vocoder ×512 (`waveform_len = 512·mel_len = 1024·L`).
pub struct AudioDecoder44k {
    vae: MelVaeDecoder,
    vocoder: BigVganVocoder,
    device: Device,
}

impl AudioDecoder44k {
    /// Assemble from the two pinned checkpoints already resolved to files (the MMAudio `v1-44.pth`
    /// and the NVIDIA `bigvgan_generator.pt`).
    pub fn load_from_paths(vae_pth: &Path, bigvgan_pth: &Path, device: &Device) -> Result<Self> {
        let vae = load_vae_44k_from_pth(vae_pth, device)?;
        let vocoder = load_bigvgan_v2_from_pth(bigvgan_pth, device)?;
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
            vae: load_vae_44k(vae_source, device)?,
            vocoder: load_bigvgan_v2(bigvgan_source, device)?,
            device: device.clone(),
        })
    }

    /// The compute device the weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Decode a latent `(B, 40, L)` to an unnormalized 128-band mel `(B, 128, 2L)`.
    pub fn decode_latent(&self, latent: &Tensor) -> CResult<Tensor> {
        self.vae.decode(latent)
    }

    /// Vocode a 128-band mel `(B, 128, T)` to a 44.1 kHz waveform `(B, 1, 512·T)`.
    pub fn vocode(&self, mel: &Tensor) -> CResult<Tensor> {
        self.vocoder.forward(mel)
    }

    /// Full 44k output path: latent `(B, 40, L)` → 44.1 kHz waveform `(B, 1, 1024·L)`.
    pub fn latent_to_waveform(&self, latent: &Tensor) -> CResult<Tensor> {
        let mel = self.decode_latent(latent)?;
        self.vocode(&mel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sixteen_k_component_licences_resolve_to_the_cc_by_nc_family() {
        use candle_audio::gen_core::{resolve_family, LicenseTerm, LICENSE_FAMILIES};
        for row in [VAE_COMPONENT_LICENSE, BIGVGAN_COMPONENT_LICENSE] {
            assert!(row.is_well_formed(LICENSE_FAMILIES), "{row:?}");
            assert_eq!(row.declared, "cc-by-nc-4.0");
            let family = resolve_family(LICENSE_FAMILIES, row.family).unwrap();
            assert_eq!(family.spdx_id, "CC-BY-NC-4.0");
            assert!(family.imposes(LicenseTerm::NonCommercialWeights));
        }
        assert_eq!(VAE_COMPONENT_LICENSE.component, VAE_MODEL_ID);
        assert_eq!(BIGVGAN_COMPONENT_LICENSE.component, BIGVGAN_MODEL_ID);
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
    fn mel_config_matches_44k_reference() {
        assert_eq!(SAMPLE_RATE_44K, 44_100);
        assert_eq!(N_FFT_44K, 2048);
        assert_eq!(NUM_MELS_44K, 128);
        assert_eq!(HOP_SIZE_44K, 512);
        assert_eq!(WIN_SIZE_44K, 2048);
        assert!((FMAX_44K - 22_050.0).abs() < 1e-3, "fmax=null → sr/2");
        // The 44k mel band count matches the 44k VAE data_dim and the NVIDIA vocoder num_mels.
        assert_eq!(NUM_MELS_44K, crate::vae::DATA_DIM_44K);
        assert_eq!(
            HOP_SIZE_44K,
            crate::bigvgan::Config::bigvgan_v2_44khz_128band_512x().hop()
        );
    }

    #[test]
    fn component_licence_rows_resolve_to_their_declared_families() {
        use candle_audio::gen_core::{resolve_family, LicenseTerm, LICENSE_FAMILIES};
        for row in [
            VAE_COMPONENT_LICENSE,
            BIGVGAN_COMPONENT_LICENSE,
            VAE_COMPONENT_LICENSE_44K,
            BIGVGAN_V2_COMPONENT_LICENSE,
        ] {
            assert!(row.is_well_formed(LICENSE_FAMILIES), "{row:?}");
            assert!(resolve_family(LICENSE_FAMILIES, row.family).is_some());
        }
        // The three MMAudio checkpoints declare CC-BY-NC-4.0, whose text restricts the weights.
        for row in [
            VAE_COMPONENT_LICENSE,
            BIGVGAN_COMPONENT_LICENSE,
            VAE_COMPONENT_LICENSE_44K,
        ] {
            assert_eq!(row.family, "cc-by-nc-4-0");
            let family = resolve_family(LICENSE_FAMILIES, row.family).unwrap();
            assert!(family.imposes(LicenseTerm::NonCommercialWeights));
        }
        // NVIDIA BigVGAN v2 declares MIT — a different family, which states no such term. It neither
        // relaxes nor tightens what the checkpoints beside it carry; the derived union is a set
        // union, not a strictest-wins collapse.
        assert_eq!(BIGVGAN_V2_COMPONENT_LICENSE.family, "mit");
        let mit = resolve_family(LICENSE_FAMILIES, "mit").unwrap();
        assert!(!mit.imposes(LicenseTerm::NonCommercialWeights));
        assert!(BIGVGAN_V2_COMPONENT_LICENSE
            .source_url
            .starts_with("https://huggingface.co/"));
    }
    #[test]
    fn bigvgan_v2_hub_revision_is_a_full_commit_sha() {
        assert_eq!(BIGVGAN_V2_HUB_REVISION.len(), 40);
        assert!(BIGVGAN_V2_HUB_REVISION
            .chars()
            .all(|c| c.is_ascii_hexdigit()));
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
