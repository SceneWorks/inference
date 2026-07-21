//! Shared test scaffolding (sc-13666): build the named-component [`LoadSpec`] the shipping MMAudio
//! generators now consume, from **env-pointed per-repo snapshot paths** with an F-029 hub-cache
//! fallback. There is no assembled snapshot directory anymore — each of the five components
//! (`clip` / `synchformer` / `dit` / `vae` / `vocoder`) is staged individually via
//! [`LoadSpec::with_component`].
//!
//! ## Test env vars (per-repo snapshot directories)
//!
//! Point these at a **repo snapshot directory**; the in-repo relative path for each checkpoint is
//! joined automatically (so one dir per HF repo covers several components):
//!
//! | Env var | HF repo | Components it supplies |
//! |---------|---------|------------------------|
//! | `MMAUDIO_MMAUDIO_SNAPSHOT` | `hkchengrex/MMAudio` | `synchformer`, `dit`, `vae`, 16k `vocoder` |
//! | `MMAUDIO_CLIP_SNAPSHOT`    | `apple/DFN5B-CLIP-ViT-H-14-384` | `clip` |
//! | `MMAUDIO_BIGVGAN_SNAPSHOT` | `nvidia/bigvgan_v2_44khz_128band_512x` | 44k `vocoder` |
//!
//! When an env var is unset the component falls back to the crate's per-component pinned hub
//! resolver (the F-029 cache path) so a warm HF cache still drives the real-weight run with no
//! configuration. These are TEST-only side channels; production `load()` takes explicit components.
#![allow(dead_code)]

use candle_audio_mmaudio as mm;
use mm::candle_audio::gen_core::{LoadSpec, WeightsSource};

/// `hkchengrex/MMAudio` repo snapshot dir (synchformer + dit + vae + 16k vocoder).
pub const MMAUDIO_SNAPSHOT_ENV: &str = "MMAUDIO_MMAUDIO_SNAPSHOT";
/// `apple/DFN5B-CLIP-ViT-H-14-384` repo snapshot dir (clip).
pub const CLIP_SNAPSHOT_ENV: &str = "MMAUDIO_CLIP_SNAPSHOT";
/// `nvidia/bigvgan_v2_44khz_128band_512x` repo snapshot dir (44k vocoder).
pub const BIGVGAN_SNAPSHOT_ENV: &str = "MMAUDIO_BIGVGAN_SNAPSHOT";

/// If `env` names a directory, join `rel` (the in-repo relative checkpoint path) onto it and return
/// a [`WeightsSource::File`]; otherwise `None` (the caller falls back to the pinned hub resolver).
fn from_env_repo(env: &str, rel: &str) -> Option<WeightsSource> {
    let dir = std::env::var(env).ok()?;
    Some(WeightsSource::File(std::path::PathBuf::from(dir).join(rel)))
}

pub fn clip_source() -> WeightsSource {
    from_env_repo(CLIP_SNAPSHOT_ENV, mm::clip::CLIP_WEIGHTS_PATH)
        .unwrap_or_else(|| mm::clip::resolve_pinned_weights().expect("resolve pinned DFN5B-CLIP"))
}

pub fn synchformer_source() -> WeightsSource {
    from_env_repo(MMAUDIO_SNAPSHOT_ENV, mm::model::WEIGHTS_PATH)
        .unwrap_or_else(|| mm::model::resolve_pinned_weights().expect("resolve pinned Synchformer"))
}

pub fn dit_16k_source() -> WeightsSource {
    from_env_repo(MMAUDIO_SNAPSHOT_ENV, mm::mmdit::WEIGHTS_PATH)
        .unwrap_or_else(|| mm::mmdit::resolve_pinned_weights().expect("resolve pinned 16k MM-DiT"))
}

pub fn vae_16k_source() -> WeightsSource {
    from_env_repo(MMAUDIO_SNAPSHOT_ENV, mm::output::VAE_WEIGHTS_PATH)
        .unwrap_or_else(|| mm::output::resolve_pinned_vae().expect("resolve pinned 16k mel-VAE"))
}

pub fn vocoder_16k_source() -> WeightsSource {
    from_env_repo(MMAUDIO_SNAPSHOT_ENV, mm::output::BIGVGAN_WEIGHTS_PATH).unwrap_or_else(|| {
        mm::output::resolve_pinned_bigvgan().expect("resolve pinned 16k BigVGAN")
    })
}

pub fn dit_44k_source() -> WeightsSource {
    from_env_repo(MMAUDIO_SNAPSHOT_ENV, mm::mmdit::WEIGHTS_PATH_44K).unwrap_or_else(|| {
        mm::mmdit::resolve_pinned_weights_large_44k_v2()
            .expect("resolve pinned large_44k_v2 MM-DiT")
    })
}

pub fn vae_44k_source() -> WeightsSource {
    from_env_repo(MMAUDIO_SNAPSHOT_ENV, mm::output::VAE_WEIGHTS_PATH_44K).unwrap_or_else(|| {
        mm::output::resolve_pinned_vae_44k().expect("resolve pinned 44k mel-VAE")
    })
}

pub fn vocoder_44k_source() -> WeightsSource {
    from_env_repo(BIGVGAN_SNAPSHOT_ENV, mm::output::BIGVGAN_V2_WEIGHTS_PATH).unwrap_or_else(|| {
        mm::output::resolve_pinned_bigvgan_v2().expect("resolve pinned NVIDIA BigVGAN v2")
    })
}

/// A placeholder base `weights` for the spec. mmaudio consumes only the five named components and
/// ignores `spec.weights`, so this is never read.
fn placeholder_weights() -> WeightsSource {
    WeightsSource::Dir(std::env::temp_dir().join("mmaudio-unused-base"))
}

/// The `mmaudio_small_16k` [`LoadSpec`] with all five components staged from env / hub.
pub fn spec_16k() -> LoadSpec {
    LoadSpec::new(placeholder_weights())
        .with_component("clip", clip_source())
        .with_component("synchformer", synchformer_source())
        .with_component("dit", dit_16k_source())
        .with_component("vae", vae_16k_source())
        .with_component("vocoder", vocoder_16k_source())
}

/// The `mmaudio_large_44k` [`LoadSpec`] with all five components staged from env / hub.
pub fn spec_44k() -> LoadSpec {
    LoadSpec::new(placeholder_weights())
        .with_component("clip", clip_source())
        .with_component("synchformer", synchformer_source())
        .with_component("dit", dit_44k_source())
        .with_component("vae", vae_44k_source())
        .with_component("vocoder", vocoder_44k_source())
}
