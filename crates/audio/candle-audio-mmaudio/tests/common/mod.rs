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
//! Each env var is **required**: inference never self-fetches or derives a cache location
//! (epic 13657), so the real-weight harness must point every component at a pre-materialized
//! snapshot. These are TEST-only side channels; production `load()` takes explicit components.
#![allow(dead_code)]

use candle_audio_mmaudio as mm;
use mm::candle_audio::gen_core::{LoadSpec, WeightsSource};

/// `hkchengrex/MMAudio` repo snapshot dir (synchformer + dit + vae + 16k vocoder).
pub const MMAUDIO_SNAPSHOT_ENV: &str = "MMAUDIO_MMAUDIO_SNAPSHOT";
/// `apple/DFN5B-CLIP-ViT-H-14-384` repo snapshot dir (clip).
pub const CLIP_SNAPSHOT_ENV: &str = "MMAUDIO_CLIP_SNAPSHOT";
/// `nvidia/bigvgan_v2_44khz_128band_512x` repo snapshot dir (44k vocoder).
pub const BIGVGAN_SNAPSHOT_ENV: &str = "MMAUDIO_BIGVGAN_SNAPSHOT";

/// Read the **required** repo-snapshot dir from `env`, join `rel` (the in-repo relative checkpoint
/// path) onto it, and return a [`WeightsSource::File`]. Panics with an actionable message when unset
/// — inference never self-fetches or derives a cache location (epic 13657).
fn from_env_repo(env: &str, rel: &str) -> WeightsSource {
    let dir = std::env::var(env)
        .unwrap_or_else(|_| panic!("set {env} to the repo snapshot dir supplying {rel}"));
    WeightsSource::File(std::path::PathBuf::from(dir).join(rel))
}

pub fn clip_source() -> WeightsSource {
    from_env_repo(CLIP_SNAPSHOT_ENV, mm::clip::CLIP_WEIGHTS_PATH)
}

pub fn synchformer_source() -> WeightsSource {
    from_env_repo(MMAUDIO_SNAPSHOT_ENV, mm::model::WEIGHTS_PATH)
}

pub fn dit_16k_source() -> WeightsSource {
    from_env_repo(MMAUDIO_SNAPSHOT_ENV, mm::mmdit::WEIGHTS_PATH)
}

pub fn vae_16k_source() -> WeightsSource {
    from_env_repo(MMAUDIO_SNAPSHOT_ENV, mm::output::VAE_WEIGHTS_PATH)
}

pub fn vocoder_16k_source() -> WeightsSource {
    from_env_repo(MMAUDIO_SNAPSHOT_ENV, mm::output::BIGVGAN_WEIGHTS_PATH)
}

pub fn dit_44k_source() -> WeightsSource {
    from_env_repo(MMAUDIO_SNAPSHOT_ENV, mm::mmdit::WEIGHTS_PATH_44K)
}

pub fn vae_44k_source() -> WeightsSource {
    from_env_repo(MMAUDIO_SNAPSHOT_ENV, mm::output::VAE_WEIGHTS_PATH_44K)
}

pub fn vocoder_44k_source() -> WeightsSource {
    from_env_repo(BIGVGAN_SNAPSHOT_ENV, mm::output::BIGVGAN_V2_WEIGHTS_PATH)
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
