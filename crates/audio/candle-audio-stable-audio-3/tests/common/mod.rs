//! Shared real-weight test-harness helpers for the Stable Audio 3 integration tests (sc-27350).
//!
//! Each test target includes this with `mod common;` and uses only what it needs, so unused
//! helpers are expected per target. These are setup helpers only — snapshot-path resolution,
//! device selection and mapping committed or pinned artifacts — never a model-execution seam.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use candle_audio_stable_audio_3::candle_audio::candle_core::safetensors::MmapedSafetensors;
use candle_audio_stable_audio_3::candle_audio::candle_core::{DType, Device};
use candle_audio_stable_audio_3::candle_audio::weights;
use candle_audio_stable_audio_3::gen_core::WeightsSource;
use candle_nn::VarBuilder;

/// The pinned immutable snapshot directory named by the `env` variable. A missing variable is a
/// hard failure (the caller is an `#[ignore]`d real-weight test that was explicitly selected).
pub fn snapshot_dir(env: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(env)
            .unwrap_or_else(|| panic!("set {env} to the pinned immutable snapshot")),
    )
}

/// [`snapshot_dir`] as a [`WeightsSource::Dir`].
pub fn snapshot_source(env: &str) -> WeightsSource {
    WeightsSource::Dir(snapshot_dir(env))
}

/// The crate's three-way real-weight device selector: `SA3_TEST_METAL` selects Metal,
/// `SA3_TEST_CUDA` selects CUDA, otherwise CPU.
///
/// Honouring `SA3_TEST_METAL` alone would silently run a case on `Device::Cpu` inside the CUDA
/// lanes, which set `SA3_TEST_CUDA`. A requested backend that is unavailable is a hard failure,
/// never a fallback.
pub fn test_device() -> Device {
    if std::env::var_os("SA3_TEST_METAL").is_some() {
        Device::new_metal(0).expect("SA3_TEST_METAL requested but Metal is unavailable")
    } else if std::env::var_os("SA3_TEST_CUDA").is_some() {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0).expect("SA3_TEST_CUDA requested but CUDA is unavailable")
        }
        #[cfg(not(feature = "cuda"))]
        {
            panic!("SA3_TEST_CUDA requires --features cuda")
        }
    } else {
        Device::Cpu
    }
}

/// Memory-map committed or pinned safetensors artifacts as an F32 [`VarBuilder`] on `device`,
/// through the audio lane's one audited mmap helper (`candle_audio::weights`, sc-27358).
pub fn mmap_f32<P: AsRef<Path>>(paths: &[P], device: &Device) -> VarBuilder<'static> {
    weights::mmap_var_builder(paths, DType::F32, device).unwrap()
}

/// Memory-map one committed or pinned safetensors artifact through the audio lane's audited helper.
pub fn mmap_artifact(path: impl AsRef<Path>) -> MmapedSafetensors {
    weights::mmap_safetensors(path).unwrap()
}
