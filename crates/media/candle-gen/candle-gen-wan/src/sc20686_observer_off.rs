//! The SC-20686 campaign observer compiled **out** (sc-27296): the default production build.
//!
//! The real observer (`sc20686_observer.rs`) is measurement tooling for the epic-20669 K/V campaign,
//! compiled only with the non-default `campaign-observer` feature (and in this crate's own unit-test
//! build, so its tests keep running). This module keeps the hook surface the Wan routes and the
//! campaign-capable examples call, as inert no-ops: no campaign is ever active, so every guarded
//! hook body is skipped. Arming a campaign from a build without the feature is a hard error rather
//! than a silent no-op.

use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_gen::gen_core::runtime::CancelFlag;

fn feature_disabled() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        "SC-20686 campaign observer is compiled out; rebuild candle-gen-wan with \
         `--features campaign-observer`",
    )
}

/// Never constructed: [`request_output`] refuses before one could exist.
pub struct CampaignOutputRequest(());

impl CampaignOutputRequest {
    pub fn arm(self) -> Self {
        self
    }

    pub fn arm_cancellation(self) -> Self {
        self
    }
}

/// Always `false`: no campaign can be armed in this build.
pub fn campaign_cancelled() -> bool {
    false
}

/// Always refuses: a campaign cannot be armed without the `campaign-observer` feature.
pub fn request_output(
    _path: impl Into<PathBuf>,
    _source_ref: impl Into<String>,
    _residency_strategy: impl Into<String>,
) -> io::Result<CampaignOutputRequest> {
    Err(feature_disabled())
}

pub struct Scope;

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn activate_requested(
    _root: &Path,
    _cancel: &CancelFlag,
    _variant: &str,
    _batch: u32,
    _frames: u32,
    _width: u32,
    _height: u32,
    _latent_frames: u32,
    _latent_height: u32,
    _latent_width: u32,
    _prompt: &str,
    _guidance: Option<f32>,
    _reference_count: u32,
) -> io::Result<Option<Scope>> {
    Ok(None)
}

#[inline(always)]
pub(crate) fn campaign_active() -> bool {
    false
}

/// Only reached under an active campaign, which this build never has.
#[inline(always)]
pub fn checked_packed_group_affine_kv_bytes(
    _batch: u64,
    _heads: u64,
    _tokens: u64,
    _width: u64,
    _group_size: u64,
) -> Option<u64> {
    None
}

pub(crate) enum ActiveAllocatorWindow {}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ActiveAllocatorMeasurement;

#[inline(always)]
pub(crate) fn begin_active_allocator_window() -> Option<ActiveAllocatorWindow> {
    None
}

#[inline(always)]
pub(crate) fn finish_active_allocator_window(
    _window: Option<ActiveAllocatorWindow>,
) -> ActiveAllocatorMeasurement {
    ActiveAllocatorMeasurement
}

#[inline(always)]
pub(crate) fn register_cache(_dense_bytes: u64, _candidate_bytes: u64, _measured: Instant) -> u64 {
    0
}

#[inline(always)]
pub(crate) fn record_cache_read(
    _cache_id: u64,
    _allocator: ActiveAllocatorMeasurement,
    _measured: Instant,
) {
}

#[inline(always)]
pub(crate) fn release_cache(_cache_id: u64) {}

#[inline(always)]
pub fn observe(_phase: &'static str, _persistent_bytes: u64, _transient_bytes: u64, _reused: u64) {}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn observe_tensor(
    _phase: &'static str,
    _operation: &'static str,
    _persistent_bytes: u64,
    _transient_bytes: u64,
    _reused: u64,
    _tensor_shape: impl Into<String>,
    _dtype: impl Into<String>,
    _mask: impl Into<String>,
    _rope: impl Into<String>,
) {
}

/// Never builds the evidence: no campaign is active.
#[inline(always)]
pub(crate) fn campaign_evidence<T>(_build: impl FnOnce() -> T) -> Option<T> {
    None
}

#[inline(always)]
pub fn observe_cancelled() {}

#[inline(always)]
pub(crate) fn bind_cross_kv_geometry(
    _layers: u32,
    _heads: u32,
    _head_dimension: u32,
    _sq: u64,
    _skv: u64,
    _dtype: impl Into<String>,
) {
}

#[inline(always)]
pub(crate) fn confirm_real_weight_projection() {}
