//! The SC-20686 campaign observer compiled **out** (sc-27288): the default production build.
//!
//! The real observer (`sc20686_observer.rs`) is measurement tooling for the epic-20669 K/V campaign,
//! compiled only with the non-default `campaign-observer` feature (and in this crate's own unit-test
//! build, so its tests keep running). This module keeps the hook surface the edit route and the
//! `flux2-edit` example call, as inert no-ops, so call sites need no `cfg`. Arming a campaign from a
//! build without the feature is a hard error rather than a silent no-op.

use std::io;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::runtime::CancelFlag;

pub const PRODUCT_ROUTE_ID: &str = "flux2_klein_9b_edit";

fn feature_disabled() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        "SC-20686 campaign observer is compiled out; rebuild candle-gen-flux2 with \
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
    _width: u32,
    _height: u32,
    _prompt: &str,
    _guidance: f32,
    _reference_count: u32,
) -> io::Result<Option<Scope>> {
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn bind_edit_geometry(
    _layers: u32,
    _batch: u32,
    _heads: u32,
    _head_dimension: u32,
    _target_sq: u64,
    _reference_skv: u64,
    _latent_height: u32,
    _latent_width: u32,
    _dtype: impl Into<String>,
) {
}

pub(crate) enum FluxReferenceKv {}

pub(crate) enum FluxKvRead {}

#[inline(always)]
pub(crate) fn record_flux_kv_created(
    _image_k: &Tensor,
    _image_v: &Tensor,
) -> Option<FluxReferenceKv> {
    None
}

#[inline(always)]
pub(crate) fn begin_flux_kv_read(_reference: Option<FluxReferenceKv>) -> Option<FluxKvRead> {
    None
}

#[inline(always)]
pub(crate) fn record_flux_kv_read(_measurement: Option<FluxKvRead>) {}

#[inline(always)]
pub fn observe(_phase: &'static str, _persistent_bytes: u64, _transient_bytes: u64, _reused: u64) {}

#[inline(always)]
pub fn observe_cancelled() {}
