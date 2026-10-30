//! The auto-duration range an LTX [`GenerationRequest`](crate::GenerationRequest) carries. The
//! duration-head frame resolution lives in `gen_core_ltx::duration_head` (sc-27316).

use crate::{Error, Result};

/// `[min_seconds, max_seconds]` an auto-duration request clamps its prediction to (upstream
/// `ltx_pipelines.utils.types.AutoDuration`). [`Default`] mirrors upstream's `AutoDuration()`:
/// 1s–20s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoDurationRange {
    pub min_seconds: f32,
    pub max_seconds: f32,
}
impl Default for AutoDurationRange {
    fn default() -> Self {
        Self {
            min_seconds: 1.0,
            max_seconds: 20.0,
        }
    }
}
impl AutoDurationRange {
    /// Validate a caller-supplied range. Upstream's `AutoDuration` is a bare dataclass with no
    /// validation; we add it here because a request-supplied range is untrusted input reaching a
    /// snap/clamp routine, and a non-finite or inverted range should fail loudly at the request
    /// boundary rather than produce a nonsensical frame count.
    pub fn new(min_seconds: f32, max_seconds: f32) -> Result<Self> {
        if !(min_seconds.is_finite() && max_seconds.is_finite()) {
            return Err(Error::Msg(format!(
                "ltx duration_head: auto-duration range must be finite, got [{min_seconds}, {max_seconds}]"
            )));
        }
        if min_seconds <= 0.0 || max_seconds <= 0.0 {
            return Err(Error::Msg(format!(
                "ltx duration_head: auto-duration range must be positive, got [{min_seconds}, {max_seconds}]"
            )));
        }
        if min_seconds > max_seconds {
            return Err(Error::Msg(format!(
                "ltx duration_head: auto-duration min_seconds {min_seconds} > max_seconds {max_seconds}"
            )));
        }
        Ok(Self {
            min_seconds,
            max_seconds,
        })
    }
}
