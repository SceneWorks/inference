//! NVFP4 projection forward dispatch and telemetry (epic sc-24128, story sc-24136).
//!
//! A [`Projection::Nvfp4`](super::projection::Projection::Nvfp4) weight has two forward
//! implementations in `candle-quant-kernels`, both over the same resident packed weight:
//!
//! - the **fused decode GEMV** ([`Nvfp4Weight::forward_gemv`]): one launch, bf16 activation of
//!   1..=[`NVFP4_GEMV_MAX_ROWS`] rows **not quantized**, f32 accumulate (W4A16 math);
//! - the **cuBLASLt W4A4 GEMM** ([`Nvfp4Weight::forward`]): activation quantized on-device to
//!   NVFP4 (one host sync for its per-tensor scale), block-scaled FP4 tensor-core GEMM.
//!
//! [`forward`] picks the GEMV when the switch is on and the kernel accepts the input (≤ 8 rows,
//! bf16, matching `K`, compiled on this device); everything else — prefill, an f32 activation, a
//! compile failure, the switch off — runs cuBLASLt. **Which path ran is never silent** (epic E2):
//! every call records itself in a per-thread [`Nvfp4PathTally`], and
//! [`DecodeRecord`](crate::decode::DecodeRecord) carries the per-request delta
//! (`nvfp4_projections`) — how many NVFP4 projection calls ran the GEMV, how many ran cuBLASLt and
//! the last cuBLASLt reason.
//!
//! **Switch.** On by default in a `cuda` build (the measured-faster path, sc-24136 evidence);
//! `CANDLE_LLM_NVFP4_GEMV=0` (also `off`, `false`, `no`, `cublaslt`) turns it off for the process,
//! and [`set_nvfp4_gemv`] overrides the environment at runtime (the decode bench and the parity
//! tests flip it to compare the two paths in one process).
//!
//! **Reasons.** A cuBLASLt run's reason is a stable label: [`REASON_DISABLED`], the GEMV's typed
//! refusal (`rows`, `dtype`, `shape`, `device`, `not_cuda` from `Nvfp4GemvRefusal::label`) or the
//! seam's cached compile error (`nvrtc`, `compute_floor`, …).

use std::cell::Cell;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

use candle_core::Tensor;
#[cfg(feature = "cuda")]
use candle_quant_kernels::Nvfp4GemvError;
use candle_quant_kernels::Nvfp4Weight;
pub use candle_quant_kernels::NVFP4_GEMV_MAX_ROWS;

use crate::error::Result;

/// Environment switch: `0` / `off` / `false` / `no` / `cublaslt` disable the fused GEMV.
pub const NVFP4_GEMV_ENV: &str = "CANDLE_LLM_NVFP4_GEMV";

/// cuBLASLt-run reason: the switch is off.
pub const REASON_DISABLED: &str = "disabled";

const POLICY_ENV: u8 = 0;
const POLICY_ON: u8 = 1;
const POLICY_OFF: u8 = 2;

static POLICY: AtomicU8 = AtomicU8::new(POLICY_ENV);

fn env_says_enabled() -> bool {
    static FROM_ENV: OnceLock<bool> = OnceLock::new();
    *FROM_ENV.get_or_init(|| {
        std::env::var(NVFP4_GEMV_ENV)
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !matches!(v.as_str(), "0" | "off" | "false" | "no" | "cublaslt")
            })
            .unwrap_or(true)
    })
}

/// Whether the fused GEMV may be tried for decode-sized NVFP4 projections.
pub fn nvfp4_gemv_enabled() -> bool {
    match POLICY.load(Ordering::Relaxed) {
        POLICY_ON => true,
        POLICY_OFF => false,
        _ => env_says_enabled(),
    }
}

/// Override the switch for the process: `Some(true)` / `Some(false)` force it, `None` returns to
/// the environment's setting.
pub fn set_nvfp4_gemv(enabled: Option<bool>) {
    let policy = match enabled {
        Some(true) => POLICY_ON,
        Some(false) => POLICY_OFF,
        None => POLICY_ENV,
    };
    POLICY.store(policy, Ordering::Relaxed);
}

/// Per-thread counts of NVFP4 projection calls by path (monotone; take deltas with
/// [`Nvfp4PathTally::since`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Nvfp4PathTally {
    /// Calls served by the fused decode GEMV.
    pub gemv: u64,
    /// Calls served by the cuBLASLt W4A4 GEMM.
    pub cublaslt: u64,
    /// Why the most recent cuBLASLt run happened (`None` if none happened).
    pub cublaslt_reason: Option<&'static str>,
}

impl Nvfp4PathTally {
    /// The counts accumulated since `start` (the reason is the latest one).
    pub fn since(&self, start: &Nvfp4PathTally) -> Nvfp4PathTally {
        Nvfp4PathTally {
            gemv: self.gemv.wrapping_sub(start.gemv),
            cublaslt: self.cublaslt.wrapping_sub(start.cublaslt),
            cublaslt_reason: if self.cublaslt != start.cublaslt {
                self.cublaslt_reason
            } else {
                None
            },
        }
    }

    /// Which path served the calls: `gemv`, `cublaslt`, `mixed` (both — e.g. a cuBLASLt prefill
    /// then GEMV decode steps), or `none` (no NVFP4 projection ran).
    pub fn label(&self) -> &'static str {
        match (self.gemv, self.cublaslt) {
            (0, 0) => "none",
            (_, 0) => "gemv",
            (0, _) => "cublaslt",
            _ => "mixed",
        }
    }
}

thread_local! {
    static TALLY: Cell<Nvfp4PathTally> = const {
        Cell::new(Nvfp4PathTally { gemv: 0, cublaslt: 0, cublaslt_reason: None })
    };
}

/// This thread's monotone tally.
pub fn nvfp4_path_tally() -> Nvfp4PathTally {
    TALLY.with(Cell::get)
}

/// Record one NVFP4 projection call served by the fused GEMV.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn note_gemv() {
    TALLY.with(|c| {
        let mut t = c.get();
        t.gemv = t.gemv.wrapping_add(1);
        c.set(t);
    });
}

/// Record one NVFP4 projection call served by cuBLASLt, with why.
pub(crate) fn note_cublaslt(reason: &'static str) {
    TALLY.with(|c| {
        let mut t = c.get();
        t.cublaslt = t.cublaslt.wrapping_add(1);
        t.cublaslt_reason = Some(reason);
        c.set(t);
    });
}

/// `x · Wᵀ (+ b)` for an NVFP4 projection: the fused GEMV when the switch is on and the kernel
/// accepts `x`, the cuBLASLt W4A4 GEMM otherwise — recorded either way. A refusal or a cached
/// compile failure falls back to cuBLASLt (and is recorded with its label); a genuine candle /
/// driver error inside the GEMV propagates.
pub fn forward(weight: &Nvfp4Weight, x: &Tensor) -> Result<Tensor> {
    if !nvfp4_gemv_enabled() {
        note_cublaslt(REASON_DISABLED);
        return Ok(weight.forward(x)?);
    }
    #[cfg(feature = "cuda")]
    {
        match weight.forward_gemv(x) {
            Ok(y) => {
                note_gemv();
                return Ok(y);
            }
            Err(Nvfp4GemvError::Candle(e)) => return Err(e.into()),
            Err(e @ (Nvfp4GemvError::Refused(_) | Nvfp4GemvError::Compile(_))) => {
                note_cublaslt(e.label());
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        // Unreachable in practice (an `Nvfp4Weight` only exists in a `cuda` build); recorded
        // honestly all the same.
        note_cublaslt("cuda_feature_off");
    }
    Ok(weight.forward(x)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tally_deltas_and_labels() {
        let start = nvfp4_path_tally();
        assert_eq!(nvfp4_path_tally().since(&start).label(), "none");
        note_gemv();
        assert_eq!(nvfp4_path_tally().since(&start).label(), "gemv");
        note_cublaslt("rows");
        let d = nvfp4_path_tally().since(&start);
        assert_eq!((d.gemv, d.cublaslt), (1, 1));
        assert_eq!(d.cublaslt_reason, Some("rows"));
        assert_eq!(d.label(), "mixed");
        let later = nvfp4_path_tally();
        note_cublaslt("dtype");
        let d = nvfp4_path_tally().since(&later);
        assert_eq!(
            d,
            Nvfp4PathTally {
                gemv: 0,
                cublaslt: 1,
                cublaslt_reason: Some("dtype")
            }
        );
        assert_eq!(d.label(), "cublaslt");
        let now = nvfp4_path_tally();
        note_gemv();
        assert_eq!(nvfp4_path_tally().since(&now).cublaslt_reason, None);
    }

    #[test]
    fn tally_is_per_thread() {
        let before = nvfp4_path_tally();
        note_gemv();
        let other = std::thread::spawn(|| {
            let start = nvfp4_path_tally();
            note_cublaslt("x");
            nvfp4_path_tally().since(&start)
        })
        .join()
        .unwrap();
        assert_eq!(other.cublaslt, 1);
        assert_eq!(nvfp4_path_tally().since(&before).cublaslt, 0);
    }

    #[test]
    fn runtime_override_beats_the_environment_and_can_be_cleared() {
        let from_env = nvfp4_gemv_enabled();
        set_nvfp4_gemv(Some(false));
        assert!(!nvfp4_gemv_enabled());
        set_nvfp4_gemv(Some(true));
        assert!(nvfp4_gemv_enabled());
        set_nvfp4_gemv(None);
        assert_eq!(nvfp4_gemv_enabled(), from_env);
    }
}
