//! Compute device + dtype selection.
//!
//! Follows the `candle-gen` convention: the backend is chosen at compile time by feature
//! (CUDA → Metal → CPU). The compute dtype is `bf16` on the GPU backends (matching the `mlx-llm`
//! reference) and `f32` on CPU, where half-precision kernels are slow or unsupported.

use candle_core::{DType, Device};

use crate::error::{Error, Result};

/// Environment switch for the CUDA stream the model runs on (story sc-24134): `own` (the
/// default) or `legacy`. See [`CudaStreamKind`].
pub const CUDA_STREAM_ENV: &str = "CANDLE_LLM_CUDA_STREAM";

/// Which CUDA stream [`select_device`] puts the model on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaStreamKind {
    /// The model's **own** stream (`Device::new_cuda_with_stream`) with cudarc's per-slice event
    /// tracking off — the default since sc-24134, and the only form a CUDA graph can be captured
    /// on.
    Own,
    /// The legacy NULL stream `Device::new_cuda` uses, with cudarc's default event tracking —
    /// the pre-sc-24134 device, kept selectable as the comparison row and the rollback lever.
    /// The CUDA-graph runner refuses it (`legacy_stream`).
    Legacy,
}

impl CudaStreamKind {
    /// Parse the switch's value: unset / empty / `own` → [`Own`](Self::Own), `legacy` →
    /// [`Legacy`](Self::Legacy), anything else a configuration error (case-insensitive).
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.map(|v| v.trim().to_ascii_lowercase()) {
            None => Ok(Self::Own),
            Some(v) if v.is_empty() || v == "own" => Ok(Self::Own),
            Some(v) if v == "legacy" => Ok(Self::Legacy),
            Some(v) => Err(Error::Config(format!(
                "unsupported {CUDA_STREAM_ENV}={v:?}; expected `own` or `legacy`"
            ))),
        }
    }

    /// The switch as the environment sets it.
    pub fn from_env() -> Result<Self> {
        let value = std::env::var(CUDA_STREAM_ENV).ok();
        Self::parse(value.as_deref())
    }

    /// Stable lower-case label for logs and evidence rows.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Own => "own",
            Self::Legacy => "legacy",
        }
    }
}

/// The process-default compute device, selected at compile time by feature:
/// CUDA (`cuda`) → Metal (`metal`) → CPU (default).
pub fn select_device() -> Result<Device> {
    if let Some(selection) = std::env::var_os("CANDLE_LLM_DEVICE") {
        let selection = selection.to_string_lossy();
        if selection.eq_ignore_ascii_case("cpu") {
            return Ok(Device::Cpu);
        }
        if !selection.eq_ignore_ascii_case("auto") {
            return Err(crate::error::Error::Config(format!(
                "unsupported CANDLE_LLM_DEVICE={selection:?}; expected `auto` or `cpu`"
            )));
        }
    }
    // The model gets its **own** CUDA stream (story sc-24134): stream capture — what the
    // CUDA-graph runner in `decode::graph` records a decode step with — is not supported on
    // the legacy NULL stream `Device::new_cuda` uses. cudarc creates two CUDA events per
    // allocation while its per-slice event tracking is on, and makes a stream wait on them as
    // soon as a second stream exists; with every tensor of the model on this one stream there
    // is nothing to order, and a wait on an event recorded before a capture would invalidate
    // the capture, so it is switched off (that is the documented use of the `unsafe`: the
    // caller vouches for single-stream ordering). `CANDLE_LLM_CUDA_STREAM=legacy` restores the
    // pre-sc-24134 device.
    #[cfg(feature = "cuda")]
    let dev = match CudaStreamKind::from_env()? {
        CudaStreamKind::Own => {
            let dev = Device::new_cuda_with_stream(0)?;
            if let Device::Cuda(cuda) = &dev {
                unsafe { cuda.disable_event_tracking() };
            }
            dev
        }
        CudaStreamKind::Legacy => Device::new_cuda(0)?,
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let dev = Device::new_metal(0)?;
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let dev = Device::Cpu;
    Ok(dev)
}

/// The dense compute dtype for a device: `bf16` on the GPU backends (CUDA / Metal — matching the
/// mlx-llm reference engine), `f32` on CPU.
pub fn compute_dtype(device: &Device) -> DType {
    if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_stream_switch_parses_own_legacy_and_refuses_the_rest() {
        assert_eq!(CudaStreamKind::parse(None).unwrap(), CudaStreamKind::Own);
        assert_eq!(
            CudaStreamKind::parse(Some("")).unwrap(),
            CudaStreamKind::Own
        );
        assert_eq!(
            CudaStreamKind::parse(Some(" Own ")).unwrap(),
            CudaStreamKind::Own
        );
        assert_eq!(
            CudaStreamKind::parse(Some("LEGACY")).unwrap(),
            CudaStreamKind::Legacy
        );
        assert!(matches!(
            CudaStreamKind::parse(Some("null")),
            Err(Error::Config(_))
        ));
        assert_eq!(CudaStreamKind::Own.label(), "own");
        assert_eq!(CudaStreamKind::Legacy.label(), "legacy");
    }
}
