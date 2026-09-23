//! Compute device + dtype selection.
//!
//! Follows the `candle-gen` convention: the backend is chosen at compile time by feature
//! (CUDA → Metal → CPU). The compute dtype is `bf16` on the GPU backends (matching the `mlx-llm`
//! reference) and `f32` on CPU, where half-precision kernels are slow or unsupported.

use candle_core::{DType, Device};

use crate::error::Result;

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
    // the legacy NULL stream `Device::new_cuda` uses. cudarc turns on per-slice event
    // tracking as soon as a second stream exists; with every tensor of the process on this one
    // stream there is nothing to order, and its waits on events recorded before a capture
    // would invalidate the capture, so it is switched off (that is the documented use of the
    // `unsafe`: the caller vouches for single-stream ordering).
    #[cfg(feature = "cuda")]
    let dev = {
        let dev = Device::new_cuda_with_stream(0)?;
        if let Device::Cuda(cuda) = &dev {
            unsafe { cuda.disable_event_tracking() };
        }
        dev
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
