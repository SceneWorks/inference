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
    #[cfg(feature = "cuda")]
    let dev = Device::new_cuda(0)?;
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
