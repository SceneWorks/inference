//! What this Candle build can serve on this host, before any model is loaded (story sc-24139).
//!
//! A product reads [`backend_capabilities`] (through its runtime bundle's
//! `text_backend_capabilities()`) to offer backend-specific controls — the NVFP4 weight format,
//! the CUDA-graph switch — or to disable them with the reason. The NVFP4 answer is settled by the
//! **same** gate a load runs ([`Nvfp4Context::require`]), so a disabled control carries exactly the
//! refusal the load would have returned.

use std::sync::OnceLock;

use candle_core::{Device, DeviceLocation};
use candle_quant_kernels::Nvfp4Context;
use core_llm::{BackendCapabilities, FeatureSupport};

use crate::decode::graph::{REASON_CUDA_FEATURE_OFF, REASON_NOT_CUDA};
use crate::device::select_device;

/// This build's execution backend label (`candle-cuda`, `candle-metal`, `candle-cpu`).
pub fn backend_label() -> &'static str {
    if cfg!(feature = "cuda") {
        "candle-cuda"
    } else if cfg!(feature = "metal") {
        "candle-metal"
    } else {
        "candle-cpu"
    }
}

/// What this build can serve on the device a load would use: the device, its CUDA compute
/// capability, whether an NVFP4 load passes the device gate, and whether the CUDA-graph switch
/// (`LoadSpec::cuda_graphs`) is honoured — each unavailable feature with its reason.
///
/// Probed once per process (the device a load selects does not change while it runs) and cached;
/// the probe opens the load device the way a load does and, on CUDA, builds the NVFP4 context
/// (a cuBLASLt handle and the fused quantizer's compile-once module, which a later NVFP4 load
/// reuses). It never reads weights.
pub fn backend_capabilities() -> BackendCapabilities {
    static CAPABILITIES: OnceLock<BackendCapabilities> = OnceLock::new();
    CAPABILITIES
        .get_or_init(|| match select_device() {
            Ok(device) => capabilities_for_device(backend_label(), &device),
            Err(error) => no_device(backend_label(), &error.to_string()),
        })
        .clone()
}

/// The capabilities of `device` for `backend` (the probe behind [`backend_capabilities`], split
/// out so the non-CUDA answers are testable on any host).
pub fn capabilities_for_device(backend: &str, device: &Device) -> BackendCapabilities {
    let nvfp4 = match Nvfp4Context::require(device) {
        Ok(_) => FeatureSupport::available(),
        Err(refusal) => FeatureSupport::unavailable(refusal.to_string()),
    };
    let (label, compute_capability) = match device.location() {
        DeviceLocation::Cpu => ("cpu".to_string(), None),
        DeviceLocation::Metal { gpu_id } => (format!("metal:{gpu_id}"), None),
        DeviceLocation::Cuda { gpu_id } => (format!("cuda:{gpu_id}"), cuda_compute_cap(device)),
    };
    let cuda_graphs = if device.is_cuda() {
        FeatureSupport::available()
    } else {
        // The same reasons the graph runner names when it refuses a step.
        let reason = if cfg!(feature = "cuda") {
            REASON_NOT_CUDA
        } else {
            REASON_CUDA_FEATURE_OFF
        };
        FeatureSupport::unavailable(format!(
            "cuda_graphs: {reason}: CUDA graphs need a CUDA load device; this runtime loads on \
             {label}"
        ))
    };
    BackendCapabilities {
        backend: backend.to_string(),
        device: label,
        compute_capability,
        nvfp4,
        cuda_graphs,
    }
}

/// The answer when no load device could be opened: every device feature is unavailable with the
/// error a load would hit first.
fn no_device(backend: &str, error: &str) -> BackendCapabilities {
    BackendCapabilities {
        backend: backend.to_string(),
        device: format!("unavailable ({error})"),
        compute_capability: None,
        nvfp4: FeatureSupport::unavailable(format!("nvfp4: no load device: {error}")),
        cuda_graphs: FeatureSupport::unavailable(format!("cuda_graphs: no load device: {error}")),
    }
}

#[cfg(feature = "cuda")]
fn cuda_compute_cap(device: &Device) -> Option<(u32, u32)> {
    let Device::Cuda(cuda) = device else {
        return None;
    };
    let (major, minor) = candle_quant_kernels::device_compute_cap(cuda).ok()?;
    Some((u32::try_from(major).ok()?, u32::try_from(minor).ok()?))
}

#[cfg(not(feature = "cuda"))]
fn cuda_compute_cap(_device: &Device) -> Option<(u32, u32)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cpu_device_refuses_nvfp4_and_graphs_with_the_load_gates_reasons() {
        let caps = capabilities_for_device(backend_label(), &Device::Cpu);
        assert_eq!(caps.backend, backend_label());
        assert_eq!(caps.device, "cpu");
        assert_eq!(caps.compute_capability, None);
        assert!(!caps.nvfp4.supported);
        let nvfp4 = caps.nvfp4.reason.as_deref().unwrap();
        // The load gate's own refusal text: it names the capability and the sm_120 floor.
        assert!(nvfp4.starts_with("nvfp4: "), "{nvfp4}");
        assert!(nvfp4.contains("sm_120"), "{nvfp4}");
        assert!(!caps.cuda_graphs.supported);
        let graphs = caps.cuda_graphs.reason.as_deref().unwrap();
        assert!(graphs.starts_with("cuda_graphs: "), "{graphs}");
        let expected = if cfg!(feature = "cuda") {
            REASON_NOT_CUDA
        } else {
            REASON_CUDA_FEATURE_OFF
        };
        assert!(graphs.contains(expected), "{graphs}");
    }

    #[test]
    fn a_missing_device_names_the_error_in_every_refusal() {
        let caps = no_device("candle-cuda", "CUDA driver not found");
        assert!(caps.device.contains("CUDA driver not found"));
        assert!(!caps.nvfp4.supported && !caps.cuda_graphs.supported);
        assert!(caps
            .nvfp4
            .reason
            .as_deref()
            .unwrap()
            .contains("CUDA driver not found"));
    }

    #[test]
    fn the_backend_label_follows_the_build() {
        let label = backend_label();
        if cfg!(feature = "cuda") {
            assert_eq!(label, "candle-cuda");
        } else if cfg!(feature = "metal") {
            assert_eq!(label, "candle-metal");
        } else {
            assert_eq!(label, "candle-cpu");
        }
    }

    /// On an sm_120 device the probe offers NVFP4 and graphs and reports the capability; below it
    /// NVFP4 carries the gate's refusal. Runs only where a CUDA device exists.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_cuda_device_reports_its_compute_capability_and_the_nvfp4_gate() {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let caps = capabilities_for_device("candle-cuda", &device);
        assert_eq!(caps.device, "cuda:0");
        assert!(caps.cuda_graphs.supported, "{:?}", caps.cuda_graphs);
        let cap = caps
            .compute_capability
            .expect("a CUDA device reports its capability");
        if cap >= (12, 0) {
            assert!(caps.nvfp4.supported, "{:?}", caps.nvfp4);
        } else {
            assert!(!caps.nvfp4.supported);
            assert!(caps.nvfp4.reason.as_deref().unwrap().contains("sm_120"));
        }
    }
}
