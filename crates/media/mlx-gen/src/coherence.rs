//! GPU-view coherence of freshly loaded weight buffers (sc-22414).
//!
//! # The defect this closes
//!
//! A safetensors `Load` runs on the CPU stream: MLX allocates a Metal shared buffer and `pread`s
//! the file into it. Under a full page cache with dirty write-back pressure, the **GPU's view of
//! that buffer can lag the CPU's**: the first Metal kernel to read it sees exact zeros for pages
//! the CPU already holds correctly, and a later kernel sees the real bytes. Measured on Mac2
//! (Mac17,6, 128 GB, macOS 26.6.1; 6/6 reproductions, see SceneWorks
//! `docs/calibration/sc-18791/mac2-cold-load/RUNBOOK.md`): LTX-2.5's once-loaded 2 GB Gemma 4
//! `embed_tokens` table read as zeros on render 1's first pass while every CPU-side hash of the
//! same buffer was correct, every layer weight was correct at both views, and the same handle read
//! correctly minutes later. An all-zero embedding maps through RMSNorm, SDPA and the residual adds
//! to an all-zero stack, so the render is deterministic garbage rather than a crash — the
//! measured-vs-warm parity breach that blocked the SC-18791 campaign.
//!
//! Ruled out there by experiment: the OS version, the sampler, a CPU→GPU fence race (a
//! materialization barrier before the forward changed nothing), read errors and short reads in
//! MLX's reader, uncached reads alone, read churn alone, and user-space staging of the read (the
//! CPU view was never wrong). Time heals it: every variant that slowed the load down passed.
//!
//! # The fix
//!
//! Verify the GPU's view against the CPU's at the load boundary, before any graph consumes the
//! buffer, and wait for them to agree. [`byte_checksum`] is a **wrapping integer sum of the raw
//! bytes** read through a given stream: modular addition is order-independent, so a CPU reduction
//! and a GPU reduction over the same bytes are bit-identical regardless of how each backend
//! partitions the sum (a float sum would not be). [`verify_gpu_view`] evaluates each array on its
//! own stream, takes the CPU checksum as the reference, and re-reads the GPU checksum with a
//! bounded back-off until it matches. Exhausting the budget is a typed
//! [`Error::IncoherentLoad`], never a silent retry-forever and never a silent render.
//!
//! The check is a GPU read of every loaded byte plus a CPU read of the same — cheap against the
//! load itself, and the GPU read is the one that would otherwise have happened inside the first
//! forward.
//!
//! # Shared with mlx-llm
//!
//! The implementation lives in [`mlx_llm::primitives::coherence`] (this crate depends on
//! mlx-llm); this module re-exports it under the names providers already call and adapts the two
//! crate-specific edges: errors come back as this crate's [`Error`] (the typed
//! [`Error::IncoherentLoad`] field for field), and [`verify_gpu_view`] reports its per-array
//! progress to the physical-iOS load-stage diagnostics. The incidence counter ([`retries`]) is
//! therefore one process-wide count across both crates.
//!
//! [`Weights::materialize`](crate::weights::Weights::materialize) and
//! [`Weights::materialize_accessed`](crate::weights::Weights::materialize_accessed) are the two
//! seams, and both call [`verify_gpu_view`] — so a loader is guarded exactly when it materializes
//! through one of them. The LTX-2.5 render path does at every load (text encoder, connector, DiT
//! resident or streamed, VAEs, vocoder, upsamplers, duration head, adapters, enhancer), as do the
//! block-window streams (krea, z-image, qwen-image, flux2, sdxl, boogu). A provider that never
//! materializes — evaluating its checkpoint inside its first forward — is **not** guarded by this
//! module; guarding it means adding the `materialize` call at its load seam, which also changes
//! when its weights become resident, so that is a per-provider decision rather than a blanket one.

use mlx_llm::primitives::coherence as shared;
use mlx_rs::{Array, StreamOrDevice};

use crate::error::{Error, Result};

pub use shared::{backoff, backoff_millis, retries, MAX_RETRIES};

/// Map an mlx-llm coherence error onto this crate's error, keeping the typed variants.
fn from_shared(e: mlx_llm::Error) -> Error {
    match e {
        mlx_llm::Error::Mlx(ex) => Error::Mlx(ex),
        mlx_llm::Error::IncoherentLoad {
            name,
            bytes,
            cpu,
            gpu,
            attempts,
        } => Error::IncoherentLoad {
            name,
            bytes,
            cpu,
            gpu,
            attempts,
        },
        other => Error::Msg(other.to_string()),
    }
}

/// Wrapping `u32` sum of the raw bytes of `a`, read through `stream`; see
/// [`mlx_llm::primitives::coherence::byte_checksum`].
pub fn byte_checksum(a: &Array, stream: StreamOrDevice) -> Result<u64> {
    shared::byte_checksum(a, stream).map_err(from_shared)
}

/// Verify that the GPU reads every one of `arrays` as the bytes the CPU holds, waiting for the
/// views to converge; see [`mlx_llm::primitives::coherence::verify_gpu_view`]. Each array's
/// progress is reported to the physical-iOS load-stage diagnostics (a no-op elsewhere).
pub fn verify_gpu_view<'a>(arrays: impl IntoIterator<Item = (&'a str, &'a Array)>) -> Result<()> {
    shared::verify_gpu_view_observed(arrays, |stage, a| {
        crate::ios_quantum::diagnostic_load_stage(stage, || a.nbytes())
    })
    .map_err(from_shared)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shared implementation's typed refusal crosses into this crate's typed variant field for
    /// field (not a stringified `Msg`), and its MLX failures stay `Error::Mlx`. The retry/back-off
    /// behaviour itself is tested once, beside the implementation in mlx-llm.
    #[test]
    fn shared_incoherent_load_maps_to_the_typed_variant() {
        let mapped = from_shared(mlx_llm::Error::IncoherentLoad {
            name: "model.embed_tokens.weight".into(),
            bytes: 4096,
            cpu: 0xdead_beef,
            gpu: 0,
            attempts: MAX_RETRIES + 1,
        });
        match mapped {
            Error::IncoherentLoad {
                name,
                bytes,
                cpu,
                gpu,
                attempts,
            } => {
                assert_eq!(name, "model.embed_tokens.weight");
                assert_eq!(bytes, 4096);
                assert_eq!(cpu, 0xdead_beef);
                assert_eq!(gpu, 0);
                assert_eq!(attempts, MAX_RETRIES + 1);
            }
            other => panic!("expected IncoherentLoad, got {other:?}"),
        }
        assert!(matches!(
            from_shared(mlx_llm::Error::Unsupported("x".into())),
            Error::Msg(m) if m.contains("unsupported: x")
        ));
    }

    /// A zero-element tensor checksums to 0 through the re-export and passes verification.
    #[test]
    fn a_zero_element_tensor_checksums_to_zero_on_both_streams() {
        let a = Array::zeros::<f32>(&[0]).unwrap();
        assert_eq!(byte_checksum(&a, StreamOrDevice::gpu()).unwrap(), 0);
        assert_eq!(byte_checksum(&a, StreamOrDevice::cpu()).unwrap(), 0);
        verify_gpu_view([("t", &a)]).unwrap();
    }
}
