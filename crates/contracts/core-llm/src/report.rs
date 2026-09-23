//! Backend-neutral reports a product reads to see which decode path ran, what a load produced,
//! and what the backend can serve on this host (epic sc-24128 E2, story sc-24139).
//!
//! A backend fills these from its own measured telemetry and hands them out through the contract
//! a product already holds: [`TextLlmOutput::decode`](crate::TextLlmOutput::decode) per
//! generation, [`TextLlm::load_report`](crate::TextLlm::load_report) per loaded provider, and a
//! runtime bundle's `text_backend_capabilities()` for the host. Every label is the backend's own
//! stable lower-case evidence label, so a product renders them as-is and a fallback is always
//! named — never a silent downgrade.

use crate::request::Quantize;
use crate::speculative::ProposerKind;

/// Whether one optional backend feature is available on this host, and why not when it is not.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FeatureSupport {
    /// The feature can be requested and the backend will act on it.
    pub supported: bool,
    /// Why the feature is unavailable, in the backend's words (`None` when supported). This is the
    /// same refusal the backend would return if a load or request asked for the feature anyway.
    pub reason: Option<String>,
}

impl FeatureSupport {
    /// An available feature.
    pub fn available() -> Self {
        Self {
            supported: true,
            reason: None,
        }
    }

    /// An unavailable feature, with the reason a product shows beside the disabled control.
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            supported: false,
            reason: Some(reason.into()),
        }
    }
}

/// What a runtime's text backend can serve on this host, independent of any loaded model — the
/// source a product uses to offer, or disable with a reason, backend-specific load and request
/// controls (sc-24139).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BackendCapabilities {
    /// The execution backend (`candle-cuda`, `candle-cpu`, `candle-metal`, `mlx`).
    pub backend: String,
    /// The device a load lands on (`cuda:0`, `cpu`, `metal`), or why no device could be opened.
    pub device: String,
    /// The CUDA compute capability `(major, minor)` of the load device; `None` off CUDA.
    pub compute_capability: Option<(u32, u32)>,
    /// Whether a load with [`Quantize::Nvfp4`] can pass the device gate here. Settled by the same
    /// check the load runs, so the reason is the refusal the load would return. (A load can still
    /// refuse NVFP4 for the *model* — a GGUF source, an unsupported architecture — with its own
    /// typed [`Unsupported`](crate::Error::Unsupported).)
    pub nvfp4: FeatureSupport,
    /// Whether [`LoadSpec::cuda_graphs`](crate::LoadSpec::cuda_graphs) is honoured. A supported
    /// switch still reports per generation whether a graph actually replayed
    /// ([`DecodeReport::cuda_graphs`]), and why not when it did not.
    pub cuda_graphs: FeatureSupport,
}

impl BackendCapabilities {
    /// A backend with no CUDA device features (MLX, Candle CPU/Metal): NVFP4 and CUDA graphs are
    /// unavailable, each with a reason naming `backend`.
    pub fn without_cuda(backend: impl Into<String>, device: impl Into<String>) -> Self {
        let backend = backend.into();
        Self {
            nvfp4: FeatureSupport::unavailable(format!(
                "nvfp4: NVFP4 weights need the Candle CUDA backend on a compute capability >= \
                 sm_120 GPU; this runtime is {backend}"
            )),
            cuda_graphs: FeatureSupport::unavailable(format!(
                "cuda_graphs: CUDA graphs need the Candle CUDA backend; this runtime is {backend}"
            )),
            backend,
            device: device.into(),
            compute_capability: None,
        }
    }
}

/// One path a request's work could take, and why the slower one ran when it did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PathReport {
    /// The backend's label for what served the request (for example `gemv` / `cublaslt` /
    /// `mixed` / `none` for NVFP4 projections, `fused` / `reference` / `mixed` / `none` for fused
    /// primitives).
    pub path: String,
    /// Why the fallback path ran, when it did (`None` when it never ran).
    pub reason: Option<String>,
}

/// The CUDA-graph runner's part in one request.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CudaGraphsReport {
    /// Whether the graph switch was on for this generation — the loaded model's
    /// [`LoadSpec::cuda_graphs`](crate::LoadSpec::cuda_graphs), else the backend's default at load.
    pub enabled: bool,
    /// `graph` (only replays), `eager` (no replay), `mixed`, or `none` (no step went through the
    /// runner — the switch was off, or this decode path does not use it).
    pub path: String,
    /// Steps served by a graph replay.
    pub replayed: u64,
    /// Steps served eagerly (fallbacks, warm-ups, prefills, self-checks).
    pub eager: u64,
    /// Graphs captured and verified.
    pub captured: u64,
    /// Why an eager step ran when a graph was wanted (for example `disabled`,
    /// `host_upload_in_capture`, `deltanet_state_unstable`).
    pub fallback_reason: Option<String>,
}

/// Which decode path served one generation, as the backend measured it (epic sc-24128 E2: the path
/// that ran is visible and a fallback is named). Carried on
/// [`TextLlmOutput::decode`](crate::TextLlmOutput::decode).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecodeReport {
    /// The decode implementation (`reference`, `step_model`, `mtp`, `prompt_lookup`,
    /// `draft_model`).
    pub path: String,
    /// Which proposer ran. `none` includes an [`MtpMode::Auto`](crate::MtpMode::Auto) request
    /// that resolved to no proposer on a model without an MTP head.
    pub proposer: ProposerKind,
    /// Draft tokens per verification pass when a proposer ran.
    pub draft_tokens: Option<u32>,
    /// The sampler path: `device`, `host:<reason>` (for example `host:penalty`), or `none`.
    pub sampler: String,
    /// The KV cache implementation (`growing`, `static`).
    pub kv_cache: String,
    /// How attention was computed (`gqa`, `expanded`).
    pub attention: String,
    /// The CUDA-graph runner's part.
    pub cuda_graphs: CudaGraphsReport,
    /// NVFP4 projection calls by path (`none` for a model without NVFP4 projections).
    pub nvfp4_projections: PathReport,
    /// Fused-versus-reference primitive runs.
    pub fused_primitives: PathReport,
    /// Target-model forward passes, including the prompt prefill.
    pub target_forwards: u64,
    /// Draft tokens proposed.
    pub proposed_tokens: u64,
    /// Draft tokens accepted by target verification.
    pub accepted_tokens: u64,
    /// Verify steps recovered by a step-start rollback plus a replay forward.
    pub replay_forwards: u64,
}

/// The resident count and bytes of one kind of projection weight after a load.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectionReport {
    /// The representation the projections hold (`dense`, `ggml`, `prism`, `nvfp4`).
    pub kind: String,
    /// Projections of this kind.
    pub count: u64,
    /// Logical weight elements.
    pub params: u64,
    /// Resident weight bytes (projections whose storage is not measured are excluded).
    pub resident_bytes: u64,
}

/// What a load produced (sc-24139): the weight format the caller requested and the projection
/// kinds the loaded decoder actually holds. Read through
/// [`TextLlm::load_report`](crate::TextLlm::load_report).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadReport {
    /// The load-time weight format requested (`None` = the checkpoint's own encoding).
    pub requested: Option<Quantize>,
    /// Resident projections by kind, only the kinds present; empty when the backend does not
    /// census this architecture.
    pub projections: Vec<ProjectionReport>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_backend_without_cuda_names_itself_in_every_refusal() {
        let caps = BackendCapabilities::without_cuda("mlx", "metal");
        assert_eq!(caps.backend, "mlx");
        assert_eq!(caps.device, "metal");
        assert_eq!(caps.compute_capability, None);
        assert!(!caps.nvfp4.supported);
        let nvfp4 = caps.nvfp4.reason.as_deref().unwrap();
        assert!(nvfp4.starts_with("nvfp4: "), "{nvfp4}");
        assert!(nvfp4.contains("sm_120") && nvfp4.contains("mlx"), "{nvfp4}");
        assert!(!caps.cuda_graphs.supported);
        let graphs = caps.cuda_graphs.reason.as_deref().unwrap();
        assert!(
            graphs.starts_with("cuda_graphs: ") && graphs.contains("mlx"),
            "{graphs}"
        );
    }

    #[test]
    fn feature_support_constructors_pair_the_flag_with_the_reason() {
        assert_eq!(
            FeatureSupport::available(),
            FeatureSupport {
                supported: true,
                reason: None
            }
        );
        let off = FeatureSupport::unavailable("why");
        assert!(!off.supported);
        assert_eq!(off.reason.as_deref(), Some("why"));
    }
}
