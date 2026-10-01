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
    /// Which proposer actually ran (`none`, `mtp`, `prompt_lookup`, `draft_model`). `none`
    /// includes a request whose speculative option resolved to no proposer — the reason is then
    /// in [`fallbacks`](Self::fallbacks).
    pub proposer: ProposerKind,
    /// Draft tokens per verification pass (the depth) when a proposer ran.
    pub draft_tokens: Option<u32>,
    /// The sampler path: `device`, `host:<reason>` (for example `host:penalty`), or `none`.
    pub sampler: String,
    /// The KV cache implementation (`growing`, `static`).
    pub kv_cache: String,
    /// How attention was computed (`gqa`, `expanded`; `decode_attention` when a Candle request's
    /// cached steps ran the length-aware decode attention, sc-24441).
    pub attention: String,
    /// The CUDA-graph runner's part.
    pub cuda_graphs: CudaGraphsReport,
    /// Which graph path served the generation's steps (epic sc-24432 E3): `captured` when a
    /// captured, verified CUDA graph replayed at least one step (warm-up, self-check and prefill
    /// steps still run eager), `eager` when steps went through the graph runner but none replayed
    /// ([`CudaGraphsReport::fallback_reason`] names why), `none` when no step went through the
    /// runner (the switch was off, or the path does not use it), or a backend without CUDA
    /// graphs.
    pub graph_path: String,
    /// NVFP4 projection calls by path (`none` for a model without NVFP4 projections).
    pub nvfp4_projections: PathReport,
    /// Fused-versus-reference primitive runs.
    pub fused_primitives: PathReport,
    /// Target-model forward passes, including the prompt prefill.
    pub target_forwards: u64,
    /// The target forwards the prompt prefill took, counted in
    /// [`target_forwards`](Self::target_forwards): one for a prompt prefilled in one pass, two
    /// when a hybrid decoder's prefill split at the cross-turn prefix cache's snapshot boundary
    /// (story sc-24437), `0` on a path that does not report it. On the speculative engine
    /// `target_forwards == prefill_forwards + verify_steps + replay_forwards`.
    pub prefill_forwards: u64,
    /// Draft tokens proposed.
    pub proposed_tokens: u64,
    /// Draft tokens accepted by target verification.
    pub accepted_tokens: u64,
    /// Target verification passes the engine took — with no proposer every decode step is a
    /// one-token verify pass; `0` on a loop that does not verify (the reference loop). The
    /// denominator of [`mean_accepted_length`](Self::mean_accepted_length).
    pub verify_steps: u64,
    /// Verify steps recovered by a step-start rollback plus a replay forward.
    pub replay_forwards: u64,
    /// Leading prompt tokens whose cache state came from the cross-turn prefix cache instead of a
    /// prefill forward (story sc-24437): the prefill ran only the prompt past them. `0` on a miss,
    /// when the cache is off, and on a request it refuses (a multimodal prompt — its reason is in
    /// [`prefix_cache`](Self::prefix_cache)'s `reason`).
    pub prefix_hit_tokens: u64,
    /// The cross-turn prefix cache's part in this request (story sc-24437): `hit` (a stored
    /// prefix was restored — [`prefix_hit_tokens`](Self::prefix_hit_tokens) of it), `miss`
    /// (looked up, nothing reusable), `off` (the load settled a zero budget), `bypassed` (the
    /// request never reads or feeds the cache — the reason names why, e.g. a multimodal prompt
    /// whose image / video / audio rows are not in the token key), or `none` (a decode path the
    /// cache is not wired to). The reason also names why a request's own state was not kept.
    pub prefix_cache: PathReport,
    /// Every fallback this request took that no sub-report above already names (epic sc-24432
    /// E2/E3): the speculative option resolving to less than it asked for, or a proposer that
    /// could not run on the path this request decoded on. Each entry leads with the feature
    /// (`speculative: …`). Empty when nothing fell back. The sampler's host reason, the
    /// CUDA-graph fallback, the NVFP4 path and the fused-primitive reason stay in
    /// [`sampler`](Self::sampler), [`cuda_graphs`](Self::cuda_graphs),
    /// [`nvfp4_projections`](Self::nvfp4_projections) and
    /// [`fused_primitives`](Self::fused_primitives).
    pub fallbacks: Vec<String>,
}

impl DecodeReport {
    /// The realized mean accepted length: draft tokens accepted per verification pass
    /// (`accepted_tokens / verify_steps`), or `None` when no proposer ran or no verify step did.
    /// Each verify step also commits one target-chosen token, so tokens per verify step is this
    /// plus one.
    pub fn mean_accepted_length(&self) -> Option<f64> {
        (self.proposer != ProposerKind::None && self.verify_steps > 0)
            .then(|| self.accepted_tokens as f64 / self.verify_steps as f64)
    }
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
    /// The CUDA-graph switch the loaded model's generations run under, as the load settled it:
    /// [`LoadSpec::cuda_graphs`](crate::LoadSpec::cuda_graphs), else the backend's default at
    /// load. `None` where the switch does not apply — a provider that never routes decode steps
    /// through a CUDA-graph runner — so a product shows the settled value, not the request.
    pub cuda_graphs: Option<bool>,
    /// Every optional accelerator the load was asked for but did not attach, each leading with
    /// the feature (`mtp_head: …`), so a product can show why (epic sc-24432 E2: the model still
    /// loaded; the accelerator is absent). Empty when everything requested was attached.
    pub fallbacks: Vec<String>,
    /// The cross-turn prefix cache's byte budget the load settled (story sc-24437): the requested
    /// budget ([`LoadSpec::prefix_cache_bytes`](crate::LoadSpec::prefix_cache_bytes), else
    /// [`DEFAULT_PREFIX_CACHE_BYTES`](crate::DEFAULT_PREFIX_CACHE_BYTES)) clamped to the headroom
    /// the load's admission left — memory the loaded model may hold beyond its weights. `Some(0)`
    /// when the cache is off or no headroom was left; `None` where the provider has no prefix
    /// cache or was assembled without a load.
    pub prefix_cache_bytes: Option<u64>,
    /// The draft model the load named ([`LoadSpec::draft_source`](crate::LoadSpec::draft_source),
    /// epic sc-24432 story sc-24436): resident, or refused with the reason named. `None` when no
    /// draft was named.
    pub draft: Option<DraftReport>,
}

/// What became of a load's named draft model (sc-24436). A draft never fails the load (epic
/// sc-24432 E2): it is resident — and `draft_model` advertised — or refused with the reason
/// named and the target loaded alone.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DraftReport {
    /// The draft source the load named.
    pub source: String,
    /// Why the draft was not loaded (a tokenizer vocabulary that is not the target's, logits over
    /// more ids than the target's, an unreadable source, no room beside the target), or `None`
    /// when it is resident.
    pub refusal: Option<String>,
}

impl DraftReport {
    /// A resident draft.
    pub fn resident(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            refusal: None,
        }
    }

    /// A refused draft, with the reason.
    pub fn refused(source: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            refusal: Some(reason.into()),
        }
    }

    /// Whether the draft is resident (and `draft_model` advertised).
    pub fn is_resident(&self) -> bool {
        self.refusal.is_none()
    }
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
    fn mean_accepted_length_is_accepted_drafts_per_verify_step() {
        let mut report = DecodeReport {
            verify_steps: 4,
            ..DecodeReport::default()
        };
        assert_eq!(report.mean_accepted_length(), None, "no proposer ran");
        report.proposer = ProposerKind::PromptLookup;
        report.verify_steps = 0;
        assert_eq!(report.mean_accepted_length(), None, "no verify step ran");
        report.verify_steps = 4;
        report.accepted_tokens = 6;
        assert_eq!(report.mean_accepted_length(), Some(1.5));
        assert!(report.fallbacks.is_empty());
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
