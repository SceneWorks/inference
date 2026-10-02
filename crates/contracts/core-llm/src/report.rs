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
    /// cached steps ran the length-aware decode attention, sc-24441; `opaque` when the engine
    /// drove a decoder only through its token-at-a-time step and cannot see how it attends — the
    /// MLX captioner / SVG step targets).
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
    /// `target_forwards == prefill_forwards + verify_steps + replay_forwards +
    /// discarded_forwards`.
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
    /// Target forwards a pipelined loop enqueued as look-ahead and then discarded unread, counted
    /// in [`target_forwards`](Self::target_forwards): the step already enqueued when the run ended
    /// on a stop token, a stop string or a cancel (MLX pipelined decode, story sc-24439) — never
    /// emitted. `0` on a loop that does not look ahead (every Candle path).
    pub discarded_forwards: u64,
    /// Where `auto`'s acceptance monitor demoted this request's proposer (sc-24446, E5): the
    /// generated-token count when the probe window
    /// ([`ACCEPTANCE_PROBE_VERIFIES`](crate::ACCEPTANCE_PROBE_VERIFIES) verify steps) closed below
    /// the proposer's break-even ([`AcceptanceMonitor`](crate::AcceptanceMonitor)); every later
    /// token decoded without a proposer — on MLX through the pipelined token-at-a-time loop.
    /// [`proposer`](Self::proposer) and [`draft_tokens`](Self::draft_tokens) still name the
    /// proposer that ran before it, and the plain steps after it are counted in
    /// [`verify_steps`](Self::verify_steps) (one-token verify passes, as on every path). `None` when
    /// nothing was demoted — always for `off` and for an explicit `{proposer, depth}` request.
    pub speculative_demoted_at: Option<u64>,
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
    /// Name what a captioner or SVG decoder (JoyCaption / LLaVA, StarVector — both backends)
    /// does not run (E2/E3): the speculative fallback for the request's `mode`
    /// ([`no_proposer_fallback`](crate::no_proposer_fallback) with
    /// [`CAPTIONER_NO_PROPOSER`](crate::CAPTIONER_NO_PROPOSER)) and why its prefix cache is
    /// `none` ([`CAPTIONER_NO_PREFIX_CACHE`](crate::CAPTIONER_NO_PREFIX_CACHE)). One call, so both
    /// backends' captioners report the same words (E8).
    pub fn with_captioner_reasons(mut self, mode: crate::Speculative) -> Self {
        self.fallbacks = crate::no_proposer_fallback(mode, crate::CAPTIONER_NO_PROPOSER)
            .into_iter()
            .collect();
        self.prefix_cache.reason = Some(crate::CAPTIONER_NO_PREFIX_CACHE.to_string());
        self
    }

    /// The realized mean accepted length: draft tokens accepted per verification pass
    /// (`accepted_tokens / verify_steps`), or `None` when no proposer ran or no verify step did.
    /// Each verify step also commits one target-chosen token, so tokens per verify step is this
    /// plus one. On a request `auto` demoted ([`speculative_demoted_at`](Self::speculative_demoted_at))
    /// the denominator includes the plain steps after the demotion.
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

/// Fused-versus-reference primitive runs on one thread (epic sc-24432 E3): each backend keeps a
/// monotone thread-local tally of the primitives that have a fused route, counting every run by
/// the route that served it, and a request reports [`since`](Self::since) its start. The same
/// labels on both backends ([`label`](Self::label)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FusedTally {
    /// Runs served by a fused kernel.
    pub fused: u64,
    /// Runs served by the reference (unfused) route.
    pub reference: u64,
    /// Why the most recent reference run happened (`None` if none happened).
    pub reference_reason: Option<&'static str>,
}

impl FusedTally {
    /// The counts accumulated since `start` (the reason is the latest one, when a reference run
    /// happened since `start`).
    pub fn since(&self, start: &FusedTally) -> FusedTally {
        FusedTally {
            fused: self.fused.wrapping_sub(start.fused),
            reference: self.reference.wrapping_sub(start.reference),
            reference_reason: if self.reference != start.reference {
                self.reference_reason
            } else {
                None
            },
        }
    }

    /// Which route served the runs: `fused` (only fused), `reference` (no fused run), `mixed`
    /// (both), or `none` (no run).
    pub fn label(&self) -> &'static str {
        match (self.fused, self.reference) {
            (0, 0) => "none",
            (_, 0) => "fused",
            (0, _) => "reference",
            _ => "mixed",
        }
    }

    /// The report a request carries ([`DecodeReport::fused_primitives`]).
    pub fn path_report(&self) -> PathReport {
        PathReport {
            path: self.label().to_string(),
            reason: self.reference_reason.map(str::to_string),
        }
    }
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
    /// Every optional accelerator the load was asked for (or the checkpoint declared) but did not
    /// attach, each leading with the feature — `mtp_head: …` (a companion head refused), `mtp: …`
    /// (a configured native MTP head the snapshot does not carry, or an unsupported head variant),
    /// `draft model: …` (a named draft refused; also in [`draft`](Self::draft)),
    /// `prefix_cache: …` (a requested cache settled to zero bytes for lack of headroom),
    /// `cuda_graphs: …` (the graph switch is on but the device, the build or the decoder's step
    /// cannot capture) — so a product can show why (epic sc-24432 E2: the model still loaded; the
    /// accelerator is absent). Empty when everything requested was attached.
    pub fallbacks: Vec<String>,
    /// The cross-turn prefix cache's byte budget the load settled (story sc-24437): the requested
    /// budget ([`LoadSpec::prefix_cache_bytes`](crate::LoadSpec::prefix_cache_bytes), else
    /// the backend's [`DecodeDefaults::prefix_cache_bytes`](crate::DecodeDefaults::prefix_cache_bytes))
    /// clamped to the headroom the load's admission left — memory the loaded model may hold
    /// beyond its weights. `Some(0)` when the cache is off or no headroom was left; `None` where
    /// the provider has no prefix cache or was assembled without a load.
    pub prefix_cache_bytes: Option<u64>,
    /// The draft model the load named ([`LoadSpec::draft_source`](crate::LoadSpec::draft_source),
    /// epic sc-24432 story sc-24436): resident, or refused with the reason named. `None` when no
    /// draft was named.
    pub draft: Option<DraftReport>,
}

impl LoadReport {
    /// Record what became of the load's named draft (sc-24436): in [`draft`](Self::draft), and —
    /// for a refused draft — its reason in [`fallbacks`](Self::fallbacks) too (E2: every fallback
    /// named in one place).
    pub fn record_draft(&mut self, draft: DraftReport) {
        self.draft = Some(draft.named_in(&mut self.fallbacks));
    }

    /// Record the prefix-cache budget a load on `backend` settled (story sc-24437) for a request
    /// of `requested` ([`LoadSpec::prefix_cache_bytes`](crate::LoadSpec::prefix_cache_bytes)): in
    /// [`prefix_cache_bytes`](Self::prefix_cache_bytes), and — when a non-zero request settled to
    /// zero because admission left no headroom — a named `prefix_cache: …` entry in
    /// [`fallbacks`](Self::fallbacks). An explicit `Some(0)` turns the cache off and names nothing.
    pub fn record_prefix_budget(
        &mut self,
        backend: crate::DecodeBackend,
        requested: Option<u64>,
        settled: u64,
    ) {
        self.fallbacks
            .extend(prefix_budget_fallback(backend, requested, settled));
        self.prefix_cache_bytes = Some(settled);
    }
}

/// The load fallback for a prefix-cache budget (story sc-24437) that settled to zero bytes on
/// `backend` although `requested` ([`LoadSpec::prefix_cache_bytes`](crate::LoadSpec::prefix_cache_bytes),
/// `None` = the backend's [`DecodeDefaults::prefix_cache_bytes`](crate::defaults::DecodeDefaults::prefix_cache_bytes)) asked for some —
/// load admission left no headroom beside the model (E2/E7). `None` when the cache settled to a
/// non-zero budget or the request turned it off (`Some(0)`).
pub fn prefix_budget_fallback(
    backend: crate::DecodeBackend,
    requested: Option<u64>,
    settled: u64,
) -> Option<String> {
    let asked = crate::requested_prefix_cache_bytes(backend, requested);
    (settled == 0 && asked > 0).then(|| {
        format!(
            "prefix_cache: the requested {asked}-byte cache settled to 0 bytes — load admission \
             left no headroom beside the model; requests decode without it"
        )
    })
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

    /// This report, with a refusal's reason also pushed onto a load's `fallbacks` (E2: every
    /// load fallback named in [`LoadReport::fallbacks`]).
    pub fn named_in(self, fallbacks: &mut Vec<String>) -> Self {
        fallbacks.extend(self.refusal.clone());
        self
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

    /// E2/E8: a captioner's report names its speculative fallback (none for `off`) and why its
    /// prefix cache is `none`, in the shared words.
    #[test]
    fn a_captioner_report_names_its_proposer_and_prefix_cache_reasons() {
        let base = DecodeReport {
            prefix_cache: PathReport {
                path: "none".into(),
                reason: None,
            },
            ..DecodeReport::default()
        };
        let auto = base
            .clone()
            .with_captioner_reasons(crate::Speculative::Auto);
        assert_eq!(auto.fallbacks.len(), 1);
        assert!(
            auto.fallbacks[0].contains(crate::CAPTIONER_NO_PROPOSER),
            "{:?}",
            auto.fallbacks
        );
        assert_eq!(auto.prefix_cache.path, "none");
        assert_eq!(
            auto.prefix_cache.reason.as_deref(),
            Some(crate::CAPTIONER_NO_PREFIX_CACHE)
        );
        let off = base.with_captioner_reasons(crate::Speculative::Off);
        assert!(off.fallbacks.is_empty());
        assert!(off.prefix_cache.reason.is_some());
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

    /// E2: a non-zero prefix-cache request settled to zero bytes is named; an explicit `0` and
    /// a cache that fits name nothing.
    #[test]
    fn a_prefix_cache_settled_to_zero_by_admission_is_named() {
        let mut load = LoadReport::default();
        load.record_prefix_budget(crate::DecodeBackend::Mlx, None, 0);
        assert_eq!(load.prefix_cache_bytes, Some(0));
        assert_eq!(load.fallbacks.len(), 1, "{:?}", load.fallbacks);
        assert!(
            load.fallbacks[0].starts_with("prefix_cache: the requested 1073741824-byte cache"),
            "{:?}",
            load.fallbacks
        );
        for (requested, settled) in [(Some(0), 0), (Some(64), 64), (None, 5)] {
            let mut load = LoadReport::default();
            load.record_prefix_budget(crate::DecodeBackend::Mlx, requested, settled);
            assert_eq!(load.prefix_cache_bytes, Some(settled));
            assert!(load.fallbacks.is_empty(), "{requested:?} -> {settled}");
        }
    }

    /// E3: the fused tally reports the route that ran since a request's start, with the latest
    /// reference reason only when a reference run happened in that window.
    #[test]
    fn the_fused_tally_reports_the_routes_since_the_start() {
        let start = FusedTally {
            fused: 2,
            reference: 1,
            reference_reason: Some("old"),
        };
        assert_eq!(start.since(&start).label(), "none");
        assert_eq!(start.since(&start).path_report().reason, None);
        let fused = FusedTally { fused: 5, ..start };
        assert_eq!(
            fused.since(&start).path_report(),
            PathReport {
                path: "fused".into(),
                reason: None
            }
        );
        let mixed = FusedTally {
            fused: 5,
            reference: 2,
            reference_reason: Some("cpu_stream"),
        };
        assert_eq!(
            mixed.since(&start).path_report(),
            PathReport {
                path: "mixed".into(),
                reason: Some("cpu_stream".into())
            }
        );
        let reference = FusedTally {
            reference: 3,
            ..mixed
        };
        assert_eq!(
            reference.since(&FusedTally { fused: 5, ..start }).label(),
            "reference"
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
