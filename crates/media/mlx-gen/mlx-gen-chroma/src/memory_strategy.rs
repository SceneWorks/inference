//! The shared image-memory ladder for **Chroma1 HD / Base / Flash** on MLX/Metal (SC-15520).
//!
//! Chroma is the FLUX MMDiT skeleton with pruned adaLN — 19 double + 38 single blocks whose
//! modulation comes from a dense `distilled_guidance_layer` Approximator rather than from per-block
//! `norm*.linear` weights — plus **T5-XXL-only** conditioning and the FLUX.1 16-channel VAE. All
//! three catalog entries (`chroma1_hd`, `chroma1_base`, `chroma1_flash`) are the *same* provider
//! route: one architecture, one loader, one contract shape; they differ only in checkpoint, sampler
//! default and schedule. Nothing here is inherited from FLUX.1 or from any other family — the epic's
//! standing rule is that a rung's presence, magnitude and candidate set are per family per backend,
//! so every number this module publishes comes from
//! `tests/memory_ladder_real_weights.rs` on this Mac's Metal GPU.
//!
//! ## What each rung does here
//!
//! Measured on **Chroma1-Base q4 at 1024², Apple/Metal**, 4 steps, true CFG 4.0. Every row is a
//! fresh generator, one discarded warm-up row of the same shape, `reset_peak_memory` after the load.
//!
//! | rung | mechanism | request peak | state |
//! |---|---|---|---|
//! | 0 resident | — | 28.0779 GiB | baseline |
//! | 1 staged residency | shed T5-XXL before the DiT/VAE load, per request | **19.2065 GiB (−31.60%)** | Implemented |
//! | 2 bounded decode | `Vae::decode_tiled` on the FLUX.1 16-ch AutoencoderKL | bounds the phase; SC-19753 preserves full-image GroupNorm statistics | route-blind Missing; exact measured rows may be Implemented, see [`DECODE_SUPPORT`] |
//! | 3 bounded attention | `sdpa_budgeted_bhsd` on both block stacks | bounds the phase; **not the binding one** | Missing, see [`ATTENTION_SUPPORT`] |
//! | 4 bounded transformer residency | `run_windowed` over the two DiT sub-stacks | **14.6932 GiB (−23.50% on rung 1)** | Implemented |
//!
//! The rung-2 row is the route-blind legacy default. A caller may now replace it for one exact
//! catalog route, tier, mode, overlay and output geometry by packaging a sealed production-latent
//! quality policy. Coordinates absent from that table remain refused; an empty table preserves the
//! default and the existing estimated-fit fallback.
//! SC-19753 changes the decoder to preserve full-activation GroupNorm statistics and refreshes the
//! packaged policy rows; the pre-SC-19753 drift figures below remain historical context for the
//! route-blind fallback, not claims about the new layer-wise tiled arithmetic.
//!
//! Rung 4's saving is attributed rather than assumed: 4.5133 GiB measured against a 4.5125 GiB
//! windowable block weight set read from the snapshot's own safetensors `data_offsets` — the whole
//! set, to within 0.8 MiB, which is what a window that replaces the resident stack should buy and
//! nothing more.
//!
//! ## What measurement decided, and what it overturned
//!
//! Two verdicts here are the opposite of what the architecture suggests, and both were reached by
//! measuring rather than by reasoning from the sibling families:
//!
//! * **The route-blind rung 2 is `Missing`.** Before SC-19753, tiling the whole FLUX.1 VAE tail saved
//!   −11% to −72% of the decode phase but its per-crop GroupNorm model produced a 28..82/255 sample.
//!   That historical result explains the fail-closed fallback. The current layer-wise decoder keeps
//!   full-image statistics, and sealed exact-coordinate evidence may adopt it. See [`DECODE_SUPPORT`].
//! * **Rung 3 is `Missing`.** MLX's fused attention kernel streams its scores, so query-row
//!   chunking has no materialized score matrix to bound: measured **−0.001%** on the request peak.
//!   See [`ATTENTION_SUPPORT`].
//! * **Rung 4 is scoped at the `Dit`, not the text encoder.** Chroma's shipped q4 tier packs only
//!   the DiT block linears, so the T5-XXL encoder is the largest single component (dense bf16, 10.15
//!   GiB against a 7.14 GiB packed DiT) — which makes the text-encoder scope look obviously right
//!   and is exactly why it needed measuring. Rung 1 has already shed the encoder before the heavy
//!   phase loads, so the request peak is the heavy phase and a `TextEncoder` window moves it by
//!   **−0.00%**. See [`TRANSFORMER_WINDOW_COMPONENT`].
//!
//! sc-16462 (inference PR #443) packs the T5/VAE auxiliaries. It changes the conditioning phase's
//! weight, so every conditioning number here is re-derived when it lands, but it does not disturb
//! the rung-3 or rung-4 verdicts; rung 2 is now governed by SC-19753's sealed coordinate evidence.

use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};
use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::gen_core::{
    adapter_stack_resident_bytes, standard_memory_strategy_safety_check, AdapterResidencyMode,
    Error as CoreError, LoadShape, LoadSpec, MemoryBackendRealization, MemoryCalibrationIdentity,
    MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent,
    MemoryRunContext, MemorySafetyDecision, MemoryStrategy, MemoryStrategyPrerequisite,
    MemoryStrategySupport, ResidentRequestMemory, TransformerComponent,
};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{GenerationRequest, OffloadPolicy, WeightsSource};

use crate::config::{ChromaTransformerConfig, ChromaVariant};

/// Static, weights-free identity for the registry conformance walk. Never a production calibration.
const STATIC_CALIBRATION: &str = "chroma1-mlx-registry-behavior-v1";

/// The production calibration key. Bound to the measured entry/tier/geometry in
/// `production_calibration_fingerprint`; every other axis fails closed.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "chroma1-base-q4-mlx-shared-ladder-2026-08-05-v1";

// ── Rung 2: the native VAE decode geometry ───────────────────────────────────────────────────────

/// Tile edges swept against the exact untiled decode of the **production** latent, in output pixels.
///
/// Recorded here rather than in a comment so `the_withheld_rungs_are_refused_by_the_production_path`
/// can re-assert every rejection against the production admission path, and
/// `native_and_pid_decode_routes_are_disjoint_and_checked` against the checked route set.
pub const DECODE_TILE_EDGES_SWEPT: &[u32] = &[960, 896, 832, 768, 640, 512, 384];
/// Feather overlaps swept beside [`DECODE_TILE_EDGES_SWEPT`].
pub const DECODE_OVERLAPS_SWEPT: &[u32] = &[64, 128, 192, 256];

/// **The route-blind rung-2 fallback is `Missing`.**
///
/// The following pre-SC-19753 history was measured on Chroma1-Base q4 at 1024² at the variant's real **28-step** schedule
/// (`decode_tile_mechanism_sweep_on_the_production_latent`), against the exact untiled decode of the
/// **production** latent — what the denoiser hands the decode phase, not a re-encoded finished image
/// whose statistics have already been through the VAE round trip:
///
/// | edge | overlap | tiles | isolated peak | vs untiled | max Δ |
/// |---:|---:|---:|---:|---:|---:|
/// | 832 | 256 | 4 | 9.4892 GiB | −32.9% | **53** |
/// | 768 | 192 | 4 | 8.0931 GiB | −42.7% | 54 |
/// | 768 | 256 | 4 | 8.0931 GiB | −42.7% | 54 |
/// | 896 | 256 | 4 | 10.9988 GiB | −22.2% | 57 |
/// | 768 | 128 | 4 | 8.0931 GiB | −42.7% | 59 |
/// | 640 | 256 | 4 | 5.8147 GiB | −58.9% | 60 |
/// | … 28 cells swept … | | | | | |
/// | 896 | 64 | 4 | 10.9988 GiB | −22.2% | 127 |
///
/// The mechanism works and the saving is large: **−11.4% to −72.4%** of the decode phase across the
/// sweep. The best cell drifts **53/255** against the harness's `SIBLING_DRIFT_BAR` of 48 — the worst
/// drift `mlx_gen_z_image` *admits* into a shipped ladder on the same tiling machinery over the same
/// `AutoencoderKL` type.
///
/// ## Historical pre-SC-19753 classification: UNRESOLVED rather than failure
///
/// **A previous revision of this doc published 105–166/255, called it "more than double the bar",
/// and built a mechanism argument on the drift being flat in the geometry. All of that was an
/// artifact of measuring on a 4-step latent.** At the real schedule the range is 53–127, the best
/// cell is ~10% over the bar rather than 120% over, and the drift is *not* flat: at overlap 256 it
/// falls monotonically 70 → 57 → 53 across edges 960 → 896 → 832, which is the tail-tile behaviour
/// the old paragraph claimed to have ruled out. That argument is withdrawn. Only the numbers stand.
///
/// **And 10% is far inside the statistic's own variance, which is measured rather than assumed.**
/// `max Δ` is an extreme-order statistic over ~3.1M subpixels — a single outlier sets it — so
/// the pre-SC-19753 version of `layerwise_decode_quality_is_resampled_across_seeds` re-took the best cell across five
/// production latents:
///
/// | seed | 1234 | 7 | 99 | 20260805 | 424242 |
/// |---|---:|---:|---:|---:|---:|
/// | max Δ | 53 | 82 | 74 | 63 | **28** |
/// | mean Δ | 1.87 | 3.57 | 2.59 | 3.45 | 0.62 |
///
/// **28..82 on one fixed geometry — a 2.9x spread, an order of magnitude wider than the margin the
/// single-image sweep appeared to establish, and it straddles the bar.** Four seeds fail it, one
/// clears it comfortably. So the rung cannot be admitted, and it also cannot honestly be called a
/// clean rejection: on this evidence `max Δ` against a borrowed 48/255 threshold does not resolve
/// this candidate in either direction.
///
/// Those results explain why the route-blind constant stays fail-closed. SC-19753 does not promote
/// the constant; it packages new exact production-latent geometry rows, leaving every absent or
/// mismatched coordinate refused.
pub const DECODE_SUPPORT: bool = false;
/// The native VAE tile ladder this provider *would* publish, and the reference domain its checked
/// [`mlx_gen_pid::DecodeRoutes`] is constructed from so the native and PiD domains stay provably
/// disjoint. With [`DECODE_SUPPORT`] `false` none of it reaches a capability's published parameters
/// and the production path refuses every bounded-decode request before this is consulted.
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512];
/// The default edge inside [`DECODE_TILE_EDGES`].
pub const DECODE_TILE_EDGE: u32 = 512;
/// The one feather overlap paired with [`DECODE_TILE_EDGES`].
pub const DECODE_OVERLAP: u32 = 128;

// ── Rung 3: the bounded-attention budget ─────────────────────────────────────────────────────────

/// The single attention-score budget this provider would publish — the shared
/// `CONSTRAINED_ATTN_SCORES_BUDGET`.
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;

/// **Rung 3 is `Missing` on this family, and it is a measured verdict rather than an omission.**
///
/// The mechanism works. Measured at the DiT seam on Chroma1-Base q4 at 1024²
/// (`attention_chunking_is_measured_at_the_dit_seam`, which drives
/// `ChromaTransformer::forward_with_attention_plan` directly because the production path refuses a
/// bounded-attention request):
///
/// | | peak | output |
/// |---|---:|---|
/// | unbounded | 7.1679 GiB | — |
/// | bounded | 7.0053 GiB | **bit-exact**, `max|Δv| = 0.000e0` |
///
/// **−2.27%, and it is confined to a phase that does not bind.** The 0.1626 GiB seam saving sits
/// inside the 12.05 GiB headroom between the denoise phase and the staged request peak, so a caller
/// pays wall clock for a request peak that does not move — which per the epic is not a saving. That
/// test asserts the relation as live arithmetic over two quantities it measures in the same run, so
/// it reddens if the denoise phase ever becomes peak-bearing rather than carrying a remembered
/// figure.
///
/// The shape is expected on MLX and the reason is structural:
/// `fast::scaled_dot_product_attention` dispatches to a fused Metal kernel that **streams** the
/// scores, so there is no `[B,H,Sq,Sk]` materialization for query-row chunking to bound — see
/// [`mlx_gen::attention`], where the same rung is worth −32% on candle and −1.7% on MLX's Z-Image
/// denoise phase. Chroma's DiT carries an additional cost candle's does not: its MMDiT mask is a
/// per-query `[B,1,S,S]` tensor (85 MiB per CFG branch at 1024²) which the bounded kernel must
/// *narrow* per chunk, adding transients on the side of the ledger the rung is supposed to help.
///
/// The mechanism is wired and correct — `sdpa_budgeted_bhsd` is threaded through both block stacks
/// and both attention sites — so a future geometry, tier or kernel that makes it pay only has to
/// flip this flag. Until one does, admitting it would sell a caller a strategy that costs wall clock
/// and buys nothing.
pub const ATTENTION_SUPPORT: bool = false;

// ── Rung 4: the transformer window ───────────────────────────────────────────────────────────────

/// Window cadences swept over the DiT sub-stacks.
pub const TRANSFORMER_WINDOW_SIZES_SWEPT: &[u32] = &[1, 2, 5, 10];
/// Whether bounded transformer residency survived measurement on the production path.
///
/// It did, and by the largest margin of any rung here: **19.2065 → 14.6932 GiB, −23.50%** on top of
/// rung 1, with a byte-identical image at every cadence.
pub const WINDOW_SUPPORT: bool = true;
/// The published window cadence domain — **equal-peak alternatives, not a latency frontier**.
///
/// All four cadences bound the request peak to 14.6932 GiB, identical to the fourth decimal, in
/// three different execution orders. They are published together because none is wrong and a caller
/// with its own cost model may prefer any of them.
///
/// What is deliberately *not* published is an ordering. Sibling providers publish their cadences as
/// a time/memory trade — SDXL measured 3654 → 1318 ms/step widening 1 → 10 — and this family's
/// ms/step column looks like a trade until the order probe is applied, at which point it stops
/// looking like anything: re-running the same four cadences in a different execution order moves the
/// times with the *slots*. Cadence 10 has measured both the slowest row and the fastest one across
/// orders, and an independent re-run reproduced neither ordering.
///
/// No table of those timings is reproduced here on purpose — a printed table reads as a finding, and
/// the only publishable statement is that the ordering is **unresolvable on this instrument**
/// (SC-17679). [`TRANSFORMER_WINDOW_SIZE`]'s default therefore rests on being the tightest weight
/// bound — what the rung exists for — and not on a latency argument.
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1, 2, 5, 10];
/// The default cadence inside [`TRANSFORMER_WINDOW_SIZES`] — the tightest weight bound.
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;
/// The **measured** default component scope, and the one a request that names none receives.
///
/// This is not a `Dit`-by-inheritance choice — it is the outcome of measuring all three scopes end
/// to end against rung 4's own control on Chroma1-Base q4 at 1024² (Apple/Metal,
/// `the_window_component_scopes_are_measured_not_inherited`):
///
/// | scope | request peak | vs control | ms/step |
/// |---|---:|---:|---:|
/// | control (no window) | 19.2065 GiB | — | 10033 |
/// | `TextEncoder` | 19.2065 GiB | **−0.00%** | 9081 |
/// | `Dit` | **14.6932 GiB** | **−23.50%** | 12243 |
/// | `Both` | 14.6932 GiB | −23.50% | 10963 |
///
/// The control and `TextEncoder` rows previously read 19.2071 GiB. That is `chroma1_hd`'s staged
/// peak, not `chroma1_base`'s — the two differ by 0.0006 GiB, and this table is labelled Base. HD
/// re-took this comparison, so the figure was real but attributed to the wrong entry. Re-taken on
/// Base at the current pin it is 19.2065 GiB, which is exactly what this family's own rung-1 row
/// publishes. The percentages and both `Dit` rows are unchanged (sc-15520 review round 2).
///
/// All three render a byte-identical image; only one moves the request peak. The reason
/// `TextEncoder` is *exactly* inert here is worth stating, because the naive reading of the phase
/// probe predicts the opposite: the conditioning phase is genuinely the largest single component
/// (T5-XXL ships dense bf16 at 10.15 GiB against a 7.14 GiB packed DiT), but **rung 1 has already
/// shed it** before the heavy phase loads, so the request peak is the heavy phase — and bounding a
/// phase that is not the request peak is not a saving.
///
/// `Both` matches `Dit` to the fourth decimal and costs 22% more wall clock, so it is dominated
/// rather than wrong. All three stay published because all three are implemented and
/// output-preserving; the numbers above are what a selector's cost model should read, and the
/// harness re-asserts them.
///
/// **This verdict is coupled to rung 2's.** Were the decode bounded, the heavy phase would fall
/// below conditioning and `TextEncoder` would become the scope that binds — which is what an earlier
/// revision of this module concluded from the phase probe alone, before rung 2 was measured and
/// withheld. Kolors' opposite verdict (SC-15521) is the same arithmetic with a different decode.
pub const TRANSFORMER_WINDOW_COMPONENT: TransformerComponent = TransformerComponent::Dit;
/// The component scopes this provider will admit, [`TRANSFORMER_WINDOW_COMPONENT`] first.
///
/// The head is load-bearing: the shared behavior fixture takes the first published candidate as its
/// representative, so a domain whose head is not the default would have the conformance walk
/// exercising a scope no unqualified request receives.
pub const TRANSFORMER_WINDOW_COMPONENTS: &[TransformerComponent] = &[
    TransformerComponent::Dit,
    TransformerComponent::TextEncoder,
    TransformerComponent::Both,
];

/// The depth the shared request scope validates window alignment against: the **longest** windowable
/// sub-stack across every admitted component scope, since each stack is windowed independently over
/// the same cadence. Chroma's are 19 double + 38 single DiT blocks and 24 T5 encoder blocks.
pub(crate) fn window_stack_depth() -> usize {
    let cfg = ChromaTransformerConfig::default();
    cfg.num_layers
        .max(cfg.num_single_layers)
        .max(mlx_gen_flux::T5_BLOCKS)
}

/// The native VAE decode ladder, declared through the checked shared constructor so it can never
/// overlap the PiD student's disjoint domain (sc-15775).
fn decode_routes(provider_id: &str) -> mlx_gen::gen_core::Result<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(
        provider_id,
        DECODE_TILE_EDGES.iter().copied(),
        DECODE_OVERLAP,
    )
}

/// Stable provider-local ids for the overlays [`resident_overlay_components`] declares.
const PID_COMPONENT_ID: &str = "chroma.pid.student_and_caption_encoder";
const ADAPTER_COMPONENT_ID: &str = "chroma.adapters.forward_residuals";

/// The **auxiliary networks this load keeps resident alongside the base three**, priced load-exact
/// (sc-15839, mirroring the Kolors fix in `mlx_gen_kolors::memory_strategy`).
///
/// Until this existed Chroma declared `overlay_bytes = 0` while advertising a PiD decode overlay,
/// and the shared validator only cross-checks the overlay legs `if overlay_bytes > 0` — so the zero
/// passed silently and a rung-1 or rung-4 selection taken for a PiD load under-predicted peak by the
/// whole student plus its caption encoder.
///
/// **The PiD carve-out, and why it is policy-scoped rather than absent.** A `pid` on the `LoadSpec`
/// does not mean the request selected it: `build_residency`'s Sequential arm threads `req.use_pid`
/// per request, so a non-PiD generate never materializes the student. That reasoning is false under
/// `Resident`, where `Residency::ensure_warm_locked` calls `load_heavy(true, …)` **unconditionally**
/// — the PiD superset is loaded once and every later request runs with it resident whether or not it
/// asked. Pricing it at zero there under-declares the ordinary path rather than avoiding an
/// overstatement, so it is priced at exactly the policy that makes it unconditional.
///
/// Projections mirror what the loader does, not what the file stores. `PidEngine::from_spec` loads
/// the student and the Gemma caption encoder verbatim (`Weights::from_file` / `from_dir`, no cast),
/// so `Stored` is exact; Chroma's own three components have no `cast_all` on any path, which is why
/// none of them needs a `ResidentProjection::Float32` correction the way the SDXL VAE did.
///
/// Adapters are priced through the shared `adapter_stack_resident_bytes` at
/// [`AdapterResidencyMode::Additive`], which is what `apply_chroma_adapters` installs — a
/// rank-decomposed factor pair held on each `AdaptableLinear` and applied at forward time, never a
/// folded `[out, in]` delta. `None` from that helper means an additive stack could not be sized, and
/// this fails closed on it rather than declaring a zero the validator would wave through.
fn resident_overlay_components(
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<Vec<MemoryResidentComponent>> {
    let mut components = Vec::new();

    if spec.offload_policy == OffloadPolicy::Resident {
        if let Some(pid) = &spec.pid {
            let (WeightsSource::Dir(checkpoint) | WeightsSource::File(checkpoint)) =
                &pid.checkpoint;
            let (WeightsSource::Dir(gemma) | WeightsSource::File(gemma)) = &pid.gemma;
            // `PidEngine::load` prefers Gemma's merged single file and falls back to the shard dir.
            let merged = gemma.join(mlx_gen_pid::engine::GEMMA_MERGED_FILE);
            let gemma_source = if merged.is_file() {
                merged
            } else {
                gemma.clone()
            };
            let student = projected_safetensors_bytes(checkpoint, |_| ResidentProjection::Stored)?;
            let caption =
                projected_safetensors_bytes(&gemma_source, |_| ResidentProjection::Stored)?;
            push_overlay(
                &mut components,
                PID_COMPONENT_ID,
                // `AdapterStack` is the closest existing kind for an auxiliary network installed
                // beside the base model's transformers; what the contract arithmetic consumes is
                // `MemoryComponentKind::is_auxiliary()`, which is true for it.
                MemoryComponentKind::AdapterStack,
                student.saturating_add(caption),
            );
        }
    }

    let adapter_bytes =
        adapter_stack_resident_bytes(&spec.adapters, AdapterResidencyMode::Additive).ok_or_else(
            || {
                CoreError::Unsupported(
                    "chroma: an adapter stack was requested but at least one source could not be \
                     sized; refusing to declare a zero the shared validator would wave through"
                        .to_owned(),
                )
            },
        )?;
    push_overlay(
        &mut components,
        ADAPTER_COMPONENT_ID,
        MemoryComponentKind::AdapterStack,
        adapter_bytes,
    );

    Ok(components)
}

/// Record one overlay, skipping a zero: the shared validator refuses a declared component with zero
/// bytes, and a component that measured zero is not evidence of residency anyway.
fn push_overlay(
    into: &mut Vec<MemoryResidentComponent>,
    id: &str,
    kind: MemoryComponentKind,
    resident_bytes: u64,
) {
    if resident_bytes == 0 {
        return;
    }
    into.push(MemoryResidentComponent {
        id: id.to_owned(),
        kind,
        resident_bytes,
        // No published rung bounds an overlay here: rung 4's window covers the DiT's two block
        // sub-stacks (and, unpublished, the T5 encoder's), and nothing else.
        bounded_by: None,
        residency: MemoryComponentResidency::WholeRender,
    });
}

fn variant_for(provider_id: &str) -> mlx_gen::gen_core::Result<ChromaVariant> {
    match provider_id {
        crate::CHROMA1_HD_ID => Ok(ChromaVariant::Hd),
        crate::CHROMA1_BASE_ID => Ok(ChromaVariant::Base),
        crate::CHROMA1_FLASH_ID => Ok(ChromaVariant::Flash),
        _ => Err(CoreError::Unsupported(format!(
            "unknown Chroma1 memory provider {provider_id}"
        ))),
    }
}

/// The `'static` provider id for a route, so every message names the same string the registry does.
fn provider_static(provider_id: &str) -> mlx_gen::gen_core::Result<&'static str> {
    Ok(variant_for(provider_id)?.id())
}

/// The overlay axes a request carries, joined into the contract's route key. `None` is the clean
/// base route — the only one whose quality-sensitive rungs are admitted.
///
/// Chroma's crate reads only `adapters` and `pid` today; the remaining axes are listed anyway so a
/// spec that carries one fails **closed** here rather than being silently treated as clean by a
/// provider that would ignore it.
fn route_overlay(spec: &LoadSpec) -> Option<String> {
    let mut axes = Vec::new();
    if !spec.adapters.is_empty() {
        axes.push("adapters");
    }
    if spec.control.is_some() {
        axes.push("control");
    }
    if !spec.extra_controls.is_empty() {
        axes.push("extra-controls");
    }
    if spec.ip_adapter.is_some() {
        axes.push("ip-adapter");
    }
    if spec.identity.is_some() {
        axes.push("identity");
    }
    if spec.text_encoder.is_some() {
        axes.push("external-text-encoder");
    }
    if spec.pid.is_some() {
        axes.push("pid");
    }
    (!axes.is_empty()).then(|| axes.join("-"))
}

fn clean(spec: &LoadSpec) -> bool {
    route_overlay(spec).is_none()
}

/// Chroma is text-to-image only (`conditioning: vec![]`), so the route is fixed.
fn route_mode_and_references(spec: &LoadSpec) -> (MemoryMode, u32) {
    let _ = spec;
    (MemoryMode::TextToImage, 0)
}

/// Whether this load can rebuild individual DiT blocks from the snapshot on demand.
///
/// `spec.quantize.is_some()` is refused deliberately: that combination means a **dense** source the
/// loader packs in place, so a window would re-quantize its blocks on every re-materialization —
/// a host-format conversion per window, which is not what
/// [`MemoryWindowMaterialization::DeviceFormatTransfer`](mlx_gen::gen_core::MemoryWindowMaterialization)
/// declares. A shipped packed tier loads with `quantize == None` (the loader packed-detects
/// `{base}.scales`), so the production Q4/Q8 route is admitted and only the re-quantizing one is not.
pub fn structurally_streamable(spec: &LoadSpec) -> bool {
    WINDOW_SUPPORT
        && spec.load_shape == LoadShape::DeferredMaterialization
        && spec.quantize.is_none()
        && clean(spec)
        && matches!(spec.weights, WeightsSource::Dir(_))
}

/// Weights-free counterpart of [`structurally_streamable`] for the finite registry surface.
///
/// On a real load `spec.quantize` means "pack this dense source now" and is therefore refused by
/// [`structurally_streamable`]. In a [`MemoryContractSurfaceSpec`](mlx_gen::gen_core::MemoryContractSurfaceSpec),
/// however, that same field is the tensor-free witness for the selector's already-resolved artifact
/// tier. Q4/Q8 catalog tiers are prepacked and reach production with `quantize == None`; treating the
/// witness as load-time quantization would erase both shipped tiers from generated capabilities.
fn weights_free_structurally_streamable(
    surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec,
) -> bool {
    let resolved_artifact_tier = surface.resolved_artifact_tier();
    let spec = &surface.spec;
    WINDOW_SUPPORT
        && matches!(
            resolved_artifact_tier,
            mlx_gen::gen_core::MemoryContractSurfaceTier::Bf16
                | mlx_gen::gen_core::MemoryContractSurfaceTier::Q4
                | mlx_gen::gen_core::MemoryContractSurfaceTier::Q8
        )
        && spec.load_shape == LoadShape::DeferredMaterialization
        && clean(spec)
        && matches!(spec.weights, WeightsSource::Dir(_))
}

#[allow(clippy::too_many_arguments)]
fn build_contract(
    provider_id: &str,
    spec: &LoadSpec,
    streamable: bool,
    calibration: Option<MemoryCalibrationIdentity>,
    footprint: mlx_gen::PerComponentBytes,
    overlays: Vec<MemoryResidentComponent>,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    variant_for(provider_id)?;
    let clean = clean(spec);
    let decode_policies = spec.decode_geometry_policies_for_loaded_contract(
        mlx_gen::gen_core::MemoryBackend::Mlx,
        loaded_tier(spec),
    )?;
    let geometry_decode = decode_policies.iter().any(|policy| {
        matches!(
            policy.disposition,
            mlx_gen::gen_core::MemoryDecodeQualityDisposition::Admitted
        )
    });
    let decode = (DECODE_SUPPORT && clean) || geometry_decode;
    let attention = ATTENTION_SUPPORT && clean;
    let routes = decode_routes(provider_id)?;
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.architecture_facts = architecture_facts();
    contract.calibration = calibration;
    contract.asset_facts.overlay_bytes = overlays
        .iter()
        .try_fold(0_u64, |total, component| {
            total.checked_add(component.resident_bytes)
        })
        .ok_or_else(|| CoreError::Msg("chroma: overlay byte sum overflow".to_owned()))?;
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::DecodeTileArea,
        MemoryFormulaVariable::AttentionChunkSize,
        MemoryFormulaVariable::TransformerWindowSize,
        MemoryFormulaVariable::OverlayBytes,
    ];
    // The component axis only where there IS a resident overlay: the shared validator refuses a
    // declared component with zero bytes, and `ComponentPhaseEnvelope` with an empty vector would
    // claim an axis this load does not use.
    contract.formula = if overlays.is_empty() {
        MemoryFormulaKind::PhaseEnvelope { phases, variables }
    } else {
        MemoryFormulaKind::ComponentPhaseEnvelope {
            phases,
            variables,
            resident_components: overlays,
        }
    };
    contract.asset_facts.base_bytes = footprint
        .text_encoder
        .saturating_add(footprint.dit)
        .saturating_add(footprint.vae);
    contract.asset_facts.conditioning_bytes = footprint.text_encoder;
    contract.asset_facts.transformer_bytes = footprint.dit;
    contract.asset_facts.decoder_bytes = footprint.vae;

    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        // Chroma uses request_scoped_from_policy: load-time offload is only the default and every
        // cached generator can execute a staged request that evicts its warm pair first.
        synchronized_phase_release: true,
        decode_tiling: decode,
        attention_chunking: attention,
        transformer_window_materialization: streamable,
    };
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
            MemoryStrategy::StagedResidency => MemoryStrategySupport::Implemented,
            MemoryStrategy::BoundedDecode if decode => {
                capability.parameters.decode_tile_edges = routes.native_edges().to_vec();
                capability.parameters.decode_overlaps = vec![DECODE_OVERLAP];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedAttention if attention => {
                capability.parameters.attention_chunk_sizes = vec![ATTENTION_CHUNK_SIZE];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency if streamable => {
                capability.parameters.transformer_window_sizes = TRANSFORMER_WINDOW_SIZES.to_vec();
                capability.parameters.transformer_window_components =
                    TRANSFORMER_WINDOW_COMPONENTS.to_vec();
                MemoryStrategySupport::Implemented
            }
            _ => MemoryStrategySupport::Missing,
        };
    }
    contract.resident_request_memory = ResidentRequestMemory::ExplicitResident;
    contract.decode_geometry_policy_authoritative = spec.decode_geometry_policy_authoritative;
    contract.adopt_decode_geometry_policies("chroma", decode_policies)?;
    if streamable {
        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedTransformerResidency,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
    }
    Ok(contract)
}

/// The measured production key.
///
/// **The tier is part of the key, and it has to be** (review of PR #496). The four axes this used to
/// test — provider, precision, `quantize.is_none()`, policy — do not distinguish a q8 or bf16
/// snapshot from the measured q4 one, because all three shipped tiers are *pre-packed* and therefore
/// all three load with `quantize == None`. Every tier of `chroma1_base` inherited the q4 key while
/// the doc above it called that key exact, so a q8 or bf16 request could have selected an optimized
/// fit on q4-measured evidence.
///
/// The discriminant is the tier directory's own packed marker, read from the transformer component's
/// `config.json` — the same `quantization.bits` the loader packed-detects on — so it names what was
/// actually measured rather than what a path happens to be called. An unreadable or absent marker
/// fails closed: no calibration, and the selector cannot reach an optimized strategy at all.
///
/// Measured: `chroma1_base`, transformer packed at **4** bits, bf16 precision, Sequential, clean
/// route. Everything else — the other two entries, the q8 and bf16 tiers, every overlay — is
/// deliberately uncalibrated until it has its own evidence (sc-17695).
fn production_calibration_fingerprint(provider_id: &str, spec: &LoadSpec) -> Option<&'static str> {
    if provider_id != crate::CHROMA1_BASE_ID
        || spec.precision != mlx_gen::Precision::Bf16
        || spec.quantize.is_some()
        || spec.offload_policy != OffloadPolicy::Sequential
        || !clean(spec)
    {
        return None;
    }
    let WeightsSource::Dir(root) = &spec.weights else {
        return None;
    };
    // `Ok(None)` is a dense (bf16) tier and `Err` an unreadable marker; both are "not the measured
    // artifact" and both fail closed here.
    (mlx_gen::quant::packed_quant_bits_at(&root.join("transformer"))
        .ok()
        .flatten()
        == Some(CALIBRATED_QUANT_BITS))
    .then_some(MEMORY_CALIBRATION_FINGERPRINT)
}

/// The packed width of the one measured tier. Part of `production_calibration_fingerprint`'s key.
pub const CALIBRATED_QUANT_BITS: i32 = 4;

/// The production contract, with filesystem-backed asset facts.
pub fn contract_for(
    provider_id: &str,
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    let calibration = production_calibration_fingerprint(provider_id, spec)
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape));
    build_contract(
        provider_id,
        spec,
        structurally_streamable(spec),
        calibration,
        crate::model::component_footprint(spec)?,
        resident_overlay_components(spec)?,
    )
}

/// Latent channels the FLUX.1 autoencoder Chroma reuses produces, and pixels per latent unit on
/// each spatial axis. Chroma loads that VAE through `mlx_gen_flux::load_vae`, so these describe the
/// same four-stage image autoencoder FLUX.1 ships.
const LATENT_CHANNELS: u32 = 16;
const VAE_SPATIAL_SCALE: u32 = 8;
/// The 2x2 latent packing applied before the DiT: `in_channels` 64 is exactly
/// `LATENT_CHANNELS * LATENT_PATCH_SIZE²`, which the tests pin against the config.
const LATENT_PATCH_SIZE: u32 = 2;

/// Architecture axes shared by all three registered Chroma routes (epic SC-22657, E2).
///
/// [`ChromaTransformerConfig`](crate::config::ChromaTransformerConfig) is this crate's mirror of the
/// reference `transformer/config.json` and is identical across HD, Base and Flash — only the weights
/// and the sampling profile differ — so one derivation serves all three.
///
/// `transformer_blocks` is the **sum** of the joint and single stacks (19 + 38): the denoiser
/// traverses both on every step. `vae_temporal_scale` stays `None` — the FLUX.1 image autoencoder
/// has no temporal axis, and a structurally absent axis is declared absent, never zero.
fn architecture_facts() -> mlx_gen::gen_core::MemoryArchitectureFacts {
    let dit = crate::config::ChromaTransformerConfig::default();
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(dit.num_attention_heads),
        head_dim: mlx_gen::architecture_facts::axis(dit.attention_head_dim),
        transformer_blocks: mlx_gen::architecture_facts::axis(
            dit.num_layers.saturating_add(dit.num_single_layers),
        ),
        patch_size: mlx_gen::architecture_facts::axis(LATENT_PATCH_SIZE),
        latent_channels: mlx_gen::architecture_facts::axis(LATENT_CHANNELS),
        vae_spatial_scale: mlx_gen::architecture_facts::axis(VAE_SPATIAL_SCALE),
        vae_temporal_scale: None,
        // The DiT runs f32 activations over its bf16 (or quantized) weights (`transformer.rs`).
        activation_dtype_width: Some(mlx_gen::architecture_facts::FLOAT32_ACTIVATION_WIDTH),
    }
}

/// Declaration-equivalent, zero-filesystem contract used only by registry conformance.
#[doc(hidden)]
pub fn weights_free_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    let route = variant_for(provider_id)?.id().replace('_', "-");
    build_contract(
        provider_id,
        spec,
        structurally_streamable(spec),
        Some(MemoryCalibrationIdentity::new(
            format!("{STATIC_CALIBRATION}-{route}"),
            spec.load_shape,
        )),
        Default::default(),
        // No overlays either: sizing one means opening its checkpoint, and this path exists exactly
        // to produce the declaration without touching a weight file.
        Vec::new(),
    )
}

/// Selector-aware finite-surface resolver. The selector's tier is the explicit resolved artifact
/// tier; downstream consumers never need to reinterpret `LoadSpec::quantize` as a packed-source
/// fact. Production continues to call [`contract_for`] and inspect the real source marker.
#[doc(hidden)]
pub fn weights_free_surface_contract(
    provider_id: &str,
    surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    let route = variant_for(provider_id)?.id().replace('_', "-");
    build_contract(
        provider_id,
        &surface.spec,
        weights_free_structurally_streamable(surface),
        Some(MemoryCalibrationIdentity::new(
            format!("{STATIC_CALIBRATION}-{route}"),
            surface.spec.load_shape,
        )),
        Default::default(),
        Vec::new(),
    )
}

/// Numeric tier the loaded Chroma artifact actually carries. Turnkey Chroma tiers deliberately keep
/// `spec.quantize == None` so their dense T5 tower is never repacked; the transformer marker is the
/// authoritative q4/q8 identity used by both calibration and decode-quality admission.
fn loaded_tier(spec: &LoadSpec) -> MemoryNumericTier {
    let packed = match &spec.weights {
        WeightsSource::Dir(root) if spec.quantize.is_none() => {
            mlx_gen::quant::packed_quant_bits_at(&root.join("transformer"))
                .ok()
                .flatten()
                .and_then(|bits| match bits {
                    4 => Some(mlx_gen::Quant::Q4),
                    8 => Some(mlx_gen::Quant::Q8),
                    _ => None,
                })
        }
        _ => None,
    };
    MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize.or(packed),
        component_precision_floors: &[],
    }
}

pub(crate) fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        let (expected_mode, expected_references) = route_mode_and_references(spec);
        if context.mode != expected_mode
            || context.geometry.reference_count != expected_references
            || context.overlay != route_overlay(spec)
        {
            return Err(CoreError::Unsupported(format!(
                "{}: memory route does not match the loaded mode/overlay",
                contract.provider_id
            )));
        }
        if context.use_pid && spec.pid.is_none() {
            return Err(CoreError::Unsupported(format!(
                "{}: PiD route requested without a loaded PiD overlay",
                contract.provider_id
            )));
        }
        if contract.engages_selection(&context.selection, MemoryStrategy::BoundedDecode)
            && contract
                .capability(MemoryStrategy::BoundedDecode)
                .is_none_or(|capability| capability.parameters.decode_geometry_policies.is_empty())
        {
            decode_routes(&contract.provider_id)?
                .validate(
                    context.use_pid,
                    context.selection.parameters.decode_tile_edge,
                    context.selection.parameters.decode_overlap,
                )
                .map_err(CoreError::Unsupported)?;
        }
        if contract.engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        ) {
            if !contract.lifecycle.transformer_window_materialization || !context.has_phases {
                return Err(CoreError::Unsupported(format!(
                    "{}: bounded transformer residency requires DeferredMaterialization with \
                     staged residency engaged in the same request",
                    contract.provider_id
                )));
            }
            let component = context.selection.parameters.window_component();
            if !TRANSFORMER_WINDOW_COMPONENTS.contains(&component) {
                return Err(CoreError::Unsupported(format!(
                    "{}: transformer window component {component:?} is outside the published \
                     domain {TRANSFORMER_WINDOW_COMPONENTS:?}",
                    contract.provider_id
                )));
            }
        }
        Ok(())
    };
    standard_memory_strategy_safety_check(
        contract,
        context,
        Some(loaded_tier(spec)),
        Some(&route_gate),
    )
}

pub(crate) fn registered_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> mlx_gen::gen_core::Result<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    if contract
        .capability(strategy)
        .is_none_or(|capability| capability.support != MemoryStrategySupport::Implemented)
    {
        return Ok(Vec::new());
    }
    let (mode, reference_count) = route_mode_and_references(spec);
    let context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        loaded_tier(spec),
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode,
            reference_count,
            use_pid: false,
            // Request-scoped residency exposes phase boundaries under either load-time default.
            has_phases: true,
            overlay: route_overlay(spec),
        },
    )?;
    Ok(vec![mlx_gen::gen_core::MemoryBehaviorFixture::new(context)])
}

fn begin_request_with_cleanup(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    let provider = provider_static(&contract.provider_id)?;
    let routes = decode_routes(provider)?;
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        window_stack_depth(),
        move |use_pid, edge, overlap| {
            routes
                .validate(use_pid, Some(edge), Some(overlap))
                .map_err(CoreError::Unsupported)
        },
    )?;
    config.load_shape = context.load_shape;
    mlx_gen::request_scope::authorize_selected_geometry_decode(&mut config, contract, context)?;
    config.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE);
    config.transformer_window = contract
        .engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .then_some(context.selection.parameters.transformer_window_size)
        .flatten();
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    )))
}

pub(crate) fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub(crate) fn begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

// ── Request-local resolution of the selected parameters ──────────────────────────────────────────

/// Rung 3's plan. An unselected request returns the exact historical unbounded/uncancellable plan.
pub(crate) fn attention_plan(req: &GenerationRequest) -> AttentionPlan<'_> {
    match req.memory {
        Some(memory) if memory.chunk_attention => {
            AttentionPlan::budgeted(AttentionBudget::CONSTRAINED).with_cancel(&req.cancel)
        }
        _ => AttentionPlan::UNBOUNDED,
    }
}

/// Rung 3's parameter domain check, on the same layer as its rung-2 and rung-4 siblings.
pub(crate) fn validate_attention_chunk(
    req: &GenerationRequest,
    provider_id: &str,
) -> mlx_gen::Result<()> {
    let Some(memory) = req.memory.filter(|memory| memory.chunk_attention) else {
        return Ok(());
    };
    match memory.attention_chunk_size {
        None | Some(ATTENTION_CHUNK_SIZE) => Ok(()),
        Some(other) => Err(mlx_gen::Error::Unsupported(format!(
            "{provider_id}: attention chunk size {other} is outside the published production \
             domain [{ATTENTION_CHUNK_SIZE}]"
        ))),
    }
}

/// Rung 4's window, or `None` when unselected. Validates the cadence and the component scope
/// against the published domains, on the production path.
pub(crate) fn transformer_window(
    req: &GenerationRequest,
    provider_id: &str,
) -> mlx_gen::Result<Option<usize>> {
    let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
        return Ok(None);
    };
    if !memory.stage_residency {
        return Err(mlx_gen::Error::Unsupported(format!(
            "{provider_id}: bounded transformer residency requires staged residency in the same \
             request"
        )));
    }
    if req.use_pid {
        return Err(mlx_gen::Error::Unsupported(format!(
            "{provider_id}: bounded transformer residency is not implemented for the PiD decode \
             overlay"
        )));
    }
    let component = memory
        .transformer_window_component
        .unwrap_or(TRANSFORMER_WINDOW_COMPONENT);
    if !TRANSFORMER_WINDOW_COMPONENTS.contains(&component) {
        return Err(mlx_gen::Error::Unsupported(format!(
            "{provider_id}: transformer window component {component:?} is outside the published \
             production domain {TRANSFORMER_WINDOW_COMPONENTS:?}"
        )));
    }
    let size = memory
        .transformer_window_size
        .unwrap_or(TRANSFORMER_WINDOW_SIZE);
    if !TRANSFORMER_WINDOW_SIZES.contains(&size) {
        return Err(mlx_gen::Error::Unsupported(format!(
            "{provider_id}: transformer window {size} is outside the published production domain \
             {TRANSFORMER_WINDOW_SIZES:?}"
        )));
    }
    Ok(Some(size as usize))
}

/// Rung 2's plan. Unselected requests return `None`, keeping the historical one-pass decode exactly
/// intact. PiD is handled before this plan at the decode call site and never inherits native VAE
/// geometry.
#[cfg(test)]
pub(crate) fn decode_tiling(
    req: &GenerationRequest,
    provider_id: &str,
) -> mlx_gen::Result<Option<TilingConfig>> {
    let Some(memory) = req.memory.filter(|memory| memory.tile_vae_decode) else {
        return Ok(None);
    };
    if req.cancel.is_cancelled() {
        return Err(mlx_gen::Error::Canceled);
    }
    let edge = memory.decode_tile_edge.unwrap_or(DECODE_TILE_EDGE);
    let overlap = memory.decode_overlap.unwrap_or(DECODE_OVERLAP);
    // The same checked route set admission used, so an out-of-domain value is refused on the
    // production render path and not only at selection time.
    decode_routes(provider_id)
        .map_err(|error| mlx_gen::Error::Unsupported(error.to_string()))?
        .validate(req.use_pid, Some(edge), Some(overlap))
        .map_err(mlx_gen::Error::Unsupported)?;
    Ok(Some(TilingConfig::spatial_only(
        edge as i32,
        overlap as i32,
    )))
}

/// Production rung-2 resolution. The mechanism helper above remains available to correctness-only
/// tests, but a real request may tile only under an exact sealed geometry row selected and armed by
/// its request scope.
pub(crate) fn decode_tiling_for_contract(
    req: &GenerationRequest,
    provider_id: &'static str,
    contract: &MemoryProviderContract,
) -> mlx_gen::Result<Option<TilingConfig>> {
    let Some(memory) = req.memory.filter(|memory| memory.tile_vae_decode) else {
        mlx_gen::diagnostics::record_geometry_decode_decision(
            mlx_gen::diagnostics::DecodePolicyDisposition::Unchanged,
            mlx_gen::diagnostics::DecodePathDisposition::Dense,
            None,
        );
        return Ok(None);
    };
    if req.cancel.is_cancelled() {
        return Err(mlx_gen::Error::Canceled);
    }
    let has_quality_table = contract
        .capability(MemoryStrategy::BoundedDecode)
        .is_some_and(|capability| !capability.parameters.decode_geometry_policies.is_empty());
    if !has_quality_table {
        return Err(mlx_gen::Error::Unsupported(format!(
            "{provider_id}: bounded decode is withheld until an exact production-latent geometry-quality row is packaged for this loaded route"
        )));
    }
    let edge = memory.decode_tile_edge.ok_or_else(|| {
        mlx_gen::Error::Unsupported(format!(
            "{provider_id}: geometry-aware bounded decode requires an exact tile edge"
        ))
    })?;
    let overlap = memory.decode_overlap.ok_or_else(|| {
        mlx_gen::Error::Unsupported(format!(
            "{provider_id}: geometry-aware bounded decode requires an exact overlap"
        ))
    })?;
    let evidence = mlx_gen::request_scope::validate_geometry_decode_authorization(
        provider_id,
        req,
        edge,
        overlap,
    )?;
    mlx_gen::diagnostics::record_geometry_decode_decision(
        mlx_gen::diagnostics::DecodePolicyDisposition::GeometryTiled,
        mlx_gen::diagnostics::DecodePathDisposition::Tiled,
        Some(&evidence),
    );
    Ok(Some(TilingConfig::spatial_only(
        edge as i32,
        overlap as i32,
    )))
}

/// Request-local conformance fault at a completed physical phase boundary (SC-15449). The shared
/// request floor authorizes this pair; production requests leave both fields unset.
pub(crate) fn calibration_fault(
    req: &GenerationRequest,
    phase: MemoryPhase,
    provider_id: &str,
) -> mlx_gen::Result<()> {
    match req.memory {
        Some(memory)
            if memory.calibration_fault_harness_authorized
                && memory.calibration_error_phase == Some(phase) =>
        {
            Err(mlx_gen::Error::Msg(format!(
                "{provider_id}: injected memory-strategy calibration error at {phase:?}"
            )))
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::GenerationMemory;
    use mlx_gen::{AdapterKind, AdapterSpec, Quant};

    fn sequential_spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
    }

    fn streamable_spec() -> LoadSpec {
        sequential_spec().with_load_shape(LoadShape::DeferredMaterialization)
    }

    /// A snapshot root whose `transformer/config.json` declares a packed width — the discriminant
    /// `production_calibration_fingerprint` keys the measured tier on. `None` writes a dense tier
    /// (no `quantization` marker), which is what a `bf16` tier looks like on disk.
    fn tier_root(tmp: &tempfile::TempDir, tag: &str, bits: Option<i32>) -> std::path::PathBuf {
        let root = tmp.path().join(format!(
            "chroma-tier-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("transformer")).unwrap();
        let config = match bits {
            Some(bits) => format!("{{\"quantization\": {{\"bits\": {bits}, \"group_size\": 64}}}}"),
            None => "{}".to_owned(),
        };
        std::fs::write(root.join("transformer/config.json"), config).unwrap();
        root
    }

    fn tier_spec(root: &std::path::Path) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
            .with_offload_policy(OffloadPolicy::Sequential)
    }

    /// AC (SC-22662): every registered Chroma route publishes the axes of the one DiT they share,
    /// derived from this crate's own config constants, and passes the shared facts check.
    #[test]
    fn architecture_facts_follow_the_crate_transformer_config() {
        let spec = sequential_spec();
        for provider in [
            crate::CHROMA1_HD_ID,
            crate::CHROMA1_BASE_ID,
            crate::CHROMA1_FLASH_ID,
        ] {
            let contract = weights_free_contract(provider, &spec).unwrap();
            assert_eq!(
                contract.architecture_facts,
                mlx_gen::gen_core::MemoryArchitectureFacts {
                    attention_heads: Some(24),
                    head_dim: Some(128),
                    // 19 joint + 38 single blocks.
                    transformer_blocks: Some(57),
                    patch_size: Some(2),
                    latent_channels: Some(16),
                    vae_spatial_scale: Some(8),
                    vae_temporal_scale: None,
                    activation_dtype_width: Some(4),
                },
                "{provider} architecture facts"
            );
            assert!(contract.architecture_facts.has_declared_architecture_axis());
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
    }

    /// The published latent geometry is the config's, not a second copy of it: the DiT's packed
    /// `in_channels` IS `latent_channels * patch²`.
    #[test]
    fn the_published_latent_geometry_reconstructs_the_dit_input_width() {
        let dit = crate::config::ChromaTransformerConfig::default();
        assert_eq!(
            LATENT_CHANNELS * LATENT_PATCH_SIZE * LATENT_PATCH_SIZE,
            dit.in_channels as u32
        );
        // `VAE_SPATIAL_SCALE`'s doc claims the x8 of the four-stage FLUX.1 autoencoder. That scale
        // is not a free constant: `SIZE_MULTIPLE` is what the sampler rounds a request to, and it
        // is exactly `vae_scale * patch`. Deriving the pin from those two keeps the published axis
        // tied to the geometry the sampler already enforces.
        assert_eq!(
            VAE_SPATIAL_SCALE,
            crate::model::SIZE_MULTIPLE / LATENT_PATCH_SIZE
        );
    }

    #[test]
    fn every_entry_publishes_the_same_ladder_and_is_internally_coherent() {
        let tmp = tempfile::tempdir().unwrap();
        for provider in [
            crate::CHROMA1_HD_ID,
            crate::CHROMA1_BASE_ID,
            crate::CHROMA1_FLASH_ID,
        ] {
            let contract = weights_free_contract(provider, &streamable_spec()).unwrap();
            assert!(
                contract.conformance_errors().is_empty(),
                "{provider}: {:?}",
                contract.conformance_errors()
            );
            assert_eq!(contract.asset_facts, Default::default());
            for (strategy, expected) in [
                (MemoryStrategy::Resident, true),
                (MemoryStrategy::StagedResidency, true),
                (MemoryStrategy::BoundedDecode, DECODE_SUPPORT),
                (MemoryStrategy::BoundedAttention, ATTENTION_SUPPORT),
                (MemoryStrategy::BoundedTransformerResidency, WINDOW_SUPPORT),
            ] {
                let support = &contract.capability(strategy).unwrap().support;
                assert_eq!(
                    matches!(support, MemoryStrategySupport::Implemented),
                    expected,
                    "{provider} {strategy:?}"
                );
            }
        }
        // Sibling entries must not be Verified by sharing code: only the measured entry, at the
        // measured tier, carries a production calibration identity.
        let measured = tier_root(&tmp, "measured", Some(CALIBRATED_QUANT_BITS));
        assert_eq!(
            production_calibration_fingerprint(crate::CHROMA1_BASE_ID, &tier_spec(&measured)),
            Some(MEMORY_CALIBRATION_FINGERPRINT)
        );
        for provider in [crate::CHROMA1_HD_ID, crate::CHROMA1_FLASH_ID] {
            assert!(production_calibration_fingerprint(provider, &tier_spec(&measured)).is_none());
        }
        std::fs::remove_dir_all(measured).ok();
    }

    #[test]
    fn the_calibration_key_is_exact_and_every_other_axis_fails_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let measured = tier_root(&tmp, "exact", Some(CALIBRATED_QUANT_BITS));
        let exact = tier_spec(&measured);
        assert_eq!(
            production_calibration_fingerprint(crate::CHROMA1_BASE_ID, &exact),
            Some(MEMORY_CALIBRATION_FINGERPRINT),
            "the measured tier must be calibrated, else every negative below is vacuous"
        );

        // **The tier is part of the key.** All three shipped tiers are pre-packed, so all three load
        // with `quantize == None`: without this discriminant a q8 or bf16 request would inherit the
        // q4-measured key and could select an optimized fit on evidence never taken for it.
        for (tag, bits) in [("q8", Some(8)), ("bf16", None)] {
            let other = tier_root(&tmp, tag, bits);
            assert!(
                production_calibration_fingerprint(crate::CHROMA1_BASE_ID, &tier_spec(&other))
                    .is_none(),
                "the {tag} tier must not inherit the q4-measured calibration"
            );
            std::fs::remove_dir_all(other).ok();
        }
        // An unreadable or absent tier marker fails closed rather than defaulting to the measured
        // one — `/nonexistent` has no `transformer/config.json` at all.
        assert!(
            production_calibration_fingerprint(crate::CHROMA1_BASE_ID, &sequential_spec())
                .is_none()
        );

        for changed in [
            exact.clone().with_quant(Quant::Q4),
            exact.clone().with_offload_policy(OffloadPolicy::Resident),
            {
                let mut spec = exact.clone();
                spec.precision = mlx_gen::Precision::Fp32;
                spec
            },
            {
                let mut spec = exact.clone();
                spec.adapters.push(AdapterSpec::new(
                    "/adapter.safetensors".into(),
                    1.0,
                    AdapterKind::Lora,
                ));
                spec
            },
            exact.clone().with_pid(
                WeightsSource::File("/pid.safetensors".into()),
                WeightsSource::Dir("/gemma".into()),
            ),
        ] {
            assert!(
                production_calibration_fingerprint(crate::CHROMA1_BASE_ID, &changed).is_none(),
                "a changed axis must not inherit the calibrated key"
            );
        }
        std::fs::remove_dir_all(measured).ok();
    }

    /// A one-tensor `.safetensors` file, so an overlay can be priced without a real checkpoint.
    fn tiny_safetensors(path: &std::path::Path, elements: usize) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let values: Vec<f32> = (0..elements).map(|i| i as f32).collect();
        mlx_rs::Array::save_safetensors(
            vec![("w", &mlx_rs::Array::from_slice(&values, &[elements as i32]))],
            None,
            path,
        )
        .unwrap();
    }

    /// **The sc-15839 class: an advertised overlay priced at zero (review of PR #496).**
    ///
    /// The shared validator only cross-checks the overlay legs `if overlay_bytes > 0`, so a zero
    /// passes silently — which is exactly how a rung-1 or rung-4 selection taken for a PiD load can
    /// under-predict peak by the whole student plus its caption encoder.
    #[test]
    fn a_resident_pid_overlay_is_priced_and_a_sequential_one_is_not() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let student = root.join("pid/student.safetensors");
        let gemma = root.join("gemma");
        tiny_safetensors(&student, 64);
        tiny_safetensors(&gemma.join("model.safetensors"), 32);
        let expected = 4 * (64 + 32);

        let with_pid = |policy: OffloadPolicy| {
            LoadSpec::new(WeightsSource::Dir(root.clone()))
                .with_offload_policy(policy)
                .with_pid(
                    WeightsSource::File(student.clone()),
                    WeightsSource::Dir(gemma.clone()),
                )
        };

        // Resident: `ensure_warm_locked` calls `load_heavy(true, …)` unconditionally, so the student
        // is resident whether or not the request asked for it — and must be priced.
        let resident = resident_overlay_components(&with_pid(OffloadPolicy::Resident)).unwrap();
        assert_eq!(resident.len(), 1, "the PiD overlay must be declared");
        assert_eq!(resident[0].id, PID_COMPONENT_ID);
        assert!(resident[0].kind.is_auxiliary());
        assert_eq!(resident[0].resident_bytes, expected);

        // Sequential: `build_residency` threads `req.use_pid` per request, so a non-PiD generate
        // never materializes it. The carve-out is scoped to exactly that policy.
        assert!(
            resident_overlay_components(&with_pid(OffloadPolicy::Sequential))
                .unwrap()
                .is_empty(),
            "a Sequential PiD overlay is request-selected and must not be priced unconditionally"
        );

        // And the contract carries it: `overlay_bytes` matches, the component axis is declared, and
        // the shared conformance validator (whose overlay legs only run when it is non-zero) passes.
        let contract = contract_for(crate::CHROMA1_BASE_ID, &with_pid(OffloadPolicy::Resident))
            .expect("contract");
        assert_eq!(contract.asset_facts.overlay_bytes, expected);
        assert_eq!(contract.auxiliary_resident_bytes(), expected);
        assert!(
            contract.conformance_errors().is_empty(),
            "{:?}",
            contract.conformance_errors()
        );

        // An adapter stack is priced too, and a stack that cannot be sized fails closed rather than
        // declaring a zero.
        let adapter = root.join("lora.safetensors");
        tiny_safetensors(&adapter, 16);
        let mut adapted = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(OffloadPolicy::Sequential);
        adapted
            .adapters
            .push(AdapterSpec::new(adapter, 1.0, AdapterKind::Lora));
        let priced = resident_overlay_components(&adapted).unwrap();
        assert_eq!(priced.len(), 1);
        assert_eq!(priced[0].id, ADAPTER_COMPONENT_ID);
        // The shared helper prices an additive stack at its on-disk file size (payload + header),
        // which is what stays resident; asserted as a bound rather than an exact figure so the
        // header's width is not pinned here.
        assert!(priced[0].resident_bytes >= 4 * 16);

        let mut unsizable = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(OffloadPolicy::Sequential);
        unsizable.adapters.push(AdapterSpec::new(
            root.join("absent.safetensors"),
            1.0,
            AdapterKind::Lora,
        ));
        assert!(
            resident_overlay_components(&unsizable).is_err(),
            "an unsizable adapter stack must fail closed"
        );

        // The clean base route declares no overlay at all, which is what keeps the ordinary
        // contract on the cheaper `PhaseEnvelope` formula.
        assert!(resident_overlay_components(&sequential_spec())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn every_overlay_withdraws_the_quality_sensitive_rungs() {
        let cases = [
            {
                let mut spec = streamable_spec();
                spec.adapters.push(AdapterSpec::new(
                    "/adapter.safetensors".into(),
                    1.0,
                    AdapterKind::Lora,
                ));
                spec
            },
            {
                let mut spec = streamable_spec();
                spec.control = Some(WeightsSource::File("/control.safetensors".into()));
                spec
            },
            {
                let mut spec = streamable_spec();
                spec.extra_controls
                    .push(WeightsSource::File("/control-2.safetensors".into()));
                spec
            },
            {
                let mut spec = streamable_spec();
                spec.ip_adapter = Some(WeightsSource::Dir("/ip".into()));
                spec
            },
            {
                let mut spec = streamable_spec();
                spec.identity = Some(Default::default());
                spec
            },
            {
                let mut spec = streamable_spec();
                spec.text_encoder = Some(WeightsSource::Dir("/external-text".into()));
                spec
            },
            streamable_spec().with_pid(
                WeightsSource::File("/pid.safetensors".into()),
                WeightsSource::Dir("/gemma".into()),
            ),
        ];
        for spec in cases {
            assert!(
                !structurally_streamable(&spec),
                "production predicate must reject {:?}",
                route_overlay(&spec)
            );
            let contract = weights_free_contract(crate::CHROMA1_BASE_ID, &spec).unwrap();
            assert!(contract.conformance_errors().is_empty());
            for strategy in [
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ] {
                assert_eq!(
                    contract.capability(strategy).unwrap().support,
                    MemoryStrategySupport::Missing,
                    "{strategy:?} must be withdrawn for {:?}",
                    route_overlay(&spec)
                );
            }
            assert!(!contract.lifecycle.decode_tiling);
            assert!(!contract.lifecycle.attention_chunking);
            assert!(!contract.lifecycle.transformer_window_materialization);
        }
    }

    #[test]
    fn production_rung_four_needs_deferred_directory_clean_route_and_no_requantization() {
        assert!(structurally_streamable(&streamable_spec()));
        assert!(
            structurally_streamable(
                &streamable_spec().with_offload_policy(OffloadPolicy::Resident)
            ),
            "materialization shape is independent from phase offload policy"
        );
        for spec in [
            // Eager materialization: no reopenable stream.
            sequential_spec(),
            // A dense source the loader would re-quantize per window.
            streamable_spec().with_quant(Quant::Q4),
            // A single-file checkpoint has no component tree to reopen.
            LoadSpec::new(WeightsSource::File("/one.safetensors".into()))
                .with_offload_policy(OffloadPolicy::Sequential)
                .with_load_shape(LoadShape::DeferredMaterialization),
        ] {
            assert!(!structurally_streamable(&spec));
        }
    }

    #[test]
    fn production_packed_and_dense_artifact_tiers_share_the_deferred_source_predicate() {
        let tmp = tempfile::tempdir().unwrap();
        for (tag, bits) in [("bf16", None), ("q4", Some(4)), ("q8", Some(8))] {
            let root = tier_root(&tmp, tag, bits);
            for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
                let spec = tier_spec(&root)
                    .with_offload_policy(policy)
                    .with_load_shape(LoadShape::DeferredMaterialization);
                assert!(
                    structurally_streamable(&spec),
                    "{tag} {policy:?} resolved artifact"
                );
                assert!(
                    !structurally_streamable(&spec.clone().with_quant(Quant::Q4)),
                    "a real load-time quantization request remains distinct from the resolved artifact tier"
                );
                let mut external_te = spec;
                external_te.text_encoder = Some(WeightsSource::Dir("/external-text".into()));
                assert!(
                    !structurally_streamable(&external_te),
                    "external text encoder must remain fail-closed"
                );
            }
            std::fs::remove_dir_all(root).ok();
        }
    }

    #[test]
    fn weights_free_resolved_tiers_expose_both_deferred_load_policies() {
        for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
            for tier in [
                mlx_gen::gen_core::MemoryContractSurfaceTier::Bf16,
                mlx_gen::gen_core::MemoryContractSurfaceTier::Q4,
                mlx_gen::gen_core::MemoryContractSurfaceTier::Q8,
            ] {
                let surface = mlx_gen::gen_core::mlx_memory_contract_surface_specs()
                    .into_iter()
                    .find(|surface| {
                        surface.selector.tier == tier
                            && surface.selector.offload_policy == policy
                            && surface.selector.load_shape == LoadShape::DeferredMaterialization
                    })
                    .unwrap();
                let contract =
                    weights_free_surface_contract(crate::CHROMA1_BASE_ID, &surface).unwrap();
                assert_eq!(
                    contract
                        .capability(MemoryStrategy::BoundedTransformerResidency)
                        .unwrap()
                        .support,
                    MemoryStrategySupport::Implemented,
                    "{policy:?} {tier:?} is a resolved prepacked surface"
                );
                assert!(contract
                    .requires(MemoryStrategy::BoundedTransformerResidency)
                    .any(|prerequisite| matches!(
                        prerequisite,
                        MemoryStrategyPrerequisite::Rung {
                            rung: MemoryStrategy::StagedResidency,
                            scope: MemoryPrerequisiteScope::EngagedInSameRequest,
                        }
                    )));

                let eager = mlx_gen::gen_core::mlx_memory_contract_surface_specs()
                    .into_iter()
                    .find(|candidate| {
                        candidate.selector.tier == tier
                            && candidate.selector.offload_policy == policy
                            && candidate.selector.load_shape == LoadShape::EagerMaterialization
                    })
                    .unwrap();
                assert_eq!(
                    weights_free_surface_contract(crate::CHROMA1_BASE_ID, &eager)
                        .unwrap()
                        .capability(MemoryStrategy::BoundedTransformerResidency)
                        .unwrap()
                        .support,
                    MemoryStrategySupport::Missing
                );
            }
        }
    }

    #[test]
    fn resident_deferred_rung_four_selection_engages_request_scoped_staging() {
        let surface = mlx_gen::gen_core::mlx_memory_contract_surface_specs()
            .into_iter()
            .find(|surface| {
                surface.selector.tier == mlx_gen::gen_core::MemoryContractSurfaceTier::Q4
                    && surface.selector.offload_policy == OffloadPolicy::Resident
                    && surface.selector.load_shape == LoadShape::DeferredMaterialization
            })
            .unwrap();
        let contract = weights_free_surface_contract(crate::CHROMA1_BASE_ID, &surface).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        assert!(contract.lifecycle.synchronized_phase_release);
        assert_eq!(
            contract.resident_request_memory,
            ResidentRequestMemory::ExplicitResident
        );

        let mut fixture = registered_fixture(
            &surface.spec,
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .remove(0);
        contract
            .validate_selection(&fixture.context.selection)
            .unwrap();
        assert!(matches!(
            safety_check(&surface.spec, &contract, &fixture.context),
            MemorySafetyDecision::Accept
        ));
        let mut scope = registered_begin_request(&surface.spec, &contract, &fixture.context)
            .unwrap()
            .unwrap();
        scope.configure_request(&mut fixture.request).unwrap();
        let memory = fixture.request.memory.expect("rung 4 request memory");
        assert!(memory.stage_residency);
        assert!(memory.stream_transformer_blocks);
    }

    #[test]
    fn the_request_scope_admits_only_the_published_parameter_domains() {
        let spec = streamable_spec();
        let contract = weights_free_contract(crate::CHROMA1_BASE_ID, &spec).unwrap();

        // Rung 2 is Missing, so it publishes no behavioral fixture at all — a caller cannot reach a
        // bounded decode through the contract, and the request scope's decode validator refuses
        // every geometry unconditionally.
        assert!(
            registered_fixture(&spec, &contract, MemoryStrategy::BoundedDecode)
                .unwrap()
                .is_empty(),
            "a Missing rung must publish no behavioral fixture"
        );

        assert!(
            registered_fixture(&spec, &contract, MemoryStrategy::BoundedAttention)
                .unwrap()
                .is_empty(),
            "a Missing rung must publish no behavioral fixture"
        );

        let mut fixture = registered_fixture(
            &spec,
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .remove(0);
        let mut scope = registered_begin_request(&spec, &contract, &fixture.context)
            .unwrap()
            .unwrap();
        scope.configure_request(&mut fixture.request).unwrap();
        let memory = fixture.request.memory.expect("rung 4 request memory");
        assert!(memory.stage_residency, "rung 4 must engage rung 1");
        assert!(memory.stream_transformer_blocks);
        assert_eq!(
            memory.transformer_window_component,
            Some(TRANSFORMER_WINDOW_COMPONENT)
        );
        let window = memory.transformer_window_size.expect("selected cadence");
        scope.materialize_transformer_window(0, window).unwrap();
        assert!(scope.materialize_transformer_window(1, window).is_err() || window == 1);
    }

    #[test]
    fn native_and_pid_decode_routes_are_disjoint_and_checked() {
        let routes = decode_routes(crate::CHROMA1_BASE_ID).unwrap();
        assert_eq!(routes.native_edges(), DECODE_TILE_EDGES);
        routes
            .validate(false, Some(DECODE_TILE_EDGE), Some(DECODE_OVERLAP))
            .unwrap();
        // A swept-but-rejected edge is refused by the same checked route set the production render
        // path uses — the rejection is enforced, not documented.
        for edge in DECODE_TILE_EDGES_SWEPT
            .iter()
            .filter(|edge| !DECODE_TILE_EDGES.contains(edge))
        {
            assert!(routes
                .validate(false, Some(*edge), Some(DECODE_OVERLAP))
                .is_err());
        }
        for overlap in DECODE_OVERLAPS_SWEPT
            .iter()
            .filter(|overlap| **overlap != DECODE_OVERLAP)
        {
            assert!(routes
                .validate(false, Some(DECODE_TILE_EDGE), Some(*overlap))
                .is_err());
        }
        assert!(routes
            .validate(true, Some(DECODE_TILE_EDGE), Some(DECODE_OVERLAP))
            .is_err());
        let pid_edge = mlx_gen_pid::DecodeRoutes::pid_edges()[0];
        let pid_overlap = mlx_gen_pid::DecodeRoutes::pid_overlap();
        routes
            .validate(true, Some(pid_edge), Some(pid_overlap))
            .unwrap();
        assert!(routes
            .validate(false, Some(pid_edge), Some(pid_overlap))
            .is_err());
    }

    #[test]
    fn request_local_resolvers_are_exact_and_default_to_the_historical_path() {
        let plain = GenerationRequest::default();
        assert_eq!(attention_plan(&plain).budget, AttentionBudget::UNBOUNDED);
        assert!(attention_plan(&plain).cancel.is_none());
        assert!(decode_tiling(&plain, crate::CHROMA1_BASE_ID)
            .unwrap()
            .is_none());
        assert!(transformer_window(&plain, crate::CHROMA1_BASE_ID)
            .unwrap()
            .is_none());
        validate_attention_chunk(&plain, crate::CHROMA1_BASE_ID).unwrap();

        let chunked = GenerationRequest {
            memory: Some(GenerationMemory {
                chunk_attention: true,
                attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            attention_plan(&chunked).budget,
            AttentionBudget::CONSTRAINED
        );
        assert!(attention_plan(&chunked).cancel.is_some());
        validate_attention_chunk(&chunked, crate::CHROMA1_BASE_ID).unwrap();
        let out_of_domain = GenerationRequest {
            memory: Some(GenerationMemory {
                chunk_attention: true,
                attention_chunk_size: Some(ATTENTION_CHUNK_SIZE - 1),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(validate_attention_chunk(&out_of_domain, crate::CHROMA1_BASE_ID).is_err());

        let tiled = GenerationRequest {
            memory: Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(DECODE_TILE_EDGE),
                decode_overlap: Some(DECODE_OVERLAP),
                ..Default::default()
            }),
            ..Default::default()
        };
        let plan = decode_tiling(&tiled, crate::CHROMA1_BASE_ID)
            .unwrap()
            .unwrap();
        let spatial = plan.spatial.expect("spatial-only plan");
        assert_eq!(spatial.tile_px, DECODE_TILE_EDGE as i32);
        assert_eq!(spatial.overlap_px, DECODE_OVERLAP as i32);
        assert!(plan.temporal.is_none());
        let canceled = tiled.clone();
        canceled.cancel.cancel();
        assert!(matches!(
            decode_tiling(&canceled, crate::CHROMA1_BASE_ID),
            Err(mlx_gen::Error::Canceled)
        ));
        let bad_edge = GenerationRequest {
            memory: Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(DECODE_TILE_EDGE + 1),
                decode_overlap: Some(DECODE_OVERLAP),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(decode_tiling(&bad_edge, crate::CHROMA1_BASE_ID).is_err());

        let windowed = GenerationRequest {
            memory: Some(GenerationMemory {
                stage_residency: true,
                stream_transformer_blocks: true,
                transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
                transformer_window_component: Some(TRANSFORMER_WINDOW_COMPONENT),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            transformer_window(&windowed, crate::CHROMA1_BASE_ID).unwrap(),
            Some(TRANSFORMER_WINDOW_SIZE as usize)
        );
        // Rung 4 without rung 1 in the same request is refused on the production path.
        let unstaged = GenerationRequest {
            memory: Some(GenerationMemory {
                stage_residency: false,
                ..windowed.memory.unwrap()
            }),
            ..Default::default()
        };
        assert!(transformer_window(&unstaged, crate::CHROMA1_BASE_ID).is_err());
        // An out-of-domain cadence is refused on the production path.
        let out_of_domain = GenerationRequest {
            memory: Some(GenerationMemory {
                transformer_window_size: Some(3),
                ..windowed.memory.unwrap()
            }),
            ..Default::default()
        };
        assert!(transformer_window(&out_of_domain, crate::CHROMA1_BASE_ID).is_err());
        // Every published scope is reachable, and each resolves to the same cadence.
        for component in TRANSFORMER_WINDOW_COMPONENTS {
            let req = GenerationRequest {
                memory: Some(GenerationMemory {
                    transformer_window_component: Some(*component),
                    ..windowed.memory.unwrap()
                }),
                ..Default::default()
            };
            assert_eq!(
                transformer_window(&req, crate::CHROMA1_BASE_ID).unwrap(),
                Some(TRANSFORMER_WINDOW_SIZE as usize),
                "published scope {component:?} must be reachable"
            );
        }
    }

    #[test]
    fn the_calibration_fault_is_authorized_phase_exact_and_request_local() {
        let mut memory = GenerationMemory::default();
        memory.authorize_calibration_fault(MemoryPhase::Decode);
        let injected = GenerationRequest {
            memory: Some(memory),
            ..Default::default()
        };
        assert!(calibration_fault(&injected, MemoryPhase::Denoise, crate::CHROMA1_BASE_ID).is_ok());
        let error = calibration_fault(&injected, MemoryPhase::Decode, crate::CHROMA1_BASE_ID)
            .unwrap_err()
            .to_string();
        assert!(error.contains(crate::CHROMA1_BASE_ID));
        assert!(error.contains("Decode"));
        assert!(calibration_fault(
            &GenerationRequest::default(),
            MemoryPhase::Decode,
            crate::CHROMA1_BASE_ID
        )
        .is_ok());
        // An unauthorized phase is inert: the pair is what the shared floor accepts.
        let unauthorized = GenerationRequest {
            memory: Some(GenerationMemory {
                calibration_error_phase: Some(MemoryPhase::Decode),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(
            calibration_fault(&unauthorized, MemoryPhase::Decode, crate::CHROMA1_BASE_ID).is_ok()
        );
    }

    #[test]
    fn the_window_stack_depth_comes_from_the_model_config() {
        let cfg = ChromaTransformerConfig::default();
        assert_eq!(window_stack_depth(), cfg.num_single_layers);
        assert!(window_stack_depth() >= cfg.num_layers);
        assert!(window_stack_depth() >= mlx_gen_flux::T5_BLOCKS);
    }
}
