//! SANA MLX adoption of the shared memory-strategy contract (SC-15449) — **phase 2**, the calibrated
//! ladder.
//!
//! Phase 1 made [`pipeline::decode_tiling`](crate::pipeline::decode_tiling) honour the contract's
//! rung-2 signal: a caller setting [`GenerationMemory::tile_vae_decode`] got a bounded decode instead
//! of being silently ignored. That fixed the *executable* half. This file is the *declared* half —
//! the published edge domain, the measured rejection set, and the calibration identity that keys
//! external evidence to this execution structure.
//!
//! Both SANA ids — `sana_1600m` and `sana_sprint_1600m` — share one Gemma-2 CHI encoder, one Linear
//! DiT trunk, one staged [`Residency`](mlx_gen::Residency) seam and one tiled DC-AE decode, so they
//! share one contract builder. Sprint differs in the scheduler and the guidance axis, neither of
//! which is a memory seam.
//!
//! ## Declared rungs
//!
//! | Rung | Support | Executable seam |
//! |---|---|---|
//! | 0 Resident | Implemented | `Residency::resident` — encoder + DiT + VAE held warm |
//! | 1 Staged residency | Implemented (**load-time**, see below) | `Residency::run_staged` (sc-13571): encode → drop Gemma → denoise → **drop DiT** → decode |
//! | 2 Bounded decode | Implemented | `DcAeDecoder` over the [`DECODE_TILE_EDGES`] ladder (`pipeline::resolve_decode_tiling`) |
//! | 3 Bounded attention | **Missing** | — |
//! | 4 Bounded transformer residency | **Missing** | — |
//!
//! **This is a three-rung ladder, and the two absences are different kinds of absence.**
//!
//! Rung 4 is plainly unimplemented: SANA's trunk has no
//! [`block_residency`](mlx_gen::block_residency) window driver, and nothing here defers block
//! materialization.
//!
//! Rung 3 is unimplemented for a reason worth recording, because it is nearly — but not quite —
//! `StructurallyNotApplicable`. SANA's **self**-attention is ReLU *linear* attention
//! (`SanaLinearAttnProcessor2_0`): it never materializes a `[B, H, Sq, Sk]` score tensor at all, so
//! there is genuinely nothing there for a score budget to bound. That is the whole point of a Linear
//! DiT. But the **cross**-attention to the caption embedding is standard softmax SDPA
//! ([`transformer::CrossAttn`](crate::transformer)) and does build an explicit score matrix before
//! [`softmax_axis`]. So a boundable score tensor exists, which is why this declares `Missing` rather
//! than claiming the architecture has no such component. It is `Missing` and not merely un-prioritized
//! because that score matrix is `[B, H, N, 300]` — [`MAX_SEQUENCE_LENGTH`](crate::MAX_SEQUENCE_LENGTH)
//! caption keys — and bounding a 300-key axis is not where this model's memory is.
//!
//! Declaring them `Missing` rather than inheriting a rung from the z-image column is deliberate.
//! `mlx-gen-z-image` carries all five; **a rung does not transfer between providers any more than it
//! transfers between backends.** Every number below was measured on this provider.
//!
//! ## Rung 1 has no request-scoped lever
//!
//! Staging is gated on [`mlx_gen::Residency::is_sequential`], which comes from the **load-time**
//! [`OffloadPolicy`], so selecting `StagedResidency` on a generator loaded `Resident` yields resident
//! behaviour — the selection is honoured only if the consumer also loaded the provider `Sequential`.
//! [`sana_generation_memory`] therefore maps rung 1 to an all-false [`GenerationMemory`]: there is
//! nothing per-request to turn on, and publishing a knob that silently does nothing is worse than
//! publishing none.
//!
//! This is the same load-time-vs-request-time seam `mlx-gen-z-image` records and Krea's CUDA adoption
//! has, so it is a shared-contract gap rather than a SANA one. It is restated here so no calibration
//! reads a rung-1 cell as request-selectable.
//!
//! The converse also holds and is the more surprising direction: **rung 2 needs no residency rung.**
//! `resolved_decode_plan` honours a request's `tile_vae_decode` *even under `Resident`*, so a
//! `BoundedDecode` selection is executable on a warm Mac generator that never sheds a component. That
//! is why [`MemoryStrategy::BoundedDecode`] is declared `Implemented` unconditionally rather than
//! conditioned on `spec.offload_policy` the way z-image conditions rung 4 on `load_shape`.
//!
//! ## What rung 2 is worth here (measured)
//!
//! Untiled is not a baseline this provider can offer on a phone. At 1024² an untiled DC-AE decode
//! measured **9177 MiB** on the host and **killed the app on device**; the tiled path completed at
//! 2751 MiB (`docs/ios-epics.md`, E5). Rung 2 is the difference between a render and a jetsam kill,
//! not a percentage saving.
//!
//! The *shape* of the saving is in [`DECODE_TILE_EDGES`], and it is the unusual one: the curve is
//! flat below 192 px. That flatness is what draws the line between the admitted and rejected sets.

use mlx_gen::gen_core::{
    Error as CoreError, GenerationMemory, GenerationRequest, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryGeometry, MemoryLifecycleCapabilities, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemoryRunOutcome,
    MemoryRuntimeSemantics, MemorySafetyDecision, MemorySelection, MemoryStrategy,
    MemoryStrategyCapability, MemoryStrategyParameters, MemoryStrategySupport, OffloadPolicy,
    Result as CoreResult,
};

use crate::pipeline::{DECODE_OVERLAP, DECODE_TILE_EDGE};

/// The production decode tile-edge ladder for rung 2, in **output pixels**.
///
/// # How the set was drawn
///
/// Swept on real weights at 1024², 4 steps, `OffloadPolicy::Sequential`, against a 4096 MiB budget
/// (`mlx-gen-ios-catalog`'s `image_budget --sequential-only --tile E --overlap 48`), reported as
/// **request peak active**, beside the pixel cost of each against a forced whole-image decode of the
/// same seed (`tiling_fidelity --overlap 48`, same geometry):
///
/// | edge | peak (MiB) | max \|Δ\| | mean \|Δ\| | >8/255 | >32/255 |
/// |---|---:|---:|---:|---:|---:|
/// | 512 | 5146 | 255 | 3.042 | 8.97% | 1.04% |
/// | 384 | 4496 | 232 | 3.374 | 10.24% | 0.98% |
/// | 256 | 3465 | 212 | 4.478 | 13.66% | 1.79% |
/// | **192** | **3294** | 225 | **5.127** | 15.65% | 2.09% |
/// | 128 | 3294 | 236 | 6.214 | 18.54% | 2.68% |
/// | 96 | 3294 | 223 | 7.349 | 21.44% | 3.20% |
/// | 64 | 3294 | 239 | 9.397 | 41.21% | 3.71% |
///
/// Every row is at **overlap 48**, this ladder's single published value, because that is the shape
/// [`DecodeRoutes`](mlx_gen_pid::DecodeRoutes) carries and therefore the shape a selector can
/// actually choose. An earlier sweep drove `MLX_GEN_SANA_DECODE_TILE`, whose overlap is hardwired to
/// `edge / 4` — so it moved two variables at once and its quality column described pairings the
/// contract cannot publish.
///
/// **The peak column is unchanged by that correction, to the megabyte** (5146 / 4496 / 3465 / 3294
/// under both overlaps), and that is itself the measurement: the decode transient is set by the tile's
/// *area*, while overlap only changes how many tiles cover the output. The quality column did move,
/// as feathering should — 512 went 2.454 → 3.042 once its overlap dropped from 128 to 48.
///
/// The 192 row is the cross-check between the two harness paths, because it is the one edge whose
/// quarter *is* 48 — the same configuration measured through the env override and then through the
/// request. It reproduced to 5.158 → 5.127, i.e. within run-to-run Metal variance rather than
/// exactly. Read that 0.6% as noise, not as a configuration difference.
///
/// **The admitted set is exactly the edges at which the peak still moves.** From 192 px down the
/// request peak is pinned at 3294 MiB — the denoise phase binds there and no smaller tile can touch
/// it — while the image keeps degrading monotonically. Every edge below 192 is therefore image
/// quality paid for no admission win, which is a strictly bad trade and the reason
/// [`DECODE_TILE_EDGES_REJECTED`] exists.
///
/// **The rejection is carried by the flat memory curve, not by a quality threshold**, and that
/// distinction matters when reading it. `mlx-gen-z-image`'s ladder separates cleanly on quality (its
/// admitted set tops out at 48/255 and its rejected set starts at 64/255, a 33% gap); SANA's does
/// not — 192's mean 5.127 and 128's 6.214 are 21% apart, and max \|Δ\| does not separate at all
/// (it is 212-255 across the whole sweep, admitted and rejected alike, which is why the mean and the
/// percentile columns are the ones worth reading). If the memory curve had kept falling, 128 would be
/// a legitimate candidate. It does not, so it is not.
///
/// # The contingency, stated up front
///
/// The 3294 MiB floor is **not set by the decode tile** — it is the denoise phase. So the rejection
/// is conditional on this provider's ladder having no denoise-bounding rung, which today is true
/// (rungs 3 and 4 are `Missing`). This is the same argument `mlx-gen-z-image`'s ladder made and then
/// had to retire when its rung 4 landed and moved the binding phase onto the decode: sub-512 edges
/// there went from "buy nothing" to "the difference between shipping and not on an 8 GB device".
///
/// **If SANA ever gains a rung that bounds denoise, 128/96/64 must be re-swept before they stay
/// rejected.** Recording that here rather than discovering it later is the whole reason the rejected
/// set is a published constant with a test rather than a deleted row.
///
/// # 512 is the largest edge measured, not the largest admissible
///
/// The sweep stopped at 512 because that is already 1568 MiB above the floor at 1024², not because
/// 768 was tried and refused. A reader must not take this ceiling for a measured exclusion — that
/// confusion (an absent value silently acquiring a second meaning) is the exact class of defect that
/// produced four wrong readings in the work that built this ladder.
///
/// # Why the larger edges are published at all
///
/// 512 and 384 cost *more* memory than the default and ship anyway, because the ladder is a
/// **domain**, not a recommendation. At a larger output a 192 px tile is many more forwards, and 256
/// buys a visibly better image (mean 4.478 vs 5.127) for +171 MiB of peak. The selector — not this
/// file — owns the peak-vs-quality-vs-latency choice against a live budget. Selection is the
/// worker's; this is the set it may choose from.
pub const DECODE_TILE_EDGES: &[u32] = &[512, 384, 256, 192];

/// Tile edges swept and **rejected** by measurement (see [`DECODE_TILE_EDGES`]), kept as a published
/// constant so the sweep can re-assert the exclusion rather than leaving it a comment that drifts.
///
/// A change that made these worth their quality cost — a denoise-bounding rung, or a DC-AE variant
/// whose peak kept falling — would show up as this list failing its rejection check. Silently
/// dropping them would leave nothing to notice that with.
///
/// The overlap correction that re-measured [`DECODE_TILE_EDGES`] does **not** reach these: their
/// peaks are the 3294 MiB floor under both overlaps, and a floor set by the denoise phase is not
/// something a decode-side parameter can move. Their quality numbers in the table above are the
/// fixed-48 ones, so the whole sweep is one comparison rather than two spliced together.
pub const DECODE_TILE_EDGES_REJECTED: &[u32] = &[128, 96, 64];

/// This provider's bounded-decode route declaration, built through the shared
/// [`DecodeRoutes`](mlx_gen_pid::DecodeRoutes) (SC-15775).
///
/// # Why a PiD type, on a provider with no PiD decoder
///
/// SANA depends on `mlx-gen-pid` for its Gemma-2 CHI caption encoder and for nothing else: there is no
/// super-resolving student here, and `model::generate` never reads
/// [`GenerationRequest::use_pid`] — the `Residency` seam's flag of that name is reused internally to
/// mean "this request needs the DC-AE *encoder*".
///
/// That does not exempt this provider from the shared declaration, and `check-workspace.py` enforces
/// as much: `mlx_gen_pid::engine::selected_decode_tiling` is provider-agnostic, so the hazard is a
/// property of the dependency graph rather than of today's wiring. `DecodeRoutes::new` proves what
/// makes route-aware admission sound — that no native edge is also a legal PiD tile — and refuses to
/// construct otherwise. Hand-rolling the same check here would be a second definition of "conforming"
/// that could drift.
///
/// Fallible and deliberately propagated rather than `expect`-ed: [`DECODE_TILE_EDGES`] is a `const`
/// this provider owns, so the `Err` arm is unreachable for any shipping load — and
/// `no_published_edge_is_a_legal_pid_tile` proves it — but a future widening of the ladder into the
/// student's range must fail typed at load rather than panic in a release build.
fn decode_routes(provider_id: &str) -> CoreResult<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new(
        provider_id,
        DECODE_TILE_EDGES.iter().copied(),
        DECODE_OVERLAP as u32,
    )
    .map_err(|errors| CoreError::Unsupported(errors.join("; ")))
}

/// Calibration content fingerprint: the key external [`MemoryEvidence`] must carry to attach to this
/// provider. It must change whenever quantization floors, tensor layout, or execution structure
/// change in a way that invalidates measurements taken against it.
///
/// **This is a key, not evidence.** Declaring it does not assert that any peak has been measured —
/// `observed_peak_bytes` lives in `MemoryEvidence`, which is external and minted by a calibration
/// harness, not here. What declaring it *does* is close admission: without an identity,
/// [`safety_check`] rejects every selection, so the conservative move is to mint the key and let the
/// handshake fail until real evidence exists.
///
/// # The suffix is load-bearing
///
/// The whole sweep above ran `Sequential`. Under `Resident` the Gemma encoder and the DiT are both
/// still live through the decode — `run_staged` sheds neither — so a peak measured one way must never
/// authorize the other. The offload policy splits this fingerprint for exactly the reason
/// `mlx-gen-z-image`'s `LoadShape` splits its own.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "sana-mlx-dcae-tiled-decode-v1-sequential";

/// The `Resident` counterpart of [`MEMORY_CALIBRATION_FINGERPRINT`]. **No sweep has been run against
/// it** — it exists so that a resident load cannot present the sequential evidence, not because
/// resident numbers are on hand.
pub const RESIDENT_MEMORY_CALIBRATION_FINGERPRINT: &str = "sana-mlx-dcae-tiled-decode-v1-resident";

pub const fn memory_calibration_fingerprint(policy: OffloadPolicy) -> &'static str {
    match policy {
        OffloadPolicy::Sequential => MEMORY_CALIBRATION_FINGERPRINT,
        OffloadPolicy::Resident => RESIDENT_MEMORY_CALIBRATION_FINGERPRINT,
    }
}

/// Build the SANA MLX provider contract for `provider_id`.
///
/// `spec` supplies two load-exact facts the declaration depends on: the component `.safetensors` sums
/// under the resolved snapshot root ([`asset_facts`]), and the offload policy that selects the
/// calibration identity.
///
/// Fallible only because [`decode_routes`] is: a native ladder that reached into the PiD student's
/// domain cannot be constructed, so it cannot be published either. Nothing else here can fail.
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    // Bound once: the declaration is checked at construction, so building it twice inside the
    // capability map would re-run the check and re-allocate for no gain.
    let routes = decode_routes(provider_id)?;
    Ok(MemoryProviderContract {
        provider_id: provider_id.to_owned(),
        backend: MemoryBackendRealization::MlxMetal {
            // Unified memory: the wired-residency budget is what the staged phases release, weights
            // are mmap-backed, and MLX's lazy graph needs an explicit `eval` before a phase drop
            // frees anything (`Residency::run_staged` owns that discipline). No host↔device transfer.
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
        strategies: MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: match strategy {
                    // See the module docs: rung 3 is Missing rather than StructurallyNotApplicable
                    // because `attn2` does build a score matrix (a 300-key one); rung 4 has no
                    // window driver in this trunk at all.
                    MemoryStrategy::BoundedAttention
                    | MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
                    _ => MemoryStrategySupport::Implemented,
                },
                parameters: match strategy {
                    // The published UNION of both routes, because the shared static validator is
                    // route-blind. `safety_check` and `configure_decode`, which do see `use_pid`,
                    // enforce the route's own subset.
                    MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                        decode_tile_edges: routes.published_edges(),
                        decode_overlaps: routes.published_overlaps(),
                        ..Default::default()
                    },
                    _ => MemoryParameterRanges::default(),
                },
            })
            .collect(),
        // SANA materializes each component eagerly when its phase loads it. There is no deferred
        // block path here, which is the same fact rung 4's `Missing` states from the other side.
        load_shape: mlx_gen::LoadShape::EagerMaterialization,
        additional_prerequisites: Vec::new(),
        lifecycle: MemoryLifecycleCapabilities {
            phases: vec![
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: false,
            transformer_window_materialization: false,
        },
        formula: MemoryFormulaKind::PhaseEnvelope {
            phases: vec![
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode,
            ],
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::BatchCount,
                MemoryFormulaVariable::ConditioningTokenCount,
                MemoryFormulaVariable::DecodeTileArea,
                // Deliberately NOT AttentionChunkSize or TransformerWindowSize: neither rung is
                // implemented, so neither is a variable of this provider's peak.
            ],
        },
        calibration: Some(MemoryCalibrationIdentity::new(
            memory_calibration_fingerprint(spec.offload_policy),
        )),
        asset_facts: asset_facts(spec),
        runtime: MemoryRuntimeSemantics::default(),
    })
}

/// Component `.safetensors` sums for the spec's snapshot root. A [`WeightsSource::File`] source has
/// no component tree, so every field stays `0` (the truthful "unknown", not a guess).
fn asset_facts(spec: &LoadSpec) -> MemoryAssetFacts {
    let Ok(components) = crate::model::component_footprint(spec) else {
        return MemoryAssetFacts::default();
    };
    MemoryAssetFacts {
        base_bytes: components
            .text_encoder
            .saturating_add(components.dit)
            .saturating_add(components.vae),
        conditioning_bytes: components.text_encoder,
        transformer_bytes: components.dit,
        decoder_bytes: components.vae,
        // SANA has no overlay checkpoint — no PiD student, no ControlNet. Zero is the fact, not a
        // placeholder.
        overlay_bytes: 0,
    }
}

/// The provider safety check both SANA ids share: the calibration handshake, the shared contract's
/// own selection validation, the measured decode pairing, then the budget. Defense in depth only — it
/// can reject, it can never swap in a different strategy or numeric tier.
pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let Some(calibration) = contract.calibration.as_ref() else {
        return MemorySafetyDecision::Reject {
            reason: format!("{}: no calibration identity declared", contract.provider_id),
        };
    };
    if context.calibration_abi != calibration.abi
        || context.calibration_fingerprint != calibration.fingerprint
    {
        return MemorySafetyDecision::Reject {
            reason: format!(
                "{}: calibration handshake mismatch (admitted abi {} fingerprint {:?}, provider abi \
                 {} fingerprint {:?})",
                contract.provider_id,
                context.calibration_abi,
                context.calibration_fingerprint,
                calibration.abi,
                calibration.fingerprint
            ),
        };
    }
    if let Err(error) = contract.validate_selection(&context.selection) {
        return MemorySafetyDecision::Reject {
            reason: error.to_string(),
        };
    }
    // SC-15805: ask the contract whether this selection ENGAGES rung 2 rather than re-deriving it
    // from the enum's numeric order.
    if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
        let routes = match decode_routes(contract.provider_id.as_str()) {
            Ok(routes) => routes,
            // Unreachable for a shipping load (the ladder is a `const`, proven disjoint by
            // `no_published_edge_is_a_legal_pid_tile`); a rejection rather than a panic if a future
            // widening ever makes it reachable.
            Err(error) => {
                return MemorySafetyDecision::Reject {
                    reason: error.to_string(),
                }
            }
        };
        if let Err(reason) = routes.validate(
            context.use_pid,
            context.selection.parameters.decode_tile_edge,
            context.selection.parameters.decode_overlap,
        ) {
            return MemorySafetyDecision::Reject { reason };
        }
        // The one place a hand-rolled check is still right, and it covers a real gap in the shared
        // type rather than duplicating it. `DecodeRoutes` assumes a PiD-*eligible* provider can run
        // the PiD route, so `validate(true, 2048, 256)` accepts — SANA is eligible by dependency
        // (Gemma-2) and has no student, so it would accept a selection it cannot execute. The
        // published union is what makes the edge reachable at all, so the rejection lives here,
        // AFTER the shared gate, where it narrows rather than replaces it.
        if context.use_pid {
            return MemorySafetyDecision::Reject {
                reason: format!(
                    "{}: bounded decode was selected on the PiD route, but this provider has no PiD \
                     decoder — it depends on mlx-gen-pid only for the Gemma-2 caption encoder",
                    contract.provider_id
                ),
            };
        }
    }
    if !context.budget.fits(context.predicted_peak_bytes) {
        return MemorySafetyDecision::Reject {
            reason: format!(
                "{}: predicted peak {} exceeds effective budget {}",
                contract.provider_id,
                context.predicted_peak_bytes,
                context.budget.effective_bytes()
            ),
        };
    }
    MemorySafetyDecision::Accept
}

/// The shared ladder → this provider's existing per-request execution controls.
///
/// Ask [`MemoryProviderContract::engages`] which rungs a selection actually engages rather than
/// re-deriving it from the enum's numeric order (SC-15805) — that seam is what stops a cost-order
/// default from switching on a lever this provider declares `Missing`.
///
/// Rung 1 maps to an all-false [`GenerationMemory`] on purpose: staging is load-time here, so there is
/// no request field for it to set (see the module docs). `Resident` returns `None`, the historical
/// fast path with [`GenerationRequest::memory`] untouched.
pub(crate) fn sana_generation_memory(
    contract: &MemoryProviderContract,
    selection: &MemorySelection,
) -> Option<GenerationMemory> {
    if selection.strategy == MemoryStrategy::Resident {
        return None;
    }
    let parameters = selection.parameters;
    let tile_vae_decode = contract.engages(selection.strategy, MemoryStrategy::BoundedDecode);
    Some(GenerationMemory {
        tile_vae_decode,
        // Each parameter is gated on its OWN rung being engaged, so a lever that is off never ships
        // the values it would have been driven with.
        decode_tile_edge: tile_vae_decode
            .then_some(parameters.decode_tile_edge)
            .flatten(),
        decode_overlap: tile_vae_decode
            .then_some(parameters.decode_overlap)
            .flatten(),
        ..Default::default()
    })
}

/// Request-scoped lifecycle state for one admitted SANA generation.
///
/// Holds no MLX arrays: its whole job is to translate the shared selection into
/// [`GenerationRequest::memory`], reject parameters this provider does not implement, and guarantee
/// the terminal synchronize-and-release on success, cancellation, **and** error.
pub(crate) struct SanaMemoryScope {
    provider_id: &'static str,
    geometry: MemoryGeometry,
    memory: Option<GenerationMemory>,
    /// Which decode route this request was admitted for, so `configure_decode` validates against the
    /// same route `safety_check` did.
    ///
    /// Carried rather than assumed `false`. It IS false on every rung-2 scope — `safety_check`
    /// refuses a PiD-routed bounded decode outright — but `MemoryRequestScope` is a trait object, and
    /// resting a validation argument on "the only constructor already checked" is the kind of
    /// invariant that holds until someone adds a second path to the value. Storing it costs a `bool`
    /// and cannot go stale.
    use_pid: bool,
    finished: bool,
}

impl SanaMemoryScope {
    fn ensure_active(&self) -> CoreResult<()> {
        if self.finished {
            Err(CoreError::Msg(format!(
                "{}: memory-strategy request scope is already finished",
                self.provider_id
            )))
        } else {
            Ok(())
        }
    }

    /// Terminal barrier + cache eviction, idempotent.
    ///
    /// MLX is lazy and its allocator retains freed buffers in a cache, so "the request is over" is
    /// only true after (a) a synchronous barrier on the default stream — which is what
    /// [`mlx_rs::Array::eval`] is — and (b) an explicit [`clear_cache`](mlx_rs::memory::clear_cache).
    /// Without both, a canceled or errored request can leave partially-resident buffers that poison
    /// the next request's budget. This runs on every exit path, including [`Drop`].
    fn synchronize_and_release(&mut self) -> CoreResult<()> {
        let barrier = mlx_rs::Array::from(0.0_f32);
        barrier.eval().map_err(mlx_gen::Error::from)?;
        drop(barrier);
        mlx_rs::memory::clear_cache();
        self.finished = true;
        Ok(())
    }
}

impl MemoryRequestScope for SanaMemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> CoreResult<()> {
        self.ensure_active()?;
        if request.width != self.geometry.width
            || request.height != self.geometry.height
            || request.count == 0
            || request.count > self.geometry.batch
        {
            return Err(CoreError::Unsupported(format!(
                "{}: request geometry {}x{} count {} does not match admitted {}x{} count {}",
                self.provider_id,
                request.width,
                request.height,
                request.count,
                self.geometry.width,
                self.geometry.height,
                self.geometry.batch
            )));
        }
        // The shared selection is authoritative and request-scoped: overwrite (never merge) whatever
        // a reused warm request carried, so a deeper prior rung cannot leak into this run.
        request.memory = self.memory;
        Ok(())
    }

    fn enter_phase(&mut self, _phase: MemoryPhase) -> CoreResult<()> {
        // The phase boundaries themselves are owned by `Residency::run_staged`, which already
        // evaluates and drops between phases; the scope only has to stay live across them.
        self.ensure_active()
    }

    fn leave_phase(&mut self, _phase: MemoryPhase) -> CoreResult<()> {
        self.ensure_active()
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> CoreResult<()> {
        self.ensure_active()?;
        // Geometry re-checked here and not only in `configure_request`: the decode peak is a function
        // of tile area *and* output area, so a scope must not execute a resolution admission refused.
        if geometry != self.geometry {
            return Err(CoreError::Unsupported(format!(
                "{}: decode geometry {}x{} count {} does not match admitted {}x{} count {}",
                self.provider_id,
                geometry.width,
                geometry.height,
                geometry.batch,
                self.geometry.width,
                self.geometry.height,
                self.geometry.batch
            )));
        }
        // The same shared gate `safety_check` uses, on the route this scope was admitted for — one
        // implementation, so the scope cannot admit a geometry admission refused, or vice versa.
        decode_routes(self.provider_id)?
            .validate(self.use_pid, Some(tile_edge), Some(overlap))
            .map_err(CoreError::Unsupported)
    }

    fn configure_attention(&mut self, chunk_size: u32) -> CoreResult<()> {
        self.ensure_active()?;
        Err(CoreError::Unsupported(format!(
            "{}: bounded attention is not implemented on this provider (its self-attention is \
             linear and materializes no score matrix), so chunk size {chunk_size} cannot be honoured",
            self.provider_id
        )))
    }

    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> CoreResult<()> {
        self.ensure_active()?;
        // Accepting an arbitrary window here would let a harness record a sweep point this provider
        // never executed — the same class of false green as declaring a rung Implemented because a
        // sibling provider has one.
        Err(CoreError::Unsupported(format!(
            "{}: bounded transformer residency is not implemented on this provider, so the window \
             at block {first_block} of {block_count} blocks was never materialized",
            self.provider_id
        )))
    }

    fn finish(&mut self, _outcome: MemoryRunOutcome) -> CoreResult<()> {
        // Deliberately outcome-independent: cancellation and error need the barrier + eviction at
        // least as much as success does.
        self.ensure_active()?;
        self.synchronize_and_release()
    }
}

impl Drop for SanaMemoryScope {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.synchronize_and_release();
        }
    }
}

/// Open a request scope after [`safety_check`] accepted `context`.
pub(crate) fn begin_request(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    Ok(Some(Box::new(SanaMemoryScope {
        provider_id,
        geometry: context.geometry,
        memory: sana_generation_memory(contract, &context.selection),
        use_pid: context.use_pid,
        finished: false,
    })))
}

/// The strategy parameters this provider accepts, for a caller that wants the production default in
/// one value (the conformance tests and the SceneWorks evidence writer both key off this).
///
/// Deliberately built from the pipeline's own [`DECODE_TILE_EDGE`] / [`DECODE_OVERLAP`] rather than
/// restating numbers: the default and the domain cannot drift, and a default that left the published
/// ladder fails `the_default_edge_is_in_the_published_ladder`.
pub fn declared_parameters() -> MemoryStrategyParameters {
    MemoryStrategyParameters {
        decode_tile_edge: Some(DECODE_TILE_EDGE as u32),
        decode_overlap: Some(DECODE_OVERLAP as u32),
        attention_chunk_size: None,
        transformer_window_size: None,
        transformer_window_component: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core_testkit::memory_strategy::memory_strategy_conformance;
    use mlx_gen::gen_core::{
        MemoryBudget, MemoryCacheState, MemoryMode, MemoryNumericTier, MEMORY_CALIBRATION_ABI,
    };
    use mlx_gen::{Precision, Quant, WeightsSource};
    use mlx_gen_pid::DecodeRoutes;

    fn spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec {
            offload_policy: policy,
            ..LoadSpec::new(WeightsSource::Dir("/nonexistent/sana-contract-test".into()))
        }
    }

    fn contract() -> MemoryProviderContract {
        memory_strategy_contract(crate::model::MODEL_ID, &spec(OffloadPolicy::Sequential)).unwrap()
    }

    /// SANA's shipping iOS tier. Immutable across every selection — strategies cannot change it.
    fn tier() -> MemoryNumericTier {
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
        }
    }

    fn selection(
        strategy: MemoryStrategy,
        parameters: MemoryStrategyParameters,
    ) -> MemorySelection {
        MemorySelection {
            strategy,
            parameters,
            tier: tier(),
        }
    }

    fn context(strategy: MemoryStrategy) -> MemoryRunContext {
        MemoryRunContext {
            selection: selection(strategy, declared_parameters()),
            calibration_abi: MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: MEMORY_CALIBRATION_FINGERPRINT.to_owned(),
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: true,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 8 << 30,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            // The measured floor from `DECODE_TILE_EDGES`, so the budget arm of `safety_check` is
            // exercised against a real number rather than a token one.
            predicted_peak_bytes: 3294 << 20,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: String::new(),
        }
    }

    #[test]
    fn the_contract_passes_shared_conformance() {
        memory_strategy_conformance(&contract());
    }

    #[test]
    fn both_sana_ids_publish_the_same_ladder() {
        let sprint =
            memory_strategy_contract(crate::SPRINT_MODEL_ID, &spec(OffloadPolicy::Sequential))
                .unwrap();
        // The whole capability, not just its parameters: `support` is half of what a rung declares,
        // and a Sprint that declared one differently would slip past a parameters-only comparison.
        for strategy in MemoryStrategy::ALL {
            assert_eq!(
                contract().capability(strategy),
                sprint.capability(strategy),
                "Sprint differs in scheduler and guidance, neither of which is a memory seam"
            );
        }
    }

    /// The three rungs this provider actually executes, and the two it does not. A rung declared
    /// `Implemented` because a sibling provider has one is the failure this pins.
    #[test]
    fn the_ladder_is_three_rungs_and_the_absences_are_declared() {
        let contract = contract();
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "{strategy:?} is executable here"
            );
        }
        for strategy in [
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Missing,
                "{strategy:?} has no seam in this provider"
            );
        }
    }

    /// Rung 2 is selectable on a `Resident` load. `resolved_decode_plan` honours a request's
    /// `tile_vae_decode` regardless of `is_sequential`, so conditioning the capability on the
    /// offload policy would refuse a selection this provider can execute.
    #[test]
    fn bounded_decode_is_implemented_on_a_resident_load_too() {
        let resident =
            memory_strategy_contract(crate::model::MODEL_ID, &spec(OffloadPolicy::Resident))
                .unwrap();
        assert_eq!(
            resident
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        assert!(crate::pipeline::resolved_decode_plan(
            Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(DECODE_TILE_EDGE as u32),
                ..Default::default()
            }),
            false,
        )
        .is_some());
    }

    /// A `Sequential` peak must never authorize a `Resident` admission: under `Resident` neither the
    /// Gemma encoder nor the DiT is shed before the decode.
    #[test]
    fn the_offload_policy_splits_the_calibration_fingerprint() {
        let sequential = contract();
        let resident =
            memory_strategy_contract(crate::model::MODEL_ID, &spec(OffloadPolicy::Resident))
                .unwrap();
        assert_ne!(
            sequential.calibration.as_ref().unwrap().fingerprint,
            resident.calibration.as_ref().unwrap().fingerprint
        );

        // And the handshake enforces it, rather than the difference merely being recorded.
        let ctx = context(MemoryStrategy::BoundedDecode);
        assert_eq!(
            safety_check(&sequential, &ctx),
            MemorySafetyDecision::Accept
        );
        assert!(matches!(
            safety_check(&resident, &ctx),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    /// The domain is a measurement (see [`DECODE_TILE_EDGES`]): the admitted set is the edges at
    /// which the request peak still moves, and the rejected set is where it has floored at
    /// 3294 MiB.
    #[test]
    fn the_published_ladder_is_the_swept_one() {
        assert_eq!(DECODE_TILE_EDGES, &[512, 384, 256, 192]);
        assert_eq!(DECODE_TILE_EDGES_REJECTED, &[128, 96, 64]);
        for edge in DECODE_TILE_EDGES_REJECTED {
            assert!(
                !DECODE_TILE_EDGES.contains(edge),
                "edge {edge} cannot be both admitted and rejected"
            );
        }
        // Descending, so a selector walking the ladder walks it in one direction.
        assert!(DECODE_TILE_EDGES.windows(2).all(|w| w[0] > w[1]));
    }

    /// The default must be a member of the domain it defaults within — otherwise the load-time path
    /// and the contract path decode differently and only one of them is published.
    #[test]
    fn the_default_edge_is_in_the_published_ladder() {
        assert!(DECODE_TILE_EDGES.contains(&(DECODE_TILE_EDGE as u32)));
        assert_eq!(
            *DECODE_TILE_EDGES.last().unwrap(),
            DECODE_TILE_EDGE as u32,
            "the default is the SMALLEST admitted edge — the one that first reaches the 3294 MiB \
             floor, so nothing below it buys admission"
        );
    }

    /// One overlap across the whole ladder, and it is the measured one. `DecodeRoutes` carries a
    /// single `native_overlap`, and the calibration sweep was re-run at exactly this value rather
    /// than the `edge / 4` the env override derives.
    #[test]
    fn the_ladder_publishes_one_measured_overlap() {
        assert_eq!(DECODE_OVERLAP, 48);
        let routes = decode_routes(crate::model::MODEL_ID).unwrap();
        assert_eq!(routes.domain(false).1, DECODE_OVERLAP as u32);
        assert_eq!(routes.native_edges(), DECODE_TILE_EDGES);
    }

    /// The precondition that makes route-aware admission sound: an edge on the wire is unambiguous,
    /// so admission can tell which route a selection was built for. `DecodeRoutes::new` refuses to
    /// construct otherwise, which is why this asserts through the constructor rather than by hand.
    #[test]
    fn no_published_edge_is_a_legal_pid_tile() {
        decode_routes(crate::model::MODEL_ID).expect("the native ladder is disjoint from PiD's");
        let pid = DecodeRoutes::pid_edges();
        for edge in DECODE_TILE_EDGES {
            assert!(!pid.contains(edge), "native edge {edge} is also a PiD tile");
        }
    }

    /// The static contract publishes the UNION of both routes, because `validate_selection` never
    /// sees `use_pid`. Publishing only the native ladder would fail every PiD selection before
    /// route-aware admission could reject it with a reason that names the route.
    #[test]
    fn the_static_contract_publishes_the_union_of_both_routes() {
        let contract = contract();
        let ranges = &contract
            .capability(MemoryStrategy::BoundedDecode)
            .unwrap()
            .parameters;
        for edge in DECODE_TILE_EDGES {
            assert!(ranges.decode_tile_edges.contains(edge));
        }
        for edge in DecodeRoutes::pid_edges() {
            assert!(ranges.decode_tile_edges.contains(&edge));
        }
        assert!(ranges.decode_overlaps.contains(&(DECODE_OVERLAP as u32)));
        assert!(ranges
            .decode_overlaps
            .contains(&DecodeRoutes::pid_overlap()));
    }

    /// The published union necessarily admits a cross-product this provider never measured — PiD's
    /// 256 px overlap is a legal value of `decode_overlaps`, so the static validator accepts it
    /// beside a native edge. Route-aware admission is what narrows it back. Narrower at admission,
    /// never broader.
    #[test]
    fn a_published_but_unmeasured_edge_overlap_pairing_is_rejected() {
        let contract = contract();
        let mut selected = selection(
            MemoryStrategy::BoundedDecode,
            MemoryStrategyParameters {
                decode_tile_edge: Some(DECODE_TILE_EDGE as u32),
                decode_overlap: Some(DecodeRoutes::pid_overlap()),
                ..declared_parameters()
            },
        );
        // Both values are individually published, so the static validator accepts them...
        contract.validate_selection(&selected).unwrap();

        // ...and the route gate is what rejects them.
        let mut ctx = context(MemoryStrategy::BoundedDecode);
        ctx.selection = selected;
        assert!(matches!(
            safety_check(&contract, &ctx),
            MemorySafetyDecision::Reject { .. }
        ));

        // The measured pairing for the same edge passes.
        selected.parameters.decode_overlap = Some(DECODE_OVERLAP as u32);
        ctx.selection = selected;
        assert_eq!(safety_check(&contract, &ctx), MemorySafetyDecision::Accept);
    }

    #[test]
    fn a_rejected_edge_is_refused_at_admission() {
        let contract = contract();
        for &edge in DECODE_TILE_EDGES_REJECTED {
            let mut ctx = context(MemoryStrategy::BoundedDecode);
            ctx.selection.parameters.decode_tile_edge = Some(edge);
            assert!(
                matches!(
                    safety_check(&contract, &ctx),
                    MemorySafetyDecision::Reject { .. }
                ),
                "edge {edge} was measured to buy no admission win and must not be selectable"
            );
        }
    }

    /// SANA has no PiD decoder — `model::generate` never reads `GenerationRequest::use_pid`. Both
    /// halves are pinned, because they fail for different reasons and only one is the shared type's:
    /// a NATIVE edge on the PiD route is rejected by `DecodeRoutes::validate`, and a PiD edge on the
    /// PiD route (which the shared type accepts, since eligibility is by dependency) is rejected by
    /// this provider's own narrowing.
    #[test]
    fn a_pid_routed_bounded_decode_is_refused_either_way() {
        let contract = contract();

        let mut native_on_pid = context(MemoryStrategy::BoundedDecode);
        native_on_pid.use_pid = true;
        assert!(matches!(
            safety_check(&contract, &native_on_pid),
            MemorySafetyDecision::Reject { .. }
        ));

        let mut pid_on_pid = context(MemoryStrategy::BoundedDecode);
        pid_on_pid.use_pid = true;
        pid_on_pid.selection.parameters.decode_tile_edge =
            DecodeRoutes::pid_edges().first().copied();
        pid_on_pid.selection.parameters.decode_overlap = Some(DecodeRoutes::pid_overlap());
        // The shared gate ACCEPTS this one — it assumes a PiD-eligible provider has a student.
        decode_routes(crate::model::MODEL_ID)
            .unwrap()
            .validate(
                true,
                pid_on_pid.selection.parameters.decode_tile_edge,
                pid_on_pid.selection.parameters.decode_overlap,
            )
            .expect("the shared type admits the PiD route for any eligible provider");
        // ...and this provider's own narrowing is what refuses it.
        assert!(matches!(
            safety_check(&contract, &pid_on_pid),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    /// Selecting a rung this provider declares `Missing` must fail at admission rather than at
    /// generate time, and the two absent rungs must not be reachable through the cost-order default.
    #[test]
    fn the_missing_rungs_are_not_selectable() {
        let contract = contract();
        for strategy in [
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert!(matches!(
                safety_check(&contract, &context(strategy)),
                MemorySafetyDecision::Reject { .. }
            ));
        }
        // And a rung-2 selection does not drag them in.
        assert!(!contract.engages(
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention
        ));
    }

    /// Rung 1 is load-time here, so it carries no request lever. An all-false `GenerationMemory` is
    /// the honest translation — the alternative is a knob that silently does nothing.
    #[test]
    fn staged_residency_carries_no_request_lever() {
        let memory = sana_generation_memory(
            &contract(),
            &selection(MemoryStrategy::StagedResidency, declared_parameters()),
        )
        .expect("an optimized rung produces a GenerationMemory");
        assert!(!memory.tile_vae_decode);
        assert_eq!(memory.decode_tile_edge, None);
        assert_eq!(memory.decode_overlap, None);
    }

    #[test]
    fn resident_leaves_the_request_untouched() {
        assert_eq!(
            sana_generation_memory(
                &contract(),
                &selection(
                    MemoryStrategy::Resident,
                    MemoryStrategyParameters::default()
                )
            ),
            None
        );
    }

    /// A rung-2 selection's parameters reach the executable path `pipeline::resolve_decode_tiling`
    /// reads. This is the seam phase 1 built; the test is what keeps the two halves attached.
    #[test]
    fn a_bounded_decode_selection_reaches_the_pipeline() {
        let memory = sana_generation_memory(
            &contract(),
            &selection(
                MemoryStrategy::BoundedDecode,
                MemoryStrategyParameters {
                    decode_tile_edge: Some(256),
                    ..declared_parameters()
                },
            ),
        )
        .unwrap();
        assert!(memory.tile_vae_decode);

        let plan = crate::pipeline::resolved_decode_plan(Some(memory), false)
            .expect("a rung-2 selection must tile");
        assert_eq!(plan.edge, 256);
        assert_eq!(plan.overlap, DECODE_OVERLAP);
        assert_eq!(plan.source, crate::pipeline::DecodeTilingSource::Request);
    }

    /// A `WeightsSource::File` has no component tree, so the asset facts stay the truthful zero.
    #[test]
    fn a_single_file_source_reports_zero_asset_facts_rather_than_a_guess() {
        let contract = memory_strategy_contract(
            crate::model::MODEL_ID,
            &LoadSpec::new(WeightsSource::File("/nonexistent/sana.safetensors".into())),
        )
        .unwrap();
        assert_eq!(contract.asset_facts, MemoryAssetFacts::default());
    }

    #[test]
    fn declared_parameters_names_only_the_rungs_this_provider_implements() {
        let parameters = declared_parameters();
        assert_eq!(parameters.attention_chunk_size, None);
        assert_eq!(parameters.transformer_window_size, None);
        assert_eq!(parameters.transformer_window_component, None);
    }
}
