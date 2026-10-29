//! Route admission: decode geometry, the profile route gate, safety checks, the registered
//! fixture and request scopes.

use super::*;

/// The canvas the weights-free behavior fixtures admit: the shipped default, exactly at
/// [`CANVAS_MAX_PIXELS`].
pub(super) const FIXTURE_WIDTH: u32 = 1344;

pub(super) const FIXTURE_HEIGHT: u32 = 768;

/// The shortest legal clip — 124 frames, 5.1667 s at 24 fps.
///
/// The shared [`standard_memory_behavior_context`] defaults to a single 1024x1024 frame, and
/// **both halves of that are illegal here**: `1024·1024` is 1.6 % over [`CANVAS_MAX_PIXELS`], and
/// `T = 1` is off the `17n + 5` lattice and does not render at all. A fixture that kept the default
/// would be admitted only by a route gate that checks nothing.
pub(super) const FIXTURE_FRAMES: u32 = LEGAL_FRAME_COUNTS[0] as u32;

/// The three render routes, each with its own evidence-key mode spelling.
///
/// `t2va` and `fl2va` map onto the shared text-to-image / image-to-image spellings; `ref2va` has no
/// shared spelling and takes [`MemoryMode::Other`], which exists for exactly this.
pub(super) fn routes() -> Vec<(MemoryMode, u32)> {
    vec![
        (MemoryMode::TextToImage, 0),
        (MemoryMode::ImageToImage, 0),
        (MemoryMode::Other("ref2va".to_owned()), 1),
    ]
}

/// Admit a bounded-decode geometry — **the** rung-2 predicate, and the only one.
///
/// Both admission seams call this: `route_gate` (the `safety_check` half) and the request scope's
/// `decode_validator` (the `configure_decode` half). Sharing one predicate is not tidiness — it is
/// what stops the pair drifting, which is the failure this family has already shipped twice, most
/// recently when `encode_clip` hardcoded its tile constants while `decode_clip` read `self.tiling`
/// so `disable_tiling()` disabled only half the VAE (sc-19008).
///
/// The domain is a singleton, so this refuses everything except the reference geometry. The message
/// names *why* rather than only *what*, because a caller reaching it has asked for the one thing
/// this rung cannot trade: output fidelity for memory.
pub fn validate_decode_geometry(edge: u32, overlap: u32) -> mlx_gen::gen_core::Result<()> {
    if edge == DECODE_TILE_EDGE && overlap == DECODE_OVERLAP {
        return Ok(());
    }
    Err(CoreError::Unsupported(format!(
        "{MODEL_ID}: bounded decode admits only the reference geometry {DECODE_TILE_EDGE}px tile / \
         {DECODE_OVERLAP}px overlap, got {edge}/{overlap}. The tile geometry is an output-\
         correctness input copied from the published VAE, not a memory lever: MiniMax-H3 was \
         released with tiling on and the released frames are the blended-tile ones, so a budgeted \
         tile would change the output (sc-18786). A zero overlap additionally starves the seams, \
         and that corruption is visible across frames rather than within one."
    )))
}

/// Reject a run context whose geometry the render itself would refuse.
///
/// This is the non-vacuous half of admission: the lattice and canvas gates are the same ones
/// `crate::pipeline::resolve_geometry` enforces, so a context that passes here is a context the
/// generator can actually run. `use_pid` is refused outright — there is no PiD decode route.
pub(super) fn route_gate(context: &MemoryRunContext) -> mlx_gen::gen_core::Result<()> {
    if context.use_pid {
        return Err(CoreError::Unsupported(format!(
            "{MODEL_ID} has no PiD decode route"
        )));
    }
    // sc-18660. Guarded by contract-aware engagement rather than an ordinal compare, so a rung the
    // contract does not implement never demands its parameters.
    if context
        .selection
        .strategy
        .engages(MemoryStrategy::BoundedDecode)
    {
        if let (Some(edge), Some(overlap)) = (
            context.selection.parameters.decode_tile_edge,
            context.selection.parameters.decode_overlap,
        ) {
            validate_decode_geometry(edge, overlap)?;
        }
    }
    let geometry = context.geometry;
    if !LEGAL_FRAME_COUNTS.contains(&(geometry.frames as i32)) {
        return Err(CoreError::Unsupported(format!(
            "{MODEL_ID}: {} frames is off the 17n+5 lattice (124..=345); T=1 does not render",
            geometry.frames
        )));
    }
    if !geometry.width.is_multiple_of(SPATIAL_STRIDE)
        || !geometry.height.is_multiple_of(SPATIAL_STRIDE)
    {
        return Err(CoreError::Unsupported(format!(
            "{MODEL_ID}: {}x{} is not a multiple of the {SPATIAL_STRIDE}px stride",
            geometry.width, geometry.height
        )));
    }
    if geometry.width.saturating_mul(geometry.height) > CANVAS_MAX_PIXELS {
        return Err(CoreError::Unsupported(format!(
            "{MODEL_ID}: {}x{} exceeds the {CANVAS_MAX_PIXELS}px canvas budget",
            geometry.width, geometry.height
        )));
    }
    if geometry.batch != 1 {
        return Err(CoreError::Unsupported(format!(
            "{MODEL_ID} renders one clip per request, got batch {}",
            geometry.batch
        )));
    }
    Ok(())
}

/// The numeric tier a spec loads at — the one immutable axis of a selection this provider varies.
///
/// The floor list is empty because every component is packed at the DiT's own tier: the packed text
/// encoder tracks whatever `{base}.scales` it ships with rather than a declared per-component floor
/// (see [`crate::model::TEXT_ENCODER_COMPONENT`]), and the two VAEs are dense on every tier.
///
/// Resolved in one place so the loaded route (`crate::model::load`, which captures this on the
/// generator) and the registry's weights-free probe cannot end up grading against different tiers.
pub fn numeric_tier(spec: &LoadSpec) -> MemoryNumericTier {
    mlx_gen::request_scope::loaded_tier(spec)
}

/// The provider's real admission check for a **loaded** route, callable before any weight file is
/// opened.
///
/// Takes the already-resolved tier rather than the spec because the loaded generator is the caller
/// that matters: `Generator::memory_strategy_safety_check` holds a contract and a tier captured once
/// at load (sc-18650), not the `LoadSpec` that produced them.
pub fn safety_check_at_tier(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    mlx_gen::gen_core::standard_safety_check(&MEMORY_PROFILE, contract, context, tier)
}

/// The MiniMax-H3 MLX memory profile (sc-27307): requested tier, the joint-AV route gate inside the
/// shared pipeline, and the per-tier fingerprint spelling with the measured bf16 cell pinned. The
/// behavior fixture stays provider-owned (the shared default geometry is off this family's frame
/// lattice, so each route's context carries the `FIXTURE_*` geometry).
pub(super) const MEMORY_PROFILE: mlx_gen::gen_core::ProviderMemoryProfile =
    mlx_gen::gen_core::ProviderMemoryProfile {
        gates: mlx_gen::gen_core::MemoryProfileGates::within(profile_route_gate),
        fingerprint: Some(mlx_gen::gen_core::MemoryFingerprintTable {
            template: "minimax-h3-{tier}-mlx-staged-joint-av-eager-abi3-v1",
            routes: &[],
            pins: &[mlx_gen::gen_core::MemoryFingerprintPin {
                route: "",
                tier: CALIBRATED_TIER,
                fingerprint: MEMORY_CALIBRATION_FINGERPRINT,
            }],
        }),
        ..mlx_gen::gen_core::ProviderMemoryProfile::STANDARD
    };

pub(super) fn profile_route_gate(
    _contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    _tier: MemoryNumericTier,
) -> mlx_gen::gen_core::Result<()> {
    route_gate(context)
}

/// The spec-taking admission check — the [`MEMORY_REGISTRATION`] entry point, which probes
/// weights-free from a spec and so resolves the tier here rather than holding one.
///
/// **The bare name is load-bearing across backends.** `scripts/check-workspace.py` compares the two
/// backends' `pub const` declarations as canonicalized *source text*, so `MEMORY_REGISTRATION` must
/// spell its `safety_check` field identically here and in `candle-gen-minimax-h3` — which registers
/// the shorthand `safety_check,` against its own spec-taking function of that name. Renaming this to
/// the wan-style `registered_safety_check` reds that gate even though nothing about the registration
/// actually changed; the loaded variant is [`safety_check_at_tier`] for that reason.
pub fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    mlx_gen::gen_core::standard_registered_safety_check(&MEMORY_PROFILE, spec, contract, context)
}

/// Weights-free executable fixtures for one implemented optimized rung — one per render route.
pub fn registered_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> mlx_gen::gen_core::Result<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || contract
            .capability(strategy)
            .is_none_or(|capability| capability.support != MemoryStrategySupport::Implemented)
    {
        return Ok(Vec::new());
    }
    let tier = numeric_tier(spec);
    routes()
        .into_iter()
        .map(|(mode, reference_count)| {
            let mut context = standard_memory_behavior_context(
                contract,
                strategy,
                tier,
                MemoryBehaviorRoute {
                    mode,
                    reference_count,
                    use_pid: false,
                    has_phases: true,
                    overlay: None,
                },
            )?;
            // The shared default geometry is illegal for this family — see `FIXTURE_FRAMES`.
            context.geometry = MemoryGeometry {
                width: FIXTURE_WIDTH,
                height: FIXTURE_HEIGHT,
                batch: 1,
                frames: FIXTURE_FRAMES,
                reference_count,
            };
            // `MemoryBehaviorFixture::new` carries `frames` across from `context.geometry` along
            // with width, height, batch and reference count (sc-19591), so the request this family
            // is graded against declares `FIXTURE_FRAMES` rather than the one-frame default that
            // would be refused as off-lattice. Asserted by
            // `the_fixture_request_declares_the_lattice_frame_count` below; no post-hoc override.
            Ok(MemoryBehaviorFixture::new(context))
        })
        .collect()
}

pub(super) fn begin_request_with_cleanup(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    mlx_gen::request_scope::begin_scope(
        safety_check_at_tier(contract, tier, context),
        cleanup,
        move || {
            let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
                MODEL_ID,
                context.geometry,
                contract.generation_memory(&context.selection),
                context.use_pid,
                DIT_BLOCKS as usize,
                // sc-18660. The same predicate the `safety_check` route gate uses, so `configure_decode`
                // and admission cannot disagree about what is legal.
                |_use_pid, edge, overlap| validate_decode_geometry(edge, overlap),
            )?;
            // sc-18662. **Declaration is not enforcement.** Publishing `transformer_window_sizes` without
            // this leaves `materialize_transformer_window` refusing every window the contract just
            // advertised — the registry conformance walk drives the declared parameter through the scope and
            // caught exactly that. Guarded by contract-aware `engages` rather than an ordinal compare, so a
            // resident load (where rung 4 is `Missing`) never admits a window.
            config.transformer_window = contract
                .engages(
                    context.selection.strategy,
                    MemoryStrategy::BoundedTransformerResidency,
                )
                .then(|| {
                    context
                        .selection
                        .parameters
                        .transformer_window_size
                        .unwrap_or(TRANSFORMER_WINDOW_SIZE)
                });
            Ok(config)
        },
    )
}

/// Registry-facing entry point: no device cleanup, because the weights-free probe never allocated.
pub fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    mlx_gen::gen_core::standard_begin_request(
        &MEMORY_PROFILE,
        MODEL_ID,
        spec,
        contract,
        context,
        registered_scope,
    )
}

pub(super) fn registered_scope(
    _provider_id: &'static str,
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    tier: MemoryNumericTier,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        contract,
        tier,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

/// Production entry point: terminal cleanup drains the MLX allocator cache.
///
/// This is what `Generator::begin_memory_strategy_request` calls on the loaded generator (sc-18650).
/// Until that override existed the function was reachable from nothing: the macro-emitted
/// `impl Generator` left the trait default in the vtable slot, so a shipped render opened no scope.
pub fn begin_request(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        contract,
        tier,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

/// The shared MLX Bf16/Q4/Q8 witness set: this provider ships all three tiers and both eager and
/// deferred (sc-18662) materialization shapes, so no selector filter is needed. A local name (the
/// wan/ltx idiom) so the fixture registration spells identically across the two backends for the
/// cross-backend geometry gate.
pub(super) fn memory_contract_surface_specs() -> Vec<mlx_gen::gen_core::MemoryContractSurfaceSpec> {
    mlx_gen::gen_core::mlx_memory_contract_surface_specs()
}

/// Request-only sparse estimate; off leaves the dense memory contract byte-for-byte unchanged.
pub fn veda_workspace_bound(
    req: mlx_gen::gen_core::veda::workspace::Request<'_>,
) -> mlx_gen::gen_core::Result<Option<mlx_gen::gen_core::veda::workspace::WorkspaceBound>> {
    mlx_gen::gen_core::veda::workspace::for_video_request(
        req,
        mlx_gen::gen_core::veda::workspace::Backend::Mlx,
    )
}

/// Loaded-generator sparse bounds use the same selected source geometry as conditioning.
pub fn veda_workspace_bound_for_contract(
    req: mlx_gen::gen_core::veda::workspace::Request<'_>,
    contract: &MemoryProviderContract,
) -> mlx_gen::gen_core::Result<Option<mlx_gen::gen_core::veda::workspace::WorkspaceBound>> {
    if req.veda.is_none_or(|veda| !veda.enabled) {
        return Ok(None);
    }
    let patch = match contract
        .phase_facts
        .as_ref()
        .and_then(|p| p.native_quant.as_ref())
    {
        Some(native) => native.selected_patch(req)?,
        None => [1, 2, 2], // unresolved inventory only; no materialized source geometry exists
    };
    mlx_gen::gen_core::veda::workspace::for_video_request_with_patch(
        req,
        mlx_gen::gen_core::veda::workspace::Backend::Mlx,
        patch,
    )
}
