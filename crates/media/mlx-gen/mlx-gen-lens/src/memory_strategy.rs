//! Lens / Lens-Turbo MLX shared image-memory ladder.
//!
//! SC-15800's dense-bf16 text-encoder-only result remains an independent legacy measurement. The
//! full ladder uses a new identity and does not relabel that result as DiT, Both, Q4, decode, or
//! attention evidence.
//!
//! # Declaration vs measurement (SC-18605)
//!
//! Before SC-18605 this module derived the *whole* contract from the measured-route key: a load that
//! was not the exact measured `lens` Q4 route or the exact legacy dense `lens_turbo` route published
//! nothing but [`MemoryStrategy::Resident`]. The engine did not agree. `generate_memory_impl` gates
//! staged residency, tiled decode and chunked attention on nothing at all — they are request levers
//! available on every load — and gates the rung-4 block window purely on
//! [`can_stream_text`]/[`can_stream_dit`]. So ten of the twelve `lens` registry surfaces and eleven
//! of the twelve `lens_turbo` surfaces carried an executable ladder that no consumer could ever
//! select, because it was undeclared. That is the inverse of the usual defect: the mechanism was
//! reachable, the declaration was not.
//!
//! The contract therefore now has two independent axes, and every consumer must read both:
//!
//! * **Support** — which rungs the engine can execute for this exact `(provider, LoadSpec)`. Derived
//!   from the engine's own predicates, never from the measured-route key.
//! * **Calibration identity** — whether a *measurement* backs this route. The two measured keys are
//!   unchanged and still bind to their exact envelopes. Every other route publishes no production
//!   calibration, so [`mlx_gen::gen_core::standard_memory_strategy_safety_check`] admits it only
//!   under an explicit [`mlx_gen::gen_core::MemoryOptimizationAuthority::Estimated`] authority.
//!
//! The weights-free declaration surface consumed by the catalog inventory substitutes a static
//! behavior identity ([`STATIC_BEHAVIOR_FINGERPRINT`]) for the absent production calibration so the
//! registry conformance walk can build a run context. That identity is never a measurement and never
//! appears on a contract built from a real load.
//!
//! ## The one asymmetry, deliberately kept
//!
//! `lens_turbo` at `bf16:sequential:deferred` is the single surface covered by the SC-15800 legacy
//! text-encoder measurement. It keeps exactly that envelope — rung 4 scoped to
//! [`TransformerComponent::TextEncoder`], with rungs 1-3 unpublished — because broadening it would
//! relabel that measurement as decode, attention or `Both` evidence, which the module contract above
//! forbids. Its five uncalibrated deferred siblings publish the full structural ladder. Closing that
//! gap needs a new `lens_turbo` measurement, not a wider declaration.

use mlx_gen::gen_core::{
    Error as CoreError, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryNumericTier, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyPrerequisite, MemoryStrategySupport,
    Result as CoreResult, TransformerComponent,
};
#[cfg(test)]
use mlx_gen::{gen_core::MemoryGeometry, GenerationRequest};
use mlx_gen::{LoadShape, LoadSpec, OffloadPolicy, Precision, WeightsSource};

pub const LEGACY_TEXT_ENCODER_FINGERPRINT: &str = "lens-text-encoder-window-2026-07-31-v1";
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "lens-mlx-shared-ladder-2026-08-03-v1";
/// Static, weights-free identity for the registry declaration walk. Never a production calibration.
///
/// A contract built from a real load leaves an unmeasured route's calibration `None` so admission
/// must name an explicit estimate authority. The weights-free surface cannot do that: the shared
/// conformance walk builds its run context through
/// [`mlx_gen::gen_core::standard_memory_behavior_context`], which requires *some* identity. This
/// constant supplies one whose value is structural, so an unmeasured declared rung is still walked
/// without any measured claim attaching to it.
pub const STATIC_BEHAVIOR_FINGERPRINT: &str = "lens-mlx-registry-behavior-v1";
pub const TEXT_ENCODER_WINDOW: u32 = 1;
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_OVERLAP: u32 = 128;
pub const ATTENTION_CHUNK_SIZE: u32 = 16_777_216;

fn decode_routes(provider_id: &str) -> CoreResult<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(provider_id, [DECODE_TILE_EDGE], DECODE_OVERLAP)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CalibrationRoute {
    FullQ4Lens,
    LegacyDenseLensTurboTextEncoder,
    Unmeasured,
}

/// The exact load shape for which the measured production rung is executable and beneficial.
pub(crate) fn is_streamable_spec(spec: &LoadSpec) -> bool {
    matches!(spec.weights, WeightsSource::Dir(_))
        && matches!(spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.precision, Precision::Bf16)
        && spec.quantize.is_none()
        && spec.adapters.is_empty()
        && spec.pid.is_none()
}

fn base_streamable(spec: &LoadSpec) -> Option<&std::path::Path> {
    if spec.load_shape != LoadShape::DeferredMaterialization
        || spec.precision != Precision::Bf16
        || spec.pid.is_some()
    {
        return None;
    }
    match &spec.weights {
        WeightsSource::Dir(root) => Some(root),
        WeightsSource::File(_) => None,
    }
}

/// Fail closed on an id this crate does not register.
///
/// This mattered less while an unmeasured route declared nothing but `Resident`. Now that an
/// unmeasured route publishes a real ladder, an unrecognized id must not be handed one: the rungs
/// below are claims about the Lens engine specifically, and nothing else is entitled to them.
fn validate_provider(provider_id: &str) -> CoreResult<()> {
    if matches!(
        provider_id,
        crate::registry::MODEL_ID_BASE | crate::registry::MODEL_ID_TURBO
    ) {
        Ok(())
    } else {
        Err(CoreError::Unsupported(format!(
            "unknown Lens provider {provider_id}"
        )))
    }
}

fn calibration_route(
    provider_id: &str,
    spec: &LoadSpec,
    text_streamable: bool,
    dit_streamable: bool,
) -> CalibrationRoute {
    if provider_id == "lens"
        && spec.quantize == Some(mlx_gen::Quant::Q4)
        && spec.adapters.is_empty()
        && text_streamable
        && dit_streamable
    {
        CalibrationRoute::FullQ4Lens
    } else if provider_id == "lens_turbo" && is_streamable_spec(spec) {
        CalibrationRoute::LegacyDenseLensTurboTextEncoder
    } else {
        CalibrationRoute::Unmeasured
    }
}

pub(crate) fn can_stream_text(spec: &LoadSpec) -> CoreResult<bool> {
    let Some(root) = base_streamable(spec) else {
        return Ok(false);
    };
    Ok(match spec.quantize {
        Some(quant) => {
            !mlx_gen::quant::needs_load_time_quant(root, "text_encoder", quant.bits(), "lens")?
        }
        None => true,
    })
}

pub(crate) fn can_stream_dit(spec: &LoadSpec) -> CoreResult<bool> {
    let Some(root) = base_streamable(spec) else {
        return Ok(false);
    };
    if !spec.adapters.is_empty() {
        return Ok(false);
    }
    Ok(match spec.quantize {
        Some(quant) => {
            !mlx_gen::quant::needs_load_time_quant(root, "transformer", quant.bits(), "lens")?
        }
        None => true,
    })
}

/// Rung-4 component scopes the engine can actually execute for one load.
///
/// `registry::resolve_transformer_windows` refuses a `TextEncoder`/`Both` window without a
/// streamable text encoder and a `Dit`/`Both` window without a streamable DiT, so this is that
/// gate's own predicate rather than a parallel list that can drift away from it.
fn structural_window_components(
    text_streamable: bool,
    dit_streamable: bool,
) -> Vec<TransformerComponent> {
    let mut components = Vec::new();
    if text_streamable && dit_streamable {
        components.push(TransformerComponent::Both);
    }
    if text_streamable {
        components.push(TransformerComponent::TextEncoder);
    }
    if dit_streamable {
        components.push(TransformerComponent::Dit);
    }
    components
}

/// Per-route static behavior identity for the weights-free declaration surface.
///
/// One shared string across every selector would let a context assembled for one route hand its
/// handshake to another. Keying the identity on the exact axes the contract shape depends on —
/// provider, numeric tier, offload policy, plus the load shape the identity already carries — keeps
/// the declaration walk fail-closed in the same way the measured identities are.
fn static_behavior_identity(provider_id: &str, spec: &LoadSpec) -> MemoryCalibrationIdentity {
    let precision = match spec.precision {
        Precision::Bf16 => "bf16",
        Precision::Fp32 => "fp32",
    };
    let quant = match spec.quantize {
        None => "dense",
        Some(mlx_gen::Quant::Q4) => "q4",
        Some(mlx_gen::Quant::Q8) => "q8",
        Some(mlx_gen::Quant::Nvfp4) => "nvfp4",
    };
    let policy = match spec.offload_policy {
        OffloadPolicy::Resident => "resident",
        OffloadPolicy::Sequential => "sequential",
    };
    // `MemoryProviderContract::conformance_errors` requires lowercase kebab tokens, and the provider
    // ids are snake_case, so `lens_turbo` has to be spelled `lens-turbo` here.
    let route = provider_id.replace('_', "-");
    MemoryCalibrationIdentity::new(
        format!("{STATIC_BEHAVIOR_FINGERPRINT}-{route}-{precision}-{quant}-{policy}"),
        spec.load_shape,
    )
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let text_streamable = can_stream_text(spec)?;
    let dit_streamable = can_stream_dit(spec)?;
    let footprint = crate::registry::component_footprint(spec)?;
    memory_strategy_contract_with_surface_facts(
        provider_id,
        spec,
        text_streamable,
        dit_streamable,
        footprint,
        // A production contract never fabricates an identity: an unmeasured route stays uncalibrated
        // so admission has to name an explicit estimate authority for it.
        None,
    )
}

/// Declaration-equivalent registry contract with no filesystem traversal.
///
/// SceneWorks catalog tiers are provisioned snapshot directories. Their q4/q8 subtrees are already
/// packed, so the weights-free surface treats a deferred directory as re-openable rather than
/// asking `needs_load_time_quant` to inspect a nonexistent witness path.
pub fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let base_streamable = matches!(spec.weights, WeightsSource::Dir(_))
        && spec.load_shape == LoadShape::DeferredMaterialization
        && spec.precision == Precision::Bf16
        && spec.pid.is_none();
    memory_strategy_contract_with_surface_facts(
        provider_id,
        spec,
        base_streamable,
        base_streamable && spec.adapters.is_empty(),
        mlx_gen::gen_core::PerComponentBytes::default(),
        Some(static_behavior_identity(provider_id, spec)),
    )
}

fn memory_strategy_contract_with_surface_facts(
    provider_id: &str,
    spec: &LoadSpec,
    text_streamable: bool,
    dit_streamable: bool,
    footprint: mlx_gen::gen_core::PerComponentBytes,
    unmeasured_identity: Option<MemoryCalibrationIdentity>,
) -> CoreResult<MemoryProviderContract> {
    validate_provider(provider_id)?;
    let calibration_route = calibration_route(provider_id, spec, text_streamable, dit_streamable);
    let legacy_text_encoder_route =
        calibration_route == CalibrationRoute::LegacyDenseLensTurboTextEncoder;
    // Each measured route publishes exactly the envelope it measured; every other route publishes
    // what the engine can execute. Keeping this one list is what stops the two claims from drifting.
    let window_components = match calibration_route {
        CalibrationRoute::FullQ4Lens => vec![TransformerComponent::Both],
        CalibrationRoute::LegacyDenseLensTurboTextEncoder => {
            vec![TransformerComponent::TextEncoder]
        }
        CalibrationRoute::Unmeasured => {
            structural_window_components(text_streamable, dit_streamable)
        }
    };
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
    let mut formula_variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::TransformerWindowSize,
    ];
    if !legacy_text_encoder_route {
        formula_variables.extend([
            MemoryFormulaVariable::DecodeTileArea,
            MemoryFormulaVariable::AttentionChunkSize,
        ]);
    }
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        variables: formula_variables,
    };
    contract.calibration = match calibration_route {
        CalibrationRoute::FullQ4Lens => Some(MemoryCalibrationIdentity::new(
            MEMORY_CALIBRATION_FINGERPRINT,
            spec.load_shape,
        )),
        CalibrationRoute::LegacyDenseLensTurboTextEncoder => Some(MemoryCalibrationIdentity::new(
            LEGACY_TEXT_ENCODER_FINGERPRINT,
            spec.load_shape,
        )),
        CalibrationRoute::Unmeasured => unmeasured_identity,
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
        synchronized_phase_release: true,
        // The legacy text-encoder measurement covers neither lever, and this surface exists to
        // report that measurement truthfully rather than to advertise the engine's full reach.
        decode_tiling: !legacy_text_encoder_route,
        attention_chunking: !legacy_text_encoder_route,
        transformer_window_materialization: !window_components.is_empty(),
    };
    for capability in &mut contract.strategies {
        match capability.strategy {
            // `Residency::run_staged_request_scoped` takes `stage_residency` as a plain request
            // argument and both Lens residency owners are request-scoped for either offload policy,
            // so shedding the encoder before the denoise/decode phase is executable on every load.
            MemoryStrategy::Resident => {
                capability.support = MemoryStrategySupport::Implemented;
            }
            MemoryStrategy::StagedResidency if !legacy_text_encoder_route => {
                capability.support = MemoryStrategySupport::Implemented;
            }
            // `generate_memory_impl` builds the tiling config and the attention budget straight from
            // the request; neither is gated on tier, offload policy or materialization shape.
            MemoryStrategy::BoundedDecode if !legacy_text_encoder_route => {
                capability.support = MemoryStrategySupport::Implemented;
                capability.parameters.decode_tile_edges = vec![DECODE_TILE_EDGE];
                capability.parameters.decode_overlaps = vec![DECODE_OVERLAP];
            }
            MemoryStrategy::BoundedAttention if !legacy_text_encoder_route => {
                capability.support = MemoryStrategySupport::Implemented;
                capability.parameters.attention_chunk_sizes = vec![ATTENTION_CHUNK_SIZE];
            }
            MemoryStrategy::BoundedTransformerResidency if !window_components.is_empty() => {
                capability.support = MemoryStrategySupport::Implemented;
                capability.parameters.transformer_window_sizes = vec![TEXT_ENCODER_WINDOW];
                capability.parameters.transformer_window_components = window_components.clone();
            }
            _ => {}
        }
    }
    if calibration_route == CalibrationRoute::FullQ4Lens {
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

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if context.use_pid
            || contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode)
        {
            let routes = decode_routes(&contract.provider_id)?;
            routes
                .validate(
                    context.use_pid,
                    context.selection.parameters.decode_tile_edge,
                    context.selection.parameters.decode_overlap,
                )
                .map_err(|reason| {
                    let detail = if context.use_pid {
                        "the Lens rung-4 calibration covers native VAE decode only, not the PiD/Gemma overlay"
                    } else {
                        "Lens decode route validation failed"
                    };
                    CoreError::Unsupported(format!(
                        "{}: {detail}: {reason}",
                        contract.provider_id
                    ))
                })?;
        }
        if context.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{}: the Lens rung-4 calibration covers native VAE decode only, not the PiD/Gemma overlay",
                contract.provider_id
            )));
        }
        Ok(())
    };
    mlx_gen::gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision,
            quant,
            component_precision_floors: &[],
        }),
        Some(&route_gate),
    )
}

pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    safety_check(contract, spec.precision, spec.quantize, context)
}

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        },
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: mlx_gen::gen_core::MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            // Derived from the contract this fixture is built for, not from a measured-route key:
            // a fixture that claimed no phases while the selection engages staged residency would
            // be rejected by the very safety gate it exists to exercise.
            has_phases: contract.engages(strategy, MemoryStrategy::StagedResidency),
            overlay: None,
        },
    )?;
    Ok(vec![mlx_gen::gen_core::MemoryBehaviorFixture::new(context)])
}

pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        provider_id,
        contract,
        spec.precision,
        spec.quantize,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub(crate) fn begin_request(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    begin_request_with_cleanup(
        provider_id,
        contract,
        precision,
        quant,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

fn begin_request_with_cleanup(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } =
        safety_check(contract, precision, quant, context)
    {
        return Err(CoreError::Unsupported(reason));
    }
    let component = context.selection.parameters.window_component();
    let routes = decode_routes(provider_id)?;
    let transformer_blocks = match component {
        TransformerComponent::TextEncoder => crate::config::GptOssConfig::lens().num_layers,
        TransformerComponent::Dit | TransformerComponent::Both => {
            crate::dit::LensDitConfig::lens().num_layers
        }
    };
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        transformer_blocks,
        move |use_pid, edge, overlap| {
            routes
                .validate(use_pid, Some(edge), Some(overlap))
                .map_err(CoreError::Unsupported)
        },
    )?;
    // The admitted chunk comes from the selection the contract just validated. Keying this on the
    // measured fingerprint instead would leave every declared-but-unmeasured bounded-attention route
    // with no chunk bound, so `configure_attention` would refuse the rung the contract published.
    config.attention_chunk_size = contract
        .engages(context.selection.strategy, MemoryStrategy::BoundedAttention)
        .then_some(context.selection.parameters.attention_chunk_size)
        .flatten();
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

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        MemoryBudget, MemoryCacheState, MemoryMode, MemoryNumericTier, MemorySelection,
        MemoryStrategyParameters, MemoryStrategySupport, TransformerComponent,
        MEMORY_CALIBRATION_ABI,
    };
    use mlx_gen::{AdapterKind, AdapterSpec, PidWeights, Quant};
    use std::sync::atomic::{AtomicU64, Ordering};

    static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    fn fixture(tmp: &tempfile::TempDir, bits: Option<i32>) -> (std::path::PathBuf, LoadSpec) {
        let root = tmp.path().join(format!(
            "mlx_gen_lens_sc15800_{}",
            FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
        ));
        for component in ["text_encoder", "transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), [0_u8; 8]).unwrap();
        }
        for component in ["text_encoder", "transformer"] {
            let config = match bits {
                Some(bits) => {
                    format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#)
                }
                None => r#"{"dtype":"bfloat16"}"#.to_owned(),
            };
            std::fs::write(root.join(component).join("config.json"), config).unwrap();
        }
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_load_shape(LoadShape::DeferredMaterialization);
        (root, spec)
    }

    fn dense_legacy_spec(tmp: &tempfile::TempDir) -> (std::path::PathBuf, LoadSpec) {
        let (root, spec) = fixture(tmp, None);
        (root, spec.with_offload_policy(OffloadPolicy::Sequential))
    }

    fn packed_spec(
        tmp: &tempfile::TempDir,
        bits: i32,
        quant: Quant,
    ) -> (std::path::PathBuf, LoadSpec) {
        let (root, mut spec) = fixture(tmp, Some(bits));
        spec.quantize = Some(quant);
        (root, spec)
    }

    /// A route that no measurement covers, checked on both of the contract's independent claims.
    ///
    /// The measured claim must stay empty — a production contract off a measured route carries no
    /// calibration identity, so `standard_memory_strategy_safety_check` refuses to admit any
    /// optimized rung on it without an explicit `Estimated` authority. The support claim is a
    /// separate question, and it must equal what the engine can execute: rungs 1-3 are unconditional
    /// request levers, and rung 4 is exactly the scopes `can_stream_text`/`can_stream_dit` allow.
    fn assert_unmeasured(
        contract: &MemoryProviderContract,
        expected_window_components: &[TransformerComponent],
    ) {
        assert!(
            contract.calibration.is_none(),
            "an unmeasured route must carry no production calibration identity"
        );
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "{strategy:?} is an unconditional Lens request lever and must stay selectable"
            );
        }
        let window = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(
            window.support,
            if expected_window_components.is_empty() {
                MemoryStrategySupport::Missing
            } else {
                MemoryStrategySupport::Implemented
            },
            "rung 4 support must equal the engine's own streamability gate"
        );
        assert_eq!(
            window.parameters.transformer_window_components,
            expected_window_components
        );
    }

    #[test]
    fn exact_q4_lens_route_publishes_the_measured_full_ladder() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = packed_spec(&tmp, 4, Quant::Q4);
        let contract = memory_strategy_contract("lens", &spec).unwrap();
        assert!(contract.conformance_errors().is_empty());
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            MEMORY_CALIBRATION_FINGERPRINT
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
        assert_eq!(decode.support, MemoryStrategySupport::Implemented);
        assert_eq!(decode.parameters.decode_tile_edges, vec![DECODE_TILE_EDGE]);
        assert_eq!(decode.parameters.decode_overlaps, vec![DECODE_OVERLAP]);
        let attention = contract
            .capability(MemoryStrategy::BoundedAttention)
            .unwrap();
        assert_eq!(attention.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            attention.parameters.attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );
        let window = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(window.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            window.parameters.transformer_window_components,
            vec![TransformerComponent::Both]
        );
        let measured = rung_four_context().selection;
        contract
            .validate_selection(&measured)
            .expect("the measured Both scope must remain selectable");
        for unmeasured in [TransformerComponent::TextEncoder, TransformerComponent::Dit] {
            let mut selection = measured;
            selection.parameters.transformer_window_component = Some(unmeasured);
            let error = contract
                .validate_selection(&selection)
                .expect_err("an unmeasured component scope must remain unpublished");
            assert!(
                error.to_string().contains("transformer_window_component")
                    && error.to_string().contains("[Both]"),
                "the refusal must identify the unadvertised component: {error}"
            );
        }
        assert!(contract.lifecycle.decode_tiling);
        assert!(contract.lifecycle.attention_chunking);
        assert!(contract.lifecycle.transformer_window_materialization);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn legacy_dense_lens_turbo_te_only_identity_remains_separate() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = dense_legacy_spec(&tmp);
        let contract = memory_strategy_contract("lens_turbo", &spec).unwrap();
        assert!(contract.conformance_errors().is_empty());
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            LEGACY_TEXT_ENCODER_FINGERPRINT
        );
        let window = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(window.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            window.parameters.transformer_window_components,
            vec![TransformerComponent::TextEncoder]
        );
        assert!(!contract.lifecycle.decode_tiling);
        assert!(!contract.lifecycle.attention_chunking);
        let MemoryFormulaKind::PhaseEnvelope { variables, .. } = &contract.formula else {
            panic!("legacy Lens-Turbo calibration must retain its phase envelope")
        };
        assert!(!variables.contains(&MemoryFormulaVariable::DecodeTileArea));
        assert!(!variables.contains(&MemoryFormulaVariable::AttentionChunkSize));
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Missing
            );
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn provider_tier_and_load_shape_mutations_do_not_fan_out_evidence() {
        const EVERY_SCOPE: &[TransformerComponent] = &[
            TransformerComponent::Both,
            TransformerComponent::TextEncoder,
            TransformerComponent::Dit,
        ];

        let tmp = tempfile::tempdir().unwrap();
        let (q4_root, q4) = packed_spec(&tmp, 4, Quant::Q4);
        // The two providers are audited independently: `lens_turbo` at the exact tier that measures
        // `lens` inherits none of that evidence, but it is the same engine, so it keeps the reach.
        assert_unmeasured(
            &memory_strategy_contract("lens_turbo", &q4).unwrap(),
            EVERY_SCOPE,
        );

        let (q8_root, q8) = packed_spec(&tmp, 8, Quant::Q8);
        assert_unmeasured(&memory_strategy_contract("lens", &q8).unwrap(), EVERY_SCOPE);

        let (dense_root, dense) = dense_legacy_spec(&tmp);
        assert_unmeasured(
            &memory_strategy_contract("lens", &dense).unwrap(),
            EVERY_SCOPE,
        );

        // Eager materialization: rungs 1-3 stay executable, but the block window has nothing left to
        // bound, which is the one shared prerequisite gen-core declares for rung 4.
        let mut eager = q4.clone();
        eager.load_shape = LoadShape::EagerMaterialization;
        assert_unmeasured(&memory_strategy_contract("lens", &eager).unwrap(), &[]);

        // A merged adapter fixes the DiT weights at load, so only the encoder half can still stream.
        let mut adapted = q4.clone();
        adapted.adapters.push(AdapterSpec::new(
            q4_root.join("adapter.safetensors"),
            1.0,
            AdapterKind::Lora,
        ));
        assert_unmeasured(
            &memory_strategy_contract("lens", &adapted).unwrap(),
            &[TransformerComponent::TextEncoder],
        );

        let mut pid = q4.clone();
        pid.pid = Some(PidWeights {
            checkpoint: WeightsSource::File(q4_root.join("pid.safetensors")),
            gemma: WeightsSource::Dir(q4_root.join("gemma")),
        });
        assert_unmeasured(&memory_strategy_contract("lens", &pid).unwrap(), &[]);

        for root in [q4_root, q8_root, dense_root] {
            std::fs::remove_dir_all(root).ok();
        }
    }

    /// Rung 4 is declared on every load the engine can actually window, for **both** providers.
    ///
    /// This is the SC-18605 repair stated as an assertion: before it, `lens` published rung 4 on the
    /// two measured Q4 deferred selectors only and `lens_turbo` on the one legacy dense selector
    /// only, while `resolve_transformer_windows` would have run the block window on all six deferred
    /// selectors of both. The declaration, not the mechanism, was the thing missing.
    #[test]
    fn every_deferred_selector_of_both_providers_declares_rung_four() {
        let tmp = tempfile::tempdir().unwrap();
        let mut declared = Vec::new();
        for provider_id in ["lens", "lens_turbo"] {
            for (bits, quant) in [
                (None, None),
                (Some(4), Some(Quant::Q4)),
                (Some(8), Some(Quant::Q8)),
            ] {
                for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
                    for load_shape in [
                        LoadShape::DeferredMaterialization,
                        LoadShape::EagerMaterialization,
                    ] {
                        let (root, mut spec) = fixture(&tmp, bits);
                        spec.quantize = quant;
                        spec = spec.with_offload_policy(policy).with_load_shape(load_shape);
                        let contract = memory_strategy_contract(provider_id, &spec).unwrap();
                        assert!(
                            contract.conformance_errors().is_empty(),
                            "{provider_id} {policy:?} {load_shape:?}"
                        );
                        let window = contract
                            .capability(MemoryStrategy::BoundedTransformerResidency)
                            .unwrap();
                        let implemented = window.support == MemoryStrategySupport::Implemented;
                        assert_eq!(
                            implemented,
                            load_shape == LoadShape::DeferredMaterialization,
                            "{provider_id} {quant:?} {policy:?} {load_shape:?}"
                        );
                        if implemented {
                            assert_eq!(
                                window.parameters.transformer_window_sizes,
                                vec![TEXT_ENCODER_WINDOW]
                            );
                            declared.push((provider_id, policy, quant));
                        }
                        std::fs::remove_dir_all(root).ok();
                    }
                }
            }
        }
        assert_eq!(
            declared.len(),
            12,
            "six deferred selectors per provider must declare rung 4: {declared:?}"
        );
    }

    /// The weights-free declaration surface substitutes a per-route static identity, never a
    /// measurement, and never lets one route's handshake stand in for another's.
    #[test]
    fn weights_free_declaration_identities_are_static_and_route_exact() {
        let mut seen = std::collections::BTreeSet::new();
        for surface in mlx_gen::gen_core::mlx_memory_contract_surface_specs() {
            for provider_id in ["lens", "lens_turbo"] {
                let contract =
                    weights_free_memory_strategy_contract(provider_id, &surface.spec).unwrap();
                assert!(
                    contract.conformance_errors().is_empty(),
                    "{provider_id} {}: {:?}",
                    surface.selector.id(),
                    contract.conformance_errors()
                );
                let fingerprint = &contract.calibration.as_ref().unwrap().fingerprint;
                let measured = fingerprint == MEMORY_CALIBRATION_FINGERPRINT
                    || fingerprint == LEGACY_TEXT_ENCODER_FINGERPRINT;
                if !measured {
                    assert!(
                        fingerprint.starts_with(STATIC_BEHAVIOR_FINGERPRINT),
                        "unmeasured declaration surfaces carry the static behavior identity: \
                         {fingerprint}"
                    );
                    assert!(
                        seen.insert(format!("{fingerprint}:{:?}", surface.spec.load_shape)),
                        "static behavior identity {fingerprint} is reused across routes"
                    );
                }
            }
        }
        // The two measured surfaces are `lens` q4 resident/sequential deferred and `lens_turbo`
        // dense sequential deferred, so every other (provider, selector) pair is static.
        assert_eq!(seen.len(), 2 * 12 - 3);
    }

    #[test]
    fn an_unregistered_provider_is_never_handed_a_lens_ladder() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = packed_spec(&tmp, 4, Quant::Q4);
        for provider_id in ["lens", "lens_turbo"] {
            assert!(memory_strategy_contract(provider_id, &spec).is_ok());
            assert!(weights_free_memory_strategy_contract(provider_id, &spec).is_ok());
        }
        for provider_id in ["lens_pro", "flux1_dev", "", "Lens"] {
            assert!(
                memory_strategy_contract(provider_id, &spec).is_err(),
                "{provider_id}"
            );
            assert!(
                weights_free_memory_strategy_contract(provider_id, &spec).is_err(),
                "{provider_id}"
            );
        }
        std::fs::remove_dir_all(root).ok();
    }

    /// A production contract never fabricates an identity, so an unmeasured optimized rung is
    /// admissible only when the caller explicitly names estimate authority.
    #[test]
    fn unmeasured_production_rungs_require_explicit_estimate_authority() {
        use mlx_gen::gen_core::MemoryOptimizationAuthority;

        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = packed_spec(&tmp, 8, Quant::Q8);
        let contract = memory_strategy_contract("lens", &spec).unwrap();
        let mut context = rung_four_context();
        context.selection.tier.quant = Some(Quant::Q8);
        context.calibration_fingerprint = String::new();
        context.optimization_authority = MemoryOptimizationAuthority::Estimated;
        assert_eq!(
            safety_check(&contract, Precision::Bf16, Some(Quant::Q8), &context),
            MemorySafetyDecision::Accept,
            "the declared-but-unmeasured Q8 rung 4 must be reachable behind an estimate"
        );
        for authority in [
            MemoryOptimizationAuthority::Calibrated,
            MemoryOptimizationAuthority::Resident,
        ] {
            let mut refused = context.clone();
            refused.optimization_authority = authority;
            assert!(
                matches!(
                    safety_check(&contract, Precision::Bf16, Some(Quant::Q8), &refused),
                    MemorySafetyDecision::Reject { .. }
                ),
                "{authority:?} must not admit an unmeasured Lens rung"
            );
        }
        std::fs::remove_dir_all(root).ok();
    }

    fn rung_four_context() -> MemoryRunContext {
        MemoryRunContext {
            optimization_authority: mlx_gen::gen_core::MemoryOptimizationAuthority::Calibrated,
            selection: MemorySelection {
                strategy: MemoryStrategy::BoundedTransformerResidency,
                parameters: MemoryStrategyParameters {
                    decode_tile_edge: Some(DECODE_TILE_EDGE),
                    decode_overlap: Some(DECODE_OVERLAP),
                    attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                    transformer_window_size: Some(TEXT_ENCODER_WINDOW),
                    transformer_window_component: Some(TransformerComponent::Both),
                },
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: &[],
                },
            },
            calibration_abi: MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: MEMORY_CALIBRATION_FINGERPRINT.to_owned(),
            load_shape: LoadShape::DeferredMaterialization,
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: true,
            geometry: MemoryGeometry {
                width: 256,
                height: 256,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 1_000_000,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "test".to_owned(),
        }
    }

    #[test]
    fn selected_contract_scope_reaches_the_generation_request_and_pid_fails_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = packed_spec(&tmp, 4, Quant::Q4);
        let contract = memory_strategy_contract("lens", &spec).unwrap();
        let context = rung_four_context();
        let mut scope = registered_begin_request("lens", &spec, &contract, &context)
            .unwrap()
            .unwrap();
        let mut request = GenerationRequest {
            width: 256,
            height: 256,
            count: 1,
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        let memory = request.memory.expect("rung 4 configures request memory");
        assert!(memory.stage_residency);
        assert!(memory.tile_vae_decode);
        assert!(memory.chunk_attention);
        assert!(memory.stream_transformer_blocks);
        assert_eq!(memory.transformer_window_size, Some(TEXT_ENCODER_WINDOW));
        assert_eq!(
            memory.transformer_window_component,
            Some(TransformerComponent::Both)
        );
        // Registry conformance uses the same tensor-free scope cleanup exercised here; the Device
        // cleanup path is covered by the real-Metal runner.
        drop(scope);

        let mut unmeasured_native = context.clone();
        unmeasured_native.selection.parameters.decode_tile_edge = Some(640);
        assert!(matches!(
            safety_check(
                &contract,
                Precision::Bf16,
                Some(Quant::Q4),
                &unmeasured_native
            ),
            MemorySafetyDecision::Reject { .. }
        ));

        let mut pid = context;
        pid.use_pid = true;
        assert!(matches!(
            safety_check(&contract, Precision::Bf16, Some(Quant::Q4), &pid),
            MemorySafetyDecision::Reject { reason } if reason.contains("native VAE decode only")
        ));
        std::fs::remove_dir_all(root).ok();
    }
}
