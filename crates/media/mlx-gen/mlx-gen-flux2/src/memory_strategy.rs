//! FLUX.2-dev provider-side memory-safety contracts.
//!
//! Text-to-image and edit share weights and the resident execution path, but they are distinct
//! provider registrations with distinct route gates and calibration identities. The edit provider
//! already bounds long multi-reference sequences internally with `MemoryConfig::LONG_SEQ`.
//! SceneWorks supplies request geometry, numeric tier, incremental live demand derived from each
//! route's evidence-owned absolute peak, and the live unified-memory budget. This module validates
//! the exact provider route and tier, then delegates the canonical budget comparison to `gen-core`;
//! calibration coefficients never live in a provider.

use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::gen_core::{
    standard_memory_strategy_safety_check, AdapterResidencyMode, Error as CoreError, LoadShape,
    LoadSpec, MemoryAssetFacts, MemoryBackendRealization, MemoryBehaviorFixture,
    MemoryBehaviorRoute, MemoryBudget, MemoryCacheState, MemoryCalibrationIdentity,
    MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryOptimizationAuthority,
    MemoryPhase, MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope,
    MemoryResidentComponent, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategyPrerequisite, MemoryStrategySupport, Quant, TransformerComponent,
};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{GenerationRequest, OffloadPolicy, WeightsSource};

use crate::config::{Flux2Variant, FLUX2_DEV_CONTROL_ID, FLUX2_DEV_EDIT_ID, FLUX2_DEV_ID};

pub const CALIBRATION_FINGERPRINT: &str = "sc-16593-flux2-dev-edit-evidence-v2";
pub const DEV_T2I_CALIBRATION_FINGERPRINT: &str = "sc-18218-flux2-dev-t2i-resident-evidence-v1";
pub const DEV_CONTROL_OVERLAY: &str = "control";

/// Architecture axes for one FLUX.2 variant (epic SC-22657, E2).
///
/// This crate mirrors the reference `transformer/config.json` as `Flux2Config`, with
/// `Flux2Config::klein_9b` behind the three Klein routes and `Flux2Config::dev` behind the three Dev
/// routes. A route's overlay (edit references, KV edit, the control branch) changes what is
/// resident, never the trunk's shape, so a variant's routes publish one set of axes.
///
/// `transformer_blocks` is the **sum** of the double and single stacks: the denoiser traverses both
/// on every step. `patch_size` is derived rather than declared — `Flux2Config` has no `patch_size`
/// field; it encodes the packing as `in_channels = num_latent_channels * patch²`.
///
/// `vae_temporal_scale` stays `None`: FLUX.2's autoencoder is an image autoencoder with no temporal
/// axis, and a structurally absent axis is declared absent, never zero.
fn architecture_facts(
    config: &crate::config::Flux2Config,
) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(config.num_heads),
        head_dim: mlx_gen::architecture_facts::axis(config.head_dim),
        transformer_blocks: mlx_gen::architecture_facts::axis(
            config
                .num_double_layers
                .saturating_add(config.num_single_layers),
        ),
        patch_size: latent_patch_size(config),
        latent_channels: mlx_gen::architecture_facts::axis(config.num_latent_channels),
        vae_spatial_scale: mlx_gen::architecture_facts::axis(config.vae_scale_factor),
        vae_temporal_scale: None,
        // Weights are bf16, but the transformer runs f32 activations over them (`model.rs`), so f32
        // is the width an activation estimate must use.
        activation_dtype_width: Some(mlx_gen::architecture_facts::FLOAT32_ACTIVATION_WIDTH),
    }
}

/// The square patch edge implied by `in_channels = num_latent_channels * patch²`.
///
/// A config whose ratio is not a small perfect square declines the axis rather than rounding one
/// into existence.
fn latent_patch_size(config: &crate::config::Flux2Config) -> Option<u32> {
    let packed = mlx_gen::architecture_facts::axis(config.in_channels)?;
    let latent = mlx_gen::architecture_facts::axis(config.num_latent_channels)?;
    (1_u32..=8).find(|edge| edge * edge * latent == packed)
}

pub fn contract_for_variant(
    variant: Flux2Variant,
    spec: &LoadSpec,
) -> mlx_gen::Result<Option<MemoryProviderContract>> {
    if variant == Flux2Variant::Dev {
        return Ok(Some(build_dev_t2i_contract_for_spec(spec)));
    }
    if variant == Flux2Variant::DevEdit {
        return Ok(Some(build_contract_for_spec(spec)));
    }
    if matches!(
        variant,
        Flux2Variant::Klein9b | Flux2Variant::Klein9bEdit | Flux2Variant::Klein9bKvEdit
    ) {
        return Ok(Some(klein_contract_for(variant.id(), spec)?));
    }
    Ok(None)
}

/// [`contract_for_variant`] over an inventory the caller has already verified, so a load that
/// holds one does not re-walk the artifact for its memory contract.
pub(crate) fn contract_for_variant_with_inventory(
    variant: Flux2Variant,
    spec: &LoadSpec,
    inventory: Option<&crate::artifact_inventory::KleinArtifactInventory>,
) -> mlx_gen::Result<Option<MemoryProviderContract>> {
    if matches!(
        variant,
        Flux2Variant::Klein9b | Flux2Variant::Klein9bEdit | Flux2Variant::Klein9bKvEdit
    ) {
        crate::model::validate_klein_load_axes(spec, variant.id())
            .map_err(|error| CoreError::Unsupported(error.to_string()))?;
        return Ok(Some(klein_contract_with_inventory(
            variant.id(),
            spec,
            inventory,
        )?));
    }
    contract_for_variant(variant, spec)
}

/// The per-component asset bytes a Dev / Dev-Edit load materializes (SC-22667, E1).
///
/// Both Dev contracts published `MemoryAssetFacts::default()` — five zeros — on every path,
/// including the loaded one: `contract_for_variant` builds the contract the generator carries from
/// load onward, and nothing ever filled the facts in. All-zero facts pass the shared conformance
/// walk vacuously (`base == cond + trans + dec` holds at `0 == 0`), so the omission never failed; it
/// handed the fit gate a contract whose `total_resident_bytes()` was 0 for a load that holds the
/// whole DiT, the Mistral-3 tower with its Pixtral surface, and the VAE.
///
/// The split is `crate::model::dev_component_footprint_for` — the registry footprint callback this
/// crate already exposes for exactly these providers, which resolves the selected language tower
/// through the same `EncoderContract` gate the load applies (including a pre-packed base selected
/// without `LoadSpec::quantize`) and prices the DiT and VAE from the snapshot subdirs the load
/// opens. It is used verbatim rather than re-derived, so the contract and the registry can never
/// disagree about the base split.
///
/// On a spec that names no materialized snapshot — the registry's contract-surface sentinel, or a
/// placeholder path — there is nothing to read and the declaration stays empty, exactly as the
/// weights-free Klein contract does. A snapshot the footprint refuses also publishes the empty
/// declaration; see [`dev_asset_facts_with`] for why that is not a contract refusal here.
///
/// The Fun-ControlNet route is deliberately not routed through here: its contract documents why its
/// facts stay empty (the registry fallback prices the base split and its consumer adds the control
/// checkpoint), and a fused control branch has no honest home in this decomposition without a typed
/// component the route's `Affine` formula cannot carry.
fn dev_asset_facts(provider_id: &str, spec: &LoadSpec) -> MemoryAssetFacts {
    dev_asset_facts_with(crate::model::dev_component_footprint_for, provider_id, spec)
}

/// [`dev_asset_facts`] with the footprint derivation passed in, so the mapping and both of its
/// empty-declaration legs are reachable from a test without a full Dev snapshot on disk.
///
/// A footprint **error** publishes the empty declaration rather than refusing the contract. The
/// registered contract callback is consulted *before* any load — the requested-vs-packed tier
/// rejection and every other registered safety check read the contract first — on snapshots that
/// may carry only their transformer tier evidence, and the Dev registry footprint fails closed on
/// anything short of a complete snapshot. Turning that into a contract refusal would mask the
/// actionable tier rejection behind an I/O error. The empty declaration is the pre-existing
/// shape and errs large: `total_resident_bytes()` is 0, so the fit gate credits nothing as already
/// resident.
///
/// The error is swallowed silently rather than logged: this crate carries no `tracing` or `log`
/// dependency, and the reachable case — a snapshot short of a complete Dev layout, read before any
/// load — is expected rather than exceptional. What keeps the `Ok` arm from silently becoming
/// unreachable is a positive test against the real footprint and a complete snapshot, not a log
/// line: see `the_production_dev_footprint_is_accepted_and_published_for_a_complete_snapshot`.
fn dev_asset_facts_with(
    footprint: impl Fn(&str, &LoadSpec) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes>,
    provider_id: &str,
    spec: &LoadSpec,
) -> MemoryAssetFacts {
    if mlx_gen::architecture_facts::materialized_root(spec).is_none() {
        return MemoryAssetFacts::default();
    }
    let Ok(footprint) = footprint(provider_id, spec) else {
        return MemoryAssetFacts::default();
    };
    MemoryAssetFacts {
        base_bytes: footprint
            .text_encoder
            .saturating_add(footprint.dit)
            .saturating_add(footprint.vae),
        conditioning_bytes: footprint.text_encoder,
        transformer_bytes: footprint.dit,
        decoder_bytes: footprint.vae,
        overlay_bytes: 0,
    }
}

fn build_contract_for_spec(spec: &LoadSpec) -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        FLUX2_DEV_EDIT_ID,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.architecture_facts = architecture_facts(&crate::config::Flux2Config::dev());
    // The only base/edit calibration receipts are eager-load captures.  A sequential lifecycle
    // releases phases after that eager assembly; it is not deferred-materialization evidence.
    contract.load_shape = LoadShape::EagerMaterialization;
    contract.formula = MemoryFormulaKind::Affine {
        variables: vec![
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::ConditioningTokenCount,
        ],
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        CALIBRATION_FINGERPRINT,
        LoadShape::EagerMaterialization,
    ));
    contract.asset_facts = dev_asset_facts(FLUX2_DEV_EDIT_ID, spec);
    configure_dev_staged_residency(&mut contract, spec, true);
    contract
}

fn build_dev_t2i_contract_for_spec(spec: &LoadSpec) -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        FLUX2_DEV_ID,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.architecture_facts = architecture_facts(&crate::config::Flux2Config::dev());
    // See the edit contract above: this fingerprint remains bound to the measured eager load.
    contract.load_shape = LoadShape::EagerMaterialization;
    contract.formula = MemoryFormulaKind::Affine {
        variables: vec![
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::ConditioningTokenCount,
        ],
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        DEV_T2I_CALIBRATION_FINGERPRINT,
        LoadShape::EagerMaterialization,
    ));
    contract.asset_facts = dev_asset_facts(FLUX2_DEV_ID, spec);
    configure_dev_staged_residency(&mut contract, spec, true);
    contract
}

#[cfg(test)]
fn build_contract() -> MemoryProviderContract {
    build_contract_for_spec(&LoadSpec::new(WeightsSource::Dir(Default::default())))
}

#[cfg(test)]
fn build_dev_t2i_contract() -> MemoryProviderContract {
    build_dev_t2i_contract_for_spec(&LoadSpec::new(WeightsSource::Dir(Default::default())))
}

/// Publish only the request-selectable lifecycle the Dev providers actually execute.  The
/// sequential residency seam materializes conditioning and releases it before the heavy phase;
/// activation/decode/transformer rungs remain Missing until route-specific evidence exists.
fn configure_dev_staged_residency(
    contract: &mut MemoryProviderContract,
    spec: &LoadSpec,
    route_is_configured: bool,
) {
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: matches!(spec.offload_policy, OffloadPolicy::Sequential),
        ..Default::default()
    };
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
            MemoryStrategy::StagedResidency
                if route_is_configured
                    && matches!(spec.offload_policy, OffloadPolicy::Sequential)
                    && spec.load_shape == LoadShape::EagerMaterialization =>
            {
                MemoryStrategySupport::Implemented
            }
            _ => MemoryStrategySupport::Missing,
        };
    }
}

/// Honest pre-calibration contract for the standalone MLX Fun-Controlnet route.
///
/// The provider has a real resident execution path, a request-selectable sequential lifecycle, and
/// a registered component-footprint fallback. It has no control-specific calibration evidence, so
/// its staged strategy remains estimate-authorized only; the route/tier safety gate below remains
/// fully load-exact.
///
/// SC-22667 review: the asset facts no longer sit on the all-zero registry fallback. The base
/// split — the selected language tower, the transformer and the VAE — is the same load-exact
/// derivation the T2I and Edit routes publish, one call away through
/// [`crate::model::dev_control_component_footprint`], which is the very callback this route's
/// registration already declares. Publishing it means the fit gate credits the bytes that are
/// genuinely resident instead of crediting nothing.
///
/// What is deliberately **omitted** is the control BRANCH itself. It has no typed home under this
/// contract's `Affine` formula (no `MemoryResidentComponent` list, and `overlay_bytes` names an
/// adapter stack, which a Fun-Controlnet checkpoint is not), and its resident width depends on a
/// runtime distinction — an effective packed base against a dense control branch — that the
/// pre-calibration evidence cannot settle. So `overlay_bytes` stays 0 and the branch remains the
/// consumer's addition on top of these base facts, exactly as before. That errs SMALL against the
/// branch alone, which is why the route's staged rung stays estimate-authorized and its load-exact
/// route/tier gate is unchanged.
fn build_dev_control_contract(spec: &LoadSpec) -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        FLUX2_DEV_CONTROL_ID,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.architecture_facts = architecture_facts(&crate::config::Flux2Config::dev());
    contract.load_shape = spec.load_shape;
    contract.asset_facts = dev_asset_facts_with(
        |_, spec| crate::model::dev_control_component_footprint(spec),
        FLUX2_DEV_CONTROL_ID,
        spec,
    );
    // Unlike base/edit, this route is not complete without its separately-addressed control
    // artifact.  Do not publish a selectable staged rung for a bare base `LoadSpec`.
    configure_dev_staged_residency(&mut contract, spec, spec.control.is_some());
    contract
}

pub fn dev_t2i_safety_check(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    expected_tier: MemoryNumericTier,
) -> MemorySafetyDecision {
    let route_gate = || {
        if context.mode != MemoryMode::TextToImage
            || context.has_reference
            || context.geometry.reference_count != 0
        {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_ID}: memory-safety context must describe reference-free text-to-image"
            )));
        }
        if expected_tier.quant == Some(Quant::Nvfp4) {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_ID}: NVFP4 is not implemented by the MLX provider"
            )));
        }
        Ok(())
    };
    standard_memory_strategy_safety_check(contract, context, Some(expected_tier), Some(&route_gate))
}

pub fn safety_check(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    expected_tier: MemoryNumericTier,
) -> MemorySafetyDecision {
    let route_accepted = std::cell::Cell::new(false);
    let route_gate = || {
        if context.mode != MemoryMode::Edit || !context.has_reference {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_EDIT_ID}: memory-safety context must describe a referenced edit"
            )));
        }
        if context.geometry.reference_count < 2 {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_EDIT_ID}: memory-safety context must name at least two references"
            )));
        }
        if expected_tier.quant == Some(Quant::Nvfp4) {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_EDIT_ID}: NVFP4 is not implemented by the MLX provider"
            )));
        }
        route_accepted.set(true);
        Ok(())
    };
    match standard_memory_strategy_safety_check(
        contract,
        context,
        Some(expected_tier),
        Some(&route_gate),
    ) {
        MemorySafetyDecision::Accept => MemorySafetyDecision::Accept,
        MemorySafetyDecision::Reject { reason }
            if route_accepted.get()
                && reason.contains("incremental live demand")
                && reason.contains("exceeds effective budget") =>
        {
            let reference_count = context.geometry.reference_count;
            let gib = 1024.0 * 1024.0 * 1024.0;
            MemorySafetyDecision::Reject {
                reason: format!(
                    "FLUX.2-dev multi-reference edit at {}×{} with {reference_count} reference \
                     images needs ~{} GB of unified memory (with headroom) but this machine has \
                     ~{} GB. Lower the output resolution, use a single reference image, choose a \
                     smaller numeric tier, or run on a Mac with more memory.",
                    context.geometry.width,
                    context.geometry.height,
                    (context
                        .budget
                        .required_total_bytes(context.predicted_peak_bytes)
                        as f64
                        / gib)
                        .round() as i64,
                    (context.budget.total_bytes as f64 / gib).round() as i64,
                ),
            }
        }
        MemorySafetyDecision::Reject { reason } => MemorySafetyDecision::Reject { reason },
    }
}

pub fn dev_control_safety_check(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    expected_tier: MemoryNumericTier,
) -> MemorySafetyDecision {
    let route_gate = || {
        if contract.provider_id != FLUX2_DEV_CONTROL_ID {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_CONTROL_ID}: memory contract resolved the wrong provider route {}",
                contract.provider_id
            )));
        }
        if context.mode != MemoryMode::TextToImage
            || !context.has_reference
            || context.geometry.reference_count != 1
            || context.overlay.as_deref() != Some(DEV_CONTROL_OVERLAY)
        {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_CONTROL_ID}: memory-safety context must describe text-to-image with exactly one control image and the control overlay"
            )));
        }
        if context.load_shape != contract.load_shape {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_CONTROL_ID}: memory context load shape does not match the loaded contract"
            )));
        }
        if context.calibration_abi != 0 || !context.calibration_fingerprint.is_empty() {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_CONTROL_ID}: uncalibrated control staging must carry the explicit empty calibration handshake"
            )));
        }
        if context.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_CONTROL_ID}: PiD decode is not implemented for the control route"
            )));
        }
        if context.has_phases {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_CONTROL_ID}: multi-phase denoise is not implemented for the control route"
            )));
        }
        if expected_tier.quant == Some(Quant::Nvfp4) {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_CONTROL_ID}: NVFP4 is not implemented by the MLX provider"
            )));
        }
        Ok(())
    };
    standard_memory_strategy_safety_check(contract, context, Some(expected_tier), Some(&route_gate))
}

pub fn registered_dev_contract(
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_contract_for_spec(spec))
}

pub fn registered_dev_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match crate::model::effective_dev_memory_numeric_tier(spec, FLUX2_DEV_EDIT_ID) {
        Ok(tier) => safety_check(contract, context, tier),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub fn registered_dev_t2i_contract(
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_dev_t2i_contract_for_spec(spec))
}

pub fn registered_dev_t2i_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match crate::model::effective_dev_memory_numeric_tier(spec, FLUX2_DEV_ID) {
        Ok(tier) => dev_t2i_safety_check(contract, context, tier),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub fn registered_dev_control_contract(
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_dev_control_contract(spec))
}

pub fn registered_dev_control_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    if let Err(error) = mlx_gen::require_control(
        spec,
        FLUX2_DEV_CONTROL_ID,
        "FLUX.2-dev-Fun-Controlnet-Union",
    ) {
        return MemorySafetyDecision::Reject {
            reason: error.to_string(),
        };
    }
    match crate::model::effective_dev_memory_numeric_tier(spec, FLUX2_DEV_CONTROL_ID) {
        Ok(tier) => dev_control_safety_check(contract, context, tier),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

/// Provider-owned, weights-free conformance fixtures for the exact FLUX.2 Dev routes.
///
/// A staged selection is meaningful only after a sequential load.  Each fixture carries the real
/// route mode, overlay and reference cardinality, so registry conformance cannot accidentally use
/// text-to-image evidence to authorize edit or control behavior.
pub(crate) fn registered_dev_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> mlx_gen::gen_core::Result<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || !matches!(
            contract
                .capability(strategy)
                .map(|capability| &capability.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }

    let tier = crate::model::effective_dev_memory_numeric_tier(spec, &contract.provider_id)?;
    let route = match contract.provider_id.as_str() {
        FLUX2_DEV_ID => MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: true,
            overlay: None,
        },
        FLUX2_DEV_EDIT_ID => MemoryBehaviorRoute {
            mode: MemoryMode::Edit,
            reference_count: 2,
            use_pid: false,
            has_phases: true,
            overlay: None,
        },
        FLUX2_DEV_CONTROL_ID => {
            return Ok(vec![dev_control_fixture(contract, strategy, tier)?]);
        }
        provider_id => {
            return Err(CoreError::Unsupported(format!(
                "unknown FLUX.2 Dev memory provider {provider_id}"
            )))
        }
    };
    let context =
        mlx_gen::gen_core::standard_memory_behavior_context(contract, strategy, tier, route)?;
    Ok(vec![executable_dev_fixture(context)])
}

fn executable_dev_fixture(context: MemoryRunContext) -> MemoryBehaviorFixture {
    let mut fixture = MemoryBehaviorFixture::new(context);
    fixture.request.prompt = "weights-free FLUX.2 Dev memory behavior".to_owned();
    if fixture.context.mode == MemoryMode::Edit {
        fixture.request.conditioning = vec![mlx_gen::Conditioning::MultiReference {
            images: (0..fixture.context.geometry.reference_count)
                .map(|_| mlx_gen::media::Image {
                    width: 1,
                    height: 1,
                    pixels: vec![0, 0, 0],
                })
                .collect(),
        }];
    }
    fixture
}

fn dev_control_fixture(
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
    tier: MemoryNumericTier,
) -> mlx_gen::gen_core::Result<MemoryBehaviorFixture> {
    let context = MemoryRunContext {
        selection: contract.representative_selection(strategy, tier, false)?,
        // The control route intentionally has no measured calibration.  Its staged lifecycle is
        // admitted under the explicit conservative estimate class, never a borrowed Dev/Edit key.
        optimization_authority: MemoryOptimizationAuthority::Estimated,
        calibration_abi: 0,
        calibration_fingerprint: String::new(),
        load_shape: contract.load_shape,
        mode: MemoryMode::TextToImage,
        // `GenerationRequest::image_reference_count` includes a control map. This is not an edit
        // reference semantically, but it is part of the exact admitted geometry seen by the scope.
        has_reference: true,
        use_pid: false,
        // The control sampler is intentionally single-phase; its sequential lifetime releases
        // model components between conditioning, denoise, and decode rather than accepting the
        // multi-phase denoise request feature.
        has_phases: false,
        geometry: mlx_gen::gen_core::MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 1,
        },
        overlay: Some(DEV_CONTROL_OVERLAY.to_owned()),
        budget: MemoryBudget {
            total_bytes: 8 * 1024 * 1024 * 1024,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1024 * 1024 * 1024,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: "weights-free-flux2-dev-control-estimate".to_owned(),
    };
    let mut fixture = MemoryBehaviorFixture::new(context);
    fixture.request.prompt = "weights-free FLUX.2 Dev control memory behavior".to_owned();
    fixture.request.conditioning = vec![mlx_gen::Conditioning::Control {
        image: mlx_gen::media::Image {
            width: 1,
            height: 1,
            pixels: vec![0, 0, 0],
        },
        kind: mlx_gen::ControlKind::Pose,
        scale: None,
    }];
    Ok(fixture)
}

pub(crate) fn registered_dev_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let tier = crate::model::effective_dev_memory_numeric_tier(spec, &contract.provider_id)?;
    begin_dev_request(contract, context, tier)
}

/// Open the shared request scope for the one non-resident Dev strategy.  Residency itself is
/// selected before load through `LoadSpec::offload_policy`; the scope records that request choice,
/// installs the typed generation-memory marker, and guarantees terminal cleanup on success,
/// cancellation, or error.  It deliberately refuses any decode validator use because no Dev
/// decode rung is advertised by this contract.
pub(crate) fn begin_dev_request(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    expected_tier: MemoryNumericTier,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let decision = match contract.provider_id.as_str() {
        FLUX2_DEV_ID => dev_t2i_safety_check(contract, context, expected_tier),
        FLUX2_DEV_EDIT_ID => safety_check(contract, context, expected_tier),
        FLUX2_DEV_CONTROL_ID => dev_control_safety_check(contract, context, expected_tier),
        provider => MemorySafetyDecision::Reject {
            reason: format!("unknown FLUX.2 Dev memory provider {provider}"),
        },
    };
    if let MemorySafetyDecision::Reject { reason } = decision {
        return Err(CoreError::Unsupported(reason));
    }
    if !context.selection.strategy.is_optimized() {
        return Ok(None);
    }
    let provider_id = match contract.provider_id.as_str() {
        FLUX2_DEV_ID => FLUX2_DEV_ID,
        FLUX2_DEV_EDIT_ID => FLUX2_DEV_EDIT_ID,
        FLUX2_DEV_CONTROL_ID => FLUX2_DEV_CONTROL_ID,
        _ => unreachable!("provider was checked above"),
    };
    let config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        false,
        48,
        |_use_pid, _edge, _overlap| {
            Err(CoreError::Unsupported(
                "FLUX.2 Dev does not expose a bounded-decode strategy".to_owned(),
            ))
        },
    )?;
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(
            config,
            mlx_gen::request_scope::MlxScopeCleanup::Device,
        ),
    )))
}

// ---- FLUX.2 Klein shared image-memory ladder (SC-15518) -------------------------------

pub const KLEIN_STATIC_BEHAVIOR_FINGERPRINT: &str = "flux2-klein-static-registry-behavior-v2";
pub const KLEIN_MEMORY_CALIBRATION_FINGERPRINT: &str =
    "flux2-klein-9b-bf16-mlx-shared-ladder-t2i-v1";
pub const KLEIN_CALIBRATED_REVISION: &str = "1d36c68041725a14c76566cdf6cea4270b264b03";
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512];
pub const DECODE_OVERLAP: u32 = 128;
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;

fn klein_decode_routes(provider_id: &str) -> mlx_gen::gen_core::Result<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(
        provider_id,
        DECODE_TILE_EDGES.iter().copied(),
        DECODE_OVERLAP,
    )
}

fn klein_overlay(spec: &LoadSpec) -> Option<String> {
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
    if !spec.components.is_empty() {
        axes.push("components");
    }
    (!axes.is_empty()).then(|| axes.join("-"))
}

pub(crate) fn klein_streamable(spec: &LoadSpec) -> bool {
    spec.offload_policy == OffloadPolicy::Sequential
        && spec.load_shape == LoadShape::DeferredMaterialization
        && spec.quantize.is_none()
        && klein_overlay(spec).is_none()
        && matches!(spec.weights, WeightsSource::Dir(_))
}

fn klein_provider_static(provider_id: &str) -> mlx_gen::gen_core::Result<&'static str> {
    match provider_id {
        crate::FLUX2_KLEIN_9B_ID => Ok(crate::FLUX2_KLEIN_9B_ID),
        crate::FLUX2_KLEIN_9B_EDIT_ID => Ok(crate::FLUX2_KLEIN_9B_EDIT_ID),
        crate::FLUX2_KLEIN_9B_KV_EDIT_ID => Ok(crate::FLUX2_KLEIN_9B_KV_EDIT_ID),
        _ => Err(CoreError::Unsupported(format!(
            "unknown FLUX.2 Klein memory provider {provider_id}"
        ))),
    }
}

/// Production calibration identity of one clean Klein route, keyed on (provider, artifact, tier).
///
/// `artifact_tag` is the admitted inventory's `KleinArtifactInventory::artifact_tag` and `tier`
/// its resolved DiT tier. The measured dense-base key [`KLEIN_MEMORY_CALIBRATION_FINGERPRINT`] is
/// returned unchanged for (`flux2_klein_9b`, `base`, bf16); every other cell is
/// `flux2-klein-9b-<tier>-mlx-shared-ladder-<artifact>-<route>-v1`, so the dense tagged artifacts
/// keep the strings they always had and the packed `rehost`/`kv-rehost` turnkeys gain one per
/// tier. Offload policy and load shape are deliberately not inputs (sc-22727): the identity names
/// the artifact the evidence was captured against, and `MemoryCalibrationIdentity::load_shape`
/// carries the materialization axis separately.
///
/// The `Err` for an unshipped tier is reachable only through this direct entry point: the
/// production path ([`klein_contract_for`]) takes `tier` from an admitted
/// `KleinArtifactInventory`, whose verification has already refused an NVFP4 artifact, so the
/// inventory stays the single fail-closed site for artifact tiers.
pub fn klein_production_calibration_fingerprint(
    provider_id: &str,
    artifact_tag: &str,
    tier: Option<Quant>,
) -> mlx_gen::gen_core::Result<String> {
    let tier = match tier {
        None => "bf16",
        Some(Quant::Q4) => "q4",
        Some(Quant::Q8) => "q8",
        Some(other) => {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: FLUX.2 Klein ships no {other:?} tier"
            )))
        }
    };
    let route = match provider_id {
        crate::FLUX2_KLEIN_9B_ID if artifact_tag == "base" && tier == "bf16" => {
            return Ok(KLEIN_MEMORY_CALIBRATION_FINGERPRINT.to_owned());
        }
        crate::FLUX2_KLEIN_9B_ID => "t2i",
        crate::FLUX2_KLEIN_9B_EDIT_ID => "edit",
        crate::FLUX2_KLEIN_9B_KV_EDIT_ID => "kv-edit",
        _ => {
            return Err(CoreError::Unsupported(format!(
                "unknown FLUX.2 Klein memory provider {provider_id}"
            )))
        }
    };
    Ok(format!(
        "flux2-klein-9b-{tier}-mlx-shared-ladder-{artifact_tag}-{route}-v1"
    ))
}

/// The production calibration identity an admitted Klein inventory publishes for `spec`, or
/// `None` when the load carries an overlay (adapters, control, IP-adapter, identity, external text
/// encoder, extra components) and therefore has no clean base cell to measure.
///
/// sc-22727 (epic sc-22723 E1/E4): this used to be published only when the load was streamable —
/// `Sequential + DeferredMaterialization + quantize.is_none()` — while the worker's resident rung
/// loads `Resident + EagerMaterialization`, so the resident anchor had no identity to bind and the
/// packed turnkeys were handed the weights-free registry string in production.
///
/// The tier is the admitted artifact's (`KleinArtifactInventory::resolved_quant`), never the
/// request knob: `validate_klein_load_axes` does not constrain `LoadSpec::quantize`, so a
/// `Some(_)` that disagrees with the artifact — a dense tagged base the loader would requantize
/// at runtime, or a packed turnkey asked for the other tier — has no measured cell and publishes
/// `None` rather than the requested tier's string.
pub(crate) fn klein_production_calibration(
    provider_id: &str,
    spec: &LoadSpec,
    inventory: &crate::artifact_inventory::KleinArtifactInventory,
) -> mlx_gen::gen_core::Result<Option<MemoryCalibrationIdentity>> {
    if klein_overlay(spec).is_some() {
        return Ok(None);
    }
    let tier = inventory.resolved_quant();
    if spec
        .quantize
        .is_some_and(|requested| Some(requested) != tier)
    {
        return Ok(None);
    }
    let fingerprint =
        klein_production_calibration_fingerprint(provider_id, inventory.artifact_tag(), tier)?;
    Ok(Some(MemoryCalibrationIdentity::new(
        fingerprint,
        spec.load_shape,
    )))
}

/// Provider-local identity of Klein's resident LoRA/LoKr factor stack.
const KLEIN_ADAPTER_COMPONENT_ID: &str = "flux2_klein.adapters.forward_residuals";

/// Load-exact bytes of the adapter stack a Klein load keeps resident.
///
/// `apply_flux2_adapters` routes into the shared `apply_adapters_strict`, which installs
/// `AdaptableLinear` residuals evaluated as `base(x) + sum(adapter.residual(x))` and never mutates
/// the packed base — [`AdapterResidencyMode::Additive`]. `None` from the shared helper means an
/// additive stack was requested and at least one source could not be sized; fail closed on it
/// rather than declare a zero the shared validator would wave through.
fn klein_adapter_bytes(spec: &LoadSpec) -> mlx_gen::gen_core::Result<u64> {
    mlx_gen::gen_core::adapter_stack_resident_bytes(&spec.adapters, AdapterResidencyMode::Additive)
        .ok_or_else(|| {
            mlx_gen::gen_core::Error::Unsupported(
                "flux2_klein: an adapter stack was requested but at least one source could not be \
                 sized; refusing to declare a zero the shared validator would wave through"
                    .to_owned(),
            )
        })
}

fn build_klein_contract(
    provider_id: &str,
    spec: &LoadSpec,
    footprint: mlx_gen::PerComponentBytes,
    streamable: bool,
    calibration: Option<MemoryCalibrationIdentity>,
    adapter_bytes: u64,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    let staged = spec.offload_policy == OffloadPolicy::Sequential;
    let clean = klein_overlay(spec).is_none();
    klein_provider_static(provider_id)?;
    let routes = klein_decode_routes(provider_id)?;
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.architecture_facts = architecture_facts(&crate::config::Flux2Config::klein_9b());
    contract.load_shape = spec.load_shape;
    contract.calibration = calibration;
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
        MemoryFormulaVariable::OverlayBytes,
        MemoryFormulaVariable::DecodeTileArea,
        MemoryFormulaVariable::TransformerWindowSize,
    ];
    // SC-22667 (E1): `klein_overlay` lists `adapters` as a live Klein axis and
    // `validate_klein_load_axes` deliberately does NOT reject it, so an adapted Klein load is a
    // reachable composition. `load_flux2_heavy` then installs the stack as forward-time residuals
    // over the (possibly quantized) transformer — the crate's own comment says so — i.e. genuinely
    // extra resident bytes, never folded into the base. Adapter files live outside the snapshot
    // tree `from_spec_subdirs` walks, so `overlay_bytes` stayed 0 while the contract already
    // declared `OverlayBytes` as a formula input. The component axis is claimed only where there
    // IS an overlay: the shared validator refuses a zero-byte component declaration.
    contract.asset_facts.overlay_bytes = adapter_bytes;
    contract.formula = if adapter_bytes == 0 {
        MemoryFormulaKind::PhaseEnvelope { phases, variables }
    } else {
        MemoryFormulaKind::ComponentPhaseEnvelope {
            phases,
            variables,
            resident_components: vec![MemoryResidentComponent {
                id: KLEIN_ADAPTER_COMPONENT_ID.to_owned(),
                kind: MemoryComponentKind::AdapterStack,
                resident_bytes: adapter_bytes,
                // No published Klein rung bounds the adapter factors: rung 4's window covers the
                // transformer blocks, and an adapted load withdraws rungs 2-4 anyway.
                bounded_by: None,
                residency: MemoryComponentResidency::WholeRender,
            }],
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
        synchronized_phase_release: staged,
        decode_tiling: clean,
        attention_chunking: clean,
        transformer_window_materialization: streamable,
    };
    if spec.pid.is_some() {
        contract.pid_decode_routes = Some(mlx_gen::gen_core::MemoryPidDecodeRoutes {
            native: mlx_gen::gen_core::MemoryDecodeRouteDomain {
                tile_edges: routes.native_edges().to_vec(),
                tile_overlap: DECODE_OVERLAP,
            },
            pid: mlx_gen::gen_core::MemoryDecodeRouteDomain {
                tile_edges: mlx_gen_pid::DecodeRoutes::pid_edges(),
                tile_overlap: mlx_gen_pid::DecodeRoutes::pid_overlap(),
            },
        });
    }
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
            MemoryStrategy::StagedResidency if staged => MemoryStrategySupport::Implemented,
            MemoryStrategy::BoundedDecode if clean => {
                if contract.pid_decode_routes.is_some() {
                    capability.parameters.decode_tile_edges = routes.published_edges();
                    capability.parameters.decode_overlaps = routes.published_overlaps();
                } else {
                    capability.parameters.decode_tile_edges = routes.native_edges().to_vec();
                    capability.parameters.decode_overlaps = vec![DECODE_OVERLAP];
                }
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedAttention if clean => {
                capability.parameters.attention_chunk_sizes = vec![ATTENTION_CHUNK_SIZE];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency if streamable => {
                capability.parameters.transformer_window_sizes = vec![TRANSFORMER_WINDOW_SIZE];
                capability.parameters.transformer_window_components =
                    vec![TransformerComponent::Dit];
                MemoryStrategySupport::Implemented
            }
            _ => MemoryStrategySupport::Missing,
        };
    }
    contract.additional_prerequisites.push((
        MemoryStrategy::BoundedTransformerResidency,
        MemoryStrategyPrerequisite::Rung {
            rung: MemoryStrategy::StagedResidency,
            scope: MemoryPrerequisiteScope::EngagedInSameRequest,
        },
    ));
    Ok(contract)
}

pub fn klein_contract_for(
    provider_id: &str,
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    crate::model::validate_klein_load_axes(spec, provider_id)
        .map_err(|error| CoreError::Unsupported(error.to_string()))?;
    let inventory =
        crate::artifact_inventory::KleinArtifactInventory::verify_for_provider(provider_id, spec)?;
    klein_contract_with_inventory(provider_id, spec, inventory.as_ref())
}

/// The production contract for an already-admitted (or absent) inventory — everything
/// [`klein_contract_for`] does after artifact verification.
fn klein_contract_with_inventory(
    provider_id: &str,
    spec: &LoadSpec,
    inventory: Option<&crate::artifact_inventory::KleinArtifactInventory>,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    // The caller already verified the inventory: the registry footprint callbacks
    // (`component_footprint` and friends) re-run `verify_for_provider`, which re-walks all three
    // component directories and re-parses every shard header, so price without them.
    let footprint = crate::model::klein_component_footprint(provider_id, spec)?;
    klein_contract_from_parts(provider_id, spec, inventory, footprint)
}

/// The rung-4 and calibration-identity seam of the production contract, over an inventory the
/// caller has verified and a footprint it has priced. Split out so the seam can be driven with a
/// sealed bounded fixture inventory that the full artifact and encoder contracts would refuse;
/// every production caller reaches it through [`klein_contract_for`].
pub(crate) fn klein_contract_from_parts(
    provider_id: &str,
    spec: &LoadSpec,
    inventory: Option<&crate::artifact_inventory::KleinArtifactInventory>,
    footprint: mlx_gen::PerComponentBytes,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    crate::model::validate_klein_load_axes(spec, provider_id)
        .map_err(|error| CoreError::Unsupported(error.to_string()))?;
    let streamable = inventory.is_some() && klein_streamable(spec);
    // The identity is a property of the admitted artifact and tier, not of the load shape: rung 4
    // stays gated on `streamable` above, the identity does not (sc-22727).
    let calibration = match inventory {
        Some(inventory) => klein_production_calibration(provider_id, spec, inventory)?,
        None => None,
    };
    build_klein_contract(
        provider_id,
        spec,
        footprint,
        streamable,
        calibration,
        klein_adapter_bytes(spec)?,
    )
}

pub fn klein_contract(spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    klein_contract_for(crate::FLUX2_KLEIN_9B_ID, spec)
}

pub(crate) fn weights_free_klein_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    build_klein_contract(
        provider_id,
        spec,
        Default::default(),
        klein_streamable(spec),
        Some(MemoryCalibrationIdentity::new(
            format!(
                "{KLEIN_STATIC_BEHAVIOR_FINGERPRINT}-{}",
                provider_id.replace('_', "-")
            ),
            spec.load_shape,
        )),
        // No adapter overlay either: sizing one means opening its checkpoint, and this path exists
        // to produce the declaration without touching a weight file.
        0,
    )
}

fn surface_selector_matches_spec(
    surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec,
) -> mlx_gen::gen_core::Result<()> {
    use mlx_gen::gen_core::MemoryContractSurfaceTier;

    let tier_matches = match surface.resolved_artifact_tier() {
        MemoryContractSurfaceTier::Bf16 => {
            surface.spec.precision == mlx_gen::Precision::Bf16 && surface.spec.quantize.is_none()
        }
        MemoryContractSurfaceTier::Q4 => surface.spec.quantize == Some(Quant::Q4),
        MemoryContractSurfaceTier::Q8 => surface.spec.quantize == Some(Quant::Q8),
        MemoryContractSurfaceTier::Nvfp4 => false,
    };
    let plain = surface.spec.adapters.is_empty()
        && surface.spec.control.is_none()
        && surface.spec.extra_controls.is_empty()
        && surface.spec.ip_adapter.is_none()
        && surface.spec.identity.is_none()
        && surface.spec.text_encoder.is_none()
        && surface.spec.components.is_empty()
        && surface.spec.pid.is_none();
    if tier_matches
        && surface.spec.precision == mlx_gen::Precision::Bf16
        && plain
        && matches!(surface.spec.weights, WeightsSource::Dir(_))
        && surface.selector.offload_policy == surface.spec.offload_policy
        && surface.selector.load_shape == surface.spec.load_shape
    {
        Ok(())
    } else {
        Err(CoreError::Unsupported(format!(
            "FLUX.2 Klein memory surface selector '{}' does not match its plain registry LoadSpec",
            surface.selector.id()
        )))
    }
}

/// Resolve the finite catalog surface from the already-selected artifact tier.
///
/// Packed Klein turnkeys reach production with `LoadSpec::quantize == None`: the tier is the
/// artifact's own (a packed DiT, and a Qwen3 tower admitted exactly as stored — dense in the
/// shipped q4/q8 tiers, sc-22727), never a load-time request. The generic witness keeps Q4/Q8 in the
/// synthetic `LoadSpec` solely to make the selector self-checking; this resolver consumes the typed
/// selector and never reinterprets it as load-time quantization.
pub(crate) fn weights_free_klein_surface_contract(
    provider_id: &str,
    surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    use mlx_gen::gen_core::MemoryContractSurfaceTier;

    klein_provider_static(provider_id)?;
    surface_selector_matches_spec(surface)?;
    let supported_tier = matches!(
        surface.resolved_artifact_tier(),
        MemoryContractSurfaceTier::Bf16
            | MemoryContractSurfaceTier::Q4
            | MemoryContractSurfaceTier::Q8
    );
    let streamable = supported_tier
        && surface.spec.offload_policy == OffloadPolicy::Sequential
        && surface.spec.load_shape == LoadShape::DeferredMaterialization;
    build_klein_contract(
        provider_id,
        &surface.spec,
        Default::default(),
        streamable,
        Some(MemoryCalibrationIdentity::new(
            format!(
                "{KLEIN_STATIC_BEHAVIOR_FINGERPRINT}-{}",
                provider_id.replace('_', "-")
            ),
            surface.spec.load_shape,
        )),
        0,
    )
}

fn klein_expected_tier(
    provider_id: &str,
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryNumericTier> {
    let quant = match spec.quantize {
        Some(quant) => Some(quant),
        None => {
            if let Some(inventory) =
                crate::artifact_inventory::KleinArtifactInventory::verify_for_provider(
                    provider_id,
                    spec,
                )?
            {
                inventory.resolved_quant()
            } else {
                let WeightsSource::Dir(root) = &spec.weights else {
                    return Err(CoreError::Unsupported(format!(
                        "{provider_id}: FLUX.2 Klein memory routes require a snapshot directory"
                    )));
                };
                match crate::loader::read_component_quant(&root.join("transformer"))
                    .map_err(|error| CoreError::Unsupported(error.to_string()))?
                {
                    Some(quant) if quant.bits == 4 && quant.group_size == 64 => Some(Quant::Q4),
                    Some(quant) if quant.bits == 8 && quant.group_size == 64 => Some(Quant::Q8),
                    Some(quant) => {
                        return Err(CoreError::Unsupported(format!(
                            "{provider_id}: unsupported packed transformer quantization bits={} group_size={}",
                            quant.bits, quant.group_size
                        )))
                    }
                    None => None,
                }
            }
        }
    };
    Ok(MemoryNumericTier {
        precision: spec.precision,
        quant,
        component_precision_floors: &[],
    })
}

pub(crate) fn klein_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    expected_tier: MemoryNumericTier,
) -> MemorySafetyDecision {
    let route_gate = || {
        klein_provider_static(&contract.provider_id)?;
        let reference_count = context.geometry.reference_count;
        let route_matches = match contract.provider_id.as_str() {
            crate::FLUX2_KLEIN_9B_ID => matches!(
                (&context.mode, reference_count),
                (MemoryMode::TextToImage, 0) | (MemoryMode::ImageToImage, 1)
            ),
            crate::FLUX2_KLEIN_9B_EDIT_ID | crate::FLUX2_KLEIN_9B_KV_EDIT_ID => {
                context.mode == MemoryMode::Edit && (1..=8).contains(&reference_count)
            }
            _ => false,
        } && context.has_reference == (reference_count > 0);
        if !route_matches {
            return Err(CoreError::Unsupported(format!(
                "{}: memory route does not match the loaded provider mode/reference domain",
                contract.provider_id
            )));
        }
        if context.has_phases {
            return Err(CoreError::Unsupported(format!(
                "{}: FLUX.2 Klein memory routes are single-phase",
                contract.provider_id
            )));
        }
        if context.use_pid && spec.pid.is_none() {
            return Err(CoreError::Unsupported(format!(
                "{}: PiD route requested without loaded PiD weights",
                contract.provider_id
            )));
        }
        if context.overlay != klein_overlay(spec) {
            return Err(CoreError::Unsupported(
                "flux2 Klein memory overlay does not match the loaded route".to_owned(),
            ));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            klein_decode_routes(&contract.provider_id)?
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
        ) && !contract.lifecycle.transformer_window_materialization
        {
            return Err(CoreError::Unsupported(
                "flux2 Klein transformer streaming requires Sequential + DeferredMaterialization"
                    .to_owned(),
            ));
        }
        Ok(())
    };
    standard_memory_strategy_safety_check(contract, context, Some(expected_tier), Some(&route_gate))
}

pub(crate) fn registered_klein_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match klein_expected_tier(&contract.provider_id, spec) {
        Ok(tier) => klein_safety_check(spec, contract, context, tier),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub(crate) fn registered_klein_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> mlx_gen::gen_core::Result<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let tier = klein_expected_tier(&contract.provider_id, spec)?;
    let routes = match contract.provider_id.as_str() {
        crate::FLUX2_KLEIN_9B_ID => {
            vec![(MemoryMode::TextToImage, 0), (MemoryMode::ImageToImage, 1)]
        }
        crate::FLUX2_KLEIN_9B_EDIT_ID | crate::FLUX2_KLEIN_9B_KV_EDIT_ID => (1..=8)
            .map(|references| (MemoryMode::Edit, references))
            .collect(),
        provider_id => {
            return Err(CoreError::Unsupported(format!(
                "unknown FLUX.2 Klein memory provider {provider_id}"
            )))
        }
    };
    let permits_pid = spec.pid.is_some()
        && contract.pid_decode_routes.is_some()
        && contract.engages(strategy, MemoryStrategy::BoundedDecode);
    let mut fixtures = Vec::new();
    for (mode, reference_count) in routes {
        for use_pid in [false, true]
            .into_iter()
            .filter(|use_pid| !*use_pid || permits_pid)
        {
            let route = mlx_gen::gen_core::MemoryBehaviorRoute {
                mode: mode.clone(),
                reference_count,
                use_pid,
                has_phases: false,
                overlay: klein_overlay(spec),
            };
            let context = mlx_gen::gen_core::standard_memory_behavior_context(
                contract, strategy, tier, route,
            )?;
            fixtures.push(executable_klein_fixture(context));
        }
    }
    Ok(fixtures)
}

fn executable_klein_fixture(context: MemoryRunContext) -> mlx_gen::gen_core::MemoryBehaviorFixture {
    let mut fixture = mlx_gen::gen_core::MemoryBehaviorFixture::new(context);
    fixture.request.prompt = "weights-free FLUX.2 Klein memory behavior".to_owned();
    let reference = || mlx_gen::media::Image {
        width: 1,
        height: 1,
        pixels: vec![0, 0, 0],
    };
    fixture.request.conditioning = match (
        &fixture.context.mode,
        fixture.context.geometry.reference_count,
    ) {
        (MemoryMode::TextToImage, 0) => Vec::new(),
        (MemoryMode::ImageToImage, 1) => vec![mlx_gen::Conditioning::Reference {
            image: reference(),
            strength: Some(1.0),
        }],
        (MemoryMode::Edit, 1) => vec![mlx_gen::Conditioning::Reference {
            image: reference(),
            strength: None,
        }],
        (MemoryMode::Edit, count @ 2..=8) => vec![mlx_gen::Conditioning::MultiReference {
            images: (0..count).map(|_| reference()).collect(),
        }],
        _ => unreachable!("provider-owned route validation constructs only executable fixtures"),
    };
    fixture.request.phases = None;
    fixture
}

pub(crate) fn registered_klein_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_klein_request_with_cleanup(
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub(crate) fn begin_klein_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_klein_request_with_cleanup(
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

fn begin_klein_request_with_cleanup(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let expected_tier = klein_expected_tier(&contract.provider_id, spec)?;
    if let MemorySafetyDecision::Reject { reason } =
        klein_safety_check(spec, contract, context, expected_tier)
    {
        return Err(CoreError::Unsupported(reason));
    }
    let routes = klein_decode_routes(&contract.provider_id)?;
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        klein_provider_static(&contract.provider_id)?,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        32,
        move |use_pid, edge, overlap| {
            routes
                .validate(use_pid, Some(edge), Some(overlap))
                .map_err(CoreError::Unsupported)
        },
    )?;
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

pub(crate) fn attention_plan(req: &GenerationRequest) -> AttentionPlan<'_> {
    match req.memory {
        Some(memory) if memory.chunk_attention => {
            AttentionPlan::budgeted(AttentionBudget::CONSTRAINED).with_cancel(&req.cancel)
        }
        _ => AttentionPlan::UNBOUNDED,
    }
}

pub(crate) fn transformer_window(req: &GenerationRequest) -> mlx_gen::Result<Option<usize>> {
    let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
        return Ok(None);
    };
    if memory
        .transformer_window_component
        .unwrap_or(TransformerComponent::Dit)
        != TransformerComponent::Dit
    {
        return Err(mlx_gen::Error::Unsupported(
            "flux2 Klein supports only the DiT transformer window component".to_owned(),
        ));
    }
    Ok(Some(
        memory
            .transformer_window_size
            .unwrap_or(TRANSFORMER_WINDOW_SIZE) as usize,
    ))
}

pub(crate) fn decode_tiling(req: &GenerationRequest) -> mlx_gen::Result<Option<TilingConfig>> {
    let Some(memory) = req.memory.filter(|memory| memory.tile_vae_decode) else {
        return Ok(None);
    };
    if req.cancel.is_cancelled() {
        return Err(mlx_gen::Error::Canceled);
    }
    Ok(Some(TilingConfig::spatial_only(
        memory.decode_tile_edge.unwrap_or(DECODE_TILE_EDGE) as i32,
        memory.decode_overlap.unwrap_or(DECODE_OVERLAP) as i32,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        MemoryBudget, MemoryCacheState, MemoryGeometry, MemoryNumericTier, MemorySelection,
    };
    use mlx_gen::{Precision, Quant};

    /// AC (SC-22662): every registered FLUX.2 route publishes the axes of the trunk it runs — the
    /// three Klein routes the 9B config, the three Dev routes the Dev config — and every contract
    /// passes the shared facts conformance check.
    #[test]
    fn architecture_facts_follow_each_variants_own_config() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let klein = mlx_gen::gen_core::MemoryArchitectureFacts {
            attention_heads: Some(32),
            head_dim: Some(128),
            // 8 double + 24 single blocks.
            transformer_blocks: Some(32),
            // `in_channels` 128 = `num_latent_channels` 32 x 2².
            patch_size: Some(2),
            latent_channels: Some(32),
            vae_spatial_scale: Some(8),
            vae_temporal_scale: None,
            // f32 activations over bf16 weights.
            activation_dtype_width: Some(4),
        };
        let dev = mlx_gen::gen_core::MemoryArchitectureFacts {
            attention_heads: Some(48),
            // 8 double + 48 single blocks.
            transformer_blocks: Some(56),
            ..klein
        };
        assert_eq!(
            architecture_facts(&crate::config::Flux2Config::klein_9b()),
            klein
        );
        assert_eq!(architecture_facts(&crate::config::Flux2Config::dev()), dev);

        for provider_id in [
            crate::config::FLUX2_KLEIN_9B_ID,
            crate::config::FLUX2_KLEIN_9B_EDIT_ID,
            crate::config::FLUX2_KLEIN_9B_KV_EDIT_ID,
        ] {
            let contract = weights_free_klein_contract(provider_id, &spec).unwrap();
            assert_eq!(contract.architecture_facts, klein, "{provider_id}");
            assert!(contract.architecture_facts.has_declared_architecture_axis());
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
        for contract in [
            registered_dev_contract(&spec).unwrap(),
            registered_dev_t2i_contract(&spec).unwrap(),
            registered_dev_control_contract(&spec).unwrap(),
        ] {
            assert_eq!(
                contract.architecture_facts, dev,
                "{} architecture facts",
                contract.provider_id
            );
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
    }

    fn context(total_gb: f64) -> MemoryRunContext {
        let contract = build_contract();
        let calibration = contract.calibration.expect("calibration");
        let bytes = |gb: f64| (gb * 1024.0 * 1024.0 * 1024.0).round() as u64;
        MemoryRunContext {
            optimization_authority: mlx_gen::gen_core::MemoryOptimizationAuthority::Calibrated,
            selection: MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: Default::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: &[],
                },
            },
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint,
            load_shape: calibration.load_shape,
            mode: MemoryMode::Edit,
            has_reference: true,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 2,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: bytes(total_gb),
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: bytes(81.0),
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "worker-owned-exact-evidence".to_owned(),
        }
    }

    fn dev_t2i_context(total_gb: f64) -> MemoryRunContext {
        let contract = build_dev_t2i_contract();
        let calibration = contract.calibration.expect("calibration");
        let mut context = context(total_gb);
        context.calibration_abi = calibration.abi;
        context.calibration_fingerprint = calibration.fingerprint;
        context.load_shape = calibration.load_shape;
        context.mode = MemoryMode::TextToImage;
        context.has_reference = false;
        context.geometry.reference_count = 0;
        context.predicted_peak_bytes = context.budget.total_bytes;
        context
    }

    #[test]
    fn dev_t2i_contract_is_distinct_conforming_and_resident_only() {
        let t2i = build_dev_t2i_contract();
        let edit = build_contract();

        assert_eq!(t2i.provider_id, FLUX2_DEV_ID);
        assert_eq!(edit.provider_id, FLUX2_DEV_EDIT_ID);
        assert_eq!(
            t2i.calibration.as_ref().unwrap().fingerprint,
            DEV_T2I_CALIBRATION_FINGERPRINT
        );
        assert_ne!(t2i.calibration, edit.calibration);
        assert!(t2i.conformance_errors().is_empty());
        assert!(t2i.strategies.iter().all(|capability| {
            capability.strategy == MemoryStrategy::Resident
                && capability.support == MemoryStrategySupport::Implemented
                || capability.strategy != MemoryStrategy::Resident
                    && capability.support == MemoryStrategySupport::Missing
        }));

        let mut non_resident = dev_t2i_context(96.0);
        non_resident.selection.strategy = MemoryStrategy::StagedResidency;
        let MemorySafetyDecision::Reject { reason } =
            dev_t2i_safety_check(&t2i, &non_resident, non_resident.selection.tier)
        else {
            panic!("a missing non-resident rung must reject");
        };
        assert!(
            reason.contains("cannot execute StagedResidency"),
            "{reason}"
        );
    }

    #[test]
    fn sequential_dev_contracts_publish_only_staged_residency_and_open_a_cleanup_scope() {
        let spec = LoadSpec::new(WeightsSource::Dir(Default::default()))
            .with_offload_policy(OffloadPolicy::Sequential);
        let t2i = build_dev_t2i_contract_for_spec(&spec);
        let edit = build_contract_for_spec(&spec);
        let control_spec = spec.clone().with_control(WeightsSource::File(
            "/nonexistent/flux2-dev-fun-controlnet-union.safetensors".into(),
        ));
        let control = build_dev_control_contract(&control_spec);
        for contract in [&t2i, &edit, &control] {
            assert!(contract.lifecycle.synchronized_phase_release);
            assert_eq!(
                contract
                    .capability(MemoryStrategy::StagedResidency)
                    .expect("complete ladder")
                    .support,
                MemoryStrategySupport::Implemented,
                "{} must expose the loaded sequential lifecycle",
                contract.provider_id
            );
            for strategy in [
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ] {
                assert_eq!(
                    contract
                        .capability(strategy)
                        .expect("complete ladder")
                        .support,
                    MemoryStrategySupport::Missing,
                    "{} must not borrow {strategy:?} evidence",
                    contract.provider_id
                );
            }
        }

        assert_eq!(
            build_dev_control_contract(&spec)
                .capability(MemoryStrategy::StagedResidency)
                .expect("complete ladder")
                .support,
            MemoryStrategySupport::Missing,
            "a control memory strategy must not be selectable before its overlay is configured"
        );
        let deferred_spec = spec
            .clone()
            .with_load_shape(LoadShape::DeferredMaterialization);
        assert_eq!(
            build_dev_t2i_contract_for_spec(&deferred_spec)
                .capability(MemoryStrategy::StagedResidency)
                .expect("complete ladder")
                .support,
            MemoryStrategySupport::Missing,
            "eager-only Dev calibration must not be relabelled as deferred-materialization evidence"
        );

        let mut context = dev_t2i_context(128.0);
        context.selection.strategy = MemoryStrategy::StagedResidency;
        context.optimization_authority = mlx_gen::gen_core::MemoryOptimizationAuthority::Estimated;
        context.load_shape = spec.load_shape;
        context.calibration_abi = t2i.calibration.as_ref().unwrap().abi;
        context.calibration_fingerprint = t2i.calibration.as_ref().unwrap().fingerprint.clone();
        let mut request = GenerationRequest::default();
        let mut scope = begin_dev_request(&t2i, &context, context.selection.tier)
            .unwrap()
            .expect("staged selection must open a request scope");
        scope.configure_request(&mut request).unwrap();
        assert!(request.memory.expect("configured request").stage_residency);
        scope
            .finish(mlx_gen::gen_core::MemoryRunOutcome::Canceled)
            .unwrap();

        let mut control_fixture =
            registered_dev_fixture(&control_spec, &control, MemoryStrategy::StagedResidency)
                .unwrap()
                .pop()
                .expect("configured control route must have a behavior fixture");
        assert!(matches!(
            registered_dev_control_safety_check(&control_spec, &control, &control_fixture.context),
            MemorySafetyDecision::Accept
        ));
        assert_eq!(
            control_fixture.context.overlay.as_deref(),
            Some(DEV_CONTROL_OVERLAY)
        );
        assert!(matches!(
            control_fixture.request.conditioning.as_slice(),
            [mlx_gen::Conditioning::Control {
                kind: mlx_gen::ControlKind::Pose,
                ..
            }]
        ));
        assert_eq!(
            control_fixture.request.image_reference_count(),
            control_fixture.context.geometry.reference_count,
            "the control map must be part of the exact admitted request geometry"
        );
        let mut crossed_control_context = control_fixture.context.clone();
        crossed_control_context.overlay = None;
        assert!(matches!(
            registered_dev_control_safety_check(&control_spec, &control, &crossed_control_context),
            MemorySafetyDecision::Reject { .. }
        ));
        let mut control_scope =
            registered_dev_begin_request(&control_spec, &control, &control_fixture.context)
                .unwrap()
                .expect("configured control staging must open a request scope");
        control_scope
            .configure_request(&mut control_fixture.request)
            .unwrap();
        control_scope
            .finish(mlx_gen::gen_core::MemoryRunOutcome::Error {
                message: "weights-free cleanup probe".to_owned(),
            })
            .unwrap();
    }

    #[test]
    fn dev_t2i_route_gate_accepts_only_reference_free_text_to_image() {
        let contract = build_dev_t2i_contract();
        let exact = dev_t2i_context(96.0);
        assert_eq!(
            dev_t2i_safety_check(&contract, &exact, exact.selection.tier),
            MemorySafetyDecision::Accept
        );

        let mut edit_shaped = exact.clone();
        edit_shaped.mode = MemoryMode::Edit;
        edit_shaped.has_reference = true;
        edit_shaped.geometry.reference_count = 2;
        let MemorySafetyDecision::Reject { reason } =
            dev_t2i_safety_check(&contract, &edit_shaped, edit_shaped.selection.tier)
        else {
            panic!("the edit route must not inherit the T2I contract");
        };
        assert!(reason.contains("reference-free text-to-image"), "{reason}");

        let mut referenced_t2i = exact.clone();
        referenced_t2i.has_reference = true;
        referenced_t2i.geometry.reference_count = 1;
        let MemorySafetyDecision::Reject { reason } =
            dev_t2i_safety_check(&contract, &referenced_t2i, referenced_t2i.selection.tier)
        else {
            panic!("a referenced request must not enter the base T2I evidence lane");
        };
        assert!(reason.contains("reference-free text-to-image"), "{reason}");
    }

    #[test]
    fn dev_t2i_handshake_numeric_tier_and_quantization_fail_closed() {
        let contract = build_dev_t2i_contract();
        let exact = dev_t2i_context(96.0);

        let mut stale = exact.clone();
        stale.calibration_fingerprint = "stale-flux2-dev-t2i".to_owned();
        let MemorySafetyDecision::Reject { reason } =
            dev_t2i_safety_check(&contract, &stale, stale.selection.tier)
        else {
            panic!("stale T2I evidence must reject");
        };
        assert!(
            reason.contains("calibration handshake mismatch"),
            "{reason}"
        );

        let q8 = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q8),
            component_precision_floors: &[],
        };
        let MemorySafetyDecision::Reject { reason } = dev_t2i_safety_check(&contract, &exact, q8)
        else {
            panic!("the selected tier must match the loaded T2I artifact");
        };
        assert!(reason.contains("does not match loaded tier"), "{reason}");

        let mut nvfp4 = exact;
        nvfp4.selection.tier.quant = Some(Quant::Nvfp4);
        let MemorySafetyDecision::Reject { reason } =
            dev_t2i_safety_check(&contract, &nvfp4, nvfp4.selection.tier)
        else {
            panic!("the unwired NVFP4 tier must reject");
        };
        assert!(reason.contains("NVFP4 is not implemented"), "{reason}");
    }

    #[test]
    fn provider_safety_uses_the_caller_owned_peak_without_recomputing_it() {
        let contract = build_contract();
        let mut exact = context(81.0);
        assert_eq!(
            safety_check(&contract, &exact, exact.selection.tier),
            MemorySafetyDecision::Accept
        );
        exact.budget.total_bytes -= 1;
        assert!(matches!(
            safety_check(&contract, &exact, exact.selection.tier),
            MemorySafetyDecision::Reject { .. }
        ));

        let mut four = context(81.0);
        four.geometry.reference_count = 4;
        assert_eq!(
            safety_check(&contract, &four, four.selection.tier),
            MemorySafetyDecision::Accept
        );
    }

    #[test]
    fn provider_safety_rejects_a_stale_calibration_identity() {
        let contract = build_contract();
        let mut stale = context(96.0);
        stale.calibration_fingerprint = "stale".to_owned();
        assert!(matches!(
            safety_check(&contract, &stale, stale.selection.tier),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn provider_contract_quarantines_structured_overlays_and_false_reference_summaries() {
        let contract = build_contract();
        assert!(contract.strategies.iter().all(|capability| {
            capability.strategy == MemoryStrategy::Resident
                || capability.support == MemoryStrategySupport::Missing
        }));

        let mut structured_overlay = context(128.0);
        structured_overlay.overlay = Some("references=2".to_owned());
        let MemorySafetyDecision::Reject { reason } = safety_check(
            &contract,
            &structured_overlay,
            structured_overlay.selection.tier,
        ) else {
            panic!("structured overlay data must reject");
        };
        assert!(reason.contains("overlay is an identity axis"), "{reason}");

        let mut inconsistent = context(128.0);
        inconsistent.has_reference = false;
        let MemorySafetyDecision::Reject { reason } =
            safety_check(&contract, &inconsistent, inconsistent.selection.tier)
        else {
            panic!("inconsistent compatibility summary must reject");
        };
        assert!(
            reason.contains("inconsistent with reference_count=2"),
            "{reason}"
        );
    }

    #[test]
    fn provider_safety_owns_tier_identity_but_not_tier_peak_estimation() {
        let contract = build_contract();
        let q4_tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        };
        for quant in [Some(Quant::Q8), None] {
            let mut larger_tier = context(128.0);
            larger_tier.selection.tier.quant = quant;
            let expected_tier = larger_tier.selection.tier;
            assert_eq!(
                safety_check(&contract, &larger_tier, expected_tier),
                MemorySafetyDecision::Accept
            );
            assert!(matches!(
                safety_check(&contract, &larger_tier, q4_tier),
                MemorySafetyDecision::Reject { .. }
            ));
        }
    }

    #[test]
    fn shared_rejections_keep_their_reason_before_provider_policy_and_budget_advice() {
        let contract = build_contract();

        let mut stale_and_wrong_route = context(1.0);
        stale_and_wrong_route.calibration_fingerprint = "stale".to_owned();
        stale_and_wrong_route.mode = MemoryMode::TextToImage;
        let MemorySafetyDecision::Reject { reason } = safety_check(
            &contract,
            &stale_and_wrong_route,
            stale_and_wrong_route.selection.tier,
        ) else {
            panic!("stale handshake must reject");
        };
        assert!(
            reason.contains("calibration handshake mismatch"),
            "{reason}"
        );
        assert!(!reason.contains("Lower the output resolution"), "{reason}");

        let mut wrong_tier = context(1.0);
        wrong_tier.selection.tier.quant = Some(Quant::Q8);
        let q4 = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        };
        let MemorySafetyDecision::Reject { reason } = safety_check(&contract, &wrong_tier, q4)
        else {
            panic!("wrong tier must reject");
        };
        assert!(reason.contains("does not match loaded tier"), "{reason}");
        assert!(!reason.contains("Lower the output resolution"), "{reason}");

        let mut invalid_selection = context(1.0);
        invalid_selection.selection.parameters.decode_tile_edge = Some(512);
        let MemorySafetyDecision::Reject { reason } = safety_check(
            &contract,
            &invalid_selection,
            invalid_selection.selection.tier,
        ) else {
            panic!("invalid selection must reject");
        };
        assert!(reason.contains("decode_tile_edge"), "{reason}");
        assert!(!reason.contains("Lower the output resolution"), "{reason}");

        let mut wrong_route = context(1.0);
        wrong_route.mode = MemoryMode::TextToImage;
        let MemorySafetyDecision::Reject { reason } =
            safety_check(&contract, &wrong_route, wrong_route.selection.tier)
        else {
            panic!("provider route policy must reject");
        };
        assert!(reason.contains("referenced edit"), "{reason}");
        assert!(!reason.contains("Lower the output resolution"), "{reason}");

        let admitted_route = context(1.0);
        let MemorySafetyDecision::Reject { reason } =
            safety_check(&contract, &admitted_route, admitted_route.selection.tier)
        else {
            panic!("under-budget request must reject");
        };
        assert!(reason.contains("Lower the output resolution"), "{reason}");
    }

    /// Feature-end review (SC-22667, E1): the Dev and Dev-Edit contracts published
    /// `MemoryAssetFacts::default()` on every path, the loaded one included, while the load holds the
    /// whole DiT, the Mistral-3 tower and the VAE. They now publish the loader's own registry
    /// footprint whenever the spec names a materialized snapshot the footprint can price, and stay
    /// empty where there is nothing to read or the footprint refuses the snapshot.
    ///
    /// The mapping is exercised through the injectable seam because the production footprint needs
    /// a complete Dev snapshot on disk. Mutations that fail this: dropping the `materialized_root`
    /// gate publishes the stub's 12 bytes for the placeholder path; mis-wiring any of the three
    /// fields or `base_bytes` reds the materialized leg; turning a footprint error into a refusal
    /// (or into non-zero facts) reds the third leg; and passing `Default::default()` instead of
    /// `dev_asset_facts(..)` at either production builder reds the registered-contract leg.
    #[test]
    fn dev_contracts_publish_the_footprint_only_for_a_materialized_snapshot() {
        let priced = |_: &str, _: &LoadSpec| {
            Ok(mlx_gen::PerComponentBytes {
                text_encoder: 3,
                dit: 5,
                vae: 4,
            })
        };
        let refused = |provider_id: &str, _: &LoadSpec| {
            Err(CoreError::Unsupported(format!(
                "{provider_id}: fixture snapshot refused"
            )))
        };

        // No materialized snapshot — the registry's contract surface and every placeholder-path
        // caller. Nothing to read, so the declaration stays empty whatever a footprint would say.
        let placeholder = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert_eq!(
            dev_asset_facts_with(priced, FLUX2_DEV_ID, &placeholder),
            MemoryAssetFacts::default()
        );

        // A materialized snapshot the footprint prices publishes exactly that split, base included.
        let materialized = tempfile::tempdir().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(materialized.path().to_path_buf()));
        assert_eq!(
            dev_asset_facts_with(priced, FLUX2_DEV_ID, &spec),
            MemoryAssetFacts {
                base_bytes: 12,
                conditioning_bytes: 3,
                transformer_bytes: 5,
                decoder_bytes: 4,
                overlay_bytes: 0,
            }
        );

        // A materialized snapshot the footprint refuses keeps the empty declaration: the registered
        // contract is read before any load, and the tier rejection behind it must stay reachable.
        assert_eq!(
            dev_asset_facts_with(refused, FLUX2_DEV_ID, &spec),
            MemoryAssetFacts::default()
        );

        // The production builders are wired to the derivation: an empty materialized directory is
        // exactly the refused shape (the Dev footprint fails closed short of a complete snapshot),
        // so both registered contracts still build and still declare nothing.
        assert!(crate::model::dev_component_footprint(&spec).is_err());
        for contract in [
            registered_dev_contract(&spec).unwrap(),
            registered_dev_t2i_contract(&spec).unwrap(),
            registered_dev_contract(&placeholder).unwrap(),
            registered_dev_t2i_contract(&placeholder).unwrap(),
        ] {
            assert_eq!(
                contract.asset_facts,
                MemoryAssetFacts::default(),
                "{}",
                contract.provider_id
            );
            assert!(contract.conformance_errors().is_empty());
        }
    }

    /// A complete Dev snapshot the **production** `dev_component_footprint_for` accepts, written
    /// against the real encoder contracts. The weight payload is a sparse `set_len` hole, so a
    /// logically multi-gigabyte tower costs no disk: the footprint reads safetensors HEADERS and
    /// never the payload. `write_typed_safetensors` is the same tiny-component trick `model.rs`
    /// uses for the generic transformer/VAE accounting.
    fn complete_dev_snapshot(fixture: &std::path::Path) -> LoadSpec {
        let root = fixture.join("base");
        gen_core_testkit::write_multimodal_encoder_contract_fixture(
            &root.join("text_encoder"),
            crate::config::DEV_ENCODER_CONTRACT,
            crate::config::DEV_VISION_ENCODER_CONTRACT,
        )
        .unwrap();
        for component in ["transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            let header = br#"{"probe":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            bytes.extend([0_u8; 2]);
            std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
        }
        // No `quantization` section: the base tier is dense, so the selected language tower is
        // admitted dense and `effective_base_quant` resolves to `None`.
        std::fs::write(root.join("transformer/config.json"), "{}").unwrap();
        LoadSpec::new(WeightsSource::Dir(root))
    }

    /// Feature-end review (SC-22667, E1): the injected-seam test above proves the MAPPING, but
    /// every leg it drives through a production builder lands on the empty declaration, so nothing
    /// proved the production footprint can be ACCEPTED at all — a `dev_asset_facts` permanently
    /// stuck on its `Err` arm would have passed the whole file. This is the positive leg: a
    /// complete snapshot, the real `dev_component_footprint_for`, non-zero facts out of the
    /// registered contract.
    ///
    /// It also pins issue 3 of the same review: Dev-Control was left on the all-zero fallback while
    /// its own registered footprint callback sat one call away. Its conditioning must be the
    /// language tower ALONE — `dev_control_component_footprint` passes
    /// `include_builtin_multimodal: false` — so it is strictly smaller than T2I's, which proves the
    /// control route is wired to its OWN footprint rather than borrowing Dev's.
    ///
    /// Mutations that fail this:
    /// * returning `MemoryAssetFacts::default()` from `dev_asset_facts_with`'s `Ok` arm — every
    ///   `assert!(.. > 0)` reds;
    /// * dropping the `contract.asset_facts = dev_asset_facts_with(..)` line from
    ///   `build_dev_control_contract` — the control legs red back to zero;
    /// * wiring Dev-Control to `dev_component_footprint_for` instead of
    ///   `dev_control_component_footprint` — the builtin multimodal surface is charged and the
    ///   strict inequality reds;
    /// * mis-summing `base_bytes` — the three equality legs red.
    #[test]
    fn the_production_dev_footprint_is_accepted_and_published_for_a_complete_snapshot() {
        let fixture = tempfile::tempdir().unwrap();
        let spec = complete_dev_snapshot(fixture.path());

        for (provider_id, contract) in [
            (FLUX2_DEV_ID, registered_dev_t2i_contract(&spec).unwrap()),
            (FLUX2_DEV_EDIT_ID, registered_dev_contract(&spec).unwrap()),
            (
                FLUX2_DEV_CONTROL_ID,
                registered_dev_control_contract(&spec).unwrap(),
            ),
        ] {
            let footprint = if provider_id == FLUX2_DEV_CONTROL_ID {
                crate::model::dev_control_component_footprint(&spec)
            } else {
                crate::model::dev_component_footprint_for(provider_id, &spec)
            }
            .unwrap_or_else(|error| {
                panic!("{provider_id}: the production footprint must accept a complete Dev snapshot: {error}")
            });

            let facts = contract.asset_facts;
            assert!(
                facts.conditioning_bytes > 0
                    && facts.transformer_bytes > 0
                    && facts.decoder_bytes > 0,
                "{provider_id}: every phase of a complete snapshot is priced"
            );
            assert_eq!(
                facts.conditioning_bytes, footprint.text_encoder,
                "{provider_id}"
            );
            assert_eq!(facts.transformer_bytes, footprint.dit, "{provider_id}");
            assert_eq!(facts.decoder_bytes, footprint.vae, "{provider_id}");
            assert_eq!(
                facts.base_bytes,
                facts.conditioning_bytes + facts.transformer_bytes + facts.decoder_bytes,
                "{provider_id}: base_bytes is the split's sum"
            );
            assert_eq!(
                facts.overlay_bytes, 0,
                "{provider_id}: no adapter stack is configured"
            );
            assert!(contract.conformance_errors().is_empty(), "{provider_id}");
        }

        // The control route omits the builtin Pixtral + projector surface that Dev and Dev-Edit
        // retain for caption upsampling, so it must price a STRICTLY smaller conditioning phase.
        assert!(
            registered_dev_control_contract(&spec)
                .unwrap()
                .asset_facts
                .conditioning_bytes
                < registered_dev_t2i_contract(&spec)
                    .unwrap()
                    .asset_facts
                    .conditioning_bytes,
            "the control route must publish its own multimodal-free footprint"
        );
    }

    /// Feature-end review (SC-22667, E1): `klein_overlay` lists `adapters` as a live Klein axis and
    /// `validate_klein_load_axes` deliberately does not reject it, so an adapted Klein load is a
    /// reachable composition. `load_flux2_heavy` installs the stack as forward-time residuals over
    /// the (possibly quantized) transformer — never folded — and those files sit outside the
    /// snapshot subtree the footprint walks, so `overlay_bytes` stayed 0 while the contract already
    /// declared `OverlayBytes` as a formula input.
    ///
    /// Mutation that fails this: passing `0` instead of `klein_adapter_bytes(spec)?` at the
    /// production call site — `overlay_bytes` drops to 0, no component is declared, and the formula
    /// falls back to `PhaseEnvelope`.
    #[test]
    fn a_klein_adapter_stack_is_priced_as_a_resident_overlay() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);

        let clean = build_klein_contract(
            crate::FLUX2_KLEIN_9B_ID,
            &spec,
            Default::default(),
            true,
            None,
            0,
        )
        .unwrap();
        assert_eq!(clean.asset_facts.overlay_bytes, 0);
        assert!(clean.resident_components().is_empty());
        assert!(matches!(
            clean.formula,
            MemoryFormulaKind::PhaseEnvelope { .. }
        ));

        let adapted = build_klein_contract(
            crate::FLUX2_KLEIN_9B_ID,
            &spec,
            Default::default(),
            true,
            None,
            2048,
        )
        .unwrap();
        assert_eq!(adapted.asset_facts.overlay_bytes, 2048);
        assert_eq!(adapted.auxiliary_resident_bytes(), 2048);
        assert_eq!(
            adapted.asset_facts.base_bytes, clean.asset_facts.base_bytes,
            "an adapter is auxiliary: it must never move the base decomposition"
        );
        let components = adapted.resident_components();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].id, KLEIN_ADAPTER_COMPONENT_ID);
        assert_eq!(components[0].kind, MemoryComponentKind::AdapterStack);
        assert_eq!(components[0].resident_bytes, 2048);
        assert!(
            adapted.conformance_errors().is_empty(),
            "{:?}",
            adapted.conformance_errors()
        );

        // The sizing helper prices an additive stack at its on-disk length and fails closed on a
        // source it cannot size, rather than declaring a zero the shared validator would accept.
        let tmp = tempfile::tempdir().unwrap();
        let lora = tmp.path().join("lora.safetensors");
        std::fs::write(&lora, vec![0_u8; 2048]).unwrap();
        let mut sized = spec.clone();
        sized.adapters.push(mlx_gen::AdapterSpec::new(
            lora,
            1.0,
            mlx_gen::AdapterKind::Lora,
        ));
        assert_eq!(klein_adapter_bytes(&sized).unwrap(), 2048);
        assert_eq!(klein_adapter_bytes(&spec).unwrap(), 0);

        let mut unsizable = spec;
        unsizable.adapters.push(mlx_gen::AdapterSpec::new(
            tmp.path().join("absent.safetensors"),
            1.0,
            mlx_gen::AdapterKind::Lora,
        ));
        assert!(klein_adapter_bytes(&unsizable).is_err());
    }

    #[test]
    fn klein_contract_declares_every_shared_rung_for_the_exact_structural_route() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        let contract = build_klein_contract(
            crate::FLUX2_KLEIN_9B_ID,
            &spec,
            Default::default(),
            true,
            Some(MemoryCalibrationIdentity::new(
                "fixture-v1",
                spec.load_shape,
            )),
            0,
        )
        .unwrap();
        assert!(
            contract.conformance_errors().is_empty(),
            "{:?}",
            contract.conformance_errors()
        );
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "{strategy:?}"
            );
        }
        let transformer = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(transformer.parameters.transformer_window_sizes, [1]);
        assert_eq!(
            transformer.parameters.transformer_window_components,
            [TransformerComponent::Dit]
        );
    }

    /// sc-22727: the identity table is keyed on (provider, artifact, tier); the measured dense-base
    /// key stays byte-identical for (`flux2_klein_9b`, `base`, bf16), the dense tagged strings
    /// keep their historical shape, and every cell is distinct from every other and from the
    /// weights-free registry strings.
    #[test]
    fn klein_production_identity_table_is_per_tier_and_preserves_the_measured_key() {
        assert_eq!(
            klein_production_calibration_fingerprint(crate::FLUX2_KLEIN_9B_ID, "base", None)
                .unwrap(),
            "flux2-klein-9b-bf16-mlx-shared-ladder-t2i-v1"
        );
        assert_eq!(
            klein_production_calibration_fingerprint(crate::FLUX2_KLEIN_9B_ID, "base", None)
                .unwrap(),
            KLEIN_MEMORY_CALIBRATION_FINGERPRINT
        );
        assert_eq!(
            klein_production_calibration_fingerprint(crate::FLUX2_KLEIN_9B_ID, "true-two", None)
                .unwrap(),
            "flux2-klein-9b-bf16-mlx-shared-ladder-true-two-t2i-v1"
        );
        let providers = [
            crate::FLUX2_KLEIN_9B_ID,
            crate::FLUX2_KLEIN_9B_EDIT_ID,
            crate::FLUX2_KLEIN_9B_KV_EDIT_ID,
        ];
        let mut published = std::collections::BTreeSet::new();
        for provider_id in providers {
            for artifact in ["base", "true-two", "rehost", "kv-rehost"] {
                for tier in [None, Some(Quant::Q4), Some(Quant::Q8)] {
                    let fingerprint =
                        klein_production_calibration_fingerprint(provider_id, artifact, tier)
                            .unwrap();
                    assert!(
                        published.insert(fingerprint.clone()),
                        "{provider_id} {artifact} {tier:?} collides on {fingerprint}"
                    );
                    assert_ne!(
                        fingerprint,
                        format!(
                            "{KLEIN_STATIC_BEHAVIOR_FINGERPRINT}-{}",
                            provider_id.replace('_', "-")
                        )
                    );
                }
            }
        }
        assert_eq!(published.len(), 3 * 4 * 3);
        // The weights-free registry surface never resolves to any production cell, under every
        // catalog surface and every registry entry point.
        for surface in mlx_gen::gen_core::mlx_memory_contract_surface_specs() {
            for provider_id in providers {
                let resolved = weights_free_klein_surface_contract(provider_id, &surface).unwrap();
                let fingerprint = resolved.calibration.unwrap().fingerprint;
                assert!(
                    !published.contains(&fingerprint),
                    "{provider_id} surface {} resolved to production identity {fingerprint}",
                    surface.selector.id()
                );
                let plain = weights_free_klein_contract(provider_id, &surface.spec).unwrap();
                let fingerprint = plain.calibration.unwrap().fingerprint;
                assert!(
                    !published.contains(&fingerprint),
                    "{provider_id} plain surface {} resolved to production identity {fingerprint}",
                    surface.selector.id()
                );
            }
        }
        assert!(
            klein_production_calibration_fingerprint("flux2_klein_alias", "base", None).is_err()
        );
        assert!(klein_production_calibration_fingerprint(
            crate::FLUX2_KLEIN_9B_ID,
            "rehost",
            Some(Quant::Nvfp4)
        )
        .is_err());
    }

    #[test]
    fn klein_rung_four_fails_closed_for_eager_quantized_and_overlay_loads() {
        let base = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        let lora = base.clone().with_adapters(vec![mlx_gen::AdapterSpec::new(
            "/nonexistent/lora.safetensors".into(),
            1.0,
            mlx_gen::AdapterKind::Lora,
        )]);
        let lokr = base.clone().with_adapters(vec![mlx_gen::AdapterSpec::new(
            "/nonexistent/lokr.safetensors".into(),
            1.0,
            mlx_gen::AdapterKind::Lokr,
        )]);
        let external_te = base
            .clone()
            .with_text_encoder(WeightsSource::Dir("/nonexistent/external-te".into()));
        for spec in [
            base.clone()
                .with_load_shape(LoadShape::EagerMaterialization),
            base.with_quant(Quant::Q4),
            lora,
            lokr,
            external_te,
        ] {
            let contract = build_klein_contract(
                crate::FLUX2_KLEIN_9B_ID,
                &spec,
                Default::default(),
                klein_streamable(&spec),
                None,
                0,
            )
            .unwrap();
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing
            );
        }
    }

    #[test]
    fn klein_selector_surfaces_are_exact_and_fail_closed_on_axis_mutation() {
        let providers = [
            crate::FLUX2_KLEIN_9B_ID,
            crate::FLUX2_KLEIN_9B_EDIT_ID,
            crate::FLUX2_KLEIN_9B_KV_EDIT_ID,
        ];
        for provider_id in providers {
            let mut implemented = 0;
            for surface in mlx_gen::gen_core::mlx_memory_contract_surface_specs() {
                let contract = weights_free_klein_surface_contract(provider_id, &surface).unwrap();
                let expected = surface.spec.offload_policy == OffloadPolicy::Sequential
                    && surface.spec.load_shape == LoadShape::DeferredMaterialization;
                assert_eq!(
                    contract
                        .capability(MemoryStrategy::BoundedTransformerResidency)
                        .unwrap()
                        .support,
                    if expected {
                        MemoryStrategySupport::Implemented
                    } else {
                        MemoryStrategySupport::Missing
                    },
                    "{provider_id}: {}",
                    surface.selector.id()
                );
                implemented += usize::from(expected);
                assert_eq!(contract.asset_facts, Default::default());
            }
            assert_eq!(implemented, 3, "{provider_id}");
        }

        let q4 = mlx_gen::gen_core::mlx_memory_contract_surface_specs()
            .into_iter()
            .find(|surface| {
                surface.resolved_artifact_tier() == mlx_gen::gen_core::MemoryContractSurfaceTier::Q4
                    && surface.spec.offload_policy == OffloadPolicy::Sequential
                    && surface.spec.load_shape == LoadShape::DeferredMaterialization
            })
            .unwrap();
        let mut mutations = Vec::new();
        let copy = || mlx_gen::gen_core::MemoryContractSurfaceSpec {
            selector: q4.selector,
            spec: q4.spec.clone(),
        };
        let mut tier = copy();
        tier.spec.quantize = Some(Quant::Q8);
        mutations.push(tier);
        let mut source = copy();
        source.spec.weights = WeightsSource::File("/klein.safetensors".into());
        mutations.push(source);
        let mut precision = copy();
        precision.spec.precision = Precision::Fp32;
        mutations.push(precision);
        let mut control = copy();
        control.spec.control = Some(WeightsSource::File("/control.safetensors".into()));
        mutations.push(control);
        let mut component = copy();
        component.spec.components.insert(
            "unknown".to_owned(),
            WeightsSource::File("/component.safetensors".into()),
        );
        mutations.push(component);
        let mut pid = copy();
        pid.spec = pid.spec.with_pid(
            WeightsSource::File("/pid.safetensors".into()),
            WeightsSource::Dir("/gemma".into()),
        );
        mutations.push(pid);
        for provider_id in providers {
            for mutation in &mutations {
                assert!(
                    weights_free_klein_surface_contract(provider_id, mutation).is_err(),
                    "{provider_id} admitted mutated selector {}",
                    mutation.selector.id()
                );
            }
        }
        assert!(weights_free_klein_surface_contract("flux2_klein_alias", &q4).is_err());
    }

    #[test]
    fn klein_behavior_routes_are_executable_reachable_and_fail_closed() {
        let q4 = mlx_gen::gen_core::mlx_memory_contract_surface_specs()
            .into_iter()
            .find(|surface| {
                surface.resolved_artifact_tier() == mlx_gen::gen_core::MemoryContractSurfaceTier::Q4
                    && surface.spec.offload_policy == OffloadPolicy::Sequential
                    && surface.spec.load_shape == LoadShape::DeferredMaterialization
            })
            .unwrap();
        for provider_id in [
            crate::FLUX2_KLEIN_9B_ID,
            crate::FLUX2_KLEIN_9B_EDIT_ID,
            crate::FLUX2_KLEIN_9B_KV_EDIT_ID,
        ] {
            let spec = q4.spec.clone().with_pid(
                WeightsSource::File("/pid.safetensors".into()),
                WeightsSource::Dir("/gemma".into()),
            );
            let contract = build_klein_contract(
                provider_id,
                &spec,
                Default::default(),
                true,
                Some(MemoryCalibrationIdentity::new(
                    format!(
                        "{KLEIN_STATIC_BEHAVIOR_FINGERPRINT}-{}",
                        provider_id.replace('_', "-")
                    ),
                    spec.load_shape,
                )),
                0,
            )
            .unwrap();
            let fixtures = registered_klein_fixture(
                &spec,
                &contract,
                MemoryStrategy::BoundedTransformerResidency,
            )
            .unwrap();
            let expected_routes = if provider_id == crate::FLUX2_KLEIN_9B_ID {
                4
            } else {
                16
            };
            assert_eq!(fixtures.len(), expected_routes, "{provider_id}");
            assert!(fixtures.iter().any(|fixture| fixture.context.use_pid));
            assert!(fixtures.iter().any(|fixture| !fixture.context.use_pid));

            let (is_edit, is_kv) = (
                provider_id != crate::FLUX2_KLEIN_9B_ID,
                provider_id == crate::FLUX2_KLEIN_9B_KV_EDIT_ID,
            );
            let descriptor = match provider_id {
                crate::FLUX2_KLEIN_9B_ID => crate::model::descriptor_klein_9b(),
                crate::FLUX2_KLEIN_9B_EDIT_ID => crate::model::descriptor_klein_9b_edit(),
                crate::FLUX2_KLEIN_9B_KV_EDIT_ID => crate::model::descriptor_klein_9b_kv_edit(),
                _ => unreachable!(),
            };
            for fixture in &fixtures {
                assert_eq!(
                    registered_klein_safety_check(&spec, &contract, &fixture.context),
                    MemorySafetyDecision::Accept,
                    "{provider_id}: {:?}",
                    fixture.context
                );
                let mut scope = registered_klein_begin_request(&spec, &contract, &fixture.context)
                    .unwrap()
                    .expect("optimized Klein fixture creates a request scope");
                let mut configured = fixture.request.clone();
                scope.configure_request(&mut configured).unwrap();
                let memory = configured.memory.expect("request memory controls");
                assert!(memory.stage_residency);
                assert!(memory.tile_vae_decode);
                assert!(memory.chunk_attention);
                assert!(memory.stream_transformer_blocks);
                assert_eq!(
                    memory.transformer_window_size,
                    Some(TRANSFORMER_WINDOW_SIZE)
                );
                assert_eq!(memory.attention_chunk_size, Some(ATTENTION_CHUNK_SIZE));
                if fixture.context.use_pid {
                    assert_eq!(memory.decode_tile_edge, Some(2048));
                    assert_eq!(memory.decode_overlap, Some(256));
                } else {
                    assert!(memory
                        .decode_tile_edge
                        .is_some_and(|edge| DECODE_TILE_EDGES.contains(&edge)));
                    assert_eq!(memory.decode_overlap, Some(DECODE_OVERLAP));
                }
                crate::model::validate_request(&descriptor, is_edit, is_kv, &fixture.request)
                    .unwrap_or_else(|error| {
                        panic!("{provider_id} fixture is not executable: {error}")
                    });
                assert_eq!(fixture.request.use_pid, fixture.context.use_pid);
                assert!(fixture.request.phases.is_none());
                assert!(!fixture.context.has_phases);
            }

            let exact = &fixtures[0].context;
            let mut mutations = Vec::new();
            let mut mode = exact.clone();
            mode.mode = if is_edit {
                MemoryMode::ImageToImage
            } else {
                MemoryMode::Edit
            };
            mutations.push(mode);
            let mut references = exact.clone();
            references.geometry.reference_count = if is_edit { 9 } else { 2 };
            references.has_reference = true;
            mutations.push(references);
            let mut phases = exact.clone();
            phases.has_phases = true;
            mutations.push(phases);
            let mut overlay = exact.clone();
            overlay.overlay = Some("references:1".to_owned());
            mutations.push(overlay);
            let mut tier = exact.clone();
            tier.selection.tier.quant = Some(Quant::Q8);
            mutations.push(tier);
            for mutation in mutations {
                assert!(matches!(
                    registered_klein_safety_check(&spec, &contract, &mutation),
                    MemorySafetyDecision::Reject { .. }
                ));
                assert!(registered_klein_begin_request(&spec, &contract, &mutation).is_err());
            }
        }
    }

    #[test]
    fn klein_admission_binds_provider_route_and_loaded_pid() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        let contract = weights_free_klein_contract(crate::FLUX2_KLEIN_9B_ID, &spec).unwrap();
        let tier = MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        };
        let route = mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        };
        let mut context = mlx_gen::gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::Resident,
            tier,
            route,
        )
        .unwrap();

        context.mode = MemoryMode::Edit;
        context.has_reference = true;
        context.geometry.reference_count = 1;
        let MemorySafetyDecision::Reject { reason } =
            klein_safety_check(&spec, &contract, &context, tier)
        else {
            panic!("base contract admitted an edit route");
        };
        assert!(
            reason.contains("provider mode/reference domain"),
            "{reason}"
        );

        context.mode = MemoryMode::TextToImage;
        context.has_reference = false;
        context.geometry.reference_count = 0;
        context.use_pid = true;
        let MemorySafetyDecision::Reject { reason } =
            klein_safety_check(&spec, &contract, &context, tier)
        else {
            panic!("base contract admitted PiD without loaded weights");
        };
        assert!(reason.contains("without loaded PiD weights"), "{reason}");
    }

    #[test]
    fn klein_contract_advertises_pid_decode_only_when_pid_weights_are_loaded() {
        let base = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        let native = weights_free_klein_contract(crate::FLUX2_KLEIN_9B_ID, &base).unwrap();
        assert!(native.pid_decode_routes.is_none());
        assert_eq!(
            native
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .parameters
                .decode_tile_edges,
            DECODE_TILE_EDGES
        );

        let pid_spec = base.with_pid(
            WeightsSource::File("/nonexistent/pid.safetensors".into()),
            WeightsSource::Dir("/nonexistent/gemma".into()),
        );
        let pid = weights_free_klein_contract(crate::FLUX2_KLEIN_9B_ID, &pid_spec).unwrap();
        assert!(pid.pid_decode_routes.is_some());
        let fixtures =
            registered_klein_fixture(&pid_spec, &pid, MemoryStrategy::BoundedDecode).unwrap();
        assert!(fixtures.iter().any(|fixture| !fixture.context.use_pid));
        assert!(fixtures.iter().any(|fixture| fixture.context.use_pid));
    }
}
