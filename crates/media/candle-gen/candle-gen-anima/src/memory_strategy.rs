//! Exact Candle/CUDA request-memory contract for the three Anima text-to-image routes (SC-20785).
//!
//! The only non-resident implementation is staged residency: Qwen3 + the bundled conditioner
//! produce immutable conditioning, are synchronized and dropped, then the DiT + VAE are loaded.
//! Decode tiling, attention chunking and transformer windows have no parity-safe implementation
//! here and are deliberately classified rather than implied by the shared ladder order.

use std::path::Path;

use candle_gen::gen_core::{
    self, LoadSpec, MemoryAssetFacts, MemoryBackendRealization, MemoryCalibrationIdentity,
    MemoryFormulaKind, MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode,
    MemoryNumericTier, MemoryPhase, MemoryProviderContract, MemoryRequestScope, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategySupport, MemoryWindowMaterialization,
    Precision, Quant,
};

use crate::config::Variant;

const IDS: &[&str] = &["anima_base", "anima_aesthetic", "anima_turbo"];
const FINGERPRINT: &str = "anima-candle-request-scoped-conditioning-v1";

/// Every Anima component is materialized at the native bf16 compute dtype
/// ([`resolved_numeric_tier`] refuses anything else), so each float element costs two bytes.
const FLOAT_WIDTH: u64 = 2;

fn variant_for(provider_id: &str) -> gen_core::Result<Variant> {
    match provider_id {
        "anima_base" => Ok(Variant::Base),
        "anima_aesthetic" => Ok(Variant::Aesthetic),
        "anima_turbo" => Ok(Variant::Turbo),
        _ => Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: not an Anima Candle memory provider"
        ))),
    }
}

/// Bytes one resolved component occupies once loaded: float tensors at the compute width, packed
/// (integer) codes at their stored width. Header-only — no tensor data is materialized.
fn component_bytes(path: &Path) -> gen_core::Result<u64> {
    gen_core::weightsmeta::safetensors_path_tensor_headers(path)?
        .iter()
        .try_fold(0_u64, |sum, header| {
            let bytes = if header.is_float() {
                header.materialized_bytes(FLOAT_WIDTH)?
            } else {
                header.data_bytes
            };
            sum.checked_add(bytes)
                .ok_or_else(|| gen_core::Error::Msg("anima: component byte sum overflow".into()))
        })
}

/// Load-exact component bytes read from the resolved `split_files/` inventory.
///
/// The DiT safetensors bundles the Cosmos DiT **and** the `AnimaTextConditioner`
/// (`{prefix}.llm_adapter.*`); both are transformer-phase residents loaded from that one file, so
/// they are priced together. `overlay_bytes` stays zero because Anima folds LoRA/LoKr into the DiT
/// (Resident) or refuses them (staged) and therefore never declares
/// [`MemoryFormulaVariable::OverlayBytes`].
fn asset_facts(spec: &LoadSpec, variant: Variant) -> gen_core::Result<MemoryAssetFacts> {
    let root = crate::loader::resolve_split_files(&spec.weights)?;
    let conditioning = component_bytes(&root.join(crate::loader::TEXT_ENCODER_FILE))?;
    let transformer = component_bytes(&root.join("diffusion_models").join(variant.dit_filename()))?;
    let decoder = component_bytes(&root.join(crate::loader::VAE_FILE))?;
    Ok(MemoryAssetFacts {
        base_bytes: conditioning
            .saturating_add(transformer)
            .saturating_add(decoder),
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: decoder,
        overlay_bytes: 0,
    })
}

/// The executable contract for a real Anima load: identical to [`weights_free_contract`] except
/// that its [`MemoryFormulaVariable::AssetBytes`] input is the exact on-disk component inventory
/// rather than a zero placeholder.
pub fn contract(provider_id: &str, spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let variant = variant_for(provider_id)?;
    let mut contract = weights_free_contract(provider_id, spec)?;
    contract.asset_facts = asset_facts(spec, variant)?;
    Ok(contract)
}

/// Registry-surface contract for a catalog build that has no resolvable weights root. It carries
/// the same shape and calibration identity as [`contract`] but no asset facts, so nothing can
/// mistake it for priced evidence.
pub fn weights_free_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    if !IDS.contains(&provider_id) {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: not an Anima Candle memory provider"
        )));
    }
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        decode_tiling: false,
        attention_chunking: false,
        transformer_window_materialization: false,
    };
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
        ],
    };
    // The three routes share an implementation but not calibration evidence.  A base calibration
    // must never admit aesthetic/turbo merely because their component layout happens to match.
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        format!("{FINGERPRINT}-{}", provider_id.replace('_', "-")),
        spec.load_shape,
    ));
    let staged = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::StagedResidency)
        .expect("compatibility contract contains every strategy");
    // Staging cannot reproduce a loaded LoRA/LoKr because its conditioner and DiT mutations span
    // the phase boundary. This classification applies only to the staged rung; Resident remains
    // available for the same adapter-bearing load surface.
    staged.support = if spec.adapters.is_empty() {
        MemoryStrategySupport::Implemented
    } else {
        MemoryStrategySupport::StructurallyNotApplicable {
            reason: "Anima adapter overlays span the conditioner and DiT load boundary; staged residency refuses them rather than applying a partial overlay".to_owned(),
        }
    };
    Ok(contract)
}

/// Anima's CUDA path is native bf16. A `Precision::Fp32` request is not an alternate loaded tier;
/// accepting it would label bf16 CUDA execution with false memory evidence.
pub fn resolved_numeric_tier(
    spec: &LoadSpec,
    physical_quant: Option<Quant>,
) -> gen_core::Result<MemoryNumericTier> {
    if spec.precision != Precision::Bf16 {
        return Err(gen_core::Error::Unsupported(
            "anima: Candle/CUDA memory residency supports only the native bf16 precision"
                .to_owned(),
        ));
    }
    Ok(MemoryNumericTier {
        precision: Precision::Bf16,
        quant: physical_quant,
        component_precision_floors: &[],
    })
}

pub fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match resolved_numeric_tier(spec, spec.quantize) {
        Ok(tier) => loaded_safety_check(
            gen_core::adapter_stack_identity(&spec.adapters).as_deref(),
            tier,
            contract,
            context,
        ),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub fn loaded_safety_check(
    expected_overlay: Option<&str>,
    loaded_tier: MemoryNumericTier,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let staged = contract.engages(context.selection.strategy, MemoryStrategy::StagedResidency);
    let route_gate = || {
        if context.mode != MemoryMode::TextToImage
            || context.geometry.reference_count != 0
            || context.has_reference
            || context.use_pid
            || context.has_phases
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: memory admission is only valid for the exact single-phase text_to_image route",
                contract.provider_id
            )));
        }
        if context.overlay.as_deref() != expected_overlay {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: memory overlay {:?} does not match the loaded adapter stack {:?}",
                contract.provider_id, context.overlay, expected_overlay
            )));
        }
        if staged && expected_overlay.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: staged residency refuses LoRA/LoKr overlays because their conditioner and DiT loads are one atomic artifact",
                contract.provider_id
            )));
        }
        Ok(())
    };
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(loaded_tier),
        Some(&route_gate),
    )
}

/// Provider-owned behavior fixture for every exact Anima Candle route.  The fixture is deliberately
/// weights-free but carries the same tier, route and calibration identity the worker will admit.
pub fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    Ok(vec![gen_core::MemoryBehaviorFixture::new(
        gen_core::standard_memory_behavior_context(
            contract,
            strategy,
            resolved_numeric_tier(spec, spec.quantize)?,
            gen_core::MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )?,
    )])
}

/// Weights-free executable admission seam.  It mirrors the loaded provider's validation before
/// creating the same request scope used by worker lifecycle telemetry.
pub fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(gen_core::Error::Unsupported(reason));
    }
    Ok(Some(Box::new(crate::AnimaMemoryScope::new(
        contract,
        context,
        candle_gen::candle_core::Device::Cpu,
    ))))
}

macro_rules! registration {
    ($name:ident, $id:literal) => {
        pub const $name: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
            provider_id: $id,
            contract: |spec| contract($id, spec),
            safety_check,
        };
    };
}

registration!(BASE_MEMORY_REGISTRATION, "anima_base");
registration!(AESTHETIC_MEMORY_REGISTRATION, "anima_aesthetic");
registration!(TURBO_MEMORY_REGISTRATION, "anima_turbo");

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{
        AdapterKind, AdapterSpec, MemoryBehaviorRoute, MemoryNumericTier, MemoryStrategy,
        Precision, WeightsSource,
    };

    #[test]
    fn all_three_exact_plain_routes_publish_only_staged_residency() {
        for provider in IDS {
            let contract = weights_free_contract(
                provider,
                &LoadSpec::new(WeightsSource::Dir("/anima".into())),
            )
            .unwrap();
            assert_eq!(
                contract
                    .capability(MemoryStrategy::StagedResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Implemented,
                "{provider}"
            );
            for strategy in [
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ] {
                assert_ne!(
                    contract.capability(strategy).unwrap().support,
                    MemoryStrategySupport::Implemented,
                    "{provider}: unsupported rung must not borrow staged evidence"
                );
            }
        }
    }

    #[test]
    fn staged_overlay_is_refused_but_resident_lora_and_lokr_remain_admitted() {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/anima".into()));
        spec.adapters.push(AdapterSpec::new(
            "/anima/overlay.safetensors".into(),
            1.0,
            AdapterKind::Lora,
        ));
        let adapter_contract = weights_free_contract("anima_base", &spec).unwrap();
        let lora_overlay = gen_core::adapter_stack_identity(&spec.adapters).unwrap();
        assert!(matches!(
            adapter_contract
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::StructurallyNotApplicable { .. }
        ));
        let resident = gen_core::standard_memory_behavior_context(
            &adapter_contract,
            MemoryStrategy::Resident,
            MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
            MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: Some(lora_overlay.clone()),
            },
        )
        .unwrap();
        assert_eq!(
            safety_check(&spec, &adapter_contract, &resident),
            MemorySafetyDecision::Accept
        );

        spec.adapters[0].kind = AdapterKind::Lokr;
        let lokr_overlay = gen_core::adapter_stack_identity(&spec.adapters).unwrap();
        let mut lokr_resident = resident.clone();
        lokr_resident.overlay = Some(lokr_overlay.clone());
        assert_eq!(
            safety_check(&spec, &adapter_contract, &lokr_resident),
            MemorySafetyDecision::Accept
        );
        assert!(matches!(
            safety_check(&spec, &adapter_contract, &resident),
            MemorySafetyDecision::Reject { .. }
        ));
        spec.adapters[0].scale = f32::from_bits(1.0_f32.to_bits() + 1);
        assert!(matches!(
            safety_check(&spec, &adapter_contract, &lokr_resident),
            MemorySafetyDecision::Reject { .. }
        ));
        spec.adapters[0].scale = 1.0;
        spec.adapters[0].path = "/anima/other.safetensors".into();
        assert!(matches!(
            safety_check(&spec, &adapter_contract, &lokr_resident),
            MemorySafetyDecision::Reject { .. }
        ));

        let plain = LoadSpec::new(WeightsSource::Dir("/anima".into()));
        let plain_contract = weights_free_contract("anima_base", &plain).unwrap();
        let mut staged =
            registered_valid_fixture(&plain, &plain_contract, MemoryStrategy::StagedResidency)
                .unwrap()
                .pop()
                .unwrap()
                .context;
        staged.overlay = Some("lokr:overlay".to_owned());
        assert!(matches!(
            safety_check(&spec, &adapter_contract, &staged),
            MemorySafetyDecision::Reject { .. }
        ));
        assert!(registered_valid_fixture(
            &spec,
            &adapter_contract,
            MemoryStrategy::StagedResidency
        )
        .is_err());
    }

    #[test]
    fn exact_route_tier_and_calibration_are_bound_before_admission() {
        let spec = LoadSpec::new(WeightsSource::Dir("/anima".into()));
        let base = weights_free_contract("anima_base", &spec).unwrap();
        let aesthetic = weights_free_contract("anima_aesthetic", &spec).unwrap();
        let context = registered_valid_fixture(&spec, &base, MemoryStrategy::StagedResidency)
            .unwrap()
            .pop()
            .unwrap()
            .context;
        assert_eq!(
            safety_check(&spec, &base, &context),
            MemorySafetyDecision::Accept
        );

        let mut crossed_tier = context.clone();
        crossed_tier.selection.tier.quant = Some(gen_core::Quant::Q4);
        assert!(matches!(
            safety_check(&spec, &base, &crossed_tier),
            MemorySafetyDecision::Reject { .. }
        ));

        let mut crossed_route = context;
        crossed_route.calibration_fingerprint = aesthetic.calibration.unwrap().fingerprint;
        assert!(matches!(
            safety_check(&spec, &base, &crossed_route),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn fp32_is_refused_instead_of_mislabelling_bf16_cuda_execution() {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/anima".into()));
        spec.precision = Precision::Fp32;
        let contract = weights_free_contract("anima_base", &spec).unwrap();
        let context = gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::StagedResidency,
            MemoryNumericTier {
                precision: Precision::Fp32,
                quant: None,
                component_precision_floors: &[],
            },
            MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .unwrap();
        assert!(matches!(
            safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn every_route_has_provider_owned_fixture_and_begin_request_seam() {
        let spec = LoadSpec::new(WeightsSource::Dir("/anima".into()));
        for provider in IDS {
            let contract = weights_free_contract(provider, &spec).unwrap();
            let fixture =
                registered_valid_fixture(&spec, &contract, MemoryStrategy::StagedResidency)
                    .unwrap()
                    .pop()
                    .unwrap();
            assert!(registered_begin_request(&spec, &contract, &fixture.context)
                .unwrap()
                .is_some());
        }
    }

    /// Write a split_files tree whose three components have deliberately distinct element counts, so
    /// a swapped component assignment cannot pass. Returns the root plus the exact per-component
    /// materialized byte totals derived from the tensors actually written.
    fn write_priced_split_files(
        temp: &tempfile::TempDir,
        variant: Variant,
    ) -> (std::path::PathBuf, u64, u64, u64) {
        use candle_gen::candle_core::{DType, Device, Tensor};
        use std::collections::HashMap;

        let root = temp.path().join("anima_priced");
        let write = |relative: &str, rows: usize, columns: usize| -> u64 {
            let path = root.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut tensors = HashMap::new();
            tensors.insert(
                "net.x_embedder.proj.1.weight".to_string(),
                Tensor::zeros((rows, columns), DType::F32, &Device::Cpu).unwrap(),
            );
            candle_gen::candle_core::safetensors::save(&tensors, &path).unwrap();
            // The written tensor is F32 on disk but bf16 once materialized: exactly two bytes per
            // element, derived from the shape written here rather than pinned to a literal.
            (rows as u64) * (columns as u64) * FLOAT_WIDTH
        };
        let transformer = write(
            &format!("diffusion_models/{}", variant.dit_filename()),
            64,
            32,
        );
        let conditioning = write(crate::loader::TEXT_ENCODER_FILE, 16, 8);
        let decoder = write(crate::loader::VAE_FILE, 4, 2);
        (root, conditioning, transformer, decoder)
    }

    #[test]
    fn contract_prices_the_declared_asset_bytes_from_the_on_disk_inventory() {
        for (provider, variant) in
            IDS.iter()
                .zip([Variant::Base, Variant::Aesthetic, Variant::Turbo])
        {
            let temp = tempfile::tempdir().unwrap();
            let (root, conditioning, transformer, decoder) =
                write_priced_split_files(&temp, variant);
            let spec = LoadSpec::new(WeightsSource::Dir(root));
            let contract = contract(provider, &spec).unwrap();

            assert!(
                contract.formula.uses(MemoryFormulaVariable::AssetBytes),
                "{provider}: the priced fact must stay a declared formula variable"
            );
            assert_eq!(
                contract.asset_facts.conditioning_bytes, conditioning,
                "{provider}: conditioning bytes"
            );
            assert_eq!(
                contract.asset_facts.transformer_bytes, transformer,
                "{provider}: transformer bytes"
            );
            assert_eq!(
                contract.asset_facts.decoder_bytes, decoder,
                "{provider}: decoder bytes"
            );
            assert_eq!(
                contract.asset_facts.base_bytes,
                conditioning + transformer + decoder,
                "{provider}: base bytes"
            );
            assert_ne!(
                contract.asset_facts,
                gen_core::MemoryAssetFacts::default(),
                "{provider}: a declared AssetBytes variable must not be pinned at zero"
            );

            // The registry-surface contract stays honestly unpriced rather than borrowing these.
            assert_eq!(
                weights_free_contract(provider, &spec).unwrap().asset_facts,
                gen_core::MemoryAssetFacts::default()
            );
        }
    }
}
