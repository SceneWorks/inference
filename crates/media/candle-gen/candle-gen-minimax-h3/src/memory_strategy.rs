//! MiniMax-H3's shared-ladder `MemoryProviderContract` on Candle/CUDA (sc-18659).
//!
//! The sibling declaration to `mlx_gen_minimax_h3::memory_strategy`. The contract carries a
//! [`MemoryBackendRealization`], so the two backends are *allowed* to differ — and here they
//! genuinely do, in three ways that are worth stating before the code:
//!
//! 1. **No fitted curve.** Candle has only the floor arm, so the formula is
//!    [`MemoryFormulaKind::AssetBytesPlusHeadroom`] rather than MLX's phase envelope, and
//!    [`MemoryProviderContract::calibration`] is `None`. A provider with no calibration identity
//!    can run its resident path and can never claim a verified optimized fit — which is exactly
//!    the truth here, and is enforced: `standard_memory_strategy_safety_check` refuses any
//!    optimized selection when the identity is absent.
//! 2. **No staged residency.** MLX declares [`MemoryStrategy::StagedResidency`] `Implemented`
//!    because `MiniMaxH3::generate_impl` releases each heavy component before mapping the next.
//!    This crate has **no pipeline at all** — it ships the DiT, the joint denoise and the two VAE
//!    decoders, and no text encoder and no generator (sc-17156 owns the end-to-end path). There is
//!    nothing to stage, so rung 1 is `Missing`.
//! 3. **No fused streaming SDPA.** The MLX verdict that attention scratch is already streamed —
//!    peak tracking `4·B·H·S·D` with no materialized score tensor — is a property of *MLX's* fused
//!    kernel and **must not be copied here**. Candle materializes scores, so bounding attention
//!    (sc-18661) may buy real memory on this backend even if it buys none on MLX.
//!
//! # Why the asset facts are the full four components anyway
//!
//! `conditioning_bytes` charges the 66.71 GB Qwen3-VL-32B text encoder even though this crate
//! cannot yet execute it. Asset facts are the render's byte floor, not a capability claim: a
//! candle render of this family needs the conditioner, and a contract that declared zero there
//! would publish a floor small enough to admit a request that cannot possibly run. Capability
//! lives in `strategies`, where every optimized rung is honestly `Missing`.
//!
//! # Stage attribution
//!
//! The ~53 GB memory floor measured for this family is the **conditioning** stage — the dense text
//! encoder in isolation — not the DiT and not activation pressure. The DiT's own denoise-resident
//! cost is a separate, genuinely tiered quantity. Those measurements were taken on MLX; this
//! backend has none of its own yet (sc-17156), which is the other reason `calibration` is `None`.

use candle_gen::gen_core::{
    safetensors_path_bytes, LoadShape, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryFormulaKind, MemoryLifecycleCapabilities, MemoryParameterRanges, MemoryProviderContract,
    MemoryRunContext, MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability,
    MemoryStrategySupport, MemoryWindowMaterialization, ResidentRequestMemory, WeightsSource,
};

use crate::MODEL_ID;

// --- measured asset facts -------------------------------------------------------------------
//
// Exact `.safetensors` bytes under each component directory of the upstream bf16 snapshot
// (`MiniMaxAI/MiniMax-H3` @ `939557dc`). Identical to the MLX sibling's, deliberately: these are
// facts about the checkpoint, not about a backend, and the two must not drift apart.

/// Qwen3-VL-32B text encoder — 14 shards, the conditioning component. 66.71 GB.
pub const TEXT_ENCODER_BYTES: u64 = 66_714_912_872;

/// One 33 B DiT partition at bf16 — 14 shards. 66.28 GB. A render loads exactly one of
/// `transformer` / `transformer_ref`, so it is charged once.
pub const DIT_BF16_BYTES: u64 = 66_280_504_216;

/// Video VAE — 3 shards; the decoder is a 36-layer transformer. 10.42 GB.
pub const VIDEO_VAE_BYTES: u64 = 10_415_558_888;

/// Audio VAE — 1 shard. 0.61 GB.
pub const AUDIO_VAE_BYTES: u64 = 605_429_340;

/// Exact bytes the AdaLN precompute-and-evict drops, asserted against the loader in
/// [`crate::dit::adaln`]. Declared here so both backends' contracts carry the same figure; sc-18665
/// turns it into a typed resident-component exclusion the ladder can see.
pub const ADALN_EVICTED_BYTES: u64 = 26_020_915_200;

/// The load shape this loader actually has, pinned rather than mirrored from the spec.
///
/// [`LoadShape::DeferredMaterialization`] means transformer blocks are materialized through a block
/// schedule. `MiniMaxH3Dit::load` builds the whole stack, so this provider is
/// [`LoadShape::EagerMaterialization`] whatever a caller asks for. sc-18662 changes it.
pub const LOAD_SHAPE: LoadShape = LoadShape::EagerMaterialization;

/// The DiT component directory a flat snapshot carries.
const DIT_COMPONENT: &str = "transformer";

struct ComponentBytes {
    text_encoder: u64,
    dit: u64,
    video_vae: u64,
    audio_vae: u64,
}

impl ComponentBytes {
    fn resolve(spec: &LoadSpec) -> Self {
        let root = match &spec.weights {
            WeightsSource::Dir(root) => root.clone(),
            WeightsSource::File(path) => path.parent().unwrap_or(path).to_path_buf(),
        };
        let dit = match spec.components.get(DIT_COMPONENT) {
            Some(WeightsSource::Dir(staged)) => staged.clone(),
            _ => root.join(DIT_COMPONENT),
        };
        Self {
            text_encoder: safetensors_path_bytes(root.join("text_encoder")),
            dit: safetensors_path_bytes(dit),
            video_vae: safetensors_path_bytes(root.join("vae")),
            audio_vae: safetensors_path_bytes(root.join("audio_vae")),
        }
    }

    /// The two decoders are one contract field; H3 is the first family with two of them.
    fn decoder(&self) -> u64 {
        self.video_vae.saturating_add(self.audio_vae)
    }

    fn base(&self) -> u64 {
        self.text_encoder
            .saturating_add(self.dit)
            .saturating_add(self.decoder())
    }
}

/// The five capability entries. Only the resident baseline is implemented on this backend.
///
/// Every entry publishes an empty [`MemoryParameterRanges`], which is correct in both directions:
/// rung 0 owns no numeric parameters, and a `Missing` rung must not publish a domain it cannot
/// honor. Flipping any of rungs 2-4 to `Implemented` without filling its domain is a conformance
/// error, not a silent under-declaration.
fn strategies() -> Vec<MemoryStrategyCapability> {
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
                // Rung 1: sc-17156 (there is no pipeline to stage). Rungs 2/3/4: sc-18660 /
                // sc-18661 / sc-18662.
                _ => MemoryStrategySupport::Missing,
            },
            parameters: MemoryParameterRanges::default(),
        })
        .collect()
}

fn build_contract(components: &ComponentBytes) -> MemoryProviderContract {
    MemoryProviderContract {
        provider_id: MODEL_ID.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            // There is no block-wise host→device materialization path in this crate at all: the
            // DiT loads as a whole stack. sc-18662 builds one.
            host_to_device_block_materialization: false,
            // Answered even though rung 4 is `Missing`, because the field is deliberately not
            // optional. It is accurate for the loader as it exists: this crate has no packed tier
            // and therefore no MLX-affine → GGML repack seam, so a window would be a mapped read
            // plus a host-to-device copy of bytes already in the accelerator's form. **sc-18662
            // must re-verify this** if a packed candle tier ever lands — that is the change that
            // turns a conforming realization into a `HostFormatConversion` one.
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies: strategies(),
        pid_decode_routes: None,
        load_shape: LOAD_SHAPE,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        // No phases and no hooks: nothing in this crate releases a component at a phase boundary,
        // tiles a decode, chunks attention or windows the block stack. Declaring a hook here would
        // be the false-declaration case `conformance_errors` cannot catch, because a hook flag is
        // not checked against an implementation.
        lifecycle: MemoryLifecycleCapabilities::default(),
        // The floor arm, and only the floor arm.
        formula: MemoryFormulaKind::AssetBytesPlusHeadroom,
        // No fitted curve exists for this backend. `None` is the honest state, and it is load
        // bearing: it makes every optimized selection fail closed at admission.
        calibration: None,
        asset_facts: MemoryAssetFacts {
            base_bytes: components.base(),
            conditioning_bytes: components.text_encoder,
            transformer_bytes: components.dit,
            decoder_bytes: components.decoder(),
            overlay_bytes: 0,
        },
        runtime: Default::default(),
    }
}

/// The production contract: asset facts read off the resolved snapshot.
pub fn contract_for(spec: &LoadSpec) -> candle_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_contract(&ComponentBytes::resolve(spec)))
}

/// The weights-free fixture contract: the identical route declaration with zero asset facts and no
/// filesystem traversal.
pub fn weights_free_contract(
    _spec: &LoadSpec,
) -> candle_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_contract(&ComponentBytes {
        text_encoder: 0,
        dit: 0,
        video_vae: 0,
        audio_vae: 0,
    }))
}

/// The provider's real admission check, callable before any weight file is opened.
///
/// The shared check is sufficient here and a route gate would be a lie: with no optimized rung and
/// no calibration identity, every admission this provider can accept is the resident baseline, and
/// the geometry gates belong to the pipeline that sc-17156 has yet to write.
pub fn safety_check(
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    candle_gen::gen_core::default_memory_strategy_safety_check(contract, context)
}

/// The memory-strategy registration for `minimax_h3` on candle.
///
/// **Not yet reachable from `candle-gen-catalog`**, and that is deliberate rather than an
/// oversight: `ProviderRegistryBuilder::build` rejects a memory-strategy registration whose
/// `provider_id` has no matching generator, and this crate ships no generator (sc-17156). Wiring it
/// through `register_composed_memory_strategy` would satisfy the builder by declaring a composition
/// root that does not exist — the exact "provider contract with no executable owner" that seam
/// exists to prevent. The constant is exercised by this module's conformance tests today and gets
/// its catalog line the moment the generator lands.
pub const MEMORY_REGISTRATION: candle_gen::gen_core::MemoryRegistration =
    candle_gen::gen_core::MemoryRegistration {
        provider_id: MODEL_ID,
        contract: contract_for,
        safety_check,
    };

/// The weights-free contract fixture paired with [`MEMORY_REGISTRATION`].
pub const MEMORY_CONTRACT_FIXTURE: candle_gen::gen_core::MemoryContractFixtureRegistration =
    candle_gen::gen_core::MemoryContractFixtureRegistration {
        provider_id: MODEL_ID,
        contract: weights_free_contract,
    };

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{
        LoadShape, MemoryBudget, MemoryCacheState, MemoryCalibrationIdentity, MemoryGeometry,
        MemoryMode, MemoryNumericTier, MemoryPhase, MemoryRunContext, MemorySelection,
        MemoryStrategyParameters, ProviderRegistryBuilder,
    };
    use std::path::Path;

    /// One named, independently applied mutation of a known-good contract.
    type ContractMutation = (&'static str, Box<dyn Fn(&mut MemoryProviderContract)>);

    fn weightless_spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_load_shape(LoadShape::DeferredMaterialization)
    }

    fn declared() -> MemoryProviderContract {
        weights_free_contract(&weightless_spec()).expect("weights-free contract")
    }

    fn candle_backend() -> MemoryBackendRealization {
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        }
    }

    /// Sparse `.safetensors` shards of exact sizes. `safetensors_path_bytes` stats rather than
    /// parses, so this costs no disk and still exercises the real directory-name wiring.
    fn sparse_snapshot(root: &Path, sizes: &[(&str, u64)]) {
        for (component, bytes) in sizes {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).expect("component dir");
            let file = std::fs::File::create(dir.join("model.safetensors")).expect("shard");
            file.set_len(*bytes).expect("sparse shard");
        }
    }

    fn full_snapshot(root: &Path) {
        sparse_snapshot(
            root,
            &[
                ("text_encoder", TEXT_ENCODER_BYTES),
                ("transformer", DIT_BF16_BYTES),
                ("vae", VIDEO_VAE_BYTES),
                ("audio_vae", AUDIO_VAE_BYTES),
                // Byte-identical to `transformer`; a render loads exactly one, so it must not be
                // charged on top.
                ("transformer_ref", DIT_BF16_BYTES),
            ],
        );
    }

    // --- AC1: declared Resident, not fallen-back Resident ---------------------------------------

    /// **The honest state of this backend is resident-only**, so the strategy table alone cannot
    /// distinguish this declaration from `compatibility_default` — asserting on it would be the
    /// false green AC1 warns about. The distinguisher is the one that actually matters to a
    /// consumer: a fallback contract publishes a **zero** byte floor for a 144 GB family, and this
    /// one publishes the measured components split across the fields the formula reads.
    #[test]
    fn resolved_contract_is_declared_and_not_the_compatibility_default() {
        let root = tempfile::tempdir().expect("tempdir");
        full_snapshot(root.path());
        let contract =
            (MEMORY_REGISTRATION.contract)(&LoadSpec::new(WeightsSource::Dir(root.path().into())))
                .expect("registered contract");

        let fallback = MemoryProviderContract::compatibility_default(MODEL_ID, candle_backend());
        assert_ne!(contract, fallback);
        assert_eq!(
            fallback.asset_facts.base_bytes, 0,
            "the fallback publishes a zero floor — that is what makes it unsafe here"
        );
        assert_eq!(
            contract.asset_facts.base_bytes,
            TEXT_ENCODER_BYTES + DIT_BF16_BYTES + VIDEO_VAE_BYTES + AUDIO_VAE_BYTES
        );
        assert_eq!(contract.asset_facts.conditioning_bytes, TEXT_ENCODER_BYTES);
        assert_eq!(contract.asset_facts.transformer_bytes, DIT_BF16_BYTES);
        assert_eq!(
            contract.asset_facts.decoder_bytes,
            VIDEO_VAE_BYTES + AUDIO_VAE_BYTES,
            "both decoders are charged, and only the decoders"
        );
        assert!(contract.conformance_errors().is_empty());
    }

    /// A misspelled provider id would produce a contract for a family that does not exist. The two
    /// registration constants and the contract must all agree on one id.
    #[test]
    fn every_declaration_agrees_on_one_provider_id() {
        assert_eq!(MODEL_ID, "minimax_h3");
        assert_eq!(declared().provider_id, MODEL_ID);
        assert_eq!(MEMORY_REGISTRATION.provider_id, MODEL_ID);
        assert_eq!(MEMORY_CONTRACT_FIXTURE.provider_id, MODEL_ID);
        assert_eq!(
            (MEMORY_CONTRACT_FIXTURE.contract)(&weightless_spec())
                .expect("fixture")
                .provider_id,
            MEMORY_REGISTRATION.provider_id
        );
    }

    /// The registration is deliberately **not** wired into `candle-gen-catalog` yet, and this pins
    /// why: `build()` rejects a memory strategy whose id has no matching generator, and this crate
    /// ships none (sc-17156). If a generator ever lands without the catalog line being added, this
    /// test starts failing and says so.
    #[test]
    fn catalog_wiring_waits_on_the_generator_that_does_not_exist_yet() {
        let orphan = ProviderRegistryBuilder::new()
            .register_memory_strategy(MEMORY_REGISTRATION)
            .register_memory_contract_fixture(MEMORY_CONTRACT_FIXTURE)
            .build();
        let Err(error) = orphan else {
            panic!("a memory strategy with no generator must not build");
        };
        assert!(
            error.to_string().contains("no matching generator"),
            "unexpected rejection: {error}"
        );
    }

    // --- AC2: nothing optimized is declared, and nothing optimized is reachable ------------------

    /// Every optimized rung is `Missing`, and each is independently refused at selection. With no
    /// calibration identity the shared safety check refuses them a second time, so the declaration
    /// and the admission path agree.
    #[test]
    fn no_optimized_rung_is_declared_or_selectable() {
        let contract = declared();
        assert!(
            contract.calibration.is_none(),
            "candle has no fitted curve for this family"
        );
        for strategy in MemoryStrategy::ALL {
            let expected = if strategy == MemoryStrategy::Resident {
                MemoryStrategySupport::Implemented
            } else {
                MemoryStrategySupport::Missing
            };
            assert_eq!(
                contract.capability(strategy).expect("entry").support,
                expected,
                "{strategy:?}"
            );
            if strategy.is_optimized() {
                assert!(
                    contract
                        .validate_selection(&MemorySelection {
                            strategy,
                            tier: MemoryNumericTier {
                                precision: candle_gen::gen_core::Precision::Bf16,
                                quant: None,
                                component_precision_floors: &[],
                            },
                            parameters: MemoryStrategyParameters::default(),
                        })
                        .is_err(),
                    "{strategy:?} must not be selectable"
                );
                assert!(
                    matches!(
                        safety_check(&weightless_spec(), &contract, &context(strategy)),
                        MemorySafetyDecision::Reject { .. }
                    ),
                    "{strategy:?} must be refused at admission"
                );
            }
        }
        // The control arm: the one rung this backend does implement is admitted, so the rejections
        // above are not an always-rejecting check.
        assert_eq!(
            safety_check(
                &weightless_spec(),
                &declared(),
                &context(MemoryStrategy::Resident)
            ),
            MemorySafetyDecision::Accept
        );
    }

    fn context(strategy: MemoryStrategy) -> MemoryRunContext {
        MemoryRunContext {
            selection: MemorySelection {
                strategy,
                tier: MemoryNumericTier {
                    precision: candle_gen::gen_core::Precision::Bf16,
                    quant: None,
                    component_precision_floors: &[],
                },
                parameters: MemoryStrategyParameters::default(),
            },
            calibration_abi: candle_gen::gen_core::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: String::new(),
            load_shape: LOAD_SHAPE,
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 1344,
                height: 768,
                batch: 1,
                frames: 124,
                reference_count: 0,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 256 * 1024 * 1024 * 1024,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1024 * 1024 * 1024,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "unit-test".to_owned(),
        }
    }

    // --- AC3: weights-free conformance, and a malformed contract fails ---------------------------

    /// The same check the registry path runs, on the same fixture factory, in the default lane.
    #[test]
    fn fixture_contract_conforms_weights_free() {
        let fixture = (MEMORY_CONTRACT_FIXTURE.contract)(&weightless_spec()).expect("fixture");
        assert_eq!(
            fixture.asset_facts,
            MemoryAssetFacts::default(),
            "the fixture must inject zero asset facts without touching the filesystem"
        );
        gen_core_testkit::memory_strategy_conformance(&fixture);

        // ...and it must not diverge from the production declaration in anything but bytes.
        let root = tempfile::tempdir().expect("tempdir");
        full_snapshot(root.path());
        let production =
            contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
        assert_eq!(fixture.strategies, production.strategies);
        assert_eq!(fixture.lifecycle, production.lifecycle);
        assert_eq!(fixture.formula, production.formula);
        assert_eq!(fixture.calibration, production.calibration);
        assert_eq!(fixture.load_shape, production.load_shape);
        assert_eq!(fixture.backend, production.backend);
    }

    /// Each mutation is applied **alone** to a known-good contract, so each guard is proven to
    /// detect its own breakage rather than the set proving itself.
    #[test]
    fn each_contract_mutation_is_independently_detected() {
        assert!(
            gen_core_testkit::check_memory_strategy_contract(&declared()).is_ok(),
            "the shipped contract must conform, or every mutation below is vacuous"
        );

        let mutations: Vec<ContractMutation> = vec![
            (
                "a dropped strategy entry",
                Box::new(|c: &mut MemoryProviderContract| {
                    c.strategies
                        .retain(|entry| entry.strategy != MemoryStrategy::BoundedDecode);
                }),
            ),
            (
                "a duplicated strategy entry",
                Box::new(|c: &mut MemoryProviderContract| {
                    let first = c.strategies[0].clone();
                    c.strategies.push(first);
                }),
            ),
            (
                "a Resident baseline that is not implemented",
                Box::new(|c: &mut MemoryProviderContract| {
                    for entry in &mut c.strategies {
                        if entry.strategy == MemoryStrategy::Resident {
                            entry.support = MemoryStrategySupport::Missing;
                        }
                    }
                }),
            ),
            (
                "an empty StructurallyNotApplicable reason",
                Box::new(|c: &mut MemoryProviderContract| {
                    for entry in &mut c.strategies {
                        if entry.strategy == MemoryStrategy::BoundedAttention {
                            entry.support = MemoryStrategySupport::StructurallyNotApplicable {
                                reason: "   ".to_owned(),
                            };
                        }
                    }
                }),
            ),
            (
                "StagedResidency implemented with no lifecycle phases",
                Box::new(|c: &mut MemoryProviderContract| {
                    for entry in &mut c.strategies {
                        if entry.strategy == MemoryStrategy::StagedResidency {
                            entry.support = MemoryStrategySupport::Implemented;
                        }
                    }
                }),
            ),
            (
                "BoundedDecode implemented with no tile domain",
                Box::new(|c: &mut MemoryProviderContract| {
                    implement_without_range(c, MemoryStrategy::BoundedDecode)
                }),
            ),
            (
                "BoundedAttention implemented with no chunk domain",
                Box::new(|c: &mut MemoryProviderContract| {
                    implement_without_range(c, MemoryStrategy::BoundedAttention)
                }),
            ),
            (
                "BoundedTransformerResidency implemented with no window domain",
                Box::new(|c: &mut MemoryProviderContract| {
                    implement_without_range(c, MemoryStrategy::BoundedTransformerResidency)
                }),
            ),
            (
                "base_bytes that does not equal its components",
                Box::new(|c: &mut MemoryProviderContract| c.asset_facts.base_bytes += 1),
            ),
            (
                "a malformed calibration fingerprint",
                Box::new(|c: &mut MemoryProviderContract| {
                    c.calibration = Some(MemoryCalibrationIdentity::new("No_Version", LOAD_SHAPE));
                }),
            ),
            (
                "a calibration load shape that disagrees with the contract",
                Box::new(|c: &mut MemoryProviderContract| {
                    c.calibration = Some(MemoryCalibrationIdentity::new(
                        "minimax-h3-candle-v1",
                        LoadShape::DeferredMaterialization,
                    ));
                }),
            ),
        ];

        for (name, mutate) in mutations {
            let mut contract = declared();
            mutate(&mut contract);
            assert!(
                gen_core_testkit::check_memory_strategy_contract(&contract).is_err(),
                "conformance must reject {name}"
            );
        }
    }

    /// AC5's failure shape: a lever declared `Implemented` with no [`MemoryParameterRanges`].
    fn implement_without_range(contract: &mut MemoryProviderContract, strategy: MemoryStrategy) {
        for entry in &mut contract.strategies {
            if entry.strategy == strategy {
                entry.support = MemoryStrategySupport::Implemented;
                entry.parameters = MemoryParameterRanges::default();
            }
        }
        match strategy {
            MemoryStrategy::BoundedDecode => contract.lifecycle.decode_tiling = true,
            MemoryStrategy::BoundedAttention => contract.lifecycle.attention_chunking = true,
            MemoryStrategy::BoundedTransformerResidency => {
                contract.lifecycle.transformer_window_materialization = true
            }
            _ => {}
        }
    }

    // --- AC5: parameter ranges are declared exactly where they are owned ------------------------

    #[test]
    fn parameter_ranges_are_owned_by_the_rung_that_consumes_them() {
        let contract = declared();
        assert!(contract.conformance_errors().is_empty());
        for capability in &contract.strategies {
            assert_eq!(
                capability.parameters,
                MemoryParameterRanges::default(),
                "{:?} owns no numeric parameters on this backend",
                capability.strategy
            );
        }
        for strategy in [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            let mut mutated = declared();
            implement_without_range(&mut mutated, strategy);
            assert!(
                !mutated.conformance_errors().is_empty(),
                "{strategy:?} implemented with no MemoryParameterRanges must fail"
            );
        }
    }

    // --- AC4: measured asset facts --------------------------------------------------------------

    /// Tolerance for the GB figures recorded on sc-18659, in bytes. The story quotes 66.73 GB for
    /// the text encoder; the measured on-disk total is 66,714,912,872 B = 66.715 GB, so the byte
    /// constants are authoritative and the GB figures are held only to this window.
    const GB_TOLERANCE_BYTES: u64 = 20_000_000;

    fn assert_within(measured: u64, story_gb: f64, what: &str) {
        let story_bytes = (story_gb * 1e9) as u64;
        let delta = measured.abs_diff(story_bytes);
        assert!(
            delta <= GB_TOLERANCE_BYTES,
            "{what}: measured {measured} B ({:.3} GB) is {delta} B from the recorded {story_gb} GB, \
             outside the {GB_TOLERANCE_BYTES} B tolerance",
            measured as f64 / 1e9
        );
    }

    #[test]
    fn measured_component_bytes_match_the_recorded_footprints() {
        assert_within(DIT_BF16_BYTES, 66.28, "33 B DiT partition at bf16");
        assert_within(TEXT_ENCODER_BYTES, 66.73, "Qwen3-VL-32B text encoder");
        assert_within(VIDEO_VAE_BYTES, 10.42, "video VAE");
        assert_within(AUDIO_VAE_BYTES, 0.61, "audio VAE");
        assert_eq!(
            ADALN_EVICTED_BYTES, 26_020_915_200,
            "the exact bytes crate::dit::adaln releases"
        );
    }

    /// The two backends must not drift apart on facts about the same checkpoint.
    #[test]
    fn the_measured_facts_are_the_same_numbers_the_mlx_sibling_declares() {
        // Mirrored deliberately rather than shared through a dependency: `candle-gen-*` crates do
        // not depend on `mlx-gen-*`. These literals are the same ones
        // `mlx_gen_minimax_h3::memory_strategy` declares, and both are asserted against the story's
        // recorded GB figures above, so a drift on either side fails there first.
        assert_eq!(TEXT_ENCODER_BYTES, 66_714_912_872);
        assert_eq!(DIT_BF16_BYTES, 66_280_504_216);
        assert_eq!(VIDEO_VAE_BYTES, 10_415_558_888);
        assert_eq!(AUDIO_VAE_BYTES, 605_429_340);
    }

    #[test]
    fn a_staged_dit_component_is_charged_at_its_own_size() {
        let root = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(root.path(), &[("transformer", DIT_BF16_BYTES)]);
        let staged = tempfile::tempdir().expect("tempdir");
        const Q4_BYTES: u64 = 18_779_970_678;
        sparse_snapshot(staged.path(), &[("transformer", Q4_BYTES)]);

        let contract = contract_for(
            &LoadSpec::new(WeightsSource::Dir(root.path().into())).with_component(
                DIT_COMPONENT,
                WeightsSource::Dir(staged.path().join(DIT_COMPONENT)),
            ),
        )
        .expect("contract");
        assert_eq!(contract.asset_facts.transformer_bytes, Q4_BYTES);
    }

    // --- the declaration facts that are easy to get wrong ----------------------------------------

    /// The load shape is pinned to the loader, not mirrored from the request.
    #[test]
    fn load_shape_is_pinned_to_the_loader_not_taken_from_the_spec() {
        let spec = weightless_spec();
        assert_eq!(spec.load_shape, LoadShape::DeferredMaterialization);
        assert_eq!(
            contract_for(&spec).expect("contract").load_shape,
            LoadShape::EagerMaterialization
        );
        // ...and the spec IS read: the asset facts come off it.
        let root = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(root.path(), &[("audio_vae", AUDIO_VAE_BYTES)]);
        let resolved =
            contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
        assert_eq!(resolved.asset_facts.decoder_bytes, AUDIO_VAE_BYTES);
    }

    /// The three ways this backend's declaration diverges from the MLX sibling's, pinned so a later
    /// slice cannot copy an MLX verdict across without the test noticing.
    #[test]
    fn the_candle_declaration_differs_from_mlx_where_the_backends_differ() {
        let contract = declared();
        // 1. No fitted curve: the floor arm only, and no calibration identity.
        assert_eq!(contract.formula, MemoryFormulaKind::AssetBytesPlusHeadroom);
        assert!(contract.calibration.is_none());
        // 2. No staged residency: there is no pipeline to stage, so no phases and no hooks.
        assert_eq!(contract.lifecycle, MemoryLifecycleCapabilities::default());
        assert!(contract.lifecycle.phases.is_empty());
        assert!(!contract.lifecycle.synchronized_phase_release);
        assert_eq!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .expect("entry")
                .support,
            MemoryStrategySupport::Missing
        );
        // 3. No fused streaming SDPA: rung 3 stays open here on its own evidence, never on MLX's.
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .expect("entry")
                .support,
            MemoryStrategySupport::Missing,
            "candle materializes attention scores; the MLX streaming verdict does not carry over"
        );
        assert!(matches!(
            contract.backend,
            MemoryBackendRealization::CandleCuda { .. }
        ));
        assert_eq!(
            contract.backend.backend_id(),
            "candle",
            "the evidence key must not be recorded under the MLX backend"
        );
    }

    /// A phase hook declared without an implementation is the false declaration conformance cannot
    /// catch. This pins the honest state: no phase is declared at all.
    #[test]
    fn no_lifecycle_phase_is_declared_without_an_implementation() {
        let contract = declared();
        for phase in [
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ] {
            assert!(
                !contract.lifecycle.phases.contains(&phase),
                "{phase:?} must not be declared until something releases it at a boundary"
            );
        }
        assert!(!contract.lifecycle.decode_tiling);
        assert!(!contract.lifecycle.attention_chunking);
        assert!(!contract.lifecycle.transformer_window_materialization);
    }
}
