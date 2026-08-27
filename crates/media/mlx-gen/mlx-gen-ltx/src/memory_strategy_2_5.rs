//! The **LTX-2.5** MLX memory-strategy registration (sc-18797, epic 18755 R9).
//!
//! This is a SECOND registration in this crate, not a replacement.
//! [`crate::memory_strategy`] keeps declaring `ltx_2_3` exactly as it did; this module declares the
//! 2.5 engine id, whose ladder is genuinely different because 2.5 is the generation that got a
//! deferred block loader ([`crate::block_stream`]) and a bounded attention seam.
//!
//! It lives in its own file rather than as more arms inside `memory_strategy.rs` so the 2.3 contract
//! is not re-shaped by a 2.5 change, and so concurrent 2.5 work on the shared file does not collide
//! here.
//!
//! # What is declared, and what backs each declaration
//!
//! | rung | support | backed by |
//! |---|---|---|
//! | 1 `StagedResidency` | Implemented | LTX's unconditional Wan-style TE staging (epic 10975) |
//! | 2 `BoundedDecode` | Implemented | the shared tiled VAE decode, as 2.3 |
//! | 3 `BoundedAttention` | Implemented | `gen_core::attention_budget` via `mlx_gen::attention::sdpa_budgeted_bhsd` |
//! | 4 `BoundedTransformerResidency` | Implemented **iff streamable** | `gen_core::block_window` via [`crate::block_stream`] |
//!
//! Rungs 3 and 4 are declared here and `Missing` on 2.3 because the code that executes them is
//! reached only through the 2.5 loader. R9 forbids declaring a rung no route can execute, so the
//! support value is computed from the load spec rather than asserted.
//!
//! # Reachability today
//!
//! The `ltx_2_5` **generator descriptor** is sc-18778 and has not landed. `ProviderRegistryBuilder`
//! refuses a memory-strategy registration whose `provider_id` matches no registered generator
//! (`registry.rs`: *"memory-strategy contract '{}' has no matching generator registration"*), so
//! [`MEMORY_REGISTRATION`] is deliberately **not** wired into
//! [`crate::register_providers`] yet — doing so would fail every catalog build. It is correct and
//! inert, exactly as the story anticipates; sc-18778 wires it in one line. Inventing a descriptor
//! here to make it "reachable" would be inventing a routing entry, which is not this story's to do.

use gen_core::{
    LoadShape, LoadSpec, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryParameterRanges, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryStrategy, MemoryStrategyCapability,
    MemoryStrategyPrerequisite, MemoryStrategySupport, ResidentRequestMemory, TransformerComponent,
    WeightsSource,
};
use mlx_gen::gen_core;

use crate::memory_strategy::{decode_tile_edges, registered_safety_check, DECODE_OVERLAP};

/// The LTX-2.5 MLX engine id.
///
/// A distinct id from [`crate::MODEL_ID`] (`ltx_2_3`) rather than a route overlay on it: the two
/// generations have different transformer deltas, different text encoders and — the reason it
/// matters here — different ladders. Sharing an id would make the 2.3 contract's `Missing` rungs and
/// this one's `Implemented` rungs two answers to the same question.
pub const LTX_2_5_MODEL_ID: &str = "ltx_2_5";

/// Rung 4's published window cadences over the 48-block AvDiT trunk.
///
/// **Distinguishable by construction.** Peak block residency is `window x per-block bytes` (linear —
/// that is the rung's contract), so each of these bounds twice the previous one: consecutive
/// candidates differ by a factor of 2 in exactly the quantity the rung claims to control, and a
/// calibration sweep across them records signal rather than noise. This is the Z-Image
/// `TRANSFORMER_WINDOW_SIZES = [1]` lesson applied: a domain whose members are indistinguishable in
/// the bounded quantity should have one member, and one whose members are distinguishable should say
/// so arithmetically rather than by taste.
///
/// Every value divides 48 exactly (48 = 2^4 x 3), so no candidate carries a ragged tail window whose
/// peak differs from its nominal size — a tail is not wrong, but it makes the sweep's independent
/// variable inexact.
///
/// `48` (fully resident) is deliberately absent: [`gen_core::block_window::BlockPlan::is_bounded`]
/// is false for an all-covering window, so publishing it would advertise a rung-4 selection that
/// bounds nothing while paying the windowing machinery's cost.
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1, 2, 4, 8, 16];

/// The default cadence — the tightest bound, which is what the rung exists for.
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;

/// The only component scope LTX-2.5 windows, and the AV-specific justification the story asks for.
///
/// LTX's audio branch is **not** a second transformer: each `transformer_blocks.{n}` entry is an
/// `AvBlock` carrying the video stack, the audio stack and both cross-modal attentions, so one
/// window over the block axis already bounds both modalities. `Dit` therefore describes what this
/// code does.
///
/// [`TransformerComponent::Both`] would additionally claim the **Gemma-4 text encoder** streams.
/// It does not: the encoder is a separate component, [`crate::block_stream`] does not window it, and
/// nothing is measured for it on this family. Declaring it would be an unreachable declaration —
/// precisely the defect class R9 exists to prevent — so the contract publishes `Dit` alone and a
/// `TextEncoder`/`Both` selection is refused rather than silently served as `Dit`.
pub const TRANSFORMER_WINDOW_COMPONENT: TransformerComponent = TransformerComponent::Dit;

/// Rung 3's published score budgets, in attention-score ELEMENTS per chunk.
///
/// Both engage at production video geometries and neither engages at the small end, which is what
/// makes them a domain rather than a pair of numbers. The arithmetic, from the shipped config
/// (`num_attention_heads = 32`) and a single batch:
///
/// - `768x512x145` -> roughly `24 x 16 x 18 = 6912` video self-attention tokens, so
///   `32 x 6912^2 ~= 1.4 Gi` scores. 64 Mi chunks that ~23 ways; 16 Mi chunks it ~91 ways.
/// - `256x256x9` -> roughly `8 x 8 x 2 = 128` tokens, so `32 x 128^2 = 0.5 Mi` scores. **Neither**
///   budget chunks it, deliberately: 2 MiB of f32 scratch is not what this rung exists to bound.
///
/// 64 Mi is [`gen_core::attention_budget::CONSTRAINED_ATTN_SCORES_BUDGET`], the shared family
/// operating point, kept first so LTX's evidence is comparable with its siblings'. 16 Mi is a 4x
/// tighter bound for hosts where the operating point is still too loose.
pub const ATTENTION_CHUNK_SIZES: &[u32] = &[67_108_864, 16_777_216];

/// The default budget — the tightest published bound.
pub const ATTENTION_CHUNK_SIZE: u32 = 16_777_216;

/// Calibration identity for the 2.5 contract. Distinct from the 2.3 fingerprint: a contract that
/// declares two more rungs is not the same measurement subject, and sharing a fingerprint would let
/// 2.3's evidence authorize a 2.5 selection.
const CALIBRATION_FINGERPRINT: &str = "sc-18797-ltx-2-5-mlx-ladder-v1";

/// Whether this load can execute rung 4.
///
/// Three conditions, each of which is a real refusal in [`crate::block_stream::LtxBlockStream::new`]
/// rather than a policy invented here — the declaration and the loader must agree, or the contract
/// promises something the loader will reject at request time:
///
/// 1. **Deferred materialization.** Rung 4's shared prerequisite. A block window over an
///    already-materialized trunk adds a copy and bounds nothing.
/// 2. **No adapters.** LTX installs LoRA onto loaded block objects, so a per-window rebuild from the
///    base component would silently carry none of them.
/// 3. **A directory bundle.** The stream reopens the resolved *transformer component* file; a
///    single-file spec is not a 2.5 split bundle.
fn streamable(spec: &LoadSpec) -> bool {
    spec.load_shape == LoadShape::DeferredMaterialization
        && spec.adapters.is_empty()
        && matches!(spec.weights, WeightsSource::Dir(_))
}

fn strategies(spec: &LoadSpec) -> Vec<MemoryStrategyCapability> {
    let windowed = streamable(spec);
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident
                | MemoryStrategy::StagedResidency
                | MemoryStrategy::BoundedDecode
                | MemoryStrategy::BoundedAttention => MemoryStrategySupport::Implemented,
                MemoryStrategy::BoundedTransformerResidency if windowed => {
                    MemoryStrategySupport::Implemented
                }
                MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
            },
            parameters: match strategy {
                MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                    decode_tile_edges: decode_tile_edges(),
                    decode_overlaps: vec![DECODE_OVERLAP],
                    ..Default::default()
                },
                MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                    attention_chunk_sizes: ATTENTION_CHUNK_SIZES.to_vec(),
                    ..Default::default()
                },
                MemoryStrategy::BoundedTransformerResidency if windowed => MemoryParameterRanges {
                    transformer_window_sizes: TRANSFORMER_WINDOW_SIZES.to_vec(),
                    transformer_window_components: vec![TRANSFORMER_WINDOW_COMPONENT],
                    ..Default::default()
                },
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect()
}

/// Build the LTX-2.5 memory contract for `spec`.
pub fn memory_strategy_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let windowed = streamable(spec);
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let mut additional_prerequisites = vec![
        // Inherited from 2.3: LTX stages Gemma before the AvDiT on every render, so a selected
        // decode rung co-engages rung 1 even though the shared cost-order default does not.
        (
            MemoryStrategy::BoundedDecode,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ),
    ];
    if windowed {
        // **Gate streaming to `OffloadPolicy::Sequential`** (the story's explicit design decision).
        //
        // Rung 4 bounds the 48-block trunk. Under a resident phase policy the Gemma-4 text encoder
        // and the VAE stay co-resident for the whole request, so the request PEAK is set by them and
        // bounding the trunk moves it by nothing — while the streamed stack still re-materializes
        // every block once per step and pays roughly 2x the latency, for identical output. That is a
        // pure loss, and the honest place to say so is the contract.
        //
        // `MemoryStrategy::engages` cannot supply this edge: rung 4 does NOT engage rung 1
        // universally (SC-15998 removed exactly that assumption, because a resident phase policy can
        // legitimately coexist with deferred block materialization on families whose other
        // components are small). LTX's components are not small, which is why this is declared here
        // as a provider-specific prerequisite — the sanctioned seam — rather than argued into the
        // shared graph.
        additional_prerequisites.push((
            MemoryStrategy::BoundedTransformerResidency,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
    }
    Ok(MemoryProviderContract {
        provider_id: LTX_2_5_MODEL_ID.to_owned(),
        backend: MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            // Also what makes the MLX realization answer `DeviceFormatTransfer` for rung 4's
            // per-window cost obligation: a window maps lazy safetensors handles and reads the
            // block's own packed triples, with no host-side repack in the per-window path.
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
        strategies: strategies(spec),
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites,
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: true,
            // Rung 3 is wired through `Attention::attn_budget` on all six attentions of every block.
            attention_chunking: true,
            // Tracks the rung-4 support value exactly: conformance rejects an `Implemented` rung 4
            // whose lifecycle does not claim the capability, and claiming it on a non-streamable
            // load would be the inverse lie.
            transformer_window_materialization: windowed,
        },
        formula: MemoryFormulaKind::PhaseEnvelope {
            phases,
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::FrameCount,
                MemoryFormulaVariable::BatchCount,
                MemoryFormulaVariable::ConditioningTokenCount,
                MemoryFormulaVariable::OverlayBytes,
                MemoryFormulaVariable::DecodeTileArea,
            ],
        },
        calibration: Some(MemoryCalibrationIdentity::new(
            CALIBRATION_FINGERPRINT,
            spec.load_shape,
        )),
        asset_facts: gen_core::MemoryAssetFacts::default(),
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    })
}

/// The LTX-2.5 MLX memory registration.
///
/// Not yet passed to [`crate::register_providers`] — see the module docs: the `ltx_2_5` generator
/// descriptor is sc-18778, and `ProviderRegistryBuilder::build` refuses a memory strategy with no
/// matching generator.
pub const MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: LTX_2_5_MODEL_ID,
    contract: memory_strategy_contract,
    safety_check: registered_safety_check,
};

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::OffloadPolicy;

    fn spec(load_shape: LoadShape) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent-ltx-2-5".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(load_shape)
    }

    fn support(contract: &MemoryProviderContract, rung: MemoryStrategy) -> MemoryStrategySupport {
        contract
            .capability(rung)
            .expect("every rung declared")
            .support
            .clone()
    }

    /// The registration must name the 2.5 id, not 2.3's. A copy-paste that left `crate::MODEL_ID` in
    /// place would register a *duplicate* `ltx_2_3` strategy, which `build()` rejects — but only
    /// once a generator exists, so the failure would surface in sc-18778 rather than here.
    #[test]
    fn the_registration_names_the_two_five_engine_id() {
        assert_eq!(MEMORY_REGISTRATION.provider_id, "ltx_2_5");
        assert_ne!(MEMORY_REGISTRATION.provider_id, crate::MODEL_ID);
        let contract = memory_strategy_contract(&spec(LoadShape::DeferredMaterialization)).unwrap();
        assert_eq!(contract.provider_id, "ltx_2_5");
    }

    /// R9's core clause, on the axis this story owns: rung 4 is declared **only** where the loader
    /// can execute it. An eager load must not advertise it.
    #[test]
    fn rung_four_is_declared_only_for_a_deferred_load() {
        let deferred = memory_strategy_contract(&spec(LoadShape::DeferredMaterialization)).unwrap();
        assert_eq!(
            support(&deferred, MemoryStrategy::BoundedTransformerResidency),
            MemoryStrategySupport::Implemented
        );
        assert!(deferred.lifecycle.transformer_window_materialization);

        let eager = memory_strategy_contract(&spec(LoadShape::EagerMaterialization)).unwrap();
        assert_eq!(
            support(&eager, MemoryStrategy::BoundedTransformerResidency),
            MemoryStrategySupport::Missing,
            "an eager load has no block window to bound"
        );
        assert!(
            !eager.lifecycle.transformer_window_materialization,
            "the lifecycle capability must track the support value, not be pinned true"
        );
    }

    /// An adapted load cannot stream — `LtxBlockStream::new` refuses it — so the contract must not
    /// declare the rung for one. Declaration and loader must give the same answer.
    #[test]
    fn an_adapted_load_does_not_declare_rung_four() {
        let mut adapted = spec(LoadShape::DeferredMaterialization);
        adapted.adapters = vec![mlx_gen::AdapterSpec {
            path: "/nonexistent/lora.safetensors".into(),
            scale: 1.0,
            kind: mlx_gen::AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }];
        let contract = memory_strategy_contract(&adapted).unwrap();
        assert_eq!(
            support(&contract, MemoryStrategy::BoundedTransformerResidency),
            MemoryStrategySupport::Missing
        );
        assert!(!contract.lifecycle.transformer_window_materialization);
    }

    /// Rung 3 is unconditional: the budget threads through `Attention` on every load shape, so it is
    /// declared on both. This is the arm that would fail if rung 3 were copy-pasted onto rung 4's
    /// `streamable` gate.
    #[test]
    fn rung_three_is_declared_on_every_load_shape() {
        for shape in [
            LoadShape::EagerMaterialization,
            LoadShape::DeferredMaterialization,
        ] {
            let contract = memory_strategy_contract(&spec(shape)).unwrap();
            assert_eq!(
                support(&contract, MemoryStrategy::BoundedAttention),
                MemoryStrategySupport::Implemented,
                "rung 3 is a scratch bound, independent of materialization shape ({shape:?})"
            );
            assert!(contract.lifecycle.attention_chunking);
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedAttention)
                    .unwrap()
                    .parameters
                    .attention_chunk_sizes,
                ATTENTION_CHUNK_SIZES,
            );
        }
    }

    /// The story's "gate streaming to `Sequential`" decision, asserted as the contract edge that
    /// implements it. Without this prerequisite a rung-4 selection under a resident phase policy is
    /// admitted and buys nothing.
    #[test]
    fn rung_four_requires_staged_residency_in_the_same_request() {
        let contract = memory_strategy_contract(&spec(LoadShape::DeferredMaterialization)).unwrap();
        assert!(
            contract.additional_prerequisites.contains(&(
                MemoryStrategy::BoundedTransformerResidency,
                MemoryStrategyPrerequisite::Rung {
                    rung: MemoryStrategy::StagedResidency,
                    scope: MemoryPrerequisiteScope::EngagedInSameRequest,
                },
            )),
            "rung 4 under a resident phase policy bounds nothing and costs ~2x; the contract must \
             say so"
        );
        // And the shared graph must NOT have been edited to say it universally.
        assert!(
            !MemoryStrategy::BoundedTransformerResidency
                .requires()
                .iter()
                .any(|prerequisite| matches!(
                    prerequisite,
                    MemoryStrategyPrerequisite::Rung {
                        rung: MemoryStrategy::StagedResidency,
                        ..
                    }
                )),
            "this is a provider-specific edge; SC-15998 removed it from the shared graph"
        );
    }

    /// The published window domain must be a real domain: every candidate distinguishable in the
    /// bounded quantity, every candidate exactly executable over 48 blocks, and none of them the
    /// degenerate all-covering window.
    #[test]
    fn published_window_candidates_are_distinguishable_and_exact() {
        let n_blocks = crate::config::LtxConfig::video_only_defaults().num_layers as usize;
        assert_eq!(n_blocks, 48);
        for pair in TRANSFORMER_WINDOW_SIZES.windows(2) {
            assert_eq!(
                pair[1],
                pair[0] * 2,
                "consecutive candidates must differ by 2x in peak block residency, which is linear \
                 in window size — anything closer records sweep noise"
            );
        }
        for &window in TRANSFORMER_WINDOW_SIZES {
            let plan = gen_core::block_window::BlockPlan::new(n_blocks, window as usize).unwrap();
            assert!(
                plan.is_bounded(),
                "window {window} must bound something; an all-covering window pays the machinery \
                 for zero saving"
            );
            assert_eq!(
                plan.window_count(),
                n_blocks / window as usize,
                "window {window} must divide 48 exactly, or the sweep's independent variable is \
                 inexact"
            );
        }
        assert!(TRANSFORMER_WINDOW_SIZES.contains(&TRANSFORMER_WINDOW_SIZE));
        assert!(ATTENTION_CHUNK_SIZES.contains(&ATTENTION_CHUNK_SIZE));
    }

    /// The contract must satisfy gen-core's own conformance rules on every load shape it accepts.
    /// An error here is contract-level — a caller that bails loses rungs 0-4, not just the offender.
    #[test]
    fn the_contract_conforms_on_every_accepted_load_shape() {
        for shape in [
            LoadShape::EagerMaterialization,
            LoadShape::DeferredMaterialization,
        ] {
            let contract = memory_strategy_contract(&spec(shape)).unwrap();
            let errors = contract.conformance_errors();
            assert!(
                errors.is_empty(),
                "{shape:?} contract is non-conforming: {errors:?}"
            );
        }
    }

    /// The AV component-scope decision, pinned so a later edit to `Both` has to argue with it.
    #[test]
    fn the_declared_window_scope_is_the_dit_alone() {
        assert_eq!(TRANSFORMER_WINDOW_COMPONENT, TransformerComponent::Dit);
        assert!(TRANSFORMER_WINDOW_COMPONENT.includes_dit());
        assert!(
            !TRANSFORMER_WINDOW_COMPONENT.includes_text_encoder(),
            "the Gemma-4 encoder is a separate component that block_stream does not window; \
             claiming it would be an unreachable declaration"
        );
        let contract = memory_strategy_contract(&spec(LoadShape::DeferredMaterialization)).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .parameters
                .transformer_window_components,
            vec![TransformerComponent::Dit]
        );
    }
}
