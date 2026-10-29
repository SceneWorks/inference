//! The provider memory contract: strategies, resident components, architecture facts and the
//! selected DiT / VAE configs it is built from.

use super::*;

/// The load shape this contract is resolved at — the spec's, because both are now implemented.
///
/// The distinction from [`LOAD_SHAPE`] is the whole of sc-18662: a provider may only mirror a
/// requested shape once it actually has a loader for it, and answering
/// [`LoadShape::DeferredMaterialization`] without one is how a rung comes to be declared over
/// nothing. Both loaders exist and `tests/block_window_real.rs` measures each.
pub(super) fn resolved_load_shape(spec: &LoadSpec) -> LoadShape {
    spec.load_shape
}

/// The five capability entries, with the parameter domain each implemented lever owns.
///
/// Every lever the contract marks `Implemented` must publish a [`MemoryParameterRanges`] for the
/// parameters that lever consumes, and must publish **none** for the ones it does not — both
/// directions are enforced by `validate_owned_parameter_domain`. Rungs 0 and 1 own no numeric
/// parameters, so their ranges are legitimately empty; rungs 2, 3 and 4 own tile/chunk/window
/// domains and cannot be flipped to `Implemented` without filling them in.
/// Rung 3's inapplicability reason, built from [`RUNG3_MEASURED_PEAK_DELTAS`] rather than typed out.
///
/// A reason string is the *only* justification a `StructurallyNotApplicable` declaration carries, and
/// `conformance_errors` refuses an empty one — but nothing stops a non-empty one from going stale
/// against the measurement it claims to report. Deriving it from the same constant the tests assert
/// against means a re-measurement that moves a number moves the declaration with it.
pub(super) fn rung3_inapplicability_reason() -> String {
    let cells = RUNG3_MEASURED_PEAK_DELTAS
        .iter()
        .map(|(tier, frames, seq, heads, rows, one_head)| {
            format!(
                "{tier} {frames}f/seq {seq}: {:+.2}% heads, {:+.2}% query rows, {:+.2}% one-head",
                heads * 100.0,
                rows * 100.0,
                one_head * 100.0
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "MLX's fused SDPA streams the scores at this family's shape, so there is no materialized \
         [B,H,S,S] tensor for rung 3 to bound: measured peak tracks the 4·B·H·S·D q/k/v/out \
         prediction exactly (5.966 GB against 5.965 GB predicted at S=104030, where materializing \
         the scores would add ~1212 GB). The remaining MLX mechanism — eval_per_chunk cutting the \
         lazy graph, worth -1.7% on Z-Image's denoise phase — was measured in-forward on the real \
         50-block q4 DiT on both chunk axes and COSTS denoise peak at every arm ({cells}). \
         MLX-scoped: candle-gen-minimax-h3 materializes its scores and must measure its own verdict."
    )
}

pub(super) fn strategies(streamable: bool) -> Vec<MemoryStrategyCapability> {
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident
                | MemoryStrategy::StagedResidency
                // sc-18660. The mechanism is `crate::spatial_tiling::BoundedStitch` behind
                // `MiniMaxH3VideoVae::decode_clip`; the domain is one geometry, because the tile
                // edge is an output-correctness input rather than a lever (module docs).
                | MemoryStrategy::BoundedDecode => MemoryStrategySupport::Implemented,
                // sc-18661. Measured, not argued — see the module docs and
                // `RUNG3_MEASURED_PEAK_DELTAS`. MLX-scoped: candle materializes its scores.
                MemoryStrategy::BoundedAttention => {
                    MemoryStrategySupport::StructurallyNotApplicable {
                        reason: rung3_inapplicability_reason(),
                    }
                }
                // sc-18662. Implemented on a `LoadShape::DeferredMaterialization` load and
                // `Missing` on a resident one — the rung's own shared prerequisite is that shape, so
                // a resident load genuinely does not have it. Not `StructurallyNotApplicable`: a
                // 50-block DiT and a 50-layer transformer encoder are exactly the components this
                // rung optimizes, and both are measured to bound ~9x.
                //
                // sc-18650: `streamable` is **not** the load shape alone — a configured adapter
                // makes it false too, because a windowed block is rebuilt from the staged tier and
                // would drop the factors. See [`streamable`] for why that gate cannot live in
                // `route_gate`.
                MemoryStrategy::BoundedTransformerResidency if streamable => {
                    MemoryStrategySupport::Implemented
                }
                MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
            },
            parameters: match strategy {
                MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                    decode_tile_edges: vec![DECODE_TILE_EDGE],
                    decode_overlaps: vec![DECODE_OVERLAP],
                    ..Default::default()
                },
                // sc-18662. `Both`, and the second half is the point: declaring `Dit` alone would
                // leave the conditioning phase — the taller of the two stages at every tier — with
                // no lever at all, which is what AC3 refuses. Both arms are measured separately in
                // `tests/block_window_real.rs`.
                MemoryStrategy::BoundedTransformerResidency if streamable => {
                    MemoryParameterRanges {
                        transformer_window_sizes: vec![TRANSFORMER_WINDOW_SIZE],
                        transformer_window_components: vec![TransformerComponent::Both],
                        ..Default::default()
                    }
                }
                // `validate_owned_parameter_domain` enforces this in BOTH directions: an
                // implemented rung that does not own the decode parameters must publish none.
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect()
}

/// The AdaLN projection stack as a typed, evictable intra-transformer component (sc-18665).
///
/// Every field but `resident_bytes` is an architecture or checkpoint fact, so the shape of this
/// component is identical in the production and weights-free contracts — which is what lets catalog
/// conformance see the exclusion without a snapshot.
///
/// * `resident_bytes` is [`ComponentBytes::adaln`], i.e. the stack **on the tier that was actually
///   resolved** — see [`resolved_adaln_bytes`]. It is passed in rather than read from
///   [`ADALN_EVICTED_BYTES`] here because that constant is the **bf16** figure, and declaring it
///   against an 18.78 GB q4 DiT declares a sub-stack larger than the stack containing it, which
///   `conformance_errors` refuses and `Registry::memory_strategy_contract` turns into a hard error
///   — a q4 render that cannot resolve a contract at all. The weights-free contract resolves
///   nothing and so passes the architecture fact through [`ComponentBytes::weights_free`].
///
/// * `kind` is [`MemoryComponentKind::TransformerSubStack`], not `Transformer`: these bytes are
///   already inside `asset_facts.transformer_bytes`, and naming a whole transformer would charge
///   them twice.
/// * `bounded_by` is `None` because **no rung bounds this**. `crate::model::generate_impl` passes
///   [`crate::dit::adaln::AdaLnResidency::PrecomputeAndEvict`] as a *literal*, so nothing a request
///   can select reaches the other arm, and `AdaLnCache::precompute_and_evict` refuses `Resident`
///   outright. The type is not vacuous — `crate::dit::model` implements a live `Resident` arm for a
///   solver whose evaluation timesteps are not enumerable up front, and `tests/adaln_cache.rs`
///   drives it — but no shipped render selects it, which is what makes the exclusion declarable as
///   an unconditional property of this provider rather than as a lever.
pub(super) fn adaln_component(resident_bytes: u64, pruned: bool) -> MemoryResidentComponent {
    if pruned {
        return MemoryResidentComponent {
            id: ADALN_COMPONENT_ID.to_owned(),
            kind: MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
            resident_bytes,
            bounded_by: None,
            residency: MemoryComponentResidency::WholeRender,
        };
    }
    MemoryResidentComponent {
        id: ADALN_COMPONENT_ID.to_owned(),
        kind: MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
        resident_bytes,
        bounded_by: None,
        residency: MemoryComponentResidency::PrecomputedThenEvicted {
            // The projections are mapped with the rest of the DiT at the head of the denoise phase,
            // consumed once to project the whole schedule's modulation, and released before the
            // first denoise step. There is no earlier phase they survive into and no later one.
            precomputed_in: MemoryPhase::Denoise,
            retained_bytes: ADALN_MODULATION_TABLE_MAX_BYTES,
            evidence: format!(
                "tests/adaln_evict_real_weights.rs and tests/adaln_evict_memory.rs drive \
                 AdaLnCache::precompute_and_evict and pin the phase pair through \
                 common::assert_adaln_phase_envelope (sc-19449); the bf16 {ADALN_EVICTED_BYTES} B \
                 is asserted against the sum over the 50 real adaln_proj tensors in \
                 crate::dit::adaln, and resident_bytes above is that stack on the resolved tier"
            ),
        },
    }
}

/// Rung 4's prerequisites, both of them, resolved from the spec.
///
/// **The deferred load shape is necessary but not sufficient** (sc-18650). A configured adapter is
/// the second gate, and it belongs *here* rather than only in the render: `route_gate` cannot enforce
/// it, because [`MemoryRunContext`] carries no adapter axis at all — a selection is geometry, tier
/// and parameters, and the adapter list is a property of the *load*. So the only place the two can
/// be made to agree is the declaration.
///
/// Without this clause the contract published rung 4 `Implemented` for an adapter-carrying deferred
/// load, `validate_selection` admitted it, the request scope opened, and
/// `MemoryProviderContract::generation_memory` set `stream_transformer_blocks` from that engagement —
/// whereupon `crate::model::MiniMaxH3::requested_transformer_window` refused the very selection the
/// contract had just promised. Declaring `Missing` closes the loop at its head instead.
///
/// The refusal it mirrors is not a caprice: a window rebuilds each block from the staged tier per
/// step, so a forward-time residual adapter would be dropped from every windowed block. Both
/// precedents this provider mirrors gate on the same thing — `mlx-gen-wan`'s adapter residency mode
/// and `mlx-gen-ltx`'s overlay legs.
pub fn streamable(spec: &LoadSpec) -> bool {
    resolved_load_shape(spec) == LoadShape::DeferredMaterialization && spec.adapters.is_empty()
}

/// Architecture axes for the MiniMax-H3 route (epic SC-22657, E2).
///
/// [`MiniMaxH3DitConfig::default`](crate::dit::config::MiniMaxH3DitConfig) mirrors the reference
/// `transformer/config.json`, and `MiniMaxH3DitConfig::from_diffusers_json` parses that same file —
/// every key required, no defaults — at load, so the two cannot disagree silently.
///
/// `head_dim` is the config's declared `attention_head_dim`, not `hidden_size / num_heads`: this DiT
/// is deliberately non-uniform (`inner_dim` 7168 is wider than `hidden_size` 5376), so the quotient
/// would be neither integral nor the head width the attention actually uses.
///
/// The VAE is a **video** autoencoder: `patch_size` pixels and `patch_size_t` frames per latent
/// unit as `MiniMaxH3VaeConfig` states them (the products of the per-level downsample factors), so
/// `vae_temporal_scale` is a real value here.
///
/// When `spec` names a materialized snapshot directory, [`dit_config`] and [`vae_config`] re-run the
/// loaders' own `from_diffusers_json` parses over that snapshot's `transformer/config.json` and
/// `vae/config.json`, so every published axis is the snapshot's rather than the preset's
/// (SC-22667: the VAE axes were preset constants although `vae.rs` parses the snapshot config).
///
/// SC-22667 review: an install that HAS a DiT to load — a materialized snapshot, or a staged
/// [`DIT_COMPONENT`] directory — but whose `transformer/config.json` is missing, partial or
/// unparseable no longer degrades into the preset. `MiniMaxH3Dit::load_dir` errors on exactly that
/// file, so a preset published here would describe a model this load will never build. A provider
/// falls back to a preset only where the LOADER falls back to it; otherwise the trunk axes are
/// declared absent. The VAE axes are their own config read (see [`vae_config`]) and normally survive
/// that branch, so the contract still declares a real architecture axis and the MLX weights-free
/// gate stays satisfied.
pub(super) fn architecture_facts(spec: &LoadSpec) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    // SC-22667 review: a `vae/config.json` that is present but does not parse declines the VAE axes
    // rather than substituting the crate mirror — `vae.rs` reads that same file through the same
    // `from_diffusers_json` and propagates the failure, so no load of this snapshot has the mirror's
    // geometry. An ABSENT file keeps the mirror: that is the shape the weights-free surface and an
    // unmaterialized VAE both have, and it is the only branch where there is nothing to contradict.
    let vae = vae_config(spec);
    let latent_channels = vae
        .as_ref()
        .and_then(|vae| mlx_gen::architecture_facts::axis(vae.latent_channels));
    let vae_spatial_scale = vae
        .as_ref()
        .and_then(|vae| mlx_gen::architecture_facts::axis(vae.patch_size));
    let vae_temporal_scale = vae
        .as_ref()
        .and_then(|vae| mlx_gen::architecture_facts::axis(vae.patch_size_t));
    let Some(dit) = dit_config(spec) else {
        return mlx_gen::gen_core::MemoryArchitectureFacts {
            attention_heads: None,
            head_dim: None,
            transformer_blocks: None,
            patch_size: None,
            latent_channels,
            vae_spatial_scale,
            vae_temporal_scale,
            activation_dtype_width: Some(if spec.precision == mlx_gen::Precision::Fp32 {
                mlx_gen::architecture_facts::FLOAT32_ACTIVATION_WIDTH
            } else {
                mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH
            }),
        };
    };
    // `patch_size` is `(t, h, w)`; only the square spatial patch has a single honest scalar, and
    // the temporal factor is already carried by `vae_temporal_scale`.
    let [_, patch_h, patch_w] = dit.patch_size;
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(dit.num_attention_heads),
        head_dim: mlx_gen::architecture_facts::axis(dit.attention_head_dim),
        transformer_blocks: mlx_gen::architecture_facts::axis(dit.num_layers),
        patch_size: (patch_h == patch_w)
            .then(|| mlx_gen::architecture_facts::axis(patch_h))
            .flatten(),
        latent_channels,
        vae_spatial_scale,
        vae_temporal_scale,
        // SC-22667: this hardcoded `HALF_ACTIVATION_WIDTH` while its own comment named the
        // exception. gen-core documents the axis as the DENOISE-phase width — explicitly not a
        // per-component byte fact — and `model.rs` selects `Dtype::Float32` for the DiT under
        // `Precision::Fp32` and `Dtype::Bfloat16` otherwise, so an Fp32 load published half the
        // width its own denoise loop computes at. The f32 audio VAE is deliberately NOT folded in
        // here: it belongs to the decode phase, which this axis does not describe.
        activation_dtype_width: Some(if spec.precision == mlx_gen::Precision::Fp32 {
            mlx_gen::architecture_facts::FLOAT32_ACTIVATION_WIDTH
        } else {
            mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH
        }),
    }
}

/// The DiT geometry a contract should describe: the config the loader would parse, or `None` when
/// there IS a DiT to load and that config cannot be read.
///
/// `from_diffusers_json` requires every key, so a partial or variant file declines rather than
/// half-defaulting into this model's numbers.
///
/// SC-22667: the DiT is honored at its **resolved** location, not under the snapshot root. It is
/// the one tiered component, so a split `q4` install stages [`DIT_COMPONENT`] outside the snapshot
/// and has no `root/transformer` at all — `model.rs` says exactly that where it probes the same
/// file, and `ComponentBytes::resolve` already applies the staged override for both the DiT and the
/// text encoder. Reading only under the root made the probe miss on every split install, so the
/// preset was published although the loaded config was one `spec.components` lookup away.
///
/// SC-22667 review: the preset is returned only where the LOADER would use it — a spec that names
/// neither a materialized root nor a staged DiT directory, i.e. the weights-free registry surface
/// with no load to describe. Once either is present, `MiniMaxH3Dit::load_dir` reads
/// `<resolved>/config.json` and propagates any read or parse failure, so this returns `None` and the
/// caller declares the trunk axes absent instead of substituting numbers the load cannot produce.
pub(super) fn dit_config(spec: &LoadSpec) -> Option<crate::dit::config::MiniMaxH3DitConfig> {
    let staged = matches!(
        spec.components.get(DIT_COMPONENT),
        Some(WeightsSource::Dir(_))
    );
    if !staged && mlx_gen::architecture_facts::materialized_root(spec).is_none() {
        return Some(crate::dit::config::MiniMaxH3DitConfig::default());
    }
    resolved_dit_config(spec)
}

/// The DiT config as the loader would resolve it, or `None` when that directory carries no
/// parseable one.
///
/// The directory precedence is `resolve_dit_dir`'s exactly: a staged [`DIT_COMPONENT`] directory
/// WINS OUTRIGHT and is never followed by a root fallback, because the loader does not fall back
/// either — a staged tier whose `config.json` is unreadable fails the load rather than silently
/// building the snapshot's base geometry.
pub(super) fn resolved_dit_config(
    spec: &LoadSpec,
) -> Option<crate::dit::config::MiniMaxH3DitConfig> {
    let dir = match spec.components.get(DIT_COMPONENT) {
        Some(WeightsSource::Dir(staged)) => staged.clone(),
        _ => mlx_gen::architecture_facts::materialized_root(spec)?
            .join(crate::model::BASE_DIT_PARTITION),
    };
    let text = mlx_gen::gen_core::__model_input_reader!(
std::fs::read_to_string(dir.join("config.json")),
"crates/media/mlx-gen/mlx-gen-minimax-h3/src/memory_strategy/contract.rs::resolved_dit_config::std::fs::read_to_string@19222",
ArtifactInput,
dir.join("config.json"),
|__model_input_path| std::fs::read_to_string(__model_input_path)
).ok()?;
    crate::dit::config::MiniMaxH3DitConfig::from_diffusers_json(&text).ok()
}

/// The video-VAE geometry a contract should describe: the materialized snapshot's own
/// `vae/config.json` — the file `vae.rs` parses through this same
/// `MiniMaxH3VaeConfig::from_diffusers_json` at load — when one exists and parses, this crate's
/// mirror of the shipped VAE when there is no such file to read, and `None` when the file is there
/// and does not parse.
///
/// That last arm is the SC-22667 review fix, and it mirrors [`dit_config`]: `from_diffusers_json`
/// requires every key, so a partial or variant file declines whole rather than half-defaulting —
/// but declining used to mean `unwrap_or_default()`, i.e. publishing the crate mirror's geometry
/// for a snapshot whose own file says something else. `MiniMaxH3Vae::load` reads that exact path
/// and propagates both the read and the parse failure, so a load of such a tree does not exist to
/// describe, and E2 declares the axes absent instead.
///
/// An ABSENT file keeps the mirror deliberately: the weights-free registry surface has no snapshot
/// at all, and a caller building a contract before the VAE component is materialized is in the same
/// position — nothing there contradicts the shipped geometry.
pub(super) fn vae_config(spec: &LoadSpec) -> Option<crate::config::MiniMaxH3VaeConfig> {
    let Some(root) = mlx_gen::architecture_facts::materialized_root(spec) else {
        return Some(crate::config::MiniMaxH3VaeConfig::default());
    };
    let Ok(text) = mlx_gen::gen_core::__model_input_reader!(
std::fs::read_to_string(root.join("vae").join("config.json")),
"crates/media/mlx-gen/mlx-gen-minimax-h3/src/memory_strategy/contract.rs::vae_config::std::fs::read_to_string@21099",
ArtifactInput,
root.join("vae").join("config.json"),
|__model_input_path| std::fs::read_to_string(__model_input_path)
) else {
        return Some(crate::config::MiniMaxH3VaeConfig::default());
    };
    crate::config::MiniMaxH3VaeConfig::from_diffusers_json(&text).ok()
}

pub(super) fn build_contract(
    spec: &LoadSpec,
    components: &ComponentBytes,
    load_shape: LoadShape,
    streamable: bool,
    // sc-22737: passed in rather than hard-coded, so the weights-free surface cannot publish the
    // production identity and the production one is keyed on the artifact's own tier.
    calibration: Option<MemoryCalibrationIdentity>,
) -> MemoryProviderContract {
    MemoryProviderContract {
        phase_facts: components.native_quant.as_ref().map(|native| {
            let mut facts = mlx_gen::gen_core::MemoryPhaseFacts::staged(
                mlx_gen::gen_core::StagedWeightSchedule::ThreeStage,
            );
            facts.native_quant = Some(native.clone());
            facts
        }),
        architecture_facts: architecture_facts(spec),
        provider_id: MODEL_ID.to_owned(),
        backend: MemoryBackendRealization::MlxMetal {
            // This flag rests on the AdaLN evict and nothing else. That evict drains the allocator
            // cache rather than migrating active → cache, because
            // `crate::dit::adaln::drain_allocator_cache` repeats `clear_cache()` while active is
            // still falling — and it is the one site sc-18665's exclusion is declared against, so
            // the declaration stands on a drain that is real where the bytes are claimed.
            //
            // It is deliberately NOT the claim it used to carry — that "the AdaLN evict AND EVERY
            // PHASE RELEASE" drain — and the retracted half is deleted rather than restated. The
            // phase releases are a separate question, owned by sc-17151. This comment therefore
            // states no `clear_cache()` call count for them: such a count is a fact about another
            // story's code, and it would go stale the moment sc-17151 lands — that story moves the
            // release path onto a shared retrying drain, which can only strengthen this flag.
            bounded_wired_residency: true,
            // MLX mmaps and materializes per tensor: a bare 66 GB `MiniMaxH3Dit::load` leaves peak
            // memory at 33 KB. This is the *intra-tensor* fact, and it is independent of
            // `LOAD_SHAPE`, which is about a block schedule.
            lazy_or_mmap_materialization: true,
            // `mlx_rs::transforms::eval` is forced before every release; under lazy evaluation a
            // drop without it frees nothing.
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
        strategies: strategies(streamable),
        // No decode-quality geometry table is declared for this route, and that is a statement of
        // fact rather than a declared-but-refused authority — identical to every other MLX video
        // provider after the sc-18325 sweep.
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape,
        // The shared graph is sufficient: rung 1 is selected explicitly and depends on nothing, and
        // no rung this provider implements adds a realization-specific edge. sc-18662 adds rung 4's.
        additional_prerequisites: Vec::new(),
        // Nothing to exclude: the cost-order default never drags in an unimplemented rung, because
        // `MemoryProviderContract::engages` already intersects with declared support.
        //
        // sc-18665 asked whether the AdaLN evict belongs here. It does not, and the two are not
        // alternatives: a `MemoryStrategyEngagementExclusion` removes a *rung* from a selection's
        // engaged composition, and the evict is not a rung — it removes *bytes* from a formula.
        // Conformance would refuse the attempt anyway, since both ends of an exclusion must be
        // implemented strategies. The evict is declared on the formula's resident component instead
        // (`adaln_component`), which is also where its unconditionality is recorded.
        default_engagement_exclusions: Vec::new(),
        // The staged phase order is hardcoded in `generate_impl`, not a load-time default a
        // `GenerationMemory` block could switch off, so an explicit all-disabled block would
        // misrepresent what a `Resident` selection gets. Leaving the block absent preserves exactly
        // the behavior the loader has.
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases(),
            synchronized_phase_release: true,
            // sc-18660. `conformance_errors` requires this hook whenever rung 2 is `Implemented`,
            // and its converse forbids declaring rung 2 `StructurallyNotApplicable` while it is
            // set — so the pair cannot drift apart.
            decode_tiling: true,
            attention_chunking: false,
            // sc-18662. `conformance_errors` requires this hook whenever rung 4 is `Implemented`
            // and forbids it while the rung is `StructurallyNotApplicable`, so the pair cannot
            // drift; here it tracks the resolved load shape for the same reason the rung does.
            transformer_window_materialization: streamable,
        },
        // A phase envelope is *max over phases*, which is the whole point for this family: the
        // floor is `max(TE, DiT, VAE)`, never the sum, because the three are never co-resident.
        //
        // `FrameCount` is the axis no image family needed. It is declared even though the peak
        // sc-18659 observed was flat across duration, because that flatness belonged to the
        // conditioning stage: the denoise and decode phase expressions are genuinely
        // frame-dependent. sc-19120's packed text-encoder tiers landed and removed the TE's mark
        // (see `CONDITIONING_STAGE_PEAK_Q4_BYTES`), so the later stages are legible and the
        // variable is load-bearing rather than anticipatory. `ConditioningTokenCount` is the
        // conditioning phase's only real input.
        //
        // sc-18665 makes this the *component* variant. The AdaLN projections are the one part of
        // the DiT that does not survive into the denoise steady state, and a plain `PhaseEnvelope`
        // has nowhere to say so — `asset_facts.transformer_bytes` is a single load-exact scalar, so
        // an estimate built from it charges 26 GB the runtime does not hold.
        formula: MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: phases(),
            variables: {
                let mut variables = vec![
                    MemoryFormulaVariable::AssetBytes,
                    MemoryFormulaVariable::PixelCount,
                    MemoryFormulaVariable::FrameCount,
                    MemoryFormulaVariable::ConditioningTokenCount,
                    // sc-18660. The decode phase's scratch is a function of the tile area, not the
                    // canvas area — one 256 px tile is decoded at a time regardless of canvas. The
                    // variable is declared because the phase expression needs it; its coefficient is
                    // calibration evidence and is NOT measured yet (see the sc-18660 notes).
                    MemoryFormulaVariable::DecodeTileArea,
                ];
                // Declared **only when an adapter is configured** (sc-18650), so an ordinary load's
                // formula is byte-for-byte the one this provider has always published and no
                // existing calibration record is disturbed. `conformance_errors` requires this
                // variable exactly when a typed auxiliary component is present, and the two are
                // built from the same condition here so they cannot drift.
                if components.overlay > 0 {
                    variables.push(MemoryFormulaVariable::OverlayBytes);
                }
                variables
            },
            resident_components: {
                let mut resident = Vec::new();
                if components.adaln > 0 {
                    let mut component =
                        adaln_component(components.adaln, components.pruning.is_some());
                    if components.pruning.is_none() {
                        if let Some(native) = &components.native_quant {
                            if let MemoryComponentResidency::PrecomputedThenEvicted {
                                retained_bytes,
                                ..
                            } = &mut component.residency
                            {
                                let rows = SourcePruneSizing::conditioned_timestep_bound(
                                    crate::model::MAX_STEPS as usize,
                                )
                                .expect("nonempty native schedule");
                                *retained_bytes = native
                                    .tasks
                                    .iter()
                                    .map(|t| t.table_bytes_per_timestep.saturating_mul(rows as u64))
                                    .max()
                                    .unwrap_or(0);
                            }
                        }
                    }
                    // A supplied small projection stack can retain more cache than it frees.
                    // Native phases still charge that cache; no generic net eviction is claimed.
                    let no_saving = components.native_quant.is_some()
                        && matches!(component.residency,
                        MemoryComponentResidency::PrecomputedThenEvicted { retained_bytes, .. }
                        if retained_bytes >= component.resident_bytes);
                    if !no_saving {
                        resident.push(component);
                    }
                }
                // The LoRA stack as a typed auxiliary component. `MemoryComponentKind::AdapterStack`
                // is the ltx spelling for the same forward-time-residual install, and
                // `WholeRender` is literal here: `crate::model::load_task_dit` installs the factors
                // onto the task's DiT before the first step and nothing releases them until the DiT
                // itself goes at the end of denoise.
                if components.overlay > 0 {
                    resident.push(MemoryResidentComponent {
                        id: "adapter_stack".to_owned(),
                        kind: MemoryComponentKind::AdapterStack,
                        resident_bytes: components.overlay,
                        bounded_by: None,
                        residency: MemoryComponentResidency::WholeRender,
                    });
                }
                resident
            },
        },
        // The calibration identity carries the RESOLVED shape, not the resident default: a
        // deferred load's peaks are a different curve from a resident load's, and an evidence
        // record that named the wrong one would be matched against measurements of the other.
        calibration,
        asset_facts: MemoryAssetFacts {
            base_bytes: components.base(),
            conditioning_bytes: components.text_encoder,
            transformer_bytes: components.dit,
            decoder_bytes: components.decoder(),
            // **The configured LoRA stack, and nothing else** (sc-18650).
            //
            // This said "MiniMax-H3 accepts no adapters … `reject_unknown_components` allows only
            // `transformer`" until this story. Both halves were false and had been since sc-18724
            // landed adapter support on 2026-08-13: the descriptor declares `supports_lora: true`,
            // `crate::model::load_task_dit` installs the published `lightx2v/Minimax-h3-Turbo`
            // exports through `crate::adapters::apply_minimax_h3_adapters`, and
            // `reject_unknown_components` is called with **two** components (`transformer` and
            // `text_encoder`). The comment simply predated the feature, and while the contract went
            // unpublished nothing read it.
            //
            // The rest of the sentence is still true and is what this field means: there is no
            // ControlNet, no IP-adapter and no identity encoder on this family, so the LoRA stack is
            // the only auxiliary residency there is.
            overlay_bytes: components.overlay,
        },
        runtime: Default::default(),
    }
}

/// The production contract: asset facts read off the resolved snapshot.
pub fn contract_for(spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    let load_shape = resolved_load_shape(spec);
    Ok(build_contract(
        spec,
        &ComponentBytes::resolve(spec)?,
        load_shape,
        streamable(spec),
        // The identity carries the RESOLVED shape, not the resident default: a deferred load's
        // peaks are a different curve from a resident load's.
        production_calibration_fingerprint(spec)
            .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, load_shape)),
    ))
}

/// The weights-free fixture contract: the identical route declaration with zero asset facts.
///
/// Catalog conformance uses this when the snapshot is unavailable. It must **not** require the
/// snapshot to exist — no byte count is read off disk here — and it must not diverge from
/// [`contract_for`] in anything but the byte counts. It does consult the spec's weights directory
/// for the DiT config when one is materialized (see the crate-private `dit_config`); that is not a
/// byte count, and on the registry's never-created sentinel path it reads nothing and publishes
/// the preset.
pub fn weights_free_contract(spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    let load_shape = resolved_load_shape(spec);
    Ok(build_contract(
        spec,
        &ComponentBytes::weights_free(),
        load_shape,
        streamable(spec),
        Some(MemoryCalibrationIdentity::new(
            STATIC_CALIBRATION_FINGERPRINT,
            load_shape,
        )),
    ))
}
