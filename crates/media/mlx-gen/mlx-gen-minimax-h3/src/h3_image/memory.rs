//! MiniMax-H3-Image's shared-ladder memory contract on MLX/Metal.
//!
//! **What is implemented, and nothing more.**
//!
//! * [`MemoryStrategy::StagedResidency`] — structural: `generate` builds the text encoder, encodes,
//!   forces the context and releases it before the DiT is mapped; denoises and releases the DiT
//!   before the VAE is mapped. There is no co-resident mode, so the descriptor sets
//!   `unconditionally_engages_staged_residency` and every selection stages.
//! * [`MemoryStrategy::BoundedDecode`] — the image VAE's shipped 256 px / 64 px tiled decode
//!   (SW-7089's [`crate::image_vae::bounded_decode_parameters`]). The domain is that singleton:
//!   upstream releases the blended-tile images, so the tile edge is an output-correctness input,
//!   not a lever. Any other geometry is refused at admission and again at decode.
//! * [`MemoryStrategy::BoundedAttention`] and [`MemoryStrategy::BoundedTransformerResidency`] are
//!   `Missing`. H3 video's measured verdicts are not copied: this family's sequence and residency
//!   differ, and the ladder for it is a later item.
//!
//! No calibration is claimed for a production load (nothing has been measured); the weights-free
//! registry surface carries its own static identity so behavior conformance can run.

use mlx_gen::gen_core::{
    safetensors_path_bytes, LoadShape, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryNumericTier, MemoryPhase, MemoryProviderContract,
    MemoryRequestScope, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategySupport, WeightsSource,
};

use crate::h3_image::dit::IMAGE_DIT_COMPONENT;
use crate::h3_image::model::IMAGE_MODEL_ID;
use crate::image_vae::{
    bounded_decode_parameters, IMAGE_DECODE_OVERLAP, IMAGE_DECODE_TILE_EDGE, IMAGE_VAE_COMPONENT,
};
use crate::model::TEXT_ENCODER_COMPONENT;

/// Calibration identity of the weights-free registry conformance walk. Never published by a
/// production load, which reads real bytes and has no measured anchor.
pub const STATIC_CALIBRATION_FINGERPRINT: &str = "minimax-h3-image-mlx-registry-behavior-v1";

/// DiT blocks at the shipped geometry, for the request scope.
const DIT_BLOCKS: usize = 50;

/// Refuse every decode geometry except the shipped one.
pub fn validate_decode_geometry(edge: u32, overlap: u32) -> mlx_gen::gen_core::Result<()> {
    if edge == IMAGE_DECODE_TILE_EDGE && overlap == IMAGE_DECODE_OVERLAP {
        return Ok(());
    }
    Err(mlx_gen::gen_core::Error::Unsupported(format!(
        "{IMAGE_MODEL_ID}: bounded decode admits only the shipped geometry {IMAGE_DECODE_TILE_EDGE}px \
         tile / {IMAGE_DECODE_OVERLAP}px overlap, got {edge}/{overlap}; the released images are \
         the blended-tile ones, so another tile edge changes the picture"
    )))
}

fn phases() -> Vec<MemoryPhase> {
    vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ]
}

/// Float width the loader opens the DiT block stack and the VAE decode half at:
/// [`mlx_gen::Precision::Fp32`] casts them to float32 (`model.rs` maps it to `Dtype::Float32` and
/// passes it to `MiniMaxH3ImageDit::load` and `MiniMaxH3ImageVae::load`); every other precision
/// casts them to bfloat16.
fn load_float_width(spec: &LoadSpec) -> u64 {
    if spec.precision == mlx_gen::Precision::Fp32 {
        4
    } else {
        2
    }
}

/// Whether the image DiT loader casts tensor `name` to the requested dtype. The token refiner and
/// the 50 blocks are cast; the top-level input/output tensors (`ImageDitProjections`: `proj_in`,
/// `context_embedder`, `time_embedder`, `norm_out`, `pdd_proj_out.*`) keep their stored dtype.
fn dit_tensor_is_cast(name: &str) -> bool {
    name.starts_with("token_refiner.") || name.starts_with("transformer_blocks.")
}

/// Whether the image VAE loader reads tensor `name` (the decode half only; it casts every one).
fn vae_tensor_is_read(name: &str) -> bool {
    name.starts_with("decoder.") || name.starts_with("post_quant_conv.")
}

/// Bytes one component holds once its loader has opened it, from the safetensors headers alone.
///
/// `cast(name)` is `None` for a tensor the loader does not read, `Some(false)` for one held at its
/// stored dtype, and `Some(true)` for a float tensor the loader casts to `float_width`. A cast
/// tensor is priced at the wider of its stored and its cast width: MLX casts lazily over the
/// evaluated source, so the stored-width source is never less than what the load holds. A missing
/// component prices 0 (the loader refuses it later); a malformed one is an error.
fn loaded_component_bytes(
    path: &std::path::Path,
    float_width: u64,
    cast: &dyn Fn(&str) -> Option<bool>,
) -> mlx_gen::gen_core::Result<u64> {
    if safetensors_path_bytes(path) == 0 {
        return Ok(0);
    }
    mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(path)?
        .iter()
        .try_fold(0_u64, |sum, header| {
            let bytes = match cast(&header.name) {
                None => 0,
                Some(true) if header.is_float() => header
                    .materialized_bytes(float_width)?
                    .max(header.data_bytes),
                Some(_) => header.data_bytes,
            };
            sum.checked_add(bytes).ok_or_else(|| {
                mlx_gen::gen_core::Error::Msg(format!(
                    "{IMAGE_MODEL_ID}: {} loaded byte sum overflow",
                    path.display()
                ))
            })
        })
}

/// The bytes each phase holds (0 for a weights-free spec). The text encoder is read at its stored
/// dtype (on-disk shard lengths); the DiT and the VAE decode half are priced at the dtype
/// `spec.precision` loads them at, from their tensor headers.
fn asset_facts(spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryAssetFacts> {
    let Some(root) = mlx_gen::gen_core::resolved_snapshot_root(spec) else {
        return Ok(MemoryAssetFacts::default());
    };
    let te_dir = match spec.components.get(TEXT_ENCODER_COMPONENT) {
        Some(WeightsSource::Dir(dir)) => dir.clone(),
        _ => root.join(TEXT_ENCODER_COMPONENT),
    };
    // t2i maps the language-model shards only (`crate::text_encoder::TE_SHARDS`).
    let conditioning: u64 = crate::text_encoder::TE_SHARDS
        .map(|i| safetensors_path_bytes(te_dir.join(format!("model-{i:05}-of-00014.safetensors"))))
        .sum();
    let width = load_float_width(spec);
    let transformer = loaded_component_bytes(&root.join(IMAGE_DIT_COMPONENT), width, &|name| {
        Some(dit_tensor_is_cast(name))
    })?;
    let decoder = loaded_component_bytes(&root.join(IMAGE_VAE_COMPONENT), width, &|name| {
        vae_tensor_is_read(name).then_some(true)
    })?;
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

fn architecture_facts(spec: &LoadSpec) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    let dit = crate::dit::config::MiniMaxH3ImageDitConfig::default().base;
    let vae = crate::config::MiniMaxH3VaeConfig::image();
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(dit.num_attention_heads),
        head_dim: mlx_gen::architecture_facts::axis(dit.attention_head_dim),
        transformer_blocks: mlx_gen::architecture_facts::axis(dit.num_layers),
        patch_size: mlx_gen::architecture_facts::axis(dit.patch_size[1]),
        latent_channels: mlx_gen::architecture_facts::axis(vae.latent_channels),
        vae_spatial_scale: mlx_gen::architecture_facts::axis(vae.patch_size),
        // An image is one latent frame; the temporal axis does not apply.
        vae_temporal_scale: None,
        activation_dtype_width: Some(if spec.precision == mlx_gen::Precision::Fp32 {
            mlx_gen::architecture_facts::FLOAT32_ACTIVATION_WIDTH
        } else {
            mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH
        }),
    }
}

fn build_contract(
    spec: &LoadSpec,
    asset_facts: MemoryAssetFacts,
    calibration: Option<MemoryCalibrationIdentity>,
) -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        IMAGE_MODEL_ID,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident
            | MemoryStrategy::StagedResidency
            | MemoryStrategy::BoundedDecode => MemoryStrategySupport::Implemented,
            MemoryStrategy::BoundedAttention | MemoryStrategy::BoundedTransformerResidency => {
                MemoryStrategySupport::Missing
            }
        };
        if capability.strategy == MemoryStrategy::BoundedDecode {
            capability.parameters = bounded_decode_parameters();
        }
    }
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: phases(),
        synchronized_phase_release: true,
        decode_tiling: true,
        attention_chunking: false,
        transformer_window_materialization: false,
    };
    contract.phase_facts = Some(mlx_gen::gen_core::MemoryPhaseFacts::staged(
        mlx_gen::gen_core::StagedWeightSchedule::ThreeStage,
    ));
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: phases(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::DecodeTileArea,
        ],
    };
    contract.load_shape = LoadShape::EagerMaterialization;
    contract.architecture_facts = architecture_facts(spec);
    contract.asset_facts = asset_facts;
    contract.calibration = calibration;
    contract
}

/// The production contract: real file bytes, no calibration identity (unmeasured).
pub fn contract_for(spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_contract(spec, asset_facts(spec)?, None))
}

/// The weights-free contract catalog conformance resolves instead of [`contract_for`].
pub fn weights_free_contract(spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_contract(
        spec,
        MemoryAssetFacts::default(),
        Some(MemoryCalibrationIdentity::new(
            STATIC_CALIBRATION_FINGERPRINT,
            LoadShape::EagerMaterialization,
        )),
    ))
}

/// The family's admission predicates the shared pipeline cannot know: the decode geometry, no PiD
/// route, one image per sequence (a batch renders sequentially), one latent frame, a positive
/// requested size whose generation canvas has a latent grid.
pub fn route_gate(context: &MemoryRunContext) -> mlx_gen::gen_core::Result<()> {
    if context.use_pid {
        return Err(mlx_gen::gen_core::Error::Unsupported(format!(
            "{IMAGE_MODEL_ID} has no PiD decode route"
        )));
    }
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
    let g = context.geometry;
    if g.frames > 1 {
        return Err(mlx_gen::gen_core::Error::Unsupported(format!(
            "{IMAGE_MODEL_ID} renders still images, got {} frames",
            g.frames
        )));
    }
    admission_canvas(g.width, g.height)?;
    Ok(())
}

/// The canvas a `width x height` request renders on, which is what its resources are sized from.
///
/// Admission takes the **requested** size, exactly as the generator does: any positive size
/// ([`mlx_gen::gen_core::h3_image::validate_size`]). The generator renders on
/// [`mlx_gen::gen_core::h3_image::generation_canvas`] at the default minimum area and resizes back,
/// so the latent grid, the packed sequence and the decode all follow the canvas, not the request.
/// The admitted geometry stays the requested size, so the request scope's equality check binds
/// the request the caller actually sends.
pub fn admission_canvas(width: u32, height: u32) -> mlx_gen::gen_core::Result<(u32, u32)> {
    use mlx_gen::gen_core::h3_image::{
        generation_canvas, latent_grid, validate_size, MIN_GENERATION_AREA_DEFAULT,
    };
    validate_size(width, height).map_err(mlx_gen::gen_core::Error::from)?;
    let canvas = generation_canvas(width, height, MIN_GENERATION_AREA_DEFAULT);
    latent_grid(canvas.0, canvas.1)?;
    Ok(canvas)
}

fn profile_route_gate(
    _contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    _tier: MemoryNumericTier,
) -> mlx_gen::gen_core::Result<()> {
    route_gate(context)
}

/// The shared-skeleton profile: requested tier, the route gate above, text-to-image fixtures.
pub(crate) const MEMORY_PROFILE: mlx_gen::gen_core::ProviderMemoryProfile =
    mlx_gen::gen_core::ProviderMemoryProfile {
        gates: mlx_gen::gen_core::MemoryProfileGates::within(profile_route_gate),
        ..mlx_gen::gen_core::ProviderMemoryProfile::STANDARD
    };

/// The loaded generator's safety check.
pub fn safety_check_at_tier(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    mlx_gen::gen_core::standard_safety_check(&MEMORY_PROFILE, contract, context, tier)
}

fn begin_request_with_cleanup(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    mlx_gen::request_scope::begin_scope(
        safety_check_at_tier(contract, tier, context),
        cleanup,
        move || {
            mlx_gen::request_scope::MlxRequestScopeConfig::new(
                IMAGE_MODEL_ID,
                context.geometry,
                contract.generation_memory(&context.selection),
                context.use_pid,
                DIT_BLOCKS,
                |_use_pid, edge, overlap| validate_decode_geometry(edge, overlap),
            )
        },
    )
}

/// The loaded generator's request scope (device cleanup on exit).
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

fn registered_scope(
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

/// The memory-strategy registration paired with the `minimax_h3_image` generator.
pub const IMAGE_MEMORY_REGISTRATION: mlx_gen::gen_core::MemoryRegistration =
    mlx_gen::gen_core::profile_memory_registration!(MEMORY_PROFILE, IMAGE_MODEL_ID, contract_for);

/// The weights-free contract fixture.
pub const IMAGE_MEMORY_CONTRACT_FIXTURE: mlx_gen::gen_core::MemoryContractFixtureRegistration =
    mlx_gen::gen_core::MemoryContractFixtureRegistration {
        provider_id: IMAGE_MODEL_ID,
        contract: weights_free_contract,
        surface_specs: memory_contract_surface_specs,
    };

/// The executable behavior seam every implemented optimized rung must have.
pub const IMAGE_MEMORY_BEHAVIOR: mlx_gen::gen_core::MemoryBehaviorRegistration = mlx_gen::gen_core::profile_memory_behavior_registration!(
    MEMORY_PROFILE,
    IMAGE_MODEL_ID,
    registered_scope
);

/// The surfaces this route publishes: the dense bf16 / fp32 eager loads (no packed tier and no
/// deferred loader exist for this family yet).
fn memory_contract_surface_specs() -> Vec<mlx_gen::gen_core::MemoryContractSurfaceSpec> {
    mlx_gen::gen_core::mlx_memory_contract_surface_specs()
        .into_iter()
        .filter(|s| {
            s.spec.quantize.is_none()
                && s.spec.load_shape == LoadShape::EagerMaterialization
                && s.spec.offload_policy == mlx_gen::gen_core::OffloadPolicy::Resident
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights_free() -> MemoryProviderContract {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-minimax-h3-image".into()));
        weights_free_contract(&spec).unwrap()
    }

    /// Exactly what is implemented: staged residency and the singleton tiled decode. Rungs 3 and
    /// 4 are `Missing`, not copied from H3 video's measured verdicts.
    #[test]
    fn declares_staged_residency_and_bounded_decode_only() {
        let c = weights_free();
        let support = |s| &c.capability(s).unwrap().support;
        assert_eq!(
            support(MemoryStrategy::StagedResidency),
            &MemoryStrategySupport::Implemented
        );
        assert_eq!(
            support(MemoryStrategy::BoundedDecode),
            &MemoryStrategySupport::Implemented
        );
        assert_eq!(
            support(MemoryStrategy::BoundedAttention),
            &MemoryStrategySupport::Missing
        );
        assert_eq!(
            support(MemoryStrategy::BoundedTransformerResidency),
            &MemoryStrategySupport::Missing
        );
        assert_eq!(
            c.capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .parameters,
            bounded_decode_parameters()
        );
        assert!(c.lifecycle.decode_tiling);
        assert!(!c.lifecycle.transformer_window_materialization);
        assert_eq!(c.provider_id, IMAGE_MODEL_ID);
    }

    /// A production contract claims no calibration: nothing has been measured.
    #[test]
    fn production_contract_is_uncalibrated() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-minimax-h3-image".into()));
        assert!(contract_for(&spec).unwrap().calibration.is_none());
        assert!(weights_free().calibration.is_some());
    }

    /// Write a header-plus-zeros safetensors file of `(name, dtype, shape)` tensors.
    fn write_safetensors(path: &std::path::Path, tensors: &[(&str, &str, &[usize])]) {
        let width = |dtype: &str| if dtype == "F32" { 4 } else { 2 };
        let mut header = serde_json::Map::new();
        let mut offset = 0_usize;
        for (name, dtype, shape) in tensors {
            let len = shape.iter().product::<usize>() * width(dtype);
            header.insert(
                (*name).into(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len]}),
            );
            offset += len;
        }
        let json = serde_json::Value::Object(header).to_string().into_bytes();
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend(json);
        bytes.resize(bytes.len() + offset, 0);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    /// A snapshot whose DiT mixes a cast bf16 block tensor (64 elements) with a stored-f32 PDD
    /// head (16) and a stored-bf16 `context_embedder` (8), and whose VAE carries a decode tensor
    /// (32, bf16), a stored-f32 decode tensor (4) and an encoder tensor the loader never reads.
    fn mixed_snapshot() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        write_safetensors(
            &root
                .path()
                .join(IMAGE_DIT_COMPONENT)
                .join("model.safetensors"),
            &[
                ("transformer_blocks.0.attn.to_q.weight", "BF16", &[8, 8]),
                ("pdd_proj_out.0.weight", "F32", &[4, 4]),
                ("context_embedder.weight", "BF16", &[2, 4]),
            ],
        );
        write_safetensors(
            &root
                .path()
                .join(IMAGE_VAE_COMPONENT)
                .join("model.safetensors"),
            &[
                ("decoder.proj_in.weight", "BF16", &[4, 8]),
                ("post_quant_conv.bias", "F32", &[4]),
                ("encoder.proj_in.weight", "BF16", &[1000]),
            ],
        );
        root
    }

    /// Fp32 casts the block stack and the VAE decode half to float32 at load, so they are priced
    /// at 4 bytes per element; the top-level DiT tensors keep their stored widths, and the
    /// encoder half of the VAE is not priced at all. Bf16 prices the same tensors at 2 bytes,
    /// never below a stored-f32 tensor's own width.
    #[test]
    fn asset_facts_price_the_dtype_the_precision_loads() {
        let root = mixed_snapshot();
        let facts = |precision| {
            let mut spec = LoadSpec::new(WeightsSource::Dir(root.path().to_path_buf()));
            spec.precision = precision;
            contract_for(&spec).unwrap().asset_facts
        };
        let fp32 = facts(mlx_gen::Precision::Fp32);
        assert_eq!(fp32.transformer_bytes, 64 * 4 + 16 * 4 + 8 * 2);
        assert_eq!(fp32.decoder_bytes, 32 * 4 + 4 * 4);
        let bf16 = facts(mlx_gen::Precision::Bf16);
        assert_eq!(bf16.transformer_bytes, 64 * 2 + 16 * 4 + 8 * 2);
        assert_eq!(bf16.decoder_bytes, 32 * 2 + 4 * 4);
        assert_eq!(fp32.conditioning_bytes, 0);
        assert_eq!(fp32.base_bytes, fp32.transformer_bytes + fp32.decoder_bytes);
    }

    /// A non-32-aligned request the generator accepts is admitted at its requested size, and its
    /// resources follow the canvas the generator renders on. The request scope still binds the
    /// requested size: the canvas itself is not the admitted request.
    #[test]
    fn a_non_aligned_request_is_admitted_and_sized_at_its_canvas() {
        use mlx_gen::gen_core::h3_image::{generation_canvas, MIN_GENERATION_AREA_DEFAULT};
        use mlx_gen::gen_core::{
            GenerationRequest, MemoryBehaviorRoute, MemoryGeometry, MemoryMode,
        };

        let canvas = admission_canvas(1000, 1001).unwrap();
        assert_eq!(
            canvas,
            generation_canvas(1000, 1001, MIN_GENERATION_AREA_DEFAULT)
        );
        assert_ne!(canvas, (1000, 1001));
        assert!(admission_canvas(0, 1001).is_err());

        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-minimax-h3-image".into()));
        let contract = weights_free_contract(&spec).unwrap();
        let tier = mlx_gen::request_scope::loaded_tier(&spec);
        let mut context = mlx_gen::gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::StagedResidency,
            tier,
            MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: true,
                overlay: None,
            },
        )
        .unwrap();
        context.geometry = MemoryGeometry {
            width: 1000,
            height: 1001,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        assert_eq!(
            safety_check_at_tier(&contract, tier, &context),
            MemorySafetyDecision::Accept
        );
        let request = |width, height| GenerationRequest {
            prompt: "a lighthouse at dusk".into(),
            width,
            height,
            ..Default::default()
        };
        let mut scope = registered_scope(IMAGE_MODEL_ID, &spec, &contract, &context, tier)
            .unwrap()
            .unwrap();
        scope.configure_request(&mut request(1000, 1001)).unwrap();
        let mut scope = registered_scope(IMAGE_MODEL_ID, &spec, &contract, &context, tier)
            .unwrap()
            .unwrap();
        assert!(scope
            .configure_request(&mut request(canvas.0, canvas.1))
            .is_err());
    }

    #[test]
    fn only_the_shipped_decode_geometry_is_admitted() {
        validate_decode_geometry(256, 64).unwrap();
        assert!(validate_decode_geometry(512, 64).is_err());
        assert!(validate_decode_geometry(256, 0).is_err());
    }
}
