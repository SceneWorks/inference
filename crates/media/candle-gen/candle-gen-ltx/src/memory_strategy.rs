//! Request-scoped memory admission for the released Candle/CUDA LTX q4 I2V and first/last-frame
//! tiers (SC-20772, SC-20773).
//!
//! This is intentionally a narrow contract: it is the split `q4/` tier that this provider loads
//! through `tier::TierPaths`, with either one fitted I2V image reference or two ordered fitted
//! first/last keyframes at strength 1.0. Dense and q8 snapshots remain usable as historical
//! generator routes, but cannot borrow this q4 evidence.

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, GenerationRequest, LoadShape, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemoryRunOutcome,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability, MemoryStrategySupport,
    MemoryWindowMaterialization, Precision, Quant, ResidentRequestMemory, WeightsSource,
};

use crate::config::{DEFAULT_FRAMES, MODEL_ID};
use candle_gen::gen_core::{MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryMode};

pub const CALIBRATION_FINGERPRINT: &str = "sc-20772-ltx-2-3-candle-q4-i2v-v1";
const STATIC_CALIBRATION_FINGERPRINT: &str = "sc-20772-ltx-2-3-candle-q4-i2v-registry-v1";
pub const DECODE_OVERLAP: u32 = 64;
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512, 448, 384, 320, 256, 192];
const SHIPPED_GEOMETRIES: &[(u32, u32)] =
    &[(768, 512), (512, 768), (640, 640), (1280, 704), (704, 1280)];
const SHIPPED_FRAMES: &[u32] = &[
    97, 121, 145, 153, 177, 193, 201, 241, 249, 289, 297, 361, 377, 449,
];
const STRENGTH_BITS: u32 = 1.0_f32.to_bits();

fn frame_count_matches_fps(fps: u32, frames: u32) -> bool {
    match fps {
        24 => [97, 145, 193, 241, 289, 361].contains(&frames),
        25 => [97, 153, 201, 249, 297, 377].contains(&frames),
        30 => [121, 177, 241, 297, 361, 449].contains(&frames),
        _ => false,
    }
}

fn reference_axis(width: u32, height: u32) -> String {
    format!("reference:image:{width}x{height}:strength:{STRENGTH_BITS:08x}")
}

fn first_last_axes(width: u32, height: u32) -> [String; 2] {
    [
        format!("keyframe:first:image:{width}x{height}:frame:0:strength:{STRENGTH_BITS:08x}"),
        format!("keyframe:last:image:{width}x{height}:frame:-1:strength:{STRENGTH_BITS:08x}"),
    ]
}
fn extend_clip_axis(frames: u32, width: u32, height: u32) -> String {
    format!(
        "clip:append:frames:{frames}:image:{width}x{height}:frame:0:strength:{STRENGTH_BITS:08x}"
    )
}
fn bridge_clip_axes(frames: u32, width: u32, height: u32) -> [String; 2] {
    [extend_clip_axis(frames, width, height), format!("clip:append:frames:{frames}:image:{width}x{height}:frame:-1:strength:{STRENGTH_BITS:08x}")]
}

fn tier_paths(spec: &LoadSpec) -> gen_core::Result<crate::tier::TierPaths> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: calibrated q4 memory admission requires the split q4 directory"
        )));
    };
    if spec.load_shape != LoadShape::EagerMaterialization || spec.precision != Precision::Bf16 {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: calibrated q4 memory admission requires eager bf16 component loading"
        )));
    }
    if spec.quantize != Some(Quant::Q4)
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || !spec.adapters.is_empty()
        || !spec.components.is_empty()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: calibrated q4 I2V/first-last admission requires the plain split q4 tier without load overlays"
        )));
    }
    let gemma = spec.text_encoder.as_ref().map(|source| match source {
        WeightsSource::Dir(path) | WeightsSource::File(path) => path.as_path(),
    });
    let paths = crate::tier::TierPaths::detect(root, gemma).ok_or_else(|| {
        gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: q4 admission requires a split tier with transformer.safetensors and quantize_config.json"
        ))
    })?;
    let config = paths
        .packed_config()
        .map_err(|error| gen_core::Error::Unsupported(error.to_string()))?;
    if config.bits != 4 || config.group_size as usize != crate::quant::GROUP_SIZE {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: q4 admission requires the released 4-bit group-{} tier",
            crate::quant::GROUP_SIZE
        )));
    }
    Ok(paths)
}

fn exact_load_receipt(spec: &LoadSpec) -> gen_core::Result<MemoryAssetFacts> {
    let paths = tier_paths(spec)?;
    // These are the actual mapped component files. Charging stored q4 codes plus their affine
    // grids is deliberately conservative against Candle's GGML repack, never an under-price; the
    // source maps live until the component constructors have completed.
    let bytes = |path: std::path::PathBuf| gen_core::safetensors_path_bytes(path);
    let conditioning =
        bytes(paths.gemma_dir.clone()) + bytes(paths.tier_dir.join("vae_encoder.safetensors"));
    let transformer = bytes(paths.tier_dir.join("transformer.safetensors"))
        + bytes(paths.tier_dir.join("connector.safetensors"))
        + bytes(
            crate::canonical_upsampler_file(&paths.tier_dir)
                .map_err(|error| gen_core::Error::Unsupported(error.to_string()))?,
        );
    let decoder = bytes(paths.tier_dir.join("vae_decoder.safetensors"));
    let base = conditioning
        .checked_add(transformer)
        .and_then(|v| v.checked_add(decoder))
        .ok_or_else(|| {
            gen_core::Error::Msg(format!("{MODEL_ID}: q4 component byte total overflows u64"))
        })?;
    if base == 0 {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: q4 component receipt is empty"
        )));
    }
    Ok(MemoryAssetFacts {
        base_bytes: base,
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: decoder,
        overlay_bytes: 0,
    })
}

fn strategies() -> Vec<MemoryStrategyCapability> {
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident | MemoryStrategy::BoundedDecode => {
                    MemoryStrategySupport::Implemented
                }
                MemoryStrategy::StagedResidency
                | MemoryStrategy::BoundedAttention
                | MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
            },
            parameters: if strategy == MemoryStrategy::BoundedDecode {
                MemoryParameterRanges {
                    decode_tile_edges: DECODE_TILE_EDGES.to_vec(),
                    decode_overlaps: vec![DECODE_OVERLAP],
                    ..Default::default()
                }
            } else {
                MemoryParameterRanges::default()
            },
        })
        .collect()
}

fn build_contract(
    spec: &LoadSpec,
    facts: MemoryAssetFacts,
    fingerprint: &str,
) -> gen_core::Result<MemoryProviderContract> {
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    Ok(MemoryProviderContract {
        provider_id: MODEL_ID.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: false,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies: strategies(),
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: false,
            transformer_window_materialization: false,
        },
        formula: MemoryFormulaKind::PhaseEnvelope {
            phases,
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::FrameCount,
                MemoryFormulaVariable::BatchCount,
                MemoryFormulaVariable::ConditioningTokenCount,
                MemoryFormulaVariable::DecodeTileArea,
            ],
        },
        calibration: Some(MemoryCalibrationIdentity::new(fingerprint, spec.load_shape)),
        asset_facts: facts,
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    })
}

pub fn memory_strategy_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    build_contract(spec, exact_load_receipt(spec)?, CALIBRATION_FINGERPRINT)
}

#[cfg(feature = "cuda")]
pub(crate) fn contract_for_loaded(
    spec: &LoadSpec,
) -> gen_core::Result<Option<(MemoryProviderContract, MemoryNumericTier)>> {
    Ok(memory_strategy_contract(spec).ok().map(|contract| {
        (
            contract,
            MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: &[],
            },
        )
    }))
}

fn weights_free_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    // The fixture shares every identity gate except physical-file inspection; catalog conformance
    // cannot depend on weights being installed on the compiler host.
    if spec.quantize != Some(Quant::Q4)
        || spec.load_shape != LoadShape::EagerMaterialization
        || spec.precision != Precision::Bf16
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: fixture is q4/eager/bf16 only"
        )));
    }
    build_contract(
        spec,
        MemoryAssetFacts::default(),
        STATIC_CALIBRATION_FINGERPRINT,
    )
}

fn validate_decode(edge: Option<u32>, overlap: Option<u32>) -> gen_core::Result<()> {
    let edge = edge.ok_or_else(|| {
        gen_core::Error::Unsupported(format!("{MODEL_ID}: bounded decode requires a tile edge"))
    })?;
    let overlap = overlap.ok_or_else(|| {
        gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: bounded decode requires a tile overlap"
        ))
    })?;
    if !DECODE_TILE_EDGES.contains(&edge) || overlap != DECODE_OVERLAP {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: decode parameters are outside the released q4 LTX domain"
        )));
    }
    Ok(())
}

fn route_gate(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    let i2v = context.mode.as_key() == "image_to_video"
        && context.geometry.reference_count == 1
        && context.has_reference;
    let first_last = context.mode.as_key() == "first_last_frame"
        && context.geometry.reference_count == 2
        && context.has_reference;
    let extend = context.mode.as_key() == "extend_clip"
        && context.geometry.reference_count == 0
        && !context.has_reference;
    let bridge = context.mode.as_key() == "video_bridge"
        && context.geometry.reference_count == 0
        && !context.has_reference;
    if (!i2v && !first_last && !extend && !bridge) || context.use_pid || context.has_phases {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: q4 memory admission requires image_to_video/one Reference or first_last_frame/two ordered Keyframes without PiD/phases"
        )));
    }
    let geometry = context.geometry;
    if geometry.batch != 1
        || !SHIPPED_GEOMETRIES.contains(&(geometry.width, geometry.height))
        || !SHIPPED_FRAMES.contains(&geometry.frames)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: unsupported q4 I2V/first-last geometry"
        )));
    }
    let expected_overlay = if i2v {
        reference_axis(geometry.width, geometry.height)
    } else if first_last {
        first_last_axes(geometry.width, geometry.height).join("+")
    } else if extend {
        extend_clip_axis(geometry.frames, geometry.width, geometry.height)
    } else {
        bridge_clip_axes(geometry.frames, geometry.width, geometry.height).join("+")
    };
    if context.overlay.as_deref() != Some(expected_overlay.as_str()) {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: q4 admission requires the exact fitted ordered conditioning receipt"
        )));
    }
    if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
        validate_decode(
            context.selection.parameters.decode_tile_edge,
            context.selection.parameters.decode_overlap,
        )?;
    }
    Ok(())
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        }),
        Some(&|| route_gate(contract, context)),
    )
}

fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    if memory_strategy_contract(spec).is_err()
        && contract
            .calibration
            .as_ref()
            .is_some_and(|value| value.fingerprint != STATIC_CALIBRATION_FINGERPRINT)
    {
        return MemorySafetyDecision::Reject {
            reason: format!("{MODEL_ID}: loaded q4 receipt no longer matches the memory contract"),
        };
    }
    safety_check(contract, context)
}

struct LtxMemoryScope {
    inner: candle_gen::request_scope::CandleRequestScopeCore,
    admitted: gen_core::MemoryGeometry,
    admitted_mode: String,
}
impl MemoryRequestScope for LtxMemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        let fitted = |image: &gen_core::Image, strength: f32| {
            image.width == request.width
                && image.height == request.height
                && strength.to_bits() == STRENGTH_BITS
        };
        let exact_conditioning = match self.admitted_mode.as_str() {
            "image_to_video" => match request.conditioning.as_slice() {
                [gen_core::Conditioning::Reference { image, strength }] => {
                    fitted(image, strength.unwrap_or(1.0))
                }
                _ => false,
            },
            "first_last_frame" => match request.conditioning.as_slice() {
                [gen_core::Conditioning::Keyframe {
                    image: first,
                    frame_idx: 0,
                    strength: first_strength,
                }, gen_core::Conditioning::Keyframe {
                    image: last,
                    frame_idx: -1,
                    strength: last_strength,
                }] => fitted(first, *first_strength) && fitted(last, *last_strength),
                _ => false,
            },
            "extend_clip" => match request.conditioning.as_slice() {
                [gen_core::Conditioning::VideoClip {
                    frames,
                    frame_idx: 0,
                    strength,
                }] if frames.len() == request.frames.unwrap_or(0) as usize && *strength == 1.0 => {
                    frames
                        .iter()
                        .all(|image| image.width == request.width && image.height == request.height)
                }
                _ => false,
            },
            "video_bridge" => match request.conditioning.as_slice() {
                [gen_core::Conditioning::VideoClip {
                    frames: left,
                    frame_idx: 0,
                    strength: left_strength,
                }, gen_core::Conditioning::VideoClip {
                    frames: right,
                    frame_idx: -1,
                    strength: right_strength,
                }] if left.len() == request.frames.unwrap_or(0) as usize
                    && right.len() == left.len()
                    && *left_strength == 1.0
                    && *right_strength == 1.0 =>
                {
                    left.iter()
                        .chain(right)
                        .all(|image| image.width == request.width && image.height == request.height)
                }
                _ => false,
            },
            _ => false,
        };
        if !exact_conditioning
            || request.fps.is_none_or(|fps| {
                !frame_count_matches_fps(fps, request.frames.unwrap_or(DEFAULT_FRAMES))
            })
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: request crossed the admitted conditioning/FPS/strength identity"
            )));
        }
        self.inner.configure_request(request)
    }
    fn enter_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.inner.enter_phase(phase)
    }
    fn leave_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.inner.leave_phase(phase)
    }
    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        geometry: gen_core::MemoryGeometry,
    ) -> gen_core::Result<()> {
        if geometry != self.admitted {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: decode geometry crossed admission"
            )));
        }
        self.inner.configure_decode(tile_edge, overlap, geometry)
    }
    fn configure_attention(&mut self, chunk_size: u32) -> gen_core::Result<()> {
        self.inner.configure_attention(chunk_size)
    }
    fn materialize_transformer_window(&mut self, first: u32, count: u32) -> gen_core::Result<()> {
        self.inner.materialize_transformer_window(first, count)
    }
    fn finish(&mut self, outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        self.inner.finish(outcome)
    }
}

fn begin(
    contract: &MemoryProviderContract,
    device: Device,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(contract, context) {
        return Err(gen_core::Error::Unsupported(reason));
    }
    let mut config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        MODEL_ID,
        device,
        context.geometry,
        contract.generation_memory(&context.selection),
        false,
        48,
        |_pid, edge, overlap| validate_decode(Some(edge), Some(overlap)),
    )?;
    config.default_frames = DEFAULT_FRAMES;
    Ok(Some(Box::new(LtxMemoryScope {
        inner: candle_gen::request_scope::CandleRequestScopeCore::new(config),
        admitted: context.geometry,
        admitted_mode: context.mode.as_key().to_owned(),
    })))
}

fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if memory_strategy_contract(spec).is_err()
        && contract
            .calibration
            .as_ref()
            .is_some_and(|value| value.fingerprint != STATIC_CALIBRATION_FINGERPRINT)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: loaded q4 receipt no longer matches the memory contract"
        )));
    }
    begin(contract, Device::Cpu, context)
}

fn registered_valid_fixtures(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || !matches!(
            contract.capability(strategy).map(|cap| &cap.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }
    let mut context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("image_to_video".into()),
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: Some(reference_axis(768, 512)),
        },
    )?;
    context.geometry.width = 768;
    context.geometry.height = 512;
    context.geometry.frames = 153;
    let mut fixture = MemoryBehaviorFixture::new(context);
    fixture.request.width = 768;
    fixture.request.height = 512;
    fixture.request.frames = Some(153);
    fixture.request.fps = Some(25);
    fixture.request.conditioning = vec![gen_core::Conditioning::Reference {
        image: gen_core::Image {
            width: 768,
            height: 512,
            pixels: vec![0; 768 * 512 * 3],
        },
        strength: Some(1.0),
    }];
    let mut first_last_context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("first_last_frame".into()),
            reference_count: 2,
            use_pid: false,
            has_phases: false,
            overlay: Some(first_last_axes(768, 512).join("+")),
        },
    )?;
    first_last_context.geometry.width = 768;
    first_last_context.geometry.height = 512;
    first_last_context.geometry.frames = 153;
    let mut first_last = MemoryBehaviorFixture::new(first_last_context);
    first_last.request.width = 768;
    first_last.request.height = 512;
    first_last.request.frames = Some(153);
    first_last.request.fps = Some(25);
    first_last.request.conditioning = vec![
        gen_core::Conditioning::Keyframe {
            image: gen_core::Image {
                width: 768,
                height: 512,
                pixels: vec![0; 768 * 512 * 3],
            },
            frame_idx: 0,
            strength: 1.0,
        },
        gen_core::Conditioning::Keyframe {
            image: gen_core::Image {
                width: 768,
                height: 512,
                pixels: vec![0; 768 * 512 * 3],
            },
            frame_idx: -1,
            strength: 1.0,
        },
    ];
    let _ = spec;
    let mut extend_context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("extend_clip".into()),
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: Some(extend_clip_axis(153, 768, 512)),
        },
    )?;
    extend_context.geometry.width = 768;
    extend_context.geometry.height = 512;
    extend_context.geometry.frames = 153;
    let mut extend = MemoryBehaviorFixture::new(extend_context);
    extend.request.width = 768;
    extend.request.height = 512;
    extend.request.frames = Some(153);
    extend.request.fps = Some(25);
    extend.request.conditioning = vec![gen_core::Conditioning::VideoClip {
        frames: vec![
            gen_core::Image {
                width: 768,
                height: 512,
                pixels: vec![0; 768 * 512 * 3]
            };
            153
        ],
        frame_idx: 0,
        strength: 1.0,
    }];
    let mut bridge_context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("video_bridge".into()),
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: Some(bridge_clip_axes(153, 768, 512).join("+")),
        },
    )?;
    bridge_context.geometry.width = 768;
    bridge_context.geometry.height = 512;
    bridge_context.geometry.frames = 153;
    let mut bridge = MemoryBehaviorFixture::new(bridge_context);
    bridge.request.width = 768;
    bridge.request.height = 512;
    bridge.request.frames = Some(153);
    bridge.request.fps = Some(25);
    let clip = vec![
        gen_core::Image {
            width: 768,
            height: 512,
            pixels: vec![0; 768 * 512 * 3]
        };
        153
    ];
    bridge.request.conditioning = vec![
        gen_core::Conditioning::VideoClip {
            frames: clip.clone(),
            frame_idx: 0,
            strength: 1.0,
        },
        gen_core::Conditioning::VideoClip {
            frames: clip,
            frame_idx: -1,
            strength: 1.0,
        },
    ];
    Ok(vec![fixture, first_last, extend, bridge])
}

fn surfaces() -> Vec<gen_core::MemoryContractSurfaceSpec> {
    gen_core::candle_memory_contract_surface_specs()
        .into_iter()
        .filter(|surface| {
            surface.selector.load_shape == LoadShape::EagerMaterialization
                && surface.selector.tier == gen_core::MemoryContractSurfaceTier::Q4
        })
        .collect()
}

// Keep the closure spelling aligned with the MLX registration. The conversion is a no-op on this
// backend today but makes the cross-backend registration gate compare the same adapter boundary.
#[allow(clippy::useless_conversion)]
pub(crate) const MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: crate::MODEL_ID,
    contract: |spec| memory_strategy_contract(spec).map_err(Into::into),
    safety_check: registered_safety_check,
};
pub(crate) const MEMORY_FIXTURE: gen_core::MemoryContractFixtureRegistration =
    gen_core::MemoryContractFixtureRegistration {
        provider_id: MODEL_ID,
        contract: weights_free_contract,
        surface_specs: surfaces,
    };
pub(crate) const MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: crate::MODEL_ID,
        valid_fixtures: registered_valid_fixtures,
        begin_request: registered_begin_request,
    };

pub(crate) fn begin_request(
    contract: &MemoryProviderContract,
    device: Device,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin(contract, device, context)
}

pub(crate) fn selected_decode_cap(
    request: &GenerationRequest,
) -> candle_gen::Result<Option<(u32, u32)>> {
    let Some(memory) = request.memory else {
        return Ok(None);
    };
    if !memory.tile_vae_decode {
        if memory.decode_tile_edge.is_some() || memory.decode_overlap.is_some() {
            return Err(candle_gen::CandleError::Msg(format!(
                "{MODEL_ID}: decode parameters supplied without bounded decode"
            )));
        }
        return Ok(None);
    }
    validate_decode(memory.decode_tile_edge, memory.decode_overlap)
        .map_err(|error| candle_gen::CandleError::Msg(error.to_string()))?;
    Ok(Some((
        memory.decode_tile_edge.expect("validated"),
        memory.decode_overlap.expect("validated"),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{MemoryBehaviorRoute, MemoryMode, MemoryStrategyParameters};

    fn fixture_spec() -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/weights/ltx/q4".into()));
        spec.quantize = Some(Quant::Q4);
        spec
    }

    #[test]
    fn q4_i2v_contract_publishes_only_executable_rungs() {
        let contract = weights_free_contract(&fixture_spec()).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::Resident)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        for rung in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                contract.capability(rung).unwrap().support,
                MemoryStrategySupport::Missing
            );
        }
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .parameters
                .decode_tile_edges,
            DECODE_TILE_EDGES
        );
    }

    #[test]
    fn i2v_identity_and_decode_mutations_fail_closed() {
        let contract = weights_free_contract(&fixture_spec()).unwrap();
        let mut context = gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::BoundedDecode,
            MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: &[],
            },
            MemoryBehaviorRoute {
                mode: MemoryMode::Other("image_to_video".into()),
                reference_count: 1,
                use_pid: false,
                has_phases: false,
                overlay: Some(reference_axis(768, 512)),
            },
        )
        .unwrap();
        context.geometry.width = 768;
        context.geometry.height = 512;
        context.geometry.frames = 153;
        assert!(matches!(
            safety_check(&contract, &context),
            MemorySafetyDecision::Accept
        ));
        context.overlay = Some(reference_axis(512, 768));
        assert!(matches!(
            safety_check(&contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
        context.overlay = Some(reference_axis(768, 512));
        context.selection.parameters = MemoryStrategyParameters {
            decode_tile_edge: Some(511),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        };
        assert!(matches!(
            safety_check(&contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn first_last_fixture_binds_ordered_two_keyframe_identity() {
        let spec = fixture_spec();
        let contract = weights_free_contract(&spec).unwrap();
        let fixture = registered_valid_fixtures(&spec, &contract, MemoryStrategy::BoundedDecode)
            .unwrap()
            .into_iter()
            .find(|fixture| fixture.context.mode.as_key() == "first_last_frame")
            .unwrap();
        assert!(matches!(
            safety_check(&contract, &fixture.context),
            MemorySafetyDecision::Accept
        ));
        let mut scope = begin(&contract, Device::Cpu, &fixture.context)
            .unwrap()
            .unwrap();
        let mut request = fixture.request.clone();
        scope.configure_request(&mut request).unwrap();

        let mut crossed = fixture.request;
        crossed.conditioning.swap(0, 1);
        let error = scope
            .configure_request(&mut crossed)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("conditioning/FPS/strength identity"),
            "{error}"
        );
        assert_eq!(crossed.memory, None);
    }
}
