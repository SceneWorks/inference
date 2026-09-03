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

/// The ordered replace-person receipt: the masked control clip, then the character contact sheet.
///
/// Two axes rather than one because the carriers are independently variable — the clip's mask mode
/// and blend weight move without changing the reference count, and a fifth reference changes the
/// composite grid without touching the clip. Folding them into one token would let a crossed pair
/// reuse the other's cell. The reference axis names the **composite** the provider actually encodes
/// (see `conditioning::compose_ordered_character_references`), so 1–4 sources at any input size
/// collapse to one target-sized image latent — the count is what changes the grid, not the sizes.
fn replace_person_axes(
    frames: u32,
    width: u32,
    height: u32,
    references: usize,
    masking_strength: f32,
    mode: gen_core::ReplacementMode,
) -> [String; 2] {
    [
        format!(
            "clip:replace:frames:{frames}:image:{width}x{height}:frame:0:mode:{}:mask:{:08x}",
            mode as u8,
            masking_strength.to_bits()
        ),
        format!("reference:sheet:{references}:image:{width}x{height}"),
    ]
}

/// Pull the exact LTX replace-person carrier out of a request: one masked `ControlClip` plus one
/// ordered 1–4 image `MultiReference`, in either order (the generator accepts both).
///
/// Returns the receipt the axes above describe, or `None` when the pair is not exactly that shape.
fn replace_person_receipt(request: &GenerationRequest) -> Option<String> {
    let (clip, images) = match request.conditioning.as_slice() {
        [gen_core::Conditioning::ControlClip { .. }, gen_core::Conditioning::MultiReference { images }]
        | [gen_core::Conditioning::MultiReference { images }, gen_core::Conditioning::ControlClip { .. }] => {
            (request.control_clip()?, images)
        }
        _ => return None,
    };
    let frames = request.frames.unwrap_or(DEFAULT_FRAMES);
    let fitted =
        |image: &gen_core::Image| image.width == request.width && image.height == request.height;
    if clip.frames.len() != frames as usize
        || clip.mask.len() != clip.frames.len()
        || clip.frames.is_empty()
        || clip.start_frame != 0
        || !clip.masking_strength.is_finite()
        || !(0.0..=1.0).contains(&clip.masking_strength)
        || !clip.frames.iter().all(fitted)
        || !clip.mask.iter().all(fitted)
        || !(1..=4).contains(&images.len())
    {
        return None;
    }
    // The composite is the actual VAE input; refusing here keeps a degenerate reference (zero-sized
    // or short buffer) from being admitted and then failing mid-render.
    crate::conditioning::compose_ordered_character_references(
        images,
        request.width,
        request.height,
    )
    .ok()?;
    Some(
        replace_person_axes(
            frames,
            request.width,
            request.height,
            images.len(),
            clip.masking_strength,
            clip.mode,
        )
        .join("+"),
    )
}

/// Does `overlay` spell exactly the two replace-person axes this `geometry` can carry?
///
/// The mask blend weight and the [`gen_core::ReplacementMode`] have no geometry counterpart, so
/// they are checked structurally (a well-formed mode ordinal and a finite `0..=1` blend weight)
/// rather than reconstructed. Everything the geometry *does* fix — frame count, output size, clip
/// start and reference cardinality — is compared exactly.
fn replace_person_overlay_shape_ok(
    overlay: Option<&str>,
    geometry: gen_core::MemoryGeometry,
) -> bool {
    let axes = overlay
        .into_iter()
        .flat_map(|value| value.split('+'))
        .collect::<Vec<_>>();
    let [clip, sheet] = axes.as_slice() else {
        return false;
    };
    if *sheet
        != format!(
            "reference:sheet:{}:image:{}x{}",
            geometry.reference_count, geometry.width, geometry.height
        )
    {
        return false;
    }
    let Some(tail) = clip.strip_prefix(&format!(
        "clip:replace:frames:{}:image:{}x{}:frame:0:mode:",
        geometry.frames, geometry.width, geometry.height
    )) else {
        return false;
    };
    let Some((mode, mask)) = tail.split_once(":mask:") else {
        return false;
    };
    if !matches!(mode, "0" | "1" | "2")
        || mask.len() != 8
        || !mask
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return false;
    }
    u32::from_str_radix(mask, 16)
        .ok()
        .map(f32::from_bits)
        .is_some_and(|weight| weight.is_finite() && (0.0..=1.0).contains(&weight))
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

/// Snapshot-gated architecture facts for the LTX-2.3 route (epic SC-22657, E2).
///
/// The LTX-2.3 loader does not parse a transformer `config.json`: `lib.rs` builds
/// [`crate::config::AvConfig::ltx_2_3`] directly, so those are the axes read here — reading a
/// config the loader ignores would publish a fact about a model that is not the one loaded. (The
/// LTX-2.5 route in [`crate::memory_strategy_2_5`] *does* read its bundle header, and reads it
/// there.) The latent geometry comes from the crate's own compression constants, the same ones
/// `pipeline::latent_dims` uses to size every latent this provider allocates.
///
/// `patch_size` is deliberately `None`: LTX patchifies inside the causal video autoencoder, not in
/// the DiT. The DiT's `patchify_proj` is a per-token `Linear(128 -> 4096)` applied to the already
/// flattened `[B, S, 128]` latent (`pipeline::flatten_latent`, one token per latent voxel), so the
/// denoiser applies no spatial patch factor at all. The VAE's own `patch_size: 4`
/// ([`crate::config::VideoVaeDeclaration`]) is a pixel-shuffle stage already folded into
/// `SPATIAL_SCALE = 32`; publishing it as the DiT's patch would double-count it.
///
/// A weights-free contract (the registry's sentinel surface, which is not on disk) publishes
/// `MemoryArchitectureFacts::default()`.
fn architecture_facts(spec: &LoadSpec) -> gen_core::MemoryArchitectureFacts {
    use candle_gen::architecture_facts as af;

    if af::snapshot_root(spec).is_none() {
        return gen_core::MemoryArchitectureFacts::default();
    }
    let video = crate::config::AvConfig::ltx_2_3().video;
    gen_core::MemoryArchitectureFacts {
        attention_heads: af::declared(video.num_heads),
        // Declared by the preset (`inner_dim == num_heads * head_dim` = 4096), not re-derived.
        head_dim: af::declared(video.head_dim),
        transformer_blocks: af::declared(video.num_layers),
        // Structurally absent: LTX patchifies inside the causal video autoencoder, not in the DiT.
        patch_size: None,
        latent_channels: af::declared(crate::config::LATENT_CHANNELS),
        vae_spatial_scale: af::declared(crate::config::SPATIAL_SCALE),
        vae_temporal_scale: af::declared(crate::config::TEMPORAL_SCALE),
        // `lib.rs: DIT_DTYPE = DType::BF16`.
        activation_dtype_width: af::dtype_width(crate::DIT_DTYPE),
    }
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
        architecture_facts: architecture_facts(spec),
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
    let spec = weights_free_behavior_spec(spec);
    build_contract(
        &spec,
        MemoryAssetFacts::default(),
        STATIC_CALIBRATION_FINGERPRINT,
    )
}

fn weights_free_behavior_spec(spec: &LoadSpec) -> LoadSpec {
    let mut exact = spec.clone();
    exact.weights = WeightsSource::Dir("registry/ltx-2-3-distilled/q4".into());
    exact.resolved_route = Some(MODEL_ID.to_owned());
    exact.precision = Precision::Bf16;
    exact.quantize = Some(Quant::Q4);
    exact.load_shape = LoadShape::EagerMaterialization;
    exact
}

fn validate_weights_free_behavior_spec(spec: &LoadSpec) -> gen_core::Result<()> {
    if spec.resolved_route.as_deref() != Some(MODEL_ID)
        || spec.quantize != Some(Quant::Q4)
        || spec.load_shape != LoadShape::EagerMaterialization
        || spec.precision != Precision::Bf16
        || !matches!(spec.weights, WeightsSource::Dir(_))
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || !spec.adapters.is_empty()
        || !spec.components.is_empty()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: registry fixture requires the exact plain q4/eager/bf16 route"
        )));
    }
    Ok(())
}

fn registered_contract_identity(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
) -> gen_core::Result<()> {
    let expected = if contract.asset_facts == MemoryAssetFacts::default()
        && contract
            .calibration
            .as_ref()
            .is_some_and(|identity| identity.fingerprint == STATIC_CALIBRATION_FINGERPRINT)
    {
        validate_weights_free_behavior_spec(spec)?;
        build_contract(
            spec,
            MemoryAssetFacts::default(),
            STATIC_CALIBRATION_FINGERPRINT,
        )?
    } else {
        memory_strategy_contract(spec)?
    };
    if expected != *contract {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: caller contract differs from the exact loaded or registry witness"
        )));
    }
    Ok(())
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
    let replace_person = context.mode.as_key() == "replace_person"
        && (1..=4).contains(&context.geometry.reference_count)
        && context.has_reference;
    if (!i2v && !first_last && !extend && !bridge && !replace_person)
        || context.use_pid
        || context.has_phases
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: q4 memory admission requires image_to_video/one Reference, \
             first_last_frame/two ordered Keyframes, extend_clip or video_bridge IC-LoRA clips, or \
             replace_person/masked ControlClip + 1-4 ordered MultiReference, without PiD/phases"
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
    if replace_person {
        // The clip's blend weight and mask mode are request axes with no geometry counterpart, so
        // admission pins the axis *shape* here and the request scope compares the whole receipt
        // string against this overlay byte-for-byte.
        if !replace_person_overlay_shape_ok(context.overlay.as_deref(), geometry) {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: replace_person admission requires the exact masked-clip + ordered \
                 character-sheet receipt"
            )));
        }
    } else {
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
    if let Err(error) = registered_contract_identity(spec, contract) {
        return MemorySafetyDecision::Reject {
            reason: error.to_string(),
        };
    }
    safety_check(contract, context)
}

struct LtxMemoryScope {
    inner: candle_gen::request_scope::CandleRequestScopeCore,
    admitted: gen_core::MemoryGeometry,
    admitted_mode: String,
    admitted_overlay: Option<String>,
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
            // The receipt is rebuilt from the executing request and compared to the admitted
            // overlay, so the mask blend weight, mask mode and reference cardinality cannot change
            // after admission even though the geometry cannot see them.
            "replace_person" => {
                replace_person_receipt(request).as_deref() == self.admitted_overlay.as_deref()
            }
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
        admitted_overlay: context.overlay.clone(),
    })))
}

fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    registered_contract_identity(spec, contract)?;
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
    let exact_spec = weights_free_behavior_spec(spec);
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
    let replace_references = 2_usize;
    let replace_mask = 1.0_f32;
    let replace_mode = gen_core::ReplacementMode::FullPersonKeepOutfit;
    let mut replace_context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("replace_person".into()),
            reference_count: replace_references as u32,
            use_pid: false,
            has_phases: false,
            overlay: Some(
                replace_person_axes(
                    153,
                    768,
                    512,
                    replace_references,
                    replace_mask,
                    replace_mode,
                )
                .join("+"),
            ),
        },
    )?;
    replace_context.geometry.width = 768;
    replace_context.geometry.height = 512;
    replace_context.geometry.frames = 153;
    let mut replace_person = MemoryBehaviorFixture::new(replace_context);
    replace_person.request.width = 768;
    replace_person.request.height = 512;
    replace_person.request.frames = Some(153);
    replace_person.request.fps = Some(25);
    let plate = gen_core::Image {
        width: 768,
        height: 512,
        pixels: vec![0; 768 * 512 * 3],
    };
    replace_person.request.conditioning = vec![
        gen_core::Conditioning::ControlClip {
            frames: vec![plate.clone(); 153],
            mask: vec![plate.clone(); 153],
            masking_strength: replace_mask,
            start_frame: 0,
            mode: replace_mode,
        },
        gen_core::Conditioning::MultiReference {
            images: vec![plate; replace_references],
        },
    ];
    Ok(vec![fixture, first_last, extend, bridge, replace_person]
        .into_iter()
        .map(|fixture| fixture.with_load_spec(exact_spec.clone()))
        .collect())
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
    fn calibrated_scope_rejects_zero_fps_before_installing_controls() {
        let generic = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let contract = weights_free_contract(&generic).unwrap();
        let fixture = registered_valid_fixtures(&generic, &contract, MemoryStrategy::BoundedDecode)
            .unwrap()
            .into_iter()
            .next()
            .expect("the calibrated LTX fixture");
        let exact = fixture
            .load_spec
            .as_ref()
            .expect("the provider-owned calibrated load spec");
        let mut scope = registered_begin_request(exact, &contract, &fixture.context)
            .unwrap()
            .expect("the calibrated LTX request scope");
        let mut zero_fps = GenerationRequest {
            fps: Some(0),
            ..fixture.request
        };
        let error = scope
            .configure_request(&mut zero_fps)
            .expect_err("calibrated Candle LTX admission must reject zero fps");
        assert!(
            matches!(error, gen_core::Error::Unsupported(_)),
            "got: {error:?}"
        );
        assert!(error
            .to_string()
            .contains("conditioning/FPS/strength identity"));
        assert_eq!(zero_fps.memory, None);
    }

    /// AC (epic SC-22657, E2): the LTX-2.3 contract publishes the architecture axes of the config
    /// the loader actually instantiates, and the weights-free surface publishes none of them.
    #[test]
    fn architecture_facts_match_the_loader_config_and_pass_conformance() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(tmp.path().to_path_buf()));
        let contract =
            build_contract(&spec, MemoryAssetFacts::default(), CALIBRATION_FINGERPRINT).unwrap();
        assert_eq!(
            contract.architecture_facts,
            gen_core::MemoryArchitectureFacts {
                // `AvConfig::ltx_2_3().video`: 32 heads x 128 = inner dim 4096, 48 layers.
                attention_heads: Some(32),
                head_dim: Some(128),
                transformer_blocks: Some(48),
                // Structurally absent: LTX patchifies inside the causal video autoencoder, not in
                // the DiT — `patchify_proj` is a per-token Linear(128 -> 4096) over the already
                // flattened `[B, S, 128]` latent, so the denoiser applies no spatial patch factor.
                patch_size: None,
                // `config::LATENT_CHANNELS`.
                latent_channels: Some(128),
                // `config::SPATIAL_SCALE` / `config::TEMPORAL_SCALE`.
                vae_spatial_scale: Some(32),
                vae_temporal_scale: Some(8),
                // `lib.rs: DIT_DTYPE = DType::BF16`.
                activation_dtype_width: Some(2),
            }
        );
        assert!(contract.architecture_facts.has_snapshot_read_axis());
        gen_core_testkit::assert_memory_contract_facts_conform(&contract);

        // The registry fixture's weights path is a sentinel that is not on disk.
        let weights_free =
            weights_free_contract(&LoadSpec::new(WeightsSource::Dir("/nonexistent".into())))
                .unwrap();
        assert!(weights_free.architecture_facts.is_empty());
    }

    #[test]
    fn catalog_fixture_normalizes_and_binds_the_exact_q4_route() {
        let generic = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let contract = weights_free_contract(&generic).unwrap();
        let fixture = registered_valid_fixtures(&generic, &contract, MemoryStrategy::BoundedDecode)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let exact = fixture
            .load_spec
            .as_ref()
            .expect("provider-owned load spec");
        assert_eq!(exact.resolved_route.as_deref(), Some(MODEL_ID));
        assert_eq!(exact.quantize, Some(Quant::Q4));
        assert_eq!(exact.load_shape, LoadShape::EagerMaterialization);
        assert!(matches!(
            registered_safety_check(exact, &contract, &fixture.context),
            MemorySafetyDecision::Accept
        ));
        assert!(matches!(
            registered_safety_check(&generic, &contract, &fixture.context),
            MemorySafetyDecision::Reject { .. }
        ));
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

    #[test]
    fn bridge_fixture_binds_ordered_two_clip_identity() {
        let spec = fixture_spec();
        let contract = weights_free_contract(&spec).unwrap();
        let fixture = registered_valid_fixtures(&spec, &contract, MemoryStrategy::BoundedDecode)
            .unwrap()
            .into_iter()
            .find(|fixture| fixture.context.mode.as_key() == "video_bridge")
            .unwrap();
        assert!(matches!(
            safety_check(&contract, &fixture.context),
            MemorySafetyDecision::Accept
        ));
        let mut scope = begin(&contract, Device::Cpu, &fixture.context)
            .unwrap()
            .unwrap();
        let mut accepted = fixture.request.clone();
        scope.configure_request(&mut accepted).unwrap();

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

    /// sc-20799: `replace_person` is implemented and advertised by this provider's generator, so it
    /// must also be admissible. It previously fell to `configure_request`'s `_ => false`, which
    /// made the mode unreachable behind an admitted memory rung.
    #[test]
    fn replace_person_fixture_binds_the_masked_clip_and_character_sheet_identity() {
        let spec = fixture_spec();
        let contract = weights_free_contract(&spec).unwrap();
        let fixture = registered_valid_fixtures(&spec, &contract, MemoryStrategy::BoundedDecode)
            .unwrap()
            .into_iter()
            .find(|fixture| fixture.context.mode.as_key() == "replace_person")
            .expect("replace_person ships a behavior fixture");
        assert!(matches!(
            safety_check(&contract, &fixture.context),
            MemorySafetyDecision::Accept
        ));
        assert_eq!(fixture.context.geometry.reference_count, 2);
        assert!(fixture.context.has_reference);
        // The shared scope core compares this against the admitted geometry; if the two counters
        // disagreed the mode would be unreachable no matter what the arms below accept.
        assert_eq!(
            fixture.request.memory_reference_count(),
            fixture.context.geometry.reference_count
        );
        let mut scope = begin(&contract, Device::Cpu, &fixture.context)
            .unwrap()
            .unwrap();
        let mut accepted = fixture.request.clone();
        scope.configure_request(&mut accepted).unwrap();
        assert!(accepted.memory.is_some());

        // The carrier order is free (the generator accepts either), so swapping must NOT be the
        // thing that fails — the crossed axes below must be.
        let mut swapped = fixture.request.clone();
        swapped.conditioning.swap(0, 1);
        scope.configure_request(&mut swapped).unwrap();

        for (label, cross) in [
            (
                "reference cardinality",
                Box::new(|request: &mut GenerationRequest| {
                    if let gen_core::Conditioning::MultiReference { images } =
                        &mut request.conditioning[1]
                    {
                        images.push(images[0].clone());
                    }
                }) as Box<dyn Fn(&mut GenerationRequest)>,
            ),
            (
                "mask blend weight",
                Box::new(|request: &mut GenerationRequest| {
                    if let gen_core::Conditioning::ControlClip {
                        masking_strength, ..
                    } = &mut request.conditioning[0]
                    {
                        *masking_strength = 0.5;
                    }
                }),
            ),
            (
                "replacement mode",
                Box::new(|request: &mut GenerationRequest| {
                    if let gen_core::Conditioning::ControlClip { mode, .. } =
                        &mut request.conditioning[0]
                    {
                        *mode = gen_core::ReplacementMode::FaceOnly;
                    }
                }),
            ),
            (
                "clip start frame",
                Box::new(|request: &mut GenerationRequest| {
                    if let gen_core::Conditioning::ControlClip { start_frame, .. } =
                        &mut request.conditioning[0]
                    {
                        *start_frame = 1;
                    }
                }),
            ),
        ] {
            let mut crossed = fixture.request.clone();
            cross(&mut crossed);
            let error = scope
                .configure_request(&mut crossed)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("conditioning/FPS/strength identity"),
                "crossing the {label} must not be admitted: {error}"
            );
            assert_eq!(crossed.memory, None, "{label}");
        }
    }

    #[test]
    fn replace_person_admission_refuses_a_crossed_or_missing_receipt() {
        let spec = fixture_spec();
        let contract = weights_free_contract(&spec).unwrap();
        let fixture = registered_valid_fixtures(&spec, &contract, MemoryStrategy::BoundedDecode)
            .unwrap()
            .into_iter()
            .find(|fixture| fixture.context.mode.as_key() == "replace_person")
            .unwrap();
        let geometry = fixture.context.geometry;
        assert!(replace_person_overlay_shape_ok(
            fixture.context.overlay.as_deref(),
            geometry
        ));
        for crossed in [
            None,
            Some(String::new()),
            // One axis only.
            Some(
                fixture
                    .context
                    .overlay
                    .as_deref()
                    .unwrap()
                    .split('+')
                    .next()
                    .unwrap()
                    .to_owned(),
            ),
            // A sheet that claims a different cardinality than the admitted geometry.
            Some(format!(
                "clip:replace:frames:{}:image:{}x{}:frame:0:mode:1:mask:{:08x}+reference:sheet:3:image:{}x{}",
                geometry.frames,
                geometry.width,
                geometry.height,
                1.0_f32.to_bits(),
                geometry.width,
                geometry.height
            )),
            // An out-of-range mask blend weight.
            Some(format!(
                "clip:replace:frames:{}:image:{}x{}:frame:0:mode:1:mask:{:08x}+reference:sheet:2:image:{}x{}",
                geometry.frames,
                geometry.width,
                geometry.height,
                2.0_f32.to_bits(),
                geometry.width,
                geometry.height
            )),
            // An undeclared replacement mode ordinal.
            Some(format!(
                "clip:replace:frames:{}:image:{}x{}:frame:0:mode:9:mask:{:08x}+reference:sheet:2:image:{}x{}",
                geometry.frames,
                geometry.width,
                geometry.height,
                1.0_f32.to_bits(),
                geometry.width,
                geometry.height
            )),
        ] {
            let mut context = fixture.context.clone();
            context.overlay = crossed.clone();
            assert!(
                !replace_person_overlay_shape_ok(context.overlay.as_deref(), geometry),
                "{crossed:?} must not pass the receipt shape gate"
            );
            assert!(
                matches!(
                    safety_check(&contract, &context),
                    MemorySafetyDecision::Reject { .. }
                ),
                "{crossed:?} must not be admitted"
            );
        }
    }
}
