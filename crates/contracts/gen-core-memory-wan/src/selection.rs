//! Selection receipts, evidence revisions and context validation.

use super::*;

pub(super) fn update_optional_u32(hash: &mut Sha256, value: Option<u32>) {
    match value {
        Some(value) => {
            hash.update([1]);
            hash.update(value.to_le_bytes());
        }
        None => hash.update([0]),
    }
}

pub(super) fn update_optional_f32(hash: &mut Sha256, value: Option<f32>) {
    match value {
        Some(value) => {
            hash.update([1]);
            hash.update(value.to_bits().to_le_bytes());
        }
        None => hash.update([0]),
    }
}

pub(super) fn update_optional_str(hash: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => {
            hash.update([1]);
            hash.update((value.len() as u64).to_le_bytes());
            hash.update(value.as_bytes());
        }
        None => hash.update([0]),
    }
}

/// Stable digest of the complete selected execution carrier.
///
/// Keep every field explicit: `None` and `Some(0)` are distinct, and fields not currently supported
/// by Wan remain identity-bearing so a future contract cannot add one without silently reusing an
/// older receipt format.
pub(super) fn selection_receipt_digest(selection: &MemorySelection) -> String {
    let parameters = selection.parameters;
    let mut hash = Sha256::new();
    hash.update(b"wan-i2v-selection-v1");
    hash.update([selection.strategy as u8]);
    update_optional_u32(&mut hash, parameters.decode_tile_edge);
    update_optional_u32(&mut hash, parameters.decode_overlap);
    update_optional_u32(&mut hash, parameters.attention_chunk_size);
    update_optional_u32(&mut hash, parameters.transformer_window_size);
    match parameters.transformer_window_component {
        Some(component) => {
            hash.update([1]);
            hash.update([component as u8]);
        }
        None => hash.update([0]),
    }
    format!("{:x}", hash.finalize())
}

pub(super) fn validate_selection(
    prepared: &PreparedWanI2vMemory,
    mode: WanPublicMode,
    selection: &MemorySelection,
) -> gen_core::Result<()> {
    if selection.tier != prepared.tier {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: selected tier does not match the sealed physical tier",
            prepared.route.provider_id()
        )));
    }
    request_contract_for_mode(prepared, mode).validate_selection(selection)
}

pub(super) fn request_evidence_revision_for_selection(
    prepared: &PreparedWanI2vMemory,
    request: &GenerationRequest,
    selection: &MemorySelection,
) -> gen_core::Result<String> {
    let mode = validate_request(prepared, request)?;
    validate_selection(prepared, mode, selection)?;
    let selection_receipt = selection_receipt_digest(selection);
    let mut hash = Sha256::new();
    hash.update(RECEIPT_VERSION.as_bytes());
    hash.update(prepared.artifact_identity.as_bytes());
    hash.update(prepared.adapter_identity.as_bytes());
    hash.update(prepared.backend.key().as_bytes());
    hash.update(prepared.route.provider_id().as_bytes());
    hash.update(mode.key().as_bytes());
    hash.update(selection_receipt.as_bytes());
    hash.update(request.width.to_le_bytes());
    hash.update(request.height.to_le_bytes());
    hash.update(request.frames.unwrap().to_le_bytes());
    hash.update(request.fps.unwrap().to_le_bytes());
    hash.update(request.steps.unwrap().to_le_bytes());
    hash.update(request.seed.unwrap().to_le_bytes());
    hash.update(request.prompt.as_bytes());
    update_optional_str(&mut hash, request.negative_prompt.as_deref());
    update_optional_f32(&mut hash, request.guidance);
    update_optional_str(&mut hash, request.sampler.as_deref());
    update_optional_f32(&mut hash, request.control_scale);
    update_optional_str(&mut hash, request.video_mode.as_deref());
    hash.update(NATIVE_SCHEDULE.as_bytes());
    match mode {
        WanPublicMode::TextToVideo => {
            text_to_video_carrier(request)?;
            // Named explicitly so a text-only receipt can never collide with an image-carrying one
            // whose carrier happened to hash to nothing.
            hash.update(b"text-only-no-carrier");
        }
        WanPublicMode::ImageToVideo => {
            let image = reference(request)?;
            hash.update(b"reference");
            hash.update(image.width.to_le_bytes());
            hash.update(image.height.to_le_bytes());
            hash.update(Sha256::digest(&image.pixels));
        }
        WanPublicMode::FirstLastFrame => {
            for (role, keyframe) in ["first", "last"]
                .into_iter()
                .zip(first_last_keyframes(request)?)
            {
                hash.update(role.as_bytes());
                hash.update(keyframe.frame_idx.to_le_bytes());
                hash.update(keyframe.strength.to_bits().to_le_bytes());
                hash.update(keyframe.image.width.to_le_bytes());
                hash.update(keyframe.image.height.to_le_bytes());
                hash.update(Sha256::digest(&keyframe.image.pixels));
            }
        }
        WanPublicMode::ExtendClip if prepared.route == WanI2vRoute::Ti2v5b => {
            let keyframe = extend_keyframe(request)?;
            hash.update(b"boundary-keyframe");
            hash.update(keyframe.frame_idx.to_le_bytes());
            hash.update(keyframe.strength.to_bits().to_le_bytes());
            hash.update(keyframe.image.width.to_le_bytes());
            hash.update(keyframe.image.height.to_le_bytes());
            hash.update(Sha256::digest(&keyframe.image.pixels));
        }
        WanPublicMode::ExtendClip => {
            let clip = extend_control_clip(request)?;
            hash.update(b"vace-control-clip");
            hash.update(clip.masking_strength.to_bits().to_le_bytes());
            hash.update(clip.start_frame.to_le_bytes());
            for (index, (frame, mask)) in clip.frames.iter().zip(clip.mask).enumerate() {
                hash.update((index as u32).to_le_bytes());
                hash.update(Sha256::digest(&frame.pixels));
                hash.update(Sha256::digest(&mask.pixels));
            }
        }
        WanPublicMode::VideoBridge if prepared.route == WanI2vRoute::Ti2v5b => {
            for (role, keyframe) in ["left-tail", "right-head"]
                .into_iter()
                .zip(bridge_keyframes(request)?)
            {
                hash.update(role.as_bytes());
                hash.update(keyframe.frame_idx.to_le_bytes());
                hash.update(keyframe.strength.to_bits().to_le_bytes());
                hash.update(keyframe.image.width.to_le_bytes());
                hash.update(keyframe.image.height.to_le_bytes());
                hash.update(Sha256::digest(&keyframe.image.pixels));
            }
        }
        WanPublicMode::VideoBridge => {
            let clip = bridge_control_clip(request)?;
            hash.update(b"vace-bridge-control-clip");
            hash.update(clip.masking_strength.to_bits().to_le_bytes());
            hash.update(clip.start_frame.to_le_bytes());
            for (index, (frame, mask)) in clip.frames.iter().zip(clip.mask).enumerate() {
                hash.update((index as u32).to_le_bytes());
                hash.update(Sha256::digest(&frame.pixels));
                hash.update(Sha256::digest(&mask.pixels));
            }
        }
        WanPublicMode::ReplacePerson => {
            let (clip, references) = replace_person_carrier(request)?;
            hash.update(b"vace-replace-person-control-clip");
            hash.update([clip.mode as u8]);
            hash.update(clip.masking_strength.to_bits().to_le_bytes());
            hash.update(clip.start_frame.to_le_bytes());
            for (index, (frame, mask)) in clip.frames.iter().zip(clip.mask).enumerate() {
                hash.update((index as u32).to_le_bytes());
                hash.update(Sha256::digest(&frame.pixels));
                hash.update(Sha256::digest(&mask.pixels));
            }
            for (ordinal, reference) in references.into_iter().enumerate() {
                hash.update((ordinal as u32).to_le_bytes());
                hash.update(reference.width.to_le_bytes());
                hash.update(reference.height.to_le_bytes());
                hash.update(Sha256::digest(&reference.pixels));
            }
        }
    }
    Ok(format!(
        "{RECEIPT_VERSION}:{}:fps{}:{}:{selection_receipt}:{:x}",
        mode.key(),
        request.fps.unwrap(),
        prepared.artifact_identity,
        hash.finalize()
    ))
}

/// The public frame rate carried in plain text by a [`RECEIPT_VERSION`] evidence revision, together
/// with the receipt tail that follows it.
///
/// Returned as `(fps, tail)` where `tail` is everything after `…:fps<fps>:` — the artifact identity,
/// selection receipt and digest. `None` when the revision is not a receipt of this version and mode.
pub(super) fn receipt_rate_and_tail(
    evidence_revision: &str,
    mode: WanPublicMode,
) -> Option<(u32, &str)> {
    let rest = evidence_revision.strip_prefix(RECEIPT_VERSION)?;
    let rest = rest.strip_prefix(':')?;
    let rest = rest.strip_prefix(mode.key())?;
    let rest = rest.strip_prefix(':')?;
    let (rate, tail) = rest.split_once(':')?;
    let rate = rate.strip_prefix("fps")?;
    // `parse` accepts a leading `+`; the emitter never writes one, so refuse a re-spelling that
    // would let two distinct receipts claim the same rate.
    if rate.is_empty() || !rate.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some((rate.parse().ok()?, tail))
}

pub fn validate_context(
    prepared: &PreparedWanI2vMemory,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    prepared.ensure_unchanged()?;
    let geometry = context.geometry;
    let mode = match context.mode.as_key() {
        // The VACE routes have no exact public I2V request contract (`request_mode` and
        // `contract_for_mode_key` both refuse it), so admission must refuse it here too rather
        // than admit a row no request can ever execute against.
        "text_to_video" if prepared.route == WanI2vRoute::T2v14b => WanPublicMode::TextToVideo,
        "image_to_video"
            if prepared.route != WanI2vRoute::Vace && prepared.route != WanI2vRoute::T2v14b =>
        {
            WanPublicMode::ImageToVideo
        }
        "first_last_frame" if prepared.route == WanI2vRoute::Ti2v5b => {
            WanPublicMode::FirstLastFrame
        }
        "extend_clip"
            if prepared.route == WanI2vRoute::Vace
                || (prepared.route == WanI2vRoute::Ti2v5b
                    && prepared.backend == WanI2vBackend::Mlx) =>
        {
            WanPublicMode::ExtendClip
        }
        "video_bridge"
            if prepared.route == WanI2vRoute::Vace
                || (prepared.route == WanI2vRoute::Ti2v5b
                    && prepared.backend == WanI2vBackend::Mlx) =>
        {
            WanPublicMode::VideoBridge
        }
        "replace_person" if matches!(prepared.route, WanI2vRoute::Vace | WanI2vRoute::VaceFun) => {
            WanPublicMode::ReplacePerson
        }
        _ => {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: crossed Wan public memory mode",
                prepared.route.provider_id()
            )))
        }
    };
    let request_contract = request_contract_for_mode(prepared, mode);
    let selection_receipt = selection_receipt_digest(&context.selection);
    // The receipt carries its own frame rate (see [`RECEIPT_VERSION`]), so the exact public rate
    // menu is pinned here instead of OR-ing 16 and 24 fps into one indistinguishable cell.
    let Some((fps, tail)) = receipt_rate_and_tail(&context.evidence_revision, mode) else {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: crossed Wan I2V memory context",
            prepared.route.provider_id()
        )));
    };
    let tail_prefix = format!("{}:{selection_receipt}:", prepared.artifact_identity);
    let rate_ok = match mode {
        WanPublicMode::TextToVideo | WanPublicMode::ImageToVideo => {
            prepared.route.accepts_rate(fps, geometry.frames)
        }
        WanPublicMode::FirstLastFrame => {
            prepared.route.accepts_first_last_rate(fps, geometry.frames)
        }
        WanPublicMode::ExtendClip | WanPublicMode::VideoBridge => match prepared.route {
            WanI2vRoute::Ti2v5b => prepared.route.accepts_first_last_rate(fps, geometry.frames),
            WanI2vRoute::Vace => prepared.route.accepts_rate(fps, geometry.frames),
            WanI2vRoute::T2v14b | WanI2vRoute::I2v14b | WanI2vRoute::VaceFun => false,
        },
        WanPublicMode::ReplacePerson => prepared.route.accepts_rate(fps, geometry.frames),
    };
    let carrier_ok = match mode {
        WanPublicMode::TextToVideo => geometry.reference_count == 0,
        WanPublicMode::ImageToVideo => geometry.reference_count == 1,
        WanPublicMode::FirstLastFrame => geometry.reference_count == 2,
        WanPublicMode::ExtendClip => geometry.reference_count == 1,
        WanPublicMode::VideoBridge => geometry.reference_count == 0,
        WanPublicMode::ReplacePerson => (1..=4).contains(&geometry.reference_count),
    };
    let selection_ok = match mode {
        WanPublicMode::ExtendClip | WanPublicMode::VideoBridge => {
            matches!(
                context.selection.strategy,
                MemoryStrategy::Resident | MemoryStrategy::BoundedDecode
            )
        }
        WanPublicMode::ReplacePerson if prepared.route == WanI2vRoute::Vace => matches!(
            context.selection.strategy,
            MemoryStrategy::Resident | MemoryStrategy::BoundedDecode
        ),
        WanPublicMode::ReplacePerson => {
            matches!(
                context.selection.strategy,
                MemoryStrategy::Resident
                    | MemoryStrategy::StagedResidency
                    | MemoryStrategy::BoundedDecode
            )
        }
        WanPublicMode::FirstLastFrame if prepared.backend == WanI2vBackend::Mlx => matches!(
            context.selection.strategy,
            MemoryStrategy::Resident | MemoryStrategy::BoundedDecode
        ),
        WanPublicMode::TextToVideo
        | WanPublicMode::ImageToVideo
        | WanPublicMode::FirstLastFrame => matches!(
            context.selection.strategy,
            MemoryStrategy::Resident
                | MemoryStrategy::StagedResidency
                | MemoryStrategy::BoundedDecode
        ),
    };
    // Which public modes carry a reference still, named rather than written as the negation of the
    // two that do not: `has_reference != !matches!(..)` said the same thing and read as a typo, and
    // `-D clippy::nonminimal_bool` refuses it on the CUDA lane.
    let carries_reference = !matches!(
        mode,
        WanPublicMode::VideoBridge | WanPublicMode::TextToVideo
    );
    if !selection_ok
        || geometry.batch != 1
        || !carrier_ok
        || context.has_reference != carries_reference
        || !prepared
            .route
            .public_geometries()
            .contains(&(geometry.width, geometry.height))
        || !rate_ok
        || context.overlay.as_deref() != Some(prepared.adapter_identity.as_str())
        || context.use_pid
        || context.has_phases
        || !tail.starts_with(&tail_prefix)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: crossed Wan I2V memory context",
            prepared.route.provider_id()
        )));
    }
    gen_core::admit_memory_safety(gen_core::standard_memory_strategy_safety_check(
        &request_contract,
        context,
        Some(prepared.tier),
        None,
    ))
}

/// Reconstruct the exact selection carried by an executing request and verify that no request
/// control was added, dropped, or relabeled after admission.
pub fn selection_from_request(
    prepared: &PreparedWanI2vMemory,
    request: &GenerationRequest,
) -> gen_core::Result<MemorySelection> {
    let mode = validate_request(prepared, request)?;
    let request_contract = request_contract_for_mode(prepared, mode);
    let memory = request.memory.ok_or_else(|| {
        gen_core::Error::Unsupported(format!(
            "{}: admitted Wan I2V request is missing its explicit memory carrier",
            prepared.route.provider_id()
        ))
    })?;
    let strategy = if memory.stream_transformer_blocks {
        MemoryStrategy::BoundedTransformerResidency
    } else if memory.chunk_attention {
        MemoryStrategy::BoundedAttention
    } else if memory.tile_vae_decode {
        MemoryStrategy::BoundedDecode
    } else if memory.stage_residency {
        MemoryStrategy::StagedResidency
    } else {
        MemoryStrategy::Resident
    };
    let selection = MemorySelection {
        strategy,
        parameters: MemoryStrategyParameters {
            stage_residency: None,
            decode_tile_edge: memory.decode_tile_edge,
            decode_overlap: memory.decode_overlap,
            attention_chunk_size: memory.attention_chunk_size,
            transformer_window_size: memory.transformer_window_size,
            transformer_window_component: memory.transformer_window_component,
        },
        tier: prepared.tier,
    };
    validate_selection(prepared, mode, &selection)?;
    if request_contract.generation_memory(&selection) != Some(memory) {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: request memory controls do not match the selected Wan I2V rung",
            prepared.route.provider_id()
        )));
    }
    Ok(selection)
}

/// Provider-facing receipt minting after admission has installed the selected memory carrier.
/// Retaining this two-argument surface keeps paired callers source-compatible while making the
/// request's exact carrier authoritative for strategy/parameter identity.
pub fn request_evidence_revision(
    prepared: &PreparedWanI2vMemory,
    request: &GenerationRequest,
) -> gen_core::Result<String> {
    let selection = selection_from_request(prepared, request)?;
    request_evidence_revision_for_selection(prepared, request, &selection)
}

/// Configure-boundary validation: the request must carry the same exact rung/parameters as the
/// admitted context, and the full selection-bound request receipt must still match byte-for-byte.
pub fn validate_request_evidence(
    prepared: &PreparedWanI2vMemory,
    request: &GenerationRequest,
    admitted_selection: &MemorySelection,
    admitted_evidence: &str,
) -> gen_core::Result<String> {
    let request_selection = selection_from_request(prepared, request)?;
    if request_selection != *admitted_selection {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: request rung/parameters crossed after admission",
            prepared.route.provider_id()
        )));
    }
    let actual = request_evidence_revision_for_selection(prepared, request, admitted_selection)?;
    if actual != admitted_evidence {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: request axes crossed after admission",
            prepared.route.provider_id()
        )));
    }
    Ok(actual)
}

pub fn geometry_from_request(request: &GenerationRequest) -> MemoryGeometry {
    MemoryGeometry {
        width: request.width,
        height: request.height,
        batch: request.count,
        frames: request.frames.unwrap_or_default(),
        // Clip carriers are execution inputs, not semantic image references.  In particular a
        // bridge has two endpoint carriers but zero conceptual references, so it cannot borrow
        // a still-image/FLF memory row merely because it has conditioning entries.
        //
        // This is deliberately the *same* function both request-scope cores compare an executing
        // request against ([`GenerationRequest::memory_reference_count`]).  Admission and execution
        // previously counted the carriers with two separate rules, and they disagreed on the
        // Wan-VACE `extend_clip` ControlClip (admission 1, execution 0) and on the Ti2V-5B
        // `video_bridge` keyframe pair (admission 0, execution 2) — so neither mode could ever pass
        // both gates.  One authority, no drift.
        reference_count: request.memory_reference_count(),
    }
}

pub fn load_policy_for_selection(selection: MemoryStrategy) -> OffloadPolicy {
    if selection == MemoryStrategy::StagedResidency || selection == MemoryStrategy::BoundedDecode {
        OffloadPolicy::Sequential
    } else {
        OffloadPolicy::Resident
    }
}
