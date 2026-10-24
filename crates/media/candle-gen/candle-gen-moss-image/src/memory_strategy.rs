//! Structural costs only; no native calibration or optimized-fit measurement is claimed.
use crate::loader::Sources;
use candle_gen::gen_core::{self, moss_image::*, *};

pub(crate) fn footprint_sources(s: &Sources) -> gen_core::Result<PerComponentBytes> {
    let mut result = PerComponentBytes::default();
    for a in &s.ordinary {
        if !a.relative.ends_with(".safetensors") {
            continue;
        }
        a.pin.verify_unchanged()?;
        let bytes = moss_image_asset(a.repo, a.revision, &a.relative)
            .ok_or_else(|| {
                gen_core::Error::Unsupported("MOSS missing frozen footprint identity".into())
            })?
            .bytes;
        match a.role {
            MossImageAssetRole::Transformer => result.dit += bytes,
            MossImageAssetRole::TextEncoder => result.text_encoder += bytes,
            MossImageAssetRole::Vae => result.vae += bytes,
            _ => (),
        }
    }
    Ok(result)
}
pub(crate) fn contract_sources(
    variant: MossImageVariant,
    s: &Sources,
) -> gen_core::Result<MemoryProviderContract> {
    let bytes = footprint_sources(s)?;
    let mut c = MemoryProviderContract::compatibility_default(
        variant.provider_id(),
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: false,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    c.load_shape = s.spec.load_shape;
    c.asset_facts = MemoryAssetFacts {
        base_bytes: bytes.dit + bytes.text_encoder + bytes.vae,
        transformer_bytes: bytes.dit,
        conditioning_bytes: bytes.text_encoder,
        decoder_bytes: bytes.vae,
        overlay_bytes: 0,
    };
    c.architecture_facts = architecture_facts(s);
    c.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        ..Default::default()
    };
    c.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: c.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::PixelCount,
        ],
    };
    // The sequential loader is executable; its optimized-fit rung remains Missing until calibrated.
    Ok(c)
}
/// Conservative request-scaled upper bound from authenticated tensor headers, never a measurement.
pub(crate) fn estimate(
    s: &Sources,
    r: &MossImageResolved,
    req: &GenerationRequest,
    images: usize,
    staged: bool,
) -> gen_core::Result<u64> {
    let f = selected_facts(s, r.acceleration)?;
    let w = workspace(
        req.width,
        req.height,
        images,
        r.max_sequence_length,
        branches(r),
    )?;
    let owned = owned_shard_bytes(s, r.acceleration)?;
    let resident = if staged {
        f.conditioning_bytes
            .saturating_add(w.text)
            .max(
                f.transformer_bytes
                    .saturating_add(f.overlay_bytes)
                    .saturating_add(w.denoise),
            )
            .max(f.decoder_bytes.saturating_add(w.decode))
    } else {
        f.base_bytes
            .saturating_add(f.overlay_bytes)
            .saturating_add(w.denoise.max(w.text).max(w.decode))
    };
    Ok(resident
        .saturating_add(512 * 1024 * 1024)
        .saturating_add(owned))
}

fn selected_facts(s: &Sources, mode: MossImageAcceleration) -> gen_core::Result<MemoryAssetFacts> {
    let mut text = 0u64;
    let mut dit = 0u64;
    let mut vae = 0u64;
    let mut extra = 0u64;
    for a in s.assets(mode)? {
        if !a.relative.ends_with(".safetensors") {
            continue;
        }
        let headers = s.headers.get(a.pin.content_sha256()).ok_or_else(|| {
            gen_core::Error::Unsupported("MOSS missing authenticated tensor header facts".into())
        })?;
        for h in headers {
            let n = h.element_count()?;
            let cost = match a.role {
                MossImageAssetRole::TextEncoder if h.name.starts_with("language_model.") => {
                    text = text.saturating_add(n * 4);
                    0
                }
                MossImageAssetRole::Transformer => {
                    dit = dit.saturating_add(
                        n * if mode == MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32) {
                            4
                        } else {
                            2
                        },
                    );
                    0
                }
                MossImageAssetRole::Vae
                    if h.name.starts_with("decoder.")
                        || h.name.starts_with("post_quant_conv.")
                        || h.name.starts_with("bn.running_") =>
                {
                    vae = vae.saturating_add(n * 4);
                    0
                }
                MossImageAssetRole::DmdLora
                | MossImageAssetRole::DmdTime
                | MossImageAssetRole::DmdDataspace => n * 4,
                _ => 0,
            };
            extra = extra.saturating_add(cost);
        }
    }
    // FP32 time/modulation islands replace BF16 originals; reserve their entire FP32 envelope.
    if mode == MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16) {
        extra = extra
            .saturating_add(4 * (256 * 3072 + 3072 * 3072 + 8 * 3072 * 3072 + 16 * 3072) as u64);
    }
    Ok(MemoryAssetFacts {
        base_bytes: text.saturating_add(dit).saturating_add(vae),
        transformer_bytes: dit,
        conditioning_bytes: text,
        decoder_bytes: vae,
        overlay_bytes: extra,
    })
}

/// Authenticated residency for only the explicitly selected source closure, without device creation.
/// Conditioning is conservatively priced at FP32; optional files do not enable or affect OFF.
pub fn selected_asset_facts(
    variant: MossImageVariant,
    spec: &LoadSpec,
    mode: MossImageAcceleration,
) -> gen_core::Result<MemoryAssetFacts> {
    selected_facts(&Sources::selected(variant, spec, mode)?, mode)
}

/// One-call static admission facts. Workspace excludes the separately reported owned-shard buffer.
#[derive(Clone, Debug)]
pub struct SelectedPreloadEstimate {
    pub asset_facts: MemoryAssetFacts,
    pub workspace_bytes: u64,
    pub owned_shard_transient_bytes: u64,
    pub peak_bytes: u64,
}
/// Validate and price the selected request before the generic provider loader creates a device.
/// The largest selected owned shard is charged once. Source authentication itself uses that host
/// buffer before returning these facts, so the caller must provide host staging headroom as well.
pub fn selected_preload_estimate(
    variant: MossImageVariant,
    spec: &LoadSpec,
    request: &GenerationRequest,
    inputs: MossImageAdvancedInputs<'_>,
) -> gen_core::Result<SelectedPreloadEstimate> {
    let r = MossImageResolved::resolve(variant, request)?;
    let images = inputs.validate(request, &r)?;
    crate::checked_host_elements(request, images)?;
    check_request(request)?;
    let s = Sources::selected_with_cancel(variant, spec, r.acceleration, &request.cancel)?;
    let asset_facts = selected_facts(&s, r.acceleration)?;
    let branches = branches(&r);
    let workspace_bytes = request_workspace_bytes(
        request.width,
        request.height,
        images,
        r.max_sequence_length,
        branches,
    )?;
    let owned_shard_transient_bytes = owned_shard_bytes(&s, r.acceleration)?;
    let staged = request
        .memory
        .map_or(spec.offload_policy == OffloadPolicy::Sequential, |m| {
            m.stage_residency
        });
    let peak_bytes = estimate(&s, &r, request, images, staged)?;
    Ok(SelectedPreloadEstimate {
        asset_facts,
        workspace_bytes,
        owned_shard_transient_bytes,
        peak_bytes,
    })
}

fn branches(r: &MossImageResolved) -> u32 {
    if matches!(r.acceleration, MossImageAcceleration::Dmd(_)) {
        1
    } else {
        1 + u32::from(r.guidance != 1.0)
            + u32::from(
                r.slg
                    .as_ref()
                    .is_some_and(|s| !s.layers.is_empty() && s.scale > 0.0),
            )
    }
}
fn owned_shard_bytes(s: &Sources, mode: MossImageAcceleration) -> gen_core::Result<u64> {
    Ok(s.assets(mode)?
        .iter()
        .filter(|a| a.relative.ends_with(".safetensors"))
        .filter_map(|a| moss_image_asset(a.repo, a.revision, &a.relative))
        .map(|a| a.bytes)
        .max()
        .unwrap_or(0))
}
struct Workspace {
    denoise: u64,
    text: u64,
    decode: u64,
}
fn workspace(
    width: u32,
    height: u32,
    images: usize,
    context: u32,
    branches: u32,
) -> gen_core::Result<Workspace> {
    if width == 0
        || height == 0
        || !width.is_multiple_of(16)
        || !height.is_multiple_of(16)
        || images == 0
        || context == 0
        || context > 4096
        || !(1..=3).contains(&branches)
    {
        return Err(gen_core::Error::Unsupported(
            "invalid MOSS estimate geometry".into(),
        ));
    }
    let pixels = u64::from(width).saturating_mul(u64::from(height));
    let tokens = pixels / 256 + u64::from(context);
    let b = images as u64;
    let attention = b
        .saturating_mul(24)
        .saturating_mul(tokens)
        .saturating_mul(tokens)
        .saturating_mul(8);
    Ok(Workspace {
        denoise: b
            .saturating_mul(tokens)
            .saturating_mul(3072 * 4 * 24)
            .saturating_mul(u64::from(branches))
            .saturating_add(attention),
        text: b
            .saturating_mul(u64::from(context))
            .saturating_mul(4096 * 4 * 32),
        decode: b.saturating_mul(pixels).saturating_mul(96 * 4 * 12),
    })
}
/// Structural scratch plus fixed overhead. Unlike MLX's existing helper, this excludes owned
/// source-shard staging, which `selected_preload_estimate` reports and adds exactly once.
pub fn request_workspace_bytes(
    width: u32,
    height: u32,
    expanded: usize,
    context: u32,
    branches: u32,
) -> gen_core::Result<u64> {
    let w = workspace(width, height, expanded, context, branches)?;
    Ok(w.denoise
        .max(w.text)
        .max(w.decode)
        .saturating_add(512 * 1024 * 1024))
}
pub(crate) fn check_request(req: &GenerationRequest) -> gen_core::Result<()> {
    if req.memory.is_some_and(|m| {
        m.tile_vae_decode
            || m.chunk_attention
            || m.stream_transformer_blocks
            || m.decode_tile_edge.is_some()
            || m.decode_overlap.is_some()
            || m.attention_chunk_size.is_some()
            || m.transformer_window_size.is_some()
            || m.transformer_window_component.is_some()
    }) {
        return Err(gen_core::Error::Unsupported(
            "MOSS does not implement VAE tiling, chunked attention or block streaming".into(),
        ));
    }
    Ok(())
}

fn architecture_facts(s: &Sources) -> MemoryArchitectureFacts {
    MemoryArchitectureFacts {
        attention_heads: Some(s.config.num_heads as u32),
        head_dim: Some((s.config.dim / s.config.num_heads) as u32),
        transformer_blocks: Some(s.config.num_layers as u32),
        patch_size: Some(1),
        latent_channels: Some(s.config.latent_channels as u32),
        vae_spatial_scale: Some(16),
        vae_temporal_scale: None,
        activation_dtype_width: Some(4),
    }
}

/// Cold typed edit admission: validates controls and exact package before device dispatch.
pub fn edit_request_peak(
    variant: MossImageVariant,
    spec: &LoadSpec,
    request: &GenerationRequest,
    inputs: gen_core::moss_image_edit::MossImageEditInputs<'_>,
) -> gen_core::Result<u64> {
    inputs.validate()?;
    let request = gen_core::moss_image_edit::edit_request(request)?;
    let resolved = MossImageResolved::resolve(variant, &request)?;
    let images = inputs.advanced.validate(&request, &resolved)?;
    crate::checked_host_elements(&request, images)?;
    check_request(&request)?;
    crate::transformer::check_cancel(&request.cancel)?;
    let edit = crate::edit::extra_peak(inputs, &request)?;
    crate::edit::EditAssets::prepare(spec)?;
    let sources =
        Sources::selected_with_cancel(variant, spec, MossImageAcceleration::Off, &request.cancel)?;
    let staged = request
        .memory
        .map_or(spec.offload_policy == OffloadPolicy::Sequential, |m| {
            m.stage_residency
        });
    estimate(&sources, &resolved, &request, images, staged)?
        .checked_add(edit)
        .ok_or_else(|| gen_core::Error::Unsupported("MOSS edit peak overflow".into()))
}
pub(crate) fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    if (context.geometry.reference_count != 0
        && !((1..=4).contains(&context.geometry.reference_count)
            && spec
                .components
                .contains_key(gen_core::moss_image_edit::EDIT_COMPONENT)))
        || context.use_pid
        || context.has_phases
    {
        return MemorySafetyDecision::Reject { reason: "MOSS accepts one to four typed edit reference slots only with its exact edit component; PID/phases are unsupported".into() };
    }
    gen_core::default_registered_memory_strategy_safety_check(spec, contract, context)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordered_edit_provider_guard_accepts_one_to_four_only_with_component() {
        let mut spec = LoadSpec::new(WeightsSource::Dir("unused-memory-guard".into()));
        let contract =
            contract_sources(MossImageVariant::Sft, &crate::tests::synthetic_sources()).unwrap();
        let mut context = MemoryRunContext {
            selection: MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: Default::default(),
                tier: MemoryNumericTier {
                    precision: spec.precision,
                    quant: None,
                    component_precision_floors: &[],
                },
            },
            optimization_authority: MemoryOptimizationAuthority::Resident,
            calibration_abi: 0,
            calibration_fingerprint: String::new(),
            load_shape: spec.load_shape,
            mode: MemoryMode::Edit,
            has_reference: true,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 512,
                height: 512,
                batch: 1,
                frames: 1,
                reference_count: 1,
            },
            reference_resources: None,
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 10_000_000_000,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1_000_000_000,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "ordered-slots-structural-test".into(),
        };
        assert!(matches!(
            safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
        spec.components.insert(
            gen_core::moss_image_edit::EDIT_COMPONENT.into(),
            WeightsSource::Dir("exact-edit-component".into()),
        );
        for count in 1..=4 {
            context.geometry.reference_count = count;
            assert_eq!(
                safety_check(&spec, &contract, &context),
                MemorySafetyDecision::Accept
            );
        }
        let references: Vec<_> = [(32, 32), (16, 48), (48, 16), (16, 48)]
            .into_iter()
            .map(|(width, height)| Image {
                width,
                height,
                pixels: vec![0; (width * height * 3) as usize],
            })
            .collect();
        let request = GenerationRequest {
            width: 512,
            height: 512,
            ..Default::default()
        };
        let mut previous = 0;
        for count in 1..=4 {
            let inputs = gen_core::moss_image_edit::MossImageEditInputs {
                references: &references[..count],
                ..Default::default()
            };
            let bytes = crate::edit::extra_peak(inputs, &request).unwrap();
            assert!(
                bytes > previous,
                "every ordered slot adds residency and attention"
            );
            previous = bytes;
        }
        // Real provider safety budget rejection, before any provider/device callback.
        context.predicted_peak_bytes = previous;
        context.budget.total_bytes = previous - 1;
        assert!(matches!(
            safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
        context.budget.total_bytes = previous;
        assert_eq!(
            safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Accept
        );
        let overflowing = GenerationRequest {
            width: u32::MAX,
            height: u32::MAX,
            ..Default::default()
        };
        assert!(crate::edit::extra_peak(
            gen_core::moss_image_edit::MossImageEditInputs {
                references: &references,
                ..Default::default()
            },
            &overflowing
        )
        .is_err());
        context.geometry.reference_count = 5;
        assert!(matches!(
            safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
        context.geometry.reference_count = 4;
        context.use_pid = true;
        assert!(matches!(
            safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
        context.use_pid = false;
        context.has_phases = true;
        assert!(matches!(
            safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
        context.has_phases = false;
        context.geometry.reference_count = 0;
        context.has_reference = false;
        context.mode = MemoryMode::TextToImage;
        spec.components.clear();
        assert_eq!(
            safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Accept
        );
    }
    #[test]
    fn static_workspace_is_shared_with_runtime_and_excludes_shard_once() {
        let s = crate::tests::synthetic_sources();
        let request = GenerationRequest::default();
        let r = MossImageResolved::resolve(MossImageVariant::Sft, &request).unwrap();
        let scratch = request_workspace_bytes(
            request.width,
            request.height,
            1,
            r.max_sequence_length,
            branches(&r),
        )
        .unwrap();
        assert_eq!(estimate(&s, &r, &request, 1, false).unwrap(), scratch);
        assert!(request_workspace_bytes(0, 16, 1, 1, 1).is_err());
        assert!(request_workspace_bytes(16, 16, 1, 0, 1).is_err());
        assert!(request_workspace_bytes(16, 16, 0, 1, 1).is_err());
        assert!(request_workspace_bytes(16, 16, 1, 1, 4).is_err());
    }
    #[test]
    fn selected_facts_price_only_selected_headers_and_charge_largest_owned_shard_once() {
        let mut s = crate::tests::synthetic_sources();
        let tmp = tempfile::tempdir().unwrap();
        let mut add = |file: &str, role, repo, revision, relative: &str, name: &str, n: usize| {
            let path = tmp.path().join(file);
            std::fs::write(&path, file).unwrap();
            let pin = PinnedWeightsFile::pin(path).unwrap();
            let h = weightsmeta::SafetensorsTensorHeader {
                name: name.into(),
                dtype: weightsmeta::Dtype::F32,
                shape: vec![n],
                data_bytes: (n * 4) as u64,
            };
            s.headers.insert(*pin.content_sha256(), vec![h]);
            crate::loader::Asset {
                pin,
                role,
                repo,
                revision,
                relative: relative.into(),
            }
        };
        let base = add(
            "base",
            MossImageAssetRole::Transformer,
            SFT_REPO,
            SFT_REVISION,
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "weight",
            6,
        );
        let full = add(
            "full",
            MossImageAssetRole::Transformer,
            DMD_REPO,
            DMD_REVISION,
            "transformer/diffusion_pytorch_model-00001-of-00004.safetensors",
            "weight",
            6,
        );
        let q = add(
            "Q",
            MossImageAssetRole::DmdDataspace,
            DMD_REPO,
            DMD_REVISION,
            "dataspace.safetensors",
            "latent_mix.Q",
            4,
        );
        let lora = add(
            "lora",
            MossImageAssetRole::DmdLora,
            DMD_REPO,
            DMD_REVISION,
            "lora.safetensors",
            "lora",
            3,
        );
        let time = add(
            "time",
            MossImageAssetRole::DmdTime,
            DMD_REPO,
            DMD_REVISION,
            "time_embedding.safetensors",
            "time",
            2,
        );
        s.ordinary = vec![base];
        s.full = Some(vec![full, q.clone()]);
        s.mixed = Some(vec![lora, time, q]);
        let off = selected_facts(&s, MossImageAcceleration::Off).unwrap();
        assert_eq!(off.transformer_bytes, 12);
        assert_eq!(off.overlay_bytes, 0);
        let full =
            selected_facts(&s, MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32)).unwrap();
        assert_eq!(full.transformer_bytes, 24);
        assert_eq!(full.overlay_bytes, 16);
        let mixed = selected_facts(
            &s,
            MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16),
        )
        .unwrap();
        assert_eq!(mixed.transformer_bytes, 12);
        assert!(mixed.overlay_bytes > 36);
        let request = GenerationRequest::default();
        let mut r = MossImageResolved::resolve(MossImageVariant::Sft, &request).unwrap();
        for mode in [
            MossImageAcceleration::Off,
            MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32),
        ] {
            r.acceleration = mode;
            let f = selected_facts(&s, mode).unwrap();
            let workspace = request_workspace_bytes(
                request.width,
                request.height,
                1,
                r.max_sequence_length,
                branches(&r),
            )
            .unwrap();
            let shard = owned_shard_bytes(&s, mode).unwrap();
            assert_eq!(
                estimate(&s, &r, &request, 1, false).unwrap(),
                f.base_bytes + f.overlay_bytes + workspace + shard
            );
        }
        let off = estimate(
            &s,
            &MossImageResolved::resolve(MossImageVariant::Sft, &request).unwrap(),
            &request,
            1,
            false,
        )
        .unwrap();
        for a in s.full.iter().flatten().chain(s.mixed.iter().flatten()) {
            std::fs::remove_file(a.pin.loader_path()).ok();
        }
        assert_eq!(
            estimate(
                &s,
                &MossImageResolved::resolve(MossImageVariant::Sft, &request).unwrap(),
                &request,
                1,
                false
            )
            .unwrap(),
            off
        );
        assert!(
            selected_facts(&s, MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32)).is_err()
        );
    }
}
