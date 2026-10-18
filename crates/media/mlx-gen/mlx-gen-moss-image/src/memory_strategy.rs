//! Structural resident admission. No unmeasured calibration or unused optimization lever.
use crate::loader::Sources;
use mlx_gen::gen_core::moss_image::MossImageVariant;
use mlx_gen::gen_core::*;

/// Asset residency for the explicitly selected policy. Optional cached files do not imply a
/// loaded renderer. Header parsing authenticates the same owned bytes and caches by content pin;
/// this function does not construct tensors or execute Metal work.
pub fn selected_asset_facts(
    variant: MossImageVariant,
    spec: &LoadSpec,
    mode: moss_image::MossImageAcceleration,
) -> Result<MemoryAssetFacts> {
    let sources = Sources::selected(variant, spec, mode).map_err(Error::from)?;
    selected_facts(&sources, mode)
}

pub(crate) fn facts(s: &Sources) -> Result<MemoryAssetFacts> {
    selected_facts(s, moss_image::MossImageAcceleration::Off)
}

pub(crate) fn selected_facts(
    s: &Sources,
    mode: moss_image::MossImageAcceleration,
) -> Result<MemoryAssetFacts> {
    selected_facts_with(s, mode, |asset| asset.headers().map_err(Error::from))
}

fn selected_facts_with(
    s: &Sources,
    mode: moss_image::MossImageAcceleration,
    mut headers_for: impl FnMut(
        &crate::loader::Asset,
    ) -> Result<Vec<weightsmeta::SafetensorsTensorHeader>>,
) -> Result<MemoryAssetFacts> {
    let headers = s
        .mode_assets(mode)
        .map_err(Error::from)?
        .into_iter()
        .filter(|a| a.relative.ends_with(".safetensors"))
        .map(|a| headers_for(a).map(|headers| (a.role, headers)))
        .collect::<Result<Vec<_>>>()?;
    sum_selected_headers(mode, &headers)
}

fn sum_selected_headers(
    mode: moss_image::MossImageAcceleration,
    selected: &[(
        moss_image::MossImageAssetRole,
        Vec<weightsmeta::SafetensorsTensorHeader>,
    )],
) -> Result<MemoryAssetFacts> {
    use moss_image::{MossDmdPrecision, MossImageAcceleration, MossImageAssetRole};
    let mut f = MemoryAssetFacts::default();
    for (role, headers) in selected {
        for h in headers {
            let target = match role {
                MossImageAssetRole::TextEncoder if h.name.starts_with("language_model.") => {
                    &mut f.conditioning_bytes
                }
                MossImageAssetRole::Vae
                    if h.name.starts_with("decoder.")
                        || h.name.starts_with("post_quant_conv.")
                        || h.name == "bn.running_mean"
                        || h.name == "bn.running_var" =>
                {
                    &mut f.decoder_bytes
                }
                MossImageAssetRole::Transformer => &mut f.transformer_bytes,
                MossImageAssetRole::DmdLora | MossImageAssetRole::DmdDataspace => {
                    &mut f.overlay_bytes
                }
                // Replacement tensors replace the original time/bias tensors. The FP32 width
                // delta is priced below; charging their full payload again would double count.
                _ => continue,
            };
            // The shared small VAE decoder is explicitly materialized in FP32.
            let bytes = if *role == MossImageAssetRole::Vae {
                h.materialized_bytes(4)?
            } else {
                h.data_bytes
            };
            *target = target
                .checked_add(bytes)
                .ok_or_else(|| Error::Unsupported("MOSS resident bytes overflow".into()))?;
            if mode == MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16)
                && *role == MossImageAssetRole::Transformer
                && (h.name.starts_with("time_embedding.")
                    || h.name.starts_with("shared_block_modulation.")
                    || h.name.starts_with("video_head_modulation."))
            {
                let extra = h
                    .materialized_bytes(4)?
                    .checked_sub(h.data_bytes)
                    .ok_or_else(|| {
                        Error::Unsupported("MOSS island source wider than FP32".into())
                    })?;
                f.overlay_bytes = f
                    .overlay_bytes
                    .checked_add(extra)
                    .ok_or_else(|| Error::Unsupported("MOSS island bytes overflow".into()))?;
            }
        }
    }
    f.base_bytes = f
        .transformer_bytes
        .checked_add(f.conditioning_bytes)
        .and_then(|n| n.checked_add(f.decoder_bytes))
        .ok_or_else(|| Error::Unsupported("MOSS base bytes overflow".into()))?;
    Ok(f)
}
pub(crate) fn contract_with_facts(
    v: MossImageVariant,
    f: MemoryAssetFacts,
) -> MemoryProviderContract {
    let mut c = MemoryProviderContract::compatibility_default(
        v.provider_id(),
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: false,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    c.asset_facts = f;
    c.architecture_facts = MemoryArchitectureFacts {
        attention_heads: Some(24),
        head_dim: Some(128),
        transformer_blocks: Some(42),
        patch_size: Some(1),
        latent_channels: Some(128),
        vae_spatial_scale: Some(16),
        vae_temporal_scale: None,
        // FP32 is the widest reachable denoise activation policy (mixed/full DMD).
        activation_dtype_width: Some(4),
    };
    c
}
pub(crate) fn contract(v: MossImageVariant, spec: &LoadSpec) -> Result<MemoryProviderContract> {
    let s = Sources::prepare(v, spec).map_err(Error::from)?;
    Ok(contract_with_facts(v, facts(&s)?))
}
pub(crate) fn fixture(v: MossImageVariant, _: &LoadSpec) -> Result<MemoryProviderContract> {
    let mut c = contract_with_facts(v, MemoryAssetFacts::default());
    c.architecture_facts = MemoryArchitectureFacts::default();
    Ok(c)
}
/// Conservative structural workspace, including expanded outputs, context and sequential branches.
/// This is an estimate, not a measured calibration rung.
pub fn request_workspace_bytes(
    width: u32,
    height: u32,
    expanded: usize,
    context: u32,
    branches: u32,
) -> Result<u64> {
    if width == 0
        || height == 0
        || !width.is_multiple_of(16)
        || !height.is_multiple_of(16)
        || expanded == 0
        || context == 0
        || context > 4096
        || !(1..=3).contains(&branches)
    {
        return Err(Error::Unsupported("invalid MOSS estimate geometry".into()));
    }
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| Error::Unsupported("MOSS pixels overflow".into()))?;
    let latent = pixels / 256;
    let seq = latent
        .checked_add(u64::from(context))
        .ok_or_else(|| Error::Unsupported("MOSS sequence overflow".into()))?;
    let b =
        u64::try_from(expanded).map_err(|_| Error::Unsupported("MOSS batch overflow".into()))?;
    let linear = seq
        .checked_mul(4 * (3072 * 12 + 8192 * 4))
        .ok_or_else(|| Error::Unsupported("MOSS scratch overflow".into()))?;
    let attention = seq
        .checked_mul(seq)
        .and_then(|n| n.checked_mul(24 * 4))
        .ok_or_else(|| Error::Unsupported("MOSS attention overflow".into()))?;
    let outputs = latent
        .checked_mul(128 * 4)
        .and_then(|n| n.checked_mul(b))
        .and_then(|n| n.checked_mul(u64::from(branches) + 4))
        .ok_or_else(|| Error::Unsupported("MOSS output scratch overflow".into()))?;
    let embeddings = u64::from(context)
        .checked_mul(4096 * 4)
        .and_then(|n| n.checked_mul(b))
        .and_then(|n| n.checked_mul(u64::from(branches)))
        .ok_or_else(|| Error::Unsupported("MOSS conditioning overflow".into()))?;
    // Small VAE FP32 activation envelope and one owned source-shard staging buffer.
    let decode = pixels
        .checked_mul(96 * 4 * 6)
        .ok_or_else(|| Error::Unsupported("MOSS decode scratch overflow".into()))?;
    linear
        .checked_add(attention)
        .and_then(|n| n.checked_add(outputs))
        .and_then(|n| n.checked_add(embeddings))
        .and_then(|n| n.checked_add(decode))
        .and_then(|n| n.checked_add(5_362_224_248))
        .ok_or_else(|| Error::Unsupported("MOSS peak overflow".into()))
}
pub(crate) fn safety(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    if context.geometry.reference_count != 0 || context.use_pid || context.has_phases {
        return MemorySafetyDecision::Reject {
            reason: "MOSS supports text-to-image only".into(),
        };
    }
    default_registered_memory_strategy_safety_check(spec, contract, context)
}
pub(crate) fn surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    mlx_memory_contract_surface_specs()
        .into_iter()
        .filter(|s| {
            s.selector.tier == MemoryContractSurfaceTier::Bf16
                && s.spec.offload_policy == OffloadPolicy::Resident
                && s.spec.load_shape == LoadShape::EagerMaterialization
        })
        .collect()
}

#[cfg(test)]
mod cpu_tests {
    use super::*;
    #[test]
    fn cached_optional_assets_only_price_the_selected_renderer() {
        use crate::loader::Asset;
        use moss_image::{MossDmdPrecision, MossImageAcceleration, MossImageAssetRole};
        use weightsmeta::{Dtype, SafetensorsTensorHeader};
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let pin = PinnedWeightsFile::pin(tmp.path()).unwrap();
        let asset = |role, relative: &str| Asset {
            role,
            relative: relative.into(),
            pin: pin.clone(),
            repo: "test",
            revision: "test",
        };
        let mut sources = Sources {
            selection: None,
            ordinary: vec![
                asset(MossImageAssetRole::Transformer, "ordinary.safetensors"),
                asset(MossImageAssetRole::TextEncoder, "text.safetensors"),
                asset(MossImageAssetRole::Vae, "vae.safetensors"),
            ],
            optional: vec![],
            config: moss_image::MossImageConfig::default(),
        };
        let header = |name: &str, dtype, bytes| SafetensorsTensorHeader {
            name: name.into(),
            dtype,
            shape: vec![2],
            data_bytes: bytes,
        };
        let headers = |a: &Asset| {
            Ok(match a.relative.as_str() {
                "ordinary.safetensors" => vec![
                    header("time_embedding.0.weight", Dtype::BF16, 4),
                    header("layers.0.q.weight", Dtype::BF16, 4),
                ],
                "full.safetensors" => vec![
                    header("time_embedding.0.weight", Dtype::F32, 8),
                    header("layers.0.q.weight", Dtype::F32, 8),
                ],
                "text.safetensors" => vec![
                    header("language_model.embed_tokens.weight", Dtype::BF16, 4),
                    header("visual.unused", Dtype::BF16, 4),
                ],
                "vae.safetensors" => vec![
                    header("decoder.conv.weight", Dtype::BF16, 4),
                    header("encoder.unused", Dtype::BF16, 4),
                ],
                "lora.safetensors" => vec![header("lora.weight", Dtype::F32, 8)],
                "time.safetensors" => vec![header("time_embedding.0.weight", Dtype::F32, 8)],
                "dataspace.safetensors" => vec![header("Q", Dtype::F32, 8)],
                _ => unreachable!(),
            })
        };
        let off_before =
            selected_facts_with(&sources, MossImageAcceleration::Off, headers).unwrap();
        sources.optional = vec![
            asset(MossImageAssetRole::Transformer, "full.safetensors"),
            asset(MossImageAssetRole::DmdLora, "lora.safetensors"),
            asset(MossImageAssetRole::DmdTime, "time.safetensors"),
            asset(MossImageAssetRole::DmdDataspace, "dataspace.safetensors"),
        ];
        let off = selected_facts_with(&sources, MossImageAcceleration::Off, |a| {
            assert!(!sources
                .optional
                .iter()
                .any(|candidate| candidate.relative == a.relative));
            headers(a)
        })
        .unwrap();
        assert_eq!(off.base_bytes, off_before.base_bytes);
        assert_eq!(off.overlay_bytes, 0);
        assert_eq!(
            (
                off.transformer_bytes,
                off.conditioning_bytes,
                off.decoder_bytes
            ),
            (8, 4, 8)
        );
        let mixed = selected_facts_with(
            &sources,
            MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16),
            headers,
        )
        .unwrap();
        assert_eq!(mixed.base_bytes, off.base_bytes);
        assert_eq!(mixed.overlay_bytes, 8 + 8 + 4); // LoRA + Q/codes + island widening delta.
        let full = selected_facts_with(
            &sources,
            MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32),
            headers,
        )
        .unwrap();
        assert_eq!(full.transformer_bytes, 16);
        assert_eq!(full.base_bytes, 16 + 4 + 8);
        assert_eq!(full.overlay_bytes, 8); // Q/codes; no second LoRA or time replacement.
    }
    #[test]
    fn expanded_batch_context_branch_work_and_overflow_are_priced() {
        let a = request_workspace_bytes(256, 256, 1, 8, 1).unwrap();
        assert!(request_workspace_bytes(256, 256, 2, 8, 1).unwrap() > a);
        assert!(request_workspace_bytes(256, 256, 1, 4096, 1).unwrap() > a);
        assert!(request_workspace_bytes(256, 256, 1, 8, 3).unwrap() > a);
        assert!(
            request_workspace_bytes(u32::MAX - 15, u32::MAX - 15, usize::MAX, 4096, 3).is_err()
        );
    }
}
