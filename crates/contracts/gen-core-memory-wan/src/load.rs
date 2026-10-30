//! Load-spec preparation: text-encoder bytes, adapter receipts and the Lightning recipe.

use super::*;

/// The width `mlx_gen_wan::model::effective_te_quant` floors the UMT5 encoder to whenever the DiT
/// is on an MLX-affine quantized tier. It is deliberately **not** the DiT's own width: Q4 costs the
/// encoder measurable prompt fidelity, so the encoder floors at Q8 for a Q4 DiT as well.
pub(super) const MLX_TE_QUANT_BITS: u64 = 8;

/// MLX affine group size, the same 64 [`packed_transformer_bytes`] assumes.
pub(super) const MLX_GROUP: u64 = 64;

/// Bytes the MLX loader materializes for the UMT5 text encoder — packed to `packed_bits` on a
/// quantized tier, dense bf16 projections on the dense tier (`None`).
///
/// `mlx_gen_wan::text_encoder` packs exactly the seven bias-less projections of each block —
/// `attn.{q,k,v,o}` and `ffn.{gate_proj,fc1,fc2}` — into an MLX affine triple on a quantized tier,
/// and leaves everything else dense. The families are priced separately because they materialize at
/// different widths, and the f32 family does so **on both tiers** (`Umt5Block::from_weights` and
/// `assemble` upcast unconditionally, before `quantize` runs):
///
/// * a packed `[out, in]` projection becomes `out * in * bits / 8` code bytes plus a `scales` and a
///   `biases` plane of `out * in / group` elements each, at the source tensor's own width; on the
///   dense tier the same projection is the shared `quant::lin` dense arm — its stored bf16 bytes;
/// * the per-block norms, the position-bias tables and the final `norm.weight` are upcast to **f32**
///   by the loader (`as_dtype(Dtype::Float32)`), so they are priced at 4 bytes per element rather
///   than at their stored bf16 width — on the dense tier as much as on a packed one;
/// * `token_embedding.weight` is cloned as stored — the loader never casts it — so it contributes
///   its stored bytes.
///
/// On a quantized tier this fails closed on any key outside those families, and on a packed
/// projection whose input width is not a whole number of groups: either means the encoder layout
/// moved and this pricing no longer describes what loads. The dense tier has no packing geometry to
/// get wrong, so a leaf outside the families keeps the `projected_dense_bytes(path, 2)` treatment
/// it always had (float at 2 B per element; packed/non-float storage refused).
pub(super) fn mlx_text_encoder_bytes(
    path: &Path,
    packed_bits: Option<u64>,
) -> gen_core::Result<u64> {
    const PACKED_LEAVES: [&str; 7] = [
        ".attn.q.weight",
        ".attn.k.weight",
        ".attn.v.weight",
        ".attn.o.weight",
        ".ffn.gate_proj.weight",
        ".ffn.fc1.weight",
        ".ffn.fc2.weight",
    ];
    const F32_LEAVES: [&str; 3] = [
        ".norm1.weight",
        ".norm2.weight",
        ".pos_embedding.embedding.weight",
    ];

    let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
    if headers.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{} contains no tensors",
            path.display()
        )));
    }
    headers.iter().try_fold(0_u64, |sum, header| {
        let name = header.name.as_str();
        let projection =
            name.starts_with("blocks.") && PACKED_LEAVES.iter().any(|leaf| name.ends_with(leaf));
        let bytes = if let (true, Some(bits)) = (projection, packed_bits) {
            let [out, input] = header.shape[..] else {
                return Err(gen_core::Error::Unsupported(format!(
                    "{} packed projection {name} is not a 2-D weight",
                    path.display()
                )));
            };
            let (out, input) = (out as u64, input as u64);
            if input == 0 || !input.is_multiple_of(MLX_GROUP) {
                return Err(gen_core::Error::Unsupported(format!(
                    "{} projection {name} has {input} input columns, not a whole number of \
                     {MLX_GROUP}-wide affine groups",
                    path.display()
                )));
            }
            let elements = tensor_elements(path, header)?;
            let codes = elements
                .checked_mul(bits)
                .and_then(|bits| bits.checked_div(8))
                .ok_or_else(|| {
                    gen_core::Error::Msg("Wan packed encoder code bytes overflow".to_owned())
                })?;
            // One `scales` and one `biases` plane, each `[out, input / group]` at the source width.
            let planes = out
                .checked_mul(input / MLX_GROUP)
                .and_then(|entries| entries.checked_mul(2))
                .and_then(|entries| entries.checked_mul(header.dtype.size() as u64))
                .ok_or_else(|| {
                    gen_core::Error::Msg("Wan packed encoder scale bytes overflow".to_owned())
                })?;
            codes.checked_add(planes).ok_or_else(|| {
                gen_core::Error::Msg("Wan packed encoder projection bytes overflow".to_owned())
            })?
        } else if projection {
            // The dense tier: `quant::lin`'s dense arm loads the projection as stored (bf16).
            header.data_bytes
        } else if name == "norm.weight"
            || (name.starts_with("blocks.") && F32_LEAVES.iter().any(|leaf| name.ends_with(leaf)))
        {
            tensor_elements(path, header)?
                .checked_mul(4)
                .ok_or_else(|| {
                    gen_core::Error::Msg("Wan encoder f32 tensor bytes overflow".to_owned())
                })?
        } else if name == "token_embedding.weight" {
            header.data_bytes
        } else if packed_bits.is_none() {
            // The dense tier has no packing geometry to get wrong, so a leaf outside the named
            // families is what `projected_dense_bytes(path, 2)` always made of it: a float tensor
            // at 2 B per element, and a refusal for packed/non-float storage.
            if !header.is_float() || name.ends_with(".scales") || name.ends_with(".biases") {
                return Err(gen_core::Error::Unsupported(format!(
                    "{} dense tensor {name} has unsupported packed/non-float storage",
                    path.display()
                )));
            }
            tensor_elements(path, header)?
                .checked_mul(2)
                .ok_or_else(|| {
                    gen_core::Error::Msg("Wan projected tensor bytes overflow".to_owned())
                })?
        } else {
            return Err(gen_core::Error::Unsupported(format!(
                "{} encoder tensor {name} is outside the packed UMT5 key surface",
                path.display()
            )));
        };
        sum.checked_add(bytes)
            .ok_or_else(|| gen_core::Error::Msg("Wan encoder byte total overflow".to_owned()))
    })
}

pub(super) fn adapter_receipts(
    spec: &LoadSpec,
    route: WanI2vRoute,
    tier: MemoryNumericTier,
) -> gen_core::Result<(Vec<WanAdapterReceipt>, String, u64, Vec<SealedFile>)> {
    let packed = tier.quant.is_some() && !matches!(route, WanI2vRoute::Vace | WanI2vRoute::VaceFun);
    let vace_quantized =
        tier.quant.is_some() && matches!(route, WanI2vRoute::Vace | WanI2vRoute::VaceFun);
    let mut receipts = Vec::new();
    let mut files = Vec::new();
    let mut identity = Sha256::new();
    identity.update(b"wan-adapters-v1");
    let mut overlay_bytes = 0_u64;
    for (ordinal, adapter) in spec.adapters.iter().enumerate() {
        if !adapter.scale.is_finite() || adapter.scale < 0.0 || adapter.pass_scales.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: adapter {ordinal} has unsupported scale/pass recipe",
                route.provider_id()
            )));
        }
        if matches!(route, WanI2vRoute::Ti2v5b | WanI2vRoute::Vace) && adapter.moe_expert.is_some()
        {
            return Err(gen_core::Error::Unsupported(
                "Wan TI2V-5B accepts shared adapters only".to_owned(),
            ));
        }
        let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(&adapter.path)?;
        if headers.is_empty() {
            return Err(gen_core::Error::Unsupported(format!(
                "adapter {} is empty",
                adapter.path.display()
            )));
        }
        let has_lokr_keys = gen_core::weightsmeta::keys_contain_lokr(
            headers.iter().map(|header| header.name.as_str()),
        );
        let has_loha_keys = gen_core::weightsmeta::keys_contain_loha(
            headers.iter().map(|header| header.name.as_str()),
        );
        if packed && has_loha_keys {
            return Err(gen_core::Error::Unsupported(
                "LoHa cannot be admitted on a packed Wan q4/q8 tier; select bf16".to_owned(),
            ));
        }
        let source_bytes = headers.iter().try_fold(0_u64, |sum, header| {
            sum.checked_add(header.data_bytes)
                .ok_or_else(|| gen_core::Error::Msg("Wan adapter byte total overflow".to_owned()))
        })?;
        if source_bytes == 0 {
            return Err(gen_core::Error::Unsupported(
                "Wan adapter has zero tensor bytes".to_owned(),
            ));
        }
        let multiplicity = if route.dual_expert() && adapter.moe_expert.is_none() {
            2
        } else {
            1
        };
        let persistent_bytes = if packed {
            source_bytes
                .checked_mul(multiplicity)
                .ok_or_else(|| gen_core::Error::Msg("Wan adapter residency overflow".to_owned()))?
        } else if vace_quantized {
            // The source factors coexist with the dense map during merge-before-quantize.
            source_bytes
        } else {
            0
        };
        overlay_bytes = overlay_bytes
            .checked_add(persistent_bytes)
            .ok_or_else(|| gen_core::Error::Msg("Wan overlay byte total overflow".to_owned()))?;
        let absolute = std::path::absolute(&adapter.path)?;
        let pin = spec
            .prepared_file_pins()
            .get(&absolute)
            .cloned()
            .unwrap_or(gen_core::PinnedWeightsFile::pin(&adapter.path)?);
        pin.ensure_unchanged()?;
        let digest = sha256_file(&adapter.path)?;
        let kind = if lightning_role(&adapter.path, route).is_some_and(|(_, exact)| exact) {
            "lightning_lora"
        } else if adapter.kind == AdapterKind::Lokr || has_lokr_keys {
            "lokr"
        } else if has_loha_keys {
            // Wan's execution adapters classify third-party LyCORIS LoHa from `hada_*` tensor
            // keys because the shared AdapterKind predates LoHa and has no LoHa variant. The
            // physical receipt must use that same classification rather than relabeling an
            // accepted dense LoHa file as LoRA.
            "loha"
        } else {
            "lora"
        };
        let expert = match adapter.moe_expert {
            None => "shared",
            Some(MoeExpert::High) => "high",
            Some(MoeExpert::Low) => "low",
        };
        let realization = if packed {
            "packed_additive_factors"
        } else if vace_quantized {
            "dense_folded_then_quantized"
        } else {
            "dense_folded"
        };
        let stability = "pinned_digest_and_tensor_geometry";
        identity.update((ordinal as u32).to_le_bytes());
        identity.update(absolute.to_string_lossy().as_bytes());
        identity.update(kind.as_bytes());
        identity.update(adapter.scale.to_bits().to_le_bytes());
        identity.update(0_u32.to_le_bytes());
        identity.update(expert.as_bytes());
        identity.update(digest.as_bytes());
        identity.update(source_bytes.to_le_bytes());
        identity.update(persistent_bytes.to_le_bytes());
        identity.update(realization.as_bytes());
        identity.update(stability.as_bytes());
        receipts.push(WanAdapterReceipt {
            ordinal: ordinal as u32,
            path: absolute.clone(),
            kind,
            scale_bits: adapter.scale.to_bits(),
            pass_scale_bits: Vec::new(),
            expert,
            digest: digest.clone(),
            source_bytes,
            persistent_bytes,
            realization,
            stability,
        });
        files.push(SealedFile {
            relative: format!("adapter:{ordinal}"),
            absolute,
            pin,
            digest,
            resident_bytes: source_bytes,
            tensor: true,
        });
    }
    validate_lightning_recipe(spec, route)?;
    Ok((
        receipts,
        format!("wan-adapters-v1:{:x}", identity.finalize()),
        overlay_bytes,
        files,
    ))
}

/// The Lightning distill subdirectory `lightx2v/Wan2.2-Lightning` ships for `route`.
///
/// The two A14B recipes are **not** cross-compatible (the manifest says so where it declares the
/// `coRequisite`), so the exact subdir is part of the recipe's identity: T2V is `Seko-V1.1`, I2V is
/// `Seko-V1`. A route with no Lightning recipe returns `None`.
pub(super) fn lightning_subdir(route: WanI2vRoute) -> Option<&'static str> {
    match route {
        WanI2vRoute::T2v14b => Some("Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1"),
        WanI2vRoute::I2v14b => Some("Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"),
        WanI2vRoute::Ti2v5b | WanI2vRoute::Vace | WanI2vRoute::VaceFun => None,
    }
}

pub(super) fn lightning_role(path: &Path, route: WanI2vRoute) -> Option<(MoeExpert, bool)> {
    let canonical = std::fs::canonicalize(path).ok()?;
    let marker = repo_dir(LIGHTNING_REPOSITORY);
    let parts = canonical
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let index = parts.iter().position(|part| part == &marker)?;
    if parts.get(index + 1).map(String::as_str) != Some("snapshots")
        || parts.get(index + 2).map(String::as_str) != Some(LIGHTNING_REVISION)
    {
        return Some((MoeExpert::High, false));
    }
    let suffix = parts[index + 3..].join("/");
    let Some(subdir) = lightning_subdir(route) else {
        return Some((MoeExpert::High, false));
    };
    if suffix == format!("{subdir}/high_noise_model.safetensors") {
        Some((MoeExpert::High, true))
    } else if suffix == format!("{subdir}/low_noise_model.safetensors") {
        Some((MoeExpert::Low, true))
    } else {
        Some((MoeExpert::High, false))
    }
}

pub(super) fn validate_lightning_recipe(
    spec: &LoadSpec,
    route: WanI2vRoute,
) -> gen_core::Result<()> {
    let roles = spec
        .adapters
        .iter()
        .enumerate()
        .filter_map(|(index, adapter)| {
            lightning_role(&adapter.path, route).map(|role| (index, adapter, role))
        })
        .collect::<Vec<_>>();
    if matches!(route, WanI2vRoute::Ti2v5b | WanI2vRoute::Vace) && !roles.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "TI2V-5B cannot use the A14B Lightning pair".to_owned(),
        ));
    }
    if roles.is_empty() {
        return Ok(());
    }
    let valid = roles.len() == 2
        && roles[0].0 == 0
        && roles[1].0 == 1
        && roles[0].2 == (MoeExpert::High, true)
        && roles[1].2 == (MoeExpert::Low, true)
        && roles[0].1.moe_expert == Some(MoeExpert::High)
        && roles[1].1.moe_expert == Some(MoeExpert::Low)
        && roles[0].1.kind == AdapterKind::Lora
        && roles[1].1.kind == AdapterKind::Lora
        && roles[0].1.scale.to_bits() == 1.0_f32.to_bits()
        && roles[1].1.scale.to_bits() == 1.0_f32.to_bits();
    if !valid {
        return Err(gen_core::Error::Unsupported(
            "Wan I2V Lightning must be the exact ordered high/low Seko-V1 pair at scale 1"
                .to_owned(),
        ));
    }
    Ok(())
}

pub(super) fn validate_load_spec<'a>(
    spec: &'a LoadSpec,
    provider_id: &str,
) -> gen_core::Result<&'a Path> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: directory weights required"
        )));
    };
    if spec.precision != Precision::Bf16
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
        || !spec.components.is_empty()
        || spec.resolved_route.as_deref() != Some(provider_id)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: load left the exact Wan I2V surface"
        )));
    }
    gen_core::reject_unknown_components(spec, &[], provider_id)?;
    spec.validate_load_shape_declaration()?;
    Ok(root)
}

pub fn prepare_load_spec(
    spec: &mut LoadSpec,
    backend: WanI2vBackend,
    provider_id: &str,
) -> gen_core::Result<()> {
    let route = WanI2vRoute::for_provider(provider_id)?;
    let tier = tier(spec, backend)?;
    // Wan-VACE constructors retain only metadata; components are built lazily for the request
    // and the Candle cache warms its component bundle on first render.  Its receipt must not
    // claim eager bulk materialization merely because the generic LoadSpec default is eager.
    if matches!(route, WanI2vRoute::Vace | WanI2vRoute::VaceFun) {
        *spec = spec.clone().with_applied_load_shape_declaration();
    }
    if backend == WanI2vBackend::Candle
        && ((route == WanI2vRoute::Vace && (tier.quant.is_some() || !spec.adapters.is_empty()))
            || (route == WanI2vRoute::VaceFun && tier.quant.is_some()))
    {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {} does not admit the selected numeric tier or load recipe",
            route.provider_id()
        )));
    }
    let root = validate_load_spec(spec, provider_id)?.to_path_buf();
    let (repository, revision) = repository_policy(backend, route, tier);
    if !matches!(route, WanI2vRoute::Vace | WanI2vRoute::VaceFun) {
        repository_revision(&root, repository, revision, tier, backend)?;
    } else if route == WanI2vRoute::VaceFun {
        vace_fun_source_receipt_revision(&root, backend)?;
    } else {
        vace_source_receipt_revision(&root, backend)?;
    }
    let inventory = match backend {
        WanI2vBackend::Mlx => validate_mlx_inventory(&root, route)?,
        WanI2vBackend::Candle => validate_candle_inventory(&root, route)?,
    };
    let mut paths = inventory
        .into_iter()
        .map(|relative| root.join(relative))
        .collect::<Vec<_>>();
    paths.extend(
        structural_files(&root, backend, route)?
            .into_iter()
            .map(|relative| root.join(relative)),
    );
    paths.extend(spec.adapters.iter().map(|adapter| adapter.path.clone()));
    spec.prepare_with_file_pins(
        paths
            .into_iter()
            .map(gen_core::PinnedWeightsFile::pin)
            .collect::<gen_core::Result<Vec<_>>>()?,
    )
}
