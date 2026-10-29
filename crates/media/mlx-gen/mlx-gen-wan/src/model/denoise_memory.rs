//! Denoise residency facts the memory strategies and the pre-flight guard share.

use super::*;

/// Estimated resident transformer bytes for the sc-4986 [`preflight_denoise_memory_guard`]: the
/// on-disk weight file size(s) scaled by the load-time quantization. A dense bf16 snapshot shrinks to
/// ~Q8/Q4 when `spec.quantize` runs at load; a pre-quantized snapshot has `quant == None` and its
/// files are already packed (ratio 1.0). The 14B MoE passes both expert files (both stay resident).
/// A missing file → 0 bytes for that file, so the guard under-counts rather than spuriously firing —
/// the real "snapshot incomplete" error then surfaces at the actual `Weights::from_file` load.
///
/// The two scaling arms are **mutually exclusive**, so the on-disk size is never double-discounted: a
/// load is *either* a dense bf16 snapshot with `spec.quantize` set (the bf16 file size is scaled down
/// by `ratio`) *or* a pre-packed Q4/Q8 snapshot with `quant == None` (the file is already the final
/// packed size, `ratio == 1.0`) — never a packed file scaled by a quant ratio again.
///
/// A path may also be a **shard directory** (the VACE lanes load `transformer/` /`transformer_2/`
/// diffusers shard dirs, sc-12459): its contents are summed under exactly the `Weights::from_dir`
/// selection — non-recursive, `*.safetensors` only, hidden entries (macOS AppleDouble
/// `._model.safetensors` sidecars) skipped via [`is_hidden_file`] — so the sizing matches what the
/// loader will actually map. Each entry is sized through `std::fs::metadata`, which **traverses
/// symlinks**: linked snapshots (the `assemble_wan_vace_*` `link == true` layout, convert.rs — the
/// default for local dev / the gated e2e) and HF-hub caches store every shard as a symlink into
/// `blobs/`, and sizing the link itself (`DirEntry::metadata`, ~dozens of bytes) would make the
/// preflight see none of the 27–54 GB expert weights and wave through requests that then die by
/// SIGKILL mid-load — the original F-008 failure mode. `pub(crate)` so the sc-12459 VACE preflight
/// reuses it.
///
/// [`is_hidden_file`]: mlx_gen::gen_core::weightsmeta::is_hidden_file
/// SC-20686 estimate-plus-reserve admission: exactly what a dense Wan route's generate-time fit gate
/// ([`preflight_denoise_memory_guard`]) prices for `req`, plus the output geometry its decode runs
/// at, resolved from the load spec, the snapshot's `config.json`/file sizes and the request alone --
/// no weights are loaded and MLX is never touched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DenoiseFacts {
    pub(crate) resident_bytes: u64,
    /// The dense (pre-packing) bytes of the largest DiT expert a load-time quantization builds
    /// before packing it -- live beside its packs while they materialize; 0 for a packed tier.
    pub(crate) load_transient_bytes: u64,
    pub(crate) tokens: usize,
    pub(crate) dim: usize,
    pub(crate) cfg_batched: bool,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) frames: u32,
}

/// [`DenoiseFacts`] of the TI2V-5B and the two A14B routes, mirroring their `generate_impl`.
pub(crate) fn dense_denoise_facts(
    route: &str,
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> Result<DenoiseFacts> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(Error::Msg(format!(
            "{route}: expected a model directory for the admission estimate"
        )));
    };
    let cfg = WanModelConfig::from_model_dir(root)?;
    let quant = resolve_load_time_quant(route, &cfg, spec.quantize)?;
    let adapter_mode = if cfg.quantization.is_some() {
        AdapterResidencyMode::Additive
    } else {
        AdapterResidencyMode::Folded
    };
    let frames = req.frames.map(|f| f as usize).unwrap_or(cfg.frame_num);
    let trim = req.trim_first_frames.unwrap_or(0) as usize;
    let unsized_adapters = || {
        Error::Msg(format!(
            "{route}: cannot size every additive adapter for the estimate"
        ))
    };
    let dense = |file: &str| {
        if quant.is_some() {
            dit_resident_bytes(&[root.join(file)], None)
        } else {
            0
        }
    };
    let (vae, resident_bytes, cfg_batched, load_transient_bytes) = match route {
        MODEL_ID => {
            let adapters = adapter_stack_resident_bytes(&spec.adapters, adapter_mode)
                .ok_or_else(unsized_adapters)?;
            let guidance = cfg.sample_guide_scale.resolve_single(req.guidance);
            (
                Ti2vProviderVae::VAE_TILING,
                dit_resident_bytes(&[root.join("model.safetensors")], quant)
                    .saturating_add(adapters),
                guidance > 1.0,
                dense("model.safetensors"),
            )
        }
        MODEL_ID_T2V_14B | MODEL_ID_I2V_14B => {
            let low = dit_resident_bytes(&[root.join("low_noise_model.safetensors")], quant);
            let high = dit_resident_bytes(&[root.join("high_noise_model.safetensors")], quant);
            let (low_adapters, high_adapters) =
                wan14b_adapter_bytes_per_expert(&spec.adapters, adapter_mode)
                    .ok_or_else(unsized_adapters)?;
            (
                A14bProviderVae::VAE_TILING,
                wan14b_denoise_resident_bytes(
                    spec.offload_policy,
                    req.sampler.as_deref(),
                    low.saturating_add(low_adapters),
                    high.saturating_add(high_adapters),
                ),
                true,
                dense("low_noise_model.safetensors").max(dense("high_noise_model.safetensors")),
            )
        }
        other => {
            return Err(Error::Msg(format!(
                "{other}: not a dense Wan route for the admission estimate"
            )))
        }
    };
    let stride = provider_vae_stride(vae);
    let gen_frames = frames + trim * stride.0;
    let (width, height) = resolve_capped_dims(req, &cfg, vae);
    let latent = latent_shape(gen_frames, height, width, cfg.vae_z_dim, stride)?;
    Ok(DenoiseFacts {
        resident_bytes,
        load_transient_bytes,
        tokens: seq_len(latent, cfg.patch_size),
        dim: cfg.dim,
        cfg_batched,
        width,
        height,
        frames: u32::try_from(gen_frames)
            .map_err(|_| Error::Msg(format!("{route}: frame count overflows")))?,
    })
}

pub(crate) fn dit_resident_bytes(files: &[PathBuf], quant: Option<Quant>) -> u64 {
    fn weight_bytes_at(p: &std::path::Path) -> u64 {
        match std::fs::metadata(p) {
            Ok(m) if m.is_dir() => std::fs::read_dir(p)
                .map(|rd| {
                    rd.flatten()
                        .map(|e| e.path())
                        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
                        .filter(|p| !mlx_gen::gen_core::weightsmeta::is_hidden_file(p))
                        // `std::fs::metadata`, NOT `DirEntry::metadata`: the latter does not
                        // traverse symlinks, and linked snapshots / HF-hub caches store shards
                        // as symlinks into blobs/ — the link's own length is ~14 bytes.
                        .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
                        .sum()
                })
                .unwrap_or(0),
            Ok(m) => m.len(),
            Err(_) => 0,
        }
    }
    let raw: u64 = files.iter().map(|p| weight_bytes_at(p)).sum();
    (raw as f64 * quant_resident_ratio(quant)) as u64
}

/// The resident fraction of a bf16 weight surface after load-time quantization.
pub(crate) fn quant_resident_ratio(quant: Option<Quant>) -> f64 {
    match quant.map(|q| q.bits()) {
        Some(4) => 0.30, // 4-bit affine: ~0.5 B/param + scales vs bf16 2 B/param
        Some(8) => 0.55, // 8-bit affine: ~1 B/param + scales
        _ => 1.0,
    }
}

/// The resident transformer bytes the sc-4986 [`preflight_denoise_memory_guard`] must budget for the
/// **dual-expert MoE denoise**, made policy + sampler aware (sc-12736). `low_bytes` / `high_bytes` are the two
/// experts' resident sizes ([`dit_resident_bytes`] per expert).
///
/// * **Expert swap** — `Sequential` with a native solver keeps only the ACTIVE expert resident, so
///   budget ONE expert (the larger of the two, conservatively). Without this the guard sums both and
///   **false-rejects** the swappable default job on exactly the one-expert-sized machines the worker
///   selects `Sequential` for — defeating the whole Pillar-1 win at the gate.
/// * **Both resident** — `Resident`, or a curated solver on callers without per-eval swapping, keeps
///   BOTH resident. Summing both makes the guard correctly
///   **reject** a both-resident job that won't fit — a catchable error before the load, not a SIGKILL
///   mid-denoise.
pub(crate) fn moe_denoise_resident_bytes(
    offload: OffloadPolicy,
    sampler: Option<&str>,
    low_bytes: u64,
    high_bytes: u64,
) -> u64 {
    let swaps = offload == OffloadPolicy::Sequential && !is_wan_curated(sampler);
    if swaps {
        // One resident expert (the swap). Take the larger so the guard never under-budgets.
        low_bytes.max(high_bytes)
    } else {
        // Both experts co-resident for the whole denoise.
        low_bytes + high_bytes
    }
}

/// Wan T2V/I2V A14B residency after sc-12795: both native and curated paths swap under
/// `Sequential`, while `Resident` retains both experts. VACE continues using
/// [`moe_denoise_resident_bytes`] because its curated path has not adopted per-eval swapping.
pub(super) fn wan14b_denoise_resident_bytes(
    offload: OffloadPolicy,
    _sampler: Option<&str>,
    low_bytes: u64,
    high_bytes: u64,
) -> u64 {
    if offload == OffloadPolicy::Sequential {
        low_bytes.max(high_bytes)
    } else {
        low_bytes.saturating_add(high_bytes)
    }
}

/// Load-exact adapter residency for each Wan A14B expert. Packed Q4/Q8 snapshots retain additive
/// residuals; dense snapshots fold factors and therefore add no independent bytes. A shared adapter
/// is installed once on each expert, while an expert-tagged adapter contributes only to its target.
/// `None` keeps the preflight fail-closed when any additive source cannot be sized.
pub(super) fn wan14b_adapter_bytes_per_expert(
    adapters: &[AdapterSpec],
    mode: AdapterResidencyMode,
) -> Option<(u64, u64)> {
    if adapters.is_empty() || mode == AdapterResidencyMode::Folded {
        return Some((0, 0));
    }
    adapters
        .iter()
        .try_fold((0_u64, 0_u64), |(low, high), adapter| {
            let bytes = mlx_gen::gen_core::weightsmeta::safetensors_path_bytes(&adapter.path);
            if bytes == 0 {
                return None;
            }
            Some(match adapter.moe_expert {
                None => (low.saturating_add(bytes), high.saturating_add(bytes)),
                Some(MoeExpert::Low) => (low.saturating_add(bytes), high),
                Some(MoeExpert::High) => (low, high.saturating_add(bytes)),
            })
        })
}
