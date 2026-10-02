//! Production compressed-KV policy for MLX generation (epic sc-20669, story sc-20679).
//!
//! The backend-neutral policy — the opt-in, the one qualification table, the fallback reasons and
//! the per-generation report — lives in [`core_llm::kv_compression`]. This module is its MLX half:
//! it maps a loaded decoder onto a table family, refuses geometry the fused reader does not
//! implement, builds the retained fused Metal reader once per provider, selects the request's cache
//! before any K/V mutation, and turns the cache's measured evidence into a
//! [`core_llm::KvCacheReport`].

use core_llm::{
    KvCacheCounters, KvCacheFallbackReason, KvCacheReport, KvCompressionFormat, KvModelFamily,
    KV_CACHE_FORMAT_VERSION,
};

use crate::config::ModelConfig;
use crate::models::CausalLm;
use crate::primitives::kv_cache::{CacheRoute, KvCache};
use crate::primitives::packed_group_affine_kv::{
    packed_metal_head_dimension_supported, DenseFallbackPackedDecoderCache,
    DenseTransitionAdmission,
};
use crate::primitives::{
    packed_metal_identity, CompiledKernelHandle, PackedCodeBits, PackedMetalGpuFamily,
    PackedMetalKernel, PACKED_METAL_QUANT_GROUP_SIZE,
};

/// The packed code width implementing `format`. Every advertised format has one.
pub(crate) const fn packed_code_bits(format: KvCompressionFormat) -> PackedCodeBits {
    match format {
        KvCompressionFormat::GroupAffineK8V8 => PackedCodeBits::Eight,
    }
}

/// How the MLX packed cache allocates and reads, for request admission (sc-20682):
///
/// * the packed arrays grow by [`KV_BLOCK_TOKENS`](crate::primitives::kv_cache::KV_BLOCK_TOKENS)
///   positions (rounded up to whole groups);
/// * a step holds at most one per-layer packed transient per in-flight evaluator buffer
///   ([`MLX_EVAL_BUFFER_WINDOW`](crate::provider::MLX_EVAL_BUFFER_WINDOW)): the prompt step's
///   flush output before its in-place write, or a decode step's pre-growth arrays beside their
///   successors;
/// * a product generation's reader dispatches are one-token decode steps (`query_heads` rows):
///   its prompt step runs on an empty cache, whose multi-row steps attend their fresh K/V through
///   dense SDPA (priced by the prefill attention term), and every qualified prompt is far longer
///   than the fused-SDPA row limit.
pub(crate) fn mlx_compressed_allocation(
    query_heads: u64,
    head_dim: u64,
) -> Option<core_llm::CompressedKvAllocation> {
    Some(core_llm::CompressedKvAllocation {
        growth_block_tokens: u64::try_from(crate::primitives::kv_cache::KV_BLOCK_TOKENS).ok()?,
        coexisting_layer_transients: crate::provider::MLX_EVAL_BUFFER_WINDOW,
        reader_scratch_bytes_per_layer:
            crate::primitives::packed_metal::packed_reader_partial_scratch_bound_bytes(
                query_heads,
                head_dim,
            )?,
    })
}

/// K/V bytes admission prices for a generation of `total_tokens` positions (prompt plus every
/// generated token) on the compressed cache of `format`: the format-derived packed representation
/// plus the MLX allocation's transients ([`core_llm::compressed_kv_cache_bytes`]).
pub(crate) fn compressed_request_kv_bytes(
    format: KvCompressionFormat,
    geometry: &core_llm::LlmMemoryGeometry,
    total_tokens: u64,
) -> Option<u64> {
    core_llm::compressed_kv_cache_bytes(
        format,
        geometry.kv_shape(),
        total_tokens,
        mlx_compressed_allocation(geometry.query_heads, geometry.head_dim)?,
    )
}

/// The table family of a provider's campaign identity (`"llama"` / `"qwen"`): the same
/// architecture check that names the SC-20671 evidence models, so a qualification row only ever
/// applies to the decoder family it was measured on.
pub(crate) fn family_for(campaign_family: Option<&str>) -> Option<KvModelFamily> {
    match campaign_family? {
        "llama" => Some(KvModelFamily::Llama),
        "qwen" => Some(KvModelFamily::Qwen3),
        _ => None,
    }
}

/// Why the fused reader cannot serve this decoder's attention, before any cache is built.
pub(crate) fn geometry_refusal(cfg: &ModelConfig) -> Option<String> {
    let head_dim = usize::try_from(cfg.head_dim).unwrap_or(0);
    if !packed_metal_head_dimension_supported(head_dim) {
        return Some(format!(
            "the fused compressed reader supports head dimension 64, 128 or 256, not {}",
            cfg.head_dim
        ));
    }
    if cfg.attn_logit_softcap.is_some() {
        return Some("attention-score soft-cap needs tanh before softmax".into());
    }
    if cfg
        .query_pre_attn_scalar
        .is_some_and(|scalar| scalar != cfg.head_dim)
    {
        return Some(
            "the fused reader scales scores by the inverse square-root head dimension".into(),
        );
    }
    if cfg.mla.is_some() || cfg.gemma4.is_some() {
        return Some("latent or shared K/V attention has no compressed-domain reader".into());
    }
    None
}

/// Build the retained fused group-affine Metal reader for `bits` (the decode kernels are compiled
/// on their first dispatch). Shared by production generation and the SC-20671 campaign arm.
pub(crate) fn group_affine_reader(bits: PackedCodeBits) -> Result<CompiledKernelHandle, String> {
    let kernel = PackedMetalKernel::for_identity_family_and_bits(
        packed_metal_identity(bits),
        // MLX's Metal backend runs only on Apple silicon, whose oldest Mac GPU (M1) is Apple
        // family 7, so every device that can dispatch this reader takes the recent-family profile.
        PackedMetalGpuFamily::Apple7OrNewer,
        bits,
    )
    .map_err(|e| e.to_string())?;
    // A provider (and a campaign worker) is single-threaded: this non-Send Metal object is owned
    // through the cache handle's `Arc` and never crosses a thread.
    #[allow(clippy::arc_with_non_send_sync)]
    let kernel = std::sync::Arc::new(kernel);
    Ok(CompiledKernelHandle::new(kernel))
}

/// The request's compressed cache, chosen before any K/V mutation: the packed group-affine cache
/// of `reader`'s width bound to `reader`, for one sequence prefilling `prompt_tokens` tokens with
/// the implicit causal mask. `Err` carries the selection's refusal with the dense cache that then
/// serves the request.
pub(crate) fn select_compressed_cache(
    model: &CausalLm,
    reader: CompiledKernelHandle,
    prompt_tokens: usize,
    transition_admission: DenseTransitionAdmission,
) -> (Box<dyn KvCache>, Option<String>) {
    let selection = model.select_cache_with_packed_reader(reader, 1, prompt_tokens, false);
    let refused = match selection.route() {
        CacheRoute::DenseFallback { reason } => Some(reason.clone()),
        CacheRoute::ExperimentalPacked => None,
    };
    let mut cache = selection.into_cache();
    // The request was admitted at the compressed price: a later dense transition is admitted
    // against fresh memory before it reconstructs the history (sc-20682).
    if let Some(packed) = cache
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
    {
        packed.set_dense_transition_admission(transition_admission);
    }
    (cache, refused)
}

/// The dense-transition admission of a compressed product generation: the transition's own
/// reconstruction bytes, or — whichever is larger — the dense cache the rest of the generation
/// grows to (`dense_final_bytes`), must fit the fresh memory budget. A refusal is the request's
/// typed [`core_llm::RequestResourceExhausted`].
pub(crate) fn dense_transition_admission(
    dense_final_bytes: u64,
    refusal: core_llm::RequestResourceExhausted,
) -> DenseTransitionAdmission {
    DenseTransitionAdmission::new(move |reconstruction| {
        let required = reconstruction.max(dense_final_bytes);
        let available = core_llm::effective_memory_budget(
            core_llm::available_host_memory_bytes(),
            core_llm::operational_memory_override()
                .map_err(|e| crate::error::Error::Msg(e.to_string()))?,
        )
        .map_err(|e| crate::error::Error::Msg(e.to_string()))?;
        if required > available {
            return Err(crate::error::Error::ResourceExhausted(
                core_llm::RequestResourceExhausted {
                    required_bytes: required,
                    available_bytes: available,
                    ..refusal
                },
            ));
        }
        Ok(())
    })
}

/// Dense K/V bytes a generation that turned dense grows to by its last position: every layer's
/// buffer at the dense cache's block-rounded capacity of `total_tokens`, plus one layer's buffer
/// again for a block growth's pre-growth copy beside its successor.
pub(crate) fn dense_final_kv_bytes(
    shape: core_llm::KvCacheShape,
    total_tokens: u64,
) -> Option<u64> {
    let block = u64::try_from(crate::primitives::kv_cache::KV_BLOCK_TOKENS).ok()?;
    let capacity = total_tokens.div_ceil(block).max(1).checked_mul(block)?;
    let dense = shape.dense_bytes(capacity)?;
    dense.checked_add(dense.checked_div(shape.layers.max(1))?)
}

/// The report of a generation that ran on a cache [`select_compressed_cache`] chose, read from the
/// cache's own evidence before it is reset. A refused selection ran dense from the start; any
/// explicit dense transition, recorded fallback or full-cache reconstruction afterwards makes it a
/// [`KvCacheFallbackReason::RuntimeFallback`].
pub(crate) fn compressed_report(
    format: KvCompressionFormat,
    refused: Option<String>,
    cache: &dyn KvCache,
) -> crate::error::Result<KvCacheReport> {
    if let Some(reason) = refused {
        return Ok(KvCacheReport::dense(
            KvCacheFallbackReason::ReaderUnavailable,
            Some(reason),
        ));
    }
    let Some(evidence) = cache.packed_evidence() else {
        return Ok(KvCacheReport::dense(
            KvCacheFallbackReason::ReaderUnavailable,
            Some("the selected cache exported no compressed evidence".into()),
        ));
    };
    if evidence.bits != packed_code_bits(format).bits()
        || evidence.quantization_group_size != PACKED_METAL_QUANT_GROUP_SIZE
        || evidence.quantization_group_size != format.group_size()
    {
        return Err(crate::error::Error::Msg(format!(
            "compressed cache ran {}-bit group {} codes, not {}",
            evidence.bits,
            evidence.quantization_group_size,
            format.id()
        )));
    }
    let compressed_cache_bytes = cache
        .compressed_storage()?
        .map_or(0, |storage| storage.device_bytes());
    let counters = KvCacheCounters {
        fused_attention_calls: u64::try_from(evidence.accepted_direct_calls).unwrap_or(u64::MAX),
        dense_fallback_events: u64::try_from(evidence.fallback_reasons.len()).unwrap_or(u64::MAX),
        full_cache_dequantizations: u64::try_from(evidence.full_cache_dequantizations)
            .unwrap_or(u64::MAX),
        compressed_cache_bytes,
    };
    let fell_back = evidence.dense_active
        || !evidence.fallback_reasons.is_empty()
        || evidence.full_cache_dequantizations > 0;
    let detail = fell_back.then(|| {
        evidence
            .fallback_reasons
            .iter()
            .map(|(operation, reason)| format!("{operation}: {reason}"))
            .collect::<Vec<_>>()
            .join("; ")
    });
    Ok(KvCacheReport {
        format_version: KV_CACHE_FORMAT_VERSION,
        format: Some(format),
        fallback: fell_back.then_some(KvCacheFallbackReason::RuntimeFallback),
        detail: detail.filter(|detail| !detail.is_empty()),
        counters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::campaign::{context_band_target, LLAMA_CANDIDATE, QWEN_CANDIDATE};
    use core_llm::{qualify_kv_compression, KvCompressionPolicy, KV_COMPRESSION_QUALIFICATIONS};

    fn row(family: KvModelFamily) -> &'static core_llm::KvQualification {
        KV_COMPRESSION_QUALIFICATIONS
            .iter()
            .find(|row| row.family == family)
            .unwrap()
    }

    /// The table's thresholds are the SC-20671 campaign coordinates of the evidence models, and
    /// each evidence model's campaign family names the table family.
    #[test]
    fn qualification_thresholds_are_the_campaign_coordinates() {
        let band = |spec: &crate::campaign::BenchmarkModelSpec, name| {
            context_band_target(spec.native_context_tokens, name).unwrap()
        };
        let qwen = row(KvModelFamily::Qwen3);
        assert_eq!(family_for(Some(QWEN_CANDIDATE.family)), Some(qwen.family));
        assert_eq!(
            qwen.min_context_tokens,
            band(&QWEN_CANDIDATE, "memory-material")
        );
        assert!(qwen.admits(band(&QWEN_CANDIDATE, "fit-boundary")));
        assert!(!qwen.admits(band(&QWEN_CANDIDATE, "medium")));

        let llama = row(KvModelFamily::Llama);
        assert_eq!(family_for(Some(LLAMA_CANDIDATE.family)), Some(llama.family));
        assert_eq!(
            llama.min_context_tokens,
            band(&LLAMA_CANDIDATE, "memory-material")
        );
        // Llama's fit-boundary coordinate is pending its noise-floor run: the row stops there.
        assert_eq!(
            llama.max_context_tokens,
            Some(band(&LLAMA_CANDIDATE, "fit-boundary"))
        );
        assert!(!llama.admits(band(&LLAMA_CANDIDATE, "medium")));
        for (spec, short) in [(&QWEN_CANDIDATE, "short"), (&LLAMA_CANDIDATE, "short")] {
            assert_eq!(
                qualify_kv_compression(
                    KvCompressionPolicy::Qualified,
                    family_for(Some(spec.family)),
                    band(spec, short),
                    1,
                ),
                Err(KvCacheFallbackReason::BelowMinimumContext)
            );
        }
    }

    /// The campaign arms build their reader here too: every arm's receipt identity and GPU-family
    /// profile must be the ones this builder uses.
    #[test]
    fn campaign_arms_record_the_reader_this_builder_makes() {
        for method in crate::campaign::CompressedKvMethod::ALL {
            assert_eq!(
                method.representation_identity(),
                packed_metal_identity(method.code_bits())
            );
            assert_eq!(
                method.kernel_gpu_family(),
                PackedMetalGpuFamily::Apple7OrNewer
            );
        }
    }

    #[test]
    fn only_the_two_evidence_families_map_onto_the_table() {
        assert_eq!(family_for(Some("llama")), Some(KvModelFamily::Llama));
        assert_eq!(family_for(Some("qwen")), Some(KvModelFamily::Qwen3));
        assert_eq!(family_for(Some("gemma")), None);
        assert_eq!(family_for(None), None);
    }

    /// Every advertised format is the 8-bit, group-32 packed cache the fused reader reads.
    #[test]
    fn the_advertised_format_is_the_packed_k8v8_reader_layout() {
        let format = KvCompressionFormat::GroupAffineK8V8;
        assert_eq!(packed_code_bits(format), PackedCodeBits::Eight);
        assert_eq!(packed_code_bits(format).bits(), format.key_bits());
        assert_eq!(packed_code_bits(format).bits(), format.value_bits());
        assert_eq!(format.group_size(), PACKED_METAL_QUANT_GROUP_SIZE);
    }

    #[test]
    fn geometry_outside_the_fused_reader_is_refused_before_selection() {
        let mut cfg = ModelConfig::from_json(&serde_json::json!({
            "architectures": ["LlamaForCausalLM"], "hidden_size": 128,
            "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 64, "vocab_size": 32, "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0, "tie_word_embeddings": false,
        }))
        .unwrap();
        assert_eq!(geometry_refusal(&cfg), None);
        cfg.head_dim = 96;
        assert!(geometry_refusal(&cfg).unwrap().contains("head dimension"));
        cfg.head_dim = 128;
        assert_eq!(geometry_refusal(&cfg), None);
        cfg.attn_logit_softcap = Some(50.0);
        assert!(geometry_refusal(&cfg).unwrap().contains("soft-cap"));
        cfg.attn_logit_softcap = None;
        cfg.query_pre_attn_scalar = Some(256);
        assert!(geometry_refusal(&cfg).unwrap().contains("square-root"));
        cfg.query_pre_attn_scalar = Some(128);
        assert_eq!(geometry_refusal(&cfg), None);
    }

    /// Llama-3.2-3B decoder geometry at bf16 (24 query heads, 8 KV heads of 128, 28 layers).
    fn llama_3b_geometry() -> core_llm::LlmMemoryGeometry {
        core_llm::LlmMemoryGeometry {
            query_heads: 24,
            kv_heads: 8,
            head_dim: 128,
            layers: 28,
            element_bytes: 2,
            score_element_bytes: 4,
            hidden_size: 3072,
            intermediate_size: 8192,
            vocab_size: 128_256,
            recurrent_bytes: 0,
        }
    }

    /// sc-20682: admission prices the MLX cache's own layout — 256-token growth blocks, one
    /// per-layer transient per in-flight evaluator buffer, and a one-token decode dispatch's
    /// split-KV partials (at most 128 splits of `D + 2` f32 per query head) for every layer.
    #[test]
    fn admission_prices_the_mlx_compressed_layout() {
        let geometry = llama_3b_geometry();
        assert_eq!(
            crate::primitives::packed_metal::packed_reader_partial_scratch_bound_bytes(24, 128),
            Some(24 * 128 * 130 * 4)
        );
        assert_eq!(
            mlx_compressed_allocation(24, 128),
            Some(core_llm::CompressedKvAllocation {
                growth_block_tokens: 256,
                coexisting_layer_transients: 11,
                reader_scratch_bytes_per_layer: 24 * 128 * 130 * 4,
            })
        );
        let format = KvCompressionFormat::GroupAffineK8V8;
        for total in [10_240 + 512, 32_768 + 1_024, 130_559] {
            let priced = compressed_request_kv_bytes(format, &geometry, total).unwrap();
            assert_eq!(
                Some(priced),
                core_llm::compressed_kv_cache_bytes(
                    format,
                    geometry.kv_shape(),
                    total,
                    mlx_compressed_allocation(24, 128).unwrap()
                )
            );
            assert!(priced < geometry.kv_shape().dense_bytes(total).unwrap());
        }
    }

    /// A generation that turns dense part-way grows to its block-rounded dense cache, plus one
    /// layer's pre-growth copy.
    #[test]
    fn dense_final_bytes_cover_block_capacity_and_one_growth_copy() {
        let shape = core_llm::KvCacheShape {
            layers: 2,
            kv_heads: 1,
            head_dim: 64,
            element_bytes: 2,
        };
        let layer = 512 * 64 * 2 * 2;
        assert_eq!(dense_final_kv_bytes(shape, 300), Some(2 * layer + layer));
        assert_eq!(dense_final_kv_bytes(shape, 512), Some(3 * layer));
        assert_eq!(dense_final_kv_bytes(shape, u64::MAX), None);
    }

    /// The dense-transition admission refuses — typed, with the request's own geometry — when the
    /// larger of the reconstruction and the final dense cache exceeds the fresh budget.
    #[test]
    fn a_dense_transition_is_admitted_against_the_fresh_budget() {
        struct Budget;
        impl Drop for Budget {
            fn drop(&mut self) {
                std::env::remove_var(core_llm::AVAILABLE_MEMORY_OVERRIDE);
            }
        }
        let _budget = Budget;
        std::env::set_var(core_llm::AVAILABLE_MEMORY_OVERRIDE, "1000");
        let refusal = core_llm::RequestResourceExhausted {
            prompt_tokens: 7,
            max_new_tokens: 3,
            max_context_tokens: 64,
            required_bytes: 0,
            available_bytes: 0,
        };
        let check = |final_bytes, reconstruction| {
            dense_transition_admission(final_bytes, refusal).admit(reconstruction)
        };
        assert!(check(500, 400).is_ok());
        assert!(check(1000, 1000).is_ok());
        for (final_bytes, reconstruction, required) in [(500, 2_000, 2_000), (5_000, 10, 5_000)] {
            match check(final_bytes, reconstruction) {
                Err(crate::error::Error::ResourceExhausted(evidence)) => assert_eq!(
                    evidence,
                    core_llm::RequestResourceExhausted {
                        required_bytes: required,
                        available_bytes: 1_000,
                        ..refusal
                    }
                ),
                other => panic!("expected a typed refusal, got {other:?}"),
            }
        }
    }
}
