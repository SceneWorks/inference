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
use crate::primitives::packed_group_affine_kv::packed_metal_head_dimension_supported;
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

/// The campaign identity of a table family: the inverse of [`family_for`].
pub(crate) const fn campaign_family(family: KvModelFamily) -> &'static str {
    match family {
        KvModelFamily::Llama => "llama",
        KvModelFamily::Qwen3 => "qwen",
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
    // `qwen3_moe` parses as the Qwen3 architecture; the evidence measured dense decoders only.
    if cfg.moe.is_some() {
        return Some(
            "a mixture-of-experts decoder is outside the dense decoders the evidence measured"
                .into(),
        );
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
) -> (Box<dyn KvCache>, Option<String>) {
    let selection = model.select_cache_with_packed_reader(reader, 1, prompt_tokens, false);
    let refused = match selection.route() {
        CacheRoute::DenseFallback { reason } => Some(reason.clone()),
        CacheRoute::ExperimentalPacked => None,
    };
    (selection.into_cache(), refused)
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
        assert!(qwen.admits(band(&QWEN_CANDIDATE, "fit-boundary"), 0));
        assert!(!qwen.admits(band(&QWEN_CANDIDATE, "medium"), 0));
        // The final context is bounded by the evidence model's native window.
        assert_eq!(
            qwen.max_context_tokens,
            Some(QWEN_CANDIDATE.native_context_tokens + 1)
        );
        let fit = band(&QWEN_CANDIDATE, "fit-boundary");
        let headroom = QWEN_CANDIDATE.native_context_tokens - fit;
        assert!(qwen.admits(fit, headroom));
        assert!(!qwen.admits(fit, headroom + 1));

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
        assert!(!llama.admits(band(&LLAMA_CANDIDATE, "medium"), 0));
        for (spec, short) in [(&QWEN_CANDIDATE, "short"), (&LLAMA_CANDIDATE, "short")] {
            assert_eq!(
                qualify_kv_compression(
                    KvCompressionPolicy::Qualified,
                    family_for(Some(spec.family)),
                    band(spec, short),
                    0,
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
        for family in [KvModelFamily::Llama, KvModelFamily::Qwen3] {
            assert_eq!(family_for(Some(campaign_family(family))), Some(family));
        }
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

        // `qwen3_moe` parses as the Qwen3 architecture (the Qwen3 table family) with experts.
        let moe = ModelConfig::from_json(&serde_json::json!({
            "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
            "hidden_size": 128, "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
            "vocab_size": 32, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
            "tie_word_embeddings": false, "num_experts": 4, "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
        }))
        .unwrap();
        assert_eq!(moe.architecture, crate::config::Architecture::Qwen3);
        assert!(moe.moe.is_some());
        assert!(geometry_refusal(&moe)
            .unwrap()
            .contains("mixture-of-experts"));
    }
}
