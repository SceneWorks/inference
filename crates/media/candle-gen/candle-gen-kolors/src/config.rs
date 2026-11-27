//! Kolors family configuration — the candle (Windows/CUDA) port of `mlx-gen-kolors`'s descriptor +
//! the ChatGLM3-6B encoder config, lifted from the diffusers `KolorsPipeline` reference.
//!
//! Kolors is a bilingual SDXL-family T2I model: it keeps the **SDXL UNet + SDXL VAE** but swaps the
//! dual-CLIP conditioning for a **ChatGLM3-6B** text encoder (penultimate hidden state = the
//! cross-attention context, last-token last-layer state = the pooled add-embedding). The UNet differs
//! from stock SDXL in exactly two places, both auto-present in the Kolors checkpoint: an
//! `encoder_hid_proj` Linear (4096→2048) projecting the ChatGLM3 context to the cross-attention width,
//! and the `text_time` add-embedding's `linear_1` taking **5632** = pooled(4096) + 6·256 time-ids (vs
//! SDXL's 2816 = pooled 1280 + 1536).
//!
//! The candle deviations from the mlx descriptor are the two backend-correct ones the SDXL / FLUX /
//! Z-Image / Chroma candle slices already make: `backend = "candle"` and `mac_only = false`. This lane
//! wires **txt2img + single-reference img2img + packed q4/q8 tiers** (sc-10819, epic 9083 — the
//! `SceneWorks/kolors-mlx` tier is packed-detected from disk). LoRA/LoKr are applied through the
//! shared SDXL-family UNet adapter loader;
//! ControlNet-pose and IP-Adapter stay on their existing bespoke Candle providers and are rejected
//! by this registered generator rather than silently dropped (the false-capability trap).

use candle_gen::gen_core::ModelDescriptor;
#[cfg(test)]
use candle_gen::gen_core::{ConditioningKind, Modality, Quant};

/// Registry id — matches the SceneWorks worker's `payload.model` for the Kolors family.
pub const MODEL_ID: &str = "kolors";

/// diffusers `KolorsPipeline` production defaults: 50 inference steps, CFG 5.0 (matches the mlx
/// `KolorsGenerator` registry defaults).
pub const DEFAULT_STEPS: u32 = 50;
pub const DEFAULT_GUIDANCE: f32 = 5.0;

/// The single Kolors sampler — diffusers `EulerDiscreteScheduler` (leading). Advertised under the same
/// name the mlx descriptor uses so a request the worker builds for one backend validates on the other.
pub const DEFAULT_SAMPLER: &str = "euler_discrete";

/// Kolors works in the SDXL VAE's /8 latent — both image dims must be multiples of 8.
pub const SIZE_MULTIPLE: u32 = 8;

/// Kolors' identity + capabilities, computed from [`crate::route_base`].
pub fn descriptor() -> ModelDescriptor {
    crate::route_base().compute_descriptor()
}

/// ChatGLM3-6B text config (the Kolors `text_encoder/config.json` values), hardcoded as in the mlx
/// provider (the snapshot config is fixed for the Kolors checkpoint).
#[derive(Clone, Copy, Debug)]
pub struct ChatGlmConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    /// Query heads (32).
    pub num_heads: usize,
    /// Multi-query KV groups (2). Broadcast to `num_heads` by the GQA-aware attention.
    pub num_kv_groups: usize,
    /// Per-head dim (`kv_channels` = 128).
    pub head_dim: usize,
    /// FFN inner width (13696). `dense_h_to_4h` emits `2 ·` this (fused gate+up).
    pub ffn_hidden: usize,
    pub rms_eps: f64,
    /// RoPE base θ (10000).
    pub rope_base: f64,
    /// Rotated head-dim prefix (`kv_channels / 2` = 64); the remaining dims pass through.
    pub rotary_dim: usize,
    pub vocab_size: usize,
}

impl ChatGlmConfig {
    /// The Kolors ChatGLM3-6B values.
    pub fn chatglm3_6b() -> Self {
        Self {
            hidden_size: 4096,
            num_layers: 28,
            num_heads: 32,
            num_kv_groups: 2,
            head_dim: 128,
            ffn_hidden: 13696,
            rms_eps: 1e-5,
            rope_base: 10_000.0,
            rotary_dim: 64,
            vocab_size: 65024,
        }
    }

    /// Fused `query_key_value` output width: `(num_heads + 2·num_kv_groups) · head_dim` = 4608.
    pub fn qkv_out(&self) -> usize {
        (self.num_heads + 2 * self.num_kv_groups) * self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **sc-18317 — the declared CFG-batching domain is reachable, and the two domains this lane has
    /// no mechanism for are refused.**
    ///
    /// The production `validate` delegates the shared contract to `Capabilities::validate_request`
    /// (asserted separately), so admitting/refusing here is admitting/refusing on the `generate` path.
    /// `Batched` is admitted because `common::cfg_batch_context` genuinely consumes it; `Sequential` is refused because the
    /// ChatGLM3 conditioning and the vendored SDXL denoise runs guidance as one batch; and the cadence / FFN-chunk domains are refused because this
    /// family has no such mechanism at all — the truthful state, not an oversight.
    #[test]
    fn execution_domains_are_declared_exactly_where_this_lane_consumes_them() {
        let caps = descriptor().capabilities;
        assert!(caps.execution.declaration_errors().is_empty());
        assert!(caps.execution.cfg_batching.is_supported());
        assert!(!caps.execution.graph_eval_cadence_blocks.is_supported());
        assert!(!caps.execution.ffn_chunk_rows.is_supported());

        let request = |memory| candle_gen::gen_core::GenerationRequest {
            prompt: "a fox".into(),
            width: 1024,
            height: 1024,
            count: 1,
            steps: Some(1),
            memory: Some(memory),
            ..Default::default()
        };

        caps.validate_request(
            "kolors",
            &request(candle_gen::gen_core::GenerationMemory {
                cfg_batching: Some(candle_gen::gen_core::CfgBatching::Batched),
                ..Default::default()
            }),
        )
        .expect("the implemented mode must be admitted");

        for (label, memory) in [
            (
                "cfg_batching",
                candle_gen::gen_core::GenerationMemory {
                    cfg_batching: Some(candle_gen::gen_core::CfgBatching::Sequential),
                    ..Default::default()
                },
            ),
            (
                "graph_eval_cadence",
                candle_gen::gen_core::GenerationMemory {
                    graph_eval_cadence: Some(candle_gen::gen_core::GraphEvalCadence::EVERY_BLOCK),
                    ..Default::default()
                },
            ),
            (
                "ffn_chunk",
                candle_gen::gen_core::GenerationMemory {
                    ffn_chunk: Some(candle_gen::gen_core::FfnChunk::new(2048).unwrap()),
                    ..Default::default()
                },
            ),
        ] {
            let error = caps
                .validate_request("kolors", &request(memory))
                .expect_err("an unimplemented execution selection must be refused");
            assert!(
                matches!(error, candle_gen::gen_core::Error::Unsupported(_)),
                "{label} must be a capability gap: {error:?}"
            );
            let message = error.to_string();
            assert!(message.contains(label), "{label}: {message}");
            assert!(
                message.contains("unset"),
                "{label} refusal must name the remedy: {message}"
            );
        }

        // And an unset selection still validates — the default-preservation half.
        caps.validate_request(
            "kolors",
            &request(candle_gen::gen_core::GenerationMemory::default()),
        )
        .expect("an unset execution selection must validate");
    }

    #[test]
    fn descriptor_advertises_txt2img_and_reference_img2img() {
        let d = descriptor();
        assert_eq!(d.id, "kolors");
        assert_eq!(d.family, "kolors");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(!d.capabilities.mac_only);
        assert_eq!(
            d.capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
        // User adapter stacks remain outside this registered base/img2img lane.
        assert!(d.capabilities.supports_lora);
        assert!(d.capabilities.supports_lokr);
        // sc-10819: packed q4/q8 MLX-tier inference is wired end-to-end, so Q4/Q8 are advertised.
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        // sc-7124: the curated ε/DDPM sampler menu + the native `euler_discrete` alias; the curated
        // scheduler axis + the legacy `discrete` alias. A curated solver name routes the new EPS path.
        assert_eq!(
            d.capabilities.samplers,
            candle_gen::menu_with_aliases(candle_gen::curated_sampler_names(), &[DEFAULT_SAMPLER])
        );
        assert!(d.capabilities.samplers.contains(&DEFAULT_SAMPLER));
        assert!(d.capabilities.samplers.contains(&"dpmpp_2m"));
        assert_eq!(
            d.capabilities.schedulers,
            candle_gen::menu_with_aliases(candle_gen::curated_scheduler_names(), &["discrete"])
        );
        assert!(d.capabilities.schedulers.contains(&"karras"));
        assert_eq!(d.capabilities.min_size, 512);
        assert_eq!(d.capabilities.max_size, 2048);
        assert_eq!(d.capabilities.max_count, 8);
    }

    #[test]
    fn chatglm3_dims() {
        let c = ChatGlmConfig::chatglm3_6b();
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.num_layers, 28);
        assert_eq!(c.qkv_out(), 4608); // (32 + 2·2)·128
        assert_eq!(c.rotary_dim, 64);
        assert_eq!(c.head_dim, 128);
    }
}
