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

use candle_gen::gen_core::{
    Capabilities, ConditioningKind, Modality, ModelDescriptor, Quant, SizeFloor,
};

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

/// Kolors' identity + the surface this candle lane wires: real classifier-free guidance (negative
/// prompt + CFG scale), txt2img, single-reference img2img, and packed **Q4/Q8** MLX-tier inference
/// (sc-10819). User LoRA/LoKr applies at load; ControlNet-pose and IP-Adapter retain separate Candle
/// providers, so this descriptor never promises a path `generate` cannot serve. Two backend-correct
/// deviations from `mlx-gen-kolors`: `backend = "candle"` and `mac_only = false`.
///
/// epic 7114 P4 (sc-7124): the native leading `euler_discrete` is the byte-exact DEFAULT, but the
/// curated ε/DDPM sampler menu (euler / euler_ancestral / heun / dpmpp_2m / dpmpp_sde / uni_pc / lcm /
/// ddim) + the curated σ-schedule axis (normal / karras / sgm_uniform / …) are ADDED over
/// `DiscreteModelSampling`; a curated solver name OR a curated scheduler (sc-8984) routes the new EPS
/// path while the default request keeps the native leading-Euler loop (see `crate::pipeline`). The
/// `discrete` scheduler alias is retained.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "kolors",
        // The tensor backend whose provider crate registered this engine (sc-3723). MLX sets "mlx".
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Kolors uses real classifier-free guidance over the ChatGLM3 conditioning.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // The registered generator accepts one Reference as a deterministic latent-init img2img.
            // ControlNet-pose and IP-Adapter remain separate, already-wired bespoke providers.
            conditioning: vec![ConditioningKind::Reference],
            // LoRA/LoKr merge into the vendored SDXL-family UNet on dense and packed tiers.
            supports_lora: true,
            supports_lokr: true,
            // epic 7114 P4 (sc-7124): the native leading EulerDiscrete (`euler_discrete`) stays the
            // byte-exact DEFAULT (N1), but the curated ε/DDPM menu (euler / euler_ancestral / heun /
            // dpmpp_2m / dpmpp_sde / uni_pc / lcm / ddim) is ADDED over `DiscreteModelSampling`, plus the
            // curated σ-schedule axis (normal / karras / sgm_uniform / …). A curated solver name OR a
            // curated scheduler (sc-8984) routes the new EPS path; `euler_discrete` (the native default)
            // and the legacy `discrete` scheduler alias keep their native behaviour.
            samplers: candle_gen::menu_with_aliases(
                candle_gen::curated_sampler_names(),
                &[DEFAULT_SAMPLER],
            ),
            schedulers: candle_gen::menu_with_aliases(
                candle_gen::curated_scheduler_names(),
                &["discrete"],
            ),
            supported_guidance_methods: vec![],
            min_size: 512,
            max_size: 2048,
            max_count: 8,
            // candle is the Windows/CUDA backend — NOT Mac-only (the MLX provider sets this true).
            // Not a distilled fixed-schedule model: any step count the shared sanity caps
            // admit is renderable (sc-19502).
            supported_steps: Vec::new(),
            mac_only: false,
            // Packed q4/q8 MLX-tier inference (sc-10819, epic 9083): the `SceneWorks/kolors-mlx` tier
            // packs the SDXL-family UNet + the four ChatGLM3 projections (VAE dense), and the candle
            // loader packed-detects it from disk (`pipeline::load_components`). Advertise Q4/Q8; the
            // `LoadSpec::quantize` overlay is an advisory no-op on an already-packed tier (as with
            // sdxl/boogu/flux2-dev). bf16 tiers stay dense (Quant::None).
            supported_quants: &[Quant::Q4, Quant::Q8],
            component_precision_floors: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
            unconditionally_engages_staged_residency: false,
            // Per-step latent previews (epic 16948, sc-16954): both lanes of this registered route
            // emit -- the curated driver lane and the native leading-Euler loop -- as do the
            // name-driven pose-control and IP-Adapter providers. Kolors adds no fit of its own; it
            // projects through `candle_gen_sdxl::preview` (one byte-identical VAE file).
            supports_preview: true,
            supports_prompt_enhancement: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            max_speakers: None,
            // No audio surface (sc-12834): pure image/video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
            size_floor: SizeFloor::RangeChecked,
        },
    }
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
