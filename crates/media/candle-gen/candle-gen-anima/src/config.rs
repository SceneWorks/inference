//! Anima configs — the candle transcription of `mlx-gen-anima`'s `config.rs`. All values verified
//! against the real `anima-*-v1.0.safetensors` checkpoint, the diffusers
//! `Cosmos-2.0-Diffusion-2B-Text2Image` transformer config, and the `AnimaTextConditioner` /
//! `Qwen3-0.6B` reference (epic 10512, sc-10515 → sc-10525).

/// The Cosmos-Predict2 DiT config for Anima (`Cosmos-2.0-Diffusion-2B-Text2Image`).
#[derive(Clone, Copy, Debug)]
pub struct DitConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub num_layers: usize,
    pub mlp_ratio: f32,
    pub text_embed_dim: usize,
    pub adaln_lora_dim: usize,
    /// `(t, h, w)` maximum latent-grid extents (post-patch), from `max_size / patch_size` =
    /// `(128,240,240) / (1,2,2)` = `(128, 120, 120)`. Guards the RoPE against position OOD.
    pub max_size: (usize, usize, usize),
    pub patch_size: (usize, usize, usize),
    pub rope_scale: (f32, f32, f32),
    /// `concat_padding_mask: true` ⇒ patch-embed input is `in_channels + 1` (17), the extra channel a
    /// full-res padding mask (all zeros for T2I).
    pub concat_padding_mask: bool,
}

impl DitConfig {
    /// The `Cosmos-2.0-Diffusion-2B-Text2Image` config Anima ships (extra_pos_embed_type=null,
    /// use_crossattn_projection=false, controlnet_block_every_n=null — all omitted here).
    pub fn anima() -> Self {
        Self {
            in_channels: 16,
            out_channels: 16,
            num_attention_heads: 16,
            attention_head_dim: 128,
            num_layers: 28,
            mlp_ratio: 4.0,
            text_embed_dim: 1024,
            adaln_lora_dim: 256,
            max_size: (128, 120, 120),
            patch_size: (1, 2, 2),
            rope_scale: (1.0, 4.0, 4.0),
            concat_padding_mask: true,
        }
    }

    /// hidden = heads · head_dim (2048).
    pub fn hidden_size(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// patch-embed input channels — 17 when `concat_padding_mask` (16 latent + 1 mask).
    pub fn patch_in_channels(&self) -> usize {
        if self.concat_padding_mask {
            self.in_channels + 1
        } else {
            self.in_channels
        }
    }
}

/// The `AnimaTextConditioner` config (verified: model_dim 1024, 6 layers, 16 heads, T5 vocab 32128,
/// min_sequence_length 512, use_self_attention=true, use_layer_norm=false).
#[derive(Clone, Copy, Debug)]
pub struct ConditionerConfig {
    pub source_dim: usize,
    pub target_dim: usize,
    pub model_dim: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub mlp_ratio: f32,
    pub target_vocab_size: usize,
    pub min_sequence_length: usize,
    pub rope_theta: f32,
    /// RMSNorm eps for every norm in the conditioner (use_layer_norm=false ⇒ RMSNorm eps 1e-6).
    pub norm_eps: f64,
}

impl ConditionerConfig {
    pub fn anima() -> Self {
        Self {
            source_dim: 1024,
            target_dim: 1024,
            model_dim: 1024,
            num_layers: 6,
            num_attention_heads: 16,
            mlp_ratio: 4.0,
            target_vocab_size: 32128,
            min_sequence_length: 512,
            rope_theta: 10000.0,
            norm_eps: 1e-6,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.model_dim / self.num_attention_heads
    }
}

/// The Qwen3-0.6B base text encoder config (hidden 1024, 28 layers, GQA 16/8, head_dim 128,
/// rope_theta 1e6, rms_norm_eps 1e-6, vocab 151936, no LM head).
#[derive(Clone, Copy, Debug)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

impl Qwen3Config {
    pub fn anima() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 1024,
            n_layers: 28,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        }
    }

    /// GQA group count — each KV head is shared by `n_heads / n_kv_heads` query heads (16/8 = 2 for
    /// Anima). The candle-gen-z-image DiT attention wart (sizes K/V by `n_kv_heads`, reshapes by
    /// `n_heads`) is MHA-only and would collapse this; the Qwen3 encoder here repeat-expands the KV
    /// heads by exactly this factor.
    pub fn kv_groups(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}

/// The three shipped Anima variants (they differ only in the DiT file + default steps/CFG).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Base,
    Aesthetic,
    Turbo,
}

impl Variant {
    /// Registry id (matches the SceneWorks worker `payload.model` / manifest `engine_id`, sc-10523).
    pub fn id(self) -> &'static str {
        match self {
            Variant::Base => "anima_base",
            Variant::Aesthetic => "anima_aesthetic",
            Variant::Turbo => "anima_turbo",
        }
    }

    /// The DiT safetensors filename under `split_files/diffusion_models/`.
    pub fn dit_filename(self) -> &'static str {
        match self {
            Variant::Base => "anima-base-v1.0.safetensors",
            Variant::Aesthetic => "anima-aesthetic-v1.0.safetensors",
            Variant::Turbo => "anima-turbo-v1.0.safetensors",
        }
    }

    /// Default denoising steps.
    pub fn default_steps(self) -> u32 {
        match self {
            Variant::Base | Variant::Aesthetic => 30,
            Variant::Turbo => 10,
        }
    }

    /// Default CFG guidance scale. Turbo is the merged CFG-free student (1.0 ⇒ single forward).
    pub fn default_guidance(self) -> f32 {
        match self {
            Variant::Base | Variant::Aesthetic => 4.5,
            Variant::Turbo => 1.0,
        }
    }

    /// Whether this variant runs classifier-free guidance (two forwards/step).
    pub fn uses_cfg(self) -> bool {
        !matches!(self, Variant::Turbo)
    }

    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "anima_base" => Some(Variant::Base),
            "anima_aesthetic" => Some(Variant::Aesthetic),
            "anima_turbo" => Some(Variant::Turbo),
            _ => None,
        }
    }
}

/// VAE spatial compression (8×) and latent channels (16) — shared with Qwen-Image.
pub const VAE_COMPRESSION: u32 = 8;
pub const VAE_CHANNELS: usize = 16;
/// patchify + VAE alignment: `vae(8) · patch(2) = 16` — W/H must be a multiple of this. Exposed as
/// the pinned-engine stride SceneWorks ties each advertised Anima image bucket to (sc-12612),
/// mirroring `wan::config::SIZE_MULTIPLE_14B`. `validate` enforces exactly this value, so the const
/// cannot drift from the check.
pub const RES_MULTIPLE: u32 = 16;
/// FlowMatchEuler static time-shift (`FlowMatchEulerDiscreteScheduler(shift=3.0)`).
pub const SIGMA_SHIFT: f32 = 3.0;
/// Qwen2 tokenizer pad id (`<|endoftext|>`); no BOS/EOS, padding="longest".
pub const QWEN_PAD_TOKEN_ID: i32 = 151643;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dit_patch_input_is_17_channels() {
        let c = DitConfig::anima();
        assert_eq!(
            c.patch_in_channels(),
            17,
            "concat_padding_mask ⇒ 16+1 channels"
        );
        assert_eq!(c.hidden_size(), 2048);
    }

    #[test]
    fn qwen3_is_gqa_16_over_8() {
        let q = Qwen3Config::anima();
        assert_eq!(q.n_heads, 16);
        assert_eq!(q.n_kv_heads, 8);
        assert_eq!(
            q.kv_groups(),
            2,
            "GQA 16/8 ⇒ each KV head shared by 2 query heads"
        );
    }

    #[test]
    fn variant_metadata() {
        assert_eq!(Variant::Base.default_steps(), 30);
        assert_eq!(Variant::Turbo.default_steps(), 10);
        assert!((Variant::Base.default_guidance() - 4.5).abs() < 1e-6);
        assert!((Variant::Turbo.default_guidance() - 1.0).abs() < 1e-6);
        assert!(Variant::Base.uses_cfg());
        assert!(!Variant::Turbo.uses_cfg());
        assert_eq!(
            Variant::from_id("anima_aesthetic"),
            Some(Variant::Aesthetic)
        );
        assert_eq!(Variant::from_id("nope"), None);
    }
}
