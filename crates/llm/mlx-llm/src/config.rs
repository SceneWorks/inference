//! Model configuration parsed from a Hugging Face `config.json`.
//!
//! Value-based parsing (no `serde` derive) matching the mlx-gen provider convention, so config keys
//! can vary and default gracefully. Story 7156 covers the Llama family; BYO architecture dispatch
//! (story 7163) layers Qwen3 on top; model breadth (story 7173) adds Phi-3, Qwen2-MoE, Gemma-2,
//! GLM-4, and DeepSeek-V2 (MLA) behind the same single generic decoder. Mirrors candle-llm's
//! `config.rs` (the cross-backend blueprint) so the two backends dispatch identically.

use crate::primitives::activation::ActivationRole;
use std::path::Path;

use serde_json::Value;

use crate::error::{Error, Result};
use crate::primitives::{QuantSpec, Rope};

// The tensor-free half of the config lives in `core_llm::model_config` (sc-27255, sc-27308), shared
// with candle-llm; this module keeps the MLX-specific assembly: `ModelConfig` (which carries the
// MLX `QuantSpec` and the activation role) and the `Rope` builders.
pub use core_llm::model_config::{
    Architecture, BidirectionalAttention, Gemma4Config, LayerAttention, LayerAttentionType,
    MlaConfig, ModelConfigError, MoeConfig, RopeScaling, RopeType, YarnConfig,
};

/// Builds a [`LayerAttention`]'s MLX [`Rope`] (the shared descriptor is tensor-free).
pub trait LayerRope {
    /// Build this layer type's RoPE. Only the schedules a per-layer-type table can express
    /// ([`RopeType`]) are built here; the whole-model schedules (llama3 / YaRN / interleaved) stay
    /// on [`ModelConfig::build_rope`], which [`ModelConfig::layer_rope`] dispatches to for every
    /// architecture that predates Gemma 4.
    fn build_rope(&self) -> Rope;
}

impl LayerRope for LayerAttention {
    fn build_rope(&self) -> Rope {
        match self.rope_type {
            RopeType::Default => Rope::standard(self.head_dim, self.rope_theta),
            RopeType::Proportional => Rope::proportional(
                self.head_dim,
                self.rope_theta,
                self.partial_rotary_factor,
                self.rope_factor,
            ),
        }
    }
}

/// Configuration for a generic causal decoder (Llama family + the breadth architectures).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    /// Model/residual width.
    pub hidden_size: i32,
    /// MLP inner width.
    pub intermediate_size: i32,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention (query) heads.
    pub num_heads: i32,
    /// Number of key/value heads (GQA; equals `num_heads` for MHA).
    pub num_kv_heads: i32,
    /// Per-head dimension.
    pub head_dim: i32,
    /// Vocabulary size.
    pub vocab_size: i32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Optional Llama-3 RoPE scaling; `None` ⇒ standard RoPE.
    pub rope_scaling: Option<RopeScaling>,
    /// Whether `lm_head` is tied to the input embeddings.
    pub tie_word_embeddings: bool,
    /// The decoder architecture (drives q/k norm, the family tag, and the breadth-specific paths).
    pub architecture: Architecture,
    /// Max context length (`max_position_embeddings`); `0` if unspecified.
    pub max_position_embeddings: i32,
    /// Present when the snapshot stores **pre-quantized** projections (a `quantization` block in
    /// `config.json`, e.g. from the GGUF converter): the group size / bit width the stored
    /// `weight`/`scales`/`biases` tensors were packed with. `None` ⇒ a dense snapshot.
    pub quantization: Option<QuantSpec>,
    /// Mixture-of-Experts parameters, present for an MoE decoder (Qwen2-MoE, DeepSeek-V2); `None` ⇒
    /// dense MLP.
    pub moe: Option<MoeConfig>,
    /// Gemma-2 attention-score soft-cap (`attn_logit_softcapping`); `None` ⇒ no cap.
    pub attn_logit_softcap: Option<f32>,
    /// Gemma-2 final-logit soft-cap (`final_logit_softcapping`); `None` ⇒ no cap.
    pub final_logit_softcap: Option<f32>,
    /// Gemma-2 attention scale denominator (`query_pre_attn_scalar`); `None` ⇒ `head_dim`.
    pub query_pre_attn_scalar: Option<i32>,
    /// Fraction of `head_dim` that RoPE rotates (`partial_rotary_factor`); `1.0` ⇒ full rotary.
    pub partial_rotary_factor: f32,
    /// Multi-head Latent Attention parameters, present for a DeepSeek-V2 decoder; `None` ⇒ GQA.
    pub mla: Option<MlaConfig>,
    /// YaRN RoPE-scaling parameters (DeepSeek-V2); `None` ⇒ no YaRN.
    pub yarn: Option<YarnConfig>,
    /// Interleaved multimodal-RoPE section `[t, h, w]` (`rope_scaling.mrope_section`, sums to
    /// `rotary_dim/2`), present for Qwen3-VL; `None` ⇒ a plain 1-D RoPE model. Drives the per-channel
    /// axis assignment for image (3-D) positions; irrelevant to the text path (where all three rows
    /// are equal, so the interleaved table is bit-identical to plain 1-D RoPE).
    pub mrope_section: Option<[i32; 3]>,
    /// Gemma 4's **per-layer-type** attention table, present only for a Gemma 4 decoder; `None` ⇒ a
    /// uniform model, where every layer resolves to the scalar `head_dim` / `num_kv_heads` /
    /// `rope_theta` above (see [`ModelConfig::layer_attention`]). Adding this block is what lets one
    /// `ModelConfig` describe a stack whose layers disagree on head dim, KV-head count, RoPE
    /// schedule, and mask — without moving any scalar an existing architecture reads.
    ///
    /// **Boxed on purpose.** `ModelConfig` is cloned into every loaded decoder, and the table is
    /// over a hundred bytes of `Vec` + two descriptors that no pre-Gemma-4 model has any use for;
    /// inlining it inflated the provider's decoder enum enough to trip `clippy::large_enum_variant`.
    /// One pointer keeps the common config exactly as cheap as it was.
    pub gemma4: Option<Box<Gemma4Config>>,
    /// What this decoder's activations feed — the key of the activation-dtype policy
    /// ([`crate::primitives::activation`]): [`ActivationRole::LlmDecode`] for every parsed config;
    /// the LTX-2.5 text encoder sets [`ActivationRole::LtxTextEncoder`] (sc-24446).
    pub activation_role: ActivationRole,
}

impl ModelConfig {
    /// Parse from an already-decoded `config.json` value.
    pub fn from_json(v: &Value) -> Result<Self> {
        let architecture = Architecture::from_config(v)?;
        // The hybrid Qwen3.6 decoder has its own config/weights path in the provider. Do not
        // mis-parse its VLM-wrapped config as a generic full-attention decoder here.
        if architecture == Architecture::Qwen35 {
            return Err(Error::Unsupported(
                "Qwen3.6 (model_type `qwen3_5`) is a hybrid linear-attention (Gated DeltaNet + gated \
                 full attention) decoder; it is routed through Qwen35Config rather than the generic \
                 ModelConfig"
                    .to_string(),
            ));
        }
        // The Qwen3-VL and Gemma 4 wrappers nest the decoder fields under `text_config`
        // (`hidden_size`, `num_hidden_layers`, `rope_theta`, …). Read the decoder fields from there,
        // falling back to the top level for a non-wrapped config. Other architectures read top-level.
        let v = if architecture.nests_text_config() {
            v.get("text_config").unwrap_or(v)
        } else {
            v
        };
        // Normalize only Qwen3-VL's modern spelling through shared tensor-neutral policy.
        // Gemma 4's per-layer map and every unrelated architecture retain their own parser.
        let normalized_rope = if architecture.is_qwen3_vl() {
            Some(core_llm::rope_config::normalize_qwen3_vl_rope(v).map_err(Error::Config)?)
        } else {
            None
        };
        let v = normalized_rope.as_ref().unwrap_or(v);
        let int =
            |key: &str| -> Option<i32> { v.get(key).and_then(|x| x.as_i64()).map(|x| x as i32) };
        let req_int = |key: &str| -> Result<i32> {
            int(key).ok_or_else(|| Error::Config(format!("config.json missing integer `{key}`")))
        };
        let f32_opt =
            |key: &str| -> Option<f32> { v.get(key).and_then(|x| x.as_f64()).map(|x| x as f32) };

        let hidden_size = req_int("hidden_size")?;
        let num_heads = req_int("num_attention_heads")?;
        // Multi-head Latent Attention (DeepSeek-V2): present iff the KV low-rank dim is configured.
        let mla = MlaConfig::from_json(v, hidden_size, num_heads);
        // MLA attends over `qk_nope_head_dim + qk_rope_head_dim` per head (no `head_dim` config key).
        let head_dim = match &mla {
            Some(m) => m.q_head_dim(),
            None => int("head_dim").unwrap_or(hidden_size / num_heads),
        };
        let num_kv_heads = int("num_key_value_heads").unwrap_or(num_heads);
        let num_layers = req_int("num_hidden_layers")? as usize;
        let intermediate_size = req_int("intermediate_size")?;
        let vocab_size = req_int("vocab_size")?;
        let rms_norm_eps = v
            .get("rms_norm_eps")
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .unwrap_or(1e-5);
        let rope_theta = v
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .unwrap_or(500_000.0);
        let tie_word_embeddings = v
            .get("tie_word_embeddings")
            .and_then(|x| x.as_bool())
            // Gemma always ties its (huge) embedding to the LM head; the config often omits the key.
            .unwrap_or(architecture.is_gemma());
        let max_position_embeddings = int("max_position_embeddings").unwrap_or(0);

        let attn_logit_softcap = f32_opt("attn_logit_softcapping");
        let final_logit_softcap = f32_opt("final_logit_softcapping");
        let query_pre_attn_scalar = int("query_pre_attn_scalar");
        let partial_rotary_factor = f32_opt("partial_rotary_factor").unwrap_or(1.0);

        // A `quantization` block marks a pre-quantized snapshot (the GGUF converter writes it).
        let quantization = v.get("quantization").and_then(|q| {
            let group_size = q.get("group_size").and_then(|x| x.as_i64())? as i32;
            let bits = q.get("bits").and_then(|x| x.as_i64())? as i32;
            Some(QuantSpec {
                group_size,
                bits,
                grid: Default::default(),
            })
        });

        // Mixture-of-Experts FFN params, YaRN (DeepSeek-V2) and Llama-3 RoPE scaling.
        let moe = MoeConfig::from_json(v, intermediate_size);
        let yarn = YarnConfig::from_json(v);
        let rope_scaling = RopeScaling::from_json(v);

        // Interleaved multimodal-RoPE section `[t, h, w]` (Qwen3-VL `rope_scaling.mrope_section`).
        let mrope_section = core_llm::model_config::mrope_section(v);

        // Gemma 4's per-layer-type attention table. Parsed **last** so it can seed its
        // `sliding_attention` descriptor from the scalars above, then reflect its own
        // `sliding_attention` values back into them: the scalar `head_dim` / `num_kv_heads` /
        // `rope_theta` / `partial_rotary_factor` stay the "layer 0" view of the model, so anything
        // that reads them (memory estimates, cache sizing, diagnostics) sees a coherent shape rather
        // than a `rope_theta` default the config never carried. Every other architecture leaves this
        // `None` and every scalar untouched — the byte-identical-parse invariant.
        let gemma4 = match architecture.is_gemma4() {
            true => Some(Box::new(Gemma4Config::from_json(
                v,
                num_layers,
                num_kv_heads,
                head_dim,
            )?)),
            false => None,
        };
        let (head_dim, num_kv_heads, rope_theta, partial_rotary_factor) = match &gemma4 {
            Some(g) => (
                g.sliding.head_dim,
                g.sliding.num_kv_heads,
                g.sliding.rope_theta,
                g.sliding.partial_rotary_factor,
            ),
            None => (head_dim, num_kv_heads, rope_theta, partial_rotary_factor),
        };

        Ok(Self {
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            rope_scaling,
            tie_word_embeddings,
            architecture,
            max_position_embeddings,
            quantization,
            moe,
            attn_logit_softcap,
            final_logit_softcap,
            query_pre_attn_scalar,
            partial_rotary_factor,
            mla,
            yarn,
            mrope_section,
            gemma4,
            activation_role: ActivationRole::LlmDecode,
        })
    }

    /// Number of head dimensions RoPE rotates (`round(head_dim · partial_rotary_factor)`, even).
    pub fn rotary_dim(&self) -> i32 {
        let rd = (self.head_dim as f32 * self.partial_rotary_factor).round() as i32;
        rd & !1 // force even (RoPE rotates in pairs)
    }

    /// The interleaved multimodal-RoPE section `[t, h, w]` (Qwen3-VL), defaulting to an even split
    /// of `rotary_dim/2` (biased toward `t` then `h`) when the config omits `mrope_section`. Used by
    /// the image (3-D) RoPE path; the text path (all rows equal) is independent of the split.
    pub fn mrope_section_resolved(&self) -> [usize; 3] {
        if let Some(s) = self.mrope_section {
            return [
                s[0].max(0) as usize,
                s[1].max(0) as usize,
                s[2].max(0) as usize,
            ];
        }
        let half = (self.rotary_dim() / 2) as usize;
        let base = half / 3;
        let rem = half % 3;
        [base + (rem > 0) as usize, base + (rem > 1) as usize, base]
    }

    /// Whether the decoder uses a Mixture-of-Experts FFN.
    pub fn is_moe(&self) -> bool {
        self.moe.is_some()
    }

    /// Whether this is a Gemma 4 decoder (and therefore carries a [`Gemma4Config`] table).
    pub fn is_gemma4(&self) -> bool {
        self.architecture.is_gemma4()
    }

    /// The attention descriptor for decoder layer `i`.
    ///
    /// Gemma 4 resolves through its [`Gemma4Config`] table (`sliding_attention` vs
    /// `full_attention`); **every** other architecture resolves every layer to the same descriptor
    /// built from the scalar `head_dim` / `num_kv_heads` / `rope_theta` / `partial_rotary_factor` —
    /// so a uniform model reads exactly the values it read before this table existed.
    pub fn layer_attention(&self, layer_idx: usize) -> LayerAttention {
        match &self.gemma4 {
            Some(g) => g.layer(layer_idx),
            None => LayerAttention {
                head_dim: self.head_dim,
                num_kv_heads: self.num_kv_heads,
                rope_type: RopeType::Default,
                rope_theta: self.rope_theta,
                partial_rotary_factor: self.partial_rotary_factor,
                rope_factor: 1.0,
                sliding_window: None,
                k_eq_v: false,
            },
        }
    }

    /// The layer type of decoder layer `i` — [`LayerAttentionType::Full`] for every uniform
    /// (pre-Gemma-4) architecture.
    pub fn layer_type(&self, layer_idx: usize) -> LayerAttentionType {
        match &self.gemma4 {
            Some(g) => g.layer_type(layer_idx),
            None => LayerAttentionType::Full,
        }
    }

    /// GQA group count for decoder layer `i` (`num_heads / layer num_kv_heads`). Equals
    /// [`ModelConfig::groups`] for every uniform architecture; Gemma 4's `full_attention` layers
    /// have a different (much larger) group count than its `sliding_attention` layers.
    pub fn layer_groups(&self, layer_idx: usize) -> i32 {
        self.layer_attention(layer_idx).groups(self.num_heads)
    }

    /// Per-head dimension for decoder layer `i`.
    pub fn layer_head_dim(&self, layer_idx: usize) -> i32 {
        self.layer_attention(layer_idx).head_dim
    }

    /// `Some(window)` when decoder layer `i` attends only the `window` most recent keys; `None` for
    /// a full causal layer (and for every uniform architecture).
    pub fn layer_sliding_window(&self, layer_idx: usize) -> Option<i32> {
        self.layer_attention(layer_idx).sliding_window
    }

    /// The RoPE for decoder layer `i`.
    ///
    /// Gemma 4 builds the layer type's schedule (`default` / `proportional`); every other
    /// architecture returns [`ModelConfig::build_rope`] unchanged — including the whole-model
    /// llama3 / YaRN / partial-interleaved schedules, which have no per-layer-type form.
    pub fn layer_rope(&self, layer_idx: usize) -> Rope {
        match &self.gemma4 {
            Some(g) => g.layer(layer_idx).build_rope(),
            None => self.build_rope(),
        }
    }

    /// Whether the decoder uses Multi-head Latent Attention (DeepSeek-V2).
    pub fn is_mla(&self) -> bool {
        self.mla.is_some()
    }

    /// Whether attention applies per-head q/k RMSNorm (Qwen3).
    pub fn has_qk_norm(&self) -> bool {
        self.architecture.has_qk_norm()
    }

    /// Read and parse `config.json` from a snapshot directory (or a file path).
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let path = if dir.is_dir() {
            dir.join("config.json")
        } else {
            dir.to_path_buf()
        };
        let text = crate::__model_input_reader!(
            std::fs::read_to_string(&path),
            "crates/llm/mlx-llm/src/config.rs::from_dir::std::fs::read_to_string@49555",
            ArtifactInput,
            &path,
            |__model_input_path| std::fs::read_to_string(__model_input_path)
        )?;
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Config(format!("parse {}: {e}", path.display())))?;
        Self::from_json(&v)
    }

    /// Build the RoPE for this config. DeepSeek-V2 (MLA) rotates only the `qk_rope_head_dim`
    /// sub-vector with YaRN-scaled, interleaved frequencies; GLM-4 uses partial/interleaved RoPE;
    /// Llama-3 scaling applies the NTK-by-parts schedule; otherwise standard full-width NeoX RoPE.
    pub fn build_rope(&self) -> Rope {
        if let Some(mla) = self.mla {
            let rope_dim = mla.qk_rope_head_dim;
            return match self.yarn {
                Some(y) => Rope::yarn(
                    rope_dim,
                    self.rope_theta,
                    y.factor,
                    y.beta_fast,
                    y.beta_slow,
                    y.original_context,
                ),
                None => Rope::partial(rope_dim, self.rope_theta, true),
            };
        }
        let rotary_dim = self.rotary_dim();
        if rotary_dim < self.head_dim || self.architecture.rope_interleaved() {
            return Rope::partial(
                rotary_dim,
                self.rope_theta,
                self.architecture.rope_interleaved(),
            );
        }
        match self.rope_scaling {
            Some(rs) => Rope::llama3(
                self.head_dim,
                self.rope_theta,
                rs.factor,
                rs.low_freq_factor,
                rs.high_freq_factor,
                rs.original_context,
            ),
            None => Rope::standard(self.head_dim, self.rope_theta),
        }
    }

    /// Number of GQA groups (`num_heads / num_kv_heads`).
    pub fn groups(&self) -> i32 {
        self.num_heads / self.num_kv_heads
    }

    /// Attention scale. MLA (DeepSeek-V2) scales by `q_head_dim^(-0.5)` multiplied by the YaRN
    /// magnitude `mscale²`; Gemma-2 scales by `query_pre_attn_scalar^(-0.5)` (a denominator that may
    /// differ from `head_dim`); **Gemma 4 scales by `1.0`** — upstream's
    /// `Gemma4UnifiedTextAttention` sets `self.scaling = 1.0`, because the per-head q/k RMSNorms
    /// already fix the score magnitude; dividing by `√head_dim` on top would shrink every logit.
    /// Otherwise the usual `head_dim^(-0.5)`.
    pub fn attn_scale(&self) -> f32 {
        if self.architecture.is_gemma4() {
            return 1.0;
        }
        if let Some(mla) = self.mla {
            let base = (mla.q_head_dim() as f32).powf(-0.5);
            let mscale = self.yarn.map_or(1.0, |y| y.softmax_mscale());
            return base * mscale * mscale;
        }
        let denom = self.query_pre_attn_scalar.unwrap_or(self.head_dim) as f32;
        denom.powf(-0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_llama31_8b_style_config() {
        let v = json!({
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192
            }
        });
        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.head_dim, 128); // 4096/32
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.groups(), 4);
        assert_eq!(cfg.rope_scaling.unwrap().factor, 8.0);
    }

    #[test]
    fn defaults_kv_heads_and_head_dim() {
        let v = json!({
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "vocab_size": 32
        });
        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(cfg.head_dim, 16); // 64/4
        assert_eq!(cfg.num_kv_heads, 4); // defaults to num_heads (MHA)
        assert!(cfg.rope_scaling.is_none());
    }

    #[test]
    fn missing_required_field_errors() {
        let v = json!({ "hidden_size": 64 });
        assert!(matches!(ModelConfig::from_json(&v), Err(Error::Config(_))));
    }

    #[test]
    fn architecture_dispatch() {
        let qwen3 = json!({ "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3" });
        assert_eq!(
            Architecture::from_config(&qwen3).unwrap(),
            Architecture::Qwen3
        );

        let llama = json!({ "architectures": ["LlamaForCausalLM"], "model_type": "llama" });
        assert_eq!(
            Architecture::from_config(&llama).unwrap(),
            Architecture::Llama
        );

        let mistral = json!({ "architectures": ["MistralForCausalLM"] });
        assert_eq!(
            Architecture::from_config(&mistral).unwrap(),
            Architecture::Llama
        );

        // Dense Qwen2 shares the Llama decoder shape (q/k/v bias auto-detected); routed to Llama.
        let qwen2 = json!({ "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2" });
        let a = Architecture::from_config(&qwen2).unwrap();
        assert_eq!(a, Architecture::Llama);
        assert!(!a.has_qk_norm());

        // Minimal config (no arch fields) defaults to Llama.
        let minimal = json!({ "hidden_size": 8 });
        assert_eq!(
            Architecture::from_config(&minimal).unwrap(),
            Architecture::Llama
        );

        // A named-but-unsupported arch is rejected.
        let unknown = json!({ "architectures": ["MambaForCausalLM"], "model_type": "mamba" });
        assert!(matches!(
            Architecture::from_config(&unknown),
            Err(ModelConfigError::Unsupported(_))
        ));

        // Phi-3 (packed qkv / gate_up; otherwise the Llama shape).
        let phi3 = json!({ "architectures": ["Phi3ForCausalLM"], "model_type": "phi3" });
        let a = Architecture::from_config(&phi3).unwrap();
        assert_eq!(a, Architecture::Phi3);
        assert_eq!(a.family(), "phi3");
        assert!(!a.has_qk_norm());

        // Qwen2-MoE (sparse FFN + q/k/v bias; no q/k norm).
        let qwen2_moe =
            json!({ "architectures": ["Qwen2MoeForCausalLM"], "model_type": "qwen2_moe" });
        let a = Architecture::from_config(&qwen2_moe).unwrap();
        assert_eq!(a, Architecture::Qwen2Moe);
        assert_eq!(a.family(), "qwen2_moe");
        assert!(!a.has_qk_norm());

        // Gemma-2 (soft-caps, sandwich norms; ties embeddings by default).
        let gemma2 = json!({ "architectures": ["Gemma2ForCausalLM"], "model_type": "gemma2" });
        let a = Architecture::from_config(&gemma2).unwrap();
        assert_eq!(a, Architecture::Gemma2);
        assert_eq!(a.family(), "gemma2");
        assert!(a.is_gemma2());
        assert!(a.is_sandwich());
        assert!(!a.has_qk_norm());

        // GLM-4 (sandwich norms, partial+interleaved RoPE; standard RMSNorm).
        let glm4 = json!({ "architectures": ["Glm4ForCausalLM"], "model_type": "glm4" });
        let a = Architecture::from_config(&glm4).unwrap();
        assert_eq!(a, Architecture::Glm4);
        assert_eq!(a.family(), "glm4");
        assert!(a.is_sandwich());
        assert!(a.rope_interleaved());
        assert!(!a.is_gemma2());
    }

    #[test]
    fn parses_glm4_config_partial_rotary() {
        let v = json!({
            "architectures": ["Glm4ForCausalLM"], "model_type": "glm4",
            "hidden_size": 4096, "intermediate_size": 13696, "num_hidden_layers": 40,
            "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
            "vocab_size": 151552, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5
        });
        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(cfg.architecture, Architecture::Glm4);
        assert_eq!(cfg.partial_rotary_factor, 0.5);
        assert_eq!(cfg.rotary_dim(), 64); // 128 * 0.5
        let rope = cfg.build_rope();
        assert_eq!(rope.dim(), 64);
        assert!(rope.interleaved());
        // A model with no partial_rotary_factor rotates the full head_dim.
        let full = ModelConfig::from_json(&json!({
            "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "vocab_size": 32
        }))
        .unwrap();
        assert_eq!(full.partial_rotary_factor, 1.0);
        assert_eq!(full.rotary_dim(), full.head_dim);
    }

    #[test]
    fn parses_gemma2_config() {
        let v = json!({
            "architectures": ["Gemma2ForCausalLM"], "model_type": "gemma2",
            "hidden_size": 2304, "intermediate_size": 9216, "num_hidden_layers": 26,
            "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 256000,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "query_pre_attn_scalar": 256,
            "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0
            // tie_word_embeddings intentionally omitted — Gemma ties by default.
        });
        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(cfg.architecture, Architecture::Gemma2);
        assert_eq!(cfg.head_dim, 256); // explicit, != 2304/8
        assert!(cfg.tie_word_embeddings, "Gemma ties by default");
        assert_eq!(cfg.attn_logit_softcap, Some(50.0));
        assert_eq!(cfg.final_logit_softcap, Some(30.0));
        assert_eq!(cfg.query_pre_attn_scalar, Some(256));
        // Attention scale uses query_pre_attn_scalar (256), not head_dim — equal here, but the path
        // is exercised.
        assert!((cfg.attn_scale() - (256f32).powf(-0.5)).abs() < 1e-9);
    }

    #[test]
    fn parses_qwen2_moe_config() {
        let v = json!({
            "architectures": ["Qwen2MoeForCausalLM"], "model_type": "qwen2_moe",
            "hidden_size": 2048, "intermediate_size": 5632, "num_hidden_layers": 24,
            "num_attention_heads": 16, "num_key_value_heads": 16, "vocab_size": 151936,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
            "num_experts": 60, "num_experts_per_tok": 4, "norm_topk_prob": false,
            "moe_intermediate_size": 1408, "shared_expert_intermediate_size": 5632
        });
        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(cfg.architecture, Architecture::Qwen2Moe);
        assert!(cfg.is_moe());
        let moe = cfg.moe.unwrap();
        assert_eq!(moe.num_experts, 60);
        assert_eq!(moe.num_experts_per_tok, 4);
        assert_eq!(moe.moe_intermediate_size, 1408);
        assert_eq!(moe.shared_expert_intermediate_size, 5632);
        assert!(!moe.norm_topk_prob);
        // A dense (non-MoE) config has no MoE block.
        assert!(!ModelConfig::from_json(&json!({
            "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
            "num_attention_heads": 2, "vocab_size": 32
        }))
        .unwrap()
        .is_moe());
    }

    #[test]
    fn parses_deepseek_v2_lite_config() {
        // DeepSeek-V2-Lite-Chat: MLA (no query LoRA), fine-grained MoE with a leading dense layer,
        // YaRN RoPE. The numbers are the real config's.
        let v = json!({
            "architectures": ["DeepseekV2ForCausalLM"], "model_type": "deepseek_v2",
            "hidden_size": 2048, "intermediate_size": 10944, "num_hidden_layers": 27,
            "num_attention_heads": 16, "num_key_value_heads": 16, "vocab_size": 102400,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 163840,
            "tie_word_embeddings": false,
            "q_lora_rank": null, "kv_lora_rank": 512,
            "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
            "n_routed_experts": 64, "num_experts_per_tok": 6, "n_shared_experts": 2,
            "moe_intermediate_size": 1408, "first_k_dense_replace": 1, "norm_topk_prob": false,
            "routed_scaling_factor": 1.0,
            "rope_scaling": {
                "type": "yarn", "factor": 40, "beta_fast": 32, "beta_slow": 1,
                "mscale": 0.707, "mscale_all_dim": 0.707,
                "original_max_position_embeddings": 4096
            }
        });
        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(cfg.architecture, Architecture::DeepseekV2);
        assert!(cfg.is_mla());
        assert!(cfg.architecture.rope_interleaved());

        let mla = cfg.mla.unwrap();
        assert_eq!(mla.q_lora_rank, None); // V2-Lite has a full q_proj
        assert_eq!(mla.kv_lora_rank, 512);
        assert_eq!(mla.qk_nope_head_dim, 128);
        assert_eq!(mla.qk_rope_head_dim, 64);
        assert_eq!(mla.v_head_dim, 128);
        assert_eq!(mla.q_head_dim(), 192);
        // head_dim is overridden to the MLA q/k head dim (no `head_dim` config key).
        assert_eq!(cfg.head_dim, 192);

        let moe = cfg.moe.unwrap();
        assert_eq!(moe.num_experts, 64);
        assert_eq!(moe.num_experts_per_tok, 6);
        assert_eq!(moe.moe_intermediate_size, 1408);
        assert_eq!(moe.shared_expert_intermediate_size, 2 * 1408); // n_shared_experts · moe_inter
        assert_eq!(moe.first_k_dense_replace, 1);
        assert!(!moe.norm_topk_prob);
        assert_eq!(moe.routed_scaling_factor, 1.0);

        let yarn = cfg.yarn.unwrap();
        assert_eq!(yarn.factor, 40.0);
        assert_eq!(yarn.original_context, 4096.0);
        // The legacy llama3 `rope_scaling` path must not also fire for a yarn block.
        assert!(cfg.rope_scaling.is_none());

        // Attention scale folds in the YaRN mscale²: q_head_dim^-0.5 · (0.1·0.707·ln40 + 1)².
        let mscale = 0.1 * 0.707 * 40f32.ln() + 1.0;
        let expected = (192f32).powf(-0.5) * mscale * mscale;
        assert!(
            (cfg.attn_scale() - expected).abs() < 1e-6,
            "{}",
            cfg.attn_scale()
        );

        // The RoPE rotates only the 64-dim sub-vector, interleaved.
        let rope = cfg.build_rope();
        assert_eq!(rope.dim(), 64);
        assert!(rope.interleaved());
    }

    #[test]
    fn parses_quantization_block() {
        let v = json!({
            "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "vocab_size": 32,
            "quantization": { "group_size": 64, "bits": 4 }
        });
        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(
            cfg.quantization,
            Some(QuantSpec {
                group_size: 64,
                bits: 4,
                grid: Default::default(),
            })
        );

        // Absent block ⇒ dense snapshot.
        let dense = json!({
            "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "vocab_size": 32
        });
        assert_eq!(ModelConfig::from_json(&dense).unwrap().quantization, None);
    }

    #[test]
    fn qwen3_config_has_qk_norm_and_explicit_head_dim() {
        let v = json!({
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 28,
            "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
            "vocab_size": 151936, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
            "tie_word_embeddings": true, "max_position_embeddings": 40960
        });
        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(cfg.architecture, Architecture::Qwen3);
        assert!(cfg.has_qk_norm());
        assert_eq!(cfg.head_dim, 128); // explicit, != 1024/16
        assert_eq!(cfg.max_position_embeddings, 40960);
        assert!(cfg.rope_scaling.is_none());
    }

    #[test]
    fn qwen35_dispatches_via_text_config_and_is_recognized_for_routing() {
        // Qwen3.6 self-identifies as `qwen3_5`, wrapped as a VLM with the real decoder under
        // `text_config` (`qwen3_5_text`). Dispatch descends into text_config and must NOT mistake
        // `qwen3_5` for plain `qwen3` (a substring).
        let v = json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": { "model_type": "qwen3_5_text" },
            "vision_config": { "model_type": "qwen3_5", "depth": 27 }
        });
        assert_eq!(Architecture::from_config(&v).unwrap(), Architecture::Qwen35);
        assert_eq!(Architecture::Qwen35.family(), "qwen3_5");

        // Plain Qwen3 is unaffected (the `qwen3_5`-before-`qwen3` ordering).
        let q3 = json!({ "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3" });
        assert_eq!(Architecture::from_config(&q3).unwrap(), Architecture::Qwen3);
    }

    #[test]
    fn qwen3vl_modern_rope_parameters_match_the_flux3_tiny_snapshot() {
        let text = json!({
            "model_type":"qwen3_vl_text", "hidden_size":32, "intermediate_size":48,
            "num_hidden_layers":36, "num_attention_heads":4, "num_key_value_heads":2,
            "head_dim":16, "vocab_size":28, "rms_norm_eps":1e-6, "rope_theta":123456,
            "rope_parameters":{"rope_type":"default","rope_theta":5000000,
                "mrope_section":[4,2,2],"mrope_interleaved":true}
        });
        for value in [&text, &json!({"model_type":"qwen3_vl","text_config":text})] {
            let cfg = ModelConfig::from_json(value).unwrap();
            assert_eq!(
                cfg.rope_theta, 5_000_000.0,
                "selected modern theta wins root legacy theta"
            );
            assert_eq!(cfg.mrope_section, Some([4, 2, 2]));
            assert_eq!(cfg.rotary_dim(), 16);
            assert!(cfg.rope_scaling.is_none());
            assert!(cfg.yarn.is_none());
        }
        let mut malformed = text.clone();
        malformed["rope_parameters"]["rope_theta"] = json!("not numeric");
        assert!(ModelConfig::from_json(&malformed)
            .unwrap_err()
            .to_string()
            .contains("rope_theta"));
        let mut unrelated = text;
        unrelated["model_type"] = json!("llama");
        unrelated["rope_parameters"] = json!("not a uniform Qwen3-VL config");
        assert_eq!(
            ModelConfig::from_json(&unrelated).unwrap().rope_theta,
            123456.0
        );
    }

    #[test]
    fn qwen3vl_dispatches_and_parses_nested_text_config() {
        // Qwen3-VL self-identifies as `qwen3_vl`, wrapped as a VLM with the standard full-attention
        // Qwen3 text decoder under `text_config` (`qwen3_vl_text`). The dispatch must pick Qwen3-VL
        // (NOT the bare `qwen3` substring, NOT the hybrid `qwen3_5`), and the config must parse the
        // decoder fields out of `text_config` along with the 256K RoPE and interleaved M-RoPE section.
        // These mirror the cached Qwen/Qwen3-VL-8B-Instruct rev 0c351dd0 `text_config`.
        let v = json!({
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "image_token_id": 151655,
            "model_type": "qwen3_vl",
            "tie_word_embeddings": false,
            "video_token_id": 151656,
            "vision_start_token_id": 151652,
            "vision_end_token_id": 151653,
            "text_config": {
                "model_type": "qwen3_vl_text",
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "rope_theta": 5000000,
                "max_position_embeddings": 262144,
                "rope_scaling": { "mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default" }
            },
            "vision_config": { "model_type": "qwen3_vl", "depth": 27 }
        });
        assert_eq!(
            Architecture::from_config(&v).unwrap(),
            Architecture::Qwen3Vl
        );
        assert_eq!(Architecture::Qwen3Vl.family(), "qwen3_vl");
        assert!(Architecture::Qwen3Vl.is_qwen3_vl());
        assert!(Architecture::Qwen3Vl.has_qk_norm());

        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(cfg.architecture, Architecture::Qwen3Vl);
        // Decoder shape read from text_config.
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 12288);
        assert_eq!(cfg.num_layers, 36);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.groups(), 4); // GQA 32/8
        assert_eq!(cfg.head_dim, 128); // explicit, != 4096/32
        assert_eq!(cfg.vocab_size, 151936);
        assert!(cfg.has_qk_norm());
        assert!(!cfg.tie_word_embeddings, "Qwen3-VL lm_head is untied");
        // RoPE / 256K context.
        assert_eq!(cfg.rope_theta, 5_000_000.0);
        assert_eq!(cfg.max_position_embeddings, 262144);
        assert_eq!(cfg.partial_rotary_factor, 1.0); // full rotary (rope_type "default")
        assert_eq!(cfg.rotary_dim(), 128);
        // Interleaved M-RoPE section parsed from rope_scaling.
        assert_eq!(cfg.mrope_section, Some([24, 20, 20]));
        assert_eq!(cfg.mrope_section_resolved(), [24, 20, 20]); // sums to dim/2 = 64
        assert_eq!(
            cfg.mrope_section_resolved().iter().sum::<usize>(),
            (cfg.rotary_dim() / 2) as usize
        );
        // The text-path RoPE is plain full-width NeoX (theta 5e6); not interleaved, not partial, no yarn.
        let rope = cfg.build_rope();
        assert_eq!(rope.dim(), 128);
        assert!(!rope.interleaved());
        assert!(cfg.yarn.is_none());
        assert!(cfg.rope_scaling.is_none()); // rope_type "default" is neither llama3 nor yarn
    }

    #[test]
    fn qwen35_modelconfig_declines_the_provider_routed_decoder() {
        // ModelConfig deliberately declines the hybrid decoder instead of producing a cryptic
        // missing-field error; the provider routes it through Qwen35Config.
        let v = json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": { "model_type": "qwen3_5_text", "hidden_size": 5120 },
            "vision_config": { "model_type": "qwen3_5" }
        });
        match ModelConfig::from_json(&v) {
            Err(Error::Unsupported(m)) => {
                assert!(m.contains("qwen3_5"), "{m}");
                assert!(m.contains("Qwen35Config"), "{m}");
                assert!(!m.contains("not yet implemented"), "{m}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn gemma4_dispatch_orders_unified_before_the_bare_family() {
        // `gemma4` is a substring of `gemma4_unified`, so the unified check must come first or every
        // unified checkpoint silently loads as the dense enhance-only variant. Both spellings the
        // checkpoints use — top-level `gemma4_unified` and the nested `gemma4_unified_text` — must
        // land on the same variant.
        let unified = json!({
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "model_type": "gemma4_unified",
            "text_config": { "model_type": "gemma4_unified_text" }
        });
        let a = Architecture::from_config(&unified).unwrap();
        assert_eq!(a, Architecture::Gemma4Unified);
        assert_eq!(a.family(), "gemma4_unified");

        let dense = json!({ "architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4" });
        let d = Architecture::from_config(&dense).unwrap();
        assert_eq!(d, Architecture::Gemma4);
        assert_eq!(d.family(), "gemma4");

        // Gemma-2 must still dispatch to Gemma-2 (and keep its `(1 + weight)` norm fold).
        let g2 = json!({ "architectures": ["Gemma2ForCausalLM"], "model_type": "gemma2" });
        let g2 = Architecture::from_config(&g2).unwrap();
        assert_eq!(g2, Architecture::Gemma2);
        assert!(g2.norm_unit_offset());

        for g4 in [a, d] {
            assert!(g4.is_gemma4());
            assert!(g4.is_gemma());
            assert!(
                !g4.is_gemma2(),
                "Gemma 4 must not answer to the Gemma-2 flag"
            );
            assert!(
                !g4.norm_unit_offset(),
                "Gemma 4 norms use the stored weight verbatim"
            );
            assert!(g4.is_sandwich());
            assert!(g4.has_qk_norm());
            assert!(g4.has_v_norm());
            assert!(g4.nests_text_config());
            assert!(!g4.rope_interleaved());
            assert!(!g4.is_mla());
        }
    }

    #[test]
    fn gemma4_minimal_config_defaults_match_upstream() {
        // A Gemma 4 text config carrying none of the optional keys must still land on upstream's
        // defaults rather than this crate's generic ones — in particular `rope_theta`, whose generic
        // default (500 000) belongs to no Gemma model, and the 512-wide `global_head_dim`.
        let v = json!({
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "model_type": "gemma4_unified",
            "text_config": {
                "model_type": "gemma4_unified_text",
                "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 12,
                "num_attention_heads": 4, "vocab_size": 256
            }
        });
        let cfg = ModelConfig::from_json(&v).unwrap();
        let g = cfg.gemma4.as_ref().expect("gemma 4 table");
        assert_eq!(g.sliding.rope_theta, 10_000.0);
        assert_eq!(
            g.sliding.head_dim, 16,
            "hidden/heads when head_dim is absent"
        );
        assert_eq!(g.full.rope_theta, 1_000_000.0);
        assert_eq!(g.full.rope_type, RopeType::Proportional);
        assert_eq!(g.full.partial_rotary_factor, 0.25);
        assert_eq!(g.full.head_dim, 512, "global_head_dim defaults to 512");
        assert_eq!(g.sliding.sliding_window, Some(1024));
        assert!(!g.full.k_eq_v, "attention_k_eq_v defaults to false");
        assert_eq!(g.num_kv_shared_layers, 0);
        assert_eq!(g.bidirectional, Some(BidirectionalAttention::Vision));
        // With `attention_k_eq_v` off, the full layers keep the ordinary KV-head count — the
        // `num_global_key_value_heads` override is gated on the flag upstream.
        assert_eq!(g.full.num_kv_heads, g.sliding.num_kv_heads);
        // The scalar view is the sliding (layer-0) one, and Gemma always ties its embeddings.
        assert_eq!(cfg.rope_theta, 10_000.0);
        assert_eq!(cfg.head_dim, 16);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.attn_scale(), 1.0);
        // 12 layers: full at indices 5 and 11 (the last is forced full either way).
        let full: Vec<usize> = (0..12)
            .filter(|&i| cfg.layer_type(i) == LayerAttentionType::Full)
            .collect();
        assert_eq!(full, vec![5, 11]);
    }
}
