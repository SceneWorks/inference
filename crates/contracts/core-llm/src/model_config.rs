//! Tensor-free decoder configuration facts parsed from a Hugging Face `config.json`, shared by the
//! MLX and Candle engines (sc-27255, sc-27308).
//!
//! Both engines used to carry a near-identical value-based `config.rs` whose copies had drifted:
//! architecture dispatch, the Qwen3.5 spelling, quantization-block parsing and the Gemma 4
//! KV-sharing helpers differed. The backend-neutral parts live here: [`Architecture`] dispatch, the
//! Gemma 4 per-layer-type attention table ([`Gemma4Config`]) with its KV-sharing helpers, the MoE /
//! MLA / YaRN / Llama-3 RoPE-scaling blocks, the multimodal-RoPE section and the neutral
//! [`RawQuantization`] block. Each engine keeps a thin `config.rs` that assembles its own
//! `ModelConfig` from these, converts [`RawQuantization`] into its quantization type, and builds its
//! RoPE tables (tensor/backend work this crate never does).

use serde_json::Value;

/// A `config.json` this module cannot describe. Each engine maps the two kinds onto its own error
/// enum's `Config` / `Unsupported` variants.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ModelConfigError {
    /// The config is malformed or missing a required field.
    #[error("{0}")]
    Config(String),
    /// The config names something no engine implements (an architecture, a RoPE schedule).
    #[error("{0}")]
    Unsupported(String),
}

/// The decoder architecture, dispatched from `config.json` (`architectures` / `model_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    /// Llama family (also Mistral — same decoder shape: no q/k norm, no QKV bias).
    Llama,
    /// Qwen3 family: adds per-head q/k RMSNorm in attention.
    Qwen3,
    /// Phi-3 family: the Llama decoder shape, but with a **packed** `qkv_proj` (q‖k‖v) and a packed
    /// `gate_up_proj` (gate‖up) — split at load into the standard projections.
    Phi3,
    /// Qwen2-MoE family: Qwen2 attention (GQA **with q/k/v bias**, no q/k norm) + a sparse
    /// mixture-of-experts FFN (router + top-k experts + a shared expert).
    Qwen2Moe,
    /// Gemma-2 family: `(1 + weight)` RMSNorm, embedding ×√hidden, GeGLU, a `query_pre_attn_scalar`
    /// attention scale, attention- and final-logit soft-capping, and a 4-norm "sandwich" block.
    Gemma2,
    /// GLM-4 family: a 4-norm sandwich block (standard RMSNorm), q/k/v bias, a packed `gate_up_proj`,
    /// and **partial, interleaved** RoPE (only `head_dim · partial_rotary_factor` dims are rotated).
    Glm4,
    /// DeepSeek-V2 family: **Multi-head Latent Attention (MLA)** — a low-rank compressed KV path
    /// (`kv_a`/`kv_b` projections, a decoupled RoPE key sub-vector) that is a distinct attention
    /// implementation rather than GQA — plus a fine-grained MoE FFN (many routed experts, several
    /// shared experts, the first layers dense) and YaRN RoPE on the `qk_rope_head_dim` sub-vector.
    DeepseekV2,
    /// Qwen3.5/3.6 (`model_type` `qwen3_5` / `qwen3_5_text`, also spelled `qwen35`, and the
    /// Qwen3-Next `qwen3_next` architecture): a **hybrid** decoder interleaving Gated DeltaNet
    /// linear-attention layers with gated full-attention layers on a fixed schedule (the 35B variant
    /// is MoE). Each backend serves it with a dedicated `Qwen35Model` / `Qwen35Config` rather than
    /// the generic full-attention decoder config.
    Qwen35,
    /// Qwen3-VL (`model_type` `qwen3_vl` / `text_config.model_type` `qwen3_vl_text`): a VLM-wrapped
    /// **standard full-attention** Qwen3 decoder (GQA + per-head q/k RMSNorm + SwiGLU, standard
    /// verbatim-weight RMSNorm) — NOT the Qwen3.6 hybrid — with a 256K context (`rope_theta` 5e6) and
    /// **interleaved multimodal RoPE** (`rope_scaling.mrope_section`). The decoder weights nest under
    /// `model.language_model.*` (the ViT tower under `model.visual.*`). Served by each backend's
    /// generic decoder; the text path is plain 1-D RoPE (interleaved M-RoPE with equal t/h/w rows is
    /// bit-identical), the image path uses the interleaved-M-RoPE cos/sin table.
    Qwen3Vl,
    /// Gemma 4 **unified** (`model_type` `gemma4_unified` / `gemma4_unified_text`): the
    /// encode-capable multimodal Gemma 4, whose text decoder is the LTX-2.5 text encoder. Structurally
    /// a Gemma-2 sandwich block, but with a **per-layer-type** attention table
    /// ([`Gemma4Config`]) rather than one uniform shape: `layer_types` alternates
    /// `sliding_attention` / `full_attention`, and the two types differ in head dim, KV-head count,
    /// RoPE schedule (`default` vs `proportional`), masking (sliding window vs full causal), and
    /// whether K and V share a projection (`attention_k_eq_v`). Norms are plain-`weight` RMSNorm —
    /// **not** Gemma-2's `(1 + weight)` fold — and the attention scale is `1.0` (the per-head q/k
    /// RMSNorms take its place).
    Gemma4Unified,
    /// Gemma 4 **dense** (`model_type` `gemma4` / `gemma4_text`): the text-only Gemma 4 upstream
    /// distinguishes from the unified variant. Same decoder shape and per-layer-type attention
    /// table as [`Architecture::Gemma4Unified`]; LTX-2.5 uses it only as a prompt *enhancer*, never
    /// as the encoder (the packed TE always carries the unified config).
    Gemma4,
}

/// Which attention flavour a decoder layer runs — Gemma 4's `layer_types` entries.
///
/// Every architecture before Gemma 4 is uniform (one shape for all layers), which is
/// [`LayerAttentionType::Full`] for the whole stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerAttentionType {
    /// `sliding_attention`: the layer only attends the `sliding_window` most recent keys.
    Sliding,
    /// `full_attention`: the layer attends the whole (causal) prefix.
    Full,
}

impl LayerAttentionType {
    /// The `config.json` spelling (`"sliding_attention"` / `"full_attention"`).
    pub fn as_str(self) -> &'static str {
        match self {
            LayerAttentionType::Sliding => "sliding_attention",
            LayerAttentionType::Full => "full_attention",
        }
    }

    /// Parse a `layer_types` entry; unrecognized spellings are `None`.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "sliding_attention" => Some(LayerAttentionType::Sliding),
            "full_attention" => Some(LayerAttentionType::Full),
            _ => None,
        }
    }
}

/// How a layer builds its RoPE inverse frequencies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeType {
    /// `rope_type: "default"` — `inv_freq[i] = theta^(-2i / head_dim)` over the full head.
    Default,
    /// `rope_type: "proportional"` — the leading `int(partial_rotary_factor · head_dim / 2)`
    /// channels carry `theta^(-2i / head_dim)` (denominator the **full** head dim) and the rest are
    /// exactly zero, so the table stays `head_dim` wide and the un-rotated channels are an identity
    /// rotation. Distinct from a leading-slice partial RoPE (GLM-4), which narrows both the
    /// denominator and the rotated span. New in Gemma 4.
    Proportional,
}

/// One layer type's attention descriptor — the per-layer replacement for the scalar `head_dim` /
/// `num_kv_heads` / `rope_theta` triple.
///
/// Uniform architectures resolve every layer to the same descriptor (each backend's
/// `ModelConfig::layer_attention`); Gemma 4 resolves `sliding_attention` and `full_attention`
/// layers to two different ones. Building the layer's RoPE table is backend work.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerAttention {
    /// Per-head dimension for this layer type (Gemma 4: 256 sliding, `global_head_dim` 512 full).
    pub head_dim: i32,
    /// Key/value head count for this layer type (Gemma 4: `num_key_value_heads` sliding,
    /// `num_global_key_value_heads` full).
    pub num_kv_heads: i32,
    /// The RoPE frequency schedule.
    pub rope_type: RopeType,
    /// RoPE base frequency (Gemma 4: 10 000 sliding, 1 000 000 full).
    pub rope_theta: f32,
    /// Fraction of `head_dim` that carries a rotation; `1.0` ⇒ every channel.
    pub partial_rotary_factor: f32,
    /// `rope_parameters.<type>.factor` — the linear divisor applied to the inverse frequencies
    /// (`inv_freq /= factor`) at the end of the reference's proportional schedule. `1.0` ⇒ no
    /// scaling, which is what every shipped Gemma 4 config carries.
    pub rope_factor: f32,
    /// `Some(w)` ⇒ the layer attends only the `w` most recent keys (inclusive of the query's own
    /// position); `None` ⇒ the full causal prefix.
    pub sliding_window: Option<i32>,
    /// `attention_k_eq_v`: K and V come from **one** shared projection (the value path takes the raw
    /// projection output and its own scale-free norm, so K and V are still different tensors).
    pub k_eq_v: bool,
}

impl LayerAttention {
    /// GQA group count for this layer (`num_heads / num_kv_heads`).
    pub fn groups(&self, num_heads: i32) -> i32 {
        num_heads / self.num_kv_heads.max(1)
    }
}

/// `use_bidirectional_attention`: which token spans attend bidirectionally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BidirectionalAttention {
    /// `"vision"` — vision tokens are bidirectional, text stays causal (the Gemma 4 default, and the
    /// only mode a text-only encoder ever sees).
    Vision,
    /// `"all"` — every token attends bidirectionally. Upstream also halves the configured
    /// `sliding_window` to `w / 2 + 1` when this is set (exclusive bounds), which
    /// [`Gemma4Config`] parsing reproduces.
    All,
}

/// Gemma 4's per-layer-type attention table.
///
/// `layer_types` has one entry per decoder layer; each entry selects [`Gemma4Config::sliding`] or
/// [`Gemma4Config::full`]. This is the block that makes a single decoder config able to describe a
/// model whose layers do not share a head dim, a KV-head count, a RoPE schedule, or a mask.
#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4Config {
    /// One entry per decoder layer (length `num_layers`).
    pub layer_types: Vec<LayerAttentionType>,
    /// The `sliding_attention` layers' descriptor.
    pub sliding: LayerAttention,
    /// The `full_attention` layers' descriptor.
    pub full: LayerAttention,
    /// `use_bidirectional_attention`; `None` when the config disables it outright.
    pub bidirectional: Option<BidirectionalAttention>,
    /// `num_kv_shared_layers`: how many trailing layers reuse the last non-sharing layer's K/V of
    /// their own layer type instead of projecting their own. `0` ⇒ no sharing.
    pub num_kv_shared_layers: usize,
    /// `use_double_wide_mlp`: the KV-sharing tail layers double `intermediate_size`.
    pub use_double_wide_mlp: bool,
}

/// Upstream's default sliding/full schedule period: every `6`th layer is full attention (5:1).
const GEMMA4_SLIDING_WINDOW_PATTERN: usize = 6;

impl Gemma4Config {
    /// The layer type of decoder layer `i` (out-of-range indices clamp to the last entry, matching
    /// how the decoder only ever asks about layers it built).
    pub fn layer_type(&self, layer_idx: usize) -> LayerAttentionType {
        self.layer_types
            .get(layer_idx)
            .copied()
            .or_else(|| self.layer_types.last().copied())
            .unwrap_or(LayerAttentionType::Full)
    }

    /// The attention descriptor of decoder layer `i`.
    pub fn layer(&self, layer_idx: usize) -> LayerAttention {
        match self.layer_type(layer_idx) {
            LayerAttentionType::Sliding => self.sliding,
            LayerAttentionType::Full => self.full,
        }
    }

    /// The descriptor for a layer *type*.
    pub fn for_type(&self, kind: LayerAttentionType) -> LayerAttention {
        match kind {
            LayerAttentionType::Sliding => self.sliding,
            LayerAttentionType::Full => self.full,
        }
    }

    /// `first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers`, or `None` when the
    /// model has no sharing tail. Upstream guards on `num_kv_shared_layers > 0`, so a `0` setting
    /// must **not** mark every layer shared (`len - 0 == len` would already say that, but the guard
    /// is what makes the intent explicit).
    pub fn first_kv_shared_layer(&self) -> Option<usize> {
        (self.num_kv_shared_layers > 0).then(|| {
            self.layer_types
                .len()
                .saturating_sub(self.num_kv_shared_layers)
        })
    }

    /// Whether decoder layer `i` is a **KV-sharing** layer: it projects no keys or values of its
    /// own (`k_proj` / `v_proj` / `k_norm` are absent from the checkpoint) and attends the stored
    /// K/V of [`Gemma4Config::kv_donor`] instead.
    pub fn is_kv_shared(&self, layer_idx: usize) -> bool {
        self.first_kv_shared_layer().is_some_and(|f| layer_idx >= f)
    }

    /// Whether layer `i` must **publish** its keys/values for the sharing tail — upstream's
    /// `store_full_length_kv`: the last layer of each type before `first_kv_shared_layer_idx`.
    pub fn stores_shared_kv(&self, layer_idx: usize) -> bool {
        let Some(first) = self.first_kv_shared_layer() else {
            return false;
        };
        if layer_idx >= first {
            return false;
        }
        let kind = self.layer_type(layer_idx);
        // The last layer of this type strictly before the tail — i.e. no later pre-tail layer
        // shares its type.
        !(layer_idx + 1..first).any(|j| self.layer_type(j) == kind)
    }

    /// The layer whose stored K/V layer `i` attends, or `None` when `i` projects its own: the last
    /// pre-tail layer of `i`'s **own type** (a sliding layer never reuses a full layer's keys —
    /// their head dims do not even match).
    pub fn kv_donor(&self, layer_idx: usize) -> Option<usize> {
        if !self.is_kv_shared(layer_idx) {
            return None;
        }
        let first = self.first_kv_shared_layer()?;
        let kind = self.layer_type(layer_idx);
        (0..first).rev().find(|&j| self.layer_type(j) == kind)
    }

    /// This layer's MLP inner width given the model's `intermediate_size` — doubled on the
    /// KV-sharing tail when `use_double_wide_mlp` is set, which is the only place upstream applies
    /// it (the layers that saved a K/V projection spend the parameters on a wider MLP instead).
    pub fn layer_intermediate_size(&self, layer_idx: usize, intermediate_size: i32) -> i32 {
        match self.use_double_wide_mlp && self.is_kv_shared(layer_idx) {
            true => intermediate_size * 2,
            false => intermediate_size,
        }
    }

    /// Resolve `layer_types`: the explicit `config.json` array when present, else upstream's default
    /// 5:1 schedule (`sliding_attention` unless `(i + 1) % 6 == 0`). Either way the **last** layer is
    /// forced to `full_attention`, as upstream does.
    pub fn resolve_layer_types(
        num_layers: usize,
        explicit: Option<&Value>,
    ) -> Vec<LayerAttentionType> {
        let mut types: Vec<LayerAttentionType> = match explicit.and_then(|v| v.as_array()) {
            Some(a) if a.len() >= num_layers => a
                .iter()
                .take(num_layers)
                .map(|e| {
                    e.as_str()
                        .and_then(LayerAttentionType::parse)
                        .unwrap_or(LayerAttentionType::Full)
                })
                .collect(),
            _ => (0..num_layers)
                .map(|i| {
                    if (i + 1) % GEMMA4_SLIDING_WINDOW_PATTERN == 0 {
                        LayerAttentionType::Full
                    } else {
                        LayerAttentionType::Sliding
                    }
                })
                .collect(),
        };
        if let Some(last) = types.last_mut() {
            *last = LayerAttentionType::Full;
        }
        types
    }

    /// Parse the Gemma 4 block out of an (already `text_config`-descended) config value.
    /// `num_layers` / `num_kv_heads` / `head_dim` are the scalars the backend's decoder config has
    /// already read, which double as the `sliding_attention` defaults.
    pub fn from_json(
        v: &Value,
        num_layers: usize,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> Result<Self, ModelConfigError> {
        let int =
            |key: &str| -> Option<i32> { v.get(key).and_then(|x| x.as_i64()).map(|x| x as i32) };
        let bidirectional = match v
            .get("use_bidirectional_attention")
            .and_then(|x| x.as_str())
        {
            Some("all") => Some(BidirectionalAttention::All),
            Some("vision") => Some(BidirectionalAttention::Vision),
            // A JSON `null` / absent key defaults to upstream's `"vision"`; any other spelling is
            // treated as "not bidirectional" rather than guessed at.
            None if v.get("use_bidirectional_attention").is_none() => {
                Some(BidirectionalAttention::Vision)
            }
            _ => None,
        };
        // Upstream halves the stored window (`w / 2 + 1`, exclusive bounds) when every token is
        // bidirectional. Serialization undoes it, so the stored value is always the un-halved one.
        let raw_window = int("sliding_window").unwrap_or(1024);
        let sliding_window = if bidirectional == Some(BidirectionalAttention::All) {
            raw_window / 2 + 1
        } else {
            raw_window
        };
        let k_eq_v = v
            .get("attention_k_eq_v")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);

        // `rope_parameters` is keyed by layer type; both entries default to upstream's table.
        // Fallible: an unsupported `factor` combination is refused rather than dropped.
        let rope_for = |kind: LayerAttentionType,
                        default_theta: f32,
                        default_type: RopeType,
                        default_partial: f32| {
            let block = v
                .get("rope_parameters")
                .and_then(|rp| rp.get(kind.as_str()));
            let rope_type = match block
                .and_then(|b| b.get("rope_type"))
                .and_then(|x| x.as_str())
            {
                Some("proportional") => RopeType::Proportional,
                Some(_) => RopeType::Default,
                None => default_type,
            };
            let theta = block
                .and_then(|b| b.get("rope_theta"))
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(default_theta);
            let partial = block
                .and_then(|b| b.get("partial_rotary_factor"))
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(default_partial);
            // `factor` divides the inverse frequencies at the end of the *proportional* schedule
            // only. Gemma 4's own `compute_default_rope_parameters` has no factor term, so a
            // `default` layer type carrying one would mean a schedule this crate does not
            // implement — refuse it rather than silently ignoring the key.
            let factor = block
                .and_then(|b| b.get("factor"))
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(1.0);
            if rope_type == RopeType::Default && factor != 1.0 {
                return Err(ModelConfigError::Unsupported(format!(
                    "rope_parameters.{}.factor = {factor} on a `default` rope_type: the reference \
                     applies `factor` only to the `proportional` schedule, so this config asks for \
                     a frequency scaling this decoder does not implement",
                    kind.as_str()
                )));
            }
            Ok((rope_type, theta, partial, factor))
        };
        let (sliding_rope, sliding_theta, sliding_partial, sliding_factor) = rope_for(
            LayerAttentionType::Sliding,
            10_000.0,
            RopeType::Default,
            1.0,
        )?;
        let (full_rope, full_theta, full_partial, full_factor) = rope_for(
            LayerAttentionType::Full,
            1_000_000.0,
            RopeType::Proportional,
            0.25,
        )?;

        // The `full_attention` layers' overrides (`per_layer_config` upstream): a wider head, and —
        // only when `attention_k_eq_v` is set — a different KV-head count.
        let global_head_dim = int("global_head_dim").unwrap_or(512);
        let global_kv_heads = if k_eq_v {
            int("num_global_key_value_heads").unwrap_or(num_kv_heads)
        } else {
            num_kv_heads
        };

        Ok(Self {
            layer_types: Self::resolve_layer_types(num_layers, v.get("layer_types")),
            sliding: LayerAttention {
                head_dim,
                num_kv_heads,
                rope_type: sliding_rope,
                rope_theta: sliding_theta,
                partial_rotary_factor: sliding_partial,
                rope_factor: sliding_factor,
                sliding_window: Some(sliding_window),
                // `attention_k_eq_v` gates only the non-sliding layers upstream
                // (`use_alternative_attention = attention_k_eq_v and not is_sliding`).
                k_eq_v: false,
            },
            full: LayerAttention {
                head_dim: global_head_dim,
                num_kv_heads: global_kv_heads,
                rope_type: full_rope,
                rope_theta: full_theta,
                partial_rotary_factor: full_partial,
                rope_factor: full_factor,
                sliding_window: None,
                k_eq_v,
            },
            bidirectional,
            num_kv_shared_layers: int("num_kv_shared_layers").unwrap_or(0).max(0) as usize,
            use_double_wide_mlp: v
                .get("use_double_wide_mlp")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
        })
    }
}

impl Architecture {
    /// Determine the architecture from a parsed `config.json`. A config with no `architectures` /
    /// `model_type` (e.g. a minimal synthetic config) defaults to [`Architecture::Llama`]; a config
    /// that names an unrecognized architecture is rejected.
    ///
    /// A VLM wrapper (LLaVA, Qwen-VL, Qwen3.5, Gemma 4) nests the language decoder under
    /// `text_config`; dispatch reads that nested decoder's `architectures` / `model_type` when it
    /// names either, else the top level (sc-27255: the MLX rule, shared by both backends).
    pub fn from_config(v: &Value) -> Result<Self, ModelConfigError> {
        let names_decoder =
            |cfg: &Value| cfg.get("architectures").is_some() || cfg.get("model_type").is_some();
        let cfg = v
            .get("text_config")
            .filter(|nested| names_decoder(nested))
            .unwrap_or(v);
        let arch = cfg
            .get("architectures")
            .and_then(|a| a.as_array())
            .and_then(|a| a.first())
            .and_then(|s| s.as_str());
        let model_type = cfg.get("model_type").and_then(|s| s.as_str());
        let hay = format!(
            "{} {}",
            arch.unwrap_or("").to_lowercase(),
            model_type.unwrap_or("").to_lowercase()
        );
        if hay.contains("qwen3_vl") || hay.contains("qwen3vl") {
            // Qwen3-VL (`qwen3_vl` / `qwen3_vl_text`) — the generic full-attention decoder. Checked
            // before `qwen3_5`/`qwen3` (all contain "qwen3"); distinct from "qwen3_5".
            Ok(Architecture::Qwen3Vl)
        } else if hay.contains("qwen3_5") || hay.contains("qwen35") || hay.contains("qwen3_next") {
            // Qwen3.5/3.6 (`qwen3_5`, MLX's `qwen35` spelling, Qwen3-Next) — the hybrid
            // Gated-DeltaNet decoder. Checked before `qwen3`, a substring of each (sc-27255).
            Ok(Architecture::Qwen35)
        } else if hay.contains("qwen3") {
            Ok(Architecture::Qwen3)
        } else if hay.contains("qwen2_moe") || hay.contains("qwen2moe") {
            Ok(Architecture::Qwen2Moe)
        } else if hay.contains("gemma4_unified") || hay.contains("gemma4unified") {
            // `gemma4_unified` must be tested before the bare `gemma4` (a substring of it).
            Ok(Architecture::Gemma4Unified)
        } else if hay.contains("gemma4") {
            Ok(Architecture::Gemma4)
        } else if hay.contains("gemma2") {
            Ok(Architecture::Gemma2)
        } else if hay.contains("glm4") {
            Ok(Architecture::Glm4)
        } else if hay.contains("deepseek") {
            Ok(Architecture::DeepseekV2)
        } else if hay.contains("phi3") {
            Ok(Architecture::Phi3)
        } else if hay.contains("llama")
            || hay.contains("mistral")
            || hay.contains("qwen2")
            || (arch.is_none() && model_type.is_none())
        {
            // Llama / Mistral / dense Qwen2 share the decoder shape — Qwen2 only adds q/k/v bias,
            // which the projection loader auto-detects by weight-key presence (no q/k norm, standard
            // RoPE, SwiGLU). A minimal config (no arch fields) also defaults here.
            Ok(Architecture::Llama)
        } else {
            Err(ModelConfigError::Unsupported(format!(
                "unsupported architecture (architectures={arch:?}, model_type={model_type:?})"
            )))
        }
    }

    /// The model-family tag.
    pub fn family(self) -> &'static str {
        match self {
            Architecture::Llama => "llama",
            Architecture::Qwen3 => "qwen3",
            Architecture::Phi3 => "phi3",
            Architecture::Qwen2Moe => "qwen2_moe",
            Architecture::Gemma2 => "gemma2",
            Architecture::Glm4 => "glm4",
            Architecture::DeepseekV2 => "deepseek_v2",
            Architecture::Qwen35 => "qwen3_5",
            Architecture::Qwen3Vl => "qwen3_vl",
            Architecture::Gemma4Unified => "gemma4_unified",
            Architecture::Gemma4 => "gemma4",
        }
    }

    /// Whether this is the Qwen3-VL decoder (drives the `text_config` descent, the
    /// `model.language_model.*` weight prefix, and the interleaved multimodal RoPE).
    pub fn is_qwen3_vl(self) -> bool {
        matches!(self, Architecture::Qwen3Vl)
    }

    /// Whether this is a Gemma-2 decoder (drives `(1+weight)` norms, embedding scaling, GeGLU, the
    /// sandwich-norm block, and logit soft-capping).
    pub fn is_gemma2(self) -> bool {
        matches!(self, Architecture::Gemma2)
    }

    /// Whether this is a Gemma 4 decoder — unified (encode-capable) or dense (enhance-only). Drives
    /// the per-layer-type attention table ([`Gemma4Config`]), the `text_config` descent, the
    /// unit attention scale, and the scale-free value norm.
    pub fn is_gemma4(self) -> bool {
        matches!(self, Architecture::Gemma4Unified | Architecture::Gemma4)
    }

    /// Whether this is any Gemma decoder. Drives the two behaviours **all** Gemma generations share:
    /// the `√hidden` embedding scale and the GeGLU (`gelu_pytorch_tanh`) MLP, plus the implicit
    /// `tie_word_embeddings` default. Deliberately distinct from [`Architecture::norm_unit_offset`]
    /// — Gemma 4 shares the scale and the activation but **not** the `(1 + weight)` norm fold.
    pub fn is_gemma(self) -> bool {
        self.is_gemma2() || self.is_gemma4()
    }

    /// Whether RMSNorm weights are stored offset by one, so the norm is `(1 + weight) · x̂`.
    ///
    /// **Gemma-2 only.** Gemma 4's `Gemma4UnifiedRMSNorm` multiplies by the stored `weight`
    /// directly (its weights initialize to ones, not zeros), so folding `+1` into a Gemma 4
    /// checkpoint would corrupt every norm in the model.
    pub fn norm_unit_offset(self) -> bool {
        matches!(self, Architecture::Gemma2)
    }

    /// Whether attention applies a **scale-free** per-head RMSNorm to the value heads (Gemma 4's
    /// `v_norm`, `with_scale=False`). Gemma 4 only.
    pub fn has_v_norm(self) -> bool {
        self.is_gemma4()
    }

    /// Whether the decoder fields live under a `text_config` VLM wrapper rather than at the top
    /// level (Qwen3-VL, and both Gemma 4 variants).
    pub fn nests_text_config(self) -> bool {
        self.is_qwen3_vl() || self.is_gemma4()
    }

    /// Whether the block uses the 4-norm "sandwich" residual (Gemma-2, Gemma 4, GLM-4) rather than
    /// the plain Llama pre-norm.
    pub fn is_sandwich(self) -> bool {
        matches!(
            self,
            Architecture::Gemma2
                | Architecture::Glm4
                | Architecture::Gemma4Unified
                | Architecture::Gemma4
        )
    }

    /// Whether RoPE uses the interleaved (GPT-J-style) pairing rather than NeoX half-split (GLM-4,
    /// and the DeepSeek MLA rope sub-vector — DeepSeek's de-interleave-then-rotate is equivalent to
    /// the interleaved convention applied to the raw projection).
    pub fn rope_interleaved(self) -> bool {
        matches!(self, Architecture::Glm4 | Architecture::DeepseekV2)
    }

    /// Whether attention uses Multi-head Latent Attention (DeepSeek-V2) rather than GQA.
    pub fn is_mla(self) -> bool {
        matches!(self, Architecture::DeepseekV2)
    }

    /// Whether attention applies per-head q/k RMSNorm (Qwen3 / Qwen3-VL / Gemma 4).
    pub fn has_qk_norm(self) -> bool {
        matches!(
            self,
            Architecture::Qwen3
                | Architecture::Qwen3Vl
                | Architecture::Gemma4Unified
                | Architecture::Gemma4
        )
    }
}

/// `rope_scaling` parameters for the Llama-3 NTK-by-parts schedule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeScaling {
    /// Scaling factor (e.g. 8.0 for Llama-3.1).
    pub factor: f32,
    /// Low-frequency factor.
    pub low_freq_factor: f32,
    /// High-frequency factor.
    pub high_freq_factor: f32,
    /// Original (pre-scaling) max context.
    pub original_context: f32,
}

/// Mixture-of-Experts FFN parameters (Qwen2-MoE, DeepSeek-V2). Present when a layer's dense MLP is
/// replaced by a router + routed expert bank + shared expert(s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeConfig {
    /// Total number of routed experts (`num_experts` / `n_routed_experts`).
    pub num_experts: usize,
    /// Experts activated per token (top-k routing).
    pub num_experts_per_tok: usize,
    /// Inner width of each routed expert's SwiGLU.
    pub moe_intermediate_size: i32,
    /// Inner width of the always-on shared expert's SwiGLU (DeepSeek packs `n_shared_experts` shared
    /// experts into one MLP, so this is `n_shared_experts · moe_intermediate_size`).
    pub shared_expert_intermediate_size: i32,
    /// Whether the top-k routing weights are renormalized to sum to 1.
    pub norm_topk_prob: bool,
    /// Multiplier applied to the routed-expert weights when they are *not* renormalized (DeepSeek's
    /// `routed_scaling_factor`; `1.0` for Qwen2-MoE).
    pub routed_scaling_factor: f32,
    /// Number of leading layers that stay a dense MLP before the MoE layers begin (DeepSeek's
    /// `first_k_dense_replace`; `0` for Qwen2-MoE — every layer is MoE).
    pub first_k_dense_replace: usize,
}

/// Multi-head Latent Attention parameters (DeepSeek-V2). Present when attention uses a low-rank
/// compressed KV path instead of GQA.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlaConfig {
    /// Low-rank dimension of the query down-projection (`q_a_proj`); `None` ⇒ a full `q_proj`
    /// (DeepSeek-V2-Lite has no query LoRA).
    pub q_lora_rank: Option<i32>,
    /// Low-rank dimension of the shared KV down-projection (`kv_a_proj_with_mqa` → `kv_a_layernorm`).
    pub kv_lora_rank: i32,
    /// Per-head non-rotary (content) query/key dimension.
    pub qk_nope_head_dim: i32,
    /// Per-head rotary (decoupled-RoPE) query/key dimension.
    pub qk_rope_head_dim: i32,
    /// Per-head value dimension (may differ from the q/k head dim).
    pub v_head_dim: i32,
}

impl MlaConfig {
    /// The full per-head query/key dimension attended over: `qk_nope_head_dim + qk_rope_head_dim`.
    pub fn q_head_dim(&self) -> i32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

/// YaRN RoPE-scaling parameters (DeepSeek-V2). A wavelength-ramped blend of extrapolated
/// (high-frequency) and interpolated (low-frequency, divided by `factor`) inverse frequencies, plus
/// an attention-softmax magnitude scale (`mscale`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YarnConfig {
    /// Context-extension factor (interpolated frequencies are divided by this).
    pub factor: f32,
    /// Fast-rotation boundary (dims with shorter wavelength are extrapolated unchanged).
    pub beta_fast: f32,
    /// Slow-rotation boundary (dims with longer wavelength are interpolated).
    pub beta_slow: f32,
    /// Original (pre-extension) max context.
    pub original_context: f32,
    /// Magnitude scale for the cos/sin tables.
    pub mscale: f32,
    /// Magnitude scale applied to the attention softmax scale (`mscale_all_dim`).
    pub mscale_all_dim: f32,
}

impl YarnConfig {
    /// YaRN attention-softmax magnitude scale: `0.1 · mscale_all_dim · ln(factor) + 1` (`1.0` when
    /// `factor ≤ 1`). The attention scale is multiplied by this **squared**.
    pub fn softmax_mscale(&self) -> f32 {
        if self.factor > 1.0 {
            0.1 * self.mscale_all_dim * self.factor.ln() + 1.0
        } else {
            1.0
        }
    }
}

/// A `quantization` block's tensor-free facts: the bit width and group size a snapshot's
/// projections are (or are to be) packed with. Each engine converts this into its own
/// quantization type and refuses (loads dense) any width it does not implement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawQuantization {
    /// Bits per quantized weight.
    pub bits: u32,
    /// Weights per quantization group.
    pub group_size: usize,
}

impl RawQuantization {
    /// Group size when a block names only `bits` (the `prepare` snapshot writer stamps
    /// `{ "bits": 4 | 8 }`).
    pub const DEFAULT_GROUP_SIZE: usize = 64;

    /// The snapshot's quantization block: top-level first (a prepared dense snapshot), else nested
    /// beside the decoder fields in `decoder` (an MLX affine packed text encoder). `bits` is
    /// required; `group_size` defaults to [`Self::DEFAULT_GROUP_SIZE`]. `None` ⇒ a dense snapshot.
    pub fn from_config(top: &Value, decoder: &Value) -> Option<Self> {
        let block = top
            .get("quantization")
            .or_else(|| decoder.get("quantization"))?;
        let bits = u32::try_from(block.get("bits")?.as_u64()?).ok()?;
        let group_size = block
            .get("group_size")
            .and_then(|group| group.as_u64())
            .map_or(Some(Self::DEFAULT_GROUP_SIZE), |group| {
                usize::try_from(group).ok()
            })?;
        Some(Self { bits, group_size })
    }
}

fn int(v: &Value, key: &str) -> Option<i32> {
    v.get(key).and_then(|x| x.as_i64()).map(|x| x as i32)
}

fn f32_opt(v: &Value, key: &str) -> Option<f32> {
    v.get(key).and_then(|x| x.as_f64()).map(|x| x as f32)
}

impl MoeConfig {
    /// The Mixture-of-Experts block of an (already `text_config`-descended) config: present iff a
    /// routed-expert count is configured (`num_experts` for Qwen2-MoE, `n_routed_experts` for
    /// DeepSeek-V2). `intermediate_size` is the dense MLP width, the default for the expert widths.
    pub fn from_json(v: &Value, intermediate_size: i32) -> Option<Self> {
        let num_experts = int(v, "num_experts").or_else(|| int(v, "n_routed_experts"))?;
        let moe_inter = int(v, "moe_intermediate_size").unwrap_or(intermediate_size);
        Some(MoeConfig {
            num_experts: num_experts as usize,
            num_experts_per_tok: int(v, "num_experts_per_tok").unwrap_or(num_experts).max(1)
                as usize,
            moe_intermediate_size: moe_inter,
            // Qwen2 gives the shared width directly; DeepSeek packs `n_shared_experts` of them.
            shared_expert_intermediate_size: int(v, "shared_expert_intermediate_size")
                .or_else(|| int(v, "n_shared_experts").map(|n| n * moe_inter))
                .unwrap_or(intermediate_size),
            norm_topk_prob: v
                .get("norm_topk_prob")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            routed_scaling_factor: f32_opt(v, "routed_scaling_factor").unwrap_or(1.0),
            first_k_dense_replace: int(v, "first_k_dense_replace").unwrap_or(0).max(0) as usize,
        })
    }
}

impl MlaConfig {
    /// The Multi-head Latent Attention block (DeepSeek-V2): present iff the KV low-rank dim is
    /// configured. `q_lora_rank` is `null` for DeepSeek-V2-Lite (a full `q_proj`), an int for the
    /// larger models; `v_head_dim` defaults to `hidden_size / num_heads`.
    pub fn from_json(v: &Value, hidden_size: i32, num_heads: i32) -> Option<Self> {
        let kv_lora_rank = int(v, "kv_lora_rank")?;
        Some(MlaConfig {
            q_lora_rank: int(v, "q_lora_rank"),
            kv_lora_rank,
            qk_nope_head_dim: int(v, "qk_nope_head_dim").unwrap_or(0),
            qk_rope_head_dim: int(v, "qk_rope_head_dim").unwrap_or(0),
            v_head_dim: int(v, "v_head_dim").unwrap_or(hidden_size / num_heads),
        })
    }
}

impl YarnConfig {
    /// YaRN RoPE scaling (DeepSeek-V2): a `rope_scaling` block of `type` / `rope_type` `"yarn"`.
    pub fn from_json(v: &Value) -> Option<Self> {
        let rs = v.get("rope_scaling")?;
        let ty = rs
            .get("type")
            .or_else(|| rs.get("rope_type"))
            .and_then(|x| x.as_str());
        if ty != Some("yarn") {
            return None;
        }
        let f = |key: &str, default: f32| f32_opt(rs, key).unwrap_or(default);
        Some(YarnConfig {
            factor: f("factor", 1.0),
            beta_fast: f("beta_fast", 32.0),
            beta_slow: f("beta_slow", 1.0),
            original_context: f("original_max_position_embeddings", 4096.0),
            mscale: f("mscale", 1.0),
            mscale_all_dim: f("mscale_all_dim", 0.0),
        })
    }
}

impl RopeScaling {
    /// The Llama-3 NTK-by-parts `rope_scaling` block. Only the `"llama3"` schedule is parsed (a bare
    /// factor block with no type counts as llama3); absent / other types (e.g. yarn) are `None`.
    pub fn from_json(v: &Value) -> Option<Self> {
        let rs = v.get("rope_scaling")?;
        let f = |key: &str, default: f32| f32_opt(rs, key).unwrap_or(default);
        let is_llama3 = rs
            .get("rope_type")
            .or_else(|| rs.get("type"))
            .and_then(|x| x.as_str())
            .map(|s| s == "llama3")
            .unwrap_or(true);
        is_llama3.then(|| RopeScaling {
            factor: f("factor", 1.0),
            low_freq_factor: f("low_freq_factor", 1.0),
            high_freq_factor: f("high_freq_factor", 4.0),
            original_context: f("original_max_position_embeddings", 8192.0),
        })
    }
}

/// The interleaved multimodal-RoPE section `[t, h, w]` (Qwen3-VL `rope_scaling.mrope_section`), or
/// `None` when the config carries no three-entry section.
pub fn mrope_section(v: &Value) -> Option<[i32; 3]> {
    v.get("rope_scaling")
        .and_then(|rs| rs.get("mrope_section"))
        .and_then(|x| x.as_array())
        .filter(|a| a.len() == 3)
        .map(|a| {
            let g = |i: usize| a[i].as_i64().unwrap_or(0) as i32;
            [g(0), g(1), g(2)]
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// sc-27255: a VLM wrapper dispatches on its nested decoder (the MLX rule both engines now
    /// share), and a nested block that names no decoder falls back to the top level.
    #[test]
    fn dispatch_descends_into_a_naming_text_config() {
        let qwen35 = json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": { "model_type": "qwen3_5_text" },
            "vision_config": { "model_type": "qwen3_5", "depth": 27 }
        });
        assert_eq!(
            Architecture::from_config(&qwen35).unwrap(),
            Architecture::Qwen35
        );
        assert_eq!(Architecture::Qwen35.family(), "qwen3_5");
        // The nested decoder wins over a wrapper name no family matches.
        let llava = json!({
            "architectures": ["LlavaForConditionalGeneration"],
            "model_type": "llava",
            "text_config": { "model_type": "llama" }
        });
        assert_eq!(
            Architecture::from_config(&llava).unwrap(),
            Architecture::Llama
        );
        // A `text_config` carrying only numerics leaves dispatch on the top level.
        let gemma4 = json!({
            "model_type": "gemma4_unified",
            "text_config": { "hidden_size": 8 }
        });
        assert_eq!(
            Architecture::from_config(&gemma4).unwrap(),
            Architecture::Gemma4Unified
        );
    }

    /// sc-27255: every Qwen3.5 spelling either engine accepted routes to the hybrid decoder, and
    /// plain Qwen3 / Qwen3-VL are unaffected by the substring ordering.
    #[test]
    fn every_qwen35_spelling_routes_to_the_hybrid_decoder() {
        for model_type in ["qwen3_5", "qwen3_5_text", "qwen35", "qwen3_next"] {
            let v = json!({ "model_type": model_type });
            assert_eq!(
                Architecture::from_config(&v).unwrap(),
                Architecture::Qwen35,
                "{model_type}"
            );
        }
        let q3 = json!({ "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3" });
        assert_eq!(Architecture::from_config(&q3).unwrap(), Architecture::Qwen3);
        let vl =
            json!({ "model_type": "qwen3_vl", "text_config": { "model_type": "qwen3_vl_text" } });
        assert_eq!(
            Architecture::from_config(&vl).unwrap(),
            Architecture::Qwen3Vl
        );
        // A minimal config defaults to Llama; an unrecognized name is refused, typed.
        assert_eq!(
            Architecture::from_config(&json!({})).unwrap(),
            Architecture::Llama
        );
        assert!(matches!(
            Architecture::from_config(&json!({ "model_type": "bert" })),
            Err(ModelConfigError::Unsupported(_))
        ));
    }

    /// sc-27255: the quantization block is read top-level first, then beside the nested decoder;
    /// `bits` is required and `group_size` defaults to 64. Width validation is the engine's.
    #[test]
    fn quantization_block_is_read_top_level_then_nested() {
        let top = json!({ "quantization": { "bits": 4 } });
        assert_eq!(
            RawQuantization::from_config(&top, &top),
            Some(RawQuantization {
                bits: 4,
                group_size: RawQuantization::DEFAULT_GROUP_SIZE
            })
        );
        let wrapper = json!({
            "text_config": { "quantization": { "bits": 8, "group_size": 32 } }
        });
        assert_eq!(
            RawQuantization::from_config(&wrapper, &wrapper["text_config"]),
            Some(RawQuantization {
                bits: 8,
                group_size: 32
            })
        );
        let both = json!({
            "quantization": { "bits": 4, "group_size": 128 },
            "text_config": { "quantization": { "bits": 8 } }
        });
        assert_eq!(
            RawQuantization::from_config(&both, &both["text_config"]).map(|q| q.bits),
            Some(4)
        );
        let no_bits = json!({ "quantization": { "group_size": 64 } });
        assert_eq!(RawQuantization::from_config(&no_bits, &no_bits), None);
        assert_eq!(RawQuantization::from_config(&json!({}), &json!({})), None);
    }

    /// sc-27255: the Gemma 4 KV-sharing helpers both engines need — the tail shares the last
    /// pre-tail layer of its own type, which publishes its K/V; the tail's MLP doubles on request.
    #[test]
    fn gemma4_kv_sharing_tail_reuses_the_last_layer_of_its_own_type() {
        let v = json!({
            "layer_types": [
                "sliding_attention", "full_attention", "sliding_attention",
                "full_attention", "sliding_attention", "full_attention"
            ],
            "num_kv_shared_layers": 2,
            "use_double_wide_mlp": true
        });
        let g = Gemma4Config::from_json(&v, 6, 2, 64).unwrap();
        assert_eq!(g.first_kv_shared_layer(), Some(4));
        assert!(!g.is_kv_shared(3) && g.is_kv_shared(4) && g.is_kv_shared(5));
        assert_eq!(g.kv_donor(4), Some(2));
        assert_eq!(g.kv_donor(5), Some(3));
        assert_eq!(g.kv_donor(3), None);
        assert!(g.stores_shared_kv(2) && g.stores_shared_kv(3));
        assert!(!g.stores_shared_kv(0) && !g.stores_shared_kv(1) && !g.stores_shared_kv(4));
        assert_eq!(g.layer_intermediate_size(3, 16), 16);
        assert_eq!(g.layer_intermediate_size(4, 16), 32);
        let unshared = Gemma4Config::from_json(&json!({}), 6, 2, 64).unwrap();
        assert_eq!(unshared.first_kv_shared_layer(), None);
        assert!(!unshared.is_kv_shared(5) && !unshared.stores_shared_kv(4));
    }

    /// A `default` rope type carrying a `factor` asks for a schedule no engine implements.
    #[test]
    fn gemma4_refuses_a_factor_on_the_default_schedule() {
        let v = json!({
            "rope_parameters": { "sliding_attention": { "rope_type": "default", "factor": 2.0 } }
        });
        assert!(matches!(
            Gemma4Config::from_json(&v, 2, 1, 64),
            Err(ModelConfigError::Unsupported(_))
        ));
    }

    #[test]
    fn leaf_blocks_parse_their_defaults() {
        let moe = MoeConfig::from_json(
            &json!({ "n_routed_experts": 64, "n_shared_experts": 2, "moe_intermediate_size": 8 }),
            32,
        )
        .unwrap();
        assert_eq!(moe.num_experts_per_tok, 64);
        assert_eq!(moe.shared_expert_intermediate_size, 16);
        assert_eq!(MoeConfig::from_json(&json!({}), 32), None);
        let mla = MlaConfig::from_json(
            &json!({ "kv_lora_rank": 512, "qk_rope_head_dim": 64 }),
            64,
            4,
        )
        .unwrap();
        assert_eq!((mla.q_lora_rank, mla.v_head_dim), (None, 16));
        let yarn = json!({ "rope_scaling": { "type": "yarn", "factor": 40.0 } });
        assert_eq!(YarnConfig::from_json(&yarn).unwrap().factor, 40.0);
        assert_eq!(RopeScaling::from_json(&yarn), None);
        let llama3 = json!({ "rope_scaling": { "factor": 8.0 } });
        assert_eq!(
            RopeScaling::from_json(&llama3).unwrap().high_freq_factor,
            4.0
        );
        assert_eq!(YarnConfig::from_json(&llama3), None);
        let mrope = json!({ "rope_scaling": { "mrope_section": [24, 20, 20] } });
        assert_eq!(mrope_section(&mrope), Some([24, 20, 20]));
    }
}
