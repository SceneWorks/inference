//! Block drafts (epic SW-F360, story SW-7193): the backend-neutral contract a block-draft
//! checkpoint is read and checked against before a backend builds it.
//!
//! A block draft is a small decoder that reads the **target's** hidden rows at its configured
//! `target_layer_ids`, borrows the target's token embedding and LM head, and proposes a whole
//! block of tokens per step: the anchor (the last committed token) followed by `depth` mask
//! tokens, one draft forward. The checkpoint's `architectures` selects the format
//! ([`BlockDraftFormat`]) — a load names it through the ordinary
//! [`LoadSpec::draft_source`](crate::LoadSpec::draft_source), and a provider that sees a block
//! draft there advertises `block_draft` instead of `draft_model`. A second format (DFlash2)
//! plugs in as another [`BlockDraftFormat`] arm with its own [`BlockDraftConfig::from_json`]
//! contract and tensor list.
//!
//! Both backends read the config through [`BlockDraftConfig::read`], check it against the target
//! with [`BlockDraftConfig::mismatches`] and the stored tensors with
//! [`BlockDraftConfig::check_tensors`], and price it with [`BlockDraftConfig::load_bytes`] /
//! [`BlockDraftConfig::request_bytes`] — one check, so the two backends accept, refuse and price
//! exactly the same drafts. Every refusal is a typed [`BlockDraftRefusal`]; the draft never
//! fails the target's load (the target loads alone and the request runs without the proposer).

use std::path::Path;

use serde_json::Value;

/// The depth (drafts per verify step, block size − 1) a block draft is advertised and run at
/// unless a request names another: block size 5, the fastest measured on quantized Qwen3.8-27B
/// targets on Apple silicon (the DSpine MLX reference).
pub const BLOCK_DRAFT_RECOMMENDED_DEPTH: u32 = 4;

/// The quantization group size a block draft's projections are quantized with at load.
pub const BLOCK_DRAFT_GROUP_SIZE: usize = 64;

/// A block-draft checkpoint format, selected by the config's `architectures`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockDraftFormat {
    /// Tencent DSpine (`DSpineDraftModel`, e.g. `tencent/Qwen3.8-27B-DSpine-b16`): a DFlash-style
    /// block draft whose every layer carries a code-bridge write, refined serially per row.
    DSpine,
}

impl BlockDraftFormat {
    /// Every supported format.
    pub const ALL: [BlockDraftFormat; 1] = [BlockDraftFormat::DSpine];

    /// The stable lower-case label (`dspine`).
    pub const fn label(self) -> &'static str {
        match self {
            BlockDraftFormat::DSpine => "dspine",
        }
    }

    /// The `architectures` entry that selects this format.
    pub const fn architecture(self) -> &'static str {
        match self {
            BlockDraftFormat::DSpine => "DSpineDraftModel",
        }
    }

    /// The block-draft format a checkpoint config declares in `architectures`, or `None` when it
    /// is not a block draft (an ordinary decoder — a `draft_model` candidate).
    pub fn from_config(config: &Value) -> Option<Self> {
        let architectures = config.get("architectures")?.as_array()?;
        Self::ALL.into_iter().find(|format| {
            architectures
                .iter()
                .any(|a| a.as_str() == Some(format.architecture()))
        })
    }

    /// [`from_config`](Self::from_config) over the `config.json` in `dir`; `None` when the file
    /// is missing or unreadable (the ordinary draft path then reports it).
    pub fn of_dir(dir: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
        Self::from_config(&serde_json::from_str(&text).ok()?)
    }
}

/// The load-time precision of a block draft's projections (E8): 8-bit by default, 4-bit on
/// request. The code bridge and the norms stay BF16 — the refinement depends on their rounding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlockDraftPrecision {
    /// 8-bit group-wise affine (the default).
    #[default]
    Q8,
    /// 4-bit group-wise affine.
    Q4,
}

impl BlockDraftPrecision {
    /// The precision a load's [`LoadSpec::draft_quantize`](crate::LoadSpec::draft_quantize) asks
    /// for: unset or `Q8` → 8-bit, `Q4` → 4-bit; any other format is refused by name.
    pub fn from_request(
        requested: Option<crate::Quantize>,
    ) -> std::result::Result<Self, BlockDraftRefusal> {
        match requested {
            None | Some(crate::Quantize::Q8) => Ok(Self::Q8),
            Some(crate::Quantize::Q4) => Ok(Self::Q4),
            Some(other) => Err(BlockDraftRefusal::Precision(format!(
                "{other:?} is not a block-draft precision (8-bit or 4-bit)"
            ))),
        }
    }

    /// Bits per quantized weight.
    pub const fn bits(self) -> u32 {
        match self {
            Self::Q8 => 8,
            Self::Q4 => 4,
        }
    }

    /// The stable label (`q8`, `q4`).
    pub const fn label(self) -> &'static str {
        match self {
            Self::Q8 => "q8",
            Self::Q4 => "q4",
        }
    }
}

/// Why a block draft was refused, typed (AC2). [`Display`](std::fmt::Display) leads with
/// `block draft:` and the [`kind`](Self::kind), and names both values of a mismatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockDraftRefusal {
    /// The config is unreadable or outside the format's inference contract.
    Format(String),
    /// The requested load precision is not one a block draft runs at.
    Precision(String),
    /// The target cannot hand a block draft its hidden rows (its family has no layer taps).
    TargetFamily(String),
    /// The draft's hidden size is not the target's.
    HiddenSize {
        /// The draft's `hidden_size`.
        draft: usize,
        /// The target's.
        target: usize,
    },
    /// The draft scores a different vocabulary than the target's LM head.
    Vocabulary {
        /// The draft's `vocab_size`.
        draft: usize,
        /// The target's.
        target: usize,
    },
    /// The draft's mask token is not a token the target embeds and its tokenizer defines.
    MaskToken {
        /// The draft's `mask_token_id`.
        mask: i64,
        /// The target tokenizer's vocabulary size.
        tokenizer: usize,
        /// The target model's `vocab_size`.
        target: usize,
    },
    /// The draft reads target layers the target does not have, or was built for another depth.
    TargetLayers {
        /// The draft's `target_layer_ids`.
        ids: Vec<usize>,
        /// The draft's `num_target_layers`.
        declared: usize,
        /// The target's decoder layer count.
        target: usize,
    },
    /// The draft's attention geometry is inconsistent or not one this runtime runs.
    AttentionGeometry(String),
    /// A stored tensor is missing, unused, or has the wrong shape or version.
    Tensors(String),
}

impl BlockDraftRefusal {
    /// The stable reason label (`format`, `precision`, `target_family`, `hidden_size`,
    /// `vocabulary`, `mask_token`, `target_layers`, `attention_geometry`, `tensors`).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Format(_) => "format",
            Self::Precision(_) => "precision",
            Self::TargetFamily(_) => "target_family",
            Self::HiddenSize { .. } => "hidden_size",
            Self::Vocabulary { .. } => "vocabulary",
            Self::MaskToken { .. } => "mask_token",
            Self::TargetLayers { .. } => "target_layers",
            Self::AttentionGeometry(_) => "attention_geometry",
            Self::Tensors(_) => "tensors",
        }
    }
}

impl std::fmt::Display for BlockDraftRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "block draft: {}: ", self.kind())?;
        match self {
            Self::Format(why)
            | Self::Precision(why)
            | Self::TargetFamily(why)
            | Self::AttentionGeometry(why)
            | Self::Tensors(why) => f.write_str(why),
            Self::HiddenSize { draft, target } => {
                write!(f, "hidden_size {draft} != target {target}")
            }
            Self::Vocabulary { draft, target } => {
                write!(f, "vocab_size {draft} != target {target}")
            }
            Self::MaskToken {
                mask,
                tokenizer,
                target,
            } => write!(
                f,
                "mask_token_id {mask} is outside the target's tokenizer ({tokenizer} tokens) or \
                 embedding ({target} rows)"
            ),
            Self::TargetLayers {
                ids,
                declared,
                target,
            } => write!(
                f,
                "target_layer_ids {ids:?} (built for {declared} target layers) do not fit a \
                 {target}-layer target"
            ),
        }
    }
}

/// Join several refusals into the one load-report reason, in order.
pub fn block_draft_refusal_reason(refusals: &[BlockDraftRefusal]) -> String {
    refusals
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}

/// What a target offers a block draft: the geometry the draft must match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockDraftTarget {
    /// The target's residual width.
    pub hidden_size: usize,
    /// The target's decoder layer count.
    pub num_layers: usize,
    /// Rows of the target's embedding and LM head.
    pub vocab_size: usize,
    /// The target tokenizer's vocabulary size (the ids it defines).
    pub tokenizer_vocab: usize,
}

/// A parsed block-draft config: everything the draft computes with.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockDraftConfig {
    /// The checkpoint format.
    pub format: BlockDraftFormat,
    /// Draft residual width (must equal the target's).
    pub hidden_size: usize,
    /// Draft decoder layers.
    pub num_hidden_layers: usize,
    /// Draft MLP width.
    pub intermediate_size: usize,
    /// Draft query heads.
    pub num_attention_heads: usize,
    /// Draft key/value heads.
    pub num_key_value_heads: usize,
    /// Draft head width.
    pub head_dim: usize,
    /// Rows of the draft's code-bridge codebook — the vocabulary it scores.
    pub vocab_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base (default RoPE only).
    pub rope_theta: f32,
    /// The block size the draft was trained with: at most `block_size - 1` drafts per step.
    pub block_size: usize,
    /// The token id the block's draft rows carry.
    pub mask_token_id: i64,
    /// The target layers whose output rows the draft reads, in order.
    pub target_layer_ids: Vec<usize>,
    /// The target depth the draft was built for.
    pub num_target_layers: usize,
    /// One sliding window over every draft layer (`None`: full attention).
    pub sliding_window: Option<usize>,
    /// Code-bridge code width.
    pub code_size: usize,
    /// Code-bridge rank.
    pub bridge_rank: usize,
    /// Code-bridge epsilon.
    pub bridge_eps: f32,
    /// Candidates per row the refinement scores (top-k of the draft row's logits).
    pub head_topk: usize,
}

fn need(cond: bool, why: impl FnOnce() -> String) -> std::result::Result<(), BlockDraftRefusal> {
    if cond {
        Ok(())
    } else {
        Err(BlockDraftRefusal::Format(why()))
    }
}

fn uint(v: &Value, key: &str) -> std::result::Result<usize, BlockDraftRefusal> {
    v.get(key)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| BlockDraftRefusal::Format(format!("`{key}` is missing or not an integer")))
}

fn float(v: &Value, key: &str) -> std::result::Result<f32, BlockDraftRefusal> {
    v.get(key)
        .and_then(Value::as_f64)
        .map(|x| x as f32)
        .ok_or_else(|| BlockDraftRefusal::Format(format!("`{key}` is missing or not a number")))
}

impl BlockDraftConfig {
    /// Read and parse `dir/config.json`.
    pub fn read(dir: &Path) -> std::result::Result<Self, BlockDraftRefusal> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| BlockDraftRefusal::Format(format!("read {}: {e}", path.display())))?;
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| BlockDraftRefusal::Format(format!("parse {}: {e}", path.display())))?;
        Self::from_json(&value)
    }

    /// Parse a block-draft config, dispatching on its `architectures`.
    pub fn from_json(raw: &Value) -> std::result::Result<Self, BlockDraftRefusal> {
        match BlockDraftFormat::from_config(raw) {
            Some(BlockDraftFormat::DSpine) => Self::dspine(raw),
            None => Err(BlockDraftRefusal::Format(format!(
                "`architectures` names no block-draft format ({})",
                BlockDraftFormat::ALL
                    .map(BlockDraftFormat::architecture)
                    .join(", ")
            ))),
        }
    }

    /// The DSpine inference contract (the MLX reference's `DraftConfig.from_dict`): every draft
    /// layer carries an independent bridge write, serial block decoding, no grouped convolution,
    /// the exact-sender last-write refinement, default RoPE and at most one sliding window.
    fn dspine(raw: &Value) -> std::result::Result<Self, BlockDraftRefusal> {
        let d = raw
            .get("dflash_config")
            .filter(|d| d.is_object())
            .ok_or_else(|| BlockDraftRefusal::Format("missing `dflash_config`".into()))?;
        let truthy = |key: &str| {
            d.get(key)
                .is_some_and(|v| !(v.is_null() || v == &Value::Bool(false) || v == &Value::from(0)))
        };
        for key in [
            "selector_rank",
            "selector_top_k",
            "projector_type",
            "attention_sink_bias",
            "attention_value_scale",
            "selected_prefix_feature_reliability",
            "selected_prefix_feature_next_token_rows",
            "conv_kernel_size",
            "conv_group_size",
        ] {
            need(!truthy(key), || {
                format!("`{key}` is a head a DSpine draft does not have")
            })?;
        }
        for (key, want) in [
            ("selected_prefix_terminal", Value::Bool(false)),
            ("selected_prefix_decode_mode", Value::from("serial")),
            ("selected_prefix_feature_writes", Value::Bool(true)),
            (
                "selected_prefix_feature_write_mode",
                Value::from("independent"),
            ),
            ("draft_block_causal", Value::Bool(true)),
        ] {
            need(d.get(key) == Some(&want), || {
                format!("DSpine requires `{key}` = {want}")
            })?;
        }
        let bridge = d
            .get("selected_prefix_code_bridge")
            .filter(|b| b.is_object())
            .ok_or_else(|| {
                BlockDraftRefusal::Format("missing `selected_prefix_code_bridge`".into())
            })?;
        let num_hidden_layers = uint(raw, "num_hidden_layers")?;
        let features: Vec<u64> = d
            .get("selected_prefix_feature_layers")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).collect())
            .unwrap_or_default();
        need(
            features == (1..=num_hidden_layers as u64).collect::<Vec<_>>(),
            || format!("every draft layer must carry a bridge write (got {features:?})"),
        )?;
        let mut widths = Vec::new();
        if let Some(refine) = d.get("selected_prefix_last_write_refine") {
            need(
                refine.get("kind") == Some(&Value::from("exact_sender"))
                    && refine.get("version").and_then(Value::as_u64) == Some(1),
                || "`selected_prefix_last_write_refine` must be exact_sender version 1".into(),
            )?;
            widths.push(uint(refine, "head_topk")?);
        }
        if let Some(second) = raw.get("code_bridge_second_pass") {
            need(
                second.get("mode") == Some(&Value::from("serial_table"))
                    && second.get("first_exact").and_then(Value::as_u64)
                        == Some(num_hidden_layers as u64 - 1),
                || "`code_bridge_second_pass` must redo the last layer as a serial table".into(),
            )?;
            widths.push(uint(second, "head_topk")?);
        }
        need(!widths.is_empty(), || {
            "the checkpoint names no refinement width".into()
        })?;
        need(widths.iter().all(|&w| w == widths[0]), || {
            format!("refinement widths disagree: {widths:?}")
        })?;
        let head_topk = widths[0];
        need(
            head_topk.is_power_of_two() && (2..=64).contains(&head_topk),
            || format!("head_topk {head_topk} must be a power of two in 2..=64"),
        )?;
        for (key, want) in [
            ("model_type", Value::from("qwen3")),
            ("hidden_act", Value::from("silu")),
            ("attention_bias", Value::Bool(false)),
            ("tie_word_embeddings", Value::Bool(false)),
        ] {
            need(raw.get(key).is_none_or(|v| v == &want), || {
                format!("DSpine requires `{key}` = {want}")
            })?;
        }
        let rope_type = raw
            .get("rope_parameters")
            .and_then(|p| p.get("rope_type").or_else(|| p.get("type")))
            .and_then(Value::as_str)
            .unwrap_or("default");
        need(
            rope_type == "default" && raw.get("rope_scaling").is_none_or(Value::is_null),
            || format!("only default RoPE is supported (got {rope_type})"),
        )?;
        let rope_theta = raw
            .get("rope_theta")
            .or_else(|| raw.get("rope_parameters").and_then(|p| p.get("rope_theta")))
            .and_then(Value::as_f64)
            .ok_or_else(|| BlockDraftRefusal::Format("missing `rope_theta`".into()))?
            as f32;
        let vocab_size = uint(raw, "vocab_size")?;
        if let Some(dv) = raw.get("draft_vocab_size").and_then(Value::as_u64) {
            need(dv as usize == vocab_size, || {
                format!("draft_vocab_size {dv} != vocab_size {vocab_size}")
            })?;
        }
        let sliding_window = sliding_window(raw, num_hidden_layers)?;
        let target_layer_ids = d
            .get("target_layer_ids")
            .and_then(Value::as_array)
            .and_then(|a| {
                a.iter()
                    .map(|x| x.as_u64().map(|n| n as usize))
                    .collect::<Option<Vec<_>>>()
            })
            .filter(|ids| !ids.is_empty())
            .ok_or_else(|| BlockDraftRefusal::Format("missing `target_layer_ids`".into()))?;
        let mask_token_id = d
            .get("mask_token_id")
            .and_then(Value::as_i64)
            .ok_or_else(|| BlockDraftRefusal::Format("missing `mask_token_id`".into()))?;
        let block_size = uint(raw, "block_size")?;
        need(block_size >= 2, || format!("block_size {block_size} < 2"))?;
        Ok(Self {
            format: BlockDraftFormat::DSpine,
            hidden_size: uint(raw, "hidden_size")?,
            num_hidden_layers,
            intermediate_size: uint(raw, "intermediate_size")?,
            num_attention_heads: uint(raw, "num_attention_heads")?,
            num_key_value_heads: uint(raw, "num_key_value_heads")?,
            head_dim: uint(raw, "head_dim")?,
            vocab_size,
            rms_norm_eps: float(raw, "rms_norm_eps")?,
            rope_theta,
            block_size,
            mask_token_id,
            target_layer_ids,
            num_target_layers: uint(raw, "num_target_layers")?,
            sliding_window,
            code_size: uint(bridge, "code_size")?,
            bridge_rank: uint(bridge, "rank")?,
            bridge_eps: float(bridge, "eps")?,
            head_topk,
        })
    }

    /// The most drafts one block proposes (`block_size - 1`).
    pub fn max_depth(&self) -> u32 {
        u32::try_from(self.block_size.saturating_sub(1)).unwrap_or(u32::MAX)
    }

    /// Target context rows a draft ever attends to: a query at position `p` sees keys
    /// `p - window + 1 ..= p`, so `window - 1` context rows before the block (E5). `None`: all.
    pub fn context_rows(&self) -> Option<usize> {
        self.sliding_window.map(|w| w.saturating_sub(1).max(1))
    }

    /// The width of one tapped target row the draft reads: `target_layer_ids.len() × hidden`.
    pub fn tap_width(&self) -> usize {
        self.target_layer_ids.len() * self.hidden_size
    }

    /// Every disagreement between this draft and `target` (empty when it fits), each typed.
    ///
    /// The draft's **own** attention geometry is checked for consistency, not equality with the
    /// target's: a block draft attends over its own projections of the target's rows, so its
    /// heads need not be the target's (DSpine-b16 runs 32 × 128 query heads beside the 27B's
    /// 24 × 256).
    pub fn mismatches(&self, target: &BlockDraftTarget) -> Vec<BlockDraftRefusal> {
        let mut out = Vec::new();
        if self.hidden_size != target.hidden_size {
            out.push(BlockDraftRefusal::HiddenSize {
                draft: self.hidden_size,
                target: target.hidden_size,
            });
        }
        if self.vocab_size != target.vocab_size {
            out.push(BlockDraftRefusal::Vocabulary {
                draft: self.vocab_size,
                target: target.vocab_size,
            });
        }
        let mask_ok = usize::try_from(self.mask_token_id).is_ok_and(|m| {
            m < target.tokenizer_vocab && m < target.vocab_size && m < self.vocab_size
        });
        if !mask_ok {
            out.push(BlockDraftRefusal::MaskToken {
                mask: self.mask_token_id,
                tokenizer: target.tokenizer_vocab,
                target: target.vocab_size,
            });
        }
        let mut sorted = self.target_layer_ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if self.num_target_layers != target.num_layers
            || sorted.len() != self.target_layer_ids.len()
            || self
                .target_layer_ids
                .iter()
                .any(|&l| l >= target.num_layers)
        {
            out.push(BlockDraftRefusal::TargetLayers {
                ids: self.target_layer_ids.clone(),
                declared: self.num_target_layers,
                target: target.num_layers,
            });
        }
        if let Some(why) = self.attention_geometry_problem() {
            out.push(BlockDraftRefusal::AttentionGeometry(why));
        }
        out
    }

    fn attention_geometry_problem(&self) -> Option<String> {
        let (q, kv, hd) = (
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        );
        if q == 0 || kv == 0 || hd == 0 || q % kv != 0 {
            return Some(format!(
                "{q} query heads over {kv} key/value heads is not a grouped-query layout"
            ));
        }
        if hd % 2 != 0 {
            return Some(format!("head_dim {hd} is odd; RoPE needs an even width"));
        }
        if self.num_hidden_layers == 0 || self.intermediate_size == 0 {
            return Some("the draft declares no layers or no MLP width".into());
        }
        None
    }

    /// Every tensor a checkpoint of this config stores, with its shape (empty shape: a scalar
    /// version marker that must equal 1). Bare names, as the published checkpoints store them.
    pub fn tensors(&self) -> Vec<(String, Vec<usize>)> {
        let (h, l) = (self.hidden_size, self.num_hidden_layers);
        let (q, kv, hd, i) = (
            self.num_attention_heads * self.head_dim,
            self.num_key_value_heads * self.head_dim,
            self.head_dim,
            self.intermediate_size,
        );
        let (c, r, v) = (self.code_size, self.bridge_rank, self.vocab_size);
        let mut out: Vec<(String, Vec<usize>)> = vec![
            ("fc.weight".into(), vec![h, self.tap_width()]),
            ("hidden_norm.weight".into(), vec![h]),
            ("norm.weight".into(), vec![h]),
            ("draft_block_causal_version".into(), vec![]),
            ("last_write_refine_version".into(), vec![]),
        ];
        for layer in 0..l {
            let p = format!("layers.{layer}.");
            for (name, shape) in [
                ("input_layernorm.weight", vec![h]),
                ("post_attention_layernorm.weight", vec![h]),
                ("self_attn.q_proj.weight", vec![q, h]),
                ("self_attn.k_proj.weight", vec![kv, h]),
                ("self_attn.v_proj.weight", vec![kv, h]),
                ("self_attn.o_proj.weight", vec![h, q]),
                ("self_attn.q_norm.weight", vec![hd]),
                ("self_attn.k_norm.weight", vec![hd]),
                ("mlp.gate_proj.weight", vec![i, h]),
                ("mlp.up_proj.weight", vec![i, h]),
                ("mlp.down_proj.weight", vec![h, i]),
            ] {
                out.push((format!("{p}{name}"), shape));
            }
        }
        let b = "selected_prefix_bridge.";
        for (name, shape) in [
            ("codebook", vec![v, c]),
            ("read_weight", vec![c, h]),
            ("early_read_weights", vec![l.saturating_sub(1), c, h]),
            ("feature_write_weights", vec![l, h, r]),
            ("gate.weight", vec![r, c]),
            ("gate.bias", vec![r]),
            ("predecessor_gate.weight", vec![r, c]),
            ("prev.weight", vec![r, c]),
            ("out.weight", vec![c, r]),
            ("independent_write_version", vec![]),
            ("terminal_disabled_version", vec![]),
        ] {
            out.push((format!("{b}{name}"), shape));
        }
        out
    }

    /// The `[out, in]` matrices a backend quantizes at load: `fc` and every layer projection.
    /// The code bridge stays BF16.
    pub fn quantized_matrices(&self) -> Vec<String> {
        let mut out = vec!["fc.weight".to_string()];
        for layer in 0..self.num_hidden_layers {
            for name in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ] {
                out.push(format!("layers.{layer}.{name}.weight"));
            }
        }
        out
    }

    /// Check the stored tensors before anything is built: every [`tensors`](Self::tensors) entry
    /// present with its shape (`shape(name)`, `None` when missing), and no tensor (`stored`) the
    /// draft does not read. Names every disagreement.
    pub fn check_tensors<'a>(
        &self,
        shape: impl Fn(&str) -> Option<Vec<usize>>,
        stored: impl IntoIterator<Item = &'a str>,
    ) -> std::result::Result<(), BlockDraftRefusal> {
        let expected = self.tensors();
        let mut wrong = Vec::new();
        for (name, want) in &expected {
            match shape(name) {
                Some(got) if &got == want || (want.is_empty() && got == [1]) => {}
                Some(got) => wrong.push(format!("`{name}` is {got:?}, expected {want:?}")),
                None => wrong.push(format!("`{name}` is missing")),
            }
        }
        let mut unused: Vec<&str> = stored
            .into_iter()
            .filter(|s| !expected.iter().any(|(name, _)| name == s))
            .collect();
        unused.sort_unstable();
        if !unused.is_empty() {
            wrong.push(format!("unused tensors {}", unused.join(", ")));
        }
        if wrong.is_empty() {
            Ok(())
        } else {
            Err(BlockDraftRefusal::Tensors(wrong.join("; ")))
        }
    }

    /// Resident bytes after a load at `precision`, plus the largest BF16 matrix the load holds
    /// while quantizing it (one matrix is materialized and quantized at a time).
    pub fn load_bytes(&self, precision: BlockDraftPrecision) -> u64 {
        let quantized = self.quantized_matrices();
        let mut resident = 0u64;
        let mut transient = 0u64;
        for (name, shape) in self.tensors() {
            let elements: u64 = shape.iter().map(|&d| d as u64).product::<u64>().max(1);
            if quantized.contains(&name) {
                let (out, inp) = (shape[0] as u64, shape[1] as u64);
                let groups = out * inp.div_ceil(BLOCK_DRAFT_GROUP_SIZE as u64);
                // Packed codes, then BF16 scales and biases per group.
                resident += elements * u64::from(precision.bits()) / 8 + groups * 4;
                transient = transient.max(elements * 2);
            } else if shape.is_empty() {
                resident += 8;
            } else {
                resident += elements * 2;
            }
        }
        resident + transient
    }

    /// The bytes one request holds for the draft beside the target, at `width` drafts per step
    /// over a `prompt_tokens + max_new_tokens` run (E8/E11): the target's tapped rows the draft
    /// reads (the prompt's, capped at the window, plus a verify step's), the draft's context
    /// cache (capped at the window, E5), its context projection over the prompt rows, and a
    /// step's transients (the block's draft logits over the vocabulary and the refinement table).
    pub fn request_bytes(
        &self,
        prompt_tokens: usize,
        max_new_tokens: u32,
        width: usize,
    ) -> Option<u64> {
        let cap = |rows: u64| match self.context_rows() {
            Some(c) => rows.min(c as u64),
            None => rows,
        };
        let element = 2u64;
        let prompt = cap(u64::try_from(prompt_tokens).ok()?);
        let run = cap(u64::try_from(prompt_tokens)
            .ok()?
            .checked_add(u64::from(max_new_tokens))?
            .checked_add(u64::try_from(width).ok()?)?
            .checked_add(1)?);
        let tap = u64::try_from(self.tap_width()).ok()?;
        let h = self.hidden_size as u64;
        let step_rows = u64::try_from(width).ok()?.checked_add(1)?;
        let taps = prompt
            .checked_add(step_rows)?
            .checked_mul(tap)?
            .checked_mul(element)?;
        let kv_row = (self.num_hidden_layers as u64)
            .checked_mul(2)?
            .checked_mul((self.num_key_value_heads * self.head_dim) as u64)?
            .checked_mul(element)?;
        let cache = run.checked_mul(kv_row)?;
        let context = prompt.checked_mul(h)?.checked_mul(element)?;
        let k = self.head_topk as u64;
        let draft_rows = u64::try_from(width).ok()?;
        let logits = draft_rows
            .checked_mul(self.vocab_size as u64)?
            .checked_mul(4)?;
        let table = draft_rows
            .checked_mul(k)?
            .checked_mul(h.checked_mul(6)?.checked_add(self.code_size as u64 * 4)?)?;
        taps.checked_add(cache)?
            .checked_add(context)?
            .checked_add(logits)?
            .checked_add(table)
    }
}

/// One sliding window shared by every draft layer, or full attention (`None`); mixed layer
/// types are refused.
fn sliding_window(
    raw: &Value,
    layers: usize,
) -> std::result::Result<Option<usize>, BlockDraftRefusal> {
    let types: Option<Vec<&str>> = raw
        .get("layer_types")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect());
    let uses = raw
        .get("use_sliding_window")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let declares = uses
        || types
            .as_ref()
            .is_some_and(|t| t.contains(&"sliding_attention"));
    if !declares {
        need(
            types
                .as_ref()
                .is_none_or(|t| t.iter().all(|&x| x == "full_attention")),
            || "unsupported `layer_types`".into(),
        )?;
        return Ok(None);
    }
    let window = raw.get("sliding_window").and_then(Value::as_u64);
    let max_window_layers = raw
        .get("max_window_layers")
        .and_then(Value::as_u64)
        .unwrap_or(layers as u64);
    need(
        uses && types.as_deref() == Some(&vec!["sliding_attention"; layers][..])
            && max_window_layers == layers as u64
            && window.is_some_and(|w| w > 0),
        || "only one sliding window on every draft layer is supported".into(),
    )?;
    Ok(window.map(|w| w as usize))
}

/// The `block_draft` advertisement a provider carries while a compatible block draft is
/// resident: up to the model's verify-depth bound (`verify_bound`) and the draft's own trained
/// block (`draft_max`), recommending the backend row's depth or the bound when shallower.
pub fn block_draft_capabilities(
    verify_bound: u32,
    draft_max: u32,
    depths: &crate::RecommendedDepths,
) -> crate::ProposerCapabilities {
    let max_depth = verify_bound.min(draft_max).max(1);
    crate::ProposerCapabilities {
        proposer: crate::SpeculativeProposer::BlockDraft,
        max_depth,
        recommended_depth: depths.block_draft.min(max_depth),
    }
}

#[cfg(test)]
mod tests;
