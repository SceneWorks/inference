//! Block drafts (epic SW-F360, story SW-7193): the backend-neutral contract a block-draft
//! checkpoint is read and checked against before a backend builds it.
//!
//! A block draft is a small decoder that reads the **target's** hidden rows at its configured
//! `target_layer_ids`, borrows the target's token embedding and LM head, and proposes a whole
//! block of tokens per step: the anchor (the last committed token) followed by `depth` mask
//! tokens, one draft forward. The checkpoint's `architectures` selects the format
//! ([`BlockDraftFormat`]) — a load names it through the ordinary
//! [`LoadSpec::draft_source`](crate::LoadSpec::draft_source), and a provider that sees a block
//! draft there advertises `block_draft` instead of `draft_model`. Each format (DSpine, DFlash2) is
//! a [`BlockDraftFormat`] arm with its own [`BlockDraftConfig::from_json`] contract, its own
//! [`BlockDraftHead`] and its own tensor list (SW-7193, SW-7194).
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

/// One group of a block draft's load ([`BlockDraftConfig::load_groups`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockDraftLoadGroup {
    /// The stored tensors the group reads.
    pub tensors: Vec<String>,
    /// The bytes the group holds beyond the draft's resident arrays while it converts: the
    /// sources it consumes into different arrays, which it then releases.
    pub window: u64,
}

/// A DFlash2 layer's two grouped dynamic convolutions (after `layers.{i}.`): around the
/// attention and around the MLP.
pub const DFLASH2_CONVS: [&str; 2] = ["attention_conv", "mlp_conv"];

/// The code bridge's tensors the built draft reads (after `selected_prefix_bridge.`).
const BRIDGE_TENSORS: [&str; 8] = [
    "codebook",
    "read_weight",
    "early_read_weights",
    "feature_write_weights",
    "gate.weight",
    "gate.bias",
    "predecessor_gate.weight",
    "prev.weight",
];

/// The bridge tensors a load converts into different arrays (packed, f32) and releases.
const BRIDGE_CONVERTED: [&str; 4] = [
    "gate.weight",
    "gate.bias",
    "predecessor_gate.weight",
    "prev.weight",
];

/// A block-draft checkpoint format, selected by the config's `architectures`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockDraftFormat {
    /// Tencent DSpine (`DSpineDraftModel`, e.g. `tencent/Qwen3.8-27B-DSpine-b16`): a DFlash-style
    /// block draft whose every layer carries a code-bridge write, refined serially per row.
    DSpine,
    /// Inco DFlash2 (`DFlash2DraftModel`, e.g. `incoai/Qwen3.8-27B-DFlash2`): a parallel block
    /// draft whose layers carry two-tap dynamic convolutions, with a candidate-path selector that
    /// rescores each row's top-`K` candidates against its predecessor.
    DFlash2,
}

impl BlockDraftFormat {
    /// Every supported format.
    pub const ALL: [BlockDraftFormat; 2] = [BlockDraftFormat::DSpine, BlockDraftFormat::DFlash2];

    /// The stable lower-case label (`dspine`, `dflash2`).
    pub const fn label(self) -> &'static str {
        match self {
            BlockDraftFormat::DSpine => "dspine",
            BlockDraftFormat::DFlash2 => "dflash2",
        }
    }

    /// The `architectures` entry that selects this format.
    pub const fn architecture(self) -> &'static str {
        match self {
            BlockDraftFormat::DSpine => "DSpineDraftModel",
            BlockDraftFormat::DFlash2 => "DFlash2DraftModel",
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
/// request. The norms and the scoring tables (DSpine's code bridge, DFlash2's selector codebooks
/// and convolution base kernels) stay BF16 — the candidate scores depend on their rounding.
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
    /// Candidates per row the format's head scores (top-k of the draft row's logits): DSpine's
    /// refinement width, DFlash2's `selector_top_k`.
    pub head_topk: usize,
    /// The format's own head.
    pub head: BlockDraftHead,
}

/// The format-specific part of a [`BlockDraftConfig`]: how a format turns a block's rows into
/// per-row candidates and their predecessor-conditioned scores.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockDraftHead {
    /// DSpine's code bridge (a write after every layer, refined per candidate sender).
    DSpine {
        /// Code-bridge code width.
        code_size: usize,
        /// Code-bridge rank.
        bridge_rank: usize,
        /// Code-bridge epsilon.
        bridge_eps: f32,
    },
    /// DFlash2's two-tap dynamic convolutions around each layer's attention and MLP, and its
    /// candidate-path selector.
    DFlash2 {
        /// Taps of each grouped dynamic causal convolution (2 in the published drafts).
        conv_kernel_size: usize,
        /// Channels per convolution group (the hidden size is a whole number of groups).
        conv_group_size: usize,
        /// Width of the selector's predecessor/successor codebooks and hidden projection.
        selector_rank: usize,
        /// Whether a block row attends only to the block rows before it (`is_causal`; the
        /// published drafts are non-causal: every block row sees the whole block).
        causal: bool,
        /// Factor on the block's input embeddings.
        input_embedding_scale: f32,
        /// Factor on the draft's logits.
        output_multiplier: f32,
        /// `tanh` soft cap on the draft's logits (`None`: none).
        final_logit_softcapping: Option<f32>,
    },
}

fn need(cond: bool, why: impl FnOnce() -> String) -> std::result::Result<(), BlockDraftRefusal> {
    if cond {
        Ok(())
    } else {
        Err(BlockDraftRefusal::Format(why()))
    }
}

/// The `dflash_config` object both formats nest their block-draft keys in.
fn dflash_config(raw: &Value) -> std::result::Result<&Value, BlockDraftRefusal> {
    raw.get("dflash_config")
        .filter(|d| d.is_object())
        .ok_or_else(|| BlockDraftRefusal::Format("missing `dflash_config`".into()))
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
            Some(BlockDraftFormat::DFlash2) => Self::dflash2(raw),
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
        let d = dflash_config(raw)?;
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
        need(
            raw.get("tie_word_embeddings")
                .is_none_or(|v| v == &Value::Bool(false)),
            || "DSpine requires `tie_word_embeddings` = false".into(),
        )?;
        let head = BlockDraftHead::DSpine {
            code_size: uint(bridge, "code_size")?,
            bridge_rank: uint(bridge, "rank")?,
            bridge_eps: float(bridge, "eps")?,
        };
        Self::common(
            raw,
            d,
            BlockDraftFormat::DSpine,
            uint(raw, "block_size")?,
            head_topk,
            head,
        )
    }

    /// The DFlash2 inference contract (the z-lab reference's `load_draft` for a
    /// `DFlash2DraftModel`): a grouped dynamic causal convolution pair around every layer's
    /// attention and MLP, a candidate-path selector over the top `selector_top_k` candidates of
    /// each row, default RoPE and at most one sliding window. DSpine's code-bridge heads are
    /// refused by name.
    fn dflash2(raw: &Value) -> std::result::Result<Self, BlockDraftRefusal> {
        let d = dflash_config(raw)?;
        for key in [
            "selected_prefix_code_bridge",
            "selected_prefix_feature_writes",
            "selected_prefix_last_write_refine",
            "projector_type",
            "attention_sink_bias",
            "attention_value_scale",
        ] {
            need(d.get(key).is_none_or(Value::is_null), || {
                format!("`{key}` is a head a DFlash2 draft does not have")
            })?;
        }
        need(raw.get("code_bridge_second_pass").is_none(), || {
            "`code_bridge_second_pass` is a head a DFlash2 draft does not have".into()
        })?;
        let hidden = uint(raw, "hidden_size")?;
        let conv_kernel_size = uint(d, "conv_kernel_size")?;
        let conv_group_size = uint(d, "conv_group_size")?;
        need(conv_kernel_size > 0, || "`conv_kernel_size` is 0".into())?;
        need(conv_group_size > 0 && hidden % conv_group_size == 0, || {
            format!("hidden_size {hidden} is not a whole number of {conv_group_size}-channel convolution groups")
        })?;
        let selector_rank = uint(d, "selector_rank")?;
        let head_topk = uint(d, "selector_top_k")?;
        need(selector_rank > 0 && head_topk > 0, || {
            "the candidate selector needs a positive `selector_rank` and `selector_top_k`".into()
        })?;
        let sliding = raw
            .get("layer_types")
            .and_then(Value::as_array)
            .is_some_and(|t| t.iter().any(|x| x.as_str() == Some("sliding_attention")));
        let causal = match raw.get("is_causal") {
            None | Some(Value::Null) => sliding,
            Some(Value::Bool(b)) => *b,
            Some(other) => {
                return Err(BlockDraftRefusal::Format(format!(
                    "`is_causal` = {other} is not a boolean"
                )))
            }
        };
        let factor = |key: &str| -> std::result::Result<f32, BlockDraftRefusal> {
            match d.get(key) {
                None | Some(Value::Null) => Ok(1.0),
                Some(v) => v
                    .as_f64()
                    .map(|x| x as f32)
                    .ok_or_else(|| BlockDraftRefusal::Format(format!("`{key}` is not a number"))),
            }
        };
        let softcap = d
            .get("final_logit_softcapping")
            .or_else(|| raw.get("final_logit_softcapping"))
            .and_then(Value::as_f64)
            .map(|x| x as f32)
            .filter(|&x| x > 0.0);
        // The reference reads `dflash_config.block_size`, then the top level, then 16.
        let block_size = if d.get("block_size").is_some() {
            uint(d, "block_size")?
        } else if raw.get("block_size").is_some() {
            uint(raw, "block_size")?
        } else {
            16
        };
        let head = BlockDraftHead::DFlash2 {
            conv_kernel_size,
            conv_group_size,
            selector_rank,
            causal,
            input_embedding_scale: factor("input_embedding_scale")?,
            output_multiplier: factor("output_multiplier")?,
            final_logit_softcapping: softcap,
        };
        let cfg = Self::common(
            raw,
            d,
            BlockDraftFormat::DFlash2,
            block_size,
            head_topk,
            head,
        )?;
        need(head_topk <= cfg.vocab_size, || {
            format!(
                "selector_top_k {head_topk} exceeds the {}-token vocabulary",
                cfg.vocab_size
            )
        })?;
        Ok(cfg)
    }

    /// What every block-draft format shares: a Qwen3 decoder body over the target's tapped rows,
    /// default RoPE, at most one sliding window, a mask token and the target layers it reads.
    fn common(
        raw: &Value,
        d: &Value,
        format: BlockDraftFormat,
        block_size: usize,
        head_topk: usize,
        head: BlockDraftHead,
    ) -> std::result::Result<Self, BlockDraftRefusal> {
        let name = match format {
            BlockDraftFormat::DSpine => "DSpine",
            BlockDraftFormat::DFlash2 => "DFlash2",
        };
        for (key, want) in [
            ("model_type", Value::from("qwen3")),
            ("hidden_act", Value::from("silu")),
            ("attention_bias", Value::Bool(false)),
        ] {
            need(raw.get(key).is_none_or(|v| v == &want), || {
                format!("{name} requires `{key}` = {want}")
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
        let num_hidden_layers = uint(raw, "num_hidden_layers")?;
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
        need(block_size >= 2, || format!("block_size {block_size} < 2"))?;
        Ok(Self {
            format,
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
            head_topk,
            head,
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
        let v = self.vocab_size;
        let mut out: Vec<(String, Vec<usize>)> = vec![
            ("fc.weight".into(), vec![h, self.tap_width()]),
            ("hidden_norm.weight".into(), vec![h]),
            ("norm.weight".into(), vec![h]),
        ];
        if self.format == BlockDraftFormat::DSpine {
            out.push(("draft_block_causal_version".into(), vec![]));
            out.push(("last_write_refine_version".into(), vec![]));
        }
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
            if let BlockDraftHead::DFlash2 {
                conv_kernel_size: k,
                conv_group_size: g,
                ..
            } = self.head
            {
                for conv in DFLASH2_CONVS {
                    out.push((format!("{p}{conv}.base_kernel"), vec![2, k, h]));
                    out.push((
                        format!("{p}{conv}.kernel_projection.weight"),
                        vec![2 * k * (h / g), h],
                    ));
                }
            }
        }
        match self.head {
            BlockDraftHead::DSpine {
                code_size: c,
                bridge_rank: r,
                ..
            } => {
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
            }
            BlockDraftHead::DFlash2 {
                selector_rank: r, ..
            } => {
                let s = "candidate_selector.";
                for (name, shape) in [
                    ("hidden_projection.weight", vec![r, h]),
                    ("predecessor_codebook", vec![v, r]),
                    ("successor_codebook", vec![v, r]),
                ] {
                    out.push((format!("{s}{name}"), shape));
                }
            }
        }
        out
    }

    /// The `[out, in]` matrices a backend quantizes at load: `fc` and every layer projection, and
    /// for DFlash2 every convolution's kernel projection and the selector's hidden projection.
    /// DSpine's code bridge, DFlash2's codebooks and base kernels, and the norms stay BF16.
    pub fn quantized_matrices(&self) -> Vec<String> {
        let mut out = vec!["fc.weight".to_string()];
        let dflash2 = self.format == BlockDraftFormat::DFlash2;
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
            if dflash2 {
                for conv in DFLASH2_CONVS {
                    out.push(format!("layers.{layer}.{conv}.kernel_projection.weight"));
                }
            }
        }
        if dflash2 {
            out.push("candidate_selector.hidden_projection.weight".into());
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

    /// The groups a load materializes, in load order (the stored checkpoint is BF16, DSpine's
    /// version markers I64). DSpine first reads its **checks** (the version markers and the
    /// disabled terminal projection, read and dropped before anything is built); then every
    /// format loads one group per draft layer, then the **head** (`fc`, the two norms outside the
    /// stack and the format's scoring tensors: DSpine's code bridge, DFlash2's selector). Each
    /// group's sources are read, converted and released before the next group reads its own, so a
    /// load holds the draft's resident arrays plus one group's
    /// [window](BlockDraftLoadGroup::window) — never the whole BF16 payload. Every stored tensor
    /// falls in exactly one group.
    pub fn load_groups(&self) -> Vec<BlockDraftLoadGroup> {
        let shapes: std::collections::HashMap<String, Vec<usize>> =
            self.tensors().into_iter().collect();
        let elements = |name: &str| -> u64 {
            shapes[name]
                .iter()
                .map(|&d| d as u64)
                .product::<u64>()
                .max(1)
        };
        // A source the draft keeps as stored (a BF16 norm, a codebook, a base kernel) shares its
        // buffer with the draft; one converted into a different array (a quantized matrix, the
        // packed or f32 bridge tensors) is consumed: held beside its conversion, then released.
        let window = |names: &[String], consumed: &dyn Fn(&str) -> bool| -> u64 {
            names
                .iter()
                .filter(|n| consumed(n))
                .map(|n| elements(n) * 2)
                .sum()
        };
        let quantized = self.quantized_matrices();
        let is_quantized = |n: &str| quantized.iter().any(|q| q == n);
        let b = "selected_prefix_bridge.";
        let mut groups = Vec::with_capacity(self.num_hidden_layers + 2);
        if self.format == BlockDraftFormat::DSpine {
            let out = format!("{b}out.weight");
            // Four I64 markers, and the terminal projection beside its `!= 0` mask.
            let checks_window = 4 * 8 + elements(&out) * 3;
            groups.push(BlockDraftLoadGroup {
                tensors: vec![
                    "draft_block_causal_version".into(),
                    "last_write_refine_version".into(),
                    format!("{b}independent_write_version"),
                    format!("{b}terminal_disabled_version"),
                    out,
                ],
                window: checks_window,
            });
        }
        for layer in 0..self.num_hidden_layers {
            let prefix = format!("layers.{layer}.");
            let tensors: Vec<String> = shapes
                .keys()
                .filter(|n| n.starts_with(&prefix))
                .cloned()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            let window = window(&tensors, &is_quantized);
            groups.push(BlockDraftLoadGroup { tensors, window });
        }
        let mut tensors: Vec<String> = vec![
            "fc.weight".into(),
            "hidden_norm.weight".into(),
            "norm.weight".into(),
        ];
        match self.format {
            BlockDraftFormat::DSpine => {
                tensors.extend(BRIDGE_TENSORS.iter().map(|s| format!("{b}{s}")));
                let window = window(&tensors, &|n| {
                    n == "fc.weight" || BRIDGE_CONVERTED.iter().any(|c| n == format!("{b}{c}"))
                });
                groups.push(BlockDraftLoadGroup { tensors, window });
            }
            BlockDraftFormat::DFlash2 => {
                tensors.extend(
                    [
                        "hidden_projection.weight",
                        "predecessor_codebook",
                        "successor_codebook",
                    ]
                    .map(|s| format!("candidate_selector.{s}")),
                );
                let window = window(&tensors, &is_quantized);
                groups.push(BlockDraftLoadGroup { tensors, window });
            }
        }
        groups
    }

    /// The draft's arrays once loaded at `precision`: every [quantized
    /// matrix](Self::quantized_matrices) packed (codes plus BF16 scales and biases per
    /// [`BLOCK_DRAFT_GROUP_SIZE`] group) and every other tensor as stored (BF16). DSpine adds the
    /// bridge's derived arrays — the packed `[gate; predecessor_gate; prev]` matrix (BF16), f32
    /// copies of `predecessor_gate`, `prev` and `gate.bias`, and the last layer's write weights in
    /// f32 — and drops its checks' tensors once read.
    pub fn resident_bytes(&self, precision: BlockDraftPrecision) -> u64 {
        let quantized = self.quantized_matrices();
        let b = "selected_prefix_bridge.";
        let checks: Vec<String> = match self.format {
            BlockDraftFormat::DSpine => self.load_groups()[0].tensors.clone(),
            BlockDraftFormat::DFlash2 => Vec::new(),
        };
        let mut resident = 0u64;
        for (name, shape) in self.tensors() {
            let elements: u64 = shape.iter().map(|&d| d as u64).product::<u64>().max(1);
            let bridge = name.strip_prefix(b);
            resident += if quantized.contains(&name) {
                let (out, inp) = (shape[0] as u64, shape[1] as u64);
                let groups = out * inp.div_ceil(BLOCK_DRAFT_GROUP_SIZE as u64);
                elements * u64::from(precision.bits()) / 8 + groups * 4
            } else if checks.contains(&name) {
                0
            } else if bridge == Some("gate.bias") {
                elements * 4
            } else if matches!(bridge, Some("predecessor_gate.weight" | "prev.weight")) {
                elements * (2 + 4)
            } else {
                elements * 2
            };
        }
        let last_write = match self.head {
            BlockDraftHead::DSpine { bridge_rank, .. } => {
                (self.hidden_size * bridge_rank) as u64 * 4
            }
            BlockDraftHead::DFlash2 { .. } => 0,
        };
        resident + last_write
    }

    /// The bytes a load at `precision` holds at its peak: the draft's
    /// [resident arrays](Self::resident_bytes), the largest [load group](Self::load_groups)'s
    /// window while it converts, and — because the Metal driver returns a released buffer
    /// asynchronously — one more such window of released sources still outstanding.
    pub fn load_bytes(&self, precision: BlockDraftPrecision) -> u64 {
        let window = self
            .load_groups()
            .iter()
            .map(|g| g.window)
            .max()
            .unwrap_or(0);
        self.resident_bytes(precision) + 2 * window
    }

    /// The bytes one request holds for the draft beside the target, at `width` drafts per step
    /// over a `prompt_tokens + max_new_tokens` run (E8/E11): the target's tapped rows the draft
    /// reads (the prompt's, capped at the window, plus a verify step's), the draft's context
    /// cache (capped at the window, E5), its context projection over the prompt rows, and a
    /// step's transients (the block's draft logits over the vocabulary and the format's candidate
    /// table: DSpine's refinement, DFlash2's selector edges). The tapped rows, the context cache
    /// and the context projection are activations at `element_bytes` per element — the backend's
    /// compute dtype (4 for f32, 2 for bf16 or f16); the logits and the candidate table are f32.
    pub fn request_bytes(
        &self,
        prompt_tokens: usize,
        max_new_tokens: u32,
        width: usize,
        element_bytes: u64,
    ) -> Option<u64> {
        let cap = |rows: u64| match self.context_rows() {
            Some(c) => rows.min(c as u64),
            None => rows,
        };
        let element = element_bytes;
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
        // Per candidate: DSpine refines a hidden row (f32 raw, refined, normalized) and gathers
        // its codebook row; DFlash2 gathers f32 predecessor and successor rows, their product
        // with the selector row, and scores a sender row of `K` edges.
        let per_candidate = match self.head {
            BlockDraftHead::DSpine { code_size, .. } => {
                h.checked_mul(6)?.checked_add(code_size as u64 * 4)?
            }
            BlockDraftHead::DFlash2 { selector_rank, .. } => (selector_rank as u64)
                .checked_mul(3 * 4)?
                .checked_add(k.checked_mul(4)?)?,
        };
        let table = draft_rows.checked_mul(k)?.checked_mul(per_candidate)?;
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
///
/// The verify bound applies to block drafts as to every proposer (sc-24438): each draft is one
/// more verify row, and `verify_bound` is the deepest verify the target's backend runs at a flat
/// per-step cost ([`verify_depth_bound`](crate::verify_depth_bound)). So a draft trained on a
/// longer block than the target verifies is advertised at the target's bound — on MLX the
/// Qwen3.8-27B target (gqa 6) verifies 4 drafts, below DFlash2's 7 (an 8-row block) and DSpine's
/// 15 — and a deeper request runs at that bound with the clamp named in the decode report's
/// fallbacks ([`resolve_speculative`](crate::resolve_speculative)). A non-causal draft still runs
/// its whole trained block and proposes its first `max_depth` rows.
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
