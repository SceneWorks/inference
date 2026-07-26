//! Bundled encoder-only T5Gemma-b-b-ul2 text conditioner.
//!
//! Frozen upstream loads `T5GemmaEncoderModel` from the checkpoint-local
//! `t5gemma-b-b-ul2/` directory after setting `is_encoder_decoder = false`. The decoder tensors
//! remain in the file but are never requested here.

use std::path::Path;

use candle_audio::candle_core::{
    bail, DType, Device, DeviceLocation, Result as CandleResult, Tensor, D,
};
use candle_audio::{AudioError, Result as AudioResult};
use candle_nn::{linear_b, Init, Linear, Module, VarBuilder};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::config::{ConditionerConfig, ModelConfig, PaddingMode, T5GemmaConfig};
use crate::weights::{SnapshotKind, SnapshotLayout};

const SHIPPED_HIDDEN: usize = 768;
const SHIPPED_LAYERS: usize = 12;
const SHIPPED_HEADS: usize = 12;
const SHIPPED_HEAD_DIM: usize = 64;
const SHIPPED_INTERMEDIATE: usize = 2048;
const SHIPPED_VOCAB: usize = 256_000;
const SHIPPED_MAX_LENGTH: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TextComputePolicy {
    compute_dtype: DType,
    raw_output_dtype: DType,
    conditioned_output_dtype: DType,
}

fn text_compute_policy(location: DeviceLocation) -> TextComputePolicy {
    let raw_output_dtype = match location {
        DeviceLocation::Cpu => DType::F32,
        DeviceLocation::Metal { .. } => DType::BF16,
        DeviceLocation::Cuda { .. } => DType::BF16,
    };
    TextComputePolicy {
        compute_dtype: DType::F32,
        raw_output_dtype,
        conditioned_output_dtype: DType::F32,
    }
}

#[derive(Debug, Clone, Deserialize)]
struct T5GemmaFileConfig {
    encoder: T5GemmaEncoderConfig,
    is_encoder_decoder: bool,
    pad_token_id: usize,
    torch_dtype: String,
}

/// Exact encoder sub-config parsed from bundled `t5gemma-b-b-ul2/config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct T5GemmaEncoderConfig {
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub attn_logit_softcapping: f64,
    pub dropout_rate: f64,
    pub head_dim: usize,
    pub hidden_activation: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub query_pre_attn_scalar: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: usize,
    pub torch_dtype: String,
    pub vocab_size: usize,
}

impl T5GemmaEncoderConfig {
    pub fn from_path(path: &Path) -> AudioResult<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| AudioError::Msg(format!("read {}: {e}", path.display())))?;
        let outer: T5GemmaFileConfig = serde_json::from_str(&text)
            .map_err(|e| AudioError::Msg(format!("parse {}: {e}", path.display())))?;
        if !outer.is_encoder_decoder || outer.pad_token_id != 0 || outer.torch_dtype != "bfloat16" {
            return Err(AudioError::Msg(format!(
                "{} is not the locked T5Gemma-b-b-ul2 config",
                path.display()
            )));
        }
        outer.encoder.validate()?;
        Ok(outer.encoder)
    }

    fn validate(&self) -> AudioResult<()> {
        let alternating = self.layer_types.len() == SHIPPED_LAYERS
            && self.layer_types.iter().enumerate().all(|(index, kind)| {
                kind == if index % 2 == 0 {
                    "sliding_attention"
                } else {
                    "full_attention"
                }
            });
        let exact = self.hidden_size == SHIPPED_HIDDEN
            && self.num_hidden_layers == SHIPPED_LAYERS
            && self.num_attention_heads == SHIPPED_HEADS
            && self.num_key_value_heads == SHIPPED_HEADS
            && self.head_dim == SHIPPED_HEAD_DIM
            && self.intermediate_size == SHIPPED_INTERMEDIATE
            && self.vocab_size == SHIPPED_VOCAB
            && self.hidden_activation == "gelu_pytorch_tanh"
            && self.rms_norm_eps == 1e-6
            && self.rope_theta == 10_000.0
            && self.query_pre_attn_scalar == 64
            && self.attn_logit_softcapping == 50.0
            && self.sliding_window == 4096
            && self.max_position_embeddings == 8192
            && self.torch_dtype == "bfloat16"
            && !self.attention_bias
            && self.attention_dropout == 0.0
            && self.dropout_rate == 0.0
            && alternating;
        if !exact {
            return Err(AudioError::Msg(format!(
                "unsupported T5Gemma encoder config: {self:?}"
            )));
        }
        Ok(())
    }
}

/// Every tensor the encoder builder requests. No decoder key can enter this list.
pub fn encoder_weight_keys() -> Vec<String> {
    let mut keys = vec![
        "model.encoder.embed_tokens.weight".to_string(),
        "model.encoder.norm.weight".to_string(),
    ];
    for layer in 0..SHIPPED_LAYERS {
        let prefix = format!("model.encoder.layers.{layer}");
        for suffix in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "pre_self_attn_layernorm.weight",
            "post_self_attn_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ] {
            keys.push(format!("{prefix}.{suffix}"));
        }
    }
    keys.sort();
    keys
}

struct T5RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl T5RmsNorm {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            weight: vb.get_with_hints(dim, "weight", Init::Const(0.0))?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        t5_rms_norm(x, &self.weight, self.eps, true)
    }
}

fn t5_rms_norm(
    x: &Tensor,
    checkpoint_weight: &Tensor,
    eps: f64,
    add_one: bool,
) -> CandleResult<Tensor> {
    let dtype = x.dtype();
    let x32 = x.to_dtype(DType::F32)?;
    let mean = x32.sqr()?.mean_keepdim(D::Minus1)?;
    let normalized = x32.broadcast_div(&(mean + eps)?.sqrt()?)?;
    let weight = checkpoint_weight
        .to_dtype(DType::F32)?
        .affine(1.0, if add_one { 1.0 } else { 0.0 })?;
    normalized.broadcast_mul(&weight)?.to_dtype(dtype)
}

fn to_heads(x: &Tensor, heads: usize, head_dim: usize) -> CandleResult<Tensor> {
    let (batch, seq, _) = x.dims3()?;
    x.reshape((batch, seq, heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()
}

fn from_heads(x: &Tensor) -> CandleResult<Tensor> {
    let (batch, heads, seq, head_dim) = x.dims4()?;
    x.transpose(1, 2)?
        .contiguous()?
        .reshape((batch, seq, heads * head_dim))
}

fn rope_tables(
    seq: usize,
    head_dim: usize,
    theta: f64,
    dtype: DType,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let half = head_dim / 2;
    // Transformers creates inv_freq on CPU in F32, then computes the position outer-product and
    // trigonometry on the execution device in F32 before casting once to the model dtype.
    let inv_freq = (0..half)
        .map(|index| {
            let exponent = 2.0f32 * index as f32 / head_dim as f32;
            1.0f32 / (theta as f32).powf(exponent)
        })
        .collect::<Vec<_>>();
    let positions = (0..seq).map(|position| position as f32).collect::<Vec<_>>();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half, 1), device)?;
    let positions = Tensor::from_vec(positions, (1, 1, seq), device)?;
    let frequencies = inv_freq.matmul(&positions)?.transpose(1, 2)?;
    let frequencies = Tensor::cat(&[&frequencies, &frequencies], D::Minus1)?.unsqueeze(1)?;
    Ok((
        frequencies.cos()?.to_dtype(dtype)?,
        frequencies.sin()?.to_dtype(dtype)?,
    ))
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
    let dim = x.dim(D::Minus1)?;
    let half = dim / 2;
    let first = x.narrow(D::Minus1, 0, half)?;
    let second = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&second.neg()?, &first], D::Minus1)?;
    x.broadcast_mul(cos)?
        .broadcast_add(&rotated.broadcast_mul(sin)?)
}

fn attention_mask(valid: &Tensor, seq: usize, dtype: DType, causal: bool) -> CandleResult<Tensor> {
    let batch = valid.dim(0)?;
    let valid = valid.reshape((batch, 1, 1, seq))?;
    let zeros = Tensor::zeros((batch, 1, 1, seq), dtype, valid.device())?;
    // Transformers uses torch.finfo(dtype).min, not -inf.
    let finite_min = match dtype {
        DType::F32 => f32::MIN,
        DType::BF16 => f32::from_bits(0xff7f_0000),
        DType::F16 => -65_504.0,
        _ => bail!("unsupported T5Gemma attention-mask dtype {dtype:?}"),
    };
    let blocked = Tensor::full(finite_min, (batch, 1, 1, seq), valid.device())?.to_dtype(dtype)?;
    let mut mask = valid.where_cond(&zeros, &blocked)?;
    if causal {
        let mut values = vec![0f32; seq * seq];
        for query in 0..seq {
            for key in (query + 1)..seq {
                values[query * seq + key] = finite_min;
            }
        }
        let causal = Tensor::from_vec(values, (1, 1, seq, seq), valid.device())?.to_dtype(dtype)?;
        mask = mask.broadcast_add(&causal)?;
    }
    Ok(mask)
}

fn scaled_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    scale_denominator: usize,
    softcap: Option<f64>,
) -> CandleResult<Tensor> {
    let mut scores = (q.matmul(&k.transpose(2, 3)?)? * (1.0 / (scale_denominator as f64).sqrt()))?;
    if let Some(cap) = softcap {
        scores = scores.affine(1.0 / cap, 0.0)?.tanh()?.affine(cap, 0.0)?;
    }
    scores = scores.broadcast_add(mask)?;
    let probabilities =
        candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?;
    probabilities.matmul(v)
}

struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    head_dim: usize,
    scale_denominator: usize,
    softcap: f64,
}

impl SelfAttention {
    fn load(cfg: &T5GemmaEncoderConfig, vb: VarBuilder) -> CandleResult<Self> {
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        Ok(Self {
            q: linear_b(cfg.hidden_size, q_dim, false, vb.pp("q_proj"))?,
            k: linear_b(cfg.hidden_size, kv_dim, false, vb.pp("k_proj"))?,
            v: linear_b(cfg.hidden_size, kv_dim, false, vb.pp("v_proj"))?,
            out: linear_b(q_dim, cfg.hidden_size, false, vb.pp("o_proj"))?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
            scale_denominator: cfg.query_pre_attn_scalar,
            softcap: cfg.attn_logit_softcapping,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> CandleResult<Tensor> {
        let q = apply_rope(
            &to_heads(&self.q.forward(x)?, self.heads, self.head_dim)?,
            cos,
            sin,
        )?;
        let k = apply_rope(
            &to_heads(&self.k.forward(x)?, self.heads, self.head_dim)?,
            cos,
            sin,
        )?;
        let v = to_heads(&self.v.forward(x)?, self.heads, self.head_dim)?;
        self.out.forward(&from_heads(&scaled_attention(
            &q,
            &k,
            &v,
            mask,
            self.scale_denominator,
            Some(self.softcap),
        )?)?)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(cfg: &T5GemmaEncoderConfig, vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            gate: linear_b(
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
                vb.pp("gate_proj"),
            )?,
            up: linear_b(
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
                vb.pp("up_proj"),
            )?,
            down: linear_b(
                cfg.intermediate_size,
                cfg.hidden_size,
                false,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate = self.gate.forward(x)?.gelu()?;
        self.down
            .forward(&gate.broadcast_mul(&self.up.forward(x)?)?)
    }
}

struct EncoderLayer {
    pre_attention: T5RmsNorm,
    attention: SelfAttention,
    post_attention: T5RmsNorm,
    pre_feedforward: T5RmsNorm,
    mlp: Mlp,
    post_feedforward: T5RmsNorm,
}

impl EncoderLayer {
    fn load(cfg: &T5GemmaEncoderConfig, vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            pre_attention: T5RmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("pre_self_attn_layernorm"),
            )?,
            attention: SelfAttention::load(cfg, vb.pp("self_attn"))?,
            post_attention: T5RmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_self_attn_layernorm"),
            )?,
            pre_feedforward: T5RmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("pre_feedforward_layernorm"),
            )?,
            mlp: Mlp::load(cfg, vb.pp("mlp"))?,
            post_feedforward: T5RmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_feedforward_layernorm"),
            )?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> CandleResult<Tensor> {
        let residual = x;
        let attention = self
            .attention
            .forward(&self.pre_attention.forward(x)?, mask, cos, sin)?;
        let x = residual.broadcast_add(&self.post_attention.forward(&attention)?)?;
        let residual = &x;
        let feedforward = self.mlp.forward(&self.pre_feedforward.forward(&x)?)?;
        residual.broadcast_add(&self.post_feedforward.forward(&feedforward)?)
    }
}

/// Frozen 12-layer T5Gemma encoder tower. The decoder is intentionally absent.
pub struct T5GemmaEncoder {
    embeddings: Tensor,
    layers: Vec<EncoderLayer>,
    norm: T5RmsNorm,
    cfg: T5GemmaEncoderConfig,
}

impl T5GemmaEncoder {
    pub fn load(cfg: &T5GemmaEncoderConfig, vb: VarBuilder) -> CandleResult<Self> {
        if !matches!(vb.dtype(), DType::BF16 | DType::F32) {
            bail!(
                "T5Gemma encoder compute must be BF16 or F32, got {:?}",
                vb.dtype()
            )
        }
        let vb = vb.pp("model.encoder");
        let embeddings = vb.get((cfg.vocab_size, cfg.hidden_size), "embed_tokens.weight")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::load(cfg, vb.pp("layers").pp(layer))?);
        }
        Ok(Self {
            embeddings,
            layers,
            norm: T5RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?,
            cfg: cfg.clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, valid: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq) = input_ids.dims2()?;
        if valid.dims2()? != (batch, seq) {
            bail!("T5Gemma ids/mask shape mismatch")
        }
        if seq > self.cfg.sliding_window || seq > self.cfg.max_position_embeddings {
            bail!("T5Gemma sequence length {seq} exceeds full-attention support")
        }
        let flat = input_ids.flatten_all()?;
        let mut x =
            self.embeddings
                .index_select(&flat, 0)?
                .reshape((batch, seq, self.cfg.hidden_size))?;
        let scale =
            Tensor::new((self.cfg.hidden_size as f32).sqrt(), x.device())?.to_dtype(x.dtype())?;
        x = x.broadcast_mul(&scale)?;
        let (cos, sin) = rope_tables(
            seq,
            self.cfg.head_dim,
            self.cfg.rope_theta,
            x.dtype(),
            x.device(),
        )?;
        let mask = attention_mask(valid, seq, x.dtype(), false)?;
        for layer in &self.layers {
            x = layer.forward(&x, &mask, &cos, &sin)?;
        }
        self.norm.forward(&x)
    }
}

/// Host tokenization result before tensor materialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizedBatch {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u8>>,
}

fn right_pad_truncate(mut ids: Vec<u32>, max_length: usize, pad_id: u32) -> (Vec<u32>, Vec<u8>) {
    ids.truncate(max_length);
    let valid = ids.len();
    let mut mask = vec![1u8; valid];
    ids.resize(max_length, pad_id);
    mask.resize(max_length, 0);
    (ids, mask)
}

enum Projection {
    Identity,
    Linear { layer: Linear, dtype: DType },
}

impl Projection {
    fn forward(&self, raw: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Identity => Ok(raw.clone()),
            Self::Linear { layer, dtype } => layer.forward(&raw.to_dtype(*dtype)?),
        }
    }
}

fn apply_padding_mode(
    mode: PaddingMode,
    padding_embedding: Option<&Tensor>,
    embeddings: &Tensor,
    valid: &Tensor,
) -> CandleResult<Tensor> {
    let (batch, seq, dim) = embeddings.dims3()?;
    let valid = valid
        .reshape((batch, seq, 1))?
        .broadcast_as((batch, seq, dim))?;
    match mode {
        PaddingMode::None => Ok(embeddings.clone()),
        PaddingMode::Zero => embeddings
            .to_dtype(DType::F32)?
            .broadcast_mul(&valid.to_dtype(DType::F32)?),
        PaddingMode::Learned => {
            let padding = padding_embedding.ok_or_else(|| {
                candle_audio::candle_core::Error::Msg(
                    "learned T5Gemma padding has no embedding".into(),
                )
            })?;
            let dtype = padding.dtype();
            let embeddings = embeddings.to_dtype(dtype)?;
            let padding = padding
                .reshape((1, 1, dim))?
                .broadcast_as((batch, seq, dim))?;
            valid.where_cond(&embeddings, &padding)
        }
    }
}

/// Complete text-conditioner output used by the later DiT story.
pub struct ConditioningOutput {
    pub embeddings: Tensor,
    pub attention_mask: Tensor,
    pub input_ids: Tensor,
}

/// Tokenizer + encoder + optional projection + None/Zero/Learned padding behavior.
pub struct T5GemmaConditioner {
    tokenizer: Tokenizer,
    encoder: T5GemmaEncoder,
    projection: Projection,
    padding_mode: PaddingMode,
    padding_embedding: Option<Tensor>,
    raw_output_dtype: DType,
    max_length: usize,
    pad_id: u32,
}

impl T5GemmaConditioner {
    /// Load the shipped conditioner from one caller-provisioned immutable snapshot.
    pub fn from_layout(layout: &SnapshotLayout, device: &Device) -> AudioResult<Self> {
        if layout.kind != SnapshotKind::Full {
            return Err(AudioError::Msg(
                "T5Gemma conditioner requires a full Stable Audio 3 snapshot".into(),
            ));
        }
        let diffusion = match &layout.config.model {
            ModelConfig::Diffusion(config) => config,
            ModelConfig::Autoencoder(_) => unreachable!(),
        };
        let (id, config) = diffusion
            .conditioning
            .configs
            .iter()
            .find_map(|entry| match entry {
                ConditionerConfig::T5gemma { id, config } => Some((id, config)),
                ConditionerConfig::Number { .. } => None,
            })
            .ok_or_else(|| AudioError::Msg("snapshot has no T5Gemma conditioner".into()))?;
        if config.max_length != SHIPPED_MAX_LENGTH
            || config.padding_mode != PaddingMode::Learned
            || config.project_out
            || diffusion.conditioning.cond_dim != SHIPPED_HIDDEN
        {
            return Err(AudioError::Msg(format!(
                "unsupported shipped T5Gemma conditioner config: {config:?}"
            )));
        }
        let text_config_path = layout
            .text_config_path
            .as_deref()
            .ok_or_else(|| AudioError::Msg("snapshot has no T5Gemma config".into()))?;
        let tokenizer_path = layout
            .tokenizer_path
            .as_deref()
            .ok_or_else(|| AudioError::Msg("snapshot has no tokenizer.json".into()))?;
        let text_config = T5GemmaEncoderConfig::from_path(text_config_path)?;
        // The checkpoint inventory remains BF16. Every backend computes in F32; the explicit
        // location policy preserves CPU raw F32 while Metal and CUDA apply one final BF16 raw
        // boundary before the F32 learned-padding operation.
        let policy = text_compute_policy(device.location());
        let builders = layout.mmap_builders_with_text_dtype(
            policy.conditioned_output_dtype,
            policy.compute_dtype,
            device,
        )?;
        let conditioner = builders
            .conditioner
            .ok_or_else(|| AudioError::Msg("snapshot has no conditioner weights".into()))?
            .pp("conditioners")
            .pp(id);
        let text = builders
            .text_encoder
            .ok_or_else(|| AudioError::Msg("snapshot has no T5Gemma weights".into()))?;
        Self::load(
            config,
            diffusion.conditioning.cond_dim,
            policy.raw_output_dtype,
            &text_config,
            tokenizer_path,
            text,
            conditioner,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load(
        config: &T5GemmaConfig,
        output_dim: usize,
        raw_output_dtype: DType,
        text_config: &T5GemmaEncoderConfig,
        tokenizer_path: &Path,
        text_vb: VarBuilder,
        conditioner_vb: VarBuilder,
    ) -> AudioResult<Self> {
        if !matches!(raw_output_dtype, DType::BF16 | DType::F32) {
            return Err(AudioError::Msg(format!(
                "unsupported T5Gemma raw output dtype {raw_output_dtype:?}"
            )));
        }
        if config.max_length == 0
            || config.max_length > text_config.sliding_window
            || config.max_length > text_config.max_position_embeddings
        {
            return Err(AudioError::Msg(format!(
                "unsupported T5Gemma max_length {}",
                config.max_length
            )));
        }
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| AudioError::Msg(format!("load {}: {e}", tokenizer_path.display())))?;
        let encoder = T5GemmaEncoder::load(text_config, text_vb)?;
        let projection = if text_config.hidden_size != output_dim || config.project_out {
            Projection::Linear {
                layer: linear_b(
                    text_config.hidden_size,
                    output_dim,
                    true,
                    conditioner_vb.pp("proj_out"),
                )?,
                dtype: conditioner_vb.dtype(),
            }
        } else {
            Projection::Identity
        };
        let padding_embedding = (config.padding_mode == PaddingMode::Learned)
            .then(|| {
                conditioner_vb.get_with_hints(
                    output_dim,
                    "padding_embedding",
                    Init::Randn {
                        mean: 0.0,
                        stdev: 0.02,
                    },
                )
            })
            .transpose()?;
        Ok(Self {
            tokenizer,
            encoder,
            projection,
            padding_mode: config.padding_mode,
            padding_embedding,
            raw_output_dtype,
            max_length: config.max_length,
            pad_id: 0,
        })
    }

    pub fn tokenize(&self, prompts: &[String]) -> AudioResult<TokenizedBatch> {
        let mut input_ids = Vec::with_capacity(prompts.len());
        let mut attention_mask = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let encoding = self
                .tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| AudioError::Msg(format!("tokenize T5Gemma prompt: {e}")))?;
            let (ids, mask) =
                right_pad_truncate(encoding.get_ids().to_vec(), self.max_length, self.pad_id);
            input_ids.push(ids);
            attention_mask.push(mask);
        }
        Ok(TokenizedBatch {
            input_ids,
            attention_mask,
        })
    }

    pub fn encode(&self, prompts: &[String]) -> AudioResult<ConditioningOutput> {
        let tokens = self.tokenize(prompts)?;
        let batch = tokens.input_ids.len();
        if batch == 0 {
            return Err(AudioError::Msg(
                "T5Gemma conditioner requires at least one prompt".into(),
            ));
        }
        let ids: Vec<u32> = tokens.input_ids.iter().flatten().copied().collect();
        let mask: Vec<u8> = tokens.attention_mask.iter().flatten().copied().collect();
        let device = self.encoder.embeddings.device();
        let input_ids = Tensor::from_vec(ids, (batch, self.max_length), device)?;
        let attention_mask = Tensor::from_vec(mask, (batch, self.max_length), device)?;
        let raw = self
            .encoder
            .forward(&input_ids, &attention_mask)?
            .to_dtype(self.raw_output_dtype)?;
        let projected = self.projection.forward(&raw)?;
        let embeddings = self.apply_padding(&projected, &attention_mask)?;
        Ok(ConditioningOutput {
            embeddings,
            attention_mask,
            input_ids,
        })
    }

    fn apply_padding(&self, embeddings: &Tensor, valid: &Tensor) -> CandleResult<Tensor> {
        apply_padding_mode(
            self.padding_mode,
            self.padding_embedding.as_ref(),
            embeddings,
            valid,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_inventory_is_exact_and_decoder_free() {
        let keys = encoder_weight_keys();
        assert_eq!(keys.len(), 134);
        assert!(keys.iter().all(|key| key.starts_with("model.encoder.")));
        assert!(!keys.iter().any(|key| key.contains(".decoder.")));
    }

    #[test]
    fn compute_policy_is_exhaustive_without_constructing_hardware_devices() {
        let cases = [
            (
                DeviceLocation::Cpu,
                TextComputePolicy {
                    compute_dtype: DType::F32,
                    raw_output_dtype: DType::F32,
                    conditioned_output_dtype: DType::F32,
                },
            ),
            (
                DeviceLocation::Metal { gpu_id: 7 },
                TextComputePolicy {
                    compute_dtype: DType::F32,
                    raw_output_dtype: DType::BF16,
                    conditioned_output_dtype: DType::F32,
                },
            ),
            (
                DeviceLocation::Cuda { gpu_id: 11 },
                TextComputePolicy {
                    compute_dtype: DType::F32,
                    raw_output_dtype: DType::BF16,
                    conditioned_output_dtype: DType::F32,
                },
            ),
        ];
        for (location, expected) in cases {
            assert_eq!(text_compute_policy(location), expected);
        }
    }

    #[test]
    fn right_truncation_and_padding_are_locked() {
        let (ids, mask) = right_pad_truncate(vec![10, 11, 12, 13, 14], 3, 0);
        assert_eq!(ids, vec![10, 11, 12]);
        assert_eq!(mask, vec![1, 1, 1]);
        let (ids, mask) = right_pad_truncate(vec![10], 3, 0);
        assert_eq!(ids, vec![10, 0, 0]);
        assert_eq!(mask, vec![1, 0, 0]);
    }

    #[test]
    fn all_padding_and_projection_branches_preserve_operation_order() {
        let device = Device::Cpu;
        let raw = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 2, 2), &device).unwrap();
        let valid = Tensor::from_vec(vec![1u8, 0], (1, 2), &device).unwrap();
        let padding = Tensor::from_vec(vec![9f32, 8.0], 2, &device).unwrap();

        let identity = Projection::Identity.forward(&raw).unwrap();
        assert_eq!(
            identity.to_vec3::<f32>().unwrap(),
            raw.to_vec3::<f32>().unwrap()
        );
        let linear = Projection::Linear {
            layer: Linear::new(
                Tensor::from_vec(vec![2f32, 0.0, 0.0, 3.0], (2, 2), &device).unwrap(),
                Some(Tensor::from_vec(vec![1f32, -1.0], 2, &device).unwrap()),
            ),
            dtype: DType::F32,
        }
        .forward(&raw)
        .unwrap();
        assert_eq!(
            linear.to_vec3::<f32>().unwrap(),
            vec![vec![vec![3.0, 5.0], vec![7.0, 11.0]]]
        );

        let none = apply_padding_mode(PaddingMode::None, None, &linear, &valid).unwrap();
        let zero = apply_padding_mode(PaddingMode::Zero, None, &linear, &valid).unwrap();
        let learned =
            apply_padding_mode(PaddingMode::Learned, Some(&padding), &linear, &valid).unwrap();
        assert_eq!(none.to_vec3::<f32>().unwrap()[0][1], vec![7.0, 11.0]);
        assert_eq!(zero.to_vec3::<f32>().unwrap()[0][1], vec![0.0, 0.0]);
        assert_eq!(learned.to_vec3::<f32>().unwrap()[0][0], vec![3.0, 5.0]);
        assert_eq!(learned.to_vec3::<f32>().unwrap()[0][1], vec![9.0, 8.0]);
    }

    #[test]
    fn rms_embedding_geglu_attention_and_padding_mutations_are_observable() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![2f32, -1.0], (1, 1, 2), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let weight = Tensor::from_vec(vec![0.5f32, -0.25], 2, &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let correct = t5_rms_norm(&x, &weight, 1e-6, true).unwrap();
        let direct = t5_rms_norm(&x, &weight, 1e-6, false).unwrap();
        assert!(
            correct
                .to_dtype(DType::F32)
                .unwrap()
                .broadcast_sub(&direct.to_dtype(DType::F32).unwrap())
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                > 0.1
        );

        let scaled = x
            .broadcast_mul(
                &Tensor::new(2f32.sqrt(), &device)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap(),
            )
            .unwrap();
        assert_ne!(
            scaled
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec3::<f32>()
                .unwrap(),
            x.to_dtype(DType::F32).unwrap().to_vec3::<f32>().unwrap()
        );

        let gate = Tensor::from_vec(vec![-1f32, 2.0], (1, 1, 2), &device).unwrap();
        let up = Tensor::from_vec(vec![3f32, -4.0], (1, 1, 2), &device).unwrap();
        let geglu = gate.gelu().unwrap().broadcast_mul(&up).unwrap();
        let swapped = up.gelu().unwrap().broadcast_mul(&gate).unwrap();
        assert!(
            geglu
                .broadcast_sub(&swapped)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                > 0.1
        );

        let q =
            Tensor::from_vec(vec![1f32, 0.0, 0.0, 1.0, 2.0, -1.0], (1, 1, 3, 2), &device).unwrap();
        let k =
            Tensor::from_vec(vec![1f32, 0.0, 0.0, 1.0, -1.0, 2.0], (1, 1, 3, 2), &device).unwrap();
        let v =
            Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 1, 3, 2), &device).unwrap();
        let valid = Tensor::ones((1, 3), DType::U8, &device).unwrap();
        let full_mask = attention_mask(&valid, 3, DType::F32, false).unwrap();
        let causal_mask = attention_mask(&valid, 3, DType::F32, true).unwrap();
        let full = scaled_attention(&q, &k, &v, &full_mask, 2, Some(0.5)).unwrap();
        let causal = scaled_attention(&q, &k, &v, &causal_mask, 2, Some(0.5)).unwrap();
        let no_cap = scaled_attention(&q, &k, &v, &full_mask, 2, None).unwrap();
        for mutated in [&causal, &no_cap] {
            assert!(
                full.broadcast_sub(mutated)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
                    > 1e-3
            );
        }
        let no_valid_keys = Tensor::zeros((1, 3), DType::U8, &device).unwrap();
        let empty_mask = attention_mask(&no_valid_keys, 3, DType::F32, false).unwrap();
        assert!(empty_mask
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
        let empty_bf16_mask = attention_mask(&no_valid_keys, 3, DType::BF16, false).unwrap();
        assert!(empty_bf16_mask
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
        let empty_attention = scaled_attention(&q, &k, &v, &empty_mask, 2, Some(0.5)).unwrap();
        assert!(empty_attention
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));

        let embeddings = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 2, 2), &device).unwrap();
        let mask = Tensor::from_vec(vec![1u8, 0], (1, 2), &device).unwrap();
        let padding = Tensor::from_vec(vec![9f32, 8.0], 2, &device).unwrap();
        let valid = mask
            .reshape((1, 2, 1))
            .unwrap()
            .broadcast_as((1, 2, 2))
            .unwrap();
        let learned = valid
            .where_cond(
                &embeddings,
                &padding
                    .reshape((1, 1, 2))
                    .unwrap()
                    .broadcast_as((1, 2, 2))
                    .unwrap(),
            )
            .unwrap();
        let zero = embeddings
            .broadcast_mul(&valid.to_dtype(DType::F32).unwrap())
            .unwrap();
        for mode_output in [&embeddings, &zero, &learned] {
            assert!(mode_output
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite()));
        }
        assert_eq!(learned.to_vec3::<f32>().unwrap()[0][1], vec![9.0, 8.0]);
        assert_eq!(zero.to_vec3::<f32>().unwrap()[0][1], vec![0.0, 0.0]);
    }

    #[test]
    fn frozen_constructor_defaults_remain_zero_and_128() {
        let config: T5GemmaConfig = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(config.max_length, 128);
        assert_eq!(config.padding_mode, PaddingMode::Zero);
        assert!(!config.project_out);
        assert_eq!(SHIPPED_MAX_LENGTH, 256);
    }
}
