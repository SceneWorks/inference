//! Z-Image Qwen3 text encoder shared by dense and packed checkpoints.
//!
//! The graph follows Candle's Z-Image Qwen3 encoder. It returns the raw second-to-last
//! decoder layer with no final normalization. Dense sources use Candle's normal projection
//! layout. Packed sources use the shared Candle spine for projections and token embeddings,
//! retaining packed weights and dequantizing only the selected embedding rows.

use std::sync::Arc;

use candle_gen::attention::repeat_kv;
use candle_gen::candle_core::{DType, Device, Module, Result, Tensor};
use candle_gen::candle_nn::{Activation, RmsNorm, VarBuilder};
use candle_gen::grounding::Rotary;
use candle_gen::quant::{self, QEmbedding, QLinear};

pub(crate) use candle_transformers::models::z_image::text_encoder::TextEncoderConfig;

#[derive(Clone, Copy)]
enum WeightStorage {
    Dense,
    DetectPacked,
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) fn combined_fp8_qwen3_q_projection(vb: &VarBuilder) -> Result<QLinear> {
    let cfg = TextEncoderConfig::z_image();
    WeightStorage::Dense.linear(
        vb,
        "q_proj",
        cfg.hidden_size,
        cfg.num_attention_heads * cfg.head_dim,
        cfg.attention_bias,
    )
}

impl WeightStorage {
    fn linear(
        self,
        vb: &VarBuilder,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<QLinear> {
        match self {
            Self::Dense if bias => QLinear::linear(input, output, vb.pp(name)),
            Self::Dense => QLinear::linear_no_bias(input, output, vb.pp(name)),
            Self::DetectPacked => QLinear::linear_detect(input, output, vb, name, bias),
        }
    }

    fn embedding(
        self,
        vb: &VarBuilder,
        name: &str,
        vocab: usize,
        hidden: usize,
    ) -> Result<QEmbedding> {
        match self {
            Self::Dense => QEmbedding::embedding(vocab, hidden, vb.pp(name)),
            Self::DetectPacked => quant::embedding(vb, name, vocab, hidden),
        }
    }
}

/// The text tower's RoPE tables. `inv_freq` is derived in **f64** (`rope_theta.powf(i / dim)` then
/// cast) — unlike [`Rotary::new`]'s f32 power — so the tables are built here and only the application
/// is shared (sc-27284).
fn rotary_embedding(dtype: DType, cfg: &TextEncoderConfig, dev: &Device) -> Result<Rotary> {
    let dim = cfg.head_dim;
    let max_seq_len = cfg.max_position_embeddings;
    let inv_freq: Vec<_> = (0..dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
        .collect();
    let inv_freq_len = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
    let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
        .to_dtype(DType::F32)?
        .reshape((max_seq_len, 1))?;
    let freqs = t.matmul(&inv_freq)?;
    Ok(Rotary::from_tables(
        freqs.cos()?.to_dtype(dtype)?,
        freqs.sin()?.to_dtype(dtype)?,
    ))
}

struct Mlp {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &TextEncoderConfig, vb: VarBuilder, storage: WeightStorage) -> Result<Self> {
        Ok(Self {
            gate_proj: storage.linear(
                &vb,
                "gate_proj",
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
            )?,
            up_proj: storage.linear(
                &vb,
                "up_proj",
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
            )?,
            down_proj: storage.linear(
                &vb,
                "down_proj",
                cfg.intermediate_size,
                cfg.hidden_size,
                false,
            )?,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(x)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<Rotary>,
}

impl Attention {
    fn new(
        cfg: &TextEncoderConfig,
        rotary_emb: Arc<Rotary>,
        vb: VarBuilder,
        storage: WeightStorage,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        // Qwen3 attention is bias-less (`cfg.attention_bias == false`).
        let bias = cfg.attention_bias;
        let q_proj = storage.linear(&vb, "q_proj", cfg.hidden_size, num_heads * head_dim, bias)?;
        let k_proj = storage.linear(
            &vb,
            "k_proj",
            cfg.hidden_size,
            num_kv_heads * head_dim,
            bias,
        )?;
        let v_proj = storage.linear(
            &vb,
            "v_proj",
            cfg.hidden_size,
            num_kv_heads * head_dim,
            bias,
        )?;
        let o_proj = storage.linear(&vb, "o_proj", num_heads * head_dim, cfg.hidden_size, bias)?;

        let q_norm = candle_gen::candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = candle_gen::candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let hidden_size = head_dim * cfg.num_attention_heads;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb,
        })
    }

    fn forward(&self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head RMSNorm (Qwen3).
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = __model_proof_output!(self.q_norm.forward(&q_flat))?;
        let k_flat = __model_proof_output!(self.k_norm.forward(&k_flat))?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply_at(&q, &k, offset)?;

        let k = repeat_kv(&k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(&v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_gen::candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        let ctx = ctx.transpose(1, 2)?.reshape((b, l, self.hidden_size))?;
        self.o_proj.forward(&ctx)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &TextEncoderConfig,
        rotary: Arc<Rotary>,
        vb: VarBuilder,
        storage: WeightStorage,
    ) -> Result<Self> {
        let self_attn = Attention::new(cfg, rotary, vb.pp("self_attn"), storage)?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"), storage)?;
        let ln1 = candle_gen::candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let ln2 = candle_gen::candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = __model_proof_output!(self.ln1.forward(x))?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = __model_proof_output!(self.ln2.forward(&x))?;
        let h2 = self.mlp.forward(&h2)?;
        x + h2
    }
}

/// Qwen3 conditioning from the checkpoint's `model.*` keys.
pub(crate) struct ZImageTextEncoder {
    embed_tokens: QEmbedding,
    layers: Vec<DecoderLayer>,
    num_hidden_layers: usize,
    device: Device,
    dtype: DType,
}

impl ZImageTextEncoder {
    /// Load a dense snapshot, normalized import, or training checkpoint.
    pub(crate) fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Self::load(cfg, vb, WeightStorage::Dense)
    }

    /// Load a packed snapshot, retaining dense tensors at keys without affine triples.
    pub(crate) fn new_packed(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Self::load(cfg, vb, WeightStorage::DetectPacked)
    }

    fn load(cfg: &TextEncoderConfig, vb: VarBuilder, storage: WeightStorage) -> Result<Self> {
        let vb_model = vb.pp("model");
        let embed_tokens =
            storage.embedding(&vb_model, "embed_tokens", cfg.vocab_size, cfg.hidden_size)?;

        let rotary = Arc::new(rotary_embedding(vb.dtype(), cfg, vb.device())?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_model.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                cfg,
                rotary.clone(),
                vb_layers.pp(i),
                storage,
            )?);
        }
        // The final norm (`model.norm.weight`) is intentionally NOT loaded — the encoder returns the
        // second-to-last layer output without the final norm.

        Ok(Self {
            embed_tokens,
            layers,
            num_hidden_layers: cfg.num_hidden_layers,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| if j <= i + offset { 0.0 } else { minf })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    /// Encode `input_ids` `(B, seq_len)` → the layer[-2] hidden states `(B, seq_len, hidden)`, WITHOUT
    /// the final RMSNorm (the Z-Image convention the pipeline's `cap_feats` relies on).
    pub(crate) fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, l) = input_ids.dims2()?;
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, 0)?)
        };

        let target_layer = self.num_hidden_layers - 2;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, causal.as_ref(), 0)?;
            if i == target_layer {
                return Ok(hidden_states);
            }
        }
        candle_gen::candle_core::bail!("z-image te: layer index out of bounds")
    }
}

#[cfg(test)]
mod parity_tests {
    //! Check both dense loading paths against the stock encoder with shared `VarMap` weights.
    use super::*;
    use candle_gen::candle_core::{Device, Tensor};
    use candle_gen::candle_nn::{VarBuilder, VarMap};
    use candle_transformers::models::z_image::text_encoder::ZImageTextEncoder as StockTe;

    #[test]
    fn dense_loading_ignores_affine_siblings() -> Result<()> {
        let dev = Device::Cpu;
        let weights = [
            (
                "tokens.weight".into(),
                Tensor::zeros((4, 64), DType::F32, &dev)?,
            ),
            (
                "tokens.scales".into(),
                Tensor::ones((4, 1), DType::F32, &dev)?,
            ),
            (
                "proj.weight".into(),
                Tensor::zeros((4, 64), DType::F32, &dev)?,
            ),
            (
                "proj.scales".into(),
                Tensor::ones((4, 1), DType::F32, &dev)?,
            ),
        ]
        .into_iter()
        .collect();
        let vb = VarBuilder::from_tensors(weights, DType::F32, &dev);
        assert!(WeightStorage::Dense
            .embedding(&vb, "tokens", 4, 64)?
            .resident_storage()
            .is_none());
        assert!(!WeightStorage::Dense
            .linear(&vb, "proj", 64, 4, false)?
            .is_quantized());
        assert!(WeightStorage::DetectPacked
            .embedding(&vb, "tokens", 4, 64)
            .is_err());
        assert!(WeightStorage::DetectPacked
            .linear(&vb, "proj", 64, 4, false)
            .is_err());
        Ok(())
    }

    /// A tiny Qwen3 config: hidden 32, 4 heads / 2 kv-heads, head_dim 8, 4 layers — enough to exercise
    /// GQA repeat + per-head norm + the layer[-2] return cheaply on CPU.
    fn tiny_cfg() -> TextEncoderConfig {
        let mut cfg = TextEncoderConfig::z_image();
        cfg.vocab_size = 64;
        cfg.hidden_size = 32;
        cfg.intermediate_size = 48;
        cfg.num_hidden_layers = 4;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 2;
        cfg.head_dim = 8;
        cfg
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn vendored_dense_te_matches_stock_forward() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // The dense constructor populates the VarMap. Both the packed loader's dense fallback
        // and the stock encoder read those same weights.
        let dense = ZImageTextEncoder::new(&cfg, vb.clone()).unwrap();
        let packed_fallback = ZImageTextEncoder::new_packed(&cfg, vb.clone()).unwrap();
        let stock = StockTe::new(&cfg, vb).unwrap();

        let ids = Tensor::from_vec(vec![1u32, 5, 12, 0, 3, 7], (1, 6), &dev).unwrap();
        let y_s = __model_proof_output!(stock.forward(&ids)).unwrap();
        for encoder in [dense, packed_fallback] {
            let y_v = __model_proof_output!(encoder.forward(&ids)).unwrap();
            assert_eq!(y_v.dims(), y_s.dims());
            let diff = (y_v - &y_s)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(diff < 1e-5, "dense TE diverged from stock by {diff}");
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED text_encoder::parity_tests::vendored_dense_te_matches_stock_forward");
    }
}
