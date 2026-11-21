//! Shared CLIP-L and T5 text encoders for FLUX.1 and Mochi.
//!
//! Based on the Candle CLIP/T5 encoders at 1e6aa85 and the FLUX.1 packed encoder port.
//! Native FLUX keeps BFL's dense projection and config conventions. Diffusers FLUX and
//! Mochi use the same encoder topology with dense or packed storage, and Mochi supplies
//! its token padding mask and activation dtype.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Activation, LayerNorm, Module, VarBuilder};

use crate::quant::{embedding, AdaptLinear, QEmbedding};

/// Checkpoint-specific storage policy used within the shared encoder graphs.
#[derive(Clone, Copy)]
enum Storage {
    Dense,
    PackedDetect,
}

impl Storage {
    fn linear(
        self,
        vb: &VarBuilder<'_>,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<AdaptLinear> {
        match self {
            Self::Dense => Ok(AdaptLinear::from_dense(
                candle_nn::linear_b(input, output, bias, vb.pp(name))?,
                input,
                output,
            )),
            Self::PackedDetect => AdaptLinear::linear_detect(input, output, vb, name, bias),
        }
    }

    fn embedding(
        self,
        vb: &VarBuilder<'_>,
        name: &str,
        vocab: usize,
        hidden: usize,
    ) -> Result<QEmbedding> {
        match self {
            Self::Dense => QEmbedding::embedding(vocab, hidden, vb.pp(name)),
            Self::PackedDetect => embedding(vb, name, vocab, hidden),
        }
    }
}

// ============================================================================================
// CLIP-L text encoder (openai/clip-vit-large-patch14 layout, `text_model.` prefix).
// ============================================================================================

/// Fixed CLIP-L text config FLUX uses (identical across schnell/dev).
pub struct ClipConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
}

impl ClipConfig {
    pub fn flux() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 768,
            intermediate_size: 3072,
            max_position_embeddings: 77,
            num_hidden_layers: 12,
            num_attention_heads: 12,
        }
    }
}

const CLIP_LN_EPS: f64 = 1e-5;

/// `nn::layer_norm` (affine) via explicit weight/bias load — the packed tier keeps the CLIP LayerNorms
/// dense (they carry no `.scales`), so a plain load suffices.
fn clip_layer_norm(dim: usize, vb: &VarBuilder, prefix: &str) -> Result<LayerNorm> {
    let w = vb.get(dim, &format!("{prefix}.weight"))?;
    let b = vb.get(dim, &format!("{prefix}.bias"))?;
    Ok(LayerNorm::new(w, b, CLIP_LN_EPS))
}

struct ClipEmbeddings {
    token_embedding: QEmbedding,
    position_embedding: QEmbedding,
    position_ids: Tensor,
}

impl ClipEmbeddings {
    fn new(cfg: &ClipConfig, vb: &VarBuilder, storage: Storage) -> Result<Self> {
        let emb = vb.pp("embeddings");
        Ok(Self {
            token_embedding: storage.embedding(
                &emb,
                "token_embedding",
                cfg.vocab_size,
                cfg.embed_dim,
            )?,
            position_embedding: storage.embedding(
                &emb,
                "position_embedding",
                cfg.max_position_embeddings,
                cfg.embed_dim,
            )?,
            position_ids: Tensor::arange(0u32, cfg.max_position_embeddings as u32, vb.device())?
                .unsqueeze(0)?,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.dim(D::Minus1)?;
        let inputs_embeds = self.token_embedding.forward(input_ids)?;
        let position_ids = self.position_ids.narrow(1, 0, seq_length)?;
        let position_embedding = self.position_embedding.forward(&position_ids)?;
        inputs_embeds.broadcast_add(&position_embedding)
    }
}

struct ClipAttention {
    q_proj: AdaptLinear,
    k_proj: AdaptLinear,
    v_proj: AdaptLinear,
    out_proj: AdaptLinear,
    head_dim: usize,
    scale: f64,
    num_heads: usize,
}

impl ClipAttention {
    fn new(cfg: &ClipConfig, vb: &VarBuilder, storage: Storage) -> Result<Self> {
        let d = cfg.embed_dim;
        let head_dim = d / cfg.num_attention_heads;
        let lin = |name| storage.linear(vb, name, d, d, true);
        Ok(Self {
            q_proj: lin("q_proj")?,
            k_proj: lin("k_proj")?,
            v_proj: lin("v_proj")?,
            out_proj: lin("out_proj")?,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
            num_heads: cfg.num_attention_heads,
        })
    }

    fn shape(&self, xs: &Tensor, seq_len: usize, bsz: usize) -> Result<Tensor> {
        xs.reshape((bsz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, xs: &Tensor, causal_mask: &Tensor) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let (bsz, seq_len, embed_dim) = xs.dims3()?;
        let proj_shape = (bsz * self.num_heads, seq_len, self.head_dim);
        let q = (self.q_proj.forward(xs)? * self.scale)?;
        let q = self
            .shape(&q, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let k = self
            .shape(&self.k_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let v = self
            .shape(&self.v_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let attn_weights = q.matmul(&k.transpose(1, 2)?)?;
        let src_len = k.dim(1)?;
        let attn_weights = attn_weights
            .reshape((bsz, self.num_heads, seq_len, src_len))?
            .broadcast_add(causal_mask)?
            .reshape((bsz * self.num_heads, seq_len, src_len))?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?.to_dtype(in_dtype)?;
        let attn_output = attn_output
            .reshape((bsz, self.num_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((bsz, seq_len, embed_dim))?;
        self.out_proj.forward(&attn_output)
    }
}

struct ClipMlp {
    fc1: AdaptLinear,
    fc2: AdaptLinear,
}

impl ClipMlp {
    fn new(cfg: &ClipConfig, vb: &VarBuilder, storage: Storage) -> Result<Self> {
        Ok(Self {
            fc1: storage.linear(vb, "fc1", cfg.embed_dim, cfg.intermediate_size, true)?,
            fc2: storage.linear(vb, "fc2", cfg.intermediate_size, cfg.embed_dim, true)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        // QuickGELU: x * sigmoid(1.702 x).
        let xs = (&xs * candle_nn::ops::sigmoid(&(&xs * 1.702f64)?)?)?;
        self.fc2.forward(&xs)
    }
}

struct ClipLayer {
    self_attn: ClipAttention,
    layer_norm1: LayerNorm,
    mlp: ClipMlp,
    layer_norm2: LayerNorm,
}

impl ClipLayer {
    fn new(cfg: &ClipConfig, vb: &VarBuilder, storage: Storage) -> Result<Self> {
        Ok(Self {
            self_attn: ClipAttention::new(cfg, &vb.pp("self_attn"), storage)?,
            layer_norm1: clip_layer_norm(cfg.embed_dim, vb, "layer_norm1")?,
            mlp: ClipMlp::new(cfg, &vb.pp("mlp"), storage)?,
            layer_norm2: clip_layer_norm(cfg.embed_dim, vb, "layer_norm2")?,
        })
    }

    fn forward(&self, xs: &Tensor, causal_mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = __model_proof_output!(self.layer_norm1.forward(xs))?;
        let xs = self.self_attn.forward(&xs, causal_mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let ys = __model_proof_output!(self.layer_norm2.forward(&xs))?;
        let ys = self.mlp.forward(&ys)?;
        ys + residual
    }
}

/// The CLIP-L text transformer — token+position embed → 12 pre-norm layers → final LayerNorm,
/// pooled at the argmax-EOT position (FLUX's `vec`/`y` conditioning).
pub struct ClipTextEncoder {
    embeddings: ClipEmbeddings,
    layers: Vec<ClipLayer>,
    final_layer_norm: LayerNorm,
}

impl ClipTextEncoder {
    /// Load from `vb` rooted at the CLIP `text_model.` prefix (the pipeline passes `vb.pp("text_model")`).
    pub fn new(cfg: &ClipConfig, vb: VarBuilder) -> Result<Self> {
        Self::load(cfg, vb, Storage::PackedDetect)
    }

    /// Load the native BFL dense CLIP-L checkpoint.
    pub fn new_bfl(cfg: &ClipConfig, vb: VarBuilder) -> Result<Self> {
        Self::load(cfg, vb, Storage::Dense)
    }

    fn load(cfg: &ClipConfig, vb: VarBuilder, storage: Storage) -> Result<Self> {
        let embeddings = ClipEmbeddings::new(cfg, &vb, storage)?;
        let enc = vb.pp("encoder").pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(ClipLayer::new(cfg, &enc.pp(i), storage)?);
        }
        let final_layer_norm = clip_layer_norm(cfg.embed_dim, &vb, "final_layer_norm")?;
        Ok(Self {
            embeddings,
            layers,
            final_layer_norm,
        })
    }

    fn causal_mask(bsz: usize, seq_len: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::MIN } else { 0. }))
            .collect();
        Tensor::from_slice(&mask, (seq_len, seq_len), device)?
            .broadcast_as((bsz, 1, seq_len, seq_len))
    }

    /// The pooled `[B, 768]` vector — the causal-attended stack read at each row's argmax (EOT) token,
    /// exactly like the stock `ClipTextTransformer` `Module::forward`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (bsz, seq_len) = input_ids.dims2()?;
        let mut xs = self.embeddings.forward(input_ids)?;
        let mask = Self::causal_mask(bsz, seq_len, input_ids.device())?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &mask)?;
        }
        let output = __model_proof_output!(self.final_layer_norm.forward(&xs))?;
        let seq_max = input_ids.argmax(D::Minus1)?.to_dtype(DType::I64)?;
        let mut indices = Vec::new();
        for (b, &s) in seq_max.to_vec1::<i64>()?.iter().enumerate() {
            indices.push(output.i((b, s as usize))?.unsqueeze(0)?);
        }
        Tensor::cat(&indices, 0)
    }
}

// ============================================================================================
// T5-XXL encoder (google/t5-v1_1-xxl layout: `shared`, `encoder.block.*`, `encoder.final_layer_norm`).
// ============================================================================================

/// Geometry and normalization configuration for the encoder-only T5 graph.
pub struct T5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub layer_norm_epsilon: f64,
}

impl T5Config {
    pub fn xxl() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 4096,
            d_kv: 64,
            d_ff: 10240,
            num_layers: 24,
            num_heads: 64,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_epsilon: 1e-6,
        }
    }
}

/// T5 RMSNorm (the "T5 layer norm": mean-of-squares in f32, scale by the dense `weight`).
struct T5LayerNorm {
    weight: Tensor,
    eps: f64,
}

impl T5LayerNorm {
    fn new(dim: usize, eps: f64, vb: &VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, &format!("{prefix}.weight"))?,
            eps,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        __model_proof_output!(xs
            .to_dtype(dtype)?
            .broadcast_mul(&self.weight.to_dtype(dtype)?))
    }
}

/// Native BFL T5 used Candle's broadcasted no-bias matrix multiply. Diffusers and
/// Mochi used the shared projection's regular/upcast path. Keep that kernel choice at
/// the projection so both routes use the same attention and feed-forward graph.
struct T5Projection {
    linear: AdaptLinear,
    storage: Storage,
}

impl T5Projection {
    fn new(
        storage: Storage,
        vb: &VarBuilder<'_>,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        Ok(Self {
            linear: storage.linear(vb, name, input, output, false)?,
            storage,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let Storage::Dense = self.storage else {
            return self.linear.forward_upcast(xs);
        };
        let dense = self
            .linear
            .dense_base()
            .ok_or_else(|| candle_core::Error::Msg("native T5 projection is not dense".into()))?;
        let weight = dense.weight().to_dtype(xs.dtype())?;
        let weight = match *xs.dims() {
            [b1, b2, _, _] => weight.broadcast_left((b1, b2))?,
            [batch, _, _] => weight.broadcast_left(batch)?,
            _ => weight,
        };
        xs.matmul(&weight.t()?)
    }
}

struct T5LayerFF {
    wi: T5Projection,
    gate: Option<T5Projection>,
    wo: T5Projection,
    activation: Activation,
    layer_norm: T5LayerNorm,
}

impl T5LayerFF {
    fn new(
        cfg: &T5Config,
        vb: &VarBuilder,
        storage: Storage,
        gated_activation: Option<Activation>,
    ) -> Result<Self> {
        let dense = vb.pp("DenseReluDense");
        let input = |name| T5Projection::new(storage, &dense, name, cfg.d_model, cfg.d_ff);
        let (wi, gate, activation) = match gated_activation {
            Some(activation) => (input("wi_0")?, Some(input("wi_1")?), activation),
            // The reference's ungated feed-forward always uses ReLU.
            None => (input("wi")?, None, Activation::Relu),
        };
        Ok(Self {
            wi,
            gate,
            wo: T5Projection::new(storage, &dense, "wo", cfg.d_ff, cfg.d_model)?,
            activation,
            layer_norm: T5LayerNorm::new(cfg.d_model, cfg.layer_norm_epsilon, vb, "layer_norm")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let normed = self.layer_norm.forward(xs)?;
        let mut hidden = self.activation.forward(&self.wi.forward(&normed)?)?;
        if let Some(gate) = &self.gate {
            hidden = hidden.broadcast_mul(&gate.forward(&normed)?)?;
        }
        xs + self.wo.forward(&hidden)?
    }
}

struct T5Attention {
    q: T5Projection,
    k: T5Projection,
    v: T5Projection,
    o: T5Projection,
    relative_attention_bias: Option<QEmbedding>,
    n_heads: usize,
    d_kv: usize,
    inner_dim: usize,
    num_buckets: usize,
    max_distance: usize,
}

impl T5Attention {
    fn new(has_rel_bias: bool, cfg: &T5Config, vb: &VarBuilder, storage: Storage) -> Result<Self> {
        let inner_dim = cfg.num_heads * cfg.d_kv;
        Ok(Self {
            q: T5Projection::new(storage, vb, "q", cfg.d_model, inner_dim)?,
            k: T5Projection::new(storage, vb, "k", cfg.d_model, inner_dim)?,
            v: T5Projection::new(storage, vb, "v", cfg.d_model, inner_dim)?,
            o: T5Projection::new(storage, vb, "o", inner_dim, cfg.d_model)?,
            relative_attention_bias: if has_rel_bias {
                Some(storage.embedding(
                    vb,
                    "relative_attention_bias",
                    cfg.relative_attention_num_buckets,
                    cfg.num_heads,
                )?)
            } else {
                None
            },
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
            inner_dim,
            num_buckets: cfg.relative_attention_num_buckets,
            max_distance: cfg.relative_attention_max_distance,
        })
    }

    /// The bidirectional relative-position-bias buckets — copied verbatim from the stock T5 encoder
    /// (`use_cache = false`, encoder path: `q_start = 0`, `q_end = kv_len`).
    fn relative_buckets(&self, q_len: usize, kv_len: usize, device: &Device) -> Result<Tensor> {
        let num_buckets = self.num_buckets as u32 / 2;
        let max_exact = num_buckets / 2;
        let rel: Vec<Vec<u32>> = (0..q_len as u32)
            .map(|i| {
                (0..kv_len as u32)
                    .map(|j| {
                        if i < j {
                            if j - i < max_exact {
                                j - i + num_buckets
                            } else {
                                let b = f32::log(
                                    (j - i) as f32 / max_exact as f32,
                                    self.max_distance as f32 / max_exact as f32,
                                ) * (num_buckets - max_exact) as f32;
                                u32::min(
                                    max_exact + num_buckets + b as u32,
                                    self.num_buckets as u32 - 1,
                                )
                            }
                        } else if i - j < max_exact {
                            i - j
                        } else {
                            let b = f32::log(
                                (i - j) as f32 / max_exact as f32,
                                self.max_distance as f32 / max_exact as f32,
                            ) * (num_buckets - max_exact) as f32;
                            u32::min(max_exact + b as u32, num_buckets - 1)
                        }
                    })
                    .collect()
            })
            .collect();
        Tensor::new(rel, device)
    }

    /// Encoder self-attention. `position_bias` is threaded from block 0 (only block 0 owns
    /// `relative_attention_bias`), matching the stock encoder. `mask` is an **optional** additive
    /// key-padding mask (broadcastable to the scores `[B, heads, q_len, kv_len]`, e.g. `[1, 1, 1, L]`
    /// with a large negative at padded keys), added after the position bias — Mochi's masked encode
    /// (`_get_t5_prompt_embeds`) supplies it; the FLUX dense path passes `None` (byte-identical to the
    /// pre-mask behavior).
    fn forward(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (b_sz, q_len) = (xs.dim(0)?, xs.dim(1)?);
        let kv_len = q_len;
        // `forward_upcast`: inert (byte-identical) when weights already match `xs`'s dtype — the FLUX /
        // Chroma path — and lets Mochi run f32 activations over bf16-resident weights.
        let q = self.q.forward(xs)?;
        let k = self.k.forward(xs)?;
        let v = self.v.forward(xs)?;
        let to_heads = |t: Tensor| -> Result<Tensor> {
            t.reshape((b_sz, q_len, self.n_heads, self.d_kv))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = to_heads(q)?;
        let k = to_heads(k)?;
        let v = to_heads(v)?;
        let scores = q.matmul(&k.t()?)?;
        let position_bias = match position_bias {
            Some(pb) => pb.clone(),
            None => {
                let buckets = self.relative_buckets(q_len, kv_len, q.device())?;
                self.relative_attention_bias
                    .as_ref()
                    .expect("block 0 owns the relative_attention_bias")
                    .forward(&buckets)?
                    .permute((2, 0, 1))?
                    .unsqueeze(0)?
                    .to_dtype(scores.dtype())?
            }
        };
        let scores = scores.broadcast_add(&position_bias)?;
        // Additive key-padding mask (Mochi's masked encode); `None` = the FLUX byte-exact path.
        let scores = match mask {
            Some(m) => scores.broadcast_add(&m.to_dtype(scores.dtype())?)?,
            None => scores,
        };
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.inner_dim))?;
        Ok((self.o.forward(&out)?, position_bias))
    }
}

struct T5LayerSelfAttention {
    self_attention: T5Attention,
    layer_norm: T5LayerNorm,
}

impl T5LayerSelfAttention {
    fn new(has_rel_bias: bool, cfg: &T5Config, vb: &VarBuilder, storage: Storage) -> Result<Self> {
        Ok(Self {
            self_attention: T5Attention::new(has_rel_bias, cfg, &vb.pp("SelfAttention"), storage)?,
            layer_norm: T5LayerNorm::new(cfg.d_model, cfg.layer_norm_epsilon, vb, "layer_norm")?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let normed = self.layer_norm.forward(xs)?;
        let (ys, pb) = self.self_attention.forward(&normed, position_bias, mask)?;
        Ok(((xs + ys)?, pb))
    }
}

struct T5Block {
    self_attn: T5LayerSelfAttention,
    ff: T5LayerFF,
}

impl T5Block {
    fn new(
        has_rel_bias: bool,
        cfg: &T5Config,
        vb: &VarBuilder,
        storage: Storage,
        gated_activation: Option<Activation>,
    ) -> Result<Self> {
        let vb = vb.pp("layer");
        Ok(Self {
            self_attn: T5LayerSelfAttention::new(has_rel_bias, cfg, &vb.pp("0"), storage)?,
            ff: T5LayerFF::new(cfg, &vb.pp("1"), storage, gated_activation)?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (xs, pb) = self.self_attn.forward(xs, position_bias, mask)?;
        Ok((self.ff.forward(&xs)?, pb))
    }
}

/// The T5 encoder — `shared` embed → 24 encoder blocks (relative bias on block 0) → final
/// RMSNorm. Returns the `[B, L, 4096]` sequence FLUX consumes as `txt`.
pub struct T5Encoder {
    shared: QEmbedding,
    blocks: Vec<T5Block>,
    final_layer_norm: T5LayerNorm,
}

impl T5Encoder {
    /// Load from `vb` rooted at the T5 checkpoint root (`shared`, `encoder.block.*`,
    /// `encoder.final_layer_norm`).
    pub fn new(cfg: &T5Config, vb: VarBuilder) -> Result<Self> {
        Self::load(
            cfg,
            vb,
            "shared",
            Storage::PackedDetect,
            Some(Activation::NewGelu),
        )
    }

    /// Load the native BFL T5 encoder and honor the config's feed-forward layout.
    /// `None` selects the legacy ungated ReLU branch.
    pub fn new_bfl(
        cfg: &T5Config,
        vb: VarBuilder,
        gated_activation: Option<Activation>,
    ) -> Result<Self> {
        let prefix = bfl_t5_embedding_prefix(|name| vb.contains_tensor(name));
        Self::load(cfg, vb, prefix, Storage::Dense, gated_activation)
    }

    fn load(
        cfg: &T5Config,
        vb: VarBuilder,
        embedding_prefix: &str,
        storage: Storage,
        gated_activation: Option<Activation>,
    ) -> Result<Self> {
        let shared = storage.embedding(&vb, embedding_prefix, cfg.vocab_size, cfg.d_model)?;
        let enc = vb.pp("encoder");
        let block_vb = enc.pp("block");
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(T5Block::new(
                i == 0,
                cfg,
                &block_vb.pp(i),
                storage,
                gated_activation,
            )?);
        }
        let final_layer_norm = T5LayerNorm::new(
            cfg.d_model,
            cfg.layer_norm_epsilon,
            &enc,
            "final_layer_norm",
        )?;
        Ok(Self {
            shared,
            blocks,
            final_layer_norm,
        })
    }

    /// Encode `input_ids` `[B, L]` → `[B, L, 4096]`. `out_dtype` casts the token embedding to the
    /// compute dtype (bf16) before the encoder, as the stock `forward_dt` does. Unmasked (FLUX runs T5
    /// unmasked); byte-identical to `forward_masked(input_ids, out_dtype, None)`.
    pub fn forward(&self, input_ids: &Tensor, out_dtype: DType) -> Result<Tensor> {
        self.forward_masked(input_ids, out_dtype, None)
    }

    /// As [`forward`](Self::forward), but with an optional **additive** key-padding mask
    /// (broadcastable to the per-block attention scores `[B, heads, L, L]`, e.g. `[1, 1, 1, L]` with a
    /// large negative at padded keys). Mochi's `_get_t5_prompt_embeds` runs T5 **with** the tokenizer
    /// padding mask (unlike FLUX, which runs it unmasked), so padded tokens don't pollute the real-token
    /// embeddings. `mask = None` is byte-identical to [`forward`](Self::forward).
    pub fn forward_masked(
        &self,
        input_ids: &Tensor,
        out_dtype: DType,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden = self.shared.forward(input_ids)?.to_dtype(out_dtype)?;
        let mut position_bias: Option<Tensor> = None;
        for block in &self.blocks {
            let (h, pb) = block.forward(&hidden, position_bias.as_ref(), mask)?;
            hidden = h;
            position_bias = Some(pb);
        }
        self.final_layer_norm.forward(&hidden)
    }
}

fn bfl_t5_embedding_prefix(contains: impl Fn(&str) -> bool) -> &'static str {
    if contains("shared.weight") {
        "shared"
    } else if contains("decoder.embed_tokens")
        || (!contains("encoder.embed_tokens.weight") && contains("decoder.embed_tokens.weight"))
    {
        "decoder.embed_tokens"
    } else {
        "encoder.embed_tokens"
    }
}

#[cfg(test)]
mod tests {
    use super::bfl_t5_embedding_prefix;

    #[test]
    fn bfl_t5_embedding_keeps_native_key_precedence_and_fallback() {
        let prefix = |present: &[&str]| bfl_t5_embedding_prefix(|name| present.contains(&name));
        assert_eq!(
            prefix(&[
                "shared.weight",
                "decoder.embed_tokens",
                "encoder.embed_tokens.weight"
            ]),
            "shared"
        );
        assert_eq!(
            prefix(&["decoder.embed_tokens", "encoder.embed_tokens.weight"]),
            "decoder.embed_tokens"
        );
        assert_eq!(
            prefix(&["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]),
            "encoder.embed_tokens"
        );
        assert_eq!(
            prefix(&["decoder.embed_tokens.weight"]),
            "decoder.embed_tokens"
        );
        assert_eq!(prefix(&[]), "encoder.embed_tokens");
    }
}
