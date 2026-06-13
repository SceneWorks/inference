//! gpt-oss-20b text encoder for Lens (candle port, sc-5108).
//!
//! Lens uses gpt-oss-20b (`GptOssForCausalLM`) **encoder-only**: it runs the decoder forward over the
//! prompt and captures intermediate hidden states (`[5, 11, 17, 23]`, sc-5110) rather than generating.
//! So this is a *non-incremental* forward — a single pass over the full sequence, no KV cache.
//!
//! The block is a from-scratch candle port: candle-transformers ships no `gpt_oss` (epic 5107 Gate-0
//! found upstream PRs #3129/#3581/#3391 all unmerged). It is adapted from the verified-parity
//! reference in candle PR #3581 (logits match HF `modeling_gpt_oss` in bf16, cosine ~0.9996), onto
//! `candle_nn` primitives and the candle-gen workspace idiom. The genuinely gpt-oss-specific pieces —
//! **attention sinks** (per-head learnable logit appended to the softmax), **alternating
//! sliding/full attention**, **YaRN RoPE**, the **clamped-SwiGLU** expert, and **MXFP4** fused-expert
//! weights — have no candle-transformers precedent and are carried over from that reference.
//!
//! Expert weights ship fused + MXFP4 (`gate_up_proj` / `down_proj` as `_blocks` + `_scales`, one e8m0
//! exponent per 32-value block); they are dequantized to bf16 at load (see [`dequant_mxfp4`]). gate/up
//! are interleaved on the output dim. Everything else (attention, router, embeddings) is bf16 per the
//! checkpoint `quantization_config.modules_to_not_convert`. The MXFP4 → GGUF Q4 `QMatMul` transcode
//! that keeps the ~12 GB footprint is a follow-up (sc-5111); this brings the encoder up in bf16 first.

use candle_gen::candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_gen::candle_nn::{
    embedding, linear, ops::sigmoid, ops::softmax_last_dim, rms_norm, Embedding, Linear, Module,
    RmsNorm, VarBuilder,
};

// --- Config -----------------------------------------------------------------

/// YaRN RoPE scaling parameters (`config.rope_parameters` / `rope_scaling`).
#[derive(Debug, Clone)]
pub struct RopeScaling {
    pub rope_type: String,
    pub factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub original_max_position_embeddings: usize,
    /// Not handled (unused for the shipped Lens config); verify at the parity gate if ever set.
    pub truncate: bool,
}

/// gpt-oss encoder config. Field names mirror the HF `GptOssConfig`. Construct via
/// [`Config::gpt_oss_20b`] (the shape Lens ships); a `config.json` loader is a later refinement.
#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub sliding_window: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub attention_bias: bool,
    pub swiglu_limit: f64,
    pub rope_scaling: Option<RopeScaling>,
}

impl Config {
    /// gpt-oss-20b reference shape (24 layers / 32 experts) — ground-truthed from the
    /// `microsoft/Lens` `text_encoder/config.json` (epic 5107).
    pub fn gpt_oss_20b() -> Self {
        Self {
            vocab_size: 201088,
            hidden_size: 2880,
            intermediate_size: 2880,
            num_hidden_layers: 24,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 64,
            num_local_experts: 32,
            num_experts_per_tok: 4,
            sliding_window: 128,
            rope_theta: 150000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 131072,
            attention_bias: true,
            swiglu_limit: 7.0,
            rope_scaling: Some(RopeScaling {
                rope_type: "yarn".to_string(),
                factor: 32.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_max_position_embeddings: 4096,
                truncate: false,
            }),
        }
    }

    /// gpt-oss alternates sliding-window and full attention, starting sliding (layer 0).
    fn is_sliding(&self, layer_idx: usize) -> bool {
        layer_idx.is_multiple_of(2)
    }
}

// --- RoPE (YaRN) ------------------------------------------------------------

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

// YaRN inverse frequencies + attention scaling (mscale), matching HF `_compute_yarn_parameters`:
// high-frequency dims keep the original frequency (extrapolate), low-frequency dims are interpolated
// by 1/factor, with a linear ramp between correction dims derived from beta_fast/beta_slow.
// `truncate` is not handled (unused for the shipped configs); re-verify at the parity gate if set.
fn yarn_inv_freq(cfg: &Config, s: &RopeScaling) -> (Vec<f32>, f32) {
    let dim = cfg.head_dim as f64;
    let base = cfg.rope_theta;
    let half = cfg.head_dim / 2;
    let orig_max = s.original_max_position_embeddings as f64;
    let correction_dim = |num_rot: f64| {
        (dim * (orig_max / (num_rot * 2.0 * std::f64::consts::PI)).ln()) / (2.0 * base.ln())
    };
    let low = correction_dim(s.beta_fast).floor().max(0.0);
    let high = correction_dim(s.beta_slow).ceil().min(dim - 1.0);
    // Guard a degenerate correction range (high == low); only hits for unusual rope_scaling.
    let denom = if (high - low).abs() < 1e-3 {
        1e-3
    } else {
        high - low
    };
    let inv_freq = (0..half)
        .map(|i| {
            let pos_freq = base.powf(2.0 * i as f64 / dim);
            let extrap = 1.0 / pos_freq;
            let interp = 1.0 / (s.factor * pos_freq);
            let ramp = ((i as f64 - low) / denom).clamp(0.0, 1.0);
            (interp * ramp + extrap * (1.0 - ramp)) as f32
        })
        .collect();
    let attn_scale = (0.1 * s.factor.ln() + 1.0) as f32;
    (inv_freq, attn_scale)
}

impl RotaryEmbedding {
    fn new(cfg: &Config, dev: &Device, dtype: DType) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_pos = cfg.max_position_embeddings;
        let (inv_freq, attn_scale) = match &cfg.rope_scaling {
            Some(s) if s.rope_type == "yarn" => yarn_inv_freq(cfg, s),
            _ => (
                (0..dim / 2)
                    .map(|i| 1f32 / (cfg.rope_theta as f32).powf(2.0 * i as f32 / dim as f32))
                    .collect::<Vec<f32>>(),
                1f32,
            ),
        };
        let inv_freq = Tensor::new(inv_freq, dev)?;
        let t = Tensor::arange(0u32, max_pos as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_pos, 1))?;
        let freqs = t.matmul(&inv_freq.reshape((1, dim / 2))?)?;
        // Fold the YaRN attention scaling (mscale) into the cos/sin tables.
        let cos = (freqs.cos()? * attn_scale as f64)?.to_dtype(dtype)?;
        let sin = (freqs.sin()? * attn_scale as f64)?.to_dtype(dtype)?;
        Ok(Self { cos, sin })
    }

    /// Apply rotary embeddings to `q`/`k` (`[b, heads, seq, head_dim]`) at absolute position 0
    /// (encoder-only: a single full-sequence pass).
    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let q = candle_gen::candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_gen::candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

// --- Attention (GQA + sinks + sliding window) -------------------------------

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    sinks: Tensor, // [num_heads] learnable per-head sink logits
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        // gpt-oss attention always carries q/k/v/o biases (config attention_bias = true). Build with
        // `linear` (loads weight + bias); a no-bias config would be a future variant.
        debug_assert!(cfg.attention_bias, "gpt-oss attention_bias must be true");
        Ok(Self {
            q_proj: linear(h, nh * hd, vb.pp("q_proj"))?,
            k_proj: linear(h, nkv * hd, vb.pp("k_proj"))?,
            v_proj: linear(h, nkv * hd, vb.pp("v_proj"))?,
            o_proj: linear(nh * hd, h, vb.pp("o_proj"))?,
            sinks: vb.get(nh, "sinks")?,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
        })
    }

    fn forward(&self, xs: &Tensor, rotary: &RotaryEmbedding, mask: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = rotary.apply(&q, &k)?;
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let scores = scores.broadcast_add(mask)?;
        // Attention sinks: append one learnable logit per head as an extra key column, softmax over
        // [scores | sink], then drop the sink column from the value-weighted sum (it only absorbs
        // probability mass). Because the sink column is always finite, no softmax row is fully masked,
        // so the (finite, dtype-min) mask values below never produce a NaN.
        let n_keys = scores.dim(D::Minus1)?;
        let sinks = self
            .sinks
            .reshape((1, self.num_heads, 1, 1))?
            .broadcast_as((b, self.num_heads, seq_len, 1))?
            .to_dtype(scores.dtype())?
            .contiguous()?;
        let logits = Tensor::cat(&[&scores, &sinks], D::Minus1)?;
        let probs = softmax_last_dim(&logits)?;
        let probs = probs.narrow(D::Minus1, 0, n_keys)?; // drop the sink column
        let out = probs.contiguous()?.matmul(&v.contiguous()?)?;
        let out = out.transpose(1, 2)?.reshape((b, seq_len, ()))?;
        self.o_proj.forward(&out)
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, n_kv, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, n_kv, n_rep, s, d))?
        .reshape((b, n_kv * n_rep, s, d))
}

// Additive causal mask, optionally sliding-window-limited, for an encoder pass (absolute offset 0).
// Query i may attend to key j when j <= i and, when `window` is set, i - j < window (gpt-oss sliding
// layers). Shape [1, 1, seq_len, seq_len]; dtype-min where masked, 0 otherwise. Uses a finite
// large-negative (HF `finfo(dtype).min`, matching the workspace convention) rather than -inf — safe
// for bf16 softmax and parity-faithful to HF's masked_fill.
const MASK_NEG: f32 = -3.389_531_4e38;

fn causal_mask(
    seq_len: usize,
    window: Option<usize>,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut data = vec![0f32; seq_len * seq_len];
    for qi in 0..seq_len {
        for kj in 0..seq_len {
            let out_of_window = window.is_some_and(|w| qi >= kj + w);
            if kj > qi || out_of_window {
                data[qi * seq_len + kj] = MASK_NEG;
            }
        }
    }
    Tensor::from_slice(&data, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)
}

// --- MoE: router + clamped-SwiGLU experts -----------------------------------
// gpt-oss expert activation: gate, up = deinterleave(gate_up); gate clamped to `limit`, up clamped to
// [-limit, limit]; glu = gate * sigmoid(alpha * gate); out = (up + 1) * glu.
const SWIGLU_ALPHA: f64 = 1.702;

// FP4 (e2m1) code -> value lookup, codes 0..7 positive, 8..15 negative.
const FP4_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Dequantize MXFP4 expert weights to `dtype`. Each byte in `blocks` packs two 4-bit e2m1 codes;
/// `scales` holds one e8m0 exponent per 32-value block: `value = FP4_LUT[code] * 2^(scale - 127)`.
/// Done once at load on CPU.
///   blocks: u8 `[E, out, nb, 16]`, scales: u8 `[E, out, nb]` -> dtype `[E, out, nb*32]`
fn dequant_mxfp4(blocks: &Tensor, scales: &Tensor, dtype: DType) -> Result<Tensor> {
    use candle_gen::candle_core::bail;
    let dev = blocks.device().clone();
    let (e, out, nb, bytes) = blocks.dims4()?;
    if scales.dims() != [e, out, nb] {
        bail!(
            "mxfp4 scales shape {:?} incompatible with blocks {:?}",
            scales.dims(),
            blocks.dims()
        );
    }
    let vals = bytes * 2; // 16 packed u8 -> 32 fp4 values
    let in_dim = nb * vals;
    let blk = blocks
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<u8>()?;
    let scl = scales
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<u8>()?;
    let rows = e * out;
    let mut data = vec![0f32; rows * in_dim];
    for row in 0..rows {
        for b in 0..nb {
            let scale = 2f32.powi(scl[row * nb + b] as i32 - 127);
            let blk_off = (row * nb + b) * bytes;
            let out_off = row * in_dim + b * vals;
            for j in 0..bytes {
                let byte = blk[blk_off + j];
                data[out_off + 2 * j] = FP4_LUT[(byte & 0x0f) as usize] * scale;
                data[out_off + 2 * j + 1] = FP4_LUT[(byte >> 4) as usize] * scale;
            }
        }
    }
    Tensor::from_vec(data, (e, out, in_dim), &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(&dev)
}

#[derive(Debug, Clone)]
struct Expert {
    gate_up_proj: Linear, // hidden -> 2*intermediate (gate/up interleaved)
    down_proj: Linear,    // intermediate -> hidden
    limit: f64,
}

impl Expert {
    fn from_weights(
        gate_up_w: Tensor,
        gate_up_b: Tensor,
        down_w: Tensor,
        down_b: Tensor,
        limit: f64,
    ) -> Self {
        Self {
            gate_up_proj: Linear::new(gate_up_w, Some(gate_up_b)),
            down_proj: Linear::new(down_w, Some(down_b)),
            limit,
        }
    }
}

impl Module for Expert {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // gate_up output is interleaved [g0, u0, g1, u1, ...]; reshape (.., I, 2) so [.., i, 0] = gate_i
        // and [.., i, 1] = up_i.
        let gate_up = self.gate_up_proj.forward(xs)?;
        let (n, two_i) = gate_up.dims2()?;
        let gate_up = gate_up.reshape((n, two_i / 2, 2))?;
        let gate = gate_up.i((.., .., 0))?.contiguous()?;
        let up = gate_up.i((.., .., 1))?.contiguous()?;
        let gate = gate.clamp(f64::NEG_INFINITY, self.limit)?;
        let up = up.clamp(-self.limit, self.limit)?;
        let glu = (&gate * sigmoid(&(&gate * SWIGLU_ALPHA)?)?)?;
        let act = ((up + 1.0)? * glu)?;
        self.down_proj.forward(&act)
    }
}

#[derive(Debug, Clone)]
struct SparseMoe {
    router: Linear, // gpt-oss router has a bias
    experts: Vec<Expert>,
    num_experts_per_tok: usize,
}

impl SparseMoe {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let router = linear(cfg.hidden_size, cfg.num_local_experts, vb.pp("router"))?;
        let e = cfg.num_local_experts;
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let dtype = vb.dtype();
        let vb_e = vb.pp("experts");
        // Fused MXFP4 expert weights, dequantized once: blocks/scales load as raw u8
        // (get_unchecked_dtype avoids the dtype coercion get() would apply).
        let gate_up_w = dequant_mxfp4(
            &vb_e.get_unchecked_dtype("gate_up_proj_blocks", DType::U8)?,
            &vb_e.get_unchecked_dtype("gate_up_proj_scales", DType::U8)?,
            dtype,
        )?; // [E, 2*inter, hidden]
        let gate_up_b = vb_e.get((e, 2 * i), "gate_up_proj_bias")?;
        let down_w = dequant_mxfp4(
            &vb_e.get_unchecked_dtype("down_proj_blocks", DType::U8)?,
            &vb_e.get_unchecked_dtype("down_proj_scales", DType::U8)?,
            dtype,
        )?; // [E, hidden, inter]
        let down_b = vb_e.get((e, h), "down_proj_bias")?;
        let experts = (0..e)
            .map(|x| {
                Ok(Expert::from_weights(
                    gate_up_w.i(x)?.contiguous()?,
                    gate_up_b.i(x)?.contiguous()?,
                    down_w.i(x)?.contiguous()?,
                    down_b.i(x)?.contiguous()?,
                    cfg.swiglu_limit,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            router,
            experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
        })
    }
}

impl Module for SparseMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq_len, hidden) = xs.dims3()?;
        let xs = xs.reshape(((), hidden))?;
        let router_logits = self.router.forward(&xs)?;
        let routing = softmax_last_dim(&router_logits)?;
        // PERF: full sort to take top-k; replace with a topk when candle exposes one. Note:
        // softmax-over-all then renormalize-over-topk is identical to HF's topk-then-softmax (the
        // normalizer cancels), so this matches the reference.
        let sel = routing
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;
        let weights = routing.gather(&sel, D::Minus1)?;
        let weights = weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let sel = sel.to_vec2::<u32>()?;

        let mut top_x = vec![vec![]; self.experts.len()];
        let mut top_w = vec![vec![]; self.experts.len()];
        for (row, (w, idxs)) in weights.iter().zip(sel.iter()).enumerate() {
            let sum: f32 = w.iter().sum();
            let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };
            for (&w, &e) in w.iter().zip(idxs.iter()) {
                top_x[e as usize].push(row as u32);
                top_w[e as usize].push(w * inv_sum); // normalize over the top-k
            }
        }
        let mut ys = xs.zeros_like()?;
        for (e, expert) in self.experts.iter().enumerate() {
            if top_x[e].is_empty() {
                continue;
            }
            let idx = Tensor::new(top_x[e].as_slice(), xs.device())?;
            let w = Tensor::new(top_w[e].as_slice(), xs.device())?
                .reshape(((), 1))?
                .to_dtype(xs.dtype())?;
            let state = xs.index_select(&idx, 0)?;
            let out = expert.forward(&state)?.broadcast_mul(&w)?;
            ys = ys.index_add(&idx, &out, 0)?;
        }
        ys.reshape((b, seq_len, hidden))
    }
}

// --- Decoder layer / Encoder ------------------------------------------------

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: SparseMoe,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    is_sliding: bool,
}

impl DecoderLayer {
    fn new(cfg: &Config, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: SparseMoe::new(cfg, vb.pp("mlp"))?,
            input_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            is_sliding: cfg.is_sliding(layer_idx),
        })
    }

    fn forward(&self, xs: &Tensor, rotary: &RotaryEmbedding, mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let h = self.input_layernorm.forward(xs)?;
        let h = self.self_attn.forward(&h, rotary, mask)?;
        let xs = (residual + h)?;
        let residual = &xs;
        let h = self.post_attention_layernorm.forward(&xs)?;
        let h = self.mlp.forward(&h)?;
        residual + h
    }
}

/// Output of an encoder forward pass.
pub struct EncoderOutput {
    /// Per-layer hidden states in HF `output_hidden_states` order: index 0 = token embeddings
    /// (pre-layer-0), index `i` = residual-stream output of layer `i-1`. Length = `num_hidden_layers + 1`.
    /// Lens captures `[5, 11, 17, 23]` from this (sc-5110).
    pub hidden_states: Vec<Tensor>,
    /// The final RMSNorm applied to the last layer's output. `[b, seq, hidden]`.
    pub last_hidden_state: Tensor,
}

/// gpt-oss-20b used encoder-only: a single full-sequence forward that captures per-layer hidden states.
pub struct GptOssTextEncoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    rotary: RotaryEmbedding,
    sliding_window: usize,
    device: Device,
    dtype: DType,
}

impl GptOssTextEncoder {
    /// `vb` is the `text_encoder` root: tensors load as `model.embed_tokens`, `model.layers.N.*`,
    /// `model.norm` (the `lm_head` is unused for the encoder).
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary = RotaryEmbedding::new(cfg, vb.device(), vb.dtype())?;
        let vb_l = vb_m.pp("layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| DecoderLayer::new(cfg, i, vb_l.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Run the encoder over `input_ids` (`[b, seq]`, u32), capturing every layer's hidden state.
    pub fn forward(&self, input_ids: &Tensor) -> Result<EncoderOutput> {
        let (_b, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        // Full-causal mask for full-attention layers, sliding-window mask for the alternating sliding
        // layers; selected per layer.
        let full = causal_mask(seq_len, None, &self.device, self.dtype)?;
        let sliding = causal_mask(seq_len, Some(self.sliding_window), &self.device, self.dtype)?;
        let mut hidden_states = Vec::with_capacity(self.layers.len() + 1);
        hidden_states.push(xs.clone());
        for layer in self.layers.iter() {
            let mask = if layer.is_sliding { &sliding } else { &full };
            xs = layer.forward(&xs, &self.rotary, mask)?;
            hidden_states.push(xs.clone());
        }
        let last_hidden_state = self.norm.forward(&xs)?;
        Ok(EncoderOutput {
            hidden_states,
            last_hidden_state,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Dequantize a single 16-byte block (32 values) at the given e8m0 scale.
    fn one_block(bytes: Vec<u8>, scale: u8) -> Vec<f32> {
        let dev = Device::Cpu;
        let blocks = Tensor::from_vec(bytes, (1usize, 1, 1, 16), &dev).unwrap();
        let scales = Tensor::from_vec(vec![scale], (1usize, 1, 1), &dev).unwrap();
        let out = dequant_mxfp4(&blocks, &scales, DType::F32).unwrap();
        assert_eq!(out.dims(), &[1, 1, 32]);
        out.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn mxfp4_codes_and_nibble_order() {
        // byte = (high << 4) | low; low nibble -> even index, high -> odd index.
        let mut bytes = vec![0u8; 16];
        bytes[0] = 0x21; // low 1 -> 0.5, high 2 -> 1.0
        bytes[1] = 0xF7; // low 7 -> 6.0, high 15 -> -6.0
        bytes[2] = 0x8A; // low 10 -> -1.0, high 8 -> -0.0
        let v = one_block(bytes, 127); // 2^(127-127) = 1.0
        assert_eq!(&v[..6], &[0.5, 1.0, 6.0, -6.0, -1.0, -0.0]);
        assert!(v[6..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn mxfp4_scale_exponent() {
        let mut bytes = vec![0u8; 16];
        bytes[0] = 0x21; // 0.5, 1.0 pre-scale
        assert_eq!(&one_block(bytes.clone(), 128)[..2], &[1.0, 2.0]); // x2
        assert_eq!(&one_block(bytes, 126)[..2], &[0.25, 0.5]); // x0.5
    }

    #[test]
    fn mxfp4_shape_multi_block() {
        let dev = Device::Cpu;
        let (e, out, nb) = (2usize, 3usize, 2usize);
        let blocks = Tensor::zeros((e, out, nb, 16), DType::U8, &dev).unwrap();
        let scales = Tensor::zeros((e, out, nb), DType::U8, &dev).unwrap();
        let t = dequant_mxfp4(&blocks, &scales, DType::F32).unwrap();
        assert_eq!(t.dims(), &[e, out, nb * 32]);
    }

    #[test]
    fn yarn_freqs_and_scale() {
        let cfg = Config::gpt_oss_20b();
        let s = cfg.rope_scaling.clone().unwrap();
        let (inv_freq, scale) = yarn_inv_freq(&cfg, &s);
        assert_eq!(inv_freq.len(), cfg.head_dim / 2);
        // mscale = 0.1*ln(factor) + 1
        assert!((scale - (0.1f32 * 32f32.ln() + 1.0)).abs() < 1e-5);
        // dim 0 is high-frequency -> pure extrapolation (original freq 1.0).
        assert!((inv_freq[0] - 1.0).abs() < 1e-6);
        // last dim is low-frequency -> interpolated by 1/factor.
        let last = inv_freq[cfg.head_dim / 2 - 1];
        let expect = 1.0 / ((cfg.rope_theta as f32).powf(62.0 / 64.0) * 32.0);
        assert!(
            (last / expect - 1.0).abs() < 1e-3,
            "last={last} expect={expect}"
        );
    }

    #[test]
    fn sliding_mask_is_tighter_than_full() {
        let dev = Device::Cpu;
        let (seq, window) = (6usize, 3usize);
        let to_vec = |w| {
            causal_mask(seq, w, &dev, DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let full = to_vec(None);
        let slide = to_vec(Some(window));
        // Anything masked by the full causal mask must also be masked when sliding.
        for (f, s) in full.iter().zip(&slide) {
            if *f == MASK_NEG {
                assert_eq!(*s, MASK_NEG);
            }
        }
        // Sliding must mask strictly more (far keys past the window).
        let extra = full
            .iter()
            .zip(&slide)
            .filter(|(f, s)| **f == 0.0 && **s == MASK_NEG)
            .count();
        assert!(extra > 0, "sliding window should mask additional far keys");
    }

    #[test]
    fn is_sliding_alternates_starting_sliding() {
        let cfg = Config::gpt_oss_20b();
        assert!(cfg.is_sliding(0));
        assert!(!cfg.is_sliding(1));
        assert!(cfg.is_sliding(22));
        assert!(!cfg.is_sliding(23));
    }
}
