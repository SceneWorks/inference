//! Gemma-2-2B-IT decoder — PiD's caption text encoder. Port of HF `Gemma2Model` (the
//! `.get_decoder()` stack PiD loads: embedding → 26 norm-sandwich decoder layers → final RMSNorm →
//! last-hidden `[B, L, 2304]`; no lm_head / final-logit-softcap needed).
//!
//! Gemma-2 specifics (vs the Gemma-3 LTX port): **attention logit soft-capping** `50·tanh(s/50)`
//! pre-softmax (so no fused SDPA — explicit attention), **no q/k norm**, RoPE is the standard HF
//! **rotate_half** convention (not PiD's interleaved), attention scale `query_pre_attn_scalar^-0.5`,
//! GQA (8 query / 4 KV heads, head_dim 256 — independent of hidden 2304), gelu-tanh MLP, and the
//! norm-sandwich block (input → attn → post_attn → +res → pre_ff → mlp → post_ff → +res). RMSNorm is
//! Gemma's `x·rsqrt(mean(x²)+eps)·(1+w)`; token embeddings are scaled by `√hidden_size`.
//!
//! PiD captions are ≤300 tokens ≪ the 4096 sliding window, so every layer is plain full-causal —
//! the local/global distinction collapses and a single causal (+ optional padding) mask suffices.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{
    add, broadcast_to, concatenate_axis, matmul, multiply, negative, softmax_axis, split, tanh,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::scalar;
use mlx_gen::nn::{gelu_tanh, TokenEmbedding};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Gemma-2 decoder configuration.
#[derive(Debug, Clone)]
pub struct Gemma2Config {
    pub hidden_size: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    pub rope_theta: f32,
    pub attn_softcap: f32,
    pub query_pre_attn_scalar: f32,
    pub rms_eps: f32,
}

impl Gemma2Config {
    /// The released `gemma-2-2b-it` config.
    pub fn gemma_2_2b() -> Self {
        Self {
            hidden_size: 2304,
            num_layers: 26,
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 256,
            intermediate_size: 9216,
            rope_theta: 10000.0,
            attn_softcap: 50.0,
            query_pre_attn_scalar: 256.0,
            rms_eps: 1e-6,
        }
    }
}

fn lin(w: &Weights, key: &str) -> Result<AdaptableLinear> {
    // Packed-detect (Group-B, sc-8489): route through the shared `quant::lin` so a pre-quantized
    // Gemma-2 (the SANA Q4/Q8 tier — `{base}.scales` present) loads its projections packed, while a
    // dense Gemma-2 (every PiD decoder — no `.scales`) loads dense EXACTLY as before (bit-identical
    // `AdaptableLinear::dense`). Callers pass the full `{base}.weight` key; Gemma-2 Linears are
    // bias-free. `embed_tokens` uses the shared packed embedding loader below.
    let base = key.strip_suffix(".weight").unwrap_or(key);
    mlx_gen::quant::lin(w, base, false, mlx_gen::quant::DEFAULT_GROUP_SIZE)
}

/// Gemma RMSNorm with the precomputed `(1 + weight)`. The reference computes the normalization in
/// **fp32** (`_norm(x.float()) · (1 + w.float())`) then casts back to the input dtype — load-bearing
/// on bf16: a bf16-internal reduction drifts hugely over Gemma-2's 104 norms with O(50) activations
/// (measured: 18% peak-rel). We upcast x + weight to f32, normalize, then cast back (f32 path is a
/// no-op cast, so the tiny f32 fixture is unaffected).
fn rms(x: &Array, one_plus_w: &Array, eps: f32) -> Result<Array> {
    let xf = x.as_dtype(Dtype::Float32)?;
    let wf = one_plus_w.as_dtype(Dtype::Float32)?;
    Ok(rms_norm(&xf, &wf, eps)?.as_dtype(x.dtype())?)
}

/// Host `(cos, sin)` `[seq, head_dim]` for HF rotate_half RoPE: `emb = cat(freqs, freqs)`.
fn rope_tables(head_dim: i32, seq: i32, theta: f32) -> (Array, Array) {
    let half = (head_dim / 2) as usize;
    let inv: Vec<f64> = (0..half)
        .map(|i| 1.0 / (theta as f64).powf((2 * i) as f64 / head_dim as f64))
        .collect();
    let hd = head_dim as usize;
    let s = seq as usize;
    let mut cos = vec![0f32; s * hd];
    let mut sin = vec![0f32; s * hd];
    for p in 0..s {
        for j in 0..hd {
            let f = inv[j % half]; // emb = cat(freqs, freqs) -> index wraps at half
            let a = p as f64 * f;
            cos[p * hd + j] = a.cos() as f32;
            sin[p * hd + j] = a.sin() as f32;
        }
    }
    (
        Array::from_slice(&cos, &[seq, head_dim]),
        Array::from_slice(&sin, &[seq, head_dim]),
    )
}

/// `rotate_half(x) = cat(-x[..,h:], x[..,:h])` for `[B,H,L,D]`.
fn rotate_half(x: &Array) -> Result<Array> {
    let p = split(x, 2, 3)?; // [x1, x2] halves along the head-dim axis
    Ok(concatenate_axis(&[&negative(&p[1])?, &p[0]], 3)?)
}

/// `q·cos + rotate_half(q)·sin` with `cos`/`sin` `[L, D]` broadcast over `[B,H,L,D]`.
fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let (l, d) = (cos.shape()[0], cos.shape()[1]);
    let cos = cos.reshape(&[1, 1, l, d])?;
    let sin = sin.reshape(&[1, 1, l, d])?;
    let xc = multiply(x, &cos.as_dtype(x.dtype())?)?;
    let xs = multiply(&rotate_half(x)?, &sin.as_dtype(x.dtype())?)?;
    Ok(add(&xc, &xs)?)
}

/// Repeat KV heads `n_rep×` along the head axis (`[B,nkv,L,D]` → `[B,nkv·n_rep,L,D]`), matching HF
/// `repeat_kv` (kv head j → q heads `[j·n_rep .. (j+1)·n_rep)`).
fn repeat_kv(x: &Array, n_rep: i32) -> Result<Array> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    let (b, nkv, l, d) = (sh[0], sh[1], sh[2], sh[3]);
    let expanded = broadcast_to(&x.reshape(&[b, nkv, 1, l, d])?, &[b, nkv, n_rep, l, d])?;
    Ok(expanded.reshape(&[b, nkv * n_rep, l, d])?)
}

struct Attention {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    softcap: f32,
}

impl Attention {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Gemma2Config) -> Result<Self> {
        Ok(Self {
            q: lin(w, &format!("{prefix}.q_proj.weight"))?,
            k: lin(w, &format!("{prefix}.k_proj.weight"))?,
            v: lin(w, &format!("{prefix}.v_proj.weight"))?,
            o: lin(w, &format!("{prefix}.o_proj.weight"))?,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            scale: cfg.query_pre_attn_scalar.powf(-0.5),
            softcap: cfg.attn_softcap,
        })
    }

    /// `x`: `[B,L,hidden]`; `cos`/`sin`: `[L,head_dim]`; `mask`: additive `[1,1,L,L]` (or broadcast).
    fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, l) = (sh[0], sh[1]);
        let hd = self.head_dim;
        let to_heads = |a: Array, nh: i32| -> Result<Array> {
            Ok(a.reshape(&[b, l, nh, hd])?.transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = apply_rope(&to_heads(self.q.forward(x)?, self.num_heads)?, cos, sin)?;
        let k = apply_rope(&to_heads(self.k.forward(x)?, self.num_kv_heads)?, cos, sin)?;
        let v = to_heads(self.v.forward(x)?, self.num_kv_heads)?;
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        // explicit attention (logit soft-cap blocks fused SDPA)
        let scores = multiply(
            &matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?,
            &scalar(self.scale).as_dtype(q.dtype())?,
        )?;
        // softcap·tanh(scores/softcap)
        let cap = scalar(self.softcap).as_dtype(scores.dtype())?;
        let scores = multiply(
            &tanh(&multiply(
                &scores,
                &scalar(1.0 / self.softcap).as_dtype(scores.dtype())?,
            )?)?,
            &cap,
        )?;
        // softmax in f32 then back to the input dtype (HF eager: `softmax(..., dtype=float32)`).
        let scores = add(
            &scores.as_dtype(Dtype::Float32)?,
            &mask.as_dtype(Dtype::Float32)?,
        )?;
        let attn = softmax_axis(&scores, -1, false)?.as_dtype(x.dtype())?;
        let out = matmul(&attn, &v)?; // [B,H,L,D]
        let out = out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, l, self.num_heads * hd])?;
        self.o.forward(&out)
    }
}

struct Mlp {
    gate: AdaptableLinear,
    up: AdaptableLinear,
    down: AdaptableLinear,
}

impl Mlp {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: lin(w, &format!("{prefix}.gate_proj.weight"))?,
            up: lin(w, &format!("{prefix}.up_proj.weight"))?,
            down: lin(w, &format!("{prefix}.down_proj.weight"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let g = gelu_tanh(&self.gate.forward(x)?)?;
        self.down.forward(&multiply(&g, &self.up.forward(x)?)?)
    }
}

struct Layer {
    input_ln: Array,
    attn: Attention,
    post_attn_ln: Array,
    pre_ff_ln: Array,
    mlp: Mlp,
    post_ff_ln: Array,
    eps: f32,
}

impl Layer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Gemma2Config) -> Result<Self> {
        let onep = |key: &str| -> Result<Array> {
            Ok(add(
                w.require(key)?,
                &scalar(1.0).as_dtype(w.require(key)?.dtype())?,
            )?)
        };
        Ok(Self {
            input_ln: onep(&format!("{prefix}.input_layernorm.weight"))?,
            attn: Attention::from_weights(w, &format!("{prefix}.self_attn"), cfg)?,
            post_attn_ln: onep(&format!("{prefix}.post_attention_layernorm.weight"))?,
            pre_ff_ln: onep(&format!("{prefix}.pre_feedforward_layernorm.weight"))?,
            mlp: Mlp::from_weights(w, &format!("{prefix}.mlp"))?,
            post_ff_ln: onep(&format!("{prefix}.post_feedforward_layernorm.weight"))?,
            eps: cfg.rms_eps,
        })
    }

    /// Bytes of every dense projection weight copy a forward at `compute` makes (its stored dtype
    /// differs from the activations), SUMMED: the layer is one quantum under one `eval`, and MLX
    /// encodes ahead of execution — a promoted copy is allocated when its cast is encoded and
    /// released only when the command buffer (Metal) or queued task (CPU) of the matmul consuming
    /// it completes, so in op order (q, k, v, o, gate, up, down) every copy can be live before the
    /// first consumer retires. Packed projections run `quantized_matmul` on the stored codes and
    /// same-dtype dense weights are used in place: 0.
    fn promoted_weight_bytes(&self, compute: Dtype) -> u64 {
        [
            &self.attn.q,
            &self.attn.k,
            &self.attn.v,
            &self.attn.o,
            &self.mlp.gate,
            &self.mlp.up,
            &self.mlp.down,
        ]
        .into_iter()
        .filter_map(|lin| lin.dense_weight())
        .filter(|(w, _)| w.dtype() != compute)
        .map(|(w, _)| w.size() as u64 * dtype_bytes(w.dtype()).max(dtype_bytes(compute)))
        .fold(0u64, u64::saturating_add)
    }

    fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let h = self
            .attn
            .forward(&rms(x, &self.input_ln, self.eps)?, cos, sin, mask)?;
        let x = add(x, &rms(&h, &self.post_attn_ln, self.eps)?)?;
        let h = self.mlp.forward(&rms(&x, &self.pre_ff_ln, self.eps)?)?;
        Ok(add(&x, &rms(&h, &self.post_ff_ln, self.eps)?)?)
    }
}

fn dtype_bytes(dtype: Dtype) -> u64 {
    match dtype {
        Dtype::Float16 | Dtype::Bfloat16 => 2,
        _ => 4,
    }
}

/// Live `[B, L, hidden]` buffers of one [`rms`], in units of its `element_bytes`-wide input: the
/// `fast::rms_norm` evaluation holds two f32 buffers at once (MLX's unfused CPU fallback computes
/// `x²`, then its mean-normalized product; the Metal kernel is fused and holds only the output, so
/// the CPU count bounds both). A narrower input adds its f32 widened copy and the cast back; for f32
/// both casts are the input itself.
fn rms_live(element_bytes: u64) -> u64 {
    if element_bytes >= 4 {
        2
    } else {
        // widened copy (2 units) + two f32 buffers (4 units) + narrow cast (1 unit), plus one
        // unit for the reduction temporaries the CPU fallback allocates around the widened input
        // (measured 7.37 units at 192x64 bf16).
        8
    }
}

/// One Gemma-2 decoder layer quantum (norm sandwich → GQA soft-capped attention → norm sandwich →
/// gelu-tanh MLP), COUNTED from `Layer::forward` at the activation width `element_bytes`.
///
/// Attention phase, in `[B, L, hidden]` buffers (q/k/v widths `heads·head_dim` and
/// `kv_heads·head_dim` never exceed `hidden` for Gemma-2, so each is priced at hidden width): the
/// residual `x`, the input-norm output plus its f32 transients (`rms_live`), the q/k/v projections
/// (3), the RoPE'd q and k with one `rotate_half` concat/negate transient each (4), and the two
/// `repeat_kv` broadcast copies (2). Scores are MATERIALIZED (soft-capping blocks fused SDPA):
/// `[B, heads, L, L]` with three f32-wide buffers live at once: each scale/softcap/cast/mask step
/// consumes its predecessor, but the softmax reads the masked logits while writing its output and
/// a third buffer (the narrow logits still referenced by their f32 widening, or the CPU backend's
/// softmax/GEMM temporary at f32) is live with them — measured up to 3.0 score buffers. MLP phase: the residual, the pre-ff norm output and its
/// transients, and the post-ff norm on the down projection; `[B, L, intermediate]` buffers: the
/// gate projection (held to gelu-tanh's final product), the gelu chain's cube/inner/tanh term and
/// its `0.5·x` factor (the chain's elementwise steps reuse their predecessor, so two live), the
/// output, and the up projection (5 live). `weight_bytes` is
/// `Layer::promoted_weight_bytes`.
pub fn gemma_layer_bytes(
    cfg: &Gemma2Config,
    batch: u64,
    tokens: u64,
    element_bytes: u64,
    weight_bytes: u64,
) -> u64 {
    use mlx_gen::ios_quantum::{attention_block_bytes, AttentionBlockGeometry, Scores};
    attention_block_bytes(AttentionBlockGeometry {
        batch,
        tokens,
        dim: cfg.hidden_size as u64,
        heads: cfg.num_heads as u64,
        context_tokens: 0,
        ff_hidden: cfg.intermediate_size as u64,
        query_rows: tokens,
        element_bytes,
        weight_bytes,
        attn_hidden_live: GEMMA_ATTN_HIDDEN_LIVE + rms_live(element_bytes),
        context_live: 0,
        scores: Scores::Materialized {
            live: 3,
            element_bytes: 4,
        },
        mlp_hidden_live: 2 + rms_live(element_bytes),
        ff_live: GEMMA_FF_LIVE,
    })
}

const GEMMA_ATTN_HIDDEN_LIVE: u64 = 1 + 3 + 4 + 2;
const GEMMA_FF_LIVE: u64 = 5;

/// The embedding/mask quantum: the rows gathered at the stored width, their f32 widening (the
/// shared embedding's output), the native cast back and the `sqrt(hidden)`-scaled copy — for f32
/// storage the widening and the cast are the gathered rows themselves, leaving two buffers; the two host-built f32 RoPE tables `[L, head_dim]`;
/// and ONE f32 `[B, 1, L, L]` causal mask — the padding term is a `[B, 1, 1, L]` row and its sum
/// with the causal table reuses the table's buffer (its only reference), plus that row.
pub fn gemma_embedding_bytes(
    cfg: &Gemma2Config,
    batch: u64,
    tokens: u64,
    element_bytes: u64,
) -> u64 {
    use mlx_gen::ios_quantum::{elementwise_bytes, tensor_bytes};
    let hidden = cfg.hidden_size as u64;
    let rows = batch.saturating_mul(tokens).saturating_mul(hidden);
    let native_copies = if element_bytes >= 4 { 1 } else { 3 };
    elementwise_bytes(rows, 1)
        .saturating_add(tensor_bytes(element_bytes, &[native_copies, rows]))
        .saturating_add(tensor_bytes(4, &[2, tokens, cfg.head_dim as u64]))
        .saturating_add(tensor_bytes(4, &[batch, tokens, tokens]))
        .saturating_add(tensor_bytes(4, &[batch, tokens]))
}

/// The final-norm quantum: `rms_live` `[B, L, hidden]` buffers at the activation width plus the
/// f32 per-row `[B, L, 1]` mean square and its reciprocal root (both live at the normalize).
pub fn gemma_final_norm_bytes(
    cfg: &Gemma2Config,
    batch: u64,
    tokens: u64,
    element_bytes: u64,
) -> u64 {
    use mlx_gen::ios_quantum::tensor_bytes;
    tensor_bytes(
        element_bytes,
        &[
            rms_live(element_bytes),
            batch,
            tokens,
            cfg.hidden_size as u64,
        ],
    )
    .saturating_add(tensor_bytes(4, &[2, batch, tokens]))
}

/// Validate the exact affine embedding layout before constructing any gather graph.
fn gemma_embedding(w: &Weights, base: &str, hidden: i32) -> Result<TokenEmbedding> {
    let weight = w.require(&format!("{base}.weight"))?;
    let scales = w.get(&format!("{base}.scales"));
    let biases = w.get(&format!("{base}.biases"));
    let valid = if let Some(scales) = scales {
        let _bits =
            mlx_gen::quant::packed_bits(weight, scales, mlx_gen::quant::DEFAULT_GROUP_SIZE)?;
        biases.is_some_and(|biases| {
            biases.shape() == scales.shape() && biases.dtype() == scales.dtype()
        }) && scales.shape()[1] * mlx_gen::quant::DEFAULT_GROUP_SIZE == hidden
    } else {
        biases.is_none()
            && weight.shape().len() == 2
            && weight.shape()[1] == hidden
            && matches!(
                weight.dtype(),
                Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
            )
    };
    if !valid {
        return Err(mlx_gen::Error::Msg(format!(
            "Gemma-2: invalid dense/packed embedding layout at {base} (hidden {hidden})"
        )));
    }
    mlx_gen::quant::embedding(w, base, mlx_gen::quant::DEFAULT_GROUP_SIZE)
}

/// The Gemma-2 decoder (caption encoder).
pub struct Gemma2 {
    embed: TokenEmbedding, // dense [vocab, hidden] or packed affine Q4/Q8
    layers: Vec<Layer>,
    norm: Array,
    cfg: Gemma2Config,
}

impl Gemma2 {
    /// `prefix` is `"model."` for the HF checkpoint layout.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &Gemma2Config) -> Result<Self> {
        let layers = (0..cfg.num_layers)
            .map(|i| Layer::from_weights(w, &format!("{prefix}layers.{i}"), cfg))
            .collect::<Result<Vec<_>>>()?;
        let norm = add(
            w.require(&format!("{prefix}norm.weight"))?,
            &scalar(1.0).as_dtype(w.require(&format!("{prefix}norm.weight"))?.dtype())?,
        )?;
        Ok(Self {
            embed: gemma_embedding(w, &format!("{prefix}embed_tokens"), cfg.hidden_size)?,
            layers,
            norm,
            cfg: cfg.clone(),
        })
    }

    /// The activation dtype of the forward: the stored embedding dtype (dense weight, or the packed
    /// table's scales), which every layer then keeps.
    pub fn compute_dtype(&self) -> Dtype {
        match &self.embed {
            TokenEmbedding::Dense(w) => w.dtype(),
            TokenEmbedding::Quantized { scales, .. } => scales.dtype(),
        }
    }

    /// The embedding/mask quantum body: scaled token embeddings `[B, L, hidden]` in `dtype`, the
    /// RoPE tables and the additive causal (+ padding) mask.
    fn embed_inputs(
        &self,
        ids: &Array,
        pad_mask: Option<&Array>,
        dtype: Dtype,
    ) -> Result<(Array, Array, Array, Array)> {
        let sh = ids.shape();
        let (b, l) = (sh[0], sh[1]);
        let hidden = self.cfg.hidden_size;
        // embed + √hidden scale (cast to embedding dtype, per the reference)
        let flat = ids.reshape(&[b * l])?;
        // TokenEmbedding gathers packed rows BEFORE dequantizing, never the full vocabulary.
        // Its common forward returns f32; Gemma must retain the stored embedding dtype.
        let emb = self
            .embed
            .forward(&flat)?
            .as_dtype(dtype)?
            .reshape(&[b, l, hidden])?;
        let normalizer = scalar((hidden as f32).sqrt()).as_dtype(emb.dtype())?;
        let x = multiply(&emb, &normalizer)?;

        let (cos, sin) = rope_tables(self.cfg.head_dim, l, self.cfg.rope_theta);
        let mask = causal_mask(b, l, pad_mask)?;
        Ok((x, cos, sin, mask))
    }

    /// `ids`: `[B, L]` (i32). `pad_mask`: optional `[B, L]` (1 = real, 0 = pad). Returns the
    /// last-hidden states `[B, L, hidden]` in the embedding dtype.
    pub fn forward(&self, ids: &Array, pad_mask: Option<&Array>) -> Result<Array> {
        let sh = ids.shape();
        let (b, l) = (sh[0], sh[1]);
        let _quantum_scope = mlx_gen::ios_quantum::Scope::new();
        let dtype = self.compute_dtype();
        let element_bytes = dtype_bytes(dtype);
        let (b64, l64) = (b as u64, l as u64);
        mlx_gen::ios_quantum::before(
            "Gemma embedding/mask",
            gemma_embedding_bytes(&self.cfg, b64, l64, element_bytes),
        )?;

        let (mut x, cos, sin, mask) = self.embed_inputs(ids, pad_mask, dtype)?;
        mlx_gen::ios_quantum::finish("Gemma embedding/mask", &[&x, &cos, &sin, &mask])?;
        for layer in &self.layers {
            let scratch = gemma_layer_bytes(
                &self.cfg,
                b64,
                l64,
                element_bytes,
                layer.promoted_weight_bytes(dtype),
            );
            x = mlx_gen::ios_quantum::run("Gemma layer", scratch, || {
                layer.forward(&x, &cos, &sin, &mask)
            })?;
        }
        mlx_gen::ios_quantum::run(
            "Gemma final norm",
            gemma_final_norm_bytes(&self.cfg, b64, l64, element_bytes),
            || rms(&x, &self.norm, self.cfg.rms_eps),
        )
    }
}

/// Additive `[B,1,L,L]` causal mask (0 where a query may attend, large-negative otherwise),
/// optionally also masking padding keys (`pad_mask[b,j]==0`).
fn causal_mask(b: i32, l: i32, pad_mask: Option<&Array>) -> Result<Array> {
    let neg = -1e9f32;
    let mut m = vec![0f32; (l * l) as usize];
    for i in 0..l as usize {
        for j in 0..l as usize {
            if j > i {
                m[i * l as usize + j] = neg;
            }
        }
    }
    let causal = Array::from_slice(&m, &[1, 1, l, l]);
    match pad_mask {
        None => Ok(causal),
        Some(pm) => {
            // pad_mask [B,L] (1 real / 0 pad) -> additive key mask [B,1,1,L]
            let pad = pm.as_dtype(Dtype::Float32)?.reshape(&[b, 1, 1, l])?;
            let one = scalar(1.0);
            let key_add = multiply(&mlx_rs::ops::subtract(&one, &pad)?, scalar(neg))?;
            Ok(add(&causal, &key_add)?)
        }
    }
}

#[cfg(test)]
mod embedding_tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn malformed_packed_embeddings_fail_at_load_not_gather() {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        for bits in [4, 8] {
            let source = Array::from_slice(&[0.25f32; 3 * 128], &[3, 128]);
            let (codes, scales, biases) = mlx_rs::ops::quantize(&source, 64, bits).unwrap();
            let valid = HashMap::from([
                ("e.weight".into(), codes),
                ("e.scales".into(), scales.clone()),
                ("e.biases".into(), biases),
            ]);
            assert!(gemma_embedding(&Weights::from_map(valid.clone()), "e", 128).is_ok());
            for missing in ["e.scales", "e.biases", "e.weight"] {
                let mut bad = valid.clone();
                bad.remove(missing);
                assert!(
                    gemma_embedding(&Weights::from_map(bad), "e", 128).is_err(),
                    "q{bits}: missing {missing}"
                );
            }
            let mut bad = valid.clone();
            bad.insert("e.biases".into(), Array::from_slice(&[0.0f32; 3], &[3, 1]));
            assert!(gemma_embedding(&Weights::from_map(bad), "e", 128).is_err());
            let mut bad = valid.clone();
            bad.insert("e.biases".into(), scales.as_dtype(Dtype::Int32).unwrap());
            assert!(gemma_embedding(&Weights::from_map(bad), "e", 128).is_err());
            assert!(gemma_embedding(&Weights::from_map(valid), "e", 64).is_err());
        }
    }

    // No real weights and no GPU: isolate the embedding + Gemma normalization from attention.
    #[test]
    fn packed_embedding_matches_dequantized_rows_in_native_dtype() {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        for prefix in ["", "model."] {
            for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
                for bits in [4, 8] {
                    let source = Array::from_slice(
                        &(0..7 * 128)
                            .map(|i| (i as f32 * 0.13).sin())
                            .collect::<Vec<_>>(),
                        &[7, 128],
                    )
                    .as_dtype(dtype)
                    .unwrap();
                    let (codes, scales, biases) = mlx_rs::ops::quantize(&source, 64, bits).unwrap();
                    let dense =
                        mlx_rs::ops::dequantize(&codes, &scales, &biases, 64, bits).unwrap();
                    let cfg = Gemma2Config {
                        hidden_size: 128,
                        num_layers: 0,
                        ..Gemma2Config::gemma_2_2b()
                    };
                    let norm = Array::from_slice(&[0.0f32; 128], &[128])
                        .as_dtype(dtype)
                        .unwrap();
                    let mut packed = HashMap::from([
                        (format!("{prefix}embed_tokens.weight"), codes),
                        (format!("{prefix}embed_tokens.scales"), scales),
                        (format!("{prefix}embed_tokens.biases"), biases),
                        (format!("{prefix}norm.weight"), norm.clone()),
                    ]);
                    let model =
                        Gemma2::from_weights(&Weights::from_map(packed.clone()), prefix, &cfg)
                            .unwrap();
                    packed.remove(&format!("{prefix}embed_tokens.scales"));
                    packed.remove(&format!("{prefix}embed_tokens.biases"));
                    packed.insert(format!("{prefix}embed_tokens.weight"), dense);
                    let reference =
                        Gemma2::from_weights(&Weights::from_map(packed), prefix, &cfg).unwrap();
                    // Repeated and out-of-order IDs across batches catch row/gather mistakes.
                    let ids = Array::from_slice(&[6i32, 0, 2, 2, 5, 1], &[2, 3]);
                    let got = model.forward(&ids, None).unwrap();
                    let expected = reference.forward(&ids, None).unwrap();
                    assert_eq!(got.shape(), &[2, 3, 128]);
                    assert_eq!(got.dtype(), dtype);
                    let delta = mlx_rs::ops::subtract(&got, &expected)
                        .unwrap()
                        .abs()
                        .unwrap()
                        .max(None)
                        .unwrap()
                        .item::<f32>();
                    assert!(delta < 1e-6, "q{bits} {dtype:?} {prefix:?}: {delta}");
                }
            }
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
#[path = "../../mlx-gen-iris/tests/common/metal_guard.rs"]
mod quantum_metal_guard;

/// Measured-peak proofs of the counted Gemma quanta (layer, embedding/mask, final norm) on the MLX
/// CPU device, Metal trapped, in a child process. Real modules at small shapes where activations
/// dominate; each at an attention-dominated and an MLP-dominated shape, in f32 and bf16.
#[cfg(all(test, target_os = "macos"))]
mod quantum_proofs {
    use super::*;
    use mlx_gen::ios_quantum::{assert_tracks_measured, measured_peak_bytes};
    use std::collections::HashMap;

    /// Maximum formula/measurement ratio every Gemma proof accepts.
    const MAX_RATIO: f64 = 1.5;

    /// Unthrottled runs per weight-dominated measurement; the proof asserts the worst (largest).
    const RUNS: usize = 5;

    fn rand(shape: &[i32], seed: u64, dtype: Dtype) -> Array {
        mlx_rs::random::normal::<f32>(shape, None, None, Some(&mlx_rs::random::key(seed).unwrap()))
            .unwrap()
            .multiply(scalar(0.05))
            .unwrap()
            .as_dtype(dtype)
            .unwrap()
    }

    fn model(cfg: &Gemma2Config, vocab: i32, dtype: Dtype) -> Gemma2 {
        model_with_projections(cfg, vocab, dtype, dtype)
    }

    /// [`model`] with the seven projection weights stored at `projection` (the embedding and the
    /// norms at `dtype`, which sets the compute dtype).
    fn model_with_projections(
        cfg: &Gemma2Config,
        vocab: i32,
        dtype: Dtype,
        projection: Dtype,
    ) -> Gemma2 {
        let (h, q, kv, i) = (
            cfg.hidden_size,
            cfg.num_heads * cfg.head_dim,
            cfg.num_kv_heads * cfg.head_dim,
            cfg.intermediate_size,
        );
        let mut m = HashMap::new();
        let mut seed = 500;
        let mut put = |m: &mut HashMap<String, Array>, k: String, shape: &[i32]| {
            seed += 1;
            // The seven projections (`*_proj`) at their own storage dtype.
            let dt = if k.ends_with("_proj.weight") {
                projection
            } else {
                dtype
            };
            m.insert(k, rand(shape, seed, dt));
        };
        put(&mut m, "embed_tokens.weight".into(), &[vocab, h]);
        put(&mut m, "norm.weight".into(), &[h]);
        for l in 0..cfg.num_layers {
            let p = format!("layers.{l}");
            for n in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                put(&mut m, format!("{p}.{n}.weight"), &[h]);
            }
            put(&mut m, format!("{p}.self_attn.q_proj.weight"), &[q, h]);
            put(&mut m, format!("{p}.self_attn.k_proj.weight"), &[kv, h]);
            put(&mut m, format!("{p}.self_attn.v_proj.weight"), &[kv, h]);
            put(&mut m, format!("{p}.self_attn.o_proj.weight"), &[h, q]);
            put(&mut m, format!("{p}.mlp.gate_proj.weight"), &[i, h]);
            put(&mut m, format!("{p}.mlp.up_proj.weight"), &[i, h]);
            put(&mut m, format!("{p}.mlp.down_proj.weight"), &[h, i]);
        }
        let arrays: Vec<&Array> = m.values().collect();
        mlx_rs::transforms::eval(arrays).unwrap();
        let model = Gemma2::from_weights(&Weights::from_map(m), "", cfg).unwrap();
        // Materialize the derived `(1 + w)` norm weights so no load work lands in a measurement.
        for layer in &model.layers {
            mlx_rs::transforms::eval([
                &layer.input_ln,
                &layer.post_attn_ln,
                &layer.pre_ff_ln,
                &layer.post_ff_ln,
            ])
            .unwrap();
        }
        model.norm.eval().unwrap();
        model
    }

    fn cfg(hidden: i32, heads: i32, kv: i32, hd: i32, inter: i32) -> Gemma2Config {
        Gemma2Config {
            hidden_size: hidden,
            num_layers: 1,
            num_heads: heads,
            num_kv_heads: kv,
            head_dim: hd,
            intermediate_size: inter,
            query_pre_attn_scalar: hd as f32,
            ..Gemma2Config::gemma_2_2b()
        }
    }

    #[test]
    #[ignore = "expensive stalled fixture capture; run through the selected iOS capture workflow"]
    fn gemma_quanta_track_their_measured_peaks() {
        const CHILD: &str = "GEMMA_QUANTUM_PROOF_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "gemma2::quantum_proofs::gemma_quanta_track_their_measured_peaks",
                    "--include-ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
                .env("IOS_CAPTURE_IN_CHILD", "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert!(quantum_metal_guard::forbid_metal());
        assert!(quantum_metal_guard::metal_forbidden());
        // (label, config, tokens, prove the embedding/final-norm quanta here): scores dominate the
        // first layer, the MLP the second; the third sizes the norm/embedding tensors well above
        // the allocator's small-buffer granularity. At the MLP-dominated shape the norm/embedding
        // tensors are 16-32 KiB, where the CPU executor's asynchronous release order alone moves
        // the measured peak by ~40% run to run, so those two quanta are proven at the other shapes.
        let cases = [
            ("attention-dominated", cfg(128, 4, 2, 32, 256), 256, true),
            ("mlp-dominated", cfg(128, 4, 2, 32, 4096), 64, false),
            ("norm-scale", cfg(512, 4, 2, 128, 512), 512, true),
        ];
        for dtype in [Dtype::Float32, Dtype::Bfloat16] {
            for (name, cfg, tokens, small_quanta) in &cases {
                let model = model(cfg, 97, dtype);
                let ids = Array::from_slice(
                    &(0..*tokens).map(|t| t % 97).collect::<Vec<i32>>(),
                    &[1, *tokens],
                );
                let pad = Array::from_slice(
                    &(0..*tokens)
                        .map(|t| if t < tokens - 3 { 1.0 } else { 0.0 })
                        .collect::<Vec<f32>>(),
                    &[1, *tokens],
                );
                ids.eval().unwrap();
                pad.eval().unwrap();
                let eb = dtype_bytes(dtype);
                let t = *tokens as u64;
                let measured = measured_peak_bytes(|| {
                    let (x, cos, sin, mask) = model.embed_inputs(&ids, Some(&pad), dtype)?;
                    Ok(vec![x, cos, sin, mask])
                })
                .unwrap();
                let formula = gemma_embedding_bytes(cfg, 1, t, eb);
                println!(
                    "[gemma embedding {name} {dtype:?}] formula {formula} measured {measured} ratio {:.3}",
                    formula as f64 / measured as f64
                );
                if *small_quanta {
                    assert_tracks_measured("gemma embedding", formula, measured, MAX_RATIO);
                }

                let (x, cos, sin, mask) = model.embed_inputs(&ids, Some(&pad), dtype).unwrap();
                mlx_rs::transforms::eval([&x, &cos, &sin, &mask]).unwrap();
                let layer = &model.layers[0];
                let measured =
                    measured_peak_bytes(|| Ok(vec![layer.forward(&x, &cos, &sin, &mask)?]))
                        .unwrap();
                let formula = gemma_layer_bytes(cfg, 1, t, eb, layer.promoted_weight_bytes(dtype));
                println!(
                    "[gemma layer {name} {dtype:?}] formula {formula} measured {measured} ratio {:.3}",
                    formula as f64 / measured as f64
                );
                assert_tracks_measured("gemma layer", formula, measured, MAX_RATIO);

                let measured =
                    measured_peak_bytes(|| Ok(vec![rms(&x, &model.norm, cfg.rms_eps)?])).unwrap();
                let formula = gemma_final_norm_bytes(cfg, 1, t, eb);
                println!(
                    "[gemma final norm {name} {dtype:?}] formula {formula} measured {measured} ratio {:.3}",
                    formula as f64 / measured as f64
                );
                if *small_quanta {
                    assert_tracks_measured("gemma final norm", formula, measured, MAX_RATIO);
                }
            }
        }

        // WEIGHT-dominated: bf16 projections against an f32 stream (f32 embedding), hidden 1024
        // over 8 tokens, so the seven promoted projection copies (~60 MB) dwarf every activation
        // (~32 KB each): the proof sees whether they are live together (summed) or one at a time.
        let cfg = cfg(1024, 8, 4, 128, 4096);
        let model = model_with_projections(&cfg, 97, Dtype::Float32, Dtype::Bfloat16);
        let tokens = 8;
        let ids = Array::from_slice(&(0..tokens).collect::<Vec<i32>>(), &[1, tokens]);
        let (x, cos, sin, mask) = model.embed_inputs(&ids, None, Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&x, &cos, &sin, &mask]).unwrap();
        let layer = &model.layers[0];
        let measured = (0..RUNS)
            .map(|_| {
                measured_peak_bytes(|| Ok(vec![layer.forward(&x, &cos, &sin, &mask)?])).unwrap()
            })
            .max()
            .unwrap();
        let copies = layer.promoted_weight_bytes(Dtype::Float32);
        let formula = gemma_layer_bytes(&cfg, 1, tokens as u64, 4, copies);
        println!(
            "[gemma layer weight copies] formula {formula} measured {measured} ratio {:.3}",
            formula as f64 / measured as f64
        );
        assert_tracks_measured("gemma layer weight copies", formula, measured, MAX_RATIO);
        // Discrimination: pricing only the largest copy (the pre-fix model) under-prices it.
        let largest = 4 * (cfg.intermediate_size as u64) * (cfg.hidden_size as u64);
        let single = gemma_layer_bytes(&cfg, 1, tokens as u64, 4, largest);
        println!("[gemma layer weight copies] largest-copy-only formula {single}");
        assert!(
            single < measured,
            "the largest-copy-only pricing ({single} B) must under-price the measured peak \
             ({measured} B), or this shape cannot tell the models apart"
        );
        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED gemma2::quantum_proofs::gemma_quanta_track_their_measured_peaks");
        }
    }
}
