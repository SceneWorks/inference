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

    fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let h = self
            .attn
            .forward(&rms(x, &self.input_ln, self.eps)?, cos, sin, mask)?;
        let x = add(x, &rms(&h, &self.post_attn_ln, self.eps)?)?;
        let h = self.mlp.forward(&rms(&x, &self.pre_ff_ln, self.eps)?)?;
        Ok(add(&x, &rms(&h, &self.post_ff_ln, self.eps)?)?)
    }
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

    /// `ids`: `[B, L]` (i32). `pad_mask`: optional `[B, L]` (1 = real, 0 = pad). Returns the
    /// last-hidden states `[B, L, hidden]` in the embedding dtype.
    pub fn forward(&self, ids: &Array, pad_mask: Option<&Array>) -> Result<Array> {
        let sh = ids.shape();
        let (b, l) = (sh[0], sh[1]);
        let hidden = self.cfg.hidden_size;
        let _quantum_scope = mlx_gen::ios_quantum::Scope::new();
        let scratch = mlx_gen::ios_quantum::gemma_layer_bytes(
            b as u64,
            l as u64,
            hidden as u64,
            self.cfg.intermediate_size as u64,
            self.cfg.num_heads as u64,
            self.cfg.head_dim as u64,
        );
        mlx_gen::ios_quantum::before("Gemma embedding/mask", scratch)?;

        // embed + √hidden scale (cast to embedding dtype, per the reference)
        let flat = ids.reshape(&[b * l])?;
        // TokenEmbedding gathers packed rows BEFORE dequantizing, never the full vocabulary.
        // Its common forward returns f32; Gemma must retain the stored embedding dtype.
        let dtype = match &self.embed {
            TokenEmbedding::Dense(w) => w.dtype(),
            TokenEmbedding::Quantized { scales, .. } => scales.dtype(),
        };
        let emb = self
            .embed
            .forward(&flat)?
            .as_dtype(dtype)?
            .reshape(&[b, l, hidden])?;
        let normalizer = scalar((hidden as f32).sqrt()).as_dtype(emb.dtype())?;
        let mut x = multiply(&emb, &normalizer)?;

        let (cos, sin) = rope_tables(self.cfg.head_dim, l, self.cfg.rope_theta);
        let mask = causal_mask(b, l, pad_mask)?;
        mlx_gen::ios_quantum::finish("Gemma embedding/mask", &[&x, &cos, &sin, &mask])?;
        for layer in &self.layers {
            x = mlx_gen::ios_quantum::run("Gemma layer", scratch, || {
                layer.forward(&x, &cos, &sin, &mask)
            })?;
        }
        mlx_gen::ios_quantum::run("Gemma final norm", scratch, || {
            rms(&x, &self.norm, self.cfg.rms_eps)
        })
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
