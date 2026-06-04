//! S3 — the LTX-2.3 **DiT** (video stack): 48 × `BasicAVTransformerBlock` (video-only path) plus the
//! preprocessor (patchify + adaLN-single) and output projection. Port of the `mlx_video` reference
//! `models/ltx/{transformer,attention,adaln,feed_forward,ltx}.py`.
//!
//! **S3a (this slice)** ports the per-block math + a single-block parity gate, the riskiest piece:
//!  - **gated attention** (`to_gate_logits → 2·sigmoid`, zero-init identity), q/k **RMSNorm** over the
//!    full inner_dim (pre-head, learned weight), **SPLIT 3-D RoPE** on q/k (reusing the S0
//!    [`crate::rope`]), SDPA, `to_out`;
//!  - **adaLN-single** with the 9-row `scale_shift_table` (gated 2.3 family): MSA rows 0..3, FF rows
//!    3..6, text-cross-attn rows 6..9 (`v_has_ca_ada`), each = `table[row] + timestep_proj[row]`;
//!  - **prompt adaLN** (`prompt_scale_shift_table`, 2 rows) modulating the text context before
//!    cross-attention;
//!  - **FeedForward** = `proj_in → gelu(tanh) → proj_out` (core dtype-preserving [`gelu_tanh`]).
//!
//! The shipped `base_q8` transformer stores the attn/ff Linears **Q8-quantized** (U32 + `scales` +
//! `biases`, group 64). The full 48-layer forward runs bf16 × Q8 quantized-matmul (S3b); this slice
//! gates the block **math in f32** by dequantizing those weights (`mx.dequantize`, bit-identical in
//! Rust and the reference) — isolating block correctness from the quant path, mirroring the S2 VAE
//! f32 gate. The small Linears (q/k-norm, gate, adaLN) are dense bf16 → f32.

use mlx_rs::fast::{rms_norm as fast_rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, dequantize, multiply, sigmoid};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{gelu_tanh, linear};
use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::Result;

use crate::config::LtxConfig;
use crate::rope::apply_split_rotary_emb;

/// Q8 quant config of the shipped transformer (`split_model.json`: bits 8, group 64).
const QUANT_BITS: i32 = 8;
const QUANT_GROUP: i32 = 64;

fn f32(w: &Weights, key: &str) -> Result<Array> {
    to_dtype(w.require(key)?, Dtype::Float32)
}

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// `x · (1 + scale) + shift` (adaLN modulation), broadcasting `scale`/`shift` `(B, S', dim)` over the
/// token axis.
fn modulate(x: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    Ok(add(&multiply(x, &add(scale, scalar(1.0))?)?, shift)?)
}

/// A dense f32 Linear. Loaded from either a dense bf16 weight or a **dequantized** Q8 weight
/// (`{prefix}.weight` U32 + `.scales` + `.biases`) — both upcast to f32 for the S3a math gate.
struct Linear {
    w: Array, // [out, in] f32
    b: Array, // [out] f32
}

impl Linear {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        let weight = match w.get(&format!("{prefix}.scales")) {
            Some(scales) => {
                let q = w.require(&format!("{prefix}.weight"))?;
                let biases = w.require(&format!("{prefix}.biases"))?;
                let dense =
                    dequantize(q, scales, Some(biases), Some(QUANT_GROUP), Some(QUANT_BITS))?;
                to_dtype(&dense, Dtype::Float32)?
            }
            None => f32(w, &format!("{prefix}.weight"))?,
        };
        Ok(Self {
            w: weight,
            b: f32(w, &format!("{prefix}.bias"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        linear(x, &self.w, &self.b)
    }
}

/// `mx.fast.rms_norm(x, ones, eps)` — the block's weightless pre-norm (feature RMS over the last axis).
fn rms_norm_noweight(x: &Array, eps: f32) -> Result<Array> {
    let dim = *x.shape().last().unwrap();
    let ones = Array::ones::<f32>(&[dim])?.as_dtype(x.dtype())?;
    Ok(fast_rms_norm(x, &ones, eps)?)
}

/// Multi-head attention with q/k RMSNorm, optional SPLIT RoPE, optional per-head gating. Self-attn
/// when `context` is `None`; cross-attn otherwise. RoPE `(cos, sin)` applies to q **and** k (self-attn
/// only; cross-attn passes `pe=None`).
struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    q_norm: Array, // [inner] f32
    k_norm: Array,
    to_out: Linear,
    gate: Option<Linear>, // to_gate_logits [heads, inner]
    heads: i32,
    dim_head: i32,
    eps: f32,
}

impl Attention {
    fn load(w: &Weights, prefix: &str, cfg: &LtxConfig) -> Result<Self> {
        let gate = if w.get(&format!("{prefix}.to_gate_logits.weight")).is_some() {
            Some(Linear::load(w, &format!("{prefix}.to_gate_logits"))?)
        } else {
            None
        };
        Ok(Self {
            to_q: Linear::load(w, &format!("{prefix}.to_q"))?,
            to_k: Linear::load(w, &format!("{prefix}.to_k"))?,
            to_v: Linear::load(w, &format!("{prefix}.to_v"))?,
            q_norm: f32(w, &format!("{prefix}.q_norm.weight"))?,
            k_norm: f32(w, &format!("{prefix}.k_norm.weight"))?,
            to_out: Linear::load(w, &format!("{prefix}.to_out"))?,
            gate,
            heads: cfg.num_attention_heads,
            dim_head: cfg.attention_head_dim,
            eps: cfg.norm_eps as f32,
        })
    }

    /// `(B, S, inner)` q-features → `(B, H, S, head_dim)`.
    fn to_heads(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        Ok(x.reshape(&[b, s, self.heads, self.dim_head])?
            .transpose_axes(&[0, 2, 1, 3])?)
    }

    fn forward(
        &self,
        x: &Array,
        context: Option<&Array>,
        mask: Option<&Array>,
        pe: Option<(&Array, &Array)>,
    ) -> Result<Array> {
        let ctx = context.unwrap_or(x);
        let q = fast_rms_norm(&self.to_q.forward(x)?, &self.q_norm, self.eps)?;
        let k = fast_rms_norm(&self.to_k.forward(ctx)?, &self.k_norm, self.eps)?;
        let v = self.to_v.forward(ctx)?;

        let mut qh = self.to_heads(&q)?;
        let mut kh = self.to_heads(&k)?;
        let vh = self.to_heads(&v)?;
        if let Some((cos, sin)) = pe {
            qh = apply_split_rotary_emb(&qh, cos, sin)?;
            kh = apply_split_rotary_emb(&kh, cos, sin)?;
        }

        let scale = (self.dim_head as f32).powf(-0.5);
        let out = match mask {
            Some(m) => scaled_dot_product_attention(&qh, &kh, &vh, scale, m, None)?,
            None => scaled_dot_product_attention(&qh, &kh, &vh, scale, None, None)?,
        };
        // (B, H, S, hd) → (B, S, inner).
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let inner = self.heads * self.dim_head;
        let mut out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, inner])?;

        if let Some(gate) = &self.gate {
            // Per-head gate: 2·sigmoid(logits) (zero-init → identity), broadcast over head_dim.
            let logits = gate.forward(x)?; // (B, S, heads)
            let gates = multiply(&sigmoid(&logits)?, scalar(2.0))?;
            let gates = gates.reshape(&[b, s, self.heads, 1])?;
            out = multiply(&out.reshape(&[b, s, self.heads, self.dim_head])?, &gates)?
                .reshape(&[b, s, inner])?;
        }
        self.to_out.forward(&out)
    }
}

/// `proj_in → gelu(tanh) → proj_out`.
struct FeedForward {
    proj_in: Linear,
    proj_out: Linear,
}

impl FeedForward {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            proj_in: Linear::load(w, &format!("{prefix}.proj_in"))?,
            proj_out: Linear::load(w, &format!("{prefix}.proj_out"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.proj_out
            .forward(&gelu_tanh(&self.proj_in.forward(x)?)?)
    }
}

/// `table[row] + timestep_proj[row]` for `row ∈ [lo, hi)`. `table` is `(num_ada, dim)`; `timestep` is
/// `(B, S', num_ada·dim)`. Returns the `hi−lo` modulation tensors, each `(B, S', dim)`.
fn ada_values(table: &Array, timestep: &Array, lo: i32, hi: i32) -> Result<Vec<Array>> {
    let num_ada = table.shape()[0];
    let dim = table.shape()[1];
    let ts = timestep.shape();
    let (b, s) = (ts[0], ts[1]);
    let ts4 = timestep.reshape(&[b, s, num_ada, dim])?;
    let mut out = Vec::with_capacity((hi - lo) as usize);
    for row in lo..hi {
        let trow = table
            .index_axis(row, 0)? // (dim,)
            .reshape(&[1, 1, dim])?;
        let tsrow = ts4.index_axis(row, 2)?; // (B, S, dim)
        out.push(add(&trow, &tsrow)?);
    }
    Ok(out)
}

/// Index a single position `i` along `axis`, dropping that axis (like `x.take(i, axis)` then squeeze).
trait IndexAxis {
    fn index_axis(&self, i: i32, axis: i32) -> Result<Array>;
}
impl IndexAxis for Array {
    fn index_axis(&self, i: i32, axis: i32) -> Result<Array> {
        let taken = self.take_axis(Array::from_int(i), axis)?;
        Ok(taken)
    }
}

/// One video transformer block (`BasicAVTransformerBlock`, video-only / gated 2.3 path).
pub struct VideoBlock {
    attn1: Attention,
    attn2: Attention,
    ff: FeedForward,
    scale_shift_table: Array,        // (9, inner)
    prompt_scale_shift_table: Array, // (2, inner)
    eps: f32,
}

impl VideoBlock {
    pub fn load(w: &Weights, prefix: &str, cfg: &LtxConfig) -> Result<Self> {
        Ok(Self {
            attn1: Attention::load(w, &format!("{prefix}.attn1"), cfg)?,
            attn2: Attention::load(w, &format!("{prefix}.attn2"), cfg)?,
            ff: FeedForward::load(w, &format!("{prefix}.ff"))?,
            scale_shift_table: f32(w, &format!("{prefix}.scale_shift_table"))?,
            prompt_scale_shift_table: f32(w, &format!("{prefix}.prompt_scale_shift_table"))?,
            eps: cfg.norm_eps as f32,
        })
    }

    /// Forward (gated, `v_has_ca_ada` = 9-row table): MSA(self, RoPE) → text cross-attn (with
    /// prompt-modulated context) → FeedForward, each adaLN-modulated and gated.
    ///
    /// * `x` — `(B, S, inner)` patch features.
    /// * `timesteps` — `(B, S', 9·inner)` adaLN-single projection.
    /// * `prompt_timestep` — optional `(B, S', 2·inner)` prompt-adaLN projection.
    /// * `context` — `(B, ctx, inner)` text embeddings; `mask` its additive attention mask.
    /// * `cos`/`sin` — SPLIT RoPE tables for the self-attention.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Array,
        timesteps: &Array,
        prompt_timestep: Option<&Array>,
        context: &Array,
        mask: Option<&Array>,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array> {
        // --- MSA (self-attention) ---
        let msa = ada_values(&self.scale_shift_table, timesteps, 0, 3)?;
        let (shift_msa, scale_msa, gate_msa) = (&msa[0], &msa[1], &msa[2]);
        let norm = modulate(&rms_norm_noweight(x, self.eps)?, scale_msa, shift_msa)?;
        let attn = self.attn1.forward(&norm, None, None, Some((cos, sin)))?;
        let mut x = add(x, &multiply(&attn, gate_msa)?)?;

        // --- prompt-adaLN on the text context ---
        let v_context = {
            let (p_shift, p_scale) = match prompt_timestep {
                Some(pt) => {
                    let p = ada_values(&self.prompt_scale_shift_table, pt, 0, 2)?;
                    (p[0].clone(), p[1].clone())
                }
                None => (
                    self.prompt_scale_shift_table.index_axis(0, 0)?,
                    self.prompt_scale_shift_table.index_axis(1, 0)?,
                ),
            };
            modulate(context, &p_scale, &p_shift)?
        };

        // --- text cross-attention (9-param adaLN rows 6..9) ---
        let ca = ada_values(&self.scale_shift_table, timesteps, 6, 9)?;
        let (shift_ca, scale_ca, gate_ca) = (&ca[0], &ca[1], &ca[2]);
        let norm_ca = modulate(&rms_norm_noweight(&x, self.eps)?, scale_ca, shift_ca)?;
        let cross = self.attn2.forward(&norm_ca, Some(&v_context), mask, None)?;
        x = add(&x, &multiply(&cross, gate_ca)?)?;

        // --- FeedForward (adaLN rows 3..6) ---
        let mlp = ada_values(&self.scale_shift_table, timesteps, 3, 6)?;
        let (shift_mlp, scale_mlp, gate_mlp) = (&mlp[0], &mlp[1], &mlp[2]);
        let norm_mlp = modulate(&rms_norm_noweight(&x, self.eps)?, scale_mlp, shift_mlp)?;
        let ff = self.ff.forward(&norm_mlp)?;
        x = add(&x, &multiply(&ff, gate_mlp)?)?;

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulate_closed_form() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let scale = Array::from_slice(&[0.0f32, 1.0, 0.0, 1.0], &[1, 1, 4]);
        let shift = Array::from_slice(&[1.0f32, 0.0, -1.0, 0.0], &[1, 1, 4]);
        let got = modulate(&x, &scale, &shift).unwrap();
        // x*(1+scale)+shift = [1*1+1, 2*2+0, 3*1-1, 4*2+0] = [2, 4, 2, 8].
        assert_eq!(got.as_slice::<f32>(), &[2.0, 4.0, 2.0, 8.0]);
    }

    #[test]
    fn ada_values_splits_rows() {
        // table (9, 2); timestep (1, 1, 18) of zeros → ada == table rows.
        let table = Array::from_slice(&(0..18).map(|v| v as f32).collect::<Vec<_>>(), &[9, 2]);
        let ts = Array::zeros::<f32>(&[1, 1, 18]).unwrap();
        let vals = ada_values(&table, &ts, 0, 3).unwrap();
        assert_eq!(vals.len(), 3);
        // row 0 = [0,1], row1 = [2,3], row2 = [4,5].
        assert_eq!(vals[0].as_slice::<f32>(), &[0.0, 1.0]);
        assert_eq!(vals[2].as_slice::<f32>(), &[4.0, 5.0]);
    }
}
