//! Joint (dual-stream) attention. Port of the fork's `QwenAttention`: separate q/k/v projections
//! for the image (`to_*`) and text (`add_*_proj`) streams, per-head q/k RMSNorm, **interleaved**
//! complex RoPE, then attention over the concatenated `[txt, img]` sequence, split back into the
//! two streams and projected (`attn_to_out.0` / `to_add_out`).

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, multiply, split, subtract};
use mlx_rs::Array;

use mlx_gen::nn::linear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::join;

const RMS_EPS: f32 = 1e-6;

pub struct QwenJointAttention {
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    add_q_w: Array,
    add_q_b: Array,
    add_k_w: Array,
    add_k_b: Array,
    add_v_w: Array,
    add_v_b: Array,
    norm_q: Array,
    norm_k: Array,
    norm_added_q: Array,
    norm_added_k: Array,
    out_w: Array,
    out_b: Array,
    add_out_w: Array,
    add_out_b: Array,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl QwenJointAttention {
    pub fn from_weights(w: &Weights, prefix: &str, num_heads: i32, head_dim: i32) -> Result<Self> {
        let g = |s: &str| w.require(&join(prefix, s)).cloned();
        Ok(Self {
            q_w: g("to_q.weight")?,
            q_b: g("to_q.bias")?,
            k_w: g("to_k.weight")?,
            k_b: g("to_k.bias")?,
            v_w: g("to_v.weight")?,
            v_b: g("to_v.bias")?,
            add_q_w: g("add_q_proj.weight")?,
            add_q_b: g("add_q_proj.bias")?,
            add_k_w: g("add_k_proj.weight")?,
            add_k_b: g("add_k_proj.bias")?,
            add_v_w: g("add_v_proj.weight")?,
            add_v_b: g("add_v_proj.bias")?,
            norm_q: g("norm_q.weight")?,
            norm_k: g("norm_k.weight")?,
            norm_added_q: g("norm_added_q.weight")?,
            norm_added_k: g("norm_added_k.weight")?,
            out_w: g("attn_to_out.0.weight")?,
            out_b: g("attn_to_out.0.bias")?,
            add_out_w: g("to_add_out.weight")?,
            add_out_b: g("to_add_out.bias")?,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `img`/`txt`: `[B, seq, dim]`; rope tables `[seq, head_dim/2]`; `mask`: optional additive
    /// `[B, 1, 1, txt+img]`. Returns `(img_attn, txt_attn)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Array,
        txt: &Array,
        img_cos: &Array,
        img_sin: &Array,
        txt_cos: &Array,
        txt_sin: &Array,
        mask: Option<&Array>,
    ) -> Result<(Array, Array)> {
        let (b, img_seq) = (img.shape()[0], img.shape()[1]);
        let txt_seq = txt.shape()[1];
        let (h, hd) = (self.num_heads, self.head_dim);
        let to_heads = |x: &Array, w: &Array, bias: &Array, seq: i32| -> Result<Array> {
            Ok(linear(x, w, bias)?.reshape(&[b, seq, h, hd])?)
        };

        let img_q = rms_norm(
            &to_heads(img, &self.q_w, &self.q_b, img_seq)?,
            &self.norm_q,
            RMS_EPS,
        )?;
        let img_k = rms_norm(
            &to_heads(img, &self.k_w, &self.k_b, img_seq)?,
            &self.norm_k,
            RMS_EPS,
        )?;
        let img_v = to_heads(img, &self.v_w, &self.v_b, img_seq)?;
        let txt_q = rms_norm(
            &to_heads(txt, &self.add_q_w, &self.add_q_b, txt_seq)?,
            &self.norm_added_q,
            RMS_EPS,
        )?;
        let txt_k = rms_norm(
            &to_heads(txt, &self.add_k_w, &self.add_k_b, txt_seq)?,
            &self.norm_added_k,
            RMS_EPS,
        )?;
        let txt_v = to_heads(txt, &self.add_v_w, &self.add_v_b, txt_seq)?;

        let img_q = apply_rope_qwen(&img_q, img_cos, img_sin)?;
        let img_k = apply_rope_qwen(&img_k, img_cos, img_sin)?;
        let txt_q = apply_rope_qwen(&txt_q, txt_cos, txt_sin)?;
        let txt_k = apply_rope_qwen(&txt_k, txt_cos, txt_sin)?;

        // joint [txt, img] over the sequence axis, then to [B, heads, seq, head_dim] for SDPA.
        let q = concatenate_axis(&[&txt_q, &img_q], 1)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = concatenate_axis(&[&txt_k, &img_k], 1)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = concatenate_axis(&[&txt_v, &img_v], 1)?.transpose_axes(&[0, 2, 1, 3])?;

        let o = match mask {
            Some(m) => scaled_dot_product_attention(&q, &k, &v, self.scale, m, None)?,
            None => scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?,
        };
        let joint = txt_seq + img_seq;
        let o = o
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, joint, h * hd])?;

        // split back along the sequence axis: text first, then image.
        let txt_idx = Array::from_slice(&(0..txt_seq).collect::<Vec<i32>>(), &[txt_seq]);
        let img_idx = Array::from_slice(&(txt_seq..joint).collect::<Vec<i32>>(), &[img_seq]);
        let txt_attn = linear(&o.take_axis(&txt_idx, 1)?, &self.add_out_w, &self.add_out_b)?;
        let img_attn = linear(&o.take_axis(&img_idx, 1)?, &self.out_w, &self.out_b)?;
        Ok((img_attn, txt_attn))
    }
}

/// Interleaved complex RoPE: pairs `(x_2i, x_2i+1)` rotated by `(cos_i, sin_i)`. `x`:
/// `[B, seq, heads, head_dim]`; `cos`/`sin`: `[seq, head_dim/2]`.
fn apply_rope_qwen(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let sh = x.shape();
    let (b, seq, heads, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let half = hd / 2;
    let x5 = x.reshape(&[b, seq, heads, half, 2])?;
    let parts = split(&x5, 2, 4)?; // even/odd lanes, each [B,seq,heads,half,1]
    let xr = parts[0].reshape(&[b, seq, heads, half])?;
    let xi = parts[1].reshape(&[b, seq, heads, half])?;
    let cos = cos.reshape(&[1, seq, 1, half])?;
    let sin = sin.reshape(&[1, seq, 1, half])?;
    let out_r = subtract(&multiply(&xr, &cos)?, &multiply(&xi, &sin)?)?;
    let out_i = add(&multiply(&xr, &sin)?, &multiply(&xi, &cos)?)?;
    let stacked = concatenate_axis(&[&out_r.expand_dims(4)?, &out_i.expand_dims(4)?], 4)?;
    Ok(stacked.reshape(&[b, seq, heads, hd])?)
}
