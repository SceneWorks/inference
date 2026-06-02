//! Model-agnostic neural-net primitives — the shared `nn` layer of `mlx-gen` core.
//!
//! These are the genuinely family-independent leaf ops: dense linear, SiLU, NHWC `conv2d`,
//! pytorch-compatible `group_norm`, and nearest `upsample`. Model-specific block assemblies
//! (attention / RoPE / SwiGLU layouts) intentionally stay in their family crates — see
//! `docs/MODEL_ARCHITECTURE.md` §3.2 ("each family crate owns its blocks"). A primitive
//! graduates here only once it is provably model-agnostic; we do not lift a block to a shared
//! abstraction off a single model.

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{
    add, broadcast_to, conv2d as conv2d_op, conv3d as conv3d_op, matmul, multiply, sigmoid,
};
use mlx_rs::Array;

use crate::Result;

/// `y = x · Wᵀ + b` for a stored `[out, in]` weight + bias (PyTorch `nn.Linear` convention).
pub fn linear(x: &Array, w: &Array, b: &Array) -> Result<Array> {
    Ok(add(&matmul(x, w.t())?, b)?)
}

/// SiLU / swish activation: `x · sigmoid(x)`.
pub fn silu(x: &Array) -> Result<Array> {
    Ok(multiply(x, &sigmoid(x)?)?)
}

/// 2-D conv over NHWC `x` with an mlx `[out, kH, kW, in]` weight (+ optional bias).
pub fn conv2d(x: &Array, w: &Array, b: Option<&Array>, stride: i32, padding: i32) -> Result<Array> {
    let mut y = conv2d_op(x, w, (stride, stride), (padding, padding), (1, 1), 1)?;
    if let Some(b) = b {
        y = add(&y, b)?;
    }
    Ok(y)
}

/// 3-D conv over NDHWC `x` with an mlx `[out, kD, kH, kW, in]` weight (+ optional bias).
/// `stride`/`padding` are per-axis `(depth, height, width)`. Qwen's causal-Conv3d VAE applies
/// its asymmetric temporal padding manually and calls this with `padding (0, 0, 0)`; future
/// video families (Wan2.2 / LTX) reuse it directly — hence it lives in shared core `nn`.
pub fn conv3d(
    x: &Array,
    w: &Array,
    b: Option<&Array>,
    stride: (i32, i32, i32),
    padding: (i32, i32, i32),
) -> Result<Array> {
    let mut y = conv3d_op(x, w, stride, padding, (1, 1, 1), 1)?;
    if let Some(b) = b {
        y = add(&y, b)?;
    }
    Ok(y)
}

/// PyTorch-compatible group normalization over NHWC `x` (`weight`/`bias` are per-channel).
/// Mirrors mlx-rs `GroupNorm::pytorch_group_norm` + affine: split channels into `num_groups`,
/// layer-norm each group, then scale/shift by `weight`/`bias`.
pub fn group_norm(
    x: &Array,
    weight: &Array,
    bias: &Array,
    num_groups: i32,
    eps: f32,
) -> Result<Array> {
    let sh = x.shape();
    let batch = sh[0];
    let dims = sh[sh.len() - 1];
    let rest = &sh[1..sh.len() - 1];
    let group_size = dims / num_groups;

    let g = x
        .reshape(&[batch, -1, num_groups, group_size])?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[batch, num_groups, -1])?;
    let g = layer_norm(&g, None, None, eps)?;
    let g = g
        .reshape(&[batch, num_groups, -1, group_size])?
        .transpose_axes(&[0, 2, 1, 3])?;

    let mut shape = vec![batch];
    shape.extend_from_slice(rest);
    shape.push(dims);
    let normed = g.reshape(&shape)?;
    Ok(add(&multiply(&normed, weight)?, bias)?)
}

/// Nearest-neighbor upsample of NHWC `x` by `scale` (broadcast + reshape).
pub fn upsample_nearest(x: &Array, scale: i32) -> Result<Array> {
    let sh = x.shape();
    let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let x6 = x.reshape(&[b, h, 1, w, 1, c])?;
    let bc = broadcast_to(&x6, &[b, h, scale, w, scale, c])?;
    Ok(bc.reshape(&[b, h * scale, w * scale, c])?)
}

/// Rotary position embedding for **text encoders** — the HF "half-split" convention (distinct from
/// the DiT's interleaved RoPE, which stays family-owned). Port of the fork's `RotaryEmbedding`:
/// `inv_freq = 1/θ^(arange(0,dim,2)/dim)`; `freqs = outer(arange(seq), inv_freq)`;
/// `emb = concat([freqs, freqs])`; `cos/sin = cos/sin(emb)[None]`. Shared by the Z-Image and
/// Qwen-Image text encoders, which use the identical layout (the second-family trigger for lifting
/// this to core — F-006).
pub struct TextRope {
    inv_freq: Vec<f32>,
    dim: i32,
}

impl TextRope {
    /// `dim` = head_dim, `theta` = rope base (1e6 for both Z-Image and Qwen-Image).
    pub fn new(dim: i32, theta: f32) -> Self {
        let half = (dim / 2) as usize;
        let inv_freq = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
            .collect();
        Self { inv_freq, dim }
    }

    /// Returns `(cos, sin)`, each `[1, seq_len, dim]`, for positions `0..seq_len`.
    pub fn forward(&self, seq_len: i32) -> Result<(Array, Array)> {
        let half = self.inv_freq.len();
        // freqs[s, j] = s * inv_freq[j]  → [seq, half]
        let mut freqs = Vec::with_capacity(seq_len as usize * half);
        for s in 0..seq_len {
            for &f in &self.inv_freq {
                freqs.push(s as f32 * f);
            }
        }
        let freqs = Array::from_slice(&freqs, &[seq_len, half as i32]);
        // emb = concat([freqs, freqs], -1) → [seq, dim]
        let emb = mlx_rs::ops::concatenate_axis(&[&freqs, &freqs], 1)?;
        let cos = mlx_rs::ops::cos(&emb)?.reshape(&[1, seq_len, self.dim])?;
        let sin = mlx_rs::ops::sin(&emb)?.reshape(&[1, seq_len, self.dim])?;
        Ok((cos, sin))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv3d_1x1x1_sums_input_channels_with_bias() {
        // NDHWC: a single voxel with 2 input channels [1, 2].
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 1, 1, 2]);
        // weight [out=1, kD=1, kH=1, kW=1, in=2] = ones -> sums over the input channels.
        let w = Array::from_slice(&[1.0f32, 1.0], &[1, 1, 1, 1, 2]);
        let bias = Array::from_slice(&[10.0f32], &[1]);
        let y = conv3d(&x, &w, Some(&bias), (1, 1, 1), (0, 0, 0)).unwrap();
        assert_eq!(y.shape(), &[1, 1, 1, 1, 1]);
        assert_eq!(y.item::<f32>(), 13.0); // 1 + 2 + bias 10
    }
}
