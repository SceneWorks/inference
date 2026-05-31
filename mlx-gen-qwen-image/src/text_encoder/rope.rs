//! HF "half-split" rotary embedding (θ=1e6). `inv_freq = 1/θ^(arange(0,dim,2)/dim)`;
//! `freqs = outer(arange(seq), inv_freq)`; `emb = concat([freqs, freqs])`; `cos/sin = cos/sin(emb)`.
//! Identical to the Z-Image text RoPE; the fork's multimodal RoPE collapses to this for text.

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::Result;

pub struct TextRope {
    inv_freq: Vec<f32>,
    dim: i32,
}

impl TextRope {
    /// `dim` = head_dim, `theta` = rope base (1e6 for Qwen-Image).
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
        let mut freqs = Vec::with_capacity(seq_len as usize * half);
        for s in 0..seq_len {
            for &f in &self.inv_freq {
                freqs.push(s as f32 * f);
            }
        }
        let freqs = Array::from_slice(&freqs, &[seq_len, half as i32]);
        let emb = concatenate_axis(&[&freqs, &freqs], 1)?;
        let cos = mlx_rs::ops::cos(&emb)?.reshape(&[1, seq_len, self.dim])?;
        let sin = mlx_rs::ops::sin(&emb)?.reshape(&[1, seq_len, self.dim])?;
        Ok((cos, sin))
    }
}
