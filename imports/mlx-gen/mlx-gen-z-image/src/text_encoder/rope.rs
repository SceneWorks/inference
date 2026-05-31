//! Rotary position embedding for the text encoder — the HF "half-split" convention (distinct
//! from the DiT's interleaved RoPE). Port of the fork's `RotaryEmbedding`:
//! `inv_freq = 1/base^(arange(0,dim,2)/dim)`; `freqs = outer(arange(seq), inv_freq)`;
//! `emb = concat([freqs, freqs])`; `cos/sin = cos/sin(emb)[None]`.

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::Result;

pub struct TextRope {
    inv_freq: Vec<f32>,
    dim: i32,
}

impl TextRope {
    /// `dim` = head_dim, `theta` = rope base (1e6 for Z-Image).
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
        let emb = concatenate_axis(&[&freqs, &freqs], 1)?;
        let cos = mlx_rs::ops::cos(&emb)?.reshape(&[1, seq_len, self.dim])?;
        let sin = mlx_rs::ops::sin(&emb)?.reshape(&[1, seq_len, self.dim])?;
        Ok((cos, sin))
    }
}
