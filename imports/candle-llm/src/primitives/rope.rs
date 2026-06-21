//! Rotary position embeddings.
//!
//! The Candle port of `mlx-llm`'s RoPE (itself byte-for-byte the mlx-gen `Llama3Rope`). Inverse
//! frequencies and the cos/sin tables are built on the host and lifted to the device; rotation uses
//! the GPT-NeoX / HF **rotate-half** (split-the-head-in-half) convention, matching
//! `candle-gen-sensenova`'s `apply_rope`. Hand-rolled (not `candle_nn::rotary_emb`) so the Llama-3
//! NTK-by-parts frequency schedule stays explicit and bit-comparable to the reference engines.
//!
//! The family covered: **standard** RoPE (also Qwen3 — same rotation, config theta) and **Llama-3
//! scaled** RoPE (the NTK-by-parts wavelength smoothing).

use candle_core::{DType, Device, Tensor};

use crate::error::Result;

/// A rotary embedding: the host-side inverse-frequency table plus the head dimension it rotates.
#[derive(Clone, Debug)]
pub struct Rope {
    /// `inv_freq[i]`, length `head_dim / 2`.
    inv_freq: Vec<f32>,
    /// Head dimension (the size of the last axis RoPE rotates).
    dim: usize,
}

impl Rope {
    /// Standard RoPE: `inv_freq[i] = theta^(-2i / head_dim)`.
    ///
    /// This also covers **Qwen3** (Qwen3 uses standard RoPE — its distinctive per-head q/k norm
    /// lives in attention, not here) by passing the model's `rope_theta` (e.g. `1_000_000`).
    pub fn standard(head_dim: i32, theta: f32) -> Self {
        let dim = head_dim as usize;
        let half = dim / 2;
        let inv_freq = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
            .collect();
        Self { inv_freq, dim }
    }

    /// Llama-3 scaled RoPE (the `rope_scaling` "llama3" NTK-by-parts schedule).
    ///
    /// Low-frequency components (long wavelength) are divided by `factor`; high-frequency components
    /// pass through unchanged; the band between is smoothly interpolated. With `factor == 1.0` this
    /// collapses to [`Rope::standard`].
    pub fn llama3(
        head_dim: i32,
        theta: f32,
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_context: f32,
    ) -> Self {
        let dim = head_dim as usize;
        let half = dim / 2;
        let low_freq_wavelen = original_context / low_freq_factor;
        let high_freq_wavelen = original_context / high_freq_factor;
        let inv_freq = (0..half)
            .map(|i| {
                let inv = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
                let wavelen = 2.0 * std::f32::consts::PI / inv;
                if wavelen > low_freq_wavelen {
                    inv / factor
                } else if wavelen < high_freq_wavelen {
                    inv
                } else {
                    let smooth = (original_context / wavelen - low_freq_factor)
                        / (high_freq_factor - low_freq_factor);
                    (1.0 - smooth) * inv / factor + smooth * inv
                }
            })
            .collect();
        Self { inv_freq, dim }
    }

    /// The head dimension this RoPE rotates.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Inverse frequencies (length `head_dim / 2`).
    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }

    /// Build `(cos, sin)` tables for `seq_len` contiguous positions starting at `offset`. Each is
    /// `[1, seq_len, head_dim]` in `dtype` (pass `DType::BF16` to match the bf16 decoders, or
    /// `DType::F32` for the CPU path).
    pub fn cos_sin(
        &self,
        seq_len: i32,
        offset: i32,
        dtype: DType,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let positions: Vec<i32> = (0..seq_len).map(|s| offset + s).collect();
        self.cos_sin_at(&positions, dtype, device)
    }

    /// Build `(cos, sin)` tables for an explicit list of positions — the building block for packed /
    /// paged batches and for multi-axis RoPE.
    pub fn cos_sin_at(
        &self,
        positions: &[i32],
        dtype: DType,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let n = positions.len();
        // emb = cat(freqs, freqs): the half-table duplicated to the full head dim.
        let mut emb = Vec::with_capacity(n * self.dim);
        for &pos in positions {
            for &f in &self.inv_freq {
                emb.push(pos as f32 * f);
            }
            for &f in &self.inv_freq {
                emb.push(pos as f32 * f);
            }
        }
        let emb = Tensor::from_vec(emb, (1, n, self.dim), device)?;
        let cos_t = emb.cos()?.to_dtype(dtype)?;
        let sin_t = emb.sin()?.to_dtype(dtype)?;
        Ok((cos_t, sin_t))
    }
}

/// Apply rotary embeddings to `x` (rotate-half / GPT-NeoX convention).
///
/// `x` is `[batch, seq, heads, head_dim]` (RoPE is applied before the transpose into
/// `[batch, heads, seq, head_dim]`); `cos`/`sin` are `[1, seq, head_dim]` and broadcast over heads.
pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let d = x.dim(3)?;
    let half = d / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let rot = Tensor::cat(&[&x2.neg()?, &x1], 3)?; // rotate_half = cat(-x2, x1)
    let cos = cos.unsqueeze(2)?; // [1, seq, 1, head_dim]
    let sin = sin.unsqueeze(2)?;
    Ok((x.broadcast_mul(&cos)? + rot.broadcast_mul(&sin)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_inv_freq_matches_formula() {
        let rope = Rope::standard(8, 10000.0);
        assert_eq!(rope.inv_freq().len(), 4);
        assert!((rope.inv_freq()[0] - 1.0).abs() < 1e-6);
        let expected1 = 1.0f32 / 10000.0f32.powf(2.0 / 8.0);
        assert!((rope.inv_freq()[1] - expected1).abs() < 1e-6);
    }

    #[test]
    fn llama3_with_unit_factor_equals_standard() {
        let std_rope = Rope::standard(128, 500000.0);
        let l3 = Rope::llama3(128, 500000.0, 1.0, 1.0, 4.0, 8192.0);
        for (a, b) in std_rope.inv_freq().iter().zip(l3.inv_freq()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn llama3_scales_low_frequencies_down() {
        let std_rope = Rope::standard(128, 500000.0);
        let l3 = Rope::llama3(128, 500000.0, 8.0, 1.0, 4.0, 8192.0);
        let last = std_rope.inv_freq().len() - 1;
        assert!(l3.inv_freq()[last] < std_rope.inv_freq()[last]);
        assert!((l3.inv_freq()[last] - std_rope.inv_freq()[last] / 8.0).abs() < 1e-9);
    }

    #[test]
    fn cos_sin_shapes() {
        let rope = Rope::standard(16, 10000.0);
        let (c, s) = rope.cos_sin(5, 0, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(c.dims(), &[1, 5, 16]);
        assert_eq!(s.dims(), &[1, 5, 16]);
    }

    #[test]
    fn position_zero_is_identity() {
        // At position 0, cos = 1 and sin = 0, so apply_rope is a no-op.
        let rope = Rope::standard(8, 10000.0);
        let (c, s) = rope.cos_sin(1, 0, DType::F32, &Device::Cpu).unwrap();
        let x = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (1, 1, 1, 8),
            &Device::Cpu,
        )
        .unwrap();
        let y = apply_rope(&x, &c, &s).unwrap();
        let yh = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let xh = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in xh.iter().zip(&yh) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn rotation_preserves_norm() {
        let rope = Rope::standard(4, 10000.0);
        let (c, s) = rope.cos_sin(3, 0, DType::F32, &Device::Cpu).unwrap();
        let x = Tensor::from_vec(
            vec![
                1.0f32, 0.5, -0.5, 2.0, 0.3, 1.0, -1.0, 0.7, 2.0, -0.2, 0.1, 0.9,
            ],
            (1, 3, 1, 4),
            &Device::Cpu,
        )
        .unwrap();
        let y = apply_rope(&x, &c, &s).unwrap();
        let xh = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let yh = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let norm = |v: &[f32]| -> f32 { v.iter().map(|a| a * a).sum::<f32>().sqrt() };
        for pos in 0..3 {
            let xs = &xh[pos * 4..pos * 4 + 4];
            let ys = &yh[pos * 4..pos * 4 + 4];
            assert!((norm(xs) - norm(ys)).abs() < 1e-4, "pos {pos}");
        }
    }
}
