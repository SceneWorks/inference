//! **Causal / non-causal 3-D convolution** for the LTX-2.3 temporal VAE — candle ships no `conv3d`,
//! and because video has `T > 1` a `kt×kh×kw` kernel does not reduce to one conv2d. It is decomposed
//! into `kt` conv2d "taps": the temporal axis is padded (see below), then the output is
//! `Σ_{kd} conv2d(x_pad[:, :, kd : kd+T], W[:, :, kd])`.
//!
//! LTX padding differs from Wan's causal-zero pad: time is padded by **frame replication** (the
//! reference `CausalConv3d`), spatial by symmetric **zeros** `(k-1)/2`. `causal=true` replicates the
//! first frame `kt-1` times at the front (encoder); `causal=false` replicates the first frame
//! `(kt-1)/2` at the front *and* the last frame `(kt-1)/2` at the back (the decoder path, which is
//! all T2V needs). Weight is the checkpoint-native PyTorch layout `[O, I, kt, kh, kw]`.

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::VarBuilder;

use crate::quant::guard_no_scales;

/// Keep every candle CUDA im2col launch well below `u32::MAX`. Larger launches truncate the
/// element count to `u32`, leaving an uninitialized output tail. The encoder's merged `B*T` batch
/// and high-resolution spatial stages are split transparently at this conservative ceiling.
const IM2COL_BUDGET: usize = 128 * 1024 * 1024;

fn checked_product(values: &[usize], what: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |acc, &value| {
        acc.checked_mul(value).ok_or_else(|| {
            candle_gen::candle_core::Error::Msg(format!("ltx conv3d: {what} size overflows usize"))
        })
    })
}

/// `x.conv2d`, split across batch and then output rows so no im2col buffer can enter candle's u32
/// overflow band. The row path explicitly pads first, preserving ordinary symmetric-padding math.
pub(crate) fn chunked_conv2d(x: &Tensor, w: &Tensor, padding: usize) -> Result<Tensor> {
    chunked_conv2d_budgeted(x, w, padding, IM2COL_BUDGET)
}

/// Ordinary same-padded 3-D convolution. Unlike the VAE's `CausalConv3d`, the
/// latent upscaler was trained with zero padding on both temporal edges.
pub(crate) struct ZeroPaddedConv3d {
    weight: Tensor,
    bias: Tensor,
    kt: usize,
    spatial_pad: usize,
}

impl ZeroPaddedConv3d {
    pub(crate) fn load(vb: VarBuilder, prefix: &str) -> Result<Self> {
        let weight = guard_no_scales(&vb, prefix, vb.dtype())?.contiguous()?;
        let bias = vb.get_unchecked(&format!("{prefix}.bias"))?;
        Self::from_parts(weight, bias)
    }

    /// Build from tensors already pulled out of a checkpoint, for callers that had to inspect the
    /// weight's shape before deciding which module to construct (the latent upsampler's resampler
    /// branch) and must not pay a second materialization to do it.
    pub(crate) fn from_parts(weight: Tensor, bias: Tensor) -> Result<Self> {
        let weight = weight.contiguous()?;
        let dims = weight.dims();
        if dims.len() != 5 {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "ltx conv3d: weight must be rank 5 [O,I,kt,kh,kw], got {dims:?}"
            )));
        }
        let (out_c, kt, kh) = (dims[0], dims[2], dims[3]);
        let bias = bias.reshape((1, out_c, 1, 1, 1))?;
        Ok(Self {
            weight,
            bias,
            kt,
            spatial_pad: (kh - 1) / 2,
        })
    }

    /// The checkpoint-native `[O, I, kt, kh, kw]` weight shape, for callers that must cross-check a
    /// declared channel count against the tensor that actually loaded.
    pub(crate) fn weight_dims(&self) -> &[usize] {
        self.weight.dims()
    }

    /// `[B,C,T,H,W]` -> `[B,O,T,H,W]`, using symmetric zero padding in every
    /// dimension (the upscaler checkpoint's native convention).
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let temporal_pad = (self.kt - 1) / 2;
        let xpad = if temporal_pad == 0 {
            x.clone()
        } else {
            x.pad_with_zeros(2, temporal_pad, temporal_pad)?
        };
        let mut acc: Option<Tensor> = None;
        for kd in 0..self.kt {
            let wk = self.weight.narrow(2, kd, 1)?.squeeze(2)?.contiguous()?;
            let frames = xpad.narrow(2, kd, t)?;
            let merged = frames
                .permute((0, 2, 1, 3, 4))?
                .reshape((b * t, c, h, w))?
                .contiguous()?;
            let y = chunked_conv2d(&merged, &wk, self.spatial_pad)?;
            acc = Some(match acc {
                Some(a) => (a + y)?,
                None => y,
            });
        }
        let y = acc.expect("kt >= 1");
        let (_, o, hp, wp) = y.dims4()?;
        y.reshape((b, t, o, hp, wp))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .broadcast_add(&self.bias)
    }
}

fn chunked_conv2d_budgeted(
    x: &Tensor,
    w: &Tensor,
    padding: usize,
    budget: usize,
) -> Result<Tensor> {
    if budget == 0 {
        return Err(candle_gen::candle_core::Error::Msg(
            "ltx conv3d: im2col budget must be positive".into(),
        ));
    }
    let (n, c, h, wd) = x.dims4()?;
    let (_o, _i, kh, kw) = w.dims4()?;
    let h_out = (h + 2 * padding).saturating_sub(kh) + 1;
    let w_out = (wd + 2 * padding).saturating_sub(kw) + 1;
    let cols_per_frame = checked_product(&[h_out, w_out, c, kh, kw], "im2col frame")?.max(1);
    if cols_per_frame > budget {
        return row_chunked_conv2d(x, w, padding, budget);
    }
    let max_batch = (budget / cols_per_frame).clamp(1, n);
    if n <= max_batch {
        return x.conv2d(w, padding, 1, 1, 1);
    }
    let mut parts = Vec::new();
    let mut start = 0;
    while start < n {
        let len = (n - start).min(max_batch);
        parts.push(x.narrow(0, start, len)?.conv2d(w, padding, 1, 1, 1)?);
        start += len;
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 0)
}

fn row_chunked_conv2d(x: &Tensor, w: &Tensor, padding: usize, budget: usize) -> Result<Tensor> {
    let (n, c, _h, _wd) = x.dims4()?;
    let (_o, _i, kh, kw) = w.dims4()?;
    let xp = if padding > 0 {
        x.pad_with_zeros(2, padding, padding)?
            .pad_with_zeros(3, padding, padding)?
    } else {
        x.clone()
    };
    let hp = xp.dim(2)?;
    let wp = xp.dim(3)?;
    let h_out = hp.saturating_sub(kh) + 1;
    let w_out = wp.saturating_sub(kw) + 1;
    let cols_per_row = checked_product(&[n, w_out, c, kh, kw], "im2col row")?.max(1);
    let band_rows = (budget / cols_per_row).max(1);
    let mut parts = Vec::new();
    let mut row = 0;
    while row < h_out {
        let rows = (h_out - row).min(band_rows);
        parts.push(xp.narrow(2, row, rows + kh - 1)?.conv2d(w, 0, 1, 1, 1)?);
        row += rows;
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 2)
}

/// A 3-D conv loaded from a `[O, I, kt, kh, kw]` weight. Temporal stride is always 1 in the LTX VAE;
/// spatial padding is "same" (`(kh-1)/2`); temporal padding is frame-replication (causal toggle).
pub struct CausalConv3d {
    weight: Tensor, // [O, I, kt, kh, kw]
    bias: Tensor,   // [1, O, 1, 1, 1]
    kt: usize,
    spatial_pad: usize,
}

impl CausalConv3d {
    /// Load `{prefix}.weight` + `{prefix}.bias`, inferring channels + kernel dims from the weight
    /// shape `[O, I, kt, kh, kw]` (channels ride on the weights, not the config).
    pub fn load(vb: VarBuilder, prefix: &str) -> Result<Self> {
        // The 3-D conv weight is NEVER MLX-affine-packed (no `.scales`); guard so a tier that ever packs
        // it errors loudly instead of loading u32 codes as garbage (sc-9417). `dtype()` preserves the
        // vb's load dtype (VAE runs f32).
        let w = guard_no_scales(&vb, prefix, vb.dtype())?.contiguous()?;
        let dims = w.dims();
        let (out_c, kt, kh) = (dims[0], dims[2], dims[3]);
        let bias = vb
            .get_unchecked(&format!("{prefix}.bias"))?
            .reshape((1, out_c, 1, 1, 1))?;
        Ok(Self {
            weight: w,
            bias,
            kt,
            spatial_pad: (kh - 1) / 2,
        })
    }

    pub(crate) fn channels(&self) -> Result<(usize, usize)> {
        let (out_channels, in_channels, _, _, _) = self.weight.dims5()?;
        Ok((out_channels, in_channels))
    }

    /// Replicate `frame` (`[B,C,1,H,W]`) `n` times along the temporal axis.
    fn repeat_frame(frame: &Tensor, n: usize) -> Result<Tensor> {
        let parts: Vec<Tensor> = (0..n).map(|_| frame.clone()).collect();
        Tensor::cat(&parts, 2)
    }

    /// `x`: `[B, C, T, H, W]` → `[B, O, T, H, W]` (spatial "same", temporal frame-replicate).
    pub fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let xpad = if self.kt > 1 {
            let first = x.narrow(2, 0, 1)?;
            if causal {
                let front = Self::repeat_frame(&first, self.kt - 1)?;
                Tensor::cat(&[&front, x], 2)?
            } else {
                let ps = (self.kt - 1) / 2;
                if ps > 0 {
                    let last = x.narrow(2, t - 1, 1)?;
                    let front = Self::repeat_frame(&first, ps)?;
                    let back = Self::repeat_frame(&last, ps)?;
                    Tensor::cat(&[&front, x, &back], 2)?
                } else {
                    x.clone()
                }
            }
        } else {
            x.clone()
        };

        let mut acc: Option<Tensor> = None;
        for kd in 0..self.kt {
            // Tap W[:, :, kd] → [O, I, kh, kw].
            let wk = self.weight.narrow(2, kd, 1)?.squeeze(2)?.contiguous()?;
            // The T frames this tap convolves: x_pad[:, :, kd : kd+T].
            let frames = xpad.narrow(2, kd, t)?;
            // [B, C, T, H, W] → [B*T, C, H, W] for conv2d.
            let merged = frames
                .permute((0, 2, 1, 3, 4))?
                .reshape((b * t, c, h, w))?
                .contiguous()?;
            let y = chunked_conv2d(&merged, &wk, self.spatial_pad)?;
            acc = Some(match acc {
                Some(a) => (a + y)?,
                None => y,
            });
        }
        let y = acc.expect("kt >= 1");
        let (_, o, hp, wp) = y.dims4()?;
        let y = y
            .reshape((b, t, o, hp, wp))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?;
        y.broadcast_add(&self.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn im2col_chunking_matches_plain_conv2d() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (3, 4, 17, 13), &dev)?;
        let w = Tensor::randn(0f32, 1f32, (5, 4, 3, 3), &dev)?;
        let want = x.conv2d(&w, 1, 1, 1, 1)?;
        // Forces both the single-frame row fallback and several output-row bands.
        let got = chunked_conv2d_budgeted(&x, &w, 1, 512)?;
        assert_eq!(got.dims(), want.dims());
        let got = got.flatten_all()?.to_vec1::<f32>()?;
        let want = want.flatten_all()?.to_vec1::<f32>()?;
        let max_error = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_error < 1e-4, "chunked conv diverged: {max_error}");
        Ok(())
    }

    #[test]
    fn im2col_size_overflow_is_catchable() {
        let error = checked_product(&[usize::MAX, 2], "test").unwrap_err();
        assert!(error.to_string().contains("overflows usize"));
    }

    #[test]
    fn training_resolution_first_conv_stays_below_u32() {
        // A 1024px training frame is patchified to 256px with 48 channels before conv_in.
        let elements = checked_product(&[256, 256, 48, 3, 3], "1024px encoder conv").unwrap();
        assert!(elements < u32::MAX as usize);
        assert!(IM2COL_BUDGET < u32::MAX as usize);
    }

    #[test]
    fn upsampler_zero_temporal_padding_is_not_vae_edge_replication() -> Result<()> {
        let d = Device::Cpu;
        let weight = Tensor::ones((1, 1, 3, 1, 1), candle_gen::candle_core::DType::F32, &d)?;
        let bias = Tensor::zeros((1, 1, 1, 1, 1), candle_gen::candle_core::DType::F32, &d)?;
        let causal = CausalConv3d {
            weight: weight.clone(),
            bias: bias.clone(),
            kt: 3,
            spatial_pad: 0,
        };
        let zero = ZeroPaddedConv3d {
            weight,
            bias,
            kt: 3,
            spatial_pad: 0,
        };
        let x = Tensor::from_vec(vec![1f32, 2.], (1, 1, 2, 1, 1), &d)?;
        assert_eq!(
            zero.forward(&x)?.flatten_all()?.to_vec1::<f32>()?,
            vec![3., 3.]
        );
        assert_eq!(
            causal.forward(&x, false)?.flatten_all()?.to_vec1::<f32>()?,
            vec![4., 5.]
        );
        Ok(())
    }
}
