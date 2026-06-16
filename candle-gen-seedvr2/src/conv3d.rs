//! Generalized **CausalConv3d** for the SeedVR2 3D causal video VAE — candle ships no `conv3d`, so a
//! `kT×kH×kW` kernel is decomposed into `kT` conv2d "taps" summed over the temporal axis (the proven
//! candle pattern: `candle-gen-wan`/`-svd` + candle-transformers `qwen3_vl::conv3d_temporal_2`). This
//! generalizes the Wan/SVD temporal-only convs to SeedVR2's needs:
//!   * spatial kernels + spatial stride (the stride-2 downsample convs) + symmetric spatial padding,
//!   * temporal stride 2 (the temporal downsamplers),
//!   * **first-frame-repeat** causal temporal padding — NOT zero-pad — matching the mflux reference
//!     `common/conv3d.py` (`causal_pad = use_padding_causal ? 2·pt : kt-1`, repeat frame 0).
//!
//! NCTHW in/out. Weight is torch-layout `[O, I, kT, kH, kW]` (loaded as-is: candle conv2d's per-tap
//! kernel is exactly `W[:, :, kd] = [O, I, kH, kW]`). All SeedVR2 convs have `ph == pw` and `sh == sw`
//! (candle conv2d takes one symmetric padding/stride); the asymmetric `(0,1)` downsample pad is added
//! by the caller (`vae::Downsample3d`) before the conv, exactly like the reference.

use candle_gen::candle_core::{Result, Tensor};

use crate::weights::Weights;

/// Max im2col elements (`batch · H_out · W_out · C · kH · kW`) per `conv2d` call. candle's CUDA conv2d
/// builds an im2col buffer of this size; above a few hundred million it silently corrupts/zeros part
/// of the output (an 8·64²·512·9 ≈ 151M call is fine, a 16·128²·512·9 ≈ 1.2B call drops frames). We
/// chunk the merged `B·T_out` batch so every conv2d stays in the proven-safe band. The image path
/// (`T_out==1`) never tripped this; it only bites multi-frame (video) clips at high resolution.
const IM2COL_BUDGET: usize = 128 * 1024 * 1024;

/// `x.conv2d`, but with the batch axis split into chunks small enough that each call's im2col stays
/// under [`IM2COL_BUDGET`]; results are concatenated back along the batch axis. Identical math to a
/// single `conv2d`, just split to dodge the large-buffer corruption.
fn chunked_conv2d(x: &Tensor, w: &Tensor, padding: usize, stride: usize) -> Result<Tensor> {
    let n = x.dim(0)?;
    let (c, h, wd) = (x.dim(1)?, x.dim(2)?, x.dim(3)?);
    let (_o, _i, kh, kw) = w.dims4()?;
    let h_out = (h + 2 * padding).saturating_sub(kh) / stride + 1;
    let w_out = (wd + 2 * padding).saturating_sub(kw) / stride + 1;
    let cols_per_frame = (h_out * w_out * c * kh * kw).max(1);
    let max_batch = (IM2COL_BUDGET / cols_per_frame).clamp(1, n);
    if n <= max_batch {
        return x.conv2d(w, padding, stride, 1, 1);
    }
    let mut parts = Vec::new();
    let mut start = 0;
    while start < n {
        let len = (n - start).min(max_batch);
        parts.push(x.narrow(0, start, len)?.conv2d(w, padding, stride, 1, 1)?);
        start += len;
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 0)
}

pub struct CausalConv3d {
    weight: Tensor, // [O, I, kT, kH, kW]
    bias: Tensor,   // [1, O, 1, 1, 1]
    kt: usize,
    st: usize,
    sh: usize,
    #[allow(dead_code)]
    sw: usize,
    pt: usize,
    ph: usize,
    #[allow(dead_code)]
    pw: usize,
    use_padding_causal: bool,
}

impl CausalConv3d {
    /// Load `{prefix}.weight` (`[O,I,kT,kH,kW]`) + `{prefix}.bias` (`[O]`). `stride`/`padding` are
    /// `(t,h,w)`; `use_padding_causal` selects the `2·pt` causal-pad rule (else `kt-1`).
    pub fn load(
        w: &Weights,
        prefix: &str,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        use_padding_causal: bool,
    ) -> candle_gen::Result<Self> {
        let weight = w.require(&format!("{prefix}.weight"))?.clone();
        let (_o, _i, kt, _kh, _kw) = weight.dims5()?;
        let bias = w.require(&format!("{prefix}.bias"))?;
        let o = bias.dim(0)?;
        let bias = bias.reshape((1, o, 1, 1, 1))?;
        debug_assert_eq!(padding.1, padding.2, "seedvr2 conv: ph == pw");
        debug_assert_eq!(stride.1, stride.2, "seedvr2 conv: sh == sw");
        Ok(Self {
            weight,
            bias,
            kt,
            st: stride.0,
            sh: stride.1,
            sw: stride.2,
            pt: padding.0,
            ph: padding.1,
            pw: padding.2,
            use_padding_causal,
        })
    }

    /// `x`: `[B, C, T, H, W]` → `[B, O, T_out, H_out, W_out]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, _t, h, w) = x.dims5()?;
        // Temporal causal padding (repeat the first frame), or symmetric zero-pad for kt==1.
        let xpad = if self.kt > 1 {
            let causal_pad = if self.use_padding_causal {
                2 * self.pt
            } else {
                self.kt - 1
            };
            if causal_pad > 0 {
                let first = x.narrow(2, 0, 1)?; // [B,C,1,H,W]
                let rep = first.broadcast_as((b, c, causal_pad, h, w))?.contiguous()?;
                Tensor::cat(&[&rep, x], 2)?
            } else {
                x.clone()
            }
        } else if self.pt > 0 {
            x.pad_with_zeros(2, self.pt, self.pt)?
        } else {
            x.clone()
        };
        let tpad = xpad.dim(2)?;
        // VALID temporal conv with temporal stride → T_out output frames.
        let t_out = (tpad - self.kt) / self.st + 1;
        let dev = x.device();

        let mut acc: Option<Tensor> = None;
        for kd in 0..self.kt {
            // Tap weight W[:, :, kd] → [O, I, kH, kW] (one conv2d).
            let wk = self.weight.narrow(2, kd, 1)?.squeeze(2)?.contiguous()?;
            // The T_out frames this tap convolves: x_pad[:, :, kd + o·st], o in 0..T_out.
            let frames = if self.st == 1 {
                xpad.narrow(2, kd, t_out)?
            } else {
                let idx: Vec<u32> = (0..t_out).map(|o| (kd + o * self.st) as u32).collect();
                let idx = Tensor::from_vec(idx, t_out, dev)?;
                xpad.index_select(&idx, 2)?
            };
            // Merge (B, T_out) into the conv2d batch axis: [B, C, T_out, H, W] → [B·T_out, C, H, W].
            let merged =
                frames
                    .permute((0, 2, 1, 3, 4))?
                    .contiguous()?
                    .reshape((b * t_out, c, h, w))?;
            // Batch-chunked so each conv2d's im2col stays under candle's CUDA limit (see [`chunked_conv2d`]).
            let y = chunked_conv2d(&merged, &wk, self.ph, self.sh)?; // [B·T_out, O, H_out, W_out]
            acc = Some(match acc {
                Some(a) => (a + y)?,
                None => y,
            });
        }
        let y = acc.expect("kt >= 1");
        let (_, o, hp, wp) = y.dims4()?;
        let y = y
            .reshape((b, t_out, o, hp, wp))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?;
        y.broadcast_add(&self.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};

    fn conv(
        weight: Tensor,
        bias: Tensor,
        st: usize,
        ph: usize,
        pt: usize,
        upc: bool,
    ) -> CausalConv3d {
        let (_o, _i, kt, _kh, _kw) = weight.dims5().unwrap();
        let o = bias.dim(0).unwrap();
        CausalConv3d {
            weight,
            bias: bias.reshape((1, o, 1, 1, 1)).unwrap(),
            kt,
            st,
            sh: 1,
            sw: 1,
            pt,
            ph,
            pw: ph,
            use_padding_causal: upc,
        }
    }

    /// A 3×3×3 causal conv (stride 1, spatial pad 1) preserves T,H,W (causal repeat = 2 frames).
    #[test]
    fn k333_preserves_dims() -> Result<()> {
        let dev = Device::Cpu;
        let (o, i) = (4usize, 3usize);
        let c = conv(
            Tensor::randn(0f32, 1.0, (o, i, 3, 3, 3), &dev)?,
            Tensor::zeros(o, DType::F32, &dev)?,
            1,
            1,
            1,
            false,
        );
        let x = Tensor::randn(0f32, 1.0, (1, i, 5, 8, 8), &dev)?;
        let y = c.forward(&x)?;
        assert_eq!(y.dims(), &[1, o, 5, 8, 8]);
        Ok(())
    }

    /// A temporal-stride-2 downsample conv (3×3×3, causal pad 2) halves T (ceil) — the encoder
    /// temporal down block.
    #[test]
    fn temporal_stride2_halves_t() -> Result<()> {
        let dev = Device::Cpu;
        let (o, i) = (2usize, 2usize);
        // stride (2,1,1), spatial pad 1 here for shape simplicity.
        let c = conv(
            Tensor::randn(0f32, 1.0, (o, i, 3, 3, 3), &dev)?,
            Tensor::zeros(o, DType::F32, &dev)?,
            2,
            1,
            1,
            false,
        );
        let x = Tensor::randn(0f32, 1.0, (1, i, 9, 4, 4), &dev)?;
        let y = c.forward(&x)?; // T_out = (9+2-3)/2 + 1 = 5
        assert_eq!(y.dims(), &[1, o, 5, 4, 4]);
        Ok(())
    }

    /// Brute-force causal conv3d reference (NCTHW, cross-correlation, first-frame causal pad, symmetric
    /// spatial zero-pad, stride). Compares the conv2d-temporal-sum decomposition against it for a
    /// **T>1, distinct-frame** input — the case the image (T=1) path never exercised. Covers temporal
    /// stride 1 (resnet/conv_in convs) AND stride 2 (the encoder temporal downsamplers).
    #[test]
    fn matches_bruteforce_conv3d_t_gt_1() -> Result<()> {
        for st in [1usize, 2usize] {
            bruteforce_conv3d_case(st)?;
        }
        Ok(())
    }

    fn bruteforce_conv3d_case(st: usize) -> Result<()> {
        let dev = Device::Cpu;
        let (o, i, kt, kh, kw) = (3usize, 2usize, 3usize, 3usize, 3usize);
        let (t, h, w) = (18usize, 5usize, 5usize); // T large enough to exercise t_out=16+
        let ph = 1usize;
        let wt = Tensor::randn(0f32, 1.0, (o, i, kt, kh, kw), &dev)?;
        let bias = Tensor::randn(0f32, 1.0, o, &dev)?;
        let x = Tensor::randn(0f32, 1.0, (1, i, t, h, w), &dev)?;
        let c = conv(wt.clone(), bias.clone(), st, ph, 1, false);
        let got = c.forward(&x)?;

        // brute force
        let causal_pad = kt - 1;
        let wv = wt.flatten_all()?.to_vec1::<f32>()?; // [o,i,kt,kh,kw] row-major
        let bv = bias.to_vec1::<f32>()?;
        let xv = x.flatten_all()?.to_vec1::<f32>()?; // [1,i,t,h,w]
        let xat = |ci: usize, ti: usize, hi: i64, wi: i64| -> f32 {
            if hi < 0 || wi < 0 || hi as usize >= h || wi as usize >= w {
                return 0.0;
            }
            // causal pad: padded indices [0..causal_pad) clamp to real frame 0.
            let real_t = ti.saturating_sub(causal_pad);
            xv[((ci * t + real_t) * h + hi as usize) * w + wi as usize]
        };
        let t_out = (t + causal_pad - kt) / st + 1;
        let wat = |oi: usize, ci: usize, a: usize, b2: usize, c2: usize| -> f32 {
            wv[(((oi * i + ci) * kt + a) * kh + b2) * kw + c2]
        };
        let mut exp = vec![0f32; o * t_out * h * w];
        for oi in 0..o {
            for to in 0..t_out {
                for ho in 0..h {
                    for wo in 0..w {
                        let mut acc = bv[oi];
                        for ci in 0..i {
                            for a in 0..kt {
                                for b2 in 0..kh {
                                    for c2 in 0..kw {
                                        let ti = to * st + a;
                                        let hi = (ho + b2) as i64 - ph as i64;
                                        let wi = (wo + c2) as i64 - ph as i64;
                                        acc += xat(ci, ti, hi, wi) * wat(oi, ci, a, b2, c2);
                                    }
                                }
                            }
                        }
                        exp[((oi * t_out + to) * h + ho) * w + wo] = acc;
                    }
                }
            }
        }
        let gv = got.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(gv.len(), exp.len());
        let max_err = gv
            .iter()
            .zip(exp.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_err < 1e-3,
            "conv3d decomposition wrong at T>1 (st={st}): max_err={max_err}"
        );
        Ok(())
    }

    /// A 1×1×1 conv is pointwise and preserves all dims (the resnet shortcut / upscale conv).
    #[test]
    fn k111_pointwise() -> Result<()> {
        let dev = Device::Cpu;
        let (o, i) = (5usize, 3usize);
        let c = conv(
            Tensor::randn(0f32, 1.0, (o, i, 1, 1, 1), &dev)?,
            Tensor::zeros(o, DType::F32, &dev)?,
            1,
            0,
            0,
            false,
        );
        let x = Tensor::randn(0f32, 1.0, (1, i, 6, 3, 3), &dev)?;
        let y = c.forward(&x)?;
        assert_eq!(y.dims(), &[1, o, 6, 3, 3]);
        Ok(())
    }
}
