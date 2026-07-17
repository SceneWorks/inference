//! Mochi AsymmVAE **3-D causal conv** (`MochiCausalConv3d`, diffusers `pad_mode="replicate"`) —
//! candle ships no `conv3d`, and because video has `T > 1` a `kt×kh×kw` kernel does not reduce to one
//! conv2d. It is decomposed into `kt` conv2d "taps" (the LTX `CausalConv3d` template): the input is
//! **replicate**-padded (temporal `(kt−1)` on the **front only** — causal; spatial `(kh−1)/2` /
//! `(kw−1)/2` symmetric), then the output is `Σ_{kd} conv2d(x_pad[:, :, kd : kd+T], W[:, :, kd])`.
//!
//! Unlike LTX's zero spatial padding, Mochi's `MochiCausalConv3d` uses `mode="replicate"` for **both**
//! the temporal front-pad and the spatial pad (the MLX port's `PadMode::Edge` over the whole tuple), so
//! the spatial replicate pad is applied explicitly here and conv2d runs with `padding = 0`. The weight
//! is the checkpoint-native PyTorch layout `[O, I, kt, kh, kw]` (candle keeps the HF layout — no MLX
//! `[O, kt, kh, kw, I]` transpose). A `1×1×1` kernel (the decoder `conv_in`) degenerates to a plain
//! conv (no padding), matching the reference.

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::VarBuilder;

/// Per-conv temporal cache threaded through a chunked decode: each slot holds the last `kt−1` frames
/// of that conv's input from the previous chunk. `idx` resets to 0 each chunk and advances once per
/// conv in the fixed traversal order, so slots stay aligned. Mirrors the MLX port's `FrameCache` and
/// the `conv_cache` diffusers threads through `AutoencoderKLMochi`'s framewise decode.
///
/// **Why a cache and not overlap+blend.** Every op in this decoder is per-frame (`GroupNorm(32)` is
/// per-frame; silu/residual/proj are elementwise or per-position) or a causal conv, so feeding a chunk
/// the previous chunk's real trailing frames reproduces the single-shot decode *exactly* — no seams to
/// blend. The decoder's temporal receptive field is **~45 latent frames** (38 stacked `kt=3` causal
/// convs), *wider than a whole 5 s clip* (26 latent frames), so a tile given the 1 frame of left
/// context that `gen_core::tiling`'s causal path allows is wrong throughout, not merely seamed. The
/// MLX port measures both claims in `mlx-gen-mochi/tests/chunked_decode.rs`.
pub struct FrameCache {
    slots: Vec<Option<Tensor>>,
    pub idx: usize,
}

impl FrameCache {
    /// A cache with one slot per causal conv, all empty (chunk 0 falls back to the replicate pad).
    pub fn new(n: usize) -> Self {
        Self {
            slots: vec![None; n],
            idx: 0,
        }
    }

    /// Reset the slot cursor to the start of the traversal (called once per chunk).
    pub fn rewind(&mut self) {
        self.idx = 0;
    }
}

/// A 3-D causal conv loaded from a `[O, I, kt, kh, kw]` weight (channels + kernel dims ride on the
/// weight, not a config). Temporal stride is always 1; spatial padding is "same" replicate
/// (`(kh−1)/2`); temporal padding is front-only replicate (`kt−1`).
pub struct CausalConv3d {
    weight: Tensor, // [O, I, kt, kh, kw]
    bias: Tensor,   // [1, O, 1, 1, 1]
    kt: usize,
    h_pad: usize,
    w_pad: usize,
}

impl CausalConv3d {
    /// Load `{name}.weight` (torch `[O, I, kt, kh, kw]`) + `{name}.bias` from `vb`. `name` is the conv
    /// leaf relative to `vb` — `"conv"` for a `CogVideoXCausalConv3d`-wrapped resnet conv
    /// (`…conv1.conv.weight`), `"conv_in"` for the plain decoder input conv.
    pub fn load(vb: &VarBuilder, name: &str) -> Result<Self> {
        let w = vb.get_unchecked(&format!("{name}.weight"))?.contiguous()?;
        let dims = w.dims();
        let (out_c, kt, kh, kw) = (dims[0], dims[2], dims[3], dims[4]);
        let bias = vb
            .get_unchecked(&format!("{name}.bias"))?
            .reshape((1, out_c, 1, 1, 1))?;
        Ok(Self {
            weight: w,
            bias,
            kt,
            h_pad: (kh - 1) / 2,
            w_pad: (kw - 1) / 2,
        })
    }

    /// `x`: `[B, C, T, H, W]` → `[B, O, T, H, W]` (spatial "same" replicate, temporal front-replicate).
    ///
    /// `cache` threads the chunked decode: when its slot for this conv is populated (every chunk after
    /// the first) the previous chunk's real trailing frames stand in for the causal front-pad, which is
    /// what makes chunked == single-shot. `None` (or an empty slot) falls back to the replicate pad.
    pub fn forward(&self, x: &Tensor, cache: Option<&mut FrameCache>) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;

        // Temporal context: real history when chunking, else the causal replicate pad of frame 0.
        let xt = if self.kt > 1 {
            match cache.as_ref().and_then(|cc| cc.slots[cc.idx].clone()) {
                Some(prev) => Tensor::cat(&[&prev, x], 2)?,
                None => {
                    let first = x.narrow(2, 0, 1)?;
                    let front = repeat_along(&first, 2, self.kt - 1)?;
                    Tensor::cat(&[&front, x], 2)?
                }
            }
        } else {
            x.clone()
        };

        // Hand the next chunk this conv's trailing frames (taken post-concat, as diffusers does, so a
        // chunk shorter than `kt−1` still carries forward the right history), then advance the slot.
        if let Some(cc) = cache {
            if self.kt > 1 {
                let n = xt.dim(2)?;
                cc.slots[cc.idx] =
                    Some(xt.narrow(2, n - (self.kt - 1), self.kt - 1)?.contiguous()?);
            }
            cc.idx += 1;
        }

        // conv2d taps over the temporal kernel; each tap replicate-pads H/W then convolves (padding 0).
        let mut acc: Option<Tensor> = None;
        for kd in 0..self.kt {
            let wk = self.weight.narrow(2, kd, 1)?.squeeze(2)?.contiguous()?; // [O, I, kh, kw]
            let frames = xt.narrow(2, kd, t)?; // [B, C, T, H, W]
            let merged = frames
                .permute((0, 2, 1, 3, 4))?
                .reshape((b * t, c, h, w))?
                .contiguous()?; // [B·T, C, H, W]
            let padded = edge_pad2d(&merged, self.h_pad, self.w_pad)?;
            let y = padded.conv2d(&wk, 0, 1, 1, 1)?; // padding 0 (already replicate-padded)
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
            .contiguous()?; // [B, O, T, H, W]
        y.broadcast_add(&self.bias)
    }
}

/// Replicate `x` `n` times along `axis` (`Tensor::cat` of `n` clones — the LTX `repeat_frame` idiom).
fn repeat_along(x: &Tensor, axis: usize, n: usize) -> Result<Tensor> {
    let parts: Vec<Tensor> = (0..n).map(|_| x.clone()).collect();
    Tensor::cat(&parts, axis)
}

/// Edge/replicate-pad the H (axis 2) and W (axis 3) of a `[B, C, H, W]` tensor by `h_pad` / `w_pad` on
/// each side (the diffusers `mode="replicate"` spatial pad).
fn edge_pad2d(x: &Tensor, h_pad: usize, w_pad: usize) -> Result<Tensor> {
    let x = edge_pad_axis(x, 2, h_pad)?;
    edge_pad_axis(&x, 3, w_pad)
}

/// Edge-replicate-pad `axis` by `p` on each side (narrow the first/last slice, replicate, concat).
fn edge_pad_axis(x: &Tensor, axis: usize, p: usize) -> Result<Tensor> {
    if p == 0 {
        return Ok(x.clone());
    }
    let n = x.dim(axis)?;
    let first = x.narrow(axis, 0, 1)?;
    let last = x.narrow(axis, n - 1, 1)?;
    let front = repeat_along(&first, axis, p)?;
    let back = repeat_along(&last, axis, p)?;
    Tensor::cat(&[&front, &x.clone(), &back], axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};

    /// Edge padding replicates the border rows/cols (not zeros): a 1×1×2×2 field padded by 1 on H/W
    /// has its corner value repeated into the new corners.
    #[test]
    fn edge_pad_replicates_borders() {
        let dev = Device::Cpu;
        // [B=1, C=1, H=2, W=2] with values [[1,2],[3,4]].
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &dev).unwrap();
        let p = edge_pad2d(&x, 1, 1).unwrap();
        assert_eq!(p.dims(), &[1, 1, 4, 4]);
        let v = p.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Top-left 3×3 corner replicates value 1; row 0 = [1,1,2,2], last row = [3,3,4,4].
        assert_eq!(&v[0..4], &[1.0, 1.0, 2.0, 2.0]);
        assert_eq!(&v[12..16], &[3.0, 3.0, 4.0, 4.0]);
    }

    /// A `1×1×1` conv (the decoder `conv_in`) is a plain 1×1 conv over each frame with no padding: the
    /// tap loop runs once and shapes pass through (T,H,W unchanged).
    #[test]
    fn conv_1x1x1_is_shape_preserving() {
        let dev = Device::Cpu;
        // weight [O=2, I=3, 1,1,1], bias [2].
        let w = Tensor::ones((2, 3, 1, 1, 1), DType::F32, &dev).unwrap();
        let b = Tensor::zeros(2, DType::F32, &dev).unwrap();
        let mut map = std::collections::HashMap::new();
        map.insert("c.weight".to_string(), w);
        map.insert("c.bias".to_string(), b);
        let vb = VarBuilder::from_tensors(map, DType::F32, &dev);
        let conv = CausalConv3d::load(&vb, "c").unwrap();
        let x = Tensor::ones((1, 3, 4, 5, 6), DType::F32, &dev).unwrap();
        let y = conv.forward(&x, None).unwrap();
        // [B, O=2, T=4, H=5, W=6]; each output = sum over 3 input channels of 1·1 = 3.
        assert_eq!(y.dims(), &[1, 2, 4, 5, 6]);
        let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|&x| (x - 3.0).abs() < 1e-5));
    }
}
