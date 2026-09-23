//! Shared **launch-bound-safe** 2-D convolution (sc-24114).
//!
//! candle's non-cuDNN CUDA `conv2d` is im2col + GEMM: it materialises a
//! `[B · H_out · W_out, C_in · k_h · k_w]` column buffer and launches the fill kernel with
//! `LaunchConfig::for_num_elems(dst_el as u32)` — the element count is **truncated to u32** at the
//! launch. A buffer past `u32::MAX` (≈4.29e9) elements is therefore only filled for its first
//! `dst_el mod 2^32` entries and the rest is whatever the allocator handed back: no error, a green
//! run, and an image whose top rows are right and whose remainder is stale memory.
//!
//! An image VAE's last decoder stage reaches that bound at the advertised render sizes. The
//! Qwen-Image 2.1 decoder runs its full-resolution stage at 144 channels (the upsampler's conv at
//! 288 in), so at the 2048² default preset one 3×3 conv's im2col is `2048² · 288 · 9 ≈ 1.09e10`
//! elements (`≈ 5.4e9` for the 144-in resnets): the truncated launch fills exactly
//! `(5.44e9 mod 2^32) / (2048 · 144 · 9) = 429.8` output rows — the 430-row correct strip the
//! sc-24114 CUDA evidence render shows above a field of noise. On CPU the same op is merely a
//! 21.7 GB f32 transient.
//!
//! [`conv2d_budgeted`] is the guard: when the im2col buffer would exceed `budget` elements it
//! zero-pads the input once and runs the convolution over **output-row chunks**, each fed exactly
//! the padded input rows its receptive field reads, then concatenates along the row axis. A
//! convolution is local, so the chunked result is *mathematically* identical to the single pass —
//! but, as with [`crate::attention`], not bitwise: a different GEMM `M` may change the accumulation
//! order, so the two agree to a tolerance (`1e-5` in this module's tests). Below the budget the
//! call is the plain `Conv2d::forward`, byte-identical to before the guard existed.

use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Module};

/// The im2col element budget one chunk may materialise — 2^30, a quarter of the u32 launch bound
/// so the `as u32` truncation can never engage, and 4 GB f32 / 2 GB bf16 of transient per chunk.
/// The Qwen-Image 2.1 decoder's ≤512² stages (and every parity fixture) stay under it — the plain
/// single pass; its 1024² full-resolution convs (1.4e9 / 2.7e9 elements, a 5.4 GB bf16 transient
/// un-chunked) split in 2–3, its 2048² ones in 6–11.
pub const CONV_IM2COL_BUDGET: usize = 1 << 30;

/// Output rows per chunk for a conv over `x_dims` (`[B, C_in, H, W]`) with `conv`'s kernel and
/// config under `budget`, or `None` when the whole im2col buffer fits and the call is a single
/// pass. Exposed so a caller can assert its own geometry engages (or never engages) the guard.
pub fn conv2d_row_plan(x_dims: [usize; 4], conv: &Conv2d, budget: usize) -> Result<Option<usize>> {
    let cfg = conv.config();
    let (_, _, k_h, k_w) = conv.weight().dims4()?;
    let [b, c_in, h, w] = x_dims;
    let (h_out, w_out) = (
        out_len(h, k_h, cfg.padding, cfg.stride, cfg.dilation),
        out_len(w, k_w, cfg.padding, cfg.stride, cfg.dilation),
    );
    let per_row = (b * w_out * c_in * k_h * k_w) as u64;
    let total = per_row * h_out as u64;
    if total <= budget as u64 || h_out <= 1 {
        return Ok(None);
    }
    Ok(Some(
        (budget as u64 / per_row.max(1)).clamp(1, h_out as u64) as usize,
    ))
}

/// `conv.forward(x)` for `x: [B, C_in, H, W]`, chunked over output rows whenever its im2col buffer
/// would exceed `budget` elements (see the module docs). Production passes
/// [`CONV_IM2COL_BUDGET`]; a test forces a small budget to drive the chunked branch at fixture
/// size.
pub fn conv2d_budgeted(x: &Tensor, conv: &Conv2d, budget: usize) -> Result<Tensor> {
    let dims = x.dims4()?;
    let Some(rows) = conv2d_row_plan([dims.0, dims.1, dims.2, dims.3], conv, budget)? else {
        return conv.forward(x);
    };
    let cfg = conv.config();
    let (p, s, d) = (cfg.padding, cfg.stride, cfg.dilation);
    let (_, _, k_h, _) = conv.weight().dims4()?;
    let h_out = out_len(dims.2, k_h, p, s, d);
    // Pad once; every chunk then convolves with padding 0 over exactly the padded rows its
    // receptive field reads, so no chunk sees an extra (or a missing) zero border.
    let padded = if p > 0 {
        x.pad_with_zeros(2, p, p)?.pad_with_zeros(3, p, p)?
    } else {
        x.clone()
    };
    let span = d * (k_h - 1) + 1;
    let mut chunks = Vec::with_capacity(h_out.div_ceil(rows));
    let mut start = 0;
    while start < h_out {
        let len = rows.min(h_out - start);
        let slice = padded
            .narrow(2, start * s, (len - 1) * s + span)?
            .contiguous()?;
        let y = slice.conv2d_with_algo(conv.weight(), 0, s, d, cfg.groups, cfg.cudnn_fwd_algo)?;
        debug_assert_eq!(y.dim(2)?, len);
        chunks.push(y);
        start += len;
    }
    let y = Tensor::cat(&chunks, 2)?;
    match conv.bias() {
        None => Ok(y),
        Some(bias) => {
            let b = bias.dims1()?;
            y.broadcast_add(&bias.reshape((1, b, 1, 1))?)
        }
    }
}

fn out_len(input: usize, k: usize, padding: usize, stride: usize, dilation: usize) -> usize {
    (input + 2 * padding - dilation * (k - 1) - 1) / stride + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::Conv2dConfig;

    fn fill(shape: (usize, usize, usize, usize), seed: usize) -> Tensor {
        let n = shape.0 * shape.1 * shape.2 * shape.3;
        let v: Vec<f32> = (0..n)
            .map(|i| (((i * 37 + seed * 11) % 97) as f32 / 97.0) - 0.5)
            .collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        assert_eq!(a.dims(), b.dims());
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// Every conv shape the Qwen-Image 2.1 VAE runs (3×3/p1/s1 resnets and upsamplers, the
    /// explicitly-padded 3×3/p0/s2 downsampler, 1×1 shortcuts and quant convs) plus a dilated
    /// case: a one-row chunking and a ragged multi-row chunking both match the single pass.
    #[test]
    fn chunked_conv_matches_single_pass_across_the_vae_shapes() {
        let (b, c_in, c_out, h, w) = (2usize, 3usize, 4usize, 11usize, 7usize);
        let x = fill((b, c_in, h, w), 0);
        for (k, padding, stride, dilation, bias) in [
            (3usize, 1usize, 1usize, 1usize, true),
            (3, 0, 2, 1, true),
            (1, 0, 1, 1, false),
            (3, 2, 1, 2, true),
        ] {
            let weight = fill((c_out, c_in, k, k), 1);
            let bias = bias.then(|| fill((1, c_out, 1, 1), 2).flatten_all().unwrap());
            let conv = Conv2d::new(
                weight,
                bias,
                Conv2dConfig {
                    padding,
                    stride,
                    dilation,
                    ..Default::default()
                },
            );
            let single = conv2d_budgeted(&x, &conv, usize::MAX).unwrap();
            assert_eq!(
                conv2d_row_plan([b, c_in, h, w], &conv, usize::MAX).unwrap(),
                None
            );
            // Exactly the un-guarded op below the budget.
            assert_eq!(max_abs_diff(&single, &conv.forward(&x).unwrap()), 0.0);

            let h_out = out_len(h, k, padding, stride, dilation);
            let w_out = out_len(w, k, padding, stride, dilation);
            let per_row = b * w_out * c_in * k * k;
            for rows in [1usize, 3] {
                let budget = per_row * rows;
                assert_eq!(
                    conv2d_row_plan([b, c_in, h, w], &conv, budget).unwrap(),
                    Some(rows),
                    "k{k} p{padding} s{stride} d{dilation}: budget {budget} must plan {rows} rows"
                );
                let chunked = conv2d_budgeted(&x, &conv, budget).unwrap();
                assert_eq!(chunked.dims(), &[b, c_out, h_out, w_out]);
                let diff = max_abs_diff(&single, &chunked);
                assert!(
                    diff <= 1e-5,
                    "k{k} p{padding} s{stride} d{dilation} rows {rows}: max |Δ| = {diff}"
                );
            }
        }
    }

    /// The production geometry the guard exists for: the Qwen-Image 2.1 decoder's full-resolution
    /// stage (sc-24114 evidence render). At 2048² both its convs exceed the u32 launch bound
    /// un-chunked — the upsampler's 288-in conv by 2.5×, the 144-in resnets by 1.27× — and chunk
    /// under the shipped budget; at 1024² they are under the bound but still chunk (a 5.4 GB bf16
    /// transient otherwise); the 512² stage is a single pass.
    #[test]
    fn the_qwen_image_2_1_full_resolution_stage_chunks_at_2048_and_1024_not_512() {
        let conv = |c_in: usize| {
            Conv2d::new(
                Tensor::zeros((144, c_in, 3, 3), candle_core::DType::F32, &Device::Cpu).unwrap(),
                None,
                Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
            )
        };
        for (c_in, side, past_launch_bound, expect_chunked) in [
            (288usize, 2048usize, true, true),
            (144, 2048, true, true),
            (288, 1024, false, true),
            (144, 1024, false, true),
            (288, 512, false, false),
            (144, 512, false, false),
        ] {
            let im2col = (side * side * c_in * 9) as u64;
            assert_eq!(
                im2col > u64::from(u32::MAX),
                past_launch_bound,
                "{c_in}-in at {side}²: im2col {im2col} vs the u32 launch bound"
            );
            let plan =
                conv2d_row_plan([1, c_in, side, side], &conv(c_in), CONV_IM2COL_BUDGET).unwrap();
            assert_eq!(
                plan.is_some(),
                expect_chunked,
                "{c_in}-in at {side}²: {plan:?}"
            );
            if let Some(rows) = plan {
                // Every chunk stays under the budget and hence far under the launch bound.
                assert!((rows * side * c_in * 9) as u64 <= CONV_IM2COL_BUDGET as u64);
                assert!(rows >= 1 && rows < side);
            }
        }
        // The truncated launch's arithmetic, pinned: 430 correct rows at 144 in — what the
        // evidence PNG shows.
        let filled = (2048u64 * 2048 * 144 * 9) % (1u64 << 32);
        assert_eq!(filled / (2048 * 144 * 9), 429);
    }
}
