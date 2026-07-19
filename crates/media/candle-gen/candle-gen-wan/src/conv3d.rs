//! **Causal 3-D convolution** for the Wan temporal VAE — candle ships no `conv3d`, and because video
//! has `T > 1` the conv3d does *not* reduce to a single conv2d (unlike a single-image VAE). Instead a
//! `kD×kH×kW` kernel is decomposed into `kD` conv2d "taps": the temporal axis is causally left-padded
//! by `kD-1` zero frames, and the output is `Σ_{kd} conv2d(x_pad[:, :, kd : kd+T], W[:, :, kd])`.
//!
//! This reproduces diffusers' `WanCausalConv3d` exactly (its `_padding = (·, ·, ·, ·, 2·pad_t, 0)`
//! left-pad + VALID conv, temporal stride 1).
//!
//! **Two decode modes (sc-5176):**
//! - *Single pass* ([`Ctx::single_pass`]): one forward over all `T` frames with the causal
//!   left-padding — what we shipped originally. Correct, but the decoder's full-resolution
//!   activations for **every frame at once** spike VAE memory (~60 GB for a 320²×17 clip) → OOM.
//! - *Streaming* ([`Ctx::streaming`]): decode one latent frame at a time, each `CausalConv3d`
//!   carrying its last `kD-1` input frames as a `feat_cache` (diffusers' frame-by-frame path). The
//!   prepended cache replaces the would-be zero-pad, so streaming is **bit-equivalent** to the single
//!   pass (a causal conv over the whole clip == the cache-streamed one) while bounding peak memory to
//!   ~one frame's activations. The caller resets caches around the decode and flips `first_chunk`.

use std::sync::Mutex;

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::VarBuilder;

/// Decode context threaded through the VAE forward. `streaming` selects the per-frame `feat_cache`
/// path; `first_chunk` marks the first latent frame (the temporal-upsample "first frame un-doubled"
/// rule — used by the upsampler/dup in `vae.rs`, not here).
#[derive(Clone, Copy)]
pub struct Ctx {
    pub streaming: bool,
    pub first_chunk: bool,
}

impl Ctx {
    /// Whole-clip single pass (causal zero-pad over all frames).
    pub fn single_pass() -> Self {
        Self {
            streaming: false,
            first_chunk: true,
        }
    }
    /// One streaming chunk (one latent frame); `first` is the leading latent frame.
    pub fn streaming(first: bool) -> Self {
        Self {
            streaming: true,
            first_chunk: first,
        }
    }
}

/// Max im2col elements (`batch · H_out · W_out · C · kH · kW`) per `conv2d` call. candle's CUDA conv2d
/// builds an im2col buffer of this size; past a few hundred million elements it **silently corrupts**
/// (finite, in-range, but wrong pixels — the `int`/`u32` index band, [`crate::vae`] / sc-12773, the same
/// class as the SeedVR2 `conv2d` bug sc-5926 / sc-10083). GPU-measured on the z16 VAE (sc-12758/sc-12773,
/// RTX PRO 6000 sm_120): an untiled 1280×720 decode's final `96-ch 3×3` conv2d im2col (~796M elems)
/// corrupts, while a 640²×5 (~354M), a 448 px tile (~173M), and a 256 px tile (~56M) all decode clean —
/// so the threshold sits between ~354M and ~796M. We keep every conv2d well under that band at 128M by
/// splitting the merged `B·T_out` batch **and**, when even a single frame's im2col is too big
/// (high-resolution VAE convs, `T_out == 1` so there is no batch to split), the output rows.
pub(crate) const IM2COL_BUDGET: usize = 128 * 1024 * 1024;

/// `x.conv2d`, but split so each call's im2col stays under [`IM2COL_BUDGET`], dodging the large-buffer
/// corruption. **Identical math** to a single `conv2d` (`dilation == groups == 1`, the only forms the Wan
/// VAE uses): at or below the budget it is one un-chunked `conv2d` (so low-res decode is byte-for-byte
/// unchanged), above it, it splits the batch axis first (cheap — video's merged `B·T_out`), and when even
/// one frame's im2col exceeds the budget (a hi-res still frame, `B·T_out == 1`) it falls back to chunking
/// the **output rows** ([`row_chunked_conv2d`]). Shared by the z16 [`crate::vae16`] and z48
/// [`crate::vae`] decoders/encoders (`Conv2dW`, `CausalConv3d`, the z16 encoder down-samplers).
pub(crate) fn chunked_conv2d(
    x: &Tensor,
    w: &Tensor,
    padding: usize,
    stride: usize,
) -> Result<Tensor> {
    chunked_conv2d_budgeted(x, w, padding, stride, IM2COL_BUDGET)
}

/// [`chunked_conv2d`] with the im2col budget injected (so the batch/row split is unit-testable against a
/// plain `conv2d` without materializing a multi-GB tensor).
fn chunked_conv2d_budgeted(
    x: &Tensor,
    w: &Tensor,
    padding: usize,
    stride: usize,
    budget: usize,
) -> Result<Tensor> {
    let n = x.dim(0)?;
    let (c, h, wd) = (x.dim(1)?, x.dim(2)?, x.dim(3)?);
    let (_o, _i, kh, kw) = w.dims4()?;
    let h_out = (h + 2 * padding).saturating_sub(kh) / stride + 1;
    let w_out = (wd + 2 * padding).saturating_sub(kw) / stride + 1;
    let cols_per_frame = (h_out * w_out * c * kh * kw).max(1);
    // A single frame's im2col already busts the budget → batch splitting can't help (still frames are
    // `B·T_out == 1`). Chunk the output rows instead (the hi-res VAE conv, sc-12773).
    if cols_per_frame > budget {
        return row_chunked_conv2d(x, w, padding, stride, budget);
    }
    let max_batch = (budget / cols_per_frame).clamp(1, n);
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

/// Compute a conv2d as a stack of output-**row** bands, each small enough to stay under `budget`, then
/// concatenate along the height axis. Bit-identical to `x.conv2d(w, padding, stride, 1, 1)`: the symmetric
/// zero padding is applied explicitly up front (so a `padding == 0` band conv matches candle's built-in
/// symmetric pad), and output rows `[ro, ro+rb)` need exactly padded-input rows
/// `[ro·stride, ro·stride + (rb-1)·stride + kH)`. Used when a single frame's im2col exceeds the budget
/// (the hi-res VAE decode/encode convs), where batch splitting can't help (sc-12773 / sc-10083).
fn row_chunked_conv2d(
    x: &Tensor,
    w: &Tensor,
    padding: usize,
    stride: usize,
    budget: usize,
) -> Result<Tensor> {
    let n = x.dim(0)?;
    let c = x.dim(1)?;
    let (_o, _i, kh, kw) = w.dims4()?;
    // Explicit symmetric zero-pad on H (dim 2) and W (dim 3); the band convs then run with padding == 0.
    let xp = if padding > 0 {
        x.pad_with_zeros(2, padding, padding)?
            .pad_with_zeros(3, padding, padding)?
    } else {
        x.clone()
    };
    let hp = xp.dim(2)?;
    let wp = xp.dim(3)?;
    let h_out = hp.saturating_sub(kh) / stride + 1;
    let w_out = wp.saturating_sub(kw) / stride + 1;
    // Rows per band so `n · rb · w_out · c · kh · kw <= budget`; at least one row (best effort even if a
    // single output row is itself over budget — vastly smaller than the whole frame either way).
    let cols_per_out_row = (n * w_out * c * kh * kw).max(1);
    let band_rows = (budget / cols_per_out_row).max(1);
    let mut parts = Vec::new();
    let mut ro = 0;
    while ro < h_out {
        let rb = (h_out - ro).min(band_rows);
        let in_start = ro * stride;
        let in_len = (rb - 1) * stride + kh;
        let band = xp.narrow(2, in_start, in_len)?;
        parts.push(band.conv2d(w, 0, stride, 1, 1)?);
        ro += rb;
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 2)
}

/// A causal Conv3d loaded from a diffusers `[O, I, kD, kH, kW]` weight. Temporal stride is always 1
/// in the Wan decoder; spatial padding is "same" (`(kH-1)/2`), temporal padding is causal (left
/// `kD-1`). In streaming mode it carries the last `kD-1` **input** frames in `cache`.
pub struct CausalConv3d {
    weight: Tensor, // [O, I, kD, kH, kW]
    bias: Tensor,   // [1, O, 1, 1, 1]
    kd: usize,
    spatial_pad: usize,
    /// Streaming `feat_cache`: the last `kD-1` input frames of the previous chunk (≤ `kD-1` while
    /// warming up). `None` = first chunk / reset. `Mutex` (not `RefCell`) keeps the generator
    /// `Send + Sync` for the worker's generator cache (`Arc<WanVae>`); decode is single-threaded so the
    /// lock is always uncontended.
    cache: Mutex<Option<Tensor>>,
}

impl CausalConv3d {
    /// Load from `vb` with an explicit kernel `(kD, kH, kW)` (kH == kW) and channel counts.
    pub fn load(
        in_c: usize,
        out_c: usize,
        kernel: (usize, usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        crate::quant::guard_dense(&vb)?;
        let (kd, kh, kw) = kernel;
        let weight = vb.get((out_c, in_c, kd, kh, kw), "weight")?.contiguous()?;
        let bias = vb.get(out_c, "bias")?.reshape((1, out_c, 1, 1, 1))?;
        Ok(Self {
            weight,
            bias,
            kd,
            spatial_pad: (kh - 1) / 2,
            cache: Mutex::new(None),
        })
    }

    /// Drop any streaming `feat_cache` (call before/after a streaming decode).
    pub fn reset_cache(&self) {
        // sc-9015 / F-031: recover from a poisoned lock (reset-on-miss streaming cache; a prior
        // panic while locked must not brick every later decode).
        *candle_gen::lock_recover(&self.cache) = None;
    }

    /// `x`: `[B, C, T, H, W]` → `[B, O, T, H, W]` (spatial "same", temporal causal).
    pub fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        // Build the temporally-padded input `xpad` with exactly `t + (kD-1)` frames, so the kD-tap
        // VALID conv below yields `t` output frames.
        let xpad = if self.kd > 1 {
            let want = self.kd - 1; // left context frames needed
            if ctx.streaming {
                let old_cache = candle_gen::lock_recover(&self.cache).clone();
                let old_n = old_cache
                    .as_ref()
                    .map(|c| c.dim(2))
                    .transpose()?
                    .unwrap_or(0);
                // `[old_cache ++ x]` is the visible input history; left-pad the still-missing
                // `want - old_n` frames with zeros (first chunks, before the cache has warmed).
                let cat_cx = match &old_cache {
                    Some(cache) => Tensor::cat(&[cache, x], 2)?,
                    None => x.clone(),
                };
                let xpad = if want > old_n {
                    cat_cx.pad_with_zeros(2, want - old_n, 0)?
                } else {
                    cat_cx.clone()
                };
                // Update the cache to the last `want` frames of the input history.
                let hn = cat_cx.dim(2)?;
                let keep = want.min(hn);
                *candle_gen::lock_recover(&self.cache) =
                    Some(cat_cx.narrow(2, hn - keep, keep)?.contiguous()?);
                xpad
            } else {
                x.pad_with_zeros(2, want, 0)?
            }
        } else {
            x.clone()
        };
        let xpad_t = xpad.dim(2)?;
        debug_assert_eq!(
            xpad_t,
            t + self.kd - 1,
            "causal pad must yield t+(kD-1) frames"
        );
        let mut acc: Option<Tensor> = None;
        for kd in 0..self.kd {
            // Tap weight W[:, :, kd] → [O, I, kH, kW].
            let wk = self.weight.narrow(2, kd, 1)?.squeeze(2)?.contiguous()?;
            // The T frames this tap convolves: x_pad[:, :, kd : kd+T].
            let frames = xpad.narrow(2, kd, t)?;
            // Merge (B, T) into the conv2d batch axis: [B, C, T, H, W] → [B*T, C, H, W].
            let merged = frames
                .permute((0, 2, 1, 3, 4))?
                .reshape((b * t, c, h, w))?
                .contiguous()?;
            // Chunked so each conv2d's im2col stays under candle's CUDA overflow band (see
            // [`chunked_conv2d`]). At the 1280×720 z16 `conv_out` (96-ch 3×3) a single output frame's
            // im2col is ~796M elems (and the merged `B·T` batch is up to 4 after the ×4 temporal upsample,
            // ~3.2B total) → the row/batch split brings it under budget; low-res stays a single pass.
            let y = chunked_conv2d(&merged, &wk, self.spatial_pad, 1)?;
            acc = Some(match acc {
                Some(a) => (a + y)?,
                None => y,
            });
        }
        let y = acc.expect("kD >= 1");
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
    use std::collections::HashMap;

    use super::*;
    use candle_gen::candle_core::{DType, Device};

    /// The streaming `feat_cache` path (one frame at a time) must be bit-equivalent to the single
    /// pass over the whole clip — the causal-conv identity sc-5176 relies on for VAE-decode parity.
    #[test]
    fn streaming_matches_single_pass() -> Result<()> {
        let dev = Device::Cpu;
        let (o, i, kd) = (5usize, 3usize, 3usize);
        let conv = CausalConv3d {
            weight: Tensor::randn(0f32, 1.0, (o, i, kd, 3, 3), &dev)?,
            bias: Tensor::randn(0f32, 1.0, o, &dev)?.reshape((1, o, 1, 1, 1))?,
            kd,
            spatial_pad: 1,
            cache: Mutex::new(None),
        };
        let t = 7usize;
        let x = Tensor::randn(0f32, 1.0, (1, i, t, 4, 4), &dev)?;

        let full = conv.forward(&x, &Ctx::single_pass())?;

        conv.reset_cache();
        let mut chunks = Vec::with_capacity(t);
        for f in 0..t {
            let xf = x.narrow(2, f, 1)?.contiguous()?;
            chunks.push(conv.forward(&xf, &Ctx::streaming(f == 0))?);
        }
        let streamed = Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 2)?;

        assert_eq!(full.dims(), streamed.dims());
        let max_diff = (full - streamed)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        assert!(
            max_diff < 1e-5,
            "streaming vs single-pass max abs diff = {max_diff}"
        );
        Ok(())
    }

    /// The im2col chunker ([`chunked_conv2d_budgeted`]) is bit-identical to a plain `conv2d` across every
    /// kernel/pad/stride the Wan VAE uses — the `CausalConv3d` 3×3 "same" tap (pad 1, stride 1), the z16
    /// encoder `SpatialDown` (pad 0, stride 2; its asymmetric pad is applied by the caller), and the 1×1
    /// taps (pad 0, stride 1) — with budgets that force each internal path: the single-frame output-row
    /// fallback, the merged-batch split, and the at-budget single pass (sc-12773 / the SeedVR2 sc-10083
    /// pattern). This is the mutation witness: an off-by-one in the row/batch stitch craters `max_err`.
    #[test]
    fn chunked_conv2d_matches_plain_conv2d() -> Result<()> {
        let dev = Device::Cpu;
        let (o, i) = (5usize, 3usize);
        for (kh, kw, padding, stride) in [
            (3usize, 3usize, 1usize, 1usize), // CausalConv3d 3×3 "same"
            (3, 3, 0, 2), // z16 encoder SpatialDown (caller pre-pads asymmetrically)
            (1, 1, 0, 1), // 1×1 taps (post_quant / mid-attn / temporal down)
        ] {
            let w = Tensor::randn(0f32, 1.0, (o, i, kh, kw), &dev)?;
            // Several merged-batch items (B·T) and a tall-enough H to split into multiple row bands.
            let x = Tensor::randn(0f32, 1.0, (4usize, i, 17, 13), &dev)?;
            let reference = x.conv2d(&w, padding, stride, 1, 1)?;
            // 64 → a single frame busts the budget ⇒ row bands; 1000 → the 1×1 shape splits the batch;
            // usize::MAX → a single un-chunked pass.
            for budget in [64usize, 1000, usize::MAX] {
                let got = chunked_conv2d_budgeted(&x, &w, padding, stride, budget)?;
                assert_eq!(
                    got.dims(),
                    reference.dims(),
                    "shape (kh={kh},stride={stride},budget={budget})"
                );
                let gv = got.flatten_all()?.to_vec1::<f32>()?;
                let rv = reference.flatten_all()?.to_vec1::<f32>()?;
                let max_err = gv
                    .iter()
                    .zip(&rv)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    max_err < 1e-4,
                    "chunked conv2d diverged (kh={kh},kw={kw},pad={padding},stride={stride},budget={budget}): {max_err}"
                );
            }
        }
        Ok(())
    }

    /// sc-12818: the A14B loads its z16 VAE at **bf16** (the decode-floor win). A `CausalConv3d` built
    /// from a bf16 [`VarBuilder`] must carry bf16 weights — `VarBuilder::get` casts each tensor to the
    /// builder's dtype. Loading only reshapes (no matmul), so this exercises bf16 *storage* on the CPU
    /// backend, which has bf16 tensors even though it lacks a bf16 matmul (the bf16 forward is CUDA-only).
    #[test]
    fn causal_conv3d_loads_weights_at_varbuilder_dtype() -> Result<()> {
        let dev = Device::Cpu;
        for dt in [DType::F32, DType::BF16] {
            let tensors = HashMap::from([
                (
                    "conv.weight".to_owned(),
                    Tensor::zeros((5, 3, 3, 3, 3), dt, &dev)?,
                ),
                ("conv.bias".to_owned(), Tensor::zeros(5, dt, &dev)?),
            ]);
            let vb = candle_gen::candle_nn::VarBuilder::from_tensors(tensors, dt, &dev);
            let conv = CausalConv3d::load(3, 5, (3, 3, 3), vb.pp("conv"))?;
            assert_eq!(
                conv.weight.dtype(),
                dt,
                "CausalConv3d weight must load at the VarBuilder dtype ({dt:?})"
            );
            assert_eq!(
                conv.bias.dtype(),
                dt,
                "bias must load at the VarBuilder dtype"
            );
        }
        Ok(())
    }

    /// At or below [`IM2COL_BUDGET`] `chunked_conv2d` must run a *single un-chunked* `conv2d` — the low-res
    /// VAE decode path is byte-for-byte unchanged (the whole point of budgeting: only the hi-res convs
    /// that would overflow candle's im2col get chunked).
    #[test]
    fn chunked_conv2d_below_budget_is_single_pass() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1.0, (4usize, 3usize, 3, 3), &dev)?;
        let x = Tensor::randn(0f32, 1.0, (2usize, 3usize, 8, 8), &dev)?; // tiny ⇒ far under 128M
        let reference = x.conv2d(&w, 1, 1, 1, 1)?;
        let got = chunked_conv2d(&x, &w, 1, 1)?;
        assert_eq!(got.dims(), reference.dims());
        let gv = got.flatten_all()?.to_vec1::<f32>()?;
        let rv = reference.flatten_all()?.to_vec1::<f32>()?;
        // Same code path (one conv2d) ⇒ bit-identical, not merely close.
        assert_eq!(
            gv, rv,
            "at/below budget the chunker must equal a plain conv2d exactly"
        );
        Ok(())
    }
}
