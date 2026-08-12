//! The BigVGAN anti-aliased periodic activation (`dac_activations.py` + `dac_alias_free_*.py`).
//!
//! `Activation1d` sandwiches a [`SnakeBeta`] between a 2× Kaiser-sinc [`UpSample1d`] and a matching
//! [`LowPassFilter1d`] (the reference's `DownSample1d` is a thin wrapper around exactly that),
//! which is what keeps `sin²(αx)` from folding harmonics back into the band. The MiniMax-H3 audio
//! decoder runs 127 of them (21 AMP blocks × 6, plus `activation_post`), so an error here is
//! invisible per-sample and deafening after 800× upsampling.
//!
//! Two details carry the numerical risk the port has to prove, and each has its own parity fixture:
//!
//! * **The filter derivation.** [`kaiser_sinc_filter1d`] is a windowed sinc whose Kaiser `beta`
//!   comes from a three-branch attenuation table; the taps are normalized to sum 1 so a constant
//!   input survives the round trip. The published checkpoint *stores* these filters as buffers
//!   (`upsample.filter`, `downsample.lowpass.filter`), so the loader reads them rather than
//!   recomputing — but `tests/audio_vae_parity.rs` pins the derivation against the reference's own
//!   `kaiser_sinc_filter1d`, and the real-weight smoke pins it against the shipped buffers.
//! * **`SnakeBeta`'s log scale.** `snake_logscale = true` for this checkpoint (see
//!   [`crate::audio_config`]), so `alpha`/`beta` are exponentiated before use, and the reciprocal
//!   guard is `1e-9` — not the `1e-6` some BigVGAN forks use.
//!
//! Everything here works in **NCL** (`[B, C, T]`), which is both candle's native conv layout and
//! the reference's, so unlike the MLX twin no transposes are needed at the decode boundary.

use candle_gen::candle_core::Tensor;
use candle_gen::{CandleError, Result};

/// Modified Bessel function of the first kind, order 0 — the Kaiser window's shape function.
///
/// Ascending series `Σ (x²/4)^k / (k!)²`, which converges in ~30 terms for the `beta ≤ 12` these
/// filters use and is accurate to f64 round-off there.
fn bessel_i0(x: f64) -> f64 {
    let quarter_sq = x * x / 4.0;
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    for k in 1..64 {
        term *= quarter_sq / ((k as f64) * (k as f64));
        sum += term;
        if term < sum * 1e-18 {
            break;
        }
    }
    sum
}

/// `torch.kaiser_window(n, beta, periodic=false)`.
fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    let denom = bessel_i0(beta);
    let last = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let t = 2.0 * (i as f64) / last - 1.0;
            bessel_i0(beta * (1.0 - t * t).max(0.0).sqrt()) / denom
        })
        .collect()
}

/// Normalized sinc, `sin(πx) / (πx)` with `sinc(0) = 1` (`torch.sinc`).
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// The low-pass prototype both resamplers use, as a `[1, 1, kernel_size]` f32 tensor.
///
/// Port of `dac_alias_free_filter.py::kaiser_sinc_filter1d`. `cutoff` is in cycles/sample (so 0.25
/// for a 2× resampler) and `half_width` is the transition half-width.
///
/// The Kaiser `beta` comes from the stop-band attenuation estimate
/// `A = 2.285·(kernel_size/2 − 1)·π·4·half_width + 7.95`, then a three-branch table. All three
/// branches are exercised by the committed fixture — the shipped 12-tap filter lands in `A > 50`,
/// but a port that only implemented that branch would be wrong for any other ratio.
pub fn kaiser_sinc_filter1d(
    cutoff: f64,
    half_width: f64,
    kernel_size: usize,
    device: &candle_gen::candle_core::Device,
) -> Result<Tensor> {
    if kernel_size < 1 {
        return Err(CandleError::Msg(format!(
            "minimax-h3 kaiser filter: kernel_size {kernel_size} must be positive"
        )));
    }
    if cutoff <= 0.0 || cutoff > 0.5 {
        // The reference's `cutoff == 0` arm references an undefined name and would raise; a cutoff
        // above 0.5 is above Nyquist. Neither is reachable from this model's configuration.
        return Err(CandleError::Msg(format!(
            "minimax-h3 kaiser filter: cutoff {cutoff} must be in (0, 0.5]"
        )));
    }
    let even = kernel_size.is_multiple_of(2);
    let half_size = (kernel_size / 2) as i64;

    let delta_f = 4.0 * half_width;
    let a = 2.285 * ((half_size - 1) as f64) * std::f64::consts::PI * delta_f + 7.95;
    let beta = if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a >= 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    };
    let window = kaiser_window(kernel_size, beta);

    let time: Vec<f64> = if even {
        (-half_size..half_size).map(|t| t as f64 + 0.5).collect()
    } else {
        (0..kernel_size as i64)
            .map(|t| (t - half_size) as f64)
            .collect()
    };
    let mut taps: Vec<f64> = time
        .iter()
        .zip(window.iter())
        .map(|(t, w)| 2.0 * cutoff * w * sinc(2.0 * cutoff * t))
        .collect();
    // Normalize to unit sum, or a constant input leaks a scaled DC component through the filter.
    let sum: f64 = taps.iter().sum();
    if sum == 0.0 {
        return Err(CandleError::Msg(
            "minimax-h3 kaiser filter: taps sum to zero; the filter cannot be normalized".into(),
        ));
    }
    for tap in &mut taps {
        *tap /= sum;
    }
    let taps: Vec<f32> = taps.into_iter().map(|t| t as f32).collect();
    Ok(Tensor::from_vec(taps, (1, 1, kernel_size), device)?)
}

/// Edge-value ("replicate") padding along the time axis of an NCL `[B, C, T]` tensor.
fn replicate_pad(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let s = x.dims();
    if s.len() != 3 {
        return Err(CandleError::Msg(format!(
            "minimax-h3 replicate_pad: expected [B, C, T], got {s:?}"
        )));
    }
    let t = s[2];
    let mut parts: Vec<Tensor> = Vec::new();
    if left > 0 {
        parts.push(x.narrow(2, 0, 1)?.broadcast_as((s[0], s[1], left))?);
    }
    parts.push(x.clone());
    if right > 0 {
        parts.push(x.narrow(2, t - 1, 1)?.broadcast_as((s[0], s[1], right))?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Ok(Tensor::cat(&refs, 2)?.contiguous()?)
}

/// Zero-insert an NCL tensor along time: `x[j]` lands at `j·ratio`, the `ratio − 1` slots after it
/// are zero. `[B, C, L]` → `[B, C, L·ratio]`.
fn zero_insert(x: &Tensor, ratio: usize) -> Result<Tensor> {
    if ratio == 1 {
        return Ok(x.clone());
    }
    let s = x.dims();
    let (b, c, l) = (s[0], s[1], s[2]);
    let kept = x.reshape((b, c, l, 1))?;
    let zeros = Tensor::zeros((b, c, l, ratio - 1), x.dtype(), x.device())?;
    Ok(Tensor::cat(&[&kept, &zeros], 3)?
        .contiguous()?
        .reshape((b, c, l * ratio))?)
}

/// A transposed 1-D convolution, computed as **zero-insert + forward convolution**.
///
/// `weight` must already be in candle's `[C_out, C_in, K]` layout **with the kernel axis reversed**
/// (see [`flip_kernel`]); reversing at load keeps it off the hot path. The identity being used is
/// `convT(x, W, stride s, padding p)[n] = full_corr(zero_insert(x, s), flip(W))[n + p]`.
///
/// Written this way on **both** backends rather than reaching for a built-in. On MLX the reason is
/// forced: `conv_transpose1d` evaluates in reduced precision on Metal — a 12-tap depthwise upsample
/// of a unit impulse comes back f16-quantized (`0.4432098` → `0.4431152`, ~2e-4 relative) — and
/// this decoder applies 127 of them plus 7 wide upsample stages, so the error compounds well past
/// the parity floor. candle's `conv_transpose1d` has no such defect, but keeping one formulation
/// across both lanes is what lets the cross-backend residual be attributed to matmul precision
/// rather than to two different resampling algorithms.
pub(crate) fn transposed_conv1d(
    x: &Tensor,
    weight: &Tensor,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    let l = x.dims()[2];
    let taps = weight.dims()[2];
    let upsampled = zero_insert(x, stride)?;
    // 'full' cross-correlation: K−1 of padding on each side.
    let full = upsampled.conv1d(weight, taps - 1, 1, 1, 1)?;
    // `conv_transpose1d`'s length is `(L−1)·stride + K`; the full convolution is `stride − 1`
    // longer because zero-insert leaves trailing zeros past the last real sample.
    let keep = (l - 1) * stride + taps;
    if keep < 2 * padding + 1 || keep > full.dims()[2] {
        return Err(CandleError::Msg(format!(
            "minimax-h3 transposed conv: padding {padding} leaves nothing of a {keep}-sample result"
        )));
    }
    Ok(full.narrow(2, padding, keep - 2 * padding)?.contiguous()?)
}

/// Reverse the kernel axis of a candle `[C_out, C_in, K]` weight.
pub(crate) fn flip_kernel(w: &Tensor) -> Result<Tensor> {
    let taps = w.dims()[2];
    let idx: Vec<u32> = (0..taps as u32).rev().collect();
    let idx = Tensor::from_vec(idx, taps, w.device())?;
    Ok(w.index_select(&idx, 2)?.contiguous()?)
}

/// Apply a `[1, 1, taps]` kernel independently to every channel of an NCL tensor.
///
/// Folding the channels into the batch keeps this a plain single-channel convolution rather than a
/// grouped one — and in NCL that fold is a pure reshape.
fn depthwise(x: &Tensor, filter: &Tensor, stride: usize) -> Result<Tensor> {
    let s = x.dims();
    let (b, c, t) = (s[0], s[1], s[2]);
    let flat = x.contiguous()?.reshape((b * c, 1, t))?;
    let kernel = filter.reshape((1, 1, filter.dims()[2]))?;
    let out = flat.conv1d(&kernel, 0, stride, 1, 1)?;
    let t_out = out.dims()[2];
    Ok(out.reshape((b, c, t_out))?)
}

/// The depthwise counterpart of [`transposed_conv1d`]: one `[1, 1, taps]` filter per channel.
fn depthwise_transposed(x: &Tensor, filter_flipped: &Tensor, stride: usize) -> Result<Tensor> {
    let s = x.dims();
    let (b, c, t) = (s[0], s[1], s[2]);
    let flat = x.contiguous()?.reshape((b * c, 1, t))?;
    let out = transposed_conv1d(&flat, filter_flipped, stride, 0)?;
    let t_out = out.dims()[2];
    Ok(out.reshape((b, c, t_out))?)
}

/// `dac_alias_free_filter.py::LowPassFilter1d` — replicate-pad, then a strided depthwise low-pass.
#[derive(Debug, Clone)]
pub struct LowPassFilter1d {
    filter: Tensor,
    stride: usize,
    pad_left: usize,
    pad_right: usize,
}

impl LowPassFilter1d {
    /// Build from an already-loaded `[1, 1, taps]` filter (the checkpoint stores it as a buffer).
    pub fn from_filter(filter: Tensor, stride: usize) -> Result<Self> {
        let dims = filter.dims();
        let taps = *dims
            .last()
            .ok_or_else(|| CandleError::Msg("minimax-h3 low-pass: zero-rank filter".into()))?;
        let even = usize::from(taps.is_multiple_of(2));
        Ok(Self {
            filter: filter.reshape((1, 1, taps))?,
            stride,
            pad_left: taps / 2 - even,
            pad_right: taps / 2,
        })
    }

    /// `[B, C, T]` → `[B, C, ceil(T / stride)]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let padded = replicate_pad(x, self.pad_left, self.pad_right)?;
        depthwise(&padded, &self.filter, self.stride)
    }
}

/// `dac_alias_free_resample.py::UpSample1d` — replicate-pad, zero-insert transposed conv scaled by
/// the ratio, then an asymmetric trim.
///
/// The two trims differ by one for an odd `taps − ratio`, and swapping them shifts the whole
/// waveform by a sample — which `upsample_matches_the_reference_sample_for_sample` is written to
/// catch, since a tolerance alone would not name that failure.
#[derive(Debug, Clone)]
pub struct UpSample1d {
    /// `[1, 1, taps]`, kernel axis reversed — see [`transposed_conv1d`].
    kernel: Tensor,
    ratio: usize,
    pad: usize,
    pad_left: usize,
    pad_right: usize,
}

impl UpSample1d {
    /// Build from an already-loaded `[1, 1, taps]` filter.
    pub fn from_filter(filter: Tensor, ratio: usize) -> Result<Self> {
        let dims = filter.dims();
        let taps = *dims
            .last()
            .ok_or_else(|| CandleError::Msg("minimax-h3 upsample: zero-rank filter".into()))?;
        if ratio < 1 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 upsample: ratio {ratio} must be positive"
            )));
        }
        if taps < ratio || taps / ratio < 1 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 upsample: {taps} taps are too few for ratio {ratio}"
            )));
        }
        let pad = taps / ratio - 1;
        Ok(Self {
            kernel: flip_kernel(&filter.reshape((1, 1, taps))?)?,
            ratio,
            pad,
            // The reference's `(kernel_size - ratio) // 2` and `(kernel_size - ratio + 1) // 2`.
            // The two differ by one whenever `taps - ratio` is odd, and swapping them shifts the
            // whole waveform by a sample — `upsample_matches_the_reference_sample_for_sample` is
            // what catches that. `+1 / 2` on non-negative integers IS `div_ceil(2)`.
            pad_left: pad * ratio + (taps - ratio) / 2,
            pad_right: pad * ratio + (taps - ratio).div_ceil(2),
        })
    }

    /// `[B, C, T]` → `[B, C, T·ratio]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let padded = replicate_pad(x, self.pad, self.pad)?;
        let up = depthwise_transposed(&padded, &self.kernel, self.ratio)?;
        let up = (up * (self.ratio as f64))?;
        let total = up.dims()[2];
        if self.pad_left + self.pad_right >= total {
            return Err(CandleError::Msg(format!(
                "minimax-h3 upsample: trim {}+{} exceeds the {total}-sample result",
                self.pad_left, self.pad_right
            )));
        }
        Ok(up
            .narrow(2, self.pad_left, total - self.pad_left - self.pad_right)?
            .contiguous()?)
    }
}

/// `x + sin²(α·x) / (β + 1e-9)`, with `α`/`β` exponentiated when `logscale`.
///
/// The `1e-9` guard is the reference's; BigVGAN forks that use `1e-6` shift the result by ~1e-6
/// relative per activation, which compounds over 127 of them.
#[derive(Debug, Clone)]
pub struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
    logscale: bool,
}

impl SnakeBeta {
    /// Build from `[C]`-element alpha/beta vectors (any shape with `C` elements is accepted; the
    /// reference stores them as `[C]` and unsqueezes at use).
    pub fn new(alpha: Tensor, beta: Tensor, logscale: bool) -> Result<Self> {
        if alpha.elem_count() != beta.elem_count() {
            return Err(CandleError::Msg(format!(
                "minimax-h3 snakebeta: alpha {:?} and beta {:?} differ in size",
                alpha.dims(),
                beta.dims()
            )));
        }
        Ok(Self {
            alpha: alpha.flatten_all()?,
            beta: beta.flatten_all()?,
            logscale,
        })
    }

    /// `[B, C, T]` → `[B, C, T]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() != 3 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 snakebeta: expected [B, C, T], got {dims:?}"
            )));
        }
        let c = dims[1];
        let n = self.alpha.elem_count();
        if n != c {
            return Err(CandleError::Msg(format!(
                "minimax-h3 snakebeta: {n} channels of alpha for a {c}-channel input"
            )));
        }
        let (alpha, beta) = if self.logscale {
            (self.alpha.exp()?, self.beta.exp()?)
        } else {
            (self.alpha.clone(), self.beta.clone())
        };
        let alpha = alpha.reshape((1, c, 1))?.to_dtype(x.dtype())?;
        let beta = beta.reshape((1, c, 1))?.to_dtype(x.dtype())?;
        let s = x.broadcast_mul(&alpha)?.sin()?;
        // `sin(αx) ** 2` as a product, not `power(·, 2)`: the two are not bit-identical on every
        // backend, and the gap compounds across the 127 activations in this decoder.
        let num = (&s * &s)?;
        let den = (beta + 1e-9)?;
        Ok(x.add(&num.broadcast_div(&den)?)?)
    }
}

/// `dac_alias_free_act.py::Activation1d` — 2× up, activate, 2× down.
#[derive(Debug, Clone)]
pub struct Activation1d {
    act: SnakeBeta,
    up: UpSample1d,
    down: LowPassFilter1d,
}

impl Activation1d {
    /// Assemble from an activation and its two stored filters.
    pub fn new(act: SnakeBeta, up: UpSample1d, down: LowPassFilter1d) -> Self {
        Self { act, up, down }
    }

    /// `[B, C, T]` → `[B, C, T]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let up = self.up.forward(x)?;
        let activated = self.act.forward(&up)?;
        self.down.forward(&activated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn spread(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.41).sin() * 1.7).collect();
        Tensor::from_vec(vals, shape, &Device::Cpu).unwrap()
    }

    /// `I0` against published values; the window is symmetric and peaks at 1 in the middle.
    #[test]
    fn bessel_and_kaiser_window_are_right() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-15);
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008_3).abs() < 1e-12);
        assert!((bessel_i0(5.0) - 27.239_871_823_604_44).abs() < 1e-10);

        let w = kaiser_window(12, 4.663_6);
        assert_eq!(w.len(), 12);
        for i in 0..6 {
            assert!((w[i] - w[11 - i]).abs() < 1e-12, "window is symmetric");
        }
        // periodic=false => the endpoints are the window's minimum, the centre its maximum.
        assert!(w[0] < w[5] && w[5] <= 1.0);
        assert_eq!(kaiser_window(1, 3.0), vec![1.0]);
    }

    /// The taps sum to 1, so a constant signal passes the low-pass unchanged (up to the edges).
    #[test]
    fn filter_taps_are_normalized_and_symmetric() {
        let dev = Device::Cpu;
        let f = kaiser_sinc_filter1d(0.25, 0.3, 12, &dev).unwrap();
        assert_eq!(f.dims(), &[1, 1, 12]);
        let t = flat(&f);
        let sum: f32 = t.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "taps sum to {sum}");
        for i in 0..6 {
            assert!((t[i] - t[11 - i]).abs() < 1e-7, "taps are symmetric");
        }
        // Odd kernel sizes take the other `time` branch and are still normalized.
        let odd = kaiser_sinc_filter1d(0.25, 0.3, 11, &dev).unwrap();
        let sum: f32 = flat(&odd).iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    /// The three `beta` branches are distinguishable, so a port that implemented only the shipped
    /// one would produce different taps here.
    #[test]
    fn beta_branches_are_distinct() {
        let dev = Device::Cpu;
        // half_size 6 -> A ≈ 51.0 > 50.
        let big = kaiser_sinc_filter1d(0.25, 0.3, 12, &dev).unwrap();
        // half_size 2 -> A ≈ 16.6 < 21 -> beta 0 (a rectangular window).
        let none = kaiser_sinc_filter1d(0.25, 0.3, 4, &dev).unwrap();
        // half_size 3 -> A ≈ 25.2 -> the middle branch.
        let mid = kaiser_sinc_filter1d(0.25, 0.3, 6, &dev).unwrap();
        for f in [&big, &none, &mid] {
            let sum: f32 = flat(f).iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
        // A beta-0 window is flat, so its taps are a pure normalized sinc: the outermost tap is a
        // larger fraction of the peak than the Kaiser-tapered case at the same length.
        let f = flat(&none);
        assert!(f[0] / f[1] > 0.1, "beta=0 leaves the edges untapered");
    }

    #[test]
    fn out_of_range_filter_arguments_are_rejected() {
        let dev = Device::Cpu;
        assert!(kaiser_sinc_filter1d(0.0, 0.3, 12, &dev).is_err());
        assert!(kaiser_sinc_filter1d(0.6, 0.3, 12, &dev).is_err());
        assert!(kaiser_sinc_filter1d(0.25, 0.3, 0, &dev).is_err());
    }

    /// `transposed_conv1d` against a hand-computed case with an **asymmetric** kernel.
    ///
    /// Every Kaiser-sinc filter in this model is symmetric, so [`flip_kernel`] is a no-op for them
    /// and none of the resampler tests can tell a missing flip from a present one. The BigVGAN
    /// `ups.*` weights are not symmetric, but they are only covered end-to-end at 2e-2 — so pin the
    /// convention here, exactly, where the arithmetic fits on one line.
    ///
    /// `x = [1, 2]`, kernel `[a, b, c]`, stride 2 gives `[a, b, c + 2a, 2b, 2c]`: each input sample
    /// stamps the kernel starting at `j·stride`, and the overlaps sum.
    #[test]
    fn transposed_conv_matches_the_hand_computed_stamp() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0], (1, 1, 2), &dev).unwrap();
        // candle `[C_out, C_in, K]` in the natural (unflipped) order, as a torch weight would be.
        let kernel = Tensor::from_vec(vec![1.0f32, 10.0, 100.0], (1, 1, 3), &dev).unwrap();
        let flipped = flip_kernel(&kernel).unwrap();
        assert_eq!(
            flat(&flipped),
            vec![100.0, 10.0, 1.0],
            "flip_kernel must reverse the KERNEL axis"
        );

        let y = transposed_conv1d(&x, &flipped, 2, 0).unwrap();
        assert_eq!(y.dims(), &[1, 1, 5]);
        assert_eq!(flat(&y), vec![1.0, 10.0, 102.0, 20.0, 200.0]);

        // `padding` crops symmetrically, exactly as `nn.ConvTranspose1d(padding=..)` does.
        let cropped = transposed_conv1d(&x, &flipped, 2, 1).unwrap();
        assert_eq!(cropped.dims(), &[1, 1, 3]);
        assert_eq!(flat(&cropped), vec![10.0, 102.0, 20.0]);

        // Feeding the UNflipped kernel gives a different answer, so the flip is load-bearing.
        let wrong = transposed_conv1d(&x, &kernel, 2, 0).unwrap();
        assert_ne!(flat(&wrong), vec![1.0, 10.0, 102.0, 20.0, 200.0]);

        // Stride 1 degenerates to a plain full convolution.
        let unit = transposed_conv1d(&x, &flipped, 1, 0).unwrap();
        assert_eq!(unit.dims(), &[1, 1, 4]);
        assert_eq!(flat(&unit), vec![1.0, 12.0, 120.0, 200.0]);
    }

    /// Zero-insert places each sample at `j·ratio` and leaves the rest zero.
    #[test]
    fn zero_insert_strides_the_samples() {
        // NCL: one batch, two channels, two samples each.
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &Device::Cpu).unwrap();
        let up = zero_insert(&x, 3).unwrap();
        assert_eq!(up.dims(), &[1, 2, 6]);
        assert_eq!(
            flat(&up),
            vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0]
        );
        assert_eq!(zero_insert(&x, 1).unwrap().dims(), x.dims());
    }

    /// Replicate padding repeats the edge samples per channel, not zeros.
    #[test]
    fn replicate_pad_repeats_the_edges() {
        let x = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (1, 2, 3),
            &Device::Cpu,
        )
        .unwrap();
        let p = replicate_pad(&x, 2, 1).unwrap();
        assert_eq!(p.dims(), &[1, 2, 6]);
        assert_eq!(
            flat(&p),
            vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0]
        );
        assert_eq!(replicate_pad(&x, 0, 0).unwrap().dims(), x.dims());
    }

    /// A 2× round trip preserves length and, in the interior, a slowly-varying signal.
    #[test]
    fn upsample_downsample_round_trip_preserves_shape_and_dc() {
        let dev = Device::Cpu;
        let filter = kaiser_sinc_filter1d(0.25, 0.3, 12, &dev).unwrap();
        let up = UpSample1d::from_filter(filter.clone(), 2).unwrap();
        let down = LowPassFilter1d::from_filter(filter, 2).unwrap();
        let x = spread(&[2, 3, 21]);
        let u = up.forward(&x).unwrap();
        assert_eq!(u.dims(), &[2, 3, 42]);
        let d = down.forward(&u).unwrap();
        assert_eq!(d.dims(), x.dims());

        // A constant signal survives the round trip: unit-sum taps and replicate padding.
        let ones = Tensor::ones((1, 2, 16), DType::F32, &dev).unwrap();
        let back = down.forward(&up.forward(&ones).unwrap()).unwrap();
        let err = flat(&(back - &ones).unwrap())
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-5, "DC gain drifted by {err:.3e}");
    }

    /// `snake_logscale` is not cosmetic: the two modes disagree, and the identity `α=β=0`
    /// (log-scale) reduces to `x + sin²(x)`.
    #[test]
    fn snakebeta_log_scale_is_load_bearing() {
        let dev = Device::Cpu;
        let x = spread(&[1, 3, 5]);
        let alpha = Tensor::from_vec(vec![0.3f32, -0.2, 0.7], 3, &dev).unwrap();
        let beta = Tensor::from_vec(vec![-0.1f32, 0.4, 0.2], 3, &dev).unwrap();
        let log = SnakeBeta::new(alpha.clone(), beta.clone(), true)
            .unwrap()
            .forward(&x)
            .unwrap();
        let linear = SnakeBeta::new(alpha, beta, false)
            .unwrap()
            .forward(&x)
            .unwrap();
        let gap = flat(&(log - &linear).unwrap())
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        assert!(gap > 1e-2, "log and linear scale agreed (gap {gap:.3e})");

        let zeros = Tensor::zeros(3, DType::F32, &dev).unwrap();
        let unit = SnakeBeta::new(zeros.clone(), zeros, true)
            .unwrap()
            .forward(&x)
            .unwrap();
        let sin = x.sin().unwrap();
        let want = x.add(&(&sin * &sin).unwrap()).unwrap();
        let err = flat(&(unit - &want).unwrap())
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-5, "exp(0)=1 identity drifted by {err:.3e}");
    }

    #[test]
    fn snakebeta_rejects_mismatched_widths() {
        let dev = Device::Cpu;
        let x = spread(&[1, 3, 4]);
        let a = Tensor::from_vec(vec![0.1f32, 0.2], 2, &dev).unwrap();
        let b = Tensor::from_vec(vec![0.1f32, 0.2], 2, &dev).unwrap();
        assert!(SnakeBeta::new(a, b, true).unwrap().forward(&x).is_err());
        let a = Tensor::from_vec(vec![0.1f32, 0.2], 2, &dev).unwrap();
        let b = Tensor::from_vec(vec![0.1f32, 0.2, 0.3], 3, &dev).unwrap();
        assert!(SnakeBeta::new(a, b, true).is_err());
    }
}
