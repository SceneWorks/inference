//! The BigVGAN anti-aliased periodic activation (`dac_activations.py` + `dac_alias_free_*.py`).
//!
//! `Activation1d` sandwiches a [`SnakeBeta`] between a 2× Kaiser-sinc [`UpSample1d`] and a matching
//! [`LowPassFilter1d`] (the reference's `DownSample1d` is a thin wrapper around exactly that), which
//! is what keeps `sin²(αx)` from folding harmonics back into the band. The
//! MiniMax-H3 audio decoder runs 127 of them (21 AMP blocks × 6, plus `activation_post`), so an
//! error here is invisible per-sample and deafening after 800× upsampling.
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
//! Everything here works in MLX-native **NLC** (`[B, T, C]`) layout; the reference's NCL tensors are
//! transposed once at the decode boundary rather than around every op.

use mlx_rs::ops::{add, broadcast_to, concatenate_axis, conv1d, divide, multiply};
use mlx_rs::Array;

use mlx_gen::{Error, Result};

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
fn kaiser_window(n: i32, beta: f64) -> Vec<f64> {
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
pub fn kaiser_sinc_filter1d(cutoff: f64, half_width: f64, kernel_size: i32) -> Result<Array> {
    if kernel_size < 1 {
        return Err(Error::Msg(format!(
            "minimax-h3 kaiser filter: kernel_size {kernel_size} must be positive"
        )));
    }
    if cutoff <= 0.0 || cutoff > 0.5 {
        // The reference's `cutoff == 0` arm references an undefined name and would raise; a cutoff
        // above 0.5 is above Nyquist. Neither is reachable from this model's configuration.
        return Err(Error::Msg(format!(
            "minimax-h3 kaiser filter: cutoff {cutoff} must be in (0, 0.5]"
        )));
    }
    let even = kernel_size % 2 == 0;
    let half_size = kernel_size / 2;

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
        (0..kernel_size).map(|t| (t - half_size) as f64).collect()
    };
    let mut taps: Vec<f64> = time
        .iter()
        .zip(window.iter())
        .map(|(t, w)| 2.0 * cutoff * w * sinc(2.0 * cutoff * t))
        .collect();
    // Normalize to unit sum, or a constant input leaks a scaled DC component through the filter.
    let sum: f64 = taps.iter().sum();
    if sum == 0.0 {
        return Err(Error::Msg(
            "minimax-h3 kaiser filter: taps sum to zero; the filter cannot be normalized".into(),
        ));
    }
    for tap in &mut taps {
        *tap /= sum;
    }
    let taps: Vec<f32> = taps.into_iter().map(|t| t as f32).collect();
    Ok(Array::from_slice(&taps, &[1, 1, kernel_size]))
}

/// A contiguous index range as an `Array`, for `take_axis`.
fn range_idx(lo: i32, hi: i32) -> Array {
    let idx: Vec<i32> = (lo..hi).collect();
    Array::from_slice(&idx, &[(hi - lo).max(0)])
}

/// Edge-value ("replicate") padding along the time axis of an NLC `[B, T, C]` tensor.
fn replicate_pad(x: &Array, left: i32, right: i32) -> Result<Array> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let s = x.shape();
    if s.len() != 3 {
        return Err(Error::Msg(format!(
            "minimax-h3 replicate_pad: expected [B, T, C], got {s:?}"
        )));
    }
    let (b, t, c) = (s[0], s[1], s[2]);
    let mut parts: Vec<Array> = Vec::new();
    if left > 0 {
        parts.push(broadcast_to(
            x.take_axis(range_idx(0, 1), 1)?,
            &[b, left, c],
        )?);
    }
    parts.push(x.clone());
    if right > 0 {
        parts.push(broadcast_to(
            x.take_axis(range_idx(t - 1, t), 1)?,
            &[b, right, c],
        )?);
    }
    let refs: Vec<&Array> = parts.iter().collect();
    Ok(concatenate_axis(&refs, 1)?)
}

/// Zero-insert an NLC tensor along time: `x[j]` lands at `j·ratio`, the `ratio − 1` slots after it
/// are zero. `[B, L, C]` → `[B, L·ratio, C]`.
fn zero_insert(x: &Array, ratio: i32) -> Result<Array> {
    if ratio == 1 {
        return Ok(x.clone());
    }
    let s = x.shape();
    let (b, l, c) = (s[0], s[1], s[2]);
    let kept = x.reshape(&[b, l, 1, c])?;
    let zeros = Array::zeros::<f32>(&[b, l, ratio - 1, c])?.as_dtype(x.dtype())?;
    Ok(concatenate_axis(&[&kept, &zeros], 2)?.reshape(&[b, l * ratio, c])?)
}

/// A transposed 1-D convolution, computed as **zero-insert + forward convolution**.
///
/// MLX's `conv_transpose1d` evaluates in reduced precision on Metal — a 12-tap depthwise upsample
/// of a unit impulse comes back quantized to f16 (`0.4432098` → `0.4431152`, ~2e-4 relative),
/// and this decoder applies 127 of them plus 7 wide upsample stages, so the error compounds well
/// past the parity floor. The identical zero-insert formulation through `conv1d` reproduces the
/// reference bit-for-bit, at the cost of multiplying by the inserted zeros.
///
/// `weight` must already be in MLX's `[C_out, K, C_in]` layout **with the kernel axis reversed**
/// (see [`flip_kernel`]); reversing at load keeps it off the hot path.
pub(crate) fn transposed_conv1d(
    x: &Array,
    weight: &Array,
    stride: i32,
    padding: i32,
) -> Result<Array> {
    let l = x.shape()[1];
    let taps = weight.shape()[1];
    let upsampled = zero_insert(x, stride)?;
    // 'full' cross-correlation: K−1 of padding on each side.
    let full = conv1d(&upsampled, weight, 1, taps - 1, 1, 1)?;
    // `conv_transpose1d`'s length is `(L−1)·stride + K`; the full convolution is `stride − 1`
    // longer because zero-insert leaves trailing zeros past the last real sample.
    let keep = (l - 1) * stride + taps;
    let (lo, hi) = (padding, keep - padding);
    if lo < 0 || hi > full.shape()[1] || lo >= hi {
        return Err(Error::Msg(format!(
            "minimax-h3 transposed conv: padding {padding} leaves nothing of a {keep}-sample \
             result"
        )));
    }
    Ok(full.take_axis(range_idx(lo, hi), 1)?)
}

/// Reverse the kernel axis of an MLX `[C_out, K, C_in]` weight.
pub(crate) fn flip_kernel(w: &Array) -> Result<Array> {
    let taps = w.shape()[1];
    let idx: Vec<i32> = (0..taps).rev().collect();
    Ok(w.take_axis(Array::from_slice(&idx, &[taps]), 1)?)
}

/// Apply a `[1, 1, taps]` kernel independently to every channel of an NLC tensor.
///
/// Folding the channels into the batch keeps this a plain single-channel convolution rather than a
/// grouped one, which is the shape the rest of `mlx-gen`'s vocoder code already uses.
fn depthwise(x: &Array, filter: &Array, stride: i32) -> Result<Array> {
    let taps = *filter
        .shape()
        .last()
        .ok_or_else(|| Error::Msg("minimax-h3 depthwise: zero-rank filter".into()))?;
    let s = x.shape();
    let (b, t, c) = (s[0], s[1], s[2]);
    let flat = x
        .transpose_axes(&[0, 2, 1])? // (B, C, T)
        .reshape(&[b * c, 1, t])?
        .transpose_axes(&[0, 2, 1])?; // (B·C, T, 1)
    let kernel = filter.reshape(&[1, taps, 1])?;
    let out = conv1d(&flat, &kernel, stride, 0, 1, 1)?;
    let t_out = out.shape()[1];
    Ok(out.reshape(&[b, c, t_out])?.transpose_axes(&[0, 2, 1])?)
}

/// The depthwise counterpart of [`transposed_conv1d`]: one `[1, 1, taps]` filter per channel.
fn depthwise_transposed(x: &Array, filter_flipped: &Array, stride: i32) -> Result<Array> {
    let s = x.shape();
    let (b, t, c) = (s[0], s[1], s[2]);
    let flat = x
        .transpose_axes(&[0, 2, 1])?
        .reshape(&[b * c, 1, t])?
        .transpose_axes(&[0, 2, 1])?; // (B·C, T, 1)
    let out = transposed_conv1d(&flat, filter_flipped, stride, 0)?;
    let t_out = out.shape()[1];
    Ok(out.reshape(&[b, c, t_out])?.transpose_axes(&[0, 2, 1])?)
}

/// `dac_alias_free_filter.py::LowPassFilter1d` — replicate-pad, then a strided depthwise low-pass.
#[derive(Debug, Clone)]
pub struct LowPassFilter1d {
    filter: Array,
    stride: i32,
    pad_left: i32,
    pad_right: i32,
}

impl LowPassFilter1d {
    /// Build from an already-loaded `[1, 1, taps]` filter (the checkpoint stores it as a buffer).
    pub fn from_filter(filter: Array, stride: i32) -> Result<Self> {
        let taps = *filter
            .shape()
            .last()
            .ok_or_else(|| Error::Msg("minimax-h3 low-pass: zero-rank filter".into()))?;
        let even = i32::from(taps % 2 == 0);
        Ok(Self {
            filter,
            stride,
            pad_left: taps / 2 - even,
            pad_right: taps / 2,
        })
    }

    /// `[B, T, C]` → `[B, ceil(T / stride), C]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let padded = replicate_pad(x, self.pad_left, self.pad_right)?;
        depthwise(&padded, &self.filter, self.stride)
    }
}

/// `dac_alias_free_resample.py::UpSample1d` — replicate-pad, zero-insert transposed conv scaled by
/// the ratio, then an asymmetric trim.
#[derive(Debug, Clone)]
pub struct UpSample1d {
    /// `[1, taps, 1]`, kernel axis reversed — see [`transposed_conv1d`].
    kernel: Array,
    ratio: i32,
    pad: i32,
    pad_left: i32,
    pad_right: i32,
}

impl UpSample1d {
    /// Build from an already-loaded `[1, 1, taps]` filter.
    pub fn from_filter(filter: Array, ratio: i32) -> Result<Self> {
        let taps = *filter
            .shape()
            .last()
            .ok_or_else(|| Error::Msg("minimax-h3 upsample: zero-rank filter".into()))?;
        if ratio < 1 {
            return Err(Error::Msg(format!(
                "minimax-h3 upsample: ratio {ratio} must be positive"
            )));
        }
        let pad = taps / ratio - 1;
        if pad < 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 upsample: {taps} taps are too few for ratio {ratio}"
            )));
        }
        Ok(Self {
            kernel: flip_kernel(&filter.reshape(&[1, taps, 1])?)?,
            ratio,
            pad,
            pad_left: pad * ratio + (taps - ratio) / 2,
            pad_right: pad * ratio + (taps - ratio + 1) / 2,
        })
    }

    /// `[B, T, C]` → `[B, T·ratio, C]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let padded = replicate_pad(x, self.pad, self.pad)?;
        let up = depthwise_transposed(&padded, &self.kernel, self.ratio)?;
        let up = multiply(&up, Array::from_f32(self.ratio as f32))?;
        let total = up.shape()[1];
        if self.pad_left + self.pad_right >= total {
            return Err(Error::Msg(format!(
                "minimax-h3 upsample: trim {}+{} exceeds the {total}-sample result",
                self.pad_left, self.pad_right
            )));
        }
        Ok(up.take_axis(range_idx(self.pad_left, total - self.pad_right), 1)?)
    }
}

/// `x + sin²(α·x) / (β + 1e-9)`, with `α`/`β` exponentiated when `logscale`.
///
/// The `1e-9` guard is the reference's; BigVGAN forks that use `1e-6` shift the result by ~1e-6
/// relative per activation, which compounds over 127 of them.
#[derive(Debug, Clone)]
pub struct SnakeBeta {
    alpha: Array,
    beta: Array,
    logscale: bool,
}

impl SnakeBeta {
    /// Build from `[C]` alpha/beta vectors.
    pub fn new(alpha: Array, beta: Array, logscale: bool) -> Result<Self> {
        if alpha.shape() != beta.shape() {
            return Err(Error::Msg(format!(
                "minimax-h3 snakebeta: alpha {:?} and beta {:?} differ in shape",
                alpha.shape(),
                beta.shape()
            )));
        }
        Ok(Self {
            alpha,
            beta,
            logscale,
        })
    }

    /// `[B, T, C]` → `[B, T, C]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let c = *x
            .shape()
            .last()
            .ok_or_else(|| Error::Msg("minimax-h3 snakebeta: zero-rank input".into()))?;
        let n = self.alpha.size() as i32;
        if n != c {
            return Err(Error::Msg(format!(
                "minimax-h3 snakebeta: {n} channels of alpha for a {c}-channel input"
            )));
        }
        let (alpha, beta) = if self.logscale {
            (self.alpha.exp()?, self.beta.exp()?)
        } else {
            (self.alpha.clone(), self.beta.clone())
        };
        let alpha = alpha.reshape(&[1, 1, c])?.as_dtype(x.dtype())?;
        let beta = beta.reshape(&[1, 1, c])?.as_dtype(x.dtype())?;
        let s = multiply(&alpha, x)?.sin()?;
        // `sin(αx) ** 2` as a product, not `power(·, 2)`: MLX's `power` is not bit-identical to a
        // multiply, and the gap compounds across the 127 activations in this decoder.
        let num = multiply(&s, &s)?;
        let den = add(&beta, Array::from_f32(1e-9))?;
        Ok(add(x, &divide(&num, &den)?)?)
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

    /// `[B, T, C]` → `[B, T, C]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let up = self.up.forward(x)?;
        let activated = self.act.forward(&up)?;
        self.down.forward(&activated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn taps(a: &Array) -> Vec<f32> {
        a.as_slice::<f32>().to_vec()
    }

    /// Deterministic host-side data — MLX's RNG needs a GPU stream that not every test worker has.
    fn spread(shape: &[i32]) -> Array {
        let n: i32 = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.41).sin() * 1.7).collect();
        Array::from_slice(&vals, shape)
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
        let f = kaiser_sinc_filter1d(0.25, 0.3, 12).unwrap();
        assert_eq!(f.shape(), &[1, 1, 12]);
        let t = taps(&f);
        let sum: f32 = t.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "taps sum to {sum}");
        for i in 0..6 {
            assert!((t[i] - t[11 - i]).abs() < 1e-7, "taps are symmetric");
        }
        // Odd kernel sizes take the other `time` branch and are still normalized.
        let odd = kaiser_sinc_filter1d(0.25, 0.3, 11).unwrap();
        let sum: f32 = taps(&odd).iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    /// The three `beta` branches are distinguishable, so a port that implemented only the shipped
    /// one would produce different taps here.
    #[test]
    fn beta_branches_are_distinct() {
        // half_size 6 -> A ≈ 51.0 > 50.
        let big = kaiser_sinc_filter1d(0.25, 0.3, 12).unwrap();
        // half_size 2 -> A ≈ 16.6 < 21 -> beta 0 (a rectangular window).
        let none = kaiser_sinc_filter1d(0.25, 0.3, 4).unwrap();
        // half_size 3 -> A ≈ 25.2 -> the middle branch.
        let mid = kaiser_sinc_filter1d(0.25, 0.3, 6).unwrap();
        for f in [&big, &none, &mid] {
            let sum: f32 = taps(f).iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
        // A beta-0 window is flat, so its taps are a pure normalized sinc: the outermost tap is a
        // larger fraction of the peak than the Kaiser-tapered case at the same length.
        let flat = taps(&none);
        assert!(flat[0] / flat[1] > 0.1, "beta=0 leaves the edges untapered");
    }

    #[test]
    fn out_of_range_filter_arguments_are_rejected() {
        assert!(kaiser_sinc_filter1d(0.0, 0.3, 12).is_err());
        assert!(kaiser_sinc_filter1d(0.6, 0.3, 12).is_err());
        assert!(kaiser_sinc_filter1d(0.25, 0.3, 0).is_err());
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
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2, 1]);
        // MLX `[C_out, K, C_in]` in the natural (unflipped) order, as a torch weight would be.
        let kernel = Array::from_slice(&[1.0f32, 10.0, 100.0], &[1, 3, 1]);
        let flipped = flip_kernel(&kernel).unwrap();
        assert_eq!(
            flipped.reshape(&[3]).unwrap().as_slice::<f32>(),
            &[100.0, 10.0, 1.0],
            "flip_kernel must reverse the KERNEL axis"
        );

        let y = transposed_conv1d(&x, &flipped, 2, 0).unwrap();
        assert_eq!(y.shape(), &[1, 5, 1]);
        assert_eq!(
            y.reshape(&[5]).unwrap().as_slice::<f32>(),
            &[1.0, 10.0, 102.0, 20.0, 200.0]
        );

        // `padding` crops symmetrically, exactly as `nn.ConvTranspose1d(padding=..)` does.
        let cropped = transposed_conv1d(&x, &flipped, 2, 1).unwrap();
        assert_eq!(cropped.shape(), &[1, 3, 1]);
        assert_eq!(
            cropped.reshape(&[3]).unwrap().as_slice::<f32>(),
            &[10.0, 102.0, 20.0]
        );

        // Feeding the UNflipped kernel gives a different answer, so the flip is load-bearing.
        let wrong = transposed_conv1d(&x, &kernel, 2, 0).unwrap();
        assert_ne!(
            wrong.reshape(&[5]).unwrap().as_slice::<f32>(),
            &[1.0, 10.0, 102.0, 20.0, 200.0]
        );

        // Stride 1 degenerates to a plain full convolution.
        let unit = transposed_conv1d(&x, &flipped, 1, 0).unwrap();
        assert_eq!(unit.shape(), &[1, 4, 1]);
        assert_eq!(
            unit.reshape(&[4]).unwrap().as_slice::<f32>(),
            &[1.0, 12.0, 120.0, 200.0]
        );
    }

    /// Zero-insert places each sample at `j·ratio` and leaves the rest zero.
    #[test]
    fn zero_insert_strides_the_samples() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let up = zero_insert(&x, 3).unwrap();
        assert_eq!(up.shape(), &[1, 6, 2]);
        assert_eq!(
            up.as_slice::<f32>(),
            &[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(zero_insert(&x, 1).unwrap().shape(), x.shape());
    }

    /// Replicate padding repeats the edge samples per channel, not zeros.
    #[test]
    fn replicate_pad_repeats_the_edges() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 3, 2]);
        let p = replicate_pad(&x, 2, 1).unwrap();
        assert_eq!(p.shape(), &[1, 6, 2]);
        assert_eq!(
            p.as_slice::<f32>(),
            &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0]
        );
        assert_eq!(replicate_pad(&x, 0, 0).unwrap().shape(), x.shape());
    }

    /// A 2× round trip preserves length and, in the interior, a slowly-varying signal.
    #[test]
    fn upsample_downsample_round_trip_preserves_shape_and_dc() {
        let filter = kaiser_sinc_filter1d(0.25, 0.3, 12).unwrap();
        let up = UpSample1d::from_filter(filter.clone(), 2).unwrap();
        let down = LowPassFilter1d::from_filter(filter, 2).unwrap();
        let x = spread(&[2, 21, 3]);
        let u = up.forward(&x).unwrap();
        assert_eq!(u.shape(), &[2, 42, 3]);
        let d = down.forward(&u).unwrap();
        assert_eq!(d.shape(), x.shape());

        // A constant signal survives the round trip: unit-sum taps and replicate padding.
        let ones = Array::ones::<f32>(&[1, 16, 2]).unwrap();
        let back = down.forward(&up.forward(&ones).unwrap()).unwrap();
        let err: f32 = mlx_rs::ops::subtract(&back, &ones)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item();
        assert!(err < 1e-5, "DC gain drifted by {err:.3e}");
    }

    /// `snake_logscale` is not cosmetic: the two modes disagree, and the identity `α=β=0`
    /// (log-scale) reduces to `x + sin²(x)`.
    #[test]
    fn snakebeta_log_scale_is_load_bearing() {
        let x = spread(&[1, 5, 3]);
        let alpha = Array::from_slice(&[0.3f32, -0.2, 0.7], &[3]);
        let beta = Array::from_slice(&[-0.1f32, 0.4, 0.2], &[3]);
        let log = SnakeBeta::new(alpha.clone(), beta.clone(), true)
            .unwrap()
            .forward(&x)
            .unwrap();
        let linear = SnakeBeta::new(alpha, beta, false)
            .unwrap()
            .forward(&x)
            .unwrap();
        let gap: f32 = mlx_rs::ops::subtract(&log, &linear)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item();
        assert!(gap > 1e-2, "log and linear scale agreed (gap {gap:.3e})");

        let zeros = Array::zeros::<f32>(&[3]).unwrap();
        let unit = SnakeBeta::new(zeros.clone(), zeros, true)
            .unwrap()
            .forward(&x)
            .unwrap();
        let sin = x.sin().unwrap();
        let want = add(&x, multiply(&sin, &sin).unwrap()).unwrap();
        let err: f32 = mlx_rs::ops::subtract(&unit, &want)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item();
        assert!(err < 1e-5, "exp(0)=1 identity drifted by {err:.3e}");
    }

    #[test]
    fn snakebeta_rejects_mismatched_widths() {
        let x = spread(&[1, 4, 3]);
        let a = Array::from_slice(&[0.1f32, 0.2], &[2]);
        let b = Array::from_slice(&[0.1f32, 0.2], &[2]);
        // Asserted by MESSAGE (sc-19488): delete the channel guard and `alpha.reshape(&[1, 1, 3])`
        // wants 3 elements from a 2-element alpha, so it errors anyway.
        let msg = SnakeBeta::new(a, b, true)
            .unwrap()
            .forward(&x)
            .expect_err("2 channels of alpha cannot serve a 3-channel input")
            .to_string();
        assert!(
            msg.contains("minimax-h3 snakebeta:") && msg.contains("channels of alpha"),
            "the channel-count guard must be what rejects this, not the reshape below it: {msg}"
        );
        let a = Array::from_slice(&[0.1f32, 0.2], &[2]);
        let b = Array::from_slice(&[0.1f32, 0.2, 0.3], &[3]);
        assert!(SnakeBeta::new(a, b, true).is_err());
    }
}
