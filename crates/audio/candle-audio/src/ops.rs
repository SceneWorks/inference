//! Tensor ops shared across the candle audio providers that candle-core's GPU backends
//! leave unimplemented, expressed with primitives every backend supports.
//!
//! candle-core does not implement `Tensor::upsample_nearest1d` on its **CUDA** (sc-13886)
//! or **Metal** (sc-13691, a hard `bail!` at `metal_backend/mod.rs`) backends — only on CPU.
//! Two audio providers hit that gap: Kokoro's iSTFT-Net vocoder (`AdainResBlk1d` ×2 time
//! upsample) and Chatterbox's S3Gen flow encoder (`Upsample1D`), so both are stuck CPU-only
//! on GPU platforms. Both upsample a `[B, C, T]` time axis by an **exact integer factor**, and
//! for an exact factor nearest-neighbour upsampling is pure repetition — so it can be expressed
//! with `unsqueeze` + `broadcast_as` + `reshape`, which every backend (cpu/cuda/metal)
//! implements, closing both gaps with one shared op. See [`nearest_upsample1d`].

use candle_core::{Result, Tensor};

/// Nearest-neighbour upsample of the time axis of a `[B, C, T]` tensor by an **exact integer
/// factor** `k` → `[B, C, T*k]`, repeating each frame `k` times.
///
/// A backend-agnostic stand-in for `Tensor::upsample_nearest1d(T*k)`, which candle-core leaves
/// unimplemented on its CUDA (sc-13886) and Metal (sc-13691) backends. For an exact integer
/// factor, nearest upsampling collapses to repetition: `out[.., t*k + i] = in[.., t]` for every
/// `i in 0..k`. candle's CPU `upsample_nearest1d` maps `dst[j] = src[min(T-1, (j * (T / (T*k)))
/// as usize)]`, computed in `f64`; for an exact integer factor the scale `1/k` truncates to the
/// integer floor `j/k` (and the `min` clamp never binds), yielding that same repetition. So
/// `unsqueeze` + `broadcast_as` + `reshape` (all pure data movement, no arithmetic — implemented
/// on every backend) reproduces it **bit-for-bit** on CPU. That bit-identity keeps the macOS
/// exact-hash regression fixture (`kokoro_regression_fixture`) valid: the fix changes which ops
/// run, not the samples produced.
///
/// `k` must be `>= 1` (the audio call sites pass `2`); `k == 1` is the identity.
pub fn nearest_upsample1d(x: &Tensor, k: usize) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    // [B, C, T] -> [B, C, T, 1] -> broadcast [B, C, T, k] -> merge to [B, C, T*k].
    // `reshape` on the (non-contiguous) broadcast view materializes it via a strided copy,
    // which the CUDA/Metal backends do support — unlike `upsample_nearest1d` itself.
    x.unsqueeze(3)?
        .broadcast_as((b, c, t, k))?
        .reshape((b, c, t * k))
}

/// Nearest-neighbour **downsample** of the time axis of a `[B, C, T]` tensor by an **exact integer
/// factor** `k` → `[B, C, T/k]`, keeping the **first** frame of every block of `k`.
///
/// The counterpart of [`nearest_upsample1d`], added for Stable Audio 3's inpaint mask (sc-14548),
/// which is constructed at audio-sample resolution over the adapted sample size and then resized to
/// latent resolution by the model's own `downsampling_ratio` (4096). candle-core has no
/// `downsample_nearest1d` at all — on any backend — so unlike its sibling this is not a
/// backend-portability workaround but the only implementation there is; it lives here rather than in
/// a provider because a mask resize is not model-specific and an ad-hoc stride buried in a provider
/// is exactly the kind of arithmetic that goes untested.
///
/// # Which sample of each block, and why that is the load-bearing choice
///
/// Nearest-neighbour resizing to a *smaller* size is `dst[j] = src[floor(j * T / out)]`, which for
/// an exact integer factor is `dst[j] = src[j * k]` — the **first** element of each block, never the
/// last, the middle, or an average. That is the same rule candle's CPU `upsample_nearest1d`
/// implements in the other direction (and the same rule `torch.nn.functional.interpolate(...,
/// mode="nearest")` implements in both), so the pair round-trips: downsampling an upsampled tensor
/// by the same factor is the identity.
///
/// For a **binary** mask the choice is not cosmetic. Under `dst[j] = src[j*k]` a zeroed audio span
/// `[start, end)` maps to the latent span `[ceil(start/k), ceil(end/k))`; under a
/// last-of-block or any-of-block rule it would map to a different, wider span, silently moving the
/// edit window. That is why this is a named, tested op and not a `narrow`/`reshape` one-liner at the
/// call site.
///
/// `k` must be `>= 1` and must divide `T` exactly; `k == 1` is the identity.
pub fn nearest_downsample1d(x: &Tensor, k: usize) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    if k == 0 {
        candle_core::bail!("nearest_downsample1d factor must be >= 1")
    }
    if t % k != 0 {
        candle_core::bail!("nearest_downsample1d factor {k} does not divide the time axis {t}")
    }
    // [B, C, T] -> [B, C, T/k, k] -> keep index 0 of each block -> [B, C, T/k].
    x.reshape((b, c, t / k, k))?
        .narrow(3, 0, 1)?
        .reshape((b, c, t / k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// The replacement must be **bit-identical** to candle's own CPU `upsample_nearest1d` for
    /// every exact integer factor — that identity is what preserves the macOS regression hash.
    #[test]
    fn matches_candle_upsample_nearest1d_bit_for_bit() {
        let dev = Device::Cpu;
        let (b, c, t) = (2usize, 3usize, 5usize);
        // Distinct, non-default values so a broken repetition/stride shows up (not a zeros no-op).
        let data: Vec<f32> = (0..(b * c * t)).map(|i| i as f32 * 0.5 - 3.0).collect();
        let x = Tensor::from_vec(data, (b, c, t), &dev).unwrap();
        for k in [1usize, 2, 3, 4] {
            let ours = nearest_upsample1d(&x, k).unwrap();
            let reference = x.upsample_nearest1d(t * k).unwrap();
            assert_eq!(ours.dims(), &[b, c, t * k], "k={k} shape");
            assert_eq!(ours.dims(), reference.dims(), "k={k} shape vs candle");
            let ours_v = ours.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let ref_v = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(
                ours_v, ref_v,
                "k={k}: not bit-identical to upsample_nearest1d"
            );
        }
    }

    /// Pin the exact repetition pattern (mutation guard: flipping the reshape to interleave
    /// instead of repeat, or dropping the broadcast, turns this red).
    #[test]
    fn repeats_each_frame_k_times() {
        let dev = Device::Cpu;
        // [1, 1, 3] = [10, 20, 30], ×2 -> [10, 10, 20, 20, 30, 30].
        let x = Tensor::from_vec(vec![10f32, 20.0, 30.0], (1usize, 1usize, 3usize), &dev).unwrap();
        let up = nearest_upsample1d(&x, 2).unwrap();
        assert_eq!(up.dims(), &[1, 1, 6]);
        assert_eq!(
            up.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![10.0, 10.0, 20.0, 20.0, 30.0, 30.0],
        );
    }

    /// Multi-channel, factor 2: each channel repeats independently (no cross-channel bleed).
    #[test]
    fn upsamples_channels_independently() {
        let dev = Device::Cpu;
        // [1, 2, 2]: ch0 = [1, 2], ch1 = [3, 4]; ×2 -> ch0 = [1,1,2,2], ch1 = [3,3,4,4].
        let x =
            Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1usize, 2usize, 2usize), &dev).unwrap();
        let up = nearest_upsample1d(&x, 2).unwrap();
        assert_eq!(up.dims(), &[1, 2, 4]);
        assert_eq!(
            up.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
        );
    }

    /// Pin the exact rule: the **first** frame of each block, per channel.
    ///
    /// Mutation guard for the three plausible alternatives — last-of-block (`narrow(3, k-1, 1)`),
    /// mean-of-block, and any-nonzero — each of which produces a different vector here.
    #[test]
    fn downsample_keeps_the_first_frame_of_each_block() {
        let dev = Device::Cpu;
        // [1, 2, 6]: ch0 = [0..6), ch1 = [10..16). k = 3 -> ch0 = [0, 3], ch1 = [10, 13].
        let data: Vec<f32> = (0..6).chain(10..16).map(|i| i as f32).collect();
        let x = Tensor::from_vec(data, (1usize, 2usize, 6usize), &dev).unwrap();
        let down = nearest_downsample1d(&x, 3).unwrap();
        assert_eq!(down.dims(), &[1, 2, 2]);
        assert_eq!(
            down.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0, 3.0, 10.0, 13.0],
            "nearest downsampling keeps src[j*k], not the block's last element or its mean"
        );
        // k = 1 is the identity.
        let same = nearest_downsample1d(&x, 1).unwrap();
        assert_eq!(
            same.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            x.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    /// A binary mask's zeroed span `[start, end)` must land on `[ceil(start/k), ceil(end/k))`.
    ///
    /// This is the property Stable Audio 3's inpaint window depends on, asserted directly rather
    /// than inferred from the element rule above: a last-of-block or any-nonzero rule widens the
    /// span and moves the edit window without changing any shape.
    #[test]
    fn downsample_maps_a_zeroed_span_to_the_ceiling_span() {
        let dev = Device::Cpu;
        let k = 4usize;
        let t = 40usize;
        for (start, end) in [(0usize, 4usize), (5, 13), (6, 8), (9, 40), (39, 40)] {
            let values: Vec<f32> = (0..t)
                .map(|i| if (start..end).contains(&i) { 0.0 } else { 1.0 })
                .collect();
            let x = Tensor::from_vec(values, (1usize, 1usize, t), &dev).unwrap();
            let down = nearest_downsample1d(&x, k)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let expected_start = start.div_ceil(k);
            let expected_end = end.div_ceil(k);
            let zeros: Vec<usize> = down
                .iter()
                .enumerate()
                .filter(|(_, value)| **value == 0.0)
                .map(|(index, _)| index)
                .collect();
            let want: Vec<usize> = (expected_start..expected_end).collect();
            assert_eq!(
                zeros, want,
                "audio span [{start},{end}) at k={k} must map to [{expected_start},{expected_end})"
            );
        }
    }

    /// The pair round-trips, which is what makes "same nearest rule, opposite direction" a claim
    /// rather than a comment.
    #[test]
    fn downsample_inverts_upsample_for_every_exact_factor() {
        let dev = Device::Cpu;
        let (b, c, t) = (2usize, 3usize, 5usize);
        let data: Vec<f32> = (0..(b * c * t)).map(|i| i as f32 * 0.25 - 1.0).collect();
        let x = Tensor::from_vec(data, (b, c, t), &dev).unwrap();
        for k in [1usize, 2, 3, 7] {
            let round_trip = nearest_downsample1d(&nearest_upsample1d(&x, k).unwrap(), k).unwrap();
            assert_eq!(
                round_trip.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                x.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                "k={k}"
            );
        }
    }

    #[test]
    fn downsample_rejects_a_factor_that_does_not_divide_the_time_axis() {
        let dev = Device::Cpu;
        let x = Tensor::zeros((1usize, 1usize, 7usize), candle_core::DType::F32, &dev).unwrap();
        assert!(nearest_downsample1d(&x, 3).is_err());
        assert!(nearest_downsample1d(&x, 0).is_err());
    }
}
