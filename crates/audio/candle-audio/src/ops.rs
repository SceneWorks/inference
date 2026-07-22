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
}
