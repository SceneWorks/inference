//! Small shape helpers shared by the video decoder modules.

use candle_gen::candle_core::Tensor;
use candle_gen::{CandleError, Result};

/// Pack an NCTHW latent into ViT tokens `[B, T·H·W, C]`.
///
/// This is `_pack_tensors_3d(x, patch_size=1, patch_size_t=1)`: with unit patches the reference's
/// eight-way permute reduces to moving the channel axis last, so token order is `(t, h, w)`
/// row-major — matching [`crate::rope::create_token_ids`]'s `ij` mesh.
pub fn pack_tokens(x: &Tensor) -> Result<Tensor> {
    let s = x.dims();
    if s.len() != 5 {
        return Err(CandleError::Msg(format!(
            "minimax-h3 pack_tokens: expected [B, C, T, H, W], got {s:?}"
        )));
    }
    let (b, c, t, h, w) = (s[0], s[1], s[2], s[3], s[4]);
    Ok(x.permute((0, 2, 3, 4, 1))?
        .contiguous()?
        .reshape((b, t * h * w, c))?)
}

/// Unpack ViT patch tokens back to NCTHW video.
///
/// `tokens` is `[B, T·H·W, out_channels · patch_size_t · patch_size²]` whose channel axis is laid
/// out `(c, pt, ph, pw)` row-major; the result is
/// `[B, out_channels, T·patch_size_t, H·patch_size, W·patch_size]` (`_unpack_tensors_3d`).
///
/// A wrong permute still produces the right SHAPE, which is why
/// `unpack_places_the_channel_major_patch_block` pins the actual placement rather than the dims.
pub fn unpack_tokens(
    tokens: &Tensor,
    out_channels: usize,
    patch_size: usize,
    patch_size_t: usize,
    lat_t: usize,
    lat_h: usize,
    lat_w: usize,
) -> Result<Tensor> {
    let s = tokens.dims();
    if s.len() != 3 {
        return Err(CandleError::Msg(format!(
            "minimax-h3 unpack_tokens: expected [B, N, C], got {s:?}"
        )));
    }
    let (b, n, c) = (s[0], s[1], s[2]);
    let expected_n = lat_t * lat_h * lat_w;
    let expected_c = out_channels * patch_size_t * patch_size * patch_size;
    if n != expected_n || c != expected_c {
        return Err(CandleError::Msg(format!(
            "minimax-h3 unpack_tokens: expected [B, {expected_n}, {expected_c}], got {s:?}"
        )));
    }
    Ok(tokens
        .reshape(&[
            b,
            lat_t,
            lat_h,
            lat_w,
            out_channels,
            patch_size_t,
            patch_size,
            patch_size,
        ])?
        .permute(&[0usize, 4, 1, 5, 2, 6, 3, 7][..])?
        .contiguous()?
        .reshape((
            b,
            out_channels,
            lat_t * patch_size_t,
            lat_h * patch_size,
            lat_w * patch_size,
        ))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    fn spread(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).sin() * 1.3).collect();
        Tensor::from_vec(vals, shape, &Device::Cpu).unwrap()
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn pack_then_unpack_round_trips_a_unit_patch() {
        // With patch sizes of 1 the pack/unpack pair is an exact inverse.
        let x = spread(&[1, 3, 2, 3, 4]);
        let tokens = pack_tokens(&x).unwrap();
        assert_eq!(tokens.dims(), &[1, 24, 3]);
        let back = unpack_tokens(&tokens, 3, 1, 1, 2, 3, 4).unwrap();
        assert_eq!(back.dims(), x.dims());
        assert_eq!(flat(&back), flat(&x));
    }

    /// The `(c, pt, ph, pw)` channel layout is what maps a token's features onto its
    /// `patch_size_t × patch_size × patch_size` pixel block.
    #[test]
    fn unpack_places_the_channel_major_patch_block() {
        // One token, 1 channel, 1×2×2 patch -> features [p0, p1, p2, p3] fill the block row-major.
        let tokens =
            Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], (1, 1, 4), &Device::Cpu).unwrap();
        let video = unpack_tokens(&tokens, 1, 2, 1, 1, 1, 1).unwrap();
        assert_eq!(video.dims(), &[1, 1, 1, 2, 2]);
        assert_eq!(flat(&video), vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn pack_tokens_orders_tokens_t_then_h_then_w() {
        // Channel 0 carries a linear ramp over (t, h, w); packing must preserve that order.
        let vals: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let x = Tensor::from_vec(vals.clone(), (1, 1, 2, 2, 3), &Device::Cpu).unwrap();
        let tokens = pack_tokens(&x).unwrap();
        assert_eq!(tokens.dims(), &[1, 12, 1]);
        assert_eq!(flat(&tokens), vals);
    }

    #[test]
    fn shape_errors_are_typed() {
        let x = spread(&[2, 5]);
        assert!(pack_tokens(&x).is_err());
        let tokens = spread(&[1, 4, 3]);
        assert!(unpack_tokens(&tokens, 3, 1, 1, 2, 3, 4).is_err());
    }
}
