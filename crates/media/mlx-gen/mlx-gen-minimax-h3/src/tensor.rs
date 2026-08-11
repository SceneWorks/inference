//! Small shape helpers shared by the decoder modules.

use mlx_rs::Array;

use mlx_gen::{Error, Result};

/// `x[..., start..end, ...]` along `axis` (half-open), as a materialized copy.
pub fn slice_axis(x: &Array, axis: i32, start: i32, end: i32) -> Result<Array> {
    let rank = x.shape().len() as i32;
    if axis < 0 || axis >= rank {
        return Err(Error::Msg(format!(
            "minimax-h3 slice_axis: axis {axis} out of range for rank {rank}"
        )));
    }
    let len = x.shape()[axis as usize];
    if start < 0 || end > len || start > end {
        return Err(Error::Msg(format!(
            "minimax-h3 slice_axis: [{start}, {end}) out of range for axis {axis} of length {len}"
        )));
    }
    let idx: Vec<i32> = (start..end).collect();
    let idx = Array::from_slice(&idx, &[idx.len() as i32]);
    Ok(x.take_axis(&idx, axis)?)
}

/// Pack an NCTHW latent into ViT tokens `[B, T·H·W, C]`.
///
/// This is `_pack_tensors_3d(x, patch_size=1, patch_size_t=1)`: with unit patches the reference's
/// eight-way permute reduces to moving the channel axis last, so token order is `(t, h, w)`
/// row-major — matching [`crate::rope::create_token_ids`]'s `ij` mesh.
pub fn pack_tokens(x: &Array) -> Result<Array> {
    let s = x.shape();
    if s.len() != 5 {
        return Err(Error::Msg(format!(
            "minimax-h3 pack_tokens: expected [B, C, T, H, W], got {s:?}"
        )));
    }
    let (b, c, t, h, w) = (s[0], s[1], s[2], s[3], s[4]);
    Ok(x.transpose_axes(&[0, 2, 3, 4, 1])?
        .reshape(&[b, t * h * w, c])?)
}

/// Unpack ViT patch tokens back to NCTHW video.
///
/// `tokens` is `[B, T·H·W, out_channels · patch_size_t · patch_size²]` whose channel axis is laid
/// out `(c, pt, ph, pw)` row-major; the result is
/// `[B, out_channels, T·patch_size_t, H·patch_size, W·patch_size]`
/// (`_unpack_tensors_3d`).
pub fn unpack_tokens(
    tokens: &Array,
    out_channels: i32,
    patch_size: i32,
    patch_size_t: i32,
    lat_t: i32,
    lat_h: i32,
    lat_w: i32,
) -> Result<Array> {
    let s = tokens.shape();
    if s.len() != 3 {
        return Err(Error::Msg(format!(
            "minimax-h3 unpack_tokens: expected [B, N, C], got {s:?}"
        )));
    }
    let (b, n, c) = (s[0], s[1], s[2]);
    let expected_n = lat_t * lat_h * lat_w;
    let expected_c = out_channels * patch_size_t * patch_size * patch_size;
    if n != expected_n || c != expected_c {
        return Err(Error::Msg(format!(
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
        .transpose_axes(&[0, 4, 1, 5, 2, 6, 3, 7])?
        .reshape(&[
            b,
            out_channels,
            lat_t * patch_size_t,
            lat_h * patch_size,
            lat_w * patch_size,
        ])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic host-side test data. MLX's RNG evaluates on the GPU stream, which is not
    /// available on every `cargo test` worker thread; building from a host slice keeps these unit
    /// tests both stream-free and reproducible.
    fn spread(shape: &[i32]) -> Array {
        let n: i32 = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).sin() * 1.3).collect();
        Array::from_slice(&vals, shape)
    }

    #[test]
    fn pack_then_unpack_round_trips_a_unit_patch() {
        // With patch sizes of 1 the pack/unpack pair is an exact inverse.
        let x = spread(&[1, 3, 2, 3, 4]);
        let tokens = pack_tokens(&x).unwrap();
        assert_eq!(tokens.shape(), &[1, 24, 3]);
        let back = unpack_tokens(&tokens, 3, 1, 1, 2, 3, 4).unwrap();
        assert_eq!(back.shape(), x.shape());
        assert_eq!(back.as_slice::<f32>(), x.as_slice::<f32>());
    }

    /// The `(c, pt, ph, pw)` channel layout is what maps a token's features onto its
    /// `patch_size_t × patch_size × patch_size` pixel block. A wrong permute still produces the
    /// right SHAPE, so pin the actual placement.
    #[test]
    fn unpack_places_the_channel_major_patch_block() {
        // One token, 1 channel, 1×2×2 patch -> features [p0, p1, p2, p3] fill the block row-major.
        let tokens = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 1, 4]);
        let video = unpack_tokens(&tokens, 1, 2, 1, 1, 1, 1).unwrap();
        assert_eq!(video.shape(), &[1, 1, 1, 2, 2]);
        assert_eq!(video.as_slice::<f32>(), &[0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn slice_axis_rejects_out_of_range() {
        let x = spread(&[2, 5]);
        assert_eq!(slice_axis(&x, 1, 1, 4).unwrap().shape(), &[2, 3]);
        assert!(slice_axis(&x, 1, 0, 9).is_err());
        assert!(slice_axis(&x, 3, 0, 1).is_err());
        assert!(slice_axis(&x, 1, 4, 2).is_err());
    }

    #[test]
    fn pack_tokens_orders_tokens_t_then_h_then_w() {
        // Channel 0 carries a linear ramp over (t, h, w); packing must preserve that order.
        let vals: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let x = Array::from_slice(&vals, &[1, 1, 2, 2, 3]);
        let tokens = pack_tokens(&x).unwrap();
        assert_eq!(tokens.shape(), &[1, 12, 1]);
        assert_eq!(tokens.as_slice::<f32>(), vals.as_slice());
    }
}
