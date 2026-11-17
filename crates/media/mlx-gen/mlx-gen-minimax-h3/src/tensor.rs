//! Small shape helpers shared by the decoder modules.

use mlx_rs::Array;

use mlx_gen::{Error, Result};

/// `x[..., start..end, ...]` along `axis` (half-open, `end` not `len`): the shared core
/// [`mlx_gen::array::slice_range`] (sc-27448), re-exported under this crate's name.
pub use mlx_gen::array::slice_range as slice_axis;

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

/// Pack an NCHW latent image into ViT tokens `[B, H·W, C]` —
/// `MiniMaxH3ImageViTDecoder2d.forward`'s `permute(0, 2, 3, 1).reshape(B, H·W, C)`, so token order
/// is `(h, w)` row-major, matching [`crate::rope::create_token_ids_2d`].
pub fn pack_tokens_2d(x: &Array) -> Result<Array> {
    let s = x.shape();
    if s.len() != 4 {
        return Err(Error::Msg(format!(
            "minimax-h3 pack_tokens_2d: expected [B, C, H, W], got {s:?}"
        )));
    }
    let (b, c, h, w) = (s[0], s[1], s[2], s[3]);
    Ok(x.transpose_axes(&[0, 2, 3, 1])?.reshape(&[b, h * w, c])?)
}

/// Unpack 2-D ViT patch tokens back to an NCHW image.
///
/// `tokens` is `[B, H·W, out_channels · patch_size²]` with the channel axis laid out
/// `(c, ph, pw)` row-major; the result is `[B, out_channels, H·patch_size, W·patch_size]` — the
/// reference's `view(B, H, W, C, p, p).permute(0, 3, 1, 4, 2, 5)`.
pub fn unpack_tokens_2d(
    tokens: &Array,
    out_channels: i32,
    patch_size: i32,
    lat_h: i32,
    lat_w: i32,
) -> Result<Array> {
    let s = tokens.shape();
    if s.len() != 3 {
        return Err(Error::Msg(format!(
            "minimax-h3 unpack_tokens_2d: expected [B, N, C], got {s:?}"
        )));
    }
    let (b, n, c) = (s[0], s[1], s[2]);
    let expected_c = out_channels * patch_size * patch_size;
    if n != lat_h * lat_w || c != expected_c {
        return Err(Error::Msg(format!(
            "minimax-h3 unpack_tokens_2d: expected [B, {}, {expected_c}], got {s:?}",
            lat_h * lat_w
        )));
    }
    Ok(tokens
        .reshape(&[b, lat_h, lat_w, out_channels, patch_size, patch_size])?
        .transpose_axes(&[0, 3, 1, 4, 2, 5])?
        .reshape(&[b, out_channels, lat_h * patch_size, lat_w * patch_size])?)
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
    fn pack_2d_orders_tokens_h_then_w_and_round_trips() {
        let vals: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let x = Array::from_slice(&vals, &[1, 2, 2, 3]);
        let tokens = pack_tokens_2d(&x).unwrap();
        assert_eq!(tokens.shape(), &[1, 6, 2]);
        // token (h=0, w=1) carries channel 0 = 1 and channel 1 = 7.
        // `as_slice` reads the raw buffer, so compare through a contiguous copy of the view.
        let dense = |a: &Array| mlx_gen::array::contiguous(a).unwrap();
        assert_eq!(&dense(&tokens).as_slice::<f32>()[2..4], &[1.0, 7.0]);
        let back = unpack_tokens_2d(&tokens, 2, 1, 2, 3).unwrap();
        assert_eq!(dense(&back).as_slice::<f32>(), x.as_slice::<f32>());
        assert!(pack_tokens_2d(&spread(&[1, 2, 1, 2, 3])).is_err());
    }

    /// A 2×2 latent with a 2×2 patch: each token's 4 features fill its own pixel block, so pixel
    /// row 0 interleaves tokens (0,0) and (0,1). A wrong permute keeps the shape and moves pixels.
    #[test]
    fn unpack_2d_places_each_token_in_its_own_block() {
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tokens = Array::from_slice(&vals, &[1, 4, 4]);
        let img =
            mlx_gen::array::contiguous(&unpack_tokens_2d(&tokens, 1, 2, 2, 2).unwrap()).unwrap();
        assert_eq!(img.shape(), &[1, 1, 4, 4]);
        assert_eq!(
            img.as_slice::<f32>(),
            &[
                0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 12.0, 13.0, 10.0, 11.0, 14.0,
                15.0
            ]
        );
        assert!(unpack_tokens_2d(&tokens, 1, 2, 2, 3).is_err());
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
