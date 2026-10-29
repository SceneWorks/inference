//! 5-point landmark alignment — the classical-CV glue between the SCRFD detector ([`crate::scrfd`])
//! and the ArcFace embedding ([`crate::iresnet`]).
//!
//! The similarity solver, the cv2-exact warp and both crops are backend-neutral and live in
//! [`candle_gen::gen_core::face_align`] (sc-27256), shared with the MLX twin; this module re-exports
//! them so `align::*` paths stay stable and adds the candle ArcFace input packing (NCHW; MLX packs
//! NHWC).

pub use candle_gen::gen_core::face_align::*;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::Result;

/// Pack 112² RGB `u8` crops into the ArcFace input batch: **NCHW** `[N,3,112,112]` f32, normalized
/// `(rgb - 127.5) / 127.5` (the antelopev2 ArcFace blob). MLX packs the same data NHWC; candle's
/// `conv2d` is NCHW, so the crops are built NHWC then permuted to NCHW here.
pub fn to_arcface_input(crops: &[Vec<u8>], device: &Device) -> Result<Tensor> {
    let n = crops.len();
    let mut data = Vec::with_capacity(n * 112 * 112 * 3);
    for crop in crops {
        debug_assert_eq!(crop.len(), 112 * 112 * 3, "crop must be 112×112×3");
        data.extend(crop.iter().map(|&v| (v as f32 - 127.5) / 127.5));
    }
    let nhwc = Tensor::from_vec(data, (n, 112, 112, 3), device)?;
    Ok(nhwc.permute((0, 3, 1, 2))?.contiguous()?)
}
