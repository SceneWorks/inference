//! 5-point landmark alignment — the classical-CV glue (sc-3083) between the SCRFD detector
//! ([`crate::scrfd`]) and the ArcFace embedding ([`crate::iresnet`]).
//!
//! The similarity solver, the cv2-exact warp and both crops are backend-neutral and live in
//! [`gen_core_family::face_align`] (sc-27256), shared with the Candle twin; this module re-exports
//! them so `align::*` paths stay stable and adds the MLX ArcFace input packing (NHWC).

pub use gen_core_family::face_align::*;

use mlx_gen::Result;
use mlx_rs::Array;

/// Pack 112² RGB `u8` crops into the ArcFace input batch: NHWC `[N,112,112,3]` f32, normalized
/// `(rgb - 127.5) / 127.5` (the antelopev2 ArcFace blob, swapRB already folded in by warping RGB).
pub fn to_arcface_input(crops: &[Vec<u8>]) -> Result<Array> {
    let n = crops.len() as i32;
    let mut data = Vec::with_capacity(crops.len() * 112 * 112 * 3);
    for crop in crops {
        debug_assert_eq!(crop.len(), 112 * 112 * 3, "crop must be 112×112×3");
        data.extend(crop.iter().map(|&v| (v as f32 - 127.5) / 127.5));
    }
    Ok(Array::from_slice(&data, &[n, 112, 112, 3]))
}
