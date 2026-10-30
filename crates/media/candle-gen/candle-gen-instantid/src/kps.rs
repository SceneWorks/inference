//! InstantID keypoint control-image renderer (sc-3111): the bit-exact OpenCV `draw_kps` port,
//! [`letterbox`] and the canonical [`VIEW_ANGLE_KPS`] multi-view landmark sets.
//!
//! The implementation is backend-neutral and lives in
//! [`gen_core_family::instantid_kps`] (sc-27256), shared with the MLX twin; this module
//! re-exports it so `kps::*` paths stay stable.

pub use gen_core_family::instantid_kps::*;
