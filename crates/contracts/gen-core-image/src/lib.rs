//! # gen-core-image
//!
//! Tensor-free HDR image I/O split out of gen-core in sc-27316: OpenEXR read/write ([`exr_io`]) and
//! the HDR conversions ([`hdr`]: the HLG master, 10-bit YUV packing, the VAE-range maps and the EXR
//! payload conversions). The colour policy a request opts into — `HdrColorSpace`, its primaries
//! and its working-space transfer — stays in `gen_core::hdr` and is re-exported from [`hdr`].

pub mod exr_io;
pub mod hdr;

pub use exr_io::{read_rgb_exr, write_rgb_exr, ExrImage, EXR_COLOR_SPACE_ATTRIBUTE};
pub use hdr::{
    exr_conditioning_to_vae_range, from_vae_range, hlg_inverse_oetf, hlg_oetf,
    rgb_signal_to_yuv420p10, to_vae_range, working_frame_to_exr_payload,
    working_frame_to_hlg_linear, HlgConverter, HlgMasterTags, Yuv420p10, HLG_MASTER_TAGS,
};
