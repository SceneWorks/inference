//! The Lens text encoder — gpt-oss-20b run encoder-only (forward to layer 23, capture hidden states
//! at layers `[5, 11, 17, 23]`). See epic 3164.
//!
//! Ported so far: the gpt-oss decoder-layer **attention core** (sc-3165), the **MoE** feed-forward +
//! full **decoder-layer** assembly (sc-3166), the MXFP4 expert dequant ([`mxfp4`]), and the 24-layer
//! **encoder stack** with multi-layer hidden capture ([`encoder::LensTextEncoder`], sc-3171). The
//! encoder front-end projection (DiT-side) follows.

pub mod encoder;
pub mod gpt_oss;
pub mod mxfp4;
