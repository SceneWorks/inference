//! # gen-core-checkpoint
//!
//! Tensor-free checkpoint readers and converters split out of gen-core in sc-27316: the torch
//! `.pth`/`.pt` zip + pickle reader ([`pth`], moved from mlx-gen in sc-24213 so the Supra2
//! converter runs on both backends) and the Supra2 checkpoint converter and pinned-bundle contract
//! ([`supra2`]). Family-specific converters that only one family's providers use (FLUX 3, LTX)
//! live with that family's sibling crate.

pub mod pth;
pub mod supra2;
