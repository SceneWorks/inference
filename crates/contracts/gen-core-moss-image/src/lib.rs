//! # gen-core-moss-image
//!
//! The tensor-free MOSS-Image helpers that only the MOSS providers need, split out of gen-core in
//! sc-27316: the sampling schedule, guidance and DMD math, package/config validation, the edit
//! adapter's reference sizing and request mapping, and the MOSS training admission/recovery
//! contract. The request and execution-receipt contract a [`gen_core::GenerationRequest`] and the
//! [`gen_core::Generator`] MOSS entry points carry stays in `gen_core::moss_image` /
//! `gen_core::moss_image_edit` and is re-exported from the matching modules here.

pub mod moss_image;
pub mod moss_image_edit;
pub mod moss_image_training;
