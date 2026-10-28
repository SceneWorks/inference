//! Text-encoder architecture contracts and header-only substitution validation.
//!
//! A generator descriptor advertises the exact conditioning encoder its denoiser was trained
//! against.  Callers may replace the bundled weights through [`crate::LoadSpec::text_encoder`], but
//! only after this module has compared both the Hugging Face `config.json` and the safetensors tensor
//! headers with that advertised contract.  Validation deliberately happens before either tensor
//! backend opens/materializes weights, so an incompatible encoder is a small, legible load error
//! rather than a late matmul-shape failure (or, worse, a plausible render with the wrong conditioning).

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

#[cfg(target_vendor = "apple")]
#[path = "encoder_guarded.rs"]
mod guarded;

use crate::weightsmeta::Dtype;
use crate::{
    safetensors_path_tensor_headers, Error, PinnedWeightsFile, Result, SafetensorsTensorHeader,
    VisionEncoderContract, WeightsSource,
};

mod types;
pub use types::*;

mod source;
use source::*;

mod facts;
pub use facts::*;

mod contract;

mod discovery;
pub use discovery::*;

#[cfg(test)]
mod tests;
