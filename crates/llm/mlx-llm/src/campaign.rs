//! Source-owned evidence primitives for the SC-20671 dense baseline campaign.
//!
//! This module deliberately keeps the receipt producer beside the product decoder.  The JSON
//! harness can validate evidence, but it must not be the component inventing model identity,
//! geometry, or lifecycle observations.  Device collection is supplied by the campaign runner;
//! these helpers make its inputs deterministic and testable without weights or Metal.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use crate::campaign_supervisor::{self, RunRequest, SafetyPolicy, SystemProbe};
// sc-27303: the engine-facing observation seam lives in `decode::observe`; re-exported so every
// `crate::campaign::…` / `mlx_llm::campaign::…` path keeps resolving.
pub use crate::decode::observe::{
    CompressedKvArm, CompressedKvMethod, InventoryFile, MemorySample, Observer, ProductGeometry,
    SnapshotInventory, TeacherForcing,
};

use core_llm::{
    Message, Role, Sampling, StreamEvent, TextLlmOutput, TextLlmRequest, ThinkingMode, Tokenizer,
    ToolSpec,
};

// The harness, split by concern (sc-27303): safety policy and quality-gate vocabulary; pinned
// benchmark models and row budgets; row admission; the receipt schema, builder and validation;
// artifact bundles and campaign publication; launch, resume and the CLI; the required schedule
// and noise floor; product fixtures, observation and quality; host and repository identity; and
// the product receipt assembly.
mod admission;
mod artifacts;
mod fixtures;
mod host;
mod launch;
mod models;
mod policy;
mod product_receipt;
mod receipt;
mod schedule;
pub use admission::*;
pub use artifacts::*;
pub use fixtures::*;
pub use host::*;
pub use launch::*;
pub use models::*;
pub use policy::*;
pub use product_receipt::*;
pub use receipt::*;
pub use schedule::*;

#[cfg(test)]
pub(crate) mod tests;
