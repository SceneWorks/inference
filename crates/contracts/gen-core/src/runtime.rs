//! Shared load/exec types used by both [`Generator`](crate::generator::Generator) and
//! [`Transform`](crate::transform::Transform): where weights come from, quantization +
//! precision knobs, adapter specs, cooperative cancellation, and progress events.

#[cfg(test)]
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::time::{Duration, SystemTime};

use sha2::{Digest, Sha256};

#[cfg(target_vendor = "apple")]
#[path = "runtime_clone.rs"]
mod bounded_clone_laws;

#[cfg(test)]
#[derive(Debug)]
struct SealHashWork(AtomicUsize);

#[cfg(test)]
thread_local! {
    static THREAD_FULL_HASH_WORK: Cell<usize> = const { Cell::new(0) };
    static THREAD_HASH_BYTES_READ: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
fn record_full_hash_work() {
    THREAD_FULL_HASH_WORK.with(|work| work.set(work.get() + 1));
}

/// Current-thread full-content hash count for regression tests that must prove a path does not
/// acquire an [`ArtifactSeal`]. This is test instrumentation only and is not artifact identity.
#[cfg(test)]
pub(crate) fn test_full_hash_work_count() -> usize {
    THREAD_FULL_HASH_WORK.with(Cell::get)
}

#[cfg(test)]
impl PartialEq for SealHashWork {
    fn eq(&self, _other: &Self) -> bool {
        // Work probes are test instrumentation, never artifact identity.
        true
    }
}

#[cfg(test)]
impl Eq for SealHashWork {}

#[cfg(test)]
impl std::hash::Hash for SealHashWork {
    fn hash<H: std::hash::Hasher>(&self, _state: &mut H) {}
}

use crate::memory_strategy::{
    MemoryBackend, MemoryDecodeGeometryPolicy, MemoryDecodeQualityRuntimeIdentity,
    MemoryNumericTier,
};

mod source;
pub use source::*;

mod seal;
pub use seal::*;

mod quant;
pub use quant::*;

mod load_spec;
pub use load_spec::*;

mod adapters;
pub use adapters::*;

mod sinks;
pub use sinks::*;

#[cfg(test)]
mod tests;
