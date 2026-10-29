//! Sealed physical and request identity shared by the native Wan2.2 I2V providers.
//!
//! Backend crates own registration, safety hooks, and request-scoped execution. This module keeps
//! the deliberately identical MLX/Candle artifact ABI in one place: canonical repository/revision,
//! exact direct-file inventory, header-derived resident facts, and ordered adapter receipts.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::{
    AdapterKind, Conditioning, GenerationRequest, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryGeometry, MemoryLifecycleCapabilities, MemoryNumericTier, MemoryParameterRanges,
    MemoryPhase, MemoryProviderContract, MemoryRunContext, MemorySelection, MemoryStrategy,
    MemoryStrategyCapability, MemoryStrategyParameters, MemoryStrategySupport, MoeExpert,
    OffloadPolicy, Precision, Quant, ResidentRequestMemory, WeightsSource,
};

mod receipts;
pub use receipts::*;

mod routes;
pub use routes::*;

mod prepared;
pub use prepared::*;

mod fingerprint;
pub use fingerprint::*;

mod inventory;
pub use inventory::*;

mod load;
pub use load::*;

mod request;
pub use request::*;

mod selection;
pub use selection::*;

#[cfg(test)]
mod tests;
