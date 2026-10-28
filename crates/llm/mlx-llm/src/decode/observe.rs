//! The engine-owned observation seam (sc-27303 / sc-27312).
//!
//! The product decode path ([`crate::provider`], [`crate::decode`], `crate::kv_policy` and the
//! packed-cache primitives) reports phase, allocation, cache and token observations through
//! [`Observer`], and runs a compressed KV arm through [`CompressedKvArm`]. Those types used to live
//! in the measurement-campaign harness (`crate::campaign`), which made the engine depend on the
//! harness. They live here now; `crate::campaign` re-exports them so its public paths are unchanged.
//! This is the first step toward compiling the campaign harness only behind an opt-in feature.

use std::path::PathBuf;

/// The narrow observation seam used by a real campaign runner.  The runner owns platform probes
/// (`footprint` and `mlx_rs::memory`); the product path owns the phase boundaries and generation.
/// Teacher forcing for the campaign's greedy-agreement measurement: the decode loop feeds the
/// forced token at each step instead of its own choice, which it still reports through
/// [`Observer::token_probability`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeacherForcing {
    /// Ordinary generation.
    Off,
    /// Feed this token at the current step.
    Token(i32),
    /// The forced stream is exhausted: stop before sampling another position.
    Exhausted,
}

pub trait Observer {
    fn phase(&mut self, name: &'static str);
    /// Teacher forcing for decode `step` of the observed generation; off unless a campaign quality
    /// observer supplies a forced stream.
    fn teacher_forced_token(&mut self, _step: usize) -> TeacherForcing {
        TeacherForcing::Off
    }
    fn allocation(&mut self, role: &'static str, lifetime: &'static str, bytes: u64);
    fn allocation_event(
        &mut self,
        _kind: &'static str,
        role: &'static str,
        lifetime: &'static str,
        bytes: u64,
    ) {
        self.allocation(role, lifetime, bytes);
    }
    /// Observe a cache-ownership release. This is deliberately separate from allocation_event:
    /// released bytes are lifecycle evidence, never positive workspace or retained allocation.
    fn release_event(&mut self, _kind: &'static str, _role: &'static str, _bytes: u64) {}
    /// Wall-clock duration of the real snapshot inventory plus provider/model load that created
    /// the bound campaign session.
    fn load_duration(&mut self, _milliseconds: f64) {}
    /// Candidate-only samples bracketing explicit model-parameter materialization.  The campaign
    /// observer replays these as the first two receipt phases so later reference allocations cannot
    /// be mistaken for candidate weights.
    fn load_boundary(&mut self, _process_start: &MemorySample, _weights_loaded: &MemorySample) {}
    /// Reset MLX's process-global high-water immediately before the measured prefill and seal the
    /// active-memory baseline independently from the historical model-load boundary.
    fn begin_prefill_memory_window(&mut self) {}
    /// Current cumulative cache ownership at a decoder boundary. The product cache supplies bytes,
    /// live length, allocated capacity, and actual MLX array scalar width so append events are not
    /// summed and the receipt cannot confuse model-weight dtype with cache dtype.
    fn cache_snapshot(&mut self, bytes: u64, _tokens: u64, _capacity: u64, _element_bytes: u64) {
        self.allocation("cache", "persistent", bytes);
    }
    /// Bind a product-owned provider-load identity before the first phase.  The default preserves
    /// existing non-campaign observers without permitting receipt assembly to invent an identity.
    fn bind_session(&mut self, _session_id: &str) {}
    /// Cache state is versioned by the provider-load session.  It prevents a same-PID reload from
    /// being mislabeled as a warmed provider/cache.
    fn cache_state(&mut self, _version: u64) {}
    /// Record a coordinate operation that actually executed on this observed product session.
    fn operation(&mut self, _operation: &'static str) {}
    /// Exact fp32 logits materialized from the production decode seam only while a campaign
    /// observer is attached.  Ordinary serving never takes this host-read path.
    fn logits(&mut self, _stage: &'static str, _values: &[f32]) {}
    /// Exact probability of the sampled production token, derived at the same decode seam as the
    /// sampler.  It is intentionally not a caller-provided quality number.
    fn token_probability(&mut self, _stage: &'static str, _token: i32, _probability: f64) {}
    /// Under teacher forcing, the model's exact probability of the forced token at the same decode
    /// seam (beside [`Self::token_probability`] of its own choice), so the candidate's likelihood
    /// is measured on the reference's tokens.
    fn forced_token_probability(&mut self, _token: i32, _probability: f64) {}
    /// Model identity is derived from the exact files resolved by the product loader, never a
    /// caller-authored digest string.
    fn snapshot_inventory(&mut self, _inventory: &SnapshotInventory) {}
    fn geometry(&mut self, _geometry: ProductGeometry) {}
    /// Immutable model-boundary evidence of one compressed (packed) decoder cache, forwarded once
    /// after decode. Dense caches never call it.
    fn packed_cache_evidence(&mut self, _evidence: &crate::primitives::PackedCacheEvidence) {}
    /// A compressed-arm operation that ran on the explicit dense path, with its reason.
    fn dense_fallback(&mut self, _operation: &str, _reason: &str) {}
    /// A packed-to-dense transition rebuilt the whole cached history as dense K/V.
    fn dense_reconstruction(&mut self, _bytes: u64) {}
    /// Measured physical storage of a live compressed cache at a decoder boundary.
    fn compressed_storage(&mut self, _storage: &crate::primitives::CompressedCacheStorage) {}
}

/// Geometry copied from the loaded decoder configuration, not an input JSON field.  Query/KV
/// lengths and batch are supplied by the actual request immediately before dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProductGeometry {
    pub query_heads: u64,
    pub kv_heads: u64,
    pub head_dimension: u64,
    pub layers: u64,
    pub element_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemorySample {
    pub captured_at: String,
    pub pid: u32,
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub mlx_active_bytes: u64,
    pub mlx_cache_bytes: u64,
    pub mlx_peak_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InventoryFile {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotInventory {
    pub root: PathBuf,
    pub files: Vec<InventoryFile>,
    pub bytes: u64,
    pub sha256: String,
}

/// Compressed KV representations the SC-20671 harness can run as a `compressed` row. The
/// producer, receipt, and validators are method-agnostic: a further candidate (for example the
/// SC-20677 RVQ/RaBitQ caches) is one variant here plus its cache selection and kernel parity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressedKvMethod {
    /// SC-20675 packed 2-bit group-affine cache read by the SC-20676 fused Metal kernel. Its
    /// identifier stays `group-affine` so earlier receipts and resume identities keep binding.
    GroupAffine,
    /// The same cache and reader with 4-bit (KIVI-style) codes, group 32.
    GroupAffine4,
    /// The same cache and reader with 8-bit codes, group 32.
    GroupAffine8,
}

impl CompressedKvMethod {
    pub const ALL: [Self; 3] = [Self::GroupAffine, Self::GroupAffine4, Self::GroupAffine8];

    /// Stable receipt/CLI identifier.
    pub fn id(self) -> &'static str {
        match self {
            Self::GroupAffine => "group-affine",
            Self::GroupAffine4 => "group-affine-4",
            Self::GroupAffine8 => "group-affine-8",
        }
    }

    /// Code width of this method's packed cache.
    pub fn code_bits(self) -> crate::primitives::PackedCodeBits {
        match self {
            Self::GroupAffine => crate::primitives::PackedCodeBits::Two,
            Self::GroupAffine4 => crate::primitives::PackedCodeBits::Four,
            Self::GroupAffine8 => crate::primitives::PackedCodeBits::Eight,
        }
    }

    /// Representation identity this method's cache and reader carry (recorded on receipts).
    pub fn representation_identity(self) -> &'static str {
        crate::primitives::packed_metal_identity(self.code_bits())
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|method| method.id() == value)
            .ok_or_else(|| {
                format!(
                    "unknown compressed KV method {value:?}; expected one of {:?}",
                    Self::ALL.map(Self::id)
                )
            })
    }

    /// GPU-family tuning profile of this method's fused reader, recorded on every compressed
    /// receipt. No device-family detector exists in this crate; none is needed because MLX's
    /// Metal backend runs only on Apple silicon, whose oldest Mac GPU (M1) is Apple family 7, so
    /// any device this arm can dispatch on qualifies for the recent-family profile.
    pub fn kernel_gpu_family(self) -> crate::primitives::PackedMetalGpuFamily {
        match self {
            Self::GroupAffine | Self::GroupAffine4 | Self::GroupAffine8 => {
                crate::primitives::PackedMetalGpuFamily::Apple7OrNewer
            }
        }
    }

    /// Build this method's retained fused reader once per campaign session.
    pub fn arm(self) -> Result<CompressedKvArm, String> {
        // The same retained reader production generation builds (`kv_policy`), at this arm's
        // code width: its identity and GPU-family profile are `representation_identity` and
        // `kernel_gpu_family`.
        let reader = match self {
            Self::GroupAffine | Self::GroupAffine4 | Self::GroupAffine8 => {
                crate::kv_policy::group_affine_reader(self.code_bits())?
            }
        };
        Ok(CompressedKvArm {
            method: self,
            reader,
        })
    }
}

/// The compressed arm of one campaign session: its method and retained fused reader.
pub struct CompressedKvArm {
    method: CompressedKvMethod,
    reader: crate::primitives::CompiledKernelHandle,
}

impl CompressedKvArm {
    /// Test-only arm bound to an arbitrary retained reader (e.g. one the cache refuses to bind).
    #[cfg(test)]
    pub(crate) fn with_reader(
        method: CompressedKvMethod,
        reader: crate::primitives::CompiledKernelHandle,
    ) -> Self {
        Self { method, reader }
    }

    pub fn method(&self) -> CompressedKvMethod {
        self.method
    }

    /// The decoder cache for one compressed request, chosen before any K/V mutation. A refusal is
    /// returned as a dense route with its reason, which the caller must record.
    pub(crate) fn select_cache(
        &self,
        model: &crate::models::CausalLm,
    ) -> crate::primitives::DecoderCacheSelection {
        match self.method {
            CompressedKvMethod::GroupAffine
            | CompressedKvMethod::GroupAffine4
            | CompressedKvMethod::GroupAffine8 => {
                model.select_cache_with_packed_reader(self.reader.clone(), 1, 1, false)
            }
        }
    }

    /// Kernel parity of this method's fused reader against an independent host-fp32
    /// dequantize-then-attend reference over the exact stored codes.
    pub fn kernel_parity_errors(&self) -> Result<Vec<f64>, String> {
        match self.method {
            CompressedKvMethod::GroupAffine
            | CompressedKvMethod::GroupAffine4
            | CompressedKvMethod::GroupAffine8 => {
                crate::primitives::group_affine_kernel_fp32_parity_errors(&self.reader)
            }
        }
    }
}
