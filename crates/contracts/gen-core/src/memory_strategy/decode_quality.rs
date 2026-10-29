//! Production-latent tiled-decode quality records and geometry policies.

use super::*;

/// One deterministic production latent used to judge a tiled-decode policy.
///
/// `production_latent_provenance` names the immutable model revision, prompt fixture, denoise
/// schedule, and harness source that reproduce the latent. The enclosing evidence SHA binds the
/// complete ordered fixture set and its observed errors; this record intentionally carries no peak,
/// duration, or allocator field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryDecodeQualityFixture {
    pub seed: u64,
    pub production_latent_provenance: String,
    pub production_latent_sha256: String,
    pub dense_output_sha256: String,
    pub tiled_output_sha256: String,
    pub observed_error: u32,
}

/// Production-latent quality verdict for one exact decode coordinate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryDecodeQualityDisposition {
    Admitted,
    Refused { reason: String },
}

/// Immutable identity of the exact model artifact whose production latents were admitted.
///
/// `fingerprint` is the caller's already-validated resolved-tree identity. For an immutable hub
/// snapshot it is normally `repository@revision:variant`; app-managed artifacts may instead carry
/// a strong `sha256:` content identity. The runtime never derives this value from a request or a
/// manifest: the loader supplies an independently resolved identity and [`LoadSpec`](crate::LoadSpec)
/// compares it before a row can enter the provider contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryDecodeArtifactIdentity {
    pub repository: String,
    pub revision: String,
    pub variant: String,
    pub fingerprint: String,
}

/// Independently resolved identity of the artifact used by one concrete load. Callers attach this
/// beside packaged rows; [`LoadSpec`](crate::LoadSpec) refuses to pass rows into a provider unless
/// it matches exactly.
///
/// This axis is *independent* of the packaged rows — the loader resolves it from the model tree it
/// actually opened — which is what makes it able to discriminate. sc-19728 removed a parallel
/// `implementation_fingerprint` axis from this struct: its only possible source was a constant
/// compiled into the same build that also stamped the rows, so once currency enforcement was
/// dropped it could only ever compare a value against itself. A vacuous axis that still reads as
/// protection is worse than no axis, so it is gone rather than defanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryDecodeQualityRuntimeIdentity {
    pub artifact: MemoryDecodeArtifactIdentity,
}

/// Geometry-aware tiled-decode policy and its immutable production-quality receipt.
///
/// Every runtime identity axis that can change the latent or decoder route is explicit. A provider
/// adopting this table may execute bounded decode only when the loaded contract and request match
/// one `Admitted` row exactly. Failed rows remain first-class records so a known-bad coordinate is
/// distinguishable from an unmeasured one; both fail closed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryDecodeGeometryPolicy {
    pub quality_abi: u32,
    pub family: String,
    pub resolved_route: String,
    pub backend: MemoryBackend,
    pub tier: MemoryNumericTier,
    pub load_shape: LoadShape,
    pub artifact: MemoryDecodeArtifactIdentity,
    /// Source closure that produced this measurement — **a forensic stamp, not an admission gate**.
    ///
    /// `scripts/ci/decode_quality_implementation_fingerprint.py` derives it from the repository at
    /// collection time and the real-weight harness records it, so it always names the code that
    /// actually ran. It is sealed into [`Self::production_evidence_sha256`], so it cannot be edited
    /// after the fact, and it answers "was this row measured before or after the tiling rewrite?".
    ///
    /// It is deliberately **not** compared against the running build (sc-19728). That comparison
    /// existed and was removed: the closure spans whole crates plus `Cargo.lock`, so every
    /// dependency bump, pin bump, and main sync invalidated every measurement, and the constant was
    /// re-derived by hand each time — a gate that is routinely restamped is not gating, it is
    /// charging a round trip. Measurements now stand until they are remeasured. The properties that
    /// still fail closed are the ones that discriminate: the ABI, the sealed evidence digest, the
    /// measured error against [`Self::maximum_error`], and the artifact identity of the loaded
    /// model. Restoring currency enforcement means bumping [`MEMORY_DECODE_QUALITY_ABI`] — a
    /// deliberate, human-owned act at the granularity where invalidating every row is the intent.
    pub implementation_fingerprint: String,
    pub mode: MemoryMode,
    pub overlay: Option<String>,
    pub geometry: MemoryGeometry,
    pub use_pid: bool,
    pub tile_edge: u32,
    pub overlap: u32,
    pub metric: String,
    pub maximum_error: u32,
    pub fixtures: Vec<MemoryDecodeQualityFixture>,
    /// SHA-256 of the canonical quality record, excluding this field.
    pub production_evidence_sha256: String,
    pub disposition: MemoryDecodeQualityDisposition,
}

/// Exact request coordinate supplied before a selector chooses tiled-decode parameters.
#[derive(Clone, Copy, Debug)]
pub struct MemoryDecodePolicyQuery<'a> {
    pub tier: MemoryNumericTier,
    pub load_shape: LoadShape,
    pub mode_key: &'a str,
    pub overlay: Option<&'a str>,
    pub geometry: MemoryGeometry,
    pub use_pid: bool,
}

impl MemoryDecodeGeometryPolicy {
    /// SHA-256 of the canonical semantic quality record. No timing, memory, or performance field is
    /// present in the input by construction.
    pub fn computed_evidence_sha256(&self) -> String {
        let component_precision_floors = self
            .tier
            .component_precision_floors
            .iter()
            .map(|floor| {
                serde_json::json!({
                    "component": floor.component.as_str(),
                    "selected_tier": quant_key(floor.selected_tier),
                    "resident_tier": quant_key(floor.resident_tier),
                })
            })
            .collect::<Vec<_>>();
        let fixtures = self
            .fixtures
            .iter()
            .map(|fixture| {
                serde_json::json!({
                    "seed": fixture.seed,
                    "production_latent_provenance": fixture.production_latent_provenance,
                    "production_latent_sha256": fixture.production_latent_sha256,
                    "dense_output_sha256": fixture.dense_output_sha256,
                    "tiled_output_sha256": fixture.tiled_output_sha256,
                    "observed_error": fixture.observed_error,
                })
            })
            .collect::<Vec<_>>();
        let disposition = match &self.disposition {
            MemoryDecodeQualityDisposition::Admitted => {
                serde_json::json!({ "kind": "admitted" })
            }
            MemoryDecodeQualityDisposition::Refused { reason } => {
                serde_json::json!({ "kind": "refused", "reason": reason })
            }
        };
        let canonical = serde_json::json!({
            "quality_abi": self.quality_abi,
            "family": self.family,
            "resolved_route": self.resolved_route,
            "backend": self.backend.as_key(),
            "tier": {
                "precision": precision_key(self.tier.precision),
                "quant": self.tier.quant.map(quant_key),
                "component_precision_floors": component_precision_floors,
            },
            "load_shape": load_shape_key(self.load_shape),
            "artifact": {
                "repository": self.artifact.repository,
                "revision": self.artifact.revision,
                "variant": self.artifact.variant,
                "fingerprint": self.artifact.fingerprint,
            },
            "implementation_fingerprint": self.implementation_fingerprint,
            "mode": self.mode.as_key(),
            "overlay": self.overlay,
            "geometry": {
                "width": self.geometry.width,
                "height": self.geometry.height,
                "batch": self.geometry.batch,
                "frames": self.geometry.frames,
                "reference_count": self.geometry.reference_count,
            },
            "use_pid": self.use_pid,
            "tile_edge": self.tile_edge,
            "overlap": self.overlap,
            "metric": self.metric,
            "maximum_error": self.maximum_error,
            "fixtures": fixtures,
            "disposition": disposition,
        });
        format!("{:x}", Sha256::digest(canonical.to_string().as_bytes()))
    }

    /// Fill the evidence identity from the canonical quality fields.
    pub fn seal(mut self) -> Self {
        self.production_evidence_sha256 = self.computed_evidence_sha256();
        self
    }

    pub(super) fn matches_request_axes(
        &self,
        contract: &MemoryProviderContract,
        query: MemoryDecodePolicyQuery<'_>,
    ) -> bool {
        self.backend == contract.backend.backend_kind()
            && self.tier == query.tier
            && self.load_shape == query.load_shape
            && self.mode.as_key() == query.mode_key
            && self.overlay.as_deref() == query.overlay
            && self.geometry == query.geometry
            && self.use_pid == query.use_pid
    }

    pub(super) fn matches_coordinate(
        &self,
        contract: &MemoryProviderContract,
        query: MemoryDecodePolicyQuery<'_>,
        tile_edge: u32,
        overlap: u32,
    ) -> bool {
        self.matches_request_axes(contract, query)
            && self.tile_edge == tile_edge
            && self.overlap == overlap
    }
}
