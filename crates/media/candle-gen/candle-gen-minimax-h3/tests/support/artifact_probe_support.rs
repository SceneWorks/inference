//! Backend-neutral policy and bounded reader for the **actual-checkpoint numeric probe**
//! (sc-26679 terminal repair).
//!
//! This file is included by BOTH H3 integration-test binaries:
//!
//! * `crates/media/candle-gen/candle-gen-minimax-h3/tests/artifact_probe.rs`
//! * `crates/media/mlx-gen/mlx-gen-minimax-h3/tests/artifact_probe.rs`
//!
//! It is deliberately free of tensor-library imports. Both backends share one manifest/oracle
//! contract, one receipt shape and one range reader; only tensor construction and the native
//! production call are backend-specific.
//!
//! # What is trusted, and why the 34 GB digest is NOT recomputed
//!
//! The manifest names an artifact OUTSIDE the repository together with the sha256 that the
//! SceneWorks staging step already verified while it installed/pinned the artifact. Staging then
//! writes a small receipt whose own sha256 is recorded in the manifest. This consumer:
//!
//! 1. hashes the *receipt* file (cheap) and requires the recorded digest to match;
//! 2. requires the receipt's `artifact_path`, `artifact_sha256` and `artifact_bytes` to equal the
//!    manifest's, and the artifact's actual byte length on disk to equal the declared length.
//!
//! That is a **trust binding**, not a second proof of the whole-file digest: re-hashing tens of
//! gigabytes on every probe would be an unbounded read the contract forbids. Receipts state this
//! explicitly (`digest_trust`, `whole_file_digest_recomputed = false`). The payload bytes that are
//! actually compared are independently hashed per selected tensor and bound to the pinned CPU
//! oracle, so the numeric expectation is still tied to the exact bytes consumed.
//!
//! # Bounded reads
//!
//! Preflight is header-only. The validated safetensors layout comes from
//! `gen_core::weightsmeta::safetensors_reader_tensor_locations` (the workspace's shared header
//! reader, which proves the payload region is contiguous and exactly covers the file). Payload
//! bytes are then read ONLY for the selected tensor ranges, by seeking to each declared offset and
//! reading exactly its declared byte count. Limits on case count, selected tensor count and total
//! selected bytes are enforced before the first payload byte is read, and preflight refusals
//! therefore report `payload_reads = 0`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

pub const MANIFEST_ENV: &str = "H3_ACTUAL_PROBE_MANIFEST";
pub const RECEIPT_ENV: &str = "H3_ACTUAL_PROBE_RECEIPT_DIR";

pub const MANIFEST_SCHEMA: &str = "sceneworks.h3.actual_checkpoint_probe";
pub const ORACLE_SCHEMA: &str = "sceneworks.h3.actual_checkpoint_oracle";
pub const RECEIPT_SCHEMA: &str = "sceneworks.h3.actual_checkpoint_probe_receipt";
pub const STAGING_SCHEMA: &str = "sceneworks.h3.staging_receipt";
pub const SCHEMA_VERSION: u64 = 1;

/// Hard ceilings. A manifest may lower these, never raise them.
pub const HARD_MAX_CASES: usize = 64;
pub const HARD_MAX_SELECTED_TENSORS: usize = 8192;
pub const HARD_MAX_SELECTED_BYTES: u64 = 512 * 1024 * 1024;
pub const HARD_MAX_INPUT_ELEMENTS: usize = 1 << 22;
pub const HARD_MAX_OUTPUT_ELEMENTS: usize = 1 << 22;
pub const HARD_MAX_TIMESTEPS: usize = 4096;
pub const HARD_MAX_CASE_ID: usize = 128;
pub const HARD_MAX_TENSOR_NAME: usize = 512;

/// Canonical direct-pruned-curve table key (same constant the contract crate owns).
pub const TABLE_KEY: &str = "adaln_t_table";

pub const KIND_W4A8: &str = "w4a8_projection";
pub const KIND_CONVROT: &str = "convrot_projection";
/// One Weights map carrying BOTH a packed W4A8 operand set and a ConvRot INT8 operand set. The
/// native loader must keep each in its own format (never densify one into the other) and the case
/// compares the concatenation `[w4a8_output, convrot_output]` along the last axis against the
/// pinned producers' combined expectation.
pub const KIND_MIXED: &str = "mixed_w4a8_convrot";
pub const KIND_INTERPOLATION: &str = "pruned_timestep_interpolation";
pub const KIND_REDUCED: &str = "pruned_reduced_projection";

/// There is deliberately **no standalone `dense_projection` kind**: no independent pinned
/// dense-only CPU oracle exists. The dense-vs-ConvRot contrast is the companion leg inside the
/// pinned ConvRot oracle's `mixedLayer` sections, so it is covered WITHIN `convrot_projection`
/// (the native `quant::lin` route assertion refuses a convrot case that resolves dense, and a
/// dense case that would resolve packed). No standalone dense fidelity is claimed.
const KINDS: &[&str] = &[
    KIND_W4A8,
    KIND_CONVROT,
    KIND_MIXED,
    KIND_INTERPOLATION,
    KIND_REDUCED,
];

/// Metadata files are small; a huge JSON must fail before any payload read. Caps are enforced on
/// the file length AND during the read.
pub const MAX_MANIFEST_BYTES: u64 = 8 * 1024 * 1024;
pub const MAX_STAGING_BYTES: u64 = 4 * 1024 * 1024;
pub const MAX_PARENT_RECEIPT_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_ORACLE_BYTES: u64 = 256 * 1024 * 1024;

/// The pinned, independent CPU producers this consumer accepts as oracle provenance. Their bytes
/// are embedded at compile time, so the expected sha256 is the checkout's exact frozen producer —
/// an arbitrary `producer` string can never satisfy the binding.
const PINNED_PRUNED_ORACLE: &[u8] =
    include_bytes!("../../../../../../scripts/h3_pruned_adaln_oracle.py");
const PINNED_CONVROT_ORACLE: &[u8] =
    include_bytes!("../../../../../../scripts/h3_convrot_int8_oracle.py");
const PINNED_W4A8_ORACLE: &[u8] =
    include_bytes!("../../../../../../scripts/ci/generate_minimax_h3_w4a8_fixture.py");

pub const PINNED_PRUNED_SCRIPT: &str = "scripts/h3_pruned_adaln_oracle.py";
pub const PINNED_CONVROT_SCRIPT: &str = "scripts/h3_convrot_int8_oracle.py";
pub const PINNED_W4A8_SCRIPT: &str = "scripts/ci/generate_minimax_h3_w4a8_fixture.py";

pub fn pinned_producer_sha256(script: &str) -> Option<String> {
    match script {
        PINNED_PRUNED_SCRIPT => Some(sha256_hex(PINNED_PRUNED_ORACLE)),
        PINNED_CONVROT_SCRIPT => Some(sha256_hex(PINNED_CONVROT_ORACLE)),
        PINNED_W4A8_SCRIPT => Some(sha256_hex(PINNED_W4A8_ORACLE)),
        _ => None,
    }
}

/// Canonical per-kind comparison ceiling (the same constants the producer bridge emits). An oracle
/// declaring a looser bound is refused, so a manifest cannot launder an arbitrary tolerance.
fn canonical_tolerance(kind: &str) -> Option<f32> {
    match kind {
        KIND_W4A8 | KIND_CONVROT | KIND_MIXED => Some(0.0),
        KIND_INTERPOLATION => Some(2e-6),
        KIND_REDUCED => Some(2e-5),
        _ => None,
    }
}

fn required_producers(kind: &str) -> &'static [&'static str] {
    match kind {
        KIND_W4A8 => &[PINNED_W4A8_SCRIPT],
        KIND_CONVROT => &[PINNED_CONVROT_SCRIPT],
        // A mixed case binds both producers because its expectation is their concatenation.
        KIND_MIXED => &[PINNED_W4A8_SCRIPT, PINNED_CONVROT_SCRIPT],
        KIND_INTERPOLATION | KIND_REDUCED => &[PINNED_PRUNED_SCRIPT],
        _ => &[],
    }
}

/// One refusal or execution error. `code` is stable and appears verbatim in receipts.
#[derive(Debug, Clone)]
pub struct ProbeError {
    pub code: &'static str,
    pub detail: String,
}

impl ProbeError {
    pub fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.detail)
    }
}

pub type PResult<T> = std::result::Result<T, ProbeError>;

fn err<T>(code: &'static str, detail: impl Into<String>) -> PResult<T> {
    Err(ProbeError::new(code, detail))
}

// =================================================================================================
// Path policy
// =================================================================================================

/// The Inference workspace root (the checkout that compiled this test binary).
pub fn workspace_root() -> PathBuf {
    let start = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut root = start.to_path_buf();
    for ancestor in start.ancestors() {
        let manifest = ancestor.join("Cargo.toml");
        if manifest.is_file() {
            if let Ok(text) = std::fs::read_to_string(&manifest) {
                if text.contains("[workspace]") {
                    root = ancestor.to_path_buf();
                }
            }
        }
    }
    root.canonicalize().unwrap_or(root)
}

/// Canonicalize an existing path, or canonicalize its parent and re-append the final component so
/// a *missing* file still yields a stable, comparison-safe absolute path.
pub fn canonicalish(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    match (path.parent(), path.file_name()) {
        (Some(parent), Some(name)) => parent
            .canonicalize()
            .map(|p| p.join(name))
            .unwrap_or_else(|_| path.to_path_buf()),
        _ => path.to_path_buf(),
    }
}

/// Require an absolute path with no `..` component and outside the Inference workspace.
///
/// Symlinks are resolved for existing paths; a path that resolves into the checkout is refused.
/// The SceneWorks checkout is a *different* tree; keeping staged artifacts and receipts out of it
/// as well is the producer's obligation, stated on every receipt.
pub fn ensure_outside_workspace(path: &Path, what: &str) -> PResult<PathBuf> {
    if !path.is_absolute() {
        return err(
            "bad_env_path",
            format!("{what} must be an absolute path, got {}", path.display()),
        );
    }
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return err(
            "bad_env_path",
            format!("{what} must not contain `..`: {}", path.display()),
        );
    }
    let root = workspace_root();
    let resolved = canonicalish(path);
    if resolved.starts_with(&root) || path.starts_with(&root) {
        return err(
            "bad_env_path",
            format!(
                "{what} must live OUTSIDE the repository ({}): {}",
                root.display(),
                resolved.display()
            ),
        );
    }
    Ok(resolved)
}

pub fn manifest_path_from_env() -> PResult<PathBuf> {
    let raw = std::env::var_os(MANIFEST_ENV).ok_or_else(|| {
        ProbeError::new(
            "bad_env_path",
            format!("{MANIFEST_ENV} is not set (absolute manifest path required)"),
        )
    })?;
    ensure_outside_workspace(Path::new(&raw), MANIFEST_ENV)
}

/// Resolve, validate and create the receipt directory. Creating it early means even a malformed
/// manifest refusal has a destination.
pub fn receipt_dir_from_env() -> PResult<PathBuf> {
    let raw = std::env::var_os(RECEIPT_ENV).ok_or_else(|| {
        ProbeError::new(
            "bad_env_path",
            format!("{RECEIPT_ENV} is not set (absolute directory required)"),
        )
    })?;
    ensure_receipt_dir(Path::new(&raw))
}

fn ensure_receipt_dir(path: &Path) -> PResult<PathBuf> {
    let resolved = ensure_outside_workspace(path, RECEIPT_ENV)?;
    std::fs::create_dir_all(&resolved).map_err(|error| {
        ProbeError::new(
            "bad_env_path",
            format!(
                "cannot create receipt directory {}: {error}",
                resolved.display()
            ),
        )
    })?;
    resolved.canonicalize().map_err(|error| {
        ProbeError::new(
            "bad_env_path",
            format!(
                "cannot canonicalize receipt directory {}: {error}",
                resolved.display()
            ),
        )
    })
}

// =================================================================================================
// Small parsing helpers (strict; a misspelled optional field is not silently ignored)
// =================================================================================================

fn object<'a>(value: &'a Value, key: &str, ctx: &str) -> PResult<&'a Map<String, Value>> {
    value.get(key).and_then(Value::as_object).ok_or_else(|| {
        ProbeError::new(
            "manifest_malformed",
            format!("{ctx}.{key} must be an object"),
        )
    })
}

fn map_object<'a>(
    map: &'a Map<String, Value>,
    key: &str,
    ctx: &str,
) -> PResult<&'a Map<String, Value>> {
    map.get(key).and_then(Value::as_object).ok_or_else(|| {
        ProbeError::new("oracle_malformed", format!("{ctx}.{key} must be an object"))
    })
}

fn field<'a>(map: &'a Map<String, Value>, key: &str, ctx: &str) -> PResult<&'a Value> {
    map.get(key)
        .ok_or_else(|| ProbeError::new("manifest_malformed", format!("{ctx}.{key} is required")))
}

fn string_field(map: &Map<String, Value>, key: &str, ctx: &str) -> PResult<String> {
    field(map, key, ctx)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| {
            ProbeError::new(
                "manifest_malformed",
                format!("{ctx}.{key} must be a string"),
            )
        })
}

fn optional_string(map: &Map<String, Value>, key: &str, ctx: &str) -> PResult<Option<String>> {
    match map.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(_) => err(
            "manifest_malformed",
            format!("{ctx}.{key} must be a string when present"),
        ),
    }
}

fn u64_field(map: &Map<String, Value>, key: &str, ctx: &str) -> PResult<u64> {
    field(map, key, ctx)?
        .as_u64()
        .ok_or_else(|| ProbeError::new("manifest_malformed", format!("{ctx}.{key} must be a u64")))
}

fn optional_usize(map: &Map<String, Value>, key: &str, ctx: &str) -> PResult<Option<usize>> {
    match map.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .map(Some)
            .ok_or_else(|| {
                ProbeError::new(
                    "manifest_malformed",
                    format!("{ctx}.{key} must be a non-negative integer"),
                )
            }),
    }
}

fn optional_bool(map: &Map<String, Value>, key: &str, ctx: &str) -> PResult<bool> {
    match map.get(key) {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(value)) => Ok(*value),
        Some(_) => err("manifest_malformed", format!("{ctx}.{key} must be a bool")),
    }
}

fn f32_value(value: &Value, ctx: &str) -> PResult<f32> {
    let number = value
        .as_f64()
        .ok_or_else(|| ProbeError::new("manifest_malformed", format!("{ctx} must be a number")))?;
    let narrowed = number as f32;
    if !narrowed.is_finite() {
        return err(
            "manifest_malformed",
            format!("{ctx} is not representable as a finite f32 ({number})"),
        );
    }
    Ok(narrowed)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn require_sha(value: &str, ctx: &str) -> PResult<String> {
    if value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit()) {
        Ok(value.to_ascii_lowercase())
    } else {
        err(
            "artifact_sha_invalid",
            format!("{ctx} must be 64 hex characters, got {value:?}"),
        )
    }
}

fn checked_elements(shape: &[usize], what: &str) -> PResult<usize> {
    if shape.is_empty() || shape.len() > 8 {
        return err(
            "manifest_malformed",
            format!("{what} must have 1..=8 dimensions, got {shape:?}"),
        );
    }
    let mut total: usize = 1;
    for &dim in shape {
        if dim == 0 {
            return err(
                "manifest_malformed",
                format!("{what} has a zero dimension: {shape:?}"),
            );
        }
        total = total.checked_mul(dim).ok_or_else(|| {
            ProbeError::new(
                "manifest_malformed",
                format!("{what} element count overflow"),
            )
        })?;
    }
    Ok(total)
}

fn require_tensor_name(name: &str, ctx: &str) -> PResult<String> {
    if name.is_empty()
        || name.len() > HARD_MAX_TENSOR_NAME
        || name.trim() != name
        || name.contains(['/', '\\'])
        || name.chars().any(char::is_control)
        || name == "."
        || name == ".."
    {
        return err(
            "selection_bounds",
            format!("{ctx}: invalid tensor identifier {name:?}"),
        );
    }
    Ok(name.to_owned())
}

/// Read a metadata file with a hard byte cap, checked on the length BEFORE the read and again on
/// the bytes actually read. A huge JSON fails with `metadata_too_large` and `payload_reads = 0`.
pub fn read_bounded(path: &Path, max_bytes: u64, what: &str) -> PResult<Vec<u8>> {
    let metadata = std::fs::metadata(path).map_err(|error| {
        ProbeError::new(
            "manifest_unreadable",
            format!("cannot stat {what} {}: {error}", path.display()),
        )
    })?;
    if metadata.len() > max_bytes {
        return err(
            "metadata_too_large",
            format!(
                "{what} {} is {} bytes, cap is {max_bytes}",
                path.display(),
                metadata.len()
            ),
        );
    }
    let file = File::open(path).map_err(|error| {
        ProbeError::new(
            "manifest_unreadable",
            format!("cannot open {what} {}: {error}", path.display()),
        )
    })?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(max_bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| {
            ProbeError::new(
                "manifest_unreadable",
                format!("cannot read {what} {}: {error}", path.display()),
            )
        })?;
    if bytes.len() as u64 > max_bytes {
        return err(
            "metadata_too_large",
            format!(
                "{what} {} grew past the {max_bytes}-byte cap",
                path.display()
            ),
        );
    }
    Ok(bytes)
}

fn remap_io_error(error: ProbeError, fallback: &'static str) -> ProbeError {
    if error.code == "metadata_too_large" {
        error
    } else {
        ProbeError::new(fallback, error.detail)
    }
}

/// Require an EXACT pinned-producer provenance set. `producer`/`producer_version` alone are labels;
/// this is the binding that refuses an arbitrary string before any payload read.
fn parse_producer_provenance(map: &Map<String, Value>) -> PResult<BTreeMap<String, String>> {
    let entries = field(map, "producer_provenance", "oracle")?
        .as_array()
        .ok_or_else(|| {
            ProbeError::new(
                "oracle_producer_mismatch",
                "oracle.producer_provenance must be an array",
            )
        })?;
    if entries.is_empty() {
        return err(
            "oracle_producer_mismatch",
            "oracle.producer_provenance must not be empty",
        );
    }
    let mut provenance = BTreeMap::new();
    for (index, entry) in entries.iter().enumerate() {
        let ctx = format!("oracle.producer_provenance[{index}]");
        let entry = entry.as_object().ok_or_else(|| {
            ProbeError::new(
                "oracle_producer_mismatch",
                format!("{ctx} must be an object"),
            )
        })?;
        let script = string_field(entry, "script", &ctx)?;
        let declared = require_sha(
            &string_field(entry, "sha256", &ctx)?,
            &format!("{ctx}.sha256"),
        )?;
        let Some(expected) = pinned_producer_sha256(&script) else {
            return Err(ProbeError::new(
                "oracle_producer_provenance",
                format!("{ctx}: {script:?} is not a pinned oracle producer"),
            ));
        };
        if declared != expected {
            return Err(ProbeError::new(
                "oracle_producer_mismatch",
                format!(
                    "{ctx}: {script} hashes to {declared}, the checkout's pinned producer is {expected}"
                ),
            ));
        }
        if provenance.insert(script.clone(), declared).is_some() {
            return err(
                "oracle_producer_mismatch",
                format!("{ctx}: duplicate producer {script:?}"),
            );
        }
    }
    Ok(provenance)
}

fn require_case_id(id: &str) -> PResult<String> {
    if id.is_empty()
        || id.len() > HARD_MAX_CASE_ID
        || !id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        return err("manifest_malformed", format!("invalid case id {id:?}"));
    }
    Ok(id.to_owned())
}

// =================================================================================================
// Manifest
// =================================================================================================

#[derive(Clone, Debug)]
pub struct ArtifactRef {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug)]
pub struct StagingRef {
    pub receipt_path: PathBuf,
    pub receipt_sha256: String,
    pub verified_artifact_sha256: String,
}

#[derive(Clone, Debug)]
pub struct OracleRef {
    pub path: PathBuf,
    pub sha256: String,
    /// Human-readable producer label. Not a provenance proof on its own — the exact pinned
    /// producer source is bound by `producer_provenance`.
    pub producer: String,
    pub producer_version: u64,
    pub artifact_sha256: String,
    /// `script (repo-relative) -> sha256` for the PINNED producer source(s) whose arithmetic the
    /// oracle values come from. The consumer embeds those scripts at compile time and refuses any
    /// mismatch, so an arbitrary producer string alone cannot satisfy preflight.
    pub producer_provenance: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_cases: usize,
    pub max_selected_tensors: usize,
    pub max_selected_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_cases: HARD_MAX_CASES,
            max_selected_tensors: HARD_MAX_SELECTED_TENSORS,
            max_selected_bytes: HARD_MAX_SELECTED_BYTES,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DitGeometry {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub time_embed_dim: usize,
}

impl Default for DitGeometry {
    fn default() -> Self {
        // Shipped MiniMax-H3 DiT geometry.
        Self {
            hidden_size: 5376,
            num_layers: 50,
            time_embed_dim: 2688,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Select {
    pub base: Option<String>,
    /// Second packed base for `mixed_w4a8_convrot` (the ConvRot INT8 operand set).
    pub convrot_base: Option<String>,
    pub bias: bool,
    pub w4a8_bias: bool,
    pub convrot_bias: bool,
    pub table: Option<String>,
    pub prefix: Option<String>,
    pub weight: Option<String>,
    pub bias_tensor: Option<String>,
    pub grid: Option<usize>,
    pub rank: Option<usize>,
    pub out_features: Option<usize>,
    pub expect_route: Option<String>,
    pub topology: bool,
    /// Raw-artifact-ID -> canonical-name aliases (naming only; never a numeric remap). Passed so a
    /// native loader that expects canonical names (e.g. `transformer_blocks.N.adaln_proj.linear.*`)
    /// can consume a raw finetuned file whose tensors are named `blocks.N.adaln_proj.*`. The
    /// oracle and per-tensor sha stay bound to the RAW bytes; the canonical name is naming only.
    pub aliases: BTreeMap<String, String>,
}

impl Select {
    pub fn raw_of(&self, canonical: &str) -> String {
        for (raw, mapped) in &self.aliases {
            if mapped == canonical {
                return raw.clone();
            }
        }
        canonical.to_owned()
    }
}

#[derive(Clone, Debug)]
pub struct Case {
    pub id: String,
    pub kind: String,
    pub select: Select,
    pub input_shape: Option<Vec<usize>>,
    pub input_values: Option<Vec<f32>>,
    /// Activation dtype for the numeric input: `F32` (default) or `BF16`. The pinned CPU
    /// producers replay BF16 activations for W4A8/ConvRot, so those cases must declare BF16; the
    /// pruned curve is F32-only.
    pub input_dtype: String,
    pub timesteps: Option<Vec<f32>>,
    pub tolerance_max_abs: f32,
    pub selected_tensors: Vec<String>,
}

impl Case {
    pub fn require_input(&self) -> PResult<(&[usize], &[f32])> {
        match (&self.input_shape, &self.input_values) {
            (Some(shape), Some(values)) => Ok((shape, values)),
            _ => err(
                "manifest_malformed",
                format!("case {} requires `input` with shape and values", self.id),
            ),
        }
    }

    pub fn require_timesteps(&self) -> PResult<&[f32]> {
        self.timesteps.as_deref().ok_or_else(|| {
            ProbeError::new(
                "manifest_malformed",
                format!("case {} requires `input.timesteps`", self.id),
            )
        })
    }
}

#[derive(Clone, Debug)]
pub struct Manifest {
    pub backend: String,
    pub artifact: ArtifactRef,
    pub staging: StagingRef,
    pub oracle: OracleRef,
    pub limits: Limits,
    pub dit: DitGeometry,
    /// Whether `dit` came from the manifest (from the actual source binding) rather than the
    /// shipped defaults. Candle pruned cases require an explicit binding.
    pub dit_explicit: bool,
    pub cases: Vec<Case>,
}

pub fn parse_manifest(bytes: &[u8]) -> PResult<Manifest> {
    let value: Value = serde_json::from_slice(bytes).map_err(|error| {
        ProbeError::new(
            "manifest_malformed",
            format!("manifest is not valid JSON: {error}"),
        )
    })?;
    let root = value
        .as_object()
        .ok_or_else(|| ProbeError::new("manifest_malformed", "manifest must be a JSON object"))?;
    if string_field(root, "schema", "manifest")? != MANIFEST_SCHEMA {
        return err(
            "manifest_malformed",
            format!("manifest.schema must be {MANIFEST_SCHEMA:?}"),
        );
    }
    if u64_field(root, "schema_version", "manifest")? != SCHEMA_VERSION {
        return err(
            "manifest_malformed",
            format!("manifest.schema_version must be {SCHEMA_VERSION}"),
        );
    }
    let backend = string_field(root, "backend", "manifest")?;
    if backend != "both" && backend != "candle" && backend != "mlx" {
        return err(
            "manifest_malformed",
            format!("manifest.backend must be candle|mlx|both, got {backend:?}"),
        );
    }

    let artifact_raw = object(&value, "artifact", "manifest")?;
    let artifact = ArtifactRef {
        path: PathBuf::from(string_field(artifact_raw, "path", "artifact")?),
        sha256: require_sha(
            &string_field(artifact_raw, "sha256", "artifact")?,
            "artifact.sha256",
        )?,
        bytes: u64_field(artifact_raw, "bytes", "artifact")?,
    };

    let staging_raw = object(&value, "staging", "manifest")?;
    let staging = StagingRef {
        receipt_path: PathBuf::from(string_field(staging_raw, "receipt_path", "staging")?),
        receipt_sha256: require_sha(
            &string_field(staging_raw, "receipt_sha256", "staging")?,
            "staging.receipt_sha256",
        )?,
        verified_artifact_sha256: require_sha(
            &string_field(staging_raw, "verified_artifact_sha256", "staging")?,
            "staging.verified_artifact_sha256",
        )?,
    };

    let oracle_raw = object(&value, "oracle", "manifest")?;
    let oracle = OracleRef {
        path: PathBuf::from(string_field(oracle_raw, "path", "oracle")?),
        sha256: require_sha(
            &string_field(oracle_raw, "sha256", "oracle")?,
            "oracle.sha256",
        )?,
        producer: string_field(oracle_raw, "producer", "oracle")?,
        producer_version: u64_field(oracle_raw, "producer_version", "oracle")?,
        artifact_sha256: require_sha(
            &string_field(oracle_raw, "artifact_sha256", "oracle")?,
            "oracle.artifact_sha256",
        )?,
        producer_provenance: parse_producer_provenance(oracle_raw)?,
    };

    let defaults = Limits::default();
    let limits = match root.get("limits") {
        None | Some(Value::Null) => defaults,
        Some(value) => {
            let map = value.as_object().ok_or_else(|| {
                ProbeError::new("manifest_malformed", "manifest.limits must be an object")
            })?;
            let mut limits = defaults;
            if let Some(value) = optional_usize(map, "max_cases", "limits")? {
                limits.max_cases = value;
            }
            if let Some(value) = optional_usize(map, "max_selected_tensors", "limits")? {
                limits.max_selected_tensors = value;
            }
            if let Some(value) = optional_usize(map, "max_selected_bytes", "limits")? {
                limits.max_selected_bytes = value as u64;
            }
            limits
        }
    };
    if limits.max_cases == 0
        || limits.max_cases > HARD_MAX_CASES
        || limits.max_selected_tensors == 0
        || limits.max_selected_tensors > HARD_MAX_SELECTED_TENSORS
        || limits.max_selected_bytes == 0
        || limits.max_selected_bytes > HARD_MAX_SELECTED_BYTES
    {
        return err(
            "limits_exceeded",
            format!(
                "manifest limits must be within 1..={HARD_MAX_CASES} cases, \
                 1..={HARD_MAX_SELECTED_TENSORS} tensors and 1..={HARD_MAX_SELECTED_BYTES} bytes"
            ),
        );
    }

    let dit_explicit = root.get("dit").is_some_and(|value| !value.is_null());
    let dit = match root.get("dit") {
        None | Some(Value::Null) => DitGeometry::default(),
        Some(value) => {
            let map = value.as_object().ok_or_else(|| {
                ProbeError::new("manifest_malformed", "manifest.dit must be an object")
            })?;
            let mut dit = DitGeometry::default();
            if let Some(value) = optional_usize(map, "hidden_size", "dit")? {
                dit.hidden_size = value;
            }
            if let Some(value) = optional_usize(map, "num_layers", "dit")? {
                dit.num_layers = value;
            }
            if let Some(value) = optional_usize(map, "time_embed_dim", "dit")? {
                dit.time_embed_dim = value;
            }
            if dit.hidden_size == 0 || dit.num_layers == 0 || dit.time_embed_dim == 0 {
                return err(
                    "manifest_malformed",
                    "manifest.dit dimensions must be nonzero",
                );
            }
            dit
        }
    };

    let cases_raw = field(root, "cases", "manifest")?
        .as_array()
        .ok_or_else(|| ProbeError::new("manifest_malformed", "manifest.cases must be an array"))?;
    if cases_raw.len() > limits.max_cases {
        return err(
            "limits_exceeded",
            format!(
                "manifest has {} cases, limit is {}",
                cases_raw.len(),
                limits.max_cases
            ),
        );
    }
    let mut cases = Vec::with_capacity(cases_raw.len());
    let mut seen = BTreeSet::new();
    for (index, raw) in cases_raw.iter().enumerate() {
        let case = parse_case(raw, index)?;
        if !seen.insert(case.id.clone()) {
            return err(
                "manifest_malformed",
                format!("duplicate case id {:?}", case.id),
            );
        }
        cases.push(case);
    }
    if cases.is_empty() {
        return err("manifest_malformed", "manifest.cases must not be empty");
    }

    Ok(Manifest {
        backend,
        artifact,
        staging,
        oracle,
        limits,
        dit,
        dit_explicit,
        cases,
    })
}

fn parse_case(raw: &Value, index: usize) -> PResult<Case> {
    let ctx = format!("cases[{index}]");
    let map = raw
        .as_object()
        .ok_or_else(|| ProbeError::new("manifest_malformed", format!("{ctx} must be an object")))?;
    let id = require_case_id(&string_field(map, "id", &ctx)?)?;
    let kind = string_field(map, "kind", &ctx)?;
    if !KINDS.contains(&kind.as_str()) {
        return err(
            "manifest_malformed",
            format!("{ctx}.kind must be one of {KINDS:?}, got {kind:?}"),
        );
    }
    let select_raw = object(raw, "select", &ctx)?;
    let opt_name = |key: &str| -> PResult<Option<String>> {
        optional_string(select_raw, key, &format!("{ctx}.select"))?
            .map(|name| require_tensor_name(&name, &format!("{ctx}.select.{key}")))
            .transpose()
    };
    let mut aliases = BTreeMap::new();
    if let Some(value) = select_raw.get("aliases").filter(|v| !v.is_null()) {
        let map = value.as_object().ok_or_else(|| {
            ProbeError::new(
                "manifest_malformed",
                format!("{ctx}.select.aliases must be an object"),
            )
        })?;
        let mut canonical_seen = BTreeSet::new();
        for (raw, canonical) in map {
            let raw = require_tensor_name(raw, &format!("{ctx}.select.aliases key"))?;
            let canonical = canonical.as_str().ok_or_else(|| {
                ProbeError::new(
                    "manifest_malformed",
                    format!("{ctx}.select.aliases[{raw:?}] must be a string"),
                )
            })?;
            let canonical =
                require_tensor_name(canonical, &format!("{ctx}.select.aliases[{raw:?}]"))?;
            if !canonical_seen.insert(canonical.clone()) {
                return err(
                    "selection_bounds",
                    format!("{ctx}.select.aliases maps two raw tensors to {canonical:?}"),
                );
            }
            aliases.insert(raw, canonical);
        }
    }
    let select = Select {
        base: opt_name("base")?,
        convrot_base: opt_name("convrot_base")?,
        bias: optional_bool(select_raw, "bias", &format!("{ctx}.select"))?,
        w4a8_bias: optional_bool(select_raw, "w4a8_bias", &format!("{ctx}.select"))?,
        convrot_bias: optional_bool(select_raw, "convrot_bias", &format!("{ctx}.select"))?,
        table: opt_name("table")?,
        prefix: opt_name("prefix")?,
        weight: opt_name("weight")?,
        bias_tensor: opt_name("bias_tensor")?,
        grid: optional_usize(select_raw, "grid", &format!("{ctx}.select"))?,
        rank: optional_usize(select_raw, "rank", &format!("{ctx}.select"))?,
        out_features: optional_usize(select_raw, "out_features", &format!("{ctx}.select"))?,
        expect_route: optional_string(select_raw, "expect_route", &format!("{ctx}.select"))?,
        topology: optional_bool(select_raw, "topology", &format!("{ctx}.select"))?,
        aliases,
    };
    if let Some(route) = select.expect_route.as_deref() {
        if !matches!(route, "w4a8" | "convrot" | "affine" | "dense") {
            return err(
                "manifest_malformed",
                format!("{ctx}.select.expect_route has unknown route {route:?}"),
            );
        }
    }

    let tolerance = object(raw, "tolerance", &ctx)?;
    let tolerance_max_abs = f32_value(
        field(tolerance, "max_abs", &format!("{ctx}.tolerance"))?,
        &format!("{ctx}.tolerance.max_abs"),
    )?;
    if tolerance_max_abs < 0.0 {
        return err(
            "manifest_malformed",
            format!("{ctx}.tolerance.max_abs must be non-negative"),
        );
    }

    let mut input_shape = None;
    let mut input_values = None;
    let mut input_dtype = "F32".to_owned();
    let mut timesteps = None;
    match raw.get("input") {
        None | Some(Value::Null) => {}
        Some(input) => {
            let input = input.as_object().ok_or_else(|| {
                ProbeError::new(
                    "manifest_malformed",
                    format!("{ctx}.input must be an object"),
                )
            })?;
            if let Some(dtype) = optional_string(input, "dtype", &format!("{ctx}.input"))? {
                if !matches!(dtype.as_str(), "F32" | "BF16") {
                    return err(
                        "manifest_malformed",
                        format!("{ctx}.input.dtype must be F32 or BF16, got {dtype:?}"),
                    );
                }
                input_dtype = dtype;
            }
            match input.get("timesteps") {
                Some(Value::Array(values)) => {
                    if values.is_empty() || values.len() > HARD_MAX_TIMESTEPS {
                        return err(
                            "limits_exceeded",
                            format!(
                                "{ctx}.input.timesteps must have 1..={HARD_MAX_TIMESTEPS} entries"
                            ),
                        );
                    }
                    let parsed = values
                        .iter()
                        .enumerate()
                        .map(|(i, value)| f32_value(value, &format!("{ctx}.input.timesteps[{i}]")))
                        .collect::<PResult<Vec<f32>>>()?;
                    timesteps = Some(parsed);
                }
                None => {
                    let shape = field(input, "shape", &format!("{ctx}.input"))?
                        .as_array()
                        .ok_or_else(|| {
                            ProbeError::new(
                                "manifest_malformed",
                                format!("{ctx}.input.shape must be an array"),
                            )
                        })?
                        .iter()
                        .enumerate()
                        .map(|(i, value)| {
                            value
                                .as_u64()
                                .and_then(|n| usize::try_from(n).ok())
                                .filter(|&n| n > 0)
                                .ok_or_else(|| {
                                    ProbeError::new(
                                        "manifest_malformed",
                                        format!(
                                            "{ctx}.input.shape[{i}] must be a positive integer"
                                        ),
                                    )
                                })
                        })
                        .collect::<PResult<Vec<usize>>>()?;
                    let count = checked_elements(&shape, &format!("{ctx}.input.shape"))?;
                    if count > HARD_MAX_INPUT_ELEMENTS {
                        return err(
                            "limits_exceeded",
                            format!("{ctx}.input has {count} elements, cap is {HARD_MAX_INPUT_ELEMENTS}"),
                        );
                    }
                    let values = field(input, "values", &format!("{ctx}.input"))?
                        .as_array()
                        .ok_or_else(|| {
                            ProbeError::new(
                                "manifest_malformed",
                                format!("{ctx}.input.values must be an array"),
                            )
                        })?
                        .iter()
                        .enumerate()
                        .map(|(i, value)| f32_value(value, &format!("{ctx}.input.values[{i}]")))
                        .collect::<PResult<Vec<f32>>>()?;
                    if values.len() != count {
                        return err(
                            "manifest_malformed",
                            format!(
                                "{ctx}.input.values has {} entries for shape {shape:?} ({count})",
                                values.len()
                            ),
                        );
                    }
                    input_shape = Some(shape);
                    input_values = Some(values);
                }
                Some(_) => {
                    return err(
                        "manifest_malformed",
                        format!("{ctx}.input.timesteps must be an array"),
                    )
                }
            }
        }
    }

    // Kind/field coherence.
    match kind.as_str() {
        KIND_W4A8 | KIND_CONVROT | KIND_MIXED => {
            if select.base.is_none() {
                return err(
                    "manifest_malformed",
                    format!("{ctx}.select.base is required for {kind}"),
                );
            }
            if kind == KIND_MIXED && select.convrot_base.is_none() {
                return err(
                    "manifest_malformed",
                    format!("{ctx}.select.convrot_base is required for {kind}"),
                );
            }
            if kind == KIND_MIXED && select.base == select.convrot_base {
                return err(
                    "manifest_malformed",
                    format!("{ctx}: mixed case needs two distinct bases"),
                );
            }
            if input_shape.is_none() {
                return err(
                    "manifest_malformed",
                    format!("{ctx} requires numeric input shape/values"),
                );
            }
            if input_dtype != "BF16" {
                return err(
                    "manifest_malformed",
                    format!(
                        "{ctx}: packed {kind} cases require input.dtype=BF16 (the pinned CPU \
                         quantizer replays BF16 activations)"
                    ),
                );
            }
        }
        KIND_INTERPOLATION => {
            if select.table.is_none() {
                return err(
                    "manifest_malformed",
                    format!("{ctx}.select.table is required for {kind}"),
                );
            }
            if timesteps.is_none() {
                return err(
                    "manifest_malformed",
                    format!("{ctx} requires input.timesteps"),
                );
            }
        }
        KIND_REDUCED => {
            if select.prefix.is_none() && select.weight.is_none() {
                return err(
                    "manifest_malformed",
                    format!("{ctx}.select.prefix or select.weight is required for {kind}"),
                );
            }
            if input_shape.is_none() {
                return err(
                    "manifest_malformed",
                    format!("{ctx} requires numeric input shape/values"),
                );
            }
            if input_dtype != "F32" {
                return err(
                    "manifest_malformed",
                    format!("{ctx}: pruned reduced projection requires input.dtype=F32"),
                );
            }
        }
        _ => unreachable!("KINDS checked above"),
    }

    let mut selected_tensors = Vec::new();
    if let Some(value) = map.get("selected_tensors").filter(|v| !v.is_null()) {
        let list = value.as_array().ok_or_else(|| {
            ProbeError::new(
                "manifest_malformed",
                format!("{ctx}.selected_tensors must be an array"),
            )
        })?;
        for (i, name) in list.iter().enumerate() {
            let name = name.as_str().ok_or_else(|| {
                ProbeError::new(
                    "manifest_malformed",
                    format!("{ctx}.selected_tensors[{i}] must be a string"),
                )
            })?;
            selected_tensors.push(require_tensor_name(
                name,
                &format!("{ctx}.selected_tensors[{i}]"),
            )?);
        }
    }

    Ok(Case {
        id,
        kind,
        select,
        input_shape,
        input_values,
        input_dtype,
        timesteps,
        tolerance_max_abs,
        selected_tensors,
    })
}

// =================================================================================================
// Oracle
// =================================================================================================

#[derive(Clone, Debug)]
pub struct OracleTensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug)]
pub struct OracleCase {
    pub selected: BTreeMap<String, OracleTensor>,
    /// Canonical input binding emitted by the pinned producer: dtype (`BF16`, `F32`, or
    /// `TIMESTEPS`), shape, and sha256 over the little-endian f32 input values. The manifest case
    /// must match this exactly; a producer/consumer input mismatch refuses before payload reads.
    pub input_dtype: String,
    pub input_shape: Vec<usize>,
    pub input_values_sha256: String,
    /// Canonical per-kind bound. The manifest may tighten it, never widen it.
    pub tolerance_max_abs: f32,
    pub output_shape: Vec<usize>,
    pub output_values: Vec<f32>,
}

/// The case's canonical input binding, computed exactly as the producer does.
pub fn input_binding(case: &Case) -> PResult<(String, Vec<usize>, String)> {
    let (dtype, shape, values) = if let Some(timesteps) = &case.timesteps {
        (
            "TIMESTEPS".to_owned(),
            vec![timesteps.len()],
            timesteps.clone(),
        )
    } else {
        let (shape, values) = case.require_input()?;
        (case.input_dtype.clone(), shape.to_vec(), values.to_vec())
    };
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok((dtype, shape, sha256_hex(&bytes)))
}

#[derive(Clone, Debug)]
pub struct Oracle {
    pub artifact_sha256: String,
    pub cases: BTreeMap<String, OracleCase>,
}

pub fn parse_oracle(bytes: &[u8]) -> PResult<Oracle> {
    let value: Value = serde_json::from_slice(bytes).map_err(|error| {
        ProbeError::new(
            "oracle_malformed",
            format!("oracle is not valid JSON: {error}"),
        )
    })?;
    let root = value
        .as_object()
        .ok_or_else(|| ProbeError::new("oracle_malformed", "oracle must be a JSON object"))?;
    if string_field(root, "schema", "oracle")? != ORACLE_SCHEMA {
        return err(
            "oracle_malformed",
            format!("oracle.schema must be {ORACLE_SCHEMA:?}"),
        );
    }
    if u64_field(root, "schema_version", "oracle")? != SCHEMA_VERSION {
        return err(
            "oracle_malformed",
            format!("oracle.schema_version must be {SCHEMA_VERSION}"),
        );
    }
    let artifact_sha256 = require_sha(
        &string_field(root, "artifact_sha256", "oracle")?,
        "oracle.artifact_sha256",
    )?;
    let cases_raw = field(root, "cases", "oracle")?
        .as_array()
        .ok_or_else(|| ProbeError::new("oracle_malformed", "oracle.cases must be an array"))?;
    let mut cases = BTreeMap::new();
    for (index, raw) in cases_raw.iter().enumerate() {
        let ctx = format!("oracle.cases[{index}]");
        let map = raw.as_object().ok_or_else(|| {
            ProbeError::new("oracle_malformed", format!("{ctx} must be an object"))
        })?;
        let id = require_case_id(&string_field(map, "id", &ctx)?)?;
        let mut selected = BTreeMap::new();
        for (i, tensor) in field(map, "selected", &ctx)?
            .as_array()
            .ok_or_else(|| {
                ProbeError::new(
                    "oracle_malformed",
                    format!("{ctx}.selected must be an array"),
                )
            })?
            .iter()
            .enumerate()
        {
            let tctx = format!("{ctx}.selected[{i}]");
            let tensor_map = tensor.as_object().ok_or_else(|| {
                ProbeError::new("oracle_malformed", format!("{tctx} must be an object"))
            })?;
            let name = require_tensor_name(&string_field(tensor_map, "name", &tctx)?, &tctx)?;
            let entry = OracleTensor {
                dtype: string_field(tensor_map, "dtype", &tctx)?,
                shape: field(tensor_map, "shape", &tctx)?
                    .as_array()
                    .ok_or_else(|| {
                        ProbeError::new(
                            "oracle_malformed",
                            format!("{tctx}.shape must be an array"),
                        )
                    })?
                    .iter()
                    .enumerate()
                    .map(|(d, value)| {
                        value
                            .as_u64()
                            .and_then(|n| usize::try_from(n).ok())
                            .ok_or_else(|| {
                                ProbeError::new(
                                    "oracle_malformed",
                                    format!("{tctx}.shape[{d}] must be a non-negative integer"),
                                )
                            })
                    })
                    .collect::<PResult<Vec<usize>>>()?,
                bytes: u64_field(tensor_map, "bytes", &tctx)?,
                sha256: require_sha(
                    &string_field(tensor_map, "sha256", &tctx)?,
                    &format!("{tctx}.sha256"),
                )?,
            };
            if selected.insert(name.clone(), entry).is_some() {
                return err(
                    "oracle_malformed",
                    format!("{ctx}.selected has duplicate tensor {name:?}"),
                );
            }
        }
        let input = map_object(map, "input", &ctx)?;
        let input_dtype = string_field(input, "dtype", &format!("{ctx}.input"))?;
        if !matches!(input_dtype.as_str(), "BF16" | "F32" | "TIMESTEPS") {
            return err(
                "oracle_malformed",
                format!("{ctx}.input.dtype must be F32, BF16 or TIMESTEPS"),
            );
        }
        let input_shape = input
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                ProbeError::new("oracle_malformed", format!("{ctx}.input.shape is required"))
            })?
            .iter()
            .map(|value| {
                value.as_u64().map(|v| v as usize).ok_or_else(|| {
                    ProbeError::new(
                        "oracle_malformed",
                        format!("{ctx}.input.shape must be integers"),
                    )
                })
            })
            .collect::<PResult<Vec<usize>>>()?;
        let input_values_sha256 = require_sha(
            &string_field(input, "values_sha256", &format!("{ctx}.input"))?,
            &format!("{ctx}.input.values_sha256"),
        )?;
        let tolerance = map_object(map, "tolerance", &ctx)?;
        let tolerance_max_abs = f32_value(
            field(tolerance, "max_abs", &format!("{ctx}.tolerance"))?,
            &format!("{ctx}.tolerance.max_abs"),
        )?;
        if tolerance_max_abs < 0.0 || !tolerance_max_abs.is_finite() {
            return err(
                "oracle_malformed",
                format!("{ctx}.tolerance.max_abs must be finite and non-negative"),
            );
        }
        let output = map_object(map, "output", &ctx)?;
        let output_shape = field(output, "shape", &format!("{ctx}.output"))?
            .as_array()
            .ok_or_else(|| {
                ProbeError::new(
                    "oracle_malformed",
                    format!("{ctx}.output.shape must be an array"),
                )
            })?
            .iter()
            .enumerate()
            .map(|(d, value)| {
                value
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| {
                        ProbeError::new(
                            "oracle_malformed",
                            format!("{ctx}.output.shape[{d}] must be a non-negative integer"),
                        )
                    })
            })
            .collect::<PResult<Vec<usize>>>()?;
        let count = checked_elements(&output_shape, &format!("{ctx}.output.shape"))?;
        if count > HARD_MAX_OUTPUT_ELEMENTS {
            return err(
                "limits_exceeded",
                format!("{ctx}.output has {count} elements, cap is {HARD_MAX_OUTPUT_ELEMENTS}"),
            );
        }
        let output_values = field(output, "values", &format!("{ctx}.output"))?
            .as_array()
            .ok_or_else(|| {
                ProbeError::new(
                    "oracle_malformed",
                    format!("{ctx}.output.values must be an array"),
                )
            })?
            .iter()
            .enumerate()
            .map(|(i, value)| f32_value(value, &format!("{ctx}.output.values[{i}]")))
            .collect::<PResult<Vec<f32>>>()?;
        if output_values.len() != count {
            return err(
                "oracle_malformed",
                format!(
                    "{ctx}.output.values has {} entries for shape {output_shape:?} ({count})",
                    output_values.len()
                ),
            );
        }
        if cases
            .insert(
                id.clone(),
                OracleCase {
                    selected,
                    input_dtype,
                    input_shape,
                    input_values_sha256,
                    tolerance_max_abs,
                    output_shape,
                    output_values,
                },
            )
            .is_some()
        {
            return err(
                "oracle_malformed",
                format!("duplicate oracle case id {id:?}"),
            );
        }
    }
    Ok(Oracle {
        artifact_sha256,
        cases,
    })
}

// =================================================================================================
// Header index + bounded payload reader
// =================================================================================================

#[derive(Clone, Debug)]
pub struct IndexedTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub file_offset: u64,
    pub data_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub struct HeaderIndex {
    pub file_len: u64,
    /// Bytes consumed by the header reader: exactly `8 + header_len`. The backend loaders verify
    /// the file position lands here, so a buffered reader that prefetched payload cannot pass as a
    /// header-only preflight.
    pub header_bytes_read: u64,
    pub tensors: BTreeMap<String, IndexedTensor>,
}

impl HeaderIndex {
    pub fn insert(&mut self, tensor: IndexedTensor) {
        self.tensors.insert(tensor.name.clone(), tensor);
    }

    pub fn require(&self, name: &str) -> PResult<&IndexedTensor> {
        self.tensors.get(name).ok_or_else(|| {
            ProbeError::new(
                "selection_missing",
                format!("artifact has no tensor {name:?}"),
            )
        })
    }
}

#[derive(Clone, Debug)]
pub struct SelectedPayload {
    /// The RAW artifact tensor id whose bytes were read and hashed.
    pub name: String,
    /// The canonical name the native loader consumes (naming only; defaults to `name`).
    pub canonical: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
    pub sha256: String,
}

/// Read exactly the declared payload ranges for `names`, in file order.
///
/// Bounds (count and total bytes) are checked before the file is opened. The artifact length is
/// re-checked against the header index so a file swapped between preflight and read is refused
/// rather than read at stale offsets.
pub fn read_selected(
    artifact: &Path,
    index: &HeaderIndex,
    names: &[String],
    aliases: &BTreeMap<String, String>,
    limits: Limits,
) -> PResult<Vec<SelectedPayload>> {
    if names.len() > limits.max_selected_tensors {
        return err(
            "limits_exceeded",
            format!(
                "case selects {} tensors, limit is {}",
                names.len(),
                limits.max_selected_tensors
            ),
        );
    }
    let mut locations = Vec::with_capacity(names.len());
    let mut total: u64 = 0;
    let mut seen = BTreeSet::new();
    for name in names {
        if !seen.insert(name.as_str()) {
            return err(
                "selection_bounds",
                format!("duplicate selected tensor {name:?}"),
            );
        }
        let location = index.require(name)?.clone();
        total = total
            .checked_add(location.data_bytes)
            .ok_or_else(|| ProbeError::new("limits_exceeded", "selected byte count overflow"))?;
        locations.push(location);
    }
    if total > limits.max_selected_bytes {
        return err(
            "limits_exceeded",
            format!(
                "case selects {total} bytes, limit is {}",
                limits.max_selected_bytes
            ),
        );
    }
    locations.sort_by_key(|location| location.file_offset);

    let mut file = File::open(artifact).map_err(|error| {
        ProbeError::new(
            "artifact_missing",
            format!("cannot open {}: {error}", artifact.display()),
        )
    })?;
    let actual_len = file
        .metadata()
        .map_err(|error| {
            ProbeError::new("artifact_missing", format!("artifact metadata: {error}"))
        })?
        .len();
    if actual_len != index.file_len {
        return err(
            "artifact_size_changed",
            format!(
                "artifact length changed between preflight and read: {} != {}",
                actual_len, index.file_len
            ),
        );
    }

    let mut selected = Vec::with_capacity(locations.len());
    for location in locations {
        let bytes = usize::try_from(location.data_bytes).map_err(|_| {
            ProbeError::new(
                "selection_bounds",
                "selected tensor is too large for this host",
            )
        })?;
        file.seek(SeekFrom::Start(location.file_offset))
            .map_err(|error| {
                ProbeError::new(
                    "selection_bounds",
                    format!("seek {}: {error}", location.name),
                )
            })?;
        let mut payload = vec![0_u8; bytes];
        file.read_exact(&mut payload).map_err(|error| {
            ProbeError::new(
                "selection_bounds",
                format!("read {} ({} bytes): {error}", location.name, bytes),
            )
        })?;
        let sha256 = sha256_hex(&payload);
        let canonical = aliases
            .get(&location.name)
            .cloned()
            .unwrap_or_else(|| location.name.clone());
        selected.push(SelectedPayload {
            name: location.name,
            canonical,
            dtype: location.dtype,
            shape: location.shape,
            bytes: payload,
            sha256,
        });
    }
    Ok(selected)
}

// =================================================================================================
// Preflight
// =================================================================================================

#[derive(Clone, Debug)]
pub struct Prepared {
    pub manifest: Manifest,
    pub oracle: Oracle,
    pub index: HeaderIndex,
    pub receipt_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub manifest_sha256: String,
    pub oracle_sha256: String,
    pub selected: BTreeMap<String, Vec<String>>,
    /// Per-case preflight refusals. A bad case does not abort the manifest: every case still gets a
    /// receipt (with `payload_reads = 0`) and the test still fails.
    pub case_errors: BTreeMap<String, ProbeError>,
    pub staging_producer: String,
    pub staging_parent_path: PathBuf,
    pub staging_parent_sha256: String,
}

/// Explicit-path form (used by the pure parser/preflight tests, which must not mutate process
/// environment variables from parallel test threads).
pub fn prepare_at(
    backend: &str,
    manifest_path: &Path,
    receipt_dir: &Path,
    load_index: impl FnOnce(&Path) -> PResult<HeaderIndex>,
    resolve: impl Fn(&Case, &HeaderIndex, &Manifest) -> PResult<Vec<String>>,
) -> PResult<Prepared> {
    let manifest_path = ensure_outside_workspace(manifest_path, MANIFEST_ENV)?;
    let receipt_dir = ensure_receipt_dir(receipt_dir)?;
    let manifest_bytes = read_bounded(&manifest_path, MAX_MANIFEST_BYTES, "manifest")?;
    let manifest_sha256 = sha256_hex(&manifest_bytes);
    let manifest = parse_manifest(&manifest_bytes)?;
    if manifest.backend != "both" && manifest.backend != backend {
        return err(
            "manifest_malformed",
            format!(
                "manifest is for backend {:?}, this binary is {backend:?}",
                manifest.backend
            ),
        );
    }

    let artifact_path = ensure_outside_workspace(&manifest.artifact.path, "artifact.path")?;
    let metadata = std::fs::metadata(&artifact_path).map_err(|error| {
        ProbeError::new(
            "artifact_missing",
            format!("cannot stat {}: {error}", artifact_path.display()),
        )
    })?;
    if !metadata.is_file() {
        return err(
            "artifact_not_file",
            format!(
                "artifact path is not a regular file: {}",
                artifact_path.display()
            ),
        );
    }
    if metadata.len() != manifest.artifact.bytes {
        return err(
            "artifact_size_mismatch",
            format!(
                "artifact declares {} bytes, disk has {}",
                manifest.artifact.bytes,
                metadata.len()
            ),
        );
    }

    // Staging trust binding (the 34 GB digest is NOT recomputed).
    let staging_path =
        ensure_outside_workspace(&manifest.staging.receipt_path, "staging.receipt_path")?;
    if !staging_path.is_file() {
        return err(
            "staging_receipt_missing",
            format!("staging receipt not found: {}", staging_path.display()),
        );
    }
    let staging_bytes = read_bounded(&staging_path, MAX_STAGING_BYTES, "staging receipt")
        .map_err(|error| remap_io_error(error, "staging_receipt_missing"))?;
    if sha256_hex(&staging_bytes) != manifest.staging.receipt_sha256 {
        return err(
            "staging_receipt_digest_mismatch",
            "staging receipt bytes do not match staging.receipt_sha256",
        );
    }
    let staging: Value = serde_json::from_slice(&staging_bytes).map_err(|error| {
        ProbeError::new(
            "staging_binding_mismatch",
            format!("staging receipt is not JSON: {error}"),
        )
    })?;
    let staging_map = staging.as_object().ok_or_else(|| {
        ProbeError::new(
            "staging_binding_mismatch",
            "staging receipt must be a JSON object",
        )
    })?;
    if string_field(staging_map, "schema", "staging")? != STAGING_SCHEMA {
        return err(
            "staging_binding_mismatch",
            format!("staging receipt schema must be {STAGING_SCHEMA:?}"),
        );
    }
    if u64_field(staging_map, "schema_version", "staging")? != SCHEMA_VERSION {
        return err(
            "staging_binding_mismatch",
            "staging receipt schema_version mismatch",
        );
    }
    let staging_artifact_path = string_field(staging_map, "artifact_path", "staging")?;
    if canonicalish(Path::new(&staging_artifact_path)) != artifact_path {
        return err(
            "staging_binding_mismatch",
            format!(
                "staging receipt names artifact {:?}, manifest names {}",
                staging_artifact_path,
                artifact_path.display()
            ),
        );
    }
    if require_sha(
        &string_field(staging_map, "artifact_sha256", "staging")?,
        "staging.artifact_sha256",
    )? != manifest.artifact.sha256
        || manifest.staging.verified_artifact_sha256 != manifest.artifact.sha256
    {
        return err(
            "staging_binding_mismatch",
            "staging receipt artifact digest does not match the manifest's verified digest",
        );
    }
    if u64_field(staging_map, "artifact_bytes", "staging")? != manifest.artifact.bytes {
        return err(
            "staging_binding_mismatch",
            "staging receipt artifact byte count does not match the manifest",
        );
    }
    // Provenance: the receipt must state that the campaign's own staging step verified the
    // FULL-FILE digest. This consumer does not re-hash 34 GB; it binds to that verification.
    if !optional_bool(staging_map, "full_file_digest_verified", "staging")? {
        return err(
            "staging_binding_mismatch",
            "staging receipt must declare full_file_digest_verified: true",
        );
    }
    let staging_producer = string_field(staging_map, "producer", "staging")?;
    if staging_producer.trim().is_empty() {
        return err(
            "staging_binding_mismatch",
            "staging receipt must name the staging producer",
        );
    }
    // Provenance to the campaign's own full staging receipt. The projection above is transport
    // only; this binds it to the receipt the staging step wrote after verifying the full file.
    let parent_receipt_path = ensure_outside_workspace(
        &PathBuf::from(string_field(staging_map, "parent_receipt_path", "staging")?),
        "staging.parent_receipt_path",
    )?;
    let parent_receipt_sha256 = require_sha(
        &string_field(staging_map, "parent_receipt_sha256", "staging")?,
        "staging.parent_receipt_sha256",
    )?;
    if !parent_receipt_path.is_file() {
        return err(
            "staging_receipt_missing",
            format!(
                "parent staging receipt not found: {}",
                parent_receipt_path.display()
            ),
        );
    }
    let parent_bytes = read_bounded(
        &parent_receipt_path,
        MAX_PARENT_RECEIPT_BYTES,
        "parent staging receipt",
    )
    .map_err(|error| remap_io_error(error, "staging_receipt_missing"))?;
    if sha256_hex(&parent_bytes) != parent_receipt_sha256 {
        return err(
            "staging_receipt_digest_mismatch",
            "parent staging receipt bytes do not match staging.parent_receipt_sha256",
        );
    }

    // Pinned CPU oracle.
    let oracle_path = ensure_outside_workspace(&manifest.oracle.path, "oracle.path")?;
    if !oracle_path.is_file() {
        return err(
            "oracle_missing",
            format!("oracle not found: {}", oracle_path.display()),
        );
    }
    let oracle_bytes = read_bounded(&oracle_path, MAX_ORACLE_BYTES, "oracle")
        .map_err(|error| remap_io_error(error, "oracle_missing"))?;
    let oracle_sha256 = sha256_hex(&oracle_bytes);
    if oracle_sha256 != manifest.oracle.sha256 {
        return err(
            "oracle_digest_mismatch",
            "oracle bytes do not match oracle.sha256",
        );
    }
    let oracle = parse_oracle(&oracle_bytes)?;
    if oracle.artifact_sha256 != manifest.artifact.sha256
        || manifest.oracle.artifact_sha256 != manifest.artifact.sha256
    {
        return err(
            "oracle_binding_mismatch",
            "oracle is not bound to the manifest's artifact sha256",
        );
    }
    if manifest.oracle.producer.trim().is_empty() {
        return err(
            "oracle_binding_mismatch",
            "oracle.producer must be non-empty",
        );
    }

    let index = load_index(&artifact_path)?;
    if index.file_len != manifest.artifact.bytes {
        return err(
            "artifact_size_mismatch",
            format!(
                "header index reports {} bytes, manifest declares {}",
                index.file_len, manifest.artifact.bytes
            ),
        );
    }

    let mut selected = BTreeMap::new();
    let mut case_errors = BTreeMap::new();
    for case in &manifest.cases {
        let checked = (|| -> PResult<Vec<String>> {
            let names = resolve(case, &index, &manifest)?;
            if names.is_empty() {
                return err(
                    "selection_bounds",
                    format!("case {} resolves to no tensors", case.id),
                );
            }
            if names.len() > manifest.limits.max_selected_tensors {
                return err(
                    "limits_exceeded",
                    format!("case {} selects {} tensors", case.id, names.len()),
                );
            }
            let mut total: u64 = 0;
            let mut unique = BTreeSet::new();
            for name in &names {
                if !unique.insert(name.as_str()) {
                    return err(
                        "selection_bounds",
                        format!("case {} repeats tensor {name:?}", case.id),
                    );
                }
                total = total
                    .checked_add(index.require(name)?.data_bytes)
                    .ok_or_else(|| {
                        ProbeError::new("limits_exceeded", "selected byte count overflow")
                    })?;
            }
            if total > manifest.limits.max_selected_bytes {
                return err(
                    "limits_exceeded",
                    format!(
                        "case {} selects {total} bytes, limit is {}",
                        case.id, manifest.limits.max_selected_bytes
                    ),
                );
            }
            if !case.selected_tensors.is_empty() {
                let declared: BTreeSet<&str> =
                    case.selected_tensors.iter().map(String::as_str).collect();
                let resolved: BTreeSet<&str> = unique.iter().copied().collect();
                if declared != resolved {
                    return err(
                        "selection_bounds",
                        format!(
                            "case {} selected_tensors {:?} does not match the resolved set {:?}",
                            case.id, declared, resolved
                        ),
                    );
                }
            }

            let oracle_case = oracle.cases.get(&case.id).ok_or_else(|| {
                ProbeError::new(
                    "oracle_case_missing",
                    format!("oracle has no case {:?}", case.id),
                )
            })?;
            let oracle_names: BTreeSet<&str> =
                oracle_case.selected.keys().map(String::as_str).collect();
            if oracle_names != unique.iter().copied().collect::<BTreeSet<_>>() {
                return err(
                    "oracle_selection_mismatch",
                    format!(
                        "case {} oracle selection {:?} does not match resolved {:?}",
                        case.id, oracle_names, unique
                    ),
                );
            }
            let (bound_dtype, bound_shape, bound_digest) = input_binding(case)?;
            if oracle_case.input_dtype != bound_dtype
                || oracle_case.input_shape != bound_shape
                || oracle_case.input_values_sha256 != bound_digest
            {
                return err(
                    "input_binding_mismatch",
                    format!(
                        "case {} input binding {:?} {:?} {} does not match the pinned producer's \
                         {:?} {:?} {}",
                        case.id,
                        bound_dtype,
                        bound_shape,
                        bound_digest,
                        oracle_case.input_dtype,
                        oracle_case.input_shape,
                        oracle_case.input_values_sha256
                    ),
                );
            }
            let cap = canonical_tolerance(&case.kind).ok_or_else(|| {
                ProbeError::new(
                    "manifest_malformed",
                    format!(
                        "case {} has no canonical tolerance for {}",
                        case.id, case.kind
                    ),
                )
            })?;
            if oracle_case.tolerance_max_abs > cap {
                return err(
                    "oracle_tolerance_invalid",
                    format!(
                        "case {} oracle tolerance {} exceeds the canonical {cap}",
                        case.id, oracle_case.tolerance_max_abs
                    ),
                );
            }
            if case.tolerance_max_abs > oracle_case.tolerance_max_abs {
                return err(
                    "tolerance_widened",
                    format!(
                        "case {} tolerance {} widens the pinned producer's {}",
                        case.id, case.tolerance_max_abs, oracle_case.tolerance_max_abs
                    ),
                );
            }
            for script in required_producers(&case.kind) {
                if !manifest.oracle.producer_provenance.contains_key(*script) {
                    return err(
                        "oracle_producer_mismatch",
                        format!(
                            "case {} ({}) requires pinned producer {script} in \
                             oracle.producer_provenance",
                            case.id, case.kind
                        ),
                    );
                }
            }
            for name in &names {
                let indexed = index.require(name)?;
                let bound = &oracle_case.selected[name];
                if bound.dtype != indexed.dtype
                    || bound.shape != indexed.shape
                    || bound.bytes != indexed.data_bytes
                {
                    return err(
                        "oracle_selection_mismatch",
                        format!(
                        "case {} tensor {name} header {:?} {:?} {} bytes disagrees with the oracle \
                         binding {:?} {:?} {} bytes",
                        case.id,
                        indexed.dtype,
                        indexed.shape,
                        indexed.data_bytes,
                        bound.dtype,
                        bound.shape,
                        bound.bytes
                    ),
                    );
                }
            }
            Ok(names)
        })();
        match checked {
            Ok(names) => {
                selected.insert(case.id.clone(), names);
            }
            Err(error) => {
                case_errors.insert(case.id.clone(), error);
            }
        }
    }

    Ok(Prepared {
        manifest,
        oracle,
        index,
        receipt_dir,
        manifest_path,
        manifest_sha256,
        oracle_sha256,
        selected,
        case_errors,
        staging_producer,
        staging_parent_path: parent_receipt_path,
        staging_parent_sha256: parent_receipt_sha256,
    })
}

// =================================================================================================
// Execution + receipts
// =================================================================================================

/// Which device the native computation actually ran on.
///
/// Recorded so a reader can distinguish REAL device execution (CUDA ordinal/UUID where available,
/// Metal device, or the MLX default device) from a CPU fallback. A CPU fallback is labeled and is
/// never counted as the device proof.
#[derive(Clone, Debug, Default)]
pub struct DeviceInfo {
    /// `cuda`, `metal`, `mlx-default-gpu`, `mlx-default-cpu`, or `cpu`.
    pub kind: String,
    /// Human-readable identity: candle `DeviceLocation`, MLX `Device` display, CUDA ordinal, ...
    pub label: String,
    pub index: Option<i64>,
    /// `true` on a candle build without the device feature, or an MLX default device that is CPU.
    pub cpu_fallback: bool,
    /// `CUDA_VISIBLE_DEVICES` as observed (candle has no UUID accessor in the pinned fork).
    pub visible_devices: Option<String>,
    pub note: Option<String>,
}

#[derive(Clone, Debug)]
pub struct NativeOutput {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
    /// The route the native loader actually selected (`w4a8`, `convrot`, `affine`, `dense`).
    pub route: Option<String>,
    /// Free-form metadata recorded on the receipt (compute dtype, elapsed note, ...).
    pub detail: Option<String>,
    pub device: DeviceInfo,
}

pub fn run_probe(
    backend: &str,
    load_index: impl FnOnce(&Path) -> PResult<HeaderIndex>,
    resolve: impl Fn(&Case, &HeaderIndex, &Manifest) -> PResult<Vec<String>>,
    compute: impl FnMut(&Case, &Manifest, &[SelectedPayload]) -> PResult<NativeOutput>,
) -> std::result::Result<Vec<Value>, String> {
    let manifest_path = match manifest_path_from_env() {
        Ok(path) => path,
        Err(error) => {
            let mut message = format!(
                "actual-checkpoint probe preflight refused [{}]: {}",
                error.code, error.detail
            );
            if let Err(write_error) = write_manifest_refusal(backend, None, None, &error) {
                message.push_str(&format!(
                    "; refusal receipt could not be written [{}]: {}",
                    write_error.code, write_error.detail
                ));
            }
            return Err(message);
        }
    };
    let receipt_dir = match receipt_dir_from_env() {
        Ok(path) => path,
        Err(error) => {
            return Err(format!(
                "actual-checkpoint probe cannot use {RECEIPT_ENV} [{}]: {}",
                error.code, error.detail
            ))
        }
    };
    run_probe_at(
        backend,
        &manifest_path,
        &receipt_dir,
        load_index,
        resolve,
        compute,
    )
}

pub fn run_probe_at(
    backend: &str,
    manifest_path: &Path,
    receipt_dir: &Path,
    load_index: impl FnOnce(&Path) -> PResult<HeaderIndex>,
    resolve: impl Fn(&Case, &HeaderIndex, &Manifest) -> PResult<Vec<String>>,
    mut compute: impl FnMut(&Case, &Manifest, &[SelectedPayload]) -> PResult<NativeOutput>,
) -> std::result::Result<Vec<Value>, String> {
    let prepared = match prepare_at(backend, manifest_path, receipt_dir, load_index, resolve) {
        Ok(prepared) => prepared,
        Err(error) => {
            let mut message = format!(
                "actual-checkpoint probe preflight refused [{}]: {}",
                error.code, error.detail
            );
            if let Err(write_error) =
                write_manifest_refusal(backend, Some(manifest_path), Some(receipt_dir), &error)
            {
                message.push_str(&format!(
                    "; refusal receipt could not be written [{}]: {}",
                    write_error.code, write_error.detail
                ));
            }
            return Err(message);
        }
    };

    let mut receipts = Vec::with_capacity(prepared.manifest.cases.len());
    let mut failures = Vec::new();
    for case in &prepared.manifest.cases {
        let names = prepared.selected.get(&case.id).cloned().unwrap_or_default();
        let receipt = match execute_case(&prepared, backend, case, &names, &mut compute) {
            Ok(receipt) => receipt,
            Err(error) => {
                return Err(format!(
                    "case {} could not be evaluated or its receipt written [{}]: {}",
                    case.id, error.code, error.detail
                ))
            }
        };
        let pass = receipt["pass"].as_bool().unwrap_or(false);
        if !pass {
            failures.push(format!(
                "case {} [{}]",
                case.id,
                receipt["status"].as_str().unwrap_or("unknown")
            ));
        }
        receipts.push(receipt);
    }
    if failures.is_empty() {
        Ok(receipts)
    } else {
        Err(format!(
            "actual-checkpoint probe failed/refused for {} case(s): {}",
            failures.len(),
            failures.join(", ")
        ))
    }
}

fn execute_case(
    prepared: &Prepared,
    backend: &str,
    case: &Case,
    names: &[String],
    compute: &mut impl FnMut(&Case, &Manifest, &[SelectedPayload]) -> PResult<NativeOutput>,
) -> PResult<Value> {
    if let Some(error) = prepared.case_errors.get(&case.id) {
        // Per-case preflight refusal: a receipt is still written, and no payload byte is read.
        return refusal_receipt(prepared, backend, case, &[], 0, error);
    }
    let selected = match read_selected(
        &prepared.manifest.artifact.path,
        &prepared.index,
        names,
        &case.select.aliases,
        prepared.manifest.limits,
    ) {
        Ok(selected) => selected,
        Err(error) => return refusal_receipt(prepared, backend, case, &[], 0, &error),
    };
    let payload_reads = selected.len();

    let oracle_case = match prepared.oracle.cases.get(&case.id) {
        Some(oracle_case) => oracle_case,
        None => {
            return refusal_receipt(
                prepared,
                backend,
                case,
                &selected,
                payload_reads,
                &ProbeError::new(
                    "oracle_case_missing",
                    "oracle case disappeared after preflight",
                ),
            )
        }
    };
    for payload in &selected {
        match oracle_case.selected.get(&payload.name) {
            Some(bound) if bound.sha256 == payload.sha256 => {}
            Some(bound) => {
                return refusal_receipt(
                    prepared,
                    backend,
                    case,
                    &selected,
                    payload_reads,
                    &ProbeError::new(
                        "tensor_sha_mismatch",
                        format!(
                            "{} selected bytes hash to {}, oracle binds {}",
                            payload.name, payload.sha256, bound.sha256
                        ),
                    ),
                )
            }
            None => {
                return refusal_receipt(
                    prepared,
                    backend,
                    case,
                    &selected,
                    payload_reads,
                    &ProbeError::new(
                        "oracle_selection_mismatch",
                        format!("oracle does not bind {}", payload.name),
                    ),
                )
            }
        }
    }

    let native = match compute(case, &prepared.manifest, &selected) {
        Ok(native) => native,
        Err(error) => {
            return refusal_receipt(prepared, backend, case, &selected, payload_reads, &error)
        }
    };

    let (numeric_pass, observed_max_abs, mut comparison) = compare(
        &native,
        &oracle_case.output_shape,
        &oracle_case.output_values,
        case.tolerance_max_abs,
    );
    if let Some(object) = comparison.as_object_mut() {
        object.insert(
            "max_abs".into(),
            serde_json::Number::from_f64(f64::from(observed_max_abs))
                .map(Value::Number)
                .unwrap_or(Value::Null),
        );
        object.insert("tolerance".into(), json!(case.tolerance_max_abs));
    }
    // A CPU fallback is never the device proof for a numeric case: the campaign must not count it,
    // and the probe itself refuses to report it as a pass.
    let pass = if native.device.cpu_fallback {
        if let Some(object) = comparison.as_object_mut() {
            object.insert(
                "code".into(),
                Value::String("cpu_fallback_not_device_proof".into()),
            );
        }
        false
    } else {
        numeric_pass
    };
    let receipt = receipt_json(
        prepared,
        backend,
        case,
        &selected,
        payload_reads,
        if pass { "pass" } else { "fail" },
        None,
        Some(&native),
        Some(comparison),
        pass,
    );
    write_receipt(prepared, backend, case, &receipt)?;
    Ok(receipt)
}

fn compare(
    native: &NativeOutput,
    oracle_shape: &[usize],
    oracle_values: &[f32],
    tolerance: f32,
) -> (bool, f32, Value) {
    if native.shape != oracle_shape {
        return (
            false,
            f32::INFINITY,
            json!({"code": "shape_mismatch", "native_shape": native.shape, "oracle_shape": oracle_shape}),
        );
    }
    if native.values.len() != oracle_values.len() {
        return (
            false,
            f32::INFINITY,
            json!({
                "code": "shape_mismatch",
                "native_elements": native.values.len(),
                "oracle_elements": oracle_values.len(),
            }),
        );
    }
    if native.values.iter().any(|value| !value.is_finite()) {
        return (
            false,
            f32::INFINITY,
            json!({"code": "non_finite", "side": "native"}),
        );
    }
    if oracle_values.iter().any(|value| !value.is_finite()) {
        return (
            false,
            f32::INFINITY,
            json!({"code": "non_finite", "side": "oracle"}),
        );
    }
    let observed = native
        .values
        .iter()
        .zip(oracle_values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let pass = observed <= tolerance;
    (
        pass,
        observed,
        json!({
            "code": if pass { "within_tolerance" } else { "max_abs_exceeded" },
            "tolerance": tolerance,
        }),
    )
}

fn selected_json(selected: &[SelectedPayload]) -> Value {
    Value::Array(
        selected
            .iter()
            .map(|payload| {
                json!({
                    "name": payload.name,
                    "canonical": payload.canonical,
                    "sha256": payload.sha256,
                    "dtype": payload.dtype,
                    "shape": payload.shape,
                    "bytes": payload.bytes.len(),
                })
            })
            .collect(),
    )
}

fn values_sha256(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    sha256_hex(&bytes)
}

fn peak(values: &[f32]) -> f32 {
    values.iter().map(|value| value.abs()).fold(0.0, f32::max)
}

#[allow(clippy::too_many_arguments)]
fn receipt_json(
    prepared: &Prepared,
    backend: &str,
    case: &Case,
    selected: &[SelectedPayload],
    payload_reads: usize,
    status: &str,
    refusal: Option<&ProbeError>,
    native: Option<&NativeOutput>,
    comparison: Option<Value>,
    pass: bool,
) -> Value {
    let oracle_case = prepared.oracle.cases.get(&case.id);
    let selected_bytes: u64 = selected
        .iter()
        .map(|payload| payload.bytes.len() as u64)
        .sum();
    let kind_coverage = if case.kind == KIND_CONVROT {
        json!({
            "dense_companion": {
                "format_reference": "pinned ConvRot oracle mixedLayer; this probe evaluates packed projections only",
                "actual_dense_comparison_executed": false,
                "standalone_dense_fidelity_claimed": false,
            }
        })
    } else if case.kind == KIND_MIXED {
        json!({
            "same_weights_map": true,
            "concat_order": ["w4a8", "convrot"],
            "dense_companion": {
                "format_reference": "pinned ConvRot oracle mixedLayer; this probe evaluates packed projections only",
                "actual_dense_comparison_executed": false,
                "standalone_dense_fidelity_claimed": false,
            }
        })
    } else {
        Value::Null
    };
    let native_receipt = native.map(|native| {
        json!({
            "shape": native.shape,
            "values": native.values,
            "max_abs": peak(&native.values),
            "values_sha256": values_sha256(&native.values),
            "route": native.route,
            "detail": native.detail,
            "device": {
                "kind": native.device.kind,
                "label": native.device.label,
                "index": native.device.index,
                "cpu_fallback": native.device.cpu_fallback,
                "visible_devices": native.device.visible_devices,
                "note": native.device.note,
            },
            "device_proof": !native.device.cpu_fallback,
        })
    });
    let oracle_receipt = oracle_case.map(|oracle_case| {
        json!({
            "shape": oracle_case.output_shape,
            "values": oracle_case.output_values,
            "max_abs": peak(&oracle_case.output_values),
            "values_sha256": values_sha256(&oracle_case.output_values),
        })
    });
    json!({
        "schema": RECEIPT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "case_id": case.id,
        "kind": case.kind,
        "evidence_scope": if matches!(case.kind.as_str(), KIND_W4A8 | KIND_CONVROT | KIND_MIXED) {
            "selected_packed_projection_comparison_only"
        } else {
            "selected_pruned_table_or_reduced_projection_only"
        },
        "qkv_ffn_tensor_mapping_evidence": false,
        "av_generation_evidence": false,
        "kind_coverage": kind_coverage,
        "input_dtype": case.input_dtype,
        "status": status,
        "pass": pass,
        "refusal": refusal.map(|error| json!({"code": error.code, "detail": error.detail})),
        "manifest_path": prepared.manifest_path,
        "manifest_sha256": prepared.manifest_sha256,
        "oracle_path": prepared.manifest.oracle.path,
        "oracle_sha256": prepared.oracle_sha256,
        "oracle_producer": prepared.manifest.oracle.producer,
        "oracle_producer_version": prepared.manifest.oracle.producer_version,
        "oracle_producer_provenance": prepared.manifest.oracle.producer_provenance,
        "receipt_dir": prepared.receipt_dir,
        "payload_reads": payload_reads,
        "artifact": {
            "path": prepared.manifest.artifact.path,
            "sha256": prepared.manifest.artifact.sha256,
            "bytes": prepared.manifest.artifact.bytes,
            "header_bytes_read": prepared.index.header_bytes_read,
            "digest_trust": "staging_verified_digest_bound_by_sha256_receipt",
            "whole_file_digest_recomputed": false,
            "staging_receipt_path": prepared.manifest.staging.receipt_path,
            "staging_receipt_sha256": prepared.manifest.staging.receipt_sha256,
            "staging_producer": prepared.staging_producer,
            "staging_parent_receipt_path": prepared.staging_parent_path,
            "staging_parent_receipt_sha256": prepared.staging_parent_sha256,
            "scope_note": "artifact and receipts must also live outside the SceneWorks checkout; \
                           this consumer enforces outside the Inference workspace",
        },
        "selected": selected_json(selected),
        "selection_topology_requested": case.select.topology,
        "declared_dit_geometry": {
            "explicit": prepared.manifest.dit_explicit,
            "hidden_size": prepared.manifest.dit.hidden_size,
            "num_layers": prepared.manifest.dit.num_layers,
            "time_embed_dim": prepared.manifest.dit.time_embed_dim,
        },
        "bounds": {
            "selected_tensors": selected.len(),
            "selected_bytes": selected_bytes,
            "hard_max_selected_tensors": HARD_MAX_SELECTED_TENSORS,
            "hard_max_selected_bytes": HARD_MAX_SELECTED_BYTES,
            "hard_max_cases": HARD_MAX_CASES,
            "materialization_note": "native materialization is bounded by selected_bytes and at \
                                     most a 4x f32 widening of the widest stored element width",
        },
        "native": native_receipt,
        "oracle": oracle_receipt,
        "tolerance": {"max_abs": case.tolerance_max_abs},
        "comparison": comparison,
    })
}

fn refusal_receipt(
    prepared: &Prepared,
    backend: &str,
    case: &Case,
    selected: &[SelectedPayload],
    payload_reads: usize,
    error: &ProbeError,
) -> PResult<Value> {
    let receipt = receipt_json(
        prepared,
        backend,
        case,
        selected,
        payload_reads,
        "refused",
        Some(error),
        None,
        None,
        false,
    );
    write_receipt(prepared, backend, case, &receipt)?;
    Ok(receipt)
}

pub fn receipt_file_name(backend: &str, case_id: &str) -> String {
    let safe: String = case_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let digest = sha256_hex(case_id.as_bytes());
    format!("{backend}.{safe}.{}.json", &digest[..8])
}

fn write_json(receipt_dir: &Path, name: &str, value: &Value) -> PResult<()> {
    let path = receipt_dir.join(name);
    let text = serde_json::to_string_pretty(value).map_err(|error| {
        ProbeError::new(
            "receipt_write_failed",
            format!("cannot serialize receipt {}: {error}", path.display()),
        )
    })?;
    std::fs::write(&path, format!("{text}\n")).map_err(|error| {
        ProbeError::new(
            "receipt_write_failed",
            format!("cannot write receipt {}: {error}", path.display()),
        )
    })
}

fn write_receipt(prepared: &Prepared, backend: &str, case: &Case, receipt: &Value) -> PResult<()> {
    write_json(
        &prepared.receipt_dir,
        &receipt_file_name(backend, &case.id),
        receipt,
    )
}

/// Best-effort refusal receipt when preflight itself could not produce a `Prepared`. The receipt
/// directory comes only from the environment, so a malformed/unreadable manifest still has a
/// destination when the directory is usable.
fn write_manifest_refusal(
    backend: &str,
    manifest_path: Option<&Path>,
    receipt_dir: Option<&Path>,
    error: &ProbeError,
) -> PResult<()> {
    let receipt_dir = match receipt_dir {
        Some(path) => ensure_receipt_dir(path)?,
        None => receipt_dir_from_env()?,
    };
    let manifest_path = manifest_path
        .map(|path| path.to_string_lossy().into_owned())
        .or_else(|| {
            std::env::var_os(MANIFEST_ENV)
                .map(PathBuf::from)
                .map(|path| path.to_string_lossy().into_owned())
        });
    let receipt = json!({
        "schema": RECEIPT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "case_id": null,
        "kind": null,
        "status": "refused",
        "pass": false,
        "refusal": {"code": error.code, "detail": error.detail},
        "manifest_path": manifest_path.clone(),
        "receipt_dir": receipt_dir.clone(),
        "payload_reads": 0,
        "whole_file_digest_recomputed": false,
    });
    write_json(
        &receipt_dir,
        &format!("{backend}.manifest-refusal.json"),
        &receipt,
    )?;
    // If the manifest itself parsed, every case still gets its own receipt.
    if let Some(path) = manifest_path.as_deref().map(PathBuf::from) {
        if let Ok(bytes) = read_bounded(&path, MAX_MANIFEST_BYTES, "manifest") {
            if let Ok(manifest) = parse_manifest(&bytes) {
                for case in &manifest.cases {
                    let case_receipt = json!({
                        "schema": RECEIPT_SCHEMA,
                        "schema_version": SCHEMA_VERSION,
                        "backend": backend,
                        "case_id": case.id,
                        "kind": case.kind,
                        "status": "refused",
                        "pass": false,
                        "refusal": {"code": error.code, "detail": error.detail},
                        "manifest_path": path.to_string_lossy(),
                        "receipt_dir": receipt_dir,
                        "payload_reads": 0,
                        "whole_file_digest_recomputed": false,
                    });
                    write_json(
                        &receipt_dir,
                        &receipt_file_name(backend, &case.id),
                        &case_receipt,
                    )?;
                }
            }
        }
    }
    Ok(())
}

// =================================================================================================
// Tiny-fixture helpers for the unignored parser/preflight tests.
// =================================================================================================

/// Write a minimal, spec-valid safetensors file from `(name, dtype, shape, payload)` entries.
///
/// Test-only: this is how the pure parser/reader tests exercise real header parsing without a
/// model, a payload of meaningful size, or an accelerator.
pub fn write_tiny_safetensors(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = Map::new();
    let mut offset = 0usize;
    for (name, dtype, shape, payload) in tensors {
        header.insert(
            (*name).to_owned(),
            json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + payload.len()],
            }),
        );
        offset += payload.len();
    }
    let mut header = serde_json::to_vec(&header).expect("header json");
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    for (_, _, _, payload) in tensors {
        file.extend(payload);
    }
    std::fs::write(path, file).expect("write tiny safetensors");
}

pub fn f32_le(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub fn staging_receipt_json(
    artifact_path: &Path,
    artifact_sha256: &str,
    bytes: u64,
    parent_receipt_path: &Path,
    parent_receipt_sha256: &str,
) -> Vec<u8> {
    serde_json::to_vec_pretty(&json!({
        "schema": STAGING_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "producer": "h3-finetune-stage-receipt projection",
        "full_file_digest_verified": true,
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": bytes,
        "parent_receipt_path": parent_receipt_path,
        "parent_receipt_sha256": parent_receipt_sha256,
    }))
    .expect("staging receipt json")
}

pub fn sha256_hex_bytes(bytes: &[u8]) -> String {
    sha256_hex(bytes)
}
