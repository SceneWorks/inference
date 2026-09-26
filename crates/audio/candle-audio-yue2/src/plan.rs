//! Exact symbolic plans — a native port of upstream `yue2.pipeline.SymbolicPlan` (sc-22990).
//!
//! A plan is the ABC stage's output as the semantic stage consumes it: the request, the ABC score
//! text, its **exact** token IDs and the exact positive prefix built from them. Re-encoding a
//! plan's decoded text does not in general reproduce the sampled IDs, so a plan is saved and
//! restored as IDs, never re-tokenized.
//!
//! * [`SymbolicPlan::prepare`] follows upstream `pipeline.plan`: `cot = off` and external ABC
//!   produce a plan directly; otherwise it returns the ABC planner's prefix
//!   ([`PlanStep::GenerateAbc`]) and [`AbcPlanning::finish`] turns the sampled IDs into the plan.
//! * [`SymbolicPlan::semantic_conditioning`] re-derives the prefix from the request and the exact
//!   ABC IDs and refuses a plan whose prefix disagrees (upstream `generate_semantic`), then builds
//!   the CFG negative prefix and checks the generation budget.
//! * [`SymbolicPlan::save`] writes upstream's layout — `plan.json`, `abc_tokens.npy`,
//!   `prefix.npy` (int32), `score.abc` when there is a score, and `plan_manifest.json` with every
//!   file's SHA-256 — so plans move between this runtime and the reference in both directions.
//!   It refuses to write over an existing plan: a saved plan is never changed in place.
//! * [`SymbolicPlan::restore`] is upstream `SymbolicPlan.load` plus the checks upstream defers to
//!   generation time: every manifest hash, token arrays equal to `plan.json`, the score file equal
//!   to the recorded ABC, the request re-validated, the ABC text consistent with its IDs, and the
//!   prefix equal to the one the request and IDs produce. An edited `score.abc` (with or without a
//!   recomputed manifest) is refused — **an edited ABC is a new request**: derive it with
//!   [`crate::protocol::SongRequest::with_abc`] and plan it again.
//!
//! # Identity
//!
//! [`SymbolicPlan::identity`] is a SHA-256 over exactly what the semantic stage is conditioned on:
//! the protocol version, every request field, the ABC text, the exact ABC token IDs, the exact
//! prefix token IDs and the truncation flag (timings are measurements and are excluded). The
//! manifest alone cannot stop a directory whose files were *all* rewritten consistently — nothing
//! without a secret can — so a caller that must prove a plan is the one it saved keeps the identity
//! [`SymbolicPlan::save`] returned outside the plan directory and restores with
//! [`SymbolicPlan::restore_expecting`].

use std::fmt;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::protocol::{
    check_generation_budget, negative_prefix, token_prefixes, CotMode, ProtocolError, Sampling,
    SongRequest, PROTOCOL_VERSION,
};
use crate::tokenizer::Yue2TextTokenizer;

/// `plan.json`.
pub const PLAN_JSON: &str = "plan.json";
/// `abc_tokens.npy` — the exact ABC token IDs.
pub const ABC_TOKENS_NPY: &str = "abc_tokens.npy";
/// `prefix.npy` — the exact positive prefix.
pub const PREFIX_NPY: &str = "prefix.npy";
/// `score.abc` — the ABC text, UTF-8 (only when the plan has a score).
pub const SCORE_ABC: &str = "score.abc";
/// `plan_manifest.json` — `{file: sha256}` over the other files.
pub const PLAN_MANIFEST: &str = "plan_manifest.json";

const REQUIRED: [&str; 3] = [PLAN_JSON, ABC_TOKENS_NPY, PREFIX_NPY];
const ALLOWED: [&str; 4] = [PLAN_JSON, ABC_TOKENS_NPY, PREFIX_NPY, SCORE_ABC];

/// Every way building, saving or restoring a plan fails.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    /// A protocol refusal (request, prefix or budget).
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
    /// Reading or writing a plan file failed.
    #[error("{}: {source}", path.display())]
    Io {
        /// The file.
        path: PathBuf,
        /// The I/O error.
        source: std::io::Error,
    },
    /// The directory already holds plan artifacts; a saved plan is never changed in place.
    #[error("{} already exists; save each plan to a fresh directory", path.display())]
    AlreadySaved {
        /// The existing artifact.
        path: PathBuf,
    },
    /// The manifest is missing required files or names a file a plan never has, or a listed
    /// file is a symlink or not a regular file.
    #[error("invalid saved plan: {0}")]
    InvalidArtifact(String),
    /// A file's bytes differ from its manifest hash.
    #[error(
        "saved plan changed ({file}); supply modified ABC as an external planner input \
         (SongRequest::with_abc) instead of editing a saved plan"
    )]
    Changed {
        /// The changed file.
        file: String,
    },
    /// A file does not parse.
    #[error("{file}: {detail}")]
    Malformed {
        /// The file.
        file: &'static str,
        /// What is wrong.
        detail: String,
    },
    /// A token array disagrees with `plan.json`.
    #[error("saved plan token array mismatch: {0} differs from plan.json")]
    TokenArrayMismatch(&'static str),
    /// `score.abc` disagrees with the recorded ABC text (or exists without one).
    #[error("saved ABC text mismatch: {0}")]
    AbcTextMismatch(&'static str),
    /// The plan's parts disagree with each other or with the request.
    #[error("inconsistent plan: {0}")]
    Inconsistent(String),
    /// The restored plan is not the one the caller saved.
    #[error("plan identity {actual} is not the expected {expected}; this is a different plan")]
    IdentityMismatch {
        /// The identity the caller kept.
        expected: PlanIdentity,
        /// The restored plan's identity.
        actual: PlanIdentity,
    },
}

/// SHA-256 identity of a plan's semantic content (see the [module docs](self)).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlanIdentity([u8; 32]);

impl PlanIdentity {
    /// The raw digest.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for PlanIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex(&self.0))
    }
}

impl fmt::Debug for PlanIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PlanIdentity({self})")
    }
}

/// Where a plan's ABC came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanKind {
    /// `cot = off`: no symbolic plan.
    Off,
    /// The request supplied the ABC; its IDs are the encoded external text.
    External,
    /// The ABC stage sampled the IDs; the text is their decoding.
    Generated,
}

/// The ABC planner's input for a request that needs a sampled plan.
#[derive(Clone, Debug, PartialEq)]
pub struct AbcPlanning {
    request: SongRequest,
    prefix: Vec<u32>,
}

impl AbcPlanning {
    /// The request being planned.
    pub fn request(&self) -> &SongRequest {
        &self.request
    }

    /// The planner prefix (`[EOD] + text + [ABC_START]`); sample with the ABC phase's
    /// [`Sampling`] until `ABC_END`, after
    /// `check_generation_budget(prefix.len(), None, sampling, 1.0)`.
    pub fn prefix(&self) -> &[u32] {
        &self.prefix
    }

    /// The plan from the sampled ABC IDs (the output tokens, excluding `ABC_END`). `timing` is
    /// recorded as-is; `truncated` says the phase stopped at `max_tokens`.
    pub fn finish(
        self,
        tokenizer: &Yue2TextTokenizer,
        abc_ids: Vec<u32>,
        timing: Map<String, Value>,
        truncated: bool,
    ) -> Result<SymbolicPlan, PlanError> {
        let prefix = token_prefixes(&self.request, tokenizer, Some(&abc_ids))?;
        Ok(SymbolicPlan {
            abc: Some(tokenizer.decode(&abc_ids)),
            request: self.request,
            abc_ids,
            prefix,
            timing,
            truncated,
        })
    }
}

/// What [`SymbolicPlan::prepare`] needs next.
#[derive(Clone, Debug, PartialEq)]
pub enum PlanStep {
    /// The plan needs no sampling (`cot = off`, or external ABC).
    Ready(SymbolicPlan),
    /// Sample the ABC from this prefix, then call [`AbcPlanning::finish`].
    GenerateAbc(AbcPlanning),
}

/// The semantic stage's exact conditioning, derived from a verified plan.
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticConditioning {
    /// The positive prefix (the plan's exact prefix).
    pub positive: Vec<u32>,
    /// The CFG negative prefix, present exactly when `cfg_scale != 1`.
    pub negative: Option<Vec<u32>>,
    /// The effective CFG scale.
    pub cfg_scale: f64,
    /// `cot = off` (upstream's historical BF16 arithmetic for that mode).
    pub legacy_off: bool,
}

/// An exact symbolic plan (see the [module docs](self)). Fields are private: a plan comes only
/// from [`SymbolicPlan::prepare`] / [`AbcPlanning::finish`] or a verified
/// [`SymbolicPlan::restore`], so its ABC can never be swapped behind its IDs.
#[derive(Clone, Debug, PartialEq)]
pub struct SymbolicPlan {
    request: SongRequest,
    abc: Option<String>,
    abc_ids: Vec<u32>,
    prefix: Vec<u32>,
    timing: Map<String, Value>,
    truncated: bool,
}

impl SymbolicPlan {
    /// Start planning `request` (upstream `pipeline.plan`).
    pub fn prepare(
        request: SongRequest,
        tokenizer: &Yue2TextTokenizer,
    ) -> Result<PlanStep, PlanError> {
        if request.cot() == CotMode::Off {
            let prefix = token_prefixes(&request, tokenizer, None)?;
            return Ok(PlanStep::Ready(SymbolicPlan {
                request,
                abc: None,
                abc_ids: Vec::new(),
                prefix,
                timing: Map::new(),
                truncated: false,
            }));
        }
        if let Some(abc) = request.abc() {
            let abc = abc.to_string();
            let abc_ids = tokenizer.encode(&abc).map_err(ProtocolError::from)?;
            let prefix = token_prefixes(&request, tokenizer, Some(&abc_ids))?;
            let mut timing = Map::new();
            timing.insert("seconds".into(), Value::from(0.0));
            timing.insert("output_tokens".into(), Value::from(0));
            timing.insert("external_prefix_tokens".into(), Value::from(abc_ids.len()));
            return Ok(PlanStep::Ready(SymbolicPlan {
                request,
                abc: Some(abc),
                abc_ids,
                prefix,
                timing,
                truncated: false,
            }));
        }
        let prefix = token_prefixes(&request, tokenizer, None)?;
        Ok(PlanStep::GenerateAbc(AbcPlanning { request, prefix }))
    }

    /// The request.
    pub fn request(&self) -> &SongRequest {
        &self.request
    }
    /// The ABC score text (`None` for `cot = off`).
    pub fn abc(&self) -> Option<&str> {
        self.abc.as_deref()
    }
    /// The exact ABC token IDs.
    pub fn abc_ids(&self) -> &[u32] {
        &self.abc_ids
    }
    /// The exact positive prefix.
    pub fn prefix(&self) -> &[u32] {
        &self.prefix
    }
    /// The recorded timing object.
    pub fn timing(&self) -> &Map<String, Value> {
        &self.timing
    }
    /// Whether the ABC phase stopped at its token limit.
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// Where the ABC came from.
    pub fn kind(&self) -> PlanKind {
        if self.request.cot() == CotMode::Off {
            PlanKind::Off
        } else if self.request.abc().is_some() {
            PlanKind::External
        } else {
            PlanKind::Generated
        }
    }

    /// The plan's identity (see the [module docs](self)).
    pub fn identity(&self) -> PlanIdentity {
        let mut h = Sha256::new();
        let mut field = |bytes: &[u8]| {
            h.update((bytes.len() as u64).to_le_bytes());
            h.update(bytes);
        };
        let r = &self.request;
        field(b"yue2-symbolic-plan");
        field(PROTOCOL_VERSION.as_bytes());
        field(r.style().as_bytes());
        field(r.lyrics().as_bytes());
        field(r.cot().as_str().as_bytes());
        field(&r.seed().to_le_bytes());
        field(&optional(r.abc().map(str::as_bytes)));
        field(&optional(
            r.cfg_scale()
                .map(|s| s.to_bits().to_le_bytes())
                .as_ref()
                .map(|b| &b[..]),
        ));
        field(r.id().as_bytes());
        field(&optional(self.abc.as_deref().map(str::as_bytes)));
        field(&ids_le(&self.abc_ids));
        field(&ids_le(&self.prefix));
        field(&[self.truncated as u8]);
        PlanIdentity(h.finalize().into())
    }

    /// Verify the plan against its request (the prefix must be the one the request and exact ABC
    /// IDs produce) and build the semantic stage's conditioning, checking the budget for
    /// `sampling` (upstream `generate_semantic` up to prefill).
    pub fn semantic_conditioning(
        &self,
        tokenizer: &Yue2TextTokenizer,
        sampling: &Sampling,
    ) -> Result<SemanticConditioning, PlanError> {
        let expected = token_prefixes(&self.request, tokenizer, Some(&self.abc_ids))?;
        if expected != self.prefix {
            return Err(PlanError::Inconsistent(
                "plan prefix disagrees with the request and exact ABC IDs".into(),
            ));
        }
        let cfg_scale = self.request.guidance();
        let negative = if cfg_scale != 1.0 {
            Some(negative_prefix(
                &self.request,
                tokenizer,
                Some(&self.abc_ids),
            )?)
        } else {
            None
        };
        check_generation_budget(
            self.prefix.len(),
            negative.as_ref().map(Vec::len),
            sampling,
            cfg_scale,
        )?;
        Ok(SemanticConditioning {
            positive: self.prefix.clone(),
            negative,
            cfg_scale,
            legacy_off: self.request.cot() == CotMode::Off,
        })
    }

    /// Check that the plan's parts agree: the right shape for its [`PlanKind`], the ABC text
    /// consistent with the exact IDs, and the prefix the request and IDs produce.
    fn verify(&self, tokenizer: &Yue2TextTokenizer) -> Result<(), PlanError> {
        let inconsistent = |detail: &str| Err(PlanError::Inconsistent(detail.into()));
        match self.kind() {
            PlanKind::Off => {
                if self.abc.is_some() || !self.abc_ids.is_empty() {
                    return inconsistent("cot=off plan carries an ABC score");
                }
            }
            PlanKind::External => {
                if self.abc.as_deref() != self.request.abc() {
                    return inconsistent("ABC text is not the request's external ABC");
                }
                let encoded = self
                    .request
                    .abc()
                    .map(|abc| tokenizer.encode(abc))
                    .transpose()
                    .map_err(ProtocolError::from)?;
                if encoded.as_deref() != Some(&self.abc_ids[..]) {
                    return inconsistent("ABC IDs are not the encoded external ABC");
                }
            }
            PlanKind::Generated => {
                if self.abc.as_deref() != Some(tokenizer.decode(&self.abc_ids).as_str()) {
                    return inconsistent(
                        "ABC text is not the decoding of the exact ABC IDs (an edited score is \
                         a new request: SongRequest::with_abc)",
                    );
                }
            }
        }
        if token_prefixes(&self.request, tokenizer, Some(&self.abc_ids))? != self.prefix {
            return inconsistent("prefix disagrees with the request and exact ABC IDs");
        }
        Ok(())
    }

    /// Save to `dir` in upstream's layout and return the plan's identity. Refuses a directory
    /// that already holds any plan artifact.
    pub fn save(&self, dir: &Path) -> Result<PlanIdentity, PlanError> {
        fs::create_dir_all(dir).map_err(|source| io(dir, source))?;
        for name in ALLOWED.iter().chain([&PLAN_MANIFEST]) {
            let path = dir.join(name);
            if fs::symlink_metadata(&path).is_ok() {
                return Err(PlanError::AlreadySaved { path });
            }
        }
        let mut files: Vec<(&str, Vec<u8>)> = Vec::new();
        if let Some(abc) = &self.abc {
            files.push((SCORE_ABC, abc.as_bytes().to_vec()));
        }
        files.push((ABC_TOKENS_NPY, npy_int32(&self.abc_ids)?));
        files.push((PREFIX_NPY, npy_int32(&self.prefix)?));
        let mut plan = Map::new();
        plan.insert("request".into(), self.request.to_json());
        plan.insert("timing".into(), Value::Object(self.timing.clone()));
        plan.insert("truncated".into(), Value::Bool(self.truncated));
        plan.insert("prefix".into(), Value::from(self.prefix.clone()));
        plan.insert("abc_ids".into(), Value::from(self.abc_ids.clone()));
        plan.insert(
            "abc".into(),
            self.abc.clone().map_or(Value::Null, Value::String),
        );
        files.push((PLAN_JSON, json_bytes(&Value::Object(plan))?));
        let manifest: Map<String, Value> = files
            .iter()
            .map(|(name, bytes)| (name.to_string(), Value::String(hex(&Sha256::digest(bytes)))))
            .collect();
        files.push((PLAN_MANIFEST, json_bytes(&Value::Object(manifest))?));
        for (name, bytes) in &files {
            write_atomic(&dir.join(name), bytes)?;
        }
        Ok(self.identity())
    }

    /// Restore a saved plan with every integrity and consistency check (see the
    /// [module docs](self)). The exact token IDs come from the saved arrays; nothing is
    /// re-tokenized from the score text.
    pub fn restore(dir: &Path, tokenizer: &Yue2TextTokenizer) -> Result<Self, PlanError> {
        let manifest_path = dir.join(PLAN_MANIFEST);
        let manifest = parse_json(PLAN_MANIFEST, &read(&manifest_path)?)?;
        let manifest = manifest
            .as_object()
            .ok_or_else(|| malformed(PLAN_MANIFEST, "not an object"))?;
        if !REQUIRED.iter().all(|name| manifest.contains_key(*name)) {
            return Err(PlanError::InvalidArtifact("incomplete saved plan".into()));
        }
        // Each listed file is read exactly once: the bytes hashed are the bytes parsed.
        let mut files: Vec<(&str, Vec<u8>)> = Vec::new();
        for (name, digest) in manifest {
            let Some(&name) = ALLOWED.iter().find(|a| **a == name.as_str()) else {
                return Err(PlanError::InvalidArtifact(format!(
                    "`{name}` is not a plan artifact"
                )));
            };
            let path = dir.join(name);
            let meta = fs::symlink_metadata(&path).map_err(|source| io(&path, source))?;
            if !meta.file_type().is_file() {
                return Err(PlanError::InvalidArtifact(format!(
                    "`{name}` is a symlink or not a regular file"
                )));
            }
            let digest = digest.as_str().ok_or_else(|| {
                malformed(PLAN_MANIFEST, format!("`{name}` hash is not a string"))
            })?;
            let bytes = read(&path)?;
            if hex(&Sha256::digest(&bytes)) != digest {
                return Err(PlanError::Changed {
                    file: name.to_string(),
                });
            }
            files.push((name, bytes));
        }
        let file = |name: &str| {
            files
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, b)| b.as_slice())
        };
        let required = |name: &str| {
            file(name).ok_or_else(|| PlanError::InvalidArtifact("incomplete saved plan".into()))
        };

        let data = parse_json(PLAN_JSON, required(PLAN_JSON)?)?;
        let field = |key: &str| {
            data.get(key)
                .ok_or_else(|| malformed(PLAN_JSON, format!("missing `{key}`")))
        };
        let abc_ids = json_ids(field("abc_ids")?, "abc_ids")?;
        let prefix = json_ids(field("prefix")?, "prefix")?;
        for (name, ids) in [(ABC_TOKENS_NPY, &abc_ids), (PREFIX_NPY, &prefix)] {
            let array = read_npy_ints(name, required(name)?)?;
            if array.len() != ids.len() || array.iter().zip(ids).any(|(a, &b)| *a != b as i128) {
                return Err(PlanError::TokenArrayMismatch(name));
            }
        }
        let abc = match field("abc")? {
            Value::Null => None,
            Value::String(s) => Some(s.clone()),
            _ => return Err(malformed(PLAN_JSON, "`abc` is neither null nor a string")),
        };
        match (&abc, file(SCORE_ABC)) {
            (Some(_), None) => {
                return Err(PlanError::AbcTextMismatch(
                    "score.abc is not in the manifest",
                ))
            }
            (Some(text), Some(score)) if score != text.as_bytes() => {
                return Err(PlanError::AbcTextMismatch(
                    "score.abc differs from plan.json",
                ))
            }
            (None, Some(_)) => {
                return Err(PlanError::AbcTextMismatch(
                    "score.abc is present but the plan records no ABC",
                ))
            }
            _ => {}
        }
        let timing = field("timing")?
            .as_object()
            .cloned()
            .ok_or_else(|| malformed(PLAN_JSON, "`timing` is not an object"))?;
        let truncated = field("truncated")?
            .as_bool()
            .ok_or_else(|| malformed(PLAN_JSON, "`truncated` is not a boolean"))?;
        let request = SongRequest::from_json(field("request")?)?;
        let plan = SymbolicPlan {
            request,
            abc,
            abc_ids,
            prefix,
            timing,
            truncated,
        };
        plan.verify(tokenizer)?;
        Ok(plan)
    }

    /// [`SymbolicPlan::restore`], then refuse any plan whose identity is not `expected` (the
    /// identity [`SymbolicPlan::save`] returned, kept outside the plan directory).
    pub fn restore_expecting(
        dir: &Path,
        tokenizer: &Yue2TextTokenizer,
        expected: &PlanIdentity,
    ) -> Result<Self, PlanError> {
        let plan = Self::restore(dir, tokenizer)?;
        let actual = plan.identity();
        if actual != *expected {
            return Err(PlanError::IdentityMismatch {
                expected: *expected,
                actual,
            });
        }
        Ok(plan)
    }
}

fn optional(value: Option<&[u8]>) -> Vec<u8> {
    match value {
        None => vec![0],
        Some(bytes) => [&[1u8][..], bytes].concat(),
    }
}

fn ids_le(ids: &[u32]) -> Vec<u8> {
    ids.iter().flat_map(|id| id.to_le_bytes()).collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn io(path: &Path, source: std::io::Error) -> PlanError {
    PlanError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn malformed(file: &'static str, detail: impl Into<String>) -> PlanError {
    PlanError::Malformed {
        file,
        detail: detail.into(),
    }
}

fn read(path: &Path) -> Result<Vec<u8>, PlanError> {
    fs::read(path).map_err(|source| io(path, source))
}

fn parse_json(file: &'static str, bytes: &[u8]) -> Result<Value, PlanError> {
    serde_json::from_slice(bytes).map_err(|e| malformed(file, e.to_string()))
}

fn json_bytes(value: &Value) -> Result<Vec<u8>, PlanError> {
    let mut bytes =
        serde_json::to_vec_pretty(value).map_err(|e| malformed(PLAN_JSON, e.to_string()))?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// A JSON array of token IDs (non-negative integers that fit `u32`).
fn json_ids(value: &Value, field: &'static str) -> Result<Vec<u32>, PlanError> {
    value
        .as_array()
        .ok_or_else(|| malformed(PLAN_JSON, format!("`{field}` is not an array")))?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| {
                    malformed(PLAN_JSON, format!("`{field}` holds a non-token value {v}"))
                })
        })
        .collect()
}

/// Write via a sibling temporary file and a rename, as upstream's `write_json` does.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), PlanError> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
    let tmp = path.with_file_name(format!(
        "{}.{}.tmp",
        name.as_deref().unwrap_or("plan"),
        std::process::id()
    ));
    let mut file = fs::File::create(&tmp).map_err(|source| io(&tmp, source))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|source| io(&tmp, source))?;
    fs::rename(&tmp, path).map_err(|source| io(path, source))
}

/// A 1-D little-endian int32 `.npy` (format 1.0), byte-identical to `np.save` of
/// `np.asarray(ids, dtype=np.int32)`.
fn npy_int32(ids: &[u32]) -> Result<Vec<u8>, PlanError> {
    let mut header = format!(
        "{{'descr': '<i4', 'fortran_order': False, 'shape': ({},), }}",
        ids.len()
    );
    // magic (6) + version (2) + header length (2) + header, padded with spaces and a final
    // newline to a multiple of 64 bytes.
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat(unpadded.next_multiple_of(64) - unpadded));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + ids.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for &id in ids {
        let id = i32::try_from(id)
            .map_err(|_| malformed(ABC_TOKENS_NPY, format!("token {id} does not fit int32")))?;
        out.extend_from_slice(&id.to_le_bytes());
    }
    Ok(out)
}

/// Read a 1-D integer `.npy` (any signed/unsigned width, either byte order; format 1.0–3.0), the
/// arrays upstream's `np.load(..., allow_pickle=False)` accepts for a plan.
fn read_npy_ints(file: &'static str, bytes: &[u8]) -> Result<Vec<i128>, PlanError> {
    let bad = |detail: &str| malformed(file, detail.to_string());
    let rest = bytes
        .strip_prefix(b"\x93NUMPY")
        .ok_or_else(|| bad("not a .npy file"))?;
    let (header_len, rest) = match rest {
        [1, _, a, b, rest @ ..] => (u16::from_le_bytes([*a, *b]) as usize, rest),
        [2 | 3, _, a, b, c, d, rest @ ..] => (u32::from_le_bytes([*a, *b, *c, *d]) as usize, rest),
        _ => return Err(bad("unsupported .npy version")),
    };
    if rest.len() < header_len {
        return Err(bad("truncated header"));
    }
    let (header, data) = rest.split_at(header_len);
    let header = std::str::from_utf8(header).map_err(|_| bad("header is not text"))?;
    let descr = dict_value(header, "descr").ok_or_else(|| bad("header has no descr"))?;
    let descr = descr
        .strip_prefix('\'')
        .and_then(|d| d.strip_suffix('\''))
        .ok_or_else(|| bad("descr is not a string"))?;
    let (order, kind, width) = match descr.as_bytes() {
        [o @ (b'<' | b'>' | b'|'), k @ (b'i' | b'u'), w @ (b'1' | b'2' | b'4' | b'8')] => {
            (*o, *k, (*w - b'0') as usize)
        }
        _ => return Err(bad("array is not integer-typed")),
    };
    if order == b'|' && width != 1 {
        return Err(bad("multi-byte dtype without byte order"));
    }
    if !matches!(dict_value(header, "fortran_order"), Some("False" | "True")) {
        return Err(bad("header has no fortran_order"));
    }
    let shape = dict_value(header, "shape").ok_or_else(|| bad("header has no shape"))?;
    let dims: Vec<&str> = shape
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| bad("shape is not a tuple"))?
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .collect();
    let [len] = dims[..] else {
        return Err(bad("array is not one-dimensional"));
    };
    let len: usize = len.parse().map_err(|_| bad("shape is not an integer"))?;
    if data.len() != len * width {
        return Err(bad("data length disagrees with the shape"));
    }
    Ok(data
        .chunks_exact(width)
        .map(|chunk| {
            let mut le = [0u8; 16];
            for (i, &b) in chunk.iter().enumerate() {
                le[if order == b'>' { width - 1 - i } else { i }] = b;
            }
            let unsigned = u128::from_le_bytes(le);
            if kind == b'i' && unsigned >> (width * 8 - 1) & 1 == 1 {
                unsigned as i128 - (1i128 << (width * 8))
            } else {
                unsigned as i128
            }
        })
        .collect())
}

/// The raw text of `key`'s value in a `.npy` header dict (`{'key': value, ...}`).
fn dict_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let start = header.find(&format!("'{key}':"))? + key.len() + 3;
    let rest = header[start..].trim_start();
    let end = if rest.starts_with('(') {
        rest.find(')')? + 1
    } else if let Some(quoted) = rest.strip_prefix('\'') {
        quoted.find('\'')? + 2
    } else {
        rest.find([',', '}'])?
    };
    Some(rest[..end].trim())
}

#[cfg(test)]
mod tests;
