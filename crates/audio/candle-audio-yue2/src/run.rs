//! Artifact-backed YuE2 runs (sc-22994): transactional run directories, identity-checked resume,
//! plan-only runs and cached-latent decoding.
//!
//! Ported from the pinned upstream `SongResult.save_artifacts`, `SymbolicPlan.save` /
//! `SymbolicPlan.load` (`src/yue2/pipeline.py`), `write_json` / `collect_hashes` /
//! `verify_result` (`src/yue2/storage.py`), the `generate --stage plan|audio --resume` flow of
//! `src/yue2/cli.py` and the skill's `run_yue2.py decode`. Deliberate differences are listed at the
//! end.
//!
//! # A run directory
//!
//! A complete run directory `D` holds upstream's layout:
//!
//! | file | what |
//! |---|---|
//! | `plan.json`, `abc_tokens.npy`, `prefix.npy`, `score.abc` (when there is a score), `plan_manifest.json` | the exact symbolic plan ([`SymbolicPlan::save`]) |
//! | `semantic.npy` | the semantic codec indices, int32 |
//! | `latent.npy`, `latent.json` | the FP32 `[frames, 64]` acoustic latents and their identity |
//! | `audio.wav` | 48 kHz stereo, IEEE float 32-bit |
//! | `request.json`, `config.json` | the request and the effective configuration |
//! | `stages/*.json` | each checkpointed stage's identity and the hashes of its artifacts |
//! | `result.json` | `status: complete`, the run identity, truncation flags, model / decoder / licence identities, per-stage identities and reuse, timings, and the SHA-256 + size of every other file |
//!
//! # Transactional publication
//!
//! Nothing is written into `D` while a run is in progress. The run works in the sibling
//! `D.partial/`, and only after every artifact and `result.json` are written and synced is
//! `D.partial` renamed to `D` (an atomic rename within one directory). A failed or cancelled run
//! therefore never leaves anything at `D`; what it leaves in `D.partial` has no `result.json` of a
//! published run and is refused by [`verify_run`].
//!
//! # Resume
//!
//! With `resume`:
//!
//! * a complete `D` is verified ([`verify_run`]: every file's size and SHA-256) and its identity must
//!   equal this request's run identity — then nothing is recomputed; a mismatch or corruption is an
//!   error and `D` is left untouched;
//! * an interrupted `D.partial` is resumed stage by stage: a checkpointed stage (plan, semantic,
//!   synthesis) is reused only when its recorded stage identity equals the one this request derives
//!   **and** every artifact's bytes hash to the recorded digests **and** the artifact itself
//!   verifies (the plan restores with its recorded plan identity; the latents match their identity
//!   sidecar and were produced by exactly this synthesis). A mismatched or corrupt checkpoint is an
//!   error — it is never recomputed over. Stages after the first missing checkpoint are computed.
//!
//! Without `resume`, an existing non-empty `D` or an existing `D.partial` is refused.
//!
//! Every stage identity binds the MoT weights, tokenizer, dtype, **device** and the runtime build
//! ([`crate::engine::SOURCE_DIGEST`], a digest of this crate's sources — upstream binds
//! `runtime_sha256` the same way), so work done on another backend or by other code is never
//! reused; the run identity additionally binds the decoder's pinned identity.
//!
//! A run holds an exclusive claim on `D.partial` for its whole duration ([`LOCK_FILE`], created
//! with `create_new`): a second run — fresh or resumed — is refused with [`RunError::Locked`]
//! while it is held. The claim is released when the run publishes, fails or is cancelled.
//!
//! A cached decode refuses an output that is, is inside, or contains its source (paths resolved
//! through symlinks and `..`), and carries the source's latent bytes over — checked against the
//! source's recorded digests — before anything is published.
//!
//! # Deliberate differences from upstream
//!
//! * Audio is `audio.wav` (IEEE float, upstream's `.wav` subtype) instead of `audio.flac` — this
//!   workspace has no FLAC encoder, and float WAV is lossless for the decoder's output.
//! * Upstream resumes only a complete result; the stage checkpoints are this port's addition.
//! * Upstream writes each file atomically but the directory incrementally (and a `failure.json`
//!   into the output on error); here the directory is published atomically and a failure leaves
//!   only `D.partial`.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read as _, Write as _};
use std::path::{Component as PathComponent, Path, PathBuf};
use std::time::Instant;

use candle_audio::gen_core;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::engine::{
    identity_of, EngineHooks, SemanticResult, SongSettings, Stage, StageEvent, Yue2Engine,
    IDENTITY_SCHEMA,
};
use crate::inventory::VaeVariant;
use crate::latent::{AcousticLatents, LatentIdentity, LatentSource, IDENTITY_FILE, LATENT_FILE};
use crate::plan::{
    npy_int32, read_npy_ints, PlanIdentity, SymbolicPlan, ABC_TOKENS_NPY, PLAN_JSON, PLAN_MANIFEST,
    PREFIX_NPY, SCORE_ABC,
};
use crate::protocol::{GenerationConfig, SongRequest, CODEC_SIZE};
use crate::vae::{variant_name, AUDIO_CHANNELS, SAMPLE_RATE};

/// `result.json` — written last; its presence with `status: complete` in a published directory
/// marks a complete run.
pub const RESULT_JSON: &str = "result.json";
/// `request.json`.
pub const REQUEST_JSON: &str = "request.json";
/// `config.json` — the effective configuration.
pub const CONFIG_JSON: &str = "config.json";
/// `semantic.npy` — int32 codec indices.
pub const SEMANTIC_NPY: &str = "semantic.npy";
/// `audio.wav` — 48 kHz stereo IEEE float.
pub const AUDIO_WAV: &str = "audio.wav";
/// `source_generation.json` — a cached decode's source record.
pub const SOURCE_GENERATION_JSON: &str = "source_generation.json";
/// `provenance.json` — a plan-only run's model identities and configuration.
pub const PROVENANCE_JSON: &str = "provenance.json";
/// The stage-checkpoint directory.
pub const STAGES_DIR: &str = "stages";
/// Suffix of the working directory of an unpublished run.
pub const PARTIAL_SUFFIX: &str = ".partial";
/// Schema of `result.json`.
pub const RUN_SCHEMA: &str = "yue2-run-v1";
/// The exclusive claim a run holds on its working directory while it assembles it.
pub const LOCK_FILE: &str = ".yue2-run.lock";

/// Every way an artifact-backed run fails.
#[derive(Debug, thiserror::Error)]
pub enum RunError {
    /// The engine failed or was cancelled ([`gen_core::Error::Canceled`]).
    #[error(transparent)]
    Engine(#[from] gen_core::Error),
    /// A fresh run was asked for a directory that already holds files.
    #[error("{} already holds a run; resume it or use a fresh output directory", .0.display())]
    Exists(PathBuf),
    /// A fresh run was asked for a directory with an interrupted run beside it.
    #[error(
        "{} holds an interrupted run; resume it or remove it before starting a fresh one",
        .0.display()
    )]
    Interrupted(PathBuf),
    /// A recorded identity is not the one this request derives.
    #[error(
        "{}: recorded {what} identity {recorded} is not this request's {expected}; the request, \
         configuration or model changed — use a fresh output directory",
        path.display()
    )]
    IdentityMismatch {
        /// The record.
        path: PathBuf,
        /// Which identity.
        what: &'static str,
        /// The recorded identity.
        recorded: String,
        /// The identity this request derives.
        expected: String,
    },
    /// An artifact is missing, changed or malformed.
    #[error("{}: {detail}", path.display())]
    Corrupt {
        /// The artifact or record.
        path: PathBuf,
        /// What is wrong.
        detail: String,
    },
    /// A file-system operation failed.
    #[error("{}: {source}", path.display())]
    Io {
        /// The path.
        path: PathBuf,
        /// The error.
        source: std::io::Error,
    },
    /// The run was asked for something it cannot do.
    #[error("{0}")]
    Invalid(String),
    /// Another run holds the working directory.
    #[error(
        "{} is claimed by another run (remove the lock file only if no run is active)",
        .0.display()
    )]
    Locked(PathBuf),
}

impl From<RunError> for gen_core::Error {
    fn from(e: RunError) -> Self {
        match e {
            RunError::Engine(e) => e,
            other => gen_core::Error::Msg(format!("YuE2 run: {other}")),
        }
    }
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> RunError + '_ {
    move |source| RunError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn corrupt(path: &Path, detail: impl Into<String>) -> RunError {
    RunError::Corrupt {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

/// What a song run starts from.
#[derive(Clone, Debug, PartialEq)]
pub enum SongInput {
    /// A request: planned (or taken from its external score / `cot = off`) by the engine.
    Request(SongRequest),
    /// An exact plan — typically restored with [`SymbolicPlan::restore`]; its token ids are used
    /// as saved.
    Plan(SymbolicPlan),
}

impl SongInput {
    /// The request.
    pub fn request(&self) -> &SongRequest {
        match self {
            SongInput::Request(r) => r,
            SongInput::Plan(p) => p.request(),
        }
    }
}

/// Where and how a run publishes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunOutput {
    /// The run directory `D` (see the [module docs](self)).
    pub dir: PathBuf,
    /// Resume a complete or interrupted run with a matching identity.
    pub resume: bool,
}

impl RunOutput {
    /// A fresh run into `dir`.
    pub fn fresh(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            resume: false,
        }
    }

    /// A resumed run in `dir`.
    pub fn resume(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            resume: true,
        }
    }
}

/// A published run.
#[derive(Clone, Debug, PartialEq)]
pub struct RunOutcome {
    /// The published directory.
    pub dir: PathBuf,
    /// Its `result.json`.
    pub result: Value,
    /// The audio, interleaved `L0 R0 L1 R1 …` at 48 kHz (empty for a plan-only run).
    pub samples: Vec<f32>,
    /// What happened to each stage (a whole reused run reports every stage as reused).
    pub stages: BTreeMap<Stage, StageEvent>,
}

/// The working directory of `dir`: `dir.partial` beside it.
pub fn partial_dir(dir: &Path) -> PathBuf {
    let mut name = dir
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(PARTIAL_SUFFIX);
    dir.with_file_name(name)
}

pub(crate) fn is_nonempty_dir(dir: &Path) -> Result<bool, RunError> {
    match fs::read_dir(dir) {
        Ok(mut entries) => Ok(entries.next().is_some()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(RunError::Io {
            path: dir.to_path_buf(),
            source: e,
        }),
    }
}

/// SHA-256 (hex) and size of the bytes of `path` as they are on disk now.
pub(crate) fn file_digest(path: &Path) -> Result<(String, u64), RunError> {
    let mut file = fs::File::open(path).map_err(io(path))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    let mut bytes = 0u64;
    loop {
        let n = file.read(&mut buf).map_err(io(path))?;
        if n == 0 {
            break;
        }
        bytes += n as u64;
        hasher.update(&buf[..n]);
    }
    Ok((crate::engine::hex(&hasher.finalize()), bytes))
}

/// Write via a sibling temporary file, `fsync`, then rename (upstream `write_json`'s shape).
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), RunError> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = path.with_file_name(format!("{name}.{}.tmp", std::process::id()));
    let mut file = fs::File::create(&tmp).map_err(io(&tmp))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(io(&tmp))?;
    fs::rename(&tmp, path).map_err(io(path))
}

pub(crate) fn write_json(path: &Path, value: &Value) -> Result<(), RunError> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|e| corrupt(path, e.to_string()))?;
    bytes.push(b'\n');
    write_atomic(path, &bytes)
}

pub(crate) fn read_json(path: &Path) -> Result<Value, RunError> {
    let bytes = fs::read(path).map_err(io(path))?;
    serde_json::from_slice(&bytes).map_err(|e| corrupt(path, format!("not JSON: {e}")))
}

pub(crate) fn sync_dir(dir: &Path) -> Result<(), RunError> {
    // Directory fsync makes the renames inside it durable; not every platform opens directories.
    #[cfg(unix)]
    {
        fs::File::open(dir)
            .and_then(|d| d.sync_all())
            .map_err(io(dir))?;
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// Every regular file under `dir` (relative, `/`-separated), `result.json` excluded, with its
/// SHA-256 and size — upstream `collect_hashes`, over the bytes on disk.
fn collect_hashes(dir: &Path) -> Result<Map<String, Value>, RunError> {
    fn walk(root: &Path, dir: &Path, out: &mut Map<String, Value>) -> Result<(), RunError> {
        let mut entries: Vec<_> = fs::read_dir(dir)
            .map_err(io(dir))?
            .collect::<Result<_, _>>()
            .map_err(io(dir))?;
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            let kind = entry.file_type().map_err(io(&path))?;
            if kind.is_dir() {
                walk(root, &path, out)?;
            } else if kind.is_file() {
                let rel = path
                    .strip_prefix(root)
                    .expect("walked below root")
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                if rel == RESULT_JSON || rel == LOCK_FILE || rel.ends_with(".tmp") {
                    continue;
                }
                let (sha256, bytes) = file_digest(&path)?;
                out.insert(rel, json!({"sha256": sha256, "bytes": bytes}));
            } else {
                return Err(corrupt(&path, "run directories hold only regular files"));
            }
        }
        Ok(())
    }
    let mut out = Map::new();
    walk(dir, dir, &mut out)?;
    Ok(out)
}

/// Verify one recorded artifact: a relative path inside `dir`, a regular file (not a symlink),
/// with exactly the recorded size and SHA-256.
fn verify_artifact(dir: &Path, name: &str, expected: &Value) -> Result<(), RunError> {
    let rel = Path::new(name);
    if rel.is_absolute()
        || rel
            .components()
            .any(|c| !matches!(c, PathComponent::Normal(_)))
    {
        return Err(corrupt(dir, format!("invalid artifact path {name:?}")));
    }
    let path = dir.join(rel);
    let meta = fs::symlink_metadata(&path).map_err(|_| corrupt(&path, "missing artifact"))?;
    if !meta.file_type().is_file() {
        return Err(corrupt(&path, "artifact is not a regular file"));
    }
    let want_sha = expected.get("sha256").and_then(Value::as_str);
    let want_bytes = expected.get("bytes").and_then(Value::as_u64);
    let (Some(want_sha), Some(want_bytes)) = (want_sha, want_bytes) else {
        return Err(corrupt(&path, "artifact record lacks sha256/bytes"));
    };
    if meta.len() != want_bytes {
        return Err(corrupt(
            &path,
            format!("{} bytes, recorded {want_bytes}", meta.len()),
        ));
    }
    let (sha, _) = file_digest(&path)?;
    if sha != want_sha {
        return Err(corrupt(
            &path,
            format!("SHA-256 {sha}, recorded {want_sha}"),
        ));
    }
    Ok(())
}

/// Upstream `verify_result`: `dir/result.json` says `complete`, its identity is `expected` (when
/// given), the files its kind requires are recorded, and every recorded file exists with exactly
/// its recorded size and SHA-256. A `.partial` working directory is never a verified run.
pub fn verify_run(dir: &Path, expected: Option<&str>) -> Result<Value, RunError> {
    if dir
        .file_name()
        .is_some_and(|n| n.to_string_lossy().ends_with(PARTIAL_SUFFIX))
    {
        return Err(corrupt(
            dir,
            "an unpublished (.partial) run is never complete",
        ));
    }
    let result_path = dir.join(RESULT_JSON);
    let result = read_json(&result_path)?;
    if result.get("status").and_then(Value::as_str) != Some("complete") {
        return Err(corrupt(&result_path, "the run did not complete"));
    }
    if result.get("schema").and_then(Value::as_str) != Some(RUN_SCHEMA) {
        return Err(corrupt(&result_path, "not a YuE2 run record"));
    }
    let recorded = result
        .get("identity")
        .and_then(Value::as_str)
        .ok_or_else(|| corrupt(&result_path, "no identity"))?;
    if let Some(expected) = expected {
        if recorded != expected {
            return Err(RunError::IdentityMismatch {
                path: result_path,
                what: "run",
                recorded: recorded.to_string(),
                expected: expected.to_string(),
            });
        }
    }
    let artifacts = result
        .get("artifacts")
        .and_then(Value::as_object)
        .ok_or_else(|| corrupt(&result_path, "no artifact manifest"))?;
    let required: &[&str] = match result.get("kind").and_then(Value::as_str) {
        Some("plan") => &[PLAN_JSON, ABC_TOKENS_NPY, PREFIX_NPY, PLAN_MANIFEST],
        Some("song" | "cached_decode") => &[
            PLAN_JSON,
            ABC_TOKENS_NPY,
            PREFIX_NPY,
            PLAN_MANIFEST,
            SEMANTIC_NPY,
            LATENT_FILE,
            IDENTITY_FILE,
            AUDIO_WAV,
            REQUEST_JSON,
            CONFIG_JSON,
        ],
        _ => return Err(corrupt(&result_path, "unknown run kind")),
    };
    if let Some(missing) = required.iter().find(|n| !artifacts.contains_key(**n)) {
        return Err(corrupt(
            &result_path,
            format!("incomplete artifact manifest: {missing} is not recorded"),
        ));
    }
    for (name, digest) in artifacts {
        verify_artifact(dir, name, digest)?;
    }
    Ok(result)
}

// ---------------------------------------------------------------------------------------------
// Audio file.
// ---------------------------------------------------------------------------------------------

/// A RIFF/WAVE IEEE-float 32-bit file of interleaved samples (format tag 3 with a `fact` chunk).
pub fn wav_f32_bytes(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let data_len = samples.len() * 4;
    let frames = samples.len() / channels.max(1) as usize;
    let mut out = Vec::with_capacity(58 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((50 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&18u32.to_le_bytes());
    out.extend_from_slice(&3u16.to_le_bytes()); // WAVE_FORMAT_IEEE_FLOAT
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * channels as u32 * 4).to_le_bytes());
    out.extend_from_slice(&(channels * 4).to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // cbSize
    out.extend_from_slice(b"fact");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&(frames as u32).to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Read a WAV written by [`wav_f32_bytes`]: `(interleaved samples, sample rate, channels)`.
pub fn read_wav_f32(bytes: &[u8]) -> Result<(Vec<f32>, u32, u16), String> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let mut at = 12;
    let mut format = None;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().expect("4 bytes")) as usize;
        let body = bytes.get(at + 8..at + 8 + len).ok_or("truncated chunk")?;
        match id {
            b"fmt " => {
                if len < 16 {
                    return Err("short fmt chunk".into());
                }
                let tag = u16::from_le_bytes([body[0], body[1]]);
                let channels = u16::from_le_bytes([body[2], body[3]]);
                let rate = u32::from_le_bytes(body[4..8].try_into().expect("4 bytes"));
                let bits = u16::from_le_bytes([body[14], body[15]]);
                if tag != 3 || bits != 32 || channels == 0 {
                    return Err("not IEEE float 32-bit audio".into());
                }
                format = Some((rate, channels));
            }
            b"data" => {
                let (rate, channels) = format.ok_or("data before fmt")?;
                if !len.is_multiple_of(4 * channels as usize) {
                    return Err("data is not whole frames".into());
                }
                let samples = body
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                return Ok((samples, rate, channels));
            }
            _ => {}
        }
        at += 8 + len + (len & 1);
    }
    Err("no data chunk".into())
}

// ---------------------------------------------------------------------------------------------
// Working directories and stage checkpoints.
// ---------------------------------------------------------------------------------------------

/// Where a run is being assembled.
struct WorkDir {
    /// The directory it publishes to.
    target: PathBuf,
    /// `target.partial`.
    work: PathBuf,
    /// This run's exclusive claim on `work`.
    _claim: Claim,
}

/// An exclusive claim on a working directory: [`LOCK_FILE`] created with `create_new` (so two runs
/// can never hold it at once), removed when the run publishes, fails or is cancelled.
struct Claim {
    path: PathBuf,
}

impl Claim {
    fn take(work: &Path) -> Result<Self, RunError> {
        let path = work.join(LOCK_FILE);
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut file) => {
                file.write_all(format!("pid {}\n", std::process::id()).as_bytes())
                    .map_err(io(&path))?;
                Ok(Self { path })
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Err(RunError::Locked(path)),
            Err(e) => Err(RunError::Io { path, source: e }),
        }
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        // After publication the working directory was renamed away and the claim travelled
        // with it (the publisher removes it there); nothing is left to remove here.
        let _ = fs::remove_file(&self.path);
    }
}

/// The state of an output directory at the start of a run.
enum Opened {
    /// A complete published run (resume only).
    Complete,
    /// A working directory to (continue to) assemble the run in.
    Work(WorkDir),
}

fn open_output(output: &RunOutput) -> Result<Opened, RunError> {
    let target = output.dir.clone();
    let work = partial_dir(&target);
    if target.as_os_str().is_empty() || target.file_name().is_none() {
        return Err(RunError::Invalid(format!(
            "{} is not a usable run directory",
            target.display()
        )));
    }
    if is_nonempty_dir(&target)? {
        if output.resume {
            return Ok(Opened::Complete);
        }
        return Err(RunError::Exists(target));
    }
    if target.exists() && !target.is_dir() {
        return Err(RunError::Exists(target));
    }
    if work.exists() {
        if !output.resume {
            return Err(RunError::Interrupted(work));
        }
        let claim = Claim::take(&work)?;
        return Ok(Opened::Work(WorkDir {
            target,
            work,
            _claim: claim,
        }));
    }
    if let Some(parent) = target.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(io(parent))?;
    }
    // `create_dir` (not `_all`) claims the working directory: a concurrent fresh run into the same
    // target fails here instead of interleaving files; the lock file then covers resumes too.
    fs::create_dir(&work).map_err(io(&work))?;
    let claim = Claim::take(&work)?;
    Ok(Opened::Work(WorkDir {
        target,
        work,
        _claim: claim,
    }))
}

impl WorkDir {
    fn path(&self, name: &str) -> PathBuf {
        self.work.join(name)
    }

    fn record_path(&self, stage: Stage) -> PathBuf {
        self.work
            .join(STAGES_DIR)
            .join(format!("{}.json", stage.name()))
    }

    /// Remove files of an unrecorded (interrupted) stage so the stage can be recomputed. Only
    /// called when the stage has no checkpoint record: such files were never part of any
    /// checkpoint or published run.
    fn clear_unrecorded(&self, names: &[&str]) -> Result<(), RunError> {
        for name in names {
            let path = self.path(name);
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(RunError::Io { path, source: e }),
            }
        }
        Ok(())
    }

    /// The checkpoint of `stage`, if recorded: verified against `identity` and the recorded
    /// artifact digests. `Ok(None)` when there is no record.
    fn checkpoint(&self, stage: Stage, identity: &str) -> Result<Option<Value>, RunError> {
        let path = self.record_path(stage);
        if fs::symlink_metadata(&path).is_err() {
            return Ok(None);
        }
        let record = read_json(&path)?;
        if record.get("stage").and_then(Value::as_str) != Some(stage.name())
            || record.get("schema").and_then(Value::as_str) != Some(IDENTITY_SCHEMA)
        {
            return Err(corrupt(&path, "not this stage's checkpoint record"));
        }
        let recorded = record
            .get("identity")
            .and_then(Value::as_str)
            .ok_or_else(|| corrupt(&path, "no identity"))?;
        if recorded != identity {
            return Err(RunError::IdentityMismatch {
                path,
                what: stage.name(),
                recorded: recorded.to_string(),
                expected: identity.to_string(),
            });
        }
        let artifacts = record
            .get("artifacts")
            .and_then(Value::as_object)
            .ok_or_else(|| corrupt(&path, "no artifact digests"))?;
        if artifacts.is_empty() {
            return Err(corrupt(&path, "a checkpoint records at least one artifact"));
        }
        for (name, digest) in artifacts {
            verify_artifact(&self.work, name, digest)?;
        }
        Ok(Some(record.get("data").cloned().unwrap_or(Value::Null)))
    }

    /// Record `stage` complete: hash the artifacts as they are on disk, then write the record
    /// atomically (the record is the completion marker, so it is written last).
    fn record(
        &self,
        stage: Stage,
        identity: &str,
        artifacts: &[&str],
        data: Value,
    ) -> Result<(), RunError> {
        let mut digests = Map::new();
        for name in artifacts {
            let path = self.path(name);
            fs::File::open(&path)
                .and_then(|f| f.sync_all())
                .map_err(io(&path))?;
            let (sha256, bytes) = file_digest(&path)?;
            digests.insert(name.to_string(), json!({"sha256": sha256, "bytes": bytes}));
        }
        let dir = self.work.join(STAGES_DIR);
        fs::create_dir_all(&dir).map_err(io(&dir))?;
        write_json(
            &self.record_path(stage),
            &json!({
                "schema": IDENTITY_SCHEMA,
                "stage": stage.name(),
                "identity": identity,
                "artifacts": digests,
                "data": data,
            }),
        )?;
        sync_dir(&self.work)
    }

    /// Write `result.json` (with every other file's digest), sync, and rename the working
    /// directory onto the target.
    fn publish(&self, mut result: Map<String, Value>) -> Result<(PathBuf, Value), RunError> {
        let artifacts = collect_hashes(&self.work)?;
        for name in artifacts.keys() {
            let path = self.work.join(name);
            fs::File::open(&path)
                .and_then(|f| f.sync_all())
                .map_err(io(&path))?;
        }
        result.insert("artifacts".into(), Value::Object(artifacts));
        let result = Value::Object(result);
        write_json(&self.work.join(RESULT_JSON), &result)?;
        let stages = self.work.join(STAGES_DIR);
        if stages.is_dir() {
            sync_dir(&stages)?;
        }
        sync_dir(&self.work)?;
        if self.target.is_dir() {
            // An empty target directory (checked when the run opened) is replaced.
            fs::remove_dir(&self.target).map_err(io(&self.target))?;
        }
        fs::rename(&self.work, &self.target).map_err(io(&self.target))?;
        // The claim moved with the directory; it is not an artifact (never hashed) — release it.
        let moved = self.target.join(LOCK_FILE);
        fs::remove_file(&moved).map_err(io(&moved))?;
        if let Some(parent) = self.target.parent().filter(|p| !p.as_os_str().is_empty()) {
            sync_dir(parent)?;
        }
        Ok((self.target.clone(), result))
    }
}

fn plan_files(plan: &SymbolicPlan) -> Vec<&'static str> {
    let mut names = vec![PLAN_JSON, ABC_TOKENS_NPY, PREFIX_NPY, PLAN_MANIFEST];
    if plan.abc().is_some() {
        names.push(SCORE_ABC);
    }
    names
}

const PLAN_FILES_ALL: [&str; 5] = [
    PLAN_JSON,
    ABC_TOKENS_NPY,
    PREFIX_NPY,
    PLAN_MANIFEST,
    SCORE_ABC,
];

fn parse_plan_identity(path: &Path, v: Option<&Value>) -> Result<String, RunError> {
    v.and_then(Value::as_str)
        .filter(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        .map(str::to_string)
        .ok_or_else(|| corrupt(path, "no plan identity recorded"))
}

/// Restore the plan saved in `dir` and require its identity to be `expected` (hex).
fn restore_plan(engine: &Yue2Engine, dir: &Path, expected: &str) -> Result<SymbolicPlan, RunError> {
    let plan = SymbolicPlan::restore(dir, engine.tokenizer())
        .map_err(|e| corrupt(&dir.join(PLAN_JSON), e.to_string()))?;
    if plan.identity().to_string() != expected {
        return Err(RunError::IdentityMismatch {
            path: dir.join(PLAN_JSON),
            what: "plan",
            recorded: plan.identity().to_string(),
            expected: expected.to_string(),
        });
    }
    Ok(plan)
}

/// Read `semantic.npy`: a 1-D integer array of codec indices.
fn read_semantic(path: &Path) -> Result<Vec<u32>, RunError> {
    let bytes = fs::read(path).map_err(io(path))?;
    read_npy_ints(SEMANTIC_NPY, &bytes)
        .map_err(|e| corrupt(path, e.to_string()))?
        .into_iter()
        .map(|v| {
            u32::try_from(v)
                .ok()
                .filter(|&c| c < CODEC_SIZE)
                .ok_or_else(|| corrupt(path, format!("{v} is not a codec index")))
        })
        .collect()
}

fn write_semantic(path: &Path, codes: &[u32]) -> Result<(), RunError> {
    let bytes = npy_int32(codes).map_err(|e| corrupt(path, e.to_string()))?;
    write_atomic(path, &bytes)
}

fn observe(
    stages: &mut BTreeMap<Stage, StageEvent>,
    hooks: &mut EngineHooks<'_>,
    stage: Stage,
    event: StageEvent,
) {
    stages.insert(stage, event);
    if event == StageEvent::Reused {
        hooks.observer.on_stage(stage, event);
    }
}

fn stage_entry(identity: &str, reused: bool) -> Value {
    json!({"identity": identity, "reused": reused})
}

/// `path` resolved component by component: every prefix that exists is canonicalized (so
/// symlinks resolve to their targets), `..` then removes one resolved component and `.` none, and
/// a component that does not exist yet — which cannot be a link — is appended as written.
fn resolve_path(path: &Path) -> Result<PathBuf, RunError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().map_err(io(path))?.join(path)
    };
    let mut resolved = PathBuf::new();
    for component in absolute.components() {
        match component {
            PathComponent::Prefix(_) | PathComponent::RootDir => resolved.push(component),
            PathComponent::CurDir => {}
            PathComponent::ParentDir => {
                resolved.pop();
            }
            PathComponent::Normal(name) => {
                let next = resolved.join(name);
                resolved = match fs::canonicalize(&next) {
                    Ok(canonical) => canonical,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => next,
                    Err(e) => {
                        return Err(RunError::Io {
                            path: next,
                            source: e,
                        })
                    }
                };
            }
        }
    }
    Ok(resolved)
}

/// Refuse an output (or its working directory) that is, is inside, or contains the source run.
fn refuse_overlap(source: &Path, output: &Path) -> Result<(), RunError> {
    let source = resolve_path(source)?;
    for candidate in [output.to_path_buf(), partial_dir(output)] {
        let candidate = resolve_path(&candidate)?;
        if candidate.starts_with(&source) || source.starts_with(&candidate) {
            return Err(RunError::Invalid(format!(
                "a cached decode never writes into its source run: {} overlaps {}",
                candidate.display(),
                source.display()
            )));
        }
    }
    Ok(())
}

/// The whole-run record fields shared by a song run and a cached decode.
fn audio_fields(
    result: &mut Map<String, Value>,
    samples: usize,
    decoder: &Value,
    latent: &LatentIdentity,
    clamped: usize,
) {
    let frames = samples / AUDIO_CHANNELS;
    result.insert("sample_rate".into(), json!(SAMPLE_RATE));
    result.insert("channels".into(), json!(AUDIO_CHANNELS));
    result.insert(
        "audio_seconds".into(),
        json!(frames as f64 / SAMPLE_RATE as f64),
    );
    result.insert("decoder".into(), decoder.clone());
    result.insert("latent".into(), latent.to_json());
    result.insert("clamped_samples".into(), json!(clamped));
}

impl Yue2Engine {
    /// The run identity of `input` with `settings` (see [`Yue2Engine::run_identity`]); a run from
    /// an exact plan also binds the plan's identity.
    pub fn input_identity(
        &self,
        input: &SongInput,
        settings: &SongSettings,
    ) -> gen_core::Result<String> {
        let request = input.request();
        Ok(identity_of(&json!({
            "schema": IDENTITY_SCHEMA,
            "kind": "song",
            "request": request.to_json(),
            "config": self.effective_config(request, settings),
            "weights": self.model_identity()?,
            "plan": match input {
                SongInput::Request(_) => Value::Null,
                SongInput::Plan(p) => json!(p.identity().to_string()),
            },
        })))
    }

    /// Generate one song into a run directory (see the [module docs](self)): transactional,
    /// identity-checked, resumable stage by stage.
    pub fn generate_to(
        &self,
        input: &SongInput,
        settings: &SongSettings,
        output: &RunOutput,
        hooks: &mut EngineHooks<'_>,
    ) -> Result<RunOutcome, RunError> {
        let start = Instant::now();
        self.check_decoder_available(settings.decoder)?;
        let run_identity = self.input_identity(input, settings)?;
        let work = match open_output(output)? {
            Opened::Complete => return self.reuse_complete(&output.dir, &run_identity, hooks),
            Opened::Work(work) => work,
        };
        // A result written just before an interrupted publish: verify it like a published run.
        if fs::symlink_metadata(work.path(RESULT_JSON)).is_ok() {
            return Err(corrupt(
                &work.path(RESULT_JSON),
                "an unpublished result exists; remove the interrupted run to regenerate it",
            ));
        }
        let mut stages = BTreeMap::new();
        let mut stage_ids = Map::new();

        // Plan.
        let plan_identity = match input {
            SongInput::Request(r) => self.plan_stage_identity(r, settings.generation.abc())?,
            SongInput::Plan(p) => identity_of(&json!([
                IDENTITY_SCHEMA,
                "plan",
                "restored",
                p.identity().to_string()
            ])),
        };
        let plan = match work.checkpoint(Stage::Plan, &plan_identity)? {
            Some(data) => {
                let recorded =
                    parse_plan_identity(&work.record_path(Stage::Plan), data.get("plan_identity"))?;
                let plan = restore_plan(self, &work.work, &recorded)?;
                if plan.request() != input.request() {
                    return Err(corrupt(
                        &work.path(PLAN_JSON),
                        "the checkpointed plan is for a different request",
                    ));
                }
                observe(&mut stages, hooks, Stage::Plan, StageEvent::Reused);
                plan
            }
            None => {
                work.clear_unrecorded(&PLAN_FILES_ALL)?;
                let plan = match input {
                    SongInput::Request(r) => self.plan(r, &settings.generation, hooks)?,
                    SongInput::Plan(p) => {
                        hooks.observer.on_stage(Stage::Plan, StageEvent::Started);
                        hooks.observer.on_stage(Stage::Plan, StageEvent::Finished);
                        p.clone()
                    }
                };
                let id = plan
                    .save(&work.work)
                    .map_err(|e| corrupt(&work.work, e.to_string()))?;
                work.record(
                    Stage::Plan,
                    &plan_identity,
                    &plan_files(&plan),
                    json!({"plan_identity": id.to_string()}),
                )?;
                observe(&mut stages, hooks, Stage::Plan, StageEvent::Finished);
                plan
            }
        };
        stage_ids.insert(
            "plan".into(),
            stage_entry(&plan_identity, stages[&Stage::Plan] == StageEvent::Reused),
        );

        // Semantic.
        let semantic_identity =
            self.semantic_stage_identity(&plan, settings.generation.semantic())?;
        let semantic = match work.checkpoint(Stage::Semantic, &semantic_identity)? {
            Some(data) => {
                let codes = read_semantic(&work.path(SEMANTIC_NPY))?;
                let truncated = data
                    .get("truncated")
                    .and_then(Value::as_bool)
                    .ok_or_else(|| corrupt(&work.record_path(Stage::Semantic), "no truncated"))?;
                let timing = data
                    .get("timing")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                observe(&mut stages, hooks, Stage::Semantic, StageEvent::Reused);
                SemanticResult {
                    plan: plan.clone(),
                    codes,
                    truncated,
                    timing,
                }
            }
            None => {
                work.clear_unrecorded(&[SEMANTIC_NPY])?;
                let semantic = self.generate_semantic(&plan, &settings.generation, hooks)?;
                write_semantic(&work.path(SEMANTIC_NPY), &semantic.codes)?;
                work.record(
                    Stage::Semantic,
                    &semantic_identity,
                    &[SEMANTIC_NPY],
                    json!({"truncated": semantic.truncated, "timing": semantic.timing}),
                )?;
                observe(&mut stages, hooks, Stage::Semantic, StageEvent::Finished);
                semantic
            }
        };
        stage_ids.insert(
            "semantic".into(),
            stage_entry(
                &semantic_identity,
                stages[&Stage::Semantic] == StageEvent::Reused,
            ),
        );

        // Acoustic synthesis.
        let synthesis_identity = self.synthesis_stage_identity(&semantic, &settings.generation)?;
        let nar_identity = self.identity_keys().nar(&semantic, &settings.generation);
        let (latents, nar_seconds) = match work.checkpoint(Stage::Synthesis, &synthesis_identity)? {
            Some(data) => {
                let latents = AcousticLatents::load(&work.work)
                    .map_err(|e| corrupt(&work.path(LATENT_FILE), e.to_string()))?;
                let recorded = data
                    .get("latent")
                    .map(LatentIdentity::from_json)
                    .transpose()
                    .map_err(|e| corrupt(&work.record_path(Stage::Synthesis), e.to_string()))?;
                if recorded.as_ref() != Some(latents.identity()) {
                    return Err(corrupt(
                        &work.path(IDENTITY_FILE),
                        "the latents are not the ones the synthesis checkpoint recorded",
                    ));
                }
                if latents.identity().source
                    != (LatentSource::Synthesis {
                        stage_identity: nar_identity.clone(),
                    })
                {
                    return Err(corrupt(
                        &work.path(IDENTITY_FILE),
                        "the latents were not produced by this synthesis",
                    ));
                }
                observe(&mut stages, hooks, Stage::Synthesis, StageEvent::Reused);
                let seconds = data.get("seconds").cloned().unwrap_or(Value::Null);
                (latents, seconds)
            }
            None => {
                work.clear_unrecorded(&[LATENT_FILE, IDENTITY_FILE])?;
                let synthesis = self.synthesize(&semantic, &settings.generation, hooks)?;
                synthesis
                    .latents
                    .save(&work.work)
                    .map_err(|e| corrupt(&work.path(LATENT_FILE), e.to_string()))?;
                work.record(
                    Stage::Synthesis,
                    &synthesis_identity,
                    &[LATENT_FILE, IDENTITY_FILE],
                    json!({
                        "latent": synthesis.latents.identity().to_json(),
                        "chunks": synthesis.chunks,
                        "seconds": synthesis.seconds,
                    }),
                )?;
                observe(&mut stages, hooks, Stage::Synthesis, StageEvent::Finished);
                (synthesis.latents, json!(synthesis.seconds))
            }
        };
        stage_ids.insert(
            "synthesis".into(),
            stage_entry(
                &synthesis_identity,
                stages[&Stage::Synthesis] == StageEvent::Reused,
            ),
        );

        // Decode (not checkpointed: it is the last stage before publication).
        work.clear_unrecorded(&[AUDIO_WAV])?;
        hooks.check_cancel()?;
        let vae_start = Instant::now();
        let audio = self.decode(&latents, settings.decoder, hooks)?;
        let vae_seconds = vae_start.elapsed().as_secs_f64();
        stages.insert(Stage::Decode, StageEvent::Finished);
        write_atomic(
            &work.path(AUDIO_WAV),
            &wav_f32_bytes(audio.samples(), SAMPLE_RATE, AUDIO_CHANNELS as u16),
        )?;
        let keys = self.identity_keys();
        let decode_identity = identity_of(&json!([
            IDENTITY_SCHEMA,
            "decode",
            latents.identity().sha256,
            audio.metadata().decoder.to_json(),
            keys.device,
            keys.runtime,
            audio.metadata().vae_decode(),
            audio.metadata().halo_frames,
            match audio.metadata().mode {
                crate::decode::DecodeMode::Tiled { core_frames } => json!(core_frames),
                crate::decode::DecodeMode::Full => Value::Null,
            },
        ]));
        stage_ids.insert("decode".into(), stage_entry(&decode_identity, false));

        let request = plan.request();
        let config = self.effective_config(request, settings);
        write_json(&work.path(REQUEST_JSON), &request.to_json())?;
        write_json(&work.path(CONFIG_JSON), &config)?;
        hooks.check_cancel()?;

        let mut timing = Map::new();
        timing.insert("abc".into(), Value::Object(plan.timing().clone()));
        timing.insert("semantic".into(), Value::Object(semantic.timing.clone()));
        timing.insert("nar_seconds".into(), nar_seconds);
        timing.insert("vae_seconds".into(), json!(vae_seconds));
        timing.insert("load".into(), Value::Object(self.load_timing().clone()));
        timing.insert("e2e_seconds".into(), json!(start.elapsed().as_secs_f64()));
        let mut result = Map::new();
        result.insert("schema".into(), json!(RUN_SCHEMA));
        result.insert("kind".into(), json!("song"));
        result.insert("status".into(), json!("complete"));
        result.insert("identity".into(), json!(run_identity));
        result.insert(
            "truncated".into(),
            json!({"abc": plan.truncated(), "semantic": semantic.truncated}),
        );
        result.insert("plan_identity".into(), json!(plan.identity().to_string()));
        audio_fields(
            &mut result,
            audio.samples().len(),
            &audio.metadata().decoder.to_json(),
            latents.identity(),
            audio.metadata().clamped_samples,
        );
        result.insert("weights".into(), self.model_identity()?);
        result.insert("license".into(), self.license_record(settings.decoder)?);
        result.insert("stages".into(), Value::Object(stage_ids));
        result.insert("timing".into(), Value::Object(timing));
        let (dir, result) = work.publish(result)?;
        Ok(RunOutcome {
            dir,
            result,
            samples: audio.samples().to_vec(),
            stages,
        })
    }

    /// Reuse a complete published run after verifying it against `identity`.
    fn reuse_complete(
        &self,
        dir: &Path,
        identity: &str,
        hooks: &mut EngineHooks<'_>,
    ) -> Result<RunOutcome, RunError> {
        let result = verify_run(dir, Some(identity))?;
        let samples = if result.get("kind").and_then(Value::as_str) == Some("plan") {
            Vec::new()
        } else {
            let path = dir.join(AUDIO_WAV);
            let bytes = fs::read(&path).map_err(io(&path))?;
            let (samples, rate, channels) = read_wav_f32(&bytes).map_err(|e| corrupt(&path, e))?;
            if rate != SAMPLE_RATE || channels as usize != AUDIO_CHANNELS {
                return Err(corrupt(&path, "not 48 kHz stereo"));
            }
            samples
        };
        let mut stages = BTreeMap::new();
        for stage in Stage::ALL {
            observe(&mut stages, hooks, stage, StageEvent::Reused);
        }
        Ok(RunOutcome {
            dir: dir.to_path_buf(),
            result,
            samples,
            stages,
        })
    }

    /// Plan only (upstream `yue2 generate --stage plan`): plan `request` and publish the exact plan
    /// transactionally into `output.dir` with its request, model provenance and a `result.json` of
    /// kind `plan`. Restore it with [`SymbolicPlan::restore`] (or [`SymbolicPlan::restore_expecting`]
    /// with the returned identity) and continue with [`SongInput::Plan`].
    pub fn plan_to(
        &self,
        request: &SongRequest,
        generation: &GenerationConfig,
        output: &RunOutput,
        hooks: &mut EngineHooks<'_>,
    ) -> Result<(SymbolicPlan, PlanIdentity, PathBuf), RunError> {
        let identity = self.plan_stage_identity(request, generation.abc())?;
        let work = match open_output(output)? {
            Opened::Complete => {
                let result = verify_run(&output.dir, Some(&identity))?;
                let recorded = parse_plan_identity(
                    &output.dir.join(RESULT_JSON),
                    result.get("plan_identity"),
                )?;
                let plan = restore_plan(self, &output.dir, &recorded)?;
                hooks.observer.on_stage(Stage::Plan, StageEvent::Reused);
                let id = plan.identity();
                return Ok((plan, id, output.dir.clone()));
            }
            Opened::Work(work) => work,
        };
        work.clear_unrecorded(&PLAN_FILES_ALL)?;
        work.clear_unrecorded(&[REQUEST_JSON, PROVENANCE_JSON, RESULT_JSON])?;
        let plan = self.plan(request, generation, hooks)?;
        let plan_id = plan
            .save(&work.work)
            .map_err(|e| corrupt(&work.work, e.to_string()))?;
        write_json(&work.path(REQUEST_JSON), &request.to_json())?;
        let settings = SongSettings {
            generation: generation.clone(),
            ..self.default_settings()
        };
        write_json(
            &work.path(PROVENANCE_JSON),
            &json!({
                "weights": self.model_identity()?,
                "config": self.effective_config(request, &settings),
                "license": self.license_record(VaeVariant::Standard)?,
            }),
        )?;
        hooks.check_cancel()?;
        let mut result = Map::new();
        result.insert("schema".into(), json!(RUN_SCHEMA));
        result.insert("kind".into(), json!("plan"));
        result.insert("status".into(), json!("complete"));
        result.insert("identity".into(), json!(identity));
        result.insert("plan_identity".into(), json!(plan_id.to_string()));
        result.insert("truncated".into(), json!({"abc": plan.truncated()}));
        let (dir, _) = work.publish(result)?;
        Ok((plan, plan_id, dir))
    }

    /// Decode the cached latents of the complete run in `source` with `decoder` (the skill's
    /// `run_yue2.py decode`): the source is verified and never written; its plan is restored with
    /// its recorded identity and its latents with theirs; the MoT identity must be the source's.
    /// With `output`, the decode is published as a new run (kind `cached_decode`) carrying the
    /// source's plan, tokens and byte-identical latents, the new audio, the source's configuration
    /// with the decoder fields replaced, and `source_generation.json`.
    pub fn decode_cached(
        &self,
        source: &Path,
        decoder: VaeVariant,
        output: Option<&RunOutput>,
        hooks: &mut EngineHooks<'_>,
    ) -> Result<RunOutcome, RunError> {
        self.check_decoder_available(decoder)?;
        if let Some(output) = output {
            refuse_overlap(source, &output.dir)?;
        }
        let original = verify_run(source, None)?;
        if original.get("kind").and_then(Value::as_str) == Some("plan") {
            return Err(RunError::Invalid(format!(
                "{} is a plan-only run; it has no latents to decode",
                source.display()
            )));
        }
        let source_result = source.join(RESULT_JSON);
        let source_identity = original
            .get("identity")
            .and_then(Value::as_str)
            .ok_or_else(|| corrupt(&source_result, "no identity"))?
            .to_string();
        let mot = self.model_identity()?;
        if original.pointer("/weights/mot") != mot.get("mot") {
            return Err(RunError::Invalid(
                "the loaded model is not the source generation's exact model identity".into(),
            ));
        }
        let plan_identity = parse_plan_identity(&source_result, original.get("plan_identity"))?;
        let plan = restore_plan(self, source, &plan_identity)?;
        let codes = read_semantic(&source.join(SEMANTIC_NPY))?;
        let latents = AcousticLatents::load(source)
            .map_err(|e| corrupt(&source.join(LATENT_FILE), e.to_string()))?;
        let recorded_latent = original
            .get("latent")
            .map(LatentIdentity::from_json)
            .transpose()
            .map_err(|e| corrupt(&source_result, e.to_string()))?;
        if recorded_latent.as_ref() != Some(latents.identity()) {
            return Err(corrupt(
                &source.join(IDENTITY_FILE),
                "the cached latents are not the ones the source run recorded",
            ));
        }
        let source_config = read_json(&source.join(CONFIG_JSON))?;

        let identity = self.identity_keys().cached_decode(
            &source_identity,
            &latents.identity().to_json(),
            &json!([variant_name(decoder), self.vae_identity(decoder)?]),
            &json!(format!("{:?}", self.options().decode)),
            &mot,
        );
        let work = match output {
            None => None,
            Some(output) => match open_output(output)? {
                Opened::Complete => return self.reuse_complete(&output.dir, &identity, hooks),
                Opened::Work(work) => Some(work),
            },
        };
        let mut stages = BTreeMap::new();
        for stage in [Stage::Plan, Stage::Semantic, Stage::Synthesis] {
            observe(&mut stages, hooks, stage, StageEvent::Reused);
        }
        let vae_start = Instant::now();
        let audio = self.decode(&latents, decoder, hooks)?;
        let vae_seconds = vae_start.elapsed().as_secs_f64();
        stages.insert(Stage::Decode, StageEvent::Finished);
        let Some(work) = work else {
            return Ok(RunOutcome {
                dir: source.to_path_buf(),
                result: original,
                samples: audio.samples().to_vec(),
                stages,
            });
        };
        work.clear_unrecorded(&PLAN_FILES_ALL)?;
        work.clear_unrecorded(&[
            SEMANTIC_NPY,
            LATENT_FILE,
            IDENTITY_FILE,
            AUDIO_WAV,
            REQUEST_JSON,
            CONFIG_JSON,
            SOURCE_GENERATION_JSON,
        ])?;
        plan.save(&work.work)
            .map_err(|e| corrupt(&work.work, e.to_string()))?;
        write_semantic(&work.path(SEMANTIC_NPY), &codes)?;
        // The latents are carried over byte for byte and checked BEFORE anything is published:
        // the copies must hash to the digests the verified source recorded, and must load as the
        // very latents that were decoded.
        for name in [LATENT_FILE, IDENTITY_FILE] {
            let (from, to) = (source.join(name), work.path(name));
            fs::copy(&from, &to).map_err(io(&to))?;
            let recorded = original
                .pointer(&format!("/artifacts/{name}/sha256"))
                .and_then(Value::as_str);
            if recorded != Some(file_digest(&to)?.0.as_str()) {
                return Err(corrupt(&to, "latents changed while they were carried over"));
            }
        }
        let carried = AcousticLatents::load(&work.work)
            .map_err(|e| corrupt(&work.path(LATENT_FILE), e.to_string()))?;
        if carried.identity() != latents.identity() {
            return Err(corrupt(
                &work.path(IDENTITY_FILE),
                "the carried-over latents are not the ones decoded",
            ));
        }
        write_atomic(
            &work.path(AUDIO_WAV),
            &wav_f32_bytes(audio.samples(), SAMPLE_RATE, AUDIO_CHANNELS as u16),
        )?;
        write_json(&work.path(REQUEST_JSON), &plan.request().to_json())?;
        let current = self.effective_config(
            plan.request(),
            &SongSettings {
                decoder,
                ..self.default_settings()
            },
        );
        let mut config = source_config.as_object().cloned().unwrap_or_default();
        for key in [
            "vae_dtype",
            "vae_decode",
            "vae_core_frames",
            "vae_halo_frames",
            "decoder_release",
        ] {
            config.insert(key.into(), current.get(key).cloned().unwrap_or(Value::Null));
        }
        let digest = |name: &str| file_digest(&source.join(name)).map(|(sha, _)| sha);
        config.insert(
            "cached_decode".into(),
            json!({
                "source_identity": source_identity,
                "source_config_sha256": digest(CONFIG_JSON)?,
                "runtime": current.get("runtime"),
                "device": current.get("device"),
            }),
        );
        write_json(&work.path(CONFIG_JSON), &Value::Object(config))?;
        write_json(
            &work.path(SOURCE_GENERATION_JSON),
            &json!({
                "source_result_sha256": digest(RESULT_JSON)?,
                "source_latent_sha256": digest(LATENT_FILE)?,
                "source_semantic_sha256": digest(SEMANTIC_NPY)?,
                "config": source_config,
                "identity": source_identity,
                "weights": original.get("weights"),
            }),
        )?;
        hooks.check_cancel()?;
        let mut timing = Map::new();
        timing.insert("operation".into(), json!("decode_cached_latents"));
        timing.insert("vae_seconds".into(), json!(vae_seconds));
        timing.insert("semantic".into(), Value::Object(Map::new()));
        timing.insert(
            "source_generation".into(),
            original.get("timing").cloned().unwrap_or(Value::Null),
        );
        let mut result = Map::new();
        result.insert("schema".into(), json!(RUN_SCHEMA));
        result.insert("kind".into(), json!("cached_decode"));
        result.insert("status".into(), json!("complete"));
        result.insert("identity".into(), json!(identity));
        result.insert(
            "truncated".into(),
            original.get("truncated").cloned().unwrap_or(Value::Null),
        );
        result.insert("plan_identity".into(), json!(plan_identity));
        audio_fields(
            &mut result,
            audio.samples().len(),
            &audio.metadata().decoder.to_json(),
            latents.identity(),
            audio.metadata().clamped_samples,
        );
        result.insert("weights".into(), mot);
        result.insert("license".into(), self.license_record(decoder)?);
        result.insert("source_identity".into(), json!(source_identity));
        result.insert("timing".into(), Value::Object(timing));
        let (dir, result) = work.publish(result)?;
        Ok(RunOutcome {
            dir,
            result,
            samples: audio.samples().to_vec(),
            stages,
        })
    }
}

#[cfg(test)]
mod tests;
