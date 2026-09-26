//! Offline resolution and integrity verification of YuE2 snapshots.
//!
//! # Verify at the load boundary
//!
//! Verification is a statement about the bytes on disk **at the moment of the call**. A loader must
//! call [`resolve_closure`] / [`resolve_component`] (or [`verify_component`]) **immediately before
//! it loads**, and must load only the paths the returned [`VerifiedComponent`] names. Caching a
//! `VerifiedClosure` across requests, or verifying at start-up and loading later, leaves a window in
//! which a file can be replaced unnoticed.
//!
//! A snapshot is found — never fetched — through [`SnapshotDirs`]: the directory the application
//! provisioned for each pinned repository. A repository with no directory, or whose directory is
//! absent, is an [`AssetError::CacheMiss`]; nothing here touches the network or derives a cache
//! location, so offline use and a cold cache fail the same explicit way.
//!
//! [`verify_component`] then checks **the bytes that will be loaded** — each pinned file at the
//! exact path handed back in [`VerifiedComponent`], following symlinks the way a loader does (a
//! download cache commonly stores snapshot entries as symlinks to content blobs):
//!
//! 1. every pinned file exists and is a regular file ([`AssetError::MissingFile`]) with the pinned
//!    size ([`AssetError::SizeMismatch`]) — all files first, so a missing or truncated shard fails
//!    before any hashing;
//! 2. no unpinned weights or index file (`*.safetensors`, `*.safetensors.index.json`, `*.bin`,
//!    `*.pt`, `*.pth`, `*.ckpt`) exists anywhere in the snapshot tree, subdirectories included
//!    ([`AssetError::UnexpectedWeightFile`]) — a stray shard or index could otherwise be picked up
//!    by a loader;
//! 3. every pinned file's SHA-256 ([`AssetError::HashMismatch`]);
//! 4. the weights file's safetensors header — tensor names, dtypes and shapes — equals the committed
//!    conversion manifest ([`AssetError::TensorTableMismatch`]).
//!
//! [`native_tensor_digests`] goes one step further for the fidelity check: it loads every tensor
//! through Candle's safetensors reader (the path the model code uses) and hashes the loaded values,
//! and [`compare_tensor_digests`] compares them with the pinned originals' digests in the manifest.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};

use candle_audio::candle_core::{self, CpuStorage, Device, Storage};
use sha2::{Digest, Sha256};

use crate::inventory::{Closure, Component, ComponentId, UpstreamRepo};
use crate::manifest::{ConversionManifest, TensorEntry};

/// Every way resolving or verifying a YuE2 asset fails. Each variant names the component and file,
/// so a failure is actionable rather than a generic load error.
#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    /// The pinned repository revision is not in the local snapshot store. Nothing is downloaded.
    #[error(
        "offline cache miss: {repo}@{revision} is not available locally (looked in {}); acquire \
         the pinned snapshot first — YuE2 never downloads at load time",
        looked_in.display()
    )]
    CacheMiss {
        /// The repository.
        repo: String,
        /// The pinned revision.
        revision: String,
        /// Where it was expected.
        looked_in: PathBuf,
    },
    /// A pinned file is absent (or a dangling symlink, or not a regular file).
    #[error("{component}: pinned file `{file}` is missing ({})", path.display())]
    MissingFile {
        /// The component key.
        component: String,
        /// The snapshot-relative path.
        file: String,
        /// The path checked.
        path: PathBuf,
    },
    /// A pinned file has the wrong size — truncated, partially downloaded or replaced.
    #[error("{component}: `{file}` is {actual} bytes, pinned {expected} (truncated or replaced)")]
    SizeMismatch {
        /// The component key.
        component: String,
        /// The snapshot-relative path.
        file: String,
        /// The pinned size.
        expected: u64,
        /// The size on disk.
        actual: u64,
    },
    /// A pinned file's content differs from the pinned revision.
    #[error(
        "{component}: `{file}` has SHA-256 {actual}, pinned {expected} (corrupt, or not the pinned \
         revision)"
    )]
    HashMismatch {
        /// The component key.
        component: String,
        /// The snapshot-relative path.
        file: String,
        /// The pinned SHA-256.
        expected: String,
        /// The SHA-256 of the bytes on disk.
        actual: String,
    },
    /// A weights file that is not pinned sits beside the pinned weights.
    #[error(
        "{component}: unexpected weights file `{file}` in {}; only the pinned shard set may be \
         present",
        dir.display()
    )]
    UnexpectedWeightFile {
        /// The component key.
        component: String,
        /// The unexpected file name.
        file: String,
        /// The snapshot directory.
        dir: PathBuf,
    },
    /// A weights file's safetensors header cannot be parsed.
    #[error("{component}: `{file}` has an invalid safetensors header: {detail}")]
    InvalidHeader {
        /// The component key.
        component: String,
        /// The snapshot-relative path.
        file: String,
        /// What is wrong.
        detail: String,
    },
    /// The weights file's tensor names / dtypes / shapes differ from the conversion manifest.
    #[error("{component}: tensor table differs from the conversion manifest: {detail}")]
    TensorTableMismatch {
        /// The component key.
        component: String,
        /// The first differences.
        detail: String,
    },
    /// Natively loaded tensor values differ from the pinned originals' digests.
    #[error("{component}: native tensor values differ from the pinned originals: {detail}")]
    TensorValueMismatch {
        /// The component key.
        component: String,
        /// The first differences.
        detail: String,
    },
    /// The committed conversion manifest is malformed or inconsistent with the inventory.
    #[error("{component}: conversion manifest: {detail}")]
    Manifest {
        /// The component key.
        component: String,
        /// What is wrong.
        detail: String,
    },
    /// Candle could not load a verified weights file.
    #[error("{component}: native load of `{file}` failed: {detail}")]
    NativeLoad {
        /// The component key.
        component: String,
        /// The snapshot-relative path.
        file: String,
        /// The loader's error.
        detail: String,
    },
    /// Reading a file failed for a reason other than its absence.
    #[error("{component}: reading {}: {source}", path.display())]
    Io {
        /// The component key.
        component: String,
        /// The path being read.
        path: PathBuf,
        /// The I/O error.
        source: std::io::Error,
    },
}

impl From<AssetError> for crate::gen_core::Error {
    fn from(e: AssetError) -> Self {
        crate::gen_core::Error::backend(e)
    }
}

/// The caller-provisioned local snapshot directory of each pinned repository.
///
/// Inference never fetches weights and never derives a download-cache location (epic 13657): the
/// application acquires each pinned repository and hands over its directory, keyed by repository
/// id (`"m-a-p/YuE2-3B"`). A plain directory records no revision, so the revision is established
/// by the pinned hashes during verification.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SnapshotDirs {
    dirs: BTreeMap<String, PathBuf>,
}

impl SnapshotDirs {
    /// No repositories provisioned.
    pub fn new() -> Self {
        Self::default()
    }

    /// Provision `repo_id`'s snapshot at `dir`.
    pub fn with(mut self, repo_id: impl Into<String>, dir: impl Into<PathBuf>) -> Self {
        self.dirs.insert(repo_id.into(), dir.into());
        self
    }

    /// The local snapshot directory for `repo`, or [`AssetError::CacheMiss`] when none was
    /// provisioned or the provisioned directory does not exist.
    pub fn snapshot_dir(&self, repo: &UpstreamRepo) -> Result<PathBuf, AssetError> {
        let miss = |looked_in: PathBuf| AssetError::CacheMiss {
            repo: repo.id.to_string(),
            revision: repo.revision.to_string(),
            looked_in,
        };
        let dir = self
            .dirs
            .get(repo.id)
            .ok_or_else(|| miss(PathBuf::from("<no directory provisioned>")))?;
        if dir.is_dir() {
            Ok(dir.clone())
        } else {
            Err(miss(dir.clone()))
        }
    }
}

/// One pinned file of a verified component, at the exact path whose bytes were hashed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedFile {
    /// The snapshot-relative pinned path.
    pub file: &'static str,
    /// The path that was verified — load from this path, not a sibling.
    pub path: PathBuf,
}

/// A component whose every pinned file was found and verified.
#[derive(Clone, Debug)]
pub struct VerifiedComponent {
    component: &'static Component,
    dir: PathBuf,
    files: Vec<VerifiedFile>,
    manifest: ConversionManifest,
}

impl VerifiedComponent {
    /// The inventory entry that was verified.
    pub fn component(&self) -> &'static Component {
        self.component
    }

    /// The snapshot directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Every verified file.
    pub fn files(&self) -> &[VerifiedFile] {
        &self.files
    }

    /// The verified path of pinned file `file`.
    pub fn path(&self, file: &str) -> Option<&Path> {
        self.files
            .iter()
            .find(|f| f.file == file)
            .map(|f| f.path.as_path())
    }

    /// The verified weights file, if the component has one.
    pub fn weights_path(&self) -> Option<&Path> {
        self.component.weights().and_then(|w| self.path(w.path))
    }

    /// The parsed conversion manifest the snapshot was checked against.
    pub fn manifest(&self) -> &ConversionManifest {
        &self.manifest
    }
}

/// Every component of a closure, verified.
///
/// Its fields are private, so [`resolve_closure`] is the only way to obtain one: holding a
/// `VerifiedClosure` means every component in it passed verification. A struct literal outside
/// this crate does not compile:
///
/// ```compile_fail
/// use candle_audio_yue2::{Closure, VerifiedClosure};
///
/// let forged = VerifiedClosure {
///     closure: Closure::Cover,
///     components: Vec::new(),
/// };
/// ```
#[derive(Clone, Debug)]
pub struct VerifiedClosure {
    closure: Closure,
    components: Vec<VerifiedComponent>,
}

impl VerifiedClosure {
    /// The closure that was resolved.
    pub fn closure(&self) -> Closure {
        self.closure
    }

    /// Its verified components, in [`Closure::components`] order.
    pub fn components(&self) -> &[VerifiedComponent] {
        &self.components
    }

    /// The verified component `id`, if it is part of this closure.
    pub fn get(&self, id: ComponentId) -> Option<&VerifiedComponent> {
        self.components.iter().find(|c| c.component.id == id)
    }
}

pub(crate) fn join_rel(dir: &Path, rel: &str) -> PathBuf {
    rel.split('/')
        .fold(dir.to_path_buf(), |p, part| p.join(part))
}

pub(crate) fn sha256_file(key: &str, path: &Path) -> Result<String, AssetError> {
    let io = |source| AssetError::Io {
        component: key.to_string(),
        path: path.to_path_buf(),
        source,
    };
    let mut file = File::open(path).map_err(io)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = file.read(&mut buf).map_err(io)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Parse the safetensors header of `path` into `name → (dtype, shape)`.
pub fn read_tensor_table(
    key: &str,
    file: &str,
    path: &Path,
) -> Result<BTreeMap<String, (String, Vec<usize>)>, AssetError> {
    let invalid = |detail: String| AssetError::InvalidHeader {
        component: key.to_string(),
        file: file.to_string(),
        detail,
    };
    let io = |source| AssetError::Io {
        component: key.to_string(),
        path: path.to_path_buf(),
        source,
    };
    let mut f = File::open(path).map_err(io)?;
    let file_len = f.metadata().map_err(io)?.len();
    let mut len = [0u8; 8];
    f.read_exact(&mut len)
        .map_err(|e| invalid(format!("cannot read the header length: {e}")))?;
    let header_len = u64::from_le_bytes(len);
    if header_len.saturating_add(8) > file_len || header_len > 100 << 20 {
        return Err(invalid(format!(
            "header length {header_len} does not fit a {file_len}-byte file"
        )));
    }
    let mut header = vec![0u8; header_len as usize];
    f.read_exact(&mut header)
        .map_err(|e| invalid(format!("cannot read the header: {e}")))?;
    let header: serde_json::Value =
        serde_json::from_slice(&header).map_err(|e| invalid(format!("header JSON: {e}")))?;
    let entries = header
        .as_object()
        .ok_or_else(|| invalid("header is not a JSON object".into()))?;
    let mut table = BTreeMap::new();
    for (name, v) in entries {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v
            .get("dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| invalid(format!("`{name}` has no dtype")))?;
        let shape = v
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| invalid(format!("`{name}` has no shape")))?
            .iter()
            .map(|d| {
                d.as_u64()
                    .map(|d| d as usize)
                    .ok_or_else(|| invalid(format!("`{name}` has a non-integer dimension")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        table.insert(name.clone(), (dtype.to_string(), shape));
    }
    Ok(table)
}

/// Compare a tensor table with the manifest's; `Err(detail)` lists the first differences.
fn diff_tables(
    manifest: &[TensorEntry],
    actual: &BTreeMap<String, (String, Vec<usize>)>,
) -> Result<(), String> {
    let mut problems = Vec::new();
    let mut expected = BTreeMap::new();
    for t in manifest {
        expected.insert(t.name.as_str(), (t.dtype.as_str(), t.shape.as_slice()));
    }
    for (name, (dtype, shape)) in &expected {
        match actual.get(*name) {
            None => problems.push(format!("`{name}` is missing")),
            Some((d, s)) if d != dtype || s.as_slice() != *shape => problems.push(format!(
                "`{name}` is {d} {s:?}, manifest says {dtype} {shape:?}"
            )),
            Some(_) => {}
        }
    }
    for name in actual.keys() {
        if !expected.contains_key(name.as_str()) {
            problems.push(format!("`{name}` is not in the manifest"));
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        let n = problems.len();
        problems.truncate(8);
        Err(format!("{n} difference(s): {}", problems.join("; ")))
    }
}

fn is_weights_file(name: &str) -> bool {
    name.ends_with(".safetensors")
        || name.ends_with(".safetensors.index.json")
        || name.ends_with(".bin")
        || name.ends_with(".pt")
        || name.ends_with(".pth")
        || name.ends_with(".ckpt")
}

/// Every weights / index file under `dir`, as `/`-separated paths relative to the snapshot root,
/// following symlinks the way a loader would. A symlink loop terminates the walk: the OS refuses
/// to resolve a path through too many symlinks (ELOOP), and that error is propagated.
pub(crate) fn collect_weight_files(
    key: &str,
    dir: &Path,
    prefix: &str,
    found: &mut Vec<String>,
) -> Result<(), AssetError> {
    let io = |path: &Path, source| AssetError::Io {
        component: key.to_string(),
        path: path.to_path_buf(),
        source,
    };
    for entry in std::fs::read_dir(dir).map_err(|e| io(dir, e))? {
        let entry = entry.map_err(|e| io(dir, e))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let rel = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        let path = entry.path();
        // `metadata` follows symlinks. A dangling link (NotFound) is not a directory, and if its
        // name is a weights name it still counts as a stray entry. Any other error — a symlink
        // loop the OS refuses to resolve, a permission error — is propagated, never skipped.
        match std::fs::metadata(&path) {
            Ok(meta) if meta.is_dir() => collect_weight_files(key, &path, &rel, found)?,
            Ok(_) => {
                if is_weights_file(&name) {
                    found.push(rel);
                }
            }
            Err(e) if e.kind() == ErrorKind::NotFound => {
                if is_weights_file(&name) {
                    found.push(rel);
                }
            }
            Err(e) => return Err(io(&path, e)),
        }
    }
    Ok(())
}

/// Verify `component` in the snapshot directory `dir` (see the module docs for the checks, in
/// order). The returned [`VerifiedComponent`] carries the exact paths whose bytes were hashed;
/// loaders must read those paths. The guarantee is as of this call — a file replaced afterwards is
/// outside it, so verify at the load boundary, not ahead of time.
pub fn verify_component(
    component: &'static Component,
    dir: &Path,
) -> Result<VerifiedComponent, AssetError> {
    let key = component.key;
    let manifest = component.conversion_manifest()?;
    if manifest.repo != component.repo.id || manifest.revision != component.repo.revision {
        return Err(AssetError::Manifest {
            component: key.to_string(),
            detail: format!(
                "describes {}@{}, the inventory pins {}@{}",
                manifest.repo, manifest.revision, component.repo.id, component.repo.revision
            ),
        });
    }

    // 1. Presence and size of every pinned file, before any hashing.
    let mut files = Vec::with_capacity(component.files.len());
    for pinned in component.files {
        let path = join_rel(dir, pinned.path);
        let missing = || AssetError::MissingFile {
            component: key.to_string(),
            file: pinned.path.to_string(),
            path: path.clone(),
        };
        let meta = match std::fs::metadata(&path) {
            Ok(meta) => meta,
            Err(e) if e.kind() == ErrorKind::NotFound => return Err(missing()),
            Err(source) => {
                return Err(AssetError::Io {
                    component: key.to_string(),
                    path,
                    source,
                })
            }
        };
        if !meta.is_file() {
            return Err(missing());
        }
        if meta.len() != pinned.bytes {
            return Err(AssetError::SizeMismatch {
                component: key.to_string(),
                file: pinned.path.to_string(),
                expected: pinned.bytes,
                actual: meta.len(),
            });
        }
        files.push(VerifiedFile {
            file: pinned.path,
            path,
        });
    }

    // 2. No unpinned weights anywhere in the snapshot tree.
    if let Some(weights) = component.weights() {
        let mut found = Vec::new();
        collect_weight_files(key, dir, "", &mut found)?;
        found.sort();
        if let Some(stray) = found.into_iter().find(|rel| rel != weights.path) {
            return Err(AssetError::UnexpectedWeightFile {
                component: key.to_string(),
                file: stray,
                dir: dir.to_path_buf(),
            });
        }
    }

    // 3. Content of every pinned file.
    for (pinned, file) in component.files.iter().zip(&files) {
        let actual = sha256_file(key, &file.path)?;
        if actual != pinned.sha256 {
            return Err(AssetError::HashMismatch {
                component: key.to_string(),
                file: pinned.path.to_string(),
                expected: pinned.sha256.to_string(),
                actual,
            });
        }
    }

    // 4. The weights header against the conversion manifest.
    if let Some(weights) = component.weights() {
        let file = files
            .iter()
            .find(|f| f.file == weights.path)
            .expect("the weights file was verified above");
        let table = read_tensor_table(key, weights.path, &file.path)?;
        diff_tables(&manifest.tensors, &table).map_err(|detail| {
            AssetError::TensorTableMismatch {
                component: key.to_string(),
                detail,
            }
        })?;
    }

    Ok(VerifiedComponent {
        component,
        dir: dir.to_path_buf(),
        files,
        manifest,
    })
}

/// Resolve component `id` from the provisioned `dirs` and verify it.
pub fn resolve_component(
    id: ComponentId,
    dirs: &SnapshotDirs,
) -> Result<VerifiedComponent, AssetError> {
    let component = id.component();
    let dir = dirs.snapshot_dir(&component.repo)?;
    verify_component(component, &dir)
}

/// Resolve and verify every component of `closure`. The first failure is returned; a closure is
/// never partially usable.
pub fn resolve_closure(
    closure: Closure,
    dirs: &SnapshotDirs,
) -> Result<VerifiedClosure, AssetError> {
    let components = closure
        .components()
        .iter()
        .map(|&id| resolve_component(id, dirs))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(VerifiedClosure {
        closure,
        components,
    })
}

fn dtype_tag(dtype: candle_core::DType) -> Option<&'static str> {
    use candle_core::DType;
    Some(match dtype {
        DType::BF16 => "BF16",
        DType::F16 => "F16",
        DType::F32 => "F32",
        DType::F64 => "F64",
        DType::U8 => "U8",
        DType::U32 => "U32",
        DType::I64 => "I64",
        _ => return None,
    })
}

/// Hash a slice's elements as little-endian bytes, in chunks (endianness-independent).
pub(crate) fn hash_le<T: Copy, const N: usize>(
    hasher: &mut Sha256,
    values: &[T],
    to_le: impl Fn(T) -> [u8; N],
) {
    let mut buf = Vec::with_capacity(64 * 1024);
    for chunk in values.chunks((64 * 1024) / N) {
        buf.clear();
        for &v in chunk {
            buf.extend_from_slice(&to_le(v));
        }
        hasher.update(&buf);
    }
}

/// Load every tensor of `verified`'s weights file through Candle's safetensors reader on the CPU and
/// digest the loaded values (name, dtype, shape, SHA-256 of the little-endian element bytes), in
/// the manifest's order. The values hashed are the tensor storage Candle hands to model code — not
/// the file bytes — so a loader that reinterpreted, converted or reordered anything would show here.
pub fn native_tensor_digests(verified: &VerifiedComponent) -> Result<Vec<TensorEntry>, AssetError> {
    let key = verified.component.key;
    let weights = verified
        .component
        .weights()
        .ok_or_else(|| AssetError::Manifest {
            component: key.to_string(),
            detail: "component has no weights file".into(),
        })?;
    let path = verified
        .path(weights.path)
        .expect("a verified component has a verified weights path");
    let native = |detail: String| AssetError::NativeLoad {
        component: key.to_string(),
        file: weights.path.to_string(),
        detail,
    };
    // SAFETY: memory-mapping is sound as long as the file is not mutated while mapped; the file was
    // just integrity-verified and is only read. Same contract as every Candle weights load.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
        .map_err(|e| native(e.to_string()))?;
    let mut names: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
    let order: BTreeMap<&str, usize> = verified
        .manifest
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.as_str(), i))
        .collect();
    names.sort_by_key(|n| {
        (
            order.get(n.as_str()).copied().unwrap_or(usize::MAX),
            n.clone(),
        )
    });

    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let t = st
            .load(&name, &Device::Cpu)
            .map_err(|e| native(format!("`{name}`: {e}")))?;
        let dtype = dtype_tag(t.dtype())
            .ok_or_else(|| native(format!("`{name}`: unsupported dtype {:?}", t.dtype())))?;
        let shape = t.dims().to_vec();
        let (storage, layout) = t.storage_and_layout();
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            return Err(native(format!("`{name}` did not load as a dense tensor")));
        }
        let n = layout.shape().elem_count();
        let mut hasher = Sha256::new();
        let Storage::Cpu(cpu) = &*storage else {
            return Err(native(format!("`{name}` did not load on the CPU")));
        };
        match cpu {
            CpuStorage::BF16(v) => hash_le(&mut hasher, &v[..n], |x| x.to_bits().to_le_bytes()),
            CpuStorage::F16(v) => hash_le(&mut hasher, &v[..n], |x| x.to_bits().to_le_bytes()),
            CpuStorage::F32(v) => hash_le(&mut hasher, &v[..n], |x| x.to_bits().to_le_bytes()),
            CpuStorage::F64(v) => hash_le(&mut hasher, &v[..n], |x| x.to_bits().to_le_bytes()),
            CpuStorage::U8(v) => hasher.update(&v[..n]),
            CpuStorage::U32(v) => hash_le(&mut hasher, &v[..n], |x| x.to_le_bytes()),
            CpuStorage::I64(v) => hash_le(&mut hasher, &v[..n], |x| x.to_le_bytes()),
            _ => return Err(native(format!("`{name}`: unsupported CPU storage"))),
        }
        out.push(TensorEntry {
            name,
            dtype: dtype.to_string(),
            shape,
            sha256: hex(&hasher.finalize()),
        });
    }
    Ok(out)
}

/// Compare natively loaded tensor digests with the pinned originals' (the manifest): the same
/// tensor set, and for each tensor the same dtype, shape and value digest.
pub fn compare_tensor_digests(
    verified: &VerifiedComponent,
    native: &[TensorEntry],
) -> Result<(), AssetError> {
    let key = verified.component.key;
    let expected = &verified.manifest.tensors;
    let mut problems = Vec::new();
    let by_name: BTreeMap<&str, &TensorEntry> =
        native.iter().map(|t| (t.name.as_str(), t)).collect();
    for want in expected {
        match by_name.get(want.name.as_str()) {
            None => problems.push(format!("`{}` was not loaded", want.name)),
            Some(got) if *got != want => problems.push(format!(
                "`{}` loaded as {} {:?} sha256 {}, original is {} {:?} sha256 {}",
                want.name, got.dtype, got.shape, got.sha256, want.dtype, want.shape, want.sha256
            )),
            Some(_) => {}
        }
    }
    if native.len() != expected.len() {
        problems.push(format!(
            "loaded {} tensors, the original has {}",
            native.len(),
            expected.len()
        ));
    }
    if problems.is_empty() {
        Ok(())
    } else {
        let n = problems.len();
        problems.truncate(8);
        Err(AssetError::TensorValueMismatch {
            component: key.to_string(),
            detail: format!("{n} difference(s): {}", problems.join("; ")),
        })
    }
}

/// [`native_tensor_digests`] + [`compare_tensor_digests`]: returns the number of tensors whose
/// natively loaded values reproduce the pinned originals.
pub fn verify_native_tensors(verified: &VerifiedComponent) -> Result<usize, AssetError> {
    let native = native_tensor_digests(verified)?;
    compare_tensor_digests(verified, &native)?;
    Ok(native.len())
}

#[cfg(test)]
mod tests;
