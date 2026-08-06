//! Content-addressed, file-backed device-format weights for Candle block windows (SC-16096).
//!
//! MLX-packed q4/q8 snapshots store every quantized projection as an affine triple:
//! `{base}.weight` (u32 codes), `{base}.scales`, and `{base}.biases`. Candle consumes those bytes as
//! GGML `Q4_1`/`Q8_0`, but converting the triple inside a block-window loop repeats an invariant
//! host conversion on every forward. This module moves that conversion to component-open time:
//!
//! 1. hash the three source tensor views (including dtype, shape, group size, and format version);
//! 2. convert each projection once through the existing [`super::repack_packed_weight`] implementation;
//! 3. atomically write the resulting GGML bytes to a one-tensor safetensors sidecar under
//!    `.candle-device-format-v1/` beside the source component, or under the external cache when the
//!    caller-provisioned component is read-only; and
//! 4. mmap that sidecar for each materialization and copy its already-device-format bytes directly
//!    to the requested device.
//!
//! The cache is content-addressed over the source bytes, not timestamps. Replacing or changing a
//! tier therefore selects a different sidecar path automatically. A window holds no anonymous host
//! allocation proportional to the tier: the mapped payload is reclaimable page cache, and
//! [`QStorage::from_data`] copies it to the device before the mapping is dropped. First creation is
//! deliberately projection-at-a-time, bounding the q8 dense conversion transient to one projection.
//! A complete valid cache is opened read-only without creating or acquiring the preparation lock.
//! After one full validation, path-owning preparation records a process-local memo keyed by canonical
//! component/cache/config identity plus the exact source file generations it opened. Artifact file
//! ID and change-time are checked around header validation, so an unchanged reopen hashes zero source
//! and payload bytes while replacement or in-place mutation drops back to full digest validation and
//! serialized recovery. Platforms without a mutation-resistant file stamp conservatively skip memoing.
//! Missing or corrupt entries still take the exclusive-lock path so recovery and publication remain
//! serialized. The external root can be supplied explicitly or through
//! `SCENEWORKS_CANDLE_DEVICE_CACHE_DIR`; otherwise the platform's per-user cache directory is used.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Error, Result, Tensor};
use fs2::FileExt;
use safetensors::tensor::{Dtype as SafeDtype, View};
use sha2::{Digest, Sha256};

use super::{repack_packed_weight, PackedConfig};
use crate::gen_core::CancelFlag;

const CACHE_DIR: &str = ".candle-device-format-v1";
const PREPARE_LOCK: &str = ".prepare.lock";
const PAYLOAD_KEY: &str = "weight";
const PAYLOAD_HASH_KEY: &str = "payload_sha256";
const FORMAT_DOMAIN: &[u8] = b"sceneworks-candle-device-format-sidecar-v1\0";
const HASH_CANCEL_CHUNK_BYTES: usize = 4 * 1024 * 1024;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
struct SidecarEntry {
    path: PathBuf,
    dtype: GgmlDType,
    out_dim: usize,
    in_dim: usize,
    payload_bytes: usize,
}

#[derive(Clone, Debug)]
struct PreparedProjection {
    entry_base: String,
    source_hex: String,
    source_base: String,
    source_slice: Option<usize>,
    bits: usize,
    group_size: usize,
    dtype: GgmlDType,
    dtype_name: &'static str,
    out_dim: usize,
    in_dim: usize,
    payload_bytes: usize,
    source_bytes: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct FileIdentity {
    path: PathBuf,
    len: u64,
    stamp: PlatformFileStamp,
}

#[cfg(unix)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct PlatformFileStamp {
    device: u64,
    inode: u64,
    change_seconds: i64,
    change_nanos: i64,
}

#[cfg(windows)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct PlatformFileStamp {
    volume: u64,
    file_id: [u8; 16],
    change_time: i64,
}

#[cfg(not(any(unix, windows)))]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct PlatformFileStamp;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ValidationMemoKey {
    component_dir: PathBuf,
    cache_identity: PathBuf,
    bits: usize,
    group_size: usize,
    base_prefix: Option<String>,
    sources: Vec<FileIdentity>,
}

#[derive(Clone, Debug)]
struct ValidationMemoValue {
    projections: Vec<PreparedProjection>,
    cache_dir: PathBuf,
    artifacts: Vec<FileIdentity>,
}

fn validation_memo() -> &'static Mutex<HashMap<ValidationMemoKey, ValidationMemoValue>> {
    static MEMO: OnceLock<Mutex<HashMap<ValidationMemoKey, ValidationMemoValue>>> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(test)]
std::thread_local! {
    static AFTER_SOURCE_OPEN_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        std::cell::RefCell::new(None);
}

#[cfg(test)]
fn run_after_source_open_hook() {
    AFTER_SOURCE_OPEN_HOOK.with(|hook| {
        if let Some(hook) = hook.borrow_mut().take() {
            hook();
        }
    });
}

/// Prepared device-format sidecars for every MLX-packed projection in one component.
///
/// A cold construction performs all source hashing and any missing conversions; an unchanged
/// process-local warm reopen opened through a path-bound helper uses the validated file-generation
/// memo. [`Self::load`] touches only the sidecar: it does not retain or consult the source
/// `MmapedSafetensors`, which is the property a block-window loader needs in order to exclude format
/// conversion from the per-window path.
#[derive(Debug)]
pub struct PackedWeightSidecars {
    entries: HashMap<String, SidecarEntry>,
    prepared: Vec<PreparedProjection>,
    cache_dir: PathBuf,
    created: usize,
    reused: usize,
    source_bytes_hashed: u64,
    payload_bytes_hashed: u64,
    sidecar_bytes: u64,
}

impl PackedWeightSidecars {
    /// Prepare all packed projections from `source`, placing content-addressed artifacts beside the
    /// component directory. Existing valid artifacts are mmap-validated and reused.
    pub fn prepare(
        source: &MmapedSafetensors,
        component_dir: &Path,
        packed: PackedConfig,
        device: &Device,
    ) -> Result<Self> {
        Self::prepare_impl(
            source,
            component_dir,
            packed,
            device,
            None,
            None,
            None,
            None,
        )
    }

    /// Open the exact source paths and prepare their sidecars with mutation-resistant warm memoing.
    ///
    /// A bare [`MmapedSafetensors`] does not expose its backing files, so [`Self::prepare`] remains a
    /// conservative full-validation helper. Callers that own paths should use this entry point: it
    /// identity-sandwiches the mmap and preparation, and only then records a zero-hash warm memo.
    pub fn open_and_prepare(
        source_paths: &[PathBuf],
        component_dir: &Path,
        packed: PackedConfig,
        device: &Device,
    ) -> Result<(MmapedSafetensors, Self)> {
        Self::open_and_prepare_impl(
            source_paths,
            component_dir,
            packed,
            device,
            None,
            None,
            None,
        )
    }

    /// Prepare packed projections with an explicit non-model cache root for read-only components.
    ///
    /// The model-adjacent cache remains preferred when it is writable. `external_cache_root` is used
    /// only when that cache cannot be written (or when an already-complete cache exists there). Each
    /// component receives a hashed namespace below the root, so unrelated snapshots do not share a
    /// lock. This is the programmatic twin of `SCENEWORKS_CANDLE_DEVICE_CACHE_DIR`.
    pub fn prepare_with_external_cache_root(
        source: &MmapedSafetensors,
        component_dir: &Path,
        packed: PackedConfig,
        device: &Device,
        external_cache_root: &Path,
    ) -> Result<Self> {
        Self::prepare_impl(
            source,
            component_dir,
            packed,
            device,
            None,
            None,
            Some(external_cache_root),
            None,
        )
    }

    /// Path-bound counterpart of [`Self::prepare_with_external_cache_root`].
    pub fn open_and_prepare_with_external_cache_root(
        source_paths: &[PathBuf],
        component_dir: &Path,
        packed: PackedConfig,
        device: &Device,
        external_cache_root: &Path,
    ) -> Result<(MmapedSafetensors, Self)> {
        Self::open_and_prepare_impl(
            source_paths,
            component_dir,
            packed,
            device,
            None,
            None,
            Some(external_cache_root),
        )
    }

    /// Cancellation-aware preparation for request-time component opens. Hashing checks between
    /// bounded chunks and conversion checks between independently addressed projections.
    pub fn prepare_cancelable(
        source: &MmapedSafetensors,
        component_dir: &Path,
        packed: PackedConfig,
        device: &Device,
        cancel: &CancelFlag,
    ) -> Result<Self> {
        Self::prepare_impl(
            source,
            component_dir,
            packed,
            device,
            Some(cancel),
            None,
            None,
            None,
        )
    }

    /// Path-bound counterpart of [`Self::prepare_cancelable`].
    pub fn open_and_prepare_cancelable(
        source_paths: &[PathBuf],
        component_dir: &Path,
        packed: PackedConfig,
        device: &Device,
        cancel: &CancelFlag,
    ) -> Result<(MmapedSafetensors, Self)> {
        Self::open_and_prepare_impl(
            source_paths,
            component_dir,
            packed,
            device,
            Some(cancel),
            None,
            None,
        )
    }

    /// Prepare only packed projections whose full tensor base starts with `base_prefix`.
    ///
    /// This is the bounded-residency variant for providers whose streamed stack is one prefix inside
    /// a larger component. It avoids hashing and materializing sidecars for weights that remain
    /// resident, while retaining the same content address and cache lifecycle as [`Self::prepare`].
    pub fn prepare_prefix_cancelable(
        source: &MmapedSafetensors,
        component_dir: &Path,
        packed: PackedConfig,
        device: &Device,
        cancel: &CancelFlag,
        base_prefix: &str,
    ) -> Result<Self> {
        if base_prefix.is_empty() {
            return Err(Error::Msg(
                "device-format sidecar: base prefix must not be empty".to_owned(),
            ));
        }
        Self::prepare_impl(
            source,
            component_dir,
            packed,
            device,
            Some(cancel),
            Some(base_prefix),
            None,
            None,
        )
    }

    /// Path-bound counterpart of [`Self::prepare_prefix_cancelable`].
    pub fn open_and_prepare_prefix_cancelable(
        source_paths: &[PathBuf],
        component_dir: &Path,
        packed: PackedConfig,
        device: &Device,
        cancel: &CancelFlag,
        base_prefix: &str,
    ) -> Result<(MmapedSafetensors, Self)> {
        if base_prefix.is_empty() {
            return Err(Error::Msg(
                "device-format sidecar: base prefix must not be empty".to_owned(),
            ));
        }
        Self::open_and_prepare_impl(
            source_paths,
            component_dir,
            packed,
            device,
            Some(cancel),
            Some(base_prefix),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn open_and_prepare_impl(
        source_paths: &[PathBuf],
        component_dir: &Path,
        packed: PackedConfig,
        device: &Device,
        cancel: Option<&CancelFlag>,
        base_prefix: Option<&str>,
        external_cache_root: Option<&Path>,
    ) -> Result<(MmapedSafetensors, Self)> {
        if source_paths.is_empty() {
            return Err(Error::Msg(
                "device-format sidecar: at least one source path is required".to_owned(),
            ));
        }
        check_cancel(cancel)?;
        let before = file_identities(source_paths)?;
        // SAFETY: read-only mappings of the exact caller-supplied source files. The robust identity
        // checks immediately before and after open detect replacement across this pathname seam.
        let source = unsafe { MmapedSafetensors::multi(source_paths)? };
        #[cfg(test)]
        run_after_source_open_hook();
        let after_open = file_identities(source_paths)?;
        if before.is_some() && before != after_open {
            return Err(Error::Msg(
                "device-format sidecar: source files changed while opening their mappings"
                    .to_owned(),
            ));
        }
        let memo_sources = after_open.as_deref();
        let cache = Self::prepare_impl(
            &source,
            component_dir,
            packed,
            device,
            cancel,
            base_prefix,
            external_cache_root,
            memo_sources,
        )?;
        let after_prepare = file_identities(source_paths)?;
        if after_open.is_some() && after_open != after_prepare {
            if let Some(sources) = memo_sources {
                if let Ok(key) = validation_memo_key(
                    component_dir,
                    external_cache_root,
                    usize::try_from(packed.bits).unwrap_or_default(),
                    usize::try_from(packed.group_size).unwrap_or_default(),
                    base_prefix,
                    sources,
                ) {
                    validation_memo()
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .remove(&key);
                }
            }
            return Err(Error::Msg(
                "device-format sidecar: source files changed during preparation".to_owned(),
            ));
        }
        Ok((source, cache))
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_impl(
        source: &MmapedSafetensors,
        component_dir: &Path,
        packed: PackedConfig,
        device: &Device,
        cancel: Option<&CancelFlag>,
        base_prefix: Option<&str>,
        external_cache_root: Option<&Path>,
        memo_sources: Option<&[FileIdentity]>,
    ) -> Result<Self> {
        check_cancel(cancel)?;
        let bits = usize::try_from(packed.bits).map_err(|_| {
            Error::Msg(format!(
                "device-format sidecar: invalid packed bit width {}",
                packed.bits
            ))
        })?;
        let group_size = usize::try_from(packed.group_size).map_err(|_| {
            Error::Msg(format!(
                "device-format sidecar: invalid group size {}",
                packed.group_size
            ))
        })?;
        if !matches!(bits, 4 | 8) || group_size == 0 {
            return Err(Error::Msg(format!(
                "device-format sidecar: expected q4/q8 and a positive group size, got bits={bits} \
                 group_size={group_size}"
            )));
        }

        let memo_key = memo_sources
            .map(|sources| {
                validation_memo_key(
                    component_dir,
                    external_cache_root,
                    bits,
                    group_size,
                    base_prefix,
                    sources,
                )
            })
            .transpose()?;
        if let Some(key) = memo_key.as_ref() {
            if let Some(value) = validation_memo_hit(key, cancel)? {
                return Ok(reused_cache(value.projections, value.cache_dir, 0, 0));
            }
        }

        let projections =
            prepare_projections(source, component_dir, bits, group_size, cancel, base_prefix)?;
        let source_bytes_hashed = projections.iter().fold(0u64, |sum, projection| {
            sum.saturating_add(projection.source_bytes as u64)
        });

        // A memo miss still takes the lock-free, read-only full validation path before attempting
        // create_dir_all or opening the writable lock. Published files are immutable, so a complete
        // valid set needs no coordination with writers.
        let adjacent_cache = component_dir.join(CACHE_DIR);
        let mut payload_bytes_hashed = 0u64;
        if cache_is_complete(
            &adjacent_cache,
            &projections,
            cancel,
            &mut payload_bytes_hashed,
        )? {
            let cache = reused_cache(
                projections,
                adjacent_cache,
                source_bytes_hashed,
                payload_bytes_hashed,
            );
            if let Some(key) = memo_key.as_ref() {
                remember_validation(key, &cache)?;
            }
            return Ok(cache);
        }

        let external_cache = external_cache_dir(component_dir, external_cache_root);
        if cache_is_complete(
            &external_cache,
            &projections,
            cancel,
            &mut payload_bytes_hashed,
        )? {
            let cache = reused_cache(
                projections,
                external_cache,
                source_bytes_hashed,
                payload_bytes_hashed,
            );
            if let Some(key) = memo_key.as_ref() {
                remember_validation(key, &cache)?;
            }
            return Ok(cache);
        }

        // Prefer the historical model-adjacent location when it is writable. A caller-provisioned
        // snapshot may legally be immutable, though, so failure to create/write there selects the
        // namespaced per-user external cache instead of disabling packed loading or retaining a full
        // tier's converted weights in anonymous memory.
        let (cache_dir, prepare_lock) = match open_writable_cache(&adjacent_cache) {
            Ok(lock) => (adjacent_cache, lock),
            Err(adjacent_error) => match open_writable_cache(&external_cache) {
                Ok(lock) => (external_cache, lock),
                Err(external_error) => {
                    return Err(Error::Msg(format!(
                        "device-format sidecar: neither model-adjacent cache {} ({adjacent_error}) \
                         nor external cache {} ({external_error}) is writable",
                        adjacent_cache.display(),
                        external_cache.display()
                    )))
                }
            },
        };

        // Serialize validation, corrupt recovery, and publication across processes. Without this
        // lock, two readers can both observe a corrupt final path and one can delete the valid
        // replacement the other just published. The lock file persists as a zero-byte cache
        // coordination artifact; the OS releases the advisory lock on every return or process exit.
        if cancel.is_some() {
            loop {
                check_cancel(cancel)?;
                match FileExt::try_lock_exclusive(&prepare_lock) {
                    Ok(()) => break,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    Err(error) => {
                        return Err(Error::Msg(format!(
                            "device-format sidecar: lock {}: {error}",
                            cache_dir.display()
                        )))
                    }
                }
            }
        } else {
            FileExt::lock_exclusive(&prepare_lock).map_err(|e| {
                Error::Msg(format!(
                    "device-format sidecar: lock {}: {e}",
                    cache_dir.display()
                ))
            })?;
        }

        let prepared = projections.clone();
        let mut entries = HashMap::with_capacity(projections.len());
        let mut created = 0usize;
        let mut reused = 0usize;
        let mut sidecar_bytes = 0u64;
        for projection in projections {
            check_cancel(cancel)?;
            let path = cache_dir.join(format!(
                "{}.{}.safetensors",
                projection.source_hex, projection.dtype_name
            ));

            let valid = validate_sidecar_counted(
                &path,
                projection.payload_bytes,
                cancel,
                &mut payload_bytes_hashed,
            )?;
            if valid {
                reused += 1;
            } else {
                build_sidecar(
                    source,
                    &path,
                    &projection.source_hex,
                    &projection.source_base,
                    projection.source_slice,
                    projection.bits,
                    projection.group_size,
                    projection.out_dim,
                    projection.in_dim,
                    projection.dtype,
                    device,
                )?;
                check_cancel(cancel)?;
                if !validate_sidecar_counted(
                    &path,
                    projection.payload_bytes,
                    cancel,
                    &mut payload_bytes_hashed,
                )? {
                    return Err(Error::Msg(format!(
                        "device-format sidecar: freshly written artifact {} failed validation",
                        path.display()
                    )));
                }
                created += 1;
            }
            sidecar_bytes = sidecar_bytes.saturating_add(projection.payload_bytes as u64);
            entries.insert(
                projection.entry_base,
                SidecarEntry {
                    path,
                    dtype: projection.dtype,
                    out_dim: projection.out_dim,
                    in_dim: projection.in_dim,
                    payload_bytes: projection.payload_bytes,
                },
            );
        }

        let cache = Self {
            entries,
            prepared,
            cache_dir,
            created,
            reused,
            source_bytes_hashed,
            payload_bytes_hashed,
            sidecar_bytes,
        };
        if let Some(key) = memo_key.as_ref() {
            remember_validation(key, &cache)?;
        }
        Ok(cache)
    }

    /// Materialize one already-converted projection on `device` from its mapped sidecar bytes.
    ///
    /// No source tensor or conversion parameter is accepted here by design: a caller cannot
    /// accidentally put the MLX-affine repack back into a block-window loop.
    pub fn load(&self, base: &str, device: &Device) -> Result<QTensor> {
        let entry = self.entries.get(base).ok_or_else(|| {
            Error::Msg(format!(
                "device-format sidecar: no prepared projection `{base}` in {}",
                self.cache_dir.display()
            ))
        })?;
        // SAFETY: immutable, atomically-published cache file. Writers never modify a published path.
        let mapped = unsafe { MmapedSafetensors::new(&entry.path) }.map_err(|e| {
            Error::Msg(format!(
                "device-format sidecar: mmap {}: {e}",
                entry.path.display()
            ))
        })?;
        let payload = mapped.get(PAYLOAD_KEY).map_err(|e| {
            Error::Msg(format!(
                "device-format sidecar: {} lacks `{PAYLOAD_KEY}`: {e}",
                entry.path.display()
            ))
        })?;
        if payload.dtype() != SafeDtype::U8 || payload.data().len() != entry.payload_bytes {
            return Err(Error::Msg(format!(
                "device-format sidecar: {} payload changed after preparation (dtype {:?}, {} \
                 bytes; expected U8, {} bytes)",
                entry.path.display(),
                payload.dtype(),
                payload.data().len(),
                entry.payload_bytes
            )));
        }
        let storage = QStorage::from_data(Cow::Borrowed(payload.data()), device, entry.dtype)
            .map_err(|e| {
                Error::Msg(format!(
                    "device-format sidecar: transfer {} to {device:?}: {e}",
                    entry.path.display()
                ))
            })?;
        QTensor::new(storage, (entry.out_dim, entry.in_dim))
    }

    /// Materialize one slice of a rank-3 packed projection (for example one MoE expert).
    /// Rank-2 callers continue to use [`Self::load`].
    pub fn load_slice(&self, base: &str, index: usize, device: &Device) -> Result<QTensor> {
        self.load(&sidecar_entry_key(base, Some(index)), device)
    }

    pub fn contains(&self, base: &str) -> bool {
        self.entries.contains_key(base)
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn created_count(&self) -> usize {
        self.created
    }

    pub fn reused_count(&self) -> usize {
        self.reused
    }

    pub fn source_bytes_hashed(&self) -> u64 {
        self.source_bytes_hashed
    }

    /// Payload bytes hashed while validating artifacts during this open. A process-local validated
    /// warm reopen reports zero; a cold-process validation reports the bytes it actually inspected.
    pub fn payload_bytes_hashed(&self) -> u64 {
        self.payload_bytes_hashed
    }

    pub fn sidecar_bytes(&self) -> u64 {
        self.sidecar_bytes
    }

    /// The exact content-addressed artifact for `base`, useful to diagnostics and evidence harnesses.
    pub fn path_for(&self, base: &str) -> Option<&Path> {
        self.entries.get(base).map(|entry| entry.path.as_path())
    }
}

fn prepare_projections(
    source: &MmapedSafetensors,
    component_dir: &Path,
    bits: usize,
    group_size: usize,
    cancel: Option<&CancelFlag>,
    base_prefix: Option<&str>,
) -> Result<Vec<PreparedProjection>> {
    let mut bases: Vec<String> = source
        .tensors()
        .into_iter()
        .filter_map(|(name, _)| name.strip_suffix(".scales").map(str::to_owned))
        .filter(|base| base_prefix.is_none_or(|prefix| base.starts_with(prefix)))
        .collect();
    bases.sort();
    bases.dedup();
    if bases.is_empty() {
        let selection = base_prefix
            .map(|prefix| format!(" matching prefix `{prefix}`"))
            .unwrap_or_default();
        return Err(Error::Msg(format!(
            "device-format sidecar: packed component {} has no `.scales` triples{selection}",
            component_dir.display(),
        )));
    }

    let mut prepared = Vec::new();
    for base in bases {
        check_cancel(cancel)?;
        let weight_key = format!("{base}.weight");
        let scales_key = format!("{base}.scales");
        let biases_key = format!("{base}.biases");
        let weight = source.get(&weight_key).map_err(|e| {
            Error::Msg(format!(
                "device-format sidecar: `{scales_key}` has no `{weight_key}` sibling: {e}"
            ))
        })?;
        let scales = source.get(&scales_key)?;
        let biases = source.get(&biases_key).map_err(|e| {
            Error::Msg(format!(
                "device-format sidecar: `{scales_key}` has no `{biases_key}` sibling: {e}"
            ))
        })?;
        let projections =
            validate_source_shapes(&base, &weight, &scales, &biases, bits, group_size)?;
        for projection in projections {
            let entry_base = sidecar_entry_key(&base, projection.index);
            let source_digest = match projection.index {
                None => source_digest(
                    &base,
                    bits,
                    group_size,
                    [
                        (&weight_key, &weight),
                        (&scales_key, &scales),
                        (&biases_key, &biases),
                    ],
                    cancel,
                )?,
                Some(index) => source_slice_digest(
                    &entry_base,
                    bits,
                    group_size,
                    index,
                    [
                        (&weight_key, &weight),
                        (&scales_key, &scales),
                        (&biases_key, &biases),
                    ],
                    cancel,
                )?,
            };
            check_cancel(cancel)?;
            let dtype = if bits == 4 {
                GgmlDType::Q4_1
            } else {
                GgmlDType::Q8_0
            };
            prepared.push(PreparedProjection {
                entry_base,
                source_hex: hex(&source_digest),
                source_base: base.clone(),
                source_slice: projection.index,
                bits,
                group_size,
                dtype,
                dtype_name: if bits == 4 { "q4_1" } else { "q8_0" },
                out_dim: projection.out_dim,
                in_dim: projection.in_dim,
                payload_bytes: payload_len(dtype, projection.out_dim, projection.in_dim)?,
                source_bytes: projection.source_bytes,
            });
        }
    }
    Ok(prepared)
}

fn cache_is_complete(
    cache_dir: &Path,
    projections: &[PreparedProjection],
    cancel: Option<&CancelFlag>,
    payload_bytes_hashed: &mut u64,
) -> Result<bool> {
    if !cache_dir.is_dir() {
        return Ok(false);
    }
    for projection in projections {
        check_cancel(cancel)?;
        let path = cache_dir.join(format!(
            "{}.{}.safetensors",
            projection.source_hex, projection.dtype_name
        ));
        let valid = validate_sidecar_counted(
            &path,
            projection.payload_bytes,
            cancel,
            payload_bytes_hashed,
        )?;
        check_cancel(cancel)?;
        if !valid {
            return Ok(false);
        }
    }
    Ok(true)
}

fn reused_cache(
    projections: Vec<PreparedProjection>,
    cache_dir: PathBuf,
    source_bytes_hashed: u64,
    payload_bytes_hashed: u64,
) -> PackedWeightSidecars {
    let prepared = projections.clone();
    let reused = projections.len();
    let mut entries = HashMap::with_capacity(reused);
    let mut sidecar_bytes = 0u64;
    for projection in projections {
        let path = cache_dir.join(format!(
            "{}.{}.safetensors",
            projection.source_hex, projection.dtype_name
        ));
        sidecar_bytes = sidecar_bytes.saturating_add(projection.payload_bytes as u64);
        entries.insert(
            projection.entry_base,
            SidecarEntry {
                path,
                dtype: projection.dtype,
                out_dim: projection.out_dim,
                in_dim: projection.in_dim,
                payload_bytes: projection.payload_bytes,
            },
        );
    }
    PackedWeightSidecars {
        entries,
        prepared,
        cache_dir,
        created: 0,
        reused,
        source_bytes_hashed,
        payload_bytes_hashed,
        sidecar_bytes,
    }
}

fn open_writable_cache(cache_dir: &Path) -> std::io::Result<fs::File> {
    fs::create_dir_all(cache_dir)?;
    // Probe create+unlink before creating/opening the coordination lock. This runs only after the
    // lock-free complete-cache path missed, proves the directory can publish an artifact, and avoids
    // leaving a new lock file in a location that cannot actually accept the sidecars. `create_new`
    // prevents following or truncating an attacker-controlled file.
    let seq = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let probe = cache_dir.join(format!(".write-probe-{}-{seq}", std::process::id()));
    fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&probe)?;
    fs::remove_file(probe)?;
    fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(cache_dir.join(PREPARE_LOCK))
}

fn external_cache_dir(component_dir: &Path, explicit_root: Option<&Path>) -> PathBuf {
    let root = explicit_root
        .map(Path::to_path_buf)
        .unwrap_or_else(default_external_cache_root);
    let identity = fs::canonicalize(component_dir).unwrap_or_else(|_| component_dir.to_path_buf());
    let mut digest = Sha256::new();
    digest.update(b"sceneworks-candle-device-format-component-v1\0");
    digest.update(identity.to_string_lossy().as_bytes());
    let namespace = hex(&digest.finalize());
    root.join("candle-device-format-v1").join(namespace)
}

fn default_external_cache_root() -> PathBuf {
    if let Some(root) =
        std::env::var_os("SCENEWORKS_CANDLE_DEVICE_CACHE_DIR").filter(|root| !root.is_empty())
    {
        return PathBuf::from(root);
    }
    #[cfg(target_os = "windows")]
    if let Some(root) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(root).join("SceneWorks");
    }
    #[cfg(target_os = "macos")]
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join("Library/Caches/SceneWorks");
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if let Some(root) = std::env::var_os("XDG_CACHE_HOME") {
            return PathBuf::from(root).join("sceneworks");
        }
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(".cache/sceneworks");
        }
    }
    // Temp directories are user-scoped on supported Windows/macOS hosts and commonly so in
    // containers. This last-resort cache is content-validated before every use; deployments that
    // need a durable or policy-controlled location should set the documented override.
    std::env::temp_dir().join("sceneworks-candle-cache")
}

fn file_identity(path: &Path) -> std::io::Result<Option<FileIdentity>> {
    let path = fs::canonicalize(path)?;
    let file = File::open(&path)?;
    let metadata = file.metadata()?;
    let Some(stamp) = platform_file_stamp(&file, &metadata)? else {
        return Ok(None);
    };
    Ok(Some(FileIdentity {
        path,
        len: metadata.len(),
        stamp,
    }))
}

fn file_identities(paths: &[PathBuf]) -> Result<Option<Vec<FileIdentity>>> {
    let mut identities = Vec::with_capacity(paths.len());
    for path in paths {
        let Some(identity) = file_identity(path)? else {
            return Ok(None);
        };
        identities.push(identity);
    }
    Ok(Some(identities))
}

#[cfg(unix)]
fn platform_file_stamp(
    _file: &File,
    metadata: &fs::Metadata,
) -> std::io::Result<Option<PlatformFileStamp>> {
    use std::os::unix::fs::MetadataExt;

    Ok(Some(PlatformFileStamp {
        device: metadata.dev(),
        inode: metadata.ino(),
        change_seconds: metadata.ctime(),
        change_nanos: metadata.ctime_nsec(),
    }))
}

#[cfg(windows)]
fn platform_file_stamp(
    file: &File,
    _metadata: &fs::Metadata,
) -> std::io::Result<Option<PlatformFileStamp>> {
    use std::mem::{size_of, MaybeUninit};
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        FileBasicInfo, FileIdInfo, GetFileInformationByHandleEx, FILE_BASIC_INFO, FILE_ID_INFO,
    };

    let handle = file.as_raw_handle();
    let mut basic = MaybeUninit::<FILE_BASIC_INFO>::uninit();
    // SAFETY: `file` owns a valid handle and both buffers have the exact Win32 structure sizes.
    let basic_ok = unsafe {
        GetFileInformationByHandleEx(
            handle,
            FileBasicInfo,
            basic.as_mut_ptr().cast(),
            size_of::<FILE_BASIC_INFO>() as u32,
        )
    };
    if basic_ok == 0 {
        return Ok(None);
    }
    let mut id = MaybeUninit::<FILE_ID_INFO>::uninit();
    // SAFETY: same handle and exact output-buffer contract as above.
    let id_ok = unsafe {
        GetFileInformationByHandleEx(
            handle,
            FileIdInfo,
            id.as_mut_ptr().cast(),
            size_of::<FILE_ID_INFO>() as u32,
        )
    };
    if id_ok == 0 {
        return Ok(None);
    }
    // SAFETY: successful calls initialized their complete output structures.
    let (basic, id) = unsafe { (basic.assume_init(), id.assume_init()) };
    Ok(Some(PlatformFileStamp {
        volume: id.VolumeSerialNumber,
        file_id: id.FileId.Identifier,
        change_time: basic.ChangeTime,
    }))
}

#[cfg(not(any(unix, windows)))]
fn platform_file_stamp(
    _file: &File,
    _metadata: &fs::Metadata,
) -> std::io::Result<Option<PlatformFileStamp>> {
    Ok(None)
}

fn validation_memo_key(
    component_dir: &Path,
    external_cache_root: Option<&Path>,
    bits: usize,
    group_size: usize,
    base_prefix: Option<&str>,
    sources: &[FileIdentity],
) -> Result<ValidationMemoKey> {
    let component_dir = fs::canonicalize(component_dir).map_err(|error| {
        Error::Msg(format!(
            "device-format sidecar: canonicalize component {}: {error}",
            component_dir.display()
        ))
    })?;
    let cache_identity = external_cache_root
        .map(Path::to_path_buf)
        .unwrap_or_else(default_external_cache_root);
    let cache_identity = fs::canonicalize(&cache_identity).unwrap_or(cache_identity);
    Ok(ValidationMemoKey {
        component_dir,
        cache_identity,
        bits,
        group_size,
        base_prefix: base_prefix.map(str::to_owned),
        sources: sources.to_vec(),
    })
}

fn validation_memo_hit(
    key: &ValidationMemoKey,
    cancel: Option<&CancelFlag>,
) -> Result<Option<ValidationMemoValue>> {
    check_cancel(cancel)?;
    let value = validation_memo()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(key)
        .cloned();
    let Some(value) = value else {
        return Ok(None);
    };
    let current_artifacts = value
        .artifacts
        .iter()
        .map(|identity| file_identity(&identity.path))
        .collect::<std::io::Result<Vec<_>>>()
        .ok()
        .and_then(|identities| identities.into_iter().collect::<Option<Vec<_>>>());
    let headers_valid = current_artifacts.as_ref().is_some_and(|current| {
        if current != &value.artifacts
            || !value.projections.iter().all(|projection| {
                let path = value.cache_dir.join(format!(
                    "{}.{}.safetensors",
                    projection.source_hex, projection.dtype_name
                ));
                validate_sidecar_header(&path, projection.payload_bytes)
            })
        {
            return false;
        }
        // Bind the header opens to the same artifact generations whose identities matched above.
        // A replacement racing between the first identity read and a pathname-based mmap changes
        // file-id/change-time and forces the serialized digest/recovery path.
        value
            .artifacts
            .iter()
            .map(|identity| file_identity(&identity.path))
            .collect::<std::io::Result<Vec<_>>>()
            .ok()
            .and_then(|identities| identities.into_iter().collect::<Option<Vec<_>>>())
            .is_some_and(|after_headers| after_headers == *current)
    });
    check_cancel(cancel)?;
    if headers_valid {
        Ok(Some(value))
    } else {
        validation_memo()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(key);
        Ok(None)
    }
}

fn remember_validation(key: &ValidationMemoKey, cache: &PackedWeightSidecars) -> Result<()> {
    let mut artifacts = Vec::with_capacity(cache.entries.len());
    for entry in cache.entries.values() {
        let Some(identity) = file_identity(&entry.path)? else {
            return Ok(());
        };
        artifacts.push(identity);
    }
    artifacts.sort_by(|a, b| a.path.cmp(&b.path));
    validation_memo()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(
            key.clone(),
            ValidationMemoValue {
                projections: cache.prepared.clone(),
                cache_dir: cache.cache_dir.clone(),
                artifacts,
            },
        );
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct SourceProjection {
    index: Option<usize>,
    out_dim: usize,
    in_dim: usize,
    source_bytes: usize,
}

fn sidecar_entry_key(base: &str, index: Option<usize>) -> String {
    match index {
        Some(index) => format!("{base}[{index}]"),
        None => base.to_owned(),
    }
}

fn validate_source_shapes(
    base: &str,
    weight: &safetensors::tensor::TensorView<'_>,
    scales: &safetensors::tensor::TensorView<'_>,
    biases: &safetensors::tensor::TensorView<'_>,
    bits: usize,
    group_size: usize,
) -> Result<Vec<SourceProjection>> {
    if weight.dtype() != SafeDtype::U32 {
        return Err(Error::Msg(format!(
            "device-format sidecar: `{base}.weight` must be U32, got {:?}",
            weight.dtype()
        )));
    }
    let (leading, out_dim, weight_cols) = match weight.shape() {
        [out_dim, weight_cols] => (None, *out_dim, *weight_cols),
        [leading, out_dim, weight_cols] => (Some(*leading), *out_dim, *weight_cols),
        shape => {
            return Err(Error::Msg(format!(
                "device-format sidecar: `{base}.weight` must be rank 2 or 3, got {shape:?}"
            )))
        }
    };
    let (scale_leading, scale_rows, scale_cols) = match scales.shape() {
        [rows, cols] => (None, *rows, *cols),
        [leading, rows, cols] => (Some(*leading), *rows, *cols),
        shape => {
            return Err(Error::Msg(format!(
                "device-format sidecar: `{base}.scales` must be rank 2 or 3, got {shape:?}"
            )))
        }
    };
    if biases.shape() != scales.shape() || leading != scale_leading || scale_rows != out_dim {
        return Err(Error::Msg(format!(
            "device-format sidecar: invalid `{base}` triple shapes: weight {:?}, scales {:?}, \
             biases {:?}",
            weight.shape(),
            scales.shape(),
            biases.shape()
        )));
    }
    let in_dim = scale_cols.checked_mul(group_size).ok_or_else(|| {
        Error::Msg(format!(
            "device-format sidecar: `{base}` input dimension overflow"
        ))
    })?;
    let codes_per_word = 32 / bits;
    if weight_cols.checked_mul(codes_per_word) != Some(in_dim) {
        return Err(Error::Msg(format!(
            "device-format sidecar: `{base}` is not an MLX q{bits} group-{group_size} triple: \
             weight {:?}, scales {:?}",
            weight.shape(),
            scales.shape()
        )));
    }
    let slices = leading.unwrap_or(1);
    let source_bytes = weight
        .data()
        .len()
        .checked_add(scales.data().len())
        .and_then(|bytes| bytes.checked_add(biases.data().len()))
        .and_then(|bytes| bytes.checked_div(slices))
        .ok_or_else(|| {
            Error::Msg("device-format sidecar: source byte count overflow".to_owned())
        })?;
    Ok((0..slices)
        .map(|index| SourceProjection {
            index: leading.map(|_| index),
            out_dim,
            in_dim,
            source_bytes,
        })
        .collect())
}

fn source_slice_digest<'a>(
    base: &str,
    bits: usize,
    group_size: usize,
    index: usize,
    views: [(&str, &safetensors::tensor::TensorView<'a>); 3],
    cancel: Option<&CancelFlag>,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(FORMAT_DOMAIN);
    hash.update((base.len() as u64).to_le_bytes());
    hash.update(base.as_bytes());
    hash.update((bits as u64).to_le_bytes());
    hash.update((group_size as u64).to_le_bytes());
    for (name, view) in views {
        let (data, shape) = source_slice(view, index)?;
        let dtype = format!("{:?}", view.dtype());
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((dtype.len() as u64).to_le_bytes());
        hash.update(dtype.as_bytes());
        hash.update((shape.len() as u64).to_le_bytes());
        for &dim in shape {
            hash.update((dim as u64).to_le_bytes());
        }
        hash.update((data.len() as u64).to_le_bytes());
        hash_source_bytes(&mut hash, data, cancel)?;
    }
    Ok(hash.finalize().into())
}

fn source_slice<'a>(
    view: &'a safetensors::tensor::TensorView<'a>,
    index: usize,
) -> Result<(&'a [u8], &'a [usize])> {
    let [leading, ..] = view.shape() else {
        return Err(Error::Msg(
            "device-format sidecar: sliced projection must have a leading dimension".to_owned(),
        ));
    };
    if index >= *leading || *leading == 0 || !view.data().len().is_multiple_of(*leading) {
        return Err(Error::Msg(format!(
            "device-format sidecar: invalid slice {index} for shape {:?}",
            view.shape()
        )));
    }
    let bytes = view.data().len() / *leading;
    Ok((
        &view.data()[index * bytes..(index + 1) * bytes],
        &view.shape()[1..],
    ))
}

fn source_digest<'a>(
    base: &str,
    bits: usize,
    group_size: usize,
    views: [(&str, &safetensors::tensor::TensorView<'a>); 3],
    cancel: Option<&CancelFlag>,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(FORMAT_DOMAIN);
    hash.update((base.len() as u64).to_le_bytes());
    hash.update(base.as_bytes());
    hash.update((bits as u64).to_le_bytes());
    hash.update((group_size as u64).to_le_bytes());
    for (name, view) in views {
        let dtype = format!("{:?}", view.dtype());
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((dtype.len() as u64).to_le_bytes());
        hash.update(dtype.as_bytes());
        hash.update((view.shape().len() as u64).to_le_bytes());
        for &dim in view.shape() {
            hash.update((dim as u64).to_le_bytes());
        }
        hash.update((view.data().len() as u64).to_le_bytes());
        hash_source_bytes(&mut hash, view.data(), cancel)?;
    }
    Ok(hash.finalize().into())
}

fn check_cancel(cancel: Option<&CancelFlag>) -> Result<()> {
    if cancel.is_some_and(CancelFlag::is_cancelled) {
        Err(Error::Msg(
            "device-format sidecar preparation cancelled".to_owned(),
        ))
    } else {
        Ok(())
    }
}

fn hash_source_bytes(hash: &mut Sha256, bytes: &[u8], cancel: Option<&CancelFlag>) -> Result<()> {
    for chunk in bytes.chunks(HASH_CANCEL_CHUNK_BYTES) {
        check_cancel(cancel)?;
        hash.update(chunk);
        #[cfg(any(test, feature = "testkit"))]
        source_cancel_test_hook(cancel);
        check_cancel(cancel)?;
    }
    Ok(())
}

#[cfg(any(test, feature = "testkit"))]
std::thread_local! {
    static CANCEL_SOURCE_AFTER_CHUNKS: std::cell::Cell<Option<usize>> = const {
        std::cell::Cell::new(None)
    };
}

/// Arm a deterministic same-thread cancellation point in the bounded source-hash loop.
///
/// Testkit-only: provider seam tests use this to prove their exact request route passes the live
/// cancellation flag into sidecar preparation rather than relying on scheduler timing.
#[cfg(any(test, feature = "testkit"))]
pub fn cancel_source_hash_after_chunks(chunks: usize) {
    assert!(
        chunks > 0,
        "source-hash cancellation needs a positive chunk count"
    );
    CANCEL_SOURCE_AFTER_CHUNKS.with(|remaining| remaining.set(Some(chunks)));
}

#[cfg(any(test, feature = "testkit"))]
fn source_cancel_test_hook(cancel: Option<&CancelFlag>) {
    if let Some(cancel) = cancel {
        CANCEL_SOURCE_AFTER_CHUNKS.with(|remaining| match remaining.get() {
            Some(1) => {
                remaining.set(None);
                cancel.cancel();
            }
            Some(chunks) => remaining.set(Some(chunks - 1)),
            None => {}
        });
    }
}

fn payload_len(dtype: GgmlDType, out_dim: usize, in_dim: usize) -> Result<usize> {
    let elements = out_dim
        .checked_mul(in_dim)
        .ok_or_else(|| Error::Msg("device-format sidecar: element-count overflow".to_string()))?;
    if !elements.is_multiple_of(32) {
        return Err(Error::Msg(format!(
            "device-format sidecar: quantized shape [{out_dim}, {in_dim}] is not block-32 aligned"
        )));
    }
    let block_bytes = match dtype {
        GgmlDType::Q4_1 => 20usize,
        GgmlDType::Q8_0 => 34usize,
        other => {
            return Err(Error::Msg(format!(
                "device-format sidecar: unsupported target dtype {other:?}"
            )))
        }
    };
    elements
        .checked_div(32)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(|| Error::Msg("device-format sidecar: payload-size overflow".to_string()))
}

#[allow(clippy::too_many_arguments)]
fn build_sidecar(
    source: &MmapedSafetensors,
    final_path: &Path,
    source_hex: &str,
    base: &str,
    slice: Option<usize>,
    bits: usize,
    group_size: usize,
    out_dim: usize,
    in_dim: usize,
    dtype: GgmlDType,
    device: &Device,
) -> Result<()> {
    let cpu = Device::Cpu;
    let load = |suffix: &str| -> Result<Tensor> {
        let view = source.get(&format!("{base}.{suffix}"))?;
        match slice {
            Some(index) => {
                let (data, shape) = source_slice(&view, index)?;
                Tensor::from_raw_buffer(data, safe_dtype(view.dtype())?, shape, &cpu)
            }
            None => {
                Tensor::from_raw_buffer(view.data(), safe_dtype(view.dtype())?, view.shape(), &cpu)
            }
        }
    };
    let wq = load("weight")?;
    let scales = load("scales")?.to_dtype(DType::F32)?;
    let biases = load("biases")?.to_dtype(DType::F32)?;
    // This is intentionally the old production conversion, on the old target device, once. Q8's
    // CUDA quantizer can therefore produce exactly the bytes the pre-change path consumed.
    let qtensor = repack_packed_weight(&wq, &scales, &biases, group_size, device)?;
    if qtensor.dtype() != dtype || qtensor.shape().dims() != [out_dim, in_dim] {
        return Err(Error::Msg(format!(
            "device-format sidecar: `{base}` conversion returned {:?} {:?}, expected {:?} \
             [{out_dim}, {in_dim}]",
            qtensor.dtype(),
            qtensor.shape(),
            dtype
        )));
    }
    let payload = qtensor.data()?;
    let payload_hash: [u8; 32] = Sha256::digest(payload.as_ref()).into();
    let seq = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp_path =
        final_path.with_extension(format!("tmp-{}-{seq}.safetensors", std::process::id()));

    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), "candle-device-format-v1".to_string());
    metadata.insert("source_sha256".to_string(), source_hex.to_string());
    metadata.insert("source_base".to_string(), sidecar_entry_key(base, slice));
    metadata.insert("source_bits".to_string(), bits.to_string());
    metadata.insert("source_group_size".to_string(), group_size.to_string());
    metadata.insert("target_dtype".to_string(), format!("{dtype:?}"));
    metadata.insert("out_dim".to_string(), out_dim.to_string());
    metadata.insert("in_dim".to_string(), in_dim.to_string());
    let views = [
        (
            PAYLOAD_KEY,
            BytesView {
                data: payload.as_ref(),
                shape: [payload.len()],
            },
        ),
        (
            PAYLOAD_HASH_KEY,
            BytesView {
                data: &payload_hash,
                shape: [payload_hash.len()],
            },
        ),
    ];
    safetensors::tensor::serialize_to_file(views, Some(metadata), &temp_path).map_err(|e| {
        Error::Msg(format!(
            "device-format sidecar: write {}: {e}",
            temp_path.display()
        ))
    })?;
    fs::OpenOptions::new()
        .write(true)
        .open(&temp_path)?
        .sync_all()?;

    if final_path.exists() {
        if validate_sidecar(final_path, payload.len(), None)? {
            let _ = fs::remove_file(&temp_path);
            return Ok(());
        }
        if let Err(e) = fs::remove_file(final_path) {
            // Another process may have removed the same corrupt cache entry after our validation.
            // Both writers then race only at atomic publish, which the rename branch below handles.
            if e.kind() != std::io::ErrorKind::NotFound {
                return Err(Error::Msg(format!(
                    "device-format sidecar: replace corrupt {}: {e}",
                    final_path.display()
                )));
            }
        }
    }
    match fs::rename(&temp_path, final_path) {
        Ok(()) => Ok(()),
        Err(_rename_error) if validate_sidecar(final_path, payload.len(), None)? => {
            let _ = fs::remove_file(&temp_path);
            Ok(())
        }
        Err(rename_error) => {
            let _ = fs::remove_file(&temp_path);
            Err(Error::Msg(format!(
                "device-format sidecar: publish {}: {rename_error}",
                final_path.display()
            )))
        }
    }
}

fn safe_dtype(dtype: SafeDtype) -> Result<DType> {
    match dtype {
        SafeDtype::U8 => Ok(DType::U8),
        SafeDtype::U32 => Ok(DType::U32),
        SafeDtype::F16 => Ok(DType::F16),
        SafeDtype::BF16 => Ok(DType::BF16),
        SafeDtype::F32 => Ok(DType::F32),
        other => Err(Error::Msg(format!(
            "device-format sidecar: unsupported source dtype {other:?}"
        ))),
    }
}

fn validate_sidecar(
    path: &Path,
    payload_bytes: usize,
    cancel: Option<&CancelFlag>,
) -> Result<bool> {
    let mut payload_bytes_hashed = 0;
    validate_sidecar_counted(path, payload_bytes, cancel, &mut payload_bytes_hashed)
}

fn validate_sidecar_header(path: &Path, payload_bytes: usize) -> bool {
    if !path.is_file() {
        return false;
    }
    // SAFETY: header-only validation maps an immutable, atomically published cache artifact.
    let Ok(mapped) = (unsafe { MmapedSafetensors::new(path) }) else {
        return false;
    };
    matches!(
        mapped.get(PAYLOAD_KEY),
        Ok(view) if view.dtype() == SafeDtype::U8 && view.data().len() == payload_bytes
    ) && matches!(
        mapped.get(PAYLOAD_HASH_KEY),
        Ok(view) if view.dtype() == SafeDtype::U8 && view.data().len() == 32
    )
}

fn validate_sidecar_counted(
    path: &Path,
    payload_bytes: usize,
    cancel: Option<&CancelFlag>,
    payload_bytes_hashed: &mut u64,
) -> Result<bool> {
    check_cancel(cancel)?;
    if !path.is_file() {
        return Ok(false);
    }
    // SAFETY: validation maps a read-only cache artifact and performs no concurrent mutation.
    let mapped = match unsafe { MmapedSafetensors::new(path) } {
        Ok(mapped) => mapped,
        Err(_) => return Ok(false),
    };
    let payload = match mapped.get(PAYLOAD_KEY) {
        Ok(view) if view.dtype() == SafeDtype::U8 && view.data().len() == payload_bytes => view,
        _ => return Ok(false),
    };
    let stored_hash = match mapped.get(PAYLOAD_HASH_KEY) {
        Ok(view) if view.dtype() == SafeDtype::U8 && view.data().len() == 32 => view,
        _ => return Ok(false),
    };
    *payload_bytes_hashed = payload_bytes_hashed.saturating_add(payload.data().len() as u64);
    let actual_hash = validation_payload_digest(payload.data(), cancel)?;
    check_cancel(cancel)?;
    Ok(stored_hash.data() == actual_hash)
}

fn validation_payload_digest(bytes: &[u8], cancel: Option<&CancelFlag>) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    for chunk in bytes.chunks(HASH_CANCEL_CHUNK_BYTES) {
        check_cancel(cancel)?;
        hash.update(chunk);
        #[cfg(test)]
        validation_cancel_test_hook(cancel);
        check_cancel(cancel)?;
    }
    Ok(hash.finalize().into())
}

#[cfg(test)]
std::thread_local! {
    static CANCEL_VALIDATION_AFTER_CHUNKS: std::cell::Cell<Option<usize>> = const {
        std::cell::Cell::new(None)
    };
}

#[cfg(test)]
fn validation_cancel_test_hook(cancel: Option<&CancelFlag>) {
    if let Some(cancel) = cancel {
        CANCEL_VALIDATION_AFTER_CHUNKS.with(|remaining| match remaining.get() {
            Some(1) => {
                remaining.set(None);
                cancel.cancel();
            }
            Some(chunks) => remaining.set(Some(chunks - 1)),
            None => {}
        });
    }
}

#[derive(Clone, Copy)]
struct BytesView<'a> {
    data: &'a [u8],
    shape: [usize; 1],
}

impl View for BytesView<'_> {
    fn dtype(&self) -> SafeDtype {
        SafeDtype::U8
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{pack_mlx_affine, repack_packed_weight};
    use candle_core::safetensors;
    use candle_core::{IndexOp, Tensor};
    use std::io::{Seek, SeekFrom, Write};

    /// A component directory that removes itself — including out of a panicking test, which the
    /// hand-written `Drop` this replaces could do, but only alongside a `$TMPDIR` path a second
    /// process could be handed again. The guard is the cleanup: one path, not two.
    struct TestDir(tempfile::TempDir);

    impl TestDir {
        fn new(label: &str) -> Self {
            Self(
                tempfile::Builder::new()
                    .prefix(&format!("candle-device-sidecar-{label}-"))
                    .tempdir()
                    .unwrap(),
            )
        }

        fn path(&self) -> &Path {
            self.0.path()
        }
    }

    fn write_source(dir: &Path, bits: usize, delta: f32) -> Result<PathBuf> {
        let values: Vec<f32> = (0..2 * 64)
            .map(|i| ((i * 17 + i / 7) % 41) as f32 / 16.0 + delta)
            .collect();
        let dense = Tensor::from_vec(values, (2, 64), &Device::Cpu)?;
        let (wq, scales, biases) = pack_mlx_affine(&dense, bits, 64)?;
        let tensors = HashMap::from([
            ("layers.0.proj.weight".to_string(), wq),
            ("layers.0.proj.scales".to_string(), scales),
            ("layers.0.proj.biases".to_string(), biases),
        ]);
        let path = dir.join("model.safetensors");
        safetensors::save(&tensors, &path)?;
        Ok(path)
    }

    fn write_multi_source(dir: &Path, bits: usize) -> Result<PathBuf> {
        let mut tensors = HashMap::new();
        for layer in 0..2 {
            let values: Vec<f32> = (0..2 * 64)
                .map(|i| ((i * 17 + i / 7 + layer * 11) % 41) as f32 / 16.0)
                .collect();
            let dense = Tensor::from_vec(values, (2, 64), &Device::Cpu)?;
            let (weight, scales, biases) = pack_mlx_affine(&dense, bits, 64)?;
            tensors.insert(format!("layers.{layer}.proj.weight"), weight);
            tensors.insert(format!("layers.{layer}.proj.scales"), scales);
            tensors.insert(format!("layers.{layer}.proj.biases"), biases);
        }
        let path = dir.join("model.safetensors");
        safetensors::save(&tensors, &path)?;
        Ok(path)
    }

    fn open(path: &Path) -> Result<MmapedSafetensors> {
        // SAFETY: immutable test fixture for the lifetime of the mapping.
        unsafe { MmapedSafetensors::new(path) }
    }

    fn open_and_prepare_path(
        path: &Path,
        component_dir: &Path,
        packed: PackedConfig,
    ) -> Result<(MmapedSafetensors, PackedWeightSidecars)> {
        PackedWeightSidecars::open_and_prepare(
            &[path.to_path_buf()],
            component_dir,
            packed,
            &Device::Cpu,
        )
    }

    fn restore_modified(path: &Path, modified: std::time::SystemTime) -> std::io::Result<()> {
        File::options()
            .write(true)
            .open(path)?
            .set_times(fs::FileTimes::new().set_modified(modified))
    }

    #[cfg(unix)]
    fn replace_file(replacement: &Path, target: &Path) -> std::io::Result<()> {
        fs::rename(replacement, target)
    }

    #[cfg(windows)]
    fn replace_file(replacement: &Path, target: &Path) -> std::io::Result<()> {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::{ReplaceFileW, REPLACEFILE_WRITE_THROUGH};

        let target: Vec<u16> = target.as_os_str().encode_wide().chain(Some(0)).collect();
        let replacement: Vec<u16> = replacement
            .as_os_str()
            .encode_wide()
            .chain(Some(0))
            .collect();
        // SAFETY: both paths are valid nul-terminated UTF-16 buffers for the duration of the call.
        let replaced = unsafe {
            ReplaceFileW(
                target.as_ptr(),
                replacement.as_ptr(),
                std::ptr::null(),
                REPLACEFILE_WRITE_THROUGH,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if replaced == 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    #[test]
    fn pre_cancelled_prepare_stops_before_creating_the_cache() -> Result<()> {
        let dir = TestDir::new("cancelled");
        let source_path = write_source(dir.path(), 4, 0.0)?;
        let source = open(&source_path)?;
        let cancel = CancelFlag::new();
        cancel.cancel();
        let error = PackedWeightSidecars::prepare_cancelable(
            &source,
            dir.path(),
            PackedConfig {
                bits: 4,
                group_size: 64,
            },
            &Device::Cpu,
            &cancel,
        )
        .unwrap_err();
        assert!(error.to_string().contains("cancelled"));
        assert!(!dir.path().join(CACHE_DIR).exists());
        Ok(())
    }

    #[test]
    fn cancellation_interrupts_waiting_for_the_cross_process_prepare_lock() -> Result<()> {
        let dir = TestDir::new("cancelled-lock");
        let source_path = write_source(dir.path(), 4, 0.0)?;
        let source = open(&source_path)?;
        let cache_dir = dir.path().join(CACHE_DIR);
        fs::create_dir_all(&cache_dir)?;
        let held = fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(cache_dir.join(PREPARE_LOCK))?;
        FileExt::lock_exclusive(&held)?;
        let cancel = CancelFlag::new();
        let trigger = cancel.clone();
        let canceller = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(30));
            trigger.cancel();
        });
        let error = PackedWeightSidecars::prepare_cancelable(
            &source,
            dir.path(),
            PackedConfig {
                bits: 4,
                group_size: 64,
            },
            &Device::Cpu,
            &cancel,
        )
        .unwrap_err();
        canceller.join().unwrap();
        FileExt::unlock(&held)?;
        assert!(error.to_string().contains("cancelled"));
        Ok(())
    }

    #[test]
    fn multi_entry_warm_caches_cancel_inside_payload_validation() -> Result<()> {
        let packed = PackedConfig {
            bits: 4,
            group_size: 64,
        };
        for external_warm_cache in [false, true] {
            let component = TestDir::new(if external_warm_cache {
                "cancel-warm-external"
            } else {
                "cancel-warm-adjacent"
            });
            let external = TestDir::new("cancel-warm-root");
            let source_path = write_multi_source(component.path(), 4)?;
            let source = open(&source_path)?;
            if external_warm_cache {
                // A regular file makes the adjacent cache unavailable on every platform, including
                // root-capable CI, so the complete warm set is definitely external.
                fs::write(
                    component.path().join(CACHE_DIR),
                    b"immutable snapshot entry",
                )?;
            }
            let first = PackedWeightSidecars::prepare_impl(
                &source,
                component.path(),
                packed,
                &Device::Cpu,
                None,
                None,
                Some(external.path()),
                None,
            )?;
            assert_eq!(first.created_count(), 2, "fixture must be non-vacuous");
            assert!(first.contains("layers.0.proj"));
            assert!(first.contains("layers.1.proj"));
            assert_eq!(
                first.cache_dir().starts_with(external.path()),
                external_warm_cache,
                "test must exercise the intended warm-cache location"
            );

            let cancel = CancelFlag::new();
            validation_memo()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clear();
            // Deterministic mutation hook: cancel after hashing the first payload chunk. This proves
            // the error originates inside artifact validation, not before source hashing or between
            // the two entries, without relying on scheduler timing.
            CANCEL_VALIDATION_AFTER_CHUNKS.with(|remaining| remaining.set(Some(1)));
            let error = PackedWeightSidecars::prepare_impl(
                &source,
                component.path(),
                packed,
                &Device::Cpu,
                Some(&cancel),
                None,
                Some(external.path()),
                None,
            )
            .unwrap_err();
            CANCEL_VALIDATION_AFTER_CHUNKS.with(|remaining| remaining.set(None));

            assert!(cancel.is_cancelled());
            assert_eq!(
                error.to_string(),
                "device-format sidecar preparation cancelled"
            );
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn complete_read_only_cache_reuses_without_creating_the_prepare_lock() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let dir = TestDir::new("warm-read-only");
        let source_path = write_source(dir.path(), 4, 0.0)?;
        let source = open(&source_path)?;
        let packed = PackedConfig {
            bits: 4,
            group_size: 64,
        };
        let first = PackedWeightSidecars::prepare(&source, dir.path(), packed, &Device::Cpu)?;
        let cache_dir = first.cache_dir().to_path_buf();
        let lock = cache_dir.join(PREPARE_LOCK);
        fs::remove_file(&lock)?;

        fs::set_permissions(&cache_dir, fs::Permissions::from_mode(0o555))?;
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555))?;
        let result = PackedWeightSidecars::prepare(&source, dir.path(), packed, &Device::Cpu);
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755))?;
        fs::set_permissions(&cache_dir, fs::Permissions::from_mode(0o755))?;

        let reused = result?;
        assert_eq!(reused.created_count(), 0);
        assert_eq!(reused.reused_count(), 1);
        assert!(
            !lock.exists(),
            "warm reuse must not recreate the prepare lock"
        );
        reused.load("layers.0.proj", &Device::Cpu)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn cold_read_only_component_uses_external_file_backed_cache() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let component = TestDir::new("cold-read-only");
        let external = TestDir::new("external-root");
        let source_path = write_source(component.path(), 4, 0.0)?;
        let source = open(&source_path)?;
        // A chmod alone is still writable to root. Making the adjacent cache pathname a regular file
        // forces the same non-writable-cache branch under privileged CI while the chmod pins the real
        // deployment contract for ordinary users.
        fs::write(
            component.path().join(CACHE_DIR),
            b"immutable snapshot entry",
        )?;
        fs::set_permissions(component.path(), fs::Permissions::from_mode(0o555))?;
        let result = PackedWeightSidecars::prepare_with_external_cache_root(
            &source,
            component.path(),
            PackedConfig {
                bits: 4,
                group_size: 64,
            },
            &Device::Cpu,
            external.path(),
        );
        fs::set_permissions(component.path(), fs::Permissions::from_mode(0o755))?;

        let cache = result?;
        assert_eq!(cache.created_count(), 1);
        assert!(cache.cache_dir().starts_with(external.path()));
        assert!(component.path().join(CACHE_DIR).is_file());
        cache.load("layers.0.proj", &Device::Cpu)?;
        Ok(())
    }

    #[test]
    fn unchanged_external_cache_warm_reopen_hashes_zero_bytes() -> Result<()> {
        let component = TestDir::new("external-warm-component");
        let external = TestDir::new("external-warm-root");
        let source_path = write_source(component.path(), 4, 0.0)?;
        fs::write(
            component.path().join(CACHE_DIR),
            b"immutable snapshot entry",
        )?;
        let packed = PackedConfig {
            bits: 4,
            group_size: 64,
        };
        let (_, first) = PackedWeightSidecars::open_and_prepare_with_external_cache_root(
            std::slice::from_ref(&source_path),
            component.path(),
            packed,
            &Device::Cpu,
            external.path(),
        )?;
        assert!(first.cache_dir().starts_with(external.path()));
        let (_, warm) = PackedWeightSidecars::open_and_prepare_with_external_cache_root(
            std::slice::from_ref(&source_path),
            component.path(),
            packed,
            &Device::Cpu,
            external.path(),
        )?;
        assert_eq!(warm.source_bytes_hashed(), 0);
        assert_eq!(warm.payload_bytes_hashed(), 0);
        assert_eq!(warm.reused_count(), 1);
        Ok(())
    }

    fn write_rank3_source(dir: &Path, bits: usize) -> Result<PathBuf> {
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        let mut biases = Vec::new();
        for expert in 0..3 {
            let values: Vec<f32> = (0..2 * 64)
                .map(|i| ((i * 13 + expert * 7) % 37) as f32 / 11.0)
                .collect();
            let dense = Tensor::from_vec(values, (2, 64), &Device::Cpu)?;
            let (weight, scale, bias) = pack_mlx_affine(&dense, bits, 64)?;
            weights.push(weight);
            scales.push(scale);
            biases.push(bias);
        }
        let tensors = HashMap::from([
            (
                "layers.0.experts.proj.weight".to_string(),
                Tensor::stack(&weights, 0)?,
            ),
            (
                "layers.0.experts.proj.scales".to_string(),
                Tensor::stack(&scales, 0)?,
            ),
            (
                "layers.0.experts.proj.biases".to_string(),
                Tensor::stack(&biases, 0)?,
            ),
        ]);
        let path = dir.join("experts.safetensors");
        safetensors::save(&tensors, &path)?;
        Ok(path)
    }

    fn assert_q4_or_q8(bits: usize) -> Result<()> {
        let dir = TestDir::new(if bits == 4 { "q4" } else { "q8" });
        let source_path = write_source(dir.path(), bits, 0.0)?;
        let packed = PackedConfig {
            bits: bits as i32,
            group_size: 64,
        };

        let (source, cache) = open_and_prepare_path(&source_path, dir.path(), packed)?;
        assert_eq!(cache.created_count(), 1);
        assert_eq!(cache.reused_count(), 0);
        assert!(cache.source_bytes_hashed() > 0);
        assert!(cache.payload_bytes_hashed() > 0);
        assert!(cache.contains("layers.0.proj"));
        let sidecar_path = cache.path_for("layers.0.proj").unwrap();
        assert!(sidecar_path.is_file());
        assert!(sidecar_path.starts_with(dir.path().join(CACHE_DIR)));

        let wq = source.load("layers.0.proj.weight", &Device::Cpu)?;
        let scales = source.load("layers.0.proj.scales", &Device::Cpu)?;
        let biases = source.load("layers.0.proj.biases", &Device::Cpu)?;
        let old = repack_packed_weight(&wq, &scales, &biases, 64, &Device::Cpu)?
            .data()?
            .into_owned();
        let new = cache
            .load("layers.0.proj", &Device::Cpu)?
            .data()?
            .into_owned();
        assert_eq!(
            new, old,
            "mapped device-format bytes must match the old path"
        );

        // A second preparation reuses the atomically-published artifact. Repeated materializations
        // below accept no source tensors and therefore cannot regress to per-window conversion.
        let (_, reused) = open_and_prepare_path(&source_path, dir.path(), packed)?;
        assert_eq!(reused.created_count(), 0);
        assert_eq!(reused.reused_count(), 1);
        assert_eq!(reused.source_bytes_hashed(), 0);
        assert_eq!(reused.payload_bytes_hashed(), 0);
        for _ in 0..3 {
            assert_eq!(
                reused.load("layers.0.proj", &Device::Cpu)?.data()?.as_ref(),
                old
            );
        }
        Ok(())
    }

    #[test]
    fn q4_sidecar_is_byte_exact_and_reused_without_source_conversion() -> Result<()> {
        assert_q4_or_q8(4)
    }

    #[test]
    fn q8_sidecar_is_byte_exact_and_reused_without_source_conversion() -> Result<()> {
        assert_q4_or_q8(8)
    }

    #[test]
    fn prefix_preparation_excludes_resident_component_weights() -> Result<()> {
        let dir = TestDir::new("prefix");
        let values: Vec<f32> = (0..2 * 64).map(|i| (i % 29) as f32 / 9.0).collect();
        let dense = Tensor::from_vec(values, (2, 64), &Device::Cpu)?;
        let (layer_w, layer_s, layer_b) = pack_mlx_affine(&dense, 4, 64)?;
        let (resident_w, resident_s, resident_b) = pack_mlx_affine(&dense, 4, 64)?;
        let path = dir.path().join("model.safetensors");
        safetensors::save(
            &HashMap::from([
                ("layers.0.proj.weight".to_owned(), layer_w),
                ("layers.0.proj.scales".to_owned(), layer_s),
                ("layers.0.proj.biases".to_owned(), layer_b),
                ("noise_refiner.0.proj.weight".to_owned(), resident_w),
                ("noise_refiner.0.proj.scales".to_owned(), resident_s),
                ("noise_refiner.0.proj.biases".to_owned(), resident_b),
            ]),
            &path,
        )?;
        let source = open(&path)?;
        let cache = PackedWeightSidecars::prepare_prefix_cancelable(
            &source,
            dir.path(),
            PackedConfig {
                bits: 4,
                group_size: 64,
            },
            &Device::Cpu,
            &CancelFlag::default(),
            "layers.",
        )?;
        assert_eq!(cache.created_count(), 1);
        assert!(cache.contains("layers.0.proj"));
        assert!(!cache.contains("noise_refiner.0.proj"));
        Ok(())
    }

    #[test]
    fn rank3_moe_slices_are_independently_addressed_and_byte_exact() -> Result<()> {
        for bits in [4, 8] {
            let dir = TestDir::new(if bits == 4 { "moe-q4" } else { "moe-q8" });
            let path = write_rank3_source(dir.path(), bits)?;
            let source = open(&path)?;
            let cache = PackedWeightSidecars::prepare(
                &source,
                dir.path(),
                PackedConfig {
                    bits: bits as i32,
                    group_size: 64,
                },
                &Device::Cpu,
            )?;
            assert_eq!(cache.created_count(), 3);
            let weight = source.load("layers.0.experts.proj.weight", &Device::Cpu)?;
            let scales = source.load("layers.0.experts.proj.scales", &Device::Cpu)?;
            let biases = source.load("layers.0.experts.proj.biases", &Device::Cpu)?;
            for expert in 0..3 {
                let expected = repack_packed_weight(
                    &weight.i(expert)?,
                    &scales.i(expert)?,
                    &biases.i(expert)?,
                    64,
                    &Device::Cpu,
                )?
                .data()?
                .into_owned();
                let got = cache
                    .load_slice("layers.0.experts.proj", expert, &Device::Cpu)?
                    .data()?
                    .into_owned();
                assert_eq!(got, expected, "q{bits} expert {expert}");
            }
        }
        Ok(())
    }

    #[test]
    fn source_byte_change_selects_a_different_content_address() -> Result<()> {
        let a = TestDir::new("hash-a");
        let b = TestDir::new("hash-b");
        let a_path = write_source(a.path(), 4, 0.0)?;
        let b_path = write_source(b.path(), 4, 0.125)?;
        let packed = PackedConfig {
            bits: 4,
            group_size: 64,
        };
        let a_source = open(&a_path)?;
        let b_source = open(&b_path)?;
        let a_cache = PackedWeightSidecars::prepare(&a_source, a.path(), packed, &Device::Cpu)?;
        let b_cache = PackedWeightSidecars::prepare(&b_source, b.path(), packed, &Device::Cpu)?;
        assert_ne!(
            a_cache.path_for("layers.0.proj").unwrap().file_name(),
            b_cache.path_for("layers.0.proj").unwrap().file_name(),
            "invalidation must follow source bytes, not a timestamp"
        );
        Ok(())
    }

    #[test]
    fn legacy_mmap_with_a_different_component_dir_cannot_seed_a_false_memo_hit() -> Result<()> {
        let a = TestDir::new("mismatched-source-a");
        let b = TestDir::new("mismatched-source-b");
        let a_path = write_source(a.path(), 4, 0.0)?;
        let b_path = write_source(b.path(), 4, 0.5)?;
        let packed = PackedConfig {
            bits: 4,
            group_size: 64,
        };
        let a_source = open(&a_path)?;
        let mismatched = PackedWeightSidecars::prepare(&a_source, b.path(), packed, &Device::Cpu)?;
        let mismatched_path = mismatched.path_for("layers.0.proj").unwrap().to_path_buf();

        let (_, normal_b) = open_and_prepare_path(&b_path, b.path(), packed)?;
        assert!(normal_b.source_bytes_hashed() > 0);
        assert_ne!(
            normal_b.path_for("layers.0.proj").unwrap(),
            mismatched_path,
            "a bare mmap must never be memoized under an unrelated component directory"
        );
        Ok(())
    }

    #[test]
    fn replacement_between_source_open_and_identity_check_is_rejected() -> Result<()> {
        let dir = TestDir::new("replace-during-open");
        let replacement_dir = TestDir::new("replace-during-open-new");
        let source_path = write_source(dir.path(), 4, 0.0)?;
        let replacement_source = write_source(replacement_dir.path(), 4, 0.75)?;
        let replacement = dir.path().join("replacement.safetensors");
        fs::copy(replacement_source, &replacement)?;
        let target = source_path.clone();
        AFTER_SOURCE_OPEN_HOOK.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move || {
                replace_file(&replacement, &target).unwrap();
            }));
        });
        let packed = PackedConfig {
            bits: 4,
            group_size: 64,
        };
        let error = match open_and_prepare_path(&source_path, dir.path(), packed) {
            Ok(_) => panic!("source replacement during mmap open must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("changed while opening"));

        let (_, reopened) = open_and_prepare_path(&source_path, dir.path(), packed)?;
        assert!(reopened.source_bytes_hashed() > 0);
        Ok(())
    }

    #[test]
    fn same_size_same_mtime_atomic_artifact_replacement_invalidates_the_memo() -> Result<()> {
        let dir = TestDir::new("artifact-replacement-identity");
        let source_path = write_source(dir.path(), 4, 0.0)?;
        let packed = PackedConfig {
            bits: 4,
            group_size: 64,
        };
        let (_, first) = open_and_prepare_path(&source_path, dir.path(), packed)?;
        let artifact = first.path_for("layers.0.proj").unwrap().to_path_buf();
        let metadata = fs::metadata(&artifact)?;
        let modified = metadata.modified()?;
        let mut bytes = fs::read(&artifact)?;
        *bytes.last_mut().unwrap() ^= 0x5a;
        let replacement = artifact.with_extension("replacement");
        fs::write(&replacement, &bytes)?;
        restore_modified(&replacement, modified)?;
        replace_file(&replacement, &artifact)?;
        let replaced = fs::metadata(&artifact)?;
        assert_eq!(replaced.len(), metadata.len());
        assert_eq!(replaced.modified()?, modified);

        let (_, rebuilt) = open_and_prepare_path(&source_path, dir.path(), packed)?;
        assert_eq!(rebuilt.created_count(), 1);
        assert!(rebuilt.source_bytes_hashed() > 0);
        assert!(rebuilt.payload_bytes_hashed() > 0);
        rebuilt.load("layers.0.proj", &Device::Cpu)?;
        Ok(())
    }

    #[test]
    fn same_size_same_mtime_in_place_artifact_corruption_invalidates_the_memo() -> Result<()> {
        let dir = TestDir::new("artifact-in-place-identity");
        let source_path = write_source(dir.path(), 8, 0.0)?;
        let packed = PackedConfig {
            bits: 8,
            group_size: 64,
        };
        let (_, first) = open_and_prepare_path(&source_path, dir.path(), packed)?;
        let artifact = first.path_for("layers.0.proj").unwrap().to_path_buf();
        let metadata = fs::metadata(&artifact)?;
        let modified = metadata.modified()?;
        let mut file = File::options().read(true).write(true).open(&artifact)?;
        file.seek(SeekFrom::End(-1))?;
        let last = fs::read(&artifact)?.last().copied().unwrap() ^ 0xa5;
        file.write_all(&[last])?;
        file.sync_all()?;
        drop(file);
        restore_modified(&artifact, modified)?;
        let corrupted = fs::metadata(&artifact)?;
        assert_eq!(corrupted.len(), metadata.len());
        assert_eq!(corrupted.modified()?, modified);

        let (_, rebuilt) = open_and_prepare_path(&source_path, dir.path(), packed)?;
        assert_eq!(rebuilt.created_count(), 1);
        assert!(rebuilt.source_bytes_hashed() > 0);
        assert!(rebuilt.payload_bytes_hashed() > 0);
        rebuilt.load("layers.0.proj", &Device::Cpu)?;
        Ok(())
    }

    #[test]
    fn same_source_path_generation_change_invalidates_the_process_memo() -> Result<()> {
        let dir = TestDir::new("same-path-source-change");
        let source_path = write_source(dir.path(), 4, 0.0)?;
        let packed = PackedConfig {
            bits: 4,
            group_size: 64,
        };
        let (_, first) = open_and_prepare_path(&source_path, dir.path(), packed)?;
        let first_path = first.path_for("layers.0.proj").unwrap().to_path_buf();
        std::thread::sleep(std::time::Duration::from_millis(20));
        write_source(dir.path(), 4, 0.25)?;
        let (_, changed) = open_and_prepare_path(&source_path, dir.path(), packed)?;
        assert!(changed.source_bytes_hashed() > 0);
        assert_ne!(changed.path_for("layers.0.proj").unwrap(), first_path);
        Ok(())
    }

    #[test]
    fn artifact_generation_change_forces_full_validation_before_reuse() -> Result<()> {
        let dir = TestDir::new("artifact-metadata-change");
        let source_path = write_source(dir.path(), 4, 0.0)?;
        let packed = PackedConfig {
            bits: 4,
            group_size: 64,
        };
        let (_, first) = open_and_prepare_path(&source_path, dir.path(), packed)?;
        let artifact = first.path_for("layers.0.proj").unwrap().to_path_buf();
        let bytes = fs::read(&artifact)?;
        std::thread::sleep(std::time::Duration::from_millis(20));
        fs::write(&artifact, bytes)?;
        let (_, reopened) = open_and_prepare_path(&source_path, dir.path(), packed)?;
        assert_eq!(reopened.created_count(), 0);
        assert_eq!(reopened.reused_count(), 1);
        assert!(reopened.source_bytes_hashed() > 0);
        assert!(reopened.payload_bytes_hashed() > 0);
        Ok(())
    }

    #[test]
    fn corrupt_sidecar_is_rebuilt_before_use() -> Result<()> {
        let dir = TestDir::new("corrupt");
        let source_path = write_source(dir.path(), 4, 0.0)?;
        let source = open(&source_path)?;
        let packed = PackedConfig {
            bits: 4,
            group_size: 64,
        };
        let first = PackedWeightSidecars::prepare(&source, dir.path(), packed, &Device::Cpu)?;
        let path = first.path_for("layers.0.proj").unwrap().to_path_buf();
        fs::write(&path, b"truncated generated cache")?;
        let rebuilt = PackedWeightSidecars::prepare(&source, dir.path(), packed, &Device::Cpu)?;
        assert_eq!(rebuilt.created_count(), 1);
        assert!(rebuilt.load("layers.0.proj", &Device::Cpu).is_ok());
        Ok(())
    }

    #[test]
    fn concurrent_corrupt_recovery_serializes_and_keeps_a_valid_artifact() -> Result<()> {
        let dir = TestDir::new("concurrent-corrupt");
        let source_path = write_source(dir.path(), 8, 0.0)?;
        let source = open(&source_path)?;
        let packed = PackedConfig {
            bits: 8,
            group_size: 64,
        };
        let first = PackedWeightSidecars::prepare(&source, dir.path(), packed, &Device::Cpu)?;
        let path = first.path_for("layers.0.proj").unwrap().to_path_buf();
        fs::write(&path, b"same corrupt entry observed by two preparers")?;

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let barrier = std::sync::Arc::clone(&barrier);
            let source_path = source_path.clone();
            let component_dir = dir.path().to_path_buf();
            threads.push(std::thread::spawn(move || -> Result<(usize, usize)> {
                let source = open(&source_path)?;
                barrier.wait();
                let cache =
                    PackedWeightSidecars::prepare(&source, &component_dir, packed, &Device::Cpu)?;
                cache.load("layers.0.proj", &Device::Cpu)?;
                Ok((cache.created_count(), cache.reused_count()))
            }));
        }
        let results: Vec<(usize, usize)> = threads
            .into_iter()
            .map(|thread| thread.join().expect("preparer thread"))
            .collect::<Result<_>>()?;
        assert_eq!(
            results.iter().map(|(created, _)| created).sum::<usize>(),
            1,
            "exactly one preparer must rebuild the corrupt artifact"
        );
        assert_eq!(
            results.iter().map(|(_, reused)| reused).sum::<usize>(),
            1,
            "the serialized follower must reuse the valid replacement"
        );
        assert!(validate_sidecar(
            &path,
            payload_len(GgmlDType::Q8_0, 2, 64)?,
            None,
        )?);
        Ok(())
    }
}
