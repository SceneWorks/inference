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
//! First preparation validates and publishes content-addressed runtime weights and a small ready
//! index. Subsequent opens trust that index, including across processes: they do not hash the
//! checkpoint, inspect file generations, or validate cached payloads. A window maps the selected
//! runtime payload and uploads its GGML bytes directly. An actual read/decode failure takes the
//! serialized repair path; normal cache hits never acquire the preparation lock.
//! First creation remains projection-at-a-time to bound conversion memory. The external root can
//! be supplied through `SCENEWORKS_CANDLE_DEVICE_CACHE_DIR`; otherwise the platform's per-user cache
//! directory is used, with an owner-only temporary fallback when unavailable.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::gen_core::snapshot_inventory::hex_lower as hex;
use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Error, Result, Tensor};
use fs2::FileExt;
use safetensors::tensor::{Dtype as SafeDtype, View};
use sha2::{Digest, Sha256};

use super::{repack_packed_weight, PackedConfig};
use crate::gen_core::CancelFlag;

// Shared with gen-core's tensor-free weight walkers, which must skip this directory: it holds derived
// copies of the sibling source file, and pricing both double-counted the component (SC-22667).
const CACHE_DIR: &str = crate::gen_core::weightsmeta::CANDLE_DEVICE_FORMAT_CACHE_DIR;
const PREPARE_LOCK: &str = ".prepare.lock";
const PAYLOAD_KEY: &str = "weight";
const PAYLOAD_HASH_KEY: &str = "payload_sha256";
const FORMAT_DOMAIN: &[u8] = b"sceneworks-candle-device-format-sidecar-v1\0";
const MAX_RUNTIME_INDEX_BYTES: usize = 64 * 1024 * 1024;
const MAX_RUNTIME_INDEX_ENTRIES: usize = 1_000_000;
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

struct RuntimeSidecarPayload {
    mapped: MmapedSafetensors,
    entry: SidecarEntry,
}
impl RuntimeSidecarPayload {
    fn upload(self, device: &Device) -> Result<QTensor> {
        let payload = self.mapped.get(PAYLOAD_KEY)?;
        let storage = QStorage::from_data(Cow::Borrowed(payload.data()), device, self.entry.dtype)
            .map_err(|e| {
                Error::Msg(format!(
                    "device-format sidecar: transfer {} to {device:?}: {e}",
                    self.entry.path.display()
                ))
            })?;
        QTensor::new(storage, (self.entry.out_dim, self.entry.in_dim))
    }
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
/// Initial construction hashes sources and converts missing weights. Published runtime indexes
/// are trusted on subsequent opens, without source or payload validation. [`Self::load`] touches only the sidecar: it does not retain or consult the source
/// `MmapedSafetensors`, which is the property a block-window loader needs in order to exclude format
/// conversion from the per-window path.
#[derive(Debug)]
pub struct PackedWeightSidecars {
    entries: HashMap<String, SidecarEntry>,
    replacement: std::sync::OnceLock<Box<PackedWeightSidecars>>,
    prepared: Vec<PreparedProjection>,
    cache_dir: PathBuf,
    created: usize,
    reused: usize,
    source_bytes_hashed: u64,
    payload_bytes_hashed: u64,
    sidecar_bytes: u64,
    rebuild: Option<std::sync::Arc<RuntimeRebuild>>,
}

#[derive(Debug)]
struct RuntimeRebuild {
    paths: Vec<PathBuf>,
    component: PathBuf,
    packed: PackedConfig,
    device: Device,
    prefix: Option<String>,
    external: Option<PathBuf>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct RuntimeIndexEntry {
    base: String,
    source_hex: String,
    bits: usize,
    group: usize,
    rows: usize,
    cols: usize,
    bytes: usize,
}

impl PackedWeightSidecars {
    /// Prepare all packed projections from `source`, placing content-addressed artifacts beside the
    /// component directory. Published runtime indexes are loaded directly.
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
            true,
        )
    }

    /// Open the exact source paths and reuse their published runtime index.
    /// Source generation checks apply only when preparing a missing cache.
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
            true,
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
            true,
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
            true,
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
        let source = crate::loader::mmap_safetensors_multi(source_paths)?;
        if let Some(cache) = trusted_runtime_cache(
            &source,
            component_dir,
            packed,
            device,
            base_prefix,
            external_cache_root,
        )? {
            return Ok((source, cache));
        }
        let before = file_identities(source_paths)?;
        // Read-only mappings of the exact caller-supplied source files. The robust identity checks
        // immediately before and after open detect replacement across this pathname seam.
        let source = crate::loader::mmap_safetensors_multi(source_paths)?;
        #[cfg(test)]
        run_after_source_open_hook();
        let after_open = file_identities(source_paths)?;
        if before.is_some() && before != after_open {
            return Err(Error::Msg(
                "device-format sidecar: source files changed while opening their mappings"
                    .to_owned(),
            ));
        }
        let cache = Self::prepare_impl(
            &source,
            component_dir,
            packed,
            device,
            cancel,
            base_prefix,
            external_cache_root,
            true,
        )?;
        let after_prepare = file_identities(source_paths)?;
        if after_open.is_some() && after_open != after_prepare {
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
        trust_ready: bool,
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

        if trust_ready {
            if let Some(cache) = trusted_runtime_cache(
                source,
                component_dir,
                packed,
                device,
                base_prefix,
                external_cache_root,
            )? {
                return Ok(cache);
            }
        }
        let projections =
            prepare_projections(source, component_dir, bits, group_size, cancel, base_prefix)?;
        let source_bytes_hashed = projections.iter().fold(0u64, |sum, projection| {
            sum.saturating_add(projection.source_bytes as u64)
        });

        // An index miss can recover existing unpublished-index artifacts before attempting
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
            write_runtime_index(&cache, packed, base_prefix, &source.source_paths());
            return Ok(with_runtime_rebuild(
                cache,
                source,
                component_dir,
                packed,
                device,
                base_prefix,
                external_cache_root,
            ));
        }

        // The external location can be unusable (a temp-dir fallback that is not a private
        // directory of this user, sc-27414); that only matters if the adjacent cache is unusable too.
        let external_cache = external_cache_dir(component_dir, external_cache_root);
        if let Ok(external_cache) = &external_cache {
            if cache_is_complete(
                external_cache,
                &projections,
                cancel,
                &mut payload_bytes_hashed,
            )? {
                let cache = reused_cache(
                    projections,
                    external_cache.clone(),
                    source_bytes_hashed,
                    payload_bytes_hashed,
                );
                write_runtime_index(&cache, packed, base_prefix, &source.source_paths());
                return Ok(with_runtime_rebuild(
                    cache,
                    source,
                    component_dir,
                    packed,
                    device,
                    base_prefix,
                    external_cache_root,
                ));
            }
        }

        // Prefer the historical model-adjacent location when it is writable. A caller-provisioned
        // snapshot may legally be immutable, though, so failure to create/write there selects the
        // namespaced per-user external cache instead of disabling packed loading or retaining a full
        // tier's converted weights in anonymous memory.
        let (cache_dir, prepare_lock) = match open_writable_cache(&adjacent_cache) {
            Ok(lock) => (adjacent_cache, lock),
            Err(adjacent_error) => match external_cache.and_then(|external_cache| {
                open_writable_cache(&external_cache)
                    .map(|lock| (external_cache.clone(), lock))
                    .map_err(|error| {
                        std::io::Error::new(
                            error.kind(),
                            format!("{}: {error}", external_cache.display()),
                        )
                    })
            }) {
                Ok(selected) => selected,
                Err(external_error) => {
                    return Err(Error::Msg(format!(
                        "device-format sidecar: neither model-adjacent cache {} ({adjacent_error}) \
                         nor the external cache ({external_error}) is writable",
                        adjacent_cache.display(),
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
            rebuild: None,
            replacement: Default::default(),
        };
        write_runtime_index(&cache, packed, base_prefix, &source.source_paths());
        Ok(with_runtime_rebuild(
            cache,
            source,
            component_dir,
            packed,
            device,
            base_prefix,
            external_cache_root,
        ))
    }

    /// Materialize one already-converted projection on `device` from its mapped sidecar bytes.
    ///
    /// No source tensor or conversion parameter is accepted here by design: a caller cannot
    /// accidentally put the MLX-affine repack back into a block-window loop.
    pub fn load(&self, base: &str, device: &Device) -> Result<QTensor> {
        self.load_with_upload(base, &mut |payload| payload.upload(device))
    }

    fn load_with_upload(
        &self,
        base: &str,
        upload: &mut dyn FnMut(RuntimeSidecarPayload) -> Result<QTensor>,
    ) -> Result<QTensor> {
        if let Some(cache) = self.replacement.get() {
            return cache.load_with_upload(base, upload);
        }
        match self.read_runtime(base) {
            // Native upload failures are device errors, not evidence of an unreadable cache.
            Ok(payload) => upload(payload),
            Err(error) => {
                let Some(rebuild) = &self.rebuild else {
                    return Err(error);
                };
                if !self.entries.contains_key(base) {
                    return Err(error);
                }
                // Bypass every trusted index during repair. The existing preparation path
                // selects a writable adjacent/external cache and serializes recovery there.
                let source = crate::loader::mmap_safetensors_multi(&rebuild.paths)?;
                let cache = Self::prepare_impl(
                    &source,
                    &rebuild.component,
                    rebuild.packed,
                    &rebuild.device,
                    None,
                    rebuild.prefix.as_deref(),
                    rebuild.external.as_deref(),
                    false,
                )?;
                let payload = cache.read_runtime(base)?;
                let _ = self.replacement.set(Box::new(cache));
                upload(payload)
            }
        }
    }

    fn read_runtime(&self, base: &str) -> Result<RuntimeSidecarPayload> {
        let entry = self.entries.get(base).ok_or_else(|| {
            Error::Msg(format!(
                "device-format sidecar: no prepared projection `{base}` in {}",
                self.cache_dir.display()
            ))
        })?;
        // Immutable, atomically-published cache file: writers never modify a published path.
        let mapped = crate::loader::mmap_safetensors(&entry.path).map_err(|e| {
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
        Ok(RuntimeSidecarPayload {
            mapped,
            entry: entry.clone(),
        })
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
        if let Some(cache) = self.replacement.get() {
            return cache.cache_dir();
        }
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

    /// Payload bytes hashed during initial preparation. Every ready-index load reports zero.
    pub fn payload_bytes_hashed(&self) -> u64 {
        self.payload_bytes_hashed
    }

    pub fn sidecar_bytes(&self) -> u64 {
        self.sidecar_bytes
    }

    /// The exact content-addressed artifact for `base`, useful to diagnostics and evidence harnesses.
    pub fn path_for(&self, base: &str) -> Option<&Path> {
        if let Some(cache) = self.replacement.get() {
            return cache.path_for(base);
        }
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
        let words = weight.shape().last().copied().unwrap_or(0);
        let width = scales
            .shape()
            .last()
            .copied()
            .unwrap_or(0)
            .checked_mul(group_size)
            .filter(|&n| n != 0)
            .ok_or_else(|| Error::Msg(format!("{base}: empty/overflowing affine width")))?;
        let inferred_bits = words
            .checked_mul(32)
            .and_then(|n| n.checked_div(width))
            .filter(|bits| {
                matches!(bits, 4 | 8) && words.checked_mul(32) == width.checked_mul(*bits)
            })
            .ok_or_else(|| {
                Error::Msg(format!(
                    "{base}: unsupported affine bit width (tier q{bits})"
                ))
            })?;
        let bits = inferred_bits;
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

fn runtime_index_path(
    dir: &Path,
    packed: PackedConfig,
    prefix: Option<&str>,
    paths: &[PathBuf],
) -> PathBuf {
    let mut identity = Sha256::new();
    identity.update(b"candle-runtime-index-v1\0");
    let prefix = prefix.unwrap_or("").as_bytes();
    identity.update((prefix.len() as u64).to_le_bytes());
    identity.update(prefix);
    for path in paths {
        let path = std::path::absolute(path).unwrap_or_else(|_| path.clone());
        let bytes = path.as_os_str().as_encoded_bytes();
        identity.update((bytes.len() as u64).to_le_bytes());
        identity.update(bytes);
    }
    let selection = hex(&identity.finalize());
    dir.join(format!(
        "runtime-index-v1-q{}-g{}-{selection}.json",
        packed.bits, packed.group_size
    ))
}

fn with_runtime_rebuild(
    mut cache: PackedWeightSidecars,
    source: &MmapedSafetensors,
    component: &Path,
    packed: PackedConfig,
    device: &Device,
    prefix: Option<&str>,
    external: Option<&Path>,
) -> PackedWeightSidecars {
    let paths = source.source_paths();
    if !paths.is_empty() {
        cache.rebuild = Some(std::sync::Arc::new(RuntimeRebuild {
            paths,
            component: component.to_owned(),
            packed,
            device: device.clone(),
            prefix: prefix.map(str::to_owned),
            external: external.map(Path::to_owned),
        }));
    }
    cache
}

fn write_runtime_index(
    cache: &PackedWeightSidecars,
    packed: PackedConfig,
    prefix: Option<&str>,
    paths: &[PathBuf],
) {
    let entries: Vec<_> = cache
        .prepared
        .iter()
        .map(|p| RuntimeIndexEntry {
            base: p.entry_base.clone(),
            source_hex: p.source_hex.clone(),
            bits: p.bits,
            group: p.group_size,
            rows: p.out_dim,
            cols: p.in_dim,
            bytes: p.payload_bytes,
        })
        .collect();
    if let Ok(bytes) = serde_json::to_vec(&entries) {
        let path = runtime_index_path(&cache.cache_dir, packed, prefix, paths);
        let tmp = path.with_extension(format!(
            "tmp-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        // Cache directories can already contain staged files. Never follow or truncate an
        // existing temporary pathname when publishing a derived index.
        if let Ok(mut file) = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
        {
            let written = file.write_all(&bytes);
            drop(file);
            if written.is_ok() {
                // A missing index is harmless: a reader can reconstruct it from runtime headers.
                let _ = fs::remove_file(&path);
                if fs::rename(&tmp, path).is_ok() {
                    return;
                }
            }
            let _ = fs::remove_file(tmp);
        }
    }
}

/// Decode runtime geometry from already-mapped headers, never source/payload contents. Cached
/// bytes are trusted, but their shape must still be representable by the native GGML decoder.
fn runtime_index_geometry(
    source: &MmapedSafetensors,
    entry: &RuntimeIndexEntry,
    group: usize,
) -> Result<()> {
    if entry.group != group {
        candle_core::bail!("runtime index group does not match its selection");
    }
    let (base, index) = if source.get(&format!("{}.weight", entry.base)).is_ok() {
        (entry.base.as_str(), None)
    } else {
        let (base, index) = entry
            .base
            .rsplit_once('[')
            .ok_or_else(|| Error::Msg("runtime index has no source projection".into()))?;
        let index = index
            .strip_suffix(']')
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| Error::Msg("runtime index has an invalid expert slice".into()))?;
        (base, Some(index))
    };
    let (leading, geometry) = source_geometry(
        base,
        &source.get(&format!("{base}.weight"))?,
        &source.get(&format!("{base}.scales"))?,
        &source.get(&format!("{base}.biases"))?,
        entry.bits,
        group,
    )?;
    let slice_matches = match (leading, index) {
        (None, None) => true,
        (Some(n), Some(i)) => i < n,
        _ => false,
    };
    let dtype = if entry.bits == 4 {
        GgmlDType::Q4_1
    } else {
        GgmlDType::Q8_0
    };
    if !slice_matches
        || entry.rows != geometry.out_dim
        || entry.cols != geometry.in_dim
        || entry.bytes != payload_len(dtype, geometry.out_dim, geometry.in_dim)?
    {
        candle_core::bail!("runtime index geometry cannot decode its source projection");
    }
    Ok(())
}

fn read_runtime_index(path: &Path) -> Option<Vec<RuntimeIndexEntry>> {
    let file = File::open(path).ok()?;
    let metadata = file.metadata().ok()?;
    if !metadata.is_file() {
        return None;
    }
    decode_runtime_index(file, metadata.len(), MAX_RUNTIME_INDEX_BYTES)
}

fn decode_runtime_index(
    reader: impl Read,
    declared_len: u64,
    limit: usize,
) -> Option<Vec<RuntimeIndexEntry>> {
    // Reject sparse/oversized files before reserving a buffer. `take` also bounds a file that
    // grows after metadata was read; this is parsing a small index, never validating weights.
    if declared_len > limit as u64 {
        return None;
    }
    let mut bytes = Vec::with_capacity(declared_len as usize);
    reader.take(limit as u64 + 1).read_to_end(&mut bytes).ok()?;
    if bytes.len() > limit {
        return None;
    }
    let entries: Vec<RuntimeIndexEntry> = serde_json::from_slice(&bytes).ok()?;
    (entries.len() <= MAX_RUNTIME_INDEX_ENTRIES).then_some(entries)
}

fn trusted_runtime_cache(
    source: &MmapedSafetensors,
    component: &Path,
    packed: PackedConfig,
    device: &Device,
    prefix: Option<&str>,
    external: Option<&Path>,
) -> Result<Option<PackedWeightSidecars>> {
    if !matches!(packed.bits, 4 | 8) || packed.group_size <= 0 {
        return Ok(None);
    }
    let source_paths = source.source_paths();
    if source_paths.is_empty() {
        return Ok(None);
    }
    let component_identity =
        std::path::absolute(component).unwrap_or_else(|_| component.to_owned());
    let legacy_owner_matches = source_paths.iter().all(|path| {
        std::path::absolute(path)
            .ok()
            .and_then(|path| path.parent().map(Path::to_owned))
            .as_ref()
            == Some(&component_identity)
    });
    let mut directories = vec![component.join(CACHE_DIR)];
    if let Ok(dir) = external_cache_dir(component, external) {
        directories.push(dir);
    }
    for dir in directories {
        let mut indexed =
            read_runtime_index(&runtime_index_path(&dir, packed, prefix, &source_paths));
        let had_exact_index = indexed.is_some();
        // A complete index for these exact source paths also serves a narrower selection.
        if indexed.is_none() && prefix.is_some() {
            indexed = read_runtime_index(&runtime_index_path(&dir, packed, None, &source_paths))
                .map(|entries| {
                    entries
                        .into_iter()
                        .filter(|entry| entry.base.starts_with(prefix.unwrap_or("")))
                        .collect()
                });
        }
        let had_index = had_exact_index;
        let entries = match indexed {
            Some(entries) => entries,
            None => {
                if !legacy_owner_matches {
                    continue;
                }
                // Once indexed, a cache belongs to its recorded source names. Do not reinterpret
                // another source's artifacts as a legacy cache merely because parameter names match.
                if fs::read_dir(&dir)
                    .into_iter()
                    .flatten()
                    .flatten()
                    .any(|entry| {
                        entry
                            .file_name()
                            .to_string_lossy()
                            .starts_with("runtime-index-v1-")
                    })
                {
                    continue;
                }
                let mut wanted = std::collections::BTreeSet::new();
                for (name, scales) in source.tensors() {
                    if let Some(base) = name.strip_suffix(".scales") {
                        if prefix.is_some_and(|prefix| !base.starts_with(prefix)) {
                            continue;
                        }
                        match scales.shape() {
                            [_, _] => {
                                wanted.insert(base.to_owned());
                            }
                            [experts, _, _] => {
                                for index in 0..*experts {
                                    wanted.insert(sidecar_entry_key(base, Some(index)));
                                }
                            }
                            _ => return Ok(None),
                        }
                    }
                }
                if wanted.is_empty() {
                    return Ok(None);
                }

                // Migrate existing sidecars by reading only their tiny safetensors headers.
                // Their published source_base/shape/format metadata already names the runtime data.
                let mut found: std::collections::BTreeMap<
                    String,
                    (Option<std::time::SystemTime>, RuntimeIndexEntry),
                > = std::collections::BTreeMap::new();
                for entry in fs::read_dir(&dir).into_iter().flatten().flatten() {
                    if entry.path().extension().and_then(|v| v.to_str()) != Some("safetensors") {
                        continue;
                    }
                    let parsed = (|| -> Option<RuntimeIndexEntry> {
                        let mut file = File::open(entry.path()).ok()?;
                        let mut size = [0u8; 8];
                        file.read_exact(&mut size).ok()?;
                        let len = usize::try_from(u64::from_le_bytes(size)).ok()?;
                        if len > 16 * 1024 * 1024 {
                            return None;
                        }
                        let mut header = vec![0; len];
                        file.read_exact(&mut header).ok()?;
                        let header: serde_json::Value = serde_json::from_slice(&header).ok()?;
                        let meta = header.get("__metadata__")?;
                        let string = |key: &str| meta.get(key)?.as_str();
                        let base = string("source_base")?.to_owned();
                        if !wanted.contains(&base) {
                            return None;
                        }
                        let bits = string("source_bits")?.parse().ok()?;
                        let group = string("source_group_size")?.parse().ok()?;
                        if !matches!(bits, 4 | 8) || group != packed.group_size as usize {
                            return None;
                        }
                        Some(RuntimeIndexEntry {
                            base,
                            source_hex: string("source_sha256")?.to_owned(),
                            bits,
                            group,
                            rows: string("out_dim")?.parse().ok()?,
                            cols: string("in_dim")?.parse().ok()?,
                            bytes: header
                                .get(PAYLOAD_KEY)?
                                .get("shape")?
                                .as_array()?
                                .first()?
                                .as_u64()?
                                .try_into()
                                .ok()?,
                        })
                    })();
                    if let Some(value) = parsed {
                        let modified = entry.metadata().and_then(|m| m.modified()).ok();
                        let replace = found
                            .get(&value.base)
                            .is_none_or(|(old, _)| modified > *old);
                        if replace {
                            found.insert(value.base.clone(), (modified, value));
                        }
                    }
                }
                if found
                    .keys()
                    .cloned()
                    .collect::<std::collections::BTreeSet<_>>()
                    != wanted
                {
                    continue;
                }
                found.into_values().map(|(_, value)| value).collect()
            }
        };
        if entries.is_empty() {
            continue;
        }
        let mut prepared = Vec::new();
        for entry in entries {
            // This is index parsing/path confinement, not a source or payload integrity pass.
            if entry.source_hex.len() != 64
                || !entry.source_hex.bytes().all(|b| b.is_ascii_hexdigit())
                || !matches!(entry.bits, 4 | 8)
                || runtime_index_geometry(source, &entry, packed.group_size as usize).is_err()
            {
                return Ok(None);
            }
            let dtype_name = if entry.bits == 4 { "q4_1" } else { "q8_0" };
            prepared.push(PreparedProjection {
                entry_base: entry.base.clone(),
                source_hex: entry.source_hex,
                source_base: entry.base,
                source_slice: None,
                bits: entry.bits,
                group_size: entry.group,
                dtype: if entry.bits == 4 {
                    GgmlDType::Q4_1
                } else {
                    GgmlDType::Q8_0
                },
                dtype_name,
                out_dim: entry.rows,
                in_dim: entry.cols,
                payload_bytes: entry.bytes,
                source_bytes: 0,
            });
        }
        let cache = reused_cache(prepared, dir, 0, 0);
        if !had_index {
            write_runtime_index(&cache, packed, prefix, &source.source_paths());
        }
        return Ok(Some(with_runtime_rebuild(
            cache, source, component, packed, device, prefix, external,
        )));
    }
    Ok(None)
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
        rebuild: None,
        replacement: Default::default(),
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

fn external_cache_dir(
    component_dir: &Path,
    explicit_root: Option<&Path>,
) -> std::io::Result<PathBuf> {
    let root = match explicit_root {
        Some(root) => root.to_path_buf(),
        None => default_external_cache_root()?,
    };
    let identity = fs::canonicalize(component_dir).unwrap_or_else(|_| component_dir.to_path_buf());
    let mut digest = Sha256::new();
    digest.update(b"sceneworks-candle-device-format-component-v1\0");
    digest.update(identity.to_string_lossy().as_bytes());
    let namespace = hex(&digest.finalize());
    Ok(root.join("candle-device-format-v1").join(namespace))
}

/// The external cache root without an explicit one: the override, else the per-user cache, else a
/// private temp-dir fallback ([`private_temp_cache_root`]).
fn default_external_cache_root() -> std::io::Result<PathBuf> {
    match per_user_cache_root() {
        Some(root) => Ok(root),
        None => private_temp_cache_root(&std::env::temp_dir()),
    }
}

fn per_user_cache_root() -> Option<PathBuf> {
    if let Some(root) = std::env::var_os(CACHE_DIR_ENV).filter(|root| !root.is_empty()) {
        return Some(PathBuf::from(root));
    }
    #[cfg(target_os = "windows")]
    if let Some(root) = std::env::var_os("LOCALAPPDATA") {
        return Some(PathBuf::from(root).join("SceneWorks"));
    }
    #[cfg(target_os = "macos")]
    if let Some(home) = std::env::var_os("HOME") {
        return Some(PathBuf::from(home).join("Library/Caches/SceneWorks"));
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if let Some(root) = std::env::var_os("XDG_CACHE_HOME") {
            return Some(PathBuf::from(root).join("sceneworks"));
        }
        if let Some(home) = std::env::var_os("HOME") {
            return Some(PathBuf::from(home).join(".cache/sceneworks"));
        }
    }
    None
}

/// The documented override of the external cache root.
const CACHE_DIR_ENV: &str = "SCENEWORKS_CANDLE_DEVICE_CACHE_DIR";

/// The last-resort cache root under `tmp`. On Unix the name carries the effective uid, so users
/// sharing `/tmp` never share (or contend for) one cache.
fn temp_cache_root_path(tmp: &Path) -> PathBuf {
    #[cfg(unix)]
    {
        // SAFETY: geteuid takes no arguments, has no preconditions and cannot fail.
        let uid = unsafe { libc::geteuid() };
        tmp.join(format!("sceneworks-candle-cache-{uid}"))
    }
    #[cfg(not(unix))]
    {
        tmp.join("sceneworks-candle-cache")
    }
}

/// The temp-dir fallback, created owner-only (0700) if missing (sc-27414). A shared `/tmp` lets
/// another local user pre-create a predictable name to block writes or plant a self-consistent
/// artifact, so on Unix an existing directory is used only if it is a real directory (not a
/// symlink) owned by the effective uid with no group/other permission bits; otherwise this is an
/// error naming [`CACHE_DIR_ENV`]. Windows/macOS temp directories are already per-user.
#[cfg(unix)]
fn private_temp_cache_root(tmp: &Path) -> std::io::Result<PathBuf> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};
    let root = temp_cache_root_path(tmp);
    match fs::DirBuilder::new().mode(0o700).create(&root) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error),
    }
    let metadata = fs::symlink_metadata(&root)?;
    // SAFETY: geteuid takes no arguments, has no preconditions and cannot fail.
    let uid = unsafe { libc::geteuid() };
    if !metadata.is_dir() || metadata.uid() != uid || metadata.mode() & 0o077 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!(
                "temp-dir cache fallback {} is not a private directory of uid {uid} (owner {}, \
                 mode {:o}); set {CACHE_DIR_ENV} (or HOME / XDG_CACHE_HOME) to a per-user cache \
                 directory",
                root.display(),
                metadata.uid(),
                metadata.mode() & 0o7777
            ),
        ));
    }
    Ok(root)
}

#[cfg(not(unix))]
fn private_temp_cache_root(tmp: &Path) -> std::io::Result<PathBuf> {
    Ok(temp_cache_root_path(tmp))
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

fn source_geometry(
    base: &str,
    weight: &safetensors::tensor::TensorView<'_>,
    scales: &safetensors::tensor::TensorView<'_>,
    biases: &safetensors::tensor::TensorView<'_>,
    bits: usize,
    group_size: usize,
) -> Result<(Option<usize>, SourceProjection)> {
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
    Ok((
        leading,
        SourceProjection {
            index: None,
            out_dim,
            in_dim,
            source_bytes,
        },
    ))
}

fn validate_source_shapes(
    base: &str,
    weight: &safetensors::tensor::TensorView<'_>,
    scales: &safetensors::tensor::TensorView<'_>,
    biases: &safetensors::tensor::TensorView<'_>,
    bits: usize,
    group_size: usize,
) -> Result<Vec<SourceProjection>> {
    let (leading, projection) = source_geometry(base, weight, scales, biases, bits, group_size)?;
    Ok((0..leading.unwrap_or(1))
        .map(|index| SourceProjection {
            index: leading.map(|_| index),
            ..projection
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
        candle_core::safetensors::with_source_bytes(
            chunk,
            crate::gen_core::LoadActivityPhase::Verifying,
            || hash.update(chunk),
        );
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
    // This producer already owns a durable runtime artifact. Avoid creating a second saved
    // quantization for its transient operands while constructing that same artifact.
    let uncached = crate::gen_core::QuantizationCacheContext::default();
    let _scope = uncached.install();
    let qtensor = candle_core::quantized::runtime_cache::without_cache(|| {
        repack_packed_weight(&wq, &scales, &biases, group_size, device)
    })?;
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
    // Validation maps a read-only cache artifact and performs no concurrent mutation.
    let mapped = match crate::loader::mmap_safetensors(path) {
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
    Ok(candle_core::safetensors::read_tensor_bytes(
        &stored_hash,
        |bytes| bytes == actual_hash,
    ))
}

fn validation_payload_digest(bytes: &[u8], cancel: Option<&CancelFlag>) -> Result<[u8; 32]> {
    check_cancel(cancel)?;
    let mut hash = Sha256::new();
    for chunk in bytes.chunks(HASH_CANCEL_CHUNK_BYTES) {
        check_cancel(cancel)?;
        candle_core::safetensors::with_source_bytes(
            chunk,
            crate::gen_core::LoadActivityPhase::Verifying,
            || hash.update(chunk),
        );
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

    /// sc-27414: the temp-dir fallback is uid-named and created owner-only; a pre-existing
    /// directory another user could control (wrong mode, or a symlink) is refused with an error
    /// naming the override instead of being used.
    #[cfg(unix)]
    #[test]
    fn temp_fallback_cache_root_is_private_to_the_effective_user() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = TestDir::new("temp-fallback");
        let root = private_temp_cache_root(tmp.path()).unwrap();
        assert_eq!(root, temp_cache_root_path(tmp.path()));
        // SAFETY: geteuid takes no arguments, has no preconditions and cannot fail.
        let uid = unsafe { libc::geteuid() };
        assert!(root.ends_with(format!("sceneworks-candle-cache-{uid}")));
        let mode = fs::metadata(&root).unwrap().permissions().mode();
        assert_eq!(mode & 0o077, 0, "created owner-only, got {mode:o}");
        // Reusing the private directory is fine.
        assert_eq!(private_temp_cache_root(tmp.path()).unwrap(), root);

        // A pre-created world-writable directory is refused, naming the override.
        fs::set_permissions(&root, fs::Permissions::from_mode(0o777)).unwrap();
        let err = private_temp_cache_root(tmp.path()).unwrap_err();
        assert!(err.to_string().contains(CACHE_DIR_ENV), "{err}");
        fs::remove_dir(&root).unwrap();

        // A symlink planted at the predictable name is refused even if it points somewhere private.
        let target = tmp.path().join("elsewhere");
        fs::create_dir(&target).unwrap();
        fs::set_permissions(&target, fs::Permissions::from_mode(0o700)).unwrap();
        std::os::unix::fs::symlink(&target, &root).unwrap();
        assert!(private_temp_cache_root(tmp.path()).is_err());
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
    fn warm_indexes_skip_payload_validation_and_still_honor_cancellation() -> Result<()> {
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
                true,
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
            // Deterministic mutation hook: cancel after hashing the first payload chunk. This proves
            // the error originates inside artifact validation, not before source hashing or between
            // the two entries, without relying on scheduler timing.
            CANCEL_VALIDATION_AFTER_CHUNKS.with(|remaining| remaining.set(Some(1)));
            let warm = PackedWeightSidecars::prepare_impl(
                &source,
                component.path(),
                packed,
                &Device::Cpu,
                Some(&cancel),
                None,
                Some(external.path()),
                true,
            )?;
            CANCEL_VALIDATION_AFTER_CHUNKS.with(|remaining| remaining.set(None));

            assert!(
                !cancel.is_cancelled(),
                "a warm hit must never enter payload validation"
            );
            assert_eq!(warm.source_bytes_hashed(), 0);
            assert_eq!(warm.payload_bytes_hashed(), 0);
            cancel.cancel();
            assert!(PackedWeightSidecars::prepare_cancelable(
                &source,
                component.path(),
                packed,
                &Device::Cpu,
                &cancel
            )
            .is_err());
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
        let activities = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = activities.clone();
        let _observer = crate::gen_core::LoadActivityObserver::new(move |activity| {
            crate::lock_recover(&captured).push(activity);
        });
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
        assert!(crate::lock_recover(&activities)
            .iter()
            .flatten()
            .any(|activity| activity.phase == crate::gen_core::LoadActivityPhase::Verifying));
        crate::lock_recover(&activities).clear();
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
        assert!(
            crate::lock_recover(&activities)
                .iter()
                .flatten()
                .all(|activity| activity.phase != crate::gen_core::LoadActivityPhase::Verifying),
            "memo hits must not claim to verify files"
        );
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
    fn mixed_q4_q8_components_reopen_directly_through_the_shared_runtime_index() -> Result<()> {
        let dir = TestDir::new("mixed-width-shared");
        let file = dir.path().join("model.safetensors");
        let dense = Tensor::arange(0f32, 128f32, &Device::Cpu)?.reshape((2, 64))?;
        let mut tensors = HashMap::new();
        for (base, bits) in [("layers.q4", 4), ("layers.q8", 8)] {
            let (weight, scales, biases) = pack_mlx_affine(&dense, bits, 64)?;
            tensors.insert(format!("{base}.weight"), weight);
            tensors.insert(format!("{base}.scales"), scales);
            tensors.insert(format!("{base}.biases"), biases);
        }
        safetensors::save(&tensors, &file)?;
        let packed = PackedConfig {
            bits: 4,
            group_size: 64,
        };
        let (_, first) = open_and_prepare_path(&file, dir.path(), packed)?;
        assert_eq!(first.created_count(), 2);
        assert_eq!(
            first.load("layers.q4", &Device::Cpu)?.dtype(),
            GgmlDType::Q4_1
        );
        assert_eq!(
            first.load("layers.q8", &Device::Cpu)?.dtype(),
            GgmlDType::Q8_0
        );
        let (_, reopened) = open_and_prepare_path(&file, dir.path(), packed)?;
        assert_eq!(reopened.source_bytes_hashed(), 0);
        assert_eq!(reopened.payload_bytes_hashed(), 0);
        assert_eq!(reopened.reused_count(), 2);
        assert_eq!(
            reopened.load("layers.q4", &Device::Cpu)?.data()?.as_ref(),
            first.load("layers.q4", &Device::Cpu)?.data()?.as_ref()
        );
        assert_eq!(
            reopened.load("layers.q8", &Device::Cpu)?.data()?.as_ref(),
            first.load("layers.q8", &Device::Cpu)?.data()?.as_ref()
        );
        Ok(())
    }

    #[test]
    fn device_upload_failure_does_not_reopen_sources_or_repair_a_valid_sidecar() -> Result<()> {
        let dir = TestDir::new("upload-failure");
        let file = write_source(dir.path(), 8, 0.)?;
        let source = open(&file)?;
        let cache = PackedWeightSidecars::prepare(
            &source,
            dir.path(),
            PackedConfig {
                bits: 8,
                group_size: 64,
            },
            &Device::Cpu,
        )?;
        drop(source);
        fs::remove_file(file)?;
        let failure = cache.load_with_upload("layers.0.proj", &mut |_| {
            Err(Error::Msg("simulated device OOM".into()))
        });
        assert!(failure
            .err()
            .unwrap()
            .to_string()
            .contains("simulated device OOM"));
        assert_eq!(
            cache.load("layers.0.proj", &Device::Cpu)?.shape().dims(),
            &[2, 64]
        );
        assert!(
            cache.replacement.get().is_none(),
            "device failures must not enter cache repair"
        );
        Ok(())
    }

    #[test]
    fn runtime_index_size_is_bounded_before_reading_and_when_a_file_grows() {
        struct NeverRead;
        impl Read for NeverRead {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                panic!("oversized index must be rejected before any read or allocation");
            }
        }
        assert!(decode_runtime_index(
            NeverRead,
            MAX_RUNTIME_INDEX_BYTES as u64 + 1,
            MAX_RUNTIME_INDEX_BYTES
        )
        .is_none());
        let mut growing = std::io::Cursor::new(b"[]                    ".as_slice());
        assert!(decode_runtime_index(&mut growing, 2, 8).is_none());
        assert_eq!(
            growing.position(),
            9,
            "a growing index must stop at the bound plus one byte"
        );
        assert!(decode_runtime_index(std::io::Cursor::new(b"[]"), 2, 8).is_some());
    }

    #[test]
    fn malformed_runtime_index_geometry_never_reaches_native_allocation() -> Result<()> {
        let dir = TestDir::new("index-geometry");
        let file = write_source(dir.path(), 8, 0.)?;
        let source = open(&file)?;
        let packed = PackedConfig {
            bits: 8,
            group_size: 64,
        };
        let first = PackedWeightSidecars::prepare(&source, dir.path(), packed, &Device::Cpu)?;
        let path = runtime_index_path(&first.cache_dir, packed, None, &source.source_paths());
        let original = fs::read(&path)?;
        for (field, value) in [
            ("rows", usize::MAX),
            ("cols", usize::MAX),
            ("bytes", 1),
            ("group", 0),
            ("bits", 4),
        ] {
            let mut index: Vec<RuntimeIndexEntry> =
                serde_json::from_slice(&original).map_err(Error::wrap)?;
            match field {
                "rows" => index[0].rows = value,
                "cols" => index[0].cols = value,
                "bytes" => index[0].bytes = value,
                "group" => index[0].group = value,
                _ => index[0].bits = value,
            }
            fs::write(&path, serde_json::to_vec(&index).map_err(Error::wrap)?)?;
            let repaired =
                PackedWeightSidecars::prepare(&source, dir.path(), packed, &Device::Cpu)?;
            assert_eq!(
                repaired.load("layers.0.proj", &Device::Cpu)?.shape().dims(),
                &[2, 64]
            );
        }
        Ok(())
    }

    #[test]
    fn prefix_repair_bypasses_the_full_component_ready_index() -> Result<()> {
        let dir = TestDir::new("prefix-repair");
        let file = write_multi_source(dir.path(), 4)?;
        let source = open(&file)?;
        let packed = PackedConfig {
            bits: 4,
            group_size: 64,
        };
        let full = PackedWeightSidecars::prepare(&source, dir.path(), packed, &Device::Cpu)?;
        let expected = full
            .load("layers.0.proj", &Device::Cpu)?
            .data()?
            .into_owned();
        let prefix = PackedWeightSidecars::prepare_prefix_cancelable(
            &source,
            dir.path(),
            packed,
            &Device::Cpu,
            &CancelFlag::default(),
            "layers.0.",
        )?;
        fs::write(&full.entries["layers.0.proj"].path, b"truncated")?;
        assert_eq!(
            prefix.load("layers.0.proj", &Device::Cpu)?.data()?.as_ref(),
            expected
        );
        assert_eq!(
            full.load("layers.1.proj", &Device::Cpu)?.dtype(),
            GgmlDType::Q4_1
        );
        Ok(())
    }

    #[test]
    fn indexed_adjacent_cache_without_writer_access_repairs_in_external_cache() -> Result<()> {
        let dir = TestDir::new("external-repair");
        let external = TestDir::new("external-repair-output");
        let file = write_source(dir.path(), 8, 0.)?;
        let source = open(&file)?;
        let packed = PackedConfig {
            bits: 8,
            group_size: 64,
        };
        let full = PackedWeightSidecars::prepare_with_external_cache_root(
            &source,
            dir.path(),
            packed,
            &Device::Cpu,
            external.path(),
        )?;
        let expected = full
            .load("layers.0.proj", &Device::Cpu)?
            .data()?
            .into_owned();
        fs::remove_file(&full.entries["layers.0.proj"].path)?;
        // Make opening an adjacent writer fail deterministically on both Windows and Unix.
        // Read-only filesystems reach the same existing writable-cache selection path.
        let lock = full.cache_dir.join(PREPARE_LOCK);
        fs::remove_file(&lock)?;
        fs::create_dir(&lock)?;
        assert_eq!(
            full.load("layers.0.proj", &Device::Cpu)?.data()?.as_ref(),
            expected
        );
        assert!(full
            .path_for("layers.0.proj")
            .unwrap()
            .starts_with(external.path()));
        drop(source);
        fs::remove_file(&file)?;
        assert_eq!(
            full.load("layers.0.proj", &Device::Cpu)?.data()?.as_ref(),
            expected,
            "subsequent windows must load the repaired runtime cache without reopening sources"
        );
        let output = external_cache_dir(dir.path(), Some(external.path()))?;
        assert!(output
            .join(full.entries["layers.0.proj"].path.file_name().unwrap())
            .is_file());
        Ok(())
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
    fn runtime_cache_does_not_validate_replaced_payload_bytes() -> Result<()> {
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
        assert_eq!(rebuilt.created_count(), 0);
        assert_eq!(rebuilt.source_bytes_hashed(), 0);
        assert_eq!(rebuilt.payload_bytes_hashed(), 0);
        rebuilt.load("layers.0.proj", &Device::Cpu)?;
        Ok(())
    }

    #[test]
    fn runtime_cache_does_not_hash_in_place_payload_changes() -> Result<()> {
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
        assert_eq!(rebuilt.created_count(), 0);
        assert_eq!(rebuilt.source_bytes_hashed(), 0);
        assert_eq!(rebuilt.payload_bytes_hashed(), 0);
        rebuilt.load("layers.0.proj", &Device::Cpu)?;
        Ok(())
    }

    #[test]
    fn ready_runtime_cache_is_authoritative_without_rehashing_source_files() -> Result<()> {
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
        assert_eq!(changed.source_bytes_hashed(), 0);
        assert_eq!(changed.payload_bytes_hashed(), 0);
        assert_eq!(changed.path_for("layers.0.proj").unwrap(), first_path);
        Ok(())
    }

    #[test]
    fn runtime_cache_reuses_ready_artifacts_without_generation_checks() -> Result<()> {
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
        assert_eq!(reopened.source_bytes_hashed(), 0);
        assert_eq!(reopened.payload_bytes_hashed(), 0);
        Ok(())
    }

    #[test]
    fn unreadable_sidecar_rebuilds_at_decode_without_preload_validation() -> Result<()> {
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
        assert_eq!(rebuilt.created_count(), 0);
        assert_eq!(rebuilt.source_bytes_hashed(), 0);
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
        assert_eq!(results.len(), 2);
        // Both readers decode successfully. Recovery happens lazily under the shared producer
        // lock; construction itself must not verify every payload merely to count repairs.
        assert!(validate_sidecar(
            &path,
            payload_len(GgmlDType::Q8_0, 2, 64)?,
            None,
        )?);
        Ok(())
    }
}
