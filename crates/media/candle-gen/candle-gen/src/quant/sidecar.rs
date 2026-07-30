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
//!    `.candle-device-format-v1/` beside the source component; and
//! 4. mmap that sidecar for each materialization and copy its already-device-format bytes directly
//!    to the requested device.
//!
//! The cache is content-addressed over the source bytes, not timestamps. Replacing or changing a
//! tier therefore selects a different sidecar path automatically. A window holds no anonymous host
//! allocation proportional to the tier: the mapped payload is reclaimable page cache, and
//! [`QStorage::from_data`] copies it to the device before the mapping is dropped. First creation is
//! deliberately projection-at-a-time, bounding the q8 dense conversion transient to one projection.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Error, Result};
use fs2::FileExt;
use safetensors::tensor::{Dtype as SafeDtype, View};
use sha2::{Digest, Sha256};

use super::{repack_packed_weight, PackedConfig};

const CACHE_DIR: &str = ".candle-device-format-v1";
const PREPARE_LOCK: &str = ".prepare.lock";
const PAYLOAD_KEY: &str = "weight";
const PAYLOAD_HASH_KEY: &str = "payload_sha256";
const FORMAT_DOMAIN: &[u8] = b"sceneworks-candle-device-format-sidecar-v1\0";
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
struct SidecarEntry {
    path: PathBuf,
    dtype: GgmlDType,
    out_dim: usize,
    in_dim: usize,
    payload_bytes: usize,
}

/// Prepared device-format sidecars for every MLX-packed projection in one component.
///
/// Construction performs all source hashing and any missing conversions. [`Self::load`] touches only
/// the sidecar: it does not retain or consult the source `MmapedSafetensors`, which is the property a
/// block-window loader needs in order to exclude format conversion from the per-window path.
#[derive(Debug)]
pub struct PackedWeightSidecars {
    entries: HashMap<String, SidecarEntry>,
    cache_dir: PathBuf,
    created: usize,
    reused: usize,
    source_bytes_hashed: u64,
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

        let cache_dir = component_dir.join(CACHE_DIR);
        fs::create_dir_all(&cache_dir).map_err(|e| {
            Error::Msg(format!(
                "device-format sidecar: create {}: {e}",
                cache_dir.display()
            ))
        })?;
        // Serialize validation, corrupt recovery, and publication across processes. Without this
        // lock, two readers can both observe a corrupt final path and one can delete the valid
        // replacement the other just published. The lock file persists as a zero-byte cache
        // coordination artifact; the OS releases the advisory lock on every return or process exit.
        let prepare_lock = fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(cache_dir.join(PREPARE_LOCK))?;
        FileExt::lock_exclusive(&prepare_lock).map_err(|e| {
            Error::Msg(format!(
                "device-format sidecar: lock {}: {e}",
                cache_dir.display()
            ))
        })?;

        let mut bases: Vec<String> = source
            .tensors()
            .into_iter()
            .filter_map(|(name, _)| name.strip_suffix(".scales").map(str::to_owned))
            .collect();
        bases.sort();
        bases.dedup();
        if bases.is_empty() {
            return Err(Error::Msg(format!(
                "device-format sidecar: packed component {} has no `.scales` triples",
                component_dir.display()
            )));
        }

        let mut entries = HashMap::with_capacity(bases.len());
        let mut created = 0usize;
        let mut reused = 0usize;
        let mut source_bytes_hashed = 0u64;
        let mut sidecar_bytes = 0u64;
        for base in bases {
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
            let (out_dim, in_dim) =
                validate_source_shapes(&base, &weight, &scales, &biases, bits, group_size)?;

            let source_digest = source_digest(
                &base,
                bits,
                group_size,
                [
                    (&weight_key, &weight),
                    (&scales_key, &scales),
                    (&biases_key, &biases),
                ],
            );
            source_bytes_hashed = source_bytes_hashed
                .saturating_add(weight.data().len() as u64)
                .saturating_add(scales.data().len() as u64)
                .saturating_add(biases.data().len() as u64);
            let source_hex = hex(&source_digest);
            let dtype = if bits == 4 {
                GgmlDType::Q4_1
            } else {
                GgmlDType::Q8_0
            };
            let dtype_name = if bits == 4 { "q4_1" } else { "q8_0" };
            let payload_bytes = payload_len(dtype, out_dim, in_dim)?;
            let path = cache_dir.join(format!("{source_hex}.{dtype_name}.safetensors"));

            let valid = validate_sidecar(&path, payload_bytes)?;
            if valid {
                reused += 1;
            } else {
                build_sidecar(
                    source,
                    &path,
                    &source_hex,
                    &base,
                    bits,
                    group_size,
                    out_dim,
                    in_dim,
                    dtype,
                    device,
                )?;
                if !validate_sidecar(&path, payload_bytes)? {
                    return Err(Error::Msg(format!(
                        "device-format sidecar: freshly written artifact {} failed validation",
                        path.display()
                    )));
                }
                created += 1;
            }
            sidecar_bytes = sidecar_bytes.saturating_add(payload_bytes as u64);
            entries.insert(
                base,
                SidecarEntry {
                    path,
                    dtype,
                    out_dim,
                    in_dim,
                    payload_bytes,
                },
            );
        }

        Ok(Self {
            entries,
            cache_dir,
            created,
            reused,
            source_bytes_hashed,
            sidecar_bytes,
        })
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

    pub fn sidecar_bytes(&self) -> u64 {
        self.sidecar_bytes
    }

    /// The exact content-addressed artifact for `base`, useful to diagnostics and evidence harnesses.
    pub fn path_for(&self, base: &str) -> Option<&Path> {
        self.entries.get(base).map(|entry| entry.path.as_path())
    }
}

fn validate_source_shapes(
    base: &str,
    weight: &safetensors::tensor::TensorView<'_>,
    scales: &safetensors::tensor::TensorView<'_>,
    biases: &safetensors::tensor::TensorView<'_>,
    bits: usize,
    group_size: usize,
) -> Result<(usize, usize)> {
    if weight.dtype() != SafeDtype::U32 {
        return Err(Error::Msg(format!(
            "device-format sidecar: `{base}.weight` must be U32, got {:?}",
            weight.dtype()
        )));
    }
    let [out_dim, weight_cols] = weight.shape() else {
        return Err(Error::Msg(format!(
            "device-format sidecar: `{base}.weight` must be rank 2, got {:?}",
            weight.shape()
        )));
    };
    let [scale_rows, scale_cols] = scales.shape() else {
        return Err(Error::Msg(format!(
            "device-format sidecar: `{base}.scales` must be rank 2, got {:?}",
            scales.shape()
        )));
    };
    if biases.shape() != scales.shape() || scale_rows != out_dim {
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
    Ok((*out_dim, in_dim))
}

fn source_digest<'a>(
    base: &str,
    bits: usize,
    group_size: usize,
    views: [(&str, &safetensors::tensor::TensorView<'a>); 3],
) -> [u8; 32] {
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
        hash.update(view.data());
    }
    hash.finalize().into()
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
    bits: usize,
    group_size: usize,
    out_dim: usize,
    in_dim: usize,
    dtype: GgmlDType,
    device: &Device,
) -> Result<()> {
    let cpu = Device::Cpu;
    let wq = source.load(&format!("{base}.weight"), &cpu)?;
    let scales = source
        .load(&format!("{base}.scales"), &cpu)?
        .to_dtype(DType::F32)?;
    let biases = source
        .load(&format!("{base}.biases"), &cpu)?
        .to_dtype(DType::F32)?;
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
    metadata.insert("source_base".to_string(), base.to_string());
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
        if validate_sidecar(final_path, payload.len())? {
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
        Err(_rename_error) if validate_sidecar(final_path, payload.len())? => {
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

fn validate_sidecar(path: &Path, payload_bytes: usize) -> Result<bool> {
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
    let actual_hash: [u8; 32] = Sha256::digest(payload.data()).into();
    Ok(stored_hash.data() == actual_hash)
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
    use candle_core::Tensor;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            let seq = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "candle-device-sidecar-{label}-{}-{seq}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
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

    fn open(path: &Path) -> Result<MmapedSafetensors> {
        // SAFETY: immutable test fixture for the lifetime of the mapping.
        unsafe { MmapedSafetensors::new(path) }
    }

    fn assert_q4_or_q8(bits: usize) -> Result<()> {
        let dir = TestDir::new(if bits == 4 { "q4" } else { "q8" });
        let source_path = write_source(&dir.0, bits, 0.0)?;
        let source = open(&source_path)?;
        let packed = PackedConfig {
            bits: bits as i32,
            group_size: 64,
        };

        let cache = PackedWeightSidecars::prepare(&source, &dir.0, packed, &Device::Cpu)?;
        assert_eq!(cache.created_count(), 1);
        assert_eq!(cache.reused_count(), 0);
        assert!(cache.contains("layers.0.proj"));
        let sidecar_path = cache.path_for("layers.0.proj").unwrap();
        assert!(sidecar_path.is_file());
        assert!(sidecar_path.starts_with(dir.0.join(CACHE_DIR)));

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
        let reused = PackedWeightSidecars::prepare(&source, &dir.0, packed, &Device::Cpu)?;
        assert_eq!(reused.created_count(), 0);
        assert_eq!(reused.reused_count(), 1);
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
    fn source_byte_change_selects_a_different_content_address() -> Result<()> {
        let a = TestDir::new("hash-a");
        let b = TestDir::new("hash-b");
        let a_path = write_source(&a.0, 4, 0.0)?;
        let b_path = write_source(&b.0, 4, 0.125)?;
        let packed = PackedConfig {
            bits: 4,
            group_size: 64,
        };
        let a_source = open(&a_path)?;
        let b_source = open(&b_path)?;
        let a_cache = PackedWeightSidecars::prepare(&a_source, &a.0, packed, &Device::Cpu)?;
        let b_cache = PackedWeightSidecars::prepare(&b_source, &b.0, packed, &Device::Cpu)?;
        assert_ne!(
            a_cache.path_for("layers.0.proj").unwrap().file_name(),
            b_cache.path_for("layers.0.proj").unwrap().file_name(),
            "invalidation must follow source bytes, not a timestamp"
        );
        Ok(())
    }

    #[test]
    fn corrupt_sidecar_is_rebuilt_before_use() -> Result<()> {
        let dir = TestDir::new("corrupt");
        let source_path = write_source(&dir.0, 4, 0.0)?;
        let source = open(&source_path)?;
        let packed = PackedConfig {
            bits: 4,
            group_size: 64,
        };
        let first = PackedWeightSidecars::prepare(&source, &dir.0, packed, &Device::Cpu)?;
        let path = first.path_for("layers.0.proj").unwrap().to_path_buf();
        fs::write(&path, b"truncated generated cache")?;
        let rebuilt = PackedWeightSidecars::prepare(&source, &dir.0, packed, &Device::Cpu)?;
        assert_eq!(rebuilt.created_count(), 1);
        assert!(rebuilt.load("layers.0.proj", &Device::Cpu).is_ok());
        Ok(())
    }

    #[test]
    fn concurrent_corrupt_recovery_serializes_and_keeps_a_valid_artifact() -> Result<()> {
        let dir = TestDir::new("concurrent-corrupt");
        let source_path = write_source(&dir.0, 8, 0.0)?;
        let source = open(&source_path)?;
        let packed = PackedConfig {
            bits: 8,
            group_size: 64,
        };
        let first = PackedWeightSidecars::prepare(&source, &dir.0, packed, &Device::Cpu)?;
        let path = first.path_for("layers.0.proj").unwrap().to_path_buf();
        fs::write(&path, b"same corrupt entry observed by two preparers")?;

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let barrier = std::sync::Arc::clone(&barrier);
            let source_path = source_path.clone();
            let component_dir = dir.0.clone();
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
            payload_len(GgmlDType::Q8_0, 2, 64)?
        )?);
        Ok(())
    }
}
