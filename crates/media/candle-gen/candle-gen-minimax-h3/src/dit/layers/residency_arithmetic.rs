//! Ignored actual-source AdaLN/FFN preparation diagnostic. No production math changes.
use super::*;
use crate::denoise::{JointGeometry, JointSchedule, PackedLayout};
use crate::dit::adaln::{AdaLnCache, AdaLnResidency};
use crate::dit::block::{AdaLnModulation, DitBlock};
use crate::dit::heads::{AdaLayerNormOut, NormOutModulation, TimestepEmbedder};
use candle_gen::candle_core::{cuda::cudarc, Device};
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::quant::{QLinear, QuantWeight};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};
const ROWS: usize = 104_554;
const HIDDEN: usize = 5376;
const HEADS: usize = 56;
const HEAD_DIM: usize = 128;
const BUDGET: u64 = 32 * 1024 * 1024;
const MAX_READBACK_ELEMENTS: usize = 1024 * 1024;
const MODEL_REVISION: &str = "137ce668c55a20bc0935fd1cf2a3de8448abb7f4";
const TURBO_REVISION: &str = "5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1";
const EXPECTED_MANIFEST_SHA256: &str =
    "9df0a1203b8a20d87b81888cb4edeaa34537f50c558aba239a79af5b3c443b01";
const DEFAULT_MANIFEST: &str =
    "D:/Codex/work/sw-f329/envelope-outputs/residency-component-manifest.v1.json";
#[cfg(not(target_endian = "little"))]
compile_error!("H3 proof bytes require little endian");
fn error(value: impl std::fmt::Display) -> CandleError {
    CandleError::Msg(value.to_string())
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path).map_err(error)?;
    let mut hash = Sha256::new();
    let mut bytes = vec![0; 1024 * 1024];
    loop {
        let count = file.read(&mut bytes).map_err(error)?;
        if count == 0 {
            break;
        }
        hash.update(&bytes[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn update_bf16_hash(hash: &mut Sha256, values: &[bf16]) {
    // SAFETY: bf16 is repr(transparent) over u16 with no padding. Read only
    // initialized bytes within the slice's exact extent, on the asserted LE host.
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    hash.update(bytes);
}

fn readback_chunk(flat: &Tensor, start: usize, len: usize) -> Result<Vec<bf16>> {
    assert!(len > 0 && len <= MAX_READBACK_ELEMENTS);
    // CUDA to_vec1 copies the WHOLE backing storage before applying the layout.
    // A contiguous narrow can still alias the full projection. Force a new,
    // chunk-sized allocation; contiguous() and copy() do not establish this bound.
    let chunk = flat.narrow(0, start, len)?.force_contiguous()?;
    assert_eq!(chunk.elem_count(), len);
    {
        let (storage, layout) = chunk.storage_and_layout();
        assert_eq!(layout.start_offset(), 0);
        let storage_elements = match &*storage {
            candle_gen::candle_core::Storage::Cpu(candle_gen::candle_core::CpuStorage::BF16(
                values,
            )) => values.len(),
            candle_gen::candle_core::Storage::Cuda(storage) => {
                storage.as_cuda_slice::<bf16>()?.len()
            }
            _ => return Err(error("BF16 readback requires CPU or CUDA storage")),
        };
        assert_eq!(storage_elements, len, "readback must own only its chunk");
    }
    let values = chunk.to_vec1::<bf16>()?;
    assert_eq!(values.len(), len);
    Ok(values)
}

fn identity_bf16(tensor: &Tensor) -> Result<Value> {
    assert_eq!(tensor.dtype(), DType::BF16);
    let flat = tensor.flatten_all()?;
    let mut hash = Sha256::new();
    for start in (0..tensor.elem_count()).step_by(MAX_READBACK_ELEMENTS) {
        let len = MAX_READBACK_ELEMENTS.min(tensor.elem_count() - start);
        update_bf16_hash(&mut hash, &readback_chunk(&flat, start, len)?);
    }
    Ok(json!({"shape":tensor.dims(),"dtype":"BF16","sha256":format!("{:x}",hash.finalize())}))
}

// Readback is bounded, and metrics cover EVERY element, including the ragged tail.
// Hash the original BF16 bits, not a rounded image or an F32 serialization.
fn comparison_bf16(left: &Tensor, right: &Tensor) -> Result<Value> {
    assert_eq!(left.dims(), right.dims());
    assert_eq!(left.dtype(), DType::BF16);
    assert_eq!(right.dtype(), DType::BF16);
    let left_flat = left.flatten_all()?;
    let right_flat = right.flatten_all()?;
    let mut lh = Sha256::new();
    let mut rh = Sha256::new();
    let mut max_abs = 0f64;
    let mut peak_bits = 0u16;
    let mut squares = 0f64;
    let mut nonfinite = 0u64;
    for start in (0..left.elem_count()).step_by(MAX_READBACK_ELEMENTS) {
        let len = MAX_READBACK_ELEMENTS.min(left.elem_count() - start);
        let a = readback_chunk(&left_flat, start, len)?;
        let b = readback_chunk(&right_flat, start, len)?;
        update_bf16_hash(&mut lh, &a);
        update_bf16_hash(&mut rh, &b);
        for (a, b) in a.into_iter().zip(b) {
            let (a_bits, b_bits) = (a.to_bits(), b.to_bits());
            // Positive finite BF16 magnitudes preserve unsigned-bit ordering.
            // Avoid billions of redundant F64 operations on byte-identical values.
            if a_bits & 0x7fff >= 0x7f80 || b_bits & 0x7fff >= 0x7f80 {
                nonfinite += 1;
                continue;
            }
            peak_bits = peak_bits.max(a_bits & 0x7fff);
            if a_bits == b_bits {
                continue;
            }
            let (a, b) = (f64::from(a.to_f32()), f64::from(b.to_f32()));
            let difference = (a - b).abs();
            max_abs = max_abs.max(difference);
            squares += difference * difference;
        }
    }
    let left_hash = format!("{:x}", lh.finalize());
    let right_hash = format!("{:x}", rh.finalize());
    let peak = f64::from(bf16::from_bits(peak_bits).to_f32());
    Ok(json!({
        "shape":left.dims(), "dtype":"BF16", "elements":left.elem_count(),
        "leftSha256":left_hash, "rightSha256":right_hash,
        "exact":left_hash == right_hash && nonfinite == 0,
        "maxAbs":max_abs, "referencePeak":peak,
        "relativePeak":if peak == 0.0 { max_abs } else { max_abs / peak },
        "rms":(squares / left.elem_count() as f64).sqrt(), "nonfinite":nonfinite,
    }))
}

fn packed_hash(layer: &LinearNoBias) -> Result<String> {
    match layer.inner.base_qlinear() {
        Some(QLinear::Quantized {
            weight: QuantWeight::Dequant(weight),
            bias: None,
            ..
        }) => {
            assert_eq!(format!("{:?}", weight.dtype()), "Q4_1");
            Ok(format!("{:x}", Sha256::digest(weight.data()?.as_ref())))
        }
        other => Err(error(format!(
            "required actual Q4_1/DequantDense, got {other:?}"
        ))),
    }
}

// SetStream resets the handle's DEFAULT workspace even if the stream is unchanged.
// Keep explicit storage alive through synchronization and reset, also on unwind.
// Never install a proof pool: its per-GEMM automatic binding would invalidate controls.
struct WorkspaceControl {
    stream: Arc<CudaStream>,
    blas: Arc<cudarc::cublas::CudaBlas>,
    storage: Option<CudaSlice<u8>>,
}

impl WorkspaceControl {
    fn reset(&self) -> Result<()> {
        self.stream.synchronize().map_err(error)?;
        // SAFETY: this test owns the live handle; every operation uses this same stream.
        unsafe {
            cudarc::cublas::sys::cublasSetStream_v2(
                *self.blas.handle(),
                self.stream.cu_stream().cast(),
            )
            .result()
        }
        .map_err(error)
    }
}

impl Drop for WorkspaceControl {
    fn drop(&mut self) {
        // Explicit finish propagates errors; this is the unwind/error safety backstop.
        // If reset fails, retain the buffer rather than free a possibly live workspace.
        if self.reset().is_err() {
            if let Some(storage) = self.storage.take() {
                std::mem::forget(storage);
            }
        }
    }
}

fn with_workspace<T>(
    device: &Device,
    user_bytes: Option<usize>,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let cuda = device.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    let blas = cuda.cublas_handle();
    let mut selected = std::ptr::null_mut();
    // SAFETY: handle and out pointer are live; no other thread uses this test's device.
    unsafe { cudarc::cublas::sys::cublasGetStream_v2(*blas.handle(), &mut selected).result() }
        .map_err(error)?;
    assert!(std::ptr::eq(
        selected.cast::<std::ffi::c_void>(),
        stream.cu_stream().cast::<std::ffi::c_void>()
    ));
    let mut guard = WorkspaceControl {
        storage: user_bytes
            .map(|bytes| stream.alloc_zeros::<u8>(bytes).map_err(error))
            .transpose()?,
        stream,
        blas,
    };
    guard.reset()?;
    if let Some(storage) = &mut guard.storage {
        let bytes = storage.len();
        let (pointer, _usage) = storage.device_ptr_mut(&guard.stream);
        assert_eq!(pointer % 256, 0, "cuBLAS workspace alignment");
        // SAFETY: owned storage remains live until completion and default restoration.
        unsafe {
            cudarc::cublas::sys::cublasSetWorkspace_v2(
                *guard.blas.handle(),
                pointer as *mut _,
                bytes,
            )
            .result()
        }
        .map_err(error)?;
    }
    let output = operation();
    guard.reset()?;
    output
}

fn update_f32_hash(hash: &mut Sha256, values: &[f32]) {
    // SAFETY: f32 has no padding. Read only
    // initialized bytes within the slice's exact extent, on the asserted LE host.
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    hash.update(bytes);
}

fn readback_f32(flat: &Tensor, start: usize, len: usize) -> Result<Vec<f32>> {
    assert!(len > 0 && len <= MAX_READBACK_ELEMENTS);
    // CUDA to_vec1 copies the WHOLE backing storage before applying the layout.
    // A contiguous narrow can still alias the full projection. Force a new,
    // chunk-sized allocation; contiguous() and copy() do not establish this bound.
    let chunk = flat.narrow(0, start, len)?.force_contiguous()?;
    assert_eq!(chunk.elem_count(), len);
    {
        let (storage, layout) = chunk.storage_and_layout();
        assert_eq!(layout.start_offset(), 0);
        let storage_elements = match &*storage {
            candle_gen::candle_core::Storage::Cpu(candle_gen::candle_core::CpuStorage::F32(
                values,
            )) => values.len(),
            candle_gen::candle_core::Storage::Cuda(storage) => {
                storage.as_cuda_slice::<f32>()?.len()
            }
            _ => return Err(error("F32 readback requires CPU or CUDA storage")),
        };
        assert_eq!(storage_elements, len, "readback must own only its chunk");
    }
    let values = chunk.to_vec1::<f32>()?;
    assert_eq!(values.len(), len);
    Ok(values)
}

fn identity_f32(tensor: &Tensor) -> Result<Value> {
    assert_eq!(tensor.dtype(), DType::F32);
    let flat = tensor.flatten_all()?;
    let mut hash = Sha256::new();
    for start in (0..tensor.elem_count()).step_by(MAX_READBACK_ELEMENTS) {
        let len = MAX_READBACK_ELEMENTS.min(tensor.elem_count() - start);
        update_f32_hash(&mut hash, &readback_f32(&flat, start, len)?);
    }
    Ok(json!({"shape":tensor.dims(),"dtype":"F32","sha256":format!("{:x}",hash.finalize())}))
}

// Readback is bounded, and metrics cover EVERY element, including the ragged tail.
// Hash the original F32 bits without sampling or serialization rounding.
fn comparison_f32(left: &Tensor, right: &Tensor) -> Result<Value> {
    assert_eq!(left.dims(), right.dims());
    assert_eq!(left.dtype(), DType::F32);
    assert_eq!(right.dtype(), DType::F32);
    let left_flat = left.flatten_all()?;
    let right_flat = right.flatten_all()?;
    let mut lh = Sha256::new();
    let mut rh = Sha256::new();
    let mut max_abs = 0f64;
    let mut peak_bits = 0u32;
    let mut squares = 0f64;
    let mut nonfinite = 0u64;
    for start in (0..left.elem_count()).step_by(MAX_READBACK_ELEMENTS) {
        let len = MAX_READBACK_ELEMENTS.min(left.elem_count() - start);
        let a = readback_f32(&left_flat, start, len)?;
        let b = readback_f32(&right_flat, start, len)?;
        update_f32_hash(&mut lh, &a);
        update_f32_hash(&mut rh, &b);
        for (a, b) in a.into_iter().zip(b) {
            let (a_bits, b_bits) = (a.to_bits(), b.to_bits());
            // Positive finite F32 magnitudes preserve unsigned-bit ordering.
            // Avoid billions of redundant F64 operations on byte-identical values.
            if a_bits & 0x7fff_ffff >= 0x7f80_0000 || b_bits & 0x7fff_ffff >= 0x7f80_0000 {
                nonfinite += 1;
                continue;
            }
            peak_bits = peak_bits.max(a_bits & 0x7fff_ffff);
            if a_bits == b_bits {
                continue;
            }
            let (a, b) = (f64::from(a), f64::from(b));
            let difference = (a - b).abs();
            max_abs = max_abs.max(difference);
            squares += difference * difference;
        }
    }
    let left_hash = format!("{:x}", lh.finalize());
    let right_hash = format!("{:x}", rh.finalize());
    let peak = f64::from(f32::from_bits(peak_bits));
    Ok(json!({
        "shape":left.dims(), "dtype":"F32", "elements":left.elem_count(),
        "leftSha256":left_hash, "rightSha256":right_hash,
        "exact":left_hash == right_hash && nonfinite == 0,
        "maxAbs":max_abs, "referencePeak":peak,
        "relativePeak":if peak == 0.0 { max_abs } else { max_abs / peak },
        "rms":(squares / left.elem_count() as f64).sqrt(), "nonfinite":nonfinite,
    }))
}

fn required_path(manifest: &Value, key: &str) -> Result<PathBuf> {
    let path = PathBuf::from(manifest[key].as_str().ok_or_else(|| error(key))?);
    if !path.is_absolute() || !path.exists() {
        return Err(error(format!(
            "missing required absolute {key}: {}",
            path.display()
        )));
    }
    Ok(path)
}

// These are fixture-verifier observations, never stronger claims about the
// underlying reader APIs. Call only after verification AND successful consumption.
fn record_verified_content(
    reader: &str,
    role: candle_gen::gen_core::model_input_proof::InputRole,
    assets: &[&Value],
    manifest_sha: &str,
) -> Result<()> {
    use candle_gen::gen_core::model_input_proof::{record_inputs, sha256_from_hex, InputIdentity};
    let paths = assets
        .iter()
        .map(|asset| {
            asset["path"]
                .as_str()
                .map(PathBuf::from)
                .ok_or_else(|| error("verified asset path missing"))
        })
        .collect::<Result<Vec<_>>>()?;
    let hashes = assets
        .iter()
        .map(|asset| {
            asset["sha256"]
                .as_str()
                .and_then(sha256_from_hex)
                .ok_or_else(|| error("verified asset digest missing"))
        })
        .collect::<Result<Vec<_>>>()?;
    let binding = format!("frozen manifest SHA256={manifest_sha}; actual bytes matched accepted receipt before successful consumption; not same-read-buffer, post-read or opened-handle/ABA attestation");
    let items = paths
        .iter()
        .zip(hashes)
        .map(|(path, hash)| {
            InputIdentity::validated(
                path,
                hash,
                "H3 fixture verify_asset + frozen manifest/accepted receipt SHA256 checks",
                &binding,
            )
        })
        .collect::<Vec<_>>();
    record_inputs(reader, role, &items);
    Ok(())
}

fn verify_asset(
    path: &Path,
    expected: &Value,
    receipt: &Value,
    repo: &str,
    revision: &str,
    relative: &str,
) -> Result<Value> {
    let expected_hash = expected["sha256"]
        .as_str()
        .ok_or_else(|| error("missing expected asset digest"))?;
    let expected_bytes = expected["bytes"]
        .as_u64()
        .ok_or_else(|| error("missing expected asset length"))?;
    let actual_path = path.canonicalize().map_err(error)?;
    let matches = receipt["files"]
        .as_array()
        .ok_or_else(|| error("missing accepted file records"))?
        .iter()
        .filter(|entry| {
            entry["repo"] == repo && entry["revision"] == revision && entry["relative"] == relative
        })
        .collect::<Vec<_>>();
    assert_eq!(
        matches.len(),
        1,
        "asset must have exactly one immutable accepted source binding"
    );
    let binding = matches[0];
    assert_eq!(binding["sha256"], expected_hash);
    assert_eq!(binding["bytes"], expected_bytes);
    let bound_path = Path::new(
        binding["path"]
            .as_str()
            .ok_or_else(|| error("missing accepted path"))?,
    )
    .canonicalize()
    .map_err(error)?;
    assert_eq!(
        actual_path, bound_path,
        "asset path differs from accepted source mapping"
    );
    assert_eq!(
        std::fs::metadata(path).map_err(error)?.len(),
        expected_bytes
    );
    let actual_hash = hash_file(path)?;
    assert_eq!(
        actual_hash,
        expected_hash,
        "source asset digest mismatch: {}",
        path.display()
    );
    Ok(
        json!({"path":actual_path,"repo":repo,"revision":revision,"relative":relative,
        "bytes":expected_bytes,"sha256":actual_hash}),
    )
}

struct FixtureInputs {
    directory: PathBuf,
    adapter_path: PathBuf,
    files: Vec<PathBuf>,
    cfg: MiniMaxH3DitConfig,
    verified: Vec<Value>,
    manifest: Value,
}

fn verified_fixture() -> Result<FixtureInputs> {
    assert!(
        std::env::var_os("SCENEWORKS_CUDA_PROOF_POOL_BYTES").is_none(),
        "workspace contrast requires NO proof-pool auto-binding; never silently override a cap"
    );
    let manifest_path = std::env::var_os("H3_RESIDENCY_ARITHMETIC_MANIFEST")
        .map_or_else(|| PathBuf::from(DEFAULT_MANIFEST), PathBuf::from);
    let manifest_bytes = std::fs::read(&manifest_path).map_err(error)?;
    let manifest_hash = format!("{:x}", Sha256::digest(&manifest_bytes));
    assert_eq!(
        manifest_hash, EXPECTED_MANIFEST_SHA256,
        "independently frozen fixture manifest required"
    );
    let manifest: Value = serde_json::from_slice(&manifest_bytes).map_err(error)?;
    assert_eq!(manifest["schemaVersion"], 1);
    assert_eq!(manifest["modelRevision"], MODEL_REVISION);
    assert_eq!(manifest["turboRevision"], TURBO_REVISION);
    let directory = required_path(&manifest, "transformerDir")?;
    let adapter_path = required_path(&manifest, "turboPath")?;
    assert_eq!(
        adapter_path.file_name().and_then(|p| p.to_str()),
        Some("minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors")
    );
    let config_path = directory.join("config.json");
    let receipt_path = required_path(&manifest["acceptedStagingReceipt"], "path")?;
    let receipt_bytes = std::fs::read(&receipt_path).map_err(error)?;
    assert_eq!(
        format!("{:x}", Sha256::digest(&receipt_bytes)),
        manifest["acceptedStagingReceipt"]["sha256"]
    );
    let receipt: Value = serde_json::from_slice(&receipt_bytes).map_err(error)?;
    assert_eq!(receipt["kind"], "sw-f329-staged-models");
    let files = candle_gen::loader::sorted_safetensors(&directory, "H3 component witness")?;
    let expected_shards = manifest["expectedShards"]
        .as_array()
        .ok_or_else(|| error("missing expected shard list"))?;
    assert_eq!(files.len(), expected_shards.len());
    assert_eq!(files.len(), 14);
    let mut verified_sources = vec![
        verify_asset(
            &config_path,
            &manifest["expectedConfig"],
            &receipt,
            "SceneWorks/minimax-h3-mlx",
            MODEL_REVISION,
            "q4/transformer/config.json",
        )?,
        verify_asset(
            &adapter_path,
            &manifest["expectedTurbo"],
            &receipt,
            "lightx2v/Minimax-h3-Turbo",
            TURBO_REVISION,
            "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
        )?,
    ];
    for (actual, expected) in files.iter().zip(expected_shards) {
        let name = expected["name"]
            .as_str()
            .ok_or_else(|| error("missing expected shard name"))?;
        assert_eq!(actual.file_name().and_then(|v| v.to_str()), Some(name));
        verified_sources.push(verify_asset(
            actual,
            expected,
            &receipt,
            "SceneWorks/minimax-h3-mlx",
            MODEL_REVISION,
            &format!("q4/transformer/{name}"),
        )?);
    }
    let config_bytes = std::fs::read(&config_path).map_err(error)?;
    let config = MiniMaxH3DitConfig::from_diffusers_json(
        std::str::from_utf8(&config_bytes).map_err(error)?,
    )?;
    assert_eq!(
        (
            config.hidden_size,
            config.num_attention_heads,
            config.attention_head_dim,
            config.num_layers,
            config.rope_freq_dim
        ),
        (HIDDEN, HEADS, HEAD_DIM, 50, 16)
    );

    Ok(FixtureInputs {
        directory,
        adapter_path,
        files,
        cfg: config,
        verified: verified_sources,
        manifest,
    })
}

fn metric(left: &Tensor, right: &Tensor) -> Result<Value> {
    assert_eq!(left.dtype(), right.dtype());
    match left.dtype() {
        DType::BF16 => comparison_bf16(left, right),
        DType::F32 => comparison_f32(left, right),
        dtype => Err(error(format!("unsupported diagnostic dtype {dtype:?}"))),
    }
}

fn identity(tensor: &Tensor) -> Result<Value> {
    match tensor.dtype() {
        DType::BF16 => identity_bf16(tensor),
        DType::F32 => identity_f32(tensor),
        dtype => Err(error(format!("unsupported diagnostic dtype {dtype:?}"))),
    }
}

fn selected(name: &str) -> bool {
    name.starts_with("transformer_blocks.0.")
        || name.starts_with("time_embedder.")
        || name.starts_with("norm_out.")
}

fn native_selected(files: &[PathBuf], device: &Device) -> Result<Weights> {
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for file in files {
        let header = candle_gen::loader::mmap_safetensors(file)?;
        for (name, view) in header.tensors() {
            if selected(&name) {
                let dtype = format!("{:?}", view.dtype());
                assert!(matches!(dtype.as_str(), "BF16" | "F32" | "U32"));
                // Integer tensors preserve their dtype in the filtered reader.
                let group = if dtype == "F32" { "F32" } else { "BF16" };
                groups.entry(group.into()).or_default().push(name);
            }
        }
    }
    assert_eq!(
        groups.len(),
        2,
        "native F32 timestep roots must not be coerced to BF16"
    );
    let mut map = HashMap::new();
    for (group, names) in groups {
        let dtype = if group == "F32" {
            DType::F32
        } else {
            DType::BF16
        };
        let prefixes = names.iter().map(String::as_str).collect::<Vec<_>>();
        let weights = Weights::from_files_filtered(files, device, dtype, &prefixes)?;
        let mut got = weights.keys().cloned().collect::<Vec<_>>();
        let mut expected = names.clone();
        got.sort();
        expected.sort();
        assert_eq!(got, expected, "prefix overlap changed native source set");
        for name in names {
            assert!(map.insert(name.clone(), weights.require(&name)?).is_none());
        }
    }
    let adaln = map
        .get("transformer_blocks.0.adaln_proj.linear.weight")
        .ok_or_else(|| error("stock AdaLN missing"))?;
    assert_eq!(adaln.dims(), [96768, 336]);
    assert_eq!(adaln.dtype(), DType::U32);
    let bias = map
        .get("transformer_blocks.0.adaln_proj.linear.bias")
        .ok_or_else(|| error("stock AdaLN bias missing"))?;
    assert_eq!(bias.dims(), [96768]);
    assert_eq!(bias.dtype(), DType::BF16);
    assert!(!map.contains_key("adaln_t_table"));
    Ok(Weights::from_map(map))
}

fn checked_sum(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(0u64, |sum, value| {
        sum.checked_add(*value)
            .ok_or_else(|| error("diagnostic planning overflow"))
    })
}
fn times(value: u64, count: u64) -> Result<u64> {
    value
        .checked_mul(count)
        .ok_or_else(|| error("diagnostic planning overflow"))
}

fn planning(directory: &Path, cfg: &MiniMaxH3DitConfig, manifest: &Value) -> Result<Value> {
    let (payloads, host_preparation, decode) =
        crate::memory_strategy::proof_dit_inventory(directory).map_err(error)?;
    let headers = candle_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(directory)
        .map_err(error)?;
    assert!(
        !headers.iter().any(|h| h.name == "adaln_t_table"),
        "stock unpruned source required"
    );
    let source = checked_sum(
        &headers
            .iter()
            .filter(|h| selected(&h.name))
            .map(|h| h.data_bytes)
            .collect::<Vec<_>>(),
    )?;
    let runtime = checked_sum(
        &payloads
            .iter()
            .filter(|(name, _)| selected(name))
            .map(|(_, bytes)| *bytes)
            .collect::<Vec<_>>(),
    )?;
    let adapter = manifest["expectedTurbo"]["bytes"]
        .as_u64()
        .ok_or_else(|| error("adapter byte bound missing"))?;
    let hidden = times(times(ROWS as u64, cfg.hidden_size as u64)?, 2)?;
    let width = times(cfg.ffn_dim as u64, 2)?;
    let block = ffn_token_block(
        1,
        width as usize,
        ROWS,
        AttentionBudget::from_score_elements(BUDGET, false),
    );
    assert_eq!((block, ROWS / block, ROWS % block), (1170, 89, 424));
    // Source-owned native runtime/host/dequant facts plus this diagnostic's live
    // coexistence. Ten full BF16 streams cover input, three production outputs,
    // trace/readback and cat lifetimes; tile intermediates are separately charged.
    // This is a conservative fixture bound, not a new product fit estimator.
    let tile = times(times(block as u64, width)?, 2)?;
    let affine = times(times(96768, 2688)?, 6)?; // F32 dequant + BF16 cast.
    let rotary = times(times(times(ROWS as u64, cfg.rotary_dim() as u64)?, 4)?, 6)?;
    let device = checked_sum(&[
        times(source, 3)?,
        times(runtime, 3)?,
        times(decode, 2)?,
        affine,
        times(hidden, 10)?,
        times(tile, 8)?,
        times(adapter, 4)?,
        rotary,
        BUDGET,
        times(MAX_READBACK_ELEMENTS as u64, 8)?,
    ])?;
    let host = checked_sum(&[
        times(host_preparation, 2)?,
        times(adapter, 4)?,
        times(hidden, 2)?,
        rotary,
        times(MAX_READBACK_ELEMENTS as u64, 8)?,
    ])?;
    assert!(
        device <= 32u64 << 30,
        "source-derived diagnostic GPU plan exceeds32GiB"
    );
    assert!(
        host <= 128u64 << 30,
        "source-derived diagnostic host plan exceeds128GiB"
    );
    Ok(
        json!({"devicePeakBytes":device,"hostPeakBytes":host,"deviceLimitBytes":32u64<<30,"hostLimitBytes":128u64<<30,"selectedSourceBytes":source,"selectedRuntimeBytes":runtime,"nativeHostPreparationBytes":host_preparation,"nativeProjectionDecodeBytes":decode,"adaLnAffineScratchBytes":affine,"hiddenStreamBytes":hidden,"boundedTileBytes":tile,"chunkRows":block,"fullChunks":89,"tailRows":424,"scope":"header-only source facts + conservative diagnostic coexistence, not observed/product memory"}),
    )
}

fn ffn_from_block(block: &mut DitBlock) -> Result<DitFeedForward> {
    let proj = block
        .adaptable_mut(&["ff", "net", "0", "proj"])
        .ok_or_else(|| error("missing FFN projection"))?
        .clone();
    let out = block
        .adaptable_mut(&["ff", "net", "2"])
        .ok_or_else(|| error("missing FFN output"))?
        .clone();
    Ok(DitFeedForward { proj, out })
}

fn install_canonical_ffn(
    host: &mut DitBlock,
    direct: &mut DitBlock,
    device: &Device,
) -> Result<()> {
    for path in [["ff", "net", "0", "proj"], ["ff", "net", "2", ""]] {
        let path = if path[3].is_empty() {
            &path[..3]
        } else {
            &path[..]
        };
        let source = host
            .adaptable_mut(path)
            .ok_or_else(|| error("missing canonical FFN"))?;
        assert_eq!(source.adapters.len(), 1);
        assert!(source.lokr_adapters.is_empty());
        let target = direct
            .adaptable_mut(path)
            .ok_or_else(|| error("missing direct FFN"))?;
        for factor in &source.adapters {
            target.push_lora(
                factor.a.to_device(device)?,
                factor.b.to_device(device)?,
                factor.scale,
            )?;
        }
    }
    Ok(())
}

fn preparation(
    host: &DitFeedForward,
    direct: &DitFeedForward,
    copied: &DitFeedForward,
) -> Result<Value> {
    let mut projections = Vec::new();
    for (host, direct, copied) in [
        (&host.proj, &direct.proj, &copied.proj),
        (&host.out, &direct.out, &copied.out),
    ] {
        let hashes = [
            packed_hash(host)?,
            packed_hash(direct)?,
            packed_hash(copied)?,
        ];
        let hf = host.inner.__model_proof_frozen_lora_factors()?;
        let df = direct.inner.__model_proof_frozen_lora_factors()?;
        let cf = copied.inner.__model_proof_frozen_lora_factors()?;
        assert_eq!((hf.len(), df.len(), cf.len()), (1, 1, 1));
        let mut factors = Vec::new();
        for ((h, d), c) in hf.iter().zip(&df).zip(&cf) {
            assert_eq!(h.2, d.2);
            assert_eq!(h.2, c.2);
            factors.push(json!({"hostVsDirectA":metric(&h.0,&d.0)?,"hostVsDirectB":metric(&h.1,&d.1)?,"hostVsCopiedA":metric(&h.0,&c.0)?,"hostVsCopiedB":metric(&h.1,&c.1)?,"hostScale":h.2,"directScale":d.2,"copiedScale":c.2,"strength":1.0}));
        }
        projections.push(json!({"packedHashes":hashes,"exact":hashes[0]==hashes[1]&&hashes[0]==hashes[2],"factors":factors}));
    }
    Ok(json!(projections))
}

struct ArithmeticStage {
    time_project: Tensor,
    temb: Tensor,
    activated: Tensor,
    raw: Tensor,
    modulation: AdaLnModulation,
    norm: NormOutModulation,
    tables: MmRopeTables,
    ff: DitFeedForward,
    cache_rows: usize,
    eviction_bytes: usize,
}

fn observed_adaln(modulation: &AdaLnModulation, temb: &Tensor) -> Result<(Tensor, Tensor)> {
    // Recover the actual projected bits by inverting ONLY forward's layout
    // split. Do not run an extra projection GEMM or warm its dequant cache.
    let parts = modulation.tables().collect::<Vec<_>>();
    let raw = Tensor::cat(&parts, 1)?.reshape((temb.dims()[0], 96768))?;
    // Stateless replay after the production call; this is not an exposed handle
    // to forward's internal activated tensor and cannot prewarm its GEMM.
    let activated = silu(temb)?.to_dtype(DType::BF16)?;
    Ok((activated, raw))
}

fn direct_stage(
    files: &[PathBuf],
    cfg: &MiniMaxH3DitConfig,
    host: &mut DitBlock,
    schedule: &crate::dit::adaln::TimestepSchedule,
    layout: &PackedLayout,
    device: &Device,
) -> Result<ArithmeticStage> {
    let weights = native_selected(files, device)?;
    let mut block = DitBlock::from_weights(&weights, "transformer_blocks.0", cfg, DType::BF16)?;
    install_canonical_ffn(host, &mut block, device)?;
    let time = TimestepEmbedder::from_weights(&weights, "time_embedder", cfg)?;
    let norm = AdaLayerNormOut::from_weights(&weights, "norm_out", cfg)?;
    drop(weights); // No loader map or projection clones survive eviction.
    let rope = MmRope::new(cfg.rope_freq_dim, cfg.rope_theta)?;
    let tables = rope.tables(layout.position_ids())?;
    let mut blocks = vec![block];
    let mut captured = None;
    let (cache, eviction_bytes) = AdaLnCache::precompute_and_evict(
        &mut blocks,
        schedule.clone(),
        AdaLnResidency::PrecomputeAndEvict,
        device,
        |ts| {
            let temb = time.forward(ts, device)?;
            captured = Some(temb.clone());
            Ok(temb)
        },
    )?;
    let temb = captured.ok_or_else(|| error("production precompute did not embed schedule"))?;
    // The resident production branch projects norm_out AFTER AdaLN eviction.
    let norm = norm.modulation(&temb)?;
    assert!(!blocks[0].holds_adaln());
    assert!(eviction_bytes > 0);
    let modulation = cache.modulation(0)?.clone();
    let (activated, raw) = observed_adaln(&modulation, &temb)?;
    let time_project =
        crate::dit::heads::timestep_sincos(schedule.distinct_timesteps(), cfg.freq_dim, device)?;
    let ff = ffn_from_block(&mut blocks[0])?;
    Ok(ArithmeticStage {
        time_project,
        temb,
        activated,
        raw,
        modulation,
        norm,
        tables,
        ff,
        cache_rows: cache.schedule().num_distinct_timesteps(),
        eviction_bytes,
    })
}

fn copied_stage(
    mut host: DitBlock,
    roots: (&TimestepEmbedder, &AdaLayerNormOut, &MmRope),
    cfg: &MiniMaxH3DitConfig,
    schedule: &crate::dit::adaln::TimestepSchedule,
    layout: &PackedLayout,
    device: &Device,
) -> Result<ArithmeticStage> {
    let (time, norm, rope) = roots;
    let time = time.copy_to_device(device)?;
    let norm = norm.copy_to_device(device)?;
    let tables = rope.tables(layout.position_ids())?;
    let temb = time.forward(schedule.distinct_timesteps(), device)?;
    // Preserve the actual opposite root order; do not normalize it to resident.
    let norm = norm.modulation(&temb)?;
    // Exact stock window precompute seam: only AdaLN moves, then CPU projection
    // eviction precedes body transfer. No curve or shared-table replacement.
    let modulation = host.modulation_for_window(&temb, device)?;
    let cache = AdaLnCache::assemble(schedule.clone(), vec![modulation], None)?;
    let eviction_bytes = host
        .evict_adaln()
        .ok_or_else(|| error("host AdaLN eviction missing"))?
        .nbytes();
    assert!(!host.holds_adaln());
    let mut body = host.copy_to_device(device)?;
    let ff = ffn_from_block(&mut body)?;
    let (activated, raw) = observed_adaln(cache.modulation(0)?, &temb)?;
    let time_project =
        crate::dit::heads::timestep_sincos(schedule.distinct_timesteps(), cfg.freq_dim, device)?;
    Ok(ArithmeticStage {
        time_project,
        temb,
        activated,
        raw,
        modulation: cache.modulation(0)?.clone(),
        norm,
        tables,
        ff,
        cache_rows: cache.schedule().num_distinct_timesteps(),
        eviction_bytes,
    })
}

fn stage_metrics(left: &ArithmeticStage, right: &ArithmeticStage) -> Result<Value> {
    let tables = left
        .modulation
        .tables()
        .zip(right.modulation.tables())
        .map(|(a, b)| metric(a, b))
        .collect::<Result<Vec<_>>>()?;
    Ok(
        json!({"timeProject":metric(&left.time_project,&right.time_project)?,"timestepMlp":metric(&left.temb,&right.temb)?,"adaLnActivated":metric(&left.activated,&right.activated)?,"adaLnRawProjection":metric(&left.raw,&right.raw)?,"modulationTables":tables,"normOutShift":metric(&left.norm.shift,&right.norm.shift)?,"normOutScale":metric(&left.norm.scale,&right.norm.scale)?,"rotaryCos":metric(&left.tables.cos,&right.tables.cos)?,"rotarySin":metric(&left.tables.sin,&right.tables.sin)?,"cacheDistinctRows":[left.cache_rows,right.cache_rows],"evictedProjectionPayloadBytes":[left.eviction_bytes,right.eviction_bytes],"evictionByteScope":"held payload arithmetic, not measured device-memory recovery"}),
    )
}

fn traced_ffn(
    direct: &DitFeedForward,
    copied: &DitFeedForward,
    input: &Tensor,
    direct_output: &Tensor,
    copied_output: &Tensor,
) -> Result<Value> {
    let mut rows = Vec::new();
    for start in (0..ROWS).step_by(1170) {
        let len = 1170.min(ROWS - start);
        let x = input.narrow(1, start, len)?.contiguous()?;
        // Exactly forward_span's operator sequence, with boundary readbacks.
        // Compare each traced result to the original production forward below.
        let a = direct.proj.forward(&x)?;
        let b = copied.proj.forward(&x)?;
        let projection = metric(&a, &b)?;
        let (ag, av) = split_gate_value(&a)?;
        let (bg, bv) = split_gate_value(&b)?;
        let aa = silu(&ag)?.mul(&av)?;
        let ba = silu(&bg)?.mul(&bv)?;
        let activation = metric(&aa, &ba)?;
        let ao = direct.out.forward(&aa)?;
        let bo = copied.out.forward(&ba)?;
        rows.push(json!({"start":start,"rows":len,"projection":projection,"activation":activation,"output":metric(&ao,&bo)?,"directProduction":metric(&ao,&direct_output.narrow(1,start,len)?)?,"copiedProduction":metric(&bo,&copied_output.narrow(1,start,len)?)?}));
    }
    assert_eq!(rows.len(), 90);
    assert_eq!(rows.last().unwrap()["rows"], 424);
    Ok(json!(rows))
}

fn differences(value: &Value, path: &str, out: &mut Vec<String>) {
    if value.get("exact") == Some(&Value::Bool(false)) {
        out.push(path.to_owned());
    }
    match value {
        Value::Object(items) => {
            for (key, value) in items {
                differences(value, &format!("{path}/{key}"), out)
            }
        }
        Value::Array(items) => {
            for (index, value) in items.iter().enumerate() {
                differences(value, &format!("{path}/{index}"), out)
            }
        }
        _ => {}
    }
}

fn ordered_differences(same: &Value, out: &mut Vec<String>) {
    differences(&same["preparation"], "samePolicy/preparation", out);
    differences(
        &same["roots"]["independentPackedPositionIds"],
        "samePolicy/roots/independentPackedPositionIds",
        out,
    );
    for key in [
        "rotaryCos",
        "rotarySin",
        "timeProject",
        "timestepMlp",
        "adaLnActivated",
        "adaLnRawProjection",
        "modulationTables",
        "normOutShift",
        "normOutScale",
    ] {
        differences(
            &same["roots"]["arithmetic"][key],
            &format!("samePolicy/roots/arithmetic/{key}"),
            out,
        );
    }
    if let Some(rows) = same["ffn"]["boundaries"].as_array() {
        for (index, row) in rows.iter().enumerate() {
            for key in [
                "projection",
                "activation",
                "output",
                "directProduction",
                "copiedProduction",
            ] {
                differences(
                    &row[key],
                    &format!("samePolicy/ffn/boundaries/{index}/{key}"),
                    out,
                );
            }
        }
    }
    differences(&same["ffn"]["fullOutput"], "samePolicy/ffn/fullOutput", out);
}

fn nonfinite_paths(value: &Value, path: &str, out: &mut Vec<String>) {
    if value
        .get("nonfinite")
        .and_then(Value::as_u64)
        .is_some_and(|count| count > 0)
    {
        out.push(path.to_owned());
    }
    match value {
        Value::Object(items) => {
            for (key, value) in items {
                nonfinite_paths(value, &format!("{path}/{key}"), out)
            }
        }
        Value::Array(items) => {
            for (index, value) in items.iter().enumerate() {
                nonfinite_paths(value, &format!("{path}/{index}"), out)
            }
        }
        _ => {}
    }
}

fn packed_layout(cfg: &MiniMaxH3DitConfig, device: &Device) -> Result<PackedLayout> {
    PackedLayout::build(
        JointGeometry::new(345, 96, 168)?,
        cfg.patch_size,
        &vec![crate::denoise::TEXT_TAG; 588],
        2,
        &[],
        device,
    )
}

fn actual_span_count() -> usize {
    LAST_SPAN_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn production_adaln_ffn_preparation_parity() -> Result<()> {
    let FixtureInputs {
        directory,
        adapter_path,
        files,
        cfg,
        verified,
        manifest,
    } = verified_fixture()?;
    let output = PathBuf::from(std::env::var("MODEL_PROOF_OUTPUT_DIR").map_err(error)?);
    std::fs::create_dir_all(&output).map_err(error)?;
    assert_eq!(
        (cfg.ffn_dim, cfg.time_embed_dim, cfg.patch_size),
        (14336, 2688, [1, 4, 4])
    );
    let plan = planning(&directory, &cfg, &manifest)?; // Before ANY model/tensor preparation.
    crate::nn::reset_attention_execution_report();
    let mut model =
        crate::dit::model::MiniMaxH3Dit::load_from_dir(&directory, &Device::Cpu, DType::BF16)?;
    let report = crate::adapters::apply_minimax_h3_adapters(
        &mut model,
        &[AdapterSpec {
            path: adapter_path,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
    )?;
    assert!(report.applied > 0 && report.unmatched_paths.is_empty());
    let mut host_block = model
        .proof_block(0)
        .ok_or_else(|| error("canonical block0 missing"))?
        .clone();
    let host_ff = ffn_from_block(&mut host_block)?;
    let time = model
        .projections()
        .time_embedder
        .as_ref()
        .ok_or_else(|| error("stock timestep MLP missing"))?
        .clone();
    let norm = model.projections().norm_out.clone();
    let rope = model.rope().clone();
    drop(model);
    let schedule = crate::denoise::adaln_schedule(&JointSchedule::new(8)?)?;
    assert_eq!(schedule.num_steps(), 8);
    let device = Device::new_cuda(0)?;
    assert!(candle_gen::candle_core::cuda::proof_pool::snapshot(0)?.is_none());
    let layout = packed_layout(&cfg, &device)?;
    let copied_layout = packed_layout(&cfg, &device)?;
    assert_eq!(layout.seq_len(), ROWS);
    let direct32 = with_workspace(&device, Some(BUDGET as usize), || {
        direct_stage(&files, &cfg, &mut host_block, &schedule, &layout, &device)
    })?;
    let copied32 = with_workspace(&device, Some(BUDGET as usize), || {
        copied_stage(
            host_block.clone(),
            (&time, &norm, &rope),
            &cfg,
            &schedule,
            &copied_layout,
            &device,
        )
    })?;
    let prep = preparation(&host_ff, &direct32.ff, &copied32.ff)?;
    let roots = json!({"arithmetic":stage_metrics(&direct32,&copied32)?,"independentPackedPositionIds":metric(layout.position_ids(),copied_layout.position_ids())?});
    for (reader, role, assets) in [
        (
            "candle-gen-minimax-h3::residency_arithmetic::verified-shards",
            candle_gen::gen_core::model_input_proof::InputRole::ModelWeights,
            verified[2..].iter().collect::<Vec<_>>(),
        ),
        (
            "candle-gen-minimax-h3::residency_arithmetic::verified-config",
            candle_gen::gen_core::model_input_proof::InputRole::Configuration,
            vec![&verified[0]],
        ),
        (
            "candle-gen-minimax-h3::residency_arithmetic::verified-turbo",
            candle_gen::gen_core::model_input_proof::InputRole::ArtifactInput,
            vec![&verified[1]],
        ),
    ] {
        record_verified_content(reader, role, &assets, EXPECTED_MANIFEST_SHA256)?;
    }
    let values = (0..ROWS * HIDDEN)
        .map(|i| {
            let value = ((i as u64 * 1_664_525 + 1_013_904_223 + 42) % 65_521) as i64 - 32_760;
            bf16::from_f32(value as f32 / 32_768.0)
        })
        .collect::<Vec<_>>();
    let input = Tensor::from_vec(values, (1, ROWS, HIDDEN), &device)?;
    let input_identity = identity(&input)?;
    assert_eq!(
        crate::nn::attention_budget_for_context(None).max_score_elements(),
        BUDGET
    );
    let budget = AttentionBudget::from_score_elements(BUDGET, false);
    let (direct_output, copied_output, ffn) =
        with_workspace(&device, Some(BUDGET as usize), || {
            let a = direct32.ff.forward_budgeted(&input, budget)?;
            assert_eq!(actual_span_count(), 90);
            let b = copied32.ff.forward_budgeted(&input, budget)?;
            assert_eq!(actual_span_count(), 90);
            let full = metric(&a, &b)?;
            let boundaries = traced_ffn(&direct32.ff, &copied32.ff, &input, &a, &b)?;
            Ok((a, b, json!({"fullOutput":full,"boundaries":boundaries})))
        })?;
    // Same actual direct constructors/operators/chunks, only the cuBLAS policy differs.
    let default_layout = packed_layout(&cfg, &device)?;
    let default = with_workspace(&device, None, || {
        direct_stage(
            &files,
            &cfg,
            &mut host_block,
            &schedule,
            &default_layout,
            &device,
        )
    })?;
    let default_output = with_workspace(&device, None, || {
        default.ff.forward_budgeted(&input, budget)
    })?;
    assert_eq!(actual_span_count(), 90);
    let observation = json!({"roots":stage_metrics(&direct32,&default)?,"ffnOutput":metric(&direct_output,&default_output)?,"policy":"library-default versus explicit32MiB; finite-only observational, no acceptance tolerance"});
    // Drop bulk tensors before emitting the compact complete metrics.
    drop((direct_output, copied_output, default_output));
    let same = json!({"preparation":prep,"roots":roots,"ffn":ffn});
    let mut failed = Vec::new();
    ordered_differences(&same, &mut failed);
    let mut nonfinite = Vec::new();
    nonfinite_paths(&observation, "workspaceDefaultVs32", &mut nonfinite);
    let attention = crate::nn::attention_execution_report();
    let evidence = json!({"kind":"h3-residency-arithmetic/v1","actualF32ReducedPrecision":candle_gen::candle_core::cuda::gemm_reduced_precision_f32(),"manifestSha256":EXPECTED_MANIFEST_SHA256,"acceptedSourcePair":manifest["acceptedSourcePair"],"verifiedAssets":verified,"planning":plan,"input":input_identity,"activationProvenance":"synthetic fixed integer LCG rounded once to BF16; NOT captured render activation","activationRecipe":{"multiplier":1664525,"increment":1013904223,"offset":42,"modulus":65521,"center":32760,"divisor":32768,"rounding":"once to BF16"},"schedule":{"steps":8,"distinctTimesteps":schedule.distinct_timesteps(),"key":schedule.key().to_string()},"dispatch":{"productionResidentElements":BUDGET,"productionWindowElements":BUDGET,"rows":ROWS,"width":28672,"chunkRows":1170,"fullChunks":89,"tailRows":424,"legacyChangedM":"no such actual C4 branch; omitted"},"rootConstruction":{"precomputeOrder":"Resident timeMLP -> AdaLNcache/evict -> norm_out; window timeMLP -> norm_out -> AdaLNwindow; preserved","rawProjectionObservation":"lossless inverse layout of actual six projected modulation tables; no extra AdaLN GEMM","activationObservation":"stateless SiLU/BF16 replay on actual captured temb after production projection, not internal handle","adapterPreparation":"Production read_adapter loads CPU tensors; transpose/alpha-rank fold occur on CPU before destination to_device for Resident as well as streamed LoRA","rotary":"MmRope retains host Vec<f64>; tables compute host f64 from F32 positions then upload independently","timeProject":"timestep_sincos regenerates host F32 frequencies/angles/sin/cos on every forward; no retained device frequency buffer","timeProjectObservation":"independently rebuilt stateless production constructor; internal forward handle not exposed","tables":"independent actual packed-layout construction, not shared deterministic substitutes"},"workspace":{"stream":"same actual stream; SetStream reset between arms","samePolicyBytes":BUDGET,"defaultVersus32":"observational only","proofPoolInstalled":false},"samePolicy":same,"workspaceDefaultVs32":observation,"firstDifference":failed.first(),"differenceOrder":"declared diagnostic semantic boundaries: preparation, root inputs/tables, timestep/AdaLN/norm_out, then sequential FFN projection/activation/output; NOT a captured render divergence","allDifferences":failed,"completed":{"directPrecomputeAndEvictStages":2,"windowPrecomputeStages":1,"additionalAdaLnBoundaryForwards":0,"productionFfnForwards":3,"pairedBoundaryChunks":90,"attentionCalls":attention.flash_calls+attention.materialized_calls},"readback":{"maximumElements":MAX_READBACK_ELEMENTS,"isolatedOffset":0,"actualTypedStorageEqualsChunk":true,"maximumF32TransferBytes":4*MAX_READBACK_ELEMENTS},"exactSamePolicy":failed.is_empty()});
    std::fs::write(
        output.join("residency-arithmetic.json"),
        serde_json::to_vec_pretty(&evidence).map_err(error)?,
    )
    .map_err(error)?;
    println!("H3_RESIDENCY_ARITHMETIC_DIAGNOSTIC {evidence}");
    assert!(
        failed.is_empty(),
        "same-policy arithmetic preparation diverged; evidence serialized first"
    );
    assert!(
        nonfinite.is_empty(),
        "workspace observation produced nonfinite values; evidence serialized first"
    );
    assert_eq!(
        attention.flash_calls + attention.materialized_calls,
        0,
        "arithmetic witness must not execute attention"
    );
    println!("\nMODEL_INFERENCE_BODY_COMPLETED dit::layers::residency_arithmetic::production_adaln_ffn_preparation_parity");
    Ok(())
}
