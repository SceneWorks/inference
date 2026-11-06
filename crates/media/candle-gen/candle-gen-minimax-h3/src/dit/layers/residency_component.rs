//! Manual real-source first-divergence witness. No production policy or arithmetic changes.
#[cfg(not(target_endian = "little"))]
compile_error!("H3 BF16 proof byte format requires a little-endian host");

use super::*;
use candle_gen::candle_core::{cuda::cudarc, Device};
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::quant::{QLinear, QuantWeight};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const ROWS: usize = 104_554;
const HIDDEN: usize = 5376;
const HEADS: usize = 56;
const HEAD_DIM: usize = 128;
const BUDGET: u64 = 32 * 1024 * 1024;
const PREFIX: &str = "transformer_blocks.0.attn.to_q";
const NORM: &str = "transformer_blocks.0.attn.norm_q.weight";
const MODEL_REVISION: &str = "137ce668c55a20bc0935fd1cf2a3de8448abb7f4";
const TURBO_REVISION: &str = "5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1";
// Frozen by the coordinator from the accepted source-staging receipt, independently
// of this witness. A matching revision label alone is never an input attestation.
const EXPECTED_MANIFEST_SHA256: &str =
    "9df0a1203b8a20d87b81888cb4edeaa34537f50c558aba239a79af5b3c443b01";
const DEFAULT_MANIFEST: &str =
    "D:/Codex/work/sw-f329/envelope-outputs/residency-component-manifest.v1.json";

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

fn tensor_identity(tensor: &Tensor) -> Result<Value> {
    assert_eq!(tensor.dtype(), DType::BF16);
    let flat = tensor.flatten_all()?;
    let mut hash = Sha256::new();
    for start in (0..tensor.elem_count()).step_by(1024 * 1024) {
        let len = (1024 * 1024).min(tensor.elem_count() - start);
        update_bf16_hash(&mut hash, &flat.narrow(0, start, len)?.to_vec1::<bf16>()?);
    }
    Ok(json!({"shape":tensor.dims(),"dtype":"BF16","sha256":format!("{:x}",hash.finalize())}))
}

// Readback is bounded, and metrics cover EVERY element, including the ragged tail.
// Hash the original BF16 bits, not a rounded image or an F32 serialization.
fn comparison(left: &Tensor, right: &Tensor) -> Result<Value> {
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
    for start in (0..left.elem_count()).step_by(1024 * 1024) {
        let len = (1024 * 1024).min(left.elem_count() - start);
        let a = left_flat.narrow(0, start, len)?.to_vec1::<bf16>()?;
        let b = right_flat.narrow(0, start, len)?.to_vec1::<bf16>()?;
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

fn base(layer: &LinearNoBias) -> LinearNoBias {
    let mut base = layer.clone();
    base.clear_adapters();
    base
}

fn residual(layer: &LinearNoBias, input: &Tensor) -> Result<Tensor> {
    let input = input.reshape((ROWS, HIDDEN))?;
    let zeros = Tensor::zeros((ROWS, HEADS * HEAD_DIM), DType::BF16, input.device())?;
    Ok(layer.inner.apply_residuals(&input, zeros)?)
}

fn projection(layer: &LinearNoBias, input: &Tensor, completed: &mut u64) -> Result<Tensor> {
    let output = layer.forward(input)?;
    *completed += 1;
    Ok(output)
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

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn production_q4_turbo_workspace_and_qk_parity() -> Result<()> {
    assert!(
        std::env::var_os("SCENEWORKS_CUDA_PROOF_POOL_BYTES").is_none(),
        "workspace contrast requires NO proof-pool auto-binding; never silently override a cap"
    );
    let manifest_path = std::env::var_os("H3_RESIDENCY_COMPONENT_MANIFEST")
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
    let output = PathBuf::from(std::env::var("MODEL_PROOF_OUTPUT_DIR").map_err(error)?);
    std::fs::create_dir_all(&output).map_err(error)?;
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
    let adapter_hash = manifest["expectedTurbo"]["sha256"]
        .as_str()
        .ok_or_else(|| error("missing Turbo digest"))?
        .to_owned();
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

    // The actual installer is the only authority for Turbo namespace, transpose,
    // alpha/rank fold, source BF16 precision, and strength. No alternate resolver.
    let host = {
        let mut model =
            crate::dit::model::MiniMaxH3Dit::load_from_dir(&directory, &Device::Cpu, DType::BF16)?;
        let report = crate::adapters::apply_minimax_h3_adapters(
            &mut model,
            &[AdapterSpec {
                path: adapter_path.clone(),
                scale: 1.0,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )?;
        assert!(report.applied > 0 && report.unmatched_paths.is_empty());
        model
            .adaptable_mut(&["transformer_blocks", "0", "attn", "to_q"])
            .ok_or_else(|| error("missing actual block0.to_q"))?
            .clone()
    };
    assert_eq!(host.inner.base_shape(), (HEADS * HEAD_DIM, HIDDEN));
    assert_eq!(host.adapters.len(), 1);
    assert!(host.lokr_adapters.is_empty());
    assert_eq!(host.adapters[0].dtype(), DType::BF16);
    assert_eq!(host.adapters[0].scale, 1.0);
    let device = Device::new_cuda(0)?;
    assert!(
        candle_gen::candle_core::cuda::proof_pool::snapshot(0)?.is_none(),
        "installed proof pool would overwrite workspace controls"
    );
    let stream = device.as_cuda_device()?.cuda_stream();
    let major = stream
        .context()
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .map_err(error)?;
    let minor = stream
        .context()
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
        .map_err(error)?;
    // Production's full reader preserves native precision. Prove the filtered
    // reader's float coerce is an identity for EVERY selected source operand,
    // rather than introduce an alternate preparation policy into this witness.
    let mut selected_header = Vec::new();
    for file in &files {
        let header = candle_gen::loader::mmap_safetensors(file)?;
        for (name, view) in header.tensors() {
            if name.starts_with(PREFIX) || name == NORM {
                let source_dtype = format!("{:?}", view.dtype());
                assert!(matches!(source_dtype.as_str(), "BF16" | "U32"));
                selected_header
                    .push(json!({"name":name,"dtype":source_dtype,"shape":view.shape()}));
            }
        }
    }
    assert_eq!(selected_header.len(), 4);
    let weights = Weights::from_files_filtered(&files, &device, DType::BF16, &[PREFIX, NORM])?;
    let norm = weights.require(NORM)?.to_dtype(DType::BF16)?;
    assert_eq!(norm.dims(), [HEAD_DIM]);
    let mut direct = LinearNoBias::from_weights(&weights, PREFIX, DType::BF16)?;
    for factor in &host.adapters {
        direct.push_lora(
            factor.a.to_device(&device)?,
            factor.b.to_device(&device)?,
            factor.scale,
        )?;
    }
    let copied = host.copy_to_device(&device)?;
    let packed = json!({"hostSha256":packed_hash(&host)?,
        "directSha256":packed_hash(&direct)?, "copiedSha256":packed_hash(&copied)?});
    let host_factors = host.inner.__model_proof_frozen_lora_factors()?;
    let direct_factors = direct.inner.__model_proof_frozen_lora_factors()?;
    let copied_factors = copied.inner.__model_proof_frozen_lora_factors()?;
    assert_eq!(host_factors.len(), 1);
    assert_eq!(host_factors.len(), direct_factors.len());
    assert_eq!(host_factors.len(), copied_factors.len());
    let factors = host_factors
        .iter()
        .zip(&direct_factors)
        .zip(&copied_factors)
        .map(|((host, direct), copied)| {
            assert_eq!(host.2, direct.2);
            assert_eq!(host.2, copied.2);
            assert_eq!(host.2, 1.0);
            Ok(
                json!({"hostVsDirectA":comparison(&host.0, &direct.0)?, "hostVsDirectB":comparison(&host.1, &direct.1)?,
                    "hostVsCopiedA":comparison(&host.0, &copied.0)?, "hostVsCopiedB":comparison(&host.1, &copied.1)?,
                    "hostScale":host.2,"directScale":direct.2,"copiedScale":copied.2,
                    "strength":1.0,"aShape":host.0.dims(),"bShape":host.1.dims(),
                    "storage":"actual executable frozen factors; provider inspection records not used"}),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    drop(weights);

    // Fixed integer formula, rounded ONCE to BF16, independent of GPU RNG/workspace.
    let input_values = (0..ROWS * HIDDEN)
        .map(|i| {
            let value = ((i as u64 * 1_664_525 + 1_013_904_223 + 42) % 65_521) as i64 - 32_760;
            bf16::from_f32(value as f32 / 32_768.0)
        })
        .collect::<Vec<_>>();
    let input = Tensor::from_vec(input_values, (1, ROWS, HIDDEN), &device)?;
    let input_identity = tensor_identity(&input)?;
    let mut completed_projections = 0;
    let same_policy = with_workspace(&device, Some(BUDGET as usize), || {
        let base_comparison = comparison(
            &projection(&base(&direct), &input, &mut completed_projections)?,
            &projection(&base(&copied), &input, &mut completed_projections)?,
        )?;
        let residual_comparison =
            comparison(&residual(&direct, &input)?, &residual(&copied, &input)?)?;
        let raw = projection(&direct, &input, &mut completed_projections)?;
        let full_comparison = comparison(
            &raw,
            &projection(&copied, &input, &mut completed_projections)?,
        )?;
        Ok((
            json!({"userWorkspaceBytes":BUDGET,"base":base_comparison,
            "residual":residual_comparison,"full":full_comparison}),
            raw,
        ))
    })?;
    let (preparation, raw_32m) = same_policy;
    let raw_default = with_workspace(&device, None, || {
        projection(&direct, &input, &mut completed_projections)
    })?;
    let default_vs_32m = comparison(&raw_default, &raw_32m)?;
    let raw_12m = with_workspace(&device, Some(12 * 1024 * 1024), || {
        projection(&direct, &input, &mut completed_projections)
    })?;
    let default_vs_12m = comparison(&raw_default, &raw_12m)?;
    let twelve_vs_32m = comparison(&raw_12m, &raw_32m)?;
    drop(raw_default);
    drop(raw_12m);
    drop(input);
    drop(direct);
    drop(copied);
    drop(host);

    let raw = raw_32m.reshape((1, ROWS, HEADS, HEAD_DIM))?;
    drop(raw_32m);
    let budget = AttentionBudget::from_score_elements(BUDGET, false);
    let original_norm = rms_weighted(&raw, &norm, config.qk_norm_eps)?;
    let (bounded_norm, norm_report) =
        precondition_projected_qk(raw.clone(), &norm, config.qk_norm_eps, None, budget, None)?;
    let norm_metrics = comparison(&original_norm, &bounded_norm)?;
    drop(bounded_norm);
    let rope = MmRope::new(config.rope_freq_dim, config.rope_theta)?;
    let positions = (0..ROWS)
        .map(|i| {
            [
                (i % 102) as f64 * 0.125,
                (i % 24) as f64 - 12.0,
                (i % 42) as f64 - 21.0,
            ]
        })
        .collect::<Vec<_>>();
    let tables = rope.tables_from_rows(&positions, &device)?;
    let original_rotary = rope.apply(&original_norm, &tables)?;
    drop(original_norm);
    let (bounded_rotary, rotary_report) = precondition_projected_qk(
        raw,
        &norm,
        config.qk_norm_eps,
        Some((&rope, &tables)),
        budget,
        None,
    )?;
    let rotary_metrics = comparison(&original_rotary, &bounded_rotary)?;
    assert_eq!(norm_report.norm_chunks, 23);
    assert_eq!(rotary_report.norm_chunks, 23);
    assert_eq!(rotary_report.rotary_chunks, 23);
    assert_eq!(rotary_report.maximum_chunk_elements, 33_553_408);
    let preparation_exact = packed["hostSha256"] == packed["directSha256"]
        && packed["hostSha256"] == packed["copiedSha256"]
        && factors.iter().all(|f| {
            [
                "hostVsDirectA",
                "hostVsDirectB",
                "hostVsCopiedA",
                "hostVsCopiedB",
            ]
            .iter()
            .all(|key| f[key]["exact"] == true)
                && f["hostScale"] == f["directScale"]
                && f["hostScale"] == f["copiedScale"]
        })
        && ["base", "residual", "full"]
            .iter()
            .all(|key| preparation[key]["exact"] == true);
    let qk_exact = norm_metrics["exact"] == true && rotary_metrics["exact"] == true;
    let exact = preparation_exact && qk_exact;
    assert_eq!(completed_projections, 6);
    assert_eq!(
        adapter_hash,
        hash_file(&adapter_path)?,
        "adapter changed during witness"
    );
    let finite_observations = [&default_vs_12m, &default_vs_32m, &twelve_vs_32m]
        .iter()
        .all(|m| m["nonfinite"] == 0);
    let evidence = json!({
        "kind":"h3-residency-component/v1", "schemaVersion":1,
        "accepted":exact && finite_observations, "completedLearnedProjectionCalls":completed_projections,
        "completedResidualOnlyScopes":2,
        "device":{"logicalOrdinal":0,"computeCapability":[major,minor],
            "cudaVisibleDevices":std::env::var("CUDA_VISIBLE_DEVICES").ok(),
            "proofPoolInstalled":false,"stream":format!("{:p}",stream.cu_stream())},
        "source":{"manifestSha256":manifest_hash,"expectedManifestSha256":EXPECTED_MANIFEST_SHA256,
            "manifest":manifest,"configSha256":format!("{:x}",Sha256::digest(&config_bytes)),
            "acceptedSourcePair":receipt["sourcePair"],
            "verifiedAssets":verified_sources,
            "turboSha256":adapter_hash, "projection":PREFIX,
            "selectedSourceHeader":selected_header,
            "runtimePacked":packed,"canonicalTurboFactors":factors,
            "normQ":tensor_identity(&norm)?},
        "input":{"identity":"integer-lcg-mod65521/seed42/bf16-once/v1", "tensor":input_identity},
        "preparation_same_policy":preparation,
        "workspace_default_vs_32m":default_vs_32m,
        "workspace_default_vs_12m":default_vs_12m,
        "workspace_12m_vs_32m":twelve_vs_32m,
        "workspaceControls":{"libraryDefaultUserBytes":null,"explicitUserBytes":[12582912,33554432],
            "acceptance":"observational contrasts only; no projection tolerance inferred",
            "rationale":"CUDA12.9 documents sm12x default recommendation 12MiB, sm10x/sm90 32MiB; policy differences are a confound, not established causality",
            "source":"https://docs.nvidia.com/cuda/archive/12.9.0/cublas/index.html#cublassetworkspace"},
        "qk_full_vs_bounded":{"norm":norm_metrics,"rotary":rotary_metrics,
            "shape":[1,ROWS,HEADS,HEAD_DIM],"budgetElements":BUDGET,"normChunks":23,
            "rotaryChunks":23,"tokensPerChunk":4681,"tailTokens":1572,
            "maximumChunkElements":rotary_report.maximum_chunk_elements,
            "positionIdentity":"deterministic-three-axis-mod102x24x42/v1",
            "effectiveCos":tensor_identity(&tables.cos.to_dtype(DType::BF16)?)?,
            "effectiveSin":tensor_identity(&tables.sin.to_dtype(DType::BF16)?)?},
        "assertions":{"samePolicyPreparationExact":preparation_exact,"fullBoundedQkExact":qk_exact,"workspaceObservationsFinite":finite_observations,
            "noAttentionPredictorDenoiseVaeOrClip":true}
    });
    std::fs::write(
        output.join("residency-component.json"),
        serde_json::to_vec_pretty(&evidence).map_err(error)?,
    )
    .map_err(error)?;
    assert!(
        exact,
        "exact same-policy/QK contract failed; inspect residency-component.json"
    );
    assert!(
        finite_observations,
        "workspace contrast produced nonfinite values"
    );
    println!("H3_RESIDENCY_COMPONENT_PROOF {}", evidence);
    println!("\nMODEL_INFERENCE_BODY_COMPLETED dit::layers::residency_component::production_q4_turbo_workspace_and_qk_parity");
    Ok(())
}
