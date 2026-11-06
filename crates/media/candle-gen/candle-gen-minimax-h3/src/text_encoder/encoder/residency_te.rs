#![cfg(test)]
//! Explicit full-shape TE first-difference diagnostic; no production math changes.
#[cfg(not(target_endian = "little"))]
compile_error!("TE F32 proof byte format requires little-endian host");
use super::*;
use candle_gen::candle_core::{cuda::cudarc, Device};
use candle_gen::gen_core::{GenerationMemory, GenerationRequest, LoadShape, TransformerComponent};
use candle_gen::quant::{QLinear, QuantWeight};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};
const ROWS: usize = 588;
const MAX_READBACK_ELEMENTS: usize = 1024 * 1024;
const WORKSPACE: usize = 32 * 1024 * 1024;
const MANIFEST: &str = "D:/Codex/work/sw-f329/envelope-outputs/residency-te-manifest.v1.json";
const MANIFEST_SHA: &str = "2bad494f27a89b6770f98fb0858ed1d89620cbc452716bcbd44a562dc92df113";
const REQUEST_SHA: &str = "b5b5ba095203fab1eddb9ba1793162959328e16ee34a0ca6e19d44e2325bce23";
const TIER_REV: &str = "137ce668c55a20bc0935fd1cf2a3de8448abb7f4";
const BASE_REV: &str = "6818f6c32d12b210915e44ad56a4228c2608f160";
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

fn update_f32_hash(hash: &mut Sha256, values: &[f32]) {
    // SAFETY: f32 has no padding. Read only
    // initialized bytes within the slice's exact extent, on the asserted LE host.
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    hash.update(bytes);
}

fn readback_chunk(flat: &Tensor, start: usize, len: usize) -> Result<Vec<f32>> {
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

fn tensor_identity(tensor: &Tensor) -> Result<Value> {
    assert_eq!(tensor.dtype(), DType::F32);
    let flat = tensor.flatten_all()?;
    let mut hash = Sha256::new();
    for start in (0..tensor.elem_count()).step_by(MAX_READBACK_ELEMENTS) {
        let len = MAX_READBACK_ELEMENTS.min(tensor.elem_count() - start);
        update_f32_hash(&mut hash, &readback_chunk(&flat, start, len)?);
    }
    Ok(json!({"shape":tensor.dims(),"dtype":"F32","sha256":format!("{:x}",hash.finalize())}))
}

// Readback is bounded, and metrics cover EVERY element, including the ragged tail.
// Hash the original F32 bits without sampling or serialization rounding.
fn comparison(left: &Tensor, right: &Tensor) -> Result<Value> {
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
        let a = readback_chunk(&left_flat, start, len)?;
        let b = readback_chunk(&right_flat, start, len)?;
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

fn packed_hash(projection: &Proj) -> Result<String> {
    match projection.inner.base_qlinear() {
        Some(QLinear::Quantized {
            weight: QuantWeight::Dequant(weight),
            bias: None,
            ..
        }) => {
            assert_eq!(format!("{:?}", weight.dtype()), "Q4_1");
            Ok(format!("{:x}", Sha256::digest(weight.data()?.as_ref())))
        }
        other => Err(error(format!(
            "actual TE Q4_1 DequantDense required: {other:?}"
        ))),
    }
}

fn layer_preparation(resident: &Qwen3DecoderLayer, prepared: &Qwen3DecoderLayer) -> Result<Value> {
    let pairs = [
        ("q", &resident.attn.q_proj, &prepared.attn.q_proj),
        ("k", &resident.attn.k_proj, &prepared.attn.k_proj),
        ("v", &resident.attn.v_proj, &prepared.attn.v_proj),
        ("o", &resident.attn.o_proj, &prepared.attn.o_proj),
        ("gate", &resident.mlp.gate, &prepared.mlp.gate),
        ("up", &resident.mlp.up, &prepared.mlp.up),
        ("down", &resident.mlp.down, &prepared.mlp.down),
    ];
    let mut projections = Vec::new();
    let mut exact = true;
    for (name, resident, prepared) in pairs {
        let a = packed_hash(resident)?;
        let b = packed_hash(prepared)?;
        exact &= a == b;
        projections.push(json!({"name":name,"residentSha256":a,"preparedSha256":b,"exact":a==b}));
    }
    let norms = [
        ("input", &resident.input_ln, &prepared.input_ln),
        ("post", &resident.post_ln, &prepared.post_ln),
        ("q", &resident.attn.q_norm, &prepared.attn.q_norm),
        ("k", &resident.attn.k_norm, &prepared.attn.k_norm),
    ]
    .into_iter()
    .map(|(name, a, b)| Ok(json!({"name":name,"metrics":comparison(a,b)?})))
    .collect::<Result<Vec<_>>>()?;
    exact &= norms.iter().all(|v| v["metrics"]["exact"] == true);
    Ok(json!({"exact":exact,"packedProjections":projections,"norms":norms}))
}

fn planning(path: &Path, cfg: &MiniMaxH3TeConfig) -> Result<Value> {
    let (facts, decode) =
        crate::memory_strategy::proof_text_encoder_inventory(path).map_err(error)?;
    let resident = facts.peak_bytes(None);
    let window = facts.peak_bytes(Some(1));
    assert!(resident < u64::MAX && window < u64::MAX);
    // Actual fixed source geometry is checked before this arithmetic. Simultaneous
    // direct/streamed states, decoder lexical temporaries and bounded readbacks.
    // Sum of phase alternatives is intentionally conservative, not a product estimator.
    let hidden = ROWS as u64 * cfg.hidden_size as u64 * 4;
    let head = ROWS as u64 * cfg.num_heads as u64 * cfg.head_dim as u64 * 4;
    let mlp = ROWS as u64 * cfg.intermediate_size as u64 * 4;
    let scores = cfg.num_heads as u64 * ROWS as u64 * ROWS as u64;
    let scratch = 16 * hidden
        + 12 * head
        + 8 * mlp
        + 16 * scores
        + 2 * MAX_READBACK_ELEMENTS as u64 * 4
        + WORKSPACE as u64;
    let peak = resident
        .checked_add(window)
        .and_then(|p| p.checked_add(decode.checked_mul(2)?))
        .and_then(|p| p.checked_add(scratch))
        .ok_or_else(|| error("TE diagnostic planning overflow"))?;
    let cap = 32u64 * 1024 * 1024 * 1024;
    assert!(
        peak <= cap,
        "diagnostic source-derived plan exceeds32GiB; do not raise cap blindly"
    );
    Ok(json!({"residentWeights":resident,"windowOneWeights":window,
        "projectionDecodeAllowance":2*decode,"carriedAndWorkingAllowance":scratch,
        "plannedDevicePeakBytes":peak,"diagnosticDeviceCapBytes":cap,
        "scope":"source-owned runtime payloads plus conservative two-route working union; no product-fit claim",
        "workingTerms":{"hiddenBytes":hidden,"headBytes":head,"mlpBytes":mlp,
            "scoreElements":scores,"hiddenCopies":16,"headCopies":12,"mlpCopies":8,"scoreScratchBytesPerElement":16},
        "inventoryIncludesUnusedVisionRootConservatively":true}))
}

fn load_encoder(
    files: &[PathBuf],
    cfg: &MiniMaxH3TeConfig,
    device: &Device,
) -> Result<MiniMaxH3TextEncoder> {
    let prefixes = crate::text_encoder::lm_prefixes(crate::text_encoder::LM_PREFIX, cfg);
    let refs = prefixes.iter().map(String::as_str).collect::<Vec<_>>();
    let weights = Weights::from_files_filtered_host_packed(
        files,
        device,
        crate::text_encoder::TE_STORE_DTYPE,
        &refs,
    )?;
    MiniMaxH3TextEncoder::from_weights(
        &weights,
        crate::text_encoder::LM_PREFIX,
        cfg,
        crate::text_encoder::TE_STORE_DTYPE,
    )
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn production_prompt_resident_window_first_difference() -> Result<()> {
    assert!(
        std::env::var_os("SCENEWORKS_CUDA_PROOF_POOL_BYTES").is_none(),
        "workspace observation requires no proof-pool auto-binding"
    );
    let manifest_path = std::env::var_os("H3_RESIDENCY_TE_MANIFEST")
        .map_or_else(|| PathBuf::from(MANIFEST), PathBuf::from);
    let raw = std::fs::read(&manifest_path).map_err(error)?;
    assert_eq!(format!("{:x}", Sha256::digest(&raw)), MANIFEST_SHA);
    let manifest: Value = serde_json::from_slice(&raw).map_err(error)?;
    assert_eq!(manifest["schemaVersion"], 1);
    assert_eq!(manifest["tierRevision"], TIER_REV);
    assert_eq!(manifest["baseRevision"], BASE_REV);
    let directory = required_path(&manifest, "textEncoderDir")?;
    let base = required_path(&manifest, "baseRoot")?;
    let receipt_path = required_path(&manifest["acceptedStagingReceipt"], "path")?;
    let receipt_raw = std::fs::read(receipt_path).map_err(error)?;
    assert_eq!(
        format!("{:x}", Sha256::digest(&receipt_raw)),
        manifest["acceptedStagingReceipt"]["sha256"]
    );
    let receipt: Value = serde_json::from_slice(&receipt_raw).map_err(error)?;
    assert_eq!(receipt["sourcePair"], manifest["acceptedSourcePair"]);
    let assets = manifest["assets"]
        .as_array()
        .ok_or_else(|| error("mandatory TE asset list missing"))?;
    assert_eq!(assets.len(), 16);
    let mut verified = Vec::new();
    for asset in assets {
        let path = required_path(asset, "path")?;
        let repo = asset["repo"].as_str().ok_or_else(|| error("asset repo"))?;
        let revision = asset["revision"]
            .as_str()
            .ok_or_else(|| error("asset revision"))?;
        let relative = asset["relative"]
            .as_str()
            .ok_or_else(|| error("asset relative"))?;
        verified.push(verify_asset(
            &path, asset, &receipt, repo, revision, relative,
        )?);
    }
    let files = candle_gen::loader::sorted_safetensors(&directory, "TE first difference")?;
    assert_eq!(files.len(), 13);
    for file in &files {
        assert!(
            verified.iter().any(|v| v["path"]
                .as_str()
                .is_some_and(|p| Path::new(p) == file.canonicalize().unwrap())),
            "unexpected source shard"
        );
    }
    for path in [
        directory.join("config.json"),
        base.join("tokenizer/tokenizer.json"),
        base.join("tokenizer/tokenizer_config.json"),
    ] {
        assert!(
            verified.iter().any(|v| v["path"]
                .as_str()
                .is_some_and(|p| Path::new(p) == path.canonicalize().unwrap())),
            "required consumed asset unbound"
        );
    }
    let request_path = required_path(&manifest, "requestPath")?;
    let request_raw = std::fs::read(request_path).map_err(error)?;
    assert_eq!(format!("{:x}", Sha256::digest(&request_raw)), REQUEST_SHA);
    assert_eq!(manifest["requestSha256"], REQUEST_SHA);
    let request: Value = serde_json::from_slice(&request_raw).map_err(error)?;
    let prompt = request["prompt"]
        .as_str()
        .ok_or_else(|| error("original prompt missing"))?;
    for name in [
        "referenceAssetIds",
        "sourceClipAssetIds",
        "referenceAudioAssetIds",
    ] {
        assert!(request[name].as_array().is_some_and(Vec::is_empty));
    }
    assert!(request["sourceAssetId"].is_null() && request["lastFrameAssetId"].is_null());
    let cfg = MiniMaxH3TeConfig::from_component_dir(&directory)?;
    assert_eq!(
        (
            cfg.hidden_size,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            cfg.intermediate_size,
            cfg.layers_to_run()
        ),
        (5120, 64, 8, 128, 25600, 50)
    );
    let plan = planning(&directory, &cfg)?;
    println!("H3_TE_DIAGNOSTIC_PLAN {plan}");
    let output = PathBuf::from(std::env::var("MODEL_PROOF_OUTPUT_DIR").map_err(error)?);
    std::fs::create_dir_all(&output).map_err(error)?;
    let device = Device::new_cuda(0)?;
    assert!(candle_gen::candle_core::cuda::proof_pool::snapshot(0)?.is_none());
    let tokenizer = crate::text_encoder::MiniMaxH3Tokenizer::from_snapshot(&base)?;
    let (ids, attention_mask) = tokenizer.encode_prompt(prompt, &device)?;
    assert_eq!(ids.dims(), [1, ROWS]);
    let id_values = ids.flatten_all()?.force_contiguous()?.to_vec1::<u32>()?;
    let ids_hash = format!(
        "{:x}",
        Sha256::digest(
            id_values
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<_>>()
        )
    );
    assert_eq!(attention_mask.dims(), [1, ROWS]);
    assert_eq!(attention_mask.dtype(), DType::U32);
    let mask_values = attention_mask
        .flatten_all()?
        .force_contiguous()?
        .to_vec1::<u32>()?;
    assert_eq!(mask_values.len(), ROWS);
    assert!(mask_values.iter().all(|&value| value == 1));
    let native_mask_identity = json!({
        "shape":attention_mask.dims(),"dtype":"U32","elements":mask_values.len(),
        "sha256":format!("{:x}",Sha256::digest(mask_values.iter().flat_map(|value|value.to_le_bytes()).collect::<Vec<_>>()))
    });
    crate::nn::reset_attention_execution_report();
    let (mut resident, resident_context, same_policy) = with_workspace(
        &device,
        Some(WORKSPACE),
        || {
            let resident = load_encoder(&files, &cfg, &device)?;
            let mut streamed = load_encoder(&files, &cfg, &Device::Cpu)?;
            assert!(
                resident.embed_tokens.embedding.is_quantized()
                    && streamed.embed_tokens.embedding.is_quantized()
            );
            let discarded = streamed.rotary.text(MAX_TEXT_POSITIONS)?;
            streamed.prepare_streamed(&cfg, &device)?;
            let rt = resident.rotary.text(MAX_TEXT_POSITIONS)?;
            let st = streamed.rotary.text(MAX_TEXT_POSITIONS)?;
            let tables = json!({"runtimeCos":comparison(&rt.0,&st.0)?,"runtimeSin":comparison(&rt.1,&st.1)?,
            "discardedCpuCosVsDirectGpu":comparison(&discarded.0,&rt.0)?,"discardedCpuSinVsDirectGpu":comparison(&discarded.1,&rt.1)?,
            "frequencyIntermediate":"not retained or observable; no synthetic frequency resolver",
            "runtimeSources":"independent direct GPU constructor and CPU-prepared then GPU-rebuilt streamed constructor"});
            let (rcos, rsin) = resident.rotary.text(ROWS)?;
            let (scos, ssin) = streamed.rotary.text(ROWS)?;
            let rmask = resident.mask(&attention_mask, 1, ROWS)?;
            let smask = streamed.mask(&attention_mask, 1, ROWS)?;
            let masks =
                json!({"resident":tensor_identity(&rmask)?,"streamed":tensor_identity(&smask)?});
            let rh = resident.embed(&ids)?;
            let memory = GenerationRequest {
                memory: Some(GenerationMemory {
                    stream_transformer_blocks: true,
                    transformer_window_size: Some(1),
                    transformer_window_component: Some(TransformerComponent::Both),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let guard = crate::block_stream::RequestGuard::enter(
                &memory,
                LoadShape::DeferredMaterialization,
                &device,
            )?;
            let sh = streamed.embed(&ids)?;
            let embeddings = comparison(&rh, &sh)?;
            let mut resident_hidden = Some(rh);
            let mut previous_streamed = Some(sh.clone());
            let mut layers = Vec::new();
            let streamed_context=streamed.run_layers(sh,&scos,&ssin,&smask,|index,hidden|{
            let rinput=resident_hidden.take().ok_or_else(||error("resident state absent"))?;
            let sinput=previous_streamed.take().ok_or_else(||error("streamed state absent"))?;
            let input=comparison(&rinput,&sinput)?;
            let preparation=layer_preparation(&resident.layers[index],&streamed.layers[index])?;
            let reference=resident.layers[index].forward(&rinput,&rcos,&rsin,&rmask)?;
            let metrics=comparison(&reference,&hidden)?;
            println!("H3_TE_LAYER_COMPLETED index={index} exact={}",metrics["exact"]);
            layers.push(json!({"index":index,"input":input,"preparation":preparation,"output":metrics}));
            resident_hidden=Some(reference);
            previous_streamed=Some(hidden.clone());
            Ok(hidden)
        })?;
            drop(guard);
            assert!(crate::block_stream::current().is_none());
            assert_eq!(layers.len(), 50);
            let resident_context = resident_hidden
                .take()
                .ok_or_else(|| error("missing resident context"))?;
            let context = comparison(&resident_context, &streamed_context)?;
            let first = if embeddings["exact"] != true {
                Some(json!({"boundary":"embedding"}))
            } else if tables["runtimeCos"]["exact"] != true || tables["runtimeSin"]["exact"] != true
            {
                Some(json!({"boundary":"runtime-rotary-tables"}))
            } else {
                layers
                    .iter()
                    .find(|v| {
                        v["input"]["exact"] != true
                            || v["preparation"]["exact"] != true
                            || v["output"]["exact"] != true
                    })
                    .map(|v| json!({"boundary":"decoder-layer","index":v["index"]}))
            };
            let exact = first.is_none()
                && context["exact"] == true
                && masks["resident"]["sha256"] == masks["streamed"]["sha256"];
            let evidence = json!({"exact":exact,"workspaceBytes":WORKSPACE,"embeddings":embeddings,"tables":tables,"masks":masks,
            "layers":layers,"context":context,"firstDifference":first,
            "referenceExecution":"original GPU-resident DecoderLayer::forward in existing streamed run_layers(after) callback; same effective32Mi budget",
            "streamedExecution":"canonical CPU-prepared encoder, actual production window1 run_layers and independent runtime tables"});
            Ok((resident, resident_context, evidence))
        },
    )?;
    let observation = with_workspace(&device, None, || {
        // Production's resident constructor builds these under library-default too.
        // Rebuild rather than reuse the explicit32Mi tables and hide a table difference.
        resident.rotary = Rotary::new(cfg.head_dim, cfg.rope_theta, MAX_TEXT_POSITIONS, &device)?;
        let default_context = resident.forward(&ids, &attention_mask)?;
        comparison(&default_context, &resident_context)
    })?;
    let routes = crate::nn::attention_execution_report();
    assert_eq!(routes.materialized_calls, 150);
    assert_eq!(routes.flash_calls, 0);
    let finite = observation["nonfinite"] == 0;
    let accepted = same_policy["exact"] == true && finite;
    let evidence = json!({"kind":"h3-residency-te/v1","schemaVersion":1,"accepted":accepted,
        "source":{"manifestSha256":MANIFEST_SHA,"manifest":manifest,"verifiedAssets":verified,"requestSha256":REQUEST_SHA},
        "input":{"idsShape":ids.dims(),"idsSha256":ids_hash,"attentionMask":native_mask_identity,"tokens":ROWS},
        "planning":plan,"same_policy":same_policy,"workspace_default_vs_32m":observation,
        "completedLayerPairs":50,"completedDefaultResidentLayers":50,"materializedAttentionCalls":routes.materialized_calls,
        "actualF32ReducedPrecision":candle_gen::candle_core::cuda::gemm_reduced_precision_f32(),
        "readback":{"maximumAllowedChunkElements":MAX_READBACK_ELEMENTS,"elementBytes":4,"everyChunkLayoutStartOffset":0,
            "everyStorageElementCountEqualsChunk":true},
        "scope":"TE component first-difference only; no DiT/Turbo/Veda/VAE/clip or fit/E5 acceptance claim"});
    std::fs::write(
        output.join("residency-te.json"),
        serde_json::to_vec_pretty(&evidence).map_err(error)?,
    )
    .map_err(error)?;
    assert!(
        accepted,
        "TE same-policy exact diagnostic failed; inspect residency-te.json before any retry"
    );
    println!("H3_RESIDENCY_TE_PROOF {evidence}");
    println!("\nMODEL_INFERENCE_BODY_COMPLETED text_encoder::encoder::residency_te::production_prompt_resident_window_first_difference");
    Ok(())
}
