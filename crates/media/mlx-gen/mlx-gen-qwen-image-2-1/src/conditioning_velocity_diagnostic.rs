//! Ignored library-test-only, fixed teacher-forced diagnostic; never acceptance.
//! No production precision policy, rendering request, or optimizer is changed.
use std::collections::BTreeMap;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use mlx_gen::adapters::loader::{apply_lokr, parse_lokr};
use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen::{GenerationRequest, Quant, Result};
use mlx_rs::{Array, Dtype};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::q4_real_weights_support as evidence;
use crate::reference::PreparedReference;
use crate::transformer::QwenImage21Transformer;
#[path = "conditioning_velocity_math.rs"]
mod math;

#[path = "conditioning_velocity_current_inputs.rs"]
mod current_inputs;
#[path = "conditioning_velocity_inputs.rs"]
mod inputs;
use inputs::header;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Binding {
    Historical,
    CurrentFailedNative768,
}

impl Binding {
    fn manifest_env(self) -> &'static str {
        match self {
            Self::Historical => "QWEN_IMAGE_2_1_VELOCITY_MANIFEST",
            Self::CurrentFailedNative768 => "QWEN_IMAGE_2_1_CURRENT_VELOCITY_MANIFEST",
        }
    }

    fn output_dir(self) -> &'static str {
        match self {
            Self::Historical => "velocity-discriminator",
            Self::CurrentFailedNative768 => "current-q4-velocity-discriminator",
        }
    }

    fn adapter_file(self) -> &'static str {
        match self {
            Self::Historical => "qwen21_edit_lokr_120step_1a853.safetensors",
            Self::CurrentFailedNative768 => "qwen21_edit_lokr.safetensors",
        }
    }

    fn training_file(self) -> &'static str {
        match self {
            Self::Historical => "qwen21_edit_lokr_120step_1a853_training.json",
            Self::CurrentFailedNative768 => "edit_lokr.json",
        }
    }

    fn protocol_file(self) -> Option<&'static str> {
        (self == Self::CurrentFailedNative768).then_some("edit-training-protocol.json")
    }

    fn adapter_identity(self) -> (&'static str, u64) {
        match self {
            Self::Historical => (inputs::DONOR, 6_759_417),
            Self::CurrentFailedNative768 => (current_inputs::ADAPTER, 6_759_417),
        }
    }

    fn training_identity(self) -> (&'static str, u64) {
        match self {
            Self::Historical => (inputs::TRAINING, 53_379),
            Self::CurrentFailedNative768 => (current_inputs::TRAINING_RECEIPT, 156_097),
        }
    }

    fn source_base(self) -> &'static str {
        match self {
            Self::Historical => inputs::BASE,
            Self::CurrentFailedNative768 => current_inputs::SOURCE_BASE,
        }
    }

    fn protocol_identity(self) -> &'static str {
        match self {
            Self::Historical => inputs::PROTOCOL,
            Self::CurrentFailedNative768 => current_inputs::PROTOCOL_RECEIPT,
        }
    }

    fn caption(self) -> &'static str {
        match self {
            Self::Historical => inputs::CAPTION,
            Self::CurrentFailedNative768 => current_inputs::CAPTION,
        }
    }

    fn shape(self) -> [i32; 3] {
        match self {
            Self::Historical => inputs::SHAPE,
            Self::CurrentFailedNative768 => current_inputs::SHAPE,
        }
    }

    fn target_hash(self) -> &'static str {
        match self {
            Self::Historical => "c2a972a40cb8fd33484ba0651ea50f922195992bdbb3fb1d0ad2a743854584f4",
            Self::CurrentFailedNative768 => current_inputs::TARGET_HASH,
        }
    }

    fn reference_hashes(self) -> [&'static str; 2] {
        match self {
            Self::Historical => inputs::REFERENCE_HASHES,
            Self::CurrentFailedNative768 => current_inputs::REFERENCE_HASHES,
        }
    }

    fn validate_header(self, value: &Value) {
        match self {
            Self::Historical => inputs::validate_header(value),
            Self::CurrentFailedNative768 => current_inputs::validate_header(value),
        }
    }

    fn validate_reference_order(self, hashes: &[String]) {
        match self {
            Self::Historical => inputs::validate_reference_order(hashes),
            Self::CurrentFailedNative768 => current_inputs::validate_reference_order(hashes),
        }
    }
}

struct PreparedBinding {
    source_candidate: String,
    manifest_path: PathBuf,
    manifest: Value,
    donor: PathBuf,
    training_path: PathBuf,
    protocol_path: Option<PathBuf>,
}

fn prepare_binding(binding: Binding) -> PreparedBinding {
    let source_candidate =
        std::env::var("GITHUB_SHA").expect("exact executing source identity required");
    assert_eq!(source_candidate.len(), 40);
    assert!(source_candidate
        .bytes()
        .all(|byte| byte.is_ascii_hexdigit()));
    let manifest_path = PathBuf::from(
        std::env::var(binding.manifest_env()).expect("strict failed donor manifest required"),
    );
    assert!(manifest_path.is_absolute());
    let manifest = json_file(&manifest_path);
    let directory = PathBuf::from(
        manifest["directory"]
            .as_str()
            .expect("absolute donor directory"),
    );
    assert!(directory.is_absolute());
    let donor = directory.join(binding.adapter_file());
    let training_path = directory.join(binding.training_file());
    let (adapter_sha, adapter_bytes) = binding.adapter_identity();
    let (training_sha, training_bytes) = binding.training_identity();
    assert_eq!(std::fs::metadata(&donor).unwrap().len(), adapter_bytes);
    assert_eq!(evidence::sha256_file(&donor), adapter_sha);
    assert_eq!(
        std::fs::metadata(&training_path).unwrap().len(),
        training_bytes
    );
    assert_eq!(evidence::sha256_file(&training_path), training_sha);
    let adapter_header = header(&donor);
    binding.validate_header(&adapter_header);
    let training = json_file(&training_path);
    let protocol_path = binding.protocol_file().map(|file| directory.join(file));
    match binding {
        Binding::Historical => inputs::validate_manifest(&manifest, &training),
        Binding::CurrentFailedNative768 => {
            let protocol_path = protocol_path.as_ref().expect("current protocol path");
            assert_eq!(std::fs::metadata(protocol_path).unwrap().len(), 91_960);
            assert_eq!(
                evidence::sha256_file(protocol_path),
                current_inputs::PROTOCOL_RECEIPT
            );
            current_inputs::validate_manifest(&manifest, &training, &json_file(protocol_path));
        }
    }
    PreparedBinding {
        source_candidate,
        manifest_path,
        manifest,
        donor,
        training_path,
        protocol_path,
    }
}

/// Declared before all native components, hence dropped after them on success
/// or unwind. The caller-thread owner remains live until this retirement ends.
struct RetireOnDrop;
impl Drop for RetireOnDrop {
    fn drop(&mut self) {
        crate::q4_diagnostic::retirement_boundary();
    }
}

fn assert_retired(baseline: u64) {
    assert!(
        mlx_rs::memory::get_active_memory() as u64 <= baseline + math::CACHE_CAP,
        "retired component still retains more than the priced CPU-cache allowance"
    );
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn json_file(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}
struct Tensor {
    shape: Vec<i32>,
    values: Vec<f32>,
    facts: Value,
}
impl Tensor {
    fn array(&self) -> Array {
        Array::from_slice(&self.values, &self.shape)
    }
    fn bytes(&self) -> u64 {
        self.values.len() as u64 * 4
    }
}
fn copy_tensor(a: &Array, name: &str, dir: &Path, cache: &mut math::Cache) -> Result<Tensor> {
    assert_eq!(
        a.dtype(),
        Dtype::Float32,
        "native output dtype must be observed, not silently recast"
    );
    let bytes = (a.size() as u64).checked_mul(4).unwrap();
    cache.reserve(bytes)?; // BEFORE contiguous materialization and owned CPU copy.
    let values = crate::q4_diagnostic::f32_values(a)?;
    assert_eq!(values.len() as u64 * 4, bytes);
    assert!(values.iter().all(|v| v.is_finite()));
    let file = format!("{name}.f32");
    let path = dir.join(&file);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut output = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    let mut hash = Sha256::new();
    for v in &values {
        let bytes = v.to_le_bytes();
        output.write_all(&bytes).unwrap();
        hash.update(bytes);
    }
    output.flush().unwrap();
    let facts = json!({"file":file,"sha256":format!("{:x}",hash.finalize()),"dtype":"Float32",
        "shape":a.shape(),"elements":values.len(),"bytes":bytes});
    Ok(Tensor {
        shape: a.shape().to_vec(),
        values,
        facts,
    })
}
struct Reference {
    pixels: Tensor,
    grid: [i32; 3],
    size: (u32, u32),
    latents: Tensor,
}
impl Reference {
    fn prepared(&self) -> PreparedReference {
        // The VAE phase is retired. encode_conditioning/joint_branch only consume
        // pixels and geometry; an empty sentinel fails if that contract changes.
        PreparedReference {
            pixel_values: self.pixels.array(),
            grid_thw: self.grid,
            size: self.size,
            vae_input: Array::from_slice(&[] as &[f32], &[0]),
        }
    }
}

fn phase_trace(
    out: &Path,
    stage: &str,
    active_limit: u64,
    cache: &math::Cache,
    rows: &mut Vec<Value>,
) {
    crate::q4_diagnostic::assert_owner();
    let active = mlx_rs::memory::get_active_memory() as u64;
    assert!(
        active <= active_limit,
        "active envelope exceeded at {stage}"
    );
    rows.push(json!({"stage":stage,"activeBytes":active,"cacheBytes":mlx_rs::memory::get_cache_memory(),
        "activePeakBytes":mlx_rs::memory::get_peak_memory(),"physicalBytes":crate::diagnostic_physical_footprint::phys_footprint().0,
        "cpuCacheLiveBytes":cache.live,"cpuCachePeakBytes":cache.peak,
        "allocatorSnapshotScope":"foreground non-atomic after existing evaluation or explicit retirement"}));
    evidence::write_json(out, "foreground-trace", &json!(rows));
}

fn header_bytes(
    root: &Path,
    component: &str,
    prefix: Option<&str>,
    float_width: Option<u64>,
) -> (u64, Value) {
    let headers = mlx_gen::gen_core::safetensors_path_tensor_headers(root.join(component)).unwrap();
    let selected = headers
        .iter()
        .filter(|h| prefix.is_none_or(|p| h.name.starts_with(p)))
        .collect::<Vec<_>>();
    assert!(!selected.is_empty());
    let rows=selected.iter().map(|h| {
        let elements=h.shape.iter().try_fold(1_u64,|v,d|v.checked_mul(*d as u64)).unwrap();
        // Retain stored parameters plus all possible F32 cast copies. Already
        // F32 arrays get a conservative extra slot too; never rely on aliasing.
        let bytes=if matches!(format!("{:?}", h.dtype).as_str(), "F16" | "BF16" | "F32" | "F64") {
            h.data_bytes.checked_add(elements.checked_mul(float_width.unwrap_or(0)).unwrap()).unwrap()
        } else {h.data_bytes};
        (bytes,json!({"name":h.name,"dtype":format!("{:?}",h.dtype),"shape":h.shape,"storedBytes":h.data_bytes,"pricedBytes":bytes}))
    }).collect::<Vec<_>>();
    (
        rows.iter()
            .try_fold(0_u64, |sum, r| sum.checked_add(r.0))
            .unwrap(),
        json!(rows.iter().map(|r| &r.1).collect::<Vec<_>>()),
    )
}
fn seal_closure(binding: Binding, dense: &Path, q4: &Path, adapter: &Path, out: &Path) -> u64 {
    assert!(
        !crate::quant::needs_load_time_quant(q4, Some(Quant::Q4)).unwrap(),
        "diagnostic requires actual pinned packed snapshot"
    );
    let (dense_lang, dl) = header_bytes(
        dense,
        "text_encoder",
        Some(crate::loader::TEXT_ENCODER_PREFIX),
        Some(4),
    );
    let (q4_lang, ql) = header_bytes(
        q4,
        "text_encoder",
        Some(crate::loader::TEXT_ENCODER_PREFIX),
        Some(4),
    );
    let (vision, v) = header_bytes(dense, "text_encoder", Some("model.visual."), Some(4));
    let (dense_dit, dd) = header_bytes(dense, "transformer", None, None);
    let (q4_dit, qd) = header_bytes(q4, "transformer", None, Some(4));
    let (vae, va) = header_bytes(dense, "vae", None, Some(4));
    let h = header(adapter);
    binding.validate_header(&h);
    let mut shapes = Vec::new();
    for (path, row) in h
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| k.ends_with("lokr_w1"))
    {
        let root = path.strip_suffix("lokr_w1").unwrap();
        let a = &h[format!("{root}lokr_w2_a")]["shape"];
        let b = &h[format!("{root}lokr_w2_b")]["shape"];
        shapes.push((
            row["shape"][0].as_u64().unwrap() * a[0].as_u64().unwrap(),
            row["shape"][1].as_u64().unwrap() * b[1].as_u64().unwrap(),
        ));
    }
    let price = crate::q4_diagnostic::math::price(&shapes).unwrap();
    let scratch = price.construction_scratch;
    let overhead = scratch + math::FREE_CACHE + math::CACHE_CAP;
    // Dense forward uses BF16 input/delta; still reserve all F32 delta widening
    // as a conservative envelope. Q4 has structured factors, no full delta.
    let phases = [
        dense_lang + vision + overhead,
        q4_lang + vision + overhead,
        dense_dit
            + vae
            + price.bf16_delta
            + price.f32_widening
            + crate::q4_diagnostic::math::FACTOR_ALLOWANCE
            + overhead,
        q4_dit + vae + crate::q4_diagnostic::math::FACTOR_ALLOWANCE + overhead,
        vae + overhead,
    ];
    let active = math::closure_active(&phases).unwrap();
    evidence::write_json(
        out,
        "phase-closure",
        &json!({"kind":"DIAGNOSTIC_ONLY","sealedBeforeNativeConstruction":true,
        "activeEnvelopeBytes":active,"activeFloorBytes":math::ACTIVE_FLOOR,"freeCacheAllowanceBytes":math::FREE_CACHE,
        "physicalBeforeBaselineBytes":active+math::FREE_CACHE,"cpuCacheCapBytes":math::CACHE_CAP,
        "phaseActiveBytes":phases,"scratchBytes":scratch,"denseDeltaBytes":price.bf16_delta,
        "allDenseDeltaF32WideningBytes":price.f32_widening,"factorAllowanceBytes":crate::q4_diagnostic::math::FACTOR_ALLOWANCE,
        "retirement":"per-target eval/default GPU stream synchronize/clear-cache; one DiT; sequential encoders/VAE",
        "headerClosure":{"denseLanguage":dl,"q4Language":ql,"fixedDenseVision":v,"denseDiT":dd,"q4DiT":qd,"fixedVAE":va}}),
    );
    active
}

/// Same public PEFT reconstruction/install dispatcher, one exact group at a time.
/// Header/hash/family/key checks precede this. Eager retirement bounds lazy
/// construction graphs without altering factors, scale, or default arithmetic.
fn install_retired(
    host: &mut QwenImage21Transformer,
    path: &Path,
    expected_sha256: &str,
    cache: &mut math::Cache,
) -> Result<()> {
    let w = Weights::from_file(path)?;
    w.materialize()?;
    let h = header(path);
    let mut raw = std::fs::File::open(path)?;
    let mut header_length = [0_u8; 8];
    raw.read_exact(&mut header_length)?;
    let payload = 8 + u64::from_le_bytes(header_length);
    let mut count = 0;
    for key in w.keys() {
        let a = w.require(key)?;
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_eq!(json!(a.shape()), h[key]["shape"]);
        let bytes = a.size() as u64 * 4;
        cache.reserve(bytes * 2)?; // native copy plus raw-byte comparison buffer.
        let actual = crate::q4_diagnostic::f32_values(a)?;
        let mut saved = vec![0_u8; bytes as usize];
        raw.seek(std::io::SeekFrom::Start(
            payload + h[key]["data_offsets"][0].as_u64().unwrap(),
        ))?;
        raw.read_exact(&mut saved)?;
        for (&value, bytes) in actual.iter().zip(saved.chunks_exact(4)) {
            assert!(value.is_finite());
            assert_eq!(
                value.to_le_bytes(),
                bytes,
                "exact raw master export/reload {key}"
            );
        }
        drop(actual);
        drop(saved);
        cache.release(bytes * 2);
        count += 1;
    }
    assert_eq!(count, 672);
    assert_eq!(
        evidence::sha256_file(path),
        expected_sha256,
        "file unchanged throughout native read"
    );
    crate::q4_diagnostic::retirement_boundary();
    let parsed = parse_lokr(&w)?;
    assert_eq!(parsed.groups.len(), 224);
    assert_eq!(parsed.alpha, 16.0);
    assert_eq!(parsed.rank, 16.0);
    for (name, group) in parsed.groups {
        let mut single = BTreeMap::new();
        for (suffix, array) in group {
            assert_eq!(array.dtype(), Dtype::Float32);
            single.insert(format!("{name}.{suffix}"), array);
        }
        let one = Weights::from_map(single.into_iter().collect());
        // parse_lokr defaults alpha=rank; exact donor alpha/rank=1. Defaults
        // without metadata are rank=1/alpha=1, the same ratio, asserted below.
        let p = parse_lokr(&one)?;
        assert_eq!(p.alpha / p.rank, 1.0);
        let report = apply_lokr(host, &one, 1.0)?;
        assert_eq!(report.applied, 1);
        assert!(report.unmatched_paths.is_empty());
        let linear = host
            .adaptable_mut(&name.split('.').collect::<Vec<_>>())
            .unwrap();
        assert_eq!(linear.adapters().len(), 1);
        assert!(
            matches!(&linear.adapters()[0],Adapter::Lokr{scale,..} if *scale==1.0)
                || matches!(&linear.adapters()[0], Adapter::LokrStructured { .. })
        );
        linear.materialize_adapters()?;
        crate::q4_diagnostic::retirement_boundary();
    }
    Ok(())
}

#[test]
#[ignore = "fixed real-weight 16-forward diagnostic; DIAGNOSTIC_ONLY, never image acceptance"]
fn diagnostic_dense_q4_conditioning_velocity() {
    run(Binding::Historical).unwrap();
}

#[test]
#[ignore = "fixed current-adapter teacher-forced diagnostic; zero renders/training; never acceptance"]
fn diagnostic_current_failed_adapter_dense_q4_conditioning_velocity() {
    run(Binding::CurrentFailedNative768).unwrap();
}

fn run(binding: Binding) -> Result<()> {
    // Hash, byte, provenance, header, source-base and FBD refusal is tensor-free
    // and intentionally precedes every MLX stream/model/array construction.
    let prepared_binding = prepare_binding(binding);
    let out = evidence::out_dir().join(binding.output_dir());
    std::fs::create_dir_all(&out)?;
    let src = evidence::edit_source(99, 768);
    let target = evidence::edit_transform(&src);
    let key = evidence::edit_key(512);
    for (name, img, expected) in [
        ("source99", &src, binding.reference_hashes()[0]),
        ("expected", &target, binding.target_hash()),
        ("palette", &key, binding.reference_hashes()[1]),
    ] {
        let path = out.join(format!("{name}.png"));
        img.save(&path).unwrap();
        assert_eq!(
            evidence::sha256_file(&path),
            expected,
            "frozen source bytes"
        );
    }
    let _owner = crate::q4_diagnostic::production_owner_scope();
    assert!(mlx_rs::task_local_default_stream().is_none());
    assert!(mlx_rs::Stream::new() == mlx_rs::Stream::gpu());
    let _retire_on_drop = RetireOnDrop;
    let source = &prepared_binding.source_candidate;
    let manifest_path = &prepared_binding.manifest_path;
    let manifest = &prepared_binding.manifest;
    let donor = &prepared_binding.donor;
    let (donor_sha, _) = binding.adapter_identity();
    let (training_sha, _) = binding.training_identity();
    let dense_spec = evidence::tier_spec("bf16", None);
    let q4_spec = evidence::tier_spec("q4", Some(Quant::Q4));
    let dense = crate::loader::snapshot_root(&dense_spec.weights)?;
    let q4 = crate::loader::snapshot_root(&q4_spec.weights)?;
    let active_limit = seal_closure(binding, dense, q4, donor, &out);
    let guard = evidence::Footprint::start(&out);
    let _current_cache_grant = if binding == Binding::CurrentFailedNative768 {
        Some(guard.admit_numeric_scoped(active_limit, math::FREE_CACHE))
    } else {
        guard.admit_numeric(active_limit, math::FREE_CACHE);
        None
    };
    guard.begin();
    let baseline_active = mlx_rs::memory::get_active_memory() as u64;
    let _bounds = crate::memory_strategy::AllocatorBounds::enter(
        active_limit - math::FREE_CACHE,
        math::FREE_CACHE,
    );
    let mut cache = math::Cache::default();
    // Retained masks/layout, serialized metadata, position ids, and small receipt
    // buffers are charged in addition to every copied full tensor below.
    cache.reserve(2 * 1024 * 1024)?;
    let mut trace = Vec::new();
    let limitations =
        if binding == Binding::CurrentFailedNative768 {
            json!(["fixed teacher-forced midpoint cannot override failed images",
            "same-activation residual capture samples 3 of 224 target modules",
            "CPU f64 aggregation and relaxed NAX observations are not a native arithmetic bound"])
        } else {
            json!([
                "fixed teacher-forced midpoint cannot override failed images",
                "DiT representation/dtype/native kernels confounded",
                "CPU f64 aggregation is not a native arithmetic bound"
            ])
        };
    let mut receipt = json!({"kind":"DIAGNOSTIC_ONLY","acceptanceEvidence":false,"accepted":false,
        "sourceCandidate":source,"sourceBase":binding.source_base(),
        "discriminatorProtocolSha256":binding.protocol_identity(),
        "runId":std::env::var("GITHUB_RUN_ID").expect("native run identity").parse::<u64>().unwrap(),
        "runAttempt":std::env::var("GITHUB_RUN_ATTEMPT").expect("native attempt identity").parse::<u64>().unwrap(),
        "inputManifestSha256":evidence::sha256_file(&manifest_path),"trainingProvenance":manifest["trainingProvenance"],
        "adapterSha256":donor_sha,"trainingReceiptSha256":training_sha,
        "forwardCount":0,"stateCount":4,"repeatCount":2,
        "adapterStrength":1,"sigma":0.5,"arithmeticBoundVerdict":math::ARITHMETIC,
        "nativePrecision":{"MLX_ENABLE_TF32":std::env::var("MLX_ENABLE_TF32").unwrap_or_else(|_|"unset: pinned MLX0.32 default1".to_owned()),
            "packedCompute":"Float32 with native defaults; relaxed NAX permission does not supply a proven mantissa/rounding bound",
            "denseCompute":"Bfloat16 intrinsic production compute", "conditioningEncodeRepeats":1,
            "forwardRepeats":2,"kernelExactnessClaimed":false},
        "vectors":[],"states":[],"cpuCachePeakBytes":0,"limitations":limitations});
    if binding == Binding::CurrentFailedNative768 {
        receipt["trainingSteps"] = json!(0);
        receipt["renderCount"] = json!(0);
        receipt["sceneWorksFbdCommit"] = manifest["sceneWorksFbdCommit"].clone();
        receipt["protocolReceiptSha256"] = json!(prepared_binding
            .protocol_path
            .as_ref()
            .map(|path| evidence::sha256_file(path)));
        receipt["trainingReceiptPath"] = json!(&prepared_binding.training_path);
    }
    evidence::write_json(&out, "receipt", &receipt);
    let req = GenerationRequest {
        prompt: binding.caption().to_owned(),
        width: 768,
        height: 768,
        steps: Some(8),
        seed: Some(24163),
        ..Default::default()
    };
    let scheduler = crate::loader::load_scheduler_config(dense)?;
    assert!(!crate::model::resolve_run_params(&scheduler, &req)?.use_negative);
    let vision = crate::loader::load_vision_config(dense)?.expect("fixed vision required");
    let image_bytes = (src.as_raw().len() + target.as_raw().len() + key.as_raw().len()) as u64;
    cache.reserve(image_bytes)?;
    let vae = crate::loader::load_vae(dense)?;
    let x0_array = crate::training::diagnostic_encode_latents(&vae, &evidence::to_image(target))?;
    let x0 = copy_tensor(&x0_array, "cpu-tensors/x0", &out, &mut cache)?;
    assert_eq!(x0.shape, binding.shape());
    drop(x0_array);
    let mut references = Vec::new();
    let mut reference_hashes = Vec::new();
    for (index, img) in [src, key].into_iter().enumerate() {
        let source_path = out.join(format!("prepared-source-{index}.png"));
        img.save(&source_path).unwrap();
        reference_hashes.push(evidence::sha256_file(&source_path));
        assert_eq!(reference_hashes[index], binding.reference_hashes()[index]);
        let rgb = evidence::to_image(img);
        let rgba = mlx_gen::RgbaImage {
            width: rgb.width,
            height: rgb.height,
            pixels: rgb
                .pixels
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect(),
        };
        let mut prepared = crate::reference::prepare_references(&[rgba], &vision)?;
        let reference = prepared.remove(0);
        assert_eq!(reference.size, (1024, 1024));
        let latents_array =
            crate::pipeline::encode_references(&vae, std::slice::from_ref(&reference))?.remove(0);
        let latents = copy_tensor(
            &latents_array,
            &format!("cpu-tensors/ref-{index}-latents"),
            &out,
            &mut cache,
        )?;
        assert_eq!(latents.shape, [1, 4096, 64]);
        drop(latents_array);
        let pixels = copy_tensor(
            &reference.pixel_values,
            &format!("cpu-tensors/ref-{index}-pixels"),
            &out,
            &mut cache,
        )?;
        references.push(Reference {
            pixels,
            grid: reference.grid_thw,
            size: reference.size,
            latents,
        });
        drop(reference);
        drop(prepared);
        crate::q4_diagnostic::retirement_boundary();
        phase_trace(
            &out,
            &format!("vae-reference-{index}"),
            active_limit,
            &cache,
            &mut trace,
        );
    }
    binding.validate_reference_order(&reference_hashes);
    assert_eq!(references.len(), 2);
    drop(vae);
    cache.release(image_bytes);
    crate::q4_diagnostic::retirement_boundary();
    assert_retired(baseline_active);
    phase_trace(&out, "vae-retired", active_limit, &cache, &mut trace);
    let noise_array = crate::pipeline::create_noise(24163, 768, 768, 64)?;
    let noise = copy_tensor(&noise_array, "cpu-tensors/noise", &out, &mut cache)?;
    assert_eq!(noise.shape, binding.shape());
    let (xt_array, target_array) =
        crate::training::diagnostic_midpoint_batch(&x0.array(), &noise_array)?;
    let xt = copy_tensor(&xt_array, "cpu-tensors/xt", &out, &mut cache)?;
    let target = copy_tensor(
        &target_array,
        "cpu-tensors/target-velocity",
        &out,
        &mut cache,
    )?;
    drop(noise_array);
    drop(xt_array);
    drop(target_array);
    crate::q4_diagnostic::retirement_boundary();
    let tokenizer = crate::loader::load_tokenizer(dense)?;
    let drop_count = crate::text_encoder::system_prompt_drop_count(&tokenizer)?;
    let mut conditioning = Vec::new();
    let mut layouts = Vec::new();
    let mut masks = Vec::new();
    for (index, root) in [dense, q4].into_iter().enumerate() {
        let te = crate::loader::diagnostic_language_with_fixed_vision(root, dense)?;
        let prepared = references
            .iter()
            .map(Reference::prepared)
            .collect::<Vec<_>>();
        let encoded =
            te.encode_conditioning(&tokenizer, binding.caption(), drop_count, &prepared)?;
        let branch = crate::pipeline::joint_branch(&encoded, &prepared, 768, 768)?;
        assert_eq!(branch.text.dtype(), Dtype::Float32);
        conditioning.push(copy_tensor(
            &branch.text,
            &format!("cpu-tensors/conditioning-{index}"),
            &out,
            &mut cache,
        )?);
        layouts.push(branch.layout.clone());
        masks.push(encoded.image_pad_mask.clone());
        drop(branch);
        drop(encoded);
        drop(prepared);
        drop(te);
        crate::q4_diagnostic::retirement_boundary();
        assert_retired(baseline_active);
        phase_trace(
            &out,
            &format!("conditioning-{index}-retired"),
            active_limit,
            &cache,
            &mut trace,
        );
    }
    assert_eq!(layouts[0], layouts[1]);
    assert_eq!(masks[0], masks[1]);
    assert_eq!(conditioning[0].shape, conditioning[1].shape);
    let (conditioning_mae, conditioning_rms, conditioning_max) =
        math::difference(&conditioning[0].values, &conditioning[1].values)?;
    let positions = layouts[0].position_ids();
    receipt["inputs"] = json!({"caption":binding.caption(),"referenceOrder":["source99","palette"],"fit":[[1024,1024],[1024,1024]],
        "x0":x0.facts,"noise":noise.facts,"xt":xt.facts,"targetVelocity":target.facts,
        "references":references.iter().map(|r|json!({"pixels":r.pixels.facts,"latents":r.latents.facts,"grid":r.grid,"size":r.size})).collect::<Vec<_>>(),
        "conditioning":conditioning.iter().map(|c|&c.facts).collect::<Vec<_>>(),"imagePadMask":masks[0],
        "layout":format!("{:?}",layouts[0]),"positionIdsSha256":sha(&serde_json::to_vec(&positions).unwrap()),
        "fixedVisionSource":dense,"fixedVAESource":dense,"conditioningRoots":[dense,q4]});
    if binding == Binding::CurrentFailedNative768 {
        receipt["inputs"]["conditioningDenseVsQ4"] = json!({
            "meanAbsoluteDifference":conditioning_mae,
            "rootMeanSquareDifference":conditioning_rms,
            "maxAbsoluteDifference":conditioning_max
        });
    }
    for r in &mut references {
        cache.release(r.pixels.bytes());
        r.pixels.values.clear();
        r.pixels.values.shrink_to_fit();
    }
    let mut vectors = Vec::new();
    let mut state_receipts = Vec::new();
    let mut gains = [[0.0; 2]; 4];
    for (state, (lang, dit)) in [(0, 0), (1, 1), (0, 1), (1, 0)].into_iter().enumerate() {
        let mut residual_capture = Value::Null;
        let root = if dit == 0 { dense } else { q4 };
        let mut model = crate::loader::load_transformer(root)?;
        for path in model.adaptable_paths() {
            assert!(
                model
                    .adaptable_mut(&path.split('.').collect::<Vec<_>>())
                    .unwrap()
                    .adapters()
                    .is_empty(),
                "base velocity must use a bare frozen DiT"
            );
        }
        assert_eq!(
            model.compute_dtype(),
            if dit == 0 {
                Dtype::Bfloat16
            } else {
                Dtype::Float32
            }
        );
        let text = conditioning[lang].array();
        let latent_refs = references
            .iter()
            .map(|r| r.latents.array())
            .collect::<Vec<_>>();
        let target_input = xt.array();
        let images = crate::pipeline::joint_images(&latent_refs, &target_input);
        let mut actual = Vec::new();
        for adapted in [false, true] {
            if adapted {
                install_retired(&mut model, donor, donor_sha, &mut cache)?;
            }
            for repeat in 0..2 {
                let velocity = if binding == Binding::CurrentFailedNative768
                    && dit == 1
                    && adapted
                    && repeat == 0
                {
                    let (velocity, captures) =
                        crate::q4_diagnostic::capture_same_activation_residuals(donor, || {
                            model.forward_joint(&text, &images, 0.5, &layouts[lang])
                        })?;
                    residual_capture = json!({
                        "scope":"same F32 activation rows; representative first-block sample only",
                        "sampledTargetModules":3,"totalTargetModules":224,
                        "acceptanceEvidence":false,"numericPassClaim":false,
                        "conditioning":if lang==0 {"denseBF16"} else {"Q4"},
                        "dit":"Q4","repeat":repeat,"captures":captures
                    });
                    velocity
                } else {
                    model.forward_joint(&text, &images, 0.5, &layouts[lang])?
                };
                assert_eq!(velocity.shape(), binding.shape());
                let mut copied = copy_tensor(
                    &velocity,
                    &format!(
                        "velocities/state-{state}-repeat-{repeat}-{}",
                        if adapted { "adapted" } else { "base" }
                    ),
                    &out,
                    &mut cache,
                )?;
                copied.facts["state"] = json!(state);
                copied.facts["repeat"] = json!(repeat);
                copied.facts["adapted"] = json!(adapted);
                vectors.push(copied.facts.clone());
                actual.push(copied);
                drop(velocity);
                crate::q4_diagnostic::retirement_boundary();
                phase_trace(
                    &out,
                    &format!("state-{state}-repeat-{repeat}-adapted-{adapted}"),
                    active_limit,
                    &cache,
                    &mut trace,
                );
                receipt["vectors"] = json!(vectors);
                receipt["forwardCount"] = json!(vectors.len());
                receipt["cpuCachePeakBytes"] = json!(cache.peak);
                evidence::write_json(&out, "receipt", &receipt);
            }
        }
        let mut pairs = Vec::new();
        for repeat in 0..2 {
            let m = math::metrics(
                &actual[repeat].values,
                &actual[repeat + 2].values,
                &target.values,
            )?;
            gains[state][repeat] = m.gain;
            pairs.push(json!({"repeat":repeat,"baseError":m.base_error,"adaptedError":m.adapted_error,"learnedGain":m.gain,
                "projection":m.projection,"deltaNorm2":m.delta_norm2,"cpuIdentityResidual":m.identity_residual,"cpuIdentityBound":m.identity_bound}));
        }
        let (bm, br) = math::variability(&actual[0].values, &actual[1].values)?;
        let (am, ar) = math::variability(&actual[2].values, &actual[3].values)?;
        let mut state_receipt = json!({"state":state,"conditioning":if lang==0 {"denseBF16"} else {"Q4"},
            "dit":if dit==0 {"denseBF16"} else {"Q4"},"computeDtype":format!("{:?}",model.compute_dtype()),
            "pairs":pairs,"repeatVariability":{"baseMaxAbs":bm,"baseRms":br,"adaptedMaxAbs":am,"adaptedRms":ar},
            "gainInterval":[gains[state][0].min(gains[state][1]),gains[state][0].max(gains[state][1])]});
        if binding == Binding::CurrentFailedNative768 {
            state_receipt["structuredVsDirectResidual"] = residual_capture;
        }
        state_receipts.push(state_receipt);
        drop(images);
        drop(text);
        drop(latent_refs);
        drop(target_input);
        drop(model);
        for v in actual {
            cache.release(v.bytes());
        }
        crate::q4_diagnostic::retirement_boundary();
        assert_retired(baseline_active);
        phase_trace(
            &out,
            &format!("state-{state}-retired"),
            active_limit,
            &cache,
            &mut trace,
        );
        receipt["states"] = json!(state_receipts);
        evidence::write_json(&out, "receipt", &receipt);
    }
    let (peak, physical) = guard.end();
    assert!(
        peak <= active_limit,
        "full active high-water exceeds sealed envelope"
    );
    assert_eq!(vectors.len(), 16);
    assert_eq!(state_receipts.len(), 4);
    let residual_capture_count = state_receipts
        .iter()
        .filter(|state| !state["structuredVsDirectResidual"].is_null())
        .count();
    assert_eq!(
        residual_capture_count,
        if binding == Binding::CurrentFailedNative768 {
            2
        } else {
            0
        }
    );
    if binding == Binding::CurrentFailedNative768 {
        receipt["residualCaptureCount"] = json!(residual_capture_count);
    }
    receipt["localizationObservation"] = json!(math::localization(&gains));
    receipt["cpuCachePeakBytes"] = json!(cache.peak);
    receipt["activePeakBytes"] = json!(peak);
    receipt["physicalPeakBytes"] = json!(physical);
    receipt["activeEnvelopeBytes"] = json!(active_limit);
    receipt["nativeRawMasterReload"] = json!({"verifiedTensorsPerState":[672,672,672,672],
        "exactRawValuesShapesDtypes":true,"originalFileSha256":donor_sha,
        "scope":"all four installs compared to the immutable raw payload; production residual representations differ by tier"});
    receipt["phaseTraceComplete"] = json!(trace.len() == 25);
    assert_eq!(trace.len(), 25);
    evidence::write_json(&out, "receipt", &receipt);
    Ok(())
}
