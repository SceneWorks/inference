//! One captured-input diagnostic. Never training, generation, donor or quality acceptance.
use std::path::{Path, PathBuf};

use mlx_gen::Result;
use mlx_gen::adapters::{Adapter, build_lokr_factors, reconstruct_lokr_delta};
use mlx_gen::weights::Weights;
use mlx_rs::ops::matmul;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::q4_real_weights_support as evidence;

const ACTIVE: u64 = 1 << 30;
const FREE_CACHE: u64 = 64 << 20;
const MAX_PHYSICAL: u64 = 4 << 30;
const PREFIX: &str = "transformer_blocks.0.attn.to_k";

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_json(root: &Path, name: &str, value: &Value) -> Result<()> {
    std::fs::write(root.join(name), serde_json::to_vec_pretty(value).unwrap())?;
    Ok(())
}

fn read_f32(root: &Path, entry: &Value) -> Result<Array> {
    let bytes = std::fs::read(root.join(entry["file"].as_str().unwrap()))?;
    assert_eq!(sha(&bytes), entry["sha256"].as_str().unwrap());
    let shape: Vec<i32> = entry["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| i32::try_from(x.as_i64().unwrap()).unwrap())
        .collect();
    assert_eq!(
        bytes.len(),
        shape.iter().map(|x| *x as usize).product::<usize>() * 4
    );
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    assert!(values.iter().all(|x| x.is_finite()));
    Ok(Array::from_slice(&values, &shape))
}

fn capture(root: &Path, name: &str, array: &Array) -> Result<Value> {
    let values = crate::q4_diagnostic::f32_values(array)?;
    assert!(values.iter().all(|x| x.is_finite()));
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let file = format!("{name}.f32");
    std::fs::write(root.join(&file), &bytes)?;
    Ok(
        json!({"file":file,"shape":array.shape(),"dtype":format!("{:?}",array.dtype()),
        "elements":values.len(),"bytes":bytes.len(),"sha256":sha(&bytes)}),
    )
}

struct Retire;
impl Drop for Retire {
    fn drop(&mut self) {
        crate::q4_diagnostic::retirement_boundary();
    }
}

#[test]
#[ignore = "one same-input linear diagnostic; zero DiT forwards/training/renders; never acceptance"]
fn diagnostic_current_q4_single_linear() {
    run().unwrap();
}

fn run() -> Result<()> {
    // Tensor-free admission binding: build source proof, prepared exact slices, and captured input.
    let root = PathBuf::from(std::env::var("QWEN_SINGLE_LINEAR_INPUT").unwrap());
    let out = PathBuf::from(std::env::var("QWEN_SINGLE_LINEAR_OUTPUT").unwrap());
    assert!(!out.exists(), "exclusive output required");
    let prepared_bytes = std::fs::read(root.join("prepared.json"))?;
    assert_eq!(
        sha(&prepared_bytes),
        std::env::var("QWEN_SINGLE_LINEAR_PREPARED_SHA").unwrap()
    );
    let prepared: Value = serde_json::from_slice(&prepared_bytes).unwrap();
    assert_eq!(prepared["kind"], "SINGLE_LINEAR_DIAGNOSTIC_ONLY");
    assert_eq!(prepared["acceptance"], false);
    assert_eq!(prepared["module"], PREFIX);
    assert_eq!(prepared["bits"], 4);
    assert_eq!(prepared["groupSize"], 64);
    assert_eq!(prepared["input"]["shape"], json!([1, 65, 4096]));
    assert_eq!(prepared["originalActivationDtype"], "Float32");
    assert_eq!(prepared["replayActivationDtype"], "Float32");
    for entry in prepared["files"].as_array().unwrap() {
        let name = entry["file"].as_str().unwrap();
        assert_eq!(
            Path::new(name).file_name().and_then(|x| x.to_str()),
            Some(name)
        );
        assert!(entry["bytes"].as_u64().unwrap() <= 16 << 20);
        let bytes = std::fs::read(root.join(entry["file"].as_str().unwrap()))?;
        assert_eq!(bytes.len() as u64, entry["bytes"].as_u64().unwrap());
        assert_eq!(sha(&bytes), entry["sha256"].as_str().unwrap());
    }
    let build_path = PathBuf::from(std::env::var("QWEN_SINGLE_LINEAR_BUILD_PROOF").unwrap());
    let build_bytes = std::fs::read(&build_path)?;
    assert_eq!(
        sha(&build_bytes),
        std::env::var("QWEN_SINGLE_LINEAR_BUILD_PROOF_SHA").unwrap()
    );
    let build: Value = serde_json::from_slice(&build_bytes).unwrap();
    assert_eq!(build["strictFloat32DispatchSourceVerified"], true);
    let mode = std::env::var("QWEN_SINGLE_LINEAR_MODE").unwrap();
    match mode.as_str() {
        "default" => assert!(std::env::var_os("MLX_ENABLE_TF32").is_none()),
        "strict" => assert_eq!(std::env::var("MLX_ENABLE_TF32").unwrap(), "0"),
        _ => panic!("unknown precision process"),
    }
    assert_eq!(build["replayM"], 65);
    // Existing linked MLX private symbol; transport verifies it in the actual archive before launch.
    #[cfg(target_os = "macos")]
    unsafe extern "C" {
        #[link_name = "_ZN3mlx4core5metal16is_nax_availableEv"]
        fn linked_nax_available() -> bool;
    }
    #[cfg(target_os = "macos")]
    let hardware_nax = unsafe { linked_nax_available() };
    #[cfg(not(target_os = "macos"))]
    let hardware_nax = false;
    assert!(hardware_nax, "default packed NAX comparison unavailable");
    std::fs::create_dir(&out)?;
    let _owner = crate::q4_diagnostic::production_owner_scope();
    let cache_before = mlx_rs::memory::set_cache_limit(0);
    assert_eq!(mlx_rs::memory::set_cache_limit(cache_before), 0);
    let memory_before = mlx_rs::memory::get_memory_limit();
    let guard = evidence::Footprint::start(&out);
    let cache_grant = guard.admit_numeric_scoped(ACTIVE, FREE_CACHE);
    let admission: Value =
        serde_json::from_slice(&std::fs::read(out.join("numeric-physical-admission.json"))?)
            .unwrap();
    assert!(
        admission["reservedPhysicalCeilingBytes"].as_u64().unwrap() <= MAX_PHYSICAL,
        "diagnostic physical envelope exceeds 4GiB"
    );
    guard.begin();
    let mut outputs = json!({});
    let operations;
    {
        let _bounds = crate::memory_strategy::AllocatorBounds::enter(ACTIVE, FREE_CACHE);
        let _retire = Retire;
        let x = read_f32(&root, &prepared["input"])?;
        let w1 = read_f32(&root, &prepared["w1"])?;
        let w2a = read_f32(&root, &prepared["w2a"])?;
        let w2b = read_f32(&root, &prepared["w2b"])?;
        // Exactly one low-rank factor GEMM. Both production helpers accept this computed full W2.
        let w2 = matmul(&w2a, &w2b)?;
        eval([&w2])?;
        outputs["rawW2"] = capture(&out, "raw-w2", &w2)?;
        let factors = build_lokr_factors(
            1.0,
            &[4096, 4096],
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
            Dtype::Bfloat16,
        )?
        .unwrap();
        eval([&factors.w1, &factors.w2])?;
        outputs["storedW1"] = capture(&out, "stored-w1", &factors.w1)?;
        outputs["storedW2"] = capture(&out, "stored-w2", &factors.w2)?;
        let structured = factors.residual(&x)?;
        eval([&structured])?;
        assert_eq!(structured.dtype(), Dtype::Float32);
        outputs["structured"] = capture(&out, "structured", &structured)?;
        let weights = Weights::from_file(root.join("selected.safetensors"))?;
        let linear = mlx_gen::quant::lin(&weights, PREFIX, false, 64)?;
        assert_eq!(linear.base_shape(), vec![4096, 4096]);
        let (packed, scales, biases, bias, gs, bits) = linear.quantized_params().unwrap();
        assert!(bias.is_none());
        assert_eq!((gs, bits), (64, 4));
        eval([packed, scales, biases])?;
        // Host-loaded quantization values must exactly match the prepared selected payload.
        assert_eq!(packed.dtype(), Dtype::Uint32);
        let packed_bytes: Vec<u8> = packed
            .as_slice::<u32>()
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        assert_eq!(
            sha(&packed_bytes),
            prepared["selected"][format!("{PREFIX}.weight")]["sha256"]
                .as_str()
                .unwrap()
        );
        outputs["loadedScales"] = capture(&out, "loaded-scales", scales)?;
        outputs["loadedBiases"] = capture(&out, "loaded-biases", biases)?;
        let base = linear.forward(&x)?;
        eval([&base])?;
        assert_eq!(base.dtype(), Dtype::Float32);
        outputs["base"] = capture(&out, "base", &base)?;
        operations = if mode == "strict" {
            let delta = reconstruct_lokr_delta(
                16.0,
                16.0,
                &[4096, 4096],
                Some(&w1),
                None,
                None,
                Some(&w2),
                None,
                None,
                Dtype::Bfloat16,
            )?;
            eval([&delta])?;
            // Stream coefficients in 64-row chunks; never retain its whole 64MiB f32 expansion.
            use mlx_rs::ops::indexing::IndexOp;
            let mut file = std::fs::File::create(out.join("delta.f32"))?;
            use std::io::Write;
            let mut digest = Sha256::new();
            for row in (0..4096).step_by(64) {
                let values = crate::q4_diagnostic::f32_values(&delta.index((row..row + 64, ..)))?;
                let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                digest.update(&bytes);
                file.write_all(&bytes)?;
            }
            outputs["delta"] = json!({"file":"delta.f32","shape":[4096,4096],
                "dtype":"Bfloat16 exported exactly as Float32","bytes":67108864,
                "sha256":format!("{:x}",digest.finalize())});
            let direct = Adapter::Lokr { delta, scale: 1.0 }.residual(&x)?;
            eval([&direct])?;
            outputs["direct"] = capture(&out, "direct", &direct)?;
            5
        } else {
            4
        };
    }
    let (active_peak, physical_peak) = guard.end();
    drop(cache_grant);
    drop(guard);
    let cache_after = mlx_rs::memory::set_cache_limit(0);
    assert_eq!(mlx_rs::memory::set_cache_limit(cache_after), 0);
    let memory_after = mlx_rs::memory::get_memory_limit();
    assert_eq!(cache_after, cache_before);
    assert_eq!(memory_after, memory_before);
    write_json(
        &out,
        "receipt.json",
        &json!({"schema":"sc24163-single-linear-native-receipt-v1",
        "kind":"SINGLE_LINEAR_DIAGNOSTIC_ONLY","accepted":false,"acceptance":false,
        "source":std::env::var("GITHUB_SHA").unwrap(),"runId":std::env::var("GITHUB_RUN_ID").unwrap(),
        "runAttempt":std::env::var("GITHUB_RUN_ATTEMPT").unwrap(),"runner":std::env::var("RUNNER_NAME").unwrap(),
        "mode":mode,"module":PREFIX,"gemmQmmEvaluations":operations,"ditForwards":0,
        "trainingSteps":0,"renderCount":0,"preparedSha256":sha(&prepared_bytes),
        "compiledSourceProofSha256":sha(&build_bytes),"activePeakBytes":active_peak,
        "physicalPeakBytes":physical_peak,"cpuPayloadRetainedStaticUpperBoundBytes":33554432,"cpuPayloadBoundMeasured":false,"retiredBeforeGuardDrop":true,"watchdogJoined":true,
        "originalActivationDtype":"Float32","replayActivationDtype":"Float32","hardwareNaxAvailable":hardware_nax,
        "originalActivationShape":[1,10583,4096],"replayShape":[1,65,4096],"replayConstruction":"capturedRow[i%3]",
        "beforeCacheLimitBytes":cache_before,"restoredCacheLimitBytes":cache_after,
        "beforeMemoryLimitBytes":memory_before,"restoredMemoryLimitBytes":memory_after,
        "outputs":outputs}),
    )?;
    Ok(())
}
