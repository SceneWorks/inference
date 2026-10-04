//! Test-binary-only diagnostic. No release selector or adapter policy changes.
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Mutex;

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear, Adapter};
use mlx_gen::train::lora::{build_lokr_targets, install_training_lokr, LoraParams};
use mlx_gen::{GenerationRequest, LoadSpec, Quant, Result};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{concatenate_axis, matmul};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};
use serde_json::{json, Value};

use crate::q4_real_weights_support as evidence;
use crate::transformer::QwenImage21Transformer;
#[path = "q4_numeric_math.rs"]
pub(crate) mod math;

const DONOR_HASH: &str = "c233129e9a64e384850331c9804b5b496d5208c08fe34aca4c680920fddb03d7";
const PROTOCOL_HASH: &str = "1e42ed9cd80f45712cdb75f6ee63d93ae4afe7b310f6bfbc9f645f8b2c634fab";
const RECEIPT_HASH: &str = "fa5b6234870f6ac0784f893067524407178ee9ee1678c50bc6bd6713d008e333";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Variant {
    Production,
    DirectTrainingDeltaRepresentation,
}
thread_local! {
    static VARIANT: Cell<Variant> = const { Cell::new(Variant::Production) };
    static CAPTURE: RefCell<Option<Capture>> = const { RefCell::new(None) };
}
static OWNER: Mutex<Option<std::thread::ThreadId>> = Mutex::new(None);
pub(crate) fn assert_owner() {
    let owner = *OWNER.lock().unwrap();
    if let Some(owner) = owner {
        assert_eq!(
            owner,
            std::thread::current().id(),
            "test diagnostic load/forward moved off its caller thread"
        );
    }
}
struct VariantScope(Variant);
impl VariantScope {
    fn enter(variant: Variant) -> Self {
        let mut owner = OWNER.lock().unwrap();
        if owner.is_some() {
            drop(owner);
            panic!("nested diagnostic scope refused");
        }
        *owner = Some(std::thread::current().id());
        drop(owner);
        Self(VARIANT.with(|v| v.replace(variant)))
    }
}
impl Drop for VariantScope {
    fn drop(&mut self) {
        CAPTURE.with(|c| {
            c.borrow_mut().take();
        });
        VARIANT.with(|v| v.set(self.0));
        *OWNER.lock().unwrap() = None;
    }
}

#[test]
#[ignore = "process-scoped test: run alone so unrelated parallel model tests cannot enter its scope"]
fn scoped_variant_refuses_nesting_cross_thread_and_resets_after_panic() {
    let panic = std::panic::catch_unwind(|| {
        let _scope = VariantScope::enter(Variant::DirectTrainingDeltaRepresentation);
        assert!(std::panic::catch_unwind(|| VariantScope::enter(Variant::Production)).is_err());
        assert!(std::thread::spawn(assert_owner).join().is_err());
        assert_eq!(
            VARIANT.with(Cell::get),
            Variant::DirectTrainingDeltaRepresentation
        );
        CAPTURE.with(|c| {
            *c.borrow_mut() = Some(Capture {
                params: LoraParams::new(),
                rows: BTreeMap::new(),
            })
        });
        panic!("intentional diagnostic unwind");
    });
    let panic = panic.expect_err("the intentional scope panic must be observed");
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"intentional diagnostic unwind")
    );
    assert!(OWNER.lock().unwrap().is_none());
    assert_eq!(VARIANT.with(Cell::get), Variant::Production);
    assert!(!capture_enabled());
    let _scope = VariantScope::enter(Variant::Production);
    assert_owner();
}

/// Extra reservation includes all possible F32 casts and construction retirement;
/// the ordinary contract already accounts for one donor-file overlay.
pub(crate) fn resident_surcharge() -> u64 {
    assert_owner();
    VARIANT.with(|v| match v.get() {
        Variant::Production => 0,
        Variant::DirectTrainingDeltaRepresentation => {
            math::EXPECTED_ELEMENTS * 6 + math::FACTOR_ALLOWANCE - 6_759_417 + 5_536_481_280
        }
    })
}

pub(crate) fn materialization_bounds() -> Option<crate::memory_strategy::AllocatorBounds> {
    (VARIANT.with(|v| v.get()) == Variant::DirectTrainingDeltaRepresentation).then(|| {
        crate::memory_strategy::AllocatorBounds::enter(
            math::BASE_RESIDENT + 6_759_417 + resident_surcharge(),
            math::TRANSIENT,
        )
    })
}

fn read_params(path: &Path) -> Result<(LoraParams, Vec<String>)> {
    let weights = mlx_gen::weights::Weights::from_file(path)?;
    let mut params = LoraParams::new();
    let mut paths = std::collections::BTreeSet::new();
    for key in weights.keys() {
        let Some((path, suffix)) = key.rsplit_once('.') else {
            return Err("numeric factor key lacks module/suffix".into());
        };
        if !matches!(suffix, "lokr_w1" | "lokr_w2_a" | "lokr_w2_b") {
            return Err(format!("numeric unsupported factor {key}").into());
        }
        let array = weights.require(key)?.clone();
        if array.dtype() != Dtype::Float32 {
            return Err(format!("numeric raw master {key} is not F32").into());
        }
        params.insert(Rc::from(key), array);
        paths.insert(path.to_owned());
    }
    if params.len() != 672 || paths.len() != 224 {
        return Err("numeric donor is not exact672factor/224target file".into());
    }
    Ok((params, paths.into_iter().collect()))
}

pub(crate) fn install_variant(host: &mut QwenImage21Transformer, spec: &LoadSpec) -> Result<()> {
    assert_owner();
    if VARIANT.with(|v| v.get()) != Variant::DirectTrainingDeltaRepresentation {
        return Ok(());
    }
    assert!(mlx_rs::task_local_default_stream().is_none());
    assert!(
        mlx_rs::Stream::new() == mlx_rs::Stream::gpu(),
        "construction stream must be the default GPU stream"
    );
    assert_eq!(host.compute_dtype(), Dtype::Float32);
    assert!(host
        .adaptable_mut(&["img_in"])
        .unwrap()
        .quantized_params()
        .is_some());
    assert_eq!(spec.adapters.len(), 1);
    let adapter = &spec.adapters[0];
    assert!(matches!(adapter.scale, 0.0 | 1.0));
    assert_eq!(evidence::sha256_file(&adapter.path), DONOR_HASH);
    let (params, paths) = read_params(&adapter.path)?;
    // Reuse the public training target descriptors and reconstruction verbatim.
    // Initializer masters are dropped; only the verified saved F32 masters install.
    let (targets, initializer) = build_lokr_targets(host, &paths, 16, -1, 42)?;
    drop(initializer);
    assert_eq!(targets.len(), 224);
    for (target, path) in targets.iter().zip(&paths) {
        install_training_lokr(
            host,
            &params,
            std::slice::from_ref(target),
            16.0 * adapter.scale,
            16.0,
            Dtype::Bfloat16,
        )?;
        let linear = host
            .adaptable_mut(&path.split('.').collect::<Vec<_>>())
            .unwrap();
        let Adapter::Lokr { delta, .. } = &linear.adapters()[0] else {
            panic!("public training installer did not produce materialized LoKr");
        };
        assert_eq!(delta.dtype(), Dtype::Bfloat16);
        // Per-target completion releases raw construction graphs, not224graphs
        // retained until one final eval. Pricing still includes11retirement slots.
        eval([delta])?;
        retirement_boundary();
    }
    Ok(())
}

fn retirement_boundary() {
    // Exact mlx-rs48ff5e7 vendored mlx-c stream.h/stream.cpp ABI. The loaded
    // library/source identity is separately captured as MLXv0.32.0 in the lane.
    // This boundary runs ONLY while constructing the alternate test overlay.
    assert!(mlx_rs::task_local_default_stream().is_none());
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct CStream {
        ctx: *mut std::ffi::c_void,
    }
    extern "C" {
        fn mlx_default_gpu_stream_new() -> CStream;
        fn mlx_synchronize(stream: CStream) -> i32;
        fn mlx_stream_free(stream: CStream) -> i32;
    }
    // SAFETY: the owned handle comes from the matching linked mlx-c library,
    // is synchronized before being freed once, and is never shared with threads.
    let (sync, free) = unsafe {
        let stream = mlx_default_gpu_stream_new();
        assert!(
            !stream.ctx.is_null(),
            "materialization stream creation failed"
        );
        let sync = mlx_synchronize(stream);
        (sync, mlx_stream_free(stream))
    };
    assert_eq!(sync, 0, "materialization synchronization failed");
    assert_eq!(free, 0, "materialization stream release failed");
    mlx_rs::memory::clear_cache();
}

fn f32_values(array: &Array) -> Result<Vec<f32>> {
    let a = array.as_dtype(Dtype::Float32)?;
    eval([&a])?;
    Ok(a.as_slice::<f32>().to_vec())
}
struct Capture {
    params: LoraParams,
    rows: BTreeMap<String, Value>,
}
pub(crate) fn capture_enabled() -> bool {
    CAPTURE.with(|c| c.borrow().is_some())
}

/// Called only from cfg(test) sites; CAPTURE is empty during all four renders.
/// The separate numeric forward records three row vectors, not a full graph.
pub(crate) fn capture_linear(path: &str, linear: &AdaptableLinear, x: &Array) -> Result<()> {
    assert_owner();
    CAPTURE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(capture) = state.as_mut() else { return Ok(()); };
        let path = format!("transformer_blocks.0.{path}");
        if capture.rows.contains_key(&path) { return Ok(()); }
        assert_eq!(x.dtype(), Dtype::Float32);
        assert_eq!(x.shape()[0], 1);
        assert_eq!(linear.adapters().len(), 1);
        let s = x.shape()[1];
        let row_indices = [0, s / 2, s - 1];
        let slices: Vec<_> = row_indices.iter().map(|&r| x.index((.., r..r+1, ..))).collect();
        let xs = concatenate_axis(&slices.iter().collect::<Vec<_>>(),1)?;
        let current = linear.adapters()[0].residual(&xs)?;
        assert_eq!(current.dtype(), Dtype::Float32);
        let get = |suffix: &str| capture.params.get(format!("{path}.{suffix}").as_str()).unwrap();
        let w1 = get("lokr_w1");
        let w2a = get("lokr_w2_a");
        let w2b = get("lokr_w2_b");
        let w2 = matmul(w2a,w2b)?;
        let a=w1.shape()[0]; let c=w1.shape()[1];
        let b=w2.shape()[0]; let d=w2.shape()[1];
        let delta = mlx_gen::adapters::reconstruct_lokr_delta(16.0,16.0,
            &[a*b,c*d],Some(w1),None,None,None,Some(w2a),Some(w2b),Dtype::Bfloat16)?;
        let direct = Adapter::Lokr { delta:delta.clone(),scale:1.0 }.residual(&xs)?;
        let components=16.min(a*b);
        let current_values=f32_values(&current.index((..,..,0..components)))?;
        let direct_values=f32_values(&direct.index((..,..,0..components)))?;
        let inputs=f32_values(&xs)?;
        let raw_w1=f32_values(w1)?; let raw_w2=f32_values(&w2)?;
        let stored_w1=f32_values(&w1.as_dtype(Dtype::Bfloat16)?)?;
        let stored_w2=f32_values(&w2.as_dtype(Dtype::Bfloat16)?)?;
        let coefficients=f32_values(&delta.index((0..components,..)))?;
        let mut checks=Vec::new();
        let mut inner_checks=true;
        let raw_a=f32_values(w2a)?; let raw_b=f32_values(w2b)?;
        for bi in 0..b as usize {
            for di in 0..d as usize {
                let mut want=0.0; let mut products=0.0;
                for rank in 0..16usize {
                    let product=raw_a[bi*16+rank] as f64*raw_b[rank*d as usize+di] as f64;
                    want+=product; products+=product.abs();
                }
                inner_checks &= math::within(raw_w2[bi*d as usize+di] as f64,want,math::dot_bound(16,products));
            }
        }
        for row in 0..3usize {
            for out in 0..components as usize {
                let (oi,oj)=(out/b as usize,out%b as usize);
                let mut structured=0.0; let mut structured_abs=0.0;
                let mut materialized=0.0; let mut materialized_abs=0.0;
                let mut reconstruction_pass=true;
                for ci in 0..c as usize {
                    for dj in 0..d as usize {
                        let input=inputs[row*(c*d) as usize+ci*d as usize+dj] as f64;
                        let product=input*stored_w1[oi*c as usize+ci] as f64*stored_w2[oj*d as usize+dj] as f64;
                        structured+=product; structured_abs+=product.abs();
                        let coefficient=coefficients[out*(c*d) as usize+ci*d as usize+dj] as f64;
                        let raw_product=raw_w1[oi*c as usize+ci] as f64*raw_w2[oj*d as usize+dj] as f64;
                        // Complete-delta BF16 round versus exact product of the
                        // actual rawF32 inner-product result; no factor preround.
                        let coefficient_bound=(2f64).powi(-8)*raw_product.abs()
                            +math::dot_bound(1,raw_product.abs());
                        reconstruction_pass &= math::within(coefficient,raw_product,coefficient_bound);
                        materialized+=input*coefficient;
                        materialized_abs+=(input*coefficient).abs();
                    }
                }
                let structured_bound=(math::gamma(c as usize+1)+math::gamma(d as usize+1)
                    +math::gamma(c as usize+1)*math::gamma(d as usize+1))*structured_abs
                    +(c*d) as f64*f32::MIN_POSITIVE as f64;
                let direct_bound=math::dot_bound((c*d) as usize,materialized_abs);
                let index=row*components as usize+out;
                let structured_pass=math::within(current_values[index] as f64,structured,structured_bound);
                let direct_pass=math::within(direct_values[index] as f64,materialized,direct_bound);
                checks.push(json!({"row":row,"component":out,"structured":current_values[index],
                    "structuredOracleF64":structured,"structuredBound":structured_bound,"structuredPass":structured_pass,
                    "directDelta":direct_values[index],"directOracleF64":materialized,"directBound":direct_bound,"directPass":direct_pass,
                    "rawInnerProductPass":inner_checks,"completeDeltaBF16RoundPass":reconstruction_pass}));
            }
        }
        use sha2::Digest;
        let input_sha256 = format!("{:x}", sha2::Sha256::digest(inputs.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()));
        capture.rows.insert(path.clone(),json!({"path":path,"inputDtype":format!("{:?}",x.dtype()),
            "residualDtype":format!("{:?}",current.dtype()),"jointRows":s,"rowIndices":row_indices,
            "factorShape":[a,b,c,d],"inputRowsSha256":input_sha256,"inputRowsF32":inputs,"rawW1F32":raw_w1,"rawW2F32":raw_w2,
            "rawW2AF32":raw_a,"rawW2BF32":raw_b,
            "storedW1BF16AsF32":stored_w1,"storedW2BF16AsF32":stored_w2,
            "selectedDirectDeltaBF16AsF32":coefficients,"componentChecks":checks}));
        Ok(())
    })
}

fn checked_file(path: &Path, bytes: u64, sha: &str) {
    assert_eq!(std::fs::metadata(path).unwrap().len(), bytes);
    assert_eq!(evidence::sha256_file(path), sha);
}
fn donor_and_pricing() -> (PathBuf, math::Pricing, Value) {
    let manifest_path = PathBuf::from(std::env::var("QWEN_IMAGE_2_1_DIAGNOSTIC_MANIFEST").unwrap());
    let manifest: Value = serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["purpose"], "DIAGNOSTIC_ONLY");
    assert_eq!(manifest["acceptanceEvidence"], false);
    assert_eq!(
        manifest["trainingProvenance"]["trainingCaption"],
        evidence::edit_protocol::TRAIN_EDIT_INSTRUCTION
    );
    assert_eq!(
        manifest["trainingProvenance"]["trainingSourceMain"],
        "b3ec3f0b8ee6880dc55e6433a35dc60116709752"
    );
    assert_eq!(
        manifest["trainingProvenance"]["trainingRun"],
        37214050997u64
    );
    assert_eq!(manifest["trainingProvenance"]["steps"], 120);
    assert_eq!(
        manifest["trainingProvenance"]["trainingJob"],
        111470848550u64
    );
    assert_eq!(
        manifest["trainingProvenance"]["datasetSha256"],
        "0d4927eedfcf33eb609278089c02c58f729a87f65d78170b0d7076f3a352073c"
    );
    let dir = manifest_path.parent().unwrap();
    let donor = dir.join("qwen21_diagnostic_edit_lokr.safetensors");
    checked_file(&donor, 6_759_417, DONOR_HASH);
    let receipt = dir.join("training-receipt.json");
    checked_file(&receipt, 50_361, RECEIPT_HASH);
    let receipt: Value = serde_json::from_slice(&std::fs::read(receipt).unwrap()).unwrap();
    assert_eq!(receipt["training"]["stepsRun"], 120);
    let losses = receipt["training"]["losses"].as_array().unwrap();
    assert_eq!(losses.len(), 120);
    assert!(losses.iter().all(|v| v.as_f64().unwrap().is_finite()));
    let metadata = mlx_gen::gen_core::weightsmeta::safetensors_file_metadata(&donor).unwrap();
    for (key, value) in [
        ("family", "qwen-image-2-1"),
        ("networkType", "lokr"),
        ("trainingMode", "edit"),
        ("rank", "16"),
        ("alpha", "16"),
        (
            "license",
            "Qwen Research License Agreement (research/evaluation only)",
        ),
    ] {
        assert_eq!(metadata.get(key).map(String::as_str), Some(value));
    }
    // Header-only geometry before any array load or materialization.
    let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&donor).unwrap();
    let mut factors: BTreeMap<String, BTreeMap<String, Vec<usize>>> = BTreeMap::new();
    for h in headers {
        assert_eq!(h.dtype, mlx_gen::gen_core::weightsmeta::Dtype::F32);
        let (module, suffix) = h.name.rsplit_once('.').unwrap();
        factors
            .entry(module.into())
            .or_default()
            .insert(suffix.into(), h.shape);
    }
    let mut shapes = Vec::new();
    for f in factors.values() {
        assert_eq!(f.len(), 3);
        let (w1, a, b) = (&f["lokr_w1"], &f["lokr_w2_a"], &f["lokr_w2_b"]);
        assert!(w1.len() == 2 && a.len() == 2 && b.len() == 2);
        assert_eq!(a[1], 16);
        assert_eq!(b[0], 16);
        shapes.push(((w1[0] * a[0]) as u64, (w1[1] * b[1]) as u64));
    }
    (donor, math::price(&shapes).unwrap(), manifest)
}

#[test]
#[ignore]
fn fixed_q4_numeric_diagnostic() {
    let out = evidence::out_dir().join("q4-numeric");
    std::fs::create_dir_all(&out).unwrap();
    let source = std::env::var("GITHUB_SHA").unwrap();
    assert_eq!(source.len(), 40);
    assert!(source.bytes().all(|c| c.is_ascii_hexdigit()));
    let (donor, pricing, manifest) = donor_and_pricing();
    evidence::write_json(
        &out,
        "DIAGNOSTIC_ONLY",
        &json!({"kind":"DIAGNOSTIC_ONLY","acceptanceEvidence":false,
        "source":source,"sourceBase":"83bd4c53ce9abb3f2c755415c0a92d39ed2096c7","protocolSha256":PROTOCOL_HASH,"renderCount":4,"originalDonorSha256":DONOR_HASH,
        "provenance":manifest["trainingProvenance"],"pricing":{"deltaElements":pricing.elements,"bf16DeltaBytes":pricing.bf16_delta,
        "f32WideningBytes":pricing.f32_widening,"scratchBytes":pricing.construction_scratch,
        "residentBytes":pricing.resident,"activeEnvelopeBytes":pricing.active,"physicalEnvelopeBeforeOverheadBytes":pricing.physical}}),
    );
    let replay = PathBuf::from(std::env::var("QWEN_IMAGE_2_1_Q4_REPLAY_DIR").unwrap());
    checked_file(
        &replay.join("q4_edit_base.png"),
        232_879,
        "ba8baf8aa439a0c00e952b9615669dbde38844ec92bb3a7aa48fe79d89708235",
    );
    checked_file(
        &replay.join("q4_edit_mlx_lokr.png"),
        306_523,
        "b8ffa6b98d4f2ddbf855060630c15d258544628fb2a1b5f0244ad880f4e7ee55",
    );
    let guard = evidence::Footprint::start(&out);
    guard.admit_numeric(pricing.active, math::TRANSIENT);
    let req = GenerationRequest {
        prompt: crate::q4_real_weights_support::edit_protocol::EDIT_INSTRUCTION.into(),
        conditioning: vec![mlx_gen::gen_core::Conditioning::MultiReference {
            images: vec![
                evidence::to_image(evidence::edit_source(99, 768)),
                evidence::to_image(evidence::edit_key(768)),
            ],
        }],
        ..evidence::t2i_request()
    };
    let spec = evidence::tier_spec("q4", Some(Quant::Q4));
    let expected = evidence::to_image(evidence::edit_transform(&evidence::edit_source(99, 768)));
    let mut images = Vec::new();
    let mut facts = Vec::new();
    for (name, variant, scale) in [
        ("base", Variant::Production, None),
        ("structured", Variant::Production, Some(1.0)),
        (
            "DirectTrainingDeltaRepresentation",
            Variant::DirectTrainingDeltaRepresentation,
            Some(1.0),
        ),
        (
            "direct_zero",
            Variant::DirectTrainingDeltaRepresentation,
            Some(0.0),
        ),
    ] {
        let _scope = VariantScope::enter(variant);
        let load = match scale {
            None => spec.clone(),
            Some(s) => spec.clone().with_adapters(vec![evidence::adapter(
                &donor,
                s,
                mlx_gen::runtime::AdapterKind::Lokr,
            )]),
        };
        let (image, mut fact) = evidence::render(name, &load, &req, &guard, &out);
        fact["diagnosticVariant"] = json!(format!("{variant:?}"));
        fact["reservationSurchargeBytes"] = json!(resident_surcharge());
        fact["diagnosticPredictedOverlayBytes"] =
            json!(fact["predictedOverlayBytes"].as_u64().unwrap() + resident_surcharge());
        fact["pipelineComputeDtype"] = json!("Float32");
        fact["pngSha256"] = json!(evidence::sha256_file(&out.join(format!("{name}.png"))));
        fact["expectedImageError"] = json!(evidence::mean_abs_diff(&image, &expected));
        images.push(image);
        facts.push(fact);
    }
    let read_png = |file: &str| image::open(replay.join(file)).unwrap().to_rgb8().into_raw();
    let replay_base = images[0].pixels == read_png("q4_edit_base.png");
    let replay_structured = images[1].pixels == read_png("q4_edit_mlx_lokr.png");
    let zero = images[0].pixels == images[3].pixels;
    let mut outcomes = Vec::new();
    for i in [1, 2] {
        let gain = evidence::mean_abs_diff(&images[0], &expected)
            - evidence::mean_abs_diff(&images[i], &expected);
        let movement = evidence::mean_abs_diff(&images[0], &images[i]);
        outcomes.push(
            json!({"variant":facts[i]["label"],"gain":gain,"movement":movement,
            "learnedGainPass":gain>=1.0,"movementPass":movement>=2.0}),
        );
    }
    let mut failures = Vec::new();
    let base_peak = facts[0]["mlxActivePeakBytes"].as_u64().unwrap();
    for fact in &mut facts {
        let overlay_pass = fact["mlxActivePeakBytes"].as_u64().unwrap()
            <= base_peak + fact["diagnosticPredictedOverlayBytes"].as_u64().unwrap() + (1 << 30);
        fact["diagnosticOverlayPass"] = json!(overlay_pass);
        if !overlay_pass {
            failures.push(format!("{}: overlay prediction", fact["label"]));
        }
    }
    for fact in &facts {
        if fact["image"]["pixelStd"].as_f64().unwrap() <= 8.0
            || fact["image"]["staticRowFraction"].as_f64().unwrap() > 0.25
        {
            failures.push(format!("{}: image sanity", fact["label"]));
        }
        if fact["mlxActivePeakBytes"].as_u64().unwrap() > pricing.active {
            failures.push(format!("{}: conservative active envelope", fact["label"]));
        }
    }
    for outcome in &outcomes {
        if outcome["learnedGainPass"] != true || outcome["movementPass"] != true {
            failures.push(format!(
                "{}: learned direction/movement floors",
                outcome["variant"]
            ));
        }
    }
    // Persist all four render outcomes before replay/oracle assertions. Negative
    // learned direction is diagnostic data, never acceptance or a waived floor.
    evidence::write_json(
        &out,
        "render-results",
        &json!({"kind":"DIAGNOSTIC_ONLY","acceptanceEvidence":false,
        "source":source,"renderCount":4,"baseReplayIdentical":replay_base,"structuredReplayIdentical":replay_structured,
        "zeroIdentical":zero,"renders":facts,"outcomes":outcomes,"failures":failures,
        "gainFloor":1.0,"movementFloor":2.0,"imageStdFloor":8.0,"staticRowMaximum":0.25}),
    );
    assert!(
        facts
            .iter()
            .all(|f| f["image"]["pixelStd"].as_f64().unwrap() > 8.0
                && f["image"]["staticRowFraction"].as_f64().unwrap() <= 0.25
                && f["mlxActivePeakBytes"].as_u64().unwrap() <= pricing.active
                && f["diagnosticOverlayPass"] == true),
        "sanity/memory failure retained; arithmetic interpretation refused"
    );
    assert!(
        replay_base && replay_structured && zero,
        "replay/zero failure; all four captures retained"
    );
    let (params, _) = read_params(&donor).unwrap();
    let _numeric_scope = VariantScope::enter(Variant::Production);
    CAPTURE.with(|c| {
        *c.borrow_mut() = Some(Capture {
            params,
            rows: BTreeMap::new(),
        })
    });
    guard.begin();
    let capture_result = crate::model::diagnostic_first_forward(
        &spec.clone().with_adapters(vec![evidence::adapter(
            &donor,
            1.0,
            mlx_gen::runtime::AdapterKind::Lokr,
        )]),
        &req,
    );
    let captures = CAPTURE.with(|c| c.borrow_mut().take().unwrap().rows);
    let (numeric_active, numeric_physical) = guard.end();
    evidence::write_json(
        &out,
        "numeric-first-forward",
        &json!({"kind":"DIAGNOSTIC_ONLY","acceptanceEvidence":false,
        "scope":"separate production firststep forward, no render instrumentation","source":source,
        "forwardResult":capture_result.as_ref().map(|_|"ok").map_err(ToString::to_string),
        "mlxActivePeakBytes":numeric_active,"physFootprintMaxBytes":numeric_physical,"captures":captures}),
    );
    capture_result.unwrap();
    assert_eq!(captures.len(), 3);
    assert!(
        captures.values().all(
            |v| v["componentChecks"]
                .as_array()
                .unwrap()
                .iter()
                .all(|c| c["structuredPass"] == true
                    && c["directPass"] == true
                    && c["rawInnerProductPass"] == true
                    && c["completeDeltaBF16RoundPass"] == true)
        ),
        "componentwise oracle failure retained"
    );
}
