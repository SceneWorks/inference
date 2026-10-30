//! Test-only bidirectional adapter exchange. The exporting test executes real native CPU
//! autograd and serializer code; an output hook is never itself a claim that export ran.
use crate::transformer::{Geometry, Transformer};
use gen_core_moss_image::moss_image::*;
use mlx_gen::gen_core::GenerationRequest;
use mlx_gen::{CancelFlag, NetworkType};
use mlx_rs::{Array, Dtype};
use serde_json::Value;
use std::path::Path;

fn read(array: &Array) -> Vec<f32> {
    crate::pipeline::logical_f32_readback(array).unwrap()
}
/// Captured source-closure provenance is retained, not compared to the current tree as a gate.
fn fixture_lineage(path: &Path) -> Value {
    use sha2::{Digest, Sha256};
    let sources: &[(&str, &[u8])] = &[
        ("Cargo.lock", include_bytes!("../../../../../Cargo.lock")),
        (
            "moss-contract",
            include_bytes!("../../../../contracts/gen-core-moss-image/src/moss_image_training.rs"),
        ),
        (
            "author-fixture",
            include_bytes!(
                "../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_author/fixture.json"
            ),
        ),
        ("transformer", include_bytes!("transformer.rs")),
        ("adapters", include_bytes!("adapters.rs")),
        ("training", include_bytes!("training.rs")),
        ("sampling", include_bytes!("pipeline.rs")),
        (
            "shared-adapter-writer",
            include_bytes!("../../src/train/lora.rs"),
        ),
        (
            "fixture-producer",
            include_bytes!("portable_exchange_tests.rs"),
        ),
    ];
    let mut tree = Sha256::new();
    for (name, source) in sources {
        tree.update((name.len() as u64).to_le_bytes());
        tree.update(name.as_bytes());
        tree.update((source.len() as u64).to_le_bytes());
        tree.update(source);
    }
    let bytes = std::fs::read(path).unwrap();
    let header = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let raw = bytes.get(8usize.checked_add(header).unwrap()..).unwrap();
    serde_json::json!({"producer_tree_sha256":format!("{:x}",tree.finalize()),"adapter_sha256":format!("{:x}",Sha256::digest(&bytes)),"raw_factor_payload_sha256":format!("{:x}",Sha256::digest(raw))})
}
fn verify_fixture_lineage(path: &Path, record: &Value) {
    let captured = &record["lineage"];
    let tree = captured["producer_tree_sha256"]
        .as_str()
        .expect("producer source-closure hash is required");
    assert!(tree.len() == 64 && tree.bytes().all(|b| b.is_ascii_hexdigit()));
    let actual = fixture_lineage(path);
    for key in ["adapter_sha256", "raw_factor_payload_sha256"] {
        assert_eq!(
            captured[key], actual[key],
            "fixture producer/raw-factor lineage mismatch: {key}"
        );
    }
}
fn json(array: &Array) -> Value {
    serde_json::json!({"shape":array.shape(),"values":read(array)})
}
fn from_json(value: &Value) -> Array {
    let shape: Vec<i32> = serde_json::from_value(value["shape"].clone()).unwrap();
    let values: Vec<f32> = serde_json::from_value(value["values"].clone()).unwrap();
    Array::from_slice(&values, &shape)
}
fn sampled(model: &Transformer, variant: MossImageVariant, x: &Array, context: &Array) -> Array {
    let request = GenerationRequest {
        width: x.shape()[3] as u32 * 16,
        height: x.shape()[2] as u32 * 16,
        steps: Some(3),
        guidance: Some(1.),
        ..Default::default()
    };
    let resolved = MossImageResolved::resolve(variant, &request).unwrap();
    crate::pipeline::Sampling {
        transformer: model,
        resolved: &resolved,
        positive: std::slice::from_ref(context),
        negative: &[],
        dataspace: None,
    }
    .run(x, &CancelFlag::new(), &mut |_| {})
    .unwrap()
}
/// Called only after the explicitly CPU-bound functional test has updated factors and verified
/// its saved adapter through the production loader. Preserve the serializer's actual bytes.
pub(crate) fn record_export(
    variant: MossImageVariant,
    network: NetworkType,
    path: &Path,
    model: &Transformer,
    x: &Array,
    context: &Array,
) {
    let Some(output) = std::env::var_os("MOSS_PORTABLE_EXCHANGE_OUTPUT") else {
        return;
    };
    let output = std::path::PathBuf::from(output);
    std::fs::create_dir_all(&output).unwrap();
    let method = if network == NetworkType::Lora {
        "lora"
    } else {
        "lokr"
    };
    let stem = format!("mlx-{}-{method}", variant.provider_id());
    let velocity = model
        .forward(
            x,
            &[370.],
            std::slice::from_ref(context),
            &[],
            &CancelFlag::new(),
        )
        .unwrap();
    let generated = sampled(model, variant, x, context);
    let record = serde_json::json!({"schema":1,"producer":"mlx","variant":variant.provider_id(),"network_type":method,"base_storage":"f32","time":370.,"sample_steps":3,"input":json(x),"context":json(context),"velocity":json(&velocity),"sampled":json(&generated),"adapter":format!("{stem}.safetensors"),"lineage":fixture_lineage(path)});
    std::fs::copy(path, output.join(format!("{stem}.safetensors"))).unwrap();
    std::fs::write(
        output.join(format!("{stem}.json")),
        serde_json::to_vec_pretty(&record).unwrap(),
    )
    .unwrap();
}
fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_author/fixture.json"
    ))
    .unwrap()
}
fn base(f: &Value, source_dtype: Dtype) -> (mlx_gen::weights::Weights, Geometry) {
    let c = &f["config"];
    let number = |key: &str| c[key].as_i64().unwrap() as i32;
    let g = Geometry {
        dim: number("dim"),
        ffn: number("ffn_dim"),
        heads: number("num_heads"),
        layers: c["num_layers"].as_u64().unwrap() as usize,
        text: number("text_dim"),
        freq: number("freq_dim"),
        channels: number("latent_channels"),
    };
    // The Candle producer deliberately starts with BF16 source values, then requests F32 compute.
    let map = f["weights"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, value)| {
            (
                key.clone(),
                from_json(value)
                    .as_dtype(source_dtype)
                    .unwrap()
                    .as_dtype(Dtype::Float32)
                    .unwrap(),
            )
        })
        .collect();
    (mlx_gen::weights::Weights::from_map(map), g)
}
fn compare(actual: &Array, expected: &Value, tolerance: f32) {
    let expected_shape: Vec<i32> = serde_json::from_value(expected["shape"].clone()).unwrap();
    let expected: Vec<f32> = serde_json::from_value(expected["values"].clone()).unwrap();
    assert_eq!(actual.shape(), expected_shape);
    let actual = read(actual);
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().chain(&expected).all(|v| v.is_finite()));
    let error = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max);
    assert!(
        error <= tolerance,
        "portable Candle→MLX max error {error} > {tolerance}"
    );
}
#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn cpu_train_export_and_import_portable_adapters() {
    // This dedicated functional seam observes no allocator, memory-limit, peak, footprint or
    // process-global resource counters. The older S1 checkpoint-counter test is not selected.
    let _device = crate::cpu_test_device();
    assert!(matches!(
        mlx_rs::Device::try_default().unwrap().get_type().unwrap(),
        mlx_rs::DeviceType::Cpu
    ));
    let cpu_stream = mlx_rs::Stream::cpu();
    let cpu_stream_index = cpu_stream.get_index().unwrap();
    mlx_rs::with_new_default_stream(cpu_stream, || {
        assert!(matches!(
            mlx_rs::Device::try_default().unwrap().get_type().unwrap(),
            mlx_rs::DeviceType::Cpu
        ));
        assert_eq!(
            mlx_rs::Stream::task_local_or_default().get_index().unwrap(),
            cpu_stream_index
        );
        use gen_core_moss_image::moss_image_training as contract;
        use mlx_gen::train::lora::{build_lokr_targets, build_lora_targets, TrainAdapter};
        let f = fixture();
        let x = mlx_gen::attention::slice_axis(&from_json(&f["inputs"]["hidden_states"]), 0, 0, 1)
            .unwrap();
        let all_context = from_json(&f["inputs"]["encoder_hidden_states"]);
        let context = mlx_gen::attention::slice_axis(&all_context, 0, 0, 1).unwrap();
        let sequence = context.shape()[1] as usize;
        let valid: Vec<i32> = f["inputs"]["encoder_attention_mask"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .take(sequence)
            .enumerate()
            .filter_map(|(i, v)| (v.as_f64().unwrap() != 0.).then_some(i as i32))
            .collect();
        let context = context
            .take_axis(Array::from_slice(&valid, &[valid.len() as i32]), 1)
            .unwrap();
        let noise = mlx_rs::ops::add(&x, Array::from_f32(0.37)).unwrap();
        for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
            for network in [NetworkType::Lora, NetworkType::Lokr] {
                let (weights, g) = base(&f, Dtype::Float32);
                let frozen: std::collections::BTreeMap<_, _> = weights
                    .keys()
                    .map(|key| (key.to_owned(), read(weights.require(key).unwrap())))
                    .collect();
                let mut model = Transformer::load(&weights, g.clone()).unwrap();
                let baseline = model
                    .forward(
                        &x,
                        &[370.],
                        std::slice::from_ref(&context),
                        &[],
                        &CancelFlag::new(),
                    )
                    .unwrap();
                let paths = contract::target_paths(
                    g.layers,
                    &["q".into(), "ffn_up_gate".into(), "ffn_down".into()],
                )
                .unwrap();
                let (adapter, mut params) = match network {
                    NetworkType::Lora => {
                        let (targets, params) =
                            build_lora_targets(&mut model, &paths, 3, 17).unwrap();
                        (TrainAdapter::Lora { targets }, params)
                    }
                    NetworkType::Lokr => {
                        let (targets, params) =
                            build_lokr_targets(&mut model, &paths, 3, 2, 17).unwrap();
                        (TrainAdapter::Lokr { targets }, params)
                    }
                };
                let before: std::collections::BTreeMap<_, _> = params
                    .iter()
                    .map(|(key, value)| (key.to_string(), read(value)))
                    .collect();
                let cfg = mlx_gen::TrainingConfig {
                    rank: 3,
                    alpha: 0.7,
                    network_type: network,
                    decompose_factor: 2,
                    seed: 17,
                    train_dtype: "f32".into(),
                    gradient_checkpointing: true,
                    ..Default::default()
                };
                let mut optimizer =
                    mlx_gen::TrainOptimizer::from_config("adamw", 0.01, 0.).unwrap();
                for update in 0..3 {
                    let (loss, grads) = crate::training::loss_grads(
                        &mut model,
                        &params,
                        &adapter,
                        &cfg,
                        &x,
                        &context,
                        &noise,
                        0.37,
                        None,
                        &CancelFlag::new(),
                    )
                    .unwrap();
                    assert!(loss.is_finite());
                    assert!(grads.values().flat_map(read).all(|v| v.is_finite()));
                    mlx_gen::train::lora::adapter_optimizer_update(
                        &mut optimizer,
                        &mut params,
                        &grads,
                        &cfg,
                        update,
                        17,
                    )
                    .unwrap();
                }
                assert!(
                    params
                        .iter()
                        .any(|(key, value)| read(value) != before[key.as_ref()]),
                    "native CPU autograd must actually update raw adapter factors"
                );
                for key in weights.keys() {
                    assert_eq!(
                        frozen[key],
                        read(weights.require(key).unwrap()),
                        "frozen backbone changed"
                    );
                }
                let temp = tempfile::tempdir().unwrap();
                let path = temp.path().join("portable.safetensors");
                let meta = contract::metadata(variant, contract::SAMPLER);
                let extra: Vec<_> = meta.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
                crate::adapters::save_validated(&mut model, &adapter, &params, &cfg, &extra, &path)
                    .unwrap();
                let mut reloaded = Transformer::load(&weights, g).unwrap();
                let kind = if network == NetworkType::Lora {
                    mlx_gen::runtime::AdapterKind::Lora
                } else {
                    mlx_gen::runtime::AdapterKind::Lokr
                };
                let spec = mlx_gen::runtime::AdapterSpec::new(path.clone(), 1., kind);
                let reports = crate::adapters::apply(&mut reloaded, variant, &[spec]).unwrap();
                assert!(reports[0].applied > 0 && reports[0].skipped.is_empty());
                let actual = reloaded
                    .forward(
                        &x,
                        &[370.],
                        std::slice::from_ref(&context),
                        &[],
                        &CancelFlag::new(),
                    )
                    .unwrap();
                assert!(
                    read(&actual)
                        .iter()
                        .zip(read(&baseline))
                        .any(|(a, b)| (a - b).abs() > 1e-7),
                    "trained adapter must change generation"
                );
                record_export(variant, network, &path, &reloaded, &x, &context);
            }
        }
        if let Some(path) = std::env::var_os("MOSS_PORTABLE_EXCHANGE_INPUT") {
            import_candle_exchange(Path::new(&path));
        }
    });
}
#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn committed_candle_cpu_adapters_generate_portably() {
    let _device = crate::cpu_test_device();
    assert!(matches!(
        mlx_rs::Device::try_default().unwrap().get_type().unwrap(),
        mlx_rs::DeviceType::Cpu
    ));
    let stream = mlx_rs::Stream::cpu();
    let expected = stream.get_index().unwrap();
    mlx_rs::with_new_default_stream(stream, || {
        assert_eq!(
            mlx_rs::Stream::task_local_or_default().get_index().unwrap(),
            expected
        );
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../contracts/gen-core/tests/fixtures/moss_image/portable_adapters");
        import_candle_exchange(&root);
    });
}
pub(crate) fn import_candle_exchange(root: &Path) {
    let _device = crate::cpu_test_device(); // Establish CPU before the first Array, including weights.
    let f = fixture();
    for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
        for method in ["lora", "lokr"] {
            let stem = format!("candle-{}-{method}", variant.provider_id());
            let record: Value = serde_json::from_slice(
                &std::fs::read(root.join(format!("{stem}.json")))
                    .expect("genuine Candle CPU fixture is required"),
            )
            .unwrap();
            assert_eq!(record["schema"], 1);
            assert_eq!(record["producer"], "candle");
            assert_eq!(record["base_storage"], "bf16");
            assert_eq!(record["variant"], variant.provider_id());
            assert_eq!(record["network_type"], method);
            let adapter_path = root.join(format!("{stem}.safetensors"));
            verify_fixture_lineage(&adapter_path, &record);
            let (weights, g) = base(&f, Dtype::Bfloat16);
            let mut model = Transformer::load(&weights, g.clone()).unwrap();
            let x = from_json(&record["input"]);
            let context = from_json(&record["context"]);
            let time = record["time"].as_f64().unwrap() as f32;
            let kind = if method == "lora" {
                mlx_gen::runtime::AdapterKind::Lora
            } else {
                mlx_gen::runtime::AdapterKind::Lokr
            };
            let spec = mlx_gen::runtime::AdapterSpec::new(adapter_path, 1., kind);
            let reports =
                crate::adapters::apply(&mut model, variant, std::slice::from_ref(&spec)).unwrap();
            assert!(reports[0].applied > 0 && reports[0].skipped.is_empty());
            let actual = model
                .forward(
                    &x,
                    &[time],
                    std::slice::from_ref(&context),
                    &[],
                    &CancelFlag::new(),
                )
                .unwrap();
            compare(&actual, &record["velocity"], 5e-5);
            compare(
                &sampled(&model, variant, &x, &context),
                &record["sampled"],
                0.0002,
            );
            let bare = Transformer::load(&weights, g).unwrap();
            let baseline = bare
                .forward(
                    &x,
                    &[time],
                    std::slice::from_ref(&context),
                    &[],
                    &CancelFlag::new(),
                )
                .unwrap();
            assert!(
                read(&actual)
                    .iter()
                    .zip(read(&baseline))
                    .any(|(a, b)| (a - b).abs() > 1e-7),
                "Candle-exported adapter must affect MLX generation"
            );
        }
    }
}

#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn strict_application_rejects_unscaled_intermediate_overflow_before_success() {
    let _device = crate::cpu_test_device();
    mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || {
        let temp = tempfile::tempdir().unwrap();
        let f = fixture();
        let (weights, geometry) = base(&f, Dtype::Float32);
        for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
            for case in [
                "lora",
                "lora-scaled",
                "lokr",
                "lokr-reversed",
                "lokr-decomposed",
            ] {
                let lokr = case.starts_with("lokr");
                let scaled = case == "lora-scaled";
                let alpha = if scaled { 32f32 } else { 0.25 };
                let mut metadata = gen_core_moss_image::moss_image_training::metadata(
                    variant,
                    gen_core_moss_image::moss_image_training::SAMPLER,
                );
                metadata.extend([
                    (
                        "networkType".into(),
                        if lokr { "lokr" } else { "lora" }.into(),
                    ),
                    ("rank".into(), "1".into()),
                    ("alpha".into(), alpha.to_string()),
                    ("decomposeFactor".into(), "-1".into()),
                ]);
                let sparse = |shape: [i32; 2]| {
                    let mut v = vec![0f32; (shape[0] * shape[1]) as usize];
                    v[0] = if scaled { 1e18 } else { 2e19 };
                    if scaled && shape[0] == 1 {
                        v.fill(1e18);
                    }
                    Array::from_slice(&v, &shape)
                };
                let factors = if case == "lokr-reversed" {
                    vec![
                        ("layers.0.proj.q.lokr_w1", sparse([1, 4])),
                        ("layers.0.proj.q.lokr_w2", sparse([16, 4])),
                    ]
                } else if case == "lokr-decomposed" {
                    vec![
                        (
                            "layers.0.proj.q.lokr_w1_a",
                            Array::from_slice(&[0f32; 4], &[4, 1]),
                        ),
                        (
                            "layers.0.proj.q.lokr_w1_b",
                            Array::from_slice(&[f32::MAX, f32::MAX, 0., 0.], &[1, 4]),
                        ),
                        (
                            "layers.0.proj.q.lokr_w2",
                            Array::from_slice(
                                &[
                                    1f32, 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
                                    1.,
                                ],
                                &[4, 4],
                            ),
                        ),
                    ]
                } else if lokr {
                    vec![
                        ("layers.0.proj.q.lokr_w1", sparse([4, 4])),
                        ("layers.0.proj.q.lokr_w2", sparse([4, 4])),
                    ]
                } else {
                    vec![
                        ("layers.0.proj.q.lora_A.weight", sparse([1, 16])),
                        ("layers.0.proj.q.lora_B.weight", sparse([16, 1])),
                        ("layers.0.proj.q.alpha", Array::from_slice(&[alpha], &[1])),
                    ]
                };
                let raw = if scaled {
                    f64::from(1e18f32).powi(2) * 16.
                } else {
                    f64::from(2e19f32).powi(2)
                };
                assert!(if scaled {
                    raw < f64::from(f32::MAX) && raw * f64::from(alpha) > f64::from(f32::MAX)
                } else {
                    raw > f64::from(f32::MAX) && raw * f64::from(alpha) < f64::from(f32::MAX)
                });
                let path = temp.path().join("overflow.safetensors");
                let metadata: std::collections::HashMap<_, _> = metadata.into_iter().collect();
                Array::save_safetensors(
                    factors.iter().map(|(k, v)| (*k, v)),
                    Some(&metadata),
                    &path,
                )
                .unwrap();
                let kind = if lokr {
                    mlx_gen::runtime::AdapterKind::Lokr
                } else {
                    mlx_gen::runtime::AdapterKind::Lora
                };
                let mut host = Transformer::load(&weights, geometry.clone()).unwrap();
                let spec = mlx_gen::runtime::AdapterSpec::new(path.clone(), 1., kind);
                let result = crate::adapters::apply(&mut host, variant, &[spec]);
                assert!(matches!(result, Err(mlx_gen::Error::Unsupported(ref message))
                    if message.contains("contraction") && message.contains("reduce factor magnitude")),
                    "unsafe raw contraction must be actionably refused before application success ({variant:?}/{case}): {result:?}");
                if !lokr {
                    let installed = mlx_gen::adapters::Adapter::Lora {
                        a: factors[0].1.t(),
                        b: factors[1].1.t(),
                        scale: alpha,
                    };
                    let validation = crate::adapters::validate_lora(&installed);
                    assert!(
                        matches!(validation, Err(mlx_gen::Error::Unsupported(ref message))
                        if message.contains("contraction")),
                        "producer must reject unsafe unfused training factors too"
                    );
                    let mut producer = Transformer::load(&weights, geometry.clone()).unwrap();
                    let paths = vec!["layers.0.proj.q".to_owned()];
                    let (targets, mut params) =
                        mlx_gen::train::lora::build_lora_targets(&mut producer, &paths, 1, 17)
                            .unwrap();
                    params.insert(
                        format!("{}.lora_a", targets[0].path).into(),
                        factors[0].1.clone(),
                    );
                    params.insert(
                        format!("{}.lora_b", targets[0].path).into(),
                        factors[1].1.clone(),
                    );
                    let adapter = mlx_gen::train::lora::TrainAdapter::Lora { targets };
                    let cfg = mlx_gen::TrainingConfig {
                        rank: 1,
                        alpha,
                        train_dtype: "f32".into(),
                        ..Default::default()
                    };
                    let output = temp.path().join("must-not-export.safetensors");
                    let export = crate::adapters::save_validated(
                        &mut producer,
                        &adapter,
                        &params,
                        &cfg,
                        &[],
                        &output,
                    );
                    assert!(matches!(export, Err(mlx_gen::Error::Unsupported(ref message))
                        if message.contains("contraction")) && !output.exists(),
                        "actual producer export must reject unsafe RAW factors before alpha folding or writing");
                }
                let mut zero_host = Transformer::load(&weights, geometry.clone()).unwrap();
                let zero = mlx_gen::runtime::AdapterSpec::new(path.clone(), 0., kind);
                assert!(crate::adapters::apply(&mut zero_host, variant, &[zero]).is_ok(),
                    "finite disabled factors must not execute or be rejected for a skipped contraction");
            }
        }
    });
}

/// External maltrainer/ComfyUI MOSS LoRAs load AS-IS (2026-10-29 product decision): the
/// `diffusion_model.` key prefix is stripped, BF16 factors upcast losslessly to the FP32 residual
/// stack, per-target alpha scalars take precedence over the kohya `ss_network_alpha` fallback,
/// per-target rank may vary (SVD-resized exports), and both Base/SFT checkpoints accept any MOSS
/// adapter regardless of its trained parent. The parity arm proves strip+upcast is semantically
/// the identity: the same factors through the external and native formats generate identically.
#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn external_prefixed_bf16_lora_loads_as_is_on_both_checkpoints() {
    let _device = crate::cpu_test_device();
    mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || {
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};
        let temp = tempfile::tempdir().unwrap();
        let f = fixture();
        let (weights, g) = base(&f, Dtype::Float32);
        let x = mlx_gen::attention::slice_axis(&from_json(&f["inputs"]["hidden_states"]), 0, 0, 1)
            .unwrap();
        let context = mlx_gen::attention::slice_axis(
            &from_json(&f["inputs"]["encoder_hidden_states"]),
            0,
            0,
            1,
        )
        .unwrap();
        let forward = |model: &Transformer| {
            read(
                &model
                    .forward(
                        &x,
                        &[370.],
                        std::slice::from_ref(&context),
                        &[],
                        &CancelFlag::new(),
                    )
                    .unwrap(),
            )
        };
        // Power-of-two factor values so the BF16 round-trip is exact and parity can be bitwise.
        let full = |v: f32, shape: [i32; 2]| {
            Array::from_slice(&vec![v; (shape[0] * shape[1]) as usize], &shape)
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        };
        let q_only: Vec<(&str, Array)> = vec![
            (
                "diffusion_model.layers.0.proj.q.lora_A.weight",
                full(0.125, [2, 16]),
            ),
            (
                "diffusion_model.layers.0.proj.q.lora_B.weight",
                full(0.25, [16, 2]),
            ),
            ("diffusion_model.layers.0.proj.q.alpha", Array::from_f32(4.)),
        ];
        // Second target with a DIFFERENT rank and no alpha scalar: covers SVD-resized per-target
        // rank variation and the kohya ss_network_alpha fallback in one file.
        let mut external: Vec<(&str, Array)> = q_only.clone();
        external.extend([
            (
                "diffusion_model.layers.1.proj.ffn_up_gate.lora_A.weight",
                full(0.0625, [3, 16]),
            ),
            (
                "diffusion_model.layers.1.proj.ffn_up_gate.lora_B.weight",
                full(0.5, [48, 3]),
            ),
        ]);
        let kohya: std::collections::HashMap<String, String> = [
            ("ss_base_model_version", "moss_image_base"),
            ("ss_network_module", "maltrainer.models.moss_image.lora"),
            ("ss_network_dim", "8"),
            ("ss_network_alpha", "9"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
        let save = |name: &str,
                    tensors: &[(&str, Array)],
                    meta: &std::collections::HashMap<String, String>| {
            let path = temp.path().join(name);
            Array::save_safetensors(tensors.iter().map(|(k, v)| (*k, v)), Some(meta), &path)
                .unwrap();
            path
        };
        let external_path = save("external.safetensors", &external, &kohya);
        let baseline = forward(&Transformer::load(&weights, g.clone()).unwrap());
        for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
            let mut host = Transformer::load(&weights, g.clone()).unwrap();
            let spec = AdapterSpec::new(external_path.clone(), 1., AdapterKind::Lora);
            let reports = crate::adapters::apply(&mut host, variant, &[spec]).unwrap();
            assert_eq!(reports[0].applied, 2);
            assert!(reports[0].skipped.is_empty());
            assert!(
                forward(&host)
                    .iter()
                    .zip(&baseline)
                    .any(|(a, b)| (a - b).abs() > 1e-7),
                "external adapter must change generation on {variant:?}"
            );
        }
        // Parity: the stripped/upcast external q target equals the native-format q target with
        // the same values — and the native file is Base-stamped while applied to an SFT host,
        // which also pins the relaxed cross-checkpoint acceptance for native adapters.
        let mut native_meta: std::collections::HashMap<String, String> =
            gen_core_moss_image::moss_image_training::metadata(
                MossImageVariant::Base,
                gen_core_moss_image::moss_image_training::SAMPLER,
            )
            .into_iter()
            .collect();
        native_meta.extend([
            ("networkType".to_owned(), "lora".to_owned()),
            ("rank".to_owned(), "2".to_owned()),
            ("alpha".to_owned(), "4".to_owned()),
        ]);
        let native_q: Vec<(&str, Array)> = q_only
            .iter()
            .map(|(k, v)| {
                (
                    k.strip_prefix("diffusion_model.").unwrap(),
                    v.as_dtype(Dtype::Float32).unwrap(),
                )
            })
            .collect();
        let native_path = save("native.safetensors", &native_q, &native_meta);
        let external_q_path = save("external-q.safetensors", &q_only, &kohya);
        let outputs: Vec<Vec<f32>> = [
            (external_q_path, MossImageVariant::Base),
            (native_path, MossImageVariant::Sft),
        ]
        .into_iter()
        .map(|(path, variant)| {
            let mut host = Transformer::load(&weights, g.clone()).unwrap();
            let spec = AdapterSpec::new(path, 1., AdapterKind::Lora);
            crate::adapters::apply(&mut host, variant, &[spec]).unwrap();
            forward(&host)
        })
        .collect();
        assert_eq!(
            outputs[0], outputs[1],
            "stripped/upcast external factors must generate identically to the native format"
        );
        // Fallback-alpha equivalence: ss_network_alpha(9) with no scalar == explicit alpha(9).
        let mut ffn_explicit: Vec<(&str, Array)> = external.clone();
        ffn_explicit.push((
            "diffusion_model.layers.1.proj.ffn_up_gate.alpha",
            Array::from_f32(9.),
        ));
        let explicit_path = save("explicit-alpha.safetensors", &ffn_explicit, &kohya);
        let outputs: Vec<Vec<f32>> = [&external_path, &explicit_path]
            .into_iter()
            .map(|path| {
                let mut host = Transformer::load(&weights, g.clone()).unwrap();
                let spec = AdapterSpec::new(path.clone(), 1., AdapterKind::Lora);
                crate::adapters::apply(&mut host, MossImageVariant::Base, &[spec]).unwrap();
                forward(&host)
            })
            .collect();
        assert_eq!(
            outputs[0], outputs[1],
            "ss_network_alpha fallback must scale exactly like a per-target alpha scalar"
        );
        // Refusals stay actionable: external LoKr, missing alpha provenance, unknown targets,
        // and a native-stamped file with non-FP32 tensors.
        let refusal = |path: std::path::PathBuf, kind: AdapterKind, needle: &str| {
            let mut host = Transformer::load(&weights, g.clone()).unwrap();
            let result = crate::adapters::apply(
                &mut host,
                MossImageVariant::Base,
                &[AdapterSpec::new(path, 1., kind)],
            );
            assert!(
                matches!(result, Err(mlx_gen::Error::Unsupported(ref m)) if m.contains(needle)),
                "expected refusal containing {needle:?}: {result:?}"
            );
            assert_eq!(forward(&host), baseline, "refusal must install nothing");
        };
        refusal(external_path.clone(), AdapterKind::Lokr, "PEFT LoRA only");
        let mut no_alpha_meta = kohya.clone();
        no_alpha_meta.remove("ss_network_alpha");
        let no_alpha: Vec<(&str, Array)> = external
            .iter()
            .filter(|(k, _)| !k.ends_with(".alpha"))
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        refusal(
            save("no-alpha.safetensors", &no_alpha, &no_alpha_meta),
            AdapterKind::Lora,
            "no alpha",
        );
        let stray: Vec<(&str, Array)> = vec![
            (
                "diffusion_model.text_encoder.q.lora_A.weight",
                full(0.125, [2, 16]),
            ),
            (
                "diffusion_model.text_encoder.q.lora_B.weight",
                full(0.25, [16, 2]),
            ),
        ];
        refusal(
            save("stray.safetensors", &stray, &kohya),
            AdapterKind::Lora,
            "not a transformer projection",
        );
        // Prefixed and unprefixed spellings of ONE factor collapse to the same logical key
        // after normalization; silently shadowing one would apply half the supplied tensors.
        let collision: Vec<(&str, Array)> = vec![
            ("layers.0.proj.q.lora_A.weight", full(0.125, [2, 16])),
            (
                "diffusion_model.layers.0.proj.q.lora_A.weight",
                full(0.125, [2, 16]),
            ),
            (
                "diffusion_model.layers.0.proj.q.lora_B.weight",
                full(0.25, [16, 2]),
            ),
        ];
        refusal(
            save("collision.safetensors", &collision, &kohya),
            AdapterKind::Lora,
            "name the same factor",
        );
        let native_bf16: Vec<(&str, Array)> = native_q
            .iter()
            .map(|(k, v)| (*k, v.as_dtype(Dtype::Bfloat16).unwrap()))
            .collect();
        refusal(
            save("native-bf16.safetensors", &native_bf16, &native_meta),
            AdapterKind::Lora,
            "must be FP32",
        );
    });
}

#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn periodic_and_final_exports_preserve_request_trigger_words() {
    let _device = crate::cpu_test_device();
    mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || {
        use mlx_gen::train::lora::{build_lokr_targets, build_lora_targets, TrainAdapter};
        let temp = tempfile::tempdir().unwrap();
        let f = fixture();
        let (weights, geometry) = base(&f, Dtype::Float32);
        for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
            for network in [NetworkType::Lora, NetworkType::Lokr] {
                let mut model = Transformer::load(&weights, geometry.clone()).unwrap();
                let paths = vec!["layers.0.proj.q".to_owned()];
                let (adapter, params) = match network {
                    NetworkType::Lora => {
                        let (targets, params) =
                            build_lora_targets(&mut model, &paths, 3, 17).unwrap();
                        (TrainAdapter::Lora { targets }, params)
                    }
                    NetworkType::Lokr => {
                        let (targets, params) =
                            build_lokr_targets(&mut model, &paths, 3, 2, 17).unwrap();
                        (TrainAdapter::Lokr { targets }, params)
                    }
                };
                let mut req = mlx_gen::TrainingRequest {
                    items: vec![],
                    output_dir: temp.path().into(),
                    file_name: "final.safetensors".into(),
                    trigger_words: vec!["ohwx".into(), "quoted\"word".into()],
                    cancel: CancelFlag::new(),
                    config: mlx_gen::TrainingConfig {
                        rank: 3,
                        alpha: 0.7,
                        decompose_factor: 2,
                        network_type: network,
                        trigger_word: Some("caption_only_not_export_words".into()),
                        train_dtype: "f32".into(),
                        ..Default::default()
                    },
                };
                for words in [req.trigger_words.clone(), vec![]] {
                    req.trigger_words = words.clone();
                    for file in [
                        mlx_gen::train::checkpoint::checkpoint_filename("adapter", 1),
                        req.file_name.clone(),
                    ] {
                        let path = temp.path().join(file);
                        crate::training::save(&mut model, &adapter, &params, variant, &req, &path)
                            .unwrap();
                        let metadata =
                            mlx_gen::gen_core::weightsmeta::safetensors_file_metadata(&path)
                                .unwrap();
                        if words.is_empty() {
                            assert!(
                                !metadata.contains_key("triggerWords"),
                                "empty legacy words must stay absent"
                            );
                        } else {
                            assert_eq!(serde_json::from_str::<Vec<String>>(&metadata["triggerWords"]).unwrap(), words,
                                "saved adapter must carry request words, not config caption injection");
                        }
                    }
                }
            }
        }
    });
}
