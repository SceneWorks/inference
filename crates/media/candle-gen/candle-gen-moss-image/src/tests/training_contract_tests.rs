//! Weights-free production-graph oracles. CPU results never assert CUDA execution.
use super::*;
use crate::training;
use candle_gen::candle_core::{backprop::GradStore, Var};
use candle_gen::gen_core::{moss_image_training as contract, NetworkType, TrainingConfig};
use candle_gen::train::{
    lora::{self, LoraSet},
    optim::TrainOptimizer,
};
use candle_gen::CandleError;
use std::collections::BTreeMap;

fn data(device: &Device, dtype: DType) -> (Transformer, Weights, Tensor, Tensor) {
    let d = fixture("fixture.json");
    let base = Weights::from_map(
        d["weights"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    tensor(v, DType::BF16)
                        .to_dtype(dtype)
                        .unwrap()
                        .to_device(device)
                        .unwrap(),
                )
            })
            .collect(),
    );
    let model = Transformer::load(&base, geometry(&d), None).unwrap();
    let x = tensor(&d["inputs"]["hidden_states"], DType::F32)
        .narrow(0, 0, 1)
        .unwrap()
        .to_device(device)
        .unwrap();
    let context = tensor(&d["inputs"]["encoder_hidden_states"], DType::F32)
        .narrow(0, 0, 1)
        .unwrap()
        .to_device(device)
        .unwrap();
    assert!(model.device().same_device(device));
    (model, base, x, context)
}
fn config(network_type: NetworkType, dtype: DType) -> TrainingConfig {
    TrainingConfig {
        rank: 3,
        alpha: 0.7,
        decompose_factor: 2,
        network_type,
        train_dtype: if dtype == DType::BF16 { "bf16" } else { "f32" }.into(),
        seed: 173,
        steps: 3,
        batch_size: 2,
        gradient_accumulation: 2,
        gradient_checkpointing: false,
        lora_target_modules: vec!["q".into(), "ffn_up_gate".into(), "ffn_down".into()],
        ..Default::default()
    }
}
fn values(t: &Tensor) -> Vec<f32> {
    loader::tensor_f32(t).unwrap()
}
fn close(a: &Tensor, b: &Tensor, tol: f32) {
    assert_eq!(a.dims(), b.dims());
    let a = values(a);
    let b = values(b);
    assert!(a.iter().chain(&b).all(|x| x.is_finite()));
    let error = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max);
    assert!(error <= tol, "max error {error} > {tol}");
}
fn nonzero(set: &LoraSet) {
    for (i, var) in set.vars.iter().enumerate() {
        let values: Vec<_> = (0..var.elem_count())
            .map(|j| 0.03 * ((j + i * 3) as f32 * 0.7).sin() + 0.01)
            .collect();
        var.set(&Tensor::from_vec(values, var.dims(), var.device()).unwrap())
            .unwrap();
    }
}
fn gradient(grads: &GradStore, var: &Var) -> Tensor {
    grads
        .get(var.as_tensor())
        .expect("actual trainable leaf must receive gradient")
        .clone()
}
fn generation(model: &Transformer, x: &Tensor, context: &Tensor) -> Tensor {
    model
        .forward(
            x,
            375.,
            context,
            &vec![1; context.dim(1).unwrap()],
            &[],
            &CancelFlag::new(),
        )
        .unwrap()
}
fn sampled_generation(
    model: &Transformer,
    x: &Tensor,
    context: &Tensor,
    variant: MossImageVariant,
) -> Tensor {
    let request = GenerationRequest {
        width: x.dim(3).unwrap() as u32 * 16,
        height: x.dim(2).unwrap() as u32 * 16,
        count: 1,
        steps: Some(3),
        guidance: Some(1.),
        seed: Some(716),
        ..Default::default()
    };
    let resolved = MossImageResolved::resolve(variant, &request).unwrap();
    let conditioning = text_encoder::Conditioning {
        rows: vec![context.clone()],
        receipts: vec![],
    };
    let noise = values(x);
    pipeline::sample_graph(
        model,
        None,
        &request,
        &resolved,
        MossImageAdvancedInputs {
            noise: Some(&noise),
            output: MossImageOutputKind::Latents,
            ..Default::default()
        },
        &conditioning,
        None,
        &mut |_| {},
    )
    .unwrap()
    .latents
}
fn adapter_spec(path: std::path::PathBuf, lokr: bool, scale: f32) -> AdapterSpec {
    AdapterSpec {
        path,
        scale,
        kind: if lokr {
            AdapterKind::Lokr
        } else {
            AdapterKind::Lora
        },
        pass_scales: None,
        moe_expert: None,
    }
}
fn save(set: &LoraSet, variant: MossImageVariant, path: &std::path::Path) {
    let meta: HashMap<_, _> = contract::metadata(variant, contract::SAMPLER)
        .into_iter()
        .collect();
    if set.kind == lora::AdapterKind::Lora {
        lora::save_lora_peft(set, "", &meta, path).unwrap();
    } else {
        lora::save_lokr(set, &meta, path).unwrap();
    }
}

/// Deterministic source-closure provenance, not a freshness gate. A receiver validates the
/// adapter/raw-factor hashes but does not require its current source tree to equal the producer.
fn fixture_lineage(path: &std::path::Path) -> Value {
    use sha2::{Digest, Sha256};
    let sources: &[(&str, &[u8])] = &[
        ("Cargo.lock", include_bytes!("../../../../../../Cargo.lock")),
        (
            "moss-contract",
            include_bytes!("../../../../../contracts/gen-core/src/moss_image_training.rs"),
        ),
        (
            "author-fixture",
            include_bytes!(
                "../../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_author/fixture.json"
            ),
        ),
        ("transformer", include_bytes!("../transformer.rs")),
        ("adapters", include_bytes!("../adapters.rs")),
        ("training", include_bytes!("../training.rs")),
        ("sampling", include_bytes!("../pipeline.rs")),
        (
            "shared-adapter-writer",
            include_bytes!("../../../candle-gen/src/train/lora.rs"),
        ),
        ("fixture-producer", include_bytes!("training_contract_tests.rs")),
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
fn verify_fixture_lineage(path: &std::path::Path, record: &Value) {
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
/// Optional evidence capture from an already-executed CPU production test. Absence of the
/// explicit output path is not cross-backend proof; committed exchange fixtures are checked
/// separately by the importing backend.
fn export_exchange(
    variant: MossImageVariant,
    network: NetworkType,
    path: &std::path::Path,
    model: &Transformer,
    x: &Tensor,
    context: &Tensor,
) {
    if !model.device().is_cpu() {
        return;
    }
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
    let stem = format!("candle-{}-{method}", variant.provider_id());
    let array = |t: &Tensor| serde_json::json!({"shape":t.dims(),"values":values(t)});
    let velocity = generation(model, x, context);
    let sampled = sampled_generation(model, x, context, variant);
    let receipt = serde_json::json!({"schema":1,"producer":"candle","variant":variant.provider_id(),"network_type":method,"base_storage":"bf16","time":375.,"sample_steps":3,"input":array(x),"context":array(context),"velocity":array(&velocity),"sampled":array(&sampled),"adapter":format!("{stem}.safetensors"),"lineage":fixture_lineage(path)});
    std::fs::copy(path, output.join(format!("{stem}.safetensors"))).unwrap();
    std::fs::write(
        output.join(format!("{stem}.json")),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
}
fn native_graph_cases(device: Device) {
    for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
        for network in [NetworkType::Lora, NetworkType::Lokr] {
            for dtype in [DType::F32, DType::BF16] {
                let (mut model, base, x, context) = data(&device, dtype);
                let snapshot: HashMap<_, _> = base
                    .keys()
                    .map(|k| (k.clone(), values(&base.require(k).unwrap())))
                    .collect();
                let mut cfg = config(network, dtype);
                if dtype == DType::BF16 {
                    let mut selective =
                        Transformer::load(&base, geometry(&fixture("fixture.json")), None).unwrap();
                    let mut selected = cfg.clone();
                    selected.lora_target_modules = vec!["o".into()];
                    let factors = selective.training_adapter(&selected).unwrap();
                    nonzero(&factors);
                    assert_eq!(selective.block_stream_dtype().unwrap(),DType::F32,"O-only adapter must retain its FP32 skip stream through a frozen BF16 FFN down projection");
                }
                let set = model.training_adapter(&cfg).unwrap();
                let portable = f64::from(cfg.alpha / cfg.rank as f32);
                assert_ne!(
                    portable,
                    f64::from(cfg.alpha) / f64::from(cfg.rank),
                    "nonbinary test scale must distinguish the portable FP32 division"
                );
                let scales = model.adapter_scales();
                assert!(
                    !scales.is_empty() && scales.iter().all(|s| s.to_bits() == portable.to_bits()),
                    "actual installed factor scales must equal MLX/import FP32 alpha/rank"
                );
                assert!(set
                    .named_vars()
                    .iter()
                    .all(|(name, var)| name.starts_with("layers.")
                        && name.contains(".proj.")
                        && var.dtype() == DType::F32
                        && var.device().same_device(&device)));
                assert!(
                    set.named_vars()
                        .iter()
                        .any(|(name, _)| name.contains("ffn_up_gate")),
                    "fused up/gate must be adapted"
                );
                let noise =
                    candle_gen::train::flow_match::sample_noise(x.dims(), 91, &device).unwrap();
                // Exercise the real producer refusal, not just a test-side missing-gradient
                // assertion: detach the installed surface while preserving the optimizer leaves.
                model.clear_residuals();
                let detached = training::loss_grads(
                    &model,
                    &set,
                    &cfg,
                    &x,
                    &context,
                    &noise,
                    0.375,
                    None,
                    &CancelFlag::new(),
                )
                .expect_err("detached trainable surface must be refused");
                assert!(detached
                    .to_string()
                    .contains("autograd omitted trainable factor"));
                let set = model.training_adapter(&cfg).unwrap();
                nonzero(&set);
                let (loss, plain) = training::loss_grads(
                    &model,
                    &set,
                    &cfg,
                    &x,
                    &context,
                    &noise,
                    0.375,
                    None,
                    &CancelFlag::new(),
                )
                .unwrap();
                let noisy = (((&x * 0.625).unwrap()) + ((&noise * 0.375).unwrap())).unwrap();
                let target = noise.sub(&x).unwrap();
                let predicted = generation(&model, &noisy, &context);
                let generation_loss = candle_gen::train::flow_match::weighted_velocity_loss(
                    &predicted, &target, None, false,
                )
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
                assert!(
                    (loss - generation_loss).abs() < 1e-6,
                    "training must use the actual generation graph and velocity convention"
                );
                cfg.gradient_checkpointing = true;
                let (recomputed, checkpointed) = training::loss_grads(
                    &model,
                    &set,
                    &cfg,
                    &x,
                    &context,
                    &noise,
                    0.375,
                    None,
                    &CancelFlag::new(),
                )
                .unwrap();
                assert!(loss.is_finite() && (loss - recomputed).abs() < 1e-5);
                for (name, var) in set.named_vars() {
                    assert!(
                        plain.get(var.as_tensor()).is_some(),
                        "missing monolithic {variant:?}/{network:?}/{dtype:?} gradient {name}"
                    );
                    assert!(
                        checkpointed.get(var.as_tensor()).is_some(),
                        "missing checkpoint {variant:?}/{network:?}/{dtype:?} gradient {name}"
                    );
                    close(
                        &gradient(&plain, &var),
                        &gradient(&checkpointed, &var),
                        if dtype == DType::F32 { 2e-5 } else { 0.005 },
                    );
                }
                let first = &set.vars[0];
                let first_grad = gradient(&plain, first);
                assert!(
                    values(&first_grad).iter().any(|v| v.abs() > 1e-8),
                    "earliest factor must receive frozen-backbone derivative"
                );
                let early_plain = values(&first_grad);
                let early_checkpoint = values(&gradient(&checkpointed, first));
                let magnitude = early_plain.iter().map(|v| v.abs()).fold(0., f32::max);
                let error = early_plain
                    .iter()
                    .zip(&early_checkpoint)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f32::max);
                assert!(early_checkpoint.iter().any(|v|v.abs()>1e-8),"checkpoint must propagate to the earliest factor, not return a missing/zero cotangent");
                assert!(
                    error <= magnitude * if dtype == DType::F32 { 0.02 } else { 0.15 } + 1e-9,
                    "earliest checkpoint gradient relative error {error}/{magnitude}"
                );
                if dtype == DType::F32 {
                    // Select the largest actual leaf derivative so finite differences cannot
                    // falsely agree with zero merely because Q's derivative is very small.
                    let (checked, index, analytic) = set
                        .vars
                        .iter()
                        .map(|var| {
                            let gradient = values(&gradient(&plain, var));
                            let (index, &value) = gradient
                                .iter()
                                .enumerate()
                                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                                .unwrap();
                            (var, index, value)
                        })
                        .max_by(|a, b| a.2.abs().total_cmp(&b.2.abs()))
                        .unwrap();
                    assert!(
                        analytic.abs() > 1e-6,
                        "finite-difference oracle must select an observable derivative"
                    );
                    let before = values(checked.as_tensor());
                    let eps = 0.01;
                    let eval = |offset: f32| {
                        let mut v = before.clone();
                        v[index] += offset;
                        checked
                            .set(&Tensor::from_vec(v, checked.dims(), &device).unwrap())
                            .unwrap();
                        // Independently reduce the actual FP32 generation prediction in f64;
                        // subtracting two rounded FP32 mean losses can hide a small derivative.
                        let prediction = values(&generation(&model, &noisy, &context));
                        let target = values(&target);
                        prediction
                            .iter()
                            .zip(&target)
                            .map(|(p, t)| (f64::from(*p) - f64::from(*t)).powi(2))
                            .sum::<f64>()
                            / prediction.len() as f64
                    };
                    let finite_difference = (eval(eps) - eval(-eps)) / (2. * f64::from(eps));
                    checked
                        .set(&Tensor::from_vec(before, checked.dims(), &device).unwrap())
                        .unwrap();
                    assert!(
                        (finite_difference - f64::from(analytic)).abs()
                            < 1e-6 + f64::from(analytic.abs()) * 0.05,
                        "finite difference {finite_difference}, actual gradient {analytic}"
                    );
                }
                let before = values(&generation(&model, &x, &context));
                let mut opt =
                    TrainOptimizer::from_config("adamw", set.vars.clone(), 0.02, 0.).unwrap();
                opt.step(&checkpointed).unwrap();
                model.validate_residuals().unwrap();
                let trained = generation(&model, &x, &context);
                assert!(
                    before
                        .iter()
                        .zip(values(&trained))
                        .any(|(a, b)| (a - b).abs() > 1e-7),
                    "actual optimizer update must change generation"
                );
                for key in base.keys() {
                    assert_eq!(
                        snapshot[key],
                        values(&base.require(key).unwrap()),
                        "frozen source {key} changed"
                    );
                }
                let temp = tempfile::tempdir().unwrap();
                let path = temp.path().join("trained.safetensors");
                save(&set, variant, &path);
                let meta = gen_core::weightsmeta::safetensors_file_metadata(&path).unwrap();
                contract::validate_metadata(variant, &meta).unwrap();
                assert_eq!(meta["alpha"], "0.7");
                let spec = adapter_spec(path.clone(), network == NetworkType::Lokr, 1.);
                let pin = PinnedWeightsFile::pin(&path).unwrap();
                let mut reloaded =
                    Transformer::load(&base, geometry(&fixture("fixture.json")), None).unwrap();
                let reports =
                    crate::adapters::apply(&mut reloaded, variant, &[spec], &[pin]).unwrap();
                assert_eq!(reports[0].applied, set.len());
                assert!(reports[0].skipped.is_empty());
                close(&trained, &generation(&reloaded, &x, &context), 1e-6);
                let sampled = sampled_generation(&model, &x, &context, variant);
                close(
                    &sampled,
                    &sampled_generation(&reloaded, &x, &context, variant),
                    1e-5,
                );
                if dtype == DType::F32 {
                    export_exchange(variant, network, &path, &reloaded, &x, &context);
                }
                let bare =
                    Transformer::load(&base, geometry(&fixture("fixture.json")), None).unwrap();
                assert!(
                    values(&sampled)
                        .iter()
                        .zip(values(&sampled_generation(&bare, &x, &context, variant)))
                        .any(|(a, b)| (a - b).abs() > 1e-7),
                    "saved adapter must affect the actual Euler generation graph"
                );
                if variant == MossImageVariant::Sft {
                    let dm = fixture("dmd_fixture.json");
                    let replacements = Weights::from_map(
                        dm["replacement_tensors"]
                            .as_object()
                            .unwrap()
                            .iter()
                            .map(|(k, v)| {
                                (k.clone(), tensor(v, DType::F32).to_device(&device).unwrap())
                            })
                            .collect(),
                    );
                    let dmd_base = transformer::install_replacements(&base, &replacements).unwrap();
                    let mut published = HashMap::new();
                    for (name, a) in dm["adapters"].as_object().unwrap() {
                        published.insert(
                            format!("diffusion_model.{name}.lora_down.weight"),
                            tensor(&a["down"], DType::F32).to_device(&device).unwrap(),
                        );
                        published.insert(
                            format!("diffusion_model.{name}.lora_up.weight"),
                            tensor(&a["up"], DType::F32).to_device(&device).unwrap(),
                        );
                        published.insert(
                            format!("diffusion_model.{name}.alpha"),
                            Tensor::new(a["alpha"].as_f64().unwrap() as f32, &device).unwrap(),
                        );
                    }
                    let published = Weights::from_map(published);
                    let mut dmd =
                        Transformer::load(&dmd_base, geometry(&dm), Some(&published)).unwrap();
                    let before = generation(&dmd, &x, &context);
                    crate::adapters::apply(
                        &mut dmd,
                        variant,
                        &[adapter_spec(path.clone(), network == NetworkType::Lokr, 0.)],
                        &[PinnedWeightsFile::pin(&path).unwrap()],
                    )
                    .unwrap();
                    close(&before, &generation(&dmd, &x, &context), 0.);
                    dmd.clear_residuals();
                    // A small trained LoKr can fall below the final frozen BF16 head's
                    // rounding quantum atop DMD. A nondefault strength makes this composition
                    // oracle discriminate installation; unit-strength reload is checked above.
                    crate::adapters::apply(
                        &mut dmd,
                        variant,
                        &[adapter_spec(
                            path.clone(),
                            network == NetworkType::Lokr,
                            16.,
                        )],
                        &[PinnedWeightsFile::pin(&path).unwrap()],
                    )
                    .unwrap();
                    assert!(
                        values(&before)
                            .iter()
                            .zip(values(&generation(&dmd, &x, &context)))
                            .any(|(a, b)| (a - b).abs() > 1e-7),
                        "custom adapter must compose with—not replace—the DMD residual/time stack: {variant:?}/{network:?}/{dtype:?}"
                    );
                }
                for strength in [-0.375, 0.] {
                    let mut target =
                        Transformer::load(&base, geometry(&fixture("fixture.json")), None).unwrap();
                    crate::adapters::apply(
                        &mut target,
                        variant,
                        &[adapter_spec(
                            path.clone(),
                            network == NetworkType::Lokr,
                            strength,
                        )],
                        &[PinnedWeightsFile::pin(&path).unwrap()],
                    )
                    .unwrap();
                    let actual = generation(&target, &x, &context);
                    if strength == 0. {
                        close(&actual, &generation(&bare, &x, &context), 0.);
                    } else {
                        assert!(values(&actual)
                            .iter()
                            .zip(values(&trained))
                            .any(|(a, b)| (a - b).abs() > 1e-7));
                    }
                }
            }
        }
    }
}
fn import_mlx_exchange(root: &std::path::Path) {
    for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
        for method in ["lora", "lokr"] {
            let stem = format!("mlx-{}-{method}", variant.provider_id());
            let record: Value = serde_json::from_slice(
                &std::fs::read(root.join(format!("{stem}.json")))
                    .expect("genuine MLX CPU fixture is required"),
            )
            .unwrap();
            assert_eq!(record["schema"], 1);
            assert_eq!(record["producer"], "mlx");
            assert_eq!(record["base_storage"], "f32");
            assert_eq!(record["variant"], variant.provider_id());
            assert_eq!(record["network_type"], method);
            let f = fixture("fixture.json");
            let base = weights(&f["weights"], DType::F32);
            let x = tensor(&record["input"], DType::F32);
            let context = tensor(&record["context"], DType::F32);
            let mut host = Transformer::load(&base, geometry(&f), None).unwrap();
            let path = root.join(format!("{stem}.safetensors"));
            verify_fixture_lineage(&path, &record);
            let reports = crate::adapters::apply(
                &mut host,
                variant,
                &[adapter_spec(path.clone(), method == "lokr", 1.)],
                &[PinnedWeightsFile::pin(path).unwrap()],
            )
            .unwrap();
            assert!(reports[0].applied > 0 && reports[0].skipped.is_empty());
            let time = record["time"].as_f64().unwrap() as f32;
            let actual = host
                .forward(
                    &x,
                    time,
                    &context,
                    &vec![1; context.dim(1).unwrap()],
                    &[],
                    &CancelFlag::new(),
                )
                .unwrap();
            compare(
                &actual,
                &record["velocity"],
                5e-5,
                "MLX-exported adapter CUDA-family CPU forward",
            );
            compare(
                &sampled_generation(&host, &x, &context, variant),
                &record["sampled"],
                0.0002,
                "MLX-exported adapter native Euler generation",
            );
            let bare = Transformer::load(&base, geometry(&f), None).unwrap();
            let baseline = bare
                .forward(
                    &x,
                    time,
                    &context,
                    &vec![1; context.dim(1).unwrap()],
                    &[],
                    &CancelFlag::new(),
                )
                .unwrap();
            assert!(
                values(&actual)
                    .iter()
                    .zip(values(&baseline))
                    .any(|(a, b)| (a - b).abs() > 1e-7),
                "MLX exported adapter must change native generation"
            );
        }
    }
}
#[test]
fn cpu_production_graph_four_cases_checkpoint_gradients_updates_and_strict_reload() {
    native_graph_cases(Device::Cpu);
    if let Some(path) = std::env::var_os("MOSS_PORTABLE_EXCHANGE_INPUT") {
        import_mlx_exchange(std::path::Path::new(&path));
    }
}
#[test]
fn committed_mlx_cpu_adapters_generate_portably() {
    // Both the fixture tensor constructor and strict host loader explicitly use Device::Cpu.
    // Missing producer bytes are a hard failure, never a skip or an empty export hook.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../contracts/gen-core/tests/fixtures/moss_image/portable_adapters");
    import_mlx_exchange(&root);
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_production_graph_four_cases_checkpoint_gradients_updates_and_strict_reload() {
    // This is actual CUDA execution, not a CPU oracle or a fallback. CI's CUDA lane owns the GPU.
    let device = Device::new_cuda(0).expect("CUDA lane requires an actual CUDA device");
    assert!(device.is_cuda());
    native_graph_cases(device);
}

#[test]
fn registered_candle_trainers_match_both_actual_generator_methods() {
    let registry = register_providers(ProviderRegistryBuilder::new())
        .build()
        .unwrap();
    for id in ["moss_image_base", "moss_image_sft"] {
        let trainer = registry
            .trainers()
            .find(|t| (t.descriptor)().id == id)
            .expect("real registered trainer");
        let d = (trainer.descriptor)();
        assert_eq!(d.backend, "candle");
        assert!(d.supports_lora && d.supports_lokr);
        let generator = registry
            .generators()
            .find(|g| (g.descriptor)().id == id)
            .unwrap();
        let g = (generator.descriptor)();
        assert!(g.capabilities.supports_lora && g.capabilities.supports_lokr);
    }
}

#[test]
fn cached_renderer_invalidates_on_adapter_replacement_or_removal() {
    for remove in [false, true] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("adapter.bin");
        std::fs::write(&path, b"original").unwrap();
        let mut generator = synthetic_generator();
        generator.sources.adapter_pins = vec![PinnedWeightsFile::pin(&path).unwrap()];
        let (model, _, _, _) = data(&Device::Cpu, DType::F32);
        candle_gen::lock_recover(&generator.state).renderer = Some(pipeline::Renderer {
            mode: MossImageAcceleration::Off,
            transformer: model,
            dataspace: None,
            adapter_reports: vec![],
        });
        assert!(candle_gen::lock_recover(&generator.state)
            .renderer
            .is_some());
        if remove {
            std::fs::remove_file(&path).unwrap();
        } else {
            std::fs::write(&path, b"replacement").unwrap();
        }
        assert!(generator.validate(&GenerationRequest::default()).is_err());
        let mut progress = |_: Progress| {};
        assert!(generator
            .generate(&GenerationRequest::default(), &mut progress)
            .is_err());
        assert!(candle_gen::lock_recover(&generator.state)
            .renderer
            .is_none());
    }
}

#[test]
fn structured_lokr_matches_independent_f64_kronecker_for_both_orders_and_decompositions() {
    for (a, b, c, d) in [(2, 3, 2, 4), (3, 2, 4, 2)] {
        for decomposed in [false, true] {
            let w1: Vec<f32> = (0..a * c).map(|i| (i as f32 - 2.) / 7.).collect();
            let w2: Vec<f32> = (0..b * d).map(|i| (i as f32 + 1.) / 11.).collect();
            let mut group = BTreeMap::new();
            for (name, rows, cols, data) in
                [("lokr_w1", a, c, w1.clone()), ("lokr_w2", b, d, w2.clone())]
            {
                if decomposed {
                    // Non-identity legs (2I × W/2) keep an independent known effective
                    // matrix while discriminating a dropped decomposed A leg.
                    let eye: Vec<f32> = (0..rows * rows)
                        .map(|i| if i / rows == i % rows { 2. } else { 0. })
                        .collect();
                    group.insert(
                        format!("{name}_a"),
                        Tensor::from_vec(eye, (rows, rows), &Device::Cpu).unwrap(),
                    );
                    group.insert(
                        format!("{name}_b"),
                        Tensor::from_vec(
                            data.into_iter().map(|v| v * 0.5).collect::<Vec<_>>(),
                            (rows, cols),
                            &Device::Cpu,
                        )
                        .unwrap(),
                    );
                } else {
                    group.insert(
                        name.into(),
                        Tensor::from_vec(data, (rows, cols), &Device::Cpu).unwrap(),
                    );
                }
            }
            let residual =
                crate::adapters::Residual::from_group(&group, a * b, c * d, true, -0.375).unwrap();
            let x: Vec<f32> = (0..2 * c * d).map(|i| (i as f32 - 3.) / 13.).collect();
            let actual = values(
                &residual
                    .forward(&Tensor::from_vec(x.clone(), (2, c * d), &Device::Cpu).unwrap())
                    .unwrap()
                    .unwrap(),
            );
            let mut expected = vec![0f64; 2 * a * b];
            for row in 0..2 {
                for i in 0..a {
                    for k in 0..b {
                        for j in 0..c {
                            for l in 0..d {
                                expected[row * a * b + i * b + k] +=
                                    f64::from(x[row * c * d + j * d + l])
                                        * f64::from(w1[i * c + j])
                                        * f64::from(w2[k * d + l])
                                        * (-0.375);
                            }
                        }
                    }
                }
            }
            assert!(actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| (f64::from(*a) - b).abs() < 2e-6));
            assert_eq!(
                residual.bytes(),
                group
                    .values()
                    .map(|t| (t.elem_count() * 4) as u64)
                    .sum::<u64>()
            );
        }
    }
    // Independent LoRA contraction at the same canonical non-binary FP32 scaling, before
    // any frozen BF16 base/head can round the residual away.
    let a: Vec<f32> = (0..15).map(|i| (i as f32 - 4.) / 17.).collect();
    let b: Vec<f32> = (0..21).map(|i| (i as f32 + 2.) / 23.).collect();
    let x: Vec<f32> = (0..10).map(|i| (i as f32 - 3.) / 11.).collect();
    let group = BTreeMap::from([
        (
            "lora_A.weight".into(),
            Tensor::from_vec(a.clone(), (3, 5), &Device::Cpu).unwrap(),
        ),
        (
            "lora_B.weight".into(),
            Tensor::from_vec(b.clone(), (7, 3), &Device::Cpu).unwrap(),
        ),
    ]);
    let scale = f64::from((0.7_f32 / 3.) * (-0.375_f32));
    let residual = crate::adapters::Residual::from_group(&group, 7, 5, false, scale).unwrap();
    let output = residual
        .forward(&Tensor::from_vec(x.clone(), (2, 5), &Device::Cpu).unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    let actual = values(&output);
    for row in 0..2 {
        for out in 0..7 {
            let expected = (0..3)
                .map(|r| {
                    (0..5)
                        .map(|i| f64::from(x[row * 5 + i]) * f64::from(a[r * 5 + i]))
                        .sum::<f64>()
                        * f64::from(b[out * 3 + r])
                })
                .sum::<f64>()
                * scale;
            assert!(
                (f64::from(actual[row * 7 + out]) - expected).abs() < 2e-6,
                "LoRA raw FP32 residual violates alpha/rank × user strength before BF16 rounding"
            );
        }
    }
}

fn write_adapter_file(
    path: &std::path::Path,
    tensors: &BTreeMap<String, Tensor>,
    meta: &BTreeMap<String, String>,
) {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    header.insert("__metadata__".into(), serde_json::to_value(meta).unwrap());
    for (name, tensor) in tensors {
        let start = payload.len();
        let dtype = match tensor.dtype() {
            DType::F32 => "F32",
            DType::BF16 => "BF16",
            other => panic!("unsupported fixture dtype {other:?}"),
        };
        for value in values(tensor) {
            if tensor.dtype() == DType::BF16 {
                payload.extend(((value.to_bits() >> 16) as u16).to_le_bytes());
            } else {
                payload.extend(value.to_le_bytes());
            }
        }
        header.insert(name.clone(),serde_json::json!({"dtype":dtype,"shape":tensor.dims(),"data_offsets":[start,payload.len()]}));
    }
    let mut json = serde_json::to_vec(&header).unwrap();
    while !json.len().is_multiple_of(8) {
        json.push(b' ');
    }
    let mut file = (json.len() as u64).to_le_bytes().to_vec();
    file.extend(json);
    file.extend(payload);
    std::fs::write(path, file).unwrap();
}
#[test]
fn strict_adapter_rejects_orphans_crossed_axes_nonfinite_and_derived_overflow() {
    let t = |v, shape| Tensor::full(v, shape, &Device::Cpu).unwrap();
    let valid = BTreeMap::from([
        ("lokr_w1".into(), t(1f32, (2, 2))),
        ("lokr_w2".into(), t(0.2, (3, 4))),
    ]);
    crate::adapters::Residual::from_group(&valid, 6, 8, true, 1.).unwrap();
    for factor in ["lokr_w1", "lokr_w2"] {
        let mut bad = valid.clone();
        bad.insert(format!("{factor}_a"), t(1., (2, 1)));
        assert!(crate::adapters::Residual::from_group(&bad, 6, 8, true, 1.).is_err());
        assert!(
            crate::adapters::Factor::from_group(&bad, factor).is_err(),
            "raw factor parser must reject competing/orphan representations too"
        );
    }
    let mut bad = valid.clone();
    bad.insert("lokr_w2".into(), t(1., (2, 6)));
    assert!(crate::adapters::Residual::from_group(&bad, 6, 8, true, 1.).is_err());
    for value in [f32::NAN, f32::INFINITY, f32::MAX] {
        let mut bad = valid.clone();
        bad.insert("lokr_w1".into(), t(value, (2, 2)));
        bad.insert("lokr_w2".into(), t(value, (3, 4)));
        assert!(crate::adapters::Residual::from_group(&bad, 6, 8, true, 1.).is_err());
    }
    let orphan = BTreeMap::from([("lora_A.weight".into(), t(1., (1, 8)))]);
    assert!(crate::adapters::Residual::from_group(&orphan, 6, 8, false, 1.).is_err());
    let temp = tempfile::tempdir().unwrap();
    let good_path = temp.path().join("good.safetensors");
    let mut metadata = contract::metadata(MossImageVariant::Base, contract::SAMPLER);
    metadata.extend([
        ("networkType".into(), "lora".into()),
        ("rank".into(), "2".into()),
        ("alpha".into(), "0.375".into()),
    ]);
    let factors = BTreeMap::from([
        ("layers.0.proj.q.lora_A.weight".into(), t(0.125, (2, 16))),
        ("layers.0.proj.q.lora_B.weight".into(), t(0.25, (16, 2))),
    ]);
    write_adapter_file(&good_path, &factors, &metadata);
    for defect in [
        "orphan",
        "unknown",
        "wrong_parent",
        "rank",
        "alpha",
        "rank_mismatch",
        "bf16",
        "nonfinite",
        "overflow",
    ] {
        let mut bad_meta = metadata.clone();
        let mut bad = factors.clone();
        match defect {
            "orphan" => {
                bad.remove("layers.0.proj.q.lora_B.weight");
            }
            "unknown" => {
                bad.insert("layers.0.proj.q.typo".into(), t(1., (1, 1)));
            }
            "wrong_parent" => {
                bad_meta.insert("base_revision".into(), "wrong".into());
            }
            "rank" => {
                bad_meta.insert("rank".into(), "oops".into());
            }
            "bf16" => {
                bad.insert(
                    "layers.0.proj.q.lora_A.weight".into(),
                    t(0.125, (2, 16)).to_dtype(DType::BF16).unwrap(),
                );
            }
            "alpha" => {
                bad_meta.insert("alpha".into(), "NaN".into());
            }
            "rank_mismatch" => {
                bad_meta.insert("rank".into(), "3".into());
            }
            "nonfinite" => {
                bad.insert("layers.0.proj.q.lora_A.weight".into(), t(f32::NAN, (2, 16)));
            }
            "overflow" => {
                bad.insert("layers.0.proj.q.lora_A.weight".into(), t(f32::MAX, (2, 16)));
                bad.insert("layers.0.proj.q.lora_B.weight".into(), t(f32::MAX, (16, 2)));
            }
            _ => unreachable!(),
        }
        let path = temp.path().join(format!("{defect}.safetensors"));
        write_adapter_file(&path, &bad, &bad_meta);
        let (mut host, _, x, context) = data(&Device::Cpu, DType::F32);
        let before = generation(&host, &x, &context);
        let specs = [
            adapter_spec(good_path.clone(), false, 1.),
            adapter_spec(path.clone(), false, 1.),
        ];
        let pins = [
            PinnedWeightsFile::pin(&good_path).unwrap(),
            PinnedWeightsFile::pin(path).unwrap(),
        ];
        assert!(
            crate::adapters::apply(&mut host, MossImageVariant::Base, &specs, &pins).is_err(),
            "malformed file {defect}"
        );
        close(&before, &generation(&host, &x, &context), 0.);
    }
}

/// A scaled finite coefficient does not make the pre-scale FP32 GEMMs finite.
#[test]
fn strict_application_rejects_unscaled_intermediate_overflow_before_success() {
    let temp = tempfile::tempdir().unwrap();
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
            let mut metadata = contract::metadata(variant, contract::SAMPLER);
            metadata.extend([
                (
                    "networkType".into(),
                    if lokr { "lokr" } else { "lora" }.into(),
                ),
                ("rank".into(), "1".into()),
                ("alpha".into(), alpha.to_string()),
                ("decomposeFactor".into(), "-1".into()),
            ]);
            let sparse = |shape: (usize, usize)| {
                let mut v = vec![0f32; shape.0 * shape.1];
                v[0] = if scaled { 1e18 } else { 2e19 };
                if scaled && shape.0 == 1 {
                    v.fill(1e18);
                }
                Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
            };
            let factors = if case == "lokr-reversed" {
                BTreeMap::from([
                    ("layers.0.proj.q.lokr_w1".into(), sparse((1, 4))),
                    ("layers.0.proj.q.lokr_w2".into(), sparse((16, 4))),
                ])
            } else if case == "lokr-decomposed" {
                // Effective first factor is zero, but its raw inner leg overflows for a
                // bounded input BEFORE multiplication by the zero outer leg.
                BTreeMap::from([
                    (
                        "layers.0.proj.q.lokr_w1_a".into(),
                        Tensor::zeros((4, 1), DType::F32, &Device::Cpu).unwrap(),
                    ),
                    (
                        "layers.0.proj.q.lokr_w1_b".into(),
                        Tensor::from_vec(vec![f32::MAX, f32::MAX, 0., 0.], (1, 4), &Device::Cpu)
                            .unwrap(),
                    ),
                    (
                        "layers.0.proj.q.lokr_w2".into(),
                        Tensor::eye(4, DType::F32, &Device::Cpu).unwrap(),
                    ),
                ])
            } else if lokr {
                BTreeMap::from([
                    ("layers.0.proj.q.lokr_w1".into(), sparse((4, 4))),
                    ("layers.0.proj.q.lokr_w2".into(), sparse((4, 4))),
                ])
            } else {
                BTreeMap::from([
                    ("layers.0.proj.q.lora_A.weight".into(), sparse((1, 16))),
                    ("layers.0.proj.q.lora_B.weight".into(), sparse((16, 1))),
                    (
                        "layers.0.proj.q.alpha".into(),
                        Tensor::from_vec(vec![alpha], (1,), &Device::Cpu).unwrap(),
                    ),
                ])
            };
            // Independent f64 witness: all raw coefficients and the scaled product are finite,
            // but the second raw contraction exceeds FP32 before alpha/rank is multiplied.
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
            write_adapter_file(&path, &factors, &metadata);
            let (mut host, _, _, _) = data(&Device::Cpu, DType::F32);
            let spec = adapter_spec(path.clone(), lokr, 1.);
            let pin = PinnedWeightsFile::pin(&path).unwrap();
            let result = crate::adapters::apply(&mut host, variant, &[spec], &[pin]);
            assert!(matches!(result, Err(CandleError::Unsupported(ref message))
                if message.contains("contraction") && message.contains("reduce factor magnitude")),
                "unsafe raw contraction must be actionably refused before application success: {result:?}");
            let zero = adapter_spec(path.clone(), lokr, 0.);
            let pin = PinnedWeightsFile::pin(&path).unwrap();
            assert!(
                crate::adapters::apply(&mut host, variant, &[zero], &[pin]).is_ok(),
                "finite disabled factors must not execute or be rejected for a skipped contraction"
            );
        }
    }
}

#[test]
fn periodic_and_final_exports_preserve_request_trigger_words() {
    let temp = tempfile::tempdir().unwrap();
    for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
        for network in [NetworkType::Lora, NetworkType::Lokr] {
            let (mut model, _, _, _) = data(&Device::Cpu, DType::F32);
            let mut cfg = config(network, DType::F32);
            cfg.trigger_word = Some("caption_only_not_export_words".into());
            let set = model.training_adapter(&cfg).unwrap();
            nonzero(&set);
            let mut req = candle_gen::gen_core::TrainingRequest {
                items: vec![],
                config: cfg,
                output_dir: temp.path().into(),
                file_name: "final.safetensors".into(),
                trigger_words: vec!["ohwx".into(), "quoted\"word".into()],
                cancel: CancelFlag::new(),
            };
            for words in [req.trigger_words.clone(), vec![]] {
                req.trigger_words = words.clone();
                for file in [
                    candle_gen::train::checkpoint::checkpoint_filename("adapter", 1),
                    req.file_name.clone(),
                ] {
                    let path = temp.path().join(file);
                    training::save(&model, &set, variant, &req, &path).unwrap();
                    let metadata = gen_core::weightsmeta::safetensors_file_metadata(&path).unwrap();
                    if words.is_empty() {
                        assert!(
                            !metadata.contains_key("triggerWords"),
                            "empty legacy words must stay absent"
                        );
                    } else {
                        assert_eq!(
                            serde_json::from_str::<Vec<String>>(&metadata["triggerWords"]).unwrap(),
                            words,
                            "saved adapter must carry request words, not config caption injection"
                        );
                    }
                }
            }
        }
    }
}
