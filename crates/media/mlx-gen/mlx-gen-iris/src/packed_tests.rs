//! The pre-quantized Iris artifacts ([`crate::packed`]) on the committed miniatures, every body in
//! a CPU child with Metal trapped: the CPU converter refuses over budget before reading anything
//! and never leaves the CPU; the packed artifact loads with no load-time quantization and is
//! BIT-IDENTICAL to the dense release quantized at load (backbone, adapters, text tower); its tier
//! is read from the layout; a request for another tier is refused; and the contract publishes no
//! calibration identity for a dense artifact requested at q8/q4.

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::iris::downstream::TaskExport;
use mlx_gen::gen_core::iris::{IrisConfig, IrisTask, TEXT_ENCODER_COMPONENT};
use mlx_gen::{AdapterKind, AdapterSpec, LoadSpec, Quant, WeightsSource};
use mlx_rs::{random, Array, Dtype};

use crate::packed::{
    artifact_tier, cpu_conversion_tensor_bytes, prequantize_cpu, text_encoder_tier, ConvertKind,
};

use crate::test_metal_guard as metal_guard;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn backbone_dir() -> PathBuf {
    fixtures().join("tiny-snapshot/iris")
}

fn in_cpu_child(test: &str, body: fn()) {
    const CHILD: &str = "IRIS_PACKED_CPU_CHILD";
    const DONE: &str = "IRIS_PACKED_CPU_CHILD_DONE";
    if std::env::var(CHILD).as_deref() == Ok(test) {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert!(metal_guard::forbid_metal());
        assert!(metal_guard::metal_forbidden());
        body();
        std::fs::write(std::env::var(DONE).unwrap(), b"ok").unwrap();
        return;
    }
    let done = tempfile::tempdir().unwrap();
    let marker = done.path().join("done");
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture", "--test-threads", "1"])
        .env(CHILD, test)
        .env(DONE, &marker)
        .status()
        .unwrap();
    assert!(status.success(), "{test} failed in its CPU child");
    assert!(marker.is_file(), "{test}: the CPU child ran no body");
}

fn host(a: &Array) -> Vec<f32> {
    let n: i32 = a.shape().iter().product();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    a.eval().unwrap();
    a.as_slice::<f32>().to_vec()
}

fn rand(shape: &[i32], seed: u64) -> Array {
    random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap())).unwrap()
}

fn bits(quant: Quant) -> i32 {
    crate::packed::quant_bits(quant).unwrap()
}

/// Convert `src` (CPU, ample budget) into a fresh dir under `root`.
fn convert(kind: ConvertKind, src: &Path, root: &Path, quant: Quant) -> PathBuf {
    let dst = root.join(format!("{kind:?}-{quant:?}").replace(['{', '}', ' ', ':'], ""));
    prequantize_cpu(kind, src, &dst, bits(quant), u64::MAX).unwrap();
    dst
}

fn load(
    dir: &Path,
    quant: Option<Quant>,
    adapters: &[AdapterSpec],
) -> mlx_gen::Result<crate::IrisDiT> {
    let config = IrisConfig::from_dir(dir)?;
    Ok(crate::model::load_backbone_quantized(
        dir,
        &config,
        Dtype::Bfloat16,
        adapters,
        IrisTask::Generation,
        crate::model::MODEL_ID,
        quant,
    )?
    .0)
}

fn converter_body() {
    let root = tempfile::tempdir().unwrap();
    let kind = ConvertKind::Export { depth: false };
    for quant in [Quant::Q4, Quant::Q8] {
        let required = cpu_conversion_tensor_bytes(kind, &backbone_dir(), bits(quant)).unwrap();
        assert!(required > 0);
        // Over budget: refused from the headers alone, before the destination exists.
        let dst = root.path().join(format!("refused-{quant:?}"));
        let refused = prequantize_cpu(kind, &backbone_dir(), &dst, bits(quant), required - 1);
        assert!(matches!(refused, Err(mlx_gen::Error::Unsupported(_))));
        assert!(!dst.exists(), "a refusal must precede any read or write");
        // Within budget, from a GPU default device: the conversion runs on the CPU (the Metal trap
        // would abort a GPU submission) and restores the caller's device.
        mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
        prequantize_cpu(kind, &backbone_dir(), &dst, bits(quant), required).unwrap();
        assert!(mlx_rs::Device::try_default().unwrap() == mlx_rs::Device::gpu());
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        // The tier is the layout's; the dense release reads as dense.
        let file = dst.join(mlx_gen::gen_core::iris::BACKBONE_WEIGHTS_FILE);
        assert_eq!(artifact_tier(&file).unwrap(), Some(quant));
        let recorded = mlx_gen::gen_core::weightsmeta::safetensors_file_metadata(&file).unwrap();
        assert_eq!(
            recorded[crate::packed::BITS_METADATA],
            bits(quant).to_string()
        );
        assert_eq!(recorded[crate::packed::GROUP_SIZE_METADATA], "64");
        assert!(dst.join("config.yaml").is_file());
        // Converting a packed artifact again is refused, not requantized.
        assert!(prequantize_cpu(kind, &dst, &root.path().join("again"), 4, u64::MAX).is_err());
    }
    assert_eq!(
        artifact_tier(&backbone_dir().join("model.safetensors")).unwrap(),
        None
    );
}

#[test]
fn the_cpu_converter_refuses_over_budget_and_never_leaves_the_cpu() {
    in_cpu_child(
        "packed_tests::the_cpu_converter_refuses_over_budget_and_never_leaves_the_cpu",
        converter_body,
    );
}

fn backbone_parity_body() {
    let root = tempfile::tempdir().unwrap();
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let m = &config.model;
    let layers = config.text_encoder.hidden_layers.len() as i32;
    let x = rand(&[2, 3, 16, 24], 3);
    let t = Array::from_slice(&[500.0f32, 250.0], &[2]);
    let states = rand(&[2, m.text_len as i32, layers, m.text_dim as i32], 4);
    let mask = vec![vec![1; m.text_len]; 2];
    let text = crate::TextBatch {
        states: &states,
        mask: &mask,
    };
    let lora = [AdapterSpec::new(
        fixtures().join("iris_lora.safetensors"),
        0.75,
        AdapterKind::Lora,
    )];
    let lokr = [AdapterSpec::new(
        fixtures().join("iris_lokr.safetensors"),
        1.0,
        AdapterKind::Lokr,
    )];
    for quant in [Quant::Q4, Quant::Q8] {
        let packed_dir = convert(
            ConvertKind::Export { depth: false },
            &backbone_dir(),
            root.path(),
            quant,
        );
        for adapters in [&[][..], &lora[..], &lokr[..]] {
            let at_load = load(&backbone_dir(), Some(quant), adapters).unwrap();
            // The packed artifact, requested by its tier and by none (the layout decides).
            for requested in [Some(quant), None] {
                let packed = load(&packed_dir, requested, adapters).unwrap();
                assert_eq!(
                    packed.resident_weight_bytes(),
                    at_load.resident_weight_bytes()
                );
                assert_eq!(
                    host(&packed.forward(&x, &t, &text).unwrap()),
                    host(&at_load.forward(&x, &t, &text).unwrap()),
                    "{quant:?} {requested:?} with {} adapter(s): the packed artifact must load \
                     bit-identical to the release quantized at load",
                    adapters.len()
                );
            }
        }
        // Another tier, or the f32 reference compute, is refused rather than requantized.
        let other = if quant == Quant::Q4 {
            Quant::Q8
        } else {
            Quant::Q4
        };
        assert!(matches!(
            load(&packed_dir, Some(other), &[]),
            Err(mlx_gen::Error::Unsupported(_))
        ));
        let f32_load = crate::model::load_backbone_quantized(
            &packed_dir,
            &config,
            Dtype::Float32,
            &[],
            IrisTask::Generation,
            crate::model::MODEL_ID,
            None,
        );
        assert!(matches!(f32_load, Err(mlx_gen::Error::Unsupported(_))));
    }
}

#[test]
fn a_packed_backbone_loads_bit_identical_to_the_release_quantized_at_load() {
    in_cpu_child(
        "packed_tests::a_packed_backbone_loads_bit_identical_to_the_release_quantized_at_load",
        backbone_parity_body,
    );
}

/// The miniature tower with an aligned MLP (`intermediate_size: 64`; the released miniature's 48
/// has no affine tier), seeded MLP weights, everything else the fixture's.
fn aligned_tiny_text_encoder(root: &Path) -> PathBuf {
    let source = fixtures().join("tiny-snapshot/text_encoder");
    let dir = root.join("text_encoder");
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(&source).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_owned();
        if name != "model.safetensors" && name != "config.json" {
            std::fs::copy(&path, dir.join(name)).unwrap();
        }
    }
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(source.join("config.json")).unwrap())
            .unwrap();
    config["text_config"]["intermediate_size"] = 64.into();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let mut tensors = Array::load_safetensors(source.join("model.safetensors")).unwrap();
    let mut names: Vec<String> = tensors.keys().cloned().collect();
    names.sort();
    for (i, name) in names.iter().enumerate() {
        let shape = if name.contains(".mlp.gate_proj.") || name.contains(".mlp.up_proj.") {
            [64, 32]
        } else if name.contains(".mlp.down_proj.") {
            [32, 64]
        } else {
            continue;
        };
        let key = random::key(26266 + i as u64).unwrap();
        tensors.insert(
            name.clone(),
            random::normal::<f32>(&shape[..], None, Some(0.02), Some(&key)).unwrap(),
        );
    }
    let refs: Vec<(&str, &Array)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(refs, None, dir.join("model.safetensors")).unwrap();
    dir
}

fn text_encoder_parity_body() {
    use crate::text_encoder::IrisTextEncoder;
    let root = tempfile::tempdir().unwrap();
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let tower = aligned_tiny_text_encoder(root.path());
    assert_eq!(text_encoder_tier(&tower).unwrap(), None);
    for quant in [Quant::Q4, Quant::Q8] {
        let packed_dir = convert(ConvertKind::TextEncoder, &tower, root.path(), quant);
        assert_eq!(text_encoder_tier(&packed_dir).unwrap(), Some(quant));
        // Only the language tower is written: no vision tensor, one file, no index.
        let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(
            packed_dir.join("model.safetensors"),
        )
        .unwrap();
        assert!(headers.iter().all(|h| h
            .name
            .starts_with(crate::text_encoder::LANGUAGE_TOWER_PREFIX)));
        assert!(!packed_dir.join("model.safetensors.index.json").exists());
        let at_load =
            IrisTextEncoder::load_quantized(&tower, &config.text_encoder, Some(quant)).unwrap();
        for requested in [Some(quant), None] {
            let packed =
                IrisTextEncoder::load_quantized(&packed_dir, &config.text_encoder, requested)
                    .unwrap();
            assert!(packed.is_quantized());
            assert_eq!(
                packed.resident_weight_bytes(),
                at_load.resident_weight_bytes()
            );
            for prompt in ["a red fox in the snow", ""] {
                let (a, b) = (
                    packed.encode(prompt).unwrap(),
                    at_load.encode(prompt).unwrap(),
                );
                assert_eq!(
                    host(&a.states),
                    host(&b.states),
                    "{quant:?}: the packed tower must encode bit-identical to the tower \
                     quantized at load"
                );
            }
        }
        let other = if quant == Quant::Q4 {
            Quant::Q8
        } else {
            Quant::Q4
        };
        assert!(
            IrisTextEncoder::load_quantized(&packed_dir, &config.text_encoder, Some(other))
                .is_err()
        );
    }
}

#[test]
fn a_packed_text_tower_loads_bit_identical_to_the_tower_quantized_at_load() {
    in_cpu_child(
        "packed_tests::a_packed_text_tower_loads_bit_identical_to_the_tower_quantized_at_load",
        text_encoder_parity_body,
    );
}

fn identity_body() {
    use crate::memory_strategy::memory_strategy_contract;
    let root = tempfile::tempdir().unwrap();
    let tower = aligned_tiny_text_encoder(root.path());
    let generation = |backbone: &Path, text: &Path, quant: Option<Quant>| {
        let mut spec = LoadSpec::new(WeightsSource::Dir(backbone.to_path_buf()));
        spec.quantize = quant;
        spec.components.insert(
            TEXT_ENCODER_COMPONENT.into(),
            WeightsSource::Dir(text.to_path_buf()),
        );
        memory_strategy_contract(crate::model::MODEL_ID, &spec)
            .unwrap()
            .calibration
            .is_some()
    };
    // The dense release at its own (dense) tier: identity. Requested at q8/q4: a load-time
    // requantization — no identity.
    assert!(generation(&backbone_dir(), &tower, None));
    for quant in [Quant::Q4, Quant::Q8] {
        assert!(
            !generation(&backbone_dir(), &tower, Some(quant)),
            "a dense artifact requested at {quant:?} must publish no calibration identity"
        );
        let backbone = convert(
            ConvertKind::Export { depth: false },
            &backbone_dir(),
            root.path(),
            quant,
        );
        let text = convert(ConvertKind::TextEncoder, &tower, root.path(), quant);
        assert!(generation(&backbone, &text, Some(quant)));
        assert!(
            generation(&backbone, &text, None),
            "the layout names the tier"
        );
        // A packed backbone beside a dense tower is not one tier.
        assert!(!generation(&backbone, &tower, Some(quant)));
        let other = if quant == Quant::Q4 {
            Quant::Q8
        } else {
            Quant::Q4
        };
        assert!(!generation(&backbone, &text, Some(other)));
    }
    // The encoder-free tasks: a packed export at its tier publishes; the dense export asked for
    // at q4 publishes none, and its loader refuses rather than requantize.
    for (task, dir, depth) in [
        (IrisTask::Depth, fixtures().join("tiny-depth"), true),
        (
            IrisTask::Restoration,
            fixtures().join("tiny-snapshot/upscaler"),
            false,
        ),
    ] {
        let id = crate::memory_strategy::provider_id(task);
        let contract = |dir: &Path, quant: Option<Quant>| {
            let spec = LoadSpec::new(WeightsSource::Dir(dir.to_path_buf())).with_quant_opt(quant);
            memory_strategy_contract(id, &spec)
                .unwrap()
                .calibration
                .is_some()
        };
        assert!(contract(&dir, None));
        assert!(!contract(&dir, Some(Quant::Q4)));
        let packed = convert(ConvertKind::Export { depth }, &dir, root.path(), Quant::Q4);
        assert!(packed.join("empty_prompt.safetensors").is_file());
        assert!(contract(&packed, Some(Quant::Q4)));
        assert!(contract(&packed, None));
        assert!(!contract(&packed, Some(Quant::Q8)));
        let spec = |dir: &Path, quant: Option<Quant>| {
            LoadSpec::new(WeightsSource::Dir(dir.to_path_buf())).with_quant_opt(quant)
        };
        match task {
            IrisTask::Depth => {
                assert!(crate::depth::load_depth(&spec(&dir, Some(Quant::Q4))).is_err());
                let loaded = crate::depth::load_depth(&spec(&packed, Some(Quant::Q4))).unwrap();
                assert!(loaded.memory_strategy_contract().calibration.is_some());
                let export = TaskExport::from_dir(&packed, task, id).unwrap();
                let rgb = rand(&[1, 3, 16, 16], 9);
                let out = crate::depth::load_depth_export(export, Dtype::Bfloat16)
                    .unwrap()
                    .forward(&rgb)
                    .unwrap();
                assert!(host(&out).iter().all(|v| v.is_finite()));
            }
            _ => {
                assert!(
                    crate::restoration::IrisRestorer::load(&spec(&dir, Some(Quant::Q4))).is_err()
                );
                let loaded = crate::restoration::IrisRestorer::load(&spec(&packed, None)).unwrap();
                assert!(loaded.memory_strategy_contract().calibration.is_some());
            }
        }
    }
}

#[test]
fn a_dense_artifact_requested_at_a_packed_tier_publishes_no_identity() {
    in_cpu_child(
        "packed_tests::a_dense_artifact_requested_at_a_packed_tier_publishes_no_identity",
        identity_body,
    );
}

trait WithQuantOpt {
    fn with_quant_opt(self, quant: Option<Quant>) -> Self;
}

impl WithQuantOpt for LoadSpec {
    fn with_quant_opt(mut self, quant: Option<Quant>) -> Self {
        self.quantize = quant;
        self
    }
}

// ---- The recorded pack (sc-26270 review): the tier is never inferred from shapes at an assumed
// group size, a partially packed artifact never loads, and the text tower's group is checked.

/// Rewrite the safetensors file at `file` in place through `edit` (tensors and `__metadata__`).
fn rewrite(
    file: &Path,
    edit: impl FnOnce(
        &mut std::collections::HashMap<String, Array>,
        &mut std::collections::HashMap<String, String>,
    ),
) {
    let (mut tensors, mut metadata) = Array::load_safetensors_with_metadata(file).unwrap();
    for array in tensors.values() {
        array.eval().unwrap();
    }
    edit(&mut tensors, &mut metadata);
    let refs: Vec<(&str, &Array)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(refs, Some(&metadata), file).unwrap();
}

fn unsupported<T>(result: mlx_gen::Result<T>, needle: &str) {
    match result {
        Err(mlx_gen::Error::Unsupported(message)) => {
            assert!(message.contains(needle), "`{needle}` not in: {message}")
        }
        Err(other) => panic!("expected a typed Unsupported refusal ({needle}), got {other}"),
        Ok(_) => panic!("expected a typed Unsupported refusal ({needle}), got Ok"),
    }
}

fn metadata(bits: &str, group: &str) -> std::collections::HashMap<String, String> {
    std::collections::HashMap::from([
        (crate::packed::BITS_METADATA.to_owned(), bits.to_owned()),
        (
            crate::packed::GROUP_SIZE_METADATA.to_owned(),
            group.to_owned(),
        ),
    ])
}

/// Build the backbone straight from the map of `file` (no `artifact_tier` gate first): the
/// [`crate::nn::Loader`]'s own checks.
fn build_from_map(file: &Path) -> mlx_gen::Result<crate::IrisDiT> {
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let weights = mlx_gen::weights::Weights::from_file(file).unwrap();
    crate::IrisDiT::from_weights_quantized(&weights, &config.model, Dtype::Bfloat16, None, None)
}

fn group_128_body() {
    let root = tempfile::tempdir().unwrap();
    // One [64, 256] projection packed at q4 over group 128: codes [64, 32], scales [64, 2] — the
    // shapes a group-64 reading takes for q8.
    let weight = rand(&[64, 256], 7).as_dtype(Dtype::Bfloat16).unwrap();
    let (codes, scales, biases) =
        mlx_llm::primitives::quant::quantize_on_stored_grid(&weight, 128, 4).unwrap();
    let file = root.path().join("model.safetensors");
    let write = |meta: Option<std::collections::HashMap<String, String>>| {
        let refs = [
            ("blocks.0.proj.weight", &codes),
            ("blocks.0.proj.scales", &scales),
            ("blocks.0.proj.biases", &biases),
        ];
        Array::save_safetensors(refs, meta.as_ref(), &file).unwrap();
    };
    // Recorded truthfully: read as q4 at group 128 and refused — the loader unpacks at 64.
    write(Some(metadata("4", "128")));
    unsupported(artifact_tier(&file), "group size 128");
    // Unrecorded: refused, never inferred as q8 at an assumed group.
    write(None);
    unsupported(artifact_tier(&file), crate::packed::BITS_METADATA);
    // Recorded at the loader's group: the shapes decode to 8 bits there, not the recorded 4.
    write(Some(metadata("4", "64")));
    unsupported(artifact_tier(&file), "not the recorded 4");

    // A whole converted artifact whose metadata records group 128: refused at tier detection, on
    // the load route, and by the loader's own map check.
    let packed = convert(
        ConvertKind::Export { depth: false },
        &backbone_dir(),
        root.path(),
        Quant::Q4,
    );
    let weights = packed.join(mlx_gen::gen_core::iris::BACKBONE_WEIGHTS_FILE);
    rewrite(&weights, |_, meta| {
        meta.insert(crate::packed::GROUP_SIZE_METADATA.into(), "128".into());
    });
    unsupported(artifact_tier(&weights), "group size 128");
    unsupported(load(&packed, None, &[]), "group size 128");
    unsupported(build_from_map(&weights), "records");
}

#[test]
fn a_q4_pack_at_group_128_is_never_read_as_q8() {
    in_cpu_child(
        "packed_tests::a_q4_pack_at_group_128_is_never_read_as_q8",
        group_128_body,
    );
}

fn partial_pack_body() {
    let root = tempfile::tempdir().unwrap();
    let packed = convert(
        ConvertKind::Export { depth: false },
        &backbone_dir(),
        root.path(),
        Quant::Q8,
    );
    let weights = packed.join(mlx_gen::gen_core::iris::BACKBONE_WEIGHTS_FILE);
    // One target projection (64 wide, quantized by `ProjectionLayout::of`) stored dense.
    let target = "blocks.0.attn.q_proj_x";
    let dense = mlx_gen::weights::Weights::from_file(backbone_dir().join("model.safetensors"))
        .unwrap()
        .require(&format!("{target}.weight"))
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    rewrite(&weights, |tensors, _| {
        tensors.remove(&format!("{target}.scales")).unwrap();
        tensors.remove(&format!("{target}.biases")).unwrap();
        tensors.insert(format!("{target}.weight"), dense);
    });
    unsupported(artifact_tier(&weights), "partially packed");
    unsupported(load(&packed, Some(Quant::Q8), &[]), "partially packed");
    unsupported(build_from_map(&weights), "partially packed");
}

#[test]
fn a_partially_packed_backbone_is_refused() {
    in_cpu_child(
        "packed_tests::a_partially_packed_backbone_is_refused",
        partial_pack_body,
    );
}

fn text_group_body() {
    use crate::text_encoder::IrisTextEncoder;
    let root = tempfile::tempdir().unwrap();
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let tower = aligned_tiny_text_encoder(root.path());
    let packed = convert(ConvertKind::TextEncoder, &tower, root.path(), Quant::Q4);
    let group = crate::text_encoder::text_encoder_quant_group(
        &mlx_llm::ModelConfig::from_dir(&tower).unwrap(),
    )
    .unwrap();
    assert_eq!(group, 32, "the miniature tower's group (hidden 32)");
    let weights = packed.join("model.safetensors");
    let file_meta = mlx_gen::gen_core::weightsmeta::safetensors_file_metadata(&weights).unwrap();
    assert_eq!(file_meta[crate::packed::BITS_METADATA], "4");
    assert_eq!(file_meta[crate::packed::GROUP_SIZE_METADATA], "32");
    let pristine = std::fs::read(&weights).unwrap();
    let config_path = packed.join("config.json");
    let pristine_config = std::fs::read_to_string(&config_path).unwrap();
    let refused = |needle: &str| {
        unsupported(crate::packed::text_encoder_tier(&packed), needle);
        unsupported(
            IrisTextEncoder::load_quantized(&packed, &config.text_encoder, None),
            needle,
        );
    };

    // config.json's group is not the tower's.
    let mut cfg: serde_json::Value = serde_json::from_str(&pristine_config).unwrap();
    cfg["text_config"]["quantization"]["group_size"] = 64.into();
    std::fs::write(&config_path, cfg.to_string()).unwrap();
    refused("the tower's group size is Some(32)");
    std::fs::write(&config_path, &pristine_config).unwrap();

    // The weight file records another group than config.json / the tower.
    rewrite(&weights, |_, meta| {
        meta.insert(crate::packed::GROUP_SIZE_METADATA.into(), "64".into());
    });
    refused("group size 64");
    std::fs::write(&weights, &pristine).unwrap();

    // One projection's codes / scales are packed at group 64 under a recorded group of 32.
    let down = format!(
        "{}layers.0.mlp.down_proj",
        crate::text_encoder::LANGUAGE_TOWER_PREFIX
    );
    let dense = Array::load_safetensors(tower.join("model.safetensors")).unwrap()
        [&format!("{down}.weight")]
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let (codes, scales, biases) =
        mlx_llm::primitives::quant::quantize_on_stored_grid(&dense, 64, 4).unwrap();
    rewrite(&weights, |tensors, _| {
        tensors.insert(format!("{down}.weight"), codes);
        tensors.insert(format!("{down}.scales"), scales);
        tensors.insert(format!("{down}.biases"), biases);
    });
    refused("not the recorded 4");
    std::fs::write(&weights, &pristine).unwrap();

    // Restored: the snapshot reads and loads at its tier again.
    assert_eq!(
        crate::packed::text_encoder_tier(&packed).unwrap(),
        Some(Quant::Q4)
    );
    IrisTextEncoder::load_quantized(&packed, &config.text_encoder, None).unwrap();
}

#[test]
fn a_text_encoder_group_size_mismatch_is_refused() {
    in_cpu_child(
        "packed_tests::a_text_encoder_group_size_mismatch_is_refused",
        text_group_body,
    );
}

fn missing_metadata_body() {
    use crate::text_encoder::IrisTextEncoder;
    let root = tempfile::tempdir().unwrap();
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let packed = convert(
        ConvertKind::Export { depth: false },
        &backbone_dir(),
        root.path(),
        Quant::Q4,
    );
    let weights = packed.join(mlx_gen::gen_core::iris::BACKBONE_WEIGHTS_FILE);
    rewrite(&weights, |_, meta| meta.clear());
    unsupported(artifact_tier(&weights), crate::packed::BITS_METADATA);
    unsupported(load(&packed, None, &[]), crate::packed::BITS_METADATA);

    let tower = aligned_tiny_text_encoder(root.path());
    let text = convert(ConvertKind::TextEncoder, &tower, root.path(), Quant::Q8);
    rewrite(&text.join("model.safetensors"), |_, meta| meta.clear());
    unsupported(
        crate::packed::text_encoder_tier(&text),
        crate::packed::BITS_METADATA,
    );
    unsupported(
        IrisTextEncoder::load_quantized(&text, &config.text_encoder, Some(Quant::Q8)),
        crate::packed::BITS_METADATA,
    );
}

#[test]
fn a_packed_artifact_without_recorded_metadata_is_refused() {
    in_cpu_child(
        "packed_tests::a_packed_artifact_without_recorded_metadata_is_refused",
        missing_metadata_body,
    );
}
