use super::*;

#[test]
fn prompt_encoding_is_required_only_for_missing_active_embeddings() {
    let values = vec![0.0; 4096];
    let supplied = MossImageEmbeddings {
        values: &values,
        batch: 1,
        sequence: 1,
        mask: None,
    };
    for positive in [None, Some(supplied)] {
        for negative in [None, Some(supplied)] {
            let inputs = MossImageAdvancedInputs {
                positive,
                negative,
                ..Default::default()
            };
            assert_eq!(
                needs_prompt_encoding(inputs, MossImageAcceleration::Off, 1.0),
                positive.is_none()
            );
            assert_eq!(
                needs_prompt_encoding(inputs, MossImageAcceleration::Off, 2.0),
                positive.is_none() || negative.is_none()
            );
            assert_eq!(
                needs_prompt_encoding(
                    inputs,
                    MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16),
                    2.0
                ),
                positive.is_none()
            );
        }
    }
}

use crate::transformer::{Geometry, Transformer};
use candle_gen::{
    candle_core::{DType, Tensor},
    weights::Weights,
};
use serde_json::Value;
use std::collections::HashMap;
fn fixture(name: &str) -> Value {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../contracts/gen-core/tests/fixtures/moss_image/tiny_author")
        .join(name);
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}
fn tensor(v: &Value, dtype: DType) -> Tensor {
    let shape: Vec<usize> = serde_json::from_value(v["shape"].clone()).unwrap();
    let data: Vec<f32> = serde_json::from_value(v["values"].clone()).unwrap();
    Tensor::from_vec(data, shape, &Device::Cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}
fn geometry(d: &Value) -> Geometry {
    let c = &d["config"];
    Geometry {
        dim: c["dim"].as_u64().unwrap() as usize,
        ffn: c["ffn_dim"].as_u64().unwrap() as usize,
        heads: c["num_heads"].as_u64().unwrap() as usize,
        layers: c["num_layers"].as_u64().unwrap() as usize,
        text: c["text_dim"].as_u64().unwrap() as usize,
        freq: c["freq_dim"].as_u64().unwrap() as usize,
        channels: c["latent_channels"].as_u64().unwrap() as usize,
    }
}
fn weights(v: &Value, dtype: DType) -> Weights {
    Weights::from_map(
        v.as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), tensor(v, dtype)))
            .collect(),
    )
}
fn checked_delta(a: &[f32], shape: &[usize], b: &[f32], expected_shape: &[usize]) -> f32 {
    assert_eq!(shape, expected_shape, "oracle shape");
    assert_eq!(a.len(), b.len(), "oracle count");
    assert_eq!(
        shape.iter().product::<usize>(),
        a.len(),
        "oracle shape/count"
    );
    assert!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "nonfinite oracle value"
    );
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}
fn compare(actual: &Tensor, expected: &Value, tolerance: f32, label: &str) {
    let a = loader::tensor_f32(actual).unwrap();
    let b: Vec<f32> = serde_json::from_value(expected["values"].clone()).unwrap();
    let shape: Vec<usize> = serde_json::from_value(expected["shape"].clone()).unwrap();
    let max = checked_delta(&a, actual.dims(), &b, &shape);
    assert!(max <= tolerance, "{label}: max abs {max} > {tolerance}");
}
#[test]
fn numerical_oracle_rejects_nan_truncation_and_shape_mismatch_sc26325() {
    assert_eq!(checked_delta(&[1., 2.], &[2], &[1., 2.], &[2]), 0.);
    for (a, shape, b, expected_shape) in [
        (vec![f32::NAN], vec![1], vec![0.], vec![1]),
        (vec![0.], vec![1], vec![f32::NAN], vec![1]),
        (vec![f32::INFINITY], vec![1], vec![0.], vec![1]),
        (vec![1.], vec![1], vec![1., 2.], vec![2]),
        (vec![1., 2.], vec![2], vec![1.], vec![2]),
        (vec![1., 2.], vec![1, 2], vec![1., 2.], vec![2, 1]),
    ] {
        assert!(
            std::panic::catch_unwind(|| checked_delta(&a, &shape, &b, &expected_shape)).is_err()
        );
    }
}
#[test]
fn descriptors_keep_acceleration_explicit() {
    for descriptor in [descriptor_base(), descriptor_sft()] {
        assert!(descriptor.capabilities.supports_lora);
        assert!(descriptor.capabilities.supports_lokr);
        assert!(descriptor.capabilities.supports_prompt_batch);
    }
    let request = GenerationRequest {
        steps: Some(8),
        ..Default::default()
    };
    assert_eq!(
        MossImageResolved::resolve(MossImageVariant::Sft, &request)
            .unwrap()
            .acceleration,
        MossImageAcceleration::Off
    );
}

#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn exact_author_graph_fp32_bf16_and_ragged_masks() {
    let d = fixture("fixture.json");
    let g = geometry(&d);
    let x = tensor(&d["inputs"]["hidden_states"], DType::F32);
    let c = tensor(&d["inputs"]["encoder_hidden_states"], DType::F32);
    let mask: Vec<f32> =
        serde_json::from_value(d["inputs"]["encoder_attention_mask"]["values"].clone()).unwrap();
    let times: Vec<f32> =
        serde_json::from_value(d["inputs"]["timestep"]["values"].clone()).unwrap();
    for case in d["cases"].as_array().unwrap() {
        let dtype = if case["dtype"] == "float32" {
            DType::F32
        } else {
            DType::BF16
        };
        let model = Transformer::load(&weights(&d["weights"], dtype), g, None).unwrap();
        let skip: Vec<u32> = serde_json::from_value(case["skip_layers"].clone()).unwrap();
        let mut outputs = Vec::new();
        for (b, time) in times.iter().enumerate() {
            outputs.push(
                model
                    .forward(
                        &x.narrow(0, b, 1).unwrap(),
                        *time,
                        &c.narrow(0, b, 1).unwrap(),
                        &mask[b * 4..(b + 1) * 4]
                            .iter()
                            .map(|v| *v as u8)
                            .collect::<Vec<_>>(),
                        &skip,
                        &CancelFlag::default(),
                    )
                    .unwrap(),
            );
        }
        compare(
            &Tensor::cat(&outputs, 0).unwrap(),
            &case["velocity"],
            if dtype == DType::F32 { 2e-6 } else { 0.001 },
            &format!("author {dtype:?}/{skip:?}"),
        );
    }
}
#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn exact_author_leaf_functions() {
    ordinary_author_time_silu(&Device::Cpu);
    let d = fixture("fixture.json");
    let l = &d["leaf_ops"];
    compare(
        &transformer::test_norm(
            &tensor(&l["rms_input"], DType::F32),
            tensor(&l["rms_weight"], DType::F32),
        )
        .unwrap(),
        &l["rms_expected"],
        2e-7,
        "RMSNorm",
    );
    compare(
        &transformer::situ(
            &tensor(&l["situ_hidden"], DType::F32),
            &tensor(&l["situ_gate"], DType::F32),
        )
        .unwrap(),
        &l["situ_expected"],
        2e-7,
        "situ",
    );
    compare(
        &transformer::rope(2, 3, 3, 8, DType::F32, &Device::Cpu).unwrap(),
        &l["rope_2x3_text3"],
        2e-7,
        "rope",
    );
}

pub(crate) fn ordinary_author_time_silu(device: &Device) {
    // CI_IMPACT: five tiny BF16 activation checks, reached only through existing
    // manual CPU/CUDA profiles. Reuse actual author nn.SiLU hooks; no new golden.
    let time_fixture: Value = serde_json::from_str(include_str!(
        "../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_v2/fixture.json"
    ))
    .unwrap();
    let mut eager_differs = false;
    for probe in time_fixture["time_probes"].as_array().unwrap() {
        let input = &probe["silu_input"];
        let shape: Vec<usize> = serde_json::from_value(input["shape"].clone()).unwrap();
        let values: Vec<f32> = serde_json::from_value(input["values"].clone()).unwrap();
        let input = Tensor::from_vec(values, shape, device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        compare(
            &transformer::ordinary_time_silu(&input).unwrap(),
            &probe["silu"],
            0.0,
            "author BF16 time SiLU",
        );
        let eager = loader::tensor_f32(&candle_gen::candle_nn::ops::silu(&input).unwrap()).unwrap();
        let expected: Vec<f32> = serde_json::from_value(probe["silu"]["values"].clone()).unwrap();
        eager_differs |= eager != expected;
    }
    assert!(
        eager_differs,
        "author time SiLU hooks must discriminate eager BF16 rounding"
    );
}

#[test]
fn same_owned_buffer_auth_rejects_substitution_even_while_original_pin_is_stable() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("transformer/pipeline_moss_image.py");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let approved=include_bytes!("../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_author/frozen_author_pipeline.py");
    std::fs::write(&path, approved).unwrap();
    let spec = LoadSpec::new(WeightsSource::Dir(tmp.path().into()));
    let asset = loader::Asset::new(
        &spec,
        path.clone(),
        MossImageAssetRole::Transformer,
        SFT_REPO,
        SFT_REVISION,
        "transformer/pipeline_moss_image.py",
    )
    .unwrap();
    let owned = asset.authenticated_bytes().unwrap();
    assert_eq!(&owned, approved);
    let mut substituted = owned.clone();
    substituted[100] ^= 1;
    asset.pin.verify_unchanged().unwrap();
    assert!(asset.verify_owned_bytes(&substituted).is_err());
    assert!(asset.verify_owned_bytes(&owned[..owned.len() - 1]).is_err());
    asset.verify_owned_bytes(&owned).unwrap();
    std::fs::write(path, b"replacement after owned read").unwrap();
    assert!(asset.authenticated_bytes().is_err());
    assert_eq!(
        &owned, approved,
        "the parser owns the exact authenticated bytes independently of later path replacement"
    );
}

#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn exact_unmerged_fp32_residual_and_fixed_noise_recurrence() {
    let d = fixture("dmd_fixture.json");
    let base = weights(&d["base_weights"], DType::BF16);
    let replace = weights(&d["replacement_tensors"], DType::F32);
    let base = transformer::install_replacements(&base, &replace).unwrap();
    let mut map = HashMap::new();
    for (name, a) in d["adapters"].as_object().unwrap() {
        map.insert(
            format!("diffusion_model.{name}.lora_down.weight"),
            tensor(&a["down"], DType::F32),
        );
        map.insert(
            format!("diffusion_model.{name}.lora_up.weight"),
            tensor(&a["up"], DType::F32),
        );
        map.insert(
            format!("diffusion_model.{name}.alpha"),
            Tensor::new(a["alpha"].as_f64().unwrap() as f32, &Device::Cpu).unwrap(),
        );
    }
    let adapter = Weights::from_map(map);
    let model = Transformer::load(&base, geometry(&d), Some(&adapter)).unwrap();
    let z = tensor(&d["noise"], DType::F32);
    let q = tensor(&d["mix_matrix"], DType::F32);
    let ctx = tensor(&d["conditioning"], DType::F32);
    let mask: Vec<f32> = serde_json::from_value(d["mask"]["values"].clone()).unwrap();
    let mask: Vec<_> = mask.iter().map(|v| *v as u8).collect();
    let codes: Vec<f32> = serde_json::from_value(d["step_codes"]["values"].clone()).unwrap();
    let mut x0 = Tensor::zeros(z.shape(), DType::F32, &Device::Cpu).unwrap();
    for (i, code) in codes.iter().enumerate() {
        let expected = &d["runs"]["correct_widened"]["steps"][i];
        let x = pipeline::dmd_input(&z, &q, &x0).unwrap();
        let previous = if i == 0 {
            Tensor::zeros(z.shape(), DType::F32, &Device::Cpu).unwrap()
        } else {
            tensor(
                &d["runs"]["correct_widened"]["steps"][i - 1]["x0"],
                DType::F32,
            )
        };
        compare(
            &pipeline::dmd_input(&z, &q, &previous).unwrap(),
            &expected["input"],
            2e-6,
            "DMD Q orientation",
        );
        compare(
            &x,
            &expected["input"],
            0.001,
            "DMD input with accumulated BF16 tolerance",
        );
        let v = model
            .forward(&x, *code, &ctx, &mask, &[], &CancelFlag::default())
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        compare(&v, &expected["velocity"], 0.001, "DMD velocity");
        x0 = z.sub(&v).unwrap();
        compare(&x0, &expected["x0"], 0.001, "DMD x0");
    }
    // Leaf residual dtype/value assertions discriminate BF16 merge or cast-down, independently
    // of end-to-end BF16 attention implementation tolerances.
    let probe = &d["projection_probe"];
    let name = probe["module"].as_str().unwrap();
    let input = tensor(&probe["input"], DType::F32);
    let p = transformer::test_projection(&base, name, geometry(&d).dim, Some(&adapter), &input)
        .unwrap();
    assert_eq!(p.dtype(), DType::F32);
    compare(&p, &probe["correct"], 2e-7, "unmerged projection");
}

#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn mask_and_cancellation_refuse_before_blocks() {
    let d = fixture("fixture.json");
    let model = Transformer::load(&weights(&d["weights"], DType::F32), geometry(&d), None).unwrap();
    let x = tensor(&d["inputs"]["hidden_states"], DType::F32);
    let c = tensor(&d["inputs"]["encoder_hidden_states"], DType::F32);
    assert!(model
        .forward(&x, 2.0, &c, &[0; 8], &[], &CancelFlag::default())
        .is_err());
    let cancel = CancelFlag::default();
    cancel.cancel();
    assert!(matches!(
        model.forward(&x, 2.0, &c, &[1; 8], &[], &cancel),
        Err(candle_gen::CandleError::Canceled)
    ));
}

pub(crate) fn synthetic_sources() -> Sources {
    Sources{variant:MossImageVariant::Sft,adapter_pins:vec![],selection:Some(MossImageAcceleration::Off),ordinary:vec![],mixed:None,full:None,headers:BTreeMap::new(),config:MossImageConfig::default(),
        text_config:candle_llm::models::qwen35::Qwen35Config::from_json(&serde_json::json!({"hidden_size":4096,"num_hidden_layers":2,"intermediate_size":24,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"vocab_size":32,"linear_num_value_heads":2,"linear_num_key_heads":1,"linear_key_head_dim":8,"linear_value_head_dim":8})).unwrap(),
        tokenizer:{let path=std::env::current_exe().unwrap();loader::Asset{#[cfg(feature="test-support")] fixture_digest:None,pin:PinnedWeightsFile::pin(path).unwrap(),role:MossImageAssetRole::Tokenizer,repo:SFT_REPO,revision:SFT_REVISION,relative:"tokenizer/tokenizer.json".into()}},
        spec:LoadSpec::new(WeightsSource::Dir("synthetic-unloaded-only".into()))}
}
fn synthetic_generator() -> MossImageGenerator {
    MossImageGenerator {
        descriptor: descriptor_sft(),
        sources: synthetic_sources(),
        device: Device::Cpu,
        state: Mutex::new(State::default()),
        contract: witness_sft(&LoadSpec::new(WeightsSource::Dir("unused".into()))).unwrap(),
        admission: Mutex::new(None),
    }
}
fn synthetic_report() -> MossImageExecutionReport {
    let r =
        MossImageResolved::resolve(MossImageVariant::Sft, &GenerationRequest::default()).unwrap();
    let assets = SNAPSHOT_FILES
        .iter()
        .filter(|p| **p != "LICENSE")
        .map(|path| {
            let role = if path.starts_with("transformer/") {
                MossImageAssetRole::Transformer
            } else if path.starts_with("text_encoder/") {
                MossImageAssetRole::TextEncoder
            } else if path.starts_with("tokenizer/") {
                MossImageAssetRole::Tokenizer
            } else if path.starts_with("scheduler/") {
                MossImageAssetRole::Scheduler
            } else {
                MossImageAssetRole::Vae
            };
            let a = moss_image_asset(SFT_REPO, SFT_REVISION, path).unwrap();
            MossImageAssetReceipt {
                role,
                repository: SFT_REPO.into(),
                revision: SFT_REVISION.into(),
                relative_path: (*path).into(),
                sha256: a.sha256.into(),
            }
        })
        .collect();
    let cond = MossImageConditioningInputReceipt {
        source: MossImageConditioningSource::Text,
        sha256: "a".repeat(64),
        sequence_length: 1,
        valid_tokens: 1,
    };
    MossImageExecutionReport::new(MossImageExecutionData {
        execution_device: MossImageExecutionDevice::Cpu,
        sampling: MossImageSamplingReceipt::Euler {
            sigmas: moss_image_schedule(r.steps, r.shift, r.training_shift).unwrap(),
        },
        resolved: r,
        assets,
        samples: vec![MossImageSampleReceipt {
            seed: Some(1),
            noise_sha256: "b".repeat(64),
        }],
        conditioning: vec![MossImageConditioningReceipt {
            positive: cond.clone(),
            negative: Some(cond),
        }],
        images_per_prompt: 1,
        latent_shape: [1, 128, 64, 64],
        output_kind: MossImageOutputKind::Images,
        component_dtypes: component_dtypes(MossImageAcceleration::Off, true),
        adapter_reports: vec![],
    })
    .unwrap()
}
#[test]
fn failure_and_cancel_clear_completed_receipt_and_allow_reentry() {
    // Synthetic metadata exercises failure state only; it cannot load or produce a model result.
    let g = synthetic_generator();
    candle_gen::lock_recover(&g.state).report = Some(synthetic_report());
    let req = GenerationRequest {
        width: 17,
        ..Default::default()
    };
    assert!(g.generate(&req, &mut |_| {}).is_err());
    assert!(g.moss_image_execution_report().is_none());
    candle_gen::lock_recover(&g.state).report = Some(synthetic_report());
    let req = GenerationRequest::default();
    req.cancel.cancel();
    assert!(matches!(
        g.generate(&req, &mut |_| {}),
        Err(gen_core::Error::Canceled)
    ));
    assert!(g.moss_image_execution_report().is_none());
    let req = GenerationRequest {
        moss_image: Some(MossImageOptions {
            acceleration: MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16),
            ..Default::default()
        }),
        ..Default::default()
    };
    assert!(g
        .generate(&req, &mut |_| {})
        .unwrap_err()
        .to_string()
        .contains("provision frozen"));
    assert!(g.moss_image_execution_report().is_none());
    assert!(g.validate(&GenerationRequest::default()).is_ok());
}
#[test]
fn supplied_embeddings_compact_noncontiguous_masks_and_hash_exact_inputs() {
    let mut values = vec![0f32; 2 * 3 * 4096];
    for (i, v) in values.iter_mut().enumerate() {
        *v = i as f32 / 10000.0;
    }
    let mask = [1, 0, 1, 0, 1, 0];
    let e = MossImageEmbeddings {
        values: &values,
        batch: 2,
        sequence: 3,
        mask: Some(&mask),
    };
    e.validate(3).unwrap();
    let c = text_encoder::supplied(e, &Device::Cpu).unwrap();
    assert_eq!(c.rows[0].dims(), [1, 2, 4096]);
    assert_eq!(c.rows[1].dims(), [1, 1, 4096]);
    assert_eq!(c.receipts[0].valid_tokens, 2);
    assert_eq!(
        c.receipts[1].sha256,
        text_encoder::digest_f32(&values[3 * 4096..], Some(&mask[3..]))
    );
    let actual = loader::tensor_f32(&c.rows[0]).unwrap();
    assert_eq!(&actual[..4096], &values[..4096]);
    assert_eq!(&actual[4096..], &values[8192..12288]);
}
#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn supplied_fp32_conditioning_preserves_precision_until_selected_projection_sc26324() {
    // sc-26324 (F-276): the same ingestion path runs on CUDA and CPU. Deliberately
    // non-BF16 values plus a ragged mask expose narrowing before a full-FP32 consumer.
    let mut values = vec![0f32; 3 * 4096];
    for (i, value) in values.iter_mut().enumerate() {
        *value = 1.001 + (i % 19) as f32 * 0.013;
    }
    let mask = [1, 0, 1];
    let supplied = text_encoder::supplied(
        MossImageEmbeddings {
            values: &values,
            batch: 1,
            sequence: 3,
            mask: Some(&mask),
        },
        &Device::Cpu,
    )
    .unwrap();
    assert_eq!(supplied.rows[0].dtype(), DType::F32);
    let compact = loader::tensor_f32(&supplied.rows[0]).unwrap();
    assert_eq!(&compact[..4096], &values[..4096]);
    assert_eq!(&compact[4096..], &values[8192..]);
    assert_eq!(
        supplied.receipts[0].sha256,
        text_encoder::digest_f32(&values, Some(&mask))
    );

    let d = fixture("fixture.json");
    let g = geometry(&d);
    let context = supplied.rows[0]
        .narrow(2, 0, g.text)
        .unwrap()
        .contiguous()
        .unwrap();
    let rounded = context
        .to_dtype(DType::BF16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    assert_ne!(
        loader::tensor_f32(&context).unwrap(),
        loader::tensor_f32(&rounded).unwrap()
    );
    let full = Transformer::load(&weights(&d["weights"], DType::F32), g, None).unwrap();
    let actual = full.project_context(&context).unwrap();
    assert_eq!(actual.dtype(), DType::F32);
    let actual = loader::tensor_f32(&actual).unwrap();
    let input = loader::tensor_f32(&context).unwrap();
    let projection: Vec<f32> =
        serde_json::from_value(d["weights"]["text_embedder.proj.weight"]["values"].clone())
            .unwrap();
    let bias: Vec<f32> =
        serde_json::from_value(d["weights"]["text_embedder.proj.bias"]["values"].clone()).unwrap();
    for (row, input) in input.chunks_exact(g.text).enumerate() {
        for output in 0..g.dim {
            let expected = input
                .iter()
                .zip(&projection[output * g.text..(output + 1) * g.text])
                .fold(0f32, |sum, (x, w)| sum + x * w)
                + bias[output];
            assert!((actual[row * g.dim + output] - expected).abs() < 1e-6);
        }
    }
    let wrong = loader::tensor_f32(&full.project_context(&rounded).unwrap()).unwrap();
    assert!(actual.iter().zip(&wrong).any(|(a, b)| (a - b).abs() > 1e-4));
    let x = tensor(&d["inputs"]["hidden_states"], DType::F32)
        .narrow(0, 0, 1)
        .unwrap();
    let output = full
        .forward(&x, 500.0, &context, &[1, 1], &[], &CancelFlag::default())
        .unwrap();
    let wrong = full
        .forward(&x, 500.0, &rounded, &[1, 1], &[], &CancelFlag::default())
        .unwrap();
    assert!(loader::tensor_f32(&output)
        .unwrap()
        .iter()
        .zip(loader::tensor_f32(&wrong).unwrap())
        .any(|(a, b)| (a - b).abs() > 1e-8));

    // Ordinary and unmerged DMD keep BF16 text projection weights; the consumer
    // still performs the author's BF16 conversion, even with FP32 time/head islands.
    let ordinary = weights(&d["weights"], DType::BF16);
    let dm = fixture("dmd_fixture.json");
    let mixed = transformer::install_replacements(
        &ordinary,
        &weights(&dm["replacement_tensors"], DType::F32),
    )
    .unwrap();
    for (name, w) in [("ordinary", ordinary), ("mixed", mixed)] {
        let model = Transformer::load(&w, g, None).unwrap();
        let output = model.project_context(&context).unwrap();
        assert_eq!(output.dtype(), DType::BF16, "{name}");
        assert_eq!(
            loader::tensor_f32(&output).unwrap(),
            loader::tensor_f32(&model.project_context(&rounded).unwrap()).unwrap(),
            "{name}"
        );
    }
}

#[test]
fn load_refuses_noncanonical_precision_and_missing_prepared_children() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("config.json");
    std::fs::write(&p, b"untrusted").unwrap();
    let mut spec = LoadSpec::new(WeightsSource::Dir(tmp.path().into()));
    spec.prepare_with_file_pins([]).unwrap();
    let e = loader::Asset::new(
        &spec,
        p.clone(),
        MossImageAssetRole::Transformer,
        SFT_REPO,
        SFT_REVISION,
        "transformer/config.json",
    )
    .err()
    .unwrap();
    assert!(e.to_string().contains("prepared source missing"));
    let mut spec = LoadSpec::new(WeightsSource::Dir(tmp.path().into()));
    spec.prepare_with_file_pins([PinnedWeightsFile::pin(&p).unwrap()])
        .unwrap();
    assert!(loader::Asset::new(
        &spec,
        p,
        MossImageAssetRole::Transformer,
        SFT_REPO,
        SFT_REVISION,
        "transformer/config.json"
    )
    .is_err());
    let mut spec = LoadSpec::new(WeightsSource::Dir(tmp.path().into()));
    spec.precision = Precision::Fp32;
    assert!(Sources::new(MossImageVariant::Sft, &spec).is_err());
}
#[test]
fn dmd_header_validation_checks_finite_values_coverage_dtype_and_mutation() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("tiny.safetensors");
    let save = |v: f32| {
        let m: HashMap<_, _> = [(
            "test".to_string(),
            Tensor::new(&[v, 2.0], &Device::Cpu).unwrap(),
        )]
        .into_iter()
        .collect();
        candle_gen::candle_core::safetensors::save(&m, &p).unwrap();
    };
    save(1.0);
    let mut a = loader::Asset {
        #[cfg(feature = "test-support")]
        fixture_digest: None,
        pin: PinnedWeightsFile::pin(&p).unwrap(),
        role: MossImageAssetRole::DmdTime,
        repo: DMD_REPO,
        revision: DMD_REVISION,
        relative: "tiny.safetensors".into(),
    };
    let expected = [("test".into(), vec![2])].into_iter().collect();
    loader::validate_fp32_bytes(&std::fs::read(&p).unwrap(), &expected).unwrap();
    assert!(loader::validate_fp32_bytes(
        &std::fs::read(&p).unwrap(),
        &[("extra".into(), vec![2])].into_iter().collect()
    )
    .is_err());
    save(f32::NAN);
    assert!(a.pin.verify_unchanged().is_err());
    a.pin = PinnedWeightsFile::pin(&p).unwrap();
    assert!(
        loader::validate_fp32_bytes(&std::fs::read(&p).unwrap(), &expected)
            .unwrap_err()
            .to_string()
            .contains("nonfinite")
    );
}
#[test]
fn architecture_facts_and_structural_cost_scale_batch_context_and_guidance() {
    let s = synthetic_sources();
    let request = GenerationRequest::default();
    let r = MossImageResolved::resolve(MossImageVariant::Sft, &request).unwrap();
    let resident = memory_strategy::estimate(&s, &r, &request, 1, false).unwrap();
    let batch = memory_strategy::estimate(&s, &r, &request, 3, false).unwrap();
    assert!(batch > resident);
    let mut short = r.clone();
    short.max_sequence_length = 1;
    assert!(memory_strategy::estimate(&s, &short, &request, 1, false).unwrap() < resident);
    let mut no_cfg = r.clone();
    no_cfg.guidance = 1.0;
    assert!(memory_strategy::estimate(&s, &no_cfg, &request, 1, false).unwrap() < resident);
    let c = memory_strategy::contract_sources(MossImageVariant::Sft, &s).unwrap();
    assert_eq!(
        c.architecture_facts.attention_heads,
        Some(s.config.num_heads as u32)
    );
    assert_eq!(
        c.architecture_facts.latent_channels,
        Some(s.config.latent_channels as u32)
    );
    assert!(c.calibration.is_none());
    assert!(c
        .strategies
        .iter()
        .filter(|s| s.strategy != MemoryStrategy::Resident)
        .all(|s| s.support == MemoryStrategySupport::Missing));
}
#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn renderer_off_on_off_never_reuses_modified_backbone() {
    let ordinary = fixture("fixture.json");
    let g = geometry(&ordinary);
    let mut state = State::default();
    let mode = MossImageAcceleration::Off;
    let make = |mode, dtype| {
        Ok(pipeline::Renderer {
            mode,
            transformer: Transformer::load(&weights(&ordinary["weights"], dtype), g, None).unwrap(),
            dataspace: None,
            adapter_reports: vec![],
            dmd_report: None,
        })
    };
    state
        .select_renderer(mode, || Ok(()), || make(mode, DType::BF16))
        .unwrap();
    let x = tensor(&ordinary["inputs"]["hidden_states"], DType::F32)
        .narrow(0, 0, 1)
        .unwrap();
    let ctx = tensor(&ordinary["inputs"]["encoder_hidden_states"], DType::F32)
        .narrow(0, 0, 1)
        .unwrap();
    let run = |s: &State| {
        s.renderer
            .as_ref()
            .unwrap()
            .transformer
            .forward(&x, 731.25, &ctx, &[1, 1, 1, 0], &[], &CancelFlag::default())
            .unwrap()
    };
    let first = loader::tensor_f32(&run(&state)).unwrap();
    let accelerated = MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32);
    state
        .select_renderer(accelerated, || Ok(()), || make(accelerated, DType::F32))
        .unwrap();
    assert_eq!(state.renderer.as_ref().unwrap().mode, accelerated);
    state
        .select_renderer(mode, || Ok(()), || make(mode, DType::BF16))
        .unwrap();
    assert_eq!(first, loader::tensor_f32(&run(&state)).unwrap());
}

#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn warm_admission_credits_only_actual_provider_weights() {
    let d = fixture("fixture.json");
    let generator = synthetic_generator();
    assert_eq!(generator.moss_image_owned_weight_bytes(), Some(0));
    let mode = MossImageAcceleration::Off;
    let model = Transformer::load(&weights(&d["weights"], DType::F32), geometry(&d), None).unwrap();
    let actual = model.resident_bytes();
    assert!(actual > 0);
    candle_gen::lock_recover(&generator.state).renderer = Some(pipeline::Renderer {
        mode,
        transformer: model,
        dataspace: None,
        adapter_reports: vec![],
        dmd_report: None,
    });
    let request = GenerationRequest::default();
    let peak = generator
        .estimated_peak_bytes(&request, MossImageAdvancedInputs::default())
        .unwrap();
    assert_eq!(generator.moss_image_owned_weight_bytes(), Some(actual));
    let budget = MemoryBudget {
        total_bytes: u64::MAX / 2,
        committed_bytes: actual + 123456,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    };
    assert_eq!(
        generator
            .estimated_incremental_peak_bytes(&request, MossImageAdvancedInputs::default(), budget)
            .unwrap(),
        peak - actual
    );
    let tiny = MemoryBudget {
        committed_bytes: 1,
        ..budget
    };
    assert_eq!(
        generator
            .estimated_incremental_peak_bytes(&request, MossImageAdvancedInputs::default(), tiny)
            .unwrap(),
        peak - 1
    );
}

#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn ordinary_sampler_executes_cfg_zero_one_slg_interval_skips_and_expanded_batch() {
    let d = fixture("fixture.json");
    let renderer = pipeline::Renderer {
        mode: MossImageAcceleration::Off,
        transformer: Transformer::load(&weights(&d["weights"], DType::F32), geometry(&d), None)
            .unwrap(),
        dataspace: None,
        adapter_reports: vec![],
        dmd_report: None,
    };
    let c = tensor(&d["inputs"]["encoder_hidden_states"], DType::F32);
    let rows = vec![
        c.narrow(0, 0, 1).unwrap().narrow(1, 0, 3).unwrap(),
        c.narrow(0, 1, 1)
            .unwrap()
            .index_select(&Tensor::new(&[0u32, 2], &Device::Cpu).unwrap(), 1)
            .unwrap(),
    ];
    let positive = text_encoder::Conditioning {
        rows: rows.clone(),
        receipts: vec![],
    };
    let negative = text_encoder::Conditioning {
        rows: rows.iter().map(|r| (r * -0.5).unwrap()).collect(),
        receipts: vec![],
    };
    let noise: Vec<f32> = (0..4 * 4 * 2 * 3)
        .map(|i| (i as f32 * 0.17).sin())
        .collect();
    for cfg in [0.0, 1.0, 3.0] {
        for slg in [false, true] {
            let req = GenerationRequest {
                width: 48,
                height: 32,
                count: 2,
                steps: Some(3),
                guidance: Some(cfg),
                prompt_batch: vec!["first".into(), "second".into()],
                moss_image: Some(MossImageOptions {
                    uncond_skip_layers: Some(vec![1]),
                    slg: slg.then_some(MossSkipLayerGuidance {
                        layers: vec![1],
                        scale: 2.0,
                        start: 0.5,
                        stop: 1.0,
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let resolved = MossImageResolved::resolve(MossImageVariant::Sft, &req).unwrap();
            let mut progress = Vec::new();
            let actual = pipeline::sample(
                &renderer,
                &req,
                &resolved,
                MossImageAdvancedInputs {
                    noise: Some(&noise),
                    output: MossImageOutputKind::Latents,
                    ..Default::default()
                },
                &positive,
                Some(&negative),
                &mut |p| progress.push(p),
            )
            .unwrap();
            let sigmas = moss_image_schedule(3, resolved.shift, resolved.training_shift).unwrap();
            let mut x = noise.clone();
            for step in 0..sigmas.len() - 1 {
                for image in 0..4 {
                    let row = image / 2;
                    let range = image * 24..(image + 1) * 24;
                    let latent =
                        Tensor::from_vec(x[range.clone()].to_vec(), (1, 4, 2, 3), &Device::Cpu)
                            .unwrap();
                    let pos = &positive.rows[row];
                    let neg = &negative.rows[row];
                    let active = resolved.skipping_active(step, sigmas.len() - 1);
                    let pred = |ctx: &Tensor, skip: &[u32]| {
                        loader::tensor_f32(
                            &renderer
                                .transformer
                                .forward(
                                    &latent,
                                    sigmas[step] * 1000.0,
                                    ctx,
                                    &vec![1; ctx.dim(1).unwrap()],
                                    skip,
                                    &req.cancel,
                                )
                                .unwrap(),
                        )
                        .unwrap()
                    };
                    let cond = pred(pos, &[]);
                    let uncond = (cfg != 1.0).then(|| pred(neg, if active { &[1] } else { &[] }));
                    let skipped = (slg && active).then(|| pred(pos, &[1]));
                    let velocity = moss_guidance(
                        &cond,
                        uncond.as_deref(),
                        skipped.as_deref(),
                        cfg,
                        if slg { 2.0 } else { 0.0 },
                    )
                    .unwrap();
                    for (i, v) in velocity.iter().enumerate() {
                        x[image * 24 + i] += v * (sigmas[step + 1] - sigmas[step]);
                    }
                }
            }
            let expected = serde_json::json!({"values":x,"shape":[4,4,2,3]});
            compare(
                &actual.latents,
                &expected,
                2e-6,
                &format!("pipeline cfg{cfg}/slg{slg}"),
            );
            assert_eq!(actual.samples.len(), 4);
            assert!(actual.samples.iter().all(|s| s.seed.is_none()));
            assert_eq!(
                actual.samples[2].noise_sha256,
                text_encoder::digest_f32(&noise[48..72], None)
            );
            assert_eq!(progress.len(), sigmas.len() - 1);
        }
    }
}
#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn sampler_seed_receipts_progress_cancel_and_warm_reentry_are_actual() {
    let d = fixture("fixture.json");
    let renderer = pipeline::Renderer {
        mode: MossImageAcceleration::Off,
        transformer: Transformer::load(&weights(&d["weights"], DType::F32), geometry(&d), None)
            .unwrap(),
        dataspace: None,
        adapter_reports: vec![],
        dmd_report: None,
    };
    let c = tensor(&d["inputs"]["encoder_hidden_states"], DType::F32)
        .narrow(0, 0, 1)
        .unwrap();
    let positive = text_encoder::Conditioning {
        rows: vec![c],
        receipts: vec![],
    };
    let req = GenerationRequest {
        width: 48,
        height: 32,
        count: 2,
        steps: Some(3),
        guidance: Some(1.0),
        seed: Some(77),
        ..Default::default()
    };
    let r = MossImageResolved::resolve(MossImageVariant::Sft, &req).unwrap();
    let run = |req: &GenerationRequest, progress: &mut dyn FnMut(Progress)| {
        pipeline::sample(
            &renderer,
            req,
            &r,
            MossImageAdvancedInputs::default(),
            &positive,
            None,
            progress,
        )
    };
    let first = run(&req, &mut |_| {}).unwrap();
    let second = run(&req, &mut |_| {}).unwrap();
    assert_eq!(
        loader::tensor_f32(&first.latents).unwrap(),
        loader::tensor_f32(&second.latents).unwrap()
    );
    assert_eq!(first.samples, second.samples);
    assert_eq!(first.samples[0].seed, Some(77));
    assert_eq!(first.samples[1].seed, Some(78));
    let canceled = req.clone();
    assert!(matches!(
        run(&canceled, &mut |_| canceled.cancel.cancel()),
        Err(candle_gen::CandleError::Canceled)
    ));
    let mut recovered = req.clone();
    recovered.cancel = CancelFlag::default();
    assert!(run(&recovered, &mut |_| {}).is_ok());
}

#[test]
fn selected_pin_authentication_ignores_unconsumed_optional_pin_mutation_sc26325() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("transformer/pipeline_moss_image.py");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let source=include_bytes!("../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_author/frozen_author_pipeline.py");
    std::fs::write(&path, source).unwrap();
    let unused = tmp.path().join("unused-full-fp32");
    std::fs::write(&unused, b"available unused").unwrap();
    let mut spec = LoadSpec::new(WeightsSource::Dir(tmp.path().into()));
    spec.prepare_with_file_pins([
        PinnedWeightsFile::pin(&path).unwrap(),
        PinnedWeightsFile::pin(&unused).unwrap(),
    ])
    .unwrap();
    std::fs::write(&unused, b"changed unused availability").unwrap();
    assert!(spec.validate_prepared_file_pins().is_err());
    let selected = loader::Asset::new(
        &spec,
        path,
        MossImageAssetRole::Transformer,
        SFT_REPO,
        SFT_REVISION,
        "transformer/pipeline_moss_image.py",
    )
    .unwrap();
    assert_eq!(selected.authenticated_bytes().unwrap(), source);
}

mod training_contract_tests;

#[test]
fn edit_failures_clear_both_reports_release_owned_renderer_and_allow_reentry() {
    let generator = synthetic_generator();
    let ordinary = fixture("fixture.json");
    let model = Transformer::load(
        &weights(&ordinary["weights"], DType::F32),
        geometry(&ordinary),
        None,
    )
    .unwrap();
    let report = MossImageEditExecutionReport {
        base: synthetic_report(),
        adapter_path: "synthetic-only".into(),
        edit_contract: EDIT_CONTRACT.into(),
        adapter_repository: EDIT_REPOSITORY.into(),
        adapter_revision: EDIT_REVISION.into(),
        adapter_sha256: EDIT_WEIGHTS_SHA256.into(),
        config_sha256: EDIT_CONFIG_SHA256.into(),
        applied_projections: 252,
        references: vec![],
        reference_max_edge: 512,
        use_kv_cache: true,
    };
    {
        let mut state = candle_gen::lock_recover(&generator.state);
        state.report = Some(synthetic_report());
        state.edit_report = Some(report.clone());
        state.renderer = Some(pipeline::Renderer {
            mode: MossImageAcceleration::Off,
            transformer: model,
            dataspace: None,
            adapter_reports: vec![],
            dmd_report: None,
        });
    }
    assert!(generator.moss_image_owned_weight_bytes().unwrap() > 0);
    assert!(generator
        .generate_moss_image_edit(
            &GenerationRequest::default(),
            MossImageEditInputs::default(),
            &mut |_| {}
        )
        .is_err());
    assert!(generator.moss_image_edit_execution_report().is_none());
    assert!(generator.moss_image_execution_report().is_none());
    assert_eq!(generator.moss_image_owned_weight_bytes(), Some(0));
    let references = [Image {
        width: 16,
        height: 16,
        pixels: vec![0; 16 * 16 * 3],
    }];
    let inputs = MossImageEditInputs {
        references: &references,
        ..Default::default()
    };
    let canceled = GenerationRequest::default();
    canceled.cancel.cancel();
    assert!(matches!(
        generator.generate_moss_image_edit(&canceled, inputs, &mut |_| {}),
        Err(gen_core::Error::Canceled)
    ));
    let request = GenerationRequest::default();
    let error = generator
        .generate_moss_image_edit(&request, inputs, &mut |_| {})
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("exact moss-image-edit package"),
        "fresh call must re-enter validation: {error}"
    );
    candle_gen::lock_recover(&generator.state).edit_report = Some(report);
    assert!(generator
        .generate(
            &GenerationRequest {
                width: 17,
                ..Default::default()
            },
            &mut |_| {}
        )
        .is_err());
    assert!(generator.moss_image_edit_execution_report().is_none());
    assert!(generator.validate(&GenerationRequest::default()).is_ok());
}

#[test]
fn edit_receipts_round_trip_ordered_mixed_aspect_and_duplicate_slots() {
    let mut base = synthetic_report().data().clone();
    base.resolved.uncond_skip_layers.clear();
    let custom_report = AdapterApplyReport {
        adapter_path: "trained-custom.safetensors".into(),
        applied: 1,
        skipped: vec![],
    };
    let mut report = MossImageEditExecutionReport {
        base: MossImageExecutionReport::new(base).unwrap(),
        adapter_path: "synthetic-only".into(),
        edit_contract: EDIT_CONTRACT.into(),
        adapter_repository: EDIT_REPOSITORY.into(),
        adapter_revision: EDIT_REVISION.into(),
        adapter_sha256: EDIT_WEIGHTS_SHA256.into(),
        config_sha256: EDIT_CONFIG_SHA256.into(),
        applied_projections: 252,
        references: vec![],
        reference_max_edge: 512,
        use_kv_cache: true,
    };
    let slot = |dimensions, digest: char| MossImageEditReferenceReceipt {
        original_dimensions: dimensions,
        encoded_dimensions: dimensions,
        pixels_sha256: digest.to_string().repeat(64),
        resized_pixels_sha256: "b".repeat(64),
        latent_sha256: "c".repeat(64),
    };
    let slots = [
        slot([32, 32], 'a'),
        slot([16, 48], 'b'),
        slot([48, 16], 'c'),
        slot([16, 48], 'b'),
    ];
    assert!(report.validate().is_err());
    for count in 1..=4 {
        report.references = slots[..count].to_vec();
        report.validate().unwrap();
        let roundtrip: MossImageEditExecutionReport =
            serde_json::from_value(serde_json::to_value(&report).unwrap()).unwrap();
        assert_eq!(roundtrip, report);
    }
    report.references.reverse();
    report.validate().unwrap();
    assert_eq!(report.references[0], slots[3]);
    let generator = synthetic_generator();
    // A valid sealed ordinary receipt remains empty even with custom provider reports.
    assert!(report.base.data().adapter_reports.is_empty());
    let mut invalid_base = report.base.data().clone();
    invalid_base.adapter_reports.push(custom_report.clone());
    assert!(MossImageExecutionReport::new(invalid_base).is_err());
    {
        let mut state = candle_gen::lock_recover(&generator.state);
        state.report = Some(report.base.clone());
        state.apply_reports = pipeline::application_reports(None, vec![custom_report.clone()]);
    }
    assert_eq!(
        generator.adapter_apply_reports(),
        vec![custom_report.clone()]
    );
    let dmd = AdapterApplyReport {
        adapter_path: "published-dmd.safetensors".into(),
        applied: 252,
        skipped: vec![],
    };
    assert_eq!(
        pipeline::application_reports(Some(&dmd), vec![custom_report.clone()]),
        vec![dmd, custom_report.clone()],
        "unmerged DMD and custom applications must appear once each"
    );
    {
        let mut state = candle_gen::lock_recover(&generator.state);
        state.report = None;
        state.edit_report = Some(report.clone());
        state.apply_reports = vec![
            custom_report.clone(),
            AdapterApplyReport {
                adapter_path: report.adapter_path.clone(),
                applied: report.applied_projections,
                skipped: vec![],
            },
        ];
    }
    assert_eq!(
        generator.adapter_apply_reports(),
        vec![
            custom_report,
            AdapterApplyReport {
                adapter_path: report.adapter_path.clone(),
                applied: report.applied_projections,
                skipped: vec![],
            }
        ],
        "edit application reports must retain the custom adapter stack"
    );
    // A failed subsequent entry cannot expose the previous completed application stack.
    assert!(generator
        .generate(
            &GenerationRequest {
                width: 17,
                ..Default::default()
            },
            &mut |_| {}
        )
        .is_err());
    assert!(generator.adapter_apply_reports().is_empty());
    report.references.push(slots[0].clone());
    assert!(report.validate().is_err());
    assert!(serde_json::from_value::<MossImageEditExecutionReport>(
        serde_json::to_value(report).unwrap()
    )
    .is_err());
}
