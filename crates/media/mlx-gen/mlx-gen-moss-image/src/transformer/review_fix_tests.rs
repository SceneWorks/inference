use super::author_fixture_tests::*;
use super::*;
use gen_core_moss_image::moss_image_training as contract;
use mlx_gen::gen_core::moss_image::MossImageVariant;
use mlx_gen::runtime::{AdapterKind, AdapterSpec};
fn tiny() -> Transformer {
    let f = fixture("ordinary");
    Transformer::load(&weights(&f["weights"], Dtype::Float32), geometry(&f)).unwrap()
}
fn file(
    temp: &tempfile::TempDir,
    values: std::collections::HashMap<String, Array>,
    meta: std::collections::BTreeMap<String, String>,
    kind: AdapterKind,
    scale: f32,
) -> AdapterSpec {
    let path = temp.path().join("adapter.safetensors");
    let meta: std::collections::HashMap<_, _> = meta.into_iter().collect();
    Array::save_safetensors(
        values.iter().map(|(k, v)| (k.as_str(), v)),
        Some(&meta),
        &path,
    )
    .unwrap();
    AdapterSpec::new(path, scale, kind)
}
fn full(shape: &[i32], value: f32) -> Array {
    Array::from_slice(
        &vec![value; shape.iter().map(|v| *v as usize).product()],
        shape,
    )
}
fn lokr_values() -> std::collections::HashMap<String, Array> {
    [
        ("layers.0.proj.q.lokr_w1".into(), full(&[2, 2], 0.5)),
        ("layers.0.proj.q.lokr_w2".into(), full(&[8, 8], 0.25)),
    ]
    .into_iter()
    .collect()
}
fn lokr_meta() -> std::collections::BTreeMap<String, String> {
    let mut m = contract::metadata(MossImageVariant::Base, contract::SAMPLER);
    m.insert("networkType".into(), "lokr".into());
    m.insert("rank".into(), "4".into());
    m.insert("alpha".into(), "6".into());
    m
}
#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn lokr_real_loader_rejects_crossed_axes_unused_factors_and_bad_scaling() {
    let _device = crate::cpu_test_device();
    for defect in [
        "axes",
        "orphan",
        "competing",
        "rank_zero",
        "rank_garbage",
        "alpha_garbage",
        "alpha_missing",
    ] {
        let mut values = lokr_values();
        let mut meta = lokr_meta();
        match defect {
            "axes" => {
                values.insert("layers.0.proj.q.lokr_w2".into(), full(&[4, 16], 0.25));
            }
            "orphan" => {
                values.insert("layers.0.proj.q.lokr_w2_a".into(), full(&[8, 1], 0.25));
            }
            "competing" => {
                values.insert("layers.0.proj.q.lokr_w2_a".into(), full(&[8, 1], 0.25));
                values.insert("layers.0.proj.q.lokr_w2_b".into(), full(&[1, 8], 0.25));
            }
            "rank_zero" => {
                meta.insert("rank".into(), "0".into());
            }
            "rank_garbage" => {
                meta.insert("rank".into(), "garbage".into());
            }
            "alpha_garbage" => {
                meta.insert("alpha".into(), "garbage".into());
            }
            "alpha_missing" => {
                meta.remove("alpha");
            }
            _ => unreachable!(),
        }
        let temp = tempfile::tempdir().unwrap();
        let spec = file(&temp, values, meta, AdapterKind::Lokr, 1.);
        assert!(
            crate::adapters::apply(&mut tiny(), MossImageVariant::Base, &[spec]).is_err(),
            "accepted {defect}"
        );
    }
}
#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn installed_adapter_validation_rejects_finite_sources_that_overflow() {
    let _device = crate::cpu_test_device();
    for defect in [
        "kron_product",
        "decomposed_factor",
        "lora_alpha_fold",
        "lora_strength",
    ] {
        let mut meta = lokr_meta();
        let mut values = lokr_values();
        let mut kind = AdapterKind::Lokr;
        let mut strength = 1.;
        match defect {
            "kron_product" => {
                values.insert("layers.0.proj.q.lokr_w1".into(), full(&[2, 2], 1e30));
                values.insert("layers.0.proj.q.lokr_w2".into(), full(&[8, 8], 1e30));
            }
            "decomposed_factor" => {
                values.remove("layers.0.proj.q.lokr_w2");
                values.insert("layers.0.proj.q.lokr_w2_a".into(), full(&[8, 1], 1e30));
                values.insert("layers.0.proj.q.lokr_w2_b".into(), full(&[1, 8], 1e30));
            }
            "lora_alpha_fold" | "lora_strength" => {
                meta.insert("networkType".into(), "lora".into());
                kind = AdapterKind::Lora;
                values = [
                    ("layers.0.proj.q.lora_A.weight".into(), full(&[2, 16], 1.)),
                    ("layers.0.proj.q.lora_B.weight".into(), full(&[16, 2], 1e30)),
                    (
                        "layers.0.proj.q.alpha".into(),
                        full(
                            &[1],
                            if defect == "lora_alpha_fold" {
                                1e30
                            } else {
                                2.
                            },
                        ),
                    ),
                ]
                .into_iter()
                .collect();
                if defect == "lora_strength" {
                    strength = 1e30;
                }
            }
            _ => unreachable!(),
        }
        assert!(
            values.values().all(|array| array
                .is_finite()
                .unwrap()
                .all(None)
                .unwrap()
                .item::<bool>()),
            "fixture must have finite source tensors"
        );
        let temp = tempfile::tempdir().unwrap();
        let spec = file(&temp, values, meta, kind, strength);
        assert!(
            crate::adapters::apply(&mut tiny(), MossImageVariant::Base, &[spec]).is_err(),
            "accepted derived overflow {defect}"
        );
    }
}
#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn all_projection_lokr_keeps_raw_factors_in_dense_and_checkpointed_backward() {
    let _device = crate::cpu_test_device();
    use mlx_gen::train::lora::{build_lokr_targets, TrainAdapter};
    let f = fixture("ordinary");
    let g = geometry(&f);
    let mut model = tiny();
    let paths = contract::target_paths(g.layers, &[]).unwrap();
    let (targets, params) = build_lokr_targets(&mut model, &paths, 2, 2, 19).unwrap();
    let adapter = TrainAdapter::Lokr { targets };
    let x = slice_axis(&array(&f["inputs"]["hidden_states"]), 0, 0, 1).unwrap();
    let context = contexts(
        &f["inputs"]["encoder_hidden_states"],
        &f["inputs"]["encoder_attention_mask"],
        g.text,
    );
    let noise = add(&x, Array::from_f32(0.7)).unwrap();
    let expected: usize = params.values().map(|a| a.nbytes()).sum();
    for checkpoint in [false, true] {
        let cfg = mlx_gen::TrainingConfig {
            rank: 2,
            alpha: 3.,
            network_type: mlx_gen::NetworkType::Lokr,
            gradient_checkpointing: checkpoint,
            ..Default::default()
        };
        let (_, grads) = crate::training::loss_grads(
            &mut model,
            &params,
            &adapter,
            &cfg,
            &x,
            &context[0],
            &noise,
            0.41,
            None,
            &CancelFlag::new(),
        )
        .unwrap();
        assert_eq!(
            model.raw_lokr_bytes(),
            expected,
            "retained reconstructed delta collection instead of raw trainable factors"
        );
        assert!(
            model
                .blocks
                .iter()
                .flat_map(|b| &b.projections)
                .all(|p| p.custom.adapters().is_empty()),
            "generic dense LoKr installation reached production graph"
        );
        assert!(
            grads
                .values()
                .map(|g| crate::pipeline::logical_f32_readback(g)
                    .unwrap()
                    .iter()
                    .map(|v| v * v)
                    .sum::<f32>())
                .sum::<f32>()
                > 0.
        );
    }
}
#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn routing_owner_tracks_widening_and_dmd_weight_replacement() {
    let _device = crate::cpu_test_device();
    let original = full(&[2, 2], 0.25).as_dtype(Dtype::Bfloat16).unwrap();
    let mut p = Projection {
        weight: original.clone(),
        bias: None,
        adapter: None,
        custom: AdaptableLinear::dense(original, None),
        lokr: Vec::new(),
        edit_adapter: false,
        edit_fused_bias: false,
    };
    p.widen().unwrap();
    assert_eq!(
        p.custom.dense_weight().unwrap().0.dtype(),
        Dtype::Float32,
        "routing retained original BF16 allocation"
    );
    p.replace_weight_bias(full(&[2, 2], 0.75), None);
    assert_eq!(
        crate::pipeline::logical_f32_readback(p.custom.dense_weight().unwrap().0).unwrap(),
        crate::pipeline::logical_f32_readback(&p.weight).unwrap(),
        "routing retained replaced time weight"
    );
}
