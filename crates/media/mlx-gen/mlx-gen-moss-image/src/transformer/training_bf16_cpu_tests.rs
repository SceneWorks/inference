use super::author_fixture_tests::*;
use super::*;
use gen_core_moss_image::moss_image_training as contract;
use mlx_gen::train::lora::{build_lokr_targets, build_lora_targets, TrainAdapter};
use mlx_gen::{gen_core::moss_image::MossImageVariant, NetworkType, TrainingConfig};
#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn production_bf16_backbone_keeps_fp32_adapter_gradients_and_portable_reload() {
    let _device = crate::cpu_test_device();
    let f = fixture("ordinary");
    let g = geometry(&f);
    let x = slice_axis(&array(&f["inputs"]["hidden_states"]), 0, 0, 1).unwrap();
    let context = contexts(
        &f["inputs"]["encoder_hidden_states"],
        &f["inputs"]["encoder_attention_mask"],
        g.text,
    );
    for targets in [
        vec!["v".into(), "o".into(), "ffn_down".into()],
        vec!["o".into()],
    ] {
        for network in [NetworkType::Lora, NetworkType::Lokr] {
            let mut model =
                Transformer::load(&weights(&f["weights"], Dtype::Bfloat16), g.clone()).unwrap();
            let frozen_values = crate::pipeline::logical_f32_readback(&model.video.weight).unwrap();
            model.cast_training_weights(Dtype::Float32).unwrap();
            assert!(model
                .owned_arrays()
                .iter()
                .all(|a| a.dtype() == Dtype::Float32));
            assert_eq!(
                crate::pipeline::logical_f32_readback(&model.video.weight).unwrap(),
                frozen_values
            );
            model.cast_training_weights(Dtype::Bfloat16).unwrap();
            assert!(model
                .owned_arrays()
                .iter()
                .all(|a| a.dtype() == Dtype::Bfloat16));
            let paths = contract::target_paths(g.layers, &targets).unwrap();
            let (adapter, mut params) = match network {
                NetworkType::Lora => {
                    let (t, p) = build_lora_targets(&mut model, &paths, 2, 71).unwrap();
                    (TrainAdapter::Lora { targets: t }, p)
                }
                NetworkType::Lokr => {
                    let (t, p) = build_lokr_targets(&mut model, &paths, 2, 2, 71).unwrap();
                    (TrainAdapter::Lokr { targets: t }, p)
                }
            };
            let cfg = TrainingConfig {
                rank: 2,
                alpha: 3.,
                network_type: network,
                ..Default::default()
            };
            let noise = add(&x, Array::from_f32(0.7)).unwrap();
            let (_, grad) = crate::training::loss_grads(
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
            assert!(grad.values().all(|g| g.dtype() == Dtype::Float32));
            assert!(
                grad.values()
                    .map(|a| crate::pipeline::logical_f32_readback(a)
                        .unwrap()
                        .iter()
                        .map(|v| v * v)
                        .sum::<f32>())
                    .sum::<f32>()
                    > 1e-12
            );
            let mut optimizer = mlx_gen::TrainOptimizer::from_config("adamw", 0.01, 0.).unwrap();
            mlx_gen::train::lora::adapter_optimizer_update(
                &mut optimizer,
                &mut params,
                &grad,
                &cfg,
                0,
                71,
            )
            .unwrap();
            model
                .install_training_adapter(&adapter, &params, 3., 2.)
                .unwrap();
            model.training = true;
            let before = model
                .forward(
                    &x,
                    &[410.],
                    std::slice::from_ref(&context[0]),
                    &[],
                    &CancelFlag::new(),
                )
                .unwrap();
            let temp = tempfile::tempdir().unwrap();
            let file = temp.path().join("adapter.safetensors");
            let metadata = contract::metadata(MossImageVariant::Base, contract::SAMPLER);
            let extra: Vec<_> = metadata
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            adapter
                .save_with_meta(&params, 3., 2., 2, "", &extra, &file)
                .unwrap();
            let kind = match network {
                NetworkType::Lora => mlx_gen::runtime::AdapterKind::Lora,
                NetworkType::Lokr => mlx_gen::runtime::AdapterKind::Lokr,
            };
            let mut reloaded =
                Transformer::load(&weights(&f["weights"], Dtype::Bfloat16), g.clone()).unwrap();
            crate::adapters::apply(
                &mut reloaded,
                MossImageVariant::Base,
                &[mlx_gen::runtime::AdapterSpec::new(file, 1., kind)],
            )
            .unwrap();
            let after = reloaded
                .forward(
                    &x,
                    &[410.],
                    std::slice::from_ref(&context[0]),
                    &[],
                    &CancelFlag::new(),
                )
                .unwrap();
            assert_eq!(
                crate::pipeline::logical_f32_readback(&before).unwrap(),
                crate::pipeline::logical_f32_readback(&after).unwrap(),
                "BF16 selected-target training/reload diverged: {:?}",
                targets
            );
            assert!(reloaded.custom_owned_bytes > 0);
            assert_eq!(
                reloaded.resident_bytes(),
                reloaded.owned_weight_bytes + reloaded.custom_owned_bytes
            );
        }
    }
}
