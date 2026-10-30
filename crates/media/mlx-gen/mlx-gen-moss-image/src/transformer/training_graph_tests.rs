use super::author_fixture_tests::*;
use super::*;
use gen_core_moss_image::moss_image_training as contract;
use mlx_gen::gen_core::moss_image::MossImageVariant;
use mlx_gen::train::lora::{build_lokr_targets, build_lora_targets, TrainAdapter};
use mlx_gen::{NetworkType, TrainingConfig};
fn read(a: &Array) -> Vec<f32> {
    crate::pipeline::logical_f32_readback(a).unwrap()
}
#[cfg(feature = "test-support")]
#[test]
#[ignore = "manual MOSS Metal model proof; CI lanes=moss-metal-fixture"]
fn moss_adamw_bias_correction_resume_and_fixed_batch_descent() {
    crate::test_support::assert_default_metal_profile_device();
    use crate::training::TrainOptimizer;
    use mlx_gen::train::lora::LoraParams;
    use std::collections::HashMap;
    use std::rc::Rc;
    let cfg = TrainingConfig {
        learning_rate: 0.001,
        weight_decay: 0.2,
        ..Default::default()
    };
    let initial = || {
        LoraParams::from([
            (Rc::from("first"), Array::from_slice(&[1f32], &[1, 1])),
            (Rc::from("second"), Array::from_slice(&[-2f32], &[1, 1])),
        ])
    };
    let mut params = initial();
    let mut opt =
        TrainOptimizer::from_config("adamw", cfg.learning_rate, cfg.weight_decay).unwrap();
    let mut reference = [1f64, -2.];
    let (mut first, mut second) = ([0f64; 2], [0f64; 2]);
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = directory.path().join("optimizer.safetensors");
    let mut resumed: Option<(TrainOptimizer, LoraParams)> = None;
    for update in 0..4 {
        // Tiny gradients expose epsilon placement; two names expose a clock
        // accidentally incremented per parameter. Later steps expose decay,
        // sign changes, LR scheduling and restoration of both moment slots.
        let gradient = if update == 0 {
            [1e-10f32, -2e-10]
        } else {
            [0.25, -0.4]
        };
        let multiplier = if update < 2 { 1. } else { 0.25 };
        let grads = LoraParams::from([
            (
                Rc::from("first"),
                Array::from_slice(&gradient[..1], &[1, 1]),
            ),
            (
                Rc::from("second"),
                Array::from_slice(&gradient[1..], &[1, 1]),
            ),
        ]);
        opt.set_lr_scaled(multiplier);
        opt.update(&mut params, &grads, &cfg, update, 1789).unwrap();
        for i in 0..2 {
            let t = f64::from(update + 1);
            let lr = f64::from(cfg.learning_rate * multiplier);
            first[i] = 0.9 * first[i] + 0.1 * f64::from(gradient[i]);
            second[i] = 0.999 * second[i] + 0.001 * f64::from(gradient[i]).powi(2);
            reference[i] = reference[i] * (1. - lr * f64::from(cfg.weight_decay))
                - lr * (first[i] / (1. - 0.9f64.powf(t)))
                    / ((second[i] / (1. - 0.999f64.powf(t))).sqrt() + 1e-8);
            let name = ["first", "second"][i];
            let actual = read(&params[name])[0];
            assert!(
                (f64::from(actual) - reference[i]).abs() < 3e-7,
                "step {update}, {name}: {actual} != host AdamW {}",
                reference[i]
            );
        }
        if let Some((loaded, restored)) = &mut resumed {
            loaded.set_lr_scaled(multiplier);
            loaded.update(restored, &grads, &cfg, update, 1789).unwrap();
            for key in params.keys() {
                assert_eq!(
                    read(&params[key]),
                    read(&restored[key]),
                    "resume factor {key}"
                );
            }
        }
        if update == 1 {
            opt.save_state(&checkpoint).unwrap();
            let mut loaded =
                TrainOptimizer::from_config("adamw", cfg.learning_rate, cfg.weight_decay).unwrap();
            loaded.load_state_checked(&checkpoint, &params, 2).unwrap();
            resumed = Some((loaded, params.clone()));
            let (arrays, metadata) =
                mlx_gen::weights::load_safetensors_on_cpu(&checkpoint).unwrap();
            mlx_rs::transforms::eval_pending_loads(arrays.values()).unwrap();
            for (policy, clock) in [
                (None, None),
                (Some("legacy"), Some("2")),
                (
                    metadata.get("moss_optimizer_policy").map(String::as_str),
                    Some("3"),
                ),
            ] {
                let mut bad = HashMap::new();
                if let Some(value) = policy {
                    bad.insert("moss_optimizer_policy".to_owned(), value.to_owned());
                }
                if let Some(value) = clock {
                    bad.insert("moss_optimizer_updates".to_owned(), value.to_owned());
                }
                let file = directory.path().join("invalid.safetensors");
                Array::save_safetensors(arrays.iter(), Some(&bad), &file).unwrap();
                let mut rejected =
                    TrainOptimizer::from_config("adamw", cfg.learning_rate, cfg.weight_decay)
                        .unwrap();
                assert!(rejected.load_state_checked(&file, &params, 2).is_err());
            }
        }
    }
    assert!(opt.update(&mut params, &initial(), &cfg, 0, 1789).is_err());
    let (loaded, _) = resumed.unwrap();
    let restored = directory.path().join("restored.safetensors");
    let final_state = directory.path().join("final.safetensors");
    loaded.save_state(&restored).unwrap();
    opt.save_state(&final_state).unwrap();
    let (a, a_meta) = mlx_gen::weights::load_safetensors_on_cpu(restored).unwrap();
    let (b, b_meta) = mlx_gen::weights::load_safetensors_on_cpu(final_state).unwrap();
    assert_eq!(a_meta, b_meta);
    assert_eq!(
        a.keys().collect::<std::collections::BTreeSet<_>>(),
        b.keys().collect()
    );
    for (key, value) in a {
        assert_eq!(read(&value), read(&b[&key]), "resume moment {key}");
    }

    // One fixed batch through the existing production BF16 graph. This
    // synthetic oracle does not replace the real-weight heldout criterion.
    let f = fixture("ordinary");
    let g = geometry(&f);
    let mut model = Transformer::load(&weights(&f["weights"], Dtype::Bfloat16), g.clone()).unwrap();
    let x = slice_axis(&array(&f["inputs"]["hidden_states"]), 0, 0, 1).unwrap();
    let context = contexts(
        &f["inputs"]["encoder_hidden_states"],
        &f["inputs"]["encoder_attention_mask"],
        g.text,
    );
    let noise = mlx_rs::ops::add(&x, Array::from_f32(0.37)).unwrap();
    let paths = contract::target_paths(g.layers, &[]).unwrap();
    let (targets, mut params) = build_lora_targets(&mut model, &paths, 4, 1789).unwrap();
    let adapter = TrainAdapter::Lora { targets };
    let cfg = TrainingConfig {
        rank: 4,
        alpha: 6.,
        learning_rate: 0.001,
        gradient_checkpointing: false,
        weight_decay: 0.,
        seed: 1789,
        ..Default::default()
    };
    let cancel = CancelFlag::new();
    let frozen = model
        .frozen_training_arrays()
        .iter()
        .map(read)
        .collect::<Vec<_>>();
    let (before, grads) = crate::training::loss_grads(
        &mut model,
        &params,
        &adapter,
        &cfg,
        &x,
        &context[0],
        &noise,
        0.375,
        None,
        &cancel,
    )
    .unwrap();
    let mut opt =
        TrainOptimizer::from_config("adamw", cfg.learning_rate, cfg.weight_decay).unwrap();
    opt.update(&mut params, &grads, &cfg, 0, cfg.seed).unwrap();
    let (after, _) = crate::training::loss_grads(
        &mut model,
        &params,
        &adapter,
        &cfg,
        &x,
        &context[0],
        &noise,
        0.375,
        None,
        &cancel,
    )
    .unwrap();
    assert!(after < before, "fixed-batch AdamW update did not descend");
    assert_eq!(
        frozen,
        model
            .frozen_training_arrays()
            .iter()
            .map(read)
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn cpu_native_backward_updates_only_factors_and_reload_changes_generation() {
    let _device = crate::cpu_test_device();
    let f = fixture("ordinary");
    let g = geometry(&f);
    let x = slice_axis(&array(&f["inputs"]["hidden_states"]), 0, 0, 1).unwrap();
    let contexts = contexts(
        &f["inputs"]["encoder_hidden_states"],
        &f["inputs"]["encoder_attention_mask"],
        g.text,
    );
    let noise = mlx_rs::ops::add(&x, Array::from_f32(0.37)).unwrap();
    let cancel = CancelFlag::new();
    for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
        for network in [NetworkType::Lora, NetworkType::Lokr] {
            let mut model =
                Transformer::load(&weights(&f["weights"], Dtype::Float32), g.clone()).unwrap();
            let frozen: Vec<_> = model.owned_arrays().iter().map(|a| read(a)).collect();
            let paths =
                contract::target_paths(g.layers, &["q".into(), "ffn_up_gate".into()]).unwrap();
            assert_eq!(paths.len(), g.layers * 2);
            assert!(model.adaptable_mut(&["text_embedder", "proj"]).is_none());
            let (adapter, mut params) = match network {
                NetworkType::Lora => {
                    let (t, p) = build_lora_targets(&mut model, &paths, 2, 17).unwrap();
                    (TrainAdapter::Lora { targets: t }, p)
                }
                NetworkType::Lokr => {
                    let (t, p) = build_lokr_targets(&mut model, &paths, 2, 2, 17).unwrap();
                    (TrainAdapter::Lokr { targets: t }, p)
                }
            };
            let cfg = TrainingConfig {
                rank: 2,
                alpha: 3.,
                network_type: network,
                decompose_factor: 2,
                ..Default::default()
            };
            let baseline = model
                .forward(
                    &x,
                    &[370.],
                    std::slice::from_ref(&contexts[0]),
                    &[],
                    &cancel,
                )
                .unwrap();
            model
                .install_training_adapter(&adapter, &params, 3., 2.)
                .unwrap();
            assert_eq!(
                read(&baseline),
                read(
                    &model
                        .forward(
                            &x,
                            &[370.],
                            std::slice::from_ref(&contexts[0]),
                            &[],
                            &cancel
                        )
                        .unwrap()
                ),
                "initial adapter must be identity"
            );
            TRAINING_BLOCK_FORWARDS.store(0, std::sync::atomic::Ordering::SeqCst);
            let (loss, grads) = crate::training::loss_grads(
                &mut model,
                &params,
                &adapter,
                &cfg,
                &x,
                &contexts[0],
                &noise,
                0.37,
                None,
                &cancel,
            )
            .unwrap();
            let dense_calls = TRAINING_BLOCK_FORWARDS.load(std::sync::atomic::Ordering::SeqCst);
            assert!(loss.is_finite());
            let norm: f32 = grads.values().flat_map(read).map(|v| v * v).sum();
            assert!(
                norm.is_finite() && norm > 1e-12,
                "backward detached adapter leaves: {norm}"
            );
            // A nonzero gradient selected from the actual adapter map, independently checked
            // against symmetric differences of the production loss (not a toy residual).
            let (key, gradient) = grads
                .iter()
                .max_by(|(_, a), (_, b)| {
                    let max = |a: &Array| read(a).into_iter().map(f32::abs).fold(0., f32::max);
                    max(a).total_cmp(&max(b))
                })
                .unwrap();
            let gv = read(gradient);
            let index = gv
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                .unwrap()
                .0;
            let epsilon = 0.01;
            let mut evaluate = |sign: f32| {
                let mut p = params.clone();
                let mut v = read(&p[key]);
                v[index] += sign * epsilon;
                p.insert(key.clone(), Array::from_slice(&v, p[key].shape()));
                crate::training::loss_grads(
                    &mut model,
                    &p,
                    &adapter,
                    &cfg,
                    &x,
                    &contexts[0],
                    &noise,
                    0.37,
                    None,
                    &cancel,
                )
                .unwrap()
                .0
            };
            let fd = (evaluate(1.) - evaluate(-1.)) / (2. * epsilon);
            assert!(
                (fd - gv[index]).abs() < 1e-6 + gv[index].abs() * 0.02,
                "finite difference {fd} vs backward {}",
                gv[index]
            );
            let cfg_checkpoint = TrainingConfig {
                gradient_checkpointing: true,
                ..cfg.clone()
            };
            TRAINING_BLOCK_FORWARDS.store(0, std::sync::atomic::Ordering::SeqCst);
            let (recomputed, rg) = crate::training::loss_grads(
                &mut model,
                &params,
                &adapter,
                &cfg_checkpoint,
                &x,
                &contexts[0],
                &noise,
                0.37,
                None,
                &cancel,
            )
            .unwrap();
            assert!(
                TRAINING_BLOCK_FORWARDS.load(std::sync::atomic::Ordering::SeqCst) > dense_calls,
                "gradient checkpoint toggle did not recompute blocks"
            );
            assert!((loss - recomputed).abs() < 1e-6);
            for (k, v) in &grads {
                assert!(read(v)
                    .iter()
                    .zip(read(&rg[k]))
                    .all(|(a, b)| (a - b).abs() < 1e-5));
            }
            let mut optimizer = mlx_gen::TrainOptimizer::from_config("adamw", 0.01, 0.0).unwrap();
            for update in 0..3 {
                let (_, grad) = crate::training::loss_grads(
                    &mut model,
                    &params,
                    &adapter,
                    &cfg,
                    &x,
                    &contexts[0],
                    &noise,
                    0.37,
                    None,
                    &cancel,
                )
                .unwrap();
                mlx_gen::train::lora::adapter_optimizer_update(
                    &mut optimizer,
                    &mut params,
                    &grad,
                    &cfg,
                    update,
                    17,
                )
                .unwrap();
            }
            assert_eq!(
                frozen,
                model
                    .owned_arrays()
                    .iter()
                    .map(|a| read(a))
                    .collect::<Vec<_>>(),
                "base/conditioning weights mutated"
            );
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("adapter.safetensors");
            let meta = contract::metadata(variant, contract::SAMPLER);
            let extra: Vec<_> = meta.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
            adapter
                .save_with_meta(&params, 3., 2., 2, "", &extra, &path)
                .unwrap();
            model
                .install_training_adapter(&adapter, &params, 3., 2.)
                .unwrap();
            let trained = model
                .forward(
                    &x,
                    &[370.],
                    std::slice::from_ref(&contexts[0]),
                    &[],
                    &cancel,
                )
                .unwrap();
            let mut reloaded =
                Transformer::load(&weights(&f["weights"], Dtype::Float32), g.clone()).unwrap();
            let kind = match network {
                NetworkType::Lora => mlx_gen::runtime::AdapterKind::Lora,
                NetworkType::Lokr => mlx_gen::runtime::AdapterKind::Lokr,
            };
            let spec = mlx_gen::runtime::AdapterSpec::new(path, 1., kind);
            let reports =
                crate::adapters::apply(&mut reloaded, variant, std::slice::from_ref(&spec))
                    .unwrap();
            assert_eq!(reports[0].applied, paths.len());
            let got = reloaded
                .forward(
                    &x,
                    &[370.],
                    std::slice::from_ref(&contexts[0]),
                    &[],
                    &cancel,
                )
                .unwrap();
            assert!(
                read(&got)
                    .iter()
                    .zip(read(&trained))
                    .all(|(a, b)| (a - b).abs() < 2e-5),
                "trained/reloaded projection mismatch"
            );
            assert!(
                read(&got)
                    .iter()
                    .zip(read(&baseline))
                    .any(|(a, b)| (a - b).abs() > 1e-5),
                "adapter did not affect generation {variant:?} {network:?}"
            );
            let mut disabled =
                Transformer::load(&weights(&f["weights"], Dtype::Float32), g.clone()).unwrap();
            let mut zero = spec.clone();
            zero.scale = 0.;
            crate::adapters::apply(&mut disabled, variant, &[zero]).unwrap();
            let disabled_output = disabled
                .forward(
                    &x,
                    &[370.],
                    std::slice::from_ref(&contexts[0]),
                    &[],
                    &cancel,
                )
                .unwrap();
            assert_eq!(
                read(&disabled_output),
                read(&baseline),
                "scale zero changed the backbone"
            );
            let (tensors, metadata) = Array::load_safetensors_with_metadata(&spec.path).unwrap();
            let first = tensors.keys().next().unwrap().clone();
            for defect in ["nan", "extra", "bf16", "missing", "shape"] {
                let mut corrupt = tensors.clone();
                match defect {
                    "nan" => {
                        corrupt.insert(
                            first.clone(),
                            multiply(&corrupt[&first], Array::from_f32(f32::NAN)).unwrap(),
                        );
                    }
                    "extra" => {
                        corrupt.insert("layers.0.proj.q.unrecognized".into(), Array::from_f32(1.));
                    }
                    "bf16" => {
                        corrupt.insert(
                            first.clone(),
                            corrupt[&first].as_dtype(Dtype::Bfloat16).unwrap(),
                        );
                    }
                    "shape" => {
                        let key = corrupt
                            .keys()
                            .find(|k| k.ends_with("lora_A.weight") || k.ends_with("lokr_w1"))
                            .expect("required factor")
                            .clone();
                        corrupt.insert(key, Array::zeros::<f32>(&[1, 1]).unwrap());
                    }
                    "missing" => {
                        let key = corrupt
                            .keys()
                            .find(|k| {
                                k.ends_with("lora_B.weight")
                                    || k.ends_with("lokr_w2_b")
                                    || k.ends_with("lokr_w2")
                            })
                            .expect("required factor")
                            .clone();
                        corrupt.remove(&key);
                    }
                    _ => unreachable!(),
                }
                let bad_path = temp.path().join(format!("{defect}.safetensors"));
                Array::save_safetensors(
                    corrupt.iter().map(|(k, v)| (k.as_str(), v)),
                    Some(&metadata),
                    &bad_path,
                )
                .unwrap();
                let mut bad = spec.clone();
                bad.path = bad_path;
                let mut fresh =
                    Transformer::load(&weights(&f["weights"], Dtype::Float32), g.clone()).unwrap();
                assert!(
                    crate::adapters::apply(&mut fresh, variant, &[bad]).is_err(),
                    "accepted malformed {defect} {network:?} adapter"
                );
            }
            let leased = temp.path().join("leased.safetensors");
            Array::save_safetensors(
                tensors.iter().map(|(k, v)| (k.as_str(), v)),
                Some(&metadata),
                &leased,
            )
            .unwrap();
            let pin = mlx_gen::gen_core::PinnedWeightsFile::pin(&leased).unwrap();
            let mut changed = tensors.clone();
            changed.insert(
                first.clone(),
                multiply(&changed[&first], Array::from_f32(2.)).unwrap(),
            );
            Array::save_safetensors(
                changed.iter().map(|(k, v)| (k.as_str(), v)),
                Some(&metadata),
                &leased,
            )
            .unwrap();
            let mut lease_spec = spec.clone();
            lease_spec.path = leased;
            let mut fresh =
                Transformer::load(&weights(&f["weights"], Dtype::Float32), g.clone()).unwrap();
            assert!(
                crate::adapters::apply_pinned(&mut fresh, variant, &[lease_spec], &[pin]).is_err(),
                "changed adapter bypassed its exact source token"
            );
            // Base and SFT share one architecture, so the sibling checkpoint accepts this
            // adapter too (2026-10-29 product decision; formerly an exact-parent refusal).
            let sibling = if variant == MossImageVariant::Base {
                MossImageVariant::Sft
            } else {
                MossImageVariant::Base
            };
            let mut cross =
                Transformer::load(&weights(&f["weights"], Dtype::Float32), g.clone()).unwrap();
            let reports = crate::adapters::apply(&mut cross, sibling, &[spec]).unwrap();
            assert!(reports[0].applied > 0 && reports[0].skipped.is_empty());
        }
    }
}
