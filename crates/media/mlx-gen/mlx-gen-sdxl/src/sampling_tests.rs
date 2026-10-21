//! Source-prepared tests for the actual resolver/factories. Execution and individually run
//! failing mutations are HELD in the current sc-26270 phase until explicit CPU admission.
use super::*;
use mlx_gen::gen_core::sampling::{cfgpp_sampler_for, CpuLatentOps};
use serde_json::json;
use std::collections::BTreeSet;

fn alpha() -> AlphaSchedule {
    AlphaSchedule::scaled_linear(1000, 0.00085, 0.012)
}
fn request(sampler: &str) -> GenerationRequest {
    GenerationRequest {
        sampler: Some(sampler.into()),
        steps: Some(3),
        ..Default::default()
    }
}

#[test]
fn metadata_recipe_pricing_regime_matches_live_controls_and_rejects_conflicting_context() {
    use crate::ios_pricing::{ios_pricing_with_recipe, Regime};
    use mlx_gen::gen_core::{
        GenerationMemory, LoadSpec, MemoryGeometry, MemoryMode, MemoryNumericTier, MemorySelection,
        MemoryStrategy, WeightsSource,
    };
    use mlx_gen::ios_pricing::IosPricingRequest;

    // No file is opened by regime resolution. In particular a refusal must precede header IO.
    let spec = LoadSpec::new(WeightsSource::Dir("unused-recipe-headers".into()));
    let selection = MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: Default::default(),
        tier: MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        },
    };
    for recipe in representative_recipes() {
        let mut generation = recipe.to_request();
        generation.width = 64;
        generation.height = 80;
        generation.count = 2;
        let execution = GenerationMemory {
            cfg_batching: Some(recipe.cfg_batching),
            ..Default::default()
        };
        let mut req = IosPricingRequest {
            provider: "sdxl",
            spec: &spec,
            mode: &MemoryMode::TextToImage,
            geometry: MemoryGeometry {
                width: 64,
                height: 80,
                batch: 2,
                frames: 1,
                reference_count: 0,
            },
            selection: &selection,
            execution: &execution,
            generation: Some(&generation),
            transform: None,
            depth: None,
            token_window: 154,
        };
        let live = Regime::from_request(&req).unwrap();
        req.geometry.reference_count = 1;
        assert!(
            matches!(Regime::from_request(&req),Err(error) if error.to_string().contains("reference counts differ"))
        );
        req.geometry.reference_count = 0;
        assert!(
            ios_pricing_with_recipe(&req, &recipe, SamplingWorkFacts::default())
                .unwrap_err()
                .to_string()
                .contains("payload-free")
        );
        req.generation = None;
        assert!(Regime::from_request(&req).is_err());
        let captured = Regime::from_recipe(&req, &recipe, SamplingWorkFacts::default()).unwrap();
        assert_eq!(captured.sampling.recipe(), live.sampling.recipe());
        assert_eq!(captured.model_rows(), live.model_rows());
        assert_eq!(captured.count, live.count);
        assert_eq!(captured.context_tokens, live.context_tokens);
        assert_eq!(captured.has_init, live.has_init);
        req.geometry.reference_count = 1;
        assert!(
            ios_pricing_with_recipe(&req, &recipe, SamplingWorkFacts::default())
                .unwrap_err()
                .to_string()
                .contains("reference counts differ")
        );
    }
    let recipe = SamplingRecipe::from_request(&request("euler_ancestral")).unwrap();
    let execution = GenerationMemory::default();
    for (mode, work, init) in [
        (
            MemoryMode::ImageToImage,
            SamplingWorkFacts {
                reference_count: 1,
                ..Default::default()
            },
            true,
        ),
        (
            MemoryMode::Edit,
            SamplingWorkFacts {
                reference_count: 1,
                mask_count: 1,
                ..Default::default()
            },
            true,
        ),
        (
            MemoryMode::Other("control".into()),
            SamplingWorkFacts {
                reference_count: 1,
                control_count: 1,
                ..Default::default()
            },
            true,
        ),
        (
            MemoryMode::Other("ip_adapter".into()),
            SamplingWorkFacts {
                reference_count: 1,
                ..Default::default()
            },
            false,
        ),
    ] {
        // These are actual configured roles, not a mode label manufacturing an unloaded overlay.
        let mut configured = spec.clone();
        match &mode {
            MemoryMode::Other(name) if name == "control" => {
                configured.control = Some(WeightsSource::Dir("unused-control-headers".into()))
            }
            MemoryMode::Other(name) if name == "ip_adapter" => {
                configured.ip_adapter = Some(WeightsSource::Dir("unused-ip-headers".into()))
            }
            _ => {}
        }
        let req = IosPricingRequest {
            provider: "sdxl",
            spec: &configured,
            mode: &mode,
            geometry: MemoryGeometry {
                width: 64,
                height: 80,
                batch: 2,
                frames: 1,
                reference_count: 1,
            },
            selection: &selection,
            execution: &execution,
            generation: None,
            transform: None,
            depth: None,
            token_window: 77,
        };
        assert_eq!(
            Regime::from_recipe(&req, &recipe, work).unwrap().has_init,
            init
        );
    }
}

#[test]
fn bounded_acceleration_families_match_actual_policies_at_repeat_boundary() {
    use std::collections::BTreeSet;
    let cfg = crate::config::DiffusionConfig::sdxl_base();
    let alpha = AlphaSchedule::scaled_linear(cfg.num_train_steps, cfg.beta_start, cfg.beta_end);
    for sampler in ["lcm", "lightning", "hyper"] {
        for steps in [1u32, 2, 49, 50, 51, 52] {
            let mut req = request(sampler);
            req.steps = Some(steps);
            let actual = ResolvedSampling::resolve(&req).unwrap();
            // Small actual runtime policy vector is the oracle. The metadata call itself never
            // constructs this requested-N vector, including for MAX below.
            let coefficients = actual.acceleration_coefficients(&alpha).unwrap();
            let n = coefficients.len();
            let expected: BTreeSet<_> = coefficients
                .into_iter()
                .enumerate()
                .map(|(step, c)| AccelerationFamily {
                    first_step: step == 0,
                    final_step: step + 1 == n,
                    input_scale_alias: c.c_in == 1.0,
                    draws_noise: c.a_noise != 0.0,
                    additive_update: c.a_x == 1.0 && c.a_noise == 0.0,
                })
                .collect();
            let bounded: BTreeSet<_> = actual
                .bounded_acceleration_families(&alpha)
                .unwrap()
                .into_iter()
                .collect();
            assert_eq!(
                bounded, expected,
                "actual {sampler} graph families steps={steps}"
            );
        }
    }
}

#[test]
fn bounded_hyper_repeats_and_max_metadata_preserve_actual_step_domain() {
    let cfg = crate::config::DiffusionConfig::sdxl_base();
    let alpha = AlphaSchedule::scaled_linear(cfg.num_train_steps, cfg.beta_start, cfg.beta_end);
    let families = |steps| {
        let mut req = request("hyper");
        req.steps = Some(steps);
        ResolvedSampling::resolve(&req)
            .unwrap()
            .bounded_acceleration_families(&alpha)
            .unwrap()
    };
    assert!(
        families(50).iter().all(|f| !f.additive_update),
        "non-repeating Hyper update must remain generic"
    );
    assert!(
        families(51)
            .iter()
            .any(|f| f.first_step && f.additive_update && !f.draws_noise),
        "Hyper above50 must include the actual first repeated-index unity branch"
    );
    for sampler in ["lcm", "lightning", "hyper"] {
        let mut req = request(sampler);
        req.steps = Some(u32::MAX);
        let actual = ResolvedSampling::resolve(&req).unwrap();
        let bounded = actual.bounded_acceleration_families(&alpha);
        assert!(matches!(&bounded,Ok(fs) if fs.iter().any(|f|f.first_step&&!f.final_step)&&fs.iter().any(|f|!f.first_step&&f.final_step)),"MAX {sampler} metadata must resolve first/terminal families without a requested schedule");
        assert!(
            matches!(&bounded,Ok(fs) if fs.iter().all(|f|f.input_scale_alias==(sampler!="lightning"))),
            "MAX {sampler} model-input alias must come from actual factory coefficients"
        );
    }
}

#[test]
fn metadata_roles_follow_actual_configured_ip_and_control_factory() {
    use crate::ios_pricing::{Regime, Route};
    use mlx_gen::gen_core::{
        GenerationMemory, LoadSpec, MemoryGeometry, MemoryMode, MemoryNumericTier, MemorySelection,
        MemoryStrategy, WeightsSource,
    };
    use mlx_gen::ios_pricing::IosPricingRequest;
    let recipe = SamplingRecipe::from_request(&request("euler_ancestral")).unwrap();
    let resolve = |spec: &LoadSpec, mode: MemoryMode, work: SamplingWorkFacts| {
        let selection = MemorySelection {
            strategy: MemoryStrategy::Resident,
            parameters: Default::default(),
            tier: MemoryNumericTier {
                precision: spec.precision,
                quant: spec.quantize,
                component_precision_floors: &[],
            },
        };
        let execution = GenerationMemory::default();
        Regime::from_recipe(
            &IosPricingRequest {
                provider: "sdxl",
                spec,
                mode: &mode,
                geometry: MemoryGeometry {
                    width: 64,
                    height: 80,
                    batch: 1,
                    frames: 1,
                    reference_count: work.reference_count as u32,
                },
                selection: &selection,
                execution: &execution,
                generation: None,
                transform: None,
                depth: None,
                token_window: 77,
            },
            &recipe,
            work,
        )
    };
    let mut configured = LoadSpec::new(WeightsSource::Dir("unused-actual-role-headers".into()));
    configured.ip_adapter = Some(WeightsSource::Dir("unused-configured-ip".into()));
    let reference = SamplingWorkFacts {
        reference_count: 1,
        ..Default::default()
    };
    assert_eq!(
        resolve(&configured, MemoryMode::Other("control".into()), reference)
            .unwrap()
            .route,
        Route::IpAdapter
    );
    configured.control = Some(WeightsSource::Dir("unused-primary-control".into()));
    let controlled = SamplingWorkFacts {
        reference_count: 1,
        control_count: 1,
        ..Default::default()
    };
    let both = resolve(&configured, MemoryMode::Other("control".into()), controlled).unwrap();
    assert_eq!((both.route, both.has_init), (Route::Control, true));
    configured
        .extra_controls
        .push(WeightsSource::Dir("unused-extra-control".into()));
    assert!(
        matches!(resolve(&configured,MemoryMode::Other("control".into()),controlled),Err(error) if error.to_string().contains("loaded branches"))
    );
    configured.control = None;
    assert_eq!(
        resolve(&configured, MemoryMode::Other("control".into()), controlled)
            .unwrap()
            .route,
        Route::Control
    );
    assert_eq!(
        resolve(&configured, MemoryMode::ImageToImage, reference)
            .unwrap()
            .route,
        Route::IpAdapter
    );
    configured.ip_adapter = None;
    assert_eq!(
        resolve(
            &configured,
            MemoryMode::Other("ip_adapter".into()),
            reference
        )
        .unwrap()
        .route,
        Route::Reference
    );
    assert_eq!(
        resolve(&configured, MemoryMode::TextToImage, controlled)
            .unwrap()
            .route,
        Route::Control
    );
}

#[test]
fn metadata_recipe_resolution_shares_actual_control_factory_and_typed_role_refusals() {
    for recipe in representative_recipes() {
        let actual = ResolvedSampling::resolve(&recipe.to_request()).unwrap();
        let metadata =
            ResolvedSampling::resolve_recipe(&recipe, SamplingWorkFacts::default()).unwrap();
        assert_eq!(metadata.route(), actual.route());
        assert_eq!(metadata.cfg_batching(), actual.cfg_batching());
        assert_eq!(metadata.recipe(), &recipe);
    }
    let ancestral = SamplingRecipe::from_request(&request("euler_ancestral")).unwrap();
    let inpaint = SamplingWorkFacts {
        reference_count: 1,
        mask_count: 1,
        control_count: 0,
        reference_image: None,
    };
    assert!(ResolvedSampling::resolve_recipe(&ancestral, inpaint).is_ok());
    assert!(ResolvedSampling::resolve_recipe(
        &ancestral,
        SamplingWorkFacts {
            reference_count: 2,
            ..Default::default()
        }
    )
    .is_err());
    assert!(ResolvedSampling::resolve_recipe(
        &ancestral,
        SamplingWorkFacts {
            mask_count: 1,
            ..Default::default()
        }
    )
    .is_err());
    let curated = SamplingRecipe::from_request(&request("heun")).unwrap();
    assert!(ResolvedSampling::resolve_recipe(&curated, inpaint).is_err());
    let accel = SamplingRecipe::from_request(&request("lcm")).unwrap();
    assert!(ResolvedSampling::resolve_recipe(
        &accel,
        SamplingWorkFacts {
            reference_count: 1,
            ..Default::default()
        }
    )
    .is_err());
    let mut value = ancestral.to_value();
    value["steps"] = json!(u32::MAX);
    value["strength"] = json!(0.2);
    value["reference_strength"] = json!(0.75);
    let captured = SamplingRecipe::from_value(value).unwrap();
    let metadata = ResolvedSampling::resolve_recipe(
        &captured,
        SamplingWorkFacts {
            reference_count: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(metadata.steps(), u32::MAX as usize);
    assert_eq!(metadata.effective_init_strength(true, false), Some(0.75));
}

#[test]
fn effective_init_strength_and_scaled_input_retention_follow_actual_mode() {
    let mut req = request("euler_ancestral");
    for (global, reference, expected) in [
        (None, None, 0.8),
        (Some(0.2), None, 0.2),
        (Some(0.2), Some(0.75), 0.75),
        (None, Some(2.0), 1.0),
    ] {
        req.strength = global;
        req.conditioning = vec![Conditioning::Reference {
            image: mlx_gen::Image::default(),
            strength: reference,
        }];
        let resolved = ResolvedSampling::resolve(&req).unwrap();
        assert_eq!(
            resolved.effective_init_strength(true, false),
            Some(expected)
        );
        assert_eq!(resolved.effective_init_strength(false, false), None);
    }
    let resolved = ResolvedSampling::resolve(&request("euler_ancestral")).unwrap();
    assert_eq!(resolved.effective_init_strength(true, true), Some(0.85));
    assert!(!ResolvedSampling::resolve(&request("lightning"))
        .unwrap()
        .retains_scaled_input_in_callback());
    assert!(ResolvedSampling::resolve(&request("euler"))
        .unwrap()
        .retains_scaled_input_in_callback());
}

#[test]
fn real_recipe_mode_validation_preserves_existing_refusals() {
    let reference = Conditioning::Reference {
        image: mlx_gen::Image::default(),
        strength: None,
    };
    let mask = Conditioning::Mask {
        image: mlx_gen::Image::default(),
    };
    let control = Conditioning::Control {
        image: mlx_gen::Image::default(),
        kind: mlx_gen::gen_core::ControlKind::Canny,
        scale: None,
    };
    for sampler in crate::model::descriptor().capabilities.samplers {
        let mut sampling_request = request(sampler);
        sampling_request.guidance = Some(7.0);
        let recipe = SamplingRecipe::from_request(&sampling_request).unwrap();
        let mut edit = GenerationRequest {
            conditioning: vec![reference.clone()],
            ..Default::default()
        };
        assert_eq!(
            recipe.apply_to(&mut edit).is_err(),
            crate::model::ACCEL_SAMPLERS.contains(&sampler)
        );
        let mut inpaint = GenerationRequest {
            conditioning: vec![reference.clone(), mask.clone()],
            ..Default::default()
        };
        assert_eq!(
            recipe.apply_to(&mut inpaint).is_ok(),
            sampler == "euler_ancestral"
        );
        let mut controlled = GenerationRequest {
            conditioning: vec![control.clone()],
            ..Default::default()
        };
        // Ordinary request defaults have CFG 7, so Lightning too must refuse this explicit guidance.
        assert_eq!(
            recipe.apply_to(&mut controlled).is_err(),
            crate::model::ACCEL_SAMPLERS.contains(&sampler)
        );
    }
    let lightning = SamplingRecipe::from_request(&GenerationRequest {
        sampler: Some("lightning".into()),
        ..Default::default()
    })
    .unwrap();
    let mut controlled = GenerationRequest {
        conditioning: vec![control],
        ..Default::default()
    };
    assert!(lightning.apply_to(&mut controlled).is_ok());
    let mut masked = GenerationRequest {
        conditioning: vec![mask],
        ..Default::default()
    };
    assert!(SamplingRecipe::from_request(&request("euler_ancestral"))
        .unwrap()
        .apply_to(&mut masked)
        .is_err());
}

#[test]
fn descriptor_routes_reach_real_host_factories_and_cover_capture_classes() {
    let capture_classes: BTreeSet<_> = representative_recipes()
        .iter()
        .map(|r| {
            ResolvedSampling::resolve(&r.to_request())
                .unwrap()
                .allocation_class()
        })
        .collect();
    let mut served_classes = BTreeSet::new();
    // Producer is SDXL's descriptor, never Solver::ALL or a duplicate advertised-name list.
    for name in crate::model::descriptor().capabilities.samplers {
        for scheduler in [None, Some("normal")] {
            for method in ["cfg", "cfg_pp"] {
                for (guidance, batching) in [
                    (1.0, CfgBatching::Batched),
                    (7.0, CfgBatching::Batched),
                    (7.0, CfgBatching::Sequential),
                ] {
                    let mut req = request(name);
                    req.scheduler = scheduler.map(str::to_owned);
                    req.guidance_method = Some(method.into());
                    req.guidance = Some(guidance);
                    req.memory.get_or_insert_with(Default::default).cfg_batching = Some(batching);
                    if crate::model::ACCEL_SAMPLERS.contains(&name) && scheduler.is_some() {
                        assert!(ResolvedSampling::resolve(&req).is_err());
                        continue;
                    }
                    let resolved = ResolvedSampling::resolve(&req).unwrap();
                    assert_eq!(resolved.sampler_name(), name);
                    served_classes.insert(resolved.allocation_class());
                    match resolved.route() {
                        SamplingRoute::Ancestral => {
                            assert_eq!(
                                resolved.ancestral_window(1000.0, None).unwrap(),
                                (3, 1000.0)
                            );
                            assert!(scheduler.is_none());
                        }
                        SamplingRoute::Lcm | SamplingRoute::Lightning | SamplingRoute::Hyper => {
                            let epochs = resolved.acceleration_epochs(&alpha()).unwrap();
                            assert_eq!(epochs.len(), 3);
                            assert!(epochs.last().unwrap().final_step);
                            assert!(!epochs.last().unwrap().draws_noise);
                            assert_eq!(
                                resolved.route(),
                                match name {
                                    "lcm" => SamplingRoute::Lcm,
                                    "lightning" => SamplingRoute::Lightning,
                                    "hyper" => SamplingRoute::Hyper,
                                    _ =>
                                        panic!("non-acceleration descriptor routed to acceleration"),
                                }
                            );
                            let factory = resolved.build_acceleration_sampler(&alpha(), 7).unwrap();
                            assert_eq!(factory.num_steps(), epochs.len());
                            for epoch in &epochs {
                                assert_eq!(
                                    factory.timestep(epoch.step),
                                    epoch.coefficients.timestep
                                );
                            }
                            if resolved.route() == SamplingRoute::Hyper {
                                assert!(epochs
                                    .iter()
                                    .all(|e| !e.draws_noise && e.input_scale_alias));
                            }
                        }
                        SamplingRoute::Curated(solver) | SamplingRoute::CfgPp(solver) => {
                            // These are the SAME Solver::boxed / cfgpp_sampler_for factories
                            // used by production run_*_sampler_with_latent_hook.
                            assert_eq!(Solver::from_name(resolved.sampler_name()), Some(solver));
                            for sigmas in [&[2.0, 1.0, 0.0][..], &[2.0, 1.0, 0.5][..], &[0.0][..]] {
                                let expected = resolved.logical_epochs(sigmas).unwrap();
                                let mut seen = Vec::new();
                                match resolved.route() {
                                    SamplingRoute::CfgPp(_) => {
                                        cfgpp_sampler_for::<CpuLatentOps>(solver)
                                            .unwrap()
                                            .sample(
                                                &CpuLatentOps,
                                                &mut |_: &Vec<f32>, sigma| {
                                                    seen.push(sigma);
                                                    Ok((vec![0.25], vec![0.125]))
                                                },
                                                vec![0.5],
                                                sigmas,
                                            )
                                            .unwrap();
                                    }
                                    _ => {
                                        solver
                                            .boxed::<CpuLatentOps>()
                                            .sample(
                                                &CpuLatentOps,
                                                &DiscreteModelSampling::sdxl(&alpha()),
                                                &mut |_: &Vec<f32>, sigma| {
                                                    seen.push(sigma);
                                                    Ok(vec![0.25])
                                                },
                                                vec![0.5],
                                                sigmas,
                                                7,
                                            )
                                            .unwrap();
                                    }
                                }
                                assert_eq!(
                                    seen,
                                    expected.iter().map(|e| e.sigma).collect::<Vec<_>>(),
                                    "{name} {method} {sigmas:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        served_classes, capture_classes,
        "representatives must cover actual factories including CFG input graphs"
    );
}

#[test]
fn defaults_scheduler_switch_and_cfgpp_fallback_are_effective_execution() {
    let default = ResolvedSampling::resolve(&GenerationRequest::default()).unwrap();
    assert_eq!(
        (default.route(), default.steps(), default.guidance()),
        (SamplingRoute::Ancestral, 30, 7.0)
    );
    for (name, steps) in [("lcm", 8), ("lightning", 4), ("hyper", 4)] {
        let req = GenerationRequest {
            sampler: Some(name.into()),
            ..Default::default()
        };
        let resolved = ResolvedSampling::resolve(&req).unwrap();
        assert_eq!((resolved.steps(), resolved.guidance()), (steps, 1.0));
    }
    for name in crate::model::descriptor().capabilities.samplers {
        let mut req = request(name);
        req.guidance_method = Some("cfg_pp".into());
        req.guidance = Some(7.0);
        let resolved = ResolvedSampling::resolve(&req).unwrap();
        assert_eq!(
            matches!(resolved.route(), SamplingRoute::CfgPp(_)),
            matches!(name, "euler" | "ddim" | "dpmpp_2m")
        );
        req.guidance = Some(1.0);
        assert!(!matches!(
            ResolvedSampling::resolve(&req).unwrap().route(),
            SamplingRoute::CfgPp(_)
        ));
    }
    let mut req = request("euler_ancestral");
    req.scheduler = Some("normal".into());
    assert_eq!(
        ResolvedSampling::resolve(&req).unwrap().route(),
        SamplingRoute::Curated(Solver::EulerAncestral)
    );
    for scheduler in [None, Some("discrete")] {
        req.scheduler = scheduler.map(str::to_owned);
        assert_eq!(
            ResolvedSampling::resolve(&req).unwrap().route(),
            SamplingRoute::Ancestral
        );
    }
    let mut euler = request("euler");
    euler.guidance_method = Some("cfg_pp".into());
    let mut ddim = request("ddim");
    ddim.guidance_method = Some("cfg_pp".into());
    assert_eq!(
        ResolvedSampling::resolve(&euler).unwrap().memory_class(),
        ResolvedSampling::resolve(&ddim).unwrap().memory_class()
    );
    euler.guidance_method = None;
    ddim.guidance_method = None;
    assert_ne!(
        ResolvedSampling::resolve(&euler).unwrap().memory_class(),
        ResolvedSampling::resolve(&ddim).unwrap().memory_class()
    );
    assert!(ResolvedSampling::for_pricing(None).is_err());
    assert!(ResolvedSampling::for_pricing(Some(&GenerationRequest::default())).is_ok());
}

#[test]
fn canonical_recipe_validates_serde_and_preserves_raw_and_effective_strength() {
    for recipe in representative_recipes() {
        let token = recipe.canonical_token();
        assert!(token
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-'));
        assert_eq!(SamplingRecipe::from_token(&token).unwrap(), recipe);
        assert_eq!(
            serde_json::from_value::<SamplingRecipe>(recipe.to_value()).unwrap(),
            recipe
        );
        assert_eq!(
            SamplingRecipe::from_request(&recipe.to_request()).unwrap(),
            recipe
        );
    }
    let mut req = request("heun");
    req.strength = Some(0.2);
    req.conditioning.push(Conditioning::Reference {
        image: mlx_gen::Image::default(),
        strength: Some(0.75),
    });
    let recipe = SamplingRecipe::from_request(&req).unwrap();
    let decoded = SamplingRecipe::from_token(&recipe.canonical_token()).unwrap();
    assert_eq!(decoded, recipe);
    assert_eq!(decoded.to_request().strength, Some(0.75));
    let mut replay = request("euler");
    replay.conditioning.push(Conditioning::Reference {
        image: mlx_gen::Image::default(),
        strength: None,
    });
    decoded.apply_to(&mut replay).unwrap();
    assert_eq!(SamplingRecipe::from_request(&replay).unwrap(), recipe);
    assert_eq!(replay.strength, Some(0.2));
    let value = recipe.to_value();
    for (field, bad) in [
        ("version", json!(2)),
        ("steps", json!(0)),
        ("sampler", json!("er_sde")),
        ("guidance", Value::Null),
        ("cfg_batching", json!("mystery")),
        ("claimed_class", json!("sdxl1-heun")),
    ] {
        let mut mutated = value.clone();
        mutated[field] = bad;
        assert!(SamplingRecipe::from_value(mutated).is_err(), "{field}");
    }
    for field in [
        "version",
        "guidance",
        "strength",
        "reference_strength",
        "pid_capture_sigma",
    ] {
        let mut mutated = value.clone();
        mutated.as_object_mut().unwrap().remove(field);
        assert!(
            SamplingRecipe::from_value(mutated).is_err(),
            "missing {field}"
        );
    }
    req.guidance = Some(f32::NAN);
    assert!(SamplingRecipe::from_request(&req).is_err());
    assert!(SamplingRecipe::from_token("sdxl1-zz").is_err());
    assert!(SamplingRecipe::from_token("sdxl1-/../../").is_err());
}

#[test]
fn schedules_use_actual_transition_count_strength_and_nonzero_terminal() {
    let mut req = request("heun");
    req.steps = Some(300);
    for scheduler in crate::model::descriptor().capabilities.schedulers {
        req.scheduler = Some(scheduler.into());
        let resolved = ResolvedSampling::resolve(&req).unwrap();
        let run = resolved.curated_sigmas(&alpha(), None).unwrap();
        let ms = DiscreteModelSampling::sdxl(&alpha());
        let expected = schedule_sigmas(
            Scheduler::from_name(scheduler).unwrap_or(Scheduler::Normal),
            &ms,
            300,
        );
        assert_eq!(run, expected);
        assert_eq!(
            resolved
                .logical_epochs(&run)
                .unwrap()
                .iter()
                .filter(|e| e.evaluation == Evaluation::Primary)
                .count(),
            run.len() - 1
        );
        let zero = resolved.curated_sigmas(&alpha(), Some(0.0)).unwrap();
        assert_eq!(zero, vec![0.0]);
        assert!(resolved.logical_epochs(&zero).unwrap().is_empty());
    }
    req.scheduler = Some("normal".into());
    req.steps = Some(4);
    req.use_pid = true;
    req.pid_capture_sigma = Some(0.9);
    let resolved = ResolvedSampling::resolve(&req).unwrap();
    let full = resolved.curated_full_sigmas(&alpha()).unwrap();
    let plan = vp_capture_plan(&full, req.pid_capture_sigma).unwrap();
    let run = resolved.curated_sigmas(&alpha(), None).unwrap();
    assert_eq!(run, full[..plan.keep]);
    assert!(*run.last().unwrap() > 0.0);
    assert_eq!(
        resolved
            .logical_epochs(&run)
            .unwrap()
            .last()
            .unwrap()
            .evaluation,
        Evaluation::HeunCorrector
    );
    let ancestral = ResolvedSampling::resolve(&request("euler_ancestral")).unwrap();
    assert_eq!(ancestral.ancestral_window(1000.0, Some(0.01)).unwrap().0, 0);
}

#[test]
fn logical_history_corrector_and_input_order_are_not_whole_phase_alias_proofs() {
    for (name, owners) in [("dpmpp_2m", 1), ("uni_pc", 2)] {
        let resolved = ResolvedSampling::resolve(&request(name)).unwrap();
        let epochs = resolved.logical_epochs(&[2.0, 1.0, 0.0]).unwrap();
        assert_eq!(epochs[0].history_owners, 0);
        assert_eq!(epochs[1].history_owners, owners);
        assert!(epochs[1].zero_landing);
        assert_eq!(epochs[1].corrector_after_callback, name == "uni_pc");
    }
    for (name, owners) in [("heun", 3), ("dpmpp_sde", 2)] {
        let resolved = ResolvedSampling::resolve(&request(name)).unwrap();
        let epochs = resolved.logical_epochs(&[2.0, 1.0, 0.0]).unwrap();
        assert_eq!(epochs[1].outer_owners, owners);
        assert_eq!(epochs.last().unwrap().evaluation, Evaluation::Primary);
        assert!(resolved.retains_initial_noise());
    }
    let mut req = request("dpmpp_2m");
    req.guidance_method = Some("cfg_pp".into());
    let resolved = ResolvedSampling::resolve(&req).unwrap();
    assert!(resolved
        .logical_epochs(&[2.0, 1.0, 0.0])
        .unwrap()
        .iter()
        .all(|e| e.denoised_outputs == 2));
    req.count = 8;
    req.memory.get_or_insert_with(Default::default).cfg_batching = Some(CfgBatching::Sequential);
    assert_eq!(ResolvedSampling::resolve(&req).unwrap().denoise_rows(), 1);
    assert_eq!(
        ResolvedSampling::resolve(&request("euler_ancestral"))
            .unwrap()
            .input_ownership(),
        InputOwnership::AncestralAliasThenCfgThenCast
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metadata_resolution_does_not_allocate_mlx_or_submit_metal() {
    const CHILD: &str = "SDXL_SAMPLING_METADATA_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "sampling::tests::metadata_resolution_does_not_allocate_mlx_or_submit_metal",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    assert!(crate::test_metal_guard::forbid_metal());
    assert!(crate::test_metal_guard::metal_forbidden());
    assert_eq!(
        unsafe { mlx_sys::mlx_set_default_stream(mlx_rs::Stream::cpu().as_ptr()) },
        0
    );
    assert_eq!(
        mlx_rs::Stream::new().get_index().unwrap(),
        mlx_rs::Stream::cpu().get_index().unwrap()
    );
    mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || {
        let before = mlx_rs::memory::get_active_memory();
        mlx_rs::memory::reset_peak_memory();
        for recipe in representative_recipes() {
            let resolved = ResolvedSampling::resolve(&recipe.to_request()).unwrap();
            let _roundtrip = SamplingRecipe::from_token(&recipe.canonical_token()).unwrap();
            if resolved.route().is_curated() {
                resolved.curated_sigmas(&alpha(), None).unwrap();
            }
            if resolved.route().is_acceleration() {
                resolved.acceleration_coefficients(&alpha()).unwrap();
            }
        }
        assert_eq!(mlx_rs::memory::get_active_memory(), before);
        assert_eq!(mlx_rs::memory::get_peak_memory(), before);
    });
}
