//! Adapt sealed, completed MOSS receipts, never submitted requests, to the shared identity.
use gen_core::generator::{
    ExecutionDraw, ExecutionFact, ExecutionIdentity, ExecutionRandomness, ExecutionReplaySupport,
    ExecutionUnavailable, NumericScheduleDigest, ResolvedExecutionIdentity,
    EXECUTION_IDENTITY_VERSION,
};
use gen_core::{
    moss_image::{
        MossImageExecutionDevice, MossImageExecutionReport, MossImageSamplingReceipt,
        MossImageSelectedStreamDevice,
    },
    moss_image_edit::MossImageEditExecutionReport,
    ModelDescriptor, Result,
};
use serde_json::json;
use std::collections::BTreeMap;

/// Call while the provider still holds its attempt lock, after success and its final cancel check.
/// Test-support graphs deliberately have no sealed source receipt; they cannot attest replay.
pub fn completed_identity(
    report: Option<&MossImageExecutionReport>,
    edit: Option<&MossImageEditExecutionReport>,
    descriptor: &ModelDescriptor,
) -> Result<ExecutionIdentity> {
    let report = match (report, edit) {
        (None, Some(edit)) => {
            edit.validate()?;
            &edit.base
        }
        (Some(report), None) => report,
        (Some(_), Some(_)) => {
            return Err(gen_core::Error::Msg(
                "ambiguous MOSS completed receipts".into(),
            ))
        }
        (None, None) => {
            return Ok(ExecutionIdentity::unavailable(ExecutionUnavailable {
                code: "moss_unsealed_source_receipt".into(),
                detail: "completed MOSS attempt has no unique sealed asset receipt; test-support sources cannot attest published assets".into(),
                missing_facts: vec!["sealed_moss_execution_receipt".into()],
            }));
        }
    };
    let data = report.data();
    let r = &data.resolved;
    let selected_stream_device = if data.execution_device == MossImageExecutionDevice::Mlx {
        match data.selected_stream_device {
            Some(MossImageSelectedStreamDevice::Cpu) => Some("mlx-cpu"),
            Some(MossImageSelectedStreamDevice::Metal) => Some("mlx-metal"),
            None => {
                return Ok(ExecutionIdentity::unavailable(ExecutionUnavailable {
                    code: "moss_mlx_device_unobserved".into(),
                    detail: "selected MLX stream device was unavailable or changed during the completed attempt".into(),
                    missing_facts: vec!["selected_stream_device".into()],
                }));
            }
        }
    } else {
        None
    };
    let backend = if data.execution_device == MossImageExecutionDevice::Mlx {
        "mlx"
    } else {
        "candle"
    };
    let (sampling, schedule) = match &data.sampling {
        MossImageSamplingReceipt::Euler { sigmas } => (
            json!({"solver":"euler", "steps":sigmas.len()-1, "grid_steps":r.steps}),
            json!({"sigmas":NumericScheduleDigest::f32_values(sigmas)?}),
        ),
        MossImageSamplingReceipt::Dmd {
            step_codes,
            mix_matrix_sha256,
        } => (
            json!({"solver":"data_space_dmd", "steps":step_codes.len()}),
            json!({"step_codes":NumericScheduleDigest::f32_values(step_codes)?, "mix_matrix_sha256":mix_matrix_sha256}),
        ),
    };
    let rng = if data.samples.iter().all(|sample| sample.seed.is_none()) {
        "explicit-f32-noise-no-RNG"
    } else {
        match data.execution_device {
            MossImageExecutionDevice::Mlx => "splitmix64-box_muller-f64-to-f32-v1",
            _ => "rand-StdRng-rand_distr-StandardNormal-f32-v1",
        }
    };
    let mut choices = BTreeMap::from([
        ("assets".into(), ExecutionFact::resolved(json!(data.assets))),
        (
            "conditioning".into(),
            ExecutionFact::resolved(json!(data.conditioning)),
        ),
        (
            "images_per_prompt".into(),
            ExecutionFact::resolved(data.images_per_prompt),
        ),
    ]);
    if let Some(device) = selected_stream_device {
        choices.insert(
            "selected_stream_device".into(),
            ExecutionFact::resolved(device),
        );
    }
    if let Some(edit) = edit {
        // Paths are operational locations, not replay identities.
        choices.insert(
            "edit".into(),
            ExecutionFact::resolved(json!({
                "contract":edit.edit_contract, "adapter_repository":edit.adapter_repository,
                "adapter_revision":edit.adapter_revision, "adapter_sha256":edit.adapter_sha256,
                "config_sha256":edit.config_sha256, "applied_projections":edit.applied_projections,
                "references":edit.references, "reference_max_edge":edit.reference_max_edge,
                "use_kv_cache":edit.use_kv_cache,
            })),
        );
    }
    let identity = ExecutionIdentity::Resolved {
        identity: Box::new(ResolvedExecutionIdentity {
            version: EXECUTION_IDENTITY_VERSION,
            provider_id: r.variant.provider_id().into(),
            backend: backend.into(),
            runtime_id: format!("{}-moss-image@{}-receipt-v1/{:?}{}", backend, env!("CARGO_PKG_VERSION"), data.execution_device,
                selected_stream_device.map_or(String::new(), |device| format!("/{device}"))),
            sampling: ExecutionFact::resolved(sampling),
            schedule: ExecutionFact::resolved(schedule),
            estimator: ExecutionFact::resolved("flow_velocity"),
            randomness: ExecutionRandomness::Draws {
                scheme: format!("{rng}; explicit-noise-overrides-seed; per-image-NCHW-f32; prompt-major-image-minor"),
                draws: data.samples.iter().enumerate().map(|(output_index, sample)| ExecutionDraw {
                    output_index, purpose: "initial_latent_noise".into(), seed: sample.seed,
                    noise_sha256: Some(sample.noise_sha256.clone()),
                }).collect(),
            },
            // The legacy sealed receipt proves executed choices, not cross-launch kernel determinism.
            deterministic_replay: ExecutionReplaySupport::NonDeterministic {
                reason: "MOSS native kernels do not attest bit-exact replay across executions; preserve runtime, device and sealed input bytes for regeneration".into(),
            },
            geometry: ExecutionFact::resolved(json!({"latent_shape":data.latent_shape, "pixel_height":data.latent_shape[2]*16, "pixel_width":data.latent_shape[3]*16})),
            quality_policy: {
                let mut quality = json!({
                    "variant":r.variant, "acceleration":r.acceleration,
                    "guidance":ExecutionFact::f32_bits(r.guidance)?,
                    "shift":ExecutionFact::f32_bits(r.shift)?,
                    "training_shift":ExecutionFact::f32_bits(r.training_shift)?,
                    "max_sequence_length":r.max_sequence_length,
                    "uncond_skip_layers":r.uncond_skip_layers,
                    "slg":r.slg.as_ref().map(|slg| Ok::<_,gen_core::Error>(json!({"layers":slg.layers, "scale":ExecutionFact::f32_bits(slg.scale)?, "start":ExecutionFact::f32_bits(slg.start)?, "stop":ExecutionFact::f32_bits(slg.stop)?}))).transpose()?,
                    "component_dtypes":data.component_dtypes,
                    "adapters":data.adapter_reports.iter().map(|a| json!({"applied":a.applied,"skipped":a.skipped})).collect::<Vec<_>>(),
                });
                if let Some(pag) = data.pag.as_ref() {
                    quality["pag"] = json!({
                        "layers": pag.layers,
                        "active_steps": pag.active_steps,
                        "total_steps": pag.total_steps,
                        "scale": ExecutionFact::f32_bits(pag.scale)?,
                        "adaptive_scale": ExecutionFact::f32_bits(pag.adaptive_scale)?,
                        "start": ExecutionFact::f32_bits(pag.start)?,
                        "stop": ExecutionFact::f32_bits(pag.stop)?,
                    });
                }
                ExecutionFact::resolved(quality)
            },
            output_policy: ExecutionFact::resolved(json!({"kind":data.output_kind, "images":"rgb8", "latents":"NCHW-f32"})),
            provider_choices: choices,
            output_units: data.samples.len(),
        }),
    };
    identity.validate(descriptor.id, descriptor.backend, data.samples.len())?;
    Ok(identity)
}

/// Retain the uniform controls of the installed custom stack in load order. The app binds each
/// ordinal to its frozen source bytes; paths and raw tensor metadata do not enter this report.
/// Only call after installation succeeded, with the attempt's actual application receipts.
pub fn record_custom_adapter_policy(
    execution: &mut ExecutionIdentity,
    specs: &[gen_core::AdapterSpec],
    applications: &[gen_core::AdapterApplyReport],
) -> Result<()> {
    let ExecutionIdentity::Resolved { identity } = execution else {
        return Ok(());
    };
    let policies = specs.iter().enumerate().map(|(ordinal,spec)| {
        let installed = applications.iter().find(|report| report.adapter_path == spec.path)
            .ok_or_else(|| gen_core::Error::Msg("completed MOSS custom adapter lacks its installation receipt".into()))?;
        Ok(json!({"ordinal":ordinal, "kind":match spec.kind { gen_core::AdapterKind::Lora => "lora", gen_core::AdapterKind::Lokr => "lokr" },
            "uniform_scale":ExecutionFact::f32_bits(spec.scale)?, "applied":installed.applied, "skipped":installed.skipped}))
    }).collect::<Result<Vec<_>>>()?;
    identity.provider_choices.insert(
        "custom_adapter_stack".into(),
        ExecutionFact::resolved(json!(policies)),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moss_image::*;
    use gen_core::{Capabilities, GenerationRequest, Modality};

    fn descriptor(backend: &'static str) -> ModelDescriptor {
        ModelDescriptor {
            id: "moss_image_sft",
            family: "moss_image",
            backend,
            modality: Modality::Image,
            capabilities: Capabilities::default(),
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
        }
    }

    // Metadata only: these bytes are never loaded or executed as model weights.
    fn receipt(
        mode: MossImageAcceleration,
        device: MossImageExecutionDevice,
    ) -> MossImageExecutionReport {
        let r = MossImageResolved::resolve(
            MossImageVariant::Sft,
            &GenerationRequest {
                extensions: gen_core::FamilyExtensions {
                    moss_image: Some(gen_core::MossImageExtension::from_options(
                        MossImageOptions {
                            acceleration: mode,
                            ..Default::default()
                        },
                    )),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let mut paths: Vec<_> = SNAPSHOT_FILES
            .iter()
            .filter(|p| **p != "LICENSE")
            .map(|p| {
                let role = if p.starts_with("transformer/") {
                    MossImageAssetRole::Transformer
                } else if p.starts_with("text_encoder/") {
                    MossImageAssetRole::TextEncoder
                } else if p.starts_with("tokenizer/") {
                    MossImageAssetRole::Tokenizer
                } else if p.starts_with("scheduler/") {
                    MossImageAssetRole::Scheduler
                } else {
                    MossImageAssetRole::Vae
                };
                (role, p.to_string())
            })
            .collect();
        match mode {
            MossImageAcceleration::Off => (),
            MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16) => paths.extend([
                (MossImageAssetRole::DmdLora, "lora.safetensors".into()),
                (
                    MossImageAssetRole::DmdTime,
                    "time_embedding.safetensors".into(),
                ),
                (
                    MossImageAssetRole::DmdDataspace,
                    "dataspace.safetensors".into(),
                ),
            ]),
            MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32) => {
                paths.retain(|(_, p)| !p.starts_with("transformer/diffusion_pytorch_model-000"));
                for i in 1..=4 {
                    paths.push((
                        MossImageAssetRole::Transformer,
                        format!("transformer/diffusion_pytorch_model-{i:05}-of-00004.safetensors"),
                    ));
                }
                paths.push((
                    MossImageAssetRole::DmdDataspace,
                    "dataspace.safetensors".into(),
                ));
            }
        }
        let assets = paths
            .into_iter()
            .map(|(role, path)| {
                let dmd = matches!(
                    role,
                    MossImageAssetRole::DmdLora
                        | MossImageAssetRole::DmdTime
                        | MossImageAssetRole::DmdDataspace
                ) || (role == MossImageAssetRole::Transformer
                    && mode == MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32));
                MossImageAssetReceipt {
                    role,
                    repository: if dmd { DMD_REPO } else { SFT_REPO }.into(),
                    revision: if dmd { DMD_REVISION } else { SFT_REVISION }.into(),
                    relative_path: path,
                    sha256: "a".repeat(64),
                }
            })
            .collect();
        let sampling = match mode {
            MossImageAcceleration::Off => MossImageSamplingReceipt::Euler {
                sigmas: moss_image_schedule(r.steps, r.shift, r.training_shift).unwrap(),
            },
            _ => MossImageSamplingReceipt::Dmd {
                step_codes: vec![0.5; r.steps as usize],
                mix_matrix_sha256: "b".repeat(64),
            },
        };
        let dtype = MossImageDtype::Bf16;
        let mut component_dtypes: BTreeMap<_, _> = [
            "transformer",
            "text_encoder",
            "vae",
            "latents",
            "time_embedding",
            "shared_block_modulation",
            "video_head_modulation",
        ]
        .into_iter()
        .map(|k| (k.into(), dtype))
        .collect();
        for k in ["vae", "latents"] {
            component_dtypes.insert(k.into(), MossImageDtype::F32);
        }
        if device == MossImageExecutionDevice::Cpu {
            component_dtypes.insert("text_encoder".into(), MossImageDtype::F32);
        }
        let mut adapter_reports = vec![];
        if let MossImageAcceleration::Dmd(precision) = mode {
            for k in [
                "time_embedding",
                "shared_block_modulation",
                "video_head_modulation",
            ] {
                component_dtypes.insert(k.into(), MossImageDtype::F32);
            }
            if precision == MossDmdPrecision::FullFp32 {
                component_dtypes.insert("transformer".into(), MossImageDtype::F32);
            } else {
                component_dtypes.insert("lora_residual".into(), MossImageDtype::F32);
                adapter_reports.push(gen_core::AdapterApplyReport {
                    adapter_path: "/not/a/model".into(),
                    applied: 252,
                    skipped: vec![],
                });
            }
        }
        let input = MossImageConditioningInputReceipt {
            source: MossImageConditioningSource::Text,
            sha256: "c".repeat(64),
            sequence_length: 2,
            valid_tokens: 2,
        };
        MossImageExecutionReport::new(MossImageExecutionData {
            resolved: r,
            execution_device: device,
            selected_stream_device: (device == MossImageExecutionDevice::Mlx)
                .then_some(MossImageSelectedStreamDevice::Metal),
            assets,
            sampling,
            samples: vec![
                MossImageSampleReceipt {
                    seed: Some(u64::MAX),
                    noise_sha256: "d".repeat(64),
                },
                MossImageSampleReceipt {
                    seed: Some(0),
                    noise_sha256: "e".repeat(64),
                },
            ],
            conditioning: vec![MossImageConditioningReceipt {
                positive: input.clone(),
                negative: (mode == MossImageAcceleration::Off).then_some(input),
            }],
            images_per_prompt: 2,
            latent_shape: [2, 128, 2, 3],
            output_kind: MossImageOutputKind::Images,
            component_dtypes,
            adapter_reports,
            pag: None,
        })
        .unwrap()
    }

    #[test]
    fn mlx_selected_stream_device_changes_identity_and_missing_observation_is_unavailable() {
        let metal = receipt(MossImageAcceleration::Off, MossImageExecutionDevice::Mlx);
        let mut cpu_data = metal.clone().into_data();
        cpu_data.selected_stream_device = Some(MossImageSelectedStreamDevice::Cpu);
        let cpu = MossImageExecutionReport::new(cpu_data).unwrap();
        let metal_identity = completed_identity(Some(&metal), None, &descriptor("mlx")).unwrap();
        let cpu_identity = completed_identity(Some(&cpu), None, &descriptor("mlx")).unwrap();
        assert_ne!(
            serde_json::to_value(metal_identity).unwrap(),
            serde_json::to_value(cpu_identity).unwrap()
        );

        let mut unknown_data = metal.into_data();
        unknown_data.selected_stream_device = None;
        let unknown = MossImageExecutionReport::new(unknown_data).unwrap();
        let identity = completed_identity(Some(&unknown), None, &descriptor("mlx")).unwrap();
        assert!(matches!(identity, ExecutionIdentity::Unavailable { .. }));
    }

    #[test]
    fn shared_reports_distinguish_effective_modes_devices_and_ordered_draws() {
        let mut reports = vec![];
        for device in [
            MossImageExecutionDevice::Cpu,
            MossImageExecutionDevice::Cuda,
            MossImageExecutionDevice::Metal,
            MossImageExecutionDevice::Mlx,
        ] {
            let backend = if device == MossImageExecutionDevice::Mlx {
                "mlx"
            } else {
                "candle"
            };
            for mode in [
                MossImageAcceleration::Off,
                MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16),
                MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32),
            ] {
                let report = receipt(mode, device);
                let shared = completed_identity(Some(&report), None, &descriptor(backend)).unwrap();
                let id = shared.resolved().unwrap();
                assert_eq!(id.output_units, 2);
                let ExecutionFact::Resolved { value: sampling } = &id.sampling else {
                    panic!("sampling facts")
                };
                let ExecutionFact::Resolved { value: quality } = &id.quality_policy else {
                    panic!("quality facts")
                };
                if mode == MossImageAcceleration::Off {
                    assert_eq!(sampling["solver"], "euler");
                    assert_eq!(sampling["steps"], 50);
                    assert_eq!(
                        quality["guidance"],
                        serde_json::to_value(ExecutionFact::f32_bits(3.0).unwrap()).unwrap()
                    );
                    assert_eq!(quality["uncond_skip_layers"], json!([22]));
                } else {
                    assert_eq!(sampling["solver"], "data_space_dmd");
                    assert_eq!(sampling["steps"], 8);
                    assert_eq!(
                        quality["guidance"],
                        serde_json::to_value(ExecutionFact::f32_bits(0.0).unwrap()).unwrap()
                    );
                    assert_eq!(quality["uncond_skip_layers"], json!([]));
                }
                assert!(!id.deterministic_replay.is_deterministic());
                let ExecutionRandomness::Draws { scheme, draws } = &id.randomness else {
                    panic!("draws")
                };
                assert!(scheme.contains(if backend == "mlx" {
                    "splitmix64"
                } else {
                    "StdRng"
                }));
                assert_eq!(
                    draws
                        .iter()
                        .map(|d| (d.output_index, d.seed))
                        .collect::<Vec<_>>(),
                    [(0, Some(u64::MAX)), (1, Some(0))]
                );
                let wire = serde_json::to_string(&shared).unwrap();
                assert!(!wire.contains("/not/a/model"));
                assert!(
                    !reports.contains(&wire),
                    "precision/mode/device must distinguish reports"
                );
                reports.push(wire);
            }
        }
    }

    #[test]
    fn supplied_noise_and_effective_nondefaults_survive_adaptation() {
        let original = receipt(MossImageAcceleration::Off, MossImageExecutionDevice::Mlx);
        let mut data = original.clone().into_data();
        data.samples[0].seed = None;
        data.resolved.steps = 3;
        data.resolved.shift = 2.0;
        data.resolved.slg = Some(MossSkipLayerGuidance {
            layers: vec![3],
            scale: 1.2,
            start: 0.1,
            stop: 0.8,
        });
        data.sampling = MossImageSamplingReceipt::Euler {
            sigmas: moss_image_schedule(3, 2.0, data.resolved.training_shift).unwrap(),
        };
        let changed = MossImageExecutionReport::new(data).unwrap();
        let base = completed_identity(Some(&original), None, &descriptor("mlx")).unwrap();
        let shared = completed_identity(Some(&changed), None, &descriptor("mlx")).unwrap();
        let id = shared.resolved().unwrap();
        assert_ne!(base.resolved().unwrap().schedule, id.schedule);
        assert_ne!(base.resolved().unwrap().quality_policy, id.quality_policy);
        let ExecutionRandomness::Draws { draws, .. } = &id.randomness else {
            panic!("draws")
        };
        assert_eq!(draws[0].seed, None);
        assert_eq!(draws[0].noise_sha256, Some("d".repeat(64)));
    }

    #[test]
    fn snapped_schedule_reports_executed_steps_instead_of_grid_request() {
        let mut data =
            receipt(MossImageAcceleration::Off, MossImageExecutionDevice::Mlx).into_data();
        data.resolved.steps = 2000;
        let sigmas =
            moss_image_schedule(2000, data.resolved.shift, data.resolved.training_shift).unwrap();
        let actual = sigmas.len() - 1;
        assert!(
            actual < 2000,
            "snapped training grid collapses repeated samples"
        );
        data.sampling = MossImageSamplingReceipt::Euler { sigmas };
        let report = MossImageExecutionReport::new(data).unwrap();
        let shared = completed_identity(Some(&report), None, &descriptor("mlx")).unwrap();
        let ExecutionFact::Resolved { value } = &shared.resolved().unwrap().sampling else {
            panic!("sampling facts")
        };
        assert_eq!(value["steps"], actual);
        assert_eq!(value["grid_steps"], 2000);
    }

    #[test]
    fn edit_cache_reference_and_latent_output_policies_survive_completion() {
        use gen_core::generator::CompletedMossImageEdit;
        use gen_core::moss_image_edit::*;
        for cache in [false, true] {
            for output_kind in [MossImageOutputKind::Images, MossImageOutputKind::Latents] {
                let mut data =
                    receipt(MossImageAcceleration::Off, MossImageExecutionDevice::Mlx).into_data();
                data.resolved.uncond_skip_layers.clear();
                data.output_kind = output_kind;
                for sample in &mut data.samples {
                    sample.seed = None;
                }
                let edit = MossImageEditExecutionReport {
                    base: MossImageExecutionReport::new(data).unwrap(),
                    adapter_path: "/private/asset/location".into(),
                    edit_contract: EDIT_CONTRACT.into(),
                    adapter_repository: EDIT_REPOSITORY.into(),
                    adapter_revision: EDIT_REVISION.into(),
                    adapter_sha256: EDIT_WEIGHTS_SHA256.into(),
                    config_sha256: EDIT_CONFIG_SHA256.into(),
                    applied_projections: 252,
                    references: vec![MossImageEditReferenceReceipt {
                        original_dimensions: [32, 48],
                        encoded_dimensions: [32, 48],
                        pixels_sha256: "1".repeat(64),
                        resized_pixels_sha256: "2".repeat(64),
                        latent_sha256: "3".repeat(64),
                    }],
                    reference_max_edge: 512,
                    use_kv_cache: cache,
                    pag: None,
                };
                let shared = completed_identity(None, Some(&edit), &descriptor("mlx")).unwrap();
                let id = shared.resolved().unwrap();
                let ExecutionFact::Resolved { value } = &id.provider_choices["edit"] else {
                    panic!("edit facts")
                };
                assert_eq!(value["use_kv_cache"], cache);
                assert_eq!(value["references"][0]["latent_sha256"], "3".repeat(64));
                let ExecutionRandomness::Draws { scheme, .. } = &id.randomness else {
                    panic!("draws")
                };
                assert!(scheme.starts_with("explicit-f32-noise-no-RNG"));
                assert!(!serde_json::to_string(&shared)
                    .unwrap()
                    .contains("/private/asset/location"));
                let output = match output_kind {
                    MossImageOutputKind::Images => MossImageAdvancedOutput::Images(vec![
                            gen_core::Image {
                                width: 48,
                                height: 32,
                                pixels: vec![0; 48 * 32 * 3]
                            };
                            2
                        ]),
                    MossImageOutputKind::Latents => MossImageAdvancedOutput::Latents {
                        values: vec![0.; 2 * 128 * 2 * 3],
                        shape: [2, 128, 2, 3],
                    },
                };
                CompletedMossImageEdit::new(output, shared, &descriptor("mlx")).unwrap();
                assert!(
                    completed_identity(Some(&edit.base), Some(&edit), &descriptor("mlx")).is_err()
                );

                // Fabricated edit.pag on a non-PAG base receipt must be rejected.
                let mut fabricated = edit.clone();
                fabricated.pag = Some(gen_core::PagExecutionReport {
                    layers: vec![20],
                    active_steps: 50,
                    total_steps: 50,
                    scale: 2.0,
                    adaptive_scale: 0.0,
                    start: 0.0,
                    stop: 1.0,
                });
                assert!(completed_identity(None, Some(&fabricated), &descriptor("mlx")).is_err());

                // Active PAG on edit base + edit receipt propagates into quality_policy.pag.
                let mut active_base = edit.base.clone().into_data();
                let pag_resolved = gen_core::PagResolved {
                    scale: 2.5,
                    adaptive_scale: 0.5,
                    layers: vec![20],
                    start: 0.0,
                    stop: 0.5,
                };
                let pag_exec = pag_resolved.execution_report(50);
                active_base.resolved.pag = Some(pag_resolved);
                active_base.pag = Some(pag_exec.clone());
                let mut active_edit = edit;
                active_edit.base = MossImageExecutionReport::new(active_base).unwrap();
                active_edit.pag = Some(pag_exec);
                let active_shared =
                    completed_identity(None, Some(&active_edit), &descriptor("mlx")).unwrap();
                let ExecutionFact::Resolved { value: quality } =
                    &active_shared.resolved().unwrap().quality_policy
                else {
                    panic!("quality facts")
                };
                assert_eq!(quality["pag"]["layers"], json!([20]));
                assert_eq!(quality["pag"]["active_steps"], 25);
                assert_eq!(quality["pag"]["total_steps"], 50);
            }
        }
    }

    #[test]
    fn custom_stack_retains_installed_scale_and_order_without_paths() {
        let report = receipt(MossImageAcceleration::Off, MossImageExecutionDevice::Mlx);
        let mut shared = completed_identity(Some(&report), None, &descriptor("mlx")).unwrap();
        let specs = [
            gen_core::AdapterSpec::new("/private/first".into(), 0.25, gen_core::AdapterKind::Lora),
            gen_core::AdapterSpec::new("/private/second".into(), -1.0, gen_core::AdapterKind::Lokr),
        ];
        let installed: Vec<_> = specs
            .iter()
            .map(|spec| gen_core::AdapterApplyReport {
                adapter_path: spec.path.clone(),
                applied: 42,
                skipped: vec![],
            })
            .collect();
        assert!(record_custom_adapter_policy(&mut shared, &specs, &[]).is_err());
        record_custom_adapter_policy(&mut shared, &specs, &installed).unwrap();
        let ExecutionFact::Resolved { value } =
            &shared.resolved().unwrap().provider_choices["custom_adapter_stack"]
        else {
            panic!("stack facts")
        };
        assert_eq!(
            value[0]["uniform_scale"],
            serde_json::to_value(ExecutionFact::f32_bits(0.25).unwrap()).unwrap()
        );
        assert_eq!(value[1]["kind"], "lokr");
        assert_eq!(value[1]["ordinal"], 1);
        assert_eq!(value[1]["applied"], 42);
        assert!(!serde_json::to_string(&shared)
            .unwrap()
            .contains("/private/"));
    }

    #[test]
    fn only_missing_sealed_receipts_report_unavailable() {
        let report = receipt(MossImageAcceleration::Off, MossImageExecutionDevice::Mlx);
        assert!(completed_identity(Some(&report), None, &descriptor("candle")).is_err());
        let mut wrong_variant = descriptor("mlx");
        wrong_variant.id = "moss_image_base";
        assert!(completed_identity(Some(&report), None, &wrong_variant).is_err());
        let shared = completed_identity(None, None, &descriptor("mlx")).unwrap();
        let ExecutionIdentity::Unavailable { reason } = shared else {
            panic!("unsealed")
        };
        assert_eq!(reason.code, "moss_unsealed_source_receipt");
    }
}
