//! Shared Perturbed-Attention Guidance (PAG) contract re-exports (SW-7173).
//!
//! Portable request, capability, step-window resolution, and execution report types live in
//! [`gen_core_pag_policy`] so native engines and `wasm32` validators consume one implementation.

pub use gen_core_pag_policy::{
    resolve_pag, PagCapability, PagConfig, PagExecutionReport, PagPolicyError, PagPolicyErrorKind,
    PagResolved, DEFAULT_PAG_MAX_SCALE,
};

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::BTreeMap;

    use super::*;
    use crate::moss_image::{
        moss_image_pag_capability, moss_image_schedule, MossDmdPrecision, MossImageAcceleration,
        MossImageAssetReceipt, MossImageAssetRole, MossImageConditioningInputReceipt,
        MossImageConditioningReceipt, MossImageConditioningSource, MossImageDtype,
        MossImageExecutionData, MossImageExecutionDevice, MossImageExecutionReport,
        MossImageOptions, MossImageOutputKind, MossImageResolved, MossImageSampleReceipt,
        MossImageSamplingReceipt, MossImageVariant, MossSkipLayerGuidance, DMD_REPO, DMD_REVISION,
        SFT_REPO, SFT_REVISION, SNAPSHOT_FILES,
    };
    use crate::moss_image_edit::{
        MossImageEditExecutionReport, MossImageEditReferenceReceipt, EDIT_CONFIG_SHA256,
        EDIT_CONTRACT, EDIT_REPOSITORY, EDIT_REVISION, EDIT_WEIGHTS_SHA256,
    };
    use crate::registry::model_descriptor_errors;
    use crate::{
        Capabilities, Error, GenerationOutput, GenerationReport, GenerationRequest, Image,
        Modality, ModelDescriptor,
    };

    const HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn pag_caps() -> Capabilities {
        Capabilities {
            min_size: 256,
            max_size: 2048,
            max_count: 4,
            pag: Some(PagCapability {
                num_blocks: 38,
                block_domain: Cow::Borrowed("single_blocks"),
                default_layers: vec![19],
                default_start: 0.0,
                default_stop: 1.0,
                recommended_scale: 2.5,
                max_scale: DEFAULT_PAG_MAX_SCALE,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn request_floor_gates_model_support_finiteness_bounds_and_block_indices() {
        let unsupported = Capabilities {
            min_size: 256,
            max_size: 2048,
            max_count: 4,
            ..Default::default()
        };
        assert!(unsupported
            .validate_request("flux1_schnell", &GenerationRequest::default())
            .is_ok());

        let err = unsupported
            .validate_request(
                "flux1_schnell",
                &GenerationRequest {
                    features: crate::FeatureParams {
                        pag: Some(PagConfig {
                            scale: 2.0,
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(
            matches!(&err, Error::Unsupported(msg) if msg.contains("Perturbed-Attention Guidance (PAG)")),
            "unexpected error: {err:?}"
        );

        let caps = pag_caps();
        // Scale 0.0 is a strict zero-pass no-op.
        let noop_req = GenerationRequest {
            features: crate::FeatureParams {
                pag: Some(PagConfig {
                    scale: 0.0,
                    layers: Some(vec![0, 37]),
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        caps.validate_request("z_image", &noop_req).unwrap();
        assert_eq!(caps.resolve_pag("z_image", &noop_req).unwrap(), None);

        // Out-of-range block index is refused even when scale == 0.0.
        let bad_zero = GenerationRequest {
            features: crate::FeatureParams {
                pag: Some(PagConfig {
                    scale: 0.0,
                    layers: Some(vec![38]),
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(caps.validate_request("z_image", &bad_zero).is_err());

        // Non-finite floats are caught by both first_nonfinite_float and validate_request.
        for (expected_field, pag) in [
            (
                "pag.scale",
                PagConfig {
                    scale: f32::NAN,
                    ..Default::default()
                },
            ),
            (
                "pag.start",
                PagConfig {
                    scale: 2.0,
                    start: Some(f32::INFINITY),
                    ..Default::default()
                },
            ),
            (
                "pag.stop",
                PagConfig {
                    scale: 2.0,
                    stop: Some(f32::NEG_INFINITY),
                    ..Default::default()
                },
            ),
            (
                "pag.adaptive_scale",
                PagConfig {
                    scale: 2.0,
                    adaptive_scale: Some(f32::NAN),
                    ..Default::default()
                },
            ),
        ] {
            let req = GenerationRequest {
                features: crate::FeatureParams {
                    pag: Some(pag),
                    ..Default::default()
                },
                ..Default::default()
            };
            assert_eq!(
                req.first_nonfinite_float().map(|(f, _)| f),
                Some(expected_field)
            );
            assert!(req.ensure_finite_floats().is_err());
            assert!(caps.validate_request("z_image", &req).is_err());
        }

        // Range and structural rejections.
        for bad in [
            PagConfig {
                scale: -1.0,
                ..Default::default()
            },
            PagConfig {
                scale: 16.0,
                ..Default::default()
            },
            PagConfig {
                scale: 2.0,
                adaptive_scale: Some(-0.5),
                ..Default::default()
            },
            PagConfig {
                scale: 2.0,
                start: Some(-0.1),
                ..Default::default()
            },
            PagConfig {
                scale: 2.0,
                stop: Some(1.1),
                ..Default::default()
            },
            PagConfig {
                scale: 2.0,
                start: Some(0.8),
                stop: Some(0.3),
                ..Default::default()
            },
            PagConfig {
                scale: 2.0,
                layers: Some(vec![]),
                ..Default::default()
            },
            PagConfig {
                scale: 2.0,
                layers: Some(vec![19, 38]),
                ..Default::default()
            },
        ] {
            let req = GenerationRequest {
                features: crate::FeatureParams {
                    pag: Some(bad.clone()),
                    ..Default::default()
                },
                ..Default::default()
            };
            assert!(
                matches!(caps.validate_request("z_image", &req), Err(Error::Msg(_))),
                "expected Msg rejection for {bad:?}"
            );
        }

        // Valid PAG resolves sorted/deduped layers and step window.
        let active_req = GenerationRequest {
            features: crate::FeatureParams {
                pag: Some(PagConfig {
                    scale: 3.0,
                    layers: Some(vec![30, 5, 30]),
                    start: Some(0.25),
                    stop: Some(0.75),
                    adaptive_scale: Some(1.0),
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        caps.validate_request("z_image", &active_req).unwrap();
        let resolved = caps
            .resolve_pag("z_image", &active_req)
            .unwrap()
            .expect("active PAG");
        assert_eq!(resolved.layers, vec![5, 30]);
        assert_eq!(resolved.active_steps(8), 4);
    }

    #[test]
    fn moss_image_pag_is_gated_validated_and_resolved_by_shared_request_floor() {
        let moss_req = |pag: Option<PagConfig>, moss_pag: Option<PagConfig>| GenerationRequest {
            features: crate::FeatureParams {
                pag,
                ..Default::default()
            },
            extensions: crate::FamilyExtensions {
                moss_image: Some(crate::MossImageExtension::from_options(MossImageOptions {
                    pag: moss_pag,
                    ..Default::default()
                })),
                ..Default::default()
            },
            ..Default::default()
        };
        // (a) moss_image.pag cannot bypass Capabilities::pag == None on MOSS or non-MOSS models.
        let unsupported_moss = Capabilities {
            min_size: 256,
            max_size: 2048,
            max_count: 4,
            pag: None,
            supported_family_extensions: vec![crate::FamilyExtensionKind::MossImage],
            ..Default::default()
        };
        let non_moss = Capabilities {
            min_size: 256,
            max_size: 2048,
            max_count: 4,
            pag: None,
            ..Default::default()
        };
        // Baseline: MOSS controls without PAG pass on moss_image_sft when Capabilities::pag == None.
        let moss_without_pag = moss_req(None, None);
        unsupported_moss
            .validate_request("moss_image_sft", &moss_without_pag)
            .unwrap();
        assert_eq!(
            unsupported_moss
                .resolve_pag("moss_image_sft", &moss_without_pag)
                .unwrap(),
            None
        );

        for scale in [2.0_f32, 0.0_f32] {
            let bypass_req = moss_req(
                None,
                Some(PagConfig {
                    scale,
                    ..Default::default()
                }),
            );
            assert!(matches!(
                unsupported_moss.validate_request("moss_image_sft", &bypass_req),
                Err(Error::Unsupported(msg)) if msg.contains("Perturbed-Attention Guidance (PAG)")
            ));
            assert!(matches!(
                unsupported_moss.resolve_pag("moss_image_sft", &bypass_req),
                Err(Error::Unsupported(msg)) if msg.contains("Perturbed-Attention Guidance (PAG)")
            ));
            assert!(matches!(
                non_moss.validate_request("flux1_schnell", &bypass_req),
                Err(Error::Unsupported(_))
            ));
        }

        // (b) Invalid moss_image.pag is rejected by Capabilities::validate_request and resolve_pag
        // when Capabilities::pag == Some(moss_image_pag_capability()).
        let moss_caps = Capabilities {
            min_size: 256,
            max_size: 2048,
            max_count: 4,
            pag: Some(moss_image_pag_capability()),
            supported_family_extensions: vec![crate::FamilyExtensionKind::MossImage],
            ..Default::default()
        };
        for bad in [
            PagConfig {
                scale: 2.0,
                layers: Some(vec![42]),
                ..Default::default()
            },
            PagConfig {
                scale: 0.0,
                layers: Some(vec![42]),
                ..Default::default()
            },
            PagConfig {
                scale: -1.0,
                ..Default::default()
            },
            PagConfig {
                scale: 2.0,
                start: Some(0.8),
                stop: Some(0.2),
                ..Default::default()
            },
            PagConfig {
                scale: f32::NAN,
                ..Default::default()
            },
        ] {
            let req = moss_req(None, Some(bad.clone()));
            assert!(
                matches!(
                    moss_caps.validate_request("moss_image_sft", &req),
                    Err(Error::Msg(_))
                ),
                "expected validate_request Msg rejection for moss_image.pag={bad:?}"
            );
            assert!(
                matches!(
                    moss_caps.resolve_pag("moss_image_sft", &req),
                    Err(Error::Msg(_))
                ),
                "expected resolve_pag Msg rejection for moss_image.pag={bad:?}"
            );
        }

        // (c) Conflicting req.pag vs moss_image.pag is rejected by effective_pag, validate_request,
        // and resolve_pag; identical values succeed.
        let conflict_req = moss_req(
            Some(PagConfig {
                scale: 2.0,
                ..Default::default()
            }),
            Some(PagConfig {
                scale: 3.0,
                ..Default::default()
            }),
        );
        assert!(matches!(
            conflict_req.effective_pag(),
            Err(Error::Msg(msg)) if msg.contains("conflicting top-level pag and moss_image.pag")
        ));
        assert!(matches!(
            moss_caps.validate_request("moss_image_sft", &conflict_req),
            Err(Error::Msg(msg)) if msg.contains("conflicting top-level pag and moss_image.pag")
        ));
        assert!(matches!(
            moss_caps.resolve_pag("moss_image_sft", &conflict_req),
            Err(Error::Msg(msg)) if msg.contains("conflicting top-level pag and moss_image.pag")
        ));

        let matching_pag = PagConfig {
            scale: 2.5,
            layers: Some(vec![20]),
            ..Default::default()
        };
        let equal_req = moss_req(Some(matching_pag.clone()), Some(matching_pag));
        moss_caps
            .validate_request("moss_image_sft", &equal_req)
            .unwrap();
        assert_eq!(
            moss_caps
                .resolve_pag("moss_image_sft", &equal_req)
                .unwrap()
                .unwrap()
                .scale,
            2.5
        );

        // (d) Valid moss_image.pag (with req.pag == None) is validated and resolved by resolve_pag.
        let moss_active_req = moss_req(
            None,
            Some(PagConfig {
                scale: 2.5,
                layers: Some(vec![41, 20, 41]),
                start: Some(0.25),
                stop: Some(0.75),
                adaptive_scale: Some(0.5),
            }),
        );
        moss_caps
            .validate_request("moss_image_sft", &moss_active_req)
            .unwrap();
        let resolved_moss = moss_caps
            .resolve_pag("moss_image_sft", &moss_active_req)
            .unwrap()
            .expect("active moss_image.pag");
        assert_eq!(
            resolved_moss,
            PagResolved {
                scale: 2.5,
                adaptive_scale: 0.5,
                layers: vec![20, 41],
                start: 0.25,
                stop: 0.75,
            }
        );

        let moss_noop_req = moss_req(
            None,
            Some(PagConfig {
                scale: 0.0,
                layers: Some(vec![20]),
                ..Default::default()
            }),
        );
        moss_caps
            .validate_request("moss_image_sft", &moss_noop_req)
            .unwrap();
        assert_eq!(
            moss_caps
                .resolve_pag("moss_image_sft", &moss_noop_req)
                .unwrap(),
            None
        );
    }

    #[test]
    fn descriptor_conformance_validates_pag_capability_definitions() {
        let mut desc = ModelDescriptor {
            id: "pag_test",
            family: "test",
            backend: "candle",
            modality: Modality::Image,
            capabilities: pag_caps(),
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
        };
        assert!(model_descriptor_errors(&desc).is_empty());

        desc.capabilities.pag.as_mut().unwrap().default_layers = vec![38];
        assert!(!model_descriptor_errors(&desc).is_empty());
    }

    #[test]
    fn generation_and_moss_execution_reports_carry_pag_execution_report() {
        let img = Image {
            width: 16,
            height: 16,
            pixels: vec![0; 16 * 16 * 3],
        };
        let mut gen_report = GenerationReport::from_output(GenerationOutput::Images(vec![img]));
        assert!(gen_report.pag.is_none());
        let pag_exec = PagExecutionReport {
            layers: vec![20],
            active_steps: 6,
            total_steps: 8,
            scale: 2.5,
            adaptive_scale: 0.5,
            start: 0.0,
            stop: 0.75,
        };
        gen_report.pag = Some(pag_exec.clone());
        assert_eq!(gen_report.pag.as_ref(), Some(&pag_exec));

        // MOSS DMD + PAG receipt validation and round-trip.
        let dmd_req = GenerationRequest {
            features: crate::FeatureParams {
                pag: Some(PagConfig {
                    scale: 2.5,
                    layers: None,
                    start: Some(0.0),
                    stop: Some(0.75),
                    adaptive_scale: Some(0.5),
                }),
                ..Default::default()
            },
            extensions: crate::FamilyExtensions {
                moss_image: Some(crate::MossImageExtension::from_options(MossImageOptions {
                    acceleration: MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16),
                    ..Default::default()
                })),
                ..Default::default()
            },
            ..Default::default()
        };
        let resolved_dmd = MossImageResolved::resolve(MossImageVariant::Sft, &dmd_req).unwrap();
        assert_eq!(
            resolved_dmd.pag.as_ref().unwrap().execution_report(8),
            pag_exec
        );

        let mut dmd_assets: Vec<_> = SNAPSHOT_FILES
            .iter()
            .copied()
            .filter(|p| *p != "LICENSE")
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
                MossImageAssetReceipt {
                    role,
                    repository: SFT_REPO.into(),
                    revision: SFT_REVISION.into(),
                    relative_path: p.into(),
                    sha256: HASH.into(),
                }
            })
            .collect();
        for (role, path) in [
            (MossImageAssetRole::DmdLora, "lora.safetensors"),
            (MossImageAssetRole::DmdTime, "time_embedding.safetensors"),
            (MossImageAssetRole::DmdDataspace, "dataspace.safetensors"),
        ] {
            dmd_assets.push(MossImageAssetReceipt {
                role,
                repository: DMD_REPO.into(),
                revision: DMD_REVISION.into(),
                relative_path: path.into(),
                sha256: HASH.into(),
            });
        }
        let cond_input = MossImageConditioningInputReceipt {
            source: MossImageConditioningSource::Text,
            sha256: HASH.into(),
            sequence_length: 8,
            valid_tokens: 8,
        };
        let dmd_dtypes: BTreeMap<String, MossImageDtype> = [
            ("transformer", MossImageDtype::Bf16),
            ("text_encoder", MossImageDtype::Bf16),
            ("vae", MossImageDtype::F32),
            ("latents", MossImageDtype::F32),
            ("time_embedding", MossImageDtype::F32),
            ("shared_block_modulation", MossImageDtype::F32),
            ("video_head_modulation", MossImageDtype::F32),
            ("lora_residual", MossImageDtype::F32),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v))
        .collect();

        let dmd_data = MossImageExecutionData {
            execution_device: MossImageExecutionDevice::Cuda,
            selected_stream_device: None,
            resolved: resolved_dmd,
            assets: dmd_assets,
            sampling: MossImageSamplingReceipt::Dmd {
                step_codes: vec![0.5; 8],
                mix_matrix_sha256: HASH.into(),
            },
            samples: vec![MossImageSampleReceipt {
                seed: Some(42),
                noise_sha256: HASH.into(),
            }],
            conditioning: vec![MossImageConditioningReceipt {
                positive: cond_input.clone(),
                negative: None,
            }],
            images_per_prompt: 1,
            latent_shape: [1, 128, 64, 64],
            output_kind: MossImageOutputKind::Images,
            component_dtypes: dmd_dtypes,
            adapter_reports: vec![crate::AdapterApplyReport {
                adapter_path: "/fixture/lora.safetensors".into(),
                applied: 252,
                skipped: vec![],
            }],
            pag: Some(pag_exec.clone()),
        };
        let dmd_report = MossImageExecutionReport::new(dmd_data.clone()).unwrap();
        assert_eq!(dmd_report.pag(), Some(&pag_exec));
        let json = serde_json::to_value(&dmd_report).unwrap();
        assert_eq!(
            serde_json::from_value::<MossImageExecutionReport>(json).unwrap(),
            dmd_report
        );

        // Missing PAG report when resolved.pag is Some is refused.
        let mut missing_pag = dmd_data;
        missing_pag.pag = None;
        assert!(MossImageExecutionReport::new(missing_pag).is_err());

        // MOSS Edit + PAG receipt validation and round-trip (SLG disallowed, PAG allowed).
        let edit_req = GenerationRequest {
            steps: Some(30),
            guidance: Some(4.0),
            features: crate::FeatureParams {
                pag: Some(PagConfig {
                    scale: 2.0,
                    layers: Some(vec![20]),
                    ..Default::default()
                }),
                ..Default::default()
            },
            extensions: crate::FamilyExtensions {
                moss_image: Some(crate::MossImageExtension::from_options(MossImageOptions {
                    uncond_skip_layers: Some(vec![]),
                    ..Default::default()
                })),
                ..Default::default()
            },
            ..Default::default()
        };
        let resolved_edit = MossImageResolved::resolve(MossImageVariant::Sft, &edit_req).unwrap();
        let edit_sigmas = moss_image_schedule(
            resolved_edit.steps,
            resolved_edit.shift,
            resolved_edit.training_shift,
        )
        .unwrap();
        let edit_steps = (edit_sigmas.len() - 1) as u32;
        let edit_pag = resolved_edit
            .pag
            .as_ref()
            .unwrap()
            .execution_report(edit_steps);
        let ordinary_assets: Vec<_> = SNAPSHOT_FILES
            .iter()
            .copied()
            .filter(|p| *p != "LICENSE")
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
                MossImageAssetReceipt {
                    role,
                    repository: SFT_REPO.into(),
                    revision: SFT_REVISION.into(),
                    relative_path: p.into(),
                    sha256: HASH.into(),
                }
            })
            .collect();
        let ordinary_dtypes: BTreeMap<String, MossImageDtype> = [
            ("transformer", MossImageDtype::Bf16),
            ("text_encoder", MossImageDtype::Bf16),
            ("vae", MossImageDtype::F32),
            ("latents", MossImageDtype::F32),
            ("time_embedding", MossImageDtype::Bf16),
            ("shared_block_modulation", MossImageDtype::Bf16),
            ("video_head_modulation", MossImageDtype::Bf16),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v))
        .collect();
        let base_edit_report = MossImageExecutionReport::new(MossImageExecutionData {
            execution_device: MossImageExecutionDevice::Cuda,
            selected_stream_device: None,
            resolved: resolved_edit,
            assets: ordinary_assets,
            sampling: MossImageSamplingReceipt::Euler {
                sigmas: edit_sigmas,
            },
            samples: vec![MossImageSampleReceipt {
                seed: Some(7),
                noise_sha256: HASH.into(),
            }],
            conditioning: vec![MossImageConditioningReceipt {
                positive: cond_input.clone(),
                negative: Some(cond_input),
            }],
            images_per_prompt: 1,
            latent_shape: [1, 128, 64, 64],
            output_kind: MossImageOutputKind::Images,
            component_dtypes: ordinary_dtypes,
            adapter_reports: vec![],
            pag: Some(edit_pag.clone()),
        })
        .unwrap();
        let edit_report = MossImageEditExecutionReport {
            base: base_edit_report,
            adapter_path: "/fixture/moss_edit.safetensors".into(),
            edit_contract: EDIT_CONTRACT.into(),
            adapter_repository: EDIT_REPOSITORY.into(),
            adapter_revision: EDIT_REVISION.into(),
            adapter_sha256: EDIT_WEIGHTS_SHA256.into(),
            config_sha256: EDIT_CONFIG_SHA256.into(),
            applied_projections: 252,
            references: vec![MossImageEditReferenceReceipt {
                original_dimensions: [512, 512],
                encoded_dimensions: [512, 512],
                pixels_sha256: HASH.into(),
                resized_pixels_sha256: HASH.into(),
                latent_sha256: HASH.into(),
            }],
            reference_max_edge: 512,
            use_kv_cache: true,
            pag: Some(edit_pag.clone()),
        };
        edit_report.validate().unwrap();
        let edit_json = serde_json::to_value(&edit_report).unwrap();
        assert_eq!(
            serde_json::from_value::<MossImageEditExecutionReport>(edit_json).unwrap(),
            edit_report
        );

        // Verify MOSS SLG serde error string remains frozen alongside PAG.
        let slg_err = serde_json::from_value::<MossSkipLayerGuidance>(
            serde_json::json!({"layers": [42], "scale": 1.0, "start": 0.0, "stop": 1.0}),
        )
        .unwrap_err();
        assert_eq!(slg_err.to_string(), "invalid MOSS SLG controls");
        assert_eq!(moss_image_pag_capability().num_blocks, 42);
        assert_eq!(moss_image_pag_capability().default_layers, vec![20]);
    }
}
