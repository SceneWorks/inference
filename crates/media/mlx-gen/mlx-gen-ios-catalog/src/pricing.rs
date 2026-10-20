//! The providers' counted iOS pricing and quantum-phase attribution, by provider id: the
//! architecture facts the anchor law ([`crate::anchor`]) scales a measured anchor by.
//!
//! Every registered iOS provider has an entry (`every_registered_provider_is_priced`). A provider
//! without one would still be admitted at its anchors' OWN keys (an exact measurement needs no
//! scaling), but nothing of it would be derivable: every other configuration refuses `no_evidence`.

use mlx_gen::ios_pricing::{IosPhase, IosPricer, IosQuantumPhase};

type Pricing = (IosPricer, IosQuantumPhase);

#[cfg(test)]
thread_local! {
    /// `Some(p)`: every provider priced by `p` (or by nothing, for `Some(None)`).
    static TEST_PRICING: std::cell::Cell<Option<Option<Pricing>>> =
        const { std::cell::Cell::new(None) };
}

/// Run `body` with every provider priced by `pricing` — `None`: no provider priced (crate tests
/// only).
#[cfg(test)]
pub(crate) fn with_test_pricing<T>(pricing: Option<Pricing>, body: impl FnOnce() -> T) -> T {
    struct Restore(Option<Option<Pricing>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_PRICING.with(|slot| slot.set(self.0));
        }
    }
    let _restore = Restore(TEST_PRICING.with(|slot| slot.replace(Some(pricing))));
    body()
}

/// `(pricing, quantum-label → phase)` of a registered iOS provider.
pub fn provider_pricing(provider: &str) -> Option<Pricing> {
    #[cfg(test)]
    if let Some(pricing) = TEST_PRICING.with(std::cell::Cell::get) {
        return pricing;
    }
    match provider {
        "sana_1600m" | "sana_sprint_1600m" => Some((
            mlx_gen_sana::ios_pricing::ios_pricing,
            mlx_gen_sana::ios_pricing::ios_quantum_phase,
        )),
        crate::anima::ANIMA_TURBO_ID => Some((
            mlx_gen_anima::ios_pricing::ios_pricing,
            mlx_gen_anima::ios_pricing::ios_quantum_phase,
        )),
        "iris_3b" | crate::ios_admission::IRIS_RESTORE_ID | crate::ios_admission::IRIS_DEPTH_ID => {
            Some((
                mlx_gen_iris::ios_pricing::ios_pricing,
                mlx_gen_iris::ios_pricing::ios_quantum_phase,
            ))
        }
        crate::sdxl::SDXL_ID => Some((
            mlx_gen_sdxl::ios_pricing::ios_pricing,
            mlx_gen_sdxl::ios_pricing::ios_quantum_phase,
        )),
        id if crate::zimage::is_z_image(id) => Some((
            mlx_gen_z_image::ios_pricing::ios_pricing,
            mlx_gen_z_image::ios_pricing::ios_quantum_phase,
        )),
        _ => None,
    }
}

/// Accepted component-additivity domains `(provider, resolved allocation class, tier, mode)`. A
/// factory/input graph name alone is NOT a whole-phase residue or mode-additivity proof. No
/// corrected literal-endpoint sampling domain has completed that proof in sc-26270 yet.
const PROVEN_ADDITIVE_DOMAINS: &[(&str, &str, &str, &str)] = &[];

/// Conservative compatibility when no sampling context is available.
pub fn anchor_variant(provider: &str, mode: &str, tier: &str) -> String {
    anchor_variant_for_sampling(provider, mode, tier, None)
}

/// Collapse a mode into its text family only in an accepted per-class matched-epoch/whole-phase
/// proof domain. Unproved modes stay expressible as distinct two-endpoint families.
pub fn anchor_variant_for_sampling(
    provider: &str,
    mode: &str,
    tier: &str,
    class: Option<&str>,
) -> String {
    let proved =
        class.is_some_and(|class| PROVEN_ADDITIVE_DOMAINS.contains(&(provider, class, tier, mode)));
    if mode == "text_to_image" || proved {
        "base".into()
    } else {
        mode.into()
    }
}

/// Recover TRUE Reference roles from the existing AGGREGATE image-carrier geometry, only in
/// the current validated Reference/Mask/Control domain. This is integer role partitioning, not
/// memory-credit subtraction. Other carrier kinds require their own source classification.
pub(crate) fn reference_role_count(
    aggregate: u32,
    masks: u32,
    controls: u32,
) -> mlx_gen::gen_core::Result<u32> {
    let non_references = masks.checked_add(controls).ok_or_else(|| {
        mlx_gen::gen_core::Error::Unsupported(
            "conditioning role counts overflow aggregate geometry".into(),
        )
    })?;
    aggregate.checked_sub(non_references).ok_or_else(|| {
        mlx_gen::gen_core::Error::Unsupported(
            "conditioning role counts exceed aggregate geometry".into(),
        )
    })
}

/// Source role graph shared by evidence and campaign grouping. `mode` is the actual validated
/// work_mode (including configured IP selection), not a caller's preferred label. Counts remain
/// exact in binding/pricing; they are NOT measurement-grid axes. Current provider source scope:
/// - SDXL model::generate resolves its single reference BEFORE uses_ip_reference selects IP;
///   model::resolve_reference/resolve_mask reject a second init/mask, including the existing IP
///   route. This does NOT establish a universal single-image IP capability rule.
/// - SANA model::resolve_reference accepts one init; Z generation/control use the shared
///   mlx_gen::img2img::resolve_reference single-init producer.
/// - Anima/Iris generation descriptors leave conditioning empty; restoration/depth are separate
///   typed tasks, not fabricated generation references.
/// Future/unknown multi-reference mechanisms remain unclassified/open, not declared unsupported.
/// MultiControlNet's actual branch topology is separately sealed by the configured overlay digest
/// (SDXL model::resolve_control + generate's loaded-count validation), so cardinality must not be
/// guessed additive from this presence partition. No transfer/allocation/physical proof is granted.
pub fn anchor_variant_for_roles(
    provider: &str,
    mode: &str,
    tier: &str,
    class: Option<&str>,
    aggregate_references: u32,
    masks: u32,
    controls: u32,
) -> Result<String, String> {
    let references = reference_role_count(aggregate_references, masks, controls)
        .map_err(|error| error.to_string())?;
    if crate::zimage::is_z_image(provider) {
        mlx_gen_z_image::ios_pricing::validate_pricing_roles(
            provider,
            aggregate_references,
            references,
            masks,
            controls,
        )
        .map_err(|error| error.to_string())?;
    }
    if references > 1 || masks > 1 {
        return Err("anchor role graph needs an explicit source classification for multiple init references/masks".into());
    }
    if matches!(mode, "restoration" | "depth") {
        if references != 0 || masks != 0 || controls != 0 {
            return Err("task anchor must not invent generation conditioning roles".into());
        }
        return Ok(mode.into());
    }
    let role_mode = if controls != 0 {
        "control"
    } else if masks != 0 {
        "edit"
    } else if mode == "ip_adapter" {
        "ip_adapter"
    } else if references != 0 {
        "image_to_image"
    } else {
        // Preserve unknown task modes, rather than relabeling them as empty text generation.
        match mode {
            "text" | "text_to_image" | "prompt_batch" => "text_to_image",
            "reference" | "image_to_image" | "mask" | "edit" | "control" => {
                return Err("anchor mode has no matching actual conditioning role".into());
            }
            other => other,
        }
    };
    if role_mode == "ip_adapter" && references == 0 {
        return Err("IP anchor has no actual image reference".into());
    }
    let variant = anchor_variant_for_sampling(provider, role_mode, tier, class);
    if controls == 0 && masks == 0 {
        return Ok(variant);
    }
    let branches = if controls > 1 {
        "multi"
    } else if controls == 1 {
        "single"
    } else {
        "none"
    };
    Ok(format!(
        "{variant}/roles-{branches}-init{}-mask{}",
        u8::from(references != 0),
        u8::from(masks != 0)
    ))
}

/// Without a class-specific proof, these non-text modes must not inherit text evidence.
pub fn non_additive_modes(provider: &str, tier: &str) -> Vec<&'static str> {
    [
        "image_to_image",
        "edit",
        "control",
        "ip_adapter",
        "restoration",
        "depth",
    ]
    .into_iter()
    .filter(|mode| anchor_variant(provider, mode, tier) != "base")
    .collect()
}

/// The phase a quantum label of `provider` runs in.
pub fn quantum_phase(provider: &str, label: &str) -> Option<IosPhase> {
    provider_pricing(provider).and_then(|(_, phase)| phase(label))
}

#[cfg(test)]
mod tests {
    /// Public live/replay guard protocol only. Empty metadata directories stop Facts construction
    /// BEFORE Model/Array construction, even when a role guard is deliberately mutated later.
    /// Reaching that exact reader error is not a successful numeric price or native memory proof.
    fn public_role_boundaries_without_tensors() {
        use mlx_gen::gen_core::{
            GenerationMemory, MemoryGeometry, MemoryMode, MemoryNumericTier, MemorySelection,
            MemoryStrategy,
        };
        use mlx_gen::ios_pricing::IosPricingRequest;
        use mlx_gen::{Conditioning, ControlKind, GenerationRequest, Image, WeightsSource};
        let image = || Image {
            width: 32,
            height: 32,
            pixels: vec![0; 32 * 32 * 3],
        };
        let scratch = tempfile::tempdir().unwrap();
        let root = scratch.path().join("empty-model");
        let control = scratch.path().join("empty-control");
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir(root.join("transformer")).unwrap();
        std::fs::create_dir(&control).unwrap();
        let control_source = WeightsSource::Dir(control);
        let z_spec = crate::zimage::z_image_turbo_load_spec(
            WeightsSource::Dir(root.clone()),
            Some(control_source.clone()),
        );
        let execution = GenerationMemory::default();
        let selection = MemorySelection {
            strategy: MemoryStrategy::Resident,
            parameters: Default::default(),
            tier: MemoryNumericTier {
                precision: z_spec.precision,
                quant: z_spec.quantize,
                component_precision_floors: &[],
            },
        };
        let mode = MemoryMode::Other("control".into());
        let provider = crate::zimage::Z_IMAGE_TURBO_CONTROL_ID;
        let reader_error = format!("iOS pricing: no .safetensors under {control_source:?}");
        let role_error = format!("{provider}: invalid Z-Image pricing conditioning-role partition");
        let mut families = Vec::new();
        for has_reference in [false, true] {
            let mut carrier = GenerationRequest {
                prompt: "role boundary".into(),
                conditioning: vec![Conditioning::Control {
                    image: image(),
                    kind: ControlKind::Pose,
                    scale: Some(0.75),
                }],
                ..Default::default()
            };
            if has_reference {
                carrier.conditioning.push(Conditioning::Reference {
                    image: image(),
                    strength: Some(0.6),
                });
            }
            let refs = u32::from(has_reference);
            let aggregate = carrier.memory_reference_count();
            assert_eq!(aggregate, 1 + refs);
            let actual_mode = crate::work_mode(
                provider,
                &z_spec,
                &crate::IosWork::Generate(carrier.clone()),
            );
            assert!(actual_mode.is_ok(), "valid actual Z carrier must classify");
            assert_eq!(actual_mode.unwrap().as_key(), "control");
            let family =
                super::anchor_variant_for_roles(provider, "control", "bf16", None, aggregate, 0, 1);
            assert!(family.is_ok(), "valid explicit Z role family must classify");
            families.push(family.unwrap());
            let sampling = crate::sampling::resolve(provider, &carrier).unwrap();
            let live = IosPricingRequest {
                provider,
                spec: &z_spec,
                mode: &mode,
                geometry: MemoryGeometry {
                    width: carrier.width,
                    height: carrier.height,
                    batch: 1,
                    frames: 1,
                    reference_count: aggregate,
                },
                selection: &selection,
                execution: &execution,
                generation: Some(&carrier),
                transform: None,
                depth: None,
                token_window: 0,
            };
            let replay = IosPricingRequest {
                generation: None,
                ..live
            };
            assert!(matches!(mlx_gen_z_image::ios_pricing::ios_pricing(&live), Err(mlx_gen::Error::Msg(reason)) if reason == reader_error),
                "valid live Z role partition must reach the metadata reader, not fabricate a numeric price");
            assert!(
                matches!(mlx_gen_z_image::ios_pricing::ios_pricing_with_token(&replay, sampling.token(), 1, refs, 0, 1), Err(mlx_gen::Error::Msg(reason)) if reason == reader_error),
                "valid scalar Z token roles must reach the same metadata reader"
            );
            assert!(
                matches!(crate::sampling::price_metadata(&replay, sampling.token(), 0, 1), Err(mlx_gen::Error::Msg(reason)) if reason == reader_error),
                "catalog replay must pass true References, not aggregate image count"
            );
            let bad_live = IosPricingRequest {
                geometry: MemoryGeometry {
                    reference_count: aggregate + 1,
                    ..live.geometry
                },
                ..live
            };
            let bad_replay = IosPricingRequest {
                generation: None,
                ..bad_live
            };
            assert!(
                matches!(mlx_gen_z_image::ios_pricing::ios_pricing(&bad_live), Err(mlx_gen::Error::Unsupported(reason)) if reason == role_error),
                "live Z aggregate mismatch must fail before metadata/model work"
            );
            assert!(
                matches!(mlx_gen_z_image::ios_pricing::ios_pricing_with_token(&bad_replay, sampling.token(), 1, refs, 0, 1), Err(mlx_gen::Error::Unsupported(reason)) if reason == role_error),
                "token Z aggregate mismatch must use the same role guard"
            );
        }
        assert_ne!(
            families[0], families[1],
            "Control-only and Control+Reference are distinct actual families"
        );
        let duplicate = GenerationRequest {
            prompt: "duplicate control".into(),
            conditioning: vec![
                Conditioning::Control {
                    image: image(),
                    kind: ControlKind::Pose,
                    scale: Some(1.0),
                },
                Conditioning::Control {
                    image: image(),
                    kind: ControlKind::Canny,
                    scale: Some(1.0),
                },
            ],
            ..Default::default()
        };
        assert_eq!(duplicate.memory_reference_count(), 2);
        assert!(
            crate::work_mode(
                provider,
                &z_spec,
                &crate::IosWork::Generate(duplicate.clone())
            )
            .is_err(),
            "two Controls are not Control+Reference"
        );
        assert!(
            super::anchor_variant_for_roles(provider, "control", "bf16", None, 2, 0, 2).is_err()
        );
        assert!(
            super::anchor_variant_for_roles(provider, "control", "bf16", None, 2, 1, 1).is_err()
        );
        let token = crate::sampling::resolve(provider, &duplicate).unwrap();
        let duplicate_req = IosPricingRequest {
            provider,
            spec: &z_spec,
            mode: &mode,
            geometry: MemoryGeometry {
                width: duplicate.width,
                height: duplicate.height,
                batch: 1,
                frames: 1,
                reference_count: 2,
            },
            selection: &selection,
            execution: &execution,
            generation: Some(&duplicate),
            transform: None,
            depth: None,
            token_window: 0,
        };
        assert!(
            matches!(mlx_gen_z_image::ios_pricing::ios_pricing(&duplicate_req), Err(mlx_gen::Error::Unsupported(reason)) if reason == role_error)
        );
        let replay = IosPricingRequest {
            generation: None,
            ..duplicate_req
        };
        assert!(
            matches!(crate::sampling::price_metadata(&replay, token.token(), 0, 2), Err(mlx_gen::Error::Unsupported(reason)) if reason == role_error)
        );
        let base = crate::zimage::Z_IMAGE_TURBO_ID;
        for (p, aggregate, refs, masks, controls) in [
            (base, 0, 0, 0, 0),
            (base, 1, 1, 0, 0),
            (provider, 1, 0, 0, 1),
            (provider, 2, 1, 0, 1),
        ] {
            assert!(
                mlx_gen_z_image::ios_pricing::validate_pricing_roles(
                    p, aggregate, refs, masks, controls
                )
                .is_ok(),
                "actual supported Z role partition must pass the pure provider rule"
            );
        }
        for (p, aggregate, refs, masks, controls) in [
            (base, 1, 0, 0, 1),
            (base, 2, 2, 0, 0),
            (provider, 2, 0, 0, 2),
            (provider, 2, 0, 1, 1),
            ("unknown-z-provider", 0, 0, 0, 0),
        ] {
            assert!(
                mlx_gen_z_image::ios_pricing::validate_pricing_roles(
                    p, aggregate, refs, masks, controls
                )
                .is_err(),
                "unknown providers or unsupported Z role partitions must not acquire a family"
            );
        }
        let mut base_spec = z_spec.clone();
        base_spec.control = None;
        let unknown = GenerationRequest {
            prompt: "unknown zero-image role".into(),
            conditioning: vec![Conditioning::ConversationHistory { turns: Vec::new() }],
            ..Default::default()
        };
        assert_eq!(unknown.memory_reference_count(), 0);
        assert!(
            crate::work_mode(base, &base_spec, &crate::IosWork::Generate(unknown.clone())).is_err(),
            "a zero-image unknown kind must not become empty text work"
        );
        let text_mode = MemoryMode::TextToImage;
        let unknown_req = IosPricingRequest {
            provider: base,
            spec: &base_spec,
            mode: &text_mode,
            geometry: MemoryGeometry {
                reference_count: 0,
                ..duplicate_req.geometry
            },
            generation: Some(&unknown),
            ..duplicate_req
        };
        assert!(
            matches!(mlx_gen_z_image::ios_pricing::ios_pricing(&unknown_req),
            Err(mlx_gen::Error::Unsupported(reason)) if reason == "Z-Image pricing: unsupported conditioning kind")
        );
        let mut sdxl_carrier = duplicate.clone();
        sdxl_carrier.conditioning.pop();
        let sdxl_spec = crate::sdxl::sdxl_load_spec("sdxl", WeightsSource::Dir(root)).unwrap();
        let sdxl_live = IosPricingRequest {
            provider: "sdxl",
            spec: &sdxl_spec,
            mode: &mode,
            geometry: MemoryGeometry {
                reference_count: 0,
                ..duplicate_req.geometry
            },
            generation: Some(&sdxl_carrier),
            ..duplicate_req
        };
        assert!(
            matches!(mlx_gen_sdxl::ios_pricing::ios_pricing(&sdxl_live), Err(mlx_gen::Error::Unsupported(reason)) if reason == "sdxl pricing generation and admitted aggregate image counts differ")
        );
        let recipe = mlx_gen_sdxl::sampling::SamplingRecipe::from_request(&sdxl_carrier).unwrap();
        let sdxl_replay = IosPricingRequest {
            generation: None,
            ..sdxl_live
        };
        assert!(
            matches!(mlx_gen_sdxl::ios_pricing::ios_pricing_with_recipe(&sdxl_replay, &recipe,
            mlx_gen_sdxl::sampling::SamplingWorkFacts { reference_count: 0, mask_count: 0, control_count: 1 }),
            Err(mlx_gen::Error::Unsupported(reason)) if reason == "sdxl captured recipe and geometry aggregate image counts differ")
        );
        let sdxl_valid_roles = IosPricingRequest {
            geometry: MemoryGeometry {
                reference_count: 1,
                ..sdxl_replay.geometry
            },
            ..sdxl_replay
        };
        assert!(matches!(crate::sampling::price_metadata(&sdxl_valid_roles, &recipe.canonical_token(), 0, 1),
            Err(mlx_gen::Error::Unsupported(reason)) if reason == "sdxl pricing Control conditioning count differs from loaded branches"),
            "valid scalar roles must reach the actual SDXL load-count guard; this is not a successful numeric price");
    }

    #[test]
    fn role_aliases_share_but_control_init_mask_and_ip_graphs_do_not() {
        public_role_boundaries_without_tensors();
        let role = |mode, refs: u32, masks: u32, controls: u32| {
            // Fixture arguments are TRUE roles; the existing wire receives their aggregate.
            let aggregate = refs
                .checked_add(masks)
                .and_then(|count| count.checked_add(controls))
                .unwrap();
            let variant = super::anchor_variant_for_roles(
                "sdxl",
                mode,
                "q4",
                Some("same-real-factory"),
                aggregate,
                masks,
                controls,
            );
            assert!(variant.is_ok(), "valid source role fixture must classify");
            variant.unwrap()
        };
        assert_eq!(role("reference", 1, 0, 0), role("image_to_image", 1, 0, 0));
        assert_eq!(role("text", 0, 0, 0), role("text_to_image", 0, 0, 0));
        let control = role("control", 0, 0, 1);
        let init = role("control", 1, 0, 1);
        let mask = role("edit", 1, 1, 0);
        assert_ne!(control, init);
        assert_ne!(
            init,
            role("image_to_image", 1, 0, 0),
            "active Control must not become plain init"
        );
        assert_ne!(
            control,
            role("control", 0, 0, 2),
            "configured multi-Control is a distinct source graph"
        );
        assert_ne!(init, mask);
        assert_ne!(role("image_to_image", 1, 0, 0), role("ip_adapter", 1, 0, 0));
        assert_ne!(
            super::anchor_variant_for_roles(
                crate::ios_admission::IRIS_RESTORE_ID,
                "restoration",
                "q4",
                None,
                0,
                0,
                0
            )
            .unwrap(),
            super::anchor_variant_for_roles(
                crate::ios_admission::IRIS_DEPTH_ID,
                "depth",
                "q4",
                None,
                0,
                0,
                0
            )
            .unwrap(),
        );

        // Real configured SDXL role selection, without loading a model or constructing an Array.
        use mlx_gen::{Conditioning, GenerationRequest, Image, WeightsSource};
        let image = || Image {
            width: 32,
            height: 32,
            pixels: vec![0; 32 * 32 * 3],
        };
        let mut spec =
            crate::sdxl::sdxl_load_spec("sdxl", WeightsSource::Dir("metadata-only".into()))
                .unwrap();
        spec.ip_adapter = Some(WeightsSource::File("metadata-only-ip".into()));
        let mut request = GenerationRequest {
            prompt: "role fixture".into(),
            strength: Some(0.9),
            conditioning: vec![Conditioning::Reference {
                image: image(),
                strength: Some(0.4),
            }],
            ..Default::default()
        };
        let actual_mode = |request: GenerationRequest, spec: &mlx_gen::LoadSpec| {
            crate::work_mode("sdxl", spec, &crate::IosWork::Generate(request))
        };
        assert_eq!(
            actual_mode(request.clone(), &spec).unwrap().as_key(),
            "ip_adapter"
        );
        request.sampler = Some("lcm".into());
        // The role predicate excludes IP, but the ACTUAL resolver currently refuses accel+Ref
        // before dispatch. Preserve that typed policy, not a fabricated accepted INIT family.
        assert!(!mlx_gen_sdxl::model::uses_ip_reference(
            true, true, false, 0, true
        ));
        let rejected = actual_mode(request.clone(), &spec);
        assert!(
            rejected.is_err(),
            "actual early acceleration+Reference gate must not be bypassed"
        );
        let rejected = rejected.unwrap_err();
        // Native sampling::invalid returns Msg, not Unsupported. Preserve its exact typed
        // failure/context across the bridge rather than relabeling it as missing evidence.
        assert!(matches!(rejected,
            crate::AdmissionError::Failed(mlx_gen::gen_core::Error::Msg(reason))
            if reason == "sdxl sampling: the \"lcm\" acceleration sampler is txt2img-only (no img2img reference) in this build"
        ));
        request.sampler = None;
        request
            .conditioning
            .push(Conditioning::Mask { image: image() });
        assert!(!mlx_gen_sdxl::model::uses_ip_reference(
            true, true, true, 0, false
        ));
        assert_eq!(
            actual_mode(request.clone(), &spec).unwrap().as_key(),
            "edit"
        );
        request.conditioning.pop();
        spec.control = Some(WeightsSource::File("metadata-only-control".into()));
        request.conditioning.push(Conditioning::Control {
            image: image(),
            kind: mlx_gen::ControlKind::Other("tile".into()),
            scale: Some(0.75),
        });
        assert_eq!(
            actual_mode(request.clone(), &spec).unwrap().as_key(),
            "control"
        );
        assert!(!mlx_gen_sdxl::model::uses_ip_reference(
            true, true, false, 1, false
        ));
        request.control_scale = Some(0.0);
        let recipe = mlx_gen_sdxl::sampling::SamplingRecipe::from_request(&request).unwrap();
        let token = recipe.canonical_token();
        request.control_scale = Some(99.0); // ignored by actual SDXL; per-Control scale stays owned.
        assert_eq!(
            token,
            mlx_gen_sdxl::sampling::SamplingRecipe::from_request(&request)
                .unwrap()
                .canonical_token()
        );
        recipe.apply_to(&mut request).unwrap();
        assert_eq!(request.strength, Some(0.9));
        assert!(
            matches!(&request.conditioning[0], Conditioning::Reference { strength: Some(value), .. } if *value == 0.4)
        );
        assert!(
            matches!(&request.conditioning[1], Conditioning::Control { scale: Some(value), .. } if *value == 0.75)
        );
        let resolved = mlx_gen_sdxl::sampling::ResolvedSampling::resolve(&request).unwrap();
        assert_eq!(resolved.effective_init_strength(true, false), Some(0.4));
    }

    #[test]
    fn unclassified_cardinality_never_becomes_a_raw_count_measurement_grid() {
        assert_eq!(super::reference_role_count(1, 0, 1).unwrap(), 0);
        assert_eq!(super::reference_role_count(2, 0, 1).unwrap(), 1);
        assert_eq!(super::reference_role_count(2, 1, 0).unwrap(), 1);
        assert!(super::reference_role_count(0, 0, 1).is_err());
        assert!(super::reference_role_count(u32::MAX, u32::MAX, 1).is_err());
        for (references, masks) in [(2, 0), (u32::MAX, 0), (3, 2)] {
            assert!(super::anchor_variant_for_roles(
                "sdxl", "edit", "q4", None, references, masks, 0,
            )
            .is_err());
        }
        assert!(
            super::anchor_variant_for_roles("sdxl", "ip_adapter", "q4", None, 0, 0, 0).is_err()
        );
        assert!(
            super::anchor_variant_for_roles("iris_3b_depth", "depth", "q4", None, 1, 0, 0).is_err()
        );
    }

    #[test]
    fn an_unproved_sampling_class_never_claims_component_additivity() {
        for mode in ["image_to_image", "edit", "control", "ip_adapter"] {
            assert_eq!(
                super::anchor_variant_for_sampling(
                    "sdxl",
                    mode,
                    "q4",
                    Some("unproved-factory-class")
                ),
                mode
            );
            assert_eq!(super::anchor_variant("sdxl", mode, "q4"), mode);
        }
        assert_eq!(
            super::anchor_variant_for_sampling(
                "sdxl",
                "text_to_image",
                "q4",
                Some("unproved-factory-class")
            ),
            "base"
        );
        assert!(super::non_additive_modes("sdxl", "q4").contains(&"image_to_image"));
    }

    #[test]
    fn every_registered_provider_is_priced() {
        let registry = crate::provider_registry().unwrap();
        for registration in registry.memory_strategy_registrations() {
            assert!(
                super::provider_pricing(registration.provider_id).is_some(),
                "{} has no counted iOS pricing",
                registration.provider_id
            );
        }
    }
}

/// HOST INTEGRATION CHECK over real snapshot HEADERS (no tensor payload, no graph, no device): the
/// catalog builds each provider's fully optimized candidate exactly as admission does
/// (`build_candidate` at the deepest implemented rung's cheapest selection) and prices it through
/// this dispatch — the request shape the anchor law hands the providers' pricing. Prints the
/// whole-run counted peak + the 512 MiB formula reserve against 6142 MiB at 512/768/1024.
/// `IOS_PRICING_SNAPSHOTS=<dir>` holds `<route>/<tier>/` snapshot directories (symlinks allowed);
/// a missing one is skipped. Run with `--ignored --nocapture`.
#[cfg(test)]
mod host_integration {
    use crate::ios_admission::{build_candidate, IosWork};
    use mlx_gen::gen_core::{
        GenerationRequest, MemoryNumericTier, MemoryStrategy, MemoryStrategySupport, Precision,
        Quant, WeightsSource,
    };
    use mlx_gen::ios_pricing::IosPricingRequest;

    #[test]
    #[ignore = "reads provisioned snapshot headers (IOS_PRICING_SNAPSHOTS)"]
    fn fully_optimized_candidates_price_through_the_catalog_dispatch() {
        let Some(root) = std::env::var_os("IOS_PRICING_SNAPSHOTS").map(std::path::PathBuf::from)
        else {
            println!("IOS_PRICING_SNAPSHOTS unset: nothing to price");
            return;
        };
        let registry = crate::provider_registry().unwrap();
        const MIB: u64 = 1 << 20;
        println!(
            "| provider | route | tier | edge | rung / selection | priced peak + 512 MiB | binding phase | vs 6142 |"
        );
        println!("|---|---|---|---:|---|---:|---|---|");
        let mut checked = 0;
        for (provider, route) in [
            ("sana_1600m", "sana_1600m"),
            ("sana_sprint_1600m", "sana_sprint_1600m"),
            ("z_image_turbo", "z_image_turbo"),
            ("sdxl", "sdxl"),
            ("sdxl", "realvisxl"),
            ("sdxl", "illustrious_xl_v2"),
        ] {
            for (tier, quant) in [
                ("q4", Some(Quant::Q4)),
                ("q8", Some(Quant::Q8)),
                ("bf16", None),
            ] {
                let dir = root.join(route).join(tier);
                if !dir.is_dir() {
                    continue;
                }
                let source = WeightsSource::Dir(dir);
                let mut spec = match provider {
                    "sdxl" => crate::sdxl::sdxl_load_spec(route, source).unwrap(),
                    id if crate::zimage::is_z_image(id) => {
                        crate::zimage::z_image_turbo_load_spec(source, None)
                    }
                    _ => crate::sana_load_spec(source),
                };
                spec.precision = Precision::Bf16;
                spec.quantize = quant;
                let contract = registry
                    .memory_strategy_contract(provider, &spec)
                    .unwrap()
                    .unwrap();
                let numeric = MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant,
                    component_precision_floors: &[],
                };
                let (strategy, selection) = MemoryStrategy::ALL
                    .into_iter()
                    .rev()
                    .filter(|s| {
                        contract.capability(*s).is_some_and(|c| {
                            matches!(c.support, MemoryStrategySupport::Implemented)
                        })
                    })
                    .find_map(|s| {
                        let space = contract.selection_space(s, numeric, false).ok()?;
                        space.last().copied().map(|sel| (s, sel))
                    })
                    .unwrap();
                for edge in [512u32, 768, 1024] {
                    let work = IosWork::Generate(GenerationRequest {
                        prompt: "A red ceramic teapot on a wooden table in daylight".into(),
                        width: edge,
                        height: edge,
                        count: 1,
                        ..Default::default()
                    });
                    let c = build_candidate(
                        &registry,
                        provider,
                        spec.clone(),
                        Some(selection),
                        strategy,
                        work,
                    )
                    .unwrap_or_else(|e| panic!("{provider}/{route}/{tier}/{edge}: {e}"));
                    let (pricer, _) = super::provider_pricing(provider).unwrap();
                    let window = crate::ios_admission::token_window(&c).unwrap();
                    let priced = pricer(&IosPricingRequest {
                        provider,
                        spec: &c.spec,
                        mode: &c.context.mode,
                        geometry: c.context.geometry,
                        selection: &c.context.selection,
                        execution: &c.execution,
                        token_window: window,
                        generation: match &c.work {
                            IosWork::Generate(request) => Some(request),
                            _ => None,
                        },
                        transform: match &c.work {
                            IosWork::Restore(request) => Some(request),
                            _ => None,
                        },
                        depth: match &c.work {
                            IosWork::Depth(request) => Some(request),
                            _ => None,
                        },
                    })
                    .unwrap_or_else(|e| panic!("{provider}/{route}/{tier}/{edge}: {e}"));
                    let worst = priced.phases.iter().max_by_key(|p| p.peak_bytes()).unwrap();
                    let total = priced.peak_bytes() + mlx_gen::ios_quantum::PROCESS_RESERVE_BYTES;
                    println!(
                        "| {provider} | {route} | {tier} | {edge} | {:?} {:?} | {} | {} ({}) | {} |",
                        strategy,
                        selection.parameters,
                        total.div_ceil(MIB),
                        worst.phase.as_key(),
                        worst.binding,
                        if total <= 6142 * MIB { "fits" } else { "OVER" }
                    );
                    assert!(
                        total <= 6142 * MIB,
                        "{provider}/{route}/{tier}/{edge} does not fit"
                    );
                    checked += 1;
                }
            }
        }
        println!("{checked} fully optimized cells priced through the catalog dispatch");
    }
}
