//! **Z-Image Turbo** on iOS (sc-26433): `z_image_turbo` (text-to-image, image-to-image, LoRA/LoKr
//! adapters) and its Fun-Controlnet-Union control route `z_image_turbo_control` (pose / canny /
//! depth, with an optional img2img init), each at every tier the provider advertises — the packed
//! Q4 / Q8 turnkeys (`SceneWorks/z-image-turbo-mlx` `q4/`, `q8/`) and the dense bf16 tier.
//!
//! Nothing here admits a load. Admission is the single evidence-table seam
//! ([`crate::ios_admission`]) with the provider's registered safety check
//! (`standard_memory_strategy_safety_check` + the Z-Image calibration handshake, E5); a capture is
//! `capture_permit`, bounded by the Z-Image quanta (`mlx_gen_z_image`'s `quantum` module). This
//! module states the iOS load defaults and the ARTIFACT policy: the published tiers are stored
//! packed, so a snapshot whose stored tier differs from the requested one (a dense snapshot asked
//! for as Q4/Q8 — a whole-model load-time requantization no capture measured — or a packed one asked
//! for at another tier) is a typed [`RefusalKind::Artifact`](crate::RefusalKind::Artifact) at
//! candidate construction. The Fun-Controlnet-Union checkpoint is published dense only; its packing
//! at load is part of the measured control route (one bounded quantum per block).

use mlx_gen::gen_core::{Error, LoadShape, LoadSpec, Result, WeightsSource};

use crate::IOS_OFFLOAD_POLICY;

/// The Qwen3-4B text-encoder contract a Z-Image Turbo snapshot's `text_encoder/` must satisfy
/// (provisioning tooling and fixtures validate against it).
pub use mlx_gen_z_image::ENCODER_CONTRACT;

/// The Z-Image Turbo generator.
pub const Z_IMAGE_TURBO_ID: &str = "z_image_turbo";
/// The Z-Image Turbo Fun-Controlnet-Union control route.
pub const Z_IMAGE_TURBO_CONTROL_ID: &str = "z_image_turbo_control";

/// Whether `provider` is one of the two Z-Image Turbo routes.
pub fn is_z_image(provider: &str) -> bool {
    provider == Z_IMAGE_TURBO_ID || provider == Z_IMAGE_TURBO_CONTROL_ID
}

/// A Z-Image Turbo load spec with the iOS defaults: phases released between conditioning, denoise
/// and decode (the sequential load every iOS model uses) and a deferred materialization, so the
/// rung-4 text-encoder / DiT / ControlNet windows are available (`Missing` from the provider's
/// contract on an eager load). `control` is the Fun-Controlnet-Union checkpoint for
/// [`Z_IMAGE_TURBO_CONTROL_ID`].
pub fn z_image_turbo_load_spec(source: WeightsSource, control: Option<WeightsSource>) -> LoadSpec {
    let mut spec = LoadSpec::new(source)
        .with_offload_policy(IOS_OFFLOAD_POLICY)
        .with_load_shape(LoadShape::DeferredMaterialization);
    spec.control = control;
    spec
}

/// The artifact policy for a Z-Image Turbo candidate: a snapshot directory with no component
/// overrides whose transformer, text encoder and VAE are each STORED at the requested tier (packed
/// Q4/Q8 at that width, or dense for bf16), and — for the control route — a provisioned ControlNet
/// checkpoint, and no PiD overlay. Any other inventory is `Unsupported` (a typed artifact refusal at
/// admission).
pub fn check_artifact(provider: &str, spec: &LoadSpec) -> Result<()> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(Error::Unsupported(format!(
            "{provider}: iOS evidence requires a snapshot directory"
        )));
    };
    if !spec.components.is_empty() || spec.ip_adapter.is_some() {
        return Err(Error::Unsupported(format!(
            "{provider}: iOS loads the snapshot's own components (no overrides)"
        )));
    }
    // A PiD overlay (the PiD student plus its Gemma encoder) is built at load by the provider,
    // outside every iOS quantum and footprint: never an iOS load.
    if spec.pid.is_some() {
        return Err(Error::Unsupported(format!(
            "{provider}: a PiD decode overlay is not an iOS load (never captured or bounded)"
        )));
    }
    match (provider == Z_IMAGE_TURBO_CONTROL_ID, &spec.control) {
        (true, None) => {
            return Err(Error::Unsupported(format!(
                "{provider}: needs a provisioned Fun-Controlnet-Union checkpoint (`control`)"
            )))
        }
        (false, Some(_)) => {
            return Err(Error::Unsupported(format!(
                "{provider}: a ControlNet checkpoint is the {Z_IMAGE_TURBO_CONTROL_ID} route's"
            )))
        }
        _ => {}
    }
    let requested = spec.quantize.map(|q| q.bits());
    for component in ["transformer", "text_encoder", "vae"] {
        // An unreadable or malformed marker is a failure, not an artifact-policy refusal.
        let stored = mlx_gen::quant::packed_quant_bits(root, component)
            .map_err(|e| Error::Msg(format!("{provider}: {component}/: {e}")))?;
        if stored != requested {
            return Err(Error::Unsupported(format!(
                "{provider}: {component}/ is stored {} but the load requests {}; iOS admits only \
                 the tier the artifact stores (a load-time requantization was never captured)",
                tier_name(stored),
                tier_name(requested)
            )));
        }
    }
    Ok(())
}

fn tier_name(bits: Option<i32>) -> String {
    bits.map_or("dense".into(), |b| format!("Q{b}"))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::evidence::{EvidenceRow, DEVICE_CAPTURE_CLASS};
    use crate::ios_admission::{admit, admit_with_rows, build_candidate, evidence_key, IosWork};
    use crate::sdxl::tests::{budget, write_tensor, DEVICE};
    use crate::RefusalKind;
    use mlx_gen::gen_core::{
        Conditioning, ControlKind, GenerationOutput, GenerationRequest, Generator, Image,
        MemoryStrategy, Progress, Quant,
    };
    use std::path::Path;

    /// A header-only Z-Image Turbo snapshot stored at `bits` (None = dense bf16): transformer and
    /// VAE tensors with their packed-tier markers, and the production Qwen3-4B text encoder's
    /// sparse contract fixture (the provider validates its exact geometry) with its tokenizer.
    pub(crate) fn snapshot(root: &Path, bits: Option<i32>) {
        let marker = |dir: &Path| {
            let mut config = serde_json::json!({"_class_name": "fixture"});
            if let Some(bits) = bits {
                config["quantization"] = serde_json::json!({"bits": bits, "group_size": 64});
            }
            std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        };
        for component in ["transformer", "vae"] {
            let dir = root.join(component);
            write_tensor(&dir.join("model.safetensors"), "BF16", 64);
            marker(&dir);
        }
        gen_core_testkit::write_encoder_contract_fixture_with_quant(
            &root.join("text_encoder"),
            mlx_gen_z_image::ENCODER_CONTRACT,
            bits,
        )
        .unwrap();
    }

    /// A header-only Fun-Controlnet-Union checkpoint.
    pub(crate) fn control_checkpoint(path: &Path) -> WeightsSource {
        write_tensor(path, "BF16", 64);
        WeightsSource::File(path.to_owned())
    }

    fn image(edge: u32) -> Image {
        Image {
            width: edge,
            height: edge,
            pixels: (0..edge * edge * 3).map(|i| (i % 251) as u8).collect(),
        }
    }

    /// Every mode of the two routes: text, image-to-image, control, control + img2img init.
    pub(crate) fn work(mode: &str, edge: u32) -> IosWork {
        let conditioning = match mode {
            "text" => vec![],
            "reference" => vec![Conditioning::Reference {
                image: image(edge),
                strength: Some(0.6),
            }],
            "control" => vec![Conditioning::Control {
                image: image(edge),
                kind: ControlKind::Pose,
                scale: Some(1.0),
            }],
            "control_reference" => vec![
                Conditioning::Control {
                    image: image(edge),
                    kind: ControlKind::Pose,
                    scale: Some(1.0),
                },
                Conditioning::Reference {
                    image: image(edge),
                    strength: Some(0.6),
                },
            ],
            other => panic!("{other}"),
        };
        IosWork::Generate(GenerationRequest {
            prompt: "a lighthouse".into(),
            width: edge,
            height: edge,
            count: 1,
            conditioning,
            ..Default::default()
        })
    }

    fn quant(bits: Option<i32>) -> Option<Quant> {
        bits.map(|b| if b == 4 { Quant::Q4 } else { Quant::Q8 })
    }

    /// E7: both routes register every mode and tier the provider advertises, and no trainer.
    #[test]
    fn z_image_turbo_registers_every_mode_and_tier() {
        use mlx_gen::gen_core::ConditioningKind::{Control, Reference};
        let registry = crate::provider_registry().unwrap();
        let descriptor = |id: &str| {
            registry
                .generators()
                .map(|r| (r.descriptor)())
                .find(|d| d.id == id)
                .unwrap_or_else(|| panic!("{id} must be registered, not just linked"))
        };
        let turbo = descriptor(Z_IMAGE_TURBO_ID);
        assert_eq!(turbo.capabilities.conditioning, vec![Reference]);
        let control = descriptor(Z_IMAGE_TURBO_CONTROL_ID);
        assert_eq!(control.capabilities.conditioning, vec![Control, Reference]);
        for d in [&turbo, &control] {
            assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
            assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        }
        for kind in [ControlKind::Pose, ControlKind::Canny, ControlKind::Depth] {
            let IosWork::Generate(mut req) = work("control", 512) else {
                unreachable!()
            };
            req.conditioning = vec![Conditioning::Control {
                image: image(512),
                kind: kind.clone(),
                scale: Some(1.0),
            }];
            control
                .capabilities
                .validate_request(Z_IMAGE_TURBO_CONTROL_ID, &req)
                .unwrap_or_else(|e| panic!("{kind:?} was dropped: {e}"));
        }
        for (id, modes) in [
            (Z_IMAGE_TURBO_ID, &["text", "reference"][..]),
            (
                Z_IMAGE_TURBO_CONTROL_ID,
                &["control", "control_reference"][..],
            ),
        ] {
            for mode in modes {
                let IosWork::Generate(req) = work(mode, 512) else {
                    unreachable!()
                };
                descriptor(id)
                    .capabilities
                    .validate_request(id, &req)
                    .unwrap_or_else(|e| panic!("{id} {mode}: {e}"));
            }
            assert!(registry
                .memory_strategy_registrations()
                .any(|r| r.provider_id == id));
        }
        assert_eq!(registry.trainers().len(), 0);
    }

    /// A request with no reviewed device row is a typed `no_evidence` refusal before any load, for
    /// every mode at every tier; the key carries the stored tier.
    #[test]
    fn every_mode_and_tier_is_refused_without_evidence_before_any_load() {
        let registry = crate::provider_registry().unwrap();
        for bits in [Some(4), Some(8), None] {
            let root = tempfile::tempdir().unwrap();
            snapshot(root.path(), bits);
            let control = control_checkpoint(&root.path().join("control.safetensors"));
            for (id, mode) in [
                (Z_IMAGE_TURBO_ID, "text"),
                (Z_IMAGE_TURBO_ID, "reference"),
                (Z_IMAGE_TURBO_CONTROL_ID, "control"),
                (Z_IMAGE_TURBO_CONTROL_ID, "control_reference"),
            ] {
                let mut spec = z_image_turbo_load_spec(
                    WeightsSource::Dir(root.path().to_owned()),
                    (id == Z_IMAGE_TURBO_CONTROL_ID).then(|| control.clone()),
                );
                spec.quantize = quant(bits);
                let candidate = build_candidate(
                    &registry,
                    id,
                    spec,
                    None,
                    MemoryStrategy::BoundedTransformerResidency,
                    work(mode, 512),
                )
                .unwrap_or_else(|e| panic!("{id} {mode} {bits:?}: {e}"));
                let key = evidence_key(&registry, &candidate, DEVICE).unwrap();
                assert_eq!(key.tier, crate::tier_label(&candidate.spec));
                let refused = admit(&registry, &[candidate], DEVICE, budget())
                    .expect_err("no device row: never admitted");
                assert_eq!(
                    refused.refusal_kind(),
                    Some(RefusalKind::NoEvidence),
                    "{id} {mode} {bits:?}: {refused}"
                );
            }
        }
    }

    /// The stored-tier artifact policy: a load whose requested tier is not the stored one, a control
    /// route without its checkpoint, or overridden components are typed ARTIFACT refusals at
    /// candidate construction.
    #[test]
    fn artifact_policy_refusals_are_typed() {
        let registry = crate::provider_registry().unwrap();
        let attempt = |root: &Path, id: &str, q: Option<Quant>, control: Option<WeightsSource>| {
            let mut spec = z_image_turbo_load_spec(WeightsSource::Dir(root.to_owned()), control);
            spec.quantize = q;
            build_candidate(
                &registry,
                id,
                spec,
                None,
                MemoryStrategy::BoundedTransformerResidency,
                work(
                    if id == Z_IMAGE_TURBO_ID {
                        "text"
                    } else {
                        "control"
                    },
                    512,
                ),
            )
            .map(|_| ())
            .expect_err("must be refused")
            .refusal_kind()
        };
        let dense = tempfile::tempdir().unwrap();
        snapshot(dense.path(), None);
        let q4 = tempfile::tempdir().unwrap();
        snapshot(q4.path(), Some(4));
        let control = control_checkpoint(&dense.path().join("control.safetensors"));
        let artifact = Some(RefusalKind::Artifact);
        // Dense requested as Q4/Q8: a load-time requantization of the whole model.
        assert_eq!(
            attempt(dense.path(), Z_IMAGE_TURBO_ID, Some(Quant::Q4), None),
            artifact
        );
        assert_eq!(
            attempt(dense.path(), Z_IMAGE_TURBO_ID, Some(Quant::Q8), None),
            artifact
        );
        // A packed turnkey requested at another tier (or as bf16).
        assert_eq!(
            attempt(q4.path(), Z_IMAGE_TURBO_ID, Some(Quant::Q8), None),
            artifact
        );
        assert_eq!(attempt(q4.path(), Z_IMAGE_TURBO_ID, None, None), artifact);
        // The control route needs its checkpoint; the base route takes none.
        assert_eq!(
            attempt(q4.path(), Z_IMAGE_TURBO_CONTROL_ID, Some(Quant::Q4), None),
            artifact
        );
        assert_eq!(
            attempt(q4.path(), Z_IMAGE_TURBO_ID, Some(Quant::Q4), Some(control)),
            artifact
        );
        // A text encoder stored at another tier than the transformer.
        let mixed = tempfile::tempdir().unwrap();
        snapshot(mixed.path(), Some(4));
        std::fs::write(
            mixed.path().join("text_encoder/config.json"),
            std::fs::read_to_string(mixed.path().join("text_encoder/config.json"))
                .unwrap()
                .replace("\"bits\":4", "\"bits\":8"),
        )
        .unwrap();
        assert_eq!(
            attempt(mixed.path(), Z_IMAGE_TURBO_ID, Some(Quant::Q4), None),
            artifact
        );
        // A PiD overlay: the provider would build the PiD student and its Gemma encoder at load,
        // outside every quantum — refused before any load.
        let mut spec = z_image_turbo_load_spec(WeightsSource::Dir(q4.path().to_owned()), None);
        spec.quantize = Some(Quant::Q4);
        spec.pid = Some(mlx_gen::gen_core::PidWeights {
            checkpoint: WeightsSource::File(q4.path().join("pid.safetensors")),
            gemma: WeightsSource::Dir(q4.path().join("gemma")),
        });
        let refused = build_candidate(
            &registry,
            Z_IMAGE_TURBO_ID,
            spec,
            None,
            MemoryStrategy::BoundedTransformerResidency,
            work("text", 512),
        )
        .map(|_| ())
        .expect_err("a PiD overlay must be refused");
        assert_eq!(refused.refusal_kind(), artifact);
        // A PiD decode request on an admitted candidate: a typed MODE refusal, not a mismatch.
        let mut spec = z_image_turbo_load_spec(WeightsSource::Dir(q4.path().to_owned()), None);
        spec.quantize = Some(Quant::Q4);
        let mut candidate = build_candidate(
            &registry,
            Z_IMAGE_TURBO_ID,
            spec,
            None,
            MemoryStrategy::BoundedTransformerResidency,
            work("text", 512),
        )
        .unwrap();
        crate::ios_admission::validate_actual(&registry, &candidate).unwrap();
        let IosWork::Generate(req) = &mut candidate.work else {
            unreachable!()
        };
        req.use_pid = true;
        let refused = crate::ios_admission::validate_actual(&registry, &candidate)
            .expect_err("use_pid must be refused");
        assert_eq!(
            refused.refusal_kind(),
            Some(RefusalKind::Mode),
            "{refused:?}"
        );
    }

    struct Probe(
        mlx_gen::gen_core::ModelDescriptor,
        std::rc::Rc<std::cell::Cell<u32>>,
    );
    impl Generator for Probe {
        fn descriptor(&self) -> &mlx_gen::gen_core::ModelDescriptor {
            &self.0
        }
        fn validate(&self, _: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
            Ok(())
        }
        fn generate(
            &self,
            _: &GenerationRequest,
            _: &mut dyn FnMut(Progress),
        ) -> mlx_gen::gen_core::Result<GenerationOutput> {
            self.1.set(self.1.get() + 1);
            Ok(GenerationOutput::Images(vec![]))
        }
    }

    /// With a reviewed row the exact work is admitted and loaded as a request-bound generator: the
    /// admitted request reaches the provider, an altered one never does.
    #[test]
    fn with_evidence_the_route_is_admitted_through_the_bound_generator() {
        let registry = crate::provider_registry().unwrap();
        let root = tempfile::tempdir().unwrap();
        snapshot(root.path(), Some(4));
        let mut spec = z_image_turbo_load_spec(WeightsSource::Dir(root.path().to_owned()), None);
        spec.quantize = Some(Quant::Q4);
        let candidate = build_candidate(
            &registry,
            Z_IMAGE_TURBO_ID,
            spec,
            None,
            MemoryStrategy::BoundedTransformerResidency,
            work("text", 768),
        )
        .unwrap();
        let row = EvidenceRow {
            key: evidence_key(&registry, &candidate, DEVICE).unwrap(),
            footprint_bytes: 3 << 30,
            baseline_footprint_bytes: 1 << 28,
            mlx_peak_bytes: 2 << 30,
            min_headroom_mib: 900.0,
            elapsed_seconds: 60.0,
            evidence_class: DEVICE_CAPTURE_CLASS.into(),
            run_id: "synthetic".into(),
            build_sha: "synthetic".into(),
        };
        let admitted = admit_with_rows(
            &registry,
            std::slice::from_ref(&candidate),
            DEVICE,
            crate::capacity_budget(8 << 30, 0, None),
            &[row],
        )
        .unwrap();
        assert!(!admitted.is_capture());
        assert_eq!(admitted.demand_bytes(), (3 << 30) - (1 << 28));
        let calls = std::rc::Rc::new(std::cell::Cell::new(0));
        let desc = registry
            .generators()
            .map(|r| (r.descriptor)())
            .find(|d| d.id == Z_IMAGE_TURBO_ID)
            .unwrap();
        let probe_calls = calls.clone();
        let bound = admitted
            .load_generator_using(
                move |_| Ok(Box::new(Probe(desc, probe_calls))),
                Box::new(|_| Ok(())),
            )
            .unwrap();
        let IosWork::Generate(req) = &candidate.work else {
            unreachable!()
        };
        let mut bigger = req.clone();
        bigger.width = 1024;
        bigger.height = 1024;
        assert!(bound.generate(&bigger, &mut |_| {}).is_err());
        assert_eq!(calls.get(), 0, "an altered request reached the provider");
        bound.generate(req, &mut |_| {}).unwrap();
        assert_eq!(calls.get(), 1);
    }
}
