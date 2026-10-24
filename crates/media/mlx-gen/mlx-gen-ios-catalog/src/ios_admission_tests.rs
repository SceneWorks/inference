//! The shared seam across providers/modes: typed mode refusals, SANA's request-bound wrapper,
//! the bound transform/depth wrappers, and the non-circular capture permit.
use super::*;
use crate::sdxl::tests::{admit_fixture_rows as admit_with_rows, budget, DEVICE};
use mlx_gen::gen_core::iris::depth::{DepthOutput, IrisDepthEstimator};
use mlx_gen::gen_core::{
    GenerationOutput, Generator, Image, MemoryStrategy, Progress, Transform, TransformDescriptor,
    WeightsSource,
};
use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;

// Reuse the existing provider CPU-child/Metal-trap mechanism, not a second guard implementation.
#[cfg(target_os = "macos")]
#[path = "../../mlx-gen-iris/tests/common/metal_guard.rs"]
mod metal_guard;

#[cfg(target_os = "macos")]
fn in_cpu_child(test: &str, body: fn()) {
    const CHILD: &str = "IOS_ADMISSION_CPU_CHILD";
    const DONE: &str = "IOS_ADMISSION_CPU_CHILD_DONE";
    if std::env::var(CHILD).as_deref() == Ok(test) {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert_eq!(
            mlx_rs::Device::try_default().expect("CPU default-device query failed"),
            mlx_rs::Device::cpu()
        );
        assert!(metal_guard::forbid_metal());
        assert!(metal_guard::metal_forbidden());
        assert_eq!(
            unsafe { mlx_sys::mlx_set_default_stream(mlx_rs::Stream::cpu().as_ptr()) },
            0
        );
        assert_eq!(
            mlx_rs::Stream::new().get_index().unwrap(),
            mlx_rs::Stream::cpu().get_index().unwrap()
        );
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

#[cfg(not(target_os = "macos"))]
fn in_cpu_child(_: &str, body: fn()) {
    body();
}

fn sana(root: &Path) -> LoadSpec {
    for (dir, key) in [
        ("transformer/a.safetensors", "proj.weight"),
        (
            "text_encoder/gemma-2-2b-it.safetensors",
            "model.embed_tokens.weight",
        ),
        ("vae/a.safetensors", "conv.weight"),
    ] {
        let path = root.join(dir);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let header =
            format!("{{\"{key}\":{{\"dtype\":\"F32\",\"shape\":[8,8],\"data_offsets\":[0,256]}}}}");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.resize(bytes.len() + 256, 0);
        std::fs::write(path, bytes).unwrap();
    }
    crate::sana_load_spec(WeightsSource::Dir(root.to_owned()))
}

fn image(edge: u32) -> Image {
    Image {
        width: edge,
        height: edge,
        pixels: (0..edge * edge * 3).map(|i| (i % 251) as u8).collect(),
    }
}

fn generation(edge: u32, conditioning: Vec<Conditioning>) -> IosWork {
    IosWork::Generate(GenerationRequest {
        prompt: "p".into(),
        width: edge,
        height: edge,
        count: 1,
        conditioning,
        ..Default::default()
    })
}

/// A synthetic completed anchor AT the candidate's own key whose measured demand is `bytes` (a
/// rung-0 candidate is a resident anchor, any other an optimized one); its margin is the floor.
fn row(registry: &ProviderRegistry, c: &IosCandidate, bytes: u64) -> AnchorRow {
    let kind = if c.context.selection.strategy == MemoryStrategy::Resident {
        AnchorKind::Resident
    } else {
        AnchorKind::Optimized
    };
    evidence::fixture::completed(
        crate::source_work::fixture_key(registry, c, DEVICE),
        kind,
        evidence::AnchorSelection::from_parts(
            c.context.selection.strategy,
            &c.context.selection.parameters,
        ),
        bytes,
    )
}

fn iris_fixtures() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../mlx-gen-iris/tests/fixtures")
}

fn kind<T: std::fmt::Debug>(r: AdmissionResult<T>) -> Option<RefusalKind> {
    match r {
        Err(e) => e.refusal_kind(),
        Ok(v) => panic!("admitted: {v:?}"),
    }
}

/// E7: an unserved mode is a TYPED `mode` refusal by admission, never silently dropped or
/// rewritten into text-to-image.
#[test]
fn unserved_modes_are_typed_mode_refusals() {
    let registry = crate::provider_registry().unwrap();
    let root = tempfile::tempdir().unwrap();
    let spec = sana(root.path());
    let control = vec![Conditioning::Control {
        image: image(8),
        kind: mlx_gen::gen_core::ControlKind::Other("tile".into()),
        scale: None,
    }];
    // A control request with a context built for it: the mode is the refusal, not drift.
    let candidate = build_candidate(
        &registry,
        "sana_1600m",
        spec.clone(),
        None,
        MemoryStrategy::BoundedDecode,
        generation(512, control),
    )
    .unwrap();
    assert_eq!(candidate.context.mode, MemoryMode::Other("control".into()));
    let refused = validate_actual(&registry, &candidate).expect_err("control on SANA");
    assert_eq!(refused.refusal_kind(), Some(RefusalKind::Mode), "{refused}");
    assert_eq!(
        kind(admit(&registry, &[candidate], DEVICE, budget())),
        Some(RefusalKind::Mode)
    );
    // SANA image-to-image IS served: admitted-reachable, refused only for evidence.
    let img2img = build_candidate(
        &registry,
        "sana_1600m",
        spec,
        None,
        MemoryStrategy::BoundedDecode,
        generation(
            512,
            vec![Conditioning::Reference {
                image: image(512),
                strength: Some(0.5),
            }],
        ),
    )
    .unwrap();
    assert_eq!(img2img.context.mode, MemoryMode::ImageToImage);
    validate_actual(&registry, &img2img).unwrap();
    assert_eq!(
        kind(admit(&registry, &[img2img], DEVICE, budget())),
        Some(RefusalKind::NoEvidence)
    );
}

struct ProbeGenerator {
    desc: mlx_gen::gen_core::ModelDescriptor,
    calls: Rc<Cell<u32>>,
}
impl Generator for ProbeGenerator {
    fn descriptor(&self) -> &mlx_gen::gen_core::ModelDescriptor {
        &self.desc
    }
    fn validate(&self, _: &GenerationRequest) -> Result<()> {
        Ok(())
    }
    fn generate(
        &self,
        _: &GenerationRequest,
        _: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.calls.set(self.calls.get() + 1);
        Ok(GenerationOutput::Images(vec![]))
    }
}

/// A dense SANA snapshot requested as q4 has no provider calibration identity: a typed ARTIFACT
/// refusal at candidate construction, before any evidence key or price exists.
#[test]
fn a_dense_snapshot_requested_as_q4_is_a_typed_artifact_refusal() {
    let registry = crate::provider_registry().unwrap();
    let root = tempfile::tempdir().unwrap();
    // Both advertised packed tiers (`supported_quants: [Q4, Q8]`), base and Sprint.
    for provider in ["sana_1600m", "sana_sprint_1600m"] {
        for quant in [mlx_gen::gen_core::Quant::Q4, mlx_gen::gen_core::Quant::Q8] {
            let mut spec = sana(root.path());
            spec.quantize = Some(quant);
            let refused = build_candidate(
                &registry,
                provider,
                spec,
                None,
                MemoryStrategy::BoundedDecode,
                generation(512, vec![]),
            )
            .expect_err("dense-as-packed has no calibrated identity");
            assert_eq!(
                refused.refusal_kind(),
                Some(RefusalKind::Artifact),
                "{provider} {quant:?}: {refused}"
            );
        }
    }
}

/// The key records the LOADED tier (never a default), alongside the layout digest.
#[test]
fn the_loaded_tier_is_recorded_in_the_key() {
    let registry = crate::provider_registry().unwrap();
    let root = crate::sdxl::tests::fixture();
    for (quant, label) in [
        (None, "bf16"),
        (Some(mlx_gen::gen_core::Quant::Q8), "q8"),
        (Some(mlx_gen::gen_core::Quant::Q4), "q4"),
    ] {
        let c = crate::sdxl::tests::candidate(
            &registry,
            root.path(),
            "sdxl",
            512,
            quant,
            "text_to_image",
        );
        assert_eq!(tier_label(&c.spec), label);
        assert_eq!(evidence_key(&registry, &c, DEVICE).unwrap().tier, label);
    }
}

#[test]
fn endpoint_family_is_shared_but_execution_and_receipt_load_forms_are_not() {
    use mlx_gen::gen_core::{LoadShape, OffloadPolicy};
    let registry = crate::provider_registry().unwrap();
    let root = tempfile::tempdir().unwrap();
    let mut resident_spec = sana(root.path());
    AnchorKind::Resident.apply_load_form(&mut resident_spec);
    let resident = build_candidate(
        &registry,
        "sana_1600m",
        resident_spec.clone(),
        None,
        MemoryStrategy::Resident,
        generation(512, vec![]),
    )
    .unwrap();
    let mut optimized_spec = resident_spec;
    AnchorKind::Optimized.apply_load_form(&mut optimized_spec);
    let optimized = build_candidate(
        &registry,
        "sana_1600m",
        optimized_spec,
        None,
        MemoryStrategy::BoundedAttention,
        generation(512, vec![]),
    )
    .unwrap();
    let resident_key = evidence_key(&registry, &resident, DEVICE).unwrap();
    let optimized_key = evidence_key(&registry, &optimized, DEVICE).unwrap();
    assert_eq!(resident_key.family(), optimized_key.family());
    assert_ne!(resident_key.layout_sha256, optimized_key.layout_sha256);
    assert_ne!(resident_key.execution, optimized_key.execution);
    assert!(resident_key
        .execution
        .starts_with("Resident|EagerMaterialization|"));
    assert!(optimized_key
        .execution
        .starts_with("Sequential|DeferredMaterialization|"));
    assert!(!resident.execution.stage_residency);
    let mut drifted = resident.clone();
    drifted.spec.load_shape = LoadShape::DeferredMaterialization;
    assert!(validate_actual(&registry, &drifted).is_err());
    // A receipt cannot substitute either axis after admission, even if a forged context copies it.
    for shape_axis in [false, true] {
        let rows = [row(&registry, &resident, 1 << 20)];
        let mut receipt = admit_with_rows(
            &registry,
            std::slice::from_ref(&resident),
            DEVICE,
            budget(),
            &rows,
        )
        .unwrap();
        if shape_axis {
            receipt.spec.load_shape = LoadShape::DeferredMaterialization;
            receipt.context.load_shape = LoadShape::DeferredMaterialization;
        } else {
            receipt.spec.offload_policy = OffloadPolicy::Sequential;
        }
        let called = Rc::new(Cell::new(false));
        let loaded = called.clone();
        assert!(receipt
            .load_generator_using(
                move |_| {
                    loaded.set(true);
                    panic!("forged load form reached loader")
                },
                Box::new(|_| Ok(()))
            )
            .is_err());
        assert!(!called.get());
    }
}

#[test]
fn sdxl_token_window_uses_effective_sampler_guidance_not_an_unconditional_negative_row() {
    let registry = crate::provider_registry().unwrap();
    let root = crate::sdxl::tests::fixture();
    let base =
        crate::sdxl::tests::candidate(&registry, root.path(), "sdxl", 512, None, "text_to_image");
    let descriptor = registry
        .generators()
        .map(|r| (r.descriptor)())
        .find(|d| d.id == "sdxl")
        .unwrap();
    for sampler in descriptor.capabilities.samplers {
        let mut candidate = base.clone();
        let IosWork::Generate(request) = &mut candidate.work else {
            unreachable!()
        };
        request.sampler = Some(sampler.into());
        request.negative_prompt = Some("many long negative prompt tokens ".repeat(1000));
        request.guidance = Some(1.0);
        let off = token_window(&candidate).unwrap();
        let IosWork::Generate(request) = &mut candidate.work else {
            unreachable!()
        };
        request.negative_prompt = None;
        assert_eq!(
            token_window(&candidate).unwrap(),
            off,
            "CFG-off {sampler} must not tokenize the negative prompt"
        );
        let IosWork::Generate(request) = &mut candidate.work else {
            unreachable!()
        };
        request.negative_prompt = Some("many long negative prompt tokens ".repeat(1000));
        request.guidance = Some(7.0);
        assert!(
            token_window(&candidate).unwrap() > off,
            "CFG-on {sampler} must include the actual negative window"
        );
        let IosWork::Generate(request) = &mut candidate.work else {
            unreachable!()
        };
        request.guidance = None;
        let cfg_on = crate::sdxl::sampling::ResolvedSampling::resolve(request)
            .unwrap()
            .guidance()
            > 1.0;
        assert_eq!(
            token_window(&candidate).unwrap() > off,
            cfg_on,
            "{sampler} actual default guidance"
        );
    }
}

/// Evidence compatibility uses resolved allocation graphs, while exact records retain the full
/// canonical recipe. Each advertised solver remains expressible; another graph needs its anchors.
#[test]
fn sdxl_sampling_classes_and_explicit_cfg_are_bound_before_loading() {
    use mlx_gen::gen_core::CfgBatching;
    let registry = crate::provider_registry().unwrap();
    let root = crate::sdxl::tests::fixture();
    let base =
        crate::sdxl::tests::candidate(&registry, root.path(), "sdxl", 512, None, "text_to_image");
    let base_key = evidence_key(&registry, &base, DEVICE).unwrap();
    let ancestral_rows = [row(&registry, &base, 3 << 20)];
    let with_sampling = |sampler: &str, guidance: f32, batching: CfgBatching| {
        let IosWork::Generate(mut request) = base.work.clone() else {
            unreachable!()
        };
        request.sampler = Some(sampler.into());
        request.guidance = Some(guidance);
        request.memory.as_mut().unwrap().cfg_batching = Some(batching);
        build_candidate(
            &registry,
            "sdxl",
            base.spec.clone(),
            None,
            base.context.selection.strategy,
            IosWork::Generate(request),
        )
        .unwrap()
    };
    let descriptor = registry
        .generators()
        .map(|r| (r.descriptor)())
        .find(|d| d.id == "sdxl")
        .unwrap();
    for sampler in descriptor.capabilities.samplers {
        let candidate = with_sampling(sampler, 7.0, CfgBatching::Sequential);
        let key = evidence_key(&registry, &candidate, DEVICE).unwrap();
        assert!(key.sampling_recipe.is_some());
        let own_rows = [row(&registry, &candidate, 3 << 20)];
        assert!(
            admit_with_rows(
                &registry,
                std::slice::from_ref(&candidate),
                DEVICE,
                budget(),
                &own_rows
            )
            .is_ok(),
            "{sampler}"
        );
        if key.family().unwrap() != base_key.family().unwrap() {
            assert_eq!(
                kind(admit_with_rows(
                    &registry,
                    &[candidate],
                    DEVICE,
                    budget(),
                    &ancestral_rows
                )),
                Some(RefusalKind::NoEvidence),
                "{sampler}"
            );
        }
    }
    let batched = with_sampling("euler", 7.0, CfgBatching::Batched);
    assert_eq!(
        batched.execution.cfg_batching,
        Some(CfgBatching::Batched),
        "the iOS default must not override an explicit caller choice"
    );
    let IosWork::Generate(actual) = &batched.work else {
        unreachable!()
    };
    assert_eq!(
        actual.memory.unwrap().cfg_batching,
        Some(CfgBatching::Batched)
    );
    let batch_rows = [row(&registry, &batched, 3 << 20)];
    assert!(admit_with_rows(
        &registry,
        std::slice::from_ref(&batched),
        DEVICE,
        budget(),
        &batch_rows
    )
    .is_ok());
    assert_eq!(
        kind(admit_with_rows(
            &registry,
            std::slice::from_ref(&batched),
            DEVICE,
            budget(),
            &ancestral_rows
        )),
        Some(RefusalKind::NoEvidence)
    );
    let continuous = with_sampling("euler", 8.25, CfgBatching::Batched);
    let k1 = evidence_key(&registry, &batched, DEVICE).unwrap();
    let k2 = evidence_key(&registry, &continuous, DEVICE).unwrap();
    assert_eq!(
        k1.family().unwrap(),
        k2.family().unwrap(),
        "continuous guidance values must not create a measurement grid"
    );
    assert_ne!(
        k1.execution, k2.execution,
        "exact execution provenance retains the actual recipe"
    );
    for missing in [true, false] {
        let mut forged = batch_rows[0].clone();
        if missing {
            forged.key.sampling_recipe = None;
        } else {
            forged.key.sampling_recipe = ancestral_rows[0].key.sampling_recipe.clone();
        }
        assert_eq!(
            kind(admit_with_rows(
                &registry,
                std::slice::from_ref(&batched),
                DEVICE,
                budget(),
                &[forged]
            )),
            Some(RefusalKind::NoEvidence)
        );
    }
}

/// Finding 4: SANA's admitted generator is request-bound. A larger geometry, an untiled or
/// different-tile decode, or an img2img request is refused BEFORE the provider is invoked, and
/// live capacity is rechecked on every call.
#[test]
fn sana_bound_generator_rechecks_geometry_tile_mode_and_capacity_before_the_provider() {
    let registry = crate::provider_registry().unwrap();
    let root = tempfile::tempdir().unwrap();
    let candidate = build_candidate(
        &registry,
        "sana_1600m",
        sana(root.path()),
        None,
        MemoryStrategy::BoundedDecode,
        generation(512, vec![]),
    )
    .unwrap();
    let rows = [row(&registry, &candidate, 1 << 20)];
    let admitted = admit_with_rows(
        &registry,
        std::slice::from_ref(&candidate),
        DEVICE,
        budget(),
        &rows,
    )
    .unwrap();
    let calls = Rc::new(Cell::new(0));
    let headroom = Rc::new(Cell::new(true));
    let desc = registry
        .generators()
        .map(|r| (r.descriptor)())
        .find(|d| d.id == "sana_1600m")
        .unwrap();
    let probe_calls = calls.clone();
    let live = headroom.clone();
    let bound = admitted
        .load_generator_using(
            move |_| {
                Ok(Box::new(ProbeGenerator {
                    desc,
                    calls: probe_calls,
                }))
            },
            Box::new(move |_| {
                if live.get() {
                    Ok(())
                } else {
                    Err(Error::Unsupported("live headroom".into()))
                }
            }),
        )
        .unwrap();
    let IosWork::Generate(req) = &candidate.work else {
        unreachable!()
    };
    let mut altered = Vec::new();
    let mut r = req.clone();
    r.width = 1024;
    r.height = 1024;
    altered.push(r);
    let mut r = req.clone();
    r.memory.as_mut().unwrap().tile_vae_decode = false;
    altered.push(r);
    let mut r = req.clone();
    let tile = r.memory.unwrap().decode_tile_edge.unwrap();
    r.memory.as_mut().unwrap().decode_tile_edge = Some(tile * 2);
    altered.push(r);
    let mut r = req.clone();
    r.conditioning.push(Conditioning::Reference {
        image: image(512),
        strength: Some(0.5),
    });
    altered.push(r);
    for r in &altered {
        assert!(bound.generate(r, &mut |_| {}).is_err());
    }
    assert_eq!(calls.get(), 0, "an altered request reached the provider");
    bound.generate(req, &mut |_| {}).unwrap();
    assert_eq!(calls.get(), 1);
    headroom.set(false);
    assert!(bound.generate(req, &mut |_| {}).is_err());
    assert_eq!(calls.get(), 1, "a capacity recheck must precede every call");
}

struct ProbeTransform(TransformDescriptor, Rc<Cell<u32>>);
impl Transform for ProbeTransform {
    fn descriptor(&self) -> &TransformDescriptor {
        &self.0
    }
    fn validate(&self, _: &TransformRequest) -> Result<()> {
        self.1.set(self.1.get() + 1);
        Ok(())
    }
    fn apply(&self, req: &TransformRequest, _: &mut dyn FnMut(Progress)) -> Result<Image> {
        self.1.set(self.1.get() + 1);
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        Ok(req.image.clone())
    }
}

struct ProbeDepth(Rc<Cell<u32>>);
impl IrisDepthEstimator for ProbeDepth {
    fn backend(&self) -> &'static str {
        "mlx"
    }
    fn estimate(&self, _: &DepthRequest, _: &mut dyn FnMut(Progress)) -> Result<DepthOutput> {
        self.0.set(self.0.get() + 1);
        Err(Error::Unsupported("probe".into()))
    }
}

/// Restoration and depth are admitted-reachable through the same evidence keying, and their bound
/// wrappers refuse an altered request before the provider.
#[test]
fn restore_and_depth_are_keyed_admitted_and_request_bound() {
    let registry = crate::provider_registry().unwrap();
    let restore_spec = LoadSpec::new(WeightsSource::Dir(
        iris_fixtures().join("tiny-snapshot/upscaler"),
    ));
    let depth_spec = LoadSpec::new(WeightsSource::Dir(iris_fixtures().join("tiny-depth")));
    let restore = build_candidate(
        &registry,
        IRIS_RESTORE_ID,
        restore_spec,
        None,
        MemoryStrategy::Resident,
        IosWork::Restore(TransformRequest {
            image: image(16),
            ..Default::default()
        }),
    )
    .unwrap();
    let depth = build_candidate(
        &registry,
        IRIS_DEPTH_ID,
        depth_spec,
        None,
        MemoryStrategy::Resident,
        IosWork::Depth(DepthRequest {
            image: image(16),
            ..Default::default()
        }),
    )
    .unwrap();
    assert_eq!(restore.context.mode.as_key(), "restoration");
    assert_eq!(depth.context.mode.as_key(), "depth");
    for c in [&restore, &depth] {
        assert_eq!(
            kind(admit(&registry, std::slice::from_ref(c), DEVICE, budget())),
            Some(RefusalKind::NoEvidence)
        );
        // Larger input is a different key.
        let key = evidence_key(&registry, c, DEVICE).unwrap();
        let mut bigger = c.clone();
        match &mut bigger.work {
            IosWork::Restore(r) => r.image = image(32),
            IosWork::Depth(r) => r.image = image(32),
            _ => unreachable!(),
        }
        bigger.context.geometry.width = 32;
        bigger.context.geometry.height = 32;
        assert_ne!(
            evidence_key(&registry, &bigger, DEVICE).unwrap().width,
            key.width
        );
    }
    let calls = Rc::new(Cell::new(0));
    let rows = [row(&registry, &restore, 1 << 20)];
    let admitted = admit_with_rows(
        &registry,
        std::slice::from_ref(&restore),
        DEVICE,
        budget(),
        &rows,
    )
    .unwrap();
    let descriptor = registry
        .transforms()
        .map(|r| (r.descriptor)())
        .find(|d| d.id == IRIS_RESTORE_ID)
        .unwrap();
    let bound = crate::bound::BoundTransform::new(
        Box::new(ProbeTransform(descriptor, calls.clone())),
        admitted.spec.clone(),
        admitted.binding,
        admitted.required_bytes(),
        Box::new(|_| Ok(())),
        None,
    );
    let IosWork::Restore(req) = &restore.work else {
        unreachable!()
    };
    let mut other = req.clone();
    other.image = image(32);
    assert!(bound.apply(&other, &mut |_| {}).is_err());
    assert!(bound.validate(&other).is_err());
    assert_eq!(calls.get(), 0);
    let mut cancelled = req.clone();
    cancelled.cancel = Default::default();
    cancelled.cancel.cancel();
    assert!(matches!(
        bound.apply(&cancelled, &mut |_| {}),
        Err(Error::Canceled)
    ));
    bound.apply(req, &mut |_| {}).unwrap();
    assert_eq!(calls.get(), 1);

    let rows = [row(&registry, &depth, 1 << 20)];
    let admitted = admit_with_rows(
        &registry,
        std::slice::from_ref(&depth),
        DEVICE,
        budget(),
        &rows,
    )
    .unwrap();
    let calls = Rc::new(Cell::new(0));
    let bound = crate::bound::BoundDepth::new(
        Box::new(ProbeDepth(calls.clone())),
        admitted.spec.clone(),
        admitted.binding,
        admitted.required_bytes(),
        Box::new(|_| Ok(())),
        None,
    );
    let IosWork::Depth(req) = &depth.work else {
        unreachable!()
    };
    let mut other = req.clone();
    other.image.pixels[0] ^= 1;
    assert!(bound.estimate(&other, &mut |_| {}).is_err());
    assert_eq!(calls.get(), 0);
    let _ = bound.estimate(req, &mut |_| {});
    assert_eq!(calls.get(), 1);
}

/// Finding 1: the capture permit is NON-circular (no evidence row consulted, the table stays
/// empty) but only exists where quanta are bounded; it is labelled and never an admission.
#[cfg(feature = "device-capture")]
#[test]
fn capture_permit_needs_bounded_quanta_not_evidence() {
    let registry = crate::provider_registry().unwrap();
    let root = tempfile::tempdir().unwrap();
    let candidate = build_candidate(
        &registry,
        "sana_1600m",
        sana(root.path()),
        None,
        MemoryStrategy::BoundedDecode,
        generation(512, vec![]),
    )
    .unwrap();
    assert!(matches!(
        capture_permit(&registry, &candidate, DEVICE),
        Err(AdmissionError::Failed(_))
    ));
    assert_eq!(
        kind(admit(
            &registry,
            std::slice::from_ref(&candidate),
            DEVICE,
            budget()
        )),
        Some(RefusalKind::NoEvidence)
    );
    // The bounded-quantum CPU fixture stands in for a device's live headroom. No MLX op runs.
    let child = std::env::var_os("IOS_CAPTURE_PERMIT_CPU_CHILD").is_some();
    if !child {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "ios_admission::tests::capture_permit_needs_bounded_quanta_not_evidence",
            ])
            .env("IOS_CAPTURE_PERMIT_CPU_CHILD", "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    let permit = mlx_gen::ios_quantum::with_cpu_diagnostic_headroom(1 << 30, || {
        Ok(capture_permit(&registry, &candidate, DEVICE).map_err(|e| e.to_string()))
    })
    .unwrap()
    .unwrap();
    assert!(permit.is_capture());
    assert_eq!(permit.required_bytes(), 0);
    assert_eq!(permit.basis(), AdmissionBasis::CapturePermit);
    assert_eq!(
        permit.evidence_key(),
        &capture_evidence_key(&registry, &candidate, DEVICE).unwrap()
    );
    // A permit still refuses what admission would refuse for the request itself.
    let mut wrong = candidate.clone();
    wrong.context.selection.strategy = MemoryStrategy::BoundedTransformerResidency;
    let refused = mlx_gen::ios_quantum::with_cpu_diagnostic_headroom(1 << 30, || {
        Ok(capture_permit(&registry, &wrong, DEVICE)
            .err()
            .and_then(|e| e.refusal_kind()))
    })
    .unwrap();
    assert_eq!(refused, Some(RefusalKind::Strategy));
}

/// E4: the anchor's peak is the CAPTURE process's absolute peak, including its own baseline. Only
/// the render's growth (peak - capture baseline) plus the anchor's measured margin is charged, and
/// what this process already holds is committed, never credited: an unrelated resident (e.g.
/// another generator) makes a load that would fit an empty process a typed capacity refusal.
#[test]
fn an_unrelated_resident_is_never_credited_as_reusable_headroom() {
    in_cpu_child(
        "ios_admission::tests::an_unrelated_resident_is_never_credited_as_reusable_headroom",
        || {
            let registry = crate::provider_registry().unwrap();
            let root = crate::sdxl::tests::fixture();
            let candidate = crate::sdxl::tests::candidate(
                &registry,
                root.path(),
                "sdxl",
                512,
                Some(mlx_gen::gen_core::Quant::Q4),
                "text_to_image",
            );
            const MIB: u64 = 1 << 20;
            let mut row = row(&registry, &candidate, 0);
            // The harness's own baseline at capture start (200 MiB) and its lifetime peak (3600 MiB).
            if let evidence::AnchorOutcome::Completed { captures } = &mut row.outcome {
                for c in captures.iter_mut() {
                    evidence::fixture::set_kernel_footprints(c, 200 * MIB, 3600 * MIB);
                }
            }
            let rows = [row];
            let budget = |held: u64, available: u64| {
                capacity_budget(6142 * MIB, held * MIB, Some(available * MIB))
            };
            // A fresh process (200 MiB held, 5900 MiB available): 3400 MiB growth + 32 MiB margin fits.
            let admitted = admit_with_rows(
                &registry,
                std::slice::from_ref(&candidate),
                DEVICE,
                budget(200, 5900),
                &rows,
            )
            .unwrap();
            assert_eq!(admitted.demand_bytes(), 3400 * MIB);
            assert_eq!(admitted.margin_bytes(), evidence::ANCHOR_MARGIN_FLOOR_BYTES);
            assert_eq!(
                admitted.basis(),
                AdmissionBasis::Measured(AnchorKind::Optimized)
            );
            // SANA (~2 GB) still resident: 2400 MiB available. The absolute-peak arithmetic would credit
            // the resident 2200 MiB (3600 - 2200 = 1400 <= 2400) and admit; the growth is 3400: refused.
            assert_eq!(
                kind(admit_with_rows(
                    &registry,
                    std::slice::from_ref(&candidate),
                    DEVICE,
                    budget(2200, 2400),
                    &rows
                )),
                Some(RefusalKind::Capacity)
            );
            // The margin is part of the boundary: growth + margin must fit, not growth alone.
            assert_eq!(
                kind(admit_with_rows(
                    &registry,
                    std::slice::from_ref(&candidate),
                    DEVICE,
                    budget(200, 3400 + 32 - 1),
                    &rows
                )),
                Some(RefusalKind::Capacity)
            );
            admit_with_rows(
                &registry,
                std::slice::from_ref(&candidate),
                DEVICE,
                budget(200, 3400 + 32),
                &rows,
            )
            .unwrap();
        },
    );
}

/// Positive exact-key admission must use the SDK BML ledger, not the older pre-request scalar.
#[test]
fn exact_key_uses_before_model_load_growth_and_rejects_masked_repeats() {
    in_cpu_child(
        "ios_admission::tests::exact_key_uses_before_model_load_growth_and_rejects_masked_repeats",
        || {
            const MIB: u64 = 1 << 20;
            let registry = crate::provider_registry().unwrap();
            let root = crate::sdxl::tests::fixture();
            let candidate = crate::sdxl::tests::candidate(
                &registry,
                root.path(),
                "sdxl",
                512,
                None,
                "text_to_image",
            );
            // This is an admission/header-contract fixture, not a runnable SDXL model. An
            // actually empty overlay must not enumerate unused base adapter targets, while ANY
            // active overlay must still use the real exporter and reject the missing targets.
            assert_eq!(
                crate::layout::configured_variant_digest("sdxl", &candidate.spec).unwrap(),
                None
            );
            let source = root.path().join("unet/diffusion_pytorch_model.safetensors");
            let missing = "down_blocks.1.attentions.0.proj_in.weight";
            // The shared positive fixture now supplies the complete inspected target graph.
            // Give this negative control its OWN graph, omitting exactly the sentinel below;
            // never damage the positive exact-key fixture or weaken the exporter assertion.
            let incomplete = crate::sdxl::tests::fixture();
            let incomplete_source = incomplete
                .path()
                .join("unet/diffusion_pytorch_model.safetensors");
            let mut headers =
                mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&incomplete_source)
                    .unwrap();
            let original_count = headers.len();
            headers.retain(|header| header.name != missing);
            assert_eq!(
                headers.len() + 1,
                original_count,
                "remove exactly one target"
            );
            crate::sdxl::tests::write_header_fixture(
                &incomplete_source,
                headers
                    .into_iter()
                    .map(|header| (header.name, header.shape)),
            );
            let quality_artifact = || mlx_gen::gen_core::MemoryDecodeArtifactIdentity {
                repository: "synthetic-negative-control".into(),
                revision: "not-measured".into(),
                variant: "presence-only".into(),
                fingerprint: "not-a-production-receipt".into(),
            };
            // Invoke ONLY variant dispatch: no artifact/key validation can mask an empty-guard
            // defect. Every case changes exactly ONE empty slot; removing that slot's predicate
            // returns Null/None and fails expect_err instead of reaching the target-table sentinel.
            for slot in [
                "lora",
                "zero_lora",
                "control",
                "extra_control",
                "ip_adapter",
                "vae_override",
                "unknown_component",
                "identity",
                "text_encoder",
                "quality_table",
                "quality_binding",
            ] {
                let mut configured = candidate.spec.clone();
                configured.weights = WeightsSource::Dir(incomplete.path().to_owned());
                match slot {
                    "lora" | "zero_lora" => {
                        configured.adapters.push(mlx_gen::gen_core::AdapterSpec {
                            path: source.clone(),
                            scale: if slot == "zero_lora" { 0.0 } else { 1.0 },
                            kind: mlx_gen::gen_core::AdapterKind::Lora,
                            pass_scales: None,
                            moe_expert: None,
                        })
                    }
                    "control" => configured.control = Some(WeightsSource::File(source.clone())),
                    "extra_control" => configured
                        .extra_controls
                        .push(WeightsSource::File(source.clone())),
                    "identity" => {
                        configured.identity = Some(mlx_gen::gen_core::IdentityWeights {
                            encoder: Some(WeightsSource::File(source.clone())),
                            eva: None,
                            face_dir: None,
                        })
                    }
                    "text_encoder" => {
                        configured.text_encoder = Some(WeightsSource::File(source.clone()))
                    }
                    "unknown_component" => {
                        configured.components.insert(
                            "unknown-negative-control".into(),
                            WeightsSource::File(source.clone()),
                        );
                    }
                    "quality_binding" => {
                        configured.decode_quality_runtime_identity =
                            Some(mlx_gen::gen_core::MemoryDecodeQualityRuntimeIdentity {
                                artifact: quality_artifact(),
                            })
                    }
                    "quality_table" => {
                        // An explicitly REFUSED/invalid synthetic row tests only table presence;
                        // it is never passed off as an admitted quality or physical-memory record.
                        configured.decode_geometry_policies.push(
                            mlx_gen::gen_core::MemoryDecodeGeometryPolicy {
                                quality_abi: 0,
                                family: "negative-control".into(),
                                resolved_route: "sdxl".into(),
                                backend: mlx_gen::gen_core::MemoryBackend::Mlx,
                                tier: candidate.context.selection.tier,
                                load_shape: candidate.spec.load_shape,
                                artifact: quality_artifact(),
                                implementation_fingerprint: "not-measured".into(),
                                mode: MemoryMode::TextToImage,
                                overlay: None,
                                geometry: candidate.context.geometry,
                                use_pid: false,
                                tile_edge: 1,
                                overlap: 0,
                                metric: "presence-only".into(),
                                maximum_error: 0,
                                fixtures: vec![],
                                production_evidence_sha256: String::new(),
                                disposition:
                                    mlx_gen::gen_core::MemoryDecodeQualityDisposition::Refused {
                                        reason:
                                            "negative presence fixture, not a quality admission"
                                                .into(),
                                    },
                            },
                        );
                    }
                    "ip_adapter" => {
                        configured.ip_adapter = Some(WeightsSource::Dir(root.path().to_owned()))
                    }
                    "vae_override" => {
                        configured.components.insert(
                            mlx_gen_sdxl::loader::VAE_FP16_FIX_COMPONENT.into(),
                            WeightsSource::File(source.clone()),
                        );
                    }
                    _ => unreachable!(),
                }
                let error = crate::layout::configured_variant_digest("sdxl", &configured)
                    .expect_err("active overlay must validate the actual base target graph");
                assert!(error.to_string().contains(missing), "{slot}: {error}");
            }
            let mut sealed = candidate.spec.clone();
            sealed.decode_geometry_policy_authoritative = true;
            assert!(matches!(
                crate::layout::configured_variant_digest("sdxl", &sealed),
                Err(Error::Unsupported(_))
            ));
            let mut pid = candidate.spec.clone();
            pid.pid = Some(mlx_gen::gen_core::PidWeights {
                checkpoint: WeightsSource::File(source),
                gemma: WeightsSource::Dir(root.path().to_owned()),
            });
            assert!(matches!(
                crate::layout::configured_variant_digest("sdxl", &pid),
                Err(Error::Unsupported(_))
            ));
            // Empty-overlay identity never substitutes for the separate base-artifact checks.
            let malformed = tempfile::tempdir().unwrap();
            let file = malformed.path().join("invalid.safetensors");
            std::fs::write(&file, b"invalid base header").unwrap();
            let malformed = LoadSpec::new(WeightsSource::File(file));
            assert_eq!(
                crate::layout::configured_variant_digest("sdxl", &malformed).unwrap(),
                None
            );
            assert!(crate::layout::artifact_layout_digest(&malformed).is_err());
            assert!(crate::layout::architecture_digest(&malformed).is_err());

            let mut anchor = row(&registry, &candidate, 256 * MIB);
            if let evidence::AnchorOutcome::Completed { captures } = &mut anchor.outcome {
                for capture in captures {
                    evidence::fixture::set_kernel_footprints(capture, 128 * MIB, 384 * MIB);
                    let bml = capture.before_model_load_snapshot.as_mut().unwrap();
                    bml.current_bytes = Some(192 * MIB);
                    bml.lifetime_peak_bytes = Some(192 * MIB);
                }
            }
            let candidates = std::slice::from_ref(&candidate);
            let admitted =
                admit_with_rows(&registry, candidates, DEVICE, budget(), &[anchor.clone()])
                    .unwrap();
            assert_eq!(admitted.demand_bytes(), 192 * MIB);
            assert_eq!(admitted.margin_bytes(), evidence::ANCHOR_MARGIN_FLOOR_BYTES);
            assert_eq!(
                admitted.required_bytes(),
                192 * MIB + evidence::ANCHOR_MARGIN_FLOOR_BYTES
            );
            assert_eq!(
                admitted.context().predicted_peak_bytes,
                admitted.required_bytes()
            );
            assert_eq!(
                admitted.basis(),
                AdmissionBasis::Measured(AnchorKind::Optimized)
            );

            // One repeat's BML highwater was already at the ending highwater, never attained by a
            // current-footprint observation. Its pre-request ledger is still exact: substituting that
            // ledger or dropping this repeat would wrongly admit the exact-key reference.
            if let evidence::AnchorOutcome::Completed { captures } = &mut anchor.outcome {
                captures[1]
                    .before_model_load_snapshot
                    .as_mut()
                    .unwrap()
                    .lifetime_peak_bytes = Some(384 * MIB);
                captures[1].observed_model_current_peak_bytes = Some(320 * MIB);
                captures[1].ending_snapshot.as_mut().unwrap().current_bytes = Some(320 * MIB);
            }
            assert_eq!(
                kind(admit_with_rows(
                    &registry,
                    candidates,
                    DEVICE,
                    budget(),
                    &[anchor]
                )),
                Some(RefusalKind::NoEvidence),
            );
        },
    );
}

#[test]
fn exact_key_requires_actual_positive_matching_request_boundary() {
    in_cpu_child(
        "ios_admission::tests::exact_key_requires_actual_positive_matching_request_boundary",
        || {
            let registry = crate::provider_registry().unwrap();
            let root = crate::sdxl::tests::fixture();
            let candidate = crate::sdxl::tests::candidate(
                &registry,
                root.path(),
                "sdxl",
                512,
                None,
                "text_to_image",
            );
            let rows = [row(&registry, &candidate, 64 << 20)];
            let candidates = std::slice::from_ref(&candidate);
            let budget = budget();
            let baseline = IosMemoryBaseline {
                domain: IosMemoryDomain::ProcessGrowth,
                coordinate: IosMemoryCoordinate::BeforeModelLoad,
                bytes: budget.committed_bytes,
                source: "explicit-synthetic-sdk-bml".into(),
            };
            let admit = |budget, boundary: Option<&IosMemoryBaseline>| {
                admit_rows_at_boundary(&registry, candidates, DEVICE, budget, &rows, boundary)
            };
            assert!(admit(budget, Some(&baseline)).is_ok());
            assert_eq!(
                kind(super::admit_with_rows(
                    &registry, candidates, DEVICE, budget, &rows
                )),
                Some(RefusalKind::NoEvidence),
            );
            // The zero control deliberately matches a ZERO committed budget, so it isolates
            // missing positive baseline evidence rather than merely tripping the mismatch check.
            // This invalid case never goes through the positive-fixture helper.
            let zero_budget = MemoryBudget {
                committed_bytes: 0,
                ..budget
            };
            for (case, request_budget, invalid) in [
                (
                    "zero",
                    zero_budget,
                    IosMemoryBaseline {
                        bytes: 0,
                        ..baseline.clone()
                    },
                ),
                (
                    "mismatch",
                    budget,
                    IosMemoryBaseline {
                        bytes: baseline.bytes + 1,
                        ..baseline.clone()
                    },
                ),
                (
                    "coordinate",
                    budget,
                    IosMemoryBaseline {
                        coordinate: IosMemoryCoordinate::BeforeRequestConstruction,
                        ..baseline.clone()
                    },
                ),
                (
                    "domain",
                    budget,
                    IosMemoryBaseline {
                        domain: IosMemoryDomain::NativeAllocatorGrowth,
                        ..baseline.clone()
                    },
                ),
                (
                    "provenance",
                    budget,
                    IosMemoryBaseline {
                        source: String::new(),
                        ..baseline.clone()
                    },
                ),
            ] {
                assert_eq!(
                    kind(admit(request_budget, Some(&invalid))),
                    Some(RefusalKind::NoEvidence),
                    "{case}"
                );
            }
        },
    );
}

#[test]
fn complete_envelope_overflow_is_refused_before_seal_and_valid_sum_is_preserved() {
    in_cpu_child("ios_admission::tests::complete_envelope_overflow_is_refused_before_seal_and_valid_sum_is_preserved", || {
    let registry = crate::provider_registry().unwrap();
    let root = crate::sdxl::tests::fixture();
    let candidate =
        crate::sdxl::tests::candidate(&registry, root.path(), "sdxl", 512, None, "text_to_image");
    let candidates = std::slice::from_ref(&candidate);
    let budget = capacity_budget(u64::MAX, 1, None);
    let margin = evidence::ANCHOR_MARGIN_FLOOR_BYTES;
    let overflowing = [row(&registry, &candidate, u64::MAX - margin + 1)];
    let error = admit_with_rows(&registry, candidates, DEVICE, budget, &overflowing)
        .expect_err("demand fits u64 but demand + margin does not");
    assert_eq!(error.refusal_kind(), Some(RefusalKind::NoEvidence));
    assert!(error
        .to_string()
        .contains("complete admitted envelope arithmetic"));

    let demand = u64::MAX - margin - budget.committed_bytes;
    let valid = [row(&registry, &candidate, demand)];
    let admitted = admit_with_rows(&registry, candidates, DEVICE, budget, &valid).unwrap();
    assert_eq!(admitted.demand_bytes(), demand);
    assert_eq!(admitted.margin_bytes(), margin);
    assert_eq!(
        admitted.required_bytes(),
        demand.checked_add(margin).unwrap()
    );
    assert_eq!(admitted.required_bytes(), budget.effective_bytes());
    assert_eq!(
        admitted.context().predicted_peak_bytes,
        admitted.required_bytes()
    );
    });
}

#[test]
fn core_pricing_errors_preserve_typed_refusal_vs_failure() {
    let unsupported = pricing_error(Error::Unsupported("unpriced configured graph".into()));
    assert_eq!(unsupported.refusal_kind(), Some(RefusalKind::NoEvidence));
    assert!(unsupported
        .to_string()
        .contains("unpriced configured graph"));
    let failed = pricing_error(Error::Msg("unreadable source header".into()));
    assert!(matches!(failed, AdmissionError::Failed(_)));
    assert!(failed.to_string().contains("unreadable source header"));
}

#[test]
fn backend_pricing_errors_preserve_typed_refusal_vs_failure() {
    let unsupported = pricing_error(mlx_gen::Error::Unsupported(
        "unpriced configured graph".into(),
    ));
    assert_eq!(unsupported.refusal_kind(), Some(RefusalKind::NoEvidence));
    assert!(unsupported
        .to_string()
        .contains("unpriced configured graph"));
    let failed = pricing_error(mlx_gen::Error::Msg("unreadable source header".into()));
    assert!(matches!(failed, AdmissionError::Failed(_)));
    assert!(failed.to_string().contains("unreadable source header"));
}

#[test]
fn actual_output_cardinality_is_checked_never_clamped_or_promoted_from_zero() {
    let mut request = GenerationRequest {
        count: 3,
        ..Default::default()
    };
    assert_eq!(generation_batch(&request).unwrap(), 3);
    request.prompt_batch = vec!["first".into(), "second".into()];
    assert_eq!(generation_batch(&request).unwrap(), 6);
    request.count = 0;
    assert!(
        matches!(generation_batch(&request), Err(AdmissionError::Refused(refusal)) if refusal.kind == RefusalKind::Mode)
    );
    request.count = u32::MAX;
    assert!(
        matches!(generation_batch(&request), Err(AdmissionError::Refused(refusal)) if refusal.kind == RefusalKind::Mode)
    );
    request.prompt_batch.clear();
    assert_eq!(generation_batch(&request).unwrap(), u32::MAX);
}
