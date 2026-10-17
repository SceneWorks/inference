//! The shared seam across providers/modes: typed mode refusals, SANA's request-bound wrapper,
//! the bound transform/depth wrappers, and the non-circular capture permit.
use super::*;
use crate::sdxl::tests::{budget, DEVICE};
use mlx_gen::gen_core::iris::depth::{DepthOutput, IrisDepthEstimator};
use mlx_gen::gen_core::{
    GenerationOutput, Generator, Image, MemoryStrategy, Progress, Transform, TransformDescriptor,
    WeightsSource,
};
use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;

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

fn row(registry: &ProviderRegistry, c: &IosCandidate, bytes: u64) -> EvidenceRow {
    EvidenceRow {
        key: evidence_key(registry, c, DEVICE).unwrap(),
        footprint_bytes: bytes,
        // Synthetic rows: a zero capture baseline makes the demand the whole figure.
        baseline_footprint_bytes: 0,
        mlx_peak_bytes: bytes,
        min_headroom_mib: 1.0,
        elapsed_seconds: 1.0,
        evidence_class: evidence::DEVICE_CAPTURE_CLASS.into(),
        run_id: "synthetic".into(),
        build_sha: "synthetic".into(),
    }
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
        admitted.spec,
        admitted.binding,
        admitted.footprint_bytes,
        Box::new(|_| Ok(())),
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
        admitted.spec,
        admitted.binding,
        admitted.footprint_bytes,
        Box::new(|_| Ok(())),
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
    assert_eq!(permit.footprint_bytes(), 0);
    assert_eq!(
        permit.evidence_key(),
        &evidence_key(&registry, &candidate, DEVICE).unwrap()
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

/// E4: the evidence peak is the CAPTURE process's absolute peak, including its own baseline. Only
/// the render's growth (peak - capture baseline) is charged, and what this process already holds
/// is committed, never credited: an unrelated resident (e.g. another generator) makes a load that
/// would fit an empty process a typed capacity refusal.
#[test]
fn an_unrelated_resident_is_never_credited_as_reusable_headroom() {
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
    let mut row = row(&registry, &candidate, 3600 * MIB);
    row.baseline_footprint_bytes = 200 * MIB; // the harness's own baseline at capture start
    let rows = [row];
    let budget = |held: u64, available: u64| {
        let mut b = capacity_budget(6142 * MIB, held * MIB, Some(available * MIB));
        b.reserved_headroom_bytes = 512 * MIB;
        b
    };
    // A fresh process (200 MiB held, 5900 MiB available): 3400 MiB growth + reserve fits.
    let admitted = admit_with_rows(
        &registry,
        std::slice::from_ref(&candidate),
        DEVICE,
        budget(200, 5900),
        &rows,
    )
    .unwrap();
    assert_eq!(admitted.demand_bytes(), 3400 * MIB);
    // SANA (~2 GB) still resident: 2400 MiB available (1888 after the reserve). The absolute-peak
    // arithmetic would credit the resident 2200 MiB (3600 - 2200 = 1400 <= 1888) and admit; the
    // growth is 3400 MiB: refused.
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
    // The reserve is part of the boundary: growth + reserve must fit, not growth alone.
    assert_eq!(
        kind(admit_with_rows(
            &registry,
            std::slice::from_ref(&candidate),
            DEVICE,
            budget(200, 3400 + 511),
            &rows
        )),
        Some(RefusalKind::Capacity)
    );
    admit_with_rows(
        &registry,
        std::slice::from_ref(&candidate),
        DEVICE,
        budget(200, 3400 + 512),
        &rows,
    )
    .unwrap();
}
