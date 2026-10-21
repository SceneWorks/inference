//! SDXL-family iOS composition (sc-26268). Five checkpoint routes share the SAME `sdxl` generator
//! and every one of its modes (text, image-to-image, edit/inpaint, control, IP-adapter).
//!
//! Admission is the shared seam in [`crate::ios_admission`]: evidence is keyed per route, tier,
//! path-free artifact layout, mode, geometry, reference count, execution levers (including
//! sequential CFG) and the encoded CLIP token window, from the single reviewed table. Mac MLX
//! allocator measurements are engine evidence, not iPhone footprint evidence, and are never rows.

use mlx_gen::gen_core::{
    CfgBatching, Error, GenerationMemory, LoadShape, LoadSpec, Result, WeightsSource,
};

/// The production host-only recipe resolver, shared by capture, evidence and execution.
pub use mlx_gen_sdxl::sampling;

pub const SDXL_ID: &str = "sdxl";
/// Catalog routes, not additional provider ids. Never substitute one sibling's capture for another.
pub const SDXL_FAMILY_ROUTES: &[&str] = &[
    "sdxl",
    "illustrious_xl_v1",
    "illustrious_xl_v2",
    "realvisxl",
    "realvisxl_lightning",
];

/// No tier is silently selected: the caller retains its requested precision/quantization. All
/// supported tiers and auxiliary components still flow through the original provider loader.
pub fn sdxl_load_spec(route: &str, source: WeightsSource) -> Result<LoadSpec> {
    if !SDXL_FAMILY_ROUTES.contains(&route) {
        return Err(Error::Unsupported(format!(
            "unknown SDXL family route {route}"
        )));
    }
    let mut spec = LoadSpec::new(source)
        .with_offload_policy(crate::IOS_OFFLOAD_POLICY)
        .with_load_shape(LoadShape::DeferredMaterialization);
    spec.resolved_route = Some(route.into());
    Ok(spec)
}

/// Low-memory execution, not an asserted device fit. The caller stages the optional fp16-fix VAE
/// via `LoadSpec::components`; it is never fetched or silently added by the catalog.
pub fn sdxl_generation_memory() -> GenerationMemory {
    GenerationMemory {
        stage_residency: true,
        tile_vae_decode: true,
        decode_tile_edge: Some(256),
        decode_overlap: Some(mlx_gen_sdxl::memory_strategy::DECODE_OVERLAP),
        cfg_batching: Some(CfgBatching::Sequential),
        ..Default::default()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::evidence::{AnchorKind, AnchorRow, AnchorSelection, ANCHOR_MARGIN_FLOOR_BYTES};
    use crate::ios_admission::{
        build_candidate, capacity_budget, evidence_key, IosCandidate, IosWork,
    };
    use crate::refusal::{AdmissionError, RefusalKind};
    use mlx_gen::gen_core::{
        Conditioning, GenerationRequest, Image, MemoryBudget, MemoryStrategy, ProviderRegistry,
        Quant,
    };
    use std::path::Path;

    pub(crate) const DEVICE: &str = "synthetic-test-device";
    /// Explicit positive unit-fixture footprint, NOT a device measurement or physical charge bound.
    pub(crate) const FIXTURE_BML_BYTES: u64 = 1;

    /// Synthetic request-boundary fact for positive admission mechanism tests. The caller must
    /// explicitly supply a positive committed budget; never synthesize a zero/default baseline
    /// or enlarge headroom. Production `admit_with_rows` remains the no-sidecar path.
    pub(crate) fn admit_fixture_rows(
        registry: &ProviderRegistry,
        candidates: &[IosCandidate],
        machine: &str,
        budget: MemoryBudget,
        rows: &[AnchorRow],
    ) -> crate::refusal::AdmissionResult<crate::Admission> {
        use mlx_gen::ios_pricing::{IosMemoryBaseline, IosMemoryCoordinate, IosMemoryDomain};
        assert!(
            budget.committed_bytes > 0,
            "fixture needs an explicit BML footprint"
        );
        let baseline = IosMemoryBaseline {
            domain: IosMemoryDomain::ProcessGrowth,
            coordinate: IosMemoryCoordinate::BeforeModelLoad,
            bytes: budget.committed_bytes,
            source: "synthetic-test-sdk-before-model-load".into(),
        };
        crate::ios_admission::admit_rows_at_boundary(
            registry,
            candidates,
            machine,
            budget,
            rows,
            Some(&baseline),
        )
    }

    use admit_fixture_rows as admit_with_rows;

    pub(crate) fn write_tensor(path: &Path, dtype: &str, count: usize) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let bytes = count * if dtype == "F32" { 4 } else { 2 };
        let header = format!(
            r#"{{"x":{{"dtype":"{dtype}","shape":[{count}],"data_offsets":[0,{bytes}]}}}}"#
        );
        let mut data = (header.len() as u64).to_le_bytes().to_vec();
        data.extend(header.as_bytes());
        data.resize(data.len() + bytes, 0);
        std::fs::write(path, data).unwrap();
    }

    /// A per-character CLIP vocabulary: enough for the real tokenizer to count windows.
    pub(crate) fn write_tokenizer(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let mut vocab = serde_json::Map::new();
        let mut id = 0;
        for c in (b'a'..=b'z').map(char::from) {
            vocab.insert(c.to_string(), id.into());
            vocab.insert(format!("{c}</w>"), (id + 1).into());
            id += 2;
        }
        vocab.insert("<|startoftext|>".into(), 49406.into());
        vocab.insert("<|endoftext|>".into(), 49407.into());
        std::fs::write(
            dir.join("vocab.json"),
            serde_json::Value::Object(vocab).to_string(),
        )
        .unwrap();
        std::fs::write(dir.join("merges.txt"), "#version: 0.2\n").unwrap();
    }

    pub(crate) fn fixture() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        populate(root.path());
        root
    }

    pub(crate) fn populate(root: &Path) {
        for component in ["unet", "vae", "text_encoder", "text_encoder_2"] {
            let name = if component.starts_with("text") {
                "model.safetensors"
            } else {
                "diffusion_pytorch_model.safetensors"
            };
            write_tensor(&root.join(component).join(name), "F16", 8);
        }
        write_tokenizer(&root.join("tokenizer"));
        write_tokenizer(&root.join("tokenizer_2"));
    }

    fn image() -> Image {
        Image {
            width: 1,
            height: 1,
            pixels: vec![128; 3],
        }
    }

    pub(crate) fn conditioning(mode: &str) -> Vec<Conditioning> {
        match mode {
            "image_to_image" | "ip_adapter" => vec![Conditioning::Reference {
                image: image(),
                strength: Some(0.5),
            }],
            "edit" => vec![
                Conditioning::Reference {
                    image: image(),
                    strength: Some(0.5),
                },
                Conditioning::Mask { image: image() },
            ],
            "control" => vec![Conditioning::Control {
                image: image(),
                kind: mlx_gen::gen_core::ControlKind::Other("tile".into()),
                scale: Some(1.0),
            }],
            _ => vec![],
        }
    }

    pub(crate) fn candidate(
        registry: &ProviderRegistry,
        root: &Path,
        route: &str,
        edge: u32,
        quant: Option<Quant>,
        mode: &str,
    ) -> IosCandidate {
        let mut spec = sdxl_load_spec(route, WeightsSource::Dir(root.to_owned())).unwrap();
        spec.quantize = quant;
        if mode == "ip_adapter" {
            let ip = root.join("ip/ip.safetensors");
            write_tensor(&ip, "F16", 8);
            spec.ip_adapter = Some(WeightsSource::File(ip));
        }
        let request = GenerationRequest {
            prompt: "synthetic fixture".into(),
            width: edge,
            height: edge,
            count: 1,
            conditioning: conditioning(mode),
            ..Default::default()
        };
        build_candidate(
            registry,
            SDXL_ID,
            spec,
            None,
            MemoryStrategy::BoundedDecode,
            IosWork::Generate(request),
        )
        .unwrap()
    }

    /// A synthetic completed anchor at `candidate`'s own key whose admission charge (measured
    /// demand + the floored margin) is exactly `bytes`.
    pub(crate) fn capture(
        registry: &ProviderRegistry,
        candidate: &IosCandidate,
        bytes: u64,
    ) -> AnchorRow {
        let kind = if candidate.context.selection.strategy == MemoryStrategy::Resident {
            AnchorKind::Resident
        } else {
            AnchorKind::Optimized
        };
        crate::evidence::fixture::completed(
            crate::source_work::fixture_key(registry, candidate, DEVICE),
            kind,
            AnchorSelection::from_parts(
                candidate.context.selection.strategy,
                &candidate.context.selection.parameters,
            ),
            bytes
                .checked_sub(ANCHOR_MARGIN_FLOOR_BYTES)
                .filter(|demand| *demand > 0)
                .expect("a completed fixture needs positive growth plus the explicit margin"),
        )
    }

    pub(crate) fn budget() -> MemoryBudget {
        // Mechanism fixture, NOT an iPhone cap or a fabricated measured model fit.
        MemoryBudget {
            total_bytes: 1 << 30,
            committed_bytes: FIXTURE_BML_BYTES,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        }
    }

    fn refusal(result: crate::refusal::AdmissionResult<crate::Admission>) -> RefusalKind {
        match result {
            Err(AdmissionError::Refused(r)) => r.kind,
            other => panic!("expected a typed refusal, got {other:?}"),
        }
    }

    fn generate(c: &mut IosCandidate) -> &mut GenerationRequest {
        match &mut c.work {
            IosWork::Generate(r) => r,
            _ => unreachable!(),
        }
    }

    #[test]
    fn evidence_key_ignores_prompt_seed_and_install_path_but_not_memory_facts() {
        let registry = crate::provider_registry().unwrap();
        let a = fixture();
        let b = fixture();
        let base = candidate(
            &registry,
            a.path(),
            "sdxl",
            512,
            Some(Quant::Q4),
            "text_to_image",
        );
        let key = evidence_key(&registry, &base, DEVICE).unwrap();
        // Same artifact elsewhere, different words with the same actual UTF8 byte basis/window, other seed.
        let mut moved = candidate(
            &registry,
            b.path(),
            "sdxl",
            512,
            Some(Quant::Q4),
            "text_to_image",
        );
        generate(&mut moved).prompt = "different fixture".into();
        let crate::IosWork::Generate(base_request) = &base.work else {
            unreachable!()
        };
        assert_eq!(generate(&mut moved).prompt.len(), base_request.prompt.len());
        generate(&mut moved).seed = Some(99);
        assert_eq!(evidence_key(&registry, &moved, DEVICE).unwrap(), key);
        let mut longer = moved.clone();
        generate(&mut longer).prompt.push(' ');
        let longer_key = evidence_key(&registry, &longer, DEVICE).unwrap();
        assert_eq!(longer_key.token_window, key.token_window);
        assert_ne!(
            longer_key, key,
            "same window does not reconstruct actual input bytes"
        );
        assert_eq!(longer_key.family(), key.family());
        // Each memory-relevant change is a different key.
        let differs = |c: &IosCandidate| evidence_key(&registry, c, DEVICE).unwrap() != key;
        assert!(differs(&candidate(
            &registry,
            a.path(),
            "sdxl",
            768,
            Some(Quant::Q4),
            "text_to_image"
        )));
        assert!(differs(&candidate(
            &registry,
            a.path(),
            "sdxl",
            512,
            Some(Quant::Q8),
            "text_to_image"
        )));
        assert!(differs(&candidate(
            &registry,
            a.path(),
            "realvisxl",
            512,
            Some(Quant::Q4),
            "text_to_image"
        )));
        assert!(differs(&candidate(
            &registry,
            a.path(),
            "sdxl",
            512,
            Some(Quant::Q4),
            "image_to_image"
        )));
        for (w, h) in [(768, 512), (512, 768)] {
            let mut aspect = base.clone();
            generate(&mut aspect).width = w;
            generate(&mut aspect).height = h;
            aspect.context.geometry.width = w;
            aspect.context.geometry.height = h;
            assert!(differs(&aspect), "{w}x{h} borrowed 512x512 evidence");
        }
        // Two modes with the same reference count, layout and execution differ only by mode.
        let img2img = candidate(
            &registry,
            a.path(),
            "sdxl",
            512,
            Some(Quant::Q4),
            "image_to_image",
        );
        let control = candidate(&registry, a.path(), "sdxl", 512, Some(Quant::Q4), "control");
        let (k1, k2) = (
            evidence_key(&registry, &img2img, DEVICE).unwrap(),
            evidence_key(&registry, &control, DEVICE).unwrap(),
        );
        assert_eq!(
            (k1.reference_count, &k1.layout_sha256, &k1.execution),
            (k2.reference_count, &k2.layout_sha256, &k2.execution)
        );
        assert_ne!(k1, k2, "control borrowed image-to-image evidence");
        let mut batch = base.clone();
        generate(&mut batch).count = 2;
        batch.context.geometry.batch = 2;
        assert!(differs(&batch));
        let mut long = base.clone();
        generate(&mut long).prompt = "a ".repeat(120);
        assert!(differs(&long), "a second CLIP window is a memory fact");
        assert_eq!(
            evidence_key(&registry, &long, DEVICE).unwrap().token_window,
            154
        );
        assert_eq!(key.token_window, 77);
        write_tensor(
            &b.path().join("unet/diffusion_pytorch_model.safetensors"),
            "F32",
            8,
        );
        assert!(
            differs(&moved),
            "a different tensor layout is a different artifact"
        );
        assert!(evidence_key(&registry, &base, "other-device").unwrap() != key);
    }

    #[test]
    fn source_identity_covers_fused_files_overrides_and_auxiliary_layouts() {
        let root = tempfile::tempdir().unwrap();
        let primary = root.path().join("fused.safetensors");
        let control = root.path().join("control.safetensors");
        let ip = root.path().join("ip.safetensors");
        let vae = root.path().join("vae.safetensors");
        for path in [&primary, &control, &ip, &vae] {
            write_tensor(path, "F16", 8);
        }
        let digest = crate::layout::artifact_layout_digest;
        let mut spec = sdxl_load_spec("illustrious_xl_v2", WeightsSource::File(primary)).unwrap();
        let plain = digest(&spec).unwrap();
        spec.control = Some(WeightsSource::File(control.clone()));
        spec.ip_adapter = Some(WeightsSource::File(ip));
        spec.components
            .insert("vae_fp16_fix".into(), WeightsSource::File(vae.clone()));
        let overlaid = digest(&spec).unwrap();
        assert_ne!(plain, overlaid);
        write_tensor(&control, "F32", 8);
        let changed_control = digest(&spec).unwrap();
        assert_ne!(overlaid, changed_control);
        write_tensor(&vae, "F32", 16);
        assert_ne!(changed_control, digest(&spec).unwrap());
    }

    #[test]
    fn budget_selection_picks_largest_offered_fit_and_refuses_next_as_capacity() {
        let registry = crate::provider_registry().unwrap();
        let root = fixture();
        let candidates = [
            candidate(&registry, root.path(), "sdxl", 1024, None, "text_to_image"),
            candidate(
                &registry,
                root.path(),
                "sdxl",
                768,
                Some(Quant::Q8),
                "text_to_image",
            ),
            candidate(
                &registry,
                root.path(),
                "sdxl",
                512,
                Some(Quant::Q4),
                "text_to_image",
            ),
        ];
        let cap = budget().effective_bytes();
        let rows = [
            capture(&registry, &candidates[0], cap + 1),
            capture(&registry, &candidates[1], cap + 1),
            capture(&registry, &candidates[2], cap),
        ];
        let admitted = admit_with_rows(&registry, &candidates, DEVICE, budget(), &rows).unwrap();
        assert_eq!((admitted.index(), admitted.required_mib()), (2, 1024));
        // The next size up is refused as CAPACITY, typed, never as a string to parse.
        assert_eq!(
            refusal(admit_with_rows(
                &registry,
                &candidates[..2],
                DEVICE,
                budget(),
                &rows
            )),
            RefusalKind::Capacity
        );
        // The production table carries no synthetic rows.
        assert_eq!(
            refusal(crate::ios_admission::admit(
                &registry,
                &candidates,
                DEVICE,
                budget()
            )),
            RefusalKind::NoEvidence
        );
        let mut committed = budget();
        committed.reserved_headroom_bytes = 1;
        assert_eq!(
            refusal(admit_with_rows(
                &registry,
                &candidates,
                DEVICE,
                committed,
                &rows
            )),
            RefusalKind::Capacity
        );
    }

    #[test]
    fn evidence_never_crosses_family_mode_artifact_tier_execution_or_hardware() {
        let registry = crate::provider_registry().unwrap();
        let root = fixture();
        for route in SDXL_FAMILY_ROUTES {
            for mode in [
                "text_to_image",
                "image_to_image",
                "edit",
                "control",
                "ip_adapter",
            ] {
                let base = candidate(&registry, root.path(), route, 512, Some(Quant::Q4), mode);
                let rows = [capture(&registry, &base, budget().total_bytes / 2)];
                let one = |c: IosCandidate, device: &str| {
                    admit_with_rows(&registry, &[c], device, budget(), &rows)
                };
                assert!(one(base.clone(), DEVICE).is_ok(), "{route}/{mode}");
                assert_eq!(
                    refusal(one(base.clone(), "other-device")),
                    RefusalKind::NoEvidence
                );
                let mut altered = base.clone();
                altered.spec.resolved_route = Some("different-route".into());
                assert_eq!(refusal(one(altered, DEVICE)), RefusalKind::Artifact);
                let mut altered = base.clone();
                altered.execution.cfg_batching = Some(CfgBatching::Batched);
                generate(&mut altered).memory = Some(altered.execution);
                assert_eq!(refusal(one(altered, DEVICE)), RefusalKind::NoEvidence);
                let mut altered = base.clone();
                altered.spec.quantize = Some(Quant::Q8);
                assert_eq!(refusal(one(altered, DEVICE)), RefusalKind::NoEvidence);
                for other in ["text_to_image", "image_to_image", "edit", "control"] {
                    if other != mode {
                        let c =
                            candidate(&registry, root.path(), route, 512, Some(Quant::Q4), other);
                        assert!(one(c, DEVICE).is_err(), "{mode} evidence admitted {other}");
                    }
                }
            }
        }
        let base = candidate(
            &registry,
            root.path(),
            "sdxl",
            512,
            Some(Quant::Q4),
            "text_to_image",
        );
        let rows = [capture(&registry, &base, budget().total_bytes / 2)];
        write_tensor(
            &root.path().join("unet/diffusion_pytorch_model.safetensors"),
            "F32",
            16,
        );
        assert_eq!(
            refusal(admit_with_rows(&registry, &[base], DEVICE, budget(), &rows)),
            RefusalKind::NoEvidence
        );
    }

    #[test]
    fn registered_standard_safety_and_actual_execution_cannot_be_bypassed_by_evidence() {
        let registry = crate::provider_registry().unwrap();
        let root = fixture();
        let mut base = candidate(
            &registry,
            root.path(),
            "sdxl",
            512,
            Some(Quant::Q4),
            "text_to_image",
        );
        let rows = [capture(&registry, &base, ANCHOR_MARGIN_FLOOR_BYTES + 1)];
        base.context.calibration_fingerprint = "wrong-provider-identity".into();
        assert_eq!(
            refusal(admit_with_rows(&registry, &[base], DEVICE, budget(), &rows)),
            RefusalKind::Strategy
        );
        let mut base = candidate(
            &registry,
            root.path(),
            "sdxl",
            512,
            Some(Quant::Q4),
            "text_to_image",
        );
        base.execution.tile_vae_decode = false;
        generate(&mut base).memory = Some(base.execution);
        let rows = [capture(&registry, &base, ANCHOR_MARGIN_FLOOR_BYTES + 1)];
        assert_eq!(
            refusal(admit_with_rows(&registry, &[base], DEVICE, budget(), &rows)),
            RefusalKind::Strategy
        );
        // A request that disagrees with its context is a failure, never a refusal.
        let mut base = candidate(
            &registry,
            root.path(),
            "sdxl",
            512,
            Some(Quant::Q4),
            "text_to_image",
        );
        let rows = [capture(&registry, &base, ANCHOR_MARGIN_FLOOR_BYTES + 1)];
        generate(&mut base).width = 768;
        assert!(matches!(
            admit_with_rows(&registry, &[base], DEVICE, budget(), &rows),
            Err(AdmissionError::Failed(_))
        ));
    }

    #[test]
    fn live_headroom_selects_fallback_with_absolute_process_accounting() {
        let registry = crate::provider_registry().unwrap();
        let root = fixture();
        let preferred = candidate(&registry, root.path(), "sdxl", 1024, None, "text_to_image");
        let fallback = candidate(
            &registry,
            root.path(),
            "sdxl",
            512,
            Some(Quant::Q4),
            "text_to_image",
        );
        let rows = [
            capture(&registry, &preferred, 3 << 30),
            capture(&registry, &fallback, 2 << 30),
        ];
        let both = [preferred.clone(), fallback.clone()];
        let live = capacity_budget(4 << 30, FIXTURE_BML_BYTES, Some(5 << 29)); // 4GiB fixed, 2.5GiB live
        assert_eq!(
            admit_with_rows(&registry, &both, DEVICE, live, &rows)
                .unwrap()
                .index(),
            1
        );
        // Captures are absolute process peaks; already committed bytes are removed exactly once.
        let held = capacity_budget(4 << 30, 1 << 29, Some(2 << 30));
        assert_eq!(
            admit_with_rows(&registry, &both, DEVICE, held, &rows)
                .unwrap()
                .index(),
            1
        );
    }

    #[test]
    fn malformed_preferred_candidate_is_a_refusal_not_a_global_error() {
        let registry = crate::provider_registry().unwrap();
        let root = fixture();
        let preferred = candidate(&registry, root.path(), "sdxl", 1024, None, "text_to_image");
        let fallback = candidate(
            &registry,
            root.path(),
            "sdxl",
            512,
            Some(Quant::Q4),
            "text_to_image",
        );
        let rows = [capture(&registry, &fallback, 2 << 30)];
        let mut missing = preferred.clone();
        missing.spec.weights = WeightsSource::Dir(root.path().join("missing"));
        let mut malformed = preferred.clone();
        let bad = tempfile::tempdir().unwrap();
        std::fs::write(bad.path().join("broken.safetensors"), b"bad").unwrap();
        malformed.spec.weights = WeightsSource::Dir(bad.path().to_owned());
        for first in [missing, malformed, preferred] {
            assert_eq!(
                admit_with_rows(
                    &registry,
                    &[first, fallback.clone()],
                    DEVICE,
                    capacity_budget(4 << 30, FIXTURE_BML_BYTES, None),
                    &rows
                )
                .unwrap()
                .index(),
                1
            );
        }
    }
}

#[cfg(test)]
mod execution_tests {
    use super::tests::admit_fixture_rows as admit_with_rows;
    use super::tests::{budget, candidate, capture, fixture, write_tensor, DEVICE};
    use super::*;
    use crate::ios_admission::{Admission, IosCandidate, IosWork};
    use mlx_gen::gen_core::*;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    type Calls = Rc<RefCell<Vec<&'static str>>>;
    struct Probe {
        desc: ModelDescriptor,
        contract: MemoryProviderContract,
        spec: LoadSpec,
        calls: Calls,
        consumed: Rc<Cell<u32>>,
    }
    impl Probe {
        fn mark(&self, name: &'static str) {
            self.calls.borrow_mut().push(name);
        }
    }
    impl Generator for Probe {
        fn descriptor(&self) -> &ModelDescriptor {
            self.mark("descriptor");
            &self.desc
        }
        fn veda_execution_report(&self) -> Option<VedaExecutionReport> {
            self.mark("veda_report");
            None
        }
        fn adapter_apply_reports(&self) -> Vec<AdapterApplyReport> {
            self.mark("adapters");
            vec![]
        }
        fn checkpoint_weight_facts(&self) -> Option<CheckpointWeightFacts> {
            self.mark("facts");
            None
        }
        fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
            self.mark("contract");
            Some(&self.contract)
        }
        fn structural_resident_evidence(&self) -> Option<&MemoryStructuralResidentEvidence> {
            self.mark("structural");
            None
        }
        fn predicted_memory_peak_from_base(&self, _: u64) -> MemoryPeakBreakdown {
            self.mark("prediction");
            MemoryPeakBreakdown::from_unattributed(123)
        }
        fn veda_workspace_bound(
            &self,
            _: veda::workspace::Request<'_>,
        ) -> Result<Option<veda::workspace::WorkspaceBound>> {
            self.mark("workspace");
            Ok(None)
        }
        fn memory_strategy_safety_check(&self, _: &MemoryRunContext) -> MemorySafetyDecision {
            self.mark("safety");
            MemorySafetyDecision::Accept
        }
        fn begin_memory_strategy_request(
            &self,
            context: &MemoryRunContext,
        ) -> Result<Option<Box<dyn MemoryRequestScope + '_>>> {
            self.mark("begin");
            Ok(Some(Box::new(ProbeScope(
                self.calls.clone(),
                self.contract.generation_memory(&context.selection),
            ))))
        }
        fn validate(&self, _: &GenerationRequest) -> Result<()> {
            self.mark("validate");
            Ok(())
        }
        fn generate(
            &self,
            req: &GenerationRequest,
            progress: &mut dyn FnMut(Progress),
        ) -> Result<GenerationOutput> {
            self.mark("generate");
            progress(Progress::Decoding);
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            // Models the production phase-reopen seam; the retained prepared spec, not a newly
            // profiled candidate, must reject replacement BEFORE consuming deferred bytes.
            self.spec.read_prepared_files_unchanged(|| {
                self.consumed.set(self.consumed.get() + 1);
                Ok(GenerationOutput::Images(vec![]))
            })
        }
        fn generate_streaming(
            &self,
            _: &GenerationRequest,
            chunk: &mut dyn FnMut(AudioChunk),
            _: &mut dyn FnMut(Progress),
        ) -> Result<GenerationOutput> {
            self.mark("streaming");
            chunk(AudioChunk {
                samples: vec![0.0],
                sample_rate: 1,
                channels: 1,
                index: 0,
            });
            Ok(GenerationOutput::Images(vec![]))
        }
        fn generate_with_report(
            &self,
            _: &GenerationRequest,
            _: &mut dyn FnMut(Progress),
        ) -> Result<GenerationReport> {
            self.mark("report");
            Ok(GenerationReport::from_output(GenerationOutput::Images(
                vec![],
            )))
        }
        fn open_conversation(
            &self,
            _: &GenerationRequest,
        ) -> Result<Box<dyn ConversationSession + '_>> {
            self.mark("conversation");
            Err(Error::Unsupported("probe conversation reached".into()))
        }
    }
    struct ProbeScope(Calls, Option<GenerationMemory>);
    impl MemoryRequestScope for ProbeScope {
        fn configure_request(&mut self, req: &mut GenerationRequest) -> Result<()> {
            self.0.borrow_mut().push("configure_request");
            req.memory = self.1;
            Ok(())
        }
        fn enter_phase(&mut self, _: MemoryPhase) -> Result<()> {
            self.0.borrow_mut().push("enter");
            Ok(())
        }
        fn leave_phase(&mut self, _: MemoryPhase) -> Result<()> {
            self.0.borrow_mut().push("leave");
            Ok(())
        }
        fn configure_decode(&mut self, _: u32, _: u32, _: MemoryGeometry) -> Result<()> {
            self.0.borrow_mut().push("decode");
            Ok(())
        }
        fn configure_attention(&mut self, _: u32) -> Result<()> {
            self.0.borrow_mut().push("attention");
            Ok(())
        }
        fn materialize_transformer_window(&mut self, _: u32, _: u32) -> Result<()> {
            self.0.borrow_mut().push("window");
            Ok(())
        }
        fn finish(&mut self, _: MemoryRunOutcome) -> Result<()> {
            self.0.borrow_mut().push("finish");
            Ok(())
        }
    }
    fn bind(
        receipt: Admission,
        registry: &ProviderRegistry,
        calls: Calls,
        consumed: Rc<Cell<u32>>,
    ) -> crate::BoundGenerator {
        let desc = registry
            .generators()
            .map(|r| (r.descriptor)())
            .find(|d| d.id == SDXL_ID)
            .unwrap();
        receipt
            .load_generator_using(
                |spec| {
                    Ok(Box::new(Probe {
                        desc,
                        contract: registry.memory_strategy_contract(SDXL_ID, spec)?.unwrap(),
                        spec: spec.clone(),
                        calls,
                        consumed,
                    }))
                },
                Box::new(|_| Ok(())),
            )
            .unwrap()
    }
    fn admission(registry: &ProviderRegistry, candidate: &IosCandidate) -> Admission {
        admit_with_rows(
            registry,
            std::slice::from_ref(candidate),
            DEVICE,
            budget(),
            &[capture(registry, candidate, 1 << 20)],
        )
        .unwrap()
    }
    fn request(c: &IosCandidate) -> GenerationRequest {
        match &c.work {
            IosWork::Generate(r) => r.clone(),
            _ => unreachable!(),
        }
    }

    #[test]
    fn actual_request_mutations_refuse_before_every_provider_entry_and_preserve_cancel() {
        let registry = crate::provider_registry().unwrap();
        let root = fixture();
        let candidate = candidate(
            &registry,
            root.path(),
            "sdxl",
            512,
            Some(Quant::Q4),
            "text_to_image",
        );
        let calls = Calls::default();
        let consumed = Rc::new(Cell::new(0));
        let bound = bind(
            admission(&registry, &candidate),
            &registry,
            calls.clone(),
            consumed.clone(),
        );
        let req = request(&candidate);
        let mut changed = Vec::new();
        let mut r = req.clone();
        r.width = 2048;
        changed.push(r);
        let mut r = req.clone();
        r.memory = Some(GenerationMemory::default());
        changed.push(r);
        let mut r = req.clone();
        r.memory.as_mut().unwrap().cfg_batching = Some(CfgBatching::Batched);
        changed.push(r);
        let mut r = req.clone();
        r.prompt = "token window expansion ".repeat(2000);
        changed.push(r);
        let mut r = req.clone();
        r.conditioning.push(Conditioning::Reference {
            image: Image {
                width: 1,
                height: 1,
                pixels: vec![0; 3],
            },
            strength: None,
        });
        changed.push(r);
        let mut r = req.clone();
        r.count += 1;
        changed.push(r);
        for r in changed {
            assert!(bound.validate(&r).is_err());
            assert!(bound.generate(&r, &mut |_| {}).is_err());
            assert!(bound
                .generate_streaming(&r, &mut |_| {}, &mut |_| {})
                .is_err());
            assert!(bound.generate_with_report(&r, &mut |_| {}).is_err());
            assert!(bound.open_conversation(&r).is_err());
        }
        assert!(
            calls.borrow().is_empty(),
            "wrong request must not reach any provider entry"
        );
        assert_eq!(consumed.get(), 0);
        let cancelled = req.clone();
        cancelled.cancel.cancel();
        assert!(matches!(
            bound.generate(&cancelled, &mut |_| {}),
            Err(Error::Canceled)
        ));
        assert!(calls.borrow().is_empty());
        let mut seeded = req.clone();
        seeded.cancel = Default::default();
        seeded.seed = Some(12345);
        bound.validate(&seeded).unwrap();
        assert_eq!(
            &*calls.borrow(),
            &["validate"],
            "memory-independent seed stays usable"
        );
        let mut running = req.clone();
        running.cancel = Default::default();
        let cancelled = bound.generate(&running, &mut |_| running.cancel.cancel());
        assert!(
            matches!(cancelled, Err(Error::Canceled)),
            "provider must receive the real cancellation handle"
        );
        assert_eq!(consumed.get(), 0);
        assert_eq!(&*calls.borrow(), &["validate", "generate"]);
    }

    #[test]
    fn directory_and_override_replacement_after_load_refuses_deferred_consumption() {
        for override_file in [false, true] {
            let registry = crate::provider_registry().unwrap();
            let root = fixture();
            let mut candidate = candidate(
                &registry,
                root.path(),
                "sdxl",
                512,
                Some(Quant::Q4),
                "text_to_image",
            );
            let source = if override_file {
                let path = root.path().join("staged-vae.safetensors");
                write_tensor(&path, "F16", 8);
                candidate
                    .spec
                    .components
                    .insert("vae_fp16_fix".into(), WeightsSource::File(path.clone()));
                let contract = registry
                    .memory_strategy_contract(SDXL_ID, &candidate.spec)
                    .unwrap()
                    .unwrap();
                let calibration = contract.calibration.unwrap();
                candidate.context.calibration_abi = calibration.abi;
                candidate.context.calibration_fingerprint = calibration.fingerprint;
                path
            } else {
                root.path().join("unet/diffusion_pytorch_model.safetensors")
            };
            let consumed = Rc::new(Cell::new(0));
            let calls = Calls::default();
            let bound = bind(
                admission(&registry, &candidate),
                &registry,
                calls.clone(),
                consumed.clone(),
            );
            let result = bound.generate(&request(&candidate), &mut |_| {
                // Same headers and payload length: a header-layout digest alone would NOT detect
                // this byte replacement. No reselection/profile scan is performed here.
                let mut bytes = std::fs::read(&source).unwrap();
                *bytes.last_mut().unwrap() ^= 1;
                let replacement = source.with_extension("new");
                std::fs::write(&replacement, bytes).unwrap();
                std::fs::rename(replacement, &source).unwrap();
            });
            assert!(result.is_err());
            assert_eq!(
                consumed.get(),
                0,
                "replacement must fail before deferred bytes"
            );
            assert_eq!(&*calls.borrow(), &["generate"]);
        }
        // Adding a shard after selection must also invalidate the retained directory inventory.
        let registry = crate::provider_registry().unwrap();
        let root = fixture();
        let candidate = candidate(
            &registry,
            root.path(),
            "sdxl",
            512,
            Some(Quant::Q4),
            "text_to_image",
        );
        let receipt = admission(&registry, &candidate);
        write_tensor(&root.path().join("unet/new-shard.safetensors"), "F16", 8);
        let loaded = Cell::new(false);
        assert!(receipt
            .load_generator_using(
                |_| {
                    loaded.set(true);
                    Err(Error::Unsupported("must not load".into()))
                },
                Box::new(|_| Ok(()))
            )
            .is_err());
        assert!(!loaded.get());
    }

    #[test]
    fn bound_generator_forwards_complete_introspection_generation_and_lifecycle_surface() {
        let registry = crate::provider_registry().unwrap();
        let root = fixture();
        let candidate = candidate(
            &registry,
            root.path(),
            "sdxl",
            512,
            Some(Quant::Q4),
            "text_to_image",
        );
        let mut receipt = admission(&registry, &candidate);
        // Test-only fake supports the extra scope methods too, independently of SDXL's declared
        // rung subset; forwarding must not accidentally erase a future provider implementation.
        receipt.context.selection.parameters.attention_chunk_size = Some(1);
        receipt.context.selection.parameters.transformer_window_size = Some(1);
        let calls = Calls::default();
        let consumed = Rc::new(Cell::new(0));
        let bound = bind(receipt, &registry, calls.clone(), consumed.clone());
        let req = &request(&candidate);
        let _ = bound.descriptor();
        let _ = bound.veda_execution_report();
        let _ = bound.adapter_apply_reports();
        let _ = bound.checkpoint_weight_facts();
        let _ = bound.memory_strategy_contract();
        let _ = bound.structural_resident_evidence();
        let _ = bound.predicted_memory_peak_from_base(1);
        bound
            .veda_workspace_bound(veda::workspace::Request::from(req))
            .unwrap();
        let context = bound.admitted_context().clone();
        assert_eq!(
            bound.memory_strategy_safety_check(&context),
            MemorySafetyDecision::Accept
        );
        let mut scope = bound
            .begin_memory_strategy_request(&context)
            .unwrap()
            .unwrap();
        scope.configure_request(&mut req.clone()).unwrap();
        scope.enter_phase(MemoryPhase::Decode).unwrap();
        scope.leave_phase(MemoryPhase::Decode).unwrap();
        let p = context.selection.parameters;
        scope
            .configure_decode(
                p.decode_tile_edge.unwrap(),
                p.decode_overlap.unwrap(),
                context.geometry,
            )
            .unwrap();
        scope.configure_attention(1).unwrap();
        scope.materialize_transformer_window(0, 1).unwrap();
        scope.finish(MemoryRunOutcome::Canceled).unwrap();
        drop(scope);
        bound.validate(req).unwrap();
        let mut progress = 0;
        bound.generate(req, &mut |_| progress += 1).unwrap();
        assert_eq!(progress, 1);
        let mut chunks = 0;
        bound
            .generate_streaming(req, &mut |_| chunks += 1, &mut |_| {})
            .unwrap();
        assert_eq!(chunks, 1);
        bound.generate_with_report(req, &mut |_| {}).unwrap();
        assert!(bound
            .open_conversation(req)
            .err()
            .unwrap()
            .to_string()
            .contains("probe conversation"));
        for expected in [
            "descriptor",
            "veda_report",
            "adapters",
            "facts",
            "contract",
            "structural",
            "prediction",
            "workspace",
            "safety",
            "begin",
            "configure_request",
            "enter",
            "leave",
            "decode",
            "attention",
            "window",
            "finish",
            "validate",
            "generate",
            "streaming",
            "report",
            "conversation",
        ] {
            assert!(
                calls.borrow().contains(&expected),
                "missing forwarding for {expected}"
            );
        }
        let mut cleanup = bound
            .begin_memory_strategy_request(&context)
            .unwrap()
            .unwrap();
        std::fs::write(root.path().join("late-source-change"), b"new").unwrap();
        assert!(cleanup.finish(MemoryRunOutcome::Canceled).is_err());
        assert_eq!(
            calls.borrow().last(),
            Some(&"finish"),
            "cleanup must forward despite invalid source"
        );
    }
}
