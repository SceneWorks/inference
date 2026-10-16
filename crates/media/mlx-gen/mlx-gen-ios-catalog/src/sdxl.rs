//! SDXL-family iOS composition and source-aware, fail-closed admission (sc-26268).
//!
//! Five checkpoint routes share the SAME `sdxl` generator and every one of its modes. Mac MLX
//! allocator measurements are engine evidence, not iPhone footprint evidence. Until a validated
//! S3 capture binds a route, actual artifact layout, execution shape and target hardware, product
//! admission refuses. Diagnostic/catalog access is not an admission verdict.

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{
    CfgBatching, Error, GenerationMemory, LoadShape, LoadSpec, MemoryBudget, MemoryRunContext,
    MemorySafetyDecision, ProviderRegistry, Result, WeightsSource,
};
use sha2::{Digest, Sha256};

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

/// A caller-ordered candidate: resolution/tier preference stays with the caller, never a hidden
/// quality downgrade. Context is the provider's ordinary typed safety input; execution also binds
/// independent levers such as sequential CFG which are not memory-ladder rungs.
#[derive(Clone, Debug)]
pub struct SdxlCandidate {
    pub spec: LoadSpec,
    pub context: MemoryRunContext,
    pub execution: GenerationMemory,
    pub request: mlx_gen::gen_core::GenerationRequest,
}

/// Header/config identity for ALL sources, overlays and component overrides in the load spec.
/// Tensor payloads are never read. Lexical paths deliberately participate: relocating a capture
/// requires rebinding, rather than guessing that a similarly named directory is the same artifact.
/// No pin/currency check and no Mac peak is turned into iOS authority.
pub fn sdxl_snapshot_profile(spec: &LoadSpec) -> Result<[u8; 32]> {
    let mut digest = Sha256::new();
    digest.update(b"sdxl-ios-artifact-layout-v1");
    digest.update(format!("{spec:?}").as_bytes());
    fn collect(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
        if path.is_file() {
            files.push(path.to_owned());
        } else {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    if mlx_gen::gen_core::weightsmeta::is_hidden_dir(&path) {
                        continue;
                    }
                    // Avoid cycles and untracked directory aliases in evidence inventory.
                    if entry.file_type()?.is_symlink() {
                        return Err(Error::Unsupported(
                            "SDXL evidence: symlinked subdirectory".into(),
                        ));
                    }
                    collect(&path, files)?;
                } else if !mlx_gen::gen_core::weightsmeta::is_hidden_file(&path) {
                    files.push(path);
                }
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    for source in spec
        .file_source_paths()
        .into_iter()
        .chain(spec.directory_source_paths())
    {
        collect(source, &mut files)?;
    }
    files.sort();
    files.dedup();
    let mut tensor_files = 0;
    for file in files {
        digest.update(file.to_string_lossy().as_bytes());
        match file.extension().and_then(|s| s.to_str()) {
            Some("safetensors") => {
                tensor_files += 1;
                for header in
                    mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&file)?
                {
                    digest.update(format!("{header:?}").as_bytes());
                }
            }
            // Configs include the packed-tier marker and component architecture; tokenizer inputs
            // affect sequence shape too. Hash these small non-tensor artifacts, never weight data.
            Some("json" | "txt") => {
                digest.update(std::fs::read(&file)?);
            }
            _ => {
                digest.update(std::fs::metadata(&file)?.len().to_le_bytes());
            }
        }
    }
    if tensor_files == 0 {
        return Err(Error::Unsupported(
            "SDXL evidence: no tensor headers in supplied sources".into(),
        ));
    }
    Ok(digest.finalize().into())
}

/// Bind the ACTUAL request exactly (including prompt/token-window inputs and conditioning), not
/// merely a claimed geometry/context. Only seed/cancellation are memory-independent; preview's
/// Debug records active/inactive without callback identity, so preview memory behavior stays bound.
pub(crate) fn actual_request_key(request: &mlx_gen::gen_core::GenerationRequest) -> [u8; 32] {
    let mut normalized = request.clone();
    normalized.cancel = Default::default();
    normalized.seed = None;
    Sha256::digest(format!("{normalized:?}").as_bytes()).into()
}

fn request_key(candidate: &SdxlCandidate) -> String {
    let c = &candidate.context;
    format!(
        "{:?}:{:?}:{:?}:{:?}:{}:{}:{:?}:{:?}:{:?}",
        c.selection,
        c.mode,
        c.geometry,
        c.overlay,
        c.use_pid,
        c.has_phases,
        c.load_shape,
        candidate.execution,
        actual_request_key(&candidate.request)
    )
}

/// Total absolute-process capacity with no speculative reclaim credit. A captured process peak
/// is converted to incremental demand exactly once at the provider check below.
pub fn capacity_budget(cap: u64, committed: u64, live_incremental: Option<u64>) -> MemoryBudget {
    MemoryBudget {
        total_bytes: live_incremental.map_or(cap, |available| {
            cap.min(committed.saturating_add(available))
        }),
        committed_bytes: committed,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    }
}

/// A sealed selection, including retained source tokens/inventory and an actual-request binding.
/// There is no public constructor and no raw generator/spec escape hatch.
#[derive(Debug)]
pub struct SdxlAdmission {
    index: usize,
    footprint_bytes: u64,
    spec: LoadSpec,
    context: MemoryRunContext,
    request: [u8; 32],
}

impl SdxlAdmission {
    pub fn index(&self) -> usize {
        self.index
    }
    pub fn footprint_mib(&self) -> u32 {
        self.footprint_bytes.div_ceil(1 << 20) as u32
    }
    pub fn context(&self) -> &MemoryRunContext {
        &self.context
    }
    pub fn load(
        self,
        registry: &ProviderRegistry,
        capacity_check: impl Fn(u64) -> Result<()> + 'static,
    ) -> Result<super::sdxl_bound::BoundSdxlGenerator> {
        self.load_using(
            |spec| registry.load(SDXL_ID, spec),
            Box::new(capacity_check),
        )
    }
    fn load_using(
        self,
        load: impl FnOnce(&LoadSpec) -> Result<Box<dyn mlx_gen::gen_core::Generator>>,
        capacity_check: Box<dyn Fn(u64) -> Result<()>>,
    ) -> Result<super::sdxl_bound::BoundSdxlGenerator> {
        capacity_check(self.footprint_bytes)?;
        let inner = self
            .spec
            .read_prepared_files_unchanged(|| load(&self.spec))?;
        Ok(super::sdxl_bound::BoundSdxlGenerator::new(
            inner,
            self.spec,
            self.context,
            self.request,
            self.footprint_bytes,
            capacity_check,
        ))
    }
}

fn validate_actual(candidate: &SdxlCandidate) -> Result<()> {
    use mlx_gen::gen_core::{Conditioning, MemoryMode};
    let req = &candidate.request;
    let c = &candidate.context;
    let references = req
        .conditioning
        .iter()
        .filter(|v| matches!(v, Conditioning::Reference { .. }))
        .count() as u32;
    let mask = req
        .conditioning
        .iter()
        .any(|v| matches!(v, Conditioning::Mask { .. }));
    let control = req
        .conditioning
        .iter()
        .any(|v| matches!(v, Conditioning::Control { .. }));
    let route_matches = match &c.mode {
        MemoryMode::TextToImage => references == 0 && !mask,
        MemoryMode::ImageToImage => references > 0 && !mask,
        MemoryMode::Edit => references > 0 && mask,
        MemoryMode::Other(mode) if mode == "control" => control,
        MemoryMode::Other(mode) if mode == "ip_adapter" => references > 0 && !mask,
        _ => false,
    };
    if req.width != c.geometry.width
        || req.height != c.geometry.height
        || req.count != c.geometry.batch
        || req.frames.unwrap_or(1) != c.geometry.frames
        || req.memory_reference_count() != c.geometry.reference_count
        || c.has_reference != (req.memory_reference_count() > 0)
        || req.use_pid != c.use_pid
        || req.memory.unwrap_or_default() != candidate.execution
        || !route_matches
    {
        return Err(Error::Unsupported(
            "SDXL actual request disagrees with admission context/execution".into(),
        ));
    }
    let registry = crate::provider_registry()?;
    let descriptor = registry
        .generators()
        .map(|r| (r.descriptor)())
        .find(|d| d.id == SDXL_ID)
        .ok_or_else(|| Error::Unsupported("SDXL generator is not registered".into()))?;
    descriptor.capabilities.validate_request(SDXL_ID, req)
}

/// An owned device capture, not constructible by API callers or an untrusted sidecar. S3 must
/// populate records with measured *process footprint*, not allocator live bytes plus a guessed cap.
struct DeviceEvidence {
    machine: String,
    route: String,
    profile: [u8; 32],
    request: String,
    footprint_bytes: u64,
}

fn select_with_evidence(
    registry: &ProviderRegistry,
    candidates: &[SdxlCandidate],
    machine: &str,
    budget: MemoryBudget,
    evidence: &[DeviceEvidence],
) -> Result<SdxlAdmission> {
    let registration = registry
        .memory_strategy_registrations()
        .find(|r| r.provider_id == SDXL_ID)
        .ok_or_else(|| {
            Error::Unsupported("iOS SDXL has no registered memory safety check".into())
        })?;
    let mut refusals = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        // A bad preferred artifact is a candidate refusal, not a failure of valid later choices.
        let attempt = || -> Result<SdxlAdmission> {
            validate_actual(candidate)?;
            let route = candidate
                .spec
                .resolved_route
                .as_deref()
                .filter(|route| SDXL_FAMILY_ROUTES.contains(route))
                .ok_or_else(|| Error::Unsupported("unknown SDXL family route".into()))?;
            let key = request_key(candidate);
            if !evidence
                .iter()
                .any(|e| e.machine == machine && e.route == route && e.request == key)
            {
                return Err(Error::Unsupported(
                    "no applicable device/request capture (source unopened)".into(),
                ));
            }
            // Capture replacement-sensitive file identities BEFORE header/config inspection and
            // carry them through load, each phase reopen and block materialization.
            let guard = mlx_gen::gen_core::SourceInventory::capture(&candidate.spec)?;
            guard.read_unchanged(|| {
                let profile = sdxl_snapshot_profile(&candidate.spec)?;
                let record = evidence
                    .iter()
                    .find(|e| {
                        e.machine == machine
                            && e.route == route
                            && e.profile == profile
                            && e.request == key
                    })
                    .ok_or_else(|| {
                        Error::Unsupported("no applicable artifact/device/request capture".into())
                    })?;
                if record.footprint_bytes == 0
                    || record.footprint_bytes.div_ceil(1 << 20) > u64::from(u32::MAX)
                {
                    return Err(Error::Unsupported("invalid captured footprint".into()));
                }
                let contract = (registration.contract)(&candidate.spec)?;
                let expected = contract
                    .generation_memory(&candidate.context.selection)
                    .unwrap_or_default();
                let actual = candidate.execution;
                if (
                    actual.stage_residency,
                    actual.tile_vae_decode,
                    actual.chunk_attention,
                    actual.stream_transformer_blocks,
                    actual.decode_tile_edge,
                    actual.decode_overlap,
                    actual.attention_chunk_size,
                    actual.transformer_window_size,
                    actual.transformer_window_component,
                ) != (
                    expected.stage_residency,
                    expected.tile_vae_decode,
                    expected.chunk_attention,
                    expected.stream_transformer_blocks,
                    expected.decode_tile_edge,
                    expected.decode_overlap,
                    expected.attention_chunk_size,
                    expected.transformer_window_size,
                    expected.transformer_window_component,
                ) {
                    return Err(Error::Unsupported(
                        "actual execution does not implement the selected ladder".into(),
                    ));
                }
                let mut context = candidate.context.clone();
                context.budget = budget;
                context.predicted_peak_bytes = record
                    .footprint_bytes
                    .saturating_sub(budget.committed_bytes);
                if let MemorySafetyDecision::Reject { reason } =
                    (registration.safety_check)(&candidate.spec, &contract, &context)
                {
                    return Err(Error::Unsupported(reason));
                }
                let mut spec = candidate.spec.clone();
                spec.bind_source_inventory(guard.clone())?;
                Ok(SdxlAdmission {
                    index,
                    footprint_bytes: record.footprint_bytes,
                    spec,
                    context,
                    request: actual_request_key(&candidate.request),
                })
            })
        };
        match attempt() {
            Ok(admitted) => return Ok(admitted),
            Err(error) => refusals.push(format!("candidate {index}: {error}")),
        }
    }
    Err(Error::Unsupported(format!("iOS SDXL admission: no artifact/route/execution-bound device footprint fits {machine};         Mac allocator peaks do not authorize an iPhone snapshot. {}", refusals.join("; "))))
}

/// First (highest caller preference) candidate admitted under the live S3 budget. Currently no
/// artifact-bound iPhone SDXL capture exists: registration is complete but product loads fail
/// closed. Synthetic evidence lives ONLY inside tests and cannot authorize a real snapshot.
pub fn select_sdxl_candidate(
    registry: &ProviderRegistry,
    candidates: &[SdxlCandidate],
    machine: &str,
    budget: MemoryBudget,
) -> Result<SdxlAdmission> {
    select_with_evidence(registry, candidates, machine, budget, &[])
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        standard_memory_behavior_context, MemoryBehaviorRoute, MemoryMode, MemoryNumericTier,
        MemoryStrategy, Quant,
    };

    pub(super) fn write_tensor(path: &Path, dtype: &str, count: usize) {
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

    pub(super) fn fixture() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        for component in ["unet", "vae", "text_encoder", "text_encoder_2"] {
            let name = if component.starts_with("text") {
                "model.safetensors"
            } else {
                "diffusion_pytorch_model.safetensors"
            };
            write_tensor(&root.path().join(component).join(name), "F16", 8);
        }
        root
    }

    pub(super) fn candidate(
        registry: &ProviderRegistry,
        root: &Path,
        route: &str,
        edge: u32,
        quant: Option<Quant>,
        mode: MemoryMode,
    ) -> SdxlCandidate {
        let mut spec = sdxl_load_spec(route, WeightsSource::Dir(root.to_owned())).unwrap();
        spec.quantize = quant;
        let contract = registry
            .memory_strategy_contract(SDXL_ID, &spec)
            .unwrap()
            .unwrap();
        let mut context = standard_memory_behavior_context(
            &contract,
            MemoryStrategy::BoundedDecode,
            MemoryNumericTier {
                precision: spec.precision,
                quant,
                component_precision_floors: &[],
            },
            MemoryBehaviorRoute {
                mode,
                reference_count: 0,
                use_pid: false,
                has_phases: true,
                overlay: None,
            },
        )
        .unwrap();
        context.selection.parameters.stage_residency = Some(true);
        context.selection.parameters.decode_tile_edge = Some(256);
        context.selection.parameters.decode_overlap =
            Some(mlx_gen_sdxl::memory_strategy::DECODE_OVERLAP);
        context.geometry.width = edge;
        context.geometry.height = edge;
        let mut execution = contract.generation_memory(&context.selection).unwrap();
        execution.cfg_batching = Some(CfgBatching::Sequential);
        let image = || mlx_gen::gen_core::Image {
            width: 1,
            height: 1,
            pixels: vec![128; 3],
        };
        use mlx_gen::gen_core::Conditioning;
        let conditioning = match &context.mode {
            MemoryMode::ImageToImage => vec![Conditioning::Reference {
                image: image(),
                strength: Some(0.5),
            }],
            MemoryMode::Edit => vec![
                Conditioning::Reference {
                    image: image(),
                    strength: Some(0.5),
                },
                Conditioning::Mask { image: image() },
            ],
            MemoryMode::Other(mode) if mode == "ip_adapter" => vec![Conditioning::Reference {
                image: image(),
                strength: Some(0.5),
            }],
            MemoryMode::Other(mode) if mode == "control" => vec![Conditioning::Control {
                image: image(),
                kind: mlx_gen::gen_core::ControlKind::Other("tile".into()),
                scale: Some(1.0),
            }],
            _ => vec![],
        };
        let references = conditioning
            .iter()
            .filter(|v| matches!(v, Conditioning::Reference { .. }))
            .count() as u32;
        context.has_reference = references > 0;
        context.geometry.reference_count = references;
        let request = mlx_gen::gen_core::GenerationRequest {
            prompt: "synthetic fixture".into(),
            width: edge,
            height: edge,
            count: context.geometry.batch,
            conditioning,
            memory: Some(execution),
            ..Default::default()
        };
        context.geometry.reference_count = request.memory_reference_count();
        context.has_reference = context.geometry.reference_count > 0;
        SdxlCandidate {
            spec,
            context,
            execution,
            request,
        }
    }

    pub(super) fn capture(candidate: &SdxlCandidate, bytes: u64) -> DeviceEvidence {
        DeviceEvidence {
            machine: "synthetic-test-device".into(),
            route: candidate.spec.resolved_route.clone().unwrap(),
            profile: sdxl_snapshot_profile(&candidate.spec).unwrap(),
            request: request_key(candidate),
            footprint_bytes: bytes,
        }
    }

    pub(super) fn budget() -> MemoryBudget {
        // Mechanism fixture, NOT an iPhone cap or a fabricated measured model fit.
        MemoryBudget {
            total_bytes: 1 << 30,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        }
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
        let mut spec = sdxl_load_spec("illustrious_xl_v2", WeightsSource::File(primary)).unwrap();
        let plain = sdxl_snapshot_profile(&spec).unwrap();
        spec.control = Some(WeightsSource::File(control.clone()));
        spec.ip_adapter = Some(WeightsSource::File(ip));
        spec.components
            .insert("vae_fp16_fix".into(), WeightsSource::File(vae.clone()));
        let overlaid = sdxl_snapshot_profile(&spec).unwrap();
        assert_ne!(plain, overlaid);
        write_tensor(&control, "F32", 8);
        let changed_control = sdxl_snapshot_profile(&spec).unwrap();
        assert_ne!(overlaid, changed_control);
        write_tensor(&vae, "F32", 16);
        assert_ne!(changed_control, sdxl_snapshot_profile(&spec).unwrap());
    }

    #[test]
    fn source_bound_budget_selection_picks_largest_offered_fit_and_refuses_next_before_load() {
        let registry = crate::provider_registry().unwrap();
        let root = fixture();
        let candidates = [
            candidate(
                &registry,
                root.path(),
                "sdxl",
                1024,
                None,
                MemoryMode::TextToImage,
            ),
            candidate(
                &registry,
                root.path(),
                "sdxl",
                768,
                Some(Quant::Q8),
                MemoryMode::TextToImage,
            ),
            candidate(
                &registry,
                root.path(),
                "sdxl",
                512,
                Some(Quant::Q4),
                MemoryMode::TextToImage,
            ),
        ];
        let cap = budget().total_bytes;
        let evidence = [
            capture(&candidates[0], cap + 1),
            capture(&candidates[1], cap + 1),
            capture(&candidates[2], cap),
        ];
        assert_eq!(
            select_with_evidence(
                &registry,
                &candidates,
                "synthetic-test-device",
                budget(),
                &evidence
            )
            .map(|receipt| (receipt.index(), receipt.footprint_mib()))
            .unwrap(),
            (2, 1024)
        );
        assert!(select_with_evidence(
            &registry,
            &candidates[..2],
            "synthetic-test-device",
            budget(),
            &evidence
        )
        .is_err());
        // The public API cannot consume synthetic captures and never uses the Mac's 3.2289 GiB.
        assert!(
            select_sdxl_candidate(&registry, &candidates, "synthetic-test-device", budget())
                .is_err()
        );
        let mut committed = budget();
        committed.reserved_headroom_bytes = 1;
        assert!(select_with_evidence(
            &registry,
            &candidates,
            "synthetic-test-device",
            committed,
            &evidence
        )
        .is_err());
    }

    #[test]
    fn evidence_never_crosses_family_mode_artifact_tier_execution_or_hardware() {
        let registry = crate::provider_registry().unwrap();
        let root = fixture();
        for route in SDXL_FAMILY_ROUTES {
            for mode in [
                MemoryMode::TextToImage,
                MemoryMode::ImageToImage,
                MemoryMode::Edit,
                MemoryMode::Other("control".into()),
                MemoryMode::Other("ip_adapter".into()),
            ] {
                let base = candidate(&registry, root.path(), route, 512, Some(Quant::Q4), mode);
                let evidence = [capture(&base, budget().total_bytes / 2)];
                assert!(select_with_evidence(
                    &registry,
                    std::slice::from_ref(&base),
                    "synthetic-test-device",
                    budget(),
                    &evidence
                )
                .is_ok());
                assert!(select_with_evidence(
                    &registry,
                    std::slice::from_ref(&base),
                    "other-device",
                    budget(),
                    &evidence
                )
                .is_err());
                let mut altered = base.clone();
                altered.spec.resolved_route = Some("different-route".into());
                assert!(select_with_evidence(
                    &registry,
                    &[altered],
                    "synthetic-test-device",
                    budget(),
                    &evidence
                )
                .is_err());
                let mut altered = base.clone();
                altered.execution.cfg_batching = Some(CfgBatching::Batched);
                assert!(select_with_evidence(
                    &registry,
                    &[altered],
                    "synthetic-test-device",
                    budget(),
                    &evidence
                )
                .is_err());
                let mut altered = base.clone();
                altered.context.mode = MemoryMode::Other("different-mode".into());
                assert!(select_with_evidence(
                    &registry,
                    &[altered],
                    "synthetic-test-device",
                    budget(),
                    &evidence
                )
                .is_err());
                let mut altered = base.clone();
                altered.spec.quantize = Some(Quant::Q8);
                assert!(select_with_evidence(
                    &registry,
                    &[altered],
                    "synthetic-test-device",
                    budget(),
                    &evidence
                )
                .is_err());
            }
        }
        let base = candidate(
            &registry,
            root.path(),
            "sdxl",
            512,
            Some(Quant::Q4),
            MemoryMode::TextToImage,
        );
        let evidence = [capture(&base, budget().total_bytes / 2)];
        write_tensor(
            &root.path().join("unet/diffusion_pytorch_model.safetensors"),
            "F32",
            16,
        );
        assert!(select_with_evidence(
            &registry,
            &[base],
            "synthetic-test-device",
            budget(),
            &evidence
        )
        .is_err());
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
            MemoryMode::TextToImage,
        );
        base.context.calibration_fingerprint = "wrong-provider-identity".into();
        let evidence = [capture(&base, 1)];
        assert!(select_with_evidence(
            &registry,
            &[base],
            "synthetic-test-device",
            budget(),
            &evidence
        )
        .is_err());
        let mut base = candidate(
            &registry,
            root.path(),
            "sdxl",
            512,
            Some(Quant::Q4),
            MemoryMode::TextToImage,
        );
        base.execution.tile_vae_decode = false;
        let evidence = [capture(&base, 1)];
        assert!(select_with_evidence(
            &registry,
            &[base],
            "synthetic-test-device",
            budget(),
            &evidence
        )
        .is_err());
    }
}

#[cfg(test)]
mod execution_tests {
    use super::tests::{budget, candidate, capture, fixture, write_tensor};
    use super::*;
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
        receipt: SdxlAdmission,
        registry: &ProviderRegistry,
        calls: Calls,
        consumed: Rc<Cell<u32>>,
    ) -> crate::BoundSdxlGenerator {
        let desc = registry
            .generators()
            .map(|r| (r.descriptor)())
            .find(|d| d.id == SDXL_ID)
            .unwrap();
        receipt
            .load_using(
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
    fn admission(registry: &ProviderRegistry, candidate: &SdxlCandidate) -> SdxlAdmission {
        select_with_evidence(
            registry,
            std::slice::from_ref(candidate),
            "synthetic-test-device",
            budget(),
            &[capture(candidate, 1 << 20)],
        )
        .unwrap()
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
            MemoryMode::TextToImage,
        );
        let calls = Calls::default();
        let consumed = Rc::new(Cell::new(0));
        let bound = bind(
            admission(&registry, &candidate),
            &registry,
            calls.clone(),
            consumed.clone(),
        );
        let req = candidate.request.clone();
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
                MemoryMode::TextToImage,
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
            let result = bound.generate(&candidate.request, &mut |_| {
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
            MemoryMode::TextToImage,
        );
        let receipt = admission(&registry, &candidate);
        write_tensor(&root.path().join("unet/new-shard.safetensors"), "F16", 8);
        let loaded = Cell::new(false);
        assert!(receipt
            .load_using(
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
    fn live_headroom_selects_fallback_with_absolute_process_accounting() {
        let registry = crate::provider_registry().unwrap();
        let root = fixture();
        let preferred = candidate(
            &registry,
            root.path(),
            "sdxl",
            1024,
            None,
            MemoryMode::TextToImage,
        );
        let fallback = candidate(
            &registry,
            root.path(),
            "sdxl",
            512,
            Some(Quant::Q4),
            MemoryMode::TextToImage,
        );
        let rows = [capture(&preferred, 3 << 30), capture(&fallback, 2 << 30)];
        let live = capacity_budget(4 << 30, 0, Some(5 << 29)); // 4GiB fixed, 2.5GiB live
        let choice = select_with_evidence(
            &registry,
            &[preferred.clone(), fallback.clone()],
            "synthetic-test-device",
            live,
            &rows,
        )
        .unwrap();
        assert_eq!(choice.index(), 1);
        // Captures are absolute process peaks; already committed bytes are removed exactly once.
        let held = capacity_budget(4 << 30, 1 << 29, Some(2 << 30));
        assert_eq!(
            select_with_evidence(
                &registry,
                &[preferred.clone(), fallback.clone()],
                "synthetic-test-device",
                held,
                &rows
            )
            .unwrap()
            .index(),
            1
        );
    }

    #[test]
    fn malformed_preferred_candidate_is_a_refusal_not_a_global_error() {
        let registry = crate::provider_registry().unwrap();
        let root = fixture();
        let preferred = candidate(
            &registry,
            root.path(),
            "sdxl",
            1024,
            None,
            MemoryMode::TextToImage,
        );
        let fallback = candidate(
            &registry,
            root.path(),
            "sdxl",
            512,
            Some(Quant::Q4),
            MemoryMode::TextToImage,
        );
        let mut invalid_contract = preferred.clone();
        let invalid_vae = tempfile::tempdir().unwrap();
        write_tensor(
            &invalid_vae.path().join("wrong-vae-name.safetensors"),
            "F16",
            8,
        );
        invalid_contract.spec.components.insert(
            "vae_fp16_fix".into(),
            WeightsSource::Dir(invalid_vae.path().to_owned()),
        );
        let rows = [
            capture(&preferred, 3 << 30),
            capture(&fallback, 2 << 30),
            capture(&invalid_contract, 1 << 20),
        ];
        let mut missing = preferred.clone();
        missing.spec.weights = WeightsSource::Dir(root.path().join("missing"));
        let mut malformed = preferred;
        let bad = tempfile::tempdir().unwrap();
        std::fs::write(bad.path().join("broken.safetensors"), b"bad").unwrap();
        malformed.spec.weights = WeightsSource::Dir(bad.path().to_owned());
        for first in [missing, malformed, invalid_contract] {
            assert_eq!(
                select_with_evidence(
                    &registry,
                    &[first, fallback.clone()],
                    "synthetic-test-device",
                    capacity_budget(4 << 30, 0, None),
                    &rows
                )
                .unwrap()
                .index(),
                1
            );
        }
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
            MemoryMode::TextToImage,
        );
        let mut receipt = admission(&registry, &candidate);
        // Test-only fake supports the extra scope methods too, independently of SDXL's declared
        // rung subset; forwarding must not accidentally erase a future provider implementation.
        receipt.context.selection.parameters.attention_chunk_size = Some(1);
        receipt.context.selection.parameters.transformer_window_size = Some(1);
        let calls = Calls::default();
        let consumed = Rc::new(Cell::new(0));
        let bound = bind(receipt, &registry, calls.clone(), consumed.clone());
        let req = &candidate.request;
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
        scope.configure_decode(256, 64, context.geometry).unwrap();
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
