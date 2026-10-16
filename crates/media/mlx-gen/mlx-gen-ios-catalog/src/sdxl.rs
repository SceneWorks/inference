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

fn request_key(candidate: &SdxlCandidate) -> String {
    let c = &candidate.context;
    format!(
        "{:?}:{:?}:{:?}:{:?}:{}:{}:{:?}:{:?}",
        c.selection,
        c.mode,
        c.geometry,
        c.overlay,
        c.use_pid,
        c.has_phases,
        c.load_shape,
        candidate.execution
    )
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
) -> Result<(usize, u32)> {
    let registration = registry
        .memory_strategy_registrations()
        .find(|r| r.provider_id == SDXL_ID)
        .ok_or_else(|| {
            Error::Unsupported("iOS SDXL has no registered memory safety check".into())
        })?;
    for (index, candidate) in candidates.iter().enumerate() {
        let Some(route) = candidate
            .spec
            .resolved_route
            .as_deref()
            .filter(|route| SDXL_FAMILY_ROUTES.contains(route))
        else {
            continue;
        };
        let profile = sdxl_snapshot_profile(&candidate.spec)?;
        let Some(record) = evidence.iter().find(|e| {
            e.machine == machine
                && e.route == route
                && e.profile == profile
                && e.request == request_key(candidate)
        }) else {
            continue;
        };
        if record.footprint_bytes == 0 {
            continue;
        }
        // Resolve the actual source-aware provider contract, not an invented iOS ladder.
        let contract = (registration.contract)(&candidate.spec)?;
        let expected = contract
            .generation_memory(&candidate.context.selection)
            .unwrap_or_default();
        let actual = candidate.execution;
        // Independent execution knobs are evidence-bound above. Ladder fields must implement the
        // very selection being checked, not claim a tiled/staged context for a resident request.
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
            continue;
        }
        let mut context = candidate.context.clone();
        context.budget = budget;
        context.predicted_peak_bytes = record.footprint_bytes;
        // The provider's real check delegates to standard_memory_strategy_safety_check, including
        // its route gates and loaded-tier/selection/calibration-identity checks.
        if matches!(
            (registration.safety_check)(&candidate.spec, &contract, &context),
            MemorySafetyDecision::Accept
        ) {
            let mib = record.footprint_bytes.div_ceil(1 << 20);
            if let Ok(mib) = u32::try_from(mib) {
                return Ok((index, mib));
            }
        }
    }
    Err(Error::Unsupported(format!(
        "iOS SDXL admission: no artifact/route/execution-bound device footprint fits {machine}; \
         Mac allocator peaks do not authorize an iPhone snapshot"
    )))
}

/// First (highest caller preference) candidate admitted under the live S3 budget. Currently no
/// artifact-bound iPhone SDXL capture exists: registration is complete but product loads fail
/// closed. Synthetic evidence lives ONLY inside tests and cannot authorize a real snapshot.
pub fn select_sdxl_candidate(
    registry: &ProviderRegistry,
    candidates: &[SdxlCandidate],
    machine: &str,
    budget: MemoryBudget,
) -> Result<(usize, u32)> {
    select_with_evidence(registry, candidates, machine, budget, &[])
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        standard_memory_behavior_context, MemoryBehaviorRoute, MemoryMode, MemoryNumericTier,
        MemoryStrategy, Quant,
    };

    fn write_tensor(path: &Path, dtype: &str, count: usize) {
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

    fn fixture() -> tempfile::TempDir {
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

    fn candidate(
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
        SdxlCandidate {
            spec,
            context,
            execution,
        }
    }

    fn capture(candidate: &SdxlCandidate, bytes: u64) -> DeviceEvidence {
        DeviceEvidence {
            machine: "synthetic-test-device".into(),
            route: candidate.spec.resolved_route.clone().unwrap(),
            profile: sdxl_snapshot_profile(&candidate.spec).unwrap(),
            request: request_key(candidate),
            footprint_bytes: bytes,
        }
    }

    fn budget() -> MemoryBudget {
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
        committed.committed_bytes = 1;
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
