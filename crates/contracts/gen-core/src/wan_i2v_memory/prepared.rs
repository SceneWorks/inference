//! `PreparedWanI2vMemory`: the sealed, load-time memory facts and the contract built from them.

use super::*;

#[derive(Clone, Debug)]
pub(super) struct SealedFile {
    pub(super) relative: String,
    pub(super) absolute: PathBuf,
    pub(super) pin: crate::PinnedWeightsFile,
    pub(super) digest: String,
    pub(super) resident_bytes: u64,
    pub(super) tensor: bool,
}

#[derive(Clone, Debug)]
pub struct PreparedWanI2vMemory {
    pub contract: MemoryProviderContract,
    pub artifact_identity: String,
    pub adapter_identity: String,
    pub repository: &'static str,
    pub revision: String,
    pub tier: MemoryNumericTier,
    pub route: WanI2vRoute,
    pub backend: WanI2vBackend,
    pub adapters: Vec<WanAdapterReceipt>,
    pub(super) root: PathBuf,
    pub(super) files: Vec<SealedFile>,
}

pub(super) fn sha256_file(path: &Path) -> crate::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub(super) fn tier(spec: &LoadSpec, backend: WanI2vBackend) -> crate::Result<MemoryNumericTier> {
    let quant = match spec.quantize {
        None if backend == WanI2vBackend::Mlx => match &spec.weights {
            WeightsSource::Dir(root) => match root.file_name().and_then(|name| name.to_str()) {
                Some("q4") => Some(Quant::Q4),
                Some("q8") => Some(Quant::Q8),
                Some("bf16") => None,
                _ => None,
            },
            WeightsSource::File(_) => None,
        },
        None => None,
        Some(Quant::Q4) => Some(Quant::Q4),
        Some(Quant::Q8) => Some(Quant::Q8),
        Some(other) => {
            return Err(crate::Error::Unsupported(format!(
                "Wan I2V memory authority does not support {other:?}"
            )))
        }
    };
    Ok(MemoryNumericTier {
        precision: Precision::Bf16,
        quant,
        component_precision_floors: &[],
    })
}

impl PreparedWanI2vMemory {
    pub fn prepare(
        spec: &LoadSpec,
        backend: WanI2vBackend,
        provider_id: &str,
    ) -> crate::Result<Self> {
        let route = WanI2vRoute::for_provider(provider_id)?;
        let tier = tier(spec, backend)?;
        if backend == WanI2vBackend::Candle
            && ((route == WanI2vRoute::Vace && (tier.quant.is_some() || !spec.adapters.is_empty()))
                || (route == WanI2vRoute::VaceFun && tier.quant.is_some()))
        {
            return Err(crate::Error::Unsupported(format!(
                "candle {} does not admit the selected numeric tier or load recipe",
                route.provider_id()
            )));
        }
        let root = validate_load_spec(spec, provider_id)?;
        spec.validate_prepared_file_pins()?;
        let (repository, expected_revision) = repository_policy(backend, route, tier);
        let revision = if route == WanI2vRoute::Vace {
            vace_source_receipt_revision(root, backend)?
        } else if route == WanI2vRoute::VaceFun {
            vace_fun_source_receipt_revision(root, backend)?
        } else {
            repository_revision(root, repository, expected_revision, tier, backend)?
        };
        let inventory = match backend {
            WanI2vBackend::Mlx => validate_mlx_inventory(root, route)?,
            WanI2vBackend::Candle => validate_candle_inventory(root, route)?,
        };
        let mut facts = MemoryAssetFacts::default();
        let mut files = Vec::new();
        let mut identity = Sha256::new();
        identity.update(RECEIPT_VERSION.as_bytes());
        identity.update(backend.key().as_bytes());
        identity.update(provider_id.as_bytes());
        identity.update(repository.as_bytes());
        identity.update(revision.as_bytes());
        identity.update(format!("{:?}", tier.quant).as_bytes());
        for relative in inventory {
            let path = root.join(&relative);
            let absolute = std::path::absolute(&path)?;
            let pin = spec
                .prepared_file_pins()
                .get(&absolute)
                .cloned()
                .ok_or_else(|| {
                    crate::Error::Unsupported(format!(
                        "{provider_id}: prepared receipt missing {}",
                        relative.display()
                    ))
                })?;
            pin.ensure_unchanged()?;
            let digest = sha256_file(&path)?;
            let relative_text = relative.to_string_lossy().replace('\\', "/");
            let phase = component_for(route, &relative_text);
            let resident_bytes = physical_resident_bytes(&path, backend, route, tier, phase)?;
            identity.update(relative_text.as_bytes());
            identity.update(digest.as_bytes());
            identity.update(resident_bytes.to_le_bytes());
            let component_total = match phase {
                MemoryPhase::Conditioning => facts.conditioning_bytes.checked_add(resident_bytes),
                MemoryPhase::Denoise => facts.transformer_bytes.checked_add(resident_bytes),
                MemoryPhase::Decode => facts.decoder_bytes.checked_add(resident_bytes),
            }
            .ok_or_else(|| crate::Error::Msg("Wan component byte overflow".to_owned()))?;
            match phase {
                MemoryPhase::Conditioning => facts.conditioning_bytes = component_total,
                MemoryPhase::Denoise => facts.transformer_bytes = component_total,
                MemoryPhase::Decode => facts.decoder_bytes = component_total,
            }
            files.push(SealedFile {
                relative: relative_text,
                absolute,
                pin,
                digest,
                resident_bytes,
                tensor: true,
            });
        }
        for relative in structural_files(root, backend, route)? {
            let path = root.join(&relative);
            let absolute = std::path::absolute(&path)?;
            let pin = spec
                .prepared_file_pins()
                .get(&absolute)
                .cloned()
                .ok_or_else(|| {
                    crate::Error::Unsupported(format!(
                        "{provider_id}: prepared receipt missing {}",
                        relative.display()
                    ))
                })?;
            pin.ensure_unchanged()?;
            let digest = sha256_file(&path)?;
            let relative_text = relative.to_string_lossy().replace('\\', "/");
            identity.update(relative_text.as_bytes());
            identity.update(digest.as_bytes());
            files.push(SealedFile {
                relative: relative_text,
                absolute,
                pin,
                digest,
                resident_bytes: 0,
                tensor: false,
            });
        }
        let (adapters, adapter_identity, overlay_bytes, adapter_files) =
            adapter_receipts(spec, route, tier)?;
        facts.overlay_bytes = overlay_bytes;
        facts.base_bytes = facts
            .conditioning_bytes
            .checked_add(facts.transformer_bytes)
            .and_then(|sum| sum.checked_add(facts.decoder_bytes))
            .ok_or_else(|| crate::Error::Msg("Wan base byte overflow".to_owned()))?;
        if facts.conditioning_bytes == 0 || facts.transformer_bytes == 0 || facts.decoder_bytes == 0
        {
            return Err(crate::Error::Unsupported(format!(
                "{provider_id}: incomplete nonzero physical facts"
            )));
        }
        identity.update(adapter_identity.as_bytes());
        let artifact_identity = format!("{:x}", identity.finalize());
        files.extend(adapter_files);
        let mut contract = contract(provider_id, route, backend, spec, facts);
        // Published only now, after the whole receipt sealed: the tier is the artifact's, not the
        // request's, and a snapshot that failed any inventory/marker check never reaches here.
        contract.calibration = production_calibration_fingerprint(route, backend, tier)
            .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape));
        let prepared = Self {
            contract,
            artifact_identity,
            adapter_identity,
            repository,
            revision,
            tier,
            route,
            backend,
            adapters,
            root: root.to_path_buf(),
            files,
        };
        prepared.ensure_unchanged()?;
        Ok(prepared)
    }

    /// The materialized snapshot directory this receipt was sealed over — the same `WeightsSource::Dir`
    /// root the backend loader parses its model config from.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Stamp the backend's architecture axes onto the sealed contract (epic SC-22657, E2).
    ///
    /// This module is backend-neutral and holds no model config, so its `contract` builder can only
    /// publish `MemoryArchitectureFacts::default()`. The contract a loaded generator hands back through
    /// `Generator::memory_strategy_contract()` **is** `self.contract`, so a backend that only
    /// overlays its axes on the per-request contract leaves the loaded path publishing the empty
    /// default. Each backend's `prepare` therefore stamps its route facts here, once, at seal time.
    pub fn with_architecture_facts(mut self, facts: crate::MemoryArchitectureFacts) -> Self {
        self.contract.architecture_facts = facts;
        self
    }

    pub fn ensure_unchanged(&self) -> crate::Result<()> {
        let current = match self.backend {
            WanI2vBackend::Mlx => validate_mlx_inventory(&self.root, self.route)?,
            WanI2vBackend::Candle => validate_candle_inventory(&self.root, self.route)?,
        };
        let expected = self
            .files
            .iter()
            .filter(|file| file.tensor && !file.relative.starts_with("adapter:"))
            .map(|file| PathBuf::from(&file.relative))
            .collect::<Vec<_>>();
        if current != expected {
            return Err(crate::Error::Unsupported(format!(
                "{}: physical inventory changed after admission",
                self.route.provider_id()
            )));
        }
        for file in &self.files {
            file.pin.ensure_unchanged()?;
            if sha256_file(&file.absolute)? != file.digest {
                return Err(crate::Error::Unsupported(format!(
                    "{} changed after admission",
                    file.absolute.display()
                )));
            }
            if file.tensor
                && if file.relative.starts_with("adapter:") {
                    stored_tensor_bytes(&file.absolute)?
                } else {
                    physical_resident_bytes(
                        &file.absolute,
                        self.backend,
                        self.route,
                        self.tier,
                        component_for(self.route, &file.relative),
                    )?
                } != file.resident_bytes
            {
                return Err(crate::Error::Unsupported(format!(
                    "{} tensor geometry changed after admission",
                    file.absolute.display()
                )));
            }
        }
        Ok(())
    }
}

pub(super) fn contract(
    provider_id: &str,
    route: WanI2vRoute,
    backend: WanI2vBackend,
    spec: &LoadSpec,
    facts: MemoryAssetFacts,
) -> MemoryProviderContract {
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    MemoryProviderContract {
        phase_facts: None,
        architecture_facts: crate::MemoryArchitectureFacts::default(),
        provider_id: provider_id.to_owned(),
        backend: backend.realization(),
        strategies: MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: if matches!(
                    strategy,
                    MemoryStrategy::Resident | MemoryStrategy::BoundedDecode
                ) || ((route != WanI2vRoute::Vace || backend == WanI2vBackend::Mlx)
                    && strategy == MemoryStrategy::StagedResidency)
                {
                    MemoryStrategySupport::Implemented
                } else {
                    MemoryStrategySupport::Missing
                },
                parameters: if strategy == MemoryStrategy::BoundedDecode {
                    MemoryParameterRanges {
                        decode_tile_edges: DECODE_TILE_EDGES.to_vec(),
                        decode_overlaps: DECODE_OVERLAPS.to_vec(),
                        ..Default::default()
                    }
                } else {
                    MemoryParameterRanges::default()
                },
            })
            .collect(),
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: ResidentRequestMemory::ExplicitResident,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: false,
            transformer_window_materialization: false,
        },
        formula: MemoryFormulaKind::PhaseEnvelope {
            phases,
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::OverlayBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::FrameCount,
                MemoryFormulaVariable::BatchCount,
            ],
        },
        calibration: None,
        asset_facts: facts,
        runtime: crate::MemoryRuntimeSemantics::default(),
    }
}
