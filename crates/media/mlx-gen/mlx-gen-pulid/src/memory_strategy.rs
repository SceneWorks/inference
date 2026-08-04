//! Shared image-memory ladder for the bespoke PuLID-FLUX MLX route.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{
    standard_memory_strategy_safety_check, Error as CoreError, MemoryBehaviorFixture,
    MemoryBehaviorRoute, MemoryCalibrationIdentity, MemoryMode, MemoryNumericTier,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, Result as CoreResult,
};
use mlx_gen::{LoadSpec, OffloadPolicy, WeightsSource};
use sha2::{Digest, Sha256};

pub const MEMORY_CALIBRATION_FINGERPRINT: &str =
    "pulid-flux-dev-q4-mlx-shared-ladder-2026-08-04-v1";
const STATIC_CALIBRATION: &str = "pulid-flux-static-registry-behavior-v1";
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen_flux::memory_strategy::ATTENTION_CHUNK_SIZE;
pub const DECODE_TILE_EDGES: &[u32] = mlx_gen_flux::memory_strategy::DECODE_TILE_EDGES;
pub const DECODE_OVERLAP: u32 = mlx_gen_flux::memory_strategy::DECODE_OVERLAP;

const EXPECTED: &[&str] = &[
    "92c41c3af322b02e58e1b32842e4601e08c8f16ec1fe80089dbe957df510f51d",
    "69417ca25d214f5e93704475f3e2a7e235fd1f1bf4a5acdf3f39ec69b51513cd",
    "7b40147a85771139e70a8d9fe6be27ffcf32f4c911770ef24b5b05c29f534eda",
    "9deff2fef8fe1b3e357a99c01f28cc478dd8acbeab0d3749d252f6d69990ee39",
    "c6b6f092ab266b13593ae0fbbc21c4546877577f6b1d2ae707f25d62ee767cac",
];

fn path(source: &WeightsSource) -> &Path {
    match source {
        WeightsSource::File(path) | WeightsSource::Dir(path) => path,
    }
}

fn identity_paths(spec: &LoadSpec) -> CoreResult<Vec<PathBuf>> {
    let identity = spec.identity.as_ref().ok_or_else(|| {
        CoreError::Unsupported("pulid_flux: exact contract requires identity weights".into())
    })?;
    let encoder = identity
        .encoder
        .as_ref()
        .ok_or_else(|| CoreError::Unsupported("pulid_flux: missing identity encoder".into()))?;
    let eva = identity
        .eva
        .as_ref()
        .ok_or_else(|| CoreError::Unsupported("pulid_flux: missing EVA tower".into()))?;
    let face = identity
        .face_dir
        .as_ref()
        .ok_or_else(|| CoreError::Unsupported("pulid_flux: missing face_dir".into()))?;
    Ok(vec![
        path(encoder).into(),
        path(eva).into(),
        path(face).join("scrfd_10g.safetensors"),
        path(face).join("arcface_iresnet100.safetensors"),
        path(face).join("bisenet_parsing.safetensors"),
    ])
}

fn hash(path: &Path) -> CoreResult<String> {
    let mut file = File::open(path).map_err(|error| {
        CoreError::Unsupported(format!("pulid_flux: open {}: {error}", path.display()))
    })?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).map_err(|error| {
            CoreError::Unsupported(format!("pulid_flux: hash {}: {error}", path.display()))
        })?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[derive(Clone)]
pub(crate) struct IdentityArtifactInventory {
    files: Vec<(PathBuf, PathBuf, String)>,
    bytes: u64,
    exact_calibration: bool,
}

impl IdentityArtifactInventory {
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn is_exact_calibration(&self) -> bool {
        self.exact_calibration
    }

    pub(crate) fn ensure_unchanged(&self) -> CoreResult<()> {
        for (path, admitted_path, expected) in &self.files {
            let current_path = path.canonicalize().map_err(|error| {
                CoreError::Unsupported(format!(
                    "pulid_flux: resolve {} after admission: {error}",
                    path.display()
                ))
            })?;
            if current_path != *admitted_path
                || !admitted_path.is_file()
                || hash(admitted_path)? != *expected
            {
                return Err(CoreError::Unsupported(format!(
                    "pulid_flux: pinned identity artifact changed after admission: {}",
                    path.display()
                )));
            }
        }
        Ok(())
    }
}

pub(crate) fn admitted_identity_inventory(
    spec: &LoadSpec,
) -> CoreResult<IdentityArtifactInventory> {
    let mut bytes = 0_u64;
    let mut exact_calibration = true;
    let files = identity_paths(spec)?
        .into_iter()
        .zip(EXPECTED.iter().copied())
        .map(|(path, expected)| {
            let admitted_path = path.canonicalize().map_err(|error| {
                CoreError::Unsupported(format!(
                    "pulid_flux: resolve {} during admission: {error}",
                    path.display()
                ))
            })?;
            let admitted_hash = hash(&admitted_path)?;
            Ok((path, admitted_path, admitted_hash, expected))
        })
        .collect::<CoreResult<Vec<_>>>()?;
    for (_, admitted_path, admitted_hash, expected) in &files {
        exact_calibration &= admitted_hash == expected;
        bytes = bytes.saturating_add(
            admitted_path
                .metadata()
                .map_err(|error| CoreError::Unsupported(error.to_string()))?
                .len(),
        );
    }
    let inventory = IdentityArtifactInventory {
        files: files
            .into_iter()
            .map(|(path, admitted_path, admitted_hash, _)| (path, admitted_path, admitted_hash))
            .collect(),
        bytes,
        exact_calibration,
    };
    inventory.ensure_unchanged()?;
    Ok(inventory)
}

pub(crate) fn backbone_spec(spec: &LoadSpec) -> LoadSpec {
    let mut base = spec.clone();
    base.identity = None;
    base
}

fn adapt(
    mut contract: MemoryProviderContract,
    spec: &LoadSpec,
    identity_bytes: Option<u64>,
    weights_free: bool,
) -> MemoryProviderContract {
    contract.provider_id = crate::pulid_flux::MODEL_ID.into();
    contract.asset_facts.base_bytes = contract
        .asset_facts
        .base_bytes
        .saturating_add(identity_bytes.unwrap_or(0));
    contract.calibration = if weights_free {
        Some(MemoryCalibrationIdentity::new(
            STATIC_CALIBRATION,
            spec.load_shape,
        ))
    } else if identity_bytes.is_some() && contract.calibration.is_some() {
        Some(MemoryCalibrationIdentity::new(
            MEMORY_CALIBRATION_FINGERPRINT,
            spec.load_shape,
        ))
    } else {
        None
    };
    contract
}

pub fn contract_for(spec: &LoadSpec) -> CoreResult<MemoryProviderContract> {
    let inventory = admitted_identity_inventory(spec)?;
    contract_with_inventory(spec, &inventory)
}

pub(crate) fn contract_with_inventory(
    spec: &LoadSpec,
    inventory: &IdentityArtifactInventory,
) -> CoreResult<MemoryProviderContract> {
    let base = backbone_spec(spec);
    let contract =
        mlx_gen_flux::memory_strategy::memory_strategy_contract(mlx_gen_flux::FLUX1_DEV_ID, &base)?;
    Ok(adapt(
        contract,
        spec,
        inventory.is_exact_calibration().then(|| inventory.bytes()),
        false,
    ))
}

pub fn weights_free_contract(spec: &LoadSpec) -> CoreResult<MemoryProviderContract> {
    let base = backbone_spec(spec);
    let contract = mlx_gen_flux::memory_strategy::weights_free_memory_strategy_contract(
        mlx_gen_flux::FLUX1_DEV_ID,
        &base,
    )?;
    Ok(adapt(contract, spec, Some(0), true))
}

pub fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route = || {
        if context.mode != MemoryMode::ImageToImage
            || context.geometry.reference_count != 1
            || context.use_pid
            || context.overlay.as_deref() != Some("identity")
        {
            return Err(CoreError::Unsupported("pulid_flux: memory route requires one identity reference, overlay=identity, and native VAE decode".into()));
        }
        if contract
            .calibration
            .as_ref()
            .is_some_and(|id| id.fingerprint == MEMORY_CALIBRATION_FINGERPRINT)
            && context.selection.strategy.is_optimized()
            && (context.geometry.width != 1024
                || context.geometry.height != 1024
                || context.geometry.batch != 1
                || context.geometry.frames != 1)
        {
            return Err(CoreError::Unsupported("pulid_flux: calibrated geometry is exactly 1024x1024, batch 1, one frame/reference".into()));
        }
        Ok(())
    };
    standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        }),
        Some(&route),
    )
}

pub fn registered_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::ImageToImage,
            reference_count: 1,
            use_pid: false,
            has_phases: spec.offload_policy == OffloadPolicy::Sequential,
            overlay: Some("identity".into()),
        },
    )?;
    Ok(vec![MemoryBehaviorFixture::new(context)])
}

pub fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_with(
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub fn begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_with(
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

fn begin_with(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        crate::pulid_flux::MODEL_ID,
        context.geometry,
        contract.generation_memory(&context.selection),
        false,
        57,
        |_pid, edge, overlap| {
            if DECODE_TILE_EDGES.contains(&edge) && overlap == DECODE_OVERLAP {
                Ok(())
            } else {
                Err(CoreError::Unsupported(format!(
                    "pulid_flux: decode tile {edge}/{overlap} is outside the production domain"
                )))
            }
        },
    )?;
    config.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE);
    config.transformer_window = contract
        .engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .then_some(context.selection.parameters.transformer_window_size)
        .flatten();
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{IdentityWeights, LoadShape, Quant};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn spec() -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_quant(Quant::Q4)
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        spec.identity = Some(IdentityWeights {
            encoder: Some(WeightsSource::File("/encoder".into())),
            eva: Some(WeightsSource::File("/eva".into())),
            face_dir: Some(WeightsSource::Dir("/face".into())),
        });
        spec
    }

    #[test]
    fn bespoke_contract_declares_every_rung_and_identity_route() {
        let spec = spec();
        let contract = weights_free_contract(&spec).unwrap();
        assert_eq!(contract.provider_id, crate::pulid_flux::MODEL_ID);
        assert!(contract.conformance_errors().is_empty());
        for strategy in MemoryStrategy::ALL {
            assert!(matches!(
                contract.capability(strategy).unwrap().support,
                mlx_gen::gen_core::MemoryStrategySupport::Implemented
            ));
        }
        let fixture = registered_fixture(
            &spec,
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .remove(0);
        assert_eq!(
            safety_check(&spec, &contract, &fixture.context),
            MemorySafetyDecision::Accept
        );
        let mut wrong = fixture.context;
        wrong.overlay = None;
        assert!(matches!(
            safety_check(&spec, &contract, &wrong),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn noncanonical_identity_is_admitted_uncalibrated_and_pinned() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "mlx-gen-pulid-identity-inventory-{}-{unique}",
            std::process::id()
        ));
        let face = root.join("face");
        std::fs::create_dir_all(&face).unwrap();
        let encoder = root.join("encoder.safetensors");
        let eva = root.join("eva.safetensors");
        std::fs::write(&encoder, b"compatible noncanonical encoder").unwrap();
        std::fs::write(&eva, b"compatible noncanonical eva").unwrap();
        for name in [
            "scrfd_10g.safetensors",
            "arcface_iresnet100.safetensors",
            "bisenet_parsing.safetensors",
        ] {
            std::fs::write(face.join(name), name.as_bytes()).unwrap();
        }

        let mut spec = spec();
        spec.identity = Some(IdentityWeights {
            encoder: Some(WeightsSource::File(encoder.clone())),
            eva: Some(WeightsSource::File(eva)),
            face_dir: Some(WeightsSource::Dir(face)),
        });
        let inventory = admitted_identity_inventory(&spec).unwrap();
        assert!(!inventory.is_exact_calibration());
        inventory.ensure_unchanged().unwrap();

        std::fs::write(&encoder, b"mutated noncanonical encoder").unwrap();
        assert!(inventory.ensure_unchanged().is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
