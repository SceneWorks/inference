//! Exact Candle/CUDA SDXL-family memory contract (SC-20793).
//!
//! The five catalog routes share code, not identity.  An adopting load must name one immutable
//! route, resolve from its immutable snapshot, prove its packed/dense tensor form, and retain a
//! complete content seal for every lazily opened base, component, control, IP, PiD, and ordered
//! adapter source.  Legacy loads without `resolved_route` remain resident-only.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use candle_gen::gen_core::{
    self, AdapterResidencyMode, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategyCapability, MemoryStrategySupport, MemoryWindowMaterialization, Precision, Quant,
    WeightsSource,
};
use sha2::{Digest, Sha256};

pub const REQUEST_EVIDENCE_REVISION: &str = "sdxl-candle-request-contract-v1";
const DECODE_TILE_EDGE: u32 = 512;
const DECODE_OVERLAP: u32 = 64;
const ATTENTION_CHUNK_SIZE: u32 = gen_core::attention_budget::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
const CLIP_L_REPO: &str = "openai/clip-vit-large-patch14";
const CLIP_L_REVISION: &str = "32bd64288804d66eefd0ccbe215aa642df71cc41";
const CLIP_BIGG_REPO: &str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k";
const CLIP_BIGG_REVISION: &str = "743c27bd53dfe508a0ade0f50698f99b39d03bec";
const VAE_REPO: &str = "madebyollin/sdxl-vae-fp16-fix";
const VAE_REVISION: &str = "207b116dae70ace3637169f1ddd2434b91b3a8cd";
const IP_REPO: &str = "h94/IP-Adapter";
const IP_REVISION: &str = "018e402774aeeddd60609b4ecdb7e298259dc729";
const TILE_CONTROL_REPO: &str = "xinsir/controlnet-tile-sdxl-1.0";
const TILE_CONTROL_REVISION: &str = "1ae8d9529efe58f7362a987363ff86a7904dc84f";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SdxlRoute {
    pub id: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
    pub edit: bool,
    pub lightning: bool,
}

pub const SDXL_ROUTES: &[SdxlRoute] = &[
    SdxlRoute {
        id: "sdxl",
        repository: "SceneWorks/sdxl-base-mlx",
        revision: "36699bb8a6353e61c920e3bf19f0e6f8e4151c55",
        edit: true,
        lightning: false,
    },
    SdxlRoute {
        id: "realvisxl",
        repository: "SceneWorks/realvisxl-mlx",
        revision: "e40202d63baef826c7df95a639a811698c1178d2",
        edit: true,
        lightning: false,
    },
    SdxlRoute {
        id: "realvisxl_lightning",
        repository: "SceneWorks/realvisxl-lightning-mlx",
        revision: "c09fd586989bdc3c658d4acd03e8ae81677ade8e",
        edit: false,
        lightning: true,
    },
    SdxlRoute {
        id: "illustrious_xl_v1",
        repository: "SceneWorks/illustrious-xl-v1-mlx",
        revision: "c5a92a902dd4e6ee99c2a57981ecf66209905dd1",
        edit: true,
        lightning: false,
    },
    SdxlRoute {
        id: "illustrious_xl_v2",
        repository: "SceneWorks/illustrious-xl-v2-mlx",
        revision: "7c5c8b2bb75a8f38a7365e70bdf84d38d6204473",
        edit: true,
        lightning: false,
    },
];

/// Which SDXL execution surface a contract describes.
///
/// The five routes share one `provider_id`, but the *registered* generator and the *bespoke*
/// eagerly-assembled providers (edit / IP-Adapter / detail) are different executables with
/// different rung support. A single contract that claimed the union would let a selector pick a
/// rung the bespoke loader hard-refuses at admission, so the surface is part of the contract's
/// identity rather than a caller-side convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdxlSurface {
    /// The registered generator: text-to-image, hires, img2img/edit and control.
    Registered,
    /// The eagerly-assembled `edit` / `ip` / `detail` providers, which materialize their whole
    /// stack up front and therefore cannot stage component residency across phases.
    Bespoke,
}

/// A route id as kebab tokens that cannot be mistaken for the fingerprint's version token.
///
/// `gen_core::validate_calibration_fingerprint` requires **exactly one** `vN` token, and two of
/// the five route ids already end in one (`illustrious_xl_v1`, `illustrious_xl_v2`). Rendered
/// naively those produce `…-illustrious-xl-v1-…-v1`, which the validator rejects — a defect the
/// old weights-free witness hid by only ever minting the base route's fingerprint. The model
/// revision keeps its number and reads as `rev1`/`rev2`, leaving the trailing `v1` as the sole
/// semantics version.
fn route_identity_tokens(route: SdxlRoute) -> String {
    route
        .id
        .replace('_', "-")
        .split('-')
        .map(|token| match token.strip_prefix('v') {
            Some(digits) if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) => {
                format!("rev{digits}")
            }
            _ => token.to_owned(),
        })
        .collect::<Vec<_>>()
        .join("-")
}

/// The exact calibration fingerprint naming one route's executable memory semantics.
///
/// This is the *whole* identity, not a prefix: `realvisxl` and `realvisxl_lightning` both render
/// `sdxl-candle-realvisxl…`, so any substring or prefix comparison silently resolves the lightning
/// route to `realvisxl` — which declares `edit: true` — and admits the edit/inpaint/detail/
/// character modes the lightning route's own row refuses.
fn route_fingerprint(route: SdxlRoute) -> String {
    format!(
        "sdxl-candle-{}-staged-decode-attention-v1",
        route_identity_tokens(route)
    )
}

/// Resolve the route a contract names from its complete calibration fingerprint.
fn route_from_fingerprint(fingerprint: &str) -> Option<SdxlRoute> {
    SDXL_ROUTES
        .iter()
        .copied()
        .find(|route| route_fingerprint(*route) == fingerprint)
}

/// The bounded-decode geometry for one request: the **selected** parameters when the selector
/// engaged the rung, otherwise this provider's own declared constants.
///
/// sc-20799: the contract declares `decode_tile_edges`/`decode_overlaps` from these constants and
/// [`request_scope`] re-validates exactly them, but the decoder executed a second hardcoded
/// `512/128` pair that no part of the contract named. Routing execution through the declaration
/// makes declared, validated and executed one value.
///
/// This is a function rather than two `pub(crate)` constants on purpose: `mlx-gen-sdxl` declares
/// its own Metal-measured `DECODE_TILE_EDGE`, and giving these visibility would mint a new
/// cross-backend shared-declaration claim that is not one.
pub(crate) fn decode_tiling_config(
    memory: Option<gen_core::GenerationMemory>,
) -> gen_core::tiling::TilingConfig {
    let edge = memory
        .and_then(|memory| memory.decode_tile_edge)
        .unwrap_or(DECODE_TILE_EDGE);
    let overlap = memory
        .and_then(|memory| memory.decode_overlap)
        .unwrap_or(DECODE_OVERLAP);
    gen_core::tiling::TilingConfig::spatial_only(edge as i32, overlap as i32)
}

pub(crate) fn backend() -> MemoryBackendRealization {
    MemoryBackendRealization::CandleCuda {
        device_residency: true,
        host_backed_weights: true,
        host_to_device_block_materialization: false,
        block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
    }
}

fn route(spec: &LoadSpec) -> gen_core::Result<SdxlRoute> {
    let id = spec.resolved_route.as_deref().ok_or_else(|| {
        gen_core::Error::Unsupported(
            "sdxl: optimized memory admission requires an exact resolved catalog route".into(),
        )
    })?;
    SDXL_ROUTES
        .iter()
        .copied()
        .find(|route| route.id == id)
        .ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "sdxl: resolved route {id:?} is not one of the five immutable SDXL routes"
            ))
        })
}

fn source_path(source: &WeightsSource) -> &Path {
    match source {
        WeightsSource::Dir(path) | WeightsSource::File(path) => path,
    }
}

fn path_has_snapshot(path: &Path, route: SdxlRoute, tier: &str) -> bool {
    let components = path
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let marker = format!("models--{}", route.repository.replace('/', "--"));
    let name = route
        .repository
        .rsplit('/')
        .next()
        .unwrap_or(route.repository);
    components
        .windows(4)
        .any(|window| window == [marker.as_str(), "snapshots", route.revision, tier])
        || components
            .windows(3)
            .any(|window| window == [name, route.revision, tier])
        || components.windows(3).any(|window| {
            window
                == [
                    format!("{}__{}", "SceneWorks", name),
                    route.revision.to_owned(),
                    tier.to_owned(),
                ]
        })
}

fn path_has_hf_revision(path: &Path, repository: &str, revision: &str) -> bool {
    let marker = format!("models--{}", repository.replace('/', "--"));
    let components = path
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    components
        .windows(3)
        .any(|window| window == [marker.as_str(), "snapshots", revision])
}

fn validate_shared_component_revisions(spec: &LoadSpec) -> gen_core::Result<()> {
    for (id, repository, revision) in [
        ("tokenizer_clip_l", CLIP_L_REPO, CLIP_L_REVISION),
        ("tokenizer_clip_bigg", CLIP_BIGG_REPO, CLIP_BIGG_REVISION),
        ("vae_fp16_fix", VAE_REPO, VAE_REVISION),
    ] {
        let source = spec.components.get(id).ok_or_else(|| {
            gen_core::Error::Unsupported(format!("sdxl: missing required component {id}"))
        })?;
        if !path_has_hf_revision(source_path(source), repository, revision) {
            return Err(gen_core::Error::Unsupported(format!(
                "sdxl: component {id} must resolve from exact {repository}@{revision}"
            )));
        }
    }
    Ok(())
}

fn packed_bits(root: &Path) -> gen_core::Result<Option<u8>> {
    let mut result = None;
    for component in ["unet", "text_encoder", "text_encoder_2"] {
        let config = root.join(component).join("config.json");
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config).map_err(|error| {
                gen_core::Error::Msg(format!("sdxl: read {}: {error}", config.display()))
            })?)
            .map_err(|error| {
                gen_core::Error::Msg(format!("sdxl: parse {}: {error}", config.display()))
            })?;
        let bits =
            candle_gen::quant::PackedConfig::from_config(&value).map(|packed| packed.bits as u8);
        if result.is_some() && bits != result {
            return Err(gen_core::Error::Unsupported(format!(
                "sdxl: crossed packed configuration between UNet and CLIP ({result:?} vs {bits:?})"
            )));
        }
        result = bits;
    }
    if let Some(bits) = result {
        if !matches!(bits, 4 | 8) {
            return Err(gen_core::Error::Unsupported(format!(
                "sdxl: packed tier declares unsupported {bits}-bit tensors"
            )));
        }
        validate_packed_headers(root, bits)?;
    } else {
        validate_dense_headers(root)?;
    }
    Ok(result)
}

fn main_tensor_paths(root: &Path, packed: bool) -> [PathBuf; 3] {
    if packed {
        [
            root.join("unet/diffusion_pytorch_model.safetensors"),
            root.join("text_encoder/model.safetensors"),
            root.join("text_encoder_2/model.safetensors"),
        ]
    } else {
        [
            root.join("unet/diffusion_pytorch_model.fp16.safetensors"),
            root.join("text_encoder/model.fp16.safetensors"),
            root.join("text_encoder_2/model.fp16.safetensors"),
        ]
    }
}

fn validate_packed_headers(root: &Path, _bits: u8) -> gen_core::Result<()> {
    for path in main_tensor_paths(root, true) {
        let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(&path)?;
        let names = headers
            .iter()
            .map(|header| header.name.as_str())
            .collect::<BTreeSet<_>>();
        let mut triples = 0_usize;
        for header in &headers {
            if let Some(base) = header.name.strip_suffix(".scales") {
                let weight = format!("{base}.weight");
                let biases = format!("{base}.biases");
                let weight_header = headers.iter().find(|candidate| candidate.name == weight);
                if weight_header
                    .is_none_or(|candidate| candidate.dtype != gen_core::weightsmeta::Dtype::U32)
                    || !names.contains(biases.as_str())
                {
                    return Err(gen_core::Error::Unsupported(format!(
                        "sdxl: packed tensor {base} lacks the required U32 weight/scales/biases triple"
                    )));
                }
                triples += 1;
            }
        }
        if triples == 0 {
            return Err(gen_core::Error::Unsupported(format!(
                "sdxl: {} claims packing but contains no packed tensor triples",
                path.display()
            )));
        }
    }
    Ok(())
}

fn validate_dense_headers(root: &Path) -> gen_core::Result<()> {
    for path in main_tensor_paths(root, false) {
        let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(&path)?;
        if headers.is_empty() || headers.iter().any(|header| !header.is_float()) {
            return Err(gen_core::Error::Unsupported(format!(
                "sdxl: dense tier {} must contain only floating tensors",
                path.display()
            )));
        }
    }
    Ok(())
}

fn tier_for(root: &Path) -> gen_core::Result<MemoryNumericTier> {
    let bits = packed_bits(root)?;
    let quant = match bits {
        Some(4) => Some(Quant::Q4),
        Some(8) => Some(Quant::Q8),
        None => None,
        _ => unreachable!("packed_bits validates the width"),
    };
    Ok(MemoryNumericTier {
        // The provider materializes the nominal bf16 catalog tier as F16 deliberately.
        precision: Precision::Bf16,
        quant,
        component_precision_floors: &[],
    })
}

fn collect_files(path: &Path, out: &mut Vec<PathBuf>) -> gen_core::Result<()> {
    let path = std::path::absolute(path)?;
    let metadata = std::fs::metadata(&path)?;
    if metadata.is_file() {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if name.starts_with('.')
            || [".part", ".tmp", ".lock", ".incomplete"]
                .iter()
                .any(|suffix| name.ends_with(suffix))
        {
            return Err(gen_core::Error::Unsupported(format!(
                "sdxl: refusing incomplete or hidden artifact {}",
                path.display()
            )));
        }
        out.push(path);
        return Ok(());
    }
    if !metadata.is_dir() {
        return Err(gen_core::Error::Unsupported(format!(
            "sdxl: artifact source is neither a file nor directory: {}",
            path.display()
        )));
    }
    let mut children = std::fs::read_dir(&path)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    children.sort();
    for child in children {
        collect_files(&child, out)?;
    }
    Ok(())
}

fn sha256(path: &Path) -> gen_core::Result<[u8; 32]> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(hash.finalize().into())
}

#[derive(Clone, Debug)]
struct SealedFile {
    pin: gen_core::PinnedWeightsFile,
    sha256: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct SdxlArtifactSeal {
    contract: MemoryProviderContract,
    tier: MemoryNumericTier,
    roots: Vec<(PathBuf, Vec<PathBuf>)>,
    files: Vec<SealedFile>,
    overlay: Option<String>,
}

impl SdxlArtifactSeal {
    /// Seal the registered generator's surface.
    pub fn capture(spec: &LoadSpec) -> gen_core::Result<Self> {
        Self::capture_for(spec, SdxlSurface::Registered)
    }

    /// Seal one named execution surface. The bespoke providers must use
    /// [`SdxlSurface::Bespoke`] so the contract they publish declares the staged-residency rung
    /// they refuse at admission as `Missing` rather than `Implemented`.
    pub fn capture_for(spec: &LoadSpec, surface: SdxlSurface) -> gen_core::Result<Self> {
        spec.validate_prepared_file_pins()?;
        let route = route(spec)?;
        let WeightsSource::Dir(root) = &spec.weights else {
            return Err(gen_core::Error::Unsupported(
                "sdxl: catalog memory receipts require an immutable snapshot directory; imported files use the explicit imported route".into(),
            ));
        };
        let root = std::fs::canonicalize(root)?;
        let tier = tier_for(&root)?;
        if spec.quantize != tier.quant {
            return Err(gen_core::Error::Unsupported(format!(
                "sdxl: requested quantization {:?} does not equal physical tensor tier {:?}",
                spec.quantize, tier.quant
            )));
        }
        let tier_name = match tier.quant {
            Some(Quant::Q4) => "q4",
            Some(Quant::Q8) => "q8",
            None => "bf16",
            _ => unreachable!(),
        };
        if !path_has_snapshot(&root, route, tier_name) {
            return Err(gen_core::Error::Unsupported(format!(
                "sdxl: {} must resolve from exact {}@{}/{}",
                route.id, route.repository, route.revision, tier_name
            )));
        }
        validate_shared_component_revisions(spec)?;
        let mut sources = vec![root.clone()];
        sources.extend(
            spec.components
                .values()
                .map(source_path)
                .map(Path::to_path_buf),
        );
        sources.extend(spec.control.iter().map(source_path).map(Path::to_path_buf));
        sources.extend(
            spec.extra_controls
                .iter()
                .map(source_path)
                .map(Path::to_path_buf),
        );
        sources.extend(
            spec.ip_adapter
                .iter()
                .map(source_path)
                .map(Path::to_path_buf),
        );
        sources.extend(spec.adapters.iter().map(|adapter| adapter.path.clone()));
        if let Some(pid) = &spec.pid {
            sources.push(source_path(&pid.checkpoint).to_path_buf());
            sources.push(source_path(&pid.gemma).to_path_buf());
        }
        sources.sort();
        sources.dedup();

        let mut roots = Vec::new();
        let mut files = Vec::new();
        let mut receipt = Sha256::new();
        receipt.update(b"sdxl-candle-physical-receipt-v1");
        receipt.update(route.id.as_bytes());
        receipt.update(route.repository.as_bytes());
        receipt.update(route.revision.as_bytes());
        receipt.update(tier_name.as_bytes());
        for source in sources {
            let absolute = std::path::absolute(&source)?;
            let mut inventory = Vec::new();
            collect_files(&absolute, &mut inventory)?;
            inventory.sort();
            inventory.dedup();
            if inventory.is_empty() {
                return Err(gen_core::Error::Unsupported(format!(
                    "sdxl: sealed source {} is empty",
                    absolute.display()
                )));
            }
            for path in &inventory {
                let pin = gen_core::PinnedWeightsFile::pin(path)?;
                let digest = pin.read_unchanged(sha256)?;
                receipt.update((path.as_os_str().len() as u64).to_le_bytes());
                receipt.update(path.as_os_str().as_encoded_bytes());
                receipt.update(digest);
                files.push(SealedFile {
                    pin,
                    sha256: digest,
                });
            }
            roots.push((absolute, inventory));
        }
        let overlay_identity = load_overlay_identity(spec);
        if let Some(identity) = &overlay_identity {
            receipt.update(identity.as_bytes());
        }
        let receipt = format!("{:x}", receipt.finalize());
        let facts = asset_facts(spec, &root, tier)?;
        let contract = build_contract(spec, surface, tier, facts, route_fingerprint(route));
        let seal = Self {
            contract,
            tier,
            roots,
            files,
            overlay: overlay_identity,
        };
        // Keep the physical digest live even though calibration identity names executable memory
        // semantics rather than mutable installation bytes. Every file digest remains retained in
        // `files` and revalidated before lazy materialization.
        debug_assert_eq!(receipt.len(), 64);
        seal.ensure_unchanged()?;
        Ok(seal)
    }

    pub fn contract(&self) -> &MemoryProviderContract {
        &self.contract
    }

    pub fn tier(&self) -> MemoryNumericTier {
        self.tier
    }

    pub fn overlay_identity(&self) -> Option<&str> {
        self.overlay.as_deref()
    }

    pub fn ensure_unchanged(&self) -> gen_core::Result<()> {
        for (root, expected) in &self.roots {
            let mut current = Vec::new();
            collect_files(root, &mut current)?;
            current.sort();
            current.dedup();
            if &current != expected {
                return Err(gen_core::Error::Unsupported(format!(
                    "sdxl: artifact inventory changed after admission: {}",
                    root.display()
                )));
            }
        }
        for file in &self.files {
            file.pin.ensure_unchanged()?;
            if file.pin.read_unchanged(sha256)? != file.sha256 {
                return Err(gen_core::Error::Unsupported(format!(
                    "sdxl: artifact content changed after admission: {}",
                    file.pin.loader_path().display()
                )));
            }
        }
        Ok(())
    }
}

fn load_overlay_identity(spec: &LoadSpec) -> Option<String> {
    let mut parts = Vec::new();
    if spec.control.is_some() {
        parts.push(format!("control:{}", 1 + spec.extra_controls.len()));
    }
    if spec.ip_adapter.is_some() {
        parts.push("ip-adapter".to_owned());
    }
    if spec.pid.is_some() {
        parts.push("pid".to_owned());
    }
    if let Some(adapters) = gen_core::adapter_stack_identity(&spec.adapters) {
        parts.push(adapters);
    }
    (!parts.is_empty()).then(|| parts.join("+"))
}

fn tensor_bytes(path: &Path, float_width: u64) -> gen_core::Result<u64> {
    gen_core::weightsmeta::safetensors_path_tensor_headers(path)?
        .iter()
        .try_fold(0_u64, |sum, header| {
            let bytes = if header.is_float() {
                header.materialized_bytes(float_width)?
            } else {
                header.data_bytes
            };
            sum.checked_add(bytes)
                .ok_or_else(|| gen_core::Error::Msg("sdxl: tensor byte sum overflow".into()))
        })
}

fn source_tensor_bytes(path: &Path, float_width: u64) -> gen_core::Result<u64> {
    let mut files = Vec::new();
    collect_files(path, &mut files)?;
    files
        .into_iter()
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))
        .try_fold(0_u64, |sum, path| {
            sum.checked_add(tensor_bytes(&path, float_width)?)
                .ok_or_else(|| gen_core::Error::Msg("sdxl: component byte sum overflow".into()))
        })
}

fn asset_facts(
    spec: &LoadSpec,
    root: &Path,
    tier: MemoryNumericTier,
) -> gen_core::Result<MemoryAssetFacts> {
    let width = 2;
    let conditioning = source_tensor_bytes(&root.join("text_encoder"), width)?
        .saturating_add(source_tensor_bytes(&root.join("text_encoder_2"), width)?);
    let transformer = source_tensor_bytes(&root.join("unet"), width)?;
    let fallback_decoder = root.join("vae");
    let decoder_source = spec
        .components
        .get("vae_fp16_fix")
        .map(source_path)
        .unwrap_or(&fallback_decoder);
    let decoder = source_tensor_bytes(decoder_source, 4)?;
    let mut overlay = spec
        .control
        .iter()
        .chain(spec.extra_controls.iter())
        .chain(spec.ip_adapter.iter())
        .map(source_path)
        .try_fold(0_u64, |sum, path| {
            Ok::<_, gen_core::Error>(sum.saturating_add(source_tensor_bytes(path, width)?))
        })?;
    let adapter_mode = if tier.quant.is_some() {
        AdapterResidencyMode::Additive
    } else {
        AdapterResidencyMode::Folded
    };
    overlay = overlay.saturating_add(
        gen_core::adapter_stack_resident_bytes(&spec.adapters, adapter_mode).ok_or_else(|| {
            gen_core::Error::Unsupported(
                "sdxl: every additive packed adapter must have an exact non-zero size".into(),
            )
        })?,
    );
    Ok(MemoryAssetFacts {
        base_bytes: conditioning
            .saturating_add(transformer)
            .saturating_add(decoder),
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: decoder,
        overlay_bytes: overlay,
    })
}

fn build_contract(
    spec: &LoadSpec,
    surface: SdxlSurface,
    _tier: MemoryNumericTier,
    asset_facts: MemoryAssetFacts,
    fingerprint: String,
) -> MemoryProviderContract {
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    MemoryProviderContract {
        provider_id: crate::MODEL_ID.to_owned(),
        backend: backend(),
        strategies: MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                // The UNet trunk is never host-backed on this route, so block windowing is
                // Missing on every surface. Staged residency is Missing on the bespoke surface:
                // the edit / IP / detail providers assemble their whole stack eagerly and
                // `validate_bespoke_context` refuses that rung, so declaring it Implemented would
                // advertise a rung the loader hard-refuses.
                support: if strategy == MemoryStrategy::BoundedTransformerResidency
                    || (surface == SdxlSurface::Bespoke
                        && strategy == MemoryStrategy::StagedResidency)
                {
                    MemoryStrategySupport::Missing
                } else {
                    MemoryStrategySupport::Implemented
                },
                parameters: match strategy {
                    MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                        decode_tile_edges: vec![DECODE_TILE_EDGE],
                        decode_overlaps: vec![DECODE_OVERLAP],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                        attention_chunk_sizes: vec![ATTENTION_CHUNK_SIZE],
                        ..Default::default()
                    },
                    _ => MemoryParameterRanges::default(),
                },
            })
            .collect(),
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: gen_core::ResidentRequestMemory::ExplicitResident,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: false,
        },
        formula: MemoryFormulaKind::PhaseEnvelope {
            phases,
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::BatchCount,
                MemoryFormulaVariable::ConditioningTokenCount,
                MemoryFormulaVariable::OverlayBytes,
                MemoryFormulaVariable::DecodeTileArea,
                MemoryFormulaVariable::AttentionChunkSize,
            ],
        },
        calibration: Some(MemoryCalibrationIdentity::new(fingerprint, spec.load_shape)),
        asset_facts,
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    }
}

pub fn provider_contract_for_spec(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    Ok(SdxlArtifactSeal::capture(spec)?.contract)
}

pub fn resolved_numeric_tier(spec: &LoadSpec) -> gen_core::Result<MemoryNumericTier> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(gen_core::Error::Unsupported(
            "sdxl: imported fused sources do not borrow catalog tier identity".into(),
        ));
    };
    tier_for(root)
}

fn same_source(left: Option<&WeightsSource>, right: &WeightsSource) -> bool {
    left.is_some_and(|left| source_path(left) == source_path(right))
}

fn validate_bespoke_spec_common(
    base: &Path,
    adapters: &[gen_core::AdapterSpec],
    tokenizer_clip_l: &WeightsSource,
    tokenizer_clip_bigg: &WeightsSource,
    vae_fp16_fix: &WeightsSource,
    spec: &LoadSpec,
) -> gen_core::Result<()> {
    if !matches!(&spec.weights, WeightsSource::Dir(path) if path == base)
        || gen_core::adapter_stack_identity(&spec.adapters)
            != gen_core::adapter_stack_identity(adapters)
        || !same_source(spec.components.get("tokenizer_clip_l"), tokenizer_clip_l)
        || !same_source(
            spec.components.get("tokenizer_clip_bigg"),
            tokenizer_clip_bigg,
        )
        || !same_source(spec.components.get("vae_fp16_fix"), vae_fp16_fix)
    {
        return Err(gen_core::Error::Unsupported(
            "sdxl: bespoke provider paths do not equal the exact admitted LoadSpec assembly".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_edit_spec(
    paths: &crate::SdxlEditPaths,
    spec: &LoadSpec,
) -> gen_core::Result<()> {
    validate_bespoke_spec_common(
        &paths.sdxl_base,
        &paths.adapters,
        &paths.tokenizer_clip_l,
        &paths.tokenizer_clip_bigg,
        &paths.vae_fp16_fix,
        spec,
    )?;
    if spec.control.is_some() || spec.ip_adapter.is_some() || !spec.extra_controls.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "sdxl edit: crossed control/IP assembly".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_ip_spec(
    paths: &crate::IpAdapterSdxlPaths,
    spec: &LoadSpec,
) -> gen_core::Result<()> {
    validate_bespoke_spec_common(
        &paths.sdxl_base,
        &paths.adapters,
        &paths.tokenizer_clip_l,
        &paths.tokenizer_clip_bigg,
        &paths.vae_fp16_fix,
        spec,
    )?;
    if !path_has_hf_revision(&paths.ip_adapter, IP_REPO, IP_REVISION)
        || !path_has_hf_revision(&paths.image_encoder, IP_REPO, IP_REVISION)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "sdxl IP-Adapter components must resolve from exact {IP_REPO}@{IP_REVISION}"
        )));
    }
    if !same_source(
        spec.ip_adapter.as_ref(),
        &WeightsSource::File(paths.ip_adapter.clone()),
    ) || !same_source(
        spec.components.get("sdxl_ip_image_encoder"),
        &WeightsSource::Dir(paths.image_encoder.clone()),
    ) || spec.control.is_some()
        || !spec.extra_controls.is_empty()
    {
        return Err(gen_core::Error::Unsupported(
            "sdxl IP-Adapter: crossed or incomplete IP assembly".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_detail_spec(
    paths: &crate::SdxlDetailPaths,
    spec: &LoadSpec,
) -> gen_core::Result<()> {
    validate_bespoke_spec_common(
        &paths.sdxl_base,
        &paths.adapters,
        &paths.tokenizer_clip_l,
        &paths.tokenizer_clip_bigg,
        &paths.vae_fp16_fix,
        spec,
    )?;
    if !path_has_hf_revision(
        source_path(&paths.tile_controlnet),
        TILE_CONTROL_REPO,
        TILE_CONTROL_REVISION,
    ) {
        return Err(gen_core::Error::Unsupported(format!(
            "sdxl detail ControlNet must resolve from exact {TILE_CONTROL_REPO}@{TILE_CONTROL_REVISION}"
        )));
    }
    if !same_source(spec.control.as_ref(), &paths.tile_controlnet)
        || spec.ip_adapter.is_some()
        || !spec.extra_controls.is_empty()
    {
        return Err(gen_core::Error::Unsupported(
            "sdxl detail: crossed or incomplete tile-ControlNet assembly".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_bespoke_request(
    seal: &SdxlArtifactSeal,
    context: &MemoryRunContext,
    width: u32,
    height: u32,
    reference_count: u32,
    use_pid: bool,
    mode: &str,
) -> gen_core::Result<()> {
    validate_bespoke_context(context)?;
    seal.ensure_unchanged()?;
    if context.geometry.width != width
        || context.geometry.height != height
        || context.geometry.batch != 1
        || context.geometry.frames != 1
        || context.geometry.reference_count != reference_count
        || context.use_pid != use_pid
    {
        return Err(gen_core::Error::Unsupported(
            "sdxl: bespoke request geometry/PiD identity changed after admission".into(),
        ));
    }
    let mode_matches = match (&context.mode, mode) {
        (MemoryMode::Edit, "edit") | (MemoryMode::ImageToImage, "edit") => true,
        (MemoryMode::Other(actual), expected) => actual == expected,
        _ => false,
    };
    if !mode_matches {
        return Err(gen_core::Error::Unsupported(format!(
            "sdxl: bespoke request mode {mode:?} does not match admitted {:?}",
            context.mode
        )));
    }
    Ok(())
}

pub(crate) fn validate_bespoke_context(context: &MemoryRunContext) -> gen_core::Result<()> {
    if context.selection.strategy == MemoryStrategy::StagedResidency {
        return Err(gen_core::Error::Unsupported(
            "sdxl: bespoke edit/IP/detail providers are eagerly assembled and do not implement staged residency"
                .into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_context(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    seal: &SdxlArtifactSeal,
) -> gen_core::Result<()> {
    seal.ensure_unchanged()?;
    validate_context_axes(contract, context, seal.tier, seal.overlay_identity())
}

fn validate_context_axes(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    tier: MemoryNumericTier,
    expected_overlay: Option<&str>,
) -> gen_core::Result<()> {
    match gen_core::standard_memory_strategy_safety_check(contract, context, Some(tier), None) {
        MemorySafetyDecision::Accept => {}
        MemorySafetyDecision::Reject { reason } => {
            return Err(gen_core::Error::Unsupported(reason));
        }
    }
    if context.geometry.width == 0
        || context.geometry.height == 0
        || context.geometry.batch == 0
        || context.geometry.frames != 1
        || context.has_reference != (context.geometry.reference_count > 0)
    {
        return Err(gen_core::Error::Unsupported(
            "sdxl: memory context has invalid image geometry/reference identity".into(),
        ));
    }
    let route_id = contract
        .calibration
        .as_ref()
        .map(|identity| identity.fingerprint.as_str())
        .unwrap_or_default();
    let route = route_from_fingerprint(route_id).ok_or_else(|| {
        gen_core::Error::Unsupported(format!(
            "sdxl: contract fingerprint {route_id:?} is not the exact identity of any SDXL route"
        ))
    })?;
    let refs = context.geometry.reference_count;
    match &context.mode {
        MemoryMode::TextToImage if refs == 0 => {}
        MemoryMode::TextToImage
            if refs > 0
                && context
                    .overlay
                    .as_deref()
                    .is_some_and(|overlay| overlay.starts_with("control:")) => {}
        MemoryMode::ImageToImage | MemoryMode::Edit if route.edit && refs == 1 => {}
        MemoryMode::Other(mode)
            if route.edit
                && matches!(
                    mode.as_str(),
                    "image_inpaint" | "image_detail" | "character_image" | "control_image"
                )
                && refs == 1 => {}
        MemoryMode::Other(mode) if mode == "hires" && refs == 0 && context.has_phases => {}
        _ => {
            return Err(gen_core::Error::Unsupported(format!(
                "sdxl: route {} does not admit mode {:?} with {refs} references/phases={}",
                route.id, context.mode, context.has_phases
            )));
        }
    }
    if context.has_phases && !matches!(&context.mode, MemoryMode::Other(mode) if mode == "hires") {
        return Err(gen_core::Error::Unsupported(
            "sdxl: only the exact Hires route may carry two phases".into(),
        ));
    }
    if context.use_pid && context.selection.strategy.is_optimized() {
        return Err(gen_core::Error::Unsupported(
            "sdxl: PiD uses a distinct decoder and cannot consume the native-VAE optimized plan"
                .into(),
        ));
    }
    if context.overlay.as_deref() != expected_overlay {
        return Err(gen_core::Error::Unsupported(format!(
            "sdxl: request overlay {:?} does not equal sealed ordered adapter identity {:?}",
            context.overlay, expected_overlay
        )));
    }
    Ok(())
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    seal: Option<&SdxlArtifactSeal>,
) -> MemorySafetyDecision {
    let Some(seal) = seal else {
        return MemorySafetyDecision::Reject {
            reason: "sdxl: optimized context has no exact artifact seal".into(),
        };
    };
    match validate_context(contract, context, seal) {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub(crate) fn request_scope(
    device: candle_gen::candle_core::Device,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<candle_gen::request_scope::CandleRequestScopeCore> {
    let mut config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        crate::MODEL_ID,
        device,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        0,
        |use_pid, edge, overlap| {
            if use_pid {
                return Err(gen_core::Error::Unsupported(
                    "sdxl: PiD has no admitted native-VAE tiled decode receipt".into(),
                ));
            }
            if edge == DECODE_TILE_EDGE && overlap == DECODE_OVERLAP {
                Ok(())
            } else {
                Err(gen_core::Error::Unsupported(format!(
                    "sdxl: decode tile {edge}/{overlap} is outside {DECODE_TILE_EDGE}/{DECODE_OVERLAP}"
                )))
            }
        },
    )?;
    config.attention_chunk_size = contract
        .engages_selection(&context.selection, MemoryStrategy::BoundedAttention)
        .then_some(context.selection.parameters.attention_chunk_size)
        .flatten();
    Ok(candle_gen::request_scope::CandleRequestScopeCore::new(
        config,
    ))
}

/// Admission for a contract minted without weights.
///
/// The overlay axis is taken from the spec, not pinned to `None`: an IP-Adapter / ControlNet /
/// PiD / LoRA spec carries an overlay identity that a sealed contract would have recorded, and
/// hardcoding `None` here refused every overlay-bearing request on the pre-load path while the
/// sealed path admitted it. The tier is the weights-free `bf16 + spec.quantize` witness — the
/// on-disk tier cannot be read without the artifact.
fn validate_weights_free_context(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    validate_context_axes(
        contract,
        context,
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: spec.quantize,
            component_precision_floors: &[],
        },
        load_overlay_identity(spec).as_deref(),
    )
}

pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let result = if contract.asset_facts == MemoryAssetFacts::default() {
        validate_weights_free_context(spec, contract, context)
    } else {
        SdxlArtifactSeal::capture(spec).and_then(|seal| validate_context(contract, context, &seal))
    };
    match result {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

/// Every route/mode the registered SDXL surface advertises, as executable behavior witnesses.
///
/// Plain text-to-image alone left the whole edit family — img2img/edit, inpaint, detail, character,
/// control and hires — advertised by [`validate_context_axes`] but never exercised by conformance,
/// so a regression in any of those admission arms was invisible. Each witness carries the exact
/// `LoadSpec` its overlay identity requires, because [`validate_weights_free_context`] compares the
/// request overlay against the spec's own overlay identity.
fn advertised_behavior_routes(
    spec: &LoadSpec,
    route: SdxlRoute,
    strategy: MemoryStrategy,
) -> Vec<(gen_core::MemoryBehaviorRoute, LoadSpec)> {
    let plain = |mode: MemoryMode, reference_count: u32, has_phases: bool| {
        (
            gen_core::MemoryBehaviorRoute {
                mode,
                reference_count,
                use_pid: false,
                has_phases,
                overlay: None,
            },
            spec.clone(),
        )
    };
    let mut routes = vec![
        plain(MemoryMode::TextToImage, 0, false),
        plain(MemoryMode::Other("hires".to_owned()), 0, true),
    ];
    // A control render carries a real ControlNet source, so its overlay identity is `control:1`
    // on both sides of the comparison rather than absent.
    let control_spec = spec.clone().with_control(WeightsSource::Dir(PathBuf::from(
        "/__weights_free__/control",
    )));
    let control_overlay = load_overlay_identity(&control_spec);
    routes.push((
        gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: control_overlay,
        },
        control_spec,
    ));
    if route.edit {
        routes.push(plain(MemoryMode::ImageToImage, 1, false));
        routes.push(plain(MemoryMode::Edit, 1, false));
        // The four `Other` modes are executed by the eagerly-assembled bespoke providers, which
        // refuse staged residency (`validate_bespoke_context`) and publish a
        // [`SdxlSurface::Bespoke`] contract that declares that rung `Missing`. Witnessing them
        // under `StagedResidency` would advertise a rung their executor rejects.
        if strategy != MemoryStrategy::StagedResidency {
            for mode in [
                "image_inpaint",
                "image_detail",
                "character_image",
                "control_image",
            ] {
                routes.push(plain(MemoryMode::Other(mode.to_owned()), 1, false));
            }
        }
    }
    routes
}

pub(crate) fn registered_valid_fixtures(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || !matches!(
            contract
                .capability(strategy)
                .map(|capability| &capability.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }
    let tier = MemoryNumericTier {
        precision: Precision::Bf16,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    let route = contract
        .calibration
        .as_ref()
        .and_then(|identity| route_from_fingerprint(&identity.fingerprint))
        .ok_or_else(|| {
            gen_core::Error::Unsupported(
                "sdxl: behavior fixtures need a contract that names an exact route".into(),
            )
        })?;
    advertised_behavior_routes(spec, route, strategy)
        .into_iter()
        .map(|(behavior_route, fixture_spec)| {
            let context = gen_core::standard_memory_behavior_context(
                contract,
                strategy,
                tier,
                behavior_route,
            )?;
            let mut fixture = gen_core::MemoryBehaviorFixture::new(context);
            fixture.load_spec = Some(fixture_spec);
            Ok(fixture)
        })
        .collect()
}

pub(crate) fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope>>> {
    if contract.asset_facts == MemoryAssetFacts::default() {
        validate_weights_free_context(spec, contract, context)?;
    } else {
        let seal = SdxlArtifactSeal::capture(spec)?;
        validate_context(contract, context, &seal)?;
    }
    Ok(Some(Box::new(request_scope(
        candle_gen::candle_core::Device::Cpu,
        contract,
        context,
    )?)))
}

pub(crate) const MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: crate::MODEL_ID,
        valid_fixtures: registered_valid_fixtures,
        begin_request: registered_begin_request,
    };

pub(crate) const fn memory_registration() -> gen_core::MemoryRegistration {
    gen_core::MemoryRegistration {
        provider_id: crate::MODEL_ID,
        contract: provider_contract_for_spec,
        safety_check: registered_safety_check,
    }
}

pub(crate) fn surface_specs() -> Vec<gen_core::MemoryContractSurfaceSpec> {
    gen_core::candle_memory_contract_surface_specs()
}

/// Weights-free contract for the route the spec names.
///
/// The five routes do not share memory semantics — `realvisxl_lightning` refuses the edit family
/// its `edit: false` row declares — so a pre-load price must be minted under the *requested*
/// route's identity. An unrouted spec (the catalog's common weights-free witness) resolves to the
/// base route explicitly rather than by falling off the end of a hardcoded index.
pub fn weights_free_contract_for_route(
    route: SdxlRoute,
    surface: SdxlSurface,
    spec: &LoadSpec,
) -> MemoryProviderContract {
    let tier = MemoryNumericTier {
        precision: Precision::Bf16,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    build_contract(
        spec,
        surface,
        tier,
        MemoryAssetFacts::default(),
        route_fingerprint(route),
    )
}

pub(crate) fn weights_free_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let route = match spec.resolved_route {
        Some(_) => route(spec)?,
        None => SDXL_ROUTES[0],
    };
    Ok(weights_free_contract_for_route(
        route,
        SdxlSurface::Registered,
        spec,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device, Tensor};

    fn write_tensor(path: &Path, dtype: DType) {
        let tensor = Tensor::zeros((1,), dtype, &Device::Cpu).unwrap();
        safetensors::serialize_to_file(vec![("x.weight".to_owned(), tensor)], None, path).unwrap();
    }

    fn write_packed_tensor(path: &Path) {
        let weight = Tensor::zeros((1,), DType::U32, &Device::Cpu).unwrap();
        let scales = Tensor::zeros((1,), DType::F16, &Device::Cpu).unwrap();
        let biases = Tensor::zeros((1,), DType::F16, &Device::Cpu).unwrap();
        safetensors::serialize_to_file(
            vec![
                ("x.weight".to_owned(), weight),
                ("x.scales".to_owned(), scales),
                ("x.biases".to_owned(), biases),
            ],
            None,
            path,
        )
        .unwrap();
    }

    fn dense_spec(temp: &tempfile::TempDir) -> (LoadSpec, PathBuf) {
        let route = SDXL_ROUTES[0];
        let root = temp
            .path()
            .join("SceneWorks__sdxl-base-mlx")
            .join(route.revision)
            .join("bf16");
        for component in ["unet", "text_encoder", "text_encoder_2"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
            std::fs::write(root.join(component).join("config.json"), b"{}").unwrap();
        }
        write_tensor(
            &root.join("unet/diffusion_pytorch_model.fp16.safetensors"),
            DType::F16,
        );
        write_tensor(
            &root.join("text_encoder/model.fp16.safetensors"),
            DType::F16,
        );
        write_tensor(
            &root.join("text_encoder_2/model.fp16.safetensors"),
            DType::F16,
        );
        let clip_l = temp
            .path()
            .join("models--openai--clip-vit-large-patch14/snapshots")
            .join(CLIP_L_REVISION)
            .join("tokenizer.json");
        let clip_g = temp
            .path()
            .join("models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots")
            .join(CLIP_BIGG_REVISION)
            .join("tokenizer.json");
        let vae = temp
            .path()
            .join("models--madebyollin--sdxl-vae-fp16-fix/snapshots")
            .join(VAE_REVISION)
            .join("diffusion_pytorch_model.safetensors");
        for path in [&clip_l, &clip_g, &vae] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        }
        std::fs::write(&clip_l, b"{}").unwrap();
        std::fs::write(&clip_g, b"{}").unwrap();
        write_tensor(&vae, DType::F16);
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_resolved_route("sdxl")
            .with_component("tokenizer_clip_l", WeightsSource::File(clip_l))
            .with_component("tokenizer_clip_bigg", WeightsSource::File(clip_g))
            .with_component("vae_fp16_fix", WeightsSource::File(vae));
        (spec, root)
    }

    #[test]
    fn route_table_is_exact_and_distinct() {
        assert_eq!(SDXL_ROUTES.len(), 5);
        let ids = SDXL_ROUTES
            .iter()
            .map(|route| route.id)
            .collect::<BTreeSet<_>>();
        let revisions = SDXL_ROUTES
            .iter()
            .map(|route| route.revision)
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), 5);
        assert_eq!(revisions.len(), 5);
        assert!(SDXL_ROUTES.iter().all(|route| route.revision.len() == 40
            && route.revision.chars().all(|c| c.is_ascii_hexdigit())));
        assert!(
            !SDXL_ROUTES
                .iter()
                .find(|route| route.lightning)
                .unwrap()
                .edit
        );
    }

    /// Weights-free witness for one route: no filesystem, exact route identity.
    fn routed_weights_free_contract(route: SdxlRoute) -> MemoryProviderContract {
        let spec = LoadSpec::new(WeightsSource::Dir("/__weights_free__".into()))
            .with_resolved_route(route.id);
        weights_free_contract(&spec).unwrap()
    }

    fn context_for(
        contract: &MemoryProviderContract,
        mode: MemoryMode,
        reference_count: u32,
    ) -> MemoryRunContext {
        gen_core::standard_memory_behavior_context(
            contract,
            MemoryStrategy::BoundedDecode,
            MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
            gen_core::MemoryBehaviorRoute {
                mode,
                reference_count,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .unwrap()
    }

    /// sc-20799 blocker 1. Resolution used to be `fingerprint.contains("sdxl-candle-realvisxl-")`,
    /// which the *lightning* fingerprint also satisfies — so the lightning route resolved to
    /// `realvisxl` (`edit: true`, declared first) and was admitted for the whole edit family its
    /// own row refuses. Each route must resolve to exactly itself.
    #[test]
    fn every_route_resolves_to_exactly_itself_from_its_contract() {
        for route in SDXL_ROUTES {
            let contract = routed_weights_free_contract(*route);
            let fingerprint = &contract.calibration.as_ref().unwrap().fingerprint;
            assert_eq!(
                route_from_fingerprint(fingerprint).map(|resolved| resolved.id),
                Some(route.id),
                "{} resolved to the wrong route from {fingerprint:?}",
                route.id
            );
        }
        let fingerprints = SDXL_ROUTES
            .iter()
            .map(|route| route_fingerprint(*route))
            .collect::<BTreeSet<_>>();
        assert_eq!(fingerprints.len(), SDXL_ROUTES.len());
        // Two route ids end in their own `vN`; the fingerprint must still carry exactly one
        // semantics version token.
        for fingerprint in &fingerprints {
            gen_core::validate_calibration_fingerprint(fingerprint)
                .unwrap_or_else(|error| panic!("{fingerprint}: {error}"));
        }
    }

    /// The behavioural consequence of the collision: the lightning route declares `edit: false`,
    /// so its own contract must refuse Edit / img2img / inpaint / detail / character, while an
    /// edit-capable route admits them.
    #[test]
    fn the_lightning_route_refuses_the_edit_family_its_row_declares_absent() {
        let lightning = SDXL_ROUTES
            .iter()
            .copied()
            .find(|route| route.lightning)
            .expect("the table ships a lightning route");
        assert!(!lightning.edit);
        let lightning_contract = routed_weights_free_contract(lightning);
        let base_contract = routed_weights_free_contract(SDXL_ROUTES[0]);
        assert!(SDXL_ROUTES[0].edit);

        for mode in [
            MemoryMode::Edit,
            MemoryMode::ImageToImage,
            MemoryMode::Other("image_inpaint".to_owned()),
            MemoryMode::Other("image_detail".to_owned()),
            MemoryMode::Other("character_image".to_owned()),
            MemoryMode::Other("control_image".to_owned()),
        ] {
            let refused = context_for(&lightning_contract, mode.clone(), 1);
            let error =
                validate_context_axes(&lightning_contract, &refused, refused.selection.tier, None)
                    .unwrap_err()
                    .to_string();
            assert!(
                error.contains("realvisxl_lightning"),
                "the refusal must name the lightning route, got {error}"
            );

            let admitted = context_for(&base_contract, mode.clone(), 1);
            validate_context_axes(&base_contract, &admitted, admitted.selection.tier, None)
                .unwrap_or_else(|error| panic!("the base route must admit {mode:?}: {error}"));
        }
    }

    /// sc-20799 blocker 2: **declared == validated == executed.**
    ///
    /// The contract publishes exactly one candidate edge and one candidate overlap,
    /// `CandleRequestScopeCore` re-validates exactly those, and the decoder must execute the same
    /// pair — the executed config used to be a second hardcoded `512/128` literal that neither the
    /// contract nor the scope named.
    #[test]
    fn the_executed_decode_geometry_is_the_declared_and_validated_one() {
        let contract = routed_weights_free_contract(SDXL_ROUTES[0]);
        let declared = contract
            .capability(MemoryStrategy::BoundedDecode)
            .expect("bounded decode is declared");
        assert_eq!(
            declared.parameters.decode_tile_edges,
            vec![DECODE_TILE_EDGE]
        );
        assert_eq!(declared.parameters.decode_overlaps, vec![DECODE_OVERLAP]);

        // What an unstated selection executes must be exactly what the contract advertises.
        let spatial = decode_tiling_config(None)
            .spatial
            .expect("the SDXL decode is spatially tiled");
        assert_eq!(
            spatial.tile_px,
            declared.parameters.decode_tile_edges[0] as i32
        );
        assert_eq!(
            spatial.overlap_px,
            declared.parameters.decode_overlaps[0] as i32
        );

        // An explicit selection executes the selected values, not the constants.
        let selected = decode_tiling_config(Some(gen_core::GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(256),
            decode_overlap: Some(32),
            ..Default::default()
        }))
        .spatial
        .expect("the SDXL decode is spatially tiled");
        assert_eq!(selected.tile_px, 256);
        assert_eq!(selected.overlap_px, 32);
    }

    /// sc-20799 blocker 3. Every route must be priceable pre-load under its **own** identity; the
    /// witness used to hardcode `SDXL_ROUTES[0]` and one literal fingerprint.
    #[test]
    fn weights_free_contracts_carry_each_routes_own_identity() {
        for route in SDXL_ROUTES {
            let contract = routed_weights_free_contract(*route);
            assert_eq!(
                contract.calibration.as_ref().unwrap().fingerprint,
                route_fingerprint(*route)
            );
            gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
        }
    }

    /// sc-20799 issue 8. The bespoke providers hard-refuse staged residency at admission, so their
    /// contract must not advertise the rung as Implemented.
    #[test]
    fn the_bespoke_surface_declares_the_rung_it_refuses_missing() {
        let spec = LoadSpec::new(WeightsSource::Dir("/__weights_free__".into()))
            .with_resolved_route(SDXL_ROUTES[0].id);
        let bespoke = weights_free_contract_for_route(SDXL_ROUTES[0], SdxlSurface::Bespoke, &spec);
        let registered =
            weights_free_contract_for_route(SDXL_ROUTES[0], SdxlSurface::Registered, &spec);
        assert_eq!(
            bespoke
                .capability(MemoryStrategy::StagedResidency)
                .map(|capability| &capability.support),
            Some(&MemoryStrategySupport::Missing)
        );
        assert_eq!(
            registered
                .capability(MemoryStrategy::StagedResidency)
                .map(|capability| &capability.support),
            Some(&MemoryStrategySupport::Implemented)
        );
        // The refusal the declaration now mirrors is still enforced at admission.
        let mut context = context_for(&bespoke, MemoryMode::Edit, 1);
        context.selection.strategy = MemoryStrategy::StagedResidency;
        assert!(validate_bespoke_context(&context).is_err());
    }

    /// sc-20799 issue 7. Conformance witnessed only plain text-to-image while
    /// `validate_context_axes` advertised the whole edit/control/hires family.
    #[test]
    fn behavior_fixtures_witness_every_advertised_surface() {
        let spec = LoadSpec::new(WeightsSource::Dir("/__weights_free__".into()))
            .with_resolved_route(SDXL_ROUTES[0].id);
        let contract = weights_free_contract(&spec).unwrap();
        let fixtures =
            registered_valid_fixtures(&spec, &contract, MemoryStrategy::BoundedDecode).unwrap();
        let modes = fixtures
            .iter()
            .map(|fixture| format!("{:?}", fixture.context.mode))
            .collect::<BTreeSet<_>>();
        for expected in [
            "TextToImage",
            "ImageToImage",
            "Edit",
            "Other(\"hires\")",
            "Other(\"image_inpaint\")",
            "Other(\"image_detail\")",
            "Other(\"character_image\")",
            "Other(\"control_image\")",
        ] {
            assert!(
                modes.iter().any(|mode| mode == expected),
                "no behavior fixture witnesses {expected}; got {modes:?}"
            );
        }
        // Every witness must pass the production admission seam it claims to exercise.
        for fixture in &fixtures {
            let fixture_spec = fixture.load_spec.as_ref().unwrap_or(&spec);
            registered_begin_request(fixture_spec, &contract, &fixture.context).unwrap_or_else(
                |error| {
                    panic!(
                        "fixture {:?} is not admissible: {error}",
                        fixture.context.mode
                    )
                },
            );
        }
        // Staged residency is the one rung the bespoke-only modes' executor refuses.
        let staged =
            registered_valid_fixtures(&spec, &contract, MemoryStrategy::StagedResidency).unwrap();
        assert!(staged.iter().all(
            |fixture| !matches!(&fixture.context.mode, MemoryMode::Other(mode) if mode != "hires")
        ));
    }

    /// sc-20799 minor: the weights-free admission branch pinned the overlay axis to `None`, which
    /// refused every overlay-bearing request the sealed path admits.
    #[test]
    fn the_weights_free_branch_admits_the_specs_own_overlay() {
        let spec = LoadSpec::new(WeightsSource::Dir("/__weights_free__".into()))
            .with_resolved_route(SDXL_ROUTES[0].id)
            .with_control(WeightsSource::Dir("/__weights_free__/control".into()));
        let overlay = load_overlay_identity(&spec).expect("a control spec carries an overlay");
        assert_eq!(overlay, "control:1");
        let contract = weights_free_contract(&spec).unwrap();
        let mut context = context_for(&contract, MemoryMode::TextToImage, 1);
        context.overlay = Some(overlay);
        validate_weights_free_context(&spec, &contract, &context).unwrap();
    }

    #[test]
    fn weights_free_contract_is_truthful() {
        for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
            let mut spec = LoadSpec::new(WeightsSource::Dir("/__weights_free__".into()));
            spec.quantize = quant;
            let contract = weights_free_contract(&spec).unwrap();
            gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .map(|capability| &capability.support),
                Some(&MemoryStrategySupport::Missing)
            );
        }
    }

    #[test]
    fn dense_receipt_seals_inventory_and_content() {
        let temp = tempfile::tempdir().unwrap();
        let (spec, root) = dense_spec(&temp);
        let seal = SdxlArtifactSeal::capture(&spec).unwrap();
        assert_eq!(seal.tier().quant, None);
        seal.ensure_unchanged().unwrap();

        std::fs::write(root.join("late-file.json"), b"{}").unwrap();
        assert!(seal.ensure_unchanged().is_err());
    }

    #[test]
    fn dense_receipt_rejects_config_and_tensor_forgery() {
        let temp = tempfile::tempdir().unwrap();
        let (spec, root) = dense_spec(&temp);
        std::fs::write(
            root.join("unet/config.json"),
            br#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        assert!(SdxlArtifactSeal::capture(&spec)
            .unwrap_err()
            .to_string()
            .contains("crossed packed configuration"));

        std::fs::write(root.join("unet/config.json"), b"{}").unwrap();
        write_tensor(
            &root.join("unet/diffusion_pytorch_model.fp16.safetensors"),
            DType::U32,
        );
        assert!(SdxlArtifactSeal::capture(&spec)
            .unwrap_err()
            .to_string()
            .contains("floating tensors"));
    }

    #[test]
    fn q4_q8_receipts_require_real_packed_triples_and_exact_tier_identity() {
        for (bits, quant) in [(4, Quant::Q4), (8, Quant::Q8)] {
            let temp = tempfile::tempdir().unwrap();
            let (mut spec, dense_root) = dense_spec(&temp);
            let root = dense_root.parent().unwrap().join(format!("q{bits}"));
            std::fs::rename(&dense_root, &root).unwrap();
            for component in ["unet", "text_encoder", "text_encoder_2"] {
                std::fs::write(
                    root.join(component).join("config.json"),
                    format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
                )
                .unwrap();
            }
            for (dense, packed) in [
                (
                    "unet/diffusion_pytorch_model.fp16.safetensors",
                    "unet/diffusion_pytorch_model.safetensors",
                ),
                (
                    "text_encoder/model.fp16.safetensors",
                    "text_encoder/model.safetensors",
                ),
                (
                    "text_encoder_2/model.fp16.safetensors",
                    "text_encoder_2/model.safetensors",
                ),
            ] {
                std::fs::remove_file(root.join(dense)).unwrap();
                write_packed_tensor(&root.join(packed));
            }
            spec.weights = WeightsSource::Dir(root.clone());
            spec.quantize = Some(quant);
            let seal = SdxlArtifactSeal::capture(&spec).unwrap();
            assert_eq!(seal.tier().quant, Some(quant));

            write_tensor(
                &root.join("unet/diffusion_pytorch_model.safetensors"),
                DType::U32,
            );
            assert!(SdxlArtifactSeal::capture(&spec)
                .unwrap_err()
                .to_string()
                .contains("packed tensor triples"));
        }
    }

    #[test]
    fn receipt_rejects_crossed_route_and_mutable_component_source() {
        let temp = tempfile::tempdir().unwrap();
        let (mut spec, _) = dense_spec(&temp);
        spec.resolved_route = Some("realvisxl".to_owned());
        assert!(SdxlArtifactSeal::capture(&spec)
            .unwrap_err()
            .to_string()
            .contains("must resolve from exact"));

        let (mut spec, _) = dense_spec(&temp);
        let mutable = temp.path().join("mutable-tokenizer.json");
        std::fs::write(&mutable, b"{}").unwrap();
        spec.components
            .insert("tokenizer_clip_l".to_owned(), WeightsSource::File(mutable));
        assert!(SdxlArtifactSeal::capture(&spec)
            .unwrap_err()
            .to_string()
            .contains("component tokenizer_clip_l"));
    }

    #[test]
    fn request_scope_binds_geometry_controls_and_cleanup() {
        use gen_core::MemoryRequestScope as _;

        let temp = tempfile::tempdir().unwrap();
        let (spec, _) = dense_spec(&temp);
        let seal = SdxlArtifactSeal::capture(&spec).unwrap();
        let contract = seal.contract();
        let context = gen_core::standard_memory_behavior_context(
            contract,
            MemoryStrategy::BoundedAttention,
            seal.tier(),
            gen_core::MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .unwrap();
        validate_context(contract, &context, &seal).unwrap();
        let staged = gen_core::standard_memory_behavior_context(
            contract,
            MemoryStrategy::StagedResidency,
            seal.tier(),
            gen_core::MemoryBehaviorRoute {
                mode: MemoryMode::Edit,
                reference_count: 1,
                use_pid: false,
                has_phases: false,
                overlay: seal.overlay_identity().map(str::to_owned),
            },
        )
        .unwrap();
        assert!(validate_bespoke_context(&staged).is_err());
        let mut scope = request_scope(Device::Cpu, contract, &context).unwrap();
        let mut request = gen_core::GenerationRequest {
            prompt: "fixture".into(),
            width: context.geometry.width,
            height: context.geometry.height,
            count: context.geometry.batch,
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        // Scratch-bounding rungs compose by cost order, while staged residency is deliberately an
        // independent, cache-evicting choice. BoundedAttention therefore enables decode+attention
        // but must preserve the warm resident cache.
        assert!(!request.memory.unwrap().stage_residency);
        assert!(request.memory.unwrap().tile_vae_decode);
        assert!(request.memory.unwrap().chunk_attention);
        scope
            .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, context.geometry)
            .unwrap();
        scope.configure_attention(ATTENTION_CHUNK_SIZE).unwrap();
        scope.finish(gen_core::MemoryRunOutcome::Canceled).unwrap();
        assert!(scope.configure_request(&mut request).is_err());

        let mut crossed = context.clone();
        crossed.mode = MemoryMode::Edit;
        crossed.geometry.reference_count = 0;
        crossed.has_reference = false;
        assert!(validate_context(contract, &crossed, &seal).is_err());
    }
}
