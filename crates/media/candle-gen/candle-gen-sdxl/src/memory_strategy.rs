//! Exact Candle/CUDA SDXL-family memory contract (SC-20793).
//!
//! The five catalog routes share code, not identity.  An adopting load must name one immutable
//! route, resolve from its immutable snapshot, prove its packed/dense tensor form, and retain a
//! complete content seal for every lazily opened base, component, control, IP, PiD, and ordered
//! adapter source.  Legacy loads without `resolved_route` remain resident-only.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use candle_gen::gen_core::{
    self, AdapterResidencyMode, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryCalibrationIdentity, MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier,
    MemoryParameterRanges, MemoryPhase, MemoryProviderContract, MemoryResidentComponent,
    MemoryRunContext, MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability,
    MemoryStrategySupport, MemoryWindowMaterialization, Precision, Quant, WeightsSource,
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

#[derive(Clone, Debug)]
struct SealedFile {
    pin: gen_core::PinnedWeightsFile,
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
                receipt.update((path.as_os_str().len() as u64).to_le_bytes());
                receipt.update(path.as_os_str().as_encoded_bytes());
                receipt.update(pin.content_sha256());
                files.push(SealedFile { pin });
            }
            roots.push((absolute, inventory));
        }
        let overlay_identity = load_overlay_identity(spec);
        if let Some(identity) = &overlay_identity {
            receipt.update(identity.as_bytes());
        }
        let receipt = format!("{:x}", receipt.finalize());
        let (facts, components) = asset_facts(spec, &root, surface, tier)?;
        let contract = build_contract(
            spec,
            surface,
            tier,
            facts,
            components,
            route_fingerprint(route),
        );
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
            file.pin.verify_unchanged()?;
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

/// Resident bytes of the tensors in one `.safetensors` file whose name `keep` admits: floats at
/// `float_width`, integer codes at their stored width. A tensor the loader never reads is not
/// charged, which is what `keep` expresses.
fn filtered_tensor_bytes(
    path: &Path,
    float_width: u64,
    keep: &dyn Fn(&str) -> bool,
) -> gen_core::Result<u64> {
    gen_core::weightsmeta::safetensors_path_tensor_headers(path)?
        .iter()
        .filter(|header| keep(&header.name))
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
    filtered_source_tensor_bytes(path, float_width, &|_| true)
}

fn filtered_source_tensor_bytes(
    path: &Path,
    float_width: u64,
    keep: &dyn Fn(&str) -> bool,
) -> gen_core::Result<u64> {
    let mut files = Vec::new();
    collect_files(path, &mut files)?;
    files
        .into_iter()
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))
        .try_fold(0_u64, |sum, path| {
            sum.checked_add(filtered_tensor_bytes(&path, float_width, keep)?)
                .ok_or_else(|| gen_core::Error::Msg("sdxl: component byte sum overflow".into()))
        })
}

/// Every SDXL-family component is materialized at `DType::F16` — the registry generator pins
/// `dtype: DType::F16` and each bespoke provider its own `const DTYPE: DType = DType::F16` — so one
/// float element of any of them costs two bytes on device. Same width as [`ACTIVATION_DTYPE`].
const FLOAT_WIDTH: u64 = 2;

/// The `LoadSpec::components` id under which the IP-Adapter route stages its CLIP ViT-H/14 image
/// encoder. `validate_ip_spec` pins it to `IpAdapterSdxlPaths::image_encoder`, which
/// `ip_provider::load` materializes at [`FLOAT_WIDTH`] beside the IP bundle.
const IP_IMAGE_ENCODER_COMPONENT: &str = "sdxl_ip_image_encoder";

/// The decode half of the diffusers `AutoencoderKL` checkpoint — the only tensors
/// [`crate::vae_decoder::SdxlVaeDecoder::new`] reads (`post_quant_conv` and everything under
/// `decoder.`, which includes `decoder.conv_norm_out` / `decoder.conv_out`). `encoder.*` and
/// `quant_conv` are never opened by that loader.
const VAE_DECODER_PREFIXES: &[&str] = &["decoder.", "post_quant_conv."];

fn is_vae_decoder_tensor(name: &str) -> bool {
    VAE_DECODER_PREFIXES
        .iter()
        .any(|prefix| name.starts_with(prefix))
}

/// Resident bytes of the `sdxl-vae-fp16-fix` checkpoint on `surface`, at the f16 width both of its
/// loaders materialize it at.
///
/// * [`SdxlSurface::Registered`] builds only [`crate::vae_decoder::SdxlVaeDecoder`]
///   (`loaders::load_sdxl_vae`, through the mmap-backed `from_file`), so only the decoder namespace
///   is resident: charging `encoder.*` + `quant_conv` billed ~half the checkpoint for tensors that
///   loader never opens (epic SC-22657, E1). A checkpoint with no decoder tensors is refused rather
///   than priced at zero — the loader would fail on it too.
/// * [`SdxlSurface::Bespoke`] prices the whole checkpoint: the edit and detail providers ALSO build
///   `VaeMomentsEncoder` from the same file (`loaders::load_sdxl_vae_encoder`, `encoder.*` +
///   `quant_conv` via mmap), so both halves are resident there. The IP provider shares that surface
///   without an encoder and is therefore over-declared by the encoder half — the surface carries no
///   per-provider discriminator, and of the two available errors only that one is conservative
///   (E3).
fn vae_tensor_bytes(source: &Path, surface: SdxlSurface) -> gen_core::Result<u64> {
    match surface {
        SdxlSurface::Bespoke => source_tensor_bytes(source, FLOAT_WIDTH),
        SdxlSurface::Registered => {
            let decoder =
                filtered_source_tensor_bytes(source, FLOAT_WIDTH, &is_vae_decoder_tensor)?;
            if decoder == 0 {
                return Err(gen_core::Error::Unsupported(format!(
                    "sdxl: VAE {} has no `decoder.` / `post_quant_conv` tensors for the decoder to load",
                    source.display()
                )));
            }
            Ok(decoder)
        }
    }
}

/// Width of one float element of the PiD student and its Gemma caption encoder once resident:
/// `PidEngine::load` reads both through `Weights::from_file(s)` at
/// [`candle_gen_pid::engine::LOAD_DTYPE`] (f32), whatever the checkpoint ships, so a bf16
/// Gemma-2-2B costs ~10.4 GB resident, not ~5.2 GB (epic SC-22657, E3). Read from the loader's own
/// constant so the two cannot drift.
fn pid_float_width() -> u64 {
    candle_gen_pid::engine::LOAD_DTYPE.size_in_bytes() as u64
}

/// The Gemma source `PidEngine::load` opens: the merged single file when the snapshot ships it,
/// else the whole shard directory — never both, so a directory carrying both is not summed twice.
fn pid_gemma_source(gemma: &Path) -> PathBuf {
    let merged = gemma.join(candle_gen_pid::engine::GEMMA_MERGED_FILE);
    if merged.is_file() {
        merged
    } else {
        gemma.to_path_buf()
    }
}

/// Resident bytes of one PiD checkpoint source (a file or a shard directory), packed-aware.
///
/// A dense float tensor lands at [`pid_float_width`]. A SANA-published packed Gemma tier
/// (`gemma2::validate_packed_tier`: every attention/MLP projection as an MLX group-64
/// `.weight` U32 / `.scales` / `.biases` triple) is repacked by `linear_from_weights` into one
/// resident GGML `QTensor` (`Q4_1` / `Q8_0`) — priced through
/// [`candle_gen::quant::mlx_packed_qtensor_resident_bytes`], with the sidecars as transient pack
/// inputs — rather than at the on-disk byte count of its three source tensors, which is neither the
/// resident size nor an upper bound of it.
fn pid_source_bytes(source: &Path) -> gen_core::Result<u64> {
    let mut files = Vec::new();
    collect_files(source, &mut files)?;
    files
        .into_iter()
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))
        .try_fold(0_u64, |sum, path| {
            sum.checked_add(pid_file_bytes(&path)?)
                .ok_or_else(|| gen_core::Error::Msg("sdxl: PiD byte sum overflow".into()))
        })
}

fn pid_file_bytes(path: &Path) -> gen_core::Result<u64> {
    let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
    let by_name = headers
        .iter()
        .map(|header| (header.name.as_str(), header))
        .collect::<std::collections::HashMap<_, _>>();
    let width = pid_float_width();
    headers.iter().try_fold(0_u64, |sum, header| {
        if header.name.ends_with(".scales") || header.name.ends_with(".biases") {
            // MLX affine sidecars: transient inputs to the repack, not separately resident.
            return Ok(sum);
        }
        let packed = header.name.strip_suffix(".weight").and_then(|base| {
            Some((
                *by_name.get(format!("{base}.scales").as_str())?,
                *by_name.get(format!("{base}.biases").as_str())?,
            ))
        });
        let bytes = match packed {
            Some((scales, biases)) => candle_gen::quant::mlx_packed_qtensor_resident_bytes(
                header,
                scales,
                biases,
                candle_gen::quant::MLX_GROUP_SIZE,
            )?,
            None if header.is_float() => header.materialized_bytes(width)?,
            None => header.data_bytes,
        };
        sum.checked_add(bytes)
            .ok_or_else(|| gen_core::Error::Msg("sdxl: PiD tensor byte sum overflow".into()))
    })
}

/// Stable provider-local ids of the typed auxiliary components this contract can declare.
const CONTROL_COMPONENT_ID: &str = "sdxl_control";
const IP_ADAPTER_COMPONENT_ID: &str = "sdxl_ip_adapter";
const PID_STUDENT_COMPONENT_ID: &str = "sdxl_pid_student";
const PID_CAPTION_ENCODER_COMPONENT_ID: &str = "sdxl_pid_caption_encoder";
const ADAPTER_STACK_COMPONENT_ID: &str = "sdxl_adapter_stack";

/// Record one auxiliary network as a typed resident component, skipping a zero: the shared
/// validator refuses a declared component with zero bytes, and a source that priced to nothing is
/// not evidence of residency (the loader fails on it before anything is resident).
fn push_overlay(
    into: &mut Vec<MemoryResidentComponent>,
    id: String,
    kind: MemoryComponentKind,
    resident_bytes: u64,
) {
    if resident_bytes == 0 {
        return;
    }
    into.push(MemoryResidentComponent {
        id,
        kind,
        resident_bytes,
        // No published rung bounds an SDXL overlay: block windowing is `Missing` on every surface
        // and staged residency releases base phases, never an auxiliary.
        bounded_by: None,
        residency: MemoryComponentResidency::WholeRender,
    });
}

/// The priced base fields plus one typed [`MemoryResidentComponent`] per auxiliary network the
/// spec stages. `overlay_bytes` is the sum of those components by construction, which is the
/// agreement `MemoryProviderContract::conformance_errors` demands of a contract that declares both
/// `AssetBytes` and `OverlayBytes` — the registered `sdxl` id seals this surface with `spec.control`
/// set (`lib.rs` routes control renders through it), so an untyped non-zero overlay there was a
/// non-conformant contract that any selector reading `conformance_errors` would refuse.
fn asset_facts(
    spec: &LoadSpec,
    root: &Path,
    surface: SdxlSurface,
    tier: MemoryNumericTier,
) -> gen_core::Result<(MemoryAssetFacts, Vec<MemoryResidentComponent>)> {
    let width = FLOAT_WIDTH;
    let conditioning = source_tensor_bytes(&root.join("text_encoder"), width)?
        .saturating_add(source_tensor_bytes(&root.join("text_encoder_2"), width)?);
    let transformer = source_tensor_bytes(&root.join("unet"), width)?;
    let fallback_decoder = root.join("vae");
    let decoder_source = spec
        .components
        .get("vae_fp16_fix")
        .map(source_path)
        .unwrap_or(&fallback_decoder);
    // The fp16-fix VAE is loaded through `SdxlVaeDecoder::from_file(.., self.dtype, ..)` — f16,
    // which is the entire point of that component: it is the checkpoint whose decode stays
    // numerically stable at f16. Pricing it at four bytes charged twice the decoder this seal can
    // admit. (The fused A1111 route, whose checkpoint VAE genuinely loads f32, never reaches here:
    // `SdxlArtifactSeal::capture_for` requires a snapshot directory.) Epic SC-22657, E1.
    let decoder = vae_tensor_bytes(decoder_source, surface)?;

    let mut components = Vec::new();
    for (index, control) in spec
        .control
        .iter()
        .chain(spec.extra_controls.iter())
        .enumerate()
    {
        // MultiControlNet holds several distinct branches at once; the id, not the kind, tells
        // them apart.
        let id = if index == 0 {
            CONTROL_COMPONENT_ID.to_owned()
        } else {
            format!("{CONTROL_COMPONENT_ID}_{}", index + 1)
        };
        push_overlay(
            &mut components,
            id,
            MemoryComponentKind::ControlBranch,
            source_tensor_bytes(source_path(control), width)?,
        );
    }
    if let Some(ip_adapter) = &spec.ip_adapter {
        push_overlay(
            &mut components,
            IP_ADAPTER_COMPONENT_ID.to_owned(),
            MemoryComponentKind::IpAdapter,
            source_tensor_bytes(source_path(ip_adapter), width)?,
        );
    }
    // The CLIP ViT-H/14 image encoder the IP-Adapter route loads beside its bundle
    // (`ip_provider::load`, `Weights::from_file(.., DTYPE)`), staged as the
    // `sdxl_ip_image_encoder` component and pinned by `validate_ip_spec`: ~1.3 GB of resident
    // auxiliary weights that no field charged (epic SC-22657, E1).
    if let Some(encoder) = spec.components.get(IP_IMAGE_ENCODER_COMPONENT) {
        push_overlay(
            &mut components,
            IP_IMAGE_ENCODER_COMPONENT.to_owned(),
            MemoryComponentKind::IpAdapter,
            source_tensor_bytes(source_path(encoder), width)?,
        );
    }
    // The PiD super-resolving decoder (epic 7840 / sc-7853). `LoadSpec::pid` makes the component
    // build load `PidEngine::from_spec` once alongside the base model — unconditionally, not per
    // request — and PiD runs on the Resident rung (`validate_context` refuses it only for the
    // *optimized* rungs), so both of its files are resident for the whole render while no field
    // charged either of them (epic SC-22657, E1).
    //
    // It is an optional add-on network standing beside the base model, which
    // `gen_core::MemoryAssetFacts` says never belongs in the three base fields: one typed
    // auxiliary component per checkpoint, summed into `overlay_bytes`. `AdapterStack` is the
    // closest existing kind for an auxiliary network installed beside the base model's own
    // networks (the Kolors PiD declaration's precedent); what the contract arithmetic consumes is
    // `MemoryComponentKind::is_auxiliary()`, which it satisfies.
    if let Some(pid) = spec.pid.as_ref() {
        push_overlay(
            &mut components,
            PID_STUDENT_COMPONENT_ID.to_owned(),
            MemoryComponentKind::AdapterStack,
            pid_source_bytes(source_path(&pid.checkpoint))?,
        );
        push_overlay(
            &mut components,
            PID_CAPTION_ENCODER_COMPONENT_ID.to_owned(),
            MemoryComponentKind::AdapterStack,
            pid_source_bytes(&pid_gemma_source(source_path(&pid.gemma)))?,
        );
    }
    let adapter_mode = if tier.quant.is_some() {
        AdapterResidencyMode::Additive
    } else {
        AdapterResidencyMode::Folded
    };
    push_overlay(
        &mut components,
        ADAPTER_STACK_COMPONENT_ID.to_owned(),
        MemoryComponentKind::AdapterStack,
        gen_core::adapter_stack_resident_bytes(&spec.adapters, adapter_mode).ok_or_else(|| {
            gen_core::Error::Unsupported(
                "sdxl: every additive packed adapter must have an exact non-zero size".into(),
            )
        })?,
    );
    let overlay = components.iter().fold(0_u64, |sum, component| {
        sum.saturating_add(component.resident_bytes)
    });
    Ok((
        MemoryAssetFacts {
            base_bytes: conditioning
                .saturating_add(transformer)
                .saturating_add(decoder),
            conditioning_bytes: conditioning,
            transformer_bytes: transformer,
            decoder_bytes: decoder,
            overlay_bytes: overlay,
        },
        components,
    ))
}

/// Activation dtype every SDXL-family Candle route computes in. `lib.rs` pins `DType::F16` on the
/// loaded generator, so this is the provider's real activation width, not a memory-model literal.
const ACTIVATION_DTYPE: candle_gen::candle_core::DType = candle_gen::candle_core::DType::F16;

/// Architecture axes for the vendored SDXL UNet + `sdxl-vae-fp16-fix` decoder (epic SC-22657, E2).
///
/// The axes come off the two Rust configs the loader actually builds from —
/// [`crate::unet::sdxl_unet_config`] and `pipeline::sdxl_vae_config` — not from any snapshot
/// `config.json`, because the vendored stack ignores the on-disk config entirely.
///
/// Four of the eight axes are structurally absent for a UNet denoiser and are therefore declared
/// absent rather than zero (E2):
///
/// * `attention_heads` — the UNet's head count is per stage (5/10/20 across `320/640/1280`); there
///   is no single uniform head count to declare.
/// * `transformer_blocks` — a UNet is a down/mid/up convolutional trunk, not a uniform stack of
///   transformer blocks.
/// * `patch_size` — the UNet consumes the latent grid directly; nothing is patchified.
/// * `vae_temporal_scale` — SDXL ships the image `AutoencoderKL`, which has no temporal axis.
///
/// `head_dim` *is* uniform: `out_channels / heads` is 64 in all three stages. It is published only
/// when every stage agrees, so a geometry that ever stopped being uniform declines the axis rather
/// than claiming a head width it does not have.
///
/// `activation_dtype` is a parameter because the same UNet geometry is loaded by sibling providers
/// at their own pinned compute width (InstantID's fp16, Kolors' f32).
pub fn sdxl_unet_family_architecture_facts(
    activation_dtype: candle_gen::candle_core::DType,
) -> gen_core::MemoryArchitectureFacts {
    use candle_gen::architecture_facts as af;

    let unet = crate::unet::sdxl_unet_config();
    // In diffusers' SDXL config `attention_head_dim` is the per-block HEAD COUNT, so the per-head
    // width is the stage quotient. Publish it only if every stage produces the same quotient.
    let mut stage_widths = unet
        .blocks
        .iter()
        .map(|block| match block.attention_head_dim {
            heads if heads != 0 && block.out_channels % heads == 0 => {
                af::declared(block.out_channels / heads)
            }
            _ => None,
        });
    let head_dim = match stage_widths.next() {
        Some(first) if first.is_some() && stage_widths.all(|width| width == first) => first,
        _ => None,
    };
    let vae = crate::pipeline::sdxl_vae_config();
    gen_core::MemoryArchitectureFacts {
        // Per-stage head counts (5/10/20): no uniform head count exists to declare.
        attention_heads: None,
        head_dim,
        // A UNet trunk is not a uniform transformer-block stack.
        transformer_blocks: None,
        // The UNet consumes the latent directly; there is no patchification.
        patch_size: None,
        latent_channels: af::declared(vae.latent_channels),
        // Each `block_out_channels` stage after the first halves both spatial axes: 4 stages => x8.
        vae_spatial_scale: vae
            .block_out_channels
            .len()
            .checked_sub(1)
            .filter(|shift| *shift <= 5)
            .map(|shift| 1_u32 << shift),
        // SDXL ships the image `AutoencoderKL`: there is no temporal axis to declare.
        vae_temporal_scale: None,
        activation_dtype_width: af::dtype_width(activation_dtype),
    }
}

/// Snapshot-scoped facts for the SDXL routes: the weights-free contract surface (the registry's
/// sentinel path, or a single-file import) resolves no snapshot, so no axis is knowable there.
fn architecture_facts(spec: &LoadSpec) -> gen_core::MemoryArchitectureFacts {
    if candle_gen::architecture_facts::snapshot_root(spec).is_none() {
        return gen_core::MemoryArchitectureFacts::default();
    }
    sdxl_unet_family_architecture_facts(ACTIVATION_DTYPE)
}

fn build_contract(
    spec: &LoadSpec,
    surface: SdxlSurface,
    _tier: MemoryNumericTier,
    asset_facts: MemoryAssetFacts,
    resident_components: Vec<MemoryResidentComponent>,
    fingerprint: String,
) -> MemoryProviderContract {
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    MemoryProviderContract {
        architecture_facts: architecture_facts(spec),
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
        // The component variant of the phase envelope: every auxiliary network `asset_facts`
        // sums into `overlay_bytes` is also declared as a typed resident component, which is what
        // lets a non-zero overlay conform. A weights-free surface declares none.
        formula: MemoryFormulaKind::ComponentPhaseEnvelope {
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
            resident_components,
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
        spec.components.get(IP_IMAGE_ENCODER_COMPONENT),
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
        Vec::new(),
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

    /// The fp16-fix VAE fixture: one f16 decoder element (`decoder.conv_out.weight`) beside eight
    /// f16 encoder elements (`encoder.conv_in.weight`) that only the bespoke surface's
    /// `VaeMomentsEncoder` opens.
    fn write_vae(path: &Path) {
        safetensors::serialize_to_file(
            vec![
                (
                    "decoder.conv_out.weight".to_owned(),
                    Tensor::zeros((1,), DType::F16, &Device::Cpu).unwrap(),
                ),
                (
                    "encoder.conv_in.weight".to_owned(),
                    Tensor::zeros((8,), DType::F16, &Device::Cpu).unwrap(),
                ),
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
        write_vae(&vae);
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_resolved_route("sdxl")
            .with_component("tokenizer_clip_l", WeightsSource::File(clip_l))
            .with_component("tokenizer_clip_bigg", WeightsSource::File(clip_g))
            .with_component("vae_fp16_fix", WeightsSource::File(vae));
        (spec, root)
    }

    /// AC (epic SC-22657, E1): every SDXL asset field prices what the loader materializes.
    ///
    /// Four defects are pinned:
    ///
    /// 1. **The fp16-fix VAE is loaded f16, decoder half only.** `SdxlVaeDecoder::from_file(..,
    ///    self.dtype, ..)` with `dtype: DType::F16` reads `decoder.*` + `post_quant_conv` and
    ///    nothing else; pricing the whole file at four bytes charged ~4x the resident decoder.
    ///    *Mutations that red this:* `source_tensor_bytes(decoder_source, 4)`, and pricing the
    ///    registered surface through `source_tensor_bytes` without the decoder filter.
    /// 2. **The bespoke surface also holds the encoder.** The edit and detail providers build
    ///    `VaeMomentsEncoder` from the same checkpoint. *Mutation that reds this:* applying the
    ///    decoder filter on `SdxlSurface::Bespoke` too.
    /// 3. **The IP-Adapter CLIP ViT-H image encoder is resident, and typed.** `ip_provider::load`
    ///    materializes the `sdxl_ip_image_encoder` component beside the IP bundle; both are
    ///    declared as `IpAdapter` components whose sum is the overlay.
    ///    *Mutation that reds this:* dropping the `IP_IMAGE_ENCODER_COMPONENT` `push_overlay`.
    /// 4. **PiD is resident when staged, at the width its engine loads, as an auxiliary.**
    ///    `PidEngine::load` materializes the student and its Gemma encoder through
    ///    `Weights::from_file(s)` at [`candle_gen_pid::engine::LOAD_DTYPE`] (f32) — four bytes per
    ///    f16 element on disk, not two — and PiD is an optional add-on network, so both land in
    ///    `overlay_bytes` as typed components and in none of the three base fields.
    ///    *Mutations that red this:* pricing PiD at `FLOAT_WIDTH`; routing the student into
    ///    `decoder_bytes` again.
    #[test]
    fn asset_facts_price_the_f16_decoder_and_every_staged_auxiliary() {
        let temp = tempfile::tempdir().unwrap();
        let (spec, root) = dense_spec(&temp);
        let base = provider_contract_for_spec(&spec).unwrap().asset_facts;
        // The fixture's `vae_fp16_fix` is one f16 decoder element beside eight f16 encoder
        // elements: two resident bytes on the registered surface, which builds only the decoder.
        assert_eq!(
            base.decoder_bytes, 2,
            "the registered surface materializes the f16 decoder namespace only"
        );
        assert_eq!(
            base.base_bytes,
            base.conditioning_bytes + base.transformer_bytes + base.decoder_bytes
        );
        assert_eq!(base.overlay_bytes, 0, "a plain route stages no auxiliary");
        // The bespoke surface (edit / detail) also builds `VaeMomentsEncoder` from that file.
        let bespoke = SdxlArtifactSeal::capture_for(&spec, SdxlSurface::Bespoke).unwrap();
        assert_eq!(
            bespoke.contract().asset_facts.decoder_bytes,
            2 + 8 * 2,
            "the bespoke surface holds both VAE halves at f16"
        );
        assert!(bespoke.contract().conformance_errors().is_empty());

        // The IP-Adapter route stages its bundle AND its CLIP ViT-H image encoder.
        let ip_bundle = root.join("ip-adapter-plus.safetensors");
        let ip_encoder = root.join("ip-image-encoder.safetensors");
        write_tensor(&ip_bundle, DType::F16);
        let encoder_tensor = Tensor::zeros((16, 16), DType::F16, &Device::Cpu).unwrap();
        safetensors::serialize_to_file(
            vec![("x.weight".to_owned(), encoder_tensor)],
            None,
            &ip_encoder,
        )
        .unwrap();
        let mut ip_spec = spec.clone();
        ip_spec.ip_adapter = Some(WeightsSource::File(ip_bundle));
        ip_spec = ip_spec.with_component(
            IP_IMAGE_ENCODER_COMPONENT,
            WeightsSource::File(ip_encoder.clone()),
        );
        let ip_contract = provider_contract_for_spec(&ip_spec).unwrap();
        assert_eq!(
            ip_contract.asset_facts.overlay_bytes,
            2 + 16 * 16 * 2,
            "the IP bundle and the CLIP ViT-H image encoder it loads beside it are both resident"
        );
        assert_eq!(
            component_table(&ip_contract),
            vec![
                (
                    IP_ADAPTER_COMPONENT_ID.to_owned(),
                    MemoryComponentKind::IpAdapter,
                    2
                ),
                (
                    IP_IMAGE_ENCODER_COMPONENT.to_owned(),
                    MemoryComponentKind::IpAdapter,
                    16 * 16 * 2
                ),
            ]
        );
        assert!(ip_contract.conformance_errors().is_empty());

        // A PiD-bearing spec stages the student checkpoint and the Gemma caption encoder.
        let pid_checkpoint = root.join("pid-student.safetensors");
        let gemma = root.join("gemma");
        std::fs::create_dir_all(&gemma).unwrap();
        let student = Tensor::zeros((4, 4), DType::F16, &Device::Cpu).unwrap();
        safetensors::serialize_to_file(
            vec![("x.weight".to_owned(), student)],
            None,
            &pid_checkpoint,
        )
        .unwrap();
        let gemma_tensor = Tensor::zeros((2, 4), DType::F16, &Device::Cpu).unwrap();
        safetensors::serialize_to_file(
            vec![("x.weight".to_owned(), gemma_tensor)],
            None,
            &gemma.join("model.safetensors"),
        )
        .unwrap();
        let mut pid_spec = spec.clone();
        pid_spec.pid = Some(gen_core::PidWeights {
            checkpoint: WeightsSource::File(pid_checkpoint),
            gemma: WeightsSource::Dir(gemma),
        });
        let pid_contract = provider_contract_for_spec(&pid_spec).unwrap();
        let pid_facts = pid_contract.asset_facts;
        // Both files are read at the engine's f32 load dtype: four bytes per f16 element on disk.
        assert_eq!(
            candle_gen_pid::engine::LOAD_DTYPE.size_in_bytes(),
            4,
            "PiD's loader materializes f32; the pricing below assumes that width"
        );
        assert_eq!(
            component_table(&pid_contract),
            vec![
                (
                    PID_STUDENT_COMPONENT_ID.to_owned(),
                    MemoryComponentKind::AdapterStack,
                    4 * 4 * 4
                ),
                (
                    PID_CAPTION_ENCODER_COMPONENT_ID.to_owned(),
                    MemoryComponentKind::AdapterStack,
                    2 * 4 * 4
                ),
            ],
            "the student and its Gemma caption encoder are typed auxiliaries at the f32 load width"
        );
        assert_eq!(pid_facts.overlay_bytes, 4 * 4 * 4 + 2 * 4 * 4);
        // An optional add-on network never joins the three base fields.
        assert_eq!(pid_facts.decoder_bytes, base.decoder_bytes);
        assert_eq!(pid_facts.conditioning_bytes, base.conditioning_bytes);
        assert_eq!(pid_facts.transformer_bytes, base.transformer_bytes);
        assert_eq!(pid_facts.base_bytes, base.base_bytes);
        assert!(
            pid_contract.conformance_errors().is_empty(),
            "{:?}",
            pid_contract.conformance_errors()
        );
    }

    /// `(id, kind, resident_bytes)` of every declared resident component, in declaration order.
    fn component_table(
        contract: &MemoryProviderContract,
    ) -> Vec<(String, MemoryComponentKind, u64)> {
        contract
            .resident_components()
            .iter()
            .map(|component| {
                assert_eq!(component.bounded_by, None);
                assert_eq!(component.residency, MemoryComponentResidency::WholeRender);
                (
                    component.id.clone(),
                    component.kind,
                    component.resident_bytes,
                )
            })
            .collect()
    }

    /// AC (epic SC-22657, E3): the Gemma caption encoder is priced from the file `PidEngine::load`
    /// opens — the merged `gemma-2-2b-it.safetensors` when present, else every shard — and a
    /// packed (SANA-published) tier is priced as the GGML `QTensor` `linear_from_weights` repacks
    /// each MLX affine triple into, not as the on-disk bytes of its three source tensors.
    ///
    /// The fixture's `[64, 128]` Q4 projection packs to `64 * 16` U32 codes + two `[64, 2]` f16
    /// sidecars (4_608 B on disk) and lands as `Q4_1`: `64 * 128 / 32` blocks of 20 B = 5_120 B.
    ///
    /// *Mutations that red this:* pricing the packed `.weight` at `header.data_bytes` (4_096 +
    /// sidecars); summing the whole `gemma/` directory instead of `pid_gemma_source` (the shard
    /// joins the merged file); pricing dense tensors at `FLOAT_WIDTH`.
    #[test]
    fn pid_gemma_is_priced_from_the_file_the_engine_opens_at_its_resident_format() {
        let temp = tempfile::tempdir().unwrap();
        let (spec, root) = dense_spec(&temp);
        let pid_checkpoint = root.join("pid-student.safetensors");
        write_tensor(&pid_checkpoint, DType::F16);
        let gemma = root.join("gemma");
        std::fs::create_dir_all(&gemma).unwrap();
        let merged = gemma.join(candle_gen_pid::engine::GEMMA_MERGED_FILE);
        safetensors::serialize_to_file(
            vec![
                (
                    "model.layers.0.mlp.down_proj.weight".to_owned(),
                    Tensor::zeros((64, 16), DType::U32, &Device::Cpu).unwrap(),
                ),
                (
                    "model.layers.0.mlp.down_proj.scales".to_owned(),
                    Tensor::zeros((64, 2), DType::F16, &Device::Cpu).unwrap(),
                ),
                (
                    "model.layers.0.mlp.down_proj.biases".to_owned(),
                    Tensor::zeros((64, 2), DType::F16, &Device::Cpu).unwrap(),
                ),
                (
                    "model.embed_tokens.weight".to_owned(),
                    Tensor::zeros((2, 4), DType::F16, &Device::Cpu).unwrap(),
                ),
            ],
            None,
            &merged,
        )
        .unwrap();
        // A shard beside the merged file: `PidEngine::load` never opens it when the merged file
        // exists, so it must not be priced either.
        let shard = gemma.join("model-00001-of-00001.safetensors");
        safetensors::serialize_to_file(
            vec![(
                "model.norm.weight".to_owned(),
                Tensor::zeros((100,), DType::F16, &Device::Cpu).unwrap(),
            )],
            None,
            &shard,
        )
        .unwrap();
        let mut pid_spec = spec.clone();
        pid_spec.pid = Some(gen_core::PidWeights {
            checkpoint: WeightsSource::File(pid_checkpoint),
            gemma: WeightsSource::Dir(gemma.clone()),
        });
        let caption_bytes = |spec: &LoadSpec| {
            component_table(&provider_contract_for_spec(spec).unwrap())
                .into_iter()
                .find(|(id, _, _)| id == PID_CAPTION_ENCODER_COMPONENT_ID)
                .map(|(_, _, bytes)| bytes)
                .unwrap()
        };
        assert_eq!(
            caption_bytes(&pid_spec),
            64 * 128 / 32 * 20 + 2 * 4 * 4,
            "the merged file alone: one Q4_1 QTensor plus the f32 embedding"
        );

        // Without the merged file the engine falls back to every shard in the directory.
        std::fs::remove_file(&merged).unwrap();
        assert_eq!(
            caption_bytes(&pid_spec),
            100 * 4,
            "the shard directory at the f32 load width"
        );
    }

    /// The registered `sdxl` id seals its contract with `spec.control` set (`lib.rs` routes control
    /// renders through `SdxlControlGenerator` under that id), so a control render's contract must
    /// conform: every overlay source is a typed `MemoryResidentComponent` whose sum is
    /// `overlay_bytes`, on the registered surface as well as the bespoke one, while the weights-free
    /// surface keeps declaring nothing.
    ///
    /// *Mutation that reds this:* leaving `overlay_bytes` an untyped aggregate (skipping every
    /// `push_overlay` and summing the sources directly) — `conformance_errors` then reports the
    /// non-zero overlay without typed auxiliary components.
    #[test]
    fn registered_control_renders_declare_typed_overlay_components_and_conform() {
        let temp = tempfile::tempdir().unwrap();
        let (spec, root) = dense_spec(&temp);
        let control = root.join("controlnet.safetensors");
        let extra = root.join("controlnet-2.safetensors");
        for (path, elements) in [(&control, 3), (&extra, 5)] {
            safetensors::serialize_to_file(
                vec![(
                    "x.weight".to_owned(),
                    Tensor::zeros((elements,), DType::F16, &Device::Cpu).unwrap(),
                )],
                None,
                path,
            )
            .unwrap();
        }
        let mut control_spec = spec.clone();
        control_spec.control = Some(WeightsSource::File(control));
        control_spec.extra_controls = vec![WeightsSource::File(extra)];

        for surface in [SdxlSurface::Registered, SdxlSurface::Bespoke] {
            let contract = SdxlArtifactSeal::capture_for(&control_spec, surface)
                .unwrap()
                .contract()
                .clone();
            assert_eq!(
                component_table(&contract),
                vec![
                    (
                        CONTROL_COMPONENT_ID.to_owned(),
                        MemoryComponentKind::ControlBranch,
                        3 * 2
                    ),
                    (
                        format!("{CONTROL_COMPONENT_ID}_2"),
                        MemoryComponentKind::ControlBranch,
                        5 * 2
                    ),
                ],
                "{surface:?}: one typed branch per ControlNet source"
            );
            assert_eq!(contract.asset_facts.overlay_bytes, 3 * 2 + 5 * 2);
            assert!(
                contract.conformance_errors().is_empty(),
                "{surface:?}: {:?}",
                contract.conformance_errors()
            );
        }

        // The weights-free surface is unchanged: no bytes, no components, still conformant.
        let weights_free = weights_free_contract(&control_spec).unwrap();
        assert_eq!(weights_free.asset_facts, MemoryAssetFacts::default());
        assert!(weights_free.resident_components().is_empty());
        assert!(weights_free.conformance_errors().is_empty());
        gen_core_testkit::assert_memory_contract_asset_facts_conform(&weights_free);
    }

    /// AC (epic SC-22657, E2): a materialized SDXL snapshot publishes the axes of the vendored
    /// UNet + VAE configs the loader builds, declines the four a UNet denoiser structurally lacks,
    /// and the weights-free surface publishes none.
    #[test]
    fn architecture_facts_match_the_loader_config_and_pass_conformance() {
        let temp = tempfile::tempdir().unwrap();
        let (spec, root) = dense_spec(&temp);
        // The shared fixture gives every component the same one-element tensor, which the
        // conformance check reads as one component borrowing another's price. Widen the encoder
        // shards AND the UNet so each is priced from its own distinct bytes. (Since SC-22667 the
        // VAE is priced at the same f16 width as the rest, so a differing dtype no longer keeps
        // the fixture's components apart by accident.)
        for component in ["text_encoder", "text_encoder_2"] {
            let path = root.join(component).join("model.fp16.safetensors");
            let tensor = Tensor::zeros((4, 4), DType::F16, &Device::Cpu).unwrap();
            safetensors::serialize_to_file(vec![("x.weight".to_owned(), tensor)], None, &path)
                .unwrap();
        }
        {
            let path = root.join("unet/diffusion_pytorch_model.fp16.safetensors");
            let tensor = Tensor::zeros((8, 8), DType::F16, &Device::Cpu).unwrap();
            safetensors::serialize_to_file(vec![("x.weight".to_owned(), tensor)], None, &path)
                .unwrap();
        }
        let contract = provider_contract_for_spec(&spec).unwrap();
        assert_eq!(
            contract.architecture_facts,
            gen_core::MemoryArchitectureFacts {
                // `sdxl_unet_config()` heads are per stage (5/10/20): no uniform head count.
                attention_heads: None,
                // Every stage's `out_channels / attention_head_dim` is 64 (320/5, 640/10, 1280/20).
                head_dim: Some(64),
                // A UNet down/mid/up trunk is not a uniform transformer-block stack.
                transformer_blocks: None,
                // The UNet consumes the latent grid directly; nothing is patchified.
                patch_size: None,
                // `sdxl_vae_config().latent_channels`.
                latent_channels: Some(4),
                // `block_out_channels` `[128,256,512,512]` = 4 stages => 3 halvings => x8.
                vae_spatial_scale: Some(8),
                // SDXL ships the image `AutoencoderKL`: no temporal axis exists to declare.
                vae_temporal_scale: None,
                // `lib.rs` pins `DType::F16` on the loaded generator.
                activation_dtype_width: Some(2),
            }
        );
        gen_core_testkit::assert_memory_contract_facts_conform(&contract);

        // The registry's weights-free surface resolves no snapshot, so no axis is knowable.
        for route in SDXL_ROUTES {
            assert!(routed_weights_free_contract(*route)
                .architecture_facts
                .is_empty());
        }
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
    fn dense_receipt_reports_the_shared_artifact_seal_grammar_for_same_size_replacement() {
        let temp = tempfile::tempdir().unwrap();
        let (spec, root) = dense_spec(&temp);
        let seal = SdxlArtifactSeal::capture(&spec).unwrap();
        let source = root.join("unet/config.json");
        let original = std::fs::read(&source).unwrap();
        let mut replacement = original.clone();
        replacement[0] ^= 1;
        std::fs::write(&source, replacement).unwrap();
        // A read-only handle cannot carry a timestamp write on Windows (`PermissionDenied`), so the
        // replacement stamp goes through a writable handle — the shape
        // `candle_gen::quant::sidecar::restore_modified` already uses.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&source)
            .unwrap()
            .set_modified(
                std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_456),
            )
            .unwrap();

        let error = seal.ensure_unchanged().unwrap_err().to_string();
        assert!(
            error.contains("artifact seal mismatch after load"),
            "Candle receipt must expose the shared seal error grammar: {error}"
        );
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
