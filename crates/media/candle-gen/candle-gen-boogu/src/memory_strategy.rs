//! Exact, request-scoped Candle/CUDA memory contract for the three public Boogu routes.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, GenerationMemory, GenerationRequest, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryFormulaKind, MemoryFormulaVariable, MemoryGeometry,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategyCapability, MemoryStrategySupport, MemoryWindowMaterialization,
    Precision, Quant, ResidentRequestMemory, WeightsSource,
};
use candle_gen::gen_core::{MemoryBehaviorFixture, MemoryBehaviorRoute};
use candle_gen::gen_core::{MemoryBudget, MemoryCacheState, MemoryOptimizationAuthority};
use sha2::{Digest, Sha256};

use crate::{BOOGU_IMAGE_EDIT_ID, BOOGU_IMAGE_ID, BOOGU_IMAGE_TURBO_ID, SIZE_MULTIPLE};

pub const CANONICAL_REPOSITORY: &str = "SceneWorks/boogu-image-mlx";
pub const CANONICAL_REVISION: &str = "a459e614d408bfdf57089c32cc3da706f5a017de";
const GROUP_SIZE: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Base,
    Turbo,
    Edit,
}

impl Route {
    fn for_provider(provider: &str) -> gen_core::Result<Self> {
        match provider {
            BOOGU_IMAGE_ID => Ok(Self::Base),
            BOOGU_IMAGE_TURBO_ID => Ok(Self::Turbo),
            BOOGU_IMAGE_EDIT_ID => Ok(Self::Edit),
            _ => Err(gen_core::Error::Unsupported(format!(
                "unknown Boogu memory route {provider}"
            ))),
        }
    }

    fn variant(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Turbo => "turbo",
            Self::Edit => "edit",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ArtifactReceipt {
    root: PathBuf,
    inventory: Vec<(PathBuf, gen_core::PinnedWeightsFile, String)>,
    pub(crate) tier: Option<Quant>,
    pub(crate) canonical: bool,
    pub(crate) facts: MemoryAssetFacts,
}

impl ArtifactReceipt {
    pub(crate) fn capture(provider: &str, spec: &LoadSpec) -> gen_core::Result<Self> {
        validate_load_spec(provider, spec)?;
        let route = Route::for_provider(provider)?;
        let WeightsSource::Dir(lexical_root) = &spec.weights else {
            unreachable!("validated directory source")
        };
        let lexical_root = std::path::absolute(lexical_root)?;
        let root = std::fs::canonicalize(&lexical_root)?;
        let tier = detect_tier(&root)?;
        if tier != spec.quantize {
            return Err(gen_core::Error::Unsupported(format!(
                "{provider}: requested tier {:?} crossed actual tensor tier {tier:?}",
                spec.quantize
            )));
        }
        validate_packing_configs(&root, tier)?;
        // PiD is a distinct decoder artifact/lifecycle. It remains supported by the resident
        // generator path, but cannot inherit the native-VAE staged receipt or its phase envelope.
        let canonical = spec.pid.is_none()
            && spec.resolved_route.as_deref() == Some(provider)
            && canonical_artifact_path(&root, route, tier);
        let facts = projected_facts(&root, route, tier)?;
        let files = recursive_files(&root)?;
        let mut inventory = Vec::with_capacity(files.len());
        for path in files {
            let pin = if spec.prepared_file_pins().is_prepared() {
                spec.prepared_file_pins()
                    .get(&path)
                    .or_else(|| {
                        spec.prepared_file_pins()
                            .get(&lexical_root.join(path.strip_prefix(&root).unwrap_or(&path)))
                    })
                    .cloned()
                    .ok_or_else(|| {
                        gen_core::Error::Unsupported(format!(
                            "{provider}: sealed artifact receipt is missing {}",
                            path.display()
                        ))
                    })?
            } else {
                gen_core::PinnedWeightsFile::pin(&path)?
            };
            let digest = pin
                .content_sha256()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            inventory.push((path, pin, digest));
        }
        let receipt = Self {
            root,
            inventory,
            tier,
            canonical,
            facts,
        };
        receipt.ensure_unchanged()?;
        Ok(receipt)
    }

    pub(crate) fn ensure_unchanged(&self) -> gen_core::Result<()> {
        let current = recursive_files(&self.root)?;
        let expected = self
            .inventory
            .iter()
            .map(|(path, _, _)| path.clone())
            .collect::<Vec<_>>();
        if current != expected {
            return Err(gen_core::Error::Unsupported(
                "boogu: artifact inventory changed after the immutable load receipt was sealed"
                    .into(),
            ));
        }
        for (_, pin, digest) in &self.inventory {
            if digest.len() != 64 {
                return Err(gen_core::Error::Unsupported(
                    "boogu: immutable artifact receipt contains an invalid digest".into(),
                ));
            }
            pin.verify_unchanged()?;
        }
        Ok(())
    }
}

fn validate_load_spec(provider: &str, spec: &LoadSpec) -> gen_core::Result<()> {
    Route::for_provider(provider)?;
    if !matches!(spec.weights, WeightsSource::Dir(_))
        || spec.precision != Precision::Bf16
        || !matches!(spec.quantize, None | Some(Quant::Q4 | Quant::Q8))
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider}: memory strategies require an exact bf16/q4/q8 turnkey directory"
        )));
    }
    if !spec.adapters.is_empty()
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider}: adapters and auxiliary conditioning are outside the Boogu memory contract"
        )));
    }
    gen_core::reject_unknown_components(spec, &[], provider)
}

fn canonical_artifact_path(root: &Path, route: Route, tier: Option<Quant>) -> bool {
    let expected = match tier {
        None => format!("{}-bf16", route.variant()),
        Some(Quant::Q4) => format!("{}-q4", route.variant()),
        Some(Quant::Q8) => route.variant().to_owned(),
        Some(_) => return false,
    };
    let parts = root
        .components()
        .filter_map(|part| part.as_os_str().to_str())
        .collect::<Vec<_>>();
    parts.ends_with(&[
        "models--SceneWorks--boogu-image-mlx",
        "snapshots",
        CANONICAL_REVISION,
        expected.as_str(),
    ]) || parts.ends_with(&[
        "SceneWorks__boogu-image-mlx",
        CANONICAL_REVISION,
        expected.as_str(),
    ])
}

pub(crate) fn canonical_load_identity(provider: &str, spec: &LoadSpec) -> bool {
    let (Ok(route), WeightsSource::Dir(root)) = (Route::for_provider(provider), &spec.weights)
    else {
        return false;
    };
    let canonical = std::fs::canonicalize(root).unwrap_or_else(|_| root.clone());
    spec.resolved_route.as_deref() == Some(provider)
        && canonical_artifact_path(&canonical, route, spec.quantize)
}

fn direct_safetensors(dir: &Path) -> gen_core::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut nested = false;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let path = entry.path();
        if ty.is_dir() {
            nested = true;
            continue;
        }
        if (ty.is_file() || ty.is_symlink())
            && path.extension().and_then(|v| v.to_str()) == Some("safetensors")
        {
            if entry.file_name().to_string_lossy().starts_with('.') {
                return Err(gen_core::Error::Unsupported(format!(
                    "boogu: hidden artifact {}",
                    path.display()
                )));
            }
            files.push(std::path::absolute(path)?);
        }
    }
    if nested || files.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "boogu: {} must contain a non-empty direct safetensors inventory with no nested directories",
            dir.display()
        )));
    }
    files.sort();
    files.dedup();
    Ok(files)
}

fn component_headers(
    root: &Path,
    component: &str,
) -> gen_core::Result<Vec<gen_core::weightsmeta::SafetensorsTensorHeader>> {
    let mut all = Vec::new();
    let mut names = BTreeSet::new();
    for file in direct_safetensors(&root.join(component))? {
        for header in gen_core::weightsmeta::safetensors_path_tensor_headers(&file)? {
            if !names.insert(header.name.clone()) {
                return Err(gen_core::Error::Unsupported(format!(
                    "boogu: duplicate tensor {:?} across {component} shards",
                    header.name
                )));
            }
            all.push(header);
        }
    }
    if all.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "boogu: empty {component} tensor inventory"
        )));
    }
    Ok(all)
}

fn detect_tier(root: &Path) -> gen_core::Result<Option<Quant>> {
    let mut observed: Option<Quant> = None;
    for component in ["transformer", "mllm"] {
        let headers = component_headers(root, component)?;
        let by_name = headers
            .iter()
            .map(|h| (h.name.as_str(), h))
            .collect::<BTreeMap<_, _>>();
        for header in headers
            .iter()
            .filter(|header| header.name.ends_with(".scales") || header.name.ends_with(".biases"))
        {
            let base = header
                .name
                .strip_suffix(".scales")
                .or_else(|| header.name.strip_suffix(".biases"))
                .unwrap();
            if !by_name.contains_key(format!("{base}.weight").as_str())
                || !by_name.contains_key(format!("{base}.scales").as_str())
                || !by_name.contains_key(format!("{base}.biases").as_str())
            {
                return Err(gen_core::Error::Unsupported(format!(
                    "boogu: orphan or incomplete packed triple {base}"
                )));
            }
        }
        for weight in headers.iter().filter(|h| h.name.ends_with(".weight")) {
            let base = weight.name.strip_suffix(".weight").unwrap();
            let scales = by_name.get(format!("{base}.scales").as_str());
            let biases = by_name.get(format!("{base}.biases").as_str());
            match (scales, biases) {
                (None, None) => {
                    if !matches!(
                        weight.dtype,
                        gen_core::weightsmeta::Dtype::BF16 | gen_core::weightsmeta::Dtype::F32
                    ) {
                        return Err(gen_core::Error::Unsupported(format!(
                            "boogu: dense tensor {:?} has non-dense dtype {:?}",
                            weight.name, weight.dtype
                        )));
                    }
                }
                (Some(scales), Some(biases)) => {
                    use gen_core::weightsmeta::Dtype;
                    if weight.dtype != Dtype::U32
                        || !matches!(scales.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
                        || !matches!(biases.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
                        || scales.shape != biases.shape
                    {
                        return Err(gen_core::Error::Unsupported(format!(
                            "boogu: malformed affine-packed triple {base}"
                        )));
                    }
                    let [out, lanes] = weight.shape.as_slice() else {
                        return Err(gen_core::Error::Unsupported(format!(
                            "boogu: packed {base} weight is not rank two"
                        )));
                    };
                    let [scale_out, groups] = scales.shape.as_slice() else {
                        return Err(gen_core::Error::Unsupported(format!(
                            "boogu: packed {base} scales are not rank two"
                        )));
                    };
                    if out != scale_out || *groups == 0 {
                        return Err(gen_core::Error::Unsupported(format!(
                            "boogu: packed {base} geometry disagrees"
                        )));
                    }
                    let logical = groups.checked_mul(GROUP_SIZE).ok_or_else(|| {
                        gen_core::Error::Msg("boogu: packed geometry overflow".into())
                    })?;
                    let bits = lanes
                        .checked_mul(32)
                        .filter(|n| n.is_multiple_of(logical))
                        .map(|n| n / logical)
                        .ok_or_else(|| {
                            gen_core::Error::Unsupported(format!(
                                "boogu: packed {base} has no exact group-32 bit ratio"
                            ))
                        })?;
                    let tier = match bits {
                        4 => Quant::Q4,
                        8 => Quant::Q8,
                        _ => {
                            return Err(gen_core::Error::Unsupported(format!(
                                "boogu: unsupported packed {bits}-bit tensor {base}"
                            )))
                        }
                    };
                    if observed.is_some_and(|prior| prior != tier) {
                        return Err(gen_core::Error::Unsupported(
                            "boogu: artifact mixes q4 and q8 packed tensors".into(),
                        ));
                    }
                    observed = Some(tier);
                }
                _ => {
                    return Err(gen_core::Error::Unsupported(format!(
                        "boogu: packed tensor {base} is missing its exact scales/biases triple"
                    )))
                }
            }
        }
    }
    Ok(observed)
}

fn validate_packing_configs(root: &Path, tier: Option<Quant>) -> gen_core::Result<()> {
    for component in ["transformer", "mllm"] {
        let path = root.join(component).join("config.json");
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path)?).map_err(|error| {
                gen_core::Error::Unsupported(format!(
                    "boogu: malformed {}: {error}",
                    path.display()
                ))
            })?;
        let quant = value.get("quantization");
        match tier {
            None if quant.is_none() => {}
            Some(tier) => {
                let bits = quant.and_then(|q| q.get("bits")).and_then(|v| v.as_u64());
                let group = quant
                    .and_then(|q| q.get("group_size"))
                    .and_then(|v| v.as_u64());
                if bits != Some(tier.bits() as u64) || group != Some(GROUP_SIZE as u64) {
                    return Err(gen_core::Error::Unsupported(format!("boogu: {component} packing marker crossed actual {:?}/group-{GROUP_SIZE} tensor geometry", tier)));
                }
            }
            _ => {
                return Err(gen_core::Error::Unsupported(format!(
                    "boogu: {component} carries a packing marker for dense bf16 tensors"
                )))
            }
        }
    }
    validate_native_vae(root)?;
    Ok(())
}

fn validate_native_vae(root: &Path) -> gen_core::Result<()> {
    let path = root.join("vae/config.json");
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path)?).map_err(|error| {
            gen_core::Error::Unsupported(format!("boogu: malformed {}: {error}", path.display()))
        })?;
    let exact = value.get("in_channels").and_then(|v| v.as_u64()) == Some(3)
        && value.get("out_channels").and_then(|v| v.as_u64()) == Some(3)
        && value.get("latent_channels").and_then(|v| v.as_u64()) == Some(16)
        && value.get("layers_per_block").and_then(|v| v.as_u64()) == Some(2)
        && value.get("norm_num_groups").and_then(|v| v.as_u64()) == Some(32)
        && value
            .get("block_out_channels")
            .and_then(|v| v.as_array())
            .is_some_and(|values| {
                values.iter().filter_map(|v| v.as_u64()).collect::<Vec<_>>() == [128, 256, 512, 512]
            })
        && value
            .get("scaling_factor")
            .and_then(|v| v.as_f64())
            .is_some_and(|v| v.to_bits() == 0.3611_f64.to_bits())
        && value
            .get("shift_factor")
            .and_then(|v| v.as_f64())
            .is_some_and(|v| v.to_bits() == 0.1159_f64.to_bits());
    if !exact {
        return Err(gen_core::Error::Unsupported(
            "boogu: native VAE config crossed the exact FLUX.1/Z-Image identity".into(),
        ));
    }

    let headers = component_headers(root, "vae")?;
    if headers.iter().any(|header| {
        header.dtype != gen_core::weightsmeta::Dtype::F32
            || header.shape.is_empty()
            || header.shape.contains(&0)
    }) {
        return Err(gen_core::Error::Unsupported(
            "boogu: native VAE tensors must be non-empty f32 encoder/decoder tensors".into(),
        ));
    }
    let by_name = headers
        .iter()
        .map(|header| (header.name.as_str(), header.shape.as_slice()))
        .collect::<BTreeMap<_, _>>();
    for (name, shape) in [
        ("encoder.conv_in.weight", &[128, 3, 3, 3][..]),
        ("encoder.conv_out.weight", &[32, 128, 3, 3][..]),
        ("decoder.conv_in.weight", &[512, 16, 3, 3][..]),
        ("decoder.conv_out.weight", &[3, 128, 3, 3][..]),
    ] {
        if by_name.get(name).copied() != Some(shape) {
            return Err(gen_core::Error::Unsupported(format!(
                "boogu: native VAE selected tensor {name} crossed its expected geometry"
            )));
        }
    }
    Ok(())
}

fn projected_facts(
    root: &Path,
    route: Route,
    tier: Option<Quant>,
) -> gen_core::Result<MemoryAssetFacts> {
    let conditioning_bytes = projected_component(root, "mllm", 2, route == Route::Edit)?;
    let transformer_bytes = projected_component(root, "transformer", 2, false)?;
    let decoder_bytes = projected_component(root, "vae", 4, false)?;
    // One network, one field (epic SC-22657, E1; feature-end ruling SC-22667). The reference
    // encoder every Edit request and the Base/Turbo img2img surface run during conditioning is
    // the same f32 VAE weights the decode phase runs, so those bytes are charged exactly once, in
    // `decoder_bytes`. They used to be folded into `conditioning_bytes` as well, which charged one
    // resident network twice against every fit decision. That the VAE is *resident during
    // conditioning* on those routes is a lifecycle fact the contract cannot yet state per base
    // component — see `MemoryAssetFacts` — and is recorded on this crate's contract doc instead.
    let base_bytes = conditioning_bytes
        .checked_add(transformer_bytes)
        .and_then(|v| v.checked_add(decoder_bytes))
        .ok_or_else(|| gen_core::Error::Msg("boogu: projected resident byte overflow".into()))?;
    let _ = (route, tier);
    Ok(MemoryAssetFacts {
        base_bytes,
        conditioning_bytes,
        transformer_bytes,
        decoder_bytes,
        overlay_bytes: 0,
    })
}

fn projected_component(
    root: &Path,
    component: &str,
    dense_width: u64,
    vision_f32: bool,
) -> gen_core::Result<u64> {
    let headers = component_headers(root, component)?;
    let by_name = headers
        .iter()
        .map(|header| (header.name.as_str(), header))
        .collect::<BTreeMap<_, _>>();
    headers.iter().try_fold(0_u64, |total, tensor| {
        if tensor.name.ends_with(".scales") || tensor.name.ends_with(".biases") {
            // MLX affine sidecars are transient pack inputs, not separately resident tensors.
            return Ok(total);
        }
        let base = tensor.name.strip_suffix(".weight");
        let scale_name = base.map(|base| format!("{base}.scales"));
        let bytes = if let Some(scales) = scale_name.as_deref().and_then(|name| by_name.get(name)) {
            let bias_name = format!("{}.biases", base.expect("packed weight has a base"));
            let biases = by_name.get(bias_name.as_str()).ok_or_else(|| {
                gen_core::Error::Unsupported(format!(
                    "boogu: packed tensor {:?} is missing its bias sidecar",
                    tensor.name
                ))
            })?;
            candle_gen::quant::mlx_packed_qtensor_resident_bytes(
                tensor, scales, biases, GROUP_SIZE,
            )?
        } else {
            let width = if vision_f32 && tensor.name.starts_with("model.visual.") {
                4
            } else {
                dense_width
            };
            tensor.materialized_bytes(width)?
        };
        total
            .checked_add(bytes)
            .ok_or_else(|| gen_core::Error::Msg("boogu: byte projection overflow".into()))
    })
}

fn recursive_files(root: &Path) -> gen_core::Result<Vec<PathBuf>> {
    fn visit(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
        let mut entries = std::fs::read_dir(dir)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            let ty = entry.file_type()?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.')
                || name.ends_with(".part")
                || name.ends_with(".partial")
                || name.ends_with(".incomplete")
                || name.ends_with(".tmp")
                || name.ends_with(".lock")
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("partial-download marker {}", path.display()),
                ));
            }
            if ty.is_dir() {
                visit(&path, out)?;
            } else if ty.is_file() || ty.is_symlink() {
                out.push(std::path::absolute(path)?);
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    visit(root, &mut files)?;
    files.sort();
    files.dedup();
    if files.is_empty() {
        return Err(gen_core::Error::Unsupported("boogu: empty artifact".into()));
    }
    Ok(files)
}

fn tier(quant: Option<Quant>) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: Precision::Bf16,
        quant,
        component_precision_floors: &[],
    }
}

fn estimated_behavior_context(
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
    numeric: MemoryNumericTier,
    route: gen_core::MemoryBehaviorRoute,
) -> gen_core::Result<MemoryRunContext> {
    Ok(MemoryRunContext {
        selection: contract.representative_selection(strategy, numeric, route.use_pid)?,
        optimization_authority: MemoryOptimizationAuthority::Estimated,
        calibration_abi: 0,
        calibration_fingerprint: String::new(),
        load_shape: contract.load_shape,
        mode: route.mode,
        has_reference: route.reference_count > 0,
        use_pid: route.use_pid,
        has_phases: route.has_phases,
        geometry: MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: route.reference_count,
        },
        overlay: route.overlay,
        budget: MemoryBudget {
            total_bytes: 8 * 1024 * 1024 * 1024,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1024 * 1024 * 1024,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: "boogu-structural-estimate".into(),
    })
}

/// Activation dtype the Boogu DiT computes in. `pipeline.rs` pins `DIT_DTYPE = DType::BF16`
/// (candle's native CUDA width for the 10 B trunk), so this is the provider's real activation
/// width rather than a memory-model literal. The FLUX.1 VAE runs f32, but the phase envelope's
/// activation term describes the denoiser.
const ACTIVATION_DTYPE: candle_gen::candle_core::DType = candle_gen::candle_core::DType::BF16;

/// Snapshot-read architecture axes for the three Boogu routes (epic SC-22657, E2).
///
/// The DiT axes come from the **same** configuration the loader builds its model from:
/// `pipeline::load_components` calls [`crate::config::BooguConfig::from_snapshot`], which parses
/// `<root>/transformer/config.json` (`num_attention_heads`, `hidden_size`, `num_layers`,
/// `patch_size`, …) and falls back per field to the published [`config::BooguConfig::base`]
/// reference. Reading the axes back off the returned struct publishes what the pipeline will
/// actually construct; a snapshot whose config disagrees with the reference publishes what it says.
///
/// `head_dim` is `hidden_size / num_attention_heads` (3360 / 28 = 120) and is published only when
/// the division is exact, so a non-uniform-head snapshot claims no head width it does not have.
///
/// `transformer_blocks` is `num_layers`, the **total** trunk: the config's
/// `num_double_stream_layers` (8) is the double-stream prefix already counted inside it, not an
/// addend.
///
/// The decoder axes come from the `VaeConfig::z_image()` the loader constructs at
/// `pipeline.rs` — `latent_channels = 16`, and a four-entry `block_out_channels` whose three
/// downsampling stages give the ×8 spatial scale — rather than from a `vae/config.json` the loader
/// never reads.
///
/// A weights-free contract — the registry's sentinel surface path, a single-file import, or a
/// snapshot whose transformer config cannot be parsed — publishes
/// `MemoryArchitectureFacts::default()`.
fn architecture_facts(spec: &LoadSpec) -> gen_core::MemoryArchitectureFacts {
    use candle_gen::architecture_facts as af;

    let Some(root) = af::snapshot_root(spec) else {
        return gen_core::MemoryArchitectureFacts::default();
    };
    let Ok(config) = crate::config::BooguConfig::from_snapshot(root) else {
        return gen_core::MemoryArchitectureFacts::default();
    };
    let vae = candle_transformers::models::z_image::vae::VaeConfig::z_image();
    let attention_heads = af::declared(config.num_attention_heads);
    gen_core::MemoryArchitectureFacts {
        attention_heads,
        head_dim: af::head_dim(af::declared(config.hidden_size), attention_heads),
        transformer_blocks: af::declared(config.num_layers),
        patch_size: af::declared(config.patch_size),
        latent_channels: af::declared(vae.latent_channels),
        vae_spatial_scale: af::spatial_scale_from_stages(
            Some(&serde_json::json!({ "block_out_channels": vae.block_out_channels })),
            &["block_out_channels"],
        ),
        // Structurally absent: the FLUX.1 16-channel AutoencoderKL is an image VAE with no
        // temporal axis at all (absent is `None`, never `Some(0)`).
        vae_temporal_scale: None,
        activation_dtype_width: af::dtype_width(ACTIVATION_DTYPE),
    }
}

/// Assemble the Boogu contract over sealed asset facts.
///
/// **Decoder residency during conditioning.** On the Edit route and the Base/Turbo img2img
/// surface the reference image is encoded through the same f32 VAE that later decodes, so the
/// decoder is resident during the `Conditioning` phase as well as `Decode`. `asset_facts` charges
/// those bytes once, in `decoder_bytes` (one network, one field — `MemoryAssetFacts`); the
/// contract has no per-phase residency declaration for a base component, so this note is where
/// that co-residency is stated until it does.
fn build_contract(
    provider: &str,
    spec: &LoadSpec,
    canonical: bool,
    facts: MemoryAssetFacts,
) -> MemoryProviderContract {
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    MemoryProviderContract {
        architecture_facts: architecture_facts(spec),
        provider_id: provider.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: false,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies: MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: match strategy {
                    MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
                    MemoryStrategy::StagedResidency if canonical => {
                        MemoryStrategySupport::Implemented
                    }
                    _ => MemoryStrategySupport::Missing,
                },
                parameters: MemoryParameterRanges::default(),
            })
            .collect(),
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: canonical,
            decode_tiling: false,
            attention_chunking: false,
            transformer_window_materialization: false,
        },
        formula: MemoryFormulaKind::PhaseEnvelope {
            phases,
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::BatchCount,
                MemoryFormulaVariable::ConditioningTokenCount,
            ],
        },
        calibration: None,
        asset_facts: facts,
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    }
}

pub(crate) struct PreparedMemory {
    pub(crate) receipt: ArtifactReceipt,
    pub(crate) contract: MemoryProviderContract,
}

impl PreparedMemory {
    pub(crate) fn prepare(provider: &str, spec: &LoadSpec) -> gen_core::Result<Self> {
        let receipt = ArtifactReceipt::capture(provider, spec)?;
        let contract = build_contract(provider, spec, receipt.canonical, receipt.facts);
        Ok(Self { receipt, contract })
    }
}

pub fn provider_contract(
    provider: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    if !canonical_load_identity(provider, spec) {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider}: memory strategies require the exact immutable Boogu turnkey artifact"
        )));
    }
    PreparedMemory::prepare(provider, spec).map(|p| p.contract)
}

fn weights_free_contract(
    provider: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    validate_load_spec(provider, spec)?;
    Ok(build_contract(
        provider,
        spec,
        true,
        MemoryAssetFacts::default(),
    ))
}

fn validate_route(
    provider: &str,
    contract: &MemoryProviderContract,
    numeric: MemoryNumericTier,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    if contract.calibration.is_none()
        && (context.calibration_abi != 0
            || !context.calibration_fingerprint.is_empty()
            || context.load_shape != contract.load_shape)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider}: structural-estimate handshake crossed ABI, fingerprint, or load shape"
        )));
    }
    let route = Route::for_provider(provider)?;
    let refs = context.geometry.reference_count;
    let valid_mode = match route {
        Route::Base | Route::Turbo if refs == 0 => {
            context.mode == MemoryMode::TextToImage && context.overlay.is_none()
        }
        Route::Base | Route::Turbo if refs == 1 => {
            context.mode == MemoryMode::ImageToImage
                && matches!(
                    context.overlay.as_deref(),
                    Some("reference_inert" | "reference_active")
                )
        }
        Route::Edit => {
            context.mode == MemoryMode::Edit && (1..=5).contains(&refs) && context.overlay.is_none()
        }
        _ => false,
    };
    if !valid_mode
        || context.has_reference != (refs > 0)
        || context.geometry.batch != 1
        || context.geometry.frames != 1
        || context.geometry.width < 256
        || context.geometry.width > 2048
        || context.geometry.height < 256
        || context.geometry.height > 2048
        || !context.geometry.width.is_multiple_of(SIZE_MULTIPLE)
        || !context.geometry.height.is_multiple_of(SIZE_MULTIPLE)
        || context.has_phases
        || (context.selection.strategy == MemoryStrategy::StagedResidency && context.use_pid)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider}: request crossed the admitted route/mode/reference/PiD/geometry identity"
        )));
    }
    match gen_core::standard_memory_strategy_safety_check(contract, context, Some(numeric), None) {
        MemorySafetyDecision::Accept => Ok(()),
        MemorySafetyDecision::Reject { reason } => Err(gen_core::Error::Unsupported(reason)),
    }
}

pub(crate) fn safety_check(
    provider: &str,
    contract: &MemoryProviderContract,
    numeric: MemoryNumericTier,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match validate_route(provider, contract, numeric, context) {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RequestBinding {
    address: usize,
    geometry: MemoryGeometry,
    memory: Option<GenerationMemory>,
    use_pid: bool,
    has_phases: bool,
    seed: Option<u64>,
    steps: Option<u32>,
    sampler: Option<String>,
    scheduler: Option<String>,
    guidance_bits: Option<u32>,
    true_cfg_bits: Option<u32>,
    shift_bits: Option<u32>,
    preview_active: bool,
    references: Vec<(u32, u32, [u8; 32], Option<u32>)>,
}

impl RequestBinding {
    fn from_request(request: &GenerationRequest) -> Self {
        let mut references = Vec::new();
        for conditioning in &request.conditioning {
            match conditioning {
                gen_core::Conditioning::Reference { image, strength } => {
                    references.push(reference_identity(image, *strength));
                }
                gen_core::Conditioning::MultiReference { images } => {
                    references.extend(images.iter().map(|image| reference_identity(image, None)))
                }
                _ => {}
            }
        }
        Self {
            address: std::ptr::from_ref(request).addr(),
            geometry: MemoryGeometry {
                width: request.width,
                height: request.height,
                batch: request.count,
                frames: request.frames.unwrap_or(1),
                reference_count: request.image_reference_count(),
            },
            memory: request.memory,
            use_pid: request.use_pid,
            has_phases: request
                .phases
                .as_ref()
                .is_some_and(|phases| !phases.is_empty()),
            seed: request.seed,
            steps: request.steps,
            sampler: request.sampler.clone(),
            scheduler: request.scheduler.clone(),
            guidance_bits: request.guidance.map(f32::to_bits),
            true_cfg_bits: request.true_cfg.map(f32::to_bits),
            shift_bits: request.scheduler_shift.map(f32::to_bits),
            preview_active: request.preview.is_active(),
            references,
        }
    }
}

fn reference_identity(
    image: &gen_core::Image,
    strength: Option<f32>,
) -> (u32, u32, [u8; 32], Option<u32>) {
    let mut digest = Sha256::new();
    digest.update(image.width.to_le_bytes());
    digest.update(image.height.to_le_bytes());
    digest.update(&image.pixels);
    (
        image.width,
        image.height,
        digest.finalize().into(),
        strength.map(f32::to_bits),
    )
}

struct ActiveAdmission {
    token: u64,
    context: MemoryRunContext,
    expected_memory: Option<GenerationMemory>,
    binding: Option<RequestBinding>,
    consumed: bool,
}

#[derive(Default)]
struct AdmissionState {
    next_token: u64,
    approved_context: Option<MemoryRunContext>,
    active: Option<ActiveAdmission>,
}

#[derive(Clone)]
pub(crate) struct AdmissionRegistry {
    provider: &'static str,
    inner: Arc<Mutex<AdmissionState>>,
}

impl AdmissionRegistry {
    pub(crate) fn new(provider: &'static str) -> Self {
        Self {
            provider,
            inner: Arc::new(Mutex::new(AdmissionState::default())),
        }
    }

    pub(crate) fn approve(&self, context: &MemoryRunContext) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.inner);
        if state.active.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: another memory request is active",
                self.provider
            )));
        }
        state.approved_context = Some(context.clone());
        Ok(())
    }

    pub(crate) fn clear_approval(&self) {
        candle_gen::lock_recover(&self.inner).approved_context = None;
    }

    fn begin(
        &self,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
    ) -> gen_core::Result<u64> {
        if contract.provider_id != self.provider {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: crossed provider contract {}",
                self.provider, contract.provider_id
            )));
        }
        let mut state = candle_gen::lock_recover(&self.inner);
        if state.active.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: another memory request scope is active",
                self.provider
            )));
        }
        let approved = state.approved_context.take().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{}: memory request skipped the safety handshake",
                self.provider
            ))
        })?;
        if approved != *context {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: memory context changed after safety approval",
                self.provider
            )));
        }
        state.next_token = state.next_token.wrapping_add(1).max(1);
        let token = state.next_token;
        state.active = Some(ActiveAdmission {
            token,
            context: context.clone(),
            expected_memory: contract.generation_memory(&context.selection),
            binding: None,
            consumed: false,
        });
        Ok(token)
    }

    fn configure(&self, token: u64, request: &GenerationRequest) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.inner);
        let active = state.active.as_mut().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{}: memory request scope is no longer active",
                self.provider
            ))
        })?;
        let binding = RequestBinding::from_request(request);
        if active.token != token
            || active.binding.is_some()
            || active.consumed
            || binding.geometry != active.context.geometry
            || binding.memory != active.expected_memory
            || binding.use_pid != active.context.use_pid
            || binding.has_phases != active.context.has_phases
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: stale or changed memory request",
                self.provider
            )));
        }
        active.binding = Some(binding);
        Ok(())
    }

    pub(crate) fn consume_for_generate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.inner);
        let constrained = request
            .memory
            .is_some_and(|memory| memory != GenerationMemory::default());
        let Some(active) = state.active.as_mut() else {
            return if constrained {
                Err(gen_core::Error::Unsupported(format!(
                    "{}: constrained request has no active admission",
                    self.provider
                )))
            } else {
                Ok(())
            };
        };
        if active.binding.as_ref() != Some(&RequestBinding::from_request(request))
            || active.consumed
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request changed or admission was already consumed",
                self.provider
            )));
        }
        active.consumed = true;
        Ok(())
    }

    fn finish(&self, token: u64) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.inner);
        if state
            .active
            .as_ref()
            .is_some_and(|active| active.token == token)
        {
            state.active = None;
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: stale memory token cannot finish",
                self.provider
            )))
        }
    }

    fn abandon(&self, token: u64) {
        let mut state = candle_gen::lock_recover(&self.inner);
        if state
            .active
            .as_ref()
            .is_some_and(|active| active.token == token)
        {
            state.active = None;
        }
    }
}

pub(crate) fn validate_generation_request(
    provider: &str,
    request: &GenerationRequest,
) -> gen_core::Result<()> {
    let route = Route::for_provider(provider)?;
    let refs = request.image_reference_count();
    let expected_mode = match route {
        Route::Base | Route::Turbo if refs == 0 => "t2i",
        Route::Base | Route::Turbo if refs == 1 => "i2i",
        Route::Edit if (1..=5).contains(&refs) => "edit",
        _ => {
            return Err(gen_core::Error::Unsupported(format!(
                "{provider}: invalid reference cardinality {refs}"
            )))
        }
    };
    if request.video_mode.as_deref() != Some(expected_mode)
        || request.frames != Some(1)
        || request.count != 1
        || request.phases.is_some()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider}: generation request crossed its admitted memory identity"
        )));
    }
    if request.memory.is_some_and(|m| {
        m.chunk_attention
            || m.stream_transformer_blocks
            || m.tile_vae_decode
            || (m.stage_residency && request.use_pid)
    }) {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider}: request carries an unsupported memory mechanism"
        )));
    }
    Ok(())
}

fn begin_with_device(
    provider: &'static str,
    contract: &MemoryProviderContract,
    numeric: MemoryNumericTier,
    device: Device,
    context: &MemoryRunContext,
    admission: Option<AdmissionRegistry>,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    validate_route(provider, contract, numeric, context)?;
    let token = admission
        .as_ref()
        .map(|admission| admission.begin(contract, context))
        .transpose()?;
    let mut config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        provider,
        device,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        0,
        |_pid, _, _| {
            Err(gen_core::Error::Unsupported(
                "boogu: bounded decode is not implemented".into(),
            ))
        },
    )?;
    config.default_frames = 1;
    Ok(Some(Box::new(BooguRequestScope {
        core: candle_gen::request_scope::CandleRequestScopeCore::new(config),
        provider,
        mode: context.mode.clone(),
        overlay: context.overlay.clone(),
        admission,
        token,
        finished: false,
    })))
}

struct BooguRequestScope {
    core: candle_gen::request_scope::CandleRequestScopeCore,
    provider: &'static str,
    mode: MemoryMode,
    overlay: Option<String>,
    admission: Option<AdmissionRegistry>,
    token: Option<u64>,
    finished: bool,
}

impl MemoryRequestScope for BooguRequestScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        if self.finished {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: memory request scope is finished",
                self.provider
            )));
        }
        let references = request.image_reference_count();
        match (Route::for_provider(self.provider)?, &self.mode) {
            (Route::Base | Route::Turbo, MemoryMode::TextToImage)
                if references == 0 && self.overlay.is_none() =>
            {
                request.video_mode = Some("t2i".into());
            }
            (Route::Base | Route::Turbo, MemoryMode::ImageToImage) if references == 1 => {
                let active = request
                    .conditioning
                    .iter()
                    .find_map(|conditioning| match conditioning {
                        gen_core::Conditioning::Reference { strength, .. } => Some(
                            strength
                                .or(request.strength)
                                .unwrap_or(crate::pipeline::DEFAULT_IMG2IMG_STRENGTH)
                                > 0.0,
                        ),
                        _ => None,
                    })
                    .unwrap_or(false);
                let expected = if active {
                    "reference_active"
                } else {
                    "reference_inert"
                };
                if self.overlay.as_deref() != Some(expected) {
                    return Err(gen_core::Error::Unsupported(format!(
                        "{}: reference strength crossed admitted {expected} identity",
                        self.provider
                    )));
                }
                request.video_mode = Some("i2i".into());
            }
            (Route::Edit, MemoryMode::Edit)
                if (1..=5).contains(&references) && self.overlay.is_none() =>
            {
                request.video_mode = Some("edit".into());
            }
            _ => {
                return Err(gen_core::Error::Unsupported(format!(
                    "{}: generation request crossed admitted mode/reference identity",
                    self.provider
                )))
            }
        }
        request.frames = Some(1);
        self.core.configure_request(request)?;
        if let (Some(admission), Some(token)) = (&self.admission, self.token) {
            admission.configure(token, request)?;
        }
        Ok(())
    }

    fn enter_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.core.enter_phase(phase)
    }
    fn leave_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.core.leave_phase(phase)
    }
    fn configure_decode(
        &mut self,
        edge: u32,
        overlap: u32,
        geometry: gen_core::MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.core.configure_decode(edge, overlap, geometry)
    }
    fn configure_attention(&mut self, chunk: u32) -> gen_core::Result<()> {
        self.core.configure_attention(chunk)
    }
    fn materialize_transformer_window(&mut self, first: u32, count: u32) -> gen_core::Result<()> {
        self.core.materialize_transformer_window(first, count)
    }
    fn finish(&mut self, outcome: gen_core::MemoryRunOutcome) -> gen_core::Result<()> {
        self.core.finish(outcome)?;
        if let (Some(admission), Some(token)) = (&self.admission, self.token) {
            admission.finish(token)?;
        }
        self.finished = true;
        Ok(())
    }
}

impl Drop for BooguRequestScope {
    fn drop(&mut self) {
        if !self.finished {
            if let (Some(admission), Some(token)) = (&self.admission, self.token) {
                admission.abandon(token);
            }
            self.finished = true;
        }
    }
}

pub(crate) fn begin_request(
    provider: &'static str,
    contract: &MemoryProviderContract,
    numeric: MemoryNumericTier,
    device: Device,
    context: &MemoryRunContext,
    admission: AdmissionRegistry,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_with_device(
        provider,
        contract,
        numeric,
        device,
        context,
        Some(admission),
    )
}

fn registered_numeric_tier(
    provider: &str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
) -> gen_core::Result<MemoryNumericTier> {
    if contract.asset_facts == MemoryAssetFacts::default() {
        validate_load_spec(provider, spec)?;
        let WeightsSource::Dir(root) = &spec.weights else {
            unreachable!()
        };
        let route = Route::for_provider(provider)?;
        if spec.resolved_route.as_deref() != Some(provider)
            || !canonical_artifact_path(root, route, spec.quantize)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{provider}: weights-free behavior crossed canonical route/revision/tier"
            )));
        }
        let expected = weights_free_contract(provider, spec)?;
        if expected != *contract {
            return Err(gen_core::Error::Unsupported(format!(
                "{provider}: caller contract differs from the sealed registry witness"
            )));
        }
        Ok(tier(spec.quantize))
    } else {
        let prepared = PreparedMemory::prepare(provider, spec)?;
        if prepared.contract != *contract {
            return Err(gen_core::Error::Unsupported(format!(
                "{provider}: caller contract differs from the sealed artifact receipt"
            )));
        }
        Ok(tier(prepared.receipt.tier))
    }
}

pub fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match registered_numeric_tier(&contract.provider_id, spec, contract) {
        Ok(numeric) => safety_check(&contract.provider_id, contract, numeric, context),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

fn weights_free_spec(provider: &str, spec: &LoadSpec) -> gen_core::Result<LoadSpec> {
    validate_load_spec(provider, spec)?;
    let route = Route::for_provider(provider)?;
    let variant = match spec.quantize {
        None => format!("{}-bf16", route.variant()),
        Some(Quant::Q4) => format!("{}-q4", route.variant()),
        Some(Quant::Q8) => route.variant().to_owned(),
        Some(other) => {
            return Err(gen_core::Error::Unsupported(format!(
                "unsupported tier {other:?}"
            )))
        }
    };
    let mut exact = spec.clone();
    exact.weights = WeightsSource::Dir(
        PathBuf::from("models--SceneWorks--boogu-image-mlx")
            .join("snapshots")
            .join(CANONICAL_REVISION)
            .join(variant),
    );
    exact.resolved_route = Some(provider.to_owned());
    Ok(exact)
}

pub fn weights_free_base(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    weights_free_contract(BOOGU_IMAGE_ID, spec)
}
pub fn weights_free_turbo(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    weights_free_contract(BOOGU_IMAGE_TURBO_ID, spec)
}
pub fn weights_free_edit(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    weights_free_contract(BOOGU_IMAGE_EDIT_ID, spec)
}
pub fn registered_base(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    provider_contract(BOOGU_IMAGE_ID, spec)
}
pub fn registered_turbo(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    provider_contract(BOOGU_IMAGE_TURBO_ID, spec)
}
pub fn registered_edit(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    provider_contract(BOOGU_IMAGE_EDIT_ID, spec)
}

pub fn valid_fixtures(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<MemoryBehaviorFixture>> {
    if strategy != MemoryStrategy::StagedResidency {
        return Ok(Vec::new());
    }
    let provider = contract.provider_id.as_str();
    let route = Route::for_provider(provider)?;
    let routes: Vec<(MemoryMode, u32, Option<String>)> = match route {
        Route::Base | Route::Turbo => vec![
            (MemoryMode::TextToImage, 0, None),
            (MemoryMode::ImageToImage, 1, Some("reference_inert".into())),
            (MemoryMode::ImageToImage, 1, Some("reference_active".into())),
        ],
        Route::Edit => (1..=5).map(|n| (MemoryMode::Edit, n, None)).collect(),
    };
    routes
        .into_iter()
        .map(|(mode, reference_count, overlay)| {
            let context = estimated_behavior_context(
                contract,
                strategy,
                tier(spec.quantize),
                MemoryBehaviorRoute {
                    mode,
                    reference_count,
                    use_pid: false,
                    has_phases: false,
                    overlay,
                },
            )?;
            let mut fixture = MemoryBehaviorFixture::new(context)
                .with_load_spec(weights_free_spec(provider, spec)?);
            fixture.request.video_mode = Some(
                match route {
                    Route::Base | Route::Turbo if reference_count == 0 => "t2i",
                    Route::Base | Route::Turbo => "i2i",
                    Route::Edit => "edit",
                }
                .into(),
            );
            for reference in &mut fixture.request.conditioning {
                if let gen_core::Conditioning::Reference { strength, .. } = reference {
                    *strength = Some(
                        if fixture.context.overlay.as_deref() == Some("reference_inert") {
                            0.0
                        } else {
                            1.0
                        },
                    );
                }
            }
            Ok(fixture)
        })
        .collect()
}

pub fn registered_begin(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let provider = match contract.provider_id.as_str() {
        BOOGU_IMAGE_ID => BOOGU_IMAGE_ID,
        BOOGU_IMAGE_TURBO_ID => BOOGU_IMAGE_TURBO_ID,
        BOOGU_IMAGE_EDIT_ID => BOOGU_IMAGE_EDIT_ID,
        other => {
            return Err(gen_core::Error::Unsupported(format!(
                "unknown Boogu route {other}"
            )))
        }
    };
    let numeric = registered_numeric_tier(provider, spec, contract)?;
    begin_with_device(provider, contract, numeric, Device::Cpu, context, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use candle_gen::candle_core::{
        safetensors as candle_safetensors, DType as CandleDType, Tensor,
    };

    /// A snapshot whose `transformer/config.json` carries the published Boogu-Image-0.1 axes the
    /// loader reads through [`config::BooguConfig::from_snapshot`]. `num_layers` is a parameter so
    /// a drifting snapshot can be exercised.
    fn architecture_spec(root: &Path, num_layers: u64) -> LoadSpec {
        std::fs::create_dir_all(root.join("transformer")).unwrap();
        std::fs::write(
            root.join("transformer").join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "patch_size": 2,
                "in_channels": 16,
                "out_channels": 16,
                "hidden_size": 3360,
                "num_layers": num_layers,
                "num_double_stream_layers": 8,
                "num_refiner_layers": 2,
                "num_attention_heads": 28,
                "num_kv_heads": 7,
                "axes_dim_rope": [40, 40, 40],
            }))
            .unwrap(),
        )
        .unwrap();
        LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
    }

    #[test]
    fn architecture_facts_match_the_loader_config_and_pass_conformance() {
        let fixture = tempfile::tempdir().unwrap();
        let spec = architecture_spec(fixture.path(), 40);
        let expected = gen_core::MemoryArchitectureFacts {
            attention_heads: Some(28),
            // hidden_size 3360 / 28 heads, published only because it divides exactly.
            head_dim: Some(120),
            // `num_layers` is the TOTAL trunk; the 8 `num_double_stream_layers` are its
            // double-stream prefix, already counted inside it.
            transformer_blocks: Some(40),
            patch_size: Some(2),
            // `VaeConfig::z_image()` — the FLUX.1 16-channel AutoencoderKL the loader constructs,
            // whose four `block_out_channels` stages give the x8 spatial scale.
            latent_channels: Some(16),
            vae_spatial_scale: Some(8),
            // Structurally absent: an image VAE has no temporal axis at all.
            vae_temporal_scale: None,
            activation_dtype_width: Some(2),
        };
        for provider in [BOOGU_IMAGE_ID, BOOGU_IMAGE_TURBO_ID, BOOGU_IMAGE_EDIT_ID] {
            let contract = build_contract(provider, &spec, true, MemoryAssetFacts::default());
            assert_eq!(contract.architecture_facts, expected, "{provider}");
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }

        // The axes are READ, not asserted: a snapshot declaring a different trunk publishes it.
        let drifted = tempfile::tempdir().unwrap();
        assert_eq!(
            build_contract(
                BOOGU_IMAGE_ID,
                &architecture_spec(drifted.path(), 32),
                true,
                MemoryAssetFacts::default(),
            )
            .architecture_facts
            .transformer_blocks,
            Some(32)
        );

        // The registry's contract surface names a sentinel that is not on disk: nothing about the
        // pipeline is resolved there, so every axis stays undeclared.
        let surface = LoadSpec::new(WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        ));
        assert!(
            build_contract(BOOGU_IMAGE_ID, &surface, true, MemoryAssetFacts::default())
                .architecture_facts
                .is_empty()
        );
    }

    fn write_tensors(path: &Path, tensors: Vec<(&str, Tensor)>) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let tensors = tensors
            .into_iter()
            .map(|(name, tensor)| (name.to_owned(), tensor))
            .collect::<HashMap<_, _>>();
        candle_safetensors::save(&tensors, path).unwrap();
    }

    fn artifact(provider: &str, quant: Option<Quant>) -> (tempfile::TempDir, PathBuf) {
        let temp = tempfile::tempdir().unwrap();
        let route = Route::for_provider(provider).unwrap();
        let name = match quant {
            None => format!("{}-bf16", route.variant()),
            Some(Quant::Q4) => format!("{}-q4", route.variant()),
            Some(Quant::Q8) => route.variant().to_owned(),
            _ => unreachable!(),
        };
        let root = temp
            .path()
            .join("models--SceneWorks--boogu-image-mlx")
            .join("snapshots")
            .join(CANONICAL_REVISION)
            .join(name);
        for component in ["transformer", "mllm"] {
            let path = root.join(component).join("model.safetensors");
            match quant {
                None => write_tensors(
                    &path,
                    vec![(
                        "layer.weight",
                        Tensor::zeros((2, 32), CandleDType::BF16, &Device::Cpu).unwrap(),
                    )],
                ),
                Some(quant) => {
                    let lanes = if quant == Quant::Q4 { 4 } else { 8 };
                    write_tensors(
                        &path,
                        vec![
                            (
                                "layer.weight",
                                Tensor::zeros((2, lanes), CandleDType::U32, &Device::Cpu).unwrap(),
                            ),
                            (
                                "layer.scales",
                                Tensor::zeros((2, 1), CandleDType::BF16, &Device::Cpu).unwrap(),
                            ),
                            (
                                "layer.biases",
                                Tensor::zeros((2, 1), CandleDType::BF16, &Device::Cpu).unwrap(),
                            ),
                        ],
                    );
                }
            }
            let config = match quant {
                Some(quant) => {
                    serde_json::json!({"quantization": {"bits": quant.bits(), "group_size": GROUP_SIZE}})
                }
                None => serde_json::json!({"architectures": ["fixture"]}),
            };
            std::fs::write(
                root.join(component).join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
        }
        write_tensors(
            &root.join("vae/model.safetensors"),
            vec![
                (
                    "encoder.conv_in.weight",
                    Tensor::zeros((128, 3, 3, 3), CandleDType::F32, &Device::Cpu).unwrap(),
                ),
                (
                    "encoder.conv_out.weight",
                    Tensor::zeros((32, 128, 3, 3), CandleDType::F32, &Device::Cpu).unwrap(),
                ),
                (
                    "decoder.conv_in.weight",
                    Tensor::zeros((512, 16, 3, 3), CandleDType::F32, &Device::Cpu).unwrap(),
                ),
                (
                    "decoder.conv_out.weight",
                    Tensor::zeros((3, 128, 3, 3), CandleDType::F32, &Device::Cpu).unwrap(),
                ),
            ],
        );
        std::fs::write(
            root.join("vae/config.json"),
            serde_json::to_vec(&serde_json::json!({
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 16,
                "block_out_channels": [128, 256, 512, 512],
                "layers_per_block": 2,
                "scaling_factor": 0.3611,
                "shift_factor": 0.1159,
                "norm_num_groups": 32
            }))
            .unwrap(),
        )
        .unwrap();
        (temp, root)
    }

    fn exact_spec(provider: &str, root: PathBuf, quant: Option<Quant>) -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir(root)).with_resolved_route(provider);
        spec.quantize = quant;
        spec
    }

    #[test]
    fn canonical_variants_bind_route_and_tier() {
        for (provider, variant) in [
            (BOOGU_IMAGE_ID, "base"),
            (BOOGU_IMAGE_TURBO_ID, "turbo"),
            (BOOGU_IMAGE_EDIT_ID, "edit"),
        ] {
            for (quant, suffix) in [
                (None, "-bf16"),
                (Some(Quant::Q4), "-q4"),
                (Some(Quant::Q8), ""),
            ] {
                let root = PathBuf::from("models--SceneWorks--boogu-image-mlx")
                    .join("snapshots")
                    .join(CANONICAL_REVISION)
                    .join(format!("{variant}{suffix}"));
                assert!(canonical_artifact_path(
                    &root,
                    Route::for_provider(provider).unwrap(),
                    quant
                ));
                assert!(!canonical_artifact_path(
                    &root.join("descendant"),
                    Route::for_provider(provider).unwrap(),
                    quant
                ));
                let crossed_repo = PathBuf::from("models--Other--boogu-image-mlx")
                    .join("snapshots")
                    .join(CANONICAL_REVISION)
                    .join(format!("{variant}{suffix}"));
                assert!(!canonical_artifact_path(
                    &crossed_repo,
                    Route::for_provider(provider).unwrap(),
                    quant
                ));
                let crossed = PathBuf::from("models--SceneWorks--boogu-image-mlx")
                    .join("snapshots")
                    .join(CANONICAL_REVISION)
                    .join(format!("base{suffix}"));
                assert_eq!(
                    canonical_artifact_path(
                        &crossed,
                        Route::for_provider(provider).unwrap(),
                        quant
                    ),
                    provider == BOOGU_IMAGE_ID
                );
                let app_root = PathBuf::from("SceneWorks__boogu-image-mlx")
                    .join(CANONICAL_REVISION)
                    .join(format!("{variant}{suffix}"));
                assert!(canonical_artifact_path(
                    &app_root,
                    Route::for_provider(provider).unwrap(),
                    quant
                ));
                let spoofed = PathBuf::from("models--SceneWorks--boogu-image-mlx")
                    .join(CANONICAL_REVISION)
                    .join("unrelated")
                    .join(format!("{variant}{suffix}"));
                assert!(!canonical_artifact_path(
                    &spoofed,
                    Route::for_provider(provider).unwrap(),
                    quant
                ));
            }
        }
    }

    #[test]
    fn weights_free_witness_is_registry_only_and_exact() {
        for provider in [BOOGU_IMAGE_ID, BOOGU_IMAGE_TURBO_ID, BOOGU_IMAGE_EDIT_ID] {
            for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
                let mut spec = LoadSpec::new(WeightsSource::Dir("unused".into()));
                spec.quantize = quant;
                let contract = weights_free_contract(provider, &spec).unwrap();
                let exact = weights_free_spec(provider, &spec).unwrap();
                assert_eq!(
                    registered_numeric_tier(provider, &exact, &contract).unwrap(),
                    tier(quant)
                );
                assert!(registered_numeric_tier(provider, &spec, &contract).is_err());
                let mut crossed = contract.clone();
                crossed.lifecycle.synchronized_phase_release = false;
                assert!(registered_numeric_tier(provider, &exact, &crossed).is_err());
            }
        }
    }

    #[test]
    fn receipt_rejects_malformed_native_vae_and_same_shape_mutation() {
        let (_temp, root) = artifact(BOOGU_IMAGE_ID, None);
        let spec = exact_spec(BOOGU_IMAGE_ID, root.clone(), None);
        let receipt = ArtifactReceipt::capture(BOOGU_IMAGE_ID, &spec).unwrap();
        let config_path = root.join("transformer/config.json");
        let mut bytes = std::fs::read(&config_path).unwrap();
        let index = bytes.iter().position(|byte| *byte == b'f').unwrap();
        bytes[index] = b'g';
        std::fs::write(&config_path, bytes).unwrap();
        assert!(receipt.ensure_unchanged().is_err());

        let (_temp, root) = artifact(BOOGU_IMAGE_ID, None);
        let vae_config = root.join("vae/config.json");
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&vae_config).unwrap()).unwrap();
        value["latent_channels"] = 8.into();
        std::fs::write(&vae_config, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(
            ArtifactReceipt::capture(BOOGU_IMAGE_ID, &exact_spec(BOOGU_IMAGE_ID, root, None))
                .is_err()
        );

        let (_temp, root) = artifact(BOOGU_IMAGE_ID, None);
        write_tensors(
            &root.join("vae/model.safetensors"),
            vec![
                (
                    "encoder.conv_in.weight",
                    Tensor::zeros((128, 3, 3, 3), CandleDType::F16, &Device::Cpu).unwrap(),
                ),
                (
                    "encoder.conv_out.weight",
                    Tensor::zeros((32, 128, 3, 3), CandleDType::F32, &Device::Cpu).unwrap(),
                ),
                (
                    "decoder.conv_in.weight",
                    Tensor::zeros((512, 16, 3, 3), CandleDType::F32, &Device::Cpu).unwrap(),
                ),
                (
                    "decoder.conv_out.weight",
                    Tensor::zeros((3, 128, 3, 3), CandleDType::F32, &Device::Cpu).unwrap(),
                ),
            ],
        );
        assert!(
            ArtifactReceipt::capture(BOOGU_IMAGE_ID, &exact_spec(BOOGU_IMAGE_ID, root, None))
                .is_err()
        );
    }

    #[test]
    fn receipts_accept_every_route_and_real_bf16_q4_q8_geometry() {
        for provider in [BOOGU_IMAGE_ID, BOOGU_IMAGE_TURBO_ID, BOOGU_IMAGE_EDIT_ID] {
            for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
                let (_temp, root) = artifact(provider, quant);
                let receipt =
                    ArtifactReceipt::capture(provider, &exact_spec(provider, root.clone(), quant))
                        .unwrap();
                assert!(receipt.canonical);
                assert_eq!(receipt.tier, quant);
                assert!(receipt.facts.decoder_bytes > 400_000);
                assert!(receipt.facts.base_bytes > receipt.facts.decoder_bytes);
                // One network, one field (SC-22667): the conditioning field is the MLLM alone —
                // the VAE the reference encoder shares with decode is charged once, in
                // `decoder_bytes` — and the base total is exactly its own decomposition.
                // Mutation that fails this: folding `decoder_bytes` into `conditioning_bytes`
                // again (the shape under review), which moves the field off the MLLM projection.
                assert_eq!(
                    receipt.facts.conditioning_bytes,
                    projected_component(&root, "mllm", 2, provider == BOOGU_IMAGE_EDIT_ID).unwrap(),
                    "{provider} {quant:?}: conditioning must be the MLLM alone"
                );
                assert_eq!(
                    receipt.facts.base_bytes,
                    receipt.facts.conditioning_bytes
                        + receipt.facts.transformer_bytes
                        + receipt.facts.decoder_bytes
                );
                // (The shared `check_memory_contract_asset_facts` is not run here: the synthetic
                // artifact gives the MLLM and the transformer identical tensor bytes, which trips
                // its repeated-total rule for a reason that is the fixture's, not the provider's.)
                assert!(receipt
                    .inventory
                    .iter()
                    .all(|(_, _, digest)| digest.len() == 64));
            }
        }
    }

    #[test]
    fn receipt_rejects_crossed_tier_route_duplicates_nested_and_mutation() {
        let (_temp, root) = artifact(BOOGU_IMAGE_ID, Some(Quant::Q4));
        assert!(ArtifactReceipt::capture(
            BOOGU_IMAGE_ID,
            &exact_spec(BOOGU_IMAGE_ID, root.clone(), Some(Quant::Q8)),
        )
        .is_err());
        assert!(
            !ArtifactReceipt::capture(
                BOOGU_IMAGE_TURBO_ID,
                &exact_spec(BOOGU_IMAGE_TURBO_ID, root.clone(), Some(Quant::Q4)),
            )
            .unwrap()
            .canonical
        );

        let duplicate = root.join("transformer/duplicate.safetensors");
        write_tensors(
            &duplicate,
            vec![(
                "layer.weight",
                Tensor::zeros((2, 4), CandleDType::U32, &Device::Cpu).unwrap(),
            )],
        );
        assert!(ArtifactReceipt::capture(
            BOOGU_IMAGE_ID,
            &exact_spec(BOOGU_IMAGE_ID, root.clone(), Some(Quant::Q4)),
        )
        .is_err());
        std::fs::remove_file(duplicate).unwrap();

        std::fs::create_dir(root.join("vae/nested")).unwrap();
        assert!(ArtifactReceipt::capture(
            BOOGU_IMAGE_ID,
            &exact_spec(BOOGU_IMAGE_ID, root.clone(), Some(Quant::Q4)),
        )
        .is_err());
        std::fs::remove_dir(root.join("vae/nested")).unwrap();

        let receipt = ArtifactReceipt::capture(
            BOOGU_IMAGE_ID,
            &exact_spec(BOOGU_IMAGE_ID, root.clone(), Some(Quant::Q4)),
        )
        .unwrap();
        std::fs::write(root.join("transformer/config.json"), b"{}\n").unwrap();
        assert!(receipt.ensure_unchanged().is_err());

        let (_temp, root) = artifact(BOOGU_IMAGE_ID, Some(Quant::Q4));
        std::fs::write(root.join(".incomplete"), b"partial").unwrap();
        assert!(ArtifactReceipt::capture(
            BOOGU_IMAGE_ID,
            &exact_spec(BOOGU_IMAGE_ID, root, Some(Quant::Q4)),
        )
        .is_err());

        let (_temp, root) = artifact(BOOGU_IMAGE_ID, Some(Quant::Q4));
        write_tensors(
            &root.join("mllm/model.safetensors"),
            vec![
                (
                    "layer.weight",
                    Tensor::zeros((2, 4), CandleDType::U32, &Device::Cpu).unwrap(),
                ),
                (
                    "layer.scales",
                    Tensor::zeros((2, 1), CandleDType::BF16, &Device::Cpu).unwrap(),
                ),
            ],
        );
        assert!(ArtifactReceipt::capture(
            BOOGU_IMAGE_ID,
            &exact_spec(BOOGU_IMAGE_ID, root, Some(Quant::Q4)),
        )
        .is_err());

        let (_temp, root) = artifact(BOOGU_IMAGE_ID, Some(Quant::Q4));
        write_tensors(
            &root.join("mllm/model.safetensors"),
            vec![
                (
                    "layer.weight",
                    Tensor::zeros((2, 8), CandleDType::U32, &Device::Cpu).unwrap(),
                ),
                (
                    "layer.scales",
                    Tensor::zeros((2, 1), CandleDType::BF16, &Device::Cpu).unwrap(),
                ),
                (
                    "layer.biases",
                    Tensor::zeros((2, 1), CandleDType::BF16, &Device::Cpu).unwrap(),
                ),
            ],
        );
        assert!(ArtifactReceipt::capture(
            BOOGU_IMAGE_ID,
            &exact_spec(BOOGU_IMAGE_ID, root, Some(Quant::Q4)),
        )
        .is_err());

        let (_temp, root) = artifact(BOOGU_IMAGE_ID, None);
        std::fs::write(root.join("vae/model.safetensors"), b"not safetensors").unwrap();
        assert!(
            ArtifactReceipt::capture(BOOGU_IMAGE_ID, &exact_spec(BOOGU_IMAGE_ID, root, None),)
                .is_err()
        );
    }

    #[test]
    fn typed_scope_binds_strength_mode_geometry_and_cleanup() {
        let spec = LoadSpec::new(WeightsSource::Dir("unused".into()));
        let contract = weights_free_contract(BOOGU_IMAGE_ID, &spec).unwrap();
        let context = estimated_behavior_context(
            &contract,
            MemoryStrategy::StagedResidency,
            tier(None),
            gen_core::MemoryBehaviorRoute {
                mode: MemoryMode::ImageToImage,
                reference_count: 1,
                use_pid: false,
                has_phases: false,
                overlay: Some("reference_active".into()),
            },
        )
        .unwrap();
        let mut scope = begin_with_device(
            BOOGU_IMAGE_ID,
            &contract,
            tier(None),
            Device::Cpu,
            &context,
            None,
        )
        .unwrap()
        .unwrap();
        let image = gen_core::Image {
            width: 1,
            height: 1,
            pixels: vec![0, 0, 0],
        };
        let mut crossed = GenerationRequest {
            prompt: "x".into(),
            width: context.geometry.width,
            height: context.geometry.height,
            count: 1,
            conditioning: vec![gen_core::Conditioning::Reference {
                image: image.clone(),
                strength: Some(0.0),
            }],
            ..Default::default()
        };
        assert!(scope.configure_request(&mut crossed).is_err());

        let mut scope = begin_with_device(
            BOOGU_IMAGE_ID,
            &contract,
            tier(None),
            Device::Cpu,
            &context,
            None,
        )
        .unwrap()
        .unwrap();
        let mut exact = GenerationRequest {
            conditioning: vec![gen_core::Conditioning::Reference {
                image,
                strength: Some(0.5),
            }],
            ..crossed
        };
        scope.configure_request(&mut exact).unwrap();
        assert_eq!(exact.video_mode.as_deref(), Some("i2i"));
        assert_eq!(exact.frames, Some(1));
        assert!(exact.memory.is_some_and(|memory| memory.stage_residency));
        scope.finish(gen_core::MemoryRunOutcome::Complete).unwrap();
        assert!(scope.finish(gen_core::MemoryRunOutcome::Complete).is_err());
    }

    #[test]
    fn admission_binds_sampling_preview_and_generation_once() {
        let spec = LoadSpec::new(WeightsSource::Dir("unused".into()));
        let contract = weights_free_contract(BOOGU_IMAGE_ID, &spec).unwrap();
        let context = estimated_behavior_context(
            &contract,
            MemoryStrategy::StagedResidency,
            tier(None),
            gen_core::MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .unwrap();

        for crossed in 0..7 {
            let admission = AdmissionRegistry::new(BOOGU_IMAGE_ID);
            admission.approve(&context).unwrap();
            let mut scope = begin_with_device(
                BOOGU_IMAGE_ID,
                &contract,
                tier(None),
                Device::Cpu,
                &context,
                Some(admission.clone()),
            )
            .unwrap()
            .unwrap();
            let mut request = GenerationRequest {
                prompt: "bound".into(),
                width: 1024,
                height: 1024,
                count: 1,
                seed: Some(7),
                steps: Some(20),
                sampler: Some("euler".into()),
                scheduler: Some("simple".into()),
                guidance: Some(4.0),
                scheduler_shift: Some(1.15),
                ..Default::default()
            };
            scope.configure_request(&mut request).unwrap();
            match crossed {
                0 => request.seed = Some(8),
                1 => request.steps = Some(21),
                2 => request.sampler = Some("lcm".into()),
                3 => request.scheduler = Some("sgm_uniform".into()),
                4 => request.guidance = Some(4.5),
                5 => request.scheduler_shift = Some(1.2),
                6 => request.preview = gen_core::PreviewSink::new(|_| {}),
                _ => unreachable!(),
            }
            assert!(admission.consume_for_generate(&request).is_err());
            scope
                .finish(gen_core::MemoryRunOutcome::Error {
                    message: "crossed".into(),
                })
                .unwrap();
        }

        let admission = AdmissionRegistry::new(BOOGU_IMAGE_ID);
        admission.approve(&context).unwrap();
        let mut scope = begin_with_device(
            BOOGU_IMAGE_ID,
            &contract,
            tier(None),
            Device::Cpu,
            &context,
            Some(admission.clone()),
        )
        .unwrap()
        .unwrap();
        let mut request = GenerationRequest {
            prompt: "bound".into(),
            width: 1024,
            height: 1024,
            count: 1,
            seed: Some(7),
            steps: Some(20),
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        admission.consume_for_generate(&request).unwrap();
        assert!(admission.consume_for_generate(&request).is_err());
        scope.finish(gen_core::MemoryRunOutcome::Complete).unwrap();
    }

    #[test]
    fn provider_rejects_batch_and_direct_count_above_one() {
        let spec = LoadSpec::new(WeightsSource::Dir("unused".into()));
        let contract = weights_free_contract(BOOGU_IMAGE_ID, &spec).unwrap();
        let exact = weights_free_spec(BOOGU_IMAGE_ID, &spec).unwrap();
        let mut context = estimated_behavior_context(
            &contract,
            MemoryStrategy::StagedResidency,
            tier(None),
            gen_core::MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .unwrap();
        context.geometry.batch = 2;
        assert!(matches!(
            registered_safety_check(&exact, &contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
        let request = GenerationRequest {
            count: 2,
            frames: Some(1),
            video_mode: Some("t2i".into()),
            ..Default::default()
        };
        assert!(validate_generation_request(BOOGU_IMAGE_ID, &request).is_err());
    }

    #[test]
    fn registered_real_contract_must_equal_the_sealed_artifact_contract() {
        let (_temp, root) = artifact(BOOGU_IMAGE_ID, None);
        let spec = exact_spec(BOOGU_IMAGE_ID, root, None);
        let prepared = PreparedMemory::prepare(BOOGU_IMAGE_ID, &spec).unwrap();
        let mut crossed = prepared.contract.clone();
        crossed.lifecycle.synchronized_phase_release = false;
        assert!(registered_numeric_tier(BOOGU_IMAGE_ID, &spec, &crossed).is_err());
    }

    #[test]
    fn every_public_route_tier_accepts_only_its_exact_staged_identity() {
        for provider in [BOOGU_IMAGE_ID, BOOGU_IMAGE_TURBO_ID, BOOGU_IMAGE_EDIT_ID] {
            for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
                let mut common = LoadSpec::new(WeightsSource::Dir("unused".into()));
                common.quantize = quant;
                let contract = weights_free_contract(provider, &common).unwrap();
                let exact = weights_free_spec(provider, &common).unwrap();
                let routes: Vec<(MemoryMode, u32, Option<String>)> =
                    match Route::for_provider(provider).unwrap() {
                        Route::Base | Route::Turbo => vec![
                            (MemoryMode::TextToImage, 0, None),
                            (MemoryMode::ImageToImage, 1, Some("reference_inert".into())),
                            (MemoryMode::ImageToImage, 1, Some("reference_active".into())),
                        ],
                        Route::Edit => (1..=5)
                            .map(|count| (MemoryMode::Edit, count, None))
                            .collect(),
                    };
                for (mode, reference_count, overlay) in routes {
                    let context = estimated_behavior_context(
                        &contract,
                        MemoryStrategy::StagedResidency,
                        tier(quant),
                        gen_core::MemoryBehaviorRoute {
                            mode,
                            reference_count,
                            use_pid: false,
                            has_phases: false,
                            overlay,
                        },
                    )
                    .unwrap();
                    assert_eq!(
                        registered_safety_check(&exact, &contract, &context),
                        MemorySafetyDecision::Accept
                    );
                    for mutation in ["abi", "fingerprint", "shape"] {
                        let mut crossed = context.clone();
                        match mutation {
                            "abi" => crossed.calibration_abi = gen_core::MEMORY_CALIBRATION_ABI,
                            "fingerprint" => crossed.calibration_fingerprint = "forged".to_owned(),
                            "shape" => {
                                crossed.load_shape = match crossed.load_shape {
                                    gen_core::LoadShape::EagerMaterialization => {
                                        gen_core::LoadShape::DeferredMaterialization
                                    }
                                    gen_core::LoadShape::DeferredMaterialization => {
                                        gen_core::LoadShape::EagerMaterialization
                                    }
                                }
                            }
                            _ => unreachable!(),
                        }
                        assert!(matches!(
                            registered_safety_check(&exact, &contract, &crossed),
                            MemorySafetyDecision::Reject { .. }
                        ));
                    }
                    let mut crossed_tier = context.clone();
                    crossed_tier.selection.tier.quant = match quant {
                        None => Some(Quant::Q8),
                        _ => None,
                    };
                    assert!(matches!(
                        registered_safety_check(&exact, &contract, &crossed_tier),
                        MemorySafetyDecision::Reject { .. }
                    ));
                    let mut crossed_pid = context;
                    crossed_pid.use_pid = true;
                    assert!(matches!(
                        registered_safety_check(&exact, &contract, &crossed_pid),
                        MemorySafetyDecision::Reject { .. }
                    ));
                }
            }
        }
    }
}
