//! Sealed physical and request identity shared by the native Wan2.2 I2V providers.
//!
//! Backend crates own registration, safety hooks, and request-scoped execution. This module keeps
//! the deliberately identical MLX/Candle artifact ABI in one place: canonical repository/revision,
//! exact direct-file inventory, header-derived resident facts, and ordered adapter receipts.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::{
    AdapterKind, Conditioning, GenerationRequest, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryFormulaKind, MemoryFormulaVariable, MemoryGeometry,
    MemoryLifecycleCapabilities, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRunContext, MemorySafetyDecision, MemorySelection,
    MemoryStrategy, MemoryStrategyCapability, MemoryStrategyParameters, MemoryStrategySupport,
    MoeExpert, OffloadPolicy, Precision, Quant, ResidentRequestMemory, WeightsSource,
};

pub const RECEIPT_VERSION: &str = "wan-i2v-structural-v2";
pub const LIGHTNING_REPOSITORY: &str = "lightx2v/Wan2.2-Lightning";
pub const LIGHTNING_REVISION: &str = "18bccf8884ec0a078eed79785eb4ef13ea16ce1e";
pub const NATIVE_SCHEDULE: &str = "wan-flow-match-native";
pub const DECODE_TILE_EDGES: &[u32] = &[192, 256, 384];
pub const DECODE_OVERLAPS: &[u32] = &[64];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WanI2vRoute {
    Ti2v5b,
    I2v14b,
}

impl WanI2vRoute {
    pub fn for_provider(provider_id: &str) -> crate::Result<Self> {
        match provider_id {
            "wan2_2_ti2v_5b" => Ok(Self::Ti2v5b),
            "wan2_2_i2v_14b" => Ok(Self::I2v14b),
            _ => Err(crate::Error::Unsupported(format!(
                "{provider_id}: not a Wan I2V memory route"
            ))),
        }
    }

    pub fn provider_id(self) -> &'static str {
        match self {
            Self::Ti2v5b => "wan2_2_ti2v_5b",
            Self::I2v14b => "wan2_2_i2v_14b",
        }
    }

    pub fn public_geometries(self) -> &'static [(u32, u32)] {
        match self {
            Self::Ti2v5b => &[(832, 480), (1280, 704), (704, 1280)],
            Self::I2v14b => &[(832, 480), (480, 832), (1280, 720), (720, 1280)],
        }
    }

    fn accepts_rate(self, fps: u32, frames: u32) -> bool {
        match self {
            Self::Ti2v5b => match fps {
                16 => [65, 81, 97, 113, 129].contains(&frames),
                24 => [97, 121, 145, 169, 193].contains(&frames),
                _ => false,
            },
            Self::I2v14b => fps == 16 && [45, 61, 77].contains(&frames),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WanI2vBackend {
    Mlx,
    Candle,
}

impl WanI2vBackend {
    fn key(self) -> &'static str {
        match self {
            Self::Mlx => "mlx",
            Self::Candle => "candle",
        }
    }

    fn realization(self) -> MemoryBackendRealization {
        match self {
            Self::Mlx => MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: false,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            },
            Self::Candle => MemoryBackendRealization::CandleCuda {
                device_residency: true,
                host_backed_weights: false,
                host_to_device_block_materialization: false,
                block_materialization: crate::MemoryWindowMaterialization::DeviceFormatTransfer,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WanAdapterReceipt {
    pub ordinal: u32,
    pub kind: &'static str,
    pub scale_bits: u32,
    pub pass_scale_bits: Vec<u32>,
    pub expert: &'static str,
    pub digest: String,
    pub source_bytes: u64,
    pub persistent_bytes: u64,
}

#[derive(Clone, Debug)]
struct SealedFile {
    relative: String,
    absolute: PathBuf,
    pin: crate::PinnedWeightsFile,
    digest: String,
    resident_bytes: u64,
    tensor: bool,
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
    root: PathBuf,
    files: Vec<SealedFile>,
}

fn sha256_file(path: &Path) -> crate::Result<String> {
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

fn tier(spec: &LoadSpec, backend: WanI2vBackend) -> crate::Result<MemoryNumericTier> {
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

fn repository_policy(
    backend: WanI2vBackend,
    route: WanI2vRoute,
    tier: MemoryNumericTier,
) -> (&'static str, &'static str) {
    match (backend, route, tier.quant) {
        (WanI2vBackend::Mlx, WanI2vRoute::Ti2v5b, _) => (
            "SceneWorks/wan2.2-ti2v-5b-mlx",
            "bb1b055249614cf9d7cf4373fbdbc184b77dee88",
        ),
        (WanI2vBackend::Mlx, WanI2vRoute::I2v14b, _) => (
            "SceneWorks/wan2.2-i2v-a14b-mlx",
            "c6c786170031eccc3a1fac0f98f1ad4ff988271e",
        ),
        (WanI2vBackend::Candle, WanI2vRoute::Ti2v5b, Some(_)) => (
            "SceneWorks/wan2.2-ti2v-5b-candle",
            "9b173dc8660334a87a11e67de58939afe68f8cb2",
        ),
        (WanI2vBackend::Candle, WanI2vRoute::I2v14b, Some(_)) => (
            "SceneWorks/wan2.2-i2v-a14b-candle",
            "d01bf1ea995c01a5bc545cefb977a320c9cb9fd0",
        ),
        (WanI2vBackend::Candle, WanI2vRoute::Ti2v5b, None) => (
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "b8fff7315c768468a5333511427288870b2e9635",
        ),
        (WanI2vBackend::Candle, WanI2vRoute::I2v14b, None) => (
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "596658fd9ca6b7b71d5057529bbf319ecbc61d74",
        ),
    }
}

fn repo_dir(repository: &str) -> String {
    format!("models--{}", repository.replace('/', "--"))
}

fn repository_revision(
    root: &Path,
    repository: &str,
    expected_revision: &str,
    tier: MemoryNumericTier,
    backend: WanI2vBackend,
) -> crate::Result<String> {
    let canonical = std::fs::canonicalize(root)?;
    let parts = canonical
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let marker = repo_dir(repository);
    let Some(index) = parts.iter().position(|part| part == &marker) else {
        return Err(crate::Error::Unsupported(format!(
            "{}: weights are not from canonical repository {repository}",
            root.display()
        )));
    };
    if parts.get(index + 1).map(String::as_str) != Some("snapshots") {
        return Err(crate::Error::Unsupported(format!(
            "{}: canonical repository path is not a snapshot",
            root.display()
        )));
    }
    let revision = parts.get(index + 2).cloned().unwrap_or_default();
    if revision != expected_revision
        || revision.len() != 40
        || !revision
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(crate::Error::Unsupported(format!(
            "{}: revision does not match immutable {repository}@{expected_revision}",
            root.display()
        )));
    }
    let expected_suffix = match tier.quant {
        Some(Quant::Q4) => Some("q4"),
        Some(Quant::Q8) => Some("q8"),
        None if backend == WanI2vBackend::Mlx => Some("bf16"),
        None => None,
        _ => unreachable!("tier validated"),
    };
    let suffix = parts.get(index + 3).map(String::as_str);
    if suffix != expected_suffix
        || index + 3 + usize::from(expected_suffix.is_some()) != parts.len()
    {
        return Err(crate::Error::Unsupported(format!(
            "{}: snapshot suffix does not match selected {:?} tier",
            root.display(),
            tier.quant
        )));
    }
    Ok(revision)
}

fn walk_safetensors(root: &Path) -> crate::Result<Vec<PathBuf>> {
    fn walk(root: &Path, at: &Path, output: &mut Vec<PathBuf>) -> crate::Result<()> {
        for entry in std::fs::read_dir(at)? {
            let entry = entry?;
            let ty = entry.file_type()?;
            let path = entry.path();
            if ty.is_symlink() && std::fs::metadata(&path)?.is_dir() {
                return Err(crate::Error::Unsupported(format!(
                    "directory symlink is not permitted in Wan inventory: {}",
                    path.display()
                )));
            }
            if std::fs::metadata(&path)?.is_dir() {
                walk(root, &path, output)?;
            } else if path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
                output.push(path.strip_prefix(root).unwrap_or(&path).to_path_buf());
            }
        }
        Ok(())
    }
    let mut output = Vec::new();
    walk(root, root, &mut output)?;
    output.sort();
    Ok(output)
}

fn validate_mlx_inventory(root: &Path, route: WanI2vRoute) -> crate::Result<Vec<PathBuf>> {
    let expected: &[&str] = match route {
        WanI2vRoute::Ti2v5b => &[
            "model.safetensors",
            "t5_encoder.safetensors",
            "vae.safetensors",
        ],
        WanI2vRoute::I2v14b => &[
            "high_noise_model.safetensors",
            "low_noise_model.safetensors",
            "t5_encoder.safetensors",
            "vae.safetensors",
        ],
    };
    for required in ["config.json", "tokenizer.json"] {
        if !root.join(required).is_file() {
            return Err(crate::Error::Unsupported(format!(
                "{}: missing required direct file {required}",
                route.provider_id()
            )));
        }
    }
    let actual = walk_safetensors(root)?;
    let expected = expected.iter().map(PathBuf::from).collect::<Vec<_>>();
    if actual != expected {
        return Err(crate::Error::Unsupported(format!(
            "{}: incomplete or extra MLX tensor inventory (expected {expected:?}, got {actual:?})",
            route.provider_id()
        )));
    }
    Ok(actual)
}

fn validate_component_inventory(root: &Path, component: &str) -> crate::Result<Vec<PathBuf>> {
    let dir = root.join(component);
    if !dir.join("config.json").is_file() {
        return Err(crate::Error::Unsupported(format!(
            "Wan Candle inventory is missing {component}/config.json"
        )));
    }
    let actual = walk_safetensors(&dir)?;
    if actual.is_empty() {
        return Err(crate::Error::Unsupported(format!(
            "Wan Candle inventory has no tensors under {component}/"
        )));
    }
    let indexes = std::fs::read_dir(&dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| name.ends_with(".safetensors.index.json"))
        })
        .collect::<Vec<_>>();
    if indexes.len() > 1 {
        return Err(crate::Error::Unsupported(format!(
            "Wan Candle {component}/ has multiple shard indexes"
        )));
    }
    if let Some(index) = indexes.first() {
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(index)?).map_err(|error| {
                crate::Error::Unsupported(format!(
                    "invalid shard index {}: {error}",
                    index.display()
                ))
            })?;
        let expected = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                crate::Error::Unsupported(format!("invalid shard index {}", index.display()))
            })?
            .values()
            .filter_map(serde_json::Value::as_str)
            .map(PathBuf::from)
            .collect::<BTreeSet<_>>();
        let actual_set = actual.iter().cloned().collect::<BTreeSet<_>>();
        if expected != actual_set {
            return Err(crate::Error::Unsupported(format!(
                "Wan Candle {component}/ shard index does not cover its exact tensor inventory"
            )));
        }
    } else if actual.len() != 1 {
        return Err(crate::Error::Unsupported(format!(
            "Wan Candle {component}/ has multiple tensors without one authoritative shard index"
        )));
    }
    Ok(actual
        .into_iter()
        .map(|relative| PathBuf::from(component).join(relative))
        .collect())
}

fn validate_candle_inventory(root: &Path, route: WanI2vRoute) -> crate::Result<Vec<PathBuf>> {
    if !root.join("model_index.json").is_file() || !root.join("tokenizer/tokenizer.json").is_file()
    {
        return Err(crate::Error::Unsupported(format!(
            "{}: incomplete Candle model/tokenizer inventory",
            route.provider_id()
        )));
    }
    let components: &[&str] = match route {
        WanI2vRoute::Ti2v5b => &["text_encoder", "transformer", "vae"],
        WanI2vRoute::I2v14b => &["text_encoder", "transformer", "transformer_2", "vae"],
    };
    let mut inventory = Vec::new();
    for component in components {
        inventory.extend(validate_component_inventory(root, component)?);
    }
    let actual = walk_safetensors(root)?;
    inventory.sort();
    if actual != inventory {
        return Err(crate::Error::Unsupported(format!(
            "{}: incomplete or extra Candle tensor inventory",
            route.provider_id()
        )));
    }
    Ok(inventory)
}

fn structural_files(
    root: &Path,
    backend: WanI2vBackend,
    route: WanI2vRoute,
) -> crate::Result<Vec<PathBuf>> {
    let mut files = match backend {
        WanI2vBackend::Mlx => vec![
            PathBuf::from("config.json"),
            PathBuf::from("tokenizer.json"),
        ],
        WanI2vBackend::Candle => {
            let components: &[&str] = match route {
                WanI2vRoute::Ti2v5b => &["text_encoder", "transformer", "vae"],
                WanI2vRoute::I2v14b => &["text_encoder", "transformer", "transformer_2", "vae"],
            };
            let mut paths = vec![
                PathBuf::from("model_index.json"),
                PathBuf::from("tokenizer/tokenizer.json"),
            ];
            paths.extend(
                components
                    .iter()
                    .map(|component| PathBuf::from(component).join("config.json")),
            );
            paths
        }
    };
    if backend == WanI2vBackend::Candle {
        fn find_indexes(root: &Path, at: &Path, output: &mut Vec<PathBuf>) -> crate::Result<()> {
            for entry in std::fs::read_dir(at)? {
                let path = entry?.path();
                if std::fs::metadata(&path)?.is_dir() {
                    find_indexes(root, &path, output)?;
                } else if path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .is_some_and(|name| name.ends_with(".safetensors.index.json"))
                {
                    output.push(path.strip_prefix(root).unwrap_or(&path).to_path_buf());
                }
            }
            Ok(())
        }
        find_indexes(root, root, &mut files)?;
        if let Some(quant) = match root.file_name().and_then(|name| name.to_str()) {
            Some("q4") => Some(Quant::Q4),
            Some("q8") => Some(Quant::Q8),
            _ => None,
        } {
            let components: &[&str] = match route {
                WanI2vRoute::Ti2v5b => &["transformer"],
                WanI2vRoute::I2v14b => &["transformer", "transformer_2"],
            };
            for component in components {
                let marker = PathBuf::from(component).join("quantize_config.json");
                let value: serde_json::Value = serde_json::from_slice(&std::fs::read(
                    root.join(&marker),
                )?)
                .map_err(|error| {
                    crate::Error::Unsupported(format!(
                        "invalid Wan packed marker {}: {error}",
                        marker.display()
                    ))
                })?;
                let bits = value.get("bits").and_then(serde_json::Value::as_u64);
                let group = value
                    .get("quantization")
                    .and_then(|value| value.get("group_size"))
                    .and_then(serde_json::Value::as_u64);
                if bits != Some(quant.bits() as u64) || group != Some(64) {
                    return Err(crate::Error::Unsupported(format!(
                        "Wan packed marker {} does not match the selected Q{} group-64 tier",
                        marker.display(),
                        quant.bits()
                    )));
                }
                files.push(marker);
            }
        }
    }
    files.sort();
    files.dedup();
    if let Some(missing) = files.iter().find(|relative| !root.join(relative).is_file()) {
        return Err(crate::Error::Unsupported(format!(
            "missing Wan structural file {}",
            missing.display()
        )));
    }
    Ok(files)
}

fn component_for(route: WanI2vRoute, relative: &str) -> MemoryPhase {
    if relative.contains("t5_encoder") || relative.starts_with("text_encoder/") {
        MemoryPhase::Conditioning
    } else if relative.contains("vae") {
        MemoryPhase::Decode
    } else {
        let _ = route;
        MemoryPhase::Denoise
    }
}

fn tensor_elements(
    path: &Path,
    header: &crate::weightsmeta::SafetensorsTensorHeader,
) -> crate::Result<u64> {
    let elements = header.shape.iter().try_fold(1_u64, |total, &dimension| {
        total
            .checked_mul(u64::try_from(dimension).map_err(|_| {
                crate::Error::Msg(format!(
                    "{} tensor shape is not representable",
                    path.display()
                ))
            })?)
            .ok_or_else(|| crate::Error::Msg(format!("{} tensor shape overflows", path.display())))
    })?;
    let stored = elements
        .checked_mul(header.dtype.size() as u64)
        .ok_or_else(|| crate::Error::Msg(format!("{} tensor bytes overflow", path.display())))?;
    if stored != header.data_bytes || stored == 0 {
        return Err(crate::Error::Unsupported(format!(
            "{} tensor {} has crossed dtype/shape/data geometry",
            path.display(),
            header.name
        )));
    }
    Ok(elements)
}

fn projected_dense_bytes(path: &Path, resident_width: u64) -> crate::Result<u64> {
    let headers = crate::weightsmeta::safetensors_path_tensor_headers(path)?;
    if headers.is_empty() {
        return Err(crate::Error::Unsupported(format!(
            "{} contains no tensors",
            path.display()
        )));
    }
    headers.into_iter().try_fold(0_u64, |sum, header| {
        if !header.is_float()
            || header.name.ends_with(".scales")
            || header.name.ends_with(".biases")
        {
            return Err(crate::Error::Unsupported(format!(
                "{} dense tensor {} has unsupported packed/non-float storage",
                path.display(),
                header.name
            )));
        }
        let bytes = tensor_elements(path, &header)?
            .checked_mul(resident_width)
            .ok_or_else(|| crate::Error::Msg("Wan projected tensor bytes overflow".to_owned()))?;
        sum.checked_add(bytes)
            .ok_or_else(|| crate::Error::Msg("Wan tensor byte total overflow".to_owned()))
    })
}

fn stored_tensor_bytes(path: &Path) -> crate::Result<u64> {
    let headers = crate::weightsmeta::safetensors_path_tensor_headers(path)?;
    if headers.is_empty() {
        return Err(crate::Error::Unsupported(format!(
            "{} contains no tensors",
            path.display()
        )));
    }
    headers.into_iter().try_fold(0_u64, |total, header| {
        tensor_elements(path, &header).and_then(|_| {
            total
                .checked_add(header.data_bytes)
                .ok_or_else(|| crate::Error::Msg("Wan stored tensor bytes overflow".to_owned()))
        })
    })
}

fn packed_transformer_bytes(
    path: &Path,
    quant: Quant,
    backend: WanI2vBackend,
) -> crate::Result<u64> {
    const GROUP: usize = 64;
    let headers = crate::weightsmeta::safetensors_path_tensor_headers(path)?;
    let tensors = headers
        .into_iter()
        .map(|header| (header.name.clone(), header))
        .collect::<BTreeMap<_, _>>();
    let mut packed_bases = BTreeSet::new();
    let mut packed = 0_u64;
    for weight in tensors
        .values()
        .filter(|header| header.name.ends_with(".weight") && header.shape.len() == 2)
    {
        let base = weight.name.strip_suffix(".weight").expect("suffix checked");
        let scales = tensors.get(&format!("{base}.scales")).ok_or_else(|| {
            crate::Error::Unsupported(format!("{} packed {base} lacks scales", path.display()))
        })?;
        let biases = tensors.get(&format!("{base}.biases")).ok_or_else(|| {
            crate::Error::Unsupported(format!("{} packed {base} lacks biases", path.display()))
        })?;
        if weight.dtype != crate::weightsmeta::Dtype::U32
            || !scales.is_float()
            || !biases.is_float()
            || scales.shape != biases.shape
            || scales.shape.len() != 2
            || scales.shape.first() != weight.shape.first()
        {
            return Err(crate::Error::Unsupported(format!(
                "{} packed {base} has invalid typed geometry",
                path.display()
            )));
        }
        let input = scales.shape[1]
            .checked_mul(GROUP)
            .ok_or_else(|| crate::Error::Msg("Wan packed input overflow".to_owned()))?;
        let packed_bits = weight.shape[1]
            .checked_mul(32)
            .and_then(|bits| bits.checked_div(input))
            .ok_or_else(|| {
                crate::Error::Unsupported("Wan packed width is inconsistent".to_owned())
            })?;
        if input == 0 || packed_bits != quant.bits() as usize {
            return Err(crate::Error::Unsupported(format!(
                "{} packed {base} does not encode Q{} group-64",
                path.display(),
                quant.bits()
            )));
        }
        let elements = u64::try_from(weight.shape[0])
            .ok()
            .and_then(|rows| {
                u64::try_from(input)
                    .ok()
                    .and_then(|cols| rows.checked_mul(cols))
            })
            .ok_or_else(|| crate::Error::Msg("Wan packed element count overflow".to_owned()))?;
        if !elements.is_multiple_of(32) {
            return Err(crate::Error::Unsupported(
                "Wan packed element count is not block aligned".to_owned(),
            ));
        }
        let bytes = match backend {
            WanI2vBackend::Mlx => weight
                .data_bytes
                .checked_add(scales.data_bytes)
                .and_then(|sum| sum.checked_add(biases.data_bytes)),
            WanI2vBackend::Candle => elements
                .checked_div(32)
                .and_then(|blocks| blocks.checked_mul(if quant == Quant::Q4 { 20 } else { 34 })),
        }
        .ok_or_else(|| crate::Error::Msg("Wan packed resident bytes overflow".to_owned()))?;
        packed = packed
            .checked_add(bytes)
            .ok_or_else(|| crate::Error::Msg("Wan packed resident total overflow".to_owned()))?;
        packed_bases.insert(base.to_owned());
    }
    if packed_bases.is_empty() {
        return Err(crate::Error::Unsupported(format!(
            "{} has no packed transformer weights",
            path.display()
        )));
    }
    let dense = tensors.values().try_fold(0_u64, |total, header| {
        if header.name.ends_with(".scales") || header.name.ends_with(".biases") {
            let base = header
                .name
                .strip_suffix(".scales")
                .or_else(|| header.name.strip_suffix(".biases"))
                .expect("suffix checked");
            if !packed_bases.contains(base) {
                return Err(crate::Error::Unsupported(format!(
                    "{} has orphan packed leaf {}",
                    path.display(),
                    header.name
                )));
            }
            return Ok(total);
        }
        if header
            .name
            .strip_suffix(".weight")
            .is_some_and(|base| packed_bases.contains(base))
        {
            return Ok(total);
        }
        if !header.is_float() {
            return Err(crate::Error::Unsupported(format!(
                "{} has unaccounted non-float tensor {}",
                path.display(),
                header.name
            )));
        }
        let bytes = tensor_elements(path, header)?
            .checked_mul(2)
            .ok_or_else(|| crate::Error::Msg("Wan dense residual bytes overflow".to_owned()))?;
        total
            .checked_add(bytes)
            .ok_or_else(|| crate::Error::Msg("Wan dense residual total overflow".to_owned()))
    })?;
    packed
        .checked_add(dense)
        .ok_or_else(|| crate::Error::Msg("Wan transformer resident bytes overflow".to_owned()))
}

fn physical_resident_bytes(
    path: &Path,
    backend: WanI2vBackend,
    route: WanI2vRoute,
    tier: MemoryNumericTier,
    phase: MemoryPhase,
) -> crate::Result<u64> {
    match (phase, tier.quant) {
        (MemoryPhase::Denoise, Some(quant)) => packed_transformer_bytes(path, quant, backend),
        (MemoryPhase::Decode, _) => projected_dense_bytes(
            path,
            if backend == WanI2vBackend::Candle && route == WanI2vRoute::Ti2v5b {
                4
            } else {
                2
            },
        ),
        _ => projected_dense_bytes(path, 2),
    }
}

fn adapter_receipts(
    spec: &LoadSpec,
    route: WanI2vRoute,
    tier: MemoryNumericTier,
) -> crate::Result<(Vec<WanAdapterReceipt>, String, u64, Vec<SealedFile>)> {
    let packed = tier.quant.is_some();
    let mut receipts = Vec::new();
    let mut files = Vec::new();
    let mut identity = Sha256::new();
    identity.update(b"wan-adapters-v1");
    let mut overlay_bytes = 0_u64;
    for (ordinal, adapter) in spec.adapters.iter().enumerate() {
        if !adapter.scale.is_finite() || adapter.scale < 0.0 || adapter.pass_scales.is_some() {
            return Err(crate::Error::Unsupported(format!(
                "{}: adapter {ordinal} has unsupported scale/pass recipe",
                route.provider_id()
            )));
        }
        if route == WanI2vRoute::Ti2v5b && adapter.moe_expert.is_some() {
            return Err(crate::Error::Unsupported(
                "Wan TI2V-5B accepts shared adapters only".to_owned(),
            ));
        }
        let headers = crate::weightsmeta::safetensors_path_tensor_headers(&adapter.path)?;
        if headers.is_empty() {
            return Err(crate::Error::Unsupported(format!(
                "adapter {} is empty",
                adapter.path.display()
            )));
        }
        let has_lokr_keys = crate::weightsmeta::keys_contain_lokr(
            headers.iter().map(|header| header.name.as_str()),
        );
        let has_loha_keys = crate::weightsmeta::keys_contain_loha(
            headers.iter().map(|header| header.name.as_str()),
        );
        if packed && has_loha_keys {
            return Err(crate::Error::Unsupported(
                "LoHa cannot be admitted on a packed Wan q4/q8 tier; select bf16".to_owned(),
            ));
        }
        let source_bytes = headers.iter().try_fold(0_u64, |sum, header| {
            sum.checked_add(header.data_bytes)
                .ok_or_else(|| crate::Error::Msg("Wan adapter byte total overflow".to_owned()))
        })?;
        if source_bytes == 0 {
            return Err(crate::Error::Unsupported(
                "Wan adapter has zero tensor bytes".to_owned(),
            ));
        }
        let multiplicity = if route == WanI2vRoute::I2v14b && adapter.moe_expert.is_none() {
            2
        } else {
            1
        };
        let persistent_bytes = if packed {
            source_bytes
                .checked_mul(multiplicity)
                .ok_or_else(|| crate::Error::Msg("Wan adapter residency overflow".to_owned()))?
        } else {
            0
        };
        overlay_bytes = overlay_bytes
            .checked_add(persistent_bytes)
            .ok_or_else(|| crate::Error::Msg("Wan overlay byte total overflow".to_owned()))?;
        let absolute = std::path::absolute(&adapter.path)?;
        let pin = spec
            .prepared_file_pins()
            .get(&absolute)
            .cloned()
            .unwrap_or(crate::PinnedWeightsFile::pin(&adapter.path)?);
        pin.ensure_unchanged()?;
        let digest = sha256_file(&adapter.path)?;
        let kind = if lightning_role(&adapter.path).is_some_and(|(_, exact)| exact) {
            "lightning_lora"
        } else if adapter.kind == AdapterKind::Lokr || has_lokr_keys {
            "lokr"
        } else if has_loha_keys {
            // Wan's execution adapters classify third-party LyCORIS LoHa from `hada_*` tensor
            // keys because the shared AdapterKind predates LoHa and has no LoHa variant. The
            // physical receipt must use that same classification rather than relabeling an
            // accepted dense LoHa file as LoRA.
            "loha"
        } else {
            "lora"
        };
        let expert = match adapter.moe_expert {
            None => "shared",
            Some(MoeExpert::High) => "high",
            Some(MoeExpert::Low) => "low",
        };
        identity.update((ordinal as u32).to_le_bytes());
        identity.update(kind.as_bytes());
        identity.update(adapter.scale.to_bits().to_le_bytes());
        identity.update(expert.as_bytes());
        identity.update(digest.as_bytes());
        identity.update(source_bytes.to_le_bytes());
        identity.update(persistent_bytes.to_le_bytes());
        receipts.push(WanAdapterReceipt {
            ordinal: ordinal as u32,
            kind,
            scale_bits: adapter.scale.to_bits(),
            pass_scale_bits: Vec::new(),
            expert,
            digest: digest.clone(),
            source_bytes,
            persistent_bytes,
        });
        files.push(SealedFile {
            relative: format!("adapter:{ordinal}"),
            absolute,
            pin,
            digest,
            resident_bytes: source_bytes,
            tensor: true,
        });
    }
    validate_lightning_recipe(spec, route)?;
    Ok((
        receipts,
        format!("wan-adapters-v1:{:x}", identity.finalize()),
        overlay_bytes,
        files,
    ))
}

fn lightning_role(path: &Path) -> Option<(MoeExpert, bool)> {
    let canonical = std::fs::canonicalize(path).ok()?;
    let marker = repo_dir(LIGHTNING_REPOSITORY);
    let parts = canonical
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let index = parts.iter().position(|part| part == &marker)?;
    if parts.get(index + 1).map(String::as_str) != Some("snapshots")
        || parts.get(index + 2).map(String::as_str) != Some(LIGHTNING_REVISION)
    {
        return Some((MoeExpert::High, false));
    }
    let suffix = parts[index + 3..].join("/");
    if suffix == "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors" {
        Some((MoeExpert::High, true))
    } else if suffix == "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors" {
        Some((MoeExpert::Low, true))
    } else {
        Some((MoeExpert::High, false))
    }
}

fn validate_lightning_recipe(spec: &LoadSpec, route: WanI2vRoute) -> crate::Result<()> {
    let roles = spec
        .adapters
        .iter()
        .enumerate()
        .filter_map(|(index, adapter)| {
            lightning_role(&adapter.path).map(|role| (index, adapter, role))
        })
        .collect::<Vec<_>>();
    if route == WanI2vRoute::Ti2v5b && !roles.is_empty() {
        return Err(crate::Error::Unsupported(
            "TI2V-5B cannot use the A14B Lightning pair".to_owned(),
        ));
    }
    if roles.is_empty() {
        return Ok(());
    }
    let valid = roles.len() == 2
        && roles[0].0 == 0
        && roles[1].0 == 1
        && roles[0].2 == (MoeExpert::High, true)
        && roles[1].2 == (MoeExpert::Low, true)
        && roles[0].1.moe_expert == Some(MoeExpert::High)
        && roles[1].1.moe_expert == Some(MoeExpert::Low)
        && roles[0].1.kind == AdapterKind::Lora
        && roles[1].1.kind == AdapterKind::Lora
        && roles[0].1.scale.to_bits() == 1.0_f32.to_bits()
        && roles[1].1.scale.to_bits() == 1.0_f32.to_bits();
    if !valid {
        return Err(crate::Error::Unsupported(
            "Wan I2V Lightning must be the exact ordered high/low Seko-V1 pair at scale 1"
                .to_owned(),
        ));
    }
    Ok(())
}

fn validate_load_spec<'a>(spec: &'a LoadSpec, provider_id: &str) -> crate::Result<&'a Path> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(crate::Error::Unsupported(format!(
            "{provider_id}: directory weights required"
        )));
    };
    if spec.precision != Precision::Bf16
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
        || !spec.components.is_empty()
        || spec.resolved_route.as_deref() != Some(provider_id)
    {
        return Err(crate::Error::Unsupported(format!(
            "{provider_id}: load left the exact Wan I2V surface"
        )));
    }
    crate::reject_unknown_components(spec, &[], provider_id)?;
    Ok(root)
}

pub fn prepare_load_spec(
    spec: &mut LoadSpec,
    backend: WanI2vBackend,
    provider_id: &str,
) -> crate::Result<()> {
    let route = WanI2vRoute::for_provider(provider_id)?;
    let tier = tier(spec, backend)?;
    let root = validate_load_spec(spec, provider_id)?.to_path_buf();
    let (repository, revision) = repository_policy(backend, route, tier);
    repository_revision(&root, repository, revision, tier, backend)?;
    let inventory = match backend {
        WanI2vBackend::Mlx => validate_mlx_inventory(&root, route)?,
        WanI2vBackend::Candle => validate_candle_inventory(&root, route)?,
    };
    let mut paths = inventory
        .into_iter()
        .map(|relative| root.join(relative))
        .collect::<Vec<_>>();
    paths.extend(
        structural_files(&root, backend, route)?
            .into_iter()
            .map(|relative| root.join(relative)),
    );
    paths.extend(spec.adapters.iter().map(|adapter| adapter.path.clone()));
    spec.prepare_with_file_pins(
        paths
            .into_iter()
            .map(crate::PinnedWeightsFile::pin)
            .collect::<crate::Result<Vec<_>>>()?,
    )
}

impl PreparedWanI2vMemory {
    pub fn prepare(
        spec: &LoadSpec,
        backend: WanI2vBackend,
        provider_id: &str,
    ) -> crate::Result<Self> {
        let route = WanI2vRoute::for_provider(provider_id)?;
        let tier = tier(spec, backend)?;
        let root = validate_load_spec(spec, provider_id)?;
        spec.validate_prepared_file_pins()?;
        let (repository, expected_revision) = repository_policy(backend, route, tier);
        let revision = repository_revision(root, repository, expected_revision, tier, backend)?;
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
        let contract = contract(provider_id, backend, spec, facts);
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

fn contract(
    provider_id: &str,
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
        provider_id: provider_id.to_owned(),
        backend: backend.realization(),
        strategies: MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: if matches!(
                    strategy,
                    MemoryStrategy::Resident
                        | MemoryStrategy::StagedResidency
                        | MemoryStrategy::BoundedDecode
                ) {
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

fn reference(request: &GenerationRequest) -> crate::Result<&crate::Image> {
    if request.conditioning.len() != 1 {
        return Err(crate::Error::Unsupported(
            "Wan public I2V requires exactly one conditioning carrier".to_owned(),
        ));
    }
    match &request.conditioning[0] {
        Conditioning::Reference { image, strength }
            if strength.is_none_or(|value| value.to_bits() == 1.0_f32.to_bits()) =>
        {
            Ok(image)
        }
        _ => Err(crate::Error::Unsupported(
            "Wan public I2V requires one full-strength Reference".to_owned(),
        )),
    }
}

pub fn validate_request(
    prepared: &PreparedWanI2vMemory,
    request: &GenerationRequest,
) -> crate::Result<()> {
    let image = reference(request)?;
    let frames = request.frames.unwrap_or_default();
    let fps = request.fps.unwrap_or_default();
    let steps = request.steps.unwrap_or_default();
    let unsupported_svd = request.motion_bucket_id.is_some()
        || request.noise_aug_strength.is_some()
        || request.decode_chunk_size.is_some()
        || request.conditioning_fps.is_some();
    let has_lightning = prepared.adapters.first().is_some_and(|receipt| {
        receipt.kind == "lightning_lora"
            && receipt.expert == "high"
            && prepared
                .adapters
                .get(1)
                .is_some_and(|low| low.kind == "lightning_lora" && low.expert == "low")
    });
    let lightning_sampling = if prepared.route == WanI2vRoute::I2v14b
        && steps == 4
        && request
            .guidance
            .is_some_and(|value| value.to_bits() == 1.0_f32.to_bits())
    {
        has_lightning
    } else {
        !has_lightning
    };
    if request.video_mode.as_deref() != Some("image_to_video")
        || request.count != 1
        || !prepared
            .route
            .public_geometries()
            .contains(&(request.width, request.height))
        || !prepared.route.accepts_rate(fps, frames)
        || !(1..=100).contains(&steps)
        || request.seed.is_none()
        || request.audio.is_some()
        || request.phases.is_some()
        || request.scheduler.is_some()
        || request.scheduler_shift.is_some()
        || request.use_pid
        || request.enhance_prompt
        || request.use_uncensored_enhancer
        || unsupported_svd
        || !lightning_sampling
        || image.width == 0
        || image.height == 0
    {
        return Err(crate::Error::Unsupported(format!(
            "{}: request left the exact public one-Reference I2V envelope",
            prepared.route.provider_id()
        )));
    }
    Ok(())
}

fn update_optional_u32(hash: &mut Sha256, value: Option<u32>) {
    match value {
        Some(value) => {
            hash.update([1]);
            hash.update(value.to_le_bytes());
        }
        None => hash.update([0]),
    }
}

/// Stable digest of the complete selected execution carrier.
///
/// Keep every field explicit: `None` and `Some(0)` are distinct, and fields not currently supported
/// by Wan remain identity-bearing so a future contract cannot add one without silently reusing an
/// older receipt format.
fn selection_receipt_digest(selection: &MemorySelection) -> String {
    let parameters = selection.parameters;
    let mut hash = Sha256::new();
    hash.update(b"wan-i2v-selection-v1");
    hash.update([selection.strategy as u8]);
    update_optional_u32(&mut hash, parameters.decode_tile_edge);
    update_optional_u32(&mut hash, parameters.decode_overlap);
    update_optional_u32(&mut hash, parameters.attention_chunk_size);
    update_optional_u32(&mut hash, parameters.transformer_window_size);
    match parameters.transformer_window_component {
        Some(component) => {
            hash.update([1]);
            hash.update([component as u8]);
        }
        None => hash.update([0]),
    }
    format!("{:x}", hash.finalize())
}

fn validate_selection(
    prepared: &PreparedWanI2vMemory,
    selection: &MemorySelection,
) -> crate::Result<()> {
    if selection.tier != prepared.tier {
        return Err(crate::Error::Unsupported(format!(
            "{}: selected tier does not match the sealed physical tier",
            prepared.route.provider_id()
        )));
    }
    prepared.contract.validate_selection(selection)
}

fn request_evidence_revision_for_selection(
    prepared: &PreparedWanI2vMemory,
    request: &GenerationRequest,
    selection: &MemorySelection,
) -> crate::Result<String> {
    validate_request(prepared, request)?;
    validate_selection(prepared, selection)?;
    let image = reference(request)?;
    let selection_receipt = selection_receipt_digest(selection);
    let mut hash = Sha256::new();
    hash.update(RECEIPT_VERSION.as_bytes());
    hash.update(prepared.artifact_identity.as_bytes());
    hash.update(prepared.adapter_identity.as_bytes());
    hash.update(prepared.backend.key().as_bytes());
    hash.update(prepared.route.provider_id().as_bytes());
    hash.update(selection_receipt.as_bytes());
    hash.update(request.width.to_le_bytes());
    hash.update(request.height.to_le_bytes());
    hash.update(request.frames.unwrap().to_le_bytes());
    hash.update(request.fps.unwrap().to_le_bytes());
    hash.update(request.steps.unwrap().to_le_bytes());
    hash.update(request.seed.unwrap().to_le_bytes());
    hash.update(request.prompt.as_bytes());
    hash.update(request.negative_prompt.as_deref().unwrap_or("").as_bytes());
    hash.update(
        request
            .guidance
            .map(f32::to_bits)
            .unwrap_or_default()
            .to_le_bytes(),
    );
    hash.update(request.sampler.as_deref().unwrap_or("default").as_bytes());
    hash.update(NATIVE_SCHEDULE.as_bytes());
    hash.update(image.width.to_le_bytes());
    hash.update(image.height.to_le_bytes());
    hash.update(Sha256::digest(&image.pixels));
    Ok(format!(
        "{RECEIPT_VERSION}:{}:{selection_receipt}:{:x}",
        prepared.artifact_identity,
        hash.finalize()
    ))
}

pub fn validate_context(
    prepared: &PreparedWanI2vMemory,
    context: &MemoryRunContext,
) -> crate::Result<()> {
    prepared.ensure_unchanged()?;
    let geometry = context.geometry;
    let selection_receipt = selection_receipt_digest(&context.selection);
    let prefix = format!(
        "{RECEIPT_VERSION}:{}:{selection_receipt}:",
        prepared.artifact_identity
    );
    let rate_ok = prepared.route.accepts_rate(
        if prepared.route == WanI2vRoute::I2v14b {
            16
        } else {
            24
        },
        geometry.frames,
    ) || (prepared.route == WanI2vRoute::Ti2v5b
        && prepared.route.accepts_rate(16, geometry.frames));
    if !matches!(
        context.selection.strategy,
        MemoryStrategy::Resident | MemoryStrategy::StagedResidency | MemoryStrategy::BoundedDecode
    ) || context.mode.as_key() != "image_to_video"
        || geometry.batch != 1
        || geometry.reference_count != 1
        || !context.has_reference
        || !prepared
            .route
            .public_geometries()
            .contains(&(geometry.width, geometry.height))
        || !rate_ok
        || context.overlay.as_deref() != Some(prepared.adapter_identity.as_str())
        || context.use_pid
        || context.has_phases
        || !context.evidence_revision.starts_with(&prefix)
    {
        return Err(crate::Error::Unsupported(format!(
            "{}: crossed Wan I2V memory context",
            prepared.route.provider_id()
        )));
    }
    match crate::standard_memory_strategy_safety_check(
        &prepared.contract,
        context,
        Some(prepared.tier),
        None,
    ) {
        MemorySafetyDecision::Accept => Ok(()),
        MemorySafetyDecision::Reject { reason } => Err(crate::Error::Unsupported(reason)),
    }
}

/// Reconstruct the exact selection carried by an executing request and verify that no request
/// control was added, dropped, or relabeled after admission.
pub fn selection_from_request(
    prepared: &PreparedWanI2vMemory,
    request: &GenerationRequest,
) -> crate::Result<MemorySelection> {
    let memory = request.memory.ok_or_else(|| {
        crate::Error::Unsupported(format!(
            "{}: admitted Wan I2V request is missing its explicit memory carrier",
            prepared.route.provider_id()
        ))
    })?;
    let strategy = if memory.stream_transformer_blocks {
        MemoryStrategy::BoundedTransformerResidency
    } else if memory.chunk_attention {
        MemoryStrategy::BoundedAttention
    } else if memory.tile_vae_decode {
        MemoryStrategy::BoundedDecode
    } else if memory.stage_residency {
        MemoryStrategy::StagedResidency
    } else {
        MemoryStrategy::Resident
    };
    let selection = MemorySelection {
        strategy,
        parameters: MemoryStrategyParameters {
            decode_tile_edge: memory.decode_tile_edge,
            decode_overlap: memory.decode_overlap,
            attention_chunk_size: memory.attention_chunk_size,
            transformer_window_size: memory.transformer_window_size,
            transformer_window_component: memory.transformer_window_component,
        },
        tier: prepared.tier,
    };
    validate_selection(prepared, &selection)?;
    if prepared.contract.generation_memory(&selection) != Some(memory) {
        return Err(crate::Error::Unsupported(format!(
            "{}: request memory controls do not match the selected Wan I2V rung",
            prepared.route.provider_id()
        )));
    }
    Ok(selection)
}

/// Provider-facing receipt minting after admission has installed the selected memory carrier.
/// Retaining this two-argument surface keeps paired callers source-compatible while making the
/// request's exact carrier authoritative for strategy/parameter identity.
pub fn request_evidence_revision(
    prepared: &PreparedWanI2vMemory,
    request: &GenerationRequest,
) -> crate::Result<String> {
    let selection = selection_from_request(prepared, request)?;
    request_evidence_revision_for_selection(prepared, request, &selection)
}

/// Configure-boundary validation: the request must carry the same exact rung/parameters as the
/// admitted context, and the full selection-bound request receipt must still match byte-for-byte.
pub fn validate_request_evidence(
    prepared: &PreparedWanI2vMemory,
    request: &GenerationRequest,
    admitted_selection: &MemorySelection,
    admitted_evidence: &str,
) -> crate::Result<String> {
    let request_selection = selection_from_request(prepared, request)?;
    if request_selection != *admitted_selection {
        return Err(crate::Error::Unsupported(format!(
            "{}: request rung/parameters crossed after admission",
            prepared.route.provider_id()
        )));
    }
    let actual = request_evidence_revision_for_selection(prepared, request, admitted_selection)?;
    if actual != admitted_evidence {
        return Err(crate::Error::Unsupported(format!(
            "{}: request axes crossed after admission",
            prepared.route.provider_id()
        )));
    }
    Ok(actual)
}

pub fn geometry_from_request(request: &GenerationRequest) -> MemoryGeometry {
    MemoryGeometry {
        width: request.width,
        height: request.height,
        batch: request.count,
        frames: request.frames.unwrap_or_default(),
        reference_count: request.conditioning.len() as u32,
    }
}

pub fn load_policy_for_selection(selection: MemoryStrategy) -> OffloadPolicy {
    if selection == MemoryStrategy::StagedResidency || selection == MemoryStrategy::BoundedDecode {
        OffloadPolicy::Sequential
    } else {
        OffloadPolicy::Resident
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AdapterSpec, Image, MemoryStrategySupport};

    fn write_safetensors(path: &Path, tensors: &[(&str, &str, &[usize])]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut offset = 0_u64;
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for &(name, dtype, shape) in tensors {
            let width = match dtype {
                "F32" | "U32" => 4,
                "F16" | "BF16" => 2,
                _ => panic!("dtype"),
            };
            let bytes = shape.iter().product::<usize>() * width;
            header.insert(
                name.to_owned(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes as u64],
                }),
            );
            data.resize(data.len() + bytes, name.len() as u8);
            offset += bytes as u64;
        }
        let mut json = serde_json::to_vec(&header).unwrap();
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend(json);
        bytes.extend(data);
        std::fs::write(path, bytes).unwrap();
    }

    fn mlx_fixture(route: WanI2vRoute, quant: Option<Quant>) -> (tempfile::TempDir, LoadSpec) {
        let tmp = tempfile::tempdir().unwrap();
        let tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant,
            component_precision_floors: &[],
        };
        let (repository, revision) = repository_policy(WanI2vBackend::Mlx, route, tier);
        let tier_name = match quant {
            Some(Quant::Q4) => "q4",
            Some(Quant::Q8) => "q8",
            None => "bf16",
            _ => unreachable!(),
        };
        let root = tmp
            .path()
            .join(repo_dir(repository))
            .join("snapshots")
            .join(revision)
            .join(tier_name);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("config.json"),
            format!(
                "{{\"route\":\"{}\",\"tier\":\"{tier_name}\"}}",
                route.provider_id()
            ),
        )
        .unwrap();
        std::fs::write(root.join("tokenizer.json"), "{}").unwrap();
        let files: &[(&str, usize)] = match route {
            WanI2vRoute::Ti2v5b => &[
                ("model.safetensors", 11),
                ("t5_encoder.safetensors", 7),
                ("vae.safetensors", 5),
            ],
            WanI2vRoute::I2v14b => &[
                ("high_noise_model.safetensors", 13),
                ("low_noise_model.safetensors", 17),
                ("t5_encoder.safetensors", 7),
                ("vae.safetensors", 5),
            ],
        };
        for &(name, logical) in files {
            if quant.is_some() && name.contains("model") {
                let packed_columns = if quant == Some(Quant::Q4) { 8 } else { 16 };
                write_safetensors(
                    &root.join(name),
                    &[
                        ("proj.weight", "U32", &[logical, packed_columns]),
                        ("proj.scales", "BF16", &[logical, 1]),
                        ("proj.biases", "BF16", &[logical, 1]),
                    ],
                );
            } else {
                write_safetensors(&root.join(name), &[("weight", "BF16", &[logical, 64])]);
            }
        }
        let mut spec =
            LoadSpec::new(WeightsSource::Dir(root)).with_resolved_route(route.provider_id());
        spec.quantize = quant;
        prepare_load_spec(&mut spec, WanI2vBackend::Mlx, route.provider_id()).unwrap();
        (tmp, spec)
    }

    fn candle_fixture(route: WanI2vRoute, quant: Option<Quant>) -> (tempfile::TempDir, LoadSpec) {
        let tmp = tempfile::tempdir().unwrap();
        let tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant,
            component_precision_floors: &[],
        };
        let (repository, revision) = repository_policy(WanI2vBackend::Candle, route, tier);
        let mut root = tmp
            .path()
            .join(repo_dir(repository))
            .join("snapshots")
            .join(revision);
        if let Some(quant) = quant {
            root = root.join(if quant == Quant::Q4 { "q4" } else { "q8" });
        }
        std::fs::create_dir_all(root.join("tokenizer")).unwrap();
        std::fs::write(root.join("model_index.json"), "{}").unwrap();
        std::fs::write(root.join("tokenizer/tokenizer.json"), "{}").unwrap();
        let components: &[&str] = match route {
            WanI2vRoute::Ti2v5b => &["text_encoder", "transformer", "vae"],
            WanI2vRoute::I2v14b => &["text_encoder", "transformer", "transformer_2", "vae"],
        };
        for (index, component) in components.iter().enumerate() {
            std::fs::create_dir_all(root.join(component)).unwrap();
            std::fs::write(root.join(component).join("config.json"), "{}").unwrap();
            if let Some(quant) = quant.filter(|_| component.starts_with("transformer")) {
                let packed_columns = if quant == Quant::Q4 { 8 } else { 16 };
                write_safetensors(
                    &root.join(component).join("model.safetensors"),
                    &[
                        ("proj.weight", "U32", &[index + 2, packed_columns]),
                        ("proj.scales", "F32", &[index + 2, 1]),
                        ("proj.biases", "F32", &[index + 2, 1]),
                    ],
                );
                std::fs::write(
                    root.join(component).join("quantize_config.json"),
                    format!(
                        "{{\"bits\":{},\"quantization\":{{\"group_size\":64}}}}",
                        quant.bits()
                    ),
                )
                .unwrap();
            } else {
                let shape = if component.starts_with("transformer") {
                    vec![index + 2, 64]
                } else {
                    vec![index + 2, index + 3]
                };
                write_safetensors(
                    &root.join(component).join("model.safetensors"),
                    &[("weight", "F32", &shape)],
                );
            }
        }
        let mut spec =
            LoadSpec::new(WeightsSource::Dir(root)).with_resolved_route(route.provider_id());
        spec.quantize = quant;
        prepare_load_spec(&mut spec, WanI2vBackend::Candle, route.provider_id()).unwrap();
        (tmp, spec)
    }

    #[test]
    fn mlx_q4_q8_bf16_facts_are_nonzero_differentiated_and_rsd_only() {
        let mut totals = Vec::new();
        for quant in [Some(Quant::Q4), Some(Quant::Q8), None] {
            let (_tmp, spec) = mlx_fixture(WanI2vRoute::I2v14b, quant);
            let prepared =
                PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
            assert!(prepared.contract.asset_facts.conditioning_bytes > 0);
            assert!(prepared.contract.asset_facts.transformer_bytes > 0);
            assert!(prepared.contract.asset_facts.decoder_bytes > 0);
            totals.push(prepared.contract.asset_facts.base_bytes);
            for strategy in MemoryStrategy::ALL {
                assert_eq!(
                    prepared.contract.capability(strategy).unwrap().support,
                    if matches!(
                        strategy,
                        MemoryStrategy::Resident
                            | MemoryStrategy::StagedResidency
                            | MemoryStrategy::BoundedDecode
                    ) {
                        MemoryStrategySupport::Implemented
                    } else {
                        MemoryStrategySupport::Missing
                    }
                );
            }
        }
        assert!(totals[0] < totals[1] && totals[1] < totals[2], "{totals:?}");

        let (_tmp, mut prepacked) = mlx_fixture(WanI2vRoute::Ti2v5b, Some(Quant::Q4));
        prepacked.quantize = None;
        let prepared = PreparedWanI2vMemory::prepare(
            &prepacked,
            WanI2vBackend::Mlx,
            WanI2vRoute::Ti2v5b.provider_id(),
        )
        .expect("the production worker selects a prepacked MLX q4 directory with quantize=None");
        assert_eq!(prepared.tier.quant, Some(Quant::Q4));
    }

    #[test]
    fn candle_routes_and_dense_commits_are_sealed() {
        for route in [WanI2vRoute::Ti2v5b, WanI2vRoute::I2v14b] {
            let mut totals = Vec::new();
            for quant in [Some(Quant::Q4), Some(Quant::Q8), None] {
                let (_tmp, spec) = candle_fixture(route, quant);
                let prepared = PreparedWanI2vMemory::prepare(
                    &spec,
                    WanI2vBackend::Candle,
                    route.provider_id(),
                )
                .unwrap();
                assert_eq!(prepared.revision.len(), 40);
                assert!(prepared.contract.asset_facts.base_bytes > 0);
                totals.push(prepared.contract.asset_facts.base_bytes);
            }
            assert!(
                totals[0] < totals[1] && totals[1] < totals[2],
                "{route:?}: {totals:?}"
            );
        }
    }

    #[test]
    fn same_length_mutation_and_inventory_or_repository_crossing_fail_closed() {
        let (_tmp, spec) = mlx_fixture(WanI2vRoute::Ti2v5b, Some(Quant::Q4));
        let prepared =
            PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan2_2_ti2v_5b").unwrap();
        let root = match &spec.weights {
            WeightsSource::Dir(root) => root,
            _ => unreachable!(),
        };
        let path = root.join("model.safetensors");
        let mut bytes = std::fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        std::fs::write(&path, bytes).unwrap();
        assert!(prepared.ensure_unchanged().is_err());

        let (_tmp, spec) = mlx_fixture(WanI2vRoute::Ti2v5b, Some(Quant::Q4));
        let root = match &spec.weights {
            WeightsSource::Dir(root) => root,
            _ => unreachable!(),
        };
        write_safetensors(&root.join("extra.safetensors"), &[("x", "F32", &[1])]);
        assert!(
            PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan2_2_ti2v_5b").is_err()
        );

        let mut crossed = spec.clone();
        crossed.weights = WeightsSource::Dir(
            root.parent()
                .unwrap()
                .join("ffffffffffffffffffffffffffffffffffffffff/q4"),
        );
        assert!(
            PreparedWanI2vMemory::prepare(&crossed, WanI2vBackend::Mlx, "wan2_2_ti2v_5b").is_err()
        );
    }

    fn request(route: WanI2vRoute) -> GenerationRequest {
        let (width, height, frames, fps, steps, guidance) = match route {
            WanI2vRoute::Ti2v5b => (832, 480, 121, 24, 20, Some(5.0)),
            WanI2vRoute::I2v14b => (1280, 720, 77, 16, 40, None),
        };
        GenerationRequest {
            prompt: "animate the still".to_owned(),
            width,
            height,
            count: 1,
            seed: Some(17),
            steps: Some(steps),
            guidance,
            frames: Some(frames),
            fps: Some(fps),
            video_mode: Some("image_to_video".to_owned()),
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 16,
                    height: 16,
                    pixels: vec![7; 16 * 16 * 3],
                },
                strength: None,
            }],
            ..Default::default()
        }
    }

    fn selection(prepared: &PreparedWanI2vMemory, strategy: MemoryStrategy) -> MemorySelection {
        MemorySelection {
            strategy,
            parameters: if strategy == MemoryStrategy::BoundedDecode {
                MemoryStrategyParameters {
                    decode_tile_edge: Some(192),
                    decode_overlap: Some(64),
                    ..Default::default()
                }
            } else {
                Default::default()
            },
            tier: prepared.tier,
        }
    }

    fn context(
        prepared: &PreparedWanI2vMemory,
        request: &GenerationRequest,
        selection: MemorySelection,
        evidence_revision: String,
    ) -> MemoryRunContext {
        MemoryRunContext {
            selection,
            optimization_authority: if selection.strategy.is_optimized() {
                crate::MemoryOptimizationAuthority::Estimated
            } else {
                crate::MemoryOptimizationAuthority::Resident
            },
            calibration_abi: crate::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: String::new(),
            load_shape: prepared.contract.load_shape,
            mode: crate::MemoryMode::Other("image_to_video".to_owned()),
            has_reference: true,
            use_pid: false,
            has_phases: false,
            geometry: geometry_from_request(request),
            overlay: Some(prepared.adapter_identity.clone()),
            budget: crate::MemoryBudget {
                total_bytes: u64::MAX,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1,
            cache_state: crate::MemoryCacheState::Cold,
            evidence_revision,
        }
    }

    #[test]
    fn request_identity_binds_reference_geometry_rate_seed_schedule_and_singular_carrier() {
        let (_tmp, spec) = mlx_fixture(WanI2vRoute::Ti2v5b, Some(Quant::Q4));
        let prepared =
            PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan2_2_ti2v_5b").unwrap();
        let resident = selection(&prepared, MemoryStrategy::Resident);
        let baseline = request_evidence_revision_for_selection(
            &prepared,
            &request(WanI2vRoute::Ti2v5b),
            &resident,
        )
        .unwrap();
        let mut crossed = request(WanI2vRoute::Ti2v5b);
        crossed.seed = Some(18);
        assert_ne!(
            baseline,
            request_evidence_revision_for_selection(&prepared, &crossed, &resident).unwrap()
        );
        let mut crossed = request(WanI2vRoute::Ti2v5b);
        crossed.conditioning.push(crossed.conditioning[0].clone());
        assert!(request_evidence_revision_for_selection(&prepared, &crossed, &resident).is_err());
        for (width, height) in [(832, 480), (1280, 704), (704, 1280)] {
            let mut accepted = request(WanI2vRoute::Ti2v5b);
            (accepted.width, accepted.height) = (width, height);
            assert!(validate_request(&prepared, &accepted).is_ok());
        }
        let mut off_menu = request(WanI2vRoute::Ti2v5b);
        (off_menu.width, off_menu.height) = (480, 832);
        assert!(validate_request(&prepared, &off_menu).is_err());
    }

    #[test]
    fn selection_receipt_binds_strategy_and_every_parameter_field() {
        let tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        };
        let baseline = MemorySelection {
            strategy: MemoryStrategy::Resident,
            parameters: Default::default(),
            tier,
        };
        let mut variants = Vec::new();
        let mut changed = baseline;
        changed.strategy = MemoryStrategy::StagedResidency;
        variants.push(changed);
        let setters: &[fn(&mut MemoryStrategyParameters)] = &[
            |parameters| parameters.decode_tile_edge = Some(192),
            |parameters| parameters.decode_overlap = Some(64),
            |parameters| parameters.attention_chunk_size = Some(1024),
            |parameters| parameters.transformer_window_size = Some(2),
            |parameters| {
                parameters.transformer_window_component =
                    Some(crate::TransformerComponent::TextEncoder)
            },
        ];
        for setter in setters {
            let mut changed = baseline;
            setter(&mut changed.parameters);
            variants.push(changed);
        }
        let baseline_digest = selection_receipt_digest(&baseline);
        let digests = variants
            .iter()
            .map(selection_receipt_digest)
            .collect::<BTreeSet<_>>();
        assert_eq!(digests.len(), variants.len());
        assert!(digests.iter().all(|digest| digest != &baseline_digest));
    }

    #[test]
    fn provider_and_configure_boundaries_reject_crossed_strategy_or_decode_parameters() {
        let (_tmp, spec) = mlx_fixture(WanI2vRoute::Ti2v5b, Some(Quant::Q4));
        let prepared =
            PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan2_2_ti2v_5b").unwrap();
        let request = request(WanI2vRoute::Ti2v5b);
        let resident = selection(&prepared, MemoryStrategy::Resident);
        let resident_evidence =
            request_evidence_revision_for_selection(&prepared, &request, &resident).unwrap();
        let resident_context = context(&prepared, &request, resident, resident_evidence.clone());
        assert!(validate_context(&prepared, &resident_context).is_ok());

        let staged = selection(&prepared, MemoryStrategy::StagedResidency);
        let mut crossed_context = resident_context.clone();
        crossed_context.selection = staged;
        crossed_context.optimization_authority = crate::MemoryOptimizationAuthority::Estimated;
        assert!(validate_context(&prepared, &crossed_context).is_err());

        let staged_evidence =
            request_evidence_revision_for_selection(&prepared, &request, &staged).unwrap();
        let staged_context = context(&prepared, &request, staged, staged_evidence.clone());
        assert!(validate_context(&prepared, &staged_context).is_ok());
        let mut staged_request = request.clone();
        staged_request.memory = prepared.contract.generation_memory(&staged);
        assert_eq!(
            validate_request_evidence(&prepared, &staged_request, &staged, &staged_evidence,)
                .unwrap(),
            staged_evidence
        );

        let bounded = selection(&prepared, MemoryStrategy::BoundedDecode);
        let bounded_evidence =
            request_evidence_revision_for_selection(&prepared, &request, &bounded).unwrap();
        let mut crossed_parameters = bounded;
        crossed_parameters.parameters.decode_tile_edge = Some(256);
        let mut crossed_context = context(
            &prepared,
            &request,
            crossed_parameters,
            bounded_evidence.clone(),
        );
        assert!(validate_context(&prepared, &crossed_context).is_err());
        crossed_context.evidence_revision =
            request_evidence_revision_for_selection(&prepared, &request, &crossed_parameters)
                .unwrap();
        assert!(validate_context(&prepared, &crossed_context).is_ok());

        let mut crossed_request = request;
        crossed_request.memory = prepared.contract.generation_memory(&crossed_parameters);
        assert!(validate_request_evidence(
            &prepared,
            &crossed_request,
            &bounded,
            &bounded_evidence,
        )
        .is_err());
    }

    #[test]
    fn adapter_residency_is_dense_zero_packed_shared_twice_targeted_once_and_loha_refuses() {
        let make_adapter = |dir: &Path, name: &str, key: &str| {
            let path = dir.join(name);
            write_safetensors(&path, &[(key, "F32", &[2, 3])]);
            path
        };
        let (tmp, dense_fixture) = mlx_fixture(WanI2vRoute::I2v14b, None);
        let mut dense =
            LoadSpec::new(dense_fixture.weights.clone()).with_resolved_route("wan2_2_i2v_14b");
        let shared = make_adapter(tmp.path(), "shared.safetensors", "lora_A.weight");
        let high = make_adapter(tmp.path(), "high.safetensors", "lora_A.weight");
        dense.adapters = vec![
            AdapterSpec::new(shared.clone(), 0.5, AdapterKind::Lora),
            AdapterSpec::new(high.clone(), 0.75, AdapterKind::Lora)
                .with_moe_expert(MoeExpert::High),
        ];
        let packed_adapters = dense.adapters.clone();
        prepare_load_spec(&mut dense, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
        let prepared =
            PreparedWanI2vMemory::prepare(&dense, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
        assert_eq!(prepared.contract.asset_facts.overlay_bytes, 0);

        let (_tmp, packed_fixture) = mlx_fixture(WanI2vRoute::I2v14b, Some(Quant::Q4));
        let mut packed =
            LoadSpec::new(packed_fixture.weights.clone()).with_resolved_route("wan2_2_i2v_14b");
        packed.quantize = Some(Quant::Q4);
        packed.adapters = packed_adapters;
        prepare_load_spec(&mut packed, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
        let prepared =
            PreparedWanI2vMemory::prepare(&packed, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
        assert_eq!(
            prepared.adapters[0].persistent_bytes,
            prepared.adapters[0].source_bytes * 2
        );
        assert_eq!(
            prepared.adapters[1].persistent_bytes,
            prepared.adapters[1].source_bytes
        );

        let loha = make_adapter(tmp.path(), "loha.safetensors", "block.hada_w1_a");
        let (_dense_loha_tmp, dense_loha_fixture) = mlx_fixture(WanI2vRoute::I2v14b, None);
        let mut dense_loha =
            LoadSpec::new(dense_loha_fixture.weights.clone()).with_resolved_route("wan2_2_i2v_14b");
        dense_loha.adapters = vec![AdapterSpec::new(loha.clone(), 1.0, AdapterKind::Lora)];
        prepare_load_spec(&mut dense_loha, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
        let dense_loha =
            PreparedWanI2vMemory::prepare(&dense_loha, WanI2vBackend::Mlx, "wan2_2_i2v_14b")
                .unwrap();
        assert_eq!(dense_loha.adapters[0].kind, "loha");
        assert_eq!(dense_loha.adapters[0].persistent_bytes, 0);

        let (_tmp, packed_loha_fixture) = mlx_fixture(WanI2vRoute::I2v14b, Some(Quant::Q4));
        let mut packed_loha = LoadSpec::new(packed_loha_fixture.weights.clone())
            .with_resolved_route("wan2_2_i2v_14b");
        packed_loha.quantize = Some(Quant::Q4);
        packed_loha.adapters = vec![AdapterSpec::new(loha, 1.0, AdapterKind::Lora)];
        prepare_load_spec(&mut packed_loha, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
        assert!(
            PreparedWanI2vMemory::prepare(&packed_loha, WanI2vBackend::Mlx, "wan2_2_i2v_14b",)
                .is_err()
        );
    }

    #[test]
    fn lightning_recipe_requires_the_exact_repository_revision_order_and_request_schedule() {
        let (tmp, fixture) = mlx_fixture(WanI2vRoute::I2v14b, None);
        let lightning = tmp
            .path()
            .join(repo_dir(LIGHTNING_REPOSITORY))
            .join("snapshots")
            .join(LIGHTNING_REVISION)
            .join("Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1");
        std::fs::create_dir_all(&lightning).unwrap();
        let high = lightning.join("high_noise_model.safetensors");
        let low = lightning.join("low_noise_model.safetensors");
        write_safetensors(&high, &[("lora_A.weight", "F32", &[2, 3])]);
        write_safetensors(&low, &[("lora_A.weight", "F32", &[2, 3])]);
        let mut spec = LoadSpec::new(fixture.weights.clone()).with_resolved_route("wan2_2_i2v_14b");
        spec.adapters = vec![
            AdapterSpec::new(high, 1.0, AdapterKind::Lora).with_moe_expert(MoeExpert::High),
            AdapterSpec::new(low, 1.0, AdapterKind::Lora).with_moe_expert(MoeExpert::Low),
        ];
        prepare_load_spec(&mut spec, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
        let prepared =
            PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
        let mut lightning_request = request(WanI2vRoute::I2v14b);
        lightning_request.steps = Some(4);
        lightning_request.guidance = Some(1.0);
        assert!(validate_request(&prepared, &lightning_request).is_ok());

        let mut native = request(WanI2vRoute::I2v14b);
        assert!(validate_request(&prepared, &native).is_err());
        native.steps = Some(4);
        native.guidance = Some(1.1);
        assert!(validate_request(&prepared, &native).is_err());

        let (_tmp, fixture) = mlx_fixture(WanI2vRoute::I2v14b, None);
        let bogus_dir = tempfile::tempdir().unwrap();
        let bogus_high = bogus_dir.path().join("high_noise_model.safetensors");
        let bogus_low = bogus_dir.path().join("low_noise_model.safetensors");
        write_safetensors(&bogus_high, &[("lora_A.weight", "F32", &[2, 3])]);
        write_safetensors(&bogus_low, &[("lora_A.weight", "F32", &[2, 3])]);
        let mut crossed =
            LoadSpec::new(fixture.weights.clone()).with_resolved_route("wan2_2_i2v_14b");
        crossed.adapters = vec![
            AdapterSpec::new(bogus_high, 1.0, AdapterKind::Lora).with_moe_expert(MoeExpert::High),
            AdapterSpec::new(bogus_low, 1.0, AdapterKind::Lora).with_moe_expert(MoeExpert::Low),
        ];
        prepare_load_spec(&mut crossed, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
        let crossed =
            PreparedWanI2vMemory::prepare(&crossed, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
        assert!(validate_request(&crossed, &lightning_request).is_err());
    }
}
