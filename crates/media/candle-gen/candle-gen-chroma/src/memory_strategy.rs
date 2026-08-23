//! Exact request-scoped Candle/CUDA memory contract for Chroma1 HD/Base/Flash (SC-20788).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, AdapterKind, GenerationMemory, GenerationRequest, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryGeometry, MemoryLifecycleCapabilities, MemoryMode,
    MemoryNumericTier, MemoryParameterRanges, MemoryPhase, MemoryProviderContract,
    MemoryRequestScope, MemoryResidentComponent, MemoryRunContext, MemoryRunOutcome,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability, MemoryStrategySupport,
    MemoryWindowMaterialization, Precision, Quant, WeightsSource,
};
#[cfg(feature = "cuda")]
use candle_gen::gen_core::{
    MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryBudget, MemoryCacheState,
    MemoryOptimizationAuthority,
};
use sha2::{Digest, Sha256};

pub const PID_FLUX_REVISION: &str = "6d5c1f1049e863f1757f68fd81c6bc850a95609d";
pub const PID_GEMMA_REVISION: &str = "684c553b5b41a1c835989d89f62f585e6269a7de";
pub const PUBLIC_GEOMETRIES: &[(u32, u32)] = &[(720, 1280), (768, 768), (1024, 1024), (1280, 720)];
const ADAPTER_OVERLAY_PREFIX: &str = "chroma.adapters.ordered-additive.sha256:";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RouteIdentity {
    pub provider: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

pub(crate) const ROUTES: &[RouteIdentity] = &[
    RouteIdentity {
        provider: crate::CHROMA1_HD_ID,
        repository: "chroma1-hd-mlx",
        revision: "9d99afe1ebca67032476756bc70d4a7152bc1bd5",
    },
    RouteIdentity {
        provider: crate::CHROMA1_BASE_ID,
        repository: "chroma1-base-mlx",
        revision: "e7330dda29d00ffdeeb719b28e92ee74cff0884c",
    },
    RouteIdentity {
        provider: crate::CHROMA1_FLASH_ID,
        repository: "chroma1-flash-mlx",
        revision: "6a9cb6178709559461506bf247f708d0d1008d00",
    },
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FileReceipt {
    pub lexical_path: PathBuf,
    pub canonical_path: PathBuf,
    pub sha256: [u8; 32],
    pub tensor_count: usize,
    pub projected_resident_bytes: u64,
    pin: gen_core::PinnedWeightsFile,
}

impl FileReceipt {
    fn capture(
        spec: &LoadSpec,
        path: &Path,
        projection_bytes: impl Fn(
            &gen_core::weightsmeta::SafetensorsTensorHeader,
        ) -> gen_core::Result<u64>,
    ) -> gen_core::Result<Self> {
        let pin = prepared_or_current_pin(spec, path)?;
        let headers = pin.read_unchanged(|stable| {
            gen_core::weightsmeta::safetensors_path_tensor_headers(stable)
        })?;
        if headers.is_empty() {
            return Err(gen_core::Error::Unsupported(format!(
                "chroma: {} contains no safetensors tensors",
                path.display()
            )));
        }
        let projected_resident_bytes = headers.iter().try_fold(0_u64, |total, header| {
            total.checked_add(projection_bytes(header)?).ok_or_else(|| {
                gen_core::Error::Unsupported("chroma: projected tensor bytes overflow".into())
            })
        })?;
        if projected_resident_bytes == 0 {
            return Err(gen_core::Error::Unsupported(format!(
                "chroma: {} projects to zero resident bytes",
                path.display()
            )));
        }
        let sha256 = pin.read_unchanged(sha256_file)?;
        let receipt = Self {
            lexical_path: pin.loader_path().to_path_buf(),
            canonical_path: pin.canonical_target_path().to_path_buf(),
            sha256,
            tensor_count: headers.len(),
            projected_resident_bytes,
            pin,
        };
        receipt.ensure_unchanged()?;
        Ok(receipt)
    }

    fn ensure_unchanged(&self) -> gen_core::Result<()> {
        self.pin.ensure_unchanged()?;
        let digest = self.pin.read_unchanged(sha256_file)?;
        if digest != self.sha256 {
            return Err(gen_core::Error::Unsupported(format!(
                "chroma: safetensors digest changed after receipt for {}",
                self.lexical_path.display()
            )));
        }
        Ok(())
    }
}

fn prepared_or_current_pin(
    spec: &LoadSpec,
    path: &Path,
) -> gen_core::Result<gen_core::PinnedWeightsFile> {
    if spec.prepared_file_pins().is_prepared() {
        let absolute = std::path::absolute(path)?;
        spec.prepared_file_pins()
            .get(&absolute)
            .cloned()
            .ok_or_else(|| {
                gen_core::Error::Unsupported(format!(
                    "chroma: prepared load receipt is missing {}",
                    absolute.display()
                ))
            })
    } else {
        gen_core::PinnedWeightsFile::pin(path)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AdapterReceipt {
    pub order: usize,
    pub file: FileReceipt,
    pub kind: AdapterKind,
    pub scale_bits: u32,
    pub pass_scale_bits: Option<Vec<u32>>,
    pub target: &'static str,
    pub additive_resident_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PidStudentTier {
    Res2k,
    Res4k,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PidReceipt {
    pub tier: PidStudentTier,
    pub student: FileReceipt,
    pub gemma: Vec<FileReceipt>,
    pub gemma_tokenizer: gen_core::PinnedWeightsFile,
    pub projected_resident_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ChromaLoadReceipt {
    pub route: RouteIdentity,
    pub canonical_root: PathBuf,
    inventory: Vec<(PathBuf, gen_core::PinnedWeightsFile, [u8; 32])>,
    pub tier: Option<Quant>,
    pub group_size: Option<usize>,
    pub transformer_config: gen_core::PinnedWeightsFile,
    pub transformer_config_sha256: [u8; 32],
    pub transformer: Vec<FileReceipt>,
    pub text_encoder: Vec<FileReceipt>,
    pub vae: Vec<FileReceipt>,
    pub adapters: Vec<AdapterReceipt>,
    pub pid: Option<PidReceipt>,
    pub components: gen_core::PerComponentBytes,
}

impl ChromaLoadReceipt {
    pub(crate) fn capture(provider_id: &str, spec: &LoadSpec) -> gen_core::Result<Self> {
        validate_load_shape(provider_id, spec)?;
        let route = route_identity(provider_id)?;
        let WeightsSource::Dir(root) = &spec.weights else {
            unreachable!("validated above")
        };
        validate_snapshot_binding(root, route)?;
        let lexical_root = std::path::absolute(root)?;
        let canonical_root = std::fs::canonicalize(&lexical_root)?;
        let inventory_paths = recursive_artifact_files(&lexical_root)?;
        let mut inventory = Vec::with_capacity(inventory_paths.len());
        for path in inventory_paths {
            let pin = if spec.prepared_file_pins().is_prepared() {
                spec.prepared_file_pins()
                    .get(&path)
                    .cloned()
                    .ok_or_else(|| {
                        gen_core::Error::Unsupported(format!(
                            "{provider_id}: sealed artifact receipt is missing {}",
                            path.display()
                        ))
                    })?
            } else {
                gen_core::PinnedWeightsFile::pin(&path)?
            };
            let digest = pin.read_unchanged(sha256_file)?;
            inventory.push((path, pin, digest));
        }
        let tier_name = root
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                gen_core::Error::Unsupported(
                    "chroma: weights path has no UTF-8 tier component".into(),
                )
            })?;
        if !matches!(tier_name, "q4" | "q8" | "bf16") {
            return Err(gen_core::Error::Unsupported(format!(
                "{provider_id}: exact turnkey path must end in q4, q8, or bf16, got {}",
                root.display()
            )));
        }
        let transformer_paths = direct_safetensors_inventory(&root.join("transformer"))?;
        let text_paths = direct_safetensors_inventory(&root.join("text_encoder"))?;
        let vae_paths = direct_safetensors_inventory(&root.join("vae"))?;
        ensure_unique_tensor_names(&transformer_paths, "transformer")?;
        ensure_unique_tensor_names(&text_paths, "text_encoder")?;
        ensure_unique_tensor_names(&vae_paths, "vae")?;
        let config_path = root.join("transformer/config.json");
        let config_pin = prepared_or_current_pin(spec, &config_path)?;
        let config_sha256 = config_pin.read_unchanged(sha256_file)?;
        config_pin.ensure_unchanged()?;
        let packed = inspect_transformer_tier(&transformer_paths, &config_pin)?;
        let expected = match tier_name {
            "q4" => Some(Quant::Q4),
            "q8" => Some(Quant::Q8),
            "bf16" => None,
            _ => unreachable!(),
        };
        if packed.0 != expected {
            return Err(gen_core::Error::Unsupported(format!(
                "{provider_id}: path tier {tier_name} crosses physical transformer tier {:?}",
                packed.0
            )));
        }
        let transformer_headers = transformer_paths
            .iter()
            .map(|path| {
                Ok((
                    path.clone(),
                    gen_core::weightsmeta::safetensors_path_tensor_headers(path)?,
                ))
            })
            .collect::<gen_core::Result<Vec<_>>>()?;
        let transformer_by_name = transformer_headers
            .iter()
            .flat_map(|(_, headers)| headers.iter())
            .map(|header| (header.name.clone(), header.clone()))
            .collect::<BTreeMap<_, _>>();
        let packed_bases = transformer_by_name
            .keys()
            .filter_map(|name| name.strip_suffix(".scales").map(str::to_owned))
            .collect::<BTreeSet<_>>();
        let transformer = transformer_paths
            .iter()
            .map(|path| {
                FileReceipt::capture(spec, path, |header| {
                    projected_transformer_header(
                        header,
                        &transformer_by_name,
                        &packed_bases,
                        packed.1,
                    )
                })
            })
            .collect::<gen_core::Result<Vec<_>>>()?;
        let text_encoder = text_paths
            .iter()
            .map(|path| FileReceipt::capture(spec, path, bf16_projection))
            .collect::<gen_core::Result<Vec<_>>>()?;
        let vae = vae_paths
            .iter()
            .map(|path| FileReceipt::capture(spec, path, bf16_projection))
            .collect::<gen_core::Result<Vec<_>>>()?;
        let adapters = capture_adapters(spec)?;
        let pid = capture_pid(spec)?;
        let sum = |files: &[FileReceipt]| {
            files.iter().fold(0_u64, |total, file| {
                total.saturating_add(file.projected_resident_bytes)
            })
        };
        let components = gen_core::PerComponentBytes {
            text_encoder: sum(&text_encoder),
            dit: sum(&transformer),
            vae: sum(&vae),
        };
        let receipt = Self {
            route,
            canonical_root,
            inventory,
            tier: packed.0,
            group_size: packed.1,
            transformer_config: config_pin,
            transformer_config_sha256: config_sha256,
            transformer,
            text_encoder,
            vae,
            adapters,
            pid,
            components,
        };
        receipt.ensure_unchanged()?;
        Ok(receipt)
    }

    pub(crate) fn ensure_unchanged(&self) -> gen_core::Result<()> {
        let mut current = recursive_artifact_files(&self.canonical_root)?
            .into_iter()
            .map(std::fs::canonicalize)
            .collect::<std::io::Result<Vec<_>>>()?;
        current.sort();
        let expected = self
            .inventory
            .iter()
            .map(|(path, _, _)| std::fs::canonicalize(path))
            .collect::<std::io::Result<Vec<_>>>()?;
        let mut expected = expected;
        expected.sort();
        if current != expected {
            return Err(gen_core::Error::Unsupported(
                "chroma: artifact inventory changed after the immutable receipt was sealed".into(),
            ));
        }
        for (_, pin, digest) in &self.inventory {
            pin.ensure_unchanged()?;
            if pin.read_unchanged(sha256_file)? != *digest {
                return Err(gen_core::Error::Unsupported(
                    "chroma: artifact contents changed after the immutable receipt was sealed"
                        .into(),
                ));
            }
        }
        self.transformer_config.ensure_unchanged()?;
        let config_digest = self.transformer_config.read_unchanged(sha256_file)?;
        if config_digest != self.transformer_config_sha256 {
            return Err(gen_core::Error::Unsupported(
                "chroma: transformer config digest changed after receipt".into(),
            ));
        }
        for file in self
            .transformer
            .iter()
            .chain(&self.text_encoder)
            .chain(&self.vae)
        {
            file.ensure_unchanged()?;
        }
        for adapter in &self.adapters {
            adapter.file.ensure_unchanged()?;
        }
        if let Some(pid) = &self.pid {
            pid.student.ensure_unchanged()?;
            for file in &pid.gemma {
                file.ensure_unchanged()?;
            }
            pid.gemma_tokenizer.ensure_unchanged()?;
        }
        Ok(())
    }

    fn adapter_overlay_identity(&self) -> Option<String> {
        if self.adapters.is_empty() {
            return None;
        }
        let mut digest = Sha256::new();
        update_framed(&mut digest, self.route.provider.as_bytes());
        update_framed(&mut digest, self.route.repository.as_bytes());
        update_framed(&mut digest, self.route.revision.as_bytes());
        for adapter in &self.adapters {
            update_framed(&mut digest, &(adapter.order as u64).to_le_bytes());
            update_framed(
                &mut digest,
                match adapter.kind {
                    AdapterKind::Lora => b"lora",
                    AdapterKind::Lokr => b"lokr",
                },
            );
            update_framed(&mut digest, &adapter.scale_bits.to_le_bytes());
            update_framed(&mut digest, adapter.target.as_bytes());
            update_framed(&mut digest, &adapter.additive_resident_bytes.to_le_bytes());
            update_framed(
                &mut digest,
                adapter.file.lexical_path.as_os_str().as_encoded_bytes(),
            );
            update_framed(
                &mut digest,
                adapter.file.canonical_path.as_os_str().as_encoded_bytes(),
            );
            update_framed(&mut digest, &adapter.file.sha256);
        }
        let mut encoded = String::with_capacity(ADAPTER_OVERLAY_PREFIX.len() + 64);
        encoded.push_str(ADAPTER_OVERLAY_PREFIX);
        for byte in digest.finalize() {
            write!(&mut encoded, "{byte:02x}").expect("writing hexadecimal to String cannot fail");
        }
        Some(encoded)
    }
}

fn update_framed(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn recursive_artifact_files(root: &Path) -> gen_core::Result<Vec<PathBuf>> {
    fn visit(root: &Path, files: &mut Vec<PathBuf>) -> gen_core::Result<()> {
        for entry in std::fs::read_dir(root)? {
            let path = entry?.path();
            let metadata = std::fs::symlink_metadata(&path)?;
            if metadata.file_type().is_symlink() {
                let target = std::fs::metadata(&path)?;
                if target.is_file() {
                    files.push(std::path::absolute(path)?);
                    continue;
                }
                return Err(gen_core::Error::Unsupported(format!(
                    "chroma: artifact symlinks must resolve directly to files: {}",
                    path.display()
                )));
            }
            if metadata.is_dir() {
                visit(&path, files)?;
            } else if metadata.is_file() {
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
                        "chroma: incomplete/hidden artifact marker is forbidden: {}",
                        path.display()
                    )));
                }
                files.push(std::path::absolute(path)?);
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    visit(root, &mut files)?;
    files.sort();
    if files.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "chroma: empty artifact inventory".into(),
        ));
    }
    Ok(files)
}

fn sha256_file(path: &Path) -> gen_core::Result<[u8; 32]> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest.finalize().into())
}

fn direct_safetensors_inventory(dir: &Path) -> gen_core::Result<Vec<PathBuf>> {
    let mut direct = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|error| {
        gen_core::Error::Unsupported(format!(
            "chroma: cannot inventory {}: {error}",
            dir.display()
        ))
    })?;
    for entry in entries {
        let path = entry?.path();
        let ty = std::fs::symlink_metadata(&path)?.file_type();
        if ty.is_dir() {
            if contains_safetensors(&path)? {
                return Err(gen_core::Error::Unsupported(format!(
                    "chroma: nested safetensors are forbidden under {}",
                    path.display()
                )));
            }
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
            && !gen_core::weightsmeta::is_hidden_file(&path)
        {
            direct.push(path);
        }
    }
    direct.sort();
    if direct.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "chroma: {} contains no direct safetensors files",
            dir.display()
        )));
    }
    Ok(direct)
}

fn contains_safetensors(dir: &Path) -> gen_core::Result<bool> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            if contains_safetensors(&path)? {
                return Ok(true);
            }
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("safetensors") {
            return Ok(true);
        }
    }
    Ok(false)
}

fn ensure_unique_tensor_names(paths: &[PathBuf], component: &str) -> gen_core::Result<()> {
    let mut names = BTreeSet::new();
    for path in paths {
        for header in gen_core::weightsmeta::safetensors_path_tensor_headers(path)? {
            if !names.insert(header.name.clone()) {
                return Err(gen_core::Error::Unsupported(format!(
                    "chroma: duplicate {component} tensor {} across direct shards",
                    header.name
                )));
            }
        }
    }
    Ok(())
}

fn route_identity(provider_id: &str) -> gen_core::Result<RouteIdentity> {
    ROUTES
        .iter()
        .copied()
        .find(|route| route.provider == provider_id)
        .ok_or_else(|| {
            gen_core::Error::Unsupported(format!("unknown Chroma memory provider {provider_id}"))
        })
}

fn validate_load_shape(provider_id: &str, spec: &LoadSpec) -> gen_core::Result<()> {
    route_identity(provider_id)?;
    if spec.resolved_route.as_deref() != Some(provider_id) {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: exact resolved_route is required for Chroma memory admission"
        )));
    }
    if spec.precision != Precision::Bf16 || spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: turnkey q4/q8/bf16 all require precision=Bf16 and LoadSpec.quantize=None"
        )));
    }
    if !matches!(spec.weights, WeightsSource::Dir(_)) {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: Chroma requires a turnkey snapshot directory"
        )));
    }
    if spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
        || !spec.components.is_empty()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: unsupported control/identity/external-component composition"
        )));
    }
    Ok(())
}

fn validate_snapshot_binding(root: &Path, route: RouteIdentity) -> gen_core::Result<()> {
    let repo_hf = format!("models--SceneWorks--{}", route.repository);
    let repo_app = format!("SceneWorks__{}", route.repository);
    let components = root
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let Some(tier) = components.last().map(String::as_str) else {
        return Err(gen_core::Error::Unsupported(
            "chroma: empty weights path".into(),
        ));
    };
    let canonical_hf = [repo_hf.as_str(), "snapshots", route.revision, tier];
    let canonical_app = [repo_app.as_str(), route.revision, tier];
    let canonical_plain = [route.repository, route.revision, tier];
    if !(components.ends_with(&canonical_hf.map(str::to_owned))
        || components.ends_with(&canonical_app.map(str::to_owned))
        || components.ends_with(&canonical_plain.map(str::to_owned)))
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: weights path {} is not bound to SceneWorks/{}@{}",
            route.provider,
            root.display(),
            route.repository,
            route.revision
        )));
    }
    Ok(())
}

fn config_quant(config_pin: &gen_core::PinnedWeightsFile) -> gen_core::Result<Option<(u8, usize)>> {
    config_pin.read_unchanged(|path| {
        let value: serde_json::Value = serde_json::from_reader(std::fs::File::open(path)?)
            .map_err(|error| {
                gen_core::Error::Unsupported(format!(
                    "chroma: malformed transformer config {}: {error}",
                    path.display()
                ))
            })?;
        let Some(packed) = candle_gen::quant::PackedConfig::from_config(&value) else {
            return Ok(None);
        };
        let bits = u8::try_from(packed.bits).map_err(|_| {
            gen_core::Error::Unsupported("chroma: transformer quantization bits overflow".into())
        })?;
        let group = usize::try_from(packed.group_size).map_err(|_| {
            gen_core::Error::Unsupported("chroma: transformer group size overflow".into())
        })?;
        Ok(Some((bits, group)))
    })
}

fn inspect_transformer_tier(
    files: &[PathBuf],
    config: &gen_core::PinnedWeightsFile,
) -> gen_core::Result<(Option<Quant>, Option<usize>)> {
    use gen_core::weightsmeta::Dtype;
    let mut headers = BTreeMap::new();
    for file in files {
        for header in gen_core::weightsmeta::safetensors_path_tensor_headers(file)? {
            if headers.insert(header.name.clone(), header).is_some() {
                return Err(gen_core::Error::Unsupported(
                    "chroma: duplicate transformer tensor across shards".into(),
                ));
            }
        }
    }
    let packed_bases = headers
        .keys()
        .filter_map(|name| name.strip_suffix(".scales").map(str::to_owned))
        .collect::<BTreeSet<_>>();
    let config_packed = config_quant(config)?;
    if packed_bases.is_empty() {
        if config_packed.is_some() {
            return Err(gen_core::Error::Unsupported(
                "chroma: config declares packing but tensor headers are dense".into(),
            ));
        }
        if headers.values().any(|header| header.dtype != Dtype::BF16) {
            return Err(gen_core::Error::Unsupported(
                "chroma: dense turnkey transformer must contain BF16 tensors only".into(),
            ));
        }
        return Ok((None, None));
    }
    let Some((bits, group)) = config_packed else {
        return Err(gen_core::Error::Unsupported(
            "chroma: packed transformer has no matching config quantization block".into(),
        ));
    };
    if group != candle_gen::quant::MLX_GROUP_SIZE || !matches!(bits, 4 | 8) {
        return Err(gen_core::Error::Unsupported(format!(
            "chroma: unsupported packed transformer {bits}-bit/group-{group}"
        )));
    }
    for base in &packed_bases {
        let weight = headers.get(&format!("{base}.weight")).ok_or_else(|| {
            gen_core::Error::Unsupported(format!("chroma: {base}.scales is missing .weight"))
        })?;
        let scales = headers
            .get(&format!("{base}.scales"))
            .expect("base came from scales");
        let biases = headers.get(&format!("{base}.biases")).ok_or_else(|| {
            gen_core::Error::Unsupported(format!("chroma: packed {base} is missing .biases"))
        })?;
        if weight.dtype != Dtype::U32 || scales.dtype != Dtype::BF16 || biases.dtype != Dtype::BF16
        {
            return Err(gen_core::Error::Unsupported(format!(
                "chroma: packed {base} must use U32/BF16/BF16"
            )));
        }
        let [weight_rows, packed_columns] = weight.shape.as_slice() else {
            return Err(gen_core::Error::Unsupported(format!(
                "chroma: packed {base}.weight must be rank 2"
            )));
        };
        let [scale_rows, scale_columns] = scales.shape.as_slice() else {
            return Err(gen_core::Error::Unsupported(format!(
                "chroma: packed {base}.scales must be rank 2"
            )));
        };
        let input = scale_columns.checked_mul(group).ok_or_else(|| {
            gen_core::Error::Unsupported("chroma: packed input width overflow".into())
        })?;
        let encoded = packed_columns
            .checked_mul(32)
            .ok_or_else(|| gen_core::Error::Unsupported("chroma: packed width overflow".into()))?;
        let resolved = encoded
            .checked_div(input)
            .filter(|_| input > 0 && encoded.is_multiple_of(input));
        if weight_rows != scale_rows
            || resolved != Some(usize::from(bits))
            || scales.shape != biases.shape
        {
            return Err(gen_core::Error::Unsupported(format!(
                "chroma: packed {base} crosses config width or affine sidecar geometry"
            )));
        }
    }
    for (name, header) in &headers {
        let base = name
            .strip_suffix(".weight")
            .or_else(|| name.strip_suffix(".scales"))
            .or_else(|| name.strip_suffix(".biases"));
        if base.is_some_and(|base| packed_bases.contains(base)) {
            continue;
        }
        if header.dtype != Dtype::BF16 {
            return Err(gen_core::Error::Unsupported(format!(
                "chroma: non-packed transformer tensor {name} must be BF16"
            )));
        }
    }
    Ok((
        Some(if bits == 4 { Quant::Q4 } else { Quant::Q8 }),
        Some(group),
    ))
}

fn projected_transformer_header(
    header: &gen_core::weightsmeta::SafetensorsTensorHeader,
    by_name: &BTreeMap<String, gen_core::weightsmeta::SafetensorsTensorHeader>,
    packed_bases: &BTreeSet<String>,
    group_size: Option<usize>,
) -> gen_core::Result<u64> {
    if header
        .name
        .strip_suffix(".scales")
        .or_else(|| header.name.strip_suffix(".biases"))
        .is_some_and(|base| packed_bases.contains(base))
    {
        return Ok(0);
    }
    if let Some(base) = header
        .name
        .strip_suffix(".weight")
        .filter(|base| packed_bases.contains(*base))
    {
        let scales = by_name.get(&format!("{base}.scales")).ok_or_else(|| {
            gen_core::Error::Unsupported(format!("chroma: packed {base} lacks scales"))
        })?;
        let biases = by_name.get(&format!("{base}.biases")).ok_or_else(|| {
            gen_core::Error::Unsupported(format!("chroma: packed {base} lacks biases"))
        })?;
        return candle_gen::quant::mlx_packed_qtensor_resident_bytes(
            header,
            scales,
            biases,
            group_size.ok_or_else(|| {
                gen_core::Error::Unsupported("chroma: packed tensor lacks group size".into())
            })?,
        );
    }
    if header.dtype != gen_core::weightsmeta::Dtype::BF16 {
        return Err(gen_core::Error::Unsupported(format!(
            "chroma: dense transformer tensor {:?} must be BF16, got {:?}",
            header.name, header.dtype
        )));
    }
    header.materialized_bytes(4)
}

fn bf16_projection(
    header: &gen_core::weightsmeta::SafetensorsTensorHeader,
) -> gen_core::Result<u64> {
    if header.dtype != gen_core::weightsmeta::Dtype::BF16 {
        return Err(gen_core::Error::Unsupported(format!(
            "chroma: auxiliary tensor {:?} must be BF16, got {:?}",
            header.name, header.dtype
        )));
    }
    header.materialized_bytes(2)
}

fn f32_projection(
    header: &gen_core::weightsmeta::SafetensorsTensorHeader,
) -> gen_core::Result<u64> {
    use gen_core::weightsmeta::Dtype;
    if !matches!(header.dtype, Dtype::BF16 | Dtype::F16 | Dtype::F32) {
        return Err(gen_core::Error::Unsupported(format!(
            "chroma: overlay tensor {:?} has unsupported dtype {:?}",
            header.name, header.dtype
        )));
    }
    header.materialized_bytes(4)
}

fn capture_adapters(spec: &LoadSpec) -> gen_core::Result<Vec<AdapterReceipt>> {
    let mut lexical = BTreeSet::new();
    let mut canonical = BTreeSet::new();
    spec.adapters
        .iter()
        .enumerate()
        .map(|(order, adapter)| {
            if !adapter.scale.is_finite()
                || adapter.pass_scales.is_some()
                || adapter.moe_expert.is_some()
            {
                return Err(gen_core::Error::Unsupported(format!(
                    "chroma: adapter {order} requires finite uniform scale and no pass/MoE target"
                )));
            }
            let file = FileReceipt::capture(spec, &adapter.path, f32_projection)?;
            if !lexical.insert(file.lexical_path.clone())
                || !canonical.insert(file.canonical_path.clone())
            {
                return Err(gen_core::Error::Unsupported(
                    "chroma: duplicate lexical or canonical adapter source".into(),
                ));
            }
            Ok(AdapterReceipt {
                order,
                additive_resident_bytes: file.projected_resident_bytes,
                file,
                kind: adapter.kind,
                scale_bits: adapter.scale.to_bits(),
                pass_scale_bits: None,
                target: "chroma.transformer",
            })
        })
        .collect()
}

fn path_has_identity(path: &Path, repo: &str, revision: &str, trailing: &[&str]) -> bool {
    let hf = format!("models--SceneWorks--{repo}");
    let app = format!("SceneWorks__{repo}");
    let parts = path
        .components()
        .map(|part| part.as_os_str().to_string_lossy())
        .collect::<Vec<_>>();
    let mut hf_suffix = vec![hf.as_str(), "snapshots", revision];
    hf_suffix.extend_from_slice(trailing);
    let mut app_suffix = vec![app.as_str(), revision];
    app_suffix.extend_from_slice(trailing);
    let mut plain_suffix = vec![repo, revision];
    plain_suffix.extend_from_slice(trailing);
    let owned = parts.iter().map(|part| part.as_ref()).collect::<Vec<_>>();
    owned.ends_with(&hf_suffix) || owned.ends_with(&app_suffix) || owned.ends_with(&plain_suffix)
}

fn capture_pid(spec: &LoadSpec) -> gen_core::Result<Option<PidReceipt>> {
    let Some(pid) = &spec.pid else {
        return Ok(None);
    };
    let WeightsSource::File(student_path) = &pid.checkpoint else {
        return Err(gen_core::Error::Unsupported(
            "chroma: PiD Flux student must be one exact safetensors file".into(),
        ));
    };
    let WeightsSource::Dir(gemma_dir) = &pid.gemma else {
        return Err(gen_core::Error::Unsupported(
            "chroma: PiD Gemma must be one exact snapshot directory".into(),
        ));
    };
    let name = student_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if !path_has_identity(student_path, "pid-flux", PID_FLUX_REVISION, &[name])
        || !path_has_identity(gemma_dir, "gemma-2-2b-it", PID_GEMMA_REVISION, &[])
    {
        return Err(gen_core::Error::Unsupported(
            "chroma: PiD requires exact SceneWorks/pid-flux and Gemma immutable revisions".into(),
        ));
    }
    let tier = match name {
        "pid_flux_2k.safetensors" => PidStudentTier::Res2k,
        "pid_flux_2kto4k.safetensors" | "pid_flux_2kto4k_v1pt5.safetensors" => {
            PidStudentTier::Res4k
        }
        _ => {
            return Err(gen_core::Error::Unsupported(format!(
                "chroma: unsupported PiD Flux student filename {name:?}"
            )))
        }
    };
    let student = FileReceipt::capture(spec, student_path, f32_projection)?;
    let merged = gemma_dir.join("gemma-2-2b-it.safetensors");
    let gemma_paths = if merged.is_file() {
        vec![merged]
    } else {
        direct_safetensors_inventory(gemma_dir)?
    };
    ensure_unique_tensor_names(&gemma_paths, "PiD Gemma")?;
    let gemma = gemma_paths
        .iter()
        .map(|path| FileReceipt::capture(spec, path, f32_projection))
        .collect::<gen_core::Result<Vec<_>>>()?;
    let tokenizer = prepared_or_current_pin(spec, &gemma_dir.join("tokenizer.json"))?;
    tokenizer.ensure_unchanged()?;
    let projected_resident_bytes =
        student
            .projected_resident_bytes
            .saturating_add(gemma.iter().fold(0_u64, |total, file| {
                total.saturating_add(file.projected_resident_bytes)
            }));
    Ok(Some(PidReceipt {
        tier,
        student,
        gemma,
        gemma_tokenizer: tokenizer,
        projected_resident_bytes,
    }))
}

pub(crate) fn provider_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let receipt = ChromaLoadReceipt::capture(provider_id, spec)?;
    Ok(contract_from_receipt(provider_id, spec, &receipt))
}

pub(crate) fn contract_from_receipt(
    provider_id: &str,
    spec: &LoadSpec,
    receipt: &ChromaLoadReceipt,
) -> MemoryProviderContract {
    let mut overlays = Vec::new();
    let adapter_bytes = receipt.adapters.iter().fold(0_u64, |total, adapter| {
        total.saturating_add(adapter.additive_resident_bytes)
    });
    if adapter_bytes > 0 {
        overlays.push(MemoryResidentComponent {
            id: receipt
                .adapter_overlay_identity()
                .expect("non-empty adapter receipt has an exact overlay identity"),
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: adapter_bytes,
            bounded_by: Some(MemoryStrategy::StagedResidency),
            residency: MemoryComponentResidency::WholeRender,
        });
    }
    if let Some(pid) = &receipt.pid {
        overlays.push(MemoryResidentComponent {
            id: "chroma.pid.flux-student-and-gemma".into(),
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: pid.projected_resident_bytes,
            bounded_by: Some(MemoryStrategy::StagedResidency),
            residency: MemoryComponentResidency::WholeRender,
        });
    }
    build_contract(
        provider_id,
        spec,
        receipt.tier,
        receipt.components,
        overlays,
    )
}

#[cfg(test)]
pub(crate) fn uncalibrated_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    validate_load_shape(provider_id, spec)?;
    Ok(build_contract(
        provider_id,
        spec,
        physical_tier_hint(spec),
        Default::default(),
        Vec::new(),
    ))
}

pub(crate) fn weights_free_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    route_identity(provider_id)?;
    let tier = match spec.quantize {
        None => None,
        Some(Quant::Q4) => Some(Quant::Q4),
        Some(Quant::Q8) => Some(Quant::Q8),
        Some(Quant::Nvfp4) => {
            return Err(gen_core::Error::Unsupported(
                "Chroma has no NVFP4 turnkey".into(),
            ))
        }
    };
    let mut normalized = spec.clone();
    normalized.quantize = None;
    normalized.resolved_route = Some(provider_id.to_owned());
    Ok(build_contract(
        provider_id,
        &normalized,
        tier,
        Default::default(),
        Vec::new(),
    ))
}

fn physical_tier_hint(spec: &LoadSpec) -> Option<Quant> {
    if spec.quantize.is_some() {
        return spec.quantize;
    }
    match &spec.weights {
        WeightsSource::Dir(root)
            if root.file_name().and_then(|name| name.to_str()) == Some("q4") =>
        {
            Some(Quant::Q4)
        }
        WeightsSource::Dir(root)
            if root.file_name().and_then(|name| name.to_str()) == Some("q8") =>
        {
            Some(Quant::Q8)
        }
        _ => None,
    }
}

pub(crate) fn weights_free_surface_contract(
    provider_id: &str,
    surface: &gen_core::MemoryContractSurfaceSpec,
) -> gen_core::Result<MemoryProviderContract> {
    route_identity(provider_id)?;
    let mut spec = surface.spec.clone();
    let tier = match surface.resolved_artifact_tier() {
        gen_core::MemoryContractSurfaceTier::Bf16 => None,
        gen_core::MemoryContractSurfaceTier::Q4 => Some(Quant::Q4),
        gen_core::MemoryContractSurfaceTier::Q8 => Some(Quant::Q8),
        gen_core::MemoryContractSurfaceTier::Nvfp4 => {
            return Err(gen_core::Error::Unsupported(
                "Chroma has no NVFP4 turnkey".into(),
            ))
        }
    };
    spec.resolved_route = Some(provider_id.to_owned());
    spec.quantize = None;
    Ok(build_contract(
        provider_id,
        &spec,
        tier,
        Default::default(),
        Vec::new(),
    ))
}

fn build_contract(
    provider_id: &str,
    spec: &LoadSpec,
    _tier: Option<Quant>,
    components: gen_core::PerComponentBytes,
    overlays: Vec<MemoryResidentComponent>,
) -> MemoryProviderContract {
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let strategies = MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident | MemoryStrategy::StagedResidency => {
                    MemoryStrategySupport::Implemented
                }
                _ => MemoryStrategySupport::Missing,
            },
            parameters: MemoryParameterRanges::default(),
        })
        .collect();
    MemoryProviderContract {
        provider_id: provider_id.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
            // No transformer-window rung is published. This value describes the base snapshot's
            // already-device-format transfer and cannot authorize a missing capability.
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies,
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: gen_core::ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: false,
            attention_chunking: false,
            transformer_window_materialization: false,
        },
        formula: if overlays.is_empty() {
            MemoryFormulaKind::PhaseEnvelope {
                phases,
                variables: vec![
                    MemoryFormulaVariable::AssetBytes,
                    MemoryFormulaVariable::PixelCount,
                    MemoryFormulaVariable::BatchCount,
                    MemoryFormulaVariable::ConditioningTokenCount,
                    MemoryFormulaVariable::OverlayBytes,
                ],
            }
        } else {
            MemoryFormulaKind::ComponentPhaseEnvelope {
                phases,
                variables: vec![
                    MemoryFormulaVariable::AssetBytes,
                    MemoryFormulaVariable::PixelCount,
                    MemoryFormulaVariable::BatchCount,
                    MemoryFormulaVariable::ConditioningTokenCount,
                    MemoryFormulaVariable::OverlayBytes,
                ],
                resident_components: overlays.clone(),
            }
        },
        // Chroma has no promoted measured curve. These bounds are structural estimates, so an
        // invented calibration identity must never upgrade selector authority to Calibrated.
        calibration: None,
        asset_facts: MemoryAssetFacts {
            base_bytes: components
                .text_encoder
                .saturating_add(components.dit)
                .saturating_add(components.vae),
            conditioning_bytes: components.text_encoder,
            transformer_bytes: components.dit,
            decoder_bytes: components.vae,
            overlay_bytes: overlays.iter().fold(0_u64, |total, component| {
                total.saturating_add(component.resident_bytes)
            }),
        },
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    }
}

fn contract_adapter_overlay_identity(
    contract: &MemoryProviderContract,
) -> gen_core::Result<Option<&str>> {
    let mut identities = contract
        .resident_components()
        .iter()
        .filter_map(|component| {
            component
                .id
                .starts_with(ADAPTER_OVERLAY_PREFIX)
                .then_some(component.id.as_str())
        });
    let identity = identities.next();
    if identities.next().is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: Chroma contract has multiple adapter overlay identities",
            contract.provider_id
        )));
    }
    Ok(identity)
}

fn validate_route(
    provider_id: &str,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    if context.mode != MemoryMode::TextToImage
        || context.geometry.reference_count != 0
        || context.has_reference
        || context.has_phases
        || context.geometry.frames != 1
        || !matches!(context.geometry.batch, 1 | 2 | 4)
        || !PUBLIC_GEOMETRIES.contains(&(context.geometry.width, context.geometry.height))
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: Chroma memory admission is exact T2I, refs=0, count 1/2/4, and one of the four public geometries only"
        )));
    }
    let expected_overlay = contract_adapter_overlay_identity(contract)?;
    if context.overlay.as_deref() != expected_overlay {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: Chroma adapter overlay identity does not match the sealed ordered LoRA/LoKr load receipt"
        )));
    }
    Ok(())
}

pub(crate) fn validate_context(
    provider_id: &str,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    tier: Option<Quant>,
) -> gen_core::Result<()> {
    if let MemorySafetyDecision::Reject { reason } = gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision: Precision::Bf16,
            quant: tier,
            component_precision_floors: &[],
        }),
        None,
    ) {
        return Err(gen_core::Error::Unsupported(reason));
    }
    validate_route(provider_id, contract, context)?;
    Ok(())
}

pub(crate) fn safety_check(
    provider_id: &str,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    tier: Option<Quant>,
) -> MemorySafetyDecision {
    match validate_context(provider_id, contract, context, tier) {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match registered_tier_and_contract(spec, contract) {
        Ok(tier) => safety_check(&contract.provider_id, contract, context, tier),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

fn registered_tier_and_contract(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
) -> gen_core::Result<Option<Quant>> {
    let provider_id = contract.provider_id.as_str();
    validate_load_shape(provider_id, spec)?;
    let route = route_identity(provider_id)?;
    let WeightsSource::Dir(root) = &spec.weights else {
        unreachable!("validated above")
    };
    validate_snapshot_binding(root, route)?;
    let tier = physical_tier_hint(spec);
    if contract.asset_facts == MemoryAssetFacts::default() {
        let expected = weights_free_contract(provider_id, spec)?;
        if expected != *contract {
            return Err(gen_core::Error::Unsupported(format!(
                "{provider_id}: caller contract differs from the exact registry witness"
            )));
        }
        Ok(tier)
    } else {
        let receipt = ChromaLoadReceipt::capture(provider_id, spec)?;
        let expected = contract_from_receipt(provider_id, spec, &receipt);
        if expected != *contract {
            return Err(gen_core::Error::Unsupported(format!(
                "{provider_id}: caller contract differs from the sealed artifact receipt"
            )));
        }
        receipt.ensure_unchanged()?;
        Ok(receipt.tier)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RequestBinding {
    address: usize,
    geometry: MemoryGeometry,
    memory: Option<GenerationMemory>,
    use_pid: bool,
    has_phases: bool,
    prompt: String,
    negative_prompt: Option<String>,
    seed: Option<u64>,
    steps: Option<u32>,
    guidance_bits: Option<u32>,
    true_cfg_bits: Option<u32>,
    sampler: Option<String>,
    scheduler: Option<String>,
    shift_bits: Option<u32>,
    preview_active: bool,
}

impl RequestBinding {
    fn from_request(request: &GenerationRequest) -> Self {
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
            prompt: request.prompt.clone(),
            negative_prompt: request.negative_prompt.clone(),
            seed: request.seed,
            steps: request.steps,
            guidance_bits: request.guidance.map(f32::to_bits),
            true_cfg_bits: request.true_cfg.map(f32::to_bits),
            sampler: request.sampler.clone(),
            scheduler: request.scheduler.clone(),
            shift_bits: request.scheduler_shift.map(f32::to_bits),
            preview_active: request.preview.is_active(),
        }
    }
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
    provider_id: &'static str,
    inner: Arc<Mutex<AdmissionState>>,
}

impl AdmissionRegistry {
    pub(crate) fn new(provider_id: &'static str) -> Self {
        Self {
            provider_id,
            inner: Arc::new(Mutex::new(AdmissionState::default())),
        }
    }

    pub(crate) fn approve(&self, context: &MemoryRunContext) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.inner);
        if state.active.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: another memory request is active",
                self.provider_id
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
        if contract.provider_id != self.provider_id {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: memory contract belongs to {}",
                self.provider_id, contract.provider_id
            )));
        }
        let mut state = candle_gen::lock_recover(&self.inner);
        if state.active.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: another memory request scope is active",
                self.provider_id
            )));
        }
        let approved = state.approved_context.take().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{}: memory request skipped the safety handshake",
                self.provider_id
            ))
        })?;
        if approved != *context {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: memory context changed after safety approval",
                self.provider_id
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
                self.provider_id
            ))
        })?;
        let binding = RequestBinding::from_request(request);
        if active.token != token
            || active.binding.is_some()
            || active.consumed
            || binding.geometry != active.context.geometry
            || binding.memory != active.expected_memory
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: stale or changed memory request",
                self.provider_id
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
                    self.provider_id
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
                self.provider_id
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
                self.provider_id
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

pub(crate) struct ChromaMemoryScope {
    core: candle_gen::request_scope::CandleRequestScopeCore,
    admission: AdmissionRegistry,
    token: u64,
    finished: bool,
}

impl ChromaMemoryScope {
    pub(crate) fn new_bound(
        provider_id: &'static str,
        device: Device,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
        admission: AdmissionRegistry,
    ) -> gen_core::Result<Self> {
        let token = admission.begin(contract, context)?;
        let core = request_scope(provider_id, device, contract, context)?;
        Ok(Self {
            core,
            admission,
            token,
            finished: false,
        })
    }
}

impl MemoryRequestScope for ChromaMemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        self.core.configure_request(request)?;
        self.admission.configure(self.token, request)
    }

    fn enter_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.core.enter_phase(phase)
    }

    fn leave_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.core.leave_phase(phase)
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.core.configure_decode(tile_edge, overlap, geometry)
    }

    fn configure_attention(&mut self, chunk_size: u32) -> gen_core::Result<()> {
        self.core.configure_attention(chunk_size)
    }

    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> gen_core::Result<()> {
        self.core
            .materialize_transformer_window(first_block, block_count)
    }

    fn finish(&mut self, outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        self.core.finish(outcome)?;
        self.admission.finish(self.token)?;
        self.finished = true;
        Ok(())
    }
}

impl Drop for ChromaMemoryScope {
    fn drop(&mut self) {
        if !self.finished {
            self.admission.abandon(self.token);
        }
    }
}

pub(crate) fn request_scope(
    provider_id: &'static str,
    device: Device,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<candle_gen::request_scope::CandleRequestScopeCore> {
    let config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        provider_id,
        device,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        crate::config::ChromaTransformerConfig::default().num_layers
            + crate::config::ChromaTransformerConfig::default().num_single_layers,
        move |_use_pid, _edge, _overlap| {
            Err(gen_core::Error::Unsupported(format!(
                "{provider_id}: bounded decode is Missing"
            )))
        },
    )?;
    Ok(candle_gen::request_scope::CandleRequestScopeCore::new(
        config,
    ))
}

#[cfg(feature = "cuda")]
pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let tier = registered_tier_and_contract(spec, contract)?;
    validate_context(provider_id, contract, context, tier)?;
    Ok(Some(Box::new(request_scope(
        provider_id,
        Device::Cpu,
        contract,
        context,
    )?)))
}

#[cfg(feature = "cuda")]
pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    if strategy != MemoryStrategy::StagedResidency {
        return Ok(Vec::new());
    }
    let context = estimated_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: physical_tier_hint(spec),
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )?;
    Ok(vec![MemoryBehaviorFixture::new(context)])
}

#[cfg(feature = "cuda")]
fn estimated_behavior_context(
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
    numeric: MemoryNumericTier,
    route: MemoryBehaviorRoute,
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
        evidence_revision: "chroma-structural-estimate".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{
        AdapterSpec, LoadShape, MemoryBudget, MemoryCacheState, MemorySelection,
        MemoryStrategyParameters, OffloadPolicy,
    };

    fn write_safetensors(path: &Path, tensors: &[(&str, &str, &[usize], usize)]) {
        let mut offset = 0_usize;
        let mut header = serde_json::Map::new();
        for (name, dtype, shape, bytes) in tensors {
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.resize(bytes.len() + offset, 0);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    fn fixture_root(tmp: &Path, route: RouteIdentity, tier: &str) -> PathBuf {
        let root = tmp
            .join(format!("models--SceneWorks--{}", route.repository))
            .join("snapshots")
            .join(route.revision)
            .join(tier);
        std::fs::create_dir_all(root.join("transformer")).unwrap();
        std::fs::create_dir_all(root.join("text_encoder")).unwrap();
        std::fs::create_dir_all(root.join("vae")).unwrap();
        write_safetensors(
            &root.join("text_encoder/model.safetensors"),
            &[("encoder.weight", "BF16", &[2], 4)],
        );
        write_safetensors(
            &root.join("vae/model.safetensors"),
            &[("decoder.weight", "BF16", &[2], 4)],
        );
        match tier {
            "bf16" => {
                std::fs::write(root.join("transformer/config.json"), b"{}").unwrap();
                write_safetensors(
                    &root.join("transformer/model.safetensors"),
                    &[
                        ("blocks.0.proj.weight", "BF16", &[2, 64], 256),
                        ("norm.weight", "BF16", &[2], 4),
                    ],
                );
            }
            "q4" | "q8" => {
                let bits = if tier == "q4" { 4 } else { 8 };
                std::fs::write(
                    root.join("transformer/config.json"),
                    serde_json::to_vec(&serde_json::json!({
                        "quantization": { "bits": bits, "group_size": 64 }
                    }))
                    .unwrap(),
                )
                .unwrap();
                let columns = if bits == 4 { 8 } else { 16 };
                write_safetensors(
                    &root.join("transformer/model.safetensors"),
                    &[
                        ("blocks.0.proj.weight", "U32", &[2, columns], 8 * columns),
                        ("blocks.0.proj.scales", "BF16", &[2, 1], 4),
                        ("blocks.0.proj.biases", "BF16", &[2, 1], 4),
                        ("norm.weight", "BF16", &[2], 4),
                    ],
                );
            }
            _ => unreachable!(),
        }
        root
    }

    fn spec(root: PathBuf, provider: &str) -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir(root)).with_resolved_route(provider);
        spec.load_shape = LoadShape::DeferredMaterialization;
        spec.offload_policy = OffloadPolicy::Sequential;
        spec
    }

    #[test]
    fn every_route_and_physical_tier_has_an_exact_runtime_projection() {
        for route in ROUTES {
            for (tier, expected_quant, transformer_bytes) in [
                ("q4", Some(Quant::Q4), 88),
                ("q8", Some(Quant::Q8), 144),
                ("bf16", None, 520),
            ] {
                let tmp = tempfile::tempdir().unwrap();
                let root = fixture_root(tmp.path(), *route, tier);
                let receipt =
                    ChromaLoadReceipt::capture(route.provider, &spec(root, route.provider))
                        .unwrap();
                assert_eq!(receipt.route, *route);
                assert_eq!(receipt.tier, expected_quant);
                assert_eq!(receipt.components.dit, transformer_bytes);
                assert_eq!(receipt.components.text_encoder, 4);
                assert_eq!(receipt.components.vae, 4);
                assert!(receipt.adapters.is_empty());
                assert!(receipt.pid.is_none());
            }
        }
    }

    #[test]
    fn path_route_revision_and_tensor_tier_crossing_fail_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let base = ROUTES[1];
        let root = fixture_root(tmp.path(), base, "q4");
        assert!(ChromaLoadReceipt::capture(
            crate::CHROMA1_HD_ID,
            &spec(root.clone(), crate::CHROMA1_HD_ID)
        )
        .is_err());

        let wrong_revision = tmp
            .path()
            .join(format!("models--SceneWorks--{}", base.repository))
            .join("snapshots/not-the-revision/q4");
        std::fs::create_dir_all(wrong_revision.parent().unwrap()).unwrap();
        assert!(
            ChromaLoadReceipt::capture(base.provider, &spec(wrong_revision, base.provider))
                .is_err()
        );

        let mut crossed = spec(root.clone(), base.provider);
        crossed.quantize = Some(Quant::Q4);
        assert!(ChromaLoadReceipt::capture(base.provider, &crossed).is_err());

        assert!(ChromaLoadReceipt::capture(
            base.provider,
            &spec(root.join("extra"), base.provider)
        )
        .is_err());

        std::fs::write(root.join("transformer/config.json"), b"{}").unwrap();
        assert!(ChromaLoadReceipt::capture(base.provider, &spec(root, base.provider)).is_err());
    }

    #[test]
    fn nested_duplicate_fake_truncated_and_mutated_inventory_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let route = ROUTES[1];
        let root = fixture_root(tmp.path(), route, "bf16");
        write_safetensors(
            &root.join("transformer/nested/fake.safetensors"),
            &[("fake", "BF16", &[1], 2)],
        );
        assert!(
            ChromaLoadReceipt::capture(route.provider, &spec(root.clone(), route.provider))
                .is_err()
        );
        std::fs::remove_dir_all(root.join("transformer/nested")).unwrap();

        let receipt =
            ChromaLoadReceipt::capture(route.provider, &spec(root.clone(), route.provider))
                .unwrap();
        std::fs::write(root.join("transformer/model.safetensors"), b"truncated").unwrap();
        assert!(receipt.ensure_unchanged().is_err());

        let tmp = tempfile::tempdir().unwrap();
        let root = fixture_root(tmp.path(), route, "bf16");
        std::fs::write(root.join("vae/model.safetensors"), b"not safetensors").unwrap();
        assert!(ChromaLoadReceipt::capture(route.provider, &spec(root, route.provider)).is_err());

        let tmp = tempfile::tempdir().unwrap();
        let root = fixture_root(tmp.path(), route, "bf16");
        write_safetensors(
            &root.join("transformer/duplicate.safetensors"),
            &[("norm.weight", "BF16", &[2], 4)],
        );
        assert!(ChromaLoadReceipt::capture(route.provider, &spec(root, route.provider)).is_err());

        let tmp = tempfile::tempdir().unwrap();
        let root = fixture_root(tmp.path(), route, "bf16");
        std::fs::write(root.join("download.incomplete"), b"marker").unwrap();
        assert!(ChromaLoadReceipt::capture(route.provider, &spec(root, route.provider)).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn hugging_face_blob_file_symlinks_are_pinned_not_rejected() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let route = ROUTES[1];
        let root = fixture_root(temp.path(), route, "bf16");
        let lexical = root.join("vae/model.safetensors");
        let blob = temp.path().join("blobs/vae");
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        std::fs::rename(&lexical, &blob).unwrap();
        symlink(&blob, &lexical).unwrap();

        let receipt = ChromaLoadReceipt::capture(route.provider, &spec(root, route.provider))
            .expect("HF snapshot file symlink remains an exact pinned member");
        receipt.ensure_unchanged().unwrap();
    }

    #[test]
    fn ordered_lora_lokr_receipts_pin_kind_scale_target_digest_and_additive_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let route = ROUTES[0];
        let root = fixture_root(tmp.path(), route, "q8");
        let a = tmp.path().join("a.safetensors");
        let b = tmp.path().join("b.safetensors");
        write_safetensors(&a, &[("lora.down.weight", "F16", &[2, 2], 8)]);
        write_safetensors(&b, &[("layer.lokr_w1", "F32", &[3], 12)]);
        let mut load = spec(root, route.provider);
        load.adapters = vec![
            AdapterSpec::new(a.clone(), 0.5, AdapterKind::Lora),
            AdapterSpec::new(b, 1.25, AdapterKind::Lokr),
        ];
        let receipt = ChromaLoadReceipt::capture(route.provider, &load).unwrap();
        assert_eq!(receipt.adapters.len(), 2);
        assert_eq!(receipt.adapters[0].order, 0);
        assert_eq!(receipt.adapters[0].kind, AdapterKind::Lora);
        assert_eq!(receipt.adapters[0].scale_bits, 0.5_f32.to_bits());
        assert_eq!(receipt.adapters[0].target, "chroma.transformer");
        assert_eq!(receipt.adapters[0].additive_resident_bytes, 16);
        assert_eq!(receipt.adapters[1].kind, AdapterKind::Lokr);
        assert_eq!(receipt.adapters[1].additive_resident_bytes, 12);
        assert_ne!(receipt.adapters[0].file.sha256, [0; 32]);

        load.adapters
            .push(AdapterSpec::new(a, 1.0, AdapterKind::Lora));
        assert!(ChromaLoadReceipt::capture(route.provider, &load).is_err());
    }

    #[test]
    fn provider_handshake_binds_the_exact_materialized_adapter_load_identity() {
        let tmp = tempfile::tempdir().unwrap();
        let route = ROUTES[1];
        let root = fixture_root(tmp.path(), route, "q4");
        let adapter = tmp.path().join("adapter.safetensors");
        write_safetensors(&adapter, &[("lora.down.weight", "F16", &[2, 2], 8)]);
        let mut load = spec(root, route.provider);
        load.adapters = vec![AdapterSpec::new(adapter.clone(), 0.75, AdapterKind::Lora)];

        let first = ChromaLoadReceipt::capture(route.provider, &load).unwrap();
        let first_contract = contract_from_receipt(route.provider, &load, &first);
        let first_identity = first.adapter_overlay_identity().unwrap();
        assert_eq!(
            contract_adapter_overlay_identity(&first_contract).unwrap(),
            Some(first_identity.as_str())
        );

        let mut bytes = std::fs::read(&adapter).unwrap();
        let last = bytes.last_mut().unwrap();
        *last ^= 0x5a;
        std::fs::write(&adapter, bytes).unwrap();
        let second = ChromaLoadReceipt::capture(route.provider, &load).unwrap();
        let second_contract = contract_from_receipt(route.provider, &load, &second);
        let second_identity = second.adapter_overlay_identity().unwrap();
        assert_ne!(first_identity, second_identity);

        let mut exact = context(&second_contract, 1);
        exact.overlay = Some(second_identity);
        assert!(
            validate_context(route.provider, &second_contract, &exact, Some(Quant::Q4)).is_ok()
        );

        let mut crossed = exact.clone();
        crossed.overlay = Some(first_identity);
        assert!(
            validate_context(route.provider, &second_contract, &crossed, Some(Quant::Q4)).is_err()
        );
        crossed.overlay = Some("lora".into());
        assert!(
            validate_context(route.provider, &second_contract, &crossed, Some(Quant::Q4)).is_err()
        );
    }

    #[test]
    fn pid_receipt_binds_flux_student_tier_and_gemma_revision() {
        let tmp = tempfile::tempdir().unwrap();
        let route = ROUTES[2];
        let root = fixture_root(tmp.path(), route, "q4");
        let pid_root = tmp
            .path()
            .join("models--SceneWorks--pid-flux/snapshots")
            .join(PID_FLUX_REVISION);
        let student = pid_root.join("pid_flux_2k.safetensors");
        write_safetensors(&student, &[("student.weight", "BF16", &[2], 4)]);
        let gemma = tmp
            .path()
            .join("models--SceneWorks--gemma-2-2b-it/snapshots")
            .join(PID_GEMMA_REVISION);
        write_safetensors(
            &gemma.join("gemma-2-2b-it.safetensors"),
            &[("model.weight", "BF16", &[2], 4)],
        );
        std::fs::write(gemma.join("tokenizer.json"), b"{}").unwrap();
        let load = spec(root, route.provider).with_pid(
            WeightsSource::File(student),
            WeightsSource::Dir(gemma.clone()),
        );
        let receipt = ChromaLoadReceipt::capture(route.provider, &load).unwrap();
        let pid = receipt.pid.unwrap();
        assert_eq!(pid.tier, PidStudentTier::Res2k);
        assert_eq!(pid.projected_resident_bytes, 16);

        let mut crossed = load;
        crossed.pid.as_mut().unwrap().gemma = WeightsSource::Dir(gemma.join("wrong"));
        assert!(ChromaLoadReceipt::capture(route.provider, &crossed).is_err());
    }

    fn context(contract: &MemoryProviderContract, count: u32) -> MemoryRunContext {
        MemoryRunContext {
            selection: MemorySelection {
                strategy: MemoryStrategy::StagedResidency,
                parameters: MemoryStrategyParameters::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: &[],
                },
            },
            optimization_authority: gen_core::MemoryOptimizationAuthority::Estimated,
            calibration_abi: 0,
            calibration_fingerprint: String::new(),
            load_shape: contract.load_shape,
            geometry: gen_core::MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: count,
                frames: 1,
                reference_count: 0,
            },
            mode: MemoryMode::TextToImage,
            budget: MemoryBudget {
                total_bytes: u64::MAX,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "test".into(),
            has_reference: false,
            use_pid: false,
            has_phases: false,
            overlay: None,
        }
    }

    #[test]
    fn staged_execution_requires_the_exact_provider_owned_request_scope() {
        let provider = crate::CHROMA1_BASE_ID;
        let mut surface = gen_core::candle_memory_contract_surface_specs()
            .into_iter()
            .find(|surface| {
                surface.selector.tier == gen_core::MemoryContractSurfaceTier::Q4
                    && surface.selector.load_shape == LoadShape::DeferredMaterialization
            })
            .unwrap();
        surface.spec.resolved_route = Some(provider.into());
        let contract = weights_free_surface_contract(provider, &surface).unwrap();
        let context = context(&contract, 1);
        let admission = AdmissionRegistry::new(provider);

        let mut bypass = GenerationRequest {
            width: 1024,
            height: 1024,
            count: 1,
            ..Default::default()
        };
        bypass.memory = contract.generation_memory(&context.selection);
        assert!(admission.consume_for_generate(&bypass).is_err());

        admission.approve(&context).unwrap();
        let mut scope = ChromaMemoryScope::new_bound(
            provider,
            Device::Cpu,
            &contract,
            &context,
            admission.clone(),
        )
        .unwrap();
        let mut request = GenerationRequest {
            width: 1024,
            height: 1024,
            count: 1,
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();

        let copied = GenerationRequest { ..request.clone() };
        assert!(admission.consume_for_generate(&copied).is_err());
        admission.consume_for_generate(&request).unwrap();
        assert!(admission.consume_for_generate(&request).is_err());
        scope.finish(MemoryRunOutcome::Complete).unwrap();
    }

    #[test]
    fn complete_cancel_error_and_panic_drop_all_release_the_admission() {
        let provider = crate::CHROMA1_BASE_ID;
        let mut surface = gen_core::candle_memory_contract_surface_specs()
            .into_iter()
            .find(|surface| {
                surface.selector.tier == gen_core::MemoryContractSurfaceTier::Q4
                    && surface.selector.load_shape == LoadShape::DeferredMaterialization
            })
            .unwrap();
        surface.spec.resolved_route = Some(provider.into());
        let contract = weights_free_surface_contract(provider, &surface).unwrap();
        let run = context(&contract, 1);
        let admission = AdmissionRegistry::new(provider);

        for outcome in [
            MemoryRunOutcome::Complete,
            MemoryRunOutcome::Canceled,
            MemoryRunOutcome::Error {
                message: "expected".into(),
            },
        ] {
            admission.approve(&run).unwrap();
            let mut scope = ChromaMemoryScope::new_bound(
                provider,
                Device::Cpu,
                &contract,
                &run,
                admission.clone(),
            )
            .unwrap();
            scope.finish(outcome).unwrap();
        }

        admission.approve(&run).unwrap();
        let scope =
            ChromaMemoryScope::new_bound(provider, Device::Cpu, &contract, &run, admission.clone())
                .unwrap();
        drop(scope);
        admission.approve(&run).unwrap();
        let mut replacement =
            ChromaMemoryScope::new_bound(provider, Device::Cpu, &contract, &run, admission)
                .unwrap();
        replacement.finish(MemoryRunOutcome::Complete).unwrap();
    }

    #[test]
    fn request_binding_seals_prompt_sampling_seed_schedule_cfg_and_preview() {
        let baseline = GenerationRequest {
            prompt: "baseline".into(),
            negative_prompt: Some("negative".into()),
            width: 1024,
            height: 1024,
            count: 1,
            seed: Some(42),
            steps: Some(40),
            true_cfg: Some(3.0),
            sampler: Some("euler".into()),
            scheduler: Some("beta".into()),
            scheduler_shift: Some(1.25),
            ..Default::default()
        };
        let sealed = RequestBinding::from_request(&baseline);
        let mut crossed = baseline.clone();
        crossed.prompt.push_str(" changed");
        assert_ne!(sealed, RequestBinding::from_request(&crossed));
        let mut crossed = baseline.clone();
        crossed.negative_prompt = None;
        assert_ne!(sealed, RequestBinding::from_request(&crossed));
        let mut crossed = baseline.clone();
        crossed.seed = Some(43);
        assert_ne!(sealed, RequestBinding::from_request(&crossed));
        let mut crossed = baseline.clone();
        crossed.steps = Some(39);
        assert_ne!(sealed, RequestBinding::from_request(&crossed));
        let mut crossed = baseline.clone();
        crossed.true_cfg = Some(2.5);
        assert_ne!(sealed, RequestBinding::from_request(&crossed));
        let mut crossed = baseline.clone();
        crossed.sampler = Some("heun".into());
        assert_ne!(sealed, RequestBinding::from_request(&crossed));
        let mut crossed = baseline.clone();
        crossed.scheduler = Some("simple".into());
        assert_ne!(sealed, RequestBinding::from_request(&crossed));
        let mut crossed = baseline.clone();
        crossed.scheduler_shift = Some(1.5);
        assert_ne!(sealed, RequestBinding::from_request(&crossed));
        let mut crossed = baseline;
        crossed.preview = gen_core::PreviewSink::new(|_| {});
        assert_ne!(sealed, RequestBinding::from_request(&crossed));
    }

    #[test]
    fn registry_admission_requires_the_sealed_contract_and_prepared_inventory() {
        let temp = tempfile::tempdir().unwrap();
        let route = ROUTES[1];
        let root = fixture_root(temp.path(), route, "q4");
        let mut load = spec(root.clone(), route.provider);
        let pins = recursive_artifact_files(&root)
            .unwrap()
            .into_iter()
            .map(gen_core::PinnedWeightsFile::pin)
            .collect::<gen_core::Result<Vec<_>>>()
            .unwrap();
        load.prepare_with_file_pins(pins).unwrap();
        let contract = provider_contract(route.provider, &load).unwrap();
        let run = context(&contract, 1);
        assert_eq!(
            registered_safety_check(&load, &contract, &run),
            MemorySafetyDecision::Accept
        );

        let mut forged = contract.clone();
        forged.asset_facts.base_bytes = forged.asset_facts.base_bytes.saturating_add(1);
        assert!(matches!(
            registered_safety_check(&load, &forged, &run),
            MemorySafetyDecision::Reject { .. }
        ));

        std::fs::write(
            root.join("transformer/config.json"),
            b"{\"quantization\":{\"bits\":4,\"group_size\":64},\"changed\":true}",
        )
        .unwrap();
        assert!(matches!(
            registered_safety_check(&load, &contract, &run),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn only_resident_and_staged_are_published_and_route_axes_are_exact() {
        for provider in [
            crate::CHROMA1_HD_ID,
            crate::CHROMA1_BASE_ID,
            crate::CHROMA1_FLASH_ID,
        ] {
            let mut surface = gen_core::candle_memory_contract_surface_specs()
                .into_iter()
                .find(|surface| {
                    surface.selector.tier == gen_core::MemoryContractSurfaceTier::Q4
                        && surface.selector.load_shape == LoadShape::DeferredMaterialization
                })
                .unwrap();
            surface.spec.resolved_route = Some(provider.into());
            let contract = weights_free_surface_contract(provider, &surface).unwrap();
            assert!(contract.calibration.is_none());
            for strategy in MemoryStrategy::ALL {
                let support = &contract.capability(strategy).unwrap().support;
                assert_eq!(
                    matches!(support, MemoryStrategySupport::Implemented),
                    matches!(
                        strategy,
                        MemoryStrategy::Resident | MemoryStrategy::StagedResidency
                    )
                );
            }
            for count in [1, 2, 4] {
                assert!(validate_context(
                    provider,
                    &contract,
                    &context(&contract, count),
                    Some(Quant::Q4)
                )
                .is_ok());
            }
            for &(width, height) in PUBLIC_GEOMETRIES {
                let mut exact = context(&contract, 1);
                exact.geometry.width = width;
                exact.geometry.height = height;
                assert!(validate_context(provider, &contract, &exact, Some(Quant::Q4)).is_ok());
            }
            for (width, height) in [(1280, 1280), (720, 720), (1024, 768)] {
                let mut crossed = context(&contract, 1);
                crossed.geometry.width = width;
                crossed.geometry.height = height;
                assert!(validate_context(provider, &contract, &crossed, Some(Quant::Q4)).is_err());
            }
            for count in [0, 3, 5] {
                assert!(validate_context(
                    provider,
                    &contract,
                    &context(&contract, count),
                    Some(Quant::Q4)
                )
                .is_err());
            }
            let mut alias = context(&contract, 1);
            alias.mode = MemoryMode::Other("style_variations".into());
            assert!(validate_context(provider, &contract, &alias, Some(Quant::Q4)).is_err());
            let mut crossed = context(&contract, 1);
            crossed.selection.tier.quant = Some(Quant::Q8);
            assert!(validate_context(provider, &contract, &crossed, Some(Quant::Q4)).is_err());
        }
    }
}
