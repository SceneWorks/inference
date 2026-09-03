//! Exact request-scoped Candle/CUDA memory contract for Ideogram 4 Base and Turbo (SC-20789).

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
    MemoryNumericTier, MemoryParameterRanges, MemoryPhase, MemoryPrerequisiteScope,
    MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent, MemoryRunContext,
    MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability,
    MemoryStrategyPrerequisite, MemoryStrategySupport, MemoryWindowMaterialization, Precision,
    Quant, TransformerComponent, WeightsSource,
};
use candle_gen::gen_core::{
    MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryBudget, MemoryCacheState,
    MemoryOptimizationAuthority,
};
use sha2::{Digest, Sha256};

use crate::config::{Ideogram4DitConfig, MODEL_ID, MODEL_ID_TURBO, TURBO_LORA_FILE};
use candle_gen::gen_core::Conditioning;

pub const PACKED_REPOSITORY: &str = "ideogram-4-mlx";
pub const PACKED_REVISION: &str = "a3095855b8819dc0d6b067cb1354aaa7da189ff8";
pub const BF16_REPOSITORY: &str = "ideogram-4";
pub const BF16_REVISION: &str = "2e8fb610109bf0db195344cc424df98b301d3cad";
pub const PID_FLUX2_REVISION: &str = "6d5c1f1049e863f1757f68fd81c6bc850a95609d";
pub const PID_GEMMA_REVISION: &str = "684c553b5b41a1c835989d89f62f585e6269a7de";
pub const PUBLIC_GEOMETRIES: &[(u32, u32)] = &[
    (1024, 1024),
    (768, 1024),
    (1024, 768),
    (1280, 720),
    (720, 1280),
];
const USER_ADAPTER_PREFIX: &str = "ideogram.adapters.ordered-additive.sha256:";
const TURBO_ADAPTER_PREFIX: &str = "ideogram.turbo-time.sha256:";
const PID_PREFIX: &str = "ideogram.pid.flux2.sha256:";
pub const PHYSICAL_RECEIPT_PREFIX: &str = "ideogram.physical.sha256:";
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_OVERLAP: u32 = 64;
pub const ATTENTION_CHUNK_SIZE: u32 = 64 * 1024 * 1024;
pub const TRANSFORMER_WINDOW_SIZE: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RouteIdentity {
    pub provider: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
    pub tier: Option<Quant>,
}

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
        projection: impl Fn(&gen_core::weightsmeta::SafetensorsTensorHeader) -> gen_core::Result<u64>,
    ) -> gen_core::Result<Self> {
        let pin = prepared_or_current_pin(spec, path)?;
        let headers = pin.read_unchanged(|stable| {
            gen_core::weightsmeta::safetensors_path_tensor_headers(stable)
        })?;
        if headers.is_empty() {
            return Err(gen_core::Error::Unsupported(format!(
                "ideogram: {} contains no tensors",
                path.display()
            )));
        }
        let projected_resident_bytes = headers.iter().try_fold(0_u64, |total, header| {
            total.checked_add(projection(header)?).ok_or_else(|| {
                gen_core::Error::Unsupported("ideogram: tensor bytes overflow".into())
            })
        })?;
        if projected_resident_bytes == 0 {
            return Err(gen_core::Error::Unsupported(format!(
                "ideogram: {} projects to zero resident bytes",
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
        if self.pin.read_unchanged(sha256_file)? != self.sha256 {
            return Err(gen_core::Error::Unsupported(format!(
                "ideogram: artifact digest changed after sealing {}",
                self.lexical_path.display()
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AdapterReceipt {
    pub order: usize,
    pub file: FileReceipt,
    pub kind: AdapterKind,
    pub scale_bits: u32,
    pub target_count: u8,
    pub materialized_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PidReceipt {
    pub student: FileReceipt,
    pub gemma: Vec<FileReceipt>,
    pub tokenizer: gen_core::PinnedWeightsFile,
    pub materialized_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ComponentBytes {
    pub text_encoder: u64,
    pub conditional_transformer: u64,
    pub unconditional_transformer: u64,
    pub vae: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct IdeogramLoadReceipt {
    pub route: RouteIdentity,
    pub lexical_root: PathBuf,
    pub canonical_root: PathBuf,
    inventory: Vec<(PathBuf, gen_core::PinnedWeightsFile, [u8; 32])>,
    pub transformer: Vec<FileReceipt>,
    pub unconditional_transformer: Vec<FileReceipt>,
    pub text_encoder: Vec<FileReceipt>,
    pub vae: Vec<FileReceipt>,
    pub turbo_adapter: Option<FileReceipt>,
    pub adapters: Vec<AdapterReceipt>,
    pub pid: Option<PidReceipt>,
    pub components: ComponentBytes,
}

impl IdeogramLoadReceipt {
    pub(crate) fn capture(provider_id: &str, spec: &LoadSpec) -> gen_core::Result<Self> {
        validate_load_shape(provider_id, spec)?;
        let WeightsSource::Dir(root) = &spec.weights else {
            unreachable!("validated directory above")
        };
        let route = route_identity(provider_id, root)?;
        validate_snapshot_binding(root, route)?;
        let lexical_root = std::path::absolute(root)?;
        let canonical_root = std::fs::canonicalize(&lexical_root)?;
        let inventory_paths =
            recursive_loader_inventory(&lexical_root, provider_id == MODEL_ID_TURBO)?;
        let mut inventory = Vec::with_capacity(inventory_paths.len());
        for path in inventory_paths {
            let pin = prepared_or_current_pin(spec, &path)?;
            let digest = pin.read_unchanged(sha256_file)?;
            inventory.push((path, pin, digest));
        }

        let transformer_paths = direct_safetensors_inventory(&root.join("transformer"))?;
        let text_paths = direct_safetensors_inventory(&root.join("text_encoder"))?;
        let vae_paths = direct_safetensors_inventory(&root.join("vae"))?;
        let uncond_paths = if provider_id == MODEL_ID {
            direct_safetensors_inventory(&root.join("unconditional_transformer"))?
        } else {
            if root.join("unconditional_transformer").exists() {
                return Err(gen_core::Error::Unsupported(
                    "ideogram turbo: unexpected unconditional_transformer inventory".into(),
                ));
            }
            Vec::new()
        };
        for (paths, component) in [
            (&transformer_paths, "conditional transformer"),
            (&text_paths, "text encoder"),
            (&vae_paths, "VAE"),
            (&uncond_paths, "unconditional transformer"),
        ] {
            if !paths.is_empty() {
                ensure_unique_tensor_names(paths, component)?;
            }
        }

        let transformer_tier = inspect_component_tier(&transformer_paths, "transformer")?;
        let text_tier = inspect_component_tier(&text_paths, "text_encoder")?;
        let uncond_tier = if uncond_paths.is_empty() {
            None
        } else {
            Some(inspect_component_tier(
                &uncond_paths,
                "unconditional_transformer",
            )?)
        };
        if transformer_tier != route.tier
            || text_tier != route.tier
            || uncond_tier.is_some_and(|tier| tier != route.tier)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{provider_id}: physical component tiers cross the sealed {:?} route",
                route.tier
            )));
        }
        let transformer = capture_component(spec, &transformer_paths, route.tier, false)?;
        let unconditional_transformer = capture_component(spec, &uncond_paths, route.tier, false)?;
        let text_encoder = capture_component(spec, &text_paths, route.tier, true)?;
        let vae = vae_paths
            .iter()
            .map(|path| FileReceipt::capture(spec, path, f32_projection))
            .collect::<gen_core::Result<Vec<_>>>()?;

        let turbo_adapter = if provider_id == MODEL_ID_TURBO {
            Some(FileReceipt::capture(
                spec,
                &root.join(TURBO_LORA_FILE),
                f32_projection,
            )?)
        } else {
            None
        };
        let adapters = capture_adapters(spec, provider_id == MODEL_ID)?;
        let pid = capture_pid(spec)?;
        let sum = |files: &[FileReceipt]| {
            files.iter().fold(0_u64, |total, file| {
                total.saturating_add(file.projected_resident_bytes)
            })
        };
        let components = ComponentBytes {
            text_encoder: sum(&text_encoder),
            conditional_transformer: sum(&transformer),
            unconditional_transformer: sum(&unconditional_transformer),
            vae: sum(&vae),
        };
        if components.text_encoder == 0
            || components.conditional_transformer == 0
            || components.vae == 0
            || (provider_id == MODEL_ID && components.unconditional_transformer == 0)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{provider_id}: exact nonzero component inventory is incomplete"
            )));
        }
        let receipt = Self {
            route,
            lexical_root,
            canonical_root,
            inventory,
            transformer,
            unconditional_transformer,
            text_encoder,
            vae,
            turbo_adapter,
            adapters,
            pid,
            components,
        };
        receipt.ensure_unchanged()?;
        Ok(receipt)
    }

    pub(crate) fn ensure_unchanged(&self) -> gen_core::Result<()> {
        let mut current = recursive_loader_inventory(
            &self.canonical_root,
            self.route.provider == MODEL_ID_TURBO,
        )?
        .into_iter()
        .map(std::fs::canonicalize)
        .collect::<std::io::Result<Vec<_>>>()?;
        let mut expected = self
            .inventory
            .iter()
            .map(|(path, _, _)| std::fs::canonicalize(path))
            .collect::<std::io::Result<Vec<_>>>()?;
        current.sort();
        expected.sort();
        if current != expected {
            return Err(gen_core::Error::Unsupported(format!(
                "ideogram: load-exact artifact inventory changed after sealing: expected {expected:?}, current {current:?}"
            )));
        }
        for (_, pin, digest) in &self.inventory {
            pin.ensure_unchanged()?;
            if pin.read_unchanged(sha256_file)? != *digest {
                return Err(gen_core::Error::Unsupported(
                    "ideogram: artifact contents changed after sealing".into(),
                ));
            }
        }
        for file in self
            .transformer
            .iter()
            .chain(&self.unconditional_transformer)
            .chain(&self.text_encoder)
            .chain(&self.vae)
            .chain(self.turbo_adapter.iter())
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
            pid.tokenizer.ensure_unchanged()?;
        }
        Ok(())
    }

    fn user_adapter_identity(&self) -> Option<String> {
        (!self.adapters.is_empty()).then(|| {
            digest_identity(USER_ADAPTER_PREFIX, |digest| {
                update_framed(digest, self.route.provider.as_bytes());
                for adapter in &self.adapters {
                    update_framed(digest, &(adapter.order as u64).to_le_bytes());
                    update_framed(
                        digest,
                        match adapter.kind {
                            AdapterKind::Lora => b"lora",
                            AdapterKind::Lokr => b"lokr",
                        },
                    );
                    update_framed(digest, &adapter.scale_bits.to_le_bytes());
                    update_framed(digest, &[adapter.target_count]);
                    update_framed(digest, &adapter.materialized_bytes.to_le_bytes());
                    update_framed(digest, &adapter.file.sha256);
                    update_framed(
                        digest,
                        adapter.file.canonical_path.as_os_str().as_encoded_bytes(),
                    );
                }
            })
        })
    }

    fn turbo_adapter_identity(&self) -> Option<String> {
        self.turbo_adapter.as_ref().map(|file| {
            digest_identity(TURBO_ADAPTER_PREFIX, |digest| {
                update_framed(digest, self.route.repository.as_bytes());
                update_framed(digest, self.route.revision.as_bytes());
                update_framed(digest, &file.sha256);
                update_framed(digest, &file.projected_resident_bytes.to_le_bytes());
            })
        })
    }

    fn pid_identity(&self) -> Option<String> {
        self.pid.as_ref().map(|pid| {
            digest_identity(PID_PREFIX, |digest| {
                update_framed(digest, b"SceneWorks/pid-flux2");
                update_framed(digest, PID_FLUX2_REVISION.as_bytes());
                update_framed(digest, &pid.student.sha256);
                update_framed(digest, b"SceneWorks/gemma-2-2b-it");
                update_framed(digest, PID_GEMMA_REVISION.as_bytes());
                for file in &pid.gemma {
                    update_framed(digest, &file.sha256);
                }
                update_framed(digest, &pid.materialized_bytes.to_le_bytes());
            })
        })
    }

    pub(crate) fn physical_identity(&self) -> String {
        digest_identity(PHYSICAL_RECEIPT_PREFIX, |digest| {
            update_framed(digest, self.route.provider.as_bytes());
            update_framed(digest, self.route.repository.as_bytes());
            update_framed(digest, self.route.revision.as_bytes());
            update_framed(
                digest,
                match self.route.tier {
                    Some(Quant::Q4) => b"q4",
                    Some(Quant::Q8) => b"q8",
                    Some(Quant::Nvfp4) => b"nvfp4",
                    None => b"bf16",
                },
            );
            for (path, _, sha256) in &self.inventory {
                let relative = path.strip_prefix(&self.lexical_root).unwrap_or(path);
                update_framed(digest, relative.as_os_str().as_encoded_bytes());
                update_framed(digest, sha256);
            }
            for bytes in [
                self.components.text_encoder,
                self.components.conditional_transformer,
                self.components.unconditional_transformer,
                self.components.vae,
            ] {
                update_framed(digest, &bytes.to_le_bytes());
            }
        })
    }
}

fn digest_identity(prefix: &str, write_parts: impl FnOnce(&mut Sha256)) -> String {
    let mut digest = Sha256::new();
    write_parts(&mut digest);
    let mut encoded = String::with_capacity(prefix.len() + 64);
    encoded.push_str(prefix);
    for byte in digest.finalize() {
        write!(&mut encoded, "{byte:02x}").expect("hexadecimal String write");
    }
    encoded
}

fn update_framed(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
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
                    "ideogram: sealed load receipt is missing {}",
                    absolute.display()
                ))
            })
    } else {
        gen_core::PinnedWeightsFile::pin(path)
    }
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

fn recursive_loader_inventory(root: &Path, turbo: bool) -> gen_core::Result<Vec<PathBuf>> {
    fn visit(
        root: &Path,
        tier_root: &Path,
        turbo: bool,
        files: &mut Vec<PathBuf>,
    ) -> gen_core::Result<()> {
        for entry in std::fs::read_dir(root)? {
            let path = entry?.path();
            let metadata = std::fs::symlink_metadata(&path)?;
            if metadata.file_type().is_symlink() {
                let target = std::fs::metadata(&path)?;
                if !target.is_file() {
                    return Err(gen_core::Error::Unsupported(format!(
                        "ideogram: symlink must resolve directly to a file: {}",
                        path.display()
                    )));
                }
            } else if metadata.is_dir() {
                visit(&path, tier_root, turbo, files)?;
                continue;
            } else if !metadata.is_file() {
                continue;
            }
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
                    "ideogram: incomplete/hidden artifact marker is forbidden: {}",
                    path.display()
                )));
            }
            let relative = path.strip_prefix(tier_root).unwrap_or(&path);
            let direct_component = relative.components().count() == 2
                && matches!(
                    relative
                        .components()
                        .next()
                        .and_then(|part| part.as_os_str().to_str()),
                    Some("transformer" | "unconditional_transformer" | "text_encoder" | "vae")
                );
            let exact_root = turbo && relative == Path::new(TURBO_LORA_FILE);
            let exact_tokenizer = relative == Path::new("tokenizer/tokenizer.json");
            let exact_config = relative == Path::new("transformer/config.json")
                || relative == Path::new("unconditional_transformer/config.json");
            if (direct_component
                && path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))
                || exact_root
                || exact_tokenizer
                || exact_config
            {
                files.push(std::path::absolute(path)?);
            }
        }
        Ok(())
    }
    let root = std::path::absolute(root)?;
    let mut files = Vec::new();
    visit(&root, &root, turbo, &mut files)?;
    files.sort();
    if files.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "ideogram: empty loader inventory".into(),
        ));
    }
    Ok(files)
}

fn direct_safetensors_inventory(dir: &Path) -> gen_core::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|error| {
        gen_core::Error::Unsupported(format!(
            "ideogram: cannot inventory {}: {error}",
            dir.display()
        ))
    })? {
        let path = entry?.path();
        if !path.is_dir()
            && path.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
            && !gen_core::weightsmeta::is_hidden_file(&path)
        {
            files.push(path);
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "ideogram: {} contains no direct safetensors",
            dir.display()
        )));
    }
    Ok(files)
}

fn ensure_unique_tensor_names(paths: &[PathBuf], component: &str) -> gen_core::Result<()> {
    let mut names = BTreeSet::new();
    for path in paths {
        for header in gen_core::weightsmeta::safetensors_path_tensor_headers(path)? {
            if !names.insert(header.name.clone()) {
                return Err(gen_core::Error::Unsupported(format!(
                    "ideogram: duplicate {component} tensor {}",
                    header.name
                )));
            }
        }
    }
    Ok(())
}

fn route_identity(provider_id: &str, root: &Path) -> gen_core::Result<RouteIdentity> {
    if !matches!(provider_id, MODEL_ID | MODEL_ID_TURBO) {
        return Err(gen_core::Error::Unsupported(format!(
            "unknown Ideogram provider {provider_id}"
        )));
    }
    let tier = root
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            gen_core::Error::Unsupported("ideogram: missing tier path component".into())
        })?;
    let (repository, revision, tier) = match tier {
        "q4" => (PACKED_REPOSITORY, PACKED_REVISION, Some(Quant::Q4)),
        "q8" => (PACKED_REPOSITORY, PACKED_REVISION, Some(Quant::Q8)),
        "bf16" => (BF16_REPOSITORY, BF16_REVISION, None),
        value => {
            return Err(gen_core::Error::Unsupported(format!(
                "ideogram: unsupported physical tier {value:?}"
            )))
        }
    };
    Ok(RouteIdentity {
        provider: if provider_id == MODEL_ID {
            MODEL_ID
        } else {
            MODEL_ID_TURBO
        },
        repository,
        revision,
        tier,
    })
}

fn validate_snapshot_binding(root: &Path, route: RouteIdentity) -> gen_core::Result<()> {
    validate_snapshot_path(root, route)?;
    let canonical = std::fs::canonicalize(root).map_err(|error| {
        gen_core::Error::Unsupported(format!(
            "{}: cannot resolve canonical snapshot root {}: {error}",
            route.provider,
            root.display()
        ))
    })?;
    validate_snapshot_path(&canonical, route)?;
    Ok(())
}

fn validate_snapshot_path(root: &Path, route: RouteIdentity) -> gen_core::Result<()> {
    let hf = format!("models--SceneWorks--{}", route.repository);
    let app = format!("SceneWorks__{}", route.repository);
    let tier = root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let components = root
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let suffixes = [
        vec![hf, "snapshots".into(), route.revision.into(), tier.into()],
        vec![app, route.revision.into(), tier.into()],
        vec![route.repository.into(), route.revision.into(), tier.into()],
    ];
    let Some(identity_depth) = suffixes
        .iter()
        .find(|suffix| components.ends_with(suffix))
        .map(Vec::len)
    else {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: weights path {} crosses SceneWorks/{}@{}",
            route.provider,
            root.display(),
            route.repository,
            route.revision
        )));
    };
    let mut identity_component = root;
    for _ in 0..identity_depth {
        if std::fs::symlink_metadata(identity_component)
            .map(|metadata| metadata.file_type().is_symlink())
            .unwrap_or(false)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: snapshot identity component {} is a symlink; exact repository/revision/tier provenance must be canonical",
                route.provider,
                identity_component.display()
            )));
        }
        identity_component = identity_component.parent().unwrap_or(identity_component);
    }
    Ok(())
}

fn validate_load_shape(provider_id: &str, spec: &LoadSpec) -> gen_core::Result<()> {
    if !matches!(provider_id, MODEL_ID | MODEL_ID_TURBO)
        || spec.resolved_route.as_deref() != Some(provider_id)
        || spec.precision != Precision::Bf16
        || spec.quantize.is_some()
        || !matches!(spec.weights, WeightsSource::Dir(_))
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: exact Ideogram directory route requires precision=Bf16 and quantize=None"
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
            "{provider_id}: unsupported control, identity, or external-component composition"
        )));
    }
    Ok(())
}

fn inspect_component_tier(paths: &[PathBuf], component: &str) -> gen_core::Result<Option<Quant>> {
    use gen_core::weightsmeta::Dtype;
    let mut headers = BTreeMap::new();
    for path in paths {
        for header in gen_core::weightsmeta::safetensors_path_tensor_headers(path)? {
            headers.insert(header.name.clone(), header);
        }
    }
    let packed = headers
        .keys()
        .filter_map(|name| name.strip_suffix(".scales").map(str::to_owned))
        .collect::<BTreeSet<_>>();
    if packed.is_empty() {
        if headers
            .values()
            .any(|header| !matches!(header.dtype, Dtype::BF16 | Dtype::F32))
        {
            return Err(gen_core::Error::Unsupported(format!(
                "ideogram: dense {component} contains a non-BF16/F32 tensor"
            )));
        }
        return Ok(None);
    }
    let mut resolved_bits = None;
    for base in &packed {
        let weight = headers.get(&format!("{base}.weight")).ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "ideogram: packed {component} {base} lacks weight"
            ))
        })?;
        let scales = headers
            .get(&format!("{base}.scales"))
            .expect("packed base from scales");
        let biases = headers.get(&format!("{base}.biases")).ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "ideogram: packed {component} {base} lacks biases"
            ))
        })?;
        if weight.dtype != Dtype::U32
            || scales.dtype != Dtype::BF16
            || biases.dtype != Dtype::BF16
            || scales.shape != biases.shape
        {
            return Err(gen_core::Error::Unsupported(format!(
                "ideogram: malformed packed {component} {base}"
            )));
        }
        let [rows, words] = weight.shape.as_slice() else {
            return Err(gen_core::Error::Unsupported(format!(
                "ideogram: packed {component} weight must be rank two"
            )));
        };
        let [scale_rows, groups] = scales.shape.as_slice() else {
            return Err(gen_core::Error::Unsupported(format!(
                "ideogram: packed {component} scales must be rank two"
            )));
        };
        let input = groups
            .checked_mul(candle_gen::quant::MLX_GROUP_SIZE)
            .ok_or_else(|| {
                gen_core::Error::Unsupported("ideogram: packed width overflow".into())
            })?;
        let bits = words
            .checked_mul(32)
            .and_then(|encoded| encoded.checked_div(input))
            .filter(|_| input > 0 && words * 32 % input == 0)
            .ok_or_else(|| {
                gen_core::Error::Unsupported("ideogram: cannot infer packed bits".into())
            })?;
        if rows != scale_rows
            || !matches!(bits, 4 | 8)
            || resolved_bits
                .replace(bits)
                .is_some_and(|prior| prior != bits)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "ideogram: mixed or invalid packed {component} geometry"
            )));
        }
    }
    Ok(Some(if resolved_bits == Some(4) {
        Quant::Q4
    } else {
        Quant::Q8
    }))
}

fn capture_component(
    spec: &LoadSpec,
    paths: &[PathBuf],
    tier: Option<Quant>,
    text_encoder: bool,
) -> gen_core::Result<Vec<FileReceipt>> {
    if paths.is_empty() {
        return Ok(Vec::new());
    }
    let mut all = BTreeMap::new();
    for path in paths {
        for header in gen_core::weightsmeta::safetensors_path_tensor_headers(path)? {
            all.insert(header.name.clone(), header);
        }
    }
    let packed = all
        .keys()
        .filter_map(|name| name.strip_suffix(".scales").map(str::to_owned))
        .collect::<BTreeSet<_>>();
    paths
        .iter()
        .map(|path| {
            FileReceipt::capture(spec, path, |header| {
                if header
                    .name
                    .strip_suffix(".scales")
                    .or_else(|| header.name.strip_suffix(".biases"))
                    .is_some_and(|base| packed.contains(base))
                {
                    return Ok(0);
                }
                if let Some(base) = header
                    .name
                    .strip_suffix(".weight")
                    .filter(|base| packed.contains(*base))
                {
                    let scales = all.get(&format!("{base}.scales")).ok_or_else(|| {
                        gen_core::Error::Unsupported("ideogram: missing packed scales".into())
                    })?;
                    let biases = all.get(&format!("{base}.biases")).ok_or_else(|| {
                        gen_core::Error::Unsupported("ideogram: missing packed biases".into())
                    })?;
                    return candle_gen::quant::mlx_packed_qtensor_resident_bytes(
                        header,
                        scales,
                        biases,
                        candle_gen::quant::MLX_GROUP_SIZE,
                    );
                }
                if tier.is_some() && header.dtype == gen_core::weightsmeta::Dtype::U32 {
                    return Err(gen_core::Error::Unsupported(
                        "ideogram: unrecognized packed U32 tensor".into(),
                    ));
                }
                let f32_normalization = text_encoder
                    && [
                        "input_layernorm.weight",
                        "post_attention_layernorm.weight",
                        "q_norm.weight",
                        "k_norm.weight",
                    ]
                    .iter()
                    .any(|suffix| header.name.ends_with(suffix));
                header.materialized_bytes(if f32_normalization { 4 } else { 2 })
            })
        })
        .collect()
}

fn f32_projection(
    header: &gen_core::weightsmeta::SafetensorsTensorHeader,
) -> gen_core::Result<u64> {
    use gen_core::weightsmeta::Dtype;
    if !matches!(header.dtype, Dtype::BF16 | Dtype::F16 | Dtype::F32) {
        return Err(gen_core::Error::Unsupported(format!(
            "ideogram: expected floating tensor {}, got {:?}",
            header.name, header.dtype
        )));
    }
    header.materialized_bytes(4)
}

fn capture_adapters(spec: &LoadSpec, base: bool) -> gen_core::Result<Vec<AdapterReceipt>> {
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
                "ideogram: adapter {order} requires finite uniform scale and no pass/MoE target"
            )));
            }
            let file = FileReceipt::capture(spec, &adapter.path, f32_projection)?;
            if !lexical.insert(file.lexical_path.clone())
                || !canonical.insert(file.canonical_path.clone())
            {
                return Err(gen_core::Error::Unsupported(
                    "ideogram: duplicate adapter source".into(),
                ));
            }
            let target_count = if base { 2 } else { 1 };
            let materialized_bytes = file
                .projected_resident_bytes
                .checked_mul(target_count)
                .ok_or_else(|| {
                    gen_core::Error::Unsupported("ideogram: adapter bytes overflow".into())
                })?;
            Ok(AdapterReceipt {
                order,
                kind: adapter.kind,
                scale_bits: adapter.scale.to_bits(),
                target_count: target_count as u8,
                materialized_bytes,
                file,
            })
        })
        .collect()
}

fn path_has_identity(path: &Path, repo: &str, revision: &str, trailing: &[&str]) -> bool {
    let hf = format!("models--SceneWorks--{repo}");
    let app = format!("SceneWorks__{repo}");
    let parts = path
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let mut suffixes = [
        vec![hf, "snapshots".into(), revision.into()],
        vec![app, revision.into()],
        vec![repo.into(), revision.into()],
    ];
    suffixes
        .iter_mut()
        .for_each(|suffix| suffix.extend(trailing.iter().map(|value| (*value).to_owned())));
    suffixes.iter().any(|suffix| parts.ends_with(suffix))
}

fn canonical_parent_has_identity(path: &Path, repo: &str, revision: &str) -> bool {
    path.parent()
        .and_then(|parent| std::fs::canonicalize(parent).ok())
        .is_some_and(|parent| path_has_identity(&parent, repo, revision, &[]))
}

fn canonical_dir_has_identity(path: &Path, repo: &str, revision: &str) -> bool {
    std::fs::canonicalize(path)
        .ok()
        .is_some_and(|canonical| path_has_identity(&canonical, repo, revision, &[]))
}

fn capture_pid(spec: &LoadSpec) -> gen_core::Result<Option<PidReceipt>> {
    let Some(pid) = &spec.pid else {
        return Ok(None);
    };
    let WeightsSource::File(student_path) = &pid.checkpoint else {
        return Err(gen_core::Error::Unsupported(
            "ideogram: PiD student must be one file".into(),
        ));
    };
    let WeightsSource::Dir(gemma_dir) = &pid.gemma else {
        return Err(gen_core::Error::Unsupported(
            "ideogram: PiD Gemma must be one directory".into(),
        ));
    };
    let name = student_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if !matches!(
        name,
        "pid_flux2_2k.safetensors"
            | "pid_flux2_2kto4k.safetensors"
            | "pid_flux2_2kto4k_v1pt5.safetensors"
    ) || !path_has_identity(student_path, "pid-flux2", PID_FLUX2_REVISION, &[name])
        || !canonical_parent_has_identity(student_path, "pid-flux2", PID_FLUX2_REVISION)
        || !path_has_identity(gemma_dir, "gemma-2-2b-it", PID_GEMMA_REVISION, &[])
        || !canonical_dir_has_identity(gemma_dir, "gemma-2-2b-it", PID_GEMMA_REVISION)
    {
        return Err(gen_core::Error::Unsupported(
            "ideogram: crossed PiD Flux2/Gemma identity".into(),
        ));
    }
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
    let materialized_bytes = student.projected_resident_bytes.saturating_add(
        gemma
            .iter()
            .map(|file| file.projected_resident_bytes)
            .sum::<u64>(),
    );
    Ok(Some(PidReceipt {
        student,
        gemma,
        tokenizer,
        materialized_bytes,
    }))
}

pub(crate) fn provider_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let receipt = IdeogramLoadReceipt::capture(provider_id, spec)?;
    Ok(contract_from_receipt(provider_id, spec, &receipt))
}

pub(crate) fn contract_from_receipt(
    provider_id: &str,
    spec: &LoadSpec,
    receipt: &IdeogramLoadReceipt,
) -> MemoryProviderContract {
    let mut resident_components = vec![MemoryResidentComponent {
        id: receipt.physical_identity(),
        kind: MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
        resident_bytes: receipt
            .components
            .conditional_transformer
            .saturating_add(receipt.components.unconditional_transformer),
        bounded_by: Some(MemoryStrategy::BoundedTransformerResidency),
        residency: MemoryComponentResidency::WholeRender,
    }];
    if let Some(file) = &receipt.turbo_adapter {
        resident_components.push(MemoryResidentComponent {
            id: receipt
                .turbo_adapter_identity()
                .expect("Turbo receipt carries mandatory adapter"),
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: file.projected_resident_bytes,
            bounded_by: Some(MemoryStrategy::StagedResidency),
            residency: MemoryComponentResidency::WholeRender,
        });
    }
    let user_bytes = receipt.adapters.iter().fold(0_u64, |total, adapter| {
        total.saturating_add(adapter.materialized_bytes)
    });
    if user_bytes > 0 {
        resident_components.push(MemoryResidentComponent {
            id: receipt
                .user_adapter_identity()
                .expect("non-empty user adapter receipt has identity"),
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: user_bytes,
            bounded_by: Some(MemoryStrategy::StagedResidency),
            residency: MemoryComponentResidency::WholeRender,
        });
    }
    if let Some(pid) = &receipt.pid {
        resident_components.push(MemoryResidentComponent {
            id: receipt.pid_identity().expect("PiD receipt has identity"),
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: pid.materialized_bytes,
            bounded_by: Some(MemoryStrategy::StagedResidency),
            residency: MemoryComponentResidency::WholeRender,
        });
    }
    build_contract(
        provider_id,
        spec,
        receipt.route.tier,
        receipt.components,
        resident_components,
    )
}

#[cfg(test)]
pub(crate) fn uncalibrated_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    if !matches!(provider_id, MODEL_ID | MODEL_ID_TURBO) {
        return Err(gen_core::Error::Unsupported(format!(
            "unknown Ideogram provider {provider_id}"
        )));
    }
    Ok(build_contract(
        provider_id,
        spec,
        physical_tier_hint(spec),
        ComponentBytes::default(),
        Vec::new(),
    ))
}

pub(crate) fn weights_free_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    if !matches!(provider_id, MODEL_ID | MODEL_ID_TURBO) {
        return Err(gen_core::Error::Unsupported(format!(
            "unknown Ideogram provider {provider_id}"
        )));
    }
    let tier = match spec.quantize {
        None => None,
        Some(Quant::Q4) => Some(Quant::Q4),
        Some(Quant::Q8) => Some(Quant::Q8),
        Some(Quant::Nvfp4) => {
            return Err(gen_core::Error::Unsupported(
                "Ideogram has no NVFP4 tier".into(),
            ))
        }
    };
    let mut normalized = spec.clone();
    normalized.resolved_route = Some(provider_id.to_owned());
    normalized.quantize = None;
    // The exhaustive registry witness must make rung four executable. Physical and surface
    // contracts preserve their actual load shape; this weights-free catalog fixture selects the
    // provider's deferred block-materialization route deliberately.
    normalized.load_shape = gen_core::LoadShape::DeferredMaterialization;
    Ok(build_contract(
        provider_id,
        &normalized,
        tier,
        ComponentBytes::default(),
        Vec::new(),
    ))
}

pub(crate) fn weights_free_surface_contract(
    provider_id: &str,
    surface: &gen_core::MemoryContractSurfaceSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let tier = match surface.resolved_artifact_tier() {
        gen_core::MemoryContractSurfaceTier::Bf16 => None,
        gen_core::MemoryContractSurfaceTier::Q4 => Some(Quant::Q4),
        gen_core::MemoryContractSurfaceTier::Q8 => Some(Quant::Q8),
        gen_core::MemoryContractSurfaceTier::Nvfp4 => {
            return Err(gen_core::Error::Unsupported(
                "Ideogram has no NVFP4 tier".into(),
            ))
        }
    };
    let mut spec = surface.spec.clone();
    spec.resolved_route = Some(provider_id.to_owned());
    spec.quantize = None;
    Ok(build_contract(
        provider_id,
        &spec,
        tier,
        ComponentBytes::default(),
        Vec::new(),
    ))
}

/// Activation dtype the loaded Ideogram 4 denoise trunk computes in. `pipeline.rs` pins
/// `DIT_DTYPE = DType::BF16` for the conditional DiT — the bottleneck this contract's phase
/// envelope prices — so this is the provider's real activation width, not a memory-model literal.
const ACTIVATION_DTYPE: candle_gen::candle_core::DType = candle_gen::candle_core::DType::BF16;

/// Architecture axes for the Ideogram 4 routes (epic SC-22657, E2).
///
/// The loader never parses `transformer/config.json`: `Ideogram4Transformer` is constructed from
/// the hardcoded [`Ideogram4DitConfig::v4`], and the latent / patch / decoder axes are the
/// `crate::config` constants the packing and decode paths execute against. Reading those same
/// declarations is what keeps the published geometry identical to the model actually built — a
/// snapshot `config.json` the loader ignores would describe a model this crate never constructs.
///
/// `vae_temporal_scale` stays `None`: Ideogram 4 decodes through the FLUX.2 `AutoencoderKLFlux2`
/// **image** autoencoder, which has no temporal axis at all. A structurally absent axis is declared
/// absent, never zero (E2).
fn architecture_facts(spec: &LoadSpec) -> gen_core::MemoryArchitectureFacts {
    use candle_gen::architecture_facts as af;

    // Weights-free contract surfaces name a sentinel path that is deliberately not on disk: no
    // pipeline has been resolved there, so no axis is knowable and every one stays `None`.
    if af::snapshot_root(spec).is_none() {
        return gen_core::MemoryArchitectureFacts::default();
    }
    let dit = Ideogram4DitConfig::v4();
    gen_core::MemoryArchitectureFacts {
        attention_heads: af::declared(dit.num_heads),
        head_dim: af::declared(dit.head_dim),
        transformer_blocks: af::declared(dit.num_layers),
        // `config::PATCH` — the DiT's 2x2 packing (`in_channels = LATENT_CHANNELS * PATCH²`).
        patch_size: af::declared(crate::config::PATCH),
        latent_channels: af::declared(crate::config::LATENT_CHANNELS),
        // `config::AE_SCALE` — the x8 spatial downscale of the FLUX.2 image autoencoder.
        vae_spatial_scale: af::declared(crate::config::AE_SCALE),
        // Structurally absent: an image autoencoder has no frames-per-latent axis to declare.
        vae_temporal_scale: None,
        activation_dtype_width: af::dtype_width(ACTIVATION_DTYPE),
    }
}

fn build_contract(
    provider_id: &str,
    spec: &LoadSpec,
    _tier: Option<Quant>,
    components: ComponentBytes,
    resident_components: Vec<MemoryResidentComponent>,
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
                MemoryStrategy::Resident
                | MemoryStrategy::StagedResidency
                | MemoryStrategy::BoundedDecode
                | MemoryStrategy::BoundedAttention
                | MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Implemented,
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
                MemoryStrategy::BoundedTransformerResidency => MemoryParameterRanges {
                    transformer_window_sizes: vec![TRANSFORMER_WINDOW_SIZE],
                    transformer_window_components: vec![TransformerComponent::Dit],
                    ..Default::default()
                },
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect();
    let transformer_bytes = components
        .conditional_transformer
        .saturating_add(components.unconditional_transformer);
    let overlay_bytes = resident_components
        .iter()
        .filter(|component| component.kind.is_auxiliary())
        .fold(0_u64, |total, component| {
            total.saturating_add(component.resident_bytes)
        });
    MemoryProviderContract {
        architecture_facts: architecture_facts(spec),
        provider_id: provider_id.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: true,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies,
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ]
        .into_iter()
        .map(|strategy| {
            (
                strategy,
                MemoryStrategyPrerequisite::Rung {
                    rung: MemoryStrategy::StagedResidency,
                    scope: MemoryPrerequisiteScope::EngagedInSameRequest,
                },
            )
        })
        .collect(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: gen_core::ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: true,
        },
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
                MemoryFormulaVariable::TransformerWindowSize,
            ],
            resident_components: resident_components.clone(),
        },
        calibration: None,
        asset_facts: MemoryAssetFacts {
            base_bytes: components
                .text_encoder
                .saturating_add(transformer_bytes)
                .saturating_add(components.vae),
            conditioning_bytes: components.text_encoder,
            transformer_bytes,
            decoder_bytes: components.vae,
            overlay_bytes,
        },
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    }
}

fn contract_overlay_identities(contract: &MemoryProviderContract) -> gen_core::Result<Vec<&str>> {
    let mut identities = contract
        .resident_components()
        .iter()
        .filter(|component| {
            component.id.starts_with(USER_ADAPTER_PREFIX)
                || component.id.starts_with(TURBO_ADAPTER_PREFIX)
                || component.id.starts_with(PID_PREFIX)
        })
        .map(|component| component.id.as_str())
        .collect::<Vec<_>>();
    identities.sort_unstable();
    let unique = identities.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != identities.len() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: duplicate physical overlay identities",
            contract.provider_id
        )));
    }
    Ok(identities)
}

fn expected_public_overlay(contract: &MemoryProviderContract) -> gen_core::Result<Option<String>> {
    let identities = contract_overlay_identities(contract)?;
    let user = identities
        .iter()
        .copied()
        .filter(|identity| identity.starts_with(USER_ADAPTER_PREFIX))
        .collect::<Vec<_>>();
    if user.len() > 1 {
        return Err(gen_core::Error::Unsupported(
            "ideogram: multiple user adapter identities".into(),
        ));
    }
    Ok(user.first().map(|value| (*value).to_owned()))
}

fn validate_route(
    provider_id: &str,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    let refs = context.geometry.reference_count;
    let valid_mode = match context.mode {
        MemoryMode::TextToImage => refs <= 1,
        MemoryMode::Edit => matches!(refs, 1 | 2),
        // SceneWorks reserves ImageToImage for the internally derived Hires refinement pass. Its
        // exact geometry is request-bound but may exceed the five public first-pass buckets.
        MemoryMode::ImageToImage => refs == 1,
        _ => false,
    };
    let (width, height) = (context.geometry.width, context.geometry.height);
    let public_geometry = PUBLIC_GEOMETRIES.contains(&(width, height));
    let hires_geometry = context.mode == MemoryMode::ImageToImage
        && width >= 256
        && height >= 256
        && width <= 8192
        && height <= 8192
        && width % 16 == 0
        && height % 16 == 0
        && width.max(height) <= width.min(height).saturating_mul(6);
    if !valid_mode
        || context.has_reference != (refs > 0)
        || context.geometry.frames != 1
        || !matches!(context.geometry.batch, 1 | 2 | 4)
        || !(public_geometry || hires_geometry)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: exact Ideogram route requires public T2I refs=0/1 or Edit refs=1/2, or one internally admitted Hires ImageToImage reference, count 1/2/4, and one image frame"
        )));
    }
    if context.overlay != expected_public_overlay(contract)? {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: public adapter overlay crosses the sealed ordered load receipt"
        )));
    }
    let identities = contract_overlay_identities(contract)?;
    let has_pid_receipt = identities
        .iter()
        .any(|identity| identity.starts_with(PID_PREFIX));
    let native_hires_over_pid_load =
        context.mode == MemoryMode::ImageToImage && !context.use_pid && has_pid_receipt;
    if context.use_pid != has_pid_receipt && !native_hires_over_pid_load {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: native/PiD request crosses its sealed decoder receipt"
        )));
    }
    let has_turbo = identities
        .iter()
        .any(|identity| identity.starts_with(TURBO_ADAPTER_PREFIX));
    let weights_free_witness = contract.asset_facts == MemoryAssetFacts::default();
    if (provider_id == MODEL_ID_TURBO) != has_turbo
        && !(weights_free_witness && provider_id == MODEL_ID_TURBO && !has_turbo)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: Turbo mandatory adapter identity is absent or crossed"
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
    if contract.calibration.is_none()
        && (context.calibration_abi != 0
            || !context.calibration_fingerprint.is_empty()
            || context.load_shape != contract.load_shape)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: structural-estimate handshake crossed ABI, fingerprint, or load shape"
        )));
    }
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
    if context.use_pid
        && contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: PiD has no admitted bounded native-VAE decode plan"
        )));
    }
    Ok(())
}

pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match registered_tier_and_contract(spec, contract)
        .and_then(|tier| validate_context(&contract.provider_id, contract, context, tier))
    {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

fn physical_tier_hint(spec: &LoadSpec) -> Option<Quant> {
    match &spec.weights {
        WeightsSource::Dir(root) => match root.file_name().and_then(|name| name.to_str()) {
            Some("q4") => Some(Quant::Q4),
            Some("q8") => Some(Quant::Q8),
            _ => None,
        },
        WeightsSource::File(_) => None,
    }
}

fn registered_tier_and_contract(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
) -> gen_core::Result<Option<Quant>> {
    let provider_id = contract.provider_id.as_str();
    validate_load_shape(provider_id, spec)?;
    let WeightsSource::Dir(root) = &spec.weights else {
        unreachable!("validated")
    };
    let route = route_identity(provider_id, root)?;
    if contract.asset_facts == MemoryAssetFacts::default() {
        // The registry witness has no installed files to canonicalize, but it still carries and
        // validates the exact provider/repository/revision/tier suffix. Physical contracts take
        // the stronger canonical-path branch below.
        validate_snapshot_path(root, route)?;
        let expected = weights_free_contract(provider_id, spec)?;
        if expected != *contract {
            return Err(gen_core::Error::Unsupported(format!(
                "{provider_id}: caller contract differs from registry witness"
            )));
        }
        Ok(physical_tier_hint(spec))
    } else {
        validate_snapshot_binding(root, route)?;
        let receipt = IdeogramLoadReceipt::capture(provider_id, spec)?;
        let expected = contract_from_receipt(provider_id, spec, &receipt);
        if expected != *contract {
            return Err(gen_core::Error::Unsupported(format!(
                "{provider_id}: caller contract differs from sealed physical receipt"
            )));
        }
        receipt.ensure_unchanged()?;
        Ok(receipt.route.tier)
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
    seed: Option<u64>,
    steps: Option<u32>,
    guidance_bits: Option<u32>,
    sampler: Option<String>,
    scheduler: Option<String>,
    shift_bits: Option<u32>,
    preview_active: bool,
    carrier: IdeogramCarrier,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IdeogramCarrier {
    PromptOnly,
    Reference,
    ReferenceAndMask,
}

fn request_carrier(request: &GenerationRequest) -> gen_core::Result<IdeogramCarrier> {
    let mut references = 0_u32;
    let mut masks = 0_u32;
    for conditioning in &request.conditioning {
        match conditioning {
            Conditioning::Reference { .. } => references += 1,
            Conditioning::Mask { .. } => masks += 1,
            _ => {
                return Err(gen_core::Error::Unsupported(
                    "ideogram: admitted request contains a non-Reference/Mask carrier".into(),
                ))
            }
        }
    }
    match (references, masks) {
        (0, 0) => Ok(IdeogramCarrier::PromptOnly),
        (1, 0) => Ok(IdeogramCarrier::Reference),
        (1, 1) => Ok(IdeogramCarrier::ReferenceAndMask),
        _ => Err(gen_core::Error::Unsupported(format!(
            "ideogram: exact carrier requires none, one Reference, or Reference+Mask; got {references} Reference and {masks} Mask"
        ))),
    }
}

fn carrier_matches_context(carrier: IdeogramCarrier, context: &MemoryRunContext) -> bool {
    matches!(
        (&context.mode, context.geometry.reference_count, carrier),
        (MemoryMode::TextToImage, 0, IdeogramCarrier::PromptOnly)
            | (MemoryMode::TextToImage, 1, IdeogramCarrier::Reference)
            | (MemoryMode::Edit, 1, IdeogramCarrier::Reference)
            | (MemoryMode::Edit, 2, IdeogramCarrier::ReferenceAndMask)
            | (MemoryMode::ImageToImage, 1, IdeogramCarrier::Reference)
    )
}

impl RequestBinding {
    fn from_request(request: &GenerationRequest) -> gen_core::Result<Self> {
        Ok(Self {
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
            seed: request.seed,
            steps: request.steps,
            guidance_bits: request.guidance.map(f32::to_bits),
            sampler: request.sampler.clone(),
            scheduler: request.scheduler.clone(),
            shift_bits: request.scheduler_shift.map(f32::to_bits),
            preview_active: request.preview.is_active(),
            carrier: request_carrier(request)?,
        })
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
        let mut state = candle_gen::lock_recover(&self.inner);
        if contract.provider_id != self.provider_id || state.active.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: crossed provider or active memory scope",
                self.provider_id
            )));
        }
        let approved = state.approved_context.take().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{}: request skipped the safety handshake",
                self.provider_id
            ))
        })?;
        if approved != *context {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: context changed after safety approval",
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
                "{}: memory scope is no longer active",
                self.provider_id
            ))
        })?;
        let binding = RequestBinding::from_request(request)?;
        if active.token != token
            || active.binding.is_some()
            || active.consumed
            || binding.geometry != active.context.geometry
            || binding.memory != active.expected_memory
            || binding.use_pid != active.context.use_pid
            || binding.has_phases != active.context.has_phases
            || !carrier_matches_context(binding.carrier, &active.context)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: stale or changed admitted request",
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
        if active.binding.as_ref() != Some(&RequestBinding::from_request(request)?)
            || active.consumed
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request changed or admission was consumed",
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
                "{}: stale token cannot finish",
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

pub(crate) struct IdeogramMemoryScope {
    core: candle_gen::request_scope::CandleRequestScopeCore,
    admission: AdmissionRegistry,
    token: u64,
    finished: bool,
}

impl IdeogramMemoryScope {
    pub(crate) fn new_bound(
        provider_id: &'static str,
        device: Device,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
        admission: AdmissionRegistry,
    ) -> gen_core::Result<Self> {
        let token = admission.begin(contract, context)?;
        Ok(Self {
            core: request_scope(provider_id, device, contract, context)?,
            admission,
            token,
            finished: false,
        })
    }
}

impl MemoryRequestScope for IdeogramMemoryScope {
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

impl Drop for IdeogramMemoryScope {
    fn drop(&mut self) {
        if !self.finished {
            self.admission.abandon(self.token);
        }
    }
}

fn request_scope(
    provider_id: &'static str,
    device: Device,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<candle_gen::request_scope::CandleRequestScopeCore> {
    let mut config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        provider_id,
        device,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        Ideogram4DitConfig::v4().num_layers,
        move |use_pid, edge, overlap| {
            if !use_pid && edge == DECODE_TILE_EDGE && overlap == DECODE_OVERLAP {
                Ok(())
            } else {
                Err(gen_core::Error::Unsupported(format!(
                    "{provider_id}: native bounded decode requires {DECODE_TILE_EDGE}/{DECODE_OVERLAP} and is unavailable for PiD"
                )))
            }
        },
    )?;
    if contract.engages(context.selection.strategy, MemoryStrategy::BoundedAttention) {
        config.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE);
    }
    if contract.engages(
        context.selection.strategy,
        MemoryStrategy::BoundedTransformerResidency,
    ) {
        config.transformer_window = Some(TRANSFORMER_WINDOW_SIZE);
    }
    Ok(candle_gen::request_scope::CandleRequestScopeCore::new(
        config,
    ))
}

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

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<MemoryBehaviorFixture>> {
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
    let exact_spec = weights_free_behavior_spec(&contract.provider_id, spec)?;
    let tier = physical_tier_hint(&exact_spec);
    for route in [
        MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::Edit,
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::Edit,
            reference_count: 2,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    ] {
        let context = estimated_behavior_context(
            contract,
            strategy,
            MemoryNumericTier {
                precision: Precision::Bf16,
                quant: tier,
                component_precision_floors: &[],
            },
            route,
        )?;
        if validate_route(&contract.provider_id, contract, &context).is_ok() {
            return Ok(vec![
                MemoryBehaviorFixture::new(context).with_load_spec(exact_spec)
            ]);
        }
    }
    Ok(Vec::new())
}

fn weights_free_behavior_spec(provider_id: &str, spec: &LoadSpec) -> gen_core::Result<LoadSpec> {
    if !matches!(provider_id, MODEL_ID | MODEL_ID_TURBO) {
        return Err(gen_core::Error::Unsupported(format!(
            "unknown Ideogram provider {provider_id}"
        )));
    }
    // The catalog-wide probe starts from a generic directory. Bind its positive control to the
    // exact dense route; packed q4/q8 routes are exercised by the surface registry.
    let mut exact = spec.clone();
    exact.weights = WeightsSource::Dir(
        PathBuf::from(format!("models--SceneWorks--{BF16_REPOSITORY}"))
            .join("snapshots")
            .join(BF16_REVISION)
            .join("bf16"),
    );
    exact.resolved_route = Some(provider_id.to_owned());
    exact.precision = Precision::Bf16;
    exact.quantize = None;
    exact.load_shape = gen_core::LoadShape::DeferredMaterialization;
    Ok(exact)
}

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
            total_bytes: 96 * 1024 * 1024 * 1024,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 64 * 1024 * 1024 * 1024,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: "ideogram-candle-structural-estimate".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{
        AdapterSpec, LoadShape, MemoryBudget, MemoryCacheState, MemorySelection,
        MemoryStrategyParameters, OffloadPolicy,
    };

    /// AC (epic SC-22657, E2): both Ideogram providers publish the geometry the loader actually
    /// builds — the `Ideogram4DitConfig::v4` trunk plus the `crate::config` latent/patch/decoder
    /// constants — the contract passes the shared facts conformance check, and the weights-free
    /// surface (whose sentinel root is not on disk) publishes nothing at all.
    #[test]
    fn architecture_facts_match_the_loader_config_and_pass_conformance() {
        let tmp = tempfile::tempdir().unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir(tmp.path().to_path_buf()));
        spec.load_shape = LoadShape::DeferredMaterialization;
        for id in [MODEL_ID, MODEL_ID_TURBO] {
            let contract = uncalibrated_contract(id, &spec).unwrap();
            assert_eq!(
                contract.architecture_facts,
                gen_core::MemoryArchitectureFacts {
                    // `Ideogram4DitConfig::v4`: num_heads / head_dim / num_layers.
                    attention_heads: Some(18),
                    head_dim: Some(256),
                    transformer_blocks: Some(34),
                    // `config::PATCH`.
                    patch_size: Some(2),
                    // `config::LATENT_CHANNELS` (32-ch `AutoencoderKLFlux2`).
                    latent_channels: Some(32),
                    // `config::AE_SCALE`.
                    vae_spatial_scale: Some(8),
                    // Structurally absent: the FLUX.2 image autoencoder has no temporal axis.
                    vae_temporal_scale: None,
                    // `pipeline.rs: DIT_DTYPE = DType::BF16`.
                    activation_dtype_width: Some(2),
                },
                "{id} architecture facts"
            );
            assert!(contract.architecture_facts.has_snapshot_read_axis());
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);

            // The registry's weights-free surface resolves nothing on disk, so no axis is knowable.
            let weights_free = LoadSpec::new(WeightsSource::Dir(
                "/__sceneworks_memory_contract_surface__".into(),
            ));
            let contract = weights_free_contract(id, &weights_free).unwrap();
            assert!(
                contract.architecture_facts.is_empty(),
                "{id} weights-free facts must be empty"
            );
            // A weights-free contract legitimately declares nothing, so the E2 snapshot-read gate
            // does not apply to it; the byte-decomposition half of the conformance walk still does.
            gen_core_testkit::assert_memory_contract_asset_facts_conform(&contract);
        }
    }

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

    fn component(path: &Path, tier: &str, prefix: &str) {
        let norm = if prefix == "text" {
            "language_model.layers.0.input_layernorm.weight".to_owned()
        } else {
            format!("{prefix}.norm.weight")
        };
        match tier {
            "q4" | "q8" => {
                let bits = if tier == "q4" { 4 } else { 8 };
                let columns = if bits == 4 { 8 } else { 16 };
                write_safetensors(
                    path,
                    &[
                        (
                            Box::leak(format!("{prefix}.proj.weight").into_boxed_str()),
                            "U32",
                            &[2, columns],
                            8 * columns,
                        ),
                        (
                            Box::leak(format!("{prefix}.proj.scales").into_boxed_str()),
                            "BF16",
                            &[2, 1],
                            4,
                        ),
                        (
                            Box::leak(format!("{prefix}.proj.biases").into_boxed_str()),
                            "BF16",
                            &[2, 1],
                            4,
                        ),
                        (Box::leak(norm.clone().into_boxed_str()), "BF16", &[2], 4),
                    ],
                );
            }
            "bf16" => write_safetensors(
                path,
                &[
                    (
                        Box::leak(format!("{prefix}.proj.weight").into_boxed_str()),
                        "BF16",
                        &[2, 64],
                        256,
                    ),
                    (Box::leak(norm.into_boxed_str()), "BF16", &[2], 4),
                ],
            ),
            _ => unreachable!(),
        }
    }

    fn fixture_root(tmp: &Path, provider: &str, tier: &str) -> PathBuf {
        let (repo, revision) = if tier == "bf16" {
            (BF16_REPOSITORY, BF16_REVISION)
        } else {
            (PACKED_REPOSITORY, PACKED_REVISION)
        };
        let root = tmp
            .join(format!("models--SceneWorks--{repo}"))
            .join("snapshots")
            .join(revision)
            .join(tier);
        component(&root.join("transformer/model.safetensors"), tier, "cond");
        component(&root.join("text_encoder/model.safetensors"), tier, "text");
        if provider == MODEL_ID {
            component(
                &root.join("unconditional_transformer/model.safetensors"),
                tier,
                "uncond",
            );
        } else {
            write_safetensors(
                &root.join(TURBO_LORA_FILE),
                &[
                    ("layers.0.lora_down.weight", "F32", &[2, 2], 16),
                    ("layers.0.lora_up.weight", "F32", &[2, 2], 16),
                ],
            );
        }
        write_safetensors(
            &root.join("vae/model.safetensors"),
            &[("decoder.weight", "F32", &[2, 2], 16)],
        );
        std::fs::create_dir_all(root.join("tokenizer")).unwrap();
        std::fs::write(root.join("tokenizer/tokenizer.json"), b"{}").unwrap();
        std::fs::write(root.join("transformer/config.json"), b"{}").unwrap();
        if provider == MODEL_ID {
            std::fs::write(root.join("unconditional_transformer/config.json"), b"{}").unwrap();
        }
        root
    }

    fn spec(root: PathBuf, provider: &'static str) -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir(root)).with_resolved_route(provider);
        spec.load_shape = LoadShape::DeferredMaterialization;
        spec.offload_policy = OffloadPolicy::Sequential;
        spec
    }

    #[test]
    fn registry_behavior_fixtures_cover_every_rung_on_exact_dense_routes() {
        let generic = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        for provider in [MODEL_ID, MODEL_ID_TURBO] {
            let contract = weights_free_contract(provider, &generic).unwrap();
            for strategy in MemoryStrategy::ALL
                .into_iter()
                .filter(|strategy| strategy.is_optimized())
            {
                let fixture = registered_valid_fixture(&generic, &contract, strategy)
                    .unwrap()
                    .pop()
                    .unwrap_or_else(|| panic!("{provider} lacks {strategy:?} fixture"));
                let exact = fixture
                    .load_spec
                    .as_ref()
                    .expect("provider-owned load spec");
                assert_eq!(exact.resolved_route.as_deref(), Some(provider));
                assert_eq!(physical_tier_hint(exact), None);
                let decision = registered_safety_check(exact, &contract, &fixture.context);
                assert!(
                    matches!(decision, MemorySafetyDecision::Accept),
                    "{provider}:{strategy:?}: {decision:?}"
                );
                let mut crossed = fixture.context.clone();
                crossed.calibration_abi = gen_core::MEMORY_CALIBRATION_ABI;
                assert!(matches!(
                    registered_safety_check(exact, &contract, &crossed),
                    MemorySafetyDecision::Reject { .. }
                ));
            }
        }
    }

    fn context(
        contract: &MemoryProviderContract,
        mode: MemoryMode,
        refs: u32,
        overlay: Option<String>,
        use_pid: bool,
    ) -> MemoryRunContext {
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
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: refs,
            },
            mode,
            budget: MemoryBudget {
                total_bytes: u64::MAX,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "fixture".into(),
            has_reference: refs > 0,
            use_pid,
            has_phases: false,
            overlay,
        }
    }

    #[test]
    fn base_and_turbo_q4_q8_bf16_receipts_are_nonzero_and_differentiated() {
        for provider in [MODEL_ID, MODEL_ID_TURBO] {
            let mut prior = 0;
            for (tier, quant) in [
                ("q4", Some(Quant::Q4)),
                ("q8", Some(Quant::Q8)),
                ("bf16", None),
            ] {
                let tmp = tempfile::tempdir().unwrap();
                let root = fixture_root(tmp.path(), provider, tier);
                let receipt =
                    IdeogramLoadReceipt::capture(provider, &spec(root, provider)).unwrap();
                assert_eq!(receipt.route.tier, quant);
                assert!(receipt.components.text_encoder > 0);
                assert!(receipt.components.conditional_transformer > 0);
                assert!(receipt.components.vae > 0);
                if provider == MODEL_ID {
                    assert!(receipt.components.unconditional_transformer > 0);
                    assert!(receipt.turbo_adapter.is_none());
                } else {
                    assert_eq!(receipt.components.unconditional_transformer, 0);
                    assert!(receipt.turbo_adapter.is_some());
                }
                let total = receipt.components.text_encoder
                    + receipt.components.conditional_transformer
                    + receipt.components.unconditional_transformer;
                assert!(
                    total > prior,
                    "{provider} {tier} must exceed its lower tier"
                );
                prior = total;
            }
        }
    }

    #[test]
    fn route_repository_revision_tier_and_mandatory_adapter_crossing_fail_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let root = fixture_root(tmp.path(), MODEL_ID_TURBO, "q4");
        std::fs::remove_file(root.join(TURBO_LORA_FILE)).unwrap();
        assert!(IdeogramLoadReceipt::capture(MODEL_ID_TURBO, &spec(root, MODEL_ID_TURBO)).is_err());

        let tmp = tempfile::tempdir().unwrap();
        let root = fixture_root(tmp.path(), MODEL_ID, "q8");
        let crossed = tmp
            .path()
            .join(format!("models--SceneWorks--{PACKED_REPOSITORY}"))
            .join("snapshots/not-the-revision/q8");
        std::fs::create_dir_all(crossed.parent().unwrap()).unwrap();
        std::fs::rename(&root, &crossed).unwrap();
        assert!(IdeogramLoadReceipt::capture(MODEL_ID, &spec(crossed, MODEL_ID)).is_err());

        let tmp = tempfile::tempdir().unwrap();
        let root = fixture_root(tmp.path(), MODEL_ID, "bf16");
        let mut crossed = spec(root, MODEL_ID);
        crossed.quantize = Some(Quant::Q4);
        assert!(IdeogramLoadReceipt::capture(MODEL_ID, &crossed).is_err());
    }

    #[test]
    fn same_length_mutation_and_direct_inventory_drift_are_detected_but_nested_ignored_files_are_not(
    ) {
        let tmp = tempfile::tempdir().unwrap();
        let root = fixture_root(tmp.path(), MODEL_ID, "bf16");
        write_safetensors(
            &root.join("transformer/nested/ignored.safetensors"),
            &[("ignored", "BF16", &[2], 4)],
        );
        let receipt =
            IdeogramLoadReceipt::capture(MODEL_ID, &spec(root.clone(), MODEL_ID)).unwrap();
        let original_identity = receipt.physical_identity();
        std::fs::write(root.join("transformer/nested/note.txt"), b"ignored too").unwrap();
        receipt.ensure_unchanged().unwrap();

        let path = root.join("transformer/model.safetensors");
        let mut bytes = std::fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 0x5a;
        std::fs::write(&path, bytes).unwrap();
        assert!(receipt.ensure_unchanged().is_err());
        let replacement =
            IdeogramLoadReceipt::capture(MODEL_ID, &spec(root.clone(), MODEL_ID)).unwrap();
        assert_ne!(replacement.physical_identity(), original_identity);

        let tmp = tempfile::tempdir().unwrap();
        let root = fixture_root(tmp.path(), MODEL_ID, "bf16");
        let receipt =
            IdeogramLoadReceipt::capture(MODEL_ID, &spec(root.clone(), MODEL_ID)).unwrap();
        write_safetensors(
            &root.join("transformer/extra.safetensors"),
            &[("extra", "BF16", &[2], 4)],
        );
        assert!(receipt.ensure_unchanged().is_err());
    }

    #[test]
    fn ordered_user_adapters_charge_both_base_stacks_but_one_turbo_stack() {
        let tmp = tempfile::tempdir().unwrap();
        let adapter = tmp.path().join("adapter.safetensors");
        write_safetensors(&adapter, &[("layer.lora_A.weight", "F16", &[2, 2], 8)]);
        for (provider, targets) in [(MODEL_ID, 2_u8), (MODEL_ID_TURBO, 1_u8)] {
            let root = fixture_root(&tmp.path().join(provider), provider, "q4");
            let mut load = spec(root, provider);
            load.adapters = vec![AdapterSpec::new(adapter.clone(), 0.75, AdapterKind::Lora)];
            let receipt = IdeogramLoadReceipt::capture(provider, &load).unwrap();
            assert_eq!(receipt.adapters[0].target_count, targets);
            assert_eq!(
                receipt.adapters[0].materialized_bytes,
                receipt.adapters[0].file.projected_resident_bytes * u64::from(targets)
            );
            assert!(receipt.user_adapter_identity().is_some());
        }
    }

    #[test]
    fn exact_public_modes_geometries_overlays_and_all_rungs_fail_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let root = fixture_root(tmp.path(), MODEL_ID_TURBO, "q4");
        let load = spec(root, MODEL_ID_TURBO);
        let receipt = IdeogramLoadReceipt::capture(MODEL_ID_TURBO, &load).unwrap();
        let contract = contract_from_receipt(MODEL_ID_TURBO, &load, &receipt);
        for (mode, refs) in [
            (MemoryMode::TextToImage, 0),
            (MemoryMode::TextToImage, 1),
            (MemoryMode::Edit, 1),
            (MemoryMode::Edit, 2),
            (MemoryMode::ImageToImage, 1),
        ] {
            let exact = context(&contract, mode, refs, None, false);
            assert!(validate_context(MODEL_ID_TURBO, &contract, &exact, Some(Quant::Q4)).is_ok());
        }
        for (mode, refs) in [
            (MemoryMode::TextToImage, 2),
            (MemoryMode::Edit, 0),
            (MemoryMode::Edit, 3),
            (MemoryMode::Other("image_to_video".into()), 1),
        ] {
            assert!(validate_context(
                MODEL_ID_TURBO,
                &contract,
                &context(&contract, mode, refs, None, false),
                Some(Quant::Q4),
            )
            .is_err());
        }
        let mut bad_geometry = context(&contract, MemoryMode::TextToImage, 0, None, false);
        bad_geometry.geometry.width = 1008;
        assert!(
            validate_context(MODEL_ID_TURBO, &contract, &bad_geometry, Some(Quant::Q4)).is_err()
        );
        let mut hires = context(&contract, MemoryMode::ImageToImage, 1, None, false);
        hires.geometry.width = 2048;
        hires.geometry.height = 2048;
        assert!(validate_context(MODEL_ID_TURBO, &contract, &hires, Some(Quant::Q4)).is_ok());
        hires.geometry.width = 2050;
        assert!(validate_context(MODEL_ID_TURBO, &contract, &hires, Some(Quant::Q4)).is_err());
        let crossed_overlay = context(
            &contract,
            MemoryMode::TextToImage,
            0,
            Some("lora".into()),
            false,
        );
        assert!(
            validate_context(MODEL_ID_TURBO, &contract, &crossed_overlay, Some(Quant::Q4)).is_err()
        );
        for implemented in [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                contract.capability(implemented).unwrap().support,
                MemoryStrategySupport::Implemented
            );
        }
        assert!(contract.lifecycle.decode_tiling);
        assert!(contract.lifecycle.attention_chunking);
        assert!(contract.lifecycle.transformer_window_materialization);
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .parameters
                .decode_tile_edges,
            vec![DECODE_TILE_EDGE]
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .parameters
                .attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .parameters
                .transformer_window_sizes,
            vec![TRANSFORMER_WINDOW_SIZE]
        );
        let selection = MemorySelection {
            strategy: MemoryStrategy::BoundedTransformerResidency,
            parameters: MemoryStrategyParameters {
                decode_tile_edge: Some(DECODE_TILE_EDGE),
                decode_overlap: Some(DECODE_OVERLAP),
                attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
                transformer_window_component: Some(TransformerComponent::Dit),
            },
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: &[],
            },
        };
        let memory = contract
            .generation_memory(&selection)
            .expect("bounded transformer execution controls");
        assert!(memory.stage_residency && memory.tile_vae_decode);
        assert!(memory.chunk_attention && memory.stream_transformer_blocks);
        assert_eq!(memory.decode_tile_edge, Some(DECODE_TILE_EDGE));
        assert_eq!(memory.decode_overlap, Some(DECODE_OVERLAP));
        assert_eq!(memory.attention_chunk_size, Some(ATTENTION_CHUNK_SIZE));
        assert_eq!(
            memory.transformer_window_size,
            Some(TRANSFORMER_WINDOW_SIZE)
        );
        assert_eq!(
            memory.transformer_window_component,
            Some(TransformerComponent::Dit)
        );
    }

    #[test]
    fn f32_text_normalization_is_accounted_at_execution_width() {
        let tmp = tempfile::tempdir().unwrap();
        let root = fixture_root(tmp.path(), MODEL_ID_TURBO, "bf16");
        let receipt =
            IdeogramLoadReceipt::capture(MODEL_ID_TURBO, &spec(root, MODEL_ID_TURBO)).unwrap();
        // Dense projection: 2x64 bf16 = 256 B. The RMSNorm is loaded by `rms_norm_f32`, so its
        // two elements occupy 8 B rather than the four container bytes.
        assert_eq!(receipt.components.text_encoder, 264);
    }

    #[cfg(unix)]
    #[test]
    fn canonical_snapshot_and_pid_provenance_reject_exact_looking_symlink_roots() {
        use std::os::unix::fs::symlink;

        for provider in [MODEL_ID, MODEL_ID_TURBO] {
            let tmp = tempfile::tempdir().unwrap();
            let foreign = fixture_root(&tmp.path().join("foreign"), provider, "q4");
            let lexical = tmp
                .path()
                .join(format!("models--SceneWorks--{PACKED_REPOSITORY}"))
                .join("snapshots")
                .join(PACKED_REVISION)
                .join("q4");
            std::fs::create_dir_all(lexical.parent().unwrap()).unwrap();
            symlink(&foreign, &lexical).unwrap();
            assert!(IdeogramLoadReceipt::capture(provider, &spec(lexical, provider)).is_err());
        }

        let tmp = tempfile::tempdir().unwrap();
        let foreign_student_dir = tmp.path().join("foreign-pid");
        let foreign_student = foreign_student_dir.join("pid_flux2_2k.safetensors");
        write_safetensors(&foreign_student, &[("student.weight", "F32", &[2], 8)]);
        let lexical_student_dir = tmp
            .path()
            .join("models--SceneWorks--pid-flux2/snapshots")
            .join(PID_FLUX2_REVISION);
        std::fs::create_dir_all(lexical_student_dir.parent().unwrap()).unwrap();
        symlink(&foreign_student_dir, &lexical_student_dir).unwrap();
        let gemma = tmp
            .path()
            .join("models--SceneWorks--gemma-2-2b-it/snapshots")
            .join(PID_GEMMA_REVISION);
        write_safetensors(
            &gemma.join("gemma-2-2b-it.safetensors"),
            &[("gemma.weight", "F32", &[2], 8)],
        );
        std::fs::write(gemma.join("tokenizer.json"), b"{}").unwrap();
        let mut load = spec(fixture_root(tmp.path(), MODEL_ID, "bf16"), MODEL_ID);
        load.pid = Some(gen_core::PidWeights {
            checkpoint: WeightsSource::File(lexical_student_dir.join("pid_flux2_2k.safetensors")),
            gemma: WeightsSource::Dir(gemma),
        });
        assert!(capture_pid(&load).is_err());
    }

    #[test]
    fn admission_binds_typed_reference_mask_carrier_pid_and_pass_axes() {
        let tmp = tempfile::tempdir().unwrap();
        let root = fixture_root(tmp.path(), MODEL_ID, "q4");
        let load = spec(root, MODEL_ID);
        let receipt = IdeogramLoadReceipt::capture(MODEL_ID, &load).unwrap();
        let contract = contract_from_receipt(MODEL_ID, &load, &receipt);
        let context = context(&contract, MemoryMode::Edit, 2, None, false);
        let admission = AdmissionRegistry::new(MODEL_ID);
        admission.approve(&context).unwrap();
        let mut scope =
            IdeogramMemoryScope::new_bound(MODEL_ID, Device::Cpu, &contract, &context, admission)
                .unwrap();
        let image = gen_core::Image {
            width: 1,
            height: 1,
            pixels: vec![0, 0, 0],
        };
        let mut crossed = GenerationRequest {
            prompt: "caption".into(),
            width: 1024,
            height: 1024,
            count: 1,
            conditioning: vec![
                Conditioning::Reference {
                    image: image.clone(),
                    strength: None,
                },
                Conditioning::Reference {
                    image,
                    strength: None,
                },
            ],
            ..Default::default()
        };
        crossed.memory = contract.generation_memory(&context.selection);
        assert!(scope.configure_request(&mut crossed).is_err());
    }

    #[test]
    fn staged_handshake_binds_request_and_cleans_up_on_finish_or_drop() {
        let provider = MODEL_ID;
        let mut surface = gen_core::candle_memory_contract_surface_specs()
            .into_iter()
            .find(|surface| {
                surface.selector.tier == gen_core::MemoryContractSurfaceTier::Q4
                    && surface.selector.load_shape == LoadShape::DeferredMaterialization
            })
            .unwrap();
        surface.spec.resolved_route = Some(provider.into());
        let contract = weights_free_surface_contract(provider, &surface).unwrap();
        let context = context(&contract, MemoryMode::TextToImage, 0, None, false);
        let admission = AdmissionRegistry::new(provider);
        let mut request = GenerationRequest {
            prompt: "structured caption".into(),
            width: 1024,
            height: 1024,
            count: 1,
            ..Default::default()
        };
        request.memory = contract.generation_memory(&context.selection);
        assert!(admission.consume_for_generate(&request).is_err());

        admission.approve(&context).unwrap();
        {
            let mut scope = IdeogramMemoryScope::new_bound(
                provider,
                Device::Cpu,
                &contract,
                &context,
                admission.clone(),
            )
            .unwrap();
            scope.configure_request(&mut request).unwrap();
            admission.consume_for_generate(&request).unwrap();
            scope.finish(MemoryRunOutcome::Complete).unwrap();
        }
        admission.approve(&context).unwrap();
        drop(
            IdeogramMemoryScope::new_bound(
                provider,
                Device::Cpu,
                &contract,
                &context,
                admission.clone(),
            )
            .unwrap(),
        );
        admission.approve(&context).unwrap();
    }
}
