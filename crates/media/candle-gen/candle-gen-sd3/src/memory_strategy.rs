//! Exact request-scoped Candle/CUDA memory contract for the three shipped SD3.5 routes.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
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
use candle_gen::gen_core::{
    MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryBudget, MemoryCacheState,
    MemoryOptimizationAuthority,
};
use sha2::{Digest, Sha256};

pub const REQUEST_EVIDENCE_REVISION: &str = "sd3.5-candle-request-contract-v1";
pub const PHYSICAL_RECEIPT_PREFIX: &str = "sd3.5.candle.physical.sha256:";
pub const ADAPTER_RECEIPT_PREFIX: &str = "sd3.5.adapters.ordered-additive.sha256:";

const COMMON_TIER_FILES: &[&str] = &[
    "LICENSE.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "text_encoder_2/model.safetensors",
    "text_encoder_3/config.json",
    "text_encoder_3/model-00001-of-00002.safetensors",
    "text_encoder_3/model-00002-of-00002.safetensors",
    "text_encoder_3/model.fp16-00001-of-00002.safetensors",
    "text_encoder_3/model.fp16-00002-of-00002.safetensors",
    "text_encoder_3/model.safetensors.index.fp16.json",
    "text_encoder_3/model.safetensors.index.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "tokenizer_3/special_tokens_map.json",
    "tokenizer_3/spiece.model",
    "tokenizer_3/tokenizer.json",
    "tokenizer_3/tokenizer_config.json",
    "transformer/config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sd35Route {
    Large,
    LargeTurbo,
    Medium,
}

impl Sd35Route {
    pub const fn provider_id(self) -> &'static str {
        match self {
            Self::Large => crate::MODEL_ID,
            Self::LargeTurbo => crate::MODEL_ID_TURBO,
            Self::Medium => crate::MODEL_ID_MEDIUM,
        }
    }

    pub const fn repository(self) -> &'static str {
        match self {
            Self::Large => "sd3.5-large-mlx",
            Self::LargeTurbo => "sd3.5-large-turbo-mlx",
            Self::Medium => "sd3.5-medium-mlx",
        }
    }

    pub const fn revision(self) -> &'static str {
        match self {
            Self::Large => "0cf819d00d30d296cee58e02c59b0daa5b8ede89",
            Self::LargeTurbo => "e9166f4632ec64f74d560be3ac778d346f89a364",
            Self::Medium => "5413e962bb326db248be2026a93b147c323392b6",
        }
    }

    fn from_provider(provider_id: &str) -> gen_core::Result<Self> {
        match provider_id {
            crate::MODEL_ID => Ok(Self::Large),
            crate::MODEL_ID_TURBO => Ok(Self::LargeTurbo),
            crate::MODEL_ID_MEDIUM => Ok(Self::Medium),
            _ => Err(gen_core::Error::Unsupported(format!(
                "unknown SD3.5 memory provider {provider_id}"
            ))),
        }
    }

    fn public_geometries(self) -> &'static [(u32, u32)] {
        const LARGE: &[(u32, u32)] = &[
            (1024, 1024),
            (1152, 896),
            (896, 1152),
            (1216, 832),
            (832, 1216),
            (1344, 768),
            (768, 1344),
        ];
        const MEDIUM: &[(u32, u32)] = &[
            (1024, 1024),
            (1440, 1440),
            (1152, 896),
            (896, 1152),
            (1216, 832),
            (832, 1216),
            (1344, 768),
            (768, 1344),
        ];
        match self {
            Self::Large | Self::LargeTurbo => LARGE,
            Self::Medium => MEDIUM,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileReceipt {
    lexical_path: PathBuf,
    canonical_path: PathBuf,
    projected_resident_bytes: u64,
    pin: gen_core::PinnedWeightsFile,
}

impl FileReceipt {
    fn capture(
        spec: &LoadSpec,
        path: &Path,
        projected_resident_bytes: u64,
    ) -> gen_core::Result<Self> {
        if projected_resident_bytes == 0 {
            return Err(gen_core::Error::Unsupported(format!(
                "SD3.5 component {} has zero realized bytes",
                path.display()
            )));
        }
        let pin = prepared_or_current_pin(spec, path)?;
        let receipt = Self {
            lexical_path: pin.loader_path().to_path_buf(),
            canonical_path: pin.canonical_target_path().to_path_buf(),
            projected_resident_bytes,
            pin,
        };
        receipt.ensure_unchanged()?;
        Ok(receipt)
    }

    fn ensure_unchanged(&self) -> gen_core::Result<()> {
        self.pin.verify_unchanged()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterReceipt {
    pub order: usize,
    pub kind: AdapterKind,
    pub scale_bits: u32,
    pub realized_bytes: u64,
    file: FileReceipt,
}

#[derive(Clone, Debug)]
pub struct Sd35LoadReceipt {
    pub route: Sd35Route,
    pub tier: Option<Quant>,
    pub group_size: Option<usize>,
    root: PathBuf,
    inventory: Vec<(PathBuf, gen_core::PinnedWeightsFile)>,
    transformer_config: gen_core::PinnedWeightsFile,
    pub components: gen_core::PerComponentBytes,
    pub adapters: Vec<AdapterReceipt>,
    physical_identity: String,
}

impl Sd35LoadReceipt {
    pub fn capture(route: Sd35Route, spec: &LoadSpec) -> gen_core::Result<Self> {
        validate_load_shape(route, spec)?;
        let WeightsSource::Dir(root) = &spec.weights else {
            unreachable!("validated directory source")
        };
        validate_snapshot_binding(root, route)?;
        let root = std::path::absolute(root)?;
        let inventory_paths = exact_inventory(&root, route)?;
        let mut inventory = Vec::with_capacity(inventory_paths.len());
        for path in inventory_paths {
            let pin = prepared_or_current_pin(spec, &path)?;
            inventory.push((path, pin));
        }
        let config = prepared_or_current_pin(spec, &root.join("transformer/config.json"))?;
        let transformer_paths = direct_safetensors(&root.join("transformer"))?;
        let (tier, group_size, transformer_bytes) =
            inspect_transformer(&transformer_paths, &config)?;
        let expected_tier = tier_from_root(&root)?;
        if tier != expected_tier {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: path tier crossed physical transformer headers",
                route.provider_id()
            )));
        }
        let text_encoder = ["text_encoder", "text_encoder_2", "text_encoder_3"]
            .iter()
            .try_fold(0_u64, |total, sub| {
                selected_float_bytes(&direct_safetensors(&root.join(sub))?, 2)
                    .and_then(|bytes| checked_add(total, bytes, "text encoder bytes"))
            })?;
        let vae_paths = direct_safetensors(&root.join("vae"))?;
        let vae_bf16 = selected_float_bytes(&vae_paths, 2)?;
        let vae_f32 = selected_float_bytes(&vae_paths, 4)?;
        let adapters = capture_adapters(spec)?;
        let components = gen_core::PerComponentBytes {
            text_encoder,
            dit: transformer_bytes,
            // The I2I encoder materializes the same sealed VAE tensors in F32 before the BF16
            // decoder phase. Price the widest VAE phase, not merely the resident decoder width.
            vae: vae_bf16.max(vae_f32),
        };
        let physical_identity = physical_identity(route, tier, &inventory, &adapters);
        let receipt = Self {
            route,
            tier,
            group_size,
            root,
            inventory,
            transformer_config: config,
            components,
            adapters,
            physical_identity,
        };
        receipt.ensure_unchanged()?;
        Ok(receipt)
    }

    pub fn ensure_unchanged(&self) -> gen_core::Result<()> {
        let current = exact_inventory(&self.root, self.route)?;
        let expected = self
            .inventory
            .iter()
            .map(|(path, _)| path.clone())
            .collect::<Vec<_>>();
        if current != expected {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: immutable tier inventory changed after admission",
                self.route.provider_id()
            )));
        }
        for (_, pin) in &self.inventory {
            pin.verify_unchanged()?;
        }
        self.transformer_config.verify_unchanged()?;
        let physical = inspect_transformer(
            &direct_safetensors(&self.root.join("transformer"))?,
            &self.transformer_config,
        )?;
        if physical.0 != self.tier || physical.1 != self.group_size {
            return Err(gen_core::Error::Unsupported(
                "SD3.5 transformer packing changed after admission".into(),
            ));
        }
        for adapter in &self.adapters {
            adapter.file.ensure_unchanged()?;
        }
        Ok(())
    }

    pub fn physical_identity(&self) -> &str {
        &self.physical_identity
    }

    pub fn adapter_identity(&self) -> Option<String> {
        if self.adapters.is_empty() {
            return None;
        }
        let mut digest = Sha256::new();
        update_framed(&mut digest, self.route.provider_id().as_bytes());
        update_framed(&mut digest, self.route.repository().as_bytes());
        update_framed(&mut digest, self.route.revision().as_bytes());
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
            update_framed(&mut digest, &adapter.realized_bytes.to_le_bytes());
            update_framed(
                &mut digest,
                adapter.file.lexical_path.as_os_str().as_encoded_bytes(),
            );
            update_framed(
                &mut digest,
                adapter.file.canonical_path.as_os_str().as_encoded_bytes(),
            );
            update_framed(&mut digest, adapter.file.pin.content_sha256());
        }
        Some(format_digest(ADAPTER_RECEIPT_PREFIX, digest.finalize()))
    }
}

fn checked_add(left: u64, right: u64, label: &str) -> gen_core::Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| gen_core::Error::Unsupported(format!("SD3.5 {label} overflow")))
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
                    "SD3.5 prepared load receipt is missing {}",
                    absolute.display()
                ))
            })
    } else {
        gen_core::PinnedWeightsFile::pin(path)
    }
}

fn validate_load_shape(route: Sd35Route, spec: &LoadSpec) -> gen_core::Result<()> {
    if spec.resolved_route.as_deref() != Some(route.provider_id()) {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: exact resolved_route is required",
            route.provider_id()
        )));
    }
    if spec.precision != Precision::Bf16 || spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: physical q4/q8/bf16 turnkeys require precision=Bf16 and quantize=None",
            route.provider_id()
        )));
    }
    if !matches!(spec.weights, WeightsSource::Dir(_)) {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: requires a turnkey snapshot directory",
            route.provider_id()
        )));
    }
    if spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
        || !spec.components.is_empty()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: control, IP, PiD, identity, and external components are unsupported",
            route.provider_id()
        )));
    }
    Ok(())
}

fn validate_snapshot_binding(root: &Path, route: Sd35Route) -> gen_core::Result<()> {
    let components = std::path::absolute(root)?
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let tier = components.last().map(String::as_str).unwrap_or_default();
    if !matches!(tier, "q4" | "q8" | "bf16") {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: path must end in q4, q8, or bf16",
            route.provider_id()
        )));
    }
    let hf = format!("models--SceneWorks--{}", route.repository());
    let app = format!("SceneWorks__{}", route.repository());
    let expected = [
        vec![hf, "snapshots".into(), route.revision().into(), tier.into()],
        vec![app, route.revision().into(), tier.into()],
        vec![
            route.repository().into(),
            route.revision().into(),
            tier.into(),
        ],
    ];
    if expected.iter().any(|suffix| components.ends_with(suffix)) {
        Ok(())
    } else {
        Err(gen_core::Error::Unsupported(format!(
            "{}: source must be exact SceneWorks/{}@{}",
            route.provider_id(),
            route.repository(),
            route.revision()
        )))
    }
}

fn expected_inventory(route: Sd35Route, tier: &str) -> Vec<String> {
    let mut files = COMMON_TIER_FILES
        .iter()
        .map(|file| (*file).to_owned())
        .collect::<Vec<_>>();
    match tier {
        "q4" | "q8" => files.push("transformer/diffusion_pytorch_model.safetensors".into()),
        "bf16" if route == Sd35Route::Medium => {
            files.push("transformer/diffusion_pytorch_model.safetensors".into())
        }
        "bf16" => {
            files.extend([
                "transformer/diffusion_pytorch_model-00001-of-00002.safetensors".into(),
                "transformer/diffusion_pytorch_model-00002-of-00002.safetensors".into(),
                "transformer/diffusion_pytorch_model.safetensors.index.json".into(),
            ]);
        }
        _ => {}
    }
    files.sort();
    files
}

fn collect_files(root: &Path, out: &mut Vec<PathBuf>) -> gen_core::Result<()> {
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        let metadata = std::fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            if std::fs::metadata(&path)?.is_file() {
                out.push(std::path::absolute(path)?);
            } else {
                return Err(gen_core::Error::Unsupported(format!(
                    "SD3.5 non-file symlink in snapshot: {}",
                    path.display()
                )));
            }
        } else if metadata.is_dir() {
            collect_files(&path, out)?;
        } else if metadata.is_file() {
            out.push(std::path::absolute(path)?);
        }
    }
    Ok(())
}

fn exact_inventory(root: &Path, route: Sd35Route) -> gen_core::Result<Vec<PathBuf>> {
    let tier = root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let mut actual = Vec::new();
    collect_files(root, &mut actual)?;
    actual.sort();
    let mut relative = actual
        .iter()
        .map(|path| {
            path.strip_prefix(root)
                .map(|value| value.to_string_lossy().replace('\\', "/"))
                .map_err(|_| {
                    gen_core::Error::Unsupported("SD3.5 inventory escaped tier root".into())
                })
        })
        .collect::<gen_core::Result<Vec<_>>>()?;
    relative.sort();
    let expected = expected_inventory(route, tier);
    if relative != expected {
        let expected_set = expected.iter().collect::<BTreeSet<_>>();
        let actual_set = relative.iter().collect::<BTreeSet<_>>();
        let missing = expected_set
            .difference(&actual_set)
            .copied()
            .collect::<Vec<_>>();
        let extra = actual_set
            .difference(&expected_set)
            .copied()
            .collect::<Vec<_>>();
        return Err(gen_core::Error::Unsupported(format!(
            "{}: immutable {tier} inventory differs from upstream; missing={missing:?} extra={extra:?}",
            route.provider_id()
        )));
    }
    Ok(actual)
}

fn direct_safetensors(dir: &Path) -> gen_core::Result<Vec<PathBuf>> {
    let mut files = std::fs::read_dir(dir)
        .map_err(|error| {
            gen_core::Error::Unsupported(format!(
                "SD3.5 cannot inventory {}: {error}",
                dir.display()
            ))
        })?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
                && !gen_core::weightsmeta::is_hidden_file(path)
        })
        .collect::<Vec<_>>();
    files.sort();
    if files.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "SD3.5 component {} contains no safetensors",
            dir.display()
        )));
    }
    Ok(files)
}

fn selected_headers(
    paths: &[PathBuf],
) -> gen_core::Result<BTreeMap<String, gen_core::weightsmeta::SafetensorsTensorHeader>> {
    let mut selected = BTreeMap::new();
    // The production mmap loader is lexical last-file-wins. Preserve that exact duplicate policy.
    for path in paths {
        for header in gen_core::weightsmeta::safetensors_path_tensor_headers(path)? {
            selected.insert(header.name.clone(), header);
        }
    }
    if selected.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "SD3.5 selected tensor inventory is empty".into(),
        ));
    }
    Ok(selected)
}

fn selected_float_bytes(paths: &[PathBuf], realized_width: u64) -> gen_core::Result<u64> {
    use gen_core::weightsmeta::Dtype;
    selected_headers(paths)?
        .values()
        .try_fold(0_u64, |total, header| {
            if header.dtype != Dtype::BF16 {
                return Err(gen_core::Error::Unsupported(format!(
                    "SD3.5 dense component tensor {} must be BF16, got {:?}",
                    header.name, header.dtype
                )));
            }
            checked_add(
                total,
                header.materialized_bytes(realized_width)?,
                "component bytes",
            )
        })
}

fn config_quant(config: &gen_core::PinnedWeightsFile) -> gen_core::Result<Option<(u8, usize)>> {
    config.read_unchanged(|path| {
        let value: serde_json::Value = serde_json::from_reader(std::fs::File::open(path)?)
            .map_err(|error| {
                gen_core::Error::Unsupported(format!(
                    "SD3.5 malformed transformer config {}: {error}",
                    path.display()
                ))
            })?;
        let Some(packed) = candle_gen::quant::PackedConfig::from_config(&value) else {
            return Ok(None);
        };
        let bits = u8::try_from(packed.bits)
            .map_err(|_| gen_core::Error::Unsupported("SD3.5 quant bits overflow".into()))?;
        let group = usize::try_from(packed.group_size)
            .map_err(|_| gen_core::Error::Unsupported("SD3.5 group size overflow".into()))?;
        Ok(Some((bits, group)))
    })
}

fn inspect_transformer(
    files: &[PathBuf],
    config: &gen_core::PinnedWeightsFile,
) -> gen_core::Result<(Option<Quant>, Option<usize>, u64)> {
    use gen_core::weightsmeta::Dtype;
    let headers = selected_headers(files)?;
    let packed_bases = headers
        .keys()
        .filter_map(|name| name.strip_suffix(".scales").map(str::to_owned))
        .collect::<BTreeSet<_>>();
    let config_packed = config_quant(config)?;
    if packed_bases.is_empty() {
        if config_packed.is_some() || headers.values().any(|header| header.dtype != Dtype::BF16) {
            return Err(gen_core::Error::Unsupported(
                "SD3.5 dense transformer requires BF16 tensors and no packing config".into(),
            ));
        }
        let bytes = headers.values().try_fold(0_u64, |total, header| {
            checked_add(
                total,
                header.materialized_bytes(2)?,
                "dense transformer bytes",
            )
        })?;
        return Ok((None, None, bytes));
    }
    let Some((bits, group)) = config_packed else {
        return Err(gen_core::Error::Unsupported(
            "SD3.5 packed transformer has no matching quantization config".into(),
        ));
    };
    if !matches!(bits, 4 | 8) || group != candle_gen::quant::MLX_GROUP_SIZE {
        return Err(gen_core::Error::Unsupported(format!(
            "SD3.5 supports only group-64 q4/q8 packed transformers, got q{bits}/group-{group}"
        )));
    }
    let mut bytes = 0_u64;
    for base in &packed_bases {
        let weight = headers.get(&format!("{base}.weight")).ok_or_else(|| {
            gen_core::Error::Unsupported(format!("SD3.5 packed {base} lacks weight"))
        })?;
        let scales = headers
            .get(&format!("{base}.scales"))
            .expect("base came from scales");
        let biases = headers.get(&format!("{base}.biases")).ok_or_else(|| {
            gen_core::Error::Unsupported(format!("SD3.5 packed {base} lacks biases"))
        })?;
        if weight.dtype != Dtype::U32 || scales.dtype != Dtype::BF16 || biases.dtype != Dtype::BF16
        {
            return Err(gen_core::Error::Unsupported(format!(
                "SD3.5 packed {base} must use U32/BF16/BF16"
            )));
        }
        let [rows, packed_columns] = weight.shape.as_slice() else {
            return Err(gen_core::Error::Unsupported(format!(
                "SD3.5 packed {base}.weight must be rank 2"
            )));
        };
        let [scale_rows, scale_columns] = scales.shape.as_slice() else {
            return Err(gen_core::Error::Unsupported(format!(
                "SD3.5 packed {base}.scales must be rank 2"
            )));
        };
        let input = scale_columns.checked_mul(group).ok_or_else(|| {
            gen_core::Error::Unsupported("SD3.5 packed input width overflow".into())
        })?;
        let encoded = packed_columns
            .checked_mul(32)
            .ok_or_else(|| gen_core::Error::Unsupported("SD3.5 packed width overflow".into()))?;
        if rows != scale_rows
            || scales.shape != biases.shape
            || input == 0
            || encoded.checked_div(input) != Some(usize::from(bits))
            || !encoded.is_multiple_of(input)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "SD3.5 packed {base} crosses config width or affine sidecar geometry"
            )));
        }
        bytes = checked_add(
            bytes,
            candle_gen::quant::mlx_packed_qtensor_resident_bytes(weight, scales, biases, group)?,
            "packed transformer bytes",
        )?;
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
                "SD3.5 non-packed transformer tensor {name} must be BF16"
            )));
        }
        bytes = checked_add(
            bytes,
            header.materialized_bytes(2)?,
            "transformer leaf bytes",
        )?;
    }
    Ok((
        Some(if bits == 4 { Quant::Q4 } else { Quant::Q8 }),
        Some(group),
        bytes,
    ))
}

fn tier_from_root(root: &Path) -> gen_core::Result<Option<Quant>> {
    match root.file_name().and_then(|name| name.to_str()) {
        Some("q4") => Ok(Some(Quant::Q4)),
        Some("q8") => Ok(Some(Quant::Q8)),
        Some("bf16") => Ok(None),
        _ => Err(gen_core::Error::Unsupported(
            "SD3.5 physical tier must be q4, q8, or bf16".into(),
        )),
    }
}

fn capture_adapters(spec: &LoadSpec) -> gen_core::Result<Vec<AdapterReceipt>> {
    let mut lexical = BTreeSet::new();
    let mut canonical = BTreeSet::new();
    spec.adapters
        .iter()
        .enumerate()
        .map(|(order, adapter)| {
            if !adapter.scale.is_finite() || adapter.pass_scales.is_some() || adapter.moe_expert.is_some() {
                return Err(gen_core::Error::Unsupported(format!(
                    "SD3.5 adapter {order} requires a finite uniform scale and no pass/MoE target"
                )));
            }
            let meta = gen_core::weightsmeta::CheckpointMeta::from_file(&adapter.path)?;
            let keys = meta.keys().collect::<Vec<_>>();
            let metadata_lokr = gen_core::weightsmeta::is_lokr_network_type(meta.metadata("networkType"));
            let keys_lokr = gen_core::weightsmeta::keys_contain_lokr(keys.iter().copied());
            let has_lora_down = keys.iter().any(|key| {
                key.ends_with(".lora_A.weight") || key.ends_with(".lora_down.weight")
            });
            let has_lora_up = keys.iter().any(|key| {
                key.ends_with(".lora_B.weight") || key.ends_with(".lora_up.weight")
            });
            match adapter.kind {
                AdapterKind::Lora if metadata_lokr || keys_lokr || !has_lora_down || !has_lora_up => {
                    return Err(gen_core::Error::Unsupported(format!(
                        "SD3.5 adapter {order} declared LoRA but its tensor layout is not an exact LoRA pair"
                    )))
                }
                AdapterKind::Lokr if !metadata_lokr || !keys_lokr => {
                    return Err(gen_core::Error::Unsupported(format!(
                        "SD3.5 adapter {order} declared LoKr without stamped LoKr factors"
                    )))
                }
                _ => {}
            }
            let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(&adapter.path)?;
            let realized_bytes = headers.iter().try_fold(0_u64, |total, header| {
                use gen_core::weightsmeta::Dtype;
                if !matches!(header.dtype, Dtype::BF16 | Dtype::F16 | Dtype::F32) {
                    return Err(gen_core::Error::Unsupported(format!(
                        "SD3.5 adapter tensor {} has unsupported additive dtype {:?}",
                        header.name, header.dtype
                    )));
                }
                checked_add(total, header.materialized_bytes(4)?, "adapter realized bytes")
            })?;
            let file = FileReceipt::capture(spec, &adapter.path, realized_bytes)?;
            if !lexical.insert(file.lexical_path.clone()) || !canonical.insert(file.canonical_path.clone()) {
                return Err(gen_core::Error::Unsupported(
                    "SD3.5 duplicate lexical or canonical adapter source".into(),
                ));
            }
            Ok(AdapterReceipt {
                order,
                kind: adapter.kind,
                scale_bits: adapter.scale.to_bits(),
                realized_bytes,
                file,
            })
        })
        .collect()
}

fn update_framed(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn format_digest(prefix: &str, bytes: impl IntoIterator<Item = u8>) -> String {
    let mut output = prefix.to_owned();
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("hex formatting into String cannot fail");
    }
    output
}

fn physical_identity(
    route: Sd35Route,
    tier: Option<Quant>,
    inventory: &[(PathBuf, gen_core::PinnedWeightsFile)],
    adapters: &[AdapterReceipt],
) -> String {
    let mut digest = Sha256::new();
    update_framed(&mut digest, route.provider_id().as_bytes());
    update_framed(&mut digest, route.repository().as_bytes());
    update_framed(&mut digest, route.revision().as_bytes());
    update_framed(&mut digest, format!("{tier:?}").as_bytes());
    for (path, pin) in inventory {
        update_framed(&mut digest, path.as_os_str().as_encoded_bytes());
        update_framed(
            &mut digest,
            pin.canonical_target_path().as_os_str().as_encoded_bytes(),
        );
        update_framed(&mut digest, pin.content_sha256());
    }
    for adapter in adapters {
        update_framed(&mut digest, &(adapter.order as u64).to_le_bytes());
        update_framed(&mut digest, &adapter.scale_bits.to_le_bytes());
        update_framed(&mut digest, &adapter.realized_bytes.to_le_bytes());
    }
    format_digest(PHYSICAL_RECEIPT_PREFIX, digest.finalize())
}

pub fn provider_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let route = Sd35Route::from_provider(provider_id)?;
    let receipt = Sd35LoadReceipt::capture(route, spec)?;
    Ok(contract_from_receipt(spec, &receipt))
}

pub fn contract_from_receipt(spec: &LoadSpec, receipt: &Sd35LoadReceipt) -> MemoryProviderContract {
    let mut components = vec![MemoryResidentComponent {
        id: receipt.physical_identity().to_owned(),
        kind: MemoryComponentKind::TransformerSubStack(gen_core::TransformerComponent::Dit),
        resident_bytes: receipt.components.dit,
        bounded_by: Some(MemoryStrategy::StagedResidency),
        residency: MemoryComponentResidency::WholeRender,
    }];
    let adapter_bytes = receipt.adapters.iter().fold(0_u64, |total, adapter| {
        total.saturating_add(adapter.realized_bytes)
    });
    if adapter_bytes > 0 {
        components.push(MemoryResidentComponent {
            id: receipt
                .adapter_identity()
                .expect("nonempty adapter stack has identity"),
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: adapter_bytes,
            bounded_by: Some(MemoryStrategy::StagedResidency),
            residency: MemoryComponentResidency::WholeRender,
        });
    }
    build_contract(receipt.route, spec, receipt.components, components)
}

pub fn weights_free_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let route = Sd35Route::from_provider(provider_id)?;
    let tier = match spec.quantize {
        None => None,
        Some(Quant::Q4) => Some(Quant::Q4),
        Some(Quant::Q8) => Some(Quant::Q8),
        Some(Quant::Nvfp4) => {
            return Err(gen_core::Error::Unsupported(
                "SD3.5 has no NVFP4 tier".into(),
            ))
        }
    };
    let mut normalized = spec.clone();
    normalized.quantize = None;
    normalized.resolved_route = Some(provider_id.to_owned());
    let _ = tier;
    Ok(build_contract(
        route,
        &normalized,
        Default::default(),
        Vec::new(),
    ))
}

pub fn weights_free_surface_contract(
    provider_id: &str,
    surface: &gen_core::MemoryContractSurfaceSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let mut spec = surface.spec.clone();
    spec.quantize = match surface.resolved_artifact_tier() {
        gen_core::MemoryContractSurfaceTier::Bf16 => None,
        gen_core::MemoryContractSurfaceTier::Q4 => Some(Quant::Q4),
        gen_core::MemoryContractSurfaceTier::Q8 => Some(Quant::Q8),
        gen_core::MemoryContractSurfaceTier::Nvfp4 => {
            return Err(gen_core::Error::Unsupported(
                "SD3.5 has no NVFP4 tier".into(),
            ))
        }
    };
    weights_free_contract(provider_id, &spec)
}

/// Activation dtype the loaded SD3.5 pipeline computes in. `lib.rs` pins `DType::BF16`
/// unconditionally on every route, so this is the provider's real activation width rather than a
/// memory-model literal.
const ACTIVATION_DTYPE: candle_gen::candle_core::DType = candle_gen::candle_core::DType::BF16;

/// Snapshot-scoped architecture axes for an SD3.5 route (epic SC-22657, E2).
///
/// SD3.5's geometry is deliberately *not* read from `transformer/config.json`: the loader builds
/// the MMDiT from the crate's own [`crate::config::Sd3Config`] preset, selected from the variant
/// exactly as `pipeline::Variant::config` does (Large/Turbo share the Large preset; Medium is the
/// MMDiT-X preset). Reading a config the loader ignores would describe a model this provider never
/// constructs, so the axes come off the same struct the loader hands to the transformer builder.
///
/// A weights-free contract — the registry's sentinel surface path, or a single-file import —
/// publishes `MemoryArchitectureFacts::default()`: nothing that *would* be loaded is resolved
/// there, so no axis is knowable.
fn architecture_facts(route: Sd35Route, spec: &LoadSpec) -> gen_core::MemoryArchitectureFacts {
    use candle_gen::architecture_facts as af;

    if af::snapshot_root(spec).is_none() {
        return gen_core::MemoryArchitectureFacts::default();
    }
    // The same variant -> preset selection `pipeline::Variant::config` performs at load.
    let config = match route {
        Sd35Route::Large | Sd35Route::LargeTurbo => crate::config::Sd3Config::large(),
        Sd35Route::Medium => crate::config::Sd3Config::medium(),
    };
    gen_core::MemoryArchitectureFacts {
        attention_heads: af::declared(config.num_heads),
        head_dim: af::declared(config.head_dim),
        transformer_blocks: af::declared(config.num_layers),
        patch_size: af::declared(config.patch_size),
        // `vae::LATENT_CHANNELS` is the encoder's own declaration of what it produces; the DiT's
        // `in_channels` is the consumer's view of the same 16 channels.
        latent_channels: af::declared(crate::vae::LATENT_CHANNELS),
        vae_spatial_scale: af::declared(crate::vae::SPATIAL_SCALE as usize),
        // SD3.5 ships the image `AutoencoderKL`: there is no temporal axis to declare at all.
        vae_temporal_scale: None,
        activation_dtype_width: af::dtype_width(ACTIVATION_DTYPE),
    }
}

fn build_contract(
    route: Sd35Route,
    spec: &LoadSpec,
    components: gen_core::PerComponentBytes,
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
                MemoryStrategy::Resident | MemoryStrategy::StagedResidency => {
                    MemoryStrategySupport::Implemented
                }
                _ => MemoryStrategySupport::Missing,
            },
            parameters: MemoryParameterRanges::default(),
        })
        .collect();
    let overlay_bytes = resident_components
        .iter()
        .filter(|component| component.kind == MemoryComponentKind::AdapterStack)
        .fold(0_u64, |total, component| {
            total.saturating_add(component.resident_bytes)
        });
    MemoryProviderContract {
        architecture_facts: architecture_facts(route, spec),
        provider_id: route.provider_id().to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
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
        formula: if resident_components.is_empty() {
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
                resident_components,
            }
        },
        calibration: None,
        asset_facts: MemoryAssetFacts {
            base_bytes: components
                .text_encoder
                .saturating_add(components.dit)
                .saturating_add(components.vae),
            conditioning_bytes: components.text_encoder,
            transformer_bytes: components.dit,
            decoder_bytes: components.vae,
            overlay_bytes,
        },
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    }
}

fn contract_adapter_identity(contract: &MemoryProviderContract) -> gen_core::Result<Option<&str>> {
    let mut found = contract
        .resident_components()
        .iter()
        .filter_map(|component| {
            component
                .id
                .starts_with(ADAPTER_RECEIPT_PREFIX)
                .then_some(component.id.as_str())
        });
    let first = found.next();
    if found.next().is_some() {
        return Err(gen_core::Error::Unsupported(
            "SD3.5 contract contains multiple adapter identities".into(),
        ));
    }
    Ok(first)
}

pub fn validate_context(
    route: Sd35Route,
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
            "{}: structural-estimate handshake crossed ABI, fingerprint, or load shape",
            route.provider_id()
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
    let exact_mode = matches!(
        (
            &context.mode,
            context.geometry.reference_count,
            context.has_reference
        ),
        (MemoryMode::TextToImage, 0, false) | (MemoryMode::ImageToImage, 1, true)
    );
    if contract.provider_id != route.provider_id()
        || !exact_mode
        || context.use_pid
        || context.has_phases
        || context.geometry.frames != 1
        || !matches!(context.geometry.batch, 1 | 2 | 4)
        || !route
            .public_geometries()
            .contains(&(context.geometry.width, context.geometry.height))
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: exact T2I/ref0 or I2I/ref1 public geometry is required",
            route.provider_id()
        )));
    }
    if context.overlay.as_deref() != contract_adapter_identity(contract)? {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: public adapter identity crossed the sealed ordered LoRA/LoKr receipt",
            route.provider_id()
        )));
    }
    if context.evidence_revision != REQUEST_EVIDENCE_REVISION {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: request evidence revision is not exact",
            route.provider_id()
        )));
    }
    Ok(())
}

fn registered_tier_and_contract(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
) -> gen_core::Result<(Sd35Route, Option<Quant>)> {
    let route = Sd35Route::from_provider(&contract.provider_id)?;
    validate_load_shape(route, spec)?;
    if contract.asset_facts == MemoryAssetFacts::default() {
        let expected = weights_free_contract(route.provider_id(), spec)?;
        if expected != *contract {
            return Err(gen_core::Error::Unsupported(
                "SD3.5 caller contract crossed the registry witness".into(),
            ));
        }
        Ok((route, physical_tier_hint(spec)))
    } else {
        let receipt = Sd35LoadReceipt::capture(route, spec)?;
        let expected = contract_from_receipt(spec, &receipt);
        if expected != *contract {
            return Err(gen_core::Error::Unsupported(
                "SD3.5 caller contract crossed the sealed artifact receipt".into(),
            ));
        }
        receipt.ensure_unchanged()?;
        Ok((route, receipt.tier))
    }
}

fn physical_tier_hint(spec: &LoadSpec) -> Option<Quant> {
    spec.quantize.or_else(|| match &spec.weights {
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
    })
}

pub fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match registered_tier_and_contract(spec, contract)
        .and_then(|(route, tier)| validate_context(route, contract, context, tier))
    {
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
    prompt: String,
    negative_prompt: Option<String>,
    conditioning_identity: [u8; 32],
    seed: Option<u64>,
    steps: Option<u32>,
    guidance_bits: Option<u32>,
    sampler: Option<String>,
    scheduler: Option<String>,
    shift_bits: Option<u32>,
    strength_bits: Option<u32>,
}

impl RequestBinding {
    fn from_request(request: &GenerationRequest) -> Self {
        let mut conditioning = Sha256::new();
        for item in &request.conditioning {
            match item {
                gen_core::Conditioning::Reference { image, strength } => {
                    update_framed(&mut conditioning, b"reference");
                    update_framed(&mut conditioning, &image.width.to_le_bytes());
                    update_framed(&mut conditioning, &image.height.to_le_bytes());
                    update_framed(&mut conditioning, &image.pixels);
                    update_framed(
                        &mut conditioning,
                        &strength.map(f32::to_bits).unwrap_or_default().to_le_bytes(),
                    );
                }
                other => update_framed(&mut conditioning, format!("{other:?}").as_bytes()),
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
            prompt: request.prompt.clone(),
            negative_prompt: request.negative_prompt.clone(),
            conditioning_identity: conditioning.finalize().into(),
            seed: request.seed,
            steps: request.steps,
            guidance_bits: request.guidance.map(f32::to_bits),
            sampler: request.sampler.clone(),
            scheduler: request.scheduler.clone(),
            shift_bits: request.scheduler_shift.map(f32::to_bits),
            strength_bits: request.strength.map(f32::to_bits),
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
    next: u64,
    approved: Option<MemoryRunContext>,
    active: Option<ActiveAdmission>,
}

#[derive(Clone)]
pub struct AdmissionRegistry {
    provider_id: &'static str,
    state: Arc<Mutex<AdmissionState>>,
}

impl AdmissionRegistry {
    pub fn new(provider_id: &'static str) -> Self {
        Self {
            provider_id,
            state: Arc::new(Mutex::new(AdmissionState::default())),
        }
    }

    pub fn approve(&self, context: &MemoryRunContext) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.state);
        if state.active.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: another request is active",
                self.provider_id
            )));
        }
        state.approved = Some(context.clone());
        Ok(())
    }

    pub fn clear(&self) {
        candle_gen::lock_recover(&self.state).approved = None;
    }

    fn begin(
        &self,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
    ) -> gen_core::Result<u64> {
        let mut state = candle_gen::lock_recover(&self.state);
        if state.active.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: another request scope is active",
                self.provider_id
            )));
        }
        let approved = state.approved.take().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{}: request skipped safety approval",
                self.provider_id
            ))
        })?;
        if contract.provider_id != self.provider_id || approved != *context {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: crossed or changed safety context",
                self.provider_id
            )));
        }
        state.next = state.next.wrapping_add(1).max(1);
        let token = state.next;
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
        let mut state = candle_gen::lock_recover(&self.state);
        let active = state.active.as_mut().ok_or_else(|| {
            gen_core::Error::Unsupported(format!("{}: inactive request scope", self.provider_id))
        })?;
        let binding = RequestBinding::from_request(request);
        if active.token != token
            || active.binding.is_some()
            || active.consumed
            || binding.geometry != active.context.geometry
            || binding.memory != active.expected_memory
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: stale or changed request",
                self.provider_id
            )));
        }
        active.binding = Some(binding);
        Ok(())
    }

    pub fn consume(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.state);
        let constrained = request
            .memory
            .is_some_and(|memory| memory != GenerationMemory::default());
        let Some(active) = state.active.as_mut() else {
            return if constrained {
                Err(gen_core::Error::Unsupported(format!(
                    "{}: constrained request lacks admission",
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
                "{}: request changed or was already consumed",
                self.provider_id
            )));
        }
        active.consumed = true;
        Ok(())
    }

    fn finish(&self, token: u64) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.state);
        if state
            .active
            .as_ref()
            .is_some_and(|active| active.token == token)
        {
            state.active = None;
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: stale request token",
                self.provider_id
            )))
        }
    }

    fn abandon(&self, token: u64) {
        let mut state = candle_gen::lock_recover(&self.state);
        if state
            .active
            .as_ref()
            .is_some_and(|active| active.token == token)
        {
            state.active = None;
        }
    }
}

struct Scope {
    core: candle_gen::request_scope::CandleRequestScopeCore,
    admission: AdmissionRegistry,
    token: u64,
    finished: bool,
}

impl Scope {
    fn new(
        route: Sd35Route,
        device: Device,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
        admission: AdmissionRegistry,
    ) -> gen_core::Result<Self> {
        let token = admission.begin(contract, context)?;
        let config = candle_gen::request_scope::CandleRequestScopeConfig::new(
            route.provider_id(),
            device,
            context.geometry,
            contract.generation_memory(&context.selection),
            false,
            match route {
                Sd35Route::Large | Sd35Route::LargeTurbo => 38,
                Sd35Route::Medium => 24,
            },
            move |_, _, _| {
                Err(gen_core::Error::Unsupported(format!(
                    "{}: bounded decode is Missing",
                    route.provider_id()
                )))
            },
        )?;
        Ok(Self {
            core: candle_gen::request_scope::CandleRequestScopeCore::new(config),
            admission,
            token,
            finished: false,
        })
    }
}

impl MemoryRequestScope for Scope {
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
        edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.core.configure_decode(edge, overlap, geometry)
    }

    fn configure_attention(&mut self, size: u32) -> gen_core::Result<()> {
        self.core.configure_attention(size)
    }

    fn materialize_transformer_window(&mut self, first: u32, count: u32) -> gen_core::Result<()> {
        self.core.materialize_transformer_window(first, count)
    }

    fn finish(&mut self, outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        self.core.finish(outcome)?;
        self.admission.finish(self.token)?;
        self.finished = true;
        Ok(())
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if !self.finished {
            self.admission.abandon(self.token);
        }
    }
}

pub fn begin_request<'a>(
    receipt: &Sd35LoadReceipt,
    contract: &MemoryProviderContract,
    admission: AdmissionRegistry,
    device: Device,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope + 'a>>> {
    receipt.ensure_unchanged()?;
    validate_context(receipt.route, contract, context, receipt.tier)?;
    Ok(Some(Box::new(Scope::new(
        receipt.route,
        device,
        contract,
        context,
        admission,
    )?)))
}

pub fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let (route, tier) = registered_tier_and_contract(spec, contract)?;
    if route.provider_id() != provider_id {
        return Err(gen_core::Error::Unsupported(
            "SD3.5 registered route crossed".into(),
        ));
    }
    validate_context(route, contract, context, tier)?;
    let admission = AdmissionRegistry::new(provider_id);
    admission.approve(context)?;
    Ok(Some(Box::new(Scope::new(
        route,
        Device::Cpu,
        contract,
        context,
        admission,
    )?)))
}

pub fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<MemoryBehaviorFixture>> {
    if strategy != MemoryStrategy::StagedResidency {
        return Ok(Vec::new());
    }
    let route = Sd35Route::from_provider(&contract.provider_id)?;
    let exact_spec = weights_free_behavior_spec(route, spec)?;
    let tier = physical_tier_hint(&exact_spec);
    [
        MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::ImageToImage,
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    ]
    .into_iter()
    .map(|request_route| {
        let context = MemoryRunContext {
            selection: contract.representative_selection(
                strategy,
                MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: tier,
                    component_precision_floors: &[],
                },
                false,
            )?,
            optimization_authority: MemoryOptimizationAuthority::Estimated,
            calibration_abi: 0,
            calibration_fingerprint: String::new(),
            load_shape: contract.load_shape,
            mode: request_route.mode,
            has_reference: request_route.reference_count == 1,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: request_route.reference_count,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 64 * 1024 * 1024 * 1024,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 32 * 1024 * 1024 * 1024,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: REQUEST_EVIDENCE_REVISION.into(),
        };
        validate_context(route, contract, &context, tier)?;
        Ok(MemoryBehaviorFixture::new(context).with_load_spec(exact_spec.clone()))
    })
    .collect()
}

fn weights_free_behavior_spec(route: Sd35Route, spec: &LoadSpec) -> gen_core::Result<LoadSpec> {
    let tier = match physical_tier_hint(spec) {
        Some(Quant::Q4) => "q4",
        Some(Quant::Q8) => "q8",
        None => "bf16",
        Some(Quant::Nvfp4) => {
            return Err(gen_core::Error::Unsupported(
                "SD3.5 has no NVFP4 tier".into(),
            ))
        }
    };
    let mut exact = spec.clone();
    exact.weights = WeightsSource::Dir(
        PathBuf::from(format!("models--SceneWorks--{}", route.repository()))
            .join("snapshots")
            .join(route.revision())
            .join(tier),
    );
    exact.resolved_route = Some(route.provider_id().to_owned());
    exact.precision = Precision::Bf16;
    exact.quantize = None;
    Ok(exact)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{
        AdapterSpec, LoadShape, MemoryBudget, MemoryCacheState, MemorySelection,
        MemoryStrategyParameters, OffloadPolicy,
    };

    fn write_safetensors(
        path: &Path,
        metadata: Option<serde_json::Value>,
        tensors: &[(&str, &str, &[usize], usize)],
    ) {
        let mut offset = 0_usize;
        let mut header = serde_json::Map::new();
        if let Some(metadata) = metadata {
            header.insert("__metadata__".into(), metadata);
        }
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
        let mut encoded = serde_json::to_vec(&header).unwrap();
        while !encoded.len().is_multiple_of(8) {
            encoded.push(b' ');
        }
        let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
        bytes.extend(encoded);
        bytes.resize(bytes.len() + offset, 0);
        std::fs::write(path, bytes).unwrap();
    }

    fn fixture(route: Sd35Route, tier: &str) -> (tempfile::TempDir, PathBuf) {
        let temp = tempfile::tempdir().unwrap();
        let root = temp
            .path()
            .join(format!("SceneWorks__{}", route.repository()))
            .join(route.revision())
            .join(tier);
        for relative in expected_inventory(route, tier) {
            let path = root.join(&relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            if relative.ends_with(".safetensors") {
                let name = relative.replace(['/', '.', '-'], "_");
                write_safetensors(&path, None, &[(&name, "BF16", &[1], 2)]);
            } else if relative.ends_with(".json") {
                std::fs::write(&path, b"{}").unwrap();
            } else {
                std::fs::write(&path, b"fixture").unwrap();
            }
        }
        let transformer = direct_safetensors(&root.join("transformer")).unwrap()[0].clone();
        match tier {
            "q4" | "q8" => {
                let bits = if tier == "q4" { 4 } else { 8 };
                let packed_columns = if bits == 4 { 8 } else { 16 };
                write_safetensors(
                    &transformer,
                    None,
                    &[
                        (
                            "blocks.0.proj.weight",
                            "U32",
                            &[1, packed_columns],
                            packed_columns * 4,
                        ),
                        ("blocks.0.proj.scales", "BF16", &[1, 1], 2),
                        ("blocks.0.proj.biases", "BF16", &[1, 1], 2),
                        ("norm.weight", "BF16", &[1], 2),
                    ],
                );
                std::fs::write(
                    root.join("transformer/config.json"),
                    format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
                )
                .unwrap();
            }
            "bf16" => {
                for (index, path) in direct_safetensors(&root.join("transformer"))
                    .unwrap()
                    .into_iter()
                    .enumerate()
                {
                    write_safetensors(
                        &path,
                        None,
                        &[(&format!("blocks.{index}.weight"), "BF16", &[2, 2], 8)],
                    );
                }
            }
            _ => unreachable!(),
        }
        (temp, root)
    }

    fn adapter(temp: &Path, kind: AdapterKind, name: &str) -> PathBuf {
        let path = temp.join(name);
        match kind {
            AdapterKind::Lora => write_safetensors(
                &path,
                Some(serde_json::json!({"networkType":"lora"})),
                &[
                    (
                        "transformer.blocks.0.proj.lora_A.weight",
                        "BF16",
                        &[1, 2],
                        4,
                    ),
                    (
                        "transformer.blocks.0.proj.lora_B.weight",
                        "BF16",
                        &[2, 1],
                        4,
                    ),
                ],
            ),
            AdapterKind::Lokr => write_safetensors(
                &path,
                Some(serde_json::json!({"networkType":"lokr"})),
                &[
                    ("blocks.0.proj.lokr_w1", "BF16", &[1, 2], 4),
                    ("blocks.0.proj.lokr_w2", "BF16", &[2, 1], 4),
                ],
            ),
        }
        path
    }

    fn spec(route: Sd35Route, root: &Path, adapters: Vec<AdapterSpec>) -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
            .with_resolved_route(route.provider_id())
            .with_load_shape(LoadShape::DeferredMaterialization)
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_adapters(adapters);
        spec.precision = Precision::Bf16;
        spec
    }

    /// AC (epic SC-22657, E2): every SD3.5 route publishes the architecture axes of the
    /// `Sd3Config` preset its loader actually builds, and the weights-free surface publishes none.
    #[test]
    fn architecture_facts_match_the_loader_config_and_pass_conformance() {
        for (route, heads, layers) in [
            (Sd35Route::Large, 38, 38),
            (Sd35Route::LargeTurbo, 38, 38),
            (Sd35Route::Medium, 24, 24),
        ] {
            let (_temp, root) = fixture(route, "bf16");
            // The shared fixture gives the encoders and the DiT the same byte total, which the
            // conformance check reads as one component borrowing another's price. Widen the DiT
            // shards so every component is priced from its own distinct bytes.
            for (index, path) in direct_safetensors(&root.join("transformer"))
                .unwrap()
                .into_iter()
                .enumerate()
            {
                write_safetensors(
                    &path,
                    None,
                    &[(&format!("blocks.{index}.weight"), "BF16", &[8, 8], 128)],
                );
            }
            let load = spec(route, &root, Vec::new());
            let receipt = Sd35LoadReceipt::capture(route, &load).unwrap();
            let contract = contract_from_receipt(&load, &receipt);
            assert_eq!(
                contract.architecture_facts,
                gen_core::MemoryArchitectureFacts {
                    // `Sd3Config::large()` / `::medium()`: `num_heads`, `head_dim`, `num_layers`,
                    // `patch_size` — the exact struct `Variant::config` hands the MMDiT builder.
                    attention_heads: Some(heads),
                    head_dim: Some(64),
                    transformer_blocks: Some(layers),
                    patch_size: Some(2),
                    // `vae::LATENT_CHANNELS` / `vae::SPATIAL_SCALE`.
                    latent_channels: Some(16),
                    vae_spatial_scale: Some(8),
                    // SD3.5 ships the image `AutoencoderKL`: no temporal axis exists to declare.
                    vae_temporal_scale: None,
                    // `lib.rs` pins `DType::BF16` on every route.
                    activation_dtype_width: Some(2),
                },
                "{} architecture facts",
                route.provider_id()
            );
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);

            // The registry's weights-free surface resolves no snapshot, so no axis is knowable.
            let weights_free = weights_free_contract(
                route.provider_id(),
                &LoadSpec::new(WeightsSource::Dir(
                    "/__sceneworks_memory_contract_surface__".into(),
                )),
            )
            .unwrap();
            assert!(weights_free.architecture_facts.is_empty());
        }
    }

    #[test]
    fn registry_behavior_fixtures_bind_each_exact_route_and_estimate_handshake() {
        let generic = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        for route in [Sd35Route::Large, Sd35Route::LargeTurbo, Sd35Route::Medium] {
            let contract = weights_free_contract(route.provider_id(), &generic).unwrap();
            for fixture in
                registered_valid_fixture(&generic, &contract, MemoryStrategy::StagedResidency)
                    .unwrap()
            {
                let exact = fixture
                    .load_spec
                    .as_ref()
                    .expect("provider-owned load spec");
                assert_eq!(exact.resolved_route.as_deref(), Some(route.provider_id()));
                assert!(matches!(
                    registered_safety_check(exact, &contract, &fixture.context),
                    MemorySafetyDecision::Accept
                ));
                let mut crossed = fixture.context.clone();
                crossed.load_shape = LoadShape::DeferredMaterialization;
                assert!(matches!(
                    registered_safety_check(exact, &contract, &crossed),
                    MemorySafetyDecision::Reject { .. }
                ));
            }
        }
    }

    fn context(receipt: &Sd35LoadReceipt, mode: MemoryMode) -> MemoryRunContext {
        let contract =
            contract_from_receipt(&spec(receipt.route, &receipt.root, Vec::new()), receipt);
        let refs = u32::from(mode == MemoryMode::ImageToImage);
        MemoryRunContext {
            selection: MemorySelection {
                strategy: MemoryStrategy::StagedResidency,
                parameters: MemoryStrategyParameters::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: receipt.tier,
                    component_precision_floors: &[],
                },
            },
            optimization_authority: gen_core::MemoryOptimizationAuthority::Estimated,
            calibration_abi: 0,
            calibration_fingerprint: String::new(),
            load_shape: contract.load_shape,
            mode,
            has_reference: refs == 1,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: refs,
            },
            overlay: receipt.adapter_identity(),
            budget: MemoryBudget {
                total_bytes: u64::MAX,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: REQUEST_EVIDENCE_REVISION.into(),
        }
    }

    #[test]
    fn complete_route_tier_adapter_and_mode_surface_is_exact() {
        for route in [Sd35Route::Large, Sd35Route::LargeTurbo, Sd35Route::Medium] {
            for tier in ["q4", "q8", "bf16"] {
                let (temp, root) = fixture(route, tier);
                for kind in [None, Some(AdapterKind::Lora), Some(AdapterKind::Lokr)] {
                    let adapters = kind
                        .map(|kind| {
                            vec![AdapterSpec::new(
                                adapter(temp.path(), kind, &format!("{tier}-{kind:?}.safetensors")),
                                0.75,
                                kind,
                            )]
                        })
                        .unwrap_or_default();
                    let load = spec(route, &root, adapters);
                    let receipt = Sd35LoadReceipt::capture(route, &load).unwrap();
                    assert_eq!(receipt.tier, tier_from_root(&root).unwrap());
                    assert!(receipt
                        .physical_identity()
                        .starts_with(PHYSICAL_RECEIPT_PREFIX));
                    if kind.is_some() {
                        assert_eq!(receipt.adapters[0].realized_bytes, 16);
                    }
                    let contract = contract_from_receipt(&load, &receipt);
                    for mode in [MemoryMode::TextToImage, MemoryMode::ImageToImage] {
                        for &(width, height) in route.public_geometries() {
                            for batch in [1, 2, 4] {
                                let mut request = context(&receipt, mode.clone());
                                request.geometry.width = width;
                                request.geometry.height = height;
                                request.geometry.batch = batch;
                                validate_context(route, &contract, &request, receipt.tier).unwrap();
                            }
                        }
                    }
                    for missing in [
                        MemoryStrategy::BoundedDecode,
                        MemoryStrategy::BoundedAttention,
                        MemoryStrategy::BoundedTransformerResidency,
                    ] {
                        assert_eq!(
                            contract.capability(missing).unwrap().support,
                            MemoryStrategySupport::Missing
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn crossed_headers_sources_receipts_and_mutations_fail_closed() {
        let (temp, root) = fixture(Sd35Route::Large, "q4");
        let load = spec(Sd35Route::Large, &root, Vec::new());
        let receipt = Sd35LoadReceipt::capture(Sd35Route::Large, &load).unwrap();
        assert!(Sd35LoadReceipt::capture(Sd35Route::Medium, &load).is_err());

        let mut crossed = context(&receipt, MemoryMode::TextToImage);
        crossed.selection.tier.quant = Some(Quant::Q8);
        let contract = contract_from_receipt(&load, &receipt);
        assert!(validate_context(Sd35Route::Large, &contract, &crossed, receipt.tier).is_err());

        std::fs::write(root.join("README.txt"), b"extra").unwrap();
        assert!(receipt.ensure_unchanged().is_err());
        std::fs::remove_file(root.join("README.txt")).unwrap();
        std::fs::write(root.join("LICENSE.md"), b"mutated").unwrap();
        assert!(receipt.ensure_unchanged().is_err());

        let bad_adapter = temp.path().join("bad.safetensors");
        write_safetensors(
            &bad_adapter,
            Some(serde_json::json!({"networkType":"lokr"})),
            &[("x.lokr_w1", "BF16", &[1], 2)],
        );
        let crossed_kind = spec(
            Sd35Route::Large,
            &root,
            vec![AdapterSpec::new(bad_adapter, 1.0, AdapterKind::Lora)],
        );
        assert!(Sd35LoadReceipt::capture(Sd35Route::Large, &crossed_kind).is_err());
    }

    #[test]
    fn packed_config_and_tensor_headers_must_agree() {
        let (_temp, root) = fixture(Sd35Route::Medium, "q4");
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":8,"group_size":64}}"#,
        )
        .unwrap();
        let load = spec(Sd35Route::Medium, &root, Vec::new());
        assert!(Sd35LoadReceipt::capture(Sd35Route::Medium, &load).is_err());
    }

    #[test]
    fn every_dense_component_and_adapter_header_is_physically_sized_and_sealed() {
        for component in ["text_encoder", "text_encoder_2", "text_encoder_3", "vae"] {
            let (_temp, root) = fixture(Sd35Route::LargeTurbo, "bf16");
            let path = direct_safetensors(&root.join(component))
                .unwrap()
                .pop()
                .unwrap();
            write_safetensors(&path, None, &[("forged.weight", "F32", &[1], 4)]);
            let load = spec(Sd35Route::LargeTurbo, &root, Vec::new());
            assert!(
                Sd35LoadReceipt::capture(Sd35Route::LargeTurbo, &load).is_err(),
                "{component}"
            );
        }

        let (temp, root) = fixture(Sd35Route::Large, "bf16");
        let unsupported = temp.path().join("f64.safetensors");
        write_safetensors(
            &unsupported,
            Some(serde_json::json!({"networkType":"lora"})),
            &[
                ("x.lora_A.weight", "F64", &[1], 8),
                ("x.lora_B.weight", "F64", &[1], 8),
            ],
        );
        let load = spec(
            Sd35Route::Large,
            &root,
            vec![AdapterSpec::new(unsupported, 1.0, AdapterKind::Lora)],
        );
        assert!(Sd35LoadReceipt::capture(Sd35Route::Large, &load).is_err());
    }

    #[test]
    fn adapter_kind_scale_order_bytes_and_mutation_are_exact() {
        let (temp, root) = fixture(Sd35Route::Medium, "q8");
        let lora = adapter(temp.path(), AdapterKind::Lora, "first.safetensors");
        let lokr = adapter(temp.path(), AdapterKind::Lokr, "second.safetensors");
        let make = |items: Vec<(PathBuf, f32, AdapterKind)>| {
            spec(
                Sd35Route::Medium,
                &root,
                items
                    .into_iter()
                    .map(|(path, scale, kind)| AdapterSpec::new(path, scale, kind))
                    .collect(),
            )
        };
        let first = make(vec![
            (lora.clone(), 0.25, AdapterKind::Lora),
            (lokr.clone(), 0.75, AdapterKind::Lokr),
        ]);
        let receipt = Sd35LoadReceipt::capture(Sd35Route::Medium, &first).unwrap();
        assert_eq!(
            receipt
                .adapters
                .iter()
                .map(|value| value.realized_bytes)
                .collect::<Vec<_>>(),
            [16, 16]
        );
        let identity = receipt.adapter_identity().unwrap();
        for crossed in [
            make(vec![
                (lokr.clone(), 0.75, AdapterKind::Lokr),
                (lora.clone(), 0.25, AdapterKind::Lora),
            ]),
            make(vec![
                (lora.clone(), 0.5, AdapterKind::Lora),
                (lokr.clone(), 0.75, AdapterKind::Lokr),
            ]),
        ] {
            assert_ne!(
                Sd35LoadReceipt::capture(Sd35Route::Medium, &crossed)
                    .unwrap()
                    .adapter_identity()
                    .unwrap(),
                identity
            );
        }
        std::fs::write(&lora, b"mutated").unwrap();
        assert!(receipt.ensure_unchanged().is_err());
    }

    #[test]
    fn admission_is_request_authoritative_and_drop_cleans_up() {
        let (_temp, root) = fixture(Sd35Route::Medium, "bf16");
        let load = spec(Sd35Route::Medium, &root, Vec::new());
        let receipt = Sd35LoadReceipt::capture(Sd35Route::Medium, &load).unwrap();
        let contract = contract_from_receipt(&load, &receipt);
        let context = context(&receipt, MemoryMode::TextToImage);
        let admission = AdmissionRegistry::new(crate::MODEL_ID_MEDIUM);
        admission.approve(&context).unwrap();
        let scope = Scope::new(
            Sd35Route::Medium,
            Device::Cpu,
            &contract,
            &context,
            admission.clone(),
        )
        .unwrap();
        drop(scope);
        admission.approve(&context).unwrap();
        assert!(Scope::new(
            Sd35Route::Medium,
            Device::Cpu,
            &contract,
            &context,
            admission,
        )
        .is_ok());
    }

    #[test]
    fn complete_cancel_error_and_panic_drop_release_for_warm_and_concurrent_reuse() {
        let (_temp, root) = fixture(Sd35Route::Large, "q4");
        let load = spec(Sd35Route::Large, &root, Vec::new());
        let receipt = Sd35LoadReceipt::capture(Sd35Route::Large, &load).unwrap();
        let contract = contract_from_receipt(&load, &receipt);
        let context = context(&receipt, MemoryMode::TextToImage);
        let admission = AdmissionRegistry::new(crate::MODEL_ID);
        for outcome in [
            MemoryRunOutcome::Complete,
            MemoryRunOutcome::Canceled,
            MemoryRunOutcome::Error {
                message: "expected".into(),
            },
        ] {
            admission.approve(&context).unwrap();
            let mut scope = Scope::new(
                Sd35Route::Large,
                Device::Cpu,
                &contract,
                &context,
                admission.clone(),
            )
            .unwrap();
            assert!(
                admission.approve(&context).is_err(),
                "concurrent reuse must refuse"
            );
            scope.finish(outcome).unwrap();
        }
        admission.approve(&context).unwrap();
        let scope = Scope::new(
            Sd35Route::Large,
            Device::Cpu,
            &contract,
            &context,
            admission.clone(),
        )
        .unwrap();
        drop(scope);
        admission.approve(&context).unwrap();
    }
}
