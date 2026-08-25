//! Candle/CUDA Kolors memory contract.
//!
//! This is intentionally a narrow contract: the registered provider owns only base T2I and the
//! single-reference edit route.  IP-Adapter and pose ControlNet have independent model identities
//! and must never borrow this base receipt or memory evidence.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::gen_core::{
    self, AdapterKind, AdapterSpec, GenerationMemory, GenerationRequest, LoadSpec,
    MemoryAssetFacts, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryComponentKind,
    MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable, MemoryGeometry,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent, MemoryRunContext,
    MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy, MemoryStrategySupport,
    MemoryWindowMaterialization, PidWeights, Precision, Quant, TransformerComponent, WeightsSource,
};
use sha2::{Digest, Sha256};

pub const REQUEST_EVIDENCE_REVISION: &str = "kolors-candle-request-contract-v1";
/// Bespoke IP-Adapter-Plus composition id. It owns a contract but no registered generator, so the
/// catalog reconciles it through an explicit `BespokeMemoryRouteWaiver` rather than a registration.
pub const IP_PROVIDER_ID: &str = "candle_kolors_ipadapter";
/// Bespoke strict-pose ControlNet composition id. Waived for the same reason as [`IP_PROVIDER_ID`].
pub const CONTROL_PROVIDER_ID: &str = "candle_kolors_control";
pub const KOLORS_REPOSITORY: &str = "SceneWorks/kolors-mlx";
pub const KOLORS_REVISION: &str = "aadbd49f53b66a33ef1be09384eac409cbc44061";
pub const IP_REPOSITORY: &str = "Kwai-Kolors/Kolors-IP-Adapter-Plus";
pub const IP_REVISION: &str = "5c72aa86cd8d9d23ff406d293c5473820e09e1d9";
pub const CONTROL_REPOSITORY: &str = "Kwai-Kolors/Kolors-ControlNet-Pose";
pub const CONTROL_REVISION: &str = "83e35a8033a89d2e75044b412d0e2474111578f7";
pub const PID_SDXL_REVISION: &str = "70b494831561dc2c181f04a7f057260b8785419a";
pub const PID_GEMMA_REVISION: &str = "684c553b5b41a1c835989d89f62f585e6269a7de";
pub const PHYSICAL_RECEIPT_PREFIX: &str = "kolors.physical.sha256:";
pub const ADAPTER_RECEIPT_PREFIX: &str = "kolors.adapters.ordered.sha256:";
pub const IP_RECEIPT_PREFIX: &str = "kolors.ip.sha256:";
pub const CONTROL_RECEIPT_PREFIX: &str = "kolors.control.sha256:";
pub const PID_RECEIPT_PREFIX: &str = "kolors.pid.sdxl.sha256:";
const IP_CACHE_NAMESPACE: &str = "ipadapter-kolors";
const CONTROL_CACHE_NAMESPACE: &str = "controlnet-kolors";

#[derive(Clone, Debug)]
struct SealedInventory {
    label: &'static str,
    root: PathBuf,
    paths: Vec<PathBuf>,
    files: Vec<(gen_core::PinnedWeightsFile, [u8; 32])>,
}

#[derive(Clone, Debug)]
pub(crate) struct KolorsLoadSeal {
    contract: MemoryProviderContract,
    tier: MemoryNumericTier,
    inventories: Vec<SealedInventory>,
}

impl KolorsLoadSeal {
    fn capture_inventory(label: &'static str, root: &Path) -> gen_core::Result<SealedInventory> {
        let root = std::path::absolute(root)?;
        let mut paths = Vec::new();
        collect_files(&root, &mut paths)?;
        paths = paths
            .into_iter()
            .map(std::path::absolute)
            .collect::<std::io::Result<Vec<_>>>()?;
        paths.sort();
        paths.dedup();
        if paths.is_empty() {
            return Err(gen_core::Error::Unsupported(format!(
                "kolors: {label} has an empty sealed inventory"
            )));
        }
        let files = paths
            .iter()
            .map(|path| {
                let pin = gen_core::PinnedWeightsFile::pin(path)?;
                let digest = pin.read_unchanged(sha256_file)?;
                Ok((pin, digest))
            })
            .collect::<gen_core::Result<Vec<_>>>()?;
        Ok(SealedInventory {
            label,
            root,
            paths,
            files,
        })
    }

    fn capture(
        provider_id: &str,
        root: &Path,
        adapters: &[AdapterSpec],
        auxiliary: Option<(&str, MemoryComponentKind, &Path, &str, &str, &str)>,
        pid: Option<&PidWeights>,
        contract: MemoryProviderContract,
    ) -> gen_core::Result<Self> {
        validate_base_source(root)?;
        if let Some((_, _, path, repository, revision, namespace)) = auxiliary {
            validate_immutable_component_source(path, repository, revision, namespace)?;
        }
        let tier = physical_tier(root)?;
        validate_float_tensor_inventory(&root.join("vae"), "vae")?;
        let mut inventories = vec![Self::capture_inventory("base", root)?];
        if let Some((_, _, path, _, _, _)) = auxiliary {
            validate_float_tensor_inventory(path, "auxiliary")?;
            inventories.push(Self::capture_inventory("auxiliary", path)?);
        }
        for adapter in adapters {
            validate_float_tensor_inventory(&adapter.path, "ordered-adapter")?;
            inventories.push(Self::capture_inventory("ordered-adapter", &adapter.path)?);
        }
        if let Some(pid) = pid {
            validate_pid_source(pid)?;
            validate_float_tensor_inventory(source_path(&pid.checkpoint), "pid-student")?;
            validate_float_tensor_inventory(source_path(&pid.gemma), "pid-gemma")?;
            inventories.push(Self::capture_inventory(
                "pid-student",
                source_path(&pid.checkpoint),
            )?);
            inventories.push(Self::capture_inventory(
                "pid-gemma",
                source_path(&pid.gemma),
            )?);
        }
        if contract.provider_id != provider_id {
            return Err(gen_core::Error::Unsupported(format!(
                "{provider_id}: sealed provider contract belongs to {}",
                contract.provider_id
            )));
        }
        let seal = Self {
            contract,
            tier,
            inventories,
        };
        seal.ensure_unchanged()?;
        let recaptured = sealed_contract(provider_id, root, adapters, auxiliary, pid)?;
        if recaptured != seal.contract {
            return Err(gen_core::Error::Unsupported(format!(
                "{provider_id}: artifact assembly changed while admission was sealed"
            )));
        }
        seal.ensure_unchanged()?;
        Ok(seal)
    }

    pub(crate) fn capture_load(
        root: &Path,
        adapters: &[AdapterSpec],
        pid: Option<&PidWeights>,
    ) -> gen_core::Result<Self> {
        let contract = sealed_contract(crate::MODEL_ID, root, adapters, None, pid)?;
        Self::capture(crate::MODEL_ID, root, adapters, None, pid, contract)
    }

    pub(crate) fn capture_ip(paths: &crate::IpAdapterKolorsPaths) -> gen_core::Result<Self> {
        let contract = provider_contract_for_ip(paths)?;
        Self::capture(
            "candle_kolors_ipadapter",
            &paths.kolors_base,
            &paths.adapters,
            Some((
                IP_RECEIPT_PREFIX,
                MemoryComponentKind::IpAdapter,
                &paths.ip_adapter,
                IP_REPOSITORY,
                IP_REVISION,
                IP_CACHE_NAMESPACE,
            )),
            None,
            contract,
        )
    }

    pub(crate) fn capture_control(
        paths: &crate::KolorsControlPaths,
        pid: Option<&PidWeights>,
    ) -> gen_core::Result<Self> {
        let contract = provider_contract_for_control(paths, pid)?;
        Self::capture(
            "candle_kolors_control",
            &paths.kolors_base,
            &paths.adapters,
            Some((
                CONTROL_RECEIPT_PREFIX,
                MemoryComponentKind::ControlBranch,
                &paths.controlnet,
                CONTROL_REPOSITORY,
                CONTROL_REVISION,
                CONTROL_CACHE_NAMESPACE,
            )),
            pid,
            contract,
        )
    }

    pub(crate) fn contract(&self) -> &MemoryProviderContract {
        &self.contract
    }

    pub(crate) fn tier(&self) -> MemoryNumericTier {
        self.tier
    }

    pub(crate) fn ensure_unchanged(&self) -> gen_core::Result<()> {
        for inventory in &self.inventories {
            let mut current = Vec::new();
            collect_files(&inventory.root, &mut current)?;
            current = current
                .into_iter()
                .map(std::path::absolute)
                .collect::<std::io::Result<Vec<_>>>()?;
            current.sort();
            current.dedup();
            if current != inventory.paths {
                return Err(gen_core::Error::Unsupported(format!(
                    "kolors: {} inventory changed after admission",
                    inventory.label
                )));
            }
            for (pin, expected_digest) in &inventory.files {
                pin.ensure_unchanged()?;
                let actual = pin.read_unchanged(sha256_file)?;
                if actual != *expected_digest {
                    return Err(gen_core::Error::Unsupported(format!(
                        "kolors: {} content changed after admission: {}",
                        inventory.label,
                        pin.loader_path().display()
                    )));
                }
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct ComponentReceipt {
    bytes: u64,
    runtime_bytes: u64,
    digest: Sha256,
}

fn update_framed(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn hex_digest(digest: Sha256) -> String {
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> gen_core::Result<()> {
    let lexical = std::path::absolute(path)?;
    let symlink = std::fs::symlink_metadata(&lexical).map_err(|error| {
        gen_core::Error::Msg(format!(
            "kolors receipt cannot inspect {}: {error}",
            lexical.display()
        ))
    })?;
    let metadata = std::fs::metadata(&lexical)?;
    if symlink.file_type().is_symlink() && !metadata.is_file() {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors receipt only permits symlinks that resolve directly to files: {}",
            lexical.display()
        )));
    }
    if metadata.is_file() {
        let name = lexical
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if name.starts_with('.')
            || [".part", ".tmp", ".lock", ".incomplete"]
                .iter()
                .any(|suffix| name.ends_with(suffix))
        {
            return Err(gen_core::Error::Unsupported(format!(
                "kolors receipt refuses incomplete/hidden artifact {}",
                lexical.display()
            )));
        }
        files.push(lexical);
        return Ok(());
    }
    if !metadata.is_dir() {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors receipt refuses non-file source {}",
            lexical.display()
        )));
    }
    let mut entries = std::fs::read_dir(&lexical)
        .map_err(|error| {
            gen_core::Error::Msg(format!(
                "kolors receipt reads {}: {error}",
                lexical.display()
            ))
        })?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            gen_core::Error::Msg(format!("kolors receipt enumeration failed: {error}"))
        })?;
    entries.sort();
    for entry in entries {
        collect_files(&entry, files)?;
    }
    Ok(())
}

fn sha256_file(path: &Path) -> gen_core::Result<[u8; 32]> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(digest.finalize().into())
}

fn validate_float_tensor_inventory(path: &Path, label: &str) -> gen_core::Result<()> {
    let mut files = Vec::new();
    collect_files(path, &mut files)?;
    files.retain(|file| file.extension().and_then(|ext| ext.to_str()) == Some("safetensors"));
    if files.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors: {label} has no safetensors tensor inventory"
        )));
    }
    let mut tensor_count = 0_usize;
    for file in files {
        for header in gen_core::weightsmeta::safetensors_path_tensor_headers(&file)? {
            tensor_count += 1;
            if !header.is_float() {
                return Err(gen_core::Error::Unsupported(format!(
                    "kolors: {label} tensor {} uses non-floating {:?} storage",
                    header.name, header.dtype
                )));
            }
        }
    }
    if tensor_count == 0 {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors: {label} tensor inventory is empty"
        )));
    }
    Ok(())
}

fn receipt_for(label: &str, path: &Path) -> gen_core::Result<ComponentReceipt> {
    let mut files = Vec::new();
    collect_files(path, &mut files)?;
    if files.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors receipt component {label} is empty"
        )));
    }
    files.sort();
    let mut receipt = ComponentReceipt::default();
    update_framed(&mut receipt.digest, label.as_bytes());
    for file in files {
        let relative = file.strip_prefix(path).unwrap_or(file.as_path());
        update_framed(&mut receipt.digest, relative.to_string_lossy().as_bytes());
        let mut reader = BufReader::new(File::open(&file).map_err(|error| {
            gen_core::Error::Msg(format!("kolors receipt opens {}: {error}", file.display()))
        })?);
        let mut file_digest = Sha256::new();
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let count = reader.read(&mut buffer).map_err(|error| {
                gen_core::Error::Msg(format!("kolors receipt reads {}: {error}", file.display()))
            })?;
            if count == 0 {
                break;
            }
            receipt.bytes = receipt.bytes.saturating_add(count as u64);
            file_digest.update(&buffer[..count]);
        }
        update_framed(&mut receipt.digest, &file_digest.finalize());
        if file.extension().and_then(|value| value.to_str()) == Some("safetensors") {
            receipt.runtime_bytes = receipt
                .runtime_bytes
                .saturating_add(f32_runtime_tensor_bytes(&file)?);
        }
    }
    Ok(receipt)
}

fn f32_runtime_tensor_bytes(path: &Path) -> gen_core::Result<u64> {
    let mut reader = BufReader::new(File::open(path).map_err(|error| {
        gen_core::Error::Msg(format!(
            "kolors receipt opens header {}: {error}",
            path.display()
        ))
    })?);
    let mut length = [0_u8; 8];
    reader.read_exact(&mut length).map_err(|error| {
        gen_core::Error::Msg(format!(
            "kolors receipt reads header {}: {error}",
            path.display()
        ))
    })?;
    let header_len = usize::try_from(u64::from_le_bytes(length)).map_err(|_| {
        gen_core::Error::Unsupported(format!(
            "kolors receipt header too large: {}",
            path.display()
        ))
    })?;
    let mut header = vec![0_u8; header_len];
    reader.read_exact(&mut header).map_err(|error| {
        gen_core::Error::Msg(format!(
            "kolors receipt reads tensor index {}: {error}",
            path.display()
        ))
    })?;
    let index: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)
        .map_err(|error| {
            gen_core::Error::Msg(format!("kolors receipt parses {}: {error}", path.display()))
        })?;
    index
        .into_iter()
        .filter(|(name, _)| name != "__metadata__")
        .try_fold(0_u64, |total, (name, value)| {
            let dtype = value
                .get("dtype")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    gen_core::Error::Unsupported(format!("kolors tensor {name} omits dtype"))
                })?;
            let width = match dtype {
                "F64" | "I64" | "U64" => 8,
                // Every floating tensor is materialized through this provider's F32 VarBuilder.
                "F32" | "F16" | "BF16" | "I32" | "U32" => 4,
                "I16" | "U16" => 2,
                "I8" | "U8" | "BOOL" => 1,
                other => {
                    return Err(gen_core::Error::Unsupported(format!(
                        "kolors tensor {name} uses unsupported dtype {other}"
                    )))
                }
            };
            let elements = value
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    gen_core::Error::Unsupported(format!("kolors tensor {name} omits shape"))
                })?
                .iter()
                .try_fold(1_u64, |product, dimension| {
                    dimension
                        .as_u64()
                        .and_then(|dimension| product.checked_mul(dimension))
                        .ok_or_else(|| {
                            gen_core::Error::Unsupported(format!(
                                "kolors tensor {name} shape overflows"
                            ))
                        })
                })?;
            total
                .checked_add(elements.saturating_mul(width))
                .ok_or_else(|| {
                    gen_core::Error::Unsupported("kolors runtime tensor bytes overflow".into())
                })
        })
}

fn source_path(source: &WeightsSource) -> &Path {
    match source {
        WeightsSource::Dir(path) | WeightsSource::File(path) => path,
    }
}

fn adapter_receipt(adapters: &[AdapterSpec]) -> gen_core::Result<Option<ComponentReceipt>> {
    if adapters.is_empty() {
        return Ok(None);
    }
    let mut combined = ComponentReceipt::default();
    update_framed(&mut combined.digest, b"ordered-kolors-adapters-v1");
    for (index, adapter) in adapters.iter().enumerate() {
        let item = receipt_for("adapter", &adapter.path)?;
        combined.bytes = combined.bytes.saturating_add(item.bytes);
        combined.runtime_bytes = combined.runtime_bytes.saturating_add(item.runtime_bytes);
        update_framed(&mut combined.digest, &(index as u64).to_le_bytes());
        update_framed(
            &mut combined.digest,
            match adapter.kind {
                AdapterKind::Lora => b"lora",
                AdapterKind::Lokr => b"lokr",
            },
        );
        update_framed(&mut combined.digest, &adapter.scale.to_bits().to_le_bytes());
        update_framed(&mut combined.digest, &item.digest.finalize());
    }
    Ok(Some(combined))
}

fn pid_receipt(pid: Option<&PidWeights>) -> gen_core::Result<Option<ComponentReceipt>> {
    let Some(pid) = pid else { return Ok(None) };
    validate_pid_source(pid)?;
    let student = receipt_for("pid-student", source_path(&pid.checkpoint))?;
    let gemma = receipt_for("pid-gemma", source_path(&pid.gemma))?;
    let mut combined = ComponentReceipt {
        bytes: student.bytes.saturating_add(gemma.bytes),
        runtime_bytes: student.runtime_bytes.saturating_add(gemma.runtime_bytes),
        ..Default::default()
    };
    update_framed(&mut combined.digest, b"kolors-pid-sdxl-v1");
    update_framed(&mut combined.digest, &student.digest.finalize());
    update_framed(&mut combined.digest, &gemma.digest.finalize());
    Ok(Some(combined))
}

fn validate_pid_source(pid: &PidWeights) -> gen_core::Result<()> {
    let WeightsSource::File(student) = &pid.checkpoint else {
        return Err(gen_core::Error::Unsupported(
            "kolors: PiD SDXL student must be one exact safetensors file".into(),
        ));
    };
    let WeightsSource::Dir(gemma) = &pid.gemma else {
        return Err(gen_core::Error::Unsupported(
            "kolors: PiD Gemma must be one exact snapshot directory".into(),
        ));
    };
    let filename = student
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if filename != "pid_sdxl_2kto4k.safetensors"
        || !(path_has_suffix(
            student,
            &[
                "models--SceneWorks--pid-sdxl",
                "snapshots",
                PID_SDXL_REVISION,
                filename,
            ],
        ) || path_has_suffix(
            student,
            &["SceneWorks__pid-sdxl", PID_SDXL_REVISION, filename],
        ) || path_has_suffix(student, &["pid-sdxl", PID_SDXL_REVISION, filename]))
        || !(path_has_suffix(
            gemma,
            &[
                "models--SceneWorks--gemma-2-2b-it",
                "snapshots",
                PID_GEMMA_REVISION,
            ],
        ) || path_has_suffix(gemma, &["SceneWorks__gemma-2-2b-it", PID_GEMMA_REVISION])
            || path_has_suffix(gemma, &["gemma-2-2b-it", PID_GEMMA_REVISION]))
    {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors: PiD requires exact SceneWorks/pid-sdxl@{PID_SDXL_REVISION} and SceneWorks/gemma-2-2b-it@{PID_GEMMA_REVISION}"
        )));
    }
    Ok(())
}

fn sealed_contract(
    provider_id: &str,
    root: &Path,
    adapters: &[AdapterSpec],
    auxiliary: Option<(&str, MemoryComponentKind, &Path, &str, &str, &str)>,
    pid: Option<&PidWeights>,
) -> gen_core::Result<MemoryProviderContract> {
    validate_base_source(root)?;
    let _ = physical_tier(root)?;
    let conditioning_tokenizer = receipt_for("kolors-tokenizer", &root.join("tokenizer"))?;
    let conditioning_text = receipt_for("kolors-chatglm", &root.join("text_encoder"))?;
    let transformer = receipt_for("kolors-unet", &root.join("unet"))?;
    let decoder = receipt_for("kolors-vae-f32", &root.join("vae"))?;
    let conditioning_bytes = conditioning_text.runtime_bytes;
    if conditioning_bytes == 0 || transformer.runtime_bytes == 0 || decoder.runtime_bytes == 0 {
        return Err(gen_core::Error::Unsupported(
            "kolors receipt has an empty runtime component inventory".into(),
        ));
    }
    let mut physical = Sha256::new();
    update_framed(&mut physical, KOLORS_REPOSITORY.as_bytes());
    update_framed(&mut physical, KOLORS_REVISION.as_bytes());
    update_framed(&mut physical, &conditioning_tokenizer.digest.finalize());
    update_framed(&mut physical, &conditioning_text.digest.finalize());
    update_framed(&mut physical, &transformer.digest.finalize());
    update_framed(&mut physical, &decoder.digest.finalize());
    let mut components = vec![MemoryResidentComponent {
        id: format!("{PHYSICAL_RECEIPT_PREFIX}{}", hex_digest(physical)),
        kind: MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
        resident_bytes: transformer.runtime_bytes,
        bounded_by: Some(MemoryStrategy::StagedResidency),
        residency: MemoryComponentResidency::WholeRender,
    }];
    let mut overlay_bytes = 0_u64;
    if let Some(adapter) = adapter_receipt(adapters)? {
        overlay_bytes = overlay_bytes.saturating_add(adapter.runtime_bytes);
        components.push(MemoryResidentComponent {
            id: format!("{ADAPTER_RECEIPT_PREFIX}{}", hex_digest(adapter.digest)),
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: adapter.runtime_bytes,
            bounded_by: Some(MemoryStrategy::StagedResidency),
            residency: MemoryComponentResidency::WholeRender,
        });
    }
    if let Some((prefix, kind, path, repository, revision, cache_namespace)) = auxiliary {
        let canonical =
            validate_immutable_component_source(path, repository, revision, cache_namespace)?;
        let receipt = receipt_for(prefix, &canonical)?;
        overlay_bytes = overlay_bytes.saturating_add(receipt.runtime_bytes);
        components.push(MemoryResidentComponent {
            id: format!("{prefix}{}", hex_digest(receipt.digest)),
            kind,
            resident_bytes: receipt.runtime_bytes,
            bounded_by: Some(MemoryStrategy::StagedResidency),
            residency: MemoryComponentResidency::WholeRender,
        });
    }
    if let Some(pid) = pid_receipt(pid)? {
        overlay_bytes = overlay_bytes.saturating_add(pid.runtime_bytes);
        components.push(MemoryResidentComponent {
            id: format!("{PID_RECEIPT_PREFIX}{}", hex_digest(pid.digest)),
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: pid.runtime_bytes,
            bounded_by: Some(MemoryStrategy::StagedResidency),
            residency: MemoryComponentResidency::WholeRender,
        });
    }
    let mut contract = provider_contract_for(provider_id);
    contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::OverlayBytes,
        ],
        resident_components: components,
    };
    contract.asset_facts = MemoryAssetFacts {
        conditioning_bytes,
        transformer_bytes: transformer.runtime_bytes,
        decoder_bytes: decoder.runtime_bytes,
        base_bytes: conditioning_bytes
            .saturating_add(transformer.runtime_bytes)
            .saturating_add(decoder.runtime_bytes),
        overlay_bytes,
    };
    Ok(contract)
}

fn path_has_suffix(path: &Path, suffix: &[&str]) -> bool {
    let components = path
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    components.ends_with(
        &suffix
            .iter()
            .map(|part| (*part).to_owned())
            .collect::<Vec<_>>(),
    )
}

fn validate_base_source(root: &Path) -> gen_core::Result<PathBuf> {
    let canonical = std::fs::canonicalize(root).map_err(|error| {
        gen_core::Error::Msg(format!(
            "kolors cannot canonicalize source {}: {error}",
            root.display()
        ))
    })?;
    let tier = canonical
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| gen_core::Error::Unsupported("kolors source has no UTF-8 tier".into()))?;
    if !matches!(tier, "q4" | "q8" | "bf16") {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors exact source tier must be q4, q8, or bf16, got {tier}"
        )));
    }
    let valid = path_has_suffix(
        &canonical,
        &[
            "models--SceneWorks--kolors-mlx",
            "snapshots",
            KOLORS_REVISION,
            tier,
        ],
    ) || path_has_suffix(
        &canonical,
        &["SceneWorks__kolors-mlx", KOLORS_REVISION, tier],
    ) || path_has_suffix(&canonical, &["kolors-mlx", KOLORS_REVISION, tier]);
    if !valid {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors source must resolve to exact immutable {KOLORS_REPOSITORY}@{KOLORS_REVISION}/{tier}"
        )));
    }
    Ok(canonical)
}

fn validate_immutable_component_source(
    path: &Path,
    repository: &str,
    revision: &str,
    cache_namespace: &str,
) -> gen_core::Result<PathBuf> {
    let lexical = std::path::absolute(path)?;
    std::fs::canonicalize(&lexical).map_err(|error| {
        gen_core::Error::Msg(format!(
            "kolors cannot canonicalize component source {}: {error}",
            path.display()
        ))
    })?;
    let repository_name = repository.rsplit('/').next().unwrap_or(repository);
    let repository_marker = format!("models--{}", repository.replace('/', "--"));
    let components = lexical
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let has_snapshot = components
        .windows(3)
        .any(|window| window == [repository_marker.as_str(), "snapshots", revision]);
    let has_app_cache = components
        .windows(2)
        .any(|window| window == [cache_namespace, revision]);
    let has_plain = components
        .windows(2)
        .any(|window| window == [repository_name, revision]);
    if !(has_snapshot || has_app_cache || has_plain) {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors component source must resolve inside immutable {repository}@{revision}"
        )));
    }
    Ok(lexical)
}

pub fn provider_contract_for_load(
    root: &Path,
    adapters: &[AdapterSpec],
    pid: Option<&PidWeights>,
) -> gen_core::Result<MemoryProviderContract> {
    sealed_contract(crate::MODEL_ID, root, adapters, None, pid)
}

pub fn provider_contract_for_ip(
    paths: &crate::IpAdapterKolorsPaths,
) -> gen_core::Result<MemoryProviderContract> {
    sealed_contract(
        IP_PROVIDER_ID,
        &paths.kolors_base,
        &paths.adapters,
        Some((
            IP_RECEIPT_PREFIX,
            MemoryComponentKind::IpAdapter,
            &paths.ip_adapter,
            IP_REPOSITORY,
            IP_REVISION,
            IP_CACHE_NAMESPACE,
        )),
        None,
    )
}

pub fn provider_contract_for_control(
    paths: &crate::KolorsControlPaths,
    pid: Option<&PidWeights>,
) -> gen_core::Result<MemoryProviderContract> {
    sealed_contract(
        CONTROL_PROVIDER_ID,
        &paths.kolors_base,
        &paths.adapters,
        Some((
            CONTROL_RECEIPT_PREFIX,
            MemoryComponentKind::ControlBranch,
            &paths.controlnet,
            CONTROL_REPOSITORY,
            CONTROL_REVISION,
            CONTROL_CACHE_NAMESPACE,
        )),
        pid,
    )
}

pub fn provider_overlay_identity(contract: &MemoryProviderContract) -> Option<String> {
    let prefixes = [
        PHYSICAL_RECEIPT_PREFIX,
        ADAPTER_RECEIPT_PREFIX,
        IP_RECEIPT_PREFIX,
        CONTROL_RECEIPT_PREFIX,
        PID_RECEIPT_PREFIX,
    ];
    let identities = contract
        .resident_components()
        .iter()
        .filter(|component| {
            prefixes
                .iter()
                .any(|prefix| component.id.starts_with(prefix))
        })
        .map(|component| component.id.as_str())
        .collect::<Vec<_>>();
    (!identities.is_empty()).then(|| identities.join("+"))
}

/// Per-provider calibration fingerprint.
///
/// The three Kolors contracts share one physical base but are deliberately independent evidence
/// identities (base T2I/edit, IP-Adapter, strict-pose ControlNet). A single shared fingerprint let
/// any one of them satisfy another's calibration check, so each mints its own route suffix — the
/// same shape SDXL/Anima use (`sdxl-candle-{route}-...`).
fn calibration_fingerprint(provider_id: &str) -> String {
    let route = match provider_id {
        IP_PROVIDER_ID => "ipadapter",
        CONTROL_PROVIDER_ID => "control",
        // The registered base id (and any future route) names itself; no two ids can collide.
        other => other,
    };
    format!("kolors-candle-{route}-staged-chatglm-unet-f32-vae-v1")
}

pub fn provider_contract() -> MemoryProviderContract {
    provider_contract_for(crate::MODEL_ID)
}

/// Bespoke IP-Adapter and ControlNet routes deliberately mint independent contracts; they may share
/// a physical Kolors base but never a provider/evidence identity.
pub fn provider_contract_for(provider_id: &str) -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    contract.lifecycle = MemoryLifecycleCapabilities {
        // Edit has a VAE-init phase, but it is deliberately represented by the decoder/VAE phase:
        // no additional advertised optimization rung is claimed for it.
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        decode_tiling: false,
        attention_chunking: false,
        transformer_window_materialization: false,
    };
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::OverlayBytes,
        ],
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        calibration_fingerprint(provider_id),
        gen_core::LoadShape::EagerMaterialization,
    ));
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident | MemoryStrategy::StagedResidency => {
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedDecode
            | MemoryStrategy::BoundedAttention
            | MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
        };
    }
    contract
}

fn component_quantization(root: &Path, component: &str) -> gen_core::Result<Option<(u8, usize)>> {
    let bytes = std::fs::read(root.join(component).join("config.json")).map_err(|error| {
        gen_core::Error::Msg(format!("kolors: read physical {component} tier: {error}"))
    })?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        gen_core::Error::Msg(format!("kolors: parse physical {component} tier: {error}"))
    })?;
    let Some(quantization) = value.get("quantization") else {
        return Ok(None);
    };
    let bits = quantization
        .get("bits")
        .and_then(serde_json::Value::as_u64)
        .and_then(|bits| u8::try_from(bits).ok())
        .ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "kolors: packed {component} lacks valid quantization.bits"
            ))
        })?;
    let group = quantization
        .get("group_size")
        .and_then(serde_json::Value::as_u64)
        .and_then(|group| usize::try_from(group).ok())
        .ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "kolors: packed {component} lacks valid quantization.group_size"
            ))
        })?;
    Ok(Some((bits, group)))
}

fn inspect_component_tier(root: &Path, component: &str) -> gen_core::Result<Option<(u8, usize)>> {
    use gen_core::weightsmeta::Dtype;
    let dir = root.join(component);
    let mut files = Vec::new();
    collect_files(&dir, &mut files)?;
    files.retain(|path| path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"));
    if files.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors: {component} has no safetensors inventory"
        )));
    }
    let mut headers = BTreeMap::new();
    for file in files {
        for header in gen_core::weightsmeta::safetensors_path_tensor_headers(&file)? {
            if headers.insert(header.name.clone(), header).is_some() {
                return Err(gen_core::Error::Unsupported(format!(
                    "kolors: duplicate {component} tensor across shards"
                )));
            }
        }
    }
    let packed_bases = headers
        .keys()
        .filter_map(|name| name.strip_suffix(".scales").map(str::to_owned))
        .collect::<BTreeSet<_>>();
    let configured = component_quantization(root, component)?;
    if packed_bases.is_empty() {
        if configured.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "kolors: {component} config claims packing but tensor headers are dense"
            )));
        }
        if headers.is_empty() || headers.values().any(|header| header.dtype != Dtype::BF16) {
            return Err(gen_core::Error::Unsupported(format!(
                "kolors: dense {component} must contain BF16 tensors only"
            )));
        }
        return Ok(None);
    }
    let Some((bits, group)) = configured else {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors: packed {component} headers have no matching quantization config"
        )));
    };
    if !matches!(bits, 4 | 8) || group != candle_gen::quant::MLX_GROUP_SIZE {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors: unsupported packed {component} {bits}-bit/group-{group}"
        )));
    }
    for base in &packed_bases {
        let weight = headers.get(&format!("{base}.weight")).ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "kolors: packed {component} {base} is missing .weight"
            ))
        })?;
        let scales = headers
            .get(&format!("{base}.scales"))
            .expect("base came from scales");
        let biases = headers.get(&format!("{base}.biases")).ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "kolors: packed {component} {base} is missing .biases"
            ))
        })?;
        if weight.dtype != Dtype::U32 || scales.dtype != Dtype::BF16 || biases.dtype != Dtype::BF16
        {
            return Err(gen_core::Error::Unsupported(format!(
                "kolors: packed {component} {base} must use U32/BF16/BF16"
            )));
        }
        let [weight_rows, packed_columns] = weight.shape.as_slice() else {
            return Err(gen_core::Error::Unsupported(format!(
                "kolors: packed {component} {base}.weight must be rank 2"
            )));
        };
        let [scale_rows, scale_columns] = scales.shape.as_slice() else {
            return Err(gen_core::Error::Unsupported(format!(
                "kolors: packed {component} {base}.scales must be rank 2"
            )));
        };
        let input = scale_columns.checked_mul(group).ok_or_else(|| {
            gen_core::Error::Unsupported("kolors: packed input width overflow".into())
        })?;
        let encoded = packed_columns.checked_mul(32).ok_or_else(|| {
            gen_core::Error::Unsupported("kolors: packed tensor width overflow".into())
        })?;
        let resolved = encoded
            .checked_div(input)
            .filter(|_| input > 0 && encoded.is_multiple_of(input));
        if weight_rows != scale_rows
            || resolved != Some(usize::from(bits))
            || scales.shape != biases.shape
        {
            return Err(gen_core::Error::Unsupported(format!(
                "kolors: packed {component} {base} crosses config width or affine geometry"
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
                "kolors: non-packed {component} tensor {name} must be BF16"
            )));
        }
    }
    Ok(Some((bits, group)))
}

/// The selected numeric tier comes from physical tensor headers plus matching configuration, never
/// from a requested UI label. q4/q8 are valid only when ChatGLM and UNet have identical packing.
pub fn physical_tier(root: &Path) -> gen_core::Result<MemoryNumericTier> {
    let text = inspect_component_tier(root, "text_encoder")?;
    let unet = inspect_component_tier(root, "unet")?;
    if text != unet {
        return Err(gen_core::Error::Unsupported(
            "kolors: ChatGLM and UNet physical tensor tiers differ".into(),
        ));
    }
    let quant = text.map(|(bits, _)| if bits == 4 { Quant::Q4 } else { Quant::Q8 });
    // The canonical bf16 snapshot executes Candle's exact F32 loader recipe and F32 VAE.  The
    // numeric identity stays BF16 (physical bytes), while execution precision is documented by the
    // provider's receipt/phase implementation rather than pretending the source artifact is F32.
    Ok(MemoryNumericTier {
        precision: Precision::Bf16,
        quant,
        component_precision_floors: &[],
    })
}

pub fn validate_context(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    tier: MemoryNumericTier,
) -> gen_core::Result<()> {
    if context.evidence_revision != REQUEST_EVIDENCE_REVISION {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors: evidence revision {} does not match {}",
            context.evidence_revision, REQUEST_EVIDENCE_REVISION
        )));
    }
    let receipt_has_pid = contract
        .resident_components()
        .iter()
        .any(|component| component.id.starts_with(PID_RECEIPT_PREFIX));
    if context.use_pid != receipt_has_pid {
        return Err(gen_core::Error::Unsupported(
            "kolors: crossed registered PiD receipt".into(),
        ));
    }
    if context.overlay != provider_overlay_identity(contract) {
        return Err(gen_core::Error::Unsupported(
            "kolors: crossed registered ordered-adapter/PiD receipt".into(),
        ));
    }
    let route = || match context.mode {
        MemoryMode::TextToImage
            if !context.has_reference && context.geometry.reference_count == 0 =>
        {
            Ok(())
        }
        MemoryMode::Edit | MemoryMode::ImageToImage
            if context.has_reference && context.geometry.reference_count == 1 =>
        {
            Ok(())
        }
        _ => Err(gen_core::Error::Unsupported(
            "kolors: memory evidence is bound only to base T2I or one-reference edit".into(),
        )),
    };
    match gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(tier),
        Some(&route),
    ) {
        MemorySafetyDecision::Accept => Ok(()),
        MemorySafetyDecision::Reject { reason } => Err(gen_core::Error::Unsupported(reason)),
    }
}

pub fn validate_bespoke_context(
    contract: &MemoryProviderContract,
    root: &Path,
    context: &MemoryRunContext,
    require_reference: bool,
    require_pid: bool,
) -> gen_core::Result<()> {
    validate_bespoke_context_with_tier(
        contract,
        context,
        physical_tier(root)?,
        require_reference,
        require_pid,
    )
}

/// The bespoke admission with the numeric tier supplied rather than read off disk.
///
/// The loaded providers pass the physically detected tier; the registry's composed pre-load probe
/// passes the tier its [`LoadSpec`] states, because a bespoke composition is assembled from typed
/// path structs the registry never sees.
fn validate_bespoke_context_with_tier(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    tier: MemoryNumericTier,
    require_reference: bool,
    require_pid: bool,
) -> gen_core::Result<()> {
    if context.evidence_revision != REQUEST_EVIDENCE_REVISION {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: evidence revision {} does not match {}",
            contract.provider_id, context.evidence_revision, REQUEST_EVIDENCE_REVISION
        )));
    }
    if context.has_reference != require_reference || context.use_pid != require_pid {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: crossed bespoke reference/PiD receipt",
            contract.provider_id
        )));
    }
    if context.overlay != provider_overlay_identity(contract) {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: crossed exact overlay/adapter receipt",
            contract.provider_id
        )));
    }
    let route_matches = match contract.provider_id.as_str() {
        IP_PROVIDER_ID => {
            context.mode == MemoryMode::Other("character_image".to_owned())
                && context.geometry.reference_count == 1
        }
        // sc-20762 review: the strict-pose route consumes exactly one conditioning image — the
        // rendered OpenPose skeleton `KolorsControl::generate` takes by value and embeds once. The
        // memory key must state that image, so this route is admitted at `reference_count == 1`
        // (never 0, which claimed a conditioning-free render the provider cannot perform). The
        // control branch's own identity travels on `context.overlay`, checked above against the
        // sealed `kolors.control.sha256:` receipt component.
        CONTROL_PROVIDER_ID => {
            (context.mode == MemoryMode::TextToImage
                || matches!(&context.mode, MemoryMode::Other(mode)
                    if mode == "style_variations" || mode == "character_image"))
                && context.geometry.reference_count == 1
        }
        _ => false,
    };
    if !route_matches {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: crossed bespoke mode/reference receipt",
            contract.provider_id
        )));
    }
    match gen_core::standard_memory_strategy_safety_check(contract, context, Some(tier), None) {
        MemorySafetyDecision::Accept => Ok(()),
        MemorySafetyDecision::Reject { reason } => Err(gen_core::Error::Unsupported(reason)),
    }
}

pub fn request_memory(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> Option<GenerationMemory> {
    contract.generation_memory(&context.selection)
}

/// Open the shared Candle request lifecycle for one bespoke IP-Adapter / ControlNet run.
///
/// The bespoke routes are driven directly by the worker rather than through the registry, so they
/// previously ran with no [`MemoryRequestScope`] at all: no geometry bind between the admitted
/// memory key and the request actually rendered, no cancellation/error cleanup contract, and no
/// double-finish rejection. Routing them through
/// [`CandleRequestScopeCore`](candle_gen::request_scope::CandleRequestScopeCore) — the same core the
/// registered route and every SDXL bespoke route use — closes all three (sc-20762 review).
///
/// The caller binds the exact request axes here, then finishes the returned scope with the run's
/// [`MemoryRunOutcome`]; `finish` performs the device synchronization the contract's cleanup
/// semantics promise, and `Drop` performs it for an early return.
pub fn bespoke_request_scope(
    provider_id: &'static str,
    device: candle_gen::candle_core::Device,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    width: u32,
    height: u32,
    use_pid: bool,
) -> gen_core::Result<candle_gen::request_scope::CandleRequestScopeCore> {
    if contract.provider_id != provider_id {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: admitted contract belongs to {}",
            contract.provider_id
        )));
    }
    if context.geometry.width != width
        || context.geometry.height != height
        || context.geometry.batch != 1
        || context.geometry.frames != 1
        || context.use_pid != use_pid
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: request {width}x{height} pid={use_pid} does not equal the admitted \
             {}x{}x{} frames={} pid={}",
            context.geometry.width,
            context.geometry.height,
            context.geometry.batch,
            context.geometry.frames,
            context.use_pid
        )));
    }
    route_request_scope(provider_id, device, contract, context)
}

/// The shared scope core every Kolors route opens: bounded decode is `Missing` on all three, so the
/// decode validator refuses every tile rather than silently accepting one.
fn route_request_scope(
    provider_id: &'static str,
    device: candle_gen::candle_core::Device,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<candle_gen::request_scope::CandleRequestScopeCore> {
    let config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        provider_id,
        device,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        1,
        move |_pid, edge, overlap| {
            Err(gen_core::Error::Unsupported(format!(
                "{provider_id}: bounded decode is Missing; tile {edge}/{overlap} was never admitted"
            )))
        },
    )?;
    Ok(candle_gen::request_scope::CandleRequestScopeCore::new(
        config,
    ))
}

// -------------------------------------------------------------------------------------------
// Registry seams (sc-20762 review).
//
// The crate previously registered no memory route at all, so a selector could not price Kolors
// before load: the contract existed only on an already-constructed `KolorsGenerator`. These
// functions are the weights-free, CUDA-free half of the same admission the loaded generator runs.
// -------------------------------------------------------------------------------------------

/// The numeric tier a weights-free probe can state without opening a single weight file.
///
/// A packed tier is auto-detected from disk at load (sc-10819), so `LoadSpec::quantize` is only
/// advisory on a real snapshot; with no snapshot present it is the sole tier evidence there is.
fn weights_free_tier(spec: &LoadSpec) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: Precision::Bf16,
        quant: spec.quantize,
        component_precision_floors: &[],
    }
}

/// The registered route's authoritative load seal, or `None` when the lazy root is absent.
///
/// This mirrors `crate::load` exactly: a present directory is sealed and priced from its real
/// bytes; a missing one is the registry-introspection case the loader also permits.
fn registered_seal(spec: &LoadSpec) -> gen_core::Result<Option<KolorsLoadSeal>> {
    match &spec.weights {
        WeightsSource::Dir(root) if root.exists() => Ok(Some(KolorsLoadSeal::capture_load(
            root,
            &spec.adapters,
            spec.pid.as_ref(),
        )?)),
        _ => Ok(None),
    }
}

/// Reproduce the loaded generator's contract from a [`LoadSpec`] alone.
pub fn registered_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    Ok(match registered_seal(spec)? {
        Some(seal) => seal.contract().clone(),
        None => provider_contract(),
    })
}

/// The exact admission `KolorsGenerator::memory_strategy_safety_check` runs, callable pre-load.
pub fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let result = registered_seal(spec).and_then(|seal| match seal {
        Some(seal) => {
            seal.ensure_unchanged()?;
            validate_context(contract, context, seal.tier())
        }
        None => validate_context(contract, context, weights_free_tier(spec)),
    });
    match result {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub fn registered_valid_fixtures(
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
    let mut context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        weights_free_tier(spec),
        gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: provider_overlay_identity(contract),
        },
    )?;
    // The shared builder stamps its own revision string; Kolors admission is bound to the
    // provider's own, so state it rather than let the fixture be rejected for the wrong reason.
    context.evidence_revision = REQUEST_EVIDENCE_REVISION.to_owned();
    Ok(vec![gen_core::MemoryBehaviorFixture::new(context)])
}

pub fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope>>> {
    match registered_seal(spec)? {
        Some(seal) => {
            seal.ensure_unchanged()?;
            validate_context(contract, context, seal.tier())?;
        }
        None => validate_context(contract, context, weights_free_tier(spec))?,
    }
    Ok(Some(Box::new(route_request_scope(
        crate::MODEL_ID,
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
        contract: registered_contract,
        safety_check: registered_safety_check,
    }
}

pub(crate) fn surface_specs() -> Vec<gen_core::MemoryContractSurfaceSpec> {
    gen_core::candle_memory_contract_surface_specs()
}

/// The declaration-only contract catalog conformance uses when no snapshot is on disk. It carries
/// the same route/strategy declaration as the sealed contract and injects zero asset facts.
pub fn weights_free_contract(_spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    Ok(provider_contract())
}

// -------------------------------------------------------------------------------------------
// Composed (generator-less) routes: IP-Adapter-Plus and strict-pose ControlNet.
//
// Both are assembled from typed path structs by the worker rather than resolved through
// `load(id, spec)`, so neither has — or should invent — a generator descriptor.
// `ProviderRegistryBuilder::register_composed_memory_strategy` is the seam gen-core provides for
// exactly that shape (the same one `z_image_control` uses); an ordinary
// `register_memory_strategy` would be rejected by `build` for having no matching generator.
// -------------------------------------------------------------------------------------------

/// The PiD identity the registry can read off a contract, matching `validate_context`'s rule.
fn contract_carries_pid(contract: &MemoryProviderContract) -> bool {
    contract
        .resident_components()
        .iter()
        .any(|component| component.id.starts_with(PID_RECEIPT_PREFIX))
}

fn composed_route_id(contract: &MemoryProviderContract) -> gen_core::Result<&'static str> {
    match contract.provider_id.as_str() {
        IP_PROVIDER_ID => Ok(IP_PROVIDER_ID),
        CONTROL_PROVIDER_ID => Ok(CONTROL_PROVIDER_ID),
        other => Err(gen_core::Error::Unsupported(format!(
            "kolors: {other} is not a bespoke Kolors composition"
        ))),
    }
}

/// The declaration-only pre-load contract for a bespoke composition.
///
/// The sealed contract with real asset facts is minted at load by [`provider_contract_for_ip`] /
/// [`provider_contract_for_control`] from the caller's typed paths — a `LoadSpec` cannot name them —
/// so the registry's answer is the route/strategy declaration with zero asset facts.
pub fn ip_composed_contract(_spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    Ok(provider_contract_for(IP_PROVIDER_ID))
}

pub fn control_composed_contract(_spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    Ok(provider_contract_for(CONTROL_PROVIDER_ID))
}

/// The bespoke admission, callable before any weight file exists. Both routes consume exactly one
/// conditioning image, so `require_reference` is unconditionally true.
pub fn composed_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let result = composed_route_id(contract).and_then(|_| {
        validate_bespoke_context_with_tier(
            contract,
            context,
            weights_free_tier(spec),
            true,
            contract_carries_pid(contract),
        )
    });
    match result {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub fn composed_valid_fixtures(
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
    let mode = match composed_route_id(contract)? {
        // The IP route is reached only through the worker's character-image stream; the pose route
        // conditions an otherwise text-to-image render.
        IP_PROVIDER_ID => MemoryMode::Other("character_image".to_owned()),
        _ => MemoryMode::TextToImage,
    };
    let mut context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        weights_free_tier(spec),
        gen_core::MemoryBehaviorRoute {
            mode,
            // The IP reference image / the rendered pose skeleton.
            reference_count: 1,
            use_pid: contract_carries_pid(contract),
            has_phases: false,
            overlay: provider_overlay_identity(contract),
        },
    )?;
    context.evidence_revision = REQUEST_EVIDENCE_REVISION.to_owned();
    Ok(vec![gen_core::MemoryBehaviorFixture::new(context)])
}

pub fn composed_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope>>> {
    let provider_id = composed_route_id(contract)?;
    validate_bespoke_context_with_tier(
        contract,
        context,
        weights_free_tier(spec),
        true,
        contract_carries_pid(contract),
    )?;
    Ok(Some(Box::new(route_request_scope(
        provider_id,
        candle_gen::candle_core::Device::Cpu,
        contract,
        context,
    )?)))
}

pub(crate) const IP_MEMORY_REGISTRATION: gen_core::MemoryRegistration =
    gen_core::MemoryRegistration {
        provider_id: IP_PROVIDER_ID,
        contract: ip_composed_contract,
        safety_check: composed_safety_check,
    };

pub(crate) const CONTROL_MEMORY_REGISTRATION: gen_core::MemoryRegistration =
    gen_core::MemoryRegistration {
        provider_id: CONTROL_PROVIDER_ID,
        contract: control_composed_contract,
        safety_check: composed_safety_check,
    };

pub(crate) const IP_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: IP_PROVIDER_ID,
        valid_fixtures: composed_valid_fixtures,
        begin_request: composed_begin_request,
    };

pub(crate) const CONTROL_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: CONTROL_PROVIDER_ID,
        valid_fixtures: composed_valid_fixtures,
        begin_request: composed_begin_request,
    };

#[derive(Clone, Debug, PartialEq, Eq)]
struct RequestBinding {
    address: usize,
    geometry: MemoryGeometry,
    memory: Option<GenerationMemory>,
    use_pid: bool,
    prompt: String,
    negative_prompt: Option<String>,
    seed: Option<u64>,
    steps: Option<u32>,
    guidance_bits: Option<u32>,
    sampler: Option<String>,
    scheduler: Option<String>,
    strength_bits: Option<u32>,
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
            prompt: request.prompt.clone(),
            negative_prompt: request.negative_prompt.clone(),
            seed: request.seed,
            steps: request.steps,
            guidance_bits: request.guidance.map(f32::to_bits),
            sampler: request.sampler.clone(),
            scheduler: request.scheduler.clone(),
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
    next_token: u64,
    approved_context: Option<MemoryRunContext>,
    active: Option<ActiveAdmission>,
}

#[derive(Clone)]
pub(crate) struct AdmissionRegistry {
    inner: Arc<Mutex<AdmissionState>>,
}

impl AdmissionRegistry {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(AdmissionState::default())),
        }
    }

    pub(crate) fn approve(&self, context: &MemoryRunContext) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.inner);
        if state.active.is_some() {
            return Err(gen_core::Error::Unsupported(
                "kolors: another memory request is active".into(),
            ));
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
        if state.active.is_some() {
            return Err(gen_core::Error::Unsupported(
                "kolors: another memory request scope is active".into(),
            ));
        }
        let approved = state.approved_context.take().ok_or_else(|| {
            gen_core::Error::Unsupported("kolors: memory request skipped safety approval".into())
        })?;
        if approved != *context {
            return Err(gen_core::Error::Unsupported(
                "kolors: memory context changed after safety approval".into(),
            ));
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
            gen_core::Error::Unsupported("kolors: memory request scope is inactive".into())
        })?;
        let binding = RequestBinding::from_request(request);
        if active.token != token
            || active.binding.is_some()
            || active.consumed
            || binding.geometry != active.context.geometry
            || binding.memory != active.expected_memory
            || binding.use_pid != active.context.use_pid
        {
            return Err(gen_core::Error::Unsupported(
                "kolors: stale or changed memory request".into(),
            ));
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
                Err(gen_core::Error::Unsupported(
                    "kolors: constrained request has no active admission".into(),
                ))
            } else {
                Ok(())
            };
        };
        if active.binding.as_ref() != Some(&RequestBinding::from_request(request))
            || active.consumed
        {
            return Err(gen_core::Error::Unsupported(
                "kolors: request changed or admission was already consumed".into(),
            ));
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
            Err(gen_core::Error::Unsupported(
                "kolors: stale memory token cannot finish".into(),
            ))
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

pub(crate) struct KolorsMemoryScope {
    core: candle_gen::request_scope::CandleRequestScopeCore,
    admission: AdmissionRegistry,
    token: u64,
    finished: bool,
}

impl KolorsMemoryScope {
    pub(crate) fn new(
        device: candle_gen::candle_core::Device,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
        admission: AdmissionRegistry,
    ) -> gen_core::Result<Self> {
        let token = admission.begin(contract, context)?;
        let config = candle_gen::request_scope::CandleRequestScopeConfig::new(
            crate::MODEL_ID,
            device,
            context.geometry,
            contract.generation_memory(&context.selection),
            context.use_pid,
            1,
            |_pid, _edge, _overlap| {
                Err(gen_core::Error::Unsupported(
                    "kolors: bounded decode is Missing".into(),
                ))
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

impl MemoryRequestScope for KolorsMemoryScope {
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

impl Drop for KolorsMemoryScope {
    fn drop(&mut self) {
        if !self.finished {
            self.admission.abandon(self.token);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{
        AdapterSpec, MemoryBudget, MemoryCacheState, MemoryGeometry, MemoryOptimizationAuthority,
        MemorySelection,
    };

    fn packed(bits: u8, group: u32) -> String {
        format!(r#"{{"quantization":{{"bits":{bits},"group_size":{group}}}}}"#)
    }

    #[test]
    fn only_resident_and_staged_are_advertised() {
        let contract = provider_contract();
        for capability in contract.strategies {
            assert_eq!(
                capability.support,
                if matches!(
                    capability.strategy,
                    MemoryStrategy::Resident | MemoryStrategy::StagedResidency
                ) {
                    MemoryStrategySupport::Implemented
                } else {
                    MemoryStrategySupport::Missing
                }
            );
        }
        assert!(contract.lifecycle.synchronized_phase_release);
        assert!(!contract.lifecycle.decode_tiling);
        assert!(!contract.lifecycle.attention_chunking);
        assert!(!contract.lifecycle.transformer_window_materialization);
    }

    #[test]
    fn physical_tier_requires_a_matching_packed_pair() {
        let temp = tempfile::tempdir().unwrap();
        for component in ["text_encoder", "unet"] {
            let dir = temp.path().join(component);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("config.json"), packed(4, 64)).unwrap();
            write_packed_tensor(&dir.join("model.safetensors"), 4);
        }
        assert_eq!(physical_tier(temp.path()).unwrap().quant, Some(Quant::Q4));

        std::fs::write(temp.path().join("text_encoder/config.json"), packed(8, 64)).unwrap();
        assert!(physical_tier(temp.path()).is_err());
        std::fs::write(temp.path().join("text_encoder/config.json"), packed(4, 32)).unwrap();
        assert!(physical_tier(temp.path()).is_err());
    }

    #[test]
    fn dense_tier_is_not_relabelled_as_requested_quantization() {
        let temp = tempfile::tempdir().unwrap();
        for component in ["text_encoder", "unet"] {
            let dir = temp.path().join(component);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("config.json"), "{}").unwrap();
            write_tensor(&dir.join("model.safetensors"), "BF16", &[2, 3]);
        }
        let tier = physical_tier(temp.path()).unwrap();
        assert_eq!(tier.precision, Precision::Bf16);
        assert_eq!(tier.quant, None);
    }

    fn write_tensor(path: &Path, dtype: &str, shape: &[u64]) {
        write_named_tensors(path, &[("weight", dtype, shape)]);
    }

    fn write_named_tensors(path: &Path, tensors: &[(&str, &str, &[u64])]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut offset = 0_u64;
        let mut header = serde_json::Map::new();
        for (name, dtype, shape) in tensors {
            let width = match *dtype {
                "BF16" | "F16" => 2,
                _ => 4,
            };
            let bytes = shape.iter().product::<u64>() * width;
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes]
                }),
            );
            offset += bytes;
        }
        let header = serde_json::Value::Object(header).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + offset as usize, 0);
        std::fs::write(path, bytes).unwrap();
    }

    fn write_packed_tensor(path: &Path, bits: u8) {
        let packed_columns = usize::from(bits) * 2;
        write_named_tensors(
            path,
            &[
                ("linear.weight", "U32", &[2, packed_columns as u64]),
                ("linear.scales", "BF16", &[2, 1]),
                ("linear.biases", "BF16", &[2, 1]),
            ],
        );
    }

    fn canonical_base(temp: &tempfile::TempDir) -> PathBuf {
        let root = temp
            .path()
            .join("models--SceneWorks--kolors-mlx")
            .join("snapshots")
            .join(KOLORS_REVISION)
            .join("q4");
        for component in ["text_encoder", "unet", "vae"] {
            if component == "vae" {
                write_tensor(
                    &root.join(component).join("model.safetensors"),
                    "BF16",
                    &[2, 3],
                );
            } else {
                write_packed_tensor(&root.join(component).join("model.safetensors"), 4);
            }
            std::fs::write(
                root.join(component).join("config.json"),
                if component == "vae" {
                    "{}"
                } else {
                    r#"{"quantization":{"bits":4,"group_size":64}}"#
                },
            )
            .unwrap();
        }
        std::fs::create_dir_all(root.join("tokenizer")).unwrap();
        std::fs::write(root.join("tokenizer/tokenizer.json"), "{}").unwrap();
        root
    }

    fn canonical_pid(temp: &tempfile::TempDir) -> (PidWeights, PathBuf, PathBuf) {
        let student = temp
            .path()
            .join("models--SceneWorks--pid-sdxl")
            .join("snapshots")
            .join(PID_SDXL_REVISION)
            .join("pid_sdxl_2kto4k.safetensors");
        write_tensor(&student, "BF16", &[2, 2]);
        let gemma = temp
            .path()
            .join("models--SceneWorks--gemma-2-2b-it")
            .join("snapshots")
            .join(PID_GEMMA_REVISION);
        write_tensor(&gemma.join("model.safetensors"), "BF16", &[2, 2]);
        (
            PidWeights {
                checkpoint: WeightsSource::File(student.clone()),
                gemma: WeightsSource::Dir(gemma.clone()),
            },
            student,
            gemma,
        )
    }

    fn context(provider: &str, strategy: MemoryStrategy) -> MemoryRunContext {
        MemoryRunContext {
            selection: MemorySelection {
                strategy,
                parameters: Default::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: &[],
                },
            },
            optimization_authority: if strategy == MemoryStrategy::Resident {
                MemoryOptimizationAuthority::Resident
            } else {
                MemoryOptimizationAuthority::Estimated
            },
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: calibration_fingerprint(provider),
            mode: if provider == IP_PROVIDER_ID {
                MemoryMode::Other("character_image".into())
            } else {
                MemoryMode::TextToImage
            },
            load_shape: gen_core::LoadShape::EagerMaterialization,
            // Both bespoke routes consume exactly one conditioning image (the IP reference / the
            // rendered pose skeleton); only the registered base T2I route has none.
            has_reference: provider != crate::MODEL_ID,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: u32::from(provider != crate::MODEL_ID),
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: u64::MAX,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: REQUEST_EVIDENCE_REVISION.to_owned(),
        }
    }

    #[test]
    fn bespoke_receipts_bind_runtime_bytes_route_and_ordered_adapters() {
        let temp = tempfile::tempdir().unwrap();
        let base = canonical_base(&temp);
        let ip = temp.path().join(IP_CACHE_NAMESPACE).join(IP_REVISION);
        write_tensor(
            &ip.join("ip_adapter_plus_general.safetensors"),
            "BF16",
            &[3, 5],
        );
        write_tensor(&ip.join("image_encoder/model.safetensors"), "BF16", &[7, 2]);
        let adapter_a = temp.path().join("a.safetensors");
        let adapter_b = temp.path().join("b.safetensors");
        write_tensor(&adapter_a, "BF16", &[2, 2]);
        write_tensor(&adapter_b, "BF16", &[3, 2]);
        let adapters = vec![
            AdapterSpec::new(adapter_a.clone(), 0.25, AdapterKind::Lora),
            AdapterSpec::new(adapter_b.clone(), 0.75, AdapterKind::Lokr),
        ];
        let paths = crate::IpAdapterKolorsPaths {
            kolors_base: base.clone(),
            ip_adapter: ip,
            adapters: adapters.clone(),
        };
        let contract = provider_contract_for_ip(&paths).unwrap();
        assert_eq!(contract.provider_id, "candle_kolors_ipadapter");
        assert_eq!(contract.asset_facts.conditioning_bytes, 80);
        assert_eq!(contract.asset_facts.transformer_bytes, 80);
        assert_eq!(contract.asset_facts.decoder_bytes, 24);
        assert!(contract
            .resident_components()
            .iter()
            .any(|component| component.id.starts_with(IP_RECEIPT_PREFIX)));
        let mut staged = context("candle_kolors_ipadapter", MemoryStrategy::StagedResidency);
        staged.overlay = provider_overlay_identity(&contract);
        validate_bespoke_context(&contract, &base, &staged, true, false).unwrap();

        let mut reversed = paths.clone();
        reversed.adapters.reverse();
        let reversed = provider_contract_for_ip(&reversed).unwrap();
        let adapter_id = |contract: &MemoryProviderContract| {
            contract
                .resident_components()
                .iter()
                .find(|component| component.id.starts_with(ADAPTER_RECEIPT_PREFIX))
                .unwrap()
                .id
                .clone()
        };
        assert_ne!(adapter_id(&contract), adapter_id(&reversed));
    }

    #[test]
    fn bespoke_context_refuses_crossed_provider_mode_reference_and_revision() {
        let temp = tempfile::tempdir().unwrap();
        let base = canonical_base(&temp);
        let control = temp
            .path()
            .join(CONTROL_CACHE_NAMESPACE)
            .join(CONTROL_REVISION)
            .join("control.safetensors");
        write_tensor(&control, "BF16", &[4, 4]);
        let paths = crate::KolorsControlPaths {
            kolors_base: base.clone(),
            controlnet: control,
            adapters: vec![],
        };
        let contract = provider_contract_for_control(&paths, None).unwrap();
        let mut crossed = context(CONTROL_PROVIDER_ID, MemoryStrategy::StagedResidency);
        crossed.overlay = provider_overlay_identity(&contract);
        // The IP route's mode with the control route's contract is still crossed.
        crossed.mode = MemoryMode::Other("character_image".into());
        assert!(validate_bespoke_context(&contract, &base, &crossed, true, false).is_ok());
        crossed.mode = MemoryMode::Edit;
        assert!(validate_bespoke_context(&contract, &base, &crossed, true, false).is_err());
        crossed.mode = MemoryMode::TextToImage;
        crossed.evidence_revision = "crossed-family".into();
        assert!(validate_bespoke_context(&contract, &base, &crossed, true, false).is_err());
    }

    /// sc-20762 review (MAJOR 9): the strict-pose route consumes a rendered OpenPose skeleton on
    /// every render, so its memory key must state that conditioning image. It previously demanded
    /// `reference_count == 0` — a key that claimed a conditioning-free render.
    #[test]
    fn control_route_requires_the_pose_conditioning_reference_it_consumes() {
        let temp = tempfile::tempdir().unwrap();
        let base = canonical_base(&temp);
        let control = temp
            .path()
            .join(CONTROL_CACHE_NAMESPACE)
            .join(CONTROL_REVISION)
            .join("control.safetensors");
        write_tensor(&control, "BF16", &[4, 4]);
        let paths = crate::KolorsControlPaths {
            kolors_base: base.clone(),
            controlnet: control,
            adapters: vec![],
        };
        let contract = provider_contract_for_control(&paths, None).unwrap();

        let mut admitted = context(CONTROL_PROVIDER_ID, MemoryStrategy::StagedResidency);
        admitted.overlay = provider_overlay_identity(&contract);
        assert!(admitted.has_reference);
        assert_eq!(admitted.geometry.reference_count, 1);
        validate_bespoke_context(&contract, &base, &admitted, true, false).unwrap();

        // The exact pre-fix shape: no typed reference at all.
        let mut unconditioned = admitted.clone();
        unconditioned.has_reference = false;
        unconditioned.geometry.reference_count = 0;
        assert!(validate_bespoke_context(&contract, &base, &unconditioned, false, false).is_err());

        // A count that does not match the single skeleton is refused in the other direction too.
        let mut doubled = admitted;
        doubled.geometry.reference_count = 2;
        assert!(validate_bespoke_context(&contract, &base, &doubled, true, false).is_err());
    }

    /// sc-20762 review (MINOR): the three Kolors contracts share one physical base but are
    /// independent evidence identities; one shared fingerprint let any of them satisfy another's
    /// calibration check.
    #[test]
    fn each_kolors_route_mints_its_own_calibration_fingerprint() {
        let temp = tempfile::tempdir().unwrap();
        let base = canonical_base(&temp);
        let ip = temp.path().join(IP_CACHE_NAMESPACE).join(IP_REVISION);
        write_tensor(
            &ip.join("ip_adapter_plus_general.safetensors"),
            "BF16",
            &[3, 5],
        );
        write_tensor(&ip.join("image_encoder/model.safetensors"), "BF16", &[7, 2]);
        let control = temp
            .path()
            .join(CONTROL_CACHE_NAMESPACE)
            .join(CONTROL_REVISION)
            .join("control.safetensors");
        write_tensor(&control, "BF16", &[4, 4]);

        let sealed = [
            provider_contract_for_load(&base, &[], None).unwrap(),
            provider_contract_for_ip(&crate::IpAdapterKolorsPaths {
                kolors_base: base.clone(),
                ip_adapter: ip,
                adapters: vec![],
            })
            .unwrap(),
            provider_contract_for_control(
                &crate::KolorsControlPaths {
                    kolors_base: base,
                    controlnet: control,
                    adapters: vec![],
                },
                None,
            )
            .unwrap(),
        ];
        let fingerprints = sealed
            .iter()
            .map(|contract| {
                contract
                    .calibration
                    .as_ref()
                    .expect("every Kolors contract carries a calibration identity")
                    .fingerprint
                    .clone()
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(
            fingerprints.len(),
            3,
            "the three sealed Kolors contracts must not share a calibration fingerprint: {fingerprints:?}"
        );
    }

    /// sc-20762 review (MAJOR 14): the bespoke routes ran with no request scope at all — nothing
    /// bound the admitted geometry to the request rendered, and nothing rejected a second finish.
    #[test]
    fn bespoke_request_scope_binds_geometry_and_owns_the_run_lifecycle() {
        use candle_gen::candle_core::Device;
        use gen_core::MemoryRequestScope as _;

        let temp = tempfile::tempdir().unwrap();
        let base = canonical_base(&temp);
        let control = temp
            .path()
            .join(CONTROL_CACHE_NAMESPACE)
            .join(CONTROL_REVISION)
            .join("control.safetensors");
        write_tensor(&control, "BF16", &[4, 4]);
        let contract = provider_contract_for_control(
            &crate::KolorsControlPaths {
                kolors_base: base,
                controlnet: control,
                adapters: vec![],
            },
            None,
        )
        .unwrap();
        let mut admitted = context(CONTROL_PROVIDER_ID, MemoryStrategy::StagedResidency);
        admitted.overlay = provider_overlay_identity(&contract);

        // A request that does not equal the admitted geometry never opens a scope.
        assert!(bespoke_request_scope(
            CONTROL_PROVIDER_ID,
            Device::Cpu,
            &contract,
            &admitted,
            512,
            1024,
            false
        )
        .is_err());
        // Nor does a crossed provider identity.
        assert!(bespoke_request_scope(
            IP_PROVIDER_ID,
            Device::Cpu,
            &contract,
            &admitted,
            1024,
            1024,
            false
        )
        .is_err());
        // Nor a PiD request against a non-PiD admission.
        assert!(bespoke_request_scope(
            CONTROL_PROVIDER_ID,
            Device::Cpu,
            &contract,
            &admitted,
            1024,
            1024,
            true
        )
        .is_err());

        let mut scope = bespoke_request_scope(
            CONTROL_PROVIDER_ID,
            Device::Cpu,
            &contract,
            &admitted,
            1024,
            1024,
            false,
        )
        .unwrap();
        // Bounded decode is Missing on this route; no tile was ever admitted.
        assert!(scope.configure_decode(512, 64, admitted.geometry).is_err());
        // A canceled run still gets the contract's cleanup, exactly once.
        scope.finish(MemoryRunOutcome::Canceled).unwrap();
        assert!(scope.finish(MemoryRunOutcome::Canceled).is_err());
        assert!(scope.enter_phase(MemoryPhase::Denoise).is_err());
    }

    #[test]
    fn mutable_or_arbitrary_base_source_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        let arbitrary = temp.path().join("kolors-main");
        for component in ["text_encoder", "unet", "vae"] {
            write_tensor(
                &arbitrary.join(component).join("model.safetensors"),
                "BF16",
                &[1],
            );
        }
        std::fs::create_dir_all(arbitrary.join("tokenizer")).unwrap();
        std::fs::write(arbitrary.join("tokenizer/tokenizer.json"), "{}").unwrap();
        assert!(provider_contract_for_load(&arbitrary, &[], None).is_err());
    }

    #[test]
    fn arbitrary_or_crossed_bespoke_component_source_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        let base = canonical_base(&temp);
        let arbitrary = temp.path().join("control-main.safetensors");
        write_tensor(&arbitrary, "BF16", &[4, 4]);
        let arbitrary_paths = crate::KolorsControlPaths {
            kolors_base: base.clone(),
            controlnet: arbitrary,
            adapters: vec![],
        };
        assert!(provider_contract_for_control(&arbitrary_paths, None).is_err());

        let crossed = temp
            .path()
            .join(IP_CACHE_NAMESPACE)
            .join(IP_REVISION)
            .join("control.safetensors");
        write_tensor(&crossed, "BF16", &[4, 4]);
        let crossed_paths = crate::KolorsControlPaths {
            kolors_base: base,
            controlnet: crossed,
            adapters: vec![],
        };
        assert!(provider_contract_for_control(&crossed_paths, None).is_err());
    }

    #[test]
    fn retained_seals_reject_mutation_of_every_lazy_route_component() {
        let temp = tempfile::tempdir().unwrap();
        let base = canonical_base(&temp);
        let adapter = temp.path().join("adapter.safetensors");
        write_tensor(&adapter, "BF16", &[2, 2]);
        let (pid, student, gemma) = canonical_pid(&temp);
        let adapters = vec![AdapterSpec::new(adapter.clone(), 0.5, AdapterKind::Lora)];

        for path in [
            base.join("tokenizer/tokenizer.json"),
            base.join("text_encoder/model.safetensors"),
            base.join("unet/model.safetensors"),
            base.join("vae/model.safetensors"),
            adapter.clone(),
            student.clone(),
            gemma.join("model.safetensors"),
        ] {
            let seal = KolorsLoadSeal::capture_load(&base, &adapters, Some(&pid)).unwrap();
            let original = std::fs::read(&path).unwrap();
            std::fs::write(&path, b"mutated-after-admission").unwrap();
            assert!(seal.ensure_unchanged().is_err(), "path={}", path.display());
            std::fs::write(path, original).unwrap();
        }

        let ip = temp.path().join(IP_CACHE_NAMESPACE).join(IP_REVISION);
        let ip_file = ip.join("ip_adapter_plus_general.safetensors");
        write_tensor(&ip_file, "BF16", &[3, 5]);
        let ip_encoder = ip.join("image_encoder/model.safetensors");
        write_tensor(&ip_encoder, "BF16", &[7, 2]);
        let ip_paths = crate::IpAdapterKolorsPaths {
            kolors_base: base.clone(),
            ip_adapter: ip,
            adapters: adapters.clone(),
        };
        for path in [ip_file, ip_encoder] {
            let seal = KolorsLoadSeal::capture_ip(&ip_paths).unwrap();
            let original = std::fs::read(&path).unwrap();
            std::fs::write(&path, b"mutated-ip").unwrap();
            assert!(seal.ensure_unchanged().is_err());
            std::fs::write(path, original).unwrap();
        }

        let control = temp
            .path()
            .join(CONTROL_CACHE_NAMESPACE)
            .join(CONTROL_REVISION)
            .join("control.safetensors");
        write_tensor(&control, "BF16", &[4, 4]);
        let control_paths = crate::KolorsControlPaths {
            kolors_base: base,
            controlnet: control.clone(),
            adapters,
        };
        let seal = KolorsLoadSeal::capture_control(&control_paths, Some(&pid)).unwrap();
        std::fs::write(&control, b"mutated-control").unwrap();
        assert!(seal.ensure_unchanged().is_err());
    }

    #[test]
    fn forged_config_and_tensor_headers_are_refused_for_all_compositions() {
        let temp = tempfile::tempdir().unwrap();
        let base = canonical_base(&temp);

        write_tensor(
            &base.join("text_encoder/model.safetensors"),
            "BF16",
            &[2, 3],
        );
        assert!(physical_tier(&base).is_err());
        write_packed_tensor(&base.join("text_encoder/model.safetensors"), 4);
        write_named_tensors(
            &base.join("unet/model.safetensors"),
            &[
                ("linear.weight", "U32", &[2, 8]),
                ("linear.scales", "BF16", &[2, 1]),
            ],
        );
        assert!(physical_tier(&base).is_err());
        write_packed_tensor(&base.join("unet/model.safetensors"), 4);

        write_tensor(&base.join("vae/model.safetensors"), "U32", &[2, 3]);
        assert!(KolorsLoadSeal::capture_load(&base, &[], None).is_err());
        write_tensor(&base.join("vae/model.safetensors"), "BF16", &[2, 3]);

        let adapter = temp.path().join("forged-adapter.safetensors");
        write_tensor(&adapter, "U32", &[2, 2]);
        let adapters = vec![AdapterSpec::new(adapter, 1.0, AdapterKind::Lora)];
        assert!(KolorsLoadSeal::capture_load(&base, &adapters, None).is_err());

        let ip = temp.path().join(IP_CACHE_NAMESPACE).join(IP_REVISION);
        write_tensor(
            &ip.join("ip_adapter_plus_general.safetensors"),
            "U32",
            &[3, 5],
        );
        let ip_paths = crate::IpAdapterKolorsPaths {
            kolors_base: base.clone(),
            ip_adapter: ip,
            adapters: vec![],
        };
        assert!(KolorsLoadSeal::capture_ip(&ip_paths).is_err());

        let control = temp
            .path()
            .join(CONTROL_CACHE_NAMESPACE)
            .join(CONTROL_REVISION)
            .join("control.safetensors");
        write_tensor(&control, "U32", &[4, 4]);
        let control_paths = crate::KolorsControlPaths {
            kolors_base: base.clone(),
            controlnet: control,
            adapters: vec![],
        };
        assert!(KolorsLoadSeal::capture_control(&control_paths, None).is_err());

        let (pid, student, _) = canonical_pid(&temp);
        write_tensor(&student, "U32", &[2, 2]);
        assert!(KolorsLoadSeal::capture_load(&base, &[], Some(&pid)).is_err());

        let arbitrary_student = temp.path().join("pid_sdxl_2kto4k.safetensors");
        write_tensor(&arbitrary_student, "BF16", &[2, 2]);
        let arbitrary_gemma = temp.path().join("gemma-main");
        write_tensor(&arbitrary_gemma.join("model.safetensors"), "BF16", &[2, 2]);
        let arbitrary_pid = PidWeights {
            checkpoint: WeightsSource::File(arbitrary_student),
            gemma: WeightsSource::Dir(arbitrary_gemma),
        };
        assert!(KolorsLoadSeal::capture_load(&base, &[], Some(&arbitrary_pid)).is_err());
    }

    #[test]
    fn registered_context_binds_physical_tier_adapters_and_pid() {
        let temp = tempfile::tempdir().unwrap();
        let base = canonical_base(&temp);
        let adapter = temp.path().join("adapter.safetensors");
        write_tensor(&adapter, "BF16", &[2, 2]);
        let (pid, _, _) = canonical_pid(&temp);
        let seal = KolorsLoadSeal::capture_load(
            &base,
            &[AdapterSpec::new(adapter, 0.75, AdapterKind::Lora)],
            Some(&pid),
        )
        .unwrap();
        let mut admitted = context("kolors", MemoryStrategy::StagedResidency);
        admitted.use_pid = true;
        admitted.overlay = provider_overlay_identity(seal.contract());
        validate_context(seal.contract(), &admitted, seal.tier()).unwrap();

        let mut crossed = admitted.clone();
        crossed.overlay = None;
        assert!(validate_context(seal.contract(), &crossed, seal.tier()).is_err());
        crossed = admitted.clone();
        crossed.use_pid = false;
        assert!(validate_context(seal.contract(), &crossed, seal.tier()).is_err());
        crossed = admitted;
        crossed.selection.tier.quant = Some(Quant::Q8);
        assert!(validate_context(seal.contract(), &crossed, seal.tier()).is_err());
    }

    #[test]
    fn registered_admission_is_once_only_request_exact_and_cleans_up() {
        let contract = provider_contract();
        let mut run = context("kolors", MemoryStrategy::StagedResidency);
        run.selection.tier.quant = None;
        run.overlay = None;
        let admission = AdmissionRegistry::new();
        admission.approve(&run).unwrap();
        let token = admission.begin(&contract, &run).unwrap();
        assert!(admission.approve(&run).is_err());
        let mut request = GenerationRequest {
            prompt: "sealed prompt".into(),
            width: run.geometry.width,
            height: run.geometry.height,
            count: run.geometry.batch,
            memory: contract.generation_memory(&run.selection),
            ..Default::default()
        };
        admission.configure(token, &request).unwrap();
        request.prompt.push_str(" crossed");
        assert!(admission.consume_for_generate(&request).is_err());
        admission.abandon(token);

        admission.approve(&run).unwrap();
        let token = admission.begin(&contract, &run).unwrap();
        request.prompt = "sealed prompt".into();
        admission.configure(token, &request).unwrap();
        admission.consume_for_generate(&request).unwrap();
        assert!(admission.consume_for_generate(&request).is_err());
        admission.finish(token).unwrap();

        admission.approve(&run).unwrap();
        let token = admission.begin(&contract, &run).unwrap();
        admission.abandon(token);
        admission.approve(&run).unwrap();
    }
}
