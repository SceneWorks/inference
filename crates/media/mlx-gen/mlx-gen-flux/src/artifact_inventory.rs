//! Exact file-identity and content pins for FLUX.1 deferred-materialization eligibility.
//!
//! A streamable snapshot has one visible `model.safetensors` in each required component directory.
//! Each file is pinned by the shared [`ArtifactSeal`] (mutation-sensitive entry/target identity and
//! a once-per-identity SHA-256), and header-validated against the selected dense/Q4/Q8 tier. The
//! composite digest covers the four model/config pairs plus the T5 `tokenizer_2/tokenizer.json`
//! consumed by prompt conditioning; it is evidence input only and never grants production
//! calibration by itself.

use mlx_gen::gen_core::weightsmeta::{
    safetensors_path_tensor_headers, Dtype, SafetensorsTensorHeader,
};
use mlx_gen::gen_core::{ArtifactSeal, Error as CoreError, Result as CoreResult};
use mlx_gen::{LoadSpec, OffloadPolicy, Precision, Quant, WeightsSource};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

pub(crate) const COMPONENTS: [&str; 4] = ["text_encoder", "text_encoder_2", "transformer", "vae"];

/// One verified snapshot file: the shared [`ArtifactSeal`] (entry, parent chain, resolution and
/// target identity plus the once-per-identity, single-flight content digest) and that digest in
/// hex.
#[derive(Clone, Debug)]
pub(crate) struct PinnedArtifact {
    seal: ArtifactSeal,
    digest: String,
}

impl PinnedArtifact {
    fn verify_file(path: &Path) -> CoreResult<Self> {
        let seal = ArtifactSeal::pin(path)?;
        let digest = seal.content_sha256_hex();
        Ok(Self { seal, digest })
    }

    pub(crate) fn canonical_path(&self) -> &Path {
        self.seal.canonical_target_path()
    }

    /// Snapshot entry consumed by format-dispatching runtime loaders.
    ///
    /// Hugging Face snapshot entries normally retain the semantic extension while their canonical
    /// blob targets do not. MLX selects the safetensors loader from that extension, so runtime
    /// opens must use this already-pinned entry rather than the extensionless canonical blob. The
    /// entry, its symlink target, and the resolved canonical file are all checked by
    /// [`ensure_unchanged`](Self::ensure_unchanged) around every load/materialization boundary.
    pub(crate) fn loader_path(&self) -> &Path {
        self.seal.loader_path()
    }

    pub(crate) fn digest(&self) -> &str {
        &self.digest
    }

    pub(crate) fn ensure_unchanged(&self) -> CoreResult<()> {
        self.seal.ensure_unchanged()
    }
}

#[derive(Clone, Debug)]
struct ComponentInventory {
    name: &'static str,
    source_directory: PathBuf,
    visible_safetensors: Vec<OsString>,
    model: PinnedArtifact,
    config: PinnedArtifact,
}

#[derive(Clone, Debug)]
pub(crate) struct PackedArtifactInventory {
    components: [ComponentInventory; 4],
    t5_tokenizer: PinnedArtifact,
    composite_sha256: String,
    /// Tier every pinned component's quantization marker and packed content encode. Equal to the
    /// admitting `LoadSpec::quantize` by construction — `discover_inventory` refuses a marker that
    /// disagrees with the selected tier — but carried on the inventory so the calibration identity
    /// is derived from the admitted artifact, never from the request knob (sc-22726).
    quant: Option<Quant>,
}

impl PackedArtifactInventory {
    /// Artifact tier of the admitted packed inventory, read from its pinned quantization markers.
    pub(crate) fn resolved_quant(&self) -> Option<Quant> {
        self.quant
    }

    pub(crate) fn transformer_source(&self) -> &PinnedArtifact {
        &self.components[2].model
    }

    pub(crate) fn clip_encoder_source(&self) -> &PinnedArtifact {
        &self.components[0].model
    }

    pub(crate) fn t5_encoder_source(&self) -> &PinnedArtifact {
        &self.components[1].model
    }

    pub(crate) fn vae_source(&self) -> &PinnedArtifact {
        &self.components[3].model
    }

    pub(crate) fn t5_tokenizer_source(&self) -> &PinnedArtifact {
        &self.t5_tokenizer
    }

    pub(crate) fn composite_sha256(&self) -> &str {
        &self.composite_sha256
    }

    pub(crate) fn ensure_unchanged(&self) -> CoreResult<()> {
        for component in &self.components {
            let current = visible_safetensors(&component.source_directory)?;
            if current != component.visible_safetensors {
                return Err(CoreError::Unsupported(format!(
                    "flux1: {} component safetensors membership changed after admission",
                    component.name
                )));
            }
            component.model.ensure_unchanged()?;
            component.config.ensure_unchanged()?;
        }
        self.t5_tokenizer.ensure_unchanged()?;
        Ok(())
    }
}

pub(crate) fn structurally_streamable(provider_id: &str, spec: &LoadSpec) -> bool {
    matches!(provider_id, crate::FLUX1_SCHNELL_ID | crate::FLUX1_DEV_ID)
        && matches!(spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(spec.load_shape, mlx_gen::LoadShape::DeferredMaterialization)
        && spec.precision == Precision::Bf16
        && matches!(spec.quantize, None | Some(Quant::Q4 | Quant::Q8))
        && spec.adapters.is_empty()
        && spec.components.is_empty()
        && spec.control.is_none()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.pid.is_none()
        && spec.identity.is_none()
        && spec.text_encoder.is_none()
        && matches!(spec.weights, WeightsSource::Dir(_))
}

fn visible_safetensors(dir: &Path) -> CoreResult<Vec<OsString>> {
    mlx_gen::gen_core::weightsmeta::visible_safetensors(dir).map_err(|error| {
        CoreError::Msg(format!(
            "flux1: read packed component {}: {error}",
            dir.display()
        ))
    })
}

fn single_component_artifact(
    root: &Path,
    component: &'static str,
) -> CoreResult<ComponentInventory> {
    let dir = root.join(component);
    let source_directory = std::path::absolute(&dir)
        .map_err(|error| CoreError::Msg(format!("flux1: resolve {}: {error}", dir.display())))?;
    let visible_safetensors = visible_safetensors(&source_directory)?;
    if visible_safetensors.as_slice() != [OsString::from("model.safetensors")] {
        return Err(CoreError::Unsupported(format!(
            "flux1: {component} must contain exactly one model.safetensors"
        )));
    }
    Ok(ComponentInventory {
        name: component,
        model: PinnedArtifact::verify_file(&source_directory.join("model.safetensors"))?,
        config: PinnedArtifact::verify_file(&source_directory.join("config.json"))?,
        source_directory,
        visible_safetensors,
    })
}

fn pinned_quant_marker(component: &ComponentInventory) -> CoreResult<Option<i32>> {
    component.config.ensure_unchanged()?;
    let marker = quant_marker_at(component.name, component.config.canonical_path())?;
    component.config.ensure_unchanged()?;
    Ok(marker)
}

/// Packed quantization marker (`{"quantization":{"bits":4|8,"group_size":64}}`) of one component
/// `config.json`, or `None` for a dense component. Shared by the pinned inventory and the
/// header-only tier resolver.
fn quant_marker_at(component: &str, config: &Path) -> CoreResult<Option<i32>> {
    let bytes = mlx_gen::gen_core::__model_input_reader!(
std::fs::read(config),
"crates/media/mlx-gen/mlx-gen-flux/src/artifact_inventory.rs::quant_marker_at::std::fs::read@16685",
ArtifactInput,
config,
|__model_input_path| std::fs::read(__model_input_path)
)
    .map_err(|error| CoreError::Msg(format!("flux1: read {component} config: {error}")))?;
    let json: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| CoreError::Msg(format!("flux1: parse {component} config: {error}")))?;
    let Some(marker) = json.get("quantization") else {
        return Ok(None);
    };
    let bits = marker.get("bits").and_then(serde_json::Value::as_i64);
    let group_size = marker.get("group_size").and_then(serde_json::Value::as_i64);
    match (bits, group_size) {
        (Some(bits @ (4 | 8)), Some(64)) => Ok(Some(bits as i32)),
        _ => Err(CoreError::Unsupported(format!(
            "flux1: {component} has an invalid packed quantization marker"
        ))),
    }
}

fn quant_for_marker(component: &str, marker: Option<i32>) -> CoreResult<Option<Quant>> {
    match marker {
        None => Ok(None),
        Some(4) => Ok(Some(Quant::Q4)),
        Some(8) => Ok(Some(Quant::Q8)),
        Some(bits) => Err(CoreError::Unsupported(format!(
            "flux1: {component} carries an unshipped Q{bits} marker"
        ))),
    }
}

/// Artifact tier a FLUX.1 snapshot directory actually encodes, proven from every component's
/// quantization marker and safetensors headers without pinning or hashing a byte of weight data
/// (sc-22726). `Ok(None)` is a dense snapshot; `Ok(Some(_))` a snapshot whose four components all
/// carry the same Q4/Q8 marker with matching packed content. `Err` when a component is unreadable,
/// its content does not encode its marker, or the components disagree — such a snapshot has no
/// measurable tier.
///
/// The pinned inventory is only discovered on the streamable route; the worker's
/// `Resident + EagerMaterialization` base load never has one, yet it must publish the same
/// per-tier identity as the deferred evidence shape. This resolver is what ties that identity to
/// the loaded artifact: a dense snapshot loaded with `LoadSpec::quantize = Some(_)` is a runtime
/// requantization the loader supports but no anchor measured, and it resolves here to `None`,
/// never to the requested tier.
pub(crate) fn resolved_artifact_tier(spec: &LoadSpec) -> CoreResult<Option<Quant>> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(CoreError::Unsupported(
            "flux1: artifact tier resolution needs a snapshot directory".to_owned(),
        ));
    };
    let mut resolved: Option<Option<Quant>> = None;
    for component in COMPONENTS {
        let dir = root.join(component);
        let marker = quant_marker_at(component, &dir.join("config.json"))?;
        let mut tensors = BTreeMap::new();
        for file in visible_safetensors(&dir)? {
            let headers = safetensors_path_tensor_headers(dir.join(&file)).map_err(|error| {
                CoreError::Unsupported(format!(
                    "flux1: {component} {} is not a valid header-readable artifact: {error}",
                    Path::new(&file).display()
                ))
            })?;
            for tensor in headers {
                validate_tensor_extent(component, &tensor)?;
                tensors.insert(tensor.name.clone(), tensor);
            }
        }
        if tensors.is_empty() {
            return Err(CoreError::Unsupported(format!(
                "flux1: {component} contains no tensors"
            )));
        }
        validate_component_tensors(component, &tensors, marker)?;
        let quant = quant_for_marker(component, marker)?;
        match resolved {
            None => resolved = Some(quant),
            Some(previous) if previous == quant => {}
            Some(previous) => {
                return Err(CoreError::Unsupported(format!(
                    "flux1: {component} tier {quant:?} disagrees with {previous:?} in an earlier component"
                )));
            }
        }
    }
    Ok(resolved.flatten())
}

fn validate_tensor_extent(component: &str, tensor: &SafetensorsTensorHeader) -> CoreResult<()> {
    let elements = tensor.shape.iter().try_fold(1_u64, |product, &dimension| {
        product.checked_mul(dimension as u64)
    });
    let expected = elements.and_then(|elements| elements.checked_mul(tensor.dtype.size() as u64));
    if expected != Some(tensor.data_bytes) {
        return Err(CoreError::Unsupported(format!(
            "flux1: {component} tensor {} has shape/dtype bytes inconsistent with its data range",
            tensor.name
        )));
    }
    Ok(())
}

fn is_float(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::BF16 | Dtype::F16 | Dtype::F32)
}

fn is_dense_quant_candidate(tensor: &SafetensorsTensorHeader) -> bool {
    tensor.shape.len() == 2 && tensor.shape[1] >= 64 && tensor.shape[1].is_multiple_of(64)
}

fn is_vae_quant_target(base: &str) -> bool {
    [".to_q", ".to_k", ".to_v", ".to_out.0"]
        .iter()
        .any(|suffix| base.ends_with(suffix))
}

fn validate_packed_triple(
    component: &str,
    base: &str,
    weight: &SafetensorsTensorHeader,
    scales: &SafetensorsTensorHeader,
    biases: &SafetensorsTensorHeader,
    expected_bits: i32,
) -> CoreResult<()> {
    if weight.dtype != Dtype::U32 || weight.shape.len() != 2 {
        return Err(CoreError::Unsupported(format!(
            "flux1: {component} packed {base}.weight must be rank-2 U32 codes"
        )));
    }
    if !is_float(scales.dtype)
        || !is_float(biases.dtype)
        || scales.shape.len() != 2
        || biases.shape.len() != 2
        || scales.dtype != biases.dtype
        || scales.shape != biases.shape
        || scales.shape[0] != weight.shape[0]
    {
        return Err(CoreError::Unsupported(format!(
            "flux1: {component} packed {base} scales/biases must be matching rank-2 floating grids with the code-row count"
        )));
    }
    let input = scales.shape[1].checked_mul(64).ok_or_else(|| {
        CoreError::Unsupported(format!("flux1: {component} packed {base} input overflow"))
    })?;
    if input < 64 || input % 64 != 0 {
        return Err(CoreError::Unsupported(format!(
            "flux1: {component} packed {base} has an invalid group-64 logical input"
        )));
    }
    let packed_width = weight.shape[1].checked_mul(32).ok_or_else(|| {
        CoreError::Unsupported(format!("flux1: {component} packed {base} width overflow"))
    })?;
    if packed_width % input != 0 || packed_width / input != expected_bits as usize {
        return Err(CoreError::Unsupported(format!(
            "flux1: {component} packed {base} content does not encode Q{expected_bits} at group 64"
        )));
    }
    Ok(())
}

fn validate_component_content(
    component: &ComponentInventory,
    quant: Option<Quant>,
) -> CoreResult<()> {
    component.model.ensure_unchanged()?;
    let headers =
        safetensors_path_tensor_headers(component.model.canonical_path()).map_err(|error| {
            CoreError::Unsupported(format!(
                "flux1: {} model.safetensors is not a valid header-readable artifact: {error}",
                component.name
            ))
        })?;
    component.model.ensure_unchanged()?;
    if headers.is_empty() {
        return Err(CoreError::Unsupported(format!(
            "flux1: {} model.safetensors contains no tensors",
            component.name
        )));
    }
    let mut tensors = BTreeMap::new();
    for tensor in headers {
        validate_tensor_extent(component.name, &tensor)?;
        tensors.insert(tensor.name.clone(), tensor);
    }

    let marker = pinned_quant_marker(component)?;
    let expected = quant.map(Quant::bits);
    if marker != expected {
        return Err(CoreError::Unsupported(format!(
            "flux1: {} pinned quantization marker does not match the selected tier",
            component.name
        )));
    }
    validate_component_tensors(component.name, &tensors, expected)
}

/// Content half of the tier check: the tensor headers of one component must encode exactly the
/// dense (`None`) or packed Q4/Q8 (`Some(bits)`) tier its marker declares. Shared by the pinned
/// inventory and [`resolved_artifact_tier`].
fn validate_component_tensors(
    component: &str,
    tensors: &BTreeMap<String, SafetensorsTensorHeader>,
    expected: Option<i32>,
) -> CoreResult<()> {
    if expected.is_none() {
        if tensors.values().any(|tensor| {
            tensor.name.ends_with(".scales")
                || tensor.name.ends_with(".biases")
                || (tensor.name.ends_with(".weight") && tensor.dtype == Dtype::U32)
        }) {
            return Err(CoreError::Unsupported(format!(
                "flux1: dense {} contains packed quantization leaves",
                component
            )));
        }
        return Ok(());
    }

    for tensor in tensors.values() {
        for suffix in [".scales", ".biases"] {
            if let Some(base) = tensor.name.strip_suffix(suffix) {
                if !tensors.contains_key(&format!("{base}.weight")) {
                    return Err(CoreError::Unsupported(format!(
                        "flux1: {} has orphan packed leaf {}",
                        component, tensor.name
                    )));
                }
            }
        }
    }

    let expected_bits = expected.expect("quantized branch checked above");
    let vae = component == "vae";
    let mut packed_count = 0_usize;
    for weight in tensors
        .values()
        .filter(|tensor| tensor.name.ends_with(".weight"))
    {
        let base = weight
            .name
            .strip_suffix(".weight")
            .expect("filtered suffix");
        let scales = tensors.get(&format!("{base}.scales"));
        let biases = tensors.get(&format!("{base}.biases"));
        match (scales, biases) {
            (Some(scales), Some(biases)) => {
                if vae && !is_vae_quant_target(base) {
                    return Err(CoreError::Unsupported(format!(
                        "flux1: VAE has unexpected packed non-attention target {base}"
                    )));
                }
                validate_packed_triple(component, base, weight, scales, biases, expected_bits)?;
                packed_count += 1;
            }
            (Some(_), None) | (None, Some(_)) => {
                return Err(CoreError::Unsupported(format!(
                    "flux1: {} has a partial packed triple for {base}",
                    component
                )));
            }
            (None, None) => {
                let required =
                    is_dense_quant_candidate(weight) && (!vae || is_vae_quant_target(base));
                if weight.dtype == Dtype::U32 || required {
                    return Err(CoreError::Unsupported(format!(
                        "flux1: {} quantized tier has an unpacked or incomplete eligible weight {base}",
                        component
                    )));
                }
            }
        }
    }
    if packed_count == 0 {
        return Err(CoreError::Unsupported(format!(
            "flux1: {} quantization marker has no packed content",
            component
        )));
    }
    Ok(())
}

fn composite_digest(components: &[ComponentInventory; 4], t5_tokenizer: &PinnedArtifact) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"flux1-packed-component-inventory-v2\0");
    for component in components {
        hasher.update((component.name.len() as u64).to_le_bytes());
        hasher.update(component.name.as_bytes());
        hasher.update(component.model.digest().as_bytes());
        hasher.update(component.config.digest().as_bytes());
    }
    let tokenizer_name = "tokenizer_2/tokenizer.json";
    hasher.update((tokenizer_name.len() as u64).to_le_bytes());
    hasher.update(tokenizer_name.as_bytes());
    hasher.update(t5_tokenizer.digest().as_bytes());
    format!("{:x}", hasher.finalize())
}

fn discover_inventory(spec: &LoadSpec) -> CoreResult<PackedArtifactInventory> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(CoreError::Unsupported(
            "flux1: deferred materialization needs a snapshot directory".to_owned(),
        ));
    };
    let t5_tokenizer = PinnedArtifact::verify_file(&root.join("tokenizer_2/tokenizer.json"))?;
    let components = [
        single_component_artifact(root, COMPONENTS[0])?,
        single_component_artifact(root, COMPONENTS[1])?,
        single_component_artifact(root, COMPONENTS[2])?,
        single_component_artifact(root, COMPONENTS[3])?,
    ];
    for component in &components {
        validate_component_content(component, spec.quantize)?;
    }
    let inventory = PackedArtifactInventory {
        components,
        t5_tokenizer,
        composite_sha256: String::new(),
        // Proven equal to every component's pinned marker by `validate_component_content` above.
        quant: spec.quantize,
    };
    inventory.ensure_unchanged()?;
    let composite_sha256 = composite_digest(&inventory.components, &inventory.t5_tokenizer);
    inventory.ensure_unchanged()?;
    Ok(PackedArtifactInventory {
        composite_sha256,
        ..inventory
    })
}

pub(crate) fn verified_stream_inventory(
    provider_id: &str,
    spec: &LoadSpec,
) -> Option<PackedArtifactInventory> {
    structurally_streamable(provider_id, spec)
        .then(|| discover_inventory(spec).ok())
        .flatten()
}

#[cfg(test)]
#[derive(Clone)]
struct TestTensor {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
}

#[cfg(test)]
fn write_test_safetensors(path: &Path, tensors: &[TestTensor]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0_usize;
    for tensor in tensors {
        let element_size = match tensor.dtype {
            "BF16" | "F16" => 2,
            "F32" | "U32" => 4,
            other => panic!("unsupported test dtype {other}"),
        };
        let bytes = tensor.shape.iter().product::<usize>() * element_size;
        header.insert(
            tensor.name.clone(),
            serde_json::json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, offset + bytes],
            }),
        );
        offset += bytes;
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = Vec::with_capacity(8 + header.len() + offset);
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(&header);
    file.resize(8 + header.len() + offset, 0);
    std::fs::write(path, file).unwrap();
}

#[cfg(test)]
fn packed_test_tensors(component: &str, bits: i32) -> Vec<TestTensor> {
    let base = if component == "vae" {
        "probe.attn.to_q"
    } else {
        "probe"
    };
    vec![
        TestTensor {
            name: format!("{base}.weight"),
            dtype: "U32",
            shape: vec![2, (bits as usize) * 2],
        },
        TestTensor {
            name: format!("{base}.scales"),
            dtype: "BF16",
            shape: vec![2, 1],
        },
        TestTensor {
            name: format!("{base}.biases"),
            dtype: "BF16",
            shape: vec![2, 1],
        },
    ]
}

/// A second, header-valid shard for one component of a [`write_test_snapshot`] snapshot: it
/// encodes the same tier as the snapshot (so the header-only tier resolver still proves it) while
/// making the component sharded, which the pinned single-file inventory refuses.
#[cfg(test)]
pub(crate) fn write_test_shard(root: &Path, component: &str, quant: Option<Quant>) {
    let tensors = match quant {
        Some(quant) => packed_test_tensors(component, quant.bits())
            .into_iter()
            .map(|tensor| TestTensor {
                name: format!("shard2.{}", tensor.name),
                ..tensor
            })
            .collect(),
        None => vec![TestTensor {
            name: "shard2.probe.weight".to_owned(),
            dtype: "BF16",
            shape: vec![2, 64],
        }],
    };
    write_test_safetensors(
        &root
            .join(component)
            .join("model-00002-of-00002.safetensors"),
        &tensors,
    );
}

#[cfg(test)]
pub(crate) fn write_test_snapshot(root: &Path, quant: Option<Quant>) {
    for component in COMPONENTS {
        let dir = root.join(component);
        std::fs::create_dir_all(&dir).unwrap();
        let tensors = match quant {
            Some(quant) => packed_test_tensors(component, quant.bits()),
            None => vec![TestTensor {
                name: "probe.weight".to_owned(),
                dtype: "BF16",
                shape: vec![2, 64],
            }],
        };
        write_test_safetensors(&dir.join("model.safetensors"), &tensors);
        let config = match quant {
            Some(quant) => format!(
                r#"{{"quantization":{{"bits":{},"group_size":64}}}}"#,
                quant.bits()
            ),
            None => "{}".to_owned(),
        };
        std::fs::write(dir.join("config.json"), config).unwrap();
    }
    std::fs::create_dir_all(root.join("tokenizer_2")).unwrap();
    std::fs::write(
        root.join("tokenizer_2/tokenizer.json"),
        br#"{"version":"1.0","model":{"type":"Unigram"}}"#,
    )
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_root(tmp: &tempfile::TempDir, label: &str) -> PathBuf {
        tmp.path().join(format!(
            "flux-artifact-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    fn write_snapshot(root: &Path, quant: Option<Quant>) {
        write_test_snapshot(root, quant);
    }

    fn eligible_spec(root: &Path, quant: Option<Quant>) -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_owned()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization);
        spec.quantize = quant;
        spec
    }

    #[test]
    fn dense_q4_and_q8_exact_single_file_inventories_are_loadable() {
        let tmp = tempfile::tempdir().unwrap();
        for (label, quant) in [
            ("dense-inventory", None),
            ("q4-inventory", Some(Quant::Q4)),
            ("q8-inventory", Some(Quant::Q8)),
        ] {
            let root = unique_root(&tmp, label);
            write_snapshot(&root, quant);
            let inventory =
                verified_stream_inventory(crate::FLUX1_DEV_ID, &eligible_spec(&root, quant))
                    .expect("exact four-file inventory");
            assert_eq!(inventory.composite_sha256().len(), 64);
            assert_eq!(
                inventory.transformer_source().canonical_path(),
                std::fs::canonicalize(root.join("transformer/model.safetensors")).unwrap()
            );
            inventory.ensure_unchanged().unwrap();
            std::fs::remove_dir_all(root).ok();
        }
    }

    #[test]
    fn pinned_loader_path_preserves_safetensors_extension_for_extensionless_hf_blob() {
        let tmp = tempfile::tempdir().unwrap();
        use std::os::unix::fs::symlink;

        let root = unique_root(&tmp, "extensionless-hf-blob");
        write_snapshot(&root, Some(Quant::Q4));
        let snapshot_path = root.join("transformer/model.safetensors");
        let blob_path = root.join("blobs/0123456789abcdef");
        std::fs::create_dir_all(blob_path.parent().unwrap()).unwrap();
        std::fs::rename(&snapshot_path, &blob_path).unwrap();
        symlink(Path::new("../blobs/0123456789abcdef"), &snapshot_path).unwrap();

        let inventory =
            verified_stream_inventory(crate::FLUX1_DEV_ID, &eligible_spec(&root, Some(Quant::Q4)))
                .expect("extensionless HF blob remains an exact streamable artifact");
        let source = inventory.transformer_source();

        assert_eq!(
            source.loader_path(),
            std::path::absolute(&snapshot_path).unwrap()
        );
        assert_eq!(
            source
                .loader_path()
                .extension()
                .and_then(|value| value.to_str()),
            Some("safetensors"),
            "MLX dispatches the file format from the pinned snapshot entry extension"
        );
        assert_eq!(
            source.canonical_path(),
            std::fs::canonicalize(&blob_path).unwrap()
        );
        assert!(source.canonical_path().extension().is_none());
        source.ensure_unchanged().unwrap();

        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn sharded_or_mismatched_component_inventory_is_not_streamable() {
        let tmp = tempfile::tempdir().unwrap();
        let root = unique_root(&tmp, "sharded-inventory");
        write_snapshot(&root, Some(Quant::Q4));
        let spec = eligible_spec(&root, Some(Quant::Q4));
        std::fs::write(
            root.join("transformer/diffusion_pytorch_model-00001-of-00002.safetensors"),
            [9_u8; 8],
        )
        .unwrap();
        assert!(verified_stream_inventory(crate::FLUX1_DEV_ID, &spec).is_none());
        std::fs::remove_file(
            root.join("transformer/diffusion_pytorch_model-00001-of-00002.safetensors"),
        )
        .unwrap();
        std::fs::write(
            root.join("vae/config.json"),
            r#"{"quantization":{"bits":8,"group_size":64}}"#,
        )
        .unwrap();
        assert!(verified_stream_inventory(crate::FLUX1_DEV_ID, &spec).is_none());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn composite_identity_and_pin_change_on_same_size_component_mutation() {
        let tmp = tempfile::tempdir().unwrap();
        let root = unique_root(&tmp, "inventory-mutation");
        write_snapshot(&root, None);
        let spec = eligible_spec(&root, None);
        let first = verified_stream_inventory(crate::FLUX1_SCHNELL_ID, &spec).unwrap();
        let first_composite = first.composite_sha256().to_owned();
        let vae_path = root.join("vae/model.safetensors");
        let mut vae = std::fs::read(&vae_path).unwrap();
        *vae.last_mut().unwrap() ^= 1;
        std::fs::write(&vae_path, vae).unwrap();
        assert!(first.ensure_unchanged().is_err());
        let second = verified_stream_inventory(crate::FLUX1_SCHNELL_ID, &spec).unwrap();
        assert_ne!(first_composite, second.composite_sha256());
        let second_composite = second.composite_sha256().to_owned();
        std::fs::write(
            root.join("tokenizer_2/tokenizer.json"),
            br#"{"version":"2.0","model":{"type":"Unigram"}}"#,
        )
        .unwrap();
        assert!(second.ensure_unchanged().is_err());
        let third = verified_stream_inventory(crate::FLUX1_SCHNELL_ID, &spec).unwrap();
        assert_ne!(second_composite, third.composite_sha256());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn missing_t5_tokenizer_fails_stream_eligibility() {
        let tmp = tempfile::tempdir().unwrap();
        let root = unique_root(&tmp, "missing-tokenizer");
        write_snapshot(&root, None);
        let spec = eligible_spec(&root, None);
        std::fs::remove_file(root.join("tokenizer_2/tokenizer.json")).unwrap();
        assert!(verified_stream_inventory(crate::FLUX1_DEV_ID, &spec).is_none());
        std::fs::remove_dir_all(root).ok();
    }

    fn assert_component_content_rejected(
        tmp: &tempfile::TempDir,
        label: &str,
        quant: Option<Quant>,
        component: &str,
        tensors: &[TestTensor],
    ) {
        let root = unique_root(tmp, label);
        write_snapshot(&root, quant);
        write_test_safetensors(&root.join(component).join("model.safetensors"), tensors);
        assert!(
            verified_stream_inventory(crate::FLUX1_DEV_ID, &eligible_spec(&root, quant)).is_none(),
            "{component} mutation unexpectedly passed admission"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn corrupt_empty_and_arbitrary_model_files_fail_in_every_component() {
        let tmp = tempfile::tempdir().unwrap();
        for component in COMPONENTS {
            let root = unique_root(&tmp, &format!("empty-{component}"));
            write_snapshot(&root, None);
            std::fs::write(root.join(component).join("model.safetensors"), []).unwrap();
            assert!(
                verified_stream_inventory(crate::FLUX1_DEV_ID, &eligible_spec(&root, None))
                    .is_none(),
                "empty {component} unexpectedly passed"
            );
            std::fs::remove_dir_all(&root).ok();

            let root = unique_root(&tmp, &format!("arbitrary-{component}"));
            write_snapshot(&root, None);
            std::fs::write(
                root.join(component).join("model.safetensors"),
                [component.len() as u8; 64],
            )
            .unwrap();
            assert!(
                verified_stream_inventory(crate::FLUX1_DEV_ID, &eligible_spec(&root, None))
                    .is_none(),
                "arbitrary {component} unexpectedly passed"
            );
            std::fs::remove_dir_all(root).ok();
        }

        assert_component_content_rejected(&tmp, "zero-tensor-header", None, "transformer", &[]);
    }

    #[test]
    fn dense_tier_rejects_packed_leaves() {
        let tmp = tempfile::tempdir().unwrap();
        assert_component_content_rejected(
            &tmp,
            "dense-with-packed",
            None,
            "transformer",
            &packed_test_tensors("transformer", 4),
        );
    }

    #[test]
    fn quantized_tier_rejects_missing_triple_for_eligible_dense_weight() {
        let tmp = tempfile::tempdir().unwrap();
        assert_component_content_rejected(
            &tmp,
            "quant-with-dense",
            Some(Quant::Q4),
            "text_encoder",
            &[TestTensor {
                name: "probe.weight".to_owned(),
                dtype: "BF16",
                shape: vec![2, 64],
            }],
        );
    }

    #[test]
    fn q4_marker_rejects_q8_content() {
        let tmp = tempfile::tempdir().unwrap();
        assert_component_content_rejected(
            &tmp,
            "q4-marker-q8-content",
            Some(Quant::Q4),
            "transformer",
            &packed_test_tensors("transformer", 8),
        );
    }

    #[test]
    fn quantized_tier_rejects_partial_and_orphan_triples() {
        let tmp = tempfile::tempdir().unwrap();
        let full = packed_test_tensors("text_encoder_2", 4);
        assert_component_content_rejected(
            &tmp,
            "partial-triple",
            Some(Quant::Q4),
            "text_encoder_2",
            &full[..2],
        );
        assert_component_content_rejected(
            &tmp,
            "orphan-triple",
            Some(Quant::Q4),
            "text_encoder_2",
            &[TestTensor {
                name: "orphan.scales".to_owned(),
                dtype: "BF16",
                shape: vec![2, 1],
            }],
        );
    }

    #[test]
    fn quantized_tier_rejects_wrong_code_dtype_and_packed_shapes() {
        let tmp = tempfile::tempdir().unwrap();
        let mut wrong_dtype = packed_test_tensors("transformer", 4);
        wrong_dtype[0].dtype = "BF16";
        assert_component_content_rejected(
            &tmp,
            "wrong-code-dtype",
            Some(Quant::Q4),
            "transformer",
            &wrong_dtype,
        );

        let mut wrong_rows = packed_test_tensors("transformer", 4);
        wrong_rows[2].shape = vec![3, 1];
        assert_component_content_rejected(
            &tmp,
            "wrong-packed-shape",
            Some(Quant::Q4),
            "transformer",
            &wrong_rows,
        );

        let mut wrong_scale_dtype = packed_test_tensors("transformer", 4);
        wrong_scale_dtype[1].dtype = "U32";
        assert_component_content_rejected(
            &tmp,
            "wrong-scale-dtype",
            Some(Quant::Q4),
            "transformer",
            &wrong_scale_dtype,
        );
    }

    #[test]
    fn quantized_tier_rejects_non_group_64_marker() {
        let tmp = tempfile::tempdir().unwrap();
        let root = unique_root(&tmp, "wrong-group-size");
        write_snapshot(&root, Some(Quant::Q4));
        std::fs::write(
            root.join("vae/config.json"),
            r#"{"quantization":{"bits":4,"group_size":32}}"#,
        )
        .unwrap();
        assert!(verified_stream_inventory(
            crate::FLUX1_DEV_ID,
            &eligible_spec(&root, Some(Quant::Q4)),
        )
        .is_none());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn vae_rejects_packed_non_attention_target() {
        let tmp = tempfile::tempdir().unwrap();
        assert_component_content_rejected(
            &tmp,
            "vae-packed-non-attention",
            Some(Quant::Q4),
            "vae",
            &packed_test_tensors("transformer", 4),
        );
    }

    #[test]
    fn vae_requires_only_eligible_attention_targets_to_be_packed() {
        let tmp = tempfile::tempdir().unwrap();
        assert_component_content_rejected(
            &tmp,
            "vae-missing-attention-triple",
            Some(Quant::Q4),
            "vae",
            &[TestTensor {
                name: "probe.attn.to_q.weight".to_owned(),
                dtype: "BF16",
                shape: vec![2, 64],
            }],
        );

        let root = unique_root(&tmp, "vae-target-only-positive");
        write_snapshot(&root, Some(Quant::Q4));
        let mut tensors = packed_test_tensors("vae", 4);
        tensors.push(TestTensor {
            name: "probe.conv.weight".to_owned(),
            dtype: "BF16",
            shape: vec![2, 64],
        });
        write_test_safetensors(&root.join("vae/model.safetensors"), &tensors);
        assert!(
            verified_stream_inventory(crate::FLUX1_DEV_ID, &eligible_spec(&root, Some(Quant::Q4)),)
                .is_some(),
            "eligible dense non-attention VAE weights must remain allowed"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn admitted_snapshot_rejects_new_visible_safetensors_member() {
        let tmp = tempfile::tempdir().unwrap();
        let root = unique_root(&tmp, "post-admission-extra-file");
        write_snapshot(&root, None);
        let inventory =
            verified_stream_inventory(crate::FLUX1_DEV_ID, &eligible_spec(&root, None)).unwrap();
        std::fs::write(root.join("transformer/extra.safetensors"), [9_u8; 8]).unwrap();
        assert!(inventory.ensure_unchanged().is_err());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn admitted_snapshot_rejects_config_replacement() {
        let tmp = tempfile::tempdir().unwrap();
        let root = unique_root(&tmp, "post-admission-config-replacement");
        write_snapshot(&root, Some(Quant::Q4));
        let inventory =
            verified_stream_inventory(crate::FLUX1_DEV_ID, &eligible_spec(&root, Some(Quant::Q4)))
                .unwrap();
        let replacement = root.join("transformer/config-replacement.json");
        std::fs::write(
            &replacement,
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        std::fs::rename(&replacement, root.join("transformer/config.json")).unwrap();
        assert!(inventory.ensure_unchanged().is_err());
        std::fs::remove_dir_all(root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn admitted_snapshot_rejects_model_symlink_retarget() {
        let tmp = tempfile::tempdir().unwrap();
        use std::os::unix::fs::symlink;

        let root = unique_root(&tmp, "post-admission-symlink-retarget");
        write_snapshot(&root, None);
        let component = root.join("transformer");
        let objects = root.join("objects");
        std::fs::create_dir_all(&objects).unwrap();
        let original = objects.join("original.safetensors");
        let replacement = objects.join("replacement.safetensors");
        std::fs::rename(component.join("model.safetensors"), &original).unwrap();
        std::fs::write(&replacement, [7_u8; 64]).unwrap();
        symlink(
            "../objects/original.safetensors",
            component.join("model.safetensors"),
        )
        .unwrap();
        let inventory =
            verified_stream_inventory(crate::FLUX1_DEV_ID, &eligible_spec(&root, None)).unwrap();
        std::fs::remove_file(component.join("model.safetensors")).unwrap();
        symlink(
            "../objects/replacement.safetensors",
            component.join("model.safetensors"),
        )
        .unwrap();
        assert!(inventory.ensure_unchanged().is_err());
        std::fs::remove_dir_all(root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn admitted_snapshot_rejects_parent_snapshot_symlink_retarget() {
        let tmp = tempfile::tempdir().unwrap();
        use std::os::unix::fs::symlink;

        let base = unique_root(&tmp, "post-admission-parent-symlink-retarget");
        let first = base.join("snapshot-a");
        let second = base.join("snapshot-b");
        write_snapshot(&first, None);
        write_snapshot(&second, None);
        let root = base.join("current");
        symlink("snapshot-a", &root).unwrap();
        let inventory =
            verified_stream_inventory(crate::FLUX1_DEV_ID, &eligible_spec(&root, None)).unwrap();
        std::fs::remove_file(&root).unwrap();
        symlink("snapshot-b", &root).unwrap();
        assert!(inventory.ensure_unchanged().is_err());
        std::fs::remove_dir_all(base).ok();
    }

    #[test]
    fn structural_gate_rejects_every_overlay_tier_policy_and_shape_mutation() {
        let tmp = tempfile::tempdir().unwrap();
        let root = unique_root(&tmp, "structural-mutations");
        write_snapshot(&root, Some(Quant::Q4));
        let base = eligible_spec(&root, Some(Quant::Q4));
        assert!(structurally_streamable(crate::FLUX1_DEV_ID, &base));

        let mut mutations = Vec::new();
        let mut spec = base.clone();
        spec.offload_policy = OffloadPolicy::Resident;
        mutations.push(spec);
        let mut spec = base.clone();
        spec.load_shape = mlx_gen::LoadShape::EagerMaterialization;
        mutations.push(spec);
        let mut spec = base.clone();
        spec.precision = Precision::Fp32;
        mutations.push(spec);
        let mut spec = base.clone();
        spec.quantize = Some(Quant::Nvfp4);
        mutations.push(spec);
        let mut spec = base.clone();
        spec.adapters.push(mlx_gen::AdapterSpec::new(
            "/adapter.safetensors".into(),
            1.0,
            mlx_gen::AdapterKind::Lora,
        ));
        mutations.push(spec);
        let mut spec = base.clone();
        spec.control = Some(WeightsSource::File("/control.safetensors".into()));
        mutations.push(spec);
        let mut spec = base.clone();
        spec.extra_controls
            .push(WeightsSource::File("/control-2.safetensors".into()));
        mutations.push(spec);
        let mut spec = base.clone();
        spec.ip_adapter = Some(WeightsSource::Dir("/ip".into()));
        mutations.push(spec);
        mutations.push(base.clone().with_pid(
            WeightsSource::File("/pid.safetensors".into()),
            WeightsSource::Dir("/gemma".into()),
        ));
        let mut spec = base.clone();
        spec.identity = Some(Default::default());
        mutations.push(spec);
        let mut spec = base.clone();
        spec.text_encoder = Some(WeightsSource::Dir("/external-text".into()));
        mutations.push(spec);

        for spec in mutations {
            assert!(!structurally_streamable(crate::FLUX1_DEV_ID, &spec));
            assert!(verified_stream_inventory(crate::FLUX1_DEV_ID, &spec).is_none());
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn same_path_same_size_content_mutation_gets_a_new_digest() {
        let tmp = tempfile::tempdir().unwrap();
        let root = unique_root(&tmp, "content-mutation");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("model.safetensors");
        std::fs::write(&path, [3_u8; 32]).unwrap();
        let first = PinnedArtifact::verify_file(&path).unwrap();
        std::fs::write(&path, [4_u8; 32]).unwrap();
        assert!(first.ensure_unchanged().is_err());
        let second = PinnedArtifact::verify_file(&path).unwrap();
        assert_ne!(first.digest(), second.digest());
        std::fs::remove_dir_all(root).ok();
    }
}
