//! The backend-neutral half of every native FLUX3 conversion (sc-26241).
//!
//! Input is ALWAYS the original dense package (`quantization: null`): never a publisher FP8R/NVFP4
//! artifact and never another backend's native package (no cross-format repack). The output is a
//! NEW owned directory published atomically with a full-checksum receipt:
//!
//! * `model.safetensors` — streamed one tensor at a time in a header fixed up front, so the same
//!   source and algorithm produce byte-identical files; each source FP8 target projection is
//!   replaced by the backend's packed tensors, every other tensor is copied byte-for-byte;
//! * `config.json` — the source config with only its `quantization` marker replaced;
//! * `LICENSE.md` / `NOTICE.md` — copied from the source;
//! * the unchanged text encoder and autoencoder are EXTERNAL components: full-hash recorded and
//!   verified again where a load reads them, never copied or re-encoded.
//!
//! The only backend-specific step is `pack`: one source target in, its exact packed tensors out.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::artifacts::NativeDitStorage;
use super::conversion::{
    hash_file, hash_source_file, publish_conversion, ComponentInventory, ComponentPlacement,
    ComponentRole, ConversionReceipt, OutputFile,
};
use super::quantization::DitFormat;
use super::safetensors_stream::{
    PlannedTensor, SafetensorsSource, SafetensorsStreamWriter, SourceTensor,
};
use super::{CONFIG_FILE, DIT_WEIGHTS_FILE};
use gen_core::{CancelFlag, Error, Result};

fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 native conversion: {detail}"))
}

/// The pinned dense source package being converted.
#[derive(Clone, Debug)]
pub struct ConversionSource {
    /// The ORIGINAL dense package root (`config.json`, `model.safetensors`, `ae.safetensors`,
    /// `text_encoder/`, `LICENSE.md`, `NOTICE.md`).
    pub root: PathBuf,
    /// Its frozen Hub identity (the receipt validates it against the frozen FLUX3 checkpoints).
    pub repository: String,
    pub revision: String,
}

/// Packed output tensors of one source target, by output name.
pub type Packed = BTreeMap<String, Vec<u8>>;

/// Every regular file under `root/rel` (sorted, `/`-separated, relative to `root`), hashed.
fn hash_tree(root: &Path, rel: &str) -> Result<BTreeMap<String, OutputFile>> {
    let mut out = BTreeMap::new();
    let start = root.join(rel);
    if std::fs::symlink_metadata(&start)?.file_type().is_symlink() {
        return Err(invalid(format!("{} links a directory", start.display())));
    }
    let mut stack = vec![root.join(rel)];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let ty = entry.file_type()?;
            let path = entry.path();
            // A snapshot links each FILE to its blob; a linked directory would let the tree escape
            // the package, so only file links are followed (and hashed by their real bytes).
            let is_dir = if ty.is_symlink() {
                if std::fs::metadata(&path)?.is_dir() {
                    return Err(invalid(format!("{} links a directory", path.display())));
                }
                false
            } else {
                ty.is_dir()
            };
            if is_dir {
                stack.push(path);
            } else {
                let name = path
                    .strip_prefix(root)
                    .map_err(|_| invalid("path escaped the source root"))?
                    .to_str()
                    .ok_or_else(|| invalid("non-UTF8 source path"))?
                    .replace('\\', "/");
                out.insert(name, hash_source_file(&path)?);
            }
        }
    }
    Ok(out)
}

fn shas(files: &BTreeMap<String, OutputFile>) -> BTreeMap<String, String> {
    files
        .iter()
        .map(|(p, f)| (p.clone(), f.sha256.clone()))
        .collect()
}

/// Convert `source` to `storage` at a NEW `output`. `pack(target, tensor, raw)` returns the exact
/// packed outputs for one `<stem>.weight` target (its codes and, for affine storage, the
/// `<stem>.scales`/`<stem>.biases` companions). Cancellation between tensors publishes nothing.
pub fn convert_native(
    source: &ConversionSource,
    storage: NativeDitStorage,
    output: &Path,
    cancel: &CancelFlag,
    mut pack: impl FnMut(&str, &SourceTensor, &[u8]) -> Result<Packed>,
) -> Result<ConversionReceipt> {
    let (algorithm, bits, group_size) = storage.receipt_algorithm()?;
    if matches!(storage, NativeDitStorage::Publisher(_)) {
        return Err(invalid(
            "publisher FP8R/NVFP4 are imported as shipped, never produced by repacking",
        ));
    }
    // Resolve the root once (it may itself be a linked snapshot directory); every input below is
    // read and hashed through it, by real bytes.
    let root_buf = std::fs::canonicalize(&source.root)
        .map_err(|e| invalid(format!("resolve {}: {e}", source.root.display())))?;
    let root = &root_buf;
    let config = super::resolve_checkpoint(root)?;
    config.validate_supported()?;
    super::conversion::validate_source_mode(
        &source.repository,
        &source.revision,
        config.dit.flow_map,
    )?;
    let dense = NativeDitStorage::from_package(root)?;
    if dense != NativeDitStorage::Publisher(DitFormat::Dense) {
        return Err(invalid(
            "the source must be the ORIGINAL dense checkpoint (quantization: null); packed \
             publisher or native artifacts are never re-quantized",
        ));
    }
    let dit_source = root.join(DIT_WEIGHTS_FILE);
    dense.validate_headers(&config.dit, &super::checkpoint_tensor_headers(root)?)?;
    if !dit_source.is_file() {
        return Err(invalid("native conversion requires the frozen single-file dense source model.safetensors; sharded metadata is not a source-byte identity"));
    }
    let reader = SafetensorsSource::open(&dit_source)?;
    let expected = storage.expected_tensors(&config.dit)?;
    let targets = storage.packed_targets(&config.dit)?;
    // Which source target owns each packed output (codes and companions).
    let mut owner = BTreeMap::new();
    for target in &targets {
        let stem = target
            .strip_suffix(".weight")
            .ok_or_else(|| invalid("packed target is not a weight"))?;
        for name in expected.keys() {
            if name == target
                || name == &format!("{stem}.scales")
                || name == &format!("{stem}.biases")
            {
                owner.insert(name.clone(), target.clone());
            }
        }
        reader.get(target)?;
    }
    let mut planned = Vec::with_capacity(expected.len());
    for (name, spec) in &expected {
        planned.push(if owner.contains_key(name) {
            PlannedTensor {
                name: name.clone(),
                dtype: format!("{:?}", spec.dtype),
                shape: spec.shape.clone(),
            }
        } else {
            let tensor = reader.get(name)?;
            PlannedTensor {
                name: name.clone(),
                dtype: tensor.dtype.clone(),
                shape: tensor.shape.clone(),
            }
        });
    }
    let copied: Vec<&String> = expected
        .keys()
        .filter(|n| !owner.contains_key(*n))
        .collect();
    if reader.tensors().len() != targets.len() + copied.len() {
        return Err(invalid(
            "source DiT inventory does not match the target layout",
        ));
    }
    let key = storage.key()?;
    let mut config_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join(CONFIG_FILE))?).map_err(invalid)?;
    config_json["quantization"] = storage.marker()?;
    let dit_artifacts = BTreeMap::from([(
        DIT_WEIGHTS_FILE.to_owned(),
        hash_source_file(&dit_source)?.sha256,
    )]);
    let ae_files = BTreeMap::from([(
        config.vae.path.clone(),
        hash_source_file(&root.join(&config.vae.path))?,
    )]);
    let te_files = hash_tree(root, &config.text_encoder.path)?;
    let mut source_artifacts = dit_artifacts.clone();
    source_artifacts.insert(
        CONFIG_FILE.into(),
        hash_source_file(&root.join(CONFIG_FILE))?.sha256,
    );

    let mut receipt = None;
    let status = publish_conversion(output, |stage| {
        let mut writer = SafetensorsStreamWriter::create(
            &stage.join(DIT_WEIGHTS_FILE),
            planned.clone(),
            &BTreeMap::from([
                ("format".to_owned(), "pt".to_owned()),
                ("flux3_native_storage".to_owned(), key.clone()),
            ]),
        )?;
        let mut pending = Packed::new();
        while let Some(name) = writer.next_name().map(str::to_owned) {
            if cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            if let Some(target) = owner.get(&name) {
                if !pending.contains_key(&name) {
                    let tensor = reader.get(target)?;
                    let packed = pack(target, tensor, &reader.read(target)?)?;
                    let want: Vec<&String> = owner
                        .iter()
                        .filter(|(_, t)| *t == target)
                        .map(|(n, _)| n)
                        .collect();
                    if packed.keys().ne(want.iter().copied()) {
                        return Err(invalid(format!(
                            "{target}: pack produced {:?}, the layout needs {want:?}",
                            packed.keys().collect::<Vec<_>>()
                        )));
                    }
                    pending.extend(packed);
                }
                let bytes = pending.remove(&name).expect("packed output present");
                writer.write(&name, &bytes)?;
            } else {
                writer.write(&name, &reader.read(&name)?)?;
            }
        }
        if !pending.is_empty() {
            return Err(invalid("packed outputs left unwritten"));
        }
        writer.finish()?;
        std::fs::write(
            stage.join(CONFIG_FILE),
            serde_json::to_vec_pretty(&config_json).map_err(invalid)?,
        )?;
        let mut package_files = BTreeMap::new();
        for name in ["LICENSE.md", "NOTICE.md"] {
            std::fs::copy(root.join(name), stage.join(name))
                .map_err(|e| invalid(format!("copy {name}: {e}")))?;
            package_files.insert(name.to_owned(), hash_file(&stage.join(name))?);
        }
        package_files.insert(CONFIG_FILE.to_owned(), hash_file(&stage.join(CONFIG_FILE))?);
        let component =
            |role, placement, artifacts, representation: String, files| ComponentInventory {
                role,
                placement,
                source_repository: source.repository.clone(),
                source_revision: source.revision.clone(),
                source_artifacts: artifacts,
                representation,
                files,
                quantized_targets: Vec::new(),
                dense_targets: Vec::new(),
            };
        let mut dit = component(
            ComponentRole::Dit,
            ComponentPlacement::Owned,
            dit_artifacts.clone(),
            format!("{key} packed targets; other tensors as stored; compute per load precision"),
            BTreeMap::from([(
                DIT_WEIGHTS_FILE.to_owned(),
                hash_file(&stage.join(DIT_WEIGHTS_FILE))?,
            )]),
        );
        dit.quantized_targets = targets.iter().cloned().collect();
        dit.dense_targets = copied.iter().map(|n| (*n).clone()).collect();
        let built = ConversionReceipt {
            source_repository: source.repository.clone(),
            source_revision: source.revision.clone(),
            source_artifacts: source_artifacts.clone(),
            algorithm: algorithm.to_owned(),
            algorithm_version: 1,
            bits,
            group_size,
            components: vec![
                dit,
                component(
                    ComponentRole::TextEncoder,
                    ComponentPlacement::External,
                    shas(&te_files),
                    "unchanged source text encoder".into(),
                    te_files.clone(),
                ),
                component(
                    ComponentRole::Autoencoder,
                    ComponentPlacement::External,
                    shas(&ae_files),
                    "unchanged source f32 autoencoder".into(),
                    ae_files.clone(),
                ),
            ],
            package_files,
        };
        receipt = Some(built.clone());
        Ok(built)
    })?;
    if let super::conversion::PublishStatus::ParentSyncFailed(error) = &status {
        eprintln!(
            "flux3 conversion published {} but its parent directory sync failed ({error}); the \n             output is complete and verified, its rename may not survive a power loss",
            output.display()
        );
    }
    receipt.ok_or_else(|| invalid("publication returned no receipt"))
}

/// MLX affine W`bits`A16 at `group`: the backend-neutral pack ([`super::affine`]) for the MLX
/// `quantized_matmul` layout. CPU-only and deterministic, so the conversion is reproducible on any
/// host; only LOADING the result needs Apple Silicon.
pub fn convert_affine(
    source: &ConversionSource,
    bits: u32,
    group: u32,
    output: &Path,
    cancel: &CancelFlag,
) -> Result<ConversionReceipt> {
    let storage = NativeDitStorage::MlxAffine {
        bits,
        group_size: group,
    };
    convert_native(source, storage, output, cancel, |target, tensor, raw| {
        let [n, k] = tensor.shape.as_slice() else {
            return Err(invalid(format!("{target}: not a matrix")));
        };
        let values = super::affine::decode_dense(&tensor.dtype, raw)?;
        let packed = super::affine::quantize(&values, *n, *k, bits, group)?;
        let (codes, scales, biases) = super::affine::to_le_bytes(&packed);
        let stem = target.strip_suffix(".weight").expect("weight target");
        Ok(Packed::from([
            (target.to_owned(), codes),
            (format!("{stem}.scales"), scales),
            (format!("{stem}.biases"), biases),
        ]))
    })
}

#[cfg(test)]
#[path = "native_convert_tests.rs"]
mod tests;
