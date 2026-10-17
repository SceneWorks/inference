//! Immutable, caller-owned FLUX3 conversion outputs and reproducibility receipts.
//!
//! No downloads or managed-cache derivation occur here. A converter fills a NEW
//! sibling directory, and publication follows complete inventory/checksum checks.
//! This module never sweeps a sibling/previous owner's files or replaces an install.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::{Error, Result};

pub const RECEIPT_FILE: &str = "flux3-conversion-receipt.json";
const RECEIPT_VERSION: u64 = 1;
static OUTPUT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 conversion: {detail}"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComponentRole {
    Dit,
    TextEncoder,
    Autoencoder,
    PromptUpsampler,
}
impl ComponentRole {
    pub fn name(self) -> &'static str {
        match self {
            Self::Dit => "dit",
            Self::TextEncoder => "text_encoder",
            Self::Autoencoder => "autoencoder",
            Self::PromptUpsampler => "prompt_upsampler",
        }
    }
    fn parse(name: &str) -> Result<Self> {
        match name {
            "dit" => Ok(Self::Dit),
            "text_encoder" => Ok(Self::TextEncoder),
            "autoencoder" => Ok(Self::Autoencoder),
            "prompt_upsampler" => Ok(Self::PromptUpsampler),
            _ => Err(invalid(format!("unknown component {name}"))),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputFile {
    pub bytes: u64,
    pub sha256: String,
}

/// Where a component's runtime files live. A converted package owns its rewritten DiT;
/// an UNCHANGED encoder/AE/PUP may stay in the original immutable source root instead of being
/// copied. External files are full-hash verified against that root by
/// [`ConversionReceipt::verify_external`]; they are never part of the owned output inventory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentPlacement {
    Owned,
    External,
}
impl ComponentPlacement {
    pub fn name(self) -> &'static str {
        match self {
            Self::Owned => "owned",
            Self::External => "external",
        }
    }
    fn parse(name: &str) -> Result<Self> {
        match name {
            "owned" => Ok(Self::Owned),
            "external" => Ok(Self::External),
            _ => Err(invalid(format!("unknown component placement {name}"))),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentInventory {
    pub role: ComponentRole,
    pub placement: ComponentPlacement,
    pub source_repository: String,
    pub source_revision: String,
    pub source_artifacts: BTreeMap<String, String>,
    /// Source/storage/compute representation is component-specific; no global
    /// "Q4 model" label may imply that AE, encoder or PUP were also quantized.
    pub representation: String,
    pub files: BTreeMap<String, OutputFile>,
    pub quantized_targets: Vec<String>,
    pub dense_targets: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversionReceipt {
    pub source_repository: String,
    pub source_revision: String,
    /// Actual FULL source-file checksums, not header-byte hashes or Hub metadata.
    pub source_artifacts: BTreeMap<String, String>,
    pub algorithm: String,
    pub algorithm_version: u32,
    pub bits: Option<u32>,
    pub group_size: Option<u32>,
    pub components: Vec<ComponentInventory>,
    pub package_files: BTreeMap<String, OutputFile>,
}

fn safe_relative(name: &str) -> bool {
    !name.is_empty()
        && !name.contains('\\')
        && !name.contains(':')
        && Path::new(name)
            .components()
            .all(|p| matches!(p, Component::Normal(_)))
        && !name
            .split('/')
            .any(|p| p.is_empty() || p == "." || p == "..")
}
fn hash_is_valid(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl ConversionReceipt {
    pub fn validate(&self) -> Result<()> {
        let revision = match self.source_repository.as_str() {
            super::DISTILLED_WEIGHTS_REPO => super::DISTILLED_WEIGHTS_REVISION,
            super::BASE_WEIGHTS_REPO => super::BASE_WEIGHTS_REVISION,
            _ => {
                return Err(invalid(
                    "source repository is not a frozen FLUX3 checkpoint",
                ))
            }
        };
        if self.source_revision != revision
            || self.algorithm_version != 1
            || self.source_artifacts.is_empty()
        {
            return Err(invalid("missing or mismatched source/algorithm identity"));
        }
        for (path, hash) in &self.source_artifacts {
            if !safe_relative(path) || !hash_is_valid(hash) {
                return Err(invalid("invalid source path or full-file checksum"));
            }
        }
        match self.algorithm.as_str() {
            "mlx-affine"
                if matches!(self.bits, Some(4 | 8))
                    && matches!(self.group_size, Some(32 | 64 | 128)) => {}
            "ggml-q4_1" if self.bits == Some(4) && self.group_size == Some(32) => {}
            "ggml-q8_0" if self.bits == Some(8) && self.group_size == Some(32) => {}
            "bfl-fp8r" | "bfl-nvfp4-import" if self.bits.is_none() && self.group_size.is_none() => {
            }
            _ => {
                return Err(invalid(
                    "unsupported algorithm/version parameters; no inferred defaults",
                ))
            }
        }
        let mut roles = BTreeSet::new();
        let mut paths = BTreeSet::new();
        for component in &self.components {
            if !roles.insert(component.role)
                || component.representation.trim().is_empty()
                || component.files.is_empty()
            {
                return Err(invalid("duplicate or incomplete component inventory"));
            }
            let (repo, revision) = if component.role == ComponentRole::PromptUpsampler {
                (super::pup::UPSAMPLER_REPO, super::pup::UPSAMPLER_REVISION)
            } else {
                (
                    self.source_repository.as_str(),
                    self.source_revision.as_str(),
                )
            };
            if component.source_repository != repo
                || component.source_revision != revision
                || component.source_artifacts.is_empty()
                || component
                    .source_artifacts
                    .iter()
                    .any(|(p, h)| !safe_relative(p) || !hash_is_valid(h))
            {
                return Err(invalid(
                    "missing or mismatched component source identity/checksum",
                ));
            }
            if component.role == ComponentRole::Dit
                && component.placement != ComponentPlacement::Owned
            {
                return Err(invalid(
                    "the converted DiT must be owned by the conversion output",
                ));
            }
            required_component_files(component)?;
            let mut targets = BTreeSet::new();
            for target in component
                .quantized_targets
                .iter()
                .chain(&component.dense_targets)
            {
                if target.is_empty() || !targets.insert(target) {
                    return Err(invalid("duplicate/overlapping target inventory"));
                }
            }
            for (path, file) in &component.files {
                validate_file(path, file, &mut paths)?;
            }
        }
        for required in [
            ComponentRole::Dit,
            ComponentRole::TextEncoder,
            ComponentRole::Autoencoder,
        ] {
            if !roles.contains(&required) {
                return Err(invalid(format!("missing {} inventory", required.name())));
            }
        }
        if self.package_files.is_empty() {
            return Err(invalid("missing package config inventory"));
        }
        for (path, file) in &self.package_files {
            validate_file(path, file, &mut paths)?;
        }
        for required in ["config.json", "LICENSE.md", "NOTICE.md"] {
            if !self.package_files.contains_key(required) {
                return Err(invalid(format!("missing package {required} inventory")));
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<Value> {
        self.validate()?;
        Ok(
            json!({"format_version":RECEIPT_VERSION,"source_repository":self.source_repository,"source_revision":self.source_revision,"source_artifacts":self.source_artifacts,"algorithm":self.algorithm,"algorithm_version":self.algorithm_version,"bits":self.bits,"group_size":self.group_size,"components":self.components.iter().map(|c|json!({"role":c.role.name(),"placement":c.placement.name(),"source_repository":c.source_repository,"source_revision":c.source_revision,"source_artifacts":c.source_artifacts,"representation":c.representation,"files":files_json(&c.files),"quantized_targets":c.quantized_targets,"dense_targets":c.dense_targets})).collect::<Vec<_>>(),"package_files":files_json(&self.package_files)}),
        )
    }

    pub fn from_json(v: &Value) -> Result<Self> {
        if v["format_version"].as_u64() != Some(RECEIPT_VERSION) {
            return Err(invalid("unsupported receipt version"));
        }
        let source_artifacts = v["source_artifacts"]
            .as_object()
            .ok_or_else(|| invalid("missing source artifacts"))?
            .iter()
            .map(|(k, v)| Ok((k.clone(), string(v)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let components = v["components"]
            .as_array()
            .ok_or_else(|| invalid("missing component inventories"))?
            .iter()
            .map(|c| {
                Ok(ComponentInventory {
                    role: ComponentRole::parse(&string(&c["role"])?)?,
                    placement: ComponentPlacement::parse(&string(&c["placement"])?)?,
                    source_repository: string(&c["source_repository"])?,
                    source_revision: string(&c["source_revision"])?,
                    source_artifacts: c["source_artifacts"]
                        .as_object()
                        .ok_or_else(|| invalid("missing component source artifacts"))?
                        .iter()
                        .map(|(k, v)| Ok((k.clone(), string(v)?)))
                        .collect::<Result<BTreeMap<_, _>>>()?,
                    representation: string(&c["representation"])?,
                    files: parse_files(&c["files"])?,
                    quantized_targets: strings(&c["quantized_targets"])?,
                    dense_targets: strings(&c["dense_targets"])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let receipt = Self {
            source_repository: string(&v["source_repository"])?,
            source_revision: string(&v["source_revision"])?,
            source_artifacts,
            algorithm: string(&v["algorithm"])?,
            algorithm_version: u32_value(&v["algorithm_version"])?,
            bits: optional_u32(&v["bits"])?,
            group_size: optional_u32(&v["group_size"])?,
            components,
            package_files: parse_files(&v["package_files"])?,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    /// Files the conversion OUTPUT owns: package files plus every owned component. External
    /// components live in the original source root and are verified there.
    fn files(&self) -> impl Iterator<Item = (&String, &OutputFile)> {
        self.package_files.iter().chain(
            self.components
                .iter()
                .filter(|c| c.placement == ComponentPlacement::Owned)
                .flat_map(|c| c.files.iter()),
        )
    }

    /// Full-byte verification of every EXTERNAL component against `source_root` (the original
    /// immutable package the conversion read). Run at install/open alongside
    /// [`Self::verify_outputs`]; a mismatch means the external files are not the converted
    /// source and the package must not load.
    pub fn verify_external(&self, source_root: &Path) -> Result<()> {
        self.verify_external_with(|_, path| Ok(source_root.join(path)))
    }

    /// [`Self::verify_external`] where each component is resolved where the LOADER reads it
    /// (e.g. a `text_encoder` component override directory). `resolve` maps a recorded
    /// source-relative path of one role to the file actually opened.
    pub fn verify_external_with(
        &self,
        resolve: impl Fn(ComponentRole, &str) -> Result<PathBuf>,
    ) -> Result<()> {
        self.validate()?;
        for component in self
            .components
            .iter()
            .filter(|c| c.placement == ComponentPlacement::External)
        {
            for (path, want) in &component.files {
                let got = hash_file(&resolve(component.role, path)?)?;
                if &got != want {
                    return Err(invalid(format!(
                        "external {} file {path} does not match the recorded source checksum",
                        component.role.name()
                    )));
                }
            }
            check_shard_index(component, &|path| resolve(component.role, path))?;
        }
        Ok(())
    }

    /// Whether any component stays in the original source root.
    pub fn has_external_components(&self) -> bool {
        self.components
            .iter()
            .any(|c| c.placement == ComponentPlacement::External)
    }

    /// Full-byte integrity verification occurs at conversion/install/open, not on
    /// every admission/cache poll. A receipt is not proof without this check.
    pub fn verify_outputs(&self, root: &Path) -> Result<()> {
        self.validate()?;
        if !fs::symlink_metadata(root)?.file_type().is_dir() {
            return Err(invalid("conversion root must be an owned real directory"));
        }
        let mut actual = BTreeSet::new();
        collect_files(root, root, &mut actual)?;
        actual.remove(RECEIPT_FILE);
        let expected: BTreeSet<String> = self.files().map(|(p, _)| p.clone()).collect();
        if actual != expected {
            return Err(invalid(
                "output file inventory is incomplete or contains unlisted files",
            ));
        }
        for (path, want) in self.files() {
            let path = root.join(path);
            if &hash_file(&path)? != want {
                return Err(invalid(format!(
                    "{}: output size or checksum mismatch",
                    path.display()
                )));
            }
        }
        for component in self
            .components
            .iter()
            .filter(|c| c.placement == ComponentPlacement::Owned)
        {
            check_shard_index(component, &|path| Ok(root.join(path)))?;
        }
        Ok(())
    }
}

/// Full-file size + SHA-256 of `path` (a regular file, never a symlink).
pub fn hash_file(path: &Path) -> Result<OutputFile> {
    if !fs::symlink_metadata(path)?.file_type().is_file() {
        return Err(invalid(format!("{} is not a regular file", path.display())));
    }
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut bytes = 0u64;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        bytes += n as u64;
        hash.update(&buf[..n]);
    }
    Ok(OutputFile {
        bytes,
        sha256: format!("{:x}", hash.finalize()),
    })
}

fn check_shard_index(
    component: &ComponentInventory,
    resolve: &dyn Fn(&str) -> Result<PathBuf>,
) -> Result<()> {
    if component.role == ComponentRole::TextEncoder {
        let index_path = "text_encoder/model.safetensors.index.json";
        if component.files.contains_key(index_path) {
            if component
                .files
                .contains_key("text_encoder/model.safetensors")
            {
                return Err(invalid("ambiguous single-file and sharded text encoder"));
            }
            let bytes = fs::read(resolve(index_path)?)?;
            let index: Value = serde_json::from_slice(&bytes).map_err(invalid)?;
            let map = index["weight_map"]
                .as_object()
                .filter(|m| !m.is_empty())
                .ok_or_else(|| invalid("empty or torn text-encoder shard index"))?;
            let mut shards = BTreeSet::new();
            for value in map.values() {
                let name = string(value)?;
                if !safe_relative(&name) || !name.ends_with(".safetensors") {
                    return Err(invalid("invalid shard path"));
                }
                let path = format!("text_encoder/{name}");
                if !component.files.contains_key(&path) {
                    return Err(invalid(format!("shard index names unowned/missing {path}")));
                }
                shards.insert(path);
            }
            let inventoried: BTreeSet<_> = component
                .files
                .keys()
                .filter(|p| p.ends_with(".safetensors"))
                .cloned()
                .collect();
            if shards != inventoried {
                return Err(invalid(
                    "shard index and component shard inventory disagree",
                ));
            }
        }
    }
    Ok(())
}

fn required_component_files(component: &ComponentInventory) -> Result<()> {
    let required: &[&str] = match component.role {
        ComponentRole::Dit => &["model.safetensors"],
        ComponentRole::TextEncoder => &[
            "text_encoder/config.json",
            "text_encoder/tokenizer.json",
            "text_encoder/tokenizer_config.json",
        ],
        ComponentRole::Autoencoder => &["ae.safetensors"],
        ComponentRole::PromptUpsampler => &[
            "prompt_upsampler/config.json",
            "prompt_upsampler/model.safetensors",
            "prompt_upsampler/tokenizer.json",
            "prompt_upsampler/tokenizer_config.json",
            "prompt_upsampler/chat_template.jinja",
            "prompt_upsampler/preprocessor_config.json",
            "prompt_upsampler/system_prompts.json",
        ],
    };
    if let Some(missing) = required
        .iter()
        .find(|path| !component.files.contains_key(**path))
    {
        return Err(invalid(format!(
            "{} missing runtime inventory {missing}",
            component.role.name()
        )));
    }
    if component.role == ComponentRole::TextEncoder
        && !component
            .files
            .contains_key("text_encoder/model.safetensors")
        && !component
            .files
            .contains_key("text_encoder/model.safetensors.index.json")
    {
        return Err(invalid(
            "text encoder needs its complete single file or shard index",
        ));
    }
    Ok(())
}

fn validate_file<'a>(path: &'a str, file: &OutputFile, seen: &mut BTreeSet<&'a str>) -> Result<()> {
    if !safe_relative(path)
        || path == RECEIPT_FILE
        || file.bytes == 0
        || !hash_is_valid(&file.sha256)
        || !seen.insert(path)
    {
        return Err(invalid(format!(
            "invalid/duplicate output inventory entry {path}"
        )));
    }
    Ok(())
}
fn string(v: &Value) -> Result<String> {
    v.as_str()
        .map(str::to_owned)
        .ok_or_else(|| invalid("expected a string"))
}
fn strings(v: &Value) -> Result<Vec<String>> {
    v.as_array()
        .ok_or_else(|| invalid("expected target array"))?
        .iter()
        .map(string)
        .collect()
}
fn u32_value(v: &Value) -> Result<u32> {
    v.as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| invalid("expected u32 parameter"))
}
fn optional_u32(v: &Value) -> Result<Option<u32>> {
    if v.is_null() {
        Ok(None)
    } else {
        u32_value(v).map(Some)
    }
}
fn files_json(files: &BTreeMap<String, OutputFile>) -> Value {
    Value::Object(
        files
            .iter()
            .map(|(p, f)| (p.clone(), json!({"bytes":f.bytes,"sha256":f.sha256})))
            .collect(),
    )
}
fn parse_files(v: &Value) -> Result<BTreeMap<String, OutputFile>> {
    v.as_object()
        .ok_or_else(|| invalid("expected file inventory object"))?
        .iter()
        .map(|(p, f)| {
            Ok((
                p.clone(),
                OutputFile {
                    bytes: f["bytes"]
                        .as_u64()
                        .ok_or_else(|| invalid("missing file size"))?,
                    sha256: string(&f["sha256"])?,
                },
            ))
        })
        .collect()
}
fn collect_files(root: &Path, dir: &Path, out: &mut BTreeSet<String>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let path = entry.path();
        if ty.is_symlink() {
            return Err(invalid("conversion output must not contain symlinks"));
        }
        if ty.is_dir() {
            collect_files(root, &path, out)?;
        } else if ty.is_file() {
            let name = path
                .strip_prefix(root)
                .map_err(|_| invalid("output escaped owned directory"))?
                .to_str()
                .ok_or_else(|| invalid("non-UTF8 output path"))?
                .replace('\\', "/");
            out.insert(name);
        } else {
            return Err(invalid("output is not a regular file/directory"));
        }
    }
    Ok(())
}

struct OwnedOutput {
    stage: PathBuf,
    lock: PathBuf,
    lock_file: Option<File>,
}
impl Drop for OwnedOutput {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.stage);
        drop(self.lock_file.take());
        let _ = fs::remove_file(&self.lock);
    }
}

/// Publish only to a NEW immutable destination. The lock is create-new: a foreign
/// or stale lock is an actionable refusal, never permission to delete it. All
/// cleanup names were created by this call; no sibling wildcard sweep is used.
pub fn publish_conversion(
    final_dir: &Path,
    write: impl FnOnce(&Path) -> Result<ConversionReceipt>,
) -> Result<()> {
    let parent = final_dir
        .parent()
        .ok_or_else(|| invalid("output needs a parent directory"))?;
    let name = final_dir
        .file_name()
        .and_then(|v| v.to_str())
        .ok_or_else(|| invalid("invalid output name"))?;
    fs::create_dir_all(parent)?;
    let lock = parent.join(format!(".{name}.flux3-conversion.lock"));
    let lock_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock)
        .map_err(|e| invalid(format!("cannot claim output lock {}: {e}", lock.display())))?;
    let stage = parent.join(format!(
        ".{name}.flux3-conversion-{}-{}",
        std::process::id(),
        OUTPUT_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    // Do not construct the cleanup owner until the stage was actually created.
    if let Err(e) = fs::create_dir(&stage) {
        drop(lock_file);
        let _ = fs::remove_file(&lock);
        return Err(e.into());
    }
    let owned = OwnedOutput {
        stage,
        lock,
        lock_file: Some(lock_file),
    };
    if final_dir.exists() {
        return Err(invalid(
            "destination already exists; choose a new owned output directory",
        ));
    }
    let receipt = write(&owned.stage)?;
    receipt.verify_outputs(&owned.stage)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(owned.stage.join(RECEIPT_FILE))?;
    file.write_all(&serde_json::to_vec_pretty(&receipt.to_json()?).map_err(invalid)?)?;
    file.sync_all()?;
    drop(file);
    for (path, _) in receipt.files() {
        OpenOptions::new()
            .read(true)
            .write(true)
            .open(owned.stage.join(path))?
            .sync_all()?;
    }
    if final_dir.exists() {
        return Err(invalid(
            "destination appeared during conversion; refusing replacement",
        ));
    }
    fs::rename(&owned.stage, final_dir)?;
    // On platforms that allow opening directories, sync the rename as well.
    if let Ok(dir) = File::open(parent) {
        dir.sync_all()?;
    }
    Ok(())
}

pub fn open_conversion(dir: &Path) -> Result<ConversionReceipt> {
    let path = dir.join(RECEIPT_FILE);
    let metadata = fs::symlink_metadata(&path)?;
    if !metadata.is_file() || metadata.len() > 16 * 1024 * 1024 {
        return Err(invalid("invalid conversion receipt file"));
    }
    let value = serde_json::from_slice(&fs::read(path)?).map_err(invalid)?;
    let receipt = ConversionReceipt::from_json(&value)?;
    receipt.verify_outputs(dir)?;
    Ok(receipt)
}

#[cfg(test)]
#[path = "conversion_tests.rs"]
mod tests;
