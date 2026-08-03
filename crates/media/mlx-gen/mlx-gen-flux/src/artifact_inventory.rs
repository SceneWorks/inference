//! Exact file-identity and content pins for FLUX.1 deferred-materialization eligibility.
//!
//! A streamable snapshot has one visible `model.safetensors` in each required component directory.
//! Each file is canonicalized, pinned by mutation-sensitive Unix identity, and SHA-256 hashed with a
//! process-local coalescing cache. The composite digest covers those four model files plus the T5
//! `tokenizer_2/tokenizer.json` consumed by prompt conditioning; it is evidence input only and never
//! grants production calibration by itself.

use mlx_gen::gen_core::{Error as CoreError, Result as CoreResult};
use mlx_gen::{LoadSpec, OffloadPolicy, Precision, Quant, WeightsSource};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufReader, Read};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex, OnceLock};

pub(crate) const COMPONENTS: [&str; 4] = ["text_encoder", "text_encoder_2", "transformer", "vae"];

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ArtifactFileIdentity {
    canonical_path: PathBuf,
    device: u64,
    inode: u64,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceEntryIdentity {
    absolute_path: PathBuf,
    is_symlink: bool,
    symlink_target: Option<PathBuf>,
    device: u64,
    inode: u64,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

#[derive(Clone, Debug)]
pub(crate) struct PinnedArtifact {
    source: SourceEntryIdentity,
    identity: ArtifactFileIdentity,
    digest: String,
}

impl PinnedArtifact {
    fn verify_file(path: &Path) -> CoreResult<Self> {
        pinned_artifact(path)
    }

    pub(crate) fn canonical_path(&self) -> &Path {
        &self.identity.canonical_path
    }

    pub(crate) fn digest(&self) -> &str {
        &self.digest
    }

    pub(crate) fn ensure_unchanged(&self) -> CoreResult<()> {
        let source = source_entry_identity(&self.source.absolute_path).map_err(|error| {
            CoreError::Msg(format!(
                "flux1: pinned snapshot entry is no longer readable: {error}"
            ))
        })?;
        if source != self.source {
            return Err(CoreError::Unsupported(
                "flux1: pinned snapshot entry or symlink target changed after verification"
                    .to_owned(),
            ));
        }
        let current = file_identity(&self.identity.canonical_path).map_err(|error| {
            CoreError::Msg(format!(
                "flux1: pinned packed artifact is no longer readable: {error}"
            ))
        })?;
        if current != self.identity {
            return Err(CoreError::Unsupported(
                "flux1: pinned packed artifact was replaced or mutated after verification"
                    .to_owned(),
            ));
        }
        let resolved = file_identity(&self.source.absolute_path).map_err(|error| {
            CoreError::Msg(format!(
                "flux1: pinned snapshot entry no longer resolves: {error}"
            ))
        })?;
        if resolved != self.identity {
            return Err(CoreError::Unsupported(
                "flux1: pinned snapshot entry resolves to a different canonical target".to_owned(),
            ));
        }
        Ok(())
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
}

impl PackedArtifactInventory {
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

fn source_entry_identity(path: &Path) -> std::io::Result<SourceEntryIdentity> {
    let absolute_path = std::path::absolute(path)?;
    let metadata = std::fs::symlink_metadata(&absolute_path)?;
    let is_symlink = metadata.file_type().is_symlink();
    Ok(SourceEntryIdentity {
        symlink_target: is_symlink
            .then(|| std::fs::read_link(&absolute_path))
            .transpose()?,
        absolute_path,
        is_symlink,
        device: metadata.dev(),
        inode: metadata.ino(),
        size: metadata.size(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    })
}

fn file_identity(path: &Path) -> std::io::Result<ArtifactFileIdentity> {
    let canonical_path = std::fs::canonicalize(path)?;
    let metadata = std::fs::metadata(&canonical_path)?;
    if !metadata.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{} is not a regular file", canonical_path.display()),
        ));
    }
    Ok(ArtifactFileIdentity {
        canonical_path,
        device: metadata.dev(),
        inode: metadata.ino(),
        size: metadata.size(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    })
}

#[derive(Clone, Debug)]
enum DigestState {
    Hashing,
    Ready(String),
}

struct DigestCache {
    entries: Mutex<HashMap<ArtifactFileIdentity, DigestState>>,
    ready: Condvar,
}

fn digest_cache() -> &'static DigestCache {
    static CACHE: OnceLock<DigestCache> = OnceLock::new();
    CACHE.get_or_init(|| DigestCache {
        entries: Mutex::new(HashMap::new()),
        ready: Condvar::new(),
    })
}

#[cfg(test)]
fn hash_operation_counts() -> &'static Mutex<HashMap<PathBuf, u64>> {
    static COUNTS: OnceLock<Mutex<HashMap<PathBuf, u64>>> = OnceLock::new();
    COUNTS.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(test)]
type HashHook = Box<dyn FnOnce() + Send>;

#[cfg(test)]
fn hash_completion_hooks() -> &'static Mutex<HashMap<PathBuf, HashHook>> {
    static HOOKS: OnceLock<Mutex<HashMap<PathBuf, HashHook>>> = OnceLock::new();
    HOOKS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn hash_exact_file(identity: &ArtifactFileIdentity) -> CoreResult<String> {
    #[cfg(test)]
    {
        *hash_operation_counts()
            .lock()
            .map_err(|_| CoreError::Msg("flux1: hash-count lock poisoned".to_owned()))?
            .entry(identity.canonical_path.clone())
            .or_default() += 1;
    }
    let file = File::open(&identity.canonical_path).map_err(|error| {
        CoreError::Msg(format!(
            "flux1: open {} for hashing: {error}",
            identity.canonical_path.display()
        ))
    })?;
    let opened = file.metadata().map_err(|error| {
        CoreError::Msg(format!(
            "flux1: stat open {}: {error}",
            identity.canonical_path.display()
        ))
    })?;
    if opened.dev() != identity.device || opened.ino() != identity.inode {
        return Err(CoreError::Unsupported(
            "flux1: packed artifact changed before hashing began".to_owned(),
        ));
    }
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer).map_err(|error| {
            CoreError::Msg(format!(
                "flux1: hash {}: {error}",
                identity.canonical_path.display()
            ))
        })?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    #[cfg(test)]
    if let Some(hook) = hash_completion_hooks()
        .lock()
        .map_err(|_| CoreError::Msg("flux1: hash-hook lock poisoned".to_owned()))?
        .remove(&identity.canonical_path)
    {
        hook();
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn pinned_artifact(path: &Path) -> CoreResult<PinnedArtifact> {
    loop {
        let source = source_entry_identity(path)
            .map_err(|error| CoreError::Msg(format!("flux1: lstat {}: {error}", path.display())))?;
        let identity = file_identity(&source.absolute_path)
            .map_err(|error| CoreError::Msg(format!("flux1: stat {}: {error}", path.display())))?;
        let cache = digest_cache();
        let mut entries = cache
            .entries
            .lock()
            .map_err(|_| CoreError::Msg("flux1: digest cache lock poisoned".to_owned()))?;
        match entries.get(&identity).cloned() {
            Some(DigestState::Ready(digest)) => {
                drop(entries);
                if source_entry_identity(&source.absolute_path).ok().as_ref() == Some(&source)
                    && file_identity(&source.absolute_path).ok().as_ref() == Some(&identity)
                {
                    return Ok(PinnedArtifact {
                        source,
                        identity,
                        digest,
                    });
                }
            }
            Some(DigestState::Hashing) => {
                entries = cache
                    .ready
                    .wait(entries)
                    .map_err(|_| CoreError::Msg("flux1: digest cache wait poisoned".to_owned()))?;
                drop(entries);
            }
            None => {
                entries.insert(identity.clone(), DigestState::Hashing);
                drop(entries);
                let digest = hash_exact_file(&identity);
                let unchanged = source_entry_identity(&source.absolute_path).ok().as_ref()
                    == Some(&source)
                    && file_identity(&source.absolute_path).ok().as_ref() == Some(&identity);
                let mut entries = cache
                    .entries
                    .lock()
                    .map_err(|_| CoreError::Msg("flux1: digest cache lock poisoned".to_owned()))?;
                entries.remove(&identity);
                let result = match (unchanged, digest) {
                    (true, Ok(digest)) => {
                        entries
                            .retain(|cached, _| cached.canonical_path != identity.canonical_path);
                        entries.insert(identity.clone(), DigestState::Ready(digest.clone()));
                        Ok(PinnedArtifact {
                            source,
                            identity,
                            digest,
                        })
                    }
                    (false, _) => Err(CoreError::Unsupported(
                        "flux1: packed artifact changed while its content was hashed".to_owned(),
                    )),
                    (true, Err(error)) => Err(error),
                };
                cache.ready.notify_all();
                return result;
            }
        }
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
    let entries = std::fs::read_dir(dir).map_err(|error| {
        CoreError::Msg(format!(
            "flux1: read packed component {}: {error}",
            dir.display()
        ))
    })?;
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| {
            CoreError::Msg(format!(
                "flux1: enumerate packed component {}: {error}",
                dir.display()
            ))
        })?;
        let path = entry.path();
        if !mlx_gen::gen_core::weightsmeta::is_hidden_file(&path)
            && path.extension().is_some_and(|ext| ext == "safetensors")
        {
            names.push(entry.file_name());
        }
    }
    names.sort();
    Ok(names)
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

fn component_quant_matches(component: &ComponentInventory, quant: Option<Quant>) -> bool {
    let expected = quant.map(Quant::bits);
    let actual = (|| -> CoreResult<Option<i32>> {
        component.config.ensure_unchanged()?;
        let bytes = std::fs::read(component.config.canonical_path()).map_err(|error| {
            CoreError::Msg(format!(
                "flux1: read pinned {} config: {error}",
                component.name
            ))
        })?;
        component.config.ensure_unchanged()?;
        let json: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
            CoreError::Msg(format!(
                "flux1: parse pinned {} config: {error}",
                component.name
            ))
        })?;
        let Some(marker) = json.get("quantization") else {
            return Ok(None);
        };
        let bits = marker.get("bits").and_then(serde_json::Value::as_i64);
        let group_size = marker.get("group_size").and_then(serde_json::Value::as_i64);
        match (bits, group_size) {
            (Some(bits @ (4 | 8)), Some(64)) => Ok(Some(bits as i32)),
            _ => Err(CoreError::Unsupported(format!(
                "flux1: {} has an invalid packed quantization marker",
                component.name
            ))),
        }
    })();
    matches!(actual, Ok(actual) if actual == expected)
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
    if !components
        .iter()
        .all(|component| component_quant_matches(component, spec.quantize))
    {
        return Err(CoreError::Unsupported(
            "flux1: every pinned component quant marker must match the selected tier".to_owned(),
        ));
    }
    let inventory = PackedArtifactInventory {
        components,
        t5_tokenizer,
        composite_sha256: String::new(),
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
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    fn unique_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "flux-artifact-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    fn write_snapshot(root: &Path, quant: Option<Quant>) {
        for (index, component) in COMPONENTS.iter().enumerate() {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), vec![index as u8; 64]).unwrap();
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

    fn eligible_spec(root: &Path, quant: Option<Quant>) -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_owned()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization);
        spec.quantize = quant;
        spec
    }

    #[test]
    fn dense_q4_and_q8_exact_single_file_inventories_are_loadable() {
        for (label, quant) in [
            ("dense-inventory", None),
            ("q4-inventory", Some(Quant::Q4)),
            ("q8-inventory", Some(Quant::Q8)),
        ] {
            let root = unique_root(label);
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
    fn sharded_or_mismatched_component_inventory_is_not_streamable() {
        let root = unique_root("sharded-inventory");
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
        let root = unique_root("inventory-mutation");
        write_snapshot(&root, None);
        let spec = eligible_spec(&root, None);
        let first = verified_stream_inventory(crate::FLUX1_SCHNELL_ID, &spec).unwrap();
        let first_composite = first.composite_sha256().to_owned();
        std::fs::write(root.join("vae/model.safetensors"), [11_u8; 64]).unwrap();
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
        let root = unique_root("missing-tokenizer");
        write_snapshot(&root, None);
        let spec = eligible_spec(&root, None);
        std::fs::remove_file(root.join("tokenizer_2/tokenizer.json")).unwrap();
        assert!(verified_stream_inventory(crate::FLUX1_DEV_ID, &spec).is_none());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn admitted_snapshot_rejects_new_visible_safetensors_member() {
        let root = unique_root("post-admission-extra-file");
        write_snapshot(&root, None);
        let inventory =
            verified_stream_inventory(crate::FLUX1_DEV_ID, &eligible_spec(&root, None)).unwrap();
        std::fs::write(root.join("transformer/extra.safetensors"), [9_u8; 8]).unwrap();
        assert!(inventory.ensure_unchanged().is_err());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn admitted_snapshot_rejects_config_replacement() {
        let root = unique_root("post-admission-config-replacement");
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
        use std::os::unix::fs::symlink;

        let root = unique_root("post-admission-symlink-retarget");
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
        use std::os::unix::fs::symlink;

        let base = unique_root("post-admission-parent-symlink-retarget");
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
        let root = unique_root("structural-mutations");
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
    fn replacement_between_hash_and_post_stat_is_rejected() {
        let root = unique_root("hash-replacement");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("model.safetensors");
        std::fs::write(&path, [1_u8; 32]).unwrap();
        let canonical = std::fs::canonicalize(&path).unwrap();
        let replacement = root.join("replacement.safetensors");
        std::fs::write(&replacement, [2_u8; 32]).unwrap();
        let target = path.clone();
        hash_completion_hooks().lock().unwrap().insert(
            canonical,
            Box::new(move || std::fs::rename(&replacement, &target).unwrap()),
        );
        let error = PinnedArtifact::verify_file(&path).unwrap_err().to_string();
        assert!(error.contains("changed while"));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn same_path_same_size_content_mutation_gets_a_new_digest() {
        let root = unique_root("content-mutation");
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

    #[test]
    fn concurrent_first_use_hashes_one_file_identity_once() {
        let root = unique_root("coalesced-hash");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("model.safetensors");
        std::fs::write(&path, vec![7_u8; 4 * 1024 * 1024]).unwrap();
        let canonical = std::fs::canonicalize(&path).unwrap();
        let before = hash_operation_counts()
            .lock()
            .unwrap()
            .get(&canonical)
            .copied()
            .unwrap_or(0);
        let barrier = Arc::new(Barrier::new(8));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let path = path.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    PinnedArtifact::verify_file(&path)
                        .unwrap()
                        .digest()
                        .to_owned()
                })
            })
            .collect();
        let digests: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert!(digests.iter().all(|digest| digest == &digests[0]));
        assert_eq!(
            hash_operation_counts()
                .lock()
                .unwrap()
                .get(&canonical)
                .copied()
                .unwrap_or(0)
                - before,
            1
        );
        std::fs::remove_dir_all(root).ok();
    }
}
