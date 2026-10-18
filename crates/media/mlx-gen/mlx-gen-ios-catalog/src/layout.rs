//! Path-free artifact layout identity for iOS evidence keys (every provider except SANA, which keeps
//! its own stricter snapshot profile in `admission.rs`).
//!
//! The digest covers every configured source by ROLE (weights, control, ip_adapter, adapters,
//! named components, ...), never by absolute path: the same artifact installed under another
//! directory has the same identity, so a capture binds the artifact rather than one install. Within
//! a directory source, tensors are keyed by their relative component directory and tensor name
//! (header dtype/shape/bytes, never payload); config and tokenizer files (json/yaml configs,
//! merges/vocab/tokenizer files) are hashed by relative path and content; every other file
//! (README, license, model card, image) is outside the identity. Tensor payloads
//! are never read. Replacement after admission is caught separately by the retained
//! `SourceInventory` pins, not by this digest.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mlx_gen::gen_core::weightsmeta::{
    is_hidden_dir, is_hidden_file, safetensors_path_tensor_headers,
};
use mlx_gen::gen_core::{Error, LoadSpec, Result, WeightsSource};
use sha2::{Digest, Sha256};

fn collect(root: &Path, dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_symlink() && path.is_dir() {
            return Err(Error::Unsupported(format!(
                "iOS evidence: symlinked subdirectory {}",
                path.strip_prefix(root).unwrap_or(&path).display()
            )));
        }
        if path.is_dir() {
            if !is_hidden_dir(&path) {
                collect(root, &path, files)?;
            }
        } else if !is_hidden_file(&path) {
            files.push(path);
        }
    }
    Ok(())
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// The non-tensor files that shape memory: model/scheduler configs and tokenizer inputs (they set
/// architecture, packed-tier markers and sequence shape). Documentation and other files are not
/// part of the identity, so adding a README never changes an evidence key.
fn is_config_or_tokenizer(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match path.extension().and_then(|e| e.to_str()) {
        Some("json" | "yaml" | "yml" | "model" | "tiktoken") => true,
        Some("txt") => name == "merges.txt" || name == "vocab.txt" || name.contains("tokenizer"),
        _ => false,
    }
}

/// Hash one role's source; returns how many tensor headers it contributed.
fn source(digest: &mut Sha256, role: &str, source: &WeightsSource) -> Result<usize> {
    digest.update((role.len() as u64).to_le_bytes());
    digest.update(role.as_bytes());
    let (root, mut files) = match source {
        WeightsSource::File(path) => (
            path.parent().map(Path::to_owned).unwrap_or_default(),
            vec![path.clone()],
        ),
        WeightsSource::Dir(path) => {
            let mut files = Vec::new();
            collect(path, path, &mut files)?;
            (path.clone(), files)
        }
    };
    files.sort();
    let single_file = matches!(source, WeightsSource::File(_));
    // Tensors per relative component directory, by name: shard file names are not identity.
    let mut tensors: BTreeMap<(String, String), String> = BTreeMap::new();
    let mut other: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for file in files {
        let rel = if single_file {
            String::new()
        } else {
            relative(&root, &file)
        };
        match file.extension().and_then(|s| s.to_str()) {
            Some("safetensors") => {
                let component = rel.rsplit_once('/').map(|(dir, _)| dir.to_owned());
                for header in safetensors_path_tensor_headers(&file)? {
                    let entry = format!(
                        "{}:{:?}:{}",
                        header.dtype.as_str(),
                        header.shape,
                        header.data_bytes
                    );
                    if tensors
                        .insert((component.clone().unwrap_or_default(), header.name), entry)
                        .is_some()
                    {
                        return Err(Error::Unsupported(format!(
                            "iOS evidence: duplicate tensor in {role} {}",
                            component.unwrap_or_default()
                        )));
                    }
                }
            }
            _ if is_config_or_tokenizer(&file) => {
                other.insert(rel, std::fs::read(&file)?);
            }
            // READMEs, licenses, model cards, images, caches: not memory identity.
            _ => {}
        }
    }
    let count = tensors.len();
    for ((component, name), entry) in tensors {
        for part in [component.as_bytes(), name.as_bytes(), entry.as_bytes()] {
            digest.update((part.len() as u64).to_le_bytes());
            digest.update(part);
        }
    }
    for (rel, content) in other {
        for part in [rel.as_bytes(), content.as_slice()] {
            digest.update((part.len() as u64).to_le_bytes());
            digest.update(part);
        }
    }
    Ok(count)
}

/// Path-free layout digest of every configured source plus the spec's memory-relevant load
/// settings. Errors: `Unsupported` for an inventory outside the admitted shape (an artifact
/// refusal), anything else for unreadable/malformed sources (a failure).
pub fn artifact_layout_digest(spec: &LoadSpec) -> Result<[u8; 32]> {
    let mut digest = Sha256::new();
    digest.update(b"ios-artifact-layout-v1");
    digest.update(
        format!(
            "{:?}|{:?}|{:?}|{:?}|{:?}",
            spec.precision,
            spec.quantize,
            spec.offload_policy,
            spec.load_shape,
            spec.resolved_route
        )
        .as_bytes(),
    );
    let mut tensors = source(&mut digest, "weights", &spec.weights)?;
    let mut roles: Vec<(String, &WeightsSource)> = Vec::new();
    if let Some(s) = &spec.control {
        roles.push(("control".into(), s));
    }
    for (i, s) in spec.extra_controls.iter().enumerate() {
        roles.push((format!("extra_control.{i}"), s));
    }
    if let Some(s) = &spec.ip_adapter {
        roles.push(("ip_adapter".into(), s));
    }
    if let Some(s) = &spec.text_encoder {
        roles.push(("text_encoder".into(), s));
    }
    if let Some(pid) = &spec.pid {
        roles.push(("pid.checkpoint".into(), &pid.checkpoint));
        roles.push(("pid.gemma".into(), &pid.gemma));
    }
    if let Some(identity) = &spec.identity {
        for (name, s) in [
            ("identity.encoder", &identity.encoder),
            ("identity.eva", &identity.eva),
            ("identity.face_dir", &identity.face_dir),
        ] {
            if let Some(s) = s {
                roles.push((name.into(), s));
            }
        }
    }
    for (name, s) in &spec.components {
        roles.push((format!("component.{name}"), s));
    }
    // Adapter kind changes the injected structure; its scale does not change memory.
    let adapters: Vec<(String, WeightsSource)> = spec
        .adapters
        .iter()
        .enumerate()
        .map(|(i, a)| {
            (
                format!("adapter.{i}.{:?}", a.kind),
                WeightsSource::File(a.path.clone()),
            )
        })
        .collect();
    for (role, s) in &adapters {
        roles.push((role.clone(), s));
    }
    for (role, s) in roles {
        tensors += source(&mut digest, &role, s)?;
    }
    if tensors == 0 {
        return Err(Error::Unsupported(
            "iOS evidence: no tensor headers in the supplied sources".into(),
        ));
    }
    Ok(digest.finalize().into())
}

/// Path-free ARCHITECTURE digest of the spec's BASE components (the weights directory, a separate
/// text encoder and named components), from safetensors HEADERS alone: per component directory and
/// tensor name, its dtype and shape (so packing — `.scales`/`.biases` tensors and packed shapes —
/// is read from the headers, never from a config marker), plus the load settings that shape memory
/// (precision, requested tier, offload policy, load shape). No tensor value, no config content, no
/// file name, no path, no route. Two checkpoints of one architecture (fine-tunes sharing every
/// tensor name, shape and dtype) share it; a distilled U-Net or another VAE that differs in any
/// header does not.
///
/// Mode-added components — a ControlNet, an IP-Adapter, adapter overlays — are deliberately NOT
/// part of it: they are component-additive and priced by the counted formulas on top of the
/// family's anchor (see `crate::anchor`), so a request that carries one still belongs to the base
/// architecture's family.
pub fn architecture_digest(spec: &LoadSpec) -> Result<[u8; 32]> {
    let mut digest = Sha256::new();
    digest.update(b"ios-architecture-v1");
    digest.update(
        format!(
            "{:?}|{:?}|{:?}|{:?}",
            spec.precision, spec.quantize, spec.offload_policy, spec.load_shape
        )
        .as_bytes(),
    );
    let mut roles: Vec<(String, &WeightsSource)> = vec![("weights".into(), &spec.weights)];
    if let Some(s) = &spec.text_encoder {
        roles.push(("text_encoder".into(), s));
    }
    for (name, s) in &spec.components {
        roles.push((format!("component.{name}"), s));
    }
    let mut tensors = 0usize;
    for (role, src) in roles {
        digest.update((role.len() as u64).to_le_bytes());
        digest.update(role.as_bytes());
        let (root, mut files) = match src {
            WeightsSource::File(path) => (
                path.parent().map(Path::to_owned).unwrap_or_default(),
                vec![path.clone()],
            ),
            WeightsSource::Dir(path) => {
                let mut files = Vec::new();
                collect(path, path, &mut files)?;
                (path.clone(), files)
            }
        };
        files.sort();
        let single = matches!(src, WeightsSource::File(_));
        let mut entries: BTreeMap<(String, String), String> = BTreeMap::new();
        for file in files {
            if file.extension().and_then(|e| e.to_str()) != Some("safetensors") {
                continue;
            }
            let component = if single {
                String::new()
            } else {
                relative(&root, &file)
                    .rsplit_once('/')
                    .map(|(dir, _)| dir.to_owned())
                    .unwrap_or_default()
            };
            for header in safetensors_path_tensor_headers(&file)? {
                entries.insert(
                    (component.clone(), header.name),
                    format!("{}:{:?}", header.dtype.as_str(), header.shape),
                );
            }
        }
        tensors += entries.len();
        for ((component, name), entry) in entries {
            for part in [component.as_bytes(), name.as_bytes(), entry.as_bytes()] {
                digest.update((part.len() as u64).to_le_bytes());
                digest.update(part);
            }
        }
    }
    if tensors == 0 {
        return Err(Error::Unsupported(
            "iOS evidence: no tensor headers in the base sources".into(),
        ));
    }
    Ok(digest.finalize().into())
}

/// Lowercase hex of a digest.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn write_tensor(path: &Path, dtype: &str, count: usize) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let bytes = count * if dtype == "F32" { 4 } else { 2 };
        let header = format!(
            r#"{{"x":{{"dtype":"{dtype}","shape":[{count}],"data_offsets":[0,{bytes}]}}}}"#
        );
        let mut data = (header.len() as u64).to_le_bytes().to_vec();
        data.extend(header.as_bytes());
        data.resize(data.len() + bytes, 0);
        std::fs::write(path, data).unwrap();
    }

    fn snapshot(root: &Path) {
        for c in ["unet", "vae", "text_encoder"] {
            write_tensor(&root.join(c).join("model.safetensors"), "F16", 8);
        }
        std::fs::write(root.join("unet/config.json"), b"{\"a\":1}").unwrap();
    }

    #[test]
    fn identity_is_install_path_free_but_layout_and_config_sensitive() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        snapshot(a.path());
        snapshot(b.path());
        let spec = |p: &Path| LoadSpec::new(WeightsSource::Dir(p.to_owned()));
        let reference = artifact_layout_digest(&spec(a.path())).unwrap();
        assert_eq!(
            reference,
            artifact_layout_digest(&spec(b.path())).unwrap(),
            "the same artifact under another install path must keep its identity"
        );
        // Renaming a shard is not a layout change.
        std::fs::rename(
            b.path().join("vae/model.safetensors"),
            b.path().join("vae/renamed.safetensors"),
        )
        .unwrap();
        assert_eq!(reference, artifact_layout_digest(&spec(b.path())).unwrap());
        // Dtype, shape, config content and component membership are.
        write_tensor(&b.path().join("vae/renamed.safetensors"), "F32", 8);
        assert_ne!(reference, artifact_layout_digest(&spec(b.path())).unwrap());
        snapshot(b.path());
        std::fs::remove_file(b.path().join("vae/renamed.safetensors")).unwrap();
        assert_eq!(reference, artifact_layout_digest(&spec(b.path())).unwrap());
        std::fs::write(b.path().join("unet/config.json"), b"{\"a\":2}").unwrap();
        assert_ne!(reference, artifact_layout_digest(&spec(b.path())).unwrap());
        snapshot(b.path());
        // Documentation is not identity; configs (yaml too) and tokenizer files are.
        std::fs::write(b.path().join("README.md"), b"# card").unwrap();
        std::fs::write(b.path().join("LICENSE.txt"), b"terms").unwrap();
        std::fs::write(b.path().join("unet/notes.txt"), b"notes").unwrap();
        assert_eq!(reference, artifact_layout_digest(&spec(b.path())).unwrap());
        std::fs::write(b.path().join("config.yaml"), b"a: 1").unwrap();
        assert_ne!(reference, artifact_layout_digest(&spec(b.path())).unwrap());
        std::fs::remove_file(b.path().join("config.yaml")).unwrap();
        std::fs::write(b.path().join("merges.txt"), b"#version").unwrap();
        assert_ne!(reference, artifact_layout_digest(&spec(b.path())).unwrap());
        std::fs::remove_file(b.path().join("merges.txt")).unwrap();
        write_tensor(&b.path().join("extra/model.safetensors"), "F16", 8);
        assert_ne!(reference, artifact_layout_digest(&spec(b.path())).unwrap());
        // Load settings and overlays participate by role.
        let mut quantized = spec(a.path());
        quantized.quantize = Some(mlx_gen::gen_core::Quant::Q4);
        assert_ne!(reference, artifact_layout_digest(&quantized).unwrap());
        let extra = tempfile::tempdir().unwrap();
        let vae = extra.path().join("fix-vae.safetensors");
        write_tensor(&vae, "F16", 8);
        let mut with_vae = spec(a.path());
        with_vae
            .components
            .insert("vae_fp16_fix".into(), WeightsSource::File(vae.clone()));
        assert_ne!(reference, artifact_layout_digest(&with_vae).unwrap());
    }

    #[test]
    fn empty_and_duplicate_inventories_are_artifact_refusals() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("config.json"), b"{}").unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
        assert!(matches!(
            artifact_layout_digest(&spec),
            Err(Error::Unsupported(_))
        ));
        write_tensor(&root.path().join("unet/a.safetensors"), "F16", 8);
        write_tensor(&root.path().join("unet/b.safetensors"), "F16", 8);
        assert!(matches!(
            artifact_layout_digest(&spec),
            Err(Error::Unsupported(_))
        ));
        std::fs::write(root.path().join("unet/b.safetensors"), b"bad").unwrap();
        assert!(
            !matches!(artifact_layout_digest(&spec), Err(Error::Unsupported(_))),
            "a malformed shard is a failure, not an artifact refusal"
        );
    }
}
