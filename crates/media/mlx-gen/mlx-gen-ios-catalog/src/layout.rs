//! Path-free artifact layout identity for iOS evidence keys (every provider except SANA, which keeps
//! its own stricter snapshot profile in `admission.rs`).
//!
//! The digest covers every configured source by ROLE (weights, control, ip_adapter, adapters,
//! named components, ...), never by absolute path: the same artifact installed under another
//! directory has the same identity, so a capture binds the artifact rather than one install. Within
//! a directory source, tensors are keyed by their relative component directory and tensor name
//! (header dtype/shape/bytes, never payload); small config/tokenizer files (`json`/`txt`) are
//! hashed by relative path and content; other files by relative path and length. Tensor payloads
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
            Some("json" | "txt") => {
                other.insert(rel, std::fs::read(&file)?);
            }
            _ => {
                other.insert(rel, std::fs::metadata(&file)?.len().to_le_bytes().to_vec());
            }
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
