//! Committed conversion manifests (`manifests/<component>.json`).
//!
//! Produced by `scripts/reference/yue2/asset_manifest.py` from the pinned originals under the
//! pinned reference environment (PyTorch + `safetensors`), and committed. A manifest records:
//!
//! * the source repository, revision, and `bytes` / `sha256` of every closure file;
//! * the conversion — `identity` for every YuE2 component (published BF16/F32 safetensors and the
//!   tiktoken ranks file are loaded natively as-is), with the native file's `bytes` / `sha256`;
//! * for weights, one [`TensorEntry`] per tensor in file order: name, dtype, shape, and the SHA-256
//!   of the tensor's little-endian bytes as PyTorch loads them.
//!
//! The snapshot verifier ([`crate::snapshot`]) checks a snapshot's safetensors header against
//! [`ConversionManifest::tensors`], and the real-weight test re-derives every tensor digest through
//! Candle ([`crate::snapshot::native_tensor_digests`]) and compares.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::snapshot::AssetError;

/// The only manifest schema this crate reads.
pub const SCHEMA: u64 = 1;

/// The conversion a component's native form went through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConversionKind {
    /// The native file *is* the pinned original, byte for byte.
    Identity,
}

/// One tensor of a weights file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorEntry {
    /// The tensor name as stored.
    pub name: String,
    /// The safetensors dtype tag (`"BF16"`, `"F32"`, …).
    pub dtype: String,
    /// The shape.
    pub shape: Vec<usize>,
    /// SHA-256 of the tensor's little-endian element bytes.
    pub sha256: String,
}

/// A size + SHA-256 pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileDigest {
    /// Exact byte size.
    pub bytes: u64,
    /// Lower-case hex SHA-256.
    pub sha256: String,
}

/// A parsed conversion manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversionManifest {
    /// The component key it describes.
    pub component: String,
    /// The source repository.
    pub repo: String,
    /// The source revision.
    pub revision: String,
    /// Every closure file of the source, by snapshot-relative path.
    pub source_files: BTreeMap<String, FileDigest>,
    /// The conversion applied.
    pub conversion: ConversionKind,
    /// The native file's snapshot-relative path.
    pub native_file: String,
    /// The native file's size and SHA-256.
    pub native: FileDigest,
    /// Every tensor of the native weights file, in file order (empty for the tokenizer).
    pub tensors: Vec<TensorEntry>,
    /// The ordinary-token count of a tiktoken ranks file (tokenizer component only).
    pub ordinary_tokens: Option<u64>,
}

fn bad(key: &str, what: impl std::fmt::Display) -> AssetError {
    AssetError::Manifest {
        component: key.to_string(),
        detail: what.to_string(),
    }
}

fn field<'v>(key: &str, v: &'v Value, name: &str) -> Result<&'v Value, AssetError> {
    v.get(name)
        .ok_or_else(|| bad(key, format!("missing field `{name}`")))
}

fn string(key: &str, v: &Value, name: &str) -> Result<String, AssetError> {
    field(key, v, name)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| bad(key, format!("`{name}` is not a string")))
}

fn uint(key: &str, v: &Value, name: &str) -> Result<u64, AssetError> {
    field(key, v, name)?
        .as_u64()
        .ok_or_else(|| bad(key, format!("`{name}` is not an unsigned integer")))
}

fn sha256(key: &str, v: &Value) -> Result<String, AssetError> {
    let sha256 = string(key, v, "sha256")?;
    if sha256.len() != 64
        || !sha256
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(bad(
            key,
            format!("`{sha256}` is not a lower-case hex SHA-256"),
        ));
    }
    Ok(sha256)
}

fn digest(key: &str, v: &Value) -> Result<FileDigest, AssetError> {
    Ok(FileDigest {
        bytes: uint(key, v, "bytes")?,
        sha256: sha256(key, v)?,
    })
}

impl ConversionManifest {
    /// Parse a manifest, requiring it to describe `key`. Every field the verifier relies on must be
    /// present and well-typed; nothing defaults.
    pub fn parse(key: &str, json: &str) -> Result<Self, AssetError> {
        let v: Value = serde_json::from_str(json).map_err(|e| bad(key, e))?;
        let schema = uint(key, &v, "schema")?;
        if schema != SCHEMA {
            return Err(bad(key, format!("schema {schema}, expected {SCHEMA}")));
        }
        let component = string(key, &v, "component")?;
        if component != key {
            return Err(bad(key, format!("describes `{component}`")));
        }
        let source = field(key, &v, "source")?;
        let mut source_files = BTreeMap::new();
        for (path, d) in field(key, source, "files")?
            .as_object()
            .ok_or_else(|| bad(key, "`source.files` is not an object"))?
        {
            source_files.insert(path.clone(), digest(key, d)?);
        }
        let conversion = match string(key, field(key, &v, "conversion")?, "kind")?.as_str() {
            "identity" => ConversionKind::Identity,
            other => return Err(bad(key, format!("unknown conversion kind `{other}`"))),
        };
        let native = field(key, &v, "native")?;
        let mut tensors = Vec::new();
        if let Some(rows) = v.get("tensors") {
            for row in rows
                .as_array()
                .ok_or_else(|| bad(key, "`tensors` is not an array"))?
            {
                let shape = field(key, row, "shape")?
                    .as_array()
                    .ok_or_else(|| bad(key, "`shape` is not an array"))?
                    .iter()
                    .map(|d| {
                        d.as_u64()
                            .map(|d| d as usize)
                            .ok_or_else(|| bad(key, "shape dimension is not an integer"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let entry = TensorEntry {
                    name: string(key, row, "name")?,
                    dtype: string(key, row, "dtype")?,
                    shape,
                    sha256: sha256(key, row)?,
                };
                tensors.push(entry);
            }
        }
        let ordinary_tokens = match v.get("tokenizer") {
            Some(t) => Some(uint(key, t, "ordinary_tokens")?),
            None => None,
        };
        Ok(Self {
            component,
            repo: string(key, source, "repo")?,
            revision: string(key, source, "revision")?,
            source_files,
            conversion,
            native_file: string(key, native, "file")?,
            native: digest(key, native)?,
            tensors,
            ordinary_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::{ComponentId, FileRole};

    /// Double-entry check: every committed manifest agrees with the Rust inventory on repository,
    /// revision and every file's size and SHA-256, and its native file is the pinned weights (or
    /// tokenizer) file byte for byte. A transcription slip in either place fails here.
    #[test]
    fn committed_manifests_agree_with_the_inventory() {
        for id in ComponentId::ALL {
            let c = id.component();
            let m = c.conversion_manifest().unwrap();
            assert_eq!(m.repo, c.repo.id, "{}", c.key);
            assert_eq!(m.revision, c.repo.revision, "{}", c.key);
            assert_eq!(m.conversion, ConversionKind::Identity);
            let pinned: BTreeMap<String, FileDigest> = c
                .files
                .iter()
                .map(|f| {
                    (
                        f.path.to_string(),
                        FileDigest {
                            bytes: f.bytes,
                            sha256: f.sha256.to_string(),
                        },
                    )
                })
                .collect();
            assert_eq!(m.source_files, pinned, "{}", c.key);
            let native = c
                .files
                .iter()
                .find(|f| matches!(f.role, FileRole::Weights | FileRole::Tokenizer))
                .unwrap();
            assert_eq!(m.native_file, native.path, "{}", c.key);
            assert_eq!(m.native.sha256, native.sha256, "{}", c.key);
            assert_eq!(m.native.bytes, native.bytes, "{}", c.key);
            if native.role == FileRole::Weights {
                assert!(!m.tensors.is_empty(), "{} has no tensor table", c.key);
                let mut names = std::collections::BTreeSet::new();
                for t in &m.tensors {
                    assert!(
                        names.insert(&t.name),
                        "{}: duplicate tensor {}",
                        c.key,
                        t.name
                    );
                    assert!(
                        matches!(t.dtype.as_str(), "BF16" | "F32"),
                        "{}: {} is {}; the identity conversion assumes Candle-native BF16/F32",
                        c.key,
                        t.name,
                        t.dtype
                    );
                }
            } else {
                assert_eq!(m.ordinary_tokens, Some(151_643));
            }
        }
    }

    #[test]
    fn a_manifest_for_another_component_or_schema_is_refused() {
        let json = crate::inventory::VAE_STANDARD.manifest_json;
        let err = ConversionManifest::parse("yue2_vae_legacy", json).unwrap_err();
        assert!(err.to_string().contains("describes `yue2_vae`"), "{err}");
        let err = ConversionManifest::parse(
            "yue2_vae",
            &json.replacen("\"schema\": 1", "\"schema\": 2", 1),
        )
        .unwrap_err();
        assert!(err.to_string().contains("schema 2"), "{err}");
        let err = ConversionManifest::parse(
            "yue2_vae",
            &json.replacen("\"kind\": \"identity\"", "\"kind\": \"requantize\"", 1),
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown conversion kind"), "{err}");
    }
}
