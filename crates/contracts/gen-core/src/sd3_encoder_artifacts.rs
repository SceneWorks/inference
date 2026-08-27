//! Deterministic SD3.5 text-encoder checkpoint identity resolution.
//!
//! Stock SD3.5 snapshots may cache full-precision and fp16 encoder families side by side. Loading
//! every safetensors file in a component directory makes the selected identity depend on directory
//! contents and backend-specific iteration behavior. This module is the tensor-free, shared source
//! of truth used by both MLX and Candle: select the authoritative `model.*` family and fail closed
//! when that family is absent, sharded where it must be singular, or internally inconsistent.

use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

use thiserror::Error;

const CLIP_L: &str = "text_encoder";
const CLIP_G: &str = "text_encoder_2";
const T5: &str = "text_encoder_3";
const MASTER_FILE: &str = "model.safetensors";
const MASTER_INDEX: &str = "model.safetensors.index.json";

/// Exact authoritative artifacts for the three SD3.5 text encoders.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sd3TextEncoderArtifacts {
    pub clip_l: PathBuf,
    pub clip_g: PathBuf,
    pub t5_index: PathBuf,
    pub t5_shards: Vec<PathBuf>,
}

/// Typed failures at the snapshot identity boundary.
#[derive(Debug, Error)]
pub enum Sd3TextEncoderArtifactError {
    #[error("sd3 text encoder component `{component}` is missing at {path}")]
    MissingComponent {
        component: &'static str,
        path: PathBuf,
    },
    #[error(
        "sd3 text encoder component `{component}` has no authoritative `{expected}` at {path}"
    )]
    MissingMaster {
        component: &'static str,
        expected: &'static str,
        path: PathBuf,
    },
    #[error("sd3 CLIP component `{component}` is sharded; expected one authoritative `{MASTER_FILE}` in {path}")]
    ShardedClip {
        component: &'static str,
        path: PathBuf,
    },
    #[error("sd3 text encoder component `{component}` has ambiguous master artifact `{artifact}`")]
    AmbiguousMaster {
        component: &'static str,
        artifact: PathBuf,
    },
    #[error("sd3 T5 index {path} is invalid: {reason}")]
    InvalidT5Index { path: PathBuf, reason: String },
    #[error("sd3 T5 index references missing shard {path}")]
    MissingT5Shard { path: PathBuf },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Resolve the same master encoder family for every SD3.5 route and backend.
pub fn resolve_sd3_text_encoder_artifacts(
    root: &Path,
) -> Result<Sd3TextEncoderArtifacts, Sd3TextEncoderArtifactError> {
    let clip_l = resolve_clip(root, CLIP_L)?;
    let clip_g = resolve_clip(root, CLIP_G)?;
    let (t5_index, t5_shards) = resolve_t5(root)?;
    Ok(Sd3TextEncoderArtifacts {
        clip_l,
        clip_g,
        t5_index,
        t5_shards,
    })
}

fn component_dir(
    root: &Path,
    component: &'static str,
) -> Result<PathBuf, Sd3TextEncoderArtifactError> {
    let dir = root.join(component);
    if !dir.is_dir() {
        return Err(Sd3TextEncoderArtifactError::MissingComponent {
            component,
            path: dir,
        });
    }
    Ok(dir)
}

fn resolve_clip(
    root: &Path,
    component: &'static str,
) -> Result<PathBuf, Sd3TextEncoderArtifactError> {
    let dir = component_dir(root, component)?;
    let master = dir.join(MASTER_FILE);
    let mut sharded = false;
    for entry in std::fs::read_dir(&dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name == MASTER_FILE || name == "model.fp16.safetensors" {
            continue;
        }
        if name == MASTER_INDEX || parse_master_shard(name).is_some() {
            sharded = true;
            continue;
        }
        if name.ends_with(".safetensors") || name.contains("safetensors.index") {
            return Err(Sd3TextEncoderArtifactError::AmbiguousMaster {
                component,
                artifact: path,
            });
        }
    }
    if sharded {
        return Err(Sd3TextEncoderArtifactError::ShardedClip {
            component,
            path: dir,
        });
    }
    if !master.is_file() {
        return Err(Sd3TextEncoderArtifactError::MissingMaster {
            component,
            expected: MASTER_FILE,
            path: dir,
        });
    }
    Ok(master)
}

fn resolve_t5(root: &Path) -> Result<(PathBuf, Vec<PathBuf>), Sd3TextEncoderArtifactError> {
    let dir = component_dir(root, T5)?;
    let index_path = dir.join(MASTER_INDEX);
    if !index_path.is_file() {
        return Err(Sd3TextEncoderArtifactError::MissingMaster {
            component: T5,
            expected: MASTER_INDEX,
            path: dir,
        });
    }
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)
        .map_err(|error| invalid_index(&index_path, format!("JSON parse failed: {error}")))?;
    let weight_map = index
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| invalid_index(&index_path, "missing object `weight_map`"))?;
    if weight_map.is_empty() {
        return Err(invalid_index(
            &index_path,
            "`weight_map` references no shards",
        ));
    }

    let mut names = BTreeSet::new();
    let mut total = None;
    for value in weight_map.values() {
        let name = value.as_str().ok_or_else(|| {
            invalid_index(&index_path, "every `weight_map` value must be a string")
        })?;
        let path = Path::new(name);
        if path.components().count() != 1
            || !matches!(path.components().next(), Some(Component::Normal(_)))
        {
            return Err(invalid_index(
                &index_path,
                format!("unsafe or nested shard path `{name}`"),
            ));
        }
        let (sequence, shard_total) = parse_master_shard(name).ok_or_else(|| {
            invalid_index(&index_path, format!("non-master shard filename `{name}`"))
        })?;
        if sequence == 0 || sequence > shard_total {
            return Err(invalid_index(
                &index_path,
                format!("invalid shard sequence `{name}`"),
            ));
        }
        match total {
            Some(expected) if expected != shard_total => {
                return Err(invalid_index(
                    &index_path,
                    format!("inconsistent shard total in `{name}`"),
                ));
            }
            None => total = Some(shard_total),
            _ => {}
        }
        names.insert(name.to_owned());
    }
    let expected_total = total.expect("non-empty weight map establishes a shard total");
    let observed_sequences: BTreeSet<usize> = names
        .iter()
        .map(|name| parse_master_shard(name).expect("validated shard name").0)
        .collect();
    let expected_sequences: BTreeSet<usize> = (1..=expected_total).collect();
    if observed_sequences != expected_sequences || names.len() != expected_total {
        return Err(invalid_index(
            &index_path,
            format!(
                "incomplete master shard family: expected 1..={expected_total}, found {observed_sequences:?}"
            ),
        ));
    }

    for entry in std::fs::read_dir(&dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let is_recognized_fp16 = (name.starts_with("model.fp16-")
            && name.ends_with(".safetensors"))
            || name == "model.safetensors.index.fp16.json";
        if is_recognized_fp16 || name == MASTER_INDEX {
            continue;
        }
        if parse_master_shard(name).is_some() {
            if names.contains(name) {
                continue;
            }
            return Err(Sd3TextEncoderArtifactError::AmbiguousMaster {
                component: T5,
                artifact: path,
            });
        }
        if name == MASTER_FILE {
            return Err(Sd3TextEncoderArtifactError::AmbiguousMaster {
                component: T5,
                artifact: path,
            });
        }
        if name.ends_with(".safetensors") || name.contains("safetensors.index") {
            return Err(Sd3TextEncoderArtifactError::AmbiguousMaster {
                component: T5,
                artifact: path,
            });
        }
    }

    let shards = names
        .into_iter()
        .map(|name| dir.join(name))
        .collect::<Vec<_>>();
    for path in &shards {
        if !path.is_file() {
            return Err(Sd3TextEncoderArtifactError::MissingT5Shard { path: path.clone() });
        }
    }
    Ok((index_path, shards))
}

fn invalid_index(path: &Path, reason: impl Into<String>) -> Sd3TextEncoderArtifactError {
    Sd3TextEncoderArtifactError::InvalidT5Index {
        path: path.to_owned(),
        reason: reason.into(),
    }
}

fn parse_master_shard(name: &str) -> Option<(usize, usize)> {
    let body = name.strip_prefix("model-")?.strip_suffix(".safetensors")?;
    let (sequence, total) = body.split_once("-of-")?;
    if sequence.len() != 5
        || total.len() != 5
        || !sequence.bytes().all(|byte| byte.is_ascii_digit())
        || !total.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    Some((sequence.parse().ok()?, total.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, contents: impl AsRef<[u8]>) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, contents).unwrap();
    }

    fn stock_fixture(root: &Path) {
        for component in [CLIP_L, CLIP_G] {
            write(&root.join(component).join(MASTER_FILE), []);
            write(&root.join(component).join("model.fp16.safetensors"), []);
        }
        let t5 = root.join(T5);
        write(
            &t5.join(MASTER_INDEX),
            br#"{"weight_map":{"a":"model-00002-of-00002.safetensors","b":"model-00001-of-00002.safetensors"}}"#,
        );
        write(&t5.join("model-00001-of-00002.safetensors"), []);
        write(&t5.join("model-00002-of-00002.safetensors"), []);
        write(&t5.join("model.fp16-00001-of-00002.safetensors"), []);
        write(&t5.join("model.fp16-00002-of-00002.safetensors"), []);
        write(&t5.join("model.safetensors.index.fp16.json"), b"{}");
    }

    #[test]
    fn stock_side_by_side_layout_selects_only_authoritative_master_family() {
        let root = tempfile::tempdir().unwrap();
        stock_fixture(root.path());

        let artifacts = resolve_sd3_text_encoder_artifacts(root.path()).unwrap();
        assert_eq!(artifacts.clip_l, root.path().join(CLIP_L).join(MASTER_FILE));
        assert_eq!(artifacts.clip_g, root.path().join(CLIP_G).join(MASTER_FILE));
        assert_eq!(artifacts.t5_index, root.path().join(T5).join(MASTER_INDEX));
        assert_eq!(
            artifacts.t5_shards,
            [
                root.path()
                    .join(T5)
                    .join("model-00001-of-00002.safetensors"),
                root.path()
                    .join(T5)
                    .join("model-00002-of-00002.safetensors"),
            ]
        );
    }

    #[test]
    fn fp16_only_clip_does_not_silently_change_encoder_identity() {
        let root = tempfile::tempdir().unwrap();
        stock_fixture(root.path());
        std::fs::remove_file(root.path().join(CLIP_L).join(MASTER_FILE)).unwrap();

        assert!(matches!(
            resolve_sd3_text_encoder_artifacts(root.path()),
            Err(Sd3TextEncoderArtifactError::MissingMaster {
                component: CLIP_L,
                ..
            })
        ));
    }

    #[test]
    fn sharded_clip_is_a_typed_failure_even_when_master_is_also_present() {
        let root = tempfile::tempdir().unwrap();
        stock_fixture(root.path());
        write(
            &root
                .path()
                .join(CLIP_G)
                .join("model-00001-of-00002.safetensors"),
            [],
        );

        assert!(matches!(
            resolve_sd3_text_encoder_artifacts(root.path()),
            Err(Sd3TextEncoderArtifactError::ShardedClip {
                component: CLIP_G,
                ..
            })
        ));
    }

    #[test]
    fn unindexed_master_t5_shard_is_rejected_as_ambiguous() {
        let root = tempfile::tempdir().unwrap();
        stock_fixture(root.path());
        write(
            &root
                .path()
                .join(T5)
                .join("model-00003-of-00003.safetensors"),
            [],
        );

        assert!(matches!(
            resolve_sd3_text_encoder_artifacts(root.path()),
            Err(Sd3TextEncoderArtifactError::AmbiguousMaster { component: T5, .. })
        ));
    }

    #[test]
    fn unsafe_or_incomplete_t5_index_is_a_typed_failure() {
        for index in [
            br#"{"weight_map":{"a":"../model-00001-of-00001.safetensors"}}"#.as_slice(),
            br#"{"weight_map":{"a":"model-00001-of-00002.safetensors"}}"#.as_slice(),
        ] {
            let root = tempfile::tempdir().unwrap();
            stock_fixture(root.path());
            write(&root.path().join(T5).join(MASTER_INDEX), index);
            assert!(matches!(
                resolve_sd3_text_encoder_artifacts(root.path()),
                Err(Sd3TextEncoderArtifactError::InvalidT5Index { .. })
            ));
        }
    }

    #[test]
    fn missing_indexed_t5_shard_is_typed() {
        let root = tempfile::tempdir().unwrap();
        stock_fixture(root.path());
        std::fs::remove_file(
            root.path()
                .join(T5)
                .join("model-00002-of-00002.safetensors"),
        )
        .unwrap();

        assert!(matches!(
            resolve_sd3_text_encoder_artifacts(root.path()),
            Err(Sd3TextEncoderArtifactError::MissingT5Shard { .. })
        ));
    }
}
