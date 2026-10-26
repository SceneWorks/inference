//! Header-authoritative native FLUX3 checkpoint metadata, without tensor payload reads.
//!
//! Only the frozen single `model.safetensors` layout is supported. Native shard indices are
//! refused consistently at identity and load boundaries. Never
//! recursively merge a package tree: it also contains AE, TE, diffusers and publisher variants.
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::{Error, Result, SafetensorsTensorHeader};

use super::{Flux3Config, Flux3MetadataConfig, DIT_WEIGHTS_FILE};

const INDEX_FILE: &str = "model.safetensors.index.json";
const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;
const FLOW_MAP_KEY: &str = "ref_time_in.in_layer.weight";
const FLOW_MAP_OUT_KEY: &str = "ref_time_in.out_layer.weight";

fn unsupported(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 native checkpoint: {detail}"))
}

fn read_header(path: &Path) -> Result<(Vec<SafetensorsTensorHeader>, u64)> {
    let mut file = File::open(path)
        .map_err(|e| unsupported(format!("cannot open transformer {}: {e}", path.display())))?;
    let file_len = file.metadata()?.len();
    read_header_from(&mut file, file_len, path)
}

fn read_header_from(
    file: &mut (impl Read + Seek),
    file_len: u64,
    path: &Path,
) -> Result<(Vec<SafetensorsTensorHeader>, u64)> {
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|e| unsupported(format!("{}: missing header prefix: {e}", path.display())))?;
    let length = u64::from_le_bytes(prefix);
    if length > MAX_HEADER_BYTES {
        return Err(unsupported(format!(
            "{}: header length {length} exceeds the 16 MiB bound",
            path.display()
        )));
    }
    // The SAME opened handle supplies the header and its actual length. The shared reader
    // validates dtype/shape/offset sizes and contiguous payload coverage using metadata only.
    // It stops at the end of the JSON header: no tensor payload byte is read.
    file.seek(SeekFrom::Start(0))?;
    let layout = crate::weightsmeta::safetensors_reader_tensor_locations(file, file_len, path)?;
    check_unique_header_names(&layout.header_json, path)?;
    let headers = layout.tensors.into_iter().map(|t| t.header).collect();
    Ok((headers, length))
}

fn check_unique_header_names(bytes: &[u8], path: &Path) -> Result<()> {
    struct UniqueNames;
    impl<'de> serde::de::Visitor<'de> for UniqueNames {
        type Value = ();
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a header with unique tensor names")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> std::result::Result<(), A::Error> {
            let mut names = BTreeSet::new();
            while let Some(name) = map.next_key::<String>()? {
                if !names.insert(name) {
                    return Err(serde::de::Error::custom(
                        "duplicate safetensors header name",
                    ));
                }
                map.next_value::<serde::de::IgnoredAny>()?;
            }
            Ok(())
        }
    }
    let mut parser = serde_json::Deserializer::from_slice(bytes);
    serde::de::Deserializer::deserialize_map(&mut parser, UniqueNames)
        .map_err(|e| unsupported(format!("{}: {e}", path.display())))
}

/// Read and structurally validate ONLY the frozen single native transformer header.
/// Every data offset must fit the ACTUAL file size; truncated/header-only fake files fail.
/// Index-only/sharded native checkpoints have no frozen provider loading contract and are
/// refused here, so identity and provider load cannot disagree about layout support.
/// Native loaders/converters separately check the full format-specific tensor geometry.
pub fn checkpoint_tensor_headers(dir: &Path) -> Result<Vec<SafetensorsTensorHeader>> {
    if dir.join(INDEX_FILE).is_file() {
        return Err(unsupported("unsupported checkpoint layout: frozen FLUX3 requires a single model.safetensors, not native transformer shards"));
    }
    let direct = dir.join(DIT_WEIGHTS_FILE);
    if !direct.is_file() {
        return Err(unsupported(format!(
            "{}: missing transformer {DIT_WEIGHTS_FILE}",
            dir.display()
        )));
    }
    let (headers, _) = read_header(&direct)?;
    validate_native_names(&headers)?;
    Ok(headers)
}

fn validate_native_names(headers: &[SafetensorsTensorHeader]) -> Result<()> {
    let mut names = BTreeSet::new();
    for header in headers {
        if header.name.is_empty() || header.name.contains('\0') || !names.insert(&header.name) {
            return Err(unsupported(
                "invalid or duplicate native transformer tensor name",
            ));
        }
    }
    // These protected native names exist in dense, publisher and converted formats. In
    // particular, do not misclassify diffusers' target_timestep_embedder layout as base.
    for anchor in ["emb_in.image.weight", "time_in.in_layer.weight"] {
        if !headers
            .iter()
            .any(|h| h.name == anchor && h.shape.len() == 2 && h.shape.iter().all(|d| *d > 0))
        {
            return Err(unsupported(format!("missing valid native FLUX3 anchor {anchor}; diffusers layouts are not native checkpoints")));
        }
    }
    let has_in = headers.iter().any(|h| h.name == FLOW_MAP_KEY);
    let has_out = headers.iter().any(|h| h.name == FLOW_MAP_OUT_KEY);
    if has_in != has_out || (!has_in && headers.iter().any(|h| h.name.starts_with("ref_time_in.")))
    {
        return Err(unsupported(
            "incomplete ref_time_in checkpoint: both native embedder weights are required",
        ));
    }
    for key in [FLOW_MAP_KEY, FLOW_MAP_OUT_KEY] {
        if let Some(header) = headers.iter().find(|h| h.name == key) {
            if header.shape.len() != 2 || header.shape.contains(&0) {
                return Err(unsupported(format!(
                    "invalid native matrix shape for {key}"
                )));
            }
        }
    }
    Ok(())
}

/// Resolve checkpoint mode from native transformer HEADER names, exactly as frozen BFL
/// `checkpoint.is_flow_map` does. Optional descriptive config metadata must agree if present.
/// Missing files never mean base. Native shard indices are explicitly unsupported; AE/TE,
/// nested publisher variants and diffusers files cannot supply or hide the marker.
///
/// Performs bounded metadata reads only, never tensor payload reads or provider/device work.
/// Use [`Flux3MetadataConfig::from_dir`] instead for pre-download planning, then bind an explicit
/// catalog model with `for_model`; that planning bind is not checkpoint identity.
pub fn resolve_checkpoint(dir: &Path) -> Result<Flux3Config> {
    let metadata = Flux3MetadataConfig::from_dir(dir)?;
    let headers = checkpoint_tensor_headers(dir)?;
    let flow_map = headers.iter().any(|header| header.name == FLOW_MAP_KEY);
    metadata.resolve_flow_map(flow_map, "checkpoint header")
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
pub(crate) mod tests;
