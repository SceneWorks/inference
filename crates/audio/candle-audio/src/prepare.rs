//! Shared, header-only safetensors accounting for the audio-lane snapshot preparers
//! (sc-27264 / sc-27265 / sc-27343).
//!
//! The passthrough preparers (ACE-Step, Chatterbox, MOSS-SFX, MOSS-TTSD, MOSS-TTS-Realtime) report
//! how many tensors a snapshot holds. They count from the safetensors **header only** through
//! gen-core's bounded reader ([`gen_core::weightsmeta::safetensors_file_tensor_locations`]: 100 MB
//! header cap, overflow-checked length prefix, validated entries) — never by reading a multi-GB
//! shard into memory. Each provider keeps its own probe predicate and component list.

use std::path::Path;

use gen_core::safetensors_shards::{resolve_safetensors_shards, snapshot_shard_roots};
use gen_core::weightsmeta::safetensors_file_tensor_locations;
use gen_core::Result;

/// Count the tensors of one safetensors file from its header only (`__metadata__` is not a
/// tensor). No tensor storage is read.
pub fn safetensors_tensor_count(path: &Path) -> Result<usize> {
    Ok(safetensors_file_tensor_locations(path)?.tensors.len())
}

/// Count the tensors of a required `{stem}` component in `dir`: every shard selected by
/// `{stem}.safetensors.index.json`, or the single-file `{stem}.safetensors` fallback. The whole
/// shard set is resolved (and confined to the snapshot's authorized roots) before any header is
/// read. A missing component is an error.
pub fn component_tensor_count(dir: &Path, stem: &str) -> Result<usize> {
    let roots = snapshot_shard_roots(dir)?;
    let shards = resolve_safetensors_shards(dir, stem, &roots)?;
    shards
        .iter()
        .map(|shard| safetensors_tensor_count(shard))
        .sum()
}

/// [`component_tensor_count`] for an optional component: `0` when neither
/// `{stem}.safetensors.index.json` nor `{stem}.safetensors` is present in `dir`.
pub fn optional_component_tensor_count(dir: &Path, stem: &str) -> Result<usize> {
    let present = |path: &Path| match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(gen_core::Error::Msg(format!(
            "inspect {}: {error}",
            path.display()
        ))),
    };
    let index = dir.join(format!("{stem}.safetensors.index.json"));
    let single = dir.join(format!("{stem}.safetensors"));
    if !present(&index)? && !present(&single)? {
        return Ok(0);
    }
    component_tensor_count(dir, stem)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal valid safetensors byte blob with zero-size F32 tensors.
    fn tiny_safetensors(names: &[&str]) -> Vec<u8> {
        let entries: Vec<String> = names
            .iter()
            .map(|n| format!(r#""{n}": {{"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}}"#))
            .collect();
        let header = format!("{{{}}}", entries.join(", "));
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out
    }

    #[test]
    fn safetensors_header_count_is_metadata_free() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("x.safetensors");
        std::fs::write(&p, tiny_safetensors(&["a", "b", "__metadata__"])).unwrap();
        assert_eq!(safetensors_tensor_count(&p).unwrap(), 2);
    }

    #[test]
    fn oversized_or_truncated_headers_fail_without_reading_the_payload() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("x.safetensors");
        // A length prefix past usize::MAX - 8 (the old `8 + n` overflow) and one past the cap.
        for len in [u64::MAX, u64::MAX - 7, 100_000_001] {
            let mut bytes = len.to_le_bytes().to_vec();
            bytes.extend_from_slice(b"{}");
            std::fs::write(&p, bytes).unwrap();
            assert!(safetensors_tensor_count(&p).is_err(), "{len}");
        }
        // A header that claims more bytes than the file holds.
        let mut bytes = 64_u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        std::fs::write(&p, bytes).unwrap();
        assert!(safetensors_tensor_count(&p).is_err());
        // Shorter than the 8-byte length prefix.
        std::fs::write(&p, [1_u8, 2, 3]).unwrap();
        assert!(safetensors_tensor_count(&p).is_err());
    }

    #[test]
    fn optional_component_is_zero_when_absent() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(
            optional_component_tensor_count(tmp.path(), "model").unwrap(),
            0
        );
        assert!(component_tensor_count(tmp.path(), "model").is_err());
        std::fs::write(
            tmp.path().join("model.safetensors"),
            tiny_safetensors(&["a", "b"]),
        )
        .unwrap();
        assert_eq!(
            optional_component_tensor_count(tmp.path(), "model").unwrap(),
            2
        );
    }

    #[test]
    fn preparer_rejects_index_paths_and_nonregular_shards() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("text_encoder");
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), tiny_safetensors(&["x"])).unwrap();
        let index = dir.join("model.safetensors.index.json");
        std::fs::write(&index, r#"{"weight_map":{"x":"model.safetensors"}}"#).unwrap();
        assert_eq!(component_tensor_count(&dir, "model").unwrap(), 1);
        assert_eq!(optional_component_tensor_count(&dir, "model").unwrap(), 1);
        for name in [
            "../outside.safetensors",
            "/tmp/outside.safetensors",
            "missing.safetensors",
        ] {
            std::fs::write(&index, format!(r#"{{"weight_map":{{"x":"{name}"}}}}"#)).unwrap();
            assert!(component_tensor_count(&dir, "model").is_err(), "{name}");
            assert!(
                optional_component_tensor_count(&dir, "model").is_err(),
                "{name}"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let outside = temp.path().join("outside.safetensors");
            std::fs::write(&outside, tiny_safetensors(&["x"])).unwrap();
            symlink(&outside, dir.join("external.safetensors")).unwrap();
            std::fs::write(&index, r#"{"weight_map":{"x":"external.safetensors"}}"#).unwrap();
            assert!(component_tensor_count(&dir, "model").is_err());
            assert!(optional_component_tensor_count(&dir, "model").is_err());
            let fifo = dir.join("pipe.safetensors");
            assert!(std::process::Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success());
            std::fs::write(&index, r#"{"weight_map":{"x":"pipe.safetensors"}}"#).unwrap();
            assert!(component_tensor_count(&dir, "model")
                .unwrap_err()
                .to_string()
                .contains("not a regular file"));
        }
    }
}
