//! The audio-lane snapshot preparers' shared skeleton (sc-27264 / sc-27265 / sc-27343).
//!
//! Every audio provider's preparer is a **validated passthrough**: a weightless probe recognizes
//! the snapshot, a requested quantization is a typed `Unsupported` (the checkpoints ship
//! dense-only), and the report counts the tensors the snapshot holds. This module owns that
//! skeleton ([`probe_json_config`], [`passthrough_report`]); each provider supplies only its probe
//! predicate and its component list.
//!
//! Tensors are counted from the safetensors **header only** through gen-core's bounded reader
//! ([`gen_core::weightsmeta::safetensors_file_tensor_locations`]: 100 MB header cap,
//! overflow-checked length prefix, validated entries) — never by reading a multi-GB shard into
//! memory.

use std::path::Path;

use gen_core::core_llm::{
    Error as CoreError, ModelFormat, PrepareReport, PrepareSpec, Result as CoreResult,
};
use gen_core::safetensors_shards::{resolve_safetensors_shards, snapshot_shard_roots};
use gen_core::weightsmeta::safetensors_file_tensor_locations;

fn prepare_error(e: impl std::fmt::Display) -> CoreError {
    CoreError::Msg(format!("prepare: {e}"))
}

/// Weightless probe skeleton: `dir` is a directory holding the regular file `weights` (a path
/// relative to `dir`), and its `config` JSON file parses and is approved by `accept`. Reads only
/// the config file, never a weight shard.
pub fn probe_json_config(
    dir: &Path,
    weights: &str,
    config: &str,
    accept: impl FnOnce(&serde_json::Value) -> bool,
) -> bool {
    if !dir.is_dir() || !dir.join(weights).is_file() {
        return false;
    }
    std::fs::read_to_string(dir.join(config))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .is_some_and(|value| accept(&value))
}

/// Whether `config[key]` is the JSON string `expected`.
pub fn json_str_is(config: &serde_json::Value, key: &str, expected: &str) -> bool {
    config.get(key).and_then(serde_json::Value::as_str) == Some(expected)
}

/// Whether the HF `config["architectures"]` array lists `name`.
pub fn lists_architecture(config: &serde_json::Value, name: &str) -> bool {
    config
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|names| names.iter().any(|n| n.as_str() == Some(name)))
}

/// The shared validated-passthrough preparer for `model` snapshots: a snapshot `is_snapshot`
/// rejects and any requested quantization are typed `Unsupported`; otherwise `count_tensors`
/// reports what the snapshot holds and the snapshot itself is the output (`passthrough: true`).
///
/// `input_format` is `Safetensors` for every audio snapshot: core-llm's own `detect_format` calls a
/// `config.json`-bearing snapshot dir Safetensors (the HF-snapshot shape), and a pickle checkpoint
/// rides inside that shape.
pub fn passthrough_report(
    spec: &PrepareSpec,
    model: &str,
    is_snapshot: impl FnOnce(&Path) -> bool,
    count_tensors: impl FnOnce(&Path) -> CoreResult<usize>,
) -> CoreResult<PrepareReport> {
    if !is_snapshot(&spec.source) {
        return Err(CoreError::Unsupported(format!(
            "prepare: {} is not a {model} snapshot",
            spec.source.display()
        )));
    }
    if let Some(q) = spec.quantize {
        return Err(CoreError::Unsupported(format!(
            "prepare: {model} snapshots have no {q:?} form — the checkpoint ships dense-only"
        )));
    }
    Ok(PrepareReport {
        input_format: ModelFormat::Safetensors,
        quantized: None,
        out_dir: spec.source.clone(),
        num_tensors: count_tensors(&spec.source)?,
        passthrough: true,
    })
}

/// Count the tensors of one safetensors file from its header only (`__metadata__` is not a
/// tensor). No tensor storage is read.
pub fn safetensors_tensor_count(path: &Path) -> CoreResult<usize> {
    Ok(safetensors_file_tensor_locations(path)
        .map_err(prepare_error)?
        .tensors
        .len())
}

/// Count the tensors of a required `{stem}` component in `dir`: every shard selected by
/// `{stem}.safetensors.index.json`, or the single-file `{stem}.safetensors` fallback. The whole
/// shard set is resolved (and confined to the snapshot's authorized roots) before any header is
/// read. A missing component is an error.
pub fn component_tensor_count(dir: &Path, stem: &str) -> CoreResult<usize> {
    let roots = snapshot_shard_roots(dir).map_err(prepare_error)?;
    let shards = resolve_safetensors_shards(dir, stem, &roots).map_err(prepare_error)?;
    shards
        .iter()
        .map(|shard| safetensors_tensor_count(shard))
        .sum()
}

/// [`component_tensor_count`] for an optional component: `0` when neither
/// `{stem}.safetensors.index.json` nor `{stem}.safetensors` is present in `dir`.
pub fn optional_component_tensor_count(dir: &Path, stem: &str) -> CoreResult<usize> {
    let present = |path: &Path| match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(prepare_error(format!(
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
    use crate::testkit::tiny_safetensors;

    fn spec(dir: &Path, quantize: Option<gen_core::core_llm::Quantize>) -> PrepareSpec {
        PrepareSpec {
            source: dir.to_path_buf(),
            out_dir: dir.join("out"),
            quantize,
        }
    }

    #[test]
    fn json_probe_needs_the_weights_file_and_an_accepted_config() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let accept = |v: &serde_json::Value| lists_architecture(v, "M");
        assert!(!probe_json_config(
            &dir.join("absent"),
            "w.bin",
            "config.json",
            accept
        ));
        assert!(!probe_json_config(dir, "w.bin", "config.json", accept));
        std::fs::write(dir.join("w.bin"), b"").unwrap();
        assert!(!probe_json_config(dir, "w.bin", "config.json", accept));
        std::fs::write(dir.join("config.json"), "{not json").unwrap();
        assert!(!probe_json_config(dir, "w.bin", "config.json", accept));
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["X"],"model_type":"m"}"#,
        )
        .unwrap();
        assert!(!probe_json_config(dir, "w.bin", "config.json", accept));
        assert!(probe_json_config(dir, "w.bin", "config.json", |v| {
            json_str_is(v, "model_type", "m")
        }));
        assert!(!probe_json_config(dir, "w.bin", "config.json", |v| {
            json_str_is(v, "architectures", "X")
        }));
        std::fs::write(dir.join("config.json"), r#"{"architectures":["X","M"]}"#).unwrap();
        assert!(probe_json_config(dir, "w.bin", "config.json", accept));
    }

    #[test]
    fn passthrough_report_types_its_refusals_and_counts_on_success() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let count = |_: &Path| Ok(7);
        assert!(matches!(
            passthrough_report(&spec(dir, None), "M", |_| false, count),
            Err(CoreError::Unsupported(_))
        ));
        assert!(matches!(
            passthrough_report(
                &spec(dir, Some(gen_core::core_llm::Quantize::Q4)),
                "M",
                |_| true,
                |_| panic!("a refused quantization must not count tensors")
            ),
            Err(CoreError::Unsupported(_))
        ));
        let report = passthrough_report(&spec(dir, None), "M", |_| true, count).unwrap();
        assert_eq!(report.num_tensors, 7);
        assert!(report.passthrough);
        assert_eq!(report.out_dir, dir);
        assert!(matches!(report.input_format, ModelFormat::Safetensors));
        let failing = passthrough_report(
            &spec(dir, None),
            "M",
            |_| true,
            |_| Err(CoreError::Msg("boom".into())),
        );
        assert!(matches!(failing, Err(CoreError::Msg(_))));
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
