//! The audio lane's one audited `unsafe` safetensors memory-map surface (sc-27259 / sc-27358).
//!
//! Every candle audio provider maps its caller-provisioned weight files through these helpers
//! instead of restating its own `unsafe { MmapedSafetensors::new(..) }` /
//! `unsafe { VarBuilder::from_mmaped_safetensors(..) }` block and its own safety justification.
//! It also holds the shared F32 tensor loaders ([`load_f32`], [`load_f32_map`]) and the lane's one
//! bounded raw-header reader ([`read_safetensors_header`], sc-27343).
//!
//! # The contract
//!
//! Inference receives every model component as a **caller-provisioned local path**
//! (`WeightsSource::Dir`/`File`, epic 13657); user-supplied models at arbitrary paths must load, so
//! nothing here can promise the file is a pinned, immutable snapshot. Memory-mapping is still the
//! right mechanism for multi-GB weights (reading the whole file would double resident memory), so
//! the supported contract is stated instead of assumed:
//!
//! - the mapping is **read-only**; nothing in this process writes or truncates a mapped file;
//! - the caller must not truncate or rewrite the files while they are mapped (for a `VarBuilder`,
//!   until the builder and every tensor borrowing it are dropped; tensors loaded to an owned
//!   device buffer stop borrowing the map). External mutation during that window is **outside the
//!   supported contract**: the process may observe torn data or be killed by `SIGBUS`.
//!
//! Small, request-supplied files (e.g. Stable Audio 3 LoRA/DoRA adapters) should not be mapped at
//! all — read them into an owned buffer (`BufferedSafetensors`) so a concurrent rewrite cannot
//! fault the process (sc-27352).

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

/// Memory-map one caller-provisioned safetensors file. See the [module contract](self).
pub fn mmap_safetensors(path: impl AsRef<Path>) -> candle_core::Result<MmapedSafetensors> {
    // SAFETY: read-only map of a caller-provisioned weight file; external mutation while mapped is
    // outside the supported contract documented on this module.
    unsafe { MmapedSafetensors::new(path) }
}

/// Memory-map caller-provisioned safetensors shard(s) into a `'static` [`VarBuilder`] that
/// materializes tensors at `dtype` on `device`. See the [module contract](self).
pub fn mmap_var_builder<P: AsRef<Path>>(
    paths: &[P],
    dtype: DType,
    device: &Device,
) -> candle_core::Result<VarBuilder<'static>> {
    // SAFETY: read-only map of caller-provisioned weight files; external mutation while mapped is
    // outside the supported contract documented on this module.
    unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device) }
}

/// Load tensor `name` of `st` onto `device` as F32 — the per-tensor step of the audio lane's
/// name-keyed checkpoint loaders (sc-27259).
pub fn load_f32(
    st: &MmapedSafetensors,
    name: &str,
    device: &Device,
) -> candle_core::Result<Tensor> {
    st.load(name, device)?.to_dtype(DType::F32)
}

/// Every tensor of `st` whose name `keep` accepts, loaded onto `device` as F32 and keyed by its full
/// name (sc-27259). A failing tensor is named in the error.
pub fn load_f32_map(
    st: &MmapedSafetensors,
    device: &Device,
    keep: impl Fn(&str) -> bool,
) -> candle_core::Result<HashMap<String, Tensor>> {
    let mut out = HashMap::new();
    for (name, _) in st.tensors() {
        if !keep(&name) {
            continue;
        }
        let tensor = load_f32(st, &name, device)
            .map_err(|e| candle_core::Error::Msg(format!("tensor {name:?}: {e}")))?;
        out.insert(name, tensor);
    }
    Ok(out)
}

/// Upper bound on a safetensors JSON header — the same 100 MB cap as gen-core's `weightsmeta`
/// reader and the `safetensors` crate itself.
pub const MAX_SAFETENSORS_HEADER_BYTES: u64 = 100_000_000;

/// Why [`read_safetensors_header`] refused a file.
#[derive(Debug, thiserror::Error)]
pub enum SafetensorsHeaderError {
    /// The file could not be opened or its length read.
    #[error("{}: {source}", path.display())]
    Io {
        /// The file.
        path: PathBuf,
        /// The underlying failure.
        source: std::io::Error,
    },
    /// The file opened but its length prefix or JSON header is not a valid safetensors header.
    #[error("{}: {detail}", path.display())]
    Invalid {
        /// The file.
        path: PathBuf,
        /// What is wrong with the header.
        detail: String,
    },
}

/// Read the raw JSON header object (`__metadata__` included) of one caller-supplied safetensors
/// file, reading only the 8-byte length prefix and the header itself (sc-27343).
///
/// The length prefix is bounded by [`MAX_SAFETENSORS_HEADER_BYTES`] and by the file length before
/// anything is allocated, so a corrupt or hostile prefix fails cleanly instead of sizing a large
/// allocation. This is the audio lane's one hand-rolled header reader; consumers that need
/// validated tensor locations use `gen_core::weightsmeta` instead.
pub fn read_safetensors_header(
    path: &Path,
) -> Result<serde_json::Map<String, serde_json::Value>, SafetensorsHeaderError> {
    let io = |source| SafetensorsHeaderError::Io {
        path: path.to_path_buf(),
        source,
    };
    let invalid = |detail: String| SafetensorsHeaderError::Invalid {
        path: path.to_path_buf(),
        detail,
    };
    let mut file = File::open(path).map_err(io)?;
    let file_len = file.metadata().map_err(io)?.len();
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|e| invalid(format!("cannot read the safetensors header length: {e}")))?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(invalid(format!(
            "safetensors header length {header_len} exceeds the \
             {MAX_SAFETENSORS_HEADER_BYTES}-byte maximum"
        )));
    }
    if header_len > file_len.saturating_sub(8) {
        return Err(invalid(format!(
            "safetensors header length {header_len} does not fit a {file_len}-byte file"
        )));
    }
    // Bounded by the 100 MB cap above, so this fits `usize` on every supported target.
    let mut header = vec![0_u8; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|e| invalid(format!("cannot read the safetensors header: {e}")))?;
    match serde_json::from_slice(&header) {
        Ok(serde_json::Value::Object(map)) => Ok(map),
        Ok(_) => Err(invalid("safetensors header is not a JSON object".into())),
        Err(e) => Err(invalid(format!("safetensors header JSON: {e}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_st(path: &Path, name: &str, values: &[f32]) {
        let mut map = std::collections::HashMap::new();
        map.insert(name.to_string(), Tensor::new(values, &Device::Cpu).unwrap());
        candle_core::safetensors::save(&map, path).unwrap();
    }

    #[test]
    fn mmap_helpers_read_tensors_and_report_missing_files() {
        let tmp = tempfile::tempdir().unwrap();
        let a = tmp.path().join("a.safetensors");
        let b = tmp.path().join("b.safetensors");
        write_st(&a, "x", &[1.0, 2.0]);
        write_st(&b, "y", &[3.0]);

        let st = mmap_safetensors(&a).unwrap();
        assert_eq!(st.get("x").unwrap().shape(), &[2]);

        let vb = mmap_var_builder(&[&a, &b], DType::F32, &Device::Cpu).unwrap();
        let y = vb.get(1, "y").unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(y, vec![3.0]);
        assert!(vb.contains_tensor("x"));

        assert!(mmap_safetensors(tmp.path().join("missing.safetensors")).is_err());
        assert!(mmap_var_builder(&[tmp.path().join("missing")], DType::F32, &Device::Cpu).is_err());
    }

    #[test]
    fn f32_loaders_convert_filter_and_name_failures() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("m.safetensors");
        let mut map = std::collections::HashMap::new();
        map.insert(
            "enc.w".to_string(),
            Tensor::new(&[1.0_f64, 2.0], &Device::Cpu).unwrap(),
        );
        map.insert(
            "dec.w".to_string(),
            Tensor::new(&[3.0_f32], &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&map, &path).unwrap();
        let st = mmap_safetensors(&path).unwrap();

        let enc = load_f32(&st, "enc.w", &Device::Cpu).unwrap();
        assert_eq!(enc.dtype(), DType::F32);
        assert_eq!(enc.to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
        assert!(load_f32(&st, "absent", &Device::Cpu).is_err());

        let only_enc = load_f32_map(&st, &Device::Cpu, |n| n.starts_with("enc.")).unwrap();
        assert_eq!(only_enc.keys().collect::<Vec<_>>(), vec!["enc.w"]);
        assert_eq!(only_enc["enc.w"].dtype(), DType::F32);
        assert_eq!(load_f32_map(&st, &Device::Cpu, |_| true).unwrap().len(), 2);
    }

    #[test]
    fn header_reader_bounds_the_length_prefix_before_allocating() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("h.safetensors");
        let with_prefix = |len: u64, body: &[u8]| {
            let mut bytes = len.to_le_bytes().to_vec();
            bytes.extend_from_slice(body);
            std::fs::write(&path, bytes).unwrap();
        };
        // Past the cap (the old unbounded / `8 + n` overflow shapes) and past the file end.
        for len in [u64::MAX, u64::MAX - 7, MAX_SAFETENSORS_HEADER_BYTES + 1, 64] {
            with_prefix(len, b"{}");
            assert!(
                matches!(
                    read_safetensors_header(&path),
                    Err(SafetensorsHeaderError::Invalid { .. })
                ),
                "{len}"
            );
        }
        // Shorter than the 8-byte prefix, non-object JSON, malformed JSON.
        std::fs::write(&path, [1_u8, 2, 3]).unwrap();
        assert!(read_safetensors_header(&path).is_err());
        with_prefix(2, b"[]");
        assert!(read_safetensors_header(&path)
            .unwrap_err()
            .to_string()
            .contains("not a JSON object"));
        with_prefix(2, b"{x");
        assert!(read_safetensors_header(&path).is_err());
        // A missing file is an I/O error, not a header error.
        assert!(matches!(
            read_safetensors_header(&tmp.path().join("missing")),
            Err(SafetensorsHeaderError::Io { .. })
        ));
        // A well-formed header keeps `__metadata__`.
        let header =
            br#"{"__metadata__":{"k":"v"},"t":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
        with_prefix(header.len() as u64, header);
        let map = read_safetensors_header(&path).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map["__metadata__"]["k"], "v");
    }
}
