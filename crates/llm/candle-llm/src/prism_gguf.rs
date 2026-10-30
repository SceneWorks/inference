//! Minimal strict GGUF reader for Prism's provisional PQ2_0/PTQ1_0 tensor types (142/143).

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::{Error, Result};

/// Header-declared counts only ever pre-reserve this many slots; larger tables grow as entries are
/// actually read, so a hostile count cannot force an allocation the file does not back (sc-27299).
const MAX_PRERESERVE: usize = 4096;
/// The smallest encoding of one metadata entry: key length (8) + type (4) + a 1-byte value.
const MIN_METADATA_ENTRY_BYTES: u64 = 8 + 4 + 1;
/// The smallest encoding of one tensor-info entry: name length (8) + rank (4) + one dimension (8)
/// + type (4) + offset (8).
const MIN_TENSOR_ENTRY_BYTES: u64 = 8 + 4 + 8 + 4 + 8;
/// Upper bound on one metadata string (keys, names, chat templates are kilobytes in practice).
const MAX_STRING_BYTES: usize = 16 << 20;

/// Reject a header-declared entry `count` that the remaining `file_len - pos` bytes cannot hold.
fn check_count(count: u64, min_entry: u64, file_len: u64, pos: u64, what: &str) -> Result<usize> {
    let remaining = file_len.saturating_sub(pos);
    if count > remaining / min_entry {
        return Err(Error::Config(format!(
            "GGUF {what} count {count} exceeds what the remaining {remaining} bytes can hold"
        )));
    }
    usize::try_from(count).map_err(|_| Error::Config(format!("GGUF {what} count overflow")))
}

#[derive(Clone, Debug)]
pub(crate) enum RawValue {
    U(u64),
    I(i64),
    F(f64),
    Bool(bool),
    String(String),
    Array(Vec<RawValue>),
}

impl RawValue {
    pub fn u64(&self) -> Option<u64> {
        match self {
            Self::U(v) => Some(*v),
            Self::I(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }
    pub fn f64(&self) -> Option<f64> {
        match self {
            Self::F(v) => Some(*v),
            _ => None,
        }
    }
    pub fn string(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }
    pub fn bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
    pub fn array(&self) -> Option<&[RawValue]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    pub fn exact_sign(&self) -> Option<i8> {
        match self {
            Self::U(1) | Self::I(1) | Self::F(1.0) => Some(1),
            Self::I(-1) | Self::F(-1.0) => Some(-1),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RawTensorInfo {
    /// GGUF order `[input, output, ...]`.
    pub dimensions: Vec<usize>,
    pub ggml_type: u32,
    pub offset: u64,
}

pub(crate) struct RawGguf {
    pub metadata: HashMap<String, RawValue>,
    pub tensors: HashMap<String, RawTensorInfo>,
    data_offset: u64,
    file: load_progress::reader::ProgressReader<std::fs::File>,
    progress: load_progress::reader::SourceProgress,
}

impl RawGguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let file_len = file.metadata()?.len();
        let progress = load_progress::reader::SourceProgress::new(path, Some(file_len));
        let _operation = progress.operation();
        let mut file = progress.reader(file)?;
        if u32le(&mut file)? != 0x4655_4747 {
            return Err(Error::Config("GGUF bad magic".into()));
        }
        let version = u32le(&mut file)?;
        if !(2..=3).contains(&version) {
            return Err(Error::Config(format!("GGUF version {version} unsupported")));
        }
        let tensor_count = u64le(&mut file)?;
        let metadata_count = u64le(&mut file)?;
        let pos = file.stream_position()?;
        let metadata_count = check_count(
            metadata_count,
            MIN_METADATA_ENTRY_BYTES,
            file_len,
            pos,
            "metadata",
        )?;
        let tensor_count = check_count(
            tensor_count,
            MIN_TENSOR_ENTRY_BYTES,
            file_len,
            pos,
            "tensor",
        )?;
        let mut metadata = HashMap::with_capacity(metadata_count.min(MAX_PRERESERVE));
        for _ in 0..metadata_count {
            let key = string(&mut file)?;
            let ty = u32le(&mut file)?;
            let value = value(&mut file, ty, 0)?;
            if metadata.insert(key.clone(), value).is_some() {
                return Err(Error::Config(format!("duplicate GGUF metadata {key}")));
            }
        }
        let mut tensors = HashMap::with_capacity(tensor_count.min(MAX_PRERESERVE));
        for _ in 0..tensor_count {
            let name = string(&mut file)?;
            let rank = usize::try_from(u32le(&mut file)?).unwrap();
            if rank == 0 || rank > 4 {
                return Err(Error::Config(format!(
                    "GGUF tensor {name} rank {rank} unsupported"
                )));
            }
            let mut dimensions = Vec::with_capacity(rank);
            for _ in 0..rank {
                dimensions.push(
                    usize::try_from(u64le(&mut file)?)
                        .map_err(|_| Error::Config("GGUF dimension overflow".into()))?,
                );
            }
            let ggml_type = u32le(&mut file)?;
            let offset = u64le(&mut file)?;
            if tensors
                .insert(
                    name.clone(),
                    RawTensorInfo {
                        dimensions,
                        ggml_type,
                        offset,
                    },
                )
                .is_some()
            {
                return Err(Error::Config(format!("duplicate GGUF tensor {name}")));
            }
        }
        let alignment = metadata
            .get("general.alignment")
            .and_then(RawValue::u64)
            .unwrap_or(32);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(Error::Config("invalid GGUF alignment".into()));
        }
        let pos = file.stream_position()?;
        let data_offset = pos.div_ceil(alignment) * alignment;
        crate::__model_input_result!(
            Ok(Self {
                metadata,
                tensors,
                data_offset,
                file,
                progress,
            }),
            "candle-llm::prism_gguf::RawGguf::open",
            ModelWeights,
            &[path]
        )
    }

    pub fn read_tensor(&mut self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.into()))?;
        let elems = info
            .dimensions
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| Error::Config(format!("GGUF tensor {name} size overflow")))?;
        let bytes = match info.ggml_type {
            0 => elems.checked_mul(4),
            30 => elems.checked_mul(2),
            142 => elems.checked_div(128).and_then(|n| n.checked_mul(34)),
            143 => elems.checked_div(128).and_then(|n| n.checked_mul(28)),
            ty => {
                return Err(Error::Unsupported(format!(
                    "Prism GGUF tensor {name} type {ty} unsupported"
                )))
            }
        }
        .ok_or_else(|| Error::Config(format!("GGUF tensor {name} has invalid block geometry")))?;
        let start = self
            .data_offset
            .checked_add(info.offset)
            .ok_or_else(|| Error::Config("GGUF tensor offset overflow".into()))?;
        // The declared extent must lie inside the file *before* allocating it (sc-27299).
        let file_len = self.file.get_ref().metadata()?.len();
        let end = u64::try_from(bytes).ok().and_then(|b| start.checked_add(b));
        if end.is_none_or(|end| end > file_len) {
            return Err(Error::Config(format!(
                "GGUF tensor {name} ({bytes} bytes at {start}) extends past the {file_len}-byte file"
            )));
        }
        let _operation = self.progress.operation();
        self.file.seek(SeekFrom::Start(start))?;
        let mut data = vec![0u8; bytes];
        self.file.read_exact(&mut data)?;
        Ok(data)
    }
}

fn read<const N: usize>(r: &mut impl Read) -> Result<[u8; N]> {
    let mut b = [0; N];
    r.read_exact(&mut b)?;
    Ok(b)
}
fn u16le(r: &mut impl Read) -> Result<u16> {
    Ok(u16::from_le_bytes(read(r)?))
}
fn i16le(r: &mut impl Read) -> Result<i16> {
    Ok(i16::from_le_bytes(read(r)?))
}
fn u32le(r: &mut impl Read) -> Result<u32> {
    Ok(u32::from_le_bytes(read(r)?))
}
fn i32le(r: &mut impl Read) -> Result<i32> {
    Ok(i32::from_le_bytes(read(r)?))
}
fn u64le(r: &mut impl Read) -> Result<u64> {
    Ok(u64::from_le_bytes(read(r)?))
}
fn i64le(r: &mut impl Read) -> Result<i64> {
    Ok(i64::from_le_bytes(read(r)?))
}
fn string(r: &mut impl Read) -> Result<String> {
    let len = usize::try_from(u64le(r)?)
        .map_err(|_| Error::Config("GGUF string length overflow".into()))?;
    if len > MAX_STRING_BYTES {
        return Err(Error::Config("GGUF string too large".into()));
    }
    let mut b = vec![0; len];
    r.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|e| Error::Config(format!("GGUF invalid UTF-8: {e}")))
}
fn value(r: &mut impl Read, ty: u32, depth: usize) -> Result<RawValue> {
    if depth > 2 {
        return Err(Error::Config("GGUF nested metadata too deep".into()));
    }
    Ok(match ty {
        0 => RawValue::U(read::<1>(r)?[0] as u64),
        1 => RawValue::I(read::<1>(r)?[0] as i8 as i64),
        2 => RawValue::U(u16le(r)? as u64),
        3 => RawValue::I(i16le(r)? as i64),
        4 => RawValue::U(u32le(r)? as u64),
        5 => RawValue::I(i32le(r)? as i64),
        6 => RawValue::F(f32::from_le_bytes(read(r)?) as f64),
        7 => match read::<1>(r)?[0] {
            0 => RawValue::Bool(false),
            1 => RawValue::Bool(true),
            x => return Err(Error::Config(format!("GGUF invalid bool {x}"))),
        },
        8 => RawValue::String(string(r)?),
        9 => {
            let inner = u32le(r)?;
            let len = usize::try_from(u64le(r)?)
                .map_err(|_| Error::Config("GGUF array length overflow".into()))?;
            if len > 10_000_000 {
                return Err(Error::Config("GGUF metadata array too large".into()));
            }
            let mut out = Vec::with_capacity(len.min(MAX_PRERESERVE));
            for _ in 0..len {
                out.push(value(r, inner, depth + 1)?);
            }
            RawValue::Array(out)
        }
        10 => RawValue::U(u64le(r)?),
        11 => RawValue::I(i64le(r)?),
        12 => RawValue::F(f64::from_le_bytes(read(r)?)),
        _ => {
            return Err(Error::Config(format!(
                "GGUF metadata type {ty} unsupported"
            )))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires frozen Bonsai GGUF via BONSAI_GGUF"]
    fn parses_frozen_bonsai_header_and_provisional_types() {
        let path = std::env::var_os("BONSAI_GGUF")
            .map(std::path::PathBuf::from)
            .expect("BONSAI_GGUF must point to the frozen PQ2_0 or PTQ1_0 file");
        let gguf = RawGguf::open(&path).unwrap();
        assert_eq!(
            gguf.metadata
                .get("general.architecture")
                .and_then(RawValue::string),
            Some("qwen35")
        );
        assert_eq!(gguf.tensors.len(), 851);
        assert!(gguf
            .tensors
            .values()
            .any(|t| matches!(t.ggml_type, 142 | 143)));
        let sections = gguf.metadata["qwen35.rope.dimension_sections"]
            .array()
            .unwrap();
        assert_eq!(
            sections
                .iter()
                .map(RawValue::u64)
                .collect::<Option<Vec<_>>>(),
            Some(vec![11, 11, 10, 0])
        );
        let signs = gguf.metadata["prism.hadamard.sign_values"].array().unwrap();
        assert!(signs.iter().all(|value| value.exact_sign().is_some()));
    }

    /// A GGUF v3 header with the given counts followed by `body`.
    fn header(tensor_count: u64, metadata_count: u64, body: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&tensor_count.to_le_bytes());
        b.extend_from_slice(&metadata_count.to_le_bytes());
        b.extend_from_slice(body);
        b
    }

    fn open_bytes(bytes: &[u8]) -> Result<RawGguf> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.gguf");
        std::fs::write(&path, bytes).unwrap();
        RawGguf::open(&path)
    }

    fn gguf_string(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    /// sc-27299: a 24-byte file declaring `u64::MAX` metadata or tensor entries is a typed error,
    /// not a capacity-overflow panic or a huge reservation.
    #[test]
    fn hostile_header_counts_are_rejected_before_allocating() {
        for (tensors, metadata) in [(0, u64::MAX), (u64::MAX, 0), (1 << 40, 1 << 40)] {
            let err = open_bytes(&header(tensors, metadata, &[]))
                .err()
                .expect("hostile count rejected");
            assert!(err.to_string().contains("count"), "{err}");
        }
        // A huge array length or string length inside an otherwise small file errors cleanly.
        let mut body = Vec::new();
        gguf_string(&mut body, "k");
        body.extend_from_slice(&9u32.to_le_bytes()); // array
        body.extend_from_slice(&4u32.to_le_bytes()); // of u32
        body.extend_from_slice(&5_000_000u64.to_le_bytes()); // declared, not backed
        assert!(open_bytes(&header(0, 1, &body)).is_err());
        let mut body = Vec::new();
        body.extend_from_slice(&u64::MAX.to_le_bytes()); // key length
        assert!(open_bytes(&header(0, 1, &body)).is_err());
    }

    /// sc-27299: a tensor whose declared extent runs past the end of the file is rejected before
    /// `read_tensor` allocates the declared byte count.
    #[test]
    fn read_tensor_rejects_an_extent_past_the_file() {
        let mut body = Vec::new();
        gguf_string(&mut body, "w");
        body.extend_from_slice(&2u32.to_le_bytes()); // rank
        body.extend_from_slice(&(1u64 << 20).to_le_bytes());
        body.extend_from_slice(&(1u64 << 20).to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes()); // F32: 4 TiB declared
        body.extend_from_slice(&0u64.to_le_bytes()); // offset
        let mut gguf = open_bytes(&header(1, 0, &body)).unwrap();
        let err = gguf.read_tensor("w").unwrap_err();
        assert!(err.to_string().contains("extends past"), "{err}");

        // A small in-bounds tensor still reads.
        let mut body = Vec::new();
        gguf_string(&mut body, "v");
        body.extend_from_slice(&1u32.to_le_bytes());
        body.extend_from_slice(&2u64.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes());
        body.extend_from_slice(&0u64.to_le_bytes());
        let mut bytes = header(1, 0, &body);
        bytes.resize(bytes.len().div_ceil(32) * 32, 0);
        bytes.extend_from_slice(&[1, 0, 0, 0, 2, 0, 0, 0]);
        let mut gguf = open_bytes(&bytes).unwrap();
        assert_eq!(gguf.read_tensor("v").unwrap(), [1, 0, 0, 0, 2, 0, 0, 0]);
    }

    #[test]
    fn exact_numeric_coercions_preserve_strict_metadata_validation() {
        assert_eq!(RawValue::I(11).u64(), Some(11));
        assert_eq!(RawValue::I(-1).u64(), None);
        assert_eq!(RawValue::I(-1).exact_sign(), Some(-1));
        assert_eq!(RawValue::F(-1.0).exact_sign(), Some(-1));
        assert_eq!(RawValue::U(1).exact_sign(), Some(1));
        for value in [
            RawValue::F(0.999),
            RawValue::F(f64::NAN),
            RawValue::I(0),
            RawValue::I(2),
            RawValue::U(2),
        ] {
            assert_eq!(value.exact_sign(), None);
        }
    }
}
