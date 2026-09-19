//! Minimal strict GGUF reader for Prism's provisional PQ2_0/PTQ1_0 tensor types (142/143).

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::{Error, Result};

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
    file: std::fs::File,
}

impl RawGguf {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        if u32le(&mut file)? != 0x4655_4747 {
            return Err(Error::Config("GGUF bad magic".into()));
        }
        let version = u32le(&mut file)?;
        if !(2..=3).contains(&version) {
            return Err(Error::Config(format!("GGUF version {version} unsupported")));
        }
        let tensor_count = usize::try_from(u64le(&mut file)?)
            .map_err(|_| Error::Config("GGUF tensor count overflow".into()))?;
        let metadata_count = usize::try_from(u64le(&mut file)?)
            .map_err(|_| Error::Config("GGUF metadata count overflow".into()))?;
        let mut metadata = HashMap::with_capacity(metadata_count);
        for _ in 0..metadata_count {
            let key = string(&mut file)?;
            let ty = u32le(&mut file)?;
            let value = value(&mut file, ty, 0)?;
            if metadata.insert(key.clone(), value).is_some() {
                return Err(Error::Config(format!("duplicate GGUF metadata {key}")));
            }
        }
        let mut tensors = HashMap::with_capacity(tensor_count);
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
        Ok(Self {
            metadata,
            tensors,
            data_offset,
            file,
        })
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
        self.file.seek(SeekFrom::Start(
            self.data_offset
                .checked_add(info.offset)
                .ok_or_else(|| Error::Config("GGUF tensor offset overflow".into()))?,
        ))?;
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
    if len > 1 << 30 {
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
            let mut out = Vec::with_capacity(len);
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
    }
}
