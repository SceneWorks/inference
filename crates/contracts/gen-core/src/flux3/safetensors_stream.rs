//! Bounded-memory, deterministic safetensors IO for native FLUX3 converters.
//!
//! A converter reads ONE source tensor at a time from the original dense checkpoint and streams
//! ONE output tensor at a time into a pre-declared layout. The header is fixed before any payload
//! is written (names sorted, offsets derived from the declared byte sizes, fixed metadata), so the
//! same source and algorithm always produce byte-identical files. Nothing here quantizes: each
//! backend's converter supplies the bytes its own loader reads.
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use crate::{Error, Result};

/// Refuse a header larger than this: no legitimate FLUX3 header approaches it, and an unbounded
/// length field would otherwise drive an unbounded allocation.
const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;

fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 safetensors: {detail}"))
}

/// Element width of a safetensors dtype string, or `None` for an unknown dtype.
pub fn dtype_size(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
        "I16" | "U16" | "F16" | "BF16" => 2,
        "I32" | "U32" | "F32" => 4,
        "I64" | "U64" | "F64" => 8,
        _ => return None,
    })
}

fn payload_bytes(dtype: &str, shape: &[usize]) -> Result<u64> {
    let width = dtype_size(dtype).ok_or_else(|| invalid(format!("unknown dtype {dtype}")))?;
    shape
        .iter()
        .try_fold(width, |acc, &d| acc.checked_mul(d as u64))
        .ok_or_else(|| invalid("tensor byte size overflow"))
}

/// One tensor of a source checkpoint, located by its absolute byte range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    start: u64,
    end: u64,
}

impl SourceTensor {
    pub fn bytes(&self) -> u64 {
        self.end - self.start
    }
}

/// A validated, read-only view of one source safetensors file. Payload is read per tensor.
#[derive(Debug)]
pub struct SafetensorsSource {
    path: PathBuf,
    tensors: BTreeMap<String, SourceTensor>,
}

impl SafetensorsSource {
    /// Parse and fully validate the header: every range inside the file, sized exactly as its
    /// dtype and shape require, non-overlapping, and covering the payload with no gap.
    pub fn open(path: &Path) -> Result<Self> {
        let mut file =
            File::open(path).map_err(|e| invalid(format!("open {}: {e}", path.display())))?;
        let file_len = file.metadata()?.len();
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let header_len = u64::from_le_bytes(len);
        if header_len > MAX_HEADER_BYTES || header_len.saturating_add(8) > file_len {
            return Err(invalid(format!(
                "{}: invalid header length",
                path.display()
            )));
        }
        let mut header = vec![0u8; header_len as usize];
        file.read_exact(&mut header)?;
        let header: Map<String, Value> = serde_json::from_slice(&header)
            .map_err(|e| invalid(format!("{}: header JSON: {e}", path.display())))?;
        let data_start = 8 + header_len;
        let mut tensors = BTreeMap::new();
        let mut ranges = Vec::new();
        for (name, entry) in &header {
            if name == "__metadata__" {
                continue;
            }
            let dtype = entry["dtype"]
                .as_str()
                .ok_or_else(|| invalid(format!("{name}: missing dtype")))?
                .to_owned();
            let shape = entry["shape"]
                .as_array()
                .ok_or_else(|| invalid(format!("{name}: missing shape")))?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| invalid(format!("{name}: invalid shape")))?;
            let offsets = entry["data_offsets"]
                .as_array()
                .filter(|o| o.len() == 2)
                .and_then(|o| Some((o[0].as_u64()?, o[1].as_u64()?)))
                .ok_or_else(|| invalid(format!("{name}: invalid data_offsets")))?;
            if offsets.1 < offsets.0 || offsets.1 - offsets.0 != payload_bytes(&dtype, &shape)? {
                return Err(invalid(format!(
                    "{name}: byte range disagrees with dtype/shape"
                )));
            }
            let (start, end) = (
                data_start
                    .checked_add(offsets.0)
                    .ok_or_else(|| invalid("offset overflow"))?,
                data_start
                    .checked_add(offsets.1)
                    .ok_or_else(|| invalid("offset overflow"))?,
            );
            if end > file_len {
                return Err(invalid(format!("{name}: payload extends past end of file")));
            }
            ranges.push((offsets.0, offsets.1));
            tensors.insert(
                name.clone(),
                SourceTensor {
                    name: name.clone(),
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
        }
        ranges.sort_unstable();
        let mut cursor = 0;
        for (start, end) in ranges {
            if start != cursor {
                return Err(invalid(format!(
                    "{}: overlapping or gapped tensor payloads",
                    path.display()
                )));
            }
            cursor = end;
        }
        if data_start + cursor != file_len {
            return Err(invalid(format!(
                "{}: trailing bytes after the declared payload",
                path.display()
            )));
        }
        Ok(Self {
            path: path.to_path_buf(),
            tensors,
        })
    }

    pub fn tensors(&self) -> &BTreeMap<String, SourceTensor> {
        &self.tensors
    }

    pub fn get(&self, name: &str) -> Result<&SourceTensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| invalid(format!("{}: missing tensor {name}", self.path.display())))
    }

    /// The raw little-endian payload of one tensor (the only allocation is that tensor's bytes).
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let tensor = self.get(name)?;
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(tensor.start))?;
        let mut bytes = vec![0u8; tensor.bytes() as usize];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }
}

/// One declared output tensor; its byte size is derived from dtype and shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

/// Writes a safetensors file whose header was fixed up front. Tensors must be written in name
/// order with exactly their declared sizes; [`Self::finish`] refuses an incomplete file.
pub struct SafetensorsStreamWriter {
    file: BufWriter<File>,
    planned: Vec<(PlannedTensor, u64)>,
    next: usize,
}

impl SafetensorsStreamWriter {
    /// Create `path` (create-new: an existing file is never replaced) with the declared layout.
    pub fn create(
        path: &Path,
        mut planned: Vec<PlannedTensor>,
        metadata: &BTreeMap<String, String>,
    ) -> Result<Self> {
        planned.sort_by(|a, b| a.name.cmp(&b.name));
        if planned.windows(2).any(|w| w[0].name == w[1].name)
            || planned
                .iter()
                .any(|t| t.name.is_empty() || t.name == "__metadata__")
        {
            return Err(invalid("duplicate or reserved output tensor name"));
        }
        let mut header = Map::new();
        if !metadata.is_empty() {
            header.insert(
                "__metadata__".into(),
                Value::Object(
                    metadata
                        .iter()
                        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                        .collect(),
                ),
            );
        }
        let mut offset = 0u64;
        let mut sized = Vec::with_capacity(planned.len());
        for tensor in planned {
            let bytes = payload_bytes(&tensor.dtype, &tensor.shape)?;
            let end = offset
                .checked_add(bytes)
                .ok_or_else(|| invalid("output size overflow"))?;
            header.insert(
                tensor.name.clone(),
                json!({"dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": [offset, end]}),
            );
            offset = end;
            sized.push((tensor, bytes));
        }
        let mut header = serde_json::to_vec(&Value::Object(header)).map_err(invalid)?;
        // The reference writer pads the header with spaces to an 8-byte boundary.
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| invalid(format!("create {}: {e}", path.display())))?;
        let mut file = BufWriter::with_capacity(1 << 20, file);
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(&header)?;
        Ok(Self {
            file,
            planned: sized,
            next: 0,
        })
    }

    /// The next tensor's name, in write order.
    pub fn next_name(&self) -> Option<&str> {
        self.planned.get(self.next).map(|(t, _)| t.name.as_str())
    }

    pub fn write(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        let (tensor, want) = self
            .planned
            .get(self.next)
            .ok_or_else(|| invalid(format!("{name}: written after the declared layout ended")))?;
        if tensor.name != name || *want != bytes.len() as u64 {
            return Err(invalid(format!(
                "{name}: out of declared order or size ({} B, expected {} {want} B)",
                bytes.len(),
                tensor.name
            )));
        }
        self.file.write_all(bytes)?;
        self.next += 1;
        Ok(())
    }

    /// Flush and fsync. Refuses a file with undeclared or missing tensors.
    pub fn finish(mut self) -> Result<()> {
        if self.next != self.planned.len() {
            return Err(invalid(format!(
                "incomplete output: {} of {} tensors written",
                self.next,
                self.planned.len()
            )));
        }
        self.file.flush()?;
        self.file
            .into_inner()
            .map_err(|e| invalid(format!("flush: {e}")))?
            .sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planned(name: &str, dtype: &str, shape: &[usize]) -> PlannedTensor {
        PlannedTensor {
            name: name.into(),
            dtype: dtype.into(),
            shape: shape.to_vec(),
        }
    }

    #[test]
    fn streamed_file_round_trips_and_is_byte_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let meta = BTreeMap::from([("format".to_string(), "pt".to_string())]);
        let write = |name: &str| {
            let path = dir.path().join(name);
            // Declared out of order: the writer sorts, so the file is order-independent.
            let mut w = SafetensorsStreamWriter::create(
                &path,
                vec![
                    planned("b.weight", "U8", &[2, 3]),
                    planned("a.bias", "F32", &[2]),
                ],
                &meta,
            )
            .unwrap();
            assert_eq!(w.next_name(), Some("a.bias"));
            w.write("a.bias", &[0u8; 8]).unwrap();
            w.write("b.weight", &[1, 2, 3, 4, 5, 6]).unwrap();
            w.finish().unwrap();
            path
        };
        let (one, two) = (write("one.safetensors"), write("two.safetensors"));
        assert_eq!(std::fs::read(&one).unwrap(), std::fs::read(&two).unwrap());
        let source = SafetensorsSource::open(&one).unwrap();
        assert_eq!(source.tensors().len(), 2);
        assert_eq!(source.read("b.weight").unwrap(), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(source.get("b.weight").unwrap().shape, vec![2, 3]);
        let header_len = u64::from_le_bytes(std::fs::read(&one).unwrap()[..8].try_into().unwrap());
        assert_eq!(header_len % 8, 0);
    }

    #[test]
    fn out_of_order_wrong_size_incomplete_and_existing_outputs_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let layout = || vec![planned("a", "U8", &[2]), planned("b", "U8", &[1])];
        let path = dir.path().join("x.safetensors");
        let mut w = SafetensorsStreamWriter::create(&path, layout(), &BTreeMap::new()).unwrap();
        assert!(w.write("b", &[0]).is_err(), "order is fixed by the header");
        assert!(w.write("a", &[0]).is_err(), "size is fixed by the header");
        w.write("a", &[0, 0]).unwrap();
        assert!(
            w.finish().is_err(),
            "a missing tensor is never a complete file"
        );
        assert!(
            SafetensorsStreamWriter::create(&path, layout(), &BTreeMap::new()).is_err(),
            "an existing file is never replaced"
        );
        assert!(SafetensorsStreamWriter::create(
            &dir.path().join("d.safetensors"),
            vec![planned("a", "U8", &[1]), planned("a", "U8", &[1])],
            &BTreeMap::new()
        )
        .is_err());
    }

    #[test]
    fn corrupt_source_ranges_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let header = br#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0u8; 4]);
        let path = dir.path().join("bad.safetensors");
        std::fs::write(&path, &bytes).unwrap();
        assert!(
            SafetensorsSource::open(&path).is_err(),
            "shape needs 8 bytes"
        );
        let header = br#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0u8; 6]);
        std::fs::write(&path, &bytes).unwrap();
        assert!(
            SafetensorsSource::open(&path).is_err(),
            "trailing payload bytes"
        );
    }
}
