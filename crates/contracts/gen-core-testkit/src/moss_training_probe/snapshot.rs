//! Bounded, header-only/streamed diagnostics over files written by the real native exporters.
//! Never loads a backbone, optimizer moment population, or whole adapter data into memory.
use gen_core::{Error, Result};
use gen_core_moss_image::moss_image_training::recovery::RunState;
use serde::{
    de::{DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor},
    Deserialize, Serialize,
};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

const CHUNK: usize = 64 * 1024;
const READ_BUFFER: usize = 8192;
const META_KEYS: [&str; 4] = ["optim.d", "optim.d_max", "optim.d_numerator", "optim.step"];
const ID_KEYS: [&str; 21] = [
    "sceneworks_adapter_format",
    "model_family",
    "baseModel",
    "base_repository",
    "base_revision",
    "shared_revision",
    "vae_posterior",
    "training_sampler",
    "training_objective",
    "adapter_compute",
    "source_checkpoint_identity",
    "dataset_fingerprint",
    "effective_training_controls",
    "rank",
    "alpha",
    "networkType",
    "decomposeFactor",
    "triggerWords",
    "training_stream_version",
    "family",
    "base_model",
];
fn invalid(message: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("diagnostic export: {message}"))
}
fn json(error: serde_json::Error) -> Error {
    invalid(error)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdapterSnapshot {
    pub path: PathBuf,
    pub sha256: String,
    pub start_step: u32,
    pub start_update: u32,
    pub factor_tensors: usize,
    pub layout_sha256: String,
    pub export_identity: Vec<(String, String)>,
    pub workspace_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactorDelta {
    pub tensors: usize,
    pub elements: u64,
    pub changed_elements: u64,
    pub max_abs_delta: f64,
    pub squared_l2_delta: f64,
    pub initial_sha256: String,
    pub final_sha256: String,
    pub layout_sha256: String,
    pub workspace_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointObservation {
    pub state: RunState,
    pub factor_delta: FactorDelta,
    pub optimizer_scalars_as_stored: BTreeMap<String, f64>,
    pub workspace_bytes: u64,
}

#[derive(Debug)]
struct Shape {
    dims: [u64; 2],
    rank: usize,
}
impl<'de> Deserialize<'de> for Shape {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Shape;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a portable scalar/vector/matrix shape")
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Shape, A::Error> {
                let mut shape = Shape {
                    dims: [1; 2],
                    rank: 0,
                };
                while let Some(n) = seq.next_element::<u64>()? {
                    if shape.rank == 2 || n == 0 {
                        return Err(serde::de::Error::custom(
                            "not a portable adapter/optimizer shape",
                        ));
                    }
                    shape.dims[shape.rank] = n;
                    shape.rank += 1;
                }
                Ok(shape)
            }
        }
        d.deserialize_seq(V)
    }
}
#[derive(Deserialize, Debug)]
struct TensorInfo {
    dtype: String,
    shape: Shape,
    data_offsets: [u64; 2],
}
type Entry = (String, TensorInfo);
struct Header {
    entries: Vec<Entry>,
    scalars: [Option<f64>; 4],
    identity: [Option<String>; ID_KEYS.len()],
}
struct Count(usize);
impl<'de> Deserialize<'de> for Count {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Count;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a safetensors header")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Count, A::Error> {
                let mut n = 0usize;
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {
                    n = n
                        .checked_add(1)
                        .ok_or_else(|| serde::de::Error::custom("header count overflow"))?;
                }
                Ok(Count(n))
            }
        }
        d.deserialize_map(V)
    }
}
struct ScalarMetadata([Option<f64>; 4], [Option<String>; ID_KEYS.len()]);
impl<'de> Deserialize<'de> for ScalarMetadata {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = ScalarMetadata;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("native optimizer metadata")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = [None; 4];
                let mut identity = std::array::from_fn(|_| None);
                while let Some(key) = map.next_key::<String>()? {
                    let value = map.next_value::<String>()?;
                    if let Some(index) = META_KEYS.iter().position(|name| *name == key) {
                        let n = value.parse::<f64>().map_err(serde::de::Error::custom)?;
                        if !n.is_finite() {
                            return Err(serde::de::Error::custom("nonfinite optimizer scalar"));
                        }
                        values[index] = Some(n);
                    }
                    if let Some(index) = ID_KEYS.iter().position(|name| *name == key) {
                        identity[index] = Some(value);
                    }
                }
                Ok(ScalarMetadata(values, identity))
            }
        }
        d.deserialize_map(V)
    }
}
struct HeaderSeed {
    count: usize,
}
impl<'de> DeserializeSeed<'de> for HeaderSeed {
    type Value = Header;
    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        d: D,
    ) -> std::result::Result<Header, D::Error> {
        struct V(usize);
        impl<'de> Visitor<'de> for V {
            type Value = Header;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("the counted native tensor header")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Header, A::Error> {
                let mut header = Header {
                    entries: Vec::with_capacity(self.0),
                    scalars: [None; 4],
                    identity: std::array::from_fn(|_| None),
                };
                while let Some(key) = map.next_key::<String>()? {
                    if key == "__metadata__" {
                        // MLX optimizer exports legitimately use null (or omit metadata);
                        // their adaptive scalars live in tensors, not this optional map.
                        if let Some(metadata) = map.next_value::<Option<ScalarMetadata>>()? {
                            header.scalars = metadata.0;
                            header.identity = metadata.1;
                        }
                    } else {
                        if header.entries.len() == self.0 {
                            return Err(serde::de::Error::custom("native header changed"));
                        }
                        header.entries.push((key, map.next_value::<TensorInfo>()?));
                    }
                }
                header.entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                Ok(header)
            }
        }
        d.deserialize_map(V(self.count))
    }
}
struct Layout {
    file: File,
    base: u64,
    header: Header,
}
fn open_layouts(paths: &[&Path], budget: u64) -> Result<(Vec<Layout>, u64)> {
    let fixed = 2 * CHUNK
        + paths.len()
            * (READ_BUFFER
                + std::mem::size_of::<Layout>()
                + std::mem::size_of::<(File, u64, u64, u64, usize)>()
                + ID_KEYS.len() * std::mem::size_of::<(String, String)>()
                + ID_KEYS.iter().map(|s| s.len()).sum::<usize>());
    let mut charge = fixed as u64;
    if charge > budget {
        return Err(invalid("stream buffers exceed admitted audit workspace"));
    }
    let mut pending = Vec::with_capacity(paths.len());
    for path in paths {
        if !std::fs::symlink_metadata(path)?.is_file() {
            return Err(invalid("snapshot is not a regular file"));
        }
        let mut file = File::open(path)?;
        let length = file.metadata()?.len();
        let mut prefix = [0; 8];
        file.read_exact(&mut prefix)?;
        let h = u64::from_le_bytes(prefix);
        let base = h
            .checked_add(8)
            .filter(|n| *n <= length)
            .ok_or_else(|| invalid("truncated native header"))?;
        // Streamed JSON retains decoded names/current value plus parser scratch capacity, not a DOM.
        charge = charge
            .checked_add(
                h.checked_mul(3)
                    .ok_or_else(|| invalid("header workspace overflow"))?,
            )
            .ok_or_else(|| invalid("workspace overflow"))?;
        if charge > budget {
            return Err(invalid("header parsing exceeds admitted audit workspace"));
        }
        file.seek(SeekFrom::Start(8))?;
        let count = serde_json::from_reader::<_, Count>(BufReader::with_capacity(
            READ_BUFFER,
            (&mut file).take(h),
        ))
        .map_err(json)?
        .0;
        charge = charge
            .checked_add(
                (count as u64)
                    .checked_mul(std::mem::size_of::<Entry>() as u64)
                    .ok_or_else(|| invalid("entry workspace overflow"))?,
            )
            .ok_or_else(|| invalid("workspace overflow"))?;
        if charge > budget {
            return Err(invalid("header index exceeds admitted audit workspace"));
        }
        pending.push((file, base, length, h, count));
    }
    let mut layouts = Vec::with_capacity(pending.len());
    for (mut file, base, length, h, count) in pending {
        file.seek(SeekFrom::Start(8))?;
        let mut d = serde_json::Deserializer::from_reader(BufReader::with_capacity(
            READ_BUFFER,
            (&mut file).take(h),
        ));
        let header = HeaderSeed { count }.deserialize(&mut d).map_err(json)?;
        d.end().map_err(json)?;
        drop(d);
        for (_, info) in &header.entries {
            let [a, b] = info.data_offsets;
            if a > b || base.checked_add(b).is_none_or(|end| end > length) {
                return Err(invalid("tensor offset outside native export"));
            }
        }
        layouts.push(Layout { file, base, header });
    }
    Ok((layouts, charge))
}
pub(super) fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = [0; CHUNK];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn layout_hash(header: &Header) -> String {
    let mut hash = Sha256::new();
    hash.update(b"portable-export-logical-layout-v1");
    for (name, info) in &header.entries {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((info.dtype.len() as u64).to_le_bytes());
        hash.update(info.dtype.as_bytes());
        hash.update((info.shape.rank as u64).to_le_bytes());
        for dim in info.shape.dims {
            hash.update(dim.to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}
pub(super) fn inspect(path: &Path, step: u32, update: u32, budget: u64) -> Result<AdapterSnapshot> {
    let (layouts, workspace) = open_layouts(&[path], budget)?;
    let tensors = layouts[0]
        .header
        .entries
        .iter()
        .filter(|(name, _)| !name.ends_with(".alpha"))
        .count();
    Ok(AdapterSnapshot {
        path: path.to_path_buf(),
        sha256: hash_file(path)?,
        start_step: step,
        start_update: update,
        factor_tensors: tensors,
        layout_sha256: layout_hash(&layouts[0].header),
        export_identity: ID_KEYS
            .iter()
            .zip(layouts[0].header.identity.iter())
            .filter_map(|(k, v)| v.as_ref().map(|v| ((*k).to_owned(), v.clone())))
            .collect(),
        workspace_bytes: workspace,
    })
}
pub(super) fn delta(initial: &Path, final_path: &Path, budget: u64) -> Result<FactorDelta> {
    let (mut layouts, workspace) = open_layouts(&[initial, final_path], budget)?;
    let initial_sha = hash_file(initial)?;
    let final_sha = hash_file(final_path)?;
    let mut right = layouts.pop().expect("two layouts");
    let mut left = layouts.pop().expect("two layouts");
    let layout = layout_hash(&left.header);
    if layout != layout_hash(&right.header) || left.header.identity != right.header.identity {
        return Err(invalid("parent/source/controls/export layout differs"));
    }
    let count = left
        .header
        .entries
        .iter()
        .filter(|(name, _)| !name.ends_with(".alpha"))
        .count();
    if count
        != right
            .header
            .entries
            .iter()
            .filter(|(name, _)| !name.ends_with(".alpha"))
            .count()
    {
        return Err(invalid("factor population differs"));
    }
    let mut stats = FactorDelta {
        tensors: count,
        elements: 0,
        changed_elements: 0,
        max_abs_delta: 0.0,
        squared_l2_delta: 0.0,
        initial_sha256: initial_sha,
        final_sha256: final_sha,
        layout_sha256: layout,
        workspace_bytes: workspace,
    };
    let (mut lhs, mut rhs) = (vec![0u8; CHUNK], vec![0u8; CHUNK]);
    for ((name, x), (other, y)) in left.header.entries.iter().zip(right.header.entries.iter()) {
        if name != other
            || x.dtype != "F32"
            || y.dtype != "F32"
            || x.shape.rank != y.shape.rank
            || x.shape.dims != y.shape.dims
        {
            return Err(invalid("raw portable factor schema differs"));
        }
        let bytes = x.data_offsets[1] - x.data_offsets[0];
        let expected = x
            .shape
            .dims
            .iter()
            .try_fold(4_u64, |n, d| n.checked_mul(*d))
            .ok_or_else(|| invalid("factor size overflow"))?;
        if bytes != expected || y.data_offsets[1] - y.data_offsets[0] != bytes {
            return Err(invalid("factor payload length differs"));
        }
        left.file
            .seek(SeekFrom::Start(left.base + x.data_offsets[0]))?;
        right
            .file
            .seek(SeekFrom::Start(right.base + y.data_offsets[0]))?;
        let mut remaining = bytes;
        while remaining > 0 {
            let n = remaining.min(CHUNK as u64) as usize;
            left.file.read_exact(&mut lhs[..n])?;
            right.file.read_exact(&mut rhs[..n])?;
            for (a, b) in lhs[..n].chunks_exact(4).zip(rhs[..n].chunks_exact(4)) {
                let a = f32::from_le_bytes(a.try_into().expect("four bytes"));
                let b = f32::from_le_bytes(b.try_into().expect("four bytes"));
                if !a.is_finite() || !b.is_finite() {
                    return Err(invalid("nonfinite factor delta"));
                }
                if name.ends_with(".alpha") {
                    if a.to_bits() != b.to_bits() {
                        return Err(invalid("portable alpha scalar differs"));
                    }
                    continue;
                }
                let difference = f64::from(b) - f64::from(a);
                stats.elements += 1;
                if a != b {
                    stats.changed_elements += 1;
                }
                stats.max_abs_delta = stats.max_abs_delta.max(difference.abs());
                stats.squared_l2_delta += difference * difference;
            }
            remaining -= n as u64;
        }
    }
    drop(lhs);
    drop(rhs);
    if hash_file(initial)? != stats.initial_sha256 || hash_file(final_path)? != stats.final_sha256 {
        return Err(invalid("export changed during streaming delta"));
    }
    Ok(stats)
}
pub(super) fn optimizer_scalars(path: &Path, budget: u64) -> Result<(BTreeMap<String, f64>, u64)> {
    let (mut layouts, workspace) = open_layouts(&[path], budget)?;
    let mut layout = layouts.pop().expect("one layout");
    let mut values = BTreeMap::new();
    for (key, n) in META_KEYS.iter().zip(layout.header.scalars) {
        if let Some(n) = n {
            values.insert((*key).into(), n);
        }
    }
    for (key, info) in &layout.header.entries {
        if !["__d", "__d_max", "__d_numerator"].contains(&key.as_str()) {
            continue;
        }
        if info.dtype != "F32" || info.data_offsets[1] - info.data_offsets[0] != 4 {
            return Err(invalid("adaptive scalar schema differs"));
        }
        layout
            .file
            .seek(SeekFrom::Start(layout.base + info.data_offsets[0]))?;
        let mut bytes = [0; 4];
        layout.file.read_exact(&mut bytes)?;
        let n = f64::from(f32::from_le_bytes(bytes));
        if !n.is_finite() {
            return Err(invalid("nonfinite adaptive scalar"));
        }
        values.insert(key.clone(), n);
    }
    Ok((values, workspace))
}
pub(super) fn copy_export(source: &Path, target: &Path) -> Result<()> {
    let mut input = File::open(source)?;
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(target)?;
    let mut bytes = [0; CHUNK];
    loop {
        let n = input.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        output.write_all(&bytes[..n])?;
    }
    output.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    // Format/parser fixtures only: never passed to a model, registered, or claimed as training output.
    fn file(path: &Path, values: [f32; 4], parent: &str, shape: serde_json::Value, alpha: f32) {
        let header = serde_json::json!({"__metadata__":{"baseModel":parent,"alpha":"0.7","optim.d":"0.000001"},
            "block.lora_A.weight":{"dtype":"F32","shape":shape,"data_offsets":[0,16]},
            "block.alpha":{"dtype":"F32","shape":[1],"data_offsets":[16,20]},
            "__d":{"dtype":"F32","shape":[1],"data_offsets":[20,24]}});
        let mut h = serde_json::to_vec(&header).unwrap();
        while !h.len().is_multiple_of(8) {
            h.push(b' ');
        }
        let mut f = File::create(path).unwrap();
        f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&h).unwrap();
        for v in values {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        f.write_all(&alpha.to_le_bytes()).unwrap();
        f.write_all(&1e-6_f32.to_le_bytes()).unwrap();
    }
    #[test]
    fn streamed_factor_deltas_match_independent_known_values_and_saved_scalars() {
        let tmp = tempfile::tempdir().unwrap();
        let a = tmp.path().join("a.safetensors");
        let b = tmp.path().join("b.safetensors");
        file(
            &a,
            [0., 1., -2., 3.],
            "base",
            serde_json::json!([2, 2]),
            0.7,
        );
        file(
            &b,
            [0., 1.125, -1.5, 3.],
            "base",
            serde_json::json!([2, 2]),
            0.7,
        );
        let observed = delta(&a, &b, super::super::AUDIT_WORKSPACE_BYTES).unwrap();
        assert_eq!(observed.changed_elements, 2);
        assert_eq!(observed.elements, 5); // Four matrix elements + the unchanged scalar fixture __d.
        assert_eq!(observed.max_abs_delta, 0.5);
        assert_eq!(observed.squared_l2_delta, 0.265625);
        assert!(observed.workspace_bytes <= super::super::AUDIT_WORKSPACE_BYTES);
        let (scalars, _) = optimizer_scalars(&a, super::super::AUDIT_WORKSPACE_BYTES).unwrap();
        assert_eq!(scalars["optim.d"], 1e-6_f64);
        assert_eq!(scalars["__d"], f64::from(1e-6_f32));
        let identity = delta(&a, &a, super::super::AUDIT_WORKSPACE_BYTES).unwrap();
        assert_eq!(identity.changed_elements, 0);
        assert_eq!(identity.squared_l2_delta, 0.0);
    }
    #[test]
    fn optimizer_metadata_null_absent_and_map_preserve_strict_scalar_validation() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("optimizer.safetensors");
        // IO-only format controls, matching the native optimizer's optional metadata.
        let write = |metadata: Option<serde_json::Value>, dtype: &str, scalar: f32| {
            let mut header = serde_json::json!({
                "__d": {"dtype":dtype,"shape":[1],"data_offsets":[0,4]}
            });
            if let Some(metadata) = metadata {
                header["__metadata__"] = metadata;
            }
            let mut bytes = serde_json::to_vec(&header).unwrap();
            while !bytes.len().is_multiple_of(8) {
                bytes.push(b' ');
            }
            let mut f = File::create(&path).unwrap();
            f.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
            f.write_all(&bytes).unwrap();
            f.write_all(&scalar.to_le_bytes()).unwrap();
        };
        let budget = super::super::AUDIT_WORKSPACE_BYTES;
        for metadata in [
            None,
            Some(serde_json::Value::Null),
            Some(serde_json::json!({})),
            Some(serde_json::json!({"optim.d":"0.000001"})),
        ] {
            let map_has_scalar = metadata
                .as_ref()
                .is_some_and(|m| m.get("optim.d").is_some());
            write(metadata, "F32", 1e-6);
            let (scalars, workspace) = optimizer_scalars(&path, budget).unwrap();
            assert_eq!(scalars["__d"], f64::from(1e-6_f32));
            assert_eq!(
                scalars.get("optim.d").copied(),
                map_has_scalar.then_some(1e-6)
            );
            assert!(workspace <= budget);
            assert!(optimizer_scalars(&path, workspace - 1).is_err());
        }
        for metadata in [
            serde_json::json!([]),
            serde_json::json!("wrong"),
            serde_json::json!(42),
            serde_json::json!({"optim.d":null}),
            serde_json::json!({"optim.d":1}),
            serde_json::json!({"optim.d":"NaN"}),
            serde_json::json!({"optim.d":"inf"}),
            serde_json::json!({"optim.d":"-inf"}),
            serde_json::json!({"unrelated":false}),
        ] {
            write(Some(metadata), "F32", 1e-6);
            assert!(optimizer_scalars(&path, budget).is_err());
        }
        for metadata in [None, Some(serde_json::Value::Null)] {
            for scalar in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                write(metadata.clone(), "F32", scalar);
                assert!(optimizer_scalars(&path, budget).is_err());
            }
            write(metadata, "F64", 1e-6);
            assert!(optimizer_scalars(&path, budget).is_err());
        }
    }
    #[test]
    fn streamed_reader_rejects_other_parent_layout_alpha_and_unadmitted_buffers() {
        let tmp = tempfile::tempdir().unwrap();
        let a = tmp.path().join("a.safetensors");
        let b = tmp.path().join("b.safetensors");
        file(
            &a,
            [0., 1., -2., 3.],
            "base",
            serde_json::json!([2, 2]),
            0.7,
        );
        for (parent, shape, alpha) in [
            ("other", serde_json::json!([2, 2]), 0.7),
            ("base", serde_json::json!([4]), 0.7),
            ("base", serde_json::json!([2, 2]), 0.75),
        ] {
            file(&b, [0., 1., -2., 3.], parent, shape, alpha);
            assert!(delta(&a, &b, super::super::AUDIT_WORKSPACE_BYTES).is_err());
        }
        assert!(delta(&a, &a, 1).is_err());
    }
}
