//! Tensor-free torch `.pth` / `.pt` (zip-of-pickle) reader, shared by both backends (sc-24213).
//!
//! First written for the native Wan converter (sc-3237 / sc-3224) inside mlx-gen, where it decoded
//! straight into MLX arrays. Supra2-IMG's checkpoint (`model_final_ema.pt`) must be converted on
//! *both* the MLX and the Candle install paths, so the pickle VM and the archive access now live here
//! and return **raw little-endian bytes** plus dtype/shape metadata — no tensor library. mlx-gen's
//! `pth` module is a thin adapter over this one (it widens to f32 and builds `Array`s), so the Wan and
//! E-LatentLPIPS readers keep their exact behavior.
//!
//! A `torch.save` archive is a ZIP holding `<prefix>/data.pkl` (a pickled object graph whose tensors
//! are `persistent_id` storage references) plus `<prefix>/data/<key>` raw storage blobs. Scope is
//! deliberately narrow — the opcode set `torch.save` emits for nested dicts of tensors and scalars
//! (protocol 2, STORED zip entries) and the globals it references (`collections.OrderedDict`,
//! `torch.<Float|BFloat16|Half|Long>Storage`, `torch._utils._rebuild_tensor_v2`). Anything else is a
//! typed error, never arbitrary code execution: this is a data-only interpreter.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::{Error, Result};

/// Torch storage element type (the `torch.<X>Storage` global in the pickle).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PthDtype {
    F32,
    BF16,
    F16,
    I64,
}

impl PthDtype {
    fn from_global(name: &str) -> Result<Self> {
        match name {
            "torch FloatStorage" => Ok(Self::F32),
            "torch BFloat16Storage" => Ok(Self::BF16),
            "torch HalfStorage" => Ok(Self::F16),
            "torch LongStorage" => Ok(Self::I64),
            other => Err(Error::Msg(format!(
                "unsupported torch storage type `{other}` in .pth (expected Float/BFloat16/Half/Long)"
            ))),
        }
    }

    /// Bytes per element.
    pub fn elem_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::BF16 | Self::F16 => 2,
            Self::I64 => 8,
        }
    }
}

/// A tensor reference inside the pickle: storage key plus the `_rebuild_tensor_v2` view metadata.
/// The bytes are read on demand through [`PthArchive::read_tensor`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PthTensorSpec {
    pub dtype: PthDtype,
    /// Logical shape, in PyTorch order.
    pub shape: Vec<i64>,
    storage_key: String,
    offset: i64,
    stride: Vec<i64>,
}

impl PthTensorSpec {
    /// Element count (validated non-negative dims; saturating only for display paths).
    pub fn numel(&self) -> usize {
        self.shape.iter().map(|&d| d.max(0) as usize).product()
    }
}

/// A decoded pickle value. Only the shapes `torch.save` produces for checkpoints are modeled.
#[derive(Clone, Debug, PartialEq)]
pub enum PthValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    /// A tuple or list.
    Seq(Vec<PthValue>),
    /// A `dict`/`OrderedDict` in insertion order. Integer keys are rendered in decimal.
    Dict(Vec<(String, PthValue)>),
    Tensor(PthTensorSpec),
}

impl PthValue {
    /// Look up `key` in a dict value.
    pub fn get(&self, key: &str) -> Option<&PthValue> {
        match self {
            PthValue::Dict(items) => items.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Every tensor reachable through nested dicts, named by its dotted path (`config.uncond_text`).
    /// A top-level `{name: tensor}` state dict therefore yields its keys unchanged.
    pub fn flatten_tensors(&self) -> Vec<(String, &PthTensorSpec)> {
        fn walk<'a>(v: &'a PthValue, prefix: &str, out: &mut Vec<(String, &'a PthTensorSpec)>) {
            match v {
                PthValue::Tensor(t) => out.push((prefix.to_string(), t)),
                PthValue::Dict(items) => {
                    for (k, child) in items {
                        let name = if prefix.is_empty() {
                            k.clone()
                        } else {
                            format!("{prefix}.{k}")
                        };
                        walk(child, &name, out);
                    }
                }
                _ => {}
            }
        }
        let mut out = Vec::new();
        walk(self, "", &mut out);
        out
    }
}

/// A raw tensor read from the archive: contiguous little-endian bytes in `dtype`.
#[derive(Clone, Debug, PartialEq)]
pub struct PthTensor {
    pub dtype: PthDtype,
    pub shape: Vec<i64>,
    pub data: Vec<u8>,
}

impl PthTensor {
    /// Widen to f32, mirroring `torch.load(...).float()` (int64 converts by value).
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        decode_to_f32(
            &self.data,
            self.dtype,
            self.data.len() / self.dtype.elem_size(),
        )
    }
}

// ------------------------------------------------------------------------------------------------
// Pickle VM
// ------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Val {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    None,
    Mark,
    Global(String),
    Tuple(Vec<Val>),
    List(Vec<Val>),
    Dict(Vec<(Val, Val)>),
    Storage {
        dtype: PthDtype,
        key: String,
    },
    Tensor {
        dtype: PthDtype,
        key: String,
        offset: i64,
        size: Vec<i64>,
        stride: Vec<i64>,
    },
}

/// Upper bound on opcodes executed by [`parse_pickle`] (a stream that loops or never reaches `STOP`).
const MAX_PICKLE_OPCODES: u64 = 64 * 1024 * 1024;

/// Cumulative budget on [`Val`] nodes *materialized by clones* (memoize / fetch) — the
/// "billion-laughs" amplification guard (F-015): `BINGET` + `TUPLE2` + `BINPUT` doubles the in-memory
/// structure every ~6 bytes, so each clone is charged before it is made.
const MAX_PICKLE_NODES: u64 = 8 * 1024 * 1024;

fn count_nodes(v: &Val) -> u64 {
    match v {
        Val::Tuple(items) | Val::List(items) => 1 + items.iter().map(count_nodes).sum::<u64>(),
        Val::Dict(items) => {
            1 + items
                .iter()
                .map(|(k, val)| count_nodes(k) + count_nodes(val))
                .sum::<u64>()
        }
        _ => 1,
    }
}

fn charge_nodes(total: &mut u64, n: u64) -> Result<()> {
    *total = total.saturating_add(n);
    if *total > MAX_PICKLE_NODES {
        return Err(Error::Msg(format!(
            "pickle: node budget exceeded ({MAX_PICKLE_NODES}); refusing to parse (possible memory-amplification attack)"
        )));
    }
    Ok(())
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn u8(&mut self) -> Result<u8> {
        let b = *self
            .buf
            .get(self.pos)
            .ok_or_else(|| Error::Msg("pickle: EOF".into()))?;
        self.pos += 1;
        Ok(b)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        let end = end.ok_or_else(|| Error::Msg("pickle: read past EOF".into()))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u16le(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32le(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32le(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64be(&mut self) -> Result<f64> {
        Ok(f64::from_be_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn line(&mut self) -> Result<String> {
        let start = self.pos;
        while *self
            .buf
            .get(self.pos)
            .ok_or_else(|| Error::Msg("pickle: unterminated line".into()))?
            != b'\n'
        {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.buf[start..self.pos])
            .map_err(|e| Error::Msg(format!("pickle: bad utf8 in global: {e}")))?
            .to_string();
        self.pos += 1;
        Ok(s)
    }
}

/// Run the pickle VM over `data.pkl`, returning the top object.
fn parse_pickle(data: &[u8]) -> Result<Val> {
    let mut r = Reader { buf: data, pos: 0 };
    let mut stack: Vec<Val> = Vec::new();
    let mut memo: HashMap<u32, Val> = HashMap::new();
    let mut opcodes: u64 = 0;
    let mut cloned_nodes: u64 = 0;

    let pop = |st: &mut Vec<Val>| {
        st.pop()
            .ok_or_else(|| Error::Msg("pickle: stack underflow".into()))
    };
    let pop_to_mark = |st: &mut Vec<Val>| -> Result<Vec<Val>> {
        let mut items = Vec::new();
        loop {
            match st.pop() {
                Some(Val::Mark) => break,
                Some(v) => items.push(v),
                None => return Err(Error::Msg("pickle: no Mark on stack".into())),
            }
        }
        items.reverse();
        Ok(items)
    };

    loop {
        opcodes += 1;
        if opcodes > MAX_PICKLE_OPCODES {
            return Err(Error::Msg(format!(
                "pickle: opcode budget exceeded ({MAX_PICKLE_OPCODES}); refusing to parse"
            )));
        }
        let op = r.u8()?;
        match op {
            0x80 => {
                r.u8()?;
            } // PROTO
            0x95 => {
                r.take(8)?;
            } // FRAME
            b'.' => break, // STOP
            b'c' => {
                let module = r.line()?;
                let name = r.line()?;
                stack.push(Val::Global(format!("{module} {name}")));
            }
            0x93 => {
                let name = pop(&mut stack)?;
                let module = pop(&mut stack)?;
                if let (Val::Str(m), Val::Str(n)) = (module, name) {
                    stack.push(Val::Global(format!("{m} {n}")));
                } else {
                    return Err(Error::Msg("pickle: STACK_GLOBAL non-string".into()));
                }
            }
            b'}' => stack.push(Val::Dict(Vec::new())),
            b')' => stack.push(Val::Tuple(Vec::new())),
            b']' => stack.push(Val::List(Vec::new())),
            b'(' => stack.push(Val::Mark),
            b'N' => stack.push(Val::None),
            0x88 => stack.push(Val::Bool(true)),
            0x89 => stack.push(Val::Bool(false)),
            b'J' => stack.push(Val::Int(r.i32le()? as i64)),
            b'K' => stack.push(Val::Int(r.u8()? as i64)),
            b'M' => stack.push(Val::Int(r.u16le()? as i64)),
            b'G' => stack.push(Val::Float(r.f64be()?)), // BINFLOAT
            0x8a => {
                // LONG1. Reject n > 8: `(b as i64) << (8 * i)` would shift by >= 64 (F-016).
                let n = r.u8()? as usize;
                if n > 8 {
                    return Err(Error::Msg(format!(
                        "pickle: LONG1 length {n} exceeds 8 bytes (i64)"
                    )));
                }
                let bytes = r.take(n)?;
                let mut v: i64 = 0;
                for (i, &b) in bytes.iter().enumerate() {
                    v |= (b as i64) << (8 * i);
                }
                if n > 0 && bytes[n - 1] & 0x80 != 0 && n < 8 {
                    v -= 1i64 << (8 * n);
                }
                stack.push(Val::Int(v));
            }
            b'X' => {
                let n = r.u32le()? as usize;
                let s = std::str::from_utf8(r.take(n)?)
                    .map_err(|e| Error::Msg(format!("pickle: bad utf8 string: {e}")))?
                    .to_string();
                stack.push(Val::Str(s));
            }
            0x8c => {
                let n = r.u8()? as usize;
                let s = std::str::from_utf8(r.take(n)?)
                    .map_err(|e| Error::Msg(format!("pickle: bad utf8 string: {e}")))?
                    .to_string();
                stack.push(Val::Str(s));
            }
            b'q' | b'r' => {
                let i = if op == b'q' {
                    r.u8()? as u32
                } else {
                    r.u32le()?
                };
                let top = stack
                    .last()
                    .ok_or_else(|| Error::Msg("pickle: BINPUT empty".into()))?;
                charge_nodes(&mut cloned_nodes, count_nodes(top))?;
                memo.insert(i, top.clone());
            }
            b'h' | b'j' => {
                let i = if op == b'h' {
                    r.u8()? as u32
                } else {
                    r.u32le()?
                };
                let v = memo
                    .get(&i)
                    .ok_or_else(|| Error::Msg("pickle: BINGET miss".into()))?;
                charge_nodes(&mut cloned_nodes, count_nodes(v))?;
                stack.push(v.clone());
            }
            0x85 => {
                let a = pop(&mut stack)?;
                stack.push(Val::Tuple(vec![a]));
            }
            0x86 => {
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Val::Tuple(vec![a, b]));
            }
            0x87 => {
                let c = pop(&mut stack)?;
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Val::Tuple(vec![a, b, c]));
            }
            b't' => {
                let items = pop_to_mark(&mut stack)?;
                stack.push(Val::Tuple(items));
            }
            b'a' => {
                // APPEND: list, value
                let value = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Val::List(items)) => items.push(value),
                    _ => return Err(Error::Msg("pickle: APPEND target not a list".into())),
                }
            }
            b'e' => {
                // APPENDS: list, MARK, v1, … → list
                let values = pop_to_mark(&mut stack)?;
                match stack.last_mut() {
                    Some(Val::List(items)) => items.extend(values),
                    _ => return Err(Error::Msg("pickle: APPENDS target not a list".into())),
                }
            }
            b'Q' => {
                let pid = pop(&mut stack)?;
                stack.push(resolve_persid(pid)?);
            }
            b'R' => {
                let args = pop(&mut stack)?;
                let callable = pop(&mut stack)?;
                stack.push(apply_reduce(callable, args)?);
            }
            b's' => {
                let value = pop(&mut stack)?;
                let key = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Val::Dict(items)) => items.push((key, value)),
                    _ => return Err(Error::Msg("pickle: SETITEM target not a dict".into())),
                }
            }
            b'u' => {
                let pairs = pop_to_mark(&mut stack)?;
                if pairs.len() % 2 != 0 {
                    return Err(Error::Msg("pickle: SETITEMS odd count".into()));
                }
                match stack.last_mut() {
                    Some(Val::Dict(items)) => {
                        let mut it = pairs.into_iter();
                        while let (Some(k), Some(v)) = (it.next(), it.next()) {
                            items.push((k, v));
                        }
                    }
                    _ => return Err(Error::Msg("pickle: SETITEMS target not a dict".into())),
                }
            }
            b'b' => {
                // BUILD: obj, state → obj. The state is an OrderedDict's instance `__dict__`
                // (e.g. `_metadata`), never tensor data, so it is discarded.
                let _state = pop(&mut stack)?;
            }
            other => {
                return Err(Error::Msg(format!(
                    "pickle: unsupported opcode 0x{other:02x} at byte {} (torch.save surface only)",
                    r.pos - 1
                )));
            }
        }
    }
    stack
        .pop()
        .ok_or_else(|| Error::Msg("pickle: empty stack at STOP".into()))
}

fn resolve_persid(pid: Val) -> Result<Val> {
    let Val::Tuple(t) = pid else {
        return Err(Error::Msg("pickle: persid not a tuple".into()));
    };
    if t.len() < 3 {
        return Err(Error::Msg("pickle: persid tuple too short".into()));
    }
    let dtype = match &t[1] {
        Val::Global(g) => PthDtype::from_global(g)?,
        _ => {
            return Err(Error::Msg(
                "pickle: persid storage type not a global".into(),
            ))
        }
    };
    let key = match &t[2] {
        Val::Str(s) => s.clone(),
        Val::Int(n) => n.to_string(),
        _ => return Err(Error::Msg("pickle: persid storage key not a string".into())),
    };
    Ok(Val::Storage { dtype, key })
}

fn apply_reduce(callable: Val, args: Val) -> Result<Val> {
    let Val::Global(g) = callable else {
        return Err(Error::Msg("pickle: REDUCE callable not a global".into()));
    };
    match g.as_str() {
        "collections OrderedDict" => Ok(Val::Dict(Vec::new())),
        "torch._utils _rebuild_tensor_v2" => {
            let Val::Tuple(a) = args else {
                return Err(Error::Msg(
                    "pickle: _rebuild_tensor_v2 args not a tuple".into(),
                ));
            };
            if a.len() < 4 {
                return Err(Error::Msg("pickle: _rebuild_tensor_v2 too few args".into()));
            }
            let (dtype, key) = match &a[0] {
                Val::Storage { dtype, key } => (*dtype, key.clone()),
                _ => {
                    return Err(Error::Msg(
                        "pickle: rebuild_tensor arg0 not a storage".into(),
                    ))
                }
            };
            let offset = match &a[1] {
                Val::Int(n) => *n,
                _ => return Err(Error::Msg("pickle: rebuild_tensor offset not int".into())),
            };
            Ok(Val::Tensor {
                dtype,
                key,
                offset,
                size: int_tuple(&a[2])?,
                stride: int_tuple(&a[3])?,
            })
        }
        other => Err(Error::Msg(format!(
            "pickle: unsupported REDUCE callable `{other}`"
        ))),
    }
}

fn int_tuple(v: &Val) -> Result<Vec<i64>> {
    let Val::Tuple(t) = v else {
        return Err(Error::Msg("pickle: expected an int tuple".into()));
    };
    t.iter()
        .map(|e| match e {
            Val::Int(n) => Ok(*n),
            _ => Err(Error::Msg("pickle: non-int in shape/stride tuple".into())),
        })
        .collect()
}

fn to_value(v: Val) -> Result<PthValue> {
    Ok(match v {
        Val::Int(n) => PthValue::Int(n),
        Val::Float(f) => PthValue::Float(f),
        Val::Str(s) => PthValue::Str(s),
        Val::Bool(b) => PthValue::Bool(b),
        Val::None => PthValue::None,
        Val::Tuple(items) | Val::List(items) => {
            PthValue::Seq(items.into_iter().map(to_value).collect::<Result<_>>()?)
        }
        Val::Dict(items) => PthValue::Dict(
            items
                .into_iter()
                .map(|(k, v)| {
                    let key = match k {
                        Val::Str(s) => s,
                        Val::Int(n) => n.to_string(),
                        other => {
                            return Err(Error::Msg(format!(
                                "pickle: unsupported dict key {other:?}"
                            )))
                        }
                    };
                    Ok((key, to_value(v)?))
                })
                .collect::<Result<_>>()?,
        ),
        Val::Tensor {
            dtype,
            key,
            offset,
            size,
            stride,
        } => PthValue::Tensor(PthTensorSpec {
            dtype,
            shape: size,
            storage_key: key,
            offset,
            stride,
        }),
        Val::Mark | Val::Global(_) | Val::Storage { .. } => {
            return Err(Error::Msg(format!(
                "pickle: unexpected bare value {v:?} in the object graph"
            )))
        }
    })
}

/// Whether `stride` is the C-contiguous stride for `size` (size-1 axes may carry any stride).
fn is_c_contiguous(size: &[i64], stride: &[i64]) -> bool {
    if size.len() != stride.len() {
        return false;
    }
    let mut expected: i64 = 1;
    for i in (0..size.len()).rev() {
        if size[i] == 1 {
            continue;
        }
        if stride[i] != expected {
            return false;
        }
        expected *= size[i];
    }
    true
}

fn decode_to_f32(bytes: &[u8], dtype: PthDtype, numel: usize) -> Result<Vec<f32>> {
    let need = numel * dtype.elem_size();
    if bytes.len() < need {
        return Err(Error::Msg(format!(
            "storage blob too small: have {} bytes, need {} ({numel} × {})",
            bytes.len(),
            need,
            dtype.elem_size()
        )));
    }
    let out = match dtype {
        PthDtype::F32 => bytes[..need]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        PthDtype::BF16 => bytes[..need]
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        PthDtype::F16 => bytes[..need]
            .chunks_exact(2)
            .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        PthDtype::I64 => bytes[..need]
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
    };
    Ok(out)
}

/// IEEE-754 half → f32: the shared [`crate::numeric`] decoder, re-exported at its historical path
/// (sc-27419).
pub use crate::numeric::f16_bits_to_f32;

/// Validate externally supplied tensor metadata and compute the contiguous `[start..end)` byte range
/// and element count. Every dimension must be non-negative and fit an `i32` axis (so a backend's
/// shape cast cannot truncate), the offset non-negative, and all products checked.
fn tensor_byte_range(
    name: &str,
    offset: i64,
    size: &[i64],
    elsize: usize,
) -> Result<(usize, usize, usize)> {
    if offset < 0 {
        return Err(Error::Msg(format!(
            "{name}: negative storage offset {offset}"
        )));
    }
    let mut numel: usize = 1;
    for &d in size {
        if d < 0 || d > i32::MAX as i64 {
            return Err(Error::Msg(format!(
                "{name}: invalid tensor dimension {d} in shape {size:?} (must be 0..={})",
                i32::MAX
            )));
        }
        numel = numel.checked_mul(d as usize).ok_or_else(|| {
            Error::Msg(format!(
                "{name}: element count overflows usize for shape {size:?}"
            ))
        })?;
    }
    let start = (offset as usize).checked_mul(elsize).ok_or_else(|| {
        Error::Msg(format!(
            "{name}: storage byte offset overflows usize (offset {offset}, elsize {elsize})"
        ))
    })?;
    let byte_len = numel.checked_mul(elsize).ok_or_else(|| {
        Error::Msg(format!(
            "{name}: storage byte length overflows usize (numel {numel}, elsize {elsize})"
        ))
    })?;
    let end = start
        .checked_add(byte_len)
        .ok_or_else(|| Error::Msg(format!("{name}: storage slice end overflows usize")))?;
    Ok((start, end, numel))
}

/// Enforce STORED-only entries (F-017): a DEFLATE entry could inflate a tiny archive to many GB.
fn require_stored(method: zip::CompressionMethod, who: &str) -> Result<()> {
    if method != zip::CompressionMethod::Stored {
        return Err(Error::Msg(format!(
            "{who}: zip entry uses {method:?} compression, but only STORED (uncompressed) entries \
             are supported — a compressed entry could inflate a tiny archive to many GB \
             (decompression-bomb DoS); re-export the checkpoint without zip compression"
        )));
    }
    Ok(())
}

fn read_entry(zip: &mut zip::ZipArchive<std::fs::File>, name: &str, who: &str) -> Result<Vec<u8>> {
    let mut entry = zip
        .by_name(name)
        .map_err(|e| Error::Msg(format!("read {name} for {who}: {e}")))?;
    require_stored(entry.compression(), name)?;
    let mut out = Vec::new();
    // F-071: a crafted header must not abort the process on allocation.
    out.try_reserve_exact(entry.compressed_size() as usize)
        .map_err(|_| Error::Msg(format!("read {name}: entry too large to reserve")))?;
    entry.read_to_end(&mut out)?;
    Ok(out)
}

/// The top-level `(name, tensor)` entries of a `torch.save(state_dict)` object graph. Errors unless
/// the top object is a dict; skips non-tensor values (e.g. a nested `_metadata`-style dict).
pub fn state_dict_entries(root: &PthValue) -> Result<Vec<(String, PthTensorSpec)>> {
    let PthValue::Dict(items) = root else {
        return Err(Error::Msg(
            "pickle: top object is not a dict (expected a torch.save state dict)".into(),
        ));
    };
    Ok(items
        .iter()
        .filter_map(|(name, value)| match value {
            PthValue::Tensor(spec) => Some((name.clone(), spec.clone())),
            _ => None,
        })
        .collect())
}

/// An opened `torch.save` archive: the parsed object graph plus lazy access to storage blobs.
pub struct PthArchive {
    path: PathBuf,
    zip: zip::ZipArchive<std::fs::File>,
    prefix: String,
    root: PthValue,
}

impl PthArchive {
    /// Open the archive and parse its `data.pkl`. No storage blob is read.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path)
            .map_err(|e| Error::Msg(format!("open {}: {e}", path.display())))?;
        let mut zip = zip::ZipArchive::new(file)
            .map_err(|e| Error::Msg(format!("read zip {}: {e}", path.display())))?;
        let names: Vec<String> = zip
            .file_names()
            .map(|name| name.map(|name| name.into_owned()))
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| Error::Msg(format!("read zip names {}: {e}", path.display())))?;
        let pkl_name = names
            .iter()
            .find(|n| n.ends_with("data.pkl"))
            .ok_or_else(|| Error::Msg(format!("{}: no data.pkl in archive", path.display())))?
            .clone();
        let prefix = pkl_name.strip_suffix("data.pkl").unwrap().to_string();
        let pkl = read_entry(&mut zip, &pkl_name, "the pickle")?;
        let root = to_value(parse_pickle(&pkl)?)?;
        Ok(Self {
            path,
            zip,
            prefix,
            root,
        })
    }

    /// The archive's top object.
    pub fn root(&self) -> &PthValue {
        &self.root
    }

    /// The archive path (diagnostics).
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn read_blob(&mut self, key: &str, who: &str) -> Result<Vec<u8>> {
        let name = format!("{}data/{key}", self.prefix);
        read_entry(&mut self.zip, &name, who)
    }

    fn slice(spec: &PthTensorSpec, name: &str, blob: &[u8]) -> Result<PthTensor> {
        let (start, end, numel) =
            tensor_byte_range(name, spec.offset, &spec.shape, spec.dtype.elem_size())?;
        if !is_c_contiguous(&spec.shape, &spec.stride) {
            return Err(Error::Msg(format!(
                "{name}: non-contiguous storage stride {:?} for size {:?} unsupported",
                spec.stride, spec.shape
            )));
        }
        if end > blob.len() {
            return Err(Error::Msg(format!(
                "{name}: storage slice [{start}..{end}] exceeds blob len {} (offset {}, numel {numel})",
                blob.len(),
                spec.offset
            )));
        }
        Ok(PthTensor {
            dtype: spec.dtype,
            shape: spec.shape.clone(),
            data: blob[start..end].to_vec(),
        })
    }

    /// Read one tensor's contiguous bytes.
    pub fn read_tensor(&mut self, name: &str, spec: &PthTensorSpec) -> Result<PthTensor> {
        // Validate before touching the archive so a malformed spec never triggers a read.
        tensor_byte_range(name, spec.offset, &spec.shape, spec.dtype.elem_size())?;
        let blob = self.read_blob(&spec.storage_key, name)?;
        Self::slice(spec, name, &blob)
    }

    /// Visit every tensor whose flattened name passes `keep`, in pickle order. Each storage blob is
    /// read once and released after its last kept view (F-018) — unkept tensors' blobs are never read,
    /// so a multi-tower checkpoint loads one tower without materializing the rest.
    pub fn visit_tensors(
        &mut self,
        keep: impl Fn(&str) -> bool,
        visit: impl FnMut(String, PthTensor) -> Result<()>,
    ) -> Result<()> {
        let specs: Vec<(String, PthTensorSpec)> = self
            .root
            .flatten_tensors()
            .into_iter()
            .filter(|(name, _)| keep(name))
            .map(|(name, spec)| (name, spec.clone()))
            .collect();
        self.visit_specs(specs, visit)
    }

    /// The flat `torch.save(state_dict)` view: the top object must be a dict, and only its
    /// **top-level** tensor values are visited under their own keys (non-tensor and nested values
    /// are skipped) — exactly the pre-sc-24213 `mlx_gen::pth` contract the Wan and E-LatentLPIPS
    /// converters were written against.
    pub fn visit_state_dict(
        &mut self,
        keep: impl Fn(&str) -> bool,
        visit: impl FnMut(String, PthTensor) -> Result<()>,
    ) -> Result<()> {
        let specs = state_dict_entries(&self.root)?
            .into_iter()
            .filter(|(name, _)| keep(name))
            .collect();
        self.visit_specs(specs, visit)
    }

    fn visit_specs(
        &mut self,
        specs: Vec<(String, PthTensorSpec)>,
        mut visit: impl FnMut(String, PthTensor) -> Result<()>,
    ) -> Result<()> {
        let mut remaining: HashMap<String, usize> = HashMap::new();
        for (_, spec) in &specs {
            *remaining.entry(spec.storage_key.clone()).or_insert(0) += 1;
        }
        let mut cache: HashMap<String, Vec<u8>> = HashMap::new();
        for (name, spec) in specs {
            tensor_byte_range(&name, spec.offset, &spec.shape, spec.dtype.elem_size())?;
            if !cache.contains_key(&spec.storage_key) {
                let blob = self.read_blob(&spec.storage_key, &name)?;
                cache.insert(spec.storage_key.clone(), blob);
            }
            let tensor = Self::slice(&spec, &name, &cache[&spec.storage_key])?;
            if let Some(refs) = remaining.get_mut(&spec.storage_key) {
                *refs -= 1;
                if *refs == 0 {
                    cache.remove(&spec.storage_key);
                }
            }
            visit(name, tensor)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn require_stored_rejects_compressed_entries() {
        assert!(require_stored(zip::CompressionMethod::Stored, "x").is_ok());
        let err = require_stored(zip::CompressionMethod::DEFLATE, "data.pkl")
            .unwrap_err()
            .to_string();
        assert!(err.contains("STORED"), "got: {err}");
        assert!(err.contains("decompression-bomb"), "got: {err}");
    }

    #[test]
    fn f16_round_trip_known_values() {
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        assert_eq!(f16_bits_to_f32(0x4000), 2.0);
        assert_eq!(f16_bits_to_f32(0xB800), -0.5);
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
    }

    #[test]
    fn decodes_bf16_f32_and_i64() {
        let v = decode_to_f32(&[0x80, 0x3F, 0x00, 0xC0], PthDtype::BF16, 2).unwrap();
        assert_eq!(v, vec![1.0, -2.0]);
        let v = decode_to_f32(&1.5f32.to_le_bytes(), PthDtype::F32, 1).unwrap();
        assert_eq!(v, vec![1.5]);
        let v = decode_to_f32(&(-3i64).to_le_bytes(), PthDtype::I64, 1).unwrap();
        assert_eq!(v, vec![-3.0]);
    }

    #[test]
    fn tensor_byte_range_validates_metadata() {
        assert_eq!(tensor_byte_range("t", 2, &[3, 4], 4).unwrap(), (8, 56, 12));
        assert_eq!(tensor_byte_range("t", 0, &[], 4).unwrap(), (0, 4, 1));
        assert_eq!(tensor_byte_range("t", 0, &[0, 5], 4).unwrap(), (0, 0, 0));
        let err = tensor_byte_range("t", -1, &[2], 4).unwrap_err().to_string();
        assert!(err.contains("negative storage offset"), "{err}");
        let neg = tensor_byte_range("t", 0, &[-1, 2], 4)
            .unwrap_err()
            .to_string();
        assert!(neg.contains("invalid tensor dimension"), "{neg}");
        let big = tensor_byte_range("t", 0, &[i32::MAX as i64 + 1], 4)
            .unwrap_err()
            .to_string();
        assert!(big.contains("invalid tensor dimension"), "{big}");
        let m = i32::MAX as i64;
        let numel = tensor_byte_range("t", 0, &[m, m, m], 4)
            .unwrap_err()
            .to_string();
        assert!(numel.contains("element count overflows usize"), "{numel}");
        let off = tensor_byte_range("t", i64::MAX, &[1], 4)
            .unwrap_err()
            .to_string();
        assert!(off.contains("storage byte offset overflows usize"), "{off}");
    }

    fn push_str(p: &mut Vec<u8>, s: &str) {
        p.push(b'X');
        p.extend_from_slice(&(s.len() as u32).to_le_bytes());
        p.extend_from_slice(s.as_bytes());
    }

    /// `_rebuild_tensor_v2(('storage', FloatStorage, '0', 'cpu', 2), 0, (2,), (1,), False, {})`.
    fn push_tensor(p: &mut Vec<u8>) {
        p.push(b'c');
        p.extend_from_slice(b"torch._utils\n_rebuild_tensor_v2\n");
        p.push(b'(');
        p.push(b'(');
        push_str(p, "storage");
        p.push(b'c');
        p.extend_from_slice(b"torch\nFloatStorage\n");
        push_str(p, "0");
        push_str(p, "cpu");
        p.extend_from_slice(&[b'K', 2, b't', b'Q', b'K', 0, b'K', 2, 0x85, b'K', 1, 0x85]);
        p.extend_from_slice(&[0x89, b'}', b't', b'R']);
    }

    #[test]
    fn parse_minimal_state_dict() {
        let mut p: Vec<u8> = vec![0x80, 2, b'c'];
        p.extend_from_slice(b"collections\nOrderedDict\n");
        p.extend_from_slice(b")R(");
        push_str(&mut p, "a");
        push_tensor(&mut p);
        p.extend_from_slice(b"u.");
        let root = to_value(parse_pickle(&p).unwrap()).unwrap();
        let flat = root.flatten_tensors();
        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].0, "a");
        assert_eq!(flat[0].1.dtype, PthDtype::F32);
        assert_eq!(flat[0].1.shape, vec![2]);
        assert_eq!(flat[0].1.stride, vec![1]);
    }

    /// sc-24213: a nested `{"model": {"w": T}, "config": {"patch": 2, "lr": 0.5, "names": ["x"]}}`
    /// keeps its structure, scalars, floats and lists, and flattens tensor names by dotted path.
    #[test]
    fn parse_nested_checkpoint_with_scalars_and_lists() {
        let mut p: Vec<u8> = vec![0x80, 2, b'}', b'('];
        push_str(&mut p, "model");
        p.extend_from_slice(b"}(");
        push_str(&mut p, "w");
        push_tensor(&mut p);
        p.push(b'u');
        push_str(&mut p, "config");
        p.extend_from_slice(b"}(");
        push_str(&mut p, "patch");
        p.extend_from_slice(&[b'K', 2]);
        push_str(&mut p, "lr");
        p.push(b'G');
        p.extend_from_slice(&0.5f64.to_be_bytes());
        push_str(&mut p, "names");
        p.extend_from_slice(b"](");
        push_str(&mut p, "x");
        p.extend_from_slice(b"euu.");
        let root = to_value(parse_pickle(&p).unwrap()).unwrap();
        let config = root.get("config").unwrap();
        assert_eq!(config.get("patch"), Some(&PthValue::Int(2)));
        assert_eq!(config.get("lr"), Some(&PthValue::Float(0.5)));
        assert_eq!(
            config.get("names"),
            Some(&PthValue::Seq(vec![PthValue::Str("x".into())]))
        );
        let flat = root.flatten_tensors();
        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].0, "model.w");
    }

    /// The flat state-dict view (mlx-gen's `load_pth_f32*` adapter) keeps the pre-sc-24213
    /// contract: a non-dict top object is refused, and only top-level tensors are visited — a
    /// nested dict's tensors never appear under dotted names.
    #[test]
    fn state_dict_view_refuses_non_dict_roots_and_skips_nested_tensors() {
        // Root = a bare tensor.
        let mut p: Vec<u8> = vec![0x80, 2];
        push_tensor(&mut p);
        p.push(b'.');
        let root = to_value(parse_pickle(&p).unwrap()).unwrap();
        let err = state_dict_entries(&root).unwrap_err().to_string();
        assert!(err.contains("top object is not a dict"), "{err}");
        // Root = a list.
        let list = to_value(parse_pickle(&[0x80, 2, b']', b'.']).unwrap()).unwrap();
        assert!(state_dict_entries(&list).is_err());

        // {"a": T, "nested": {"w": T}, "n": 3} → only "a".
        let mut p: Vec<u8> = vec![0x80, 2, b'}', b'('];
        push_str(&mut p, "a");
        push_tensor(&mut p);
        push_str(&mut p, "nested");
        p.extend_from_slice(b"}(");
        push_str(&mut p, "w");
        push_tensor(&mut p);
        p.push(b'u');
        push_str(&mut p, "n");
        p.extend_from_slice(&[b'K', 3]);
        p.extend_from_slice(b"u.");
        let root = to_value(parse_pickle(&p).unwrap()).unwrap();
        let names: Vec<String> = state_dict_entries(&root)
            .unwrap()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(names, vec!["a".to_string()]);
        // The general nested view still sees both.
        assert_eq!(root.flatten_tensors().len(), 2);
    }

    /// On a real nested archive (the Supra2 tiny checkpoint: `{"model": {…}, "config": {…}}`) the
    /// flat state-dict view mlx-gen's `load_pth_f32*` uses visits no tensors — as before sc-24213,
    /// nested values are skipped, never surfaced under dotted names — while the nested view sees all.
    #[test]
    fn archive_state_dict_view_never_surfaces_nested_tensors() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/supra2/tiny_model_with_uncond.pt");
        let mut flat = Vec::new();
        PthArchive::open(&path)
            .unwrap()
            .visit_state_dict(
                |_| true,
                |name, _| {
                    flat.push(name);
                    Ok(())
                },
            )
            .unwrap();
        assert!(flat.is_empty(), "{flat:?}");
        let mut nested = Vec::new();
        PthArchive::open(&path)
            .unwrap()
            .visit_tensors(
                |_| true,
                |name, _| {
                    nested.push(name);
                    Ok(())
                },
            )
            .unwrap();
        assert!(nested.contains(&"model.pos_embed".to_string()));
        assert!(nested.contains(&"config.uncond_text".to_string()));
    }

    #[test]
    fn parse_rejects_oversized_long1() {
        let mut p: Vec<u8> = vec![0x80, 2, 0x8a, 9];
        p.extend_from_slice(&[0u8; 9]);
        p.push(b'.');
        let err = parse_pickle(&p).unwrap_err().to_string();
        assert!(err.contains("LONG1 length 9 exceeds 8 bytes"), "{err}");
    }

    #[test]
    fn count_nodes_counts_nested_structure() {
        assert_eq!(count_nodes(&Val::Int(7)), 1);
        let nested = Val::Tuple(vec![
            Val::Tuple(vec![Val::Int(1), Val::Int(2)]),
            Val::Int(3),
        ]);
        assert_eq!(count_nodes(&nested), 5);
    }

    #[test]
    fn parse_rejects_billion_laughs_amplification() {
        let mut p: Vec<u8> = vec![0x80, 2, b')', b'q', 0];
        for _ in 0..40 {
            p.extend_from_slice(&[b'h', 0, b'h', 0, 0x86, b'q', 0]);
        }
        p.push(b'.');
        let err = parse_pickle(&p).unwrap_err().to_string();
        assert!(err.contains("node budget exceeded"), "{err}");
    }

    #[test]
    fn parse_rejects_arbitrary_callables() {
        let mut p: Vec<u8> = vec![0x80, 2, b'c'];
        p.extend_from_slice(b"os\nsystem\n");
        push_str(&mut p, "echo");
        p.extend_from_slice(&[0x85, b'R', b'.']);
        let err = parse_pickle(&p).unwrap_err().to_string();
        assert!(
            err.contains("unsupported REDUCE callable `os system`"),
            "{err}"
        );
    }

    #[test]
    fn c_contiguous_check() {
        assert!(is_c_contiguous(&[160, 12, 3, 3, 3], &[324, 27, 9, 3, 1]));
        assert!(is_c_contiguous(&[4, 5], &[5, 1]));
        assert!(is_c_contiguous(&[3, 1, 1, 1], &[1, 3, 3, 3]));
        assert!(!is_c_contiguous(&[4, 5], &[1, 4]));
        assert!(is_c_contiguous(&[], &[]));
    }

    /// The committed `torch.save`d BatchNorm state dict (sc-24833's fixture) reads with its
    /// float tensors, the int64 `num_batches_tracked` counter and the discarded `_metadata` state.
    #[test]
    fn reads_a_torch_state_dict_with_batchnorm_counters() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/latent_perceptual/tiny_bn_state_dict.pth");
        let mut archive = PthArchive::open(path).unwrap();
        let mut seen = Vec::new();
        archive
            .visit_tensors(
                |_| true,
                |name, t| {
                    seen.push((name, t.dtype, t.shape.clone()));
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(seen.len(), 7, "{seen:?}");
        assert!(seen.contains(&("1.num_batches_tracked".into(), PthDtype::I64, vec![])));
        assert!(seen.contains(&("0.weight".into(), PthDtype::F32, vec![2, 4, 3, 3])));
    }
}
