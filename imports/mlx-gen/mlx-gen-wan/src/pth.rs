//! Minimal torch `.pth` (zip-of-pickle) reader for the native Wan converter (sc-3237 / sc-3224).
//!
//! Native Wan checkpoints ship the T5 encoder (`models_t5_umt5-xxl-enc-bf16.pth`) and the VAE
//! (`Wan2.x_VAE.pth`) as PyTorch `torch.save` archives — a ZIP holding `<prefix>/data.pkl` (a
//! pickled `OrderedDict` whose tensors are `persistent_id` storage references) plus `<prefix>/data/<n>`
//! raw storage blobs. The reference Python loads these with `torch.load(...).float()` (every tensor
//! → f32), so [`load_pth_f32`] mirrors that exactly: it returns each tensor as an **f32** MLX
//! [`Array`] in PyTorch layout (the converter's sanitizers then transpose conv weights to
//! channels-last and cast per component).
//!
//! Scope is deliberately narrow — exactly the opcode set `torch.save` emits (protocol 2/4, STORED
//! zip entries) and the three globals it references (`collections.OrderedDict`,
//! `torch.FloatStorage`/`BFloat16Storage`/`HalfStorage`, `torch._utils._rebuild_tensor_v2`). The
//! `zip` crate handles the archive (incl. data-descriptor entries and zip64 for the >4 GB T5).
//! Vendored rather than pulling in candle-core (a whole second tensor framework) for one file read.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use mlx_gen::{Error, Result};
use mlx_rs::Array;

/// Torch storage element type (the `torch.<X>Storage` global in the pickle). We only need the float
/// storages the Wan T5/VAE use; everything is decoded to f32 (mirroring `.float()`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StorageDtype {
    Float32,
    BFloat16,
    Float16,
}

impl StorageDtype {
    fn from_global(name: &str) -> Result<Self> {
        match name {
            "torch FloatStorage" => Ok(StorageDtype::Float32),
            "torch BFloat16Storage" => Ok(StorageDtype::BFloat16),
            "torch HalfStorage" => Ok(StorageDtype::Float16),
            other => Err(Error::Msg(format!(
                "unsupported torch storage type `{other}` in .pth (expected Float/BFloat16/Half)"
            ))),
        }
    }

    fn elem_size(self) -> usize {
        match self {
            StorageDtype::Float32 => 4,
            StorageDtype::BFloat16 | StorageDtype::Float16 => 2,
        }
    }
}

/// A pickle stack value. Only the variants `torch.save` produces are modeled.
#[derive(Clone, Debug)]
enum Val {
    Int(i64),
    Str(String),
    /// `requires_grad` (NEWTRUE/NEWFALSE) — a stack placeholder; the value is never read.
    Bool,
    None,
    Mark,
    Global(String),
    Tuple(Vec<Val>),
    Dict(Vec<(Val, Val)>),
    /// A resolved `persistent_id` storage reference: `('storage', <dtype>, <key>, <loc>, <numel>)`.
    Storage {
        dtype: StorageDtype,
        key: String,
    },
    /// The result of `_rebuild_tensor_v2(storage, offset, size, stride, …)`.
    Tensor {
        dtype: StorageDtype,
        key: String,
        offset: i64,
        size: Vec<i64>,
    },
}

/// A cursor over the pickle byte stream.
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
    /// Read a `\n`-terminated line (for the `GLOBAL` opcode's module / name).
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
        self.pos += 1; // skip '\n'
        Ok(s)
    }
}

/// Run the pickle VM over `data.pkl`, returning the `(name, Tensor-spec)` entries of the top dict.
fn parse_pickle(data: &[u8]) -> Result<Vec<(String, Val)>> {
    let mut r = Reader { buf: data, pos: 0 };
    let mut stack: Vec<Val> = Vec::new();
    let mut memo: HashMap<u32, Val> = HashMap::new();

    let pop = |st: &mut Vec<Val>| {
        st.pop()
            .ok_or_else(|| Error::Msg("pickle: stack underflow".into()))
    };
    // Collect items back to the topmost Mark (exclusive), restoring original order.
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
                // GLOBAL module '\n' name '\n'
                let module = r.line()?;
                let name = r.line()?;
                stack.push(Val::Global(format!("{module} {name}")));
            }
            0x93 => {
                // STACK_GLOBAL: name, module on stack
                let name = pop(&mut stack)?;
                let module = pop(&mut stack)?;
                if let (Val::Str(m), Val::Str(n)) = (module, name) {
                    stack.push(Val::Global(format!("{m} {n}")));
                } else {
                    return Err(Error::Msg("pickle: STACK_GLOBAL non-string".into()));
                }
            }
            b'}' => stack.push(Val::Dict(Vec::new())), // EMPTY_DICT
            b')' => stack.push(Val::Tuple(Vec::new())), // EMPTY_TUPLE
            b']' => stack.push(Val::Tuple(Vec::new())), // EMPTY_LIST (modeled as tuple; unused)
            b'(' => stack.push(Val::Mark),             // MARK
            b'N' => stack.push(Val::None),             // NONE
            0x88 => stack.push(Val::Bool),             // NEWTRUE
            0x89 => stack.push(Val::Bool),             // NEWFALSE
            b'J' => stack.push(Val::Int(r.i32le()? as i64)), // BININT
            b'K' => stack.push(Val::Int(r.u8()? as i64)), // BININT1
            b'M' => stack.push(Val::Int(r.u16le()? as i64)), // BININT2
            0x8a => {
                // LONG1: 1-byte length, then little-endian signed
                let n = r.u8()? as usize;
                let bytes = r.take(n)?;
                let mut v: i64 = 0;
                for (i, &b) in bytes.iter().enumerate() {
                    v |= (b as i64) << (8 * i);
                }
                if n > 0 && bytes[n - 1] & 0x80 != 0 && n < 8 {
                    v -= 1i64 << (8 * n); // sign-extend
                }
                stack.push(Val::Int(v));
            }
            b'X' => {
                // BINUNICODE: u32 length + utf8
                let n = r.u32le()? as usize;
                let s = std::str::from_utf8(r.take(n)?)
                    .map_err(|e| Error::Msg(format!("pickle: bad utf8 string: {e}")))?
                    .to_string();
                stack.push(Val::Str(s));
            }
            0x8c => {
                // SHORT_BINUNICODE: u8 length + utf8
                let n = r.u8()? as usize;
                let s = std::str::from_utf8(r.take(n)?)
                    .map_err(|e| Error::Msg(format!("pickle: bad utf8 string: {e}")))?
                    .to_string();
                stack.push(Val::Str(s));
            }
            b'q' => {
                let i = r.u8()? as u32;
                memo.insert(
                    i,
                    stack
                        .last()
                        .cloned()
                        .ok_or_else(|| Error::Msg("pickle: BINPUT empty".into()))?,
                );
            }
            b'r' => {
                let i = r.u32le()?;
                memo.insert(
                    i,
                    stack
                        .last()
                        .cloned()
                        .ok_or_else(|| Error::Msg("pickle: LONG_BINPUT empty".into()))?,
                );
            }
            b'h' => {
                let i = r.u8()? as u32;
                stack.push(
                    memo.get(&i)
                        .cloned()
                        .ok_or_else(|| Error::Msg("pickle: BINGET miss".into()))?,
                );
            }
            b'j' => {
                let i = r.u32le()?;
                stack.push(
                    memo.get(&i)
                        .cloned()
                        .ok_or_else(|| Error::Msg("pickle: LONG_BINGET miss".into()))?,
                );
            }
            0x85 => {
                let a = pop(&mut stack)?;
                stack.push(Val::Tuple(vec![a]));
            } // TUPLE1
            0x86 => {
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Val::Tuple(vec![a, b]));
            } // TUPLE2
            0x87 => {
                let c = pop(&mut stack)?;
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Val::Tuple(vec![a, b, c]));
            } // TUPLE3
            b't' => {
                let items = pop_to_mark(&mut stack)?;
                stack.push(Val::Tuple(items));
            } // TUPLE
            b'Q' => {
                // BINPERSID: pop the storage tuple, resolve.
                let pid = pop(&mut stack)?;
                stack.push(resolve_persid(pid)?);
            }
            b'R' => {
                // REDUCE: callable, args → result
                let args = pop(&mut stack)?;
                let callable = pop(&mut stack)?;
                stack.push(apply_reduce(callable, args)?);
            }
            b's' => {
                // SETITEM: dict, key, value
                let value = pop(&mut stack)?;
                let key = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Val::Dict(items)) => items.push((key, value)),
                    _ => return Err(Error::Msg("pickle: SETITEM target not a dict".into())),
                }
            }
            b'u' => {
                // SETITEMS: dict, MARK, k1, v1, … → dict
                let pairs = pop_to_mark(&mut stack)?;
                if pairs.len() % 2 != 0 {
                    return Err(Error::Msg("pickle: SETITEMS odd count".into()));
                }
                match stack.last_mut() {
                    Some(Val::Dict(items)) => {
                        for kv in pairs.chunks_exact(2) {
                            items.push((kv[0].clone(), kv[1].clone()));
                        }
                    }
                    _ => return Err(Error::Msg("pickle: SETITEMS target not a dict".into())),
                }
            }
            b'b' => {
                // BUILD: obj, state → obj  (we discard state: OrderedDict items arrive via SETITEMS,
                // state is the empty instance __dict__).
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

    // The top object is the OrderedDict of {name: tensor}.
    let top = stack
        .pop()
        .ok_or_else(|| Error::Msg("pickle: empty stack at STOP".into()))?;
    let Val::Dict(items) = top else {
        return Err(Error::Msg("pickle: top object is not a dict".into()));
    };
    let mut out = Vec::with_capacity(items.len());
    for (k, v) in items {
        let Val::Str(name) = k else {
            return Err(Error::Msg("pickle: non-string state_dict key".into()));
        };
        out.push((name, v));
    }
    Ok(out)
}

/// `('storage', <FloatStorage global>, <key str>, <location str>, <numel int>)` → `Val::Storage`.
fn resolve_persid(pid: Val) -> Result<Val> {
    let Val::Tuple(t) = pid else {
        return Err(Error::Msg("pickle: persid not a tuple".into()));
    };
    if t.len() < 3 {
        return Err(Error::Msg("pickle: persid tuple too short".into()));
    }
    let dtype = match &t[1] {
        Val::Global(g) => StorageDtype::from_global(g)?,
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

/// Apply a `REDUCE`: only `OrderedDict()` and `_rebuild_tensor_v2(...)` are produced by `torch.save`.
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
            // (storage, storage_offset, size, stride, requires_grad, backward_hooks[, ...])
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
            let size = int_tuple(&a[2])?;
            Ok(Val::Tensor {
                dtype,
                key,
                offset,
                size,
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

/// Decode a raw little-endian storage blob (in `dtype`) of `numel` elements to an f32 vector,
/// mirroring `torch.load(...).float()`.
fn decode_to_f32(bytes: &[u8], dtype: StorageDtype, numel: usize) -> Result<Vec<f32>> {
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
        StorageDtype::Float32 => bytes[..need]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        StorageDtype::BFloat16 => bytes[..need]
            .chunks_exact(2)
            // bf16 occupies the high 16 bits of the f32 — widen by a 16-bit left shift.
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        StorageDtype::Float16 => bytes[..need]
            .chunks_exact(2)
            .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
    };
    Ok(out)
}

/// IEEE-754 half → f32 (no `half` crate dep).
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = (h & 0x8000) as u32;
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 16
        } else {
            // subnormal — normalize
            let mut e: i32 = -1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 16) | (((127 - 15 + 1 + e) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 16) | 0x7f80_0000 | (mant << 13) // inf / nan
    } else {
        (sign << 16) | (((exp as i32 - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Load a torch `.pth` checkpoint, returning every tensor as an f32 MLX [`Array`] in PyTorch layout
/// (`torch.load(...).float()` semantics). Conv-weight transposes + key renames are the caller's job.
pub fn load_pth_f32(path: impl AsRef<Path>) -> Result<HashMap<String, Array>> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .map_err(|e| Error::Msg(format!("open {}: {e}", path.display())))?;
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|e| Error::Msg(format!("read zip {}: {e}", path.display())))?;

    // Locate `<prefix>data.pkl` and derive the storage-blob prefix.
    let names: Vec<String> = zip.file_names().map(String::from).collect();
    let pkl_name = names
        .iter()
        .find(|n| n.ends_with("data.pkl"))
        .ok_or_else(|| Error::Msg(format!("{}: no data.pkl in archive", path.display())))?
        .clone();
    let prefix = pkl_name.strip_suffix("data.pkl").unwrap().to_string();

    let mut pkl_bytes = Vec::new();
    zip.by_name(&pkl_name)
        .map_err(|e| Error::Msg(format!("read {pkl_name}: {e}")))?
        .read_to_end(&mut pkl_bytes)?;

    let specs = parse_pickle(&pkl_bytes)?;

    let mut out = HashMap::with_capacity(specs.len());
    for (name, spec) in specs {
        let Val::Tensor {
            dtype,
            key,
            offset,
            size,
        } = spec
        else {
            // Non-tensor entries (rare) are skipped — the reference keeps only `torch.Tensor`s.
            continue;
        };
        let numel: usize = size.iter().product::<i64>().max(0) as usize;
        if offset != 0 {
            return Err(Error::Msg(format!(
                "{name}: non-zero storage_offset {offset} unsupported (saved weights are contiguous)"
            )));
        }
        let blob_name = format!("{prefix}data/{key}");
        let mut blob = Vec::new();
        zip.by_name(&blob_name)
            .map_err(|e| Error::Msg(format!("read storage {blob_name} for {name}: {e}")))?
            .read_to_end(&mut blob)?;
        let floats = decode_to_f32(&blob, dtype, numel)?;
        let shape: Vec<i32> = size.iter().map(|&d| d as i32).collect();
        out.insert(name, Array::from_slice(&floats, &shape));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip_known_values() {
        // 1.0 = 0x3C00, 2.0 = 0x4000, -0.5 = 0xB800, 0 = 0x0000
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        assert_eq!(f16_bits_to_f32(0x4000), 2.0);
        assert_eq!(f16_bits_to_f32(0xB800), -0.5);
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
    }

    #[test]
    fn bf16_decode_high_bits() {
        // bf16 1.0 = 0x3F80 → f32 1.0; -2.0 bf16 = 0xC000 → f32 -2.0
        let v = decode_to_f32(&[0x80, 0x3F, 0x00, 0xC0], StorageDtype::BFloat16, 2).unwrap();
        assert_eq!(v, vec![1.0, -2.0]);
    }

    #[test]
    fn f32_decode_little_endian() {
        let v = decode_to_f32(&1.5f32.to_le_bytes(), StorageDtype::Float32, 1).unwrap();
        assert_eq!(v, vec![1.5]);
    }

    /// A hand-built minimal protocol-2 pickle of `OrderedDict({"a": rebuild_tensor(FloatStorage 0,
    /// offset 0, size (2,), stride (1,), False, {})})` exercises the VM end to end.
    #[test]
    fn parse_minimal_state_dict() {
        let mut p: Vec<u8> = Vec::new();
        p.extend_from_slice(&[0x80, 2]); // PROTO 2
                                         // OrderedDict()
        p.push(b'c');
        p.extend_from_slice(b"collections\nOrderedDict\n");
        p.push(b')'); // EMPTY_TUPLE
        p.push(b'R'); // REDUCE → dict
        p.push(b'('); // MARK
        p.push(b'X'); // BINUNICODE "a"
        p.extend_from_slice(&1u32.to_le_bytes());
        p.push(b'a');
        // _rebuild_tensor_v2
        p.push(b'c');
        p.extend_from_slice(b"torch._utils\n_rebuild_tensor_v2\n");
        p.push(b'('); // MARK (args)
                      // storage tuple ('storage', FloatStorage, '0', 'cpu', 2)
        p.push(b'('); // MARK
        p.push(b'X');
        p.extend_from_slice(&7u32.to_le_bytes());
        p.extend_from_slice(b"storage");
        p.push(b'c');
        p.extend_from_slice(b"torch\nFloatStorage\n");
        p.push(b'X');
        p.extend_from_slice(&1u32.to_le_bytes());
        p.push(b'0');
        p.push(b'X');
        p.extend_from_slice(&3u32.to_le_bytes());
        p.extend_from_slice(b"cpu");
        p.push(b'K');
        p.push(2); // numel
        p.push(b't'); // TUPLE → storage tuple
        p.push(b'Q'); // BINPERSID
        p.push(b'K');
        p.push(0); // offset
        p.push(b'K');
        p.push(2);
        p.push(0x85); // TUPLE1 → size (2,)
        p.push(b'K');
        p.push(1);
        p.push(0x85); // TUPLE1 → stride (1,)
        p.push(0x89); // NEWFALSE requires_grad
        p.push(b'}'); // EMPTY_DICT backward_hooks
        p.push(b't'); // TUPLE → args
        p.push(b'R'); // REDUCE → tensor
        p.push(b'u'); // SETITEMS (pops back to the MARK after the dict)
        p.push(b'.'); // STOP

        let specs = parse_pickle(&p).unwrap();
        assert_eq!(specs.len(), 1);
        let (name, v) = &specs[0];
        assert_eq!(name, "a");
        match v {
            Val::Tensor {
                dtype,
                key,
                offset,
                size,
            } => {
                assert_eq!(*dtype, StorageDtype::Float32);
                assert_eq!(key, "0");
                assert_eq!(*offset, 0);
                assert_eq!(size, &vec![2]);
            }
            _ => panic!("expected a tensor spec"),
        }
    }
}
