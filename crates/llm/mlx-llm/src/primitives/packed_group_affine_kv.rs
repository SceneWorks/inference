//! CPU reference for the physically packed 2-bit group-affine KV representation (SC-20675).
//!
//! This module deliberately has no MLX dependency in its implementation.  It is the deterministic
//! storage/lifecycle seam that a later Metal reader can adopt: codes are four 2-bit values per byte,
//! while each group has an f32 scale and zero.  A dense reader is an explicit, instrumented fallback;
//! this type never retains a dense mirror.

use std::convert::TryInto;

use crate::error::{Error, Result};
use half::f16;

const MAGIC: &[u8; 8] = b"SW20675\0";
const VERSION: u32 = 1;
const BITS: u8 = 2;

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepresentationMetadata {
    pub identity: String,
    pub version: u32,
    pub group_size: usize,
    pub bits: u8,
    pub batch: usize,
    pub kv_heads: usize,
    pub head_dimension: usize,
    pub logical_len: usize,
    pub capacity: usize,
    pub absolute_offset: usize,
    pub allocated_bytes: usize,
    pub key_grouping: &'static str,
    pub value_grouping: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DenseFallbackEvent {
    pub operation: String,
    pub reason: String,
    pub logical_len: usize,
    pub allocated_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledHandleMetadata {
    pub device: String,
    pub queue: String,
    pub context: String,
    pub cache_identity: String,
}

#[derive(Clone, Debug)]
struct PackedTensor {
    rows: usize,
    width: usize,
    groups: usize,
    codes: Vec<u8>,
    scales: Vec<f16>,
    zeros: Vec<f16>,
}

impl PackedTensor {
    fn new(rows: usize, width: usize, group_size: usize, capacity_rows: usize) -> Self {
        let groups = width.div_ceil(group_size);
        Self {
            rows,
            width,
            groups,
            codes: Vec::with_capacity(capacity_rows * width.div_ceil(4)),
            scales: Vec::with_capacity(capacity_rows * groups),
            zeros: Vec::with_capacity(capacity_rows * groups),
        }
    }

    fn append(&mut self, values: &[f32], group_size: usize) -> Result<()> {
        if values.len() != self.width {
            return Err(Error::Config("packed KV row width mismatch".into()));
        }
        for group in 0..self.groups {
            let start = group * group_size;
            let end = (start + group_size).min(self.width);
            let slice = &values[start..end];
            let min = slice.iter().copied().fold(f32::INFINITY, f32::min);
            let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = ((max - min) / 3.0).max(f32::EPSILON);
            self.scales.push(f16::from_f32(scale));
            self.zeros.push(f16::from_f32(min));
            for (i, value) in slice.iter().copied().enumerate() {
                let code = ((value - min) / scale).round().clamp(0.0, 3.0) as u8;
                let index = self.rows * self.width + start + i;
                if index % 4 == 0 {
                    self.codes.push(code);
                } else {
                    let byte = self.codes.last_mut().expect("code byte");
                    *byte |= code << ((index % 4) * 2);
                }
            }
        }
        self.rows += 1;
        Ok(())
    }

    fn row(&self, row: usize, group_size: usize) -> Result<Vec<f32>> {
        if row >= self.rows {
            return Err(Error::Config("packed KV row out of range".into()));
        }
        let mut out = Vec::with_capacity(self.width);
        for i in 0..self.width {
            let code =
                (self.codes[(row * self.width + i) / 4] >> (((row * self.width + i) % 4) * 2)) & 3;
            let group = i / group_size;
            out.push(
                self.zeros[row * self.groups + group].to_f32()
                    + self.scales[row * self.groups + group].to_f32() * code as f32,
            );
        }
        Ok(out)
    }

    fn truncate(&mut self, rows: usize, group_size: usize) {
        self.rows = rows;
        self.codes.truncate(rows * self.width.div_ceil(4));
        self.scales.truncate(rows * self.groups);
        self.zeros.truncate(rows * self.groups);
        let _ = group_size;
    }

    fn bytes(&self) -> usize {
        self.codes.len() + (self.scales.len() + self.zeros.len()) * std::mem::size_of::<f16>()
    }
    fn allocated_bytes(&self) -> usize {
        self.codes.capacity()
            + (self.scales.capacity() + self.zeros.capacity()) * std::mem::size_of::<f16>()
    }
}

#[derive(Clone, Debug)]
struct LayerStorage {
    keys: PackedTensor,
    values: PackedTensor,
}

#[derive(Clone, Debug)]
pub struct PackedGroupAffineKvCache {
    identity: String,
    group_size: usize,
    batch: usize,
    kv_heads: usize,
    head_dimension: usize,
    capacity: usize,
    logical_len: usize,
    absolute_offset: usize,
    layers: Vec<Option<LayerStorage>>,
    fallback_events: Vec<DenseFallbackEvent>,
    cancelled: bool,
    handle: CompiledHandleMetadata,
}

impl PackedGroupAffineKvCache {
    pub fn new(
        identity: impl Into<String>,
        layers: usize,
        batch: usize,
        kv_heads: usize,
        head_dimension: usize,
        group_size: usize,
    ) -> Result<Self> {
        if group_size == 0 || head_dimension == 0 || batch == 0 || kv_heads == 0 || layers == 0 {
            return Err(Error::Config("invalid packed KV shape".into()));
        }
        Ok(Self {
            identity: identity.into(),
            group_size,
            batch,
            kv_heads,
            head_dimension,
            capacity: 0,
            logical_len: 0,
            absolute_offset: 0,
            layers: vec![None; layers],
            fallback_events: Vec::new(),
            cancelled: false,
            handle: CompiledHandleMetadata {
                device: "cpu-reference".into(),
                queue: "unbound".into(),
                context: "sc-20675-experimental".into(),
                cache_identity: "uncompiled".into(),
            },
        })
    }

    fn rows(&self) -> usize {
        self.batch * self.kv_heads
    }
    fn row_width(&self) -> usize {
        self.head_dimension
    }
    fn validate_step(&self, values: &[f32], step: usize) -> Result<()> {
        if self.cancelled {
            return Err(Error::Canceled);
        }
        if step == 0 || values.len() != step * self.rows() * self.row_width() {
            return Err(Error::Config("packed KV append shape mismatch".into()));
        }
        Ok(())
    }
    fn ensure_layer(&mut self, layer: usize) -> Result<&mut LayerStorage> {
        let cap = self.capacity.max(1);
        let width = self.row_width();
        let group_size = self.group_size;
        let rows = self.rows();
        let slot = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| Error::Config("layer out of range".into()))?;
        if slot.is_none() {
            *slot = Some(LayerStorage {
                keys: PackedTensor::new(self.logical_len, width, group_size, rows * cap),
                values: PackedTensor::new(self.logical_len, width, group_size, rows * cap),
            });
        }
        Ok(slot.as_mut().expect("initialized layer"))
    }
    fn grow(&mut self, required: usize) {
        while self.capacity < required {
            self.capacity = self.capacity.max(1) * 2;
        }
    }

    /// Append a contiguous `[batch, kv_heads, step, head_dimension]` slice. Rows are appended
    /// token-major, preserving logical and absolute offsets without copying prior codes.
    pub fn append(
        &mut self,
        layer: usize,
        keys: &[f32],
        values: &[f32],
        step: usize,
    ) -> Result<()> {
        self.validate_step(keys, step)?;
        self.validate_step(values, step)?;
        if layer >= self.layers.len() {
            return Err(Error::Config("layer out of range".into()));
        }
        self.grow(self.logical_len + step);
        let rows = self.rows();
        let width = self.row_width();
        let group = self.group_size;
        if self
            .layers
            .get(layer)
            .and_then(Option::as_ref)
            .is_some_and(|l| l.keys.rows != self.logical_len * rows)
        {
            return Err(Error::Msg(
                "layer append is ahead of the atomic commit".into(),
            ));
        }
        {
            let store = self.ensure_layer(layer)?;
            for token in 0..step {
                for row in 0..rows {
                    let index = (token * rows + row) * width;
                    store.keys.append(&keys[index..index + width], group)?;
                    store.values.append(&values[index..index + width], group)?;
                }
            }
        }
        if self.layers.iter().all(Option::is_some)
            && self
                .layers
                .iter()
                .flatten()
                .all(|l| l.keys.rows == (self.logical_len + step) * rows)
        {
            self.logical_len += step;
        }
        Ok(())
    }

    /// Stage every layer against a clone and commit only after all layer shapes/quantization pass.
    /// The caller owns the layer order; no partially appended representation is observable.
    pub fn append_all_layers(&mut self, updates: &[(&[f32], &[f32])], step: usize) -> Result<()> {
        if updates.len() != self.layers.len() {
            return Err(Error::Config("atomic append layer count mismatch".into()));
        }
        let mut staged = self.clone();
        for (layer, (keys, values)) in updates.iter().enumerate() {
            staged.append(layer, keys, values, step)?;
        }
        if staged.logical_len != self.logical_len + step {
            return Err(Error::Msg("atomic append did not commit all layers".into()));
        }
        *self = staged;
        Ok(())
    }

    pub fn trim(&mut self, len: usize) -> Result<()> {
        if len > self.logical_len {
            return Err(Error::Config("trim exceeds logical length".into()));
        }
        for layer in self.layers.iter_mut().flatten() {
            layer.keys.truncate(len * self.rows(), self.group_size);
            layer.values.truncate(len * self.rows(), self.group_size);
        }
        self.logical_len = len;
        Ok(())
    }
    pub fn rollback(&mut self, len: usize) -> Result<()> {
        self.trim(len)
    }
    pub fn clear(&mut self) {
        self.layers.iter_mut().for_each(|l| *l = None);
        self.logical_len = 0;
        self.capacity = 0;
        self.cancelled = false;
    }
    pub fn cancel(&mut self) {
        self.cancelled = true;
        self.layers.iter_mut().for_each(|layer| *layer = None);
        self.logical_len = 0;
        self.capacity = 0;
    }
    pub fn logical_len(&self) -> usize {
        self.logical_len
    }
    pub fn allocated_len(&self) -> usize {
        self.capacity
    }
    pub fn absolute_offset(&self) -> usize {
        self.absolute_offset
    }
    pub fn set_absolute_offset(&mut self, offset: usize) {
        self.absolute_offset = offset;
    }
    pub fn layers(&self) -> usize {
        self.layers.len()
    }
    pub fn allocated_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|l| l.keys.bytes() + l.values.bytes())
            .sum()
    }
    pub fn logical_stored_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|l| l.keys.bytes() + l.values.bytes())
            .sum()
    }
    pub fn allocated_vec_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|l| l.keys.allocated_bytes() + l.values.allocated_bytes())
            .sum()
    }
    pub fn process_visible_bytes_estimate(&self) -> usize {
        self.allocated_vec_bytes() + std::mem::size_of_val(self)
    }
    pub fn representation(&self) -> RepresentationMetadata {
        RepresentationMetadata {
            identity: self.identity.clone(),
            version: VERSION,
            group_size: self.group_size,
            bits: BITS,
            batch: self.batch,
            kv_heads: self.kv_heads,
            head_dimension: self.head_dimension,
            logical_len: self.logical_len,
            capacity: self.capacity,
            absolute_offset: self.absolute_offset,
            allocated_bytes: self.allocated_bytes(),
            key_grouping: "token-axis groups [B,H,ceil(S/group_size),D]",
            value_grouping: "channel-axis groups [B,H,S,ceil(D/group_size)]",
        }
    }
    pub fn read_row(&self, layer: usize, token: usize, row: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        if token >= self.logical_len || row >= self.rows() {
            return Err(Error::Config("packed KV read index out of bounds".into()));
        }
        let l = self
            .layers
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::Config("layer is not resident".into()))?;
        let index = token * self.rows() + row;
        Ok((
            l.keys.row(index, self.group_size)?,
            l.values.row(index, self.group_size)?,
        ))
    }
    pub fn dense_read_fallback(
        &mut self,
        operation: impl Into<String>,
        reason: impl Into<String>,
    ) -> DenseFallbackEvent {
        let event = DenseFallbackEvent {
            operation: operation.into(),
            reason: reason.into(),
            logical_len: self.logical_len,
            allocated_bytes: self.allocated_bytes(),
        };
        self.fallback_events.push(event.clone());
        event
    }
    pub fn fallback_events(&self) -> &[DenseFallbackEvent] {
        &self.fallback_events
    }
    pub fn no_dense_mirror(&self) -> bool {
        true
    }
    pub fn handle_metadata(&self) -> &CompiledHandleMetadata {
        &self.handle
    }
    pub fn bind_compiled_handle(&mut self, handle: CompiledHandleMetadata) -> Result<()> {
        if handle.cache_identity != self.identity {
            return Err(Error::Config("compiled handle identity mismatch".into()));
        }
        self.handle = handle;
        Ok(())
    }
    pub fn preflight(
        &mut self,
        backend: &str,
        query_length: usize,
        mask: bool,
    ) -> crate::primitives::CacheRoute {
        if backend != "mlx-metal" || query_length == 0 || mask {
            let reason = if backend != "mlx-metal" {
                "unsupported backend"
            } else if query_length == 0 {
                "empty query"
            } else {
                "mask requires dense fallback"
            };
            self.dense_read_fallback("preflight", reason);
            crate::primitives::CacheRoute::DenseFallback {
                reason: reason.into(),
            }
        } else {
            crate::primitives::CacheRoute::ExperimentalPacked
        }
    }

    /// Versioned, identity-bound snapshot. Restore is all-or-nothing and rejects mismatched shape,
    /// quantization, or identity before installing any state.
    pub fn save(&self) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        out.extend(VERSION.to_le_bytes());
        out.extend((self.identity.len() as u32).to_le_bytes());
        out.extend(self.identity.as_bytes());
        for n in [
            self.group_size,
            self.batch,
            self.kv_heads,
            self.head_dimension,
            self.capacity,
            self.logical_len,
            self.absolute_offset,
            self.layers.len(),
        ] {
            out.extend((n as u64).to_le_bytes());
        }
        out.push(BITS);
        for layer in &self.layers {
            out.push(layer.is_some() as u8);
            if let Some(layer) = layer {
                for tensor in [&layer.keys, &layer.values] {
                    out.extend((tensor.rows as u64).to_le_bytes());
                    out.extend((tensor.codes.len() as u64).to_le_bytes());
                    out.extend(&tensor.codes);
                    for values in [&tensor.scales, &tensor.zeros] {
                        out.extend((values.len() as u64).to_le_bytes());
                        for value in values {
                            out.extend(value.to_bits().to_le_bytes());
                        }
                    }
                }
            }
        }
        out.extend(checksum(&out).to_le_bytes());
        out
    }

    pub fn restore(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() < 8
            || u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().unwrap())
                != checksum(&bytes[..bytes.len() - 8])
        {
            return Err(Error::Config("snapshot checksum mismatch".into()));
        }
        let bytes = &bytes[..bytes.len() - 8];
        let mut p = 0;
        let take = |p: &mut usize, n: usize| -> Result<&[u8]> {
            let end = p
                .checked_add(n)
                .ok_or_else(|| Error::Config("snapshot overflow".into()))?;
            let s = bytes
                .get(*p..end)
                .ok_or_else(|| Error::Config("truncated snapshot".into()))?;
            *p = end;
            Ok(s)
        };
        if take(&mut p, 8)? != MAGIC {
            return Err(Error::Config("snapshot magic mismatch".into()));
        }
        if u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap()) != VERSION {
            return Err(Error::Config("snapshot version mismatch".into()));
        }
        let id_len = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap()) as usize;
        if take(&mut p, id_len)? != self.identity.as_bytes() {
            return Err(Error::Config("snapshot identity mismatch".into()));
        }
        let mut nums = [0u64; 8];
        for n in &mut nums {
            *n = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
        }
        if nums[0] as usize != self.group_size
            || nums[1] as usize != self.batch
            || nums[2] as usize != self.kv_heads
            || nums[3] as usize != self.head_dimension
            || nums[7] as usize != self.layers.len()
            || take(&mut p, 1)?[0] != BITS
        {
            return Err(Error::Config(
                "snapshot quantization or shape mismatch".into(),
            ));
        }
        if nums[5] > nums[4] || nums[4] > usize::MAX as u64 {
            return Err(Error::Config(
                "snapshot capacity/logical bounds mismatch".into(),
            ));
        }
        let mut restored = Vec::with_capacity(self.layers.len());
        let rows = self.rows();
        for _ in 0..self.layers.len() {
            let present = take(&mut p, 1)?[0] != 0;
            if !present {
                restored.push(None);
                continue;
            }
            let mut tensors = Vec::new();
            for _ in 0..2 {
                let tensor_rows = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap()) as usize;
                let code_len = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap()) as usize;
                let expected_rows = nums[5] as usize * rows;
                let expected_codes = expected_rows * self.head_dimension.div_ceil(4);
                let expected_metadata =
                    expected_rows * self.head_dimension.div_ceil(self.group_size);
                if tensor_rows != expected_rows || code_len != expected_codes {
                    return Err(Error::Config("snapshot code shape mismatch".into()));
                }
                let codes = take(&mut p, code_len)?.to_vec();
                let mut metadata = Vec::new();
                for _ in 0..2 {
                    let count = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap()) as usize;
                    if count != expected_metadata {
                        return Err(Error::Config("snapshot scale/zero count mismatch".into()));
                    }
                    let mut values = Vec::with_capacity(count);
                    for _ in 0..count {
                        values.push(f16::from_bits(u16::from_le_bytes(
                            take(&mut p, 2)?.try_into().unwrap(),
                        )));
                    }
                    metadata.push(values);
                }
                tensors.push(PackedTensor {
                    rows: tensor_rows,
                    width: self.head_dimension,
                    groups: self.head_dimension.div_ceil(self.group_size),
                    codes,
                    scales: metadata.remove(0),
                    zeros: metadata.remove(0),
                });
            }
            if tensors[0].rows != nums[5] as usize * rows {
                return Err(Error::Config("snapshot logical shape mismatch".into()));
            }
            restored.push(Some(LayerStorage {
                keys: tensors.remove(0),
                values: tensors.remove(0),
            }));
        }
        if p != bytes.len() {
            return Err(Error::Config("snapshot trailing bytes".into()));
        }
        self.capacity = nums[4] as usize;
        self.logical_len = nums[5] as usize;
        self.absolute_offset = nums[6] as usize;
        self.layers = restored;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn data(step: usize, rows: usize, width: usize, bias: f32) -> Vec<f32> {
        (0..step * rows * width)
            .map(|i| bias + i as f32 * 0.25)
            .collect()
    }
    #[test]
    fn packed_append_is_quantized_and_non_mirroring() {
        let mut c = PackedGroupAffineKvCache::new("model", 1, 1, 2, 7, 4).unwrap();
        let k = data(3, 2, 7, -2.0);
        c.append(0, &k, &k, 3).unwrap();
        assert_eq!(c.logical_len(), 3);
        assert!(c.allocated_bytes() > 0);
        assert!(c.no_dense_mirror());
        assert_eq!(c.representation().bits, 2);
    }
    #[test]
    fn chunking_offsets_and_rollback() {
        let mut c = PackedGroupAffineKvCache::new("m", 1, 1, 1, 5, 4).unwrap();
        c.set_absolute_offset(19);
        let a = data(2, 1, 5, 0.0);
        c.append(0, &a, &a, 2).unwrap();
        let b = data(3, 1, 5, 9.0);
        c.append(0, &b, &b, 3).unwrap();
        assert_eq!(
            (c.logical_len(), c.allocated_len(), c.absolute_offset()),
            (5, 8, 19)
        );
        c.rollback(2).unwrap();
        assert_eq!(c.logical_len(), 2);
    }
    #[test]
    fn snapshot_rejects_identity_and_round_trips() {
        let mut c = PackedGroupAffineKvCache::new("m", 1, 1, 1, 8, 4).unwrap();
        let x = data(1, 1, 8, 1.0);
        c.append(0, &x, &x, 1).unwrap();
        let bytes = c.save();
        let mut other = PackedGroupAffineKvCache::new("wrong", 1, 1, 1, 8, 4).unwrap();
        assert!(other.restore(&bytes).is_err());
        let mut restored = PackedGroupAffineKvCache::new("m", 1, 1, 1, 8, 4).unwrap();
        restored.restore(&bytes).unwrap();
        assert_eq!(restored.representation(), c.representation());
    }
    #[test]
    fn fallback_is_explicit_and_cancel_is_safe() {
        let mut c = PackedGroupAffineKvCache::new("m", 1, 1, 1, 3, 2).unwrap();
        let event = c.dense_read_fallback("read", "unsupported mask");
        assert_eq!(event.reason, "unsupported mask");
        c.cancel();
        assert!(c.append(0, &[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 1).is_err());
    }

    #[test]
    fn deterministic_extremes_zero_tail_and_non_aligned_width() {
        let mut c = PackedGroupAffineKvCache::new("m", 1, 1, 1, 11, 4).unwrap();
        let mut values = vec![0.0; 11];
        values[0] = -1000.0;
        values[4] = 1000.0;
        values[10] = 7.0;
        c.append(0, &values, &values, 1).unwrap();
        let (row, _) = c.read_row(0, 0, 0).unwrap();
        assert_eq!(row.len(), 11);
        assert!(row.iter().all(|value| value.is_finite()));
        assert_eq!(c.representation().logical_len, 1);
    }

    #[test]
    fn corrupted_snapshot_version_and_trailing_bytes_reject_without_mutation() {
        let mut c = PackedGroupAffineKvCache::new("m", 1, 1, 1, 8, 4).unwrap();
        let x = data(2, 1, 8, -3.0);
        c.append(0, &x, &x, 2).unwrap();
        let before = c.representation();
        let mut version = c.save();
        version[8] = 2;
        assert!(c.restore(&version).is_err());
        assert_eq!(c.representation(), before);
        let mut trailing = c.save();
        trailing.push(0);
        assert!(c.restore(&trailing).is_err());
        assert_eq!(c.representation(), before);
    }

    #[test]
    fn atomic_layers_and_preflight_are_reachable() {
        let mut c = PackedGroupAffineKvCache::new("m", 2, 1, 1, 5, 4).unwrap();
        let x = data(2, 1, 5, 0.0);
        let updates = [(&x[..], &x[..]), (&x[..x.len() - 1], &x[..x.len() - 1])];
        assert!(c.append_all_layers(&updates, 2).is_err());
        assert_eq!(c.logical_len(), 0);
        assert!(matches!(
            c.preflight("cpu", 1, false),
            crate::primitives::CacheRoute::DenseFallback { .. }
        ));
        assert!(matches!(
            c.preflight("mlx-metal", 1, false),
            crate::primitives::CacheRoute::ExperimentalPacked
        ));
        c.bind_compiled_handle(CompiledHandleMetadata {
            device: "metal0".into(),
            queue: "q0".into(),
            context: "ctx".into(),
            cache_identity: "m".into(),
        })
        .unwrap();
        assert_eq!(c.handle_metadata().device, "metal0");
    }
}
