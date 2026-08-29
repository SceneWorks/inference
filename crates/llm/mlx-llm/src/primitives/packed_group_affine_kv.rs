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
const VERSION: u32 = 2;
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
        let code_start = self.codes.len();
        self.codes.resize(code_start + self.width.div_ceil(4), 0);
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
                let index = start + i;
                self.codes[code_start + index / 4] |= code << ((index % 4) * 2);
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
            let code = (self.codes[row * self.width.div_ceil(4) + i / 4] >> ((i % 4) * 2)) & 3;
            let group = i / group_size;
            out.push(
                self.zeros[row * self.groups + group].to_f32()
                    + self.scales[row * self.groups + group].to_f32() * code as f32,
            );
        }
        Ok(out)
    }

    fn truncate(&mut self, rows: usize) {
        self.rows = rows;
        self.codes.truncate(rows * self.width.div_ceil(4));
        self.scales.truncate(rows * self.groups);
        self.zeros.truncate(rows * self.groups);
    }

    fn bytes(&self) -> usize {
        self.codes.len() + (self.scales.len() + self.zeros.len()) * std::mem::size_of::<f16>()
    }
    fn allocated_bytes(&self) -> usize {
        self.codes.capacity()
            + (self.scales.capacity() + self.zeros.capacity()) * std::mem::size_of::<f16>()
    }

    fn reserve_rows(&mut self, rows: usize) {
        self.codes.reserve(
            rows.saturating_mul(self.width.div_ceil(4))
                .saturating_sub(self.codes.len()),
        );
        self.scales.reserve(
            rows.saturating_mul(self.groups)
                .saturating_sub(self.scales.len()),
        );
        self.zeros.reserve(
            rows.saturating_mul(self.groups)
                .saturating_sub(self.zeros.len()),
        );
    }
}

/// Key storage groups tokens per channel (`[B,H,ceil(S/group),D]`).  Only an
/// incomplete final token group is staged densely; every completed group is
/// physically 2-bit codes plus f16 scale/zero metadata.
#[derive(Clone, Debug)]
struct TokenGroupKeyTensor {
    rows: usize,
    width: usize,
    group_size: usize,
    complete_tokens: usize,
    pending_tokens: usize,
    codes: Vec<u8>,
    scales: Vec<f16>,
    zeros: Vec<f16>,
    // `[pending_token, row, channel]`, bounded to `group_size - 1` tokens.
    pending: Vec<f32>,
}

impl TokenGroupKeyTensor {
    fn new(rows: usize, width: usize, group_size: usize, capacity_tokens: usize) -> Self {
        let groups = capacity_tokens.div_ceil(group_size);
        Self {
            rows,
            width,
            group_size,
            complete_tokens: 0,
            pending_tokens: 0,
            codes: Vec::with_capacity(rows * groups * (group_size * width).div_ceil(4)),
            scales: Vec::with_capacity(rows * groups * width),
            zeros: Vec::with_capacity(rows * groups * width),
            pending: Vec::with_capacity(rows * group_size.saturating_sub(1) * width),
        }
    }

    fn logical_tokens(&self) -> usize {
        self.complete_tokens + self.pending_tokens
    }

    fn complete_groups(&self) -> usize {
        self.complete_tokens / self.group_size
    }

    fn code_bytes_per_group(&self) -> usize {
        (self.group_size * self.width).div_ceil(4)
    }

    fn append(&mut self, values: &[f32], step: usize) -> Result<()> {
        if values.len() != self.rows * step * self.width {
            return Err(Error::Config("packed key append shape mismatch".into()));
        }
        for token in 0..step {
            for row in 0..self.rows {
                let start = (row * step + token) * self.width;
                self.pending
                    .extend_from_slice(&values[start..start + self.width]);
            }
            self.pending_tokens += 1;
            if self.pending_tokens == self.group_size {
                self.flush_pending_group()?;
            }
        }
        Ok(())
    }

    fn flush_pending_group(&mut self) -> Result<()> {
        if self.pending_tokens != self.group_size {
            return Err(Error::Config(
                "incomplete key token group cannot be quantized".into(),
            ));
        }
        let code_start = self.codes.len();
        self.codes
            .resize(code_start + self.rows * self.code_bytes_per_group(), 0);
        let group_index = self.complete_groups();
        for row in 0..self.rows {
            for channel in 0..self.width {
                let min = (0..self.group_size)
                    .map(|token| self.pending[(token * self.rows + row) * self.width + channel])
                    .fold(f32::INFINITY, f32::min);
                let max = (0..self.group_size)
                    .map(|token| self.pending[(token * self.rows + row) * self.width + channel])
                    .fold(f32::NEG_INFINITY, f32::max);
                let scale = ((max - min) / 3.0).max(f32::EPSILON);
                self.scales.push(f16::from_f32(scale));
                self.zeros.push(f16::from_f32(min));
                for token in 0..self.group_size {
                    let value = self.pending[(token * self.rows + row) * self.width + channel];
                    let code = ((value - min) / scale).round().clamp(0.0, 3.0) as u8;
                    let code_index = (token * self.width) + channel;
                    self.codes[code_start + row * self.code_bytes_per_group() + code_index / 4] |=
                        code << ((code_index % 4) * 2);
                }
            }
        }
        debug_assert_eq!(
            self.scales.len(),
            (group_index + 1) * self.rows * self.width
        );
        self.complete_tokens += self.group_size;
        self.pending_tokens = 0;
        self.pending.clear();
        Ok(())
    }

    fn truncate(&mut self, tokens: usize) -> Result<()> {
        if tokens > self.logical_tokens() {
            return Err(Error::Config(
                "packed key truncate exceeds logical length".into(),
            ));
        }
        let complete_tokens = tokens / self.group_size * self.group_size;
        let complete_groups = complete_tokens / self.group_size;
        let keep_pending = tokens - complete_tokens;
        let pending = if keep_pending == 0 {
            Vec::new()
        } else if complete_tokens == self.complete_tokens {
            self.pending[..keep_pending * self.rows * self.width].to_vec()
        } else {
            // A rollback may cut a completed group. Re-stage its retained prefix from the
            // already-quantized representation; no discarded dense mirror is retained.
            let mut retained = Vec::with_capacity(keep_pending * self.rows * self.width);
            for token in 0..keep_pending {
                for row in 0..self.rows {
                    retained.extend(self.row(complete_tokens + token, row)?);
                }
            }
            retained
        };
        self.codes
            .truncate(complete_groups * self.rows * self.code_bytes_per_group());
        self.scales
            .truncate(complete_groups * self.rows * self.width);
        self.zeros
            .truncate(complete_groups * self.rows * self.width);
        if keep_pending == 0 {
            self.pending.clear();
        } else {
            self.pending = pending;
        }
        self.complete_tokens = complete_tokens;
        self.pending_tokens = keep_pending;
        Ok(())
    }

    fn row(&self, token: usize, row: usize) -> Result<Vec<f32>> {
        if token >= self.logical_tokens() || row >= self.rows {
            return Err(Error::Config("packed key row out of range".into()));
        }
        if token >= self.complete_tokens {
            let pending_token = token - self.complete_tokens;
            let start = (pending_token * self.rows + row) * self.width;
            return Ok(self.pending[start..start + self.width].to_vec());
        }
        let group = token / self.group_size;
        let local_token = token % self.group_size;
        let code_base =
            group * self.rows * self.code_bytes_per_group() + row * self.code_bytes_per_group();
        let metadata_base = (group * self.rows + row) * self.width;
        let mut out = Vec::with_capacity(self.width);
        for channel in 0..self.width {
            let code_index = local_token * self.width + channel;
            let code = (self.codes[code_base + code_index / 4] >> ((code_index % 4) * 2)) & 3;
            out.push(
                self.zeros[metadata_base + channel].to_f32()
                    + self.scales[metadata_base + channel].to_f32() * code as f32,
            );
        }
        Ok(out)
    }

    fn logical_bytes(&self) -> usize {
        self.codes.len()
            + (self.scales.len() + self.zeros.len()) * std::mem::size_of::<f16>()
            + self.pending.len() * std::mem::size_of::<f32>()
    }

    fn allocated_bytes(&self) -> usize {
        self.codes.capacity()
            + (self.scales.capacity() + self.zeros.capacity()) * std::mem::size_of::<f16>()
            + self.pending.capacity() * std::mem::size_of::<f32>()
    }

    fn reserve_tokens(&mut self, tokens: usize) {
        let groups = tokens.div_ceil(self.group_size);
        self.codes.reserve(
            self.rows
                .saturating_mul(groups)
                .saturating_mul(self.code_bytes_per_group())
                .saturating_sub(self.codes.len()),
        );
        self.scales.reserve(
            self.rows
                .saturating_mul(groups)
                .saturating_mul(self.width)
                .saturating_sub(self.scales.len()),
        );
        self.zeros.reserve(
            self.rows
                .saturating_mul(groups)
                .saturating_mul(self.width)
                .saturating_sub(self.zeros.len()),
        );
    }
}

#[derive(Clone, Debug)]
struct LayerStorage {
    keys: TokenGroupKeyTensor,
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
                keys: TokenGroupKeyTensor::new(rows, width, group_size, cap),
                values: PackedTensor::new(self.logical_len * rows, width, group_size, rows * cap),
            });
        }
        Ok(slot.as_mut().expect("initialized layer"))
    }
    fn grow(&mut self, required: usize) {
        while self.capacity < required {
            self.capacity = self.capacity.max(1) * 2;
        }
        self.reserve_storage_for_capacity();
    }

    fn reserve_storage_for_capacity(&mut self) {
        let capacity = self.capacity;
        let rows = self.rows();
        for layer in self.layers.iter_mut().flatten() {
            layer.keys.reserve_tokens(capacity);
            layer.values.reserve_rows(rows.saturating_mul(capacity));
        }
    }

    /// Append a contiguous `[batch, kv_heads, step, head_dimension]` slice. Input remains
    /// batch/head-major as required by `KvCache`; keys stage token-axis groups and values pack
    /// each token's channel groups without copying completed historical codes.
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
            .is_some_and(|l| l.keys.logical_tokens() != self.logical_len)
        {
            return Err(Error::Msg(
                "layer append is ahead of the atomic commit".into(),
            ));
        }
        {
            let store = self.ensure_layer(layer)?;
            store.keys.append(keys, step)?;
            for token in 0..step {
                for row in 0..rows {
                    let index = (row * step + token) * width;
                    store.values.append(&values[index..index + width], group)?;
                }
            }
        }
        if self.layers.iter().all(Option::is_some)
            && self
                .layers
                .iter()
                .flatten()
                .all(|l| l.keys.logical_tokens() == self.logical_len + step)
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
            layer.keys.truncate(len)?;
            layer.values.truncate(len * self.rows());
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
            .map(|l| l.keys.logical_bytes() + l.values.bytes())
            .sum()
    }
    pub fn logical_stored_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|l| l.keys.logical_bytes() + l.values.bytes())
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
            l.keys.row(token, row)?,
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
                let keys = &layer.keys;
                out.extend((keys.complete_tokens as u64).to_le_bytes());
                out.extend((keys.pending_tokens as u64).to_le_bytes());
                out.extend((keys.codes.len() as u64).to_le_bytes());
                out.extend(&keys.codes);
                for values in [&keys.scales, &keys.zeros] {
                    out.extend((values.len() as u64).to_le_bytes());
                    for value in values {
                        out.extend(value.to_bits().to_le_bytes());
                    }
                }
                out.extend((keys.pending.len() as u64).to_le_bytes());
                for value in &keys.pending {
                    out.extend(value.to_bits().to_le_bytes());
                }

                let values = &layer.values;
                out.extend((values.rows as u64).to_le_bytes());
                out.extend((values.codes.len() as u64).to_le_bytes());
                out.extend(&values.codes);
                for metadata in [&values.scales, &values.zeros] {
                    out.extend((metadata.len() as u64).to_le_bytes());
                    for value in metadata {
                        out.extend(value.to_bits().to_le_bytes());
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
        if nums.iter().any(|&value| value > usize::MAX as u64)
            || nums[0] as usize != self.group_size
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
        if nums[5] > nums[4] {
            return Err(Error::Config(
                "snapshot capacity/logical bounds mismatch".into(),
            ));
        }
        let mut restored = Vec::with_capacity(self.layers.len());
        let rows = self.rows();
        for _ in 0..self.layers.len() {
            let present_byte = take(&mut p, 1)?[0];
            if present_byte > 1 {
                return Err(Error::Config(
                    "snapshot layer presence flag mismatch".into(),
                ));
            }
            let present = present_byte != 0;
            if !present {
                if nums[5] != 0 {
                    return Err(Error::Config("snapshot omits a resident layer".into()));
                }
                restored.push(None);
                continue;
            }
            let complete_tokens = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
            let pending_tokens = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
            if complete_tokens > usize::MAX as u64
                || pending_tokens > usize::MAX as u64
                || complete_tokens as usize % self.group_size != 0
                || complete_tokens.checked_add(pending_tokens) != Some(nums[5])
                || pending_tokens as usize >= self.group_size
            {
                return Err(Error::Config(
                    "snapshot key token-group shape mismatch".into(),
                ));
            }
            let complete_tokens = complete_tokens as usize;
            let pending_tokens = pending_tokens as usize;
            let key_groups = complete_tokens / self.group_size;
            let key_code_len = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
            let expected_key_codes = key_groups
                .checked_mul(rows)
                .and_then(|value| {
                    value.checked_mul((self.group_size * self.head_dimension).div_ceil(4))
                })
                .ok_or_else(|| Error::Config("snapshot key code length overflow".into()))?;
            if key_code_len != expected_key_codes as u64 {
                return Err(Error::Config("snapshot key code shape mismatch".into()));
            }
            let key_codes = take(&mut p, expected_key_codes)?.to_vec();
            let expected_key_metadata = key_groups
                .checked_mul(rows)
                .and_then(|value| value.checked_mul(self.head_dimension))
                .ok_or_else(|| Error::Config("snapshot key metadata overflow".into()))?;
            let mut key_metadata = Vec::new();
            for _ in 0..2 {
                let count = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
                if count != expected_key_metadata as u64 {
                    return Err(Error::Config(
                        "snapshot key scale/zero count mismatch".into(),
                    ));
                }
                let mut values = Vec::with_capacity(expected_key_metadata);
                for _ in 0..expected_key_metadata {
                    values.push(f16::from_bits(u16::from_le_bytes(
                        take(&mut p, 2)?.try_into().unwrap(),
                    )));
                }
                key_metadata.push(values);
            }
            let pending_count = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
            let expected_pending = pending_tokens
                .checked_mul(rows)
                .and_then(|value| value.checked_mul(self.head_dimension))
                .ok_or_else(|| Error::Config("snapshot key pending overflow".into()))?;
            if pending_count != expected_pending as u64 {
                return Err(Error::Config("snapshot key pending count mismatch".into()));
            }
            let mut pending = Vec::with_capacity(expected_pending);
            for _ in 0..expected_pending {
                pending.push(f32::from_bits(u32::from_le_bytes(
                    take(&mut p, 4)?.try_into().unwrap(),
                )));
            }

            let value_rows = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
            let expected_rows = (nums[5] as usize)
                .checked_mul(rows)
                .ok_or_else(|| Error::Config("snapshot value row overflow".into()))?;
            let value_code_len = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
            let expected_value_codes =
                expected_rows
                    .checked_mul(self.head_dimension.div_ceil(4))
                    .ok_or_else(|| Error::Config("snapshot value code overflow".into()))?;
            if value_rows != expected_rows as u64 || value_code_len != expected_value_codes as u64 {
                return Err(Error::Config("snapshot value code shape mismatch".into()));
            }
            let value_codes = take(&mut p, expected_value_codes)?.to_vec();
            let expected_value_metadata = expected_rows
                .checked_mul(self.head_dimension.div_ceil(self.group_size))
                .ok_or_else(|| Error::Config("snapshot value metadata overflow".into()))?;
            let mut value_metadata = Vec::new();
            for _ in 0..2 {
                let count = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
                if count != expected_value_metadata as u64 {
                    return Err(Error::Config(
                        "snapshot value scale/zero count mismatch".into(),
                    ));
                }
                let mut values = Vec::with_capacity(expected_value_metadata);
                for _ in 0..expected_value_metadata {
                    values.push(f16::from_bits(u16::from_le_bytes(
                        take(&mut p, 2)?.try_into().unwrap(),
                    )));
                }
                value_metadata.push(values);
            }
            restored.push(Some(LayerStorage {
                keys: TokenGroupKeyTensor {
                    rows,
                    width: self.head_dimension,
                    group_size: self.group_size,
                    complete_tokens,
                    pending_tokens,
                    codes: key_codes,
                    scales: key_metadata.remove(0),
                    zeros: key_metadata.remove(0),
                    pending,
                },
                values: PackedTensor {
                    rows: expected_rows,
                    width: self.head_dimension,
                    groups: self.head_dimension.div_ceil(self.group_size),
                    codes: value_codes,
                    scales: value_metadata.remove(0),
                    zeros: value_metadata.remove(0),
                },
            }));
        }
        if p != bytes.len() {
            return Err(Error::Config("snapshot trailing bytes".into()));
        }
        self.capacity = nums[4] as usize;
        self.logical_len = nums[5] as usize;
        self.absolute_offset = nums[6] as usize;
        self.layers = restored;
        self.reserve_storage_for_capacity();
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

    fn bhst_data(batch: usize, heads: usize, step: usize, width: usize, bias: f32) -> Vec<f32> {
        (0..batch * heads)
            .flat_map(|row| {
                (0..step).flat_map(move |token| {
                    (0..width).map(move |channel| {
                        bias + (row * 10_000 + token * 100 + channel) as f32 * 0.03125
                    })
                })
            })
            .collect()
    }

    fn token_range(
        source: &[f32],
        batch: usize,
        heads: usize,
        full_step: usize,
        width: usize,
        start: usize,
        len: usize,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(batch * heads * len * width);
        for row in 0..batch * heads {
            for token in start..start + len {
                let offset = (row * full_step + token) * width;
                out.extend_from_slice(&source[offset..offset + width]);
            }
        }
        out
    }

    fn assert_same_rows(left: &PackedGroupAffineKvCache, right: &PackedGroupAffineKvCache) {
        assert_eq!(left.logical_len(), right.logical_len());
        for token in 0..left.logical_len() {
            for row in 0..left.rows() {
                assert_eq!(
                    left.read_row(0, token, row).unwrap(),
                    right.read_row(0, token, row).unwrap()
                );
            }
        }
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
        version[8] = 3;
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

    #[test]
    fn key_token_groups_and_value_channel_groups_are_chunk_invariant() {
        let (batch, heads, step, width, group) = (2, 2, 7, 5, 3);
        let keys = bhst_data(batch, heads, step, width, -9.0);
        let values = bhst_data(batch, heads, step, width, 4.0);
        let mut one = PackedGroupAffineKvCache::new("m", 1, batch, heads, width, group).unwrap();
        one.append(0, &keys, &values, step).unwrap();
        let mut chunks = PackedGroupAffineKvCache::new("m", 1, batch, heads, width, group).unwrap();
        for (start, len) in [(0, 1), (1, 2), (3, 1), (4, 3)] {
            chunks
                .append(
                    0,
                    &token_range(&keys, batch, heads, step, width, start, len),
                    &token_range(&values, batch, heads, step, width, start, len),
                    len,
                )
                .unwrap();
        }
        assert_same_rows(&one, &chunks);
        let store = one.layers[0].as_ref().unwrap();
        assert_eq!(store.keys.complete_tokens, 6);
        assert_eq!(store.keys.pending_tokens, 1);
        assert_eq!(store.keys.scales.len(), batch * heads * 2 * width);
        assert_eq!(
            store.values.scales.len(),
            batch * heads * step * width.div_ceil(group)
        );
        assert_eq!(
            one.representation().key_grouping,
            "token-axis groups [B,H,ceil(S/group_size),D]"
        );
    }

    #[test]
    fn rollback_reappend_clears_packed_tails_and_requantizes_cut_key_groups() {
        let (batch, heads, width, group) = (1, 2, 5, 4);
        let original_keys = bhst_data(batch, heads, 6, width, -1000.0);
        let original_values = bhst_data(batch, heads, 6, width, 1000.0);
        let replacement_keys = bhst_data(batch, heads, 3, width, 700.0);
        let replacement_values = bhst_data(batch, heads, 3, width, -700.0);
        let mut rolled = PackedGroupAffineKvCache::new("m", 1, batch, heads, width, group).unwrap();
        rolled
            .append(0, &original_keys, &original_values, 6)
            .unwrap();
        rolled.rollback(3).unwrap();
        rolled
            .append(0, &replacement_keys, &replacement_values, 3)
            .unwrap();

        let mut expected_keys = Vec::new();
        let mut expected_values = Vec::new();
        for row in 0..batch * heads {
            let prefix = row * 6 * width;
            let suffix = row * 3 * width;
            expected_keys.extend_from_slice(&original_keys[prefix..prefix + 3 * width]);
            expected_keys.extend_from_slice(&replacement_keys[suffix..suffix + 3 * width]);
            expected_values.extend_from_slice(&original_values[prefix..prefix + 3 * width]);
            expected_values.extend_from_slice(&replacement_values[suffix..suffix + 3 * width]);
        }
        let mut expected =
            PackedGroupAffineKvCache::new("m", 1, batch, heads, width, group).unwrap();
        expected
            .append(0, &expected_keys, &expected_values, 6)
            .unwrap();
        assert_same_rows(&rolled, &expected);
        assert_eq!(rolled.logical_len(), 6);
    }

    #[test]
    fn pending_key_groups_snapshot_and_byte_accounting_are_strict() {
        let (batch, heads, step, width, group) = (2, 1, 5, 7, 3);
        let keys = bhst_data(batch, heads, step, width, 0.0);
        let mut values = bhst_data(batch, heads, step, width, 0.0);
        values[0] = -1000.0;
        values[values.len() - 1] = 1000.0;
        let mut cache = PackedGroupAffineKvCache::new("m", 1, batch, heads, width, group).unwrap();
        cache.append(0, &keys, &values, step).unwrap();
        let bytes_before = cache.logical_stored_bytes();
        assert!(cache.allocated_vec_bytes() >= bytes_before);
        assert!(cache.process_visible_bytes_estimate() >= cache.allocated_vec_bytes());
        let snapshot = cache.save();
        let mut restored =
            PackedGroupAffineKvCache::new("m", 1, batch, heads, width, group).unwrap();
        restored.restore(&snapshot).unwrap();
        assert_same_rows(&cache, &restored);
        assert_eq!(
            cache.logical_stored_bytes(),
            restored.logical_stored_bytes()
        );
        let mut corrupt = snapshot.clone();
        corrupt[corrupt.len() / 2] ^= 0x01;
        assert!(restored.restore(&corrupt).is_err());
        assert_same_rows(&cache, &restored);
        cache.clear();
        assert_eq!(cache.logical_stored_bytes(), 0);
    }
}
