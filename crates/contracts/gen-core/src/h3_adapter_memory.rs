//! MiniMax-H3 adapter memory profile carried by a
//! [`NativeTaskAllocation`](crate::h3_memory::NativeTaskAllocation). Header parsing and LoRA/LoKr
//! pricing live in `gen_core_h3::h3_adapter_memory` (sc-27316).

use crate::h3_memory::{NativeBackend, NativeRows, RowDomain, FFN_INTERMEDIATE_BUDGET};
use crate::weightsmeta::Dtype;
use std::collections::BTreeMap;

/// A real adapter-addressable projection in the selected source partition. Build
/// this from the loader's target paths and source headers, never the requested tier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3AdapterTarget {
    pub path: String,
    pub input: u64,
    pub output: u64,
    pub packed: bool,
    pub activation_width: u64,
    pub activation_dtype: Dtype,
    pub rows: RowDomain,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct H3AdapterMemoryProfile {
    /// Complete installed payload, including first-forward prepared caches.
    pub resident_bytes: u64,
    /// Complete adapter-owned install high-water, NOT additional to resident_bytes.
    /// Includes the current source map/buffer and construction intermediates; previous
    /// files contribute installed payload only, not future prepared caches.
    pub install_peak_bytes: u64,
    pub projections: Vec<H3AdapterProjectionMemory>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3AdapterProjectionMemory {
    pub target: H3AdapterTarget,
    /// Installed originals before the first forward.
    pub installed_bytes: u64,
    /// Originals plus independently owned prepared cache tensors.
    pub resident_bytes: u64,
    /// Nonresident casts/transposes alive during the forward.
    pub forward_fixed_bytes: u64,
    /// Residual intermediates plus accumulation outputs, excluding input/base output.
    pub forward_bytes_per_row: u64,
}
impl H3AdapterProjectionMemory {
    pub fn forward_working_bytes(&self, backend: NativeBackend, rows: NativeRows) -> u64 {
        let t = &self.target;
        let full = match t.rows {
            RowDomain::Packed => rows.packed,
            RowDomain::Conditioning => rows.conditioning,
            RowDomain::Schedule => rows.schedule,
        };
        if full == 0 {
            return 0;
        }
        let projected = if t.path.ends_with(".ff.net.0.proj") {
            Some(t.output)
        } else if t.path.ends_with(".ff.net.2") {
            Some(t.input.saturating_mul(2))
        } else {
            None
        };
        let m = match (backend, projected) {
            (NativeBackend::Candle { .. }, Some(width)) => {
                crate::attention_budget::AttentionBudget::from_score_elements(
                    FFN_INTERMEDIATE_BUDGET,
                    false,
                )
                .query_block_rows(width, full)
            }
            _ => full,
        };
        self.forward_fixed_bytes
            .saturating_add(m.saturating_mul(self.forward_bytes_per_row))
    }
}
impl H3AdapterMemoryProfile {
    /// Sequential layers do not coexist. Residuals of one MLX target may share a lazy graph;
    /// Candle evaluates each residual eagerly. Keep their actual domain and FFN span bounds.
    pub fn forward_working_bytes_for(
        &self,
        backend: NativeBackend,
        rows: NativeRows,
        domain: RowDomain,
    ) -> u64 {
        let mut targets = BTreeMap::<&str, u64>::new();
        for p in self.projections.iter().filter(|p| p.target.rows == domain) {
            let n = p.forward_working_bytes(backend, rows);
            let entry = targets.entry(&p.target.path).or_default();
            *entry = if matches!(backend, NativeBackend::Mlx { .. }) {
                entry.saturating_add(n)
            } else {
                (*entry).max(n)
            };
        }
        targets.into_values().max().unwrap_or(0)
    }

    pub fn forward_working_bytes(&self, backend: NativeBackend, rows: NativeRows) -> u64 {
        [
            RowDomain::Packed,
            RowDomain::Conditioning,
            RowDomain::Schedule,
        ]
        .into_iter()
        .map(|domain| self.forward_working_bytes_for(backend, rows, domain))
        .max()
        .unwrap_or(0)
    }
}
