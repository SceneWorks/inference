//! Native H3 deferred-DiT residency. MLX source: `block_stream.rs::{DitBlockStream,
//! precompute_adaln_windowed}` and `dit/model.rs::{load_dir_deferred,new_windowed,
//! forward_windowed}`. Fresh weights views and forced carried state bound each ACTUAL aligned
//! BlockPlan window; refiner/root tensors are not streamed. This is structural arithmetic, not a
//! measured peak. CUDA eligibility additionally requires source-owned `CudaWorkingSet`
//! facts matching its eager prepared-host/body-transfer loader and bounded attention/FFN.

use std::collections::{BTreeMap, BTreeSet};

use super::{NativeBackend, NativeProjection, NativeRows, NativeTaskAllocation, RowDomain};
use crate::block_window::BlockPlan;
use crate::{Error, Result};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Partition {
    payload: u64,
    /// Subset forced by native descriptor validation even before block source verification.
    eager_payload: u64,
    /// Largest one source-to-compute cast, not a sum of all cast outputs.
    forward_cast: u64,
    descriptors: u64,
    rotations: BTreeSet<u64>,
}
impl Partition {
    fn resident(&self) -> Result<u64> {
        plus(self.payload, rotation_bytes(&self.rotations)?)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Layer {
    body: Partition,
    adaln: Partition,
}

/// Payload-only partition, built from the selected task's actual materialized tensors. Descriptor
/// blobs and synthesized shared rotations must NOT appear in `resident_by_name`. Root adjustment
/// is only the existing S5 compute/retained-source/cache DELTA over these payloads. Pruned reduced
/// projections remain root-resident, even though their source names contain block indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowAllocation {
    root: Partition,
    layers: Vec<Layer>,
    pruned: bool,
    cuda_working: Option<CudaWorkingSet>,
    cuda_adaln: Option<Vec<CudaAdalnWorkingSet>>,
}

/// Actual full-width AdaLN projection consumed by CUDA's modulation_for_window. Native
/// descriptor routes keep their existing working facts; compatible dense/affine sources
/// supply these header-owned logical dimensions before any tensors are materialized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaAdalnWorkingSet {
    pub input: u64,
    pub output: u64,
    pub affine: bool,
}
impl CudaAdalnWorkingSet {
    fn working_bytes(self, rows: u64, element_bytes: u64) -> Result<u64> {
        if self.input == 0 || self.output == 0 || rows == 0 || !matches!(element_bytes, 2 | 4) {
            return Err(invalid());
        }
        let input = times(rows, self.input)?;
        let output = times(rows, self.output)?;
        let weight = times(self.input, self.output)?;
        // The F32 temb remains alive across every projection. Silu briefly owns both
        // sigmoid and multiplied F32 arrays; activated then stays at the compute dtype.
        let activation = times(12, input)?;
        let carried_input = times(4 + element_bytes, input)?;
        // QLinear's F32 dequantization and its BF16 cast coexist at the let statement,
        // then the F32 tensor drops BEFORE Linear::forward. F32 compute aliases the cast.
        let dequant = if self.affine {
            plus(
                times(
                    if element_bytes == 4 {
                        4
                    } else {
                        4 + element_bytes
                    },
                    weight,
                )?,
                carried_input,
            )?
        } else {
            0
        };
        // Linear's matmul and bias-added outputs overlap. The later six contiguous parts
        // overlap the complete projected tensor instead; both phases own two output arrays.
        let outputs = times(times(2, element_bytes)?, output)?;
        let linear = plus(
            plus(carried_input, outputs)?,
            if self.affine {
                times(element_bytes, weight)?
            } else {
                0
            },
        )?;
        Ok(activation.max(dequant).max(linear))
    }
}

/// Source-config dimensions and largest runtime dequantization, independent of a checkpoint
/// name or measured peak. CUDA's dense/affine routes need these even without native descriptors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaWorkingSet {
    pub hidden: u64,
    pub heads: u64,
    pub head_dim: u64,
    pub rotary_dim: u64,
    pub ffn_inner: u64,
    pub element_bytes: u64,
    pub projection_decode_bytes: u64,
}
impl CudaWorkingSet {
    fn attention_tables_bytes(self, rows: u64) -> Result<u64> {
        // MmRopeTables retains two F32 [S,R] tables across every block. The existing
        // 64-byte row envelope also covers device positions (12), inverse permutation (4),
        // resident local/class indices (8), active AdaLN/timestep indices (8), and one
        // gather/scatter U32 upload (4). Host F64 position rows and Vec indices are excluded.
        times(rows, plus(times(8, self.rotary_dim)?, 64)?)
    }
    pub fn carried_attention_bytes(self, rows: u64) -> Result<u64> {
        plus(
            times(times(times(2, self.hidden)?, rows)?, self.element_bytes)?,
            self.attention_tables_bytes(rows)?,
        )
    }
    fn qk_norm_peak(self, rows: u64, inner: u64) -> Result<u64> {
        // DitAttention's shadowed original Q/K/V stay alive through the inner scope. During
        // the second rms_weighted, normalized Q is also retained: four stream buffers.
        // nn::rms_weighted retains xf and normed through the final weighted cast. F32 xf
        // aliases K only on an F32 stream; the final cast likewise aliases its F32 input.
        let f32_inner = times(times(times(rows, self.heads)?, self.head_dim)?, 4)?;
        let reduction = times(times(rows, self.heads)?, 4)?;
        let (xf, cast) = if self.element_bytes == 4 {
            (0, 0)
        } else {
            (f32_inner, inner)
        };
        let square_reduce = plus(f32_inner, reduction)?;
        let divide = plus(f32_inner, times(2, reduction)?)?;
        let weighted_cast = plus(
            plus(times(2, f32_inner)?, cast)?,
            times(4, self.head_dim)?, // at most one F32 norm-weight cast
        )?;
        plus(
            plus(times(4, inner)?, xf)?,
            square_reduce.max(divide).max(weighted_cast),
        )
    }
    fn rotary_peak(self, rows: u64, inner: u64) -> Result<u64> {
        let rotated = times(
            times(times(rows, self.heads)?, self.rotary_dim)?,
            self.element_bytes,
        )?;
        let partial = self.rotary_dim != self.head_dim;
        let head = if partial { rotated } else { 0 };
        let table_casts = if self.element_bytes == 4 {
            0
        } else {
            times(times(times(2, rows)?, self.rotary_dim)?, self.element_bytes)?
        };
        // During the second apply: original QKV, normalized Q/K and already rotated Q
        // remain (six stream buffers). apply retains head, rotated, both products and
        // their sum at the multiply-add; partial rotary then cats out+tail into a full
        // buffer. The head view is a copy only for partial rotary. Narrow views alias.
        let multiply_add = plus(head, times(4, rotated)?)?;
        let final_cat = if partial {
            plus(plus(head, times(2, rotated)?)?, inner)?
        } else {
            0 // out is already contiguous
        };
        plus(
            plus(times(6, inner)?, table_casts)?,
            multiply_add.max(final_cat),
        )
    }
    fn operation_peak(self, rows: u64) -> Result<u64> {
        self.operation_peak_with_adapter(rows, 0)
    }
    fn operation_peak_with_adapter(self, rows: u64, adapter_work: u64) -> Result<u64> {
        let hidden = times(times(rows, self.hidden)?, self.element_bytes)?;
        let inner = times(
            times(times(rows, self.heads)?, self.head_dim)?,
            self.element_bytes,
        )?;
        let budget = crate::veda::workspace::CUDA_DENSE_SCORE_ELEMENTS;
        let score_row = times(rows, self.heads)?;
        let scores = times(
            crate::attention_budget::AttentionBudget::from_score_elements(budget, false)
                .query_block_rows(score_row, rows),
            score_row,
        )?;
        // Materialized SDPA: caller QKV (3), its own contiguous Q/Kt/V (3), retained
        // output spans (1) and final cat (1). H3 uses softmax_last_dim, so 16 bytes per
        // score covers the two score/scaling buffers plus softmax output for F32 or BF16.
        let materialized = plus(times(8, inner)?, times(16, scores)?)?;
        // The current vendored fixed-length FlashAttention wrapper allocates an F32
        // [B,128,H,S] LSE buffer, not [B,H,S]. Its kernel overlaps caller QKV+output;
        // after return the output's transpose/contiguous overlaps one more full buffer.
        let flash_lse = times(times(rows, self.heads)?, 128 * 4)?;
        let fused = plus(times(4, inner)?, flash_lse)?.max(times(5, inner)?);
        // Head merge retains the original SDPA output and its contiguous merged copy;
        // to_out then allocates hidden output while caller QKV remain.
        let output_projection = plus(times(5, inner)?, hidden)?;
        // Scoped attention retains input and modulated hidden throughout these alternatives.
        let attention = plus(
            times(2, hidden)?,
            self.qk_norm_peak(rows, inner)?
                .max(self.rotary_peak(rows, inner)?)
                .max(materialized)
                .max(fused)
                .max(output_projection),
        )?;
        // Residual adapters execute INSIDE a projection, never during later Q/K RMS,
        // rotary or SDPA. The larger to_out envelope (QKV + SDPA/merged outputs + base
        // hidden output) also covers earlier Q/K/V projections retaining prior outputs.
        let adapted_projection = plus(plus(times(2, hidden)?, output_projection)?, adapter_work)?;
        // SwiGLU chunks bound the expanded projection. Prior output spans and final cat
        // coexist with full hidden input; token-axis chunking never changes the key domain.
        let ffn = plus(
            times(4, hidden)?,
            times(
                3 * self.element_bytes,
                times(rows, times(2, self.ffn_inner)?)?
                    .min(budget)
                    .max(times(2, self.ffn_inner)?),
            )?,
        )?;
        plus(
            plus(
                attention
                    .max(adapted_projection)
                    .max(plus(ffn, adapter_work)?)
                    .max(times(8, hidden)?),
                self.attention_tables_bytes(rows)?,
            )?,
            self.projection_decode_bytes,
        )
    }
    fn root_peak(self, rows: NativeRows, conditioning: bool) -> Result<u64> {
        if conditioning {
            return self.operation_peak(rows.conditioning);
        }
        plus(
            times(
                times(times(8, rows.packed)?, self.hidden)?,
                self.element_bytes,
            )?,
            self.projection_decode_bytes,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Location {
    Root,
    Body(usize),
    Adaln(usize),
}
fn invalid() -> Error {
    Error::Msg(
        "H3 native window allocation: inconsistent, unsupported, or overflowing facts".into(),
    )
}
fn plus(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(invalid)
}
fn times(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(invalid)
}
fn rotation_bytes(groups: &BTreeSet<u64>) -> Result<u64> {
    groups
        .iter()
        .try_fold(0, |bytes, &g| plus(bytes, times(times(g, g)?, 4)?))
}

fn locate(name: &str) -> Result<Location> {
    let Some(tail) = name.strip_prefix("transformer_blocks.") else {
        return Ok(Location::Root);
    };
    let (index, suffix) = tail.split_once('.').ok_or_else(invalid)?;
    let layer = index.parse::<usize>().map_err(|_| invalid())?;
    if index != layer.to_string() || suffix.is_empty() {
        return Err(invalid());
    }
    if suffix.starts_with("adaln_proj.linear") {
        Ok(Location::Adaln(layer))
    } else {
        Ok(Location::Body(layer))
    }
}

impl WindowAllocation {
    pub fn from_resident_tensors(
        resident_by_name: &[(String, u64)],
        projections: &[NativeProjection],
        root_adjustment: u64,
        pruned: bool,
    ) -> Result<Self> {
        let mut root = Partition {
            payload: root_adjustment,
            ..Default::default()
        };
        let mut layers = BTreeMap::<usize, Layer>::new();
        // Constructors validate these small operands through as_slice BEFORE staged source
        // verification. Codes and ordinary dense bodies remain lazy; do not force them into
        // precompute merely because their source Array handles already exist.
        let eager_names = projections
            .iter()
            .flat_map(|p| {
                let suffixes: &[&str] = if p.w4a8 {
                    &["weight_s_rel", "weight_s_channel", "weight_codebook"]
                } else {
                    &["weight_scale"]
                };
                suffixes
                    .iter()
                    .map(move |suffix| format!("{}.{suffix}", p.base))
            })
            .collect::<BTreeSet<_>>();
        let mut names = BTreeSet::new();
        for (name, bytes) in resident_by_name {
            if name.ends_with(".comfy_quant") || !names.insert(name) {
                return Err(invalid());
            }
            let destination = match locate(name)? {
                Location::Root => &mut root,
                Location::Adaln(i) if pruned => {
                    layers.entry(i).or_default();
                    &mut root
                }
                Location::Adaln(i) => &mut layers.entry(i).or_default().adaln,
                Location::Body(i) => &mut layers.entry(i).or_default().body,
            };
            destination.payload = plus(destination.payload, *bytes)?;
            if eager_names.contains(name) {
                destination.eager_payload = plus(destination.eager_payload, *bytes)?;
            }
        }
        if layers.is_empty() || layers.keys().copied().ne(0..layers.len()) {
            return Err(invalid());
        }
        for p in projections {
            let destination = match locate(&p.base)? {
                Location::Root => &mut root,
                Location::Adaln(i) if pruned => {
                    if !layers.contains_key(&i) {
                        return Err(invalid());
                    }
                    &mut root
                }
                Location::Adaln(i) => &mut layers.get_mut(&i).ok_or_else(invalid)?.adaln,
                Location::Body(i) => &mut layers.get_mut(&i).ok_or_else(invalid)?.body,
            };
            destination.descriptors = plus(destination.descriptors, p.descriptor_bytes)?;
            // ConvRot matrices are shared by group inside one Weights view. W4A8 uses a
            // radix FHT and has no dense resident rotation, regardless of rotation_group.
            if !p.w4a8 {
                if p.rotation_group == 0 {
                    return Err(invalid());
                }
                destination.rotations.insert(p.rotation_group);
            }
        }
        Ok(Self {
            root,
            layers: layers.into_values().collect(),
            pruned,
            cuda_working: None,
            cuda_adaln: None,
        })
    }

    pub fn with_cuda_working_set(mut self, facts: CudaWorkingSet) -> Result<Self> {
        if facts.hidden == 0
            || facts.heads == 0
            || facts.head_dim == 0
            || facts.rotary_dim == 0
            || !facts.rotary_dim.is_multiple_of(2)
            || facts.rotary_dim > facts.head_dim
            || facts.ffn_inner == 0
            || !matches!(facts.element_bytes, 2 | 4)
        {
            return Err(invalid());
        }
        self.cuda_working = Some(facts);
        Ok(self)
    }

    pub fn with_cuda_adaln_working_sets(mut self, facts: Vec<CudaAdalnWorkingSet>) -> Result<Self> {
        if self.pruned || facts.len() != self.layers.len() {
            return Err(invalid());
        }
        for fact in &facts {
            // Validate unknown/overflowing dimensions independently of the eventual schedule.
            fact.working_bytes(1, 2)?;
        }
        self.cuda_adaln = Some(facts);
        Ok(self)
    }

    /// Denoising only: root + modulation tables + one actual aligned body window. This excludes
    /// successive precompute allocations and operation scratch, allowing attention alternatives
    /// to compose with a shared buffer identity rather than adding complete phase peaks.
    pub fn denoise_resident_bytes(
        &self,
        task: &NativeTaskAllocation,
        backend: NativeBackend,
        rows: NativeRows,
        window: usize,
    ) -> Result<u64> {
        let base = plus(self.root.resident()?, self.table_bytes(task, rows)?)?;
        self.plan(backend, window)?
            .windows()
            .try_fold(base, |peak, range| {
                Ok(peak.max(plus(
                    base,
                    self.window_partition(range, false)?.resident()?,
                )?))
            })
    }

    pub fn denoise_with_veda_bytes(
        &self,
        task: &NativeTaskAllocation,
        backend: NativeBackend,
        rows: NativeRows,
        window: usize,
        veda: &crate::veda::workspace::WorkspaceBound,
    ) -> Result<u64> {
        let facts = self.cuda_working.ok_or_else(invalid)?;
        let dense = self.denoise_peak_bytes(task, backend, rows, window)?;
        let attention = plus(
            self.denoise_resident_bytes(task, backend, rows, window)?,
            plus(
                facts.carried_attention_bytes(rows.packed)?,
                veda.device_peak_bytes(),
            )?,
        )?;
        Ok(dense.max(attention))
    }

    /// Add only actual current source-to-compute cast OUTPUT sizes supplied by the selected
    /// loader policy. Endpoint payloads are already max(source, compute), not their sum. A cast
    /// can first evaluate during forward; it therefore overlaps operation work, not just load.
    pub fn with_source_casts(mut self, casts: &[(String, u64)]) -> Result<Self> {
        for (name, bytes) in casts {
            let partition = match locate(name)? {
                Location::Root => &mut self.root,
                Location::Adaln(i) if self.pruned => {
                    if i >= self.layers.len() {
                        return Err(invalid());
                    }
                    &mut self.root
                }
                Location::Adaln(i) => &mut self.layers.get_mut(i).ok_or_else(invalid)?.adaln,
                Location::Body(i) => &mut self.layers.get_mut(i).ok_or_else(invalid)?.body,
            };
            partition.forward_cast = partition.forward_cast.max(*bytes);
        }
        Ok(self)
    }

    pub fn n_blocks(&self) -> usize {
        self.layers.len()
    }

    pub fn streamed_weight_facts(&self) -> Result<crate::StreamedWeightFacts> {
        Ok(crate::StreamedWeightFacts {
            resident_bytes: self.root.resident()?,
            stacks: vec![self
                .layers
                .iter()
                .map(|layer| layer.body.resident())
                .collect::<Result<Vec<_>>>()?],
        })
    }

    /// CUDA prepares/folds all adapter factors on the HOST before prepare_streamed moves
    /// root/refiner and the current body window. The per-copy prepared caches are independently
    /// owned and drop with that window. Ordinary clones retain their existing shared semantics.
    pub fn adapter_resident_bytes(
        &self,
        task: &NativeTaskAllocation,
        backend: NativeBackend,
        window: usize,
    ) -> Result<u64> {
        let plan = self.plan(backend, window)?;
        let Some(adapter) = &task.adapter else {
            return Ok(0);
        };
        let mut root = 0;
        let mut blocks = vec![0; self.layers.len()];
        for projection in &adapter.projections {
            let dest = match locate(&projection.target.path)? {
                Location::Root => &mut root,
                Location::Body(i) => blocks.get_mut(i).ok_or_else(invalid)?,
                Location::Adaln(_) => return Err(invalid()), // not adapter-addressable
            };
            *dest = plus(*dest, projection.resident_bytes)?;
        }
        let peak = plan.windows().try_fold(root, |peak, range| {
            let body = blocks[range].iter().try_fold(0, |n, &b| plus(n, b))?;
            Ok(peak.max(plus(root, body)?))
        });
        peak
    }

    /// Resident-only counterpart for the ordinary native weight floor. Includes root/refiner,
    /// actual fresh-view groups and retained modulation cache, but NO operation scratch or
    /// descriptor blobs. This avoids placing a full-component ordinary floor underneath a real
    /// window=1 allocation prediction. Full-component single-view group storage is not forced to
    /// equal deferred root+fresh-window storage; those are genuinely different lifetimes.
    pub fn resident_peak_bytes(
        &self,
        task: &NativeTaskAllocation,
        backend: NativeBackend,
        rows: NativeRows,
        window: usize,
    ) -> Result<u64> {
        let plan = self.plan(backend, window)?;
        let root = self.root.resident()?;
        let table = self.table_bytes(task, rows)?;
        let mut peak = plus(root, table)?;
        for range in plan.windows() {
            let body = self.window_partition(range.clone(), false)?.resident()?;
            peak = peak.max(plus(plus(root, table)?, body)?);
            if !self.pruned {
                let adaln = self.window_partition(range.clone(), true)?.resident()?;
                let produced = times(table / self.layers.len() as u64, range.end as u64)?;
                peak = peak.max(plus(plus(root, produced)?, adaln)?);
            }
        }
        Ok(peak)
    }

    fn plan(&self, backend: NativeBackend, window: usize) -> Result<BlockPlan> {
        if matches!(backend, NativeBackend::Candle { .. }) && self.cuda_working.is_none() {
            return Err(invalid());
        }
        BlockPlan::new(self.layers.len(), window)
    }

    fn table_bytes(&self, task: &NativeTaskAllocation, rows: NativeRows) -> Result<u64> {
        // Pruned source operands and the admitted-domain retained cache already live in the S5
        // root adjustment; those projections are never evicted or rematerialized per window.
        if self.pruned {
            if task.table_bytes_per_timestep != 0 || task.evicted_bytes != 0 {
                return Err(invalid());
            }
            return Ok(0);
        }
        if !task
            .table_bytes_per_timestep
            .is_multiple_of(self.layers.len() as u64)
        {
            return Err(invalid());
        }
        times(task.table_bytes_per_timestep, rows.schedule)
    }

    fn is_root(&self, projection: &NativeProjection) -> Result<bool> {
        Ok(match locate(&projection.base)? {
            Location::Root => true,
            Location::Adaln(_) => self.pruned,
            Location::Body(_) => false,
        })
    }

    fn root_working(
        &self,
        task: &NativeTaskAllocation,
        backend: NativeBackend,
        rows: NativeRows,
        conditioning: bool,
    ) -> Result<u64> {
        let mut peak = 0;
        for p in &task.projections {
            if self.is_root(p)? && (p.rows == RowDomain::Conditioning) == conditioning {
                let work = p.working_bytes(backend, rows);
                // S5 already counts the pruned modulation cache backing these output arrays.
                let work = if self.pruned && p.rows == RowDomain::Schedule {
                    work.saturating_sub(
                        p.output
                            .saturating_mul(p.activation_width)
                            .saturating_mul(rows.schedule),
                    )
                } else {
                    work
                };
                peak = peak.max(work);
            }
        }
        if matches!(backend, NativeBackend::Candle { .. }) {
            peak = peak.max(
                self.cuda_working
                    .ok_or_else(invalid)?
                    .root_peak(rows, conditioning)?,
            );
        }
        plus(peak, self.root.forward_cast)
    }

    /// Exact aligned window's payload and fresh-view shared matrices. Root and window groups
    /// intentionally do NOT deduplicate against each other: their Weights views are distinct and
    /// both are live. Consecutive windows release their view, so they take a maximum, not a sum.
    fn window_partition(&self, range: std::ops::Range<usize>, adaln: bool) -> Result<Partition> {
        let mut result = Partition::default();
        for layer in &self.layers[range] {
            let p = if adaln { &layer.adaln } else { &layer.body };
            result.payload = plus(result.payload, p.payload)?;
            result.eager_payload = plus(result.eager_payload, p.eager_payload)?;
            result.forward_cast = result.forward_cast.max(p.forward_cast);
            result.descriptors = plus(result.descriptors, p.descriptors)?;
            result.rotations.extend(p.rotations.iter().copied());
        }
        Ok(result)
    }

    /// A staged full stack has ONE shared Weights context, unlike deferred root/fresh-window
    /// views. Its ConvRot group matrices deduplicate across root, body and AdaLN together.
    fn full_view(&self, backend: NativeBackend) -> Result<(u64, Partition, Partition)> {
        self.plan(backend, self.layers.len())?;
        let body = self.window_partition(0..self.layers.len(), false)?;
        let adaln = self.window_partition(0..self.layers.len(), true)?;
        let mut groups = self.root.rotations.clone();
        groups.extend(body.rotations.iter().copied());
        groups.extend(adaln.rotations.iter().copied());
        Ok((
            plus(self.root.payload, rotation_bytes(&groups)?)?,
            body,
            adaln,
        ))
    }

    /// Ordinary resident-weight counterpart for the no-window MLX schedule. The full AdaLN
    /// stack is precomputed and evicted BEFORE body source verification. Native scale/codebook
    /// operands are the exception: constructors already read them for validation, and their
    /// backing arrays remain live throughout precompute. Descriptors and casts are work, not
    /// endpoint residency, and are intentionally excluded here.
    pub fn resident_peak_bytes_full(
        &self,
        task: &NativeTaskAllocation,
        backend: NativeBackend,
        rows: NativeRows,
    ) -> Result<u64> {
        let (root, body, adaln) = self.full_view(backend)?;
        let table = self.table_bytes(task, rows)?;
        let precompute = plus(plus(plus(root, body.eager_payload)?, adaln.payload)?, table)?;
        let denoise = plus(plus(root, body.payload)?, table)?;
        Ok(precompute.max(denoise))
    }

    pub fn precompute_peak_bytes_full(
        &self,
        task: &NativeTaskAllocation,
        backend: NativeBackend,
        rows: NativeRows,
    ) -> Result<u64> {
        let (root, body, adaln) = self.full_view(backend)?;
        let table = self.table_bytes(task, rows)?;
        // Root verification/refiner precedes AdaLN verification. Both native companion sets
        // are eagerly read by constructors; unverified AdaLN/body descriptor handles still exist.
        let eager = plus(body.eager_payload, adaln.eager_payload)?;
        let pending_descriptors = plus(body.descriptors, adaln.descriptors)?;
        let cold_or_refiner = plus(
            plus(plus(root, eager)?, pending_descriptors)?,
            self.root
                .descriptors
                .max(self.root_working(task, backend, rows, true)?),
        )?;
        // Final/root modulation runs after dense eviction but BEFORE body verification; body
        // companions/descriptors still coexist with this work and the complete retained cache.
        let retained_body = plus(plus(root, body.eager_payload)?, body.descriptors)?;
        let after_projection = plus(
            plus(retained_body, table)?,
            self.root_working(task, backend, rows, false)?,
        )?;
        let mut peak = cold_or_refiner.max(after_projection);
        if self.pruned {
            return Ok(peak);
        }
        let mut work = 0;
        for p in &task.projections {
            if matches!(locate(&p.base)?, Location::Adaln(_)) {
                work = work.max(
                    p.working_bytes(backend, rows).saturating_sub(
                        p.output
                            .saturating_mul(p.activation_width)
                            .saturating_mul(rows.schedule),
                    ),
                );
            }
        }
        let weights = plus(retained_body, adaln.payload)?;
        // AdaLN descriptors clear on source verification BEFORE projection work. The body
        // descriptors do not clear until the later body verification, so they are NOT maxed
        // away against AdaLN work. Cast output overlaps the active projection only once.
        let verifying = plus(weights, adaln.descriptors)?;
        let forwarding = plus(plus(weights, table)?, plus(work, adaln.forward_cast)?)?;
        peak = peak.max(verifying).max(forwarding);
        Ok(peak)
    }

    pub fn denoise_peak_bytes_full(
        &self,
        task: &NativeTaskAllocation,
        backend: NativeBackend,
        rows: NativeRows,
    ) -> Result<u64> {
        let (root, body, _) = self.full_view(backend)?;
        let base = plus(plus(root, body.payload)?, self.table_bytes(task, rows)?)?;
        let mut body_work = 0;
        for p in &task.projections {
            if matches!(locate(&p.base)?, Location::Body(_)) {
                body_work = body_work.max(p.working_bytes(backend, rows));
            }
        }
        // Verification keeps descriptor handles but does not evaluate derived dtype casts.
        // Forward starts after those handles clear, and may first evaluate one such cast.
        let working = body
            .descriptors
            .max(plus(body_work, body.forward_cast)?)
            .max(self.root_working(task, backend, rows, false)?);
        plus(base, working)
    }

    pub fn precompute_peak_bytes(
        &self,
        task: &NativeTaskAllocation,
        backend: NativeBackend,
        rows: NativeRows,
        window: usize,
    ) -> Result<u64> {
        let plan = self.plan(backend, window)?;
        let root = self.root.resident()?;
        let table = self.table_bytes(task, rows)?;
        // Root constructor descriptors are transient at load; refiner runs before AdaLN windows.
        // Root/head operations after precompute retain the full modulation table but no window.
        let cold_or_refiner = plus(
            root,
            self.root
                .descriptors
                .max(self.root_working(task, backend, rows, true)?),
        )?;
        let mut peak = cold_or_refiner.max(plus(
            plus(root, table)?,
            self.root_working(task, backend, rows, false)?,
        )?);
        if self.pruned {
            return Ok(peak);
        }
        if let NativeBackend::Candle { .. } = backend {
            let cuda = self.cuda_working.ok_or_else(invalid)?;
            if table == 0 && self.layers.iter().all(|layer| layer.adaln.payload == 0) {
                return Ok(peak);
            }
            let projections = self.cuda_adaln.as_ref().ok_or_else(invalid)?;
            // text_rows and F32 RoPE tables precede precompute; norm_out's F32 shift/scale
            // is computed once before the block loop and is retained alongside its cache.
            let carried = plus(
                plus(
                    cuda.attention_tables_bytes(rows.packed)?,
                    times(times(rows.conditioning, cuda.hidden)?, cuda.element_bytes)?,
                )?,
                times(times(rows.schedule, cuda.hidden)?, 8)?,
            )?;
            let per_layer = table / self.layers.len() as u64;
            for (index, projection) in projections.iter().enumerate() {
                if times(times(projection.output, cuda.element_bytes)?, rows.schedule)? != per_layer
                {
                    return Err(invalid());
                }
                // CUDA modulation_for_window copies and drops ONE projection per iteration,
                // even inside a multi-block callback. Only earlier blocks' cache parts exist;
                // working_bytes includes current projected output and its six returned parts.
                let prior = times(per_layer, index as u64)?;
                let live = plus(
                    plus(plus(root, carried)?, prior)?,
                    plus(
                        self.layers[index].adaln.resident()?,
                        projection.working_bytes(rows.schedule, cuda.element_bytes)?,
                    )?,
                )?;
                peak = peak.max(live);
            }
            return Ok(peak);
        }
        for range in plan.windows() {
            let partition = self.window_partition(range.clone(), true)?;
            let mut work = 0;
            for p in &task.projections {
                if let Location::Adaln(index) = locate(&p.base)? {
                    if range.contains(&index) {
                        // This output becomes a retained modulation table, not a second copy.
                        work = work.max(
                            p.working_bytes(backend, rows).saturating_sub(
                                p.output
                                    .saturating_mul(p.activation_width)
                                    .saturating_mul(rows.schedule),
                            ),
                        );
                    }
                }
            }
            // H3's one common hidden/modality width gives equal table bytes per block. The
            // cache grows until THIS aligned range's end; do not give the first window the final
            // window's cache, or combine one window's large weights with another's large work.
            let produced = times(table / self.layers.len() as u64, range.end as u64)?;
            let live = plus(
                plus(plus(root, partition.resident()?)?, produced)?,
                partition
                    .descriptors
                    .max(plus(work, partition.forward_cast)?),
            )?;
            peak = peak.max(live);
        }
        Ok(peak)
    }

    pub fn denoise_peak_bytes(
        &self,
        task: &NativeTaskAllocation,
        backend: NativeBackend,
        rows: NativeRows,
        window: usize,
    ) -> Result<u64> {
        let plan = self.plan(backend, window)?;
        let base = plus(self.root.resident()?, self.table_bytes(task, rows)?)?;
        let mut peak = plus(base, self.root_working(task, backend, rows, false)?)?;
        for range in plan.windows() {
            let partition = self.window_partition(range.clone(), false)?;
            let mut work = 0;
            for p in &task.projections {
                if let Location::Body(index) = locate(&p.base)? {
                    if range.contains(&index) {
                        work = work.max(p.working_bytes(backend, rows));
                    }
                }
            }
            if matches!(backend, NativeBackend::Candle { .. }) {
                work = work.max(
                    self.cuda_working
                        .ok_or_else(invalid)?
                        .operation_peak_with_adapter(
                            rows.packed,
                            task.adapter.as_ref().map_or(0, |a| {
                                a.forward_working_bytes_for(backend, rows, RowDomain::Packed)
                            }),
                        )?,
                );
            }
            peak = peak.max(plus(
                plus(base, partition.resident()?)?,
                partition
                    .descriptors
                    .max(plus(work, partition.forward_cast)?),
            )?);
        }
        Ok(peak)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const BACKEND: NativeBackend = NativeBackend::Mlx { tile_columns: 16 };
    fn rows() -> NativeRows {
        NativeRows {
            task: super::super::NativeTaskSelection::All,
            packed: 2,
            conditioning: 1,
            schedule: 2,
            visual_elements: 0,
            global: 0,
        }
    }
    fn task(projections: Vec<NativeProjection>, table: u64, pruned: bool) -> NativeTaskAllocation {
        NativeTaskAllocation {
            adapter: None,
            resident_bytes: u64::MAX / 2,
            constructor_bytes: 0,
            patch_size: [1, 2, 2],
            reference_task: false,
            evicted_bytes: if pruned { 0 } else { 10 },
            table_bytes_per_timestep: table,
            projections,
            window: None,
        }
    }
    fn projection(layer: usize, adaln: bool, group: u64, w4a8: bool) -> NativeProjection {
        NativeProjection {
            base: format!(
                "transformer_blocks.{layer}.{}",
                if adaln {
                    "adaln_proj.linear"
                } else {
                    "attn.to_q"
                }
            ),
            input: 16,
            output: 16,
            rotation_group: group,
            activation_width: 2,
            w4a8,
            bias_width: None,
            bias_dtype: None,
            rows: if adaln {
                RowDomain::Schedule
            } else {
                RowDomain::Packed
            },
            descriptor_bytes: 0,
        }
    }
    fn payloads(bodies: &[u64], adaln: &[u64]) -> Vec<(String, u64)> {
        let mut v = vec![("token_refiner.proj.weight".into(), 10)];
        for (i, bytes) in bodies.iter().enumerate() {
            v.push((format!("transformer_blocks.{i}.attn.to_q.weight"), *bytes));
            v.push((
                format!("transformer_blocks.{i}.adaln_proj.linear.weight"),
                adaln[i],
            ));
        }
        v
    }
    #[test]
    fn dense_window_uses_aligned_actual_payload_not_full_component() {
        let data = payloads(&[100, 200, 300, 400, 500], &[10, 20, 30, 40, 50]);
        let p = WindowAllocation::from_resident_tensors(&data, &[], 5, false).unwrap();
        let task = task(vec![], 50, false);
        assert_eq!(p.n_blocks(), 5);
        // [0,2)=300, [2,4)=700, [4,5)=500: never sliding [3,5)=900.
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 2).unwrap(),
            15 + 100 + 700
        );
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 1).unwrap(),
            15 + 100 + 500
        );
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 99).unwrap(),
            15 + 100 + 1500
        );
        assert!(p.denoise_peak_bytes(&task, BACKEND, rows(), 1).unwrap() < task.resident_bytes);
    }
    #[test]
    fn cuda_dense_windows_price_real_geometry_without_native_descriptors() {
        let data = payloads(&[100, 200], &[10, 10]);
        let cuda = NativeBackend::Candle {
            tile_rows: 16,
            workspace_bytes: 0,
        };
        let unknown = WindowAllocation::from_resident_tensors(&data, &[], 0, false).unwrap();
        let task = task(vec![], 20, false);
        assert!(unknown.denoise_peak_bytes(&task, cuda, rows(), 1).is_err());
        let facts = CudaWorkingSet {
            hidden: 16,
            heads: 2,
            head_dim: 8,
            rotary_dim: 6,
            ffn_inner: 32,
            element_bytes: 2,
            projection_decode_bytes: 512,
        };
        let profile = unknown.with_cuda_working_set(facts).unwrap();
        let small = profile.denoise_peak_bytes(&task, cuda, rows(), 1).unwrap();
        assert!(
            small
                > profile
                    .denoise_resident_bytes(&task, cuda, rows(), 1)
                    .unwrap()
        );
        let mut larger = rows();
        larger.packed = 4096;
        assert!(profile.denoise_peak_bytes(&task, cuda, larger, 1).unwrap() > small);
        assert!(profile.denoise_peak_bytes(&task, cuda, rows(), 2).unwrap() > small);
        assert_eq!(
            profile.streamed_weight_facts().unwrap().peak_bytes(Some(1)),
            210
        );
        assert!(facts.operation_peak(u64::MAX).is_err());

        // Source-lifetime regression: the second BF16 RMS norm's final cast retains
        // 4 stream inputs + xf/normed/weighted F32 (6 stream equivalents) + cast (1).
        let stock = CudaWorkingSet {
            hidden: 5376,
            heads: 56,
            head_dim: 128,
            rotary_dim: 96,
            ffn_inner: 14336,
            element_bytes: 2,
            projection_decode_bytes: 0,
        };
        let packed = 104_000;
        let inner = packed * 56 * 128 * 2;
        assert_eq!(stock.qk_norm_peak(packed, inner).unwrap(), 11 * inner + 512);
        let rotary = packed * 56 * 96 * 2;
        let cast_tables = 2 * packed * 96 * 2;
        assert_eq!(
            stock.rotary_peak(packed, inner).unwrap(),
            6 * inner + cast_tables + (5 * rotary).max(inner + 3 * rotary)
        );
        // Tables/layout survive both attention and FFN; dequantization is a separate,
        // simultaneous maximum supplied by the selected weight inventory.
        let expected = 2 * packed * 5376 * 2 + 11 * inner + 512 + packed * (8 * 96 + 64);
        assert_eq!(stock.operation_peak(packed).unwrap(), expected);
        assert_eq!(expected, 18_723_328_512);
        assert_eq!(
            CudaWorkingSet {
                element_bytes: 4,
                ..stock
            }
            .qk_norm_peak(packed, inner * 2)
            .unwrap(),
            6 * inner * 2 + 512
        );
        // Partial rotary below half a head is dominated by its final full-width cat,
        // and an F32 stream aliases its already-F32 cos/sin tables.
        let narrow = CudaWorkingSet {
            rotary_dim: 2,
            element_bytes: 4,
            ..facts
        };
        assert_eq!(narrow.rotary_peak(3, 3 * 2 * 8 * 4).unwrap(), 1488);
        // For short sequences/small heads the vendored 128-wide LSE, not a multiple
        // of inner, determines fused scratch. This fails if LSE is priced as [B,H,S].
        assert_eq!(
            CudaWorkingSet {
                hidden: 1,
                heads: 1,
                ffn_inner: 1,
                projection_decode_bytes: 0,
                ..facts
            }
            .operation_peak(1)
            .unwrap(),
            692
        );
        for rotary_dim in [0, 3, 10] {
            assert!(profile
                .clone()
                .with_cuda_working_set(CudaWorkingSet {
                    rotary_dim,
                    ..facts
                })
                .is_err());
        }
    }
    #[test]
    fn cuda_adaln_precompute_prices_current_projection_and_cache_prefix() {
        let cuda = NativeBackend::Candle {
            tile_rows: 16,
            workspace_bytes: 0,
        };
        let dims = CudaAdalnWorkingSet {
            input: 1024,
            output: 256,
            affine: true,
        };
        assert_eq!(
            dims.working_bytes(3, 2).unwrap(),
            6 * 1024 * 256 + 6 * 3 * 1024
        );
        assert_eq!(
            dims.working_bytes(3, 4).unwrap(),
            4 * 1024 * 256 + 8 * 3 * 1024 + 8 * 3 * 256
        );
        assert_eq!(
            CudaAdalnWorkingSet {
                affine: false,
                ..dims
            }
            .working_bytes(3, 2)
            .unwrap(),
            12 * 3 * 1024
        );
        assert_eq!(
            CudaAdalnWorkingSet {
                input: 64,
                output: 1152,
                affine: true
            }
            .working_bytes(100, 2)
            .unwrap(),
            646_656,
            "linear+bias/part outputs also need a bound"
        );
        let facts = CudaWorkingSet {
            hidden: 1,
            heads: 1,
            head_dim: 2,
            rotary_dim: 2,
            ffn_inner: 2,
            element_bytes: 2,
            projection_decode_bytes: 0,
        };
        let mut rows = rows();
        rows.packed = 1;
        rows.schedule = 3;
        let task = task(vec![], 2 * 256 * 2, false);
        for packed_payload in [20 * 1024 * 256 / 32, 34 * 1024 * 256 / 32] {
            let data = payloads(&[100, 100], &[packed_payload + 512, 2 * 1024 * 256 + 512]);
            let bare = WindowAllocation::from_resident_tensors(&data, &[], 0, false)
                .unwrap()
                .with_cuda_working_set(facts)
                .unwrap();
            assert!(
                bare.precompute_peak_bytes(&task, cuda, rows, 1).is_err(),
                "unknown affine/dense facts cannot become zero work"
            );
            let profile = bare
                .clone()
                .with_cuda_adaln_working_sets(vec![
                    dims,
                    CudaAdalnWorkingSet {
                        affine: false,
                        ..dims
                    },
                ])
                .unwrap();
            // The largest first packed projection overlaps NO future block-cache prefix.
            // Current outputs are in working_bytes; only the prior cache is added separately.
            let expected =
                10 + (80 + 2 + 24) + packed_payload + 512 + 6 * 1024 * 256 + 6 * 3 * 1024;
            assert_eq!(
                profile.precompute_peak_bytes(&task, cuda, rows, 1).unwrap(),
                expected
            );
            assert_eq!(
                profile.precompute_peak_bytes(&task, cuda, rows, 2).unwrap(),
                expected,
                "CUDA still copies one AdaLN projection at a time inside a block callback"
            );
            assert_eq!(
                profile
                    .precompute_peak_bytes(&task, BACKEND, rows, 1)
                    .unwrap(),
                bare.precompute_peak_bytes(&task, BACKEND, rows, 1).unwrap(),
                "MLX lifetime composition is unchanged"
            );
            assert!(bare
                .clone()
                .with_cuda_adaln_working_sets(vec![dims])
                .is_err());
            assert!(profile
                .precompute_peak_bytes(
                    &task,
                    cuda,
                    NativeRows {
                        schedule: 0,
                        ..rows
                    },
                    1
                )
                .is_err());
            assert!(profile
                .precompute_peak_bytes(
                    &task,
                    cuda,
                    NativeRows {
                        schedule: u64::MAX,
                        ..rows
                    },
                    1
                )
                .is_err());
        }
        for bad in [
            CudaAdalnWorkingSet { input: 0, ..dims },
            CudaAdalnWorkingSet { output: 0, ..dims },
            CudaAdalnWorkingSet {
                input: u64::MAX,
                ..dims
            },
        ] {
            assert!(bad.working_bytes(1, 2).is_err());
        }
        // The shipped affine matrix alone owns F32 decode plus BF16 cast, not merely BF16.
        assert_eq!(6u64 * 96768 * 2688, 1_560_674_304);
    }

    #[test]
    fn dense_precompute_cache_grows_by_actual_window_end_and_evicts_adaln() {
        let data = payloads(&[100, 100, 100], &[1000, 10, 10]);
        let p = WindowAllocation::from_resident_tensors(&data, &[], 0, false).unwrap();
        let task = task(vec![], 300, false);
        // First window: root10 + adaln1000 + 1/3 of 600byte cache =1210.
        assert_eq!(
            p.precompute_peak_bytes(&task, BACKEND, rows(), 1).unwrap(),
            1210
        );
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 1).unwrap(),
            710
        );
    }
    #[test]
    fn root_and_fresh_window_rotations_coexist_but_group_duplicates_share() {
        let data = payloads(&[100, 100], &[10, 10]);
        let mut projections = vec![
            projection(0, false, 16, false),
            projection(1, false, 16, false),
        ];
        let mut root = projection(0, false, 16, false);
        root.base = "token_refiner.proj".into();
        root.rows = RowDomain::Conditioning;
        projections.push(root);
        let p = WindowAllocation::from_resident_tensors(&data, &projections, 0, false).unwrap();
        let task = task(projections, 0, false);
        let work = task.projections[0].working_bytes(BACKEND, rows());
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 2).unwrap(),
            10 + 1024 + 200 + 1024 + work
        );
    }
    #[test]
    fn w4a8_fht_has_no_resident_group_square_and_work_is_per_operation() {
        let data = payloads(&[100_000, 200], &[10, 10]);
        let mut a = projection(0, false, 16, true);
        let mut b = projection(1, false, 16, true);
        b.input = 4096;
        // Heavy weights and heavy scratch are deliberately in DIFFERENT windows.
        a.descriptor_bytes = 0;
        let projections = vec![a, b];
        let p = WindowAllocation::from_resident_tensors(&data, &projections, 0, false).unwrap();
        let task = task(projections, 0, false);
        let expected = (110 + task.projections[0].working_bytes(BACKEND, rows()))
            .max(210 + task.projections[1].working_bytes(BACKEND, rows()));
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 1).unwrap(),
            expected
        );
        let mut changed = task.projections.clone();
        changed[1].rotation_group = 8192;
        let no_dense_matrix =
            WindowAllocation::from_resident_tensors(&data, &changed, 0, false).unwrap();
        assert_eq!(p, no_dense_matrix);
    }
    #[test]
    fn pruned_operands_compute_and_retained_cache_are_root_no_eviction() {
        let data = payloads(&[100, 200], &[30, 40]);
        let p = WindowAllocation::from_resident_tensors(&data, &[], 1000, true).unwrap();
        let task = task(vec![], 0, true);
        assert_eq!(
            p.precompute_peak_bytes(&task, BACKEND, rows(), 1).unwrap(),
            1080
        );
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 1).unwrap(),
            1280
        );
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 2).unwrap(),
            1380
        );
    }
    #[test]
    fn descriptors_are_transient_max_not_permanent_or_all_layer_sum() {
        let data = payloads(&[100, 100], &[10, 10]);
        let mut a = projection(0, false, 16, true);
        a.descriptor_bytes = 9000;
        let mut b = projection(1, false, 16, true);
        b.descriptor_bytes = 8000;
        let projections = vec![a, b];
        let p = WindowAllocation::from_resident_tensors(&data, &projections, 0, false).unwrap();
        let task = task(projections, 0, false);
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 1).unwrap(),
            10 + 100 + 9000
        );
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 2).unwrap(),
            10 + 200 + 17000
        );
    }
    #[test]
    fn ordinary_resident_floor_has_actual_cache_groups_but_no_descriptors_or_work() {
        let data = payloads(&[100, 200], &[1000, 10]);
        let mut a = projection(0, false, 16, false);
        a.descriptor_bytes = 9000;
        let mut b = projection(1, false, 16, false);
        b.input = 4096;
        let projections = vec![a, b];
        let p = WindowAllocation::from_resident_tensors(&data, &projections, 0, false).unwrap();
        let task = task(projections, 200, false);
        // Body1 200+matrix1024+fullcache400 exceeds AdaLN0 1000+prefixcache200.
        // Root10 is held throughout. The heavy body's scratch/descriptor is NOT a weight floor.
        let expected = 10 + 200 + 1024 + 400;
        assert_eq!(
            p.resident_peak_bytes(&task, BACKEND, rows(), 1).unwrap(),
            expected
        );
        assert!(p.denoise_peak_bytes(&task, BACKEND, rows(), 1).unwrap() > expected);
        let mut more_schedule = rows();
        more_schedule.schedule = 4;
        assert!(
            p.resident_peak_bytes(&task, BACKEND, more_schedule, 1)
                .unwrap()
                > expected
        );
    }
    #[test]
    fn staged_full_dense_precompute_does_not_hold_lazy_body_stack() {
        let data = payloads(&[100, 200], &[1000, 20]);
        let profile = WindowAllocation::from_resident_tensors(&data, &[], 0, false).unwrap();
        let task = task(vec![], 200, false);
        assert_eq!(
            profile
                .precompute_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            10 + 1020 + 400
        );
        assert_eq!(
            profile
                .denoise_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            10 + 300 + 400
        );
        assert_eq!(
            profile
                .resident_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            10 + 1020 + 400
        );
        assert!(
            profile
                .resident_peak_bytes_full(&task, BACKEND, rows())
                .unwrap()
                < 10 + 1020 + 300 + 400
        );
    }
    #[test]
    fn staged_full_eager_companions_pending_descriptors_and_single_view_groups() {
        let mut data = payloads(&[100, 200], &[1000, 20]);
        data.extend([
            ("transformer_blocks.0.attn.to_q.weight_scale".into(), 7),
            ("transformer_blocks.1.attn.to_q.weight_scale".into(), 11),
            (
                "transformer_blocks.0.adaln_proj.linear.weight_scale".into(),
                5,
            ),
        ]);
        let mut a = projection(0, false, 16, false);
        a.descriptor_bytes = 3;
        let mut b = projection(1, false, 16, false);
        b.descriptor_bytes = 4;
        let mut adaln = projection(0, true, 16, false);
        adaln.descriptor_bytes = 6;
        let mut root_projection = projection(0, false, 16, false);
        root_projection.base = "token_refiner.proj".into();
        root_projection.rows = RowDomain::Conditioning;
        root_projection.descriptor_bytes = 2;
        let projections = vec![a, b, adaln, root_projection];
        let profile =
            WindowAllocation::from_resident_tensors(&data, &projections, 0, false).unwrap();
        let task = task(projections, 200, false);
        let root = 10 + 4 * 16 * 16; // ONE whole-view group shared by root, both bodies, AdaLN.
        let table = 400;
        let adaln_work = task.projections[2].working_bytes(BACKEND, rows()) - 16 * 2 * 2;
        let expected_precompute = root + (7 + 11) + (3 + 4) + (1000 + 20 + 5) + table + adaln_work;
        assert_eq!(
            profile
                .precompute_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            expected_precompute
        );
        assert_eq!(
            profile
                .resident_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            root + (7 + 11) + (1000 + 20 + 5) + table
        );
        let body_work = task.projections[0].working_bytes(BACKEND, rows());
        assert_eq!(
            profile
                .denoise_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            root + (100 + 200 + 7 + 11) + table + body_work.max(3 + 4)
        );
    }
    #[test]
    fn staged_full_w4a8_companions_eager_but_codes_and_dense_bodies_lazy() {
        let mut data = payloads(&[100, 200], &[10, 20]);
        data.extend([
            ("transformer_blocks.0.attn.to_q.weight_s_rel".into(), 5),
            ("transformer_blocks.0.attn.to_q.weight_s_channel".into(), 7),
            ("transformer_blocks.0.attn.to_q.weight_codebook".into(), 11),
        ]);
        let mut projection = projection(0, false, 16, true);
        projection.descriptor_bytes = 3;
        let projections = vec![projection];
        let profile =
            WindowAllocation::from_resident_tensors(&data, &projections, 0, false).unwrap();
        let task = task(projections, 0, false);
        assert_eq!(
            profile
                .precompute_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            10 + 30 + (5 + 7 + 11) + 3
        );
        // Eager companions are a subset of body payload, never another copy at denoise.
        assert_eq!(
            profile
                .resident_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            10 + 300 + 5 + 7 + 11
        );
    }
    #[test]
    fn actual_source_casts_overlap_forward_once_but_never_resident_endpoints() {
        let mut data = payloads(&[100, 200], &[10, 20]);
        data.extend([
            ("transformer_blocks.0.ff.net.2.weight".into(), 32),
            ("transformer_blocks.1.ff.net.2.weight".into(), 32),
        ]);
        let projections = vec![
            projection(0, false, 16, true),
            projection(1, false, 16, true),
            projection(0, true, 16, true),
        ];
        let base = WindowAllocation::from_resident_tensors(&data, &projections, 0, false).unwrap();
        let casted = base
            .clone()
            .with_source_casts(&[
                ("token_refiner.proj.weight".into(), 13),
                ("transformer_blocks.0.ff.net.2.weight".into(), 17),
                ("transformer_blocks.1.ff.net.2.weight".into(), 11),
                ("transformer_blocks.1.adaln_proj.linear.weight".into(), 23),
            ])
            .unwrap();
        let task = task(projections, 64, false);
        assert_eq!(
            casted
                .resident_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            base.resident_peak_bytes_full(&task, BACKEND, rows())
                .unwrap()
        );
        assert_eq!(
            casted
                .resident_peak_bytes(&task, BACKEND, rows(), 2)
                .unwrap(),
            base.resident_peak_bytes(&task, BACKEND, rows(), 2).unwrap()
        );
        assert_eq!(
            casted
                .denoise_peak_bytes_full(&task, BACKEND, rows())
                .unwrap()
                - base
                    .denoise_peak_bytes_full(&task, BACKEND, rows())
                    .unwrap(),
            17
        );
        assert_eq!(
            casted
                .precompute_peak_bytes_full(&task, BACKEND, rows())
                .unwrap()
                - base
                    .precompute_peak_bytes_full(&task, BACKEND, rows())
                    .unwrap(),
            23
        );
        assert_eq!(
            casted
                .denoise_peak_bytes(&task, BACKEND, rows(), 2)
                .unwrap()
                - base.denoise_peak_bytes(&task, BACKEND, rows(), 2).unwrap(),
            17
        );
        assert!(base
            .with_source_casts(&[("transformer_blocks.9.ff.weight".into(), 1)])
            .is_err());
    }
    #[test]
    fn pruned_full_operands_stay_root_with_zero_eviction() {
        let data = payloads(&[100, 200], &[30, 40]);
        let profile = WindowAllocation::from_resident_tensors(&data, &[], 1000, true).unwrap();
        let task = task(vec![], 0, true);
        assert_eq!(
            profile
                .precompute_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            1080
        );
        assert_eq!(
            profile
                .denoise_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            1380
        );
        assert_eq!(
            profile
                .resident_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            1380
        );
    }
    #[test]
    fn invalid_windows_gaps_descriptors_and_overflow_are_refused() {
        let data = payloads(&[100], &[10]);
        let p = WindowAllocation::from_resident_tensors(&data, &[], 0, false).unwrap();
        let task = task(vec![], 0, false);
        assert!(p.denoise_peak_bytes(&task, BACKEND, rows(), 0).is_err());
        assert!(p
            .denoise_peak_bytes(
                &task,
                NativeBackend::Candle {
                    tile_rows: 16,
                    workspace_bytes: 0
                },
                rows(),
                1
            )
            .is_err());
        assert!(WindowAllocation::from_resident_tensors(
            &[("transformer_blocks.2.attn.weight".into(), 1)],
            &[],
            0,
            false
        )
        .is_err());
        assert!(WindowAllocation::from_resident_tensors(
            &[("transformer_blocks.X.attn.weight".into(), 1)],
            &[],
            0,
            false
        )
        .is_err());
        assert!(WindowAllocation::from_resident_tensors(
            &[("root.comfy_quant".into(), 1)],
            &[],
            0,
            false
        )
        .is_err());
        assert!(WindowAllocation::from_resident_tensors(&data, &[], u64::MAX, false).is_err());
    }
}
