//! H3's native integer allocation facts, not measured peaks or a precision conversion policy.
//! Both providers build these from the same validated descriptors their loaders consume. Source
//! payloads remain untouched. Working storage is per operation/selected task, never per-layer
//! resident weight or a second host copy of device storage on unified memory.

pub mod window;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::convrot_int8::ConvRotInt8;
use crate::w4a8_int8::W4A8Int8;
use crate::weightsmeta::{self, Dtype, SafetensorsTensorHeader};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeBackend {
    Mlx {
        tile_columns: u64,
    },
    Candle {
        tile_rows: u64,
        workspace_bytes: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowDomain {
    Packed,
    Conditioning,
    Schedule,
}

/// Actual native partition choice. All is an explicit route-independent envelope, never the
/// result of request resolution; text/keyframes use Base and unpositioned references use Reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeTaskSelection {
    Base,
    Reference,
    All,
}

impl NativeTaskSelection {
    fn includes(self, task: &NativeTaskAllocation) -> bool {
        match self {
            Self::Base => !task.reference_task,
            Self::Reference => task.reference_task,
            Self::All => true,
        }
    }
}

/// Actual rows supplied by the selected request, including reference/audio/text tokens and the
/// deduplicated precompute schedule. The consumer derives them from H3's existing geometry helpers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeRows {
    pub task: NativeTaskSelection,
    pub packed: u64,
    pub conditioning: u64,
    pub schedule: u64,
    pub visual_elements: u64,
    pub global: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeProjection {
    pub base: String,
    /// Source-only descriptor bytes held by the constructor's source map, not steady residency.
    pub descriptor_bytes: u64,
    pub input: u64,
    pub output: u64,
    pub rotation_group: u64,
    pub activation_width: u64,
    pub w4a8: bool,
    pub bias_width: Option<u64>,
    pub bias_dtype: Option<Dtype>,
    pub rows: RowDomain,
}

impl NativeProjection {
    /// Live input plus bounded source-level working tensors. Each sum names allocations that
    /// coexist; consecutive FHT stages and projection tiles take a maximum, not a sum. MLX may
    /// fuse/reuse these buffers, so this is a structural upper bound, never a measured high-water.
    /// Activation width follows the actual caller, not the F32 storage dtype of row scales.
    pub fn working_bytes(&self, backend: NativeBackend, rows: NativeRows) -> u64 {
        let full_rows = match self.rows {
            RowDomain::Packed => rows.packed,
            RowDomain::Conditioning => rows.conditioning,
            RowDomain::Schedule => rows.schedule,
        };
        if full_rows == 0 {
            return 0;
        }
        let (k, n, g) = (self.input, self.output, self.rotation_group);
        let ff_in = self.base.ends_with(".ff.net.0.proj");
        let ff_out = self.base.ends_with(".ff.net.2");
        let ff_hidden = if ff_in {
            k
        } else if ff_out {
            n
        } else {
            0
        };
        let m = if matches!(backend, NativeBackend::Candle { .. }) && (ff_in || ff_out) {
            let projected = if ff_in { n } else { k.saturating_mul(2) };
            crate::attention_budget::AttentionBudget::from_score_elements(
                FFN_INTERMEDIATE_BUDGET,
                false,
            )
            .query_block_rows(projected, full_rows)
        } else {
            full_rows
        };
        let mk = m.saturating_mul(k);
        let mn = m.saturating_mul(n);
        let e = self.activation_width;
        let input = mk.saturating_mul(e);
        let bias = match self.bias_dtype {
            None => {
                if matches!(backend, NativeBackend::Mlx { .. }) {
                    n.saturating_mul(4)
                } else {
                    0
                }
            }
            Some(Dtype::F32) if self.w4a8 || e == 4 => 0,
            Some(dtype) if !self.w4a8 && e != 4 && dtype != Dtype::BF16 => n.saturating_mul(e + 4),
            Some(_) => n.saturating_mul(4),
        };
        // One radix-4 stage: old F32 cur (4MK), ab (MK), four outputs (4MK),
        // concatenation (4MK) and scaled result (4MK). Views share cur, no dense rotation.
        let fht = mk.saturating_mul(17).saturating_add(4);
        let (rotation, projection) = match backend {
            NativeBackend::Mlx { tile_columns } => {
                let rotation = if self.w4a8 {
                    fht
                } else {
                    // Source-dtype rotated values plus the dtype-cast shared F32 rotation.
                    mk.saturating_mul(e).saturating_add(if e == 4 {
                        0
                    } else {
                        g.saturating_mul(g).saturating_mul(e)
                    })
                };
                let tile = n.min(tile_columns);
                let weight = if self.w4a8 { tile.saturating_mul(k) } else { 0 };
                // Retained rotated input, I8 activation codes, F32 row scales, F32 epilogue
                // bias, one decoded weight tile, and output tiles + final concatenation.
                let outputs = mn.saturating_mul(e);
                let tile_peak = weight.saturating_add(outputs);
                let join_peak = outputs.saturating_mul(if self.w4a8 && n > tile { 2 } else { 1 });
                let projection = mk
                    .saturating_mul(if self.w4a8 { 5 } else { e + 1 })
                    .saturating_add(m.saturating_mul(4))
                    .saturating_add(bias)
                    .saturating_add(tile_peak.max(join_peak));
                (rotation, projection)
            }
            NativeBackend::Candle { tile_rows, .. } => {
                let rotation = if self.w4a8 {
                    fht
                } else {
                    // F32 cast input + F32 dense rotation output. The shared matrix is resident.
                    mk.saturating_mul(8)
                };
                let tile = if self.w4a8 { n.min(tile_rows) } else { n };
                let mt = m.saturating_mul(tile);
                let weight = if self.w4a8 { tile.saturating_mul(k) } else { 0 };
                // qx/sx and rotated stay in scope. `acc` is F32 after an I32→F32 copy;
                // two eager scale multiplies overlap with acc, then bias/output casts overlap.
                // A biased BF16/F16 result keeps acc, the pre-bias y, the biased y and its
                // narrowed output live (4+4+4+e bytes/element). Unbiased scale folding peaks
                // at acc plus two F32 multiply outputs. W4A8 converts only the current bias tile.
                let epilogue_width = if self.bias_width.is_some() && e != 4 {
                    12 + e
                } else {
                    12
                };
                let tile_bias = if self.w4a8 { bias / n * tile } else { bias };
                // Bias-cast temporaries die before the final output narrowing. Compare those
                // actual scopes instead of adding the bias and cast high-waters together.
                let epilogue = mt
                    .saturating_mul(12)
                    .saturating_add(tile_bias)
                    .max(mt.saturating_mul(epilogue_width));
                let retained_tiles = if self.w4a8 {
                    mn.saturating_sub(mt).saturating_mul(e)
                } else {
                    0
                };
                let project = mk
                    .saturating_mul(if self.w4a8 { 5 } else { e + 1 })
                    .saturating_add(m.saturating_mul(4))
                    .saturating_add(weight)
                    .saturating_add(epilogue)
                    .saturating_add(retained_tiles);
                // Tensor::cat copies even a single retained tile. The last tile's locals have
                // left scope here: do not charge its decoded weight or accumulator again.
                let cat = mk
                    .saturating_mul(5)
                    .saturating_add(m.saturating_mul(4))
                    .saturating_add(mn.saturating_mul(e).saturating_mul(if self.w4a8 {
                        2
                    } else {
                        1
                    }));
                (rotation, project.max(cat))
            }
        };
        // FF output projection runs while the first projection's 2*K tensor and its K-wide
        // SiLU temporary are still held by forward_span/forward. The original hidden input is
        // held too; its storage is distinct from the gated K-wide activation counted above.
        let enclosing = if ff_out {
            m.saturating_mul(k.saturating_mul(3).saturating_add(n))
                .saturating_mul(e)
        } else {
            0
        };
        let operation = input
            .saturating_add(rotation.max(projection))
            .saturating_add(enclosing);
        if m == full_rows {
            return operation;
        }
        // Candle's outer FFN token loop retains preceding span outputs and finally cats them.
        // Its narrow input is a view of the full hidden input: charge that storage once, not
        // both as a full tensor and as an independently allocated current-span input.
        let outer_input = full_rows.saturating_mul(ff_hidden).saturating_mul(e);
        let prior_outputs = full_rows
            .saturating_sub(m)
            .saturating_mul(ff_hidden)
            .saturating_mul(e);
        let current_input_credit = m.saturating_mul(ff_hidden).saturating_mul(e);
        let span_peak = outer_input
            .saturating_add(prior_outputs)
            .saturating_add(operation.saturating_sub(current_input_credit));
        let cat_peak = outer_input.saturating_add(
            2u64.saturating_mul(full_rows)
                .saturating_mul(ff_hidden)
                .saturating_mul(e),
        );
        span_peak.max(cat_peak)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeTaskAllocation {
    /// Includes native payloads, shared device context/rotations, dense companions and retained
    /// source-pruned operands/tables. The provider applies its existing SourcePruneSizing first.
    pub resident_bytes: u64,
    /// Eager Candle source map plus actually derived device weights before that map drops.
    pub constructor_bytes: u64,
    pub patch_size: [u64; 3],
    pub reference_task: bool,
    pub window: Option<window::WindowAllocation>,
    pub evicted_bytes: u64,
    /// Dense modulation cache bytes per DISTINCT timestep, already including the modality axis.
    /// Source-pruned retained tables are included by SourcePruneSizing in resident_bytes instead.
    pub table_bytes_per_timestep: u64,
    pub projections: Vec<NativeProjection>,
    /// Selected-task adapter originals/caches and allocation lifetimes. Steady overlay is
    /// reported separately by the provider and added once by the consumer.
    pub adapter: Option<crate::h3_adapter_memory::H3AdapterMemoryProfile>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeAllocationFacts {
    pub conditioning: Option<crate::h3_conditioning_memory::H3ConditioningMemoryFacts>,
    pub conditioning_stream: Option<crate::StreamedWeightFacts>,
    pub conditioning_resident_bytes: u64,
    pub conditioning_load_bytes: u64,
    pub audio_decode_load_bytes: u64,
    pub video_decode_resident_bytes: u64,
    pub video_decode_load_bytes: u64,
    pub audio_decode_resident_bytes: u64,
    pub default_evaluations: usize,
    pub backend: NativeBackend,
    pub tasks: Vec<NativeTaskAllocation>,
}

impl NativeAllocationFacts {
    fn selected_tasks(&self, rows: NativeRows) -> impl Iterator<Item = &NativeTaskAllocation> {
        self.tasks
            .iter()
            .filter(move |task| rows.task.includes(task))
    }

    /// Header-selected partition geometry for conditioning and sparse admission. No weights load.
    pub fn selected_patch(&self, req: crate::veda::workspace::Request<'_>) -> Result<[usize; 3]> {
        let rows = request_rows(req, self.default_evaluations)?;
        let task = self
            .selected_tasks(rows)
            .find(|t| t.resident_bytes > 0)
            .ok_or_else(|| invalid("H3 source omits the partition selected by this request"))?;
        let patch = task.patch_size.map(|n| usize::try_from(n).unwrap_or(0));
        if patch.contains(&0) {
            return Err(invalid("invalid selected H3 patch geometry"));
        }
        Ok(patch)
    }

    pub fn adapter_resident_bytes(&self, rows: NativeRows) -> u64 {
        self.selected_tasks(rows)
            .filter_map(|t| t.adapter.as_ref())
            .map(|p| p.resident_bytes)
            .max()
            .unwrap_or(0)
    }

    // Overlay residency remains separate to preserve existing consumers' exactly-once addition.
    fn with_adapter_precompute(
        &self,
        task: &NativeTaskAllocation,
        rows: NativeRows,
        base: u64,
    ) -> u64 {
        let Some(adapter) = &task.adapter else {
            return base;
        };
        let work = adapter.forward_working_bytes_for(self.backend, rows, RowDomain::Conditioning);
        base.saturating_add(work).max(
            task.resident_bytes
                .max(task.constructor_bytes)
                .saturating_add(
                    adapter
                        .install_peak_bytes
                        .saturating_sub(adapter.resident_bytes),
                ),
        )
    }

    fn with_adapter_denoise(
        &self,
        task: &NativeTaskAllocation,
        rows: NativeRows,
        base: u64,
    ) -> u64 {
        base.saturating_add(task.adapter.as_ref().map_or(0, |p| {
            p.forward_working_bytes_for(self.backend, rows, RowDomain::Packed)
        }))
    }

    pub fn request_rows(
        &self,
        req: crate::veda::workspace::Request<'_>,
        evaluations: usize,
    ) -> Result<NativeRows> {
        let rows = request_rows(req, evaluations)?;
        let (target, _) = crate::veda::workspace::h3_visual_grids(req)?;
        let latent = [target[0], target[1] * 2, target[2] * 2];
        let mut found = false;
        for task in self.selected_tasks(rows).filter(|t| t.resident_bytes > 0) {
            found = true;
            for (axis, patch) in latent.into_iter().zip(task.patch_size) {
                if patch == 0 || !axis.is_multiple_of(patch) {
                    return Err(invalid(format!("minimax_h3: target latent {latent:?} is not divisible by source patch {:?}", task.patch_size)));
                }
            }
        }
        if !found {
            return Err(invalid(
                "H3 source omits the partition selected by this request",
            ));
        }
        Ok(rows)
    }

    fn task_rows(task: &NativeTaskAllocation, rows: NativeRows) -> NativeRows {
        if rows.visual_elements == 0 {
            return rows;
        }
        let volume = task
            .patch_size
            .iter()
            .fold(1u64, |n, &p| n.saturating_mul(p));
        NativeRows {
            packed: rows
                .global
                .saturating_add(rows.visual_elements.div_ceil(volume.max(1))),
            ..rows
        }
    }

    fn full_precompute(&self, task: &NativeTaskAllocation, rows: NativeRows) -> Result<u64> {
        if matches!(self.backend, NativeBackend::Mlx { .. }) {
            if let Some(profile) = &task.window {
                return profile.precompute_peak_bytes_full(task, self.backend, rows);
            }
        }
        let tables = task.table_bytes_per_timestep.saturating_mul(rows.schedule);
        let refiner = task
            .projections
            .iter()
            .filter(|p| p.rows == RowDomain::Conditioning)
            .map(|p| p.working_bytes(self.backend, rows))
            .max()
            .unwrap_or(0);
        let precompute = task
            .projections
            .iter()
            .filter(|p| p.rows == RowDomain::Schedule)
            .map(|p| {
                p.working_bytes(self.backend, rows).saturating_sub(
                    p.output
                        .saturating_mul(p.activation_width)
                        .saturating_mul(rows.schedule),
                )
            })
            .max()
            .unwrap_or(0);
        let descriptors = task
            .projections
            .iter()
            .fold(0u64, |n, p| n.saturating_add(p.descriptor_bytes));
        Ok(task.constructor_bytes.max(
            task.resident_bytes
                .saturating_add(descriptors.max(refiner.max(tables.saturating_add(precompute)))),
        ))
    }

    fn full_denoise(&self, task: &NativeTaskAllocation, rows: NativeRows) -> Result<u64> {
        if matches!(self.backend, NativeBackend::Mlx { .. }) {
            if let Some(profile) = &task.window {
                return profile.denoise_peak_bytes_full(task, self.backend, rows);
            }
        }
        let working = task
            .projections
            .iter()
            .filter(|p| p.rows == RowDomain::Packed)
            .map(|p| p.working_bytes(self.backend, rows))
            .max()
            .unwrap_or(0);
        Ok(task
            .resident_bytes
            .saturating_sub(task.evicted_bytes)
            .saturating_add(task.table_bytes_per_timestep.saturating_mul(rows.schedule))
            .saturating_add(working))
    }

    /// Scalar compatibility helpers fail closed at an unrepresentable bound. Admission uses the
    /// Result-returning methods below, which preserve the actual source/geometry error.
    pub fn precompute_peak_bytes(&self, rows: NativeRows) -> u64 {
        self.selected_tasks(rows)
            .map(|task| {
                self.full_precompute(task, Self::task_rows(task, rows))
                    .map(|base| {
                        self.with_adapter_precompute(task, Self::task_rows(task, rows), base)
                    })
                    .unwrap_or(u64::MAX)
            })
            .max()
            .unwrap_or(0)
    }
    pub fn denoise_peak_bytes(&self, rows: NativeRows) -> u64 {
        self.selected_tasks(rows)
            .map(|task| {
                self.full_denoise(task, Self::task_rows(task, rows))
                    .map(|base| self.with_adapter_denoise(task, Self::task_rows(task, rows), base))
                    .unwrap_or(u64::MAX)
            })
            .max()
            .unwrap_or(0)
    }

    pub fn precompute_peak_bytes_windowed(
        &self,
        rows: NativeRows,
        window: Option<u32>,
    ) -> Result<u64> {
        let Some(window) = window else {
            return self.selected_tasks(rows).try_fold(0u64, |peak, task| {
                Ok(peak.max(self.with_adapter_precompute(
                    task,
                    Self::task_rows(task, rows),
                    self.full_precompute(task, Self::task_rows(task, rows))?,
                )))
            });
        };
        self.selected_tasks(rows)
            .filter(|t| t.resident_bytes > 0)
            .try_fold(0u64, |peak, task| {
                let profile = task.window.as_ref().ok_or_else(|| {
                    invalid("H3 source has no native block-window allocation profile")
                })?;
                Ok(peak.max(self.with_adapter_precompute(
                    task,
                    Self::task_rows(task, rows),
                    profile.precompute_peak_bytes(
                        task,
                        self.backend,
                        Self::task_rows(task, rows),
                        window as usize,
                    )?,
                )))
            })
    }

    pub fn denoise_peak_bytes_windowed(
        &self,
        rows: NativeRows,
        window: Option<u32>,
    ) -> Result<u64> {
        let Some(window) = window else {
            return self.selected_tasks(rows).try_fold(0u64, |peak, task| {
                Ok(peak.max(self.with_adapter_denoise(
                    task,
                    Self::task_rows(task, rows),
                    self.full_denoise(task, Self::task_rows(task, rows))?,
                )))
            });
        };
        self.selected_tasks(rows)
            .filter(|t| t.resident_bytes > 0)
            .try_fold(0u64, |peak, task| {
                let profile = task.window.as_ref().ok_or_else(|| {
                    invalid("H3 source has no native block-window allocation profile")
                })?;
                Ok(peak.max(self.with_adapter_denoise(
                    task,
                    Self::task_rows(task, rows),
                    profile.denoise_peak_bytes(
                        task,
                        self.backend,
                        Self::task_rows(task, rows),
                        window as usize,
                    )?,
                )))
            })
    }

    /// Correct the ordinary estimator's weight floor without counting operation workspace twice.
    pub fn resident_peak_bytes_windowed(
        &self,
        rows: NativeRows,
        window: Option<u32>,
    ) -> Result<u64> {
        let Some(window) = window else {
            return self.selected_tasks(rows).try_fold(0u64, |peak, task| {
                let value = if matches!(self.backend, NativeBackend::Mlx { .. }) {
                    match &task.window {
                        Some(profile) => profile.resident_peak_bytes_full(
                            task,
                            self.backend,
                            Self::task_rows(task, rows),
                        )?,
                        None => task.resident_bytes.saturating_add(
                            task.table_bytes_per_timestep.saturating_mul(rows.schedule),
                        ),
                    }
                } else {
                    task.resident_bytes
                        .saturating_add(task.table_bytes_per_timestep.saturating_mul(rows.schedule))
                };
                Ok(peak.max(value))
            });
        };
        self.selected_tasks(rows)
            .filter(|t| t.resident_bytes > 0)
            .try_fold(0u64, |peak, task| {
                let profile = task.window.as_ref().ok_or_else(|| {
                    invalid("H3 source has no native block-window allocation profile")
                })?;
                Ok(peak.max(profile.resident_peak_bytes(
                    task,
                    self.backend,
                    Self::task_rows(task, rows),
                    window as usize,
                )?))
            })
    }

    pub fn peak_bytes(&self, rows: NativeRows) -> u64 {
        self.precompute_peak_bytes(rows)
            .max(self.denoise_peak_bytes(rows))
    }
}

/// Full request upper envelope from the same native reference/vision/audio packing bounds Veda
/// admission uses. This does not enable Veda or allocate/load a predictor. Schedule cardinality is
/// bounded across the admitted shifts by S5's existing two-stream bound, not a default grid.
pub fn request_rows(
    req: crate::veda::workspace::Request<'_>,
    evaluations: usize,
) -> Result<NativeRows> {
    validate_request_geometry(req.width, req.height, u64::from(req.frames), req.count)?;
    let references = req.conditioning.iter().any(|c| {
        matches!(
            c,
            crate::Conditioning::Reference { .. }
                | crate::Conditioning::MultiReference { .. }
                | crate::Conditioning::ReferenceVideo { .. }
                | crate::Conditioning::ReferenceAudio { .. }
        )
    });
    let keyframes = req
        .conditioning
        .iter()
        .any(|c| matches!(c, crate::Conditioning::Keyframe { .. }));
    // Match MiniMaxH3Task::resolve in both native providers. Mixing those carriers is not a task.
    if references && keyframes {
        return Err(invalid(
            "H3 requests cannot mix keyframes and unpositioned references",
        ));
    }
    let task = if references {
        NativeTaskSelection::Reference
    } else {
        NativeTaskSelection::Base
    };
    let (packed, conditioning) = crate::veda::workspace::h3_row_bound(req)?;
    let schedule = crate::pruned_adaln::SourcePruneSizing::conditioned_timestep_bound(evaluations)
        .map_err(invalid)? as u64;
    let (target, refs) = crate::veda::workspace::h3_visual_grids(req)?;
    let visual_elements =
        refs.iter()
            .chain(std::iter::once(&target))
            .try_fold(0u64, |n, grid| {
                let volume = grid
                    .iter()
                    .try_fold(4u64, |v, &axis| v.checked_mul(axis))
                    .ok_or_else(|| invalid("H3 visual size overflow"))?;
                n.checked_add(volume)
                    .ok_or_else(|| invalid("H3 visual size overflow"))
            })?;
    let global = packed.saturating_sub(visual_elements / 4);
    Ok(NativeRows {
        task,
        packed,
        conditioning,
        schedule,
        visual_elements,
        global,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeComponentAllocation {
    pub resident_by_name: Vec<(String, u64)>,
    pub phase_resident_by_name: Vec<(String, u64)>,
    pub source_cast_by_name: Vec<(String, u64)>,
    pub constructor_bytes: u64,
    pub resolved: bool,
    pub resident_bytes: u64,
    pub adaln_bytes: u64,
    pub table_bytes_per_timestep: u64,
    pub projections: Vec<NativeProjection>,
}

/// Header-only dense projection plus bounded reads of authoritative descriptor blobs. `width`
/// is supplied by the owning loader, not inferred from precision labels; None preserves source
/// dtype for mixed-precision heads/native companions. Missing components are unresolved, not errors.
pub fn component_allocation(
    path: &Path,
    backend: NativeBackend,
    activation_width: u64,
    width: impl Fn(&SafetensorsTensorHeader) -> Option<u64>,
) -> Result<NativeComponentAllocation> {
    if weightsmeta::safetensors_path_bytes(path) == 0 {
        return Ok(NativeComponentAllocation {
            resident_by_name: Vec::new(),
            phase_resident_by_name: Vec::new(),
            source_cast_by_name: Vec::new(),
            constructor_bytes: 0,
            resolved: false,
            resident_bytes: 0,
            adaln_bytes: 0,
            table_bytes_per_timestep: 0,
            projections: Vec::new(),
        });
    }
    let headers = weightsmeta::safetensors_path_tensor_headers(path)?;
    let payloads = weightsmeta::read_safetensors_path_tensor_payloads(
        path,
        |h| h.name.ends_with(".comfy_quant"),
        4096,
    )?;
    allocation_from_headers(&headers, &payloads, backend, activation_width, width)
}

/// The actual Candle H3 outer FFN token-span planner's element budget.
pub const FFN_INTERMEDIATE_BUDGET: u64 = 1_000_000_000;
/// H3's packed audio axis is channel-major over both output channels.
pub const AUDIO_OUTPUT_CHANNELS: u16 = 2;
pub const SPATIAL_STRIDE: u32 = 32;
pub const CANVAS_MAX_PIXELS: u32 = 768 * 1344;
/// The short edge a resolved MiniMax-H3 canvas is laid at before the [`CANVAS_MAX_PIXELS`] cap.
pub const CANVAS_SHORT_EDGE: u32 = 768;

/// Shared H3 request laws used by both native geometry resolvers and allocation admission.
/// Reject rather than rounding an unrenderable request into a smaller memory estimate.
pub fn validate_request_geometry(width: u32, height: u32, frames: u64, count: u32) -> Result<()> {
    if count != 1 {
        return Err(invalid("minimax_h3: batch count must be 1"));
    }
    if width == 0 || height == 0 {
        return Err(invalid(format!(
            "minimax_h3: width/height must be positive, got {width}x{height}"
        )));
    }
    if !width.is_multiple_of(SPATIAL_STRIDE) || !height.is_multiple_of(SPATIAL_STRIDE) {
        return Err(invalid(format!(
            "minimax_h3: width/height must be a multiple of {SPATIAL_STRIDE}, got {width}x{height}"
        )));
    }
    if u64::from(width) * u64::from(height) > u64::from(CANVAS_MAX_PIXELS) {
        return Err(invalid(format!(
            "minimax_h3: {width}x{height} is over the {CANVAS_MAX_PIXELS} px canvas budget"
        )));
    }
    if frames < 5 || !(frames - 5).is_multiple_of(17) {
        return Err(invalid(format!(
            "minimax_h3: {frames} frames must be on the 17n + 5 lattice"
        )));
    }
    if !(124..=345).contains(&frames) {
        return Err(invalid(format!(
            "minimax_h3: {frames} frames is outside the model's 124-345 frame range"
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Unsupported(message.into())
}

pub fn allocation_from_headers(
    headers: &[SafetensorsTensorHeader],
    payloads: &BTreeMap<String, Vec<u8>>,
    backend: NativeBackend,
    activation_width: u64,
    width: impl Fn(&SafetensorsTensorHeader) -> Option<u64>,
) -> Result<NativeComponentAllocation> {
    let by_name = headers
        .iter()
        .map(|h| (h.name.as_str(), h))
        .collect::<BTreeMap<_, _>>();
    let mut native = BTreeSet::new();
    let mut rotations = BTreeSet::new();
    let mut projections = Vec::new();
    let mut copied_native_companions = 0u64;
    for h in headers {
        for suffix in [
            ".weight_scale",
            ".weight_s_rel",
            ".weight_s_channel",
            ".weight_codebook",
        ] {
            if let Some(base) = h.name.strip_suffix(suffix) {
                if !by_name.contains_key(format!("{base}.comfy_quant").as_str()) {
                    return Err(invalid(format!(
                        "{base}: native operands require a descriptor"
                    )));
                }
            }
        }
        if h.dtype == Dtype::I8 && h.name.ends_with(".weight") {
            let base = h.name.strip_suffix(".weight").unwrap();
            if !by_name.contains_key(format!("{base}.comfy_quant").as_str()) {
                return Err(invalid(format!(
                    "{base}: native I8 codes require a descriptor"
                )));
            }
        }
    }
    let mut resident_bytes = 0u64;
    let mut resident_by_name = Vec::new();
    let mut adaln_bytes = 0u64;
    let mut table_bytes_per_timestep = 0u64;
    for h in headers.iter().filter(|h| h.name.ends_with(".comfy_quant")) {
        let base = h.name.strip_suffix(".comfy_quant").unwrap();
        let payload = payloads
            .get(&h.name)
            .ok_or_else(|| invalid("missing native descriptor payload"))?;
        if h.dtype != Dtype::U8 || h.shape != [payload.len()] {
            return Err(invalid(format!(
                "{}: invalid native descriptor header",
                h.name
            )));
        }
        let get = |suffix: &str, dtype: Dtype| -> Result<&SafetensorsTensorHeader> {
            let key = format!("{base}.{suffix}");
            let tensor = by_name
                .get(key.as_str())
                .ok_or_else(|| invalid(format!("missing {key}")))?;
            if tensor.dtype != dtype {
                return Err(invalid(format!("{key}: wrong native storage dtype")));
            }
            Ok(tensor)
        };
        let w = get("weight", Dtype::I8)?;
        if by_name.contains_key(format!("{base}.scales").as_str())
            || by_name.contains_key(format!("{base}.biases").as_str())
        {
            return Err(invalid(format!(
                "{base}: contradictory native/affine operands"
            )));
        }
        let value: serde_json::Value =
            serde_json::from_slice(payload).map_err(|e| invalid(e.to_string()))?;
        let w4a8 = value.get("format").and_then(|v| v.as_str()) == Some("asym_w4a8_int8");
        if w4a8 && by_name.contains_key(format!("{base}.weight_scale").as_str()) {
            return Err(invalid(format!(
                "{base}: contradictory W4A8/ConvRot operands"
            )));
        }
        if !w4a8
            && ["weight_s_rel", "weight_s_channel", "weight_codebook"]
                .iter()
                .any(|suffix| by_name.contains_key(format!("{base}.{suffix}").as_str()))
        {
            return Err(invalid(format!(
                "{base}: contradictory ConvRot/W4A8 operands"
            )));
        }
        let (n, k, group, bytes, companions) = if w4a8 {
            let rel = get("weight_s_rel", Dtype::F8_E4M3)?;
            let channel = get("weight_s_channel", Dtype::F32)?;
            let book = get("weight_codebook", Dtype::F32)?;
            let d = W4A8Int8::parse(payload, &w.shape, &rel.shape, &channel.shape, &book.shape)
                .map_err(invalid)?;
            if matches!(backend, NativeBackend::Candle { .. }) {
                // W4A8Linear rebuilds these companions with Tensor::from_vec. Every model-owned
                // replacement survives while the original complete device map is still held.
                // This accumulates across projections, unlike the one-current decode tile.
                copied_native_companions = copied_native_companions
                    .checked_add(rel.data_bytes)
                    .and_then(|n| n.checked_add(book.data_bytes))
                    .ok_or_else(|| invalid("native constructor companion size overflow"))?;
            }
            (
                d.out_features,
                d.in_features,
                d.rotation_group,
                d.resident_bytes(),
                vec!["weight_s_rel", "weight_s_channel", "weight_codebook"],
            )
        } else {
            let scales = get("weight_scale", Dtype::F32)?;
            let d = ConvRotInt8::parse(payload, &w.shape, &scales.shape).map_err(invalid)?;
            rotations.insert(d.rotation_group as u64);
            (
                d.out_features,
                d.in_features,
                d.rotation_group,
                d.resident_bytes(),
                vec!["weight_scale"],
            )
        };
        if matches!(backend, NativeBackend::Candle { .. }) && !n.is_multiple_of(16) {
            return Err(invalid(format!(
                "{base}: native CUDA output width must be a multiple of 16"
            )));
        }
        for suffix in ["weight", "comfy_quant"].into_iter().chain(companions) {
            let name = format!("{base}.{suffix}");
            if suffix != "comfy_quant" {
                resident_by_name.push((name.clone(), by_name[name.as_str()].data_bytes));
            }
            native.insert(name);
        }
        let before = resident_bytes;
        let bias_width = if let Some(bias) = by_name.get(format!("{base}.bias").as_str()) {
            if !matches!(bias.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32) || bias.shape != [n] {
                return Err(invalid("invalid native bias"));
            }
            native.insert(bias.name.clone());
            resident_by_name.push((bias.name.clone(), bias.data_bytes));
            resident_bytes = resident_bytes
                .checked_add(bias.data_bytes)
                .ok_or_else(|| invalid("native byte overflow"))?;
            Some(bias.data_bytes / n as u64)
        } else {
            None
        };
        resident_bytes = resident_bytes
            .checked_add(bytes as u64)
            .ok_or_else(|| invalid("native byte overflow"))?;
        let activation_width = if w4a8
            && base.ends_with(".adaln_proj.linear")
            && matches!(backend, NativeBackend::Mlx { .. })
        {
            2
        } else {
            activation_width
        };
        if base.ends_with(".adaln_proj.linear") {
            adaln_bytes = adaln_bytes.saturating_add(resident_bytes - before);
            table_bytes_per_timestep =
                table_bytes_per_timestep.saturating_add(n as u64 * activation_width);
        }
        projections.push(NativeProjection {
            base: base.to_owned(),
            descriptor_bytes: h.data_bytes,
            input: k as u64,
            output: n as u64,
            rotation_group: group as u64,
            activation_width,
            w4a8,
            bias_width,
            bias_dtype: by_name
                .get(format!("{base}.bias").as_str())
                .map(|h| h.dtype),
            rows: if base.ends_with(".adaln_proj.linear") {
                RowDomain::Schedule
            } else if base.starts_with("token_refiner.") {
                RowDomain::Conditioning
            } else {
                RowDomain::Packed
            },
        });
    }
    // Affine packs are already encoded. Neither their codes nor scale/bias companions follow a
    // dense materialization policy. Native descriptors are source-only, not resident arrays.
    let affine = headers
        .iter()
        .filter_map(|h| h.name.strip_suffix(".scales"))
        .collect::<BTreeSet<_>>();
    for h in headers.iter().filter(|h| !native.contains(&h.name)) {
        let packed = [".weight", ".scales", ".biases"].iter().any(|suffix| {
            h.name
                .strip_suffix(suffix)
                .is_some_and(|base| affine.contains(base))
        });
        let bytes = if packed {
            h.data_bytes
        } else if let Some(w) = width(h).filter(|_| h.is_float()) {
            h.materialized_bytes(w)?
        } else {
            h.data_bytes
        };
        resident_bytes = resident_bytes
            .checked_add(bytes)
            .ok_or_else(|| invalid("component byte overflow"))?;
        resident_by_name.push((h.name.clone(), bytes));
        if h.name.contains(".adaln_proj.linear.") {
            adaln_bytes = adaln_bytes.saturating_add(bytes);
            if h.name.ends_with(".weight") {
                if let Some(&n) = h.shape.first() {
                    let output_width = if packed && matches!(backend, NativeBackend::Mlx { .. }) {
                        let base = h.name.strip_suffix(".weight").unwrap();
                        by_name[format!("{base}.scales").as_str()].dtype.size() as u64
                    } else {
                        activation_width
                    };
                    table_bytes_per_timestep =
                        table_bytes_per_timestep.saturating_add(n as u64 * output_width);
                }
            }
        }
    }
    let tensor_bytes = resident_by_name.iter().try_fold(0u64, |n, (_, bytes)| {
        n.checked_add(*bytes)
            .ok_or_else(|| invalid("tensor size overflow"))
    })?;
    let mut phase_resident_by_name = resident_by_name.clone();
    let mut source_cast_by_name = Vec::new();
    let mut derived = 0u64;
    for (name, bytes) in &mut phase_resident_by_name {
        let h = by_name[name.as_str()];
        let packed = [".weight", ".scales", ".biases"].iter().any(|suffix| {
            h.name
                .strip_suffix(suffix)
                .is_some_and(|base| affine.contains(base))
        });
        if native.contains(name) || packed || !h.is_float() {
            continue;
        }
        if let Some(w) = width(h) {
            let target = if w == 4 { Dtype::F32 } else { Dtype::BF16 };
            if h.dtype != target {
                if matches!(backend, NativeBackend::Mlx { .. }) {
                    source_cast_by_name.push((name.clone(), h.data_bytes.min(*bytes)));
                    *bytes = (*bytes).max(h.data_bytes);
                } else {
                    derived = derived
                        .checked_add(*bytes)
                        .ok_or_else(|| invalid("constructor size overflow"))?;
                }
            }
        }
    }
    for g in rotations {
        resident_bytes = resident_bytes
            .checked_add(4 * g * g)
            .ok_or_else(|| invalid("rotation byte overflow"))?;
    }
    if !projections.is_empty() {
        if let NativeBackend::Candle {
            workspace_bytes, ..
        } = backend
        {
            resident_bytes = resident_bytes
                .checked_add(workspace_bytes)
                .ok_or_else(|| invalid("workspace byte overflow"))?;
        }
    }
    let constructor_bytes = if matches!(backend, NativeBackend::Candle { .. }) {
        let source = headers.iter().try_fold(0u64, |n, h| {
            n.checked_add(h.data_bytes)
                .ok_or_else(|| invalid("source size overflow"))
        })?;
        source
            .checked_add(derived)
            .and_then(|n| n.checked_add(copied_native_companions))
            .and_then(|n| n.checked_add(resident_bytes - tensor_bytes))
            .ok_or_else(|| invalid("constructor size overflow"))?
    } else {
        0
    };
    Ok(NativeComponentAllocation {
        resident_by_name,
        phase_resident_by_name,
        source_cast_by_name,
        constructor_bytes,
        resolved: headers
            .iter()
            .any(|h| h.name.ends_with(".weight") || h.name == "adaln_t_table"),
        resident_bytes,
        adaln_bytes,
        table_bytes_per_timestep,
        projections,
    })
}

/// Source/compute dtype casts at the VAE load boundary. MLX's preloaded source arrays are
/// consumed progressively; Candle casts one loaded tensor at a time. Bound the larger resident
/// endpoint plus one actual cast operand/output, never a second full host copy or host RSS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VideoAllocation {
    pub resident_bytes: u64,
    pub load_extra_bytes: u64,
    pub forward_extra_bytes: u64,
}

pub fn video_allocation(
    headers: &[SafetensorsTensorHeader],
    target: Dtype,
    lazy_source: bool,
) -> Result<VideoAllocation> {
    let width = target.size() as u64;
    let final_bytes =
        weightsmeta::materialized_header_bytes(headers, width, Path::new("H3 video VAE"))?;
    let mut source = 0u64;
    let mut cast = 0u64;
    for h in headers {
        source = source
            .checked_add(h.data_bytes)
            .ok_or_else(|| invalid("VAE source byte overflow"))?;
        if h.is_float() && h.dtype != target {
            cast = cast.max(if lazy_source {
                h.data_bytes.min(h.materialized_bytes(width)?)
            } else {
                h.data_bytes
            });
        }
    }
    Ok(VideoAllocation {
        resident_bytes: if lazy_source {
            source.max(final_bytes)
        } else {
            final_bytes
        },
        load_extra_bytes: cast,
        forward_extra_bytes: if lazy_source { cast } else { 0 },
    })
}

/// Weight allocations in the audio constructor/forward. Canonical native tensor-name lists select
/// the readset. Weight-norm g/v become one v-shaped fused weight; Snake log parameters are F32.
/// Source maps and derivatives are charged only when they actually coexist, not as host RSS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AudioAllocation {
    pub encoder_resident: u64,
    pub decoder_resident: u64,
    pub encoder_load: u64,
    pub decoder_load: u64,
    pub encoder_forward_extra: u64,
    pub decoder_forward_extra: u64,
}

pub fn audio_allocation(
    path: &Path,
    encoder_names: &[String],
    decoder_names: &[String],
    decode_width: u64,
    mlx: bool,
    latent_channels: u64,
) -> Result<AudioAllocation> {
    if weightsmeta::safetensors_path_bytes(path) == 0 {
        return Ok(AudioAllocation::default());
    }
    let headers = weightsmeta::safetensors_path_tensor_headers(path)?;
    let by_name = headers
        .iter()
        .map(|h| (h.name.as_str(), h))
        .collect::<BTreeMap<_, _>>();
    let size = |names: &[String], width: u64, keep_source: bool| -> Result<(u64, u64, u64)> {
        let mut source = 0u64;
        let mut final_bytes = 0u64;
        let mut additional = 0u64;
        let mut largest_fusion = 0u64;
        let mut largest_cast = 0u64;
        for name in names {
            let h = by_name
                .get(name.as_str())
                .ok_or_else(|| invalid(format!("missing audio operand {name}")))?;
            if !h.is_float() {
                return Err(invalid(format!("non-float audio operand {name}")));
            }
            source = source.saturating_add(h.data_bytes);
            if name.ends_with(".weight_g") {
                continue;
            }
            let snake = name.ends_with(".act.alpha") || name.ends_with(".act.beta");
            let target_width = if snake { 4 } else { width };
            let resident = h.materialized_bytes(target_width)?;
            final_bytes = final_bytes.saturating_add(resident);
            let fused = name.ends_with(".weight_v");
            if fused {
                // F32 v/norm/fused constructor tensors, bounded by the largest projection,
                // never by all layers' normalization work at once.
                largest_fusion = largest_fusion.max(
                    h.materialized_bytes(4)?
                        .saturating_mul(3)
                        .saturating_add(h.shape.first().copied().unwrap_or(0) as u64 * 8),
                );
            }
            let map_width = if mlx { h.dtype.size() as u64 } else { width };
            if fused || map_width != target_width {
                additional = additional.saturating_add(resident);
            }
            if map_width != target_width {
                largest_cast = largest_cast.max(resident);
            }
        }
        let normalization = latent_channels.saturating_mul(width).saturating_mul(2);
        let resident = if keep_source {
            source.saturating_add(additional)
        } else if mlx {
            source.max(final_bytes)
        } else {
            final_bytes
        };
        // Candle eagerly loads the complete map at its requested dtype, including the unused half;
        // MLX materializes only the constructor readset at source dtype.
        let load_source = if mlx {
            source
        } else {
            weightsmeta::materialized_header_bytes(&headers, width, path)?
        };
        let constructor_work = additional
            .saturating_add(largest_fusion)
            .saturating_add(normalization);
        let file_cast = if mlx {
            0
        } else {
            let target = if width == 4 { Dtype::F32 } else { Dtype::BF16 };
            headers
                .iter()
                .filter(|h| h.is_float() && h.dtype != target)
                .map(|h| h.data_bytes)
                .max()
                .unwrap_or(0)
        };
        let load = load_source.saturating_add(constructor_work.max(file_cast));
        let forward_extra = if mlx {
            largest_fusion.max(largest_cast)
        } else {
            0
        };
        Ok((resident.saturating_add(normalization), load, forward_extra))
    };
    let (encoder_resident, encoder_load, encoder_forward_extra) = size(encoder_names, 4, false)?;
    let (decoder_resident, decoder_load, decoder_forward_extra) =
        size(decoder_names, decode_width, mlx)?;
    Ok(AudioAllocation {
        encoder_resident,
        decoder_resident,
        encoder_load,
        decoder_load,
        encoder_forward_extra,
        decoder_forward_extra,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(name: impl Into<String>, dtype: Dtype, shape: &[usize]) -> SafetensorsTensorHeader {
        SafetensorsTensorHeader {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            data_bytes: (shape.iter().product::<usize>() * dtype.size()) as u64,
        }
    }
    fn fixture(
        w4a8: bool,
        groups: &[usize],
    ) -> (Vec<SafetensorsTensorHeader>, BTreeMap<String, Vec<u8>>) {
        let mut headers = Vec::new();
        let mut payloads = BTreeMap::new();
        for (i, &group) in groups.iter().enumerate() {
            let base = format!("transformer_blocks.{i}.attn.to_q");
            let payload = if w4a8 {
                br#"{"format":"asym_w4a8_int8","group_size":16,"convrot_groupsize":256}"#.to_vec()
            } else {
                format!(
                    r#"{{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":{group}}}"#
                )
                .into_bytes()
            };
            let k = group * 2;
            headers.push(header(
                format!("{base}.weight"),
                Dtype::I8,
                &[32, if w4a8 { k / 2 } else { k }],
            ));
            headers.push(header(
                format!("{base}.comfy_quant"),
                Dtype::U8,
                &[payload.len()],
            ));
            if w4a8 {
                headers.push(header(
                    format!("{base}.weight_s_rel"),
                    Dtype::F8_E4M3,
                    &[32, k / 16],
                ));
                headers.push(header(
                    format!("{base}.weight_s_channel"),
                    Dtype::F32,
                    &[32],
                ));
                headers.push(header(format!("{base}.weight_codebook"), Dtype::F32, &[16]));
            } else {
                headers.push(header(format!("{base}.weight_scale"), Dtype::F32, &[32, 1]));
            }
            payloads.insert(format!("{base}.comfy_quant"), payload);
        }
        (headers, payloads)
    }
    const MLX: NativeBackend = NativeBackend::Mlx { tile_columns: 256 };
    const CUDA: NativeBackend = NativeBackend::Candle {
        tile_rows: 64,
        workspace_bytes: 32 << 20,
    };
    const ROWS: NativeRows = NativeRows {
        task: NativeTaskSelection::All,
        packed: 17,
        conditioning: 11,
        schedule: 5,
        visual_elements: 0,
        global: 0,
    };

    #[test]
    fn native_resident_codes_scales_descriptor_exclusion_and_unique_rotations() {
        let (headers, payloads) = fixture(false, &[64, 256, 64, 256]);
        let expected =
            2 * (32 * 128 + 4 * 32) + 2 * (32 * 512 + 4 * 32) + 4 * (64 * 64 + 256 * 256);
        for width in [2, 4] {
            let mlx =
                allocation_from_headers(&headers, &payloads, MLX, width, |_| Some(width)).unwrap();
            assert_eq!(
                mlx.resident_bytes, expected,
                "native I8/F32 remains original width; descriptor is source-only"
            );
            let cuda =
                allocation_from_headers(&headers, &payloads, CUDA, width, |_| Some(width)).unwrap();
            assert_eq!(
                cuda.resident_bytes,
                expected + (32 << 20),
                "one actual context allocation, not one per layer"
            );
        }
    }

    #[test]
    fn w4a8_mixed_companion_preserves_payloads_without_resident_fht_rotation() {
        let (mut headers, mut payloads) = fixture(true, &[256, 256]);
        let (other, descriptors) = fixture(false, &[64]);
        for mut h in other {
            h.name = h
                .name
                .replacen("transformer_blocks.0", "token_refiner.0", 1);
            headers.push(h);
        }
        payloads.extend(descriptors.into_iter().map(|(key, value)| {
            (
                key.replacen("transformer_blocks.0", "token_refiner.0", 1),
                value,
            )
        }));
        headers.push(header("proj_in.weight", Dtype::F32, &[4, 4]));
        headers.push(header("dense16.weight", Dtype::F16, &[4, 4]));
        headers.push(header("densebf.weight", Dtype::BF16, &[4, 4]));
        let narrow = allocation_from_headers(&headers, &payloads, MLX, 2, |_| None).unwrap();
        assert_eq!(
            narrow.resident_bytes,
            2 * (32 * 512 / 2 + 32 * 512 / 16 + 4 * 32 + 64)
                + (32 * 128 + 4 * 32)
                + 4 * 64 * 64
                + (64 + 32 + 32)
        );
        let wide = allocation_from_headers(&headers, &payloads, MLX, 4, |_| Some(4)).unwrap();
        assert_eq!(
            wide.resident_bytes - narrow.resident_bytes,
            64,
            "only F16/BF16 dense companions widen; native tables/codebooks stay F32"
        );
        assert_eq!(narrow.projections[0].rows, RowDomain::Packed);
        assert_eq!(narrow.projections[2].rows, RowDomain::Conditioning);
    }

    #[test]
    fn working_storage_is_dynamic_and_maximum_of_operations_not_layer_scratch_sum() {
        let (headers, payloads) = fixture(true, &[256, 256]);
        let allocation = allocation_from_headers(&headers, &payloads, MLX, 2, |_| None).unwrap();
        let op = allocation.projections[0].clone();
        let task = NativeTaskAllocation {
            adapter: None,
            patch_size: [1, 2, 2],
            reference_task: false,
            window: None,
            resident_bytes: allocation.resident_bytes,
            constructor_bytes: 0,
            evicted_bytes: 0,
            table_bytes_per_timestep: 0,
            projections: allocation.projections,
        };
        let mut facts = NativeAllocationFacts {
            conditioning_stream: None,
            conditioning: None,
            conditioning_resident_bytes: 0,
            conditioning_load_bytes: 0,
            audio_decode_load_bytes: 0,
            video_decode_resident_bytes: 0,
            video_decode_load_bytes: 0,
            audio_decode_resident_bytes: 0,
            default_evaluations: 50,
            backend: MLX,
            tasks: vec![task],
        };
        assert_eq!(
            facts.denoise_peak_bytes(ROWS),
            facts.tasks[0].resident_bytes + op.working_bytes(MLX, ROWS)
        );
        let before = facts.denoise_peak_bytes(ROWS);
        facts.tasks[0].projections.push(op.clone());
        assert_eq!(
            before,
            facts.denoise_peak_bytes(ROWS),
            "sequential projections add no concurrent scratch"
        );
        let bigger = NativeRows {
            packed: 2 * ROWS.packed,
            ..ROWS
        };
        assert!(facts.denoise_peak_bytes(bigger) > before);
        let mut group = op;
        group.rotation_group = 64;
        assert_eq!(
            group.working_bytes(MLX, ROWS),
            facts.tasks[0].projections[0].working_bytes(MLX, ROWS),
            "F32 FHT has no group-squared resident rotation"
        );
        assert!(group.working_bytes(CUDA, ROWS) > 0);
    }

    #[test]
    fn fht_and_final_partial_tile_lifetimes_have_discriminating_formulas() {
        let op = NativeProjection {
            base: "x".into(),
            descriptor_bytes: 0,
            input: 256,
            output: 16,
            rotation_group: 256,
            activation_width: 2,
            w4a8: true,
            bias_width: None,
            bias_dtype: None,
            rows: RowDomain::Packed,
        };
        let one = NativeRows { packed: 1, ..ROWS };
        assert_eq!(
            op.working_bytes(MLX, one),
            2 * 256 + 5 * 256 + 4 + 4 * 16 + 16 * 256 + 2 * 16,
            "bounded final partial decoded tile, not 256*K scratch"
        );
        let many = NativeRows {
            packed: 1024,
            ..ROWS
        };
        assert_eq!(
            op.working_bytes(MLX, many),
            19 * 1024 * 256 + 4,
            "live unfused radix-4 stage plus original input"
        );
        assert_eq!(op.working_bytes(CUDA, many), 19 * 1024 * 256 + 4);
    }

    #[test]
    fn task_stage_maximum_preserves_source_pruned_zero_eviction_and_dense_cache() {
        let facts = NativeAllocationFacts {
            conditioning_stream: None,
            conditioning: None,
            conditioning_resident_bytes: 0,
            conditioning_load_bytes: 0,
            audio_decode_load_bytes: 0,
            video_decode_resident_bytes: 0,
            video_decode_load_bytes: 0,
            audio_decode_resident_bytes: 0,
            default_evaluations: 50,
            backend: MLX,
            tasks: vec![
                NativeTaskAllocation {
                    adapter: None,
                    patch_size: [1, 2, 2],
                    reference_task: false,
                    window: None,
                    resident_bytes: 1000,
                    constructor_bytes: 0,
                    evicted_bytes: 0,
                    table_bytes_per_timestep: 0,
                    projections: vec![],
                },
                NativeTaskAllocation {
                    adapter: None,
                    patch_size: [1, 2, 2],
                    reference_task: false,
                    window: None,
                    resident_bytes: 1200,
                    constructor_bytes: 0,
                    evicted_bytes: 800,
                    table_bytes_per_timestep: 16,
                    projections: vec![],
                },
            ],
        };
        assert_eq!(facts.precompute_peak_bytes(ROWS), 1280);
        assert_eq!(
            facts.denoise_peak_bytes(ROWS),
            1000,
            "pruned operands/tables cannot receive dense eviction credit"
        );
        assert_eq!(
            facts.peak_bytes(ROWS),
            1280,
            "one task, not both checkpoints resident"
        );
    }

    #[test]
    fn full_request_bounds_include_reference_audio_and_do_not_enable_veda() {
        let mut req = crate::GenerationRequest {
            width: 1344,
            height: 768,
            frames: Some(124),
            prompt: "a scene".into(),
            ..Default::default()
        };
        let base = request_rows((&req).into(), 4).unwrap();
        req.conditioning.push(crate::Conditioning::ReferenceAudio {
            audio: crate::AudioTrack {
                sample_rate: 32000,
                channels: 1,
                samples: vec![0.; 32000],
                stems: vec![],
            },
            strength: None,
        });
        let referenced = request_rows((&req).into(), 4).unwrap();
        assert!(referenced.packed > base.packed);
        assert!(referenced.conditioning >= base.conditioning);
        assert!(req.veda.is_none());
        assert_eq!(
            referenced.schedule, 10,
            "existing S5 two-stream bound includes fixed timesteps"
        );
        assert!(request_rows((&req).into(), 0).is_err());
    }

    #[test]
    fn selected_adapter_partition_prices_install_and_forward_without_overlay_double_charge() {
        use crate::h3_adapter_memory::{
            H3AdapterMemoryProfile, H3AdapterProjectionMemory, H3AdapterTarget,
        };
        let profile = |resident, install, per_row| H3AdapterMemoryProfile {
            resident_bytes: resident,
            install_peak_bytes: install,
            projections: vec![H3AdapterProjectionMemory {
                target: H3AdapterTarget {
                    path: "transformer_blocks.0.attn.to_q".into(),
                    input: 8,
                    output: 8,
                    packed: false,
                    activation_width: 2,
                    activation_dtype: Dtype::BF16,
                    rows: RowDomain::Packed,
                },
                installed_bytes: resident,
                resident_bytes: resident,
                forward_fixed_bytes: 0,
                forward_bytes_per_row: per_row,
            }],
        };
        let base = NativeTaskAllocation {
            adapter: Some(profile(40, 200, 8)),
            resident_bytes: 1000,
            constructor_bytes: 0,
            patch_size: [1, 1, 1],
            reference_task: false,
            window: None,
            evicted_bytes: 0,
            table_bytes_per_timestep: 0,
            projections: vec![],
        };
        let mut reference = base.clone();
        reference.reference_task = true;
        reference.patch_size = [1, 2, 2];
        reference.adapter = Some(profile(10_000, 90_000, 100));
        let mut facts = NativeAllocationFacts {
            backend: CUDA,
            tasks: vec![base, reference],
            default_evaluations: 4,
            conditioning: None,
            conditioning_stream: None,
            conditioning_resident_bytes: 0,
            conditioning_load_bytes: 0,
            video_decode_resident_bytes: 0,
            video_decode_load_bytes: 0,
            audio_decode_resident_bytes: 0,
            audio_decode_load_bytes: 0,
        };
        let rows = NativeRows {
            task: NativeTaskSelection::Base,
            packed: 3,
            ..ROWS
        };
        assert_eq!(facts.adapter_resident_bytes(rows), 40);
        assert_eq!(facts.precompute_peak_bytes(rows), 1160);
        assert_eq!(facts.denoise_peak_bytes(rows), 1024);
        assert_eq!(
            facts.precompute_peak_bytes_windowed(rows, None).unwrap(),
            1160
        );
        assert_eq!(facts.denoise_peak_bytes_windowed(rows, None).unwrap(), 1024);
        assert_eq!(
            facts.resident_peak_bytes_windowed(rows, None).unwrap(),
            1000,
            "overlay is separate; consumers add selected residency exactly once"
        );
        facts.tasks[1].adapter.as_mut().unwrap().install_peak_bytes = 1_000_000;
        facts.tasks[1].adapter.as_mut().unwrap().resident_bytes = 900_000;
        assert_eq!(facts.adapter_resident_bytes(rows), 40);
        assert_eq!(
            facts.peak_bytes(rows),
            1160,
            "unselected representation cannot affect admission"
        );
        let req = crate::GenerationRequest {
            width: 256,
            height: 256,
            frames: Some(124),
            ..Default::default()
        };
        assert_eq!(facts.selected_patch((&req).into()).unwrap(), [1, 1, 1]);
    }

    #[test]
    fn source_patch_geometry_controls_rows_and_only_selected_task_validation() {
        let (headers, payloads) = fixture(true, &[256]);
        let allocation = allocation_from_headers(&headers, &payloads, MLX, 2, |_| None).unwrap();
        let base = NativeTaskAllocation {
            adapter: None,
            resident_bytes: allocation.resident_bytes,
            constructor_bytes: 0,
            patch_size: [1, 2, 2],
            reference_task: false,
            window: None,
            evicted_bytes: 0,
            table_bytes_per_timestep: 0,
            projections: allocation.projections,
        };
        let mut reference = base.clone();
        reference.reference_task = true;
        reference.patch_size = [2, 2, 2];
        let mut facts = NativeAllocationFacts {
            conditioning: None,
            conditioning_stream: None,
            conditioning_resident_bytes: 0,
            conditioning_load_bytes: 0,
            audio_decode_load_bytes: 0,
            video_decode_resident_bytes: 0,
            video_decode_load_bytes: 0,
            audio_decode_resident_bytes: 0,
            default_evaluations: 50,
            backend: MLX,
            tasks: vec![base, reference],
        };
        let mut req = crate::GenerationRequest {
            width: 1344,
            height: 768,
            frames: Some(124),
            ..Default::default()
        };
        let rows = facts.request_rows((&req).into(), 4).unwrap();
        let stock = facts.denoise_peak_bytes(rows);
        facts.tasks[0].patch_size = [1, 1, 1];
        assert!(
            facts.denoise_peak_bytes(facts.request_rows((&req).into(), 4).unwrap()) > stock,
            "smaller source patches increase visual rows without changing text/audio rows"
        );
        req.conditioning.push(crate::Conditioning::Reference {
            image: crate::Image {
                width: 320,
                height: 576,
                pixels: vec![],
            },
            strength: None,
        });
        assert!(
            facts.request_rows((&req).into(), 4).is_err(),
            "incompatible selected reference-task temporal patch"
        );
        req.conditioning.clear();
        for (width, height, frames, count) in [
            (1345, 768, 124, 1),
            (1344, 1344, 124, 1),
            (1344, 768, 123, 1),
            (1344, 768, 124, 2),
        ] {
            req.width = width;
            req.height = height;
            req.frames = Some(frames);
            req.count = count;
            assert!(
                facts.request_rows((&req).into(), 4).is_err(),
                "native geometry refuses instead of rounding smaller"
            );
        }
    }

    #[test]
    fn candle_ffn_uses_actual_token_spans_and_retains_outer_outputs() {
        let (headers, payloads) = fixture(true, &[256]);
        let allocation = allocation_from_headers(&headers, &payloads, MLX, 2, |_| None).unwrap();
        let mut op = allocation.projections[0].clone();
        let large = NativeRows {
            packed: 4_000_000,
            ..ROWS
        };
        let unchunked = op.working_bytes(CUDA, large);
        op.base = "transformer_blocks.0.ff.net.2".into();
        let chunked = op.working_bytes(CUDA, large);
        assert!(
            chunked < unchunked,
            "native CUDA follows the existing outer FFN row planner"
        );
        assert!(
            chunked >= 3 * large.packed * op.output * op.activation_width,
            "full hidden input plus retained spans and final cat really coexist"
        );
        let mut bare = op.clone();
        bare.base = "projection".into();
        assert!(
            op.working_bytes(MLX, ROWS) > bare.working_bytes(MLX, ROWS),
            "FF output keeps the first projection and SiLU while native second projection runs"
        );
    }

    #[test]
    fn malformed_native_headers_refuse_instead_of_becoming_dense() {
        let (mut headers, payloads) = fixture(false, &[64]);
        headers[0].dtype = Dtype::F32;
        assert!(allocation_from_headers(&headers, &payloads, MLX, 4, |_| Some(4)).is_err());
        let (mut headers, payloads) = fixture(true, &[256]);
        headers.retain(|h| !h.name.ends_with("weight_s_rel"));
        assert!(allocation_from_headers(&headers, &payloads, CUDA, 2, |_| None).is_err());
    }
}
