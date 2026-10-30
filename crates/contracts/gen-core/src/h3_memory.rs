//! MiniMax-H3 native low-precision allocation facts carried by
//! [`MemoryPhaseFacts`](crate::MemoryPhaseFacts), and the frame lattice a request resolves onto.
//! Request/component allocation pricing lives in `gen_core_h3::h3_memory` (sc-27316).

use crate::weightsmeta::Dtype;
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
/// The frame rate every MiniMax-H3 duration is expressed at.
pub const FPS: f64 = 24.0;
/// The shortest legal render (the `17n + 5` lattice floor, 5.1667 s).
pub const MIN_FRAMES: u32 = 124;
/// The longest legal render (the lattice ceiling, 14.375 s).
pub const MAX_FRAMES: u32 = 345;
/// The duration a request naming neither `frames` nor `duration` renders at — the lattice floor.
pub const DEFAULT_DURATION_SECONDS: f32 = 5.1667;
/// Align a requested `duration` **upward** onto the `17n + 5` lattice, refusing one outside the
/// model's `MIN_FRAMES..=MAX_FRAMES` seconds range. The single definition the Candle engine
/// renders with and allocation admission prices with (sc-26691).
pub fn align_frames_for_duration(seconds: f32) -> Result<u32> {
    if !seconds.is_finite() {
        return Err(Error::Msg(format!(
            "minimax_h3: duration must be finite, got {seconds}"
        )));
    }
    let (min_s, max_s) = (f64::from(MIN_FRAMES) / FPS, f64::from(MAX_FRAMES) / FPS);
    let seconds = f64::from(seconds);
    if !(min_s - 1e-6..=max_s + 1e-6).contains(&seconds) {
        return Err(Error::Msg(format!(
            "minimax_h3: duration {seconds} s is outside the model's {min_s}-{max_s} s range (the \
             legal frame counts {MIN_FRAMES}-{MAX_FRAMES} at {FPS} fps; the reference pipeline \
             advertises 5-15 s but its own 17n+5 lattice tops out at 14.375 s)"
        )));
    }
    let requested = (seconds * FPS).round().max(1.0) as u32;
    // Round up to the next `17n + 5`.
    Ok(requested + (17 + 5 - requested % 17) % 17)
}
/// The frame count a request renders: `frames` wins over `duration`, which aligns upward; neither
/// renders [`DEFAULT_DURATION_SECONDS`]. Shared by the Candle engine's geometry resolution and
/// [`crate::veda::workspace`] pricing, so admission prices the clip the engine will render.
pub fn resolve_request_frames(frames: Option<u32>, duration: Option<f32>) -> Result<u32> {
    match (frames, duration) {
        (Some(frames), _) => Ok(frames),
        (None, Some(seconds)) => align_frames_for_duration(seconds),
        (None, None) => align_frames_for_duration(DEFAULT_DURATION_SECONDS),
    }
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
pub mod window;

/// The actual Candle H3 outer FFN token-span planner's element budget.
pub const FFN_INTERMEDIATE_BUDGET: u64 = 1_000_000_000;
