//! The Veda workspace bound a [`Generator`](crate::Generator) reports, and the borrowed request
//! shape it is priced from. The pricing itself lives in `gen_core_h3::veda::workspace` (sc-27316).

use super::{contract::TileShape, VedaRequest};
use crate::{Conditioning, GenerationRequest};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Mlx,
    Cuda,
}
/// The CUDA H3 materializing fallback's default score domain. Sixteen bytes per score
/// conservatively cover scores, scaling/softmax transients and probabilities (512 MiB).
/// Runtime and request pricing use this same limit; fused attention remains eligible.
pub const CUDA_DENSE_SCORE_ELEMENTS: u64 = 32 * 1024 * 1024;

/// Allocations simultaneously live in one attention phase, in separate physical pools.
/// `shared_device` is the input QKV storage included in `device`, not an extra allocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkspacePoolBytes {
    pub host: u64,
    pub device: u64,
    pub shared_device: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkspaceLifetimes {
    pub predictor_load: WorkspacePoolBytes,
    pub self_test: WorkspacePoolBytes,
    pub sparse: WorkspacePoolBytes,
    pub dense_fallback: WorkspacePoolBytes,
}
impl WorkspaceLifetimes {
    pub fn phases(self) -> [WorkspacePoolBytes; 4] {
        [
            self.predictor_load,
            self.self_test,
            self.sparse,
            self.dense_fallback,
        ]
    }
}
#[derive(Clone, Debug, Default)]
pub struct Chunk {
    pub heads: usize,
    pub query_tiles: usize,
    pub global_tiles: usize,
    pub selected_keys: usize,
    pub copies_bytes: u64,
    pub attention_bytes: u64,
    pub minimum_copies_bytes: u64,
    pub minimum_attention_bytes: u64,
    pub refusal: Option<String>,
}
#[derive(Clone, Debug)]
pub struct WorkspaceBound {
    pub backend: Backend,
    pub plan: String,
    pub execution_grid: [usize; 3],
    pub plan_grid: [usize; 3],
    pub predictor_host_bytes: u64,
    pub predictor_device_bytes: u64,
    pub predictor_load_bytes: u64,
    pub layout_host_bytes: u64,
    pub position_rows_bytes: u64,
    pub tile_order_bytes: u64,
    pub scorer_bytes: u64,
    pub selector_bytes: u64,
    pub gather_bytes: u64,
    pub kernel_scratch_bytes: u64,
    pub retained_output_bytes: u64,
    pub input_qkv_bytes: u64,
    pub self_test_bytes: u64,
    pub dense_fallback_bytes: u64,
    pub chunks: BTreeMap<TileShape, Chunk>,
    pub total_bytes: u64,
    /// CUDA uses phase maxima independently per pool. MLX's unified-memory bound retains
    /// the conservative lazy-graph/cache lifetimes represented by `total_bytes`.
    pub lifetimes: WorkspaceLifetimes,
}
impl WorkspaceBound {
    pub fn host_peak_bytes(&self) -> u64 {
        self.lifetimes
            .phases()
            .iter()
            .map(|p| p.host)
            .max()
            .unwrap_or(0)
    }
    pub fn device_peak_bytes(&self) -> u64 {
        self.lifetimes
            .phases()
            .iter()
            .map(|p| p.device)
            .max()
            .unwrap_or(0)
    }
    pub fn unified_peak_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Credit only independently proven shared caller QKV, once in each overlapping phase.
    /// No credit is granted for weights, outputs, or a historical dense measured peak.
    pub fn incremental_device_peak_bytes(&self, shared_qkv_credit: u64) -> crate::Result<u64> {
        let mut peak = 0;
        for phase in self.lifetimes.phases() {
            if phase.shared_device > phase.device {
                return Err(crate::Error::Unsupported(
                    "invalid Veda shared-device allocation facts".into(),
                ));
            }
            peak = peak.max(phase.device - phase.shared_device.min(shared_qkv_credit));
        }
        Ok(peak)
    }

    /// Check pools separately; equality fits, including the caller's explicit reserve.
    pub fn fits_pools(&self, host_available: u64, device_available: u64, reserve: u64) -> bool {
        let device = if self.backend == Backend::Mlx {
            self.unified_peak_bytes()
        } else {
            self.device_peak_bytes()
        };
        self.host_peak_bytes() <= host_available
            && device
                .checked_add(reserve)
                .is_some_and(|need| need <= device_available)
    }
}
/// Borrowed product/native admission shape; avoids cloning decoded media just to price it.
#[derive(Clone, Copy)]
pub struct Request<'a> {
    pub veda: Option<&'a VedaRequest>,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub count: u32,
    pub prompt: &'a str,
    pub conditioning: &'a [Conditioning],
    pub reference_image_short_edge: Option<u32>,
    /// sc-26691: `None` ⇒ [`crate::ReferenceImageResizeMode::Match`].
    pub reference_image_resize_mode: Option<crate::ReferenceImageResizeMode>,
}
impl<'a> From<&'a GenerationRequest> for Request<'a> {
    fn from(r: &'a GenerationRequest) -> Self {
        Self {
            veda: r.veda.as_ref(),
            width: r.width,
            height: r.height,
            // The engine's own resolution (`frames` wins, `duration` aligns upward, neither is the
            // lattice floor). A duration the engine refuses prices at the lattice ceiling — the
            // conservative envelope; the render itself is refused before allocating.
            frames: crate::h3_memory::resolve_request_frames(r.frames, r.duration)
                .unwrap_or(crate::h3_memory::MAX_FRAMES),
            count: r.count,
            prompt: &r.prompt,
            conditioning: &r.conditioning,
            reference_image_short_edge: r.reference_image_short_edge,
            reference_image_resize_mode: r.reference_image_resize_mode,
        }
    }
}
