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
