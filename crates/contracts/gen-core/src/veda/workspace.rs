//! Explicit request bounds for Veda. No allocator/free-memory query, predictor load or device work.
//! Bounds sum overlapping owned lifetimes conservatively, including lazy graph intermediates.
use super::{
    contract::{Layout, Plan, TileShape, FROZEN_PLANS, TILE},
    VedaBudget, VedaRequest,
};
use crate::{Conditioning, Error, GenerationRequest, Result};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Mlx,
    Cuda,
}
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub copies: u64,
    pub attention: u64,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            copies: 512 << 20,
            attention: 512 << 20,
        }
    }
}
impl Limits {
    /// Allowances a backend's runtime batches heads under. CUDA scores, selects and launches a
    /// whole head batch at once (one score readback and one kernel launch per batch), so its
    /// gather/scorer copy allowance admits several heads: 4 GiB, below the dense SDPA score
    /// transient (`dense_fallback_bytes`) the sparse call replaces. MLX keeps the default.
    pub fn for_backend(backend: Backend) -> Self {
        match backend {
            Backend::Mlx => Self::default(),
            Backend::Cuda => Self {
                copies: 4 << 30,
                ..Self::default()
            },
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct LayoutSize {
    pub seq: usize,
    pub slots: usize,
    pub video: usize,
    pub reference: usize,
    pub ref_tokens: usize,
    pub target_tokens: usize,
}
impl From<&Layout> for LayoutSize {
    fn from(l: &Layout) -> Self {
        Self {
            seq: l.seq,
            slots: l.perm.len(),
            video: l.video,
            reference: l.reference,
            ref_tokens: l.ref_tokens,
            target_tokens: l.target_tokens,
        }
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
}
fn bad(s: &str) -> Error {
    Error::Msg(format!("Veda workspace: {s}"))
}
fn bytes(n: u128) -> Result<u64> {
    u64::try_from(n).map_err(|_| bad("size overflow"))
}
fn keys(b: VedaBudget, tokens: usize, cols: usize) -> usize {
    if cols == 0 {
        return 0;
    }
    let n = tokens.div_ceil(TILE) as f64;
    let cost = match b {
        VedaBudget::SparsityPercent(p) => (1. - p / 100.) * n * n / cols as f64,
        VedaBudget::KeyTiles(n) => n as f64,
    };
    // Fractional selector can select floor+1 even near an integer boundary. Dense 0% is exact.
    if matches!(b, VedaBudget::SparsityPercent(0.)) {
        cols
    } else {
        ((cost.floor() as usize).saturating_add(1)).clamp(1, cols)
    }
}
/// All temporary copies/pooling/scoring per head, before its first allocation.
fn copy_cost(l: LayoutSize, d: usize, element: usize) -> Result<u64> {
    let (s, v, d, e) = (l.slots as u128, l.video as u128, d as u128, element as u128);
    // Original-head gather, padded gather, masked tile, casts, +/- infinity where operands,
    // pooled mean/max/min/concat, projections/residuals, logits + downloaded host scores.
    bytes(
        16 * s
            + 8 * l.seq as u128
            + 64
            + 12 * s * d * e
            + 16 * v * TILE as u128 * d
            + 80 * v * d
            + 12 * v * v
            + 48 * d * d,
    )
}
fn sparse_row_cost(l: LayoutSize, h: usize, k: usize, d: usize, element: usize) -> Result<u64> {
    let (h, k, d, e, t) = (
        h as u128,
        k as u128,
        d as u128,
        element as u128,
        TILE as u128,
    );
    let _ = l;
    // K/V gather, query/returned output, host/device indices and validity, additive masks,
    // conservative SDPA scores/probabilities/workspace even when fused kernel avoids them.
    bytes(h * (2 * k * t * d * e + 2 * t * d * e + 12 * k + 2 * k * t + 12 * t * k * t))
}
fn global_row_cost(l: LayoutSize, h: usize, d: usize, element: usize) -> Result<u64> {
    bytes(
        h as u128
            * (12 * TILE as u128 * l.slots as u128
                + 2 * TILE as u128 * d as u128 * element as u128)
            + 2 * l.slots as u128,
    )
}
/// The smallest batch is always priced even when it exceeds an allowance. Refusal is an outcome,
/// not an excuse to charge zero bytes; callers check it BEFORE gather/pooling allocation.
pub fn chunk(
    l: LayoutSize,
    group: usize,
    dim: usize,
    element: usize,
    backend: Backend,
    request: &VedaRequest,
    limits: Limits,
) -> Result<Chunk> {
    if group == 0
        || dim == 0
        || l.video == 0
        || l.reference >= l.video
        || !l.slots.is_multiple_of(TILE)
        || l.video > l.slots / TILE
        || !matches!(element, 2 | 4)
    {
        return Err(bad("invalid layout/dtype"));
    }
    let mut copy = copy_cost(l, dim, element)?;
    if backend == Backend::Cuda {
        copy = copy
            .checked_add(bytes(
                20 * l.slots as u128 * dim as u128
                    + 32 * (l.slots / TILE) as u128 * (l.slots / TILE) as u128
                    + 64 * (l.slots / TILE) as u128
                    + 64,
            )?)
            .ok_or_else(|| bad("size overflow"))?;
    }
    // Both backends batch up to the whole same-shape head group within the copy allowance.
    let mut h = (limits.copies / copy.max(1)).min(group as u64).max(1) as usize;
    let k = keys(request.reference, l.ref_tokens, l.reference)
        + keys(request.generated, l.target_tokens, l.video - l.reference)
        + l.slots / TILE
        - l.video;
    if backend == Backend::Mlx {
        let minimum = sparse_row_cost(l, 1, k, dim, element)?.max(if l.slots / TILE > l.video {
            global_row_cost(l, 1, dim, element)?
        } else {
            0
        });
        h = h.min((limits.attention / minimum.max(1)).max(1) as usize);
    }
    let row = sparse_row_cost(l, h, k, dim, element)?;
    let global = global_row_cost(l, h, dim, element)?;
    let qr = (limits.attention / row.max(1)).min(l.video as u64).max(1) as usize;
    let ng = l.slots / TILE - l.video;
    let gr = if ng == 0 {
        0
    } else {
        (limits.attention / global.max(1)).min(ng as u64).max(1) as usize
    };
    let min_attention = if backend == Backend::Mlx {
        row.max(if ng > 0 { global } else { 0 })
    } else {
        0
    };
    let refusal = if copy > limits.copies {
        Some("minimum tile/scorer head exceeds copy allowance".into())
    } else if min_attention > limits.attention {
        Some("minimum query/global batch exceeds attention allowance".into())
    } else {
        None
    };
    Ok(Chunk {
        heads: h,
        query_tiles: qr,
        global_tiles: gr,
        selected_keys: k,
        copies_bytes: copy.saturating_mul(h as u64),
        attention_bytes: if backend == Backend::Mlx {
            row.saturating_mul(qr as u64)
                .max(global.saturating_mul(gr as u64))
        } else {
            0
        },
        minimum_copies_bytes: copy,
        minimum_attention_bytes: min_attention,
        refusal,
    })
}
/// Sum full-layer lifetimes rather than treating one gather allowance as the layer bound.
#[allow(clippy::too_many_arguments)]
pub fn bound(
    request: &VedaRequest,
    backend: Backend,
    plan: &Plan,
    grid: [usize; 3],
    layouts: &BTreeMap<TileShape, LayoutSize>,
    layers: usize,
    heads: usize,
    dim: usize,
    batch: usize,
    element: usize,
    limits: Limits,
) -> Result<WorkspaceBound> {
    request.validate(layers, usize::MAX).map_err(Error::Msg)?;
    if layouts.is_empty()
        || batch == 0
        || heads == 0
        || plan.heads.len() != layers
        || plan
            .heads
            .iter()
            .any(|row| row.len() != heads || row.iter().any(|&s| s >= plan.shapes.len()))
    {
        return Err(bad("empty problem"));
    }
    let seq = layouts.values().next().unwrap().seq;
    let mut chunks = BTreeMap::new();
    let (mut gather, mut selector, mut scratch, mut device, mut layout_bytes) =
        (0u64, 0u64, 0u64, 0u64, 0u64);
    for (&shape, &l) in layouts {
        if l.seq != seq {
            return Err(bad("layout sequence mismatch"));
        }
        let group = plan
            .heads
            .iter()
            .map(|row| row.iter().filter(|&&s| plan.shapes[s] == shape).count())
            .max()
            .unwrap_or(heads)
            .max(1);
        let c = chunk(l, group, dim, element, backend, request, limits)?;
        let (s, n, h, d) = (
            l.slots as u128,
            (l.slots / TILE) as u128,
            c.heads as u128,
            dim as u128,
        );
        layout_bytes = layout_bytes
            .checked_add(bytes(8 * s + 16 * n + 4 * seq as u128 + 256)?)
            .ok_or_else(|| bad("size overflow"))?;
        gather = gather.max(c.attention_bytes);
        // selected rows Vec + indices, per-column topk candidates, scores owned host/device.
        selector = selector.max(bytes(
            16 * h * n * n + 48 * h * n + 24 * n + 8 * h * (l.video as u128).pow(2),
        )?);
        device = device.max(bytes(24 * h * d * d)?);
        // CUDA conversion inputs/output/quant/scales and expanded CSR 64-key masks on host/device,
        // for every head of one batched launch.
        if backend == Backend::Cuda {
            scratch = scratch.max(bytes(h * (20 * s * d + 32 * n * n + 32 * n) + 64)?);
        }
        chunks.insert(shape, c);
    }
    let construction = layouts
        .values()
        .map(|l| 8 * l.slots as u128 + 16 * (l.slots / TILE) as u128 + 16 * l.seq as u128 + 1024)
        .max()
        .unwrap_or(0);
    layout_bytes = layout_bytes
        .checked_add(bytes(construction)?)
        .ok_or_else(|| bad("size overflow"))?;
    let predictor_host = bytes(12 * layers as u128 * heads as u128 * (dim as u128).pow(2))?;
    let plans = 32 * FROZEN_PLANS.len() as u64;
    let load = 275_415_649 + plans + bytes(24 * heads as u128 * (dim as u128).pow(2))?;
    let position = bytes(6 * 24 * seq as u128)?; // packed layout/model clones, including Vec growth capacity
    let outputs =
        bytes(12 * batch as u128 * heads as u128 * seq as u128 * dim as u128 * element as u128)?;
    let inputs =
        bytes(3 * batch as u128 * heads as u128 * seq as u128 * dim as u128 * element as u128)?;
    let self_test = match backend {
        Backend::Cuda => bytes(64 * 512 * dim as u128 + 12 * 512 * 512 + 64 * 512)?,
        Backend::Mlx => bytes(128 * 1024 * dim as u128 + 32 * 2 * 1024 * 1024 + 256 * 1024)?,
    };
    // A failed route keeps predictor/layouts alive while executing dense. Price explicit eager
    // score/probabilities, masks, output/casts. MLX fused execution may use less; never assumes it.
    let fallback = match backend {
        Backend::Mlx => outputs
            .checked_add(inputs)
            .ok_or_else(|| bad("size overflow"))?,
        // Production Candle dense SDPA has a 1e9 score-element query budget. Minimum one
        // query exceeds it when B*H*S > budget; that minimum is explicitly priced too.
        Backend::Cuda => {
            let row = batch as u128 * heads as u128 * seq as u128;
            let rows = (1_000_000_000u128 / row.max(1)).max(1).min(seq as u128);
            bytes(16 * row * rows + outputs as u128 + inputs as u128)?
        }
    };
    let tile_bytes = layouts
        .iter()
        .map(|(shape, l)| {
            12 * l.slots as u128 * dim as u128 * element as u128 * chunks[shape].heads as u128
        })
        .max()
        .unwrap_or(0);
    let tile_bytes = bytes(tile_bytes)?;
    let scorer_bytes = layouts
        .iter()
        .map(|(shape, l)| {
            chunks[shape].copies_bytes.saturating_sub(
                (12 * l.slots as u128 * dim as u128 * element as u128 * chunks[shape].heads as u128)
                    as u64,
            )
        })
        .max()
        .unwrap_or(0);
    let total = bytes(
        predictor_host as u128
            + device as u128
            + load as u128
            + layout_bytes as u128
            + plans as u128
            + position as u128
            + tile_bytes as u128
            + scorer_bytes as u128
            + selector as u128
            + gather as u128
            + scratch as u128
            + outputs as u128
            + inputs as u128
            + self_test as u128
            + fallback as u128,
    )?;
    Ok(WorkspaceBound {
        backend,
        plan: plan.name.clone(),
        execution_grid: grid,
        plan_grid: plan.grid,
        predictor_host_bytes: predictor_host,
        predictor_device_bytes: device,
        predictor_load_bytes: load,
        layout_host_bytes: layout_bytes + plans,
        position_rows_bytes: position,
        tile_order_bytes: tile_bytes,
        scorer_bytes,
        selector_bytes: selector,
        gather_bytes: gather,
        kernel_scratch_bytes: scratch,
        retained_output_bytes: outputs,
        input_qkv_bytes: inputs,
        self_test_bytes: self_test,
        dense_fallback_bytes: fallback,
        chunks,
        total_bytes: total,
    })
}
/// Choose from the exact immutable checkpoint plan table without reading weights.
pub fn frozen_plan(grid: [usize; 3]) -> Result<Plan> {
    if grid.contains(&0) {
        return Err(bad("empty patch grid"));
    }
    let plans: serde_json::Value =
        serde_json::from_str(FROZEN_PLANS).map_err(|_| bad("frozen plans"))?;
    let bundle = super::contract::Bundle {
        identity: String::new(),
        layers: 50,
        heads: 56,
        dim: 128,
        plans: plans
            .as_object()
            .unwrap()
            .values()
            .map(|p| Plan::parse(p, 50, 56))
            .collect::<Result<_>>()?,
        q: vec![],
        k: vec![],
    };
    Ok(bundle.choose(grid))
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
}
impl<'a> From<&'a GenerationRequest> for Request<'a> {
    fn from(r: &'a GenerationRequest) -> Self {
        Self {
            veda: r.veda.as_ref(),
            width: r.width,
            height: r.height,
            frames: r.frames.unwrap_or(124),
            count: r.count,
            prompt: &r.prompt,
            conditioning: &r.conditioning,
            reference_image_short_edge: r.reference_image_short_edge,
        }
    }
}
/// Output pixels per DiT patch edge: 16x VAE spatial compression times the 2x2 spatial patch.
const PATCH_PIXELS: usize = 32;
pub fn for_request(req: &GenerationRequest, backend: Backend) -> Result<Option<WorkspaceBound>> {
    for_video_request(Request::from(req), backend)
}
/// Admission before prompt encoding/packing. Byte-tokenized prompt is bounded by UTF-8 bytes;
/// grounded image tokens use the native vision processor's 16,777,216 pixel cap / 32² merger.
/// Reference image grids follow short-edge stride rounding; clips use the full generated duration
/// at their own aspect. These upper envelopes intentionally exceed actual encoded layouts.
pub fn for_video_request(req: Request<'_>, backend: Backend) -> Result<Option<WorkspaceBound>> {
    let Some(r) = req.veda.filter(|r| r.enabled) else {
        return Ok(None);
    };
    let (plan, grid, layouts) = price_layouts(req)?;
    bound(
        r,
        backend,
        &plan,
        grid,
        &layouts,
        50,
        56,
        128,
        req.count.max(1) as usize,
        2,
        Limits::for_backend(backend),
    )
    .map(Some)
}
/// Upper-envelope native layout sizes for an enabled request, in DiT patch units.
fn price_layouts(req: Request<'_>) -> Result<(Plan, [usize; 3], BTreeMap<TileShape, LayoutSize>)> {
    let frames = req.frames as usize;
    if frames < 5 || req.width == 0 || req.height == 0 {
        return Err(bad("admission requires resolved geometry"));
    }
    // Veda tiles the DiT *patch* grid, as runtime and upstream `h3_layout.py` do: the VAE's
    // 16x spatial latent divided by the DiT's 2x2 spatial patch, i.e. pixels / 32.
    let grid = [
        (frames - 5).div_ceil(17) * 5 + 2,
        req.height as usize / PATCH_PIXELS,
        req.width as usize / PATCH_PIXELS,
    ];
    let mut spans = vec![grid];
    let mut globals = req
        .prompt
        .len()
        .saturating_add(512)
        .saturating_add((frames * 40).div_ceil(24));
    let image_grid = |w: u32, h: u32| -> Result<[usize; 3]> {
        if w == 0 || h == 0 {
            return Err(bad("empty reference image"));
        }
        let edge = req.reference_image_short_edge.unwrap_or(2048) as f64;
        let scale = edge / w.min(h) as f64;
        // Stride-32 lattice in pixels == one DiT patch per 32 pixels.
        Ok([
            1,
            (h as f64 * scale / 32.).round_ties_even().max(1.) as usize,
            (w as f64 * scale / 32.).round_ties_even().max(1.) as usize,
        ])
    };
    // Same factor/min/max geometry as native Qwen3-VL preprocessing. Extreme aspects
    // can exceed its nominal max area because each dimension retains at least one factor;
    // price those actual snapped dimensions rather than treating MAX_PIXELS as an absolute cap.
    let vision_tokens = |height: usize, width: usize| -> Result<usize> {
        let (h, w) = (height as f64, width as f64);
        if h <= 0. || w <= 0. {
            return Err(bad("empty vision image"));
        }
        let (mut rh, mut rw) = (
            (h / 32.).round_ties_even() * 32.,
            (w / 32.).round_ties_even() * 32.,
        );
        if rh * rw > 16_777_216. {
            let scale = (h * w / 16_777_216.).sqrt();
            rh = (h / scale / 32.).floor().max(1.) * 32.;
            rw = (w / scale / 32.).floor().max(1.) * 32.;
        } else if rh * rw < 65_536. {
            let scale = (65_536. / (h * w)).sqrt();
            rh = (h * scale / 32.).ceil() * 32.;
            rw = (w * scale / 32.).ceil() * 32.;
        }
        let tokens = rh / 32. * (rw / 32.);
        if !tokens.is_finite() || tokens >= usize::MAX as f64 {
            return Err(bad("vision size overflow"));
        }
        Ok(tokens as usize)
    };
    let audio_rows = |a: &crate::AudioTrack| -> Result<usize> {
        if a.sample_rate == 0 || a.channels == 0 {
            return Err(bad("invalid reference audio"));
        }
        Ok(
            ((a.samples.len() as u128 * 40).div_ceil(a.sample_rate as u128 * a.channels as u128))
                .min(usize::MAX as u128) as usize,
        )
    };
    for c in req.conditioning {
        match c {
            Conditioning::Reference { image, .. } => {
                let g = image_grid(image.width, image.height)?;
                spans.insert(spans.len() - 1, g);
                globals = globals.saturating_add(
                    vision_tokens(
                        g[1].checked_mul(PATCH_PIXELS)
                            .ok_or_else(|| bad("image size overflow"))?,
                        g[2].checked_mul(PATCH_PIXELS)
                            .ok_or_else(|| bad("image size overflow"))?,
                    )? + 64,
                );
            }
            Conditioning::MultiReference { images } => {
                for image in images {
                    let g = image_grid(image.width, image.height)?;
                    spans.insert(spans.len() - 1, g);
                    globals = globals.saturating_add(
                        vision_tokens(
                            g[1].checked_mul(PATCH_PIXELS)
                                .ok_or_else(|| bad("image size overflow"))?,
                            g[2].checked_mul(PATCH_PIXELS)
                                .ok_or_else(|| bad("image size overflow"))?,
                        )? + 64,
                    );
                }
            }
            Conditioning::Keyframe { .. } => {
                spans.insert(spans.len() - 1, [1, grid[1], grid[2]]);
                globals = globals
                    .saturating_add(vision_tokens(req.height as usize, req.width as usize)? + 64);
            }
            Conditioning::ReferenceVideo {
                frames: refs,
                audio,
                ..
            } => {
                // Native clip fit never exceeds the 1,032,192-pixel canvas. Square envelope
                // prices padding for every shape, without relying on reference carrier counts.
                let first = refs.first().ok_or_else(|| bad("empty reference video"))?;
                let w = first.width.max(1) as f64;
                let h = first.height.max(1) as f64;
                let scale = (1_032_192. / (w * h)).sqrt();
                let g = [
                    grid[0],
                    (h * scale / PATCH_PIXELS as f64).ceil() as usize + 1,
                    (w * scale / PATCH_PIXELS as f64).ceil() as usize + 1,
                ];
                spans.insert(spans.len() - 1, g);
                // 2-fps samples of a 24-fps normalized clip, temporally merged in pairs.
                let blocks = frames.div_ceil(12).div_ceil(2);
                globals = globals.saturating_add(
                    blocks.saturating_mul(
                        vision_tokens(
                            g[1].checked_mul(PATCH_PIXELS)
                                .ok_or_else(|| bad("image size overflow"))?,
                            g[2].checked_mul(PATCH_PIXELS)
                                .ok_or_else(|| bad("image size overflow"))?,
                        )? + 64,
                    ),
                );
                if let Some(a) = audio {
                    globals = globals.saturating_add(audio_rows(a)?);
                }
            }
            Conditioning::ReferenceAudio { audio, .. } => {
                globals = globals.saturating_add(audio_rows(audio)?)
            }
            _ => return Err(bad("unsupported H3 conditioning carrier")),
        }
    }
    let volume = |g: &[usize; 3]| {
        g.iter()
            .try_fold(1usize, |n, &axis| n.checked_mul(axis))
            .ok_or_else(|| bad("grid size overflow"))
    };
    let target = volume(&grid)?;
    let refs = spans[..spans.len() - 1].iter().try_fold(0usize, |n, g| {
        n.checked_add(volume(g)?)
            .ok_or_else(|| bad("reference size overflow"))
    })?;
    let seq = target
        .checked_add(refs)
        .and_then(|n| n.checked_add(globals))
        .ok_or_else(|| bad("size overflow"))?;
    let plan = frozen_plan(grid)?;
    let layouts = plan
        .shapes
        .iter()
        .map(|&s| {
            let tiles = |g: &[usize; 3]| {
                (0..3)
                    .try_fold(1usize, |n, i| n.checked_mul(g[i].div_ceil(s.0[i])))
                    .ok_or_else(|| bad("tile size overflow"))
            };
            let reference = spans[..spans.len() - 1].iter().try_fold(0usize, |n, g| {
                n.checked_add(tiles(g)?)
                    .ok_or_else(|| bad("reference tile overflow"))
            })?;
            let video = reference
                .checked_add(tiles(&grid)?)
                .ok_or_else(|| bad("tile size overflow"))?;
            Ok((
                s,
                LayoutSize {
                    seq,
                    slots: video
                        .checked_add(globals.div_ceil(TILE))
                        .and_then(|n| n.checked_mul(TILE))
                        .ok_or_else(|| bad("padding size overflow"))?,
                    video,
                    reference,
                    ref_tokens: refs,
                    target_tokens: target,
                },
            ))
        })
        .collect::<Result<_>>()?;
    Ok((plan, grid, layouts))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn req() -> GenerationRequest {
        GenerationRequest {
            width: 1344,
            height: 768,
            frames: Some(124),
            veda: Some(VedaRequest {
                enabled: true,
                ..Default::default()
            }),
            ..Default::default()
        }
    }
    fn fixture_choice(grid: [usize; 3]) -> [usize; 3] {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/veda/frozen.json")).unwrap();
        let choice = fixture["choices"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["grid"] == serde_json::json!(grid))
            .expect("fixture choice for the priced grid");
        assert_eq!(choice["exact"], true);
        serde_json::from_value(choice["plan_grid"].clone()).unwrap()
    }
    #[test]
    fn reference_and_keyframe_spans_are_priced_in_patch_units() {
        // A 320x576 reference at the default 2048 short edge is 2048x3680 px on the
        // stride-32 lattice: 64x115 patches, i.e. 7,360 reference tokens.
        let base = for_request(&req(), Backend::Cuda).unwrap().unwrap();
        let mut refs = req();
        refs.conditioning.push(Conditioning::Reference {
            image: crate::Image {
                width: 320,
                height: 576,
                pixels: vec![],
            },
            strength: None,
        });
        let rb = for_request(&refs, Backend::Cuda).unwrap().unwrap();
        let sizes = |r: &GenerationRequest| {
            let (_, _, layouts) = price_layouts(Request::from(r)).unwrap();
            *layouts.values().next().unwrap()
        };
        let (b, r) = (sizes(&req()), sizes(&refs));
        assert_eq!(b.ref_tokens, 0);
        assert_eq!(b.target_tokens, 37 * 24 * 42);
        assert_eq!(r.ref_tokens, 115 * 64);
        assert_eq!(r.target_tokens, b.target_tokens);
        let mut keyframe = req();
        keyframe.conditioning.push(Conditioning::Keyframe {
            image: crate::Image {
                width: 1344,
                height: 768,
                pixels: vec![],
            },
            frame_idx: 0,
            strength: 1.0,
        });
        assert_eq!(sizes(&keyframe).ref_tokens, 24 * 42);
        assert!(rb.total_bytes > base.total_bytes);
    }
    #[test]
    fn malformed_huge_request_is_refused_without_allocating_or_panicking() {
        let mut r = req();
        r.width = u32::MAX;
        r.height = u32::MAX;
        r.frames = Some(u32::MAX);
        assert!(for_request(&r, Backend::Mlx).is_err());
    }

    #[test]
    fn off_never_prices_or_parses_sparse_state() {
        let mut r = GenerationRequest::default();
        for b in [Backend::Mlx, Backend::Cuda] {
            assert!(for_request(&r, b).unwrap().is_none());
            r.veda = Some(VedaRequest::default());
            assert!(for_request(&r, b).unwrap().is_none());
        }
    }
    #[test]
    fn request_bound_owns_full_lifetimes_and_no_free_memory_axis() {
        for backend in [Backend::Mlx, Backend::Cuda] {
            let b = for_request(&req(), backend).unwrap().unwrap();
            assert!(b.predictor_host_bytes > 500_000_000);
            for cost in [
                b.predictor_device_bytes,
                b.predictor_load_bytes,
                b.layout_host_bytes,
                b.position_rows_bytes,
                b.tile_order_bytes,
                b.scorer_bytes,
                b.selector_bytes,
                b.retained_output_bytes,
                b.input_qkv_bytes,
                b.self_test_bytes,
                b.dense_fallback_bytes,
            ] {
                assert!(cost > 0);
            }
            let sum = b.predictor_host_bytes
                + b.predictor_device_bytes
                + b.predictor_load_bytes
                + b.layout_host_bytes
                + b.position_rows_bytes
                + b.tile_order_bytes
                + b.scorer_bytes
                + b.selector_bytes
                + b.gather_bytes
                + b.kernel_scratch_bytes
                + b.retained_output_bytes
                + b.input_qkv_bytes
                + b.self_test_bytes
                + b.dense_fallback_bytes;
            assert_eq!(sum, b.total_bytes);
            assert_eq!(
                for_request(&req(), backend).unwrap().unwrap().total_bytes,
                b.total_bytes
            );
            assert!(!b.plan.is_empty());
            // Patch grid (pixels / 32), the same grid runtime packs and the frozen
            // fixture chooses an exact plan for.
            assert_eq!(b.execution_grid, [37, 24, 42]);
            assert_eq!(b.plan_grid, fixture_choice([37, 24, 42]));
            assert_eq!(b.plan, frozen_plan([37, 24, 42]).unwrap().name);
        }
    }
    #[test]
    fn cuda_cold_envelope_prices_every_control_budget_and_forced_fallback() {
        let base = for_request(&req(), Backend::Cuda)
            .unwrap()
            .unwrap()
            .total_bytes;
        for generated in [
            VedaBudget::SparsityPercent(0.),
            VedaBudget::KeyTiles(1),
            VedaBudget::KeyTiles(u32::MAX),
        ] {
            let mut r = req();
            let controls = r.veda.as_mut().unwrap();
            controls.generated = generated;
            controls.reference = generated;
            controls.full_attention_layers = vec![0, 49];
            controls.full_attention_steps = vec![0];
            assert_eq!(
                for_request(&r, Backend::Cuda).unwrap().unwrap().total_bytes,
                base
            );
        }
    }

    #[test]
    fn cuda_batches_heads_within_its_copy_allowance() {
        let cuda = for_request(&req(), Backend::Cuda).unwrap().unwrap();
        let mlx = for_request(&req(), Backend::Mlx).unwrap().unwrap();
        for (shape, c) in &cuda.chunks {
            eprintln!(
                "cuda {shape:?}: {} heads, {} bytes",
                c.heads, c.copies_bytes
            );
            assert!(c.heads > 1, "{shape:?} CUDA batches several heads");
            assert!(c.copies_bytes <= Limits::for_backend(Backend::Cuda).copies);
            assert!(c.refusal.is_none());
        }
        for c in mlx.chunks.values() {
            assert!(c.copies_bytes <= Limits::default().copies.max(c.minimum_copies_bytes));
        }
        // The batched launch's conversion/quant scratch is priced for every head of the batch.
        let r = req();
        let (plan, grid, layouts) = price_layouts(Request::from(&r)).unwrap();
        let single = bound(
            r.veda.as_ref().unwrap(),
            Backend::Cuda,
            &plan,
            grid,
            &layouts,
            50,
            56,
            128,
            1,
            2,
            Limits::default(),
        )
        .unwrap();
        assert!(single.chunks.values().all(|c| c.heads == 1));
        let heads = cuda.chunks.values().map(|c| c.heads as u64).max().unwrap();
        assert!(cuda.kernel_scratch_bytes > 10 * single.kernel_scratch_bytes);
        assert!(cuda.kernel_scratch_bytes <= heads * single.kernel_scratch_bytes);
    }
    #[test]
    fn tiny_limits_price_minimum_batch_and_refuse_before_gather() {
        let l = LayoutSize {
            seq: 15000,
            slots: 16384,
            video: 110,
            reference: 60,
            ref_tokens: 7200,
            target_tokens: 5200,
        };
        let r = VedaRequest {
            enabled: true,
            reference: VedaBudget::SparsityPercent(0.),
            ..Default::default()
        };
        for backend in [Backend::Mlx, Backend::Cuda] {
            let tiny = chunk(
                l,
                56,
                128,
                2,
                backend,
                &r,
                Limits {
                    copies: 1,
                    attention: 1,
                },
            )
            .unwrap();
            assert_eq!(tiny.heads, 1);
            assert!(tiny.minimum_copies_bytes > 1);
            assert!(tiny.refusal.is_some());
            let normal = chunk(l, 56, 128, 2, backend, &r, Limits::default()).unwrap();
            assert!(normal.heads >= tiny.heads);
            assert!(normal.copies_bytes >= tiny.minimum_copies_bytes);
            if backend == Backend::Mlx {
                assert!(tiny.minimum_attention_bytes > 1);
            }
        }
    }
    #[test]
    fn long_nearest_reference_and_dense_fallback_are_priced() {
        let base = for_request(&req(), Backend::Cuda).unwrap().unwrap();
        let mut long = req();
        long.frames = Some(345);
        let b = for_request(&long, Backend::Cuda).unwrap().unwrap();
        assert!(b.total_bytes > base.total_bytes);
        assert!(b.position_rows_bytes > base.position_rows_bytes);
        long.frames = Some(141);
        let nearest = for_request(&long, Backend::Mlx).unwrap().unwrap();
        assert_ne!(nearest.execution_grid, nearest.plan_grid);
        let mut refs = req();
        refs.conditioning.push(Conditioning::Reference {
            image: crate::Image {
                width: 320,
                height: 576,
                pixels: vec![],
            },
            strength: None,
        });
        let rb = for_request(&refs, Backend::Cuda).unwrap().unwrap();
        assert!(rb.total_bytes > base.total_bytes);
        assert!(rb.dense_fallback_bytes >= base.dense_fallback_bytes);
        assert!(rb.position_rows_bytes > base.position_rows_bytes);
        let mut dense = req();
        dense.veda.as_mut().unwrap().generated = VedaBudget::SparsityPercent(0.);
        let db = for_request(&dense, Backend::Mlx).unwrap().unwrap();
        let sparse = for_request(&req(), Backend::Mlx).unwrap().unwrap();
        for (shape, chunk) in &db.chunks {
            assert!(chunk.selected_keys >= sparse.chunks[shape].selected_keys);
        }
        assert!(db.dense_fallback_bytes > 0);
    }
}
