//! The Z VAE's evaluated CPU-stream graph, mirrored from `vae/*.rs`, `nn.rs`, and
//! `vae_tiling.rs`. These are native array descriptors and allocation requests, not
//! logical convolution/quantum prices. Evaluation, dropping a tile, and finishing a
//! stage never subtract anything from this graph's cumulative allocator bound.
use super::z_graph::Catalog;
use super::*;
use crate::quantum as q;
use mlx_gen::tiling::{TilingConfig, VaeTiling};

const GROUPS: u64 = 32;
const ENCODE_TILE: u64 = 256;
const HALF_STATS_CHUNK: u64 = 1 << 24;

fn error(message: impl Into<String>) -> Error {
    Error::Msg(format!("Z VAE CPU source graph: {}", message.into()))
}

fn nhwc(x: &Tensor) -> Result<[u64; 4]> {
    let shape: [u64; 4] = x.shape.clone().try_into().map_err(|_| error("NHWC rank"))?;
    if shape.contains(&0) || shape.iter().any(|&d| d > i32::MAX as u64) {
        return Err(error("empty or non-native NHWC dimension"));
    }
    Ok(shape)
}

/// `crate::array::scalar` is a host-created [1] f32 array, not a rank-zero scalar.
fn scalar(g: &mut Graph, site: &str) -> Result<Tensor> {
    g.immediate(&[1], Dtype::Float32, site)
}

fn pair(g: &mut Graph, c: &mut Catalog, base: &str) -> Result<(Tensor, Tensor)> {
    let weight = format!("{base}.weight");
    let bias = format!("{base}.bias");
    c.preload(g, [weight.clone(), bias.clone()])?;
    Ok((c.value(&weight)?, c.value(&bias)?))
}

fn conv(
    g: &mut Graph,
    c: &mut Catalog,
    base: &str,
    x: &Tensor,
    stride: u64,
    padding: u64,
) -> Result<Tensor> {
    let (weight, bias) = pair(g, c, base)?;
    g.conv2d(x, &weight, Some(&bias), stride, padding, base)
}

/// `ops.cpp::take`: Gather allocates [indices.shape..., slice_sizes...], then
/// Transpose moves the index axis into place and Squeeze removes the size-1
/// source axis. `common.cpp::{Transpose,Squeeze}` share that same backing and
/// preserve data_size/contiguous. In particular this is NOT a compact Slice view.
fn take_axis(g: &mut Graph, x: &Tensor, axis: usize, len: u64, site: &str) -> Result<Tensor> {
    if axis >= x.shape.len() || len == 0 || len > x.shape[axis] || len > i32::MAX as u64 {
        return Err(error("invalid host-index gather"));
    }
    g.immediate(&[len], Dtype::Int32, site)?;
    let mut shape = vec![len];
    shape.extend_from_slice(&x.shape);
    shape[axis + 1] = 1;
    let gathered = g.allocation(&shape, x.dtype, "cpu/indexing.cpp Gather::eval_cpu", site)?;
    let mut axes: Vec<usize> = (1..=axis).collect();
    axes.push(0);
    axes.extend(axis + 1..shape.len());
    let mut taken = gathered.transpose(&axes)?;
    // Squeeze::eval copies shared-buffer strides, flags and data_size verbatim.
    taken.shape.remove(axis + 1);
    taken.strides.remove(axis + 1);
    Ok(taken)
}

/// The native concatenate op is an identity for ONE input (ops.cpp:1275).
fn concat(g: &mut Graph, xs: &[Tensor], axis: usize, site: &str) -> Result<Tensor> {
    match xs {
        [x] => Ok(x.clone()),
        [] => Err(error("empty tile assembly")),
        _ => g.concat(xs, axis, site),
    }
}

/// Pinned mlx-rs `ops/shapes.rs::pad_device`: None constructs a rank-zero
/// Array::from_int(0), casts it to a.dtype(), then calls mlx_pad. The native
/// ops.cpp Pad casts it again (identity), and CPU Pad's Scalar copy allocates
/// the ENTIRE output even if every pad amount is zero.
fn pad_bottom_right(
    g: &mut Graph,
    x: &Tensor,
    bottom: u64,
    right: u64,
    site: &str,
) -> Result<Tensor> {
    let [b, h, w, c] = nhwc(x)?;
    let zero = g.scalar(Dtype::Int32, site)?;
    g.cast(&zero, x.dtype, site)?;
    g.allocation(
        &[b, add(h, bottom)?, add(w, right)?, c],
        x.dtype,
        "cpu/primitives.cpp Pad::eval_cpu Scalar copy",
        site,
    )
}

/// `nn::group_norm`: flatten spatial axes, transpose groups forward, flatten
/// again, affine-free CPU fast::layer_norm fallback, undo the regrouping, affine.
fn group_norm(
    g: &mut Graph,
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    site: &str,
) -> Result<Tensor> {
    let [b, h, w, c] = nhwc(x)?;
    check_affine(c, weight, bias)?;
    let spatial = mul(h, w)?;
    let grouped = g.reshape(x, &[b, spatial, GROUPS, c / GROUPS], site)?;
    let grouped = grouped.transpose(&[0, 2, 1, 3])?;
    let grouped = g.reshape(&grouped, &[b, GROUPS, mul(spatial, c / GROUPS)?], site)?;
    let grouped = g.layer_norm(&grouped, site)?;
    let grouped = g.reshape(&grouped, &[b, GROUPS, spatial, c / GROUPS], site)?;
    let grouped = grouped.transpose(&[0, 2, 1, 3])?;
    let normalized = g.reshape(&grouped, &[b, h, w, c], site)?;
    let out = g.binary(&normalized, weight, site)?;
    g.binary(&out, bias, site)
}

fn check_affine(channels: u64, weight: &Tensor, bias: &Tensor) -> Result<()> {
    if !channels.is_multiple_of(GROUPS) || weight.shape != [channels] || bias.shape != [channels] {
        return Err(error("GroupNorm affine shape/groups"));
    }
    Ok(())
}

fn norm(g: &mut Graph, c: &mut Catalog, base: &str, x: &Tensor) -> Result<Tensor> {
    let (weight, bias) = pair(g, c, base)?;
    group_norm(g, x, &weight, &bias, base)
}

struct GlobalNorm {
    mean: Tensor,
    inv_std: Tensor,
    weight: Tensor,
    bias: Tensor,
    channels: u64,
}

impl GlobalNorm {
    fn new(g: &mut Graph, x: &Tensor, weight: Tensor, bias: Tensor, site: &str) -> Result<Self> {
        let [b, h, w, c] = nhwc(x)?;
        check_affine(c, &weight, &bias)?;
        let (mean, inv_std) = if x.dtype == Dtype::Float16 {
            half_stats(g, x, site)?
        } else {
            // GlobalGroupNorm::new's f32/bf16 path is FULL resolution. Native
            // var(a) independently computes a second mean, square(subtract),
            // sum, NumberOfElements(inverted), multiply (ops.cpp:2319).
            let grouped = g.reshape(x, &[b, h, w, GROUPS, c / GROUPS], site)?;
            let grouped = grouped.transpose(&[0, 3, 1, 2, 4])?;
            let mean = g.mean(&grouped, &[2, 3, 4], site)?;
            let var_mean = g.mean(&grouped, &[2, 3, 4], site)?;
            let deviation = g.binary(&grouped, &var_mean, site)?;
            let squared = g.unary(&deviation, site)?;
            let sum = g.reduce(&squared, &[2, 3, 4], site)?;
            // Native var casts through at_least_float: the inverted count is f32
            // even for a bf16 sum, which promotes the variance to f32.
            let inverse_count = g.allocation(
                &[],
                Dtype::Float32,
                "common/common.cpp NumberOfElements inverted variance normalizer",
                site,
            )?;
            let variance = g.binary(&sum, &inverse_count, site)?;
            let eps = scalar(g, site)?;
            // bf16 + the actual f32 epsilon PROMOTES. Do not cast it back.
            let variance = g.binary(&variance, &eps, site)?;
            let inv_std = g.unary(&variance, site)?;
            (mean, inv_std)
        };
        Ok(Self {
            mean,
            inv_std,
            weight,
            bias,
            channels: c,
        })
    }

    fn from_catalog(g: &mut Graph, c: &mut Catalog, base: &str, x: &Tensor) -> Result<Self> {
        let (weight, bias) = pair(g, c, base)?;
        Self::new(g, x, weight, bias, base)
    }

    fn apply(&self, g: &mut Graph, tile: &Tensor, site: &str) -> Result<Tensor> {
        let [b, h, w, c] = nhwc(tile)?;
        if c != self.channels {
            return Err(error("GlobalGroupNorm crop channels"));
        }
        let grouped = g.reshape(tile, &[b, h, w, GROUPS, c / GROUPS], site)?;
        let grouped = grouped.transpose(&[0, 3, 1, 2, 4])?;
        let deviation = g.binary(&grouped, &self.mean, site)?;
        let normalized = g.binary(&deviation, &self.inv_std, site)?;
        let normalized = normalized.transpose(&[0, 2, 3, 1, 4])?;
        let normalized = g.reshape(&normalized, &[b, h, w, c], site)?;
        let out = g.binary(&normalized, &self.weight, site)?;
        let out = g.binary(&out, &self.bias, site)?;
        if tile.dtype == Dtype::Float16 && out.dtype != tile.dtype {
            g.cast(&out, tile.dtype, site)
        } else {
            Ok(out)
        }
    }
}

fn half_grouped(g: &mut Graph, x: &Tensor, rows: u64, site: &str) -> Result<Tensor> {
    let [b, _, w, c] = nhwc(x)?;
    // vae_tiling::slice_axis takes host i32 indices + Gather, including a full
    // row-span. This is not attention::slice_axis's zero-allocation Split.
    let crop = take_axis(g, x, 1, rows, site)?;
    let crop = g.cast(&crop, Dtype::Float32, site)?;
    let crop = g.reshape(&crop, &[b, rows, w, GROUPS, c / GROUPS], site)?;
    crop.transpose(&[0, 3, 1, 2, 4])
}

fn half_stats(g: &mut Graph, x: &Tensor, site: &str) -> Result<(Tensor, Tensor)> {
    let [b, h, w, c] = nhwc(x)?;
    let row_elements = mul(mul(w, c)?, b)?;
    let rows = (HALF_STATS_CHUNK / row_elements.max(1)).clamp(1, h);
    let mut sum: Option<Tensor> = None;
    let mut y = 0;
    while y < h {
        let end = add(y, rows)?.min(h);
        let grouped = half_grouped(g, x, end - y, site)?;
        let part = g.reduce(&grouped, &[2, 3, 4], site)?;
        sum = Some(match sum {
            Some(acc) => g.binary(&acc, &part, site)?,
            None => part,
        });
        y = end;
    }
    let divisor = scalar(g, site)?;
    let mean = g.binary(
        &sum.ok_or_else(|| error("empty half mean"))?,
        &divisor,
        site,
    )?;
    let mut centred: Option<Tensor> = None;
    y = 0;
    while y < h {
        let end = add(y, rows)?.min(h);
        let grouped = half_grouped(g, x, end - y, site)?;
        let deviation = g.binary(&grouped, &mean, site)?;
        // Source multiplies the deviation by itself (rather than native Square).
        let squared = g.binary(&deviation, &deviation, site)?;
        let part = g.reduce(&squared, &[2, 3, 4], site)?;
        centred = Some(match centred {
            Some(acc) => g.binary(&acc, &part, site)?,
            None => part,
        });
        y = end;
    }
    let divisor = scalar(g, site)?;
    let variance = g.binary(
        &centred.ok_or_else(|| error("empty half variance"))?,
        &divisor,
        site,
    )?;
    let eps = scalar(g, site)?;
    let variance = g.binary(&variance, &eps, site)?;
    Ok((mean, g.unary(&variance, site)?))
}

fn preprocess(g: &mut Graph, x: &Tensor, norm: Option<&GlobalNorm>, site: &str) -> Result<Tensor> {
    match norm {
        Some(norm) => {
            let x = norm.apply(g, x, site)?;
            g.silu(&x, site)
        }
        None => Ok(x.clone()),
    }
}

/// Shared normal (NOT f32_accum) tiled 3x3 helper. Four actual i32-index
/// Gathers per tile, with exact intermediate layouts; horizontal then vertical
/// concatenate. The fast path calls preprocess/conv once, with no crop/assembly.
fn tiled_conv(
    p: &mut Program,
    c: &mut Catalog,
    base: &str,
    x: &Tensor,
    edge: u64,
    norm: Option<&GlobalNorm>,
    labels: &q::VaeLabels,
) -> Result<Tensor> {
    let [_, h, w, _] = nhwc(x)?;
    let (weight, bias) = pair(&mut p.graph, c, base)?;
    if edge < 3 || weight.shape.len() != 4 || weight.shape[1..3] != [3, 3] {
        return Err(error("tiled 3x3 kernel/edge"));
    }
    if h <= edge && w <= edge {
        let tile = preprocess(&mut p.graph, x, norm, base)?;
        let out = p.graph.conv2d(&tile, &weight, Some(&bias), 1, 1, base)?;
        p.mark(labels.tile)?;
        return Ok(out);
    }
    let core = edge - 2;
    // Canonical Model::tile_extents enumerates DISTINCT actual halo extents (cartesian product of
    // the sorted per-axis extent sets), not cores. Bind every real core to the point of its exact
    // extent class; the stored upper is the max cumulative prefix over all cores in that class.
    let axis = |n: u64| -> Result<Vec<u64>> {
        let mut set = BTreeSet::new();
        let mut start = 0;
        while start < n {
            let end = add(start, core)?.min(n);
            set.insert(add(end, 1)?.min(n) - start.saturating_sub(1));
            start = end;
        }
        Ok(set.into_iter().collect())
    };
    let (hs, ws) = (axis(h)?, axis(w)?);
    let tile_base = p.points.get(labels.tile).copied().unwrap_or(0);
    let mut rows = Vec::new();
    let mut y0 = 0;
    while y0 < h {
        let y1 = add(y0, core)?.min(h);
        let iy0 = y0.saturating_sub(1);
        let iy1 = add(y1, 1)?.min(h);
        let mut cores = Vec::new();
        let mut x0 = 0;
        while x0 < w {
            let x1 = add(x0, core)?.min(w);
            let ix0 = x0.saturating_sub(1);
            let ix1 = add(x1, 1)?.min(w);
            let tile = take_axis(&mut p.graph, x, 1, iy1 - iy0, base)?;
            let tile = take_axis(&mut p.graph, &tile, 2, ix1 - ix0, base)?;
            let tile = preprocess(&mut p.graph, &tile, norm, base)?;
            let convolved = p.graph.conv2d(&tile, &weight, Some(&bias), 1, 1, base)?;
            let out = take_axis(&mut p.graph, &convolved, 1, y1 - y0, base)?;
            let out = take_axis(&mut p.graph, &out, 2, x1 - x0, base)?;
            let ih = hs
                .binary_search(&(iy1 - iy0))
                .map_err(|_| error("tile height outside canonical extent classes"))?;
            let iw = ws
                .binary_search(&(ix1 - ix0))
                .map_err(|_| error("tile width outside canonical extent classes"))?;
            let offset = ih
                .checked_mul(ws.len())
                .and_then(|v| v.checked_add(iw))
                .and_then(|v| v.checked_add(tile_base))
                .ok_or_else(|| error("tile point overflow"))?;
            p.mark_at(labels.tile, offset)?;
            cores.push(out);
            x0 = x1;
        }
        rows.push(concat(&mut p.graph, &cores, 2, base)?);
        y0 = y1;
    }
    let next = hs
        .len()
        .checked_mul(ws.len())
        .and_then(|v| v.checked_add(tile_base))
        .ok_or_else(|| error("tile point overflow"))?;
    p.points.insert(labels.tile.into(), next);
    let out = concat(&mut p.graph, &rows, 1, base)?;
    // vae_tiling calls array::contiguous after assembly. Concatenations are
    // row-major, but a one-input concat preserves its gather-transposed view.
    let flat = p.graph.reshape(&out, &[out.elements()?, 1], base)?;
    p.graph.reshape(&flat, &out.shape, base)
}

fn shortcut(g: &mut Graph, c: &mut Catalog, base: &str, x: &Tensor) -> Result<Tensor> {
    let base = format!("{base}.conv_shortcut");
    // Exact constructor: one present key alone does NOT enable the shortcut.
    if c.has(&format!("{base}.weight")) && c.has(&format!("{base}.bias")) {
        conv(g, c, &base, x, 1, 0)
    } else {
        Ok(x.clone())
    }
}

fn resnet(
    p: &mut Program,
    c: &mut Catalog,
    base: &str,
    input: &Tensor,
    edge: Option<u64>,
    labels: &q::VaeLabels,
) -> Result<Tensor> {
    let x = input.transpose(&[0, 2, 3, 1])?;
    let h = if let Some(edge) = edge {
        let n1 = GlobalNorm::from_catalog(&mut p.graph, c, &format!("{base}.norm1"), &x)?;
        let h = tiled_conv(p, c, &format!("{base}.conv1"), &x, edge, Some(&n1), labels)?;
        let n2 = GlobalNorm::from_catalog(&mut p.graph, c, &format!("{base}.norm2"), &h)?;
        tiled_conv(p, c, &format!("{base}.conv2"), &h, edge, Some(&n2), labels)?
    } else {
        let h = norm(&mut p.graph, c, &format!("{base}.norm1"), &x)?;
        let h = p.graph.silu(&h, base)?;
        let h = conv(&mut p.graph, c, &format!("{base}.conv1"), &h, 1, 1)?;
        let h = norm(&mut p.graph, c, &format!("{base}.norm2"), &h)?;
        let h = p.graph.silu(&h, base)?;
        conv(&mut p.graph, c, &format!("{base}.conv2"), &h, 1, 1)?
    };
    let residual = shortcut(&mut p.graph, c, base, &x)?;
    let out = p.graph.binary(&residual, &h, base)?;
    // ONE stage encloses BOTH tiled convolutions, both global statistics,
    // optional shortcut, residual sum, and all tile/concat operations.
    p.mark(if edge.is_some() {
        labels.stage
    } else {
        labels.resnet
    })?;
    out.transpose(&[0, 3, 1, 2])
}

fn attention(
    p: &mut Program,
    c: &mut Catalog,
    base: &str,
    input: &Tensor,
    label: &str,
) -> Result<Tensor> {
    let x = input.transpose(&[0, 2, 3, 1])?;
    let [b, h, w, channels] = nhwc(&x)?;
    let n = mul(h, w)?;
    let normalized = norm(&mut p.graph, c, &format!("{base}.group_norm"), &x)?;
    let mut qkv = Vec::new();
    for name in ["to_q", "to_k", "to_v"] {
        let projected = c.linear(&mut p.graph, &format!("{base}.{name}"), &normalized, true)?;
        let projected = p.graph.reshape(&projected, &[b, n, 1, channels], base)?;
        qkv.push(projected.transpose(&[0, 2, 1, 3])?);
    }
    let rows = i32::try_from(n).map_err(|_| error("attention token count"))?;
    let budget = mlx_gen::attention::AttentionBudget::from_score_elements(
        mlx_gen::ios_quantum::VAE_ATTENTION_SCORE_ELEMENTS,
        true,
    );
    // The source asks the shared planner itself: query_block(b, h, sq, sk).
    let block = budget.query_block(1, 1, rows, rows) as u64;
    let out = if block >= n {
        p.graph.sdpa(&qkv[0], &qkv[1], &qkv[2], None, base)?
    } else {
        let mut outs = Vec::new();
        let mut start = 0;
        while start < n {
            let len = block.min(n - start);
            // attention::slice_axis uses boundary Split views, not host indices.
            let query = qkv[0].split(2, len)?;
            outs.push(p.graph.sdpa(&query, &qkv[1], &qkv[2], None, base)?);
            start = add(start, len)?;
        }
        concat(&mut p.graph, &outs, 2, base)?
    };
    let out = out.transpose(&[0, 2, 1, 3])?;
    let out = p.graph.reshape(&out, &[b, h, w, channels], base)?;
    let out = c.linear(&mut p.graph, &format!("{base}.to_out.0"), &out, true)?;
    let out = p.graph.binary(&x, &out, base)?;
    p.mark(label)?;
    out.transpose(&[0, 3, 1, 2])
}

fn mid(
    p: &mut Program,
    c: &mut Catalog,
    base: &str,
    input: &Tensor,
    labels: &q::VaeLabels,
) -> Result<Tensor> {
    let h = resnet(p, c, &format!("{base}.resnets.0"), input, None, labels)?;
    let h = attention(p, c, &format!("{base}.attentions.0"), &h, labels.attention)?;
    resnet(p, c, &format!("{base}.resnets.1"), &h, None, labels)
}

fn nearest(g: &mut Graph, x: &Tensor, site: &str) -> Result<Tensor> {
    let [b, h, w, c] = nhwc(x)?;
    let x6 = g.reshape(x, &[b, h, 1, w, 1, c], site)?;
    let broadcast = x6.broadcast(&[b, h, 2, w, 2, c])?;
    g.reshape(&broadcast, &[b, mul(h, 2)?, mul(w, 2)?, c], site)
}

fn upsample(
    p: &mut Program,
    c: &mut Catalog,
    base: &str,
    input: &Tensor,
    edge: Option<u64>,
) -> Result<Tensor> {
    let x = input.transpose(&[0, 2, 3, 1])?;
    let up = nearest(&mut p.graph, &x, base)?;
    let out = if let Some(edge) = edge {
        tiled_conv(p, c, base, &up, edge, None, &q::DECODE_LABELS)?
    } else {
        conv(&mut p.graph, c, base, &up, 1, 1)?
    };
    p.mark(if edge.is_some() {
        q::VAE_DECODE_TILED_STAGE
    } else {
        q::VAE_UPSAMPLE
    })?;
    out.transpose(&[0, 3, 1, 2])
}

/// All real stride-2 cores are evaluated. The canonical encode window list has
/// ONE representative tile point per downsample, unlike its stride-1 tile list.
/// Bind each actual core to that one point via mark_at/max. Cumulative upper()
/// never decreases, so the last core proves every earlier native guard instant;
/// counter advancement occurs once, BEFORE the final row/output concats.
fn downsample(
    p: &mut Program,
    c: &mut Catalog,
    base: &str,
    input: &Tensor,
    edge: Option<u64>,
) -> Result<Tensor> {
    let x = input.transpose(&[0, 2, 3, 1])?;
    if let Some(edge) = edge {
        let [_, h, w, _] = nhwc(&x)?;
        if h < 2 || w < 2 || edge == 0 {
            return Err(error("downsample tiled extent"));
        }
        let (oh, ow) = (add((h - 2) / 2, 1)?, add((w - 2) / 2, 1)?);
        let (weight, bias) = pair(&mut p.graph, c, base)?;
        let tile_point = p.points.get(q::VAE_ENCODE_TILE).copied().unwrap_or(0);
        let mut rows = Vec::new();
        let mut y0 = 0;
        while y0 < oh {
            let y1 = add(y0, edge)?.min(oh);
            let in_y0 = mul(2, y0)?;
            let need_h = add(mul(2, y1 - y0)?, 1)?;
            let have_h = need_h.min(h - in_y0);
            let mut cores = Vec::new();
            let mut x0 = 0;
            while x0 < ow {
                let x1 = add(x0, edge)?.min(ow);
                let in_x0 = mul(2, x0)?;
                let need_w = add(mul(2, x1 - x0)?, 1)?;
                let have_w = need_w.min(w - in_x0);
                // This helper imports attention::slice_axis: TWO Split views,
                // never the Gather-based vae_tiling::slice_axis above.
                let crop = x.split(1, have_h)?.split(2, have_w)?;
                let crop =
                    pad_bottom_right(&mut p.graph, &crop, need_h - have_h, need_w - have_w, base)?;
                let out = p.graph.conv2d(&crop, &weight, Some(&bias), 2, 0, base)?;
                p.mark_at(q::VAE_ENCODE_TILE, tile_point)?;
                if y1 == oh && x1 == ow {
                    p.mark(q::VAE_ENCODE_TILE)?;
                }
                cores.push(out);
                x0 = x1;
            }
            rows.push(concat(&mut p.graph, &cores, 2, base)?);
            y0 = y1;
        }
        let out = concat(&mut p.graph, &rows, 1, base)?;
        p.mark(q::VAE_ENCODE_TILED_STAGE)?;
        out.transpose(&[0, 3, 1, 2])
    } else {
        let padded = pad_bottom_right(&mut p.graph, &x, 1, 1, base)?;
        let out = conv(&mut p.graph, c, base, &padded, 2, 0)?;
        p.mark(q::VAE_DOWNSAMPLE)?;
        out.transpose(&[0, 3, 1, 2])
    }
}

fn input_conv(
    p: &mut Program,
    c: &mut Catalog,
    base: &str,
    input: &Tensor,
    edge: Option<u64>,
    label: &str,
    labels: &q::VaeLabels,
) -> Result<Tensor> {
    let x = input.transpose(&[0, 2, 3, 1])?;
    let out = if let Some(edge) = edge {
        tiled_conv(p, c, base, &x, edge, None, labels)?
    } else {
        conv(&mut p.graph, c, base, &x, 1, 1)?
    };
    p.mark(if edge.is_some() { labels.stage } else { label })?;
    out.transpose(&[0, 3, 1, 2])
}

fn head(
    p: &mut Program,
    c: &mut Catalog,
    prefix: &str,
    input: &Tensor,
    edge: Option<u64>,
) -> Result<Tensor> {
    let x = input.transpose(&[0, 2, 3, 1])?;
    let norm_base = format!("{prefix}conv_norm_out.norm");
    let conv_base = format!("{prefix}conv_out.conv");
    let out = if let Some(edge) = edge {
        let norm = GlobalNorm::from_catalog(&mut p.graph, c, &norm_base, &x)?;
        tiled_conv(p, c, &conv_base, &x, edge, Some(&norm), &q::DECODE_LABELS)?
    } else {
        let x = norm(&mut p.graph, c, &norm_base, &x)?;
        let x = p.graph.silu(&x, &conv_base)?;
        conv(&mut p.graph, c, &conv_base, &x, 1, 1)?
    };
    // Encoder head is marked by encode AFTER latent_mean's Gather and affine.
    if prefix.is_empty() {
        p.mark(if edge.is_some() {
            q::VAE_DECODE_TILED_STAGE
        } else {
            q::VAE_DECODE_OUT
        })?;
    }
    out.transpose(&[0, 3, 1, 2])
}

/// NCHW [1,3,H,W] -> [1,16,H/8,W/8]. Same automatic bounded encode
/// selection as `Vae::encode` under the explicitly bound iOS quantum scope AND the >256px extent.
pub(super) fn encode(p: &mut Program, catalog: &mut Catalog, input: &Tensor) -> Result<Tensor> {
    if input.shape.len() != 4 || input.shape[0] != 1 || input.shape[1] != 3 {
        return Err(error("encode requires [1,3,H,W]"));
    }
    // `Vae::encode` gates on ios_quantum::enabled(). The certificate never samples the CALLER'S
    // thread-local state: window_certificates requires the explicit execution fact that the run
    // being certified executes inside the iOS quantum scope (device cfg or CPU-fixture headroom).
    if !p.ios_quantum_scope {
        return Err(error(
            "encode certificate requires the explicit iOS quantum scope execution fact",
        ));
    }
    let tiled = input.shape[2] > ENCODE_TILE || input.shape[3] > ENCODE_TILE;
    let labels = &q::ENCODE_LABELS;
    let mut h = input_conv(
        p,
        catalog,
        "encoder.conv_in.conv",
        input,
        tiled.then_some(ENCODE_TILE),
        q::VAE_ENCODE_IN,
        labels,
    )?;
    let mut scale = 1;
    for block in 0..4 {
        let edge = ENCODE_TILE.div_ceil(scale).max(3);
        for layer in 0..2 {
            h = resnet(
                p,
                catalog,
                &format!("encoder.down_blocks.{block}.resnets.{layer}"),
                &h,
                tiled.then_some(edge),
                labels,
            )?;
        }
        if block < 3 {
            scale = mul(scale, 2)?;
            h = downsample(
                p,
                catalog,
                &format!("encoder.down_blocks.{block}.downsamplers.0.conv"),
                &h,
                tiled.then_some(ENCODE_TILE.div_ceil(scale).max(1)),
            )?;
        }
    }
    h = mid(p, catalog, "encoder.mid_block", &h, labels)?;
    h = head(p, catalog, "encoder.", &h, None)?;
    if h.shape[1] % 2 != 0 {
        return Err(error("latent distribution must have even channels"));
    }
    let mean = take_axis(
        &mut p.graph,
        &h,
        1,
        h.shape[1] / 2,
        "vae/decoder.rs latent_mean indices/take",
    )?;
    let shift = scalar(&mut p.graph, "vae/decoder.rs latent_mean shift")?;
    let mean = p
        .graph
        .binary(&mean, &shift, "vae/decoder.rs latent_mean subtract")?;
    let scaling = scalar(&mut p.graph, "vae/decoder.rs latent_mean scaling")?;
    let out = p
        .graph
        .binary(&mean, &scaling, "vae/decoder.rs latent_mean multiply")?;
    p.mark(q::VAE_ENCODE_OUT)?;
    Ok(out)
}

fn edge_at(tile_px: i32, scale: u64) -> Result<u64> {
    let scale = i32::try_from(scale).map_err(|_| error("tile scale"))?;
    let numerator = tile_px
        .checked_add(scale - 1)
        .ok_or_else(|| error("tile edge overflow"))?;
    Ok((numerator / scale).max(3) as u64)
}

/// NCHW [1,16,H/8,W/8] -> [1,3,H,W]. The singleton frame axis added
/// by Vae::decode/decode_tiled is a view and is omitted from this caller API.
pub(super) fn decode(
    p: &mut Program,
    catalog: &mut Catalog,
    latents: &Tensor,
    tiling: Option<&TilingConfig>,
) -> Result<Tensor> {
    if latents.shape.len() != 4 || latents.shape[0] != 1 || latents.shape[1] != 16 {
        return Err(error("decode requires [1,16,h,w]"));
    }
    let lh = i32::try_from(latents.shape[2]).map_err(|_| error("decode latent height"))?;
    let lw = i32::try_from(latents.shape[3]).map_err(|_| error("decode latent width"))?;
    let tiled = tiling.filter(|t| t.needs_tiling(VaeTiling::QWEN_IMAGE, 1, lh, lw));
    let tile_px = tiled
        .map(|t| {
            t.spatial
                .as_ref()
                .map(|s| s.tile_px)
                .ok_or_else(|| error("tiled decode requires spatial tiling"))
        })
        .transpose()?;
    // Both real f32 [1] constants and their arithmetic are INSIDE DECODE_IN.
    let scale = scalar(&mut p.graph, "vae/decoder.rs decode reciprocal scale")?;
    let scaled = p
        .graph
        .binary(latents, &scale, "vae/decoder.rs decode scale multiply")?;
    let shift = scalar(&mut p.graph, "vae/decoder.rs decode shift")?;
    let scaled = p
        .graph
        .binary(&scaled, &shift, "vae/decoder.rs decode shift add")?;
    let mut h = input_conv(
        p,
        catalog,
        "conv_in.conv",
        &scaled,
        None,
        q::VAE_DECODE_IN,
        &q::DECODE_LABELS,
    )?;
    h = mid(p, catalog, "mid_block", &h, &q::DECODE_LABELS)?;
    let mut remaining = 8;
    for block in 0..4 {
        let edge = tile_px.map(|t| edge_at(t, remaining)).transpose()?;
        for layer in 0..3 {
            h = resnet(
                p,
                catalog,
                &format!("up_blocks.{block}.resnets.{layer}"),
                &h,
                edge,
                &q::DECODE_LABELS,
            )?;
        }
        if block < 3 {
            remaining /= 2;
            let edge = tile_px.map(|t| edge_at(t, remaining)).transpose()?;
            h = upsample(
                p,
                catalog,
                &format!("up_blocks.{block}.upsamplers.0.conv"),
                &h,
                edge,
            )?;
        }
    }
    head(p, catalog, "", &h, tile_px.map(|t| t.max(3) as u64))
}

/// Exact shared image.rs CPU-readback chain. Four host f32 constants (half is
/// reused), five binary ops, Round(decimals=0), then NHWC flatten/readback.
/// The Rust Vec<u8> conversion is host-only: there is NO MLX UInt8 array.
pub(super) fn output_image(p: &mut Program, decoded: &Tensor) -> Result<()> {
    if decoded.shape.len() != 4 || decoded.shape[0] != 1 || decoded.shape[1] != 3 {
        return Err(error("output image requires [1,3,H,W]"));
    }
    let site = "image.rs decoded_to_interleaved_u8";
    let decoded = p.graph.cast(decoded, Dtype::Float32, site)?;
    let half = scalar(&mut p.graph, site)?;
    let x = p.graph.binary(&decoded, &half, site)?;
    let x = p.graph.binary(&x, &half, site)?;
    let zero = scalar(&mut p.graph, site)?;
    let x = p.graph.binary(&x, &zero, site)?;
    let one = scalar(&mut p.graph, site)?;
    let x = p.graph.binary(&x, &one, site)?;
    let x = x.transpose(&[0, 2, 3, 1])?;
    let scale = scalar(&mut p.graph, site)?;
    let x = p.graph.binary(&x, &scale, site)?;
    let x = p.graph.unary(&x, site)?;
    p.graph.reshape(&x, &[x.elements()?], site)?;
    p.mark(q::DECODED_TO_IMAGE)
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests operate only on host descriptors and checked allocator-request
    // metadata. They never construct/evaluate an mlx_rs::Array or query a device.
    #[test]
    fn take_axis_preserves_native_gather_then_transpose_layout() {
        let mut g = Graph::new(16_384).unwrap();
        let x = Tensor::row(&[1, 7, 11, 32], Dtype::Float16).unwrap();
        let y = take_axis(&mut g, &x, 1, 4, "host take height").unwrap();
        assert_eq!(y.shape, [1, 4, 11, 32]);
        assert_eq!(y.strides, [352, 352, 32, 1]);
        assert_eq!(y.data_size, 4 * 11 * 32);
        let y = take_axis(&mut g, &y, 2, 5, "host take width").unwrap();
        assert_eq!(y.shape, [1, 4, 5, 32]);
        assert_eq!(y.strides, [128, 32, 128, 1]);
        assert_eq!(y.data_size, 4 * 5 * 32);
        assert!(y.contiguous);
        assert!(!y.row_contiguous());
    }

    #[test]
    fn global_norm_dtype_paths_follow_real_scalar_promotion_and_half_castback() {
        for (dtype, expected) in [
            (Dtype::Float32, Dtype::Float32),
            (Dtype::Bfloat16, Dtype::Float32),
            (Dtype::Float16, Dtype::Float16),
        ] {
            let mut g = Graph::new(16_384).unwrap();
            let x = Tensor::row(&[1, 4, 5, 32], dtype).unwrap();
            let weight = Tensor::row(&[32], dtype).unwrap();
            let bias = Tensor::row(&[32], dtype).unwrap();
            let norm = GlobalNorm::new(&mut g, &x, weight, bias, "host norm").unwrap();
            assert_eq!(norm.inv_std.dtype, Dtype::Float32);
            let out = norm.apply(&mut g, &x, "host crop").unwrap();
            assert_eq!(out.dtype, expected);
            assert_eq!(out.shape, x.shape);
        }
    }
}
