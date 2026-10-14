//! Frozen predictor bundle and tile layout. No device work occurs in this module.
use crate::weightsmeta::{CheckpointMeta, Dtype};
use crate::{Error, Result};
use serde_json::Value;
use std::{collections::BTreeSet, path::Path};

pub const UPSTREAM_REVISION: &str = "60bfae688897dc41bc8685ca9f3130d7ed8f3b0c";
pub const FROZEN_PLANS: &str = include_str!("frozen-plans.json");
pub const TILE: usize = 128;
pub const FORMAT: &str = "miowtion-veda-predictor-v1";
pub const COMPONENT: &str = "veda_predictor";
pub const DIGEST: &str = "2a8d8845c5342756a2781e8e69563940e4bb573c9a40ebb534915ff8fd76573a";

fn bad(message: impl Into<String>) -> Error {
    Error::Msg(format!("Veda predictor: {}", message.into()))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TileShape(pub [usize; 3]);
impl TileShape {
    pub fn parse(s: &str) -> Result<Self> {
        let v: Vec<usize> = s
            .split('x')
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| bad("invalid tile shape"))?;
        if v.len() != 3
            || v.contains(&0)
            || v.iter().try_fold(1usize, |n, &v| n.checked_mul(v)) != Some(TILE)
        {
            return Err(bad("tile shape must hold 128 rows"));
        }
        Ok(Self([v[0], v[1], v[2]]))
    }
    pub fn tiles(self, g: [usize; 3]) -> usize {
        (0..3).map(|i| g[i].div_ceil(self.0[i])).product()
    }
    pub fn transpose(self) -> Self {
        Self([self.0[0], self.0[2], self.0[1]])
    }
}
#[derive(Clone, Debug)]
pub struct Plan {
    pub name: String,
    pub grid: [usize; 3],
    pub shapes: Vec<TileShape>,
    pub heads: Vec<Vec<usize>>,
    /// Frozen search/training provenance, retained without interpreting it as runtime controls.
    pub provenance: Option<Value>,
}
impl Plan {
    pub fn parse(v: &Value, layers: usize, heads: usize) -> Result<Self> {
        if !v.as_object().is_some_and(|m| {
            m.keys().all(|k| {
                ["geometry", "grid", "shapes", "head_shape", "provenance"].contains(&k.as_str())
            }) && ["geometry", "grid", "shapes", "head_shape"]
                .iter()
                .all(|k| m.contains_key(*k))
        }) {
            return Err(bad("invalid plan fields"));
        }
        let grid = v["grid"]
            .as_array()
            .ok_or_else(|| bad("missing plan grid"))?;
        let grid: Vec<usize> = grid
            .iter()
            .map(|x| {
                x.as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .filter(|&v| v > 0)
                    .ok_or_else(|| bad("invalid plan grid"))
            })
            .collect::<Result<_>>()?;
        if grid.len() != 3
            || grid
                .iter()
                .try_fold(1usize, |n, &v| n.checked_mul(v))
                .is_none_or(|n| n > i32::MAX as usize)
        {
            return Err(bad("plan grid must have three axes"));
        }
        let shapes = v["shapes"]
            .as_array()
            .ok_or_else(|| bad("missing shapes"))?
            .iter()
            .map(|s| TileShape::parse(s.as_str().unwrap_or("")))
            .collect::<Result<Vec<_>>>()?;
        let rows = v["head_shape"]
            .as_array()
            .ok_or_else(|| bad("missing head_shape"))?;
        if rows.len() != layers
            || shapes.is_empty()
            || shapes.iter().collect::<BTreeSet<_>>().len() != shapes.len()
        {
            return Err(bad("plan layer count mismatch"));
        }
        let mut hs = Vec::new();
        for row in rows {
            let row = row
                .as_array()
                .ok_or_else(|| bad("invalid head_shape row"))?;
            if row.len() != heads {
                return Err(bad("plan head count mismatch"));
            }
            hs.push(
                row.iter()
                    .map(|x| {
                        x.as_u64()
                            .and_then(|v| usize::try_from(v).ok())
                            .filter(|&v| v < shapes.len())
                            .ok_or_else(|| bad("plan shape index out of range"))
                    })
                    .collect::<Result<_>>()?,
            );
        }
        Ok(Self {
            name: v["geometry"]
                .as_str()
                .filter(|v| !v.is_empty())
                .ok_or_else(|| bad("missing geometry"))?
                .into(),
            grid: [grid[0], grid[1], grid[2]],
            shapes,
            heads: hs,
            provenance: v.get("provenance").cloned(),
        })
    }
    pub fn transposed(&self) -> Self {
        Self {
            name: format!("{}_T", self.name),
            grid: [self.grid[0], self.grid[2], self.grid[1]],
            shapes: self.shapes.iter().map(|s| s.transpose()).collect(),
            heads: self.heads.clone(),
            provenance: self.provenance.clone(),
        }
    }
}
pub struct Bundle {
    pub identity: String,
    pub layers: usize,
    pub heads: usize,
    pub dim: usize,
    pub plans: Vec<Plan>,
    pub q: Vec<Vec<u16>>,
    pub k: Vec<Vec<u16>>,
}
/// Round-to-nearest-even BF16, AFTER FP8 decode and per-head FP32 scaling.
pub fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
}
pub fn fp8(b: u8) -> f32 {
    let e = (b >> 3) & 15;
    let m = b & 7;
    let sign = if b & 128 == 0 { 1.0 } else { -1.0 };
    if e == 15 && m == 7 {
        f32::NAN
    } else if e == 0 {
        sign * (m as f32) * 2.0f32.powi(-9)
    } else {
        sign * (1.0 + m as f32 / 8.0) * 2.0f32.powi(e as i32 - 7)
    }
}
fn floats(dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>> {
    Ok(match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16))
            .collect(),
        Dtype::F8_E4M3 => bytes.iter().map(|&b| fp8(b)).collect(),
        _ => return Err(bad("unsupported storage dtype")),
    })
}
impl Bundle {
    pub fn load(path: &Path, layers: usize, heads: usize, dim: usize) -> Result<Self> {
        use sha2::{Digest, Sha256};
        // Hash and parse one owned buffer, so a replaced path cannot pass integrity validation.
        use std::io::Read;
        let file = std::fs::File::open(path).map_err(|e| {
            bad(format!(
                "{}: {e}; repair/install the managed component",
                path.display()
            ))
        })?;
        if file.metadata()?.len() != 275_415_648 {
            return Err(bad(
                "size mismatch; repair/install the frozen managed component",
            ));
        }
        let mut bytes = Vec::with_capacity(275_415_649);
        file.take(275_415_649).read_to_end(&mut bytes)?;
        if bytes.len() != 275_415_648 {
            return Err(bad(
                "size mismatch; repair/install the frozen managed component",
            ));
        }
        if format!("{:x}", Sha256::digest(&bytes)) != DIGEST {
            return Err(bad(
                "digest mismatch; repair/install the frozen managed component",
            ));
        }
        let meta = CheckpointMeta::from_bytes(bytes)?;
        let mut bundle = Self::from_meta(&meta, layers, heads, dim)?;
        if (layers, heads, dim) != (50, 56, 128)
            || serde_json::from_str::<Value>(meta.metadata("plans").unwrap())
                .map_err(|e| bad(e.to_string()))?
                != serde_json::from_str::<Value>(FROZEN_PLANS).unwrap()
        {
            return Err(bad("frozen model dimensions/complete plan table mismatch"));
        }
        bundle.identity = DIGEST.into();
        Ok(bundle)
    }
    pub fn from_meta(
        meta: &CheckpointMeta,
        layers: usize,
        heads: usize,
        dim: usize,
    ) -> Result<Self> {
        if layers == 0
            || heads == 0
            || dim == 0
            || dim
                .checked_mul(dim)
                .and_then(|n| n.checked_mul(3))
                .and_then(|n| n.checked_mul(heads))
                .is_none()
        {
            return Err(bad("invalid predictor dimensions"));
        }
        if meta.metadata("format") != Some(FORMAT) {
            return Err(bad("format mismatch"));
        }
        for (key, expected) in [
            ("num_layers", layers),
            ("num_heads", heads),
            ("head_dim", dim),
        ] {
            if meta.metadata(key).and_then(|s| s.parse::<usize>().ok()) != Some(expected) {
                return Err(bad(format!("{key} mismatch")));
            }
        }
        let keep = meta
            .metadata("keep_ratio")
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0 && *v <= 1.0)
            .ok_or_else(|| bad("invalid keep_ratio"))?;
        if (keep - 0.1).abs() > 1e-12 {
            return Err(bad("frozen predictor keep_ratio mismatch"));
        }
        let storage = match meta.metadata("dtype").unwrap_or("float32") {
            "float8_e4m3fn" => Dtype::F8_E4M3,
            "bfloat16" => Dtype::BF16,
            "float32" => Dtype::F32,
            _ => return Err(bad("unknown storage dtype")),
        };
        let pj: Value =
            serde_json::from_str(meta.metadata("plans").ok_or_else(|| bad("missing plans"))?)
                .map_err(|e| bad(format!("invalid plans JSON: {e}")))?;
        let plans = pj
            .as_object()
            .ok_or_else(|| bad("plans must be an object"))?
            .values()
            .map(|v| Plan::parse(v, layers, heads))
            .collect::<Result<Vec<_>>>()?;
        if plans.is_empty()
            || plans.iter().map(|p| &p.name).collect::<BTreeSet<_>>().len() != plans.len()
        {
            return Err(bad("empty plan table"));
        }
        let mut expected = BTreeSet::new();
        let mut q = Vec::new();
        let mut k = Vec::new();
        for layer in 0..layers {
            for (name, out) in [("proj_q", &mut q), ("proj_k", &mut k)] {
                let key = format!("layers.{layer}.{name}");
                expected.insert(key.clone());
                let t = meta
                    .tensor(&key)
                    .ok_or_else(|| bad(format!("missing {key}")))?;
                if t.shape != [heads, 3 * dim, dim] || t.dtype != storage {
                    return Err(bad(format!("{key} shape/dtype mismatch")));
                }
                let scale = if storage == Dtype::F8_E4M3 {
                    let sk = format!("{key}.__scale");
                    expected.insert(sk.clone());
                    let st = meta
                        .tensor(&sk)
                        .ok_or_else(|| bad("missing per-head scale"))?;
                    if st.shape != [heads] || st.dtype != Dtype::F32 {
                        return Err(bad("scale shape/dtype mismatch"));
                    }
                    floats(st.dtype, st.data)?
                } else {
                    vec![1.0; heads]
                };
                if scale.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                    return Err(bad("invalid scale"));
                }
                let values = floats(t.dtype, t.data)?;
                let mut rounded = Vec::with_capacity(values.len());
                for (i, v) in values.into_iter().enumerate() {
                    let v = v * scale[i / (3 * dim * dim)];
                    if !v.is_finite() {
                        return Err(bad("non-finite projection"));
                    }
                    let rounded_value = bf16(v);
                    if !f32::from_bits((rounded_value as u32) << 16).is_finite() {
                        return Err(bad("projection exceeds BF16 storage range"));
                    }
                    rounded.push(rounded_value);
                }
                out.push(rounded);
            }
        }
        if meta.keys().map(str::to_owned).collect::<BTreeSet<_>>() != expected {
            return Err(bad("unexpected tensor keys"));
        }
        Ok(Self {
            identity: "fixture-unverified".into(),
            layers,
            heads,
            dim,
            plans,
            q,
            k,
        })
    }
    pub fn choose(&self, grid: [usize; 3]) -> Plan {
        let cost = |p: &Plan| {
            (
                rounded_aspect_cost(
                    ((p.grid[2] as f64 / p.grid[1] as f64).ln()
                        - (grid[2] as f64 / grid[1] as f64).ln())
                    .abs(),
                ),
                p.grid[0].abs_diff(grid[0]),
                p.shapes.iter().map(|s| s.tiles(grid)).sum::<usize>(),
            )
        };
        self.plans
            .iter()
            .cloned()
            .chain(self.plans.iter().map(Plan::transposed))
            .min_by_key(cost)
            .unwrap()
    }
    pub fn projection(&self, layer: usize, heads: &[i32], query: bool) -> Vec<f32> {
        let data = if query {
            &self.q[layer]
        } else {
            &self.k[layer]
        };
        let n = 3 * self.dim * self.dim;
        heads
            .iter()
            .flat_map(|&h| {
                data[h as usize * n..(h as usize + 1) * n]
                    .iter()
                    .map(|&v| f32::from_bits((v as u32) << 16))
            })
            .collect()
    }
    pub fn host_bytes(&self) -> u64 {
        (self
            .q
            .iter()
            .chain(&self.k)
            .map(|v| v.len() * 2)
            .sum::<usize>()) as u64
    }
}

// Python round(distance, 3) rounds the original binary64 to the nearest
// decimal, ties to even. Multiplying by 1000 and rounding can move a midpoint
// (e.g. 1.2345), changing which temporal/padding tie-breaker gets considered.
fn rounded_aspect_cost(distance: f64) -> u64 {
    format!("{distance:.3}")
        .replace('.', "")
        .parse()
        .unwrap_or(u64::MAX)
}

/// One contiguous native T/H/W row-major span. References precede the target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TiledSpan {
    pub start: usize,
    pub grid: [usize; 3],
    pub shape: TileShape,
}
/// Frozen column budget. Absolute tile budgets may be fractional (Bresenham).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Budget {
    Ratio(f64),
    Tiles(f64),
}
impl Budget {
    pub fn sparsity(percent: f64) -> Result<Self> {
        if !percent.is_finite() || !(0.0..100.0).contains(&percent) {
            return Err(bad("sparsity must be in [0,100)"));
        }
        Ok(Self::Ratio(1.0 - percent / 100.0))
    }
    fn per_row(self, real_tokens: usize, cols: usize) -> Result<f64> {
        let v = match self {
            Self::Ratio(v) | Self::Tiles(v) => v,
        };
        if !v.is_finite() || v <= 0.0 {
            return Err(bad("budget must be finite and positive"));
        }
        Ok(match self {
            Self::Ratio(r) => {
                let n = real_tokens.div_ceil(TILE) as f64;
                r * n * n / cols as f64
            }
            Self::Tiles(t) => t,
        })
    }
}
fn split_budget(budget: f64, cols: usize) -> (usize, usize, f64) {
    let low = (budget.floor() as usize).clamp(1, cols);
    if budget >= cols as f64 {
        return (cols, cols, 0.0);
    }
    let fraction = (budget - low as f64).clamp(0.0, 1.0);
    // Python round(x, 12) rounds the original binary64 value. Scaling first
    // can erase which side of the decimal midpoint it lies on.
    let fraction = format!("{fraction:.12}").parse::<f64>().unwrap();
    (low, (low + 1).min(cols), fraction)
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub perm: Vec<i32>,
    pub valid: Vec<usize>,
    /// Tiled reference and generated rows; trailing rows are dense globals.
    pub video: usize,
    pub reference: usize,
    pub ref_tokens: usize,
    pub target_tokens: usize,
    pub seq: usize,
    pub inverse: Vec<i32>,
}
impl Layout {
    pub fn t2va(start: usize, grid: [usize; 3], shape: TileShape, seq: usize) -> Result<Self> {
        Self::from_spans(&[TiledSpan { start, grid, shape }], seq)
    }
    pub fn from_spans(spans: &[TiledSpan], seq: usize) -> Result<Self> {
        if spans.is_empty() || seq > i32::MAX as usize {
            return Err(bad("invalid span list/sequence"));
        }
        let mut perm = Vec::new();
        let mut valid = Vec::new();
        let mut reference = 0;
        let mut ref_tokens = 0;
        let mut stop = 0;
        let mut target_tokens = 0;
        let mut covered = vec![false; seq];
        for (i, s) in spans.iter().enumerate() {
            let count = s
                .grid
                .iter()
                .try_fold(1usize, |n, &v| n.checked_mul(v))
                .ok_or_else(|| bad("grid overflow"))?;
            let next = s
                .start
                .checked_add(count)
                .ok_or_else(|| bad("span overflow"))?;
            if s.grid.contains(&0)
                || s.shape.0.contains(&0)
                || s.shape.0.iter().try_fold(1usize, |n, &v| n.checked_mul(v)) != Some(TILE)
                || s.start < stop
                || next > seq
            {
                return Err(bad("invalid/overlapping/out-of-order span"));
            }
            covered[s.start..next].fill(true);
            stop = next;
            target_tokens = count;
            let [st, sh, sw] = s.shape.0;
            for hb in 0..s.grid[1].div_ceil(sh) {
                for wb in 0..s.grid[2].div_ceil(sw) {
                    for tb in 0..s.grid[0].div_ceil(st) {
                        let mut rows = Vec::with_capacity(TILE);
                        for t in tb * st..(tb + 1) * st {
                            for h in hb * sh..(hb + 1) * sh {
                                for w in wb * sw..(wb + 1) * sw {
                                    if t < s.grid[0] && h < s.grid[1] && w < s.grid[2] {
                                        rows.push(
                                            (s.start + (t * s.grid[1] + h) * s.grid[2] + w) as i32,
                                        );
                                    }
                                }
                            }
                        }
                        valid.push(rows.len());
                        rows.resize(TILE, -1);
                        perm.extend(rows);
                    }
                }
            }
            if i + 1 < spans.len() {
                reference = valid.len();
                ref_tokens += count;
            }
        }
        let video = valid.len();
        let globals: Vec<i32> = covered
            .iter()
            .enumerate()
            .filter(|(_, v)| !**v)
            .map(|(i, _)| i as i32)
            .collect();
        for rows in globals.chunks(TILE) {
            valid.push(rows.len());
            perm.extend(rows);
            perm.resize(perm.len() + TILE - rows.len(), -1);
        }
        if perm.len() > i32::MAX as usize {
            return Err(bad("padded sequence exceeds index range"));
        }
        let mut inverse = vec![0; seq];
        for (slot, &r) in perm.iter().enumerate() {
            if r >= 0 {
                inverse[r as usize] = slot as i32;
            }
        }
        Ok(Self {
            perm,
            valid,
            video,
            reference,
            ref_tokens,
            target_tokens,
            seq,
            inverse,
        })
    }
    pub fn select(&self, scores: &[f32], heads: usize, ratio: f64) -> Result<Vec<Vec<usize>>> {
        self.select_with_budgets(scores, heads, Budget::Ratio(ratio), Budget::Ratio(1.0))
    }
    /// Exact block-mask selection, shared by every backend. Head-major rows,
    /// ascending key IDs, including mandatory global queries/keys.
    pub fn select_with_budgets(
        &self,
        scores: &[f32],
        heads: usize,
        generated: Budget,
        reference: Budget,
    ) -> Result<Vec<Vec<usize>>> {
        let n = self.video;
        if heads == 0
            || n == 0
            || self.reference >= n
            || self.valid.iter().any(|&n| n > TILE)
            || scores.len()
                != heads
                    .checked_mul(n)
                    .and_then(|n1| n1.checked_mul(n))
                    .unwrap_or(0)
            || scores.iter().any(|v| !v.is_finite())
        {
            return Err(bad("invalid predictor scores/layout"));
        }
        let mut blocks = Vec::new();
        if self.reference > 0 {
            blocks.push((0, self.reference, reference, self.ref_tokens));
        }
        blocks.push((self.reference, n, generated, self.target_tokens));
        let mut splits = Vec::new();
        for &(start, stop, budget, tokens) in &blocks {
            let cols = stop - start;
            let b = budget.per_row(tokens, cols)?;
            let (low, hi, frac) = split_budget(b, cols);
            splits.push((start, stop, budget, low, hi, frac));
        }
        let mut rows = Vec::with_capacity(heads * self.valid.len());
        for h in 0..heads {
            for row in 0..self.valid.len() {
                if row >= n {
                    rows.push(
                        (0..self.valid.len())
                            .filter(|&k| self.valid[k] > 0)
                            .collect(),
                    );
                    continue;
                }
                let mut keys = Vec::new();
                for &(start, stop, budget, low, hi, frac) in &splits {
                    if matches!(budget,Budget::Ratio(r) if r>=1.0) {
                        keys.extend((start..stop).filter(|&k| self.valid[k] > 0));
                        continue;
                    }
                    let candidates = (start..stop)
                        .map(|k| {
                            let score = if k == row {
                                f32::INFINITY
                            } else if self.valid[k] == 0 {
                                f32::NEG_INFINITY
                            } else {
                                scores[(h * n + row) * n + k]
                            };
                            (score, k)
                        })
                        .collect();
                    let mut selected = super::topk::topk(candidates, hi);
                    // nth leaves the kth item last; fractional rows keep either k or k-1.
                    let count = low
                        + usize::from(
                            ((row + 1) as f64 * frac).floor() > (row as f64 * frac).floor(),
                        );
                    selected.truncate(count);
                    keys.extend(selected.into_iter().filter(|&k| self.valid[k] > 0));
                }
                keys.extend((n..self.valid.len()).filter(|&k| self.valid[k] > 0));
                keys.sort_unstable();
                rows.push(keys);
            }
        }
        Ok(rows)
    }
}
/// Infer a row-major reference grid exactly as frozen h3_layout._segment_grid.
/// Invalid/unsorted/duplicate position products stay global at the caller.
pub fn segment_grid(rows: &[[f64; 3]]) -> Option<[usize; 3]> {
    if rows.is_empty() || rows.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    let mut axes: [Vec<f64>; 3] = std::array::from_fn(|i| rows.iter().map(|r| r[i]).collect());
    for a in &mut axes {
        a.sort_by(f64::total_cmp);
        a.dedup();
    }
    let grid = axes.each_ref().map(Vec::len);
    if grid.iter().product::<usize>() != rows.len() {
        return None;
    }
    let mut row = 0;
    for &t in &axes[0] {
        for &h in &axes[1] {
            for &w in &axes[2] {
                if rows[row] != [t, h, w] {
                    return None;
                }
                row += 1;
            }
        }
    }
    Some(grid)
}
/// Least padding reference shape: fitting shapes, then cubic spread, then lexicographic.
pub fn least_padding_shape(grid: [usize; 3]) -> TileShape {
    let all: Vec<TileShape> = (0..8)
        .flat_map(|i| (0..8 - i).map(move |j| TileShape([1 << i, 1 << j, 1 << (7 - i - j)])))
        .collect();
    let fitting: Vec<_> = all
        .iter()
        .copied()
        .filter(|s| (0..3).all(|i| s.0[i] <= grid[i]))
        .collect();
    let candidates = if fitting.is_empty() { &all } else { &fitting };
    *candidates
        .iter()
        .min_by(|a, b| {
            a.tiles(grid)
                .cmp(&b.tiles(grid))
                .then_with(|| {
                    let spread = |s: &TileShape| {
                        *s.0.iter().max().unwrap() as f64 / *s.0.iter().min().unwrap() as f64
                    };
                    spread(a).total_cmp(&spread(b))
                })
                .then(a.cmp(b))
        })
        .unwrap()
}
/// Host TripPool over `[head,tile,128,dim]`; padding must be zero.
/// Backend accelerated pooling/projectors use this same contract and captured fixtures.
pub fn pool_video_tiles(x: &[f32], layout: &Layout, heads: usize, dim: usize) -> Result<Vec<f32>> {
    if dim == 0 || heads == 0 || x.len() != heads * layout.valid.len() * TILE * dim {
        return Err(bad("pool shape mismatch"));
    }
    let mut f = vec![0.0; heads * layout.video * 3 * dim];
    for h in 0..heads {
        for tile in 0..layout.video {
            let count = layout.valid[tile];
            if count == 0 {
                continue;
            }
            for d in 0..dim {
                let mut sum = 0.0f32;
                let mut max = f32::NEG_INFINITY;
                let mut min = f32::INFINITY;
                for row in 0..TILE {
                    let value = x[((h * layout.valid.len() + tile) * TILE + row) * dim + d];
                    if !value.is_finite() {
                        return Err(bad("non-finite pooled input"));
                    }
                    sum += value;
                    if row < count {
                        max = max.max(value);
                        min = min.min(value);
                    } else if value != 0.0 {
                        return Err(bad("nonzero padding"));
                    }
                }
                let base = (h * layout.video + tile) * 3 * dim + d;
                f[base] = sum / count as f32;
                f[base + dim] = max;
                f[base + 2 * dim] = min;
            }
        }
    }
    Ok(f)
}
/// Residual FP32 projection/logits. Projections have already passed BF16 storage rounding.
pub fn tile_logits(
    fq: &[f32],
    fk: &[f32],
    pq: &[f32],
    pk: &[f32],
    heads: usize,
    tiles: usize,
    dim: usize,
) -> Result<Vec<f32>> {
    if dim == 0
        || heads == 0
        || fq.len() != heads * tiles * 3 * dim
        || fk.len() != fq.len()
        || pq.len() != heads * 3 * dim * dim
        || pk.len() != pq.len()
        || fq
            .iter()
            .chain(fk)
            .chain(pq)
            .chain(pk)
            .any(|v| !v.is_finite())
    {
        return Err(bad("logit shape/value mismatch"));
    }
    let project = |f: &[f32], p: &[f32]| {
        let mut out = vec![0.0; heads * tiles * dim];
        for h in 0..heads {
            for tile in 0..tiles {
                for d in 0..dim {
                    let mut v = 0.0f32;
                    for i in 0..3 * dim {
                        v += f[(h * tiles + tile) * 3 * dim + i] * p[(h * 3 * dim + i) * dim + d];
                    }
                    out[(h * tiles + tile) * dim + d] = v + f[(h * tiles + tile) * 3 * dim + d];
                }
            }
        }
        out
    };
    let q = project(fq, pq);
    let k = project(fk, pk);
    let mut scores = vec![0.0; heads * tiles * tiles];
    for h in 0..heads {
        for r in 0..tiles {
            for c in 0..tiles {
                let mut v = 0.0f32;
                for d in 0..dim {
                    v += q[(h * tiles + r) * dim + d] * k[(h * tiles + c) * dim + d];
                }
                scores[(h * tiles + r) * tiles + c] = v / (dim as f32).sqrt();
            }
        }
    }
    Ok(scores)
}

/// Tensor-neutral primitive seam for an accelerated scorer. The shared driver
/// owns projection lookup/decode, residual mean addition and head-dimension
/// scaling; a backend supplies only pooling and tensor arithmetic primitives.
/// Implementations must pass the frozen scorer's exact-mask fixtures.
pub trait PredictorOps {
    type Tensor;
    type Error: From<Error>;
    fn pool(
        &self,
        x: &Self::Tensor,
        layout: &Layout,
        heads: usize,
        dim: usize,
    ) -> std::result::Result<Self::Tensor, Self::Error>;
    fn project(
        &self,
        features: &Self::Tensor,
        projection: &[f32],
        heads: usize,
        dim: usize,
    ) -> std::result::Result<Self::Tensor, Self::Error>;
    fn prefix(
        &self,
        features: &Self::Tensor,
        dim: usize,
    ) -> std::result::Result<Self::Tensor, Self::Error>;
    fn add(
        &self,
        a: &Self::Tensor,
        b: &Self::Tensor,
    ) -> std::result::Result<Self::Tensor, Self::Error>;
    fn dot_scaled(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        divisor: f32,
    ) -> std::result::Result<Vec<f32>, Self::Error>;
}
#[allow(clippy::too_many_arguments)]
pub fn learned_scores<M: PredictorOps>(
    bundle: &Bundle,
    layer: usize,
    heads: &[i32],
    layout: &Layout,
    q: &M::Tensor,
    k: &M::Tensor,
    ops: &M,
) -> std::result::Result<Vec<f32>, M::Error> {
    if layer >= bundle.layers
        || heads.is_empty()
        || heads.iter().any(|&h| h < 0 || h as usize >= bundle.heads)
    {
        return Err(bad("invalid scorer layer/heads").into());
    }
    let fq = ops.pool(q, layout, heads.len(), bundle.dim)?;
    let fk = ops.pool(k, layout, heads.len(), bundle.dim)?;
    let pq = ops.project(
        &fq,
        &bundle.projection(layer, heads, true),
        heads.len(),
        bundle.dim,
    )?;
    let pk = ops.project(
        &fk,
        &bundle.projection(layer, heads, false),
        heads.len(),
        bundle.dim,
    )?;
    let qh = ops.add(&pq, &ops.prefix(&fq, bundle.dim)?)?;
    let kh = ops.add(&pk, &ops.prefix(&fk, bundle.dim)?)?;
    ops.dot_scaled(&qh, &kh, (bundle.dim as f32).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_product_proof_grids_choose_all_twelve_original_plans_before_transposes() {
        let value: Value = serde_json::from_str(include_str!("frozen-plans.json")).unwrap();
        let bundle = Bundle {
            identity: "chooser_only_no_weights_or_device".into(),
            layers: 50,
            heads: 56,
            dim: 128,
            plans: value
                .as_object()
                .unwrap()
                .values()
                .map(|v| Plan::parse(v, 50, 56).unwrap())
                .collect(),
            q: vec![],
            k: vec![],
        };
        let mut observed = BTreeSet::new();
        for (grid_hw, name) in [
            ([10, 18], "16x9"),
            ([18, 10], "9x16"),
            ([24, 24], "1x1"),
            ([24, 32], "4x3"),
        ] {
            for t in [37, 72, 102] {
                let grid = [t, grid_hw[0], grid_hw[1]];
                let plan = bundle.choose(grid);
                assert_eq!(plan.name, format!("{name}_t{t}"));
                assert_eq!(plan.heads.len(), 50);
                assert!(plan.heads.iter().all(|h| h.len() == 56));
                println!(
                    "H3_NATIVE_CHOOSER grid={grid:?} plan={} layers={} heads={} exact={}",
                    plan.name,
                    plan.heads.len(),
                    plan.heads[0].len(),
                    plan.grid == grid
                );
                observed.insert(plan.name);
            }
        }
        assert_eq!(
            observed,
            value.as_object().unwrap().keys().cloned().collect()
        );
        assert_eq!(bundle.choose([42, 10, 18]).name, "16x9_t37");
    }
    struct HostOps {
        heads: usize,
        tiles: usize,
        dim: usize,
    }
    impl PredictorOps for HostOps {
        type Tensor = Vec<f32>;
        type Error = Error;
        fn pool(&self, x: &Vec<f32>, l: &Layout, h: usize, d: usize) -> Result<Vec<f32>> {
            pool_video_tiles(x, l, h, d)
        }
        fn project(&self, f: &Vec<f32>, p: &[f32], heads: usize, dim: usize) -> Result<Vec<f32>> {
            let mut out = vec![0.0; heads * self.tiles * dim];
            for h in 0..heads {
                for tile in 0..self.tiles {
                    for d in 0..dim {
                        for i in 0..3 * dim {
                            out[(h * self.tiles + tile) * dim + d] += f
                                [(h * self.tiles + tile) * 3 * dim + i]
                                * p[(h * 3 * dim + i) * dim + d];
                        }
                    }
                }
            }
            Ok(out)
        }
        fn prefix(&self, f: &Vec<f32>, dim: usize) -> Result<Vec<f32>> {
            Ok(f.chunks(3 * dim)
                .flat_map(|f| f[..dim].iter().copied())
                .collect())
        }
        fn add(&self, a: &Vec<f32>, b: &Vec<f32>) -> Result<Vec<f32>> {
            Ok(a.iter().zip(b).map(|(a, b)| a + b).collect())
        }
        fn dot_scaled(&self, q: &Vec<f32>, k: &Vec<f32>, divisor: f32) -> Result<Vec<f32>> {
            let mut out = vec![];
            for h in 0..self.heads {
                for r in 0..self.tiles {
                    for c in 0..self.tiles {
                        let mut v = 0.0;
                        for d in 0..self.dim {
                            v += q[(h * self.tiles + r) * self.dim + d]
                                * k[(h * self.tiles + c) * self.dim + d];
                        }
                        out.push(v / divisor);
                    }
                }
            }
            Ok(out)
        }
    }
    fn fixture() -> Value {
        serde_json::from_str(include_str!("../../tests/fixtures/veda/frozen.json")).unwrap()
    }
    fn us(v: &Value) -> usize {
        v.as_u64().unwrap() as usize
    }
    fn usize_vec(v: &Value) -> Vec<usize> {
        v.as_array().unwrap().iter().map(us).collect()
    }
    fn float_vec(v: &Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    }
    fn layout(v: &Value) -> Layout {
        let spans = v["spans"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| TiledSpan {
                start: us(&s["start"]),
                grid: usize_vec(&s["grid"]).try_into().unwrap(),
                shape: TileShape(usize_vec(&s["shape"]).try_into().unwrap()),
            })
            .collect::<Vec<_>>();
        Layout::from_spans(&spans, us(&v["seq"])).unwrap()
    }
    fn rows(v: &Value) -> Vec<Vec<usize>> {
        v.as_array().unwrap().iter().map(usize_vec).collect()
    }
    fn budget(v: &Value) -> Budget {
        if v[0] == "ratio" {
            Budget::Ratio(v[1].as_f64().unwrap())
        } else {
            Budget::Tiles(v[1].as_f64().unwrap())
        }
    }
    #[test]
    fn frozen_log_aspect_cost_keeps_python_decimal_rounding() {
        for c in fixture()["rounding_probes"].as_array().unwrap() {
            assert_eq!(
                rounded_aspect_cost(c["input"].as_f64().unwrap()),
                us(&c["cost"]) as u64
            );
        }
    }
    #[test]
    fn frozen_fractional_budget_keeps_python_decimal_rounding() {
        for c in fixture()["split_budget_probes"].as_array().unwrap() {
            let expected = &c["split"];
            assert_eq!(
                split_budget(c["budget"].as_f64().unwrap(), us(&c["cols"])),
                (
                    us(&expected[0]),
                    us(&expected[1]),
                    expected[2].as_f64().unwrap()
                )
            );
        }
    }
    #[test]
    fn frozen_all_plans_exact_nearest_transposed() {
        let pj: Value = serde_json::from_str(FROZEN_PLANS).unwrap();
        let plans = pj
            .as_object()
            .unwrap()
            .values()
            .map(|p| Plan::parse(p, 50, 56).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(plans.len(), 12);
        assert!(plans
            .iter()
            .all(|p| p.heads.len() == 50 && p.heads.iter().all(|h| h.len() == 56)));
        let bundle = Bundle {
            identity: String::new(),
            layers: 50,
            heads: 56,
            dim: 128,
            plans,
            q: vec![],
            k: vec![],
        };
        for c in fixture()["choices"].as_array().unwrap() {
            let grid = usize_vec(&c["grid"]).try_into().unwrap();
            let p = bundle.choose(grid);
            assert!(p.provenance.is_some());
            assert_eq!(p.name, c["name"].as_str().unwrap());
            assert_eq!(p.grid.to_vec(), usize_vec(&c["plan_grid"]));
            assert_eq!(p.grid == grid, c["exact"].as_bool().unwrap());
            assert_eq!(
                p.shapes
                    .iter()
                    .map(|s| s
                        .0
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join("x"))
                    .collect::<Vec<_>>(),
                c["shapes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_owned())
                    .collect::<Vec<_>>()
            );
        }
    }
    #[test]
    fn frozen_prefix_order_padding_reference_global() {
        for v in fixture()["layouts"].as_array().unwrap() {
            let l = layout(v);
            assert_eq!(
                l.perm,
                v["perm"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_i64().unwrap() as i32)
                    .collect::<Vec<_>>()
            );
            assert_eq!(l.valid, usize_vec(&v["valid"]));
            assert_eq!(l.reference, us(&v["reference"]));
            assert_eq!(l.ref_tokens, us(&v["ref_tokens"]));
            for (r, &slot) in l.inverse.iter().enumerate() {
                assert_eq!(l.perm[slot as usize], r as i32);
            }
            for (tile, valid) in l.perm.chunks(TILE).zip(&l.valid) {
                assert!(tile[..*valid].iter().all(|&n| n >= 0));
                assert!(tile[*valid..].iter().all(|&n| n == -1));
            }
        }
    }
    #[test]
    fn frozen_topk_cutoffs_preserve_tied_partition_and_heap_order() {
        for c in fixture()["topk_probes"].as_array().unwrap() {
            let mut scores = float_vec(&c["scores"]);
            scores[us(&c["own"])] = f32::INFINITY;
            let k = us(&c["k"]);
            let mut selected = super::super::topk::topk(
                scores
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| (v, i))
                    .collect(),
                k,
            );
            selected.truncate(k - 1);
            selected.sort_unstable();
            assert_eq!(
                selected,
                usize_vec(&c["prefix"]),
                "n={} k={} own={}",
                c["scores"].as_array().unwrap().len(),
                k,
                c["own"]
            );
        }
    }
    #[test]
    fn frozen_masks_independent_fractional_ties_near_ties_empty_keys() {
        for c in fixture()["selections"].as_array().unwrap() {
            let l = Layout {
                perm: vec![],
                valid: usize_vec(&c["valid"]),
                video: us(&c["video"]),
                reference: us(&c["reference"]),
                ref_tokens: us(&c["ref_tokens"]),
                target_tokens: us(&c["target_tokens"]),
                seq: 0,
                inverse: vec![],
            };
            let result = l
                .select_with_budgets(
                    &float_vec(&c["scores"]),
                    us(&c["heads"]),
                    budget(&c["generated"]),
                    budget(&c["reference_budget"]),
                )
                .unwrap();
            assert_eq!(result, rows(&c["rows"]), "{}", c["name"]);
            if l.valid[..l.video].iter().all(|&v| v > 0) {
                assert_eq!(
                    result,
                    rows(&c["upstream_rows"]),
                    "unaltered frozen mask {}",
                    c["name"]
                );
            }
            assert!(result.iter().flatten().all(|&k| l.valid[k] > 0));
        }
    }
    #[test]
    fn frozen_fp8_bf16_trip_pool_projection_scores_and_masks() {
        for c in fixture()["scorers"].as_array().unwrap() {
            let l = layout(&c["layout"]);
            let heads = us(&c["heads"]);
            let dim = us(&c["dim"]);
            let pq: Vec<f32> = c["fp8"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let decoded = bf16(
                        fp8(us(v) as u8)
                            * c["scales"][i / (3 * dim * dim)].as_f64().unwrap() as f32,
                    );
                    assert_eq!(decoded, c["proj"][i].as_i64().unwrap() as u16);
                    f32::from_bits((decoded as u32) << 16)
                })
                .collect();
            let mut pk = vec![0.0; pq.len()];
            for h in 0..heads {
                for i in 0..3 * dim {
                    for d in 0..dim {
                        pk[(h * 3 * dim + i) * dim + d] =
                            pq[(h * 3 * dim + 3 * dim - 1 - i) * dim + d];
                    }
                }
            }
            let fq = pool_video_tiles(&float_vec(&c["q"]), &l, heads, dim).unwrap();
            let fk = pool_video_tiles(&float_vec(&c["k"]), &l, heads, dim).unwrap();
            assert_eq!(fq, float_vec(&c["fq"]));
            assert_eq!(fk, float_vec(&c["fk"]));
            let scores = tile_logits(&fq, &fk, &pq, &pk, heads, l.video, dim).unwrap();
            let b = Bundle {
                identity: "fixture".into(),
                layers: 1,
                heads,
                dim,
                plans: vec![],
                q: vec![pq.iter().map(|v| bf16(*v)).collect()],
                k: vec![pk.iter().map(|v| bf16(*v)).collect()],
            };
            let shared = learned_scores(
                &b,
                0,
                &(0..heads as i32).collect::<Vec<_>>(),
                &l,
                &float_vec(&c["q"]),
                &float_vec(&c["k"]),
                &HostOps {
                    heads,
                    tiles: l.video,
                    dim,
                },
            )
            .unwrap();
            assert_eq!(shared, scores);
            for (a, b) in scores.iter().zip(float_vec(&c["logits"])) {
                assert!((*a - b).abs() < 0.00001);
            }
            // Output/logit tolerance NEVER replaces this exact selected-mask assertion.
            assert_eq!(
                l.select_with_budgets(&scores, heads, Budget::Tiles(2.3), Budget::Tiles(1.5))
                    .unwrap(),
                rows(&c["rows"])
            );
        }
    }
    fn meta_fixture(
        change: impl FnOnce(&mut serde_json::Map<String, Value>, &mut Vec<u8>),
    ) -> CheckpointMeta {
        use serde_json::json;
        let metadata = json!({"format":FORMAT,"num_layers":"1","num_heads":"2","head_dim":"2","keep_ratio":"0.1","dtype":"float8_e4m3fn","plans":json!({"tiny":{"geometry":"tiny","grid":[1,2,2],"shapes":["1x8x16"],"head_shape":[[0,0]]}}).to_string()});
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), metadata);
        let mut bytes = vec![];
        for name in ["proj_q", "proj_k"] {
            let key = format!("layers.0.{name}");
            let start = bytes.len();
            bytes.extend([0x18u8; 24]);
            header.insert(
                key.clone(),
                json!({"dtype":"F8_E4M3","shape":[2,6,2],"data_offsets":[start,bytes.len()]}),
            );
            let start = bytes.len();
            bytes.extend(0.03003f32.to_le_bytes());
            bytes.extend(0.07111f32.to_le_bytes());
            header.insert(
                format!("{key}.__scale"),
                json!({"dtype":"F32","shape":[2],"data_offsets":[start,bytes.len()]}),
            );
        }
        change(&mut header, &mut bytes);
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(bytes);
        CheckpointMeta::from_bytes(file).unwrap()
    }
    #[test]
    fn strict_bundle_metadata_projection_and_scales() {
        let m = meta_fixture(|_, _| {});
        let b = Bundle::from_meta(&m, 1, 2, 2).unwrap();
        assert_eq!(b.q[0].len(), 24);
        assert_ne!(b.q[0][0], b.q[0][12]);
        for key in [
            "format",
            "num_layers",
            "num_heads",
            "head_dim",
            "keep_ratio",
            "dtype",
            "plans",
        ] {
            let m = meta_fixture(|h, _| {
                h["__metadata__"]
                    .as_object_mut()
                    .unwrap()
                    .insert(key.into(), Value::String("invalid".into()));
            });
            assert!(Bundle::from_meta(&m, 1, 2, 2).is_err(), "{key}");
        }
        let m = meta_fixture(|_, bytes| bytes[24..28].copy_from_slice(&f32::NAN.to_le_bytes()));
        assert!(Bundle::from_meta(&m, 1, 2, 2).is_err());
        let m = meta_fixture(|_, bytes| bytes[0] = 0x7f);
        assert!(Bundle::from_meta(&m, 1, 2, 2).is_err());
        let m = meta_fixture(|h, _| {
            h["layers.0.proj_q.__scale"]["shape"] = serde_json::json!([1, 2]);
        });
        assert!(Bundle::from_meta(&m, 1, 2, 2).is_err());
        let m = meta_fixture(|h, _| {
            let t = h.remove("layers.0.proj_q").unwrap();
            h.insert("wrong".into(), t);
        });
        assert!(Bundle::from_meta(&m, 1, 2, 2).is_err());
        let mut p: Value = serde_json::from_str(FROZEN_PLANS).unwrap();
        let plan = p.as_object_mut().unwrap().values_mut().next().unwrap();
        for (key, value) in [
            ("grid", serde_json::json!([1, 0, 3])),
            ("shapes", serde_json::json!(["1x1x127"])),
            ("head_shape", serde_json::json!([[99]])),
        ] {
            let mut bad_plan = plan.clone();
            bad_plan[key] = value;
            assert!(Plan::parse(&bad_plan, 50, 56).is_err());
        }
    }
    #[test]
    fn reference_grid_and_boundaries_are_verified() {
        let rows = vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ];
        assert_eq!(segment_grid(&rows), Some([2, 1, 2]));
        let mut bad_rows = rows.clone();
        bad_rows.swap(1, 2);
        assert_eq!(segment_grid(&bad_rows), None);
        assert!(Layout::from_spans(
            &[
                TiledSpan {
                    start: 2,
                    grid: [1, 2, 2],
                    shape: TileShape([1, 8, 16])
                },
                TiledSpan {
                    start: 3,
                    grid: [1, 1, 1],
                    shape: TileShape([1, 8, 16])
                }
            ],
            10
        )
        .is_err());
        assert!(Layout::t2va(0, [usize::MAX, 2, 2], TileShape([1, 8, 16]), 10).is_err());
        assert!(Budget::sparsity(f64::NAN).is_err());
        assert!(Budget::sparsity(100.0).is_err());
        assert_eq!(fp8(0x7e), 448.0);
        assert_eq!(fp8(1), 2.0f32.powi(-9));
        assert!(fp8(0x7f).is_nan());
        assert_eq!(bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
    }
}
