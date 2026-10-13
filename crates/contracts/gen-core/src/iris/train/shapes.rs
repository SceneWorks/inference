//! Iris training **data shapes and batching** (story sc-25689): the shape policies that decide
//! what `(H, W)` a sample trains at, the aspect-ratio bucket tables, the per-rank ranged walk with
//! its world-portable resume cursor, and the shape-homogeneous batch grouping — upstream's
//! `data/shapes.py`, `data/buckets.py`, `data/samplers.py` and the `PixelDataset` transform of
//! `data/datasets.py` @ [`super::super::UPSTREAM_CODE_REVISION`], ported line for line.
//!
//! Everything here is host-side and tensor-free: the MLX and Candle trainers walk the **same**
//! plan, draw the same positional randomness for the same batch ordinal, and persist the same
//! resume cursor (the JSON object upstream stores as `data_position`, key for key), so a run
//! resumes — and reshards to another rank count — identically on both backends.
//!
//! Python semantics the port keeps exactly: `round()` is round-half-to-even; `min()` over a dict
//! returns the **first** minimal key in insertion order; the ranged walk leaves the tail shorter
//! than one canonical chunk unassigned; a shape-varying policy drops each shape group's partial
//! batch (`drop_last=True`) while the holdout walk keeps it.

use std::collections::BTreeMap;

use serde_json::{json, Value as JsonValue};

use crate::{Error, Result};

/// `samplers.CANONICAL_CHUNKS`: a rank owns a contiguous block of these chunks, so a checkpoint's
/// data position reconstructs at any world size that divides it.
pub const CANONICAL_CHUNKS: usize = 640;
const STATE_VERSION: u64 = 1;
const SHAPE_STATE_VERSION: u64 = 2;

/// Python's `round(x)` on a float: round half to even.
pub fn py_round(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && (r as i64) % 2 != 0 {
        (r - x.signum()) as i64
    } else {
        r as i64
    }
}

// =============================================================================================
// Shapes and policies (`data/shapes.py`, `data/buckets.py`)
// =============================================================================================

/// A concrete training shape in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Shape {
    pub height: usize,
    pub width: usize,
}

impl Shape {
    pub fn new(height: usize, width: usize) -> Result<Self> {
        if height == 0 || width == 0 {
            return Err(Error::Msg(format!(
                "iris data: shape must be positive, got {height}x{width}"
            )));
        }
        Ok(Self { height, width })
    }

    /// The `(rows, cols)` patch grid; refused when a side is not divisible by `patch`.
    pub fn grid(&self, patch: usize) -> Result<(usize, usize)> {
        if !self.height.is_multiple_of(patch) || !self.width.is_multiple_of(patch) {
            return Err(Error::Msg(format!(
                "iris data: {}x{} is not divisible by patch {patch}",
                self.height, self.width
            )));
        }
        Ok((self.height / patch, self.width / patch))
    }

    pub fn tokens(&self, patch: usize) -> Result<usize> {
        let (r, c) = self.grid(patch)?;
        Ok(r * c)
    }

    /// The grouping key (`"{height}x{width}"`).
    pub fn key(&self) -> String {
        format!("{}x{}", self.height, self.width)
    }
}

/// The `(height, width)` rows of `SHARED_21_512`, in upstream's dict insertion order.
pub const SHARED_21_512: [(usize, usize); 21] = [
    (256, 1024),
    (288, 928),
    (288, 896),
    (320, 832),
    (320, 800),
    (352, 736),
    (384, 672),
    (416, 640),
    (448, 576),
    (480, 544),
    (512, 512),
    (544, 480),
    (576, 448),
    (640, 416),
    (672, 384),
    (736, 352),
    (800, 320),
    (832, 320),
    (896, 288),
    (928, 288),
    (1024, 256),
];

/// The bucket table names of `TRAIN_BUCKETS`.
pub const BUCKET_TABLES: [&str; 2] = ["shared21-512", "shared21-1024"];

/// The `(height, width)` rows of a named bucket table (`shared21-1024` doubles every
/// `shared21-512` row, same order).
pub fn bucket_table(name: &str) -> Result<Vec<(usize, usize)>> {
    match name {
        "shared21-512" => Ok(SHARED_21_512.to_vec()),
        "shared21-1024" => Ok(SHARED_21_512.iter().map(|&(h, w)| (2 * h, 2 * w)).collect()),
        other => Err(Error::Msg(format!(
            "iris data: unknown bucket table '{other}' (known: {})",
            BUCKET_TABLES.join(", ")
        ))),
    }
}

/// `closest_ratio`: the table row whose `h / w` is nearest `height / width` — the **first** such
/// row in table order on a tie (Python's `min` over the dict keys).
pub fn closest_ratio(height: f64, width: f64, table: &[(usize, usize)]) -> (usize, usize) {
    let ratio = height / width;
    let mut best = table[0];
    let mut best_d = f64::INFINITY;
    for &(h, w) in table {
        let d = ((h as f64 / w as f64) - ratio).abs();
        if d < best_d {
            best_d = d;
            best = (h, w);
        }
    }
    best
}

/// `data.shape_policy`.
#[derive(Clone, Debug, PartialEq)]
pub enum ShapePolicy {
    /// One square side for every sample; shortest-side resize then center crop.
    Fixed { size: usize },
    /// Snap to the nearest row of a fixed aspect-ratio table.
    Bucket {
        table: String,
        rows: Vec<(usize, usize)>,
    },
    /// Table-free: the native ratio at `tokens` patches, both sides rounded to `align`, the
    /// long/short ratio clamped to `max_ratio`.
    Area {
        tokens: usize,
        patch: usize,
        align: usize,
        max_ratio: f64,
    },
}

impl ShapePolicy {
    /// `build_shape_policy`, with `DataConfig.validate`'s checks.
    pub fn build(
        policy: &str,
        image_size: usize,
        patch: usize,
        bucket_table_name: &str,
        align: usize,
        max_ratio: f64,
    ) -> Result<Self> {
        match policy {
            "fixed" | "area" if !image_size.is_multiple_of(patch) || image_size == 0 => {
                Err(Error::Msg(format!(
                    "iris data: data.image_size {image_size} is not a positive multiple of \
                     model.patch_size {patch}"
                )))
            }
            "fixed" => Ok(Self::Fixed { size: image_size }),
            "bucket" => {
                let rows = bucket_table(bucket_table_name)?;
                for &(h, w) in &rows {
                    Shape::new(h, w)?.grid(patch)?;
                }
                Ok(Self::Bucket {
                    table: bucket_table_name.to_string(),
                    rows,
                })
            }
            "area" => {
                if align == 0 || !align.is_multiple_of(patch) {
                    return Err(Error::Msg(format!(
                        "iris data: data.shape_align {align} is not a positive multiple of \
                         model.patch_size {patch}"
                    )));
                }
                if !(max_ratio.is_finite() && max_ratio >= 1.0) {
                    return Err(Error::Msg(format!(
                        "iris data: data.shape_max_ratio must be >= 1, got {max_ratio}"
                    )));
                }
                let side = image_size / patch;
                Ok(Self::Area {
                    tokens: side * side,
                    patch,
                    align,
                    max_ratio,
                })
            }
            other => Err(Error::Msg(format!(
                "iris data: unknown data.shape_policy '{other}' (fixed | bucket | area)"
            ))),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Fixed { .. } => "fixed",
            Self::Bucket { .. } => "bucket",
            Self::Area { .. } => "area",
        }
    }

    /// True when every sample resolves to the same shape (the walk skips grouping).
    pub fn uniform(&self) -> bool {
        matches!(self, Self::Fixed { .. })
    }

    /// The training shape of a sample of native size `height × width`.
    pub fn target(&self, height: usize, width: usize) -> Shape {
        match self {
            Self::Fixed { size } => Shape {
                height: *size,
                width: *size,
            },
            Self::Bucket { rows, .. } => {
                let (h, w) = closest_ratio(height as f64, width as f64, rows);
                Shape {
                    height: h,
                    width: w,
                }
            }
            Self::Area {
                tokens,
                patch,
                align,
                max_ratio,
            } => {
                let area = (tokens * patch * patch) as f64;
                let ratio = (height as f64 / width as f64).clamp(1.0 / max_ratio, *max_ratio);
                let raw_w = (area / ratio).sqrt();
                let raw_h = raw_w * ratio;
                let round = |v: f64| -> usize {
                    let r = py_round(v / *align as f64).max(0) as usize * align;
                    r.max(*align)
                };
                Shape {
                    height: round(raw_h),
                    width: round(raw_w),
                }
            }
        }
    }

    /// The grouping key of a sample.
    pub fn key(&self, height: usize, width: usize) -> String {
        self.target(height, width).key()
    }

    /// Every realizable shape, or `None` when the set is open (`area`).
    pub fn shapes(&self) -> Option<Vec<Shape>> {
        match self {
            Self::Fixed { size } => Some(vec![Shape {
                height: *size,
                width: *size,
            }]),
            Self::Bucket { rows, .. } => Some(
                rows.iter()
                    .map(|&(h, w)| Shape {
                        height: h,
                        width: w,
                    })
                    .collect(),
            ),
            Self::Area { .. } => None,
        }
    }

    /// The largest pixel count any sample can train at (the memory bound of a stage), when the
    /// shape set is closed.
    pub fn max_pixels(&self) -> Option<usize> {
        self.shapes()
            .map(|s| s.iter().map(|s| s.height * s.width).max().unwrap_or(0))
    }
}

/// One preprocessed training image: `[3, height, width]` f32 CHW in `[−1, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapedImage {
    pub height: usize,
    pub width: usize,
    pub chw: Vec<f32>,
}

/// `PixelDataset._get`'s resize for `policy` on an RGB8 buffer (`h · w · 3`): a uniform policy
/// resizes the shortest side to the square side (PIL bicubic) and center-crops it; any other
/// policy resizes so the target rectangle is covered (`_resize_covering`, PIL bicubic) and
/// center-crops it, and uses a sample already at its target size as is. Then `x / 127.5 − 1`.
pub fn preprocess_to_shape(
    rgb: &[u8],
    w: usize,
    h: usize,
    policy: &ShapePolicy,
) -> Result<ShapedImage> {
    let target = policy.target(h, w);
    if policy.uniform() {
        let img = super::preprocess_rgb(rgb, w, h, target.height)?;
        return Ok(ShapedImage {
            height: target.height,
            width: target.width,
            chw: img.chw,
        });
    }
    let (th, tw) = (target.height, target.width);
    let (src, sh, sw): (Vec<f32>, usize, usize) = if (h, w) == (th, tw) {
        (rgb.iter().map(|&v| v as f32).collect(), h, w)
    } else {
        // `_resize_covering(img, oh, ow, bh, bw)`.
        let (rh, rw) = if th as f64 / h as f64 > tw as f64 / w as f64 {
            (th, py_round(w as f64 * th as f64 / h as f64) as usize)
        } else {
            (py_round(h as f64 * tw as f64 / w as f64) as usize, tw)
        };
        (
            crate::imageops::resize_bicubic_u8(rgb, h, w, rh, rw)?,
            rh,
            rw,
        )
    };
    if sh < th || sw < tw {
        return Err(Error::Msg(format!(
            "iris data: a {sh}x{sw} resize cannot cover the {th}x{tw} crop"
        )));
    }
    // `_center_crop`: top/left = int(round((dim − crop) / 2)).
    let top = py_round((sh - th) as f64 / 2.0) as usize;
    let left = py_round((sw - tw) as f64 / 2.0) as usize;
    let mut chw = vec![0f32; 3 * th * tw];
    for y in 0..th {
        for x in 0..tw {
            let s = ((top + y) * sw + (left + x)) * 3;
            for c in 0..3 {
                chw[c * th * tw + y * tw + x] = src[s + c] / 127.5 - 1.0;
            }
        }
    }
    Ok(ShapedImage {
        height: th,
        width: tw,
        chw,
    })
}

/// [`preprocess_to_shape`] on an image file (decoded to RGB8).
pub fn preprocess_file_to_shape(
    path: &std::path::Path,
    policy: &ShapePolicy,
) -> Result<ShapedImage> {
    let img = image::open(path)
        .map_err(|e| Error::Msg(format!("iris training: decode {}: {e}", path.display())))?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    preprocess_to_shape(img.as_raw(), w, h, policy)
}

// =============================================================================================
// The ranged walk (`RangedSampler`)
// =============================================================================================

fn req_int(name: &str, v: Option<&JsonValue>, minimum: u64) -> Result<u64> {
    match v.and_then(JsonValue::as_u64) {
        Some(x) if x >= minimum => Ok(x),
        _ => Err(Error::Msg(format!(
            "iris data: {name} must be a {} integer, got {}",
            if minimum == 1 {
                "positive"
            } else {
                "nonnegative"
            },
            v.map(|v| v.to_string()).unwrap_or_else(|| "nothing".into())
        ))),
    }
}

type Ranges = Vec<(usize, usize)>;

/// `RangedSampler`: walk an ordered physical-range plan with an exact, world-portable resume.
#[derive(Clone, Debug, PartialEq)]
pub struct RangedSampler {
    pub num_samples: usize,
    pub rank: usize,
    pub world_size: usize,
    pub canonical_chunks: usize,
    pub chunk_size: usize,
    bounds: Vec<usize>,
    ranges_by_rank: Vec<Ranges>,
    progress: usize,
}

impl RangedSampler {
    pub fn new(num_samples: usize, rank: usize, world_size: usize) -> Result<Self> {
        Self::with_chunks(num_samples, rank, world_size, CANONICAL_CHUNKS)
    }

    pub fn with_chunks(
        num_samples: usize,
        rank: usize,
        world_size: usize,
        canonical_chunks: usize,
    ) -> Result<Self> {
        if world_size == 0 || canonical_chunks == 0 {
            return Err(Error::Msg(
                "iris data: world_size and canonical_chunks must be positive".into(),
            ));
        }
        if rank >= world_size {
            return Err(Error::Msg(format!(
                "iris data: rank {rank} out of range for world size {world_size}"
            )));
        }
        let chunks = canonical_chunks.min(num_samples);
        if chunks < world_size {
            return Err(Error::Msg(format!(
                "iris data: canonical_chunks={chunks} is below the world size ({world_size}); a \
                 rank must own at least one chunk"
            )));
        }
        let chunk_size = num_samples / chunks;
        let bounds = (0..=chunks).map(|k| k * chunk_size).collect();
        let mut s = Self {
            num_samples,
            rank,
            world_size,
            canonical_chunks: chunks,
            chunk_size,
            bounds,
            ranges_by_rank: Vec::new(),
            progress: 0,
        };
        s.ranges_by_rank = s.normal_ranges(world_size);
        Ok(s)
    }

    fn normal_ranges(&self, world_size: usize) -> Vec<Ranges> {
        let capacity = (self.canonical_chunks / world_size) * self.chunk_size;
        (0..world_size)
            .map(|r| vec![(r * capacity, (r + 1) * capacity)])
            .collect()
    }

    fn ranges_len(r: &[(usize, usize)]) -> usize {
        r.iter().map(|(lo, hi)| hi - lo).sum()
    }

    fn trim(ranges: &[(usize, usize)], mut skip: usize) -> Ranges {
        let mut out = Vec::new();
        for &(lo, hi) in ranges {
            let span = hi - lo;
            if skip >= span {
                skip -= span;
                continue;
            }
            out.push((lo + skip, hi));
            skip = 0;
        }
        out
    }

    fn split(ranges: &[(usize, usize)], world: usize, per_rank: usize) -> Vec<Ranges> {
        let mut plans: Vec<Ranges> = vec![Vec::new(); world];
        if per_rank == 0 {
            return plans;
        }
        let (mut target, mut left) = (0usize, per_rank);
        'outer: for &(plo, phi) in ranges {
            let mut lo = plo;
            while lo < phi {
                let take = (phi - lo).min(left);
                let hi = lo + take;
                let plan = &mut plans[target];
                match plan.last_mut() {
                    Some(last) if last.1 == lo => last.1 = hi,
                    _ => plan.push((lo, hi)),
                }
                lo = hi;
                left -= take;
                if left == 0 {
                    target += 1;
                    if target == world {
                        break 'outer;
                    }
                    left = per_rank;
                }
            }
        }
        plans
    }

    /// The covered prefix of the dataset (`chunks · chunk_size`); the tail is never walked.
    pub fn covered(&self) -> usize {
        self.bounds[self.bounds.len() - 1]
    }

    pub fn set_epoch(&mut self, _epoch: u32) {
        self.ranges_by_rank = self.normal_ranges(self.world_size);
        self.progress = 0;
    }

    pub fn set_start(&mut self, start: usize) -> Result<()> {
        let cap = Self::ranges_len(&self.ranges_by_rank[self.rank]);
        if start > cap {
            return Err(Error::Msg(format!(
                "iris data: start={start} exceeds this rank's capacity of {cap} samples"
            )));
        }
        self.progress = start;
        Ok(())
    }

    pub fn set_start_batches(&mut self, batches: usize, batch_size: usize) -> Result<()> {
        if batch_size == 0 {
            return Err(Error::Msg("iris data: batch_size must be positive".into()));
        }
        self.set_start(batches * batch_size)
    }

    fn require_divisible(&self, world: usize, label: &str) -> Result<()> {
        if !self.canonical_chunks.is_multiple_of(world) {
            return Err(Error::Msg(format!(
                "iris data: canonical_chunks={} must be divisible by the {label} world_size={world} \
                 for an exact resume",
                self.canonical_chunks
            )));
        }
        Ok(())
    }

    fn resume_layout(&self, world: usize) -> Result<JsonValue> {
        self.require_divisible(world, "prior")?;
        self.require_divisible(self.world_size, "current")?;
        Ok(json!({
            "canonical_chunks": self.canonical_chunks,
            "num_samples": self.num_samples,
            "world_size": world,
        }))
    }

    /// `resume_state`: this plan plus equal progress within every rank's virtual walk.
    pub fn resume_state(
        &self,
        batches_per_rank: usize,
        batch_size: usize,
        world_size: Option<usize>,
        samples_per_rank: Option<usize>,
    ) -> Result<JsonValue> {
        if batch_size == 0 {
            return Err(Error::Msg("iris data: batch_size must be positive".into()));
        }
        let prior = world_size.unwrap_or(self.world_size);
        let layout = self.resume_layout(prior)?;
        let plans = if prior == self.world_size {
            self.ranges_by_rank.clone()
        } else {
            self.normal_ranges(prior)
        };
        let caps: std::collections::BTreeSet<usize> =
            plans.iter().map(|r| Self::ranges_len(r)).collect();
        if caps.len() != 1 {
            return Err(Error::Msg(
                "iris data: resume plan must assign equal sample capacity to every rank".into(),
            ));
        }
        let capacity = *caps.iter().next().expect("one");
        let nominal = batches_per_rank * batch_size;
        let samples = match samples_per_rank {
            None => nominal,
            Some(s) => {
                let minimum = (nominal + 1).saturating_sub(batch_size);
                if !(minimum..=nominal).contains(&s) {
                    return Err(Error::Msg(format!(
                        "iris data: samples_per_rank={s} is incompatible with {batches_per_rank} \
                         batches of size {batch_size}; expected {minimum}..{nominal}"
                    )));
                }
                s
            }
        };
        if samples > capacity {
            return Err(Error::Msg(format!(
                "iris data: samples_per_rank={samples} exceeds the prior rank capacity of \
                 {capacity}"
            )));
        }
        let mut state = layout;
        let obj = state.as_object_mut().expect("object");
        obj.insert("version".into(), json!(STATE_VERSION));
        obj.insert("kind".into(), json!("contiguous_samples"));
        obj.insert(
            "ranges_by_rank".into(),
            JsonValue::Array(
                plans
                    .iter()
                    .map(|r| JsonValue::Array(r.iter().map(|&(lo, hi)| json!([lo, hi])).collect()))
                    .collect(),
            ),
        );
        obj.insert("samples_per_rank".into(), json!(samples));
        Ok(state)
    }

    fn validate_layout(&self, layout: &JsonValue) -> Result<usize> {
        let chunks = req_int("canonical_chunks", layout.get("canonical_chunks"), 1)? as usize;
        let num = req_int("num_samples", layout.get("num_samples"), 0)? as usize;
        let prior = req_int("world_size", layout.get("world_size"), 1)? as usize;
        if chunks != self.canonical_chunks {
            return Err(Error::Msg(format!(
                "iris resume: checkpoint used canonical_chunks={chunks} but this run uses {}; the \
                 partition is fixed for a run's lifetime",
                self.canonical_chunks
            )));
        }
        if num != self.num_samples {
            return Err(Error::Msg(format!(
                "iris resume: checkpoint indexed {num} samples but this dataset has {}; the \
                 dataset is fixed for a run's lifetime",
                self.num_samples
            )));
        }
        self.require_divisible(prior, "prior")?;
        self.require_divisible(self.world_size, "current")?;
        Ok(prior)
    }

    fn validate_plans(&self, raw: &JsonValue, prior: usize) -> Result<Vec<Ranges>> {
        let arr = raw.as_array().filter(|a| a.len() == prior).ok_or_else(|| {
            Error::Msg(format!(
                "iris resume: ranges_by_rank must contain exactly {prior} rank plans"
            ))
        })?;
        let covered_end = self.covered();
        let mut plans = Vec::with_capacity(prior);
        let mut all: Ranges = Vec::new();
        let mut caps = std::collections::BTreeSet::new();
        for (rank, rr) in arr.iter().enumerate() {
            let rr = rr.as_array().ok_or_else(|| {
                Error::Msg(format!(
                    "iris resume: ranges_by_rank[{rank}] must be a sequence"
                ))
            })?;
            let mut plan = Vec::new();
            let mut prev_hi: Option<usize> = None;
            for (pos, pair) in rr.iter().enumerate() {
                let p = pair.as_array().filter(|p| p.len() == 2).ok_or_else(|| {
                    Error::Msg(format!(
                        "iris resume: ranges_by_rank[{rank}][{pos}] must be a [lo, hi] pair"
                    ))
                })?;
                let lo = req_int("range lo", p.first(), 0)? as usize;
                let hi = req_int("range hi", p.get(1), 0)? as usize;
                if lo >= hi {
                    return Err(Error::Msg(format!(
                        "iris resume: ranges_by_rank[{rank}][{pos}] must be nonempty"
                    )));
                }
                if hi > covered_end {
                    return Err(Error::Msg(format!(
                        "iris resume: ranges_by_rank[{rank}][{pos}] exceeds the covered corpus \
                         bound {covered_end}"
                    )));
                }
                if prev_hi.is_some_and(|p| lo < p) {
                    return Err(Error::Msg(format!(
                        "iris resume: ranges_by_rank[{rank}] must be sorted and non-overlapping"
                    )));
                }
                plan.push((lo, hi));
                all.push((lo, hi));
                prev_hi = Some(hi);
            }
            caps.insert(Self::ranges_len(&plan));
            plans.push(plan);
        }
        if caps.len() != 1 {
            return Err(Error::Msg(
                "iris resume: ranges_by_rank must assign equal sample capacity to every rank"
                    .into(),
            ));
        }
        all.sort();
        let mut prev = None;
        for (lo, hi) in all {
            if prev.is_some_and(|p| lo < p) {
                return Err(Error::Msg(
                    "iris resume: ranges_by_rank contains globally overlapping intervals".into(),
                ));
            }
            prev = Some(hi);
        }
        Ok(plans)
    }

    /// `load_resume_state`: trim the equal source progress, then balance the exact unread plan
    /// over **this** run's world size (which may differ from the checkpoint's).
    pub fn load_resume_state(&mut self, state: &JsonValue) -> Result<()> {
        let kind = state.get("kind").and_then(JsonValue::as_str);
        if let Some(k) = kind {
            if k != "contiguous_samples" {
                return Err(Error::Msg(format!(
                    "iris resume: the ranged walk cannot restore resume state kind {k:?}"
                )));
            }
            let v = req_int("version", state.get("version"), 1)?;
            if v != STATE_VERSION {
                return Err(Error::Msg(format!(
                    "iris resume: unsupported sampler resume state version {v}"
                )));
            }
        }
        let prior = self.validate_layout(state)?;
        let samples = req_int("samples_per_rank", state.get("samples_per_rank"), 0)? as usize;
        let plans = match state.get("ranges_by_rank") {
            Some(r) => self.validate_plans(r, prior)?,
            None if kind.is_none() => self.normal_ranges(prior),
            None => {
                return Err(Error::Msg(
                    "iris resume: resume state is missing 'ranges_by_rank'".into(),
                ))
            }
        };
        let capacity = Self::ranges_len(&plans[0]);
        if samples > capacity {
            return Err(Error::Msg(format!(
                "iris resume: samples_per_rank={samples} exceeds the prior rank capacity of \
                 {capacity}"
            )));
        }
        let unread: Ranges = plans.iter().flat_map(|p| Self::trim(p, samples)).collect();
        let total_remaining = prior * (capacity - samples);
        if !total_remaining.is_multiple_of(self.world_size) {
            return Err(Error::Msg(format!(
                "iris resume: {total_remaining} remaining samples are not divisible by the \
                 current world_size={}; refusing unequal per-rank work",
                self.world_size
            )));
        }
        self.ranges_by_rank =
            Self::split(&unread, self.world_size, total_remaining / self.world_size);
        self.progress = 0;
        Ok(())
    }

    /// This rank's unread ranges.
    pub fn ranges(&self) -> Ranges {
        Self::trim(&self.ranges_by_rank[self.rank], self.progress)
    }

    pub fn len(&self) -> usize {
        Self::ranges_len(&self.ranges())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The indices this rank walks this pass, in order.
    pub fn indices(&self) -> Vec<usize> {
        self.ranges()
            .into_iter()
            .flat_map(|(lo, hi)| lo..hi)
            .collect()
    }
}

// =============================================================================================
// Shape-homogeneous batching (`ShapeBatchSampler`)
// =============================================================================================

/// `ShapeBatchSampler`: groups the ranged walk's indices into shape-homogeneous batches. An index
/// joins the group named by its key; a group is emitted once it holds `batch_size` indices.
/// `keys[i]` is the key of dataset index `i` (`None` = an unreadable sample, skipped). The epoch
/// plan is materialized once, so `len` and iteration cannot disagree.
#[derive(Clone, Debug)]
pub struct ShapeBatchSampler {
    pub sampler: RangedSampler,
    pub batch_size: usize,
    pub drop_last: bool,
    plan: Option<Vec<Vec<usize>>>,
    limit: Option<usize>,
    skip: usize,
    installed_skip: usize,
    history: Vec<(usize, usize)>,
    consumed_by_key: BTreeMap<String, usize>,
}

impl ShapeBatchSampler {
    pub fn new(sampler: RangedSampler, batch_size: usize, drop_last: bool) -> Result<Self> {
        if batch_size == 0 {
            return Err(Error::Msg("iris data: batch_size must be positive".into()));
        }
        Ok(Self {
            sampler,
            batch_size,
            drop_last,
            plan: None,
            limit: None,
            skip: 0,
            installed_skip: 0,
            history: Vec::new(),
            consumed_by_key: BTreeMap::new(),
        })
    }

    pub fn set_epoch(&mut self, epoch: u32) {
        self.sampler.set_epoch(epoch);
        self.plan = None;
        self.limit = None;
        self.skip = 0;
        self.installed_skip = 0;
        self.history.clear();
        self.consumed_by_key.clear();
    }

    /// The epoch's batches (materialized once per epoch / resume).
    pub fn plan(&mut self, keys: &[Option<String>]) -> &Vec<Vec<usize>> {
        if self.plan.is_none() {
            let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
            // Emission order follows the walk: a batch is emitted the moment its group fills.
            let mut batches: Vec<Vec<usize>> = Vec::new();
            let mut skip = self.consumed_by_key.clone();
            let mut order: Vec<String> = Vec::new();
            for idx in self.sampler.indices() {
                let Some(key) = keys.get(idx).cloned().flatten() else {
                    continue;
                };
                if let Some(s) = skip.get_mut(&key) {
                    if *s > 0 {
                        *s -= 1;
                        continue;
                    }
                }
                let g = groups.entry(key.clone()).or_insert_with(|| {
                    order.push(key.clone());
                    Vec::new()
                });
                g.push(idx);
                if g.len() == self.batch_size {
                    batches.push(std::mem::take(g));
                }
            }
            if !self.drop_last {
                // `groups.values()` in first-insertion order (Python dict order).
                for k in &order {
                    if let Some(g) = groups.get(k) {
                        if !g.is_empty() {
                            batches.push(g.clone());
                        }
                    }
                }
            }
            self.plan = Some(batches);
        }
        self.plan.as_ref().expect("planned")
    }

    fn capacity(&mut self, keys: &[Option<String>]) -> usize {
        let total = self.plan(keys).len();
        match self.limit {
            None => total,
            Some(l) => total.min(l),
        }
    }

    fn validate_skip(&mut self, batches: usize, keys: &[Option<String>]) -> Result<usize> {
        let cap = self.capacity(keys);
        if batches > cap {
            return Err(Error::Msg(format!(
                "iris data: batches_per_rank={batches} exceeds this rank's capacity of {cap} \
                 shape batches"
            )));
        }
        Ok(batches)
    }

    pub fn set_start_batches(
        &mut self,
        batches: usize,
        batch_size: usize,
        keys: &[Option<String>],
    ) -> Result<()> {
        if batch_size != self.batch_size {
            return Err(Error::Msg(format!(
                "iris data: resume batch_size={batch_size} does not match the shape sampler's \
                 batch_size={}",
                self.batch_size
            )));
        }
        self.installed_skip = 0;
        self.skip = self.validate_skip(batches, keys)?;
        Ok(())
    }

    /// `limit_batches`: truncate the upcoming pass to `limit` remaining batches.
    pub fn limit_batches(&mut self, limit: Option<usize>) {
        self.limit = limit.map(|l| self.skip + l);
    }

    /// Remaining batches of this pass.
    pub fn len(&mut self, keys: &[Option<String>]) -> usize {
        self.capacity(keys) - self.skip
    }

    pub fn is_empty(&mut self, keys: &[Option<String>]) -> bool {
        self.len(keys) == 0
    }

    /// This pass's batches (consumes the installed skip, as iterating upstream does).
    pub fn batches(&mut self, keys: &[Option<String>]) -> Vec<Vec<usize>> {
        let cap = self.capacity(keys);
        let skip = std::mem::take(&mut self.skip);
        self.plan(keys)[skip..cap].to_vec()
    }

    pub fn resume_state(
        &mut self,
        plan_local_batches: usize,
        batch_size: usize,
        samples_per_rank: Option<usize>,
        keys: &[Option<String>],
    ) -> Result<JsonValue> {
        if batch_size != self.batch_size {
            return Err(Error::Msg(format!(
                "iris data: resume batch_size={batch_size} does not match the shape sampler's \
                 batch_size={}",
                self.batch_size
            )));
        }
        let batches = self.validate_skip(self.installed_skip + plan_local_batches, keys)?;
        if let Some(s) = samples_per_rank {
            let expected = if self.drop_last {
                plan_local_batches * batch_size
            } else {
                let plan = self.plan(keys).clone();
                (self.installed_skip..batches).map(|i| plan[i].len()).sum()
            };
            if s != expected {
                return Err(Error::Msg(format!(
                    "iris data: shape-batched samples_per_rank does not match emitted batches: \
                     got {s}, expected {expected}"
                )));
            }
        }
        let layout = self.sampler.resume_layout(self.sampler.world_size)?;
        Ok(json!({
            "version": SHAPE_STATE_VERSION,
            "kind": "shape_batches",
            "batches_per_rank": batches,
            "batch_size": batch_size,
            "drop_last": self.drop_last,
            "batch_limit": self.limit,
            "regroup_history": self.history.iter().map(|&(size, count)| json!({
                "batch_size": size, "batches_per_rank": count
            })).collect::<Vec<_>>(),
            "layout": layout,
        }))
    }

    pub fn load_resume_state(&mut self, state: &JsonValue, keys: &[Option<String>]) -> Result<()> {
        if state.get("kind").and_then(JsonValue::as_str) != Some("shape_batches") {
            return Err(Error::Msg(
                "iris resume: shape-batched data needs resume state kind 'shape_batches'; a \
                 contiguous sample offset cannot restore grouped batches exactly"
                    .into(),
            ));
        }
        let version = req_int("version", state.get("version"), 1)?;
        if version != STATE_VERSION && version != SHAPE_STATE_VERSION {
            return Err(Error::Msg(format!(
                "iris resume: unsupported shape sampler resume state version {version}"
            )));
        }
        let layout = state
            .get("layout")
            .ok_or_else(|| Error::Msg("iris resume: resume state is missing 'layout'".into()))?;
        let mut batches = req_int("batches_per_rank", state.get("batches_per_rank"), 0)? as usize;
        let batch_size = req_int("batch_size", state.get("batch_size"), 1)? as usize;
        let prior = self.sampler.validate_layout(layout)?;
        if prior != self.sampler.world_size {
            return Err(Error::Msg(format!(
                "iris resume: shape-batched resume requires an unchanged world size; checkpoint \
                 world_size={prior}, current world_size={}",
                self.sampler.world_size
            )));
        }
        let mut history: Vec<(usize, usize)> = Vec::new();
        let mut limit = self.limit;
        if version == SHAPE_STATE_VERSION {
            let drop_last = state.get("drop_last").and_then(JsonValue::as_bool);
            if drop_last != Some(self.drop_last) {
                return Err(Error::Msg(
                    "iris resume: shape-batched resume requires unchanged drop_last".into(),
                ));
            }
            let raw = state
                .get("regroup_history")
                .and_then(JsonValue::as_array)
                .ok_or_else(|| Error::Msg("iris resume: regroup_history must be a list".into()))?;
            limit = match state.get("batch_limit") {
                None | Some(JsonValue::Null) => None,
                Some(v) => Some(req_int("batch_limit", Some(v), 0)? as usize),
            };
            for e in raw {
                let size = req_int("history batch_size", e.get("batch_size"), 1)? as usize;
                let count =
                    req_int("history batches_per_rank", e.get("batches_per_rank"), 0)? as usize;
                if count > 0 {
                    history.push((size, count));
                }
            }
            if limit.is_some_and(|l| batches > l) {
                return Err(Error::Msg(
                    "iris resume: batches_per_rank exceeds the checkpoint batch_limit".into(),
                ));
            }
        }
        if batch_size != self.batch_size {
            if batches > 0 {
                history.push((batch_size, batches));
            }
            batches = 0;
            limit = None;
        }
        self.sampler.set_start(0)?;
        let consumed = self.replay(&history, keys)?;
        self.history = history;
        self.consumed_by_key = consumed;
        self.plan = None;
        self.limit = limit;
        self.installed_skip = self.validate_skip(batches, keys)?;
        self.skip = self.installed_skip;
        Ok(())
    }

    /// Recover the consumed shape prefixes of a regroup history.
    fn replay(
        &self,
        history: &[(usize, usize)],
        keys: &[Option<String>],
    ) -> Result<BTreeMap<String, usize>> {
        let mut consumed: BTreeMap<String, usize> = BTreeMap::new();
        for &(size, count) in history {
            let mut skip = consumed.clone();
            let mut groups: BTreeMap<String, usize> = BTreeMap::new();
            let mut order: Vec<String> = Vec::new();
            let mut emitted = 0usize;
            let mut broke = false;
            for idx in self.sampler.indices() {
                let Some(key) = keys.get(idx).cloned().flatten() else {
                    continue;
                };
                if let Some(s) = skip.get_mut(&key) {
                    if *s > 0 {
                        *s -= 1;
                        continue;
                    }
                }
                if !groups.contains_key(&key) {
                    order.push(key.clone());
                }
                let pending = groups.get(&key).copied().unwrap_or(0) + 1;
                groups.insert(key.clone(), pending);
                if pending == size {
                    groups.insert(key.clone(), 0);
                    *consumed.entry(key).or_insert(0) += size;
                    emitted += 1;
                    if emitted == count {
                        broke = true;
                        break;
                    }
                }
            }
            if !broke && !self.drop_last {
                for key in &order {
                    let pending = groups.get(key).copied().unwrap_or(0);
                    if pending > 0 {
                        *consumed.entry(key.clone()).or_insert(0) += pending;
                        emitted += 1;
                        if emitted == count {
                            break;
                        }
                    }
                }
            }
            if emitted != count {
                return Err(Error::Msg(format!(
                    "iris resume: history batches_per_rank={count} exceeds this rank's capacity of \
                     {emitted} shape batches at batch_size={size}"
                )));
            }
        }
        Ok(consumed)
    }
}

/// `shape_policy_id`: the identity of everything that changes a sample's shape key.
pub fn shape_policy_id(
    policy: &str,
    image_size: usize,
    bucket: &str,
    align: usize,
    max_ratio: f64,
    patch: usize,
) -> String {
    format!(
        "{policy}:{image_size}:{bucket}:{align}:{}:{patch}",
        py_float_repr(max_ratio)
    )
}

/// Python's `repr(float)` (what an f-string prints): the shortest round-trip digits, always with a
/// fractional part or an exponent (`4.0`, `2.5`, `1e-05`, `1e+16`, `inf`, `nan`).
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    let a = x.abs();
    if a != 0.0 && !(1e-4..1e16).contains(&a) {
        // Rust `{:e}` is `1e-5` / `1.5e16`; Python writes a signed, two-digit exponent.
        let e = format!("{x:e}");
        let (mantissa, exp) = e.split_once('e').expect("exponent");
        let exp: i32 = exp.parse().expect("integer exponent");
        let sign = if exp < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:02}", exp.abs());
    }
    let s = format!("{x}");
    if s.contains('.') {
        s
    } else {
        format!("{s}.0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_policy_ids_print_floats_like_python() {
        // `builder.shape_policy_id` with the DataConfig defaults (shape_max_ratio: float = 4.0)
        assert_eq!(
            shape_policy_id("bucket", 512, "shared21-512", 32, 4.0, 16),
            "bucket:512:shared21-512:32:4.0:16"
        );
        for (x, py) in [
            (4.0, "4.0"),
            (2.5, "2.5"),
            (0.1, "0.1"),
            (-3.0, "-3.0"),
            (0.0, "0.0"),
            (1e-5, "1e-05"),
            (1.5e16, "1.5e+16"),
            (1e16, "1e+16"),
            (123456.0, "123456.0"),
            (f64::INFINITY, "inf"),
        ] {
            assert_eq!(py_float_repr(x), py, "{x}");
        }
    }

    #[test]
    fn python_round_is_half_to_even() {
        assert_eq!(py_round(0.5), 0);
        assert_eq!(py_round(1.5), 2);
        assert_eq!(py_round(2.5), 2);
        assert_eq!(py_round(-0.5), 0);
        assert_eq!(py_round(-1.5), -2);
        assert_eq!(py_round(2.4999), 2);
        assert_eq!(py_round(3.0), 3);
    }

    #[test]
    fn bucket_snaps_to_the_first_nearest_row() {
        let p = ShapePolicy::build("bucket", 512, 16, "shared21-512", 32, 4.0).unwrap();
        assert_eq!(p.target(1000, 1000), Shape::new(512, 512).unwrap());
        assert_eq!(p.target(300, 1200), Shape::new(256, 1024).unwrap());
        assert_eq!(p.target(2000, 500), Shape::new(1024, 256).unwrap());
        let p2 = ShapePolicy::build("bucket", 1024, 16, "shared21-1024", 32, 4.0).unwrap();
        assert_eq!(p2.target(1000, 1000), Shape::new(1024, 1024).unwrap());
        assert_eq!(p2.shapes().unwrap().len(), 21);
        assert!(ShapePolicy::build("bucket", 512, 16, "nope", 32, 4.0).is_err());
    }

    #[test]
    fn area_keeps_the_ratio_at_the_token_budget() {
        let p = ShapePolicy::build("area", 256, 16, "shared21-512", 32, 4.0).unwrap();
        // square: exactly the budget
        assert_eq!(p.target(500, 500), Shape::new(256, 256).unwrap());
        // 2:1 → sqrt(65536/2)=181.02 → 192 (6·32) wide; 362.04 → 352 tall
        assert_eq!(p.target(1000, 500), Shape::new(352, 192).unwrap());
        // clamp to 4:1
        assert_eq!(p.target(10000, 100), p.target(400, 100));
        assert!(ShapePolicy::build("area", 256, 16, "", 24, 4.0).is_err());
        assert!(ShapePolicy::build("area", 250, 16, "", 32, 4.0).is_err());
        assert!(p.shapes().is_none());
    }

    #[test]
    fn ranged_walk_partitions_and_drops_the_tail() {
        let s0 = RangedSampler::new(1300, 0, 2).unwrap();
        let s1 = RangedSampler::new(1300, 1, 2).unwrap();
        // 640 chunks of 2 → 1280 covered, 640 per rank.
        assert_eq!(s0.ranges(), vec![(0, 640)]);
        assert_eq!(s1.ranges(), vec![(640, 1280)]);
        assert_eq!(s0.covered(), 1280);
        let single = RangedSampler::new(10, 0, 1).unwrap();
        assert_eq!(single.indices(), (0..10).collect::<Vec<_>>());
        assert!(RangedSampler::new(1, 0, 2).is_err());
        assert!(RangedSampler::new(10, 2, 2).is_err());
    }

    #[test]
    fn ranged_resume_reshards_to_another_world() {
        // 2 ranks consumed 3 batches of 4 each; resume on 4 ranks.
        let s0 = RangedSampler::new(640, 0, 2).unwrap();
        let state = s0.resume_state(3, 4, None, None).unwrap();
        let mut seen = Vec::new();
        for r in 0..4 {
            let mut s = RangedSampler::new(640, r, 4).unwrap();
            s.load_resume_state(&state).unwrap();
            assert_eq!(s.len(), (640 - 2 * 12) / 4);
            seen.extend(s.indices());
        }
        seen.sort();
        let mut want: Vec<usize> = (12..320).chain(332..640).collect();
        want.sort();
        assert_eq!(seen, want);
        // an exact same-world resume continues where it stopped
        let mut s = RangedSampler::new(640, 1, 2).unwrap();
        s.load_resume_state(&state).unwrap();
        assert_eq!(s.indices().first(), Some(&332));
        // a world that does not divide the chunks is refused
        assert!(RangedSampler::new(640, 0, 3)
            .unwrap()
            .load_resume_state(&state)
            .is_err());
    }

    fn keys(n: usize) -> Vec<Option<String>> {
        (0..n)
            .map(|i| Some(if i % 3 == 0 { "a" } else { "b" }.to_string()))
            .collect()
    }

    #[test]
    fn shape_batches_group_and_resume_exactly() {
        let k = keys(30);
        let mut s = ShapeBatchSampler::new(RangedSampler::new(30, 0, 1).unwrap(), 2, true).unwrap();
        s.set_epoch(1);
        let all = s.batches(&k);
        // a: 0,3,6,…,27 (10) → 5 batches; b: 20 → 10 batches
        assert_eq!(all.len(), 15);
        assert_eq!(all[0], vec![1, 2]);
        assert_eq!(all[1], vec![0, 3]);
        // resume after 4 emitted batches
        let mut s2 =
            ShapeBatchSampler::new(RangedSampler::new(30, 0, 1).unwrap(), 2, true).unwrap();
        s2.set_epoch(1);
        s2.set_start_batches(0, 2, &k).unwrap();
        let state = s2.resume_state(4, 2, Some(8), &k).unwrap();
        let mut s3 =
            ShapeBatchSampler::new(RangedSampler::new(30, 0, 1).unwrap(), 2, true).unwrap();
        s3.set_epoch(1);
        s3.load_resume_state(&state, &k).unwrap();
        assert_eq!(s3.batches(&k), all[4..].to_vec());
        // a microbatch change regroups the unconsumed suffix of each shape
        let mut s4 =
            ShapeBatchSampler::new(RangedSampler::new(30, 0, 1).unwrap(), 3, true).unwrap();
        s4.set_epoch(1);
        s4.load_resume_state(&state, &k).unwrap();
        let rest: usize = s4.batches(&k).iter().map(Vec::len).sum();
        // the first 4 batches were [1,2] [0,3] [4,5] [7,8]: 2 a and 6 b consumed → 8 a and 14 b
        // left → 2 + 4 batches of 3
        assert_eq!(rest, 18);
    }

    #[test]
    fn holdout_keeps_partial_groups() {
        let k = keys(7);
        let mut s = ShapeBatchSampler::new(RangedSampler::new(7, 0, 1).unwrap(), 2, false).unwrap();
        s.set_epoch(0);
        let b = s.batches(&k);
        let n: usize = b.iter().map(Vec::len).sum();
        assert_eq!(n, 7);
    }

    #[test]
    fn covering_resize_and_crop_match_the_policy_shape() {
        let p = ShapePolicy::build("bucket", 512, 16, "shared21-512", 32, 4.0).unwrap();
        let (w, h) = (300usize, 600usize);
        let rgb = vec![128u8; w * h * 3];
        let img = preprocess_to_shape(&rgb, w, h, &p).unwrap();
        assert_eq!((img.height, img.width), (736, 352));
        assert_eq!(img.chw.len(), 3 * 736 * 352);
        assert!(img
            .chw
            .iter()
            .all(|v| (v - (128.0 / 127.5 - 1.0)).abs() < 1e-6));
        // already at its target: used as is
        let rgb2: Vec<u8> = (0..352 * 736 * 3).map(|i| (i % 251) as u8).collect();
        let img2 = preprocess_to_shape(&rgb2, 352, 736, &p).unwrap();
        assert_eq!(img2.chw[0], rgb2[0] as f32 / 127.5 - 1.0);
    }
}
