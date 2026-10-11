//! Frozen predictor bundle and tile layout. No device work occurs in this module.
use mlx_gen::gen_core::weightsmeta::{CheckpointMeta, Dtype};
use mlx_gen::{Error, Result};
use serde_json::Value;
use std::{collections::BTreeSet, path::Path};

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
    fn parse(s: &str) -> Result<Self> {
        let v: Vec<usize> = s
            .split('x')
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| bad("invalid tile shape"))?;
        if v.len() != 3 || v.contains(&0) || v.iter().product::<usize>() != TILE {
            return Err(bad("tile shape must hold 128 rows"));
        }
        Ok(Self([v[0], v[1], v[2]]))
    }
    fn tiles(self, g: [usize; 3]) -> usize {
        (0..3).map(|i| g[i].div_ceil(self.0[i])).product()
    }
    fn transpose(self) -> Self {
        Self([self.0[0], self.0[2], self.0[1]])
    }
}
#[derive(Clone, Debug)]
pub struct Plan {
    pub name: String,
    pub grid: [usize; 3],
    pub shapes: Vec<TileShape>,
    pub heads: Vec<Vec<usize>>,
}
impl Plan {
    fn parse(v: &Value, layers: usize, heads: usize) -> Result<Self> {
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
        if grid.len() != 3 {
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
        if rows.len() != layers || shapes.is_empty() {
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
                            .map(|v| v as usize)
                            .filter(|&v| v < shapes.len())
                            .ok_or_else(|| bad("plan shape index out of range"))
                    })
                    .collect::<Result<_>>()?,
            );
        }
        Ok(Self {
            name: v["geometry"]
                .as_str()
                .ok_or_else(|| bad("missing geometry"))?
                .into(),
            grid: [grid[0], grid[1], grid[2]],
            shapes,
            heads: hs,
        })
    }
    fn transposed(&self) -> Self {
        Self {
            name: format!("{}_T", self.name),
            grid: [self.grid[0], self.grid[2], self.grid[1]],
            shapes: self.shapes.iter().map(|s| s.transpose()).collect(),
            heads: self.heads.clone(),
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
fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
}
fn fp8(b: u8) -> f32 {
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
        let bytes = std::fs::read(path).map_err(|e| {
            bad(format!(
                "{}: {e}; repair/install the managed component",
                path.display()
            ))
        })?;
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
        bundle.identity = DIGEST.into();
        Ok(bundle)
    }
    pub fn from_meta(
        meta: &CheckpointMeta,
        layers: usize,
        heads: usize,
        dim: usize,
    ) -> Result<Self> {
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
        if plans.is_empty() {
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
                    rounded.push(bf16(v));
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
                ((((p.grid[2] as f64 / p.grid[1] as f64).ln()
                    - (grid[2] as f64 / grid[1] as f64).ln())
                .abs()
                    * 1000.0)
                    .round() as u64),
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
#[derive(Clone, Debug)]
pub struct Layout {
    pub perm: Vec<i32>,
    pub valid: Vec<usize>,
    pub video: usize,
    pub target_tokens: usize,
    pub seq: usize,
    pub inverse: Vec<i32>,
}
impl Layout {
    /// Native T2VA generated rows are contiguous T/H/W row-major; every other packed row is global.
    pub fn t2va(start: usize, grid: [usize; 3], shape: TileShape, seq: usize) -> Result<Self> {
        let count = grid.iter().product::<usize>();
        if grid.contains(&0)
            || start.checked_add(count).is_none_or(|n| n > seq)
            || seq > i32::MAX as usize
        {
            return Err(bad("invalid target span"));
        }
        let mut perm = Vec::new();
        let mut valid = Vec::new();
        let [st, sh, sw] = shape.0;
        // Frozen permute(2,4,0,1,3,5): h-block, w-block, t-block; then t,h,w inside.
        for hb in 0..grid[1].div_ceil(sh) {
            for wb in 0..grid[2].div_ceil(sw) {
                for tb in 0..grid[0].div_ceil(st) {
                    let mut rows = Vec::new();
                    for t in tb * st..(tb + 1) * st {
                        for h in hb * sh..(hb + 1) * sh {
                            for w in wb * sw..(wb + 1) * sw {
                                if t < grid[0] && h < grid[1] && w < grid[2] {
                                    rows.push((start + (t * grid[1] + h) * grid[2] + w) as i32);
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
        let video = valid.len();
        let global: Vec<i32> = (0..seq)
            .filter(|&r| r < start || r >= start + count)
            .map(|r| r as i32)
            .collect();
        for rows in global.chunks(TILE) {
            valid.push(rows.len());
            perm.extend(rows);
            perm.resize(perm.len() + TILE - rows.len(), -1);
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
            target_tokens: count,
            seq,
            inverse,
        })
    }
    pub fn select(&self, scores: &[f32], heads: usize, ratio: f64) -> Result<Vec<Vec<usize>>> {
        let n = self.video;
        if scores.len() != heads * n * n || scores.iter().any(|v| !v.is_finite()) {
            return Err(bad("invalid predictor scores"));
        }
        let ideal = self.target_tokens.div_ceil(TILE);
        let budget = ratio * (ideal * ideal) as f64 / n as f64;
        let low = (budget.floor() as usize).clamp(1, n);
        let hi = (low + 1).min(n);
        let frac = ((budget - low as f64).clamp(0.0, 1.0) * 1e12).round() / 1e12;
        let mut rows = Vec::new();
        for h in 0..heads {
            for r in 0..self.valid.len() {
                if r >= n {
                    rows.push(
                        (0..self.valid.len())
                            .filter(|&k| self.valid[k] > 0)
                            .collect(),
                    );
                    continue;
                }
                let k = if budget >= n as f64 {
                    n
                } else if ((r + 1) as f64 * frac).floor() > (r as f64 * frac).floor() {
                    hi
                } else {
                    low
                };
                let mut candidates: Vec<usize> = (0..n).filter(|&k| self.valid[k] > 0).collect();
                // Explicit deterministic tie order. S2 frozen tied-score fixtures must establish upstream parity.
                candidates.sort_by(|&a, &b| {
                    if a == b {
                        std::cmp::Ordering::Equal
                    } else if a == r {
                        std::cmp::Ordering::Less
                    } else if b == r {
                        std::cmp::Ordering::Greater
                    } else {
                        scores[(h * n + r) * n + b]
                            .total_cmp(&scores[(h * n + r) * n + a])
                            .then(a.cmp(&b))
                    }
                });
                candidates.truncate(k);
                candidates.extend((n..self.valid.len()).filter(|&k| self.valid[k] > 0));
                candidates.sort_unstable();
                rows.push(candidates);
            }
        }
        Ok(rows)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fp8_and_bf16_order() {
        assert_eq!(fp8(0x7e), 448.0);
        assert_eq!(fp8(1), 2.0f32.powi(-9));
        assert!(fp8(0x7f).is_nan());
        assert_eq!(bf16(1.00390625), 0x3f80);
    }
    #[test]
    fn tile_order_padding_global_and_budget() {
        let l = Layout::t2va(3, [3, 5, 19], TileShape([2, 4, 16]), 291).unwrap();
        assert_eq!(&l.perm[..5], &[3, 4, 5, 6, 7]);
        assert_eq!(l.valid.iter().sum::<usize>(), 291);
        for (r, &slot) in l.inverse.iter().enumerate() {
            assert_eq!(l.perm[slot as usize], r as i32);
        }
        let selected = l.select(&vec![0.0; l.video * l.video], 1, 0.1).unwrap();
        for (r, keys) in selected.iter().enumerate() {
            assert!(keys.contains(&r) || l.valid[r] == 0);
            assert!(keys.iter().all(|&k| l.valid[k] > 0));
            if r < l.video {
                assert!(keys.contains(&l.video));
            } else {
                assert_eq!(keys.len(), l.valid.iter().filter(|&&c| c > 0).count());
            }
        }
    }
}
