//! Request-scoped MLX Veda route. Dense remains the default and fallback.
//! Numerical and batching rules ported from Veda-on-ComfyUI at
//! `60bfae688897dc41bc8685ca9f3130d7ed8f3b0c`; upstream MIT notice: `LICENSE.upstream`.
pub mod contract;
use crate::denoise::PackedLayout;
use contract::{Budget, Bundle, Layout, Plan, PredictorOps, TileShape, TILE};
use mlx_gen::attention::{sdpa_bounded_bhsd, slice_axis, BoundedAttention};
use mlx_gen::gen_core::{VedaBudget, VedaExecutionReport, VedaRequest};
use mlx_gen::{Error, Result};
use mlx_rs::{
    ops::{concatenate_axis, r#where},
    Array, Dtype,
};
use std::{cell::RefCell, collections::BTreeMap, path::Path, rc::Rc};

pub type ReportSink = Rc<RefCell<VedaExecutionReport>>;
const COPY_BYTES: u64 = 256 * 1024 * 1024;
const GATHER_BYTES: u64 = 512 * 1024 * 1024;
pub const KERNEL: &str = "mlx-batched-gather-sdpa-v1";
pub const SELF_TEST_ATOL: f32 = 0.025;

pub struct VedaRuntime {
    bundle: Bundle,
    plan: Plan,
    layouts: BTreeMap<TileShape, Layout>,
    request: VedaRequest,
    report: ReportSink,
    failed: RefCell<bool>,
    #[cfg(test)]
    pub(crate) calls: RefCell<Vec<(usize, usize, VedaRequest)>>,
}
impl VedaRuntime {
    /// Called only for an enabled request, after the native packed layout is established.
    pub fn prepare(
        request: &VedaRequest,
        path: Option<&Path>,
        packed: &PackedLayout,
        layers: usize,
        heads: usize,
        dim: usize,
        report: ReportSink,
    ) -> Option<Self> {
        if !request.enabled {
            return None;
        }
        report.borrow_mut().predictor_load_attempts += 1;
        let result = (|| {
            request.validate(layers, usize::MAX).map_err(Error::Msg)?;
            let path = path.ok_or_else(|| {
                Error::Msg(
                    "Veda predictor component is missing; repair/install the managed component"
                        .into(),
                )
            })?;
            let bundle = Bundle::load(path, layers, heads, dim)?;
            Self::from_bundle(bundle, request, packed, report.clone(), device_self_test)
        })();
        match result {
            Ok(runtime) => Some(runtime),
            Err(e) => {
                report.borrow_mut().fallback_reason = Some(e.to_string());
                None
            }
        }
    }
    fn from_bundle(
        bundle: Bundle,
        request: &VedaRequest,
        packed: &PackedLayout,
        report: ReportSink,
        self_test: impl FnOnce() -> Result<()>,
    ) -> Result<Self> {
        let g = packed.geometry();
        // Infer patch grid from the actual native layout, never from output pixels.
        let rows = packed.rows_per_frame() as usize;
        let grid_h = (g.latent_height / 2) as usize;
        let grid_w = (g.latent_width / 2) as usize;
        if rows != grid_h * grid_w {
            return Err(Error::Msg("Veda target patch grid mismatch".into()));
        }
        let grid = [g.num_latent_frames as usize, grid_h, grid_w];
        let plan = bundle.choose(grid);
        let mut layouts = BTreeMap::new();
        for &shape in &plan.shapes {
            layouts.insert(shape, packed.veda_layout(shape)?);
        }
        // Actual-device check precedes sparse production calls. Failure leaves this request dense.
        {
            let mut r = report.borrow_mut();
            r.predictor = Some(bundle.identity.clone());
            r.plan = Some(plan.name.clone());
            r.exact_plan = plan.grid == grid;
            r.execution_grid = Some(grid);
            r.plan_grid = Some(plan.grid);
        }
        self_test()?;
        Ok(Self {
            bundle,
            plan,
            layouts,
            request: request.clone(),
            report,
            failed: RefCell::new(false),
            #[cfg(test)]
            calls: Default::default(),
        })
    }
    #[cfg(test)]
    pub(crate) fn prepare_fixture(
        request: &VedaRequest,
        packed: &PackedLayout,
        report: ReportSink,
        fail: bool,
    ) -> Option<Self> {
        report.borrow_mut().predictor_load_attempts += 1;
        let result = Self::from_bundle(
            crate::veda_fixture::predictor(),
            request,
            packed,
            report.clone(),
            || {
                if fail {
                    Err(Error::Msg(
                        "Veda actual-device self-test forced failure".into(),
                    ))
                } else {
                    device_self_test()
                }
            },
        );
        match result {
            Ok(v) => Some(v),
            Err(e) => {
                report.borrow_mut().fallback_reason = Some(e.to_string());
                None
            }
        }
    }
    pub fn attention(
        &self,
        layer: usize,
        step: usize,
        q: &Array,
        k: &Array,
        v: &Array,
        bounded: BoundedAttention<'_>,
    ) -> Result<Array> {
        #[cfg(test)]
        self.calls
            .borrow_mut()
            .push((layer, step, self.request.clone()));
        if *self.failed.borrow() || self.request.full_attention(layer, step) {
            self.record_dense(q);
            return sdpa_bounded_bhsd(
                q,
                k,
                v,
                1.0 / (self.bundle.dim as f32).sqrt(),
                None,
                bounded,
            );
        }
        let result = self.execute(layer, q, k, v);
        match result {
            Ok(out) => {
                let mut report = self.report.borrow_mut();
                report.sparse_calls += 1;
                report.kernel = Some(KERNEL.into());
                Ok(out)
            }
            Err(e) => {
                *self.failed.borrow_mut() = true;
                let mut r = self.report.borrow_mut();
                r.fallback_reason = Some(format!(
                    "Veda execution failed: {e}; dense attention executed"
                ));
                drop(r);
                self.record_dense(q);
                sdpa_bounded_bhsd(
                    q,
                    k,
                    v,
                    1.0 / (self.bundle.dim as f32).sqrt(),
                    None,
                    bounded,
                )
            }
        }
    }
    fn record_dense(&self, q: &Array) {
        let mut r = self.report.borrow_mut();
        r.dense_calls += 1;
        let shape = q.shape();
        let pairs = shape[0] as u64 * shape[1] as u64 * shape[2] as u64 * shape[2] as u64;
        r.selected_token_pairs += pairs;
        r.dense_token_pairs += pairs;
    }

    fn execute(&self, layer: usize, q: &Array, k: &Array, v: &Array) -> Result<Array> {
        if q.shape().len() != 4
            || q.shape()[0] != 1
            || q.shape()[1] != self.bundle.heads as i32
            || q.shape()[3] != self.bundle.dim as i32
            || q.shape() != k.shape()
            || q.shape() != v.shape()
            || layer >= self.bundle.layers
        {
            return Err(Error::Msg("Veda QKV shape/layer mismatch".into()));
        }
        let dim = self.bundle.dim;
        let seq = q.shape()[2] as usize;
        let bytes = match q.dtype() {
            Dtype::Bfloat16 | Dtype::Float16 => 2,
            Dtype::Float32 => 4,
            _ => return Err(Error::Msg("Veda unsupported compute dtype".into())),
        };
        let mut grouped: BTreeMap<TileShape, Vec<i32>> = BTreeMap::new();
        for (h, &si) in self.plan.heads[layer].iter().enumerate() {
            grouped
                .entry(self.plan.shapes[si])
                .or_default()
                .push(h as i32);
        }
        let mut pieces = Vec::new();
        let mut head_order: Vec<i32> = Vec::new();
        let mut pairs = 0u64;
        let mut high = 0u64;
        for (shape, heads) in grouped {
            let layout = &self.layouts[&shape];
            // Bound the copies AND scorer before constructing any gathered tensor.
            let slots = layout.perm.len();
            let per_head = 3 * slots as u64 * dim as u64 * bytes
                + 2 * (layout.video * 3 * dim * 4) as u64
                + (layout.video * layout.video * 4) as u64
                + (2 * 3 * dim * dim * 4) as u64;
            let max_heads = (COPY_BYTES / per_head).min(heads.len() as u64) as usize;
            if max_heads == 0 {
                return Err(Error::Msg(
                    "Veda minimum tile-copy/scorer head exceeds workspace allowance".into(),
                ));
            }
            for chunk in heads.chunks(max_heads) {
                let n = chunk.len();
                let index = Array::from_slice(chunk, &[n as i32]);
                let perm: Vec<i32> = layout.perm.iter().map(|&r| r.max(0)).collect();
                let perm = Array::from_slice(&perm, &[slots as i32]);
                let valid: Vec<bool> = layout.perm.iter().map(|&r| r >= 0).collect();
                let valid = Array::from_slice(&valid, &[1, slots as i32, 1]);
                let tile = |a: &Array| -> Result<Array> {
                    let a = a
                        .take_axis(&index, 1)?
                        .reshape(&[n as i32, seq as i32, dim as i32])?
                        .take_axis(&perm, 1)?;
                    Ok(
                        r#where(&valid, &a, &Array::from_f32(0.0).as_dtype(a.dtype())?)?.reshape(
                            &[n as i32, layout.valid.len() as i32, TILE as i32, dim as i32],
                        )?,
                    )
                };
                let (qt, kt, vt) = (tile(q)?, tile(k)?, tile(v)?);
                let scores = learned_scores(&self.bundle, layer, chunk, layout, &qt, &kt)?;
                self.report.borrow_mut().predictor_score_calls += 1;
                let budget = |b: &VedaBudget| match b {
                    VedaBudget::SparsityPercent(p) => Budget::Ratio(1.0 - p / 100.0),
                    VedaBudget::KeyTiles(n) => Budget::Tiles(*n as f64),
                };
                let rows = layout.select_with_budgets(
                    &scores,
                    n,
                    budget(&self.request.generated),
                    budget(&self.request.reference),
                )?;
                let chunk_pairs: u64 = rows
                    .iter()
                    .enumerate()
                    .map(|(r, keys)| {
                        layout.valid[r % layout.valid.len()] as u64
                            * keys.iter().map(|&k| layout.valid[k] as u64).sum::<u64>()
                    })
                    .sum();
                pairs += chunk_pairs;
                let (out, scratch) = gather_attention(layout, &rows, &qt, &kt, &vt, dim, bytes)?;
                high = high.max(
                    per_head * n as u64
                        + scratch
                        + 2 * self.bundle.heads as u64 * seq as u64 * dim as u64 * bytes
                        + self.bundle.host_bytes(),
                );
                pieces.push(out.reshape(&[1, n as i32, seq as i32, dim as i32])?);
                head_order.extend(chunk);
            }
        }
        let refs: Vec<&Array> = pieces.iter().collect();
        let merged = concatenate_axis(&refs, 1)?;
        let mut inverse = vec![0i32; self.bundle.heads];
        for (i, &h) in head_order.iter().enumerate() {
            inverse[h as usize] = i as i32;
        }
        let out = merged.take_axis(Array::from_slice(&inverse, &[self.bundle.heads as i32]), 1)?;
        mlx_rs::transforms::eval([&out])?;
        let mut report = self.report.borrow_mut();
        report.selected_token_pairs += pairs;
        report.dense_token_pairs += self.bundle.heads as u64 * seq as u64 * seq as u64;
        report.workspace_bytes = report.workspace_bytes.max(high);
        Ok(out)
    }
}
fn learned_scores(
    bundle: &Bundle,
    layer: usize,
    heads: &[i32],
    layout: &Layout,
    q: &Array,
    k: &Array,
) -> Result<Vec<f32>> {
    contract::learned_scores(bundle, layer, heads, layout, q, k, &MlxPredictorOps)
}
struct MlxPredictorOps;
impl PredictorOps for MlxPredictorOps {
    type Tensor = Array;
    type Error = Error;
    fn pool(&self, a: &Array, layout: &Layout, _heads: usize, _dim: usize) -> Result<Array> {
        let n = layout.video as i32;
        let valid: Vec<bool> = layout.valid[..layout.video]
            .iter()
            .flat_map(|&c| (0..TILE).map(move |r| r < c))
            .collect();
        let valid = Array::from_slice(&valid, &[1, n, TILE as i32, 1]);
        let counts: Vec<f32> = layout.valid[..layout.video]
            .iter()
            .map(|&c| c.max(1) as f32)
            .collect();
        let counts = Array::from_slice(&counts, &[1, n, 1]);
        let ok: Vec<bool> = layout.valid[..layout.video]
            .iter()
            .map(|&c| c > 0)
            .collect();
        let ok = Array::from_slice(&ok, &[1, n, 1]);
        let a = slice_axis(a, 1, 0, n)?.as_dtype(Dtype::Float32)?;
        let mean = a.sum_axis(2, false)?.divide(&counts)?;
        let max = r#where(&valid, &a, Array::from_f32(f32::NEG_INFINITY))?.max_axis(2, false)?;
        let min = r#where(&valid, &a, Array::from_f32(f32::INFINITY))?.min_axis(2, false)?;
        let f = concatenate_axis(&[&mean, &max, &min], -1)?;
        Ok(r#where(&ok, &f, Array::from_f32(0.0))?)
    }
    fn project(&self, f: &Array, p: &[f32], heads: usize, dim: usize) -> Result<Array> {
        Ok(f.matmul(Array::from_slice(
            p,
            &[heads as i32, (3 * dim) as i32, dim as i32],
        ))?)
    }
    fn prefix(&self, f: &Array, dim: usize) -> Result<Array> {
        slice_axis(f, 2, 0, dim as i32)
    }
    fn add(&self, a: &Array, b: &Array) -> Result<Array> {
        Ok(a.add(b)?)
    }
    fn dot_scaled(&self, q: &Array, k: &Array, divisor: f32) -> Result<Vec<f32>> {
        let scores = q
            .matmul(&k.transpose_axes(&[0, 2, 1])?)?
            .divide(Array::from_f32(divisor))?
            .reshape(&[-1])?;
        // Logical flatten, because physical MLX result strides need not be row-major.
        mlx_rs::transforms::eval([&scores])?;
        Ok(scores.as_slice::<f32>().to_vec())
    }
}

fn gather_attention(
    l: &Layout,
    selected: &[Vec<usize>],
    q: &Array,
    k: &Array,
    v: &Array,
    dim: usize,
    bytes: u64,
) -> Result<(Array, u64)> {
    let h = q.shape()[0] as usize;
    let n = l.valid.len();
    let nv = l.video;
    let kmax = (0..h)
        .flat_map(|head| selected[head * n..head * n + nv].iter().map(Vec::len))
        .max()
        .unwrap_or(0);
    if kmax == 0 {
        return Err(Error::Msg("Veda empty selection".into()));
    }
    let per_row = h as u64 * kmax as u64 * TILE as u64 * (2 * dim as u64 * bytes + TILE as u64 * 4);
    let rmax = (GATHER_BYTES / per_row).min(nv as u64) as usize;
    if rmax == 0 {
        return Err(Error::Msg(
            "Veda minimum sparse batch exceeds gather allowance".into(),
        ));
    }
    let flatk = k.reshape(&[(h * n) as i32, TILE as i32, dim as i32])?;
    let flatv = v.reshape(&[(h * n) as i32, TILE as i32, dim as i32])?;
    let mut outputs = Vec::new();
    for start in (0..nv).step_by(rmax) {
        let rows = rmax.min(nv - start);
        let mut indices = Vec::with_capacity(h * rows * kmax);
        let mut valid = Vec::with_capacity(h * rows * kmax * TILE);
        for head in 0..h {
            for row in start..start + rows {
                let keys = &selected[head * n + row];
                for i in 0..kmax {
                    let tile = keys.get(i).copied().unwrap_or(0);
                    indices.push((head * n + tile) as i32);
                    valid.extend((0..TILE).map(|s| i < keys.len() && s < l.valid[tile]));
                }
            }
        }
        let idx = Array::from_slice(&indices, &[indices.len() as i32]);
        let shape = [(h * rows) as i32, 1, (kmax * TILE) as i32, dim as i32];
        let keys = flatk.take_axis(&idx, 0)?.reshape(&shape)?;
        let values = flatv.take_axis(&idx, 0)?.reshape(&shape)?;
        let query = slice_axis(q, 1, start as i32, rows as i32)?.reshape(&[
            (h * rows) as i32,
            1,
            TILE as i32,
            dim as i32,
        ])?;
        let mask = Array::from_slice(&valid, &[(h * rows) as i32, 1, 1, (kmax * TILE) as i32]);
        let out = sdpa_bounded_bhsd(
            &query,
            &keys,
            &values,
            1.0 / (dim as f32).sqrt(),
            Some(&mask),
            BoundedAttention::UNBOUNDED,
        )?
        .reshape(&[h as i32, (rows * TILE) as i32, dim as i32])?;
        mlx_rs::transforms::eval([&out])?;
        outputs.push(out);
    }
    let global_per_row = h as u64 * TILE as u64 * l.perm.len() as u64 * 4;
    let global_rows = (GATHER_BYTES / global_per_row).min((n - nv) as u64) as usize;
    if n > nv && global_rows == 0 {
        return Err(Error::Msg(
            "Veda minimum global batch exceeds workspace allowance".into(),
        ));
    }
    if n > nv {
        let keys = k.reshape(&[h as i32, 1, l.perm.len() as i32, dim as i32])?;
        let values = v.reshape(&[h as i32, 1, l.perm.len() as i32, dim as i32])?;
        let valid: Vec<bool> = l.perm.iter().map(|&r| r >= 0).collect();
        let mask = Array::from_slice(&valid, &[1, 1, 1, l.perm.len() as i32]);
        for start in (nv..n).step_by(global_rows) {
            let rows = global_rows.min(n - start);
            let query = slice_axis(q, 1, start as i32, rows as i32)?.reshape(&[
                h as i32,
                1,
                (rows * TILE) as i32,
                dim as i32,
            ])?;
            let out = sdpa_bounded_bhsd(
                &query,
                &keys,
                &values,
                1.0 / (dim as f32).sqrt(),
                Some(&mask),
                BoundedAttention::UNBOUNDED,
            )?
            .reshape(&[h as i32, (rows * TILE) as i32, dim as i32])?;
            mlx_rs::transforms::eval([&out])?;
            outputs.push(out);
        }
    }
    let refs: Vec<&Array> = outputs.iter().collect();
    let tiled = concatenate_axis(&refs, 1)?;
    let out = tiled.take_axis(Array::from_slice(&l.inverse, &[l.seq as i32]), 1)?;
    mlx_rs::transforms::eval([&out])?;
    let scratch = (rmax as u64 * per_row).max(global_rows as u64 * global_per_row)
        + (l.perm.len() * dim * h) as u64 * bytes * 2;
    Ok((out, scratch))
}
/// Bounded actual-device oracle, including partial video tiles, global rows, and a non-dense mask.
pub fn device_self_test() -> Result<()> {
    let l = Layout::t2va(17, [2, 9, 17], TileShape([2, 8, 8]), 323)?;
    let heads = 2usize;
    let dim = 128usize;
    let seq = l.seq;
    let drive = |offset: usize| {
        let values: Vec<f32> = (0..heads * seq * dim)
            .map(|i| ((i * 17 + offset) % 113) as f32 / 113.0 - 0.5)
            .collect();
        Array::from_slice(&values, &[1, heads as i32, seq as i32, dim as i32])
            .as_dtype(Dtype::Bfloat16)
    };
    let (q, k, v) = (drive(1)?, drive(3)?, drive(7)?);
    let mut rows = Vec::new();
    for _ in 0..heads {
        for r in 0..l.valid.len() {
            rows.push(if r < l.video {
                vec![r, l.video]
            } else {
                (0..l.valid.len()).collect()
            });
        }
    }
    let perm: Vec<i32> = l.perm.iter().map(|&r| r.max(0)).collect();
    let valid: Vec<bool> = l.perm.iter().map(|&r| r >= 0).collect();
    let valid = Array::from_slice(&valid, &[1, l.perm.len() as i32, 1]);
    let tile = |a: &Array| -> Result<Array> {
        let a = a
            .reshape(&[heads as i32, seq as i32, dim as i32])?
            .take_axis(Array::from_slice(&perm, &[perm.len() as i32]), 1)?;
        Ok(
            r#where(&valid, &a, &Array::from_f32(0.0).as_dtype(a.dtype())?)?.reshape(&[
                heads as i32,
                l.valid.len() as i32,
                TILE as i32,
                dim as i32,
            ])?,
        )
    };
    let sparse = gather_attention(&l, &rows, &tile(&q)?, &tile(&k)?, &tile(&v)?, dim, 2)?
        .0
        .reshape(&[1, heads as i32, seq as i32, dim as i32])?;
    let mut mask = vec![false; heads * seq * seq];
    let mut pairs = 0usize;
    for head in 0..heads {
        for row in 0..l.valid.len() {
            for qi in 0..l.valid[row] {
                let qr = l.perm[row * TILE + qi] as usize;
                for &kt in &rows[head * l.valid.len() + row] {
                    for ki in 0..l.valid[kt] {
                        mask[(head * seq + qr) * seq + l.perm[kt * TILE + ki] as usize] = true;
                        pairs += 1;
                    }
                }
            }
        }
    }
    if pairs >= heads * seq * seq {
        return Err(Error::Msg(
            "Veda self-test did not execute sparse selection".into(),
        ));
    }
    let mask = Array::from_slice(&mask, &[1, heads as i32, seq as i32, seq as i32]);
    let reference = sdpa_bounded_bhsd(
        &q,
        &k,
        &v,
        1.0 / (dim as f32).sqrt(),
        Some(&mask),
        BoundedAttention::UNBOUNDED,
    )?;
    let error = sparse
        .as_dtype(Dtype::Float32)?
        .subtract(&reference.as_dtype(Dtype::Float32)?)?
        .abs()?
        .max(None)?
        .item::<f32>();
    #[cfg(test)]
    println!("Veda device self-test: kernel={KERNEL}, D128 BF16, max_abs={error}, tolerance={SELF_TEST_ATOL}, selected_pairs={pairs}, dense_pairs={}",heads*seq*seq);
    if !error.is_finite() || error > SELF_TEST_ATOL {
        return Err(Error::Msg(format!(
            "Veda {KERNEL} actual-device self-test failed: {error} > {SELF_TEST_ATOL}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod frozen_tests {
    use super::*;
    /// Requires the coordinator's exclusive Metal token; never runs in CPU gates.
    #[test]
    #[ignore = "requires exclusive coordinator admission for an actual MLX device"]
    fn frozen_upstream_device_scores_select_exact_masks() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/gen-core/tests/fixtures/veda/frozen.json"
        )))
        .unwrap();
        let uv = |v: &serde_json::Value| {
            v.as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap() as usize)
                .collect::<Vec<_>>()
        };
        let fv = |v: &serde_json::Value| {
            v.as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        for c in fixture["scorers"].as_array().unwrap() {
            let spans = c["layout"]["spans"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| contract::TiledSpan {
                    start: s["start"].as_u64().unwrap() as usize,
                    grid: uv(&s["grid"]).try_into().unwrap(),
                    shape: TileShape(uv(&s["shape"]).try_into().unwrap()),
                })
                .collect::<Vec<_>>();
            let layout =
                Layout::from_spans(&spans, c["layout"]["seq"].as_u64().unwrap() as usize).unwrap();
            let heads = c["heads"].as_u64().unwrap() as usize;
            let dim = c["dim"].as_u64().unwrap() as usize;
            let pq = c["proj"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as u16)
                .collect::<Vec<_>>();
            let mut pk = vec![0; pq.len()];
            for h in 0..heads {
                for i in 0..3 * dim {
                    for d in 0..dim {
                        pk[(h * 3 * dim + i) * dim + d] =
                            pq[(h * 3 * dim + 3 * dim - 1 - i) * dim + d];
                    }
                }
            }
            let bundle = Bundle {
                identity: "frozen-capture-fixture".into(),
                layers: 1,
                heads,
                dim,
                plans: vec![],
                q: vec![pq],
                k: vec![pk],
            };
            let shape = [
                heads as i32,
                layout.valid.len() as i32,
                TILE as i32,
                dim as i32,
            ];
            let q = Array::from_slice(&fv(&c["q"]), &shape)
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let k = Array::from_slice(&fv(&c["k"]), &shape)
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let scores = learned_scores(
                &bundle,
                0,
                &(0..heads as i32).collect::<Vec<_>>(),
                &layout,
                &q,
                &k,
            )
            .unwrap();
            let masks = layout
                .select_with_budgets(&scores, heads, Budget::Tiles(2.3), Budget::Tiles(1.5))
                .unwrap();
            let expected = c["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(&uv)
                .collect::<Vec<_>>();
            assert_eq!(
                masks, expected,
                "device logits must select frozen masks exactly"
            );
        }
    }
}
