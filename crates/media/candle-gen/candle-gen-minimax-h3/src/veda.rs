//! Request-local native Candle Veda. Default/off never constructs this runtime.
use crate::denoise::PackedLayout;
pub use candle_gen::gen_core::veda::contract;
use candle_gen::{
    candle_core::{DType, Device, Tensor},
    gen_core::{VedaBudget, VedaExecutionReport, VedaRequest},
    quant::veda_int8,
    CandleError, Result,
};
use contract::{Budget, Bundle, Layout, Plan, PredictorOps, TileShape, TILE};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};
pub const COMPONENT: &str = "veda_predictor";
pub use candle_gen::quant::veda_int8::KERNEL;
pub type ReportSink = Rc<RefCell<VedaExecutionReport>>;
const WORKSPACE_ALLOWANCE: u64 = 512 * 1024 * 1024;
struct Prepared {
    bundle: Bundle,
    plan: Plan,
    layouts: BTreeMap<TileShape, Layout>,
}
pub struct VedaRuntime {
    request: VedaRequest,
    report: ReportSink,
    prepared: Option<Prepared>,
    failure: RefCell<Option<String>>,
    // Job-local and actual Candle device/context identity. No unrelated process-global pass.
    tested: RefCell<Option<(Device, DType, usize)>>,
    self_test: fn(&Device, DType, usize) -> Result<()>,
    #[cfg(test)]
    pub(crate) calls: RefCell<Vec<AttentionCall>>,
}
#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AttentionCall {
    pub layer: usize,
    pub step: usize,
    pub full_attention: bool,
    pub request: VedaRequest,
    pub shape: Vec<usize>,
}
impl VedaRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        request: &VedaRequest,
        predictor: Option<&std::path::Path>,
        packed: &PackedLayout,
        layers: usize,
        heads: usize,
        dim: usize,
        device: &Device,
        report: ReportSink,
    ) -> Option<Self> {
        if !request.enabled {
            return None;
        }
        let result = (|| -> Result<Prepared> {
            request
                .validate(layers, usize::MAX)
                .map_err(CandleError::Msg)?;
            let path = predictor.ok_or_else(|| {
                CandleError::Msg(
                    "Veda predictor component is missing; repair/install the managed component"
                        .into(),
                )
            })?;
            if !matches!(device, Device::Cuda(_)) {
                return Err(
                    "Veda sparse kernel unavailable: requires CUDA; this Candle device is unsupported".into(),
                );
            }
            report.borrow_mut().predictor_load_attempts += 1;
            let bundle = Bundle::load(path, layers, heads, dim)?;
            Self::build(bundle, packed, &report)
        })();
        let (prepared, failure) = match result {
            Ok(p) => (Some(p), None),
            Err(e) => (None, Some(e.to_string())),
        };
        Some(Self {
            request: request.clone(),
            report,
            prepared,
            failure: RefCell::new(failure),
            tested: RefCell::new(None),
            self_test: device_self_test,
            #[cfg(test)]
            calls: Default::default(),
        })
    }
    fn build(bundle: Bundle, packed: &PackedLayout, report: &ReportSink) -> Result<Prepared> {
        let g = packed.geometry();
        let grid = [g.num_latent_frames, g.latent_height / 2, g.latent_width / 2];
        let plan = bundle.choose(grid);
        let mut layouts = BTreeMap::new();
        for &shape in &plan.shapes {
            layouts.insert(shape, packed.veda_layout(shape)?);
        }
        let mut r = report.borrow_mut();
        r.predictor = Some(bundle.identity.clone());
        r.plan = Some(plan.name.clone());
        r.exact_plan = plan.grid == grid;
        r.workspace_bytes = r.workspace_bytes.max(bundle.host_bytes());
        Ok(Prepared {
            bundle,
            plan,
            layouts,
        })
    }
    pub fn attention(
        &self,
        layer: usize,
        step: usize,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f64,
    ) -> Result<Tensor> {
        #[cfg(test)]
        self.calls.borrow_mut().push(AttentionCall {
            layer,
            step,
            full_attention: self.request.full_attention(layer, step),
            request: self.request.clone(),
            shape: q.dims().to_vec(),
        });
        if !self.request.full_attention(layer, step) && self.failure.borrow().is_none() {
            match self.execute(layer, q, k, v, scale as f32) {
                Ok((out, pairs, workspace)) => {
                    let (b, h, n, _) = q.dims4()?;
                    let mut r = self.report.borrow_mut();
                    r.sparse_calls += 1;
                    r.kernel = Some(KERNEL.into());
                    r.selected_token_pairs += pairs;
                    r.dense_token_pairs += b as u64 * h as u64 * n as u64 * n as u64;
                    r.workspace_bytes = r.workspace_bytes.max(workspace);
                    return Ok(out);
                }
                Err(e) => *self.failure.borrow_mut() = Some(format!("Veda execution failed: {e}")),
            }
        }
        if !self.request.full_attention(layer, step) {
            if let Some(reason) = self.failure.borrow().as_ref() {
                self.report.borrow_mut().fallback_reason =
                    Some(format!("{reason}; dense attention executed"));
            }
        }
        let out = crate::nn::sdpa(q, k, v, scale)?;
        let (b, h, n, _) = q.dims4()?;
        let pairs = b as u64 * h as u64 * n as u64 * n as u64;
        let mut r = self.report.borrow_mut();
        r.dense_calls += 1;
        r.selected_token_pairs += pairs;
        r.dense_token_pairs += pairs;
        Ok(out)
    }
    fn execute(
        &self,
        layer: usize,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
    ) -> Result<(Tensor, u64, u64)> {
        let p = self
            .prepared
            .as_ref()
            .ok_or_else(|| CandleError::Msg("Veda predictor/layout unavailable".into()))?;
        let (batch, heads, seq, dim) = q.dims4()?;
        if heads != p.bundle.heads
            || dim != p.bundle.dim
            || seq != p.layouts.values().next().unwrap().seq
            || layer >= p.bundle.layers
            || k.dims() != q.dims()
            || v.dims() != q.dims()
            || !matches!(q.dtype(), DType::BF16 | DType::F16)
            || !matches!(q.device(), Device::Cuda(_))
        {
            return Err("Veda unsupported device/dtype/layout".into());
        }
        let tested = self
            .tested
            .borrow()
            .as_ref()
            .is_some_and(|(d, t, n)| d.same_device(q.device()) && *t == q.dtype() && *n == dim);
        if !tested {
            let mut report = self.report.borrow_mut();
            report.workspace_bytes = report
                .workspace_bytes
                .max(p.bundle.host_bytes() + self_test_workspace_bytes(dim));
            drop(report);
            (self.self_test)(q.device(), q.dtype(), dim)?;
            *self.tested.borrow_mut() = Some((q.device().clone(), q.dtype(), dim));
        }
        let budget = |b: &VedaBudget| match b {
            VedaBudget::SparsityPercent(p) => Budget::Ratio(1. - p / 100.),
            VedaBudget::KeyTiles(n) => Budget::Tiles(*n as f64),
        };
        let mut batches = Vec::new();
        let mut pairs = 0;
        let mut high = 0;
        for b in 0..batch {
            let mut outputs = Vec::new();
            for h in 0..heads {
                let layout = &p.layouts[&p.plan.shapes[p.plan.heads[layer][h]]];
                let slots = layout.perm.len();
                let tiles = layout.valid.len();
                // Bound copies, pooling, projected features, logits, worst-case mask, kernel scratch,
                // retained and final outputs BEFORE any gather (even minimum single-head batches).
                let minimum = 6 * slots as u64 * dim as u64
                    + 20 * slots as u64 * dim as u64
                    + (8 * layout.video * 3 * dim
                        + 8 * layout.video * dim
                        + 4 * layout.video * layout.video
                        + 24 * dim * dim) as u64
                    + (8 * tiles * tiles + 8 * tiles + 4 * (tiles + 1)) as u64;
                if minimum > WORKSPACE_ALLOWANCE {
                    return Err(
                        "Veda minimum head workspace exceeds problem-bounded allowance".into(),
                    );
                }
                let qt = tile(q, b, h, layout)?;
                let kt = tile(k, b, h, layout)?;
                let vt = tile(v, b, h, layout)?;
                let scores = contract::learned_scores(
                    &p.bundle,
                    layer,
                    &[h as i32],
                    layout,
                    &qt,
                    &kt,
                    &CandlePredictorOps,
                )?;
                self.report.borrow_mut().predictor_score_calls += 1;
                let rows = layout.select_with_budgets(
                    &scores,
                    1,
                    budget(&self.request.generated),
                    budget(&self.request.reference),
                )?;
                pairs += rows
                    .iter()
                    .enumerate()
                    .map(|(r, ks)| {
                        layout.valid[r] as u64
                            * ks.iter().map(|&c| layout.valid[c] as u64).sum::<u64>()
                    })
                    .sum::<u64>();
                let mask = veda_int8::Mask::new(&rows, &layout.valid, 1)?;
                let out = veda_int8::attend(
                    &qt.reshape((1, slots, dim))?,
                    &kt.reshape((1, slots, dim))?,
                    &vt.reshape((1, slots, dim))?,
                    &mask,
                    scale,
                )?;
                let inverse = Tensor::from_vec(
                    layout.inverse.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                    seq,
                    q.device(),
                )?;
                outputs.push(out.index_select(&inverse, 1)?.reshape((1, 1, seq, dim))?);
                high = high.max(
                    minimum
                        + 8 * batch as u64 * heads as u64 * seq as u64 * dim as u64
                        + p.bundle.host_bytes(),
                );
            }
            batches.push(Tensor::cat(&outputs, 1)?);
        }
        Ok((Tensor::cat(&batches, 0)?, pairs, high))
    }
}
fn tile(x: &Tensor, b: usize, h: usize, l: &Layout) -> Result<Tensor> {
    let (_, _, seq, dim) = x.dims4()?;
    let perm = Tensor::from_vec(
        l.perm.iter().map(|&x| x.max(0) as u32).collect::<Vec<_>>(),
        l.perm.len(),
        x.device(),
    )?;
    let valid = Tensor::from_vec(
        l.perm.iter().map(|&x| u8::from(x >= 0)).collect::<Vec<_>>(),
        (l.perm.len(), 1),
        x.device(),
    )?
    .broadcast_as((l.perm.len(), dim))?;
    let a = x
        .get(b)?
        .get(h)?
        .reshape((seq, dim))?
        .index_select(&perm, 0)?;
    let zero = Tensor::zeros(a.dims(), a.dtype(), a.device())?;
    Ok(valid
        .where_cond(&a, &zero)?
        .reshape((1, l.valid.len(), TILE, dim))?)
}
struct CandlePredictorOps;
impl PredictorOps for CandlePredictorOps {
    type Tensor = Tensor;
    type Error = CandleError;
    fn pool(&self, x: &Tensor, l: &Layout, h: usize, d: usize) -> Result<Tensor> {
        let n = l.video;
        let x = x.narrow(1, 0, n)?.to_dtype(DType::F32)?;
        let valid = Tensor::from_vec(
            l.valid[..n]
                .iter()
                .flat_map(|&c| (0..TILE).map(move |r| u8::from(r < c)))
                .collect::<Vec<_>>(),
            (1, n, TILE, 1),
            x.device(),
        )?
        .broadcast_as((h, n, TILE, d))?;
        let counts = Tensor::from_vec(
            l.valid[..n]
                .iter()
                .map(|&c| c.max(1) as f32)
                .collect::<Vec<_>>(),
            (1, n, 1),
            x.device(),
        )?;
        let mean = x.sum(2)?.broadcast_div(&counts)?;
        let max = valid
            .where_cond(&x, &Tensor::full(f32::NEG_INFINITY, x.dims(), x.device())?)?
            .max(2)?;
        let min = valid
            .where_cond(&x, &Tensor::full(f32::INFINITY, x.dims(), x.device())?)?
            .min(2)?;
        let f = Tensor::cat(&[mean, max, min], 2)?;
        let ok = Tensor::from_vec(
            l.valid[..n]
                .iter()
                .map(|&c| u8::from(c > 0))
                .collect::<Vec<_>>(),
            (1, n, 1),
            x.device(),
        )?
        .broadcast_as(f.dims())?;
        Ok(ok.where_cond(&f, &Tensor::zeros(f.dims(), DType::F32, x.device())?)?)
    }
    fn project(&self, f: &Tensor, p: &[f32], h: usize, d: usize) -> Result<Tensor> {
        Ok(f.contiguous()?
            .matmul(&Tensor::from_vec(p.to_vec(), (h, 3 * d, d), f.device())?)?)
    }
    fn prefix(&self, f: &Tensor, d: usize) -> Result<Tensor> {
        Ok(f.narrow(2, 0, d)?)
    }
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Ok((a + b)?)
    }
    fn dot_scaled(&self, q: &Tensor, k: &Tensor, divisor: f32) -> Result<Vec<f32>> {
        Ok(
            (q.contiguous()?.matmul(&k.transpose(1, 2)?.contiguous()?)? / divisor as f64)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        )
    }
}
/// Fixed fixture problem bound, including device/host QKV, conversion/quant buffers,
/// retained host oracles and dense additive masks/scores/probabilities. Independent
/// of free memory and accounted even when the first-use check refuses sparse execution.
pub fn self_test_workspace_bytes(dim: usize) -> u64 {
    let slots = 4 * TILE;
    64 * slots as u64 * dim as u64 + 12 * slots as u64 * slots as u64 + 64 * slots as u64
}
/// Require actual CUDA; there is no skip-success path. Uses the same production kernel.
pub fn device_self_test(device: &Device, dtype: DType, dim: usize) -> Result<()> {
    if !matches!(device, Device::Cuda(_)) {
        return Err("Veda actual-device self-test requires CUDA".into());
    }
    if !matches!(dtype, DType::BF16 | DType::F16) || dim == 0 || dim > 128 || !dim.is_multiple_of(4)
    {
        return Err("Veda self-test unsupported dtype/head width".into());
    }
    let valid = vec![17, 128, 9, 3]; // reference, generated, partial generated, dense global
    let rows = vec![vec![0, 3], vec![1, 3], vec![2, 3], vec![0, 1, 2, 3]];
    let mask = veda_int8::Mask::new(&rows, &valid, 1)?;
    let slots = mask.slots();
    let n = slots * dim;
    let q: Vec<f32> = (0..n)
        .map(|i| ((i * 17 % 101) as f32 - 50.) / 80.)
        .collect();
    let k: Vec<f32> = (0..n)
        .map(|i| ((i * 31 % 103) as f32 - 51.) / 80.)
        .collect();
    let v: Vec<f32> = (0..n)
        .map(|i| (i / dim / 128) as f32 * 0.6 - 0.9 + ((i * 13 % 29) as f32 - 14.) / 90.)
        .collect();
    let tensor = |a: Vec<f32>| -> Result<Tensor> {
        Ok(Tensor::from_vec(a, (1, slots, dim), device)?.to_dtype(dtype)?)
    };
    let (q, k, v) = (tensor(q)?, tensor(k)?, tensor(v)?);
    let host = |x: &Tensor| -> Result<Vec<f32>> {
        Ok(x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
    };
    let (qh, kh, vh) = (host(&q)?, host(&k)?, host(&v)?);
    let scale = 1. / (dim as f32).sqrt();
    let got = host(&veda_int8::attend(&q, &k, &v, &mask, scale)?)?;
    let oracle = veda_int8::reference(&qh, &kh, &vh, dim, scale, &mask)?;
    // Same arithmetic: two output ULPs plus FP32 reduction roundoff. This is not a
    // tolerance for changing masks/INT8 scales. BF16 2^-7, F16 2^-10.
    let ulp = if dtype == DType::BF16 {
        1. / 128.
    } else {
        1. / 1024.
    };
    for (i, (&a, &b)) in got.iter().zip(&oracle).enumerate() {
        if !a.is_finite() || (a - b).abs() > 2. * ulp * b.abs() + 2e-4 {
            return Err(
                format!("Veda self-test frozen arithmetic mismatch at {i}: {a} vs {b}").into(),
            );
        }
    }
    // Independently dense FP32 oracle with exactly the selected mask. Quantization error
    // bound: input rounding <= block_scale/2; bound every logit error, then total
    // variation <= tanh(max_logit_error), plus FP16 block-dot and output rounding.
    let (masked, bound) = masked_reference(&qh, &kh, &vh, dim, scale, &mask);
    let device_masked = host(&device_masked_reference(&q, &k, &v, scale, &mask)?)?;
    for (i, (&a, &b)) in device_masked.iter().zip(&masked).enumerate() {
        if !a.is_finite() || (a - b).abs() > 2e-4 {
            return Err(format!("Veda self-test device dense oracle mismatch at {i}").into());
        }
    }
    for (i, (&a, &b)) in got.iter().zip(&masked).enumerate() {
        if (a - b).abs() > bound + 2. * ulp * b.abs() + 2e-4 {
            return Err(format!("Veda self-test masked dense mismatch at {i}").into());
        }
    }
    let dense_mask = veda_int8::Mask::new(&vec![vec![0, 1, 2, 3]; 4], &valid, 1)?;
    let dense = host(&device_masked_reference(&q, &k, &v, scale, &dense_mask)?)?;
    let delta = got
        .iter()
        .zip(&dense)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0f32, f32::max);
    if delta < 0.1 {
        return Err("Veda self-test output equals unmasked dense attention".into());
    }
    Ok(())
}
fn device_masked_reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: &veda_int8::Mask,
) -> Result<Tensor> {
    let (heads, slots, dim) = q.dims3()?;
    let mut additive = vec![f32::NEG_INFINITY; heads * slots * slots];
    let mut query_valid = vec![0f32; slots];
    for h in 0..heads {
        for row in 0..slots {
            let tile = row / 128;
            if row % 128 < (mask.valid[2 * tile] + mask.valid[2 * tile + 1]) as usize {
                query_valid[row] = 1.;
            }
            let ri = h * (slots / 128) + tile;
            for &block in &mask.indices[mask.offsets[ri] as usize..mask.offsets[ri + 1] as usize] {
                for r in 0..mask.valid[block as usize] as usize {
                    additive[(h * slots + row) * slots + block as usize * 64 + r] = 0.;
                }
            }
        }
    }
    let additive = Tensor::from_vec(additive, (1, heads, slots, slots), q.device())?;
    let shape = (1, heads, slots, dim);
    let out = candle_gen::sdpa_budgeted_bhsd(
        &q.to_dtype(DType::F32)?.reshape(shape)?,
        &k.to_dtype(DType::F32)?.reshape(shape)?,
        &v.to_dtype(DType::F32)?.reshape(shape)?,
        scale as f64,
        Some(&additive),
        candle_gen::candle_nn::ops::softmax_last_dim,
        candle_gen::ATTN_SCORES_BUDGET,
    )?;
    let query_valid = Tensor::from_vec(query_valid, (1, 1, slots, 1), q.device())?;
    Ok(out
        .broadcast_mul(&query_valid)?
        .reshape((heads, slots, dim))?)
}
fn masked_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dim: usize,
    scale: f32,
    m: &veda_int8::Mask,
) -> (Vec<f32>, f32) {
    let slots = m.slots();
    let mut out = vec![0.; q.len()];
    let vmax = v.iter().map(|x| x.abs()).fold(0f32, f32::max);
    let peakq = q.iter().map(|x| x.abs()).fold(0f32, f32::max);
    let peakk = k.iter().map(|x| x.abs()).fold(0f32, f32::max);
    let eq = peakq / 254.;
    let ek = peakk / 254.;
    let epsilon = dim as f32 * scale * (eq * peakk + ek * peakq + eq * ek);
    // 64-product FP16 dot result rounding, FP16 P/V rounding and rescaling over
    // retained blocks: conservatively 4 half ULPs per block plus one output ULP.
    let blocks = m.offsets.windows(2).map(|a| a[1] - a[0]).max().unwrap_or(0) as f32;
    let bound = 2. * vmax * epsilon.tanh() + vmax * (4. * blocks + 1.) / 1024.;
    for h in 0..m.heads {
        for row in 0..slots {
            let tile = row / 128;
            if row % 128 >= (m.valid[2 * tile] + m.valid[2 * tile + 1]) as usize {
                continue;
            }
            let ri = h * (slots / 128) + tile;
            let keys: Vec<usize> = m.indices[m.offsets[ri] as usize..m.offsets[ri + 1] as usize]
                .iter()
                .flat_map(|&b| (0..m.valid[b as usize] as usize).map(move |r| b as usize * 64 + r))
                .collect();
            let logits: Vec<f32> = keys
                .iter()
                .map(|&key| {
                    (0..dim)
                        .map(|d| q[(h * slots + row) * dim + d] * k[(h * slots + key) * dim + d])
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let p: Vec<f32> = logits.iter().map(|&x| (x - peak).exp()).collect();
            let sum = p.iter().sum::<f32>();
            for d in 0..dim {
                out[(h * slots + row) * dim + d] = keys
                    .iter()
                    .zip(&p)
                    .map(|(&key, &p)| p * v[(h * slots + key) * dim + d])
                    .sum::<f32>()
                    / sum;
            }
        }
    }
    (out, bound)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::denoise::{JointGeometry, TEXT_TAG};
    fn packed(dev: &Device) -> PackedLayout {
        PackedLayout::build(
            JointGeometry::new(5, 16, 16).unwrap(),
            [1, 2, 2],
            &[TEXT_TAG; 3],
            32,
            &[],
            dev,
        )
        .unwrap()
    }
    fn predictor(layers: usize, heads: usize, dim: usize) -> Bundle {
        use serde_json::json;
        let mut metadata = serde_json::Map::new();
        for (k, v) in [
            ("format", "miowtion-veda-predictor-v1".to_string()),
            ("num_layers", layers.to_string()),
            ("num_heads", heads.to_string()),
            ("head_dim", dim.to_string()),
            ("keep_ratio", "0.1".into()),
            ("dtype", "float8_e4m3fn".into()),
        ] {
            metadata.insert(k.into(), json!(v));
        }
        metadata.insert("plans".into(),json!(json!({"fixture":{"geometry":"16x9_t2","grid":[2,8,8],"shapes":["2x4x16"],"head_shape":vec![vec![0;heads];layers]}}).to_string()));
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), json!(metadata));
        let mut bytes = vec![];
        for layer in 0..layers {
            for name in ["proj_q", "proj_k"] {
                let key = format!("layers.{layer}.{name}");
                let start = bytes.len();
                bytes.extend(
                    (0..heads * 3 * dim * dim).map(|i| if i % 7 == 0 { 0x9b } else { 0x1a }),
                );
                header.insert(key.clone(),json!({"dtype":"F8_E4M3","shape":[heads,3*dim,dim],"data_offsets":[start,bytes.len()]}));
                let start = bytes.len();
                for _ in 0..heads {
                    bytes.extend(0.03f32.to_le_bytes());
                }
                header.insert(
                    format!("{key}.__scale"),
                    json!({"dtype":"F32","shape":[heads],"data_offsets":[start,bytes.len()]}),
                );
            }
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ')
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(bytes);
        let meta = candle_gen::gen_core::weightsmeta::CheckpointMeta::from_bytes(file).unwrap();
        Bundle::from_meta(&meta, layers, heads, dim).unwrap()
    }
    #[cfg(feature = "cuda")]
    fn runtime(dev: &Device, report: ReportSink) -> VedaRuntime {
        let p = packed(dev);
        VedaRuntime {
            prepared: Some(VedaRuntime::build(predictor(2, 2, 16), &p, &report).unwrap()),
            request: VedaRequest {
                enabled: true,
                generated: VedaBudget::KeyTiles(1),
                reference: VedaBudget::SparsityPercent(0.),
                ..Default::default()
            },
            report,
            failure: RefCell::new(None),
            tested: RefCell::new(None),
            self_test: device_self_test,
            calls: Default::default(),
        }
    }
    fn tensor(n: usize, dev: &Device, dtype: DType, phase: usize) -> Tensor {
        Tensor::from_vec(
            (0..2 * 2 * n * 16)
                .map(|i| ((i * 17 + phase) % 101) as f32 / 80. - 0.5)
                .collect::<Vec<_>>(),
            (2, 2, n, 16),
            dev,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
    }
    #[test]
    fn omitted_off_never_initializes_and_failure_is_visible_dense() {
        let p = packed(&Device::Cpu);
        let report = ReportSink::default();
        assert!(VedaRuntime::prepare(
            &VedaRequest::default(),
            None,
            &p,
            2,
            2,
            16,
            &Device::Cpu,
            report.clone()
        )
        .is_none());
        assert_eq!(report.borrow().predictor_load_attempts, 0);
        let req = VedaRequest {
            enabled: true,
            ..Default::default()
        };
        for path in [None, Some(std::path::Path::new("/managed/predictor"))] {
            let r = VedaRuntime::prepare(&req, path, &p, 2, 2, 16, &Device::Cpu, report.clone())
                .unwrap();
            let q = tensor(p.seq_len(), &Device::Cpu, DType::F32, 0);
            let k = tensor(p.seq_len(), &Device::Cpu, DType::F32, 1);
            let v = tensor(p.seq_len(), &Device::Cpu, DType::F32, 2);
            let got = r.attention(0, 0, &q, &k, &v, 0.25).unwrap();
            let want = crate::nn::sdpa(&q, &k, &v, 0.25).unwrap();
            assert_eq!(
                got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                want.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            );
            assert!(report.borrow().fallback_reason.as_ref().unwrap().contains(
                if path.is_none() {
                    "missing"
                } else {
                    "unsupported"
                }
            ));
        }
        assert_eq!(report.borrow().sparse_calls, 0);
    }
    #[test]
    fn candle_trip_pool_and_projection_use_shared_frozen_scorer() {
        let dev = Device::Cpu;
        let p = packed(&dev);
        let l = p.veda_layout(TileShape([2, 4, 16])).unwrap();
        let n = p.seq_len();
        let q = tensor(n, &dev, DType::F32, 0);
        let k = tensor(n, &dev, DType::F32, 3);
        let qt = tile(&q, 0, 0, &l).unwrap();
        let kt = tile(&k, 0, 0, &l).unwrap();
        let bundle = predictor(2, 2, 16);
        let scores =
            contract::learned_scores(&bundle, 0, &[0], &l, &qt, &kt, &CandlePredictorOps).unwrap();
        let qhost = qt.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let khost = kt.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let qpool = contract::pool_video_tiles(&qhost, &l, 1, 16).unwrap();
        let kpool = contract::pool_video_tiles(&khost, &l, 1, 16).unwrap();
        let oracle = contract::tile_logits(
            &qpool,
            &kpool,
            &bundle.projection(0, &[0], true),
            &bundle.projection(0, &[0], false),
            1,
            l.video,
            16,
        )
        .unwrap();
        for (&a, &b) in scores.iter().zip(&oracle) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        let actual = l
            .select_with_budgets(&scores, 1, Budget::Tiles(1.), Budget::Ratio(1.))
            .unwrap();
        let expected = l
            .select_with_budgets(&oracle, 1, Budget::Tiles(1.), Budget::Ratio(1.))
            .unwrap();
        assert_eq!(actual, expected);
    }
    /// Coordinator must explicitly admit a CUDA device before running this gate.
    /// Missing/unsupported CUDA is failure, never an ignored or early-return success.
    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_kernel_gate() {
        let dev = Device::new_cuda(0).expect("S3 requires an exclusively admitted CUDA device");
        for dtype in [DType::BF16, DType::F16] {
            for dim in [16, 128] {
                device_self_test(&dev, dtype, dim).unwrap();
            }
            let report = ReportSink::default();
            let mut r = runtime(&dev, report.clone());
            let n = r
                .prepared
                .as_ref()
                .unwrap()
                .layouts
                .values()
                .next()
                .unwrap()
                .seq;
            let q = tensor(n, &dev, dtype, 0);
            let k = tensor(n, &dev, dtype, 3);
            let v = tensor(n, &dev, dtype, 7);
            r.attention(0, 0, &q, &k, &v, 0.25).unwrap();
            assert_eq!(report.borrow().sparse_calls, 1);
            assert!(report.borrow().predictor_score_calls > 0);
            assert!(report.borrow().selected_token_pairs < report.borrow().dense_token_pairs);
            assert_eq!(report.borrow().kernel.as_deref(), Some(KERNEL));
            assert!(r.tested.borrow().is_some());
            r.request.full_attention_steps = vec![1];
            let full = r.attention(0, 1, &q, &k, &v, 0.25).unwrap();
            let dense = crate::nn::sdpa(&q, &k, &v, 0.25).unwrap();
            let host = |t: Tensor| {
                t.to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            };
            assert_eq!(host(full), host(dense));
            let unsupported_report = ReportSink::default();
            let unsupported = runtime(&dev, unsupported_report.clone());
            let (qf, kf, vf) = (
                q.to_dtype(DType::F32).unwrap(),
                k.to_dtype(DType::F32).unwrap(),
                v.to_dtype(DType::F32).unwrap(),
            );
            let got = unsupported.attention(0, 0, &qf, &kf, &vf, 0.25).unwrap();
            assert_eq!(
                host(got),
                host(crate::nn::sdpa(&qf, &kf, &vf, 0.25).unwrap())
            );
            assert!(unsupported_report
                .borrow()
                .fallback_reason
                .as_ref()
                .unwrap()
                .contains("unsupported"));
            assert_eq!(unsupported_report.borrow().sparse_calls, 0);
            assert!(unsupported.tested.borrow().is_none());
            let failed_report = ReportSink::default();
            let mut failed = runtime(&dev, failed_report.clone());
            failed.self_test = |_, _, _| Err("Veda actual-device self-test forced failure".into());
            let got = failed.attention(0, 0, &q, &k, &v, 0.25).unwrap();
            assert_eq!(host(got), host(crate::nn::sdpa(&q, &k, &v, 0.25).unwrap()));
            assert_eq!(failed_report.borrow().sparse_calls, 0);
            assert!(failed_report
                .borrow()
                .fallback_reason
                .as_ref()
                .unwrap()
                .contains("forced failure"));
            assert!(failed.tested.borrow().is_none());
            let _ = failed.attention(0, 2, &q, &k, &v, 0.25).unwrap();
            assert_eq!(failed_report.borrow().dense_calls, 2);
            let Device::Cuda(cuda) = &dev else {
                unreachable!()
            };
            println!(
                "S3_DEVICE_RECEIPT {}",
                serde_json::json!({
                    "schema":"sc-25771-actual-cuda-kernel-v1", "kernel":KERNEL,
                    "device_ordinal":cuda.cuda_stream().context().ordinal(),
                    "dtype":format!("{dtype:?}"), "self_test_dimensions":[16,128],
                    "native_batch":2, "native_heads":2,
                    "sparse_calls":report.borrow().sparse_calls,
                    "predictor_score_calls":report.borrow().predictor_score_calls,
                    "forced_failure_dense":true, "unsupported_dtype_dense":true,
                    "intentional_dense_identity":true,
                })
            );
        }
    }
    #[test]
    fn independent_masks_include_dense_globals_and_valid_reference_rows() {
        use contract::TiledSpan;
        let l = Layout::from_spans(
            &[
                TiledSpan {
                    start: 2,
                    grid: [1, 1, 17],
                    shape: TileShape([2, 4, 16]),
                },
                TiledSpan {
                    start: 21,
                    grid: [1, 1, 137],
                    shape: TileShape([2, 4, 16]),
                },
            ],
            161,
        )
        .unwrap();
        let scores = vec![0f32; l.video * l.video];
        let selected = l
            .select_with_budgets(&scores, 1, Budget::Tiles(1.), Budget::Ratio(1.))
            .unwrap();
        let globals: Vec<usize> = (l.video..l.valid.len())
            .filter(|&t| l.valid[t] > 0)
            .collect();
        for (q, keys) in selected.iter().enumerate() {
            for &g in &globals {
                assert!(keys.contains(&g));
            }
            if q < l.video && l.valid[q] > 0 {
                assert!(keys.contains(&q));
            }
            if q >= l.video {
                assert_eq!(
                    keys,
                    &(0..l.valid.len())
                        .filter(|&t| l.valid[t] > 0)
                        .collect::<Vec<_>>()
                );
            }
            for r in 0..l.reference {
                if l.valid[r] > 0 {
                    assert!(keys.contains(&r));
                }
            }
        }
        let m = veda_int8::Mask::new(&selected, &l.valid, 1).unwrap();
        m.validate().unwrap();
        assert!(l.valid.iter().any(|&v| v > 0 && v < 128));
    }
    #[test]
    fn actual_device_self_test_cannot_pass_on_cpu() {
        assert!(device_self_test(&Device::Cpu, DType::BF16, 128).is_err());
    }
}
