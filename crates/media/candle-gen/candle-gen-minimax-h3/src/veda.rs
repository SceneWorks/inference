//! Request-local native Candle Veda. Default/off never constructs this runtime.
use crate::denoise::PackedLayout;
pub use candle_gen::gen_core::veda::contract;
use candle_gen::{
    candle_core::{DType, Device, Tensor},
    gen_core::{VedaBudget, VedaExecutionReport, VedaRequest},
    quant::{veda_int8, veda_score},
    CandleError, Result,
};
use contract::{Budget, Bundle, Layout, Plan, TileShape, TILE};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};
pub const COMPONENT: &str = "veda_predictor";
pub use candle_gen::quant::veda_int8::KERNEL;
pub type ReportSink = Rc<RefCell<VedaExecutionReport>>;
use candle_gen::gen_core::veda::workspace::{self, Backend, LayoutSize, Limits};
struct Prepared {
    bundle: Bundle,
    plan: Plan,
    grid: [usize; 3],
    layouts: BTreeMap<TileShape, Layout>,
}
impl Prepared {
    fn bound(&self, request: &VedaRequest, batch: usize) -> Result<workspace::WorkspaceBound> {
        let sizes = self
            .layouts
            .iter()
            .map(|(&s, l)| (s, LayoutSize::from(l)))
            .collect();
        Ok(workspace::bound(
            request,
            Backend::Cuda,
            &self.plan,
            self.grid,
            &sizes,
            self.bundle.layers,
            self.bundle.heads,
            self.bundle.dim,
            batch,
            2,
            Limits::for_backend(Backend::Cuda),
        )?)
    }
    /// Same-shape head batches of `layer` (ascending heads), each at most the bound's batch size.
    /// A refused shape fails before any gather.
    fn batches(
        &self,
        layer: usize,
        bound: &workspace::WorkspaceBound,
    ) -> Result<Vec<(TileShape, Vec<usize>)>> {
        let mut groups: BTreeMap<TileShape, Vec<usize>> = BTreeMap::new();
        for (h, &shape) in self.plan.heads[layer].iter().enumerate() {
            groups.entry(self.plan.shapes[shape]).or_default().push(h);
        }
        let mut batches = Vec::new();
        for (shape, heads) in groups {
            let chunk = &bound.chunks[&shape];
            if let Some(reason) = &chunk.refusal {
                return Err(format!("Veda {reason}").into());
            }
            batches.extend(heads.chunks(chunk.heads).map(|h| (shape, h.to_vec())));
        }
        Ok(batches)
    }
}
/// Selected rows of sparse calls as `(sample, head, rows)`.
#[cfg(test)]
pub(crate) type Masks = Vec<(usize, usize, Vec<Vec<usize>>)>;
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
    /// Every executed head's selected rows as `(sample, head, rows)`, in execution order.
    #[cfg(test)]
    pub(crate) masks: RefCell<Masks>,
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
            let g = packed.geometry();
            let grid = [g.num_latent_frames, g.latent_height / 2, g.latent_width / 2];
            let plan = workspace::frozen_plan(grid)?;
            let sizes = plan
                .shapes
                .iter()
                .map(|&s| packed.veda_layout(s).map(|l| (s, LayoutSize::from(&l))))
                .collect::<Result<BTreeMap<_, _>>>()?;
            // Account loading even when checkpoint integrity/decode fails.
            let initial = workspace::bound(
                request,
                Backend::Cuda,
                &plan,
                grid,
                &sizes,
                layers,
                heads,
                dim,
                1,
                2,
                Limits::for_backend(Backend::Cuda),
            )?;
            report.borrow_mut().workspace_bytes = initial.total_bytes;
            let bundle = Bundle::load(path, layers, heads, dim)?;
            Self::build(bundle, request, packed, &report)
        })();
        let (prepared, failure) = match result {
            Ok(p) => (Some(p), None),
            Err(e) => {
                // Setup failure is observed now, whether or not any later call is
                // sparse-eligible: every call of this request runs dense.
                report.borrow_mut().fallback_reason =
                    Some(format!("{e}; dense attention executed"));
                (None, Some(e.to_string()))
            }
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
            #[cfg(test)]
            masks: Default::default(),
        })
    }
    fn build(
        bundle: Bundle,
        request: &VedaRequest,
        packed: &PackedLayout,
        report: &ReportSink,
    ) -> Result<Prepared> {
        let g = packed.geometry();
        let grid = [g.num_latent_frames, g.latent_height / 2, g.latent_width / 2];
        let plan = bundle.choose(grid);
        let mut layouts = BTreeMap::new();
        for &shape in &plan.shapes {
            layouts.insert(shape, packed.veda_layout(shape)?);
        }
        let sizes = layouts
            .iter()
            .map(|(&s, l)| (s, LayoutSize::from(l)))
            .collect();
        let bound = workspace::bound(
            request,
            Backend::Cuda,
            &plan,
            grid,
            &sizes,
            bundle.layers,
            bundle.heads,
            bundle.dim,
            1,
            2,
            Limits::for_backend(Backend::Cuda),
        )?;
        let mut r = report.borrow_mut();
        r.predictor = Some(bundle.identity.clone());
        r.plan = Some(plan.name.clone());
        r.exact_plan = plan.grid == grid;
        r.execution_grid = Some(grid);
        r.plan_grid = Some(plan.grid);
        r.dense_segments = packed.veda_dense_segments();
        r.workspace_bytes = r.workspace_bytes.max(bound.total_bytes);
        Ok(Prepared {
            bundle,
            plan,
            grid,
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
                Err(e) => {
                    *self.failure.borrow_mut() = Some(format!("Veda execution failed: {e}"));
                    // Failed before any sparse call (e.g. the first-use self-test): no call
                    // ever executed under the chosen plan, so none is reported.
                    let mut r = self.report.borrow_mut();
                    if r.sparse_calls == 0 {
                        r.plan = None;
                        r.plan_grid = None;
                        r.exact_plan = false;
                    }
                }
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
        let bound = p.bound(&self.request, batch)?;
        {
            let mut r = self.report.borrow_mut();
            r.workspace_bytes = r.workspace_bytes.max(bound.total_bytes);
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
        let batches = p.batches(layer, &bound)?;
        // Each same-shape head batch is gathered, scored with one host readback, selected and
        // attended with one kernel launch over all of its heads.
        let mut samples = Vec::with_capacity(batch);
        let mut pairs = 0;
        for b in 0..batch {
            let mut order = Vec::with_capacity(heads);
            let mut outputs = Vec::new();
            for (shape, group) in &batches {
                let layout = &p.layouts[shape];
                let tiles = layout.valid.len();
                let slots = layout.perm.len();
                let g = group.len();
                let qt = tile(q, b, group, layout)?;
                let kt = tile(k, b, group, layout)?;
                let vt = tile(v, b, group, layout)?;
                let rows = select(p, &self.request, layer, group, layout, &qt, &kt)?;
                self.report.borrow_mut().predictor_score_calls += 1;
                #[cfg(test)]
                self.masks.borrow_mut().extend(
                    group
                        .iter()
                        .zip(rows.chunks(tiles))
                        .map(|(&h, r)| (b, h, r.to_vec())),
                );
                pairs += rows
                    .iter()
                    .enumerate()
                    .map(|(r, ks)| {
                        layout.valid[r % tiles] as u64
                            * ks.iter().map(|&c| layout.valid[c] as u64).sum::<u64>()
                    })
                    .sum::<u64>();
                let mask = veda_int8::Mask::new(&rows, &layout.valid, g)?;
                let out = veda_int8::attend(
                    &qt.reshape((g, slots, dim))?,
                    &kt.reshape((g, slots, dim))?,
                    &vt.reshape((g, slots, dim))?,
                    &mask,
                    scale,
                )?;
                let inverse = Tensor::from_vec(
                    layout.inverse.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                    seq,
                    q.device(),
                )?;
                outputs.push(out.index_select(&inverse, 1)?);
                order.extend_from_slice(group);
            }
            samples.push(assemble(&outputs, &order)?.unsqueeze(0)?);
        }
        let out = Tensor::cat(&samples, 0)?;
        ensure_finite(&out)?;
        Ok((out, pairs, bound.total_bytes))
    }
}
/// Concatenates per-batch `[heads, seq, dim]` outputs, whose heads are `order`, back into head order.
fn assemble(outputs: &[Tensor], order: &[usize]) -> Result<Tensor> {
    let out = Tensor::cat(outputs, 0)?;
    if order.iter().enumerate().all(|(i, &h)| i == h) {
        return Ok(out);
    }
    let mut position = vec![0u32; order.len()];
    for (i, &h) in order.iter().enumerate() {
        position[h] = i as u32;
    }
    Ok(out.index_select(&Tensor::from_vec(position, order.len(), out.device())?, 0)?)
}
/// The frozen learned tile logits `[heads, video, video]` of one same-shape head batch of one
/// sample, equal bit for bit to gen-core's host `pool_video_tiles` + `tile_logits` for every head
/// on its own: on CUDA the exact device scorer with one host readback for the whole batch, on any
/// other device the host contract itself. Masks therefore never depend on the batch size.
fn scores(
    p: &Prepared,
    layer: usize,
    heads: &[usize],
    layout: &Layout,
    qt: &Tensor,
    kt: &Tensor,
) -> Result<Vec<f32>> {
    if layer >= p.bundle.layers || heads.is_empty() || heads.iter().any(|&h| h >= p.bundle.heads) {
        return Err("Veda invalid scorer layer/heads".into());
    }
    let ids: Vec<i32> = heads.iter().map(|&h| h as i32).collect();
    let (pq, pk) = (
        p.bundle.projection(layer, &ids, true),
        p.bundle.projection(layer, &ids, false),
    );
    if matches!(qt.device(), Device::Cuda(_)) {
        return Ok(veda_score::tile_logits(
            qt,
            kt,
            &layout.valid,
            layout.video,
            &pq,
            &pk,
        )?);
    }
    let pool = |x: &Tensor| -> Result<Vec<f32>> {
        let x = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        Ok(contract::pool_video_tiles(
            &x,
            layout,
            heads.len(),
            p.bundle.dim,
        )?)
    };
    Ok(contract::tile_logits(
        &pool(qt)?,
        &pool(kt)?,
        &pq,
        &pk,
        heads.len(),
        layout.video,
        p.bundle.dim,
    )?)
}
/// Learned selection for one same-shape head batch of one sample: one scoring pass over every
/// head of the batch ([`scores`]) and the frozen host selection. Rows are head-major, as
/// [`veda_int8::Mask`] expects.
fn select(
    p: &Prepared,
    request: &VedaRequest,
    layer: usize,
    heads: &[usize],
    layout: &Layout,
    qt: &Tensor,
    kt: &Tensor,
) -> Result<Vec<Vec<usize>>> {
    let scores = scores(p, layer, heads, layout, qt, kt)?;
    let (generated, reference) = (budget(&request.generated), budget(&request.reference));
    // Heads select independently (head h reads only its own `[tiles, tiles]` scores), so the
    // frozen host selection runs on contiguous head ranges in parallel and concatenates them in
    // head order: the rows are identical to one `select_with_budgets` over every head.
    let per_head = layout.video * layout.video;
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
    let span = heads.len().div_ceil(workers.clamp(1, heads.len().max(1)));
    if per_head == 0 || scores.len() != heads.len() * per_head || span >= heads.len() {
        return Ok(layout.select_with_budgets(&scores, heads.len(), generated, reference)?);
    }
    let parts = std::thread::scope(|scope| {
        let jobs: Vec<_> = scores
            .chunks(span * per_head)
            .map(|chunk| {
                scope.spawn(move || {
                    layout.select_with_budgets(chunk, chunk.len() / per_head, generated, reference)
                })
            })
            .collect();
        jobs.into_iter()
            .map(|job| job.join().expect("Veda selection worker panicked"))
            .collect::<Vec<_>>()
    });
    let mut rows = Vec::with_capacity(heads.len() * layout.valid.len());
    for part in parts {
        rows.extend(part?);
    }
    Ok(rows)
}
/// Refuses a sparse attention output holding any NaN/Inf, so `attention` recomputes that call
/// densely and records the reason instead of letting the poison into the hidden state (a poisoned
/// residual decodes to a flat black clip with silent audio — the sc-25777 CUDA A/B signature).
/// One f32 sum reduction plus a scalar readback per sparse call, beside the one score readback per
/// head batch. A NaN or Inf anywhere makes the sum non-finite.
pub(crate) fn ensure_finite(out: &Tensor) -> Result<()> {
    let sum = out.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    if sum.is_finite() {
        Ok(())
    } else {
        Err(format!("Veda sparse attention produced non-finite output (sum {sum})").into())
    }
}
fn budget(b: &VedaBudget) -> Budget {
    match b {
        VedaBudget::SparsityPercent(p) => Budget::Ratio(1. - p / 100.),
        VedaBudget::KeyTiles(n) => Budget::Tiles(*n as f64),
    }
}
/// Gathers `heads` of sample `b` into padded tile order `[heads, tiles, TILE, dim]`; padding is zero.
fn tile(x: &Tensor, b: usize, heads: &[usize], l: &Layout) -> Result<Tensor> {
    let (_, all, _, dim) = x.dims4()?;
    let mut a = x.get(b)?.contiguous()?;
    if heads.len() != all || heads.iter().enumerate().any(|(i, &h)| i != h) {
        let ids = Tensor::from_vec(
            heads.iter().map(|&h| h as u32).collect::<Vec<_>>(),
            heads.len(),
            x.device(),
        )?;
        a = a.index_select(&ids, 0)?;
    }
    let perm = Tensor::from_vec(
        l.perm.iter().map(|&x| x.max(0) as u32).collect::<Vec<_>>(),
        l.perm.len(),
        x.device(),
    )?;
    let a = a.index_select(&perm, 1)?;
    let valid = Tensor::from_vec(
        l.perm.iter().map(|&x| u8::from(x >= 0)).collect::<Vec<_>>(),
        (1, l.perm.len(), 1),
        x.device(),
    )?
    .broadcast_as(a.dims())?;
    let zero = Tensor::zeros(a.dims(), a.dtype(), a.device())?;
    Ok(valid
        .where_cond(&a, &zero)?
        .reshape((heads.len(), l.valid.len(), TILE, dim))?)
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
    use crate::dit::positions::KeyframeAnchor;
    fn packed(dev: &Device) -> PackedLayout {
        packed_with(dev, &[])
    }
    /// `t2va`, or `fl2va` with keyframe anchors packed as reference segments: one video tile.
    fn packed_with(dev: &Device, anchors: &[KeyframeAnchor]) -> PackedLayout {
        packed_geometry(dev, anchors, JointGeometry::new(5, 16, 16).unwrap())
    }
    /// A `t2va` grid of several video tiles, so learned selection has real choices per head.
    fn packed_large(dev: &Device) -> PackedLayout {
        packed_geometry(dev, &[], JointGeometry::new(22, 16, 32).unwrap())
    }
    fn packed_geometry(
        dev: &Device,
        anchors: &[KeyframeAnchor],
        geometry: JointGeometry,
    ) -> PackedLayout {
        PackedLayout::build(geometry, [1, 2, 2], &[TEXT_TAG; 3], 32, anchors, dev).unwrap()
    }
    fn predictor(layers: usize, heads: usize, dim: usize) -> Bundle {
        predictor_with(
            layers,
            heads,
            dim,
            &["2x4x16"],
            vec![vec![0; heads]; layers],
        )
    }
    /// Layer 0 gives head 0 `2x4x16` and head 1 `1x8x16`; same-shape batches run in shape order,
    /// so layer 0's run out of head order and must be reassembled.
    fn mixed_predictor() -> Bundle {
        predictor_with(
            2,
            2,
            16,
            &["2x4x16", "1x8x16"],
            vec![vec![0, 1], vec![1, 0]],
        )
    }
    fn predictor_with(
        layers: usize,
        heads: usize,
        dim: usize,
        shapes: &[&str],
        head_shape: Vec<Vec<usize>>,
    ) -> Bundle {
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
        metadata.insert("plans".into(),json!(json!({"fixture":{"geometry":"16x9_t2","grid":[2,8,8],"shapes":shapes,"head_shape":head_shape}}).to_string()));
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), json!(metadata));
        let mut bytes = vec![];
        for layer in 0..layers {
            for (kind, name) in ["proj_q", "proj_k"].into_iter().enumerate() {
                let key = format!("layers.{layer}.{name}");
                let start = bytes.len();
                // Distinct per layer, projection and head (so heads select different rows): finite
                // E4M3 bytes of either sign from a multiplicative hash.
                bytes.extend((0..heads * 3 * dim * dim).map(|i| {
                    let x = (i as u64 + 1)
                        .wrapping_mul(2654435761)
                        .wrapping_add((layer * 2 + kind) as u64 * 40503);
                    ((x >> 9) % 0x58) as u8 | (((x >> 20) & 1) as u8) << 7
                }));
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
        runtime_for(dev, report, &[], VedaBudget::SparsityPercent(0.))
    }
    fn runtime_for(
        dev: &Device,
        report: ReportSink,
        anchors: &[KeyframeAnchor],
        reference: VedaBudget,
    ) -> VedaRuntime {
        runtime_with(dev, report, anchors, reference, predictor(2, 2, 16))
    }
    fn runtime_with(
        dev: &Device,
        report: ReportSink,
        anchors: &[KeyframeAnchor],
        reference: VedaBudget,
        bundle: Bundle,
    ) -> VedaRuntime {
        runtime_packed(report, reference, bundle, &packed_with(dev, anchors))
    }
    /// A sparse `t2va` runtime over [`packed_large`] keeping three generated key tiles per query
    /// tile, so each head's learned choice beyond its forced diagonal shows in its rows.
    fn runtime_large(dev: &Device, report: ReportSink, bundle: Bundle) -> VedaRuntime {
        let mut r = runtime_packed(
            report,
            VedaBudget::SparsityPercent(0.),
            bundle,
            &packed_large(dev),
        );
        r.request.generated = VedaBudget::KeyTiles(3);
        r
    }
    fn runtime_packed(
        report: ReportSink,
        reference: VedaBudget,
        bundle: Bundle,
        p: &PackedLayout,
    ) -> VedaRuntime {
        VedaRuntime {
            prepared: Some(
                VedaRuntime::build(
                    bundle,
                    &VedaRequest {
                        enabled: true,
                        ..Default::default()
                    },
                    p,
                    &report,
                )
                .unwrap(),
            ),
            request: VedaRequest {
                enabled: true,
                generated: VedaBudget::KeyTiles(1),
                reference,
                ..Default::default()
            },
            report,
            failure: RefCell::new(None),
            tested: RefCell::new(None),
            self_test: device_self_test,
            calls: Default::default(),
            masks: Default::default(),
        }
    }
    /// Host values of `x` and, per sample and head, its token-order rows gathered into the head's
    /// tile slots straight from `layout.perm` (padding zero).
    fn host_slots(x: &Tensor, p: &Prepared, layer: usize) -> Vec<Vec<f32>> {
        let (batch, heads, seq, dim) = x.dims4().unwrap();
        let x = x
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut out = Vec::new();
        for b in 0..batch {
            for h in 0..heads {
                let layout = &p.layouts[&p.plan.shapes[p.plan.heads[layer][h]]];
                let base = (b * heads + h) * seq;
                out.push(
                    layout
                        .perm
                        .iter()
                        .flat_map(|&t| {
                            let x = &x;
                            (0..dim).map(move |d| {
                                if t >= 0 {
                                    x[(base + t as usize) * dim + d]
                                } else {
                                    0.
                                }
                            })
                        })
                        .collect(),
                );
            }
        }
        out
    }
    /// The masks of one sparse call, independent of the runtime's batching, scorer and selection
    /// code: per sample and per head on its own, the frozen host `pool_video_tiles` +
    /// `tile_logits` over [`host_slots`], then the frozen `select_with_budgets`.
    fn oracle_masks(r: &VedaRuntime, layer: usize, q: &Tensor, k: &Tensor) -> Masks {
        let p = r.prepared.as_ref().unwrap();
        let (_, heads, _, dim) = q.dims4().unwrap();
        let (qs, ks) = (host_slots(q, p, layer), host_slots(k, p, layer));
        (0..qs.len())
            .map(|i| {
                let (b, h) = (i / heads, i % heads);
                let layout = &p.layouts[&p.plan.shapes[p.plan.heads[layer][h]]];
                let logits = contract::tile_logits(
                    &contract::pool_video_tiles(&qs[i], layout, 1, dim).unwrap(),
                    &contract::pool_video_tiles(&ks[i], layout, 1, dim).unwrap(),
                    &p.bundle.projection(layer, &[h as i32], true),
                    &p.bundle.projection(layer, &[h as i32], false),
                    1,
                    layout.video,
                    dim,
                )
                .unwrap();
                let rows = layout
                    .select_with_budgets(
                        &logits,
                        1,
                        budget(&r.request.generated),
                        budget(&r.request.reference),
                    )
                    .unwrap();
                (b, h, rows)
            })
            .collect()
    }
    /// Host oracle for one sparse call: the [`oracle_masks`], masked FP32 attention per head over
    /// exactly the selected keys, scattered back through `layout.inverse`. Returns the expected
    /// output, the INT8 quantization error bound of the worst head, and the masks the runtime
    /// must have selected exactly.
    fn masked_dense_oracle(
        r: &VedaRuntime,
        layer: usize,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
    ) -> (Vec<f32>, f32, Masks) {
        masked_dense_oracle_for(r, layer, q, k, v, scale, |_, _| true)
    }
    /// [`masked_dense_oracle`] with the (host-costly) attention output computed only for the
    /// `(sample, head)` pairs `keep` admits; the others stay NaN and are not compared. The masks
    /// always cover every head.
    #[allow(clippy::too_many_arguments)]
    fn masked_dense_oracle_for(
        r: &VedaRuntime,
        layer: usize,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        keep: impl Fn(usize, usize) -> bool,
    ) -> (Vec<f32>, f32, Masks) {
        let p = r.prepared.as_ref().unwrap();
        let (_, heads, seq, dim) = q.dims4().unwrap();
        let masks = oracle_masks(r, layer, q, k);
        let (qs, ks, vs) = (
            host_slots(q, p, layer),
            host_slots(k, p, layer),
            host_slots(v, p, layer),
        );
        let mut out = vec![f32::NAN; q.elem_count()];
        let mut worst = 0f32;
        for (i, (b, h, rows)) in masks.iter().enumerate() {
            if !keep(*b, *h) {
                continue;
            }
            let layout = &p.layouts[&p.plan.shapes[p.plan.heads[layer][*h]]];
            let base = (b * heads + h) * seq;
            let mask = veda_int8::Mask::new(rows, &layout.valid, 1).unwrap();
            let (slots, bound) = masked_reference(&qs[i], &ks[i], &vs[i], dim, scale, &mask);
            worst = worst.max(bound);
            for t in 0..seq {
                let slot = layout.inverse[t] as usize;
                out[(base + t) * dim..][..dim].copy_from_slice(&slots[slot * dim..][..dim]);
            }
        }
        (out, worst, masks)
    }
    /// The runtime's recorded masks, sorted like the oracle's.
    #[cfg(feature = "cuda")]
    fn runtime_masks(r: &VedaRuntime) -> Masks {
        let mut masks = r.masks.borrow().clone();
        masks.sort_by_key(|&(b, h, _)| (b, h));
        masks
    }
    #[test]
    fn masked_dense_oracle_is_dense_attention_when_every_key_is_kept() {
        // Guards the CUDA gate's oracle: gather, selection, masking and inverse scatter.
        let dev = Device::Cpu;
        for anchors in [vec![], vec![KeyframeAnchor::First, KeyframeAnchor::Last]] {
            let mut r = runtime_for(
                &dev,
                ReportSink::default(),
                &anchors,
                VedaBudget::SparsityPercent(0.),
            );
            r.request.generated = VedaBudget::SparsityPercent(0.);
            let layout = r
                .prepared
                .as_ref()
                .unwrap()
                .layouts
                .values()
                .next()
                .unwrap();
            // FL2VA keyframe anchors are tiled reference segments, as the CUDA gate requires.
            assert_eq!(layout.reference > 0, !anchors.is_empty());
            let n = layout.seq;
            let (q, k, v) = (
                tensor(n, &dev, DType::F32, 0),
                tensor(n, &dev, DType::F32, 3),
                tensor(n, &dev, DType::F32, 7),
            );
            let flat = |t: Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let dense = flat(crate::nn::sdpa(&q, &k, &v, 0.25).unwrap());
            let (want, _, _) = masked_dense_oracle(&r, 0, &q, &k, &v, 0.25);
            for (i, (&a, &b)) in want.iter().zip(&dense).enumerate() {
                assert!((a - b).abs() < 1e-4, "{anchors:?} {i}: {a} vs {b}");
            }
            // A one-tile generated budget really masks keys.
            r.request.generated = VedaBudget::KeyTiles(1);
            let (sparse, _, _) = masked_dense_oracle(&r, 0, &q, &k, &v, 0.25);
            assert!(sparse.iter().zip(&dense).any(|(a, b)| (a - b).abs() > 1e-3));
        }
    }
    /// Batched against single-head scoring and selection of every layer and sample, against the
    /// independent host oracle per head: logits bit for bit (`to_bits`), rows exactly, gathers
    /// identical and every head scored exactly once. Returns every failure, so one run names all.
    fn batched_scoring_failures(r: &VedaRuntime, q: &Tensor, k: &Tensor) -> Vec<String> {
        let p = r.prepared.as_ref().unwrap();
        let (batch, heads, _, dim) = q.dims4().unwrap();
        let flat = |t: &Tensor| {
            t.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let bound = p.bound(&r.request, batch).unwrap();
        let mut failures = Vec::new();
        for layer in 0..p.bundle.layers {
            let batches = p.batches(layer, &bound).unwrap();
            let mut order: Vec<usize> = batches.iter().flat_map(|(_, g)| g.clone()).collect();
            order.sort_unstable();
            if order != (0..heads).collect::<Vec<_>>() {
                failures.push(format!("layer {layer}: heads not batched exactly once"));
            }
            let mut batched = Vec::new();
            for b in 0..batch {
                for (shape, group) in &batches {
                    let l = &p.layouts[shape];
                    let (tiles, per) = (l.valid.len(), l.video * l.video);
                    let (qt, kt) = (tile(q, b, group, l).unwrap(), tile(k, b, group, l).unwrap());
                    let s = scores(p, layer, group, l, &qt, &kt).unwrap();
                    let rows = select(p, &r.request, layer, group, l, &qt, &kt).unwrap();
                    for (i, &h) in group.iter().enumerate() {
                        let at = format!("layer {layer} sample {b} head {h} of {group:?}");
                        let (qh, kh) = (tile(q, b, &[h], l).unwrap(), tile(k, b, &[h], l).unwrap());
                        if flat(&qt.get(i).unwrap()) != flat(&qh.get(0).unwrap()) {
                            failures.push(format!("{at}: batched gather differs"));
                        }
                        let single = scores(p, layer, &[h], l, &qh, &kh).unwrap();
                        let host = contract::tile_logits(
                            &contract::pool_video_tiles(&flat(&qh), l, 1, dim).unwrap(),
                            &contract::pool_video_tiles(&flat(&kh), l, 1, dim).unwrap(),
                            &p.bundle.projection(layer, &[h as i32], true),
                            &p.bundle.projection(layer, &[h as i32], false),
                            1,
                            l.video,
                            dim,
                        )
                        .unwrap();
                        let mine = &s[i * per..(i + 1) * per];
                        if bits(mine) != bits(&single) {
                            failures.push(format!("{at}: batched logits differ from single-head"));
                        }
                        if bits(mine) != bits(&host) {
                            failures.push(format!("{at}: logits differ from host tile_logits"));
                        }
                        let one = select(p, &r.request, layer, &[h], l, &qh, &kh).unwrap();
                        let rows = rows[i * tiles..(i + 1) * tiles].to_vec();
                        if rows != one {
                            failures.push(format!("{at}: batched rows differ from single-head"));
                        }
                        batched.push((b, h, rows));
                    }
                }
            }
            batched.sort_by_key(|&(b, h, _)| (b, h));
            if batched != oracle_masks(r, layer, q, k) {
                failures.push(format!(
                    "layer {layer}: batched rows differ from host oracle"
                ));
            }
        }
        failures
    }
    #[test]
    fn head_batches_select_and_reassemble_exactly_as_single_heads() {
        let dev = Device::Cpu;
        for (bundle, heads, dim) in [
            (predictor(2, 2, 16), 2, 16),
            (mixed_predictor(), 2, 16),
            (predictor(2, 12, 128), 12, 128),
        ] {
            let r = runtime_large(&dev, ReportSink::default(), bundle);
            let p = r.prepared.as_ref().unwrap();
            assert!(
                p.layouts.values().all(|l| l.video > 2),
                "selection has choices"
            );
            let n = p.layouts.values().next().unwrap().seq;
            let (q, k) = (
                tensor_hd(n, heads, dim, &dev, DType::F32, 0),
                tensor_hd(n, heads, dim, &dev, DType::F32, 3),
            );
            let failures = batched_scoring_failures(&r, &q, &k);
            assert!(failures.is_empty(), "{failures:#?}");
            // Single-shape plans batch every head at once; mixed plans one batch per shape, with
            // layer 0's batches running out of head order.
            let bound = p.bound(&r.request, 2).unwrap();
            let batches = p.batches(0, &bound).unwrap();
            if p.plan.shapes.len() == 1 {
                assert_eq!(batches.len(), 1);
            } else {
                let order: Vec<usize> = batches.iter().flat_map(|(_, g)| g.clone()).collect();
                assert_eq!(order, vec![1, 0]);
            }
        }
    }
    #[test]
    fn batches_follow_the_bound_and_refuse_before_gather() {
        let r = runtime_for(
            &Device::Cpu,
            ReportSink::default(),
            &[],
            VedaBudget::SparsityPercent(0.),
        );
        let p = r.prepared.as_ref().unwrap();
        let mut bound = p.bound(&r.request, 1).unwrap();
        let shape = *bound.chunks.keys().next().unwrap();
        bound.chunks.get_mut(&shape).unwrap().heads = 1;
        let batches = p.batches(0, &bound).unwrap();
        assert_eq!(batches, vec![(shape, vec![0]), (shape, vec![1])]);
        bound.chunks.get_mut(&shape).unwrap().refusal = Some("over allowance".into());
        assert!(p
            .batches(0, &bound)
            .unwrap_err()
            .to_string()
            .contains("over allowance"));
    }
    #[test]
    fn assemble_restores_head_order() {
        let dev = Device::Cpu;
        let head = |h: usize| Tensor::full(h as f32, (1, 3, 2), &dev).unwrap();
        let parts = [
            Tensor::cat(&[head(1), head(3)], 0).unwrap(),
            Tensor::cat(&[head(0), head(2)], 0).unwrap(),
        ];
        let out = assemble(&parts, &[1, 3, 0, 2]).unwrap();
        for h in 0..4 {
            assert_eq!(
                out.get(h)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                vec![h as f32; 6]
            );
        }
        let identity = assemble(&[head(0), head(1)], &[0, 1]).unwrap();
        assert_eq!(identity.dims(), &[2, 3, 2]);
    }
    #[test]
    fn dense_reference_segments_reach_the_execution_report() {
        use crate::dit::positions::ReferenceLatentGeometry;
        use crate::reference::ReferenceKind;
        let refs = [
            ReferenceLatentGeometry {
                kind: ReferenceKind::Image,
                num_latent_frames: 1,
                latent_height: 4,
                latent_width: 6,
                num_audio_latents: 0,
            },
            ReferenceLatentGeometry {
                kind: ReferenceKind::Audio,
                num_latent_frames: 0,
                latent_height: 0,
                latent_width: 0,
                num_audio_latents: 9,
            },
        ];
        let p = PackedLayout::build_ref2va(
            JointGeometry::new(5, 16, 16).unwrap(),
            [1, 2, 2],
            &[TEXT_TAG; 3],
            2,
            &refs,
            &Device::Cpu,
        )
        .unwrap();
        let report = ReportSink::default();
        VedaRuntime::build(
            predictor(2, 2, 16),
            &VedaRequest {
                enabled: true,
                ..Default::default()
            },
            &p,
            &report,
        )
        .unwrap();
        let segments = report.borrow().dense_segments.clone();
        assert_eq!(segments, p.veda_dense_segments());
        assert_eq!(segments.len(), 1);
        assert!(segments[0].starts_with("reference audio"));
        assert_eq!(
            report.borrow().outcome_json()["denseSegments"],
            serde_json::json!(segments)
        );
    }
    #[test]
    fn setup_failure_is_reported_without_a_sparse_eligible_call() {
        let p = packed(&Device::Cpu);
        let report = ReportSink::default();
        // Every layer forced dense: no call is ever sparse-eligible.
        let req = VedaRequest {
            enabled: true,
            full_attention_layers: vec![0, 1],
            ..Default::default()
        };
        let r =
            VedaRuntime::prepare(&req, None, &p, 2, 2, 16, &Device::Cpu, report.clone()).unwrap();
        let reason = report.borrow().fallback_reason.clone();
        assert!(
            reason.as_deref().is_some_and(|r| r.contains("missing")),
            "{reason:?}"
        );
        let q = tensor(p.seq_len(), &Device::Cpu, DType::F32, 0);
        r.attention(0, 0, &q, &q, &q, 0.25).unwrap();
        assert_eq!(report.borrow().fallback_reason, reason);
        assert_eq!(report.borrow().dense_calls, 1);
    }
    #[test]
    fn failure_before_any_sparse_call_reports_no_executed_plan() {
        let dev = Device::Cpu;
        let report = ReportSink::default();
        let r = runtime_for(&dev, report.clone(), &[], VedaBudget::SparsityPercent(0.));
        assert!(report.borrow().plan.is_some() && report.borrow().plan_grid.is_some());
        let n = r
            .prepared
            .as_ref()
            .unwrap()
            .layouts
            .values()
            .next()
            .unwrap()
            .seq;
        let q = tensor(n, &dev, DType::F32, 0);
        // Sparse-eligible, but CPU/F32 is refused before the first-use self-test.
        r.attention(0, 0, &q, &q, &q, 0.25).unwrap();
        let rep = report.borrow();
        assert_eq!(rep.sparse_calls, 0);
        assert!(rep.fallback_reason.is_some());
        assert!(rep.plan.is_none() && rep.plan_grid.is_none() && !rep.exact_plan);
        assert!(rep.outcome_json()["planMatch"].is_null());
    }
    fn tensor(n: usize, dev: &Device, dtype: DType, phase: usize) -> Tensor {
        tensor_hd(n, 2, 16, dev, dtype, phase)
    }
    /// `[2, heads, n, dim]` deterministic inputs.
    fn tensor_hd(
        n: usize,
        heads: usize,
        dim: usize,
        dev: &Device,
        dtype: DType,
        phase: usize,
    ) -> Tensor {
        Tensor::from_vec(
            (0..2 * heads * n * dim)
                .map(|i| ((i * 17 + phase) % 101) as f32 / 80. - 0.5)
                .collect::<Vec<_>>(),
            (2, heads, n, dim),
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
    fn host_scores_are_the_frozen_host_contract() {
        // Off CUDA the scorer is gen-core's host contract itself (the CUDA gate holds the device
        // scorer to it bit for bit).
        let dev = Device::Cpu;
        let r = runtime_with(
            &dev,
            ReportSink::default(),
            &[],
            VedaBudget::SparsityPercent(0.),
            predictor(2, 2, 16),
        );
        let p = r.prepared.as_ref().unwrap();
        let l = p.layouts.values().next().unwrap();
        let n = l.seq;
        let (q, k) = (
            tensor(n, &dev, DType::F32, 0),
            tensor(n, &dev, DType::F32, 3),
        );
        let (qt, kt) = (
            tile(&q, 0, &[0, 1], l).unwrap(),
            tile(&k, 0, &[0, 1], l).unwrap(),
        );
        let got = scores(p, 1, &[0, 1], l, &qt, &kt).unwrap();
        let host = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let fq = contract::pool_video_tiles(&host(&qt), l, 2, 16).unwrap();
        let fk = contract::pool_video_tiles(&host(&kt), l, 2, 16).unwrap();
        let want = contract::tile_logits(
            &fq,
            &fk,
            &p.bundle.projection(1, &[0, 1], true),
            &p.bundle.projection(1, &[0, 1], false),
            2,
            l.video,
            16,
        )
        .unwrap();
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(scores(p, 2, &[0], l, &qt, &kt).is_err());
        assert!(scores(p, 0, &[2], l, &qt, &kt).is_err());
    }
    /// The tiled production attention kernel against the frozen per-row `pointer-v1` kernel it
    /// replaced (`veda_pointer_v1.cu`, verbatim from main), on one problem with three heads whose
    /// rows differ, reference/generated/partial tiles and a dense global tile: the FP32 outputs
    /// must be equal bit for bit, and the harness's tiled launch must be the production route.
    #[cfg(feature = "cuda")]
    fn tiled_kernel_bit_failures(dev: &Device, dtype: DType, dim: usize) -> Vec<String> {
        use candle_gen::candle_core::{cuda_backend::cudarc, CudaStorage, Storage};
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let Device::Cuda(cuda) = dev else {
            unreachable!()
        };
        let heads = 3;
        let valid = vec![17, 128, 9, 3]; // reference, generated, partial generated, dense global
        let rows: Vec<Vec<usize>> = vec![
            vec![0, 3],
            vec![1, 3],
            vec![2, 3],
            vec![0, 1, 2, 3],
            vec![0, 1, 3],
            vec![1, 3],
            vec![0, 2, 3],
            vec![0, 1, 2, 3],
            vec![3],
            vec![0, 1, 2, 3],
            vec![1, 2, 3],
            vec![0, 1, 2, 3],
        ];
        let mask = veda_int8::Mask::new(&rows, &valid, heads).unwrap();
        let slots = mask.slots();
        let n = heads * slots * dim;
        let make = |a: usize, b: usize, c: usize| {
            Tensor::from_vec(
                (0..n)
                    .map(|i| ((i * a % b) as f32 - (b / 2) as f32) / c as f32)
                    .collect::<Vec<_>>(),
                (heads, slots, dim),
                dev,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
        };
        let (q, k, v) = (make(17, 101, 40), make(31, 103, 40), make(13, 97, 60));
        let scale = 1. / (dim as f32).sqrt();
        let mut v1 = veda_int8::SOURCE;
        v1.name = "cuda-veda-sage-int8-pointer-v1";
        v1.src = include_str!("veda_pointer_v1.cu");
        let stream = cuda.cuda_stream();
        let mut outputs = Vec::new();
        for (source, rows_per_cta) in [(veda_int8::SOURCE, 32), (v1, 1)] {
            let module = source.compiled(cuda).unwrap();
            let quant = module.function("veda_quant").unwrap();
            let attention = module.function("veda_attend").unwrap();
            let mut qi = unsafe { cuda.alloc::<u8>(n) }.unwrap();
            let mut ki = unsafe { cuda.alloc::<u8>(n) }.unwrap();
            let mut qs = unsafe { cuda.alloc::<f32>(heads * slots / 128) }.unwrap();
            let mut ks = unsafe { cuda.alloc::<f32>(heads * slots / 64) }.unwrap();
            let mut out = unsafe { cuda.alloc::<f32>(n) }.unwrap();
            let offsets = stream.clone_htod(&mask.offsets).unwrap();
            let indices = stream.clone_htod(&mask.indices).unwrap();
            let valid = stream.clone_htod(&mask.valid).unwrap();
            let f32s: Vec<Tensor> = [&q, &k, &v]
                .iter()
                .map(|t| t.to_dtype(DType::F32).unwrap().contiguous().unwrap())
                .collect();
            let storages: Vec<_> = f32s.iter().map(|t| t.storage_and_layout()).collect();
            let views: Vec<_> = storages
                .iter()
                .map(|(s, l)| {
                    let Storage::Cuda(s) = &**s else {
                        unreachable!()
                    };
                    let s: &CudaStorage = s;
                    s.as_cuda_slice::<f32>().unwrap().slice(l.start_offset()..)
                })
                .collect();
            let (sl, d, qb, kb) = (slots as i32, dim as i32, 128i32, 64i32);
            let (pre, one) = (scale * veda_int8::LOG2E, 1f32);
            let cfg = |rows: usize| LaunchConfig {
                grid_dim: (rows as u32, heads as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&quant)
                    .arg(&views[0])
                    .arg(&mut qi)
                    .arg(&mut qs)
                    .arg(&sl)
                    .arg(&d)
                    .arg(&qb)
                    .arg(&pre)
                    .launch(cfg(slots / 128))
                    .unwrap();
                stream
                    .launch_builder(&quant)
                    .arg(&views[1])
                    .arg(&mut ki)
                    .arg(&mut ks)
                    .arg(&sl)
                    .arg(&d)
                    .arg(&kb)
                    .arg(&one)
                    .launch(cfg(slots / 64))
                    .unwrap();
                stream
                    .launch_builder(&attention)
                    .arg(&qi)
                    .arg(&ki)
                    .arg(&views[2])
                    .arg(&qs)
                    .arg(&ks)
                    .arg(&offsets)
                    .arg(&indices)
                    .arg(&valid)
                    .arg(&mut out)
                    .arg(&sl)
                    .arg(&d)
                    .launch(cfg(slots / rows_per_cta))
                    .unwrap();
            }
            outputs.push(stream.clone_dtoh(&out).unwrap());
        }
        let mut failures = Vec::new();
        let label = format!("{dtype:?} D{dim}");
        if let Some(i) = (0..n).find(|&i| outputs[0][i].to_bits() != outputs[1][i].to_bits()) {
            failures.push(format!(
                "{label}: tiled kernel differs from pointer-v1 at {i}: {:e} vs {:e}",
                outputs[0][i], outputs[1][i]
            ));
        }
        let host = |t: Tensor| {
            t.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        };
        let production = host(veda_int8::attend(&q, &k, &v, &mask, scale).unwrap());
        let harness = Tensor::from_vec(outputs[0].clone(), (heads, slots, dim), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        if production != host(harness) {
            failures.push(format!(
                "{label}: production attend differs from the tiled launch"
            ));
        }
        failures
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
            // Exactness checks collect every failure before any tolerance assertion, so one run
            // names all of them.
            let mut failures = Vec::new();
            for dim in [16, 128] {
                failures.extend(tiled_kernel_bit_failures(&dev, dtype, dim));
            }
            // Batched device scoring/selection against single heads and the host oracle,
            // including a many-head D128 plan.
            for (bundle, heads, dim) in [
                (predictor(2, 2, 16), 2, 16),
                (mixed_predictor(), 2, 16),
                (predictor(2, 12, 128), 12, 128),
            ] {
                let r = runtime_large(&dev, ReportSink::default(), bundle);
                let n = r
                    .prepared
                    .as_ref()
                    .unwrap()
                    .layouts
                    .values()
                    .next()
                    .unwrap()
                    .seq;
                let (q, k) = (
                    tensor_hd(n, heads, dim, &dev, dtype, 0),
                    tensor_hd(n, heads, dim, &dev, dtype, 3),
                );
                failures.extend(
                    batched_scoring_failures(&r, &q, &k)
                        .into_iter()
                        .map(|f| format!("{dtype:?} H{heads} D{dim} {f}")),
                );
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
            let host = |t: Tensor| {
                t.to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            };
            // Same tolerance shape as the first-use self-test: INT8 quantization bound plus
            // two output ULPs and FP32 reduction roundoff. The masks themselves are exact.
            let ulp = if dtype == DType::BF16 {
                1. / 128.
            } else {
                1. / 1024.
            };
            let check = |label: &str,
                         runtime: &VedaRuntime,
                         got: Tensor,
                         (want, bound, masks): (Vec<f32>, f32, Masks),
                         failures: &mut Vec<String>| {
                if runtime_masks(runtime) != masks {
                    failures.push(format!(
                        "{label} {dtype:?}: runtime masks differ from oracle"
                    ));
                }
                assert!(failures.is_empty(), "exactness failures: {failures:#?}");
                let got = host(got);
                assert_eq!(got.len(), want.len());
                assert!(
                    want.iter().any(|b| !b.is_nan()),
                    "{label}: nothing compared"
                );
                for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
                    if b.is_nan() {
                        continue; // head not computed by the oracle
                    }
                    assert!(
                        a.is_finite() && (a - b).abs() <= bound + 2. * ulp * b.abs() + 2e-4,
                        "{label} {dtype:?} {i}: sparse {a} vs masked dense {b} (bound {bound})"
                    );
                }
            };
            let sparse = r.attention(0, 0, &q, &k, &v, 0.25).unwrap();
            let oracle = masked_dense_oracle(&r, 0, &q, &k, &v, 0.25);
            check("t2va", &r, sparse, oracle, &mut failures);
            assert_eq!(report.borrow().sparse_calls, 1);
            // One batched scoring pass per sample: both heads share the fixture's single shape.
            assert_eq!(report.borrow().predictor_score_calls, 2);
            assert!(report.borrow().selected_token_pairs < report.borrow().dense_token_pairs);
            assert_eq!(report.borrow().kernel.as_deref(), Some(KERNEL));
            assert!(r.tested.borrow().is_some());
            r.request.full_attention_steps = vec![1];
            let full = r.attention(0, 1, &q, &k, &v, 0.25).unwrap();
            let dense = crate::nn::sdpa(&q, &k, &v, 0.25).unwrap();
            assert_eq!(host(full), host(dense));
            // FL2VA: keyframe anchors packed as reference segments, with a nonzero reference
            // budget so reference tiles go through learned selection too.
            let fl_report = ReportSink::default();
            let fl = runtime_for(
                &dev,
                fl_report.clone(),
                &[KeyframeAnchor::First, KeyframeAnchor::Last],
                VedaBudget::KeyTiles(1),
            );
            let layout = fl
                .prepared
                .as_ref()
                .unwrap()
                .layouts
                .values()
                .next()
                .unwrap();
            assert!(layout.reference > 0);
            let n = layout.seq;
            let (fq, fk, fv) = (
                tensor(n, &dev, dtype, 1),
                tensor(n, &dev, dtype, 5),
                tensor(n, &dev, dtype, 9),
            );
            let sparse = fl.attention(1, 0, &fq, &fk, &fv, 0.25).unwrap();
            let oracle = masked_dense_oracle(&fl, 1, &fq, &fk, &fv, 0.25);
            check("fl2va", &fl, sparse, oracle, &mut failures);
            assert_eq!(fl_report.borrow().sparse_calls, 1);
            assert!(fl_report.borrow().fallback_reason.is_none());
            // Mixed tile shapes: per-shape head batches launch out of head order and are
            // reassembled; every head must still match its own masked dense oracle.
            let mixed_report = ReportSink::default();
            let mixed = runtime_with(
                &dev,
                mixed_report.clone(),
                &[],
                VedaBudget::SparsityPercent(0.),
                mixed_predictor(),
            );
            let sparse = mixed.attention(0, 0, &q, &k, &v, 0.25).unwrap();
            let oracle = masked_dense_oracle(&mixed, 0, &q, &k, &v, 0.25);
            check("mixed", &mixed, sparse, oracle, &mut failures);
            assert_eq!(mixed_report.borrow().sparse_calls, 1);
            assert_eq!(mixed_report.borrow().predictor_score_calls, 4);
            assert!(mixed_report.borrow().fallback_reason.is_none());
            // Many heads at production width: twelve D128 heads through the batched route.
            let many_report = ReportSink::default();
            let many = runtime_large(&dev, many_report.clone(), predictor(2, 12, 128));
            let n = many
                .prepared
                .as_ref()
                .unwrap()
                .layouts
                .values()
                .next()
                .unwrap()
                .seq;
            let (mq, mk, mv) = (
                tensor_hd(n, 12, 128, &dev, dtype, 2),
                tensor_hd(n, 12, 128, &dev, dtype, 4),
                tensor_hd(n, 12, 128, &dev, dtype, 6),
            );
            let scale = 1. / 128f32.sqrt();
            let sparse = many.attention(1, 0, &mq, &mk, &mv, scale as f64).unwrap();
            // Every head's mask is compared exactly; the host FP32 attention (~13 s per head in a
            // debug build) is computed for the first head of sample 0 and the last of sample 1.
            let oracle = masked_dense_oracle_for(&many, 1, &mq, &mk, &mv, scale, |b, h| {
                (b, h) == (0, 0) || (b, h) == (1, 11)
            });
            check("many", &many, sparse, oracle, &mut failures);
            assert_eq!(many_report.borrow().sparse_calls, 1);
            assert!(many_report.borrow().fallback_reason.is_none());
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
            // A non-finite sparse output (NaN V: the scores read Q/K only) is recomputed densely
            // and reported, never returned into the hidden state.
            let nan_report = ReportSink::default();
            let nan = runtime(&dev, nan_report.clone());
            let nan_v = (v.ones_like().unwrap() * f64::NAN).unwrap();
            let _ = nan.attention(0, 0, &q, &k, &nan_v, 0.25).unwrap();
            assert_eq!(nan_report.borrow().sparse_calls, 0);
            assert_eq!(nan_report.borrow().dense_calls, 1);
            assert!(nan_report
                .borrow()
                .fallback_reason
                .as_ref()
                .unwrap()
                .contains("non-finite output"));
            let Device::Cuda(cuda) = &dev else {
                unreachable!()
            };
            println!(
                "S3_DEVICE_RECEIPT {}",
                serde_json::json!({
                    "schema":"sc-25777-actual-cuda-kernel-v2", "kernel":KERNEL,
                    "device_ordinal":cuda.cuda_stream().context().ordinal(),
                    "dtype":format!("{dtype:?}"), "self_test_dimensions":[16,128],
                    "native_batch":2, "native_heads":2,
                    "sparse_calls":report.borrow().sparse_calls,
                    "predictor_score_batches":report.borrow().predictor_score_calls,
                    "exact_scorer_and_masks":true, "tiled_kernel_bits_equal_pointer_v1":true,
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
        // Full denominator uses real rows across generated/reference/global segments, never padded slots.
        let valid = l.valid.iter().sum::<usize>() as u64;
        assert_eq!(valid, 161);
        let pairs = selected
            .iter()
            .enumerate()
            .map(|(q, keys)| {
                l.valid[q] as u64 * keys.iter().map(|&k| l.valid[k] as u64).sum::<u64>()
            })
            .sum::<u64>();
        let global_rows = globals.iter().map(|&t| l.valid[t] as u64).sum::<u64>();
        let global_pairs = global_rows * valid;
        assert!(pairs >= global_pairs);
        assert!(pairs < valid * valid);
        let report = VedaExecutionReport {
            sparse_calls: 1,
            dense_calls: 2,
            selected_token_pairs: pairs + 2 * valid * valid,
            dense_token_pairs: 3 * valid * valid,
            ..Default::default()
        };
        assert_eq!(
            report.total_attention_fraction(),
            Some((pairs + 2 * 161 * 161) as f64 / (3 * 161 * 161) as f64)
        );
    }
    #[test]
    fn ensure_finite_refuses_nan_and_inf_attention_outputs() {
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let ok = Tensor::new(&[[0.5f32, -2.0], [3.0, 0.0]], &Device::Cpu)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            ensure_finite(&ok).unwrap();
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let t = Tensor::new(&[[0.5f32, bad], [3.0, 0.0]], &Device::Cpu)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let e = ensure_finite(&t).unwrap_err().to_string();
                assert!(e.contains("non-finite output"), "{dtype:?} {bad}: {e}");
            }
        }
    }
    #[test]
    fn actual_device_self_test_cannot_pass_on_cpu() {
        assert!(device_self_test(&Device::Cpu, DType::BF16, 128).is_err());
    }
}
