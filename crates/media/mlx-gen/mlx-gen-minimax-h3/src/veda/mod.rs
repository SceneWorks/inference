//! Request-scoped MLX Veda route. Dense remains the default and fallback.
//! Numerical and batching rules ported from Veda-on-ComfyUI at
//! `60bfae688897dc41bc8685ca9f3130d7ed8f3b0c`; upstream MIT notice: `LICENSE.upstream`.
pub mod contract;
#[cfg(test)]
mod device_gate;
pub mod metal;
use crate::denoise::PackedLayout;
use contract::{Budget, Bundle, Layout, Plan, TileShape, TILE};
use mlx_gen::attention::{sdpa_bounded_bhsd, slice_axis, BoundedAttention};
use mlx_gen::gen_core::{VedaBudget, VedaExecutionReport, VedaRequest};
use mlx_gen::{Error, Result};
use mlx_rs::{
    ops::{concatenate_axis, r#where},
    Array, Dtype,
};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    path::Path,
    rc::Rc,
    sync::{Arc, Mutex},
};

pub type ReportSink = Rc<RefCell<VedaExecutionReport>>;
/// One executed head's selected rows: `(layer, head, rows)`.
#[cfg(test)]
pub(crate) type ExecutedMask = (usize, usize, Vec<Vec<usize>>);
use mlx_gen::gen_core::veda::workspace::{self, Backend, Chunk, LayoutSize, Limits};
/// Metal execution (sc-26214): the exact device tile scorer plus [`metal::attend`]'s block-sparse
/// attention over selected key tiles read in place, and one dense call for the global query rows.
/// FP32 `simdgroup_matrix` products, on a GPU without MLX's neural-accelerator kernels.
pub const KERNEL: &str = "mlx-tile-sparse-metal-v2";
/// The same route with `matmul2d` products on the GPU's neural accelerators (16-bit Q/K/V).
pub const NAX_KERNEL: &str = "mlx-tile-sparse-nax-v2";
/// MLX CPU device execution (tests, no Metal): the host scorer contract and gather + SDPA.
pub const CPU_KERNEL: &str = "mlx-batched-gather-sdpa-v1";

fn on_gpu() -> Result<bool> {
    Ok(matches!(
        mlx_rs::Device::try_default()?.get_type()?,
        mlx_rs::DeviceType::Gpu
    ))
}
/// The one sparse kernel route of the current device (E4): its self-test cache key and report id.
pub(crate) fn kernel_id() -> Result<&'static str> {
    Ok(if !on_gpu()? {
        CPU_KERNEL
    } else if metal::nax_available() {
        NAX_KERNEL
    } else {
        KERNEL
    })
}
pub const SELF_TEST_ATOL: f32 = 0.025;

fn predictor_path(path: Option<&Path>) -> Result<&Path> {
    let path = path.ok_or_else(|| {
        Error::Msg(
            "Veda predictor component is missing; repair/install the managed component".into(),
        )
    })?;
    // Diagnose a missing managed file before layout work. This probe reads no weights and
    // grants no load authority: the bounded loader still opens, sizes and hashes one buffer.
    std::fs::metadata(path).map_err(|e| {
        Error::Msg(format!(
            "Veda predictor {}: {e}; repair/install the managed component",
            path.display()
        ))
    })?;
    Ok(path)
}

/// Process-wide first-use self-test outcomes keyed on (device type, device index, kernel id).
/// A pass or a failure is decided once per actual device and kernel, not once per request.
type SelfTestKey = (u32, i32, &'static str);
type SelfTestCache = Mutex<BTreeMap<SelfTestKey, std::result::Result<(), String>>>;
static SELF_TESTS: SelfTestCache = Mutex::new(BTreeMap::new());

fn cached_self_test(
    cache: &SelfTestCache,
    key: SelfTestKey,
    run: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let mut cache = cache.lock().unwrap_or_else(|e| e.into_inner());
    cache
        .entry(key)
        .or_insert_with(|| run().map_err(|e| e.to_string()))
        .clone()
        .map_err(Error::Msg)
}

/// The production first-use check: [`device_self_test`] once per process per default MLX
/// device and [`KERNEL`].
pub fn cached_device_self_test() -> Result<()> {
    let device = mlx_rs::Device::try_default()?;
    let key = (
        u32::from(device.get_type()?),
        device.get_index()?,
        kernel_id()?,
    );
    cached_self_test(&SELF_TESTS, key, device_self_test)
}

pub struct VedaRuntime {
    bundle: Arc<Bundle>,
    plan: Plan,
    grid: [usize; 3],
    layouts: BTreeMap<TileShape, Layout>,
    /// Device-resident permutation/validity/assembly operands per tile shape (Metal only).
    device: BTreeMap<TileShape, metal::DeviceLayout>,
    /// Device-resident BF16 `[heads, 3*dim, dim]` (Q, K) projections per layer, uploaded on the
    /// layer's first sparse call and kept for every later step.
    projections: RefCell<BTreeMap<usize, (Array, Array)>>,
    limits: Limits,
    request: VedaRequest,
    report: ReportSink,
    failed: RefCell<bool>,
    #[cfg(test)]
    pub(crate) calls: RefCell<Vec<(usize, usize, VedaRequest)>>,
    /// Every executed head's selected rows as `(layer, head, rows)`, in execution order.
    #[cfg(test)]
    pub(crate) masks: RefCell<Vec<ExecutedMask>>,
    /// Accumulated wall seconds per execution phase (device gate bench).
    #[cfg(test)]
    pub(crate) profile: RefCell<BTreeMap<&'static str, f64>>,
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
            let path = predictor_path(path)?;
            let g = packed.geometry();
            let grid = [
                g.num_latent_frames as usize,
                (g.latent_height / 2) as usize,
                (g.latent_width / 2) as usize,
            ];
            let plan = workspace::frozen_plan(grid)?;
            let sizes = plan
                .shapes
                .iter()
                .map(|&s| packed.veda_layout(s).map(|l| (s, LayoutSize::from(&l))))
                .collect::<Result<BTreeMap<_, _>>>()?;
            // Account loading even when checkpoint integrity/decode fails.
            let initial = workspace::bound(
                request,
                Backend::Mlx,
                &plan,
                grid,
                &sizes,
                layers,
                heads,
                dim,
                1,
                2,
                Limits::for_backend(Backend::Mlx),
            )?;
            report.borrow_mut().workspace_bytes = initial.total_bytes;
            let bundle = Bundle::load_cached(path, layers, heads, dim)?;
            Self::from_bundle(
                bundle,
                request,
                packed,
                report.clone(),
                cached_device_self_test,
            )
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
        bundle: Arc<Bundle>,
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
        let sizes = layouts
            .iter()
            .map(|(&s, l)| (s, LayoutSize::from(l)))
            .collect();
        let bound = workspace::bound(
            request,
            Backend::Mlx,
            &plan,
            grid,
            &sizes,
            bundle.layers,
            bundle.heads,
            bundle.dim,
            1,
            2,
            Limits::for_backend(Backend::Mlx),
        )?;
        report.borrow_mut().workspace_bytes = bound.total_bytes;
        // Actual-device check precedes sparse production calls. Failure leaves this request dense.
        {
            let mut r = report.borrow_mut();
            r.predictor = Some(bundle.identity.clone());
            r.plan = Some(plan.name.clone());
            r.exact_plan = plan.grid == grid;
            r.execution_grid = Some(grid);
            r.plan_grid = Some(plan.grid);
            r.dense_segments = packed.veda_dense_segments();
        }
        if let Err(e) = self_test() {
            // No sparse call will run under this plan; do not report it as the executed plan.
            let mut r = report.borrow_mut();
            r.plan = None;
            r.plan_grid = None;
            r.exact_plan = false;
            return Err(e);
        }
        let device = layouts
            .iter()
            .map(|(&s, l)| (s, metal::DeviceLayout::new(l)))
            .collect();
        Ok(Self {
            bundle,
            plan,
            grid,
            layouts,
            device,
            projections: Default::default(),
            limits: Limits::for_backend(Backend::Mlx),
            request: request.clone(),
            report,
            failed: RefCell::new(false),
            #[cfg(test)]
            calls: Default::default(),
            #[cfg(test)]
            masks: Default::default(),
            #[cfg(test)]
            profile: Default::default(),
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
            Arc::new(crate::veda_fixture::predictor()),
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
        let result = self.execute(layer, q, k, v, bounded);
        match result {
            Ok(out) => {
                let kernel = kernel_id()?;
                let mut report = self.report.borrow_mut();
                report.sparse_calls += 1;
                report.kernel = Some(kernel.into());
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

    /// Device-resident BF16 `[heads, 3*dim, dim]` Q and K projections of `layer`.
    fn projection(&self, layer: usize) -> Result<(Array, Array)> {
        if let Some(p) = self.projections.borrow().get(&layer) {
            return Ok(p.clone());
        }
        let (h, d) = (self.bundle.heads as i32, self.bundle.dim as i32);
        let all: Vec<i32> = (0..h).collect();
        // Projections are stored BF16 (E2): the f32 values below are exactly representable.
        let upload = |query: bool| -> Result<Array> {
            Ok(
                Array::from_slice(&self.bundle.projection(layer, &all, query), &[h, 3 * d, d])
                    .as_dtype(Dtype::Bfloat16)?,
            )
        };
        let p = (upload(true)?, upload(false)?);
        mlx_rs::transforms::eval([&p.0, &p.1])?;
        self.projections.borrow_mut().insert(layer, p.clone());
        Ok(p)
    }

    fn execute(
        &self,
        layer: usize,
        q: &Array,
        k: &Array,
        v: &Array,
        bounded: BoundedAttention<'_>,
    ) -> Result<Array> {
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
        let heads = self.bundle.heads;
        let seq = q.shape()[2] as usize;
        let bytes = match q.dtype() {
            Dtype::Bfloat16 | Dtype::Float16 => 2,
            Dtype::Float32 => 4,
            _ => return Err(Error::Msg("Veda unsupported compute dtype".into())),
        };
        let sizes = self
            .layouts
            .iter()
            .map(|(&s, l)| (s, LayoutSize::from(l)))
            .collect();
        let bound = workspace::bound(
            &self.request,
            Backend::Mlx,
            &self.plan,
            self.grid,
            &sizes,
            self.bundle.layers,
            heads,
            dim,
            1,
            bytes as usize,
            self.limits,
        )?;
        {
            let mut r = self.report.borrow_mut();
            r.workspace_bytes = r.workspace_bytes.max(bound.total_bytes);
        }
        // Same-shape head batches (ascending heads), each at most the bound's batch size. A
        // refused shape fails before any scorer or attention allocation.
        let mut grouped: BTreeMap<TileShape, Vec<i32>> = BTreeMap::new();
        for (h, &si) in self.plan.heads[layer].iter().enumerate() {
            grouped
                .entry(self.plan.shapes[si])
                .or_default()
                .push(h as i32);
        }
        let mut batches = Vec::new();
        for (shape, group) in grouped {
            let reservation = &bound.chunks[&shape];
            if let Some(reason) = &reservation.refusal {
                return Err(Error::Msg(format!("Veda {reason}")));
            }
            batches.extend(group.chunks(reservation.heads).map(|h| (shape, h.to_vec())));
        }
        let gpu = on_gpu()?;
        if gpu && dim != metal::HEAD_DIM {
            return Err(Error::Msg(format!(
                "Veda Metal kernel supports head dim {}, not {dim}",
                metal::HEAD_DIM
            )));
        }
        // The DiT's Q/K/V are head-transposed views of native `[seq, heads, dim]` rows, which is
        // what the Metal kernels index (through each layout's permutation), so no tile copy exists.
        let native = |a: &Array| -> Result<Array> {
            Ok(a.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[seq as i32, heads as i32, dim as i32])?)
        };
        let natives = if gpu {
            Some((native(q)?, native(k)?, native(v)?, self.projection(layer)?))
        } else {
            None
        };
        let budgets = (
            budget(&self.request.generated),
            budget(&self.request.reference),
        );
        let mut pieces = Vec::with_capacity(batches.len());
        let mut head_order: Vec<i32> = Vec::with_capacity(heads);
        let mut pairs = 0u64;
        #[cfg(test)]
        let lap = |name: &'static str, from: std::time::Instant| {
            *self.profile.borrow_mut().entry(name).or_default() += from.elapsed().as_secs_f64();
            std::time::Instant::now()
        };
        #[cfg(test)]
        let mut at = std::time::Instant::now();
        for (shape, group) in &batches {
            let layout = &self.layouts[shape];
            let scores = match &natives {
                Some((nq, nk, _, (pq, pk))) => {
                    metal::scores(&self.device[shape], nq, nk, pq, pk, group, 0, dim)?
                }
                None => host_scores(&self.bundle, layer, group, layout, q, k)?,
            };
            self.report.borrow_mut().predictor_score_calls += 1;
            #[cfg(test)]
            {
                at = lap("score", at);
            }
            let rows = select(layout, &scores, group.len(), budgets)?;
            #[cfg(test)]
            {
                at = lap("select", at);
            }
            #[cfg(test)]
            self.masks.borrow_mut().extend(
                group
                    .iter()
                    .zip(rows.chunks(layout.valid.len()))
                    .map(|(&h, r)| (layer, h as usize, r.to_vec())),
            );
            pairs += rows
                .iter()
                .enumerate()
                .map(|(r, keys)| {
                    layout.valid[r % layout.valid.len()] as u64
                        * keys.iter().map(|&k| layout.valid[k] as u64).sum::<u64>()
                })
                .sum::<u64>();
            let out = match &natives {
                Some((nq, nk, nv, _)) => metal_batch(
                    &self.device[shape],
                    group,
                    &rows,
                    [q, k, v],
                    [nq, nk, nv],
                    dim,
                    bounded,
                )?,
                None => cpu_batch(
                    layout,
                    group,
                    &rows,
                    [q, k, v],
                    dim,
                    bytes,
                    &bound.chunks[shape],
                )?,
            };
            // Materialize each bounded batch output before the next batch's scorer graph (E8).
            mlx_rs::transforms::eval([&out])?;
            #[cfg(test)]
            {
                at = lap("attend", at);
            }
            pieces.push(out);
            head_order.extend(group);
        }
        let refs: Vec<&Array> = pieces.iter().collect();
        let mut out = concatenate_axis(&refs, 0)?;
        if head_order.iter().enumerate().any(|(i, &h)| i as i32 != h) {
            let mut inverse = vec![0i32; heads];
            for (i, &h) in head_order.iter().enumerate() {
                inverse[h as usize] = i as i32;
            }
            out = out.take_axis(Array::from_slice(&inverse, &[heads as i32]), 0)?;
        }
        let out = out.reshape(&[1, heads as i32, seq as i32, dim as i32])?;
        mlx_rs::transforms::eval([&out])?;
        ensure_finite(&out)?;
        let mut report = self.report.borrow_mut();
        report.selected_token_pairs += pairs;
        report.dense_token_pairs += heads as u64 * seq as u64 * seq as u64;
        Ok(out)
    }
}
fn budget(b: &VedaBudget) -> Budget {
    match b {
        VedaBudget::SparsityPercent(p) => Budget::Ratio(1.0 - p / 100.0),
        VedaBudget::KeyTiles(n) => Budget::Tiles(*n as f64),
    }
}
/// The frozen host selection of one batch (head-major rows). Heads select independently (head `h`
/// reads only its own `[video, video]` scores), so contiguous head ranges run in parallel and are
/// concatenated in head order: identical to one `select_with_budgets` over every head.
fn select(
    layout: &Layout,
    scores: &[f32],
    heads: usize,
    (generated, reference): (Budget, Budget),
) -> Result<Vec<Vec<usize>>> {
    let per_head = layout.video * layout.video;
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
    let span = heads.div_ceil(workers.clamp(1, heads.max(1)));
    if per_head == 0 || scores.len() != heads * per_head || span >= heads {
        return Ok(layout.select_with_budgets(scores, heads, generated, reference)?);
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
    let mut rows = Vec::with_capacity(heads * layout.valid.len());
    for part in parts {
        rows.extend(part?);
    }
    Ok(rows)
}
/// One batch on Metal: video query tiles through [`metal::attend`], the global query rows as one
/// dense attention over every real key (frozen dense globals), assembled to `[heads, seq, dim]`.
fn metal_batch(
    dev: &metal::DeviceLayout,
    heads: &[i32],
    rows: &[Vec<usize>],
    [q, k, v]: [&Array; 3],
    [nq, nk, nv]: [&Array; 3],
    dim: usize,
    bounded: BoundedAttention<'_>,
) -> Result<Array> {
    let tiled = metal::attend(dev, nq, nk, nv, heads, rows)?;
    let combined = match &dev.globals {
        Some(globals) => {
            let all = heads.len() == q.shape()[1] as usize
                && heads.iter().enumerate().all(|(i, &h)| i as i32 == h);
            let ids = Array::from_slice(heads, &[heads.len() as i32]);
            let pick = |a: &Array| -> Result<Array> {
                Ok(if all {
                    a.clone()
                } else {
                    a.take_axis(&ids, 1)?
                })
            };
            let (qb, kb, vb) = (pick(q)?, pick(k)?, pick(v)?);
            let query = qb.take_axis(globals, 2)?;
            let rows = globals.shape()[0];
            let out =
                sdpa_bounded_bhsd(&query, &kb, &vb, 1.0 / (dim as f32).sqrt(), None, bounded)?
                    .reshape(&[heads.len() as i32, rows, dim as i32])?;
            concatenate_axis(&[&tiled, &out], 1)?
        }
        None => tiled,
    };
    Ok(combined.take_axis(&dev.compact, 1)?)
}
/// One batch on the MLX CPU device: padded tile gathers and the gather + SDPA reference route.
#[allow(clippy::too_many_arguments)]
fn cpu_batch(
    layout: &Layout,
    heads: &[i32],
    rows: &[Vec<usize>],
    [q, k, v]: [&Array; 3],
    dim: usize,
    bytes: u64,
    reservation: &Chunk,
) -> Result<Array> {
    let (qt, kt, vt) = (
        tile(q, heads, layout, dim)?,
        tile(k, heads, layout, dim)?,
        tile(v, heads, layout, dim)?,
    );
    Ok(gather_attention(layout, rows, &qt, &kt, &vt, dim, bytes, reservation)?.0)
}
/// Gathers `heads` of `[1, H, seq, dim]` into padded tile order `[heads, tiles, TILE, dim]`.
fn tile(a: &Array, heads: &[i32], layout: &Layout, dim: usize) -> Result<Array> {
    let (n, seq, slots) = (heads.len() as i32, a.shape()[2], layout.perm.len() as i32);
    let perm: Vec<i32> = layout.perm.iter().map(|&r| r.max(0)).collect();
    let valid: Vec<bool> = layout.perm.iter().map(|&r| r >= 0).collect();
    let a = a
        .take_axis(Array::from_slice(heads, &[n]), 1)?
        .reshape(&[n, seq, dim as i32])?
        .take_axis(Array::from_slice(&perm, &[slots]), 1)?;
    Ok(r#where(
        Array::from_slice(&valid, &[1, slots, 1]),
        &a,
        &Array::from_f32(0.0).as_dtype(a.dtype())?,
    )?
    .reshape(&[n, layout.valid.len() as i32, TILE as i32, dim as i32])?)
}
/// gen-core's host scorer contract itself (`pool_video_tiles` + `tile_logits`).
fn host_scores(
    bundle: &Bundle,
    layer: usize,
    heads: &[i32],
    layout: &Layout,
    q: &Array,
    k: &Array,
) -> Result<Vec<f32>> {
    let host = |a: &Array| -> Result<Vec<f32>> {
        let a = tile(a, heads, layout, bundle.dim)?
            .as_dtype(Dtype::Float32)?
            .reshape(&[-1])?;
        mlx_rs::transforms::eval([&a])?;
        Ok(contract::pool_video_tiles(
            a.as_slice::<f32>(),
            layout,
            heads.len(),
            bundle.dim,
        )?)
    };
    Ok(contract::tile_logits(
        &host(q)?,
        &host(k)?,
        &bundle.projection(layer, heads, true),
        &bundle.projection(layer, heads, false),
        heads.len(),
        layout.video,
        bundle.dim,
    )?)
}
/// Refuses a sparse attention output holding any NaN/Inf, so the caller recomputes that call
/// densely and records the reason instead of letting the poison into the hidden state (a poisoned
/// residual decodes to a flat black clip with silent audio). `out` is already materialized, so
/// this is one f32 sum reduction plus a scalar readback per sparse call; a NaN or Inf anywhere
/// makes the sum non-finite.
fn ensure_finite(out: &Array) -> Result<()> {
    let sum = out.as_dtype(Dtype::Float32)?.sum(None)?.item::<f32>();
    if sum.is_finite() {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "Veda sparse attention produced non-finite output (sum {sum})"
        )))
    }
}

#[allow(clippy::too_many_arguments)]
fn gather_attention(
    l: &Layout,
    selected: &[Vec<usize>],
    q: &Array,
    k: &Array,
    v: &Array,
    dim: usize,
    bytes: u64,
    reservation: &Chunk,
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
    if let Some(reason) = &reservation.refusal {
        return Err(Error::Msg(format!("Veda {reason}")));
    }
    if kmax > reservation.selected_keys || h > reservation.heads {
        return Err(Error::Msg(
            "Veda selection exceeds admitted gather bound".into(),
        ));
    }
    let rmax = reservation.query_tiles;
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
    // Globals never join sparse video query batches.
    let global_rows = reservation.global_tiles;
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
    let scratch = reservation.attention_bytes + (l.perm.len() * dim * h) as u64 * bytes * 2;
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
    let reservation = workspace::chunk(
        LayoutSize::from(&l),
        heads,
        dim,
        2,
        Backend::Mlx,
        &VedaRequest {
            enabled: true,
            generated: VedaBudget::KeyTiles(2),
            reference: VedaBudget::KeyTiles(2),
            ..Default::default()
        },
        Limits::for_backend(Backend::Mlx),
    )?;
    // The production route of the current device, on the production batch shape.
    let ids: Vec<i32> = (0..heads as i32).collect();
    let sparse = if on_gpu()? {
        let native = |a: &Array| -> Result<Array> {
            Ok(a.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[seq as i32, heads as i32, dim as i32])?)
        };
        metal_batch(
            &metal::DeviceLayout::new(&l),
            &ids,
            &rows,
            [&q, &k, &v],
            [&native(&q)?, &native(&k)?, &native(&v)?],
            dim,
            BoundedAttention::UNBOUNDED,
        )?
    } else {
        cpu_batch(&l, &ids, &rows, [&q, &k, &v], dim, 2, &reservation)?
    }
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
    // Independent oracle: explicit f32 softmax(QK^T * scale + mask) V on the host, sharing no
    // MLX attention kernel with the sparse path under test.
    let host = |a: &Array| -> Result<Vec<f32>> {
        let a = a.as_dtype(Dtype::Float32)?.reshape(&[-1])?;
        mlx_rs::transforms::eval([&a])?;
        Ok(a.as_slice::<f32>().to_vec())
    };
    let reference = host_masked_attention(
        &host(&q)?,
        &host(&k)?,
        &host(&v)?,
        &mask,
        heads,
        seq,
        dim,
        1.0 / (dim as f32).sqrt(),
    );
    let got = host(&sparse)?;
    // The oracle must be able to tell the selected mask from unmasked dense attention,
    // otherwise a gather that ignored the selection would pass.
    let (qh, kh, vh) = (host(&q)?, host(&k)?, host(&v)?);
    let unmasked = host_masked_attention(
        &qh,
        &kh,
        &vh,
        &vec![true; heads * seq * seq],
        heads,
        seq,
        dim,
        1.0 / (dim as f32).sqrt(),
    );
    let discrimination = reference
        .iter()
        .zip(&unmasked)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    #[cfg(test)]
    println!("Veda self-test discrimination: masked-vs-dense max_abs={discrimination}");
    if discrimination.is_nan() || discrimination <= 4.0 * SELF_TEST_ATOL {
        return Err(Error::Msg(format!(
            "Veda self-test cannot discriminate its mask from dense attention: {discrimination}"
        )));
    }
    let error = got
        .iter()
        .zip(&reference)
        .map(|(a, b)| {
            if a.is_finite() {
                (a - b).abs()
            } else {
                f32::INFINITY
            }
        })
        .fold(0f32, f32::max);
    #[cfg(test)]
    println!("Veda device self-test: kernel={}, D128 BF16, max_abs={error}, tolerance={SELF_TEST_ATOL}, selected_pairs={pairs}, dense_pairs={}",kernel_id()?,heads*seq*seq);
    if !error.is_finite() || error > SELF_TEST_ATOL {
        return Err(Error::Msg(format!(
            "Veda {} actual-device self-test failed: {error} > {SELF_TEST_ATOL}",
            kernel_id()?
        )));
    }
    Ok(())
}

/// Row-major `[heads, seq, dim]` f32 attention with a boolean `[heads, seq, seq]` key mask.
/// A query row with no admitted key is zero.
#[allow(clippy::too_many_arguments)]
fn host_masked_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[bool],
    heads: usize,
    seq: usize,
    dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; heads * seq * dim];
    let mut logits = vec![0f32; seq];
    for h in 0..heads {
        for row in 0..seq {
            let qrow = &q[(h * seq + row) * dim..][..dim];
            let admitted = &mask[(h * seq + row) * seq..][..seq];
            let mut peak = f32::NEG_INFINITY;
            for key in 0..seq {
                logits[key] = if admitted[key] {
                    let krow = &k[(h * seq + key) * dim..][..dim];
                    qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>() * scale
                } else {
                    f32::NEG_INFINITY
                };
                peak = peak.max(logits[key]);
            }
            if peak == f32::NEG_INFINITY {
                continue;
            }
            let mut sum = 0f32;
            let o = &mut out[(h * seq + row) * dim..][..dim];
            for key in (0..seq).filter(|&key| admitted[key]) {
                let p = (logits[key] - peak).exp();
                sum += p;
                let vrow = &v[(h * seq + key) * dim..][..dim];
                for (o, &x) in o.iter_mut().zip(vrow) {
                    *o += p * x;
                }
            }
            for o in o.iter_mut() {
                *o /= sum;
            }
        }
    }
    out
}

#[cfg(test)]
mod frozen_tests {
    use super::*;
    #[test]
    fn host_self_test_oracle_is_masked_softmax_attention() {
        // heads=1, seq=3, dim=2. Row 0 admits keys {0, 2}; row 1 admits only key 1;
        // row 2 admits nothing and is zero.
        let q = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let k = [1.0, 0.0, 0.0, 1.0, 2.0, 0.0];
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = [true, false, true, false, true, false, false, false, false];
        let got = host_masked_attention(&q, &k, &v, &mask, 1, 3, 2, 0.5);
        // Row 0 logits: key0 = 0.5, key2 = 1.0.
        let (p0, p2) = ((0.5f32 - 1.0).exp(), 1.0f32);
        let want0 = [
            (p0 + 5.0 * p2) / (p0 + p2),
            (2.0 * p0 + 6.0 * p2) / (p0 + p2),
        ];
        for (a, b) in got[..2].iter().zip(want0) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        assert_eq!(&got[2..4], &[3.0, 4.0]);
        assert_eq!(&got[4..], &[0.0, 0.0]);
    }
    #[test]
    fn failed_self_test_reports_no_plan_and_records_dense_segments() {
        use crate::denoise::{JointGeometry, PackedLayout, TEXT_TAG};
        use crate::dit::positions::ReferenceLatentGeometry;
        use crate::reference::ReferenceKind;
        let refs = [ReferenceLatentGeometry {
            kind: ReferenceKind::Audio,
            num_latent_frames: 0,
            latent_height: 0,
            latent_width: 0,
            num_audio_latents: 9,
        }];
        let packed = PackedLayout::build_ref2va(
            JointGeometry::new(124, 4, 6).unwrap(),
            [1, 2, 2],
            &[TEXT_TAG; 5],
            2,
            &refs,
        )
        .unwrap();
        let request = VedaRequest {
            enabled: true,
            ..Default::default()
        };
        let report = ReportSink::default();
        assert!(VedaRuntime::prepare_fixture(&request, &packed, report.clone(), true).is_none());
        let r = report.borrow();
        assert!(r
            .fallback_reason
            .as_deref()
            .is_some_and(|e| e.contains("self-test forced failure")));
        assert!(r.plan.is_none() && r.plan_grid.is_none() && !r.exact_plan);
        assert!(r.outcome_json()["planMatch"].is_null());
        assert_eq!(r.dense_segments, packed.veda_dense_segments());
        assert!(r.dense_segments[0].starts_with("reference audio"));
    }
    /// A sparse call whose output is non-finite must not reach the hidden state: that call is
    /// recomputed densely, it is not counted as sparse, and the reason is reported. Runs on the
    /// MLX **CPU** device with the fixture predictor; NaN values poison only the attention output
    /// (the predictor scores read Q/K alone), exactly the CUDA A/B failure's signature.
    #[test]
    fn non_finite_sparse_output_falls_back_to_dense_and_is_reported() {
        use crate::denoise::{JointGeometry, PackedLayout, TEXT_TAG};
        struct Cpu;
        impl Drop for Cpu {
            fn drop(&mut self) {
                mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
            }
        }
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        let _restore = Cpu;
        let packed = PackedLayout::build(
            JointGeometry::new(124, 4, 6).unwrap(),
            [1, 2, 2],
            &[TEXT_TAG; 5],
            2,
            &[],
        )
        .unwrap();
        let request = VedaRequest {
            enabled: true,
            ..Default::default()
        };
        let seq = packed.seq_len();
        let shape = [1, 4, seq, 128];
        let n = (4 * seq * 128) as usize;
        let ramp = |phase: f32| -> Array {
            let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37 + phase).sin()).collect();
            Array::from_slice(&x, &shape)
        };
        let (q, k, finite_v) = (ramp(0.0), ramp(1.0), ramp(2.0));
        let poisoned_v = Array::from_slice(&vec![f32::NAN; n], &shape);

        let run = |v: &Array| {
            let report = ReportSink::default();
            let runtime = VedaRuntime::prepare_fixture(&request, &packed, report.clone(), false)
                .expect("fixture Veda runtime");
            runtime
                .attention(1, 1, &q, &k, v, BoundedAttention::UNBOUNDED)
                .unwrap();
            let r = report.borrow().clone();
            r
        };
        let healthy = run(&finite_v);
        assert_eq!((healthy.sparse_calls, healthy.dense_calls), (1, 0));
        assert!(healthy.fallback_reason.is_none());

        let poisoned = run(&poisoned_v);
        assert_eq!((poisoned.sparse_calls, poisoned.dense_calls), (0, 1));
        assert!(
            poisoned
                .fallback_reason
                .as_deref()
                .is_some_and(|e| e.contains("non-finite output")),
            "{:?}",
            poisoned.fallback_reason
        );
        assert!(poisoned.kernel.is_none());
    }
    #[test]
    fn self_test_outcome_is_cached_per_device_and_kernel() {
        let cache: SelfTestCache = Mutex::new(BTreeMap::new());
        let runs = std::cell::Cell::new(0);
        let pass = || {
            runs.set(runs.get() + 1);
            Ok(())
        };
        let fail = || {
            runs.set(runs.get() + 1);
            Err(Error::Msg("forced self-test failure".into()))
        };
        let gpu = (1, 0, KERNEL);
        cached_self_test(&cache, gpu, pass).unwrap();
        cached_self_test(&cache, gpu, fail).unwrap();
        assert_eq!(runs.get(), 1, "same device/kernel must not re-run");
        // A failure is an outcome too: cached, and reported on every later request.
        let other = (1, 1, KERNEL);
        assert!(cached_self_test(&cache, other, fail).is_err());
        let again = cached_self_test(&cache, other, pass)
            .unwrap_err()
            .to_string();
        assert!(again.contains("forced self-test failure"));
        assert_eq!(runs.get(), 2);
        cached_self_test(&cache, (1, 0, "another-kernel"), pass).unwrap();
        assert_eq!(runs.get(), 3);
    }
    /// The full first-use check on the MLX **CPU** device: proves the host oracle and the
    /// sparse gather agree within the declared tolerance without admitting a Metal device.
    #[test]
    fn device_self_test_passes_on_the_mlx_cpu_device() {
        struct Cpu;
        impl Drop for Cpu {
            fn drop(&mut self) {
                mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
            }
        }
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        let _restore = Cpu;
        device_self_test().unwrap();
    }
    #[test]
    fn managed_component_probe_preserves_bounded_load_authority() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("predictor.safetensors");
        let absent = predictor_path(None).unwrap_err().to_string();
        assert!(absent.contains("repair/install the managed component"));
        let missing = predictor_path(Some(&path)).unwrap_err().to_string();
        assert!(missing.contains(&path.display().to_string()));
        assert!(missing.contains("repair/install the managed component"));
        // A metadata probe cannot bless an oversized checkpoint or bypass workspace bounds.
        std::fs::File::create(&path)
            .unwrap()
            .set_len(275_415_649)
            .unwrap();
        assert_eq!(predictor_path(Some(&path)).unwrap(), path);
        let error = Bundle::load(&path, 50, 56, 128)
            .err()
            .expect("oversized component must fail its authoritative load")
            .to_string();
        assert!(error.contains("size mismatch"));
        let plan = workspace::frozen_plan([37, 24, 42]).unwrap();
        let request = VedaRequest {
            enabled: true,
            ..Default::default()
        };
        let error = workspace::bound(
            &request,
            Backend::Mlx,
            &plan,
            plan.grid,
            &BTreeMap::new(),
            2,
            4,
            128,
            1,
            2,
            Limits::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("empty problem"));
    }
}
