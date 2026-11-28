//! Sol-Attn sparse block-routed attention (SW-6137, feature SW-F287 S12): an **opt-in, lossy**
//! training-free sparse attention for long video self-attention.
//!
//! A port of comfy-kitchen's `sol_attn` CUDA family (Apache-2.0, Comfy Org; the Sol-Attn
//! algorithm of arXiv 2607.24027), flattened into one nvrtc source (`sol_attention.cu`, after the
//! shared `int8_attention_common.cuh` primitives it shares with the SW-6131 INT8 QK kernel).
//! Attribution is in that file's header, `sol_attention-NOTICE.md` and
//! `docs/licensing/third-party-kernel-sources.md`.
//!
//! # Algorithm
//!
//! The sequence is cut into [`SOL_BLOCK`]-token blocks. Each key block is pooled (K mean, V sum);
//! each query block's centroid (its mean query) scores every pooled key block on the INT8 tensor
//! cores. A query block then attends **exactly** (INT8 `Q·Kᵀ`, U8 `P·V`, per-token scales) to a
//! routed subset of key blocks — its own block and both neighbours (the always-exact band), the
//! sink blocks, and either the top `topk_ratio` fraction of the remaining blocks by pooled score
//! ([`SolSelection::TopK`]) or those over a `tau`-sigma threshold ([`SolSelection::Tau`]) — while
//! every unrouted block still contributes one pooled term (its centroid score, weighted by its
//! token count, against its summed values), so the whole sequence stays in the softmax denominator.
//! `topk_ratio >= 1` routes every block, which is dense attention through the same INT8 pipeline.
//!
//! # Envelope
//!
//! Checked by the pure [`check_sol_attention`] before anything launches (E5): CUDA sm_80+ (the
//! source's `cc_floor`), bf16 or f16 operands of one dtype, `[B, H, S, D]` self-attention
//! (`Sq == Sk`, K and V shaped like Q: no GQA, no cross attention), head dim exactly
//! [`SOL_HEAD_DIM`], at most [`SOL_MAX_KEY_BLOCKS`] blocks (the routed lists are 16-bit and the
//! top-k selection holds one row of pooled scores in shared memory), `B·H` within the grid limit,
//! and every INT8 / routing buffer at most `i32::MAX` elements (E6). A ragged last block is padded
//! internally. There is no mask, causal mode, key bias, padded-tile table or key prefix.
//!
//! # Chunked-query variant
//!
//! [`SolConfig::query_chunk_blocks`] runs the route + exact stages over windows of query blocks
//! after one K/V preprocessing pass, bounding the routed-list workspace (`B·H·NQ·NK` u16 — the one
//! quadratic term) to one window. Every per-query-block quantity is independent of the window, so
//! the chunked result is **bitwise** the unchunked one.

use crate::nvrtc::KernelSource;
use candle_core::{DType, Tensor};

/// The Sol-Attn kernels (preprocess, V transpose, top-k threshold, route, exact), after the shared
/// INT8-attention device primitives.
pub const SOURCE: KernelSource = KernelSource {
    name: "sol-sparse-block-attention-v1",
    src: concat!(
        include_str!("int8_attention_common.cuh"),
        include_str!("sol_attention.cu")
    ),
    cc_floor: (8, 0),
};

/// The compute-capability floor (packed, `80` = sm_8.0): `cp.async` and the INT8 / bf16 `mma.sync`.
pub const SOL_MIN_COMPUTE_CAP: u32 = 80;

/// The only head dim the kernels handle (comfy-kitchen bakes it into the layout).
pub const SOL_HEAD_DIM: usize = 128;

/// Routing granularity, in tokens.
pub const SOL_BLOCK: usize = 64;

/// The most 64-token blocks one call routes (524,288 tokens): the routed block ids are `u16` and
/// the top-k selection keeps one query block's pooled scores (4 B each) in 32 KiB of shared memory.
/// Wan2.2 A14B at its 1280x720, 81-frame ceiling is 75,600 tokens (1,182 blocks).
pub const SOL_MAX_KEY_BLOCKS: usize = 8192;

/// How the routed key blocks are chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolSelection {
    /// SLA-style per-query-block top-k: keep [`topk_count`] of the non-sink key blocks with the
    /// best pooled scores (ties at the boundary are all kept). In `(0, 1]`; `1` routes everything.
    TopK(f32),
    /// comfy-kitchen's default threshold: route a block whose pooled score clears `tau` standard
    /// deviations of the query block's proxy score distribution. Finite.
    Tau(f32),
}

/// One Sol-Attn call's configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolConfig {
    /// Block selection.
    pub selection: SolSelection,
    /// `[start, end)` key blocks every query block attends exactly (on top of the budget).
    pub sink_blocks: (u32, u32),
    /// `[start, end)` query blocks that attend every key block exactly.
    pub sink_q: (u32, u32),
    /// Keep the pooled term of the unrouted blocks (comfy-kitchen's default); `false` is the
    /// softmax over the routed blocks only.
    pub tail: bool,
    /// Query blocks per route + exact window; `0` runs one window (see the module docs).
    pub query_chunk_blocks: usize,
}

impl SolConfig {
    /// Top-k selection at `ratio`, no sinks, the pooled tail on, one window.
    pub const fn top_k(ratio: f32) -> Self {
        Self {
            selection: SolSelection::TopK(ratio),
            sink_blocks: (0, 0),
            sink_q: (0, 0),
            tail: true,
            query_chunk_blocks: 0,
        }
    }

    /// Why this configuration is invalid, if it is (comfy-kitchen's `sol_attn_common_call_rule`,
    /// with the top-k range widened to include `1`).
    pub fn validate(&self) -> Result<(), &'static str> {
        match self.selection {
            SolSelection::TopK(r) if !(r > 0.0 && r <= 1.0) => {
                return Err("topk_ratio must be in (0, 1]")
            }
            SolSelection::Tau(t) if !t.is_finite() => return Err("tau must be finite"),
            _ => {}
        }
        if self.sink_blocks.1 < self.sink_blocks.0 {
            return Err("sink_blocks must be [start, end) with end >= start");
        }
        if self.sink_q.1 < self.sink_q.0 {
            return Err("sink_q must be [start, end) with end >= start");
        }
        // The kernels take block indices as signed 32-bit ints.
        let i32_max = i32::MAX as u32;
        if self.sink_blocks.1 > i32_max || self.sink_q.1 > i32_max {
            return Err("sink_blocks / sink_q endpoints must fit the kernels' i32 block index");
        }
        Ok(())
    }
}

/// Key blocks a query block keeps under top-k out of `n` selectable (non-sink) blocks: comfy-kitchen's
/// `_topk_count` (`round(ratio * n)`, ties to even, clamped to `[1, n - 1]`; `0` when `n <= 1`),
/// except that `ratio >= 1` keeps all `n` (dense through the sparse pipeline).
pub fn topk_count(n: usize, ratio: f32) -> usize {
    if ratio >= 1.0 {
        return n;
    }
    if n <= 1 {
        return 0;
    }
    // In f32, as the ratio is: 0.35 x 10 is 3.5 (ties to 4), as in comfy-kitchen's Python.
    let want = (ratio * n as f32).round_ties_even() as usize;
    want.clamp(1, n - 1)
}

/// Sink blocks inside `[0, n)` (they are always exact, so the top-k budget counts the others).
pub fn sink_count(n: usize, sinks: (u32, u32)) -> usize {
    (sinks.1 as usize)
        .min(n)
        .saturating_sub((sinks.0 as usize).min(n))
}

/// Why a call cannot take the Sol-Attn kernel. Every variant is decided before launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolRefusal {
    /// Not bf16 / f16.
    Dtype(DType),
    /// q, k and v do not share one dtype.
    MixedDtype,
    /// The head dim is not [`SOL_HEAD_DIM`] (or q / k / v differ).
    HeadDim(usize),
    /// Rank, batch, K/V shape, empty sequence, or grouped KV heads.
    Shape,
    /// `Sq != Sk`: cross attention (block routing needs one token sequence).
    CrossAttention,
    /// More than [`SOL_MAX_KEY_BLOCKS`] 64-token blocks.
    SequenceTooLong(usize),
    /// A carrier / routing buffer would exceed `i32::MAX` elements, or `B·H` the grid limit.
    IndexOverflow,
}

/// The launch geometry [`check_sol_attention`] accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolPlan {
    /// Batch.
    pub batch: usize,
    /// Heads.
    pub heads: usize,
    /// Tokens (query = key).
    pub len: usize,
    /// `len` padded to [`SOL_BLOCK`].
    pub t_pad: usize,
    /// Key (= query) blocks.
    pub blocks: usize,
    /// `blocks` rounded up to 64 (the route kernel reads pooled keys in 64-block groups).
    pub blocks_pad: usize,
}

impl SolPlan {
    /// Every launch one call makes, by kernel, as its `[x, y, z]` grid, at the largest query
    /// window (a chunked call's route / exact grids are never wider) — what the envelope
    /// ([`check_sol_attention`]) holds against CUDA's grid limits. Every grid puts `B·H` on y and
    /// a block count (at most [`SOL_MAX_KEY_BLOCKS`]) or `ceil(T / 256)` on x.
    pub fn launch_grids(&self) -> [(&'static str, [usize; 3]); 9] {
        let (bh, n, np) = (self.bh(), self.blocks, self.blocks_pad);
        [
            ("sol_reduce_kv", [np, bh, 1]),
            ("sol_pooled_stats", [bh, 1, 1]),
            ("sol_pooled_quant", [np, bh, 1]),
            ("sol_prep_q", [n, bh, 1]),
            ("sol_prep_k", [n, bh, 1]),
            ("sol_vtranspose", [self.len.div_ceil(256), bh, 1]),
            ("sol_topk_threshold", [n, bh, 1]),
            ("sol_route", [n.div_ceil(64), bh, 1]),
            ("sol_exact", [n, bh, 1]),
        ]
    }

    /// `sol_topk_threshold`'s dynamic shared memory: one `u32` key per block.
    pub fn topk_smem_bytes(&self) -> usize {
        self.blocks * 4
    }

    /// `B·H`.
    pub fn bh(&self) -> usize {
        self.batch * self.heads
    }

    /// Query blocks per route + exact window for `chunk` (`0` = all).
    pub fn window_blocks(&self, chunk: usize) -> usize {
        if chunk == 0 {
            self.blocks
        } else {
            chunk.min(self.blocks)
        }
    }

    /// Bytes of device workspace one call allocates besides its output, with route + exact windows
    /// of `chunk` query blocks.
    pub fn workspace_bytes(&self, chunk: usize) -> u64 {
        let bh = self.bh() as u64;
        let d = SOL_HEAD_DIM as u64;
        let (t, tp, n, np) = (
            self.len as u64,
            self.t_pad as u64,
            self.blocks as u64,
            self.blocks_pad as u64,
        );
        let w = self.window_blocks(chunk) as u64;
        let q_side = t * (d + 4) + n * (d + 4 + 4); // qiP + qs, cen8 + cens + thr
        let k_side = tp * (d + 8) + d * tp; // kiP + ksb, vTi
        let pooled = np * (4 * d + d + 4 + 2 * d) + 4 * d * 3; // kc, kciP, kcs, vcT, vsc/kmean/kcvar
        let window = w * (2 * n + 4 + 2 * d + 8); // idx, cnt, oPart, m/l
        bh * (q_side + k_side + pooled + window)
    }
}

/// The pure envelope check: `q`, `k` and `v` are `[B, H, S, D]` of one dtype.
pub fn check_sol_attention(
    q_dims: &[usize],
    k_dims: &[usize],
    v_dims: &[usize],
    q_dtype: DType,
    k_dtype: DType,
    v_dtype: DType,
) -> Result<SolPlan, SolRefusal> {
    if !matches!(q_dtype, DType::BF16 | DType::F16) {
        return Err(SolRefusal::Dtype(q_dtype));
    }
    if k_dtype != q_dtype || v_dtype != q_dtype {
        return Err(SolRefusal::MixedDtype);
    }
    let (&[b, h, sq, d], &[kb, kh, sk, kd]) = (q_dims, k_dims) else {
        return Err(SolRefusal::Shape);
    };
    if v_dims != k_dims || kb != b || kh != h || b == 0 || h == 0 || sq == 0 || sk == 0 {
        return Err(SolRefusal::Shape);
    }
    if d != SOL_HEAD_DIM || kd != d {
        return Err(SolRefusal::HeadDim(d));
    }
    if sq != sk {
        return Err(SolRefusal::CrossAttention);
    }
    let blocks = sq.div_ceil(SOL_BLOCK);
    if blocks > SOL_MAX_KEY_BLOCKS {
        return Err(SolRefusal::SequenceTooLong(blocks));
    }
    let t_pad = blocks * SOL_BLOCK;
    let blocks_pad = blocks.next_multiple_of(64);
    let fits = |parts: [usize; 3]| {
        parts
            .into_iter()
            .try_fold(1usize, |acc, x| acc.checked_mul(x))
            .is_some_and(|n| n <= i32::MAX as usize)
    };
    let bh = b.checked_mul(h).ok_or(SolRefusal::IndexOverflow)?;
    // `B·H` rides on every launch's grid.y; its 65,535 cap is checked with the grids below.
    if !fits([bh, t_pad, d]) || !fits([bh, blocks, blocks]) || !fits([bh, blocks_pad, d]) {
        return Err(SolRefusal::IndexOverflow);
    }
    let plan = SolPlan {
        batch: b,
        heads: h,
        len: sq,
        t_pad,
        blocks,
        blocks_pad,
    };
    // Every launch grid within CUDA's limits, and the top-k pass's dynamic shared memory within
    // the 48 KiB a launch gets without an opt-in attribute.
    if !plan
        .launch_grids()
        .iter()
        .all(|(_, grid)| crate::nvrtc::cuda_grid_fits(*grid))
        || plan.topk_smem_bytes() > SOL_TOPK_SMEM_MAX
    {
        return Err(SolRefusal::IndexOverflow);
    }
    Ok(plan)
}

/// The dynamic shared memory a launch may take without the opt-in attribute (48 KiB).
pub const SOL_TOPK_SMEM_MAX: usize = 48 * 1024;

// ---------------------------------------------------------------------------------------------
// The reference: comfy-kitchen's eager `sol_attn` (full precision), as candle ops on any device.
// ---------------------------------------------------------------------------------------------

/// The always-exact band of query block `qb` (its own block and both neighbours).
fn in_band(qb: usize, kb: usize) -> bool {
    kb + 1 >= qb && kb <= qb + 1
}

/// The exact-routing table `[NQ][NK]` the reference derives for one `(batch, head)` from its pooled
/// scores `colmean[qb][kb]` (log2 units) and proxy thresholds.
fn routing_table(colmean: &[f32], tau_thr: &[f32], n: usize, cfg: &SolConfig) -> Vec<bool> {
    let mut exact = vec![false; n * n];
    let sink = |kb: usize| kb >= cfg.sink_blocks.0 as usize && kb < cfg.sink_blocks.1 as usize;
    for qb in 0..n {
        let row = &colmean[qb * n..(qb + 1) * n];
        let out = &mut exact[qb * n..(qb + 1) * n];
        match cfg.selection {
            SolSelection::TopK(ratio) => {
                let kk = topk_count(n - sink_count(n, cfg.sink_blocks), ratio);
                if kk > 0 {
                    let mut ranked: Vec<f32> =
                        (0..n).filter(|&kb| !sink(kb)).map(|kb| row[kb]).collect();
                    ranked.sort_by(|a, b| b.total_cmp(a));
                    let thr = ranked[kk - 1];
                    for kb in 0..n {
                        out[kb] = !sink(kb) && row[kb] >= thr;
                    }
                }
            }
            SolSelection::Tau(_) => {
                for kb in 0..n {
                    out[kb] = row[kb] > tau_thr[qb];
                }
            }
        }
        let q_sink = qb >= cfg.sink_q.0 as usize && qb < cfg.sink_q.1 as usize;
        for (kb, e) in out.iter_mut().enumerate() {
            *e |= in_band(qb, kb) || sink(kb) || q_sink;
        }
    }
    exact
}

/// The full-precision reference of one `[B, H, S, 128]` Sol-Attn call (comfy-kitchen's eager
/// `sol_attn`, the parity oracle of the kernel): f32 arithmetic over the same block selection,
/// returned in `q`'s dtype as `[B, H, S, D]`. Materializes `[rows, S]` scores per head in query-row
/// windows of ~2^26 score elements, so it is O(S²) work — a test / investigation tool, never a
/// serving path. Any float dtype (f32 included); the shape rules are the kernel's. The kernel's INT8
/// arithmetic agrees to ~0.999 cosine, not bitwise.
pub fn sol_attention_reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    cfg: &SolConfig,
) -> candle_core::Result<Tensor> {
    if k.dtype() != q.dtype() || v.dtype() != q.dtype() {
        candle_core::bail!("sol reference: q, k and v must share one dtype");
    }
    let bf = DType::BF16;
    let plan = check_sol_attention(q.dims(), k.dims(), v.dims(), bf, bf, bf)
        .map_err(|r| candle_core::Error::Msg(format!("sol reference refused: {r:?}")))?;
    cfg.validate().map_err(candle_core::Error::msg)?;
    let (b, h, t, n) = (plan.batch, plan.heads, plan.len, plan.blocks);
    let d = SOL_HEAD_DIM;
    let log2s = (scale * std::f64::consts::LOG2_E) as f32;
    let dev = q.device().clone();
    let lengths: Vec<f32> = (0..n)
        .map(|i| (t - i * SOL_BLOCK).min(SOL_BLOCK) as f32)
        .collect();
    let len_t = Tensor::from_vec(lengths.clone(), (n, 1), &dev)?;
    let mut heads = Vec::with_capacity(b * h);
    for bi in 0..b {
        for hi in 0..h {
            let take = |x: &Tensor| -> candle_core::Result<Tensor> {
                x.narrow(0, bi, 1)?
                    .narrow(1, hi, 1)?
                    .squeeze(0)?
                    .squeeze(0)?
                    .to_dtype(DType::F32)?
                    .contiguous()
            };
            let (fq, fk, fv) = (take(q)?, take(k)?, take(v)?);
            let pad = |x: &Tensor| -> candle_core::Result<Tensor> {
                if plan.t_pad == t {
                    return Ok(x.clone());
                }
                Tensor::cat(
                    &[
                        x.clone(),
                        Tensor::zeros((plan.t_pad - t, d), DType::F32, &dev)?,
                    ],
                    0,
                )
            };
            let sums = |x: &Tensor| -> candle_core::Result<Tensor> {
                pad(x)?.reshape((n, SOL_BLOCK, d))?.sum(1)
            };
            let kc = sums(&fk)?.broadcast_div(&len_t)?; // [N, D] summary keys
            let vc = sums(&fv)?; // [N, D] summed values
            let centroid = sums(&fq)?.broadcast_div(&len_t)?; // [N, D]
            let k_mean = kc.mean_keepdim(0)?; // [1, D]
            let kcc = kc.broadcast_sub(&k_mean)?;
            let kc_var = kcc.sqr()?.mean_keepdim(0)?; // [1, D]
            let colmean = (centroid.matmul(&kcc.t()?)? * log2s as f64)?; // [N, N]
            let var = centroid.sqr()?.broadcast_mul(&kc_var)?.sum(1)?; // [N]
            let tau = match cfg.selection {
                SolSelection::Tau(tau) => tau,
                SolSelection::TopK(_) => 0.0,
            };
            let thr: Vec<f32> = var
                .to_vec1::<f32>()?
                .into_iter()
                .map(|v| tau * (v * log2s * log2s + 1e-6).sqrt())
                .collect();
            let cm = colmean.flatten_all()?.to_vec1::<f32>()?;
            let exact = routing_table(&cm, &thr, n, cfg);
            let exact_t = Tensor::from_vec(
                exact
                    .iter()
                    .map(|&e| if e { 1f32 } else { 0f32 })
                    .collect::<Vec<_>>(),
                (n, n),
                &dev,
            )?;
            let kh = fk.broadcast_sub(&k_mean)?.t()?.contiguous()?; // [D, T]
                                                                    // One query block window at a time: rows x T scores.
            let rows_per = ((1usize << 26) / t.max(1)).max(SOL_BLOCK) / SOL_BLOCK;
            let mut outs = Vec::new();
            let mut qb0 = 0;
            while qb0 < n {
                let nb = rows_per.min(n - qb0);
                let r0 = qb0 * SOL_BLOCK;
                let rows = (nb * SOL_BLOCK).min(t - r0);
                let qs = fq.narrow(0, r0, rows)?;
                let qblk: Vec<u32> = (0..rows).map(|r| ((r0 + r) / SOL_BLOCK) as u32).collect();
                let qblk = Tensor::from_vec(qblk, rows, &dev)?;
                let ex_blk = exact_t.index_select(&qblk, 0)?; // [rows, N] 1 = exact
                let ex_tok = ex_blk
                    .unsqueeze(2)?
                    .broadcast_as((rows, n, SOL_BLOCK))?
                    .reshape((rows, plan.t_pad))?
                    .narrow(1, 0, t)?;
                let neg = |mask_exact: &Tensor, keep_when: f64| -> candle_core::Result<Tensor> {
                    // 0 where kept, -1e30 where dropped.
                    (mask_exact - keep_when)?.abs()? * -1e30
                };
                let s_tok = (qs.matmul(&kh)? * log2s as f64)?.add(&neg(&ex_tok, 1.0)?)?;
                let cm_rows = colmean.index_select(&qblk, 0)?; // [rows, N]
                let s_blk = if cfg.tail {
                    cm_rows.add(&neg(&ex_blk, 0.0)?)?
                } else {
                    (cm_rows.zeros_like()? - 1e30)?
                };
                let m = s_tok.max_keepdim(1)?.maximum(&s_blk.max_keepdim(1)?)?;
                let p_tok = exp2(&s_tok.broadcast_sub(&m)?)?;
                let p_blk = exp2(&s_blk.broadcast_sub(&m)?)?;
                let num = p_tok.matmul(&fv)?.add(&p_blk.matmul(&vc)?)?;
                let den = p_tok
                    .sum_keepdim(1)?
                    .add(&p_blk.matmul(&len_t)?)?
                    .clamp(1e-30, f64::MAX)?;
                outs.push(num.broadcast_div(&den)?);
                qb0 += nb;
            }
            heads.push(Tensor::cat(&outs, 0)?);
        }
    }
    Tensor::stack(&heads, 0)?
        .reshape((b, h, t, d))?
        .to_dtype(q.dtype())
}

fn exp2(x: &Tensor) -> candle_core::Result<Tensor> {
    (x * std::f64::consts::LN_2)?.exp()
}

/// The dense f32 reference `softmax(q·kᵀ·scale)·v` (`[B, H, S, D]` in `q`'s dtype), in query-row
/// windows — the "exact attention" the sparse result is measured against.
pub fn dense_attention_reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
) -> candle_core::Result<Tensor> {
    let (b, h, t, d) = q.dims4()?;
    let sk = k.dim(2)?;
    let mut heads = Vec::with_capacity(b * h);
    for bi in 0..b {
        for hi in 0..h {
            let take = |x: &Tensor| -> candle_core::Result<Tensor> {
                x.narrow(0, bi, 1)?
                    .narrow(1, hi, 1)?
                    .squeeze(0)?
                    .squeeze(0)?
                    .to_dtype(DType::F32)
            };
            let (fq, fk, fv) = (take(q)?, take(k)?, take(v)?);
            let kt = fk.t()?.contiguous()?;
            let rows_per = ((1usize << 26) / sk.max(1)).max(1);
            let mut outs = Vec::new();
            let mut r0 = 0;
            while r0 < t {
                let rows = rows_per.min(t - r0);
                let s = (fq.narrow(0, r0, rows)?.matmul(&kt)? * scale)?;
                let p = candle_nn::ops::softmax_last_dim(&s)?;
                outs.push(p.matmul(&fv)?);
                r0 += rows;
            }
            heads.push(Tensor::cat(&outs, 0)?);
        }
    }
    Tensor::stack(&heads, 0)?
        .reshape((b, h, t, d))?
        .to_dtype(q.dtype())
}

/// `(cosine, rel-L2)` of `a` against `b`, in f32.
pub fn similarity(a: &Tensor, b: &Tensor) -> candle_core::Result<(f32, f32)> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?;
    let dot = (&a * &b)?.sum_all()?.to_scalar::<f32>()?;
    let na = a.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    let nb = b.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    let diff = (&a - &b)?.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    Ok((
        dot / (na * nb).max(f32::MIN_POSITIVE),
        diff / nb.max(f32::MIN_POSITIVE),
    ))
}

/// Per-call routing statistics (`sol_attention_with_stats`, cuda builds).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SolStats {
    /// Routed (exact) `(query block, key block)` pairs over every `(batch, head)`.
    pub routed: u64,
    /// All pairs: `B·H·NQ·NK`.
    pub total: u64,
}

impl SolStats {
    /// The routed fraction (1 = dense).
    pub fn density(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.routed as f64 / self.total as f64
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::{sol_attention, sol_attention_with_stats};

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use candle_core::cuda_backend::cudarc::driver::{
        CudaSlice, CudaView, DeviceRepr, LaunchConfig, PushKernelArg,
    };
    use candle_core::cuda_backend::CudaDType;
    use candle_core::{op::BackpropOp, CudaStorage, Device, Result, Shape, Storage};

    fn msg(text: impl Into<String>) -> candle_core::Error {
        candle_core::Error::Msg(text.into())
    }

    fn refusal(r: SolRefusal) -> candle_core::Error {
        msg(format!("sol attention refused: {r:?}"))
    }

    fn drv<E: std::fmt::Debug>(e: E) -> candle_core::Error {
        msg(format!("sol attention: {e:?}"))
    }

    /// `t` itself when its last dim is unit-stride and every row / head / batch stride and the
    /// start offset are multiples of 8 elements (the kernels' 16-byte loads), else a fresh
    /// zero-offset contiguous copy. `force_contiguous`, not `contiguous`: an already-contiguous
    /// view (a flat buffer narrowed at element 1, or an odd stride on a size-1 dim) is returned
    /// as-is by `contiguous`, offset and all, and would issue misaligned `uint4` loads.
    fn operand(t: &Tensor) -> Result<Tensor> {
        let st = t.stride();
        let off = t.layout().start_offset();
        let ok = st.len() == 4
            && st[3] == 1
            && st[..3].iter().all(|s| s.is_multiple_of(8))
            && off.is_multiple_of(8);
        if ok {
            Ok(t.clone())
        } else {
            t.force_contiguous()
        }
    }

    /// The `(batch, token, head)` element strides of a `[B, H, S, D]` operand.
    fn strides_bth(t: &Tensor) -> (i64, i64, i64) {
        let s = t.stride();
        (s[0] as i64, s[2] as i64, s[1] as i64)
    }

    fn view<'a, T: CudaDType>(st: &'a Storage, t: &Tensor) -> Result<CudaView<'a, T>> {
        let Storage::Cuda(cs) = st else {
            return Err(msg("sol attention: expected CUDA storage"));
        };
        Ok(cs.as_cuda_slice::<T>()?.slice(t.layout().start_offset()..))
    }

    fn wrap<T: CudaDType>(
        slice: CudaSlice<T>,
        dev: &candle_core::CudaDevice,
        shape: impl Into<Shape>,
    ) -> Tensor {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, dev.clone())),
            shape.into(),
            BackpropOp::none(),
            false,
        )
    }

    fn i32_of(x: usize) -> Result<i32> {
        i32::try_from(x).map_err(|_| refusal(SolRefusal::IndexOverflow))
    }

    /// A launch over the `(x, y)` grid, refused with a typed error past CUDA's grid limits (the
    /// envelope already refuses every such plan; this keeps a launch from reaching the driver).
    fn cfg(grid: (usize, usize), block: u32, smem: u32) -> Result<LaunchConfig> {
        if !crate::nvrtc::cuda_grid_fits([grid.0, grid.1, 1]) {
            return Err(refusal(SolRefusal::IndexOverflow));
        }
        let dim = |x: usize| u32::try_from(x).map_err(|_| refusal(SolRefusal::IndexOverflow));
        Ok(LaunchConfig {
            grid_dim: (dim(grid.0)?, dim(grid.1)?, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: smem,
        })
    }

    /// One Sol-Attn call over `[B, H, S, 128]` self-attention operands: the contiguous
    /// `[B, H, S, 128]` output in the operands' dtype. `scale` is the softmax scale.
    pub fn sol_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f64,
        config: &SolConfig,
    ) -> Result<Tensor> {
        Ok(run(q, k, v, scale, config, false)?.0)
    }

    /// [`sol_attention`] plus the routed-pair count (one extra device reduction and sync).
    pub fn sol_attention_with_stats(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f64,
        config: &SolConfig,
    ) -> Result<(Tensor, SolStats)> {
        let (out, stats) = run(q, k, v, scale, config, true)?;
        Ok((out, stats.unwrap_or_default()))
    }

    fn run(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f64,
        config: &SolConfig,
        stats: bool,
    ) -> Result<(Tensor, Option<SolStats>)> {
        let plan = check_sol_attention(
            q.dims(),
            k.dims(),
            v.dims(),
            q.dtype(),
            k.dtype(),
            v.dtype(),
        )
        .map_err(refusal)?;
        config
            .validate()
            .map_err(|e| msg(format!("sol attention: {e}")))?;
        if !(scale.is_finite() && scale > 0.0) {
            return Err(msg(format!(
                "sol attention: scale {scale} must be finite and positive"
            )));
        }
        let Device::Cuda(dev) = q.device() else {
            return Err(msg("sol attention: expected a CUDA tensor"));
        };
        if !q.device().same_device(k.device()) || !q.device().same_device(v.device()) {
            return Err(msg("sol attention: q, k and v on different devices"));
        }
        let module = SOURCE.compiled(dev).map_err(candle_core::Error::wrap)?;
        match q.dtype() {
            DType::BF16 => {
                run_typed::<half::bf16>(dev, &module, q, k, v, scale, config, &plan, "bf16", stats)
            }
            _ => run_typed::<half::f16>(dev, &module, q, k, v, scale, config, &plan, "f16", stats),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_typed<T: CudaDType + DeviceRepr>(
        dev: &candle_core::CudaDevice,
        module: &crate::nvrtc::CompiledKernel,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f64,
        config: &SolConfig,
        p: &SolPlan,
        suffix: &str,
        want_stats: bool,
    ) -> Result<(Tensor, Option<SolStats>)> {
        let f = |name: &str| module.function(name).map_err(candle_core::Error::wrap);
        let f_reduce = f(&format!("sol_reduce_kv_{suffix}"))?;
        let f_stats = f("sol_pooled_stats")?;
        let f_pquant = f("sol_pooled_quant")?;
        let f_prep_q = f(&format!("sol_prep_q_{suffix}"))?;
        let f_prep_k = f(&format!("sol_prep_k_{suffix}"))?;
        let f_vt = f(&format!("sol_vtranspose_{suffix}"))?;
        let f_topk = f("sol_topk_threshold")?;
        let (q, k, v) = (operand(q)?, operand(k)?, operand(v)?);
        let (qsb, qst, qsh) = strides_bth(&q);
        let (ksb_, kst, ksh) = strides_bth(&k);
        let (vsb, vst, vsh) = strides_bth(&v);
        let (bh, d) = (p.bh(), SOL_HEAD_DIM);
        let (t, tp, n, np) = (p.len, p.t_pad, p.blocks, p.blocks_pad);
        let (ti, tpi, ni, npi, hi) = (
            i32_of(t)?,
            i32_of(tp)?,
            i32_of(n)?,
            i32_of(np)?,
            i32_of(p.heads)?,
        );
        let scale_log2 = (scale as f32) * std::f32::consts::LOG2_E;
        let stream = dev.cuda_stream();

        let (qs_st, _) = q.storage_and_layout();
        let (ks_st, _) = k.storage_and_layout();
        let (vs_st, _) = v.storage_and_layout();
        let qv = view::<T>(&qs_st, &q)?;
        let kv = view::<T>(&ks_st, &k)?;
        let vv = view::<T>(&vs_st, &v)?;

        // SAFETY (every `alloc` below): each buffer is fully written by the stage that produces it
        // before any stage reads it (pads included: reduce_kv / pooled_quant write the 64-group
        // padding, prep_k the dead rows of the last block, vtranspose every padded key column).
        let mut kc = unsafe { stream.alloc::<f32>(bh * np * d) }.map_err(drv)?;
        let mut vct = unsafe { stream.alloc::<u16>(bh * d * np) }.map_err(drv)?;
        let mut vsc = stream.alloc_zeros::<f32>(bh * d).map_err(drv)?; // |max| accumulator first
        let mut kmean = unsafe { stream.alloc::<f32>(bh * d) }.map_err(drv)?;
        let mut kcvar = unsafe { stream.alloc::<f32>(bh * d) }.map_err(drv)?;
        let mut kcip = unsafe { stream.alloc::<u8>(bh * np * d) }.map_err(drv)?;
        let mut kcs = unsafe { stream.alloc::<f32>(bh * np) }.map_err(drv)?;
        let mut qip = unsafe { stream.alloc::<u8>(bh * t * d) }.map_err(drv)?;
        let mut qsc = unsafe { stream.alloc::<f32>(bh * t) }.map_err(drv)?;
        let mut thr = unsafe { stream.alloc::<f32>(bh * n) }.map_err(drv)?;
        let mut cen8 = unsafe { stream.alloc::<u8>(bh * n * d) }.map_err(drv)?;
        let mut cens = unsafe { stream.alloc::<f32>(bh * n) }.map_err(drv)?;
        let mut kip = unsafe { stream.alloc::<u8>(bh * tp * d) }.map_err(drv)?;
        let mut kscale = unsafe { stream.alloc::<f32>(bh * tp * 2) }.map_err(drv)?;
        let mut vti = unsafe { stream.alloc::<u8>(bh * d * tp) }.map_err(drv)?;

        // pass 1: K block means, V block sums, V |max|
        let mut l = stream.launch_builder(&f_reduce);
        l.arg(&kv).arg(&vv).arg(&mut kc).arg(&mut vct).arg(&mut vsc);
        l.arg(&ti).arg(&hi).arg(&ni).arg(&npi);
        l.arg(&ksb_)
            .arg(&kst)
            .arg(&ksh)
            .arg(&vsb)
            .arg(&vst)
            .arg(&vsh);
        // SAFETY: matches `sol_reduce_kv_*`; grid (NPAD, BH) x 128 channels.
        unsafe { l.launch(cfg((np, bh), 128, 0)?) }.map_err(drv)?;
        // pass 2
        let mut l = stream.launch_builder(&f_stats);
        l.arg(&kc)
            .arg(&mut kmean)
            .arg(&mut vsc)
            .arg(&mut kcvar)
            .arg(&ni)
            .arg(&npi);
        // SAFETY: matches `sol_pooled_stats`; grid BH x 128.
        unsafe { l.launch(cfg((bh, 1), 128, 0)?) }.map_err(drv)?;
        // pass 3
        let mut l = stream.launch_builder(&f_pquant);
        l.arg(&kc)
            .arg(&kmean)
            .arg(&mut kcip)
            .arg(&mut kcs)
            .arg(&ni)
            .arg(&npi);
        // SAFETY: matches `sol_pooled_quant`; grid (NPAD, BH) x 128.
        unsafe { l.launch(cfg((np, bh), 128, 0)?) }.map_err(drv)?;
        // pass 4: Q
        let tau = match config.selection {
            SolSelection::Tau(tau) => tau,
            SolSelection::TopK(_) => 0.0,
        };
        let mut l = stream.launch_builder(&f_prep_q);
        l.arg(&qv)
            .arg(&kcvar)
            .arg(&mut qip)
            .arg(&mut qsc)
            .arg(&mut thr)
            .arg(&mut cen8)
            .arg(&mut cens);
        l.arg(&ti).arg(&hi).arg(&ni).arg(&tau).arg(&scale_log2);
        l.arg(&qsb).arg(&qst).arg(&qsh);
        // SAFETY: matches `sol_prep_q_*`; grid (NQ, BH) x 128.
        unsafe { l.launch(cfg((n, bh), 128, 0)?) }.map_err(drv)?;
        // pass 5: K
        let mut l = stream.launch_builder(&f_prep_k);
        l.arg(&kv).arg(&kmean).arg(&mut kip).arg(&mut kscale);
        l.arg(&ti).arg(&tpi).arg(&hi).arg(&ksb_).arg(&kst).arg(&ksh);
        // SAFETY: matches `sol_prep_k_*`; grid (NTB, BH) x 128.
        unsafe { l.launch(cfg((n, bh), 128, 0)?) }.map_err(drv)?;
        // V transpose
        let mut l = stream.launch_builder(&f_vt);
        l.arg(&vv).arg(&vsc).arg(&mut vti);
        l.arg(&ti).arg(&tpi).arg(&hi).arg(&vsb).arg(&vst).arg(&vsh);
        // SAFETY: matches `sol_vtranspose_*`; grid (ceil(T/256), BH) x 256.
        unsafe { l.launch(cfg((t.div_ceil(256), bh), 256, 0)?) }.map_err(drv)?;
        drop((qs_st, ks_st, vs_st));

        // Clamp the sink ranges to the actual block count before the i32 kernel ABI: a range past
        // the sequence would otherwise count sink blocks that do not exist (`validate` already
        // refuses endpoints beyond i32).
        let block = |v: u32| v.min(n as u32) as i32;
        let (sink_s, sink_e) = (block(config.sink_blocks.0), block(config.sink_blocks.1));
        let (sink_qs, sink_qe) = (block(config.sink_q.0), block(config.sink_q.1));
        if let SolSelection::TopK(ratio) = config.selection {
            let kk = i32_of(topk_count(n - sink_count(n, config.sink_blocks), ratio))?;
            let mut l = stream.launch_builder(&f_topk);
            l.arg(&cen8).arg(&cens).arg(&kcip).arg(&kcs).arg(&mut thr);
            l.arg(&ni)
                .arg(&ni)
                .arg(&npi)
                .arg(&kk)
                .arg(&sink_s)
                .arg(&sink_e)
                .arg(&scale_log2);
            // SAFETY: matches `sol_topk_threshold`; NTB u32 keys of dynamic smem (<= 32 KiB).
            unsafe { l.launch(cfg((n, bh), 128, (n * 4) as u32)?) }.map_err(drv)?;
        }

        // route + exact per query-block window
        let w = p.window_blocks(config.query_chunk_blocks);
        // SAFETY: route writes every entry the exact kernel reads for the window's live blocks.
        let mut idx = unsafe { stream.alloc::<u16>(bh * w * n) }.map_err(drv)?;
        let mut cnt = unsafe { stream.alloc::<i32>(bh * w) }.map_err(drv)?;
        let mut opart = unsafe { stream.alloc::<u16>(bh * w * d) }.map_err(drv)?;
        let mut mpart = unsafe { stream.alloc::<f32>(bh * w) }.map_err(drv)?;
        let mut lpart = unsafe { stream.alloc::<f32>(bh * w) }.map_err(drv)?;
        // SAFETY: the exact kernel writes every element of every row < T.
        let mut out = unsafe { stream.alloc::<T>(bh * t * d) }.map_err(drv)?;
        let tail = i32::from(config.tail);
        let (diag_lo, diag_span) = (-1i32, 2i32);
        let route = f("sol_route")?;
        let exact = f(&format!("sol_exact_{suffix}"))?;
        let mut routed = 0u64;
        let mut qb0 = 0usize;
        while qb0 < n {
            let nqc = w.min(n - qb0);
            let (qb0i, nqci) = (i32_of(qb0)?, i32_of(nqc)?);
            let mut l = stream.launch_builder(&route);
            l.arg(&cen8)
                .arg(&cens)
                .arg(&kcip)
                .arg(&kcs)
                .arg(&vct)
                .arg(&vsc)
                .arg(&thr);
            l.arg(&mut idx)
                .arg(&mut cnt)
                .arg(&mut opart)
                .arg(&mut mpart)
                .arg(&mut lpart);
            l.arg(&tail)
                .arg(&ti)
                .arg(&ni)
                .arg(&npi)
                .arg(&ni)
                .arg(&qb0i)
                .arg(&nqci);
            l.arg(&sink_s)
                .arg(&sink_e)
                .arg(&sink_qs)
                .arg(&sink_qe)
                .arg(&diag_lo)
                .arg(&diag_span);
            l.arg(&scale_log2);
            // SAFETY: matches `sol_route`; grid (ceil(nqc / 64), BH) x 128.
            unsafe { l.launch(cfg((nqc.div_ceil(64), bh), 128, 0)?) }.map_err(drv)?;
            let mut l = stream.launch_builder(&exact);
            l.arg(&qip)
                .arg(&qsc)
                .arg(&kip)
                .arg(&kscale)
                .arg(&vti)
                .arg(&vsc);
            l.arg(&idx)
                .arg(&cnt)
                .arg(&opart)
                .arg(&mpart)
                .arg(&lpart)
                .arg(&mut out);
            l.arg(&ti)
                .arg(&tpi)
                .arg(&ni)
                .arg(&qb0i)
                .arg(&nqci)
                .arg(&scale_log2);
            // SAFETY: matches `sol_exact_*`; grid (nqc, BH) x 128.
            unsafe { l.launch(cfg((nqc, bh), 128, 0)?) }.map_err(drv)?;
            if want_stats {
                let counts = stream.clone_dtoh(&cnt).map_err(drv)?;
                for b in 0..bh {
                    for j in 0..nqc {
                        routed += counts[b * nqc + j].max(0) as u64;
                    }
                }
            }
            qb0 += nqc;
        }
        let stats = want_stats.then_some(SolStats {
            routed,
            total: (bh * n * n) as u64,
        });
        Ok((wrap(out, dev, (p.batch, p.heads, t, d)), stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// PR #1530 review: every Sol-Attn launch's grid is held to CUDA's limits in the envelope.
    /// `B·H` rides on grid.y (65,535 launches, 65,536 refuses); every x extent is a block count
    /// (at most [`SOL_MAX_KEY_BLOCKS`]) or `ceil(T / 256)`, and the top-k pass's shared memory stays
    /// within 48 KiB, at the longest sequence the envelope admits.
    #[test]
    fn every_launch_grid_is_inside_cuda_limits_at_the_boundaries() {
        use crate::nvrtc::{cuda_grid_fits, CUDA_GRID_YZ_MAX};
        let chk = |dims: &[usize]| {
            check_sol_attention(dims, dims, dims, DType::BF16, DType::BF16, DType::BF16)
        };
        for (b, h, want_ok) in [
            (1, CUDA_GRID_YZ_MAX, true),
            (1, CUDA_GRID_YZ_MAX + 1, false),
            (CUDA_GRID_YZ_MAX, 1, true),
            (CUDA_GRID_YZ_MAX + 1, 1, false),
            (257, 255, true),
            (257, 256, false),
        ] {
            let got = chk(&[b, h, 64, SOL_HEAD_DIM]);
            if want_ok {
                let plan = got.unwrap();
                assert!(plan.launch_grids().iter().all(|(_, g)| cuda_grid_fits(*g)));
            } else {
                assert_eq!(got, Err(SolRefusal::IndexOverflow), "b={b} h={h}");
            }
        }
        let longest = SOL_MAX_KEY_BLOCKS * SOL_BLOCK;
        let plan = chk(&[1, 1, longest, SOL_HEAD_DIM]).unwrap();
        assert_eq!(plan.blocks, SOL_MAX_KEY_BLOCKS);
        for (name, grid) in plan.launch_grids() {
            assert!(cuda_grid_fits(grid), "{name}: {grid:?}");
            assert!(
                grid[0] <= SOL_MAX_KEY_BLOCKS.max(longest.div_ceil(256)),
                "{name}"
            );
        }
        assert!(plan.topk_smem_bytes() <= SOL_TOPK_SMEM_MAX);
    }

    #[test]
    fn the_envelope_admits_wan_self_attention_and_refuses_the_rest() {
        // Wan2.2 A14B at 480x832, 81 frames: 21 x 30 x 52 = 32,760 tokens, 40 heads x 128.
        let wan = [1, 40, 32_760, 128];
        let plan =
            check_sol_attention(&wan, &wan, &wan, DType::BF16, DType::BF16, DType::BF16).unwrap();
        assert_eq!(
            (plan.blocks, plan.t_pad, plan.blocks_pad),
            (512, 32_768, 512)
        );
        let chk = |q: &[usize], k: &[usize], dt: DType| check_sol_attention(q, k, k, dt, dt, dt);
        assert_eq!(
            chk(&wan, &wan, DType::F32),
            Err(SolRefusal::Dtype(DType::F32))
        );
        assert_eq!(
            check_sol_attention(&wan, &wan, &wan, DType::BF16, DType::F16, DType::BF16),
            Err(SolRefusal::MixedDtype)
        );
        for d in [64, 96, 256] {
            let x = [1, 4, 640, d];
            assert_eq!(chk(&x, &x, DType::BF16), Err(SolRefusal::HeadDim(d)));
        }
        // Wan's text cross attention (512 text keys) and GQA.
        assert_eq!(
            chk(&wan, &[1, 40, 512, 128], DType::BF16),
            Err(SolRefusal::CrossAttention)
        );
        assert_eq!(
            chk(&wan, &[1, 8, 32_760, 128], DType::BF16),
            Err(SolRefusal::Shape)
        );
        assert_eq!(
            chk(&[1, 4, 0, 128], &[1, 4, 0, 128], DType::BF16),
            Err(SolRefusal::Shape)
        );
        assert_eq!(
            chk(&[4, 64, 128], &[4, 64, 128], DType::BF16),
            Err(SolRefusal::Shape)
        );
        // A ragged tail pads internally.
        let ragged = [2, 3, 1000, 128];
        let plan = chk(&ragged, &ragged, DType::F16).unwrap();
        assert_eq!((plan.blocks, plan.t_pad, plan.blocks_pad), (16, 1024, 64));
        // Past the block-id / top-k limit, and past i32 indexing.
        let long = [1, 1, SOL_MAX_KEY_BLOCKS * SOL_BLOCK + 1, 128];
        assert_eq!(
            chk(&long, &long, DType::BF16),
            Err(SolRefusal::SequenceTooLong(SOL_MAX_KEY_BLOCKS + 1))
        );
        let wide = [1, 600, 32_768, 128];
        assert_eq!(
            chk(&wide, &wide, DType::BF16),
            Err(SolRefusal::IndexOverflow)
        );
        let grid = [65_536, 1, 64, 128];
        assert_eq!(
            chk(&grid, &grid, DType::BF16),
            Err(SolRefusal::IndexOverflow)
        );
    }

    #[test]
    fn topk_counts_follow_comfy_kitchen_and_one_routes_everything() {
        assert_eq!(topk_count(512, 0.1), 51);
        assert_eq!(topk_count(512, 0.25), 128);
        assert_eq!(topk_count(10, 0.25), 2); // 2.5 rounds to even
        assert_eq!(topk_count(10, 0.35), 4); // 3.5 rounds to even
        assert_eq!(topk_count(10, 0.01), 1); // at least one
        assert_eq!(topk_count(10, 0.99), 9); // at most n - 1 below 1.0
        assert_eq!(topk_count(10, 1.0), 10);
        assert_eq!(topk_count(1, 0.5), 0);
        assert_eq!(topk_count(0, 0.5), 0);
        assert_eq!(sink_count(32, (0, 4)), 4);
        assert_eq!(sink_count(32, (28, 40)), 4);
        assert_eq!(sink_count(32, (40, 50)), 0);
    }

    #[test]
    fn configs_are_validated() {
        assert!(SolConfig::top_k(0.25).validate().is_ok());
        assert!(SolConfig::top_k(1.0).validate().is_ok());
        for bad in [0.0, -0.1, 1.5, f32::NAN] {
            assert!(SolConfig::top_k(bad).validate().is_err(), "{bad}");
        }
        let tau = SolConfig {
            selection: SolSelection::Tau(f32::INFINITY),
            ..SolConfig::top_k(0.5)
        };
        assert!(tau.validate().is_err());
        let sinks = SolConfig {
            sink_blocks: (3, 1),
            ..SolConfig::top_k(0.5)
        };
        assert!(sinks.validate().is_err());
        let sink_q = SolConfig {
            sink_q: (2, 1),
            ..SolConfig::top_k(0.5)
        };
        assert!(sink_q.validate().is_err());
        // Endpoints beyond the kernels' i32 block index are refused, not wrapped negative.
        for (blocks, q) in [
            ((2_147_483_648, 2_147_483_649), (0, 1)),
            ((0, 1), (0, 2_147_483_648)),
            ((0, u32::MAX), (0, 1)),
        ] {
            let wide = SolConfig {
                sink_blocks: blocks,
                sink_q: q,
                ..SolConfig::top_k(0.5)
            };
            assert!(wide.validate().is_err(), "{blocks:?} {q:?}");
        }
        let edge = SolConfig {
            sink_blocks: (0, i32::MAX as u32),
            sink_q: (0, i32::MAX as u32),
            ..SolConfig::top_k(0.5)
        };
        assert!(edge.validate().is_ok());
        // Sink ranges past the sequence count only the blocks that exist.
        assert_eq!(sink_count(1, (0, i32::MAX as u32)), 1);
        assert_eq!(sink_count(4, (2, 9)), 2);
    }

    #[test]
    fn workspace_covers_the_carriers_and_one_routing_window() {
        let x = [1, 40, 32_760, 128];
        let p = check_sol_attention(&x, &x, &x, DType::BF16, DType::BF16, DType::BF16).unwrap();
        let full = p.workspace_bytes(0);
        let chunked = p.workspace_bytes(64);
        // The INT8 Q/K/V carriers alone are 3 x 40 x 32k x 128 B.
        assert!(full > 3 * 40 * 32_760 * 128, "{full}");
        // One 64-block window keeps 1/8 of the routed lists (40 x 512 x 512 x 2 B in full).
        assert!(full - chunked > 40 * 448 * 512 * 2, "{full} {chunked}");
        assert_eq!(p.window_blocks(0), 512);
        assert_eq!(p.window_blocks(64), 64);
        assert_eq!(p.window_blocks(10_000), 512);
    }

    fn operands(b: usize, h: usize, t: usize, seed: u64) -> (Tensor, Tensor, Tensor) {
        // Deterministic pseudo-random operands without a device RNG.
        let mk = |salt: u64, s: f32| {
            let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(salt);
            let data: Vec<f32> = (0..b * h * t * SOL_HEAD_DIM)
                .map(|_| {
                    x = x
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (((x >> 40) as f32) / ((1u64 << 24) as f32) - 0.5) * 2.0 * s
                })
                .collect();
            Tensor::from_vec(data, (b, h, t, SOL_HEAD_DIM), &Device::Cpu).unwrap()
        };
        (mk(1, 1.0), mk(2, 1.0), mk(3, 1.0))
    }

    /// The reference with every block routed (top-k 1.0, or tau with every query block a query
    /// sink) is dense attention: E2's f32 bound.
    #[test]
    fn the_reference_routing_everything_is_dense_attention() -> candle_core::Result<()> {
        let (q, k, v) = operands(1, 2, 200, 7); // ragged: 4 blocks, the last 8 rows
        let scale = (SOL_HEAD_DIM as f64).powf(-0.5);
        let dense = dense_attention_reference(&q, &k, &v, scale)?;
        let all = sol_attention_reference(&q, &k, &v, scale, &SolConfig::top_k(1.0))?;
        let sink_q = SolConfig {
            selection: SolSelection::Tau(1.0),
            sink_q: (0, 4),
            ..SolConfig::top_k(1.0)
        };
        let all_q = sol_attention_reference(&q, &k, &v, scale, &sink_q)?;
        for got in [all, all_q] {
            let err = (&got - &dense)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let mag = dense.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(err <= 1e-5 * mag.max(1.0), "max |d| {err} vs {mag}");
        }
        // A sparse budget is not dense, but stays close on this smooth fixture.
        let sparse = sol_attention_reference(&q, &k, &v, scale, &SolConfig::top_k(0.25))?;
        let (cos, rel) = similarity(&sparse, &dense)?;
        assert!(rel > 1e-4 && cos > 0.9, "cos {cos} rel {rel}");
        Ok(())
    }

    /// The pooled-tail bookkeeping: with the tail off and the band covering every block (a two-block
    /// sequence), the result is dense; with one far block pooled, the tail keeps it in the
    /// denominator (closer to dense than dropping it).
    #[test]
    fn the_reference_tail_keeps_unrouted_blocks_in_the_softmax() -> candle_core::Result<()> {
        let scale = (SOL_HEAD_DIM as f64).powf(-0.5);
        let (q, k, v) = operands(1, 1, 128, 3);
        let dense = dense_attention_reference(&q, &k, &v, scale)?;
        let no_tail = SolConfig {
            tail: false,
            ..SolConfig::top_k(0.5)
        };
        let got = sol_attention_reference(&q, &k, &v, scale, &no_tail)?;
        assert!(similarity(&got, &dense)?.1 < 1e-5);
        let (q, k, v) = operands(1, 1, 64 * 8, 5);
        let dense = dense_attention_reference(&q, &k, &v, scale)?;
        let with_tail = sol_attention_reference(&q, &k, &v, scale, &SolConfig::top_k(0.2))?;
        let dropped = sol_attention_reference(
            &q,
            &k,
            &v,
            scale,
            &SolConfig {
                tail: false,
                ..SolConfig::top_k(0.2)
            },
        )?;
        let (rel_tail, rel_drop) = (
            similarity(&with_tail, &dense)?.1,
            similarity(&dropped, &dense)?.1,
        );
        assert!(rel_tail < rel_drop, "tail {rel_tail} vs dropped {rel_drop}");
        Ok(())
    }
}

// -------------------------------------------------------------------------------------------------
// CUDA compute - `#[ignore]`d, never part of normal CI. Run on a pinned validation GPU:
//
// CUDA_VISIBLE_DEVICES=<gpu> cargo test -p candle-quant-kernels --features cuda --lib \
//   sol_attention::cuda_tests:: -- --ignored --nocapture --test-threads=1
// -------------------------------------------------------------------------------------------------
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn device() -> Device {
        Device::new_cuda(0).expect("CUDA device 0")
    }

    /// comfy-kitchen's `_qkv`: q, k ~ N(0, 0.25), v ~ N(0, 1), bf16, `[B, H, S, D]`.
    fn qkv(
        b: usize,
        h: usize,
        t: usize,
        dtype: DType,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let dev = device();
        let mk = |s: f64| -> candle_core::Result<Tensor> {
            (Tensor::randn(0f32, 1.0, (b, h, t, SOL_HEAD_DIM), &dev)? * s)?.to_dtype(dtype)
        };
        Ok((mk(0.5)?, mk(0.5)?, mk(1.0)?))
    }

    const SCALE: f64 = 0.088_388_347_648_318_44; // 128^-0.5

    /// Every entry point compiles through the seam with zero local memory.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_sol_kernels_compile_without_local_memory() {
        let dev = device();
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let module = SOURCE.compiled(cuda).expect("compiles");
        for entry in module.entry_points() {
            let f = module.function(entry).unwrap();
            let local = f.local_size_bytes().unwrap();
            println!(
                "{entry}: {local} B local, {} registers",
                f.num_regs().unwrap()
            );
            assert_eq!(local, 0, "{entry} spills to local memory");
        }
        assert_eq!(module.entry_points().len(), 14);
    }

    /// AC1: with every block routed (`topk_ratio = 1.0`) the kernel is dense attention up to its
    /// INT8 arithmetic: cosine >= 0.999 and rel-L2 < 0.016 against the f32 dense reference — the
    /// bound comfy-kitchen's `test_exact_branch_quantization_error` declares for the all-routed
    /// exact branch (per-token INT8 Q/K, per-channel INT8 V, per-tile U8 P: ~1% rel-L2). Every
    /// query block routes all of its key blocks (counted), bf16 and f16, ragged and aligned
    /// lengths, batch > 1.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_sol_ratio_one_is_dense_attention() -> candle_core::Result<()> {
        for (b, h, t, dtype) in [
            (1, 8, 4096, DType::BF16),
            (2, 4, 3137, DType::BF16),
            (1, 4, 1000, DType::F16),
            (1, 2, 40, DType::BF16),
        ] {
            let (q, k, v) = qkv(b, h, t, dtype)?;
            let (got, stats) = sol_attention_with_stats(&q, &k, &v, SCALE, &SolConfig::top_k(1.0))?;
            assert_eq!(stats.routed, stats.total, "every block routed: {stats:?}");
            let dense = dense_attention_reference(&q, &k, &v, SCALE)?;
            let (cos, rel) = similarity(&got, &dense)?;
            println!("ratio 1.0 [{b},{h},{t}] {dtype:?}: cosine {cos:.6} rel-L2 {rel:.4e}");
            assert!(
                cos >= 0.999 && rel < 0.016,
                "[{b},{h},{t}] {dtype:?}: cos {cos} rel {rel}"
            );
            assert_eq!(got.dims(), q.dims());
            assert_eq!(got.dtype(), dtype);
        }
        Ok(())
    }

    /// AC1: the chunked-query variant is bitwise the unchunked call, for top-k and tau selection,
    /// with sinks, at window sizes that do and do not divide the block count.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_sol_chunked_equals_unchunked() -> candle_core::Result<()> {
        let (q, k, v) = qkv(1, 4, 64 * 150 + 17, DType::BF16)?;
        for base in [
            SolConfig::top_k(0.2),
            SolConfig {
                selection: SolSelection::Tau(1.4),
                sink_blocks: (0, 2),
                sink_q: (3, 5),
                ..SolConfig::top_k(0.2)
            },
        ] {
            let whole = sol_attention(&q, &k, &v, SCALE, &base)?;
            for chunk in [1, 7, 64, 150] {
                let cfg = SolConfig {
                    query_chunk_blocks: chunk,
                    ..base
                };
                let got = sol_attention(&q, &k, &v, SCALE, &cfg)?;
                let diff = (&got.to_dtype(DType::F32)? - &whole.to_dtype(DType::F32)?)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?;
                assert_eq!(diff, 0.0, "chunk {chunk} {base:?}");
            }
        }
        Ok(())
    }

    /// The kernel follows the f32 reference at sparse budgets (comfy-kitchen's bars: top-k 0.995,
    /// tau 0.998 cosine — INT8 rounding can flip a boundary block), and a bigger budget moves it
    /// toward dense attention.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_sol_matches_the_reference_at_sparse_budgets() -> candle_core::Result<()> {
        for t in [4096, 1000, 3137] {
            let (q, k, v) = qkv(1, 4, t, DType::BF16)?;
            let dense = dense_attention_reference(&q, &k, &v, SCALE)?;
            let mut last = 0f32;
            for ratio in [0.05, 0.2, 0.5] {
                let cfg = SolConfig::top_k(ratio);
                let got = sol_attention(&q, &k, &v, SCALE, &cfg)?;
                let want = sol_attention_reference(&q, &k, &v, SCALE, &cfg)?;
                let (cos, rel) = similarity(&got, &want)?;
                let (dcos, _) = similarity(&got, &dense)?;
                println!("t={t} top-k {ratio}: vs reference cos {cos:.6} rel {rel:.3e}; vs dense cos {dcos:.6}");
                assert!(cos > 0.995, "t={t} ratio {ratio}: {cos}");
                assert!(
                    dcos >= last - 1e-4,
                    "t={t}: budget {ratio} moved away from dense"
                );
                last = dcos;
            }
            for tau in [0.5, 2.0] {
                let cfg = SolConfig {
                    selection: SolSelection::Tau(tau),
                    ..SolConfig::top_k(0.5)
                };
                let got = sol_attention(&q, &k, &v, SCALE, &cfg)?;
                let (cos, _) =
                    similarity(&got, &sol_attention_reference(&q, &k, &v, SCALE, &cfg)?)?;
                println!("t={t} tau {tau}: vs reference cos {cos:.6}");
                assert!(cos > 0.998, "t={t} tau {tau}: {cos}");
            }
        }
        Ok(())
    }

    /// A sink-dependent fixture: key block 0 holds keys that point exactly along every query, in
    /// +u / -u pairs, so its pooled mean (and centroid score) is ~0 and top-k never routes it, while
    /// half its tokens carry most of the softmax mass. With block 0 a forced sink the kernel stays
    /// close to dense attention; without the forced sink (the mutation: drop the forced blocks) the
    /// same budget loses that mass and fails the bound.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_sol_forced_sinks_carry_a_sink_dependent_fixture() -> candle_core::Result<()> {
        let dev = device();
        let (h, t, d) = (2usize, 64 * 64, SOL_HEAD_DIM);
        let u = Tensor::randn(0f32, 1.0, (1, h, 1, d), &dev)?;
        let u = u.broadcast_div(&u.sqr()?.sum_keepdim(3)?.sqrt()?)?; // unit vector per head
        let noise = |s: f64| -> candle_core::Result<Tensor> {
            Tensor::randn(0f32, 1.0, (1, h, t, d), &dev)? * s
        };
        // Queries along u (plus noise); keys noise except block 0: alternating +-8u.
        let q = (u.broadcast_as((1, h, t, d))? * 4.0)?.add(&noise(0.3)?)?;
        let signs: Vec<f32> = (0..t)
            .map(|i| {
                if i >= 64 {
                    0.0
                } else if i % 2 == 0 {
                    8.0
                } else {
                    -8.0
                }
            })
            .collect();
        let signs = Tensor::from_vec(signs, (1, 1, t, 1), &dev)?;
        let k = u
            .broadcast_mul(&signs.broadcast_as((1, h, t, 1))?)?
            .add(&noise(0.3)?)?;
        // Block 0's values are distinctive, so losing their mass shows in the output.
        let vmark: Vec<f32> = (0..t).map(|i| if i < 64 { 3.0 } else { 0.0 }).collect();
        let vmark = Tensor::from_vec(vmark, (1, 1, t, 1), &dev)?;
        let v = noise(1.0)?.broadcast_add(&vmark)?;
        let (q, k, v) = (
            q.to_dtype(DType::BF16)?,
            k.to_dtype(DType::BF16)?,
            v.to_dtype(DType::BF16)?,
        );
        let dense = dense_attention_reference(&q, &k, &v, SCALE)?;
        let sinks = SolConfig {
            sink_blocks: (0, 1),
            ..SolConfig::top_k(0.1)
        };
        let with_sink = similarity(&sol_attention(&q, &k, &v, SCALE, &sinks)?, &dense)?;
        let without = similarity(
            &sol_attention(&q, &k, &v, SCALE, &SolConfig::top_k(0.1))?,
            &dense,
        )?;
        println!(
            "sink fixture: with forced sink cos {:.6} rel {:.3e}; without cos {:.6} rel {:.3e}",
            with_sink.0, with_sink.1, without.0, without.1
        );
        assert!(
            with_sink.0 >= 0.99,
            "with the forced sink block: {with_sink:?}"
        );
        assert!(
            without.0 < 0.99,
            "the fixture must depend on the forced sink: {without:?}"
        );
        Ok(())
    }

    /// One Wan-shaped call (A14B 480x832x81: 40 heads, 32,760 tokens) per top-k budget, its
    /// density and ms (quick smoke; the bench example has the alternated A/B rows).
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_sol_wan_shape_smoke() -> candle_core::Result<()> {
        let (q, k, v) = qkv(1, 40, 32_760, DType::BF16)?;
        let dev = device();
        for ratio in [0.1f32, 0.25, 0.5, 1.0] {
            let cfg = SolConfig::top_k(ratio);
            let (_, stats) = sol_attention_with_stats(&q, &k, &v, SCALE, &cfg)?;
            for _ in 0..3 {
                drop(sol_attention(&q, &k, &v, SCALE, &cfg)?);
            }
            dev.synchronize()?;
            let t0 = std::time::Instant::now();
            for _ in 0..5 {
                drop(sol_attention(&q, &k, &v, SCALE, &cfg)?);
            }
            dev.synchronize()?;
            let ms = t0.elapsed().as_secs_f64() * 1e3 / 5.0;
            println!(
                "wan 480p ratio {ratio}: density {:.4}, {ms:.2} ms/call",
                stats.density()
            );
            assert!(stats.density() > 0.0 && stats.density() <= 1.0);
        }
        Ok(())
    }

    /// `t`'s values as a *contiguous* view whose storage offset is `off` elements: a flat buffer
    /// narrowed at `off` and reshaped back, exactly the layout `Tensor::contiguous` returns
    /// unchanged.
    fn offset_view(t: &Tensor, off: usize) -> candle_core::Result<Tensor> {
        let pad = Tensor::zeros(off, t.dtype(), t.device())?;
        let flat = Tensor::cat(&[&pad, &t.flatten_all()?], 0)?;
        let view = flat.narrow(0, off, t.elem_count())?.reshape(t.dims())?;
        assert!(view.is_contiguous() && view.layout().start_offset() == off);
        Ok(view)
    }

    /// Review regression (PR #1530): a contiguous bf16/f16 operand whose storage offset is not a
    /// multiple of 8 elements is copied to a zero-offset buffer before launch (the `uint4` loads
    /// need 16-byte rows), so the call succeeds and is bitwise the aligned call. Before the fix
    /// `Tensor::contiguous` returned the view unchanged and the kernels read misaligned
    /// addresses.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_sol_contiguous_view_at_an_unaligned_offset() -> candle_core::Result<()> {
        for (b, h, t, dtype) in [(1, 1, 64, DType::BF16), (1, 2, 1000, DType::F16)] {
            let (q, k, v) = qkv(b, h, t, dtype)?;
            let cfg = SolConfig::top_k(0.5);
            let want = sol_attention(&q, &k, &v, SCALE, &cfg)?;
            let (qo, ko, vo) = (
                offset_view(&q, 1)?,
                offset_view(&k, 3)?,
                offset_view(&v, 5)?,
            );
            let got = sol_attention(&qo, &ko, &vo, SCALE, &cfg)?;
            let diff = (&got.to_dtype(DType::F32)? - &want.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert_eq!(diff, 0.0, "[{b},{h},{t}] {dtype:?}");
        }
        Ok(())
    }
}
