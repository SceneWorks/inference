//! The sparse Mixture-of-Experts feed-forward — the **one** routing module every MoE family in this
//! crate reaches (sc-24440): Qwen2-MoE and DeepSeek-V2 through the generic
//! [`CausalLm`](crate::models::CausalLm), Qwen3.6-MoE (`qwen3_5_moe`) through
//! [`Qwen35Model`](crate::models::Qwen35Model).
//!
//! **Routing is on the device.** The router's softmax, the top-k (`arg_sort` + `narrow` +
//! `gather`) and the renormalization / routed scaling are tensor ops; the router this replaced
//! pulled every token's probabilities to the host each layer to pick the experts there, which
//! drained the GPU pipeline once per MoE layer and made every MoE model refuse CUDA-graph capture
//! (`moe_router_host_read`).
//!
//! **Dispatch** depends on how the routed experts are stored:
//!
//! * A **dense** bank (every expert a plain `[out, in]` weight, the unquantized load) is held
//!   stacked (`[experts, out, in]`). A decode-sized step — at most [`DEVICE_DISPATCH_MAX_ROWS`]
//!   tokens, and no more (token, slot) pairs than experts — gathers each pair's expert weights by
//!   its device index and runs one batched matmul per projection: no host read and shapes fixed by
//!   the step, so nothing in it stops a graph capture (whether a recorded step replays is still
//!   the graph runner's census to decide — candle uploads some ops' layouts from the host). Every
//!   (token, slot) pair is its own one-row product, so a row never depends on its batch; the
//!   products are accumulated in ascending expert order, the order the grouped loop below adds
//!   them, so a one-token step reproduces the grouped dispatch bit for bit.
//! * Larger batches (a prefill), and banks Candle cannot index by a device id — GGML-quantized,
//!   NVFP4 or Prism experts, each its own kernel with no gathered matmul — are dispatched
//!   grouped: the device-computed routes are read back once (one counted host sync), each expert
//!   runs on just its tokens, and the weighted outputs are `index_add`ed back. That is the
//!   routing arithmetic of the pre-change block, fed by the device router. A model with such a
//!   bank therefore still reads the device during a decode step and says so in
//!   [`SparseMoe::graph_refusal`] (`moe_expert_host_dispatch`).

use candle_core::{DType, Tensor};

use crate::error::{Error, Result};
use crate::primitives::host_sync::note_host_sync;
use crate::primitives::nn::swiglu;
use crate::primitives::projection::{Projection, WeightCensus};

/// The most tokens a step may route through the stacked device dispatch. Decode steps (one token
/// per sequence, a handful of sequences) fit; a prefill goes grouped, where each expert's weights
/// are read once for all of its tokens instead of gathered once per (token, slot) pair.
pub const DEVICE_DISPATCH_MAX_ROWS: usize = 8;

/// The graph-capture refusal of a model whose MoE bank is dispatched from host-read routes.
pub const REASON_EXPERT_HOST_DISPATCH: &str = "moe_expert_host_dispatch";

/// A SwiGLU MLP over three projections: a routed expert of a bank Candle cannot stack, or the
/// always-on shared expert.
pub struct SwiGlu {
    /// Gate projection.
    pub gate: Projection,
    /// Up projection.
    pub up: Projection,
    /// Down projection.
    pub down: Projection,
}

impl SwiGlu {
    /// `down(silu(gate(x)) · up(x))`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(x)?;
        let up = self.up.forward(x)?;
        self.down.forward(&swiglu(&g, &up)?)
    }

    /// Add the three projections to a census.
    pub fn record(&self, census: &mut WeightCensus) {
        for p in [&self.gate, &self.up, &self.down] {
            census.projections.record(p);
        }
    }

    fn part(&self, part: ExpertPart) -> &Projection {
        match part {
            ExpertPart::Gate => &self.gate,
            ExpertPart::Up => &self.up,
            ExpertPart::Down => &self.down,
        }
    }
}

/// One projection of a routed expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertPart {
    /// The gate projection.
    Gate,
    /// The up projection.
    Up,
    /// The down projection.
    Down,
}

/// The routed experts.
enum ExpertBank {
    /// Dense experts stacked per projection: `gate` / `up` `[experts, inter, hidden]`, `down`
    /// `[experts, hidden, inter]`.
    Stacked {
        gate: Tensor,
        up: Tensor,
        down: Tensor,
    },
    /// Experts Candle cannot index by a device id (quantized / NVFP4 / Prism, or a mixed bank).
    PerExpert(Vec<SwiGlu>),
}

impl ExpertBank {
    fn new(experts: Vec<SwiGlu>) -> Result<Self> {
        let dense = |p: &Projection| match p {
            Projection::Dense(l) if l.bias().is_none() => Some(l.weight().clone()),
            _ => None,
        };
        let stackable = experts.iter().all(|e| {
            [&e.gate, &e.up, &e.down]
                .into_iter()
                .all(|p| dense(p).is_some())
        });
        if !stackable {
            return Ok(Self::PerExpert(experts));
        }
        let mut parts: [Vec<Tensor>; 3] = Default::default();
        for e in &experts {
            for (i, p) in [&e.gate, &e.up, &e.down].into_iter().enumerate() {
                parts[i].extend(dense(p));
            }
        }
        let [gate, up, down] = parts;
        Ok(Self::Stacked {
            gate: Tensor::stack(&gate, 0)?,
            up: Tensor::stack(&up, 0)?,
            down: Tensor::stack(&down, 0)?,
        })
    }

    /// Expert `e`'s `part` applied to `x` `[n, in]`.
    fn project(&self, e: usize, part: ExpertPart, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Stacked { gate, up, down } => {
                let w = match part {
                    ExpertPart::Gate => gate,
                    ExpertPart::Up => up,
                    ExpertPart::Down => down,
                };
                // The same `x · wᵀ` a dense `Linear` runs on its own `[out, in]` weight.
                Ok(x.matmul(&w.get(e)?.t()?)?)
            }
            Self::PerExpert(experts) => experts[e].part(part).forward(x),
        }
    }

    /// Expert `e`'s SwiGLU on `x` `[n, hidden]`.
    fn expert(&self, e: usize, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Stacked { .. } => {
                let g = self.project(e, ExpertPart::Gate, x)?;
                let up = self.project(e, ExpertPart::Up, x)?;
                self.project(e, ExpertPart::Down, &swiglu(&g, &up)?)
            }
            Self::PerExpert(experts) => experts[e].forward(x),
        }
    }
}

/// How the router turns probabilities into expert weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeRouting {
    /// Experts each token is routed to.
    pub experts_per_tok: usize,
    /// Renormalize the top-k probabilities to sum to 1 (Qwen3.6-MoE; Qwen2-MoE when its config
    /// says so). Otherwise they are multiplied by `routed_scaling_factor`.
    pub norm_topk_prob: bool,
    /// Multiplier on the un-normalized routed weights (DeepSeek-V2); `1.0` for Qwen2-MoE. Ignored
    /// when `norm_topk_prob`.
    pub routed_scaling_factor: f32,
}

/// A sparse Mixture-of-Experts FFN: a softmax router over the routed experts (top-k per token)
/// plus an always-on shared expert, sigmoid-gated (Qwen2-MoE, Qwen3.6-MoE) or added ungated
/// (DeepSeek-V2). `n_group` / `topk_group` group-limited routing (DeepSeek-V2-236B / V3) is not
/// modelled — the verification model (V2-Lite) uses plain greedy top-k.
pub struct SparseMoe {
    /// Router weight `[experts, hidden]`.
    router: Tensor,
    bank: ExpertBank,
    num_experts: usize,
    shared: SwiGlu,
    /// Shared-expert sigmoid gate `[1, hidden]`; `None` ⇒ the shared expert is added ungated.
    shared_gate: Option<Tensor>,
    routing: MoeRouting,
}

impl SparseMoe {
    /// Assemble the block from the router `[experts, hidden]`, the per-expert SwiGLUs (stacked
    /// here when every projection is dense), the shared expert and its optional gate.
    pub fn new(
        router: Tensor,
        experts: Vec<SwiGlu>,
        shared: SwiGlu,
        shared_gate: Option<Tensor>,
        routing: MoeRouting,
    ) -> Result<Self> {
        let num_experts = experts.len();
        if num_experts == 0 || router.dim(0)? != num_experts {
            return Err(Error::Config(format!(
                "an MoE router over {} experts needs as many experts, got {num_experts}",
                router.dim(0)?
            )));
        }
        Ok(Self {
            router,
            bank: ExpertBank::new(experts)?,
            num_experts,
            shared,
            shared_gate,
            routing,
        })
    }

    /// The routed-expert count.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    fn top_k(&self) -> usize {
        self.routing.experts_per_tok.clamp(1, self.num_experts)
    }

    /// Why a step through this block cannot be captured as a CUDA graph, if it cannot: a bank
    /// Candle cannot index by a device id is dispatched from host-read routes.
    pub fn graph_refusal(&self) -> Option<&'static str> {
        match self.bank {
            ExpertBank::Stacked { .. } => None,
            ExpertBank::PerExpert(_) => Some(REASON_EXPERT_HOST_DISPATCH),
        }
    }

    /// Route every token on the device: the top-k expert ids `[t, k]` (u32, most probable first)
    /// and their f32 weights `[t, k]`, from `xf` `[t, hidden]`.
    ///
    /// The arithmetic is the pre-change host router's, op for op: an f32 softmax, the k most
    /// probable experts, then either `p / Σ top-k p` (the sum taken in the same most-probable-first
    /// order) or `p · routed_scaling_factor`. The host router clamped the sum at
    /// `f32::MIN_POSITIVE`; it can never reach it — the top-k include the largest probability,
    /// which is at least `1 / experts` — so no clamp is taken here.
    pub fn route(&self, xf: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = self.top_k();
        let logits = xf.matmul(&self.router.t()?)?; // [t, E]
        let probs = candle_nn::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?;
        let ids = probs
            .arg_sort_last_dim(false)?
            .narrow(1, 0, k)?
            .contiguous()?; // [t, k]
        let top = probs.gather(&ids, 1)?; // [t, k]
        let weights = if self.routing.norm_topk_prob {
            let mut denom = top.narrow(1, 0, 1)?;
            for j in 1..k {
                denom = (denom + top.narrow(1, j, 1)?)?;
            }
            top.broadcast_div(&denom)?
        } else {
            (top * f64::from(self.routing.routed_scaling_factor))?
        };
        Ok((ids, weights))
    }

    /// The block's output for `x` `[batch, seq, hidden]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let t = b * s;
        let xf = x.reshape((t, h))?;
        let (ids, weights) = self.route(&xf)?;
        let routed = match &self.bank {
            ExpertBank::Stacked { gate, up, down }
                if t <= DEVICE_DISPATCH_MAX_ROWS && t * self.top_k() <= self.num_experts =>
            {
                dispatch_stacked(&xf, &ids, &weights, [gate, up, down])?
            }
            _ => self.dispatch_grouped(&xf, &ids, &weights)?,
        };

        // Always-on shared expert: Qwen2 / Qwen3.6 gate it by sigmoid(x · shared_gateᵀ);
        // DeepSeek packs several shared experts into one MLP and adds them ungated.
        let shared = self.shared.forward(&xf)?;
        let shared = match &self.shared_gate {
            Some(g) => {
                let sg = candle_nn::ops::sigmoid(&xf.matmul(&g.t()?)?)?; // [t, 1]
                shared.broadcast_mul(&sg)?
            }
            None => shared,
        };
        Ok((routed + shared)?.reshape((b, s, h))?)
    }

    /// Group the routed (token, weight) pairs per expert on the host, run each expert on just its
    /// tokens, and `index_add` the weighted outputs back.
    fn dispatch_grouped(&self, xf: &Tensor, ids: &Tensor, weights: &Tensor) -> Result<Tensor> {
        let (t, h) = xf.dims2()?;
        let (dtype, device) = (xf.dtype(), xf.device());
        let mut routed: Vec<Vec<(u32, f32)>> = vec![Vec::new(); self.num_experts];
        for (ti, row) in read_routes(ids, weights)?.into_iter().enumerate() {
            for (e, w) in row {
                routed[e as usize].push((ti as u32, w));
            }
        }
        let mut out = Tensor::zeros((t, h), dtype, device)?;
        for (e, toks) in routed.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let n = toks.len();
            let idx = Tensor::from_vec(
                toks.iter().map(|&(ti, _)| ti).collect::<Vec<u32>>(),
                (n,),
                device,
            )?;
            let wts = Tensor::from_vec(
                toks.iter().map(|&(_, w)| w).collect::<Vec<f32>>(),
                (n, 1),
                device,
            )?
            .to_dtype(dtype)?;
            let xe = xf.index_select(&idx, 0)?; // [n, h]
            let ye = self.bank.expert(e, &xe)?.broadcast_mul(&wts)?; // [n, h]
            out = out.index_add(&idx, &ye, 0)?;
        }
        Ok(out)
    }

    /// Expert `e`'s `part` applied to `x` `[n, in]` — the per-expert view a test compares
    /// representations through, whether the bank is stacked or not.
    #[cfg(test)]
    pub(crate) fn expert_projection(
        &self,
        e: usize,
        part: ExpertPart,
        x: &Tensor,
    ) -> Result<Tensor> {
        self.bank.project(e, part, x)
    }

    /// Which representation expert `e`'s `part` holds.
    #[cfg(test)]
    pub(crate) fn expert_kind(
        &self,
        e: usize,
        part: ExpertPart,
    ) -> crate::primitives::projection::ProjectionKind {
        match &self.bank {
            ExpertBank::Stacked { .. } => crate::primitives::projection::ProjectionKind::Dense,
            ExpertBank::PerExpert(experts) => experts[e].part(part).kind(),
        }
    }

    /// Add the block's weights to a census. A stacked dense bank counts as its per-expert dense
    /// projections, exactly as the un-stacked bank did.
    pub fn record(&self, census: &mut WeightCensus) {
        census.record_tensor(&self.router);
        if let Some(gate) = &self.shared_gate {
            census.record_tensor(gate);
        }
        match &self.bank {
            ExpertBank::Stacked { gate, up, down } => {
                for w in [gate, up, down] {
                    let experts = w.dim(0).unwrap_or(0) as u64;
                    let tally = &mut census.projections.dense;
                    tally.count += experts;
                    tally.params += w.elem_count() as u64;
                    tally.resident_bytes += (w.elem_count() * w.dtype().size_in_bytes()) as u64;
                }
            }
            ExpertBank::PerExpert(experts) => {
                for expert in experts {
                    expert.record(census);
                }
            }
        }
        self.shared.record(census);
    }
}

/// The stacked device dispatch: gather each (token, slot) pair's expert weights by its device
/// index and run one batched matmul per projection. Everything stays on the device and every
/// shape is fixed by `(t, k)`, so the step is graph-replayable.
fn dispatch_stacked(
    xf: &Tensor,
    ids: &Tensor,
    weights: &Tensor,
    [gate, up, down]: [&Tensor; 3],
) -> Result<Tensor> {
    let (t, h) = xf.dims2()?;
    let k = ids.dim(1)?;
    // Visit each token's experts in ascending id order: the grouped dispatch adds them in that
    // order, and matching it keeps the two paths bit-identical.
    let (ids, perm) = ids.sort_last_dim(true)?;
    let weights = weights.gather(&perm, 1)?.to_dtype(xf.dtype())?; // [t, k]
    let flat = ids.flatten_all()?; // [t·k]
    let xs = xf
        .unsqueeze(1)?
        .broadcast_as((t, k, h))?
        .contiguous()?
        .reshape((t * k, 1, h))?;
    let project = |x: &Tensor, w: &Tensor| -> Result<Tensor> {
        Ok(x.matmul(&w.index_select(&flat, 0)?.transpose(1, 2)?)?)
    };
    let g = project(&xs, gate)?; // [t·k, 1, inter]
    let u = project(&xs, up)?;
    let y = project(&swiglu(&g, &u)?, down)?; // [t·k, 1, h]
    let y = y
        .reshape((t, k, h))?
        .broadcast_mul(&weights.unsqueeze(2)?)?;
    let mut out = Tensor::zeros((t, h), xf.dtype(), xf.device())?;
    for j in 0..k {
        out = (out + y.narrow(1, j, 1)?.squeeze(1)?)?;
    }
    Ok(out)
}

/// Read the routes back to the host — the grouped dispatch's one device→host transfer, counted
/// as one host sync. The ids ride along as f32 (exact below 2²⁴ experts) so ids and weights move
/// in a single copy.
fn read_routes(ids: &Tensor, weights: &Tensor) -> Result<Vec<Vec<(u32, f32)>>> {
    let k = ids.dim(1)?;
    let both = Tensor::cat(&[&ids.to_dtype(DType::F32)?, weights], 1)?;
    note_host_sync();
    let rows = both.to_vec2::<f32>()?;
    Ok(rows
        .into_iter()
        .map(|row| (0..k).map(|j| (row[j] as u32, row[k + j])).collect())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::host_sync::host_sync_count;
    use crate::primitives::projection::QuantSpec;
    use crate::primitives::sampler::{SplitMix64, TokenRng};
    use candle_core::Device;

    fn randn(shape: &[usize], rng: &mut SplitMix64) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn block(
        e: usize,
        h: usize,
        inter: usize,
        routing: MoeRouting,
        quant: Option<QuantSpec>,
    ) -> SparseMoe {
        let mut rng = SplitMix64::new(0x2444_0100);
        let router = randn(&[e, h], &mut rng);
        let swiglu = |rng: &mut SplitMix64, quant: Option<QuantSpec>| SwiGlu {
            gate: Projection::load(randn(&[inter, h], rng), quant).unwrap(),
            up: Projection::load(randn(&[inter, h], rng), quant).unwrap(),
            down: Projection::load(randn(&[h, inter], rng), quant).unwrap(),
        };
        let experts = (0..e).map(|_| swiglu(&mut rng, quant)).collect();
        let shared = swiglu(&mut rng, None);
        let gate = randn(&[1, h], &mut rng);
        SparseMoe::new(router, experts, shared, Some(gate), routing).unwrap()
    }

    const NORM: MoeRouting = MoeRouting {
        experts_per_tok: 2,
        norm_topk_prob: true,
        routed_scaling_factor: 1.0,
    };
    const SCALED: MoeRouting = MoeRouting {
        experts_per_tok: 3,
        norm_topk_prob: false,
        routed_scaling_factor: 2.5,
    };

    /// The device router picks the top-k and weights them exactly as the host router did.
    #[test]
    fn device_routes_match_a_host_top_k() {
        for routing in [NORM, SCALED] {
            let moe = block(8, 32, 16, routing, None);
            let mut rng = SplitMix64::new(9);
            let x = randn(&[5, 32], &mut rng);
            let (ids, weights) = moe.route(&x).unwrap();
            let probs =
                candle_nn::ops::softmax_last_dim(&x.matmul(&moe.router.t().unwrap()).unwrap())
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap();
            let ids = ids.to_vec2::<u32>().unwrap();
            let weights = weights.to_vec2::<f32>().unwrap();
            for (ti, row) in probs.iter().enumerate() {
                let mut idx: Vec<usize> = (0..8).collect();
                idx.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]));
                let top = &idx[..routing.experts_per_tok];
                let denom: f32 = top.iter().map(|&e| row[e]).sum();
                for (j, &e) in top.iter().enumerate() {
                    assert_eq!(ids[ti][j] as usize, e, "token {ti} slot {j}");
                    let want = if routing.norm_topk_prob {
                        row[e] / denom
                    } else {
                        row[e] * routing.routed_scaling_factor
                    };
                    assert_eq!(
                        weights[ti][j].to_bits(),
                        want.to_bits(),
                        "token {ti} slot {j}"
                    );
                }
            }
        }
    }

    /// A single-token step through the stacked device dispatch is bit-identical to the grouped
    /// dispatch (each routed expert sees exactly one row either way). A multi-row device step is
    /// row-for-row bit-identical to running its tokens one at a time — every (token, slot) pair is
    /// its own one-row product — and agrees with the grouped dispatch (whose per-expert GEMMs
    /// span several rows) to rounding.
    #[test]
    fn stacked_device_dispatch_matches_the_grouped_dispatch() {
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for routing in [NORM, SCALED] {
            let moe = block(8, 32, 16, routing, None);
            let ExpertBank::Stacked { gate, up, down } = &moe.bank else {
                panic!("a dense bank stacks")
            };
            let device = |x: &Tensor| {
                let (ids, weights) = moe.route(x).unwrap();
                host(&dispatch_stacked(x, &ids, &weights, [gate, up, down]).unwrap())
            };
            let grouped = |x: &Tensor| {
                let (ids, weights) = moe.route(x).unwrap();
                host(&moe.dispatch_grouped(x, &ids, &weights).unwrap())
            };
            let max_t = 8 / routing.experts_per_tok;
            for t in 1..=max_t {
                let mut rng = SplitMix64::new(0x2444_0200 + t as u64);
                let x = randn(&[t, 32], &mut rng);
                let (d, g) = (device(&x), grouped(&x));
                if t == 1 {
                    assert_eq!(bits(&d), bits(&g), "{routing:?}: device vs grouped");
                } else {
                    let md = d
                        .iter()
                        .zip(&g)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f32::max);
                    assert!(md < 1e-6, "t={t} {routing:?}: device vs grouped {md}");
                }
                for ti in 0..t {
                    let row = device(&x.narrow(0, ti, 1).unwrap());
                    assert_eq!(
                        bits(&d[ti * 32..(ti + 1) * 32]),
                        bits(&row),
                        "t={t} {routing:?}: row {ti} depends on its batch"
                    );
                }
            }
        }
    }

    /// A decode step through a dense bank reads nothing back; a quantized bank's grouped dispatch
    /// reads its routes once — and says so as its graph refusal.
    #[test]
    fn host_reads_follow_the_bank() {
        let x = randn(&[1, 1, 64], &mut SplitMix64::new(3));
        let dense = block(8, 64, 64, NORM, None);
        let before = host_sync_count();
        dense.forward(&x).unwrap();
        assert_eq!(host_sync_count() - before, 0);
        assert_eq!(dense.graph_refusal(), None);

        let q8 = block(8, 64, 64, NORM, Some(QuantSpec::q8()));
        assert_eq!(
            q8.expert_kind(0, ExpertPart::Gate),
            crate::primitives::projection::ProjectionKind::Ggml
        );
        let before = host_sync_count();
        q8.forward(&x).unwrap();
        assert_eq!(host_sync_count() - before, 1);
        assert_eq!(q8.graph_refusal(), Some(REASON_EXPERT_HOST_DISPATCH));
    }

    /// Every device→host read in this module goes through [`read_routes`], so the host-sync
    /// counter cannot miss one.
    #[test]
    fn host_reads_go_through_read_routes() {
        const RAW_READS: [&str; 5] = [".to_vec0", ".to_vec1", ".to_vec2", ".to_vec3", ".to_scalar"];
        let source = include_str!("moe.rs");
        let code = &source[..source
            .find("#[cfg(test)]\nmod tests {")
            .expect("tests marker")];
        let mut current_fn = "";
        let mut raw = Vec::new();
        for (n, line) in code.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            if let Some(at) = trimmed.find("fn ") {
                current_fn = trimmed[at + 3..]
                    .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .next()
                    .unwrap_or_default();
            }
            if RAW_READS.iter().any(|r| line.contains(r)) && current_fn != "read_routes" {
                raw.push(format!("line {}: {}", n + 1, line.trim()));
            }
        }
        assert!(raw.is_empty(), "uncounted host reads:\n{}", raw.join("\n"));
    }
}
