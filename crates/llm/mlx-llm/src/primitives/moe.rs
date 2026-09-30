//! The sparse Mixture-of-Experts feed-forward — the **one** routing module every MoE family in this
//! crate reaches (sc-24440): Qwen2-MoE and DeepSeek-V2 through the generic
//! [`CausalLm`](crate::models::CausalLm), Qwen3.6-MoE (`qwen3_5_moe`) through
//! [`Qwen35Model`](crate::models::Qwen35Model).
//!
//! Routing never leaves the device. The router's softmax, the top-k (`argpartition`), the
//! renormalization / routed scaling and the expert dispatch are all lazy MLX ops, so a decode
//! step's MoE layers add nothing that forces an evaluation mid-step — the router this replaced
//! read its probabilities back with `as_slice` once per layer to pick the experts on the host,
//! draining the GPU pipeline every layer.
//!
//! The routed experts are held **stacked** (`[experts, out, in]` per projection, dense or
//! group-wise quantized) and dispatched with `gather_mm` / `gather_qmm`: each (token, slot) pair
//! multiplies against the expert its index names, so only the routed experts' weights are read.
//! Prefill-sized batches sort the pairs by expert first (the `sorted_indices` fast path) and
//! unsort afterwards, exactly as mlx-lm's `SwitchGLU` does.

use mlx_rs::ops::indexing::take_along_axis;
use mlx_rs::ops::{
    add, argpartition_axis, argsort_axis, divide, floor_divide, gather_mm, gather_qmm, maximum,
    multiply, negative, quantize, sigmoid, softmax_axis, split_sections, stack_axis, sum_axis,
};
use mlx_rs::{Array, Dtype};

use crate::error::{Error, Result};
use crate::primitives::nn::{linear, silu};
use crate::primitives::projection::{Projection, QuantSpec};

/// Routed (token, slot) pairs at or above this count are sorted by expert before the gathered
/// matmuls — mlx-lm's `SwitchGLU` threshold. Below it (every single-sequence decode step) the
/// indices go in unsorted, as the router produced them.
const SORT_THRESHOLD: i32 = 64;

/// One routed-expert projection, stacked across the bank: expert `e`'s `[out, in]` weight is
/// slice `e` of the leading axis.
#[derive(Debug)]
pub enum SwitchLinear {
    /// Dense `[experts, out, in]` weights, dispatched with `gather_mm`.
    Dense {
        /// The stacked weight.
        weight: Array,
    },
    /// Group-wise quantized weights (`[experts, out, packed]` plus per-group scales / biases),
    /// dispatched with `gather_qmm`.
    Quantized {
        /// Packed quantized weight.
        weight: Array,
        /// Per-group scales.
        scales: Array,
        /// Per-group biases (zero-points).
        biases: Array,
        /// Elements per quantization group.
        group_size: i32,
        /// Bits per weight.
        bits: i32,
    },
}

impl SwitchLinear {
    /// From a dense stacked `[experts, out, in]` weight, quantized group-wise (per expert row,
    /// the same groups a per-expert quantize forms) when `quant` is set.
    pub fn load(weight: Array, quant: Option<QuantSpec>) -> Result<Self> {
        match quant {
            None => Ok(Self::Dense { weight }),
            Some(q) => {
                let (weight, scales, biases) = quantize(&weight, q.group_size, q.bits)?;
                Ok(Self::Quantized {
                    weight,
                    scales,
                    biases,
                    group_size: q.group_size,
                    bits: q.bits,
                })
            }
        }
    }

    /// Stack per-expert projections (a checkpoint that stores each expert under its own key).
    ///
    /// Every part must share one representation: all dense, or all quantized with one group size
    /// and bit width, and none may carry an additive bias (no MoE checkpoint has expert biases).
    /// A mixed bank, or a Prism one, has no gathered kernel and is refused at load.
    pub fn stack(parts: Vec<Projection>) -> Result<Self> {
        let Some(first) = parts.first() else {
            return Err(Error::Config(
                "an MoE expert bank needs at least one expert".into(),
            ));
        };
        match first {
            Projection::Dense { .. } => {
                let mut weights = Vec::with_capacity(parts.len());
                for p in &parts {
                    match p {
                        Projection::Dense { weight, bias: None } => weights.push(weight),
                        _ => return Err(mixed_bank()),
                    }
                }
                Ok(Self::Dense {
                    weight: stack_axis(&weights, 0)?,
                })
            }
            Projection::Quantized(q0) => {
                let (group_size, bits) = (q0.group_size, q0.bits);
                let (mut ws, mut ss, mut bs) = (Vec::new(), Vec::new(), Vec::new());
                for p in &parts {
                    match p {
                        Projection::Quantized(q)
                            if q.group_size == group_size && q.bits == bits && q.bias.is_none() =>
                        {
                            ws.push(&q.weight);
                            ss.push(&q.scales);
                            bs.push(&q.biases);
                        }
                        _ => return Err(mixed_bank()),
                    }
                }
                Ok(Self::Quantized {
                    weight: stack_axis(&ws, 0)?,
                    scales: stack_axis(&ss, 0)?,
                    biases: stack_axis(&bs, 0)?,
                    group_size,
                    bits,
                })
            }
            Projection::Prism(_) => Err(Error::Unsupported(
                "Prism packed MoE experts have no gathered matmul".into(),
            )),
        }
    }

    /// `x @ weight[idx]ᵀ` for every index: `x` `[.., 1, in]` broadcast against `idx`'s shape,
    /// giving `[.., 1, out]` per index.
    fn forward(&self, x: &Array, idx: &Array, sorted: bool) -> Result<Array> {
        Ok(match self {
            Self::Dense { weight } => gather_mm(x, weight.swap_axes(-1, -2)?, None, idx, sorted)?,
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => gather_qmm(
                x,
                weight,
                scales,
                biases,
                None,
                idx,
                true,
                *group_size,
                *bits,
                sorted,
            )?,
        })
    }

    /// Whether the bank is quantized.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }
}

fn mixed_bank() -> Error {
    Error::Unsupported(
        "MoE experts must share one representation (all dense, or all quantized with one group \
         size / bit width, without bias) to be dispatched with a gathered matmul"
            .into(),
    )
}

/// A SwiGLU MLP over three projections — the always-on shared expert.
#[derive(Debug)]
pub struct SwiGlu {
    /// Gate projection.
    pub gate: Projection,
    /// Up projection.
    pub up: Projection,
    /// Down projection.
    pub down: Projection,
}

impl SwiGlu {
    fn forward(&self, x: &Array) -> Result<Array> {
        let g = silu(&self.gate.forward(x)?)?;
        let up = self.up.forward(x)?;
        self.down.forward(&multiply(&g, &up)?)
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

/// A sparse Mixture-of-Experts FFN: a softmax router over the stacked routed experts (top-k per
/// token) plus an always-on shared expert, optionally sigmoid-gated (Qwen2-MoE, Qwen3.6-MoE) or
/// added ungated (DeepSeek-V2). `n_group` / `topk_group` group-limited routing (DeepSeek-V2-236B /
/// V3) is not modelled — V2-Lite uses plain greedy top-k.
#[derive(Debug)]
pub struct SparseMoe {
    /// Router weight `[experts, hidden]`.
    router: Array,
    gate: SwitchLinear,
    up: SwitchLinear,
    down: SwitchLinear,
    num_experts: usize,
    shared: SwiGlu,
    /// Shared-expert sigmoid gate `[1, hidden]`; `None` ⇒ the shared expert is added ungated.
    shared_gate: Option<Array>,
    routing: MoeRouting,
}

impl SparseMoe {
    /// Assemble the block. `router` is `[experts, hidden]`; `gate` / `up` / `down` are the stacked
    /// routed-expert projections (`[experts, inter, hidden]` / `[experts, hidden, inter]`).
    pub fn new(
        router: Array,
        gate: SwitchLinear,
        up: SwitchLinear,
        down: SwitchLinear,
        shared: SwiGlu,
        shared_gate: Option<Array>,
        routing: MoeRouting,
    ) -> Result<Self> {
        let num_experts = router.shape().first().copied().unwrap_or(0) as usize;
        if num_experts == 0 {
            return Err(Error::Config(
                "an MoE router needs at least one expert".into(),
            ));
        }
        Ok(Self {
            router,
            gate,
            up,
            down,
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

    /// Whether the routed experts are quantized.
    pub fn is_quantized(&self) -> bool {
        self.gate.is_quantized()
    }

    /// Route every token on the device: the top-k expert indices `[t, k]` (uint32) and their
    /// weights `[t, k]` (f32), from `xf` `[t, hidden]`.
    pub fn route(&self, xf: &Array) -> Result<(Array, Array)> {
        let k = self.routing.experts_per_tok.clamp(1, self.num_experts) as i32;
        let logits = linear(xf, &self.router, None)?; // [t, E]
        let probs = softmax_axis(logits.as_dtype(Dtype::Float32)?, -1, true)?;
        // The k largest probabilities (unordered) sit in the first k positions of a partition.
        let part = argpartition_axis(negative(&probs)?, k - 1, -1)?;
        let inds = if k as usize == self.num_experts {
            part
        } else {
            split_sections(&part, &[k], -1)?.swap_remove(0)
        };
        let top = take_along_axis(&probs, &inds, -1)?; // [t, k]
        let weights = if self.routing.norm_topk_prob {
            let denom = sum_axis(&top, -1, true)?;
            let denom = maximum(&denom, Array::from_f32(f32::MIN_POSITIVE))?;
            divide(&top, &denom)?
        } else {
            multiply(&top, Array::from_f32(self.routing.routed_scaling_factor))?
        };
        Ok((inds, weights))
    }

    /// The routed experts' SwiGLU outputs `[t, k, hidden]` for `xf` `[t, hidden]` and `inds`
    /// `[t, k]`.
    fn experts(&self, xf: &Array, inds: &Array) -> Result<Array> {
        let sh = inds.shape();
        let (t, k) = (sh[0], sh[1]);
        let h = xf.shape()[1];
        let swiglu = |x: &Array, idx: &Array, sorted: bool| -> Result<Array> {
            let g = silu(&self.gate.forward(x, idx, sorted)?)?;
            let up = self.up.forward(x, idx, sorted)?;
            self.down.forward(&multiply(&g, &up)?, idx, sorted)
        };
        if t * k >= SORT_THRESHOLD {
            // Sort the (token, slot) pairs by expert so each expert's rows are contiguous, run
            // the gathered matmuls on the sorted pairs, and put the rows back in (token, slot)
            // order.
            let flat = inds.flatten(None, None)?; // [t*k]
            let order = argsort_axis(&flat, 0)?;
            let inverse = argsort_axis(&order, 0)?;
            let rows = floor_divide(&order, Array::from_int(k).as_dtype(order.dtype())?)?;
            let xs = xf.reshape(&[t, 1, h])?.take_axis(&rows, 0)?; // [t*k, 1, h]
            let idx = flat.take_axis(&order, 0)?;
            let y = swiglu(&xs, &idx, true)?; // [t*k, 1, h]
            Ok(y.take_axis(&inverse, 0)?.reshape(&[t, k, h])?)
        } else {
            let x = xf.reshape(&[t, 1, 1, h])?;
            Ok(swiglu(&x, inds, false)?.reshape(&[t, k, h])?) // [t, k, 1, h] → [t, k, h]
        }
    }

    /// The block's output for `x` `[batch, seq, hidden]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s, h) = (sh[0], sh[1], sh[2]);
        let dtype = x.dtype();
        let xf = x.reshape(&[b * s, h])?;

        let (inds, weights) = self.route(&xf)?;
        let y = self.experts(&xf, &inds)?; // [t, k, h]
        let w = weights.as_dtype(dtype)?.expand_dims(-1)?; // [t, k, 1]
        let routed = sum_axis(&multiply(&y, &w)?, 1, false)?; // [t, h]

        let shared = self.shared.forward(&xf)?;
        let shared = match &self.shared_gate {
            Some(g) => multiply(&shared, &sigmoid(&linear(&xf, g, None)?)?)?,
            None => shared,
        };
        Ok(add(&routed, &shared)?.reshape(&[b, s, h])?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::sampler::{SplitMix64, TokenRng};

    fn randn(shape: &[i32], rng: &mut SplitMix64) -> Array {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
        Array::from_slice(&data, shape)
    }

    fn host(a: &Array) -> Vec<f32> {
        a.as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    }

    /// A small f32 block with `e` experts, per-expert dense weights kept for the reference.
    struct Fixture {
        moe: SparseMoe,
        router: Vec<f32>,
        experts: Vec<[Array; 3]>,
        shared: [Array; 3],
        shared_gate: Array,
        h: i32,
    }

    fn fixture(
        e: i32,
        h: i32,
        inter: i32,
        routing: MoeRouting,
        quant: Option<QuantSpec>,
    ) -> Fixture {
        let mut rng = SplitMix64::new(0x2444_0100);
        let router = randn(&[e, h], &mut rng);
        let experts: Vec<[Array; 3]> = (0..e)
            .map(|_| {
                [
                    randn(&[inter, h], &mut rng),
                    randn(&[inter, h], &mut rng),
                    randn(&[h, inter], &mut rng),
                ]
            })
            .collect();
        let shared = [
            randn(&[inter, h], &mut rng),
            randn(&[inter, h], &mut rng),
            randn(&[h, inter], &mut rng),
        ];
        let shared_gate = randn(&[1, h], &mut rng);
        let bank = |i: usize| {
            SwitchLinear::stack(
                experts
                    .iter()
                    .map(|w| Projection::load(w[i].clone(), quant).unwrap())
                    .collect(),
            )
            .unwrap()
        };
        let moe = SparseMoe::new(
            router.clone(),
            bank(0),
            bank(1),
            bank(2),
            SwiGlu {
                gate: Projection::load(shared[0].clone(), None).unwrap(),
                up: Projection::load(shared[1].clone(), None).unwrap(),
                down: Projection::load(shared[2].clone(), None).unwrap(),
            },
            Some(shared_gate.clone()),
            routing,
        )
        .unwrap();
        Fixture {
            router: host(&router),
            moe,
            experts,
            shared,
            shared_gate,
            h,
        }
    }

    /// The textbook block on the host, token by token: softmax → top-k → renormalize or scale →
    /// Σ wᵢ·SwiGLUᵢ(x) + σ(x·g)·SwiGLU_shared(x).
    fn reference(f: &Fixture, x: &[f32], routing: MoeRouting) -> Vec<f32> {
        let h = f.h as usize;
        let e = f.experts.len();
        let dense = |w: &Array| (host(w), w.shape()[0] as usize, w.shape()[1] as usize);
        let mv = |(w, out, inn): &(Vec<f32>, usize, usize), v: &[f32]| -> Vec<f32> {
            (0..*out)
                .map(|o| (0..*inn).map(|i| w[o * inn + i] * v[i]).sum())
                .collect()
        };
        let swiglu = |ws: &[Array; 3], v: &[f32]| -> Vec<f32> {
            let (g, u, d) = (dense(&ws[0]), dense(&ws[1]), dense(&ws[2]));
            let a: Vec<f32> = mv(&g, v)
                .iter()
                .zip(mv(&u, v))
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            mv(&d, &a)
        };
        let gate = host(&f.shared_gate);
        let mut out = Vec::new();
        for v in x.chunks(h) {
            let logits: Vec<f32> = (0..e)
                .map(|j| (0..h).map(|i| f.router[j * h + i] * v[i]).sum())
                .collect();
            let m = logits.iter().copied().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|p| p / sum).collect();
            let mut idx: Vec<usize> = (0..e).collect();
            idx.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
            let top = &idx[..routing.experts_per_tok];
            let denom: f32 = top.iter().map(|&j| probs[j]).sum();
            let mut acc = vec![0.0f32; h];
            for &j in top {
                let w = if routing.norm_topk_prob {
                    probs[j] / denom
                } else {
                    probs[j] * routing.routed_scaling_factor
                };
                for (a, y) in acc.iter_mut().zip(swiglu(&f.experts[j], v)) {
                    *a += w * y;
                }
            }
            let sg = 1.0 / (1.0 + (-(0..h).map(|i| gate[i] * v[i]).sum::<f32>()).exp());
            for (a, y) in acc.iter_mut().zip(swiglu(&f.shared, v)) {
                *a += sg * y;
            }
            out.extend(acc);
        }
        out
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// Both dispatch shapes (unsorted decode, sorted prefill) and both routing branches match the
    /// textbook block.
    #[test]
    fn sparse_moe_matches_the_reference_block() {
        for routing in [
            MoeRouting {
                experts_per_tok: 2,
                norm_topk_prob: true,
                routed_scaling_factor: 1.0,
            },
            MoeRouting {
                experts_per_tok: 3,
                norm_topk_prob: false,
                routed_scaling_factor: 2.5,
            },
        ] {
            let f = fixture(8, 32, 16, routing, None);
            // t = 1 (decode, unsorted) and t = 40 (t·k ≥ 64: the sorted path).
            for t in [1, 40] {
                let mut rng = SplitMix64::new(0x2444_0200 + t as u64);
                let x = randn(&[1, t, f.h], &mut rng);
                let got = host(&f.moe.forward(&x).unwrap());
                let want = reference(&f, &host(&x), routing);
                let d = max_diff(&got, &want);
                assert!(d < 1e-4, "t={t} {routing:?}: max abs diff {d}");
            }
        }
    }

    /// A quantized bank (gather_qmm) matches the same block with each expert quantized on its own
    /// (quantized_matmul) — the representation the per-expert router ran.
    #[test]
    fn quantized_bank_matches_per_expert_quantized_experts() {
        let routing = MoeRouting {
            experts_per_tok: 2,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
        };
        let q = QuantSpec::q8();
        let f = fixture(8, 64, 64, routing, Some(q));
        assert!(f.moe.is_quantized());
        for t in [1, 40] {
            let mut rng = SplitMix64::new(0x2444_0300 + t as u64);
            let x = randn(&[t, f.h], &mut rng);
            let (inds, weights) = f.moe.route(&x).unwrap();
            let (inds_h, w_h) = (
                inds.as_dtype(Dtype::Int32)
                    .unwrap()
                    .as_slice::<i32>()
                    .to_vec(),
                host(&weights),
            );
            let got = host(&f.moe.forward(&x.reshape(&[1, t, f.h]).unwrap()).unwrap());
            // Per-expert quantized experts, accumulated on the host with the router's own picks.
            let per: Vec<[Projection; 3]> = f
                .experts
                .iter()
                .map(|ws| ws.clone().map(|w| Projection::load(w, Some(q)).unwrap()))
                .collect();
            let shared = host(&f.moe.shared.forward(&x).unwrap());
            let sg = host(&sigmoid(linear(&x, &f.shared_gate, None).unwrap()).unwrap());
            let h = f.h as usize;
            let mut want = vec![0.0f32; t as usize * h];
            for ti in 0..t as usize {
                let row = x
                    .take_axis(Array::from_slice(&[ti as i32], &[1]), 0)
                    .unwrap();
                for s in 0..2 {
                    let e = inds_h[ti * 2 + s] as usize;
                    let p = &per[e];
                    let g = silu(&p[0].forward(&row).unwrap()).unwrap();
                    let u = p[1].forward(&row).unwrap();
                    let y = host(&p[2].forward(&multiply(&g, &u).unwrap()).unwrap());
                    for i in 0..h {
                        want[ti * h + i] += w_h[ti * 2 + s] * y[i];
                    }
                }
                for i in 0..h {
                    want[ti * h + i] += sg[ti] * shared[ti * h + i];
                }
            }
            let d = max_diff(&got, &want);
            assert!(d < 1e-4, "t={t}: max abs diff {d}");
        }
    }

    /// Mixed and Prism banks are refused at load, never dispatched on a wrong kernel.
    #[test]
    fn a_mixed_bank_is_refused() {
        let mut rng = SplitMix64::new(1);
        let w = randn(&[64, 64], &mut rng);
        let parts = vec![
            Projection::load(w.clone(), None).unwrap(),
            Projection::load(w, Some(QuantSpec::q8())).unwrap(),
        ];
        assert!(matches!(
            SwitchLinear::stack(parts),
            Err(Error::Unsupported(_))
        ));
    }
}
