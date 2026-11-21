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
//! A checkpoint that stores gate‖up as one fused bank (Qwen3.6's `experts.gate_up_proj`,
//! `[experts, 2·inter, in]`) keeps it fused: one gathered matmul produces both halves and the
//! per-token **activations** are split, instead of splitting the bank into two weight copies at
//! load (sc-24446) — which held a second copy of ~44 GB of Qwen3.6-35B-A3B expert weights.
//! Prefill-sized batches sort the pairs by expert first (the `sorted_indices` fast path) and
//! unsort afterwards, exactly as mlx-lm's `SwitchGLU` does.

use mlx_rs::ops::indexing::take_along_axis;
use mlx_rs::ops::{
    add, argpartition_axis, argsort_axis, divide, floor_divide, gather_mm, gather_qmm, maximum,
    multiply, negative, quantize, sigmoid, softmax_axis, split_sections, stack_axis, sum_axis,
};
use mlx_rs::{ops::indexing::IndexOp, Array, Dtype};

use crate::error::{Error, Result};
#[cfg(test)]
use crate::primitives::nn::linear;
use crate::primitives::nn::silu;
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
    /// Original per-expert source operators, indexed in the same order as the router.
    Source {
        parts: Vec<Projection>,
        shape: [i32; 2],
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

    /// Retain the slices of a publisher's [experts, output, input] FP8 bank.
    pub(crate) fn from_checkpoint(
        w: &super::weights::Weights,
        key: &str,
        shape: [i32; 3],
        dense: impl FnOnce() -> Result<Array>,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        use mlx_layers::convrot_int8::MlxWeightSource;
        use mlx_layers::ComfyQuantFormat;
        if let Some(blob) = w.get(&format!("{key}.comfy_quant")) {
            let desc = mlx_layers::convrot_int8::descriptor(blob, key)?;
            if !matches!(
                desc.format,
                ComfyQuantFormat::Float8E4M3Fn | ComfyQuantFormat::Float8E5M2
            ) || desc.full_precision_matrix_mult
            {
                return Err(Error::Config(format!(
                    "{key}: stacked source requires a scalar FP8 scale for the entire expert bank"
                )));
            }
            let codes = w.require(key)?;
            if codes.shape() != shape || shape.iter().any(|n| *n <= 0) {
                return Err(Error::Config(format!(
                    "{key}: invalid expert bank geometry"
                )));
            }
            let scales = w.get(&format!("{key}.weight_scale")).cloned();
            let input = w.get(&format!("{key}.input_scale")).cloned();
            let observation = w.observation(key);
            let [e, n, k] = shape;
            let all_rows = e.checked_mul(n).ok_or_else(|| {
                Error::Config(format!("{key}: expert bank exceeds MLX row range"))
            })?;
            let matrix = mlx_layers::fp8::Fp8Linear::new(
                desc.format,
                codes.reshape(&[all_rows, k])?,
                scales,
                input,
                &[all_rows as usize, k as usize],
                None,
            )?;
            let mut parts = Vec::with_capacity(e as usize);
            for i in 0..e {
                let mut op = matrix.slice_rows(i * n, (i + 1) * n)?;
                if let Some(obs) = &observation {
                    op.observe(obs.expert(i as usize, n as usize, k as usize)?)?;
                }
                parts.push(Projection::Native(mlx_layers::NativeProjection::Fp8(op)));
            }
            return Ok(Self::Source {
                parts,
                shape: [n, k],
            });
        }
        let weight = dense()?;
        if weight.shape() != shape {
            return Err(Error::Config(format!(
                "{key}: expected {shape:?}, got {:?}",
                weight.shape()
            )));
        }
        Self::load(weight, quant)
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
        if parts.iter().any(|p| matches!(p, Projection::Native(_))) {
            let shape = first.expert_shape()?;
            for p in &parts {
                if p.expert_shape()? != shape {
                    return Err(Error::Config(
                        "routed experts disagree on input/output geometry".into(),
                    ));
                }
            }
            return Ok(Self::Source { parts, shape });
        }
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
            Projection::Native(_) => unreachable!("source bank handled above"),
            Projection::Prism(_) => Err(Error::Unsupported(
                "Prism packed MoE experts have no gathered matmul".into(),
            )),
        }
    }

    /// `x @ weight[idx]ᵀ` for every index: `x` `[.., 1, in]` broadcast against `idx`'s shape,
    /// giving `[.., 1, out]` per index.
    fn forward(
        &self,
        x: &Array,
        idx: &Array,
        sorted: bool,
        offsets: Option<&[i32]>,
    ) -> Result<Array> {
        Ok(match self {
            Self::Source { parts, .. } => {
                let offsets = offsets.ok_or_else(|| {
                    Error::Config("source expert bank requires sorted router offsets".into())
                })?;
                let mut outputs = Vec::new();
                for (p, span) in parts.iter().zip(offsets.windows(2)) {
                    if span[0] < span[1] {
                        outputs.push(p.forward(&x.index(span[0]..span[1]))?);
                    }
                }
                mlx_rs::ops::concatenate_axis(&outputs.iter().collect::<Vec<_>>(), 0)?
            }
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

    /// The experts stacked on the leading axis.
    pub fn num_experts(&self) -> usize {
        match self {
            Self::Source { parts, .. } => parts.len(),
            Self::Dense { weight } | Self::Quantized { weight, .. } => {
                weight.shape().first().copied().unwrap_or(0) as usize
            }
        }
    }

    /// Whether the bank is quantized.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. } | Self::Source { .. })
    }

    /// Output rows per expert (`out` of `[experts, out, in]`).
    fn out_features(&self) -> i32 {
        match self {
            Self::Source { shape, .. } => shape[0],
            Self::Dense { weight } | Self::Quantized { weight, .. } => {
                weight.shape().get(1).copied().unwrap_or(0)
            }
        }
    }

    /// The arrays the bank holds, for load-time materialization (sc-24446).
    pub(crate) fn push_arrays(&self, out: &mut Vec<Array>) {
        match self {
            Self::Source { parts, .. } => {
                for p in parts {
                    p.push_arrays(out);
                }
            }
            Self::Dense { weight } => out.push(weight.clone()),
            Self::Quantized {
                weight,
                scales,
                biases,
                ..
            } => out.extend([weight.clone(), scales.clone(), biases.clone()]),
        }
    }
}

/// The routed experts' gate and up projections.
#[derive(Debug)]
pub enum GateUp {
    /// One bank over gate‖up rows (`[experts, 2·inter, in]`, gate rows first) — the layout
    /// Qwen3.6 ships as `experts.gate_up_proj`. One gathered matmul yields `[.., 2·inter]`, split
    /// into gate and up per token.
    Fused(SwitchLinear),
    /// Separate gate and up banks (per-expert checkpoints, stacked at load).
    Split {
        /// The gate bank `[experts, inter, in]`.
        gate: SwitchLinear,
        /// The up bank `[experts, inter, in]`.
        up: SwitchLinear,
    },
}

impl GateUp {
    /// `(gate, up)` activations for `x` routed through `idx`.
    fn forward(
        &self,
        x: &Array,
        idx: &Array,
        sorted: bool,
        offsets: Option<&[i32]>,
    ) -> Result<(Array, Array)> {
        match self {
            Self::Fused(bank) => {
                let gu = bank.forward(x, idx, sorted, offsets)?;
                let mut halves = split_sections(&gu, &[bank.out_features() / 2], -1)?;
                let up = halves.pop().expect("two halves");
                let gate = halves.pop().expect("two halves");
                Ok((gate, up))
            }
            Self::Split { gate, up } => Ok((
                gate.forward(x, idx, sorted, offsets)?,
                up.forward(x, idx, sorted, offsets)?,
            )),
        }
    }

    fn banks(&self) -> Vec<(&'static str, &SwitchLinear)> {
        match self {
            Self::Fused(bank) => vec![("gate_up", bank)],
            Self::Split { gate, up } => vec![("gate", gate), ("up", up)],
        }
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

    /// The arrays the three projections hold, for load-time materialization (sc-24446).
    pub(crate) fn push_arrays(&self, out: &mut Vec<Array>) {
        for p in [&self.gate, &self.up, &self.down] {
            p.push_arrays(out);
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

/// A sparse Mixture-of-Experts FFN: a softmax router over the stacked routed experts (top-k per
/// token) plus an always-on shared expert, optionally sigmoid-gated (Qwen2-MoE, Qwen3.6-MoE) or
/// added ungated (DeepSeek-V2). `n_group` / `topk_group` group-limited routing (DeepSeek-V2-236B /
/// V3) is not modelled — V2-Lite uses plain greedy top-k.
#[derive(Debug)]
pub struct SparseMoe {
    /// Router weight `[experts, hidden]`.
    router: Projection,
    gate_up: GateUp,
    down: SwitchLinear,
    num_experts: usize,
    shared: SwiGlu,
    /// Shared-expert sigmoid gate `[1, hidden]`; `None` ⇒ the shared expert is added ungated.
    shared_gate: Option<Projection>,
    routing: MoeRouting,
}

impl SparseMoe {
    /// Assemble the block. `router` is `[experts, hidden]`; `gate` / `up` / `down` are the stacked
    /// routed-expert projections (`[experts, inter, hidden]` / `[experts, hidden, inter]`). A bank
    /// whose expert count disagrees with the router's rows is refused: the gathered matmuls would
    /// index past it without complaint.
    pub fn new(
        router: Array,
        gate: SwitchLinear,
        up: SwitchLinear,
        down: SwitchLinear,
        shared: SwiGlu,
        shared_gate: Option<Array>,
        routing: MoeRouting,
    ) -> Result<Self> {
        Self::with_gate_up(
            Projection::load(router, None)?,
            GateUp::Split { gate, up },
            down,
            shared,
            shared_gate.map(|w| Projection::load(w, None)).transpose()?,
            routing,
        )
    }

    /// Assemble the block over a **fused** gate‖up bank `[experts, 2·inter, hidden]` (gate rows
    /// first — Qwen3.6's `experts.gate_up_proj`), dispatched as one gathered matmul whose output
    /// is split per token. A bank with an odd row count has no gate/up split and is refused.
    pub fn fused(
        router: Array,
        gate_up: SwitchLinear,
        down: SwitchLinear,
        shared: SwiGlu,
        shared_gate: Option<Array>,
        routing: MoeRouting,
    ) -> Result<Self> {
        if gate_up.out_features() % 2 != 0 || gate_up.out_features() == 0 {
            return Err(Error::Config(format!(
                "a fused MoE gate_up bank needs an even, non-zero row count (gate ‖ up), got {}",
                gate_up.out_features()
            )));
        }
        Self::with_gate_up(
            Projection::load(router, None)?,
            GateUp::Fused(gate_up),
            down,
            shared,
            shared_gate.map(|w| Projection::load(w, None)).transpose()?,
            routing,
        )
    }

    pub(crate) fn with_gate_up(
        router: Projection,
        gate_up: GateUp,
        down: SwitchLinear,
        shared: SwiGlu,
        shared_gate: Option<Projection>,
        routing: MoeRouting,
    ) -> Result<Self> {
        let num_experts = router.expert_shape()?[0] as usize;
        if matches!(&gate_up, GateUp::Fused(bank) if bank.out_features()==0 || bank.out_features()%2!=0)
        {
            return Err(Error::Config(
                "fused gate/up requires an even nonzero output width".into(),
            ));
        }
        if num_experts == 0 {
            return Err(Error::Config(
                "an MoE router needs at least one expert".into(),
            ));
        }
        let mut banks = gate_up.banks();
        banks.push(("down", &down));
        for (name, bank) in banks {
            if bank.num_experts() != num_experts {
                return Err(Error::Config(format!(
                    "an MoE router over {num_experts} experts needs as many {name} experts, got {}",
                    bank.num_experts()
                )));
            }
        }
        Ok(Self {
            router,
            gate_up,
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
        self.down.is_quantized()
    }

    /// Whether gate and up are dispatched as one fused bank.
    pub fn is_fused(&self) -> bool {
        matches!(self.gate_up, GateUp::Fused(_))
    }

    /// The always-on shared expert's projections.
    pub(crate) fn for_each_shared_projection(&self, f: &mut dyn FnMut(&Projection)) {
        for p in [&self.shared.gate, &self.shared.up, &self.shared.down] {
            f(p);
        }
    }

    /// Every array the block holds, for load-time materialization (sc-24446).
    pub(crate) fn push_arrays(&self, out: &mut Vec<Array>) {
        self.router.push_arrays(out);
        for (_, bank) in self.gate_up.banks() {
            bank.push_arrays(out);
        }
        self.down.push_arrays(out);
        self.shared.push_arrays(out);
        if let Some(g) = &self.shared_gate {
            g.push_arrays(out);
        }
    }

    /// Route every token on the device: the top-k expert indices `[t, k]` (uint32) and their
    /// weights `[t, k]` (f32), from `xf` `[t, hidden]`.
    pub fn route(&self, xf: &Array) -> Result<(Array, Array)> {
        let k = self.routing.experts_per_tok.clamp(1, self.num_experts) as i32;
        let logits = self.router.forward(xf)?; // [t, E]
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
        let has_source = self
            .gate_up
            .banks()
            .iter()
            .any(|(_, bank)| matches!(bank, SwitchLinear::Source { .. }))
            || matches!(self.down, SwitchLinear::Source { .. });
        let swiglu =
            |x: &Array, idx: &Array, sorted: bool, offsets: Option<&[i32]>| -> Result<Array> {
                let (g, up) = self.gate_up.forward(x, idx, sorted, offsets)?;
                self.down
                    .forward(&multiply(&silu(&g)?, &up)?, idx, sorted, offsets)
            };
        if has_source || t * k >= SORT_THRESHOLD {
            // Sort the (token, slot) pairs by expert so each expert's rows are contiguous, run
            // the gathered matmuls on the sorted pairs, and put the rows back in (token, slot)
            // order.
            let flat = inds.flatten(None, None)?; // [t*k]
            let order = argsort_axis(&flat, 0)?;
            let inverse = argsort_axis(&order, 0)?;
            let rows = floor_divide(&order, Array::from_int(k).as_dtype(order.dtype())?)?;
            let xs = xf.reshape(&[t, 1, h])?.take_axis(&rows, 0)?; // [t*k, 1, h]
            let idx = flat.take_axis(&order, 0)?;
            let boundaries = has_source
                .then(|| routed_offsets(&idx, self.num_experts))
                .transpose()?;
            let y = swiglu(&xs, &idx, true, boundaries.as_deref())?; // [t*k, 1, h]
            Ok(y.take_axis(&inverse, 0)?.reshape(&[t, k, h])?)
        } else {
            let x = xf.reshape(&[t, 1, 1, h])?;
            Ok(swiglu(&x, inds, false, None)?.reshape(&[t, k, h])?) // [t, k, 1, h] → [t, k, h]
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
            Some(g) => multiply(&shared, &sigmoid(&g.forward(&xf)?)?)?,
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

    const NORM: MoeRouting = MoeRouting {
        experts_per_tok: 2,
        norm_topk_prob: true,
        routed_scaling_factor: 1.0,
    };

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// Both dispatch shapes (unsorted decode, sorted prefill) and both routing branches match the
    /// textbook block.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
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
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::moe::tests::sparse_moe_matches_the_reference_block");
    }

    /// A quantized bank (gather_qmm) matches the same block with each expert quantized on its own
    /// (quantized_matmul) — the representation the per-expert router ran.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
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
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::moe::tests::quantized_bank_matches_per_expert_quantized_experts");
    }

    /// sc-24446: a fused gate‖up bank (Qwen3.6's `experts.gate_up_proj`) dispatched as ONE
    /// gathered matmul whose output is split per token is bit-identical to the gate and up banks
    /// the loader used to cut out of it — dense f32 and bf16, Q8 and Q4, on the unsorted decode
    /// shape (t = 1) and the sorted prefill shape (t·k ≥ 64).
    ///
    /// MUTATION: swap the halves in `GateUp::forward` (gate ↔ up), and every case goes RED.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn a_fused_gate_up_bank_matches_the_split_banks_exactly() {
        let (e, h, inter) = (8, 64, 64);
        let routing = MoeRouting {
            experts_per_tok: 2,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
        };
        for dtype in [Dtype::Float32, Dtype::Bfloat16] {
            for quant in [None, Some(QuantSpec::q8()), Some(QuantSpec::q4())] {
                let mut rng = SplitMix64::new(0x2444_6100);
                let cast = |a: Array| a.as_dtype(dtype).unwrap();
                let router = cast(randn(&[e, h], &mut rng));
                let gate_up = cast(randn(&[e, 2 * inter, h], &mut rng));
                let down = cast(randn(&[e, h, inter], &mut rng));
                let shared: [Array; 3] = [
                    cast(randn(&[inter, h], &mut rng)),
                    cast(randn(&[inter, h], &mut rng)),
                    cast(randn(&[h, inter], &mut rng)),
                ];
                let shared_gate = cast(randn(&[1, h], &mut rng));
                let swiglu = || SwiGlu {
                    gate: Projection::load(shared[0].clone(), None).unwrap(),
                    up: Projection::load(shared[1].clone(), None).unwrap(),
                    down: Projection::load(shared[2].clone(), None).unwrap(),
                };
                let fused = SparseMoe::fused(
                    router.clone(),
                    SwitchLinear::load(gate_up.clone(), quant).unwrap(),
                    SwitchLinear::load(down.clone(), quant).unwrap(),
                    swiglu(),
                    Some(shared_gate.clone()),
                    routing,
                )
                .unwrap();
                let halves = split_sections(&gate_up, &[inter], 1).unwrap();
                let split = SparseMoe::new(
                    router,
                    SwitchLinear::load(halves[0].clone(), quant).unwrap(),
                    SwitchLinear::load(halves[1].clone(), quant).unwrap(),
                    SwitchLinear::load(down, quant).unwrap(),
                    swiglu(),
                    Some(shared_gate),
                    routing,
                )
                .unwrap();
                assert!(fused.is_fused() && !split.is_fused());
                for t in [1, 40] {
                    let mut rng = SplitMix64::new(0x2444_6200 + t as u64);
                    let x = cast(randn(&[1, t, h], &mut rng));
                    let (a, b) = (
                        host(&fused.forward(&x).unwrap()),
                        host(&split.forward(&x).unwrap()),
                    );
                    assert!(a.iter().any(|v| v.abs() > 1e-3), "non-degenerate output");
                    assert_eq!(a, b, "{dtype:?} {quant:?} t={t}: fused != split");
                }
            }
        }
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::moe::tests::a_fused_gate_up_bank_matches_the_split_banks_exactly");
    }

    /// A fused bank with no even gate/up split is refused at load.
    #[test]
    fn a_fused_bank_with_an_odd_row_count_is_refused() {
        let mut rng = SplitMix64::new(7);
        let (e, h) = (4, 64);
        let shared = || SwiGlu {
            gate: Projection::load(randn(&[64, h], &mut SplitMix64::new(3)), None).unwrap(),
            up: Projection::load(randn(&[64, h], &mut SplitMix64::new(4)), None).unwrap(),
            down: Projection::load(randn(&[h, 64], &mut SplitMix64::new(5)), None).unwrap(),
        };
        let odd = SparseMoe::fused(
            randn(&[e, h], &mut rng),
            SwitchLinear::load(randn(&[e, 63, h], &mut rng), None).unwrap(),
            SwitchLinear::load(randn(&[e, h, 31], &mut rng), None).unwrap(),
            shared(),
            None,
            NORM,
        );
        assert!(matches!(odd, Err(Error::Config(_))));
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

    /// A router whose row count disagrees with the expert banks is refused at load (dense and
    /// quantized banks alike), never handed to `gather_mm` / `gather_qmm` as an out-of-range index.
    #[test]
    fn a_router_that_disagrees_with_the_bank_is_refused() {
        let mut rng = SplitMix64::new(2);
        let (e, h, inter) = (4, 64, 64);
        for quant in [None, Some(QuantSpec::q8())] {
            let bank = |rng: &mut SplitMix64, out: i32, inn: i32| {
                SwitchLinear::load(randn(&[e, out, inn], rng), quant).unwrap()
            };
            let shared = || SwiGlu {
                gate: Projection::load(randn(&[inter, h], &mut SplitMix64::new(3)), None).unwrap(),
                up: Projection::load(randn(&[inter, h], &mut SplitMix64::new(4)), None).unwrap(),
                down: Projection::load(randn(&[h, inter], &mut SplitMix64::new(5)), None).unwrap(),
            };
            let (gate, up, down) = (
                bank(&mut rng, inter, h),
                bank(&mut rng, inter, h),
                bank(&mut rng, h, inter),
            );
            let short = SparseMoe::new(
                randn(&[e - 1, h], &mut rng),
                gate,
                up,
                down,
                shared(),
                None,
                NORM,
            );
            assert!(
                matches!(short, Err(Error::Config(_))),
                "{quant:?}: a router one row short is accepted"
            );
            let whole = SparseMoe::new(
                randn(&[e, h], &mut rng),
                bank(&mut rng, inter, h),
                bank(&mut rng, inter, h),
                bank(&mut rng, h, inter),
                shared(),
                None,
                NORM,
            );
            assert!(whole.is_ok(), "{quant:?}: a matching router is refused");
        }
    }
}

// Read only E+1 boundaries for the three operators, regardless of prompt length. Token data,
// sorted IDs, and expert operands stay on device. Each active expert receives a contiguous batch.
fn routed_offsets(ids: &Array, experts: usize) -> Result<Vec<i32>> {
    use mlx_rs::fast::MetalKernel;
    thread_local! {static KERNEL:std::cell::RefCell<Option<MetalKernel>>=const{std::cell::RefCell::new(None)};}
    const SOURCE: &str = r#"
    uint e=thread_position_in_grid.x;
    if(e<=uint(E)){
        uint lo=0,hi=uint(P);
        while(lo<hi){uint mid=lo+(hi-lo)/2u;if(uint(ids[mid])<e){lo=mid+1u;}else{hi=mid;}}
        offsets[e]=int(lo);
    }"#;
    let count = i32::try_from(experts + 1)
        .map_err(|_| Error::Config("expert count exceeds Metal indexing".into()))?;
    let pairs = i32::try_from(ids.size())
        .map_err(|_| Error::Config("token routes exceed Metal indexing".into()))?;
    let offsets = KERNEL.with(|cell| -> Result<Array> {
        let mut cache = cell.borrow_mut();
        if cache.is_none() {
            *cache = Some(MetalKernel::with_options(
                "sc_routed_offsets_v1",
                &["ids"],
                &["offsets"],
                SOURCE,
                "",
                true,
                false,
            )?);
        }
        Ok(cache
            .as_ref()
            .unwrap()
            .apply()
            .inputs([ids])
            .output_shape([count], Dtype::Int32)
            .template_arg("E", count - 1)
            .template_arg("P", pairs)
            .grid(count, 1, 1)
            .thread_group(32, 1, 1)
            .run()?
            .remove(0))
    })?;
    offsets.eval()?;
    let offsets = offsets.as_slice::<i32>().to_vec();
    if offsets.first() != Some(&0) || offsets.last() != Some(&pairs) {
        return Err(Error::Config("routed expert ID outside bank".into()));
    }
    Ok(offsets)
}
