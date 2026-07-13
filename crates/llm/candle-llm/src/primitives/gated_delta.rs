//! Gated DeltaNet linear attention — the recurrence (story sc-7632, the candle mirror of mlx-llm
//! sc-7627).
//!
//! Qwen3.6 (`model_type` `qwen3_5`, the Qwen3-Next architecture) interleaves 3 **Gated DeltaNet**
//! linear-attention layers with 1 gated full-attention layer. Unlike softmax attention over a
//! growing KV cache, a linear layer carries a **fixed-size recurrent state** `S ∈ [Dv, Dk]` per head
//! and updates it with the gated delta rule each step — so it costs O(1) memory in sequence length.
//!
//! This module ports the **ops path** of `mlx_lm.models.gated_delta` (`gated_delta_ops` /
//! `_gated_delta_step_ops`) — the sequential reference the MLX engine itself falls back to off-GPU —
//! into Candle, byte-for-byte with the verified `mlx-llm` port. The recurrence is validated against
//! the same numeric fixture (see the tests). The per-step update, for head state `S` (decayed by the
//! gate `g`, `β` the delta strength):
//!
//! ```text
//!   S      = S · g                          # forget (per-head scalar decay)
//!   kv_mem = (S · kᵀ) summed over Dk         # what the current key already recalls  → [Dv]
//!   Δ      = (v − kv_mem) · β                # the correction to write               → [Dv]
//!   S      = S + Δ ⊗ k                        # delta-rule outer-product write         → [Dv, Dk]
//!   y      = (S · qᵀ) summed over Dk          # read out with the query                → [Dv]
//! ```
//!
//! The gate and delta strength come from the layer's learned projections via [`compute_g`] (`g =
//! exp(−exp(A_log) · softplus(a + dt_bias))`) and `β = sigmoid(b)`; the surrounding short-conv,
//! normalisation, and in/out projections (the full layer) build on this in the decoder story. GQA is
//! handled by repeating each of the `Hk` key/query heads to the `Hv` value heads.
//!
//! The math runs in the inputs' dtype, matching the ops reference; the decoder lifts the recurrence
//! to f32 (casting the projected q/k/v/g/β and keeping the SSM state in f32) to match the GPU kernel.

use candle_core::{DType, Tensor};

use crate::error::Result;
use crate::primitives::nn::{rms_norm, silu};

/// Numerically-stable softplus `ln(1 + eˣ) = relu(x) + ln(1 + e^−|x|)`, matching the reference's
/// `softplus`. Evaluated in `x`'s dtype.
fn softplus(x: &Tensor) -> Result<Tensor> {
    let relu = x.relu()?;
    let log1p = x.abs()?.affine(-1.0, 0.0)?.exp()?.affine(1.0, 1.0)?.log()?; // ln(1 + e^−|x|)
    Ok(relu.broadcast_add(&log1p)?)
}

/// The per-step gate `g = exp(−exp(A_log) · softplus(a + dt_bias))` (a faithful port of
/// `mlx_lm.models.gated_delta.compute_g`). `a` is `[B, T, Hv]` (the gating projection), `A_log` and
/// `dt_bias` are per-value-head `[Hv]`. The inner exponentials are evaluated in f32 (matching the
/// reference's `.astype(float32)`) and the result is cast back to `a`'s dtype.
pub fn compute_g(a: &Tensor, a_log: &Tensor, dt_bias: &Tensor) -> Result<Tensor> {
    let orig = a.dtype();
    let a32 = a.to_dtype(DType::F32)?;
    let dt32 = dt_bias.to_dtype(DType::F32)?;
    let al32 = a_log.to_dtype(DType::F32)?;
    let sp = softplus(&a32.broadcast_add(&dt32)?)?; // softplus(a + dt_bias)        [B,T,Hv]
    let coeff = al32.exp()?.broadcast_mul(&sp)?; // exp(A_log) · softplus(...)       [B,T,Hv]
    let g = coeff.affine(-1.0, 0.0)?.exp()?; // exp(−coeff)
    Ok(g.to_dtype(orig)?)
}

/// Run the gated delta recurrence over a `[B, T, ·]` chunk, a faithful port of the
/// `mlx_lm.models.gated_delta` ops path.
///
/// Shapes: `q`, `k` are `[B, T, Hk, Dk]`; `v` is `[B, T, Hv, Dv]`; `g` (the per-step gate from
/// [`compute_g`]) and `beta` are `[B, T, Hv]`; `state` (the carried recurrent state, or `None` to
/// start from zeros) is `[B, Hv, Dv, Dk]`. Returns the per-step output `y` `[B, T, Hv, Dv]` and the
/// final `state` `[B, Hv, Dv, Dk]` — feed `state` back in for the next chunk / decode step (T = 1).
///
/// GQA: when `Hv > Hk` each key/query head is repeated `Hv / Hk` times so it pairs with the value
/// heads (`Hv` must be a multiple of `Hk`).
pub fn gated_delta_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (b, t, hk, dk) = q.dims4()?;
    let (_, _, hv, dv) = v.dims4()?;

    // GQA: repeat each of the Hk key/query heads to the Hv value heads (contiguous).
    let (q, k) = if hv != hk {
        let r = hv / hk;
        (repeat_heads(q, r)?, repeat_heads(k, r)?)
    } else {
        (q.clone(), k.clone())
    };

    let mut state = match state {
        Some(s) => s.clone(),
        None => Tensor::zeros((b, hv, dv, dk), q.dtype(), q.device())?,
    };

    let mut ys: Vec<Tensor> = Vec::with_capacity(t);
    for ti in 0..t {
        let qt = q.narrow(1, ti, 1)?.squeeze(1)?.contiguous()?; // [B,Hv,Dk]
        let kt = k.narrow(1, ti, 1)?.squeeze(1)?.contiguous()?; // [B,Hv,Dk]
        let vt = v.narrow(1, ti, 1)?.squeeze(1)?.contiguous()?; // [B,Hv,Dv]
        let gt = g.narrow(1, ti, 1)?.squeeze(1)?.contiguous()?; // [B,Hv]
        let bt = beta.narrow(1, ti, 1)?.squeeze(1)?.contiguous()?; // [B,Hv]
        let (y, next) = delta_step(&qt, &kt, &vt, &gt, &bt, &state, b, hv, dk, dv)?;
        state = next;
        ys.push(y.unsqueeze(1)?); // [B,1,Hv,Dv]
    }
    let refs: Vec<&Tensor> = ys.iter().collect();
    let y = Tensor::cat(&refs, 1)?; // [B,T,Hv,Dv]
    Ok((y, state))
}

/// Causal depthwise short convolution over `[B, S, C]` with per-channel kernel `weight` `[C, K]`
/// (the HF/MLX depthwise `Conv1d`, no bias), left-seeded by `conv_state` `[B, K-1, C]` (the previous
/// step's tail). Returns `(silu(conv) [B,S,C], new_conv_state [B,K-1,C])` — a port of the Qwen3-Next
/// short-conv path: `out[b,s,c] = silu(Σ_j weight[c,j] · concat(conv_state, x)[b, s+j, c])`. Mixing
/// q/k/v through this 1-D conv before the recurrence is what gives Gated DeltaNet its local context.
pub fn causal_depthwise_conv(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (_b, s, c) = x.dims3()?;
    let kk = weight.dim(1)?; // kernel size K (weight is [C, K])
    let cat = Tensor::cat(&[conv_state, x], 1)?; // [B, S+K-1, C]
    let mut acc: Option<Tensor> = None;
    for j in 0..kk {
        let window = cat.narrow(1, j, s)?; // cat[:, j:j+S, :] → [B,S,C]
        let wj = weight.narrow(1, j, 1)?.contiguous()?.reshape((1, 1, c))?; // weight[:,j] → [1,1,C]
        let term = window.broadcast_mul(&wj)?;
        acc = Some(match acc {
            None => term,
            Some(a) => a.broadcast_add(&term)?,
        });
    }
    let out = silu(&acc.expect("conv kernel size must be >= 1"))?; // [B,S,C]
    let new_state = cat.narrow(1, s, kk - 1)?.contiguous()?; // last K-1 of conv_in → [B,K-1,C]
    Ok((out, new_state))
}

/// Gated RMSNorm (`Qwen3NextRMSNormGated`): `rms_norm(x, weight, eps) · silu(gate)`. Applied to the
/// delta-net output before the out-projection (`x`, `gate`, and `weight` share the head-value dim).
pub fn rms_norm_gated(x: &Tensor, weight: &Tensor, gate: &Tensor, eps: f64) -> Result<Tensor> {
    let normed = rms_norm(x, weight, eps)?;
    Ok(normed.broadcast_mul(&silu(gate)?)?)
}

/// The recurrent state of one Gated DeltaNet layer — the linear-attention analog of a KV-cache slot
/// (the Mamba/SSM cache). It holds the short-conv tail `conv_state` `[B, K-1, conv_dim]` and the
/// delta-rule `ssm_state` `[B, Hv, Dv, Dk]`, both **fixed size** in sequence length (unlike the
/// growing KV cache). A hybrid decoder keeps one of these per linear layer alongside a
/// [`KvCache`](super::KvCache) per full-attention layer (the decoder assembles the mixed list).
#[derive(Clone, Debug, Default)]
pub struct DeltaNetCache {
    /// The short-conv history (previous `K-1` tokens), or `None` before the first step.
    pub conv_state: Option<Tensor>,
    /// The delta-rule recurrent state, or `None` before the first step.
    pub ssm_state: Option<Tensor>,
    offset: i32,
}

impl DeltaNetCache {
    /// An empty cache (no conv history, zero recurrent state).
    pub fn new() -> Self {
        Self::default()
    }

    /// Positions consumed so far (the linear-layer analog of [`KvCache::offset`](super::KvCache::offset)).
    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// Store the post-step `(conv_state, ssm_state)` and advance the position by `step` tokens.
    pub fn update(&mut self, conv_state: Tensor, ssm_state: Tensor, step: i32) {
        self.conv_state = Some(conv_state);
        self.ssm_state = Some(ssm_state);
        self.offset += step;
    }

    /// Drop all state, returning the cache to its freshly-constructed condition.
    pub fn reset(&mut self) {
        self.conv_state = None;
        self.ssm_state = None;
        self.offset = 0;
    }
}

/// One recurrent step (`_gated_delta_step_ops`). `q`,`k` `[B,Hv,Dk]`; `v` `[B,Hv,Dv]`; `g`,`beta`
/// `[B,Hv]`; `state` `[B,Hv,Dv,Dk]`. Returns `(y [B,Hv,Dv], new_state [B,Hv,Dv,Dk])`.
#[allow(clippy::too_many_arguments)]
fn delta_step(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    b: usize,
    hv: usize,
    dk: usize,
    dv: usize,
) -> Result<(Tensor, Tensor)> {
    let decay = g.reshape((b, hv, 1, 1))?; // [B,Hv,1,1]
    let state = state.broadcast_mul(&decay)?; // S · g
    let k_r = k.reshape((b, hv, 1, dk))?; // [B,Hv,1,Dk]
    let kv_mem = state.broadcast_mul(&k_r)?.sum(3)?; // (S·k).sum(Dk) → [B,Hv,Dv]
    let delta = v
        .broadcast_sub(&kv_mem)?
        .broadcast_mul(&beta.reshape((b, hv, 1))?)?; // (v−kv)·β → [B,Hv,Dv]
    let state = state.broadcast_add(&k_r.broadcast_mul(&delta.reshape((b, hv, dv, 1))?)?)?; // S + Δ⊗k
    let q_r = q.reshape((b, hv, 1, dk))?;
    let y = state.broadcast_mul(&q_r)?.sum(3)?; // (S·q).sum(Dk) → [B,Hv,Dv]
    Ok((y, state))
}

/// Repeat each head of `x` `[B,T,H,D]` `r` times along the head axis (contiguous), giving
/// `[B,T,H·r,D]` — the GQA expansion (`mx.repeat(x, r, axis=-2)`).
fn repeat_heads(x: &Tensor, r: usize) -> Result<Tensor> {
    let (b, t, h, d) = x.dims4()?;
    Ok(x.unsqueeze(3)? // [B,T,H,1,D]
        .broadcast_as((b, t, h, r, d))? // [B,T,H,r,D]
        .contiguous()?
        .reshape((b, t, h * r, d))?) // [B,T,H·r,D]
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // Numeric oracle generated from `mlx_lm.models.gated_delta.gated_delta_update(..., use_kernel=
    // False)` (the ops reference) on seeded inputs — B=1, T=3, Hk=2, Hv=4 (GQA ×2), Dk=2, Dv=2.
    // Identical fixture to the verified mlx-llm port (sc-7627); a framework-independent check.
    const Q: &[f32] = &[
        -0.8805123, -0.2520431, 0.6974789, -0.8069373, 0.5367976, -0.5163696, 0.4404979, -0.721925,
        0.7900037, -0.5997525, 0.0088437, -0.2685511,
    ];
    const K: &[f32] = &[
        -0.5530653, -0.0336156, 0.34984, -0.9507952, 0.3385786, 0.7130073, 0.4130355, 0.7005113,
        -0.9424242, -0.8679754, 0.0911343, -0.1635016,
    ];
    const V: &[f32] = &[
        0.9033704, 0.9242766, -0.5594231, 0.5815381, -0.9665546, 0.8363392, 0.8484675, -0.1261052,
        -0.8719618, 0.0562458, 0.8790255, 0.7971791, -0.8360307, -0.2071904, -0.6701862,
        -0.7691332, -0.2697922, -0.7519733, 0.0098643, -0.2587476, 0.5248392, 0.9719371,
        -0.0304079, -0.2898675,
    ];
    const A: &[f32] = &[
        -0.6942793, -1.9834223, 1.6954608, 1.8882596, 1.5180013, 1.9647338, -0.7497661, -1.9337821,
        -1.3342639, 1.7648504, -0.3198055, -1.4922521,
    ];
    const B: &[f32] = &[
        -1.6042936, 1.7927692, 0.1977825, 0.2890182, 0.152185, -0.4371433, 0.8649859, 0.2619474,
        -1.2190499, -1.3681375, -1.4745429, 1.3650055,
    ];
    const A_LOG: &[f32] = &[2.0919125, 1.5201275, 2.7469416, 0.104467];
    const DT_BIAS: &[f32] = &[-0.2865368, -0.2390987, 0.5489618, 0.1692053];
    const EXP_Y: &[f32] = &[
        0.0749167, 0.0766504, -0.2376069, 0.2469999, -0.5368805, 0.4645513, 0.490568, -0.0729116,
        0.0874512, -0.0056413, -0.0642828, -0.0583461, 0.1904479, 0.0472361, 0.4258981, 0.0956538,
        0.0364119, 0.0369535, -0.0004748, 0.0117344, 0.0043713, 0.0080946, 0.1139456, 0.041226,
    ];
    const EXP_STATE: &[f32] = &[
        0.0431024, -0.0039364, 0.1626127, 0.1525815, -0.0018656, -0.0016657, 0.0495011, 0.0456383,
        0.0089079, -0.015984, 0.0164975, -0.0295984, 0.0197504, -0.4236471, -0.1824342, -0.1595202,
    ];

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn host(x: &Tensor) -> Vec<f32> {
        x.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    #[test]
    fn recurrence_matches_python_ops_reference() {
        let q = Tensor::from_slice(Q, (1, 3, 2, 2), &Device::Cpu).unwrap();
        let k = Tensor::from_slice(K, (1, 3, 2, 2), &Device::Cpu).unwrap();
        let v = Tensor::from_slice(V, (1, 3, 4, 2), &Device::Cpu).unwrap();
        let a = Tensor::from_slice(A, (1, 3, 4), &Device::Cpu).unwrap();
        let b_raw = Tensor::from_slice(B, (1, 3, 4), &Device::Cpu).unwrap();
        let a_log = Tensor::from_slice(A_LOG, (4,), &Device::Cpu).unwrap();
        let dt_bias = Tensor::from_slice(DT_BIAS, (4,), &Device::Cpu).unwrap();

        // beta = sigmoid(b); g = compute_g(a, A_log, dt_bias) — exactly what gated_delta_update does.
        let beta = candle_nn::ops::sigmoid(&b_raw).unwrap();
        let g = compute_g(&a, &a_log, &dt_bias).unwrap();
        let (y, state) = gated_delta_recurrence(&q, &k, &v, &g, &beta, None).unwrap();

        assert_eq!(y.dims(), &[1, 3, 4, 2]);
        assert_eq!(state.dims(), &[1, 4, 2, 2]);

        let yh = host(&y);
        let sh = host(&state);
        assert!(
            max_abs_diff(&yh, EXP_Y) < 1e-4,
            "y diff {}",
            max_abs_diff(&yh, EXP_Y)
        );
        assert!(
            max_abs_diff(&sh, EXP_STATE) < 1e-4,
            "state diff {}",
            max_abs_diff(&sh, EXP_STATE)
        );
    }

    #[test]
    fn decode_step_matches_chunked_prefill() {
        // Feeding the sequence one token at a time (carrying state) must equal one T-step call —
        // the prefill/decode equivalence the hybrid cache relies on.
        let q = Tensor::from_slice(Q, (1, 3, 2, 2), &Device::Cpu).unwrap();
        let k = Tensor::from_slice(K, (1, 3, 2, 2), &Device::Cpu).unwrap();
        let v = Tensor::from_slice(V, (1, 3, 4, 2), &Device::Cpu).unwrap();
        let g = Tensor::from_slice(
            &[
                0.6f32, 0.7, 0.8, 0.9, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95, 0.4, 0.45,
            ],
            (1, 3, 4),
            &Device::Cpu,
        )
        .unwrap();
        let beta = Tensor::from_slice(
            &[
                0.2f32, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.15, 0.25, 0.35,
            ],
            (1, 3, 4),
            &Device::Cpu,
        )
        .unwrap();

        let (y_full, s_full) = gated_delta_recurrence(&q, &k, &v, &g, &beta, None).unwrap();

        // Step one token at a time, carrying the state.
        let pick =
            |x: &Tensor, t: usize| -> Tensor { x.narrow(1, t, 1).unwrap().contiguous().unwrap() };
        let mut state: Option<Tensor> = None;
        let mut ys = Vec::new();
        for t in 0..3 {
            let (y, s) = gated_delta_recurrence(
                &pick(&q, t),
                &pick(&k, t),
                &pick(&v, t),
                &pick(&g, t),
                &pick(&beta, t),
                state.as_ref(),
            )
            .unwrap();
            ys.push(y);
            state = Some(s);
        }
        let refs: Vec<&Tensor> = ys.iter().collect();
        let y_step = Tensor::cat(&refs, 1).unwrap();

        assert!(
            max_abs_diff(&host(&y_full), &host(&y_step)) < 1e-5,
            "prefill vs step y: {}",
            max_abs_diff(&host(&y_full), &host(&y_step))
        );
        assert!(
            max_abs_diff(&host(&s_full), &host(&state.unwrap())) < 1e-5,
            "prefill vs step state"
        );
    }

    #[test]
    fn compute_g_is_in_unit_interval_and_shaped() {
        // g = exp(−positive) ∈ (0, 1]: a per-head forget gate.
        let a = Tensor::from_slice(A, (1, 3, 4), &Device::Cpu).unwrap();
        let a_log = Tensor::from_slice(A_LOG, (4,), &Device::Cpu).unwrap();
        let dt_bias = Tensor::from_slice(DT_BIAS, (4,), &Device::Cpu).unwrap();
        let g = compute_g(&a, &a_log, &dt_bias).unwrap();
        assert_eq!(g.dims(), &[1, 3, 4]);
        for x in host(&g) {
            assert!(x > 0.0 && x <= 1.0 + 1e-6, "gate out of (0,1]: {x}");
        }
    }

    // Conv oracle from MLX's depthwise `nn.Conv1d` + silu — C=3, K=4, S=2, conv_state K-1=3.
    const CW: &[f32] = &[
        0.2422637, 0.3207079, 0.3157361, 0.493552, 0.6888506, 0.2758986, -0.2604986, 0.5719915,
        -0.677193, -0.1786878, -0.1721832, 0.024104,
    ];
    const CX: &[f32] = &[
        -0.3929182, -0.15279, -0.0340949, -0.1223795, -0.5179096, 0.7106992,
    ];
    const CSTATE: &[f32] = &[
        0.2526282, 0.8012137, 0.0075967, -0.7815045, -0.8113322, -0.8019857, 0.0879031, 0.8514872,
        0.00599,
    ];
    const CEXP_OUT: &[f32] = &[
        -0.1465172, 0.0095216, 0.0727914, -0.1432332, -0.2082713, 0.3602719,
    ];
    const CEXP_STATE: &[f32] = &[
        0.0879031, 0.8514872, 0.00599, -0.3929182, -0.15279, -0.0340949, -0.1223795, -0.5179096,
        0.7106992,
    ];

    #[test]
    fn causal_conv_matches_mlx_conv1d() {
        // weight stored [C,K,1] in the checkpoint; the helper takes [C,K] (squeezed).
        let weight = Tensor::from_slice(CW, (3, 4), &Device::Cpu).unwrap();
        let x = Tensor::from_slice(CX, (1, 2, 3), &Device::Cpu).unwrap();
        let state = Tensor::from_slice(CSTATE, (1, 3, 3), &Device::Cpu).unwrap();
        let (out, new_state) = causal_depthwise_conv(&x, &weight, &state).unwrap();
        assert_eq!(out.dims(), &[1, 2, 3]);
        assert_eq!(new_state.dims(), &[1, 3, 3]);
        assert!(max_abs_diff(&host(&out), CEXP_OUT) < 1e-5);
        // new conv_state is the last K-1 tokens of [conv_state ++ x] — exact (a slice, no arithmetic).
        assert_eq!(host(&new_state), CEXP_STATE.to_vec());
    }

    #[test]
    fn delta_cache_tracks_state_and_offset() {
        let mut cache = DeltaNetCache::new();
        assert_eq!(cache.offset(), 0);
        assert!(cache.conv_state.is_none() && cache.ssm_state.is_none());
        let conv = Tensor::zeros((1, 3, 3), DType::F32, &Device::Cpu).unwrap();
        let ssm = Tensor::zeros((1, 4, 2, 2), DType::F32, &Device::Cpu).unwrap();
        cache.update(conv, ssm, 5);
        assert_eq!(cache.offset(), 5);
        assert!(cache.conv_state.is_some() && cache.ssm_state.is_some());
        cache.reset();
        assert_eq!(cache.offset(), 0);
        assert!(cache.conv_state.is_none());
    }
}
