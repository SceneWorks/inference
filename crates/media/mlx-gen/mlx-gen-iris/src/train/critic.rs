//! The restoration recipe's **critic**, native MLX — HYPIR's `ImageConvNextDiscriminator`
//! (`HYPIR/model/D.py`, `backbone.py`), training-only:
//!
//! * a **frozen** OpenCLIP ConvNeXt-XXL vision trunk ([`ConvNextTrunk`]: timm `convnext_xxlarge` +
//!   OpenCLIP's `head.proj`) on `(x·0.5 + 0.5 − mean) / std`, read out as HYPIR's
//!   `_visual_forward(return_pooled_feats=True)`: the stage-0/1/2 maps and the pooled embedding;
//! * the **trained** `MultiLevelDConv(level=4, down=2)` head — per map `BlurPool(pad 2) → SN-conv3×3
//!   s2 → LeakyReLU(0.2) → BlurPool(pad 1) → SN-conv1×1 s2 → [B, h, w]`, and `SN-Linear →
//!   LeakyReLU → SN-Linear → [B, 1]` on the embedding — with `torch.nn.utils.spectral_norm`
//!   (`W / uᵀWv`, one power iteration per forward in train mode, none in eval mode);
//! * `multilevel_loss(alpha)`: BCE-with-logits against `alpha` (real / generator) or 0 (fake),
//!   spatially averaged per level and summed over levels → `[B]`.
//!
//! Precision mirrors HYPIR's `precision="bf16"` trunk under accelerate's bf16 autocast: weights of
//! convs / linears in the compute dtype, LayerNorms and every reduction in f32 (`fp32` computes
//! everything in f32, the parity path).

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use gen_core::iris::restoration_train::{
    sn_keys, ConvNextConfig, CriticHeadConfig, HostTensor, BLUR_TAPS, CLIP_MEAN, CLIP_STD,
    CRITIC_LEAKY_SLOPE, SPECTRAL_EPS,
};
use mlx_gen::array::contiguous;
use mlx_gen::gen_core;
use mlx_gen::nn::conv2d_general;
use mlx_gen::{Error, Result};
use mlx_rs::ops::{abs, erf, exp, log1p, matmul, maximum, pad, sqrt};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use super::optim::Params;

fn f32s(v: f64) -> Array {
    Array::from_f32(v as f32)
}

/// LayerNorm over the last axis, f32, affine.
fn layer_norm(x: &Array, w: &Array, b: &Array, eps: f64) -> Result<Array> {
    let x = x.as_dtype(Dtype::Float32)?;
    let mean = x.mean_axes(&[-1], true)?;
    let xc = x.subtract(&mean)?;
    let var = xc.square()?.mean_axes(&[-1], true)?;
    Ok(xc
        .multiply(&mlx_rs::ops::rsqrt(&var.add(f32s(eps))?)?)?
        .multiply(w)?
        .add(b)?)
}

/// `nn.GELU()` (erf form).
fn gelu(x: &Array) -> Result<Array> {
    let half = f32s(0.5);
    let inv_sqrt2 = f32s(std::f64::consts::FRAC_1_SQRT_2);
    Ok(x.multiply(&half)?
        .multiply(&erf(&x.multiply(&inv_sqrt2)?)?.add(f32s(1.0))?)?)
}

/// `y = x · Wᵀ + b` with `x` cast to the compute dtype, output f32. Both GEMM operands are made
/// row-contiguous first (one dense `[in, out]` weight read by every call).
fn linear(x: &Array, w: &Array, b: Option<&Array>, compute: Dtype) -> Result<Array> {
    linear_t(x, &contiguous(&w.t())?, b, compute)
}

/// [`linear`] with an already transposed, contiguous `[in, out]` weight.
fn linear_t(x: &Array, wt: &Array, b: Option<&Array>, compute: Dtype) -> Result<Array> {
    let y = matmul(&contiguous(&x.as_dtype(compute)?)?, wt)?.as_dtype(Dtype::Float32)?;
    Ok(match b {
        Some(b) => y.add(b)?,
        None => y,
    })
}

/// NHWC conv with an OHWI weight in the compute dtype, f32 output.
fn conv(
    x: &Array,
    w: &Array,
    b: Option<&Array>,
    stride: i32,
    padding: i32,
    groups: i32,
    compute: Dtype,
) -> Result<Array> {
    // Dense input (the blur's shifted slices are strided views).
    let y = conv2d_general(
        &contiguous(&x.as_dtype(compute)?)?,
        w,
        None,
        (stride, stride),
        (padding, padding),
        (1, 1),
        groups,
    )?
    .as_dtype(Dtype::Float32)?;
    Ok(match b {
        Some(b) => y.add(b)?,
        None => y,
    })
}

// =============================================================================================
// The frozen trunk
// =============================================================================================

struct Block {
    dw_w: Array,
    dw_b: Array,
    ln_w: Array,
    ln_b: Array,
    fc1_w: Array,
    fc1_b: Array,
    fc2_w: Array,
    fc2_b: Array,
    gamma: Array,
}

struct Stage {
    down: Option<(Array, Array, Array, Array)>,
    blocks: Vec<Block>,
}

/// The frozen OpenCLIP ConvNeXt vision trunk.
pub struct ConvNextTrunk {
    cfg: ConvNextConfig,
    compute: Dtype,
    stem_w: Array,
    stem_b: Array,
    stem_ln: (Array, Array),
    stages: Vec<Stage>,
    head_ln: (Array, Array),
    proj: Array,
}

impl ConvNextTrunk {
    /// Build from the OpenCLIP `visual.*` tensors (torch layouts). Conv / linear weights are stored
    /// in `compute`, LayerNorm parameters, biases and LayerScale in f32.
    pub fn from_map(
        map: &HashMap<String, Array>,
        cfg: &ConvNextConfig,
        compute: Dtype,
    ) -> Result<Self> {
        for (k, shape) in cfg.tensor_shapes() {
            let a = map
                .get(&k)
                .ok_or_else(|| Error::Msg(format!("critic trunk: {k} missing")))?;
            let s: Vec<usize> = a.shape().iter().map(|&d| d as usize).collect();
            if s != shape {
                return Err(Error::Msg(format!(
                    "critic trunk: {k} is {s:?}, the configured trunk expects {shape:?}"
                )));
            }
        }
        let f = |k: &str| -> Result<Array> { Ok(map[k].as_dtype(Dtype::Float32)?) };
        let cw = |k: &str| -> Result<Array> {
            // OIHW → OHWI, compute dtype.
            contiguous(&map[k].transpose_axes(&[0, 2, 3, 1])?.as_dtype(compute)?)
        };
        // Linear weights pre-transposed to a contiguous `[in, out]` (see [`linear`]).
        let lw = |k: &str| -> Result<Array> { contiguous(&map[k].t().as_dtype(compute)?) };
        let t = "visual.trunk";
        let mut stages = Vec::new();
        for (s, &depth) in cfg.depths.iter().enumerate() {
            let p = format!("{t}.stages.{s}");
            let down = if s > 0 {
                Some((
                    f(&format!("{p}.downsample.0.weight"))?,
                    f(&format!("{p}.downsample.0.bias"))?,
                    cw(&format!("{p}.downsample.1.weight"))?,
                    f(&format!("{p}.downsample.1.bias"))?,
                ))
            } else {
                None
            };
            let mut blocks = Vec::with_capacity(depth);
            for b in 0..depth {
                let q = format!("{p}.blocks.{b}");
                blocks.push(Block {
                    dw_w: cw(&format!("{q}.conv_dw.weight"))?,
                    dw_b: f(&format!("{q}.conv_dw.bias"))?,
                    ln_w: f(&format!("{q}.norm.weight"))?,
                    ln_b: f(&format!("{q}.norm.bias"))?,
                    fc1_w: lw(&format!("{q}.mlp.fc1.weight"))?,
                    fc1_b: f(&format!("{q}.mlp.fc1.bias"))?,
                    fc2_w: lw(&format!("{q}.mlp.fc2.weight"))?,
                    fc2_b: f(&format!("{q}.mlp.fc2.bias"))?,
                    gamma: f(&format!("{q}.gamma"))?,
                });
            }
            stages.push(Stage { down, blocks });
        }
        let trunk = Self {
            cfg: cfg.clone(),
            compute,
            stem_w: cw(&format!("{t}.stem.0.weight"))?,
            stem_b: f(&format!("{t}.stem.0.bias"))?,
            stem_ln: (
                f(&format!("{t}.stem.1.weight"))?,
                f(&format!("{t}.stem.1.bias"))?,
            ),
            stages,
            head_ln: (
                f(&format!("{t}.head.norm.weight"))?,
                f(&format!("{t}.head.norm.bias"))?,
            ),
            proj: lw("visual.head.proj.weight")?,
        };
        trunk.materialize()?;
        Ok(trunk)
    }

    fn materialize(&self) -> Result<()> {
        let mut all: Vec<&Array> = vec![
            &self.stem_w,
            &self.stem_b,
            &self.stem_ln.0,
            &self.stem_ln.1,
            &self.head_ln.0,
            &self.head_ln.1,
            &self.proj,
        ];
        for s in &self.stages {
            if let Some((a, b, c, d)) = &s.down {
                all.extend([a, b, c, d]);
            }
            for b in &s.blocks {
                all.extend([
                    &b.dw_w, &b.dw_b, &b.ln_w, &b.ln_b, &b.fc1_w, &b.fc1_b, &b.fc2_w, &b.fc2_b,
                    &b.gamma,
                ]);
            }
        }
        for chunk in all.chunks(128) {
            eval(chunk.iter().copied())?;
        }
        Ok(())
    }

    /// Load the pinned OpenCLIP checkpoint (only its `visual.*` tensors are read).
    pub fn load(path: &Path, cfg: &ConvNextConfig, compute: Dtype) -> Result<Self> {
        let map = mlx_gen::pth::load_pth_f32_filtered(path, |k| k.starts_with("visual."))?;
        Self::from_map(&map, cfg, compute)
    }

    /// `_visual_forward(…, return_pooled_feats=True)` on an NCHW image in `[−1, 1]` (after HYPIR's
    /// CLIP normalization): `[stage0, stage1, stage2]` NHWC maps (f32) and the pooled `[B, E]`
    /// embedding.
    pub fn features(&self, x: &Array) -> Result<(Vec<Array>, Array)> {
        let eps = self.cfg.norm_eps;
        let mean = Array::from_slice(&CLIP_MEAN, &[1, 1, 1, 3]);
        let std = Array::from_slice(&CLIP_STD, &[1, 1, 1, 3]);
        let x = x
            .as_dtype(Dtype::Float32)?
            .transpose_axes(&[0, 2, 3, 1])?
            .multiply(f32s(0.5))?
            .add(f32s(0.5))?
            .subtract(&mean)?
            .divide(&std)?;
        let mut h = conv(&x, &self.stem_w, Some(&self.stem_b), 4, 0, 1, self.compute)?;
        h = layer_norm(&h, &self.stem_ln.0, &self.stem_ln.1, eps)?;
        let mut maps = Vec::with_capacity(3);
        for (s, stage) in self.stages.iter().enumerate() {
            if let Some((lw, lb, cw, cb)) = &stage.down {
                h = layer_norm(&h, lw, lb, eps)?;
                h = conv(&h, cw, Some(cb), 2, 0, 1, self.compute)?;
            }
            let c = self.cfg.dims[s] as i32;
            for b in &stage.blocks {
                let r = conv(&h, &b.dw_w, Some(&b.dw_b), 1, 3, c, self.compute)?;
                let r = layer_norm(&r, &b.ln_w, &b.ln_b, eps)?;
                let r = gelu(&linear_t(&r, &b.fc1_w, Some(&b.fc1_b), self.compute)?)?;
                let r = linear_t(&r, &b.fc2_w, Some(&b.fc2_b), self.compute)?;
                h = h.add(&r.multiply(&b.gamma)?)?;
            }
            if s < 3 {
                maps.push(h.clone());
            }
        }
        // trunk.forward_head: global average pool → LayerNorm; visual.head.proj (no bias).
        let pooled = h.mean_axes(&[1, 2], false)?;
        let pooled = layer_norm(&pooled, &self.head_ln.0, &self.head_ln.1, eps)?;
        let emb = linear_t(&pooled, &self.proj, None, self.compute)?;
        Ok((maps, emb))
    }
}

// =============================================================================================
// The trained head
// =============================================================================================

/// The head's power-iteration buffers (`weight_u`, `weight_v` per spectral layer).
pub type SpectralState = HashMap<String, (Array, Array)>;

/// Split a host init / a checkpoint into the trained parameters (`weight_orig`, `bias`) and the
/// spectral buffers.
pub fn head_from_host(
    cfg: &CriticHeadConfig,
    init: &[HostTensor],
) -> Result<(Params, SpectralState)> {
    let map: HashMap<&str, Array> = init
        .iter()
        .map(|(k, s, v)| {
            let shape: Vec<i32> = s.iter().map(|&d| d as i32).collect();
            (k.as_str(), Array::from_slice(v, &shape))
        })
        .collect();
    head_from_arrays(cfg, |k| map.get(k).cloned())
}

/// [`head_from_host`] from any keyed source of arrays.
pub fn head_from_arrays(
    cfg: &CriticHeadConfig,
    get: impl Fn(&str) -> Option<Array>,
) -> Result<(Params, SpectralState)> {
    let mut params = Params::new();
    let mut sn = SpectralState::new();
    for layer in cfg.layers() {
        let [w, b, u, v] = sn_keys(&layer.prefix);
        let need = |k: &str| -> Result<Array> {
            Ok(get(k)
                .ok_or_else(|| Error::Msg(format!("critic head: {k} missing")))?
                .as_dtype(Dtype::Float32)?)
        };
        let wa = need(&w)?;
        let shape: Vec<usize> = wa.shape().iter().map(|&d| d as usize).collect();
        if shape != layer.shape {
            return Err(Error::Msg(format!(
                "critic head: {w} is {shape:?}, expected {:?}",
                layer.shape
            )));
        }
        params.insert(Rc::from(w.as_str()), wa);
        params.insert(Rc::from(b.as_str()), need(&b)?);
        sn.insert(layer.prefix.clone(), (need(&u)?, need(&v)?));
    }
    let refs: Vec<&Array> = params
        .values()
        .chain(sn.values().flat_map(|(u, v)| [u, v]))
        .collect();
    eval(refs)?;
    Ok((params, sn))
}

fn normalize(v: &Array) -> Result<Array> {
    let n = sqrt(&v.square()?.sum(None)?)?;
    Ok(v.divide(&maximum(&n, f32s(SPECTRAL_EPS as f64))?)?)
}

/// One `torch.nn.utils.spectral_norm` power iteration on every layer (train mode, before a
/// forward): `v ← normalize(Wᵀu)`, `u ← normalize(W v)` on the current `weight_orig`. No gradient.
pub fn power_iteration(
    cfg: &CriticHeadConfig,
    params: &Params,
    sn: &SpectralState,
) -> Result<SpectralState> {
    let mut out = SpectralState::new();
    for layer in cfg.layers() {
        let (rows, cols) = layer.matrix();
        let [wk, ..] = sn_keys(&layer.prefix);
        let w =
            mlx_rs::stop_gradient(&params[wk.as_str()])?.reshape(&[rows as i32, cols as i32])?;
        let (u, _) = &sn[&layer.prefix];
        let v = normalize(
            &matmul(&contiguous(&w.t())?, &u.reshape(&[rows as i32, 1])?)?
                .reshape(&[cols as i32])?,
        )?;
        let u = normalize(&matmul(&w, &v.reshape(&[cols as i32, 1])?)?.reshape(&[rows as i32])?)?;
        eval([&u, &v])?;
        out.insert(layer.prefix.clone(), (u, v));
    }
    Ok(out)
}

/// `W / σ`, `σ = uᵀ W v` (`u`, `v` constants; the gradient flows through both `W` and `σ`).
fn spectral_weight(w: &Array, u: &Array, v: &Array) -> Result<Array> {
    let sh = w.shape();
    let rows = sh[0];
    let cols: i32 = sh[1..].iter().product();
    let wm = w.reshape(&[rows, cols])?;
    let wv = matmul(&wm, &v.reshape(&[cols, 1])?)?.reshape(&[rows])?;
    let sigma = u.multiply(&wv)?.sum(None)?;
    Ok(w.divide(&sigma)?)
}

/// `BlurPool(filt_size=4, stride=1)` with zero padding `pad` on every side, depthwise (compute
/// dtype, f32 output) — as the 16 shifted, weighted slices of the padded map (the kernel is the
/// same `[1, 3, 3, 1]ᵀ[1, 3, 3, 1] / 64` for every channel).
fn blur(x: &Array, pad_px: i32, compute: Dtype) -> Result<Array> {
    use mlx_rs::ops::indexing::IndexOp;
    let sh = x.shape();
    let (h, w) = (sh[1] + 2 * pad_px - 3, sh[2] + 2 * pad_px - 3);
    let xp = pad(
        &x.as_dtype(compute)?,
        &[(0, 0), (pad_px, pad_px), (pad_px, pad_px), (0, 0)],
        None,
        None,
    )?;
    let mut acc: Option<Array> = None;
    for (i, a) in BLUR_TAPS.iter().enumerate() {
        for (j, b) in BLUR_TAPS.iter().enumerate() {
            let (i, j) = (i as i32, j as i32);
            let term = xp
                .index_device(
                    (.., i..i + h, j..j + w, ..),
                    mlx_rs::Stream::task_local_or_default(),
                )
                .multiply(Array::from_f32(a * b / 64.0).as_dtype(compute)?)?;
            acc = Some(match acc {
                None => term,
                Some(t) => t.add(&term)?,
            });
        }
    }
    Ok(acc.expect("16 taps").as_dtype(Dtype::Float32)?)
}

fn leaky(x: &Array) -> Result<Array> {
    let slope = f32s(CRITIC_LEAKY_SLOPE as f64);
    Ok(maximum(x, &x.multiply(&slope)?)?)
}

/// `MultiLevelDConv.forward`: three `[B, h, w]` logit maps and the `[B, 1]` embedding logit (f32).
pub fn head_logits(
    cfg: &CriticHeadConfig,
    params: &Params,
    sn: &SpectralState,
    maps: &[Array],
    emb: &Array,
    compute: Dtype,
) -> Result<Vec<Array>> {
    let sw = |prefix: &str| -> Result<(Array, Array)> {
        let [w, b, ..] = sn_keys(prefix);
        let (u, v) = &sn[prefix];
        Ok((
            spectral_weight(&params[w.as_str()], u, v)?,
            params[b.as_str()].clone(),
        ))
    };
    let mut out = Vec::with_capacity(4);
    for (i, x) in maps.iter().enumerate().take(cfg.in_ch1.len()) {
        let (w1, b1) = sw(&format!("decoder.{i}.1"))?;
        let (w4, b4) = sw(&format!("decoder.{i}.4"))?;
        let h = blur(x, 2, compute)?;
        let h = conv(
            &h,
            &contiguous(&w1.transpose_axes(&[0, 2, 3, 1])?.as_dtype(compute)?)?,
            Some(&b1),
            2,
            0,
            1,
            compute,
        )?;
        let h = blur(&leaky(&h)?, 1, compute)?;
        let h = conv(
            &h,
            &contiguous(&w4.transpose_axes(&[0, 2, 3, 1])?.as_dtype(compute)?)?,
            Some(&b4),
            2,
            0,
            1,
            compute,
        )?;
        let sh = h.shape();
        out.push(h.reshape(&[sh[0], sh[1], sh[2]])?);
    }
    let (w, b) = sw("decoder.3.0")?;
    let h = leaky(&linear(emb, &w.as_dtype(compute)?, Some(&b), compute)?)?;
    let (w, b) = sw("out")?;
    out.push(linear(&h, &w.as_dtype(compute)?, Some(&b), compute)?);
    Ok(out)
}

/// `nn.BCEWithLogitsLoss(reduction="none")`: `max(x, 0) − x·t + log(1 + e^{−|x|})`.
fn bce(x: &Array, target: f64) -> Result<Array> {
    let zero = f32s(0.0);
    Ok(maximum(x, &zero)?
        .subtract(&x.multiply(f32s(target))?)?
        .add(&log1p(&exp(&abs(x)?.negative()?)?)?)?)
}

/// `multilevel_loss(alpha)(logits, target)` → `[B]`: each conv level's BCE averaged over
/// `(h, w)`, plus the embedding level's, summed.
pub fn multilevel_loss(logits: &[Array], target: f64) -> Result<Array> {
    let mut total: Option<Array> = None;
    for l in logits {
        let e = bce(l, target)?;
        let per = if e.ndim() > 2 {
            e.mean_axes(&[1, 2], false)?
        } else {
            e.reshape(&[-1])?
        };
        total = Some(match total {
            None => per,
            Some(t) => t.add(&per)?,
        });
    }
    total.ok_or_else(|| Error::Msg("critic: no logits".into()))
}

/// The frozen trunk + head geometry of one run.
pub struct Critic {
    pub trunk: ConvNextTrunk,
    pub head: CriticHeadConfig,
    pub compute: Dtype,
}

impl Critic {
    /// `D(x, …)`'s logits with the given head state.
    pub fn logits(&self, params: &Params, sn: &SpectralState, x: &Array) -> Result<Vec<Array>> {
        let (maps, emb) = self.trunk.features(x)?;
        head_logits(&self.head, params, sn, &maps, &emb, self.compute)
    }

    /// `D(x, for_G=True)` (eval mode: no power iteration) → `[B]`.
    pub fn generator_term(
        &self,
        params: &Params,
        sn: &SpectralState,
        x: &Array,
        label: f64,
    ) -> Result<Array> {
        multilevel_loss(&self.logits(params, sn, x)?, label)
    }
}
