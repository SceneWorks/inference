//! The restoration recipe's **critic** on Candle — the twin of `mlx_gen_iris::train::critic`
//! (HYPIR's `ImageConvNextDiscriminator`: a frozen OpenCLIP ConvNeXt-XXL trunk read out as
//! `_visual_forward(return_pooled_feats=True)`, the trained spectral-normalized
//! `MultiLevelDConv(level=4, down=2)` head, `multilevel_loss(alpha)`), training-only.
//!
//! NCHW. Depthwise convolutions (the trunk's 7×7 `conv_dw`, the head's BlurPool) are computed as
//! sums of shifted, per-channel-weighted slices: Candle lowers `groups > 1` to one convolution per
//! group (3072 of them on the last stage), and the slice form is the same arithmetic with an
//! autograd path. Precision mirrors HYPIR's bf16 trunk under bf16 autocast: conv / linear operands
//! in the compute dtype, LayerNorm and reductions in f32.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor, Var, D};
use candle_gen::{CandleError as Error, Result};
use gen_core_iris::restoration_train::{
    sn_keys, ConvNextConfig, CriticHeadConfig, HostTensor, BLUR_TAPS, CLIP_MEAN, CLIP_STD,
    CRITIC_LEAKY_SLOPE, SPECTRAL_EPS,
};

use super::optim::{Params, Tensors};

/// LayerNorm over the channel axis 1 of an NCHW map (or the last axis of `[B, C]`), f32, affine.
fn layer_norm_c(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64, axis: usize) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(axis)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(axis)?;
    let y = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    let mut shape = vec![1usize; x.rank()];
    shape[axis] = w.dim(0)?;
    Ok(y.broadcast_mul(&w.reshape(shape.as_slice())?)?
        .broadcast_add(&b.reshape(shape.as_slice())?)?)
}

/// `x · Wᵀ + b` over the last axis, operands in `compute`, f32 out.
fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>, compute: DType) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let inn = *dims.last().unwrap();
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let y = x
        .to_dtype(compute)?
        .reshape((rows, inn))?
        .matmul(&w.to_dtype(compute)?.t()?)?
        .to_dtype(DType::F32)?;
    let out = w.dim(0)?;
    let mut shape = dims[..dims.len() - 1].to_vec();
    shape.push(out);
    let y = y.reshape(shape)?;
    Ok(match b {
        Some(b) => y.broadcast_add(b)?,
        None => y,
    })
}

/// A plain (groups = 1) conv, operands in `compute`, f32 out, bias `[C]`.
fn conv(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    stride: usize,
    padding: usize,
    compute: DType,
) -> Result<Tensor> {
    let y = x
        .to_dtype(compute)?
        .conv2d(&w.to_dtype(compute)?, padding, stride, 1, 1)?
        .to_dtype(DType::F32)?;
    Ok(match b {
        Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1, 1))?)?,
        None => y,
    })
}

/// Depthwise `k × k` stride-1 correlation with per-channel `[C, 1, k, k]` weights and zero padding
/// `pad`, as `Σ_{i,j} w[:, i, j] · x_pad[:, :, i:i+H', j:j+W']` (compute dtype, f32 out). `fold`
/// puts the sum on the tape as one node; a caller that is itself replayed inside one passes `false`.
fn depthwise(x: &Tensor, w: &Tensor, pad: usize, compute: DType, fold: bool) -> Result<Tensor> {
    let (_, c, h, wd) = x.dims4()?;
    let k = w.dim(2)?;
    let (oh, ow) = (h + 2 * pad + 1 - k, wd + 2 * pad + 1 - k);
    let xp = x
        .to_dtype(compute)?
        .pad_with_zeros(2, pad, pad)?
        .pad_with_zeros(3, pad, pad)?;
    let w = w.to_dtype(compute)?.reshape((c, k * k))?;
    // One tape node (sc-25690) when the taps are constants and the padded map is this call's own:
    // the plain chain keeps `2·k² − 1` feature maps on the tape per call.
    let fold = fold && !w.track_op() && xp.id() != x.id();
    let sum = move |xp: &Tensor| -> candle_gen::candle_core::Result<Tensor> {
        let mut acc: Option<Tensor> = None;
        for i in 0..k {
            for j in 0..k {
                let tap = w.narrow(1, i * k + j, 1)?.reshape((1, c, 1, 1))?;
                let term = xp.narrow(2, i, oh)?.narrow(3, j, ow)?.broadcast_mul(&tap)?;
                acc = Some(match acc {
                    None => term,
                    Some(a) => a.add(&term)?,
                });
            }
        }
        Ok(acc.expect("k ≥ 1"))
    };
    let acc = if fold {
        crate::tape::fold1("iris-depthwise", &xp, sum)?
    } else {
        sum(&xp)?
    };
    Ok(acc.to_dtype(DType::F32)?)
}

fn gelu(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu_erf()?)
}

// =============================================================================================
// The frozen trunk
// =============================================================================================

#[derive(Clone)]
struct Block {
    dw_w: Tensor,
    dw_b: Tensor,
    ln_w: Tensor,
    ln_b: Tensor,
    fc1_w: Tensor,
    fc1_b: Tensor,
    fc2_w: Tensor,
    fc2_b: Tensor,
    gamma: Tensor,
}

impl Block {
    /// `residual + γ · mlp(norm(dwconv(x)))` — timm's `ConvNeXtBlock` (`conv_mlp=False`), with the
    /// block input named once per reader.
    fn forward(&self, residual: &Tensor, x: &Tensor, eps: f64, compute: DType) -> Result<Tensor> {
        let r = __model_proof_output!(depthwise(x, &self.dw_w, 3, compute, false)?
            .broadcast_add(&self.dw_b.reshape((1, self.dw_b.dim(0)?, 1, 1))?))?;
        // channels-last MLP (timm `conv_mlp=False`)
        let r = r.permute((0, 2, 3, 1))?;
        let r = layer_norm_c(&r, &self.ln_w, &self.ln_b, eps, 3)?;
        let r = gelu(&__model_proof_output!(linear(
            &r,
            &self.fc1_w,
            Some(&self.fc1_b),
            compute
        ))?)?;
        let r = __model_proof_output!(linear(&r, &self.fc2_w, Some(&self.fc2_b), compute))?;
        let r = __model_proof_output!(r.broadcast_mul(&self.gamma))?.permute((0, 3, 1, 2))?;
        Ok(residual.add(&r)?)
    }

    fn frozen(&self) -> bool {
        ![
            &self.dw_w,
            &self.dw_b,
            &self.ln_w,
            &self.ln_b,
            &self.fc1_w,
            &self.fc1_b,
            &self.fc2_w,
            &self.fc2_b,
            &self.gamma,
        ]
        .iter()
        .any(|t| t.track_op())
    }
}

struct Stage {
    down: Option<(Tensor, Tensor, Tensor, Tensor)>,
    blocks: Vec<Block>,
}

/// The frozen OpenCLIP ConvNeXt vision trunk.
pub struct ConvNextTrunk {
    cfg: ConvNextConfig,
    compute: DType,
    device: Device,
    stem_w: Tensor,
    stem_b: Tensor,
    stem_ln: (Tensor, Tensor),
    stages: Vec<Stage>,
    head_ln: (Tensor, Tensor),
    proj: Tensor,
}

impl ConvNextTrunk {
    /// Build from the OpenCLIP `visual.*` tensors. Conv / linear weights in `compute`, the rest f32.
    pub fn from_map(
        map: &HashMap<String, Tensor>,
        cfg: &ConvNextConfig,
        compute: DType,
        device: &Device,
    ) -> Result<Self> {
        for (k, shape) in cfg.tensor_shapes() {
            let a = map
                .get(&k)
                .ok_or_else(|| Error::Msg(format!("critic trunk: {k} missing")))?;
            if a.dims() != shape.as_slice() {
                return Err(Error::Msg(format!(
                    "critic trunk: {k} is {:?}, the configured trunk expects {shape:?}",
                    a.dims()
                )));
            }
        }
        let f = |k: &str| -> Result<Tensor> { Ok(map[k].to_dtype(DType::F32)?.to_device(device)?) };
        let c = |k: &str| -> Result<Tensor> { Ok(map[k].to_dtype(compute)?.to_device(device)?) };
        let t = "visual.trunk";
        let mut stages = Vec::new();
        for (s, &depth) in cfg.depths.iter().enumerate() {
            let p = format!("{t}.stages.{s}");
            let down = if s > 0 {
                Some((
                    f(&format!("{p}.downsample.0.weight"))?,
                    f(&format!("{p}.downsample.0.bias"))?,
                    c(&format!("{p}.downsample.1.weight"))?,
                    f(&format!("{p}.downsample.1.bias"))?,
                ))
            } else {
                None
            };
            let mut blocks = Vec::with_capacity(depth);
            for b in 0..depth {
                let q = format!("{p}.blocks.{b}");
                blocks.push(Block {
                    dw_w: c(&format!("{q}.conv_dw.weight"))?,
                    dw_b: f(&format!("{q}.conv_dw.bias"))?,
                    ln_w: f(&format!("{q}.norm.weight"))?,
                    ln_b: f(&format!("{q}.norm.bias"))?,
                    fc1_w: c(&format!("{q}.mlp.fc1.weight"))?,
                    fc1_b: f(&format!("{q}.mlp.fc1.bias"))?,
                    fc2_w: c(&format!("{q}.mlp.fc2.weight"))?,
                    fc2_b: f(&format!("{q}.mlp.fc2.bias"))?,
                    gamma: f(&format!("{q}.gamma"))?,
                });
            }
            stages.push(Stage { down, blocks });
        }
        Ok(Self {
            cfg: cfg.clone(),
            compute,
            device: device.clone(),
            stem_w: c(&format!("{t}.stem.0.weight"))?,
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
            proj: c("visual.head.proj.weight")?,
        })
    }

    /// Load the pinned OpenCLIP checkpoint (only its `visual.*` tensors are kept).
    pub fn load(
        path: &Path,
        cfg: &ConvNextConfig,
        compute: DType,
        device: &Device,
    ) -> Result<Self> {
        let pth = candle_gen::candle_core::pickle::PthTensors::new(path, None)
            .map_err(|e| Error::Msg(format!("critic trunk: read {}: {e}", path.display())))?;
        let mut map = HashMap::new();
        for (k, _) in cfg.tensor_shapes() {
            let t = pth
                .get(&k)
                .map_err(|e| Error::Msg(format!("critic trunk: {k}: {e}")))?
                .ok_or_else(|| Error::Msg(format!("critic trunk: {k} missing")))?;
            map.insert(k, t);
        }
        Self::from_map(&map, cfg, compute, device)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `_visual_forward(…, return_pooled_feats=True)` on an NCHW image in `[−1, 1]`: the
    /// `[stage0, stage1, stage2]` NCHW maps (f32) and the pooled `[B, E]` embedding.
    pub fn features(&self, x: &Tensor) -> Result<(Vec<Tensor>, Tensor)> {
        let eps = self.cfg.norm_eps;
        let dev = &self.device;
        let mean = Tensor::new(&CLIP_MEAN, dev)?.reshape((1, 3, 1, 1))?;
        let std = Tensor::new(&CLIP_STD, dev)?.reshape((1, 3, 1, 1))?;
        let x = x
            .to_dtype(DType::F32)?
            .affine(0.5, 0.5)?
            .broadcast_sub(&mean)?
            .broadcast_div(&std)?;
        let mut h = conv(&x, &self.stem_w, Some(&self.stem_b), 4, 0, self.compute)?;
        h = layer_norm_c(&h, &self.stem_ln.0, &self.stem_ln.1, eps, 1)?;
        let mut maps = Vec::with_capacity(3);
        for (s, stage) in self.stages.iter().enumerate() {
            if let Some((lw, lb, cw, cb)) = &stage.down {
                h = layer_norm_c(&h, lw, lb, eps, 1)?;
                h = conv(&h, cw, Some(cb), 2, 0, self.compute)?;
            }
            for b in &stage.blocks {
                h = if b.frozen() {
                    // One tape node per frozen block (sc-25690): a gradient to the image crosses
                    // the trunk, and the plain chain keeps ~125 feature maps per block on the tape
                    // for it (the depthwise sum alone 97). The block reads its input twice (the
                    // residual sum and the convolution), so it is passed once per reader; nothing
                    // else reads a block's input.
                    let (block, compute) = (b.clone(), self.compute);
                    crate::tape::fold3("iris-convnext-block", &h, &h, &b.gamma, move |r, x, _| {
                        block
                            .forward(r, x, eps, compute)
                            .map_err(candle_gen::candle_core::Error::wrap)
                    })?
                } else {
                    b.forward(&h, &h, eps, self.compute)?
                };
            }
            if s < 3 {
                maps.push(h.clone());
            }
        }
        let pooled = h.mean(D::Minus1)?.mean(D::Minus1)?;
        let pooled = layer_norm_c(&pooled, &self.head_ln.0, &self.head_ln.1, eps, 1)?;
        let emb = linear(&pooled, &self.proj, None, self.compute)?;
        Ok((maps, emb))
    }
}

// =============================================================================================
// The trained head
// =============================================================================================

/// The power-iteration buffers (`weight_u`, `weight_v`) per spectral layer.
pub type SpectralState = BTreeMap<String, (Tensor, Tensor)>;

/// Split a host init into the trained head parameters (`Var`s) and the spectral buffers.
pub fn head_from_host(
    cfg: &CriticHeadConfig,
    init: &[HostTensor],
    device: &Device,
) -> Result<(Params, SpectralState)> {
    let map: HashMap<&str, Tensor> = init
        .iter()
        .map(|(k, s, v)| {
            Ok((
                k.as_str(),
                Tensor::from_vec(v.clone(), s.as_slice(), device)?,
            ))
        })
        .collect::<Result<_>>()?;
    head_from_tensors(cfg, |k| map.get(k).cloned())
}

/// [`head_from_host`] from any keyed tensor source.
pub fn head_from_tensors(
    cfg: &CriticHeadConfig,
    get: impl Fn(&str) -> Option<Tensor>,
) -> Result<(Params, SpectralState)> {
    let mut params = Params::new();
    let mut sn = SpectralState::new();
    for layer in cfg.layers() {
        let [w, b, u, v] = sn_keys(&layer.prefix);
        let need = |k: &str| -> Result<Tensor> {
            Ok(get(k)
                .ok_or_else(|| Error::Msg(format!("critic head: {k} missing")))?
                .to_dtype(DType::F32)?)
        };
        let wt = need(&w)?;
        if wt.dims() != layer.shape.as_slice() {
            return Err(Error::Msg(format!(
                "critic head: {w} is {:?}, expected {:?}",
                wt.dims(),
                layer.shape
            )));
        }
        params.insert(w, Var::from_tensor(&wt)?);
        params.insert(b.clone(), Var::from_tensor(&need(&b)?)?);
        sn.insert(layer.prefix.clone(), (need(&u)?, need(&v)?));
    }
    Ok((params, sn))
}

fn normalize(v: &Tensor) -> Result<Tensor> {
    let n = v
        .sqr()?
        .sum_all()?
        .sqrt()?
        .to_vec0::<f32>()?
        .max(SPECTRAL_EPS);
    Ok((v / n as f64)?)
}

/// One `spectral_norm` power iteration on every layer (train mode, before a forward). No gradient.
pub fn power_iteration(
    cfg: &CriticHeadConfig,
    params: &Tensors,
    sn: &SpectralState,
) -> Result<SpectralState> {
    let mut out = SpectralState::new();
    for layer in cfg.layers() {
        let (rows, cols) = layer.matrix();
        let [wk, ..] = sn_keys(&layer.prefix);
        let w = params[&wk].detach().reshape((rows, cols))?;
        let (u, _) = &sn[&layer.prefix];
        let v = normalize(&w.t()?.matmul(&u.reshape((rows, 1))?)?.reshape(cols)?)?;
        let u = normalize(&w.matmul(&v.reshape((cols, 1))?)?.reshape(rows)?)?;
        out.insert(layer.prefix.clone(), (u, v));
    }
    Ok(out)
}

/// `W / (uᵀ W v)` (`u`, `v` constants).
fn spectral_weight(w: &Tensor, u: &Tensor, v: &Tensor) -> Result<Tensor> {
    let rows = w.dim(0)?;
    let cols = w.elem_count() / rows;
    let wm = w.reshape((rows, cols))?;
    let sigma = u
        .reshape((1, rows))?
        .matmul(&wm.matmul(&v.reshape((cols, 1))?)?)?
        .reshape(())?;
    Ok(w.broadcast_div(&sigma)?)
}

/// `BlurPool(filt_size=4, stride=1)`, zero padding `pad`, depthwise.
fn blur(x: &Tensor, pad: usize, compute: DType) -> Result<Tensor> {
    let c = x.dim(1)?;
    let mut taps = Vec::with_capacity(16 * c);
    for _ in 0..c {
        for a in BLUR_TAPS {
            for b in BLUR_TAPS {
                taps.push(a * b / 64.0);
            }
        }
    }
    let k = Tensor::from_vec(taps, (c, 1, 4, 4), x.device())?;
    depthwise(x, &k, pad, compute, true)
}

fn leaky(x: &Tensor) -> Result<Tensor> {
    Ok(x.maximum(&(x * CRITIC_LEAKY_SLOPE as f64)?)?)
}

/// `MultiLevelDConv.forward`: three `[B, h, w]` logit maps and the `[B, 1]` embedding logit.
pub fn head_logits(
    cfg: &CriticHeadConfig,
    params: &Tensors,
    sn: &SpectralState,
    maps: &[Tensor],
    emb: &Tensor,
    compute: DType,
) -> Result<Vec<Tensor>> {
    let sw = |prefix: &str| -> Result<(Tensor, Tensor)> {
        let [w, b, ..] = sn_keys(prefix);
        let (u, v) = &sn[prefix];
        Ok((spectral_weight(&params[&w], u, v)?, params[&b].clone()))
    };
    let mut out = Vec::with_capacity(4);
    for (i, x) in maps.iter().enumerate().take(cfg.in_ch1.len()) {
        let (w1, b1) = sw(&format!("decoder.{i}.1"))?;
        let (w4, b4) = sw(&format!("decoder.{i}.4"))?;
        let h = blur(x, 2, compute)?;
        let h = conv(&h, &w1, Some(&b1), 2, 0, compute)?;
        let h = blur(&leaky(&h)?, 1, compute)?;
        let h = conv(&h, &w4, Some(&b4), 2, 0, compute)?;
        out.push(h.squeeze(1)?);
    }
    let (w, b) = sw("decoder.3.0")?;
    let h = leaky(&linear(emb, &w, Some(&b), compute)?)?;
    let (w, b) = sw("out")?;
    out.push(linear(&h, &w, Some(&b), compute)?);
    Ok(out)
}

/// `multilevel_loss(alpha)(logits, target)` → `[B]`.
pub fn multilevel_loss(logits: &[Tensor], target: f64) -> Result<Tensor> {
    let mut total: Option<Tensor> = None;
    for l in logits {
        // max(x, 0) − x·t + log(1 + e^{−|x|})
        let e = l
            .relu()?
            .sub(&(l * target)?)?
            .add(&l.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?)?;
        let per = if e.rank() > 2 {
            e.flatten_from(1)?.mean(1)?
        } else {
            e.flatten_all()?
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
    pub compute: DType,
}

impl Critic {
    pub fn logits(&self, params: &Tensors, sn: &SpectralState, x: &Tensor) -> Result<Vec<Tensor>> {
        let (maps, emb) = self.trunk.features(x)?;
        head_logits(&self.head, params, sn, &maps, &emb, self.compute)
    }

    /// `D(x, for_G=True)` (eval mode) → `[B]`.
    pub fn generator_term(
        &self,
        params: &Tensors,
        sn: &SpectralState,
        x: &Tensor,
        label: f64,
    ) -> Result<Tensor> {
        multilevel_loss(&self.logits(params, sn, x)?, label)
    }
}
