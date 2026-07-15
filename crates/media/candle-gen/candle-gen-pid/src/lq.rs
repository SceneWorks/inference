//! Sigma-aware LQ adapter (`LQProjection2D` + `SigmaAwareGatePerTokenPerDim`) and the `PidNet`
//! wrapper that injects its controlnet-style gate between the backbone's patch blocks. Faithful port
//! of `pid/_src/networks/lq_projection_2d.py` + `pid_net.py` (the inference subset).
//!
//! Scope: the **latent-only** path every in-scope catalog student uses (`lq_in_channels=0`,
//! `z_to_patch_ratio = (sr_scale·lsdf)/patch_size` → nearest-upsample, `lq_interval=2`). The
//! image branch and the merge path are never exercised by any released catalog checkpoint and are
//! intentionally not ported.
//!
//! candle simplification vs the MLX port: candle conv2d/group-norm are native **NCHW**, so the adapter
//! stays in the checkpoint's NCHW layout — no NHWC transpose, and the conv weights load `[out,in,kH,kW]`
//! as-is (the MLX port transposed to `[out,kH,kW,in]`). Runs f32 throughout.

use candle_gen::candle_core::Tensor;
use candle_gen::candle_nn::ops::sigmoid;
use candle_gen::candle_nn::{Conv2d, Conv2dConfig, GroupNorm, Linear, Module};
use candle_gen::{Result, Weights};

use crate::backbone::{PatchInjector, PixDiT};
use crate::config::{ConvPadding, PidConfig};
use crate::nn::linear;

const GN_EPS: f64 = 1e-5; // torch nn.GroupNorm default
const GN_GROUPS: usize = 4; // ResBlock default num_groups

/// A same-padding (stride 1) NCHW conv with a selectable padding mode. `Zeros` lets candle's `Conv2d`
/// pad; `Replicate` (PiD v1.5) edge-pads H/W via `pad_with_same` then convs `valid` — candle's `Conv2d`
/// only zero-pads, mirroring `nn.Conv2d(padding_mode="replicate")`.
struct Conv {
    conv: Conv2d,
    /// `Some(p)` → replicate-pad H/W by `p` before a padding-0 conv; `None` → the conv zero-pads.
    replicate_pad: Option<usize>,
}

impl Conv {
    fn from_weights(w: &Weights, prefix: &str, padding: usize, mode: ConvPadding) -> Result<Self> {
        let weight = w.require(&format!("{prefix}.weight"))?;
        let bias_key = format!("{prefix}.bias");
        let bias = if w.contains(&bias_key) {
            Some(w.require(&bias_key)?)
        } else {
            None
        };
        let (conv_pad, replicate_pad) = match mode {
            ConvPadding::Zeros => (padding, None),
            ConvPadding::Replicate => (0, Some(padding)),
        };
        let cfg = Conv2dConfig {
            padding: conv_pad,
            ..Default::default()
        };
        Ok(Self {
            conv: Conv2d::new(weight, bias, cfg),
            replicate_pad,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = match self.replicate_pad {
            // NCHW axes 2=H, 3=W.
            Some(p) => x.pad_with_same(2, p, p)?.pad_with_same(3, p, p)?,
            None => x.clone(),
        };
        Ok(self.conv.forward(&x)?)
    }
}

/// Per-channel GroupNorm (NCHW), 4 groups, eps 1e-5.
fn group_norm(w: &Weights, prefix: &str, channels: usize) -> Result<GroupNorm> {
    let weight = w.require(&format!("{prefix}.weight"))?;
    let bias = w.require(&format!("{prefix}.bias"))?;
    Ok(GroupNorm::new(weight, bias, channels, GN_GROUPS, GN_EPS)?)
}

/// Pre-activation residual block: `x + Conv(SiLU(GN(Conv(SiLU(GN(x))))))`. Indices match the torch
/// `nn.Sequential` (0 GN, 2 Conv, 3 GN, 5 Conv).
struct ResBlock {
    gn0: GroupNorm,
    conv2: Conv,
    gn3: GroupNorm,
    conv5: Conv,
}

impl ResBlock {
    fn from_weights(w: &Weights, prefix: &str, channels: usize, mode: ConvPadding) -> Result<Self> {
        Ok(Self {
            gn0: group_norm(w, &format!("{prefix}.block.0"), channels)?,
            conv2: Conv::from_weights(w, &format!("{prefix}.block.2"), 1, mode)?,
            gn3: group_norm(w, &format!("{prefix}.block.3"), channels)?,
            conv5: Conv::from_weights(w, &format!("{prefix}.block.5"), 1, mode)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv2.forward(&self.gn0.forward(x)?.silu()?)?;
        let h = self.conv5.forward(&self.gn3.forward(&h)?.silu()?)?;
        Ok((x + h)?)
    }
}

/// `Conv(in→hidden) → SiLU → Conv(hidden→hidden) → ResBlock×N` over NCHW.
struct ConvStack {
    conv0: Conv,
    conv2: Conv,
    res: Vec<ResBlock>,
}

impl ConvStack {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        hidden: usize,
        num_res_blocks: i32,
        mode: ConvPadding,
    ) -> Result<Self> {
        Ok(Self {
            conv0: Conv::from_weights(w, &format!("{prefix}.0"), 1, mode)?,
            conv2: Conv::from_weights(w, &format!("{prefix}.2"), 1, mode)?,
            res: (0..num_res_blocks)
                .map(|i| ResBlock::from_weights(w, &format!("{prefix}.{}", i + 3), hidden, mode))
                .collect::<Result<_>>()?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv2.forward(&self.conv0.forward(x)?.silu()?)?;
        for rb in &self.res {
            x = rb.forward(&x)?;
        }
        Ok(x)
    }
}

/// Sigma-aware LQ gate: `out = x + sigmoid(content_proj([x;lq]) − exp(log_alpha)·σ)·lq`. One impl covers
/// both released variants — the difference is the loaded `content_proj` output width, which
/// `broadcast_mul`s against `lq`: v1 per-dim (`content_proj` `[D,2D]` → gate `[B,N,D]`) and v1.5
/// per-token scalar (`content_proj` `[1,2D]` → gate `[B,N,1]`, broadcast across `D`).
struct SigmaGate {
    content_proj: Linear,
    log_alpha: Tensor,
}

impl SigmaGate {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            content_proj: linear(w, &format!("{prefix}.content_proj"))?,
            log_alpha: w.require(&format!("{prefix}.log_alpha"))?,
        })
    }

    /// `x`, `lq`: `[B, N, D]`; `sigma`: `[B]`.
    fn forward(&self, x: &Tensor, lq: &Tensor, sigma: &Tensor) -> Result<Tensor> {
        let cat = Tensor::cat(&[x, lq], candle_gen::candle_core::D::Minus1)?;
        let logit = self.content_proj.forward(&cat)?; // [B,N,D]
        let b = sigma.dim(0)?;
        // exp(log_alpha)·σ  →  [D] · [B,1,1] = [B,1,D] (broadcast)
        let alpha = self.log_alpha.exp()?;
        let sigma_off = sigma.reshape((b, 1, 1))?.broadcast_mul(&alpha)?;
        let gate = sigmoid(&logit.broadcast_sub(&sigma_off)?)?;
        // broadcast_mul (not mul): v1.5's scalar gate is [B,N,1] and must broadcast across lq's D.
        Ok((x + gate.broadcast_mul(lq)?)?)
    }
}

/// Infer the LQ un-patchify factor `f` from the shape gap between the registry's advertised LQ latent
/// channel count and the first conv's actual input width: `latent_channels = proj_in · f²`. Returns 1
/// when they match (every student except flux2 v1.5). Errors if the ratio isn't a perfect square.
fn infer_unpatchify(latent_channels: i32, proj_in: i32) -> Result<i32> {
    if proj_in == latent_channels {
        return Ok(1);
    }
    if proj_in <= 0 || latent_channels % proj_in != 0 {
        return Err(candle_gen::CandleError::Msg(format!(
            "pid: LQ latent channels ({latent_channels}) not a square multiple of the conv input ({proj_in})"
        )));
    }
    let sq = latent_channels / proj_in;
    let f = (f64::from(sq)).sqrt().round() as i32;
    if f * f != sq {
        return Err(candle_gen::CandleError::Msg(format!(
            "pid: LQ un-patchify ratio {sq} (= {latent_channels}/{proj_in}) is not a perfect square"
        )));
    }
    Ok(f)
}

/// Un-patchify a packed NCHW latent: `[B, C, H, W] → [B, C/f², H·f, W·f]` (the reference's
/// `LQProjection2D._unpatchify_latent_if_needed`, factor `f`). No BN inverse-norm — a pure reshuffle.
fn unpatchify_nchw(x: &Tensor, f: i32) -> Result<Tensor> {
    let (b, c, h, wd) = x.dims4()?;
    let f = f as usize;
    let cc = c / (f * f);
    Ok(x.reshape((b, cc, f, f, h, wd))?
        .permute((0, 1, 4, 2, 5, 3))?
        .contiguous()?
        .reshape((b, cc, h * f, wd * f))?)
}

/// `LQProjection2D` (latent-only): nearest-upsample the latent to the patch grid, run the conv stack,
/// then project to `num_outputs` per-block token feature sets; plus the per-block sigma gates.
pub struct LqAdapter {
    latent_proj: ConvStack,
    output_heads: Vec<Linear>,
    /// PiD v1.5 only (`lq_proj.pit_head`): projects the shared LQ tokens for the PiT-stream injection.
    pit_head: Option<Linear>,
    gates: Vec<SigmaGate>,
    interval: i32,
    /// Channel-unpatchify factor for a packed latent (`lq_projection_2d.py::latent_unpatchify_factor`):
    /// flux2 **v1.5** feeds the 128-ch packed latent but its first conv takes **32**, so the adapter must
    /// un-patchify `[B,128,H,W] → [B,32,2H,2W]` first. `1` for every other student (flux2 v1.0 feeds 128
    /// directly; flux/qwen are 16→16).
    unpatchify_factor: i32,
    upsample_ratio: i32,
}

impl LqAdapter {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &PidConfig) -> Result<Self> {
        let num_outputs = cfg.num_lq_outputs();
        // Infer the un-patchify factor from the shape gap (weights are the truth, like the version
        // sniff): f = isqrt(lq_latent_channels / proj_in). f>1 only for flux2 v1.5.
        let proj_in = w
            .require(&format!("{prefix}.latent_proj.0.weight"))?
            .dim(1)? as i32;
        let unpatchify_factor = infer_unpatchify(cfg.lq_latent_channels, proj_in)?;
        // After un-patchify the latent grid is `f×` finer, so the upsample to the patch grid drops by f.
        let z_to_patch =
            (cfg.sr_scale * cfg.latent_spatial_down_factor) / (cfg.patch_size * unpatchify_factor);
        Ok(Self {
            latent_proj: ConvStack::from_weights(
                w,
                &format!("{prefix}.latent_proj"),
                cfg.lq_hidden_dim as usize,
                cfg.lq_num_res_blocks,
                cfg.lq_conv_padding,
            )?,
            output_heads: (0..num_outputs)
                .map(|i| linear(w, &format!("{prefix}.output_heads.{i}")))
                .collect::<Result<_>>()?,
            pit_head: cfg
                .pit_lq_inject
                .then(|| linear(w, &format!("{prefix}.pit_head")))
                .transpose()?,
            gates: (0..num_outputs)
                .map(|i| SigmaGate::from_weights(w, &format!("{prefix}.gate_modules.{i}")))
                .collect::<Result<_>>()?,
            interval: cfg.lq_interval,
            unpatchify_factor,
            upsample_ratio: z_to_patch.max(1),
        })
    }

    /// Project an LQ latent `[B, z_dim, zH, zW]` to `num_outputs` per-patch-block token feature sets
    /// `[B, N, out_dim]` (`N = pH·pW`), plus (v1.5) the single PiT-stream feature from `pit_head`.
    /// Stays NCHW through the conv stack, then flattens to tokens.
    pub fn forward(
        &self,
        lq_latent: &Tensor,
        _p_h: i32,
        _p_w: i32,
    ) -> Result<(Vec<Tensor>, Option<Tensor>)> {
        let (b, _c, _zh, _zw) = lq_latent.dims4()?;
        // flux2 v1.5: un-patchify the packed latent (128→32 ch, spatial ×f) before the conv stack.
        let mut x = if self.unpatchify_factor > 1 {
            unpatchify_nchw(lq_latent, self.unpatchify_factor)?
        } else {
            lq_latent.clone()
        };
        if self.upsample_ratio > 1 {
            let (_b, _c, h, wd) = x.dims4()?;
            let r = self.upsample_ratio as usize;
            x = x.upsample_nearest2d(h * r, wd * r)?;
        }
        let x = self.latent_proj.forward(&x)?; // [B, hidden, pH, pW]
        let (_b, hidden, ph, pw) = x.dims4()?;
        // NCHW -> NHWC -> [B, pH·pW, hidden]
        let tokens = x
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .reshape((b, ph * pw, hidden))?;
        let feats = self
            .output_heads
            .iter()
            .map(|h| Ok(h.forward(&tokens)?))
            .collect::<Result<Vec<_>>>()?;
        let pit = self
            .pit_head
            .as_ref()
            .map(|h| h.forward(&tokens))
            .transpose()?;
        Ok((feats, pit))
    }

    /// Whether the gate fires at this patch-block index (`interval>1` → every `interval`-th block).
    pub fn is_gate_active(&self, block_idx: i32) -> bool {
        self.interval <= 1 || block_idx % self.interval == 0
    }

    /// Map a patch-block index to its output-head / gate index.
    pub fn output_index(&self, block_idx: i32) -> i32 {
        if self.interval > 1 {
            block_idx / self.interval
        } else {
            block_idx
        }
    }

    /// Apply the `out_idx`-th sigma-aware gate: `x + sigmoid(content_proj([x;lq]) − α·σ)·lq`.
    pub fn gate(&self, out_idx: usize, x: &Tensor, lq: &Tensor, sigma: &Tensor) -> Result<Tensor> {
        self.gates[out_idx].forward(x, lq, sigma)
    }
}

/// `PidNet` — the backbone plus the LQ adapter wired as a between-blocks gate injector.
pub struct PidNet {
    backbone: PixDiT,
    lq: LqAdapter,
    /// PiD v1.5 only (top-level `pit_lq_gate`): gates the pixel-stream conditioning before the PiT
    /// blocks, fed by [`LqAdapter::pit_head`]. `None` for base students.
    pit_lq_gate: Option<SigmaGate>,
    patch_size: i32,
}

impl PidNet {
    /// `prefix` is `""` for a bare-key fixture or `"net."` for the released checkpoint's nesting.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &PidConfig) -> Result<Self> {
        Ok(Self {
            backbone: PixDiT::from_weights(w, prefix, cfg)?,
            lq: LqAdapter::from_weights(w, &format!("{prefix}lq_proj"), cfg)?,
            pit_lq_gate: cfg
                .pit_lq_inject
                .then(|| SigmaGate::from_weights(w, &format!("{prefix}pit_lq_gate")))
                .transpose()?,
            patch_size: cfg.patch_size,
        })
    }

    /// `x`: `[B, 3, H, W]`; `t`: `[B]`; `y`: caption embeddings `[B, Ltxt, txt_embed_dim]`;
    /// `lq_latent`: `[B, z_dim, zH, zW]`; `sigma`: per-sample LQ noise level `[B]`.
    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        y: &Tensor,
        lq_latent: &Tensor,
        sigma: &Tensor,
    ) -> Result<Tensor> {
        let (_b, _c, h, w) = x.dims4()?;
        let (p_h, p_w) = (h as i32 / self.patch_size, w as i32 / self.patch_size);
        let (feats, pit_feat) = self.lq.forward(lq_latent, p_h, p_w)?;
        let inj = LqInjection {
            lq: &self.lq,
            feats,
            sigma: sigma.clone(),
            pit_gate: self.pit_lq_gate.as_ref(),
            pit_feat,
        };
        self.backbone.forward_with(x, t, y, &inj)
    }

    /// Access the LQ adapter (e.g. to parity-test its projection in isolation).
    pub fn lq(&self) -> &LqAdapter {
        &self.lq
    }
}

/// Binds the LQ adapter + this generation's precomputed features + sigma into the backbone's injection
/// hooks: the per-patch-block gate (`inject`) and (v1.5) the PiT pixel-stream gate (`inject_pit`).
struct LqInjection<'a> {
    lq: &'a LqAdapter,
    feats: Vec<Tensor>,
    sigma: Tensor,
    /// PiD v1.5 only: the top-level `pit_lq_gate` + its precomputed feature. `None`/`None` for base.
    pit_gate: Option<&'a SigmaGate>,
    pit_feat: Option<Tensor>,
}

impl PatchInjector for LqInjection<'_> {
    fn inject(&self, block_idx: i32, s_main: &Tensor) -> Result<Tensor> {
        if self.lq.is_gate_active(block_idx) {
            let out_idx = self.lq.output_index(block_idx) as usize;
            if out_idx < self.feats.len() {
                return self
                    .lq
                    .gate(out_idx, s_main, &self.feats[out_idx], &self.sigma);
            }
        }
        Ok(s_main.clone())
    }

    /// PiD v1.5: gate the pixel-stream conditioning `s = silu(s_main + t_emb)` with `pit_lq_gate` before
    /// the PiT blocks. No-op for base students (no `pit_gate`/`pit_feat`).
    fn inject_pit(&self, s: &Tensor) -> Result<Tensor> {
        match (self.pit_gate, &self.pit_feat) {
            (Some(gate), Some(feat)) => gate.forward(s, feat, &self.sigma),
            _ => Ok(s.clone()),
        }
    }
}
