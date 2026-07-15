//! Sigma-aware LQ adapter (`LQProjection2D` + `SigmaAwareGatePerTokenPerDim`) and the `PidNet`
//! wrapper that injects its controlnet-style gate between the backbone's patch blocks. Faithful port
//! of `pid/_src/networks/lq_projection_2d.py` + `pid_net.py` (the inference subset).
//!
//! Scope: the **latent-only** path every in-scope catalog student uses (`lq_in_channels=0`,
//! `z_to_patch_ratio = (sr_scale·lsdf)/patch_size = 2` → nearest-upsample, `lq_interval=2`). The
//! image branch (`lq_in_channels>0`, PixelUnshuffle + bilinear align) and the merge path are never
//! exercised by any released catalog checkpoint and are intentionally not ported (additive if one
//! ever ships an image-conditioned student); the latent `z_to_patch_ratio<1` fold branch likewise
//! never occurs for the 16-/4-channel catalog spaces.

use mlx_rs::ops::{add, concatenate_axis, exp, multiply, pad, sigmoid, subtract, PadMode};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::{conv2d, group_norm, silu, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::backbone::{PatchInjector, PixDiT};
use crate::config::{ConvPadding, PidConfig};

const GN_EPS: f32 = 1e-5; // torch nn.GroupNorm default
const GN_GROUPS: i32 = 4; // ResBlock default num_groups

/// Load a dense Linear (`[out, in]` weight + optional bias).
fn lin(w: &Weights, prefix: &str) -> Result<AdaptableLinear> {
    let weight = w.require(&format!("{prefix}.weight"))?.clone();
    let bias = w.get(&format!("{prefix}.bias")).cloned();
    Ok(AdaptableLinear::dense(weight, bias))
}

/// A Conv2d that stores its weight in mlx NHWC `[out, kH, kW, in]` (transposed from the torch
/// `[out, in, kH, kW]` at load) and runs over NHWC activations. `padding_mode` selects zero (torch
/// default) vs replicate/edge padding — the latter for the PiD v1.5 students (`lq_projection_2d.py`).
struct Conv2d {
    weight: Array,
    bias: Option<Array>,
    padding: i32,
    padding_mode: ConvPadding,
}

impl Conv2d {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        padding: i32,
        padding_mode: ConvPadding,
    ) -> Result<Self> {
        let weight = w
            .require(&format!("{prefix}.weight"))?
            .transpose_axes(&[0, 2, 3, 1])?; // [out,in,kH,kW] -> [out,kH,kW,in]
        Ok(Self {
            weight,
            bias: w.get(&format!("{prefix}.bias")).cloned(),
            padding,
            padding_mode,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        match self.padding_mode {
            // Zero padding (torch default): let the conv op pad.
            ConvPadding::Zeros => conv2d(x, &self.weight, self.bias.as_ref(), 1, self.padding),
            // Replicate/edge padding (v1.5): edge-pad H/W on the NHWC activation, then conv `valid`
            // (padding 0). NHWC axes 1=H, 2=W. Mirrors `nn.Conv2d(padding_mode="replicate")` and the
            // mlx-gen-mochi VAE's edge-pad precedent.
            ConvPadding::Replicate if self.padding > 0 => {
                let p = self.padding;
                let x = pad(
                    x,
                    &[(0, 0), (p, p), (p, p), (0, 0)][..],
                    None,
                    Some(PadMode::Edge),
                )?;
                conv2d(&x, &self.weight, self.bias.as_ref(), 1, 0)
            }
            ConvPadding::Replicate => conv2d(x, &self.weight, self.bias.as_ref(), 1, 0),
        }
    }
}

/// Per-activation GroupNorm (NHWC).
struct GroupNorm {
    weight: Array,
    bias: Array,
}

impl GroupNorm {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: w.require(&format!("{prefix}.weight"))?.clone(),
            bias: w.require(&format!("{prefix}.bias"))?.clone(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        group_norm(x, &self.weight, &self.bias, GN_GROUPS, GN_EPS)
    }
}

/// Pre-activation residual block: `x + Conv(SiLU(GN(Conv(SiLU(GN(x))))))`. Indices match the torch
/// `nn.Sequential` (0 GN, 2 Conv, 3 GN, 5 Conv).
struct ResBlock {
    gn0: GroupNorm,
    conv2: Conv2d,
    gn3: GroupNorm,
    conv5: Conv2d,
}

impl ResBlock {
    fn from_weights(w: &Weights, prefix: &str, padding_mode: ConvPadding) -> Result<Self> {
        Ok(Self {
            gn0: GroupNorm::from_weights(w, &format!("{prefix}.block.0"))?,
            conv2: Conv2d::from_weights(w, &format!("{prefix}.block.2"), 1, padding_mode)?,
            gn3: GroupNorm::from_weights(w, &format!("{prefix}.block.3"))?,
            conv5: Conv2d::from_weights(w, &format!("{prefix}.block.5"), 1, padding_mode)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.conv2.forward(&silu(&self.gn0.forward(x)?)?)?;
        let h = self.conv5.forward(&silu(&self.gn3.forward(&h)?)?)?;
        Ok(add(x, &h)?)
    }
}

/// `Conv(in→hidden) → SiLU → Conv(hidden→hidden) → ResBlock×N` over NHWC.
struct ConvStack {
    conv0: Conv2d,
    conv2: Conv2d,
    res: Vec<ResBlock>,
}

impl ConvStack {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        num_res_blocks: i32,
        padding_mode: ConvPadding,
    ) -> Result<Self> {
        Ok(Self {
            conv0: Conv2d::from_weights(w, &format!("{prefix}.0"), 1, padding_mode)?,
            conv2: Conv2d::from_weights(w, &format!("{prefix}.2"), 1, padding_mode)?,
            res: (0..num_res_blocks)
                .map(|i| ResBlock::from_weights(w, &format!("{prefix}.{}", i + 3), padding_mode))
                .collect::<Result<_>>()?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = self.conv2.forward(&silu(&self.conv0.forward(x)?)?)?;
        for rb in &self.res {
            x = rb.forward(&x)?;
        }
        Ok(x)
    }
}

/// Sigma-aware LQ gate: `out = x + sigmoid(content_proj([x;lq]) − exp(log_alpha)·σ)·lq`.
///
/// One implementation covers both released variants — the difference is purely the loaded
/// `content_proj` output width, which broadcasts against `lq`:
/// - v1 `SigmaAwarePerTokenAndDim`: `content_proj` is `[D, 2·D]` → gate `[B,N,D]` (per-channel);
/// - v1.5 `SigmaAwarePerToken`: `content_proj` is `[1, 2·D]` → gate `[B,N,1]` (per-token scalar),
///   broadcast-multiplied across the `D` channels of `lq`.
struct SigmaGate {
    content_proj: AdaptableLinear,
    log_alpha: Array,
}

impl SigmaGate {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            content_proj: lin(w, &format!("{prefix}.content_proj"))?,
            log_alpha: w.require(&format!("{prefix}.log_alpha"))?.clone(),
        })
    }

    /// `x`, `lq`: `[B, N, D]`; `sigma`: `[B]`.
    fn forward(&self, x: &Array, lq: &Array, sigma: &Array) -> Result<Array> {
        let logit = self
            .content_proj
            .forward(&concatenate_axis(&[x, lq], -1)?)?; // [B,N,D]
        let b = sigma.shape()[0];
        let sigma_off = multiply(&exp(&self.log_alpha)?, &sigma.reshape(&[b, 1, 1])?)?; // exp(log_alpha)·σ
        let gate = sigmoid(&subtract(&logit, &sigma_off)?)?;
        Ok(add(x, &multiply(&gate, lq)?)?)
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
        return Err(mlx_gen::Error::Msg(format!(
            "pid: LQ latent channels ({latent_channels}) not a square multiple of the conv input ({proj_in})"
        )));
    }
    let sq = latent_channels / proj_in;
    let f = (f64::from(sq)).sqrt().round() as i32;
    if f * f != sq {
        return Err(mlx_gen::Error::Msg(format!(
            "pid: LQ un-patchify ratio {sq} (= {latent_channels}/{proj_in}) is not a perfect square"
        )));
    }
    Ok(f)
}

/// Un-patchify a packed NCHW latent: `[B, C, H, W] → [B, C/f², H·f, W·f]` (the reference's
/// `LQProjection2D._unpatchify_latent_if_needed`, factor `f`). No BN inverse-norm — a pure reshuffle.
fn unpatchify_nchw(x: &Array, f: i32) -> Result<Array> {
    let sh = x.shape();
    let (b, c, h, w) = (sh[0], sh[1], sh[2], sh[3]);
    let cc = c / (f * f);
    Ok(x.reshape(&[b, cc, f, f, h, w])?
        .transpose_axes(&[0, 1, 4, 2, 5, 3])?
        .reshape(&[b, cc, h * f, w * f])?)
}

/// `LQProjection2D` (latent-only): nearest-upsample the latent to the patch grid, run the conv stack,
/// then project to `num_outputs` per-block token feature sets; plus the per-block sigma gates.
pub struct LqAdapter {
    latent_proj: ConvStack,
    output_heads: Vec<AdaptableLinear>,
    /// PiD v1.5 only (`lq_proj.pit_head`): a dedicated head that projects the shared LQ tokens for the
    /// PiT pixel-stream injection. `None` for the base students.
    pit_head: Option<AdaptableLinear>,
    gates: Vec<SigmaGate>,
    interval: i32,
    /// Channel-unpatchify factor for a packed latent (`lq_projection_2d.py::latent_unpatchify_factor`):
    /// the released **flux2 v1.5** student consumes the 128-ch packed latent but its first LQ conv takes
    /// **32** channels, so the adapter must un-patchify `[B,128,H,W] → [B,32,2H,2W]` before the conv.
    /// `1` for every other student (flux/qwen 16→16; flux2 **v1.0** feeds the 128-ch latent directly).
    unpatchify_factor: i32,
    upsample_ratio: i32,
}

impl LqAdapter {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &PidConfig) -> Result<Self> {
        let num_outputs = cfg.num_lq_outputs();
        // Infer the un-patchify factor from the shape gap between the registry's advertised LQ latent
        // channel count and the first conv's actual input width (weights are the truth, same principle as
        // the version sniff): f = isqrt(lq_latent_channels / proj_in). f>1 only for flux2 v1.5.
        let proj_in = w
            .require(&format!("{prefix}.latent_proj.0.weight"))?
            .shape()[1];
        let unpatchify_factor = infer_unpatchify(cfg.lq_latent_channels, proj_in)?;
        // After un-patchify the latent grid is `f×` finer, so the upsample to the patch grid drops by f.
        let z_to_patch =
            (cfg.sr_scale * cfg.latent_spatial_down_factor) / (cfg.patch_size * unpatchify_factor);
        Ok(Self {
            latent_proj: ConvStack::from_weights(
                w,
                &format!("{prefix}.latent_proj"),
                cfg.lq_num_res_blocks,
                cfg.lq_conv_padding,
            )?,
            output_heads: (0..num_outputs)
                .map(|i| lin(w, &format!("{prefix}.output_heads.{i}")))
                .collect::<Result<_>>()?,
            pit_head: cfg
                .pit_lq_inject
                .then(|| lin(w, &format!("{prefix}.pit_head")))
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
    pub fn forward(
        &self,
        lq_latent: &Array,
        p_h: i32,
        p_w: i32,
    ) -> Result<(Vec<Array>, Option<Array>)> {
        let b = lq_latent.shape()[0];
        // flux2 v1.5: un-patchify the packed latent (128→32 ch, spatial ×f) before the conv stack.
        let lq = if self.unpatchify_factor > 1 {
            unpatchify_nchw(lq_latent, self.unpatchify_factor)?
        } else {
            lq_latent.clone()
        };
        let mut x = lq.transpose_axes(&[0, 2, 3, 1])?; // NCHW -> NHWC
        if self.upsample_ratio > 1 {
            x = upsample_nearest(&x, self.upsample_ratio)?;
        }
        let x = self.latent_proj.forward(&x)?; // [B, pH, pW, hidden]
        let hidden = x.shape()[3];
        let tokens = x.reshape(&[b, p_h * p_w, hidden])?;
        let feats = self
            .output_heads
            .iter()
            .map(|h| h.forward(&tokens))
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
    pub fn gate(&self, out_idx: usize, x: &Array, lq: &Array, sigma: &Array) -> Result<Array> {
        self.gates[out_idx].forward(x, lq, sigma)
    }
}

/// `PidNet` — the backbone plus the LQ adapter wired as a between-blocks gate injector.
pub struct PidNet {
    backbone: PixDiT,
    lq: LqAdapter,
    /// PiD v1.5 only (top-level `pit_lq_gate`): the sigma gate applied to the pixel-stream conditioning
    /// (`silu(t_emb + s_main)`) before the PiT blocks, fed by [`LqAdapter::pit_head`]. `None` for base.
    pit_lq_gate: Option<SigmaGate>,
    patch_size: i32,
    lq_latent_channels: i32,
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
            lq_latent_channels: cfg.lq_latent_channels,
        })
    }

    /// The LQ latent-branch channel count this net was built for (`[B, z, zH, zW]` → `z`), so the
    /// decoder can validate a caller-supplied LQ latent's contract before the forward (F-100).
    pub fn lq_latent_channels(&self) -> i32 {
        self.lq_latent_channels
    }

    /// `x`: `[B, 3, H, W]`; `t`: `[B]`; `y`: caption embeddings `[B, Ltxt, txt_embed_dim]`;
    /// `lq_latent`: `[B, z_dim, zH, zW]`; `sigma`: per-sample LQ noise level `[B]`.
    pub fn forward(
        &self,
        x: &Array,
        t: &Array,
        y: &Array,
        lq_latent: &Array,
        sigma: &Array,
    ) -> Result<Array> {
        let sh = x.shape();
        let (p_h, p_w) = (sh[2] / self.patch_size, sh[3] / self.patch_size);
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
    feats: Vec<Array>,
    sigma: Array,
    /// PiD v1.5 only: the top-level `pit_lq_gate` + its precomputed feature. `None`/`None` for base.
    pit_gate: Option<&'a SigmaGate>,
    pit_feat: Option<Array>,
}

impl PatchInjector for LqInjection<'_> {
    fn inject(&self, block_idx: i32, s_main: &Array) -> Result<Array> {
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

    /// PiD v1.5: gate the pixel-stream conditioning `s = silu(t_emb + s_main)` with `pit_lq_gate` before
    /// the PiT blocks (ref `pid_net.py`: `s_cond_tokens = pit_lq_gate(s, pit_lq_feature, σ)`). No-op for
    /// the base students (no `pit_gate`/`pit_feat`).
    fn inject_pit(&self, s: &Array) -> Result<Array> {
        match (self.pit_gate, &self.pit_feat) {
            (Some(gate), Some(feat)) => gate.forward(s, feat, &self.sigma),
            _ => Ok(s.clone()),
        }
    }
}
