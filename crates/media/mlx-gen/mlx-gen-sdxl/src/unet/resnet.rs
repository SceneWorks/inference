//! UNet `ResnetBlock2D` (NHWC): GroupNorm→SiLU→Conv3×3, add the projected time embedding, then
//! GroupNorm→SiLU→Conv3×3, plus a residual (a 1×1-conv-as-Linear shortcut when channels change).
//! Port of the vendored `unet.ResnetBlock2D`. The whole UNet runs NHWC (mlx conv layout), so unlike
//! the Z-Image VAE port there is no per-conv transpose — only the stored weights are transposed at
//! load (NCHW→NHWC).

use mlx_rs::ops::add;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableConv2d, AdaptableHost, AdaptableLinear};
use mlx_gen::nn::{conv2d, group_norm};
use mlx_gen::vae_tiling::{tiled_conv2d_3x3_nhwc, GlobalGroupNorm};

use crate::silu_glue;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Result};

use super::nchw_to_nhwc;

/// UNet GroupNorm: 32 groups, eps 1e-5 (mlx `nn.GroupNorm` default, which the vendored uses).
const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-5;

#[derive(Clone)]
pub struct ResnetBlock2D {
    norm1_w: Array,
    norm1_b: Array,
    /// 3×3 conv (NHWC) — a conv-layer LoRA target (sc-2919).
    conv1: AdaptableConv2d,
    norm2_w: Array,
    norm2_b: Array,
    /// 3×3 conv (NHWC) — a conv-layer LoRA target (sc-2919).
    conv2: AdaptableConv2d,
    /// Time-embedding projection — `Some` for UNet resnets, `None` for VAE resnets (no temb).
    time_emb_proj: Option<AdaptableLinear>,
    /// 1×1-conv-as-Linear residual projection when in≠out channels. A conv-shortcut LoRA (sc-2919)
    /// merges here via a 1×1 `[out,in,1,1]→[out,in]` reshape (kept a Linear so its forward — a
    /// matmul over NHWC channels — is unchanged; switching to `conv2d` would risk a 1-ULP shift on
    /// the chaos-sensitive ancestral sampler).
    shortcut: Option<AdaptableLinear>,
}

impl ResnetBlock2D {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let g = |n: &str| w.require(&format!("{prefix}.{n}")).cloned();
        let shortcut = match w.get(&format!("{prefix}.conv_shortcut.weight")) {
            Some(sw) => {
                // [out, in, 1, 1] → [out, in] Linear over NHWC channels.
                let sh = sw.shape();
                let w2 = sw.reshape(&[sh[0], sh[1]])?;
                let b = w.require(&format!("{prefix}.conv_shortcut.bias"))?.clone();
                Some(AdaptableLinear::dense(w2, Some(b)))
            }
            None => None,
        };
        // VAE resnets carry no time embedding (`temb_channels=None`). The U-Net's `time_emb_proj` is
        // quantized (packed-detect, sc-8746); the VAE resnets never reach this crate's quantize path.
        // Detect presence by the dense `.weight` OR a packed `.scales` (a pre-quantized snapshot has
        // no dense `.weight` key beyond the u32 codes, but always carries `.scales`).
        let has_time_proj = w.get(&format!("{prefix}.time_emb_proj.weight")).is_some()
            || w.get(&format!("{prefix}.time_emb_proj.scales")).is_some();
        let time_emb_proj = if has_time_proj {
            Some(crate::quant::lin(
                w,
                &format!("{prefix}.time_emb_proj"),
                true,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1_w: g("norm1.weight")?,
            norm1_b: g("norm1.bias")?,
            conv1: AdaptableConv2d::new(nchw_to_nhwc(&g("conv1.weight")?)?, Some(g("conv1.bias")?)),
            time_emb_proj,
            norm2_w: g("norm2.weight")?,
            norm2_b: g("norm2.bias")?,
            conv2: AdaptableConv2d::new(nchw_to_nhwc(&g("conv2.weight")?)?, Some(g("conv2.bias")?)),
            shortcut,
        })
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        if let Some(t) = &mut self.time_emb_proj {
            crate::quant::quantize_linear(t, bits)?;
        }
        // `shortcut` is the 1×1 `conv_shortcut` stored as a Linear (for the NHWC channel-matmul
        // forward), not a true Linear — so it stays DENSE like every other conv in the U-Net
        // (`conv_in`/`conv_out`, resnet `conv1`/`conv2`, the up/down samplers; see
        // `UNet2DConditionModel::quantize`). It sits directly on the resnet residual/skip path, so
        // int8 rounding error there injects straight into the residual stream and, at 1024², compounds
        // across the denoise loop into a runaway latent outlier that blows out the VAE → a flat image
        // (sc-3329). The defect is the int8 rounding itself, not a dtype/overflow issue (f32 scales +
        // f32 activations reproduce it identically), and it stayed sub-threshold at ≤512² so sc-2641
        // missed it. The `mlx_sd` reference also stores `conv_shortcut` as `nn.Linear` and quantizes
        // it, so leaving it dense is a deliberate, correct divergence from a reference that shares the
        // bug. (LoRA still merges into `shortcut` as a reshaped conv delta — that path is unaffected.)
        Ok(())
    }

    pub(crate) fn materialize_weights(&self) -> Result<()> {
        mlx_rs::transforms::eval([&self.norm1_w, &self.norm1_b, &self.norm2_w, &self.norm2_b])?;
        self.conv1.materialize_weights()?;
        self.conv2.materialize_weights()?;
        if let Some(projection) = &self.time_emb_proj {
            projection.materialize_weights()?;
        }
        if let Some(shortcut) = &self.shortcut {
            shortcut.materialize_weights()?;
        }
        Ok(())
    }

    /// Cast every dtype-bearing leaf to `dtype` (sc-4941 bf16 training): the GroupNorm weights/biases,
    /// both convs, the time-embedding projection, and the 1×1 shortcut.
    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        super::cast_array(&mut self.norm1_w, dtype)?;
        super::cast_array(&mut self.norm1_b, dtype)?;
        super::cast_array(&mut self.norm2_w, dtype)?;
        super::cast_array(&mut self.norm2_b, dtype)?;
        self.conv1.cast_weights(dtype)?;
        self.conv2.cast_weights(dtype)?;
        if let Some(t) = &mut self.time_emb_proj {
            t.cast_weights(dtype)?;
        }
        if let Some(sc) = &mut self.shortcut {
            sc.cast_weights(dtype)?;
        }
        Ok(())
    }

    /// `x`: NHWC `[B, H, W, in]`; `temb`: `Some([B, temb_dim])` for the UNet, `None` for the VAE.
    pub fn forward(&self, x: &Array, temb: Option<&Array>) -> Result<Array> {
        self.forward_planned(x, temb, crate::plan::SdxlForwardPlan::UNBOUNDED)
    }

    /// [`forward`](Self::forward) under a U-Net forward plan. Only the plan's
    /// [`cfg_row`](crate::plan::SdxlForwardPlan::cfg_row) reaches a resnet: a sequential CFG half
    /// hands in the guided pair's full-batch `temb`, projects it at that batch, and keeps its own
    /// row (sc-26268).
    pub fn forward_planned(
        &self,
        x: &Array,
        temb: Option<&Array>,
        plan: crate::plan::SdxlForwardPlan<'_>,
    ) -> Result<Array> {
        self.forward_core(x, temb, plan, false)
    }

    /// Dense VAE inference with evaluation boundaries between convolutions. No spatial slicing or
    /// arithmetic change; unlike the lazy/autograd forward this releases each convolution's
    /// transients before constructing the next one.
    pub(crate) fn forward_evaluated_vae(&self, x: &Array) -> Result<Array> {
        self.forward_core(x, None, crate::plan::SdxlForwardPlan::UNBOUNDED, true)
    }

    fn forward_core(
        &self,
        x: &Array,
        temb: Option<&Array>,
        plan: crate::plan::SdxlForwardPlan<'_>,
        evaluate: bool,
    ) -> Result<Array> {
        let y = group_norm(x, &self.norm1_w, &self.norm1_b, GN_GROUPS, GN_EPS)?;
        let mut y = conv2d(
            &silu_glue(&y)?,
            self.conv1.weight(),
            self.conv1.bias(),
            1,
            1,
        )?;
        if evaluate {
            y.eval()?;
        }
        // Add the projected time embedding (UNet only), broadcast over H,W.
        if let (Some(proj), Some(t)) = (&self.time_emb_proj, temb) {
            let tp = plan.select_cfg_row(proj.forward(&silu_glue(t)?)?, x.shape()[0])?;
            let tb = tp.shape();
            y = add(&y, &tp.reshape(&[tb[0], 1, 1, tb[1]])?)?;
        }
        let y = group_norm(&y, &self.norm2_w, &self.norm2_b, GN_GROUPS, GN_EPS)?;
        let y = conv2d(
            &silu_glue(&y)?,
            self.conv2.weight(),
            self.conv2.bias(),
            1,
            1,
        )?;

        if evaluate {
            y.eval()?;
        }
        let residual = match &self.shortcut {
            Some(sc) => sc.forward(x)?,
            None => x.clone(),
        };
        let out = add(&residual, &y)?;
        if evaluate {
            out.eval()?;
        }
        Ok(out)
    }

    /// iOS quantum scratch (sc-26270) for this resnet over its (pre-concat) input `x` and the
    /// time embedding `temb` it adds, counted by [`crate::quantum::resnet_bytes`] from the loaded
    /// `conv1`/`conv2` weights (whose `conv1` input width tells whether the up path's skip
    /// concatenation runs inside the quantum), `x`'s dtype, and the dtype the projected `temb`
    /// promotes the second half to (a packed `time_emb_proj` with BF16 scales, or an f32 `temb`,
    /// makes it f32 — [`crate::quantum::linear_output_dtype`]), on the running backend.
    pub(crate) fn quantum_scratch(&self, x: &Array, temb: Option<&Array>) -> u64 {
        let act_in = crate::quantum::element_bytes(x);
        let act_out = match (&self.time_emb_proj, temb) {
            (Some(proj), Some(t)) => {
                let projected = crate::quantum::linear_output_dtype(proj, t.dtype());
                crate::quantum::dtype_bytes(Dtype::from_promoting_types(x.dtype(), projected))
            }
            _ => act_in,
        };
        crate::quantum::resnet_bytes(
            x.shape(),
            act_in,
            act_out,
            self.conv1.weight(),
            self.conv2.weight(),
            crate::quantum::backend(),
        )
    }

    /// The dtype [`Self::forward_core`] returns for an `input`-typed input (the up path's
    /// concatenation already applied) and a `temb`-typed time embedding: MLX promotes through
    /// every weight, bias and norm affine the forward touches and the projected embedding
    /// ([`crate::quantum::linear_output_dtype`]). Header-only pricing threads it.
    pub(crate) fn output_dtype(&self, input: Dtype, temb: Option<Dtype>) -> Dtype {
        let mut arrays: Vec<&Array> = vec![&self.norm1_w, &self.norm1_b, self.conv1.weight()];
        arrays.extend(self.conv1.bias());
        arrays.extend([&self.norm2_w, &self.norm2_b, self.conv2.weight()]);
        arrays.extend(self.conv2.bias());
        let mut y = crate::quantum::promote_all(input, &arrays);
        if let (Some(proj), Some(t)) = (&self.time_emb_proj, temb) {
            y = Dtype::from_promoting_types(y, crate::quantum::linear_output_dtype(proj, t));
        }
        let residual = match &self.shortcut {
            Some(sc) => {
                Dtype::from_promoting_types(input, crate::quantum::linear_output_dtype(sc, input))
            }
            None => input,
        };
        Dtype::from_promoting_types(residual, y)
    }

    /// The `[cin, cout]` of `conv1` and `cout` of `conv2` (header-only pricing).
    pub(crate) fn conv_weights(&self) -> (&Array, &Array) {
        (self.conv1.weight(), self.conv2.weight())
    }

    /// [`forward_planned`](Self::forward_planned) as one guarded iOS quantum (sc-26270). `skip`
    /// (the up path's popped residual) is concatenated INSIDE the quantum, so the concat is checked
    /// and materialized with the resnet that consumes it. A pass-through off device.
    pub(crate) fn forward_quantum(
        &self,
        x: &Array,
        skip: Option<Array>,
        temb: Option<&Array>,
        plan: crate::plan::SdxlForwardPlan<'_>,
    ) -> Result<Array> {
        let scratch = self.quantum_scratch(x, temb);
        mlx_gen::ios_quantum::run(crate::quantum::UNET_RESNET, scratch, || {
            let input = match skip {
                Some(skip) => mlx_rs::ops::concatenate_axis(&[x, &skip], -1)?,
                None => x.clone(),
            };
            self.forward_planned(&input, temb, plan)
        })
    }

    /// VAE-only, normalization-correct bounded forward (sc-19753).
    ///
    /// The U-Net continues to use [`Self::forward`]. VAE resnets carry no time embedding, allowing
    /// their GroupNorms to use full-image statistics while only the two 3×3 convolutions tile.
    pub(crate) fn forward_tiled_vae(
        &self,
        x: &Array,
        tile_edge: i32,
        cancel: Option<&CancelFlag>,
        labels: crate::quantum::VaeLabels,
    ) -> Result<Array> {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        if self.time_emb_proj.is_some() {
            return Err(mlx_gen::Error::Msg(
                "forward_tiled_vae called on a time-conditioned U-Net resnet".into(),
            ));
        }
        // sc-26270: the tiled layer is one iOS stage quantum (full-resolution input, both conv
        // outputs, shortcut and sum), inside which every conv tile is checked on its own.
        let s = x.shape();
        let (cin, cout) = crate::quantum::conv_channels(self.conv1.weight());
        mlx_gen::ios_quantum::before(
            labels.tiled_stage,
            crate::quantum::tiled_stage_bytes(s, cin, cout, crate::quantum::element_bytes(x), true),
        )?;
        let norm1 = GlobalGroupNorm::new(x, &self.norm1_w, &self.norm1_b, GN_GROUPS, GN_EPS)?;
        let y = tiled_conv2d_3x3_nhwc(
            x,
            self.conv1.weight(),
            self.conv1.bias(),
            tile_edge,
            cancel,
            |tile| {
                // sc-26270: each tile is checked before its norm/conv graph is constructed.
                crate::quantum::before_tile(labels.tile, tile, self.conv1.weight(), false)?;
                silu_glue(&norm1.apply(tile)?)
            },
        )?;
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        crate::vae::trace_decode_stage("resnet.conv1", &y)?;
        let norm2 = GlobalGroupNorm::new(&y, &self.norm2_w, &self.norm2_b, GN_GROUPS, GN_EPS)?;
        let y = tiled_conv2d_3x3_nhwc(
            &y,
            self.conv2.weight(),
            self.conv2.bias(),
            tile_edge,
            cancel,
            |tile| {
                crate::quantum::before_tile(labels.tile, tile, self.conv2.weight(), false)?;
                silu_glue(&norm2.apply(tile)?)
            },
        )?;
        crate::vae::trace_decode_stage("resnet.conv2", &y)?;
        let residual = match &self.shortcut {
            Some(sc) => sc.forward(x)?,
            None => x.clone(),
        };
        let out = add(&residual, &y)?;
        out.eval()?;
        mlx_gen::ios_quantum::finish(labels.tiled_stage, &[&out])?;
        Ok(out)
    }

    /// Header-only pricing of [`Self::forward_tiled_vae`] over the shape-only `x`: the stage and
    /// each convolution's widest tile, with the guards' own formulas. Returns the shape-only output.
    pub(crate) fn price_tiled_vae(
        &self,
        x: &Array,
        tile_edge: i32,
        labels: crate::quantum::VaeLabels,
        out: &mut crate::quantum::Quanta,
    ) -> Result<Array> {
        let s = x.shape();
        let (cin, cout) = crate::quantum::conv_channels(self.conv1.weight());
        out.push((
            labels.tiled_stage,
            crate::quantum::tiled_stage_bytes(s, cin, cout, crate::quantum::element_bytes(x), true),
        ));
        let tile = |input: &[i32], channels: i32, width: u64, weight: &Array| {
            let (th, tw) = crate::vae::max_conv_tile(input[1], input[2], tile_edge);
            crate::quantum::tile_bytes(
                &[input[0], th, tw, channels],
                width,
                weight,
                false,
                crate::quantum::backend(),
            )
        };
        out.push((
            labels.tile,
            tile(
                s,
                s[3],
                crate::quantum::element_bytes(x),
                self.conv1.weight(),
            ),
        ));
        let mut conv1 = vec![&self.norm1_w, &self.norm1_b, self.conv1.weight()];
        conv1.extend(self.conv1.bias());
        let y = crate::quantum::promote_all(x.dtype(), &conv1);
        let mid = [s[0], s[1], s[2], self.conv2.weight().shape()[3]];
        out.push((
            labels.tile,
            tile(
                &mid,
                mid[3],
                crate::quantum::dtype_bytes(y),
                self.conv2.weight(),
            ),
        ));
        crate::quantum::shaped(
            &[s[0], s[1], s[2], cout],
            self.output_dtype(x.dtype(), None),
        )
    }

    /// The one LoRA-targetable **Linear** on a U-Net resnet — `time_emb_proj`. The convs
    /// (`conv1`/`conv2`/`conv_shortcut`) are conv-layer targets, enumerated by
    /// [`conv_target_paths`](Self::conv_target_paths) instead (sc-2919).
    pub fn lora_target_paths(&self, prefix: &str, out: &mut Vec<String>) {
        if self.time_emb_proj.is_some() {
            out.push(format!("{prefix}.time_emb_proj"));
        }
    }

    /// This resnet's conv-layer LoRA targets (sc-2919): `conv1`, `conv2`, and — when present —
    /// `conv_shortcut`. Merged only under [`crate::adapters::LoraCoverage::Complete`]; the
    /// Linear-only vendored coverage drops them (preserving byte-parity with the retired Python path).
    pub fn conv_target_paths(&self, prefix: &str, out: &mut Vec<String>) {
        out.push(format!("{prefix}.conv1"));
        out.push(format!("{prefix}.conv2"));
        if self.shortcut.is_some() {
            out.push(format!("{prefix}.conv_shortcut"));
        }
    }
}

impl AdaptableHost for ResnetBlock2D {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["time_emb_proj"] => self.time_emb_proj.as_mut(),
            // conv_shortcut is a 1×1 conv stored as a Linear; a conv LoRA merges into it as a
            // reshaped 2-D delta (sc-2919, routed by the SDXL adapter's conv path).
            ["conv_shortcut"] => self.shortcut.as_mut(),
            _ => None,
        }
    }

    fn adaptable_conv_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableConv2d> {
        match path {
            ["conv1"] => Some(&mut self.conv1),
            ["conv2"] => Some(&mut self.conv2),
            _ => None,
        }
    }
}
