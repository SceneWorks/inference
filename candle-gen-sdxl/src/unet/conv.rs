//! The only `candle-transformers::models::with_tracing` items the vendored SDXL UNet stack uses are
//! `Conv2d` + `conv2d` (sc-5165). They are a thin tracing wrapper over `candle_nn::Conv2d`; we vendor
//! a tracing-free equivalent so this crate need not depend on `candle-transformers`' internals. The
//! computation is identical to `candle_nn::Conv2d` (tracing spans are observation-only).
//!
//! sc-11682: the wrapper now also carries a forward-time **additive conv-LoRA residual** channel so a
//! conv-layer LoRA rides the UNet convs WITHOUT folding into the weight — the base `Conv2d` stays a
//! pristine (unmutated) mmap, which epic-10765 offload/eviction can drop-and-restore cheaply (a folded
//! weight is not disk-re-derivable → pinned). With no residual installed the forward is byte-identical
//! to before, so the plain / dense-fold paths are unchanged.
use candle_core::{Result, Tensor};
use candle_nn::{Conv2dConfig, Module};

/// A forward-time additive conv-LoRA residual (sc-11682): `scale · conv(conv(x, down, base_cfg), up,
/// 1×1)`, the two-conv deferred form of the fused delta [`candle_gen::train::lora::conv_lora_delta`]
/// folds. `down` `[rank, in, kH, kW]` runs at the base conv's config (so its output spatial matches the
/// base's — a stride-2 downsampler residual downsamples too), `up` `[out, rank, 1, 1]` is a 1×1 channel
/// mix; the `[out,in,kH,kW]` delta is never formed and the base weight is never touched. Factors are
/// held f32 and cast to the activation dtype per forward.
#[derive(Debug, Clone)]
struct ConvResidual {
    down: Tensor,
    up: Tensor,
    scale: f64,
}

#[derive(Debug, Clone)]
pub struct Conv2d {
    inner: candle_nn::Conv2d,
    /// The base conv's config — the additive `down` leg runs at this config so its output spatial size
    /// matches `inner`'s.
    cfg: Conv2dConfig,
    /// The dotted module path (`vs.prefix()` at construction, e.g. `down_blocks.0.resnets.0.conv1`) — a
    /// conv-LoRA install matches its resolved target against this.
    path: String,
    /// Forward-time additive conv-LoRA residuals, applied in push order after the base. Empty on the
    /// plain / dense-fold path ⇒ forward is byte-identical to the bare conv.
    residuals: Vec<ConvResidual>,
}

impl Conv2d {
    /// The dotted module path captured from the `VarBuilder` prefix — drives conv-LoRA target matching.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Whether any additive conv-LoRA residual is attached (sc-11682).
    pub fn has_additive(&self) -> bool {
        !self.residuals.is_empty()
    }

    /// The base conv's `[out_channels, in_channels, kH, kW]` weight shape — a conv-LoRA factor is
    /// shape-checked against it before install.
    pub fn weight_dims(&self) -> Vec<usize> {
        self.inner.weight().dims().to_vec()
    }

    /// Attach a forward-time additive conv-LoRA residual `scale · conv(conv(x, down), up)` (sc-11682).
    /// `down` `[rank, in, kH, kW]`, `up` `[out, rank, 1, 1]`; valid on the frozen mmap base (never
    /// mutated). Multiple pushes stack.
    pub fn push_additive_conv(&mut self, down: Tensor, up: Tensor, scale: f64) {
        self.residuals.push(ConvResidual { down, up, scale });
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = self.inner.forward(x)?;
        if self.residuals.is_empty() {
            return Ok(y);
        }
        let xd = x.dtype();
        // 1×1 config for the `up` leg (pure channel mix, no spatial): stride 1, no padding.
        let up_cfg = Conv2dConfig {
            stride: 1,
            padding: 0,
            groups: 1,
            dilation: 1,
            ..Default::default()
        };
        for r in &self.residuals {
            let down = candle_nn::Conv2d::new(r.down.to_dtype(xd)?, None, self.cfg);
            let up = candle_nn::Conv2d::new(r.up.to_dtype(xd)?, None, up_cfg);
            let res = up.forward(&down.forward(x)?)?;
            y = (y + (res * r.scale)?)?;
        }
        Ok(y)
    }
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv2dConfig,
    vs: candle_nn::VarBuilder,
) -> Result<Conv2d> {
    let path = vs.prefix();
    let inner = candle_nn::conv2d(in_channels, out_channels, kernel_size, cfg, vs)?;
    Ok(Conv2d {
        inner,
        cfg,
        path,
        residuals: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_gen::train::lora::conv_lora_delta;

    /// sc-11682: the forward-time additive conv-LoRA residual `scale·conv(conv(x, down), up)` reproduces
    /// the **folded** conv forward `conv(x, W + δ)` with `δ = conv_lora_delta(down, up, alpha, rank,
    /// scale)` on a dense 3×3 conv base — the additive == folded identity (the conv analog of the Linear
    /// parity, tight in f32). This is what lets the dense conv base stay an unmutated mmap (residual) yet
    /// render exactly like the fold.
    #[test]
    fn additive_conv_matches_folded() {
        let dev = Device::Cpu;
        let (out_c, in_c, rank, k) = (6usize, 4usize, 2usize, 3usize);
        let cfg = Conv2dConfig {
            stride: 1,
            padding: 1,
            groups: 1,
            dilation: 1,
            ..Default::default()
        };
        let base_w = Tensor::randn(0f32, 1f32, (out_c, in_c, k, k), &dev).unwrap();
        let down = Tensor::randn(0f32, 1f32, (rank, in_c, k, k), &dev).unwrap(); // [rank, in, kH, kW]
        let up = Tensor::randn(0f32, 1f32, (out_c, rank, 1, 1), &dev).unwrap(); // [out, rank, 1, 1]
        let (alpha, user_scale) = (4.0f32, 0.7f32); // ratio = alpha/rank = 2.0 ⇒ full = 1.4
        let full = (alpha as f64 / rank as f64) * user_scale as f64;

        // Additive: base conv + the residual at the full baked scale.
        let mut c = Conv2d {
            inner: candle_nn::Conv2d::new(base_w.clone(), None, cfg),
            cfg,
            path: String::new(),
            residuals: Vec::new(),
        };
        c.push_additive_conv(down.clone(), up.clone(), full);
        assert!(c.has_additive());

        // Folded reference: δ = conv_lora_delta(...); W_merged = W + δ.
        let delta = conv_lora_delta(&down, &up, alpha, rank as f32, user_scale).unwrap();
        let folded = candle_nn::Conv2d::new((base_w + delta).unwrap(), None, cfg);

        let x = Tensor::randn(0f32, 1f32, (1usize, in_c, 8, 8), &dev).unwrap();
        let dmax = (c.forward(&x).unwrap() - folded.forward(&x).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            dmax < 1e-4,
            "additive conv vs folded conv deviates by {dmax}"
        );

        // A scale-0 residual is an exact no-op vs the bare base conv.
        let mut z = Conv2d {
            inner: candle_nn::Conv2d::new(
                Tensor::randn(0f32, 1f32, (out_c, in_c, k, k), &dev).unwrap(),
                None,
                cfg,
            ),
            cfg,
            path: String::new(),
            residuals: Vec::new(),
        };
        let bare = z.inner.clone();
        z.push_additive_conv(down, up, 0.0);
        let z0 = (z.forward(&x).unwrap() - bare.forward(&x).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(z0, 0.0, "a scale-0 conv residual must be an exact no-op");
    }
}
