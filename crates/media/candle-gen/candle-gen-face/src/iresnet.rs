//! ArcFace recognition embedding — the candle port of antelopev2 `glintr100` (iresnet100), sibling
//! of mlx-gen-face's `iresnet.rs`. The fidelity-critical core of the face stack: PuLID-FLUX and
//! InstantID were trained on *this exact checkpoint's* 512-d embeddings, so the port reproduces the
//! onnx output numerically (embedding cosine ≈ 1.0 vs the canonical pure-math reference).
//!
//! Architecture: `Conv(3→64)+PReLU` stem → layers `[3,13,30,3]` of IBasicBlock (`bn1 → conv1 →
//! PReLU → conv2(stride) → +downsample-identity`) → `bn2 → flatten → Linear(25088→512) → features`.
//! Every Conv carries a folded-BN bias; the pre-activation BNs (`bn1`/`bn2`/`features`) are folded to
//! per-channel affine at conversion. Runs f32.
//!
//! **Layout note:** MLX runs NHWC and transposes NHWC→NCHW before the head flatten to match the onnx
//! `Flatten`. candle is already NCHW, so the flatten is a plain channel-major `reshape` — no transpose.

use candle_gen::candle_core::Tensor;
use candle_gen::Result;

use crate::common::{Conv, Weights};

/// iresnet100 block counts per layer (`layer1..layer4`) — antelopev2 `glintr100`.
pub const IRESNET100_LAYERS: [usize; 4] = [3, 13, 30, 3];
/// iresnet50 block counts per layer — buffalo_l `w600k_r50`, the upstream ai-toolkit-perceptual
/// identity-loss checkpoint (sc-24831).
pub const IRESNET50_LAYERS: [usize; 4] = [3, 4, 14, 3];

/// The per-layer block counts a converted checkpoint carries, read from its keys
/// (`layer{l}.{b}.conv1.weight`) — the twin of mlx-gen-face's `infer_layers` (sc-24831).
pub fn infer_layers(has_key: impl Fn(&str) -> bool) -> Result<[usize; 4]> {
    let mut layers = [0usize; 4];
    for (li, n) in layers.iter_mut().enumerate() {
        while has_key(&format!("layer{}.{}.conv1.weight", li + 1, *n)) {
            *n += 1;
        }
        if *n == 0 {
            return Err(candle_gen::CandleError::Msg(format!(
                "ArcFace checkpoint has no layer{l} blocks (missing layer{l}.0.conv1.weight)",
                l = li + 1
            )));
        }
    }
    Ok(layers)
}

/// PReLU with a per-channel `slope` (`[1,C,1,1]`, broadcast over an NCHW map):
/// `max(x,0) + slope · min(x,0)`. `min(x,0) = x − relu(x)`, avoiding a scalar-min op.
fn prelu(x: &Tensor, slope: &Tensor) -> Result<Tensor> {
    let pos = x.relu()?;
    let neg = x.broadcast_sub(&pos)?; // min(x, 0)
    Ok(pos.broadcast_add(&neg.broadcast_mul(slope)?)?)
}

/// A folded BatchNorm (per-channel `scale`/`shift`, stored 1-D `[C]`). `forward_spatial` broadcasts
/// over an NCHW map (`[1,C,1,1]`); `forward_vec` broadcasts over a `[N,C]` feature vector (`[1,C]`).
struct Affine {
    scale: Tensor,
    shift: Tensor,
}

impl Affine {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            scale: w.require(&format!("{prefix}.scale"))?,
            shift: w.require(&format!("{prefix}.shift"))?,
        })
    }

    fn forward_spatial(&self, x: &Tensor) -> Result<Tensor> {
        let c = self.scale.elem_count();
        let scale = self.scale.reshape((1, c, 1, 1))?;
        let shift = self.shift.reshape((1, c, 1, 1))?;
        Ok(x.broadcast_mul(&scale)?.broadcast_add(&shift)?)
    }

    fn forward_vec(&self, x: &Tensor) -> Result<Tensor> {
        let c = self.scale.elem_count();
        let scale = self.scale.reshape((1, c))?;
        let shift = self.shift.reshape((1, c))?;
        Ok(x.broadcast_mul(&scale)?.broadcast_add(&shift)?)
    }
}

/// One IBasicBlock: `bn1 → conv1 → prelu → conv2(stride) → + downsample(identity)`.
struct Block {
    bn1: Affine,
    conv1: Conv,
    prelu: Tensor, // [1,C,1,1]
    conv2: Conv,
    stride: usize,
    downsample: Option<Conv>,
}

impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t = self.bn1.forward_spatial(x)?;
        let t = self.conv1.forward(&t, 1, 1)?;
        let t = prelu(&t, &self.prelu)?;
        let t = self.conv2.forward(&t, self.stride, 1)?;
        let identity = match &self.downsample {
            Some(ds) => ds.forward(x, self.stride, 0)?,
            None => x.clone(),
        };
        Ok(t.broadcast_add(&identity)?)
    }
}

/// ArcFace iresnet100 recognition network → 512-d embedding.
pub struct ArcFace {
    stem_conv: Conv,
    stem_prelu: Tensor, // [1,C,1,1]
    layers: Vec<Vec<Block>>,
    bn2: Affine,
    fc_w: Tensor, // `[512, 25088]` Linear weight
    fc_b: Tensor, // `[512]`
    features: Affine,
}

impl ArcFace {
    /// Load from a converted IResNet checkpoint (shared with the MLX path) — any depth: the block
    /// counts are read from the keys ([`infer_layers`]) and every width from the tensors' shapes.
    ///
    /// The epic-2123 identity loss (sc-24831) uses the shipped glintr100 (iresnet100); upstream's
    /// buffalo_l `w600k_r50` (iresnet50) loads here unchanged once converted with
    /// `ARCFACE_ONNX=<w600k_r50.onnx> ARCFACE_LAYERS=3,4,14,3 python3
    /// crates/media/mlx-gen/tools/convert_glintr100.py`.
    pub(crate) fn from_weights(w: &Weights) -> Result<Self> {
        let depth = infer_layers(|k| w.contains(k))?;
        let mut layers = Vec::with_capacity(depth.len());
        for (li, &nb) in depth.iter().enumerate() {
            let l = li + 1;
            let mut blocks = Vec::with_capacity(nb);
            for b in 0..nb {
                let p = format!("layer{l}.{b}");
                let stride = if b == 0 { 2 } else { 1 };
                let downsample = if b == 0 {
                    Some(Conv::load(w, &format!("{p}.downsample"))?)
                } else {
                    None
                };
                blocks.push(Block {
                    bn1: Affine::load(w, &format!("{p}.bn1"))?,
                    conv1: Conv::load(w, &format!("{p}.conv1"))?,
                    prelu: w.require_channel4d(&format!("{p}.prelu.weight"))?,
                    conv2: Conv::load(w, &format!("{p}.conv2"))?,
                    stride,
                    downsample,
                });
            }
            layers.push(blocks);
        }
        Ok(Self {
            stem_conv: Conv::load(w, "stem.conv")?,
            stem_prelu: w.require_channel4d("stem.prelu.weight")?,
            layers,
            bn2: Affine::load(w, "bn2")?,
            fc_w: w.require("fc.weight")?, // [512, 25088], used as a Linear (no OHWI transpose)
            fc_b: w.require("fc.bias")?,
            features: Affine::load(w, "features")?,
        })
    }

    /// Compute the 512-d recognition embedding for a batch of aligned face crops.
    ///
    /// `x`: NCHW `[N, 3, 112, 112]` f32, normalized `(rgb - 127.5) / 127.5`. Returns the raw
    /// `[N, 512]` embedding (un-normalized — L2-normalize at the call site for cosine).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.stem_conv.forward(x, 1, 1)?;
        h = prelu(&h, &self.stem_prelu)?;
        for blocks in &self.layers {
            for blk in blocks {
                h = blk.forward(&h)?;
            }
        }
        // Head: bn2, then flatten in NCHW (channel-major) order — already the onnx `Flatten` layout.
        h = self.bn2.forward_spatial(&h)?;
        let n = h.dim(0)?;
        let flat = h.elem_count() / n;
        h = h.contiguous()?.reshape((n, flat))?;
        // fc Linear: [N,25088] @ [25088,512] + [512].
        let fc_b = self.fc_b.reshape((1, self.fc_b.elem_count()))?;
        h = h.matmul(&self.fc_w.t()?)?.broadcast_add(&fc_b)?;
        self.features.forward_vec(&h)
    }
}
