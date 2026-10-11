//! Backend-neutral **restoration training contract** of Iris-3B (epic sc-25678, story sc-25688):
//! everything about a restoration fine-tuning run that is not a tensor — the request surface, the
//! resolved recipe, the training-only resources (frozen VGG16/LPIPS, the frozen OpenCLIP
//! ConvNeXt-XXL critic trunk), the critic head's architecture and init law, the per-item training
//! pair (synthetic second-order degradation or a validated aligned pair), the generator/critic
//! alternation, the durable checkpoint state and the export identity.
//!
//! ## The recipe and where each piece comes from
//!
//! The frozen Iris release (`speridlabs/iris-3b` @ [`super::UPSTREAM_CODE_REVISION`]) ships the
//! restoration *export* and its inference (`downstream/restoration.py`) but **no training loop**.
//! The recipe is the paper's (arXiv 2610.09450v1 §5.3, Eq. 6), which "follows the one-step recipe
//! of HYPIR"; every choice the paper leaves open is HYPIR's reference trainer
//! (`XPixelGroup/HYPIR` @ [`HYPIR_REVISION`], `HYPIR/trainer/base.py`, `HYPIR/model/D.py`,
//! `HYPIR/model/backbone.py`, `configs/sd2_train.yaml`). The row-by-row trace is the
//! "Restoration training" table of `mlx-gen-iris/UPSTREAM.md`.
//!
//! * **Generator** — the restoration backbone itself, one step: `x̂ = x_lq − σ·v(x_lq, σ·N, ∅)`
//!   on the bicubic-upsampled degraded image (exactly the `iris_3b_restore` forward), trained in
//!   full (the paper) or through a LoRA/LoKr adapter merged at export (HYPIR's own LoRA arm).
//! * **Loss** (Eq. 6): `‖x̂ − y‖₂² + 5·LPIPS_vgg(x̂, y) + 0.5·L_adv(x̂)` — HYPIR's `F.mse_loss`
//!   (mean), `lpips.LPIPS(net="vgg")(x̂, y).mean()`, and the critic's generator term.
//! * **Critic** — a frozen OpenCLIP ConvNeXt-XXL trunk (`convnext_xxlarge`,
//!   `laion2b_s34b_b82k_augreg_soup`) whose stage-1/2/3 maps and pooled embedding feed
//!   `MultiLevelDConv(level=4, in_ch1=[384, 768, 1536], in_ch2=1024, out_ch=512, down=2)` —
//!   spectral-normalized convs with BlurPool, LeakyReLU(0.2) — scored by `multilevel_loss(alpha=0.8)`:
//!   the non-saturating BCE-with-logits GAN with one-sided label smoothing (real/generator target
//!   0.8, fake 0), summed over the four levels.
//! * **Alternation** — HYPIR's `run`: each accumulation window of micro-batches is a generator
//!   window or a critic window, alternating (`(batch_count // accum) % 2`); the critic window
//!   regenerates `x̂` with the current generator under no-grad; one **step** is one generator
//!   window plus one critic window. Gradients are clipped (`max_grad_norm = 1.0`) per network; the EMA
//!   (decay 0.999) is updated after every generator optimizer step.
//! * **Optimizers** — AdamW for both networks, `lr = 1e-5`, `betas = (0.9, 0.999)`, torch's default
//!   `weight_decay = 0.01` and `eps = 1e-8`, constant schedule (paper: "constant LR").

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Map as JsonMap, Value as JsonValue};
use sha2::{Digest, Sha256};

pub use super::degradation::HYPIR_REVISION;
use super::degradation::{
    self, degradation_key, DegradationConfig, KernelConfig, KernelType, Planes, StageConfig,
};
use super::restoration::{bicubic_scale_factor, RestorationResources};
use super::train::{
    mix_seed, select_adapter_targets, HostRng, LrShape, MixedPrecision, ResumeDataPolicy,
    WeightsSelect, META_ARTIFACT, META_BASE, META_STEPS, META_TASK, META_UPSTREAM, META_WEIGHTS,
};
use super::{IrisConfig, IrisTask};
use crate::runtime::{LoadSpec, WeightsSource};
use crate::train::{LrSchedule, NetworkType, TrainingConfig, TrainingItem, TrainingRequest};
use crate::{Error, Result};

/// Registry id of the restoration trainer — the same id as the transform it trains for.
pub const TRAINER_ID: &str = super::restoration::MODEL_ID;
/// `TrainingConfig::model_options` key carrying the restoration options bag.
pub const OPTIONS_KEY: &str = "iris_restore";
/// `TrainingItem::model_options` key naming an item's **aligned degraded input** (a path). Present
/// on every item = paired training; absent on every item = synthetic degradation; a mix is refused.
pub const ITEM_DEGRADED_KEY: &str = "degraded_image";

/// `LoadSpec::components` key of the frozen VGG16 trunk file (torchvision `vgg16-397923af.pth`).
pub const VGG16_COMPONENT: &str = "lpips_vgg16";
/// `LoadSpec::components` key of the LPIPS v0.1 linear-head file (`lpips/weights/v0.1/vgg.pth`).
pub const LPIPS_LIN_COMPONENT: &str = "lpips_lin";
/// `LoadSpec::components` key of the frozen OpenCLIP ConvNeXt-XXL checkpoint
/// (`open_clip_pytorch_model.bin`; only its `visual.*` tensors are read).
pub const CRITIC_TRUNK_COMPONENT: &str = "critic_trunk";
/// Optional `LoadSpec::components` key of a generation backbone directory to start from
/// (`init: "parent"`, the paper's "adapted from the parent checkpoint").
pub const PARENT_COMPONENT: &str = "parent_backbone";

// =============================================================================================
// Training-only resource identities
// =============================================================================================

/// The pinned identity of one training-only weight file. The trainer reads it from the path the
/// caller stages under [`Self::component`]; the caller downloads it from `url` and checks
/// `sha256` (see [`verify_resource`]) — nothing downloads during training.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingResource {
    pub component: &'static str,
    /// What it is.
    pub name: &'static str,
    /// The pinned download URL (an immutable revision).
    pub url: &'static str,
    /// The repository / project and its pinned revision.
    pub repo: &'static str,
    pub revision: &'static str,
    /// The file name inside the revision.
    pub file: &'static str,
    /// sha256 of the file bytes (lowercase hex).
    pub sha256: &'static str,
    pub bytes: u64,
    /// SPDX-ish licence of the artifact.
    pub license: &'static str,
}

/// torchvision's ImageNet VGG16 (`VGG16_Weights.IMAGENET1K_V1`, `vgg16-397923af.pth`), the LPIPS
/// trunk — as timm's pinned safetensors republication `timm/vgg16.tv_in1k`, whose `features.*`
/// tensors are bit-identical to the torchvision file's (checked tensor by tensor when this pin was
/// taken; the extra classifier tensors are never read).
pub const VGG16_RESOURCE: TrainingResource = TrainingResource {
    component: VGG16_COMPONENT,
    name: "VGG16 ImageNet (torchvision IMAGENET1K_V1, timm safetensors)",
    url: "https://huggingface.co/timm/vgg16.tv_in1k/resolve/b8d8aa2dd860af9233c8c67385a8097fd6c35d3f/model.safetensors",
    repo: "timm/vgg16.tv_in1k",
    revision: "b8d8aa2dd860af9233c8c67385a8097fd6c35d3f",
    file: "model.safetensors",
    sha256: "57b026918159a6bf9faf8405c3a551903768e7138989d9c6224a14227203fad8",
    bytes: 553_432_986,
    license: "BSD-3-Clause (torchvision); VGG weights CC-BY-4.0 (Simonyan & Zisserman)",
};

/// LPIPS v0.1 VGG linear heads (`lpips==0.1.4`, the package HYPIR pins) as the pinned safetensors
/// republication `vivym/lpips`: its five `{k}.weight` tensors are bit-identical to the
/// `lin{k}.model.1.weight` tensors of
/// `richzhang/PerceptualSimilarity@082bb24f84c091ea94de2867d34c4544f68e0963:lpips/weights/v0.1/vgg.pth`
/// (checked tensor by tensor when this pin was taken; that file is a legacy, non-zip torch archive).
pub const LPIPS_LIN_RESOURCE: TrainingResource = TrainingResource {
    component: LPIPS_LIN_COMPONENT,
    name: "LPIPS v0.1 VGG linear layers (safetensors)",
    url: "https://huggingface.co/vivym/lpips/resolve/270571f1fb2a2c4c5f920cec96cd87838c345982/vgg_lpips_linear.safetensors",
    repo: "vivym/lpips",
    revision: "270571f1fb2a2c4c5f920cec96cd87838c345982",
    file: "vgg_lpips_linear.safetensors",
    sha256: "0c6387bf2e51e434dedbc0ca4c5894f605c415b8d128bfe720e13456eb4333ac",
    bytes: 6_264,
    license: "BSD-2-Clause",
};

/// The lin-head key of tap `k` in either published layout (`lin{k}.model.1.weight` — the LPIPS
/// package's — or `{k}.weight`, the pinned safetensors').
pub fn lpips_lin_keys(k: usize) -> [String; 2] {
    [format!("lin{k}.model.1.weight"), format!("{k}.weight")]
}

/// The OpenCLIP ConvNeXt-XXL checkpoint HYPIR's critic trunk loads
/// (`open_clip.create_model_and_transforms("convnext_xxlarge", pretrained="laion2b_s34b_b82k_augreg_soup")`).
pub const CRITIC_TRUNK_RESOURCE: TrainingResource = TrainingResource {
    component: CRITIC_TRUNK_COMPONENT,
    name: "OpenCLIP ConvNeXt-XXLarge laion2B-s34B-b82K-augreg-soup",
    url: "https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup/resolve/9f3e8ee3f383c672388d9178afe70af9e63ac9df/open_clip_pytorch_model.bin",
    repo: "laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup",
    revision: "9f3e8ee3f383c672388d9178afe70af9e63ac9df",
    file: "open_clip_pytorch_model.bin",
    sha256: "d571d992c3fc1d9d6f771ad4eb0ae4cb63745d3bd0cf3363ecde5ff249e2f155",
    bytes: 4_802_571_739,
    license: "MIT",
};

/// Every training-only resource the restoration trainer reads.
pub const TRAINING_RESOURCES: [TrainingResource; 3] =
    [VGG16_RESOURCE, LPIPS_LIN_RESOURCE, CRITIC_TRUNK_RESOURCE];

/// sha256 of a file (lowercase hex), streamed.
pub fn sha256_file(path: &Path) -> Result<String> {
    use std::io::Read as _;
    let mut f = std::fs::File::open(path)
        .map_err(|e| Error::Msg(format!("iris restore: open {}: {e}", path.display())))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| Error::Msg(format!("iris restore: read {}: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// The caller's integrity check of a staged resource file (size, then sha256).
pub fn verify_resource(path: &Path, res: &TrainingResource) -> Result<()> {
    let len = std::fs::metadata(path)
        .map_err(|e| Error::Msg(format!("iris restore: stat {}: {e}", path.display())))?
        .len();
    if len != res.bytes {
        return Err(Error::Msg(format!(
            "iris restore: {} is {len} bytes, the pinned {} ({}) is {}",
            path.display(),
            res.name,
            res.revision,
            res.bytes
        )));
    }
    let got = sha256_file(path)?;
    if got != res.sha256 {
        return Err(Error::Msg(format!(
            "iris restore: {} sha256 {got} is not the pinned {} ({})",
            path.display(),
            res.name,
            res.sha256
        )));
    }
    Ok(())
}

// =============================================================================================
// The perceptual network (lpips 0.1.4 `LPIPS(net="vgg", version="0.1")`)
// =============================================================================================

/// `ScalingLayer.shift` / `.scale` (inputs in `[−1, 1]`).
pub const LPIPS_SHIFT: [f32; 3] = [-0.030, -0.088, -0.188];
pub const LPIPS_SCALE: [f32; 3] = [0.458, 0.448, 0.450];
/// `normalize_tensor` eps: `x / (‖x‖₂ over channels + 1e-10)`.
pub const LPIPS_NORM_EPS: f32 = 1e-10;

/// One op of the VGG16 feature stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VggOp {
    /// 3×3 conv, padding 1, bias, then ReLU: `(features index, in, out)`.
    ConvRelu(usize, usize, usize),
    /// 2×2 / stride-2 max-pool.
    MaxPool,
}

/// The VGG16 feature stack split at LPIPS's five taps (`relu1_2, relu2_2, relu3_3, relu4_3,
/// relu5_3` — torchvision `features[0:4] / [4:9] / [9:16] / [16:23] / [23:30]`), for the given
/// channel widths (`[64, 128, 256, 512, 512]` for the real network).
pub fn vgg_slices(widths: [usize; 5]) -> Vec<Vec<VggOp>> {
    let convs = [2usize, 2, 3, 3, 3];
    let mut idx = 0usize;
    let mut cin = 3usize;
    let mut out = Vec::with_capacity(5);
    for (s, (&n, &w)) in convs.iter().zip(&widths).enumerate() {
        let mut ops = Vec::new();
        if s > 0 {
            ops.push(VggOp::MaxPool);
            idx += 1;
        }
        for _ in 0..n {
            ops.push(VggOp::ConvRelu(idx, cin, w));
            cin = w;
            idx += 2;
        }
        out.push(ops);
    }
    out
}

/// The real network's widths.
pub const VGG16_WIDTHS: [usize; 5] = [64, 128, 256, 512, 512];

// =============================================================================================
// The critic (`HYPIR/model/D.py` + `backbone.py`, timm `convnext_xxlarge`)
// =============================================================================================

/// CLIP's input normalization (`image_mean` / `image_std`) applied to `x·0.5 + 0.5` (HYPIR's
/// literals, verbatim).
pub const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
#[allow(clippy::excessive_precision)]
pub const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

/// A timm ConvNeXt trunk (`conv_mlp=False`, 4×4/4 stem, 7×7 depthwise blocks, MLP ratio 4,
/// LayerScale `gamma`) with the OpenCLIP `TimmModel` head (`timm_pool=""` ⇒ the trunk's own
/// avg-pool + LayerNorm head, then `head.proj` Linear without bias).
#[derive(Clone, Debug, PartialEq)]
pub struct ConvNextConfig {
    pub depths: Vec<usize>,
    pub dims: Vec<usize>,
    pub norm_eps: f64,
    pub embed_dim: usize,
}

impl ConvNextConfig {
    /// `convnext_xxlarge` (`depths=[3, 4, 30, 3]`, `dims=[384, 768, 1536, 3072]`,
    /// `norm_eps=1e-5`), CLIP embed 1024.
    pub fn xxlarge() -> Self {
        Self {
            depths: vec![3, 4, 30, 3],
            dims: vec![384, 768, 1536, 3072],
            norm_eps: 1e-5,
            embed_dim: 1024,
        }
    }

    /// Every trunk tensor `(key, shape)` under the OpenCLIP state-dict names.
    pub fn tensor_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let mut out = Vec::new();
        let t = "visual.trunk";
        let d0 = self.dims[0];
        out.push((format!("{t}.stem.0.weight"), vec![d0, 3, 4, 4]));
        out.push((format!("{t}.stem.0.bias"), vec![d0]));
        out.push((format!("{t}.stem.1.weight"), vec![d0]));
        out.push((format!("{t}.stem.1.bias"), vec![d0]));
        for (s, (&depth, &dim)) in self.depths.iter().zip(&self.dims).enumerate() {
            let p = format!("{t}.stages.{s}");
            if s > 0 {
                let prev = self.dims[s - 1];
                out.push((format!("{p}.downsample.0.weight"), vec![prev]));
                out.push((format!("{p}.downsample.0.bias"), vec![prev]));
                out.push((format!("{p}.downsample.1.weight"), vec![dim, prev, 2, 2]));
                out.push((format!("{p}.downsample.1.bias"), vec![dim]));
            }
            for b in 0..depth {
                let q = format!("{p}.blocks.{b}");
                out.push((format!("{q}.gamma"), vec![dim]));
                out.push((format!("{q}.conv_dw.weight"), vec![dim, 1, 7, 7]));
                out.push((format!("{q}.conv_dw.bias"), vec![dim]));
                out.push((format!("{q}.norm.weight"), vec![dim]));
                out.push((format!("{q}.norm.bias"), vec![dim]));
                out.push((format!("{q}.mlp.fc1.weight"), vec![4 * dim, dim]));
                out.push((format!("{q}.mlp.fc1.bias"), vec![4 * dim]));
                out.push((format!("{q}.mlp.fc2.weight"), vec![dim, 4 * dim]));
                out.push((format!("{q}.mlp.fc2.bias"), vec![dim]));
            }
        }
        let last = *self.dims.last().expect("dims");
        out.push((format!("{t}.head.norm.weight"), vec![last]));
        out.push((format!("{t}.head.norm.bias"), vec![last]));
        out.push(("visual.head.proj.weight".into(), vec![self.embed_dim, last]));
        out
    }

    /// The critic head geometry this trunk feeds (`in_ch1` = stage 0–2 widths, `in_ch2` = embed).
    pub fn head(&self, out_ch: usize) -> CriticHeadConfig {
        CriticHeadConfig {
            in_ch1: [self.dims[0], self.dims[1], self.dims[2]],
            in_ch2: self.embed_dim,
            out_ch,
        }
    }
}

/// `MultiLevelDConv(level=4, down=2)` geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CriticHeadConfig {
    pub in_ch1: [usize; 3],
    pub in_ch2: usize,
    pub out_ch: usize,
}

/// HYPIR's head width (`out_ch=512`).
pub const CRITIC_HEAD_WIDTH: usize = 512;
/// `LeakyReLU(0.2)`.
pub const CRITIC_LEAKY_SLOPE: f32 = 0.2;
/// `BlurPool(filt_size=4)`: `[1, 3, 3, 1]ᵀ[1, 3, 3, 1] / 64`.
pub const BLUR_TAPS: [f32; 4] = [1.0, 3.0, 3.0, 1.0];
/// `multilevel_loss(alpha=0.8)`: the real / generator BCE target.
pub const GAN_REAL_LABEL: f64 = 0.8;
/// `torch.nn.utils.spectral_norm` eps (`normalize(·, eps=1e-12)`).
pub const SPECTRAL_EPS: f32 = 1e-12;

/// One spectral-normalized layer of the head: its parameter prefix, weight shape, and whether it
/// is a conv (4-D) or a linear.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnLayer {
    pub prefix: String,
    pub shape: Vec<usize>,
}

impl SnLayer {
    /// `weight_mat` rows / cols (`weight.reshape(out, -1)`).
    pub fn matrix(&self) -> (usize, usize) {
        (self.shape[0], self.shape[1..].iter().product())
    }
}

impl CriticHeadConfig {
    /// Every spectral-normalized layer, HYPIR's module names: `decoder.{0,1,2}.1` (3×3 conv,
    /// stride 2), `decoder.{0,1,2}.4` (1×1 conv to one channel, stride 2), `decoder.3.0` (Linear),
    /// `out` (Linear).
    pub fn layers(&self) -> Vec<SnLayer> {
        let mut out = Vec::new();
        for (i, &c) in self.in_ch1.iter().enumerate() {
            out.push(SnLayer {
                prefix: format!("decoder.{i}.1"),
                shape: vec![self.out_ch, c, 3, 3],
            });
            out.push(SnLayer {
                prefix: format!("decoder.{i}.4"),
                shape: vec![1, self.out_ch, 1, 1],
            });
        }
        out.push(SnLayer {
            prefix: "decoder.3.0".into(),
            shape: vec![self.out_ch, self.in_ch2],
        });
        out.push(SnLayer {
            prefix: "out".into(),
            shape: vec![1, self.out_ch],
        });
        out
    }
}

/// Key of a spectral layer's trained weight (`weight_orig`), bias, and power-iteration buffers.
pub fn sn_keys(prefix: &str) -> [String; 4] {
    [
        format!("{prefix}.weight_orig"),
        format!("{prefix}.bias"),
        format!("{prefix}.weight_u"),
        format!("{prefix}.weight_v"),
    ]
}

/// Stream tag of the critic head's initialization.
const CRITIC_STREAM: u64 = 0x4352_4954_4943; // "CRITIC"

fn fnv64(key: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in key.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// One host tensor: key, shape, row-major f32 values.
pub type HostTensor = (String, Vec<usize>, Vec<f32>);

fn l2_normalize(v: &mut [f32]) {
    let n = v
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(SPECTRAL_EPS);
    for x in v.iter_mut() {
        *x /= n;
    }
}

/// The critic head's fresh state: PyTorch's default `Conv2d` / `Linear` init (`kaiming_uniform_(a=√5)`
/// ⇒ `U(±1/√fan_in)` weights, `U(±1/√fan_in)` biases) and `spectral_norm`'s `u, v ~ normalize(N(0, 1))`,
/// drawn from the host stream keyed on `mix_seed(seed, CRITIC, fnv(key))` — identical on every
/// backend (the torch stream is not reproduced; the law is).
pub fn critic_head_init(cfg: &CriticHeadConfig, seed: u64) -> Vec<HostTensor> {
    let mut out = Vec::new();
    for layer in cfg.layers() {
        let (rows, cols) = layer.matrix();
        let bound = 1.0 / (cols as f64).sqrt();
        let [w_key, b_key, u_key, v_key] = sn_keys(&layer.prefix);
        let uni = |key: &str, n: usize| -> Vec<f32> {
            let mut rng = HostRng::new(mix_seed(&[seed, CRITIC_STREAM, fnv64(key)]));
            (0..n)
                .map(|_| ((rng.uniform() * 2.0 - 1.0) * bound) as f32)
                .collect()
        };
        let nrm = |key: &str, n: usize| -> Vec<f32> {
            let mut rng = HostRng::new(mix_seed(&[seed, CRITIC_STREAM, fnv64(key)]));
            let mut v = rng.normals_f32(n);
            l2_normalize(&mut v);
            v
        };
        out.push((w_key.clone(), layer.shape.clone(), uni(&w_key, rows * cols)));
        out.push((b_key.clone(), vec![rows], uni(&b_key, rows)));
        out.push((u_key.clone(), vec![rows], nrm(&u_key, rows)));
        out.push((v_key.clone(), vec![cols], nrm(&v_key, cols)));
    }
    out
}

/// The 3-D output geometry of one conv level for an `h × w` feature map:
/// `BlurPool(pad 2, stride 1)` → `+1`; 3×3 stride-2 valid conv; `BlurPool(pad 1)` → `−1`; 1×1
/// stride-2 conv. Returns the logit map size, or `None` when the map is too small.
pub fn critic_level_size(h: usize, w: usize) -> Option<(usize, usize)> {
    let f = |n: usize| -> Option<usize> {
        let a = n + 1; // blur, pad 2, filt 4, stride 1
        if a < 3 {
            return None;
        }
        let b = (a - 3) / 2 + 1; // conv 3x3 stride 2, pad 0
        if b < 2 {
            return None;
        }
        let c = b - 1; // blur, pad 1
        Some(c.div_ceil(2)) // conv 1x1 stride 2
    };
    Some((f(h)?, f(w)?))
}

/// `nn.BCEWithLogitsLoss` on one logit: `max(x, 0) − x·t + log(1 + e^{−|x|})`.
pub fn bce_with_logits(x: f64, t: f64) -> f64 {
    x.max(0.0) - x * t + (-x.abs()).exp().ln_1p()
}

// =============================================================================================
// Options and the resolved plan
// =============================================================================================

/// How a synthetic-degradation item is cropped to the training square.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CropMode {
    /// guided-diffusion `random_crop_arr(min_crop_frac=0.7)` (BOX halving, bicubic resize of the
    /// short side to a random size in `[S, ⌈S/0.7⌉]`, random crop) — the paper's "1024² crops".
    Random,
    /// `center_crop_arr`.
    Center,
    /// HYPIR's `crop_type: none`: every image must already be `S × S`.
    None,
}

/// How the generator weights start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestoreInit {
    /// The staged restoration export's own `model.safetensors` (continue fine-tuning the release).
    Export,
    /// The [`PARENT_COMPONENT`] generation backbone's weights (the paper's parent checkpoint).
    Parent,
}

/// What a run trains (the export is always the full `upscaler/` layout).
#[derive(Clone, Debug, PartialEq)]
pub enum RestoreArtifact {
    Full,
    Lora {
        rank: usize,
        alpha: f32,
        targets: Vec<String>,
    },
    Lokr {
        rank: usize,
        alpha: f32,
        decompose_factor: i32,
        targets: Vec<String>,
    },
}

impl RestoreArtifact {
    pub fn kind(&self) -> &'static str {
        match self {
            RestoreArtifact::Full => "full",
            RestoreArtifact::Lora { .. } => "lora",
            RestoreArtifact::Lokr { .. } => "lokr",
        }
    }
    pub fn targets(&self) -> &[String] {
        match self {
            RestoreArtifact::Full => &[],
            RestoreArtifact::Lora { targets, .. } | RestoreArtifact::Lokr { targets, .. } => {
                targets
            }
        }
    }
}

/// The data source of a run.
#[derive(Clone, Debug, PartialEq)]
pub enum PairSource {
    /// Clean images degraded on the fly.
    Synthetic(Box<DegradationConfig>),
    /// Aligned degraded/clean pairs ([`ITEM_DEGRADED_KEY`]); `hflip` is the shared augmentation.
    Paired { hflip: bool },
}

/// AdamW hyper-parameters of one network.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWPlan {
    pub lr: f64,
    pub betas: (f64, f64),
    pub weight_decay: f64,
    pub eps: f64,
    pub clip: f64,
}

/// The Eq. 6 weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LossWeights {
    pub l2: f64,
    pub lpips: f64,
    pub gan: f64,
}

/// Paper Eq. 6 / HYPIR `lambda_l2`, `lambda_lpips`, `lambda_gan`.
pub const RECIPE_LOSS_WEIGHTS: LossWeights = LossWeights {
    l2: 1.0,
    lpips: 5.0,
    gan: 0.5,
};
/// HYPIR `opt_kwargs.betas`.
pub const RECIPE_BETAS: (f64, f64) = (0.9, 0.999);
/// HYPIR `max_grad_norm`.
pub const RECIPE_CLIP: f64 = 1.0;
/// Paper (Iris arm, "EMA of weights with decay 0.999 is scored") / HYPIR `ema_decay`.
pub const RECIPE_EMA_DECAY: f64 = 0.999;
/// `torch.optim.AdamW` eps.
pub const ADAMW_EPS: f64 = 1e-8;

const OPTION_KEYS: [&str; 26] = [
    "loss_l2",
    "loss_lpips",
    "loss_gan",
    "gan_real_label",
    "lr_critic",
    "betas",
    "critic_betas",
    "critic_weight_decay",
    "gradient_clip",
    "critic_gradient_clip",
    "ema_enabled",
    "ema_decay",
    "crop",
    "degradation",
    "critic_dtype",
    "init",
    "num_epochs",
    "nan_loss_tolerance",
    "keep_last_checkpoints",
    "milestone_steps",
    "resume_from",
    "resume_data_policy",
    "export_weights",
    "export_dtype",
    "preview_weights",
    "lora_init",
];

/// The `TrainingConfig::model_options["iris_restore"]` object (every key optional).
///
/// | key | source | default |
/// | --- | --- | --- |
/// | `loss_l2`, `loss_lpips`, `loss_gan` | Eq. 6 / HYPIR `lambda_*` | 1, 5, 0.5 |
/// | `gan_real_label` | `multilevel_loss(alpha=0.8)` | 0.8 |
/// | `lr_critic` | HYPIR `lr_D` | `TrainingConfig::learning_rate` (paper: same 1e-5) |
/// | `betas`, `critic_betas` | HYPIR `opt_kwargs.betas` | `[0.9, 0.999]` |
/// | `critic_weight_decay` | torch `AdamW` default | `TrainingConfig::weight_decay` |
/// | `gradient_clip`, `critic_gradient_clip` | HYPIR `max_grad_norm` | 1.0 |
/// | `ema_enabled`, `ema_decay` | HYPIR `use_ema`, `ema_decay` / paper | true, 0.999 |
/// | `crop` | paper "1024² crops" | `"random"` (`"center"`, `"none"`) |
/// | `degradation` | HYPIR `sd2_train.yaml` dataset + batch_transform | the yaml (any key overrides) |
/// | `critic_dtype` | HYPIR `ImageConvNextDiscriminator(precision="bf16")` | `"bf16"` (`"fp32"`) |
/// | `init` | paper "parent checkpoint" | `"export"` (`"parent"` needs the parent component) |
/// | `lora_init` | HYPIR `init_lora_weights="gaussian"` | `"gaussian"` (`"kaiming"`) |
/// | `num_epochs`, `nan_loss_tolerance`, `keep_last_checkpoints`, `milestone_steps`, `resume_from`, `resume_data_policy` | lifecycle | as the generation trainer |
/// | `export_weights` / `preview_weights` | paper "EMA … is scored" | `"ema"` when EMA is on |
/// | `export_dtype` | the release export is FP32 | `"fp32"` (`"bf16"`) |
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RestoreOptions(pub JsonMap<String, JsonValue>);

impl RestoreOptions {
    pub fn from_config(cfg: &TrainingConfig) -> Result<Self> {
        let map = match cfg.model_options.get(OPTIONS_KEY) {
            None | Some(JsonValue::Null) => JsonMap::new(),
            Some(JsonValue::Object(m)) => m.clone(),
            Some(other) => {
                return Err(Error::Msg(format!(
                    "iris restore training: model_options.{OPTIONS_KEY} must be an object, got \
                     {other}"
                )))
            }
        };
        for key in map.keys() {
            if !OPTION_KEYS.contains(&key.as_str()) {
                return Err(Error::Unsupported(format!(
                    "iris restore training: model_options.{OPTIONS_KEY}.{key} is not a restoration \
                     training option (known: {})",
                    OPTION_KEYS.join(", ")
                )));
            }
        }
        for other in cfg.model_options.keys() {
            if other != OPTIONS_KEY {
                return Err(Error::Unsupported(format!(
                    "iris restore training: model_options.{other} is not read by the restoration \
                     trainer (its options live under model_options.{OPTIONS_KEY})"
                )));
            }
        }
        Ok(Self(map))
    }

    fn get(&self, key: &str) -> Option<&JsonValue> {
        self.0.get(key).filter(|v| !v.is_null())
    }
    fn f64(&self, key: &str, default: f64) -> Result<f64> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| {
                Error::Msg(format!(
                    "iris restore training: {key} must be a finite number, got {v}"
                ))
            }),
        }
    }
    fn u64(&self, key: &str) -> Result<Option<u64>> {
        self.get(key)
            .map(|v| {
                v.as_u64().ok_or_else(|| {
                    Error::Msg(format!(
                        "iris restore training: {key} must be a non-negative integer, got {v}"
                    ))
                })
            })
            .transpose()
    }
    fn bool(&self, key: &str, default: bool) -> Result<bool> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v.as_bool().ok_or_else(|| {
                Error::Msg(format!(
                    "iris restore training: {key} must be a bool, got {v}"
                ))
            }),
        }
    }
    fn string(&self, key: &str) -> Result<Option<String>> {
        self.get(key)
            .map(|v| {
                v.as_str().map(str::to_string).ok_or_else(|| {
                    Error::Msg(format!(
                        "iris restore training: {key} must be a string, got {v}"
                    ))
                })
            })
            .transpose()
    }
    fn pair(&self, key: &str, default: (f64, f64)) -> Result<(f64, f64)> {
        match self.get(key) {
            None => Ok(default),
            Some(JsonValue::Array(b)) if b.len() == 2 => {
                let f = |v: &JsonValue| v.as_f64().filter(|x| (0.0..1.0).contains(x));
                match (f(&b[0]), f(&b[1])) {
                    (Some(a), Some(c)) => Ok((a, c)),
                    _ => Err(Error::Msg(format!(
                        "iris restore training: {key} must be two numbers in [0, 1)"
                    ))),
                }
            }
            Some(v) => Err(Error::Msg(format!(
                "iris restore training: {key} must be a pair, got {v}"
            ))),
        }
    }
}

/// Parse the `degradation` options object over the HYPIR defaults (key names as in
/// `sd2_train.yaml`). An unknown key is a refusal.
pub fn parse_degradation(v: Option<&JsonValue>) -> Result<DegradationConfig> {
    let mut cfg = DegradationConfig::default();
    let Some(v) = v.filter(|v| !v.is_null()) else {
        return Ok(cfg);
    };
    let JsonValue::Object(m) = v else {
        return Err(Error::Msg(
            "iris restore training: degradation must be an object".into(),
        ));
    };
    let num = |k: &str, v: &JsonValue| -> Result<f64> {
        v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| {
            Error::Msg(format!(
                "iris restore training: degradation.{k} must be a number"
            ))
        })
    };
    let range = |k: &str, v: &JsonValue| -> Result<(f64, f64)> {
        match v {
            JsonValue::Array(a) if a.len() == 2 => Ok((num(k, &a[0])?, num(k, &a[1])?)),
            _ => Err(Error::Msg(format!(
                "iris restore training: degradation.{k} must be a [lo, hi] pair"
            ))),
        }
    };
    let probs = |k: &str, v: &JsonValue| -> Result<Vec<f64>> {
        match v {
            JsonValue::Array(a) => a.iter().map(|x| num(k, x)).collect(),
            _ => Err(Error::Msg(format!(
                "iris restore training: degradation.{k} must be a list of numbers"
            ))),
        }
    };
    let kinds = |k: &str, v: &JsonValue| -> Result<Vec<KernelType>> {
        match v {
            JsonValue::Array(a) => a
                .iter()
                .map(|x| {
                    x.as_str()
                        .ok_or_else(|| {
                            Error::Msg(format!(
                                "iris restore training: degradation.{k} must list kernel names"
                            ))
                        })
                        .and_then(KernelType::parse)
                })
                .collect(),
            _ => Err(Error::Msg(format!(
                "iris restore training: degradation.{k} must be a list"
            ))),
        }
    };
    let three = |k: &str, v: &JsonValue| -> Result<[f64; 3]> {
        let p = probs(k, v)?;
        p.try_into().map_err(|_| {
            Error::Msg(format!(
                "iris restore training: degradation.{k} must be [up, down, keep]"
            ))
        })
    };
    let boolean = |k: &str, v: &JsonValue| -> Result<bool> {
        v.as_bool().ok_or_else(|| {
            Error::Msg(format!(
                "iris restore training: degradation.{k} must be a bool"
            ))
        })
    };
    #[allow(clippy::type_complexity)]
    fn kernel_key(
        k: &mut KernelConfig,
        base: &str,
        v: &JsonValue,
        num: &dyn Fn(&str, &JsonValue) -> Result<f64>,
        range: &dyn Fn(&str, &JsonValue) -> Result<(f64, f64)>,
        probs: &dyn Fn(&str, &JsonValue) -> Result<Vec<f64>>,
        kinds: &dyn Fn(&str, &JsonValue) -> Result<Vec<KernelType>>,
    ) -> Result<bool> {
        match base {
            "kernel_list" => k.kernel_list = kinds(base, v)?,
            "kernel_prob" => k.kernel_prob = probs(base, v)?,
            "sinc_prob" => k.sinc_prob = num(base, v)?,
            "blur_sigma" => k.blur_sigma = range(base, v)?,
            "betag_range" => k.betag_range = range(base, v)?,
            "betap_range" => k.betap_range = range(base, v)?,
            _ => return Ok(false),
        }
        Ok(true)
    }
    #[allow(clippy::type_complexity)]
    fn stage_key(
        s: &mut StageConfig,
        base: &str,
        v: &JsonValue,
        num: &dyn Fn(&str, &JsonValue) -> Result<f64>,
        range: &dyn Fn(&str, &JsonValue) -> Result<(f64, f64)>,
        three: &dyn Fn(&str, &JsonValue) -> Result<[f64; 3]>,
    ) -> Result<bool> {
        match base {
            "resize_prob" => s.resize_prob = three(base, v)?,
            "resize_range" => s.resize_range = range(base, v)?,
            "gaussian_noise_prob" => s.gaussian_noise_prob = num(base, v)?,
            "noise_range" => s.noise_range = range(base, v)?,
            "poisson_scale_range" => s.poisson_scale_range = range(base, v)?,
            "gray_noise_prob" => s.gray_noise_prob = num(base, v)?,
            "jpeg_range" => s.jpeg_range = range(base, v)?,
            _ => return Ok(false),
        }
        Ok(true)
    }
    for (key, val) in m {
        let handled = match key.as_str() {
            "final_sinc_prob" => {
                cfg.final_sinc_prob = num(key, val)?;
                true
            }
            "use_hflip" => {
                cfg.use_hflip = boolean(key, val)?;
                true
            }
            "use_rot" => {
                cfg.use_rot = boolean(key, val)?;
                true
            }
            "use_sharpener" => {
                cfg.use_sharpener = boolean(key, val)?;
                true
            }
            "second_blur_prob" => {
                cfg.second_blur_prob = num(key, val)?;
                true
            }
            "stage2_scale" => {
                cfg.stage2_scale = num(key, val)?;
                true
            }
            "resize_back" => {
                cfg.resize_back = boolean(key, val)?;
                true
            }
            k => {
                if let Some(base) = k.strip_suffix('2') {
                    kernel_key(&mut cfg.kernel2, base, val, &num, &range, &probs, &kinds)?
                        || stage_key(&mut cfg.stage2, base, val, &num, &range, &three)?
                } else {
                    kernel_key(&mut cfg.kernel1, k, val, &num, &range, &probs, &kinds)?
                        || stage_key(&mut cfg.stage1, k, val, &num, &range, &three)?
                }
            }
        };
        if !handled {
            return Err(Error::Unsupported(format!(
                "iris restore training: degradation.{key} is not a Real-ESRGAN degradation key"
            )));
        }
    }
    if !cfg.resize_back {
        return Err(Error::Unsupported(
            "iris restore training: degradation.resize_back false would train on an input the \
             restorer never sees (it always upsamples its input to the output size first)"
                .into(),
        ));
    }
    cfg.validate()?;
    Ok(cfg)
}

/// The fully resolved, validated plan of one restoration training run.
#[derive(Clone, Debug, PartialEq)]
pub struct RestoreTrainPlan {
    pub artifact: RestoreArtifact,
    pub init: RestoreInit,
    pub lora_gaussian_init: bool,
    /// The training square `S`.
    pub image_size: usize,
    pub crop: CropMode,
    pub source: PairSource,
    pub batch_size: usize,
    pub grad_accum: usize,
    /// Steps (one generator window + one critic window each).
    pub max_steps: u32,
    pub num_epochs: Option<u32>,
    pub seed: u64,
    pub mixed_precision: MixedPrecision,
    pub critic_bf16: bool,
    /// The export's one-step noise level and the model time `σ · N`.
    pub sigma: f64,
    pub model_time: f32,
    pub weights: LossWeights,
    pub gan_real_label: f64,
    pub generator: AdamWPlan,
    pub critic: AdamWPlan,
    pub schedule: LrShape,
    /// Warmup in steps.
    pub warmup_steps: u64,
    pub ema: Option<f64>,
    pub nan_loss_tolerance: u32,
    /// Checkpoint / preview cadence in steps (0 = off).
    pub save_every: u32,
    pub preview_every: u32,
    pub keep_last_checkpoints: usize,
    pub milestone_steps: Vec<u64>,
    pub resume: bool,
    pub resume_from: Option<PathBuf>,
    pub resume_data_policy: ResumeDataPolicy,
    pub export_weights: WeightsSelect,
    pub preview_weights: WeightsSelect,
    pub export_bf16: bool,
}

impl RestoreTrainPlan {
    /// Micro-batches per step (`2 · grad_accum`: a generator window then a critic window).
    pub fn micro_per_step(&self) -> usize {
        2 * self.grad_accum
    }

    /// Resolve and validate a request against the staged export. `linear_paths` resolves adapter
    /// targets (empty = skip, the request-only `validate`). `has_parent` says whether the parent
    /// component is staged.
    pub fn resolve(
        req: &TrainingRequest,
        export: &RestorationResources,
        linear_paths: &[String],
        has_parent: bool,
    ) -> Result<Self> {
        let cfg = &req.config;
        let opts = RestoreOptions::from_config(cfg)?;
        let unsupported = |what: String| -> Result<Self> {
            Err(Error::Unsupported(format!("iris restore training: {what}")))
        };
        if cfg.gradient_checkpointing {
            return unsupported(
                "gradient (activation) checkpointing is not implemented by the native trainer"
                    .into(),
            );
        }
        // The shared request defaults (`loss_type "mse"`, `timestep_type "sigmoid"`,
        // `timestep_bias "balanced"`) stand for "not set"; any other value names a control the
        // one-step restoration recipe does not have and is refused, never ignored. (`mse` is also
        // exactly the recipe's pixel term.)
        let loss = cfg.loss_type.trim().to_ascii_lowercase();
        if !(loss.is_empty() || loss == "mse") {
            return unsupported(format!(
                "loss_type {:?} — the restoration objective is the fixed pixel (mse) + LPIPS + \
                 adversarial recipe (weights under model_options.{OPTIONS_KEY})",
                cfg.loss_type
            ));
        }
        for (field, value, default) in [
            ("timestep_type", &cfg.timestep_type, "sigmoid"),
            ("timestep_bias", &cfg.timestep_bias, "balanced"),
        ] {
            let v = value.trim().to_ascii_lowercase();
            if !(v.is_empty() || v == default) {
                return unsupported(format!(
                    "{field} {value:?} — the restorer is one-step at the export's fixed sigma"
                ));
            }
        }
        if !cfg.sample_prompts.is_empty() {
            return unsupported(
                "sample_prompts — restoration previews restore the first dataset item; they take \
                 no prompt"
                    .into(),
            );
        }
        if cfg.control_type.is_some() {
            return unsupported("control_type (no control branch)".into());
        }
        if cfg.trigger_word.is_some() || !req.trigger_words.is_empty() {
            return unsupported("trigger words (the restorer takes no prompt)".into());
        }
        let mixed_precision = match cfg.train_dtype.trim().to_ascii_lowercase().as_str() {
            "bf16" | "bfloat16" => MixedPrecision::Bf16,
            "f32" | "fp32" | "float32" | "no" => MixedPrecision::Fp32,
            other => return unsupported(format!("train_dtype {other:?} (bf16 | fp32)")),
        };
        let schedule =
            match cfg.lr_scheduler {
                LrSchedule::Constant => LrShape::Constant,
                LrSchedule::Cosine => LrShape::Cosine,
                LrSchedule::Linear => return unsupported(
                    "lr_scheduler linear (the Iris arm trains at a constant LR; constant | cosine)"
                        .into(),
                ),
            };
        if !cfg.optimizer.trim().eq_ignore_ascii_case("adamw") {
            return unsupported(format!(
                "optimizer {:?} — the recipe is AdamW for both networks",
                cfg.optimizer
            ));
        }
        if !(cfg.learning_rate.is_finite() && cfg.learning_rate >= 0.0) {
            return Err(Error::Msg(
                "iris restore training: learning_rate must be >= 0".into(),
            ));
        }
        if !(cfg.weight_decay.is_finite() && cfg.weight_decay >= 0.0) {
            return Err(Error::Msg(
                "iris restore training: weight_decay must be >= 0".into(),
            ));
        }
        let batch_size = cfg.batch_size.max(1) as usize;
        let grad_accum = cfg.gradient_accumulation.max(1) as usize;
        let per_step = 2 * grad_accum as u32;
        if cfg.steps == 0 {
            return Err(Error::Msg(
                "iris restore training: steps must be >= 1".into(),
            ));
        }
        for (field, value) in [
            ("steps", cfg.steps),
            ("save_every", cfg.save_every),
            ("sample_every", cfg.sample_every),
        ] {
            if !value.is_multiple_of(per_step) {
                return Err(Error::Msg(format!(
                    "iris restore training: {field} = {value} micro-steps is not a multiple of \
                     2 · gradient_accumulation = {per_step} (one step is a generator window then \
                     a critic window)"
                )));
            }
        }

        // ---- export geometry -------------------------------------------------------------------
        let config = export.config();
        let patch = config.model.patch_size;
        let image_size = cfg.resolution as usize;
        if image_size == 0 || !image_size.is_multiple_of(patch) {
            return Err(Error::Msg(format!(
                "iris restore training: resolution {image_size} must be a positive multiple of the \
                 backbone patch_size {patch}"
            )));
        }
        let sigma = export.settings.sigma;
        let model_time = export.settings.model_time(config);

        // ---- artifact --------------------------------------------------------------------------
        let artifact = if cfg.full_finetune {
            RestoreArtifact::Full
        } else {
            if cfg.rank == 0 {
                return Err(Error::Msg(
                    "iris restore training: adapter rank must be >= 1".into(),
                ));
            }
            if !(cfg.alpha.is_finite() && cfg.alpha > 0.0) {
                return Err(Error::Msg(
                    "iris restore training: adapter alpha must be > 0".into(),
                ));
            }
            let targets = if linear_paths.is_empty() {
                Vec::new()
            } else {
                select_adapter_targets(linear_paths, &cfg.lora_target_modules)?
            };
            match cfg.network_type {
                NetworkType::Lora => RestoreArtifact::Lora {
                    rank: cfg.rank as usize,
                    alpha: cfg.alpha,
                    targets,
                },
                NetworkType::Lokr => RestoreArtifact::Lokr {
                    rank: cfg.rank as usize,
                    alpha: cfg.alpha,
                    decompose_factor: cfg.decompose_factor,
                    targets,
                },
            }
        };
        let lora_gaussian_init = match opts.string("lora_init")?.as_deref() {
            None | Some("gaussian") => true,
            Some("kaiming") => false,
            Some(o) => {
                return Err(Error::Msg(format!(
                    "iris restore training: lora_init {o:?} (gaussian | kaiming)"
                )))
            }
        };
        let init = match opts.string("init")?.as_deref() {
            None | Some("export") => RestoreInit::Export,
            Some("parent") => RestoreInit::Parent,
            Some(o) => {
                return Err(Error::Msg(format!(
                    "iris restore training: init {o:?} (export | parent)"
                )))
            }
        };
        if init == RestoreInit::Parent && !has_parent {
            return Err(Error::Msg(format!(
                "iris restore training: init parent needs the generation backbone staged as the \
                 '{PARENT_COMPONENT}' component"
            )));
        }
        if init == RestoreInit::Export && has_parent {
            return unsupported(format!(
                "the '{PARENT_COMPONENT}' component is staged but init is export — it would be \
                 ignored (set init: parent or drop it)"
            ));
        }

        // ---- data ------------------------------------------------------------------------------
        let crop = match opts.string("crop")?.as_deref() {
            None | Some("random") => CropMode::Random,
            Some("center") => CropMode::Center,
            Some("none") => CropMode::None,
            Some(o) => {
                return Err(Error::Msg(format!(
                    "iris restore training: crop {o:?} (random | center | none)"
                )))
            }
        };
        let paired: Vec<bool> = req
            .items
            .iter()
            .map(|i| i.model_options.contains_key(ITEM_DEGRADED_KEY))
            .collect();
        let source = if paired.iter().all(|&p| p) && !paired.is_empty() {
            let deg = parse_degradation(opts.get("degradation"))?;
            if let Some(JsonValue::Object(m)) = opts.get("degradation") {
                if m.keys().any(|k| k != "use_hflip") {
                    return unsupported(
                        "degradation options other than use_hflip with paired items — aligned \
                         pairs are not synthetically degraded"
                            .into(),
                    );
                }
            }
            PairSource::Paired {
                hflip: deg.use_hflip,
            }
        } else if paired.iter().any(|&p| p) {
            return Err(Error::Msg(format!(
                "iris restore training: some items carry {ITEM_DEGRADED_KEY} and some do not — a \
                 run is either all aligned pairs or all clean images"
            )));
        } else {
            let deg = parse_degradation(opts.get("degradation"))?;
            deg.check_crop(image_size)?;
            PairSource::Synthetic(Box::new(deg))
        };

        // ---- optimizers ------------------------------------------------------------------------
        let lr = cfg.learning_rate as f64;
        let generator = AdamWPlan {
            lr,
            betas: opts.pair("betas", RECIPE_BETAS)?,
            weight_decay: cfg.weight_decay as f64,
            eps: ADAMW_EPS,
            clip: opts.f64("gradient_clip", RECIPE_CLIP)?,
        };
        let critic = AdamWPlan {
            lr: opts.f64("lr_critic", lr)?,
            betas: opts.pair("critic_betas", generator.betas)?,
            weight_decay: opts.f64("critic_weight_decay", generator.weight_decay)?,
            eps: ADAMW_EPS,
            clip: opts.f64("critic_gradient_clip", RECIPE_CLIP)?,
        };
        for (n, p) in [("generator", &generator), ("critic", &critic)] {
            if !(p.lr >= 0.0 && p.weight_decay >= 0.0 && p.clip > 0.0) {
                return Err(Error::Msg(format!(
                    "iris restore training: the {n} optimizer needs lr >= 0, weight_decay >= 0 and \
                     a positive gradient clip"
                )));
            }
        }
        let weights = LossWeights {
            l2: opts.f64("loss_l2", RECIPE_LOSS_WEIGHTS.l2)?,
            lpips: opts.f64("loss_lpips", RECIPE_LOSS_WEIGHTS.lpips)?,
            gan: opts.f64("loss_gan", RECIPE_LOSS_WEIGHTS.gan)?,
        };
        if weights.l2 < 0.0 || weights.lpips < 0.0 || weights.gan < 0.0 {
            return Err(Error::Msg(
                "iris restore training: loss weights must be >= 0".into(),
            ));
        }
        let gan_real_label = opts.f64("gan_real_label", GAN_REAL_LABEL)?;
        if !(0.0..=1.0).contains(&gan_real_label) {
            return Err(Error::Msg(
                "iris restore training: gan_real_label must be in [0, 1]".into(),
            ));
        }
        let critic_bf16 = match opts.string("critic_dtype")?.as_deref() {
            None | Some("bf16") => true,
            Some("fp32") => false,
            Some(o) => {
                return Err(Error::Msg(format!(
                    "iris restore training: critic_dtype {o:?} (bf16 | fp32)"
                )))
            }
        };
        let ema = if opts.bool("ema_enabled", true)? {
            let d = opts.f64("ema_decay", RECIPE_EMA_DECAY)?;
            if !(0.0..=1.0).contains(&d) {
                return Err(Error::Msg(
                    "iris restore training: ema_decay must be in [0, 1]".into(),
                ));
            }
            Some(d)
        } else {
            None
        };
        let default_weights = if ema.is_some() {
            WeightsSelect::Ema
        } else {
            WeightsSelect::Raw
        };
        let select = |key: &str| -> Result<WeightsSelect> {
            let w = match opts.string(key)?.as_deref() {
                None => default_weights,
                Some("ema") => WeightsSelect::Ema,
                Some("raw") => WeightsSelect::Raw,
                Some(o) => {
                    return Err(Error::Msg(format!(
                        "iris restore training: {key} {o:?} (ema | raw)"
                    )))
                }
            };
            if w == WeightsSelect::Ema && ema.is_none() {
                return Err(Error::Msg(format!(
                    "iris restore training: {key} ema needs ema_enabled"
                )));
            }
            Ok(w)
        };
        let export_weights = select("export_weights")?;
        let preview_weights = select("preview_weights")?;
        let export_bf16 = match opts.string("export_dtype")?.as_deref() {
            None | Some("fp32") => false,
            Some("bf16") => true,
            Some(o) => {
                return Err(Error::Msg(format!(
                    "iris restore training: export_dtype {o:?} (fp32 | bf16)"
                )))
            }
        };
        let num_epochs = opts.u64("num_epochs")?.map(|v| v as u32);
        if num_epochs == Some(0) {
            return Err(Error::Msg(
                "iris restore training: num_epochs must be >= 1".into(),
            ));
        }
        let resume_data_policy = match opts.string("resume_data_policy")?.as_deref() {
            None | Some("exact") => ResumeDataPolicy::Exact,
            Some("new_phase") => ResumeDataPolicy::NewPhase,
            Some(o) => {
                return Err(Error::Msg(format!(
                    "iris restore training: resume_data_policy {o:?} (exact | new_phase)"
                )))
            }
        };
        let milestone_steps = match opts.get("milestone_steps") {
            None => Vec::new(),
            Some(JsonValue::Array(a)) => a
                .iter()
                .map(|v| {
                    v.as_u64().ok_or_else(|| {
                        Error::Msg("iris restore training: milestone_steps must be integers".into())
                    })
                })
                .collect::<Result<_>>()?,
            Some(v) => {
                return Err(Error::Msg(format!(
                    "iris restore training: milestone_steps must be a list, got {v}"
                )))
            }
        };
        Ok(Self {
            artifact,
            init,
            lora_gaussian_init,
            image_size,
            crop,
            source,
            batch_size,
            grad_accum,
            max_steps: cfg.steps / per_step,
            num_epochs,
            seed: cfg.seed,
            mixed_precision,
            critic_bf16,
            sigma,
            model_time,
            weights,
            gan_real_label,
            generator,
            critic,
            schedule,
            warmup_steps: cfg.lr_warmup_steps.div_ceil(per_step) as u64,
            ema,
            nan_loss_tolerance: opts.u64("nan_loss_tolerance")?.unwrap_or(20) as u32,
            save_every: cfg.save_every / per_step,
            preview_every: cfg.sample_every / per_step,
            keep_last_checkpoints: opts.u64("keep_last_checkpoints")?.unwrap_or(0) as usize,
            milestone_steps,
            resume: cfg.resume,
            resume_from: opts.string("resume_from")?.map(PathBuf::from),
            resume_data_policy,
            export_weights,
            preview_weights,
            export_bf16,
        })
    }

    /// The identity a resume must match (the trained state's shape).
    pub fn state_identity(&self) -> String {
        let artifact = match &self.artifact {
            RestoreArtifact::Full => "full".to_string(),
            RestoreArtifact::Lora {
                rank,
                alpha,
                targets,
            } => format!("lora:{rank}:{alpha}:{}", digest(targets)),
            RestoreArtifact::Lokr {
                rank,
                alpha,
                decompose_factor,
                targets,
            } => format!("lokr:{rank}:{alpha}:{decompose_factor}:{}", digest(targets)),
        };
        format!(
            "task=restoration;artifact={artifact};ema={};critic=convnext_xxlarge:{}",
            self.ema.is_some(),
            CRITIC_HEAD_WIDTH
        )
    }

    /// The data-shaping identity an `exact` resume also requires.
    pub fn data_identity(&self) -> String {
        format!(
            "image_size={};crop={:?};batch={};accum={};seed={};source={:?}",
            self.image_size, self.crop, self.batch_size, self.grad_accum, self.seed, self.source
        )
    }
}

fn digest(items: &[String]) -> String {
    let mut h = Sha256::new();
    for s in items {
        h.update((s.len() as u64).to_le_bytes());
        h.update(s.as_bytes());
    }
    format!("{:x}", h.finalize())[..16].to_string()
}

/// The request's dataset fingerprint: the shared request fingerprint plus every paired item's
/// degraded-image bytes.
pub fn dataset_fingerprint(req: &TrainingRequest) -> Result<String> {
    let base = crate::train::resume::request_fingerprint(req)?;
    let mut h = Sha256::new();
    h.update(base.as_bytes());
    for item in &req.items {
        if let Some(p) = degraded_path(item)? {
            let bytes = std::fs::read(&p).map_err(|e| {
                Error::Msg(format!("iris restore training: read {}: {e}", p.display()))
            })?;
            h.update(Sha256::digest(&bytes));
        }
        h.update([0u8]);
    }
    Ok(format!("{:x}", h.finalize()))
}

// =============================================================================================
// Resources
// =============================================================================================

/// Every resource of a restoration training load, identity-checked.
#[derive(Clone, Debug, PartialEq)]
pub struct RestoreTrainResources {
    /// The `upscaler/` export the run fine-tunes (its config, empty prompt and — under
    /// `init: export` — weights).
    pub export: RestorationResources,
    pub vgg16: PathBuf,
    pub lpips_lin: PathBuf,
    pub critic_trunk: PathBuf,
    /// The generation backbone directory (`init: parent`).
    pub parent: Option<PathBuf>,
}

fn component_file(spec: &LoadSpec, res: &TrainingResource) -> Result<PathBuf> {
    match spec.components.get(res.component) {
        Some(WeightsSource::File(f)) if f.is_file() => Ok(f.clone()),
        Some(WeightsSource::File(f)) => Err(Error::Msg(format!(
            "{TRAINER_ID} trainer: the '{}' resource {} does not exist",
            res.component,
            f.display()
        ))),
        Some(WeightsSource::Dir(d)) => {
            let p = d.join(Path::new(res.file).file_name().unwrap_or_default());
            if p.is_file() {
                Ok(p)
            } else {
                Err(Error::Msg(format!(
                    "{TRAINER_ID} trainer: the '{}' resource directory {} has no {}",
                    res.component,
                    d.display(),
                    res.file
                )))
            }
        }
        None => Err(Error::Msg(format!(
            "{TRAINER_ID} trainer: stage the {} ({}@{}, sha256 {}) as the '{}' component — a \
             training-only resource the restoration recipe needs",
            res.name, res.repo, res.revision, res.sha256, res.component
        ))),
    }
}

impl RestoreTrainResources {
    /// Resolve from a load spec: `spec.weights` = the restoration export directory;
    /// [`VGG16_COMPONENT`], [`LPIPS_LIN_COMPONENT`], [`CRITIC_TRUNK_COMPONENT`] (files, or
    /// directories holding them); optional [`PARENT_COMPONENT`] (a generation backbone directory).
    /// A text encoder or any other component is refused (E4).
    pub fn from_spec(spec: &LoadSpec) -> Result<Self> {
        crate::control::reject_unknown_components(
            spec,
            &[
                VGG16_COMPONENT,
                LPIPS_LIN_COMPONENT,
                CRITIC_TRUNK_COMPONENT,
                PARENT_COMPONENT,
            ],
            TRAINER_ID,
        )?;
        if spec.text_encoder.is_some() {
            return Err(Error::Unsupported(format!(
                "{TRAINER_ID} trainer: the restoration task never loads a text encoder"
            )));
        }
        // TaskExport refuses components; resolve the export from a spec without them.
        let mut export_spec = LoadSpec::new(spec.weights.clone());
        export_spec.precision = spec.precision;
        let export = RestorationResources::from_spec(&export_spec, TRAINER_ID)?;
        let parent = match spec.components.get(PARENT_COMPONENT) {
            None => None,
            Some(WeightsSource::Dir(d)) => {
                for f in [super::BACKBONE_CONFIG_FILE, super::BACKBONE_WEIGHTS_FILE] {
                    if !d.join(f).is_file() {
                        return Err(Error::Msg(format!(
                            "{TRAINER_ID} trainer: the parent backbone {} has no {f}",
                            d.display()
                        )));
                    }
                }
                let text = std::fs::read_to_string(d.join(super::BACKBONE_CONFIG_FILE))
                    .map_err(|e| Error::Msg(format!("iris: read parent config: {e}")))?;
                if super::downstream::parse_task_settings(&text)?.is_some() {
                    return Err(Error::Unsupported(format!(
                        "{TRAINER_ID} trainer: the '{PARENT_COMPONENT}' component {} is a \
                         downstream export, not the generation backbone",
                        d.display()
                    )));
                }
                let parent_cfg = IrisConfig::parse(&text)?;
                if parent_cfg.model != export.export.config.model {
                    return Err(Error::Msg(format!(
                        "{TRAINER_ID} trainer: the parent backbone's model section differs from \
                         the restoration export's ({})",
                        d.display()
                    )));
                }
                Some(d.clone())
            }
            Some(WeightsSource::File(f)) => {
                return Err(Error::Msg(format!(
                    "{TRAINER_ID} trainer: the '{PARENT_COMPONENT}' resource must be a backbone \
                     directory, not the file {}",
                    f.display()
                )))
            }
        };
        Ok(Self {
            vgg16: component_file(spec, &VGG16_RESOURCE)?,
            lpips_lin: component_file(spec, &LPIPS_LIN_RESOURCE)?,
            critic_trunk: component_file(spec, &CRITIC_TRUNK_RESOURCE)?,
            export,
            parent,
        })
    }

    /// The weights file the generator starts from.
    pub fn init_weights(&self, init: RestoreInit) -> PathBuf {
        match (init, &self.parent) {
            (RestoreInit::Parent, Some(p)) => p.join(super::BACKBONE_WEIGHTS_FILE),
            _ => self.export.export.weights_path(),
        }
    }
}

// =============================================================================================
// Training pairs
// =============================================================================================

/// One training pair at the training square, CHW f32 in `[−1, 1]` (`x·2 − 1` of the `[0, 1]`
/// pipeline): `gt` the target, `lq` the generator input at the GT size.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainPair {
    pub size: usize,
    pub gt: Vec<f32>,
    pub lq: Vec<f32>,
}

/// An item's aligned degraded-image path, if it carries one.
pub fn degraded_path(item: &TrainingItem) -> Result<Option<PathBuf>> {
    match item.model_options.get(ITEM_DEGRADED_KEY) {
        None => Ok(None),
        Some(JsonValue::String(s)) => Ok(Some(PathBuf::from(s))),
        Some(other) => Err(Error::Msg(format!(
            "iris restore training: item {} {ITEM_DEGRADED_KEY} must be a path string, got {other}",
            item.image_path.display()
        ))),
    }
}

struct Rgb {
    w: usize,
    h: usize,
    px: Vec<u8>,
}

fn decode(path: &Path) -> Result<Rgb> {
    let img = image::open(path)
        .map_err(|e| {
            Error::Msg(format!(
                "iris restore training: decode {}: {e}",
                path.display()
            ))
        })?
        .to_rgb8();
    Ok(Rgb {
        w: img.width() as usize,
        h: img.height() as usize,
        px: img.into_raw(),
    })
}

fn to_u8(v: Vec<f32>) -> Vec<u8> {
    v.into_iter().map(|x| x as u8).collect()
}

/// Python `round` (half to even).
fn py_round(v: f64) -> usize {
    v.round_ties_even().max(0.0) as usize
}

/// guided-diffusion's crop: BOX-halve while the short side is ≥ 2·`small`, PIL-bicubic the short
/// side to `small`, then crop `size × size` at `(top, left)` from `pick(range_h, range_w)`.
fn crop_resized(
    img: Rgb,
    size: usize,
    small: usize,
    pick: &mut dyn FnMut(usize, usize) -> (usize, usize),
) -> Result<Rgb> {
    let mut img = img;
    while img.w.min(img.h) >= 2 * small {
        let (nw, nh) = (img.w / 2, img.h / 2);
        let px = to_u8(crate::imageops::resize_box_u8(
            &img.px, img.h, img.w, nh, nw,
        )?);
        img = Rgb { w: nw, h: nh, px };
    }
    let scale = small as f64 / img.w.min(img.h) as f64;
    let (nw, nh) = (
        py_round(img.w as f64 * scale),
        py_round(img.h as f64 * scale),
    );
    if (nw, nh) != (img.w, img.h) {
        let px = to_u8(crate::imageops::resize_bicubic_u8(
            &img.px, img.h, img.w, nh, nw,
        )?);
        img = Rgb { w: nw, h: nh, px };
    }
    if img.w < size || img.h < size {
        return Err(Error::Msg(format!(
            "iris restore training: a {}×{} resize cannot hold a {size}px crop",
            img.w, img.h
        )));
    }
    let (top, left) = pick(img.h - size + 1, img.w - size + 1);
    let mut px = vec![0u8; size * size * 3];
    for y in 0..size {
        let src = ((top + y) * img.w + left) * 3;
        px[y * size * 3..(y + 1) * size * 3].copy_from_slice(&img.px[src..src + size * 3]);
    }
    Ok(Rgb {
        w: size,
        h: size,
        px,
    })
}

fn planes(img: &Rgb) -> Planes {
    let n = img.w * img.h;
    let mut p = Planes::zeros(3, img.h, img.w);
    for i in 0..n {
        for c in 0..3 {
            p.data[c * n + i] = img.px[i * 3 + c] as f32 / 255.0;
        }
    }
    p
}

fn signed(p: &Planes) -> Vec<f32> {
    p.data.iter().map(|v| v * 2.0 - 1.0).collect()
}

const CROP_STREAM: u64 = 0x4352_4f50; // "CROP"

/// The crop key of item `index` in `epoch`.
pub fn crop_key(seed: u64, epoch: u32, index: usize) -> u64 {
    mix_seed(&[seed, CROP_STREAM, epoch as u64, index as u64])
}

/// Check one item before training (decodable; a pair's sizes are aligned and hold the crop).
pub fn validate_item(item: &TrainingItem, plan: &RestoreTrainPlan) -> Result<()> {
    let (w, h) = image::image_dimensions(&item.image_path).map_err(|e| {
        Error::Msg(format!(
            "iris restore training: item {} is not a readable image: {e}",
            item.image_path.display()
        ))
    })?;
    let (w, h) = (w as usize, h as usize);
    let s = plan.image_size;
    match &plan.source {
        PairSource::Synthetic(_) => {
            if plan.crop == CropMode::None && (w, h) != (s, s) {
                return Err(Error::Msg(format!(
                    "iris restore training: crop none needs every image at {s}×{s}; {} is {w}×{h}",
                    item.image_path.display()
                )));
            }
        }
        PairSource::Paired { .. } => {
            let lq = degraded_path(item)?.expect("paired");
            let (lw, lh) = image::image_dimensions(&lq).map_err(|e| {
                Error::Msg(format!(
                    "iris restore training: degraded image {} is not readable: {e}",
                    lq.display()
                ))
            })?;
            pair_factor(w, h, lw as usize, lh as usize, s, &item.image_path)?;
            if plan.crop == CropMode::None && (w, h) != (s, s) {
                return Err(Error::Msg(format!(
                    "iris restore training: crop none needs every target at {s}×{s}; {} is {w}×{h}",
                    item.image_path.display()
                )));
            }
        }
    }
    Ok(())
}

/// The integer upscale between an aligned pair, checked: the clean image must be exactly `f×` the
/// degraded one on both axes, at least the crop, and the crop a multiple of `f`.
fn pair_factor(w: usize, h: usize, lw: usize, lh: usize, s: usize, what: &Path) -> Result<usize> {
    if lw == 0 || lh == 0 || !w.is_multiple_of(lw) || !h.is_multiple_of(lh) || w / lw != h / lh {
        return Err(Error::Msg(format!(
            "iris restore training: {} ({w}×{h}) and its degraded image ({lw}×{lh}) are not an \
             aligned integer-scale pair",
            what.display()
        )));
    }
    let f = w / lw;
    if w < s || h < s || !s.is_multiple_of(f) {
        return Err(Error::Msg(format!(
            "iris restore training: {} ({w}×{h}, ×{f} pair) cannot give an aligned {s}px crop",
            what.display()
        )));
    }
    Ok(f)
}

/// Build item `index`'s training pair for `epoch` (deterministic in `(seed, epoch, index)`).
pub fn training_pair(
    item: &TrainingItem,
    plan: &RestoreTrainPlan,
    epoch: u32,
    index: usize,
) -> Result<TrainPair> {
    let s = plan.image_size;
    let mut crop_rng = HostRng::new(crop_key(plan.seed, epoch, index));
    match &plan.source {
        PairSource::Synthetic(cfg) => {
            let img = decode(&item.image_path)?;
            let img = match plan.crop {
                CropMode::None => img,
                CropMode::Center => {
                    crop_resized(img, s, s, &mut |rh, rw| ((rh - 1) / 2, (rw - 1) / 2))?
                }
                CropMode::Random => {
                    // `randrange(ceil(S / 1.0), ceil(S / 0.7) + 1)`
                    let lo = s;
                    let hi = (s as f64 / 0.7).ceil() as usize;
                    let small = lo + crop_rng.below(hi - lo + 1);
                    crop_resized(img, s, small, &mut |rh, rw| {
                        (crop_rng.below(rh), crop_rng.below(rw))
                    })?
                }
            };
            let pair =
                degradation::degrade(cfg, &planes(&img), degradation_key(plan.seed, epoch, index))?;
            Ok(TrainPair {
                size: s,
                gt: signed(&pair.gt),
                lq: signed(&pair.lq),
            })
        }
        PairSource::Paired { hflip } => {
            let gt = decode(&item.image_path)?;
            let lq_path = degraded_path(item)?.expect("paired");
            let lq = decode(&lq_path)?;
            let f = pair_factor(gt.w, gt.h, lq.w, lq.h, s, &item.image_path)?;
            let (rh, rw) = ((gt.h - s) / f + 1, (gt.w - s) / f + 1);
            let (ty, tx) = match plan.crop {
                CropMode::Random => (crop_rng.below(rh), crop_rng.below(rw)),
                CropMode::Center | CropMode::None => ((rh - 1) / 2, (rw - 1) / 2),
            };
            let gt_p = planes(&gt).crop(ty * f, tx * f, s, s);
            let lq_p = planes(&lq).crop(ty, tx, s / f, s / f);
            let lq_up = if f == 1 {
                lq_p
            } else {
                bicubic_scale_factor(&lq_p, f as f64)
            };
            let flip =
                *hflip && HostRng::new(degradation_key(plan.seed, epoch, index)).uniform() < 0.5;
            let gt_p = degradation::augment(&gt_p, flip, false, false);
            let lq_up = degradation::augment(&lq_up, flip, false, false);
            Ok(TrainPair {
                size: s,
                gt: signed(&gt_p),
                lq: signed(&lq_up),
            })
        }
    }
}

// =============================================================================================
// The alternation
// =============================================================================================

/// Which network a micro-batch trains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Generator,
    Critic,
}

/// HYPIR's `generator_step = ((batch_count // accum) % 2) == 0` for the 0-based micro-batch count
/// within the run, and whether this micro-batch closes its window (`sync_gradients`).
pub fn phase_of(batch_count: u64, accum: usize) -> (Phase, bool) {
    let accum = accum.max(1) as u64;
    let phase = if (batch_count / accum).is_multiple_of(2) {
        Phase::Generator
    } else {
        Phase::Critic
    };
    (phase, (batch_count + 1).is_multiple_of(accum))
}

// =============================================================================================
// Checkpoint state and export identity
// =============================================================================================

/// Checkpoint format tag.
pub const CHECKPOINT_FORMAT: &str = "iris-restoration-train-checkpoint-v1";
/// The critic head (trained weights + spectral power-iteration buffers).
pub const CKPT_CRITIC: &str = "critic.safetensors";
/// The critic's AdamW slots.
pub const CKPT_CRITIC_OPTIMIZER: &str = "critic_optimizer.safetensors";

/// The resumable, non-tensor state of a restoration run. Tensors beside it:
/// [`super::train::CKPT_TRAINABLE`] / [`super::train::CKPT_EMA`] /
/// [`super::train::CKPT_OPTIMIZER`] (the generator, keyed as trained), [`CKPT_CRITIC`],
/// [`CKPT_CRITIC_OPTIMIZER`].
#[derive(Clone, Debug, PartialEq)]
pub struct RestoreCheckpointState {
    pub step: u64,
    pub epoch: u32,
    pub batches_consumed: usize,
    pub scheduler_step: u64,
    pub lr: f64,
    pub lr_critic: f64,
    pub nan_count: u32,
    /// The last step's mean losses: generator total, pixel, LPIPS, adversarial; critic.
    pub losses: [f64; 5],
    pub artifact: String,
    pub ema_decay: Option<f64>,
    pub seed: u64,
    pub backend: String,
    pub state_identity: String,
    pub data_identity: String,
    pub dataset_fingerprint: String,
    pub base_identity: String,
}

impl RestoreCheckpointState {
    pub fn to_json(&self) -> JsonValue {
        json!({
            "format": CHECKPOINT_FORMAT,
            "trainer": TRAINER_ID,
            "task": IrisTask::Restoration.name(),
            "step": self.step,
            "epoch": self.epoch,
            "batches_consumed": self.batches_consumed,
            "scheduler_step": self.scheduler_step,
            "lr": self.lr,
            "lr_critic": self.lr_critic,
            "nan_count": self.nan_count,
            "losses": self.losses.to_vec(),
            "artifact": self.artifact,
            "ema_decay": self.ema_decay,
            "seed": self.seed.to_string(),
            "rng": "positional: degradation mix_seed(seed, RESTORE1, epoch, item); crops mix_seed(seed, CROP, epoch, item)",
            "backend": self.backend,
            "state_identity": self.state_identity,
            "data_identity": self.data_identity,
            "dataset_fingerprint": self.dataset_fingerprint,
            "base_identity": self.base_identity,
        })
    }

    pub fn from_json(v: &JsonValue) -> Result<Self> {
        let bad = |k: &str| {
            Error::Msg(format!(
                "iris restore checkpoint: state.json field {k:?} is missing or malformed"
            ))
        };
        if v.get("format").and_then(JsonValue::as_str) != Some(CHECKPOINT_FORMAT) {
            return Err(Error::Msg(format!(
                "iris restore checkpoint: not a {CHECKPOINT_FORMAT} state.json (a generation \
                 checkpoint cannot resume a restoration run)"
            )));
        }
        let u = |k: &str| v.get(k).and_then(JsonValue::as_u64).ok_or_else(|| bad(k));
        let f = |k: &str| v.get(k).and_then(JsonValue::as_f64).ok_or_else(|| bad(k));
        let s = |k: &str| {
            v.get(k)
                .and_then(JsonValue::as_str)
                .map(str::to_string)
                .ok_or_else(|| bad(k))
        };
        let losses: Vec<f64> = v
            .get("losses")
            .and_then(JsonValue::as_array)
            .map(|a| a.iter().filter_map(JsonValue::as_f64).collect())
            .ok_or_else(|| bad("losses"))?;
        Ok(Self {
            step: u("step")?,
            epoch: u("epoch")? as u32,
            batches_consumed: u("batches_consumed")? as usize,
            scheduler_step: u("scheduler_step")?,
            lr: f("lr")?,
            lr_critic: f("lr_critic")?,
            nan_count: u("nan_count")? as u32,
            losses: losses.try_into().map_err(|_| bad("losses"))?,
            artifact: s("artifact")?,
            ema_decay: v.get("ema_decay").and_then(JsonValue::as_f64),
            seed: s("seed")?.parse().map_err(|_| bad("seed"))?,
            backend: s("backend")?,
            state_identity: s("state_identity")?,
            data_identity: s("data_identity")?,
            dataset_fingerprint: s("dataset_fingerprint")?,
            base_identity: s("base_identity")?,
        })
    }
}

/// Read a restoration checkpoint's `state.json`.
pub fn read_checkpoint_state(dir: &Path) -> Result<RestoreCheckpointState> {
    let path = dir.join(super::train::CKPT_STATE);
    let text = std::fs::read_to_string(&path).map_err(|e| {
        Error::Msg(format!(
            "iris restore checkpoint: read {}: {e}",
            path.display()
        ))
    })?;
    let v: JsonValue = serde_json::from_str(&text).map_err(|e| {
        Error::Msg(format!(
            "iris restore checkpoint: parse {}: {e}",
            path.display()
        ))
    })?;
    RestoreCheckpointState::from_json(&v)
}

/// Check a checkpoint against the resuming run.
pub fn check_resume(
    saved: &RestoreCheckpointState,
    plan: &RestoreTrainPlan,
    dataset_fingerprint: &str,
    base_identity: &str,
) -> Result<()> {
    if saved.state_identity != plan.state_identity() {
        return Err(Error::Msg(format!(
            "iris restore resume: the checkpoint trained a different state ({}) than this run \
             ({})",
            saved.state_identity,
            plan.state_identity()
        )));
    }
    if saved.base_identity != base_identity {
        return Err(Error::Msg(
            "iris restore resume: the checkpoint was trained from a different base".into(),
        ));
    }
    if plan.resume_data_policy == ResumeDataPolicy::Exact {
        if saved.data_identity != plan.data_identity() {
            return Err(Error::Msg(format!(
                "iris restore resume: data shaping differs (saved {}, requested {}); \
                 resume_data_policy new_phase starts the current dataset fresh",
                saved.data_identity,
                plan.data_identity()
            )));
        }
        if saved.dataset_fingerprint != dataset_fingerprint {
            return Err(Error::Msg(
                "iris restore resume: the dataset differs from the checkpoint's".into(),
            ));
        }
    }
    Ok(())
}

/// The `model.safetensors` metadata of a restoration export: `format = "pt"`,
/// [`META_ARTIFACT`] `= "full_model"`, [`META_TASK`] `= "restoration"`, [`META_WEIGHTS`],
/// [`META_STEPS`], [`META_BASE`], [`META_UPSTREAM`], plus the recipe revision.
pub fn export_metadata(
    weights: WeightsSelect,
    steps: u64,
    base_identity: &str,
) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert("format".into(), "pt".into());
    m.insert(META_ARTIFACT.into(), "full_model".into());
    m.insert(META_TASK.into(), IrisTask::Restoration.name().into());
    m.insert(META_WEIGHTS.into(), weights.as_str().into());
    m.insert(META_STEPS.into(), steps.to_string());
    m.insert(META_BASE.into(), base_identity.into());
    m.insert(META_UPSTREAM.into(), super::UPSTREAM_CODE_REVISION.into());
    m.insert("irisRecipe".into(), format!("HYPIR@{HYPIR_REVISION}"));
    m
}

/// Copy the base export's `config.yaml` (task / tile / sigma / conditioning identity) and
/// `empty_prompt.safetensors` verbatim into `dir` — the exported restorer runs at exactly the
/// geometry and time it was trained at, on exactly the conditioning it was trained with.
pub fn write_export_sidecars(export: &RestorationResources, dir: &Path) -> Result<()> {
    let cfg = std::fs::read(export.export.dir.join(super::BACKBONE_CONFIG_FILE))
        .map_err(|e| Error::Msg(format!("iris restore export: read config: {e}")))?;
    super::train::atomic_write(&dir.join(super::BACKBONE_CONFIG_FILE), &cfg)?;
    let prompt = std::fs::read(export.export.empty_prompt_path())
        .map_err(|e| Error::Msg(format!("iris restore export: read empty prompt: {e}")))?;
    super::train::atomic_write(&dir.join(super::downstream::EMPTY_PROMPT_FILE), &prompt)?;
    Ok(())
}

/// The base identity of a run: the export's config bytes + the init weights' header.
pub fn base_identity(export: &RestorationResources, init_weights: &Path) -> Result<String> {
    let mut h = Sha256::new();
    h.update(export.export.config_sha256.as_bytes());
    let mut f = std::fs::File::open(init_weights)
        .map_err(|e| Error::Msg(format!("iris: open {}: {e}", init_weights.display())))?;
    let size = f
        .metadata()
        .map_err(|e| Error::Msg(format!("iris: stat {}: {e}", init_weights.display())))?
        .len();
    let mut len = [0u8; 8];
    std::io::Read::read_exact(&mut f, &mut len)
        .map_err(|e| Error::Msg(format!("iris: read {}: {e}", init_weights.display())))?;
    let n = u64::from_le_bytes(len).min(64 << 20) as usize;
    let mut header = vec![0u8; n];
    std::io::Read::read_exact(&mut f, &mut header)
        .map_err(|e| Error::Msg(format!("iris: read {}: {e}", init_weights.display())))?;
    h.update(size.to_le_bytes());
    h.update(&header);
    Ok(format!("{:x}", h.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_alternation_is_hypirs() {
        // accum 1: G, D, G, D …; every micro-batch closes its window.
        let p: Vec<_> = (0..4).map(|b| phase_of(b, 1)).collect();
        assert_eq!(
            p,
            [
                (Phase::Generator, true),
                (Phase::Critic, true),
                (Phase::Generator, true),
                (Phase::Critic, true)
            ]
        );
        // accum 2: G G D D G …; windows close on odd counts.
        let p: Vec<_> = (0..5).map(|b| phase_of(b, 2)).collect();
        assert_eq!(p[0], (Phase::Generator, false));
        assert_eq!(p[1], (Phase::Generator, true));
        assert_eq!(p[2], (Phase::Critic, false));
        assert_eq!(p[3], (Phase::Critic, true));
        assert_eq!(p[4], (Phase::Generator, false));
    }

    #[test]
    fn the_head_layout_is_hypirs() {
        let head = ConvNextConfig::xxlarge().head(CRITIC_HEAD_WIDTH);
        assert_eq!(head.in_ch1, [384, 768, 1536]);
        assert_eq!(head.in_ch2, 1024);
        let layers = head.layers();
        assert_eq!(layers.len(), 8);
        assert_eq!(layers[0].matrix(), (512, 384 * 9));
        assert_eq!(layers[7].prefix, "out");
        let init = critic_head_init(&head, 7);
        assert_eq!(init.len(), 32);
        // u, v unit vectors; weights inside the kaiming bound
        for (k, shape, v) in &init {
            if k.ends_with("weight_u") || k.ends_with("weight_v") {
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!((n - 1.0).abs() < 1e-5, "{k}");
            }
            if k.ends_with("weight_orig") {
                let fan: usize = shape[1..].iter().product();
                let b = 1.0 / (fan as f32).sqrt();
                assert!(v.iter().all(|x| x.abs() <= b));
            }
        }
        assert_eq!(init, critic_head_init(&head, 7));
        assert_ne!(init, critic_head_init(&head, 8));
    }

    #[test]
    fn the_trunk_has_the_released_parameter_count() {
        let n: usize = ConvNextConfig::xxlarge()
            .tensor_shapes()
            .iter()
            .map(|(_, s)| s.iter().product::<usize>())
            .sum();
        // The `visual.*` tensors of open_clip_pytorch_model.bin @ 9f3e8ee.
        assert_eq!(n, 846_543_744);
    }

    #[test]
    fn critic_level_geometry() {
        // 1024² input: stride-4 stage 0 → 256 → (257 → 128 → 127 → 64)
        assert_eq!(critic_level_size(256, 256), Some((64, 64)));
        assert_eq!(critic_level_size(2, 2), None);
    }

    #[test]
    fn bce_matches_the_closed_form() {
        for (x, t) in [(0.3f64, 0.8), (-2.0, 0.0), (5.0, 0.8), (-30.0, 0.8)] {
            let s = 1.0 / (1.0 + (-x).exp());
            let want = -(t * s.ln() + (1.0 - t) * (1.0 - s).ln());
            assert!((bce_with_logits(x, t) - want).abs() < 1e-9, "{x} {t}");
        }
    }

    #[test]
    fn vgg_slices_follow_torchvision_indices() {
        let s = vgg_slices(VGG16_WIDTHS);
        assert_eq!(s.len(), 5);
        assert_eq!(
            s[0],
            [VggOp::ConvRelu(0, 3, 64), VggOp::ConvRelu(2, 64, 64)]
        );
        assert_eq!(s[1][0], VggOp::MaxPool);
        assert_eq!(s[1][1], VggOp::ConvRelu(5, 64, 128));
        assert_eq!(s[4][3], VggOp::ConvRelu(28, 512, 512));
    }

    #[test]
    fn degradation_options_override_and_refuse() {
        let c = parse_degradation(Some(&json!({
            "stage2_scale": 2,
            "resize_range2": [0.5, 1.5],
            "kernel_list": ["iso"],
            "kernel_prob": [1.0],
        })))
        .unwrap();
        assert_eq!(c.stage2_scale, 2.0);
        assert_eq!(c.stage2.resize_range, (0.5, 1.5));
        assert_eq!(c.kernel1.kernel_list, [KernelType::Iso]);
        assert_eq!(c.kernel2.kernel_list.len(), 6);
        assert!(parse_degradation(Some(&json!({"queue_size": 256}))).is_err());
        assert!(parse_degradation(Some(&json!({"resize_back": false}))).is_err());
    }

    #[test]
    fn checkpoint_state_round_trips() {
        let s = RestoreCheckpointState {
            step: 3,
            epoch: 2,
            batches_consumed: 1,
            scheduler_step: 3,
            lr: 1e-5,
            lr_critic: 2e-5,
            nan_count: 0,
            losses: [1.0, 0.5, 0.25, 0.125, 0.75],
            artifact: "full".into(),
            ema_decay: Some(0.999),
            seed: u64::MAX,
            backend: "mlx".into(),
            state_identity: "a".into(),
            data_identity: "b".into(),
            dataset_fingerprint: "c".into(),
            base_identity: "d".into(),
        };
        assert_eq!(RestoreCheckpointState::from_json(&s.to_json()).unwrap(), s);
        let mut gen = s.to_json();
        gen["format"] = json!(super::super::train::CHECKPOINT_FORMAT);
        assert!(RestoreCheckpointState::from_json(&gen).is_err());
    }
}
