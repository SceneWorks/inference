//! S4 — the LTX **latent upsamplers**. Port of the `mlx_video` reference `models/ltx/upsampler.py`
//! (`LatentUpsampler` + `upsample_latents`) — and, for LTX-2.5, of upstream
//! `ltx_core/model/upsampler/model.py` at `Lightricks/LTX-2` @ `d1511477` (v1.2.0) — gated against
//! both (`tests/upsampler_parity.rs`, real checkpoints).
//!
//! Two shipped variants, distinguished by [`LatentUpsamplerMode`] (sc-18773):
//!
//! * **spatial ×2** — LTX-2.3 `upsampler.safetensors` and LTX-2.5
//!   `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` (`mid_channels` 1024). `H,W → 2H,2W`;
//!   the frame count is untouched. The 2.5 file is a drop-in for the 2.3 architecture.
//! * **temporal ×2** — LTX-2.5 `ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors`
//!   (`mid_channels` 512). `Conv3d(mid, 2·mid) + PixelShuffleND(1)` on the frame axis, then the
//!   **leading frame is dropped**, so `F → 2F−1`; `H,W` untouched. Dropping is not an off-by-one
//!   trim: latent frame 0 encodes a single pixel frame, and `2·(8k+1) − 1 = 16k+1` is what keeps
//!   LTX's `n % 8 == 1` latent-frame invariant.
//!
//! Which variant a checkpoint is comes from the **rank of `upsampler.0.weight`** (4 → Conv2d
//! spatial, 5 → Conv3d temporal), not from a file name and not from a config: SceneWorks-converted
//! 2.3 trees carry no `__metadata__` at all. `gen_core::ltx_checkpoint::LatentUpsamplerConfig` reads
//! the declared config where one exists, and
//! [`LatentUpsampler::assert_matches_config`] cross-checks the two —
//! [`LatentUpsampler::from_checkpoint`] is the path-taking constructor that runs both, so the
//! cross-check is on the production load path rather than only in the parity tests.
//!
//! Sits between the two-stage distilled denoise (S5): stage-1 runs at half resolution, its latents
//! are upsampled 2× spatially here, then stage-2 refines at full resolution. The reference loads the
//! `ltx-2-spatial-upscaler-x2` checkpoint **bf16** and runs the whole path bf16 (weights, latents,
//! and the un-/re-normalize `latents_mean`/`latents_std` are all bf16) — so this matches that exactly
//! rather than the VAE's f32 (which is its own gated choice). The one f32 island is `GroupNorm3d`,
//! which the reference upcasts to f32 internally and casts back — replicated here verbatim.
//!
//! Architecture (`num_blocks_per_stage = 4`, structure-from-weights):
//!   `initial_conv 128→mid` → `initial_norm` → SiLU → 4× pre-`ResBlock3D` → resampler → 4×
//!   post-`ResBlock3D` → `final_conv mid→128`. I/O is channels-first `NCFHW`, transposed to `NFHWC`
//!   only for the conv ops. The resampler is the frame-by-frame `Conv2d mid→4·mid` +
//!   `PixelShuffle2D(2)` (spatial) or the `Conv3d mid→2·mid` + frame-axis `PixelShuffle1D(2)` +
//!   leading-frame drop (temporal).
//!
//! Reference quirks carried over verbatim:
//!  - Conv weights are on-disk **PyTorch** layout (Conv3d `[O,I,D,H,W]`, Conv2d `[O,I,H,W]`) — unlike
//!    the VAE's pre-transposed MLX layout — so they're transposed to MLX `[O,…,I]` at load. The
//!    Conv2d lives under `upsampler.0.*` on disk (the reference renames it `upsampler.conv.*`).
//!  - `GroupNorm3d` (32 groups, eps 1e-5) reshapes `NFHWC → (N, F·H·W, groups, C/groups)`, takes
//!    mean/var over the spatial+within-group axes `(1, 3)`, normalizes, then scale/shifts — all in
//!    **f32**, output cast back to the input dtype.
//!  - `ResBlock3D` applies its SiLU **after** the residual add (`silu(conv2(norm)→ + residual)`).
//!  - `PixelShuffle2D` is the channels-last `(N,H,W,C·r²) → (N,H·r,W·r,C)` rearrange.
//!
//! The parity gate honors "divergence is not rounding": every op here is the same mlx op the
//! reference uses at the same dtype, so a >1% gap would be a real bug, not bf16 noise.

use mlx_rs::ops::{add, divide, mean_axes, multiply, subtract, var_axes};
use mlx_rs::{Array, Dtype};

use mlx_gen::gen_core::ltx_checkpoint::{
    upsampled_latent_frames, LatentUpsamplerConfig, LatentUpsamplerMode, LtxCheckpointMetadata,
};
use mlx_gen::nn::{conv2d, conv3d, silu};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// `GroupNorm3d` group count (`GroupNorm3d(32, …)` throughout the reference).
const GROUPS: i32 = 32;
/// `GroupNorm3d` epsilon (`GroupNorm3d.__init__(eps=1e-5)`).
const NORM_EPS: f32 = 1e-5;

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// A bias-carrying conv whose on-disk weight is **PyTorch** layout; transposed to MLX `[O,…,I]` at
/// load. Stride 1, padding 1 (kernel 3, "same"), matching every conv in the upsampler.
struct Conv {
    w: Array,
    b: Array,
    /// `true` → 3-D (`NDHWC`), `false` → 2-D (`NHWC`).
    is_3d: bool,
}

impl Conv {
    /// `is_3d` picks the PyTorch→MLX transpose: 3-D `[O,I,D,H,W]→[O,D,H,W,I]` (`0,2,3,4,1`); 2-D
    /// `[O,I,H,W]→[O,H,W,I]` (`0,2,3,1`). Weights stay at their on-disk dtype (bf16).
    fn load(w: &Weights, prefix: &str, is_3d: bool) -> Result<Self> {
        let raw = w.require(&format!("{prefix}.weight"))?;
        let weight = if is_3d {
            raw.transpose_axes(&[0, 2, 3, 4, 1])?
        } else {
            raw.transpose_axes(&[0, 2, 3, 1])?
        };
        Ok(Self {
            w: weight,
            b: w.require(&format!("{prefix}.bias"))?.clone(),
            is_3d,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        if self.is_3d {
            conv3d(x, &self.w, Some(&self.b), (1, 1, 1), (1, 1, 1))
        } else {
            conv2d(x, &self.w, Some(&self.b), 1, 1)
        }
    }
}

/// `GroupNorm3d` — group norm over `NFHWC` computed in **f32** then cast back. Mirrors the reference
/// reshape `(N, F·H·W, groups, C/groups)` + mean/var over `(1, 3)` exactly (mlx `mean`/`var`, ddof 0).
struct GroupNorm {
    weight: Array,
    bias: Array,
}

impl GroupNorm {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: w.require(&format!("{prefix}.weight"))?.clone(),
            bias: w.require(&format!("{prefix}.bias"))?.clone(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let input_dtype = x.dtype();
        let x = x.as_dtype(Dtype::Float32)?;
        let sh = x.shape(); // (n, f, h, w, c)
        let (n, f, h, wd, c) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let gs = c / GROUPS;
        let xr = x.reshape(&[n, f * h * wd, GROUPS, gs])?;
        let mean = mean_axes(&xr, &[1, 3], true)?;
        let var = var_axes(&xr, &[1, 3], true, None)?; // ddof 0 — matches mx.var
        let denom = add(&var, scalar(NORM_EPS))?.sqrt()?;
        let normed = divide(&subtract(&xr, &mean)?, &denom)?.reshape(&[n, f, h, wd, c])?;
        let wf = self.weight.as_dtype(Dtype::Float32)?;
        let bf = self.bias.as_dtype(Dtype::Float32)?;
        let out = add(&multiply(&normed, &wf)?, &bf)?;
        Ok(out.as_dtype(input_dtype)?)
    }
}

/// `PixelShuffle2D(r)` — channels-last `(N, H, W, C·r²) → (N, H·r, W·r, C)`.
fn pixel_shuffle_2d(x: &Array, r: i32) -> Result<Array> {
    let sh = x.shape(); // (n, h, w, c)
    let (n, h, wd, c) = (sh[0], sh[1], sh[2], sh[3]);
    if r == 0 || c % (r * r) != 0 {
        return Err(Error::Msg(format!(
            "ltx upsampler: PixelShuffle({r}) cannot divide {c} channels"
        )));
    }
    let out_c = c / (r * r);
    let x = x.reshape(&[n, h, wd, out_c, r, r])?;
    let x = x.transpose_axes(&[0, 1, 4, 2, 5, 3])?;
    Ok(x.reshape(&[n, h * r, wd * r, out_c])?)
}

/// `PixelShuffleND(1)` — channels-last `(N, F, H, W, C·r) → (N, F·r, H, W, C)`.
///
/// Upstream is `rearrange(x, "b (c p1) f h w -> b c (f p1) h w")`: the channel axis decomposes
/// `c`-major / `p1`-minor, and the shuffled frame index is `f·r + p1`. Channels-last that is a
/// `(…, C, r)` split of the last axis moved in front of `H`/`W` and folded into `F` — the `r` sub-axis
/// stays *minor* to `F`, which is what makes the two consecutive output frames come from the same
/// input frame.
fn pixel_shuffle_1d_frames(x: &Array, r: i32) -> Result<Array> {
    let sh = x.shape(); // (n, f, h, w, c)
    let (n, f, h, wd, c) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
    if r == 0 || c % r != 0 {
        return Err(Error::Msg(format!(
            "ltx upsampler: PixelShuffle1d({r}) cannot divide {c} channels"
        )));
    }
    let out_c = c / r;
    let x = x.reshape(&[n, f, h, wd, out_c, r])?;
    let x = x.transpose_axes(&[0, 1, 5, 2, 3, 4])?;
    Ok(x.reshape(&[n, f * r, h, wd, out_c])?)
}

/// Drop frame 0 of an `NFHWC` tensor (`x[:, 1:]`).
fn drop_leading_frame(x: &Array) -> Result<Array> {
    let frames = x.shape()[1];
    if frames < 2 {
        return Err(Error::Msg(format!(
            "ltx latent upsampler: temporal resampler produced {frames} frame(s); the leading-frame \
             drop needs at least 2"
        )));
    }
    let idx: Vec<i32> = (1..frames).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[frames - 1]), 1)?)
}

/// The resampler stage — the one place the two shipped checkpoints differ.
enum Resampler {
    /// Frame-by-frame 2× spatial upsample: fold frames into the batch, one `Conv2d mid→4·mid`,
    /// `PixelShuffle2D(2)`, unfold. Frame count unchanged.
    Spatial2x { conv: Conv },
    /// 2× temporal upsample: one `Conv3d mid→2·mid`, frame-axis `PixelShuffle1D(2)`, then drop the
    /// leading frame. `F → 2F−1`.
    Temporal2x { conv: Conv },
}

impl Resampler {
    /// Structure-from-weights: `{prefix}.0.weight` is rank 4 for the spatial Conv2d and rank 5 for
    /// the temporal Conv3d. (On disk the conv is `{prefix}.0.*` — the `torch.nn.Sequential` index —
    /// which the `mlx_video` reference renames `{prefix}.conv.*`.)
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        let key = format!("{prefix}.0");
        let raw = w.require(&format!("{key}.weight"))?;
        match raw.ndim() {
            4 => Ok(Resampler::Spatial2x {
                conv: Conv::load(w, &key, false)?,
            }),
            5 => Ok(Resampler::Temporal2x {
                conv: Conv::load(w, &key, true)?,
            }),
            n => Err(mlx_gen::Error::Msg(format!(
                "ltx latent upsampler: {key}.weight has rank {n}, expected 4 (spatial Conv2d) or \
                 5 (temporal Conv3d)"
            ))),
        }
    }

    fn mode(&self) -> LatentUpsamplerMode {
        match self {
            Resampler::Spatial2x { .. } => LatentUpsamplerMode::Spatial2x,
            Resampler::Temporal2x { .. } => LatentUpsamplerMode::Temporal2x,
        }
    }

    /// `NFHWC → NFHWC`.
    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape(); // (n, f, h, w, c)
        let (n, f, h, wd, c) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        match self {
            Resampler::Spatial2x { conv } => {
                let x = x.reshape(&[n * f, h, wd, c])?;
                let x = conv.forward(&x)?;
                let x = pixel_shuffle_2d(&x, 2)?;
                Ok(x.reshape(&[n, f, h * 2, wd * 2, c])?)
            }
            Resampler::Temporal2x { conv } => {
                let x = conv.forward(x)?; // (n, f, h, w, 2c)
                let x = pixel_shuffle_1d_frames(&x, 2)?; // (n, 2f, h, w, c)
                                                         // Drop the leading frame: latent frame 0 encodes ONE pixel frame, so its shuffled
                                                         // pair would duplicate it. `2f − 1` is also what preserves `n % 8 == 1`.
                drop_leading_frame(&x)
            }
        }
    }
}

/// `ResBlock3D` — `conv1 → norm1 → SiLU → conv2 → norm2`, then `SiLU(· + residual)`.
struct ResBlock {
    conv1: Conv,
    norm1: GroupNorm,
    conv2: Conv,
    norm2: GroupNorm,
}

impl ResBlock {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv1: Conv::load(w, &format!("{prefix}.conv1"), true)?,
            norm1: GroupNorm::load(w, &format!("{prefix}.norm1"))?,
            conv2: Conv::load(w, &format!("{prefix}.conv2"), true)?,
            norm2: GroupNorm::load(w, &format!("{prefix}.norm2"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let residual = x.clone();
        let h = self.conv1.forward(x)?;
        let h = self.norm1.forward(&h)?;
        let h = silu(&h)?;
        let h = self.conv2.forward(&h)?;
        let h = self.norm2.forward(&h)?;
        silu(&add(&h, &residual)?)
    }
}

/// An LTX latent upsampler — spatial ×2 or temporal ×2, selected by the loaded weights.
/// `num_blocks_per_stage` is read from the checkpoint (count of `res_blocks.{i}`), `mid_channels`
/// follows from the conv weights.
pub struct LatentUpsampler {
    initial_conv: Conv,
    initial_norm: GroupNorm,
    res_blocks: Vec<ResBlock>,
    upsampler: Resampler,
    post_upsample_res_blocks: Vec<ResBlock>,
    final_conv: Conv,
}

impl LatentUpsampler {
    /// Build from a loaded latent-upsampler checkpoint — LTX-2.3 `upsampler.safetensors`, LTX-2.5
    /// `…latent-spatial-upscaler-x2…`, or LTX-2.5 `…latent-temporal-upscaler-x2…`.
    pub fn from_weights(w: &Weights) -> Result<Self> {
        // Structure-from-weights with a **floor**: a stage whose `.0` key is absent is a truncated
        // checkpoint, not a zero-block network. Without this the `while` loop returns an empty
        // `Vec` and a shallower model runs silently against real weights.
        let load_stage = |stem: &str| -> Result<Vec<ResBlock>> {
            let mut blocks = Vec::new();
            let mut i = 0;
            while w.get(&format!("{stem}.{i}.conv1.weight")).is_some() {
                blocks.push(ResBlock::load(w, &format!("{stem}.{i}"))?);
                i += 1;
            }
            if blocks.is_empty() {
                return Err(Error::Msg(format!(
                    "ltx latent upsampler: no residual blocks under {stem}.* — the checkpoint is \
                     missing {stem}.0.conv1.weight"
                )));
            }
            Ok(blocks)
        };
        let res_blocks = load_stage("res_blocks")?;
        let post_upsample_res_blocks = load_stage("post_upsample_res_blocks")?;
        // Upstream builds both stages from the one `num_blocks_per_stage`, so they are the same
        // count by construction; a file where they differ has lost blocks from one side.
        if res_blocks.len() != post_upsample_res_blocks.len() {
            return Err(Error::Msg(format!(
                "ltx latent upsampler: res_blocks has {} block(s) but post_upsample_res_blocks has \
                 {} — both stages are built from the same num_blocks_per_stage",
                res_blocks.len(),
                post_upsample_res_blocks.len()
            )));
        }
        Ok(Self {
            initial_conv: Conv::load(w, "initial_conv", true)?,
            initial_norm: GroupNorm::load(w, "initial_norm")?,
            res_blocks,
            upsampler: Resampler::load(w, "upsampler")?,
            post_upsample_res_blocks,
            final_conv: Conv::load(w, "final_conv", true)?,
        })
    }

    /// Build from a latent-upsampler `.safetensors` **file**, cross-checking the declared config.
    ///
    /// This is the only path-taking constructor, and the one every production load site uses:
    /// [`Self::from_weights`] reads the structure out of the tensors, and when the file carries a
    /// `__metadata__["config"]` object this reads that too and runs
    /// [`Self::assert_matches_config`] before returning. Loading through a bare `from_weights` on a
    /// stamped file would let the rank silently win over a config that disagrees — which is the
    /// whole point of having two authorities.
    ///
    /// A file with no `__metadata__["config"]` (every SceneWorks-converted LTX-2.3 tree) simply
    /// skips the cross-check; that is a checkpoint that declares nothing, not one that disagrees.
    pub fn from_checkpoint(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let w = Weights::from_file(path)?;
        let up = Self::from_weights(&w)?;
        let meta = LtxCheckpointMetadata::from_file(path)?;
        if meta.config().is_some() {
            let config = LatentUpsamplerConfig::from_metadata(path, &meta)?;
            up.assert_matches_config(&config)?;
        }
        Ok(up)
    }

    /// Which axis this checkpoint rescales, as read from its weights.
    pub fn mode(&self) -> LatentUpsamplerMode {
        self.upsampler.mode()
    }

    /// Latent frame count this upsampler produces from `frames` input frames.
    pub fn output_frames(&self, frames: usize) -> Result<usize> {
        Ok(upsampled_latent_frames(frames, self.mode())?)
    }

    /// Assert the structure read from the weights agrees with the config the checkpoint declares.
    ///
    /// Both are authorities on the same fact and they are read independently, so a disagreement
    /// means one of them is being misread — never something to paper over by preferring one.
    /// Checkpoints that carry no `__metadata__` (SceneWorks-converted LTX-2.3 trees) simply never
    /// reach this.
    pub fn assert_matches_config(&self, config: &LatentUpsamplerConfig) -> Result<()> {
        let declared = config.mode()?;
        if declared != self.mode() {
            return Err(Error::Msg(format!(
                "ltx latent upsampler: config declares {declared:?} but the weights are {:?}",
                self.mode()
            )));
        }
        // Both stages are the same length by the time `from_weights` returns, so one comparison
        // covers `num_blocks_per_stage`.
        let mismatch = |what: &str, declared: u64, loaded: i64| -> Error {
            Error::Msg(format!(
                "ltx latent upsampler: config declares {what}={declared} but the weights carry \
                 {loaded}"
            ))
        };
        let loaded_blocks = self.res_blocks.len() as i64;
        if config.num_blocks_per_stage as i64 != loaded_blocks {
            return Err(mismatch(
                "num_blocks_per_stage",
                config.num_blocks_per_stage,
                loaded_blocks,
            ));
        }
        // `initial_conv` is MLX layout `[mid, D, H, W, in]` after the load transpose; `final_conv`
        // is `[in, D, H, W, mid]`. Both are checked, so a swapped pair cannot cancel out.
        let initial = self.initial_conv.w.shape();
        let final_ = self.final_conv.w.shape();
        for (what, declared, loaded) in [
            ("in_channels", config.in_channels, i64::from(initial[4])),
            ("in_channels", config.in_channels, i64::from(final_[0])),
            ("mid_channels", config.mid_channels, i64::from(initial[0])),
            ("mid_channels", config.mid_channels, i64::from(final_[4])),
        ] {
            if declared as i64 != loaded {
                return Err(mismatch(what, declared, loaded));
            }
        }
        Ok(())
    }

    /// Upsample a channels-first `NCFHW` latent.
    ///
    /// [`LatentUpsamplerMode::Spatial2x`] → `NCF(2H)(2W)`; [`LatentUpsamplerMode::Temporal2x`] →
    /// `NC(2F−1)HW`.
    pub fn forward(&self, latent_ncfhw: &Array) -> Result<Array> {
        let frames_in = latent_ncfhw.shape()[2] as usize;
        let frames_out = self.output_frames(frames_in)?;
        // NCFHW → NFHWC for the channels-last conv ops.
        let mut x = latent_ncfhw.transpose_axes(&[0, 2, 3, 4, 1])?;
        x = self.initial_conv.forward(&x)?;
        x = self.initial_norm.forward(&x)?;
        x = silu(&x)?;
        for b in &self.res_blocks {
            x = b.forward(&x)?;
        }
        x = self.upsampler.forward(&x)?;
        // Checked here, on the resampler's own output, rather than on the returned tensor: this is
        // the only stage that moves the frame axis, so a wrong count localises to it.
        let produced = x.shape()[1] as usize;
        if produced != frames_out {
            return Err(Error::Msg(format!(
                "ltx latent upsampler: {:?} resampler turned {frames_in} latent frames into \
                 {produced}, expected {frames_out}",
                self.mode()
            )));
        }
        for b in &self.post_upsample_res_blocks {
            x = b.forward(&x)?;
        }
        x = self.final_conv.forward(&x)?;
        // NFHWC → NCFHW.
        Ok(x.transpose_axes(&[0, 4, 1, 2, 3])?)
    }
}

/// `upsample_latents` / upstream `upsample_video` — un-normalize (`latent·std + mean`), upsample
/// (spatially or temporally, per the checkpoint), re-normalize
/// (`(latent − mean)/std`). `latent_mean`/`latent_std` are the VAE `per_channel_statistics` (bf16,
/// shape `[C]`), reshaped to broadcast over `NCFHW`.
pub fn upsample_latents(
    latent: &Array,
    upsampler: &LatentUpsampler,
    latent_mean: &Array,
    latent_std: &Array,
) -> Result<Array> {
    let mean = latent_mean.reshape(&[1, -1, 1, 1, 1])?;
    let std = latent_std.reshape(&[1, -1, 1, 1, 1])?;
    let unnorm = add(&multiply(latent, &std)?, &mean)?;
    let up = upsampler.forward(&unnorm)?;
    Ok(divide(&subtract(&up, &mean)?, &std)?)
}
