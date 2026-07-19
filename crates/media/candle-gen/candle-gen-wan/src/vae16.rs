//! The **Wan 2.1 `AutoencoderKLWan`** (z16, stride 4×8×8) — the temporal VAE used by **both** A14B
//! MoE variants (`wan2_2_t2v_14b` / `wan2_2_i2v_14b`, sc-5174). Decode (always) + encode (I2V
//! channel-concat conditioning), ported from the diffusers checkpoint
//! (`Wan-AI/Wan2.2-T2V-A14B-Diffusers/vae`).
//!
//! Distinct from the 5B's z48 [`crate::vae`] `AutoencoderKLWan` on three structural axes (`vae/config.json`):
//!  - **z16, base_dim 96** (`dim_mult [1,2,4,4]`) vs the z48 base 256.
//!  - **non-residual** — no `DupUp3D`/`AvgDown3D` block-level shortcuts (the z48's `is_residual`).
//!  - **no spatial patchify** — `conv_out` emits 3 channels directly (the z48 unpatchifies a 2×2 grid),
//!    so the spatial scale is **8×** (3 up/down stages), not 16×.
//!  - diffusers names the up-sampler `up_blocks.N.upsamplers.0.…` (plural) vs the z48's `upsampler.…`.
//!
//! It reuses the proven z48 building blocks (`ChanNorm`, `Conv2dW`, `Resnet`, `MidAttn`,
//! `Upsampler`, `causal`) and the from-scratch [`CausalConv3d`](crate::conv3d) — only the encoder's
//! stride-2 spatial/temporal downsamplers are new here. Decode **streams one latent frame at a time**
//! (the sc-5176 fix, bit-equivalent to a single pass via the causal `feat_cache`); encode mirrors the
//! diffusers **chunked** causal encode (frame 0 alone, then 4-frame chunks).
//!
//! **Dtype (sc-12818):** the decoder runs in the loaded weight dtype ([`WanVae16::dtype`]) — the A14B
//! loads it **bf16**, ~halving the z16 decode's fixed VRAM floor (weights + the un-tileable f32
//! activations) so the 1280×720/81f A14B decode fits a 24 GiB card; the 5B z48 path and the CPU tests
//! stay f32. The channel-L2 `ChanNorm` and the `MidAttn` softmax reduce in f32 regardless (bf16's
//! 8-bit mantissa is too coarse for those sums), applying the result back in the working dtype so the
//! activation itself stays bf16 — the UMT5 RMS-norm idiom (sc-12778). candle's **CPU** backend has no
//! bf16 matmul, so the bf16 decode is CUDA-only; CPU tests exercise the f32 reference forward.

use std::sync::Mutex;

use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tiling::{SpatialTiling, TileCandidates, TilingConfig, VaeTiling};
use candle_gen::vae_tiling;

use crate::config::{Vae16Config, LATENTS16_MEAN, LATENTS16_STD};
use crate::conv3d::{chunked_conv2d, CausalConv3d, Ctx};
use crate::vae::{causal, ChanNorm, Conv2dW, MidAttn, Resnet, Upsampler};

/// One z16 decoder up-stage: residual blocks then an optional spatial/temporal upsampler (no `Dup`
/// residual — the z16 VAE is non-residual).
struct UpBlock {
    resnets: Vec<Resnet>,
    upsampler: Option<Upsampler>,
}

impl UpBlock {
    fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let mut h = x.clone();
        for r in &self.resnets {
            h = r.forward(&h, ctx)?;
        }
        if let Some(up) = &self.upsampler {
            h = up.forward(&h, ctx)?;
        }
        Ok(h)
    }

    fn reset_cache(&self) {
        for r in &self.resnets {
            r.reset_cache();
        }
        if let Some(up) = &self.upsampler {
            up.reset_cache();
        }
    }
}

/// Encoder spatial 2× downsample: `ZeroPad2d((0,1,0,1))` per frame + a stride-2 3×3 conv (the diffusers
/// `WanResample` `resample.1`). Operates per-frame (no temporal cache).
struct SpatialDown {
    w: Tensor,
    b: Tensor, // [1, O, 1, 1]
}

impl SpatialDown {
    fn load(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        crate::quant::guard_dense(&vb)?;
        Ok(Self {
            w: vb.get((out_c, in_c, 3, 3), "weight")?.contiguous()?,
            b: vb.get(out_c, "bias")?.reshape((1, out_c, 1, 1))?,
        })
    }

    /// `x`: `[B,C,T,H,W]` → `[B,C,T,H/2,W/2]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let merged = x
            .permute((0, 2, 1, 3, 4))?
            .reshape((b * t, c, h, w))?
            .contiguous()?;
        // ZeroPad2d((left,right,top,bottom)) = (0,1,0,1): pad right (W, dim 3) + bottom (H, dim 2).
        let padded = merged.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?;
        // im2col-chunked (stride-2): a hi-res I2V conditioning frame's first spatial down (96-ch 3×3 at
        // 720×1280 ⇒ ~199M-elem im2col) would otherwise hit candle's conv2d corruption band (sc-12773).
        let y = chunked_conv2d(&padded, &self.w, 0, 2)?.broadcast_add(&self.b)?;
        let (_, oc, oh, ow) = y.dims4()?;
        y.reshape((b, t, oc, oh, ow))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()
    }
}

/// Encoder temporal 2× downsample (`time_conv`, a causal stride-2 `(3,1,1)` conv). Chunked like the
/// diffusers `WanResample` `downsample3d`: the **first chunk** stashes its last frame and passes
/// through un-downsampled; later chunks prepend the stash as the single causal left-context frame
/// (`causal = kt − st = 1`, fully covered by the cache → no zero-pad) and run the stride-2 conv.
struct TemporalDown {
    w: Tensor,                    // [O, I, 3, 1, 1]
    b: Tensor,                    // [1, O, 1, 1, 1]
    cache: Mutex<Option<Tensor>>, // previous chunk's last frame
}

impl TemporalDown {
    fn load(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        crate::quant::guard_dense(&vb)?;
        Ok(Self {
            w: vb.get((out_c, in_c, 3, 1, 1), "weight")?.contiguous()?,
            b: vb.get(out_c, "bias")?.reshape((1, out_c, 1, 1, 1))?,
            cache: Mutex::new(None),
        })
    }

    fn reset_cache(&self) {
        // sc-9015 / F-031: recover from a poisoned lock (reset-on-miss streaming cache).
        *candle_gen::lock_recover(&self.cache) = None;
    }

    fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let last = x.narrow(2, x.dim(2)? - 1, 1)?.contiguous()?;
        if ctx.first_chunk {
            // First chunk: passthrough (no temporal downsample), stash the last frame for next chunk.
            // sc-9015 / F-031: recover from a poisoned lock (overwrite-on-miss streaming cache).
            *candle_gen::lock_recover(&self.cache) = Some(last);
            return Ok(x.clone());
        }
        // sc-9015 / F-031: recover from a poisoned lock; the `.expect` below is on the cached
        // `Option`, not the lock (a warmed cache is a real precondition on the non-first chunk).
        let prev = candle_gen::lock_recover(&self.cache)
            .clone()
            .expect("TemporalDown: non-first chunk needs a warmed cache");
        let xcat = Tensor::cat(&[&prev, x], 2)?; // T+1 frames; cache supplies the 1 causal context frame
        let out = self.strided_conv(&xcat)?;
        *candle_gen::lock_recover(&self.cache) = Some(last);
        Ok(out)
    }

    /// Stride-2, kernel-3 temporal conv over `[B,C,Tc,H,W]` (1×1 spatial) → `[B,O,(Tc-3)/2+1,H,W]`.
    /// Three taps `out[o] = Σ_k W[:,:,k]·x[2o+k]`, each a per-frame 1×1 conv2d, summed.
    fn strided_conv(&self, xcat: &Tensor) -> Result<Tensor> {
        let (b, c, tc, h, w) = xcat.dims5()?;
        let out_t = (tc - 3) / 2 + 1;
        let dev = xcat.device();
        let mut acc: Option<Tensor> = None;
        for k in 0..3 {
            let idx: Vec<u32> = (0..out_t).map(|o| (2 * o + k) as u32).collect();
            let sel = Tensor::from_vec(idx, out_t, dev)?;
            let frames = xcat.index_select(&sel, 2)?; // [B,C,out_t,H,W]
            let merged = frames
                .permute((0, 2, 1, 3, 4))?
                .reshape((b * out_t, c, h, w))?
                .contiguous()?;
            let wk = self.w.narrow(2, k, 1)?.squeeze(2)?.contiguous()?; // [O,I,1,1]
                                                                        // im2col-chunked for parity with the other VAE convs (1×1 ⇒ tiny im2col at these dims, so in
                                                                        // practice a single un-chunked pass; keeps every z16 encoder conv uniformly safe, sc-12773).
            let yk = chunked_conv2d(&merged, &wk, 0, 1)?;
            acc = Some(match acc {
                Some(a) => (a + yk)?,
                None => yk,
            });
        }
        let y = acc.expect("kernel 3 has >= 1 tap");
        let (_, oc, oh, ow) = y.dims4()?;
        y.reshape((b, out_t, oc, oh, ow))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .broadcast_add(&self.b)
    }
}

/// One encoder down-stage entry: a residual block or a spatial/temporal downsample.
enum DownLayer {
    Res(Resnet),
    Down {
        spatial: SpatialDown,
        temporal: Option<TemporalDown>,
    },
}

impl DownLayer {
    fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        match self {
            DownLayer::Res(r) => r.forward(x, ctx),
            DownLayer::Down { spatial, temporal } => {
                let x = spatial.forward(x)?;
                match temporal {
                    Some(t) => t.forward(&x, ctx),
                    None => Ok(x),
                }
            }
        }
    }

    fn reset_cache(&self) {
        match self {
            DownLayer::Res(r) => r.reset_cache(),
            DownLayer::Down { temporal, .. } => {
                if let Some(t) = temporal {
                    t.reset_cache();
                }
            }
        }
    }
}

/// The z16 encoder (`conv_in → flat down_blocks → mid → norm/SiLU/conv_out`) → `2·z` moments. Chunked
/// causal: drive each chunk with [`Ctx::streaming`]; the convs carry their `feat_cache`.
struct Encoder {
    conv_in: CausalConv3d,
    down_blocks: Vec<DownLayer>,
    mid_resnet0: Resnet,
    mid_attn: MidAttn,
    mid_resnet1: Resnet,
    norm_out: ChanNorm,
    conv_out: CausalConv3d,
}

impl Encoder {
    fn new(cfg: &Vae16Config, vb: VarBuilder) -> Result<Self> {
        let b = cfg.base_dim;
        // dim_mult [1,2,4,4] → stage dims [96,192,384,384]; downsample after stages 0,1,2.
        let stage_dim = [b, b * 2, b * 4, b * 4];
        let temporal_down = [false, true, true];

        let conv_in = causal(3, b, (3, 3, 3), vb.pp("conv_in"))?;
        let mut down_blocks = Vec::new();
        let mut idx = 0usize;
        for (s, &out_d) in stage_dim.iter().enumerate() {
            let in_d = if s == 0 { b } else { stage_dim[s - 1] };
            for j in 0..cfg.num_res_blocks {
                let rin = if j == 0 { in_d } else { out_d };
                down_blocks.push(DownLayer::Res(Resnet::new(
                    rin,
                    out_d,
                    vb.pp("down_blocks").pp(idx),
                )?));
                idx += 1;
            }
            if s < 3 {
                let db = vb.pp("down_blocks").pp(idx);
                let spatial = SpatialDown::load(out_d, out_d, db.pp("resample").pp("1"))?;
                let temporal = if temporal_down[s] {
                    Some(TemporalDown::load(out_d, out_d, db.pp("time_conv"))?)
                } else {
                    None
                };
                down_blocks.push(DownLayer::Down { spatial, temporal });
                idx += 1;
            }
        }

        let mid = vb.pp("mid_block");
        let mid_dim = b * 4;
        Ok(Self {
            conv_in,
            down_blocks,
            mid_resnet0: Resnet::new(mid_dim, mid_dim, mid.pp("resnets").pp("0"))?,
            mid_attn: MidAttn::new(mid_dim, mid.pp("attentions").pp("0"))?,
            mid_resnet1: Resnet::new(mid_dim, mid_dim, mid.pp("resnets").pp("1"))?,
            norm_out: ChanNorm::new(mid_dim, vb.pp("norm_out"))?,
            conv_out: causal(mid_dim, 2 * cfg.z_dim, (3, 3, 3), vb.pp("conv_out"))?,
        })
    }

    fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x, ctx)?;
        for layer in &self.down_blocks {
            h = layer.forward(&h, ctx)?;
        }
        h = self.mid_resnet0.forward(&h, ctx)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_resnet1.forward(&h, ctx)?;
        let h = self.norm_out.forward(&h)?.silu()?;
        self.conv_out.forward(&h, ctx)
    }

    fn reset_cache(&self) {
        self.conv_in.reset_cache();
        for layer in &self.down_blocks {
            layer.reset_cache();
        }
        self.mid_resnet0.reset_cache();
        self.mid_resnet1.reset_cache();
        self.conv_out.reset_cache();
    }
}

/// Reject a conditioning-video temporal length the chunked causal encode cannot consume without
/// dropping frames (F-126, sc-11220). The encode reads frame 0 alone then 4-frame chunks, i.e.
/// exactly `1 + 4·k` frames; any other `t` leaves the trailing `(t - 1) % 4` frames unencoded.
fn require_aligned_encode_frames(t: usize) -> Result<()> {
    if t == 0 || !(t - 1).is_multiple_of(4) {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "WanVae16::encode: frame count must be 1 + 4·k (got {t}); the causal encode would \
             silently drop the trailing {} frame(s)",
            t.saturating_sub(1) % 4
        )));
    }
    Ok(())
}

/// The Wan 2.1 z16 VAE: a decoder (always) plus an optional encoder (I2V conditioning), with
/// per-channel latent normalization.
pub struct WanVae16 {
    mean: Tensor, // [1,16,1,1,1]
    std: Tensor,
    post_quant_conv: CausalConv3d,
    conv_in: CausalConv3d,
    mid_resnet0: Resnet,
    mid_attn: MidAttn,
    mid_resnet1: Resnet,
    up_blocks: Vec<UpBlock>,
    norm_out: ChanNorm,
    conv_out: CausalConv3d,
    encoder: Option<(Encoder, CausalConv3d)>, // (encoder, quant_conv)
    z_dim: usize,
    /// The dtype the weights loaded at (the [`VarBuilder`] dtype), i.e. the dtype the decode runs in.
    /// The A14B loads this **bf16** (sc-12818, ~halving the z16 decode's fixed VRAM floor); the 5B z48
    /// path and the CPU tests load f32. [`Self::unnormalize`] casts the (f32-computed) unnormalized
    /// latent to this so the decoder convs see a matching-dtype activation.
    dtype: DType,
}

impl WanVae16 {
    /// Build a **decode-only** z16 VAE from a diffusers `vae/` snapshot (T2V — no I2V conditioning).
    pub fn new(cfg: &Vae16Config, vb: VarBuilder) -> Result<Self> {
        Self::build(cfg, vb, false)
    }

    /// Build a z16 VAE **with the encoder** (I2V — the conditioning image's first-frame latent).
    pub fn new_with_encoder(cfg: &Vae16Config, vb: VarBuilder) -> Result<Self> {
        Self::build(cfg, vb, true)
    }

    fn build(cfg: &Vae16Config, vb: VarBuilder, with_encoder: bool) -> Result<Self> {
        let device = vb.device();
        let dtype = vb.dtype();
        let z = cfg.z_dim;
        let mean = Tensor::from_vec(LATENTS16_MEAN.to_vec(), (1, z, 1, 1, 1), device)?;
        let std = Tensor::from_vec(LATENTS16_STD.to_vec(), (1, z, 1, 1, 1), device)?;
        let post_quant_conv = causal(z, z, (1, 1, 1), vb.pp("post_quant_conv"))?;

        let dec = vb.pp("decoder");
        let b = cfg.base_dim;
        // Per-up-block resnet output dims base·[4,4,2,1]; the spatial resample halves channels into the
        // next block's input. temperal_upsample = reversed([false,true,true]) = [true,true,false].
        let resnet_out = [b * 4, b * 4, b * 2, b];
        let has_up = [true, true, true, false];
        let temporal = [true, true, false, false];
        let conv_in = causal(z, resnet_out[0], (3, 3, 3), dec.pp("conv_in"))?;

        let mid = dec.pp("mid_block");
        let mid_dim = b * 4;
        let mid_resnet0 = Resnet::new(mid_dim, mid_dim, mid.pp("resnets").pp("0"))?;
        let mid_attn = MidAttn::new(mid_dim, mid.pp("attentions").pp("0"))?;
        let mid_resnet1 = Resnet::new(mid_dim, mid_dim, mid.pp("resnets").pp("1"))?;

        let mut up_blocks = Vec::with_capacity(4);
        let mut block_in = resnet_out[0]; // conv_in output feeds up_block 0
        for i in 0..4 {
            let out_c = resnet_out[i];
            let ub = dec.pp("up_blocks").pp(i);
            let mut resnets = Vec::with_capacity(cfg.num_res_blocks + 1);
            let mut cur = block_in;
            for j in 0..(cfg.num_res_blocks + 1) {
                resnets.push(Resnet::new(cur, out_c, ub.pp("resnets").pp(j))?);
                cur = out_c;
            }
            let upsampler = if has_up[i] {
                let us = ub.pp("upsamplers").pp("0");
                let resample = Conv2dW::load(out_c, out_c / 2, 3, 1, us.pp("resample").pp("1"))?;
                Some(if temporal[i] {
                    Upsampler::Temporal {
                        time_conv: causal(out_c, out_c * 2, (3, 1, 1), us.pp("time_conv"))?,
                        resample,
                    }
                } else {
                    Upsampler::Spatial { resample }
                })
            } else {
                None
            };
            up_blocks.push(UpBlock { resnets, upsampler });
            block_in = out_c / 2; // the resample halves channels into the next block
        }

        let norm_out = ChanNorm::new(b, dec.pp("norm_out"))?;
        let conv_out = causal(b, cfg.out_channels, (3, 3, 3), dec.pp("conv_out"))?;

        let encoder = if with_encoder {
            Some((
                Encoder::new(cfg, vb.pp("encoder"))?,
                causal(2 * z, 2 * z, (1, 1, 1), vb.pp("quant_conv"))?,
            ))
        } else {
            None
        };

        Ok(Self {
            mean,
            std,
            post_quant_conv,
            conv_in,
            mid_resnet0,
            mid_attn,
            mid_resnet1,
            up_blocks,
            norm_out,
            conv_out,
            encoder,
            z_dim: z,
            dtype,
        })
    }

    /// The dtype the VAE weights loaded at (and the decode runs in) — **bf16** for the A14B (sc-12818),
    /// f32 for the CPU/test path. Exposed so a CUDA parity harness can assert the weights really loaded
    /// bf16 (the VRAM-floor win) before comparing the bf16 vs f32 decode.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Decode latents `[B,16,T,H,W]` → RGB frames `[B,3, 1+(T-1)·4, 8H, 8W]` in `[-1,1]`. **Streams one
    /// latent frame at a time** (sc-5176), carrying the causal `feat_cache` while bounding peak memory
    /// to ~one frame's activations — the 14B's heavier clips would otherwise OOM the
    /// VAE-decode stage exactly as the 5B did.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let z = self.unnormalize(z)?;
        let t_lat = z.dim(2)?;
        self.reset_caches();
        // Collect the per-frame decoded chunks and `cat` once (sc-9037): cat-ing onto a growing
        // accumulator each iteration re-copies every prior frame → O(T²) copy traffic and briefly
        // holds old+new. A single `Tensor::cat` at the end is O(T) and equivalent (same frames, same
        // order along the temporal axis).
        let mut chunks: Vec<Tensor> = Vec::with_capacity(t_lat);
        for i in 0..t_lat {
            let zi = z.narrow(2, i, 1)?.contiguous()?;
            chunks.push(self.decode_inner(&zi, &Ctx::streaming(i == 0))?);
        }
        self.reset_caches();
        assert!(!chunks.is_empty(), "decode needs >= 1 latent frame");
        Tensor::cat(&chunks, 2)?.clamp(-1f32, 1f32)
    }

    /// Decode with **spatial tiling** for the memory-bounded A14B decode (sc-12758) — the z16 twin of the
    /// z48 [`WanVae::decode_tiled`](crate::vae::WanVae::decode_tiled).
    ///
    /// Like the z48 path, [`decode`](Self::decode) already **streams one latent frame at a time**, so the
    /// temporal axis is memory-bounded (peak ≈ one frame's decode). The only remaining spike is a single
    /// **high-resolution frame** through the z16 decoder — GPU-measured at 42 GB for a 1280×720/81f A14B
    /// T2V decode (sc-12771), the sole thing keeping the 14B off a 24 GB card. Spatial tiling caps that.
    ///
    /// Shares the pure [`gen_core::tiling`](candle_gen::gen_core::tiling) geometry + the seam-free
    /// blend/stitch DRIVER ([`vae_tiling::decode_tiled`]) with the z48/LTX halves, but with the **z16**
    /// geometry (`WAN_Z16`: ×8 spatial — not the z48's ×16 — ×4 **causal** temporal): each spatial tile
    /// is decoded via the per-frame streaming `decode`, then trapezoidally blended into the full video.
    /// Because the z16 `MidAttn` is **global per-frame spatial attention**, a tile attends only within
    /// itself, so this is an *approximation* softened by the overlapping blend (seam-free ≈ PSNR ~35 dB,
    /// not bit-exact) — exactly the z48 tradeoff. Falls back to a single streaming `decode` when `cfg`
    /// does not fire for these dims. `cfg` is expected to carry **spatial** tiling only.
    pub fn decode_tiled(&self, z: &Tensor, cfg: &TilingConfig) -> Result<Tensor> {
        // The tile/narrow/blend/slice-accumulate/normalize DRIVER is the shared
        // `candle_gen::vae_tiling::decode_tiled`; what stays z16-specific is the `WAN_Z16` geometry and
        // the per-frame-streaming `decode` closure. With a spatial-only `cfg`, `plan.t` is a single
        // full-extent temporal tile, so each `decode` call streams the whole clip (temporal bound kept).
        //
        // Each tile is decoded in the VAE's working dtype (bf16 on the A14B, sc-12818 — where the
        // per-tile im2col/conv activations, the decode's dominant transient, get the VRAM win), then
        // upcast to f32 so the shared seam-blend accumulates against its f32 trapezoidal mask (the
        // accumulator is small vs. the per-tile activations, so f32 there costs little and keeps the
        // stitch precise). A no-op cast on the f32 path.
        vae_tiling::decode_tiled(WAN_Z16, "wan z16 vae", z, cfg, |tile| {
            self.decode(tile)?.to_dtype(DType::F32)
        })
    }

    /// **Free-aware budgeted** decode (sc-12758): derive the decoded output dims from the z16 latent
    /// geometry (×8 spatial, ×4 **causal** temporal ⇒ `out_f = 1 + (T_lat−1)·4`), pick a budgeted
    /// **spatial** tiling via [`auto_tiling_budgeted_wan_z16`] (which budgets against **FREE** VRAM, the
    /// sc-12734 resolver, and additionally forces tiling below the candle conv2d im2col-safe cap
    /// `WAN_Z16_VAE_IM2COL_SAFE_PX`), and run [`decode_tiled`](Self::decode_tiled) — or the streaming
    /// [`decode`](Self::decode) only when a single high-res frame both fits the budget **and** stays
    /// under the im2col cap (so a hi-res / low-frame-count decode on a large-VRAM card still tiles rather
    /// than silently corrupting). An over-budget decode returns a **catchable** error here instead of
    /// OOM-ing the worker. The z16 analogue of the z48
    /// [`WanVae::decode_budgeted`](crate::vae::WanVae::decode_budgeted).
    pub fn decode_budgeted(&self, z: &Tensor) -> Result<Tensor> {
        let (_b, _c, f, h, w) = z.dims5()?;
        let out_f = 1 + (f as i32 - 1) * WAN_Z16.temporal_scale; // causal ×4
        let out_h = h as i32 * WAN_Z16.spatial_scale; // ×8
        let out_w = w as i32 * WAN_Z16.spatial_scale;
        match auto_tiling_budgeted_wan_z16(out_h, out_w, out_f)? {
            Some(cfg) => self.decode_tiled(z, &cfg),
            None => self.decode(z),
        }
    }

    fn decode_inner(&self, z: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let mut h = self.post_quant_conv.forward(z, ctx)?;
        h = self.conv_in.forward(&h, ctx)?;
        h = self.mid_resnet0.forward(&h, ctx)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_resnet1.forward(&h, ctx)?;
        for ub in &self.up_blocks {
            h = ub.forward(&h, ctx)?;
        }
        let h = self.norm_out.forward(&h)?.silu()?;
        self.conv_out.forward(&h, ctx) // [B,3,T',8H,8W] — no unpatchify (z16 has no spatial patchify)
    }

    /// Encode a conditioning video `[B,3,T,H,W]` (`T = 1 + 4·k`, values in `[-1,1]`) → normalized latent
    /// `[B,16,T_lat,H/8,W/8]`. Mirrors the diffusers **chunked** causal encode (frame 0 alone, then
    /// 4-frame chunks; the convs carry their `feat_cache`), then `quant_conv` → take the posterior mean →
    /// `(μ − mean)/std`. Requires encoder weights ([`Self::new_with_encoder`]).
    pub fn encode(&self, video: &Tensor) -> Result<Tensor> {
        let (encoder, quant_conv) = self.encoder.as_ref().ok_or_else(|| {
            candle_gen::candle_core::Error::Msg("WanVae16: encode needs encoder weights".into())
        })?;
        let t = video.dim(2)?;
        // The chunked causal encode consumes exactly `1 + 4·(num_chunks-1)` frames (frame 0 alone,
        // then 4-frame chunks). For `t % 4 != 1` the trailing `(t - 1) % 4` frames would silently
        // vanish from the latent. All in-repo callers pre-align, but this method (and `Scail2Job`)
        // is `pub`, so reject the unaligned length with a typed error rather than dropping frames.
        require_aligned_encode_frames(t)?;
        // sc-12894: the encoder convs run in the VAE's working dtype (bf16 on the A14B since sc-12818);
        // the I2V conditioning video arrives f32, so cast it to match — the encode-side mirror of
        // [`unnormalize`]'s decode cast. A no-op on the f32 5B/test path; without it the bf16 encoder's
        // first conv2d hits a `dtype mismatch, lhs: F32, rhs: BF16` and every I2V render crashes.
        let video = video.to_dtype(self.dtype)?;
        let num_chunks = 1 + (t - 1) / 4;
        encoder.reset_cache();
        // Collect the per-chunk encoded features and `cat` once (sc-9037): cat-ing onto a growing
        // accumulator each iteration re-copies every prior chunk → O(T²) copy traffic. A single
        // `Tensor::cat` at the end is O(T) and equivalent (same chunks, same temporal order).
        let mut chunks: Vec<Tensor> = Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let chunk = if i == 0 {
                video.narrow(2, 0, 1)?
            } else {
                video.narrow(2, 1 + 4 * (i - 1), 4)?
            }
            .contiguous()?;
            chunks.push(encoder.forward(&chunk, &Ctx::streaming(i == 0))?);
        }
        encoder.reset_cache();
        assert!(!chunks.is_empty(), "encode needs >= 1 frame");
        let out = Tensor::cat(&chunks, 2)?;
        // quant_conv (1×1×1) over the full moments, take the mean (first z channels), normalize. The
        // moments run in the VAE dtype (bf16 on the A14B, sc-12818); upcast to f32 for the per-channel
        // normalize (`mean`/`std` are f32) so the returned conditioning latent stays f32 exactly as
        // before — a no-op cast on the f32 path.
        let moments = quant_conv.forward(&out, &Ctx::single_pass())?;
        let mu = moments.narrow(1, 0, self.z_dim)?.to_dtype(DType::F32)?;
        mu.broadcast_sub(&self.mean)?.broadcast_div(&self.std)
    }

    /// `z_pixel = z·std + mean` (the inverse of the encoder's per-channel normalize). The affine is
    /// computed in **f32** (`mean`/`std` are f32) for precision, then cast to the VAE's working
    /// [`dtype`](Self::dtype) so the decoder convs get a matching-dtype activation — bf16 for the A14B
    /// (sc-12818), a no-op f32 clone for the 5B/test path.
    fn unnormalize(&self, z: &Tensor) -> Result<Tensor> {
        z.to_dtype(DType::F32)?
            .broadcast_mul(&self.std)?
            .broadcast_add(&self.mean)?
            .to_dtype(self.dtype)
    }

    fn reset_caches(&self) {
        self.post_quant_conv.reset_cache();
        self.conv_in.reset_cache();
        self.mid_resnet0.reset_cache();
        self.mid_resnet1.reset_cache();
        for ub in &self.up_blocks {
            ub.reset_cache();
        }
        self.conv_out.reset_cache();
    }
}

// --- sc-12758: free-aware budgeted z16 spatial-tiling for the A14B decode ------------------------
//
// **sc-12818 — tiling is NOT the A14B memory lever.** Re-measured with the driver's accurate
// concurrent-live peak (`CU_MEMPOOL_ATTR_USED_MEM_HIGH`, which the nvidia-smi poll under-sampled ~2×),
// the sequential A14B q4 @1280×720/81f decode's true peak was a **FIXED ~30.1 GiB floor, independent of
// the VAE tile budget** (budget 20 → 30.1; budget 10 / 192 px tile → still 30.1) — the floor is the VAE
// weights + the un-tileable f32 decode activations, not something spatial tiling shrinks. The lever that
// lands it under 24 GiB is running the z16 VAE **bf16** (`WanVae16::dtype`), which ~halves that floor.
// This tiling machinery stays as the candle conv2d **im2col-safety** cap (`WAN_Z16_VAE_IM2COL_SAFE_PX`)
// and the big-VRAM correctness bound — not as the 24 GiB fit mechanism.
//
// The z16 twin of the z48 budgeted tiler (`crate::vae`, sc-7111/sc-12734). It reuses the shared
// budgeted-tiling DRIVER + free-aware budget resolver + selector in `candle_gen::vae_tiling`, supplying
// only the **z16-specific** geometry, cost constants, and candidate grid. Two axes make z16 its OWN
// path, not a reuse of the z48 `WAN22` constants (the story's explicit ask):
//   - **×8 spatial** (3 up-stages), not the z48's ×16 (4 up-stages + 2×2 unpatchify): for a given output
//     resolution the z16 latent is 2× larger per side, and the decoder is far narrower (base_dim 96,
//     stage dims ≤384 vs the z48's base 256, ≤1024) — so the per-output-pixel activation is smaller.
//   - **16-ch, non-residual, no patchify** — a different activation profile, so `FRAME` is recalibrated.
//
// **Structural match to z48** (why this is a spatial-only mirror): the candle `decode` STREAMS one
// latent frame at a time, so the temporal axis is already memory-bounded — peak ≈ one frame's decode,
// not the whole clip. So the candidate grid carries **no temporal candidates** and the cost model's
// per-tile term is per-output-**pixel** (one streamed frame's activations), not per-voxel.

/// z16 tiling geometry for the **candle causal streaming decode**: ×8 spatial, ×4 **causal** temporal
/// (`out_f = 1 + (f−1)·4`), `full_res_channels 96` (the materialized full-res res-block width before
/// `conv_out` drops 96→3). Distinct from the shared [`VaeTiling::WAN`], which is the **mlx** z16
/// preset and is **non-causal** (`out_f = f·4`) — the candle decode is causal, so the plan's `out_f`
/// must match `decode`'s frame count. Kept local (not a new `gen_core` preset) to keep this the z16's
/// own path with zero blast radius on the shared contract.
const WAN_Z16: VaeTiling = VaeTiling {
    spatial_scale: 8,
    temporal_scale: 4,
    causal_temporal: true,
    full_res_channels: 96,
};

const GIB_F64: f64 = 1024.0 * 1024.0 * 1024.0;
/// Env override read by the shared [`vae_tiling::free_aware_safe_budget_gib`] resolver — the SAME
/// deterministic injection point as the z48 tiler (only one Wan VAE runs per process), per sc-12758.
const WAN_Z16_VAE_BUDGET_ENV: &str = "WAN_VAE_BUDGET_GIB";
/// Fraction of **FREE** VRAM treated as safe (sc-12734). The decode runs *after* the denoise (and in
/// the sequential path *after* the experts+TE are offloaded), so the fraction is applied to what is
/// genuinely free (`total − resident`), not `0.85×TOTAL`. 0.85 matches the z48 / ltx / seedvr2 headroom.
const WAN_Z16_VAE_BUDGET_SAFE_FRAC: f64 = 0.85;
/// Fallback budget when neither the env override nor `nvidia-smi` yields a value.
const WAN_Z16_VAE_DEFAULT_BUDGET_GIB: f64 = 16.0;

// Cost-model constants — **CUDA-calibrated (sc-12758)** from real-weight peak-VRAM anchors measured by
// `tests/vae16_decode_sweep.rs` on this RTX PRO 6000 Blackwell (sm_120, CUDA 12.9, f32, device-level
// `nvidia-smi` peak, ~0.8 GB baseline) over the real Wan2.2-T2V-A14B z16 VAE. Anchors (output 1280×720×81
// / largest spatial tile → measured decode peak) — see the `wan_z16_decode_peak_matches_cuda_anchors`
// test:
//   single-pass (full 1280×720 frame) → 48.10 GiB   (the untiled A14B T2V spike; cf. sc-12771's 42 GB
//                                                     render-context delta — the isolated decode peaks higher)
//   tiled 512px                       → 19.82 GiB
//   tiled 256px                       →  5.85 GiB    (accumulator-dominated ⇒ the ACCUM floor)
// Because `decode` STREAMS one latent frame at a time, the per-tile activation spike scales with ONE
// output frame's area (`tile_h·tile_w`), not the whole tile's volume — so the model is `ACCUM·out_vox +
// FRAME·frame_px`. `ACCUM` is the full-output accumulator floor (the shared blend `output`/`weights`
// buffers + the streaming per-tile `cat`); the z16 floor measures ~17 B/voxel (near the naive 3+1-ch f32
// blend buffers) — far below the z48's 160, since the z16 blend holds fewer full-output transients.
// `FRAME` is one full-res z16 frame's decoder activations. The real per-px cost RISES for smaller tiles
// (worse per-px decoder amortization: ~56 kB/px at full-frame, effectively higher tiled), so a single
// linear model cannot be tight on all three anchors — the constants are rounded to the **conservative**
// (over-predicting) side (ratios 1.14× / 1.14× / 1.86× for single / 512 / 256) so the selector never OKs
// a tile / single-pass that OOMs. Over-prediction also serves the "fit as small as we can go" directive
// (it errs toward smaller tiles). Re-run the sweep after a decoder or candle-allocator change.
const WAN_Z16_VAE_ACCUM_BYTES_PER_VOXEL: f64 = 100.0;
const WAN_Z16_VAE_FRAME_BYTES_PER_OUT_PX: f64 = 64_000.0;

/// Candidate spatial tile sizes (output px, multiples of the z16 ×8 scale, overlap 64). Coarser at the
/// top (fewer tiles = faster) down to a 192 px floor (past which per-tile decoder overhead amortizes
/// worse — the speed cliff). All are multiples of 8.
///
/// **Capped at 512** (sc-12758): 512 is the largest tile this PR GPU-validated clean (parity at 448 &
/// 256) — the earlier 768 (≈2.04e9 B im2col) / 640 (≈1.42e9 B) entries were never GPU-validated and sit
/// near candle's conv2d im2col-overflow band ([`WAN_Z16_VAE_IM2COL_SAFE_PX`]), so they are removed. The
/// selector now only ever picks a GPU-validated-clean tile.
const WAN_Z16_VAE_SPATIAL_PX: [i32; 6] = [512, 448, 384, 320, 256, 192];
/// Spatial tile overlap (output px) stamped onto whichever tile the selector / im2col cap picks.
const WAN_Z16_VAE_SPATIAL_OVERLAP_PX: i32 = 64;
/// **No temporal candidates** — the candle z16 decode is already temporally streaming-bounded, so the
/// budgeted selector only ever tiles the spatial axes (see the module note above).
const WAN_Z16_VAE_TEMPORAL_FR: [(i32, i32); 0] = [];

/// candle **conv2d im2col spatial cap** (output px per side) — historically a *correctness* bound the
/// VRAM budget and the MLX `i32` write bound both miss (sc-12758). candle's CUDA `conv2d` unrolls each
/// decoded frame into an im2col buffer of `out_h·out_w·C·kH·kW` elements; the z16 decoder's widest
/// full-res stage is 96-ch 3×3, so a single frame's im2col is `out_h·out_w·96·9`. Past a few hundred
/// million elements *an un-chunked* buffer **silently corrupts**. Reviewer GPU data on this z16 decoder:
/// 640² per frame (≈1.42e9 B ⇒ 3.5e8 elems) is CLEAN, 1280×720 (≈3.18e9 B ⇒ 8.0e8 elems) CORRUPTS.
///
/// **As of sc-12773 the corruption is fixed at the source**: every VAE conv2d now im2col-chunks
/// ([`crate::conv3d::chunked_conv2d`], the 128M-elem splitter — the twin of the `candle-gen-seedvr2`
/// one sc-10023 / sc-11744), so the plain untiled [`WanVae16::decode`] is correct at any resolution
/// (GPU-validated: untiled 1280×720 now matches the trusted 256 px tiling at ~52.6 dB, was ~15.6 dB
/// corrupt). **This cap is therefore no longer required for correctness** — it is retained only as a
/// conservative memory/tile bound (redundant defense-in-depth). Removing it (and re-adding the 640/768
/// tile candidates dropped by sc-12758) so hi-res decodes on big-VRAM cards run un-tiled — higher quality
/// than the tiling approximation — is a deliberate **follow-up** (it changes sc-12758's shipped A14B
/// decode routing, so it needs a render-level GPU re-validation of the 24 GB fit, out of scope here).
/// 512 is GPU-validated clean; `512²·96·9 = 2.26e8` elems sits well under the corruption band.
const WAN_Z16_VAE_IM2COL_SAFE_PX: i32 = 512;

/// Clamp a budgeted plan so **every** decoded tile's per-frame spatial extent stays ≤
/// [`WAN_Z16_VAE_IM2COL_SAFE_PX`] — the candle conv2d im2col-safe cap that neither the VRAM budget nor
/// the MLX write bound can see (sc-12758). The effective spatial tile is `min(VRAM-budget tile,
/// im2col-safe tile)`:
///  - output already ≤ cap on both sides ⇒ per-frame im2col is safe, keep the budget's decision
///    verbatim (including a single-pass `None`);
///  - output over the cap on either side ⇒ a full-frame (or over-cap) decode would corrupt, so force a
///    spatial tile ≤ cap: override a single-pass `None` to the cap tile, or shrink an already-tiled
///    plan's spatial tile to the cap (never enlarge). A forced cap tile always fits a budget that
///    admitted single-pass — the streaming cost model is monotonic in tile area, so a smaller tile can
///    only lower the peak.
fn cap_spatial_for_im2col(
    plan: Option<TilingConfig>,
    out_h: i32,
    out_w: i32,
) -> Option<TilingConfig> {
    let cap = WAN_Z16_VAE_IM2COL_SAFE_PX;
    if out_h <= cap && out_w <= cap {
        return plan; // per-frame extent already im2col-safe — the budget's decision stands
    }
    let capped_tile = |tile_px: Option<i32>| SpatialTiling {
        tile_px: tile_px.map_or(cap, |t| t.min(cap)),
        overlap_px: WAN_Z16_VAE_SPATIAL_OVERLAP_PX,
    };
    Some(match plan {
        // Budget said single-pass fits, but a full 1280×720-class frame's per-frame im2col is over the
        // cap — force the largest capped spatial tile (temporal already streams). Post-sc-12773 the
        // untiled conv2d self-chunks and is correct, so this is a conservative memory/tile bound, not a
        // correctness necessity (see WAN_Z16_VAE_IM2COL_SAFE_PX).
        None => TilingConfig {
            spatial: Some(capped_tile(None)),
            temporal: None,
        },
        // Budget already tiles; clamp its spatial tile down to the im2col cap (preserving any temporal).
        Some(cfg) => TilingConfig {
            spatial: Some(capped_tile(cfg.spatial.map(|s| s.tile_px))),
            temporal: cfg.temporal,
        },
    })
}

/// Estimated concurrent VRAM peak (GiB) of the streaming z16 decode while assembling an `out_*` video,
/// the largest spatial tile spanning `tile_h·tile_w` output px. `ACCUM·out_voxels` is the unavoidable
/// full-output accumulator floor; `FRAME·tile_h·tile_w` is one streamed frame's decoder activations
/// (independent of `out_f`/`tile_f` — the temporal axis streams). Single-pass is `tile_* == out_*`; a
/// zero tile yields the accumulator-only floor (the `budgeted_plan` contract). `tile_f` is unused.
fn estimated_wan_z16_decode_peak_gib(
    out_f: i64,
    out_h: i64,
    out_w: i64,
    _tile_f: i64,
    tile_h: i64,
    tile_w: i64,
) -> f64 {
    let out_voxels = (out_f * out_h * out_w) as f64;
    let frame_px = (tile_h * tile_w) as f64;
    (WAN_Z16_VAE_ACCUM_BYTES_PER_VOXEL * out_voxels + WAN_Z16_VAE_FRAME_BYTES_PER_OUT_PX * frame_px)
        / GIB_F64
}

/// The safe peak-GiB budget for the z16 decode tiler — **free-aware** (sc-12734/sc-12758). The decode
/// runs after the denoise, so it budgets against **FREE** VRAM, not `0.85×TOTAL`. Resolved in order:
/// `WAN_VAE_BUDGET_GIB` env override (positive float — the deterministic test/worker injection point) →
/// **free** VRAM × `WAN_Z16_VAE_BUDGET_SAFE_FRAC` (via the live `nvidia-smi memory.free` probe
/// [`candle_gen::gpu::nvidia_smi_min_free_gib`], i.e. `total − used`) → `WAN_Z16_VAE_DEFAULT_BUDGET_GIB`.
pub fn wan_z16_vae_safe_budget_gib() -> f64 {
    vae_tiling::free_aware_safe_budget_gib(
        WAN_Z16_VAE_BUDGET_ENV,
        WAN_Z16_VAE_BUDGET_SAFE_FRAC,
        WAN_Z16_VAE_DEFAULT_BUDGET_GIB,
    )
}

/// **Memory-budgeted** spatial tiling for the z16 decode — routes the shared `budgeted_plan` selector
/// through the z16 cost model, then clamps the result to the candle im2col-safe spatial cap
/// (`cap_spatial_for_im2col`). Caller passes the **output** dims. `Ok(None)` → a single high-res frame
/// both fits the budget **and** is under the im2col cap (streaming single-pass); `Ok(Some)` → the largest
/// spatial tile that fits and is im2col-safe; `Err` → a catchable over-budget signal returned before the
/// decode (not an OOM).
pub fn auto_tiling_budgeted_wan_z16(
    height: i32,
    width: i32,
    out_frames: i32,
) -> Result<Option<TilingConfig>> {
    plan_wan_z16_tiling(height, width, out_frames, wan_z16_vae_safe_budget_gib())
}

/// Pure z16 spatial tile selector (the `safe_gib` ceiling injected so it is unit-testable without a
/// GPU). Supplies the z16 candidate grid + cost model to the shared [`vae_tiling::plan_tiling`]; same
/// `Ok(None)` / `Ok(Some)` / catchable-`Err` contract as the z48 half.
fn plan_wan_z16_tiling(
    height: i32,
    width: i32,
    out_frames: i32,
    safe_gib: f64,
) -> Result<Option<TilingConfig>> {
    let candidates = TileCandidates {
        spatial_px: &WAN_Z16_VAE_SPATIAL_PX,
        spatial_overlap_px: WAN_Z16_VAE_SPATIAL_OVERLAP_PX,
        temporal: &WAN_Z16_VAE_TEMPORAL_FR,
    };
    let budget_plan = vae_tiling::plan_tiling(
        "wan z16 vae decode",
        WAN_Z16,
        height,
        width,
        out_frames,
        safe_gib,
        candidates,
        estimated_wan_z16_decode_peak_gib,
    )?;
    // The VRAM budget (and the MLX write bound gen-core enforces) can OK a single-pass / over-cap tile
    // that overflows candle's conv2d im2col and silently corrupts — bound every decoded tile to the
    // im2col-safe spatial cap on top of the budget decision (sc-12758 / sc-12773).
    Ok(cap_spatial_for_im2col(budget_plan, height, width))
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    #[test]
    fn wan_z16_tiling_single_pass_when_small() {
        // A short, low-res clip fits a single streaming decode → no tiling.
        assert!(plan_wan_z16_tiling(256, 256, 25, 60.0).unwrap().is_none());
    }

    #[test]
    fn wan_z16_im2col_cap_forces_tiling_at_low_frame_count() {
        // sc-12758 BLOCKER: a 1280×720 render with a LOW frame count on a LARGE-VRAM card. The MLX
        // write bound admits single-pass (24-frame cap ≥ these frame counts) and the budget is ample,
        // so the pure memory/write-bound selector returns `None` → the plain untiled `decode` → candle
        // conv2d im2col corruption (sc-12773). The im2col spatial cap must force a TILED plan for EVERY
        // frame count, independent of the (here effectively infinite) budget.
        for frames in [5, 9, 13, 17, 21] {
            let cfg = plan_wan_z16_tiling(720, 1280, frames, 1_000_000.0)
                .unwrap()
                .unwrap_or_else(|| {
                    panic!(
                        "1280×720/{frames}f must tile under the im2col cap, not take single-pass"
                    )
                });
            let s = cfg
                .spatial
                .expect("the im2col cap must tile the spatial axis");
            assert!(
                s.tile_px <= WAN_Z16_VAE_IM2COL_SAFE_PX,
                "chosen tile {} exceeds the im2col-safe cap {}",
                s.tile_px,
                WAN_Z16_VAE_IM2COL_SAFE_PX
            );
            assert!(
                cfg.temporal.is_none(),
                "temporal axis streams — the cap only tiles the spatial axes"
            );
        }
    }

    #[test]
    fn wan_z16_im2col_cap_keeps_single_pass_below_cap() {
        // At/under the cap on both sides the per-frame im2col is safe (512² is GPU-validated clean), so
        // an ample budget still returns a single-pass `None` — the cap must not tile needlessly.
        assert!(plan_wan_z16_tiling(512, 512, 17, 1_000_000.0)
            .unwrap()
            .is_none());
    }

    #[test]
    fn cap_spatial_for_im2col_forces_and_clamps() {
        let cap = WAN_Z16_VAE_IM2COL_SAFE_PX;
        // Below the cap on both sides: the budget decision (incl. a single-pass `None`) is kept verbatim.
        assert!(cap_spatial_for_im2col(None, cap, cap).is_none());
        // Over the cap: a single-pass `None` is forced to a cap-sized spatial tile; temporal still streams.
        let forced =
            cap_spatial_for_im2col(None, 720, 1280).expect("over-cap None must force a tile");
        assert_eq!(forced.spatial.expect("forced spatial tile").tile_px, cap);
        assert!(forced.temporal.is_none());
        // Over the cap: an already-tiled plan's spatial tile is clamped DOWN to the cap, never enlarged.
        let over = TilingConfig::spatial_only(cap + 256, WAN_Z16_VAE_SPATIAL_OVERLAP_PX);
        let clamped = cap_spatial_for_im2col(Some(over), 720, 1280).unwrap();
        assert_eq!(clamped.spatial.expect("clamped spatial tile").tile_px, cap);
        // A smaller-than-cap tile is left as-is (min, not overwrite).
        let small = TilingConfig::spatial_only(256, WAN_Z16_VAE_SPATIAL_OVERLAP_PX);
        let kept = cap_spatial_for_im2col(Some(small), 720, 1280).unwrap();
        assert_eq!(kept.spatial.expect("kept spatial tile").tile_px, 256);
    }

    #[test]
    fn wan_z16_tiling_bounds_high_res_frame_peak() {
        // The GPU-measured A14B T2V spike: 1280×720×81 single-pass adds ~42 GB (sc-12771). Emulate a
        // ~24 GB card (budget ≈ 20 GiB) and require the selector to tile spatially and keep the
        // recomputed peak under the safe ceiling (bounded/catchable), driving the decode toward the
        // "fit as small as we can go" target.
        let safe = 20.0;
        let cfg = plan_wan_z16_tiling(720, 1280, 81, safe)
            .unwrap()
            .expect("the 42 GB high-res A14B frame must tile spatially under a 20 GiB budget");
        // Spatial-only: the selector never tiles the temporal axis here (candle decode streams).
        assert!(cfg.spatial.is_some(), "expected a spatial tile");
        assert!(
            cfg.temporal.is_none(),
            "candle z16 decode streams temporally — no temporal tiling"
        );
        let th = cfg.spatial.map(|s| (s.tile_px as i64).min(720)).unwrap();
        let tw = cfg.spatial.map(|s| (s.tile_px as i64).min(1280)).unwrap();
        let peak = estimated_wan_z16_decode_peak_gib(81, 720, 1280, 81, th, tw);
        assert!(peak <= safe, "chosen peak {peak:.1} over safe {safe:.1}");
        // …and strictly below the single-pass spike it replaces.
        let single = estimated_wan_z16_decode_peak_gib(81, 720, 1280, 81, 720, 1280);
        assert!(
            peak < single,
            "tiling must lower the peak ({peak:.1} vs {single:.1})"
        );
    }

    #[test]
    fn wan_z16_tiling_errors_when_unfittable() {
        // 4K × 257 frames under 8 GiB: the output accumulators alone blow it → catchable, not an OOM.
        assert!(plan_wan_z16_tiling(2160, 3840, 257, 8.0).is_err());
    }

    #[test]
    fn wan_z16_budget_env_override_wins() {
        // The deterministic injection point the worker/tests use — the same env as the z48 tiler.
        std::env::set_var("WAN_VAE_BUDGET_GIB", "37.5");
        assert_eq!(wan_z16_vae_safe_budget_gib(), 37.5);
        std::env::remove_var("WAN_VAE_BUDGET_GIB");
    }

    #[test]
    fn wan_z16_free_aware_budget_picks_smaller_tile_than_total() {
        // sc-12734 core AC (z16 half): with N GB left resident, the FREE-aware budget picks a strictly
        // smaller decode tile than `0.85×TOTAL` would. 1280²×49 (not the 720p/81f render) so the ×8
        // write bound stays loose enough that the *budget*, not the write cap, is the deciding lever.
        let total = 96.0;
        let frac = WAN_Z16_VAE_BUDGET_SAFE_FRAC; // 0.85
        let resident = 70.0; // weights + cudarc pool the denoise left resident

        let total_budget = total * frac;
        let free_budget = vae_tiling::free_aware_budget_gib(total - resident, frac);
        assert!(
            free_budget < total_budget,
            "resident weights must shrink the budget: free {free_budget:.1} !< total {total_budget:.1}"
        );

        let big = plan_wan_z16_tiling(1280, 1280, 49, total_budget)
            .unwrap()
            .expect("total-based budget still tiles this high-res frame")
            .spatial
            .expect("spatial tile")
            .tile_px;
        let small = plan_wan_z16_tiling(1280, 1280, 49, free_budget)
            .unwrap()
            .expect("free-aware budget tiles")
            .spatial
            .expect("spatial tile")
            .tile_px;
        assert!(
            small < big,
            "free-aware selection must pick a strictly smaller tile: {small} !< {big}"
        );
        // …but NO smaller than needed: it takes the largest tile that fits, not the floor candidate
        // (budgeted_plan never over-tiles past the speed cliff).
        let smallest_candidate = *WAN_Z16_VAE_SPATIAL_PX.last().unwrap();
        assert!(
            small > smallest_candidate,
            "must not over-tile to the smallest candidate {smallest_candidate} when a larger tile fits"
        );
    }

    #[test]
    fn wan_z16_free_aware_budget_respects_accumulator_floor() {
        // If the denoise leaves so little free that even the full-output accumulator floor won't fit,
        // the tiler returns a catchable Err (AccumulatorsExceedBudget) rather than tiling BELOW the
        // floor into a guaranteed OOM. 1280×720×81's accumulator floor is ~11 GiB (ACCUM·out_vox); a
        // ~6.8 GiB free budget (8 GiB free × 0.85) is under it.
        let tiny_free = vae_tiling::free_aware_budget_gib(8.0, WAN_Z16_VAE_BUDGET_SAFE_FRAC);
        assert!(
            plan_wan_z16_tiling(720, 1280, 81, tiny_free).is_err(),
            "a below-floor free budget must be a catchable Err, not a sub-floor tile"
        );
    }

    #[test]
    fn wan_z16_tiling_plan_for_a14b_720p() {
        // Plan-level check for the real A14B T2V dims (1280×720/81f). Under a ~24 GB-card budget the
        // selector must tile spatially, keep the temporal axis whole, and the chosen tile must be
        // strictly smaller than the full 1280 spatial extent (so the decode footprint drops).
        let cfg = plan_wan_z16_tiling(720, 1280, 81, 20.0)
            .unwrap()
            .expect("A14B 720p/81f must tile at a 20 GiB budget");
        let s = cfg.spatial.expect("spatial tile");
        assert!(
            cfg.temporal.is_none(),
            "no temporal tiling for the streaming z16 decode"
        );
        assert!(
            (s.tile_px as i64) < 1280,
            "the chosen tile {} must be smaller than the 1280 spatial extent",
            s.tile_px
        );
    }

    /// sc-12758: the calibrated streaming cost model must stay **conservative** against the real CUDA
    /// peak-VRAM anchors (RTX PRO 6000 Blackwell, sm_120, f32, real Wan2.2-T2V-A14B z16 weights) it was
    /// fit from — `estimated ≥ measured` for every anchor (never under-predict ⇒ the selector never OKs
    /// a tile / single-pass that OOMs), and not absurdly over (≤ 2.5×). Regenerate the tiled anchors with
    /// `cargo test -p candle-gen-wan --features cuda --release --test vae16_decode_sweep -- --ignored
    /// --nocapture` after a decoder or candle-allocator change.
    #[test]
    fn wan_z16_decode_peak_matches_cuda_anchors() {
        // (out_f, out_h, out_w, tile_h, tile_w, measured_peak_gib). `tile_f` is unused by the streaming
        // model (passed as out_f). Single-pass ⇒ tile == out (the full-frame spike). All three are the
        // real `vae16_decode_sweep` CUDA measurements on the A14B z16 VAE (sc-12758) that pin ACCUM vs
        // FRAME; the single-pass 48.10 GiB is the isolated-decode analogue of sc-12771's 42 GB render delta.
        let anchors: [(i64, i64, i64, i64, i64, f64); 3] = [
            (81, 720, 1280, 720, 1280, 48.10), // single-pass A14B T2V spike (vae16_decode_sweep)
            (81, 720, 1280, 512, 512, 19.82),  // tiled 512px (vae16_decode_sweep)
            (81, 720, 1280, 256, 256, 5.85), // tiled 256px (vae16_decode_sweep, accumulator-dominated)
        ];
        for (of, oh, ow, th, tw, measured) in anchors {
            let est = estimated_wan_z16_decode_peak_gib(of, oh, ow, of, th, tw);
            assert!(
                est >= measured,
                "under-predicts {ow}x{oh}x{of} tile {tw}x{th}: est {est:.2} < measured {measured:.2} GiB"
            );
            assert!(
                est <= measured * 2.5,
                "over-predicts {ow}x{oh}x{of} tile {tw}x{th}: est {est:.2} > 2.5x measured {measured:.2} GiB"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    //! sc-9037 (F-053): the streaming decode/encode loops used to accumulate frame chunks by
    //! `cat`-ing onto a growing accumulator each iteration — O(T²) copy traffic. The fix collects
    //! the chunks into a `Vec<Tensor>` and does a single `Tensor::cat` at the end (O(T)). This must
    //! be **bit-identical** to the old incremental accumulation (same frames, same temporal order,
    //! no boundary blend — the streaming loops here are a plain temporal concatenation). These tests
    //! pin that equivalence at the tensor level so the refactor is provably output-preserving
    //! without needing real VAE weights.

    use candle_gen::candle_core::{DType, Device, Tensor};

    /// Old accumulator: `out = cat([out, chunk], dim)` folded over the chunk list (the pre-sc-9037
    /// pattern). Returns `None` for an empty list (mirrors the old `Option<Tensor>` seed).
    fn incremental_cat(chunks: &[Tensor], dim: usize) -> Option<Tensor> {
        let mut out: Option<Tensor> = None;
        for c in chunks {
            out = Some(match out {
                Some(o) => Tensor::cat(&[&o, c], dim).unwrap(),
                None => c.clone(),
            });
        }
        out
    }

    /// A known sequence of temporal chunks shaped like the streaming decode output
    /// `[B, C, t_i, H, W]` with per-chunk frame counts mimicking `1 + (T-1)·4` (frame 0 → 1 output
    /// frame, later latent frames → 4). Deterministic ascending values so any reorder/duplication
    /// would change the bytes.
    fn make_chunks(dev: &Device) -> Vec<Tensor> {
        let (b, c, h, w) = (1usize, 3usize, 2usize, 2usize);
        let per_chunk_frames = [1usize, 4, 4, 4, 4]; // 5 latent frames → 17 output frames
        let mut base = 0f32;
        let mut chunks = Vec::new();
        for &tf in &per_chunk_frames {
            let n = b * c * tf * h * w;
            let data: Vec<f32> = (0..n).map(|k| base + k as f32).collect();
            base += n as f32;
            chunks.push(Tensor::from_vec(data, (b, c, tf, h, w), dev).unwrap());
        }
        chunks
    }

    /// The single-cat replacement is byte-for-byte identical to the incremental accumulator over a
    /// realistic multi-frame chunk sequence (the decode temporal axis, dim 2).
    #[test]
    fn single_cat_matches_incremental_accumulation() {
        let dev = Device::Cpu;
        let chunks = make_chunks(&dev);

        let single = Tensor::cat(&chunks, 2).unwrap();
        let incremental = incremental_cat(&chunks, 2).expect("non-empty");

        assert_eq!(single.dims(), incremental.dims(), "shape must match");
        let a = single
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = incremental
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Bit-identical (concatenation copies bytes verbatim — no arithmetic, no blend).
        assert_eq!(a, b, "single-cat must be bit-identical to incremental cat");
    }

    /// The total output frame count along the temporal axis is `1 + (T-1)·4` — the streaming
    /// contract — and is preserved by the single cat.
    #[test]
    fn single_cat_preserves_frame_count() {
        let dev = Device::Cpu;
        let chunks = make_chunks(&dev);
        let single = Tensor::cat(&chunks, 2).unwrap();
        let t_lat = chunks.len();
        assert_eq!(single.dim(2).unwrap(), 1 + (t_lat - 1) * 4);
    }

    /// F-126 (sc-11220): `encode` must reject a temporal length it cannot consume without dropping
    /// trailing frames. Aligned `1 + 4·k` lengths pass; any other (including 0) errors, and the
    /// message reports how many frames would have been dropped.
    #[test]
    fn require_aligned_encode_frames_gates_unaligned_lengths() {
        for t in [1usize, 5, 9, 17, 33] {
            assert!(
                super::require_aligned_encode_frames(t).is_ok(),
                "t={t} is 1+4k and must be accepted"
            );
        }
        // t=0 and every t with (t-1)%4 != 0 are rejected.
        assert!(super::require_aligned_encode_frames(0).is_err());
        for (t, dropped) in [(2usize, 1usize), (3, 2), (4, 3), (18, 1), (20, 3)] {
            let err = super::require_aligned_encode_frames(t)
                .expect_err("unaligned length must be rejected");
            let msg = err.to_string();
            assert!(
                msg.contains(&format!("{dropped}")) && msg.contains(&format!("got {t}")),
                "message reports dropped count and t: {msg}"
            );
        }
    }
}
