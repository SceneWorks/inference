//! Qwen-Image causal-Conv3d VAE — encoder (image → 16-ch latent) + decoder (latent → image),
//! ported 1:1 from the fork (`qwen_vae.py`, `qwen_image_{encoder,decoder}_3d.py`). NCTHW I/O;
//! for T2I the temporal axis is a singleton (T=1).
//!
//! Latents are scaled by per-channel `mean`/`std` (fork `QwenVAE.LATENTS_{MEAN,STD}`). Weight keys
//! mirror the fork's *internal* module tree (`decoder.conv_in.conv3d.weight`,
//! `encoder.down_blocks.0.resnets.0.…`, etc.), so a `Weights` dumped from a loaded fork VAE drops
//! straight in. The on-disk-snapshot key remapping (diffusers → this tree) is applied by the loader
//! (`remap_vae_keys`).

pub mod blocks;

use mlx_rs::ops::{add, divide, multiply, split, subtract};
use mlx_rs::Array;

use mlx_gen::gen_core::{QWEN_WAN_Z16_MEAN as LATENTS_MEAN, QWEN_WAN_Z16_STD as LATENTS_STD};
use mlx_gen::nn::silu;
use mlx_gen::tiling::{TilingConfig, VaeTiling};
use mlx_gen::vae_tiling::{tiled_decode, tiled_decode_with_finalizer};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, LatentDecoder, Result};

use blocks::{
    channel_norm_bytes, decode_quantum, element_bytes, ncthw, rms_norm_channels, CausalConv3d,
    DownBlock3D, MidBlock3D, UpBlock3D, NORM_EPS,
};

/// iOS decode-quantum labels (see [`mlx_gen::ios_quantum`]); the per-resnet/attention/upsample
/// labels live in [`blocks`].
pub const DECODE_HEAD_QUANTUM: &str = "Qwen VAE decode head";
pub const DECODE_OUTPUT_QUANTUM: &str = "Qwen VAE decode output";
pub const DECODE_TILE_FOLD_QUANTUM: &str = "Qwen VAE tile fold";
pub const DECODE_FINAL_NORMALIZE_QUANTUM: &str = "Qwen VAE final normalize";

/// Image → 32-ch (sliced to 16) latent. 4 down-stages (3 spatial-downsample), then a mid-block.
pub struct Encoder3D {
    conv_in: CausalConv3d,
    down_blocks: Vec<DownBlock3D>,
    mid_block: MidBlock3D,
    norm_out: Array,
    conv_out: CausalConv3d,
}

impl Encoder3D {
    pub fn from_weights(w: &Weights) -> Result<Self> {
        // (dim transitions 96→96→192→384→384; stages 0–2 downsample, stage 3 does not.)
        let mut down_blocks = Vec::with_capacity(4);
        for i in 0..4 {
            down_blocks.push(DownBlock3D::from_weights(
                w,
                &format!("encoder.down_blocks.{i}"),
                2,
                i < 3,
            )?);
        }
        Ok(Self {
            conv_in: CausalConv3d::from_weights(w, "encoder.conv_in", 1)?,
            down_blocks,
            mid_block: MidBlock3D::from_weights(w, "encoder.mid_block")?,
            norm_out: w.require("encoder.norm_out.weight")?.clone(),
            conv_out: CausalConv3d::from_weights(w, "encoder.conv_out", 1)?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = self.conv_in.forward(x)?;
        for block in &self.down_blocks {
            x = block.forward(&x)?;
        }
        x = self.mid_block.forward(&x)?;
        x = rms_norm_channels(&x, &self.norm_out, NORM_EPS)?;
        self.conv_out.forward(&silu(&x)?)
    }
}

/// 16-ch latent → 3-ch image. Mid-block, then 4 up-stages (3 spatial-upsample).
pub struct Decoder3D {
    conv_in: CausalConv3d,
    mid_block: MidBlock3D,
    up_blocks: Vec<UpBlock3D>,
    norm_out: Array,
    conv_out: CausalConv3d,
}

impl Decoder3D {
    pub fn from_weights(w: &Weights) -> Result<Self> {
        // up_block0/1/2 upsample (2× spatial each → 8×); up_block3 does not.
        let mut up_blocks = Vec::with_capacity(4);
        for i in 0..4 {
            up_blocks.push(UpBlock3D::from_weights(
                w,
                &format!("decoder.up_block{i}"),
                2,
                i < 3,
            )?);
        }
        Ok(Self {
            conv_in: CausalConv3d::from_weights(w, "decoder.conv_in", 1)?,
            mid_block: MidBlock3D::from_weights(w, "decoder.mid_block")?,
            up_blocks,
            norm_out: w.require("decoder.norm_out.weight")?.clone(),
            conv_out: CausalConv3d::from_weights(w, "decoder.conv_out", 1)?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = self.forward_pre_upsample(x)?;
        self.forward_upsample_tail(&x)
    }

    /// The **pre-upsample head**: `conv_in → mid_block` (the mid-block carries a *global* per-frame
    /// spatial self-attention over all H·W tokens). Runs at LATENT resolution and is spatially GLOBAL,
    /// so a tiled decode (sc-11747) must run this ONCE on the full latent — tiling it would make each
    /// tile attend only within itself, changing the whole image. It runs at latent resolution, but its
    /// single 384-wide head is not one the fused SDPA kernel serves, so its `H·W x H·W` scores
    /// materialize; on a physical iOS device the attention is therefore query-chunked
    /// (`AttentionBlock3D::forward_bounded`), bounding the scores per chunk without changing what any
    /// query attends over.
    pub(super) fn forward_pre_upsample(&self, x: &Array) -> Result<Array> {
        let x = decode_quantum(
            DECODE_HEAD_QUANTUM,
            self.conv_in.quantum_bytes(ncthw(x), element_bytes(x)),
            || self.conv_in.forward(x),
        )?;
        self.mid_block.forward_decode(&x)
    }

    /// The **upsample tail**: `up_blocks → norm_out → SiLU → conv_out`. Every op here is spatially LOCAL
    /// — resnet convs, nearest-2× + conv upsamplers, the per-position channel-L2 `norm_out`, and the head
    /// conv — so it is **normalization-correct under tiling**: no statistic here is scoped to a tile, and
    /// a tile's interior is bit-identical to the dense decode's (proved in
    /// `tests/vae_tiling_normalization_proof.rs`, sc-19753). This is also where the decode memory spike
    /// lives (the 8× spatial growth), so tiling it is what bounds the peak (sc-11747). `x` is the
    /// pre-upsample head output at latent resolution.
    ///
    /// ⚠️ Tile-*safe* is not seam-*free*. The tail's receptive field is **98 output px** — each stage
    /// contributes 6 (3 resnets × 2 `k=3` convs) at its own resolution, each upsample doubles the
    /// accumulated radius and adds 1, `conv_out` adds 1 — which is **wider than the 64 px overlap**
    /// `TilingConfig::auto` ships. The trapezoidal blend attenuates that conv halo; it does not remove
    /// it, so a residual seam term is expected by design. What tiling this stage cannot introduce is a
    /// per-tile *statistic*, which is the sc-19753 property and the one the tests pin.
    ///
    /// On a physical iOS device (and the CPU diagnostic fixture) every resnet, upsampler and the
    /// output head is one execution quantum, checked before its graph is built; elsewhere the
    /// quanta are pass-throughs and this is the historical single lazy graph.
    pub(super) fn forward_upsample_tail(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for block in &self.up_blocks {
            x = block.forward_decode(&x)?;
        }
        decode_quantum(
            DECODE_OUTPUT_QUANTUM,
            self.output_quantum_bytes(ncthw(&x), element_bytes(&x)),
            || self.forward_output(&x),
        )
    }

    /// The output head: `norm_out → SiLU → conv_out`.
    pub(super) fn forward_output(&self, x: &Array) -> Result<Array> {
        let x = rms_norm_channels(x, &self.norm_out, NORM_EPS)?;
        self.conv_out.forward(&silu(&x)?)
    }

    /// Bytes [`Self::forward_output`] allocates over an `[B, C, T, H, W]` input: the channel norm,
    /// the SiLU (1 full width) and `conv_out`.
    pub(super) fn output_quantum_bytes(&self, x: [u64; 5], eb: u64) -> u64 {
        channel_norm_bytes(x, eb)
            .saturating_add(mlx_gen::ios_quantum::allocated_tensor_bytes(eb, &x))
            .saturating_add(self.conv_out.quantum_bytes(x, eb))
    }
}

/// The Qwen-Image VAE: `encode` (image → scaled 16-ch latent) and `decode` (latent → image).
pub struct QwenVae {
    encoder: Encoder3D,
    decoder: Decoder3D,
    quant_conv: CausalConv3d,
    post_quant_conv: CausalConv3d,
    mean: Array,
    std: Array,
}

impl QwenVae {
    pub fn from_weights(w: &Weights) -> Result<Self> {
        Ok(Self {
            encoder: Encoder3D::from_weights(w)?,
            decoder: Decoder3D::from_weights(w)?,
            quant_conv: CausalConv3d::from_weights(w, "quant_conv", 0)?,
            post_quant_conv: CausalConv3d::from_weights(w, "post_quant_conv", 0)?,
            mean: Array::from_slice(&LATENTS_MEAN, &[1, 16, 1, 1, 1]),
            std: Array::from_slice(&LATENTS_STD, &[1, 16, 1, 1, 1]),
        })
    }

    /// 16-ch latent (NCHW or NCTHW) → image `(B, 3, 1, H, W)`. Denormalizes by `std`/`mean`.
    pub fn decode(&self, latents: &Array) -> Result<Array> {
        let _quantum_scope = mlx_gen::ios_quantum::Scope::new();
        self.decode_upsample_tail(&self.decode_pre_upsample(latents)?)
    }

    /// The decode **head** at LATENT resolution: denormalize (`·std + mean`) → `post_quant_conv` →
    /// decoder `conv_in` → mid-block. The mid-block carries a *global* per-frame spatial self-attention,
    /// so a tiled decode (sc-11747) must run this ONCE on the full latent — it is spatially global, so
    /// tiling it would corrupt the whole image (on iOS its scores are bounded by query chunking
    /// instead; see `Decoder3D::forward_pre_upsample`). Output is the
    /// pre-upsample feature map `[B, C, T, H, W]` (spatially unchanged from the latent).
    pub fn decode_pre_upsample(&self, latents: &Array) -> Result<Array> {
        let l = to_5d(latents)?;
        // One iOS head quantum: denormalize → `post_quant_conv` → decoder `conv_in`.
        let x = decode_quantum(
            DECODE_HEAD_QUANTUM,
            self.head_quantum_bytes(ncthw(&l), element_bytes(&l)),
            || self.decode_head(&l),
        )?;
        self.decoder.mid_block.forward_decode(&x)
    }

    /// The decode head quantum's body: denormalize (`·std + mean`) → `post_quant_conv` → decoder
    /// `conv_in`, over a 5-D latent.
    pub(super) fn decode_head(&self, l: &Array) -> Result<Array> {
        let l = add(&multiply(l, &self.std)?, &self.mean)?;
        let l = self.post_quant_conv.forward(&l)?;
        self.decoder.conv_in.forward(&l)
    }

    /// Bytes [`Self::decode_head`] allocates over an `[B, 16, T, H, W]` latent: the denormalized
    /// product and sum (2 latents), `post_quant_conv` and `conv_in`.
    pub(super) fn head_quantum_bytes(&self, l: [u64; 5], eb: u64) -> u64 {
        (2 * mlx_gen::ios_quantum::allocated_tensor_bytes(eb, &l))
            .saturating_add(self.post_quant_conv.quantum_bytes(l, eb))
            .saturating_add(self.decoder.conv_in.quantum_bytes(l, eb))
    }

    /// The decode **upsample tail**: `up_blocks → norm_out → SiLU → conv_out`, taking the
    /// [`decode_pre_upsample`](Self::decode_pre_upsample) head and producing the `(B, 3, 1, 8·H, 8·W)`
    /// image. Every op is spatially LOCAL, so this is the stage a tiled decode splits into blended tiles
    /// (and where the decode memory spike lives). Kept public so the tiling seam ([`Self::decode_tiled`])
    /// and its parity tests can drive it directly.
    pub fn decode_upsample_tail(&self, head: &Array) -> Result<Array> {
        self.decoder.forward_upsample_tail(head)
    }

    /// **Tiled** decode for a memory-bounded large-image decode (sc-11747): split the latent into
    /// overlapping spatial tiles, run `post_quant_conv` + the decoder stack per tile, and trapezoidally
    /// blend them into the full `(B, 3, 1, 8·H, 8·W)` image — so the end-of-generation decode spike
    /// scales with the tile size, not the full resolution (the peak obstacle to a Krea pose-control
    /// render fitting a 32 GB Mac). Falls back to the single-pass [`decode`](Self::decode) when `cfg`
    /// doesn't fire for these dims (small image / large-memory machine → zero tiling overhead).
    ///
    /// **Denormalize + `post_quant_conv` + the decoder's pre-upsample head run ONCE on the full latent**
    /// (the head carries a *global* spatial self-attention, so tiling it would change the whole image and
    /// it is cheap at latent resolution), then only the **upsample tail** — the spatially-local, memory-
    /// spiking stage — is tiled and trapezoidally blended. The Qwen-Image VAE is a still-image VAE
    /// (spatial ×8, singleton temporal axis, [`VaeTiling::QWEN_IMAGE`]) so only H/W tile; the shared blend
    /// geometry lives in [`mlx_gen::tiling`] and the Array loop in [`mlx_gen::vae_tiling`]. No clamp here
    /// (the single-pass [`decode`](Self::decode) doesn't clamp either — the `[-1,1]` clamp is applied
    /// later by the engine's `decoded_to_image`), so the tiled output matches the untiled one to within
    /// the blend tolerance — the *conv-halo* seam term described on
    /// `Decoder3D::forward_upsample_tail`, which the shipped 64 px overlap attenuates rather than
    /// eliminates. It is never a per-tile normalization or attention term: that is the sc-19753
    /// property, pinned by `tests/vae_tiling_normalization_proof.rs`.
    pub fn decode_tiled(
        &self,
        latents: &Array,
        cfg: &TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        let (l, f, h, w, needs_tiling) = tiled_decode_preamble(latents, cfg, cancel)?;
        if !needs_tiling {
            return self.decode(latents);
        }
        let _quantum_scope = mlx_gen::ios_quantum::Scope::new();
        // Head (denormalize → post_quant_conv → conv_in → mid-block global attention) runs ONCE on the
        // full latent — identical to single-pass `decode` up to the up-blocks, so parity is exact here.
        let head = self.decode_pre_upsample(&l)?;
        // Tile the upsample tail. The head output is at latent resolution (H×W), so the same
        // latent-shape plan drives the tiling; `VaeTiling::QWEN_IMAGE`'s ×8 spatial scale maps each
        // tile to its 8× output slab. NCTHW: channel axis at 1, tiled axes [2, 3, 4] (T is the singleton).
        let plan = cfg.plan(VaeTiling::QWEN_IMAGE, f, h, w);
        if !mlx_gen::ios_quantum::enabled() {
            return tiled_decode(&head, &plan, [2, 3, 4], cancel, |tile| {
                self.decode_upsample_tail(tile)
            });
        }
        // iOS: each tile's tail stages are their own quanta (inside `decode_upsample_tail`); the
        // fold of a decoded tile into the full-size accumulators and the final normalization are
        // checked before their graphs are built ([`tile_fold_bytes`], [`final_normalize_bytes`]).
        let latent = ncthw(&l);
        let output_bytes = tile_fold_bytes(latent);
        let normalize_bytes = final_normalize_bytes(latent);
        let out = tiled_decode_with_finalizer(
            &head,
            &plan,
            [2, 3, 4],
            cancel,
            |tile| {
                let decoded = self.decode_upsample_tail(tile)?;
                mlx_gen::ios_quantum::before(DECODE_TILE_FOLD_QUANTUM, output_bytes)?;
                Ok(decoded)
            },
            |accumulators| mlx_gen::ios_quantum::finish(DECODE_TILE_FOLD_QUANTUM, accumulators),
            || {
                mlx_gen::residency::drain_allocator_cache();
            },
            |_| mlx_gen::ios_quantum::before(DECODE_FINAL_NORMALIZE_QUANTUM, normalize_bytes),
        )?;
        mlx_gen::ios_quantum::finish(DECODE_FINAL_NORMALIZE_QUANTUM, &[&out])?;
        Ok(out)
    }

    /// Image `(B, 3, H, W)` (or NCTHW) → scaled 16-ch latent `(B, 16, 1, H/8, W/8)`.
    pub fn encode(&self, image: &Array) -> Result<Array> {
        let x = to_5d(image)?;
        let e = self.encoder.forward(&x)?;
        let e = self.quant_conv.forward(&e)?;
        let e16 = split(&e, 2, 1)?.swap_remove(0); // keep first 16 of 32 channels
        Ok(divide(&subtract(&e16, &self.mean)?, &self.std)?)
    }
}

/// The native decoder for the Qwen-Image latent space (the behavior-preserving default of the
/// PiD decode seam, sc-7844). Delegates to the inherent [`QwenVae::decode`]; a PiD decoder for this
/// same latent space (`mlx-gen-pid`, sc-7843/7845) implements the same trait so an engine can swap
/// between them at the decode call site.
impl LatentDecoder for QwenVae {
    fn input_latent_space(&self) -> Option<&mlx_gen::gen_core::LatentSpace> {
        Some(&mlx_gen::gen_core::QWEN_KREA_Z16_LATENT_SPACE)
    }

    fn decode(&self, latents: &Array) -> Result<Array> {
        QwenVae::decode(self, latents)
    }

    fn decode_tiled(
        &self,
        latents: &Array,
        tiling: &TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(mlx_gen::Error::Canceled);
        }
        QwenVae::decode_tiled(self, latents, tiling, cancel)
    }
}

/// The tile-fold quantum over a `[B, C, F, H, W]` latent: a fold pads the weighted tile into the
/// full-size slab, sums it into the slab accumulator and the slab into the output — 3 f32 RGB
/// outputs `[B, 3, F, 8H, 8W]` (proven in `quantum_tests::tile_fold_formula_tracks_the_measured_peak`).
pub fn tile_fold_bytes([b, _, f, h, w]: [u64; 5]) -> u64 {
    let scale = VaeTiling::QWEN_IMAGE.spatial_scale as u64;
    3 * mlx_gen::ios_quantum::allocated_tensor_bytes(4, &[b, 3, f, h * scale, w * scale])
}

/// The final-normalization quantum over a `[B, C, F, H, W]` latent: the separable coverage
/// normalizer built as one f32 `[B, 1, F, 8H, 8W]` plane from its three per-axis vectors and the
/// clamp's f32 `1e-8` floor scalar. Its clamp and the quotient are elementwise outputs MLX donates
/// into the normalizer and into the folded output, whose last handle is released before the graph
/// runs.
pub fn final_normalize_bytes([b, _, f, h, w]: [u64; 5]) -> u64 {
    let scale = VaeTiling::QWEN_IMAGE.spatial_scale as u64;
    // The plane is one (page-rounded) allocation; the per-axis vectors and the scalar are sub-page.
    mlx_gen::ios_quantum::allocated_tensor_bytes(4, &[b, 1, f, h * scale, w * scale])
        .saturating_add(mlx_gen::ios_quantum::tensor_bytes(
            4,
            &[f + (h + w) * scale + 1],
        ))
}

impl QwenVae {
    /// Every decode quantum's counted scratch for a `[B, 16, F, H, W]` latent at `element_bytes`,
    /// from the loaded module shapes alone (no tensor is read): the head, the mid-block at the full
    /// latent (its attention under `attention`), the upsample tail at the largest tile `tiling`
    /// cuts (the full latent when it does not tile), and the tile fold and final normalization when
    /// it does. Labels are the quanta's own. For planning predictions; a decode checks each quantum
    /// itself.
    pub fn decode_quanta(
        &self,
        latent: [u64; 5],
        tiling: Option<&TilingConfig>,
        attention: mlx_gen::attention::AttentionPlan<'_>,
        element_bytes: u64,
    ) -> Vec<(&'static str, u64)> {
        let eb = element_bytes;
        let [b, _, f, h, w] = latent;
        let mut out = vec![(DECODE_HEAD_QUANTUM, self.head_quantum_bytes(latent, eb))];
        let conv_in = self.decoder.conv_in.out_channels().max(0) as u64;
        let mid =
            self.decoder
                .mid_block
                .decode_quanta([b, conv_in, f, h, w], eb, attention, &mut out);
        let dim = |x: u64| x.min(i32::MAX as u64) as i32;
        let tiled =
            tiling.filter(|cfg| cfg.needs_tiling(VaeTiling::QWEN_IMAGE, dim(f), dim(h), dim(w)));
        let (th, tw) = match tiled {
            Some(cfg) => {
                let plan = cfg.plan(VaeTiling::QWEN_IMAGE, dim(f), dim(h), dim(w));
                let largest = |tiles: &[mlx_gen::tiling::AxisTile]| {
                    tiles
                        .iter()
                        .map(|t| (t.end - t.start).max(0) as u64)
                        .max()
                        .unwrap_or(0)
                };
                (largest(&plan.h), largest(&plan.w))
            }
            None => (h, w),
        };
        let mut x = [b, mid[1], f, th, tw];
        for block in &self.decoder.up_blocks {
            x = block.decode_quanta(x, eb, &mut out);
        }
        out.push((
            DECODE_OUTPUT_QUANTUM,
            self.decoder.output_quantum_bytes(x, eb),
        ));
        if tiled.is_some() {
            out.push((DECODE_TILE_FOLD_QUANTUM, tile_fold_bytes(latent)));
            out.push((
                DECODE_FINAL_NORMALIZE_QUANTUM,
                final_normalize_bytes(latent),
            ));
        }
        out
    }

    /// The attention plan an iOS decode runs the mid-block attention under (the query-chunked
    /// plan; see `AttentionBlock3D::forward_bounded`), for [`Self::decode_quanta`].
    pub fn ios_decode_attention_plan() -> mlx_gen::attention::AttentionPlan<'static> {
        mlx_gen::attention::AttentionPlan::budgeted(
            mlx_gen::attention::AttentionBudget::from_score_elements(
                blocks::DECODE_ATTENTION_SCORE_ELEMENTS,
                true,
            ),
        )
    }
}

/// Add a singleton temporal axis to a 4-D `(B, C, H, W)` tensor → `(B, C, 1, H, W)`.
fn to_5d(x: &Array) -> Result<Array> {
    if x.shape().len() == 4 {
        let s = x.shape();
        Ok(x.reshape(&[s[0], s[1], 1, s[2], s[3]])?)
    } else {
        Ok(x.clone())
    }
}

/// The complete work-free preamble for Qwen's native tiled decode. Cancellation is intentionally
/// first: a pre-tripped request must not validate/reshape a malformed tensor, choose a plan, enter
/// the monolithic fallback, or execute the globally-scoped head.
fn tiled_decode_preamble(
    latents: &Array,
    cfg: &TilingConfig,
    cancel: Option<&CancelFlag>,
) -> Result<(Array, i32, i32, i32, bool)> {
    if cancel.is_some_and(CancelFlag::is_cancelled) {
        return Err(mlx_gen::Error::Canceled);
    }
    let latent5 = to_5d(latents)?;
    let shape = latent5.shape();
    let (frames, height, width) = (shape[2], shape[3], shape[4]);
    let needs_tiling = cfg.needs_tiling(VaeTiling::QWEN_IMAGE, frames, height, width);
    Ok((latent5, frames, height, width, needs_tiling))
}

#[cfg(test)]
mod tiled_cancel_tests {
    use super::*;

    #[test]
    fn pre_cancel_wins_for_firing_and_non_firing_plans_before_validation() {
        let malformed = Array::from_slice(&[1.0f32], &[1]);
        let cancel = CancelFlag::new();
        cancel.cancel();
        for cfg in [
            TilingConfig::spatial_only(8, 2),
            TilingConfig::spatial_only(4096, 64),
        ] {
            assert!(matches!(
                tiled_decode_preamble(&malformed, &cfg, Some(&cancel)),
                Err(mlx_gen::Error::Canceled)
            ));
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
#[path = "../../../mlx-gen-iris/tests/common/metal_guard.rs"]
mod quantum_metal_guard;

/// iOS bounded-quantum guards on the Qwen-Image VAE decode (the Anima/Qwen/Krea decode path), on a
/// structurally faithful synthetic VAE (the real 4 up-blocks × 3 resnets, 3 upsamplers, mid
/// resnet/attention/resnet) at a narrow width, through the CPU diagnostic fixture. Every test
/// re-executes itself in a CPU-pinned child with Metal made fatal.
#[cfg(all(test, target_os = "macos"))]
mod quantum_tests {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    use mlx_gen::ios_quantum::{
        take_refusal, with_cpu_diagnostic_headroom, with_cpu_diagnostic_observer, QuantumEvent,
        PROCESS_RESERVE_BYTES,
    };

    use super::blocks::{DECODE_ATTENTION_QUANTUM, DECODE_RESNET_QUANTUM, DECODE_UPSAMPLE_QUANTUM};
    use super::*;

    const CHILD: &str = "QWEN_VAE_IOS_QUANTUM_CPU_CHILD";
    const AMPLE: u64 = 1 << 34;

    fn in_cpu_child(name: &str) -> bool {
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    &format!("vae::quantum_tests::{name}"),
                    "--include-ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
                .env("IOS_CAPTURE_IN_CHILD", "1")
                .status()
                .unwrap();
            assert!(status.success(), "CPU child for {name} failed");
            return false;
        }
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert!(super::quantum_metal_guard::forbid_metal());
        assert!(super::quantum_metal_guard::metal_forbidden());
        true
    }

    fn values(shape: &[i32], phase: f32, scale: f32) -> Array {
        let count = shape.iter().product::<i32>();
        let data = (0..count)
            .map(|i| ((i as f32 + phase) * 0.071).sin() * scale)
            .collect::<Vec<_>>();
        Array::from_slice(&data, shape)
    }

    /// A synthetic Qwen-Image VAE at decoder width `c` (the encoder half is required by
    /// `QwenVae::from_weights` but unused).
    fn vae(c: i32) -> QwenVae {
        let mut t: HashMap<String, Array> = HashMap::new();
        let phase = std::cell::Cell::new(0.0f32);
        let next = || {
            phase.set(phase.get() + 1.0);
            phase.get()
        };
        let conv3d = |t: &mut HashMap<String, Array>, p: &str, i: i32, o: i32, k: i32| {
            let ph = next();
            t.insert(
                format!("{p}.conv3d.weight"),
                values(&[o, k, k, k, i], ph, 0.09),
            );
            t.insert(format!("{p}.conv3d.bias"), values(&[o], ph + 0.5, 0.01));
        };
        let norm = |t: &mut HashMap<String, Array>, key: String| {
            t.insert(key, Array::from_slice(&vec![1.0f32; c as usize], &[c]));
        };
        let resnet = |t: &mut HashMap<String, Array>, p: &str| {
            norm(t, format!("{p}.norm1.weight"));
            conv3d(t, &format!("{p}.conv1"), c, c, 3);
            norm(t, format!("{p}.norm2.weight"));
            conv3d(t, &format!("{p}.conv2"), c, c, 3);
        };
        let resample = |t: &mut HashMap<String, Array>, p: &str, ph: f32| {
            t.insert(
                format!("{p}.resample_conv.weight"),
                values(&[c, 3, 3, c], ph, 0.09),
            );
            t.insert(format!("{p}.resample_conv.bias"), values(&[c], ph, 0.01));
        };
        let attn = |t: &mut HashMap<String, Array>, p: &str| {
            norm(t, format!("{p}.norm.weight"));
            t.insert(
                format!("{p}.to_qkv.weight"),
                values(&[3 * c, 1, 1, c], 11.0, 0.25),
            );
            t.insert(format!("{p}.to_qkv.bias"), values(&[3 * c], 12.0, 0.02));
            t.insert(
                format!("{p}.proj.weight"),
                values(&[c, 1, 1, c], 13.0, 0.25),
            );
            t.insert(format!("{p}.proj.bias"), values(&[c], 14.0, 0.02));
        };
        for side in ["decoder", "encoder"] {
            resnet(&mut t, &format!("{side}.mid_block.resnets.0"));
            attn(&mut t, &format!("{side}.mid_block.attentions.0"));
            resnet(&mut t, &format!("{side}.mid_block.resnets.1"));
            norm(&mut t, format!("{side}.norm_out.weight"));
        }
        conv3d(&mut t, "decoder.conv_in", 16, c, 3);
        for block in 0..4 {
            for r in 0..3 {
                resnet(&mut t, &format!("decoder.up_block{block}.resnets.{r}"));
            }
            if block < 3 {
                resample(
                    &mut t,
                    &format!("decoder.up_block{block}.upsamplers.0"),
                    80.0,
                );
            }
        }
        conv3d(&mut t, "decoder.conv_out", c, 3, 3);
        conv3d(&mut t, "encoder.conv_in", 3, c, 3);
        for block in 0..4 {
            for r in 0..2 {
                resnet(&mut t, &format!("encoder.down_blocks.{block}.resnets.{r}"));
            }
            if block < 3 {
                resample(
                    &mut t,
                    &format!("encoder.down_blocks.{block}.downsamplers.0"),
                    150.0,
                );
            }
        }
        conv3d(&mut t, "encoder.conv_out", c, 32, 3);
        conv3d(&mut t, "quant_conv", 32, 32, 1);
        conv3d(&mut t, "post_quant_conv", 16, 16, 1);
        QwenVae::from_weights(&Weights::from_map(t)).unwrap()
    }

    fn latents(hw: i32) -> Array {
        values(&[1, 16, 1, hw, hw], 0.3, 1.0)
    }

    fn observed(
        headroom: u64,
        body: impl FnOnce() -> Result<Array>,
    ) -> (Result<Array>, Vec<QuantumEvent>) {
        let events = Rc::new(RefCell::new(Vec::new()));
        let sink = events.clone();
        let result = with_cpu_diagnostic_headroom(headroom, || {
            with_cpu_diagnostic_observer(move |e| sink.borrow_mut().push(e.clone()), body)
        });
        let events = events.borrow().clone();
        (result, events)
    }

    fn assert_checked_before_synchronized(events: &[QuantumEvent]) {
        assert!(
            events.len().is_multiple_of(2),
            "unpaired quantum events: {events:?}"
        );
        for pair in events.chunks(2) {
            match pair {
                [QuantumEvent::Checked { label: c, .. }, QuantumEvent::Synchronized { label: s, .. }] =>
                {
                    assert_eq!(c, s, "{events:?}")
                }
                _ => panic!("a quantum was not checked before it synchronized: {events:?}"),
            }
        }
    }

    fn checks(events: &[QuantumEvent]) -> Vec<(String, u64)> {
        events
            .iter()
            .filter_map(|e| match e {
                QuantumEvent::Checked { label, scratch } => Some((label.clone(), *scratch)),
                QuantumEvent::Synchronized { .. } => None,
            })
            .collect()
    }

    fn count(events: &[QuantumEvent], label: &str) -> usize {
        checks(events).iter().filter(|(l, _)| l == label).count()
    }

    fn bytes(a: &Array) -> Vec<f32> {
        a.eval().unwrap();
        a.as_slice::<f32>().to_vec()
    }

    #[test]
    fn decode_stages_are_guarded_quanta() {
        if !in_cpu_child("decode_stages_are_guarded_quanta") {
            return;
        }
        let vae = vae(8);
        let latent = latents(4);
        let run = || vae.decode(&latent);

        let host = bytes(&run().unwrap());
        let (guarded, events) = observed(AMPLE, run);
        let guarded = guarded.unwrap();
        assert_eq!(guarded.shape(), &[1, 3, 1, 32, 32]);
        assert_eq!(bytes(&guarded), host, "guards changed numerics");
        assert_checked_before_synchronized(&events);
        assert_eq!(count(&events, DECODE_HEAD_QUANTUM), 1);
        assert_eq!(count(&events, DECODE_RESNET_QUANTUM), 2 + 4 * 3);
        assert_eq!(count(&events, DECODE_ATTENTION_QUANTUM), 1);
        assert_eq!(count(&events, DECODE_UPSAMPLE_QUANTUM), 3);
        assert_eq!(count(&events, DECODE_OUTPUT_QUANTUM), 1);
        assert_eq!(
            checks(&events).len(),
            20,
            "one check per decode stage: {events:?}"
        );

        // Refusal before construction at the first quantum.
        take_refusal();
        let (refused, refused_events) = observed(0, run);
        assert!(refused.is_err());
        assert_eq!(take_refusal().unwrap().label, DECODE_HEAD_QUANTUM);
        assert!(refused_events.is_empty(), "{refused_events:?}");

        // Admit every quantum before the first one whose scratch exceeds all earlier ones; that one
        // refuses and nothing at or past it runs.
        let all = checks(&events);
        let index = (1..all.len())
            .find(|&i| all[i].1 > all[..i].iter().map(|c| c.1).max().unwrap())
            .unwrap();
        let admitted = all[..index].iter().map(|c| c.1).max().unwrap();
        let (refused, partial) = observed(admitted + PROCESS_RESERVE_BYTES, run);
        assert!(refused.is_err());
        let refusal = take_refusal().unwrap();
        assert_eq!((refusal.label, refusal.scratch_bytes), all[index].clone());
        assert_checked_before_synchronized(&partial);
        assert_eq!(partial.len(), 2 * index, "{partial:?}");
    }

    #[test]
    fn tiled_decode_tiles_are_guarded_quanta() {
        if !in_cpu_child("tiled_decode_tiles_are_guarded_quanta") {
            return;
        }
        let vae = vae(8);
        let latent = latents(8);
        let tiling = TilingConfig::spatial_only(32, 8);
        assert!(tiling.needs_tiling(VaeTiling::QWEN_IMAGE, 1, 8, 8));
        let plan = tiling.plan(VaeTiling::QWEN_IMAGE, 1, 8, 8);
        let tiles = plan.t.len() * plan.h.len() * plan.w.len();
        assert!(tiles > 1, "fixture must genuinely tile");
        let run = || vae.decode_tiled(&latent, &tiling, None);

        let host = bytes(&run().unwrap());
        let (guarded, events) = observed(AMPLE, run);
        let guarded = guarded.unwrap();
        assert_eq!(guarded.shape(), &[1, 3, 1, 64, 64]);
        assert_eq!(bytes(&guarded), host, "guards changed numerics");
        assert_checked_before_synchronized(&events);
        assert_eq!(count(&events, DECODE_TILE_FOLD_QUANTUM), tiles);
        // The tail's stages run once per tile; the head and mid-block once.
        assert_eq!(count(&events, DECODE_UPSAMPLE_QUANTUM), 3 * tiles);
        assert_eq!(count(&events, DECODE_OUTPUT_QUANTUM), tiles);
        assert_eq!(count(&events, DECODE_HEAD_QUANTUM), 1);
        assert_eq!(count(&events, DECODE_ATTENTION_QUANTUM), 1);
        let all = checks(&events);
        assert_eq!(
            all.last().map(|c| c.0.as_str()),
            Some(DECODE_FINAL_NORMALIZE_QUANTUM),
            "finalization is checked after the last fold: {events:?}"
        );
        assert_eq!(count(&events, DECODE_FINAL_NORMALIZE_QUANTUM), 1);

        take_refusal();
        let (refused, refused_events) = observed(0, run);
        assert!(refused.is_err());
        assert_eq!(take_refusal().unwrap().label, DECODE_HEAD_QUANTUM);
        assert!(refused_events.is_empty(), "{refused_events:?}");
    }

    #[test]
    fn decode_scratch_grows_with_latent_extent_and_loaded_width() {
        if !in_cpu_child("decode_scratch_grows_with_latent_extent_and_loaded_width") {
            return;
        }
        let first = |c: i32, hw: i32, label: &str| {
            let vae = vae(c);
            let latent = latents(hw);
            let (result, events) = observed(AMPLE, || vae.decode(&latent));
            result.unwrap();
            checks(&events)
                .into_iter()
                .find(|(l, _)| l == label)
                .unwrap()
                .1
        };
        // The first mid-block resnet at latent extent: exactly its counted bytes.
        let vae8 = vae(8);
        let (resnet, _) = vae8.decoder.mid_block.parts();
        assert_eq!(
            first(8, 4, DECODE_RESNET_QUANTUM),
            resnet.quantum_bytes([1, 8, 1, 4, 4], 4)
        );
        assert!(first(8, 8, DECODE_RESNET_QUANTUM) > first(8, 4, DECODE_RESNET_QUANTUM));
        assert!(first(12, 4, DECODE_RESNET_QUANTUM) > first(8, 4, DECODE_RESNET_QUANTUM));
        // An upsampler is priced at its doubled output extent.
        let up = vae8.decoder.up_blocks[0].upsampler().unwrap();
        assert_eq!(
            first(8, 4, DECODE_UPSAMPLE_QUANTUM),
            up.quantum_bytes([1, 8, 1, 4, 4], 4)
        );
        assert!(first(8, 8, DECODE_ATTENTION_QUANTUM) > first(8, 4, DECODE_ATTENTION_QUANTUM));
    }

    /// The shape-only decode-quantum predictor matches what a decode actually checks: the exact
    /// sequence for a single-pass decode, and every label's largest check for a tiled one.
    #[test]
    fn decode_quanta_predicts_the_checked_scratch() {
        if !in_cpu_child("decode_quanta_predicts_the_checked_scratch") {
            return;
        }
        let vae = vae(8);
        let plan = QwenVae::ios_decode_attention_plan();
        let latent = latents(8);
        let (result, events) = observed(AMPLE, || vae.decode(&latent));
        result.unwrap();
        let predicted = vae.decode_quanta([1, 16, 1, 8, 8], None, plan, 4);
        let predicted: Vec<(String, u64)> = predicted
            .into_iter()
            .map(|(l, b)| (l.to_string(), b))
            .collect();
        assert_eq!(checks(&events), predicted);

        let tiling = TilingConfig::spatial_only(32, 8);
        let (result, events) = observed(AMPLE, || vae.decode_tiled(&latent, &tiling, None));
        result.unwrap();
        let predicted = vae.decode_quanta([1, 16, 1, 8, 8], Some(&tiling), plan, 4);
        let largest = |list: &[(String, u64)], label: &str| {
            list.iter().filter(|(l, _)| l == label).map(|c| c.1).max()
        };
        let predicted: Vec<(String, u64)> = predicted
            .into_iter()
            .map(|(l, b)| (l.to_string(), b))
            .collect();
        let observed_checks = checks(&events);
        for label in [
            DECODE_HEAD_QUANTUM,
            DECODE_RESNET_QUANTUM,
            DECODE_ATTENTION_QUANTUM,
            DECODE_UPSAMPLE_QUANTUM,
            DECODE_OUTPUT_QUANTUM,
            DECODE_TILE_FOLD_QUANTUM,
            DECODE_FINAL_NORMALIZE_QUANTUM,
        ] {
            assert_eq!(
                largest(&observed_checks, label),
                largest(&predicted, label),
                "{label}"
            );
        }
    }

    /// MLX's peak while `body` runs, above what was live before it (inputs pre-materialized):
    /// the worst of three runs, since MLX's CPU stream retires buffers asynchronously and one
    /// quantum's peak varies run to run.
    fn measured(body: impl Fn() -> Result<Array>) -> u64 {
        (0..MEASURE_RUNS)
            .map(|_| mlx_gen::ios_quantum::measured_peak_bytes(|| Ok(vec![body()?])).unwrap())
            .max()
            .unwrap()
    }

    /// Runs per measurement; each proof takes the worst (largest) peak.
    const MEASURE_RUNS: usize = 3;

    /// The largest ratio any counted VAE formula may hold over its measured peak.
    const MAX_RATIO: f64 = 1.5;

    /// Every decode quantum type's counted formula, proven against MLX's measured peak on the REAL
    /// stage modules at a shape where activations dominate the (pre-materialized, used-in-place)
    /// weights: head, resnet, attention (unchunked and query-chunked), upsample and output.
    #[test]
    #[ignore = "expensive fixture capture; run through the selected iOS capture workflow"]
    fn decode_stage_formulas_track_the_measured_peak() {
        if !in_cpu_child("decode_stage_formulas_track_the_measured_peak") {
            return;
        }
        let c = 32;
        let vae = vae(c);
        let hw = 16;
        let latent = latents(hw);
        let x = values(&[1, c, 1, hw, hw], 0.7, 1.0);
        let shape = [1, c as u64, 1, hw as u64, hw as u64];
        let ratios = std::cell::RefCell::new(Vec::new());
        let prove = |label: &str, formula: u64, measured: u64| {
            println!(
                "{label}: formula {formula} B, measured {measured} B, ratio {:.3}",
                formula as f64 / measured as f64
            );
            mlx_gen::ios_quantum::assert_tracks_measured(label, formula, measured, MAX_RATIO);
            ratios.borrow_mut().push(label.to_string());
        };

        prove(
            DECODE_HEAD_QUANTUM,
            vae.head_quantum_bytes(ncthw(&latent), 4),
            measured(|| vae.decode_head(&latent)),
        );
        let (resnet, attn) = vae.decoder.mid_block.parts();
        prove(
            DECODE_RESNET_QUANTUM,
            resnet.quantum_bytes(shape, 4),
            measured(|| resnet.forward(&x)),
        );
        let unbounded = mlx_gen::attention::AttentionPlan::UNBOUNDED;
        prove(
            "Qwen VAE decode attention (unchunked)",
            mlx_gen::ios_quantum::attention_block_bytes(attn.quantum_geometry(shape, 4, unbounded)),
            measured(|| attn.forward_bounded(&x, unbounded)),
        );
        let tokens = (hw * hw) as u64;
        let chunked = mlx_gen::attention::AttentionPlan::budgeted(
            mlx_gen::attention::AttentionBudget::from_score_elements(tokens * tokens / 4, true),
        );
        let geometry = attn.quantum_geometry(shape, 4, chunked);
        assert_eq!(
            geometry.query_rows,
            tokens / 4,
            "the fixture must genuinely chunk"
        );
        prove(
            "Qwen VAE decode attention (chunked)",
            mlx_gen::ios_quantum::attention_block_bytes(geometry),
            measured(|| attn.forward_bounded(&x, chunked)),
        );
        let up = vae.decoder.up_blocks[0].upsampler().unwrap();
        prove(
            DECODE_UPSAMPLE_QUANTUM,
            up.quantum_bytes(shape, 4),
            measured(|| up.forward(&x)),
        );
        prove(
            DECODE_OUTPUT_QUANTUM,
            vae.decoder.output_quantum_bytes(shape, 4),
            measured(|| vae.decoder.forward_output(&x)),
        );
        // The causal 3-D convolution every stage formula sums, on its own: a narrow (3-channel)
        // and a full-width (C-channel) conv, whose padded staging copy and im2col buffer are each
        // one page-rounded allocation (`allocated_tensor_bytes`); the exact byte count
        // under-prices them at these extents.
        let act = values(&[1, c, 1, hw, hw], 0.3, 1.0);
        act.eval().unwrap();
        prove(
            "Qwen VAE causal conv (3 channels out)",
            vae.decoder.conv_out.quantum_bytes(shape, 4),
            measured(|| vae.decoder.conv_out.forward(&act)),
        );
        prove(
            "Qwen VAE causal conv (C channels out)",
            resnet.conv1().quantum_bytes(shape, 4),
            measured(|| resnet.conv1().forward(&act)),
        );
        assert_eq!(ratios.borrow().len(), 8);
        eprintln!("IOS_CAPTURE_BODY_COMPLETED vae::quantum_tests::decode_stage_formulas_track_the_measured_peak");
    }

    /// The query-chunked mid-block attention computes the same attention as the historical single
    /// call (each query row's softmax is over every key either way), checked on CPU; the unbounded
    /// plan IS the historical single call, and a decode off device takes it.
    #[test]
    fn chunked_decode_attention_matches_the_single_call() {
        if !in_cpu_child("chunked_decode_attention_matches_the_single_call") {
            return;
        }
        let c = 16;
        let vae = vae(c);
        let (_, attn) = vae.decoder.mid_block.parts();
        let x = values(&[1, c, 1, 12, 12], 0.4, 1.5);
        // The historical path is one SDPA call: `forward` is `forward_bounded(UNBOUNDED)`, whose
        // unbounded branch is that single `scaled_dot_product_attention` (no chunk loop).
        mlx_gen::attention::reset_chunk_count();
        let single = bytes(&attn.forward(&x).unwrap());
        assert_eq!(mlx_gen::attention::last_chunk_count(), 1);
        // 144 tokens, 1000 scores per chunk -> 6 query rows per chunk, 24 chunks.
        let chunked_plan = mlx_gen::attention::AttentionPlan::budgeted(
            mlx_gen::attention::AttentionBudget::from_score_elements(1000, true),
        );
        let chunked = bytes(&attn.forward_bounded(&x, chunked_plan).unwrap());
        assert_eq!(mlx_gen::attention::last_chunk_count(), 24);
        let max_abs = single
            .iter()
            .zip(&chunked)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let scale = single.iter().map(|a| a.abs()).fold(0f32, f32::max);
        println!("chunked vs single: max |delta| {max_abs:e} over max |x| {scale}");
        assert!(max_abs <= 1e-5 * scale.max(1.0), "max |delta| {max_abs}");

        // The decode routes through the chunked plan exactly when the iOS quanta are live: a 72 x 72
        // latent (5184 tokens) chunks at DECODE_ATTENTION_SCORE_ELEMENTS, and the decode under the
        // fixture still matches the host (single-call) decode.
        let vae = self::vae(4);
        let big = latents(72);
        mlx_gen::attention::reset_chunk_count();
        let host = bytes(&vae.decode(&big).unwrap());
        assert_eq!(
            mlx_gen::attention::last_chunk_count(),
            1,
            "off device the decode attention is the single historical call"
        );
        let (guarded, events) = observed(AMPLE, || vae.decode(&big));
        let guarded = bytes(&guarded.unwrap());
        assert!(
            mlx_gen::attention::last_chunk_count() > 1,
            "under the iOS quanta the decode attention is query-chunked"
        );
        assert_eq!(count(&events, DECODE_ATTENTION_QUANTUM), 1);
        // ... and is priced per chunk: the checked scratch is the chunked geometry's.
        let (_, attn) = vae.decoder.mid_block.parts();
        let plan = mlx_gen::attention::AttentionPlan::budgeted(
            mlx_gen::attention::AttentionBudget::from_score_elements(
                super::blocks::DECODE_ATTENTION_SCORE_ELEMENTS,
                true,
            ),
        );
        let geometry = attn.quantum_geometry([1, 4, 1, 72, 72], 4, plan);
        assert!(geometry.query_rows < 5184, "{geometry:?}");
        let attention_scratch = checks(&events)
            .into_iter()
            .find(|(l, _)| l == DECODE_ATTENTION_QUANTUM)
            .unwrap()
            .1;
        assert_eq!(
            attention_scratch,
            mlx_gen::ios_quantum::attention_block_bytes(geometry)
        );
        let max_abs = host
            .iter()
            .zip(&guarded)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("iOS (chunked) vs host decode at 72x72 latent: max |delta| {max_abs:e}");
        assert!(max_abs <= 1e-4, "max |delta| {max_abs}");
    }

    /// The tile fold and the final normalization are the shared tiling loop's own quanta: each is
    /// measured in situ, between its pre-construction check and its synchronization, and its
    /// counted formula proven against that peak.
    #[test]
    #[ignore = "expensive fixture capture; run through the selected iOS capture workflow"]
    fn tile_fold_formula_tracks_the_measured_peak() {
        if !in_cpu_child("tile_fold_formula_tracks_the_measured_peak") {
            return;
        }
        let vae = vae(4);
        let latent = latents(16);
        let tiling = TilingConfig::spatial_only(64, 16);
        assert!(tiling.needs_tiling(VaeTiling::QWEN_IMAGE, 1, 16, 16));
        // One in-situ decode per run; each fold's peak is the worst of the runs (MLX's CPU stream
        // retires buffers asynchronously, so one quantum's peak varies run to run).
        let run_once = || {
            let peaks: Rc<RefCell<Vec<(String, u64, u64)>>> = Rc::new(RefCell::new(Vec::new()));
            let sink = peaks.clone();
            let baseline = std::cell::Cell::new(0u64);
            let result = with_cpu_diagnostic_headroom(AMPLE, || {
                with_cpu_diagnostic_observer(
                    move |e| match e {
                        QuantumEvent::Checked { label, scratch }
                            if label == DECODE_TILE_FOLD_QUANTUM
                                || label == DECODE_FINAL_NORMALIZE_QUANTUM =>
                        {
                            mlx_rs::memory::clear_cache();
                            baseline.set(mlx_rs::memory::get_active_memory() as u64);
                            mlx_rs::memory::reset_peak_memory();
                            sink.borrow_mut().push((label.clone(), *scratch, 0));
                        }
                        QuantumEvent::Synchronized { label, .. }
                            if label == DECODE_TILE_FOLD_QUANTUM
                                || label == DECODE_FINAL_NORMALIZE_QUANTUM =>
                        {
                            let peak = mlx_rs::memory::get_peak_memory() as u64;
                            if let Some(last) = sink.borrow_mut().last_mut() {
                                last.2 = peak.saturating_sub(baseline.get());
                            }
                        }
                        _ => {}
                    },
                    || vae.decode_tiled(&latent, &tiling, None),
                )
            });
            result.unwrap();
            peaks.take()
        };
        let mut peaks = run_once();
        for _ in 1..MEASURE_RUNS {
            let again = run_once();
            assert_eq!(again.len(), peaks.len(), "every run folds the same tiles");
            for (worst, p) in peaks.iter_mut().zip(again) {
                assert_eq!(
                    (&worst.0, worst.1),
                    (&p.0, p.1),
                    "same quantum, same formula"
                );
                worst.2 = worst.2.max(p.2);
            }
        }
        assert!(
            peaks.len() > 2,
            "the fixture must fold several tiles: {peaks:?}"
        );
        for (label, formula, measured) in peaks.iter() {
            println!(
                "{label}: formula {formula} B, measured {measured} B, ratio {:.3}",
                *formula as f64 / (*measured).max(1) as f64
            );
        }
        let largest = |label: &str| {
            peaks
                .iter()
                .filter(|p| p.0 == label)
                .max_by_key(|p| p.2)
                .cloned()
                .unwrap()
        };
        for label in [DECODE_TILE_FOLD_QUANTUM, DECODE_FINAL_NORMALIZE_QUANTUM] {
            let (_, formula, measured) = largest(label);
            // Every fold must stay under its formula; the largest must be tracked within MAX_RATIO.
            for p in peaks.iter().filter(|p| p.0 == label) {
                assert!(
                    p.1 >= p.2,
                    "{label}: formula {} below measured {}",
                    p.1,
                    p.2
                );
            }
            mlx_gen::ios_quantum::assert_tracks_measured(label, formula, measured, MAX_RATIO);
        }
        eprintln!("IOS_CAPTURE_BODY_COMPLETED vae::quantum_tests::tile_fold_formula_tracks_the_measured_peak");
    }
}
