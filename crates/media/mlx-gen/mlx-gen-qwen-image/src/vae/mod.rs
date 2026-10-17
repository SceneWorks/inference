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
    decode_quantum, rms_norm_channels, CausalConv3d, DownBlock3D, MidBlock3D, UpBlock3D, NORM_EPS,
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
    /// tile attend only within itself, changing the whole image. It is also cheap (latent res, fused
    /// SDPA), so running it full adds no meaningful memory over the single-pass decode.
    pub(super) fn forward_pre_upsample(&self, x: &Array) -> Result<Array> {
        let x = decode_quantum(
            DECODE_HEAD_QUANTUM,
            x,
            self.conv_in.out_channels(),
            1,
            0,
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
            &x,
            self.conv_out.out_channels(),
            1,
            0,
            || {
                let x = rms_norm_channels(&x, &self.norm_out, NORM_EPS)?;
                self.conv_out.forward(&silu(&x)?)
            },
        )
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
    /// so a tiled decode (sc-11747) must run this ONCE on the full latent — it is cheap (latent res,
    /// fused SDPA) and spatially global, so tiling it would corrupt the whole image. Output is the
    /// pre-upsample feature map `[B, C, T, H, W]` (spatially unchanged from the latent).
    pub fn decode_pre_upsample(&self, latents: &Array) -> Result<Array> {
        let l = to_5d(latents)?;
        // One iOS head quantum: denormalize → `post_quant_conv` → decoder `conv_in`.
        let x = decode_quantum(
            DECODE_HEAD_QUANTUM,
            &l,
            self.decoder.conv_in.out_channels(),
            1,
            0,
            || {
                let l = add(&multiply(&l, &self.std)?, &self.mean)?;
                let l = self.post_quant_conv.forward(&l)?;
                self.decoder.conv_in.forward(&l)
            },
        )?;
        self.decoder.mid_block.forward_decode(&x)
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
        // checked against the full f32 RGB output before their graphs are built. A fold holds the
        // accumulator, the padded placement and their sum (3 outputs); finalization holds output,
        // normalizer, maximum, quotient and contiguous copy (5) — the SANA `decode_tiled` accounting.
        let scale = VaeTiling::QWEN_IMAGE.spatial_scale as u64;
        let output_bytes = (l.shape()[0].max(0) as u64)
            .saturating_mul(f.max(0) as u64)
            .saturating_mul(h.max(0) as u64)
            .saturating_mul(w.max(0) as u64)
            .saturating_mul(scale)
            .saturating_mul(scale)
            .saturating_mul(3 * 4);
        let out = tiled_decode_with_finalizer(
            &head,
            &plan,
            [2, 3, 4],
            cancel,
            |tile| {
                let decoded = self.decode_upsample_tail(tile)?;
                mlx_gen::ios_quantum::before(
                    DECODE_TILE_FOLD_QUANTUM,
                    output_bytes.saturating_mul(3),
                )?;
                Ok(decoded)
            },
            |accumulators| mlx_gen::ios_quantum::finish(DECODE_TILE_FOLD_QUANTUM, accumulators),
            || {
                mlx_gen::residency::drain_allocator_cache();
            },
            |_| {
                mlx_gen::ios_quantum::before(
                    DECODE_FINAL_NORMALIZE_QUANTUM,
                    output_bytes.saturating_mul(5),
                )
            },
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
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
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
        // The first mid-block resnet at latent extent: exactly the conv-stage floor of its input.
        assert_eq!(
            first(8, 4, DECODE_RESNET_QUANTUM),
            mlx_gen::ios_quantum::conv_stage_bytes(1, 4, 4, 8)
        );
        assert!(first(8, 8, DECODE_RESNET_QUANTUM) > first(8, 4, DECODE_RESNET_QUANTUM));
        assert!(first(12, 4, DECODE_RESNET_QUANTUM) > first(8, 4, DECODE_RESNET_QUANTUM));
        // An upsampler is priced at its doubled output extent.
        assert_eq!(
            first(8, 4, DECODE_UPSAMPLE_QUANTUM),
            mlx_gen::ios_quantum::conv_stage_bytes(1, 8, 8, 8)
        );
        assert!(first(8, 8, DECODE_ATTENTION_QUANTUM) > first(8, 4, DECODE_ATTENTION_QUANTUM));
    }
}
