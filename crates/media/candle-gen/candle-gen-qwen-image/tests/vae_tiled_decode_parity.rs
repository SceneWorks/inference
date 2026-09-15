//! sc-19753 — the Qwen-Image VAE's tail tiling is **normalization-correct**, proven weights-free.
//!
//! The audit verdict for this crate is `LOCAL_ONLY`: [`QwenVae::decode_mid`] carries the only global
//! op (`mid_attn`, a softmax over all H·W tokens) and runs once on the whole latent, while every op
//! in [`QwenVae::decode_tail`] is spatially local — convolutions, nearest upsample, and `ChanNorm`,
//! whose reduction `x.sqr().sum_keepdim(1)` is over the **channel** axis of NCHW, per `(b, h, w)`
//! position. There is **no GroupNorm anywhere in this crate** (an adversarial review claimed the tail
//! had "GroupNorm-bearing up_blocks + norm_out"; that is refuted — `vae.rs`'s own module doc says the
//! norm is "NOT GroupNorm and NOT a feature-axis RMSNorm", and the only `num_groups` in the crate is
//! an unrelated patch-merger count in `vision.rs`).
//!
//! `QwenVae::new` hard-codes the channel widths (16 → 384 → 192 → 96), so a synthetic checkpoint has
//! to carry the **real** channel counts and may only shrink the spatial extent. That makes the full
//! decoder ~71 M parameters (~286 MB f32); the fixture below builds exactly that, so these tests run
//! against the true module graph, and the spatial sizes are kept as small as the property under test
//! allows.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen_qwen_image::vae::QwenVae;

// ---------------------------------------------------------------------------------------------------
// Synthetic checkpoint
// ---------------------------------------------------------------------------------------------------

/// Emits every key [`QwenVae::new`] requires, at the shapes it demands.
struct Ckpt {
    map: HashMap<String, Tensor>,
    dev: Device,
    phase: usize,
}

impl Ckpt {
    fn new(dev: &Device) -> Self {
        Self {
            map: HashMap::new(),
            dev: dev.clone(),
            phase: 0,
        }
    }

    /// Deterministic values with an explicit scale. A cheap LCG rather than `sin` — this fills ~71 M
    /// elements and the test crate itself compiles at `opt-level = 0`.
    fn put(&mut self, shape: &[usize], name: String, scale: f32, offset: f32) {
        let n: usize = shape.iter().product();
        let mut s = (self.phase as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.phase += 1;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // top 31 bits → [0, 1) → [-1, 1)
            let u = ((s >> 33) as f32) / ((1u64 << 31) as f32) * 2.0 - 1.0;
            v.push(u * scale + offset);
        }
        self.map.insert(
            name,
            Tensor::from_vec(v, shape, &self.dev).expect("synthetic tensor"),
        );
    }

    /// A `CausalConv3d` as it ships: `[O, I, k, k, k]` + bias. Scaled by `1/sqrt(fan_in)` over the
    /// **used** taps (`causal_conv2d` keeps only the last depth slice), so activations stay O(1)
    /// through 16 residual blocks instead of overflowing to inf.
    fn causal_conv(&mut self, prefix: &str, out_c: usize, in_c: usize, k: usize) {
        let scale = 1.0 / ((in_c * k * k) as f32).sqrt();
        self.put(
            &[out_c, in_c, k, k, k],
            format!("{prefix}.weight"),
            scale,
            0.0,
        );
        self.put(&[out_c], format!("{prefix}.bias"), 0.05, 0.0);
    }

    /// A native 2-D conv (`[O, I, k, k]`) — the spatial resample and attention 1×1 convs.
    fn conv2d(&mut self, prefix: &str, out_c: usize, in_c: usize, k: usize) {
        let scale = 1.0 / ((in_c * k * k) as f32).sqrt();
        self.put(&[out_c, in_c, k, k], format!("{prefix}.weight"), scale, 0.0);
        self.put(&[out_c], format!("{prefix}.bias"), 0.05, 0.0);
    }

    /// A `ChanNorm` gamma. `ChanNorm::new` uses `get_unchecked` + `flatten_all`, so only the element
    /// count matters; ship the real ranks (`[C,1,1,1]` for resnet/`norm_out`, `[C,1,1]` for attention)
    /// so a future rank assumption would still be exercised. Centred on 1.0 — a gamma of 0 would make
    /// the norm invisible.
    fn gamma(&mut self, prefix: &str, c: usize, rank4: bool) {
        let shape: &[usize] = if rank4 { &[c, 1, 1, 1] } else { &[c, 1, 1] };
        self.put(shape, format!("{prefix}.gamma"), 0.25, 1.0);
    }

    fn resnet(&mut self, prefix: &str, in_c: usize, out_c: usize) {
        self.gamma(&format!("{prefix}.norm1"), in_c, true);
        self.causal_conv(&format!("{prefix}.conv1"), out_c, in_c, 3);
        self.gamma(&format!("{prefix}.norm2"), out_c, true);
        self.causal_conv(&format!("{prefix}.conv2"), out_c, out_c, 3);
        if in_c != out_c {
            self.causal_conv(&format!("{prefix}.conv_shortcut"), out_c, in_c, 1);
        }
    }
}

/// The complete decode-side key set of [`QwenVae::new`], mirroring its hard-coded `up_cfg` table.
fn checkpoint(dev: &Device) -> HashMap<String, Tensor> {
    let mut c = Ckpt::new(dev);
    c.causal_conv("post_quant_conv", 16, 16, 1);
    c.causal_conv("decoder.conv_in", 384, 16, 3);

    c.resnet("decoder.mid_block.resnets.0", 384, 384);
    c.gamma("decoder.mid_block.attentions.0.norm", 384, false);
    c.conv2d("decoder.mid_block.attentions.0.to_qkv", 384 * 3, 384, 1);
    c.conv2d("decoder.mid_block.attentions.0.proj", 384, 384, 1);
    c.resnet("decoder.mid_block.resnets.1", 384, 384);

    // (resnet0_in, block_width, upsampler_out?) — the table `QwenVae::new` hard-codes.
    let up_cfg: [(usize, usize, Option<usize>); 4] = [
        (384, 384, Some(192)),
        (192, 384, Some(192)),
        (192, 192, Some(96)),
        (96, 96, None),
    ];
    for (i, &(in_c, width, up_out)) in up_cfg.iter().enumerate() {
        for j in 0..3 {
            let rin = if j == 0 { in_c } else { width };
            c.resnet(&format!("decoder.up_blocks.{i}.resnets.{j}"), rin, width);
        }
        if let Some(out) = up_out {
            c.conv2d(
                &format!("decoder.up_blocks.{i}.upsamplers.0.resample.1"),
                out,
                width,
                3,
            );
        }
    }

    c.gamma("decoder.norm_out", 96, true);
    c.causal_conv("decoder.conv_out", 3, 96, 3);
    c.map
}

fn decoder(dev: &Device) -> QwenVae {
    QwenVae::new(VarBuilder::from_tensors(checkpoint(dev), DType::F32, dev))
        .expect("synthetic Qwen VAE loads through the real constructor")
}

/// Deterministic signed latents with a spatial ramp, so a crop's global attention statistics differ
/// observably from the whole field's — the property every test here turns on.
fn latent(h: usize, w: usize, dev: &Device) -> Tensor {
    let v = (0..(16 * h * w))
        .map(|i| {
            let y = (i / w % h) as f32;
            let x = (i % w) as f32;
            (i as f32 * 0.037).sin() + y * 0.05 - x * 0.08
        })
        .collect::<Vec<f32>>();
    Tensor::from_vec(v, (1, 16, h, w), dev).expect("latent")
}

fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
    (a.contiguous().unwrap() - b.contiguous().unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

/// PSNR (dB) of `b` against reference `a`, peak = `a`'s dynamic range — the codebase's seam metric
/// (PiD sc-10087 called ≥~34.75 dB "no measurable seam"). `INFINITY` when equal.
fn psnr_db(a: &Tensor, b: &Tensor) -> f64 {
    let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (lo, hi) = av
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
    let peak = (hi - lo).max(1e-6) as f64;
    let mse: f64 = av
        .iter()
        .zip(&bv)
        .map(|(x, y)| ((x - y) as f64).powi(2))
        .sum::<f64>()
        / av.len().max(1) as f64;
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (peak * peak / mse).log10()
}

/// The exact spatial radius, in mid-feature pixels, that [`QwenVae::decode_tail`] reaches: the
/// `up_blocks` contribute 6 padded 3×3 convs each at ×1/×2/×4/×8 of the mid resolution, plus the
/// upsampler and `conv_out` taps. Measured (not assumed) by `decode_tail_is_crop_decomposable_but_
/// decode_mid_is_not`, which finds the first contaminated output column exactly 12.25 mid-px from a
/// crop edge. Production keeps its tile **overlap** (512/8 = 64 px tile, 128/8 = 16 mid-px overlap)
/// above this radius, which is what lets the trapezoidal ramp hide every zero-pad boundary.
const TAIL_RADIUS_MID_PX: f32 = 12.25;

/// sc-19753 — the head/tail decomposition is an **exact** identity of the single-pass decode.
///
/// [`QwenVae::decode`] at or below the sc-10023 im2col threshold must be literally
/// `decode_tail ∘ decode_mid` — the same op sequence, not merely a close one. That is the premise the
/// other two tests rest on: they compare tiled and cropped variants against `decode`, so if `decode`
/// were quietly taking some other route those comparisons would measure the wrong thing.
///
/// Measured: max|Δ| = **0.00e0** (bit-exact) at a 8×8 latent → 64×64 output.
///
/// Mutation-discrimination (measured, not assumed): making the automatic policy tile this
/// sub-threshold decode — `DECODE_TILE_ABOVE_PX` `1536` → `32` together with the legacy fallback
/// `tile.unwrap_or((512, 128))` → `(32, 8)`, since the shipped 512-px tile spans more than this whole
/// fixture and would otherwise plan a single tile — breaks the identity: max|Δ| = **9.909301e-1** and
/// the `== 0.0` guard fires.
#[test]
fn head_tail_decomposition_is_the_single_pass_decode() {
    let dev = Device::Cpu;
    let vae = decoder(&dev);
    let z = latent(8, 8, &dev);

    let single = vae.decode(&z).unwrap();
    let composed = vae.decode_tail(&vae.decode_mid(&z).unwrap()).unwrap();
    assert_eq!(single.dims(), &[1, 3, 64, 64], "×8 spatial upsample");
    assert_eq!(composed.dims(), single.dims());

    let delta = max_abs(&single, &composed);
    assert_eq!(
        delta, 0.0,
        "the default decode must be exactly `decode_tail ∘ decode_mid`; a divergence means \
         `decode` took some other route and every tiling comparison here measures the wrong \
         reference; max|Δ|={delta:.3e}"
    );
}

/// sc-19753 — the **mechanism** proof behind this crate's `LOCAL_ONLY` audit verdict: the tail is
/// crop-decomposable and the head is not. This is a *binary* property rather than a tolerance, so it
/// cannot be satisfied accidentally.
///
/// [`QwenVae::decode_tail`] is convolutions, nearest upsample and `ChanNorm` — all spatially local —
/// so decoding a *crop* of the mid feature map reproduces the full decode **bit-exactly** everywhere
/// outside the crop's own zero-pad reach. [`QwenVae::decode_mid`] carries `mid_attn`, a softmax over
/// all H·W tokens, so a crop perturbs **every** position, including the one farthest from the crop
/// boundary. That asymmetry is the entire reason only the tail may be tiled.
///
/// Measured on this fixture (8×48 latent, CPU f32):
///  - the tail's first contaminated output column is **158 of 256**, i.e. bit-exact until exactly
///    **12.25 mid-px** from the crop edge ([`TAIL_RADIUS_MID_PX`]); the asserted 12-mid-px interior
///    sits 20 mid-px clear of it
///  - the head's residual at **column 0** — the farthest column from the crop — is **1.835257e-1**,
///    and stays ~1.8e-1 across the interior rather than decaying, which is the signature of a global
///    reduction rather than a padding artifact
///
/// Mutation-discrimination (measured, not assumed), each guard mutated alone:
///  - Make the tail's norm spatial — `ChanNorm::forward`'s `x.sqr().sum_keepdim(1)` (channel axis) →
///    `sum_keepdim(3)` (the W axis): the tail interior residual becomes **1.079054e0** and the
///    bit-exactness guard fires.
///  - Drop the global op — delete `h = self.mid_attn.forward(&h)?;` from `decode_mid`: the head's
///    column-0 residual becomes **0.00e0** and the "no safe interior" guard fires.
#[test]
fn decode_tail_is_crop_decomposable_but_decode_mid_is_not() {
    let dev = Device::Cpu;
    let vae = decoder(&dev);

    // A wide, short latent: width carries the crop question, height is kept small for CPU cost.
    let z = latent(8, 48, &dev);
    let mid = vae.decode_mid(&z).unwrap();

    const CROP: usize = 32;
    const INTERIOR: usize = 12; // mid-px, leaving a 20 mid-px margin > TAIL_RADIUS_MID_PX
    assert!(
        (CROP - INTERIOR) as f32 > TAIL_RADIUS_MID_PX,
        "the asserted interior must sit clear of the tail's receptive field, else bit-exactness \
         would be a claim about padding rather than about locality"
    );

    // Tail: spatially local ⇒ the interior is bit-exact under cropping.
    let tail_full = vae.decode_tail(&mid).unwrap();
    let tail_crop = vae
        .decode_tail(&mid.narrow(3, 0, CROP).unwrap().contiguous().unwrap())
        .unwrap();
    let tail_residual = max_abs(
        &tail_full.narrow(3, 0, INTERIOR * 8).unwrap(),
        &tail_crop.narrow(3, 0, INTERIOR * 8).unwrap(),
    );
    assert_eq!(
        tail_residual, 0.0,
        "the tail must be spatially local: decoding a crop of the mid map has to reproduce the full \
         decode bit-exactly away from the crop's own zero-pad reach, else tiling it would not be \
         normalization-correct; max|Δ|={tail_residual:.3e}"
    );

    // Head: global ⇒ no interior is safe, including the column farthest from the crop edge.
    let mid_crop = vae
        .decode_mid(&z.narrow(3, 0, CROP).unwrap().contiguous().unwrap())
        .unwrap();
    let head_col0 = max_abs(
        &mid.narrow(3, 0, 1).unwrap(),
        &mid_crop.narrow(3, 0, 1).unwrap(),
    );
    assert!(
        head_col0 > 1e-3,
        "`mid_attn` normalizes over every H·W token, so cropping the latent must perturb even \
         column 0 — the column farthest from the crop boundary. Contamination there is what proves \
         the divergence is the global softmax and not padding; max|Δ|={head_col0:.3e}"
    );
}

/// sc-19753 — the **executed defect control**: the shipped tail-only tiling tracks the single-pass
/// decode, while applying the same tiling to the *whole* decoder does not.
///
/// This is the crate's PSNR floor, and the evidence that the tiled decode is a close approximation
/// rather than an equality: each tile's padded convolutions zero-pad at their own crop boundary, so
/// the residual is a handful of sub-pixel seam values the trapezoidal partition-of-unity smooths.
/// The second arm reproduces the naive shape this story rejects — decode independent latent crops
/// through the whole decoder and stitch — where each crop's `mid_attn` softmaxes over only its own
/// quarter of the tokens.
///
/// The tile geometry mirrors the production design property rather than its absolute size: the
/// overlap (128 px = 16 mid-px) exceeds [`TAIL_RADIUS_MID_PX`], exactly as the shipped 512/128 pair
/// does, so the contaminated band of each tile falls inside the ramp where its neighbour is clean.
/// A tile *smaller* than the receptive field cannot reach this bar and would be measuring padding.
///
/// Measured on this fixture (32×32 latent, 256×256 output, CPU f32):
///  - shipped (tail-only tiling, 192/128): **47.80 dB**
///  - defect (whole-decoder quadrant crops, stitched): **21.48 dB** — a 26.32 dB gap, i.e. ~21× the
///    RMS error, and far below the ≥~34.75 dB "no measurable seam" bar this codebase uses
///
/// Mutation-discrimination (measured, not assumed): making the tail's norm spatial —
/// `ChanNorm::forward`'s `x.sqr().sum_keepdim(1)` → `sum_keepdim(3)`, i.e. a per-row global statistic
/// that a tile then computes over its own crop — collapses the shipped arm to **24.40 dB** and the gap
/// to **6.11 dB** (the defect arm moves to 18.30 dB, since the mutation degrades every arm), firing
/// both the `>= 40.0` floor and the `>= 20.0` gap guard.
#[test]
fn tail_only_tiling_tracks_the_single_pass_decode_but_whole_decoder_tiling_does_not() {
    let dev = Device::Cpu;
    let vae = decoder(&dev);
    let z = latent(32, 32, &dev);
    let single = vae.decode(&z).unwrap();

    // Arm A — the shipped shape: `decode_mid` once on the whole latent, only the tail tiled.
    let (tile_px, overlap_px) = (192u32, 128u32);
    assert!(
        (overlap_px / 8) as f32 > TAIL_RADIUS_MID_PX,
        "the overlap must exceed the tail's receptive field, as production's 512/128 pair does"
    );
    let shipped = vae
        .decode_with_tile(&z, Some((tile_px, overlap_px)))
        .unwrap();
    assert_eq!(shipped.dims(), single.dims());
    assert!(
        max_abs(&shipped, &single) != 0.0,
        "the tile plan must actually split, else this arm is the single-pass decode in disguise"
    );
    let shipped_psnr = psnr_db(&single, &shipped);

    // Arm B — the defect: tile the WHOLE decoder. Each quadrant runs its own `decode_mid`, so its
    // softmax normalizes over a quarter of the tokens.
    let quadrant = |y: usize, x: usize| {
        vae.decode(
            &z.narrow(2, y, 16)
                .unwrap()
                .narrow(3, x, 16)
                .unwrap()
                .contiguous()
                .unwrap(),
        )
        .unwrap()
    };
    let top = Tensor::cat(&[quadrant(0, 0), quadrant(0, 16)], 3).unwrap();
    let bottom = Tensor::cat(&[quadrant(16, 0), quadrant(16, 16)], 3).unwrap();
    let whole_tiled = Tensor::cat(&[top, bottom], 2).unwrap();
    assert_eq!(whole_tiled.dims(), single.dims());
    let defect_psnr = psnr_db(&single, &whole_tiled);

    assert!(
        shipped_psnr >= 40.0,
        "tail-only tiling must track the single-pass decode; PSNR={shipped_psnr:.2} dB (< 40)"
    );
    assert!(
        defect_psnr < 34.75,
        "whole-decoder tiling must fall below the codebase's no-measurable-seam bar, else the floor \
         above proves nothing; PSNR={defect_psnr:.2} dB"
    );
    assert!(
        shipped_psnr - defect_psnr >= 20.0,
        "the head/tail split must buy a large margin over whole-decoder tiling; \
         shipped={shipped_psnr:.2} dB defect={defect_psnr:.2} dB"
    );
}
