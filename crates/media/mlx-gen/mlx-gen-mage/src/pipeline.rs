//! Rectified-flow sampler, native-resolution packing, and the prompt→image / edit paths —
//! **owned by sc-14041** (native-resolution work continues in sc-14043, edit in sc-14048).
//!
//! Port of `_vendor/mage_flow/pipeline.py`. The pieces, with their pinned answers:
//!
//! - **Schedule.** `FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=6.0,
//!   use_dynamic_shifting=false)` with `set_timesteps(sigmas=linspace(1, 1/N, N))` (`:37-50`) —
//!   static shift `6σ/(1+5σ)` plus a terminal 0. **Turbo is the same ladder at N = 4**, not a
//!   distilled timestep table: N=4 gives sigmas `[1.0, 0.94736844, 0.85714287, 0.66666669, 0.0]`.
//!   The step is plain Euler, `x += (σ_next − σ_cur) · v` (`:343`).
//! - **Initial latent.** [`crate::latent`], never plain `randn`.
//! - **Packing.** Latents flatten to a variable-length token sequence (`patch_size == 1`) and are
//!   packed under a fixed budget with per-sample cumulative offsets (`cu_seqlens`) instead of
//!   block-diagonal masks. Sides must be multiples of
//!   [`SIZE_MULTIPLE`]; the native range is
//!   [`MIN_SIZE`]–[`MAX_SIZE`] per side.
//! - **CFG.** `use_neg = cfg > 1.0` (`:326`, `:535`): at cfg ≤ 1 the reference builds **no**
//!   unconditional branch at all — one segment, one `cu_seqlens` pair, positive conditioning only.
//!   Both Turbo variants default there, so the CFG-off path is a first-class case, not an edge one.
//!   Under `batch_cfg` the duplicated uncond branch rotates at msrope frame index 1 — see
//!   [`crate::rope_embedder`].
//! - **Edit sequence.** `[noisy_target, ref_1, …, ref_N]` — **target first** (`:552-555`), which
//!   corrects the epic's `[τ, z_src, noisy z_tgt]` on both ordering and τ-placement: τ is the
//!   *separate text stream*, not part of the image sequence. Refs are clean latents,
//!   re-concatenated every step, and only the target tokens are stepped (`:557-565`). Frame index:
//!   target 0, ref_j → j. Refs are VAE-encoded at *target* resolution; the copy fed to the VL
//!   vision tower is long-edge capped at
//!   [`VL_COND_LONG_EDGE`](crate::config::VL_COND_LONG_EDGE).
//! - **Decode.** `vae.decode(unpack(tokens.float(), h, w))` (`:121-127`); `unpack` reshapes at
//!   `ceil(height/16) × ceil(width/16)` (`models/utils.py:36`).
//!
//! Boundary goldens for every one of these stages, and the hardened checker that verifies them
//! (76 invariants at cfg > 1, 71 at cfg ≤ 1, with `--self-test`), live in
//! `crates/media/mlx-gen/tools/`.

use mlx_gen::{Error, Progress, Result};
use mlx_rs::ops::{concatenate_axis, maximum, minimum};
use mlx_rs::{Array, Dtype};

use crate::config::{LATENT_CHANNELS, MAX_SIZE, MIN_SIZE, SIZE_MULTIPLE, VAE_DOWNSAMPLE_FACTOR};
use crate::latent::{encode_noise, GsKey};
use crate::rope_embedder::{ImgShape, PackLayout};
use crate::text_encoder::{MageTextEncoder, PromptKind};
use crate::transformer::MageTransformer;
use crate::vae::MageVae;
use std::path::Path;

/// The published Mage-Flow scheduler shift.
pub const STATIC_SHIFT: f32 = 6.0;

/// The frozen pipeline constructs an unconditional branch only above guidance scale 1.
pub const fn uses_cfg(cfg: f32) -> bool {
    cfg > 1.0
}

/// Upstream's native-resolution image-token budget for one packed DiT invocation.
///
/// This budget is applied before the optional fused-CFG duplication, exactly like the reference's
/// sample packer. It partitions samples; it never rounds or bucket-quantizes their dimensions.
pub const MAX_PACKED_IMAGE_TOKENS: i32 = 50_000;

/// One independently seeded, native-resolution generation request.
#[derive(Debug, Clone, Copy)]
pub struct GenerationSample<'a> {
    pub prompt: &'a str,
    pub negative_prompt: &'a str,
    pub height: u32,
    pub width: u32,
    pub seed: i64,
}

/// A consecutive group of samples that fits one native-resolution DiT pack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationPack {
    pub sample_range: std::ops::Range<usize>,
    pub grids: Vec<(i32, i32)>,
    pub image_tokens: i32,
}

/// Per-sample observable outputs from variable-geometry generation.
pub struct BatchGenerationTrace {
    pub samples: Vec<GenerationTrace>,
    pub packs: Vec<GenerationPack>,
}

/// The four loaded components needed for the Mage-Flow generation path.
pub struct MageFlowPipeline {
    pub text_encoder: MageTextEncoder,
    pub transformer: MageTransformer,
    pub vae: MageVae,
}

/// Observable boundary tensors retained by the real-weight parity gate.
pub struct GenerationTrace {
    pub final_tokens: Array,
    pub final_latent: Array,
    pub trajectories: Vec<Array>,
    /// Exact reference byte conversion, HWC `Uint8`.
    pub image_u8: Array,
}

impl MageFlowPipeline {
    /// Load a published diffusers snapshot (`text_encoder/`, `transformer/`, `vae/`).
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_quant(root, None)
    }

    /// Load and optionally quantize all live Linears in the TE, DiT, and VAE.
    pub fn load_with_quant(root: impl AsRef<Path>, quant_bits: Option<i32>) -> Result<Self> {
        let root = root.as_ref();
        let mut pipeline = Self {
            text_encoder: crate::text_encoder::load(root)?,
            transformer: MageTransformer::load(root.join("transformer"))?,
            vae: crate::vae::load(
                root.join("vae"),
                crate::vae::VaePart::Decode,
                Dtype::Bfloat16,
            )?,
        };
        if let Some(bits) = quant_bits {
            pipeline.text_encoder.quantize(bits)?;
            pipeline.transformer.quantize(bits)?;
            pipeline.vae.quantize(bits)?;
            verify_quantized_counts(
                "text encoder",
                pipeline.text_encoder.quantized_linear_count(),
                253,
            )?;
            verify_quantized_counts("DiT", pipeline.transformer.quantized_linear_count(), 174)?;
            let (expected_vae, packed_vae) = pipeline.vae.quantization_count();
            verify_quantized_counts("VAE", packed_vae, expected_vae)?;
        }
        Ok(pipeline)
    }

    /// Generate one decoded NCHW image in the reference's `[0,255]` float range.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: &str,
        negative_prompt: &str,
        height: u32,
        width: u32,
        steps: usize,
        cfg: f32,
        seed: i64,
        gs_key: &GsKey,
        renormalize: bool,
    ) -> Result<Array> {
        Ok(self
            .generate_trace(
                prompt,
                negative_prompt,
                height,
                width,
                steps,
                cfg,
                seed,
                gs_key,
                renormalize,
                &mut |_| {},
            )?
            .image_u8)
    }

    /// Generate while retaining parity boundaries and reporting normal platform progress.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_trace(
        &self,
        prompt: &str,
        negative_prompt: &str,
        height: u32,
        width: u32,
        steps: usize,
        cfg: f32,
        seed: i64,
        gs_key: &GsKey,
        renormalize: bool,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationTrace> {
        let mut batch = self.generate_batch_trace(
            &[GenerationSample {
                prompt,
                negative_prompt,
                height,
                width,
                seed,
            }],
            steps,
            cfg,
            gs_key,
            renormalize,
            on_progress,
        )?;
        Ok(batch.samples.swap_remove(0))
    }

    /// Generate a variable-geometry batch without resizing or bucket quantization.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_batch(
        &self,
        samples: &[GenerationSample<'_>],
        steps: usize,
        cfg: f32,
        gs_key: &GsKey,
        renormalize: bool,
    ) -> Result<Vec<Array>> {
        Ok(self
            .generate_batch_trace(samples, steps, cfg, gs_key, renormalize, &mut |_| {})?
            .samples
            .into_iter()
            .map(|trace| trace.image_u8)
            .collect())
    }

    /// Generate variable-length native-resolution packs while retaining per-sample boundaries.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_batch_trace(
        &self,
        samples: &[GenerationSample<'_>],
        steps: usize,
        cfg: f32,
        gs_key: &GsKey,
        renormalize: bool,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<BatchGenerationTrace> {
        let packs = plan_generation_packs(samples)?;
        let sigmas = mage_flow_sigmas(steps)?;
        let total_steps = steps
            .checked_mul(packs.len())
            .ok_or_else(|| Error::Msg("mage_flow: progress step count overflow".into()))?;
        let mut traces = Vec::with_capacity(samples.len());

        for (pack_index, pack) in packs.iter().enumerate() {
            let pack_samples = &samples[pack.sample_range.clone()];
            let mut texts = pack_samples
                .iter()
                .map(|sample| sample.prompt)
                .collect::<Vec<_>>();
            if uses_cfg(cfg) {
                texts.extend(pack_samples.iter().map(|sample| sample.negative_prompt));
            }
            let conditioning = self.text_encoder.encode(&texts, PromptKind::Gen)?;
            let sample_count = pack_samples.len();
            let positive_lens = conditioning.seq_lens[..sample_count]
                .iter()
                .map(|&len| len as i32)
                .collect::<Vec<_>>();
            let positive_tokens: i32 = positive_lens.iter().sum();
            let hidden = self.transformer.config().context_in_dim;
            let (cond_flat, neg_flat) = if uses_cfg(cfg) {
                let parts = conditioning.txt.split_axis(&[positive_tokens], 0)?;
                (parts[0].clone(), Some(parts[1].clone()))
            } else {
                (conditioning.txt.clone(), None)
            };
            let cond = cond_flat.reshape(&[1, positive_tokens, hidden])?;
            let negative_lens = if uses_cfg(cfg) {
                Some(
                    conditioning.seq_lens[sample_count..]
                        .iter()
                        .map(|&len| len as i32)
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            let negative = match (neg_flat, negative_lens) {
                (Some(txt), Some(lens)) => {
                    let tokens = lens.iter().sum();
                    Some((txt.reshape(&[1, tokens, hidden])?, lens))
                }
                _ => None,
            };

            let initial = pack_samples
                .iter()
                .map(|sample| {
                    initial_tokens(
                        sample.height,
                        sample.width,
                        sample.seed,
                        gs_key,
                        Dtype::Bfloat16,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let initial_refs = initial.iter().collect::<Vec<_>>();
            let tokens = concatenate_axis(&initial_refs, 1)?;
            let layout = generation_layout(&pack.grids, positive_lens)?;
            let mut trajectories = Vec::with_capacity(2);
            let out = denoise_capture(
                &self.transformer,
                tokens,
                &cond,
                layout,
                negative.as_ref().map(|(txt, lens)| (txt, lens.clone())),
                cfg,
                renormalize,
                &sigmas,
                &mut |step, latent| {
                    if sample_count == 1 && step < 2 {
                        trajectories.push(latent.clone());
                    }
                    on_progress(Progress::Step {
                        current: (pack_index * steps + step + 1) as u32,
                        total: total_steps as u32,
                    });
                },
            )?;

            let boundaries = pack
                .grids
                .iter()
                .scan(0, |offset, &(gh, gw)| {
                    *offset += gh * gw;
                    Some(*offset)
                })
                .collect::<Vec<_>>();
            let split_points = &boundaries[..boundaries.len().saturating_sub(1)];
            let outputs = out.split_axis(split_points, 1)?;
            for ((tokens, &(gh, gw)), sample) in
                outputs.into_iter().zip(pack.grids.iter()).zip(pack_samples)
            {
                let final_latent = unpack_tokens(&tokens, gh, gw)?;
                on_progress(Progress::Decoding);
                let image_u8 = decode(&self.vae, &tokens, gh, gw)?;
                debug_assert_eq!(
                    image_u8.shape(),
                    [sample.height as i32, sample.width as i32, 3]
                );
                traces.push(GenerationTrace {
                    final_tokens: tokens,
                    final_latent,
                    trajectories: if sample_count == 1 {
                        std::mem::take(&mut trajectories)
                    } else {
                        Vec::new()
                    },
                    image_u8,
                });
            }
        }
        Ok(BatchGenerationTrace {
            samples: traces,
            packs,
        })
    }
}

fn verify_quantized_counts(component: &str, packed: usize, expected: usize) -> Result<()> {
    if packed != expected {
        return Err(Error::Msg(format!(
            "mage_flow: {component} quantization packed {packed}/{expected} required projections"
        )));
    }
    Ok(())
}

/// Validate exact requested geometry and greedily preserve input order under the 50k-token budget.
pub fn plan_generation_packs(samples: &[GenerationSample<'_>]) -> Result<Vec<GenerationPack>> {
    if samples.is_empty() {
        return Err(Error::Msg(
            "mage_flow: generation batch must not be empty".into(),
        ));
    }
    let mut packs = Vec::new();
    let mut start = 0;
    let mut grids = Vec::new();
    let mut tokens = 0;
    for (index, sample) in samples.iter().enumerate() {
        let grid = latent_hw(sample.height, sample.width)?;
        let sample_tokens = grid.0 * grid.1;
        if !grids.is_empty() && tokens + sample_tokens > MAX_PACKED_IMAGE_TOKENS {
            packs.push(GenerationPack {
                sample_range: start..index,
                grids: std::mem::take(&mut grids),
                image_tokens: tokens,
            });
            start = index;
            tokens = 0;
        }
        grids.push(grid);
        tokens += sample_tokens;
    }
    packs.push(GenerationPack {
        sample_range: start..samples.len(),
        grids,
        image_tokens: tokens,
    });
    Ok(packs)
}

/// Build the exact diffusers static-shift schedule used by Mage-Flow.
///
/// The input ladder is `linspace(1, 1/N, N)`, each value is mapped through
/// `shift*s/(1+(shift-1)*s)`, and the terminal zero is appended.
pub fn mage_flow_sigmas(steps: usize) -> Result<Vec<f32>> {
    if steps == 0 {
        return Err(Error::Msg(
            "mage_flow: steps must be greater than zero".into(),
        ));
    }
    // Match ATen's endpoint-aware f32 linspace fill before applying Diffusers' f32 static shift.
    // A one-direction `start + i*step` accumulates several one-ULP errors late in the 20-step
    // ladder, which feed back through every subsequent nonlinear DiT call.
    let n = steps as f32;
    let start = 1.0f32;
    let end = 1.0 / n;
    let step = if steps > 1 {
        (end - start) / (steps - 1) as f32
    } else {
        0.0
    };
    let mut sigmas = (0..steps)
        .map(|i| {
            // ATen fills from the nearer endpoint on each half, avoiding the extra endpoint error
            // a one-direction `start + i*step` accumulates.
            let s = if i < steps / 2 {
                step.mul_add(i as f32, start)
            } else {
                (-step).mul_add((steps - i - 1) as f32, end)
            };
            STATIC_SHIFT * s / (1.0 + (STATIC_SHIFT - 1.0) * s)
        })
        .collect::<Vec<_>>();
    sigmas.push(0.0);
    Ok(sigmas)
}

/// Validate and convert output pixels to Mage's latent grid.
pub fn latent_hw(height: u32, width: u32) -> Result<(i32, i32)> {
    if !(MIN_SIZE..=MAX_SIZE).contains(&height) || !(MIN_SIZE..=MAX_SIZE).contains(&width) {
        return Err(Error::Msg(format!(
            "mage_flow: {width}x{height} must be within {MIN_SIZE}..={MAX_SIZE} per side"
        )));
    }
    if !height.is_multiple_of(SIZE_MULTIPLE) || !width.is_multiple_of(SIZE_MULTIPLE) {
        return Err(Error::Msg(format!(
            "mage_flow: {width}x{height} must be divisible by {SIZE_MULTIPLE}"
        )));
    }
    Ok((
        (height / VAE_DOWNSAMPLE_FACTOR) as i32,
        (width / VAE_DOWNSAMPLE_FACTOR) as i32,
    ))
}

/// Create the required Gaussian-Shading initial latent and flatten it to image tokens.
pub fn initial_tokens(
    height: u32,
    width: u32,
    seed: i64,
    key: &GsKey,
    dtype: Dtype,
) -> Result<Array> {
    let (gh, gw) = latent_hw(height, width)?;
    encode_noise(
        (LATENT_CHANNELS as usize, gh as usize, gw as usize),
        key,
        seed,
        dtype,
    )?
    .transpose_axes(&[0, 2, 3, 1])?
    .reshape(&[1, gh * gw, LATENT_CHANNELS])
    .map_err(Into::into)
}

/// Convert one packed token stream back to the NCHW latent consumed by Mage-VAE.
pub fn unpack_tokens(tokens: &Array, gh: i32, gw: i32) -> Result<Array> {
    if tokens.shape() != [1, gh * gw, LATENT_CHANNELS] {
        return Err(Error::Msg(format!(
            "mage_flow: expected tokens [1, {}, {}], got {:?}",
            gh * gw,
            LATENT_CHANNELS,
            tokens.shape()
        )));
    }
    Ok(tokens
        .reshape(&[1, gh, gw, LATENT_CHANNELS])?
        .transpose_axes(&[0, 3, 1, 2])?)
}

/// Combine conditional and unconditional velocity exactly as the reference.
pub fn cfg_velocity(cond: &Array, unc: &Array, cfg: f32, renormalize: bool) -> Result<Array> {
    if cond.shape() != unc.shape() {
        return Err(Error::Msg("mage_flow: CFG velocity shapes differ".into()));
    }
    // Torch keeps tensor/scalar CFG arithmetic at the velocity tensor's dtype. An f32 MLX
    // scalar would promote the guided velocity, causing the scheduler to retain f32 latents
    // after its documented cast back to `model_output.dtype`.
    let scale = Array::from_slice(&[cfg], &[1]).as_dtype(cond.dtype())?;
    let guided = unc.add(&cond.subtract(unc)?.multiply(&scale)?)?;
    if !renormalize {
        return guided.as_dtype(cond.dtype()).map_err(Into::into);
    }
    // Per-token L2 norm over channels, matching torch.norm(..., dim=-1, keepdim=True).
    let cond_norm = cond.multiply(cond)?.sum_axis(-1, true)?.sqrt()?;
    let guided_norm = guided
        .multiply(&guided)?
        .sum_axis(-1, true)?
        .sqrt()?
        .add(Array::from_slice(&[1e-6f32], &[1]))?;
    Ok(guided
        .multiply(&cond_norm.divide(&guided_norm)?)?
        .as_dtype(cond.dtype())?)
}

/// One deterministic Diffusers `FlowMatchEulerDiscreteScheduler.step`.
fn flow_euler_step(sample: &Array, model_output: &Array, delta: f32) -> Result<Array> {
    let model_dtype = model_output.dtype();
    Ok(sample
        .as_dtype(Dtype::Float32)?
        .add(
            &model_output
                .as_dtype(Dtype::Float32)?
                .multiply(Array::from_slice(&[delta], &[1]))?,
        )?
        .as_dtype(model_dtype)?)
}

/// Run Mage's rectified-flow Euler loop over already-packed text and image streams.
///
/// At `cfg > 1`, conditional and unconditional branches are packed into one transformer
/// forward. [`PackLayout::fused_cfg`] deliberately preserves the reference's shifted frame index
/// for the appended unconditional image shapes.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    transformer: &MageTransformer,
    img: Array,
    cond_txt: &Array,
    cond_layout: PackLayout,
    negative: Option<(&Array, Vec<i32>)>,
    cfg: f32,
    renormalize: bool,
    sigmas: &[f32],
) -> Result<Array> {
    denoise_capture(
        transformer,
        img,
        cond_txt,
        cond_layout,
        negative,
        cfg,
        renormalize,
        sigmas,
        &mut |_, _| {},
    )
}

#[allow(clippy::too_many_arguments)]
fn denoise_capture(
    transformer: &MageTransformer,
    mut img: Array,
    cond_txt: &Array,
    cond_layout: PackLayout,
    negative: Option<(&Array, Vec<i32>)>,
    cfg: f32,
    renormalize: bool,
    sigmas: &[f32],
    on_step: &mut dyn FnMut(usize, &Array),
) -> Result<Array> {
    if sigmas.len() < 2 {
        return Err(Error::Msg(
            "mage_flow: scheduler needs a terminal sigma".into(),
        ));
    }
    let use_cfg = uses_cfg(cfg);
    if use_cfg && negative.is_none() {
        return Err(Error::Msg(
            "mage_flow: cfg > 1 requires negative conditioning".into(),
        ));
    }
    let cond_ctx = transformer.pack_context(cond_layout.clone())?;
    for (step, pair) in sigmas.windows(2).enumerate() {
        let velocity = if use_cfg {
            let (neg_txt, neg_lens) = negative.as_ref().unwrap();
            let fused_layout = cond_layout.fused_cfg(neg_lens)?;
            let fused_ctx = transformer.pack_context(fused_layout)?;
            let fused_img = concatenate_axis(&[&img, &img], 1)?;
            on_step(step, &fused_img);
            let fused_txt = concatenate_axis(&[cond_txt, neg_txt], 1)?;
            let sigma = Array::from_slice(
                &vec![pair[0]; fused_ctx.segments()],
                &[fused_ctx.segments() as i32],
            );
            let out = transformer.forward(&fused_img, &fused_txt, &sigma, &fused_ctx)?;
            let n = img.shape()[1];
            let parts = out.split_axis(&[n], 1)?;
            cfg_velocity(&parts[0], &parts[1], cfg, renormalize)?
        } else {
            on_step(step, &img);
            let sigma = Array::from_slice(
                &vec![pair[0]; cond_ctx.segments()],
                &[cond_ctx.segments() as i32],
            );
            transformer.forward(&img, cond_txt, &sigma, &cond_ctx)?
        };
        // Diffusers' FlowMatchEulerDiscreteScheduler upcasts the sample, performs the complete
        // Euler addition in f32, then casts the result back to model_output.dtype. Casting the
        // scaled velocity before the add introduces an extra bf16 rounding at every step.
        img = flow_euler_step(&img, &velocity, pair[1] - pair[0])?;
        mlx_rs::transforms::eval([&img])?;
    }
    Ok(img)
}

/// Decode a final token stream and apply the reference's clamp/range conversion.
pub fn decode(vae: &MageVae, tokens: &Array, gh: i32, gw: i32) -> Result<Array> {
    let pixels = vae.decode(&unpack_tokens(tokens, gh, gw)?)?;
    let pixels = maximum(&pixels, Array::from_slice(&[-1.0f32], &[1]))?;
    let pixels = minimum(&pixels, Array::from_slice(&[1.0f32], &[1]))?;
    let scaled = pixels
        .add(Array::from_slice(&[1.0f32], &[1]))?
        .multiply(Array::from_slice(&[127.5f32], &[1]))?;
    Ok(scaled
        .as_dtype(Dtype::Uint8)?
        .transpose_axes(&[0, 2, 3, 1])?
        .reshape(&[
            gh * VAE_DOWNSAMPLE_FACTOR as i32,
            gw * VAE_DOWNSAMPLE_FACTOR as i32,
            3,
        ])?)
}

/// Generation layout for a list of latent grids and encoded prompt lengths.
pub fn generation_layout(grids: &[(i32, i32)], txt_lens: Vec<i32>) -> Result<PackLayout> {
    PackLayout::generation(
        grids.iter().map(|&(h, w)| ImgShape::latent(h, w)).collect(),
        txt_lens,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb8_conversion_truncates_like_torch_byte() {
        let pixels = Array::from_slice(&[-1.0f32, -0.5, 0.0, 0.5, 1.0], &[1, 1, 1, 5]);
        let scaled = pixels
            .add(Array::from_slice(&[1.0f32], &[1]))
            .unwrap()
            .multiply(Array::from_slice(&[127.5f32], &[1]))
            .unwrap()
            .as_dtype(Dtype::Uint8)
            .unwrap();
        mlx_rs::transforms::eval([&scaled]).unwrap();
        assert_eq!(scaled.as_slice::<u8>(), &[0, 63, 127, 191, 255]);
    }

    #[test]
    fn exact_twenty_step_static_shift_schedule() {
        let s = mage_flow_sigmas(20).unwrap();
        assert_eq!(
            s,
            vec![
                1.0, 0.99130434, 0.98181814, 0.97142863, 0.96000004, 0.94736844, 0.9333333,
                0.917647, 0.90000004, 0.88000005, 0.85714287, 0.83076924, 0.8, 0.76363635, 0.72,
                0.6666667, 0.6, 0.51428574, 0.4, 0.24000001, 0.0,
            ],
            "every sigma must match the torch/diffusers CPU oracle bit-for-bit"
        );
        assert!(s.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn exact_thirty_step_base_static_shift_schedule() {
        let s = mage_flow_sigmas(30).unwrap();
        assert_eq!(
            s,
            vec![
                1.0, 0.9942857, 0.9882353, 0.98181814, 0.97499996, 0.96774185, 0.96000004,
                0.9517242, 0.9428571, 0.9333334, 0.92307687, 0.912, 0.90000004, 0.8869566,
                0.87272733, 0.85714287, 0.8399999, 0.8210527, 0.79999995, 0.77647054, 0.75,
                0.71999997, 0.68571424, 0.6461538, 0.59999996, 0.54545456, 0.48, 0.39999998, 0.3,
                0.17142859, 0.0,
            ],
            "all 31 Base sigmas must match the frozen Torch/Diffusers CPU oracle bit-for-bit"
        );
        assert!(s.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn turbo_uses_the_same_shifted_four_step_ladder_and_no_cfg_branch() {
        let sigmas = mage_flow_sigmas(4).unwrap();
        assert_eq!(sigmas.len(), 5);
        let expected = [1.0, 0.94736844, 0.85714287, 0.6666667, 0.0];
        for (index, (got, want)) in sigmas.iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "Turbo sigma {index}: got {got}, want {want}"
            );
        }
        assert!(!uses_cfg(1.0));
        assert!(uses_cfg(1.0001));
    }

    #[test]
    fn cfg_and_optional_norm_match_reference_formula() {
        let cond = Array::from_slice(&[3.0f32, 4.0], &[1, 1, 2]);
        let unc = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2]);
        let plain = cfg_velocity(&cond, &unc, 5.0, false).unwrap();
        assert_eq!(plain.as_slice::<f32>(), &[11.0, 12.0]);
        let normed = cfg_velocity(&cond, &unc, 5.0, true).unwrap();
        let norm =
            (normed.as_slice::<f32>()[0].powi(2) + normed.as_slice::<f32>()[1].powi(2)).sqrt();
        assert!((norm - 5.0).abs() < 1e-5);

        let cond_bf16 = cond.as_dtype(Dtype::Bfloat16).unwrap();
        let unc_bf16 = unc.as_dtype(Dtype::Bfloat16).unwrap();
        assert_eq!(
            cfg_velocity(&cond_bf16, &unc_bf16, 5.0, false)
                .unwrap()
                .dtype(),
            Dtype::Bfloat16,
            "Torch scalar CFG arithmetic preserves the model-output dtype"
        );
    }

    #[test]
    fn euler_add_rounds_only_after_the_f32_add() {
        let sample = Array::from_slice(&[1.390625f32], &[1])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let velocity = Array::from_slice(&[1.859375f32], &[1])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let delta = -0.05263156f32;
        let got = flow_euler_step(&sample, &velocity, delta).unwrap();
        let pre_add_rounded = sample
            .add(
                velocity
                    .as_dtype(Dtype::Float32)
                    .unwrap()
                    .multiply(Array::from_slice(&[delta], &[1]))
                    .unwrap()
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap(),
            )
            .unwrap();
        mlx_rs::transforms::eval([&got, &pre_add_rounded]).unwrap();
        let got_f32 = got.as_dtype(Dtype::Float32).unwrap();
        let wrong_f32 = pre_add_rounded.as_dtype(Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&got_f32, &wrong_f32]).unwrap();
        assert_ne!(
            got_f32.as_slice::<f32>(),
            wrong_f32.as_slice::<f32>(),
            "fixture must discriminate Diffusers' post-add cast from pre-add rounding"
        );
    }

    #[test]
    fn size_and_pack_round_trip_are_explicit() {
        assert_eq!(latent_hw(1024, 1024).unwrap(), (64, 64));
        assert_eq!(latent_hw(512, 2048).unwrap(), (32, 128));
        assert_eq!(latent_hw(2048, 512).unwrap(), (128, 32));
        assert_eq!(latent_hw(688, 1232).unwrap(), (43, 77));
        assert!(latent_hw(496, 512).is_err());
        assert!(latent_hw(512, 2064).is_err());
        assert!(latent_hw(1023, 1024).is_err());
        let x = Array::from_slice(
            &vec![0.0f32; 2 * 3 * LATENT_CHANNELS as usize],
            &[1, 2 * 3, LATENT_CHANNELS],
        );
        assert_eq!(unpack_tokens(&x, 2, 3).unwrap().shape(), [1, 128, 2, 3]);
    }

    #[test]
    fn native_geometry_packs_are_exact_and_budgeted() {
        let samples = [
            GenerationSample {
                prompt: "a",
                negative_prompt: "",
                height: 2048,
                width: 2048,
                seed: 1,
            },
            GenerationSample {
                prompt: "b",
                negative_prompt: "",
                height: 2048,
                width: 2048,
                seed: 2,
            },
            GenerationSample {
                prompt: "c",
                negative_prompt: "",
                height: 2048,
                width: 2048,
                seed: 3,
            },
            GenerationSample {
                prompt: "d",
                negative_prompt: "",
                height: 512,
                width: 2048,
                seed: 4,
            },
            GenerationSample {
                prompt: "e",
                negative_prompt: "",
                height: 688,
                width: 1232,
                seed: 5,
            },
        ];
        let packs = plan_generation_packs(&samples).unwrap();
        assert_eq!(packs.len(), 2);
        assert_eq!(packs[0].sample_range, 0..3);
        assert_eq!(packs[0].image_tokens, 49_152);
        assert_eq!(packs[0].grids, vec![(128, 128); 3]);
        assert_eq!(packs[1].sample_range, 3..5);
        assert_eq!(packs[1].grids, vec![(32, 128), (43, 77)]);
        assert!(packs
            .iter()
            .all(|pack| pack.image_tokens <= MAX_PACKED_IMAGE_TOKENS));
    }

    #[test]
    fn mixed_layout_carries_axes_boundaries_and_fused_cfg_segments() {
        let layout = generation_layout(&[(32, 128), (128, 32), (43, 77)], vec![7, 11, 13]).unwrap();
        // These image/text lens and cumulative boundaries are the frozen reference's concrete
        // representation of the story's "axes_lens"; MSRoPE additionally consumes img_shapes.
        assert_eq!(layout.img_lens(), &[4096, 4096, 3311]);
        assert_eq!(layout.img_cu(), vec![0, 4096, 8192, 11503]);
        assert_eq!(layout.txt_cu(), vec![0, 7, 18, 31]);
        assert_eq!(layout.img_shapes().len(), 3);

        let fused = layout.fused_cfg(&[5, 9, 15]).unwrap();
        assert_eq!(fused.segments(), 6);
        assert_eq!(fused.img_lens(), &[4096, 4096, 3311, 4096, 4096, 3311]);
        assert_eq!(fused.txt_lens(), &[7, 11, 13, 5, 9, 15]);
        assert_eq!(fused.img_tokens(), 23_006);
    }

    #[test]
    fn quant_audit_rejects_one_omitted_small_projection() {
        // Mutation discriminator: skipping even one small projection (for example a timestep
        // `linear_2`) changes 253→252 and must fail, independent of total memory movement.
        assert!(verify_quantized_counts("text encoder", 252, 253).is_err());
        assert!(verify_quantized_counts("DiT", 173, 174).is_err());
        assert!(verify_quantized_counts("VAE", 4, 5).is_err());
        assert!(verify_quantized_counts("VAE", 5, 5).is_ok());
    }
}
