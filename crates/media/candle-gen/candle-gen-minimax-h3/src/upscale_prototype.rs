//! Experimental short-window H3 x2 primitives, separate from generation validation.
//! Recipe: dntpi/ComfyUI-H3-Video-Upsampler @ 36cb612ec3df30094eeb4fec66f1528925ebba73.
//! Network layout derives from LBH-123-AI (MIT; weights Apache-2.0); runner/layout derive from
//! dntpi (MIT). See scripts/acceptance/h3-upscale/THIRD_PARTY_NOTICES.md.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::{CandleError, Result};

use crate::dit::positions::{audio_position_ids, frame_grid, temporal_grid, text_position_ids};

pub const REFERENCE_COMMIT: &str = "36cb612ec3df30094eeb4fec66f1528925ebba73";
pub const CAPTION: &str = "h3upscale, increase resolution and restore fine details while preserving the original content and visual style.";

fn refuse(message: impl Into<String>) -> CandleError {
    CandleError::Msg(format!("h3-upscale prototype: {}", message.into()))
}

/// Comfy's H3 VAE encode boundary is normalized, before the learned network's
/// separate published-stat normalization. Both normalizations are intentional.
pub fn normalize_vae_raw(raw: &Tensor) -> Result<Tensor> {
    let mean = Tensor::from_slice(&crate::LATENTS_MEAN, (1, 24, 1, 1, 1), raw.device())?;
    let std = Tensor::from_slice(&crate::LATENTS_STD, (1, 24, 1, 1, 1), raw.device())?;
    Ok(raw
        .to_dtype(DType::F32)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)?)
}

/// One Euler update from the tail of Comfy's simple discrete-flow grid.
/// `denoise` selects the grid length, never the sigma itself.
pub fn recipe_sigma(denoise: f32) -> Result<Option<f32>> {
    if !denoise.is_finite() || !(0.0..=1.0).contains(&denoise) {
        return Err(refuse("denoise must be finite in [0,1]"));
    }
    if denoise == 0.0 {
        return Ok(None);
    }
    let steps = (1.0 / denoise) as usize;
    if steps > 1000 {
        return Err(refuse(
            "denoise below 0.001 is outside the discrete reference grid",
        ));
    }
    let index = 999 - (((steps - 1) as f64 * 1000.0 / steps as f64) as usize);
    let base = (index + 1) as f32 / 1000.0;
    Ok(Some(crate::denoise::schedule::shift_sigma(base, 12.0)))
}

/// Require the trained x2 guide identity, including fields upstream defaults.
pub fn validate_lora_metadata(meta: &serde_json::Value) -> Result<()> {
    for (key, want) in [
        ("reference_downscale_factor", "2"),
        ("minimax_h3_guide_spatial_version", "target_crop_v2"),
        ("minimax_h3_guide_position_version", "target_grid_stride_v1"),
        ("guide_latent_only", "true"),
        ("ss_base_model_version", "minimax_h3_ref2va"),
    ] {
        if meta.get(key).and_then(|v| v.as_str()) != Some(want) {
            return Err(refuse(format!("LoRA metadata {key} must be {want}")));
        }
    }
    let rope: serde_json::Value = serde_json::from_str(
        meta.get("minimax_h3_reference_rope")
            .and_then(|v| v.as_str())
            .ok_or_else(|| refuse("missing explicit guide RoPE metadata"))?,
    )
    .map_err(|e| refuse(e.to_string()))?;
    if rope.get("guide_rope_layout").and_then(|v| v.as_str()) != Some("overlap") {
        return Err(refuse("guide RoPE must overlap the target"));
    }
    Ok(())
}

/// Packed order `[text, guide, frozen audio, target]`. Guide covers every time
/// but only every second target patch in each spatial axis.
#[derive(Debug)]
pub struct GuideLayout {
    pub positions: Vec<[f64; 3]>,
    pub classes: Vec<u32>,
    pub tags: Vec<u32>,
    pub text_indices: Vec<u32>,
    pub video_indices: Vec<u32>,
    pub audio_indices: Vec<u32>,
    pub update_indices: Vec<u32>,
    pub guide_rows: usize,
}

impl GuideLayout {
    pub fn new(
        text: usize,
        time: usize,
        height: usize,
        width: usize,
        audio: usize,
        guided: bool,
    ) -> Result<Self> {
        if text == 0
            || time == 0
            || height == 0
            || width == 0
            || audio == 0
            || !height.is_multiple_of(4)
            || !width.is_multiple_of(4)
        {
            return Err(refuse(
                "guided canvas must align to 64px with nonempty text/media",
            ));
        }
        let (grid, widths) = frame_grid(height, width, 2, 2)?;
        let clock = temporal_grid(time, text as f64)?;
        let mut guide = Vec::new();
        let mut target = Vec::new();
        for t in clock {
            for (index, &[h, w]) in grid.iter().enumerate() {
                target.push([t, h, w]);
                if guided
                    && (index / (width / 2)).is_multiple_of(2)
                    && (index % (width / 2)).is_multiple_of(2)
                {
                    guide.push([t, h, w]);
                }
            }
        }
        let guide_rows = guide.len();
        let audio_start = text + guide_rows;
        let video_start = audio_start + 2 * audio;
        let seq = video_start + target.len();
        let mut positions = text_position_ids(text)?;
        positions.extend(guide);
        positions.extend(audio_position_ids(text, audio, 2, &widths)?);
        positions.extend(target);
        let mut classes = vec![0; seq];
        classes[text..audio_start].fill(2);
        // Frozen zero audio is supplied at CLEAN audio timestep, not the video schedule.
        classes[audio_start..video_start].fill(3);
        let mut tags = vec![0; seq];
        tags[..text].fill(1);
        tags[audio_start..video_start].fill(2);
        Ok(Self {
            positions,
            classes,
            tags,
            text_indices: (0..text as u32).collect(),
            video_indices: (text as u32..audio_start as u32)
                .chain(video_start as u32..seq as u32)
                .collect(),
            audio_indices: (audio_start as u32..video_start as u32).collect(),
            update_indices: (guide_rows as u32..(guide_rows + seq - video_start) as u32).collect(),
            guide_rows,
        })
    }
}

/// Torch align_corners=false linear interpolation on one axis. Accumulate in
/// f32 and cast to the input dtype, like the reference's fp16 interpolate.
pub fn resize_axis(x: &Tensor, axis: usize, size: usize) -> Result<Tensor> {
    let old = x.dims()[axis];
    if old == size {
        return Ok(x.clone());
    }
    if old == 0 || size == 0 {
        return Err(refuse("empty interpolation axis"));
    }
    let mut lo = Vec::with_capacity(size);
    let mut hi = Vec::with_capacity(size);
    let mut weights = Vec::with_capacity(size);
    for n in 0..size {
        let at = ((n as f64 + 0.5) * old as f64 / size as f64 - 0.5).max(0.0);
        let lower = (at.floor() as usize).min(old - 1);
        lo.push(lower as u32);
        hi.push((lower + 1).min(old - 1) as u32);
        weights.push((at - lower as f64) as f32);
    }
    let xf = x.to_dtype(DType::F32)?;
    let low = xf.index_select(&Tensor::from_vec(lo, size, x.device())?, axis)?;
    let high = xf.index_select(&Tensor::from_vec(hi, size, x.device())?, axis)?;
    let mut shape = vec![1; x.rank()];
    shape[axis] = size;
    let w = Tensor::from_vec(weights, size, x.device())?.reshape(shape)?;
    Ok(
        (low.broadcast_mul(&w.affine(-1.0, 1.0)?)? + high.broadcast_mul(&w)?)?
            .to_dtype(x.dtype())?,
    )
}

/// Torch's antialiased bicubic kernel (a=-0.5), boundary renormalization,
/// align_corners=false. Used twice on the SOURCE decode, before enlargement.
pub fn bicubic_axis(x: &Tensor, axis: usize, size: usize) -> Result<Tensor> {
    let old = x.dims()[axis];
    if size == 0 || old == 0 {
        return Err(refuse("empty bicubic axis"));
    }
    let scale = old as f64 / size as f64;
    let filter_scale = scale.max(1.);
    let cubic = |v: f64| {
        let a = v.abs();
        if a < 1. {
            ((1.5 * a - 2.5) * a) * a + 1.
        } else if a < 2. {
            ((-0.5 * a + 2.5) * a - 4.) * a + 2.
        } else {
            0.
        }
    };
    let mut matrix = vec![0f32; size * old];
    for dest in 0..size {
        let center = (dest as f64 + 0.5) * scale;
        let mut sum = 0.;
        for source in 0..old {
            let value = cubic((source as f64 + 0.5 - center) / filter_scale);
            matrix[dest * old + source] = value as f32;
            sum += value;
        }
        for source in 0..old {
            matrix[dest * old + source] /= sum as f32;
        }
    }
    let mut permutation: Vec<usize> = (0..x.rank()).filter(|&i| i != axis).collect();
    permutation.push(axis);
    let trans = x
        .to_dtype(DType::F32)?
        .permute(permutation.clone())?
        .contiguous()?;
    let mut shape = trans.dims().to_vec();
    *shape.last_mut().unwrap() = size;
    let weights = Tensor::from_vec(matrix, (size, old), x.device())?;
    let output = trans
        .reshape((x.elem_count() / old, old))?
        .matmul(&weights.t()?.contiguous()?)?
        .reshape(shape)?;
    let mut inverse = vec![0; x.rank()];
    for (i, &p) in permutation.iter().enumerate() {
        inverse[p] = i;
    }
    Ok(output.permute(inverse)?.contiguous()?.to_dtype(x.dtype())?)
}

pub fn source_guide_pixels(
    decoded_rgb: &Tensor,
    target_height: usize,
    target_width: usize,
) -> Result<Tensor> {
    let target = bicubic_axis(
        &bicubic_axis(decoded_rgb, 3, target_height)?,
        4,
        target_width,
    )?
    .clamp(0f32, 1f32)?;
    Ok(bicubic_axis(
        &bicubic_axis(&target, 3, target_height / 2)?,
        4,
        target_width / 2,
    )?
    .clamp(0f32, 1f32)?)
}

/// One native Ref2VA pass with frozen clean audio. The target alone receives
/// the reversed H3 velocity Euler update. Guide and audio are never scattered
/// back into the output. This does not call the ordinary generation validator.
pub fn refine_once(
    dit: &crate::MiniMaxH3Dit,
    context: &Tensor,
    enlarged: &Tensor,
    guide: Option<&Tensor>,
    noise: &Tensor,
    guide_noise: Option<&Tensor>,
    sigma: f32,
) -> Result<Tensor> {
    use crate::dit::model::{BlockModulation, PackedForward};
    let (_, _, time, height, width) = enlarged.dims5()?;
    let frames = (time / 5) * 17 + [0, 1, 5, 9, 13][time % 5];
    let audio = (frames as f64 * 5. / 3.).round() as usize;
    let layout = GuideLayout::new(
        context.dims()[1],
        time,
        height,
        width,
        audio,
        guide.is_some(),
    )?;
    let device = enlarged.device();
    let initial = (enlarged.affine((1. - sigma) as f64, 0.)? + noise.affine(sigma as f64, 0.)?)?;
    let target_rows = crate::patchify_video_latents(&initial, [1, 2, 2])?;
    let video_rows = if let Some(guide) = guide {
        // Clean source anchors at max(t, .999), matching upstream condition aug.
        let guide_rows = crate::patchify_video_latents(guide, [1, 2, 2])?;
        let guide_noise =
            guide_noise.ok_or_else(|| refuse("missing pinned seed444 guide noise fixture"))?;
        if guide_noise.dims() != guide_rows.dims() {
            return Err(refuse("guide noise shape mismatch"));
        }
        let guide_rows = (guide_rows.affine(0.999, 0.)? + guide_noise.affine(0.001, 0.)?)?;
        if guide_rows.dims()[1] != layout.guide_rows {
            return Err(refuse("guide latent shape mismatches strided layout"));
        }
        Tensor::cat(&[&guide_rows, &target_rows], 1)?
    } else {
        target_rows
    };
    let audio_rows = Tensor::zeros((1, 2 * audio, 32), DType::F32, device)?;
    let text_rows = dit
        .embed_context(context)
        .map_err(|e| refuse(format!("context refinement: {e}")))?;
    let timestep = 1. - sigma;
    let temb = dit
        .embed_timesteps(&[timestep, 1., timestep.max(0.999), 1.])
        .map_err(|e| refuse(format!("timestep embedding: {e}")))?;
    let modulation = dit.projections().norm_out.modulation(&temb)?;
    let classes = Tensor::from_vec(layout.classes.clone(), layout.classes.len(), device)?;
    let adaln: Vec<u32> = layout
        .classes
        .iter()
        .zip(&layout.tags)
        .map(|(&c, &tag)| c * 3 + tag)
        .collect();
    let adaln = Tensor::from_vec(adaln, layout.classes.len(), device)?;
    let tables = dit.rope().tables_from_rows(&layout.positions, device)?;
    let packed = PackedForward {
        video_rows: &video_rows,
        audio_rows: &audio_rows,
        text_rows: &text_rows,
        adaln_indices: &adaln,
        timestep_indices: &classes,
        tables: &tables,
        text_indices: &layout.text_indices,
        video_indices: &layout.video_indices,
        audio_indices: &layout.audio_indices,
    };
    let (velocity, _) = dit
        .forward_packed(&packed, BlockModulation::Temb(&temb), &modulation)
        .map_err(|e| refuse(format!("packed Ref2VA forward: {e}")))?;
    let velocity = velocity.narrow(1, layout.guide_rows, layout.update_indices.len())?;
    let velocity = crate::unpatchify_video_rows(&velocity, 24, time, height, width, [1, 2, 2])?
        .to_dtype(initial.dtype())?;
    Ok((initial + velocity.affine(sigma as f64, 0.)?)?)
}

/// Learned 24-channel Conv3d upscaler. Strict key-set loading avoids ignoring
/// a checkpoint with a different architecture or block rhythm.
pub struct LatentUpscaler {
    weights: HashMap<String, Tensor>,
    channels: usize,
    blocks: [Vec<bool>; 2],
}

impl LatentUpscaler {
    pub fn load(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        Self::from_weights(
            candle_gen::candle_core::safetensors::load(path, device)?,
            dtype,
        )
    }
    pub fn from_weights(raw: HashMap<String, Tensor>, dtype: DType) -> Result<Self> {
        let prefixed = raw.keys().any(|k| k.starts_with("upscaler."));
        let weights: HashMap<_, _> = raw
            .into_iter()
            .filter_map(|(k, v)| {
                if prefixed {
                    k.strip_prefix("upscaler.").map(|s| (s.to_owned(), v))
                } else {
                    Some((k, v))
                }
            })
            .map(|(k, v)| Ok((k, v.to_dtype(dtype)?)))
            .collect::<Result<_>>()?;
        let channels = weights
            .get("conv_in.weight")
            .ok_or_else(|| refuse("missing conv_in"))?
            .dims5()?
            .0;
        if weights["conv_in.weight"].dims()[1] != 24 || !channels.is_multiple_of(32) {
            return Err(refuse(
                "network must have 24 input channels and 32-group hidden width",
            ));
        }
        let mut expected: BTreeSet<String> = BTreeSet::new();
        for name in ["conv_in", "conv_out", "embed.0", "embed.2", "norm_out"] {
            for suffix in ["weight", "bias"] {
                expected.insert(format!("{name}.{suffix}"));
            }
        }
        let mut blocks = [Vec::new(), Vec::new()];
        for (side, name) in ["in_blocks", "out_blocks"].iter().enumerate() {
            let indices: BTreeSet<usize> = weights
                .keys()
                .filter_map(|k| {
                    k.strip_prefix(&format!("{name}."))?
                        .split('.')
                        .next()?
                        .parse()
                        .ok()
                })
                .collect();
            for (slot, index) in indices.into_iter().enumerate() {
                if slot != index {
                    return Err(refuse("gaps in network block list"));
                }
                let stem = format!("{name}.{slot}");
                let temporal = weights.contains_key(&format!("{stem}.dwconv.weight"));
                blocks[side].push(temporal);
                let modules: &[&str] = if temporal {
                    &["norm", "dwconv", "pwconv"]
                } else {
                    &[
                        "in_layers.0",
                        "in_layers.2",
                        "emb_layers.1",
                        "out_norm",
                        "out_layers.2",
                    ]
                };
                for module in modules {
                    for suffix in ["weight", "bias"] {
                        expected.insert(format!("{stem}.{module}.{suffix}"));
                    }
                }
            }
        }
        let actual: BTreeSet<_> = weights.keys().cloned().collect();
        if actual != expected {
            return Err(refuse(format!(
                "checkpoint key mismatch: missing {:?}; extra {:?}",
                expected.difference(&actual).collect::<Vec<_>>(),
                actual.difference(&expected).collect::<Vec<_>>()
            )));
        }
        Ok(Self {
            weights,
            channels,
            blocks,
        })
    }
    fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        crate::nn::linear(
            x,
            &self.weights[&format!("{name}.weight")],
            &self.weights[&format!("{name}.bias")],
        )
    }
    fn norm(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let grouped = x
            .to_dtype(DType::F32)?
            .reshape((b, 32, c / 32 * t * h * w))?;
        let mean = grouped.mean_keepdim(2)?;
        let centered = grouped.broadcast_sub(&mean)?;
        let norm = centered
            .broadcast_div(&(centered.sqr()?.mean_keepdim(2)? + 1e-5)?.sqrt()?)?
            .reshape((b, c, t, h, w))?;
        Ok(norm
            .broadcast_mul(
                &self.weights[&format!("{name}.weight")]
                    .to_dtype(DType::F32)?
                    .reshape((1, c, 1, 1, 1))?,
            )?
            .broadcast_add(
                &self.weights[&format!("{name}.bias")]
                    .to_dtype(DType::F32)?
                    .reshape((1, c, 1, 1, 1))?,
            )?
            .to_dtype(x.dtype())?)
    }
    fn conv(&self, x: &Tensor, name: &str, groups: usize) -> Result<Tensor> {
        let weight = &self.weights[&format!("{name}.weight")];
        let (out, _, kt, kh, kw) = weight.dims5()?;
        let padded = x
            .pad_with_zeros(2, kt / 2, kt / 2)?
            .pad_with_zeros(3, kh / 2, kh / 2)?
            .pad_with_zeros(4, kw / 2, kw / 2)?;
        if groups == 1 {
            return crate::vae_encoder::conv3d_ncthw(
                &padded,
                weight,
                Some(&self.weights[&format!("{name}.bias")]),
                (1, 1, 1),
            );
        }
        // Depthwise temporal convolution: multiply all channels together per tap,
        // avoiding a launch per channel and preserving the grouped weight layout.
        if groups != self.channels || out != groups || kh != 1 || kw != 1 {
            return Err(refuse("unsupported grouped conv"));
        }
        let (b, c, t, h, w) = x.dims5()?;
        let mut y = Tensor::zeros((b, c, t, h, w), x.dtype(), x.device())?;
        for tap in 0..kt {
            let coeff = weight.narrow(2, tap, 1)?.reshape((1, c, 1, 1, 1))?;
            y = (y + padded.narrow(2, tap, t)?.broadcast_mul(&coeff)?)?;
        }
        Ok(y.broadcast_add(&self.weights[&format!("{name}.bias")].reshape((1, c, 1, 1, 1))?)?)
    }
    fn block(&self, x: &Tensor, emb: &Tensor, side: usize, slot: usize) -> Result<Tensor> {
        let name = format!(
            "{}.{}",
            if side == 0 { "in_blocks" } else { "out_blocks" },
            slot
        );
        if self.blocks[side][slot] {
            let h = self.conv(
                &crate::nn::silu(&self.norm(x, &format!("{name}.norm"))?)?,
                &format!("{name}.dwconv"),
                self.channels,
            )?;
            return Ok((x + self.conv(&h, &format!("{name}.pwconv"), 1)?)?);
        }
        let h = self.conv(
            &crate::nn::silu(&self.norm(x, &format!("{name}.in_layers.0"))?)?,
            &format!("{name}.in_layers.2"),
            1,
        )?;
        let e = self.linear(&crate::nn::silu(emb)?, &format!("{name}.emb_layers.1"))?;
        let (b, c, _, _, _) = h.dims5()?;
        let scale = e
            .narrow(1, 0, c)?
            .reshape((b, c, 1, 1, 1))?
            .affine(1., 1.)?;
        let shift = e.narrow(1, c, c)?.reshape((b, c, 1, 1, 1))?;
        let h = self
            .norm(&h, &format!("{name}.out_norm"))?
            .broadcast_mul(&scale)?
            .broadcast_add(&shift)?;
        Ok((x + self.conv(&crate::nn::silu(&h)?, &format!("{name}.out_layers.2"), 1)?)?)
    }
    pub fn forward_normalized(&self, x: &Tensor, height: usize, width: usize) -> Result<Tensor> {
        let (b, c, _, h, w) = x.dims5()?;
        if c != 24 || height < h || width < w {
            return Err(refuse("expected 24-channel enlargement"));
        }
        let scale = (height as f32 / h as f32 + width as f32 / w as f32) / 2.;
        let e = Tensor::from_vec(vec![scale - 1.; b], (b, 1), x.device())?.to_dtype(x.dtype())?;
        let emb = self.linear(&crate::nn::silu(&self.linear(&e, "embed.0")?)?, "embed.2")?;
        let mut hidden = self.conv(x, "conv_in", 1)?;
        for slot in 0..self.blocks[0].len() {
            hidden = self.block(&hidden, &emb, 0, slot)?;
        }
        hidden = resize_axis(&resize_axis(&hidden, 3, height)?, 4, width)?;
        for slot in 0..self.blocks[1].len() {
            hidden = self.block(&hidden, &emb, 1, slot)?;
        }
        self.conv(
            &crate::nn::silu(&self.norm(&hidden, "norm_out")?)?,
            "conv_out",
            1,
        )
    }
    /// Takes Comfy VAE normalized latents and applies the network's additional
    /// mean/std transform. Its output remains in the VAE/DiT normalized domain.
    pub fn upscale_latents(
        &self,
        vae_normalized: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let mean = Tensor::from_slice(
            &crate::config::LATENTS_MEAN,
            (1, 24, 1, 1, 1),
            vae_normalized.device(),
        )?
        .to_dtype(vae_normalized.dtype())?;
        let std = Tensor::from_slice(
            &crate::config::LATENTS_STD,
            (1, 24, 1, 1, 1),
            vae_normalized.device(),
        )?
        .to_dtype(vae_normalized.dtype())?;
        let normalized = vae_normalized.broadcast_sub(&mean)?.broadcast_div(&std)?;
        Ok(self
            .forward_normalized(&normalized, height, width)?
            .broadcast_mul(&std)?
            .broadcast_add(&mean)?)
    }

    /// Compose the exact posterior-to-upscale boundary. The result is already
    /// normalized for Ref2VA and VAE decode; no further VAE normalization follows.
    pub fn upscale_vae_raw(
        &self,
        posterior: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let source =
            normalize_vae_raw(posterior)?.to_dtype(self.weights["conv_in.weight"].dtype())?;
        Ok(self
            .upscale_latents(&source, height, width)?
            .to_dtype(DType::F32)?)
    }
}
