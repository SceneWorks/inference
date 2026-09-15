//! Native Candle FLUX.2-dev Mistral3/Pixtral caption upsampling.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::imageops::{checked_image_buffer_len, resize_lanczos_u8};
use candle_gen::gen_core::tokenizer::TextTokenizer;
use candle_gen::gen_core::{CancelFlag, Image};
use candle_gen::{CandleError, Result};
use candle_llm::llava::splice_image_features;

use crate::text_encoder::{Flux2PromptEncoder, UpsampleSampling};
use crate::vision::{Mistral3Projector, PixtralVisionTower};

pub const IMAGE_TOKEN_ID: i32 = 10;
const IMAGE_BREAK_TOKEN_ID: i32 = 12;
const IMAGE_END_TOKEN_ID: i32 = 13;
pub const EOS_TOKEN_ID: i32 = 2;
const PATCH_SIZE: usize = 14;
const SPATIAL_MERGE: usize = 2;
const MERGE_PATCH: usize = PATCH_SIZE * SPATIAL_MERGE;
const UPSAMPLING_MAX_AREA: f64 = 768.0 * 768.0;
const IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const IMAGE_STD: [f32; 3] = [0.268_629_55, 0.261_302_6, 0.275_777_1];

pub const DEFAULT_TEMPERATURE: f32 = 0.15;
pub const DEFAULT_MAX_NEW_TOKENS: usize = 512;
pub const MAX_NEW_TOKENS_CAP: usize = 2048;

pub const SYSTEM_MESSAGE_UPSAMPLING_T2I: &str = "You are an expert prompt engineer for FLUX.2 by Black Forest Labs. Rewrite user prompts to be more descriptive while strictly preserving their core subject and intent.\n\nGuidelines:\n1. Structure: Keep structured inputs structured (enhance within fields). Convert natural language to detailed paragraphs.\n2. Details: Add concrete visual specifics - form, scale, textures, materials, lighting (quality, direction, color), shadows, spatial relationships, and environmental context.\n3. Text in Images: Put ALL text in quotation marks, matching the prompt's language. Always provide explicit quoted text for objects that would contain text in reality (signs, labels, screens, etc.) - without it, the model generates gibberish.\n\nOutput only the revised prompt and nothing else.";

pub const SYSTEM_MESSAGE_UPSAMPLING_I2I: &str = "You are FLUX.2 by Black Forest Labs, an image-editing expert. You convert editing requests into one concise instruction (50-80 words, ~30 for brief requests).\n\nRules:\n- Single instruction only, no commentary\n- Use clear, analytical language (avoid \"whimsical,\" \"cascading,\" etc.)\n- Specify what changes AND what stays the same (face, lighting, composition)\n- Reference actual image elements\n- Turn negatives into positives (\"don't change X\" → \"keep X\")\n- Make abstractions concrete (\"futuristic\" → \"glowing cyan neon, metallic panels\")\n- Keep content PG-13\n\nOutput only the final instruction in plain text and nothing else.";

pub struct CaptionUpsampler {
    vision: Option<PixtralVisionTower>,
    projector: Option<Mistral3Projector>,
}

impl CaptionUpsampler {
    /// Load the native caption head. The Pixtral path is optional so txt2img does not retain the
    /// edit-only dense vision tower/projector; edit callers explicitly request it.
    pub fn new(
        vb: candle_gen::candle_nn::VarBuilder<'static>,
        include_vision: bool,
    ) -> candle_gen::candle_core::Result<Self> {
        let (vision, projector) = if include_vision {
            (
                Some(PixtralVisionTower::new(vb.clone())?),
                Some(Mistral3Projector::new(vb)?),
            )
        } else {
            (None, None)
        };
        Ok(Self { vision, projector })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upsample_prompt(
        &self,
        tokenizer: &TextTokenizer,
        encoder: &Flux2PromptEncoder,
        prompt: &str,
        references: &[Image],
        temperature: f32,
        max_new_tokens: usize,
        seed: u64,
        cancel: &CancelFlag,
    ) -> Result<String> {
        candle_gen::check_cancel(cancel)?;
        let (system, image) = if references.is_empty() {
            (SYSTEM_MESSAGE_UPSAMPLING_T2I, None)
        } else {
            (
                SYSTEM_MESSAGE_UPSAMPLING_I2I,
                Some(preprocess_image(references, encoder_device(encoder))?),
            )
        };
        let merged_grid = image
            .as_ref()
            .map(|(_, (gh, gw))| (gh / SPATIAL_MERGE, gw / SPATIAL_MERGE));
        let ids = build_input_ids(tokenizer, system, prompt, merged_grid)?;
        if ids.is_empty() {
            return Err(CandleError::Msg(
                "flux2 caption-upsample: formatted prompt tokenized to an empty sequence".into(),
            ));
        }
        let device = encoder_device(encoder);
        let input_ids = Tensor::from_vec(
            ids.iter().map(|&id| id.max(0) as u32).collect::<Vec<_>>(),
            (1, ids.len()),
            device,
        )?;
        let mut embeds = encoder.embed(&input_ids)?;
        if let Some((pixels, grid)) = image {
            candle_gen::check_cancel(cancel)?;
            let vision = self.vision.as_ref().ok_or_else(|| {
                CandleError::Msg(
                    "flux2 caption-upsample: image enhancement components were not loaded".into(),
                )
            })?;
            let projector = self.projector.as_ref().ok_or_else(|| {
                CandleError::Msg(
                    "flux2 caption-upsample: image enhancement projector was not loaded".into(),
                )
            })?;
            let features = vision.forward(&[&pixels], &[grid], cancel)?;
            let projected = projector.forward(&features, &[grid])?;
            embeds = splice_image_features(&embeds, &ids, &projected, IMAGE_TOKEN_ID)
                .map_err(|error| CandleError::Msg(format!("flux2 caption-upsample: {error}")))?;
        }
        let tokens = encoder.generate_from_embeds(
            &embeds,
            EOS_TOKEN_ID,
            UpsampleSampling {
                temperature,
                max_new_tokens: max_new_tokens.min(MAX_NEW_TOKENS_CAP),
                seed,
            },
            cancel,
        )?;
        let tokens: Vec<u32> = tokens.into_iter().map(|id| id.max(0) as u32).collect();
        Ok(tokenizer.decode(&tokens, true)?.trim().to_owned())
    }
}

// Candle tensors carry their device; this helper keeps the caption module independent from the
// encoder's private weight fields by deriving it from a zero-length-safe embedding probe.
fn encoder_device(encoder: &Flux2PromptEncoder) -> &Device {
    encoder.device()
}

pub fn build_input_ids(
    tokenizer: &TextTokenizer,
    system: &str,
    prompt: &str,
    merged_grid: Option<(usize, usize)>,
) -> Result<Vec<i32>> {
    let cleaned = prompt.replace("[IMG]", "");
    let text = match merged_grid {
        Some(_) => format!(
            "[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST][IMG][/INST][INST]{cleaned}[/INST]"
        ),
        None => format!("[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{cleaned}[/INST]"),
    };
    let ids = tokenizer.encode_ids(&text, true)?;
    Ok(match merged_grid {
        Some(grid) => expand_image_tokens(&ids, grid),
        None => ids,
    })
}

pub fn expand_image_tokens(ids: &[i32], grid: (usize, usize)) -> Vec<i32> {
    let (gh, gw) = grid;
    let mut out = Vec::with_capacity(ids.len() + gh.saturating_mul(gw + 1));
    let mut expanded = false;
    for &id in ids {
        if id == IMAGE_TOKEN_ID && !expanded {
            for row in 0..gh {
                out.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, gw));
                out.push(if row + 1 == gh {
                    IMAGE_END_TOKEN_ID
                } else {
                    IMAGE_BREAK_TOKEN_ID
                });
            }
            expanded = true;
        } else {
            out.push(id);
        }
    }
    out
}

fn preprocess_image(references: &[Image], device: &Device) -> Result<(Tensor, (usize, usize))> {
    let concat = concatenate_horizontal(references)?;
    let area = concat.width as f64 * concat.height as f64;
    let scale = if area > UPSAMPLING_MAX_AREA {
        (UPSAMPLING_MAX_AREA / area).sqrt()
    } else {
        1.0
    };
    let round_up = |value: u32| {
        let value = (value as f64 * scale).round().max(1.0) as usize;
        value.div_ceil(MERGE_PATCH) * MERGE_PATCH
    };
    let (width, height) = (round_up(concat.width), round_up(concat.height));
    let resized = resize_lanczos_u8(
        &concat.pixels,
        concat.height as usize,
        concat.width as usize,
        height,
        width,
    )?;
    let mut nchw = vec![0f32; 3 * height * width];
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let src = resized[(y * width + x) * 3 + c];
                nchw[(c * height + y) * width + x] = (src / 255.0 - IMAGE_MEAN[c]) / IMAGE_STD[c];
            }
        }
    }
    Ok((
        Tensor::from_vec(nchw, (1, 3, height, width), device)?.to_dtype(DType::F32)?,
        (height / PATCH_SIZE, width / PATCH_SIZE),
    ))
}

fn concatenate_horizontal(images: &[Image]) -> Result<Image> {
    if images.is_empty() {
        return Err(CandleError::Msg(
            "flux2 caption-upsample: no reference image to preprocess".into(),
        ));
    }
    for image in images {
        let expected = checked_image_buffer_len(image.width as usize, image.height as usize, 3)
            .ok_or_else(|| {
                CandleError::Msg("flux2 caption-upsample: image buffer length overflow".into())
            })?;
        if image.pixels.len() != expected {
            return Err(CandleError::Msg(format!(
                "flux2 caption-upsample: malformed {}x{} RGB image ({} bytes, expected {expected})",
                image.width,
                image.height,
                image.pixels.len()
            )));
        }
    }
    if images.len() == 1 {
        return Ok(images[0].clone());
    }
    let width: u32 = images.iter().try_fold(0u32, |sum, image| {
        sum.checked_add(image.width).ok_or_else(|| {
            CandleError::Msg("flux2 caption-upsample: concatenated width overflow".into())
        })
    })?;
    let height = images.iter().map(|image| image.height).max().unwrap_or(1);
    let output_len =
        checked_image_buffer_len(width as usize, height as usize, 3).ok_or_else(|| {
            CandleError::Msg("flux2 caption-upsample: concatenated image buffer overflow".into())
        })?;
    let mut pixels = vec![255u8; output_len];
    let mut x_offset = 0usize;
    for image in images {
        let y_offset = (height as usize - image.height as usize) / 2;
        for y in 0..image.height as usize {
            let src = y * image.width as usize * 3;
            let dst = ((y + y_offset) * width as usize + x_offset) * 3;
            let len = image.width as usize * 3;
            pixels[dst..dst + len].copy_from_slice(&image.pixels[src..src + len]);
        }
        x_offset += image.width as usize;
    }
    Ok(Image {
        width,
        height,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_placeholder_expands_to_projector_grid() {
        let ids = expand_image_tokens(&[1, IMAGE_TOKEN_ID, 2], (2, 3));
        assert_eq!(ids, vec![1, 10, 10, 10, 12, 10, 10, 10, 13, 2]);
        assert_eq!(ids.iter().filter(|&&id| id == IMAGE_TOKEN_ID).count(), 6);
    }

    #[test]
    fn literal_extra_image_tokens_are_not_expanded_twice() {
        let ids = expand_image_tokens(&[IMAGE_TOKEN_ID, 99, IMAGE_TOKEN_ID], (1, 2));
        assert_eq!(ids, vec![10, 10, 13, 99, 10]);
    }

    #[test]
    fn image_features_replace_only_placeholder_rows() {
        let device = Device::Cpu;
        let embeds = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (1, 3, 2), &device).unwrap();
        let projected = Tensor::from_vec(vec![9f32, 8.], (1, 2), &device).unwrap();
        let got =
            splice_image_features(&embeds, &[1, IMAGE_TOKEN_ID, 2], &projected, IMAGE_TOKEN_ID)
                .unwrap()
                .to_vec3::<f32>()
                .unwrap();
        assert_eq!(got, vec![vec![vec![1., 2.], vec![9., 8.], vec![5., 6.]]]);
    }

    #[test]
    fn concatenation_centers_short_images_on_white() {
        let a = Image {
            width: 1,
            height: 1,
            pixels: vec![1, 2, 3],
        };
        let b = Image {
            width: 1,
            height: 3,
            pixels: vec![4, 5, 6, 7, 8, 9, 10, 11, 12],
        };
        let out = concatenate_horizontal(&[a, b]).unwrap();
        assert_eq!((out.width, out.height), (2, 3));
        assert_eq!(&out.pixels[0..3], &[255, 255, 255]);
        assert_eq!(&out.pixels[6..9], &[1, 2, 3]);
    }
}
