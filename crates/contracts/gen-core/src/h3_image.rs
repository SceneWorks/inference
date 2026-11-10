//! MiniMax-H3-Image request contract: size and canvas rules, the published aspect table, the fixed
//! 6-step sigma list, and the refusals a request must pass before any backend runs. Tensor-free:
//! MLX, Candle and the shared-generation runner all read these as data.
//!
//! Sources (read 2026-11-10): `huggingface/diffusers` main,
//! `src/diffusers/modular_pipelines/minimax_h3_image/` (`before_denoise.py` canvas and validation,
//! `modular_pipeline.py` multiple/max sequence length, `decoders.py` resize back), plus the
//! MiniMaxAI/MiniMax-H3-Image model card and `pdd_config.json`.
//!
//! Upstream enforces NO maximum output area or edge: the requested size only has to be positive.
//! The 32 px stride applies to the derived generation canvas, not to the requested size, which is
//! resized back from the canvas. The card's "about 4 MP to 20 MP" range is advisory only. This
//! family is NOT bound by H3 video's `CANVAS_MAX_PIXELS` (1.03 MP) or `MAX_CANVAS_EDGE` (2016).

use crate::Error;
use thiserror::Error as ThisError;

/// Canvas multiple, pixels per token edge: VAE 16x spatial times DiT patch 2x2
/// (`canvas_multiple`, modular_pipeline.py:69).
pub const SIZE_STRIDE: u32 = 32;
/// Smallest accepted `min_generation_area` (upstream parameter range, pixels).
pub const MIN_GENERATION_AREA_MIN: u64 = 1024;
/// Largest accepted `min_generation_area`.
pub const MIN_GENERATION_AREA_MAX: u64 = 16_777_216;
/// Upstream default `min_generation_area` (2048x2048).
pub const MIN_GENERATION_AREA_DEFAULT: u64 = 4_194_304;
/// Advisory working range (not enforced; upstream has no maximum): about 4 MP to 20 MP.
pub const WORKING_AREA_MIN: u64 = 4 * 1_048_576;
/// Upper end of the advisory working range.
pub const WORKING_AREA_MAX: u64 = 20 * 1_048_576;
/// Maximum reference images for edit (model card; diffusers does not enforce a count).
pub const MAX_REFERENCES: usize = 16;
/// Prompt is truncated upstream to this many tokens.
pub const MAX_PROMPT_TOKENS: usize = 4096;
/// The only step count the distilled sampler supports.
pub const FIXED_STEPS: u32 = 6;
/// The only guidance value (no classifier-free guidance).
pub const FIXED_GUIDANCE: f32 = 1.0;

/// The 6-step schedule from `pdd_config.json` (7 boundaries, `1.0` down to `0.0`). The scheduler
/// uses velocity euler updates and does NOT shift these sigmas.
pub const SIGMAS: [f32; 7] = [
    1.0,
    0.967_523_7,
    0.887_548_5,
    0.75,
    0.532_773_9,
    0.232_008_88,
    0.0,
];

/// [`SIGMAS`] exactly as `pdd_config.json` / diffusers' `sample_sigmas` publish them (float64).
/// The sampler derives its float32 grid from these, not from the rounded [`SIGMAS`]; see
/// [`sampler_sigmas`].
pub const SIGMAS_F64: [f64; 7] = [
    1.0,
    0.967_523_694_038_391_1,
    0.887_548_506_259_918_2,
    0.75,
    0.532_773_911_952_972_4,
    0.232_008_874_416_351_32,
    0.0,
];

/// The float32 sigma grid the released sampler steps on (`MiniMaxH3ImageSetTimestepsStep`,
/// before_denoise.py): every published sigma is rounded through its float32 timestep
/// `t = f32(1 - sigma)` and back, `sigma' = 1 - t` in float32, with the terminal `0.0` kept. Below
/// `sigma = 0.5` that round trip moves a sigma by an ulp, so this is not [`SIGMAS`].
pub fn sampler_sigmas() -> [f32; 7] {
    let mut out = [0.0f32; 7];
    for (slot, &s) in out.iter_mut().zip(SIGMAS_F64.iter()).take(6) {
        let t = (1.0f64 - s) as f32;
        *slot = 1.0f32 - t;
    }
    out
}

/// One published size: label and exact pixel dimensions (about 16 MP each).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AspectEntry {
    pub label: &'static str,
    pub width: u32,
    pub height: u32,
}

/// Published "Common 16 MP sizes", including the portrait forms the card lists.
pub const ASPECT_TABLE: [AspectEntry; 8] = [
    AspectEntry {
        label: "1:1",
        width: 4096,
        height: 4096,
    },
    AspectEntry {
        label: "4:3",
        width: 4736,
        height: 3552,
    },
    AspectEntry {
        label: "3:4",
        width: 3552,
        height: 4736,
    },
    AspectEntry {
        label: "3:2",
        width: 5024,
        height: 3360,
    },
    AspectEntry {
        label: "2:3",
        width: 3360,
        height: 5024,
    },
    AspectEntry {
        label: "16:9",
        width: 5472,
        height: 3072,
    },
    AspectEntry {
        label: "9:16",
        width: 3072,
        height: 5472,
    },
    AspectEntry {
        label: "21:9",
        width: 6272,
        height: 2688,
    },
];

/// Why a H3-Image request was refused. Each variant names the limit it broke.
#[derive(Debug, Clone, PartialEq, ThisError)]
pub enum H3ImageRefusal {
    #[error("size {width}x{height} must be positive on both sides")]
    SizeNotPositive { width: u32, height: u32 },
    #[error("min_generation_area {value} is outside {MIN_GENERATION_AREA_MIN}..={MIN_GENERATION_AREA_MAX}")]
    MinGenerationAreaRange { value: u64 },
    #[error("steps {steps} refused: H3-Image is fixed at {FIXED_STEPS} steps")]
    Steps { steps: u32 },
    #[error("guidance {guidance} refused: H3-Image has no CFG; guidance must be {FIXED_GUIDANCE} or omitted")]
    Guidance { guidance: f32 },
    #[error("negative prompt refused: H3-Image does not support one")]
    NegativePrompt,
    #[error("{count} reference images refused: at most {MAX_REFERENCES} are supported")]
    TooManyReferences { count: usize },
    #[error(
        "reference image {index} is {width}x{height}: every reference needs at least \
         {MIN_REFERENCE_EDGE} pixels on each side"
    )]
    ReferenceTooSmall {
        index: usize,
        width: u32,
        height: u32,
    },
    #[error(
        "reference image {index} carries strength {strength}: H3-Image edit has no per-reference \
         strength; omit it"
    )]
    ReferenceStrength { index: usize, strength: f32 },
}

impl From<H3ImageRefusal> for Error {
    fn from(r: H3ImageRefusal) -> Self {
        Error::Unsupported(r.to_string())
    }
}

/// Python `round` (half to even) to the nearest multiple, floored at one multiple.
fn round_to_multiple(value: f64) -> u32 {
    let m = f64::from(SIZE_STRIDE);
    ((value / m).round_ties_even() * m).max(m) as u32
}

/// The `(width, height)` canvas upstream generates a `width x height` output on
/// (`get_image_canvas_size`, before_denoise.py:94). An image below `min_generation_area` is scaled
/// up by `sqrt(min_generation_area / area)` at the same aspect; either way both sides round to the
/// nearest multiple of [`SIZE_STRIDE`]. The output is resized back from this canvas to the
/// requested size when they differ (decoders.py:165).
pub fn generation_canvas(width: u32, height: u32, min_generation_area: u64) -> (u32, u32) {
    let (w, h) = (f64::from(width), f64::from(height));
    let scale = (min_generation_area as f64 / (w * h)).sqrt().max(1.0);
    (round_to_multiple(w * scale), round_to_multiple(h * scale))
}

/// Check an output size: upstream only requires it to be positive (before_denoise.py:212).
pub fn validate_size(width: u32, height: u32) -> Result<(), H3ImageRefusal> {
    if width == 0 || height == 0 {
        return Err(H3ImageRefusal::SizeNotPositive { width, height });
    }
    Ok(())
}

/// Check an optional `min_generation_area` against upstream's range (before_denoise.py:214).
pub fn validate_min_generation_area(value: Option<u64>) -> Result<(), H3ImageRefusal> {
    match value {
        Some(v) if !(MIN_GENERATION_AREA_MIN..=MIN_GENERATION_AREA_MAX).contains(&v) => {
            Err(H3ImageRefusal::MinGenerationAreaRange { value: v })
        }
        _ => Ok(()),
    }
}

/// Check the whole request. `guidance` of `None` means omitted.
pub fn validate_request(
    width: u32,
    height: u32,
    min_generation_area: Option<u64>,
    steps: u32,
    guidance: Option<f32>,
    has_negative_prompt: bool,
    reference_count: usize,
) -> Result<(), H3ImageRefusal> {
    validate_size(width, height)?;
    validate_min_generation_area(min_generation_area)?;
    if steps != FIXED_STEPS {
        return Err(H3ImageRefusal::Steps { steps });
    }
    if let Some(g) = guidance {
        if g != FIXED_GUIDANCE {
            return Err(H3ImageRefusal::Guidance { guidance: g });
        }
    }
    if has_negative_prompt {
        return Err(H3ImageRefusal::NegativePrompt);
    }
    if reference_count > MAX_REFERENCES {
        return Err(H3ImageRefusal::TooManyReferences {
            count: reference_count,
        });
    }
    Ok(())
}

/// Per-row modality tag of an image row (`MINIMAX_H3_IMAGE_TAG`, modular_pipeline.py). Tags index
/// the transformer's AdaLN table, so the values are a checkpoint contract: image rows reuse H3's
/// video slot.
pub const IMAGE_TAG: i32 = 0;
/// Per-row modality tag of a text row (`MINIMAX_H3_TEXT_TAG`).
pub const TEXT_TAG: i32 = 1;
/// VAE spatial compression: one latent cell per 16x16 pixels.
pub const VAE_SPATIAL_RATIO: u32 = 16;

/// The `text2image` presentation's token ids: the prompt tokenized verbatim (no chat template, no
/// special tokens) and truncated to [`MAX_PROMPT_TOKENS`] (encoders.py,
/// `MiniMaxH3ImageTextEncoderStep`). An empty presentation is refused, as upstream refuses an empty
/// prompt.
pub fn t2i_prompt_ids(mut ids: Vec<i32>) -> Result<Vec<i32>, Error> {
    if ids.is_empty() {
        return Err(Error::Unsupported(
            "MiniMax-H3-Image needs a non-empty prompt: the packed sequence has no text rows \
             without one"
                .into(),
        ));
    }
    ids.truncate(MAX_PROMPT_TOKENS);
    Ok(ids)
}

/// The `(latent_height, latent_width)` a `width x height` canvas packs to. The canvas must be a
/// [`SIZE_STRIDE`] multiple, which [`generation_canvas`] guarantees.
pub fn latent_grid(canvas_width: u32, canvas_height: u32) -> Result<(u32, u32), Error> {
    if canvas_width == 0
        || canvas_height == 0
        || !canvas_width.is_multiple_of(SIZE_STRIDE)
        || !canvas_height.is_multiple_of(SIZE_STRIDE)
    {
        return Err(Error::Msg(format!(
            "MiniMax-H3-Image canvas {canvas_width}x{canvas_height} must be a positive multiple of \
             {SIZE_STRIDE}"
        )));
    }
    Ok((
        canvas_height / VAE_SPATIAL_RATIO,
        canvas_width / VAE_SPATIAL_RATIO,
    ))
}

// ── edit (reference images) ─────────────────────────────────────────────────────────────────────
//
// Sources (read 2026-11-10): diffusers main `modular_pipelines/minimax_h3_image/`
// `before_encoder.py` (`bound_short_edge`, `MiniMaxH3ImageEditResizeStep`), `encoders.py`
// (`MiniMaxH3ImageEditTextEncoderStep`, `MiniMaxH3ImageEditVaeEncoderStep`) and
// `modular_pipeline.py` (`reference_noise_aug`).

/// The smallest side a reference may have: one canvas multiple (`before_encoder.py:123`).
pub const MIN_REFERENCE_EDGE: u32 = SIZE_STRIDE;
/// The VAE copy of a reference keeps its pixels while its short edge is within
/// `[REFERENCE_MIN_SHORT_EDGE, REFERENCE_MAX_SHORT_EDGE]` (`reference_min_short_edge`).
pub const REFERENCE_MIN_SHORT_EDGE: u32 = 1024;
/// Upper bound of the VAE copy's short edge (`reference_max_short_edge`).
pub const REFERENCE_MAX_SHORT_EDGE: u32 = 2048;
/// The text-encoder copy of a reference always has this short edge (`vision_min_short_edge` =
/// `vision_max_short_edge` = 1024).
pub const VISION_SHORT_EDGE: u32 = 1024;
/// The `t` the reference rows of an edit are noised to and held at for every step
/// (`reference_noise_aug`, modular_pipeline.py): just short of clean in H3's `t = 1` clean
/// convention.
pub const REFERENCE_NOISE_AUG: f32 = 0.999;
/// Qwen3-VL merges `2 x 2` vision patches into one token (`merge_size ** 2`).
pub const VISION_MERGE_TOKENS: usize = 4;

/// The `(width, height)` one consumer of a reference resizes it to (`bound_short_edge`,
/// before_encoder.py): unchanged when the short edge is within `[min_short_edge, max_short_edge]`,
/// otherwise scaled at the same aspect so the short edge lands on the nearer bound, both sides
/// rounded (Python `round`, half to even) to `multiple` and floored at one `multiple`.
pub fn bound_short_edge(
    width: u32,
    height: u32,
    min_short_edge: u32,
    max_short_edge: u32,
    multiple: u32,
) -> (u32, u32) {
    let short = width.min(height);
    if (min_short_edge..=max_short_edge).contains(&short) {
        return (width, height);
    }
    let bound = if short < min_short_edge {
        min_short_edge
    } else {
        max_short_edge
    };
    let scale = f64::from(bound) / f64::from(short);
    let m = f64::from(multiple);
    let side = |v: u32| ((f64::from(v) * scale / m).round_ties_even() * m).max(m) as u32;
    (side(width), side(height))
}

/// The size the VAE copy of a `width x height` reference is resized to.
pub fn reference_vae_size(width: u32, height: u32) -> (u32, u32) {
    bound_short_edge(
        width,
        height,
        REFERENCE_MIN_SHORT_EDGE,
        REFERENCE_MAX_SHORT_EDGE,
        SIZE_STRIDE,
    )
}

/// The size the text-encoder (vision tower) copy of a `width x height` reference is resized to.
pub fn reference_vision_size(width: u32, height: u32) -> (u32, u32) {
    bound_short_edge(
        width,
        height,
        VISION_SHORT_EDGE,
        VISION_SHORT_EDGE,
        SIZE_STRIDE,
    )
}

/// The ordered reference images of an edit request: every [`crate::Conditioning::Reference`] and every
/// image of every [`crate::Conditioning::MultiReference`], in request order (`<Picture 1>` is the first).
/// Zero references is a `text2image` request. Refuses a per-reference strength other than the
/// no-op `1.0`, more than [`MAX_REFERENCES`] images, and a reference under
/// [`MIN_REFERENCE_EDGE`] on either side. Other conditioning kinds are left to the descriptor's
/// allowlist.
pub fn edit_reference_images(
    conditioning: &[crate::Conditioning],
) -> Result<Vec<&crate::Image>, H3ImageRefusal> {
    let mut images = Vec::new();
    for c in conditioning {
        match c {
            crate::Conditioning::Reference { image, strength } => {
                if let Some(s) = strength.filter(|&s| s != 1.0) {
                    return Err(H3ImageRefusal::ReferenceStrength {
                        index: images.len() + 1,
                        strength: s,
                    });
                }
                images.push(image);
            }
            crate::Conditioning::MultiReference { images: many } => images.extend(many.iter()),
            _ => {}
        }
        if images.len() > MAX_REFERENCES {
            return Err(H3ImageRefusal::TooManyReferences {
                count: conditioning
                    .iter()
                    .map(|c| match c {
                        crate::Conditioning::Reference { .. } => 1,
                        crate::Conditioning::MultiReference { images } => images.len(),
                        _ => 0,
                    })
                    .sum(),
            });
        }
    }
    validate_reference_sizes(images.iter().map(|i| (i.width, i.height)))?;
    Ok(images)
}

/// Refuse a reference under [`MIN_REFERENCE_EDGE`] on either side (before_encoder.py:123).
/// `index` in the refusal is 1-based, the `<Picture N>` the prompt names it by.
pub fn validate_reference_sizes(
    sizes: impl IntoIterator<Item = (u32, u32)>,
) -> Result<(), H3ImageRefusal> {
    for (i, (width, height)) in sizes.into_iter().enumerate() {
        if width < MIN_REFERENCE_EDGE || height < MIN_REFERENCE_EDGE {
            return Err(H3ImageRefusal::ReferenceTooSmall {
                index: i + 1,
                width,
                height,
            });
        }
    }
    Ok(())
}

/// The label that precedes reference `index`'s (0-based) vision block: `"Picture {index + 1}: "`
/// with no angle brackets — the `add_vision_id` form of the processor chat template
/// (`processor/chat_template.json`), unlike the `<Picture N>` the prompt refers to it by.
pub fn edit_reference_label(index: usize) -> String {
    format!("Picture {}: ", index + 1)
}

/// The three special ids a vision block is spelled with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionBlockIds {
    pub vision_start: i32,
    pub image_pad: i32,
    pub vision_end: i32,
}

/// The `edit` presentation (`MiniMaxH3ImageEditTextEncoderStep`, encoders.py): per reference, its
/// [`edit_reference_label`] tokenized verbatim and truncated to [`MAX_PROMPT_TOKENS`], then
/// `<|vision_start|>`, `vision_tokens[i]` `<|image_pad|>` and `<|vision_end|>`; then the prompt
/// tokenized verbatim and truncated to [`MAX_PROMPT_TOKENS`]. No chat template, no special tokens.
/// Returns `(ids, tags)` built in lockstep: label and prompt rows are [`TEXT_TAG`], every row of a
/// vision block is [`IMAGE_TAG`].
///
/// `encode` is the backend tokenizer (`tokenizer(text, add_special_tokens=False)`); it is called
/// once per label and once for the prompt.
pub fn edit_presentation(
    prompt: &str,
    vision_tokens: &[usize],
    ids_of: VisionBlockIds,
    mut encode: impl FnMut(&str) -> Result<Vec<i32>, Error>,
) -> Result<(Vec<i32>, Vec<i32>), Error> {
    if vision_tokens.is_empty() {
        return Err(Error::Unsupported(
            "MiniMax-H3-Image edit needs at least one reference image".into(),
        ));
    }
    if vision_tokens.len() > MAX_REFERENCES {
        return Err(H3ImageRefusal::TooManyReferences {
            count: vision_tokens.len(),
        }
        .into());
    }
    let mut ids = Vec::new();
    let mut tags = Vec::new();
    for (i, &n) in vision_tokens.iter().enumerate() {
        if n == 0 {
            return Err(Error::Msg(format!(
                "MiniMax-H3-Image edit: reference {} contributes no vision tokens",
                i + 1
            )));
        }
        let mut label = encode(&edit_reference_label(i))?;
        label.truncate(MAX_PROMPT_TOKENS);
        tags.extend(std::iter::repeat_n(TEXT_TAG, label.len()));
        ids.extend(label);
        ids.push(ids_of.vision_start);
        ids.extend(std::iter::repeat_n(ids_of.image_pad, n));
        ids.push(ids_of.vision_end);
        tags.extend(std::iter::repeat_n(IMAGE_TAG, n + 2));
    }
    let prompt_ids = encode(prompt)?;
    if prompt_ids.is_empty() {
        return Err(Error::Unsupported(
            "MiniMax-H3-Image needs a non-empty prompt".into(),
        ));
    }
    let prompt_ids = &prompt_ids[..prompt_ids.len().min(MAX_PROMPT_TOKENS)];
    tags.extend(std::iter::repeat_n(TEXT_TAG, prompt_ids.len()));
    ids.extend_from_slice(prompt_ids);
    Ok((ids, tags))
}

/// The `(latent_height, latent_width)` of one VAE-resized `width x height` reference after the
/// encode crops (`MiniMaxH3ImageEditVaeEncoderStep`): pixels cropped at the top left to whole
/// [`VAE_SPATIAL_RATIO`] cells, then the latent cropped to whole `patch` cells.
pub fn reference_latent_grid(width: u32, height: u32, patch: (u32, u32)) -> (u32, u32) {
    let lh = height / VAE_SPATIAL_RATIO;
    let lw = width / VAE_SPATIAL_RATIO;
    (lh / patch.0 * patch.0, lw / patch.1 * patch.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_ids_truncate_at_4096_and_refuse_empty() {
        assert_eq!(t2i_prompt_ids(vec![5, 6]).unwrap(), vec![5, 6]);
        let long: Vec<i32> = (0..5000).collect();
        let kept = t2i_prompt_ids(long).unwrap();
        assert_eq!(kept.len(), MAX_PROMPT_TOKENS);
        assert_eq!(kept[4095], 4095);
        assert!(matches!(
            t2i_prompt_ids(Vec::new()),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn latent_grid_is_the_canvas_over_16() {
        assert_eq!(latent_grid(2048, 2048).unwrap(), (128, 128));
        assert_eq!(latent_grid(1664, 2496).unwrap(), (156, 104));
        assert!(latent_grid(1000, 1000).is_err());
        assert_eq!((IMAGE_TAG, TEXT_TAG), (0, 1));
    }

    #[test]
    fn table_entries_are_stride_aligned_near_16mp() {
        for e in ASPECT_TABLE {
            validate_size(e.width, e.height).unwrap();
            assert_eq!((e.width % SIZE_STRIDE, e.height % SIZE_STRIDE), (0, 0));
            let a = u64::from(e.width) * u64::from(e.height);
            assert!((15_000_000..=17_000_000).contains(&a), "{}", e.label);
        }
    }

    #[test]
    fn size_only_needs_to_be_positive() {
        validate_size(1, 1).unwrap();
        validate_size(1000, 1001).unwrap();
        // No maximum area or edge upstream, and not H3 video's 1.03 MP / 2016 caps.
        validate_size(8192, 8192).unwrap();
        assert_eq!(
            validate_size(0, 32),
            Err(H3ImageRefusal::SizeNotPositive {
                width: 0,
                height: 32
            })
        );
    }

    #[test]
    fn canvas_scales_small_images_and_rounds() {
        // At or above the minimum area: only rounding to the stride.
        assert_eq!(
            generation_canvas(4096, 4096, MIN_GENERATION_AREA_DEFAULT),
            (4096, 4096)
        );
        assert_eq!(generation_canvas(1000, 1001, 1024), (992, 992));
        // 1024x1024 with the 4 MP default scales 2x.
        assert_eq!(
            generation_canvas(1024, 1024, MIN_GENERATION_AREA_DEFAULT),
            (2048, 2048)
        );
        // min_generation_area 1024 never enlarges; tiny sizes floor at one stride.
        assert_eq!(generation_canvas(8, 8, 1024), (32, 32));
        // 512x768 (0.39 MP) at 4 MP: scale sqrt(4194304/393216)=3.266 -> 1672.5, 2508.8
        assert_eq!(
            generation_canvas(512, 768, MIN_GENERATION_AREA_DEFAULT),
            (1664, 2496)
        );
    }

    #[test]
    fn min_generation_area_range() {
        validate_min_generation_area(None).unwrap();
        validate_min_generation_area(Some(1024)).unwrap();
        validate_min_generation_area(Some(16_777_216)).unwrap();
        assert!(validate_min_generation_area(Some(1023)).is_err());
        let e = validate_min_generation_area(Some(16_777_217)).unwrap_err();
        assert!(e.to_string().contains("16777216"));
    }

    #[test]
    fn refusals_are_distinct() {
        let ok = |s, g, n, r| validate_request(2048, 2048, None, s, g, n, r);
        ok(6, None, false, 16).unwrap();
        ok(6, Some(1.0), false, 0).unwrap();
        let errs = [
            ok(8, None, false, 0).unwrap_err(),
            ok(6, Some(4.0), false, 0).unwrap_err(),
            ok(6, None, true, 0).unwrap_err(),
            ok(6, None, false, 17).unwrap_err(),
        ];
        for i in 0..errs.len() {
            for j in i + 1..errs.len() {
                assert_ne!(
                    std::mem::discriminant(&errs[i]),
                    std::mem::discriminant(&errs[j])
                );
            }
        }
        assert!(matches!(
            Error::from(errs[0].clone()),
            Error::Unsupported(_)
        ));
    }

    #[test]
    fn the_sampler_grid_is_the_float32_round_trip_of_the_published_sigmas() {
        let g = sampler_sigmas();
        for (i, (&f, &d)) in SIGMAS.iter().zip(SIGMAS_F64.iter()).enumerate() {
            assert_eq!(
                f, d as f32,
                "SIGMAS[{i}] is the float32 rounding of SIGMAS_F64[{i}]"
            );
        }
        assert_eq!(g[0], 1.0);
        assert_eq!(g[3], 0.75);
        assert_eq!(g[6], 0.0);
        assert!(g.windows(2).all(|w| w[0] > w[1]));
        // The round trip moves a sigma by at most one float32 ulp of its timestep.
        for i in 0..6 {
            let d = (g[i] - SIGMAS[i]).abs();
            assert!(d <= f32::EPSILON, "{i}: {} vs {}", g[i], SIGMAS[i]);
        }
    }

    #[test]
    fn sigmas_are_the_published_list() {
        assert_eq!(SIGMAS.len(), FIXED_STEPS as usize + 1);
        assert_eq!(SIGMAS[0], 1.0);
        assert_eq!(SIGMAS[6], 0.0);
        assert!(SIGMAS.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn bound_short_edge_keeps_in_range_and_scales_to_the_nearer_bound() {
        // In range: untouched, even off the 32 grid.
        assert_eq!(reference_vae_size(1500, 1100), (1500, 1100));
        // Below 1024: short edge scaled up to 1024, long side rounded to 32.
        assert_eq!(reference_vae_size(512, 768), (1024, 1536));
        // Above 2048: down to 2048.
        assert_eq!(reference_vae_size(4096, 3000), (2784, 2048));
        // The vision copy is pinned at a 1024 short edge both ways.
        assert_eq!(reference_vision_size(1500, 1100), (1408, 1024));
        assert_eq!(reference_vision_size(1024, 1024), (1024, 1024));
        assert_eq!(reference_vision_size(64, 32), (2048, 1024));
        // Python round is half-to-even: 1.5 multiples -> 2, 2.5 -> 2.
        assert_eq!(bound_short_edge(48, 64, 64, 64, 32), (64, 96));
        assert_eq!(bound_short_edge(80, 32, 32, 32, 32), (80, 32));
        assert_eq!(bound_short_edge(80, 16, 32, 32, 32), (160, 32));
    }

    fn img(w: u32, h: u32) -> crate::Image {
        crate::Image {
            width: w,
            height: h,
            pixels: vec![0; (w * h * 3) as usize],
        }
    }

    #[test]
    fn edit_references_flatten_in_order_and_refuse_17_small_and_strength() {
        use crate::Conditioning;
        let conditioning = vec![
            Conditioning::Reference {
                image: img(32, 40),
                strength: None,
            },
            Conditioning::MultiReference {
                images: vec![img(64, 32), img(33, 33)],
            },
            Conditioning::Reference {
                image: img(50, 50),
                strength: Some(1.0),
            },
        ];
        let refs = edit_reference_images(&conditioning).unwrap();
        let sizes: Vec<(u32, u32)> = refs.iter().map(|i| (i.width, i.height)).collect();
        assert_eq!(sizes, vec![(32, 40), (64, 32), (33, 33), (50, 50)]);
        assert!(edit_reference_images(&[]).unwrap().is_empty());

        let sixteen = vec![Conditioning::MultiReference {
            images: (0..16).map(|_| img(32, 32)).collect(),
        }];
        assert_eq!(edit_reference_images(&sixteen).unwrap().len(), 16);
        let seventeen = vec![
            sixteen[0].clone(),
            Conditioning::Reference {
                image: img(32, 32),
                strength: None,
            },
        ];
        assert_eq!(
            edit_reference_images(&seventeen).unwrap_err(),
            H3ImageRefusal::TooManyReferences { count: 17 }
        );
        let small = vec![Conditioning::MultiReference {
            images: vec![img(32, 32), img(31, 400)],
        }];
        assert_eq!(
            edit_reference_images(&small).unwrap_err(),
            H3ImageRefusal::ReferenceTooSmall {
                index: 2,
                width: 31,
                height: 400
            }
        );
        let strength = vec![Conditioning::Reference {
            image: img(32, 32),
            strength: Some(0.5),
        }];
        assert!(matches!(
            edit_reference_images(&strength).unwrap_err(),
            H3ImageRefusal::ReferenceStrength { index: 1, .. }
        ));
    }

    /// The presentation is `Picture i: ` + vision block per reference, then the prompt; vision rows
    /// are image-tagged, labels and prompt text-tagged.
    #[test]
    fn edit_presentation_matches_the_reference_layout() {
        let specials = VisionBlockIds {
            vision_start: 900,
            image_pad: 901,
            vision_end: 902,
        };
        let mut seen = Vec::new();
        let (ids, tags) = edit_presentation("make it red", &[2, 1], specials, |s| {
            seen.push(s.to_owned());
            Ok(s.bytes().take(3).map(i32::from).collect())
        })
        .unwrap();
        assert_eq!(seen, vec!["Picture 1: ", "Picture 2: ", "make it red"]);
        let p = |s: &str| s.bytes().take(3).map(i32::from).collect::<Vec<i32>>();
        let mut want = p("Picture 1: ");
        want.extend([900, 901, 901, 902]);
        want.extend(p("Picture 2: "));
        want.extend([900, 901, 902]);
        want.extend(p("make it red"));
        assert_eq!(ids, want);
        assert_eq!(tags, vec![1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]);
        assert!(edit_presentation("x", &[], specials, |_| Ok(vec![1])).is_err());
        assert!(edit_presentation("x", &[1; 17], specials, |_| Ok(vec![1])).is_err());
        assert!(edit_presentation("x", &[0], specials, |_| Ok(vec![1])).is_err());
        assert!(edit_presentation("", &[1], specials, |_| Ok(vec![])).is_err());
        // The prompt truncates at 4096 tokens.
        let (ids, _) = edit_presentation("long", &[1], specials, |s| {
            Ok(if s == "long" { vec![7; 5000] } else { vec![1] })
        })
        .unwrap();
        assert_eq!(ids.len(), 1 + 3 + MAX_PROMPT_TOKENS);
    }

    #[test]
    fn reference_latent_grid_crops_to_cells_then_patches() {
        assert_eq!(reference_latent_grid(1024, 1536, (2, 2)), (96, 64));
        // 100 px -> 6 latent cells -> 6; 90 px -> 5 -> 4.
        assert_eq!(reference_latent_grid(90, 100, (2, 2)), (6, 4));
        assert_eq!(REFERENCE_NOISE_AUG, 0.999);
    }
}
