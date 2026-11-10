//! MiniMax-H3-Image request contract: size rules, published aspect table, the fixed 6-step sigma
//! list, and the refusals a request must pass before any backend runs. Tensor-free: MLX, Candle and
//! the shared-generation runner all read these as data.
//!
//! Source: the MiniMaxAI/MiniMax-H3-Image model card and `pdd_config.json` (read 2026-11-10). The
//! diffusers pipeline source was not reachable, so the upstream size ceiling is taken from the card:
//! it states a working range of "about 4 MP to 20 MP" and no hard maximum. We refuse above 20 MP
//! (`20 * 2^20` px, the same megapixel unit as the card's 4 MP default `2048x2048` and 16 MP
//! `4096x4096`). This is stricter than upstream, which is not documented to refuse.
//!
//! This family is NOT bound by H3 video's `CANVAS_MAX_PIXELS` (1.03 MP) or `MAX_CANVAS_EDGE` (2016).

use crate::Error;
use thiserror::Error as ThisError;

/// Pixels per generated token edge: VAE 16x spatial times DiT patch 2x2.
pub const SIZE_STRIDE: u32 = 32;
/// Smallest accepted `min_generation_area` (upstream parameter range, pixels).
pub const MIN_GENERATION_AREA_MIN: u64 = 1024;
/// Largest accepted `min_generation_area`.
pub const MIN_GENERATION_AREA_MAX: u64 = 16_777_216;
/// Upstream default `min_generation_area` (2048x2048).
pub const MIN_GENERATION_AREA_DEFAULT: u64 = 4_194_304;
/// Largest output area we accept (20 MP, `20 * 2^20`).
pub const MAX_OUTPUT_AREA: u64 = 20 * 1_048_576;
/// Maximum reference images for edit.
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
    #[error("size {width}x{height} must be non-zero multiples of {SIZE_STRIDE}")]
    SizeStride { width: u32, height: u32 },
    #[error("size {width}x{height} is {area} px, above the {MAX_OUTPUT_AREA} px (20 MP) limit")]
    AreaAboveMax { width: u32, height: u32, area: u64 },
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
}

impl From<H3ImageRefusal> for Error {
    fn from(r: H3ImageRefusal) -> Self {
        Error::Unsupported(r.to_string())
    }
}

/// Check an output size. Sizes below 4 MP are valid: upstream renders them on a larger canvas.
pub fn validate_size(width: u32, height: u32) -> Result<(), H3ImageRefusal> {
    if width == 0
        || height == 0
        || !width.is_multiple_of(SIZE_STRIDE)
        || !height.is_multiple_of(SIZE_STRIDE)
    {
        return Err(H3ImageRefusal::SizeStride { width, height });
    }
    let area = u64::from(width) * u64::from(height);
    if area > MAX_OUTPUT_AREA {
        return Err(H3ImageRefusal::AreaAboveMax {
            width,
            height,
            area,
        });
    }
    Ok(())
}

/// Check an optional `min_generation_area` against the published range.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_entries_are_valid_sizes_near_16mp() {
        for e in ASPECT_TABLE {
            validate_size(e.width, e.height).unwrap();
            let a = u64::from(e.width) * u64::from(e.height);
            assert!((15_000_000..=17_000_000).contains(&a), "{}", e.label);
        }
    }

    #[test]
    fn size_range() {
        validate_size(32, 32).unwrap();
        validate_size(6272, 2688).unwrap();
        // Above H3 video's 1.03 MP cap and 2016 edge: allowed here.
        validate_size(4096, 4096).unwrap();
        assert_eq!(
            validate_size(0, 32),
            Err(H3ImageRefusal::SizeStride {
                width: 0,
                height: 32
            })
        );
        assert!(matches!(
            validate_size(1000, 1024),
            Err(H3ImageRefusal::SizeStride { .. })
        ));
        let e = validate_size(6400, 3360).unwrap_err();
        assert!(matches!(e, H3ImageRefusal::AreaAboveMax { .. }));
        assert!(e.to_string().contains("20 MP"));
    }

    #[test]
    fn min_generation_area_range() {
        validate_min_generation_area(None).unwrap();
        validate_min_generation_area(Some(1024)).unwrap();
        validate_min_generation_area(Some(16_777_216)).unwrap();
        assert!(validate_min_generation_area(Some(1023)).is_err());
        assert!(validate_min_generation_area(Some(16_777_217)).is_err());
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
    fn sigmas_are_the_published_list() {
        assert_eq!(SIGMAS.len(), FIXED_STEPS as usize + 1);
        assert_eq!(SIGMAS[0], 1.0);
        assert_eq!(SIGMAS[6], 0.0);
        assert!(SIGMAS.windows(2).all(|w| w[0] > w[1]));
    }
}
