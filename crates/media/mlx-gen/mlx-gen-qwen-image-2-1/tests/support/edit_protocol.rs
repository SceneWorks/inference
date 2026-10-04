//! Fixed semantic diagnostic fixtures, executable as a standalone CPU Rust test.

pub const TRAIN_EDIT_INSTRUCTION: &str = "zxq edit: invert each RGB colour channel of image 1 independently, then quantize each channel to the four numeric levels 0, 85, 170, 255; preserve the shapes and keep the result in colour";
pub const EDIT_INSTRUCTION: &str = "zxq edit: invert each RGB colour channel of image 1 independently, then quantize each channel to the four numeric levels 0, 85, 170, 255; preserve the shapes and keep the result in colour; image 2 is only the RGB level palette; do not copy its layout";
pub const T2I_EVAL_PROMPT: &str = "zxq style, zxq edit, zxq invert: a lighthouse made of concentric teal and orange rings on yellow";
pub const PALETTE_ROLE: &str = "; image 2 is only the RGB level palette; do not copy its layout";

/// All 64 Cartesian RGB combinations, arranged in fixed row-major 8×8 swatches.
pub fn palette_pixel(x: u32, y: u32, edge: u32) -> [u8; 3] {
    let levels = [0, 85, 170, 255];
    let index = ((y * 8 / edge) * 8 + x * 8 / edge) as usize;
    [
        levels[index / 16],
        levels[(index / 4) % 4],
        levels[index % 4],
    ]
}

/// The original independent-channel transform, unchanged by the new palette key.
pub fn transformed_channel(value: u8) -> u8 {
    ((255 - value) / 64).min(3) * 85
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn key_covers_every_rgb_combination_and_no_other_levels() {
        for edge in [512, 768] {
            let colors: BTreeSet<_> = (0..edge)
                .flat_map(|y| (0..edge).map(move |x| palette_pixel(x, y, edge)))
                .collect();
            let expected: BTreeSet<_> = [0, 85, 170, 255]
                .into_iter()
                .flat_map(|r| {
                    [0, 85, 170, 255]
                        .into_iter()
                        .flat_map(move |g| [0, 85, 170, 255].into_iter().map(move |b| [r, g, b]))
                })
                .collect();
            assert_eq!(colors, expected);
            assert_eq!(palette_pixel(0, 0, edge), [0, 0, 0]);
            assert_eq!(palette_pixel(edge - 1, edge - 1, edge), [255, 255, 255]);
        }
    }

    #[test]
    fn target_math_keeps_original_bins_and_independent_channels() {
        for (range, expected) in [(0..64, 255), (64..128, 170), (128..192, 85), (192..256, 0)] {
            for value in range {
                assert_eq!(transformed_channel(value as u8), expected);
            }
        }
        assert_eq!([0, 85, 170].map(transformed_channel), [255, 170, 85]);
    }

    #[test]
    fn evaluation_only_adds_an_honest_palette_role() {
        assert_eq!(
            EDIT_INSTRUCTION,
            format!("{TRAIN_EDIT_INSTRUCTION}{PALETTE_ROLE}")
        );
        assert!(!TRAIN_EDIT_INSTRUCTION.contains("image 2"));
        assert!(TRAIN_EDIT_INSTRUCTION.contains("independently"));
        assert!(TRAIN_EDIT_INSTRUCTION.contains("keep the result in colour"));
        assert!(EDIT_INSTRUCTION.ends_with("do not copy its layout"));
        for prefix in ["zxq style", "zxq edit", "zxq invert"] {
            assert!(T2I_EVAL_PROMPT.contains(prefix));
        }
        assert!(T2I_EVAL_PROMPT
            .contains("lighthouse made of concentric teal and orange rings on yellow"));
    }
}
