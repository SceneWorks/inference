//! Fixed held-out evaluation protocol for the six terminal stack renders.
//! Pure CPU helpers use the real backend-neutral request and image carriers.

use super::{edit_protocol, gen_core, RENDER_EDGE, RENDER_STEPS, SEED, TRAIN_EDGE};
use gen_core::{Conditioning, GenerationRequest, Image, RgbaImage};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub fn request(source: Image, key: Image) -> GenerationRequest {
    assert_eq!((source.width, source.height), (RENDER_EDGE, RENDER_EDGE));
    assert_eq!((key.width, key.height), (TRAIN_EDGE, TRAIN_EDGE));
    GenerationRequest {
        prompt: edit_protocol::EDIT_INSTRUCTION.to_owned(),
        width: RENDER_EDGE,
        height: RENDER_EDGE,
        steps: Some(RENDER_STEPS),
        seed: Some(SEED),
        strength: Some(1.0),
        conditioning: vec![Conditioning::MultiReference {
            images: vec![source, key],
        }],
        ..Default::default()
    }
}

/// Hash the actual ordered RGBA inputs and their common vision/VAE fitted bytes, using
/// the same core Lanczos operation and u8 conversion as production `prepare_reference`.
/// `sizes` comes from production `reference_fit`, rather than a guessed fit.
pub fn reference_receipt(references: &[RgbaImage], sizes: &[(u32, u32)]) -> Value {
    assert_eq!(references.len(), 2);
    assert_eq!(sizes.len(), references.len());
    let ordered: Vec<_> = references.iter().zip(sizes).enumerate().map(|(index, (image, &(width, height)))| {
        let fitted = if (image.width, image.height) == (width, height) {
            image.pixels.clone()
        } else {
            gen_core::imageops::resize_lanczos_rgba_u8(&image.pixels,
                image.height as usize, image.width as usize, height as usize, width as usize)
                .unwrap().into_iter().map(|v| v.clamp(0.0, 255.0) as u8).collect()
        };
        json!({"index":index,"role":if index == 0 {"heldout-source-99"} else {"RGB-level-palette"},
            "nativeWidth":image.width,"nativeHeight":image.height,
            "nativeRgbaSha256":format!("{:x}",Sha256::digest(&image.pixels)),
            "fittedWidth":width,"fittedHeight":height,
            "fittedRgbaSha256":format!("{:x}",Sha256::digest(&fitted)),
            "fittedRgbaBytes":fitted.len(),"latentTokens":u64::from(width/16)*u64::from(height/16)})
    }).collect();
    json!({"route":"two_reference_edit","referenceCount":references.len(),
        "inputHashSchema":"SHA256(raw row-major RGBA8 bytes, alpha255 for RGB inputs)",
        "orderedReferences":ordered})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(edge: u32, value: u8) -> Image {
        Image {
            width: edge,
            height: edge,
            pixels: vec![value; (edge * edge * 3) as usize],
        }
    }

    #[test]
    fn stack_is_ordered_two_reference_edit_with_fixed_evaluation_knobs() {
        let source = image(RENDER_EDGE, 31);
        let key = image(TRAIN_EDGE, 170);
        let req = request(source.clone(), key.clone());
        assert_eq!(req.prompt, edit_protocol::EDIT_INSTRUCTION);
        assert_eq!((req.width, req.height), (768, 768));
        assert_eq!(
            (req.steps, req.seed, req.strength),
            (Some(8), Some(24163), Some(1.0))
        );
        assert_eq!(req.memory_reference_count(), 2);
        assert!(req.negative_prompt.is_none());
        assert!(req.guidance.is_none() && req.true_cfg.is_none());
        let [Conditioning::MultiReference { images }] = req.conditioning.as_slice() else {
            panic!("ordered multi-reference conditioning required")
        };
        assert_eq!(images, &[source, key]);
    }

    #[test]
    #[should_panic]
    fn stack_refuses_transfer_key_geometry() {
        request(image(RENDER_EDGE, 31), image(RENDER_EDGE, 170));
    }

    #[test]
    fn receipt_binds_order_geometry_native_and_fitted_bytes() {
        let source = RgbaImage {
            width: 2,
            height: 2,
            pixels: [31, 31, 31, 255].repeat(4),
        };
        let key = RgbaImage {
            width: 1,
            height: 1,
            pixels: vec![170, 170, 170, 255],
        };
        let refs = [source, key];
        let receipt = reference_receipt(&refs, &[(4, 4), (4, 4)]);
        assert_eq!(receipt["route"], "two_reference_edit");
        assert_eq!(receipt["referenceCount"], 2);
        let ordered = receipt["orderedReferences"].as_array().unwrap();
        assert_eq!(ordered[0]["nativeWidth"], 2);
        assert_eq!(ordered[1]["nativeWidth"], 1);
        assert_eq!(ordered[0]["fittedWidth"], 4);
        assert_eq!(ordered[0]["fittedRgbaBytes"], 64);
        assert_eq!(
            ordered[0]["nativeRgbaSha256"],
            format!("{:x}", Sha256::digest([31, 31, 31, 255].repeat(4)))
        );
        assert_eq!(
            ordered[1]["fittedRgbaSha256"],
            format!("{:x}", Sha256::digest([170, 170, 170, 255].repeat(16)))
        );
        let reversed = reference_receipt(&[refs[1].clone(), refs[0].clone()], &[(4, 4), (4, 4)]);
        assert_ne!(receipt, reversed);
        let mut changed = refs.clone();
        changed[0].pixels[0] = 99;
        let changed = reference_receipt(&changed, &[(4, 4), (4, 4)]);
        assert_ne!(
            ordered[0]["nativeRgbaSha256"],
            changed["orderedReferences"][0]["nativeRgbaSha256"]
        );
        assert_ne!(
            ordered[0]["fittedRgbaSha256"],
            changed["orderedReferences"][0]["fittedRgbaSha256"]
        );
    }
}
