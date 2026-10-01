//! SC-20686 product admission estimate for one MLX FLUX.2 Klein edit request
//! (estimate-plus-reserve-v1).
//!
//! Priced from the product's own memory model, resolved exactly as the run resolves it (the same
//! snapshot and request), without loading a weight or touching MLX:
//!
//! * text encoder: the Qwen3 text-encoder files at their stored width (conditioning phase);
//! * DiT + VAE: the transformer and VAE files, resident through denoise and decode;
//! * activation: the provider's registered warm 1024² activation transient
//!   ([`crate::KLEIN_ACTIVATION_MEMORY_REGISTRATION`], 4096 image tokens), scaled by the square of the
//!   request's token ratio when it attends over more tokens (the target plus every reference, each
//!   resized to the target -- `preprocess_ref_image`), and doubled for true CFG (`guidance > 1`);
//! * KV edit route: the cached reference K/V of every transformer block at bf16.
//!
//! The staged `Sequential` product residency releases the text encoder before the DiT loads, so the
//! estimate is the MAX of the conditioning and denoise phase totals.

use std::path::Path;

use mlx_gen::gen_core::weightsmeta::{materialized_path_bytes, safetensors_path_bytes};
use mlx_gen::{Conditioning, Error, GenerationRequest, LoadSpec, Result, WeightsSource};

use crate::{FLUX2_KLEIN_9B_EDIT_ID, FLUX2_KLEIN_9B_KV_EDIT_ID};

/// Image tokens of the 1024² activation anchor: `(1024 / 16)²`.
const ANCHOR_TOKENS: u64 = 4096;
/// Output pixels per image token on each axis (VAE ×8, then 2×2 patchify).
const PIXELS_PER_TOKEN_EDGE: u32 = 16;

/// One request's priced components and phase totals (bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flux2AdmissionEstimate {
    pub text_encoder_bytes: u64,
    pub transformer_bytes: u64,
    pub vae_bytes: u64,
    pub activation_bytes: u64,
    pub reference_kv_bytes: u64,
}

impl Flux2AdmissionEstimate {
    /// `(phase, bytes)` for the two staged phases.
    pub fn phases(&self) -> [(&'static str, u64); 2] {
        [
            ("conditioning", self.text_encoder_bytes),
            (
                "denoise",
                self.transformer_bytes
                    .saturating_add(self.vae_bytes)
                    .saturating_add(self.activation_bytes)
                    .saturating_add(self.reference_kv_bytes),
            ),
        ]
    }

    /// The admission estimate: the larger phase.
    pub fn peak_bytes(&self) -> u64 {
        self.phases()
            .iter()
            .map(|(_, bytes)| *bytes)
            .max()
            .unwrap_or(0)
    }
}

fn component_bytes(root: &Path, component: &str, float_width: u64) -> Result<u64> {
    let path = root.join(component);
    Ok(safetensors_path_bytes(&path).max(materialized_path_bytes(&path, float_width)?))
}

/// The product admission estimate for the Klein edit `route` loaded with `spec` and asked `req`.
pub fn product_admission_estimate(
    route: &str,
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> Result<Flux2AdmissionEstimate> {
    if route != FLUX2_KLEIN_9B_EDIT_ID && route != FLUX2_KLEIN_9B_KV_EDIT_ID {
        return Err(Error::Msg(format!(
            "unsupported SC-20686 FLUX.2 route for the admission estimate: {route}"
        )));
    }
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(Error::Msg(format!(
            "{route}: expected a snapshot directory for the admission estimate"
        )));
    };
    let overflow = || Error::Msg(format!("{route}: admission estimate overflows"));
    let references = req
        .conditioning
        .iter()
        .map(|conditioning| match conditioning {
            Conditioning::Reference { .. } => 1_u64,
            Conditioning::MultiReference { images } => images.len() as u64,
            _ => 0,
        })
        .sum::<u64>();
    let target_tokens = u64::from(req.width / PIXELS_PER_TOKEN_EDGE)
        .checked_mul(u64::from(req.height / PIXELS_PER_TOKEN_EDGE))
        .ok_or_else(overflow)?;
    let reference_tokens = target_tokens.checked_mul(references).ok_or_else(overflow)?;
    let tokens = target_tokens
        .checked_add(reference_tokens)
        .ok_or_else(overflow)?;
    let anchor = crate::KLEIN_ACTIVATION_MEMORY_REGISTRATION
        .anchor
        .bytes_1024;
    // Attention grows with the square of the token count; below the anchor the anchor itself.
    let scaled = if tokens <= ANCHOR_TOKENS {
        anchor
    } else {
        u128::from(anchor)
            .checked_mul(u128::from(tokens) * u128::from(tokens))
            .map(|bytes| bytes.div_ceil(u128::from(ANCHOR_TOKENS * ANCHOR_TOKENS)))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(overflow)?
    };
    let cfg_forwards = if req.guidance.is_some_and(|guidance| guidance > 1.0) {
        2
    } else {
        1
    };
    let activation_bytes = scaled.checked_mul(cfg_forwards).ok_or_else(overflow)?;
    let reference_kv_bytes = if route == FLUX2_KLEIN_9B_KV_EDIT_ID {
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("transformer/config.json"))
                .map_err(|error| Error::Msg(format!("{route}: transformer config: {error}")))?,
        )
        .map_err(|error| Error::Msg(format!("{route}: transformer config: {error}")))?;
        let field = |key: &str| {
            config
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| Error::Msg(format!("{route}: transformer config lacks {key}")))
        };
        // K and V, every double- and single-stream block, every reference token, bf16.
        [
            2,
            field("num_layers")?
                .checked_add(field("num_single_layers")?)
                .ok_or_else(overflow)?,
            field("num_attention_heads")?,
            field("attention_head_dim")?,
            reference_tokens,
            2,
        ]
        .into_iter()
        .try_fold(1_u64, |product, factor| product.checked_mul(factor))
        .ok_or_else(overflow)?
    } else {
        0
    };
    Ok(Flux2AdmissionEstimate {
        text_encoder_bytes: component_bytes(root, "text_encoder", 2)?,
        transformer_bytes: component_bytes(root, "transformer", 2)?,
        // The VAE runs f32: every float tensor at four bytes.
        vae_bytes: component_bytes(root, "vae", 4)?,
        activation_bytes,
        reference_kv_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{Image, OffloadPolicy};

    fn safetensors(path: &Path, bytes: u64) {
        let elements = bytes / 2;
        let mut header = serde_json::json!({
            "w": {"dtype": "BF16", "shape": [elements], "data_offsets": [0, elements * 2]}
        })
        .to_string();
        while !header.len().is_multiple_of(8) {
            header.push(' ');
        }
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        std::io::Write::write_all(&mut file, &(header.len() as u64).to_le_bytes()).unwrap();
        std::io::Write::write_all(&mut file, header.as_bytes()).unwrap();
        file.set_len(8 + header.len() as u64 + elements * 2)
            .unwrap();
    }

    fn snapshot(root: &Path) {
        safetensors(&root.join("text_encoder/model.safetensors"), 16_000_000_000);
        safetensors(
            &root.join("transformer/diffusion_pytorch_model.safetensors"),
            5_000_000_000,
        );
        safetensors(
            &root.join("vae/diffusion_pytorch_model.safetensors"),
            84_000_000,
        );
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"num_layers": 8, "num_single_layers": 24, "num_attention_heads": 32,
                "attention_head_dim": 128, "quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();
    }

    fn request(width: u32, height: u32, references: usize, guidance: f32) -> GenerationRequest {
        let image = Image {
            width: 8,
            height: 8,
            pixels: vec![0; 8 * 8 * 3],
        };
        GenerationRequest {
            width,
            height,
            guidance: Some(guidance),
            conditioning: if references == 1 {
                vec![Conditioning::Reference {
                    image,
                    strength: None,
                }]
            } else {
                vec![Conditioning::MultiReference {
                    images: vec![image; references],
                }]
            },
            ..Default::default()
        }
    }

    #[test]
    fn klein_edit_estimate_prices_tokens_cfg_and_cached_references() {
        let dir = tempfile::tempdir().unwrap();
        snapshot(dir.path());
        let spec = |route| {
            crate::product_load::product_load_spec(route, dir.path(), OffloadPolicy::Sequential)
                .unwrap()
        };
        let anchor = crate::KLEIN_ACTIVATION_MEMORY_REGISTRATION
            .anchor
            .bytes_1024;
        // 512² + one 512² reference: 2048 tokens, under the 4096-token anchor -> the anchor.
        let small = product_admission_estimate(
            FLUX2_KLEIN_9B_EDIT_ID,
            &spec(FLUX2_KLEIN_9B_EDIT_ID),
            &request(512, 512, 1, 1.0),
        )
        .unwrap();
        assert_eq!(small.activation_bytes, anchor);
        assert_eq!(small.reference_kv_bytes, 0);
        assert_eq!(small.vae_bytes, 168_000_000, "the VAE is priced at f32");
        // 768x512 + two references: 4608 tokens -> (4608/4096)², and true CFG doubles it.
        let large = product_admission_estimate(
            FLUX2_KLEIN_9B_EDIT_ID,
            &spec(FLUX2_KLEIN_9B_EDIT_ID),
            &request(768, 512, 2, 2.0),
        )
        .unwrap();
        assert_eq!(
            large.activation_bytes,
            2 * (u128::from(anchor) * 4608 * 4608).div_ceil(4096 * 4096) as u64
        );
        assert_eq!(
            large.peak_bytes(),
            large.transformer_bytes + large.vae_bytes + large.activation_bytes
        );
        // The KV route also holds K/V of every block for every reference token, at bf16.
        let kv = product_admission_estimate(
            FLUX2_KLEIN_9B_KV_EDIT_ID,
            &spec(FLUX2_KLEIN_9B_KV_EDIT_ID),
            &request(768, 512, 2, 2.0),
        )
        .unwrap();
        assert_eq!(kv.reference_kv_bytes, 2 * 32 * 32 * 128 * 3072 * 2);
        assert_eq!(kv.peak_bytes(), large.peak_bytes() + kv.reference_kv_bytes);
        // Text encoding is its own (released) phase: a max, not a sum.
        assert!(small.peak_bytes() < small.phases().iter().map(|(_, bytes)| bytes).sum::<u64>());
        assert!(product_admission_estimate(
            "flux2_klein_9b",
            &spec(FLUX2_KLEIN_9B_EDIT_ID),
            &request(512, 512, 1, 1.0)
        )
        .is_err());
    }
}
