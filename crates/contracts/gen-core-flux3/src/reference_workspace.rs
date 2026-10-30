//! Pre-fit reference workspace pricing for the FLUX3 routes (sc-26241).
//!
//! A worker resolves its owned references into [`gen_core::MemoryReferenceResourceFacts`] BEFORE fit
//! selection and admission. This module turns those facts into the additional bytes the
//! reference work needs, from the package's own architecture, so the fit decision sees references
//! instead of only refusing a mismatch after admission. The formula and its inputs are shared by
//! the Candle and MLX providers; a backend contributes only its compute width.
//!
//! This is a DERIVED conservative bound, not a measurement (measurement belongs to the epic's
//! terminal campaign). It has three terms:
//!
//! 1. **Autoencoder encode peak.** References are encoded one at a time, so only the LARGEST
//!    canvas counts: `width · height · encoder_base_channels · ENCODER_LIVE_MAPS · 4` (the encoder
//!    runs in f32; a resnet block at full resolution holds its input, normalized copy, conv output
//!    and residual sum at once).
//! 2. **Retained reference latents.** Every reference's packed tokens stay resident through denoise:
//!    `conditioning_tokens · latent_channels · 4`.
//! 3. **DiT sequence growth.** Reference tokens join the stream of every step. Per extra token and
//!    CFG row, one block holds q/k/v/attention output (`4 · hidden`) and the MLP input/activation
//!    (`3 · mlp_hidden`) at the compute width. The distilled flow map has one row; the base CFG
//!    route packs references into BOTH rows.
//!
//! No references prices zero, so every decision without references is unchanged.
use std::path::Path;

use gen_core::weightsmeta::safetensors_path_tensor_headers;
use gen_core::{Error, LoadSpec, MemoryReferenceResourceFacts, Result};

/// Full-resolution feature maps live at once inside one encoder resnet block.
pub const ENCODER_LIVE_MAPS: u64 = 4;
/// The autoencoder (and so its encoder) runs in f32 on both backends.
const AE_WIDTH: u64 = 4;
/// Reference latents are held as f32 tokens.
const LATENT_WIDTH: u64 = 4;

fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 reference workspace: {detail}"))
}

/// Everything the formula reads, from the package itself plus the backend's compute width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReferenceWorkspaceInputs {
    pub hidden: u64,
    pub mlp_hidden: u64,
    pub latent_channels: u64,
    pub encoder_base_channels: u64,
    /// Bytes per element of the DiT compute dtype on this backend.
    pub compute_width: u64,
    /// Rows the reference context occupies per step: 1 (flow map) or 2 (base CFG).
    pub cfg_rows: u64,
}

impl ReferenceWorkspaceInputs {
    /// Read the inputs from the package `spec` names (config + autoencoder header; no tensor
    /// payload is read). `model_id` selects the route's resource resolution.
    pub fn from_spec(spec: &LoadSpec, model_id: &str, compute_width: u64) -> Result<Self> {
        let dir = match &spec.weights {
            gen_core::WeightsSource::Dir(dir) => dir.clone(),
            gen_core::WeightsSource::File(file) => {
                return Err(invalid(format!(
                    "the weights resource must be the package directory, not {}",
                    file.display()
                )))
            }
        };
        let config = super::resolve_checkpoint(&dir)?;
        config.validate_route(model_id)?;
        super::artifacts::NativeDitStorage::from_package(&dir)?
            .validate_headers(&config.dit, &super::checkpoint_tensor_headers(&dir)?)?;
        let resources = super::GenerationResources::from_spec(spec, &config, model_id)?;
        Self::from_parts(&config, &resources.autoencoder_file, compute_width)
    }

    pub fn from_parts(
        config: &super::Flux3Config,
        autoencoder: &Path,
        compute_width: u64,
    ) -> Result<Self> {
        if !matches!(compute_width, 2 | 4) {
            return Err(invalid("compute width must be 2 or 4 bytes"));
        }
        let headers = safetensors_path_tensor_headers(autoencoder)?;
        let conv_in = headers
            .iter()
            .find(|h| h.name == "encoder.conv_in.weight")
            .ok_or_else(|| invalid("autoencoder has no encoder.conv_in.weight"))?;
        let encoder_base_channels = *conv_in
            .shape
            .first()
            .ok_or_else(|| invalid("encoder.conv_in.weight has no output axis"))?
            as u64;
        Ok(Self {
            hidden: config.dit.hidden as u64,
            mlp_hidden: config.dit.mlp_hidden() as u64,
            latent_channels: config.dit.image_in_channels as u64,
            encoder_base_channels,
            compute_width,
            cfg_rows: if config.dit.flow_map { 1 } else { 2 },
        })
    }
}

/// Additional bytes the reference work of one request needs. `None` (no references) is 0.
pub fn reference_workspace_bytes(
    inputs: &ReferenceWorkspaceInputs,
    facts: Option<&MemoryReferenceResourceFacts>,
) -> Result<u64> {
    let Some(facts) = facts else {
        return Ok(0);
    };
    if facts.prepared_canvases.is_empty()
        || facts.prepared_canvases.len() > super::references::MAX_REFERENCES
    {
        return Err(invalid(
            "reference facts need 1..=MAX_REFERENCES prepared canvases",
        ));
    }
    let token_sum: u64 = facts
        .prepared_canvases
        .iter()
        .map(|&(w, h)| u64::from(w / 16) * u64::from(h / 16))
        .sum();
    if token_sum != facts.conditioning_tokens {
        return Err(invalid(
            "conditioning_tokens disagree with the prepared canvases; derive both from one ReferencePlan",
        ));
    }
    let overflow = || invalid("reference workspace byte overflow");
    let largest = facts
        .prepared_canvases
        .iter()
        .map(|&(w, h)| u64::from(w) * u64::from(h))
        .max()
        .unwrap_or(0);
    let encode = largest
        .checked_mul(inputs.encoder_base_channels)
        .and_then(|v| v.checked_mul(ENCODER_LIVE_MAPS))
        .and_then(|v| v.checked_mul(AE_WIDTH))
        .ok_or_else(overflow)?;
    let latents = facts
        .conditioning_tokens
        .checked_mul(inputs.latent_channels)
        .and_then(|v| v.checked_mul(LATENT_WIDTH))
        .ok_or_else(overflow)?;
    let per_token = inputs
        .hidden
        .checked_mul(4)
        .and_then(|v| v.checked_add(inputs.mlp_hidden.checked_mul(3)?))
        .and_then(|v| v.checked_mul(inputs.compute_width))
        .ok_or_else(overflow)?;
    let dit = facts
        .conditioning_tokens
        .checked_mul(inputs.cfg_rows)
        .and_then(|v| v.checked_mul(per_token))
        .ok_or_else(overflow)?;
    encode
        .checked_add(latents)
        .and_then(|v| v.checked_add(dit))
        .ok_or_else(overflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(cfg_rows: u64) -> ReferenceWorkspaceInputs {
        ReferenceWorkspaceInputs {
            hidden: 3072,
            mlp_hidden: 9216,
            latent_channels: 128,
            encoder_base_channels: 128,
            compute_width: 2,
            cfg_rows,
        }
    }

    fn facts(canvases: &[(u32, u32)]) -> MemoryReferenceResourceFacts {
        MemoryReferenceResourceFacts {
            conditioning_tokens: canvases
                .iter()
                .map(|&(w, h)| u64::from(w / 16) * u64::from(h / 16))
                .sum(),
            prepared_canvases: canvases.to_vec(),
        }
    }

    #[test]
    fn no_references_price_zero() {
        assert_eq!(reference_workspace_bytes(&inputs(1), None).unwrap(), 0);
    }

    /// *Mutations that red this:* pricing only the first canvas, dropping the CFG row factor, or
    /// summing encode peaks across canvases (references encode one at a time).
    #[test]
    fn workspace_is_monotone_in_count_and_size_and_doubles_dit_for_cfg() {
        let one = reference_workspace_bytes(&inputs(1), Some(&facts(&[(512, 512)]))).unwrap();
        let two =
            reference_workspace_bytes(&inputs(1), Some(&facts(&[(512, 512), (512, 512)]))).unwrap();
        let big = reference_workspace_bytes(&inputs(1), Some(&facts(&[(1024, 1024)]))).unwrap();
        assert!(one > 0 && two > one && big > one);
        // A second equal canvas adds tokens but not a second encode peak.
        let encode = 512 * 512 * 128 * ENCODER_LIVE_MAPS * 4;
        let per_ref = two - one;
        assert_eq!(one - per_ref, encode);
        // Order-independent largest-canvas encode peak.
        let a = reference_workspace_bytes(&inputs(1), Some(&facts(&[(256, 256), (1024, 512)])))
            .unwrap();
        let b = reference_workspace_bytes(&inputs(1), Some(&facts(&[(1024, 512), (256, 256)])))
            .unwrap();
        assert_eq!(a, b);
        let base = reference_workspace_bytes(&inputs(2), Some(&facts(&[(512, 512)]))).unwrap();
        let tokens = 32 * 32;
        assert_eq!(base - one, tokens * (4 * 3072 + 3 * 9216) * 2);
    }

    #[test]
    fn inconsistent_or_empty_facts_are_refused() {
        let mut bad = facts(&[(512, 512)]);
        bad.conditioning_tokens += 1;
        assert!(reference_workspace_bytes(&inputs(1), Some(&bad)).is_err());
        assert!(reference_workspace_bytes(&inputs(1), Some(&facts(&[]))).is_err());
    }

    /// Both backends read the SAME package inputs; only the compute width differs.
    #[test]
    fn inputs_come_from_the_package_config_and_autoencoder_header() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../media/mlx-gen/mlx-gen-flux3/tests/fixtures/tiny-snapshot");
        let config = super::super::resolve_checkpoint(&dir).unwrap();
        let candle =
            ReferenceWorkspaceInputs::from_parts(&config, &dir.join("ae.safetensors"), 4).unwrap();
        let mlx =
            ReferenceWorkspaceInputs::from_parts(&config, &dir.join("ae.safetensors"), 2).unwrap();
        assert_eq!(candle.hidden, 64);
        assert_eq!(candle.mlp_hidden, 192);
        assert_eq!(candle.latent_channels, 16);
        assert_eq!(
            candle.cfg_rows, 1,
            "the tiny package is the distilled flow map"
        );
        assert!(candle.encoder_base_channels > 0);
        assert_eq!(
            ReferenceWorkspaceInputs {
                compute_width: 2,
                ..candle
            },
            mlx
        );
        assert!(
            ReferenceWorkspaceInputs::from_parts(&config, &dir.join("ae.safetensors"), 3).is_err()
        );
    }
}
