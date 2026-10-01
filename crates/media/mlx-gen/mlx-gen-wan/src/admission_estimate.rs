//! SC-20686 product admission estimate for one MLX Wan route request (estimate-plus-reserve-v1).
//!
//! The campaign supervisor admits a coordinate when host available memory covers its estimated
//! peak plus the host reserve. This module prices that peak from the product's own memory model,
//! resolved exactly as the run resolves it (the same [`LoadSpec`] and request), without loading a
//! weight or touching MLX:
//!
//! * text encoder: the UMT5 file at its stored width (conditioning phase);
//! * DiT: the generate-time fit gate's resident bytes ([`crate::pipeline::preflight_denoise_memory_guard`]
//!   inputs: one expert under `Sequential`, both under `Resident`, plus additive adapters) and its
//!   `72 B · batch · tokens · dim` activation working set;
//! * VAE encode: the provider's conservative encode working set
//!   ([`crate::conservative_video_encode_memory_profile`], sc-20686 E8);
//! * VAE decode: the provider's conservative single-pass decode working set
//!   ([`crate::conservative_video_decode_memory_profile`]), which bounds every tiled plan the
//!   runtime's free-aware selector can choose.
//!
//! Phases are staged, so the estimate is the MAX of the phase totals, not their sum. The DiT stays
//! resident through decode (measured on the SC-20686 Metal lane: the post-denoise window still holds
//! the expert), so the denoise and decode phases both carry it, beside the VAE weights.

use mlx_gen::gen_core::weightsmeta::{materialized_path_bytes, safetensors_path_bytes};
use mlx_gen::{Conditioning, Error, GenerationRequest, LoadSpec, Result, WeightsSource};

use crate::model::{MODEL_ID, MODEL_ID_I2V_14B, MODEL_ID_T2V_14B};
use crate::model_vace::{MODEL_ID_VACE, MODEL_ID_VACE_FUN};

/// The VACE forward's documented activation under-fit of the dense 72 B/token/dim coefficient
/// (`model_vace.rs`: the hint stack and 96-channel control patch-embed add ~20-30%): the estimate
/// prices the upper end so it never under-admits a VACE denoise.
const VACE_ACTIVATION_UNDERFIT_PERCENT: u64 = 130;

/// One route request's priced components and phase totals (bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WanAdmissionEstimate {
    pub text_encoder_bytes: u64,
    pub vae_bytes: u64,
    pub dit_resident_bytes: u64,
    pub denoise_activation_bytes: u64,
    pub encode_working_set_bytes: u64,
    pub decode_working_set_bytes: u64,
}

impl WanAdmissionEstimate {
    /// `(phase, bytes)` for the four staged phases.
    pub fn phases(&self) -> [(&'static str, u64); 4] {
        let vae_and_dit = self.vae_bytes.saturating_add(self.dit_resident_bytes);
        [
            ("conditioning", self.text_encoder_bytes),
            (
                "encode",
                self.vae_bytes.saturating_add(self.encode_working_set_bytes),
            ),
            (
                "denoise",
                vae_and_dit.saturating_add(self.denoise_activation_bytes),
            ),
            (
                "decode",
                vae_and_dit.saturating_add(self.decode_working_set_bytes),
            ),
        ]
    }

    /// The admission estimate: the largest phase.
    pub fn peak_bytes(&self) -> u64 {
        self.phases()
            .iter()
            .map(|(_, bytes)| *bytes)
            .max()
            .unwrap_or(0)
    }
}

/// The product admission estimate for `route` loaded with `spec` and asked `req`.
pub fn product_admission_estimate(
    route: &str,
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> Result<WanAdmissionEstimate> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(Error::Msg(format!(
            "{route}: expected a model directory for the admission estimate"
        )));
    };
    let (facts, activation_percent) = match route {
        MODEL_ID | MODEL_ID_T2V_14B | MODEL_ID_I2V_14B => {
            (crate::model::dense_denoise_facts(route, spec, req)?, 100)
        }
        MODEL_ID_VACE | MODEL_ID_VACE_FUN => (
            crate::model_vace::vace_denoise_facts(route, spec, req)?,
            VACE_ACTIVATION_UNDERFIT_PERCENT,
        ),
        other => {
            return Err(Error::Msg(format!(
                "unsupported SC-20686 Wan route for the admission estimate: {other}"
            )))
        }
    };
    let overflow = || Error::Msg(format!("{route}: admission estimate overflows"));
    let denoise_activation_bytes =
        crate::pipeline::denoise_activation_bytes(facts.tokens, facts.dim, facts.cfg_batched)
            .and_then(|bytes| bytes.checked_mul(activation_percent))
            .map(|bytes| bytes.div_ceil(100))
            .ok_or_else(overflow)?;
    let text_encoder = root.join("t5_encoder.safetensors");
    let text_encoder_bytes = safetensors_path_bytes(&text_encoder)
        .max(materialized_path_bytes(&text_encoder, 2).map_err(Error::from)?);
    // Both Wan VAEs are opened at f32 or cast down from it: pricing every float tensor at four bytes
    // never under-counts the resident decoder.
    let vae_bytes =
        materialized_path_bytes(root.join("vae.safetensors"), 4).map_err(Error::from)?;
    let references = req
        .conditioning
        .iter()
        .map(|conditioning| match conditioning {
            Conditioning::Reference { .. } => 1,
            Conditioning::MultiReference { images } => images.len(),
            _ => 0,
        })
        .sum::<usize>();
    // The request's video mode when it states one; otherwise the mode its conditioning implies.
    let mode = req.video_mode.as_deref().unwrap_or(if references == 0 {
        "text_to_video"
    } else {
        "image_to_video"
    });
    let encode_working_set_bytes = crate::conservative_video_encode_memory_profile(
        route,
        mode,
        facts.width,
        facts.height,
        facts.frames,
        u32::try_from(references).map_err(|_| overflow())?,
    )
    .map_or(0, |profile| profile.working_set_bytes());
    let decode_working_set_bytes = crate::conservative_video_decode_memory_profile(
        route,
        facts.width,
        facts.height,
        facts.frames,
    )
    .ok_or_else(|| Error::Msg(format!("{route}: no conservative decode memory profile")))?
    .working_set_bytes();
    Ok(WanAdmissionEstimate {
        text_encoder_bytes,
        vae_bytes,
        dit_resident_bytes: facts.resident_bytes,
        denoise_activation_bytes,
        encode_working_set_bytes,
        decode_working_set_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::OffloadPolicy;
    use std::path::Path;

    const GIB: u64 = 1 << 30;

    /// A sparse file of `bytes` (only its length is read for DiT / adapter sizing).
    fn sparse(path: &Path, bytes: u64) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::File::create(path).unwrap().set_len(bytes).unwrap();
    }

    /// A one-tensor safetensors file of `bytes` stored BF16 bytes (payload sparse).
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

    /// The SC-20686 W2 q4 tier layouts (`.github/kv-poc/models-w2.tsv` sizes; payloads sparse).
    fn snapshot(root: &Path, model_type: &str, dual: bool) {
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(
            root.join("config.json"),
            serde_json::json!({
                "model_type": model_type,
                "dual_model": dual,
                "quantization": {"bits": 4, "group_size": 64},
            })
            .to_string(),
        )
        .unwrap();
        safetensors(&root.join("t5_encoder.safetensors"), 11_361_845_504);
        if dual {
            sparse(&root.join("high_noise_model.safetensors"), 8_377_000_000);
            sparse(&root.join("low_noise_model.safetensors"), 8_377_000_000);
            safetensors(&root.join("vae.safetensors"), 507_591_212 / 2);
        } else {
            sparse(&root.join("model.safetensors"), 2_942_000_000);
            safetensors(&root.join("vae.safetensors"), 2_826_000_000 / 2);
        }
    }

    fn request(width: u32, height: u32, frames: u32, guidance: f32) -> GenerationRequest {
        GenerationRequest {
            prompt: "SC-20686".into(),
            width,
            height,
            frames: Some(frames),
            count: 1,
            seed: Some(42),
            steps: Some(4),
            guidance: Some(guidance),
            ..Default::default()
        }
    }

    fn estimate(
        route: &str,
        root: &Path,
        policy: OffloadPolicy,
        req: &GenerationRequest,
    ) -> WanAdmissionEstimate {
        let spec =
            crate::product_load::product_load_spec(route, root, policy, false, None).unwrap();
        product_admission_estimate(route, &spec, req).unwrap()
    }

    #[test]
    fn the_estimate_is_the_largest_staged_phase_of_the_product_model() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("t2v/q4");
        snapshot(&root, "t2v", true);
        let req = request(512, 512, 17, 5.0);
        let priced = estimate(MODEL_ID_T2V_14B, &root, OffloadPolicy::Sequential, &req);
        // Sequential holds one expert; Resident both (the fit gate's residency).
        assert_eq!(priced.dit_resident_bytes, 8_377_000_000);
        let resident = estimate(MODEL_ID_T2V_14B, &root, OffloadPolicy::Resident, &req);
        assert_eq!(resident.dit_resident_bytes, 2 * 8_377_000_000);
        // The provider's own decode/activation pricing, not a re-derivation.
        assert_eq!(
            priced.decode_working_set_bytes,
            crate::conservative_video_decode_memory_profile(MODEL_ID_T2V_14B, 512, 512, 17)
                .unwrap()
                .working_set_bytes()
        );
        // 17 frames -> 5 latent frames x 64 x 64 / (2 x 2) patches; A14B CFG is batched.
        assert_eq!(
            priced.denoise_activation_bytes,
            crate::pipeline::denoise_activation_bytes(5 * 32 * 32, 5_120, true).unwrap()
        );
        assert_eq!(priced.encode_working_set_bytes, 0, "T2V encodes nothing");
        let phases = priced.phases();
        let max = phases.iter().map(|(_, bytes)| *bytes).max().unwrap();
        let sum: u64 = phases.iter().map(|(_, bytes)| *bytes).sum();
        assert_eq!(priced.peak_bytes(), max);
        assert!(
            priced.peak_bytes() < sum,
            "staged phases are a max, not a sum"
        );
        assert_eq!(phases[3].0, "decode");
        assert_eq!(
            phases[3].1,
            priced.vae_bytes + priced.dit_resident_bytes + priced.decode_working_set_bytes
        );
    }

    /// The SC-20686 Metal lane's accepted D units (nax-macos-2, run 36907062374): the product
    /// estimate must cover every measured process `phys_footprint` peak (else the product
    /// under-admits: an E8 admission bug).
    #[test]
    fn the_estimate_covers_every_measured_sc20686_metal_peak() {
        let dir = tempfile::tempdir().unwrap();
        let ti2v = dir.path().join("ti2v/q4");
        snapshot(&ti2v, "ti2v", false);
        let t2v = dir.path().join("t2v/q4");
        snapshot(&t2v, "t2v", true);
        // Lightning: the product's T2V pair (1.14 GiB per expert) in a hub-shaped snapshot dir.
        let lightning = dir.path().join("hub/snapshots/rev");
        for expert in ["high", "low"] {
            sparse(
                &lightning
                    .join("Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1")
                    .join(format!("{expert}_noise_model.safetensors")),
                1_225_000_000,
            );
        }
        let lightning_spec = crate::product_load::product_load_spec(
            MODEL_ID_T2V_14B,
            &t2v,
            OffloadPolicy::Sequential,
            true,
            Some(&lightning),
        )
        .unwrap();
        let cases = [
            (
                estimate(
                    MODEL_ID,
                    &ti2v,
                    OffloadPolicy::Sequential,
                    &request(768, 512, 33, 4.0),
                ),
                40_334_756_240_u64,
            ),
            (
                estimate(
                    MODEL_ID,
                    &ti2v,
                    OffloadPolicy::Sequential,
                    &request(512, 512, 17, 5.0),
                ),
                16_446_179_632,
            ),
            (
                estimate(
                    MODEL_ID_T2V_14B,
                    &t2v,
                    OffloadPolicy::Sequential,
                    &request(768, 512, 33, 4.0),
                ),
                67_621_094_696,
            ),
            (
                product_admission_estimate(
                    MODEL_ID_T2V_14B,
                    &lightning_spec,
                    &request(768, 512, 33, 1.0),
                )
                .unwrap(),
                67_541_059_000,
            ),
        ];
        for (priced, measured) in cases {
            assert!(
                priced.peak_bytes() >= measured,
                "estimate {:.2} GiB under-prices the measured {:.2} GiB peak: {priced:?}",
                priced.peak_bytes() as f64 / GIB as f64,
                measured as f64 / GIB as f64
            );
        }
    }
}
