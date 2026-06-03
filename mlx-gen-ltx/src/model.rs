//! `mlx-gen-ltx` model entry: the LTX-2.3 **video-only T2V** descriptor, the config-driven
//! `load`, and registry self-registration.
//!
//! **Scope (sc-2679 S0):** this slice ships the foundation — crate scaffold, registry wiring, the
//! `embedded_config.json`-driven config, SPLIT 3-D RoPE (double-precision), the f32 position grid,
//! the distilled sigma schedules, and the legacy Euler step. The denoise pipeline itself (Gemma-3
//! TE → connector → 48-layer DiT → video VAE → 2-stage upsample/refine) lands across S1–S5, so
//! `generate` returns an explicit "not yet wired" error until then. `load` already reads + validates
//! the real model's `embedded_config.json` so the config seam is exercised end-to-end now.

use mlx_gen::{
    Capabilities, Error, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Precision, Progress, Result, WeightsSource,
};

use crate::config::LtxConfig;

/// Public registry id: `mlx_gen::load("ltx_2_3", spec)`.
pub const MODEL_ID: &str = "ltx_2_3";

/// Stable identity + advertised capabilities for the LTX-2.3 video-only core.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "ltx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // Distilled 2-stage path: CFG is forced to 1.0, so no guidance / negative prompt in
            // the core. (I2V, LoRA, LoKr, Q4/Q8, and the audio half are sibling slices.)
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            conditioning: Vec::new(),
            supports_lora: false,
            supports_lokr: false,
            samplers: Vec::new(),
            schedulers: Vec::new(),
            // height/width must be divisible by 64 (stage-1 runs at //2//32).
            min_size: 64,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// The loaded LTX-2.3 model. S0 holds the resolved config; the network components (TE, connector,
/// transformer, VAE, upsampler) attach across S1–S5.
pub struct Ltx {
    descriptor: ModelDescriptor,
    #[allow(dead_code)] // consumed by the S1–S5 pipeline.
    config: LtxConfig,
}

/// Load the model from a snapshot directory. Reads + validates `embedded_config.json` (the
/// config seam). Dense f32 activations are the target for LTX (quality + dodging the pmetal bf16
/// GEMM bug); quantization + adapters are sibling slices, rejected here for now.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p,
            WeightsSource::File(_) => return Err(Error::Msg(
                "ltx_2_3: expected a model directory (split-weight snapshot), not a single file"
                    .into(),
            )),
        };
    if spec.precision != Precision::Bf16 {
        // The dense LTX path runs f32 activations regardless; no precision override is wired yet.
        return Err(Error::Msg(
            "ltx_2_3: precision override is not wired (the dense path runs f32 activations)".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(Error::Msg(
            "ltx_2_3: Q4/Q8 quantization is a sibling slice (sc-2686), not yet wired".into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "ltx_2_3: LoRA/LoKr adapters are sibling slices (sc-2687 / sc-2393), not yet wired"
                .into(),
        ));
    }

    let config = LtxConfig::from_model_dir(root)?;
    Ok(Box::new(Ltx {
        descriptor: descriptor(),
        config,
    }))
}

impl Generator for Ltx {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        if !req.width.is_multiple_of(64) || !req.height.is_multiple_of(64) {
            return Err(Error::Msg(format!(
                "ltx_2_3: width/height must be divisible by 64 (got {}x{})",
                req.width, req.height
            )));
        }
        if let Some(frames) = req.frames {
            if frames % 8 != 1 {
                return Err(Error::Msg(format!(
                    "ltx_2_3: num_frames must be 1 + 8·k (got {frames})"
                )));
            }
        }
        Ok(())
    }

    fn generate(
        &self,
        _req: &GenerationRequest,
        _on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        Err(Error::Msg(
            "ltx_2_3: T2V denoise pipeline is not yet wired — S0 ships the scaffold, config, \
             RoPE, position grid, sigma schedules, and Euler step; the TE/connector/DiT/VAE/2-stage \
             pipeline lands in S1–S5 (sc-2679)"
                .into(),
        ))
    }
}

inventory::submit! {
    mlx_gen::ModelRegistration { descriptor, load }
}
