//! The **training adapter** on the candle H3-Image trainer — ai-toolkit's `assistant_lora_path`,
//! the Z-Image precedent (sc-25213) carried to H3-Image: a frozen de-distill LoRA
//! (`ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter`, kohya keys, rank 16 over the
//! attention and MLP of all 50 blocks) applied to the six-step distilled base for every training
//! forward and backward, so the user's adapter learns the concept instead of un-distilling the
//! model (a plain flow-matching fine-tune of the distilled base samples blurry, with ghosted,
//! multiply-exposed limbs).
//!
//! # How it is applied
//!
//! Through the **inference** H3-Image install, [`apply_minimax_h3_image_adapters`] — the exact key
//! mapping a render uses for a community LoRA (kohya `lora_unet_blocks_*`, the ComfyUI fused
//! `qkv_proj` split and SwiGLU half swap, diffusers/PEFT, LoKr, the alpha precedence chain). It
//! lands as a **frozen forward-time residual** on each targeted
//! [`AdaptLinear`](candle_gen::quant::AdaptLinear): the base weight is
//! never touched (a packed q4/q8 tier stays packed — a merge could not run there at all), and the
//! factors are plain tensors, never `Var`s, so no gradient or optimizer state reaches them.
//!
//! It is installed right after the DiT loads, before the trainable factors, the streamed-window
//! preparation and the AdaLN precompute — the same point the generator installs a render's
//! adapters — so it rides every block residency the trainer has: a window copy of a host block
//! carries its residual as it carries the trainable leaves, the resident promotion moves it with
//! the block, and the non-checkpointed load installs it on the device directly.
//!
//! # Why previews and the saved adapter never see it
//!
//! A preview is rendered by a **fresh** registered generator whose load names only the
//! in-progress trainable factors (`PreviewSetup::render`), never the training DiT — so a preview
//! renders the bare distilled base plus the user's adapter, exactly what the saved file renders at
//! generation time, without any strength switch to remember. The saved adapter (and every
//! checkpoint and resume bundle) is written from the run's [`LoraSet`](candle_gen::train::lora::LoraSet)
//! `Var`s alone, which hold no de-distill tensor.
//!
//! # Memory
//!
//! Priced from the inference adapter memory profile of the file (header-only), reduced by
//! [`TrainingAdapterBytes`]: every factor on the device for a resident run; the trunk share plus
//! one block's window for a streamed one (the blocks' share stays on the host beside the stack);
//! plus the residuals' retained forward rows of one live block (checkpointed) or every block.

use std::path::Path;

use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::{CandleError as Error, Result};

use gen_core_train::h3_image::{TrainingAdapterBytes, TRAINER_ID};

use crate::adapters::{apply_minimax_h3_image_adapters, image_adapter_target_paths};
use crate::dit::{MiniMaxH3Dit, MiniMaxH3DitConfig};

/// The strength the training adapter is applied at — ai-toolkit's merge strength.
pub const TRAINING_ADAPTER_STRENGTH: f32 = 1.0;

/// The install spec of the training adapter at `file` and `strength`.
pub(crate) fn training_adapter_spec(file: &Path, strength: f32) -> AdapterSpec {
    AdapterSpec::new(file.to_path_buf(), strength, AdapterKind::Lora)
}

/// Install the training adapter at `file` on `dit` at `strength` as a frozen residual, through the
/// inference install. Returns the number of projections it adapts. Strict like a render's install:
/// a file that names a module the DiT does not have, or matches nothing, is a named error.
pub(crate) fn install_training_adapter(
    dit: &mut MiniMaxH3Dit,
    file: &Path,
    strength: f32,
) -> Result<usize> {
    let report = apply_minimax_h3_image_adapters(dit, &[training_adapter_spec(file, strength)])
        .map_err(|e| {
            Error::Msg(format!(
                "{TRAINER_ID} trainer: training adapter {} could not be applied: {e}",
                file.display()
            ))
        })?;
    Ok(report.applied)
}

/// The training adapter's footprint on a DiT staged at `dit_dir` with geometry `dit_cfg` over
/// `projections` (the staged partition's native projections, from the memory facts): the
/// inference adapter memory profile of the file alone, reduced for the trainer. Headers only.
/// Zero when no adapter is configured.
pub(crate) fn training_adapter_bytes(
    dit_dir: &Path,
    dit_cfg: &MiniMaxH3DitConfig,
    projections: &[candle_gen::gen_core::h3_memory::NativeProjection],
    file: Option<&Path>,
    strength: f32,
) -> Result<TrainingAdapterBytes> {
    let Some(file) = file else {
        return Ok(TrainingAdapterBytes::default());
    };
    let profile = crate::memory_strategy::adapter_profile_for(
        dit_dir,
        &image_adapter_target_paths(dit_cfg),
        &[training_adapter_spec(file, strength)],
        projections,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "{TRAINER_ID} trainer: training adapter {}: {e}",
            file.display()
        ))
    })?;
    Ok(TrainingAdapterBytes::from_profile(&profile))
}
