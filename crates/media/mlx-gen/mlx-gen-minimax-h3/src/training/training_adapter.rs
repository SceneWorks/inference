//! The **training adapter** on the MLX H3-Image trainer — the twin of
//! `candle_gen_minimax_h3::training::training_adapter`, and the Z-Image precedent (sc-25213)
//! carried to H3-Image: a frozen de-distill LoRA
//! (`ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter`, kohya keys, rank 16 over the
//! attention and MLP of all 50 blocks) applied to the six-step distilled base for every training
//! forward and backward, so the user's adapter learns the concept instead of un-distilling the
//! model.
//!
//! # How it is applied
//!
//! Through the **inference** H3-Image install, [`apply_minimax_h3_image_adapters`] — the key
//! mapping a render uses for a community LoRA (kohya, the ComfyUI fused `qkv_proj` split and SwiGLU
//! half swap, diffusers/PEFT, LoKr, the alpha precedence chain) — as a **frozen forward-time
//! residual** over the untouched, possibly packed, base. The Z-Image MLX trainer merges its adapter
//! into a dense base instead; H3-Image trains on packed q4/q8 tiers and streams its blocks from
//! the staged tier, so there is no dense resident weight to merge into.
//!
//! It is installed right after the DiT loads, before the AdaLN precompute and the trainable
//! factors — the point the generator installs a render's adapters. On the deferred (streamed) DiT
//! it lands in the block stream's adapter overlay, which re-applies it to every block a window
//! rebuilds; on the resident DiT it lands on the block linears directly.
//!
//! The trainable install ([`TrainAdapter::install_as`](mlx_gen::train::lora::TrainAdapter::install_as))
//! **replaces** a target leaf's residual stack with the trainable factor. So the frozen residuals
//! of every *trainable target* leaf are captured once here ([`FrozenResiduals`]) and appended back
//! after each trainable install ([`FrozenResiduals::reattach`]) — the resident traced `loss_fn`,
//! the streamed forward's concrete install and each block's `vjp` closure alike. A leaf the user
//! does not train keeps the residual its load (or window rebuild) gave it.
//!
//! # Why previews and the saved adapter never see it
//!
//! A preview is rendered by a **fresh** registered generator whose load names only the
//! in-progress trainable factors, never the training DiT; the saved adapter, every checkpoint and
//! every resume bundle are written from the trainable [`LoraParams`](mlx_gen::train::lora::LoraParams)
//! alone. The frozen factors are plain arrays outside that map, so no gradient reaches them.
//!
//! # Memory
//!
//! Priced from the inference adapter memory profile of the file (header-only, through
//! [`ImageMemoryFacts::resolve`]) reduced by [`TrainingAdapterBytes`]: unified memory holds every
//! factor for the whole run, plus the residuals' retained forward rows.

use std::collections::BTreeMap;
use std::path::Path;

use mlx_rs::error::{Exception, Result as MlxResult};

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::gen_core::{AdapterKind, AdapterSpec, LoadSpec};
use mlx_gen::{Error, Result};

use gen_core_train::h3_image::{target_block, TrainingAdapterBytes, TRAINER_ID};

use crate::adapters::apply_minimax_h3_image_adapters;
use crate::dit::config::MiniMaxH3ImageDitConfig;
use crate::h3_image::admission::ImageMemoryFacts;
use crate::h3_image::dit::{image_adapter_target_paths, MiniMaxH3ImageDit};

/// The strength the training adapter is applied at — ai-toolkit's merge strength.
pub const TRAINING_ADAPTER_STRENGTH: f32 = 1.0;

/// The install spec of the training adapter at `file`.
pub(crate) fn training_adapter_spec(file: &Path) -> AdapterSpec {
    AdapterSpec::new(
        file.to_path_buf(),
        TRAINING_ADAPTER_STRENGTH,
        AdapterKind::Lora,
    )
}

/// The frozen training-adapter residuals of the run's **trainable target** leaves, by dotted
/// module path — what each trainable install's stack replacement must append back. Empty when no
/// adapter is configured (every method is then a no-op).
#[derive(Clone, Default)]
pub(crate) struct FrozenResiduals {
    by_path: BTreeMap<String, Vec<Adapter>>,
}

fn segments(path: &str) -> Vec<&str> {
    path.split('.').collect()
}

impl FrozenResiduals {
    /// Append the frozen residuals back onto the trainable target leaves `host` resolves — of
    /// block `block` only (a one-block host), or of every block (`None`, the resident DiT) — after
    /// a trainable install replaced each leaf's stack with its trainable factor.
    pub(crate) fn reattach<H: AdaptableHost>(
        &self,
        host: &mut H,
        block: Option<usize>,
    ) -> MlxResult<()> {
        for (path, frozen) in &self.by_path {
            if block.is_some_and(|b| target_block(path) != Some(b)) {
                continue;
            }
            let lin = host.adaptable_mut(&segments(path)).ok_or_else(|| {
                Exception::custom(format!(
                    "{TRAINER_ID} trainer: training adapter target {path} does not resolve"
                ))
            })?;
            let mut stack = lin.adapters().to_vec();
            stack.extend(frozen.iter().cloned());
            lin.set_training_adapters(stack);
        }
        Ok(())
    }

    /// Frozen residuals keyed by trainable target path — a test seam.
    #[cfg(test)]
    pub(crate) fn from_paths(by_path: BTreeMap<String, Vec<Adapter>>) -> Self {
        Self { by_path }
    }
}

/// Install the training adapter at `file` on `dit` as a frozen residual through the inference
/// install, force its factors once (so no traced step pulls a lazy adapter-file read), and capture
/// the residuals of the `target_paths` the run trains. Strict like a render's install: a file
/// naming a module the DiT does not have, or matching nothing, is a named error.
pub(crate) fn install_training_adapter(
    dit: &mut MiniMaxH3ImageDit,
    file: &Path,
    target_paths: &[String],
) -> Result<FrozenResiduals> {
    apply_minimax_h3_image_adapters(dit, &[training_adapter_spec(file)]).map_err(|e| {
        Error::Msg(format!(
            "{TRAINER_ID} trainer: training adapter {} could not be applied: {e}",
            file.display()
        ))
    })?;
    // A deferred DiT holds its block residuals in the stream overlay (forced here); its trunk and
    // a resident DiT's every leaf hold theirs on the linears.
    dit.materialize_streamed_adapters()?;
    let deferred = dit.is_deferred();
    for path in image_adapter_target_paths(&MiniMaxH3ImageDitConfig::default()) {
        if deferred && path.starts_with("transformer_blocks.") {
            continue;
        }
        if let Some(lin) = dit.adaptable_mut(&segments(&path)) {
            lin.materialize_adapters()?;
        }
    }
    let mut by_path = BTreeMap::new();
    for path in target_paths {
        let lin = dit.adaptable_mut(&segments(path)).ok_or_else(|| {
            Error::Msg(format!(
                "{TRAINER_ID} trainer: target {path} does not resolve on the DiT"
            ))
        })?;
        let frozen = lin.adapters().to_vec();
        if !frozen.is_empty() {
            by_path.insert(path.clone(), frozen);
        }
    }
    Ok(FrozenResiduals { by_path })
}

/// The training adapter's footprint: the inference adapter memory profile of the file alone at
/// `spec`'s snapshot, reduced for the trainer. Headers only; zero when none is configured.
pub(crate) fn training_adapter_bytes(
    spec: &LoadSpec,
    file: Option<&Path>,
) -> Result<TrainingAdapterBytes> {
    let Some(file) = file else {
        return Ok(TrainingAdapterBytes::default());
    };
    let mut spec = spec.clone();
    spec.adapters = vec![training_adapter_spec(file)];
    let facts = ImageMemoryFacts::resolve(&spec).map_err(|e| {
        Error::Msg(format!(
            "{TRAINER_ID} trainer: training adapter {}: {e}",
            file.display()
        ))
    })?;
    Ok(TrainingAdapterBytes::from_profile(&facts.adapters))
}
