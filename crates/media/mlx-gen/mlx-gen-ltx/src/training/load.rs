//! Lazy trainer construction: the LTX-2.3 split-weight snapshot and the LTX-2.5 split bundle.

use super::*;

/// Construct the trainer from an LTX-2.3 split-weight snapshot directory (transformer / VAE /
/// connector + the Gemma-3-12B text-encoder snapshot resolved like inference). The transformer loads
/// at **f32 activations × quantized weights** (`quant_f32`) for clean autograd — the base is frozen,
/// gradients flow only through the trainable LoRA factors. Registered via [`mlx_gen::TrainerRegistration`].
///
/// The weights load lazily (sc-2124): construction only checks the spec (and that the Gemma-3
/// override exists), so `validate` and `train`'s refusal floors never read weights; see
/// [`LazyTrainer`](mlx_gen::train::lazy::LazyTrainer).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    snapshot_root_23(spec)?;
    crate::model::resolve_gemma_dir(spec.text_encoder.as_ref())?;
    Ok(Box::new(mlx_gen::train::lazy::LazyTrainer::new(
        trainer_descriptor(),
        validate_floors,
        {
            let spec = spec.clone();
            move || load_weights_23(&spec)
        },
    )))
}

/// The LTX-2.3 snapshot directory a trainer spec names — a single file is refused.
pub(super) fn snapshot_root_23(spec: &LoadSpec) -> Result<&Path> {
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(mlx_gen::Error::Msg(
            "ltx_2_3 trainer expects a split-weight snapshot directory (transformer.safetensors \
             / vae_*.safetensors / connector.safetensors), not a single file"
                .into(),
        )),
    }
}

/// The LTX-2.3 weight load behind [`load_trainer`], run by [`LazyTrainer`](mlx_gen::train::lazy::LazyTrainer) on first need.
pub(super) fn load_weights_23(spec: &LoadSpec) -> Result<LtxTrainer> {
    load_trainer_from_dir(snapshot_root_23(spec)?, spec.text_encoder.as_ref())
}

/// Construct the LTX-2.5 trainer from its split Gemma-4 bundle.  This follows the ordinary
/// provider's component resolver so training cannot quietly combine a 2.5 DiT with a Gemma-3
/// override or a 2.3-shaped VAE configuration.
///
/// The weights load lazily (sc-2124): construction only checks the spec, so `validate` and `train`'s
/// refusal floors never read weights; see [`LazyTrainer`](mlx_gen::train::lazy::LazyTrainer).
pub fn load_trainer_25(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    snapshot_root_25(spec)?;
    Ok(Box::new(mlx_gen::train::lazy::LazyTrainer::new(
        trainer_descriptor_25(),
        validate_floors,
        {
            let spec = spec.clone();
            move || load_weights_25(&spec)
        },
    )))
}

/// The LTX-2.5 split-component directory a trainer spec names — a single checkpoint is refused.
pub(super) fn snapshot_root_25(spec: &LoadSpec) -> Result<&Path> {
    match &spec.weights {
        WeightsSource::Dir(path) => Ok(path),
        WeightsSource::File(_) => Err(mlx_gen::Error::Msg(
            "ltx_2_5 trainer expects a split-component directory, not a single checkpoint".into(),
        )),
    }
}

/// The LTX-2.5 weight load behind [`load_trainer_25`], run by [`LazyTrainer`](mlx_gen::train::lazy::LazyTrainer) on first need.
pub(super) fn load_weights_25(spec: &LoadSpec) -> Result<LtxTrainer> {
    let root = snapshot_root_25(spec)?;
    let bundle = crate::bundle::resolve_split_bundle(spec)?;
    crate::bundle::assert_gemma_version(&bundle)?;
    ensure_ltx25_training_variant(crate::dev_sampler::from_bundle(&bundle)?)?;
    let split = SplitModel::from_model_dir(root)?;
    ensure_ltx25_training_tier(split)?;
    let cfg = LtxConfig::from_bundle(&bundle)?;
    let vae_cfg = LtxVaeConfig::from_bundle(&bundle, LtxComponent::ConvVideoVae)?;
    let transformer_w = Weights::from_file(bundle.require(LtxComponent::Transformer)?.path())?;
    let video_w = Weights::from_file(bundle.require(LtxComponent::ConvVideoVae)?.path())?;
    let connector_path = bundle
        .require(LtxComponent::Transformer)?
        .path()
        .with_file_name("connector.safetensors");
    if !connector_path.is_file() {
        return Err(mlx_gen::Error::Msg(format!(
            "ltx_2_5 trainer: missing connector.safetensors beside {}",
            bundle.require(LtxComponent::Transformer)?.path().display()
        )));
    }
    let connector_w = Weights::from_file(&connector_path)?;
    let text_encoder = Ltx25TextEncoder::from_bundle_av(
        &bundle,
        &connector_w,
        &cfg,
        Precision::quant_bf16(split.bits, split.group),
        mlx_gen::gen_core::OffloadPolicy::Resident,
    )?;
    let tokenizer =
        Ltx25Tokenizer::from_packed_te_file(bundle.require(LtxComponent::TextEncoder)?.path())?;
    let transformer = AvDiT::from_weights(
        &transformer_w,
        &cfg,
        Precision::quant_f32(split.bits, split.group),
    )?;
    // Converted 2.5 tiers split the encoder into a sibling file; raw upstream split bundles keep
    // it inside the ConvVAE component.  Match the ordinary provider's documented preference rather
    // than making training depend on one publication layout.
    let split_encoder = root.join("vae_encoder.safetensors");
    let encoder_path = if split_encoder.is_file() {
        split_encoder
    } else {
        bundle
            .require(LtxComponent::ConvVideoVae)?
            .path()
            .to_path_buf()
    };
    let vae = LtxVideoVae::from_weights_lazy_encoder(&video_w, encoder_path, &vae_cfg)?;
    // Materialize at load (sc-24245; see mlx_gen_qwen_image::loader::load_transformer_with).
    for w in [&connector_w, &transformer_w, &video_w] {
        w.materialize_accessed()?;
    }
    Ok(LtxTrainer {
        descriptor: trainer_descriptor_25(),
        tokenizer: Some(TrainingTokenizer::Gemma4(tokenizer)),
        text_encoder: Some(TrainingTextEncoder::Gemma4(text_encoder)),
        vae,
        transformer: TrainingTransformer::Ltx25(transformer),
        cfg,
    })
}

/// The concrete-typed loader behind [`load_trainer`] (sc-4942 — the first-step memory harness needs
/// the concrete [`LtxTrainer`] to reach `.transformer` / `.vae`, which a `Box<dyn Trainer>` hides).
///
/// `te_override` is `LoadSpec::text_encoder` — the bundled Gemma-3 dir the self-contained LTX install
/// ships beside the tier weights (sc-8827/sc-9989). Threaded into [`resolve_gemma_dir`] exactly as the
/// inference path does (`model.rs`), which as of sc-13664 **requires** the slot: `None` is a load-time
/// error (no env / HF-cache fallback), so the trainer must be handed an explicit TE dir.
pub(super) fn load_trainer_from_dir(
    root: &Path,
    te_override: Option<&WeightsSource>,
) -> Result<LtxTrainer> {
    // Resolve (and validate) the Gemma-3 TE location up front — an absent/bad `LoadSpec::text_encoder`
    // override fails fast here, ahead of the split-weight load, and keeps the wiring unit-testable
    // without the ~24 GB base snapshot (sc-9989). Only the path is resolved here; the heavy
    // `Weights::from_dir` load stays below with the rest of the component loads.
    let gemma_dir = crate::model::resolve_gemma_dir(te_override)?;

    let split = SplitModel::from_model_dir(root)?;
    let cfg = LtxConfig::from_model_dir(root)?;
    let vae_config = LtxVaeConfig::from_model_dir(root)?;

    let gemma_w = Weights::from_dir(&gemma_dir)?;
    let gemma_quant = crate::model::resolve_gemma_quant(&gemma_dir)?;
    let connector_w = Weights::from_file(root.join("connector.safetensors"))?;
    let transformer_w = Weights::from_file(root.join("transformer.safetensors"))?;
    let vae_dec_w = Weights::from_file(root.join("vae_decoder.safetensors"))?;
    let vae_enc_w = Weights::from_file(root.join("vae_encoder.safetensors"))?;

    // Video-only text encoder (bf16, the reference TE dtype); we cast its embeds to f32 per-item for
    // the f32 training forward.
    let text_encoder = LtxTextEncoder::from_weights(
        &gemma_w,
        &connector_w,
        GemmaConfig::gemma_3_12b(),
        gemma_quant,
        &cfg,
        // bf16 activations at the checkpoint's own quant geometry: the connector and the
        // feature-extractor Linear take the packed arm on an LTX-2.5 tier and the dense arm on
        // LTX-2.3's dense `connector.safetensors`, decided per tensor by `.scales` presence.
        Precision::quant_bf16(split.bits, split.group),
    )?;
    let transformer = LtxDiT::from_weights(
        &transformer_w,
        &cfg,
        Precision::quant_f32(split.bits, split.group),
    )?;
    let vae = LtxVideoVae::from_weights(&vae_dec_w, Some(&vae_enc_w), &vae_config)?;
    // Materialize at load (sc-24245; see mlx_gen_qwen_image::loader::load_transformer_with).
    for w in [
        &gemma_w,
        &connector_w,
        &transformer_w,
        &vae_dec_w,
        &vae_enc_w,
    ] {
        w.materialize_accessed()?;
    }
    let tokenizer = LtxTokenizer::from_dir(&gemma_dir)?;

    Ok(LtxTrainer {
        descriptor: trainer_descriptor(),
        tokenizer: Some(TrainingTokenizer::Gemma3(tokenizer)),
        text_encoder: Some(TrainingTextEncoder::Gemma3(text_encoder)),
        vae,
        transformer: TrainingTransformer::Ltx23(transformer),
        cfg,
    })
}
