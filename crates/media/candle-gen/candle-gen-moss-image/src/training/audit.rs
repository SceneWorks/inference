//! Explicit test-support/terminal acceptance observations from the actual native tensors.
use super::*;
use gen_core_testkit::moss_training_probe::{
    EvaluationContentHasher, EvaluationObservation, EvaluationTensorDescriptor, RunProbe,
    EVALUATION_CONTENT_ENCODING,
};
use sha2::{Digest, Sha256};
const CHUNK: usize = 4_194_304;

pub(super) fn digest(tensors: impl IntoIterator<Item = Tensor>) -> Result<String> {
    let mut hash = Sha256::new();
    for tensor in tensors {
        hash.update(format!("{:?}:{:?};", tensor.dtype(), tensor.dims()).as_bytes());
        let flat = tensor.flatten_all()?;
        for start in (0..flat.elem_count()).step_by(CHUNK) {
            let values = flat
                .narrow(0, start, CHUNK.min(flat.elem_count() - start))?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let bytes = values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            hash.update(&bytes);
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}
pub(super) fn base_handles(weights: &candle_gen::weights::Weights) -> Result<Vec<Tensor>> {
    let mut keys = weights.keys().collect::<Vec<_>>();
    keys.sort();
    keys.into_iter().map(|key| weights.require(key)).collect()
}
pub(super) fn factors(set: &LoraSet) -> Result<String> {
    let mut named = set.named_vars();
    named.sort_by(|(a, _), (b, _)| a.cmp(b));
    digest(named.into_iter().map(|(_, v)| v.as_tensor().clone()))
}
pub(super) fn gradients(probe: &RunProbe, gradients: &GradStore, set: &LoraSet) -> Result<()> {
    let mut finite = true;
    let mut nonzero = false;
    let named = set.named_vars();
    let count = named.len() as u64;
    for (_, var) in named {
        let value = gradients
            .get(var.as_tensor())
            .ok_or_else(|| CandleError::Msg("observed gradient missing".into()))?
            .flatten_all()?;
        for start in (0..value.elem_count()).step_by(CHUNK) {
            let values = value
                .narrow(0, start, CHUNK.min(value.elem_count() - start))?
                .to_vec1::<f32>()?;
            finite &= values.iter().all(|v| v.is_finite());
            nonzero |= values.iter().any(|v| *v != 0.);
        }
    }
    probe.gradients(count, finite, nonzero)?;
    Ok(())
}
pub(super) fn mask(probe: &RunProbe, mask: &Tensor, latent: &Tensor) -> Result<()> {
    let width = latent.dim(3)?;
    let height = latent.dim(2)?;
    let row = mask.flatten_all()?.narrow(0, 0, width)?.to_vec1::<f32>()?;
    probe.update(|r| {
        r.masks
            .insert(format!("{}x{}", width * 16, height * 16), row);
    })?;
    Ok(())
}
pub(super) struct Snapshot {
    pub request: TrainingRequest,
    pub plan: contract::recovery::DataPlan,
}
impl Snapshot {
    pub(super) fn new(
        request: &TrainingRequest,
        manifest: &gen_core_testkit::moss_training_probe::EvaluationManifest,
        host_budget: u64,
    ) -> Result<Self> {
        let request = manifest.request(request)?;
        let plan = contract::recovery::DataPlan::from_request_budget(&request, host_budget)?;
        manifest.verify_snapshot(&plan, request.config.subject_mask_loss.is_some())?;
        Ok(Self { request, plan })
    }
}
pub(super) struct Evaluation {
    latent: Tensor,
    context: Tensor,
    mask: Option<Tensor>,
    noise: Tensor,
    sigma: f32,
}
pub(super) fn prepare(
    sources: &Sources,
    snapshot: &Snapshot,
    probe: &RunProbe,
    encoder: &candle_llm::models::qwen35::Qwen35Model,
    vae: &candle_gen_flux2::vae::Flux2Vae,
    device: &Device,
    host_budget: u64,
) -> Result<Evaluation> {
    let evaluation = &snapshot.request;
    let plan = &snapshot.plan;
    let (latent, mask) =
        encode_entry(evaluation, &plan.entries[0], plan, host_budget, vae, device)?;
    let item = &evaluation.items[0];
    let context = text_encoder::strings(
        sources,
        encoder,
        &[caption(
            &item.caption,
            evaluation.config.trigger_word.as_deref(),
        )],
        4096,
        device,
        &evaluation.cancel,
    )?
    .rows
    .remove(0)
    .detach();
    let noise = sample_noise(latent.dims(), probe.evaluation().noise_seed, device)?;
    let context_tokens = context.dim(1)? as u64;
    probe.update(|r| r.actual_context_max = r.actual_context_max.max(context_tokens))?;
    Ok(Evaluation {
        latent,
        context,
        mask,
        noise,
        sigma: probe.evaluation().sigma,
    })
}
impl Evaluation {
    fn descriptor(tensor: &Tensor) -> Result<EvaluationTensorDescriptor> {
        let mut hash = EvaluationContentHasher::default();
        let flat = tensor.flatten_all()?;
        for start in (0..flat.elem_count()).step_by(CHUNK) {
            let values = flat
                .narrow(0, start, CHUNK.min(flat.elem_count() - start))?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            hash.update(values);
        }
        Ok(EvaluationTensorDescriptor {
            shape: tensor.dims().to_vec(),
            strides: tensor.stride().to_vec(),
            dtype: format!("{:?}", tensor.dtype()),
            content_encoding: EVALUATION_CONTENT_ENCODING.into(),
            content_sha256: hash.finish(),
        })
    }
    pub(super) fn observation(&self, loss: f32) -> Result<EvaluationObservation> {
        let mut tensors = std::collections::BTreeMap::new();
        for (name, tensor) in [
            ("latent", &self.latent),
            ("context", &self.context),
            ("noise", &self.noise),
        ] {
            tensors.insert(name.into(), Self::descriptor(tensor)?);
        }
        if let Some(mask) = &self.mask {
            tensors.insert("mask".into(), Self::descriptor(mask)?);
        }
        Ok(EvaluationObservation {
            tensors,
            sigma: self.sigma,
            loss,
        })
    }

    pub(super) fn loss(
        &self,
        model: &Transformer,
        set: &LoraSet,
        request: &TrainingRequest,
    ) -> Result<f32> {
        let (loss, _) = loss_grads(
            model,
            set,
            &request.config,
            &self.latent,
            &self.context,
            &self.noise,
            self.sigma,
            self.mask.as_ref(),
            &request.cancel,
        )?;
        if !loss.is_finite() {
            return Err(CandleError::Msg("heldout objective is nonfinite".into()));
        }
        Ok(loss)
    }
}
