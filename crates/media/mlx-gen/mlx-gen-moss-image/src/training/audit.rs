//! Actual-tensor, bounded-chunk observations for explicit native acceptance only.
use super::*;
use gen_core_testkit::moss_training_probe::RunProbe;
use sha2::{Digest, Sha256};
const CHUNK: usize = 4_194_304;
pub(super) fn digest(arrays: impl IntoIterator<Item = Array>) -> Result<String> {
    let mut hash = Sha256::new();
    for array in arrays {
        hash.update(format!("{:?}:{:?};", array.dtype(), array.shape()).as_bytes());
        let flat = array.reshape(&[-1])?;
        let n = usize::try_from(flat.shape()[0])
            .map_err(|_| Error::Msg("unrepresentable audit tensor".into()))?;
        for start in (0..n).step_by(CHUNK) {
            let end = start + CHUNK.min(n - start);
            let part = mlx_gen::attention::slice_axis(&flat, 0, start as i32, end as i32)?;
            let values = crate::pipeline::logical_f32_readback(&part)?;
            let bytes = values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            hash.update(&bytes);
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}
pub(super) fn factors(params: &LoraParams) -> Result<String> {
    let mut ordered = params.iter().collect::<Vec<_>>();
    ordered.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
    digest(ordered.into_iter().map(|(_, v)| v.clone()))
}
pub(super) fn gradients(probe: &RunProbe, grads: &LoraParams) -> Result<()> {
    let mut finite = true;
    let mut nonzero = false;
    for gradient in grads.values() {
        finite &= gradient.is_finite()?.all(None)?.item::<bool>();
        nonzero |= gradient.abs()?.max(None)?.item::<f32>() > 0.;
    }
    probe.gradients(grads.len() as u64, finite, nonzero)?;
    Ok(())
}
pub(super) fn mask(probe: &RunProbe, mask: &Array, latent: &Array) -> Result<()> {
    let width = latent.shape()[3];
    let height = latent.shape()[2];
    let row = crate::pipeline::logical_f32_readback(&mlx_gen::attention::slice_axis(
        &mask.reshape(&[-1])?,
        0,
        0,
        width,
    )?)?;
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
    latent: Array,
    context: Array,
    mask: Option<Array>,
    noise: Array,
    sigma: f32,
}
pub(super) fn prepare(
    sources: &Sources,
    snapshot: &Snapshot,
    probe: &RunProbe,
    encoder: &Encoder,
    vae: &Flux2Vae,
    host_budget: u64,
) -> Result<Evaluation> {
    let evaluation = &snapshot.request;
    let plan = &snapshot.plan;
    let (latent, mask) = encode_entry(evaluation, &plan.entries[0], plan, host_budget, vae)?;
    let item = &evaluation.items[0];
    let context = encoder
        .encode(
            &caption(&item.caption, evaluation.config.trigger_word.as_deref()),
            4096,
            &evaluation.cancel,
        )?
        .0;
    let noise = random::normal::<f32>(
        latent.shape(),
        None,
        None,
        Some(&random::key(probe.evaluation().noise_seed)?),
    )?;
    eval([&noise])?;
    let context_tokens = u64::try_from(context.shape()[1])
        .map_err(|_| Error::Msg("invalid heldout context shape".into()))?;
    probe.update(|r| r.actual_context_max = r.actual_context_max.max(context_tokens))?;
    sources.ensure_unchanged()?;
    Ok(Evaluation {
        latent,
        context,
        mask,
        noise,
        sigma: probe.evaluation().sigma,
    })
}
impl Evaluation {
    pub(super) fn loss(
        &self,
        model: &mut Transformer,
        params: &LoraParams,
        adapter: &TrainAdapter,
        request: &TrainingRequest,
    ) -> Result<f32> {
        let (loss, _) = loss_grads(
            model,
            params,
            adapter,
            &request.config,
            &self.latent,
            &self.context,
            &self.noise,
            self.sigma,
            self.mask.as_ref(),
            &request.cancel,
        )?;
        if !loss.is_finite() {
            return Err(Error::Unsupported("heldout objective is nonfinite".into()));
        }
        Ok(loss)
    }
}
