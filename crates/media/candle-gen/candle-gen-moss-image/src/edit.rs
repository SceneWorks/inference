//! Frozen edit package loading, deterministic reference encoding and request-local sampling.
use crate::{
    loader::{load_weights, tensor_f32, Sources},
    text_encoder::digest_f32,
    transformer::check_cancel as check,
    transformer::Transformer,
};
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::driver::{
    run_candle_driver_slices, DriverEvaluationKind, DriverRefusalReasons, DriverRunState,
    DriverStepInfo, DriverSupport, FamilyDriver, SliceInput, SliceSpec, StepHooks,
};
use candle_gen::gen_core::guidance::{cfg as guidance_cfg, pag_apply};
use candle_gen::gen_core::snapshot_inventory::hex_lower as hex;
use candle_gen::gen_core::{
    weightsmeta::{CheckpointMeta, Dtype as StoredDtype},
    BoundaryState, PinnedWeightsFile,
};
use candle_gen::gen_core::{CancelFlag, LoadSpec, Progress, WeightsSource};
use candle_gen::weights::Weights;
use candle_gen::{CandleError as Error, CandleLatentOps, Result};
use candle_gen_flux2::vae::{Flux2Vae, Flux2VaeDecoderConfig};
use gen_core_moss_image::{moss_image::*, moss_image_edit::*};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, path::Path};

/// Preserve the declared source precision policy without mutating process-global CUDA flags.
pub(crate) fn check_precision(device: &Device) -> Result<()> {
    #[cfg(feature = "cuda")]
    if device.is_cuda()
        && (candle_gen::candle_core::cuda_backend::gemm_reduced_precision_f32()
            || candle_gen::candle_core::cuda_backend::gemm_reduced_precision_bf16())
    {
        return Err(Error::Unsupported(
            "MOSS edit requires strict CUDA FP32/BF16 GEMM accumulation".into(),
        ));
    }
    let _ = device;
    Ok(())
}

pub(crate) struct EditAssets {
    config: PinnedWeightsFile,
    weights: PinnedWeightsFile,
}
fn pin(spec: &LoadSpec, path: &Path, digest: &str) -> Result<PinnedWeightsFile> {
    let path = std::path::absolute(path).map_err(candle_gen::gen_core::Error::from)?;
    let pin = if spec.prepared_file_pins().is_prepared() {
        if !spec.prepared_file_pins().is_finalized() {
            return Err(Error::Unsupported(
                "MOSS edit requires finalized prepared pins".into(),
            ));
        }
        spec.prepared_file_pins()
            .get(&path)
            .cloned()
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "MOSS edit prepared identity missing {}",
                    path.display()
                ))
            })?
    } else {
        PinnedWeightsFile::pin(&path)?
    };
    if pin.loader_path() != path || hex(pin.require_content_sha256()?) != digest {
        return Err(Error::Unsupported(format!(
            "MOSS edit frozen package hash mismatch: {}",
            path.display()
        )));
    }
    pin.verify_unchanged()?;
    Ok(pin)
}
fn authenticated_bytes(pin: &PinnedWeightsFile, digest: &str) -> Result<Vec<u8>> {
    pin.read_unchanged(|path| {
        let bytes = std::fs::read(path)?;
        if hex(&Sha256::digest(&bytes)) != digest {
            return Err(candle_gen::gen_core::Error::Unsupported(
                "consumed MOSS edit bytes changed".into(),
            ));
        }
        Ok(bytes)
    })
    .map_err(Into::into)
}
impl EditAssets {
    pub(crate) fn prepare(spec: &LoadSpec) -> Result<Self> {
        let root = match spec.components.get(EDIT_COMPONENT) {
            Some(WeightsSource::Dir(path)) => path,
            _ => {
                return Err(Error::Unsupported(
                    "MOSS edit requires the exact moss-image-edit package directory".into(),
                ))
            }
        };
        let config = pin(spec, &root.join("adapter_config.json"), EDIT_CONFIG_SHA256)?;
        let weights = pin(
            spec,
            &root.join("adapter_model.safetensors"),
            EDIT_WEIGHTS_SHA256,
        )?;
        // The exact pinned config is the authority, including all 252 module options/targets.
        let value: serde_json::Value =
            serde_json::from_slice(&authenticated_bytes(&config, EDIT_CONFIG_SHA256)?)
                .map_err(|e| Error::Msg(e.to_string()))?;
        validate_edit_config(&value)?;
        Ok(Self { config, weights })
    }
    pub(crate) fn path(&self) -> &Path {
        self.weights.loader_path()
    }
    pub(crate) fn ensure_unchanged(&self) -> Result<()> {
        self.config.verify_unchanged()?;
        self.weights.verify_unchanged()?;
        Ok(())
    }
    pub(crate) fn install(&self, transformer: &mut Transformer, cancel: &CancelFlag) -> Result<()> {
        check(cancel)?;
        let meta =
            CheckpointMeta::from_bytes(authenticated_bytes(&self.weights, EDIT_WEIGHTS_SHA256)?)?;
        let headers = meta
            .keys()
            .map(|name| {
                let t = meta.tensor(name).expect("indexed edit tensor");
                candle_gen::gen_core::weightsmeta::SafetensorsTensorHeader {
                    name: name.into(),
                    shape: t.shape.to_vec(),
                    dtype: t.dtype,
                    data_bytes: t.data.len() as u64,
                }
            })
            .collect::<Vec<_>>();
        validate_edit_adapter_headers(&headers)?;
        let expected = edit_adapter_shapes();
        if meta.keys().count() != expected.len() {
            return Err(Error::Unsupported(
                "MOSS edit requires exactly 504 A/B tensors".into(),
            ));
        }
        // Validate every owned payload before any adapter tensor reaches the device.
        for (name, shape) in &expected {
            check(cancel)?;
            let t = meta
                .tensor(name)
                .ok_or_else(|| Error::Unsupported(format!("missing edit tensor {name}")))?;
            if t.dtype != StoredDtype::BF16
                || t.shape != shape.as_slice()
                || t.data
                    .chunks_exact(2)
                    .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80)
            {
                return Err(Error::Unsupported(format!(
                    "MOSS edit invalid/nonfinite BF16 tensor {name}"
                )));
            }
        }
        let mut arrays = HashMap::new();
        for (name, shape) in &expected {
            check(cancel)?;
            let t = meta.tensor(name).expect("validated edit tensor");
            arrays.insert(
                name.clone(),
                Tensor::from_raw_buffer(
                    t.data,
                    DType::BF16,
                    shape.as_slice(),
                    transformer.device(),
                )?,
            );
        }
        transformer.install_edit(&Weights::from_map(arrays), 64)?;
        transformer.device().synchronize()?;
        check(cancel)?;
        self.ensure_unchanged()
    }
}
pub(crate) fn extra_peak(
    inputs: MossImageEditInputs<'_>,
    request: &candle_gen::gen_core::GenerationRequest,
) -> Result<u64> {
    let references = MossEditReferenceMemory::checked(inputs)?;
    let reference_tokens = references.tokens;
    let total = references
        .persistent_bytes
        .checked_add(references.encoder_bytes)
        .ok_or_else(|| Error::Unsupported("MOSS edit reference bound overflow".into()))?;
    // Edit-only biased projections cast temporarily, never retaining an FP32 backbone.
    // Bound the text embedder plus one evaluated text block: text*d + 4*d*d + 3*d*f
    // weight elements, plus both latent projections (2*latent*d). This exceeds
    // the complete timestep path (freq*d + 9*d*d).
    // Scratch counts input, dot and biased output for every projection conservatively;
    // text blocks evaluate before the next block, so their cast weights do not overlap.
    const D: u64 = 3072;
    const F: u64 = 8192;
    const TEXT: u64 = 4096;
    const LATENT: u64 = 128;
    let text_tokens = request
        .extensions
        .moss_image
        .as_ref()
        .and_then(|ext| ext.options.as_ref())
        .map_or(4096, |o| o.max_sequence_length) as u64;
    let image_tokens = (request.width as u64 * request.height as u64 / 256).max(reference_tokens);
    let precise_bytes = text_tokens
        .checked_mul(TEXT + 16 * D + 5 * F)
        .and_then(|n| {
            image_tokens
                .checked_mul(3 * D + 3 * LATENT)
                .and_then(|m| n.checked_add(m))
        })
        .and_then(|n| {
            n.checked_add(TEXT * D + 4 * D * D + 3 * D * F + 2 * LATENT * D + 256 + 23 * D)
        })
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::Unsupported("MOSS edit precise projection bound overflow".into()))?;
    // SiTU retains source BF16 boundaries while evaluating sigmoid and scalar division in F32.
    // Eight F32-width FFN intermediates conservatively cover casts, op outputs, and rounded
    // operands/results in one evaluated text/ref/body block; the FFN width exceeds D.
    let situ_bytes = text_tokens
        .checked_add(image_tokens)
        .and_then(|n| n.checked_mul(F * 8 * 4))
        .ok_or_else(|| Error::Unsupported("MOSS edit SiTU scratch overflow".into()))?;
    // F32 scores, exponentials, normalization and BF16-rounded operand scratch.
    // Two F32 matrices are already reserved by the ordinary B*B workspace; replace
    // that reservation with five full matrices at max(R*R, B*(B+R)). This includes
    // the rounded BF16 operand and avoids adding the old per-reference sum(r_i*r_i).
    let body_tokens = text_tokens
        .checked_add(request.width as u64 * request.height as u64 / 256)
        .ok_or_else(|| Error::Unsupported("MOSS edit body scratch overflow".into()))?;
    // Ordinary scratch scales with body tokens. Reference blocks run on the whole
    // prefix; reserve the excess linear/FFN activations when R exceeds B.
    let reference_work = reference_tokens
        .saturating_sub(body_tokens)
        .checked_mul(4 * (12 * D + 4 * F))
        .ok_or_else(|| Error::Unsupported("MOSS edit reference scratch overflow".into()))?;
    let attention_bytes = references
        .attention_elements(body_tokens)?
        .checked_mul(5)
        .and_then(|n| {
            body_tokens
                .checked_mul(body_tokens)
                .and_then(|ordinary| ordinary.checked_mul(2))
                .and_then(|ordinary| n.checked_sub(ordinary))
        })
        .and_then(|n| n.checked_mul(24 * 4))
        .ok_or_else(|| Error::Unsupported("MOSS edit attention scratch overflow".into()))?;
    total
        .checked_add(precise_bytes)
        .and_then(|n| n.checked_add(reference_work))
        .and_then(|n| n.checked_add(situ_bytes))
        .and_then(|n| n.checked_add(attention_bytes))
        .and_then(|n| n.checked_add(2_000_000_000 + 297_271_296 * 2))
        .ok_or_else(|| Error::Unsupported("MOSS edit bound overflow".into()))
}

pub(crate) fn encode_references(
    sources: &Sources,
    inputs: MossImageEditInputs<'_>,
    device: &Device,
    cancel: &CancelFlag,
) -> Result<(Vec<Tensor>, Vec<MossImageEditReferenceReceipt>)> {
    check(cancel)?;
    let weights = load_weights(
        &sources.ordinary,
        MossImageAssetRole::Vae,
        device,
        DType::F32,
        &[
            "encoder.",
            "decoder.",
            "quant_conv.",
            "post_quant_conv.",
            "bn.running_mean",
            "bn.running_var",
        ],
    )?;
    // Source BN requires finite means and finite, nonnegative variances.
    let mean = tensor_f32(&weights.require("bn.running_mean")?)?;
    let var = tensor_f32(&weights.require("bn.running_var")?)?;
    if mean.len() != 128
        || var.len() != 128
        || mean.iter().any(|v| !v.is_finite())
        || var.iter().any(|v| !v.is_finite() || *v < 0.)
    {
        return Err(Error::Unsupported(
            "MOSS edit invalid VAE BN statistics".into(),
        ));
    }
    let map = weights
        .keys()
        .map(|k| Ok((k.clone(), weights.require(k)?)))
        .collect::<Result<HashMap<_, _>>>()?;
    let vae = Flux2Vae::new_with_encoder_and_decoder_config(
        VarBuilder::from_tensors(map, DType::F32, device),
        &Flux2VaeDecoderConfig {
            block_out_channels: MOSS_VAE_DECODER_CHANNELS,
        },
    )?;
    let mut references = Vec::new();
    let mut receipts = Vec::new();
    for original in inputs.references {
        check(cancel)?;
        let image = resize_reference(original, inputs.reference_max_edge)?;
        let values: Vec<f32> = image
            .pixels
            .iter()
            .map(|p| *p as f32 / 127.5 - 1.)
            .collect();
        let pixels = Tensor::from_vec(
            values,
            (1, image.height as usize, image.width as usize, 3),
            device,
        )?
        .permute((0, 3, 1, 2))?
        .contiguous()?;
        let latent = reference_latent(&vae, &pixels, cancel)?;
        let values = tensor_f32(&latent)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Unsupported(
                "nonfinite MOSS reference encoding".into(),
            ));
        }
        receipts.push(MossImageEditReferenceReceipt {
            original_dimensions: [original.width, original.height],
            encoded_dimensions: [image.width, image.height],
            pixels_sha256: hex(&Sha256::digest(&original.pixels)),
            resized_pixels_sha256: hex(&Sha256::digest(&image.pixels)),
            latent_sha256: digest_f32(&values, None),
        });
        references.push(latent);
    }
    device.synchronize()?;
    check(cancel)?;
    Ok((references, receipts))
}
fn reference_latent(vae: &Flux2Vae, image: &Tensor, cancel: &CancelFlag) -> Result<Tensor> {
    check(cancel)?;
    let latent = vae.encode_packed(image)?;
    image.device().synchronize()?;
    check(cancel)?;
    Ok(latent)
}
/// Source CFG and Euler use FP32 even when network predictions are BF16.
#[cfg(test)]
fn cfg_euler_step(
    current: &Tensor,
    conditional: &Tensor,
    unconditional: Option<&Tensor>,
    guidance: f32,
    delta: f32,
) -> Result<Tensor> {
    cfg_pag_euler_step(
        current,
        conditional,
        unconditional,
        None,
        guidance,
        0.0,
        delta,
    )
}
fn cfg_pag_euler_step(
    current: &Tensor,
    conditional: &Tensor,
    unconditional: Option<&Tensor>,
    perturbed: Option<&Tensor>,
    guidance: f32,
    pag_scale: f32,
    delta: f32,
) -> Result<Tensor> {
    let ops = CandleLatentOps;
    let cond_f32 = conditional.to_dtype(DType::F32)?;
    let mut prediction = cond_f32.clone();
    if let Some(negative) = unconditional {
        let negative = negative.to_dtype(DType::F32)?;
        prediction = guidance_cfg(&ops, &prediction, &negative, guidance)?;
    }
    if let Some(perturbed) = perturbed.filter(|_| pag_scale > 0.0) {
        let perturbed = perturbed.to_dtype(DType::F32)?;
        prediction = pag_apply(&ops, &prediction, &cond_f32, &perturbed, pag_scale)?;
    }
    Ok(current
        .to_dtype(DType::F32)?
        .add(&(prediction * f64::from(delta))?)?)
}

/// Typed `FamilyDriver` for Candle MOSS-Image Edit's per-sample CFG/PAG Euler loop (`SW-6969`).
pub struct MossEditDriver<F> {
    support: DriverSupport,
    sigmas: Vec<f32>,
    step_fn: F,
}

impl<F> MossEditDriver<F>
where
    F: Fn(&Tensor, usize, f32, f32, &CancelFlag) -> Result<Tensor>,
{
    pub fn new(sigmas: Vec<f32>, step_fn: F) -> Self {
        Self {
            support: DriverSupport {
                driver_id: "candle_moss_image_edit",
                arbitrary_slice: true,
                carry_history: true,
                reset_history: true,
                renoise_from_clean: false,
                geometry_change: false,
                read_only_boundary_estimate: false,
                previous_estimate_capture: false,
                uses_global_rng: false,
                refusal_reasons: DriverRefusalReasons {
                    renoise_from_clean: "MOSS-Image Edit driver refuses external Renoise transitions",
                    geometry_change: "MOSS-Image Edit driver does not support mid-trajectory spatial resampling",
                    read_only_boundary_estimate: "MOSS-Image Edit per-sample step driver does not expose a separate read-only boundary estimator",
                    ..DriverRefusalReasons::default()
                },
            },
            sigmas,
            step_fn,
        }
    }
}

impl<F> FamilyDriver<Tensor> for MossEditDriver<F>
where
    F: Fn(&Tensor, usize, f32, f32, &CancelFlag) -> Result<Tensor>,
{
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Tensor>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Tensor>,
    ) -> candle_gen::gen_core::Result<BoundaryState<Tensor>> {
        self.support.validate_slice(&input, state)?;
        let expected_steps = self.sigmas.len().saturating_sub(1);
        if input.total_steps != expected_steps {
            return Err(candle_gen::gen_core::Error::Unsupported(format!(
                "{}: total_steps={} does not match edit sigma schedule steps={expected_steps}",
                self.support.driver_id, input.total_steps
            )));
        }
        state.enter_slice(&input)?;

        let mut latents = match input.state {
            BoundaryState::XSigma(t)
            | BoundaryState::CleanEstimate { latent: t, .. }
            | BoundaryState::RouteSpecific { state: t, .. } => t,
        };

        for index in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(candle_gen::gen_core::Error::Canceled);
            }
            let sigma = self.sigmas[index];
            let next_sigma = self.sigmas[index + 1];
            let info = DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: index,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: DriverEvaluationKind::Trajectory,
                sigma,
                next_sigma,
            };
            hooks.observe(&info, &latents)?;
            latents = (self.step_fn)(&latents, index, sigma, next_sigma, input.cancel)
                .map_err(candle_gen::gen_core::Error::from)?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);
            latents = hooks.post_step(&info, latents)?;
            hooks.materialize(&info, &latents)?;
            if input.cancel.is_cancelled() {
                return Err(candle_gen::gen_core::Error::Canceled);
            }
            state.record_outer_step(index);
            hooks.emit_progress(&info);
        }
        state.finish_slice(input.span.end);
        Ok(BoundaryState::XSigma(latents))
    }
}

#[cfg(test)]
std::thread_local! {
    pub(crate) static SAMPLING_TRACE: std::cell::RefCell<Option<Vec<[Tensor; 4]>>> = const { std::cell::RefCell::new(None) };
}
pub(crate) struct EditSampling<'a> {
    pub transformer: &'a Transformer,
    pub resolved: &'a MossImageResolved,
    pub positive: &'a [Tensor],
    pub negative: &'a [Tensor],
    pub references: &'a [Tensor],
    pub use_cache: bool,
    pub images_per_prompt: usize,
}
impl EditSampling<'_> {
    pub(crate) fn run(
        &self,
        noise: &Tensor,
        cancel: &CancelFlag,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let cache = if self.use_cache {
            Some(
                self.transformer
                    .prepare_edit_cache(self.references, cancel)?,
            )
        } else {
            None
        };
        let sigmas = moss_image_schedule(
            self.resolved.steps,
            self.resolved.shift,
            self.resolved.training_shift,
        )?;
        let total_steps = sigmas.len() - 1;
        let latents = noise.to_dtype(DType::F32)?;
        let driver = MossEditDriver::new(sigmas, |latents, index, sigma, next_sigma, cancel| {
            let pag_scale = self.resolved.pag_step_scale(index, total_steps);
            let pag_layers = self
                .resolved
                .pag
                .as_ref()
                .map_or(&[][..], |pag| pag.step_layers(index, total_steps));
            let mut outputs = Vec::new();
            for sample in 0..latents.dim(0)? {
                check(cancel)?;
                let row = sample / self.images_per_prompt;
                let current = latents.narrow(0, sample, 1)?;
                let predict = |context: &Tensor, layers: &[u32]| {
                    self.transformer.forward_edit_with_pag(
                        &current,
                        sigma,
                        crate::transformer::EditConditioning {
                            context,
                            references: self.references,
                            cache: cache.as_ref(),
                        },
                        layers,
                        cancel,
                    )
                };
                let prediction = predict(&self.positive[row], &[])?;
                let negative = if self.resolved.guidance != 1. {
                    Some(predict(&self.negative[row], &[])?)
                } else {
                    None
                };
                let perturbed = if pag_scale > 0.0 {
                    Some(predict(&self.positive[row], pag_layers)?)
                } else {
                    None
                };
                let next = cfg_pag_euler_step(
                    &current,
                    &prediction,
                    negative.as_ref(),
                    perturbed.as_ref(),
                    self.resolved.guidance,
                    pag_scale,
                    next_sigma - sigma,
                )?;
                #[cfg(test)]
                SAMPLING_TRACE.with(|trace| {
                    if let Some(trace) = trace.borrow_mut().as_mut() {
                        trace.push([
                            current.clone(),
                            prediction.to_dtype(DType::F32).unwrap(),
                            negative
                                .as_ref()
                                .unwrap_or(&prediction)
                                .to_dtype(DType::F32)
                                .unwrap(),
                            next.clone(),
                        ]);
                    }
                });
                outputs.push(next);
            }
            Ok(Tensor::cat(&outputs, 0)?)
        });
        let mut run_state = DriverRunState::new();
        let mut on_materialize = |_info: &DriverStepInfo, lat: &Tensor| {
            lat.device()
                .synchronize()
                .map_err(|e| candle_gen::gen_core::Error::Msg(e.to_string()))
        };
        let mut hooks = StepHooks::new(progress);
        hooks.on_materialize = Some(&mut on_materialize);
        let out = run_candle_driver_slices(
            &driver,
            [SliceSpec::full(total_steps)],
            BoundaryState::XSigma(latents),
            total_steps,
            0,
            cancel,
            None,
            &mut run_state,
            &mut hooks,
            |_, state, _| Ok(state),
        )?;
        check(cancel)?;
        Ok(match out {
            BoundaryState::XSigma(t)
            | BoundaryState::CleanEstimate { latent: t, .. }
            | BoundaryState::RouteSpecific { state: t, .. } => t,
        })
    }
}

#[cfg(test)]
mod driver_conformance_tests {
    use super::*;
    use candle_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

    #[test]
    fn candle_moss_edit_driver_passes_tier0_conformance() {
        let sigmas = moss_image_schedule(4, 1.0, 1.0).unwrap();
        let report = verify_driver_conformance(ConformanceCase {
            total_steps: 4,
            split_at: 2,
            seed: 5,
            base_geometry: (512, 512),
            alt_geometry: (1024, 1024),
            make_driver: move || {
                Box::new(MossEditDriver::new(
                    sigmas.clone(),
                    |latents, step, sigma, next_sigma, _cancel| {
                        let cond = (latents * (0.2 + 0.1 * f64::from(sigma) + 0.01 * step as f64))?;
                        let uncond = (latents * 0.05)?;
                        cfg_pag_euler_step(
                            latents,
                            &cond,
                            Some(&uncond),
                            None,
                            2.5,
                            0.0,
                            next_sigma - sigma,
                        )
                    },
                ))
            },
            make_initial: || {
                Tensor::from_vec(vec![0.4f32, -0.2, 0.6, -0.8], (1, 1, 2, 2), &Device::Cpu).unwrap()
            },
            latents_equal: |a, b| {
                let va = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let vb = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                va.len() == vb.len()
                    && va
                        .iter()
                        .zip(vb.iter())
                        .all(|(x, y)| x.to_bits() == y.to_bits())
            },
        })
        .expect("Candle MossEditDriver must pass Tier 0 conformance");
        assert!(report.verified_inert_split);
        assert!(report.verified_cancellation);
        assert_eq!(report.unsplit_evaluations, 4);
    }
}

#[cfg(test)]
mod tests;
