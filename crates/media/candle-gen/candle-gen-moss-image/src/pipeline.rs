//! FP32 author Euler/CFG/SLG and fixed-noise DMD recurrence.
use crate::{
    loader::{load_weights, tensor_f32, Asset, Sources},
    text_encoder::{digest_f32, Conditioning},
    transformer::{check_cancel, install_replacements, Geometry, Transformer},
};
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::driver::{
    run_candle_driver_slices, DriverEvaluationKind, DriverRefusalReasons, DriverRunState,
    DriverStepInfo, DriverSupport, FamilyDriver, SliceInput, SliceSpec, StepHooks,
};
use candle_gen::gen_core::guidance::{cfg as guidance_cfg, pag_apply};
use candle_gen::gen_core::{
    self, AdapterApplyReport, BoundaryState, CancelFlag, EstimateSource, GenerationRequest, Image,
    Progress,
};
use candle_gen::{CandleError, CandleLatentOps, Result};
use candle_gen_flux2::vae::Flux2Vae;
use gen_core_moss_image::moss_image::*;
use rand::{rngs::StdRng, SeedableRng};

pub(crate) struct Renderer {
    pub mode: MossImageAcceleration,
    pub transformer: Transformer,
    pub dataspace: Option<(Tensor, Vec<f32>)>,
    pub adapter_reports: Vec<AdapterApplyReport>,
    pub dmd_report: Option<AdapterApplyReport>,
}
pub(crate) fn application_reports(
    dmd_report: Option<&AdapterApplyReport>,
    custom_reports: Vec<AdapterApplyReport>,
) -> Vec<AdapterApplyReport> {
    dmd_report
        .into_iter()
        .cloned()
        .chain(custom_reports)
        .collect()
}
impl Renderer {
    pub fn load(sources: &Sources, mode: MossImageAcceleration, device: &Device) -> Result<Self> {
        let assets = sources.assets(mode)?;
        let dtype = if mode == MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32) {
            DType::F32
        } else {
            DType::BF16
        };
        let base = load_weights(
            &assets,
            MossImageAssetRole::Transformer,
            device,
            dtype,
            &[""],
        )?;
        let mixed = mode == MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16);
        let adapter = if mixed {
            Some(load_weights(
                &assets,
                MossImageAssetRole::DmdLora,
                device,
                DType::F32,
                &[""],
            )?)
        } else {
            None
        };
        let base = if mixed {
            install_replacements(
                &base,
                &load_weights(
                    &assets,
                    MossImageAssetRole::DmdTime,
                    device,
                    DType::F32,
                    &[""],
                )?,
            )?
        } else {
            base
        };
        let mut transformer =
            Transformer::load(&base, Geometry::from(&sources.config), adapter.as_ref())?;
        let dataspace = if matches!(mode, MossImageAcceleration::Dmd(_)) {
            let w = load_weights(
                &assets,
                MossImageAssetRole::DmdDataspace,
                device,
                DType::F32,
                &[""],
            )?;
            Some((
                w.require("latent_mix.Q")?,
                tensor_f32(&w.require("step_codes")?)?,
            ))
        } else {
            None
        };
        let dmd_report = if mixed {
            Some(AdapterApplyReport {
                adapter_path: assets
                    .iter()
                    .find(|a| a.role == MossImageAssetRole::DmdLora)
                    .expect("validated mixed assets")
                    .pin
                    .loader_path()
                    .into(),
                applied: 252,
                skipped: vec![],
            })
        } else {
            None
        };
        let custom_reports = crate::adapters::apply(
            &mut transformer,
            sources.variant,
            &sources.spec.adapters,
            &sources.adapter_pins,
        )?;
        let adapter_reports = application_reports(dmd_report.as_ref(), custom_reports);
        sources.ensure_unchanged()?;
        Ok(Self {
            mode,
            transformer,
            dataspace,
            adapter_reports,
            dmd_report,
        })
    }
}
pub(crate) struct Sampling {
    pub latents: Tensor,
    pub samples: Vec<MossImageSampleReceipt>,
    pub sampling: MossImageSamplingReceipt,
}
pub(crate) fn sample(
    renderer: &Renderer,
    req: &GenerationRequest,
    resolved: &MossImageResolved,
    inputs: MossImageAdvancedInputs<'_>,
    positive: &Conditioning,
    negative: Option<&Conditioning>,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Sampling> {
    sample_graph(
        &renderer.transformer,
        renderer.dataspace.as_ref(),
        req,
        resolved,
        inputs,
        positive,
        negative,
        on_progress,
    )
}
/// Typed `FamilyDriver` for Candle MOSS-Image's fixed-noise DMD recurrence (`SW-6969`).
pub struct MossDmdDriver<F> {
    support: DriverSupport,
    noise: Tensor,
    q: Tensor,
    codes: Vec<f32>,
    forward: F,
}

impl<F> MossDmdDriver<F>
where
    F: Fn(&Tensor, f32, usize, &CancelFlag) -> Result<Tensor>,
{
    pub fn new(noise: Tensor, q: Tensor, codes: Vec<f32>, forward: F) -> Self {
        Self {
            support: DriverSupport {
                driver_id: "candle_moss_image_dmd",
                arbitrary_slice: true,
                carry_history: true,
                reset_history: true,
                renoise_from_clean: false,
                geometry_change: false,
                read_only_boundary_estimate: true,
                previous_estimate_capture: false,
                uses_global_rng: false,
                refusal_reasons: DriverRefusalReasons {
                    renoise_from_clean: "MOSS-Image DMD uses a fixed-noise linear mix and refuses external Renoise transitions",
                    geometry_change: "MOSS-Image DMD does not support mid-trajectory spatial resampling",
                    ..DriverRefusalReasons::default()
                },
            },
            noise,
            q,
            codes,
            forward,
        }
    }
}

impl<F> FamilyDriver<Tensor> for MossDmdDriver<F>
where
    F: Fn(&Tensor, f32, usize, &CancelFlag) -> Result<Tensor>,
{
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Tensor>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Tensor>,
    ) -> gen_core::Result<BoundaryState<Tensor>> {
        self.support.validate_slice(&input, state)?;
        if input.total_steps > self.codes.len() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: total_steps={} exceeds DMD step_codes len={}",
                self.support.driver_id,
                input.total_steps,
                self.codes.len()
            )));
        }
        state.enter_slice(&input)?;

        let mut x = match input.state {
            BoundaryState::XSigma(t)
            | BoundaryState::CleanEstimate { latent: t, .. }
            | BoundaryState::RouteSpecific { state: t, .. } => t,
        };

        for step in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            let code = self.codes[step];
            let next_code = self.codes.get(step + 1).copied().unwrap_or(0.0);
            let info = DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: DriverEvaluationKind::Trajectory,
                sigma: code,
                next_sigma: next_code,
            };
            hooks.observe(&info, &x)?;
            let mixed_input = dmd_input(&self.noise, &self.q, &x).map_err(gen_core::Error::from)?;
            let v = (self.forward)(&mixed_input, code, step, input.cancel)
                .map_err(gen_core::Error::from)?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);
            x = self
                .noise
                .sub(&v)
                .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
            x = hooks.post_step(&info, x)?;
            hooks.materialize(&info, &x)?;
            if input.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }
        state.finish_slice(input.span.end);

        if input.request_boundary_estimate {
            if input.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            let est = if input.span.end < input.total_steps {
                let code = self.codes[input.span.end];
                let mixed_input =
                    dmd_input(&self.noise, &self.q, &x).map_err(gen_core::Error::from)?;
                let v = (self.forward)(&mixed_input, code, input.span.end, input.cancel)
                    .map_err(gen_core::Error::from)?;
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                self.noise
                    .sub(&v)
                    .map_err(|e| gen_core::Error::Msg(e.to_string()))?
            } else {
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                x.clone()
            };
            return Ok(BoundaryState::CleanEstimate {
                latent: est,
                source: EstimateSource::EndpointEvaluation,
            });
        }

        if input.span.end == input.total_steps {
            Ok(BoundaryState::CleanEstimate {
                latent: x,
                source: EstimateSource::EndpointEvaluation,
            })
        } else {
            Ok(BoundaryState::XSigma(x))
        }
    }
}

/// Typed `FamilyDriver` for Candle MOSS-Image's ordinary FP32 flow-Euler loop (`SW-6969`).
pub struct MossEulerDriver<F> {
    support: DriverSupport,
    sigmas: Vec<f32>,
    forward: F,
}

impl<F> MossEulerDriver<F>
where
    F: Fn(&Tensor, usize, f32, &CancelFlag) -> Result<Tensor>,
{
    pub fn new(sigmas: Vec<f32>, forward: F) -> Self {
        Self {
            support: DriverSupport {
                driver_id: "candle_moss_image_euler",
                arbitrary_slice: true,
                carry_history: true,
                reset_history: true,
                renoise_from_clean: false,
                geometry_change: false,
                read_only_boundary_estimate: true,
                previous_estimate_capture: false,
                uses_global_rng: false,
                refusal_reasons: DriverRefusalReasons {
                    renoise_from_clean:
                        "MOSS-Image Euler driver refuses external Renoise transitions",
                    geometry_change:
                        "MOSS-Image Euler driver does not support mid-trajectory spatial resampling",
                    ..DriverRefusalReasons::default()
                },
            },
            sigmas,
            forward,
        }
    }
}

impl<F> FamilyDriver<Tensor> for MossEulerDriver<F>
where
    F: Fn(&Tensor, usize, f32, &CancelFlag) -> Result<Tensor>,
{
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Tensor>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Tensor>,
    ) -> gen_core::Result<BoundaryState<Tensor>> {
        self.support.validate_slice(&input, state)?;
        let expected_steps = self.sigmas.len().saturating_sub(1);
        if input.total_steps != expected_steps {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: total_steps={} does not match sigma schedule steps={expected_steps}",
                self.support.driver_id, input.total_steps
            )));
        }
        state.enter_slice(&input)?;

        let mut x = match input.state {
            BoundaryState::XSigma(t)
            | BoundaryState::CleanEstimate { latent: t, .. }
            | BoundaryState::RouteSpecific { state: t, .. } => t,
        };

        for step in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            let sigma = self.sigmas[step];
            let next_sigma = self.sigmas[step + 1];
            let info = DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: DriverEvaluationKind::Trajectory,
                sigma,
                next_sigma,
            };
            hooks.observe(&info, &x)?;
            let v = (self.forward)(&x, step, sigma, input.cancel).map_err(gen_core::Error::from)?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);
            let dv = (v * f64::from(next_sigma - sigma))
                .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
            x = x
                .add(&dv)
                .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
            x = hooks.post_step(&info, x)?;
            hooks.materialize(&info, &x)?;
            if input.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }
        state.finish_slice(input.span.end);

        if input.request_boundary_estimate {
            if input.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            let est = if input.span.end < input.total_steps {
                let sigma = self.sigmas[input.span.end];
                let v = (self.forward)(&x, input.span.end, sigma, input.cancel)
                    .map_err(gen_core::Error::from)?;
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                let sv = (v * f64::from(sigma)).map_err(|e| gen_core::Error::Msg(e.to_string()))?;
                x.sub(&sv)
                    .map_err(|e| gen_core::Error::Msg(e.to_string()))?
            } else {
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                x.clone()
            };
            return Ok(BoundaryState::CleanEstimate {
                latent: est,
                source: EstimateSource::EndpointEvaluation,
            });
        }

        Ok(BoundaryState::XSigma(x))
    }
}

/// The same sampling graph is used for generation and in-training previews.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_graph(
    transformer: &Transformer,
    dataspace: Option<&(Tensor, Vec<f32>)>,
    req: &GenerationRequest,
    resolved: &MossImageResolved,
    inputs: MossImageAdvancedInputs<'_>,
    positive: &Conditioning,
    negative: Option<&Conditioning>,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Sampling> {
    let device = transformer.device();
    let count = req.count as usize;
    let images = positive.rows.len() * count;
    let ops = CandleLatentOps;
    let (z, samples) = initial_noise(req, inputs, positive, transformer.channels(), device)?;

    let eval_batch = |input: &Tensor,
                      time: f32,
                      step: usize,
                      total_steps: usize,
                      is_dmd: bool,
                      cancel: &CancelFlag|
     -> Result<Tensor> {
        let pag_scale = resolved.pag_step_scale(step, total_steps);
        let pag_layers = resolved
            .pag
            .as_ref()
            .map_or(&[][..], |pag| pag.step_layers(step, total_steps));
        let mut velocities = Vec::with_capacity(images);
        for image in 0..images {
            check_cancel(cancel)?;
            let row = image / count;
            let latent = input.narrow(0, image, 1)?;
            let ctx = &positive.rows[row];
            let mask = vec![1; ctx.dim(1)?];
            let cond = transformer
                .forward(&latent, time, ctx, &mask, &[], cancel)?
                .to_dtype(DType::F32)?;
            let mut v = cond.clone();
            if !is_dmd {
                if resolved.guidance != 1.0 {
                    let neg = &negative.expect("validated CFG conditioning").rows[row];
                    let active = resolved.skipping_active(step, total_steps);
                    let skip = if active {
                        resolved.uncond_skip_layers.as_slice()
                    } else {
                        &[]
                    };
                    let uncond = transformer
                        .forward(&latent, time, neg, &vec![1; neg.dim(1)?], skip, cancel)?
                        .to_dtype(DType::F32)?;
                    v = guidance_cfg(&ops, &cond, &uncond, resolved.guidance)?;
                }
                if let Some(slg) = &resolved.slg {
                    if !slg.layers.is_empty()
                        && slg.scale > 0.0
                        && resolved.skipping_active(step, total_steps)
                    {
                        let skipped = transformer
                            .forward(&latent, time, ctx, &mask, &slg.layers, cancel)?
                            .to_dtype(DType::F32)?;
                        if resolved.guidance == 1.0 {
                            v = skipped.clone();
                        }
                        v = pag_apply(&ops, &v, &cond, &skipped, slg.scale)?;
                    }
                }
            }
            if pag_scale > 0.0 {
                let perturbed = transformer
                    .forward_with_pag(&latent, time, ctx, &mask, &[], pag_layers, cancel)?
                    .to_dtype(DType::F32)?;
                v = pag_apply(&ops, &v, &cond, &perturbed, pag_scale)?;
            }
            velocities.push(v);
        }
        Ok(Tensor::cat(&velocities, 0)?)
    };

    let (x, sampling) = if let Some((q, codes)) = dataspace {
        let prefix = moss_dmd_step_codes(codes, resolved.steps)?.to_vec();
        let total_steps = prefix.len();
        let receipt = MossImageSamplingReceipt::Dmd {
            step_codes: prefix.clone(),
            mix_matrix_sha256: digest_f32(&tensor_f32(q)?, None),
        };
        let initial = Tensor::zeros(z.shape(), DType::F32, device)?;
        let driver = MossDmdDriver::new(z, q.clone(), prefix, |input, code, step, cancel| {
            eval_batch(input, code, step, total_steps, true, cancel)
        });
        let mut run_state = DriverRunState::new();
        let mut on_materialize = |_info: &DriverStepInfo, _lat: &Tensor| {
            device
                .synchronize()
                .map_err(|e| gen_core::Error::Msg(e.to_string()))
        };
        let mut hooks = StepHooks::new(on_progress);
        hooks.on_materialize = Some(&mut on_materialize);
        let out = run_candle_driver_slices(
            &driver,
            [SliceSpec::full(total_steps)],
            BoundaryState::XSigma(initial),
            total_steps,
            0,
            &req.cancel,
            None,
            &mut run_state,
            &mut hooks,
            |_, state, _| Ok(state),
        )?;
        let x = match out {
            BoundaryState::XSigma(t)
            | BoundaryState::CleanEstimate { latent: t, .. }
            | BoundaryState::RouteSpecific { state: t, .. } => t,
        };
        (x, receipt)
    } else {
        let sigmas = moss_image_schedule(resolved.steps, resolved.shift, resolved.training_shift)?;
        let total_steps = sigmas.len() - 1;
        let receipt = MossImageSamplingReceipt::Euler {
            sigmas: sigmas.clone(),
        };
        let driver = MossEulerDriver::new(sigmas, |input, step, sigma, cancel| {
            eval_batch(input, sigma * 1000.0, step, total_steps, false, cancel)
        });
        let mut run_state = DriverRunState::new();
        let mut on_materialize = |_info: &DriverStepInfo, _lat: &Tensor| {
            device
                .synchronize()
                .map_err(|e| gen_core::Error::Msg(e.to_string()))
        };
        let mut hooks = StepHooks::new(on_progress);
        hooks.on_materialize = Some(&mut on_materialize);
        let out = run_candle_driver_slices(
            &driver,
            [SliceSpec::full(total_steps)],
            BoundaryState::XSigma(z),
            total_steps,
            0,
            &req.cancel,
            None,
            &mut run_state,
            &mut hooks,
            |_, state, _| Ok(state),
        )?;
        let x = match out {
            BoundaryState::XSigma(t)
            | BoundaryState::CleanEstimate { latent: t, .. }
            | BoundaryState::RouteSpecific { state: t, .. } => t,
        };
        (x, receipt)
    };
    check_cancel(&req.cancel)?;
    if tensor_f32(&x)?.iter().any(|v| !v.is_finite()) {
        return Err(CandleError::Msg("MOSS nonfinite sampled latent".into()));
    }
    Ok(Sampling {
        latents: x,
        samples,
        sampling,
    })
}
pub(crate) fn dmd_input(z: &Tensor, q: &Tensor, x0: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = z.dims4()?;
    let flat = x0.reshape((b, c, h * w))?.transpose(1, 2)?;
    let mixed = candle_gen::candle_nn::Linear::new(q.clone(), None).forward(&flat)?;
    Ok(z.add(&mixed.transpose(1, 2)?.reshape((b, c, h, w))?)?)
}
use candle_gen::candle_nn::Module;
pub(crate) fn decode(
    vae: &Flux2Vae,
    latents: &Tensor,
    req: &GenerationRequest,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    check_cancel(&req.cancel)?;
    on_progress(Progress::Decoding);
    let mut images = Vec::new();
    for i in 0..latents.dim(0)? {
        check_cancel(&req.cancel)?;
        let pixels = vae
            .decode_packed(&latents.narrow(0, i, 1)?.to_dtype(DType::F32)?)?
            .clamp(-1.0, 1.0)?;
        let pixels = pixels
            .squeeze(0)?
            .permute((1, 2, 0))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if pixels.iter().any(|v| !v.is_finite()) {
            return Err(CandleError::Msg("MOSS nonfinite decoded pixels".into()));
        }
        let pixels = pixels
            .iter()
            .map(|v| ((v + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8)
            .collect();
        check_cancel(&req.cancel)?;
        images.push(Image {
            width: req.width,
            height: req.height,
            pixels,
        });
    }
    check_cancel(&req.cancel)?;
    Ok(images)
}
pub(crate) fn receipts(assets: &[Asset]) -> gen_core::Result<Vec<MossImageAssetReceipt>> {
    assets.iter().map(Asset::receipt).collect()
}

fn initial_noise(
    req: &GenerationRequest,
    inputs: MossImageAdvancedInputs<'_>,
    positive: &Conditioning,
    channels: usize,
    device: &Device,
) -> Result<(Tensor, Vec<MossImageSampleReceipt>)> {
    let count = req.count as usize;
    let images = positive.rows.len() * count;
    let h = req.height as usize / 16;
    let w = req.width as usize / 16;

    let image_len = channels * h * w;
    let mut noises = Vec::with_capacity(images * image_len);
    let mut samples = Vec::with_capacity(images);
    let initial_seed = req.seed.unwrap_or_else(gen_core::default_seed);
    for i in 0..images {
        check_cancel(&req.cancel)?;
        let (seed, noise) = if let Some(n) = inputs.noise {
            (None, n[i * image_len..(i + 1) * image_len].to_vec())
        } else {
            let seed = initial_seed.wrapping_add(i as u64);
            let mut rng = StdRng::seed_from_u64(seed);
            (
                Some(seed),
                candle_gen::seeded_normal_vec(&mut rng, image_len),
            )
        };
        samples.push(MossImageSampleReceipt {
            seed,
            noise_sha256: digest_f32(&noise, None),
        });
        noises.extend(noise);
    }
    let z = Tensor::from_vec(noises, (images, channels, h, w), device)?;
    Ok((z, samples))
}
pub(crate) fn sample_edit(
    renderer: &Renderer,
    req: &GenerationRequest,
    resolved: &MossImageResolved,
    inputs: gen_core::moss_image_edit::MossImageEditInputs<'_>,
    conditioning: (&Conditioning, Option<&Conditioning>),
    references: &[Tensor],
    progress: &mut dyn FnMut(Progress),
) -> Result<Sampling> {
    let (positive, negative) = conditioning;
    let (noise, samples) = initial_noise(
        req,
        inputs.advanced,
        positive,
        renderer.transformer.channels(),
        renderer.transformer.device(),
    )?;
    let latents = crate::edit::EditSampling {
        transformer: &renderer.transformer,
        resolved,
        positive: &positive.rows,
        negative: negative.map_or(&[], |n| n.rows.as_slice()),
        references,
        use_cache: inputs.use_kv_cache,
        images_per_prompt: req.count as usize,
    }
    .run(&noise, &req.cancel, progress)?;
    if tensor_f32(&latents)?.iter().any(|v| !v.is_finite()) {
        return Err(CandleError::Msg("MOSS nonfinite edit latent".into()));
    }
    Ok(Sampling {
        latents,
        samples,
        sampling: MossImageSamplingReceipt::Euler {
            sigmas: moss_image_schedule(resolved.steps, resolved.shift, resolved.training_shift)?,
        },
    })
}

#[cfg(test)]
mod driver_conformance_tests {
    use super::*;
    use candle_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

    fn tensors_bit_equal(a: &Tensor, b: &Tensor) -> bool {
        let va = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vb = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        va.len() == vb.len()
            && va
                .iter()
                .zip(vb.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn candle_moss_dmd_and_euler_drivers_pass_tier0_conformance() {
        let q = Tensor::from_vec(vec![1.0f32, 0.25, -0.1, 0.9], (2, 2), &Device::Cpu).unwrap();
        let noise = Tensor::from_vec(
            vec![0.5f32, -0.25, 0.75, 0.1, -0.3, 0.2, 0.4, -0.6],
            (1, 2, 2, 2),
            &Device::Cpu,
        )
        .unwrap();
        let codes = vec![1000.0f32, 750.0, 500.0, 250.0];
        let dmd_report = verify_driver_conformance(ConformanceCase {
            total_steps: 4,
            split_at: 2,
            seed: 13,
            base_geometry: (512, 512),
            alt_geometry: (1024, 1024),
            make_driver: {
                let noise = noise.clone();
                let q = q.clone();
                let codes = codes.clone();
                move || {
                    Box::new(MossDmdDriver::new(
                        noise.clone(),
                        q.clone(),
                        codes.clone(),
                        |input, code, step, _cancel| {
                            let scale = 0.1 + 0.0001 * f64::from(code) + 0.02 * step as f64;
                            Ok((input * scale)?)
                        },
                    ))
                }
            },
            make_initial: || Tensor::zeros((1, 2, 2, 2), DType::F32, &Device::Cpu).unwrap(),
            latents_equal: tensors_bit_equal,
        })
        .expect("Candle MossDmdDriver must pass Tier 0 conformance");
        assert!(dmd_report.verified_inert_split);
        assert!(dmd_report.verified_cancellation);
        assert_eq!(dmd_report.unsplit_evaluations, 4);

        let sigmas = moss_image_schedule(4, 1.0, 1.0).unwrap();
        let euler_report = verify_driver_conformance(ConformanceCase {
            total_steps: 4,
            split_at: 2,
            seed: 13,
            base_geometry: (512, 512),
            alt_geometry: (1024, 1024),
            make_driver: move || {
                Box::new(MossEulerDriver::new(
                    sigmas.clone(),
                    |latent, step, sigma, _cancel| {
                        let scale = 0.2 + 0.1 * f64::from(sigma) + 0.03 * step as f64;
                        Ok((latent * scale)?)
                    },
                ))
            },
            make_initial: move || noise.clone(),
            latents_equal: tensors_bit_equal,
        })
        .expect("Candle MossEulerDriver must pass Tier 0 conformance");
        assert!(euler_report.verified_inert_split);
        assert!(euler_report.verified_cancellation);
        assert_eq!(euler_report.unsplit_evaluations, 4);
    }
}
