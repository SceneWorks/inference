//! Prompt-to-image Mage-Flow RL pipeline.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_transformers::models::z_image::sampling::postprocess_image;

use crate::config::LATENT_CHANNELS;
use crate::rope::{ImgShape, PackLayout};
use crate::scheduler;
use crate::{MageComponentDirs, MageConfig, MageTextEncoder, MageTransformer, MageVae};
use candle_gen::driver::{
    run_candle_driver_slices, DriverEvaluationKind, DriverHistoryPolicy, DriverRefusalReasons,
    DriverRunState, DriverStepInfo, DriverSupport, FamilyDriver, SliceInput, SliceSpec,
    SliceTransition, StepHooks,
};
use candle_gen::gen_core::{
    AdapterSpec, BoundaryState, CancelFlag, GenerationMemory, Progress, Quant,
};

pub(crate) struct MageEncoded {
    positive: Tensor,
    negative: Option<Tensor>,
}

pub(crate) struct MageHeavy {
    pub(crate) transformer: MageTransformer,
    pub(crate) vae: MageVae,
}

pub struct MagePipeline {
    unobserved_adapter_count: usize,
    loaded_quant: crate::quant::LoadedQuant,
    text: MageTextEncoder,
    transformer: MageTransformer,
    vae: MageVae,
}

impl MagePipeline {
    pub fn load(root: &Path, device: &Device) -> Result<Self> {
        Self::load_with_quant(root, None, device)
    }

    pub fn load_with_quant(root: &Path, quant: Option<Quant>, device: &Device) -> Result<Self> {
        Self::load_components(&MageComponentDirs::flat(root), quant, device, &[])
    }

    pub(crate) fn load_components(
        dirs: &MageComponentDirs,
        quant: Option<Quant>,
        device: &Device,
        adapters: &[AdapterSpec],
    ) -> Result<Self> {
        let cfg_text = candle_gen::gen_core::__model_input_choose!(
            std::fs::read_to_string(dirs.transformer.join("config.json")),
            candle_gen::gen_core::model_input_proof::read_configuration_to_string_for(
                dirs.transformer.join("config.json"),
                "candle-gen-mage::pipeline::MagePipeline::load_components",
            )
        )?;
        let cfg = MageConfig::from_json(&cfg_text)?;

        let mut transformer =
            MageTransformer::load_with_quant(&dirs.transformer, &cfg, quant, device)?;
        candle_gen::quant::install_dotted_adapters("mage", adapters, device, |visitor| {
            transformer.visit_adaptable_mut(visitor)
        })
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let text =
            MageTextEncoder::load_component_with_quant(&dirs.text_encoder, false, quant, device)?;
        let loaded_quant = crate::quant::LoadedQuant::from_loaded(
            quant,
            transformer.has_packed_source(),
            text.has_packed_source(),
        );
        Ok(Self {
            unobserved_adapter_count: adapters.len(),
            loaded_quant,
            text,
            transformer,
            vae: MageVae::load(&dirs.vae, device)?,
        })
    }

    pub(crate) fn load_text(
        dirs: &MageComponentDirs,
        quant: Option<Quant>,
        device: &Device,
    ) -> Result<MageTextEncoder> {
        MageTextEncoder::load_component_with_quant(&dirs.text_encoder, false, quant, device)
    }

    pub(crate) fn load_heavy(
        dirs: &MageComponentDirs,
        quant: Option<Quant>,
        device: &Device,
        stream_transformer_blocks: bool,
        cancel: &candle_gen::gen_core::CancelFlag,
        adapters: &[AdapterSpec],
    ) -> Result<MageHeavy> {
        let cfg_text = candle_gen::gen_core::__model_input_choose!(
            std::fs::read_to_string(dirs.transformer.join("config.json")),
            candle_gen::gen_core::model_input_proof::read_configuration_to_string_for(
                dirs.transformer.join("config.json"),
                "candle-gen-mage::pipeline::MagePipeline::load_heavy",
            )
        )?;
        let cfg = MageConfig::from_json(&cfg_text)?;

        if stream_transformer_blocks && !adapters.is_empty() {
            candle_core::bail!(
                "Mage adapters require resident transformer blocks; disable block streaming"
            );
        }
        let mut transformer = if stream_transformer_blocks {
            MageTransformer::load_block_streamed(&dirs.transformer, &cfg, quant, device, cancel)?
        } else {
            MageTransformer::load_with_quant(&dirs.transformer, &cfg, quant, device)?
        };
        candle_gen::quant::install_dotted_adapters("mage", adapters, device, |visitor| {
            transformer.visit_adaptable_mut(visitor)
        })
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        Ok(MageHeavy {
            transformer,
            vae: MageVae::load(&dirs.vae, device)?,
        })
    }

    pub(crate) fn encode_prompt(
        text: &MageTextEncoder,
        prompt: &str,
        negative_prompt: &str,
        guidance: f32,
    ) -> Result<MageEncoded> {
        Ok(MageEncoded {
            positive: text.encode(prompt)?,
            negative: (guidance != 1.0)
                .then(|| text.encode(negative_prompt))
                .transpose()?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    ///
    /// Complete this standalone route with observations from this same attempt.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_execution_report(
        &self,
        prompt: &str,
        negative_prompt: &str,
        width: u32,
        height: u32,
        steps: usize,
        guidance: f32,
        seed: u64,
        on_progress: &mut dyn FnMut(candle_gen::gen_core::Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        let mut report_progress =
            |event| candle_gen::execution::outside_attempt(|| on_progress(event));
        crate::quant::capture_direct_with_loaded_quant(
            &crate::descriptor(),
            &candle_gen::gen_core::CancelFlag::default(),
            self.loaded_quant,
            self.unobserved_adapter_count,
            "MagePipeline",
            "mage/pipeline/generate",
            || {
                let image = self
                    .generate(
                        prompt,
                        negative_prompt,
                        width,
                        height,
                        steps,
                        guidance,
                        seed,
                        &mut report_progress,
                    )
                    .map_err(candle_gen::CandleError::from)?;
                Ok(candle_gen::gen_core::GenerationOutput::Images(vec![image]))
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: &str,
        negative_prompt: &str,
        width: u32,
        height: u32,
        steps: usize,
        guidance: f32,
        seed: u64,
        on_progress: &mut dyn FnMut(candle_gen::gen_core::Progress),
    ) -> Result<candle_gen::gen_core::Image> {
        self.generate_with_memory(
            prompt,
            negative_prompt,
            width,
            height,
            steps,
            guidance,
            seed,
            None,
            None,
            &candle_gen::gen_core::CancelFlag::default(),
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_with_memory(
        &self,
        prompt: &str,
        negative_prompt: &str,
        width: u32,
        height: u32,
        steps: usize,
        guidance: f32,
        seed: u64,
        scheduler_shift: Option<f32>,
        memory: Option<GenerationMemory>,
        cancel: &candle_gen::gen_core::CancelFlag,
        on_progress: &mut dyn FnMut(candle_gen::gen_core::Progress),
    ) -> Result<candle_gen::gen_core::Image> {
        let encoded = candle_gen::gen_core::with_prompt_encoding(on_progress, || {
            Self::encode_prompt(&self.text, prompt, negative_prompt, guidance)
        })?;
        Self::sample(
            &self.transformer,
            &self.vae,
            encoded,
            width,
            height,
            steps,
            guidance,
            seed,
            scheduler_shift,
            memory,
            cancel,
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample(
        transformer: &MageTransformer,
        vae: &MageVae,
        encoded: MageEncoded,
        width: u32,
        height: u32,
        steps: usize,
        guidance: f32,
        seed: u64,
        scheduler_shift: Option<f32>,
        memory: Option<GenerationMemory>,
        cancel: &candle_gen::gen_core::CancelFlag,
        on_progress: &mut dyn FnMut(candle_gen::gen_core::Progress),
    ) -> Result<candle_gen::gen_core::Image> {
        let gh = height as usize / 16;
        let gw = width as usize / 16;
        let MageEncoded { positive, negative } = encoded;
        let device = positive.device().clone();
        let use_cfg = negative.is_some();
        candle_gen::execution::f32_policy(
            "mage/effective_cfg",
            if use_cfg { guidance } else { 1.0 },
        );
        candle_gen::execution::policy("mage/pipeline/use_cfg", format!("{use_cfg:?}").into());
        let base = PackLayout::generation(vec![ImgShape::latent(gh, gw)], vec![positive.dim(1)?])?;
        let (layout, text) = if let Some(negative) = &negative {
            (
                base.fused_cfg(&[negative.dim(1)?])?,
                Tensor::cat(&[positive.clone(), negative.clone()], 1)?,
            )
        } else {
            (base, positive)
        };

        let noise =
            crate::latent::watermarked_noise(LATENT_CHANNELS, gh, gw, seed, DType::BF16, &device)?;
        let initial_tokens = noise
            .permute((0, 2, 3, 1))?
            .reshape((1, gh * gw, LATENT_CHANNELS))?;
        let native = scheduler::sigmas(steps)?;
        let ladder = candle_gen::resolve_flow_schedule_with_shift(
            None,
            scheduler_shift,
            6.0f32.ln(),
            steps,
            &native,
        )
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let attention_budget = memory
            .filter(|memory| memory.chunk_attention)
            .and_then(|memory| memory.attention_chunk_size)
            .unwrap_or(candle_gen::ATTN_SCORES_BUDGET as u32)
            as usize;
        candle_gen::execution::policy(
            "mage/pipeline/attention_budget",
            format!("{attention_budget:?}").into(),
        );
        let transformer_window = memory
            .filter(|memory| memory.stream_transformer_blocks)
            .and_then(|memory| memory.transformer_window_size)
            .map(|window| window as usize)
            .unwrap_or(crate::memory_strategy::DEFAULT_TRANSFORMER_WINDOW);
        candle_gen::execution::policy(
            "mage/pipeline/transformer_window",
            format!("{transformer_window:?}").into(),
        );
        candle_gen::execution::schedule("mage-euler", &ladder, "flow-velocity");
        candle_gen::execution::runtime(initial_tokens.device(), initial_tokens.dtype());
        let tokens = run_mage_candle_driver(
            "mage_candle_generation_driver",
            transformer,
            initial_tokens,
            gh * gw,
            None,
            &text,
            &layout,
            use_cfg,
            guidance,
            &ladder,
            attention_budget,
            transformer_window,
            seed,
            cancel,
            on_progress,
        )?;
        decode_tokens_to_image(vae, &tokens, gh, gw, cancel, "mage", on_progress)
    }
}

/// Forward-evaluation seam for [`MageCandleDriver`] (`PIPE-7`, `DEC-13`).
pub trait MageCandleForward {
    /// Whether this forward evaluates fused `[cond, uncond]` CFG branches along axis 1.
    fn uses_cfg(&self) -> bool;

    /// Predict transformer velocity for `sequence` at noise level `sigma` and outer step `step`.
    fn forward_sequence(
        &self,
        sequence: &Tensor,
        sigma: f32,
        step: usize,
        cancel: &CancelFlag,
    ) -> Result<Tensor>;
}

struct TransformerMageCandleForward<'a> {
    transformer: &'a MageTransformer,
    text: &'a Tensor,
    layout: &'a PackLayout,
    use_cfg: bool,
    attention_budget: usize,
    transformer_window: usize,
}

impl MageCandleForward for TransformerMageCandleForward<'_> {
    fn uses_cfg(&self) -> bool {
        self.use_cfg
    }

    fn forward_sequence(
        &self,
        sequence: &Tensor,
        sigma: f32,
        _step: usize,
        cancel: &CancelFlag,
    ) -> Result<Tensor> {
        let device = sequence.device();
        let sigma_tensor = if self.use_cfg {
            Tensor::new(&[sigma, sigma], device)?
        } else {
            Tensor::new(&[sigma], device)?
        };
        self.transformer.forward_with_memory(
            sequence,
            self.text,
            &sigma_tensor,
            self.layout,
            self.attention_budget,
            self.transformer_window,
            cancel,
        )
    }
}

struct MageCandleDriverHistory {
    total_steps: usize,
    sigmas_bits: Vec<u32>,
    expected_next_step: usize,
}

/// Typed [`FamilyDriver`] for Candle Mage-Flow rectified-flow Euler denoising across T2I and
/// target-first clean-reference Edit sequences (`PIPE-7`, `DEC-13`).
pub struct MageCandleDriver<F> {
    sigmas: Vec<f32>,
    sigmas_bits: Vec<u32>,
    cfg: f32,
    target_tokens: usize,
    reference_tokens: Option<Tensor>,
    forward: F,
    support: DriverSupport,
}

impl<F: MageCandleForward> MageCandleDriver<F> {
    pub fn build_support(driver_id: &'static str) -> DriverSupport {
        DriverSupport {
            driver_id,
            arbitrary_slice: true,
            carry_history: true,
            reset_history: true,
            renoise_from_clean: false,
            geometry_change: false,
            read_only_boundary_estimate: false,
            previous_estimate_capture: false,
            uses_global_rng: false,
            refusal_reasons: DriverRefusalReasons {
                renoise_from_clean:
                    "native Candle Mage flow driver does not re-noise from clean estimates",
                geometry_change:
                    "native Candle Mage flow driver does not support geometry-changing phase boundaries",
                read_only_boundary_estimate:
                    "native Candle Mage flow driver does not expose read-only boundary clean estimates",
                ..DriverRefusalReasons::default()
            },
        }
    }

    pub fn new(
        driver_id: &'static str,
        sigmas: Vec<f32>,
        cfg: f32,
        target_tokens: usize,
        reference_tokens: Option<Tensor>,
        forward: F,
    ) -> Result<Self> {
        if sigmas.len() < 2 {
            candle_core::bail!("mage_flow: scheduler needs a terminal sigma");
        }
        if target_tokens == 0 {
            candle_core::bail!("mage_flow: target token count must be positive");
        }
        let sigmas_bits = sigmas.iter().map(|s| s.to_bits()).collect();
        Ok(Self {
            sigmas,
            sigmas_bits,
            cfg,
            target_tokens,
            reference_tokens,
            forward,
            support: Self::build_support(driver_id),
        })
    }
}

impl<F: MageCandleForward> FamilyDriver<Tensor> for MageCandleDriver<F> {
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
        let total_steps = self.sigmas.len() - 1;
        if input.total_steps != total_steps || input.span.end > total_steps {
            return Err(candle_gen::gen_core::Error::Unsupported(format!(
                "{}: slice {:?} / total_steps={} disagrees with Mage schedule length {total_steps}",
                self.support.driver_id, input.span, input.total_steps
            )));
        }

        let carrying = input.transition == SliceTransition::Continue
            && input.history == DriverHistoryPolicy::Carry;
        if carrying {
            let Some(existing) = state.history_ref::<MageCandleDriverHistory>() else {
                return Err(candle_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry requested without prior Mage driver history in DriverRunState",
                    self.support.driver_id
                )));
            };
            if existing.total_steps != input.total_steps || existing.sigmas_bits != self.sigmas_bits
            {
                return Err(candle_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry refused across incompatible Mage sigma schedule",
                    self.support.driver_id
                )));
            }
            if existing.expected_next_step != input.span.start {
                return Err(candle_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry refused at non-contiguous step {} (expected {})",
                    self.support.driver_id, input.span.start, existing.expected_next_step
                )));
            }
        }

        state.enter_slice(&input)?;
        if !carrying {
            state.set_history(MageCandleDriverHistory {
                total_steps,
                sigmas_bits: self.sigmas_bits.clone(),
                expected_next_step: input.span.start,
            });
        }

        let mut target = match input.state {
            BoundaryState::XSigma(x)
            | BoundaryState::CleanEstimate { latent: x, .. }
            | BoundaryState::RouteSpecific { state: x, .. } => x,
        };
        let got_tokens = target
            .dim(1)
            .map_err(|e| candle_gen::gen_core::Error::Msg(e.to_string()))?;
        if got_tokens != self.target_tokens {
            return Err(candle_gen::gen_core::Error::Msg(format!(
                "{}: expected target tokens length {}, got {got_tokens}",
                self.support.driver_id, self.target_tokens
            )));
        }

        let use_cfg = self.forward.uses_cfg();
        for step in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(candle_gen::gen_core::Error::Canceled);
            }
            let info = DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: DriverEvaluationKind::Trajectory,
                sigma: self.sigmas[step],
                next_sigma: self.sigmas[step + 1],
            };
            hooks.observe(&info, &target)?;

            let sequence = if let Some(refs) = &self.reference_tokens {
                Tensor::cat(&[target.clone(), refs.clone()], 1)
                    .map_err(|e| candle_gen::gen_core::Error::Msg(e.to_string()))?
            } else {
                target.clone()
            };
            let seq_tokens = sequence
                .dim(1)
                .map_err(|e| candle_gen::gen_core::Error::Msg(e.to_string()))?;
            let model_input = if use_cfg {
                Tensor::cat(&[sequence.clone(), sequence], 1)
                    .map_err(|e| candle_gen::gen_core::Error::Msg(e.to_string()))?
            } else {
                sequence
            };

            let raw_out = self
                .forward
                .forward_sequence(&model_input, self.sigmas[step], step, input.cancel)
                .map_err(|e| {
                    if input.cancel.is_cancelled() {
                        candle_gen::gen_core::Error::Canceled
                    } else {
                        candle_gen::gen_core::Error::Msg(e.to_string())
                    }
                })?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);

            let velocity = if use_cfg {
                let cond = raw_out
                    .narrow(1, 0, seq_tokens)
                    .map_err(|e| candle_gen::gen_core::Error::Msg(e.to_string()))?;
                let unc = raw_out
                    .narrow(1, seq_tokens, seq_tokens)
                    .map_err(|e| candle_gen::gen_core::Error::Msg(e.to_string()))?;
                (&unc
                    + ((&cond - &unc)
                        .map_err(|e| candle_gen::gen_core::Error::Msg(e.to_string()))?
                        * self.cfg as f64)
                        .map_err(|e| candle_gen::gen_core::Error::Msg(e.to_string()))?)
                .map_err(|e| candle_gen::gen_core::Error::Msg(e.to_string()))?
            } else {
                raw_out
            };

            let target_velocity = if self.reference_tokens.is_some() {
                velocity
                    .narrow(1, 0, self.target_tokens)
                    .map_err(|e| candle_gen::gen_core::Error::Msg(e.to_string()))?
            } else {
                velocity
            };

            target = scheduler::euler_step(
                &target,
                &target_velocity,
                self.sigmas[step],
                self.sigmas[step + 1],
            )
            .map_err(|e| candle_gen::gen_core::Error::Msg(e.to_string()))?
            .to_dtype(DType::BF16)
            .map_err(|e| candle_gen::gen_core::Error::Msg(e.to_string()))?;
            target = hooks.post_step(&info, target)?;
            hooks.materialize(&info, &target)?;
            if let Some(history) = state.history_mut::<MageCandleDriverHistory>() {
                history.expected_next_step = step + 1;
            }
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }

        state.finish_slice(input.span.end);
        Ok(BoundaryState::XSigma(target))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_mage_candle_driver(
    driver_id: &'static str,
    transformer: &MageTransformer,
    initial_target: Tensor,
    target_tokens: usize,
    reference_tokens: Option<Tensor>,
    text: &Tensor,
    layout: &PackLayout,
    use_cfg: bool,
    guidance: f32,
    sigmas: &[f32],
    attention_budget: usize,
    transformer_window: usize,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Tensor> {
    let forward = TransformerMageCandleForward {
        transformer,
        text,
        layout,
        use_cfg,
        attention_budget,
        transformer_window,
    };
    let driver = MageCandleDriver::new(
        driver_id,
        sigmas.to_vec(),
        guidance,
        target_tokens,
        reference_tokens,
        forward,
    )?;
    let total_steps = sigmas.len() - 1;
    let mut state = DriverRunState::new();
    let mut hooks = StepHooks::new(on_progress);
    let out = run_candle_driver_slices(
        &driver,
        [SliceSpec::full(total_steps)],
        BoundaryState::XSigma(initial_target),
        total_steps,
        seed,
        cancel,
        None,
        &mut state,
        &mut hooks,
        |_, out, _| Ok(out),
    )
    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let BoundaryState::XSigma(final_target) = out else {
        unreachable!("MageCandleDriver always emits BoundaryState::XSigma");
    };
    Ok(final_target)
}

pub(crate) fn decode_tokens_to_image(
    vae: &MageVae,
    target: &Tensor,
    gh: usize,
    gw: usize,
    cancel: &CancelFlag,
    label: &str,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<candle_gen::gen_core::Image> {
    crate::begin_decode(cancel, label, on_progress)?;
    let latent = target
        .reshape((1, gh, gw, LATENT_CHANNELS))?
        .permute((0, 3, 1, 2))?;
    let decoded = vae.decode(&latent)?.to_dtype(DType::F32)?;
    if cancel.is_cancelled() {
        candle_core::bail!("{label} canceled");
    }
    let image = postprocess_image(&decoded)?.i(0)?.to_device(&Device::Cpu)?;
    let (c, h, w) = image.dims3()?;
    if c != 3 {
        candle_core::bail!("{label}: VAE returned {c} channels");
    }
    Ok(candle_gen::gen_core::Image {
        width: w as u32,
        height: h as u32,
        pixels: image.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?,
    })
}

#[cfg(test)]
mod driver_tests {
    use super::*;
    use candle_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};
    use std::cell::RefCell;
    use std::rc::Rc;

    type SeenLog = Rc<RefCell<Vec<(usize, Vec<usize>, f32, Tensor)>>>;

    struct MockMageCandleForward {
        use_cfg: bool,
        target_tokens: usize,
        seen: SeenLog,
    }

    impl MageCandleForward for MockMageCandleForward {
        fn uses_cfg(&self) -> bool {
            self.use_cfg
        }

        fn forward_sequence(
            &self,
            sequence: &Tensor,
            sigma: f32,
            step: usize,
            _cancel: &CancelFlag,
        ) -> Result<Tensor> {
            self.seen
                .borrow_mut()
                .push((step, sequence.dims().to_vec(), sigma, sequence.clone()));
            let dtype = sequence.dtype();
            let seq_f32 = sequence.to_dtype(DType::F32)?;
            let out = if self.use_cfg {
                let half = sequence.dim(1)? / 2;
                let cond = seq_f32.narrow(1, 0, half)?;
                let unc = seq_f32.narrow(1, half, half)?;
                let c = (cond.affine(-0.5 - 0.1 * sigma as f64, 0.05 * (step + 1) as f64))?;
                let u = (unc.affine(-0.25, -0.02 * self.target_tokens as f64))?;
                Tensor::cat(&[&c, &u], 1)?
            } else {
                seq_f32.affine(-0.4 - 0.1 * sigma as f64, 0.03 * (step + 1) as f64)?
            };
            out.to_dtype(dtype)
        }
    }

    fn tensor_bf16_bit_equal(a: &Tensor, b: &Tensor) -> bool {
        if a.dims() != b.dims() {
            return false;
        }
        let va = a
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let vb = b
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        va.len() == vb.len() && va.iter().zip(vb).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn mage_candle_driver_passes_conformance_t2i_turbo_and_edit() {
        let dev = Device::Cpu;
        let target_tokens = 4usize;
        let ref_tokens = Tensor::from_vec(
            vec![9.0f32, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0],
            (1, 4, 2),
            &dev,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let init_target = Tensor::from_vec(
            vec![0.5f32, -0.25, 1.0, -1.5, 0.25, 0.75, -0.5, 0.125],
            (1, 4, 2),
            &dev,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

        for (name, cfg, refs) in [
            ("mage_candle_t2i_cfg", 5.0f32, None),
            ("mage_candle_t2i_turbo_nocfg", 1.0f32, None),
            ("mage_candle_edit_cfg", 5.0f32, Some(ref_tokens)),
        ] {
            let sigmas = scheduler::sigmas(4).unwrap();
            let report = verify_driver_conformance(ConformanceCase {
                total_steps: 4,
                split_at: 2,
                seed: 77,
                base_geometry: (64, 64),
                alt_geometry: (128, 128),
                make_driver: || {
                    Box::new(
                        MageCandleDriver::new(
                            name,
                            sigmas.clone(),
                            cfg,
                            target_tokens,
                            refs.clone(),
                            MockMageCandleForward {
                                use_cfg: cfg != 1.0,
                                target_tokens,
                                seen: Rc::new(RefCell::new(Vec::new())),
                            },
                        )
                        .unwrap(),
                    )
                },
                make_initial: || init_target.clone(),
                latents_equal: tensor_bf16_bit_equal,
            })
            .unwrap_or_else(|err| panic!("{name} conformance failed: {err}"));
            assert_eq!(report.total_steps, 4);
            assert!(report.verified_inert_split);
            assert!(report.verified_cancellation);
        }
    }

    #[test]
    fn mage_candle_edit_driver_reconcatenates_clean_references_and_switches_cfg() {
        let dev = Device::Cpu;
        let target_tokens = 3usize;
        let init = Tensor::from_vec(vec![0.5f32, -0.5, 0.25, -0.25, 1.0, -1.0], (1, 3, 2), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let refs = Tensor::from_vec(vec![10.0f32, 11.0, 20.0, 21.0], (1, 2, 2), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let sigmas = scheduler::sigmas(4).unwrap();
        let cancel = CancelFlag::default();

        let seen_cfg: SeenLog = Rc::new(RefCell::new(Vec::new()));
        let drv_cfg = MageCandleDriver::new(
            "mage_candle_edit_invariant",
            sigmas.clone(),
            5.0,
            target_tokens,
            Some(refs.clone()),
            MockMageCandleForward {
                use_cfg: true,
                target_tokens,
                seen: seen_cfg.clone(),
            },
        )
        .unwrap();
        let mut st = DriverRunState::new();
        let mut progress = Vec::new();
        let mut on_progress = |p: Progress| progress.push(p);
        let mut hooks = StepHooks::new(&mut on_progress);
        let out = run_candle_driver_slices(
            &drv_cfg,
            [SliceSpec::full(4)],
            BoundaryState::XSigma(init.clone()),
            4,
            0,
            &cancel,
            None,
            &mut st,
            &mut hooks,
            |_, out, _| Ok(out),
        )
        .unwrap();
        let BoundaryState::XSigma(final_target) = out else {
            panic!("expected XSigma");
        };
        assert_eq!(final_target.dims(), &[1, 3, 2]);
        assert_eq!(final_target.dtype(), DType::BF16);
        assert_eq!(progress.len(), 4);

        for (step, dims, _, seq) in seen_cfg.borrow().iter() {
            assert_eq!(dims.as_slice(), &[1, 10, 2], "step {step} fused CFG edit");
            let cond_t = seq.narrow(1, 0, 3).unwrap();
            let cond_r = seq.narrow(1, 3, 2).unwrap();
            let unc_t = seq.narrow(1, 5, 3).unwrap();
            let unc_r = seq.narrow(1, 8, 2).unwrap();
            assert!(tensor_bf16_bit_equal(&cond_r, &refs));
            assert!(tensor_bf16_bit_equal(&unc_r, &refs));
            assert!(tensor_bf16_bit_equal(&cond_t, &unc_t));
        }

        let seen_turbo: SeenLog = Rc::new(RefCell::new(Vec::new()));
        let drv_turbo = MageCandleDriver::new(
            "mage_candle_turbo_invariant",
            sigmas,
            1.0,
            target_tokens,
            Some(refs),
            MockMageCandleForward {
                use_cfg: false,
                target_tokens,
                seen: seen_turbo.clone(),
            },
        )
        .unwrap();
        let mut st_t = DriverRunState::new();
        let mut noop = |_: Progress| {};
        let mut hooks_t = StepHooks::new(&mut noop);
        let _ = run_candle_driver_slices(
            &drv_turbo,
            [SliceSpec::full(4)],
            BoundaryState::XSigma(init),
            4,
            0,
            &cancel,
            None,
            &mut st_t,
            &mut hooks_t,
            |_, out, _| Ok(out),
        )
        .unwrap();
        for (step, dims, _, _) in seen_turbo.borrow().iter() {
            assert_eq!(
                dims.as_slice(),
                &[1, 5, 2],
                "step {step} cfg==1.0 must not duplicate uncond branch"
            );
        }
    }

    #[test]
    fn mage_candle_initial_noise_uses_watermarked_noise() {
        let dev = Device::Cpu;
        let a =
            crate::latent::watermarked_noise(LATENT_CHANNELS, 2, 2, 42, DType::BF16, &dev).unwrap();
        let b =
            crate::latent::watermarked_noise(LATENT_CHANNELS, 2, 2, 42, DType::BF16, &dev).unwrap();
        let c =
            crate::latent::watermarked_noise(LATENT_CHANNELS, 2, 2, 43, DType::BF16, &dev).unwrap();
        assert_eq!(a.dims(), &[1, LATENT_CHANNELS, 2, 2]);
        assert!(tensor_bf16_bit_equal(&a, &b));
        assert!(!tensor_bf16_bit_equal(&a, &c));
    }
}
