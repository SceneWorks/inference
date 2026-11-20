//! The candle Chroma **txt2img** pipeline — the T5-XXL prompt encode → the Chroma DiT (true-CFG
//! flow-match Euler) → the FLUX 16-ch AutoencoderKL, driven through the backend-neutral
//! [`gen_core::Generator`] contract and parity-matched to the macOS `mlx-gen-chroma` provider.
//!
//! Parity choices (grounded in the mlx `model.rs`):
//! - **Packing**: noise is drawn in the VAE's /8 latent `[1, 16, h/8, w/8]`, then 2×2-packed to
//!   `[1, Si, 64]` exactly as candle FLUX's `State::new` (so the row-major `img_ids` line up). The
//!   denoised packed latent is `flux::sampling::unpack`ed back to `[1, 16, h/8, w/8]` for the VAE.
//! - **Sigmas**: Chroma's scheduler is `use_dynamic_shifting=false`. HD/Flash use the static-shift
//!   `linspace(1, 1/N, N)` (`σ' = shift·σ/(1+(shift-1)·σ)`); Base uses the beta-spaced schedule
//!   ([`crate::beta`]). NOT FLUX's resolution-dependent exp-shift.
//! - **True CFG**: `pred = neg + g·(pos − neg)`; at `g ≤ 1.0` the negative branch is skipped and
//!   `pred = pos` exactly (`chroma1_flash` is distilled to single-forward), a 2× per-step saving.
//! - **Deterministic seeding (sc-3673)**: initial noise from a fixed-algorithm CPU RNG (`StdRng`,
//!   ChaCha) seeded by `seed`, moved to the device — launch-portable per seed.
//! - **Step-invariants once per step/branch**: the Approximator modulation table (`pooled_temb`,
//!   timestep-only) is computed once per step and shared across both CFG branches; the RoPE table is
//!   built once per branch.
//!
//! The DiT is loaded at **f32**. The dense T5-XXL and VAE stay at their native **bf16** width; their
//! outputs are promoted at the existing F32 activation boundaries. Components are cached by the
//! generator across `generate` calls.

use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{self, AdapterSpec, GenerationRequest, Image, PidWeights, Progress};
// Shared per-image batch seed (`base + index`) — one home in `candle-gen` (sc-9043 / F-059).
use candle_gen::{CandleError, LatentDecoder, Result};
use candle_gen_pid::{PidDecoder, PidEngine};
use candle_transformers::models::flux::sampling::unpack;

/// The PiD backbone (latent-space) tag for Chroma (epic 7840 / sc-7853). Chroma is a FLUX.1-lineage
/// DiT over the FLUX.1 16-ch VAE, so its latent space is `flux` — the same 4× SR student FLUX resolves.
const PID_BACKBONE: &str = "flux";
use candle_transformers::models::t5::T5EncoderModel;
use rand::{rngs::StdRng, SeedableRng};
use tokenizers::Tokenizer;

use crate::config::{ChromaTransformerConfig, ChromaVariant};
use crate::rope;
use crate::text;
use crate::transformer::ChromaTransformer;
use crate::vae::Vae;

/// The VAE latent channel count (the DiT works on the 2×2-packed 64-ch form).
pub(crate) const LATENT_CHANNELS: usize = 16;

/// A light pipeline handle: the snapshot `root`, variant, and compute device. Heavy components load
/// via [`load_components`](Self::load_components) and are owned/cached by the generator.
pub(crate) struct Pipeline {
    variant: ChromaVariant,
    root: PathBuf,
    device: Device,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), built into the cached
    /// [`Components`] so the PiD engine loads once alongside the base model. `None` ⇒ native VAE decode.
    pid_spec: Option<PidWeights>,
    adapters: Vec<AdapterSpec>,
}

/// The loaded Chroma components, `Arc`-shared so the generator can cache them across `generate`
/// calls. The T5 encoder is behind a `Mutex` (its `forward` takes `&mut self` for the
/// relative-position-bias cache) — locked only for the once-per-request text encode.
#[derive(Clone)]
pub(crate) struct Components {
    tokenizer: Arc<Tokenizer>,
    t5: Arc<Mutex<T5EncoderModel>>,
    transformer: Arc<ChromaTransformer>,
    vae: Arc<Vae>,
    cfg: ChromaTransformerConfig,
    /// Optional NVIDIA PiD super-resolving decoder (epic 7840 / sc-7853); None ⇒ native VAE decode.
    pid: Option<Arc<PidEngine>>,
}

struct TextComponents {
    tokenizer: Tokenizer,
    t5: Mutex<T5EncoderModel>,
    cfg: ChromaTransformerConfig,
}

struct DenoiseComponents {
    transformer: ChromaTransformer,
}

enum DecoderComponents {
    Native(Box<Vae>),
    Pid(Box<PidEngine>),
}

trait PhaseSynchronizer {
    fn synchronize(&self) -> Result<()>;
}

impl PhaseSynchronizer for Device {
    fn synchronize(&self) -> Result<()> {
        Ok(Device::synchronize(self)?)
    }
}

/// Own one staged phase's loaded model until queued backend work has synchronized. The guard is
/// intentionally responsible for the component's drop: ordinary returns, `?` errors, cancellation,
/// and panic unwinding all run this ordering. If synchronization itself fails, leaking the component
/// is safer than releasing storage which may still be referenced by queued device work.
struct SynchronizedPhase<T, S: PhaseSynchronizer = Device> {
    component: Option<T>,
    synchronizer: S,
    phase: &'static str,
}

impl<T, S: PhaseSynchronizer> SynchronizedPhase<T, S> {
    fn new(component: T, synchronizer: S, phase: &'static str) -> Self {
        Self {
            component: Some(component),
            synchronizer,
            phase,
        }
    }

    fn synchronize_before_release(&mut self) -> Result<()> {
        let Some(component) = self.component.take() else {
            return Ok(());
        };
        match self.synchronizer.synchronize() {
            Ok(()) => {
                drop(component);
                Ok(())
            }
            Err(error) => {
                std::mem::forget(component);
                Err(CandleError::Msg(format!(
                    "chroma: synchronize before releasing {}: {error}",
                    self.phase
                )))
            }
        }
    }

    fn release(mut self) -> Result<()> {
        self.synchronize_before_release()
    }
}

impl<T, S: PhaseSynchronizer> Deref for SynchronizedPhase<T, S> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.component
            .as_ref()
            .expect("a released staged phase is consumed")
    }
}

impl<T, S: PhaseSynchronizer> Drop for SynchronizedPhase<T, S> {
    fn drop(&mut self) {
        let _ = self.synchronize_before_release();
    }
}

/// Last-resort synchronization for every staged exit, including unwinding. Phase boundaries also
/// synchronize explicitly before their component is dropped; this guard covers early errors and
/// panics between those boundaries.
struct StagedCleanup {
    device: Device,
}

impl Drop for StagedCleanup {
    fn drop(&mut self) {
        let _ = self.device.synchronize();
    }
}

impl Pipeline {
    pub(crate) fn load(
        variant: ChromaVariant,
        root: &Path,
        device: &Device,
        pid_spec: Option<PidWeights>,
        adapters: Vec<AdapterSpec>,
    ) -> Self {
        Self {
            variant,
            root: root.to_path_buf(),
            device: device.clone(),
            pid_spec,
            adapters,
        }
    }

    /// Load the four heavy components from the Chroma diffusers snapshot (`tokenizer/` vendored,
    /// `text_encoder/` T5, `transformer/` DiT, `vae/` AutoencoderKL). The DiT stays F32; the T5 and
    /// VAE stay at their native BF16 checkpoint width.
    ///
    /// The DiT loads through the shared packed-detect seam (sc-9409): a pre-quantized MLX-packed tier
    /// (`SceneWorks/chroma1-*-mlx` q4/q8, whose `transformer/config.json` carries a `quantization`
    /// block) loads straight from the packed parts (no dense bf16 staging); a dense diffusers snapshot
    /// takes the plain path unchanged. The **T5-XXL encoder and VAE ship dense bf16 in every tier** (the
    /// convert job quantizes only the transformer), so their loaders retain those native bytes. The packed
    /// `group_size` is read from the transformer `config.json` (default 64 when absent — never a silent
    /// dense read of the u32 codes).
    pub(crate) fn load_components(&self) -> Result<Components> {
        let text = self.load_text_components()?;
        let denoise = self.load_denoise_components()?;
        let vae = self.load_native_decoder()?;
        let pid = self.load_pid_decoder()?.map(Arc::new);
        Ok(Components {
            tokenizer: Arc::new(text.tokenizer),
            t5: Arc::new(text.t5),
            transformer: Arc::new(denoise.transformer),
            vae: Arc::new(vae),
            cfg: text.cfg,
            pid,
        })
    }

    fn load_text_components(&self) -> Result<TextComponents> {
        Ok(TextComponents {
            tokenizer: text::load_tokenizer()?,
            t5: Mutex::new(text::load_t5(&self.root, &self.device)?),
            cfg: ChromaTransformerConfig::default(),
        })
    }

    fn load_denoise_components(&self) -> Result<DenoiseComponents> {
        let cfg = ChromaTransformerConfig::default();

        let dit_dir = self.root.join("transformer");
        let gs = self.transformer_group_size(&dit_dir);
        let mut transformer = ChromaTransformer::new_gs(cfg, self.vb(&dit_dir, DType::F32)?, gs)?;
        candle_gen::quant::install_dotted_adapters(
            "chroma",
            &self.adapters,
            &self.device,
            |visitor| transformer.visit_adaptable_mut(visitor),
        )?;
        Ok(DenoiseComponents { transformer })
    }

    fn load_native_decoder(&self) -> Result<Vae> {
        let vae_dtype = crate::native_component_dtype(crate::NativeComponent::Vae);
        Ok(Vae::new(self.vb(&self.root.join("vae"), vae_dtype)?)?)
    }

    fn load_pid_decoder(&self) -> Result<Option<PidEngine>> {
        self.pid_spec
            .as_ref()
            .map(|spec| PidEngine::from_spec(spec, PID_BACKBONE, &self.device))
            .transpose()
    }

    /// The MLX packed `group_size` for the DiT, read from `transformer/config.json`'s `quantization`
    /// block (a packed tier). Absent/dense ⇒ the shared default 64 ([`candle_gen::quant::PackedConfig`]
    /// already resolves a missing `group_size` to 64, so a packed tier with only `bits` still loads
    /// packed rather than silently reading its u32 codes dense). The per-tensor `.scales` detect in
    /// [`crate::quant::QLinear::linear_detect_gs`] is what actually routes each projection dense vs
    /// packed — this only supplies the group size the packed branch repacks at.
    fn transformer_group_size(&self, dit_dir: &Path) -> usize {
        std::fs::read_to_string(dit_dir.join("config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| candle_gen::quant::PackedConfig::from_config(&v))
            .map(|pc| {
                candle_gen::gen_core::__model_input_choose!(
                    pc.group_size as usize,
                    {
                        let group_size = pc.group_size as usize;
                        candle_gen::gen_core::model_input_proof::record_reader_paths(
                            "candle-gen-chroma::pipeline::Pipeline::transformer_group_size/configuration",
                            candle_gen::gen_core::model_input_proof::InputRole::Configuration,
                            &[dit_dir.join("config.json")],
                        );
                        group_size
                    }
                )
            })
            .unwrap_or(candle_gen::quant::MLX_GROUP_SIZE)
    }

    /// mmap a [`VarBuilder`] at `dtype` over every `.safetensors` in `dir`.
    fn vb(&self, dir: &Path, dtype: DType) -> Result<VarBuilder<'static>> {
        candle_gen::load_sorted_mmap(dir, dtype, &self.device, "chroma")
    }

    pub(crate) fn render_planned(
        &self,
        req: &GenerationRequest,
        run: &gen_core::RenderRun<'_, '_>,
        components: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let guidance = run.plan().guidance();
        candle_gen::execution::policy("chroma/pipeline/guidance", format!("{guidance:?}").into());
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = run.plan().base_seed();

        let (pos_embeds, neg, pid_decoder) = if req.use_pid {
            run.begin_conditioning().map_err(CandleError::from)?;
            let (pos_embeds, neg) = candle_gen::gen_core::with_prompt_encoding(
                on_progress,
                || -> std::result::Result<_, candle_gen::CandleError> {
                    let pos_embeds = self.encode(components, &req.prompt)?;
                    let neg = if guidance > 1.0 {
                        Some(self.encode(components, negative)?)
                    } else {
                        None
                    };
                    Ok((pos_embeds, neg))
                },
            )?;
            let pid_decoder = candle_gen_pid::resolve_pid_decoder(
                components.pid.as_deref(),
                req,
                base_seed,
                self.variant.id(),
            )?;
            self.device.synchronize()?;
            run.end_conditioning().map_err(CandleError::from)?;
            (pos_embeds, neg, pid_decoder)
        } else {
            let (pos_embeds, neg) = run.encode(|| {
                let encoded = candle_gen::gen_core::with_prompt_encoding(
                    on_progress,
                    || -> std::result::Result<_, candle_gen::CandleError> {
                        let pos_embeds = self.encode(components, &req.prompt)?;
                        let neg = if guidance > 1.0 {
                            Some(self.encode(components, negative)?)
                        } else {
                            None
                        };
                        Ok((pos_embeds, neg))
                    },
                )?;
                self.device.synchronize()?;
                Ok::<_, candle_gen::CandleError>(encoded)
            })?;
            (pos_embeds, neg, None)
        };

        let build_ropes =
            |w: u32, h: u32| -> Result<(rope::RopeTable, Option<rope::RopeTable>)> {
                let h2 = (h as usize).div_ceil(16);
                let w2 = (w as usize).div_ceil(16);
                let rope_pos =
                    rope::build_for(&components.cfg, pos_embeds.dim(1)?, h2, w2, &self.device)?;
                let rope_neg = match &neg {
                    Some(n) => Some(rope::build_for(
                        &components.cfg,
                        n.dim(1)?,
                        h2,
                        w2,
                        &self.device,
                    )?),
                    None => None,
                };
                Ok((rope_pos, rope_neg))
            };
        let (phase0_w, phase0_h) = run.plan().phase_output_size(0);
        let (full_w, full_h) = run.plan().output_size();
        let phase0_ropes = build_ropes(phase0_w, phase0_h)?;
        let phase1_ropes = if (phase0_w, phase0_h) != (full_w, full_h) {
            Some(build_ropes(full_w, full_h)?)
        } else {
            None
        };

        run.images_with_denoise_phases(
            on_progress,
            |seed| {
                let output_index = seed.wrapping_sub(base_seed) as usize;
                candle_gen::execution::draw("image_initial_noise", seed, output_index);
                self.initial_packed_noise(seed, phase0_h, phase0_w)
            },
            |seed, latents, on_progress| {
                let preview = crate::preview::hook(&req.preview, req.width, req.height);
                self.denoise_planned(
                    run,
                    &components.transformer,
                    latents,
                    &pos_embeds,
                    neg.as_ref(),
                    &phase0_ropes,
                    phase1_ropes.as_ref(),
                    seed,
                    &preview,
                    &req.cancel,
                    on_progress,
                )
            },
            |seed, latents, on_progress| {
                let output_index = seed.wrapping_sub(base_seed) as usize;
                candle_gen::execution::output_index(output_index);
                on_progress(Progress::Decoding);
                self.decode(
                    &components.vae,
                    pid_decoder.as_ref(),
                    &latents,
                    req.height,
                    req.width,
                )
            },
        )
    }

    pub(crate) fn render_staged_planned(
        &self,
        req: &GenerationRequest,
        run: &gen_core::RenderRun<'_, '_>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let _cleanup = StagedCleanup {
            device: self.device.clone(),
        };
        candle_gen::check_cancel(&req.cancel)?;
        let guidance = run.plan().guidance();
        candle_gen::execution::policy("chroma/pipeline/guidance", format!("{guidance:?}").into());
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = run.plan().base_seed();

        run.begin_conditioning().map_err(CandleError::from)?;
        check_calibration_fault(req, gen_core::MemoryPhase::Conditioning)?;
        let text = SynchronizedPhase::new(
            self.load_text_components()?,
            self.device.clone(),
            "conditioning components",
        );
        let (pos_embeds, neg) = candle_gen::gen_core::with_prompt_encoding(
            on_progress,
            || -> std::result::Result<_, candle_gen::CandleError> {
                let pos_embeds = self.encode_parts(&text.tokenizer, &text.t5, &req.prompt)?;
                let neg = if guidance > 1.0 {
                    Some(self.encode_parts(&text.tokenizer, &text.t5, negative)?)
                } else {
                    None
                };
                Ok((pos_embeds, neg))
            },
        )?;
        let build_ropes =
            |w: u32, h: u32| -> Result<(rope::RopeTable, Option<rope::RopeTable>)> {
                let h2 = (h as usize).div_ceil(16);
                let w2 = (w as usize).div_ceil(16);
                let rope_pos =
                    rope::build_for(&text.cfg, pos_embeds.dim(1)?, h2, w2, &self.device)?;
                let rope_neg = neg
                    .as_ref()
                    .map(|value| rope::build_for(&text.cfg, value.dim(1)?, h2, w2, &self.device))
                    .transpose()?;
                Ok((rope_pos, rope_neg))
            };
        let (phase0_w, phase0_h) = run.plan().phase_output_size(0);
        let (full_w, full_h) = run.plan().output_size();
        let phase0_ropes = build_ropes(phase0_w, phase0_h)?;
        let phase1_ropes = if (phase0_w, phase0_h) != (full_w, full_h) {
            Some(build_ropes(full_w, full_h)?)
        } else {
            None
        };
        text.release()?;
        run.end_conditioning().map_err(CandleError::from)?;

        candle_gen::check_cancel(&req.cancel)?;
        check_calibration_fault(req, gen_core::MemoryPhase::Denoise)?;
        let denoise = std::cell::RefCell::new(Some(SynchronizedPhase::new(
            self.load_denoise_components()?,
            self.device.clone(),
            "denoise components",
        )));

        let images = run.denoise_then_decode_with_phases(
            on_progress,
            |seed| {
                let output_index = seed.wrapping_sub(base_seed) as usize;
                candle_gen::execution::draw("image_initial_noise", seed, output_index);
                self.initial_packed_noise(seed, phase0_h, phase0_w)
            },
            |seed, noise, on_progress| {
                let denoise_guard = denoise.borrow();
                let denoise_ref = denoise_guard
                    .as_ref()
                    .expect("denoise components remain loaded during denoise");
                let preview = crate::preview::hook(&req.preview, req.width, req.height);
                self.denoise_planned(
                    run,
                    &denoise_ref.transformer,
                    noise,
                    &pos_embeds,
                    neg.as_ref(),
                    &phase0_ropes,
                    phase1_ropes.as_ref(),
                    seed,
                    &preview,
                    &req.cancel,
                    on_progress,
                )
            },
            || {
                if let Some(d) = denoise.borrow_mut().take() {
                    d.release()?;
                }
                candle_gen::check_cancel(&req.cancel)?;
                check_calibration_fault(req, gen_core::MemoryPhase::Decode)?;
                let decoder = SynchronizedPhase::new(
                    if req.use_pid {
                        DecoderComponents::Pid(Box::new(self.load_pid_decoder()?.ok_or_else(
                            || {
                                CandleError::Msg(format!(
                                    "{}: PiD was requested but no exact PiD load receipt is present",
                                    self.variant.id()
                                ))
                            },
                        )?))
                    } else {
                        DecoderComponents::Native(Box::new(self.load_native_decoder()?))
                    },
                    self.device.clone(),
                    "decode components",
                );
                Ok(decoder)
            },
            |decoder, seed, latent, on_progress| {
                let output_index = seed.wrapping_sub(base_seed) as usize;
                candle_gen::execution::output_index(output_index);
                on_progress(Progress::Decoding);
                let pid_decoder = match &**decoder {
                    DecoderComponents::Pid(pid) => Some(
                        candle_gen_pid::resolve_pid_decoder(
                            Some(pid),
                            req,
                            base_seed,
                            self.variant.id(),
                        )?
                        .ok_or_else(|| {
                            CandleError::Msg(format!(
                                "{}: PiD decoder resolution returned none",
                                self.variant.id()
                            ))
                        })?,
                    ),
                    DecoderComponents::Native(_) => None,
                };
                match &**decoder {
                    DecoderComponents::Native(vae) => {
                        self.decode(vae, None, &latent, req.height, req.width)
                    }
                    DecoderComponents::Pid(_) => self.decode_tensor(
                        None,
                        pid_decoder.as_ref(),
                        &latent,
                        req.height,
                        req.width,
                    ),
                }
            },
        )?;
        Ok(images)
    }

    /// Encode a prompt to its T5 sequence embedding `[1, L, 4096]` (natural length).
    fn encode(&self, components: &Components, prompt: &str) -> Result<Tensor> {
        self.encode_parts(&components.tokenizer, &components.t5, prompt)
    }

    fn encode_parts(
        &self,
        tokenizer: &Tokenizer,
        t5: &Mutex<T5EncoderModel>,
        prompt: &str,
    ) -> Result<Tensor> {
        let mut t5 = candle_gen::lock_recover(t5);
        let encoded = text::encode_prompt(tokenizer, &mut t5, prompt, &self.device)?;
        candle_gen::execution::component_runtime("chroma/t5", encoded.device(), encoded.dtype());
        Ok(encoded)
    }

    /// Chroma's flow-match sigma schedule (length `steps + 1`, descending to a trailing `0`). HD/Flash
    /// use the static-shift `linspace(1, 1/N, N)`; Base uses the beta-spaced schedule.
    #[cfg(test)]
    pub(crate) fn sigmas(&self, steps: usize) -> Vec<f32> {
        variant_sigmas(self.variant, steps)
    }

    /// sc-3673 deterministic, launch-portable initial noise in candle's get_noise shape, 2×2-packed to
    /// the DiT's `[1, Si, 64]`. N(0,1) from a fixed-algorithm CPU RNG seeded by `seed`.
    fn initial_packed_noise(&self, seed: u64, height: u32, width: u32) -> Result<Tensor> {
        let lat_h = (height as usize).div_ceil(16) * 2; // = h/8 for a multiple-of-16 request
        let lat_w = (width as usize).div_ceil(16) * 2;
        let n = LATENT_CHANNELS * lat_h * lat_w;
        let mut rng = StdRng::seed_from_u64(candle_gen::execution::seed("chroma/pipeline", seed));
        let noise = candle_gen::seeded_normal_vec(&mut rng, n);
        let noise = Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat_h, lat_w), &Device::Cpu)?
            .to_device(&self.device)?;
        pack(&noise)
    }

    #[allow(clippy::too_many_arguments)]
    fn denoise_planned(
        &self,
        run: &gen_core::RenderRun<'_, '_>,
        transformer: &ChromaTransformer,
        latents: Tensor,
        pos_embeds: &Tensor,
        neg_embeds: Option<&Tensor>,
        phase0_ropes: &(rope::RopeTable, Option<rope::RopeTable>),
        phase1_ropes: Option<&(rope::RopeTable, Option<rope::RopeTable>)>,
        seed: u64,
        preview: &candle_gen::preview::PreviewHook<'_>,
        cancel: &gen_core::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let guidance = run.plan().guidance();
        let phase0_tokens = run.plan().phase_packed_tokens(0);
        if let Some(low_res_report) = run
            .plan()
            .low_res_start_execution_report("bf16", 2, "base_dit")
            .map_err(CandleError::from)?
        {
            candle_gen::execution::policy(
                "chroma.low_res_start",
                serde_json::to_value(&low_res_report).unwrap_or(serde_json::Value::Null),
            );
        }
        candle_gen::run_flow_sampler_spans(
            Some(run.plan().sampler()),
            TimestepConvention::Sigma,
            run.plan().denoise_sigmas(),
            &run.plan().denoise_spans(),
            Some(run),
            false,
            latents,
            seed,
            cancel,
            on_progress,
            Some(preview),
            |latents, sigma| -> Result<Tensor> {
                let (rope_pos, rope_neg) = match phase1_ropes {
                    Some(ropes) if latents.dims().get(1).copied() != Some(phase0_tokens) => {
                        (&ropes.0, ropes.1.as_ref())
                    }
                    _ => (&phase0_ropes.0, phase0_ropes.1.as_ref()),
                };
                let ts = Tensor::from_vec(vec![sigma], 1, &self.device)?;
                let pooled = transformer.pooled_temb(&ts)?;
                let pos = transformer.forward_prepared(latents, pos_embeds, &pooled, rope_pos)?;
                candle_gen::execution::component_runtime(
                    "chroma/transformer",
                    pos.device(),
                    pos.dtype(),
                );
                match (neg_embeds, rope_neg) {
                    (Some(neg), Some(rope_n)) => {
                        let neg = transformer.forward_prepared(latents, neg, &pooled, rope_n)?;
                        Ok((&neg + ((&pos - &neg)? * guidance as f64)?)?)
                    }
                    _ => Ok(pos),
                }
            },
        )
    }

    /// Unpack the denoised packed latent `[1, Si, 64]` → `[1, 16, H/8, W/8]` and decode to an RGB8
    /// [`Image`] (the `[-1, 1]` output mapped to `[0, 255]`). Native path: the FLUX-lineage `Vae` applies
    /// its own scale/shift un-scale internally. When a PiD decoder resolved (epic 7840 / sc-7853), the
    /// super-resolving `flux`-student consumes the SAME unpacked latent the VAE receives (a zero-transform
    /// seam) and emits a larger `[1,3,4H,4W]` tensor; the size is read from the tensor (never `latent*8`).
    fn decode(
        &self,
        vae: &Vae,
        pid: Option<&PidDecoder>,
        latents: &Tensor,
        height: u32,
        width: u32,
    ) -> Result<Image> {
        self.decode_tensor(Some(vae), pid, latents, height, width)
    }

    fn decode_tensor(
        &self,
        vae: Option<&Vae>,
        pid: Option<&PidDecoder>,
        latents: &Tensor,
        height: u32,
        width: u32,
    ) -> Result<Image> {
        let latents = unpack(latents, height as usize, width as usize)?;
        let decoded = match pid {
            Some(pid) => {
                candle_gen::ensure_decoder_compatible(
                    Some(&candle_gen::gen_core::FLUX1_LATENT_SPACE),
                    pid,
                )?;
                let decoded = pid.decode(&latents)?;
                candle_gen::execution::component_runtime(
                    "chroma/pid_decoder",
                    decoded.device(),
                    decoded.dtype(),
                );
                decoded
            }
            None => {
                let decoded = vae
                    .ok_or_else(|| CandleError::Msg("chroma: native decode lacks VAE".into()))?
                    .decode(&latents)?;
                candle_gen::execution::component_runtime(
                    "chroma/vae",
                    decoded.device(),
                    decoded.dtype(),
                );
                decoded.to_dtype(DType::F32)? // [1, 3, H, W] in [-1, 1]
            }
        };
        let scaled = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
        let img = candle_gen::round_rgb8(&scaled)?;
        let img = img.i(0)?.to_device(&Device::Cpu)?; // [3, H, W]
        let (c, h, w) = img.dims3()?;
        if c != 3 {
            return Err(CandleError::Msg(format!("expected 3 channels, got {c}")));
        }
        let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
        Ok(Image {
            width: w as u32,
            height: h as u32,
            pixels,
        })
    }
}

fn check_calibration_fault(req: &GenerationRequest, phase: gen_core::MemoryPhase) -> Result<()> {
    if req.memory.is_some_and(|memory| {
        memory.calibration_fault_harness_authorized && memory.calibration_error_phase == Some(phase)
    }) {
        Err(CandleError::Msg(format!(
            "chroma: authorized calibration fault at {phase:?}"
        )))
    } else {
        Ok(())
    }
}

pub(crate) fn variant_sigmas(variant: ChromaVariant, steps: usize) -> Vec<f32> {
    if variant.use_beta_sigmas() {
        crate::beta::base_sigmas(steps)
    } else {
        let shift = variant.sigma_shift();
        let n = steps.max(1);
        let smax = 1.0f32;
        let smin = 1.0 / n as f32;
        let mut s = Vec::with_capacity(n + 1);
        for i in 0..n {
            let lin = if n == 1 {
                0.0
            } else {
                i as f32 / (n - 1) as f32
            };
            let sigma = smax + (smin - smax) * lin;
            s.push(shift * sigma / (1.0 + (shift - 1.0) * sigma));
        }
        s.push(0.0);
        s
    }
}

pub(crate) fn resolved_sigmas(
    variant: ChromaVariant,
    req: &GenerationRequest,
) -> gen_core::Result<Vec<f32>> {
    let steps = req
        .steps
        .map(|steps| steps as usize)
        .unwrap_or(variant.default_steps() as usize);
    let native = variant_sigmas(variant, steps);
    let mu = variant.sigma_shift().ln();
    candle_gen::resolve_flow_schedule_with_shift(
        req.scheduler.as_deref(),
        req.scheduler_shift,
        mu,
        steps,
        &native,
    )
}

/// 2×2 pack `[1, 16, h, w] → [1, h/2·w/2, 64]` — candle FLUX's `State::new` image packing (so the
/// row-major `img_ids` in [`crate::rope`] line up with the packed token order).
fn pack(x: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    Ok(
        x.reshape((b, c, h / 2, 2, w / 2, 2))? // (b, c, h, ph, w, pw)
            .permute((0, 2, 4, 1, 3, 5))? // (b, h, w, c, ph, pw)
            .reshape((b, h / 2 * w / 2, c * 4))?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct RecordingSynchronizer {
        events: Arc<Mutex<Vec<&'static str>>>,
        fail: bool,
    }

    impl PhaseSynchronizer for RecordingSynchronizer {
        fn synchronize(&self) -> Result<()> {
            candle_gen::lock_recover(&self.events).push("synchronize");
            if self.fail {
                Err(CandleError::Msg("expected synchronize failure".into()))
            } else {
                Ok(())
            }
        }
    }

    struct RecordingComponent(Arc<Mutex<Vec<&'static str>>>);

    impl Drop for RecordingComponent {
        fn drop(&mut self) {
            candle_gen::lock_recover(&self.0).push("release");
        }
    }

    fn early_staged_exit(
        events: Arc<Mutex<Vec<&'static str>>>,
        message: &'static str,
    ) -> Result<()> {
        let _phase = SynchronizedPhase::new(
            RecordingComponent(events.clone()),
            RecordingSynchronizer {
                events,
                fail: false,
            },
            "fixture",
        );
        Err(CandleError::Msg(message.into()))
    }

    #[test]
    fn staged_component_release_synchronizes_on_normal_error_cancel_and_panic() {
        let events = Arc::new(Mutex::new(Vec::new()));
        SynchronizedPhase::new(
            RecordingComponent(events.clone()),
            RecordingSynchronizer {
                events: events.clone(),
                fail: false,
            },
            "normal",
        )
        .release()
        .unwrap();
        assert_eq!(
            *candle_gen::lock_recover(&events),
            ["synchronize", "release"]
        );

        for message in ["expected error", "expected cancellation"] {
            candle_gen::lock_recover(&events).clear();
            assert!(early_staged_exit(events.clone(), message).is_err());
            assert_eq!(
                *candle_gen::lock_recover(&events),
                ["synchronize", "release"],
                "{message}"
            );
        }

        candle_gen::lock_recover(&events).clear();
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let events = events.clone();
            move || {
                let _phase = SynchronizedPhase::new(
                    RecordingComponent(events.clone()),
                    RecordingSynchronizer {
                        events,
                        fail: false,
                    },
                    "panic",
                );
                panic!("expected panic");
            }
        }));
        assert!(unwind.is_err());
        assert_eq!(
            *candle_gen::lock_recover(&events),
            ["synchronize", "release"]
        );

        candle_gen::lock_recover(&events).clear();
        let result = SynchronizedPhase::new(
            RecordingComponent(events.clone()),
            RecordingSynchronizer {
                events: events.clone(),
                fail: true,
            },
            "failed synchronization",
        )
        .release();
        assert!(result.is_err());
        assert_eq!(
            *candle_gen::lock_recover(&events),
            ["synchronize"],
            "a failed synchronization must leak rather than release active storage"
        );
    }

    #[test]
    fn calibration_faults_are_authorized_and_phase_exact() {
        let plain = GenerationRequest::default();
        for phase in [
            gen_core::MemoryPhase::Conditioning,
            gen_core::MemoryPhase::Denoise,
            gen_core::MemoryPhase::Decode,
        ] {
            assert!(check_calibration_fault(&plain, phase).is_ok());
            let mut memory = gen_core::GenerationMemory {
                calibration_error_phase: Some(phase),
                ..Default::default()
            };
            let unauthorized = GenerationRequest {
                memory: Some(memory),
                ..Default::default()
            };
            assert!(check_calibration_fault(&unauthorized, phase).is_ok());

            memory.authorize_calibration_fault(phase);
            let authorized = GenerationRequest {
                memory: Some(memory),
                ..Default::default()
            };
            assert!(check_calibration_fault(&authorized, phase).is_err());
        }
    }

    /// `transformer_group_size` reads the packed `transformer/config.json`'s `quantization.group_size`
    /// (the sc-9409 packed tier), defaults to the shared 64 for a `bits`-only or a dense config, and
    /// never returns a *non*-value that would silently read u32 codes dense. Writes small JSONs — no
    /// weights, GPU-free.
    #[test]
    fn transformer_group_size_reads_quantization_block() {
        let tmp_guard = tempfile::tempdir().unwrap();
        let tmp = tmp_guard.path().to_path_buf();
        let dir = tmp.join("transformer");
        std::fs::create_dir_all(&dir).ok();
        let pipe = Pipeline::load(ChromaVariant::Base, &tmp, &Device::Cpu, None, Vec::new());

        // A real Chroma packed tier: bits 4, group 64.
        std::fs::write(
            dir.join("config.json"),
            r#"{ "num_layers": 19, "quantization": { "bits": 4, "group_size": 64 } }"#,
        )
        .unwrap();
        assert_eq!(pipe.transformer_group_size(&dir), 64);

        // A non-64 packed tier (the sc-9410 group-32 shape) is threaded verbatim.
        std::fs::write(
            dir.join("config.json"),
            r#"{ "quantization": { "bits": 4, "group_size": 32 } }"#,
        )
        .unwrap();
        assert_eq!(pipe.transformer_group_size(&dir), 32);

        // `bits`-only (no group_size) ⇒ the shared default 64 (PackedConfig resolves it), NOT a dense
        // read of the packed codes.
        std::fs::write(
            dir.join("config.json"),
            r#"{ "quantization": { "bits": 8 } }"#,
        )
        .unwrap();
        assert_eq!(pipe.transformer_group_size(&dir), 64);

        // A dense snapshot (no `quantization`) ⇒ default 64 (inert — the per-tensor detect takes the
        // dense path regardless).
        std::fs::write(dir.join("config.json"), r#"{ "num_layers": 19 }"#).unwrap();
        assert_eq!(pipe.transformer_group_size(&dir), 64);

        // Absent config ⇒ default 64 (no panic).
        assert_eq!(
            pipe.transformer_group_size(&tmp.join("missing")),
            candle_gen::quant::MLX_GROUP_SIZE
        );
    }

    /// HD's static shift moves the interior sigmas but keeps a descending 1→0 schedule of length N+1.
    #[test]
    fn hd_sigmas_descend_with_shift() {
        let pipe = Pipeline::load(
            ChromaVariant::Hd,
            Path::new("/x"),
            &Device::Cpu,
            None,
            Vec::new(),
        );
        let s = pipe.sigmas(8);
        assert_eq!(s.len(), 9);
        assert!(
            (s[0] - 1.0).abs() < 1e-6,
            "starts at shift·1/(1+ (shift-1)) = 1: {s:?}"
        );
        assert!(s[8].abs() < 1e-9, "ends at 0: {s:?}");
        for w in s.windows(2) {
            assert!(w[0] > w[1], "must descend: {s:?}");
        }
    }

    /// Flash uses shift 1.0 → the schedule is the raw `linspace(1, 1/N, N)` + trailing 0.
    #[test]
    fn flash_sigmas_are_unshifted_linspace() {
        let pipe = Pipeline::load(
            ChromaVariant::Flash,
            Path::new("/x"),
            &Device::Cpu,
            None,
            Vec::new(),
        );
        let s = pipe.sigmas(4);
        // linspace(1, 1/4, 4) = [1, 0.75, 0.5, 0.25], then 0.
        let want = [1.0, 0.75, 0.5, 0.25, 0.0];
        for (g, w) in s.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w} in {s:?}");
        }
    }

    /// Base routes through the beta-spaced schedule (distinct from the linspace).
    #[test]
    fn base_sigmas_use_beta_schedule() {
        let pipe = Pipeline::load(
            ChromaVariant::Base,
            Path::new("/x"),
            &Device::Cpu,
            None,
            Vec::new(),
        );
        let s = pipe.sigmas(4);
        assert_eq!(s, crate::beta::base_sigmas(4));
        // 0.79344 (beta) ≠ 0.75 (linspace) at index 1.
        assert!((s[1] - 0.75).abs() > 1e-3);
    }

    /// 2×2 pack folds `[1,16,4,4] → [1,4,64]` (Si = (4/2)·(4/2) = 4, 16·4 = 64).
    #[test]
    fn pack_shapes() {
        let x = Tensor::zeros((1, 16, 4, 4), DType::F32, &Device::Cpu).unwrap();
        let p = pack(&x).unwrap();
        assert_eq!(p.dims(), &[1, 4, 64]);
    }
}
